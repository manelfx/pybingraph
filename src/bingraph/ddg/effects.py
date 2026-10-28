"""Read instruction effects from VEX with bounded local value evaluation.

Only straight-line local address values are evaluated, through angr's existing
Claripy IROp implementations and register storage. Memory loads yield fresh
unknown values: disambiguation must not depend on guessed concrete addresses.
No CFG, path exploration or callee execution is involved.
Read-only repeated x86 string comparisons use maximum-footprint summaries,
not guessed iteration counts or final register values.
"""

from dataclasses import replace
from itertools import takewhile
from typing import Any

import angr
from angr import claripy
from archinfo import Arch
from capstone import x86_const as x86
import pyvex
from angr.engines.vex.claripy.irop import vexop_to_simop
from angr.engines.vex.claripy.datalayer import value as vex_value
from pyvex.vex_ffi import guest_offsets

from bingraph.helpers.capstone import InsnSemantics
from bingraph.helpers.vex import _vex_get_key
from .models import Effects, Resource
from .memory import address_distance, address_label, coalesce_memory, offset_label

MAX_VALUE_NODES = 64
MAX_INSTRUCTION_ITERATIONS = 256
VEX_REGISTER_NAMES = {
    (arch, offset): name for (arch, name), offset in guest_offsets.items()
}
REPEATED_READ_IDS = {
    x86.X86_INS_CMPSB,
    x86.X86_INS_CMPSW,
    x86.X86_INS_CMPSD,
    x86.X86_INS_CMPSQ,
    x86.X86_INS_SCASB,
    x86.X86_INS_SCASW,
    x86.X86_INS_SCASD,
    x86.X86_INS_SCASQ,
}


def _register_label(arch: Arch, offset: int, width: int) -> str:
    """Name missing register views without changing their byte-range identity.

    Keep Archinfo's established aliases. PyVEX can name otherwise undeclared
    guest fields; declared parent registers identify interior slices, with bit
    numbers determined by register endianness. Gaps remain explicit guest slots,
    not guessed architectural names or offsets mistaken for register names.
    """
    name = arch.translate_register_name(offset, width)
    if not name.isdecimal():
        return name
    vex_arch = (arch.vex_arch or "").removeprefix("VexArch").lower()
    if name := VEX_REGISTER_NAMES.get((vex_arch, offset)):
        return name
    for reg in arch.register_list:
        if (
            reg.vex_offset is not None
            and reg.vex_offset <= offset
            and offset + width <= reg.vex_offset + reg.size
        ):
            byte = offset - reg.vex_offset
            if arch.register_endness == "Iend_BE":
                byte = reg.size - byte - width
            low = byte * arch.byte_width
            return f"{reg.name}[{low + width * arch.byte_width - 1}:{low}]"
    return f"guest[{offset:#x}:{width}]"


def instruction_effects(
    project: angr.Project, instructions: tuple[Any, ...], size: int
) -> list[Effects]:
    """Lift an exact block and retain reads/writes grouped by instruction.

    Lift without IR optimization: otherwise lazy flag computations can move
    across instruction markers and acquire a false producer. Pure temporary
    reads are committed only when consumed, so cancelled operands do not add
    dependencies. Value evaluation is bounded independently of resource reads.
    Capstone/archinfo alias ranges narrow full-width VEX GETs
    followed by truncations; implicit VEX resources absent from Capstone remain.
    Concrete instruction-internal backedges replay this same evaluator within
    a fixed budget. Known-count CMPS/SCAS instead summarize maximum reads and
    forget final registers. Other symbolic or excessive repetition fails rather
    than reporting one iteration as the complete instruction's effects.
    """

    start = instructions[0].address
    effects = {i.address: Effects(i) for i in instructions}
    # Identify real REP string reads, not SSE comparisons or other ISAs whose
    # instruction IDs happen to match. VEX still supplies all register effects.
    repeated_reads = {
        i.address
        for i in instructions
        if project.arch.name in {"X86", "AMD64"}
        and i.id in REPEATED_READ_IDS
        and i.prefix[0] in {0xF2, 0xF3}
    }
    blocks = []
    cursor = start
    # VEX caps individual lifts (normally at 99 instructions), independently
    # of the requested bytes. Continue only at a linear instruction boundary,
    # including summarized repeated comparisons. Register state spans chunks,
    # but each IRSB has its own temporaries.
    while cursor < start + size:
        vex = project.factory.block(
            cursor,
            size=start + size - cursor,
            opt_level=0,
            cross_insn_opt=False,
            strict_block_end=True,
        ).vex
        end = cursor + vex.size
        if (
            not vex.size
            or end > start + size
            or vex.jumpkind in {"Ijk_NoDecode", "Ijk_EmFail"}
        ):
            raise ValueError("DDG region must be one fully liftable basic block")
        blocks.append(vex)
        if end < start + size:
            markers = [s for s in vex.statements if isinstance(s, pyvex.stmt.IMark)]
            last = (
                effects.get(markers[-1].addr + markers[-1].delta) if markers else None
            )
            if (
                end not in effects
                or vex.jumpkind != "Ijk_Boring"
                or (
                    vex.exit_statements
                    and last is not None
                    and last.insn.address not in repeated_reads
                )
                or not isinstance(vex.next, pyvex.expr.Const)
                or vex.next.con.value != end
                or last is None
                or InsnSemantics(last.insn).is_control_transfer()
            ):
                raise ValueError("DDG region must be one fully liftable basic block")
        cursor = end
    # Some delay-slot lifters use one IMark for branch + slot. Split effects,
    # not code: lift the branch against a synthetic NOP slot, then analyze the
    # real slot independently using the same register-value state. Branch
    # target reads precede slot writes; link-register writes remain visible.
    fused = [
        s
        for vex in blocks
        for s in vex.statements
        if isinstance(s, pyvex.stmt.IMark)
        and any(i.address == s.addr + s.delta and i.size < s.len for i in instructions)
    ]
    if fused:
        if (
            not project.arch.branch_delay_slot
            or len(fused) != 1
            or len(instructions) < 2
        ):
            raise ValueError("Unsupported fused VEX instruction markers")
        branch, slot = instructions[-2:]
        # The validated VEX bundle is authoritative: Capstone omits the
        # generic call/jump groups for some transfers (notably MIPS BAL).
        if (
            fused[0].addr != branch.address
            or fused[0].len != branch.size + slot.size
            or InsnSemantics(slot).is_control_transfer()
        ):
            raise ValueError(
                "Unsupported instruction arrangement in a delay-slot bundle"
            )
        nop = project.arch.nop_instruction
        if len(nop) != slot.size:
            raise ValueError("Unsupported delay-slot instruction size")
        tail_start = blocks[-1].addr
        tail_size = start + size - tail_start
        data = (
            b"".join(
                bytes(i.bytes) for i in instructions[:-1] if i.address >= tail_start
            )
            + nop
        )
        blocks[-1:] = [
            project.factory.block(
                tail_start,
                byte_string=data,
                size=tail_size,
                opt_level=0,
                cross_insn_opt=False,
            ).vex,
            project.factory.block(
                slot.address, size=slot.size, opt_level=0, cross_insn_opt=False
            ).vex,
        ]
    state = project.factory.blank_state(
        addr=start, add_options={angr.options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS}
    )
    # Stable versioned roots make address descriptors deterministic across
    # requests and avoid assigning concrete values to unknown entry registers.
    for reg in project.arch.register_list:
        if reg.vex_offset is not None:
            state.registers.store(
                reg.vex_offset,
                claripy.BVS(f"entry_{reg.name}", reg.size * 8, explicit_name=True),
                size=reg.size,
                endness=project.arch.register_endness,
            )
    state.regs.ip = start
    ignored_offsets = {project.arch.ip_offset}
    # VEX's restart counter (including its accumulator subfield) is internal,
    # not an architectural live-in. Completed instructions leave it zero.
    counter = guest_offsets.get((project.arch.name.lower(), "counter"))
    if counter is not None:
        state.registers.store(
            counter, 0, size=project.arch.bytes, endness=project.arch.register_endness
        )
        ignored_offsets.update(range(counter, counter + project.arch.bytes))
    temporaries: dict[int, tuple[Any, int, tuple[Resource, ...]]] = {}
    temporary_labels: dict[int, str] = {}
    address_registers: dict[str, Any] = {}
    current = effects[start]
    serial = 0
    tmp_base = 0
    repeated = set()
    itstate = project.arch.registers.get("itstate")
    itstate_resets_only = itstate is not None

    def unknown(bits: int):
        nonlocal serial
        serial += 1
        return claripy.BVS(f"ddg_unknown_{serial}", bits, explicit_name=True)

    def small_values(*values) -> bool:
        """Bound expanded expression size, not just depth or unique DAG nodes.

        Associative arithmetic can flatten repeated values into exponentially
        wide, shallow expressions. Losing an optional value is conservative;
        resource reads have already been collected independently.
        """
        pending = list(values)
        budget = MAX_VALUE_NODES
        while pending:
            value = pending.pop()
            budget -= 1
            if budget < 0 or len(value.args) > budget:
                return False
            pending.extend(
                arg for arg in value.args if isinstance(arg, claripy.ast.Base)
            )
        return True

    def register(offset: int, width: int) -> Resource:
        return Resource(
            "register",
            offset,
            width,
            _register_label(project.arch, offset, width),
        )

    def memory(address: Any, width: int) -> Resource:
        if address is None:
            return Resource("memory", 0, width, "mem?", None)
        text = address_label(address)
        display = None
        if any(name.startswith("ddg_unknown_") for name in address.variables):
            display = "[?]"
            # Name loaded/opaque pointers using registers actually read by
            # this instruction, not the temporary's arbitrary symbolic ID.
            # Keep the original expression for alias decisions and tooltips.
            for name, value in address_registers.items():
                delta = address_distance(value, address)
                if delta is not None:
                    display = "[" + offset_label(name, delta, len(address)) + "]"
                    break
        return Resource("memory", 0, width, f"mem[{text}]:{width}", address, display)

    def summarize_repeated_reads(count: int) -> None:
        """Widen one VEX iteration to a possible footprint, not executed bytes.

        With unknown DF, cover both legal directions. Address-size overrides
        can wrap inside the wider pointer space: keep those addresses opaque.
        These instructions have no memory writes, so maximum extents never
        become producer-killing stores. All final register values stay unknown.
        """
        direction = state.regs.d
        reads = []
        for resource in current.reads:
            if resource.kind == "memory":
                span = (count - 1) * resource.size
                address = resource.address
                width = span + resource.size
                if not direction.concrete:
                    address = address - span if address is not None else None
                    width += span
                elif direction.concrete_value != 1:
                    address = address - span if address is not None else None
                if current.insn.addr_size != project.arch.bytes and count > 1:
                    address = None
                resource = memory(address, min(width, 1 << project.arch.bits))
                resource = replace(resource, maximum_footprint=True)
            reads.append(resource)
        current.reads = reads
        for resource in current.writes:
            if resource.kind == "register":
                state.registers.store(
                    resource.offset,
                    unknown(resource.size * 8),
                    size=resource.size,
                    endness=project.arch.register_endness,
                )

    def expression(expr: Any):
        """Evaluate supported scalar expressions using angr, retaining accesses.

        Unsupported arithmetic becomes an unconstrained value, but all operand
        accesses are still recorded. Unsupported stateful statements are errors.
        """
        bits = expr.result_size(vex.tyenv)
        if isinstance(expr, pyvex.expr.Const):
            return vex_value(expr.con.type, expr.con.value)
        if isinstance(expr, pyvex.expr.RdTmp):
            value, owner, reads = temporaries[expr.tmp]
            if owner != current.insn.address:
                effects[owner].reads.extend(reads)
                current.reads.append(
                    Resource(
                        "temporary",
                        tmp_base + expr.tmp,
                        1,
                        temporary_labels.get(expr.tmp, "lifted value"),
                    )
                )
            else:
                current.reads.extend(reads)
            return value
        if isinstance(expr, pyvex.expr.Get):
            value = state.registers.load(
                expr.offset, size=bits // 8, endness=project.arch.register_endness
            )
            if expr.offset not in ignored_offsets:
                resource = register(expr.offset, bits // 8)
                # VEX may read guest bytes it already wrote in this same
                # instruction (including its own zeroing normalization).
                # Only the still-unwritten bytes are external inputs. Reads
                # captured earlier in temporaries retain their original inputs.
                ranges = [(resource.offset, resource.offset + resource.size)]
                for written in current.writes:
                    if written.kind != "register":
                        continue
                    ranges = [
                        part
                        for lo, hi in ranges
                        for part in (
                            (lo, min(hi, written.offset)),
                            (max(lo, written.offset + written.size), hi),
                        )
                        if part[0] < part[1]
                    ]
                current.reads.extend(register(lo, hi - lo) for lo, hi in ranges)
                if bits == project.arch.bits:
                    # Preserve the value at its first read even if a later
                    # writeback updates the register within this instruction.
                    address_registers.setdefault(resource.label, value)
            return value
        if isinstance(expr, pyvex.expr.Load):
            address = expression(expr.addr)
            current.reads.append(memory(address, bits // 8))
            return unknown(bits)
        if isinstance(expr, pyvex.expr.ITE):
            guard = expression(expr.cond)
            if guard.concrete:
                return expression(expr.iftrue if guard.concrete_value else expr.iffalse)
            yes, no = expression(expr.iftrue), expression(expr.iffalse)
            if not small_values(guard, yes, no):
                return unknown(bits)
            return claripy.If(guard != 0, yes, no)
        if isinstance(expr, pyvex.expr.Binop):
            # Ask angr whether equal *register views* cancel, using a fresh
            # input rather than current values. Equal constants in different
            # registers must not erase a real dependency (e.g. xor eax,ebx).
            left = _vex_get_key(expr.args[0], definitions, vex)
            if (
                left is not None
                and left == _vex_get_key(expr.args[1], definitions, vex)
                and not any(
                    r.kind == "register"
                    and r.offset < left[0] + (left[1] + 7) // 8
                    and left[0] < r.offset + r.size
                    for r in current.writes
                )
            ):
                try:
                    probe = claripy.BVS(
                        "ddg_operand", expr.args[0].result_size(vex.tyenv)
                    )
                    folded = vexop_to_simop(expr.op).calculate(probe, probe)
                    if folded.concrete:
                        return folded
                except (
                    angr.errors.SimError,
                    claripy.ClaripyError,
                    TypeError,
                    ValueError,
                    KeyError,
                ):
                    pass
        args = [expression(arg) for arg in getattr(expr, "args", ())]
        if isinstance(
            expr, (pyvex.expr.Unop, pyvex.expr.Binop, pyvex.expr.Triop, pyvex.expr.Qop)
        ):
            # Preserve all reads above even when optional symbolic values grow
            # too large. Opaque values weaken alias proofs, never dependencies.
            if not small_values(*args):
                return unknown(bits)
            try:
                return vexop_to_simop(expr.op).calculate(*args)
            except (
                angr.errors.SimError,
                claripy.ClaripyError,
                TypeError,
                ValueError,
                KeyError,
            ):
                return unknown(bits)
        if isinstance(expr, pyvex.expr.CCall):
            return unknown(bits)
        raise ValueError(f"Unsupported DDG VEX expression: {type(expr).__name__}")

    for vex in blocks:
        definitions = {}
        temporaries.clear()
        temporary_labels.clear()
        index = loop_start = iterations = 0
        repeat_count = None
        while index < len(vex.statements):
            statement = vex.statements[index]
            index += 1
            if isinstance(statement, pyvex.stmt.IMark):
                address = statement.addr + statement.delta
                if address not in effects:
                    raise ValueError("VEX/assembly instruction boundaries disagree")
                if address != current.insn.address:
                    iterations = 0
                loop_start = index - 1
                current = effects[address]
                address_registers.clear()
                # A register GET from an older instruction is a snapshot, not
                # the same input as another GET after intervening writes.
                definitions.clear()
                if address in repeated_reads:
                    if address + current.insn.size != vex.addr + vex.size:
                        raise ValueError(
                            "Unsupported DDG repeated-instruction boundary"
                        )
                    name = {2: "cx", 4: "ecx", 8: "rcx"}[current.insn.addr_size]
                    offset, width = project.arch.registers[name]
                    count = state.registers.load(
                        offset, size=width, endness=project.arch.register_endness
                    )
                    if not count.concrete:
                        raise ValueError(
                            f"Unsupported DDG symbolic repeat count at {address:#x}"
                        )
                    repeat_count = count.concrete_value
                    if not repeat_count:
                        # Zero repetitions do not touch pointers, memory or flags.
                        current.reads.append(register(offset, width))
                        state.regs.ip = address + current.insn.size
                        index = len(vex.statements)
            elif isinstance(statement, pyvex.stmt.WrTmp):
                first = len(current.reads)
                value = expression(statement.data)
                reads = tuple(dict.fromkeys(current.reads[first:]))
                if isinstance(statement.data, pyvex.expr.Load):
                    # Loads execute even when their result is unused. Pure
                    # register reads, however, belong only to live consumers.
                    reads = ()
                else:
                    del current.reads[first:]
                temporaries[statement.tmp] = (
                    value,
                    current.insn.address,
                    reads,
                )
                definitions[statement.tmp] = statement.data
                current.writes.append(
                    Resource("temporary", tmp_base + statement.tmp, 1, "lifted value")
                )
            elif isinstance(statement, pyvex.stmt.Put):
                value = expression(statement.data)
                width = statement.data.result_size(vex.tyenv) // 8
                if (
                    itstate is not None
                    and statement.offset < itstate[0] + itstate[1]
                    and itstate[0] < statement.offset + width
                ):
                    itstate_resets_only &= (
                        (statement.offset, width) == itstate
                        and value.concrete
                        and value.concrete_value == 0
                    )
                state.registers.store(
                    statement.offset,
                    value,
                    size=width,
                    endness=project.arch.register_endness,
                )
                if statement.offset not in ignored_offsets:
                    resource = register(statement.offset, width)
                    current.writes.append(resource)
                    if isinstance(statement.data, pyvex.expr.RdTmp):
                        temporary_labels[statement.data.tmp] = resource.label
            elif isinstance(statement, pyvex.stmt.Store):
                address = expression(statement.addr)
                expression(statement.data)
                current.writes.append(
                    memory(address, statement.data.result_size(vex.tyenv) // 8)
                )
            elif isinstance(statement, pyvex.stmt.LoadG):
                guard = expression(statement.guard)
                if not guard.concrete:
                    raise ValueError("Unsupported DDG conditional LoadG")
                if guard.concrete_value:
                    read_type, _ = statement.cvt_types
                    value = expression(
                        pyvex.expr.Load(statement.end, read_type, statement.addr)
                    )
                    if not statement.cvt.startswith("ILGop_Ident"):
                        value = vexop_to_simop(
                            statement.cvt.replace("ILGop_", "Iop_")
                        ).calculate(value)
                else:
                    value = expression(statement.alt)
                temporaries[statement.dst] = (value, current.insn.address, ())
                current.writes.append(
                    Resource("temporary", tmp_base + statement.dst, 1, "lifted value")
                )
            elif isinstance(statement, pyvex.stmt.Exit):
                guard = expression(statement.guard)
                if current.insn.address in repeated_reads:
                    continue  # One iteration supplies effects; the summary supplies extent.
                if (
                    statement.jumpkind == "Ijk_Boring"
                    and statement.dst.value == current.insn.address
                    and not InsnSemantics(current.insn).is_control_transfer()
                ):
                    if current.insn.address not in repeated:
                        # An additional ordinary exit could skip iterations or
                        # stores. Do not turn such conditional writes into a
                        # complete range that kills earlier producers.
                        body = takewhile(
                            lambda s: not isinstance(s, pyvex.stmt.IMark),
                            vex.statements[loop_start + 1 :],
                        )
                        if any(
                            isinstance(s, pyvex.stmt.Exit)
                            and s is not statement
                            and s.jumpkind == "Ijk_Boring"
                            for s in body
                        ):
                            raise ValueError(
                                "Unsupported DDG multi-exit instruction-internal loop"
                            )
                    if not guard.concrete:
                        raise ValueError(
                            "Unsupported DDG symbolic instruction-internal loop"
                        )
                    repeated.add(current.insn.address)
                    if guard.concrete_value:
                        iterations += 1
                        if iterations >= MAX_INSTRUCTION_ITERATIONS:
                            raise ValueError(
                                "DDG instruction-internal loop exceeded iteration budget"
                            )
                        # Same instruction and effects owner; no CFG traversal,
                        # path splitting, or loss of the original memory inputs.
                        index = loop_start
            elif isinstance(statement, pyvex.stmt.MBE):
                current.reads.append(memory(None, 1))
                current.writes.append(memory(None, 1))
            elif isinstance(statement, (pyvex.stmt.NoOp, pyvex.stmt.AbiHint)):
                continue
            else:
                raise ValueError(
                    f"Unsupported DDG VEX statement: {type(statement).__name__}"
                )
        expression(vex.next)
        if repeat_count:
            summarize_repeated_reads(repeat_count)
        if (
            vex.jumpkind == "Ijk_Boring"
            and isinstance(vex.next, pyvex.expr.Const)
            and vex.next.con.value == current.insn.address
            and not InsnSemantics(current.insn).is_control_transfer()
        ):
            raise ValueError("Unsupported DDG tail instruction-internal loop")
        tmp_base += len(vex.tyenv.types)

    for effect in effects.values():
        if effect.insn.address in repeated:
            effect.reads = coalesce_memory(effect.reads)
            effect.writes = coalesce_memory(effect.writes)
        # Capstone describes the actual architectural read width (al/ah/eax,
        # SIMD aliases, etc.). Only narrow matching ranges; never discard an
        # implicit register or VEX condition-state read absent from Capstone.
        try:
            reads, _ = effect.insn.regs_access()
            ranges = [
                project.arch.registers[name]
                for reg in reads
                if (name := effect.insn.reg_name(reg)) in project.arch.registers
            ]
        except Exception:
            ranges = []
        narrowed = []
        for resource in effect.reads:
            matches = [
                (
                    max(resource.offset, offset),
                    min(resource.offset + resource.size, offset + width),
                )
                for offset, width in ranges
                if resource.kind == "register"
                and resource.offset < offset + width
                and offset < resource.offset + resource.size
            ]
            if matches:
                narrowed.extend(register(lo, hi - lo) for lo, hi in matches)
            else:
                narrowed.append(resource)
        effect.reads = list(dict.fromkeys(narrowed))
    # VEX resets inactive Thumb IT state even on unrelated instructions. These
    # resets are not output dependencies. Keep all effects if any consumed read
    # or nonzero/unknown/partial write could represent real predication; never
    # assume that a block starts outside an IT region. Check after deferred
    # temporary reads have been attributed to their actual consumers.
    if itstate_resets_only and itstate is not None:
        if not any(
            r.kind == "register"
            and r.offset < itstate[0] + itstate[1]
            and itstate[0] < r.offset + r.size
            for effect in effects.values()
            for r in effect.reads
        ):
            for effect in effects.values():
                effect.writes = [
                    r
                    for r in effect.writes
                    if not (r.kind == "register" and (r.offset, r.size) == itstate)
                ]
    return list(effects.values())
