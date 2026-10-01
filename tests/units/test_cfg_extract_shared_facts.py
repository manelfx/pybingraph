"""Must-fact safety at joins, loops, aliases, calls, and VEX read positions."""

from dataclasses import dataclass
from types import SimpleNamespace

import archinfo
import networkx as nx
import pyvex
import pytest

from bingraph.cfg.models import FunctionBounds
from bingraph.cfg_extract.shared_facts import PredecessorFacts


@dataclass(eq=False)
class Node:
    addr: int
    block: object
    size: int = 16


ARCH = archinfo.ArchAMD64()
RBX = ARCH.registers["rbx"][0]
RAX = ARCH.registers["rax"][0]


def const(value, bits=64):
    return pyvex.expr.Const(getattr(pyvex.const, f"U{bits}")(value))


def get(offset, bits=64):
    return pyvex.expr.Get(offset, f"Ity_I{bits}")


def node(addr, *statements, jumpkind="Ijk_Boring", types=()):
    tyenv = pyvex.IRTypeEnv(ARCH, list(types))
    return Node(
        addr,
        SimpleNamespace(
            vex=SimpleNamespace(
                statements=list(statements),
                tyenv=tyenv,
                jumpkind=jumpkind,
                next=get(ARCH.ip_offset),
            )
        ),
    )


def setup(nodes, edges, *, max_steps=20000, writable=False, seeds=None, arch=ARCH):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for source, target in edges:
        graph.add_edge(source, target, jumpkind="Ijk_Boring")
    project = SimpleNamespace(
        arch=arch,
        loader=SimpleNamespace(
            main_object=SimpleNamespace(os="UNIX - System V"),
            find_section_containing=lambda addr: SimpleNamespace(
                is_writable=writable, max_addr=0x500010
            ),
            memory=SimpleNamespace(
                load=lambda addr, size: (37).to_bytes(size, "little")
            ),
        ),
    )
    bounds = FunctionBounds(
        nodes[0].addr, 0x401000, 0x1000, SimpleNamespace(name="test")
    )
    return PredecessorFacts(project, graph, bounds, seeds or {}, max_steps=max_steps)


@pytest.mark.parametrize("other,expected", [(7, 7), (8, None), (None, None)])
def test_every_path_must_define_the_same_constant(other, expected):
    entry = node(0x400000)
    left = node(0x400010, pyvex.stmt.Put(const(7), RBX))
    right = node(
        0x400020, *([] if other is None else [pyvex.stmt.Put(const(other), RBX)])
    )
    use = node(0x400030)
    facts = setup(
        [entry, left, right, use],
        [(entry, left), (entry, right), (left, use), (right, use)],
    )
    assert facts.value(use, get(RBX)) == expected


@pytest.mark.parametrize("increment,expected", [(False, 7), (True, None)])
def test_loop_without_writes_preserves_but_loop_carried_computation_does_not(
    increment, expected
):
    entry = node(0x400000, pyvex.stmt.Put(const(7), RBX))
    body = node(
        0x400010,
        *(
            [pyvex.stmt.Put(pyvex.expr.Binop("Iop_Add64", [get(RBX), const(1)]), RBX)]
            if increment
            else []
        ),
    )
    use = node(0x400020)
    facts = setup([entry, body, use], [(entry, body), (body, body), (body, use)])
    assert facts.value(use, get(RBX)) == expected


def test_partial_alias_write_invalidates_full_register():
    entry = node(0x400000, pyvex.stmt.Put(const(0x12345678), RAX))
    partial = node(0x400010, pyvex.stmt.Put(const(9, 8), RAX + 1))
    use = node(0x400020)
    facts = setup([entry, partial, use], [(entry, partial), (partial, use)])
    assert facts.value(use, get(RAX)) is None


def test_temporary_read_keeps_value_before_later_register_write():
    entry = node(0x400000, pyvex.stmt.Put(const(7), RBX))
    use = node(
        0x400010,
        pyvex.stmt.WrTmp(0, get(RBX)),
        pyvex.stmt.Put(const(99), RBX),
        pyvex.stmt.Put(pyvex.expr.RdTmp(0), RAX),
        types=("Ity_I64",),
    )
    facts = setup([entry, use], [(entry, use)])
    assert facts.value(use, get(RAX)) == 7
    assert facts.value(use, get(RBX)) == 99


@pytest.mark.parametrize("offset,expected", [(RBX, 7), (RAX, None)])
def test_calls_apply_abi_preservation(offset, expected):
    entry = node(0x400000, pyvex.stmt.Put(const(7), offset))
    call = node(0x400010, jumpkind="Ijk_Call")
    use = node(0x400020)
    facts = setup([entry, call, use], [(entry, call), (call, use)])
    assert facts.value(use, get(offset)) == expected
    assert facts.values(use, get(offset)) == (
        frozenset({expected}) if expected is not None else None
    )
    facts.project.loader.main_object.os = "unknown"
    unknown_abi = PredecessorFacts(facts.project, facts.graph, facts.bounds, {})
    assert unknown_abi.value(use, get(offset)) is None


def test_callee_entry_does_not_inherit_callers_preserved_registers():
    entry = node(0x400000, pyvex.stmt.Put(const(7), RBX), jumpkind="Ijk_Call")
    callee = node(0x400100)
    facts = setup([entry, callee], [(entry, callee)])
    assert facts.value(callee, get(RBX)) is None


def test_taken_exit_ignores_later_fallthrough_writes():
    entry = node(
        0x400000,
        pyvex.stmt.Put(const(7), RBX),
        pyvex.stmt.Exit(
            const(1, 1), pyvex.const.U64(0x400020), "Ijk_Boring", ARCH.ip_offset
        ),
        pyvex.stmt.Put(const(99), RBX),
    )
    use = node(0x400020)
    entry.block.vex.next = const(0x400010)
    facts = setup([entry, use], [(entry, use)])
    assert facts.value(use, get(RBX)) == 7
    assert facts.values(use, get(RBX)) == frozenset({7})


@pytest.mark.parametrize("arch", [ARCH, archinfo.ArchMIPS32()])
def test_exact_dispatch_edges_pin_only_the_destination_carrying_register(arch):
    offset = arch.registers["rbx" if arch.name == "AMD64" else "t0"][0]
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(
            0, pyvex.expr.Load("Iend_LE", f"Ity_I{arch.bits}", get(ARCH.sp_offset))
        ),
        pyvex.stmt.Put(pyvex.expr.RdTmp(0), offset),
        types=(f"Ity_I{arch.bits}",),
    )
    entry.block.vex.next = pyvex.expr.RdTmp(0)
    left, right = node(0x400010), node(0x400020)
    use = node(0x400030)
    facts = setup(
        [entry, left, right, use],
        [(entry, left), (entry, right), (left, use), (right, use)],
        arch=arch,
    )
    for target in (left, right):
        facts.graph.edges[entry, target]["proven_dispatch"] = True
        assert facts.value(target, get(offset, arch.bits)) == target.addr
        assert facts.values(target, get(offset, arch.bits)) == frozenset({target.addr})
    assert facts.value(use, get(offset, arch.bits)) is None
    assert facts.values(use, get(offset, arch.bits)) == frozenset(
        {left.addr, right.addr}
    )
    assert facts.values(use, get(ARCH.sp_offset)) is None
    assert not facts.exhausted


@pytest.mark.parametrize(
    "failure", ["unmarked", "candidate", "unresolved", "overwrite", "alias", "exit"]
)
def test_dispatch_edge_facts_require_exact_flow_and_an_unchanged_carrier(failure):
    statements = [
        pyvex.stmt.WrTmp(0, get(RBX)),
        pyvex.stmt.Put(pyvex.expr.RdTmp(0), RBX),
    ]
    if failure == "overwrite":
        statements.append(pyvex.stmt.Put(get(RAX), RBX))
    elif failure == "alias":
        statements.append(pyvex.stmt.Put(const(1, 8), RBX + 1))
    elif failure == "exit":
        statements.append(
            pyvex.stmt.Exit(
                get(RAX, 1), pyvex.const.U64(0x400020), "Ijk_Boring", ARCH.ip_offset
            )
        )
    entry = node(0x400000, *statements, types=("Ity_I64",))
    entry.block.vex.next = pyvex.expr.RdTmp(0)
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    edge = facts.graph.edges[entry, use]
    edge["proven_dispatch"] = failure != "unmarked"
    edge["candidate"] = failure == "candidate"
    edge["unresolved_indirect"] = failure == "unresolved"
    assert facts.value(use, get(RBX)) is None
    assert facts.values(use, get(RBX)) is None


def test_dispatch_fact_does_not_drop_an_unknown_incoming_path():
    entry = node(0x400000, pyvex.stmt.WrTmp(0, get(RBX)), types=("Ity_I64",))
    entry.block.vex.next = pyvex.expr.RdTmp(0)
    unknown, use = node(0x400010), node(0x400020)
    facts = setup([entry, unknown, use], [(entry, use), (unknown, use)])
    facts.graph.edges[entry, use]["proven_dispatch"] = True
    assert facts.value(use, get(RBX)) is None
    assert facts.values(use, get(RBX)) is None


@pytest.mark.parametrize("offset,preserved", [(RBX, True), (RAX, False)])
def test_dispatch_carrier_facts_obey_call_clobbers(offset, preserved):
    entry = node(0x400000, pyvex.stmt.WrTmp(0, get(offset)), types=("Ity_I64",))
    entry.block.vex.next = pyvex.expr.RdTmp(0)
    call = node(0x400010, jumpkind="Ijk_Call")
    use = node(0x400020)
    facts = setup([entry, call, use], [(entry, call), (call, use)])
    facts.graph.edges[entry, call]["proven_dispatch"] = True
    assert facts.values(use, get(offset)) == (
        frozenset({call.addr}) if preserved else None
    )


@pytest.mark.parametrize("writable,expected", [(False, 37), (True, None)])
def test_generic_memory_reads_require_immutable_bytes(writable, expected):
    entry = node(0x400000)
    facts = setup([entry], [], writable=writable)
    load = pyvex.expr.Load("Iend_LE", "Ity_I64", const(0x500000))
    assert facts.value(entry, load) == expected


def test_dynamic_stack_load_is_unknown():
    entry = node(0x400000)
    facts = setup([entry], [])
    load = pyvex.expr.Load("Iend_LE", "Ity_I64", get(ARCH.sp_offset))
    assert facts.value(entry, load) is None


@pytest.mark.parametrize("bits", [1, 8, 16])
def test_storage_width_alone_does_not_prove_a_dispatch_domain(bits):
    entry = node(0x400000)
    facts = setup([entry], [])
    assert facts.values(entry, get(RBX, bits)) is None


def test_unknown_byte_load_and_width_views_remain_unknown():
    entry = node(0x400000)
    facts = setup([entry], [], writable=True)
    load = pyvex.expr.Load("Iend_LE", "Ity_I8", const(0x500000))
    assert facts.values(entry, load) is None
    assert facts.values(entry, pyvex.expr.Unop("Iop_8Uto64", [load])) is None
    assert facts.values(entry, pyvex.expr.Unop("Iop_64to8", [get(RBX)])) is None
    assert facts.values(entry, pyvex.expr.Unop("Iop_Not8", [get(RBX, 8)])) is None


def test_byte_constants_masks_and_immutable_reads_still_prove_domains():
    entry = node(0x400000, pyvex.stmt.Put(const(7, 8), RBX))
    facts = setup([entry], [])
    assert facts.values(entry, get(RBX, 8)) == frozenset({7})
    assert facts.values(
        entry, pyvex.expr.Load("Iend_LE", "Ity_I8", const(0x500000))
    ) == frozenset({37})
    assert facts.values(
        entry, pyvex.expr.Binop("Iop_And8", [get(RAX, 8), const(3, 8)])
    ) == frozenset(range(4))


def test_matching_byte_guard_still_proves_a_domain():
    entry = node(
        0x400000,
        pyvex.stmt.Exit(
            pyvex.expr.Binop("Iop_CmpLT8U", [get(RBX, 8), const(4, 8)]),
            pyvex.const.U64(0x400020),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
    )
    entry.block.vex.next = const(0x400030)
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RBX, 8)) == frozenset(range(4))


def test_adapter_linkage_root_is_distinct_from_arbitrary_writable_memory():
    entry = node(0x400000)
    facts = setup([entry], [], writable=True)
    facts.linkage_slots = frozenset({0x500000})
    assert (
        facts.value(entry, pyvex.expr.Load("Iend_LE", "Ity_I64", const(0x500000))) == 37
    )
    assert (
        facts.value(entry, pyvex.expr.Load("Iend_LE", "Ity_I64", const(0x500008)))
        is None
    )
    assert (
        facts.value(entry, pyvex.expr.Load("Iend_LE", "Ity_I32", const(0x500000)))
        is None
    )


def test_syscall_and_candidate_edges_are_barriers():
    entry = node(0x400000, pyvex.stmt.Put(const(7), RBX), jumpkind="Ijk_Sys_syscall")
    use = node(0x400010)
    facts = setup([entry, use], [(entry, use)])
    assert facts.value(use, get(RBX)) is None
    entry.block.vex.jumpkind = "Ijk_Boring"
    facts.graph.edges[entry, use]["candidate"] = True
    candidate = PredecessorFacts(facts.project, facts.graph, facts.bounds, {})
    assert candidate.value(use, get(RBX)) is None


def test_budget_limits_work_and_cached_queries_do_not_rescan():
    nodes = [node(0x400000 + 16 * i) for i in range(8)]
    nodes[0] = node(0x400000, pyvex.stmt.Put(const(7), RBX))
    facts = setup(nodes, list(zip(nodes, nodes[1:])))
    assert facts.value(nodes[-1], get(RBX)) == 7
    steps = facts.steps
    assert facts.value(nodes[-1], get(RBX)) == 7
    assert facts.steps == steps + 1
    small = setup(nodes, list(zip(nodes, nodes[1:])), max_steps=3)
    assert small.value(nodes[-1], get(RBX)) is None
    assert small.exhausted


def test_distinct_uses_share_intermediate_scans_within_budget():
    entry = node(0x400000, pyvex.stmt.Put(const(7), RBX))
    middle = node(0x400010, *[pyvex.stmt.Put(const(i), RAX) for i in range(80)])
    first, second = node(0x400020), node(0x400030)
    facts = setup(
        [entry, middle, first, second],
        [(entry, middle), (middle, first), (middle, second)],
        max_steps=100,
    )
    assert facts.value(first, get(RBX)) == 7
    steps = facts.steps
    assert facts.value(second, get(RBX)) == 7
    assert facts.steps - steps < 10
    assert not facts.exhausted


def test_cached_scans_do_not_share_conflicting_join_results():
    entry = node(0x400000)
    left = node(0x400010, pyvex.stmt.Put(const(7), RBX))
    right = node(0x400020, pyvex.stmt.Put(const(8), RBX))
    first, second = node(0x400030), node(0x400040)
    facts = setup(
        [entry, left, right, first, second],
        [
            (entry, left),
            (entry, right),
            (left, first),
            (right, first),
            (left, second),
            (right, second),
        ],
    )
    assert facts.value(first, get(RBX)) is None
    assert facts.value(second, get(RBX)) is None
    assert facts.value(left, get(RBX)) == 7
    assert facts.value(right, get(RBX)) == 8


def test_cached_scans_distinguish_positions_and_alias_widths():
    entry = node(
        0x400000,
        pyvex.stmt.Put(const(7), RBX),
        pyvex.stmt.Put(const(9, 8), RBX + 1),
    )
    facts = setup([entry], [])
    assert facts.value(entry, get(RBX), before=1) == 7
    assert facts.value(entry, get(RBX)) is None
    assert facts.value(entry, get(RBX + 1, 8)) == 9


def test_interrupted_local_scan_is_not_cached():
    entry = node(0x400000, pyvex.stmt.Put(const(7), RBX))
    use = node(0x400010, *[pyvex.stmt.Put(const(i), RAX) for i in range(8)])
    facts = setup([entry, use], [(entry, use)], max_steps=3)
    assert facts.value(use, get(RBX)) is None
    assert facts.exhausted
    assert (use, 8, RBX, 64) not in facts._write_cache


@pytest.mark.parametrize("arch", [ARCH, archinfo.ArchMIPS32()])
def test_finite_mask_flows_through_register_copies_on_multiple_architectures(arch):
    first, second = list(arch.registers.values())[:2]
    offset, copied = first[0], second[0]
    bits = arch.bits
    entry = node(
        0x400000,
        pyvex.stmt.Put(
            pyvex.expr.Binop(f"Iop_And{bits}", [get(offset, bits), const(5, bits)]),
            offset,
        ),
    )
    copy = node(0x400010, pyvex.stmt.Put(get(offset, bits), copied))
    use = node(0x400020)
    facts = setup([entry, copy, use], [(entry, copy), (copy, use)], arch=arch)
    assert facts.values(use, get(copied, bits)) == frozenset({0, 1, 4, 5})


@pytest.mark.parametrize("other,expected", [(8, frozenset({7, 8})), (None, None)])
def test_finite_domains_union_all_paths_without_dropping_unknown_inputs(
    other, expected
):
    entry = node(0x400000)
    left = node(0x400010, pyvex.stmt.Put(const(7), RBX))
    right = node(
        0x400020, *([] if other is None else [pyvex.stmt.Put(const(other), RBX)])
    )
    use = node(0x400030)
    facts = setup(
        [entry, left, right, use],
        [(entry, left), (entry, right), (left, use), (right, use)],
    )
    assert facts.values(use, get(RBX)) == expected


@pytest.mark.parametrize("taken", [True, False])
def test_unsigned_guard_proves_a_domain_on_taken_and_fallthrough_paths(taken):
    comparison = pyvex.expr.Binop(
        "Iop_CmpLT64U", [get(RBX), const(4)] if taken else [const(3), get(RBX)]
    )
    entry = node(
        0x400000,
        pyvex.stmt.Exit(
            comparison,
            pyvex.const.U64(0x400020 if taken else 0x400030),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
    )
    entry.block.vex.next = const(0x400030 if taken else 0x400020)
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RBX)) == frozenset(range(4))


def test_nonzero_mask_path_does_not_include_the_zero_table_row():
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(0, pyvex.expr.Binop("Iop_And64", [get(RBX), const(3)])),
        pyvex.stmt.Put(pyvex.expr.RdTmp(0), RBX),
        pyvex.stmt.Exit(
            pyvex.expr.Binop("Iop_CmpEQ64", [pyvex.expr.RdTmp(0), const(0)]),
            pyvex.const.U64(0x400030),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=("Ity_I64",),
    )
    entry.block.vex.next = const(0x400020)
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RBX)) == frozenset({1, 2, 3})


def test_guard_of_old_register_does_not_bound_a_later_overwrite():
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(0, get(RBX)),
        pyvex.stmt.Put(get(RAX), RBX),
        pyvex.stmt.Exit(
            pyvex.expr.Binop("Iop_CmpLT64U", [pyvex.expr.RdTmp(0), const(4)]),
            pyvex.const.U64(0x400020),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=("Ity_I64",),
    )
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RBX)) is None


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("bound", [64, 128])
@pytest.mark.parametrize("taken", [True, False])
def test_guard_domain_flows_through_subtraction_and_restore(bits, bound, taken):
    old = pyvex.expr.RdTmp(0)
    guard = pyvex.expr.Binop(
        f"Iop_CmpLT{bits}U" if taken else f"Iop_CmpLE{bits}U",
        [old, const(bound, bits)] if taken else [const(bound, bits), old],
    )
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(0, get(RBX, bits)),
        pyvex.stmt.Put(
            pyvex.expr.Binop(f"Iop_Sub{bits}", [old, const(bound, bits)]), RBX
        ),
        pyvex.stmt.Exit(
            guard,
            pyvex.const.U64(0x400020 if taken else 0x400000),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=(f"Ity_I{bits}",),
    )
    entry.block.vex.next = const(0x400000 if taken else 0x400020)
    use = node(
        0x400020,
        pyvex.stmt.Put(
            pyvex.expr.Binop(f"Iop_Add{bits}", [get(RBX, bits), const(bound, bits)]),
            RBX,
        ),
    )
    facts = setup([entry, use], [(entry, entry), (entry, use)])
    # A bounded exit proves the remainder without following loop iterations.
    mask = (1 << bits) - 1
    assert facts.values(use, get(RBX, bits), before=0) == frozenset(
        (value - bound) & mask for value in range(bound)
    )
    assert facts.values(use, get(RBX, bits)) == frozenset(range(bound))
    assert not facts.exhausted


@pytest.mark.parametrize("bound", [1, 4, 16])
def test_guard_domain_maps_nested_arithmetic_without_assuming_correlations(bound):
    old = pyvex.expr.RdTmp(0)
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(0, get(RBX)),
        pyvex.stmt.Put(
            pyvex.expr.Binop(
                "Iop_Add64",
                [
                    const(0x500000),
                    pyvex.expr.Binop("Iop_Shl64", [old, const(2, 8)]),
                ],
            ),
            RAX,
        ),
        pyvex.stmt.Exit(
            pyvex.expr.Binop("Iop_CmpLT64U", [old, const(bound)]),
            pyvex.const.U64(0x400020),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=("Ity_I64",),
    )
    entry.block.vex.next = const(0x400030)
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RAX)) == frozenset(
        0x500000 + 4 * value for value in range(bound)
    )


@pytest.mark.parametrize("failure", ["unrelated", "alias", "width", "unsupported"])
def test_arithmetic_guard_proof_rejects_unmatched_or_unsupported_inputs(failure):
    old = pyvex.expr.RdTmp(0)
    operand = get(RAX) if failure == "unrelated" else get(RBX)
    statements = [pyvex.stmt.WrTmp(0, get(RBX))]
    if failure == "alias":
        statements.append(pyvex.stmt.Put(const(1, 8), RBX + 1))
    if failure == "width":
        operand = pyvex.expr.Unop("Iop_32Uto64", [get(RBX, 32)])
    operation = "Iop_Xor64" if failure == "unsupported" else "Iop_Sub64"
    statements.extend(
        [
            pyvex.stmt.Put(pyvex.expr.Binop(operation, [operand, const(4)]), RBX),
            pyvex.stmt.Exit(
                pyvex.expr.Binop("Iop_CmpLT64U", [old, const(4)]),
                pyvex.const.U64(0x400020),
                "Ijk_Boring",
                ARCH.ip_offset,
            ),
        ]
    )
    entry = node(0x400000, *statements, types=("Ity_I64",))
    entry.block.vex.next = const(0x400030)
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RBX)) is None


def test_arithmetic_guard_proof_does_not_drop_an_unknown_join_path():
    old = pyvex.expr.RdTmp(0)
    entry = node(0x400000)
    guarded = node(
        0x400010,
        pyvex.stmt.WrTmp(0, get(RBX)),
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_Sub64", [old, const(4)]), RBX),
        pyvex.stmt.Exit(
            pyvex.expr.Binop("Iop_CmpLT64U", [old, const(4)]),
            pyvex.const.U64(0x400030),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=("Ity_I64",),
    )
    guarded.block.vex.next = const(0x400040)
    unknown, use = node(0x400020), node(0x400030)
    facts = setup(
        [entry, guarded, unknown, use],
        [(entry, guarded), (entry, unknown), (guarded, use), (unknown, use)],
    )
    assert facts.values(use, get(RBX)) is None


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("taken", [True, False])
@pytest.mark.parametrize("reversed_operands", [True, False])
def test_signed_guard_filters_a_proven_wrapped_domain_before_restoration(
    bits, taken, reversed_operands
):
    old = pyvex.expr.RdTmp(0)
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(0, get(RBX, bits)),
        pyvex.stmt.Put(
            pyvex.expr.Binop(f"Iop_Sub{bits}", [old, const(128, bits)]), RBX
        ),
        pyvex.stmt.Exit(
            pyvex.expr.Binop(f"Iop_CmpLT{bits}U", [old, const(128, bits)]),
            pyvex.const.U64(0x400010),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=(f"Ity_I{bits}",),
    )
    entry.block.vex.next = const(0x400000)
    signed_bound = const((1 << bits) - 64, bits)
    # Both orientations express old < -64, before restoring old+128.
    comparison = pyvex.expr.Binop(
        f"Iop_CmpLE{bits}S" if reversed_operands else f"Iop_CmpLT{bits}S",
        [signed_bound, old] if reversed_operands else [old, signed_bound],
    )
    branch_taken = taken != reversed_operands
    branch = node(
        0x400010,
        pyvex.stmt.WrTmp(0, get(RBX, bits)),
        pyvex.stmt.Put(
            pyvex.expr.Binop(f"Iop_Add{bits}", [old, const(128, bits)]), RBX
        ),
        pyvex.stmt.Exit(
            comparison,
            pyvex.const.U64(0x400020 if branch_taken else 0x400030),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=(f"Ity_I{bits}",),
    )
    branch.block.vex.next = const(0x400030 if branch_taken else 0x400020)
    use = node(0x400020)
    facts = setup([entry, branch, use], [(entry, branch), (branch, use)])
    assert facts.values(use, get(RBX, bits)) == frozenset(
        range(64) if taken else range(64, 128)
    )


def test_guard_equality_maps_arithmetic_but_exclusion_does_not_assume_injectivity():
    old = pyvex.expr.RdTmp(0)
    entry = node(
        0x400000,
        pyvex.stmt.WrTmp(0, get(RBX)),
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [old, const(1)]), RBX),
        pyvex.stmt.Exit(
            pyvex.expr.Binop("Iop_CmpEQ64", [old, const(0)]),
            pyvex.const.U64(0x400020),
            "Ijk_Boring",
            ARCH.ip_offset,
        ),
        types=("Ity_I64",),
    )
    entry.block.vex.next = const(0x400030)
    equal, other = node(0x400020), node(0x400030)
    facts = setup([entry, equal, other], [(entry, equal), (entry, other)])
    assert facts.values(equal, get(RBX)) == frozenset({0})
    # Nonzero old values can still map to zero after masking.
    assert facts.values(other, get(RBX)) == frozenset({0, 1})


def test_arithmetic_results_share_work_without_caching_an_interrupted_product():
    entry = node(0x400000)
    facts = setup([entry], [])
    left, right = frozenset(range(128)), frozenset({128})
    expected = frozenset((v - 128) & ((1 << 64) - 1) for v in left)
    assert facts._combine("Iop_Sub64", 64, left, right) == expected
    steps = facts.steps
    assert facts._combine("Iop_Sub64", 64, left, right) == expected
    assert facts.steps == steps
    small = setup([entry], [], max_steps=2)
    assert small._combine("Iop_Sub64", 64, left, right) is None
    assert small.exhausted
    assert not small._combinations


@pytest.mark.parametrize(
    "guard",
    [
        pyvex.expr.Binop("Iop_CmpLT64S", [get(RBX), const(4)]),
        pyvex.expr.Unop(
            "Iop_Not1", [pyvex.expr.Binop("Iop_CmpLT64U", [get(RBX), const(4)])]
        ),
        pyvex.expr.Binop("Iop_CmpLT64U", [get(RBX), const(257)]),
    ],
)
def test_signed_inverted_and_oversized_guards_are_not_treated_as_small_domains(guard):
    entry = node(
        0x400000,
        pyvex.stmt.Exit(guard, pyvex.const.U64(0x400020), "Ijk_Boring", ARCH.ip_offset),
    )
    use = node(0x400020)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(use, get(RBX)) is None


def test_finite_domains_wrap_arithmetic_and_stop_loop_carried_searches():
    entry = node(0x400000, pyvex.stmt.Put(const((1 << 64) - 1), RBX))
    use = node(0x400010)
    facts = setup([entry, use], [(entry, use)])
    assert facts.values(
        use, pyvex.expr.Binop("Iop_Add64", [get(RBX), const(1)])
    ) == frozenset({0})
    loop = node(
        0x400020,
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_Add64", [get(RBX), const(1)]), RBX),
    )
    cyclic = setup([entry, loop, use], [(entry, loop), (loop, loop), (loop, use)])
    assert cyclic.values(use, get(RBX)) is None
    assert not cyclic.exhausted
    small = setup([entry, use], [(entry, use)], max_steps=2)
    assert small.values(use, get(RBX)) is None
    assert small.exhausted
