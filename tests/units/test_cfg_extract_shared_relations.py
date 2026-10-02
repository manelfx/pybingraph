"""Bounded relational facts, copied definitions, and lazy-flag semantics."""

from angr.engines.vex.claripy.ccall import data as x86_cc_data
import archinfo
import pyvex
import pytest

from test_cfg_extract_shared_facts import ARCH, RAX, RBX, const, get, node, setup


def exit_to(comparison, address, arch=ARCH):
    return pyvex.stmt.Exit(
        comparison, pyvex.const.U64(address), "Ijk_Boring", arch.ip_offset
    )


def ordered_fixture(*, arch=ARCH, equality=True, swapped=True, bypass=False):
    first, second = {
        "AMD64": ("rbx", "rax"),
        "MIPS32": ("s0", "s1"),
        "S390X": ("r2", "r3"),
    }[arch.name]
    a, b = (arch.registers[name][0] for name in (first, second))
    bits = arch.bits
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(
                pyvex.expr.Binop(f"Iop_And{bits}", [get(r, bits), const(15, bits)]), r
            )
            for r in (a, b)
        ),
        *(
            [
                exit_to(
                    pyvex.expr.Binop(f"Iop_CmpEQ{bits}", [get(a, bits), get(b, bits)]),
                    0x400070,
                    arch,
                )
            ]
            if equality
            else []
        ),
        arch=arch,
    )
    entry.block.vex.next = const(0x400010)
    branch = node(
        0x400010,
        exit_to(
            pyvex.expr.Binop(f"Iop_CmpLE{bits}U", [get(a, bits), get(b, bits)]),
            0x400030,
            arch,
        ),
        arch=arch,
    )
    branch.block.vex.next = const(0x400020)
    swap = node(
        0x400020,
        pyvex.stmt.WrTmp(0, get(a, bits)),
        pyvex.stmt.WrTmp(1, get(b, bits)),
        pyvex.stmt.Put(pyvex.expr.RdTmp(1), a),
        pyvex.stmt.Put(pyvex.expr.RdTmp(0), b),
        types=(f"Ity_I{bits}", f"Ity_I{bits}"),
        arch=arch,
    )
    swap.block.vex.next = const(0x400030)
    use = node(0x400030, arch=arch)
    nodes, edges = [entry, branch, use], [(entry, branch), (branch, use)]
    if swapped:
        nodes.append(swap)
        edges.extend([(branch, swap), (swap, use)])
    if bypass:
        # A separately bounded but unordered path remains relevant at the join.
        other = node(0x400050, *entry.block.vex.statements[:2], arch=arch)
        other.block.vex.next = const(use.addr)
        nodes.append(other)
        edges.append((other, use))
    expression = pyvex.expr.Binop(
        f"Iop_Sub{bits}",
        [
            pyvex.expr.Binop(f"Iop_Add{bits}", [get(a, bits), const(15, bits)]),
            get(b, bits),
        ],
    )
    return setup(nodes, edges, arch=arch), use, expression, a, b


@pytest.mark.parametrize("arch", [ARCH, archinfo.ArchMIPS32(), archinfo.ArchS390X()])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("equality", [False, True])
def test_ordered_finite_operands_retain_their_relation_through_swaps(
    arch, swapped, equality
):
    facts, use, expr, _, _ = ordered_fixture(
        arch=arch, equality=equality, swapped=swapped
    )
    assert facts.values(use, expr) == frozenset(range(15 if equality else 16))
    assert not facts.exhausted


def test_an_unordered_join_path_is_not_dropped():
    facts, use, expr, _, _ = ordered_fixture(bypass=True)
    assert facts.values(use, expr) == frozenset(range(31))


def test_signed_relations_and_byte_arithmetic_preserve_machine_wrapping():
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(
                pyvex.expr.Binop(
                    "Iop_Sub8",
                    [
                        pyvex.expr.Binop("Iop_And8", [get(r, 8), const(15, 8)]),
                        const(8, 8),
                    ],
                ),
                r,
            )
            for r in (RBX, RAX)
        ),
        exit_to(pyvex.expr.Binop("Iop_CmpLT8S", [get(RBX, 8), get(RAX, 8)]), 0x400010),
    )
    entry.block.vex.next = const(0x400020)
    use = node(0x400010)
    facts = setup([entry, use], [(entry, use)])
    expr = pyvex.expr.Binop("Iop_Sub8", [get(RBX, 8), get(RAX, 8)])
    assert facts.values(use, expr) == frozenset(range(241, 256))


def test_candidate_edges_cannot_establish_a_relation():
    facts, use, expr, _, _ = ordered_fixture()
    source = next(facts.graph.predecessors(use))
    facts.graph.get_edge_data(source, use)["candidate"] = True
    assert facts._relations.values(use, expr, 0) is None


def test_a_later_overwrite_cannot_inherit_the_old_value_relation():
    facts, use, expr, a, _ = ordered_fixture()
    use.block.vex.statements.append(pyvex.stmt.Put(const(0), a))
    # A newly assigned zero and the old second operand can legitimately be
    # equal. The old a<b constraint cannot remove that new equality.
    assert 15 in facts.values(use, expr)


def test_unknown_alias_writes_and_predecessors_decline_refinement():
    facts, use, expr, a, _ = ordered_fixture()
    use.block.vex.statements.append(
        pyvex.stmt.Put(get(ARCH.registers["dl"][0], 8), a + 1)
    )
    assert facts.values(use, expr) is None


def test_old_temporaries_do_not_observe_later_overwrites():
    facts, use, expr, a, _ = ordered_fixture()
    use.block.vex.statements.extend(
        [
            pyvex.stmt.WrTmp(0, get(a)),
            pyvex.stmt.Put(const(15), a),
        ]
    )
    use.block.vex.tyenv.types.append("Ity_I64")
    old = pyvex.expr.RdTmp(0)
    expr.args[0].args[0] = old
    assert facts.values(use, expr) == frozenset(range(15))


def test_relational_cache_does_not_repeat_a_completed_query():
    facts, use, expr, _, _ = ordered_fixture()
    assert facts._relations.values(use, expr, 0) == frozenset(range(15))
    steps = facts.steps
    assert facts._relations.values(use, expr, 0) == frozenset(range(15))
    assert facts.steps == steps


@pytest.mark.parametrize("bits,mask", [(32, 31), (64, 31)])
def test_pair_enumeration_cap_and_wrapping_remain_conservative(bits, mask):
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(
                pyvex.expr.Binop(f"Iop_And{bits}", [get(r, bits), const(mask, bits)]), r
            )
            for r in (RBX, RAX)
        ),
    )
    expr = pyvex.expr.Binop(f"Iop_Sub{bits}", [get(RBX, bits), get(RAX, bits)])
    facts = setup([entry], [])
    assert facts._relations.values(entry, expr, 2) is None
    assert not facts.exhausted


def test_relational_budget_failure_is_not_cached_as_a_proof():
    facts, use, expr, _, _ = ordered_fixture()
    facts.max_steps = 100
    assert facts._relations.values(use, expr, 0) is None
    assert facts.exhausted
    assert not facts._relations.answers


@pytest.mark.parametrize("operand", ["rbx", "rax"])
def test_extra_guard_operands_obey_call_clobbers_even_when_query_registers_are_preserved(
    operand,
):
    rbp = ARCH.registers["rbp"][0]
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(r), const(15)]), r)
            for r in (RBX, rbp)
        ),
        pyvex.stmt.Put(get(RBX), RAX),
    )
    entry.block.vex.next = const(0x400010)
    call = node(0x400010, jumpkind="Ijk_Call")
    call.block.vex.next = const(0x500000)
    branch = node(
        0x400020,
        exit_to(
            pyvex.expr.Binop(
                "Iop_CmpLT64U", [get(ARCH.registers[operand][0]), get(rbp)]
            ),
            0x400030,
        ),
    )
    branch.block.vex.next = const(0x400040)
    use = node(0x400030)
    facts = setup(
        [entry, call, branch, use], [(entry, call), (call, branch), (branch, use)]
    )
    expr = pyvex.expr.Binop(
        "Iop_Sub64", [pyvex.expr.Binop("Iop_Add64", [get(RBX), const(15)]), get(rbp)]
    )
    assert facts.values(use, expr) == frozenset(range(15 if operand == "rbx" else 31))


def test_doubling_loop_rewrites_remain_bounded_without_expanding_a_tree():
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(r), const(15)]), r)
            for r in (RBX, RAX)
        ),
    )
    loop = node(
        0x400010,
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_Add64", [get(RBX), get(RBX)]), RBX),
    )
    use = node(0x400020)
    facts = setup([entry, loop, use], [(entry, loop), (loop, loop), (loop, use)])
    query = pyvex.expr.Binop("Iop_Sub64", [get(RBX), get(RAX)])
    assert facts._relations.values(use, query, 0) is None
    assert facts.steps < facts.max_steps
    assert len(facts._relations.terms) < 1000


@pytest.mark.parametrize(
    "mask,expected",
    [
        (2, {15}),
        (4, set(range(16, 31))),
        (8, set(range(15))),
        (10, set(range(16))),
        (6, set(range(15, 31))),
    ],
)
def test_angr_ordinal_comparisons_keep_their_relation_to_finite_operands(
    mask, expected
):
    arch = archinfo.ArchPPC64(endness="Iend_LE")
    a, b = (arch.registers[name][0] for name in ("r14", "r15"))
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(r), const(15)]), r)
            for r in (a, b)
        ),
        arch=arch,
    )
    comparison = pyvex.expr.Binop("Iop_CmpORD64U", [get(a), get(b)])
    guard = pyvex.expr.Binop(
        "Iop_CmpNE64",
        [
            pyvex.expr.Binop(
                "Iop_And64", [pyvex.expr.Unop("Iop_32Uto64", [comparison]), const(mask)]
            ),
            const(0),
        ],
    )
    branch = node(0x400010, exit_to(guard, 0x400020, arch), arch=arch)
    branch.block.vex.next = const(0x400030)
    use = node(0x400020, arch=arch)
    facts = setup([entry, branch, use], [(entry, branch), (branch, use)], arch=arch)
    query = pyvex.expr.Binop(
        "Iop_Sub64", [pyvex.expr.Binop("Iop_Add64", [get(a), const(15)]), get(b)]
    )
    assert facts.values(use, query) == frozenset(expected)


def test_angr_flag_helpers_support_logic_without_another_flag_adapter():
    registers = ARCH.registers
    query = pyvex.expr.Binop("Iop_Add64", [get(RBX), get(RAX)])
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(r), const(15)]), r)
            for r in (RBX, RAX)
        ),
        pyvex.stmt.Put(
            const(x86_cc_data["AMD64"]["OpTypes"]["G_CC_OP_LOGICQ"]),
            registers["cc_op"][0],
        ),
        pyvex.stmt.Put(query, registers["cc_dep1"][0]),
        pyvex.stmt.Put(const(0), registers["cc_dep2"][0]),
    )
    guard = pyvex.expr.CCall(
        "Ity_I64",
        pyvex.IRCallee(0, "amd64g_calculate_condition", 0),
        [
            const(x86_cc_data["AMD64"]["CondTypes"]["CondZ"]),
            *(get(registers[name][0]) for name in ("cc_op", "cc_dep1", "cc_dep2")),
            const(0),
        ],
    )
    branch = node(0x400010, exit_to(guard, 0x400020))
    branch.block.vex.next = const(0x400030)
    use = node(0x400020)
    facts = setup([entry, branch, use], [(entry, branch), (branch, use)])
    assert facts.values(use, query) == frozenset({0})


def test_angr_integer_operations_reuse_finite_roots_and_unknown_memory_stays_unknown():
    entry = node(0x400000)
    facts = setup([entry], [])
    masks = [pyvex.expr.Binop("Iop_And64", [get(r), const(15)]) for r in (RBX, RAX)]
    query = pyvex.expr.Binop("Iop_Xor64", masks)
    assert facts._relations.values(entry, query, 0) == frozenset(range(16))
    assert (
        facts._relations.values(
            entry, pyvex.expr.Load("Iend_LE", "Ity_I64", get(RBX)), 0
        )
        is None
    )
    assert facts._relations.values(entry, pyvex.expr.GSPTR(), 0) is None


def test_different_loop_iterations_cannot_share_a_mask_definition_identity():
    r12 = ARCH.registers["r12"][0]
    entry = node(0x400000, pyvex.stmt.Put(const(0), RBX))
    loop = node(
        0x400010,
        pyvex.stmt.Put(get(RBX), r12),
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(RAX), const(15)]), RBX),
        pyvex.stmt.WrTmp(0, pyvex.expr.Binop("Iop_Sub64", [get(r12), get(RBX)])),
        pyvex.stmt.Put(pyvex.expr.RdTmp(0), ARCH.registers["rdx"][0]),
        types=("Ity_I64",),
    )
    facts = setup([entry, loop], [(entry, loop), (loop, loop)])
    # The query observes values from consecutive iterations. The same static
    # mask instruction can produce two different values; subtraction is not 0.
    assert facts._relations.values(loop, pyvex.expr.RdTmp(0), 4) is None


@pytest.mark.parametrize("with_call", [False, True])
def test_lazy_flags_cannot_survive_an_intervening_call(with_call):
    rbp = ARCH.registers["rbp"][0]
    registers = ARCH.registers
    entry = node(
        0x400000,
        *(
            pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(r), const(15)]), r)
            for r in (RBX, rbp)
        ),
        pyvex.stmt.Put(const(7), registers["cc_op"][0]),
        pyvex.stmt.Put(get(RBX), registers["cc_dep1"][0]),
        pyvex.stmt.Put(get(rbp), registers["cc_dep2"][0]),
    )
    comparison = pyvex.expr.CCall(
        "Ity_I64",
        pyvex.IRCallee(0, "amd64g_calculate_condition", 0),
        [
            const(2),
            *(get(registers[name][0]) for name in ("cc_op", "cc_dep1", "cc_dep2")),
            const(0),
        ],
    )
    branch = node(0x400020, exit_to(comparison, 0x400030))
    branch.block.vex.next = const(0x400040)
    use = node(0x400030)
    nodes, edges = [entry, branch, use], [(entry, branch), (branch, use)]
    entry.block.vex.next = const(branch.addr)
    if with_call:
        call = node(0x400010, jumpkind="Ijk_Call")
        call.block.vex.next = const(0x500000)
        entry.block.vex.next = const(call.addr)
        nodes.append(call)
        edges = [(entry, call), (call, branch), (branch, use)]
    facts = setup(nodes, edges)
    expr = pyvex.expr.Binop(
        "Iop_Sub64", [pyvex.expr.Binop("Iop_Add64", [get(RBX), const(15)]), get(rbp)]
    )
    assert facts.values(use, expr) == frozenset(range(31 if with_call else 15))


@pytest.mark.parametrize(
    "arch,bits",
    [(arch, bits) for arch in (ARCH, archinfo.ArchX86()) for bits in (8, 16, 32)]
    + [(ARCH, 64)],
)
@pytest.mark.parametrize(
    "condition", ["B", "NB", "BE", "NBE", "Z", "NZ", "L", "NL", "LE", "NLE"]
)
def test_subtraction_flag_adapter_preserves_width_signedness_and_polarity(
    arch, bits, condition
):
    entry = node(0x400000, arch=arch)
    facts = setup([entry], [], arch=arch)
    data = x86_cc_data[arch.name]
    suffix = {8: "B", 16: "W", 32: "L", 64: "Q"}[bits]
    helper = (
        "amd64g_calculate_condition"
        if arch.name == "AMD64"
        else "x86g_calculate_condition"
    )
    comparison = pyvex.expr.CCall(
        f"Ity_I{arch.bits}",
        pyvex.IRCallee(0, helper, 0),
        [
            const(data["CondTypes"][f"Cond{condition}"], arch.bits),
            const(data["OpTypes"][f"G_CC_OP_SUB{suffix}"], arch.bits),
            get(arch.registers["cc_dep1"][0], arch.bits),
            get(arch.registers["cc_dep2"][0], arch.bits),
            const(0, arch.bits),
        ],
    )
    result, before, inverted = facts._guard_comparison(entry, comparison, 0)
    relation = (
        "EQ"
        if condition in {"Z", "NZ"}
        else "LE"
        if condition.removeprefix("N") in {"BE", "LE"}
        else "LT"
    )
    sign = (
        ""
        if relation == "EQ"
        else "U"
        if condition.removeprefix("N") in {"B", "BE"}
        else "S"
    )
    assert result.op == f"Iop_Cmp{relation}{bits}{sign}"
    assert all(arg.result_size(entry.block.vex.tyenv) == bits for arg in result.args)
    assert before == 0
    assert inverted == condition.startswith("N")


@pytest.mark.parametrize("arch", [ARCH, archinfo.ArchX86()])
@pytest.mark.parametrize("operation", ["ADD", "LOGIC", "COPY", "unknown"])
def test_non_subtraction_or_unknown_flag_producers_are_rejected(arch, operation):
    entry = node(0x400000, arch=arch)
    facts = setup([entry], [], arch=arch)
    data = x86_cc_data[arch.name]
    first, second = (arch.registers[name][0] for name in ("ebx", "eax"))
    op = (
        get(first, arch.bits)
        if operation == "unknown"
        else const(
            data["OpTypes"][
                "G_CC_OP_COPY" if operation == "COPY" else f"G_CC_OP_{operation}L"
            ],
            arch.bits,
        )
    )
    guard = pyvex.expr.CCall(
        f"Ity_I{arch.bits}",
        pyvex.IRCallee(
            0,
            "amd64g_calculate_condition"
            if arch.name == "AMD64"
            else "x86g_calculate_condition",
            0,
        ),
        [
            const(6, arch.bits),
            op,
            get(first, arch.bits),
            get(second, arch.bits),
            const(0, arch.bits),
        ],
    )
    assert facts._guard_comparison(entry, guard, 0) is None
