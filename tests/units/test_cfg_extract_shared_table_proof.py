"""Independent shared-table migration and remaining legacy shadow proofs."""

from pathlib import Path
from types import SimpleNamespace

from angr import KnowledgeBase, options as sim_options
import pytest

from bingraph.cfg.jumps import plan_dynamic_selector_table_candidates
from bingraph.cfg_extract.builder import _ExtractionSession, build_extracted_cfg
from bingraph.cfg_extract import builder as builder_module
from bingraph.cfg_extract import shared_table_proof as proof_module
from bingraph.cfg_extract.shared_table_proof import (
    shared_register_targets,
    shared_table_targets,
    table_predecessor_facts,
)
from bingraph.core.project import load_project
from test_cfg_extract_shared_facts import RBX, const, get, node, setup
import pyvex


@pytest.mark.parametrize(
    "function,dispatcher",
    [
        (0x426490, 0x426583),
        (0x4286D0, 0x4287B7),
        (0x42A280, 0x42A347),
        (0x42C6B0, 0x42C723),
        (0x4354F0, 0x4355E3),
        (0x4911C0, 0x4912D3),
        (0x493810, 0x493917),
        (0x4957E0, 0x4958C7),
        (0x497190, 0x4972A3),
    ],
)
def test_shared_affine_tables_prove_exact_targets_without_candidates(
    function: int, dispatcher: int
) -> None:
    """Resolve ordered affine dispatches without speculative table rows."""

    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    session = _ExtractionSession(project, KnowledgeBase(project), function)
    session._decode_all_blocks()
    graph, nodes = session._analysis_graph({})

    # Exact register-derived targets must not seed heuristic table candidates.
    assert (
        plan_dynamic_selector_table_candidates(
            project, graph, session.bounds, nodes[dispatcher]
        )
        is None
    )

    session._discover_static_jump_targets()

    assert len(session.static_targets[dispatcher]) == 15
    assert session.stats.exact_jump_proofs_by_flavor == {"shared_finite_table": 1}
    assert dispatcher not in session.unresolved_dispatcher_reasons
    assert not session.static_target_candidates
    assert session.stats.shared_fact_budget_exhausted == 0


@pytest.mark.parametrize(
    ("binary", "function", "expect_match"),
    [
        ("mipsel/busybox", 0x40FDC0, True),
        ("mips64/ld.so.1", 0x402988, True),
        ("mips64/ld.so.1", 0x41ABD8, True),
        ("ppc64el/fauxware_static", 0x10002390, True),
        ("ppc64el/fauxware_static", 0x100985E0, True),
        # These bases require shared predecessor facts rather than the old
        # same-block evaluator or MIPS-only fallback.
        ("i386/nl", 0x402710, True),
        ("x86_64/elf_with_static_libc_ubuntu_2004", 0x48EF40, True),
        ("x86_64/bomb", 0x400F43, False),
        # Static GOT bytes without local relocation evidence remain unknown.
        ("mipsel/mips_syscall_demo", 0x408814, False),
    ],
)
def test_shared_shadow_proof_does_not_change_exact_targets(
    monkeypatch: pytest.MonkeyPatch, binary: str, function: int, expect_match: bool
) -> None:
    monkeypatch.setenv("BINGRAPH_SHADOW_TABLE_PROOFS", "1")
    project = load_project(Path("angr-binaries/tests") / binary)
    cfg = build_extracted_cfg(project, KnowledgeBase(project), function)

    assert cfg.extract_stats.shadow_table_attempts >= 1
    if expect_match:
        assert cfg.extract_stats.shadow_table_matches >= 1
    else:
        assert cfg.extract_stats.shadow_table_inconclusive >= 1
    assert cfg.extract_stats.shadow_table_disagreements == 0
    assert cfg.extract_stats.shadow_table_attempts == (
        cfg.extract_stats.shadow_table_matches
        + cfg.extract_stats.shadow_table_inconclusive
        + cfg.extract_stats.shadow_table_disagreements
    )


def test_repeated_dispatch_searches_share_scans_without_exhausting_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BINGRAPH_SHADOW_TABLE_PROOFS", "1")
    project = load_project(Path("angr-binaries/tests/mipsel/mips_syscall_demo"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x42C460)

    # This large function previously repeated the same backward statement
    # scans at each dispatch and exhausted the shared 20K allowance.
    assert cfg.extract_stats.shadow_fact_steps < 20000
    assert cfg.extract_stats.shadow_fact_budget_exhausted == 0
    assert cfg.extract_stats.shadow_table_attempts == 14
    assert cfg.extract_stats.shadow_table_matches == 1
    assert cfg.extract_stats.shadow_table_inconclusive == 13
    assert cfg.extract_stats.shadow_table_disagreements == 0


@pytest.mark.parametrize("conditional_lift_fails", [False, True])
@pytest.mark.parametrize(
    "binary,function,proofs",
    [
        ("x86_64/static", 0x42C6B0, 1),
        ("x86_64/static", 0x493810, 1),
        ("x86_64/rust_hello_world", 0x411330, 2),
        ("x86_64/rust_hello_world", 0x426600, 2),
        ("s390x/test-instr_s390x", 0x80067160, 1),
        ("s390x/test-instr_s390x", 0x80067F90, 1),
        ("i386/bronze_ropchain", 0x80A6D90, 1),
        ("i386/nl", 0x403CF0, 1),
        ("armel/RTOSDemo.axf.issue_685", 0xA59, 1),
        ("armel/lwip_udpecho_bm.elf", 0x2CA9, 2),
        ("armhf/amp_challenge_07.gcc.dyn.unstripped", 0x401CB9, 1),
    ],
)
def test_shared_primary_discovers_cfg_without_any_legacy_table_rescue(
    monkeypatch, binary, function, proofs, conditional_lift_fails
):
    """Start from entry decoding, not code already recovered by a legacy proof."""

    project = load_project(Path("angr-binaries/tests") / binary)
    baseline = _ExtractionSession(project, KnowledgeBase(project), function)
    baseline.build()
    if conditional_lift_fails:
        # Failure of the separate conditional-PC re-lift must not hide the
        # usable VEX already attached to an ordinary table's decoded block.
        monkeypatch.setattr(
            builder_module,
            "conditional_pc_dispatch_targets",
            lambda *_args: (None, "no_vex"),
        )

    def forbidden(*_args, **_kwargs):
        pytest.fail("A migrated table fell back to a legacy resolver")

    for name in (
        "plan_static_jump_table",
        "plan_mips_pic_relative_jump_table",
    ):
        monkeypatch.setattr(builder_module, name, forbidden)
    monkeypatch.setenv("BINGRAPH_SHADOW_TABLE_PROOFS", "1")
    independent = _ExtractionSession(project, KnowledgeBase(project), function)
    independent.build()

    def edges(session):
        return {
            (source.addr, source.is_simprocedure, target.addr, target.is_simprocedure)
            for source, target in session.graph.edges()
        }

    assert independent.blocks == baseline.blocks
    assert edges(independent) == edges(baseline)
    assert independent.static_targets == baseline.static_targets
    assert independent.stats.exact_jump_proofs_by_flavor == {
        "shared_finite_table": proofs
    }
    assert independent.stats.legacy_table_fallback_attempts == 0
    assert independent.stats.shadow_table_attempts == 0
    assert independent.stats.unresolved_indirect_targets == 0
    assert independent.stats.shared_fact_budget_exhausted == 0
    assert independent.stats.sweep_runs == 0
    assert independent.stats.output_anomaly_count == 0


@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize("failure", [None, "writable", "unmapped", "non_executable"])
def test_shared_proof_requires_every_row_and_destination_to_be_valid(relative, failure):
    entry = node(
        0x400000,
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(RBX), const(1)]), RBX),
    )
    use = node(0x400010)
    facts = setup([entry, use], [(entry, use)], writable=failure == "writable")
    address = pyvex.expr.Binop(
        "Iop_Add64",
        [const(0x500000), pyvex.expr.Binop("Iop_Shl64", [get(RBX), const(3, 8)])],
    )
    load = pyvex.expr.Load("Iend_LE", "Ity_I64", address)
    use.block.vex.next = (
        pyvex.expr.Binop("Iop_Add64", [const(0x400000), load]) if relative else load
    )

    reads = []

    def read(address, size):
        reads.append(address)
        if failure == "unmapped" and address == 0x500008:
            raise KeyError(address)
        target = 0x400020 if address == 0x500000 else 0x400030
        return (target - 0x400000 if relative else target).to_bytes(size, "little")

    facts.project.loader.memory.load = read
    facts.project.loader.find_object_containing = lambda address: SimpleNamespace(
        find_section_containing=lambda address: SimpleNamespace(
            is_executable=not (failure == "non_executable" and address == 0x400030)
        )
    )
    assert shared_table_targets(facts.project, use, facts) == (
        (0x400020, 0x400030) if failure is None else None
    )
    if failure is None:
        steps = facts.steps
        assert shared_table_targets(facts.project, use, facts) == (0x400020, 0x400030)
        assert reads == [0x500000, 0x500008]
        assert facts.steps <= steps + 1  # The scalar relative base is rechecked.
    else:
        assert all(key[0] != 0x500008 for key in facts._table_rows)


@pytest.mark.parametrize(
    "binary,function,dispatcher,expected",
    [
        (
            "armhf/amp_challenge_07.gcc.dyn.unstripped",
            0x401D29,
            0x401D31,
            {0x401D39, 0x401D59, 0x401D61},
        ),
        (
            "armel/lwip_udpecho_bm.elf",
            0x41DD,
            0x4747,
            {0x4775, 0x4865, 0x4937, 0x493F, 0x49D1, 0x4A05, 0x4A43},
        ),
    ],
)
def test_shared_engine_recovers_compact_table_edges_without_legacy_plans(
    monkeypatch, binary, function, dispatcher, expected
):
    """Keep byte/halfword target and leader regressions after matcher removal."""

    def forbidden(*_args, **_kwargs):
        pytest.fail("A compact table depended on a legacy plan")

    monkeypatch.setattr(builder_module, "plan_static_jump_table", forbidden)
    project = load_project(Path("angr-binaries/tests") / binary)
    cfg = build_extracted_cfg(project, KnowledgeBase(project), function)
    nodes = {node.addr: node for node in cfg.graph if not node.is_simprocedure}
    assert {node.addr for node in cfg.graph.successors(nodes[dispatcher])} == expected
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.exact_jump_proofs_by_flavor == {"shared_finite_table": 1}
    assert cfg.extract_stats.static_jump_target_edges_added == len(expected)
    assert cfg.extract_stats.unresolved_indirect_targets == 0
    assert cfg.extract_stats.shared_fact_budget_exhausted == 0


def expression_fixture(raw, bits=8, endian="little", **kwargs):
    """Use immutable file bytes and executable destinations, not a legacy plan."""

    use = node(0x400000)
    facts = setup([use], [], **kwargs)
    facts.project.loader.memory.load = lambda _address, size: raw.to_bytes(size, endian)
    facts.project.loader.find_object_containing = lambda _address: SimpleNamespace(
        find_section_containing=lambda _address: SimpleNamespace(is_executable=True)
    )
    load = pyvex.expr.Load(
        "Iend_LE" if endian == "little" else "Iend_BE",
        f"Ity_I{bits}",
        const(0x500000),
    )
    return use, facts, load


@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize(
    "raw,bits,operations,expected",
    [
        (7, 8, [("Iop_8Uto64",), ("Iop_Shl64", 1), ("Iop_Add64", 0x400000)], 0x40000E),
        (0xFE, 8, [("Iop_8Sto64",), ("Iop_Add64", 0x400000)], 0x3FFFFE),
        (0xFFFE, 16, [("Iop_16Sto64",), ("Iop_Add64", 0x400000)], 0x3FFFFE),
        (
            3,
            8,
            [
                ("Iop_8Uto64",),
                ("Iop_Add64", 0x400000),
                ("Iop_And64", 0xFFFFFFFFFFFFFFFC),
            ],
            0x400000,
        ),
        (6, 8, [("Iop_8Uto64",), ("Iop_Shr64", 1), ("Iop_Or64", 0x400000)], 0x400003),
        (3, 8, [("Iop_8Uto64",), ("Iop_Sub64", 0x400020, False)], 0x40001D),
        (
            1,
            8,
            [
                ("Iop_8Uto64",),
                ("Iop_Add64", 0xFFFFFFFFFFFFFFFF),
                ("Iop_Add64", 0x400000),
            ],
            0x400000,
        ),
        (
            0x100000002,
            64,
            [("Iop_64to32",), ("Iop_32Uto64",), ("Iop_Add64", 0x400000)],
            0x400002,
        ),
    ],
)
def test_table_target_program_preserves_width_order_and_operand_direction(
    raw, bits, operations, expected, endian
):
    use, facts, expression = expression_fixture(raw, bits, endian)
    for operation in operations:
        op, *args = operation
        if not args:
            expression = pyvex.expr.Unop(op, [expression])
        else:
            operands = [expression, const(args[0], 8 if "Sh" in op else 64)]
            if len(args) > 1 and not args[1]:
                operands.reverse()
            expression = pyvex.expr.Binop(op, operands)
    use.block.vex.next = expression
    assert shared_table_targets(facts.project, use, facts) == (expected,)
    steps = facts.steps
    assert shared_table_targets(facts.project, use, facts) == (expected,)
    assert facts.steps == steps


@pytest.mark.parametrize(
    "failure", ["unknown", "two_loads", "unsupported", "shift", "budget"]
)
def test_unknown_or_unsupported_target_program_preserves_fallback(failure):
    use, facts, load = expression_fixture(
        7, 64, max_steps=1 if failure == "budget" else 20000
    )
    other = (
        get(RBX)
        if failure == "unknown"
        else pyvex.expr.Load("Iend_LE", "Ity_I64", get(RBX))
        if failure == "two_loads"
        else const(64)
    )
    op = (
        "Iop_Mul64"
        if failure == "unsupported"
        else "Iop_Shl64"
        if failure == "shift"
        else "Iop_Add64"
    )
    if failure == "shift":
        other = const(64, 8)
    use.block.vex.next = pyvex.expr.Binop(op, [load, other])
    assert shared_table_targets(facts.project, use, facts) is None
    assert not facts._table_rows
    if failure == "budget":
        assert not facts._table_expressions


def test_target_program_cache_keeps_operation_chains_distinct():
    use, facts, load = expression_fixture(7, 64)
    for op, expected in [("Iop_Add64", 0x400027), ("Iop_Sub64", 0x400019)]:
        use.block.vex.next = pyvex.expr.Binop(op, [const(0x400020), load])
        assert shared_table_targets(facts.project, use, facts) == (expected,)
    assert len(facts._table_rows) == 2


def test_table_expression_retains_the_loads_register_read_position():
    use, facts, _ = expression_fixture(0x400020, 64)
    use.block.vex.statements = [
        pyvex.stmt.Put(const(0x500000), RBX),
        pyvex.stmt.WrTmp(0, pyvex.expr.Load("Iend_LE", "Ity_I64", get(RBX))),
        pyvex.stmt.Put(const(0x600000), RBX),
    ]
    use.block.vex.tyenv.add("Ity_I64")
    use.block.vex.next = pyvex.expr.RdTmp(0)
    reads = []

    def read(address, size):
        reads.append(address)
        return (0x400020).to_bytes(size, "little")

    facts.project.loader.memory.load = read
    assert shared_table_targets(facts.project, use, facts) == (0x400020,)
    assert reads == [0x500000]


@pytest.mark.parametrize(
    "binary,function,source,table,count,size,endian",
    [
        ("s390x/test-instr_s390x", 0x80067160, 0x80067188, 0x8008A220, 13, 8, "big"),
        ("s390x/test-instr_s390x", 0x80067F90, 0x80068288, 0x8008A408, 5, 8, "big"),
        (
            "x86_64/1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51",
            0x43F940,
            0x43F988,
            0x45F0B8,
            13,
            4,
            "little",
        ),
        ("x86_64/rust_hello_world", 0x426600, 0x42664B, 0x44A060, 13, 4, "little"),
        ("x86_64/rust_hello_world", 0x426600, 0x426725, 0x44A094, 5, 4, "little"),
    ],
)
def test_typed_guards_independently_bound_rust_and_s390_dispatches(
    binary, function, source, table, count, size, endian
):
    project = load_project(Path("angr-binaries/tests") / binary)
    session = _ExtractionSession(project, KnowledgeBase(project), function)
    session._decode_all_blocks()
    session._discover_static_jump_targets()
    # Legacy recovery makes decoded code available, but supplies no selector
    # bounds or exact dispatch edges to this independent shared-engine query.
    graph, nodes = session._analysis_graph()
    facts = table_predecessor_facts(project, graph, session.bounds)
    actual = shared_table_targets(project, nodes[source], facts)
    expected = {
        (
            table
            + int.from_bytes(
                project.loader.memory.load(table + index * size, size),
                endian,
                signed=True,
            )
        )
        & ((1 << project.arch.bits) - 1)
        for index in range(count)
    }
    assert actual == tuple(sorted(expected))
    assert not facts.exhausted


@pytest.mark.parametrize(
    "address,dispatcher,count",
    [(0x42F210, 0x42F284, 15), (0x432280, 0x432308, 15), (0x42DBC0, 0x42DBD0, 79)],
)
def test_shared_finite_proof_recovers_masked_or_guarded_static_tables(
    address, dispatcher, count
):
    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), address)
    nodes = {node.addr: node for node in cfg.graph if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[dispatcher]))
    assert len(successors) == count
    assert all(not node.is_simprocedure for node in successors)
    assert (
        cfg.extract_stats.exact_jump_proofs_by_flavor.get("shared_finite_table", 0) >= 1
    )
    assert cfg.extract_stats.shared_fact_budget_exhausted == 0


@pytest.mark.parametrize(
    "function,dispatchers,base,count,remaining_ujts",
    [
        (0x42DBC0, (0x42E09F, 0x42E11D, 0x42E5AF, 0x42E629), 0x4A2C70, 64, 0),
        (
            0x42F210,
            (0x42F417, 0x431AE8, 0x431BB6, 0x431CA4),
            0x4A2DB0,
            64,
            0,
        ),
        (0x42F210, (0x42F635, 0x431D64), 0x4A2DB0, 64, 0),
        (
            0x432280,
            (
                0x432407,
                0x432544,
                0x4326C4,
                0x432844,
                0x4329C4,
                0x432B44,
                0x432CC4,
                0x432E44,
                0x432FC4,
                0x433144,
                0x4332C4,
                0x433444,
                0x4335C5,
                0x433744,
                0x4338C4,
                0x433A44,
                0x433CB1,
            ),
            0x4A31B0,
            128,
            0,
        ),
        (
            0x432280,
            (
                0x432488,
                0x432601,
                0x432781,
                0x432901,
                0x432A81,
                0x432C01,
                0x432D81,
                0x432F01,
                0x433081,
                0x433201,
                0x433381,
                0x433501,
                0x433681,
                0x433801,
                0x433981,
                0x433B01,
                0x433E5F,
            ),
            0x4A2F70,
            128,
            0,
        ),
    ],
)
def test_arithmetic_guard_proofs_resolve_all_memcpy_tail_rows(
    function, dispatchers, base, count, remaining_ujts
):
    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), function)
    expected = {
        base
        + int.from_bytes(
            project.loader.memory.load(base + 4 * index, 4), "little", signed=True
        )
        for index in range(count)
    }
    nodes = {node.addr: node for node in cfg.graph if not node.is_simprocedure}
    for address in dispatchers:
        successors = tuple(cfg.graph.successors(nodes[address]))
        assert all(not node.is_simprocedure for node in successors)
        assert {node.addr for node in successors} == expected
        assert all(
            not cfg.graph.get_edge_data(nodes[address], target).get("candidate")
            for target in successors
        )
    unresolved = [
        source
        for source in cfg.graph
        if any(
            target.is_simprocedure and target.name == "UnresolvableJumpTarget"
            for target in cfg.graph.successors(source)
        )
    ]
    assert len(unresolved) == remaining_ujts
    assert cfg.extract_stats.shared_fact_budget_exhausted == 0


@pytest.mark.parametrize("reject_target", [False, True])
@pytest.mark.parametrize("jumpkind", ["Ijk_Boring", "Ijk_Call"])
def test_finite_register_targets_require_every_destination_to_be_valid(
    reject_target, jumpkind
):
    from types import SimpleNamespace

    entry = node(
        0x400000,
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(RBX), const(1)]), RBX),
    )
    use = node(0x400010, jumpkind=jumpkind)
    use.block.vex.next = pyvex.expr.Binop(
        "Iop_Add64",
        [const(0x400020), pyvex.expr.Binop("Iop_Shl64", [get(RBX), const(4, 8)])],
    )
    facts = setup([entry, use], [(entry, use)])
    facts.project.loader.find_object_containing = lambda address: SimpleNamespace(
        find_section_containing=lambda address: SimpleNamespace(
            is_executable=not (reject_target and address == 0x400030)
        )
    )
    assert shared_register_targets(facts.project, use, facts) == (
        None if reject_target else (0x400020, 0x400030)
    )


@pytest.mark.parametrize("jumpkind", ["Ijk_Boring", "Ijk_Call"])
def test_folded_register_call_is_exact_but_linear_next_is_not_a_dispatch(jumpkind):
    use = node(0x400000, jumpkind=jumpkind)
    use.block.vex.next = const(0x400010)
    facts = setup([use], [])
    facts.project.loader.find_object_containing = lambda address: SimpleNamespace(
        find_section_containing=lambda address: SimpleNamespace(is_executable=True)
    )

    assert shared_register_targets(facts.project, use, facts) == (
        (0x400010,) if jumpkind == "Ijk_Call" else None
    )


@pytest.mark.parametrize("failure", [None, "unknown", "alias", "budget", "callee"])
def test_shared_call_uses_scalar_loop_facts_without_losing_safety(monkeypatch, failure):
    target = 0x500000
    entry = node(0x400000, pyvex.stmt.Put(const(target), RBX))
    loop = node(
        0x400010,
        *([pyvex.stmt.Put(const(1, 8), RBX + 1)] if failure == "alias" else []),
    )
    use = node(0x400020, jumpkind="Ijk_Call")
    use.block.vex.next = get(RBX)
    facts = setup([entry, loop, use], [(entry, loop), (loop, loop), (loop, use)])
    if failure == "unknown":
        unknown = node(0x400030)
        facts.graph.add_edge(unknown, use, jumpkind="Ijk_Boring")
    elif failure == "budget":
        facts.max_steps = 1
    monkeypatch.setattr(
        proof_module,
        "_is_static_pointer_call_target",
        lambda project, addr: addr == target and failure != "callee",
    )

    assert shared_register_targets(facts.project, use, facts) == (
        (target,) if failure is None else None
    )


def test_shared_relro_load_resolves_register_call_to_memcpy() -> None:
    """Recover a register-carried memcpy target from protected RELRO bytes."""

    project = load_project(Path("angr-binaries/tests/x86_64/fmt-rust"))
    session = _ExtractionSession(project, KnowledgeBase(project), 0x4D9A10)
    session._decode_all_blocks()
    session._discover_static_jump_targets()

    call = session.blocks[0x4D9A80]
    assert 0x4D9A8B in call.instruction_addrs
    assert call.direct_targets
    symbol = project.loader.find_symbol(call.direct_targets[0])
    assert symbol is not None and symbol.name == "memcpy"
    assert session.stats.shared_fact_budget_exhausted == 0


@pytest.mark.parametrize(
    "function,source,table,count",
    [
        (0x42F210, 0x42F352, 0x4A2DB0, 65),
        (0x42F210, 0x42F576, 0x4A2DB0, 65),
        (0x42F210, 0x42F506, 0x4A2DB0, 32),
        (0x42F210, 0x42F726, 0x4A2DB0, 32),
        (0x42DBC0, 0x42E319, 0x4A2C70, 32),
        (0x42DBC0, 0x42DE10, 0x4A2C70, 32),
        (0x42DBC0, 0x42E524, 0x4A2C70, 32),
        (0x42DBC0, 0x42E00C, 0x4A2C70, 32),
    ],
)
def test_lower_guards_independently_bound_every_memmove_and_memcmp_tail_row(
    function, source, table, count
):
    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    session = _ExtractionSession(project, KnowledgeBase(project), function)
    session._decode_all_blocks()
    session._discover_static_jump_targets()
    # Decoding supplies instructions, not legacy selector domains or dispatch
    # edges. Read the table only after the shared engine proves the row set.
    graph, nodes = session._analysis_graph()
    facts = table_predecessor_facts(project, graph, session.bounds)
    rdx = project.arch.registers["rdx"][0]
    assert facts.values(nodes[source], get(rdx), before=0) == frozenset(range(count))
    expected = {
        table
        + int.from_bytes(
            project.loader.memory.load(table + 4 * index, 4), "little", signed=True
        )
        for index in range(count)
    }
    assert set(shared_table_targets(project, nodes[source], facts)) == expected
    assert not facts.exhausted


@pytest.mark.parametrize(
    "source,destination,origin,length",
    [(0x42F352, 0x10000000, 0x20000000, 160), (0x42F576, 0x20000040, 0x20000000, 144)],
)
def test_memmove_row_64_has_a_concrete_path_from_function_entry(
    source, destination, origin, length
):
    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    state = project.factory.blank_state(
        addr=0x42F210,
        add_options={
            sim_options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            sim_options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.regs.rdi, state.regs.rsi, state.regs.rdx = destination, origin, length
    state.memory.store(origin, bytes(range(length)))
    state.memory.store(0x6CA0B0, (4096).to_bytes(8, "little"))
    # Execute concrete instructions, independently of both CFG implementations.
    # The legacy proof's missing endpoint is reachable in each copy direction.
    for _ in range(32):
        if state.addr == source:
            assert state.solver.eval(state.regs.rdx) == 64
            successors = project.factory.successors(state).flat_successors
            assert len(successors) == 1
            assert successors[0].addr == 0x430B30
            break
        successors = project.factory.successors(
            state, extra_stop_points={source}
        ).flat_successors
        assert len(successors) == 1
        state = successors[0]
    else:
        pytest.fail("The concrete path did not reach the tail dispatcher")


@pytest.mark.parametrize(
    "function,source,table",
    [
        (0x426490, 0x426583, 0x4A2B70),
        (0x4286D0, 0x4287B7, 0x4A2AF0),
        (0x42A280, 0x42A347, 0x4A2B30),
        (0x42C6B0, 0x42C723, 0x4A2C30),
        (0x4354F0, 0x4355E3, 0x4A3470),
        (0x4911C0, 0x4912D3, 0x4BD500),
        (0x493810, 0x493917, 0x4BD480),
        (0x4957E0, 0x4958C7, 0x4BD4C0),
        (0x497190, 0x4972A3, 0x4BD540),
    ],
)
def test_relational_guards_independently_bound_comparison_table_rows(
    function, source, table
):
    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    session = _ExtractionSession(project, KnowledgeBase(project), function)
    session._decode_all_blocks()
    session._discover_static_jump_targets()
    graph, nodes = session._analysis_graph()
    facts = table_predecessor_facts(project, graph, session.bounds)
    # An independently proven a<b narrows a+15-b to 0..14. No legacy plan,
    # selector bound, table shape or dispatch edge is supplied to the query.
    expected = {
        table
        + int.from_bytes(
            project.loader.memory.load(table + 4 * index, 4), "little", signed=True
        )
        for index in range(15)
    }
    assert set(shared_table_targets(project, nodes[source], facts)) == expected
    assert not facts.exhausted


def test_memmove_retains_dispatch_carriers_through_initial_and_loop_jumps():
    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x42F210)
    nodes = {node.addr: node for node in cfg.graph if not node.is_simprocedure}
    for base in (0x4A2EF0, 0x4A2F30):
        for index in range(1, 16):
            entry = base + int.from_bytes(
                project.loader.memory.load(base + 4 * index, 4), "little", signed=True
            )
            adjustment = (
                project.factory.block(entry).capstone.insns[0].insn.operands[1].mem.disp
            )
            expected = {entry + adjustment, entry + adjustment - 7}
            # Instruction lengths vary in the alignment-eight routine. Derive
            # the two destinations from its actual LEA, not guessed offsets.
            sources = [
                source
                for address, source in nodes.items()
                if entry <= address < entry + 0xB0
                and any(
                    insn.mnemonic == "jmp" and insn.op_str == "r9"
                    for insn in project.factory.block(
                        address, size=source.size
                    ).capstone.insns
                )
            ]
            assert len(sources) == 2
            for source in sources:
                successors = tuple(cfg.graph.successors(source))
                assert all(not target.is_simprocedure for target in successors)
                assert {target.addr for target in successors} == expected
    # The same shared facts can now resolve carriers in the earlier ABI stage,
    # rather than waiting for the non-table jump fallback.
    assert (
        cfg.extract_stats.exact_jump_proofs_by_flavor.get("shared_finite_register", 0)
        + cfg.extract_stats.abi_static_jump_targets_resolved
    ) == 60
    assert cfg.extract_stats.unresolved_indirect_targets == 0
    assert cfg.extract_stats.shared_fact_budget_exhausted == 0
    assert cfg.extract_stats.output_anomaly_count == 0


@pytest.mark.parametrize(
    "binary,function,dispatcher",
    [
        (
            "x86_64/1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51",
            0x431280,
            0x431578,
        ),
        ("x86_64/rust_hello_world", 0x4230B0, 0x4230ED),
        ("x86_64/rust_hello_world", 0x4231A0, 0x4231FE),
    ],
)
def test_unconstrained_rust_bytes_keep_ujt_without_external_table_expansion(
    binary, function, dispatcher
):
    project = load_project(Path("angr-binaries/tests") / binary)
    cfg = build_extracted_cfg(project, KnowledgeBase(project), function)
    source = next(
        node
        for node in cfg.graph
        if node.addr == dispatcher and not node.is_simprocedure
    )
    successors = tuple(cfg.graph.successors(source))
    assert any(
        node.is_simprocedure and node.name == "UnresolvableJumpTarget"
        for node in successors
    )
    # Reading 256 rows previously produced over a hundred external SIMP nodes.
    # Preserve existing in-function dashed candidates, but no guessed externals.
    assert all(
        not node.is_simprocedure or node.name == "UnresolvableJumpTarget"
        for node in successors
    )
    assert (
        cfg.extract_stats.exact_jump_proofs_by_flavor.get("shared_finite_table", 0) == 0
    )


@pytest.mark.parametrize(
    ("binary", "function"),
    [
        ("mipsel/busybox", 0x40FDC0),
        ("i386/nl", 0x402710),
        ("x86_64/elf_with_static_libc_ubuntu_2004", 0x48EF40),
    ],
)
def test_shadow_proof_does_not_change_cfg(
    monkeypatch: pytest.MonkeyPatch, binary: str, function: int
) -> None:
    project = load_project(Path("angr-binaries/tests") / binary)
    monkeypatch.delenv("BINGRAPH_SHADOW_TABLE_PROOFS", raising=False)
    baseline = build_extracted_cfg(project, KnowledgeBase(project), function)
    monkeypatch.setenv("BINGRAPH_SHADOW_TABLE_PROOFS", "1")
    shadowed = build_extracted_cfg(project, KnowledgeBase(project), function)

    def shape(cfg):
        nodes = {(node.addr, node.is_simprocedure) for node in cfg.graph.nodes()}
        edges = {
            (source.addr, source.is_simprocedure, target.addr, target.is_simprocedure)
            for source, target in cfg.graph.edges()
        }
        return nodes, edges

    assert shape(shadowed) == shape(baseline)
    assert baseline.extract_stats.shadow_table_attempts == 0
    assert shadowed.extract_stats.shadow_table_matches >= 1
