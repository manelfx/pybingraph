"""Shadow comparisons for the shared relative-table proof."""

from pathlib import Path

from angr import KnowledgeBase, options as sim_options
import pytest

from bingraph.cfg_extract.builder import _ExtractionSession, build_extracted_cfg
from bingraph.cfg_extract.shared_table_proof import (
    shared_register_targets,
    shared_table_targets,
    table_predecessor_facts,
)
from bingraph.core.project import load_project
from test_cfg_extract_shared_facts import RBX, const, get, node, setup
import pyvex


@pytest.mark.parametrize(
    ("binary", "function", "expect_match"),
    [
        ("mipsel/busybox", 0x40FDC0, True),
        ("mips64/ld.so.1", 0x402988, True),
        ("mips64/ld.so.1", 0x41ABD8, True),
        ("ppc64el/fauxware_static", 0x10002390, True),
        ("ppc64el/fauxware_static", 0x100985E0, True),
        ("x86_64/static", 0x42C6B0, True),
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


@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize("failure", [None, "writable", "unmapped", "non_executable"])
def test_shared_proof_requires_every_row_and_destination_to_be_valid(relative, failure):
    from types import SimpleNamespace

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

    def read(address, size):
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
def test_finite_register_targets_require_every_destination_to_be_valid(reject_target):
    from types import SimpleNamespace

    entry = node(
        0x400000,
        pyvex.stmt.Put(pyvex.expr.Binop("Iop_And64", [get(RBX), const(1)]), RBX),
    )
    use = node(0x400010)
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
    assert cfg.extract_stats.exact_jump_proofs_by_flavor["shared_finite_register"] == 60
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
