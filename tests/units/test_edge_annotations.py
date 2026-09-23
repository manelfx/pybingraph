"""Fast tests for control-flow edge classification used by render styles."""

from types import SimpleNamespace

import pytest
from capstone import CS_GRP_JUMP, CS_OP_IMM
from capstone.x86 import X86_INS_JMP

import bingraph.core.annotators as annotators
from bingraph.core.annotators import _edge_type


def _node(
    addr: int,
    *,
    size: int = 0,
    vex: SimpleNamespace | None = None,
    capstone_insns: tuple[SimpleNamespace, ...] = (),
    graph: SimpleNamespace | None = None,
    arch_name: str = "X86",
    is_simprocedure: bool = False,
    simprocedure_name: str | None = None,
) -> SimpleNamespace:
    """Return the minimal visualization-node shape consumed by ``_edge_type``."""

    block = (
        SimpleNamespace(
            vex=vex,
            capstone=SimpleNamespace(
                insns=[SimpleNamespace(insn=insn) for insn in capstone_insns]
            ),
        )
        if vex is not None
        else None
    )
    return SimpleNamespace(
        obj=SimpleNamespace(
            addr=addr,
            size=size,
            block=block,
            is_simprocedure=is_simprocedure,
            simprocedure_name=simprocedure_name,
        ),
        project=SimpleNamespace(arch=SimpleNamespace(name=arch_name)),
        graph=graph,
    )


def _edge(
    source: SimpleNamespace,
    destination: SimpleNamespace,
    *,
    jumpkind: str,
    **meta: object,
) -> SimpleNamespace:
    """Return the minimal visualization-edge shape consumed by ``_edge_type``."""

    return SimpleNamespace(
        meta={"jumpkind": jumpkind, **meta},
        src=source,
        dst=destination,
    )


def _vex(
    jumpkind: str,
    *,
    next_addr: int | None = None,
    exit_targets: tuple[int, ...] = (),
    exit_jumpkind: str = "Ijk_Boring",
) -> SimpleNamespace:
    """Build the subset of VEX state used to classify ordinary edges."""

    if next_addr is None:
        next_expression = SimpleNamespace()
    else:
        next_expression = SimpleNamespace(con=SimpleNamespace(value=next_addr))
    return SimpleNamespace(
        jumpkind=jumpkind,
        next=next_expression,
        exit_statements=[
            (
                None,
                None,
                SimpleNamespace(
                    dst=SimpleNamespace(value=target),
                    jumpkind=exit_jumpkind,
                ),
            )
            for target in exit_targets
        ],
    )


@pytest.mark.parametrize(
    ("jumpkind", "expected"),
    [
        ("Ijk_Ret", "RET"),
        ("Ijk_FakeRet", "FAKE_RET"),
        ("Ijk_Call", "CALL"),
        ("Ijk_Sys_syscall", "CALL"),
        ("Ijk_Sys_int128", "CALL"),
    ],
)
def test_direct_vex_jumpkinds_use_expected_style(jumpkind: str, expected: str) -> None:
    """Map direct VEX control-flow jump kinds to stable render styles."""

    edge = _edge(_node(0x1000), _node(0x2000), jumpkind=jumpkind)

    assert _edge_type(edge) == expected


def test_explicit_unresolved_indirect_metadata_has_highest_precedence() -> None:
    """Keep repaired unresolved-indirect edges visible despite their jump kind."""

    edge = _edge(
        _node(0x1000),
        _node(0x2000),
        jumpkind="Ijk_Ret",
        unresolved_indirect=True,
    )

    assert _edge_type(edge) == "UNRESOLVED_INDIRECT"


def test_exceptional_metadata_uses_its_own_known_flow_style() -> None:
    """Render LSDA-proven unwinding separately from ordinary call flow."""

    edge = _edge(
        _node(0x1000),
        _node(0x2000),
        jumpkind="Ijk_Boring",
        exceptional=True,
    )

    assert _edge_type(edge) == "EXCEPTION"


@pytest.mark.parametrize("placeholder_side", ["source", "destination"])
def test_unresolvable_jump_target_edges_remain_unresolved(
    placeholder_side: str,
) -> None:
    """Classify both directions of angr's indirect-jump placeholder equally."""

    placeholder = _node(
        0x80100000,
        is_simprocedure=True,
        simprocedure_name="UnresolvableJumpTarget",
    )
    normal_node = _node(0x1000)
    source, destination = (
        (placeholder, normal_node)
        if placeholder_side == "source"
        else (normal_node, placeholder)
    )

    assert _edge_type(_edge(source, destination, jumpkind="Ijk_Boring")) == (
        "UNRESOLVED_INDIRECT"
    )


@pytest.mark.parametrize(
    ("vex", "destination", "expected"),
    [
        (_vex("Ijk_Boring", next_addr=0x1004), 0x1004, "NEXT"),
        (_vex("Ijk_Boring", next_addr=0x2000), 0x2000, "UNCONDITIONAL"),
        (
            _vex("Ijk_Boring", next_addr=0x1004, exit_targets=(0x2000,)),
            0x2000,
            "CONDITIONAL_TRUE",
        ),
        (
            _vex("Ijk_Boring", next_addr=0x2000, exit_targets=(0x2000,)),
            0x1004,
            "CONDITIONAL_FALSE",
        ),
        (
            _vex("Ijk_Boring", exit_targets=(0x1004,)),
            0x2000,
            "INDIRECT",
        ),
        (_vex("Ijk_Boring"), 0x2000, "INDIRECT"),
        (_vex("Ijk_Boring", next_addr=0x2000), 0x3000, "UNKNOWN"),
    ],
)
def test_boring_edges_use_vex_terminator_semantics(
    vex: SimpleNamespace, destination: int, expected: str
) -> None:
    """Distinguish ordinary fall-through, branch, and indirect CFG edges."""

    source = _node(0x1000, size=4, vex=vex)

    assert (
        _edge_type(_edge(source, _node(destination), jumpkind="Ijk_Boring")) == expected
    )


def test_boring_edge_to_direct_vex_call_target_is_a_call() -> None:
    """Recover a lost CFGFast call tag only for VEX's direct call target."""

    source = _node(0x1000, size=4, vex=_vex("Ijk_Call", next_addr=0x2000))

    assert _edge_type(_edge(source, _node(0x2000), jumpkind="Ijk_Boring")) == "CALL"
    assert _edge_type(_edge(source, _node(0x1004), jumpkind="Ijk_Boring")) == "UNKNOWN"


def test_conditional_indirect_call_uses_its_explicit_fallthrough_exit() -> None:
    """Style a conditional indirect call's not-taken path as a branch edge."""

    source = _node(
        0x1000,
        size=4,
        vex=_vex("Ijk_Call", exit_targets=(0x1004,)),
    )

    assert (
        _edge_type(_edge(source, _node(0x1004), jumpkind="Ijk_Boring"))
        == "CONDITIONAL_FALSE"
    )


def test_capstone_conditional_branch_overrides_folded_vex_default() -> None:
    """Keep both static successors when VEX has folded a branch condition."""

    target = _node(0x2000)
    fallthrough = _node(0x1008)
    graph = SimpleNamespace(successors=lambda _node: (target.obj, fallthrough.obj))
    branch = SimpleNamespace(
        address=0x1004,
        size=4,
        id=0,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Boring", next_addr=0x2000),
        capstone_insns=(branch,),
        graph=graph,
    )

    assert (
        _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "CONDITIONAL_TRUE"
    )
    assert (
        _edge_type(_edge(source, fallthrough, jumpkind="Ijk_Boring"))
        == "CONDITIONAL_FALSE"
    )


def test_capstone_conditional_branch_ignores_stale_internal_vex_exit() -> None:
    """Use CFGFast's instruction provenance when VEX stopped before the tail."""

    target = _node(0x2000)
    fallthrough = _node(0x1008)
    graph = SimpleNamespace(successors=lambda _node: (target.obj, fallthrough.obj))
    branch = SimpleNamespace(
        address=0x1004,
        size=4,
        id=0,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Boring", next_addr=0x1004, exit_targets=(0x1004,)),
        capstone_insns=(branch,),
        graph=graph,
    )

    assert (
        _edge_type(_edge(source, target, jumpkind="Ijk_Boring", ins_addr=0x1004))
        == "CONDITIONAL_TRUE"
    )
    assert (
        _edge_type(_edge(source, fallthrough, jumpkind="Ijk_Boring", ins_addr=0x1004))
        == "CONDITIONAL_FALSE"
    )


def test_capstone_conditional_branch_overrides_non_branch_vex_jumpkind() -> None:
    """Recover a direct branch after a VEX-modeled x86 ``pause`` instruction."""

    target = _node(0x2000)
    fallthrough = _node(0x1008)
    graph = SimpleNamespace(successors=lambda _node: (target.obj, fallthrough.obj))
    branch = SimpleNamespace(
        address=0x1004,
        size=4,
        id=0,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Yield", next_addr=0x1004),
        capstone_insns=(branch,),
        graph=graph,
    )

    assert (
        _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "CONDITIONAL_TRUE"
    )
    assert (
        _edge_type(_edge(source, fallthrough, jumpkind="Ijk_Boring"))
        == "CONDITIONAL_FALSE"
    )


@pytest.mark.parametrize(
    ("branch_addr", "displacement", "target_addr", "fallthrough_addr"),
    [
        (0x84C2, -0x2BC, 0x8206, 0x84C6),
        (0x87F6, 0xA, 0x8800, 0x87FA),
    ],
)
def test_capstone_riscv_relative_conditional_branch_overrides_truncated_vex(
    branch_addr: int,
    displacement: int,
    target_addr: int,
    fallthrough_addr: int,
) -> None:
    """Resolve RISC-V's PC-relative direct branch operands for edge styles."""

    target = _node(target_addr)
    fallthrough = _node(fallthrough_addr)
    graph = SimpleNamespace(successors=lambda _node: (target.obj, fallthrough.obj))
    branch = SimpleNamespace(
        address=branch_addr,
        size=4,
        id=0,
        groups=(CS_GRP_JUMP,),
        operands=(
            SimpleNamespace(type=0, imm=0),
            SimpleNamespace(type=CS_OP_IMM, imm=displacement),
        ),
    )
    source = _node(
        0x8206,
        size=branch_addr + 4 - 0x8206,
        vex=_vex("Ijk_Boring", next_addr=0x8374),
        capstone_insns=(branch,),
        graph=graph,
        arch_name="RISCV64",
    )

    assert (
        _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "CONDITIONAL_TRUE"
    )
    assert (
        _edge_type(_edge(source, fallthrough, jumpkind="Ijk_Boring"))
        == "CONDITIONAL_FALSE"
    )


@pytest.mark.parametrize("arch_name", ["RISCV64", "AMD64"])
def test_capstone_linear_tail_overrides_truncated_vex_default(
    arch_name: str,
) -> None:
    """Keep the decoded fall-through when VEX stops inside a long repaired block."""

    fallthrough = _node(0x403C)
    graph = SimpleNamespace(successors=lambda _node: (fallthrough.obj,))
    source = _node(
        0x3EA2,
        size=0x19A,
        vex=_vex("Ijk_Boring", next_addr=0x401A),
        capstone_insns=(
            SimpleNamespace(
                address=0x403A,
                size=2,
                id=0,
                groups=(),
                operands=(),
            ),
        ),
        graph=graph,
        arch_name=arch_name,
    )

    assert _edge_type(_edge(source, fallthrough, jumpkind="Ijk_Boring")) == "NEXT"


@pytest.mark.parametrize("jumpkind", ["Ijk_EmWarn", "Ijk_EmFail"])
def test_capstone_linear_tail_overrides_vex_error_jumpkind(jumpkind: str) -> None:
    """Classify a non-branch instruction despite VEX's error jumpkind."""

    fallthrough = _node(0x40B692)
    graph = SimpleNamespace(successors=lambda _node: (fallthrough.obj,))
    source = _node(
        0x40B68A,
        size=8,
        vex=_vex(jumpkind, next_addr=0x40B692),
        capstone_insns=(
            SimpleNamespace(
                address=0x40B68E,
                size=4,
                id=0,
                groups=(),
                operands=(),
            ),
        ),
        graph=graph,
        arch_name="S390X",
    )

    assert _edge_type(_edge(source, fallthrough, jumpkind="Ijk_Boring")) == "NEXT"


@pytest.mark.parametrize("jumpkind", ["Ijk_Yield", "Ijk_Sys_syscall"])
def test_capstone_direct_jump_overrides_non_branch_vex_jumpkind(jumpkind: str) -> None:
    """Recover a direct jump following a VEX-obscuring instruction."""

    target = _node(0x2000)
    graph = SimpleNamespace(successors=lambda _node: (target.obj,))
    jump = SimpleNamespace(
        address=0x1004,
        size=4,
        id=X86_INS_JMP,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex(jumpkind, next_addr=0x1004),
        capstone_insns=(jump,),
        graph=graph,
    )

    assert _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "UNCONDITIONAL"


def test_capstone_direct_jump_overrides_spurious_vex_conditional_state() -> None:
    """Style a proven direct branch even when VEX retains a false exit."""

    target = _node(0x2000)
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Boring", next_addr=0x1008, exit_targets=(0x2000,)),
        capstone_insns=(
            SimpleNamespace(
                address=0x1004,
                size=4,
                id=X86_INS_JMP,
                groups=(CS_GRP_JUMP,),
                operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
            ),
        ),
    )

    assert _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "UNCONDITIONAL"


def test_tail_lift_disambiguates_one_operand_direct_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the transfer-tail lift for generic unconditional branch encodings."""

    target = _node(0x2000)
    graph = SimpleNamespace(successors=lambda _node: (target.obj,))
    jump = SimpleNamespace(
        address=0x1004,
        size=4,
        id=0,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Boring", next_addr=0x1004, exit_targets=(0x1004,)),
        capstone_insns=(jump,),
        graph=graph,
        arch_name="MIPS32",
    )
    monkeypatch.setattr(
        annotators,
        "_lift_control_transfer_tail",
        lambda _edge, _tail: _vex("Ijk_Boring", next_addr=0x2000),
    )

    assert _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "UNCONDITIONAL"


def test_tail_lift_overrides_stale_full_block_exit_for_direct_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the direct branch tail when the full VEX lift stopped earlier."""

    target = _node(0x2000)
    graph = SimpleNamespace(successors=lambda _node: (target.obj,))
    jump = SimpleNamespace(
        address=0x1004,
        size=4,
        id=X86_INS_JMP,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x2000),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Boring", next_addr=0x1004, exit_targets=(0x1004,)),
        capstone_insns=(jump,),
        graph=graph,
    )
    monkeypatch.setattr(
        annotators,
        "_lift_control_transfer_tail",
        lambda _edge, _tail: _vex("Ijk_Boring", next_addr=0x2000),
    )

    assert _edge_type(_edge(source, target, jumpkind="Ijk_Boring")) == "UNCONDITIONAL"


def test_folded_conditional_with_sequential_target_remains_fallthrough() -> None:
    """Do not turn a degenerate direct target into a taken branch edge."""

    destination = _node(0x1008)
    graph = SimpleNamespace(successors=lambda _node: (destination.obj,))
    branch = SimpleNamespace(
        address=0x1004,
        size=4,
        id=0,
        groups=(CS_GRP_JUMP,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=0x1008),),
    )
    source = _node(
        0x1000,
        size=8,
        vex=_vex("Ijk_Boring", next_addr=0x1008),
        capstone_insns=(branch,),
        graph=graph,
    )

    assert _edge_type(_edge(source, destination, jumpkind="Ijk_Boring")) == "NEXT"


def test_atomic_vex_self_exit_remains_a_linear_fallthrough() -> None:
    """Ignore VEX's internal atomic exit when Capstone sees no branch."""

    source = _node(
        0x1000,
        size=7,
        vex=_vex("Ijk_Boring", next_addr=0x1007, exit_targets=(0x1000,)),
        capstone_insns=(SimpleNamespace(address=0x1000, groups=(), operands=()),),
    )

    assert _edge_type(_edge(source, _node(0x1007), jumpkind="Ijk_Boring")) == "NEXT"


def test_terminal_vex_return_with_boring_exit_is_conditional() -> None:
    """Style a conditional return's explicit non-returning exit as not taken."""

    source = _node(
        0x1000,
        size=4,
        vex=_vex("Ijk_Ret", exit_targets=(0x1004,)),
    )

    assert (
        _edge_type(_edge(source, _node(0x1004), jumpkind="Ijk_Boring"))
        == "CONDITIONAL_FALSE"
    )


def test_non_boring_vex_exit_is_not_a_conditional_branch() -> None:
    """Ignore VEX exception exits when classifying ordinary CFG edges."""

    source = _node(
        0x1000,
        size=4,
        vex=_vex(
            "Ijk_Boring",
            next_addr=0x1004,
            exit_targets=(0x2000,),
            exit_jumpkind="Ijk_SigFPE_IntDiv",
        ),
    )

    assert _edge_type(_edge(source, _node(0x1004), jumpkind="Ijk_Boring")) == "NEXT"
