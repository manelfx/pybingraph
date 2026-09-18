"""Unit tests for static jump-table target recovery helpers."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import archinfo
import networkx as nx
import pyvex

from bingraph.cfg.jumps import (
    _amd64_sysv_register_layout,
    _abi_transfer_static_targets,
    arithmetic_pc_dispatch_targets,
    _jump_table_target_addr,
    plan_static_jump_table,
    _read_static_jump_table_targets,
    _vex_direct_jump_table,
    _vex_guarded_index_upper_bound,
    _vex_relative_jump_table,
    _x86_pc_thunk_base_addr,
    _x86_pc_thunk_guarded_entry_count,
    static_jump_target_rejection_reason,
)
from bingraph.cfg import jumps as jumps_module
from bingraph.cfg.models import BlockSpec, FunctionBounds, StaticJumpTable


@dataclass(frozen=True)
class _Node:
    """Provide the small hashable CFG-node surface used by jump helpers."""

    addr: int
    size: int
    vex: object = field(compare=False, hash=False)

    @property
    def block(self) -> SimpleNamespace:
        """Expose the node VEX through angr's block-shaped API."""

        return SimpleNamespace(vex=self.vex)


def _table(
    *, target_displacement: int = 0, index_register_offset: int = 12
) -> StaticJumpTable:
    """Create a 32-bit relative table description for focused helper tests."""

    return StaticJumpTable(
        base_register_offset=20,
        base_bits=32,
        table_displacement=0,
        index_register_offset=index_register_offset,
        index_bits=32,
        entry_size=4,
        endness="Iend_LE",
        signed_entries=False,
        target_displacement=target_displacement,
    )


def test_amd64_sysv_alias_write_invalidates_a_static_register_target() -> None:
    """A high-byte write must invalidate the enclosing 64-bit register value."""

    arch = archinfo.ArchAMD64()
    executable = SimpleNamespace(
        find_section_containing=lambda _addr: SimpleNamespace(is_executable=True)
    )
    vex = pyvex.lift(bytes.fromhex("b4 12 ff e0"), 0x400000, arch)
    project = SimpleNamespace(
        arch=arch,
        loader=SimpleNamespace(
            main_object=SimpleNamespace(os="UNIX - System V"),
            extern_object=None,
            find_object_containing=lambda _addr: executable,
        ),
        factory=SimpleNamespace(
            block=lambda *_args, **_kwargs: SimpleNamespace(vex=vex)
        ),
    )
    layout = _amd64_sysv_register_layout(project)
    assert layout is not None
    alias_writes, _ = layout
    rax_offset = arch.registers["rax"][0]
    assert alias_writes[arch.registers["ah"][0]] == rax_offset

    output, target = _abi_transfer_static_targets(
        project,
        BlockSpec(0x400000, 4, (0x400000, 0x400002), "Ijk_Boring"),
        alias_writes,
        {rax_offset: 0x401000},
    )

    assert rax_offset not in output
    assert target is None


def test_amd64_sysv_transfer_ignores_a_direct_vex_target() -> None:
    """Only a register-valued VEX transfer can be materialized by this pass."""

    arch = archinfo.ArchAMD64()
    vex = pyvex.lift(bytes.fromhex("e9 fb 0f 00 00"), 0x400000, arch)
    project = SimpleNamespace(
        arch=arch,
        loader=SimpleNamespace(main_object=SimpleNamespace(os="UNIX - System V")),
        factory=SimpleNamespace(
            block=lambda *_args, **_kwargs: SimpleNamespace(vex=vex)
        ),
    )
    layout = _amd64_sysv_register_layout(project)
    assert layout is not None
    alias_writes, _ = layout

    _, target = _abi_transfer_static_targets(
        project,
        BlockSpec(0x400000, 5, (0x400000,), "Ijk_Boring"),
        alias_writes,
        {},
    )

    assert target is None


def test_x86_pc_thunk_proves_the_dispatcher_base_register() -> None:
    """Accept a fallthrough from the matching GCC PC thunk call."""

    call_target = 0x4000
    predecessor = _Node(
        0x1000,
        5,
        SimpleNamespace(
            jumpkind="Ijk_Call",
            next=pyvex.expr.Const(pyvex.const.U32(call_target)),
            statements=(),
        ),
    )
    dispatcher = _Node(0x1005, 4, SimpleNamespace())
    graph = nx.DiGraph([(predecessor, dispatcher)])
    project = SimpleNamespace(
        arch=SimpleNamespace(name="X86", bits=32, register_names={20: "ebx"}),
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(name="__x86.get_pc_thunk.bx")
                if addr == call_target
                else None
            )
        ),
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    assert (
        _x86_pc_thunk_base_addr(project, graph, bounds, dispatcher, _table()) == 0x1005
    )


def test_x86_pc_thunk_rejects_a_call_to_the_wrong_register_thunk() -> None:
    """Reject a direct call whose thunk does not initialize the table register."""

    call_target = 0x4000
    predecessor = _Node(
        0x1000,
        5,
        SimpleNamespace(
            jumpkind="Ijk_Call",
            next=pyvex.expr.Const(pyvex.const.U32(call_target)),
            statements=(),
        ),
    )
    dispatcher = _Node(0x1005, 4, SimpleNamespace())
    graph = nx.DiGraph([(predecessor, dispatcher)])
    project = SimpleNamespace(
        arch=SimpleNamespace(name="X86", bits=32, register_names={20: "ebx"}),
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(name="__x86.get_pc_thunk.ax")
                if addr == call_target
                else None
            )
        ),
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    assert _x86_pc_thunk_base_addr(project, graph, bounds, dispatcher, _table()) is None


def test_x86_pc_thunk_prefers_an_immediate_dispatcher_predecessor() -> None:
    """Keep one local PC-thunk proof from conflicting with unrelated PIC setup."""

    call_target = 0x4000
    call_vex = SimpleNamespace(
        jumpkind="Ijk_Call",
        next=pyvex.expr.Const(pyvex.const.U32(call_target)),
        statements=(),
    )
    predecessor = _Node(0x1000, 5, call_vex)
    dispatcher = _Node(0x1005, 4, SimpleNamespace())
    other_thunk_call = _Node(0x1020, 5, call_vex)
    other_fallthrough = _Node(
        0x1025,
        4,
        pyvex.lift(bytes.fromhex("83c320c3"), 0x1025, archinfo.ArchX86()),
    )
    graph = nx.DiGraph(
        [(predecessor, dispatcher), (other_thunk_call, other_fallthrough)]
    )
    project = SimpleNamespace(
        arch=SimpleNamespace(name="X86", bits=32, register_names={20: "ebx"}),
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(name="__x86.get_pc_thunk.bx")
                if addr == call_target
                else None
            )
        ),
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    assert (
        _x86_pc_thunk_base_addr(project, graph, bounds, dispatcher, _table()) == 0x1005
    )


def test_x86_pc_thunk_recovers_the_guarded_table_length() -> None:
    """Carry an unsigned index bound through the thunk's fake-return edge."""

    call_target = 0x4000
    guard = _Node(
        0x1000,
        5,
        pyvex.lift(bytes.fromhex("83f8207200"), 0x1000, archinfo.ArchX86()),
    )
    thunk_call = _Node(
        0x1005,
        5,
        SimpleNamespace(
            jumpkind="Ijk_Call",
            next=pyvex.expr.Const(pyvex.const.U32(call_target)),
            statements=(),
        ),
    )
    dispatcher = _Node(0x100A, 4, SimpleNamespace())
    graph = nx.DiGraph([(guard, thunk_call), (thunk_call, dispatcher)])
    project = SimpleNamespace(
        arch=SimpleNamespace(name="X86", bits=32, register_names={20: "ebx"}),
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(name="__x86.get_pc_thunk.bx")
                if addr == call_target
                else None
            )
        ),
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    assert (
        _x86_pc_thunk_guarded_entry_count(
            project,
            graph,
            bounds,
            dispatcher,
            _table(index_register_offset=8),
        )
        == 32
    )


def test_relative_table_target_wraps_to_the_architecture_width() -> None:
    """Interpret a raw 32-bit relative entry using machine-width arithmetic."""

    assert (
        _jump_table_target_addr(
            0x80626C1, _table(target_displacement=0x4B9C7), 0xFFFB46A8
        )
        == 0x8062730
    )


def test_direct_jump_table_keeps_absolute_entries() -> None:
    """Recognize a VEX-lifted absolute table dispatch without rebasing entries."""

    # jmp dword ptr [ebx + eax * 4]
    vex = pyvex.lift(bytes.fromhex("ff2483"), 0x1000, archinfo.ArchX86())

    table = _vex_direct_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.base_register_offset == 20  # ebx
    assert table.index_register_offset == 8  # eax
    assert table.entry_size == 4
    assert not table.entries_are_relative
    assert _jump_table_target_addr(0x1000, table, 0x2000) == 0x2000


def test_direct_jump_table_recovers_a_finite_masked_index() -> None:
    """Recover the exact domain of a constant-mask table index."""

    # mov edx, ecx; shr edx, 12; and edx, 15; jmp [rdx * 8 + 0x4a3648]
    vex = pyvex.lift(
        bytes.fromhex("89cac1ea0c83e20fff24d548364a00"),
        0x43DAC8,
        archinfo.ArchAMD64(),
    )

    table = _vex_direct_jump_table(
        vex,
        allow_full_width_index=True,
        allow_masked_index_values=True,
        allow_static_base=True,
    )

    assert table is not None
    assert table.index_register_offset is None
    assert table.index_values == tuple(range(16))


def test_direct_jump_table_keeps_sparse_mask_domains_exact() -> None:
    """Do not expand a non-contiguous mask into its numerical range."""

    # and edx, 5; jmp [rdx * 8 + 0x4a3648]
    vex = pyvex.lift(
        bytes.fromhex("83e205ff24d548364a00"), 0x1000, archinfo.ArchAMD64()
    )

    table = _vex_direct_jump_table(
        vex,
        allow_full_width_index=True,
        allow_masked_index_values=True,
        allow_static_base=True,
    )

    assert table is not None
    assert table.index_values == (0, 1, 4, 5)


def test_shared_static_table_plan_is_graph_strategy_neutral(monkeypatch) -> None:
    """Keep table recognition reusable by fixup and independent extraction."""

    table = StaticJumpTable(
        base_register_offset=None,
        base_bits=32,
        table_displacement=0,
        index_register_offset=8,
        index_bits=32,
        entry_size=4,
        endness="Iend_LE",
        signed_entries=False,
        static_base_addr=0x2000,
    )
    dispatcher = _Node(0x1000, 4, SimpleNamespace())
    graph = nx.DiGraph()
    graph.add_node(dispatcher)
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))
    project = SimpleNamespace(arch=SimpleNamespace(name="ARMEL", bits=32))
    monkeypatch.setattr(
        jumps_module, "_vex_relative_jump_table", lambda *_args, **_kwargs: table
    )
    monkeypatch.setattr(
        jumps_module, "_vex_direct_jump_table", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        jumps_module, "_guarded_jump_table_entry_count", lambda *_args: 3
    )

    plan, reason = plan_static_jump_table(project, graph, bounds, dispatcher)

    assert reason is None
    assert plan is not None
    assert plan.base_addr == 0x2000
    assert plan.entry_count == 3


def test_relative_jump_table_accepts_a_guarded_full_width_index() -> None:
    """Recognize the common AMD64 signed-relative table dispatch form."""

    # movsxd r9, dword ptr [r11 + r9 * 4]; lea r9, [r11 + r9]; jmp r9
    vex = pyvex.lift(
        bytes.fromhex("4f630c8b4f8d0c0b41ffe1"), 0x1000, archinfo.ArchAMD64()
    )

    table = _vex_relative_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.base_bits == 64
    assert table.entry_size == 4
    assert table.signed_entries
    assert table.entries_are_relative


def test_relative_jump_table_normalizes_signed_entry_casts() -> None:
    """Recognize a signed entry after a VEX zero-extend/truncate detour."""

    # mov eax, eax; lea rdx, [rax * 4]; lea rax, [rip + table];
    # mov eax, [rdx + rax]; cdqe; lea rdx, [rip + base]; add rax, rdx; jmp rax
    vex = pyvex.lift(
        bytes.fromhex(
            "89c0488d148500000000488d05000000008b04024898488d15000000004801d03effe0"
        ),
        0x1000,
        archinfo.ArchAMD64(),
    )

    table = _vex_relative_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.entry_size == 4
    assert table.signed_entries


def test_relative_jump_table_accepts_a_vex_folded_static_base() -> None:
    """Recognize a RIP-relative table base folded to a VEX constant."""

    # lea r11, [rip + 0x7389d]; movsxd rdx, [r11 + rdx * 4];
    # lea rdx, [r11 + rdx]; jmp rdx
    vex = pyvex.lift(
        bytes.fromhex("4c8d1d9d38070049631493498d1413ffe2"),
        0x42F50C,
        archinfo.ArchAMD64(),
    )

    table = _vex_relative_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.base_register_offset is None
    assert table.static_base_addr == 0x4A2DB0
    assert table.index_bits == 64


def test_guarded_jump_table_bound_accepts_unsigned_strict_less_than() -> None:
    """Treat an unsigned ``index < limit`` guard as ``limit`` table entries."""

    # cmp rdx, 0x20; jb 0x1006
    vex = pyvex.lift(bytes.fromhex("4883fa207200"), 0x1000, archinfo.ArchAMD64())

    assert _vex_guarded_index_upper_bound(vex, 0x1006, (32, 64)) == 31


def test_guarded_jump_table_bound_tracks_a_narrowed_register_view() -> None:
    """Match a 32-bit guard against the 64-bit register used to index a table."""

    # cmp r8d, 0x11; jbe 0x1006
    vex = pyvex.lift(bytes.fromhex("4183f8117600"), 0x1000, archinfo.ArchAMD64())

    assert _vex_guarded_index_upper_bound(vex, 0x1006, (80, 64)) == 17


def test_guarded_jump_table_bound_tracks_a_16_bit_register_view() -> None:
    """Normalize VEX's shifted unsigned comparison for an x86 ``ax`` guard."""

    # cmp ax, 0x2b; jbe 0x1006
    vex = pyvex.lift(bytes.fromhex("6683f82b7600"), 0x1000, archinfo.ArchAMD64())

    assert _vex_guarded_index_upper_bound(vex, 0x1006, (16, 16)) == 43


def test_guarded_jump_table_bound_tracks_a_post_decrement_byte_selector() -> None:
    """Match VEX's masked flags against the byte index after decrementing."""

    # add al, -1; cmp al, 5; jbe 0x1008
    vex = pyvex.lift(bytes.fromhex("04ff3c057602"), 0x1000, archinfo.ArchAMD64())

    assert _vex_guarded_index_upper_bound(vex, 0x1008, (16, 8)) == 5


def test_guarded_table_bound_tracks_a_zero_extended_index_value() -> None:
    """Match a narrow guard after its value was written into a wider register."""

    # mov eax, [rip]; cmp eax, 4; ja 0x1015
    # The not-taken edge enters the dispatcher at 0x100f with rax holding the
    # zero-extended 32-bit value used by the comparison.
    vex = pyvex.lift(
        bytes.fromhex("8b050000000083f8040f8706000000"),
        0x1000,
        archinfo.ArchAMD64(),
    )

    assert _vex_guarded_index_upper_bound(vex, 0x100F, (16, 64)) == 4


def test_guarded_table_bound_tracks_a_right_shifted_index_value() -> None:
    """Accept a guard on an x86-64 selector narrowed by a logical shift."""

    # shr rdi, 32; cmp edi, 0x28; ja 0x2000
    # The not-taken path enters the dispatcher at 0x100e.
    vex = pyvex.lift(
        bytes.fromhex("48c1ef204883ff280f87f20f0000"),
        0x1000,
        archinfo.ArchAMD64(),
    )
    rdi_offset = archinfo.ArchAMD64().registers["rdi"][0]

    assert _vex_guarded_index_upper_bound(vex, 0x100E, (rdi_offset, 64)) == 40


def test_static_table_accepts_a_guarded_stack_selector() -> None:
    """Recover an x86 switch table indexed by one guarded stack argument."""

    # push ebp; mov ebp, esp; sub esp, 16; cmp dword ptr [ebp + 8], 12;
    # ja default. The selector therefore needs the prologue's ebp assignment
    # normalized before it can match the dispatcher's stack load.
    predecessor_vex = pyvex.lift(
        bytes.fromhex("5589e583ec10837d080c7755"), 0x1000, archinfo.ArchX86()
    )
    # mov eax, [ebp + 8]; shl eax, 2; add eax, table; mov eax, [eax]; jmp eax
    dispatcher_vex = pyvex.lift(
        bytes.fromhex("8b4508c1e00205708704088b00ffe0"),
        0x100C,
        archinfo.ArchX86(),
    )
    predecessor = _Node(0x1000, 12, predecessor_vex)
    dispatcher = _Node(0x100C, 15, dispatcher_vex)
    graph = nx.DiGraph([(predecessor, dispatcher)])
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))
    project = SimpleNamespace(arch=archinfo.ArchX86())

    plan, reason = plan_static_jump_table(
        project,
        graph,
        bounds,
        dispatcher,
        allow_static_bases=True,
        allow_guarded_expression_indices=True,
    )

    assert reason is None
    assert plan is not None
    assert plan.base_addr == 0x8048770
    assert plan.entry_indices == tuple(range(13))

    unguarded_graph = nx.DiGraph()
    unguarded_graph.add_node(dispatcher)
    unguarded_plan, unguarded_reason = plan_static_jump_table(
        project,
        unguarded_graph,
        bounds,
        dispatcher,
        allow_static_bases=True,
        allow_guarded_expression_indices=True,
    )

    assert unguarded_plan is None
    assert unguarded_reason == "no_table_shape"


def test_guarded_table_bound_handles_an_ite_index_expression() -> None:
    """Use VEX's type environment while tracing a conditional-move index."""

    # cmp rcx, 6; mov eax, 2; cmovb rax, rcx; cmp rax, 4; ja 0x1013
    # The conditional move produces an ITE expression whose type depends on
    # the block type environment rather than being derivable in isolation.
    vex = pyvex.lift(
        bytes.fromhex("4883f906b802000000480f42c14883f8047700"),
        0x1000,
        archinfo.ArchAMD64(),
    )

    assert _vex_guarded_index_upper_bound(vex, 0x1013, (16, 64)) == 4


def test_relative_jump_table_accepts_a_finite_ite_index_domain() -> None:
    """Recover a table domain selected by a VEX conditional move."""

    # value -= 2; index = value <u 5 ? value : 1; jump table[index]
    vex = pyvex.lift(
        bytes.fromhex(
            "498b064883c0fe4883f805b901000000480f42c8488d050000000048630c884801c1ffe1"
        ),
        0x1000,
        archinfo.ArchAMD64(),
    )

    table = _vex_relative_jump_table(
        vex,
        allow_full_width_index=True,
        allow_inline_index_values=True,
    )

    assert table is not None
    assert table.index_register_offset is None
    assert table.index_values == (0, 1, 2, 3, 4)

    dispatcher = _Node(0x1000, 4, vex)
    graph = nx.DiGraph()
    graph.add_node(dispatcher)
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))
    project = SimpleNamespace(arch=SimpleNamespace(name="AMD64", bits=64))

    plan, reason = plan_static_jump_table(
        project,
        graph,
        bounds,
        dispatcher,
        allow_inline_index_values=True,
    )

    assert reason is None
    assert plan is not None
    assert plan.entry_indices == (0, 1, 2, 3, 4)


def test_guarded_jump_table_bound_tracks_a_same_block_index_assignment() -> None:
    """Use the index value written before the guard rather than only a raw GET."""

    # mov ecx, [esp + 0x10]; cmp ecx, 0x20; jb 0x1009
    vex = pyvex.lift(bytes.fromhex("8b4c241083f9207200"), 0x1000, archinfo.ArchX86())

    assert _vex_guarded_index_upper_bound(vex, 0x1009, (12, 32)) == 31


def test_unreadable_static_jump_table_returns_none() -> None:
    """Distinguish an unreadable table from a readable table without targets."""

    project = SimpleNamespace(
        loader=SimpleNamespace(
            memory=SimpleNamespace(
                load=lambda *_args: (_ for _ in ()).throw(ValueError("unmapped"))
            )
        )
    )

    assert _read_static_jump_table_targets(project, _table(), 0x1000, (0, 1)) is None


def test_arithmetic_pc_dispatch_keeps_only_conditionally_scaled_targets() -> None:
    """Prune CFGFast's byte-stride over-approximation using local VEX proof."""

    scale = _Node(
        0x1000,
        4,
        pyvex.lift(bytes.fromhex("82208210"), 0x1000, archinfo.ArchARMEL()),
    )
    neutral = _Node(
        0x1004,
        4,
        pyvex.lift(bytes.fromhex("0000a0e3"), 0x1004, archinfo.ArchARMEL()),
    )
    dispatch = _Node(
        0x1008,
        4,
        pyvex.lift(bytes.fromhex("02f18f10"), 0x1008, archinfo.ArchARMEL()),
    )
    fallthrough = _Node(0x100C, 4, SimpleNamespace())
    candidates = tuple(
        _Node(addr, 4, SimpleNamespace()) for addr in range(0x1010, 0x1040, 4)
    )
    graph = nx.DiGraph(
        [
            (scale, neutral),
            (neutral, dispatch),
            (dispatch, fallthrough),
            *((dispatch, candidate) for candidate in candidates),
        ]
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    project = SimpleNamespace()
    assert arithmetic_pc_dispatch_targets(project, graph, bounds, dispatch) == tuple(
        range(0x1010, 0x1040, 12)
    )


@dataclass(frozen=True)
class _Region:
    """Model the executable mapping bit used by static-target validation."""

    is_executable: bool


def _target_project(
    section: _Region | None, *, synthetic: bool = False
) -> SimpleNamespace:
    """Create the minimal loader surface required by target validation."""

    obj = SimpleNamespace(
        find_section_containing=lambda _addr: section,
        find_segment_containing=lambda _addr: None,
    )
    loader = SimpleNamespace(find_object_containing=lambda _addr: obj)
    if synthetic:
        loader.extern_object = obj
    return SimpleNamespace(loader=loader)


def test_static_jump_target_accepts_executable_in_function_code() -> None:
    """Accept an in-function entry backed by an executable section."""

    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=True)), 0x1080
        )
        is None
    )


def test_static_jump_target_rejects_non_code_or_synthetic_values() -> None:
    """Keep data, unmapped, and synthetic table values out of CFG recovery."""

    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=False)), 0x1080
        )
        == "non_executable"
    )
    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=True)), 0x1200
        )
        is None
    )
    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=True), synthetic=True),
            0x1080,
        )
        == "synthetic"
    )
    unmapped = SimpleNamespace(
        loader=SimpleNamespace(find_object_containing=lambda _addr: None)
    )
    assert static_jump_target_rejection_reason(unmapped, 0x1080) == "unmapped"
