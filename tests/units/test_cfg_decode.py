"""Fast tests for Capstone-backed CFG node inspection."""

from types import SimpleNamespace

import pytest

from bingraph.cfg import decode as decode_module
from bingraph.cfg.decode import (
    DecodedNode,
    decode_raw_capstone_insns,
    is_post_prefix_instruction_entry,
    target_is_known_nonreturning,
)


def _insn(addr: int, size: int) -> SimpleNamespace:
    """Create the minimal instruction shape required by ``DecodedNode``."""

    return SimpleNamespace(address=addr, size=size)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("abort", True),
        ("_Unwind_Resume", True),
        ("__assert_fail", True),
        ("__libc_assert_fail", True),
        ("__malloc_assert", True),
        ("__stack_chk_fail", True),
        ("malloc", False),
        ("_Unwind_Resume_or_Rethrow", False),
    ],
)
def test_nonreturning_runtime_plt_requires_an_exact_known_entry(
    monkeypatch, name: str, expected: bool
) -> None:
    """Use PLT metadata, not stub-address symbols or name-prefix guesses."""

    obj = SimpleNamespace(reverse_plt={0x1000: name})
    project = SimpleNamespace(
        loader=SimpleNamespace(find_object_containing=lambda _addr: obj)
    )
    monkeypatch.setattr(
        decode_module, "target_is_hooked_nonreturning", lambda *_: False
    )
    monkeypatch.setattr(
        decode_module, "_symbol_is_declared_nonreturning", lambda *_: False
    )

    assert target_is_known_nonreturning(project, 0x1000) is expected
    assert not target_is_known_nonreturning(project, 0x1001)


def test_decoded_node_reports_exact_instruction_coverage() -> None:
    """Accept a contiguous instruction stream covering the complete node span."""

    decoded = DecodedNode((_insn(0x1000, 2), _insn(0x1002, 3)))
    node = SimpleNamespace(addr=0x1000, size=5)

    assert decoded.has_exact_coverage(node)
    assert not decoded.contains_mid_instruction_addr(0x1002)
    assert decoded.contains_mid_instruction_addr(0x1003)


def test_decoded_node_rejects_a_gap_or_trailing_bytes() -> None:
    """Reject instruction streams that do not match the declared node range."""

    node = SimpleNamespace(addr=0x1000, size=5)

    assert not DecodedNode((_insn(0x1000, 2), _insn(0x1003, 2))).has_exact_coverage(
        node
    )
    assert not DecodedNode((_insn(0x1000, 2),)).has_exact_coverage(node)


def test_post_prefix_entry_is_a_valid_alternate_instruction_stream() -> None:
    """Accept only the byte directly after all leading instruction prefixes."""

    instruction = SimpleNamespace(address=0x1000, size=3, prefix=(0xF0, 0, 0, 0))

    assert is_post_prefix_instruction_entry(instruction, 0x1001)
    assert not is_post_prefix_instruction_entry(instruction, 0x1000)
    assert not is_post_prefix_instruction_entry(instruction, 0x1002)


def test_decoded_node_handles_missing_capstone_inspection() -> None:
    """Treat a missing Capstone inspection as non-empty-coverage failure."""

    decoded = DecodedNode(None, KeyError("block"))

    assert decoded.is_empty
    assert decoded.last is None
    assert not decoded.has_exact_coverage(SimpleNamespace(addr=0x1000, size=1))


def test_decoded_node_falls_back_to_raw_capstone_when_vex_stops() -> None:
    """Use the architecture decoder when VEX-backed Capstone has no instruction."""

    instruction = _insn(0x1000, 2)
    project = SimpleNamespace(
        loader=SimpleNamespace(
            memory=SimpleNamespace(load=lambda _addr, _size: b"\x90\x90")
        ),
        arch=SimpleNamespace(
            capstone=SimpleNamespace(
                disasm=lambda _data, _addr, *, count: [instruction]
            )
        ),
    )
    node = SimpleNamespace(
        addr=0x1000,
        size=2,
        block=SimpleNamespace(
            capstone=SimpleNamespace(insns=()),
            _project=project,
        ),
    )

    assert DecodedNode.from_node(node).insns == (instruction,)


def test_raw_decode_uses_thumb_mode_and_untagged_memory_address() -> None:
    """Respect ARM's tagged Thumb addresses when raw decoding is required."""

    instruction = _insn(0x1001, 2)
    memory_reads: list[tuple[int, int]] = []
    thumb_calls: list[tuple[bytes, int, int]] = []
    project = SimpleNamespace(
        loader=SimpleNamespace(
            memory=SimpleNamespace(
                load=lambda addr, size: memory_reads.append((addr, size)) or b"\x00\x00"
            )
        ),
        arch=SimpleNamespace(
            is_thumb=lambda _addr: True,
            capstone=SimpleNamespace(disasm=lambda *_args, **_kwargs: ()),
            capstone_thumb=SimpleNamespace(
                disasm=lambda data, addr, *, count: (
                    thumb_calls.append((data, addr, count)) or [instruction]
                )
            ),
        ),
    )

    assert decode_raw_capstone_insns(project, 0x1001, 2) == (instruction,)
    assert memory_reads == [(0x1000, 2)]
    assert thumb_calls == [(b"\x00\x00", 0x1001, 0)]


def test_decoded_node_caches_one_live_node_inspection() -> None:
    """Avoid recreating angr's Capstone view for repeated node checks."""

    instruction = _insn(0x1000, 1)

    class Node:
        """Provide a hashable node with a counted block lookup."""

        addr = 0x1000
        size = 1

        def __init__(self) -> None:
            self.block_reads = 0

        @property
        def block(self):
            self.block_reads += 1
            return SimpleNamespace(
                capstone=SimpleNamespace(insns=(SimpleNamespace(insn=instruction),))
            )

    node = Node()

    assert DecodedNode.from_node(node).insns == (instruction,)
    assert DecodedNode.from_node(node).insns == (instruction,)
    assert node.block_reads == 1


def test_decoded_node_does_not_reuse_an_equal_replacement_node() -> None:
    """Keep a same-address recovered replacement separate from its stale node."""

    class Node:
        """Model angr's address-based CFG-node equality with distinct objects."""

        addr = 0x1000
        size = 1

        def __init__(self, instruction) -> None:
            self.instruction = instruction

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Node) and self.addr == other.addr

        def __hash__(self) -> int:
            return hash(self.addr)

        @property
        def block(self):
            return SimpleNamespace(
                capstone=SimpleNamespace(
                    insns=(SimpleNamespace(insn=self.instruction),)
                )
            )

    stale = Node(_insn(0x1000, 1))
    replacement = Node(_insn(0x1000, 2))

    assert DecodedNode.from_node(stale).insns == (_insn(0x1000, 1),)
    assert DecodedNode.from_node(replacement).insns == (_insn(0x1000, 2),)
