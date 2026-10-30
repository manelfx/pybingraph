"""Small analysis records; presentation creates its own mutable node wrappers."""

from dataclasses import dataclass, field
from typing import Any, Literal

from angr import claripy

from bingraph.helpers.settings import MemoryPolicy as MemoryPolicy

DependencyKind = Literal["RAW", "WAR", "WAW"]


@dataclass(frozen=True)
class Resource:
    """A register/temporary range, opaque reservation state or memory range.

    Register offsets come from archinfo, so aliases share the same storage.
    ``address=None`` denotes opaque memory (e.g. a fence), never no memory.
    Labels describe the original access even when its range is split internally.
    ``display_label`` is an optional operand-like caption, never alias evidence.
    ``maximum_footprint`` marks a possible extent, not an executed byte count.
    ``conditional`` means a memory access may not execute; such a write cannot
    prove that an earlier producer has been overwritten.
    The reservation resource orders LL/SC attempts and explicit cancellation,
    independently of memory alias policy; it does not predict success.
    """

    kind: Literal["register", "memory", "temporary", "reservation"]
    offset: int
    size: int
    label: str
    address: claripy.ast.BV | None = field(default=None, compare=False)
    display_label: str | None = field(default=None, compare=False)
    maximum_footprint: bool = False
    conditional: bool = False


@dataclass
class Effects:
    """Explicit/implicit effects of one instruction, including address reads."""

    insn: Any
    reads: list[Resource] = field(default_factory=list)
    writes: list[Resource] = field(default_factory=list)


@dataclass(frozen=True)
class Dependency:
    """Full resource descriptions and separate, position-aligned display captions.

    Presentation metadata does not participate in dependency identity.
    """

    source: int
    target: int
    kind: DependencyKind
    resources: tuple[str, ...]
    display_resources: tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class DDG:
    """Cached read-only graph data. Instructions are retained even if isolated."""

    instructions: tuple[Any, ...]
    dependencies: tuple[Dependency, ...]
