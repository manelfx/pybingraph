"""Construct instruction-order dependencies from normalized VEX effects.

No architectural instruction matcher or whole-function inference lives here.
Unknown memory relations remain may-alias facts. The edge policy either keeps
all relations, excludes proven separation, or includes only proven overlap.
The last policy is intentionally incomplete, not proof of safe reordering.
All edges point forward in the requested block.
"""

from collections import defaultdict
from functools import lru_cache

from angr import Project

from bingraph.helpers.disassembly import decode_raw_capstone_insns
from .effects import instruction_effects
from .memory import covers, distance, overlap
from .models import DDG, Dependency, MemoryPolicy, Resource

MAX_BLOCK_BYTES = 4096
MAX_INSTRUCTIONS = 256


@lru_cache(maxsize=128)
def get_ddg(
    project: Project, address: int, size: int, memory: MemoryPolicy = "must"
) -> DDG:
    """Analyze exactly one bounded region; cached results have no render state.

    Each register byte keeps its latest writer. Memory producers are retained
    unless a later unconditional must-alias store fully covers the load. WAR/WAW
    relations preserve ordering even when they are transitively redundant.
    Memory labels explicitly say may-alias when overlap is not established.
    """
    if memory not in {"always", "may", "must"}:
        raise ValueError("DDG memory policy must be 'always', 'may', or 'must'")
    if address < 0 or not 0 < size <= MAX_BLOCK_BYTES:
        raise ValueError(f"DDG size must be between 1 and {MAX_BLOCK_BYTES} bytes")
    instructions = decode_raw_capstone_insns(project, address, size)
    if not instructions or len(instructions) > MAX_INSTRUCTIONS:
        raise ValueError(f"DDG requires 1 to {MAX_INSTRUCTIONS} decoded instructions")
    cursor = address
    for insn in instructions:
        if insn.address != cursor:
            raise ValueError("DDG region contains undecodable bytes")
        cursor += insn.size
    if cursor != address + size:
        raise ValueError(
            "DDG size ends inside an instruction or contains undecodable bytes"
        )
    effects = instruction_effects(project, instructions, size)
    edges = defaultdict(dict)
    writers: dict[tuple[str, int], tuple[int, str]] = {}
    readers = defaultdict(set)
    stores: list[tuple[int, Resource]] = []
    loads: list[tuple[int, Resource]] = []

    def add(source, target, kind, label, display_label=None):
        if source < target:
            edges[(source, target, kind)][label] = display_label or label

    def memory_label(left, right):
        return (
            right.label
            + (" (conditional)" if left.conditional or right.conditional else "")
            + (
                " (maximum footprint)"
                if left.maximum_footprint or right.maximum_footprint
                else ""
            )
            + (
                " (may alias)"
                if distance(left, right) is None or not overlap(left, right, "may")
                else ""
            )
        )

    for index, effect in enumerate(effects):
        for resource in effect.reads:
            if resource.kind == "memory":
                for position, (owner, writer) in enumerate(stores):
                    if overlap(writer, resource, memory) and not any(
                        covers(later, resource) for _, later in stores[position + 1 :]
                    ):
                        add(
                            owner,
                            index,
                            "RAW",
                            memory_label(writer, resource),
                            resource.display_label,
                        )
            else:
                for byte in range(resource.offset, resource.offset + resource.size):
                    if (previous := writers.get((resource.kind, byte))) is not None:
                        add(previous[0], index, "RAW", resource.label)
        for resource in effect.writes:
            if resource.kind == "memory":
                for owner, prior in loads:
                    if overlap(prior, resource, memory):
                        add(
                            owner,
                            index,
                            "WAR",
                            memory_label(prior, resource),
                            resource.display_label,
                        )
                for owner, prior in stores:
                    if overlap(prior, resource, memory):
                        add(
                            owner,
                            index,
                            "WAW",
                            memory_label(prior, resource),
                            resource.display_label,
                        )
            else:
                for byte in range(resource.offset, resource.offset + resource.size):
                    key = resource.kind, byte
                    for owner in readers[key]:
                        add(owner, index, "WAR", resource.label)
                    if (prior := writers.get(key)) is not None:
                        add(prior[0], index, "WAW", resource.label)
        # Update only after collecting the instruction's reads and writes, so
        # read/modify/write operands never create self-edges or backward edges.
        for resource in effect.reads:
            if resource.kind == "memory":
                loads.append((index, resource))
            else:
                for byte in range(resource.offset, resource.offset + resource.size):
                    readers[resource.kind, byte].add(index)
        for resource in effect.writes:
            if resource.kind == "memory":
                stores.append((index, resource))
            else:
                for byte in range(resource.offset, resource.offset + resource.size):
                    key = resource.kind, byte
                    writers[key] = index, resource.label
                    readers[key].clear()
    dependencies = []
    for (source, target, kind), labels in sorted(edges.items()):
        resources = tuple(sorted(labels))
        dependencies.append(
            Dependency(
                source,
                target,
                kind,
                resources,
                tuple(labels[resource] for resource in resources),
            )
        )
    return DDG(instructions, tuple(dependencies))
