"""Byte-range alias policies and labels for versioned DDG resources.

Only concrete translations of identical roots can prove separation. Different
pointer registers never imply distinct memory objects; no SAT queries are used.
"""

from dataclasses import replace

from angr import claripy

from .models import MemoryPolicy, Resource


def address_label(address) -> str:
    """Show compact signed base offsets without changing their alias identity."""
    if address.concrete:
        return hex(address.concrete_value)
    base, offset = base_offset(address)
    if base.op != "BVS":
        return str(address)
    return offset_label(str(base.args[0]), offset, len(address))


def offset_label(base: str, offset: int, bits: int) -> str:
    """Format a modular pointer displacement as a compact signed offset."""
    modulus = 1 << bits
    offset %= modulus
    if offset >= modulus // 2:
        offset -= modulus
    suffix = f"{'+' if offset > 0 else '-'}{abs(offset):#x}" if offset else ""
    return base + suffix


def distance(left: Resource, right: Resource) -> int | None:
    """Compare memory access addresses, independently of their display labels."""
    return address_distance(left.address, right.address)


def address_distance(left, right) -> int | None:
    """Return a proven modular address difference without invoking a solver."""
    if left is None or right is None:
        return None
    delta = claripy.simplify(right - left)
    if delta.concrete:
        return delta.concrete_value

    left_base, left_offset = base_offset(left)
    right_base, right_offset = base_offset(right)
    # Rust Claripy's identical() matches expression shape, including differently
    # named symbolic leaves. It is NOT proof that two entry pointers are equal.
    if claripy.is_true(left_base == right_base):
        return (right_offset - left_offset) % (1 << len(left))
    return None


def overlap(left: Resource, right: Resource, policy: MemoryPolicy) -> bool:
    """Select overlapping accesses under the requested memory-edge policy.

    ``must`` omits unknown relations, but never converts that omission into a
    disjointness fact. Address identity and producer-killing proofs are separate.
    """
    if left.kind != right.kind:
        return False
    if left.kind != "memory":
        return (
            left.offset < right.offset + right.size
            and right.offset < left.offset + left.size
        )
    delta = distance(left, right)
    if policy == "always":
        return True
    if delta is None:
        return policy == "may"
    assert left.address is not None
    modulus = 1 << len(left.address)
    return delta < left.size or delta > modulus - right.size


def covers(writer: Resource, reader: Resource) -> bool:
    """Require must-alias full coverage before killing a possible producer."""
    if writer.kind != reader.kind:
        return False
    if writer.kind != "memory":
        delta = reader.offset - writer.offset
    else:
        delta = distance(writer, reader)
    return delta is not None and 0 <= delta and delta + reader.size <= writer.size


def coalesce_memory(resources: list[Resource]) -> list[Resource]:
    """Combine increasing, overlapping/adjacent accesses of one instruction.

    Use only after completing a proven instruction-internal loop. A contiguous
    write must cover its entire range for later producer-killing proofs; merely
    retaining individual byte stores would leave overwritten producers alive.
    Unknown address relations never merge, regardless of the display policy.
    """
    ranges, other = [], []
    for resource in resources:
        if resource.kind != "memory" or resource.address is None:
            other.append(resource)
            continue
        for index, prior in enumerate(ranges):
            delta = distance(prior, resource)
            if delta is not None and 0 <= delta <= prior.size:
                width = max(prior.size, delta + resource.size)
                ranges[index] = replace(
                    prior,
                    size=width,
                    label=f"mem[{address_label(prior.address)}]:{width}",
                )
                break
        else:
            ranges.append(resource)
    return list(dict.fromkeys(other)) + ranges


def base_offset(value):
    """Peel bounded constant translations from a versioned address expression."""
    offset = 0
    for _ in range(32):
        if value.op not in {"__add__", "__sub__"} or len(value.args) != 2:
            break
        base, constant = value.args
        if not constant.concrete:
            if value.op == "__add__" and base.concrete:
                base, constant = constant, base
            else:
                break
        offset += constant.concrete_value * (-1 if value.op == "__sub__" else 1)
        value = base
    return value, offset
