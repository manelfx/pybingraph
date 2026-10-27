"""Select a frozen DDG corpus from reviewed CFG block boundaries.

Run explicitly, never during pytest collection. The bounded survey balances
architecture directories, binaries and functions before invoking DDG analysis;
it is not an exhaustive audit of every block. Selection uses stable hashes,
instruction counts and dependency features, never machine-dependent timings.
Existing checkpoint rows are protected regression cases. No DOT is generated
or promoted here; --report records unsupported surveyed blocks for review.
"""

import argparse
from collections import Counter, defaultdict, deque
import csv
from dataclasses import dataclass, replace
from hashlib import sha256
from html import unescape
from io import StringIO
import json
from pathlib import Path
import re

from angr import Project
from capstone import CsError

from bingraph.ddg.builder import get_ddg
from bingraph.helpers.disassembly import decode_raw_capstone_insns

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("filepath", "address", "size", "checkpoint", "purpose")


@dataclass(frozen=True)
class Block:
    """One candidate boundary, then its validated size and coverage features."""

    filepath: str
    address: int
    size: int = 0
    addresses: tuple[int, ...] = ()
    function: str = ""
    architecture: str = ""
    features: frozenset[str] = frozenset()
    purpose: str = ""
    shape: str = ""

    @property
    def key(self):
        return self.filepath, self.address, self.size

    @property
    def rank(self):
        # Unlike hash(), this order is stable across Python processes/platforms.
        return sha256(repr((self.key, self.addresses)).encode()).hexdigest()


def read_seeds(path):
    """Protect existing checkpoints; an old manifest without flags is all seeds."""
    with path.open(newline="") as stream:
        return [
            Block(
                row["filepath"],
                int(row["address"], 0),
                int(row["size"]),
                purpose=row["purpose"],
                function=f"regression:{row['address']}",
            )
            for row in csv.DictReader(stream)
            if row.get("checkpoint", "true") == "true"
        ]


def inventory(goldens, functions):
    """Read only assembly rows of our line-oriented CFG DOT output, not SIMPs.

    Address sequences preserve Thumb mode bits and exact existing splits. The
    binary decoder later validates them and determines the last instruction's
    length. This deliberately is not a general DOT parser or a new CFG builder.
    """
    with functions.open(newline="") as stream:
        paths = {row["filepath"] for row in csv.DictReader(stream)}
    lookup = {}
    for path in sorted(paths):
        flat = re.sub(r"[^A-Za-z0-9,._-]+", "_", path.replace("/", ",")).strip(".,")[
            :160
        ]
        if flat in lookup and lookup[flat] != path:
            raise ValueError(f"Ambiguous CFG binary prefix: {flat}")
        lookup[flat] = path
    candidates, counts = {}, Counter()
    for artifact in sorted(goldens.glob("*.dot")):
        parts = re.split(r",(0x[0-9a-f]+),", artifact.name, maxsplit=1)
        if len(parts) != 3 or parts[0] not in lookup:
            raise ValueError(f"Cannot identify CFG binary: {artifact.name}")
        filepath = lookup[parts[0]]
        for line in artifact.read_text().splitlines():
            if not re.match(r'^"0x[0-9a-f]+" \[label=', line):
                continue
            addresses = tuple(
                int(a, 16) for a in re.findall(r">0x([0-9a-f]+):", unescape(line))
            )
            if not addresses:
                continue
            key = filepath, addresses
            if key in candidates:
                continue
            # Keep the total inventory independent of the eligibility filter.
            candidates[key] = Block(
                filepath, addresses[0], addresses=addresses, function=parts[1]
            )
            counts[filepath.split("/")[0]] += 1
    eligible = [
        b
        for b in candidates.values()
        if 4 <= len(b.addresses) <= 64 and b.addresses[-1] - b.address <= 4096
    ]
    return eligible, dict(sorted(counts.items()))


def balanced(blocks, limit, *, per_binary, per_function, initial=()):
    """Round-robin architecture queues with hard binary/function contribution caps."""
    selected = list(initial)
    binaries = Counter(b.filepath for b in selected)
    functions = Counter((b.filepath, b.function) for b in selected)
    keys = {b.key for b in selected}
    shapes = {b.shape for b in selected if b.shape}
    groups = defaultdict(deque)
    for block in blocks:
        groups[block.architecture or block.filepath.split("/")[0]].append(block)
    while groups and len(selected) < limit:
        for arch in sorted(tuple(groups)):
            queue = groups[arch]
            while queue:
                block = queue.popleft()
                function = block.filepath, block.function
                if (
                    block.key in keys
                    or (block.shape and block.shape in shapes)
                    or binaries[block.filepath] >= per_binary
                    or functions[function] >= per_function
                ):
                    continue
                selected.append(block)
                keys.add(block.key)
                if block.shape:
                    shapes.add(block.shape)
                binaries[block.filepath] += 1
                functions[function] += 1
                break
            if not queue:
                del groups[arch]
            if len(selected) == limit:
                break
    return selected


def inspect_block(project, block):
    """Validate boundaries with the production analyzer, then summarize diversity.

    Features guide test selection only. They never feed alias analysis or CFG
    discovery. Architecture-specific Capstone groups are useful selection tags,
    not a separate instruction-semantics implementation.
    """
    size = block.size
    if not size:
        last = decode_raw_capstone_insns(project, block.addresses[-1], 16, count=1)
        if not last:
            raise ValueError("Cannot decode the last instruction")
        size = block.addresses[-1] + last[0].size - block.address
    ddg = get_ddg(project, block.address, size)
    insns = ddg.instructions
    if block.addresses and tuple(i.address for i in insns) != block.addresses:
        raise ValueError("Binary decode differs from CFG instruction boundaries")
    count = len(insns)
    features = {
        "size:" + ("small" if count <= 8 else "medium" if count <= 24 else "large")
    }
    views = set()
    for insn in insns:
        features.update(f"group:{insn.group_name(g)}" for g in insn.groups)
        try:
            reads, writes = insn.regs_access()
            views.update(
                project.arch.registers[name]
                for r in reads + writes
                if (name := insn.reg_name(r)) in project.arch.registers
            )
        except CsError:
            # Some Capstone backends omit access metadata; VEX still validates
            # the block and provides its actual dependency graph.
            pass
    if any(offset == project.arch.sp_offset for offset, _ in views):
        features.add("stack-register")
    if any(
        a != b and a[0] < b[0] + b[1] and b[0] < a[0] + a[1]
        for a in views
        for b in views
    ):
        features.add("register-aliases")
    for edge in ddg.dependencies:
        for resource in edge.resources:
            kind = (
                "memory"
                if resource.startswith("mem")
                else "flags"
                if resource.startswith("cc_")
                else "register"
            )
            features.add(f"{kind}:{edge.kind}")
    participants = {
        index for edge in ddg.dependencies for index in (edge.source, edge.target)
    }
    if len(participants) < count:
        features.add("independent-instructions")
    if not ddg.dependencies:
        features.add("no-edges")
    # Preserve mode/endian distinctions that a path prefix alone cannot express.
    thumb = bool(getattr(project.arch, "is_thumb", lambda _: False)(block.address))
    arch = f"{project.arch.name}/{project.arch.memory_endness}" + (
        "/Thumb" if thumb else ""
    )
    # Ignore relocated immediates but retain instruction/operand structure and
    # dependency topology, so repeated library implementations cannot dominate.
    shape = sha256(
        repr(
            (
                arch,
                [
                    (i.mnemonic, re.sub(r"0x[0-9a-f]+|\b\d+\b", "#", i.op_str))
                    for i in insns
                ],
                [(d.source, d.target, d.kind) for d in ddg.dependencies],
            )
        ).encode()
    ).hexdigest()
    return replace(
        block,
        size=size,
        architecture=arch,
        features=frozenset(features),
        addresses=tuple(i.address for i in insns),
        shape=shape,
    )


def checkpoints(blocks, seeds, limit):
    """Greedily cover architecture/feature pairs, preferring small representatives."""
    by_key = {b.key: b for b in blocks}
    selected = [by_key[b.key] for b in seeds]
    if len(selected) > limit:
        raise ValueError("Checkpoint target is smaller than the protected seed set")
    covered = {(b.architecture, f) for b in selected for f in b.features}
    keys = {b.key for b in selected}
    binaries = Counter(b.filepath for b in selected)
    functions = Counter((b.filepath, b.function) for b in selected)
    remaining = sorted(blocks, key=lambda b: (len(b.addresses), b.rank))
    while len(selected) < limit:
        choices = [
            b
            for b in remaining
            if b.key not in keys
            and binaries[b.filepath] < 12
            and functions[b.filepath, b.function] < 2
        ]
        if not choices:
            raise ValueError("Not enough distinct blocks for the checkpoint target")
        best = max(
            choices,
            key=lambda b: (
                sum((b.architecture, f) not in covered for f in b.features),
                -len(b.addresses),
            ),
        )
        selected.append(best)
        keys.add(best.key)
        binaries[best.filepath] += 1
        functions[best.filepath, best.function] += 1
        covered.update((best.architecture, f) for f in best.features)
    return selected


def main():
    """Survey, report and optionally replace the manifest; never touch goldens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "angr-binaries/tests")
    parser.add_argument(
        "--goldens", type=Path, default=ROOT / "tests/goldens/cfg_mode_custom"
    )
    parser.add_argument(
        "--functions", type=Path, default=ROOT / "tests/playground_functions.csv"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "tests/playground_blocks.csv"
    )
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--checkpoints", type=int, default=100)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--write", action="store_true", help="explicitly replace the manifest"
    )
    args = parser.parse_args()
    if not 0 < args.checkpoints <= args.limit:
        parser.error("Require 0 < checkpoints <= limit")
    seeds = read_seeds(args.manifest)
    candidates, inventory_counts = inventory(args.goldens, args.functions)
    owners = {(b.filepath, b.address): b.function for b in reversed(candidates)}
    seeds = [
        replace(b, function=owners.get((b.filepath, b.address), b.function))
        for b in seeds
    ]
    survey = balanced(
        sorted(candidates, key=lambda b: b.rank),
        args.limit * 3,
        per_binary=600,
        per_function=12,
    )
    print(
        f"Inventory: {sum(inventory_counts.values())} blocks, {len(candidates)} eligible; surveying {len(survey)} plus {len(seeds)} protected checkpoints",
        flush=True,
    )
    supported, rejected = {}, []
    grouped = defaultdict(list)
    for block in seeds + survey:
        grouped[block.filepath].append(block)
    for filepath, blocks in sorted(grouped.items()):
        try:
            project = Project(str(args.root / filepath), auto_load_libs=False)
        except Exception as exc:
            rejected.append({"filepath": filepath, "reason": str(exc)})
            continue
        for block in blocks:
            try:
                validated = inspect_block(project, block)
                # First occurrence retains protected purposes and stable ownership.
                supported.setdefault(validated.key, validated)
            except Exception as exc:
                rejected.append(
                    {
                        "filepath": filepath,
                        "address": hex(block.address),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        get_ddg.cache_clear()
        print(
            f"Surveyed {filepath}: {len(supported)} supported, {len(rejected)} rejected so far",
            flush=True,
        )
    missing = [b.key for b in seeds if b.key not in supported]
    args.report.write_text(
        json.dumps(
            {
                "inventory": inventory_counts,
                "eligible": len(candidates),
                "surveyed": len(survey),
                "supported": len(supported),
                "rejected": rejected,
                "missing_seeds": missing,
            },
            indent=2,
        )
        + "\n"
    )
    if missing:
        raise ValueError(f"Protected checkpoints failed validation: {missing}")
    selected_checkpoints = checkpoints(
        list(supported.values()), seeds, args.checkpoints
    )
    frequency = Counter(
        (b.architecture, f) for b in supported.values() for f in b.features
    )
    ranked = sorted(
        supported.values(),
        key=lambda b: (
            -sum(1 / frequency[b.architecture, f] for f in sorted(b.features)),
            b.rank,
        ),
    )
    selected = balanced(
        ranked, args.limit, per_binary=300, per_function=6, initial=selected_checkpoints
    )
    if len(selected) != args.limit:
        raise ValueError(
            f"Only {len(selected)} blocks fit the contribution caps; inspect the report before changing limits"
        )
    checkpoint_keys = {b.key for b in selected_checkpoints}
    report = json.loads(args.report.read_text())
    report["selected"] = dict(sorted(Counter(b.architecture for b in selected).items()))
    report["checkpoints"] = dict(
        sorted(Counter(b.architecture for b in selected_checkpoints).items())
    )
    report["max_per_binary"] = max(Counter(b.filepath for b in selected).values())
    report["max_per_function"] = max(
        Counter((b.filepath, b.function) for b in selected).values()
    )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for b in sorted(selected, key=lambda b: b.key):
        writer.writerow(
            dict(
                filepath=b.filepath,
                address=hex(b.address),
                size=b.size,
                checkpoint=str(b.key in checkpoint_keys).lower(),
                purpose=b.purpose or "; ".join(sorted(b.features)),
            )
        )
    content = stream.getvalue()
    report["manifest_sha256"] = sha256(content.encode()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.write:
        args.manifest.write_text(content)
    print(
        json.dumps(
            {k: report[k] for k in ("selected", "checkpoints", "max_per_binary")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
