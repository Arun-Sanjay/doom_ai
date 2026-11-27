"""A training manifest that is all of one map set plus as many maps of a second set, tier by tier.

    python -m training.rl.mix_manifests OUT_MANIFEST FIRST SECOND [--seed 4321]

Training-time scaffolding, never imported by the agent.

OUT_MANIFEST holds every row of FIRST and, for every tier that FIRST has, as many rows of SECOND
of the same tier as FIRST has in that tier, chosen at random without replacement (one generator
seeded with `--seed`, tiers in ascending order, the chosen rows kept in SECOND's order). So both
sets fill half of every tier, and a tier that FIRST does not have is left out. The rows are copied
as they are: their `path` entries still name the WAD files of the two original sets, which
`training.rl.deps.load_manifest` finds from the repository root, so no map is copied.

The clone run of `training/configs/clone_mix_full.json` trained on such a mix:

    python -m training.rl.mix_manifests data/maps/train_mix_ref50/manifest.jsonl \\
        data/maps/train_ref_v1/manifest.jsonl data/maps/train_v3/manifest.jsonl --seed 4321

which gives 3,000 organizer-style maps and 3,000 maps of the main run's set: 450, 750, 750 and
1,050 of each in tiers 1 to 4, and none of the main set's tier 0 maps.

Refused: a tier of FIRST for which SECOND has fewer rows, a row without a tier, a map name that
would appear twice in the output.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(manifest, lines: dict | None = None) -> list[dict]:
    """The rows of a manifest. `lines`, when given, receives id(row) -> the row's own text, so it can be written back unchanged."""
    rows = []
    for number, line in enumerate(Path(manifest).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "tier" not in row:
            raise ValueError(f"{manifest}: line {number} has no tier")
        rows.append(row)
        if lines is not None:
            lines[id(row)] = line
    return rows


def mix(first: list[dict], second: list[dict], seed: int = 4321) -> list[dict]:
    """Every row of `first`, then per tier of `first` as many randomly chosen rows of `second` of that tier."""
    rng = random.Random(seed)
    wanted = Counter(int(row["tier"]) for row in first)
    out = list(first)
    for tier in sorted(wanted):
        pool = [k for k, row in enumerate(second) if int(row["tier"]) == tier]
        if len(pool) < wanted[tier]:
            raise ValueError(f"tier {tier}: the first set has {wanted[tier]} maps, the second only {len(pool)}")
        out += [second[k] for k in sorted(rng.sample(pool, wanted[tier]))]
    names = Counter(str(row.get("name") or row.get("path")) for row in out)
    twice = sorted(name for name, count in names.items() if count > 1)
    if twice:
        raise ValueError(f"{len(twice)} map name(s) would appear twice, for example {twice[:3]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="All of FIRST plus as many maps of SECOND, tier by tier")
    ap.add_argument("out", help="the manifest to write")
    ap.add_argument("first", help="manifest.jsonl taken whole")
    ap.add_argument("second", help="manifest.jsonl sampled tier by tier")
    ap.add_argument("--seed", type=int, default=4321)
    args = ap.parse_args(argv)
    lines: dict = {}
    try:
        first, second = read_rows(args.first, lines), read_rows(args.second, lines)
        rows = mix(first, second, args.seed)
    except ValueError as error:
        raise SystemExit(f"refused: {error}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines[id(row)] + "\n" for row in rows), encoding="utf-8")
    per_tier = Counter(int(row["tier"]) for row in first)
    print(f"wrote {out}: {len(rows)} maps, {len(first)} of {args.first} and {len(rows) - len(first)} of {args.second} "
          f"(seed {args.seed}); per tier from each set: " + ", ".join(f"tier {t}: {per_tier[t]}" for t in sorted(per_tier)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
