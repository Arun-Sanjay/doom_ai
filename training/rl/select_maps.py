"""Write a sub-manifest of the maps that match simple filters on the generator's counts.

    python -m training.rl.select_maps --maps data/maps/train/manifest.jsonl --out data/maps/smoke_arena.jsonl \
        --tier 1 --exits 0 --max-doors 0 --max-rooms 1

Training-time scaffolding, never imported by the agent. Used to pick the single-room arenas of the
learning smoke test; WAD paths are rewritten as absolute paths so the output can live anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training import paths
from training.rl.deps import start_on_grid


def select(source: str, tier=None, exits=None, max_doors=None, max_rooms=None, max_monsters=None, max_open_cells=None,
           tiers=None, on_grid: bool = False) -> list:
    """Manifest rows (dicts) passing every given filter; a filter left at None is not applied.

    ``tiers`` keeps the maps of several tiers; ``on_grid`` keeps the maps whose start lies on the
    64-unit grid facing along an axis (the maps the navigation teacher labels).
    """
    src = Path(source)
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        c = row.get("counts", {})
        if tier is not None and int(row.get("tier", -1)) != tier:
            continue
        if tiers is not None and int(row.get("tier", -1)) not in tiers:
            continue
        if on_grid and not start_on_grid(row.get("options")):
            continue
        if exits is not None and int(c.get("exits", -1)) != exits:
            continue
        if any(limit is not None and int(c.get(key, 10**9)) > limit for key, limit in
               (("doors", max_doors), ("rooms", max_rooms), ("monsters", max_monsters), ("open_cells", max_open_cells))):
            continue
        raw = Path(row["path"])
        if not raw.is_absolute():
            raw = next((b / raw for b in (src.parent, paths.ROOT) if (b / raw).exists()), src.parent / raw)
        row["path"] = str(raw.resolve())
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter a map manifest")
    ap.add_argument("--maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", type=int)
    ap.add_argument("--exits", type=int, help="exact number of exits")
    ap.add_argument("--max-doors", type=int)
    ap.add_argument("--max-rooms", type=int)
    ap.add_argument("--max-monsters", type=int)
    ap.add_argument("--max-open-cells", type=int)
    ap.add_argument("--tiers", help="comma-separated tiers to keep, e.g. 0,2")
    ap.add_argument("--on-grid", action="store_true", help="only maps the navigation teacher labels (start on the 64-unit grid)")
    args = ap.parse_args()
    rows = select(args.maps, args.tier, args.exits, args.max_doors, args.max_rooms, args.max_monsters, args.max_open_cells,
                  None if not args.tiers else {int(t) for t in args.tiers.split(",")}, args.on_grid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} maps -> {out}")


if __name__ == "__main__":
    main()
