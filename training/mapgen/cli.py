"""Command line of the map generator. Training-time scaffolding, never imported by the agent.

Write a training set (WAD files plus `manifest.jsonl`); `--n` is the number of maps per tier::

    python -m training.mapgen.cli --out data/maps/train --n 300 --tiers 0,1,2,3,4 --seed 0 --randomize 0.5

Write the same maps without their monsters, to train and measure navigation apart from combat::

    python -m training.mapgen.cli --out data/maps/train_nav --n 300 --tiers 0,2,3,4 --seed 0 --randomize 0.5 --strip-monsters

`--strip-monsters` turns every monster cell into floor and drops the ambush flags and monster
facings from the options. Walls, doors, keys, pickups, start pose, looks, tick cap and map name
stay, so write the copy to a directory of its own. It also works with `--from-suite` and `--freeze`.

Rebuild the WADs of a frozen suite (the suite stores grids, so this never depends on the layout code)::

    python -m training.mapgen.cli --from-suite training/suites/heldout_v1.jsonl --out data/maps/heldout_v1

Write a new frozen suite from fresh seeds, or from the maps of an existing suite (`--tiers` chooses
which of its tiers; this is how the navigation suites were made, see training/suites/README.md)::

    python -m training.mapgen.cli --freeze training/suites/NAME.jsonl --n 100 --tiers 1,2,3,4 --seed 9000000
    python -m training.mapgen.cli --freeze training/suites/NAME_nav.jsonl --from-suite training/suites/NAME.jsonl \\
        --tiers 2,3,4 --strip-monsters

Print per-tier statistics of a manifest or suite::

    python -m training.mapgen.cli --stats data/maps/train/manifest.jsonl

Check every map of a manifest or suite inside the engine (loads, plays without warnings, and the
reference walker reaches the exit on a copy without monsters)::

    python -m training.mapgen.cli --verify data/maps/train/manifest.jsonl --workers 3

A manifest row has: name, path (as given with --out, so relative to the directory the command ran
in), file, tier, seed, episode_ticks, style, exit_style, randomized, grid, options (keyword
arguments of `udmf.build_wad`) and counts. Map (tier, seed) is always the same map. Seeds from
8,000,000 upwards are reserved for the frozen suites and refused for training sets, and a training
set never contains a grid that is in a suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from . import layouts, udmf
from .randomize import random_options

RESERVED_SEED = 8_000_000
SUITE_DIR = Path(__file__).resolve().parents[1] / "suites"
SUITE_KEYS = ("name", "tier", "seed", "episode_ticks", "style", "exit_style", "grid", "options", "counts")


def grid_hash(grid: str) -> str:
    return hashlib.sha1(grid.encode()).hexdigest()


def make_map(tier: int, seed: int, randomize: float = 0.0, strip: bool = False) -> dict:
    """Map (tier, seed) with its build options; a fraction `randomize` of maps gets randomised options.
    With `strip` the map comes without its monsters: same walls, doors, keys, pickups, start pose,
    looks and tick cap, so the copy of a map is that map as a pure navigation task."""
    m = layouts.generate(tier, seed)
    rng = random.Random(f"mapgen-options-v1/{tier}/{seed}")
    m["randomized"] = rng.random() < randomize
    if m["randomized"]:
        m["options"] = random_options(rng, m["grid"], m["options"])
        m["counts"]["ambush"] = len(m["options"].get("ambush", ()))
    m["name"] = f"t{tier}_{seed:07d}"
    return layouts.without_monsters(m) if strip else m


def read_rows(path) -> list[dict]:
    """Rows of a manifest or suite file (JSON lines)."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def suite_rows(path, tiers=None, strip: bool = False) -> list[dict]:
    """Rows of a suite file in file order: only those of `tiers` if given, without monsters if `strip`."""
    rows = [row for row in read_rows(path) if tiers is None or row["tier"] in tiers]
    return [layouts.without_monsters(row) for row in rows] if strip else rows


def frozen_grids() -> set[str]:
    """Hashes of every grid stored in the frozen suites."""
    found = set()
    for path in sorted(SUITE_DIR.glob("*.jsonl")):
        found |= {grid_hash(row["grid"]) for row in read_rows(path)}
    return found


def make_set(n: int, tiers, seed: int, randomize: float, exclude: set[str], strip: bool = False) -> list[dict]:
    """`n` distinct maps per tier from consecutive seeds, skipping duplicates and excluded grids.
    With `strip` every map comes without its monsters (see `make_map`)."""
    maps, seen = [], set(exclude)
    for tier in tiers:
        made, s = 0, seed
        while made < n:
            if s - seed > 50 * n + 1000:
                raise RuntimeError(f"tier {tier}: could not find {n} distinct maps")
            m = make_map(tier, s, randomize, strip)
            s += 1
            if grid_hash(m["grid"]) not in seen:
                seen.add(grid_hash(m["grid"]))
                maps.append(m)
                made += 1
    return maps


def write_wads(maps: list[dict], out) -> Path:
    """Compile every map into `out` and write `manifest.jsonl` next to the WADs. Returns the manifest path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for m in maps:
        path = udmf.write_wad(m["grid"], out / f"{m['name']}.wad", **m["options"])
        row = {"name": m["name"], "path": path.as_posix(), "file": path.name}
        row.update({k: m.get(k) for k in ("tier", "seed", "episode_ticks", "style", "exit_style")})
        row.update(randomized=bool(m.get("randomized", False)), grid=m["grid"], options=m["options"], counts=m["counts"])
        lines.append(json.dumps(row))
    manifest = out / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _write_suite(path: Path, maps: list[dict]) -> list[dict]:
    """Write `maps` as a suite file, named `<file stem>_t<tier>_<number within the tier>`."""
    index: Counter = Counter()
    rows = []
    for m in maps:
        row = {k: m[k] for k in SUITE_KEYS}
        row["name"] = f"{path.stem}_t{row['tier']}_{index[row['tier']]:03d}"
        index[row["tier"]] += 1
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows


def freeze_suite(path, n: int, tiers, seed: int, strip: bool = False) -> list[dict]:
    """Write a new suite file of `n` maps per tier in the reference dialect (without monsters if
    `strip` is set). Never overwrites."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen: suites are written once and never regenerated")
    if seed < RESERVED_SEED:
        raise ValueError(f"suite seeds start at {RESERVED_SEED}")
    return _write_suite(path, make_set(n, tiers, seed, randomize=0.0, exclude=frozen_grids(), strip=strip))


def derive_suite(path, source, tiers=None, strip: bool = False) -> list[dict]:
    """Write a new suite file from the maps of an existing one: the rows of `source` whose tier is in
    `tiers` (all tiers if None), in the same order, without their monsters if `strip` is set.
    Never overwrites.

    A tier is always taken whole and numbered from zero again, so map i of a tier is the same layout
    with the same tick cap in both files: `heldout_nav_v1_t3_004` is `heldout_v1_t3_004` without its
    monsters. Scoring a policy on both files separates navigation from combat.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen: suites are written once and never regenerated")
    maps = suite_rows(source, tiers, strip)
    if not maps:
        raise ValueError(f"{source} has no map of the tiers asked for")
    return _write_suite(path, maps)


def suite_checksums(path) -> dict:
    """The entry of `checksums.json` for a suite file: number of maps, SHA-256 of the file, and
    SHA-256 of the WADs built from its rows, taken together in file order."""
    path = Path(path)
    rows = read_rows(path)
    wads = hashlib.sha256()
    for row in rows:
        wads.update(udmf.build_wad(row["grid"], **row["options"]))
    return {"maps": len(rows), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "wads_sha256": wads.hexdigest()}


def verify_map(row: dict) -> dict:
    """Engine check of one manifest or suite row: the map loads in `doom_env` with the expected
    number of sectors, 25 random steps raise no warning and, if there is an exit, the reference
    walker reaches it on a copy without monsters. Returns the name, whether the exit was reached
    (None without an exit) and the share of the tick cap the walker used."""
    import tempfile
    import warnings

    import numpy as np
    from doom_env import make_env

    from . import walker

    out = {"name": row["name"], "reached": None, "cap_used": None}
    rng = np.random.default_rng(row["seed"])
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("error")
        env = make_env(str(udmf.write_wad(row["grid"], Path(tmp) / "map.wad", **row["options"])),
                       episode_ticks=row["episode_ticks"], render_buffers=False)
        env.reset(seed=0)
        if len(env.privileged_state()["sector_floor"]) != row["counts"]["sectors"]:
            raise RuntimeError(f"{row['name']}: sector count differs from the grid")
        for _ in range(25):
            _, _, terminated, truncated, _ = env.step(rng.integers(0, 2, 9).astype(np.int8))
            if terminated or truncated:
                break
        env.close()
        if row["counts"]["exits"]:
            bare = layouts.without_monsters(row)
            env = make_env(str(udmf.write_wad(bare["grid"], Path(tmp) / "bare.wad", **bare["options"])),
                           episode_ticks=row["episode_ticks"], render_buffers=False)
            result = walker.walk(env, bare["grid"])
            env.close()
            out.update(reached=result["exit_reached"], cap_used=4 * result["steps"] / row["episode_ticks"])
    return out


def verify(rows: list[dict], workers: int = 1) -> bool:
    """Run `verify_map` over all rows, print a summary and return True if every map passed."""
    if workers > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap_unordered(verify_map, rows, chunksize=8))
    else:
        results = [verify_map(row) for row in rows]
    walked = [r for r in results if r["reached"] is not None]
    failed = sorted(r["name"] for r in walked if not r["reached"])
    used = [r["cap_used"] for r in walked if r["reached"]]
    print(f"{len(results)} maps loaded and played without warnings; {len(walked)} have an exit; "
          f"the reference walker reached it on {len(walked) - len(failed)}")
    if used:
        print(f"share of the tick cap used by the walker: mean {statistics.mean(used):.2f}, max {max(used):.2f}")
    if failed:
        print("exit not reached:", " ".join(failed))
    return not failed


def tier_stats(rows: list[dict]) -> str:
    """Per-tier table: size, rooms, doors, locks, monsters and tick caps (minimum / mean / maximum)."""
    out = []
    for tier in sorted({row["tier"] for row in rows}):
        part = [row for row in rows if row["tier"] == tier]

        def span(key: str) -> str:
            values = [row["counts"][key] for row in part]
            return f"{min(values)}/{statistics.mean(values):.1f}/{max(values)}"

        caps = sorted(Counter(row["episode_ticks"] for row in part).items())
        out.append(
            f"tier {tier}: {len(part)} maps | width {span('width')} height {span('height')} open cells {span('open_cells')}"
            f" | rooms {span('rooms')} doors {span('doors')} locks {span('locks')} keys {span('keys')}"
            f" | monsters {span('monsters')} pickups {span('pickups')} ambush flags {span('ambush')}"
            f" | dead ends {span('dead_ends')} route cells {span('route_cells')}\n"
            f"        with exit {sum(row['counts']['exits'] > 0 for row in part)}, door-free {sum(row['counts']['doors'] == 0 for row in part)},"
            f" monster-free {sum(row['counts']['monsters'] == 0 for row in part)},"
            f" with a lock and no monster {sum(row['counts']['locks'] > 0 and row['counts']['monsters'] == 0 for row in part)},"
            f" randomized {sum(bool(row.get('randomized')) for row in part)},"
            f" styles {dict(sorted(Counter(row.get('style') for row in part).items()))}\n"
            f"        tick caps {' '.join(f'{cap}:{count}' for cap, count in caps)}"
        )
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m training.mapgen.cli", description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="directory for the WAD files and manifest.jsonl")
    ap.add_argument("--n", type=int, default=200, help="maps per tier")
    ap.add_argument("--tiers", help="comma-separated tiers (default: 0,1,2,3,4, or every tier of the --from-suite file)")
    ap.add_argument("--seed", type=int, default=0, help="first map seed; map i of a tier uses seed + i")
    ap.add_argument("--randomize", type=float, nargs="?", const=0.5, default=0.0, metavar="P",
                    help="fraction of maps built with randomised options (0.5 when given without a value)")
    ap.add_argument("--from-suite", metavar="FILE",
                    help="take the maps of a frozen suite (those of --tiers): rebuild their WADs into --out, "
                         "or with --freeze write them as a new suite")
    ap.add_argument("--freeze", metavar="FILE", help="write a new frozen suite (refuses to overwrite)")
    ap.add_argument("--strip-monsters", action="store_true",
                    help="write every map without its monsters (training sets, --from-suite and --freeze): "
                         "the same layouts as pure navigation tasks")
    ap.add_argument("--stats", metavar="FILE", help="print per-tier statistics of a manifest or suite")
    ap.add_argument("--verify", metavar="FILE", help="check every map of a manifest or suite in the engine")
    ap.add_argument("--workers", type=int, default=1, help="processes for --verify")
    args = ap.parse_args(argv)
    tiers = [int(t) for t in args.tiers.split(",")] if args.tiers else None     # None: no choice was made
    began = time.perf_counter()

    if args.stats:
        print(tier_stats(read_rows(args.stats)))
        return
    if args.verify:
        ok = verify(read_rows(args.verify), args.workers)
        print(f"{time.perf_counter() - began:.0f} s")
        raise SystemExit(0 if ok else 1)
    if args.freeze:
        if args.from_suite:
            rows = derive_suite(args.freeze, args.from_suite, tiers, args.strip_monsters)
        else:
            rows = freeze_suite(args.freeze, args.n, tiers or sorted(layouts.TIERS), args.seed, args.strip_monsters)
        print(f"froze {len(rows)} maps in {args.freeze}")
        print(tier_stats(rows))
        print("entry for checksums.json:", json.dumps({Path(args.freeze).name: suite_checksums(args.freeze)}))
        return
    if not args.out:
        ap.error("--out is required")
    if args.from_suite:
        maps = suite_rows(args.from_suite, tiers, args.strip_monsters)
    else:
        if args.seed < 0 or args.seed + 50 * args.n + 1000 >= RESERVED_SEED:
            ap.error(f"training seeds must stay below {RESERVED_SEED} (reserved for the frozen suites)")
        maps = make_set(args.n, tiers or sorted(layouts.TIERS), args.seed, args.randomize, exclude=frozen_grids(),
                        strip=args.strip_monsters)
    made = time.perf_counter()
    manifest = write_wads(maps, args.out)
    done = time.perf_counter()
    print(f"wrote {len(maps)} maps to {args.out} ({manifest})")
    print(f"layout {made - began:.2f} s, compile and write {done - made:.2f} s, {len(maps) / (done - began):.0f} maps/s")
    print(tier_stats(read_rows(manifest)))


if __name__ == "__main__":
    main()
