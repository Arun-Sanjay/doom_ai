"""Command line of the map generator. Training-time scaffolding, never imported by the agent.

Write a training set (WAD files plus `manifest.jsonl`); `--n` is the number of maps per tier::

    python -m training.mapgen.cli --out data/maps/train --n 300 --tiers 0,1,2,3,4 --seed 0 --randomize 0.5

Write the same maps without their monsters, to train and measure navigation apart from combat::

    python -m training.mapgen.cli --out data/maps/train_nav --n 300 --tiers 0,2,3,4 --seed 0 --randomize 0.5 --strip-monsters

Write a training set whose starts are mostly on the 64-unit grid, as in every reference map. The
three `--start-*` flags give the start pose its own draw on every map of the set, whether or not
its looks are randomised: a quarter turn (still on the grid), an arbitrary heading, a sub-cell
offset. This is the command that wrote `data/maps/train_v2`::

    python -m training.mapgen.cli --out data/maps/train_v2 --n 600 --tiers 0,1,2,3,4 --seed 6000000 --randomize 0.5 \\
        --start-quarter 0.5 --start-any 0.05 --start-offset 0.05

Without a `--start-*` flag the start is randomised as in the first set: only on the maps of the
`--randomize` fraction, and off the grid on about four of five of those.

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

Write maps of the wide profile (`layouts.WIDE`: larger arenas, more monsters, and in tiers 2 to 4 a
long route under a tick cap of 1.5 to 3 times its cost). `--profile wide` applies to the maps the
command generates, `--n` may give one number per tier, and `--extend` first copies an existing set
(rows and WAD files, byte for byte) into `--out`; every row then names its profile. This is the
command that wrote `data/maps/train_v3`, the 3,000 maps of `train_v2` plus 1,500 wide ones::

    python -m training.mapgen.cli --out data/maps/train_v3 --extend data/maps/train_v2/manifest.jsonl --profile wide \\
        --n 150,350,500,500 --tiers 1,2,3,4 --seed 7000000 --randomize 0.5 \\
        --start-quarter 0.5 --start-any 0.05 --start-offset 0.05

and these froze the two wide suites::

    python -m training.mapgen.cli --freeze training/suites/heldout_wide_v1.jsonl --profile wide --n 50 --tiers 1,2,3,4 --seed 9500000
    python -m training.mapgen.cli --freeze training/suites/dev_wide_v1.jsonl --profile wide --n 15 --tiers 1,2,3,4 --seed 8500000

Write maps in the legend of the reference maps (`layouts.REFERENCE`: zombieman, imp and chaingunner
as the only monsters, the stimpack as the only pickup, an exit on every map). `--profile reference`
has the base sizes, `--profile reference_wide` the long routes of the wide profile with crowded
rooms; both have tiers 1 to 4. A set of both is written in two steps, the second one extending the
first (PART is any directory outside the set), and `--mix PROFILE:SEED` freezes a suite whose tiers hold `--n` maps of each profile, with
the profile named in every row. These wrote `data/maps/train_ref_v1` and the two suites::

    python -m training.mapgen.cli --out PART --profile reference \\
        --n 225,375,375,525 --tiers 1,2,3,4 --seed 7200000 --randomize 0.5 \\
        --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
    python -m training.mapgen.cli --out data/maps/train_ref_v1 --extend PART/manifest.jsonl \\
        --profile reference_wide --n 225,375,375,525 --tiers 1,2,3,4 --seed 7300000 --randomize 0.5 \\
        --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
    python -m training.mapgen.cli --freeze training/suites/dev_ref_v1.jsonl --profile reference --n 10 \\
        --tiers 1,2,3,4 --seed 8700000 --mix reference_wide:8750000
    python -m training.mapgen.cli --freeze training/suites/heldout_ref_v1.jsonl --profile reference --n 30 \\
        --tiers 1,2,3,4 --seed 9700000 --mix reference_wide:9750000

Print per-tier statistics of a manifest or suite (per profile when the rows name one; with
`--profile` only the rows of that profile)::

    python -m training.mapgen.cli --stats data/maps/train/manifest.jsonl

Check every map of a manifest or suite inside the engine (loads, plays without warnings, and the
reference walker reaches the exit on a copy without monsters; `--profile` chooses rows as above)::

    python -m training.mapgen.cli --verify data/maps/train/manifest.jsonl --workers 3

A manifest row has: name, path (as given with --out, so relative to the directory the command ran
in), file, tier, seed, episode_ticks, style, exit_style, randomized, on_grid, grid, options (keyword
arguments of `udmf.build_wad`) and counts. `on_grid` is true when the player starts on a cell centre
facing along an axis, so the walls lie on the 64-unit cells of the agent's start frame; the
navigation teacher gives labels on those maps only. A set written with `--profile` or `--extend`
also has `profile` ("base" or "wide") after `on_grid`. Map (tier, seed) is always the same map
under the same flags. Seeds from
8,000,000 upwards are reserved for the frozen suites and refused for training sets, and a training
set never contains a grid that is in a suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

from . import layouts, udmf
from .randomize import START_DEFAULTS, random_options, random_start, start_on_grid

RESERVED_SEED = 8_000_000
SUITE_DIR = Path(__file__).resolve().parents[1] / "suites"
SUITE_KEYS = ("name", "tier", "seed", "episode_ticks", "style", "exit_style", "grid", "options", "counts")
ROW_KEYS = ("name", "path", "file", "tier", "seed", "episode_ticks", "style", "exit_style", "randomized", "on_grid",
            "profile", "grid", "options", "counts")


def grid_hash(grid: str) -> str:
    return hashlib.sha1(grid.encode()).hexdigest()


def make_map(tier: int, seed: int, randomize: float = 0.0, strip: bool = False, start=None, profile: str = "base") -> dict:
    """Map (tier, seed) with its build options; a fraction `randomize` of maps gets randomised options.
    With `strip` the map comes without its monsters: same walls, doors, keys, pickups, start pose,
    looks and tick cap, so the copy of a map is that map as a pure navigation task.

    `start` is None or the three probabilities of `randomize.random_start` (quarter turn, arbitrary
    heading, sub-cell offset). None is the first training set: the start pose is randomised together
    with the looks, on the maps of the `randomize` fraction only. With probabilities the start pose
    has a draw of its own on every map, from a random stream of its own, and the looks of the
    `randomize` fraction are drawn without it.

    `profile` is the layout profile of `layouts.generate`. A profile other than "base" gives another
    map for the same (tier, seed), with looks and start drawn from streams of its own."""
    m = layouts.generate(tier, seed, profile)
    own = "" if profile == "base" else f"{profile}/"
    rng = random.Random(f"mapgen-options-v1/{own}{tier}/{seed}")
    m["randomized"] = rng.random() < randomize
    if m["randomized"]:
        m["options"] = random_options(rng, m["grid"], m["options"], start=start is None)
        m["counts"]["ambush"] = len(m["options"].get("ambush", ()))
    if start is not None:
        m["options"] = {**m["options"], **random_start(random.Random(f"mapgen-start-v1/{own}{tier}/{seed}"), *start)}
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


def make_set(n, tiers, seed: int, randomize: float, exclude: set[str], strip: bool = False, start=None,
             profile: str = "base") -> list[dict]:
    """`n` distinct maps per tier from consecutive seeds, skipping duplicates and excluded grids.
    `n` is one number for every tier or a list with one number per tier. With `strip` every map
    comes without its monsters, `start` sets how the start pose is drawn and `profile` is the
    layout profile (see `make_map`)."""
    tiers = list(tiers)
    counts = [n] * len(tiers) if isinstance(n, int) else list(n)
    if len(counts) != len(tiers):
        raise ValueError("one number of maps for all tiers, or one per tier")
    maps, seen = [], set(exclude)
    for tier, n in zip(tiers, counts):
        made, s = 0, seed
        while made < n:
            if s - seed > 50 * n + 1000:
                raise RuntimeError(f"tier {tier}: could not find {n} distinct maps")
            m = make_map(tier, s, randomize, strip, start, profile)
            s += 1
            if grid_hash(m["grid"]) not in seen:
                seen.add(grid_hash(m["grid"]))
                maps.append(m)
                made += 1
    return maps


def copy_set(manifest, out) -> list[dict]:
    """Copy the WAD files of an existing training set into `out`, byte for byte, and return its
    manifest rows for the new place: `path` points at the copy and `profile` is added ("base" for
    a row that names none). Everything else in a row is as it was; no map is generated or compiled."""
    source, out = Path(manifest), Path(out)
    if source.parent.resolve() == out.resolve():
        raise ValueError("--extend copies a set into another directory")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_rows(source):
        target = out / row["file"]
        shutil.copyfile(source.parent / row["file"], target)
        full = {**row, "path": target.as_posix(), "on_grid": row.get("on_grid", start_on_grid(row.get("options"))),
                "profile": row.get("profile", "base")}
        rows.append({**{k: full[k] for k in ROW_KEYS}, **{k: v for k, v in full.items() if k not in ROW_KEYS}})
    return rows


def write_wads(maps: list[dict], out, profiles: bool = False, copied: list[dict] | None = None) -> Path:
    """Compile every map into `out` and write `manifest.jsonl` next to the WADs. Returns the manifest path.
    With `profiles` every row names its layout profile. `copied` are the rows of `copy_set`, whose
    files are in `out` already: they come first in the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in copied or ()]
    for m in maps:
        path = udmf.write_wad(m["grid"], out / f"{m['name']}.wad", **m["options"])
        row = {"name": m["name"], "path": path.as_posix(), "file": path.name}
        row.update({k: m.get(k) for k in ("tier", "seed", "episode_ticks", "style", "exit_style")})
        row.update(randomized=bool(m.get("randomized", False)), on_grid=start_on_grid(m["options"]))
        if profiles:
            row.update(profile=m.get("profile", "base"))
        row.update(grid=m["grid"], options=m["options"], counts=m["counts"])
        lines.append(json.dumps(row))
    manifest = out / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _write_suite(path: Path, maps: list[dict], profiles: bool = False) -> list[dict]:
    """Write `maps` as a suite file, named `<file stem>_t<tier>_<number within the tier>`. With
    `profiles` every row also names its layout profile, after the suite keys."""
    index: Counter = Counter()
    rows = []
    for m in maps:
        row = {k: m[k] for k in SUITE_KEYS}
        if profiles:
            row["profile"] = m.get("profile", "base")
        row["name"] = f"{path.stem}_t{row['tier']}_{index[row['tier']]:03d}"
        index[row["tier"]] += 1
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows


def freeze_suite(path, n, tiers, seed: int, strip: bool = False, profile: str = "base") -> list[dict]:
    """Write a new suite file of `n` maps per tier in the reference dialect (without monsters if
    `strip` is set), from the layout profile `profile`. Never overwrites."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen: suites are written once and never regenerated")
    if seed < RESERVED_SEED:
        raise ValueError(f"suite seeds start at {RESERVED_SEED}")
    return _write_suite(path, make_set(n, tiers, seed, randomize=0.0, exclude=frozen_grids(), strip=strip, profile=profile))


def freeze_mixed_suite(path, n, tiers, parts, strip: bool = False) -> list[dict]:
    """Write a new suite file whose tiers each hold `n` maps of every (profile, seed) of `parts`, in
    that order: the first `n` maps of a tier are of the first profile, the next `n` of the second.
    Reference dialect, no grid twice, every row names its profile. Never overwrites."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen: suites are written once and never regenerated")
    if min(seed for _, seed in parts) < RESERVED_SEED:
        raise ValueError(f"suite seeds start at {RESERVED_SEED}")
    exclude, maps = frozen_grids(), []
    for profile, seed in parts:
        part = make_set(n, tiers, seed, randomize=0.0, exclude=exclude, strip=strip, profile=profile)
        exclude |= {grid_hash(m["grid"]) for m in part}
        maps += part
    maps.sort(key=lambda m: list(tiers).index(m["tier"]))          # stable: profiles stay in order inside a tier
    return _write_suite(path, maps, profiles=True)


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


def budget(row: dict) -> float | None:
    """Tick cap of a map divided by the cost of its shortest route in tics (None without an exit)."""
    steps = row["counts"].get("optimal_steps") if row["counts"].get("exits") else None
    return row["episode_ticks"] / (layouts.FRAME_SKIP * steps) if steps else None


def profile_stats(rows: list[dict]) -> str:
    """`tier_stats` of the rows, one table per layout profile when the rows name more than one."""
    names = sorted({row.get("profile", "base") for row in rows})
    if len(names) < 2:
        return tier_stats(rows)
    return "\n".join(f"profile {name}\n" + tier_stats([row for row in rows if row.get("profile", "base") == name])
                     for name in names)


def tier_stats(rows: list[dict]) -> str:
    """Per-tier table: size, rooms, doors, locks, monsters and tick caps (minimum / mean / maximum)."""
    out = []
    for tier in sorted({row["tier"] for row in rows}):
        part = [row for row in rows if row["tier"] == tier]
        budgets = [b for b in map(budget, part) if b is not None]

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
            f" start on the grid {sum(start_on_grid(row.get('options')) for row in part)},"
            f" styles {dict(sorted(Counter(row.get('style') for row in part).items()))}\n"
            f"        tick caps {' '.join(f'{cap}:{count}' for cap, count in caps)}"
            + (f" | tick cap over route cost {min(budgets):.2f}/{statistics.mean(budgets):.2f}/{max(budgets):.2f}" if budgets else "")
        )
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m training.mapgen.cli", description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="directory for the WAD files and manifest.jsonl")
    ap.add_argument("--n", default="200", help="maps per tier: one number, or one per tier of --tiers (150,350,500,500)")
    ap.add_argument("--tiers", help="comma-separated tiers (default: 0,1,2,3,4, or every tier of the --from-suite file)")
    ap.add_argument("--seed", type=int, default=0, help="first map seed; map i of a tier uses seed + i")
    ap.add_argument("--randomize", type=float, nargs="?", const=0.5, default=0.0, metavar="P",
                    help="fraction of maps built with randomised options (0.5 when given without a value)")
    ap.add_argument("--start-quarter", type=float, metavar="P",
                    help="probability that a map starts with a quarter turn (facing north, west or south; still on the grid)")
    ap.add_argument("--start-any", type=float, metavar="P",
                    help="probability that a map starts with a heading that is not a multiple of 90 degrees (off the grid)")
    ap.add_argument("--start-offset", type=float, metavar="P",
                    help="probability that a map starts up to 16 units off its cell centre (off the grid). Any --start-* "
                         "flag gives the start pose its own draw on every map; the flags left out then take "
                         f"{START_DEFAULTS[0]}, {START_DEFAULTS[1]} and {START_DEFAULTS[2]}. With none of them the "
                         "start is randomised with the looks, as in the first training set")
    ap.add_argument("--profile", choices=layouts.ALL_PROFILES,
                    help="layout profile of the maps this command generates (default: base). 'wide' has wider "
                         "limits, see layouts.WIDE; 'reference' and 'reference_wide' keep to the legend of the "
                         "reference maps, see layouts.REFERENCE; the manifest rows then name their profile. With "
                         "--stats and --verify: only the rows of this profile")
    ap.add_argument("--mix", metavar="PROFILE:SEED",
                    help="with --freeze: every tier also gets --n maps of this profile from this seed, after those "
                         "of --profile; the suite rows then name their profile")
    ap.add_argument("--extend", metavar="MANIFEST",
                    help="training sets: first copy the maps of this manifest (rows and WAD files, unchanged) into "
                         "--out, then add the generated maps; the rows name their profile")
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
    profile = args.profile or "base"
    try:
        n = [int(v) for v in args.n.split(",")]
    except ValueError:
        ap.error("--n is a number or a comma-separated list of numbers")
    all_tiers = sorted(layouts.WIDE["size"]) if profile != "base" else sorted(layouts.TIERS)
    if min(n) < 0 or (len(n) > 1 and len(n) != len(tiers or all_tiers)):
        ap.error("--n takes one number, or one number for every tier of --tiers")
    n = n[0] if len(n) == 1 else n
    if profile != "base" and not set(tiers or all_tiers) <= set(all_tiers) and not (args.stats or args.verify or args.from_suite):
        ap.error(f"the {profile} profile has tiers 1 to 4")
    mix = None
    if args.mix:
        name, _, mix_seed = args.mix.partition(":")
        if name not in layouts.ALL_PROFILES or not mix_seed.isdigit() or not args.freeze or args.from_suite:
            ap.error("--mix PROFILE:SEED names a layout profile and a seed, for --freeze without --from-suite")
        if name != "base" and not set(tiers or all_tiers) <= set(layouts.WIDE["size"]):
            ap.error(f"the {name} profile has tiers 1 to 4")
        mix = (name, int(mix_seed))
    if args.from_suite and (args.profile or args.extend):
        ap.error("--profile and --extend apply to generated maps, not to --from-suite")
    if args.extend and (args.freeze or args.strip_monsters):
        ap.error("--extend copies a training set as it is: not with --freeze or --strip-monsters")
    start = (args.start_quarter, args.start_any, args.start_offset)
    start = None if all(p is None for p in start) else tuple(d if p is None else p for p, d in zip(start, START_DEFAULTS))
    if start is not None and (args.from_suite or args.freeze or min(start) < 0 or max(start) > 1 or start[0] + start[1] > 1):
        ap.error("--start-* are probabilities for a training set (--out without --from-suite), "
                 "and --start-quarter plus --start-any may not exceed 1")
    began = time.perf_counter()

    def chosen(file) -> list[dict]:
        """Rows of a manifest or suite; only those of --profile when it is given."""
        return [row for row in read_rows(file) if args.profile in (None, row.get("profile", "base"))]

    if args.stats:
        print(profile_stats(chosen(args.stats)))
        return
    if args.verify:
        ok = verify(chosen(args.verify), args.workers)
        print(f"{time.perf_counter() - began:.0f} s")
        raise SystemExit(0 if ok else 1)
    if args.freeze:
        if args.from_suite:
            rows = derive_suite(args.freeze, args.from_suite, tiers, args.strip_monsters)
        elif mix:
            rows = freeze_mixed_suite(args.freeze, n, tiers or all_tiers, ((profile, args.seed), mix), args.strip_monsters)
        else:
            rows = freeze_suite(args.freeze, n, tiers or all_tiers, args.seed, args.strip_monsters, profile)
        print(f"froze {len(rows)} maps in {args.freeze}")
        print(profile_stats(rows) if mix else tier_stats(rows))
        print("entry for checksums.json:", json.dumps({Path(args.freeze).name: suite_checksums(args.freeze)}))
        return
    if not args.out:
        ap.error("--out is required")
    copied = None
    if args.from_suite:
        maps = suite_rows(args.from_suite, tiers, args.strip_monsters)
    else:
        most = n if isinstance(n, int) else max(n)
        if args.seed < 0 or args.seed + 50 * most + 1000 >= RESERVED_SEED:
            ap.error(f"training seeds must stay below {RESERVED_SEED} (reserved for the frozen suites)")
        exclude = frozen_grids()
        if args.extend:
            if Path(args.extend).parent.resolve() == Path(args.out).resolve():
                ap.error("--extend copies a set into another directory than its own")
            exclude |= {grid_hash(row["grid"]) for row in read_rows(args.extend)}
        maps = make_set(n, tiers or all_tiers, args.seed, args.randomize, exclude=exclude,
                        strip=args.strip_monsters, start=start, profile=profile)
        if args.extend:
            taken = {row["name"] for row in read_rows(args.extend)} & {m["name"] for m in maps}
            if taken:
                ap.error(f"--seed gives {len(taken)} maps a name of the extended set (for example {sorted(taken)[0]}): use other seeds")
            copied = copy_set(args.extend, args.out)
    made = time.perf_counter()
    manifest = write_wads(maps, args.out, profiles=bool(args.profile or args.extend), copied=copied)
    done = time.perf_counter()
    print(f"wrote {len(maps)} maps to {args.out} ({manifest})" + (f", after {len(copied)} copied from {args.extend}" if copied else ""))
    print(f"layout {made - began:.2f} s, compile and write {done - made:.2f} s, {len(maps) / (done - began):.0f} maps/s")
    print(profile_stats(read_rows(manifest)))


if __name__ == "__main__":
    main()
