"""Score a policy on a frozen suite, log the result, and compare two logged results.

Training-time scaffolding, never imported by the agent.

    python -m training.evaluate --policy training.runner:AgentPolicy \\
        --suite training/suites/heldout_v1.jsonl --tag NAME [--workers 3] [--seeds 0]
        [--time-scale 2.5 | --no-time-limits] [--strict]
    python -m training.evaluate --compare TAG_A TAG_B      # exit status 0 only if B beats A
    python -m training.evaluate --table                    # latest result of every tag, best first

A suite is a JSON-lines file with one map per line, stored as an ASCII grid so that later changes
to the map generator cannot alter it:

    {"grid": "#####\\n#P.E#\\n#####", "tier": 2, "episode_ticks": 2800, "options": {...}, ...}

The WADs are rebuilt from the grids with `training.mapgen.udmf.build_wad(grid, **options)` into
`data/suites/<suite name>/` before every run. A line that carries `path` (or `wad`) to a WAD that
exists, which is how a generator manifest looks, is played from that file instead.

Every map is played once per seed with fixed engine seeds. One row per tier, plus a row with
tier `all`, is appended to `training/logs/results.csv` (columns: RESULT_COLUMNS). The per-episode
rows go to `runs/eval/<tag>.csv` and make the comparison a paired one.

The official time limits apply (docs/EVALUATION.md; `training.runner`, "Time limits"): 10 s for
`reset()` and for the first `act()` of a map, 5 s for any other `act()`, and 40 ms times the
decision steps in the map's tick cap for all `act()` calls of the map together. An agent that
exceeds one stops and stands still for the rest of the map, as it would in the evaluation.
`--time-scale X` multiplies the measured time by X first, to judge the agent as a machine X times
slower than this one would (an estimate: an agent that watches its own clock still sees this
machine's time); `--no-time-limits` is for training-only policies such as a teacher.

Whatever went wrong is printed below the table as a WARNING with a count and the first message,
and counted in the table's last three columns: `agent` (episodes in which the agent raised,
returned an invalid action or ran out of time, and stood still from then on), `hook` (a
telemetry hook of the policy failed; the score is not affected) and `err` (the episode could not
be played, or is void, and counts as score 0). The result is logged either way; with `--strict`
the exit status is 2 when `agent` or `err` is not zero.

`--compare A B` is the promote-only-if-better check: B is better only if its tier-weighted score
exceeds A's by more than one composite point and by more than two standard errors of the
difference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from doom_env.limits import AGENT_MS_PER_STEP, FIRST_ACT_TIMEOUT_S, agent_budget_s
from training import paths
from training.runner import ROW_FIELDS, find_wad, is_timeout, normalise_map, run_episodes, spec_name
from training.scorer import SUMMARY_FIELDS, TIER_WEIGHTS, suite_score, summarise

RESULTS_CSV = paths.ROOT / "training" / "logs" / "results.csv"
EPISODES_DIR = paths.ROOT / "runs" / "eval"
MAPS_DIR = paths.ROOT / "data" / "suites"
# Columns of results.csv: what was run, then the aggregated metrics of scorer.summarise().
# `time_scale` is the factor the official time limits were applied with (empty: they were not applied).
RESULT_COLUMNS = ("stamp", "tag", "commit", "suite", "suite_sha", "policy", "seeds", "time_scale", "tier") + SUMMARY_FIELDS
NOISE_POINTS = 1.0   # composite points: differences below this are a tie whatever the statistics say
NOISE_SIGMAS = 2.0   # ... and so are differences under this many standard errors
HEADROOM = 0.6       # warn when the agent uses more than this fraction of a map's compute budget


# ------------------------------------------------------------------------------------ suite
def _suite_path(path) -> Path:
    """The suite file as given, or relative to the repository root when it is not found as given."""
    path = Path(path)
    if not path.is_file() and not path.is_absolute() and (paths.ROOT / path).is_file():
        return paths.ROOT / path
    return path


def load_suite(path) -> list[dict]:
    """Read a suite or manifest: one dict per map with id, tier, episode_ticks and grid+options or wad."""
    path = _suite_path(path)
    if not path.is_file():
        raise SystemExit(f"suite file not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        grid = raw.get("grid")
        if isinstance(grid, list):
            grid = "\n".join(grid)
        index = len(rows)
        rows.append({
            "index": index,
            "id": str(raw.get("id") or raw.get("name") or raw.get("map_id") or f"{path.stem}-{index:04d}"),
            "tier": int(raw.get("tier", 0) or 0),
            "episode_ticks": int(raw.get("episode_ticks") or raw.get("ticks") or raw.get("tick_cap") or 2100),
            "grid": grid, "options": dict(raw.get("options") or raw.get("opts") or {}),
            "wad": raw.get("wad") or raw.get("path"), "map": raw.get("map") or "MAP01",
        })
    if not rows:
        raise SystemExit(f"suite {path} has no maps")
    if len({row["id"] for row in rows}) != len(rows):   # results are matched across runs by map id
        raise SystemExit(f"suite {path} uses the same map name or id on more than one line")
    return rows


def _write_if_changed(path: Path, data: bytes) -> None:
    """Atomic write that leaves an identical file untouched (several evaluations may share the directory)."""
    if path.is_file() and path.read_bytes() == data:
        return
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def suite_maps(path, maps_dir=MAPS_DIR, builder=None) -> list[dict]:
    """Map entries for the runner, building the WAD of every row that is stored as a grid.

    `builder(grid, **options) -> bytes` defaults to `training.mapgen.udmf.build_wad`.
    """
    path = _suite_path(path)
    rows = load_suite(path)
    out_dir = Path(maps_dir) / path.stem
    maps = []
    for row in rows:
        wad = find_wad(row["wad"], path.parent) if row["wad"] else None   # a WAD that is already built wins
        if wad is None and row["grid"] is None:
            raise SystemExit(f"{path}: map {row['index'] + 1} ({row['id']}) has no grid and its WAD was not found: {row['wad']}")
        if wad is None:
            if builder is None:
                try:
                    from training.mapgen.udmf import build_wad as builder
                except ImportError as e:
                    raise SystemExit(
                        f"{path} stores its maps as ASCII grids, and building their WADs needs the map generator "
                        f"(training.mapgen.udmf.build_wad), which cannot be imported here: {e}. "
                        "Add the generator, or pass a manifest whose lines have a 'path' to existing WADs.") from None
            out_dir.mkdir(parents=True, exist_ok=True)
            wad = out_dir / f"{row['index']:04d}.wad"
            _write_if_changed(wad, builder(row["grid"], **row["options"]))
        maps.append(normalise_map({"id": row["id"], "wad": str(wad), "map": row["map"], "tier": row["tier"],
                                   "episode_ticks": row["episode_ticks"]}))
    return maps


# ---------------------------------------------------------------------------------- logging
def git_commit() -> str:
    """Short hash of HEAD, with "+dirty" when the working tree differs from it (modified or
    untracked files that are not ignored); "unknown" outside git. Both commands only read."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "--no-optional-locks", "-C", str(paths.ROOT), *args],
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = git("rev-parse", "--short", "HEAD")
    if not head:
        return "unknown"
    return head + ("+dirty" if git("status", "--porcelain") else "")


def _cell(value) -> str:
    """CSV text of a value: empty for missing or NaN, at most four decimals for floats."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return str(int(value)) if value == int(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def tier_rows(rows: list[dict], info: dict) -> list[dict]:
    """The results.csv rows of one evaluation: one per tier present, then tier `all`.

    In the `all` row the three scores are tier-weighted (see scorer.suite_score); its other
    columns are plain aggregates over all episodes.
    """
    total = suite_score(rows)
    out = [{**info, "tier": str(tier), **summary} for tier, summary in total["tiers"].items()]
    overall = summarise(rows)
    overall.update(score=total["score"], score_se=total["score_se"],
                   score_exit_heavy=suite_score(rows, "score_exit_heavy")["score"],
                   score_balanced=suite_score(rows, "score_balanced")["score"])
    out.append({**info, "tier": "all", **overall})
    return out


def append_results(path, new_rows: list[dict]) -> None:
    """Append rows to results.csv under a file lock; the header is extended if columns were added since."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except ImportError:   # no advisory locks on this platform
            pass
        f.seek(0)
        existing = list(csv.reader(f))
        header = existing[0] if existing else []
        columns = list(RESULT_COLUMNS) + [c for c in header if c and c not in RESULT_COLUMNS]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", restval="", lineterminator="\n")
        if header != columns:   # new file, or an older layout: rewrite it under the current header
            f.seek(0)
            f.truncate()
            writer.writeheader()
            writer.writerows(dict(zip(header, old)) for old in existing[1:])
        else:
            f.seek(0, os.SEEK_END)
        for row in new_rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})


def read_results(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def write_episodes(path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(ROW_FIELDS) + sorted({k for r in rows for k in r} - set(ROW_FIELDS))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})
    os.replace(tmp, path)


SCORES = ("score", "score_exit_heavy", "score_balanced")   # the default weighting and the two alternatives


def read_episodes(path) -> dict[tuple[str, str], dict]:
    """(map id, seed) -> {"tier": int, "score": ..., "score_exit_heavy": ..., "score_balanced": ...}
    from a per-episode file; {} if the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {(r["map_id"], r["seed"]): {"tier": int(r["tier"] or 0), **{k: float(r.get(k) or 0.0) for k in SCORES}}
                for r in csv.DictReader(f)}


# --------------------------------------------------------------------------------- evaluate
def _first(rows: list[dict], key: str) -> str:
    """`map seed n: last line of the message` for the first of `rows`, whose field `key` holds a message."""
    row = rows[0]
    return f"{row['map_id']} seed {row['seed']}: {str(row[key]).strip().splitlines()[-1]}"


def budget_use(row: dict, time_scale: float = 1.0) -> float:
    """Fraction of the map's compute budget (40 ms x decision steps in its tick cap) that act() used.

    For an episode in which the agent acted on every step; `time_scale` as in `evaluate`.
    """
    return row["act_ms_mean"] * row["steps"] * time_scale / 1000.0 / agent_budget_s(int(row["episode_ticks"]))


def problems(rows: list[dict], time_limits: bool = True, time_scale: float = 1.0) -> list[str]:
    """One line for every kind of thing that went wrong in a run (empty if nothing did).

    Covers episodes that could not be played, agents that failed or ran out of time, telemetry
    hooks that failed, agent time close to (or, when the limits were not applied, beyond) the
    compute budget, and maps on which the engine spawned other things than the WAD lists.
    """
    out = []
    errors = [r for r in rows if r["error"]]
    if errors:
        out.append(f"{len(errors)} of {len(rows)} episodes could not be played or are void, and count as score 0; "
                   f"first: {_first(errors, 'error')}")
    failed = [r for r in rows if r["agent_error"]]
    if failed:
        late = sum(1 for r in failed if is_timeout(r))
        out.append(f"the agent failed in {len(failed)} of {len(rows)} episodes and stood still from then on "
                   f"({late} stopped by a time limit, {len(failed) - late} by an error or an invalid action); "
                   f"first: {_first(failed, 'agent_error')}")
    hooks = [r for r in rows if r["hook_error"]]
    if hooks:
        out.append(f"a telemetry hook of the policy failed in {len(hooks)} of {len(rows)} episodes and was not called "
                   f"again there (scores are not affected; the odometry or exit-belief columns of those episodes are "
                   f"incomplete); first: {_first(hooks, 'hook_error')}")
    scale = time_scale if time_limits else 1.0
    clean = [r for r in rows if not r["error"] and not r["agent_error"] and r["steps"] > 0]   # the agent acted on every step
    if clean:
        use, worst = max(((budget_use(r, scale), r) for r in clean), key=lambda pair: pair[0])
        over = sum(1 for r in clean if budget_use(r, scale) > 1.0 or r["act_ms_max"] > 1000.0 * FIRST_ACT_TIMEOUT_S)
        if over and not time_limits:
            out.append(f"the time limits were not applied, and in {over} of {len(rows)} episodes act() took longer than "
                       f"they allow (the map's compute budget of {AGENT_MS_PER_STEP:g} ms x decision steps in the tick cap, "
                       f"or {FIRST_ACT_TIMEOUT_S:g} s for one call): the evaluation would have stopped the agent there")
        elif use > HEADROOM:
            out.append(f"little headroom: act() used {100.0 * use:.0f}% of the map's compute budget on {worst['map_id']} "
                       f"seed {worst['seed']} (mean {worst['act_ms_mean']:.1f} ms"
                       + ("" if scale == 1.0 else f", counted x {scale:g}") + f"; the budget is {AGENT_MS_PER_STEP:g} ms "
                       "per decision step of the tick cap)")
    if any(r["census_ok"] == 0 for r in rows):
        out.append("on some maps the engine spawned different things than the WAD lists (see census_ok)")
    return out


def evaluate(policy, suite, tag: str, *, workers: int = 3, seeds=(0,), tiers=None, limit: int | None = None,
             out=RESULTS_CSV, episodes_dir=EPISODES_DIR, maps_dir=MAPS_DIR, render_buffers: bool = True,
             time_limits: bool = True, time_scale: float = 1.0, episode_timeout_s: float | None = None,
             builder=None, quiet: bool = False) -> list[dict]:
    """Play the suite, append the per-tier rows to `out` and return them (as written).

    time_limits, time_scale   the official limits on agent time, on by default (module docstring).
    episode_timeout_s         seconds before a stuck episode is abandoned (default: see runner.patience_s).
    Unless `quiet`, prints the table and a WARNING for everything `problems` finds.
    """
    suite = _suite_path(suite)
    maps = suite_maps(suite, maps_dir, builder)
    if tiers is not None:
        maps = [m for m in maps if m["tier"] in set(tiers)]
    if limit is not None:   # the first `limit` maps of every tier: a fixed subset for quick checks
        seen: dict[int, int] = {}
        kept = []
        for m in maps:
            seen[m["tier"]] = seen.get(m["tier"], 0) + 1
            if seen[m["tier"]] <= limit:
                kept.append(m)
        maps = kept
    if not maps:
        raise SystemExit("no maps left to play after --tiers/--limit")
    seeds = [int(s) for s in seeds]
    started = time.perf_counter()
    told = {"error": 0, "agent_error": 0, "hook_error": 0}   # failures already shown while the run is going

    def progress(done: int, total: int, row: dict) -> None:
        if quiet:
            return
        for key, what in (("error", "not played"), ("agent_error", "the agent failed"), ("hook_error", "a hook failed")):
            if row[key] and told[key] < 3:   # the first few as they happen: a broken agent shows in the first minute
                told[key] += 1
                print(f"  {what}: {_first([row], key)}", file=sys.stderr, flush=True)
        if done == total or done % max(1, total // 20) == 0:
            print(f"  {done}/{total} episodes, {time.perf_counter() - started:.0f} s", file=sys.stderr, flush=True)

    rows = run_episodes(policy, maps, workers=workers, seeds=seeds, render_buffers=render_buffers,
                        time_limits=time_limits, time_scale=time_scale, episode_timeout_s=episode_timeout_s,
                        progress=progress)
    subset = [f"tiers={','.join(str(t) for t in sorted(set(tiers)))}"] if tiers is not None else []
    subset += [f"limit={limit}"] if limit is not None else []
    info = {
        "stamp": datetime.now().astimezone().isoformat(timespec="seconds"), "tag": tag, "commit": git_commit(),
        "suite": suite.name + (f"[{';'.join(subset)}]" if subset else ""),
        "suite_sha": hashlib.sha1(suite.read_bytes()).hexdigest()[:12],
        "policy": spec_name(policy), "seeds": " ".join(str(s) for s in seeds),
        "time_scale": float(time_scale) if time_limits else None,
    }
    summary = tier_rows(rows, info)
    write_episodes(Path(episodes_dir) / f"{_safe(tag)}.csv", rows)
    append_results(out, summary)
    if not quiet:
        print_rows(summary)
        for line in problems(rows, time_limits, time_scale):
            print(f"WARNING: {line}")
    return summary


def print_rows(rows: list[dict]) -> None:
    """A compact table of results.csv rows (as dicts of numbers or of CSV text).

    The last three columns count episodes: `agent` those in which the agent failed or ran out of
    time, `hook` those with a failed telemetry hook, `err` those that could not be played.
    """
    shown = (("score", 6, 2), ("score_se", 5, 2), ("exit_rate", 6, 3), ("c_prog", 6, 3), ("c_expl", 6, 3),
             ("c_combat", 6, 3), ("death_rate", 6, 3), ("stuck_rate", 6, 3), ("door_reversals", 5, 2),
             ("act_ms_mean", 6, 2))
    names = ("score", "+-", "exit", "prog", "expl", "combat", "dead", "stuck", "rev", "ms")
    counts = (("agent_errors", "agent", 5), ("hook_errors", "hook", 4), ("errors", "err", 4))
    print(f"{'tag':<24} {'tier':>4} {'n':>5} " + " ".join(f"{n:>{w}}" for n, (_, w, _) in zip(names, shown))
          + "".join(f" {name:>{w}}" for _, name, w in counts))
    for r in rows:
        cells = []
        for key, width, precision in shown:
            value = float(r[key]) if r.get(key) not in ("", None) else float("nan")
            cells.append(f"{value:{width}.{precision}f}" if value == value else f"{'-':>{width}}")
        for key, _, width in counts:   # a result logged before the column existed has no count
            cells.append(f"{int(float(r[key])):>{width}}" if r.get(key) not in ("", None) else f"{'-':>{width}}")
        print(f"{str(r['tag'])[:24]:<24} {r['tier']:>4} {int(float(r['episodes'])):>5} " + " ".join(cells))


# ---------------------------------------------------------------------------------- compare
def _latest(results: list[dict], tag: str) -> dict[str, dict]:
    """tier -> row of the evaluation most recently appended under `tag` (rows are in time order)."""
    mine = [r for r in results if r.get("tag") == tag]
    return {r["tier"]: r for r in mine if r["stamp"] == mine[-1]["stamp"]}


def paired_difference(ep_a: dict, ep_b: dict, key: str = "score") -> tuple[float, float, int]:
    """Tier-weighted mean of B - A over the episodes both runs played, its standard error, and their number.

    The weights are those of scorer.suite_score, so with identical episode sets the mean equals
    the difference of the two logged `all` scores. The standard error comes from the spread of
    the per-episode differences inside each tier.
    """
    by_tier: dict[int, list[float]] = {}
    for episode in sorted(set(ep_a) & set(ep_b)):
        by_tier.setdefault(ep_a[episode]["tier"], []).append(ep_b[episode][key] - ep_a[episode][key])
    weighted = [t for t in by_tier if TIER_WEIGHTS.get(t, 0.0) > 0.0]
    total = num = var = 0.0
    for t in (weighted or list(by_tier)):
        d, w = by_tier[t], (TIER_WEIGHTS[t] if weighted else 1.0)
        mean = sum(d) / len(d)
        total += w * len(d)
        num += w * sum(d)
        var += w * w * len(d) * (sum((x - mean) ** 2 for x in d) / (len(d) - 1) if len(d) > 1 else 0.0)
    n = sum(len(d) for d in by_tier.values())
    return (num / total, math.sqrt(var) / total, n) if total else (float("nan"), float("nan"), 0)


def compare(tag_a: str, tag_b: str, out=RESULTS_CSV, episodes_dir=EPISODES_DIR, quiet: bool = False) -> dict:
    """Does B beat A by more than noise?

    Returns {"better", "worse", "delta", "se", "margin", "paired", "n", "other", "notes"}. With
    both per-episode files present the difference is taken episode by episode on the maps and
    seeds the two runs share (a map is hard or easy for both, so this is far tighter than
    comparing two means); otherwise it falls back to the two logged scores and their standard
    errors. `other` holds the same difference under the two alternative weightings, to show
    whether the verdict depends on the guessed weights.
    """
    results = read_results(out)
    a, b = _latest(results, tag_a), _latest(results, tag_b)
    for tag, rows in ((tag_a, a), (tag_b, b)):
        if "all" not in rows:
            raise SystemExit(f"no result logged under tag {tag!r} in {out}")
    ep_a = read_episodes(Path(episodes_dir) / f"{_safe(tag_a)}.csv")
    ep_b = read_episodes(Path(episodes_dir) / f"{_safe(tag_b)}.csv")
    notes = []
    if a["all"].get("suite_sha") != b["all"].get("suite_sha") or a["all"].get("seeds") != b["all"].get("seeds"):
        notes.append("the two results come from different suites or seeds")
    paired = bool(set(ep_a) & set(ep_b))
    if paired:
        delta, se, n = paired_difference(ep_a, ep_b)
        other = {key: paired_difference(ep_a, ep_b, key)[0] for key in SCORES[1:]}
        if n < max(len(ep_a), len(ep_b)):
            notes.append(f"only {n} of {max(len(ep_a), len(ep_b))} episodes are shared by both runs")
    else:
        notes.append("per-episode files not found for both tags: unpaired comparison of the logged scores")
        delta = float(b["all"]["score"]) - float(a["all"]["score"])
        se = math.hypot(float(a["all"]["score_se"] or 0.0), float(b["all"]["score_se"] or 0.0))
        n = int(float(b["all"]["episodes"]))
        other = {key: float(b["all"][key]) - float(a["all"][key]) for key in SCORES[1:]}
    if a["all"].get("time_scale", "") != b["all"].get("time_scale", ""):
        notes.append("the two results were measured under different time limits (see the time_scale column)")
    for tag, row in ((tag_a, a["all"]), (tag_b, b["all"])):
        for key, what in (("errors", "could not be played (counted as score 0)"),
                          ("agent_errors", "had the agent fail or run out of time and stand still")):
            if int(float(row.get(key) or 0)):
                notes.append(f"{tag}: {int(float(row[key]))} of {int(float(row['episodes']))} episodes {what}")
    margin = max(NOISE_POINTS, NOISE_SIGMAS * se)
    verdict = {"better": bool(delta > margin), "worse": bool(delta < -margin), "delta": delta, "se": se,
               "margin": margin, "paired": paired, "n": n, "other": other, "notes": notes}
    if not quiet:
        print_rows([a[t] for t in sorted(a)] + [b[t] for t in sorted(b)])
        for note in notes:
            print(f"note: {note}")
        print(f"{tag_b} - {tag_a}: {delta:+.2f} +- {se:.2f} composite points ({'paired' if paired else 'unpaired'}, "
              f"n = {n}); needs more than {margin:.2f} to count")
        print("with the other weightings: " + ", ".join(f"{k.removeprefix('score_')} {v:+.2f}" for k, v in other.items()))
        if verdict["better"]:
            print(f"PROMOTE: {tag_b} beats {tag_a} by more than noise")
        elif verdict["worse"]:
            print(f"KEEP {tag_a}: {tag_b} is worse by more than noise")
        else:
            print(f"KEEP {tag_a}: the difference is within noise")
    return verdict


def table(out=RESULTS_CSV) -> None:
    """Latest logged result of every tag, best overall score first."""
    results = read_results(out)
    latest = [_latest(results, tag).get("all") for tag in sorted({r["tag"] for r in results})]
    print_rows(sorted((r for r in latest if r), key=lambda r: -float(r["score"] or 0.0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("--policy", help="module:factory, for example training.runner:AgentPolicy")
    ap.add_argument("--kw", action="append", default=[], metavar="KEY=VALUE", help="policy keyword argument (JSON value)")
    ap.add_argument("--suite", help="suite of ASCII grids (training/suites/*.jsonl) or a manifest with WAD paths")
    ap.add_argument("--tag", help="name of this result in results.csv")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", default="0", help="comma-separated engine seeds; every map is played once per seed")
    ap.add_argument("--tiers", help="comma-separated tiers to play (default: all)")
    ap.add_argument("--limit", type=int, help="play only the first N maps of every tier")
    ap.add_argument("--no-time-limits", action="store_true",
                    help="do not apply the official limits on reset() and act() time (for training-only policies)")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiply measured agent time by this before comparing it with the limits "
                         "(2.5 = judge the agent as a machine 2.5 times slower than this one would)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds before a stuck episode is abandoned (default: the map's reset limit and compute "
                         "budget plus 60 s; 300 s with --no-time-limits)")
    ap.add_argument("--strict", action="store_true",
                    help="exit status 2 if the agent failed in any episode or an episode could not be played")
    ap.add_argument("--no-render", action="store_true", help="leave rgb/depth as zeros (policies that ignore pixels)")
    ap.add_argument("--out", default=str(RESULTS_CSV), help="results file (default training/logs/results.csv)")
    ap.add_argument("--episodes-dir", default=str(EPISODES_DIR), help="where per-episode rows are kept")
    ap.add_argument("--maps-dir", default=str(MAPS_DIR), help="where suite WADs are built")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), help="does TAG_B beat TAG_A by more than noise?")
    ap.add_argument("--table", action="store_true", help="print the latest result of every tag")
    args = ap.parse_args(argv)

    if args.table:
        table(args.out)
        return 0
    if args.compare:
        return 0 if compare(args.compare[0], args.compare[1], args.out, args.episodes_dir)["better"] else 1
    if not (args.policy and args.suite and args.tag):
        ap.error("--policy, --suite and --tag are required (or use --compare / --table)")
    kwargs = {}
    for item in args.kw:
        key, _, value = item.partition("=")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    summary = evaluate((args.policy, kwargs) if kwargs else args.policy, args.suite, args.tag, workers=args.workers,
                       seeds=[int(s) for s in args.seeds.split(",")],
                       tiers=[int(t) for t in args.tiers.split(",")] if args.tiers else None, limit=args.limit,
                       out=args.out, episodes_dir=args.episodes_dir, maps_dir=args.maps_dir,
                       render_buffers=not args.no_render, time_limits=not args.no_time_limits,
                       time_scale=args.time_scale, episode_timeout_s=args.timeout)
    overall = summary[-1]   # the `all` row
    return 2 if args.strict and (overall["errors"] or overall["agent_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
