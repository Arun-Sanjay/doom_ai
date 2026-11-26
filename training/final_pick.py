"""The report for the final choice of agent: score on the four held-out suites, compare, decide.

Training-time scaffolding, never imported by the agent.

    python -m training.final_pick score CHECKPOINT_OR_CANDIDATE --tag NAME [--workers 5] [--log]
    python -m training.final_pick report NAME [NAME2 ...]

score
    Plays one agent on the four held-out suites (`SUITES`: file, engine seeds and weight of each)
    through the real agent with the official time limits. CHECKPOINT is turned into a candidate
    under `runs/promote/NAME/` by `training.promote.build_candidate`; a directory that already
    holds a `submission/` (a candidate of `training.promote score`) is played as it is and nothing
    is built. Every suite is played by `training.promote.score_local`, so the agent is loaded and
    timed exactly as for a promotion. Per-episode rows go to `runs/final/NAME/<suite>.csv`, the
    per-tier rows to `runs/final/NAME/results.csv`, and how long it took to
    `runs/final/NAME/score.json`. Nothing is appended to `training/logs/results.csv` unless `--log`
    is given (the rows are then tagged `NAME:<suite>`).
    These suites are held out: a result from here is for the final choice between finished agents,
    not for choosing training settings.

report
    Reads those per-episode rows and prints, for every agent named, one row per agent under each
    cell so that the same column of two agents can be read off directly:
      * per suite and tier: composite (the scorer's default weighting), exit rate, kills over
        monsters, death rate, timeout rate and door reversals per episode, each with its standard
        error. Tier 1 is counted on maps that have an exit; the figure for all tier 1 maps is
        printed below it;
      * the same split into maps with a heavy monster, without one, and with reference monsters
        only, per suite and for the four suites pooled;
      * how many maps fall in each split (a `*` marks a cell with fewer than `THIN_MAPS` maps);
      * the headline (`headline`): suite scores combined with the suite weights, where a suite
        score is the mean over its maps weighted by tier (1, 2, 3, 4) and by `LIGHT_MAP_WEIGHT` for
        a map without a heavy monster; the same number without that factor is printed beside it;
      * for every pair of agents the paired difference on the headline and on every suite, with
        its standard error and a verdict (the two tier 4 lines marked `info` are not part of the
        rule), and the decision.
    The whole text is also written to `runs/final/report_<names>.txt`.

How the numbers are made
    A map of a suite that is played with several seeds enters every statistic once, as the mean
    over its seeds; standard errors come from the spread between maps. Weighted means use the
    linearised (ratio) standard error with the tiers as strata, which for one seed and plain tier
    weights is exactly `training.scorer.suite_score`. `kills` is all kills over all monsters.
    In a row for `all` tiers the composite is tier-weighted and the other columns are plain means
    over maps, as in `training.evaluate`.
    A map is `heavy` when its grid holds the symbol of a hell knight or a cacodemon, `ref only`
    when it has at least one monster and all of them are zombiemen, imps or chaingunners; the
    symbols are read from `training.mapgen.udmf.LEGEND`.
    The paired difference is taken map by map on the seeds both agents played. Its verdict uses
    the noise rule of `training.evaluate.compare`: a difference counts only beyond
    `max(NOISE_POINTS, NOISE_SIGMAS * standard error)`.
    Decision: B replaces A only if B's weighted headline is better by more than noise and B is not
    worse by more than noise on any single suite. With more than two names the first is the agent
    that holds the place, and the others challenge whoever holds it, in the order given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training import evaluate, paths, promote
from training.mapgen.udmf import LEGEND
from training.scorer import TIER_WEIGHTS

FINAL_DIR = paths.ROOT / "runs" / "final"
LIGHT_MAP_WEIGHT = 3.0   # in the headline a map without a heavy monster counts this many times a map with one
THIN_MAPS = 10           # a cell with fewer maps than this is marked `*`: too few to read a difference from
HEAVY_MONSTERS = ("hell knight", "cacodemon")
REFERENCE_MONSTERS = ("zombieman", "imp", "chaingunner")


@dataclass(frozen=True)
class Suite:
    name: str
    file: str                 # relative to the repository root, or absolute
    seeds: tuple[int, ...]    # engine seeds every map is played with
    weight: float             # share of the headline


SUITES = (
    Suite("heldout_v1", "training/suites/heldout_v1.jsonl", (0,), 0.35),
    Suite("heldout_wide_v1", "training/suites/heldout_wide_v1.jsonl", (0,), 0.20),
    Suite("authored_v1", "training/suites/authored_v1.jsonl", (0, 1, 2, 3, 4), 0.20),
    Suite("authored_hard_v1", "training/suites/authored_hard_v1.jsonl", (0, 1, 2, 3, 4), 0.25),
)

NAN = float("nan")
BETTER, WORSE, LEVEL, UNKNOWN = "better by more than noise", "worse by more than noise", "level", "not comparable"
# Per-episode columns the report reads (the others of runner.ROW_FIELDS are not needed).
NUMBERS = ("score", "has_exit", "exit", "dead", "timeout", "kills", "monsters_total", "door_reversals",
           "act_ms_mean", "act_ms_max")


def _chars(names) -> frozenset:
    found = frozenset(c for c, s in LEGEND.items() if s.group == "monster" and s.name in names)
    if len(found) != len(names):
        raise RuntimeError(f"training.mapgen.udmf.LEGEND no longer lists all of {', '.join(names)} as monsters")
    return found


HEAVY_CHARS, REFERENCE_CHARS = _chars(HEAVY_MONSTERS), _chars(REFERENCE_MONSTERS)
MONSTER_CHARS = frozenset(c for c, s in LEGEND.items() if s.group == "monster")
EXIT_CHARS = frozenset(c for c, s in LEGEND.items() if s.kind == "exit")


# --------------------------------------------------------------------------------------- maps
@dataclass(frozen=True)
class MapInfo:
    """What the report needs to know about one map of a suite, all of it read from the grid."""

    suite: str
    map_id: str
    tier: int
    has_exit: bool
    heavy: bool            # a hell knight or a cacodemon is on the map
    reference_only: bool   # at least one monster, and only zombiemen, imps and chaingunners
    monsters: int

    @property
    def counted(self) -> bool:
        """In the headline and in the tier rows: tiers 1-4, and of tier 1 only the maps with an exit."""
        return self.tier in TIER_WEIGHTS and (self.tier != 1 or self.has_exit)


def classify(grid: str) -> dict:
    """{"has_exit", "heavy", "reference_only", "monsters"} of an ASCII grid."""
    cells = [c for c in grid if c in MONSTER_CHARS]
    kinds = set(cells)
    return {"has_exit": any(c in EXIT_CHARS for c in grid), "heavy": bool(kinds & HEAVY_CHARS),
            "reference_only": bool(kinds) and kinds <= REFERENCE_CHARS, "monsters": len(cells)}


def load_maps(suite: Suite) -> dict[str, MapInfo]:
    """map id -> MapInfo for every map of a suite file, in file order."""
    out = {}
    for row in evaluate.load_suite(suite.file):
        if row["grid"] is None:
            raise SystemExit(f"{suite.file}: map {row['id']} has no grid: its monsters and exit cannot be read")
        out[row["id"]] = MapInfo(suite.name, row["id"], row["tier"], **classify(row["grid"]))
    return out


# --------------------------------------------------------------------------------------- rows
@dataclass
class Run:
    """The per-episode rows of one agent: episodes[suite][map id][seed] -> row, and what was odd about them."""

    name: str
    episodes: dict
    notes: list
    facts: dict
    complete: bool


def read_rows(path) -> list[dict]:
    """The rows of a per-episode file with the columns of NUMBERS as floats (NaN when empty)."""
    with open(path, newline="") as f:
        out = []
        for r in csv.DictReader(f):
            row = {"map_id": r["map_id"], "seed": int(float(r["seed"] or 0)), "tier": int(float(r.get("tier") or 0)),
                   "error": r.get("error") or "", "agent_error": r.get("agent_error") or ""}
            for key in NUMBERS:
                row[key] = float(r[key]) if r.get(key) not in (None, "") else NAN
            if row["score"] != row["score"]:   # an episode without a score was not played: it counts as 0
                row["score"] = 0.0
            out.append(row)
    return out


def load_run(name: str, suites=SUITES, maps: dict | None = None, runs_dir=None) -> Run:
    """Read `runs_dir/NAME/<suite>.csv` for every suite and check the rows against the suite files."""
    home = Path(runs_dir or FINAL_DIR) / promote._safe(name)
    maps = maps if maps is not None else {s.name: load_maps(s) for s in suites}
    episodes, notes, complete = {}, [], True
    for suite in suites:
        path, table = home / f"{suite.name}.csv", {}
        if not path.is_file():
            notes.append(f"{suite.name}: no rows ({path} is missing): no headline for this agent")
            complete = False
            continue
        unknown = foreign = failed = stopped = 0
        for row in read_rows(path):
            info = maps[suite.name].get(row["map_id"])
            if info is None:
                unknown += 1
            elif row["seed"] not in suite.seeds:
                foreign += 1
            else:
                table.setdefault(row["map_id"], {})[row["seed"]] = row
                failed += bool(row["error"])
                stopped += bool(row["agent_error"])
        episodes[suite.name] = table
        played = sum(len(seeds) for seeds in table.values())
        expected = len(maps[suite.name]) * len(suite.seeds)
        if played < expected:
            complete = False
            notes.append(f"{suite.name}: {played} of {expected} episodes (maps x seeds) are there: the suite is incomplete")
        if unknown:
            notes.append(f"{suite.name}: {unknown} rows of maps that are not in {suite.file} were left out")
        if foreign:
            notes.append(f"{suite.name}: {foreign} rows with other seeds than {list(suite.seeds)} were left out")
        if failed:
            notes.append(f"{suite.name}: {failed} episodes could not be played and count as score 0")
        if stopped:
            notes.append(f"{suite.name}: in {stopped} episodes the agent failed or ran out of time and stood still")
    try:
        facts = json.loads((home / "score.json").read_text())
    except (OSError, ValueError):
        facts = {}
    if facts.get("partial"):
        complete = False
        notes.append("scored with --limit or --suites: not a full run")
    return Run(name, episodes, notes, facts if isinstance(facts, dict) else {}, complete)


# ---------------------------------------------------------------------------------- statistics
def _mean(values) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else NAN


def map_means(seeds: dict) -> dict:
    """One map's rows (seed -> row) -> the mean of every number over its seeds (NaN values left out)."""
    return {key: _mean(row[key] for row in seeds.values()) for key in NUMBERS}


def weighted_mean(values, weights=None, strata=None) -> tuple[float, float, int]:
    """(mean, standard error, n) of sum(w * x) / sum(w) over the entries whose value is known.

    The standard error is the linearised one of a ratio of sums, with the spread taken inside each
    stratum (a stratum with one entry adds none). With equal weights and one stratum it is the
    ordinary standard error of the mean; with tier weights and tiers as strata it is the one of
    `training.scorer.suite_score`.
    """
    values = list(values)
    weights = [1.0] * len(values) if weights is None else list(weights)
    strata = [0] * len(values) if strata is None else list(strata)
    kept = [(x, w, s) for x, w, s in zip(values, weights, strata) if x == x and w == w and w > 0.0]
    total = sum(w for _, w, _ in kept)
    if not kept or total <= 0.0:
        return NAN, NAN, 0
    mean = sum(w * x for x, w, _ in kept) / total
    groups: dict = {}
    for x, w, s in kept:
        groups.setdefault(s, []).append(w * (x - mean) / total)
    var = 0.0
    for e in groups.values():
        if len(e) > 1:
            centre = sum(e) / len(e)
            var += len(e) / (len(e) - 1) * sum((v - centre) ** 2 for v in e)
    return mean, math.sqrt(var), len(kept)


def cell_stats(means: dict, infos: dict, keys, tier_weighted: bool = False) -> dict:
    """The statistics of one cell: `keys` are the maps in it, `means[key]` their seed-averaged numbers.

    Returns {"maps", "score", "exit", "kills", "dead", "timeout", "door_reversals"}, the last six
    as (value, standard error, maps it is over). The exit rate is over the maps that have an exit,
    `kills` is kills over monsters pooled over the maps that have monsters, and with
    `tier_weighted` the composite is weighted by tier (the other columns never are).
    """
    keys = [k for k in keys if k in means]
    column = lambda name: [means[k][name] for k in keys]  # noqa: E731
    if tier_weighted:
        score = weighted_mean(column("score"), [TIER_WEIGHTS.get(infos[k].tier, 0.0) for k in keys],
                              [infos[k].tier for k in keys])
    else:
        score = weighted_mean(column("score"))
    with_exit = [k for k in keys if infos[k].has_exit]
    fought = [k for k in keys if means[k]["monsters_total"] > 0.0 and means[k]["kills"] == means[k]["kills"]]
    return {
        "maps": len(keys), "score": score,
        "exit": weighted_mean([means[k]["exit"] for k in with_exit]),
        "kills": weighted_mean([means[k]["kills"] / means[k]["monsters_total"] for k in fought],
                               [means[k]["monsters_total"] for k in fought]),
        "dead": weighted_mean(column("dead")), "timeout": weighted_mean(column("timeout")),
        "door_reversals": weighted_mean(column("door_reversals")),
    }


def map_weight(info: MapInfo, light_weight: float = LIGHT_MAP_WEIGHT) -> float:
    """Weight of a map inside its suite: its tier weight, times `light_weight` when it has no heavy monster."""
    return TIER_WEIGHTS[info.tier] * (1.0 if info.heavy else light_weight) if info.counted else 0.0


def suite_score(values: dict, infos: dict, light_weight: float = LIGHT_MAP_WEIGHT) -> tuple[float, float, int]:
    """(score, standard error, maps) of one suite from per-map values (map id -> composite, or a difference)."""
    keys = [k for k in values if k in infos and infos[k].counted]
    return weighted_mean([values[k] for k in keys], [map_weight(infos[k], light_weight) for k in keys],
                         [infos[k].tier for k in keys])


def headline(per_suite: dict, suites=SUITES) -> tuple[float, float]:
    """(headline, standard error) from {suite name: (score, standard error, ...)}; NaN when a suite is missing."""
    total = var = 0.0
    for suite in suites:
        score, se = (per_suite.get(suite.name) or (NAN, NAN))[:2]
        total += suite.weight * score
        var += (suite.weight * se) ** 2
    weight = sum(s.weight for s in suites)
    return total / weight, math.sqrt(var) / weight


def agent_scores(run: Run, maps: dict, suites=SUITES, light_weight: float = LIGHT_MAP_WEIGHT) -> dict:
    """{"suites": {suite name: (score, se, maps)}, "headline": (value, se)} of one agent."""
    per_suite = {}
    for suite in suites:
        if suite.name in run.episodes:
            values = {k: map_means(seeds)["score"] for k, seeds in run.episodes[suite.name].items()}
            per_suite[suite.name] = suite_score(values, maps[suite.name], light_weight)
    return {"suites": per_suite, "headline": headline(per_suite, suites)}


# ----------------------------------------------------------------------------------- comparison
def verdict(delta: float, se: float) -> dict:
    """{"delta", "se", "margin", "verdict"} by the noise rule of `training.evaluate.compare`."""
    if delta != delta or se != se:
        return {"delta": delta, "se": se, "margin": NAN, "verdict": UNKNOWN}
    margin = max(evaluate.NOISE_POINTS, evaluate.NOISE_SIGMAS * se)
    return {"delta": delta, "se": se, "margin": margin,
            "verdict": BETTER if delta > margin else WORSE if delta < -margin else LEVEL}


def paired_differences(run_a: Run, run_b: Run, suite: str) -> dict:
    """map id -> mean of (B - A) composite over the seeds of that map which both agents played."""
    a, b = run_a.episodes.get(suite, {}), run_b.episodes.get(suite, {})
    out = {}
    for key in a.keys() & b.keys():
        shared = a[key].keys() & b[key].keys()
        if shared:
            out[key] = sum(b[key][s]["score"] - a[key][s]["score"] for s in shared) / len(shared)
    return out


def paired(run_a: Run, run_b: Run, maps: dict, suites=SUITES, light_weight: float = LIGHT_MAP_WEIGHT) -> dict:
    """B minus A on the same maps and seeds: {"headline": verdict, "suites": {name: verdict}, "maps": {name: n}}."""
    per_suite, counts = {}, {}
    for suite in suites:
        differences = paired_differences(run_a, run_b, suite.name)
        delta, se, n = suite_score(differences, maps[suite.name], light_weight)
        per_suite[suite.name], counts[suite.name] = (delta, se), n
    return {"headline": verdict(*headline(per_suite, suites)),
            "suites": {name: verdict(*pair) for name, pair in per_suite.items()}, "maps": counts}


def decide(head: dict, per_suite: dict, complete: bool = True) -> tuple[bool, str]:
    """(B replaces A?, why) from the paired verdicts on the headline and on every suite."""
    if not complete:
        return False, "one of the two was not scored on every map and seed of the four suites"
    if head["verdict"] == UNKNOWN or any(v["verdict"] == UNKNOWN for v in per_suite.values()):
        return False, "the two could not be compared on every suite"
    if head["verdict"] != BETTER:
        return False, (f"B's weighted headline is {head['verdict']} "
                       f"({head['delta']:+.2f}, needs more than {head['margin']:.2f})")
    worse = [f"{name} ({v['delta']:+.2f}, margin {v['margin']:.2f})" for name, v in per_suite.items() if v["verdict"] == WORSE]
    if worse:
        return False, (f"B's weighted headline is better by more than noise ({head['delta']:+.2f}), "
                       f"but B is worse by more than noise on {', '.join(worse)}")
    return True, (f"B's weighted headline is better by more than noise ({head['delta']:+.2f}, margin {head['margin']:.2f}) "
                  "and B is not worse by more than noise on any suite")


# --------------------------------------------------------------------------------------- report
SPLITS = (("heavy", lambda i: i.heavy), ("no heavy", lambda i: not i.heavy), ("ref only", lambda i: i.reference_only))
COLUMNS = (("score", "score", 2), ("exit", "exit", 3), ("kills", "kills", 3), ("dead", "dead", 3),
           ("timeout", "tmo", 3), ("door_reversals", "rev", 2))


def _pm(pair, decimals: int) -> str:
    value, se = pair[0], pair[1]
    if value != value:
        return f"{'-':>7} {'':<7}"
    return f"{value:>7.{decimals}f} {'+-' + format(se, f'.{decimals}f'):<7}"


def _table(title: str, cells: list, runs: list, width: int) -> list[str]:
    """Lines of one table. `cells` are (suite, tier, split, maps in the cell, [stats per agent])."""
    lines = ["", title,
             f"{'suite':<17}{'tier':<5}{'split':<10}{'maps':>5}  {'agent':<{width}} "
             + " ".join(f"{head:>7} {'':<7}" for _, head, _ in COLUMNS)]
    for suite, tier, split, count, stats in cells:
        label = f"{suite:<17}{tier:<5}{split:<10}{count:>5}{'*' if count < THIN_MAPS else ' '} "
        for run, stat in zip(runs, stats):
            lines.append(label + f"{run.name[:width]:<{width}} " + " ".join(_pm(stat[key], d) for key, _, d in COLUMNS))
            label = " " * len(label)
    return lines


def build_report(names, suites=SUITES, runs_dir=None, light_weight: float = LIGHT_MAP_WEIGHT) -> tuple[list[str], dict]:
    """(lines of the report, the numbers behind it) for the agents scored under `names`."""
    names = list(names)
    if len(set(names)) != len(names):
        raise SystemExit("the same name was given twice")
    maps = {s.name: load_maps(s) for s in suites}
    runs = [load_run(name, suites, maps, runs_dir) for name in names]
    width = max(len("agent"), min(28, max(len(n) for n in names)))
    means = [{(s, k): map_means(seeds) for s, table in run.episodes.items() for k, seeds in table.items()} for run in runs]
    infos = {(s, k): info for s, table in maps.items() for k, info in table.items()}
    tiers = sorted({i.tier for i in infos.values() if i.tier in TIER_WEIGHTS})

    def cell(suite, tier, split, keys, tier_weighted=False):
        keys = list(keys)
        return (suite, str(tier), split, len(keys), [cell_stats(m, infos, keys, tier_weighted) for m in means])

    def select(suite=None, tier=None, test=None, counted=True):
        return [k for k, i in infos.items() if (suite is None or k[0] == suite) and (tier is None or i.tier == tier)
                and (not counted or i.counted) and (test is None or test(i))]

    lines = [f"FINAL PICK REPORT  {datetime.now().astimezone().isoformat(timespec='seconds')}",
             "agents: " + ", ".join(names),
             "suites (weight, maps x seeds): " + ", ".join(f"{s.name} ({s.weight:g}, {len(maps[s.name])} x {len(s.seeds)})"
                                                          for s in suites),
             f"headline = suite weights x suite score; suite score = mean over maps weighted by tier "
             f"({', '.join(f'{TIER_WEIGHTS[t]:g}' for t in sorted(TIER_WEIGHTS))}) x {light_weight:g} for a map without a "
             "heavy monster; tier 1 maps without an exit are left out",
             f"heavy = {' or '.join(HEAVY_MONSTERS)} on the map; ref only = at least one monster, all of them "
             f"{', '.join(REFERENCE_MONSTERS)}",
             "every value is followed by its standard error (+-); a multi-seed map counts once, as the mean over its seeds",
             f"a difference counts beyond max({evaluate.NOISE_POINTS:g} point, {evaluate.NOISE_SIGMAS:g} standard errors); "
             f"* = fewer than {THIN_MAPS} maps in the cell"]
    for run in runs:
        facts = run.facts
        if facts:
            lines.append(f"{run.name}: scored {facts.get('stamp', '?')} in {float(facts.get('seconds', NAN)):.0f} s with "
                         f"{facts.get('workers', '?')} workers, {facts.get('episodes', '?')} episodes, candidate "
                         f"{facts.get('candidate', '?')}, policy sha256 {str(facts.get('policy_sha256', '?'))[:12]}")

    # ---- headline
    weighted = [agent_scores(run, maps, suites, light_weight) for run in runs]
    plain = [agent_scores(run, maps, suites, 1.0) for run in runs]
    lines += ["", f"HEADLINE (weighted: light maps x {light_weight:g}; unweighted: every map of a tier alike)",
              f"{'':<24}{'weight':>6}  {'agent':<{width}} {'weighted':>8} {'':<7} {'unweighted':>10} {'':<7} {'maps':>5}"]
    for label, weight in [("headline", sum(s.weight for s in suites))] + [(s.name, s.weight) for s in suites]:
        head = f"{label:<24}{weight:>6.2f}  "
        for run, w, p in zip(runs, weighted, plain):
            if label == "headline":
                a, b, count = w["headline"], p["headline"], sum(v[2] for v in w["suites"].values())
            else:
                a, b = w["suites"].get(label, (NAN, NAN, 0)), p["suites"].get(label, (NAN, NAN, 0))
                count = a[2]
            lines.append(head + f"{run.name[:width]:<{width}} {_pm(a, 2):>16} {_pm(b, 2):>18} {count:>5}")
            head = " " * len(head)

    # ---- per suite and tier
    cells = []
    for s in suites:
        for tier in tiers:
            cells.append(cell(s.name, tier, "exit maps" if tier == 1 else "all", select(s.name, tier)))
            if tier == 1:
                cells.append(cell(s.name, tier, "all maps", select(s.name, tier, counted=False)))
        cells.append(cell(s.name, "all", "counted", select(s.name), tier_weighted=True))
    lines += _table("PER SUITE AND TIER (tier 1 on maps with an exit; `all maps` adds the tier 1 maps without one; "
                    "`all` rows: tier-weighted composite, the counted maps)", cells, runs, width)

    # ---- heavy / no heavy / reference only
    cells = []
    for label, suite in [(s.name, s.name) for s in suites] + [("ALL SUITES", None)]:
        for tier in [t for t in reversed(tiers)] + ["all"]:
            for split, test in SPLITS:
                keys = select(suite, None if tier == "all" else tier, test)
                if keys:
                    cells.append(cell(label, tier, split, keys, tier_weighted=tier == "all"))
    lines += _table("SPLIT BY MONSTERS (counted maps; ALL SUITES pools the maps of the four suites, each map once)",
                    cells, runs, width)

    # ---- map counts
    kinds = (("maps", lambda i: True), ("exit", lambda i: i.has_exit), ("no exit", lambda i: not i.has_exit),
             ("counted", lambda i: i.counted), ("heavy", lambda i: i.counted and i.heavy),
             ("no heavy", lambda i: i.counted and not i.heavy), ("ref only", lambda i: i.counted and i.reference_only),
             ("no monst", lambda i: i.counted and i.monsters == 0))
    lines += ["", "MAPS IN EACH SPLIT (heavy, no heavy, ref only and no monst are of the counted maps)",
              f"{'suite':<17}{'tier':<5}" + "".join(f"{name:>9}" for name, _ in kinds)]
    counts = {}
    for label, suite in [(s.name, s.name) for s in suites] + [("ALL SUITES", None)]:
        for tier in tiers + ["all"]:
            keys = select(suite, None if tier == "all" else tier, counted=False)
            row = [sum(1 for k in keys if test(infos[k])) for _, test in kinds]
            counts[(label, str(tier))] = dict(zip((name for name, _ in kinds), row))
            lines.append(f"{label:<17}{str(tier):<5}" + "".join(
                f"{n:>8}{'*' if 0 < n < THIN_MAPS and name not in ('no exit', 'no monst') else ' '}"
                for (name, _), n in zip(kinds, row)))

    # ---- paired comparisons and the decision
    pairs, decisions = {}, []
    for j, run_b in enumerate(runs):
        for run_a in runs[:j]:
            result = paired(run_a, run_b, maps, suites, light_weight)
            flat = paired(run_a, run_b, maps, suites, 1.0)
            pairs[(run_a.name, run_b.name)] = result
            lines += ["", f"PAIRED  B = {run_b.name}  minus  A = {run_a.name}  (same maps, same seeds; composite points)",
                      f"{'':<34}{'maps':>5} {'B - A':>8} {'se':>6} {'margin':>7}  verdict"]
            entries = [("headline (weighted)", sum(result["maps"].values()), result["headline"]),
                       ("headline (unweighted)", sum(flat["maps"].values()), flat["headline"])]
            entries += [(f"{s.name} (weighted)", result["maps"][s.name], result["suites"][s.name]) for s in suites]
            entries += [(f"{s.name} (unweighted)", flat["maps"][s.name], flat["suites"][s.name]) for s in suites]
            for label, test in (("tier 4 heavy, all suites (info)", lambda i: i.heavy),
                                ("tier 4 no heavy, all suites (info)", lambda i: not i.heavy)):
                differences = {(s.name, k): d for s in suites for k, d in paired_differences(run_a, run_b, s.name).items()}
                delta, se, n = weighted_mean([d for k, d in differences.items() if infos[k].tier == 4 and test(infos[k])])
                entries.append((label, n, verdict(delta, se)))
            for label, n, v in entries:
                lines.append(f"{label:<34}{n:>5} {v['delta']:>+8.2f} {v['se']:>6.2f} {v['margin']:>7.2f}  {v['verdict']}")
    if len(runs) > 1:
        holder = runs[0]
        lines.append("")
        for challenger in runs[1:]:
            result = pairs.get((holder.name, challenger.name)) or paired(holder, challenger, maps, suites, light_weight)
            replace, why = decide(result["headline"], result["suites"], holder.complete and challenger.complete)
            decisions.append({"a": holder.name, "b": challenger.name, "replace": replace, "why": why})
            lines.append(f"DECISION (A = {holder.name}, B = {challenger.name}): "
                         + (f"REPLACE {holder.name} WITH {challenger.name}" if replace else f"KEEP {holder.name}") + f": {why}")
            holder = challenger if replace else holder
        lines.append(f"FINAL PICK: {holder.name}")

    notes = [f"{run.name}: {note}" for run in runs for note in run.notes]
    watched = ("counted", "heavy", "no heavy", "ref only")
    for label in [s.name for s in suites] + ["ALL SUITES"]:
        thin = [f"tier {tier}: " + ", ".join(f"{row[name]} {name}" for name in watched if 0 < row[name] < THIN_MAPS)
                for (where, tier), row in counts.items()
                if where == label and tier != "all" and any(0 < row[name] < THIN_MAPS for name in watched)]
        if thin:
            notes.append(f"{label}: cells with fewer than {THIN_MAPS} maps, too few to read a difference from: " + "; ".join(thin))
    notes.append("maps with reference monsters only, the four suites together: "
                 + ", ".join(f"tier {t}: {counts[('ALL SUITES', str(t))]['ref only']}" for t in tiers))
    lines += [""] + [f"note: {note}" for note in notes]
    return lines, {"runs": runs, "weighted": weighted, "plain": plain, "pairs": pairs, "decisions": decisions,
                   "counts": counts, "maps": maps}


def report(names, suites=SUITES, runs_dir=None, out=None, echo=print) -> dict:
    """Print the report and write it to `out` (default `runs_dir/report_<names>.txt`). Returns its numbers."""
    lines, numbers = build_report(names, suites, runs_dir)
    target = Path(out) if out else Path(runs_dir or FINAL_DIR) / f"report_{'_vs_'.join(promote._safe(n) for n in names)}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    for line in lines:
        echo(line)
    echo(f"written to {target}")
    return dict(numbers, file=target)


# ---------------------------------------------------------------------------------------- score
def candidate(source, name: str, check: str = "require", echo=print) -> Path:
    """The candidate directory to play: `source` itself when it is a directory, else one built from the file."""
    source = Path(source)
    if source.is_dir():
        promote.submission_dir(source)     # stops here when there is no agent in it
        return source
    promote.build_candidate(source, name, check, echo=echo)
    return promote.CANDIDATES / promote._safe(name)


def score(source, name: str, *, workers: int = 5, suites=SUITES, runs_dir=None, log: bool = False,
          check: str = "require", limit: int | None = None, echo=print) -> dict:
    """Play the agent of `source` on every suite and write the rows under `runs_dir/NAME/`. Returns the facts written.

    source  a checkpoint or exported weights file (a candidate is built from it under
            `runs/promote/NAME/`), or a directory that holds a `submission/` (played as it is).
    log     also append the per-tier rows to `training/logs/results.csv`, tagged `NAME:<suite>`.
    limit   play only the first N maps of every tier (a trial of the tool; the report says so).
    """
    home = candidate(source, name, check, echo)
    submission = promote.submission_dir(home)
    target = Path(runs_dir or FINAL_DIR).resolve() / promote._safe(name)
    target.mkdir(parents=True, exist_ok=True)
    echo("note: these suites are held out: this result is for the final choice between finished agents, "
         "not for choosing training settings")
    started, timings, episodes = time.perf_counter(), {}, 0
    for suite in suites:
        echo(f"\n== {suite.name}: seeds {list(suite.seeds)}, {workers} workers")
        began = time.perf_counter()
        rows = promote.score_local(home, suite.name, suite.file, workers=workers, seeds=suite.seeds, limit=limit,
                                   out=target / "results.csv", episodes_dir=target, echo=echo)
        if log:
            evaluate.append_results(evaluate.RESULTS_CSV, [{**row, "tag": f"{name}:{suite.name}"} for row in rows])
        count = int(float(rows[-1]["episodes"]))
        episodes += count
        timings[suite.name] = {"seconds": round(time.perf_counter() - began, 1), "episodes": count,
                               "seeds": list(suite.seeds)}
        echo(f"{suite.name}: {count} episodes in {timings[suite.name]['seconds']:.0f} s")
    facts = {"name": name, "stamp": datetime.now().astimezone().isoformat(timespec="seconds"),
             "candidate": promote._relative(home), "policy_sha256": promote.sha256(submission / promote.POLICY_NAME),
             "workers": int(workers), "seconds": round(time.perf_counter() - started, 1), "episodes": episodes,
             "partial": limit is not None or tuple(s.name for s in suites) != tuple(s.name for s in SUITES),
             "logged": bool(log), "suites": timings}
    (target / "score.json").write_text(json.dumps(facts, indent=1) + "\n")
    echo(f"\nscored {name}: {episodes} episodes in {facts['seconds']:.0f} s with {workers} workers; rows in "
         f"{promote._relative(target)}")
    echo(f"next: python -m training.final_pick report {name}")
    return facts


# ------------------------------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.final_pick", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score one agent on the four held-out suites through the real agent")
    s.add_argument("source", help="a checkpoint or exported weights file, or a candidate directory (one with a submission/)")
    s.add_argument("--tag", required=True, help="name of the result: rows go to runs/final/NAME/")
    s.add_argument("--workers", type=int, default=5, help="worker processes (default 5)")
    s.add_argument("--log", action="store_true", help="also append the per-tier rows to training/logs/results.csv")
    s.add_argument("--suites", help="comma-separated subset of the suites (a trial; the report will call it incomplete)")
    s.add_argument("--limit", type=int, help="play only the first N maps of every tier (a trial)")
    s.add_argument("--allow-unverified", action="store_true", help="accept a checkpoint that carries no provenance record")
    s.add_argument("--allow-mismatch", action="store_true",
                   help="accept a checkpoint whose record the current code no longer reproduces")
    s.add_argument("--runs-dir", help="where the rows are kept (default runs/final)")
    r = sub.add_parser("report", help="the report and the decision, from the rows of `score`")
    r.add_argument("names", nargs="+", help="tags given to `score`; the first is the agent that holds the place")
    r.add_argument("--runs-dir", help="where the rows are kept (default runs/final)")
    r.add_argument("--out", help="file the report is written to (default runs/final/report_<names>.txt)")
    args = ap.parse_args(argv)

    if args.command == "report":
        report(args.names, runs_dir=args.runs_dir, out=args.out)
        return 0
    suites = SUITES
    if args.suites:
        wanted = [n.strip() for n in args.suites.split(",") if n.strip()]
        unknown = sorted(set(wanted) - {s.name for s in SUITES})
        if unknown:
            ap.error(f"unknown suite: {', '.join(unknown)} (the suites are {', '.join(s.name for s in SUITES)})")
        suites = tuple(s for s in SUITES if s.name in wanted)
    mode = "warn" if args.allow_mismatch else "strict" if args.allow_unverified else "require"
    try:
        home = candidate(args.source, args.tag, mode)
    except Exception as exc:  # noqa: BLE001 - the export's refusal is the answer; show it without a traceback
        if type(exc).__name__ not in ("CheckpointMismatch", "ValueError", "RuntimeError", "FileNotFoundError"):
            raise
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    score(home, args.tag, workers=args.workers, suites=suites, runs_dir=args.runs_dir, log=args.log, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
