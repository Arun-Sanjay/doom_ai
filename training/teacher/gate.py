"""Training-time scaffolding. Never imported by the agent in submission/.

Gate evaluation of the navigation teacher.

    python -m training.teacher.gate --tag clean                      # tiers 2-4 of dev_v1, clean exit observations
    python -m training.teacher.gate --tag noisy --exit noisy         # the training condition
    python -m training.teacher.gate --tag turned --start random --limit 5

The teacher does not fight, so it is measured on monster-free rebuilds of a frozen suite: the
stored grids with every monster symbol replaced by floor, compiled into `data/teacher/`. Episodes
are played by `training.runner.run_episodes` and scored by the episode scorer. The assumed time
budget handed to the teacher is the true cap of each map (or, with `--budget N`, the same N
decision steps on every map); it is set by the recorder below from the task description, so the
policy itself never asks the environment for it.

Outputs: one row per tier in `training/logs/teacher_gate.csv`, the failed episodes with their
cause in `training/logs/teacher_failures.json`, per-episode rows in `runs/teacher/<tag>.csv` and,
for every failed episode, a text view of the teacher's final map in `runs/teacher/<tag>/`.

`--start random` rebuilds each map with a random start heading and an offset of up to 16 units
inside the start cell, so the walls no longer line up with the cells of the agent's start frame.

`--start keep` plays every map with the start pose it was stored with. A training manifest shifts
or turns the start on part of its maps (581 of the 1500 maps of `data/maps/train`, 454 of the 1200
in tiers 0, 2, 3 and 4); the teacher's 64-unit cells do not fit those maps and it gives no
controller labels there (`Teacher.grid_fits`, `start_on_grid` below). With mixed starts the table
therefore gets two more groups of rows, marked `keep:on-grid` and `keep:off-grid` in the `start`
column: the first describes the episodes the student is given labels on, the second the ones it
learns from reward alone.

    python -m training.teacher.gate --tag train_keep --suite data/maps/train/manifest.jsonl --tiers 0,2,3,4 --start keep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from training import paths
from doomnet import spec
from training.mapgen import udmf
from training.runner import run_episodes

LOG_CSV = paths.ROOT / "training" / "logs" / "teacher_gate.csv"
FAILURES_JSON = paths.ROOT / "training" / "logs" / "teacher_failures.json"
MAPS_DIR = paths.ROOT / "data" / "teacher"
RUNS_DIR = paths.ROOT / "runs" / "teacher"
POLICY = "training.teacher.policy:TeacherPolicy"
RECORDER = "training.teacher.gate:GateRecorder"
MODES = ("run", "slow", "line", "door", "blocked", "look", "arrive", "idle")     # steering modes of core.Teacher
# Gate of the build spec: (minimum exit rate per tier, minimum sector coverage, door reversals allowed).
GATE_EXIT = {2: 0.90, 3: 0.90, 4: 0.75}
GATE_SECTORS = 0.85
EXTRA_FIELDS = ("budget_steps", "exit_step", "leave_step", "leave_travel", "label_ms", "front_ms", "teacher_errors",
                "teacher_error", "searches", "replans", "final_subgoal", "final_mode", "locks_given_up", "slipped", "use_masked",
                "grid_fits", "labelled", "unstuck", "cause")
TABLE_FIELDS = ("stamp", "tag", "suite", "exit_source", "start", "budget", "seeds", "tier", "episodes", "exit_rate", "exit_gate",
                "sectors_frac", "doors_frac", "locked_frac", "keys_frac", "door_reversals", "exit_seen_timeout",
                "stuck_rate", "stuck_at_door_rate", "steps_to_exit", "steps_frac_of_cap", "label_ms", "front_ms",
                "odom_err_max", "teacher_errors", "agent_errors", "use_masked", "slipped_frac", "policy_kwargs", "passes",
                "labelled_frac")      # share of steps whose label carried controller labels (blank in rows logged before it existed)


# ----------------------------------------------------------------------------------- maps
def strip_monsters(grid: str) -> str:
    """The grid with every monster symbol replaced by floor."""
    return "".join("." if ch in udmf.MONSTER_CHARS else ch for ch in grid)


def start_on_grid(options) -> bool:
    """True when build options leave the player start at a cell centre, facing along an axis.

    Only then do the walls lie on the 64-unit cells of the agent's start frame, which is what the
    teacher plans on. A training loop that knows the options of its maps can use this to decide
    which maps get teacher labels; the teacher makes the same decision by itself from its first
    observation (`Teacher.grid_fits`), and the gate table reports how often the two agree.
    """
    options = options or {}
    offset = options.get("start_offset") or (0, 0)
    quarter = float(options.get("start_angle") or 0) % 90.0
    return all(abs(float(v)) < 1e-6 for v in offset) and min(quarter, 90.0 - quarter) < 1e-6


def bare_maps(suite, tiers=None, limit: int | None = None, start: str = "aligned", out_dir=MAPS_DIR,
              monsters: bool = False) -> list[dict]:
    """Monster-free WADs of a suite file, as map entries for the runner.

    start="aligned" starts every map at a cell centre facing east (the reference dialect),
    start="random" gives every map a random start heading and an offset inside its start cell
    (fixed by the map's name, so reruns play the same maps), start="offset" only the offset,
    start="quarter" only a heading of 90, 180 or 270 degrees, start="keep" leaves the start the
    row was stored with (a training manifest randomises about half of its maps).
    monsters=True keeps the monsters (not the gate: it shows what navigation alone is worth there).
    """
    suite = Path(suite)
    if not suite.is_file() and (paths.ROOT / suite).is_file():
        suite = paths.ROOT / suite
    out = Path(out_dir) / f"{suite.stem}_{'full' if monsters else 'bare'}{'' if start == 'aligned' else '_' + start}"
    maps, per_tier = [], {}
    for line in suite.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tier = int(row.get("tier", 0))
        if tiers is not None and tier not in tiers:
            continue
        per_tier[tier] = per_tier.get(tier, 0) + 1
        if limit is not None and per_tier[tier] > limit:
            continue
        options = {k: v for k, v in (row.get("options") or {}).items() if monsters or k not in ("ambush", "thing_angles")}
        if start != "keep":
            options.pop("start_angle", None)
            options.pop("start_offset", None)
        rng = random.Random(f"teacher-start/{row['name']}")
        if start == "random":
            options["start_angle"] = rng.randrange(360)
            options["start_offset"] = [rng.randint(-16, 16), rng.randint(-16, 16)]
        elif start == "offset":                          # still facing east, off the cell centre
            options["start_offset"] = [rng.randint(-16, 16), rng.randint(-16, 16)]
        elif start == "quarter":                         # at the cell centre, facing north, west or south
            options["start_angle"] = rng.choice((90, 180, 270))
        wad = udmf.write_wad(row["grid"] if monsters else strip_monsters(row["grid"]), out / f"{row['name']}.wad", **options)
        maps.append({"id": row["name"], "wad": str(wad), "tier": tier, "episode_ticks": int(row["episode_ticks"]),
                     "on_grid": start_on_grid(options)})
    return maps


# ------------------------------------------------------------------------------- recorder
class GateRecorder:
    """Runner hook: sets the teacher's time budget (the true cap by default) and reports its own statistics.

    Adds the columns of EXTRA_FIELDS to every episode row and, when `dump` is given, writes a text
    view of the teacher's map at the end of every episode that had an exit and did not reach it.
    """

    def __init__(self, dump: str = "", budget: float | str = "cap"):
        self.dump = Path(dump) if dump else None
        self.budget = budget            # "cap": the true cap of each map; a number: that many decision steps on every map
        self.policy = None
        self.task = None

    def begin(self, task, env, meta, policy, obs) -> None:
        self.policy, self.task = policy, task
        if hasattr(policy, "set_budget"):
            policy.set_budget(task["episode_ticks"] / spec.FRAME_SKIP if self.budget == "cap" else float(self.budget))

    def end(self, row: dict) -> dict:
        policy = self.policy
        if policy is None or not hasattr(policy, "debug_state"):
            return {}
        state = policy.debug_state()
        extra = {
            "budget_steps": state["budget"], "exit_step": state["exit_step"], "leave_step": state["leave_step"],
            "leave_travel": state["leave_travel"], "leave_last": state["leave_last"],
            "leave_cells": state["leave_route"][0], "leave_gates": state["leave_route"][1],
            "leave_turns": state["leave_route"][2], "label_ms": state["label_ms"], "front_ms": state["front_ms"],
            "teacher_errors": state["errors"], "teacher_error": state["last_error"], "searches": state["searches"],
            "replans": state["replans"], "final_subgoal": state.get("subgoal", ""), "final_mode": state.get("mode", ""),
            "locks_given_up": len(state["failed_locks"]), "slipped": state["slipped"], "use_masked": state["use_masked"],
            "grid_fits": int(state.get("grid_fits", True)), "labelled": state.get("labelled", 0),
            "unstuck": state.get("unstuck", 0),
        }
        steps = max(1, row["steps"])
        extra.update({f"mode_{m}": state["mode_steps"].get(m, 0) / steps for m in MODES})
        extra.update({f"sub_{n}": c / steps for n, c in state["subgoal_steps"].items()})
        extra["cause"] = classify({**row, **extra})
        if self.dump is not None and row["has_exit"] and not row["exit"]:
            from . import debug

            self.dump.mkdir(parents=True, exist_ok=True)
            text = (f"{row['map_id']} seed {row['seed']}: {extra['cause']}\n"
                    + json.dumps({k: v for k, v in {**row, **extra}.items() if k not in ("wad",)}, default=str) + "\n"
                    + debug.frame(policy) + "\n")
            (self.dump / f"{row['map_id']}_s{row['seed']}.txt").write_text(text)
        return extra


# --------------------------------------------------------------------------- classification
def classify(row: dict) -> str:
    """Why an episode did not reach the exit, from the scorer's row and the teacher's own counters."""
    if row.get("error"):
        return "episode could not be played"
    if not row.get("has_exit"):
        return "map has no exit"
    if row.get("exit"):
        return "ok"
    if row.get("agent_error") or (row.get("teacher_errors") or 0) > 0:
        return "teacher raised"
    if row.get("dead"):
        return "died"
    stuck = float(row.get("stuck_rate") or 0.0) > 0.25
    if (row.get("exit_step") if row.get("exit_step") is not None else -1) >= 0:      # the teacher had the exit on its map
        if (row.get("leave_step") if row.get("leave_step") is not None else -1) >= 0:
            return "left for the exit too late" + (" (stuck on the way)" if stuck else "")
        return "exit known but never left for it" + (" (stuck)" if stuck else "")
    if row.get("keys_picked", 0) < row.get("keys_total", 0) and row.get("locked_opened", 0) < row.get("locked_total", 0):
        return "key not collected" + (" (stuck)" if stuck else "")
    if row.get("locked_opened", 0) < row.get("locked_total", 0):
        return "locked door not opened" + (" (stuck)" if stuck else "")
    visible = row.get("exit_visible_tic")
    if visible is not None and visible == visible and visible >= 0:
        return "exit was in view but not registered" + (" (stuck)" if stuck else "")
    if stuck:
        return "stuck before the exit was seen"
    return "exit never came into view before the cap"


# ---------------------------------------------------------------------------------- table
def _mean(values) -> float:
    values = [float(v) for v in values if v is not None and v == v]
    return sum(values) / len(values) if values else float("nan")


def _pooled(rows, part: str, total: str) -> float:
    den = sum(r[total] for r in rows)
    return sum(r[part] for r in rows) / den if den else float("nan")


def gate_table(rows: list[dict]) -> list[dict]:
    """One dict per tier (and one for all tiers) with the numbers the gate asks for."""
    out = []
    tiers = sorted({r["tier"] for r in rows})
    for tier in tiers + ["all"]:
        part = [r for r in rows if tier == "all" or r["tier"] == tier]
        with_exit = [r for r in part if r["has_exit"]]
        reached = [r for r in with_exit if r["exit"]]
        exit_rate = len(reached) / len(with_exit) if with_exit else float("nan")
        sectors = _pooled(part, "sectors_visited", "sectors_total")
        reversals = sum(r["door_reversals"] for r in part)
        gate = GATE_EXIT.get(tier)
        passes = "" if gate is None else int(exit_rate >= gate and sectors >= GATE_SECTORS and reversals == 0)
        out.append({
            "tier": tier, "episodes": len(part), "exit_rate": exit_rate, "exit_gate": gate if gate is not None else "",
            "sectors_frac": sectors, "doors_frac": _pooled(part, "doors_opened", "doors_total"),
            "locked_frac": _pooled(part, "locked_opened", "locked_total"),
            "keys_frac": _pooled(part, "keys_picked", "keys_total"),
            "door_reversals": reversals,
            "exit_seen_timeout": sum(1 for r in part if r.get("exit_seen_timeout") == 1.0),
            "stuck_rate": _mean(r["stuck_rate"] for r in part),
            "stuck_at_door_rate": _mean(r["stuck_at_door_rate"] for r in part),
            "steps_to_exit": _mean(r["steps"] for r in reached),
            "steps_frac_of_cap": _mean(4.0 * r["steps"] / r["episode_ticks"] for r in reached),
            "label_ms": _mean(r.get("label_ms") for r in part), "front_ms": _mean(r.get("front_ms") for r in part),
            "odom_err_max": max([r["odom_err_max"] for r in part if r["odom_err_max"] == r["odom_err_max"]], default=float("nan")),
            "teacher_errors": sum(int(r.get("teacher_errors") or 0) for r in part),
            "use_masked": sum(int(r.get("use_masked") or 0) for r in part),
            "slipped_frac": _pooled(part, "slipped", "steps") if all("slipped" in r for r in part) else float("nan"),
            "labelled_frac": _pooled(part, "labelled", "steps") if all("labelled" in r for r in part) else float("nan"),
            "agent_errors": sum(1 for r in part if r.get("agent_error") or r.get("error")),
            "passes": passes,
        })
    return out


def failures(rows: list[dict]) -> list[dict]:
    """The episodes that had an exit and did not reach it, with their cause."""
    keep = ("map_id", "tier", "seed", "cause", "steps", "episode_ticks", "exit_step", "leave_step", "leave_travel",
            "keys_picked", "keys_total", "doors_opened", "doors_total", "locked_opened", "locked_total",
            "sectors_visited", "sectors_total", "stuck_rate", "exit_dist_min", "exit_visible_tic", "final_subgoal")
    return [{k: r.get(k) for k in keep} for r in rows if r["has_exit"] and not r["exit"]]


def _text(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def print_table(table: list[dict]) -> None:
    cols = (("tier", 4), ("episodes", 4), ("exit_rate", 6), ("sectors_frac", 8), ("doors_frac", 6), ("locked_frac", 7),
            ("keys_frac", 6), ("door_reversals", 4), ("exit_seen_timeout", 8), ("stuck_rate", 6), ("steps_to_exit", 7),
            ("label_ms", 6), ("teacher_errors", 4), ("passes", 4))
    names = ("tier", "n", "exit", "sectors", "doors", "locked", "keys", "rev", "seen+TO", "stuck", "steps", "ms", "err", "gate")
    print(" ".join(f"{n:>{w}}" for n, (_, w) in zip(names, cols)))
    for row in table:
        cells = []
        for key, width in cols:
            v = row[key]
            cells.append(f"{v:>{width}.3f}" if isinstance(v, float) and v == v else f"{'-' if v != v else v!s:>{width}}")
        print(" ".join(cells))


def run_gate(tag: str, suite="training/suites/dev_v1.jsonl", tiers=(2, 3, 4), exit_source: str = "label",
             start: str = "aligned", seeds=(0,), workers: int = 3, limit: int | None = None, log: bool = True,
             policy_kwargs: dict | None = None, quiet: bool = False, budget: float | str = "cap",
             monsters: bool = False) -> tuple[list[dict], list[dict]]:
    """Play the monster-free suite with the teacher; returns (episode rows, gate table).

    budget: "cap" hands the teacher the true cap of each map as its time budget (the gate's
    condition); a number hands it that many decision steps on every map, whatever the cap.
    """
    maps = bare_maps(suite, tiers, limit, start, monsters=monsters)
    kwargs = {"exit_source": exit_source, **(policy_kwargs or {})}
    dump = RUNS_DIR / tag
    began = time.perf_counter()
    rows = run_episodes((POLICY, kwargs), maps, workers=workers, seeds=list(seeds), render_buffers=False,
                        record=(RECORDER, {"dump": str(dump), "budget": budget}))
    on_grid = {m["id"]: m["on_grid"] for m in maps}
    for r in rows:
        r["on_grid"] = int(on_grid.get(r["map_id"], True))
    info = {"stamp": datetime.now().astimezone().isoformat(timespec="seconds"), "tag": tag, "suite": Path(suite).name,
            "exit_source": exit_source, "start": start + ("+monsters" if monsters else ""), "budget": str(budget),
            "seeds": " ".join(str(s) for s in seeds),
            "policy_kwargs": json.dumps(policy_kwargs or {}, sort_keys=True)}
    table = [{**info, **row} for row in gate_table(rows)]
    groups = [(rows, "")]
    if len({r["on_grid"] for r in rows}) == 2:               # mixed starts: the labelled part and the rest, apart
        for flag, name in ((1, "on-grid"), (0, "off-grid")):
            part = [r for r in rows if r["on_grid"] == flag]
            groups.append((part, name))
            table += [{**info, **row, "start": f"{info['start']}:{name}"} for row in gate_table(part)]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with open(RUNS_DIR / f"{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: _text(v) for k, v in r.items()} for r in rows)
    if log:
        LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        logged = []
        if LOG_CSV.is_file():                                # rewritten whole, so rows of an older column set stay aligned
            with open(LOG_CSV, newline="") as f:
                logged = list(csv.DictReader(f))
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS, extrasaction="ignore", restval="", lineterminator="\n")
            writer.writeheader()
            writer.writerows(logged)
            writer.writerows({k: _text(v) for k, v in row.items()} for row in table)
        book = json.loads(FAILURES_JSON.read_text()) if FAILURES_JSON.is_file() else {}
        # Episodes on maps the teacher's grid does not fit carry no labels: their failures are
        # counted by cause, not listed one by one.
        fitting = [r for r in rows if r["on_grid"]]
        entry = {**info, "episodes": len(rows), "failures": failures(fitting)}
        if len(fitting) < len(rows):
            causes: dict[str, int] = {}
            for r in failures([r for r in rows if not r["on_grid"]]):
                causes[r["cause"]] = causes.get(r["cause"], 0) + 1
            entry["off_grid_episodes"] = len(rows) - len(fitting)
            entry["off_grid_failure_causes"] = dict(sorted(causes.items(), key=lambda kv: -kv[1]))
        book[tag] = entry
        FAILURES_JSON.write_text(json.dumps(book, indent=1, default=str) + "\n")
    if not quiet:
        print(f"{tag}: {len(rows)} episodes in {time.perf_counter() - began:.0f} s, exit observations: {exit_source}, "
              f"start: {start}, budget: {budget}")
        at = 0
        for part, name in groups:
            tiers_here = len({r["tier"] for r in part}) + 1
            if name:
                agree = sum(1 for r in part if r.get("grid_fits") == r["on_grid"])
                print(f"-- {name} starts: {len(part)} episodes, {len({r['map_id'] for r in part})} maps; the teacher's own "
                      f"grid check agrees on {agree}; steps with controller labels: "
                      f"{sum(r.get('labelled', 0) for r in part)} of {sum(r['steps'] for r in part)}")
            print_table(table[at:at + tiers_here])
            at += tiers_here
        causes: dict[str, int] = {}
        for r in failures(rows):
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        for cause, count in sorted(causes.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3d}  {cause}")
        for r in failures([r for r in rows if r["on_grid"]]):
            print(f"       {r['map_id']} seed {r['seed']}: {r['cause']} (keys {r['keys_picked']}/{r['keys_total']}, "
                  f"locked {r['locked_opened']}/{r['locked_total']}, sectors {r['sectors_visited']}/{r['sectors_total']}, "
                  f"exit on map at step {r['exit_step']}, left at {r['leave_step']})")
    return rows, table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.teacher.gate", description=__doc__.split("\n")[2])
    ap.add_argument("--tag", required=True, help="name of this run in the logs")
    ap.add_argument("--suite", default="training/suites/dev_v1.jsonl")
    ap.add_argument("--tiers", default="2,3,4")
    ap.add_argument("--exit", default="label", choices=("label", "noisy", "none"), help="source of exit observations")
    ap.add_argument("--start", default="aligned", choices=("aligned", "random", "offset", "quarter", "keep"), help="player start of the rebuilt maps")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first N maps of every tier only")
    ap.add_argument("--budget", default="cap", help="'cap' (the true cap of each map) or a number of decision steps")
    ap.add_argument("--monsters", action="store_true", help="keep the monsters (the teacher does not fight them)")
    ap.add_argument("--no-log", action="store_true", help="do not append to training/logs")
    ap.add_argument("--kw", action="append", default=[], metavar="KEY=VALUE", help="TeacherPolicy keyword argument (JSON value)")
    args = ap.parse_args(argv)
    kwargs = {}
    for item in args.kw:
        key, _, value = item.partition("=")
        kwargs[key] = json.loads(value)
    run_gate(args.tag, args.suite, [int(t) for t in args.tiers.split(",")], args.exit, args.start,
             [int(s) for s in args.seeds.split(",")], args.workers, args.limit, not args.no_log, kwargs,
             budget="cap" if args.budget == "cap" else float(args.budget), monsters=args.monsters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
