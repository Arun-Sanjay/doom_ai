"""Latency of the whole agent: `DoomAgent.act()` on one thread, on real maps, driven by its own actions.

Training-time scaffolding, never imported by the agent.

    .venv/bin/python training/bench/agent_latency.py                       # 3 maps x 700 decisions x 3 repeats
    .venv/bin/python training/bench/agent_latency.py --steps 1050 --repeats 5 --json runs/latency.json
    .venv/bin/python training/bench/agent_latency.py --quick               # short smoke run

What is measured
    The agent is built once, exactly as the scorer builds it, and then plays three maps of the
    frozen suites: the smallest, a mid-size one and the largest (by grid area). Each observation
    comes from the engine and each action is the agent's own, so the pose estimate, the memory map
    and the recurrent state evolve as they do at evaluation. When an episode ends (exit, death or
    tick cap) the map and the agent are reset and the run continues until `--steps` decisions were
    timed on that map. Only the time inside `act()` counts; the engine's time does not, as in the
    official budget.

What is printed
    mean, p95 and max of `act()` in milliseconds for every map and for all maps together; the time
    of the very first `act()` after the agent was built and of the first `act()` after each
    `reset()` (the first call of a map counts towards the map's budget like any other); how long
    building the agent took; and a per-stage breakdown from the agent's own stage hook:
    odometry (with the input checks), exit detector, memory update, crop, encoder and masks, policy
    network on manager steps and on other steps, and decoding. The hook costs about a microsecond
    per decision.

Also counted, because either would make the times mean something else: decisions on which the
agent's time governor was engaged (it rests the networks on every second decision, so such a
decision is cheaper than a full one), and faults inside the agent: decisions answered with no
buttons, pose updates that raised and were survived, and measurements that failed inside the
odometry module (its own `errors` counter, read from `debug_state()` before every reset). The
first two kinds of fault count as failures of the run; the third is printed as a warning.

The whole measurement is repeated `--repeats` times. The tables show the repeat with the lowest
mean, and the range of the repeats' means says how noisy the machine was; the load average at the
start is printed too, with a warning when other jobs were using the machine. Targets: TARGET_MAC_MS
mean on the development Mac and TARGET_CI_MS mean on an x86 CI runner (the binding number). The
exit status is 1 when the best repeat's mean is over the target of the machine it ran on.

Without committed weights the agent refuses to start; set DOOMNET_ALLOW_UNTRAINED=1 to time the
same networks with their initial weights (the cost of a pass does not depend on the weights).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, so the script also runs by path

from training import paths  # noqa: E402  (puts submission/ on sys.path)

import numpy as np  # noqa: E402

TARGET_MAC_MS = 8.0          # design target on the development machine (Apple silicon)
TARGET_CI_MS = 20.0          # binding target on an x86 CI runner
SUITES = ("training/suites/dev_v1.jsonl", "training/suites/heldout_v1.jsonl")
DEFAULT_STEPS = 700          # per map: 3 maps -> 2,100 decisions per repeat
DEFAULT_REPEATS = 3


def target_ms() -> float:
    """The mean-latency target of the machine this runs on."""
    return TARGET_MAC_MS if (platform.system() == "Darwin" and platform.machine() == "arm64") else TARGET_CI_MS


def pick_maps() -> list[dict]:
    """The smallest, the median and the largest map of the frozen suites, by grid area (ties by name)."""
    rows = []
    for suite in SUITES:
        path = paths.ROOT / suite
        if path.is_file():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise FileNotFoundError(f"no frozen suite found under {paths.ROOT / 'training' / 'suites'}")
    rows.sort(key=lambda r: (r["counts"]["width"] * r["counts"]["height"], r["name"]))
    chosen = {"small": rows[0], "mid-size": rows[len(rows) // 2], "largest": rows[-1]}
    return [dict(row, label=label) for label, row in chosen.items()]


class StageClock:
    """The agent's stage hook: called with a stage name when that stage has finished."""

    def __init__(self, stages):
        self.seconds = {name: [] for name in stages}
        self.mark = 0.0

    def start(self) -> None:
        self.mark = time.perf_counter()

    def __call__(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name].append(now - self.mark)
        self.mark = now


def build_agent():
    """(agent module, agent, seconds it took to import and build it)."""
    from doom_env import action_space, observation_space

    started = time.perf_counter()
    import agent as agent_module

    agent = agent_module.DoomAgent(observation_space(), action_space(), "cpu")
    return agent_module, agent, time.perf_counter() - started


def time_map(agent, env, steps: int, seed: int, clock: StageClock) -> dict:
    """Play `steps` decisions on one map with the agent's own actions. Returns the act() times in ms."""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    ms, first, manager, episodes = [], [], [], 1
    governed = module_errors = 0
    fresh = True
    manager_calls = clock.seconds["policy_manager"]
    for _ in range(steps):
        before = len(manager_calls)
        clock.start()
        t0 = time.perf_counter()
        action = agent.act(obs)
        dt = 1000.0 * (time.perf_counter() - t0)
        ms.append(dt)
        manager.append(len(manager_calls) > before)
        governed += int(getattr(agent, "governor_level", 0) > 0)
        if fresh:
            first.append(dt)
            fresh = False
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            seed += 1
            episodes += 1
            module_errors += odometry_errors(agent)
            obs, _ = env.reset(seed=seed)
            agent.reset()
            fresh = True
    module_errors += odometry_errors(agent)
    # "failures" and "odometry_failures" are the agent's own counters, which run from the moment it was built.
    return {"ms": np.array(ms), "manager": np.array(manager), "first_ms": first, "episodes": episodes,
            "failures": int(agent.failures) + int(getattr(agent, "odometry_failures", 0)),
            "odometry_failures": int(getattr(agent, "odometry_failures", 0)),
            "odometry_errors": module_errors, "governed": governed}


def odometry_errors(agent) -> int:
    """The odometry module's own error count for the map being played (it starts again at every reset())."""
    return int(agent.debug_state().get("odometry_errors") or 0)


def stats(ms) -> dict:
    ms = np.asarray(ms, np.float64)
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)), "max": float(ms.max()), "n": int(len(ms))}


def run(steps: int, repeats: int, seed: int = 0, log=print) -> dict:
    """The whole measurement; returns everything that is printed, as plain data."""
    from doom_env import make_env
    from training.mapgen import udmf

    maps = pick_maps()
    agent_module, agent, build_s = build_agent()
    stages = agent_module.STAGES
    result = {"machine": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
              "cores": os.cpu_count(), "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
              "target_ms": target_ms(), "build_s": build_s, "steps_per_map": steps, "repeats": [],
              "maps": [{"label": m["label"], "name": m["name"], "tier": m["tier"],
                        "size": [m["counts"]["width"], m["counts"]["height"]], "open_cells": m["counts"]["open_cells"]} for m in maps]}
    with tempfile.TemporaryDirectory(prefix="agent-latency-") as tmp:
        envs = []
        try:
            for m in maps:
                wad = udmf.write_wad(m["grid"], Path(tmp) / f"{m['name']}.wad", **m["options"])
                envs.append(make_env(str(wad), episode_ticks=int(m["episode_ticks"])))
            for repeat in range(repeats):
                clock = StageClock(stages)
                agent.stage_hook = clock
                per_map = []
                for m, env in zip(maps, envs):
                    per_map.append(time_map(agent, env, steps, seed + 1000 * repeat, clock))
                agent.stage_hook = None
                every = np.concatenate([p["ms"] for p in per_map])
                manager = np.concatenate([p["manager"] for p in per_map])
                n = len(every)
                result["repeats"].append({
                    "all": stats(every),
                    "manager_steps": stats(every[manager]), "other_steps": stats(every[~manager]),
                    "maps": [dict(stats(p["ms"]), episodes=p["episodes"], first_ms=p["first_ms"]) for p in per_map],
                    "first_act_ms": per_map[0]["first_ms"][0],
                    "failures": per_map[-1]["failures"],
                    "odometry_failures": per_map[-1]["odometry_failures"],
                    "odometry_errors": sum(p["odometry_errors"] for p in per_map),
                    "governed": sum(p["governed"] for p in per_map),
                    "stages": {name: {"calls": len(v), "mean_us": 1e6 * float(np.mean(v)) if v else 0.0,
                                      "p95_us": 1e6 * float(np.percentile(v, 95)) if v else 0.0,
                                      "per_step_us": 1e6 * float(np.sum(v)) / n} for name, v in clock.seconds.items()},
                })
                log(f"repeat {repeat + 1}/{repeats}: mean {result['repeats'][-1]['all']['mean']:.3f} ms over {n} decisions")
        finally:
            for env in envs:
                env.close()
    means = [r["all"]["mean"] for r in result["repeats"]]
    result["best"] = int(np.argmin(means))
    result["mean_range_ms"] = [float(min(means)), float(max(means))]
    result["first_act_after_build_ms"] = result["repeats"][0]["first_act_ms"]
    return result


def report(result: dict, log=print) -> bool:
    """Print the tables of the best repeat. Returns True when its mean meets the machine's target."""
    best = result["repeats"][result["best"]]
    lo, hi = result["mean_range_ms"]
    log("")
    log(f"DoomAgent.act() on one thread, {result['machine']}, python {result['python']}")
    log(f"building the agent (imports, weights, warm-up): {result['build_s']:.2f} s")
    log(f"best of {len(result['repeats'])} repeats; means of the repeats ranged from {lo:.3f} to {hi:.3f} ms")
    load, cores = result.get("load_average"), result.get("cores") or 1
    if load:
        busy = load[0] > 0.5 * cores
        log(f"load average when the run started: {load[0]:.1f} on {cores} cores" + (
            " -> the machine was busy: these times are inflated, measure again on an idle machine for the definitive number"
            if busy else ""))
    log("")
    log(f"{'map':<34}{'cells':>6}{'episodes':>9}{'steps':>7}{'mean':>9}{'p95':>9}{'p99':>9}{'max':>9}   ms")
    for m, s in zip(result["maps"], best["maps"]):
        name = f"{m['label']}: {m['name']} {m['size'][0]}x{m['size'][1]}"
        log(f"{name:<34}{m['open_cells']:>6}{s['episodes']:>9}{s['n']:>7}{s['mean']:>9.3f}{s['p95']:>9.3f}{s['p99']:>9.3f}{s['max']:>9.3f}")
    a = best["all"]
    for label, s in (("all maps", a), ("  decisions with a manager pass", best["manager_steps"]),
                     ("  other decisions", best["other_steps"])):
        log(f"{label:<34}{'':>6}{'':>9}{s['n']:>7}{s['mean']:>9.3f}{s['p95']:>9.3f}{s['p99']:>9.3f}{s['max']:>9.3f}")
    log("")
    log(f"{'stage':<18}{'calls':>7}{'mean':>10}{'p95':>10}{'per decision':>15}   microseconds")
    for name, s in best["stages"].items():
        log(f"{name:<18}{s['calls']:>7}{s['mean_us']:>10.1f}{s['p95_us']:>10.1f}{s['per_step_us']:>15.1f}")
    log(f"{'sum of stages':<18}{'':>7}{'':>10}{'':>10}{sum(s['per_step_us'] for s in best['stages'].values()):>15.1f}")
    log("")
    firsts = [f for r in result["repeats"] for m in r["maps"] for f in m["first_ms"]]
    log(f"first act() after the agent was built: {result['first_act_after_build_ms']:.3f} ms")
    log(f"first act() after reset(), {len(firsts)} resets: mean {np.mean(firsts):.3f} ms, max {np.max(firsts):.3f} ms "
        f"(it always includes a manager pass; mean of such decisions {best['manager_steps']['mean']:.3f} ms)")
    last = result["repeats"][-1]                 # the agent's counters are never reset: the last repeat holds the totals
    failures, lost_poses = last["failures"], last.get("odometry_failures", 0)
    if failures - lost_poses:
        log(f"WARNING: {failures - lost_poses} decisions raised inside the agent and were answered with no buttons")
    if lost_poses:
        log(f"WARNING: {lost_poses} pose updates raised inside the agent; it kept its pose and went on")
    module_errors = sum(r.get("odometry_errors", 0) for r in result["repeats"])
    if module_errors:
        log(f"WARNING: {module_errors} measurements failed inside the odometry module, which kept its model's answer "
            "(see debug_state()['odometry_last_error'])")
    governed = sum(r.get("governed", 0) for r in result["repeats"])
    if governed:
        log(f"WARNING: the time governor was engaged on {governed} decisions, so those were not full decisions: "
            "the machine was too slow or too busy for this to be the agent's own time")
    ok = a["mean"] <= result["target_ms"]
    log(f"mean {a['mean']:.3f} ms  p95 {a['p95']:.3f} ms  max {a['max']:.3f} ms  "
        f"(target on this machine: mean <= {result['target_ms']:g} ms) -> {'OK' if ok else 'OVER TARGET'}")
    return ok and not failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="decisions timed per map and repeat")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="120 decisions per map, one repeat")
    ap.add_argument("--json", help="also write every number to this file")
    args = ap.parse_args(argv)
    if args.quick:
        args.steps, args.repeats = 120, 1
    result = run(args.steps, args.repeats, args.seed)
    ok = report(result)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
