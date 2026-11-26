"""Episode runner: plays a policy on maps in a pool of worker processes and scores every episode.

Training-time scaffolding, never imported by the agent. This is the tool that ranks agents, so
it is built to be boring: the same (policy, map, seed) always gives the same row, rows come back
in input order whatever the workers did, and a worker that crashes or hangs costs one episode,
not the run.

    rows = run_episodes("training.runner:RandomPolicy", ["maps/two_rooms.wad"], workers=3, seeds=[0, 1])

Policy interface
----------------
A policy is any object with `reset()` and `act(obs) -> (9,)` array of 0/1. Optional hooks:
    attach(env, meta)   training-only policies that read privileged state (called before reset)
    seed(n)             called with the episode seed before reset (for policies with their own RNG)
    pose_estimate()     (x, y, theta) in the agent's start frame, compared with the truth
    exit_believed()     True once the agent believes it knows where an exit is
For each episode the runner calls: env.reset(seed) -> attach -> seed -> reset -> act, act, ...

A policy that raises, or returns an invalid action, is treated as the official scorer treats it:
the player stands still for the rest of the map, the episode is still scored, and the message is
kept in the row's `agent_error`.

`pose_estimate()` and `exit_believed()` are telemetry that the official scorer never calls, so
they cannot cost the agent anything here either: a hook that raises or returns something
unusable is not called again in that episode, the message goes to the row's `hook_error`, and
the agent plays on.

Time limits
-----------
With `time_limits=True` the runner applies the limits of docs/EVALUATION.md, with the numbers of
`doom_env.limits` and the bookkeeping of the loop in validate.py:

    reset()                   10 s    over it: 0 points for the map (`error` is set as well)
    first act() of a map      10 s    over it: the agent stops
    any other act()            5 s    over it: the agent stops
    all act() calls of a map  40 ms x decision steps in the tick cap (21 s for 2,100 tics);
                              used up, or not enough left for the call in progress: the agent stops

An agent that stops stands still for the rest of the map, exactly as after an exception, and
`agent_error` says `timeout` and which limit. `time_scale` multiplies every measured duration
before it is compared, to judge an agent as a slower machine would (2.5 = a machine that needs
2.5 times as long). The 60 s limit on building the agent is checked by validate.py, not here.

The limits are off by default because they describe the submitted agent: training-only policies
(a teacher, a data collector) have no time budget. `training.evaluate` turns them on.

The official scorer waits for the agent on a socket and gives up when a limit passes. Here the
policy runs in the worker's own thread, so a call can only be judged once it returns. In worker
processes the runner therefore also sets a timer that interrupts a call still running at its
limit (`timed_call`); the worker then builds a fresh policy for its next map, as the official
scorer restarts the agent. A call that cannot be interrupted (one stuck in native code) is ended
by the pool's episode timeout, and with `workers=0` nothing can end it at all.

`policy_spec` is `"module:factory"`, or `("module:factory", {kwargs})`; the factory is called
once per worker process (and again after a call was interrupted). With `workers=0` everything
runs in the calling process and `policy_spec` may also be a ready policy object. Workers are
fresh interpreters (the "spawn" start method), each limited to one compute thread like the
evaluation sandbox; as with any multiprocessing code, a script that calls the runner needs the
`if __name__ == "__main__":` guard.

Rows
----
One dict per episode with the keys of ROW_FIELDS: `map_id`, `tier`, `seed`, `wad`, `map` and
`policy`; every metric of `scorer.RESULT_FIELDS`; `census_ok` (1 when the engine spawned exactly
the things the WAD lists at skill 3), `agent_error`, `hook_error` and `error` (set when the
episode could not be played at all, or is void because reset() ran out of time: it then counts
as score 0); and the timing fields `act_ms_mean`, `act_ms_max` (as measured, never scaled) and
`wall_s`, the only ones that differ between two runs of the same thing. With time limits on that
holds for a policy that stays inside them; whether a slow one is stopped depends on the clock.

Command line
------------
    python -m training.runner --bench                       throughput by worker count
    python -m training.runner --policy SPEC --maps M ...    play and print a summary
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np

from doom_env.limits import (ACT_TIMEOUT_S, AGENT_MS_PER_STEP, FIRST_ACT_TIMEOUT_S, RESET_TIMEOUT_S, THREAD_ENV,
                             agent_budget_s)
from doom_env.remote import validate_action
from training import paths
from training.mapmeta import MapTooLarge, engine_census, load_meta
from training.scorer import RESULT_FIELDS, EpisodeScorer, empty_result, suite_score

DEFAULT_TICKS = 2100
NOOP = np.zeros(9, np.int8)
EPISODE_TIMEOUT_S = 300.0   # how long the pool waits for one episode when no time limits apply
EPISODE_MARGIN_S = 60.0     # with time limits: engine, scorer and recorder time on top of the agent's own limits
# Every key of a result row, in order: identity, the scorer's metrics, status, timing.
ID_FIELDS = ("map_id", "tier", "seed", "wad", "map", "policy")
STATUS_FIELDS = ("census_ok", "agent_error", "hook_error", "error")
TIMING_FIELDS = ("act_ms_mean", "act_ms_max", "wall_s")   # the only fields that vary between runs
ROW_FIELDS = ID_FIELDS + RESULT_FIELDS + STATUS_FIELDS + TIMING_FIELDS


# ------------------------------------------------------------------------------------ specs
def parse_spec(spec) -> tuple[str, dict]:
    """`"module:factory"`, `("module:factory", {kwargs})` or `{"factory": ..., "kwargs": {...}}` -> (target, kwargs).

    A string may carry its keyword arguments as JSON, `'module:factory{"p": 0.3}'`, which is the
    form `spec_name` writes into result rows, so a logged policy can be pasted back in.
    """
    if isinstance(spec, str):
        target, brace, rest = spec.partition("{")
        kwargs = json.loads(brace + rest) if brace else {}
    elif isinstance(spec, dict):
        target, kwargs = spec["factory"], dict(spec.get("kwargs") or {})
    else:
        target, kwargs = spec[0], dict(spec[1] if len(spec) > 1 and spec[1] else {})
    if ":" not in target:
        raise ValueError(f"spec {target!r} is not of the form 'module:factory'")
    return target, kwargs


def spec_name(spec) -> str:
    """Stable text form of a spec, written to every row."""
    if not isinstance(spec, (str, dict, tuple, list)):
        return type(spec).__name__
    target, kwargs = parse_spec(spec)
    return target + (json.dumps(kwargs, sort_keys=True) if kwargs else "")


def build(spec):
    """Import `module` and call `factory(**kwargs)`."""
    target, kwargs = parse_spec(spec)
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)(**kwargs)


def _resolve_path(p) -> str:
    """Absolute path of a WAD given relative to the working directory or to the repository root."""
    p = Path(p)
    if not p.is_absolute() and not p.exists() and (paths.ROOT / p).exists():
        p = paths.ROOT / p
    return str(p.resolve())


def normalise_map(m) -> dict:
    """A map given as `"file.wad"`, `"file.wad:MAP05"` or a dict -> {id, wad, map, tier, episode_ticks}.

    Dict keys: `wad` (or `path`), `map` (default MAP01), `id`, `tier` (default 0) and
    `episode_ticks` (default 2100), which is how manifest rows of the map generator look.
    """
    if isinstance(m, (str, Path)):
        text = str(m)
        wad, _, name = (text, "", "") if text.lower().endswith(".wad") else text.rpartition(":")
        m = {"wad": wad, "map": name or "MAP01"}
    wad = m.get("wad") or m.get("path")
    if not wad:
        raise ValueError(f"map entry without a WAD path: {m}")
    name = str(m.get("map") or m.get("map_name") or "MAP01").upper()
    default_id = Path(str(wad)).stem + ("" if name == "MAP01" else f":{name}")
    return {
        "id": str(m.get("id") or m.get("map_id") or m.get("name") or default_id),
        "wad": _resolve_path(wad), "map": name, "tier": int(m.get("tier", 0) or 0),
        "episode_ticks": int(m.get("episode_ticks") or m.get("ticks") or DEFAULT_TICKS),
    }


def find_wad(ref, manifest_dir: Path) -> Path | None:
    """The WAD a manifest line refers to, or None if it is not there.

    A relative path is tried against the manifest's directory, the repository root and the
    working directory; a manifest that was moved together with its WADs is found by file name.
    """
    ref = Path(ref)
    tries = [ref] if ref.is_absolute() else [manifest_dir / ref, paths.ROOT / ref, Path.cwd() / ref]
    return next((c for c in tries + [manifest_dir / ref.name] if c.is_file()), None)


def load_maps(source, tier: int = 0, episode_ticks: int = DEFAULT_TICKS) -> list[dict]:
    """Maps from a `manifest.jsonl`, a directory of WADs, or one `file.wad[:MAP]`."""
    p = Path(str(source))
    if p.is_dir():
        return [normalise_map({"wad": w, "tier": tier, "episode_ticks": episode_ticks}) for w in sorted(p.glob("*.wad"))]
    if p.suffix.lower() == ".jsonl":
        out = []
        for number, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            wad = find_wad(row.get("wad") or row["path"], p.parent)
            if wad is None:
                raise FileNotFoundError(f"{p}: line {number}: WAD not found: {row.get('wad') or row['path']}")
            out.append(normalise_map({**row, "wad": str(wad), "path": None}))
        return out
    return [normalise_map({**normalise_map(str(source)), "tier": tier, "episode_ticks": episode_ticks})]


def episode_tasks(maps, seeds) -> list[dict]:
    """maps x seeds as a flat task list, map by map, in input order."""
    seeds = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    return [{**normalise_map(m), "seed": s} for m in maps for s in seeds]


# ---------------------------------------------------------------------------------- policies
class RandomPolicy:
    """Presses every button independently with probability `p`: the floor every agent must beat."""

    def __init__(self, p: float = 0.5, attack: bool = True):
        self.p, self.attack = float(p), bool(attack)
        self._seed = 0
        self.rng = np.random.default_rng(0)

    def seed(self, seed: int) -> None:
        self._seed = int(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)

    def act(self, obs) -> np.ndarray:
        action = (self.rng.random(9) < self.p).astype(np.int8)
        if not self.attack:
            action[6] = 0
        return action


class ScriptPolicy:
    """Replays a fixed button script, then stands still. For hand-checkable episodes and tests.

    The script is space-separated tokens; a token is one or more button letters pressed together,
    optionally followed by `*count`:
        F forward  B back  < strafe left  > strafe right  L turn left  R turn right
        A attack   U use   W next weapon  N nothing
    Example: `"F*12 N*6 U N*2 U N*30"` walks, waits, presses USE, waits two steps, presses again.
    """

    LETTERS = {"F": 0, "B": 1, "<": 2, ">": 3, "L": 4, "R": 5, "A": 6, "U": 7, "W": 8}

    def __init__(self, script: str = ""):
        self.actions = []
        for token in script.split():
            buttons, _, count = token.partition("*")
            action = np.zeros(9, np.int8)
            for letter in buttons.upper():
                if letter != "N":
                    action[self.LETTERS[letter]] = 1
            self.actions += [action] * (int(count) if count else 1)
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def act(self, obs) -> np.ndarray:
        action = self.actions[self.t] if self.t < len(self.actions) else NOOP
        self.t += 1
        return action


class CorridorPolicy:
    """Scripted reference for straight corridors: walk forward, open the door ahead, wait for it.

    Training-time reference only (hand-written rules are not allowed in the agent). It presses
    USE once when the ray straight ahead ends on a door within reach, then keeps its hands off
    USE for `wait` steps so the door is never pressed twice while it moves.
    """

    def __init__(self, wait: int = 12, reach: float = 64.0):
        self.wait, self.reach = int(wait), float(reach)
        self.cooldown = 0

    def reset(self) -> None:
        self.cooldown = 0

    def act(self, obs) -> np.ndarray:
        action = np.zeros(9, np.int8)
        action[0] = 1
        self.cooldown = max(0, self.cooldown - 1)
        door_ahead = int(obs["rays"]["hit_type"][0]) in (2, 3) and float(obs["rays"]["distance"][0]) * 1024.0 <= self.reach
        if door_ahead and self.cooldown == 0:
            action[7] = 1
            self.cooldown = self.wait
        return action


class AgentPolicy:
    """The submission's `DoomAgent` behind the policy interface, loaded the way the scorer loads it.

    `pose_estimate()` and `exit_believed()` are passed through when the agent defines them.
    """

    def __init__(self, device: str = "cpu", repo: str | None = None):
        from doom_env import action_space, observation_space

        root = Path(repo).resolve() if repo else paths.ROOT
        for entry in (str(root / "submission"), str(root)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        spec = importlib.util.spec_from_file_location("submission_agent", root / "submission" / "agent.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["submission_agent"] = module
        spec.loader.exec_module(module)
        self.agent = module.DoomAgent(observation_space(), action_space(), device)

    def reset(self) -> None:
        self.agent.reset()

    def act(self, obs) -> np.ndarray:
        return self.agent.act(obs)

    def __getattr__(self, name):
        if name in ("pose_estimate", "exit_believed"):
            return getattr(self.agent, name)
        raise AttributeError(name)


# ------------------------------------------------------------------------------ time limits
class CallTimeout(BaseException):
    """Raised inside a policy call that is still running when its time limit passes.

    Not an `Exception`, so an agent that catches its own errors does not swallow it.
    """


_interrupts_on = False    # this process may cut calls short (worker processes, see allow_interrupts)
_timer_armed = False      # a limited call is running right now
_calls_cut = 0            # how many calls this process has cut short


def _on_alarm(signum, frame) -> None:
    """Signal handler of the interval timer: end the call that is being timed."""
    global _timer_armed, _calls_cut
    if _timer_armed:   # a timer that fires just after its call returned is ignored
        _timer_armed = False
        _calls_cut += 1
        raise CallTimeout()


def allow_interrupts() -> bool:
    """Let `timed_call` cut short calls that run past their limit in this process.

    It takes over the process's real-time interval timer and its SIGALRM handler, so only the
    runner's own worker processes call it. Needs POSIX and the main thread; returns whether it
    is on.
    """
    global _interrupts_on
    if hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGALRM, _on_alarm)
        _interrupts_on = True
    return _interrupts_on


def timed_call(fn, limit_s: float | None = None):
    """Call `fn()` and time it: returns (value, exception or None, seconds taken, cut short).

    With a limit, in a process where `allow_interrupts()` has run, a call that is still running
    after `limit_s` seconds is cut short by raising CallTimeout inside it; value and exception
    are then None. Everywhere else the call runs to its end and the caller compares the time.
    The timer can only raise between two Python instructions of the main thread: it ends a
    sleep, a wait or a loop, but not a call stuck inside native code.
    """
    global _timer_armed
    cut_after = limit_s if _interrupts_on else None
    value = error = None
    started = time.perf_counter()
    try:
        try:
            if cut_after is not None:
                _timer_armed = True
                signal.setitimer(signal.ITIMER_REAL, max(cut_after, 1e-4))
            value = fn()
        finally:
            if cut_after is not None:
                _timer_armed = False
                signal.setitimer(signal.ITIMER_REAL, 0.0)
    except CallTimeout:
        return None, None, time.perf_counter() - started, True
    except Exception as e:  # noqa: BLE001 - the caller reports it
        error = e
    return value, error, time.perf_counter() - started, False


def _timeout(text: str, scale: float) -> str:
    """The `agent_error` text of a time limit: always contains ": timeout: " once the step is put in front."""
    return f"timeout: {text}" + ("" if scale == 1.0 else f" (measured time x {scale:g})")


def is_timeout(row) -> bool:
    """True if the agent of this row was stopped by a time limit (rather than by an error of its own)."""
    return ": timeout: " in str(row.get("agent_error") or "")


def patience_s(task: dict, time_scale: float | None = None, fixed: float | None = None) -> float:
    """Seconds the pool waits for one episode before it gives its worker up.

    `fixed` if given. Otherwise 300 s, or with time limits the most agent time an episode can
    use inside them (the reset limit plus the map's compute budget, in real seconds) plus a
    margin of 60 s for the engine, the scorer and a recorder.
    """
    if fixed is not None:
        return float(fixed)
    if time_scale is None:
        return EPISODE_TIMEOUT_S
    return (RESET_TIMEOUT_S + agent_budget_s(task["episode_ticks"])) / time_scale + EPISODE_MARGIN_S


# ------------------------------------------------------------------------------- one episode
def error_row(task: dict, message: str, policy_name: str = "", *, has_exit: bool | None = None,
              agent_error: str = "", wall_s: float = 0.0) -> dict:
    """Row for an episode that could not be played, or is void: score 0, measurements NaN, `error` set."""
    if has_exit is None:
        try:
            has_exit = load_meta(task["wad"], task["map"]).has_exit
        except Exception:  # noqa: BLE001 - the map itself may be the problem
            has_exit = True
    row = {"map_id": task["id"], "tier": task["tier"], "seed": task["seed"], "wad": task["wad"], "map": task["map"],
           "policy": policy_name, **empty_result(task["episode_ticks"], has_exit),
           "census_ok": float("nan"), "agent_error": agent_error[:500], "hook_error": "", "error": message.strip()[-2000:],
           "act_ms_mean": float("nan"), "act_ms_max": float("nan"), "wall_s": wall_s}
    return {k: row[k] for k in ROW_FIELDS}


HOOKS = {"pose_estimate": "pose", "exit_believed": "exit_believed"}   # policy hook -> argument of scorer.agent_report


def report_hooks(hooks: dict, scorer, t: int) -> str:
    """Hand the policy's optional reports for step `t` to the scorer; returns "" or what went wrong.

    `hooks` maps the names of HOOKS to the policy's bound methods (None where it has none). A
    hook that raises, or whose value the scorer cannot use, is set to None so that it is not
    called again: telemetry must never cost the agent a map.
    """
    message = ""
    for name, fn in tuple(hooks.items()):
        if fn is None:
            continue
        try:
            scorer.agent_report(**{HOOKS[name]: fn()})
        except Exception as e:  # noqa: BLE001
            hooks[name] = None
            message = message or f"step {t}: {name}(): {type(e).__name__}: {e}"
    return message


def run_episode(policy, env, meta, task: dict, recorder=None, policy_name: str = "",
                time_scale: float | None = None) -> dict:
    """Play one episode on an environment built for the task's map; returns the row (ROW_FIELDS).

    `time_scale` None (the default) lets the policy take as long as it likes. A number applies
    the official time limits to the measured time multiplied by it (1.0 = this machine is the
    evaluation machine); see "Time limits" in the module docstring.
    `recorder` may define `begin(task, env, meta, policy, obs)`,
    `step(t, obs, action, next_obs, reward, done, env, scorer)` and `end(row) -> dict | None`.
    """
    started = time.perf_counter()
    obs, _ = env.reset(seed=task["seed"])
    try:   # how many monsters the engine really spawned; the WAD count is the fallback
        census = engine_census(env)
        monsters, census_ok = sum(census["monster"].values()), int(census == meta.census())
    except Exception:  # noqa: BLE001 - an environment without an engine object list
        monsters, census_ok = None, float("nan")
    scorer = EpisodeScorer(env, meta, monsters)
    limited = time_scale is not None
    scale = float(time_scale) if limited else 1.0
    budget = agent_budget_s(scorer.cap)   # the map's compute budget: 40 ms x decision steps in its tick cap

    agent_error = hook_error = ""
    try:
        if hasattr(policy, "attach"):
            policy.attach(env, meta)
        if hasattr(policy, "seed"):
            policy.seed(task["seed"])
    except Exception as e:  # noqa: BLE001
        agent_error = f"reset: {type(e).__name__}: {e}"
    if not agent_error:
        _, error, dt, cut = timed_call(lambda: policy.reset(), RESET_TIMEOUT_S / scale if limited else None)
        if limited and (cut or dt * scale > RESET_TIMEOUT_S):   # officially: 0 points for the affected map
            slow = _timeout(f"reset() took longer than {RESET_TIMEOUT_S:g} s", scale)
            return error_row(task, f"the map is void (0 points): {slow}", policy_name, has_exit=meta.has_exit,
                             agent_error=f"reset: {slow}", wall_s=time.perf_counter() - started)
        if error is not None:
            agent_error = f"reset: {type(error).__name__}: {error}"
    hooks = {name: getattr(policy, name, None) for name in HOOKS}
    if recorder is not None and hasattr(recorder, "begin"):
        recorder.begin(task, env, meta, policy, obs)

    budget_text = (f"the agent's compute budget for this map ({budget:g} s = {AGENT_MS_PER_STEP:g} ms x "
                   f"{round(1000.0 * budget / AGENT_MS_PER_STEP)} decision steps)")
    act_total = act_max = 0.0   # seconds spent in act(), as measured
    acts = t = 0
    done = False
    while not done:
        action = NOOP
        left = budget - act_total * scale   # what remains of the map's compute budget
        if limited and not agent_error and left <= 0.0:
            agent_error = f"step {t}: " + _timeout(f"{budget_text} is used up", scale)
        if not agent_error:   # after a failure the player stands still, as in the official scorer
            per_call = FIRST_ACT_TIMEOUT_S if acts == 0 else ACT_TIMEOUT_S
            allowed = min(per_call, left)   # how long the official loop waits for this reply
            value, error, dt, cut = timed_call(lambda: policy.act(obs), allowed / scale if limited else None)
            act_total += dt
            act_max = max(act_max, dt)
            acts += 1
            if limited and (cut or dt * scale > allowed):   # too late: the action is not used
                if left < per_call:
                    agent_error = f"step {t}: " + _timeout(f"act() ran past the rest of {budget_text}", scale)
                else:
                    call = "the first act() of the map" if acts == 1 else "act()"
                    agent_error = f"step {t}: " + _timeout(f"{call} took longer than {per_call:g} s", scale)
            elif error is not None:
                agent_error = f"step {t}: {type(error).__name__}: {error}"
            else:
                try:
                    action = validate_action(value)   # the official check: shape (9,), values 0/1
                except Exception as e:  # noqa: BLE001
                    agent_error = f"step {t}: {type(e).__name__}: {e}"
            if not agent_error:
                failed = report_hooks(hooks, scorer, t)
                hook_error = hook_error or failed
        next_obs, reward, terminated, truncated, _ = env.step(action)
        scorer.step()
        done = bool(terminated or truncated)
        if recorder is not None and hasattr(recorder, "step"):
            recorder.step(t, obs, action, next_obs, reward, done, env, scorer)
        obs = next_obs
        t += 1

    row = {"map_id": task["id"], "tier": task["tier"], "seed": task["seed"], "wad": task["wad"], "map": task["map"],
           "policy": policy_name, **scorer.result(),
           "census_ok": census_ok, "agent_error": agent_error[:500], "hook_error": hook_error[:500], "error": "",
           "act_ms_mean": 1000.0 * act_total / max(1, acts), "act_ms_max": 1000.0 * act_max,
           "wall_s": time.perf_counter() - started}
    row = {k: row[k] for k in ROW_FIELDS}
    if recorder is not None and hasattr(recorder, "end"):
        row.update(recorder.end(row) or {})
    return row


class _Player:
    """One policy plus the environment of the map it is currently playing (one per worker)."""

    def __init__(self, policy_spec, record_spec, render_buffers: bool, time_scale: float | None = None):
        self.policy_spec = policy_spec
        self.policy = policy_spec if hasattr(policy_spec, "act") else build(policy_spec)
        self.policy_name = spec_name(policy_spec)
        is_spec = isinstance(record_spec, (str, dict, tuple, list))
        self.recorder = build(record_spec) if is_spec else record_spec
        self.render_buffers, self.time_scale = render_buffers, time_scale
        self.cut_short = False   # the policy was interrupted in the middle of a call (see timed_call)
        self._key, self._env, self._meta = None, None, None

    def _open(self, task: dict):
        key = (task["wad"], task["map"], task["episode_ticks"])
        if key != self._key:
            self.close()
            from doom_env import make_env

            self._meta = load_meta(task["wad"], task["map"])   # raises MapTooLarge before the engine starts
            self._env = make_env(task["wad"], task["map"], episode_ticks=task["episode_ticks"],
                                 render_buffers=self.render_buffers)
            self._key = key
        return self._env, self._meta

    def play(self, task: dict) -> dict:
        """Play one task. Never raises: a failure of the environment becomes an error row."""
        message = ""
        for _attempt in range(2):   # an engine that died is replaced once
            try:
                if self.cut_short:   # its state cannot be trusted any more: a fresh policy for the next map
                    self.policy, self.cut_short = build(self.policy_spec), False   # (the official scorer restarts the agent)
                env, meta = self._open(task)
                cuts = _calls_cut
                try:
                    return run_episode(self.policy, env, meta, task, self.recorder, self.policy_name, self.time_scale)
                finally:
                    self.cut_short = _calls_cut != cuts
            except MapTooLarge as e:
                return error_row(task, f"skipped: {e}", self.policy_name)
            except Exception:  # noqa: BLE001
                message = traceback.format_exc()
                self.close()
        return error_row(task, message, self.policy_name)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
        self._key, self._env, self._meta = None, None, None


# ------------------------------------------------------------------------------ worker pool
def _worker_main(conn, policy_spec, record_spec, render_buffers, time_scale=None) -> None:
    """Worker process: build the policy once, then play the chunks of episodes the parent sends."""
    try:
        os.setsid()   # own process group, so the parent can stop this worker and its engine together
    except (AttributeError, OSError):
        pass
    if time_scale is not None:
        allow_interrupts()   # a call that runs past its time limit is ended there, not waited for
    try:
        player = _Player(policy_spec, record_spec, render_buffers, time_scale)
    except BaseException:  # noqa: BLE001 - report everything, including SystemExit
        conn.send(("init_error", traceback.format_exc()))
        return
    conn.send(("ready",))
    try:
        while True:
            chunk = conn.recv()
            if chunk is None:
                break
            for index, task in chunk:
                conn.send(("start", index))
                conn.send(("result", index, player.play(task)))
            conn.send(("idle",))
    except (EOFError, OSError, KeyboardInterrupt):
        pass   # the parent went away
    finally:
        player.close()


@dataclass
class _Worker:
    proc: mp.Process
    conn: object
    deadline: float                    # monotonic time by which it must report again
    patience: float                    # the seconds it was given for that
    state: str = "starting"            # starting -> idle <-> busy; dead once it is gone
    chunk: list = field(default_factory=list)
    current: int | None = None         # index of the episode being played


class _Pool:
    """Hands chunks of episodes (all seeds of one map) to worker processes and survives their failures.

    The parent never blocks on one worker: it waits on all pipes at once and checks deadlines. A
    worker that died or ran past its deadline is killed together with its engine and replaced,
    and the rest of its chunk goes back to the front of the queue. The episode of a worker that
    died is retried `retries` times and then reported as an error row. An episode that ran past
    its deadline (see `patience_s`) becomes an error row at once: what hung will hang again.
    """

    def __init__(self, policy_spec, record_spec, workers, render_buffers, episode_timeout_s, init_timeout_s, retries,
                 progress, time_scale=None):
        self.args = (policy_spec, record_spec, render_buffers, time_scale)
        self.n, self.retries, self.progress = workers, retries, progress
        self.episode_timeout_s, self.init_timeout_s, self.time_scale = episode_timeout_s, init_timeout_s, time_scale
        self.ctx = mp.get_context("spawn")   # a clean interpreter per worker on every platform
        self.workers: list[_Worker] = []
        self.start_failures = 0

    def run(self, tasks: list[dict]) -> list[dict]:
        self.tasks = tasks
        self.rows: list = [None] * len(tasks)
        self.attempts = [0] * len(tasks)
        self.done = 0
        self.pending: deque = deque()
        for index, task in enumerate(tasks):   # consecutive episodes of one map share a chunk and an engine
            same_map = index > 0 and all(task[k] == tasks[index - 1][k] for k in ("wad", "map", "episode_ticks"))
            if same_map:
                self.pending[-1].append((index, task))
            else:
                self.pending.append([(index, task)])
        try:
            while self.done < len(tasks):
                self._staff()
                self._collect()
                self._check_deadlines()
        finally:
            self._shutdown()
        return self.rows

    # ---- workers
    def _spawn(self) -> None:
        parent_conn, child_conn = self.ctx.Pipe()
        proc = self.ctx.Process(target=_worker_main, args=(child_conn, *self.args), daemon=True)
        saved = {k: os.environ.get(k) for k in THREAD_ENV}
        os.environ.update(THREAD_ENV)   # one compute thread per worker, as in the evaluation sandbox
        try:
            proc.start()
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        child_conn.close()
        self.workers.append(_Worker(proc, parent_conn, time.monotonic() + self.init_timeout_s, self.init_timeout_s))

    def _expect(self, w: _Worker, index: int) -> None:
        """Restart a worker's clock: it has the time allowed for episode `index` to report again."""
        w.patience = patience_s(self.tasks[index], self.time_scale, self.episode_timeout_s)
        w.deadline = time.monotonic() + w.patience

    @staticmethod
    def _kill(w: _Worker) -> None:
        """Stop a worker for good. One that already left cleanly (exit code 0) has closed its engine itself."""
        if w.proc.exitcode != 0:
            try:   # the worker leads its own process group: this also stops the engine it started
                os.killpg(w.proc.pid, signal.SIGKILL)
            except (AttributeError, OSError):
                pass
        if w.proc.is_alive():
            w.proc.kill()
        w.proc.join(timeout=5)
        try:
            w.conn.close()
        except OSError:
            pass
        w.state = "dead"

    def _staff(self) -> None:
        """Replace dead workers while there is queued work, and give every idle worker a chunk."""
        self.workers = [w for w in self.workers if w.state != "dead"]
        busy = sum(1 for w in self.workers if w.state == "busy")
        while len(self.workers) < min(self.n, len(self.pending) + busy):
            self._spawn()
        for w in self.workers:
            if w.state == "idle" and self.pending:
                try:
                    w.conn.send(self.pending[0])
                except OSError:   # it died while idle: nothing was lost, it is replaced next time round
                    self._kill(w)
                    continue
                w.chunk, w.current, w.state = self.pending.popleft(), None, "busy"
                self._expect(w, w.chunk[0][0])
        if not any(w.state in ("starting", "busy") for w in self.workers):
            raise RuntimeError("internal error: episodes left without a worker")   # never spin without progress

    # ---- results
    def _finish(self, index: int, row: dict) -> None:
        self.rows[index] = row
        self.done += 1
        if self.progress:
            self.progress(self.done, len(self.tasks), row)

    def _collect(self) -> None:
        """Wait up to half a second for messages and handle whatever arrived."""
        by_conn = {w.conn: w for w in self.workers if w.state != "dead"}
        for conn in wait(list(by_conn), timeout=0.5):
            w = by_conn[conn]
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                self._lost(w, f"worker process died (exit code {w.proc.exitcode})")
                continue
            if msg[0] == "ready":
                w.state = "idle"
            elif msg[0] == "init_error":
                raise RuntimeError(f"the policy could not be built in a worker:\n{msg[1]}")
            elif msg[0] == "start":
                w.current = msg[1]
                self._expect(w, msg[1])
            elif msg[0] == "result":
                w.current = None
                self._expect(w, msg[1])
                self._finish(msg[1], msg[2])
            elif msg[0] == "idle":
                w.state, w.chunk = "idle", []

    def _check_deadlines(self) -> None:
        now = time.monotonic()
        for w in self.workers:
            if w.state not in ("starting", "busy"):
                continue
            if now > w.deadline:
                what = "did not start" if w.state == "starting" else "timed out: the policy or the engine did not come back"
                self._lost(w, f"worker {what} within {w.patience:.0f} s", retry=False)
            elif not w.proc.is_alive() and not w.conn.poll():   # gone without closing its pipe
                self._lost(w, f"worker process died (exit code {w.proc.exitcode})")

    def _lost(self, w: _Worker, reason: str, retry: bool = True) -> None:
        """A worker is gone: retry or fail the episode it was on and requeue the rest of its chunk."""
        was_starting = w.state == "starting"
        self._kill(w)
        if was_starting:
            self.start_failures += 1
            if self.start_failures >= 3:
                raise RuntimeError(f"worker processes keep failing during start-up ({reason})")
            return
        remaining = [(i, t) for i, t in w.chunk if self.rows[i] is None]
        if not remaining:
            return
        victim = w.current if w.current is not None and self.rows[w.current] is None else remaining[0][0]
        self.attempts[victim] += 1
        later = [(i, t) for i, t in remaining if i != victim]
        if later:
            self.pending.appendleft(later)
        if retry and self.attempts[victim] <= self.retries:
            self.pending.appendleft([(victim, self.tasks[victim])])
        else:
            self._finish(victim, error_row(self.tasks[victim], reason, spec_name(self.args[0])))

    def _shutdown(self) -> None:
        """Ask idle workers to leave, give them a moment, and kill whatever is still there."""
        for w in self.workers:
            if w.state == "idle":
                try:
                    w.conn.send(None)
                except OSError:
                    pass
        deadline = time.monotonic() + 5.0
        for w in self.workers:
            if w.state == "idle":
                w.proc.join(timeout=max(0.0, deadline - time.monotonic()))
            if w.state != "dead":
                self._kill(w)


def run_episodes(policy_spec, maps, workers: int = 3, seeds=(0,), record=None, *, render_buffers: bool = True,
                 time_limits: bool = False, time_scale: float = 1.0, episode_timeout_s: float | None = None,
                 init_timeout_s: float = 180.0, retries: int = 1, progress=None) -> list[dict]:
    """Play every map with every seed and return one row per episode, in input order.

    policy_spec  "module:factory" or ("module:factory", {kwargs}); see the module docstring.
    maps         "file.wad", "file.wad:MAP05" or dicts {wad, map, id, tier, episode_ticks}.
    workers      worker processes; 0 plays in the calling process.
    seeds        engine seeds (a list, or an int n for 0..n-1); every map is played with each.
    record       optional recorder spec ("module:factory", built once per worker), see run_episode.
    render_buffers  False leaves rgb/depth as zeros (for policies that do not look at pixels).
    time_limits  True applies the official limits on reset() and act() time (module docstring,
                 "Time limits"): an agent that exceeds one stops and stands still. Off by default,
                 for training-only policies.
    time_scale   with time_limits: measured time is multiplied by this before it is compared
                 (2.5 = judge the agent as a machine 2.5 times slower than this one would).
    retries      the episode of a worker that died is played again this many times, then reported
                 with `error` set.
    episode_timeout_s  a worker that takes longer than this over one episode is stopped and the
                 episode is reported with `error` set, without a retry. Default: 300 s, or with
                 time_limits the map's reset limit and compute budget plus 60 s (`patience_s`).
    progress     optional callback(done, total, row) called in this process as rows arrive.

    Each row has the keys of ROW_FIELDS: map id, tier, seed, every scorer metric, `agent_error`,
    `hook_error`, `error` and timing. Everything except TIMING_FIELDS is identical between runs.
    """
    if time_limits and not time_scale > 0.0:
        raise ValueError(f"time_scale must be positive, got {time_scale!r}")
    scale = float(time_scale) if time_limits else None
    tasks = episode_tasks(maps, seeds)
    if not tasks:
        return []
    if workers <= 0:
        player = _Player(policy_spec, record, render_buffers, scale)
        rows = []
        try:
            for task in tasks:
                rows.append(player.play(task))
                if progress:
                    progress(len(rows), len(tasks), rows[-1])
        finally:
            player.close()
        return rows
    for name, spec in (("policy_spec", policy_spec), ("record", record)):
        if spec is not None and not isinstance(spec, (str, dict, tuple, list)):
            raise TypeError(f"{name} must be a 'module:factory' spec when workers > 0 (objects only work with workers=0)")
    parse_spec(policy_spec)
    pool = _Pool(policy_spec, record, int(workers), render_buffers, episode_timeout_s, init_timeout_s, retries,
                 progress, scale)
    return pool.run(tasks)


# -------------------------------------------------------------------------------- benchmark
def default_bench_maps() -> list[dict]:
    """The small maps shipped with the template: the grid-layout test maps and two_rooms."""
    wads = sorted((paths.ROOT / "tests" / "fixtures").glob("*.wad")) + [paths.ROOT / "maps" / "two_rooms.wad"]
    return [normalise_map(str(w)) for w in wads]


def bench(maps=None, worker_counts=(1, 2, 3), seeds: int = 4, policy="training.runner:RandomPolicy") -> list[dict]:
    """Throughput by worker count, measured end to end.

    `steps_per_s` counts everything a ranking run pays for: starting the workers, starting one
    engine per map, playing and scoring. `in_episode_steps_per_s` is the rate of one worker while
    an episode is running (steps over the summed episode times), i.e. the ceiling per worker.
    """
    maps = maps or default_bench_maps()
    out = []
    for n in worker_counts:
        t0 = time.perf_counter()
        rows = run_episodes(policy, maps, workers=n, seeds=seeds, render_buffers=False)
        wall = time.perf_counter() - t0
        steps = sum(r["steps"] for r in rows)
        inside = sum(r["wall_s"] for r in rows)
        out.append({"workers": n, "episodes": len(rows), "steps": steps, "wall_s": wall,
                    "episodes_per_s": len(rows) / wall, "steps_per_s": steps / wall,
                    "in_episode_steps_per_s": steps / inside if inside else float("nan"),
                    "errors": sum(1 for r in rows if r["error"])})
        print(f"workers {n}: {len(rows)} episodes, {steps} steps in {wall:.1f} s = {steps / wall:,.0f} steps/s, "
              f"{len(rows) / wall:.1f} episodes/s end to end; {out[-1]['in_episode_steps_per_s']:,.0f} steps/s per "
              f"worker inside episodes; errors {out[-1]['errors']}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.runner", description=__doc__.split("\n")[0])
    ap.add_argument("--bench", action="store_true", help="measure throughput by worker count with the random policy")
    ap.add_argument("--policy", default="training.runner:RandomPolicy", help="module:factory")
    ap.add_argument("--kw", action="append", default=[], metavar="KEY=VALUE", help="policy keyword argument (JSON value)")
    ap.add_argument("--maps", nargs="*", default=[], help="manifest.jsonl, directory of WADs or file.wad[:MAP]")
    ap.add_argument("--seeds", default="0", help="comma-separated engine seeds, or N for 0..N-1 with --bench")
    ap.add_argument("--workers", default="3", help="worker processes (comma-separated list with --bench)")
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="tick cap for maps given without one")
    ap.add_argument("--time-limits", action="store_true", help="apply the official limits on reset() and act() time")
    ap.add_argument("--time-scale", type=float, default=1.0, help="with --time-limits: multiply measured time by this")
    args = ap.parse_args(argv)

    maps = [m for src in args.maps for m in load_maps(src, episode_ticks=args.ticks)]
    if args.bench:
        counts = [int(x) for x in args.workers.split(",")]
        bench(maps or None, counts if len(counts) > 1 else (1, 2, 3), seeds=int(args.seeds) or 4, policy=args.policy)
        return 0
    if not maps:
        ap.error("--maps is required (or use --bench)")
    kwargs = {}
    for item in args.kw:
        key, _, value = item.partition("=")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    rows = run_episodes((args.policy, kwargs), maps, workers=int(args.workers), seeds=[int(s) for s in args.seeds.split(",")],
                        time_limits=args.time_limits, time_scale=args.time_scale)
    for r in rows:
        trouble = r["error"] or r["agent_error"] or (f"hook: {r['hook_error']}" if r["hook_error"] else "")
        trouble = trouble.strip().splitlines()[-1] if trouble else ""
        print(f"{r['map_id']:>24} tier {r['tier']} seed {r['seed']}: score {r['score']:6.2f} exit {r['exit']} "
              f"steps {r['steps']:4d} kills {r['kills']}/{r['monsters_total']} {trouble}")
    total = suite_score(rows)
    print(f"score {total['score']:.2f} +- {total['score_se']:.2f} over {total['episodes']} episodes, "
          f"agent failures {sum(1 for r in rows if r['agent_error'])}, errors {total['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
