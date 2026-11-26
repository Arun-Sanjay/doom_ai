"""Fuzz test of the evaluation entry point, submission/agent.py::DoomAgent.

The scorer stops an agent for the rest of a map when act() raises, returns anything other than nine
0/1 values, or runs out of time (docs/EVALUATION.md). This file feeds the agent observations it
should never see in normal play and checks that none of that happens.

How the agent is run
    In its own process, behind the socket protocol of doom_env.remote, exactly as validate.py and
    the scorer run it (AgentProcess below). Nothing that other tests did to this pytest process can
    help or hurt it, a crash cannot take pytest down, and act() time is measured the way the scorer
    counts it. The agent is started with DOOMNET_ALLOW_UNTRAINED=1, so the file works before any
    weights exist and uses the real weights once they do. An agent that cannot be built (an import
    error, a constructor that raises, an initialisation that runs out of time) fails every test
    that needs it: with that variable set there is no acceptable reason, and at evaluation such an
    agent scores nothing. The tests are skipped only while submission/agent.py does not exist.

How the inputs are made
    ordinary() builds a random observation with exactly the keys, shapes and dtypes of
    doom_env.spaces.observation_space(); two tests check that against the space and against a real
    observation from the engine. Every entry of ODD then edits such an observation: NaN and
    infinity in the vitals, all-zero and all-one rays, 32 visible entities, zero and 200 health,
    extreme velocities, wrong but castable dtypes, and values outside the documented ranges.

What must hold (docs/CONTRACT.md section 6, and the agent's own rule that act() never raises)
    * act() never raises and always returns nine values that the scorer's own check accepts;
    * reset() may be called at any time, also in the middle of an episode, and afterwards the agent
      works normally again;
    * a bad observation costs at most the decision it arrives on: the agent goes on pressing
      buttons on the ordinary observations that follow, without waiting for reset();
    * decisions in a row without reset() are handled for longer than any map lasts: 1,060 in the
      fast tests (the longest map has 1,050) and 1,200 in a test marked slow;
    * act() stays within the time budget on ordinary and on odd inputs: its mean within the budget
      (the scorer stops an agent whose average over a map is higher) and its 99th percentile within
      the budget plus 5 ms. The budget is the scorer's 40 ms unless DOOMNET_FUZZ_BUDGET_MS says
      otherwise; CI sets it to the agent's own target for that machine;
    * the agent does not depend on the `device` argument: built with device="cuda" on a machine
      without one it still works, never tries to initialise CUDA and never loads the engine or the
      training code.

An agent that swallows an internal error answers with all zeros, which is a valid action. So that
the checks above cannot pass on an agent that is failing inside, the tests that feed ordinary
observations also require at least one pressed button, and so do the odd observations that the
engine can really produce (PLAUSIBLE), such as 32 visible enemies or no ray hitting anything.

The last part of the file checks the checks: stand-in agents with one known defect each (a
constructor that raises, a buffer shorter than the longest map, silence after one NaN until
reset(), a steady 42 ms per call, a try at the GPU with a fallback) must fail.

Run as a script (`python test_fuzz.py DEVICE`) this file is the device check used by one of the tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pytest
from doom_env import limits, remote
from doom_env import spaces as S

ROOT = Path(__file__).resolve().parents[2]
AGENT_PATH = ROOT / "submission" / "agent.py"

# The scorer allows 40 ms of agent time per decision, averaged over a map, and that is the default
# budget here. The agent's own targets are lower (8 ms on the development machine, 20 ms on the CI
# runner, measured by training/bench/agent_latency.py). DOOMNET_FUZZ_BUDGET_MS sets the budget this
# file enforces: the CI workflow sets it to 20, and it can be set by hand to check a tighter one.
BUDGET_MS = float(os.environ.get("DOOMNET_FUZZ_BUDGET_MS", limits.AGENT_MS_PER_STEP))
P99_SLACK_MS = 5.0
WARMUP_STEPS = 20        # act() calls made before timing starts
LATENCY_STEPS = 300      # timed act() calls in the fast latency test
PAST_LONGEST_MAP_STEPS = 1060  # more than the 1,050 decisions of the longest map (fast test)
LONG_RUN_STEPS = 1200          # the run without reset() that the build spec asks for (slow test)
LIVENESS_STEPS = 60      # ordinary observations within which a working agent presses some button
ODD_STEPS = 3            # consecutive observations of one odd kind
PLAUSIBLE_STEPS = 10     # the same for the kinds listed in PLAUSIBLE, which must also get a button pressed
MARK = "@@fuzz "         # prefix of the result line printed by the device check

V = S.VITALS_INDEX
RAY = {name: i for i, name in enumerate(S.RAY_HIT_TYPES)}
ENT = {name: i for i, name in enumerate(S.ENTITY_TYPES)}


# ---------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------
def ordinary(rng: np.random.Generator) -> dict:
    """A random, plausible observation with exactly the wrapper's keys, shapes and dtypes."""
    distance = rng.uniform(0.02, 1.0, S.NUM_RAYS).astype(np.float32)
    hit_type = rng.choice(len(S.RAY_HIT_TYPES), S.NUM_RAYS, p=[0.10, 0.70, 0.05, 0.03, 0.06, 0.06]).astype(np.int64)
    distance[hit_type == RAY["none"]] = 1.0  # the wrapper reports "nothing hit" at full range

    n = int(rng.integers(0, 7))  # visible entities, nearest first, the other slots zero
    angle = rng.uniform(-np.pi / 4, np.pi / 4, n)
    features = np.zeros((S.MAX_ENTITIES, len(S.ENTITY_FEATURES)), np.float32)
    features[:n] = np.column_stack([np.sort(rng.uniform(0.03, 3.0, n)), np.sin(angle), np.cos(angle), rng.normal(0, 3, (n, 2))])
    entity_type = np.zeros(S.MAX_ENTITIES, np.int64)
    entity_type[:n] = rng.integers(1, len(S.ENTITY_TYPES), n)
    mask = np.zeros(S.MAX_ENTITIES, np.int8)
    mask[:n] = 1

    vitals = np.zeros(len(S.VITALS_FIELDS), np.float32)
    vitals[V["health"]] = rng.integers(1, 101)
    vitals[V["armor"]] = rng.integers(0, 101)
    vitals[V["selected_weapon"]] = rng.integers(1, 8)
    vitals[V["selected_weapon_ammo"]:V["ammo_cells"] + 1] = rng.integers(0, 200, 5)
    vitals[V["key_blue_card"]:V["key_red_skull"] + 1] = rng.random(6) < 0.2
    vitals[V["vel_forward"]:V["vel_up"] + 1] = rng.normal([4.0, 0.0, 0.0], [3.0, 2.0, 0.1])
    return {
        "rgb": rng.integers(0, 256, (S.SCREEN_HEIGHT, S.SCREEN_WIDTH, 3), dtype=np.uint8),
        "depth": rng.integers(0, 256, (S.SCREEN_HEIGHT, S.SCREEN_WIDTH), dtype=np.uint8),
        "vitals": vitals,
        "rays": {"distance": distance, "hit_type": hit_type},
        "visible_entities": {"features": features, "entity_type": entity_type, "mask": mask},
    }


def leaves(tree, prefix: str = "") -> dict:
    """Flatten a nested observation (or the nested space) to {"rays/distance": leaf, ...}."""
    out = {}
    for key, value in (tree.spaces if hasattr(tree, "spaces") else tree).items():
        if isinstance(value, dict) or hasattr(value, "spaces"):
            out.update(leaves(value, f"{prefix}{key}/"))
        else:
            out[prefix + key] = value
    return out


FLOAT_LEAVES = ("vitals", "rays/distance", "visible_entities/features")
INDEX_LEAVES = ("rays/hit_type", "visible_entities/entity_type", "visible_entities/mask")
SCREEN_LEAVES = ("rgb", "depth")
ALL_LEAVES = FLOAT_LEAVES + INDEX_LEAVES + SCREEN_LEAVES


def leaf(obs: dict, path: str) -> np.ndarray:
    """The array at a path such as "rays/distance"."""
    for key in path.split("/"):
        obs = obs[key]
    return obs


def replace_leaves(obs: dict, paths, change) -> None:
    """obs[path] = change(obs[path]) for every path."""
    for path in paths:
        *parents, last = path.split("/")
        parent = leaf(obs, "/".join(parents)) if parents else obs
        parent[last] = change(parent[last])


# ---- editors: each returns a function (obs, rng) -> None that changes the observation in place ----
def set_vitals(**fields):
    """Named vitals set to the given values, for example set_vitals(health=0)."""
    def edit(obs, rng):
        for name, value in fields.items():
            obs["vitals"][V[name]] = value
    return edit


def all_vitals(*values):
    """The given values repeated over all 17 fields."""
    def edit(obs, rng):
        obs["vitals"][:] = np.resize(np.array(values, np.float32), len(S.VITALS_FIELDS))
    return edit


def velocity(value):
    """All three velocity fields at the given magnitude."""
    return set_vitals(vel_forward=value, vel_right=-value, vel_up=value)


def all_rays(distance, hit_type):
    """Every one of the 64 rays with the same distance and hit type."""
    def edit(obs, rng):
        obs["rays"]["distance"][:] = distance
        obs["rays"]["hit_type"][:] = RAY[hit_type]
    return edit


def crowd(*types, distance=None, speed=None):
    """All 32 entity slots in use, with types drawn from `types`."""
    def edit(obs, rng):
        n, ents = S.MAX_ENTITIES, obs["visible_entities"]
        angle = rng.uniform(-np.pi / 4, np.pi / 4, n)
        dist = np.sort(rng.uniform(0.03, 4.0, n)) if distance is None else np.full(n, distance)
        vel = rng.normal(0, 3, (n, 2)) if speed is None else np.full((n, 2), speed)
        ents["features"][:] = np.column_stack([dist, np.sin(angle), np.cos(angle), vel])
        ents["entity_type"][:] = rng.choice([ENT[t] for t in types], n)
        ents["mask"][:] = 1
    return edit


def screen(colour, depth):
    """A frame of one colour and a depth buffer of one value."""
    def edit(obs, rng):
        obs["rgb"][:] = colour
        obs["depth"][:] = depth
    return edit


def cast(dtype, paths):
    """The same values in another dtype."""
    return lambda obs, rng: replace_leaves(obs, paths, lambda a: a.astype(dtype))


def put(path, value, index=np.s_[:]):
    """Write `value` into part of one array; used for values outside the documented ranges."""
    def edit(obs, rng):
        leaf(obs, path)[index] = value
    return edit


def together(*edits):
    """Several edits applied one after the other."""
    def edit(obs, rng):
        for one in edits:
            one(obs, rng)
    return edit


EVERY_KIND = [t for t in S.ENTITY_TYPES if t != "pad"]
ENEMIES = ("enemy_melee", "enemy_hitscan", "enemy_projectile")
F32_MAX = float(np.finfo(np.float32).max)
AHEAD = [62, 63, 0, 1, 2]  # the rays within 11 degrees of straight ahead
DOOR_AHEAD = together(put("rays/distance", 0.03, AHEAD), put("rays/hit_type", RAY["door"], AHEAD))  # 31 units away: USE reaches it

ODD = {
    # ---- NaN and infinity in the vitals
    "vitals all NaN": all_vitals(np.nan),
    "vitals all +inf": all_vitals(np.inf),
    "vitals all -inf": all_vitals(-np.inf),
    "vitals NaN, +inf and -inf mixed": all_vitals(np.nan, np.inf, -np.inf, 1.0),
    "health NaN": set_vitals(health=np.nan),
    "selected weapon NaN": set_vitals(selected_weapon=np.nan),
    "selected weapon +inf": set_vitals(selected_weapon=np.inf),
    "ammo NaN": set_vitals(selected_weapon_ammo=np.nan, ammo_bullets=np.nan, ammo_shells=np.inf),
    "key flags NaN": set_vitals(key_blue_card=np.nan, key_red_skull=np.nan),
    "velocity NaN": velocity(np.nan),
    "velocity +inf and -inf": velocity(np.inf),
    # ---- health and the other vitals at and beyond their limits
    "health 0": set_vitals(health=0),
    "health 200, armor 200": set_vitals(health=200, armor=200),
    "health negative": set_vitals(health=-50),
    "health and armor huge": set_vitals(health=1e9, armor=1e9),
    "weapon slot 0": set_vitals(selected_weapon=0),
    "weapon slot 7": set_vitals(selected_weapon=7),
    "weapon slot 9": set_vitals(selected_weapon=9),
    "weapon slot -1": set_vitals(selected_weapon=-1),
    "weapon slot 1e6": set_vitals(selected_weapon=1e6),
    "no ammo, every key held": together(all_vitals(0.0), set_vitals(health=100, selected_weapon=2, key_blue_card=1, key_yellow_card=1, key_red_card=1,
                                                                key_blue_skull=1, key_yellow_skull=1, key_red_skull=1)),
    "ammo huge": set_vitals(selected_weapon_ammo=1e9, ammo_bullets=1e9, ammo_shells=1e9, ammo_rockets=1e9, ammo_cells=1e9),
    # ---- extreme velocities
    "velocity 1e3": velocity(1e3),
    "velocity -1e6": velocity(-1e6),
    "velocity 1e30": velocity(1e30),
    "velocity at the float32 maximum": velocity(F32_MAX),
    "velocity denormal": velocity(1e-42),
    # ---- all-zero and all-one rays
    "rays all zero, wall": all_rays(0.0, "wall"),
    "rays all zero, nothing hit": all_rays(0.0, "none"),
    "rays all one, nothing hit": all_rays(1.0, "none"),
    "rays all one, wall": all_rays(1.0, "wall"),
    "rays all door within reach": all_rays(0.03, "door"),
    "rays all locked door within reach": all_rays(0.03, "locked_door"),
    "rays all enemy": all_rays(0.05, "enemy"),
    "rays all item": all_rays(0.05, "item"),
    # ---- 32 visible entities
    "32 entities of every kind": crowd(*EVERY_KIND),
    "32 enemies": crowd(*ENEMIES),
    "32 enemies and a door straight ahead": together(crowd(*ENEMIES), DOOR_AHEAD),
    "32 projectiles": crowd("projectile"),
    "32 keys": crowd("key"),
    "32 entities at the same point": crowd(*EVERY_KIND, distance=0.0, speed=0.0),
    "32 entities at the 4096-unit limit": crowd(*EVERY_KIND, distance=4.0),
    "32 entities with extreme relative velocity": crowd(*ENEMIES, speed=1e6),
    # ---- screens
    "screen all black (render_buffers=False)": screen(0, 0),
    "screen all white": screen(255, 255),
    "screen all red": screen((255, 0, 0), 128),
    # ---- wrong but castable dtypes (the socket protocol passes any numeric dtype on to the agent)
    "float64 floats": cast(np.float64, FLOAT_LEAVES),
    "float16 floats": cast(np.float16, FLOAT_LEAVES),
    "int32 index fields": cast(np.int32, INDEX_LEAVES),
    "uint8 index fields": cast(np.uint8, INDEX_LEAVES),
    "float32 index fields": cast(np.float32, INDEX_LEAVES),
    "bool mask": cast(np.bool_, ("visible_entities/mask",)),
    "int64 mask and int16 screens": together(cast(np.int64, ("visible_entities/mask",)), cast(np.int16, SCREEN_LEAVES)),
    "float32 screens": cast(np.float32, SCREEN_LEAVES),
    "float64 everything": cast(np.float64, ALL_LEAVES),
    # ---- values outside the documented ranges (the engine never sends these; act() must still answer)
    "hit type 6 and -1": together(put("rays/hit_type", 6, np.s_[::2]), put("rays/hit_type", -1, np.s_[1::4])),
    "entity type 12 and -1": together(crowd(*EVERY_KIND), put("visible_entities/entity_type", 12, np.s_[::2]),
                                     put("visible_entities/entity_type", -1, np.s_[1::4])),
    "mask value 2": together(crowd(*EVERY_KIND), put("visible_entities/mask", 2)),
    "ray distance -1 and 2": together(put("rays/distance", -1.0, np.s_[::2]), put("rays/distance", 2.0, np.s_[1::2])),
    "ray distance NaN": put("rays/distance", np.nan),
    "entity features NaN and inf": together(crowd(*EVERY_KIND), put("visible_entities/features", np.nan, np.s_[::2]),
                                           put("visible_entities/features", np.inf, np.s_[1::2])),
    # ---- several at once
    "NaN vitals, zero rays, 32 enemies, black screen, float64": together(all_vitals(np.nan), all_rays(0.0, "wall"), crowd(*ENEMIES), screen(0, 0),
                                                                        cast(np.float64, FLOAT_LEAVES)),
}

# Odd observations that the engine can really produce: they lie inside the observation space and
# describe a possible situation (a large open arena, a crowd, a full inventory). On these it is not
# enough to answer: the agent must still be acting. Answering every one of them with all zeros
# means an internal error is being swallowed, and the agent would stand still in that situation.
PLAUSIBLE = (
    "health 200, armor 200",
    "weapon slot 0",
    "weapon slot 7",
    "no ammo, every key held",
    "rays all one, nothing hit",
    "32 entities of every kind",
    "32 enemies",
    "32 enemies and a door straight ahead",
    "32 projectiles",
    "32 entities at the 4096-unit limit",
)


# ---------------------------------------------------------------------------------------------
# The agent, in its own process
# ---------------------------------------------------------------------------------------------
class AgentProcess:
    """DoomAgent in its own process behind the scorer's socket protocol (doom_env.remote).

    This is how validate.py and the scorer run the agent: a fresh interpreter, one compute thread,
    one core where the system allows pinning, observations and actions as raw arrays over a local
    socket. reset() and act() raise remote.AgentFailure when the agent raises, answers with anything
    but nine 0/1 values, takes too long or dies; act() returns the action as the scorer accepted it.
    """

    def __init__(self, device: str = "cpu", repo: Path = ROOT):
        self.folder = Path(tempfile.mkdtemp(prefix="doom-fuzz-"))  # short path: socket names are length-limited
        self.server = remote.listen(self.folder / "agent.sock")
        self.log = open(self.folder / "agent.log", "wb")
        self.conn = None
        self.broken = ""  # set after a timeout or a crash: the connection is then out of step
        env = {**os.environ, **limits.THREAD_ENV, "DOOMNET_ALLOW_UNTRAINED": "1", "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(self.server),
             "--repo", str(repo), "--device", device],
            stdout=self.log, stderr=subprocess.STDOUT, env=env, cwd=self.folder)
        try:
            self._pin_to_one_core()
            self.conn = remote.AgentConnection.accept(self.server, limits.INIT_TIMEOUT_S, alive=lambda: self.proc.poll() is None)
            self.conn.init(device, timeout=limits.INIT_TIMEOUT_S)
        except BaseException as error:
            error.agent_output = self.output(30)  # close() deletes the log; keep what the agent printed
            self.close()
            raise

    def _pin_to_one_core(self) -> None:
        """One CPU for the agent, as at evaluation (Linux only; elsewhere the agent keeps all cores)."""
        if hasattr(os, "sched_setaffinity"):
            cores = sorted(os.sched_getaffinity(0))
            try:
                if len(cores) > 1:
                    os.sched_setaffinity(self.proc.pid, {cores[-1]})
            except OSError:
                pass

    def _call(self, method, *args):
        if self.broken:
            raise remote.AgentFailure("crash", f"the agent process was given up after: {self.broken}", "an earlier call")
        try:
            return method(*args)
        except remote.AgentFailure as failure:
            if failure.kind in ("timeout", "crash", "protocol"):
                self.broken = str(failure)
            raise

    def reset(self) -> None:
        self._call(self.conn.reset, limits.RESET_TIMEOUT_S)

    def act(self, observation: dict) -> np.ndarray:
        return self._call(self.conn.act, observation, limits.ACT_TIMEOUT_S)

    def output(self, lines: int = 12) -> str:
        """The last lines the agent process printed."""
        text = (self.folder / "agent.log").read_bytes().decode("utf-8", "replace").strip().splitlines()
        return "\n".join(text[-lines:])

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.server.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        shutil.rmtree(self.folder, ignore_errors=True)


def valid_action(action, where: str) -> np.ndarray:
    """The scorer's own check: nine numbers, each 0 or 1. Returns them as int8."""
    array = np.asarray(action)
    assert array.dtype.kind in "biuf", f"act() returned {action!r}, which is not an array of numbers [{where}]"
    try:
        return remote.validate_action(array)
    except remote.AgentFailure as failure:
        raise AssertionError(f"act() returned an invalid action {action!r}: {failure.message} [{where}]") from None


def clue(agent) -> str:
    """The agent's last output, for failure messages: an agent that swallows an error usually logs it."""
    text = agent.output() if hasattr(agent, "output") else ""
    return f"\nlast output of the agent process:\n{text}" if text else ""


def act(agent, obs: dict, where: str) -> tuple[np.ndarray, float]:
    """One timed act() call. Returns (action, seconds), the time including the trip through the socket.

    Fails if the agent raised, answered with an invalid action, ran out of time or died; `where` names
    the observation. A failure inside the agent process arrives with the agent's own traceback.
    """
    start = time.perf_counter()
    try:
        action = agent.act(obs)
    except Exception as error:
        detail = getattr(error, "traceback", "") or clue(agent)
        raise AssertionError(f"act() failed: {error} [{where}]\n{detail}") from (None if isinstance(error, remote.AgentFailure) else error)
    seconds = time.perf_counter() - start
    return valid_action(action, where), seconds


def presses(agent, rng: np.random.Generator, steps: int, where: str) -> int:
    """How many of `steps` ordinary observations were answered with at least one pressed button."""
    return sum(int(act(agent, ordinary(rng), f"{where}, step {i}")[0].any()) for i in range(steps))


def start_agent(repo: Path = ROOT) -> AgentProcess:
    """Start the agent of `repo` the way the scorer starts it and reset it once.

    An agent that cannot be built or reset fails the test that asked for it. That is never a reason
    to skip: DOOMNET_ALLOW_UNTRAINED=1 is set, so missing weights are no excuse, and at evaluation an
    agent that raises or runs out of time in __init__ scores nothing on every map.
    """
    try:
        process = AgentProcess("cpu", repo)
    except remote.AgentFailure as failure:
        detail = failure.traceback or getattr(failure, "agent_output", "")
        pytest.fail(f"DoomAgent cannot be built: {failure}\n{detail}".rstrip(), pytrace=False)
    try:
        process.reset()
    except remote.AgentFailure as failure:
        detail = failure.traceback or process.output()
        process.close()
        pytest.fail(f"the first reset() failed: {failure}\n{detail}".rstrip(), pytrace=False)
    return process


@pytest.fixture(scope="module")
def agent():
    """One agent process for the whole file. Skipped only while there is no agent.py at all."""
    if not AGENT_PATH.is_file():
        pytest.skip("submission/agent.py does not exist yet")
    process = start_agent()
    try:
        yield process
    finally:
        process.close()


def report(capsys, text: str) -> None:
    """Print a line that is shown even when pytest captures output (CI logs keep the numbers)."""
    with capsys.disabled():
        print(f"\n[fuzz] {text}")


def summary(ms: np.ndarray) -> str:
    """Mean, percentiles and maximum of a list of act() times in milliseconds."""
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return f"mean {ms.mean():.3f}  p50 {p50:.3f}  p95 {p95:.3f}  p99 {p99:.3f}  max {ms.max():.3f} ms over {len(ms)} calls"


# ---------------------------------------------------------------------------------------------
# Tests of the inputs themselves (no agent needed)
# ---------------------------------------------------------------------------------------------
def test_ordinary_observation_matches_the_spaces():
    """Every array ordinary() makes has the dtype and shape of its leaf in the official observation space."""
    space = S.observation_space()
    want = leaves(space)
    assert set(want) == set(ALL_LEAVES)
    rng = np.random.default_rng(0)
    for _ in range(25):
        obs = ordinary(rng)
        assert space.contains(obs)
        got = leaves(obs)
        assert got.keys() == want.keys()
        for path, array in got.items():
            assert (array.dtype, array.shape) == (want[path].dtype, want[path].shape), path
            assert array.flags.writeable and array.flags.c_contiguous, path


def test_ordinary_observation_matches_a_real_observation():
    """Same keys, dtypes, shapes and flags as an observation produced by the engine through the wrapper."""
    try:
        from doom_env import make_env

        env = make_env(ROOT / "maps" / "two_rooms.wad")
    except Exception as error:
        pytest.skip(f"the engine cannot start here ({type(error).__name__}: {error})")
    try:
        real = leaves(env.reset(seed=0)[0])
    finally:
        env.close()
    made = leaves(ordinary(np.random.default_rng(0)))

    def describe(tree: dict) -> dict:
        return {path: (a.dtype, a.shape, a.flags.writeable, a.flags.c_contiguous) for path, a in tree.items()}

    assert describe(made) == describe(real)


@pytest.mark.parametrize("name", list(ODD))
def test_odd_observation_keeps_keys_and_shapes(name):
    """An edit changes values, dtypes or memory layout, never the structure of the observation."""
    rng = np.random.default_rng(0)
    obs = ordinary(rng)
    ODD[name](obs, rng)
    want = leaves(S.observation_space())
    got = leaves(obs)
    assert got.keys() == want.keys()
    for path, array in got.items():
        assert array.shape == want[path].shape, path
        array.astype(want[path].dtype)  # "castable": converting to the contract dtype does not raise
    if name in PLAUSIBLE:
        assert S.observation_space().contains(obs), "a plausible observation must lie inside the observation space"


def test_plausible_names_exist():
    """PLAUSIBLE only names entries of ODD."""
    assert set(PLAUSIBLE) <= set(ODD)


def test_action_check_rejects_what_the_scorer_rejects():
    """The validity check used on every action accepts 0/1 values of any numeric dtype and nothing else."""
    ok = valid_action(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], np.int64), "ok")
    assert ok.dtype == np.int8 and ok.tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 0]
    valid_action([0.0, 1.0, 0, 0, 0, 0, 0, 0, 1], "floats that are 0 or 1")
    valid_action(np.zeros(9, bool), "booleans")
    bad = [np.zeros(8, np.int8), np.zeros((1, 9), np.int8), np.full(9, 2), np.full(9, 0.5), np.full(9, np.nan), np.full(9, -1),
           None, "forward", [None] * 9, 1]
    for action in bad:
        with pytest.raises(AssertionError):
            valid_action(action, "bad")


# ---------------------------------------------------------------------------------------------
# Tests of the agent
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("index", range(len(ODD)), ids=list(ODD))
def test_odd_observations(agent, index, capsys):
    """Several odd observations of one kind in a row, then an ordinary one.

    The agent's state carries over from one kind to the next, except that a plausible kind starts a new
    episode: whether the agent still acts on it must not depend on what was fed before.
    """
    name = list(ODD)[index]
    rng = np.random.default_rng([7, index])
    steps = PLAUSIBLE_STEPS if name in PLAUSIBLE else ODD_STEPS
    if name in PLAUSIBLE:
        agent.reset()
    pressed = 0
    for step in range(steps):
        obs = ordinary(rng)
        ODD[name](obs, rng)
        pressed += int(act(agent, obs, f"{name}, step {step}")[0].any())
    act(agent, ordinary(rng), f"ordinary observation after: {name}")
    if name in PLAUSIBLE:
        assert pressed > 0, (f"{steps} observations of the kind '{name}', which the engine can produce, were all "
                             "answered with all zeros: act() is probably failing on them" + clue(agent))
    elif pressed == 0:
        report(capsys, f"note: '{name}' was answered with all zeros {steps} times in a row; the agent may be swallowing an error on it")


def test_reset_mid_episode(agent, capsys):
    """reset() at any point, twice in a row, and after the state was fed NaN: the agent works normally afterwards."""
    rng = np.random.default_rng(11)
    slowest = 0.0
    for episode, steps in enumerate((40, 1, 25)):
        start = time.perf_counter()
        agent.reset()
        slowest = max(slowest, time.perf_counter() - start)
        for step in range(steps):
            act(agent, ordinary(rng), f"episode {episode}, step {step}")
    agent.reset()
    agent.reset()
    for step in range(5):
        obs = ordinary(rng)
        ODD["vitals all NaN"](obs, rng)
        ODD["ray distance NaN"](obs, rng)
        act(agent, obs, f"NaN observation {step}")
    agent.reset()
    pressed = presses(agent, rng, LIVENESS_STEPS, "after NaN observations and reset()")
    report(capsys, f"reset(): slowest {1000 * slowest:.2f} ms; after NaN and reset() {pressed} of {LIVENESS_STEPS} actions press a button")
    assert slowest < limits.RESET_TIMEOUT_S
    assert pressed > 0, (f"after reset() the agent answered {LIVENESS_STEPS} ordinary observations with all zeros: "
                         "it is failing on every step, or reset() did not clear the bad state" + clue(agent))


BAD_INPUTS = ("vitals all NaN", "ray distance NaN", "entity features NaN and inf", "float32 index fields")


def presses_after_bad_input(agent, seed: int) -> int:
    """Feed the kinds of BAD_INPUTS in mid-episode, then LIVENESS_STEPS ordinary observations with no
    reset() in between. Returns how many of the ordinary ones were answered with a pressed button."""
    rng = np.random.default_rng(seed)
    agent.reset()
    presses(agent, rng, 10, "before the bad observations")
    for name in BAD_INPUTS:
        obs = ordinary(rng)
        ODD[name](obs, rng)
        act(agent, obs, f"{name}, in mid-episode")
    return presses(agent, rng, LIVENESS_STEPS, "after bad observations, without reset()")


def test_agent_keeps_acting_after_bad_input_without_reset(agent, capsys):
    """A bad observation may cost the decision it arrives on, never the rest of the map: the scorer
    calls reset() only between maps, so an agent that stays silent until then stands still for good."""
    pressed = presses_after_bad_input(agent, 13)
    report(capsys, f"after NaN and wrong-dtype observations, without reset(): {pressed} of {LIVENESS_STEPS} actions press a button")
    assert pressed > 0, (f"after bad observations the agent answered {LIVENESS_STEPS} ordinary ones with all zeros: "
                         "its state stays broken until reset()" + clue(agent))


def timed_run(agent, seed: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """`steps` timed act() calls after a warm-up, every second observation odd.

    Returns (milliseconds per call, True where the observation of that call was odd).
    """
    rng = np.random.default_rng(seed)
    names = list(ODD)
    agent.reset()
    ms, odd = [], []
    for i in range(WARMUP_STEPS + steps):
        obs, name = ordinary(rng), "ordinary"
        if i % 2:
            name = names[(i // 2) % len(names)]
            ODD[name](obs, rng)
        seconds = act(agent, obs, f"{name}, step {i} of the timed run")[1]
        if i >= WARMUP_STEPS:
            ms.append(1000 * seconds)
            odd.append(name != "ordinary")
    return np.array(ms), np.array(odd)


LIMITS = f"limits: mean <= {BUDGET_MS:g} ms, p99 <= {BUDGET_MS + P99_SLACK_MS:g} ms"


def over_budget(ms: np.ndarray, budget_ms: float = BUDGET_MS) -> list[str]:
    """The time bounds that the act() times `ms` break, as text; empty when both hold.

    Mean <= budget: the scorer adds up the agent's time over a map and stops the agent when the sum
    passes budget x decisions, so the mean is what decides. p99 <= budget + slack: single slow calls
    must stay rare and small (the build spec's bound).
    """
    mean, p99 = float(ms.mean()), float(np.percentile(ms, 99))
    problems = []
    if mean > budget_ms:
        problems.append(f"the mean, {mean:.3f} ms, is over the {budget_ms:g} ms budget")
    if p99 > budget_ms + P99_SLACK_MS:
        problems.append(f"p99, {p99:.3f} ms, is over the {budget_ms:g} ms budget + {P99_SLACK_MS:g} ms")
    return problems


def test_latency_on_odd_inputs(agent, capsys):
    """Odd inputs must not make act() slow.

    Over a mixed stream, measured after a warm-up: the mean and p99 of all calls are bounded, and so
    is the mean of the calls on odd observations alone, which a fast ordinary half would otherwise
    hide. The machine is shared, so a measurement that is over a bound is repeated once.
    """
    ms, odd = timed_run(agent, 1, LATENCY_STEPS)
    if over_budget(ms) or over_budget(ms[odd]):
        ms, odd = timed_run(agent, 2, LATENCY_STEPS)
    report(capsys, f"act() on mixed ordinary and odd observations: {summary(ms)}; mean on ordinary ones {ms[~odd].mean():.3f} ms, "
                   f"on odd ones {ms[odd].mean():.3f} ms ({LIMITS}, mean on odd ones <= {BUDGET_MS:g} ms)")
    problems = over_budget(ms)
    assert not problems, f"act() is too slow on a mixed stream: {'; '.join(problems)}. {summary(ms)}"
    assert ms[odd].mean() <= BUDGET_MS, (f"odd observations make act() slow: mean {ms[odd].mean():.3f} ms on them against "
                                         f"{ms[~odd].mean():.3f} ms on ordinary ones (budget {BUDGET_MS:g} ms)")


def long_run(agent, seed: int, steps: int) -> tuple[np.ndarray, int]:
    """`steps` ordinary observations without reset(): (milliseconds per call, actions with a button pressed).

    Only ordinary observations are used, so that nothing makes the agent clear its own state on the way
    and its step counters, time-budget inputs and map really do run for `steps` decisions.
    """
    rng = np.random.default_rng(seed)
    agent.reset()
    ms, pressed = [], 0
    for step in range(steps):
        obs = ordinary(rng)
        obs["vitals"][V["vel_forward"]] = 8.0 + rng.normal()  # walking speed on every step: far off any map by the end
        action, seconds = act(agent, obs, f"step {step} of {steps} without reset()")
        ms.append(1000 * seconds)
        pressed += int(action.any())
    return np.array(ms[WARMUP_STEPS:]), pressed


def check_long_run(agent, capsys, steps: int) -> None:
    """A long run must stay valid (checked inside long_run), stay within the time budget and keep acting.

    The machine is shared, so a run that is over a time bound is repeated once.
    """
    ms, pressed = long_run(agent, 3, steps)
    if over_budget(ms):
        ms, pressed = long_run(agent, 4, steps)
    report(capsys, f"{steps} steps without reset(): {pressed} actions press a button; {summary(ms)} ({LIMITS})")
    problems = over_budget(ms)
    assert not problems, f"act() is too slow over {steps} steps without reset(): {'; '.join(problems)}. {summary(ms)}"
    assert pressed > 0, f"{steps} ordinary observations were all answered with all zeros" + clue(agent)


def test_run_past_the_longest_map_without_reset(agent, capsys):
    """Hidden maps run for up to 1,050 decisions, twice the validator's episode, and the long ones
    carry the most weight. This is the run that every push gets, so it covers the longest map."""
    check_long_run(agent, capsys, PAST_LONGEST_MAP_STEPS)


@pytest.mark.slow
def test_1200_steps_without_reset(agent, capsys):
    """Longer than any map: nothing in the agent may overflow or run out when reset() does not come."""
    check_long_run(agent, capsys, LONG_RUN_STEPS)


def test_agent_presses_buttons_on_a_real_map(agent, capsys):
    """100 decisions on maps/two_rooms.wad through the wrapper: valid actions, and not all of them zeros."""
    try:
        from doom_env import make_env

        env = make_env(ROOT / "maps" / "two_rooms.wad")
    except Exception as error:
        pytest.skip(f"the engine cannot start here ({type(error).__name__}: {error})")
    try:
        obs, _ = env.reset(seed=0)
        agent.reset()
        pressed = steps = 0
        for steps in range(1, 101):
            action = act(agent, obs, f"two_rooms.wad, step {steps}")[0]
            pressed += int(action.any())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    report(capsys, f"two_rooms.wad: {pressed} of {steps} actions press a button")
    assert pressed > 0, "the agent stood still for the whole run: act() is probably failing on every step" + clue(agent)


# ---------------------------------------------------------------------------------------------
# The device check. It needs to look inside the agent's interpreter (which modules are loaded, whether
# anything reached for CUDA), so it builds the agent directly, in a new interpreter started for the purpose.
# ---------------------------------------------------------------------------------------------
def load_agent_class(repo: Path = ROOT):
    """Import submission/agent.py of `repo` the way the evaluation server does (doom_env.remote)."""
    for path in (str(repo / "submission"), str(repo)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("submission_agent", repo / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent"] = module
    spec.loader.exec_module(module)
    return module.DoomAgent


def watch_cuda() -> list[str]:
    """Record every attempt to initialise CUDA from now on. Returns the list that fills up, one entry
    per attempt: the line of the agent's code that made it.

    Putting a tensor or a module on a CUDA device makes torch call torch.cuda._lazy_init() first, also
    on a machine without a GPU, where that call then raises. Counting the calls shows an agent that
    tries the GPU and falls back to the CPU when there is none. On a CPU-only machine such an agent
    passes every other check; at evaluation, where there is a GPU, it would run on it.
    """
    attempts: list[str] = []
    try:
        import torch
    except ImportError:
        return attempts  # an agent without torch has no way to a GPU
    real = getattr(torch.cuda, "_lazy_init", None)
    if real is None:
        return attempts  # this torch has no such function; torch.cuda.is_initialized() is still checked
    library = str(Path(torch.__file__).resolve().parent)

    def spy(*args, **kwargs):
        stack = traceback.extract_stack()[:-1]
        own = [f for f in stack if not str(Path(f.filename).resolve()).startswith(library)]  # skip frames inside torch
        frame = (own or stack)[-1]
        attempts.append(f"{Path(frame.filename).name}:{frame.lineno}: {frame.line}")
        return real(*args, **kwargs)

    torch.cuda._lazy_init = spy
    return attempts


def device_check(device: str, repo: Path = ROOT) -> dict:
    """Runs in the new interpreter: build the agent of `repo` with `device`, step it, report what happened."""
    start = time.perf_counter()
    cuda_attempts = watch_cuda()
    cls = load_agent_class(repo)
    agent = cls(S.observation_space(), S.action_space(), device)
    init_seconds = time.perf_counter() - start  # imports included, as the scorer counts it
    agent.reset()
    rng = np.random.default_rng(0)
    actions, seconds = zip(*(act(agent, ordinary(rng), f"device={device}, step {i}") for i in range(LIVENESS_STEPS)))
    torch = sys.modules.get("torch")
    return {
        "init_seconds": init_seconds,
        "first_ms": 1000 * seconds[0],
        "mean_ms": 1000 * float(np.mean(seconds)),
        "pressed": int(sum(a.any() for a in actions)),
        "cuda_attempts": cuda_attempts,
        "cuda_initialized": bool(torch is not None and torch.cuda.is_initialized()),
        "forbidden_modules": sorted(m for m in ("vizdoom", "training", "stable_baselines3") if m in sys.modules),
    }


def run_device_check(device: str, cwd: Path, repo: Path = ROOT) -> tuple[dict | None, str]:
    """Run device_check in a new interpreter, with the scorer's one-thread settings and `cwd` as the
    working directory. Returns (result, problem); the problem is "" when everything is in order."""
    env = {**os.environ, **limits.THREAD_ENV, "DOOMNET_ALLOW_UNTRAINED": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), device, str(repo)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=limits.INIT_TIMEOUT_S + 120)
    lines = [line for line in proc.stdout.splitlines() if line.startswith(MARK)]
    if proc.returncode != 0 or not lines:
        return None, "it did not finish:\n" + "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
    result = json.loads(lines[-1][len(MARK):])
    if result["pressed"] == 0:
        return result, "every action was all zeros, so act() is probably failing on every step"
    if result["cuda_attempts"]:
        return result, (f"it tried to initialise CUDA {len(result['cuda_attempts'])} time(s), first at {result['cuda_attempts'][0]!r}. "
                        "The agent must stay on the CPU whatever `device` says; on a machine without a GPU the attempt "
                        "fails and may be caught, at evaluation it would succeed")
    if result["cuda_initialized"]:
        return result, "it initialised CUDA; the agent must run on the CPU whatever `device` says"
    if result["forbidden_modules"]:
        return result, f"it loaded {result['forbidden_modules']}"
    if result["init_seconds"] >= limits.INIT_TIMEOUT_S:
        return result, f"__init__ took {result['init_seconds']:.1f} s (limit {limits.INIT_TIMEOUT_S:g} s)"
    return result, ""


def test_agent_does_not_depend_on_the_device_argument(agent, tmp_path, capsys):
    """The scorer's machine has a GPU and may pass device="cuda"; every other check here passes "cpu".

    `agent` is requested only so that an agent that cannot be built at all is reported once, by the
    fixture, and not a second time from here.
    """
    result, problem = run_device_check("cuda", tmp_path)
    if problem:
        if run_device_check("cpu", tmp_path)[1]:
            cause = "With device='cpu' the same run fails too, so the device argument is not the cause."
        else:
            cause = "With device='cpu' the same run passes, so the agent depends on the device argument."
        pytest.fail(f"built with device='cuda' the agent failed: {problem}\n{cause}", pytrace=False)
    report(capsys, f"device='cuda', direct calls without the socket: init with imports {result['init_seconds']:.2f} s, "
                   f"first act() {result['first_ms']:.3f} ms, mean {result['mean_ms']:.3f} ms, "
                   f"{result['pressed']} of {LIVENESS_STEPS} actions press a button, "
                   f"attempts to initialise CUDA: {len(result['cuda_attempts'])}, CUDA initialised: {result['cuda_initialized']}")


# ---------------------------------------------------------------------------------------------
# Checks of the checks. Every stand-in agent below has one known defect. It is written to a
# temporary folder laid out like the repository (never to submission/), and the functions that
# judge the real agent must reject it.
# ---------------------------------------------------------------------------------------------
STAND_IN = '''import numpy as np


class Base:
    def __init__(self, observation_space, action_space, device):
        self.rng = np.random.default_rng(0)
        self.steps = 0

    def reset(self):
        self.steps = 0

    def act(self, observation):
        self.steps += 1
        return self.rng.integers(0, 2, 9, dtype=np.int8)
'''

CANNOT_BE_BUILT = {
    # defect: (source of submission/agent.py, words of the agent's own error that the failure must quote)
    "the constructor raises": (STAND_IN + '''

class DoomAgent(Base):
    def __init__(self, *args):
        raise RuntimeError("weights/policy.pt: size mismatch for controller.gru.weight_ih_l0")
''', "size mismatch for controller.gru.weight_ih_l0"),
    "an import fails": ("from doomnet.modle import PolicyNet  # misspelt\n" + STAND_IN + "\n\nclass DoomAgent(Base):\n    pass\n",
                        "ModuleNotFoundError"),
    "there is no DoomAgent class": (STAND_IN, "does not define DoomAgent"),
}

SHORT_BUFFER = STAND_IN + '''

class DoomAgent(Base):
    """Keeps its trail in 700 rows: enough for the shortest map (525 decisions), not for the longest (1,050)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.trail = np.zeros((700, 2), np.float32)

    def act(self, observation):
        self.trail[self.steps] = observation["vitals"][14:16]
        return super().act(observation)
'''

SILENT_AFTER_NAN = STAND_IN + '''

class DoomAgent(Base):
    """One NaN observation and every later answer is all zeros, until reset()."""

    def __init__(self, *args):
        super().__init__(*args)
        self.broken = False

    def reset(self):
        super().reset()
        self.broken = False

    def act(self, observation):
        self.broken = self.broken or not np.isfinite(observation["vitals"]).all()
        return np.zeros(9, np.int8) if self.broken else super().act(observation)
'''

TRIES_THE_GPU = '''import numpy as np
import torch


class DoomAgent:
    """Moves its network to the `device` argument and falls back to the CPU when that fails."""

    def __init__(self, observation_space, action_space, device):
        self.rng = np.random.default_rng(0)
        self.net = torch.nn.Linear(17, 9)
        try:
            self.net = self.net.to(device)
            self.device = device
        except Exception:
            self.device = "cpu"

    def reset(self):
        pass

    def act(self, observation):
        with torch.inference_mode():
            self.net(torch.as_tensor(np.nan_to_num(observation["vitals"]), dtype=torch.float32).to(self.device))
        return self.rng.integers(0, 2, 9, dtype=np.int8)
'''


def stand_in_repo(folder: Path, source: str) -> Path:
    """A folder laid out like the repository whose submission/agent.py is `source`."""
    (folder / "submission").mkdir(parents=True)
    (folder / "submission" / "agent.py").write_text(source)
    return folder


@pytest.mark.parametrize("defect", list(CANNOT_BE_BUILT))
def test_an_agent_that_cannot_be_built_fails_the_tests(tmp_path, defect):
    """It must fail them, with the agent's own error in the message. Skipping would let the whole file
    pass on an agent that scores nothing."""
    source, quoted = CANNOT_BE_BUILT[defect]
    try:
        start_agent(stand_in_repo(tmp_path, source)).close()
    except BaseException as outcome:  # pytest's skip and fail are both BaseException, so they are told apart here
        assert isinstance(outcome, pytest.fail.Exception), f"expected a failure, got {type(outcome).__name__}: {outcome}"
        assert quoted in str(outcome), str(outcome)
    else:
        raise AssertionError(f"an agent of which {defect} was started without complaint")


def test_long_run_catches_a_buffer_shorter_than_the_longest_map(tmp_path):
    """The defect shows at decision 701, which a run only a little longer than the shortest map never reaches."""
    agent = start_agent(stand_in_repo(tmp_path, SHORT_BUFFER))
    try:
        with pytest.raises(AssertionError, match=rf"IndexError.*step 700 of {PAST_LONGEST_MAP_STEPS} without reset"):
            long_run(agent, 3, PAST_LONGEST_MAP_STEPS)
    finally:
        agent.close()


def test_bad_input_check_catches_an_agent_that_stays_silent_until_reset(tmp_path):
    """Every answer of this stand-in is a valid action, and it recovers at reset(): only the count of
    pressed buttons after the bad input, with no reset() in between, gives it away."""
    agent = start_agent(stand_in_repo(tmp_path, SILENT_AFTER_NAN))
    try:
        assert presses_after_bad_input(agent, 13) == 0
        agent.reset()
        assert presses(agent, np.random.default_rng(0), LIVENESS_STEPS, "after reset()") > 0
    finally:
        agent.close()


def test_time_bounds_catch_a_steady_42_ms_per_call():
    """Numbers only. An agent that takes 42.5 ms on every call uses 22.3 s of the 21 s that a
    525-decision map allows, yet its p99 is within 40 + 5 ms: the mean has to be bounded as well."""
    steady = np.full(540, 42.5)
    assert np.percentile(steady, 99) <= 40.0 + P99_SLACK_MS and steady.sum() / 1000 > limits.agent_budget_s(2100) / 525 * 540
    assert [p.split(",")[0] for p in over_budget(steady, 40.0)] == ["the mean"]
    spikes = np.r_[np.full(530, 5.0), np.full(10, 60.0)]  # fast on average, 60 ms on one call in 54
    assert [p.split(",")[0] for p in over_budget(spikes, 40.0)] == ["p99"]
    assert over_budget(np.full(540, 39.9), 40.0) == []
    assert [p.split(",")[0] for p in over_budget(np.full(540, 21.0), 20.0)] == ["the mean"]  # the budget CI sets
    assert [p.split(",")[0] for p in over_budget(np.full(540, 50.0), 40.0)] == ["the mean", "p99"]


def test_device_check_catches_an_agent_that_tries_the_gpu_and_falls_back(tmp_path):
    """Built with device="cuda" on a machine without a GPU, this stand-in ends up on the CPU and
    answers every observation. Only the record of its attempt gives it away."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    result, problem = run_device_check("cuda", tmp_path, stand_in_repo(tmp_path / "repo", TRIES_THE_GPU))
    assert "CUDA" in problem and "it did not finish" not in problem, problem
    assert result["pressed"] > 0 and (result["cuda_attempts"] or result["cuda_initialized"]), result
    if result["cuda_attempts"]:
        assert "agent.py" in result["cuda_attempts"][0] and "self.net.to(device)" in result["cuda_attempts"][0], result


if __name__ == "__main__":  # python test_fuzz.py DEVICE [REPOSITORY]
    print(MARK + json.dumps(device_check(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT)))
