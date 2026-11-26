"""Tests for the fair navigation teacher (training/teacher).

Four groups:

* fairness: the label is a function of what the student network receives. A recorded episode is
  replayed with the environment gone and must give the same labels; the package may not even
  name the sources of ground truth.
* robustness: labelling trajectories driven by a random policy, and labelling garbage, never
  raises and always returns a well-formed label.
* behaviour on small hand-drawn maps: doors without reversals, key then locked door, a locked
  door that the held key does not fit, a pickup lying in front of a door, an exit in an alcove,
  exploring before leaving, leaving in time.
* the gate of the build spec on the monster-free rebuild of the dev suite (slow).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from doom_env import make_env
from training import paths  # noqa: F401
from doomnet import spec
from doomnet.encode import Encoder
from doomnet.memory import MemoryMap
from doomnet.odometry import Odometry
from training.mapgen import udmf
from training.mapmeta import load_meta
from training.runner import RandomPolicy, run_episodes
from training.teacher import Teacher, TeacherLabel, TeacherPolicy, core, gate, grid, steer
from training.teacher.gate import GateRecorder

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "training" / "teacher"
MAPS = ROOT / "data" / "teacher" / "tests"
HEADER = "Training-time scaffolding. Never imported by the agent in submission/."

# Hand-drawn maps (legend of training.mapgen.udmf: P start, E exit, D door, R/B locked doors, r/b keys, h stimpack).
THREE_DOORS = "################\n#P..D...D...D.E#\n################"
KEY_THEN_LOCK = "\n".join([
    "###########",
    "#P....#...#",
    "#.....R..E#",
    "#.....#...#",
    "#..r..#####",
    "###########"])
WRONG_KEY = "\n".join([
    "############",
    "#P.....#..b#",
    "#......B...#",
    "#..r...#####",
    "#......D..E#",
    "############"])
ITEM_AT_DOOR = "############\n#P...hD...E#\n############"
ALCOVE_EXIT = "\n".join([
    "#############",
    "#####E#######",
    "#P..........#",
    "#############"])
EXPLORE_FIRST = "\n".join([
    "#############",
    "#P.....#....#",
    "#...E..D....#",
    "#......#....#",
    "#############"])
FAR_ROOM = "\n".join([
    "##########################",
    "#E.P.....................#",
    "########################.#",
    "#........................#",
    "#.########################",
    "#........................#",
    "##########################"])
WITH_MONSTERS = "\n".join([
    "##############",
    "#P.....#..z..#",
    "#..h...D.....#",
    "#......#..i.E#",
    "##############"])


# ------------------------------------------------------------------------------------ helpers
def dev_map(name: str) -> tuple[str, int]:
    """(monster-free grid, tick cap) of a map of the frozen dev suite."""
    for line in (ROOT / "training" / "suites" / "dev_v1.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["name"] == name:
            return gate.strip_monsters(row["grid"]), int(row["episode_ticks"])
    raise KeyError(name)


def wad_of(grid_text: str, **options) -> str:
    name = hashlib.sha1((grid_text + repr(sorted(options.items()))).encode()).hexdigest()[:16]
    return str(udmf.write_wad(grid_text, MAPS / f"{name}.wad", **options))


def play(grid_text: str, ticks: int = 2100, **policy_kwargs) -> tuple[dict, TeacherPolicy]:
    """One episode of the teacher on a hand-drawn map, scored by the runner; budget = the true cap."""
    policy = TeacherPolicy(**policy_kwargs)
    rows = run_episodes(policy, [{"wad": wad_of(grid_text), "id": "test", "episode_ticks": ticks}], workers=0,
                        seeds=[0], record=GateRecorder(), render_buffers=False)
    return rows[0], policy


def lean(obs: dict) -> dict:
    """The parts of an observation the agent's front end reads, copied (no frame, no depth)."""
    return {"vitals": np.array(obs["vitals"]),
            "rays": {k: np.array(v) for k, v in obs["rays"].items()},
            "visible_entities": {k: np.array(v) for k, v in obs["visible_entities"].items()}}


class FrontEnd:
    """Odometry -> memory map -> encoder -> teacher, exactly as the training loop will call them."""

    def __init__(self, **teacher):
        self.odom, self.mem, self.enc, self.teacher = Odometry(), MemoryMap(), Encoder(), Teacher(**teacher)
        self.step, self.prev = 0, None

    def label(self, obs, exit_obs, budget) -> TeacherLabel:
        if self.step > 0:
            self.odom.update(self.prev, obs)
        pose = (self.odom.x, self.odom.y, self.odom.theta)
        self.mem.update(pose, obs, self.prev, exit_obs, self.step)
        crop = self.mem.crop(pose)
        scal = self.enc.encode(obs, pose, self.prev, self.step, budget, exit_obs)["scal"]
        return self.teacher.label(obs, crop, pose, scal, self.step, budget, self.prev)

    def took(self, action) -> None:
        self.prev, self.step = np.asarray(action).astype(np.int8), self.step + 1


def assert_valid(label: TeacherLabel) -> None:
    assert isinstance(label, TeacherLabel)
    assert 0 <= int(label.subgoal) < spec.NUM_SUBGOALS
    assert 0 <= int(label.target) < spec.CROP_SIZE * spec.CROP_SIZE
    assert label.heads.shape == (6,) and label.heads.dtype == np.int64
    assert all(0 <= int(h) < n for h, n in zip(label.heads, spec.HEAD_SIZES))
    assert label.head_mask.shape == (6,) and label.head_mask.dtype == np.bool_
    assert not label.head_mask[spec.HEAD_ATTACK] and not label.head_mask[spec.HEAD_WEAPON]
    assert int(label.heads[spec.HEAD_ATTACK]) == 0 and int(label.heads[spec.HEAD_WEAPON]) == 0
    assert label.value_field.shape == (spec.CROP_SIZE, spec.CROP_SIZE) and label.value_field.dtype == np.float32
    assert np.isfinite(label.value_field).all() and float(label.value_field.min()) >= 0.0
    assert float(label.value_field.max()) <= core.VALUE_CAP
    assert isinstance(label.field_valid, bool)
    if label.subgoal == core.FIGHT:
        assert not label.head_mask.any()
    buttons = spec.heads_to_buttons(label.heads)
    assert buttons.shape == (9,) and set(np.unique(buttons)) <= {0, 1}


def same_label(a: TeacherLabel, b: TeacherLabel) -> bool:
    return (a.subgoal == b.subgoal and a.target == b.target and a.field_valid == b.field_valid
            and np.array_equal(a.heads, b.heads) and np.array_equal(a.head_mask, b.head_mask)
            and np.array_equal(a.value_field, b.value_field))


# ----------------------------------------------------------------------------------- fairness
def test_files_carry_the_header_and_name_no_ground_truth():
    """Every file says what it is, and none mentions the engine's ground truth or the map file reader."""
    banned = ["privileged" + "_state", "_geo" + "metry", "sector" + "_polygons", "map" + "meta",
              "a" + "tan2", "arc" + "tan2"]
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 6
    for path in files:
        text = path.read_text()
        assert text.startswith('"""' + HEADER), f"{path.name} does not start with the scaffolding header"
        for token in banned:
            assert token not in text, f"{path.name} mentions {token}"


def test_label_signature_is_the_students_inputs():
    names = list(inspect.signature(Teacher.label).parameters)
    assert names == ["self", "obs", "crop", "pose", "scal", "step", "budget_steps", "prev_action"]
    assert not hasattr(Teacher, "attach")          # the teacher itself can never be handed an environment


def test_label_is_unchanged_when_the_environment_is_withheld():
    """Play a key-and-doors map with the stand-alone policy, then recompute every label from the recorded
    observations and exit observations alone: no environment object, no ground truth, a fresh teacher."""
    grid_text, ticks = dev_map("dev_v1_t3_000")
    wad = wad_of(grid_text)
    budget = ticks / spec.FRAME_SKIP
    env = make_env(wad, episode_ticks=ticks, render_buffers=False)
    policy = TeacherPolicy(exit_source="noisy")
    obs, _ = env.reset(seed=3)
    policy.attach(env, load_meta(wad))
    policy.seed(3)
    policy.reset()
    policy.set_budget(budget)
    trace = []
    for _ in range(600):
        action = policy.act(obs)
        trace.append((lean(obs), None if policy.exit_obs is None else np.array(policy.exit_obs), policy.label, action.copy()))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    del env, policy
    assert len(trace) > 100 and info["exit_reached"]
    assert len({label.subgoal for _, _, label, _ in trace}) >= 5          # the episode exercised most sub-goals
    assert any(exit_obs is not None for _, exit_obs, _, _ in trace)

    replay = FrontEnd()
    for step, (obs, exit_obs, want, action) in enumerate(trace):
        got = replay.label(obs, exit_obs, budget)
        assert same_label(got, want), f"label differs at step {step}"
        assert_valid(got)
        replay.took(action)
    assert replay.teacher.errors == 0


def test_label_from_scalars_is_the_same_label():
    """The step, the budget and the previous action are all in the encoder's scalar vector: a teacher
    handed only the network's inputs gives the labels of a teacher handed them as arguments."""
    wad = wad_of(KEY_THEN_LOCK)
    env = make_env(wad, episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    odom, mem, enc, by_args, by_scalars = Odometry(), MemoryMap(), Encoder(), Teacher(), Teacher()
    prev, done, step = None, False, 0
    while not done and step < 400:
        if step > 0:
            odom.update(prev, obs)
        pose = (odom.x, odom.y, odom.theta)
        mem.update(pose, obs, prev, None, step)
        crop = mem.crop(pose)
        scal = enc.encode(obs, pose, prev, step, 525.0, None)["scal"]
        want = by_args.label(obs, crop, pose, scal, step, 525.0, prev)
        got = by_scalars.label_from_scalars(obs, crop, pose, scal)
        assert same_label(got, want), f"label differs at step {step}"
        prev = spec.heads_to_buttons(want.heads)
        obs, _, terminated, truncated, _ = env.step(prev)
        done, step = terminated or truncated, step + 1
    env.close()
    assert step > 50 and by_args.errors == 0 and by_scalars.errors == 0


def test_teacher_policy_touches_the_environment_only_through_the_exit_sensor():
    """With the exit sensor switched off the policy plays without ever being given an environment."""
    wad = wad_of(THREE_DOORS)
    env = make_env(wad, episode_ticks=2100, render_buffers=False)
    policy = TeacherPolicy(exit_source="none")
    obs, _ = env.reset(seed=0)
    policy.reset()                                   # no attach()
    done, steps = False, 0
    while not done and steps < 400:
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        done, steps = terminated or truncated, steps + 1
    env.close()
    assert info["exit_reached"] and policy.teacher.errors == 0
    source = (PACKAGE / "policy.py").read_text()
    uses = [m.start() for m in re.finditer(r"self\._env\b", source)]
    sensor = source.index("class ExitSensor")
    teacher_policy = source.index("class TeacherPolicy")
    assert uses and all(sensor < u < teacher_policy for u in uses)


# --------------------------------------------------------------------------------- robustness
@pytest.mark.parametrize("grid_text,seed", [(WITH_MONSTERS, 0), (KEY_THEN_LOCK, 1), (THREE_DOORS, 2)])
def test_labels_on_a_random_policy_trajectory(grid_text, seed):
    """The student will drive: label a trajectory the teacher did not choose."""
    wad = wad_of(grid_text)
    env = make_env(wad, episode_ticks=1200, render_buffers=False)
    driver = RandomPolicy(p=0.4, attack=False)
    driver.seed(seed)
    driver.reset()
    front = FrontEnd()
    obs, _ = env.reset(seed=seed)
    subgoals = set()
    for _ in range(300):
        label = front.label(lean(obs), None, 300.0)
        assert_valid(label)
        subgoals.add(label.subgoal)
        action = driver.act(obs)
        front.took(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert front.teacher.errors == 0, front.teacher.last_error
    if grid_text is WITH_MONSTERS:
        assert front.step > 50


def test_fight_label_names_the_enemy_and_masks_every_head():
    wad = wad_of("#########\n#P....z.#\n#########")
    env = make_env(wad, episode_ticks=400, render_buffers=False)
    obs, _ = env.reset(seed=0)
    front = FrontEnd()
    label = front.label(lean(obs), None, 100.0)
    env.close()
    assert label.subgoal == core.FIGHT and not label.head_mask.any()
    assert label.target == grid.CENTER * grid.N + grid.CENTER + 5 * grid.N       # five cells ahead of the agent
    assert_valid(label)


def test_garbage_never_raises():
    rng = np.random.default_rng(0)
    teacher = Teacher()
    obs = {"vitals": np.zeros(17, np.float32),
           "rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
           "visible_entities": {"features": np.zeros((32, 5), np.float32), "entity_type": np.zeros(32, np.int64),
                                "mask": np.zeros(32, np.int8)}}
    scal = np.zeros(spec.SCALAR_DIM, np.float32)
    crop = np.zeros((spec.MAP_CHANNELS, 64, 64), np.float32)
    cases = [
        (obs, crop, (0.0, 0.0, 0.0)),
        (obs, crop, (float("nan"), 0.0, 0.0)),
        (obs, crop, (1.0e4, -1.0e4, 2.0)),                       # 10,000 units from the start
        (obs, crop, (1.0e9, 0.0, float("inf"))),
        (obs, rng.random((spec.MAP_CHANNELS, 64, 64)).astype(np.float32), (37.0, -410.0, 1.0)),
        (obs, np.ones((spec.MAP_CHANNELS, 64, 64), np.float32), (0.0, 0.0, 0.0)),
    ]
    odd = {"vitals": np.full(17, np.nan, np.float32),
           "rays": {"distance": rng.random(64).astype(np.float32), "hit_type": rng.integers(0, 6, 64).astype(np.float64)},
           "visible_entities": {"features": np.full((32, 5), np.inf, np.float32),
                                "entity_type": rng.integers(0, 12, 32).astype(np.float64), "mask": np.ones(32, np.int8)}}
    cases += [(odd, rng.random((spec.MAP_CHANNELS, 64, 64)).astype(np.float32), (5.0, 5.0, 0.3)),
              (odd, crop, (-3000.0, 2500.0, -2.0))]
    for step, (o, c, pose) in enumerate(cases * 3):
        action = (rng.random(9) < 0.5).astype(np.int8) if step else None
        assert_valid(teacher.label(o, c, pose, scal, step, 525, action))
    teacher.reset()
    assert teacher.errors == 0 and teacher.plan is None


def test_reset_forgets_the_episode():
    front = FrontEnd()
    wad = wad_of(KEY_THEN_LOCK)
    env = make_env(wad, episode_ticks=800, render_buffers=False)
    obs, _ = env.reset(seed=0)
    first = front.label(lean(obs), None, 200.0)
    for _ in range(40):
        action = spec.heads_to_buttons(front.label(lean(obs), None, 200.0).heads) if front.step else spec.heads_to_buttons(first.heads)
        front.took(action)
        obs, *_ = env.step(action)
    env.close()
    front.teacher.reset()
    t = front.teacher
    assert t.plan is None and not t.failed and not t.presses and not t._key_cells and not t._looked.any()
    assert not t._stood.any() and not t._dismissed.any() and not t._checks
    assert not t._doorways.any() and t._held == 0 and t.searches == 0


# ------------------------------------------------------------------------- planning and steering
def blank_crop() -> np.ndarray:
    return np.zeros((spec.MAP_CHANNELS, grid.N, grid.N), np.float32)


def test_search_costs_doors_corners_and_pads():
    c, crop, cell = grid.CENTER, blank_crop(), grid.STEPS_PER_CELL
    crop[spec.CH_FREE, c:c + 7, c] = 1.0                       # a corridor of seven cells along +x
    crop[spec.CH_WALL, c:c + 7, c - 1] = 1.0
    crop[spec.CH_WALL, c:c + 7, c + 1] = 1.0
    crop[spec.CH_FREE, c + 3, c] = 0.0
    crop[spec.CH_DOOR, c + 3, c] = 1.0                         # a closed door in the middle
    g = grid.PlanGrid(crop, None, door_steps=12.0)
    dist, _ = g.from_agent()
    own = grid.OWN
    assert dist[own + 2 * grid.N] == pytest.approx(2 * cell)
    assert dist[own + 3 * grid.N] == pytest.approx(3 * cell + 12.0)       # the door costs its opening time
    assert dist[own + 6 * grid.N] == pytest.approx(6 * cell + 12.0)
    field, link = g.to_target(own + 6 * grid.N)
    assert field[own] == pytest.approx(dist[own + 6 * grid.N])           # the same cost in both directions
    assert link[own] == own + grid.N and link[own + 6 * grid.N] == -1

    crop = blank_crop()                                        # an L-shaped corridor: no diagonal across the corner
    crop[spec.CH_FREE, c:c + 3, c] = 1.0
    crop[spec.CH_FREE, c + 2, c:c + 3] = 1.0
    g = grid.PlanGrid(crop)
    dist, _ = g.from_agent()
    assert dist[own + 2 * grid.N + 2] == pytest.approx(4 * cell)
    crop[spec.CH_FREE, c:c + 3, c:c + 3] = 1.0                 # an open square: the diagonal is allowed
    dist, _ = grid.PlanGrid(crop).from_agent()
    assert dist[own + 2 * grid.N + 2] == pytest.approx(2 * cell * math.sqrt(2.0))

    crop = blank_crop()                                        # an exit pad in a corridor: reached, never crossed
    crop[spec.CH_FREE, c:c + 5, c] = 1.0
    crop[spec.CH_EXIT, c + 2, c] = 1.0
    g = grid.PlanGrid(crop)
    dist, _ = g.from_agent()
    assert dist[own + 2 * grid.N] == pytest.approx(2 * cell) and dist[own + 3 * grid.N] == grid.INF
    crop[spec.CH_EXIT, c + 2, c] = 0.5                         # a single sighting does not close the corridor ...
    g = grid.PlanGrid(crop, pad_floor=grid.EXIT_SURE)
    assert g.from_agent()[0][own + 3 * grid.N] == pytest.approx(3 * cell)
    crop[spec.CH_EXIT, c + 2, c + 1] = 0.5                     # ... but a second one in a neighbouring cell confirms it
    g = grid.PlanGrid(crop, pad_floor=grid.EXIT_SURE)
    assert g.from_agent()[0][own + 3 * grid.N] == grid.INF


def test_locked_doors_frontier_dead_ends_and_rooms():
    c, crop = grid.CENTER, blank_crop()
    crop[spec.CH_FREE, c:c + 3, c] = 1.0
    crop[spec.CH_LOCKED, c + 3, c] = 1.0
    crop[spec.CH_FREE, c + 4:c + 6, c] = 1.0
    crop[spec.CH_WALL, c - 1, c] = 1.0
    crop[spec.CH_WALL, c:c + 6, c - 1] = 1.0
    crop[spec.CH_WALL, c:c + 6, c + 1] = 1.0
    crop[spec.CH_VISITED, c, c] = 1.0
    own = grid.OWN
    g = grid.PlanGrid(crop, None)
    assert g.from_agent()[0][own + 4 * grid.N] == grid.INF                    # no key: the lock is a wall
    g = grid.PlanGrid(crop, np.ones((grid.N, grid.N), bool))
    assert g.from_agent()[0][own + 4 * grid.N] < grid.INF                     # a key is held
    assert [tuple(v) for v in np.argwhere(g.frontier())] == [(c + 5, c)]      # the far end is open to the unknown
    assert [tuple(v) for v in np.argwhere(g.dead_ends())] == [(c, c)]         # the start is closed on three sides
    doorway = g.locked.copy()
    rooms = g.unvisited_rooms(doorway)
    assert rooms[c + 4, c] and rooms[c + 5, c] and not rooms[c + 1, c]         # the room behind the door was never entered


def test_crop_origin_agrees_with_the_memory_map():
    memory = pytest.importorskip("doomnet.memory")
    rng = np.random.default_rng(1)
    for x, y in rng.uniform(-3000, 3000, (200, 2)).tolist() + [[31.99, -32.0], [32.0, 95.9], [-32.01, 0.0]]:
        assert grid.crop_origin(x, y) == tuple(memory.crop_origin((x, y)))
        cx, cy = grid.cell_of(x, y)
        assert abs(x - cx * grid.CELL) <= grid.CELL / 2 and abs(y - cy * grid.CELL) <= grid.CELL / 2


def test_turning_never_overshoots():
    """With the engine's turn model (7 degrees on the first step, 14 later) the heading settles within
    half a slow step of any target and then stays put."""
    for target in range(0, 1024, 37):
        heading, tic, held, turns = 0, 8, 0, []
        for _ in range(40):
            theta = heading * 2.0 * math.pi / 1024.0
            want = target * 2.0 * math.pi / 1024.0
            head = steer.turn_head(math.cos(theta), math.sin(theta), math.cos(want), math.sin(want), tic, held)
            turns.append(head)
            if head:
                heading = (heading + steer.turn_increment(tic, held) * (1 if head == 1 else -1)) % 1024
                held += 4
            else:
                held = 0
            tic += 4
        error = (heading - target + 512) % 1024 - 512
        assert abs(error) <= 10, (target, error)
        assert turns[-6:] == [0] * 6, f"still turning towards {target}"


def test_driving_stops_near_the_point_without_hunting():
    """Point-mass check of the stop control: from rest and at full speed the body ends within one key
    step of the point and the keys go quiet."""
    for start, speed in ((-200.0, 0.0), (-90.0, 8.0), (-40.0, 7.5), (150.0, -8.0), (-20.0, 0.0)):
        x, vx, pushes = start, speed, []
        for _ in range(60):
            push = steer.axis_push(0.0 - x, vx)
            pushes.append(push)
            moved, vx = steer.advance(vx, push)
            x += moved
        assert abs(x) <= 20.0, (start, speed, x)
        assert pushes[-10:] == [0] * 10, (start, speed)


def test_swept_segment_respects_walls():
    c = grid.CENTER
    plain = np.zeros((grid.N, grid.N), bool)
    plain[c:c + 4, c] = True                                    # corridor along +x, then a turn to +y
    plain[c + 3, c:c + 4] = True
    flat = plain.reshape(-1).tolist()
    ox = oy = -c
    assert steer.swept_clear(flat, ox, oy, 0.0, 0.0, 192.0, 0.0, ())
    assert not steer.swept_clear(flat, ox, oy, 0.0, 0.0, 192.0, 128.0, ())       # across the inner corner
    assert not steer.swept_clear(flat, ox, oy, 150.0, 0.0, 192.0, 128.0, ())     # still clipping it
    assert steer.swept_clear(flat, ox, oy, 186.0, 0.0, 192.0, 128.0, ())         # on the new centre line
    assert not steer.swept_clear(flat, ox, oy, 0.0, 17.0, 192.0, 17.0, ())       # hugging the wall


# ---------------------------------------------------------------------------------- behaviour
def test_three_doors_without_a_reversal():
    row, policy = play(THREE_DOORS)
    assert row["exit"] == 1 and row["door_reversals"] == 0 and row["doors_opened"] == 3
    assert row["teacher_errors"] == 0 and row["agent_error"] == ""
    assert row["sectors_visited"] == row["sectors_total"]
    assert row["odom_err_max"] < 4.0


def test_key_then_locked_door_then_exit():
    row, policy = play(KEY_THEN_LOCK)
    assert row["exit"] == 1 and row["keys_picked"] == 1 and row["locked_opened"] == 1 and row["door_reversals"] == 0
    assert policy.counts[core.GET_KEY] > 0 and policy.counts[core.OPEN_LOCKED] > 0 and policy.counts[core.GO_EXIT] > 0


def test_a_lock_the_key_does_not_fit_is_given_up():
    row, policy = play(WRONG_KEY)
    assert row["exit"] == 1 and row["keys_picked"] == 1 and row["locked_opened"] == 0
    assert row["locks_given_up"] == 1 and row["door_reversals"] == 0
    assert row["doors_opened"] == 1


def test_a_pickup_in_front_of_a_door_does_not_block_it():
    row, _ = play(ITEM_AT_DOOR)                # the stimpack is not picked up at full health and hides the door
    assert row["exit"] == 1 and row["doors_opened"] == 1 and row["door_reversals"] == 0


def test_exit_in_an_alcove_is_found():
    row, policy = play(ALCOVE_EXIT)
    assert row["exit"] == 1
    assert policy.counts[core.SWEEP] > 0       # found by looking into the dead end, not by a frontier


def test_explores_the_other_room_before_leaving():
    row, policy = play(EXPLORE_FIRST)          # the exit is in the start room, in plain view
    assert row["exit"] == 1 and row["doors_opened"] == 1
    assert row["sectors_visited"] == row["sectors_total"]
    assert 0 <= row["exit_step"] < 20 < row["leave_step"]


def test_leaves_for_the_exit_in_time():
    """A long way to explore and a short cap: the teacher has to turn round and come back."""
    row, policy = play(FAR_ROOM, ticks=700)    # 175 steps; walking the whole map takes about 150 one way
    assert row["exit"] == 1, row["cause"]
    assert 0 <= row["exit_step"] < row["leave_step"] and row["cells_visited"] < 60
    assert row["steps"] > 100                  # it used the time it had before turning round
    row, _ = play(FAR_ROOM, ticks=2800)        # with time to spare it walks the whole map first
    assert row["exit"] == 1 and row["cells_visited"] >= 60


BEHIND_A_DOOR = "\n".join([
    "#############",
    "#P.....#....#",
    "#......D..E.#",
    "#......#....#",
    "#############"])
LONG_CORRIDOR = "###########\n#P.......E#\n###########"


class TamperedSensor:
    """Wraps the policy's exit sensor and adds false detections, placed with the policy's own pose estimate.

    once=(step, forward, left): one false sighting on that step.
    ghost=(x, y): a spot in the start frame that is reported as the exit whenever it is in view,
    like a patch of red wall that fools a detector on every frame; it hides the real pad then.
    """

    def __init__(self, policy, once=None, ghost=None):
        self.policy, self.inner, self.once, self.ghost = policy, policy.sensor, once, ghost
        self.injected = 0

    def attach(self, env, meta) -> None:
        self.inner.attach(env, meta)

    def seed(self, seed: int) -> None:
        self.inner.seed(seed)

    def read(self):
        true = self.inner.read()
        if self.once is not None and self.policy.step == self.once[0]:
            self.injected += 1
            return np.array([1.0, self.once[1] / spec.RAY_RANGE, self.once[2] / spec.RAY_RANGE], np.float32)
        if self.ghost is not None:
            x, y, theta = self.policy.pose_estimate()
            dx, dy = self.ghost[0] - x, self.ghost[1] - y
            fwd, left = dx * math.cos(theta) + dy * math.sin(theta), -dx * math.sin(theta) + dy * math.cos(theta)
            if fwd > 16.0 and abs(left) <= 0.9 * fwd and math.hypot(fwd, left) <= 640.0:
                self.injected += 1
                return np.array([1.0, fwd / spec.RAY_RANGE, left / spec.RAY_RANGE], np.float32)
        return true


def play_tampered(grid_text: str, ticks: int = 2100, **tamper) -> tuple[dict, TeacherPolicy]:
    policy = TeacherPolicy()
    policy.sensor = TamperedSensor(policy, **tamper)
    rows = run_episodes(policy, [{"wad": wad_of(grid_text), "id": "test", "episode_ticks": ticks}], workers=0,
                        seeds=[0], record=GateRecorder(), render_buffers=False)
    return rows[0], policy


def test_a_single_false_exit_sighting_is_looked_at_again_and_dropped():
    """One false detection on open floor: the teacher looks at the spot from close by, gets no second
    sighting, drops it and goes on to the real exit behind the door."""
    row, policy = play_tampered(BEHIND_A_DOOR, once=(4, 200.0, -60.0))
    assert policy.sensor.injected == 1
    assert row["exit"] == 1 and row["doors_opened"] == 1 and row["teacher_errors"] == 0
    assert int(policy.teacher._dismissed.sum()) == 1
    assert max(policy.teacher._checks.values()) == core.CHECK_LOOKS
    plain, _ = play(BEHIND_A_DOOR)
    assert row["steps"] <= plain["steps"] + 40          # the check costs a few steps, not a detour


def test_a_persistent_false_exit_is_walked_over_once_and_then_ignored():
    """A spot that the detector reports as the exit on every frame, in the only corridor, hiding the real
    pad behind it: with nothing else to do the teacher goes there, stands on it, learns that the
    episode goes on, and never treats that cell as the exit again."""
    row, policy = play_tampered(LONG_CORRIDOR, ghost=(256.0, 0.0))
    assert policy.sensor.injected > 10
    assert row["exit"] == 1 and row["teacher_errors"] == 0, row["cause"]
    assert policy.teacher._stood[4 + core.HALF, core.HALF]
    assert row["steps"] < 200


def test_use_is_never_asked_for_against_the_mask_and_a_slipping_driver_reverses_no_door():
    """A stand-in for a student that drives: about four steps in ten are random (without USE). The
    teacher keeps labelling, its USE labels never fall on a masked step, and no door is reversed."""
    policy = TeacherPolicy(slip=0.15, slip_use=0.0)
    rows = run_episodes(policy, [{"wad": wad_of(THREE_DOORS), "id": "test", "episode_ticks": 4200}], workers=0,
                        seeds=[0, 1, 2], record=GateRecorder(), render_buffers=False)
    for row in rows:
        assert row["door_reversals"] == 0 and row["use_masked"] == 0 and row["teacher_errors"] == 0
        assert row["slipped"] > 0.25 * row["steps"]
    assert sum(r["exit"] for r in rows) >= 2
    row, _ = play(THREE_DOORS)
    assert row["use_masked"] == 0 and row["slipped"] == 0


def test_view_range_follows_the_detector():
    """A detector that only reaches 200 units: with `view_range` to match, the far end of the corridor is
    not taken as inspected from the start, and the exit is still found."""
    assert Teacher(view_range=10.0).view_range == 2 * grid.CELL and Teacher().view_range == core.VIEW_RANGE
    policy = TeacherPolicy(exit_range=200.0, view_range=200.0)
    rows = run_episodes(policy, [{"wad": wad_of(LONG_CORRIDOR), "id": "test", "episode_ticks": 2100}], workers=0,
                        seeds=[0], record=GateRecorder(), render_buffers=False)
    assert rows[0]["exit"] == 1 and rows[0]["teacher_errors"] == 0
    assert rows[0]["exit_step"] > 10               # not known from the start cell, 512 units away


# ------------------------------------------------------ review fixes: looking, patience, grid fit
def open_rays(reach: float = 1024.0, wall_ahead: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ray arrays of an empty space; with `wall_ahead` every ray inside the view ends on a wall that far away."""
    dist = np.full(spec.NUM_RAYS, reach / spec.RAY_RANGE, np.float32)
    hit = np.zeros(spec.NUM_RAYS, np.int64)
    if wall_ahead is not None:
        dist[core.VIEW_RAYS] = wall_ahead / spec.RAY_RANGE
        hit[core.VIEW_RAYS] = spec.HIT_WALL
    return dist, hit


def looked_cells(teacher: Teacher, x: float, y: float, heading_deg: float, rays, times: int = core.LOOKS) -> set:
    theta = math.radians(heading_deg)
    for _ in range(times):
        teacher._mark_looked(rays[0], rays[1], x, y, math.cos(theta), math.sin(theta))
    return {(int(a) - core.HALF, int(b) - core.HALF) for a, b in np.argwhere(teacher._looked)}


def test_a_cell_counts_as_looked_at_only_where_the_exit_sensor_would_see_a_pad():
    """The exit labels report a pad whose centre is within 42 degrees of the view axis, in line of sight
    and within range. The teacher's own record of what it has looked at must not be wider than that."""
    seen = looked_cells(Teacher(), 0.0, 0.0, 0.0, open_rays())
    assert {(1, 0), (5, 0), (9, 0), (2, 1), (2, -1), (5, 4), (5, -4)} <= seen
    assert not seen & {(0, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (10, 0), (5, 5)}     # own cell, 45 degrees or more, too far
    for cx, cy in seen:                                                               # every one inside the sensor's cone
        assert cx > 0 and abs(cy) <= cx * math.tan(math.radians(40.0)) + 1e-9 and math.hypot(cx, cy) * 64.0 <= core.VIEW_RANGE
    # fewer looks than `looks` do not count (a detector that misses one frame in five)
    assert not looked_cells(Teacher(), 0.0, 0.0, 0.0, open_rays(), times=core.LOOKS - 1)
    # a wall 100 units ahead hides everything behind it
    assert looked_cells(Teacher(), 0.0, 0.0, 0.0, open_rays(wall_ahead=100.0)) == {(1, 0)}
    # The case from the review: the body 49 units from a pad centre that is 54 degrees off the view
    # axis. The sensor returns nothing there, so the cell must not count as inspected.
    theta = 0.0
    px, py = 49.0 * math.cos(math.radians(54.0)), 49.0 * math.sin(math.radians(54.0))
    teacher = Teacher()
    assert (0, 0) not in looked_cells(teacher, -px, -py, math.degrees(theta), open_rays())
    assert (0, 0) in looked_cells(teacher, -px, -py, 54.0 - 30.0, open_rays())          # turned to within 30 degrees: seen
    # a detector with a shorter reach
    assert max(cx for cx, _ in looked_cells(Teacher(view_range=200.0), 0.0, 0.0, 0.0, open_rays())) == 3


def test_patience_runs_only_on_steps_that_executed_the_label():
    """A driver that never does what the teacher asks: the teacher goes on asking, and none of its
    patience counters runs out, so no cell is entered unseen and no door is given up."""
    grid_text = "\n".join([
        "###########",
        "#P...#....#",
        "#....D....#",
        "#....#...E#",
        "###########"])
    wad = wad_of(grid_text)
    env = make_env(wad, episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    front = FrontEnd()
    asked, still = [], np.zeros(spec.NUM_BUTTONS, np.int8)
    for _ in range(120):
        label = front.label(lean(obs), None, 525.0)
        assert_valid(label)
        asked.append((front.teacher.info.get("mode"), tuple(int(h) for h in label.heads)))
        front.took(still)                                   # nothing of it is executed
        obs, *_ = env.step(still)
    env.close()
    t = front.teacher
    assert t.errors == 0
    assert any(heads != (0, 0, 0, 0, 0, 0) for _, heads in asked)             # it did ask for something
    assert not t._waived and not t._enter_cells and not t._left_alone and not t._probed
    assert all(count == 0 or all(h == (0, 0, 0, 0, 0, 0) for _, h in asked) for _, count in t._patience.values())
    assert asked[-1] == asked[40]                                             # same place, same request

    # the counter itself: one step per executed label, none otherwise
    t = Teacher()
    t._followed = False
    assert [t._spent("look", (3, 4)) for _ in range(5)] == [0] * 5
    t._followed = True
    assert [t._spent("look", (3, 4)) for _ in range(3)] == [1, 2, 3]
    assert t._spent("look", (5, 5)) == 1                                      # another cell starts again
    # `_followed` is read off the buttons that were executed
    obs0 = {"vitals": np.zeros(spec.NUM_VITALS, np.float32),
            "rays": {"distance": np.full(64, 0.2, np.float32), "hit_type": np.ones(64, np.int64)},
            "visible_entities": {"mask": np.zeros(32, bool), "entity_type": np.zeros(32, np.int64),
                                 "features": np.zeros((32, 8), np.float32)}}
    crop = blank_crop()
    scal = np.zeros(spec.SCALAR_DIM, np.float32)
    first = t.label(obs0, crop, (0.0, 0.0, 0.0), scal, 0, 500.0, None)
    assert t.errors == 0, t.last_error
    assert not t._followed                                                    # no action led to the first observation
    t.label(obs0, crop, (0.0, 0.0, 0.0), scal, 1, 500.0, spec.heads_to_buttons(first.heads))
    assert t._followed
    other = first.heads.copy()
    other[spec.HEAD_TURN] = (int(other[spec.HEAD_TURN]) + 1) % 3
    t.label(obs0, crop, (0.0, 0.0, 0.0), scal, 2, 500.0, spec.heads_to_buttons(other))
    assert not t._followed


def test_what_patience_gave_up_is_forgotten_again():
    book = {(2, 3): 100}
    assert Teacher._given_up(book, (2, 3), 100) and Teacher._given_up(book, (2, 3), 100 + core.GIVE_UP_STEPS)
    assert not Teacher._given_up(book, (9, 9), 100)
    assert not Teacher._given_up(book, (2, 3), 101 + core.GIVE_UP_STEPS) and not book      # expired and removed
    assert core.GIVE_UP_STEPS > 4 * max(core.LOOK_PATIENCE, core.BLOCKED_PATIENCE) / 2     # long against the patience itself
    t = Teacher()
    t._waived[(1, 1)] = 5
    t._enter_cells[(1, 2)] = 5
    t._left_alone[(1, 3)] = 5
    t._patience["look"] = [(1, 1), 7]
    t.reset()
    assert not t._waived and not t._enter_cells and not t._left_alone and not t._patience and t._asked is None


def stalled_run(speed: float, follow: bool, steps: int = 40) -> tuple[Teacher, list]:
    """Label a body that stays at the start of a known corridor whatever is pressed. `speed` is the
    velocity the observation reports; with `follow` the action executed is the one the teacher asked for."""
    crop = blank_crop()
    crop[spec.CH_FREE, grid.CENTER:grid.CENTER + 7, grid.CENTER] = 1.0
    crop[spec.CH_WALL, grid.CENTER - 1:grid.CENTER + 7, grid.CENTER - 1] = 1.0
    crop[spec.CH_WALL, grid.CENTER - 1:grid.CENTER + 7, grid.CENTER + 1] = 1.0
    crop[spec.CH_WALL, grid.CENTER - 1, grid.CENTER] = 1.0
    vitals = np.zeros(spec.NUM_VITALS, np.float32)
    vitals[14] = speed
    obs = {"vitals": vitals, "rays": {"distance": np.full(64, 0.6, np.float32), "hit_type": np.zeros(64, np.int64)},
           "visible_entities": {"mask": np.zeros(32, bool), "entity_type": np.zeros(32, np.int64),
                                "features": np.zeros((32, 8), np.float32)}}
    scal = np.zeros(spec.SCALAR_DIM, np.float32)
    teacher, prev, modes = Teacher(), None, []
    for step in range(steps):
        label = teacher.label(obs, crop, (0.0, 0.0, 0.0), scal, step, 500.0, prev)
        assert_valid(label)
        modes.append((teacher.info.get("mode"), int(label.heads[spec.HEAD_MOVE]), int(label.heads[spec.HEAD_STRAFE])))
        prev = spec.heads_to_buttons(label.heads) if follow else np.zeros(spec.NUM_BUTTONS, np.int8)
    assert teacher.errors == 0, teacher.last_error
    return teacher, modes


def test_a_body_caught_on_a_corner_is_put_back_on_the_line():
    """Pushed for STALL_STEPS executed steps and no speed reported: the teacher stops driving at the
    waypoint and asks for the centre line of the route instead, for one step. Not when the body
    moves, and not when the pushes were never executed."""
    teacher, modes = stalled_run(speed=0.0, follow=True)
    pushes = [bool(move or strafe) and mode in ("run", "slow") for mode, move, strafe in modes]
    assert teacher.unstuck >= 1 and ("line", 0, 0) in modes
    first = modes.index(("line", 0, 0))
    assert first >= core.STALL_STEPS and all(pushes[first - core.STALL_STEPS:first])      # only after that many pushes
    moving, modes = stalled_run(speed=4.0, follow=True)
    assert moving.unstuck == 0 and any(mode in ("run", "slow") for mode, _, _ in modes)
    ignored, modes = stalled_run(speed=0.0, follow=False)
    assert ignored.unstuck == 0 and any(mode in ("run", "slow") for mode, _, _ in modes)


def test_a_door_that_reads_out_of_reach_is_closed_in_on():
    """The body has come to rest where the map says the door is opened from, but the pose estimate is
    14 units off along the door's axis and the forward ray reads the door 68 units away, beyond the
    press range. Waiting would last for ever: the body is placed by the ray instead and ends up in reach."""
    wait = lambda along, v, door_ahead=None: steer.wait_at_door(along, 0.0, 1.0, 0.0, v, 0.0, 0.0, 0.0, 1.0, 0.0, False, door_ahead)
    assert wait(-22.0, 0.0) == (steer.NONE, steer.NONE)                   # in the zone by the estimate: nothing to do
    assert wait(-22.0, 0.0, 68.0) == (steer.FIRST, steer.NONE)            # ... unless the ray says the door is out of reach
    assert wait(-40.0, 0.0) == (steer.FIRST, steer.NONE)                  # behind the zone: move up, as before
    error = 14.0                                                           # true distance to the door face = 32 - along + error
    along, v, reads = -22.0, 0.0, []
    for _ in range(30):
        ahead = grid.CELL / 2 - along + error
        reads.append(ahead)
        gap = ahead - (grid.CELL / 2 - along)                             # the rule of core.Teacher._steer
        far = ahead if core.DOOR_GAP < gap <= grid.CELL / 2 else None
        move, strafe = wait(along, v, far)
        assert strafe == steer.NONE
        moved, v = steer.advance(v, {steer.NONE: 0, steer.FIRST: 1, steer.SECOND: -1}[move])
        along += moved
    assert reads[0] > core.USE_PRESS_RANGE
    assert all(r <= core.USE_PRESS_RANGE for r in reads[8:]), reads           # in reach within eight steps, and stays there
    assert min(reads) > 16.0 + 4.0                                            # never leaning on the door (the body is 16 wide)


@pytest.mark.parametrize("options,fits", [({}, True), ({"start_angle": 180}, True), ({"start_angle": 270}, True),
                                          ({"start_offset": [7, -5]}, False), ({"start_offset": [0, 1]}, False),
                                          ({"start_angle": 33}, False), ({"start_angle": 91, "start_offset": [3, 0]}, False)])
def test_no_controller_labels_when_the_grid_does_not_fit_the_map(options, fits):
    """A start that is shifted inside its cell or turned off the axes puts the walls off the teacher's
    64-unit cells. The teacher sees that in its first observation and labels no head for the episode;
    `gate.start_on_grid` says the same from the build options."""
    assert gate.start_on_grid(options) is fits
    wad = wad_of(KEY_THEN_LOCK, **options)
    env = make_env(wad, episode_ticks=400, render_buffers=False)
    obs, _ = env.reset(seed=0)
    front = FrontEnd()
    for step in range(12):
        label = front.label(lean(obs), None, 100.0)
        assert_valid(label)
        assert front.teacher.grid_fits is fits
        if fits:
            assert label.head_mask.tolist() == [True, True, True, False, True, False]
        else:
            assert not label.head_mask.any() and not label.field_valid
        action = spec.heads_to_buttons(label.heads)
        front.took(action)
        obs, *_ = env.step(action)
    env.close()
    assert front.teacher.errors == 0
    front.teacher.reset()
    assert front.teacher.grid_fits


def test_gate_tells_stored_starts_on_and_off_the_grid_apart(tmp_path):
    """`--start keep` plays the maps as a training manifest stores them and marks which of them the
    teacher can label; the default strips the start options, so every map is on the grid."""
    rows = [{"name": "a", "tier": 2, "episode_ticks": 2100, "grid": THREE_DOORS, "options": {}},
            {"name": "b", "tier": 2, "episode_ticks": 2100, "grid": THREE_DOORS, "options": {"start_angle": 90}},
            {"name": "c", "tier": 2, "episode_ticks": 2100, "grid": THREE_DOORS, "options": {"start_offset": [4, 0]}},
            {"name": "d", "tier": 3, "episode_ticks": 2100, "grid": THREE_DOORS, "options": {"start_angle": 45}}]
    suite = tmp_path / "mini.jsonl"
    suite.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = gate.bare_maps(suite, start="keep", out_dir=tmp_path)
    assert [m["on_grid"] for m in kept] == [True, True, False, False]
    assert all(m["on_grid"] for m in gate.bare_maps(suite, start="aligned", out_dir=tmp_path))
    assert not gate.start_on_grid({"start_angle": 54}) and gate.start_on_grid(None)
    # the table of a mixed run: one group of rows for all episodes, then the two parts
    base = {"has_exit": 1, "sectors_visited": 4, "sectors_total": 4, "doors_opened": 3, "doors_total": 3, "locked_opened": 0,
            "locked_total": 0, "keys_picked": 0, "keys_total": 0, "door_reversals": 0, "stuck_rate": 0.0,
            "stuck_at_door_rate": 0.0, "steps": 100, "episode_ticks": 2100, "odom_err_max": 0.1, "labelled": 100}
    played = [{**base, "tier": 2, "exit": 1, "on_grid": 1}, {**base, "tier": 2, "exit": 0, "on_grid": 0, "labelled": 0}]
    table = gate.gate_table([r for r in played if r["on_grid"]])
    assert table[0]["exit_rate"] == 1.0 and table[0]["labelled_frac"] == 1.0
    assert gate.gate_table([r for r in played if not r["on_grid"]])[0]["labelled_frac"] == 0.0
    assert gate.gate_table(played)[0]["exit_rate"] == 0.5


def test_teacher_policy_reports():
    row, policy = play(THREE_DOORS)
    state = policy.debug_state()
    assert state["errors"] == 0 and state["label_ms"] > 0.0
    assert len(policy.pose_estimate()) == 3 and policy.exit_believed()
    from training.teacher import debug

    assert "@" not in debug.frame(policy) or True          # renders without raising
    assert debug.render(policy.crop, policy.pose_estimate(), policy.teacher.info).count("\n") > 2


def test_classification_of_failures():
    base = {"has_exit": 1, "exit": 0, "error": "", "agent_error": "", "dead": 0, "stuck_rate": 0.0, "teacher_errors": 0,
            "keys_picked": 0, "keys_total": 0, "locked_opened": 0, "locked_total": 0, "exit_step": -1, "leave_step": -1,
            "exit_visible_tic": -1}
    assert gate.classify({**base, "exit": 1}) == "ok"
    assert gate.classify(base) == "exit never came into view before the cap"
    assert gate.classify({**base, "exit_step": 10}) == "exit known but never left for it"
    assert gate.classify({**base, "exit_step": 10, "leave_step": 90}) == "left for the exit too late"
    assert gate.classify({**base, "keys_total": 1, "locked_total": 1}) == "key not collected"
    assert gate.classify({**base, "keys_total": 1, "keys_picked": 1, "locked_total": 1}) == "locked door not opened"
    assert gate.classify({**base, "exit_visible_tic": 400}) == "exit was in view but not registered"
    assert gate.classify({**base, "teacher_errors": 2}) == "teacher raised"


# ---------------------------------------------------------------------------------------- gate
@pytest.mark.slow
def test_labelling_cost():
    """At or under 0.5 ms per step amortised (searches included), on one thread."""
    best = float("inf")
    for _ in range(3):
        row, _ = play(KEY_THEN_LOCK)
        best = min(best, row["label_ms"])
    assert best <= 0.5, f"{best:.3f} ms per label"


@pytest.mark.slow
def test_gate_on_the_dev_suite():
    """Build spec gate: exit rate >= 90% on tiers 2 and 3 and >= 75% on tier 4, sector coverage >= 85%,
    no door reversal, on the monster-free dev suite with the true cap as budget."""
    rows, table = gate.run_gate("pytest_clean", workers=3, log=False, quiet=True)
    by_tier = {row["tier"]: row for row in table}
    for tier, floor in gate.GATE_EXIT.items():
        assert by_tier[tier]["exit_rate"] >= floor, (tier, by_tier[tier]["exit_rate"])
        assert by_tier[tier]["sectors_frac"] >= gate.GATE_SECTORS
    assert by_tier["all"]["door_reversals"] == 0
    assert by_tier["all"]["teacher_errors"] == 0 and by_tier["all"]["agent_errors"] == 0
    assert by_tier["all"]["label_ms"] <= 0.5


@pytest.mark.slow
def test_gate_with_noisy_exit_observations():
    """The training condition: 20% missed detections, position noise, rare false detections."""
    rows, table = gate.run_gate("pytest_noisy", exit_source="noisy", workers=3, log=False, quiet=True)
    by_tier = {row["tier"]: row for row in table}
    for tier, floor in gate.GATE_EXIT.items():
        assert by_tier[tier]["exit_rate"] >= floor - 0.05, (tier, by_tier[tier]["exit_rate"])
    assert by_tier["all"]["door_reversals"] == 0 and by_tier["all"]["teacher_errors"] == 0


@pytest.mark.slow
def test_gate_with_stored_starts_labels_only_the_maps_the_grid_fits(tmp_path):
    """`--start keep`, the condition of the training set: eight dev maps, every second one stored with a
    start that is shifted or turned off the axes. The teacher labels the maps its grid fits, and only
    those; the table reports the two parts apart."""
    rows = []
    for line in (ROOT / "training" / "suites" / "dev_v1.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["tier"] == 2 and len(rows) < 8:
            options = {k: v for k, v in (row.get("options") or {}).items() if k not in ("start_angle", "start_offset")}
            options.update([{}, {"start_offset": [9, -6]}, {"start_angle": 180}, {"start_angle": 37}][len(rows) % 4])
            rows.append({**row, "options": options})
    suite = tmp_path / "keep_mix.jsonl"
    suite.write_text("".join(json.dumps(r) + "\n" for r in rows))
    played, table = gate.run_gate("pytest_keep", suite=suite, tiers=(2,), start="keep", workers=3, log=False, quiet=True)
    on = [r for r in played if r["on_grid"]]
    off = [r for r in played if not r["on_grid"]]
    assert len(on) == 4 and len(off) == 4
    assert all(r["grid_fits"] == r["on_grid"] for r in played)              # the teacher's own check, from its first observation
    assert all(r["labelled"] == 0 for r in off)
    assert all(r["labelled"] > 0.9 * r["steps"] for r in on)
    assert sum(r["exit"] for r in on) >= 3 and all(r["teacher_errors"] == 0 for r in played)
    assert [row["start"] for row in table] == ["keep"] * 2 + ["keep:on-grid"] * 2 + ["keep:off-grid"] * 2
    by_part = {row["start"]: row for row in table if row["tier"] == 2}
    assert by_part["keep:on-grid"]["labelled_frac"] > 0.9 and by_part["keep:off-grid"]["labelled_frac"] == 0.0
