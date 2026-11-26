"""Tests for the odometry, the memory map and the encoder (submission/doomnet/{odometry,memory,encode}.py).

Ground truth comes from `env.privileged_state()` (position and angle) and from the ASCII grid a map
was compiled from. Both are test-side only: the modules under test never see them.

Fast tests run in CI. The measurements (position error distributions over long episodes, map
accuracy, the speed budget) are marked `slow`; run them with

    .venv/bin/python -m pytest training/tests/test_memory.py -m slow -s

to print the tables.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest
import vizdoom

from doom_env import make_env
from doomnet import spec
from doomnet.encode import Encoder
from doomnet.memory import CROP_CENTER, MemoryMap, crop_origin
from doomnet.odometry import Odometry
from fixtures import fixture_wad

FREEDOOM = os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad")
NOOP = np.zeros(9, np.int8)
FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], np.int8)
I = spec.SCALAR_INDEX

# Grid maps for the tests that need more than the small reference maps: rooms with doors, one-cell
# corridors, a maze and an arena with pillars and monsters. One character = one 64-unit cell.
GRIDS = {
    "rooms": """
####################
#P.....#...........#
#......#...........#
#......D...........#
#......#.....##....#
####.###.....##....#
#......#...........#
#......#####D#######
#..........#.......#
#......#...#.......#
#......#...........#
####################""",
    "corridors": """
######################
#P...................#
####.##########.####.#
#....#........#.#....#
#.####.######.#.#.####
#.#....#....#.#.#....#
#.#.####.##.#.#.####.#
#.#......##...#......#
#.##########D#######.#
#....................#
######################""",
    "maze": """
################
#P.#.....#.....#
#..#.###.#.###.#
#.##.#.....#...#
#....#.#####.#.#
####.#.#.....#.#
#....#.#.#####.#
#.####.#.......#
#......#######.#
#.####.........#
#......#.#####.#
################""",
    "arena": """
##################
#P...............#
#................#
#....##....##....#
#....##....##....#
#................#
#.......z........#
#....##....##....#
#....##....##.i..#
#................#
#..z.............#
##################""",
}
GRIDS = {k: v.strip("\n") for k, v in GRIDS.items()}
DOOR_CHARS = "DRBY"


# ---------------------------------------------------------------------------------------------
# test-side helpers: maps, drivers, ground truth
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def grid_maps(tmp_path_factory):
    """{name: wad path} for GRIDS, plus the same layouts with a rotated and offset player start."""
    udmf = pytest.importorskip("training.mapgen.udmf")
    root = tmp_path_factory.mktemp("memory_maps")
    out = {}
    for name, grid in GRIDS.items():
        out[name] = str(udmf.write_wad(grid, root / f"{name}.wad"))
        out[name + "_rot"] = str(udmf.write_wad(grid, root / f"{name}_rot.wad", start_angle=37, start_offset=(9.0, -5.0)))
    return out


def open_env(wad, map_name="MAP01", steps=1050):
    """Environment with pixels off and the player made unkillable (test only), so long runs do not end early."""
    env = make_env(wad, map_name, episode_ticks=spec.FRAME_SKIP * steps + 40, render_buffers=False)
    obs, _ = env.reset(seed=0)
    return env, obs


def reset(env, seed):
    obs, _ = env.reset(seed=seed)
    env._game.send_game_command("god")
    return obs


class Truth:
    """Pose in the start frame from the engine's absolute position and angle (the scorer's side of the wall)."""

    def __init__(self, env):
        s = env.privileged_state()
        self.x0, self.y0, self.a0 = s["x"], s["y"], math.radians(s["angle"])

    def pose(self, env):
        s = env.privileged_state()
        dx, dy = s["x"] - self.x0, s["y"] - self.y0
        c, sn = math.cos(self.a0), math.sin(self.a0)
        return dx * c + dy * sn, -dx * sn + dy * c, math.radians(s["angle"]) - self.a0


def angle_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class Explorer:
    """Scripted test driver: heads for open space, turns away from walls, opens doors, strafes now and then."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.turn, self.hold, self.use_wait = 0, 0, 0

    def act(self, obs):
        d, t, rng = obs["rays"]["distance"], obs["rays"]["hit_type"], self.rng
        a = np.zeros(9, np.int8)
        if self.hold > 0:
            self.hold -= 1
        else:
            front = np.r_[d[-10:], d[:11]]                                    # rays -10 .. +10
            smooth = np.convolve(front, np.ones(3) / 3, mode="same") + rng.normal(0, 0.01, 21)
            k = int(np.argmax(smooth)) - 10
            if d[0] < 0.06 and smooth.max() < 0.12:                           # dead end: turn around
                self.turn, self.hold = (1 if rng.random() < 0.5 else -1), int(rng.integers(4, 9))
            elif abs(k) >= 2 and (d[0] < 0.25 or rng.random() < 0.3):
                self.turn, self.hold = (1 if k > 0 else -1), int(min(abs(k) // 2, 3))
            else:
                self.turn = 0
                if rng.random() < 0.03:
                    self.turn, self.hold = (1 if rng.random() < 0.5 else -1), int(rng.integers(2, 10))
        a[4], a[5] = self.turn > 0, self.turn < 0
        a[0] = d[0] > 0.04
        if rng.random() < 0.08:
            a[2 + int(rng.integers(2))] = 1
        self.use_wait -= 1
        if t[0] in (spec.HIT_DOOR, spec.HIT_LOCKED) and d[0] * spec.RAY_RANGE < 60 and self.use_wait <= 0:
            a[7], self.use_wait = 1, 12
        return a


class RandomDriver:
    """Uniform random buttons, redrawn every third step (no attack, no weapon change)."""

    def __init__(self, seed):
        self.rng, self.t, self.a = np.random.default_rng(seed), 0, NOOP.copy()

    def act(self, obs):
        if self.t % 3 == 0:
            self.a = self.rng.integers(0, 2, size=9).astype(np.int8)
            self.a[6] = self.a[8] = 0
        self.t += 1
        return self.a


DRIVERS = {"explorer": Explorer, "random": RandomDriver}


def run_episode(wad, map_name, driver, seed, steps, with_map=False, record=None):
    """Drive one episode. Returns position errors per step, the largest heading error, and the memory map."""
    env, _ = open_env(wad, map_name, steps)
    obs = reset(env, seed)
    truth, od, mem, drv = Truth(env), Odometry(), MemoryMap(), DRIVERS[driver](seed)
    if with_map:
        mem.update(od.pose, obs, NOOP, None, 0)
    if record is not None:
        record.append((None, obs))
    errors, heading = [], 0.0
    for t in range(1, steps + 1):
        a = drv.act(obs).copy()
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
        od.update(a, obs)
        if with_map:
            mem.update(od.pose, obs, a, None, t)
        if record is not None:
            record.append((a, obs))
        tx, ty, tth = truth.pose(env)
        errors.append(math.hypot(od.x - tx, od.y - ty))
        heading = max(heading, angle_error(od.theta, tth))
    env.close()
    return np.array(errors), heading, mem, od


def grid_truth(grid):
    """{(i, j) crop cell seen from the start pose: grid character} and the start cell, for a grid started on 'P'."""
    rows = grid.splitlines()
    pr, pc = next((r, row.index("P")) for r, row in enumerate(rows) if "P" in row)
    return {(CROP_CENTER + c - pc, CROP_CENTER + pr - r): ch for r, row in enumerate(rows) for c, ch in enumerate(row)}


def map_accuracy(mem, grid):
    """Share of observed wall / floor cells of the crop (taken at the start pose) typed as in the grid."""
    crop = mem.crop((0.0, 0.0, 0.0))
    good = bad = doors_marked = doors = 0
    for (i, j), ch in grid_truth(grid).items():
        wall, free = crop[spec.CH_WALL, i, j] > 0, crop[spec.CH_FREE, i, j] > 0
        door = crop[spec.CH_DOOR, i, j] > 0 or crop[spec.CH_LOCKED, i, j] > 0
        if ch in DOOR_CHARS:
            doors += wall or free or door
            doors_marked += door or (free and not wall)          # closed and marked, or last seen open
        elif wall or free or door:
            good += (wall and not free and not door) if ch == "#" else (free and not wall and not door)
            bad += not ((wall and not free and not door) if ch == "#" else (free and not wall and not door))
    return good / max(good + bad, 1), good + bad, doors_marked, doors


# ---------------------------------------------------------------------------------------------
# shapes, dtypes, scalars
# ---------------------------------------------------------------------------------------------
def test_shapes_and_dtypes_match_spec():
    env, obs = open_env("maps/two_rooms.wad", steps=40)
    od, mem, enc = Odometry(), MemoryMap(), Encoder()
    for step in range(12):
        if step:
            a = env.action_space.sample().astype(np.int8)
            obs, *_ = env.step(a)
            od.update(a, obs)
        else:
            a = None
        mem.update(od.pose, obs, a, None, step)
        crop = mem.crop(od.pose)
        f = enc.encode(obs, od.pose, a, step, spec.DEFAULT_BUDGET_STEPS, None)
        assert crop.shape == (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE) and crop.dtype == np.float32
        assert crop.min() >= 0.0 and crop.max() <= 1.0
        assert f["rays"].shape == (spec.NUM_RAYS, spec.RAY_FEATS) and f["rays"].dtype == np.float32
        assert f["ents"].shape == (spec.MAX_ENTITIES, spec.ENT_FEATS) and f["ents"].dtype == np.float32
        assert f["ent_mask"].shape == (spec.MAX_ENTITIES,) and f["ent_mask"].dtype == np.bool_
        assert f["scal"].shape == (spec.SCALAR_DIM,) and f["scal"].dtype == np.float32
        assert all(np.isfinite(v).all() for v in f.values())
        np.testing.assert_allclose(f["rays"][:, 1:].sum(axis=1), 1.0)                 # one hit type per ray
        np.testing.assert_allclose(f["ents"][:, 5:].sum(axis=1), f["ent_mask"])       # one type per real entity
        assert not f["ents"][~f["ent_mask"]].any()
        np.testing.assert_allclose(f["rays"][:, 0], obs["rays"]["distance"])
        assert f["scal"][I["weapon_0"]:I["weapon_0"] + 8].sum() == 1.0
    assert isinstance(od.x, float) and isinstance(od.y, float) and isinstance(od.theta, float) and od.step == 11
    env.close()


def test_scalars_follow_the_observation():
    env, obs = open_env(fixture_wad("#########\n#P..h..z#\n#########"), steps=40)
    enc = Encoder()
    pose = (70.0, -40.0, math.pi / 2)
    prev = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1], np.int8)
    s = enc.encode(obs, pose, prev, 105, 525, (0.9, 0.25, -0.5))["scal"]
    v = obs["vitals"]
    assert s[I["health"]] == pytest.approx(v[0] / 100) and s[I["armor"]] == pytest.approx(v[1] / 100)
    assert s[I["weapon_0"] + int(v[2])] == 1.0
    assert s[I["sel_ammo"]] == pytest.approx(v[3] / 50) and s[I["ammo_bullets"]] == pytest.approx(v[4] / 50)
    assert s[I["step_frac"]] == pytest.approx(0.1) and s[I["budget_frac"]] == pytest.approx(0.5)
    assert s[I["budget_left"]] == pytest.approx(0.4)
    assert s[I["sin_theta"]] == pytest.approx(1.0) and s[I["cos_theta"]] == pytest.approx(0.0, abs=1e-6)
    assert s[I["subcell_x"]] == pytest.approx(6 / 64) and s[I["subcell_y"]] == pytest.approx(24 / 64)   # cells 1 and -1
    np.testing.assert_array_equal(s[I["prev_fwd"]:I["prev_fwd"] + 9], prev)
    assert (s[I["exit_visible"]], s[I["exit_x"]], s[I["exit_y"]]) == (1.0, 0.25, -0.5)
    # An exit observation below half confidence, or none at all, leaves the three exit scalars at zero.
    for exit_obs in (None, (0.2, 0.25, -0.5)):
        s = enc.encode(obs, pose, prev, 105, 525, exit_obs)["scal"]
        assert not s[I["exit_visible"]:I["exit_visible"] + 3].any()
    # Sub-cell offsets stay in [-0.5, 0.5) and are 0 at a cell centre.
    for x in (-1000.3, -32.0, -0.01, 0.0, 31.99, 32.0, 777.7):
        sub = Encoder().encode(obs, (x, x, 0.0), None, 0, 525, None)["scal"][I["subcell_x"]]
        assert -0.5 <= sub < 0.5 and sub == pytest.approx((x + 32) % 64 / 64 - 0.5, abs=1e-5)
    env.close()


def test_use_age_and_budget_scalars():
    env, obs = open_env(fixture_wad("#####\n#P..#\n#####"), steps=40)
    enc = Encoder()
    age = lambda a, step=0, budget=525: enc.encode(obs, (0, 0, 0), a, step, budget, None)["scal"][I["use_age"]]
    assert age(None) == 1.0                                   # never pressed: saturated
    assert age(NOOP) == 1.0
    assert age(USE) == 0.0                                    # the observation that follows the press
    assert age(USE) == pytest.approx(1 / 16)                  # holding USE is still one press
    assert age(NOOP) == pytest.approx(2 / 16)
    assert age(USE) == 0.0                                    # released and pressed again
    for k in range(1, 20):
        assert age(NOOP) == pytest.approx(min(k / 16, 1.0))   # saturates at 16 steps
    enc.reset()
    assert age(NOOP) == 1.0

    budget = lambda step, b: Encoder().encode(obs, (0, 0, 0), NOOP, step, b, None)["scal"][[I["step_frac"], I["budget_frac"], I["budget_left"]]]
    np.testing.assert_allclose(budget(0, 525), [0.0, 0.5, 0.5])
    np.testing.assert_allclose(budget(525, 525), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(budget(1050, 525), [1.0, 0.5, -0.5])
    np.testing.assert_allclose(budget(2000, 525), [2000 / 1050, 0.5, -1.0])          # budget_left is clipped
    np.testing.assert_allclose(budget(0, 5000), [0.0, 4.0, 1.0])
    env.close()


def test_encoder_survives_corrupt_values():
    env, obs = open_env("maps/two_rooms.wad", steps=40)
    bad = {"vitals": np.full(17, np.nan, np.float32),
           "rays": {"distance": np.array([np.nan, np.inf, -np.inf, 5.0] * 16, np.float32), "hit_type": np.full(64, 99, np.int64)},
           "visible_entities": {"features": np.full((32, 5), np.inf, np.float32), "entity_type": np.full(32, -3, np.int64),
                                "mask": np.ones(32, np.int8)}}
    bad["vitals"][0], bad["vitals"][14] = np.inf, -np.inf
    od, mem, enc = Odometry(), MemoryMap(), Encoder()
    od.update(FWD, obs)
    for pose in ((0.0, 0.0, 0.0), (float("nan"), 1.0, 0.0), (0.0, float("inf"), float("nan"))):
        f = enc.encode(bad, pose, FWD, 3, float("nan"), (float("nan"), float("inf"), 0.0))
        assert all(np.isfinite(v).all() for v in f.values())
        assert f["rays"][:, 0].min() >= 0.0 and f["rays"][:, 0].max() <= 1.0
        mem.update(pose, bad, FWD, (1.0, float("nan"), float("inf")), 4)
        assert np.isfinite(mem.crop(pose)).all()
    od.update(FWD, bad)                                       # NaN velocity: the model's prediction is kept
    od.update(FWD, obs)
    assert math.isfinite(od.x) and math.isfinite(od.y) and math.isfinite(od.theta)
    env.close()


def test_odd_observations_never_raise_or_warn():
    """Odd but castable inputs, 1,200 steps without a reset, then a reset: finite outputs and no numpy warning."""
    import warnings

    rng = np.random.default_rng(0)

    def make(dist, hit, vitals=None, feats=None, etype=None, mask=None):
        base = np.r_[100, 0, 2, 50, 50, 0, 0, 0, np.zeros(9)]
        return {"vitals": np.asarray(base if vitals is None else vitals, np.float32),
                "rays": {"distance": np.asarray(dist, np.float32), "hit_type": np.asarray(hit, np.int64)},
                "visible_entities": {"features": np.asarray(np.zeros((32, 5)) if feats is None else feats, np.float32),
                                     "entity_type": np.asarray(np.zeros(32) if etype is None else etype, np.int64),
                                     "mask": np.asarray(np.zeros(32) if mask is None else mask, np.int8)}}

    far = np.c_[np.full(32, 4.0), np.zeros(32), np.ones(32), np.full(32, 50.0), np.full(32, -50.0)]
    cases = [
        make(np.zeros(64), np.ones(64)),                                           # all-zero rays
        make(np.ones(64), np.zeros(64)),                                           # all-one rays
        make(np.ones(64), np.ones(64)),
        make(np.full(64, np.nan), np.ones(64)),
        make(np.full(64, np.inf), np.full(64, 3)),
        make(rng.random(64), np.ones(64)),                                         # walls that jump every step
        make(rng.random(64), rng.integers(0, 6, 64), feats=far, etype=rng.integers(0, 12, 32), mask=np.ones(32)),
        make(rng.random(64), rng.integers(0, 6, 64), vitals=np.r_[np.nan, np.inf, -np.inf, np.nan, np.zeros(10), np.inf, -np.inf, np.nan]),
        make(rng.random(64), rng.integers(0, 6, 64), vitals=np.r_[0, 200, 7, 999, 999, 999, 999, 999, np.ones(6), 1e6, -1e6, 1e30]),
        make(rng.random(64), np.ones(64), vitals=np.r_[200, 200, 7, 9, 9, 9, 9, 9, np.ones(6), 100.0, -90.0, 3.0]),
        make(rng.random(64), rng.integers(0, 6, 64), feats=np.full((32, 5), np.nan), etype=np.full(32, 99), mask=np.ones(32)),
        {"vitals": [100.0] * 17,                                                   # wrong but castable dtypes
         "rays": {"distance": np.linspace(0, 1, 64), "hit_type": np.ones(64, np.uint8)},
         "visible_entities": {"features": np.zeros((32, 5)), "entity_type": np.ones(32, np.int8), "mask": np.ones(32, bool)}},
    ]
    actions = [np.array(a, np.int8) for a in ([1, 0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0], [1] * 9, [0] * 9)]
    od, mem, enc = Odometry(), MemoryMap(), Encoder()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for step in range(1200):
            obs, a = cases[step % len(cases)], actions[(step // 5) % 4]
            od.update(a, obs)
            assert all(math.isfinite(v) for v in od.pose)
            mem.update(od.pose, obs, a, (0.9, 0.2, 0.1), step)
            crop = mem.crop(od.pose)
            f = enc.encode(obs, od.pose, a, step, 525, (0.9, 0.2, 0.1))
            assert np.isfinite(crop).all() and crop.min() >= 0.0 and crop.max() <= 1.0
            assert all(np.isfinite(v).all() for v in f.values())
    od.reset(), mem.reset(), enc.reset()
    assert od.pose == (0.0, 0.0, 0.0) and not mem.crop(od.pose).any()


# ---------------------------------------------------------------------------------------------
# odometry
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("wad,map_name", [("maps/two_rooms.wad", "MAP01"),
                                          (fixture_wad("############\n#P.r..R...E#\n############"), "MAP01"),
                                          (FREEDOOM, "MAP07")])
def test_heading_is_exact(wad, map_name):
    """Random buttons, both turn keys together included: the heading matches the engine at every step."""
    for seed in range(2):
        errors, heading, _, od = run_episode(wad, map_name, "random", seed, 520)
        assert len(errors) > 200 and od.step == len(errors)
        assert heading < 1e-6, f"heading error {math.degrees(heading):.6f} degrees"


def test_free_motion_is_exact():
    """No wall is touched on this walk through the open start room: every step is predicted, none is measured."""
    env, _ = open_env("maps/two_rooms.wad", steps=200)
    obs = reset(env, 0)
    truth, od = Truth(env), Odometry()
    script = [NOOP] * 2 + [FWD] * 6 + [TURN_L] * 3 + [NOOP] * 8 + [np.array([0, 1, 0, 1, 0, 0, 0, 0, 0], np.int8)] * 3 + [NOOP] * 12
    for a in script:
        obs, *_ = env.step(a)
        od.update(a, obs)
        tx, ty, tth = truth.pose(env)
        assert math.hypot(od.x - tx, od.y - ty) < 0.02 and angle_error(od.theta, tth) < 1e-6
    assert od.disturbed_steps == 0 and math.hypot(od.x, od.y) > 60
    env.close()


def test_position_survives_walls(grid_maps):
    """Short runs that ram and slide along walls: the drift stays far below one map cell."""
    for name, driver, steps in (("maze", "random", 300), ("corridors", "explorer", 300), ("rooms_rot", "random", 300)):
        errors, heading, _, od = run_episode(grid_maps[name], "MAP01", driver, 1, steps)
        assert od.disturbed_steps > 20, "the run was meant to hit walls"
        assert errors.max() < 8.0 and heading < 1e-6, f"{name}/{driver}: max error {errors.max():.2f}"


def test_pushing_into_a_wall_does_not_move_the_estimate():
    env, _ = open_env(fixture_wad("####\n#P.#\n####"), steps=200)
    obs = reset(env, 0)
    truth, od = Truth(env), Odometry()
    back = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], np.int8)
    for a in [back] * 40 + [FWD] * 60 + [np.array([1, 0, 1, 0, 0, 0, 0, 0, 0], np.int8)] * 40:
        obs, *_ = env.step(a)
        od.update(a, obs)
    tx, ty, _ = truth.pose(env)
    assert math.hypot(od.x - tx, od.y - ty) < 0.5
    assert abs(od.x - 80.0) < 0.5 and abs(od.y - 16.0) < 0.5      # the front-left corner of the two-cell room
    env.close()


@pytest.mark.parametrize("grid,limit", [("##########\n#P......i#\n##########", 12.0), ("#########\n#P..h..z#\n#########", 32.0)])
def test_monster_in_the_way_does_not_run_the_estimate_through_it(grid, limit):
    """One-cell corridor, a monster at its end, the walls ahead hidden behind it: pushing into the monster for
    hundreds of steps must not move the estimate (a model that takes the stop for a wall slide runs on for
    about 7 units per step)."""
    env, _ = open_env(fixture_wad(grid), steps=300)
    obs = reset(env, 0)
    truth, od = Truth(env), Odometry()
    back = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], np.int8)
    worst, blocked = 0.0, 0
    for t in range(250):
        a = FWD if (t // 60) % 2 == 0 else back
        obs, *_ = env.step(a)
        od.update(a, obs)
        tx, ty, _ = truth.pose(env)
        worst = max(worst, math.hypot(od.x - tx, od.y - ty))
        blocked += od.last_mode == "rays+body"
    assert blocked > 20, "the run was meant to push against the monster"
    assert worst < limit, f"worst error {worst:.1f}"
    env.close()


def test_odometry_reset():
    env, obs = open_env("maps/two_rooms.wad", steps=60)
    od = Odometry()
    for _ in range(20):
        a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], np.int8)
        obs, *_ = env.step(a)
        od.update(a, obs)
    assert od.pose != (0.0, 0.0, 0.0) and od.step == 20
    od.reset()
    assert od.pose == (0.0, 0.0, 0.0) and od.step == 0 and od.disturbed_steps == 0
    env.close()


# ---------------------------------------------------------------------------------------------
# memory map
# ---------------------------------------------------------------------------------------------
def first_view(grid, **kw):
    """Memory map after the first observation of a reference grid map."""
    env, obs = open_env(fixture_wad(grid), steps=400)
    mem = MemoryMap()
    mem.update((0.0, 0.0, 0.0), obs, None, None, 0)
    return env, obs, mem


def cell(crop, ch, di, dj=0):
    """Channel value of the crop cell `di` cells forward and `dj` cells to the left of the agent's cell."""
    return crop[ch, CROP_CENTER + di, CROP_CENTER + dj]


def test_crop_cells_coincide_with_grid_cells_at_the_start():
    grid = "##########\n#P..D...E#\n##########"
    env, obs, mem = first_view(grid)
    crop = mem.crop((0.0, 0.0, 0.0))
    assert crop_origin((0.0, 0.0)) == (-CROP_CENTER, -CROP_CENTER)
    for di in (0, 1, 2):                                       # the corridor up to the door
        assert cell(crop, spec.CH_FREE, di) == 1 and cell(crop, spec.CH_WALL, di) == 0
        assert cell(crop, spec.CH_WALL, di, 1) == 1 and cell(crop, spec.CH_WALL, di, -1) == 1      # side walls
    assert cell(crop, spec.CH_WALL, -1) == 1                   # the wall behind the start
    assert cell(crop, spec.CH_DOOR, 3) == 1 and cell(crop, spec.CH_WALL, 3) == 0 and cell(crop, spec.CH_FREE, 3) == 0
    assert crop[spec.CH_DOOR].sum() == 1 and crop[spec.CH_LOCKED].sum() == 0
    assert crop[spec.CH_FREE].sum() == 3                       # nothing beyond the closed door
    # Every marked crop cell is a grid cell of that kind, and nothing is marked outside the grid.
    truth = grid_truth(grid)
    for ch, kinds in ((spec.CH_WALL, "#"), (spec.CH_FREE, ".P"), (spec.CH_DOOR, "D")):
        for i, j in zip(*np.nonzero(crop[ch])):
            assert truth.get((i, j)) is not None and truth[(i, j)] in kinds, (ch, i, j, truth.get((i, j)))
    assert cell(crop, spec.CH_VISITED, 0) == 1 and crop[spec.CH_VISITED].sum() == 1
    assert cell(crop, spec.CH_RECENCY, 0) == 1.0
    env.close()


def test_crop_follows_the_agent_in_whole_cells():
    """The crop is centred on the agent's 64-unit cell: it shifts by one index when a cell border is crossed."""
    env, obs, mem = first_view("##########\n#P..D...E#\n##########")
    at = lambda x, y: mem.crop((x, y, 0.3))
    base = at(0.0, 0.0)
    np.testing.assert_array_equal(at(31.9, -32.0), base)                       # same cell: [-32, 32) on both axes
    np.testing.assert_array_equal(at(32.0, 0.0)[:, :-1], base[:, 1:])          # one cell forward
    np.testing.assert_array_equal(at(0.0, -32.1)[:, :, 1:], base[:, :, :-1])   # one cell to the right
    assert crop_origin((64.0, -64.0)) == (1 - CROP_CENTER, -1 - CROP_CENTER)
    env.close()


def test_locked_door_item_enemy_and_key_channels():
    env, obs, mem = first_view("#######\n#P..B.E#\n#######")
    crop = mem.crop((0.0, 0.0, 0.0))
    assert cell(crop, spec.CH_LOCKED, 3) == 1 and crop[spec.CH_LOCKED].sum() == 1 and crop[spec.CH_DOOR].sum() == 0
    env.close()

    env, obs, mem = first_view("#########\n#P..h..z#\n#########")
    crop = mem.crop((0.0, 0.0, 0.0))
    assert cell(crop, spec.CH_ITEM, 3) == 1 and crop[spec.CH_ITEM].sum() == 1
    assert cell(crop, spec.CH_ENEMY, 6) == 1.0 and (crop[spec.CH_ENEMY] > 0).sum() == 1
    assert crop[spec.CH_KEY].sum() == 0
    # An enemy sighting fades: half gone after 32 steps without seeing it, gone after 64.
    hidden = {**obs, "visible_entities": {k: np.zeros_like(v) for k, v in obs["visible_entities"].items()}}
    mem.update((0.0, 0.0, 0.0), hidden, None, None, 32)
    assert cell(mem.crop((0.0, 0.0, 0.0)), spec.CH_ENEMY, 6) == pytest.approx(0.5)
    mem.update((0.0, 0.0, 0.0), hidden, None, None, 64)
    assert mem.crop((0.0, 0.0, 0.0))[spec.CH_ENEMY].sum() == 0
    env.close()

    env, obs, mem = first_view("############\n#P.r..R...E#\n############")
    assert cell(mem.crop((0.0, 0.0, 0.0)), spec.CH_KEY, 2) == 1
    od = Odometry()
    for t in range(1, 30):                                     # walk over the key: it is picked up
        obs, *_ = env.step(FWD)
        od.update(FWD, obs)
        mem.update(od.pose, obs, FWD, None, t)
    assert obs["vitals"][10] == 1 and od.x > 128
    crop = mem.crop((0.0, 0.0, 0.0))
    assert crop[spec.CH_KEY].sum() == 0
    assert cell(crop, spec.CH_LOCKED, 5) == 1                  # the locked door further on is still remembered
    assert [cell(crop, spec.CH_VISITED, d) for d in (0, 1, 2, 3)] == [1, 1, 1, 1 if od.x >= 160 else 0]
    env.close()


def test_door_rule_open_door_stops_showing_and_returns_when_closed():
    """Closed door: door bit. Seen open (rays pass through its cell): bit cleared, cell free. Closed again: bit back."""
    env, obs, mem = first_view("##########\n#P..D...E#\n##########")
    od = Odometry()
    step = 0

    def advance(a):
        nonlocal obs, step
        step += 1
        obs, *_ = env.step(a)
        od.update(a, obs)
        mem.update(od.pose, obs, a, None, step)
        return mem.crop((0.0, 0.0, 0.0))

    while obs["rays"]["distance"][0] * spec.RAY_RANGE > 62:    # walk up to the door (it is 160 units ahead)
        crop = advance(FWD)
    for _ in range(8):
        crop = advance(NOOP)
    assert cell(crop, spec.CH_DOOR, 3) == 1 and crop[spec.CH_DOOR].sum() == 1
    crop = advance(USE)
    assert cell(crop, spec.CH_USED, 3) == 1.0                  # the press is remembered on the door's cell
    opened = None
    for k in range(1, 30):
        crop = advance(NOOP)
        if obs["rays"]["hit_type"][0] != spec.HIT_DOOR:
            opened = k
            break
        assert cell(crop, spec.CH_DOOR, 3) == 1                # still closing the view: still a door
    assert opened is not None and opened <= 8
    assert crop[spec.CH_DOOR].sum() == 0                       # seen open: no longer shown as a door
    assert cell(crop, spec.CH_FREE, 3) == 1 and cell(crop, spec.CH_WALL, 3) == 0
    assert cell(crop, spec.CH_FREE, 5) == 1                    # and the corridor behind it has been seen
    for _ in range(70):                                        # it closes by itself after 150 tics
        crop = advance(NOOP)
    assert obs["rays"]["hit_type"][0] == spec.HIT_DOOR
    assert cell(crop, spec.CH_DOOR, 3) == 1 and crop[spec.CH_DOOR].sum() == 1
    assert cell(crop, spec.CH_USED, 3) == 0.0                  # the USE mark has faded (64 steps)
    env.close()


def test_exit_used_visited_and_recency_channels():
    env, obs, mem = first_view("#######\n#P...E#\n#######")
    here = (0.0, 0.0, 0.0)
    assert mem.crop(here)[spec.CH_EXIT].sum() == 0
    mem.update(here, obs, None, (0.2, 0.25, 0.0), 1)                           # low confidence: ignored
    assert mem.crop(here)[spec.CH_EXIT].sum() == 0
    mem.update(here, obs, None, (1.0, 0.25, 0.0), 2)                           # 256 units ahead = 4 cells
    assert cell(mem.crop(here), spec.CH_EXIT, 4) == pytest.approx(0.5)
    mem.update(here, obs, None, (0.8, 0.26, 0.01), 3)                          # a second sighting, same cell
    assert cell(mem.crop(here), spec.CH_EXIT, 4) == pytest.approx(0.9)
    mem.update(here, obs, None, (1.0, 0.25, 0.0), 4)
    assert cell(mem.crop(here), spec.CH_EXIT, 4) == 1.0 and mem.crop(here)[spec.CH_EXIT].sum() == 1.0
    # Seen from a turned pose the offsets rotate with the heading: forward is +y when facing left.
    mem.update((0.0, 0.0, math.pi / 2), obs, None, (1.0, 0.125, 0.0), 5)
    assert cell(mem.crop(here), spec.CH_EXIT, 0, 2) == pytest.approx(0.5)

    # USE marks the cell straight ahead, just past the hit if something is within reach, and fades.
    mem.reset()
    mem.update(here, obs, USE, None, 10)                       # nothing within reach: the cell 64 units ahead
    assert cell(mem.crop(here), spec.CH_USED, 1) == 1.0 and (mem.crop(here)[spec.CH_USED] > 0).sum() == 1
    mem.update(here, obs, NOOP, None, 42)
    assert cell(mem.crop(here), spec.CH_USED, 1) == pytest.approx(0.5)
    mem.update(here, obs, NOOP, None, 74)
    assert mem.crop(here)[spec.CH_USED].sum() == 0

    # Visited is permanent, recency fades over 128 steps.
    mem.reset()
    mem.update(here, obs, NOOP, None, 0)
    mem.update((64.0, 0.0, 0.0), obs, NOOP, None, 64)
    crop = mem.crop(here)
    assert cell(crop, spec.CH_VISITED, 0) == 1 and cell(crop, spec.CH_VISITED, 1) == 1 and crop[spec.CH_VISITED].sum() == 2
    assert cell(crop, spec.CH_RECENCY, 0) == pytest.approx(0.5) and cell(crop, spec.CH_RECENCY, 1) == 1.0
    mem.update((64.0, 0.0, 0.0), obs, NOOP, None, 200)
    crop = mem.crop(here)
    assert cell(crop, spec.CH_RECENCY, 0) == 0.0 and cell(crop, spec.CH_VISITED, 0) == 1
    env.close()


def test_memory_reset_clears_everything():
    env, obs, mem = first_view("############\n#P.r..R...E#\n############")
    mem.update((0.0, 0.0, 0.0), obs, USE, (1.0, 0.3, 0.0), 5)
    assert mem.crop((0.0, 0.0, 0.0)).sum() > 0 and mem.full().sum() > 0
    mem.reset()
    assert mem.crop((0.0, 0.0, 0.0)).sum() == 0 and mem.full().sum() == 0
    env.close()


def test_full_map_agrees_with_crop():
    errors, _, mem, od = run_episode("maps/two_rooms.wad", "MAP01", "explorer", 0, 150, with_map=True)
    full = mem.full()
    assert full.shape == (spec.MAP_CHANNELS, spec.MAP_SIZE, spec.MAP_SIZE) and full.dtype == np.float32
    for pose in ((0.0, 0.0, 0.0), od.pose, (-500.0, 333.0, 1.0)):
        oi, oj = crop_origin(pose)
        i0, j0 = 2 * oi + 127, 2 * oj + 127                    # crop cell k pools 32-unit cells 2k + 127 and 2k + 128
        ref = full[:, i0:i0 + 128, j0:j0 + 128].reshape(spec.MAP_CHANNELS, 64, 2, 64, 2).max(axis=(2, 4))
        np.testing.assert_allclose(mem.crop(pose), ref)


def test_bounds_far_poses_never_raise():
    """Writes outside the array are dropped, crops past the edge are zero-padded, absurd poses are harmless."""
    env, obs = open_env("maps/two_rooms.wad", steps=40)
    mem = MemoryMap()
    mem.update((0.0, 0.0, 0.0), obs, USE, (1.0, 0.2, 0.1), 1)
    before = (mem.occ.copy(), mem.flags.copy(), mem.stamp.copy(), mem.exit.copy())
    for pose in ((10_000.0, 0.0, 0.0), (0.0, -10_000.0, 2.0), (-10_000.0, 10_000.0, -1.0), (1e12, -1e12, 0.5),
                 (float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0, float("nan"))):
        mem.update(pose, obs, USE, (1.0, 0.2, 0.1), 2)         # 10,000 units from the start: nothing lands
        crop = mem.crop(pose)
        assert crop.shape == (spec.MAP_CHANNELS, 64, 64) and np.isfinite(crop).all()
        assert not crop.any() or pose[:2] == (0.0, 0.0)        # only the last pose still looks at the start
    for a, b in zip(before, (mem.occ, mem.flags, mem.stamp, mem.exit)):
        np.testing.assert_array_equal(a, b)

    # Half in, half out: the part of the view inside the array is written, the rest is dropped, and
    # the part of the crop that lies beyond the stored area reads zero.
    edge = spec.MAP_SIZE * spec.MAP_RES / 2                    # 4096 units from the start
    for pose in ((edge - 100.0, 0.0, 0.0), (-edge + 10.0, edge - 10.0, 1.0), (edge + 300.0, -edge - 300.0, 3.0)):
        mem.reset()
        mem.update(pose, obs, USE, (1.0, 0.2, 0.1), 3)
        assert mem.occ.any() == (abs(pose[0]) < edge)          # the agent's own cell, if it is inside
        crop = mem.crop(pose)
        oi, oj = crop_origin(pose)
        stored_i = np.abs(np.arange(64) + oi) <= 64            # 64-unit cells that exist in the store
        stored_j = np.abs(np.arange(64) + oj) <= 64
        assert not crop[:, ~stored_i, :].any() and not crop[:, :, ~stored_j].any()
        assert np.isfinite(crop).all()
        assert crop.any() == (abs(pose[0]) < edge)
    # A crop taken far from everything that was written is all zeros.
    mem.reset()
    mem.update((0.0, 0.0, 0.0), obs, None, None, 0)
    assert mem.crop((0.0, 0.0, 0.0)).any() and not mem.crop((3500.0, 3500.0, 0.0)).any()
    env.close()


def test_map_matches_the_grid_after_a_short_exploration(grid_maps):
    errors, _, mem, _ = run_episode(grid_maps["maze"], "MAP01", "explorer", 0, 300, with_map=True)
    accuracy, observed, _, _ = map_accuracy(mem, GRIDS["maze"])
    assert observed > 100 and accuracy >= 0.97, f"accuracy {accuracy:.3f} over {observed} cells, drift {errors[-1]:.2f}"


# ---------------------------------------------------------------------------------------------
# measurements (slow): position error distributions, map accuracy, speed
# ---------------------------------------------------------------------------------------------
def position_table(maps, drivers, seeds, steps=1050):
    """{(map, driver): errors array (seeds, steps)} with NaN after an episode's end."""
    table = {}
    for name, (wad, map_name) in maps.items():
        for driver in drivers:
            runs = np.full((len(seeds), steps), np.nan)
            for k, seed in enumerate(seeds):
                errors, heading, _, _ = run_episode(wad, map_name, driver, seed, steps)
                assert heading < 1e-6
                runs[k, :len(errors)] = errors
            table[(name, driver)] = runs
    return table


def describe(errors):
    e = errors[np.isfinite(errors)]
    return (float(np.median(e)), float(np.percentile(e, 90)), float(e.max())) if len(e) else (float("nan"),) * 3


@pytest.mark.slow
def test_position_error_distribution(grid_maps):
    """SPEC targets after 520 steps of a sensibly moving driver: median under 16 units, 90th percentile under 32.

    Measured on six grid layouts (three with a rotated and offset start), the bundled test map and
    one Freedoom level, 4 seeds each, at steps 520 and 1050, for the scripted explorer and for
    random buttons.
    """
    maps = {name: (grid_maps[name], "MAP01") for name in ("rooms", "corridors", "maze", "rooms_rot", "corridors_rot", "maze_rot")}
    maps["two_rooms"] = ("maps/two_rooms.wad", "MAP01")
    maps["freedoom07"] = (FREEDOOM, "MAP07")
    table = position_table(maps, ("explorer", "random"), seeds=range(4))
    print("\nposition error in map units: median / p90 / max over 4 seeds")
    print(f"{'map':14s} {'driver':9s} {'step 520':>24s} {'step 1050':>24s}")
    for (name, driver), runs in table.items():
        a, b = describe(runs[:, 519]), describe(runs[:, 1049])
        print(f"{name:14s} {driver:9s} {a[0]:8.2f} {a[1]:7.2f} {a[2]:7.2f} {b[0]:8.2f} {b[1]:7.2f} {b[2]:7.2f}")
    for driver in ("explorer", "random"):
        for step in (520, 1050):
            med, p90, worst = describe(np.concatenate([runs[:, step - 1] for (n, d), runs in table.items() if d == driver]))
            print(f"ALL {driver:9s} step {step:4d}: median {med:.2f}  p90 {p90:.2f}  max {worst:.2f}")
            assert med < 16.0 and p90 < 32.0, f"{driver} at step {step}: median {med:.1f}, p90 {p90:.1f}"
    explorer_520 = np.concatenate([runs[:, 519] for (n, d), runs in table.items() if d == "explorer"])
    assert np.nanmedian(explorer_520) < 2.0 and np.nanmax(explorer_520) < 16.0     # far inside the targets


@pytest.mark.slow
def test_map_accuracy_after_exploration(grid_maps):
    """Share of observed crop cells typed as in the true grid after 1050 steps (walls and floor; doors apart)."""
    print("\nmap accuracy after 1050 steps: correct share of observed wall/floor cells (cells), doors marked/seen, drift")
    worst = 1.0
    for name in ("rooms", "corridors", "maze", "arena"):
        for driver in ("explorer", "random"):
            for seed in range(2):
                errors, _, mem, _ = run_episode(grid_maps[name], "MAP01", driver, seed, 1050, with_map=True)
                accuracy, observed, doors_marked, doors = map_accuracy(mem, GRIDS[name])
                worst = min(worst, accuracy)
                print(f"{name:10s} {driver:9s} seed {seed}: {accuracy:.4f} ({observed:3d})  doors {doors_marked}/{doors}  drift {errors[-1]:.2f}")
    assert worst >= 0.95


def stage_times(tape, repeats=3):
    """Mean microseconds per step of each stage over a recorded episode (best of `repeats`), one thread."""
    best = None
    clock = time.perf_counter
    for _ in range(repeats):
        od, mem, enc = Odometry(), MemoryMap(), Encoder()
        total = np.zeros(4)
        for step, (a, obs) in enumerate(tape):
            t0 = clock()
            if step:
                od.update(a, obs)
            t1 = clock()
            pose = (od.x, od.y, od.theta)
            mem.update(pose, obs, a, None, step)
            t2 = clock()
            mem.crop(pose)
            t3 = clock()
            enc.encode(obs, pose, a, step, spec.DEFAULT_BUDGET_STEPS, None)
            t4 = clock()
            total += (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        total *= 1e6 / len(tape)
        if best is None or total.sum() < best.sum():
            best = total
    return best


def test_per_step_cost_is_sane():
    """Loose ceiling for CI machines; the 0.6 ms budget itself is checked by the slow benchmark below."""
    tape = []
    run_episode("maps/two_rooms.wad", "MAP01", "explorer", 0, 300, record=tape)
    assert stage_times(tape, repeats=2).sum() < 2000.0


@pytest.mark.slow
def test_speed_budget(grid_maps):
    """Odometry.update + MemoryMap.update + MemoryMap.crop + Encoder.encode: at or under 0.6 ms mean, one thread."""
    cases = (("two_rooms", "maps/two_rooms.wad", "MAP01", "explorer"), ("rooms", grid_maps["rooms"], "MAP01", "explorer"),
             ("corridors", grid_maps["corridors"], "MAP01", "explorer"), ("corridors", grid_maps["corridors"], "MAP01", "random"),
             ("maze", grid_maps["maze"], "MAP01", "random"), ("arena", grid_maps["arena"], "MAP01", "explorer"),
             ("freedoom01", FREEDOOM, "MAP01", "explorer"))
    print("\nmicroseconds per step, mean over a 1050-step episode (best of 3 passes)")
    print(f"{'map':12s} {'driver':9s} {'odometry':>9s} {'mem.update':>11s} {'mem.crop':>9s} {'encode':>7s} {'total':>7s}")
    rows = []
    for name, wad, map_name, driver in cases:
        tape = []
        run_episode(wad, map_name, driver, 0, 1050, record=tape)
        t = stage_times(tape)
        rows.append(t)
        print(f"{name:12s} {driver:9s} {t[0]:9.1f} {t[1]:11.1f} {t[2]:9.1f} {t[3]:7.1f} {t.sum():7.1f}")
    mean = np.mean(rows, axis=0)
    print(f"{'mean':22s} {mean[0]:9.1f} {mean[1]:11.1f} {mean[2]:9.1f} {mean[3]:7.1f} {mean.sum():7.1f}")
    assert max(t.sum() for t in rows) <= 600.0
