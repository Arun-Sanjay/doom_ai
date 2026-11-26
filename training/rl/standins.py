"""Stand-ins for the modules the RL core takes from other areas, plus toy environments.

Training-time scaffolding, never imported by the agent. Everything here is numpy only, so
rollout workers can import it without loading torch.

* ``StubOdometry``, ``StubMemoryMap``, ``StubEncoder`` follow the interfaces of the agent's
  odometry, memory map and encoder (same method names, shapes and dtypes as
  ``doomnet.spec``). They are simple versions: good enough to develop and unit-test the vector
  environment, the buffer and the loss, and to measure throughput before the real modules land.
* ``basic_meta`` reads the few static map facts the reward needs straight from a UDMF map. It is
  used only when ``training.mapmeta`` is not importable.
* ``ToyBandit`` and ``ToyRecall`` are tiny environments with the ``doom_env`` interface
  (``reset``, ``step``, ``privileged_state``). They return observations in the wrapper's format,
  so the whole pipeline (workers, encoder, buffer, loss) runs on them unchanged.
* ``ToyFragile`` is a bandit that crashes or hangs once on request: the failure injection for the
  tests of worker recovery.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from training import paths  # noqa: F401  (puts submission/ on sys.path)
from doomnet import spec

# ---------------------------------------------------------------------------------------------
# Odometry, memory map, encoder
# ---------------------------------------------------------------------------------------------
_TURN_SLOW = 360.0 * 320 / 65536      # degrees per tic while a turn has been held under 6 tics
_TURN_FAST = 360.0 * 640 / 65536      # degrees per tic afterwards
_DEAD_TICS = 7                        # the engine ignores input for the first tics of an episode
_RAY_ANGLES = np.arange(spec.NUM_RAYS) * (2.0 * np.pi / spec.NUM_RAYS)
_HALF = spec.MAP_SIZE // 2


class StubOdometry:
    """Pose in the start frame from the agent's own actions and reported ego velocity.

    Heading uses the engine's turn rates; position integrates the observed velocity over the four
    tics of a step (no friction model, so it drifts more than the real module).
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.x = self.y = self.theta = 0.0
        self.step = 0
        self._deg = 0.0
        self._held = 0
        self._tic = 0

    def update(self, prev_action, obs) -> None:
        a = np.asarray(prev_action)
        turn = int(a[4]) - int(a[5])
        for _ in range(spec.FRAME_SKIP):
            self._tic += 1
            self._held = self._held + 1 if turn else 0
            if self._tic > _DEAD_TICS:
                self._deg += (_TURN_SLOW if self._held < 6 else _TURN_FAST) * turn
        self.theta = math.radians(self._deg)
        vf, vr = float(obs["vitals"][14]), float(obs["vitals"][15])
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x += spec.FRAME_SKIP * (vf * c + vr * s)
        self.y += spec.FRAME_SKIP * (vf * s - vr * c)
        self.step += 1


class StubMemoryMap:
    """Occupancy memory in the start frame, array layout [channel, x index, y index].

    Free cells are marked along every ray up to its hit, the cell just past a wall or door hit is
    typed, and the agent's own cell is marked visited. ``crop`` returns the 2x2 max-pooled window
    centred on the agent's cell; cells outside the stored array read as zero.
    """

    _SAMPLES = np.arange(0.0, spec.RAY_RANGE + 1.0, spec.MAP_RES, dtype=np.float64)

    def __init__(self):
        self.grid = np.zeros((spec.MAP_CHANNELS, spec.MAP_SIZE, spec.MAP_SIZE), np.float32)

    def reset(self) -> None:
        self.grid[:] = 0.0

    @staticmethod
    def _cells(px, py):
        ix = np.floor(px / spec.MAP_RES).astype(np.int64) + _HALF
        iy = np.floor(py / spec.MAP_RES).astype(np.int64) + _HALF
        ok = (ix >= 0) & (ix < spec.MAP_SIZE) & (iy >= 0) & (iy < spec.MAP_SIZE)
        return ix, iy, ok

    def update(self, pose, obs, prev_action=None, exit_obs=None, step=0) -> None:
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        g = self.grid
        dist = np.asarray(obs["rays"]["distance"], np.float64) * spec.RAY_RANGE
        hit = np.asarray(obs["rays"]["hit_type"])
        c, s = np.cos(theta + _RAY_ANGLES), np.sin(theta + _RAY_ANGLES)
        t = np.minimum(self._SAMPLES[None, :], dist[:, None])
        ix, iy, ok = self._cells(x + c[:, None] * t, y + s[:, None] * t)
        g[spec.CH_FREE, ix[ok], iy[ok]] = 1.0
        ix, iy, ok = self._cells(x + c * (dist + 1.0), y + s * (dist + 1.0))
        for ch, code in ((spec.CH_WALL, spec.HIT_WALL), (spec.CH_DOOR, spec.HIT_DOOR), (spec.CH_LOCKED, spec.HIT_LOCKED)):
            m = ok & (hit == code)
            g[ch, ix[m], iy[m]] = 1.0
        g[spec.CH_RECENCY] *= 0.98
        ix, iy, ok = self._cells(np.array([x]), np.array([y]))
        if ok[0]:
            g[spec.CH_VISITED, ix[0], iy[0]] = 1.0
            g[spec.CH_RECENCY, ix[0], iy[0]] = 1.0
        if exit_obs is not None and float(exit_obs[0]) > 0.5:
            fx, fl = float(exit_obs[1]) * spec.RAY_RANGE, float(exit_obs[2]) * spec.RAY_RANGE
            ct, st = math.cos(theta), math.sin(theta)
            ix, iy, ok = self._cells(np.array([x + fx * ct - fl * st]), np.array([y + fx * st + fl * ct]))
            if ok[0]:
                g[spec.CH_EXIT, ix[0], iy[0]] = 1.0

    def crop(self, pose) -> np.ndarray:
        n = spec.CROP_SIZE * spec.CROP_MULT
        cell = spec.MAP_RES * spec.CROP_MULT
        i0 = spec.CROP_MULT * int(math.floor((float(pose[0]) + cell / 2) / cell)) + _HALF - n // 2 - 1
        j0 = spec.CROP_MULT * int(math.floor((float(pose[1]) + cell / 2) / cell)) + _HALF - n // 2 - 1
        win = np.zeros((spec.MAP_CHANNELS, n, n), np.float32)
        a0, a1 = max(i0, 0), min(i0 + n, spec.MAP_SIZE)
        b0, b1 = max(j0, 0), min(j0 + n, spec.MAP_SIZE)
        if a0 < a1 and b0 < b1:
            win[:, a0 - i0:a1 - i0, b0 - j0:b1 - j0] = self.grid[:, a0:a1, b0:b1]
        # 2x2 max pooling written as slices (several times faster than a reshape + max in numpy).
        return np.maximum(np.maximum(win[:, 0::2, 0::2], win[:, 1::2, 0::2]),
                          np.maximum(win[:, 0::2, 1::2], win[:, 1::2, 1::2]))


class StubEncoder:
    """Observation dict -> network inputs with the layout of ``doomnet.spec``."""

    _I = spec.SCALAR_INDEX

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._use_age = 16
        self._prev_use = 0

    def encode(self, obs, pose, prev_action, step, budget_steps, exit_obs) -> dict:
        rays = np.zeros((spec.NUM_RAYS, spec.RAY_FEATS), np.float32)
        rays[:, 0] = obs["rays"]["distance"]
        rays[np.arange(spec.NUM_RAYS), 1 + np.asarray(obs["rays"]["hit_type"])] = 1.0

        ve = obs["visible_entities"]
        mask = np.asarray(ve["mask"]).astype(bool)
        ents = np.zeros((spec.MAX_ENTITIES, spec.ENT_FEATS), np.float32)
        ents[:, :5] = ve["features"]
        ents[np.arange(spec.MAX_ENTITIES), 5 + np.asarray(ve["entity_type"])] = 1.0
        ents[~mask] = 0.0

        v = np.nan_to_num(np.asarray(obs["vitals"], np.float32))
        prev = np.zeros(spec.NUM_BUTTONS, np.float32) if prev_action is None else np.asarray(prev_action, np.float32)
        use = int(prev[7])
        self._use_age = 0 if (use and not self._prev_use) else min(self._use_age + 1, 16)
        self._prev_use = use

        i = self._I
        scal = np.zeros(spec.SCALAR_DIM, np.float32)
        scal[i["health"]], scal[i["armor"]] = v[0] / 100.0, v[1] / 100.0
        scal[i["weapon_0"] + int(np.clip(v[2], 0, 7))] = 1.0
        scal[i["sel_ammo"]:i["sel_ammo"] + 5] = np.clip(v[3:8] / 50.0, 0.0, 4.0)
        scal[i["key_blue_card"]:i["key_blue_card"] + 6] = v[8:14]
        scal[i["vel_forward"]:i["vel_forward"] + 3] = v[14:17] / 10.0
        scal[i["step_frac"]] = step / spec.STEP_NORM
        scal[i["budget_frac"]] = budget_steps / spec.STEP_NORM
        scal[i["budget_left"]] = np.clip((budget_steps - step) / spec.STEP_NORM, -1.0, 1.0)
        scal[i["sin_theta"]], scal[i["cos_theta"]] = math.sin(pose[2]), math.cos(pose[2])
        cell = spec.MAP_RES * spec.CROP_MULT
        scal[i["subcell_x"]] = ((pose[0] + cell / 2) % cell) / cell - 0.5
        scal[i["subcell_y"]] = ((pose[1] + cell / 2) % cell) / cell - 0.5
        scal[i["prev_fwd"]:i["prev_fwd"] + 9] = prev
        scal[i["use_age"]] = self._use_age / 16.0
        if exit_obs is not None and float(exit_obs[0]) > 0.5:
            scal[i["exit_visible"]:i["exit_visible"] + 3] = (1.0, float(exit_obs[1]), float(exit_obs[2]))
        return {"rays": rays, "ents": ents, "ent_mask": mask, "scal": scal}


# ---------------------------------------------------------------------------------------------
# Static map facts (fallback for training.mapmeta)
# ---------------------------------------------------------------------------------------------
KEY_THING_TYPES = frozenset({5, 6, 13, 38, 39, 40})
MONSTER_THING_TYPES = frozenset({3004, 9, 65, 3001, 3002, 58, 3006, 3005, 69, 3003, 64, 66, 67, 68, 71, 16, 7, 84})
EXIT_LINE_SPECIALS = frozenset({243, 244})


@dataclass
class BasicMeta:
    """The static facts the reward and the exit oracle need about one map."""

    n_keys: int = 0             # distinct key kinds on the map (the wrapper reports one flag per kind)
    n_monsters: int = 0
    exit_centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float64))
    known: bool = True          # False when the map format could not be read (totals are unknown)
    meta: object = None         # the full training.mapmeta.MapMeta when that module provided the facts
    pads: object = None         # exit-pad centres, filled on first use by deps.exit_observation


def basic_meta(wad: str, map_name: str = "MAP01") -> BasicMeta:
    """Count keys and monsters present at skill 3 and locate exit sectors of a UDMF map."""
    from doom_env.mapinfo import parse_udmf
    from doom_env.wad import map_lumps, read_wad

    lumps = map_lumps(read_wad(wad), map_name)
    if "TEXTMAP" not in lumps:
        return BasicMeta(known=False)
    b = parse_udmf(lumps["TEXTMAP"].decode("utf-8", "replace"))
    things = [t for t in b.get("thing", []) if t.get("skill3", False)]
    n_keys = len({t.get("type") for t in things if t.get("type") in KEY_THING_TYPES})   # distinct key kinds
    n_monsters = sum(1 for t in things if t.get("type") in MONSTER_THING_TYPES)

    verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in b.get("vertex", [])]
    sides = b.get("sidedef", [])
    per_sector: dict[int, list] = {}
    exit_pairs = []
    for line in b.get("linedef", []):
        secs = [int(sides[line[k]].get("sector", -1)) for k in ("sidefront", "sideback")
                if isinstance(line.get(k), int) and 0 <= line[k] < len(sides)]
        for sec in secs:
            per_sector.setdefault(sec, []).extend([verts[int(line["v1"])], verts[int(line["v2"])]])
        if int(line.get("special", 0) or 0) in EXIT_LINE_SPECIALS:
            exit_pairs.append(secs)
    # The exit region is the smaller of the two sectors a walk-over exit line separates.
    exit_sectors = {min(secs, key=lambda s: len(per_sector.get(s, ()))) for secs in exit_pairs if secs}
    centres = [np.unique(np.array(per_sector[s], np.float64), axis=0).mean(axis=0) for s in sorted(exit_sectors)]
    return BasicMeta(n_keys, n_monsters, np.array(centres, np.float64).reshape(-1, 2))


# ---------------------------------------------------------------------------------------------
# Toy environments
# ---------------------------------------------------------------------------------------------
def blank_observation() -> dict:
    """An observation in the wrapper's format with nothing in view (no pixel buffers)."""
    vitals = np.zeros(spec.NUM_VITALS, np.float32)
    vitals[0] = 100.0
    return {
        "vitals": vitals,
        "rays": {"distance": np.ones(spec.NUM_RAYS, np.float32), "hit_type": np.zeros(spec.NUM_RAYS, np.int64)},
        "visible_entities": {
            "features": np.zeros((spec.MAX_ENTITIES, 5), np.float32),
            "entity_type": np.zeros(spec.MAX_ENTITIES, np.int64),
            "mask": np.zeros(spec.MAX_ENTITIES, np.int8),
        },
    }


class ToyBandit:
    """Two-armed bandit with the ``doom_env`` interface.

    The arms are the two directions of the turn head. Each step one arm is the good one and pays
    +1; everything else pays 0. With ``contextual=True`` the good arm is redrawn every step and
    shown in the ray distances (0.25 = turn left, 0.75 = turn right); otherwise the good arm is
    always "turn left". ``p_terminate`` ends an episode early with that probability per step
    (termination); the tick cap ends it by truncation.
    """

    native_reward = True       # the vector environment uses the reward returned by step()

    def __init__(self, episode_ticks: int = 128, contextual: bool = True, p_terminate: float = 0.0):
        self.episode_ticks = int(episode_ticks)
        self.contextual = contextual
        self.p_terminate = p_terminate
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.good = 1
        self.hits = 0

    def _draw(self) -> None:
        self.good = int(self.rng.integers(1, 3)) if self.contextual else 1

    def _obs(self) -> dict:
        obs = blank_observation()
        obs["rays"]["distance"][:] = 0.25 if self.good == 1 else 0.75
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t, self.hits = 0, 0
        self._draw()
        return self._obs(), {"tic": 0, "exit_reached": False}

    def step(self, action):
        arm = int(spec.buttons_to_heads(action)[spec.HEAD_TURN])
        reward = 1.0 if arm == self.good else 0.0
        self.hits += int(reward)
        self.t += 1
        terminated = bool(self.rng.random() < self.p_terminate)
        truncated = (not terminated) and self.t * spec.FRAME_SKIP >= self.episode_ticks
        self._draw()
        return self._obs(), reward, terminated, truncated, {"tic": self.t * spec.FRAME_SKIP, "exit_reached": False}

    def teacher_label(self):
        """(head labels (6,), sub-goal, target cell, mask): the action an oracle would take now."""
        heads = np.zeros(len(spec.HEAD_SIZES), np.int64)
        heads[spec.HEAD_TURN] = self.good
        return heads, 0, 0, True

    def privileged_state(self) -> dict:
        return {"tic": self.t * spec.FRAME_SKIP, "episode_ticks": self.episode_ticks, "x": 0.0, "y": 0.0,
                "health": 100.0, "kills": self.hits, "damage_taken": 0.0, "dead": False, "exit_reached": False}

    def close(self) -> None:
        pass


class ToyRecall(ToyBandit):
    """Bandit whose cue is visible only in the first observation of an episode.

    The good arm is fixed for the episode and shown at step 0; afterwards the rays read 0.5.
    Earning reward on later steps needs the recurrent state to carry the cue, so this exercises
    state resets at episode starts and truncated backpropagation through time.
    """

    def _draw(self) -> None:
        if self.t == 0:
            self.good = int(self.rng.integers(1, 3))

    def _obs(self) -> dict:
        obs = blank_observation()
        obs["rays"]["distance"][:] = (0.25 if self.good == 1 else 0.75) if self.t == 0 else 0.5
        return obs


FRAGILE_ENV = "DOOMRL_FRAGILE"     # names the file that arms ToyFragile (worker processes inherit the variable)


class ToyFragile(ToyBandit):
    """A bandit that fails once on request, for the tests of worker recovery.

    Before every step it looks for the file named by the environment variable ``DOOMRL_FRAGILE``.
    If the file is there it is taken away (a rename, so exactly one environment in one process
    fails, once) and the failure its text asks for happens: "exit" ends the whole process at once,
    as a crash of the worker would; "hang" never returns from the step, as a call into a dead
    engine does.
    """

    def step(self, action):
        path = os.environ.get(FRAGILE_ENV)
        if path and os.path.exists(path):
            taken = f"{path}.{os.getpid()}"
            try:
                os.rename(path, taken)
                with open(taken) as f:
                    mode = f.read().strip()
                os.remove(taken)
            except OSError:
                mode = ""
            if mode == "exit":
                os._exit(3)
            if mode == "hang":
                threading.Event().wait()
        return super().step(action)


TOY_ENVS = {"toy:bandit": lambda ticks: ToyBandit(ticks, contextual=True),
            "toy:fixed": lambda ticks: ToyBandit(ticks, contextual=False),
            "toy:episodic": lambda ticks: ToyBandit(ticks, contextual=True, p_terminate=0.05),
            "toy:recall": lambda ticks: ToyRecall(ticks),
            "toy:fragile": lambda ticks: ToyFragile(ticks, contextual=True)}
