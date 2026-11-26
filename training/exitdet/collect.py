"""Collect labelled frames for the exit detector.

Training-time scaffolding. Never imported by the agent in submission/.

    python -m training.exitdet.collect --sets unseen,heldout,val,train --workers 3

Maps. Every set is built from fresh generator seeds of tiers 0-4 (SETS below; the training set of
the policy uses seeds 0-300 and the frozen suites 8,000,000 upwards). A grid that is in a frozen
suite or in another detector set is refused, so the sets share no layout. Part of each set keeps
the plain reference look; the rest gets random room floors, walls, ceilings and light levels drawn
from one side of the texture split in `textures.py`. Half of the maps also get extra red keys, red
doors and imps (`textures.add_distractors`), because a far red key or the foot of a red door is
what a far exit pad is most easily confused with.

    train     textures of the training side, 35% reference look
    val       same recipe, other maps: watched during training
    calib     same recipe, other maps: with val, the frames the decision threshold is chosen on
    heldout   same recipe, other maps: 50% reference look
    unseen    every room floor, wall, ceiling and pad wall from the held-out textures

Poses. One engine is started per map. The player is then teleported (the engine's `warp` console
command) to a random spot of a random floor cell and driven by one of five scripted behaviours for
a few dozen steps; a new episode and a new spot follow. Frames are taken at random steps, so they
include turning, walking (the view bobs by up to 8 units), the weapon rising, firing and being hit.
In some episodes the player is given a shotgun or a chaingun, or uses the fist, because each weapon
covers a different part of the lower frame.

    exit_view   start in sight of the pad at a distance drawn from four bands, turn until the pad
                sits at a random bearing inside or just outside the view, then move about
    approach    face the pad and walk onto it
    spin        turn a full circle on the spot
    thing_view  like exit_view, aimed at a red key, a red door, a monster (and shoot at it), another
                key or door, or a pickup
    wander      walk where the range rays show room, open doors, shoot at monsters now and then

These behaviours read the true pose and the map; they only choose where the camera goes. Labels
come from `training.exit_label.exit_observation`, tags (what else is in view) from the engine's
object list and the map geometry. Bearings are compared through dot and cross products.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np

from training import paths
from training.exit_label import exit_observation
from training.exitdet import textures
from training.exitdet.data import (FIELDS, HURT_MIN_FLASH, KIND_INDEX, KINDS, PAIRS, TAG_BIT, TAGS, ShardWriter, describe,
                                   empty_fields, load_set)
from training.mapgen import layouts, udmf
from training.mapgen.cli import frozen_grids, grid_hash

DATA_DIR = paths.ROOT / "data" / "exitdet"

# name -> first generator seed, texture side of the random looks, share of maps left in the
# reference look, default number of maps. Planned in this order, each set refusing the grids of the
# sets before it: the evaluation sets come first, so the later sets can grow without changing them.
SETS = {
    "unseen": dict(seed=2_500_000, side="heldout", reference=0.0, maps=320),
    "heldout": dict(seed=2_000_000, side="train", reference=0.5, maps=320),
    "val": dict(seed=1_500_000, side="train", reference=0.4, maps=160),
    "train": dict(seed=1_000_000, side="train", reference=0.35, maps=1600),
    "calib": dict(seed=1_750_000, side="train", reference=0.4, maps=400),
}
TIER_SHARE = {0: 0.15, 1: 0.15, 2: 0.20, 3: 0.25, 4: 0.25}
EDIT_SHARE = 0.5                 # maps that get extra red keys, red doors and imps
FRAMES_PER_MAP = 64
SEGMENT_FRAMES = {"exit_view": 16, "approach": 16, "spin": 10, "thing_view": 10, "wander": 12}   # most frames per episode
MAX_SEGMENTS = 60
EPISODE_TICKS = 4200

FWD, BACK, LEFT, RIGHT, TURN_L, TURN_R, ATTACK, USE, WEAPON = range(9)
CELL = float(udmf.CELL)
WARP_JITTER = 14                 # the player (radius 16) stays inside its 64-unit cell
VIEW_BANDS = (((48.0, 200.0), 0.18), ((200.0, 400.0), 0.34), ((400.0, 640.0), 0.40), ((640.0, 900.0), 0.08))
TAN_TAG = math.tan(math.radians(48.0))     # half-angle inside which a thing counts as "in the frame"
SIN_ALIGNED = math.sin(math.radians(8.0))
COS_LOST = math.cos(math.radians(58.0))
TAG_RANGE = {"red_key": 800.0, "other_key": 800.0, "imp_fireball": 1000.0, "other_projectile": 1000.0,
             "blood": 500.0, "door": 900.0, "surface": 700.0}
RED_KEYS = ("RedCard", "RedSkull")
OTHER_KEYS = ("BlueCard", "YellowCard", "BlueSkull", "YellowSkull")
DOOR_SHUT = 96.0                 # a door whose opening is below this still shows most of its face
# Projection of the 160 x 120 frame, measured in the engine: 90 degrees across the columns (80 pixels
# per unit of left / forward), and rows stretched by the engine's 1.2 pixel aspect (96 pixels per
# unit of height / forward). A floor point d units ahead is on row 60 + 96 * eye height / d.
FOCAL_X, FOCAL_Y, HORIZON, CENTRE = 80.0, 96.0, 60.0, 80.0
EYE_LOW, EYE_HIGH = 33.0, 49.0   # eye height above the floor: 41 units, plus or minus 8 while walking
PAD_RANGE = 950.0
_PAD_GRID = np.array([(i, j) for i in (-1.0, -0.5, 0.0, 0.5, 1.0) for j in (-1.0, -0.5, 0.0, 0.5, 1.0)]) * (CELL / 2 - 1.0)

# weights of the behaviours, with and without a pad that can be seen from some floor cell
KIND_WEIGHTS_PAD = (("exit_view", 0.52), ("approach", 0.06), ("spin", 0.05), ("thing_view", 0.18), ("wander", 0.19))
KIND_WEIGHTS_BARE = (("thing_view", 0.40), ("wander", 0.45), ("spin", 0.15))
# What the player holds in an episode. Maps hand out shotguns and chainguns, and each weapon covers
# a different part of the lower frame (the pumping shotgun reaches the middle of it).
WEAPONS = (("Pistol", 0.62), ("Shotgun", 0.16), ("Chaingun", 0.12), ("Fist", 0.10))
THING_WEIGHTS = (("red_key", 0.30), ("red_door", 0.18), ("monster", 0.30), ("other_key", 0.07), ("other_door", 0.08),
                 ("pickup", 0.07))


# ---- planning ------------------------------------------------------------------------------------------
def make_job(set_name: str, layout: dict, index: int) -> dict:
    """One map of a set: the layout with its edits and build options, all drawn from a seed of its own."""
    cfg = SETS[set_name]
    tier, seed = layout["tier"], layout["seed"]
    rng = random.Random(f"exitdet/{set_name}/{tier}/{seed}")
    grid, edits = layout["grid"], {}
    if rng.random() < EDIT_SHARE:
        grid, edits = textures.add_distractors(rng, grid, tier)
    reference = rng.random() < cfg["reference"]
    options = dict(layout["options"])
    if not reference:
        options = textures.detector_options(rng, grid, cfg["side"], options)
    return {"index": index, "set": set_name, "name": f"{set_name}_t{tier}_{seed}", "tier": tier, "seed": seed,
            "grid": grid, "base_grid_sha1": grid_hash(layout["grid"]), "options": options,
            "look": "reference" if reference else "random", "side": "train" if reference else cfg["side"],
            "edits": edits, "rng_seed": rng.getrandbits(32)}


def plan(sizes: dict | None = None) -> dict[str, list[dict]]:
    """The maps of every set. Deterministic; no grid appears in two sets or in a frozen suite."""
    sizes = sizes or {}
    seen = set(frozen_grids())
    out: dict[str, list[dict]] = {}
    for name, cfg in SETS.items():
        total = int(sizes.get(name, cfg["maps"]))
        jobs: list[dict] = []
        for tier, share in TIER_SHARE.items():
            want, seed = int(round(total * share)), cfg["seed"]
            while want > 0:
                layout = layouts.generate(tier, seed)
                seed += 1
                key = grid_hash(layout["grid"])
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(make_job(name, layout, len(jobs)))
                want -= 1
        out[name] = jobs
    return out


# ---- geometry --------------------------------------------------------------------------------------------
def ego(px: float, py: float, heading: float, pts) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 2) -> forward and left coordinates in the player's frame."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = pts[:, 0] - px, pts[:, 1] - py
    return dx * c + dy * s, -dx * s + dy * c


def sight_clear(env, origin, pts) -> np.ndarray:
    """Is the straight line from `origin` to each point free of lines that block sight right now?"""
    from doom_env.sensors import _segments_blocked

    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    geo = env._geometry
    idx = np.flatnonzero(geo.sight_block)
    return ~_segments_blocked(np.asarray(origin, np.float64), pts, geo.p1[idx], geo.p2[idx])


class Scene:
    """Static facts of one map that the behaviours and the tags need."""

    def __init__(self, env, meta, job: dict):
        self.env, self.meta = env, meta
        rows = udmf.parse_grid(job["grid"])
        self.h, self.w = len(rows), len(rows[0])
        sector, kinds = udmf.assign_sectors(rows)
        self.sector = np.array(sector, np.int64)
        themes = job["options"].get("themes") or udmf.DEFAULT_THEMES
        exit_wall = job["options"].get("exit_wall")
        floor_red = np.zeros(len(kinds), bool)
        wall_red = np.zeros(len(kinds), bool)
        for i, (kind, _) in enumerate(kinds):
            floor, wall = themes[i % len(themes)]
            floor_red[i] = kind == "room" and textures.is_red_floor(floor)
            wall_red[i] = (kind == "room" and textures.is_red_wall(wall)) or \
                          (kind == "exit" and textures.is_red_wall(exit_wall or wall))
        self.floor_red, self.wall_red = floor_red, wall_red

        def centres(test) -> np.ndarray:
            pts = [udmf.cell_centre(c, r, self.h) for r in range(self.h) for c in range(self.w) if test(c, r)]
            return np.array(pts, np.float64).reshape(-1, 2)

        self.cells = centres(lambda c, r: rows[r][c] in ".P")                    # where the player may be put
        self.red_floor_pts = centres(lambda c, r: sector[r][c] >= 0 and floor_red[sector[r][c]])
        self.red_wall_pts = centres(lambda c, r: sector[r][c] >= 0 and wall_red[sector[r][c]])
        self.pads = np.array([e.centre for e in meta.exits if e.sector >= 0], np.float64).reshape(-1, 2)

        # Door faces: a point just outside each side of a door cell that borders open floor.
        pts, secs, red = [], [], []
        for door in meta.doors:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (door.centre[0] + 33.0 * dx, door.centre[1] + 33.0 * dy)
                s = self.sector_at(*p)
                if s >= 0 and kinds[s][0] == "room":
                    pts.append(p)
                    secs.append(door.sector)
                    red.append(door.colour == "red")
        self.door_pts = np.array(pts, np.float64).reshape(-1, 2)
        self.door_sector = np.array(secs, np.int64)
        self.door_red = np.array(red, bool)

        def things(*classes_or_kinds) -> np.ndarray:
            pts = [(t.x, t.y) for t in meta.all_things if t.cls in classes_or_kinds or t.kind in classes_or_kinds]
            return np.array(pts, np.float64).reshape(-1, 2)

        self.targets = {"red_key": things(*RED_KEYS), "other_key": things(*OTHER_KEYS), "monster": things("monster"),
                        "pickup": things("health", "armor", "ammo", "weapon"),
                        "red_door": self.door_pts[self.door_red], "other_door": self.door_pts[~self.door_red]}
        self._views: dict = {}
        self.pad_views = [self.views(p) for p in self.pads]                    # needs a reset environment

    def sector_at(self, x: float, y: float) -> int:
        col, row = int(x // CELL), self.h - 1 - int(y // CELL)
        return int(self.sector[row, col]) if 0 <= row < self.h and 0 <= col < self.w else -1

    def views(self, target) -> tuple[np.ndarray, np.ndarray]:
        """(distance, in line of sight) from every floor cell centre to a point, doors as they are at reset."""
        key = (float(target[0]), float(target[1]))
        if key not in self._views:
            dist = np.hypot(self.cells[:, 0] - key[0], self.cells[:, 1] - key[1])
            clear = sight_clear(self.env, key, self.cells) if len(self.cells) else np.zeros(0, bool)
            self._views[key] = (dist, clear)
        return self._views[key]

    def view_cell(self, rng, target, lo: float, hi: float) -> np.ndarray | None:
        """A floor cell centre with a clear line to `target` at a distance in [lo, hi], or None."""
        dist, clear = self.views(target)
        idx = np.flatnonzero(clear & (dist >= lo) & (dist <= hi))
        return self.cells[int(rng.choice(idx))] if len(idx) else None

    def pad_seen_from_somewhere(self) -> bool:
        return any(bool((clear & (dist >= 48.0) & (dist <= 900.0)).any()) for dist, clear in self.pad_views)

    def pad_pixels(self, st: dict, rgb: np.ndarray) -> int:
        """Pad-coloured pixels inside the screen box a pad in front of the player can cover.

        The label is geometric; this says whether the renderer actually drew the pad. The box is
        the projection of the pad square for every eye height the view bob allows, grown by one
        pixel. A pixel is pad-coloured when it is red with almost no green or blue, which is what
        the pad looks like at every light level until it fades to black.
        """
        best = 0
        px, py, heading = st["x"], st["y"], math.radians(st["angle"])
        for pad in self.pads:
            fwd, left = ego(px, py, heading, pad + _PAD_GRID)
            ok = fwd >= 8.0
            if not ok.any() or float(np.hypot(pad[0] - px, pad[1] - py)) > PAD_RANGE:
                continue
            fwd, left = fwd[ok], left[ok]
            cols = CENTRE - FOCAL_X * left / fwd
            c0, c1 = int(math.floor(cols.min())) - 1, int(math.ceil(cols.max())) + 2
            r0 = int(math.floor(HORIZON + FOCAL_Y * EYE_LOW / fwd.max())) - 1
            r1 = int(math.ceil(HORIZON + FOCAL_Y * EYE_HIGH / fwd.min())) + 2
            box = rgb[max(r0, int(HORIZON)):min(r1, rgb.shape[0]), max(c0, 0):min(c1, rgb.shape[1])].astype(np.int32)
            if box.size:
                red = (box[..., 0] >= 40) & (np.maximum(box[..., 1], box[..., 2]) * 5 <= box[..., 0] * 2)
                best = max(best, int(red.sum()))
        return best

    # ---- tags
    def tags(self, st: dict, obs: dict, flash: float, fired: bool) -> int:
        """Bit set of TAGS for the frame just rendered at the pose in `st`."""
        from doom_env.sensors import classify

        env = self.env
        px, py, heading = st["x"], st["y"], math.radians(st["angle"])
        bits = 0
        pts, names = [], []
        state = env._game.get_state()
        for o in (state.objects if state is not None else ()):
            name = o.name
            if name in RED_KEYS:
                tag = "red_key"
            elif name in OTHER_KEYS:
                tag = "other_key"
            elif name == "DoomImpBall":
                tag = "imp_fireball"
            elif name == "Blood" or o.category in ("Gibs", "Gore"):
                tag = "blood"
            elif name != "BulletPuff" and classify(name, o.category) == "projectile":
                tag = "other_projectile"
            else:
                continue
            pts.append((o.position_x, o.position_y))
            names.append(tag)
        shut = (st["sector_ceiling"][self.door_sector] - st["sector_floor"][self.door_sector]) < DOOR_SHUT \
            if len(self.door_pts) else np.zeros(0, bool)
        for p, red in zip(self.door_pts[shut], self.door_red[shut]):
            pts.append((p[0], p[1]))
            names.append("red_door" if red else "other_door")
        for group, tag in ((self.red_floor_pts, "red_floor"), (self.red_wall_pts, "red_wall")):
            for p in group:
                pts.append((p[0], p[1]))
                names.append(tag)
        if pts:
            arr = np.array(pts, np.float64)
            fwd, left = ego(px, py, heading, arr)
            dist = np.hypot(fwd, left)
            reach = np.array([TAG_RANGE["door" if "door" in n else "surface" if n.startswith("red_f") or n == "red_wall"
                                        else n] for n in names])
            near = (fwd > 4.0) & (np.abs(left) <= fwd * TAN_TAG + 16.0) & (dist <= reach)
            idx = np.flatnonzero(near)
            if len(idx):
                clear = sight_clear(env, (px, py), arr[idx])
                for k in idx[clear]:
                    bits |= TAG_BIT[names[k]]
        here = self.sector_at(px, py)
        if here >= 0:
            if self.floor_red[here]:
                bits |= TAG_BIT["red_floor"]
            if self.wall_red[here]:
                bits |= TAG_BIT["red_wall"]
        ents = obs["visible_entities"]
        kinds = np.asarray(ents["entity_type"])[np.asarray(ents["mask"]).astype(bool)]
        if ((kinds >= 1) & (kinds <= 3)).any():
            bits |= TAG_BIT["monster"]
        if flash >= HURT_MIN_FLASH:
            bits |= TAG_BIT["hurt"]
        if fired:
            bits |= TAG_BIT["firing"]
        return bits


# ---- behaviours --------------------------------------------------------------------------------------------
class Gaze:
    """Turn until a target sits at a chosen bearing, then move about while it stays roughly there."""

    def __init__(self, rng, target, approach: bool = False, attack: float = 0.04, dwell_p: float = 0.55):
        self.rng, self.target, self.approach, self.attack = rng, np.asarray(target, np.float64), approach, attack
        self.dwell_p = dwell_p
        self.buttons: tuple = ()
        self.left = 0
        self.aligned = False
        self.offset = 0.0
        self.retarget()

    def retarget(self) -> None:
        spread = 8.0 if self.approach else 50.0
        self.offset = math.radians(self.rng.uniform(-spread, spread))    # wanted bearing of the target, left positive
        self.aligned = False

    @property
    def record_p(self) -> float:
        return self.dwell_p if self.aligned else 0.2

    def act(self, obs, st) -> np.ndarray:
        rng = self.rng
        a = np.zeros(9, np.int8)
        fwd, left = ego(st["x"], st["y"], math.radians(st["angle"]), self.target)
        fwd, left = float(fwd[0]), float(left[0])
        dist = max(math.hypot(fwd, left), 1e-6)
        cb, sb = math.cos(self.offset), math.sin(self.offset)
        sin_e, cos_e = (left * cb - fwd * sb) / dist, (fwd * cb + left * sb) / dist   # bearing minus wanted bearing
        if self.aligned and (fwd < dist * COS_LOST or (self.approach and abs(sin_e) > 0.25)):
            self.retarget()
        if not self.aligned:
            if cos_e > 0.0 and abs(sin_e) <= SIN_ALIGNED:
                self.aligned = True
            else:
                a[TURN_L if sin_e > 0.0 else TURN_R] = 1
                if rng.random() < 0.3:
                    a[int(rng.choice((FWD, LEFT, RIGHT)))] = 1
                return a
        if self.left <= 0:
            if self.approach:
                self.buttons, self.left = (FWD,), 3
            else:
                moves = ((FWD,), (BACK,), (LEFT,), (RIGHT,), (FWD, LEFT), (FWD, RIGHT), (), (TURN_L,), (TURN_R,), None)
                weights = (0.24, 0.12, 0.14, 0.14, 0.05, 0.05, 0.10, 0.05, 0.05, 0.06)
                move = moves[int(rng.choice(len(moves), p=weights))]
                if move is None:
                    self.retarget()
                    a[TURN_L if rng.random() < 0.5 else TURN_R] = 1
                    return a
                self.buttons = move
                self.left = int(rng.integers(1, 3)) if move in ((TURN_L,), (TURN_R,)) else int(rng.integers(1, 7))
        for b in self.buttons:
            a[b] = 1
        self.left -= 1
        if rng.random() < self.attack:
            a[ATTACK] = 1
        return a


class Spin:
    """Turn on the spot through a full circle, sometimes drifting."""

    record_p = 0.5

    def __init__(self, rng):
        self.rng = rng
        self.turn = TURN_L if rng.random() < 0.5 else TURN_R
        self.move = ((), (), (FWD,), (LEFT,), (RIGHT,), (BACK,))[int(rng.integers(6))]

    def act(self, obs, st) -> np.ndarray:
        a = np.zeros(9, np.int8)
        a[self.turn] = 1
        if self.move and self.rng.random() < 0.3:
            a[self.move[0]] = 1
        return a


class Wander:
    """Walk where the range rays show room; open doors; shoot at monsters now and then."""

    record_p = 0.3

    def __init__(self, rng):
        self.rng = rng
        self.turn = int(rng.integers(-13, 14))        # steps of turning left (positive) or right (negative)
        self.use_wait = 0

    def act(self, obs, st) -> np.ndarray:
        rng = self.rng
        a = np.zeros(9, np.int8)
        dist = np.asarray(obs["rays"]["distance"], np.float64) * 1024.0
        hit = np.asarray(obs["rays"]["hit_type"])
        front = float(min(dist[0], dist[1], dist[63]))
        self.use_wait = max(0, self.use_wait - 1)
        if int(hit[0]) == 2 and self.turn == 0:       # an ordinary door straight ahead: go and open it
            if dist[0] > 60.0:
                a[FWD] = 1
            elif self.use_wait == 0:
                a[USE] = 1
                self.use_wait = 14
            return a
        if self.turn == 0:
            if front < rng.uniform(56.0, 140.0):
                sign = 1 if dist[4:20].mean() > dist[45:61].mean() else -1
                if rng.random() < 0.15:
                    sign = -sign
                self.turn = sign * int(rng.integers(2, 9))
            elif rng.random() < 0.07:
                self.turn = int(rng.integers(-6, 7))
        if self.turn:
            a[TURN_L if self.turn > 0 else TURN_R] = 1
            self.turn -= 1 if self.turn > 0 else -1
        if front > 48.0 and (not a[TURN_L] and not a[TURN_R] or rng.random() < 0.6):
            a[FWD] = 1
        if rng.random() < 0.06:
            a[LEFT if rng.random() < 0.5 else RIGHT] = 1
        ents = obs["visible_entities"]
        kinds = np.asarray(ents["entity_type"])[np.asarray(ents["mask"]).astype(bool)]
        if rng.random() < (0.3 if ((kinds >= 1) & (kinds <= 3)).any() else 0.02):
            a[ATTACK] = 1
        if rng.random() < 0.01:
            a[WEAPON] = 1
        return a


# ---- one map -----------------------------------------------------------------------------------------------
def _choose(rng, weighted):
    names, weights = zip(*weighted)
    w = np.array(weights, np.float64)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _segment(scene: Scene, rng):
    """Pick a behaviour and where it starts: (kind, behaviour, start point, steps) or None."""
    kind = _choose(rng, KIND_WEIGHTS_PAD if scene.pad_seen_from_somewhere() else KIND_WEIGHTS_BARE)
    if kind in ("exit_view", "approach") or (kind == "spin" and len(scene.pads) and rng.random() < 0.7):
        pad = scene.pads[int(rng.integers(len(scene.pads)))]
        bands = [band for band, _ in VIEW_BANDS]
        order = rng.choice(len(bands), size=len(bands), replace=False, p=[w for _, w in VIEW_BANDS])
        start = next((s for s in (scene.view_cell(rng, pad, *bands[i]) for i in order) if s is not None), None)
        if start is None:
            return None
        if kind == "spin":
            return kind, Spin(rng), start, 28
        if kind == "approach":
            return kind, Gaze(rng, pad, approach=True), start, 70
        return kind, Gaze(rng, pad), start, int(rng.integers(28, 46))
    if kind == "thing_view":
        have = [(name, w) for name, w in THING_WEIGHTS if len(scene.targets[name])]
        if have:
            name = _choose(rng, have)
            target = scene.targets[name][int(rng.integers(len(scene.targets[name])))]
            start = scene.view_cell(rng, target, 70.0, 420.0 if rng.random() < 0.7 else 700.0)
            if start is not None:
                if name == "monster":        # stay longer: the monster has to wake up, attack and bleed
                    return kind, Gaze(rng, target, attack=0.3, dwell_p=0.3), start, int(rng.integers(40, 64))
                return kind, Gaze(rng, target), start, int(rng.integers(26, 42))
        kind = "wander"
    if not len(scene.cells):
        return None
    start = scene.cells[int(rng.integers(len(scene.cells)))]
    if kind == "spin":
        return kind, Spin(rng), start, 28
    return "wander", Wander(rng), start, int(rng.integers(40, 70))


def collect_map(job: dict) -> dict:
    """Worker entry: play one map and return its frames. Never raises: an error is reported in the result."""
    try:
        return _collect_map(job)
    except Exception:  # noqa: BLE001 - one broken map must not stop the run
        return {"index": job["index"], "error": traceback.format_exc(), "rgb": np.zeros((0, 120, 160, 3), np.uint8),
                "fields": empty_fields(0), "wall_s": 0.0}


def _collect_map(job: dict) -> dict:
    from doom_env import make_env
    from training.mapmeta import load_meta

    began = time.perf_counter()
    rng = np.random.default_rng(job["rng_seed"])
    wad = udmf.write_wad(job["grid"], job["wad"], **job["options"])
    meta = load_meta(wad)
    env = make_env(str(wad), episode_ticks=EPISODE_TICKS)
    frames: list[np.ndarray] = []
    rec = {k: [] for k in FIELDS}
    want = int(job["frames"])
    steps = segments = failed = 0
    try:
        env.reset(seed=int(job["rng_seed"]) % (2**31))
        scene = Scene(env, meta, job)
        while len(frames) < want and segments < MAX_SEGMENTS:
            segments += 1
            seg = _segment(scene, rng)
            if seg is None:
                failed += 1
                continue
            kind, behaviour, start, length = seg
            x = int(round(start[0])) + int(rng.integers(-WARP_JITTER, WARP_JITTER + 1))
            y = int(round(start[1])) + int(rng.integers(-WARP_JITTER, WARP_JITTER + 1))
            obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
            env._game.send_game_command(f"warp {x} {y}")
            weapon = _choose(rng, WEAPONS)
            if weapon != "Pistol":
                if weapon != "Fist":
                    env._game.send_game_command(f"give {weapon}")
                env._game.send_game_command(f"use {weapon}")
            obs, _, terminated, truncated, _ = env.step(np.zeros(9, np.int8))
            st = env.privileged_state()
            if terminated or truncated or abs(st["x"] - x) > 3.0 or abs(st["y"] - y) > 3.0:
                failed += 1
                continue
            health, flash, fired, taken = st["health"], 0.0, False, 0
            for t in range(length + 1):
                if t == 0:
                    keep = rng.random() < 0.25                   # the first frame: the weapon is still rising
                else:
                    action = behaviour.act(obs, st)
                    obs, _, terminated, truncated, _ = env.step(action)
                    steps += 1
                    if terminated or truncated:
                        break
                    st = env.privileged_state()
                    # the engine's damage counter: plus the damage taken, minus one per tic
                    flash = min(100.0, max(0.0, flash + max(0.0, health - st["health"]) - 4.0))
                    health, fired = st["health"], bool(action[ATTACK])
                    keep = rng.random() < behaviour.record_p
                if not keep:
                    continue
                visible, x_fwd, y_left, ambiguous = exit_observation(env, meta)
                pad_dist, pad_ego, pad_los = -1.0, (0.0, 0.0), False
                if len(scene.pads):
                    fwd, left = ego(st["x"], st["y"], math.radians(st["angle"]), scene.pads)
                    k = int(np.argmin(np.hypot(fwd, left)))
                    pad_dist, pad_ego = float(math.hypot(fwd[k], left[k])), (float(fwd[k]), float(left[k]))
                    pad_los = bool(sight_clear(env, (st["x"], st["y"]), scene.pads[k:k + 1])[0])
                frames.append(obs["rgb"].copy())
                rec["visible"].append(visible)
                rec["ambiguous"].append(ambiguous)
                rec["pos"].append((x_fwd, y_left))
                rec["pad_dist"].append(pad_dist)
                rec["pad_ego"].append(pad_ego)
                rec["pad_los"].append(pad_los)
                rec["pad_px"].append(min(scene.pad_pixels(st, obs["rgb"]), 65535))
                rec["tags"].append(scene.tags(st, obs, flash, fired))
                rec["flash"].append(int(flash))
                rec["weapon"].append(int(obs["vitals"][2]))
                rec["kind"].append(KIND_INDEX[kind])
                rec["map_id"].append(job["index"])
                rec["step"].append(t)
                taken += 1
                if taken >= SEGMENT_FRAMES[kind] or len(frames) >= want:
                    break
    finally:
        env.close()
    fields = {k: np.asarray(rec[k], dt).reshape((-1, 2) if k in PAIRS else (-1,)) for k, dt in FIELDS.items()}
    rgb = np.stack(frames) if frames else np.zeros((0, 120, 160, 3), np.uint8)
    return {"index": job["index"], "error": "", "rgb": rgb, "fields": fields, "steps": steps, "segments": segments,
            "failed_segments": failed, "wall_s": time.perf_counter() - began}


# ---- one set -----------------------------------------------------------------------------------------------
def collect_set(name: str, jobs: list[dict], out_dir, workers: int = 3, frames_per_map: int = FRAMES_PER_MAP,
                progress: bool = True) -> dict:
    """Play every map of a set and write its shards, maps.jsonl and summary.json. Returns the summary."""
    import multiprocessing as mp

    from doom_env.limits import THREAD_ENV

    out_dir = Path(out_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    jobs = [{**job, "frames": frames_per_map, "wad": str(out_dir / "maps" / f"{job['name']}.wad")} for job in jobs]
    writer = ShardWriter(out_dir)
    rows, errors, began, steps = [], [], time.perf_counter(), 0

    def take(result: dict) -> None:
        nonlocal steps
        job = jobs[result["index"]]
        if result["error"]:
            errors.append({"map": job["name"], "error": result["error"][-1500:]})
        writer.add(result["rgb"], result["fields"])
        steps += result.get("steps", 0)
        rows.append({k: job[k] for k in ("index", "name", "tier", "seed", "look", "side", "edits", "grid", "options",
                                         "base_grid_sha1", "rng_seed")}
                    | {"frames": int(len(result["rgb"])), "visible": int(result["fields"]["visible"].sum()),
                       "textures": textures.option_textures(job["options"]), "error": bool(result["error"])})
        if progress and (len(rows) % 100 == 0 or len(rows) == len(jobs)):
            print(f"  {name}: {len(rows)}/{len(jobs)} maps, {writer.total} frames, "
                  f"{time.perf_counter() - began:.0f} s, errors {len(errors)}", flush=True)

    if workers <= 0:
        for job in jobs:
            take(collect_map(job))
    else:
        saved = {k: os.environ.get(k) for k in THREAD_ENV}
        os.environ.update(THREAD_ENV)            # one compute thread per worker
        try:
            with mp.get_context("spawn").Pool(workers) as pool:
                for result in pool.imap(collect_map, jobs, chunksize=2):
                    take(result)
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    writer.close()
    (out_dir / "maps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"set": name, "maps": len(jobs), "frames_per_map": frames_per_map, "engine_steps": steps,
               "wall_s": round(time.perf_counter() - began, 1), "workers": workers, "errors": errors,
               "looks": {look: sum(r["look"] == look for r in rows) for look in ("reference", "random")},
               "edited_maps": sum(bool(r["edits"]) for r in rows), "tags": list(TAGS), "kinds": list(KINDS),
               **describe(load_set(out_dir, with_rgb=False))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.exitdet.collect", description=__doc__.split("\n")[0])
    ap.add_argument("--sets", default="unseen,heldout,val,train", help="comma-separated sets to collect")
    ap.add_argument("--out", default=str(DATA_DIR), help="directory that holds one sub-directory per set")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--frames", type=int, default=FRAMES_PER_MAP, help="frames per map")
    ap.add_argument("--maps", default="", help="override map counts, e.g. train=2000,val=100")
    args = ap.parse_args(argv)
    sizes = {k: int(v) for k, v in (item.split("=") for item in args.maps.split(",") if item)}
    plans = plan(sizes)
    for name in [s for s in args.sets.split(",") if s]:
        summary = collect_set(name, plans[name], Path(args.out) / name, args.workers, args.frames)
        print(f"{name}: {summary['frames']} frames from {summary['maps']} maps in {summary['wall_s']:.0f} s; "
              f"visible {summary['visible']} ({summary['visible_share']:.1%}; by band {summary['visible_0_200']}/"
              f"{summary['visible_200_400']}/{summary['visible_400_640']}; drawn by the renderer "
              f"{summary['rendered_0_200']}/{summary['rendered_200_400']}/{summary['rendered_400_640']}), "
              f"ambiguous {summary['ambiguous']}, errors {len(summary['errors'])}")
        print("  negatives by tag: " + ", ".join(f"{t} {summary['negative_' + t]}" for t in TAGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
