"""Spatial memory: a start-frame map of what the rays, the entity list and the exit detector have shown.

The map is an observation encoding. It is written by fixed geometry from the agent's own pose
estimate and observations (nothing here is learned and nothing here chooses an action); the
networks read it through `crop()`.

Stored state, start pose at the centre, index [x cell, y cell]:

* `occ`    (256 x 256 cells of 32 units) occupancy evidence. Each step, a cell a ray passes through
           loses 1 and a cell in which a ray ends on a wall gains `WALL_WEIGHT`. Negative = free,
           positive = wall, 0 = never observed. Votes instead of bits, so a wall point that lands
           in the wrong cell once (the pose is an estimate) is outvoted by the rays that keep
           passing through it.
* `flags`  (256 x 256) bits: closed door last seen here, locked door last seen here, pickup seen
           here, key seen here, the agent has stood here.
* `stamp`  (3 x 129 x 129 cells of 64 units) decision step of the last visit, the last enemy
           sighting and the last USE press. The decaying channels (recency, enemy, used) are
           computed from these in `crop()`, so nothing has to be decayed every step.
* `exit`   (129 x 129 cells of 64 units) belief that the exit pad is here, raised by every sighting
           reported by the exit detector.

The last two are written at single points and only ever read at the crop's 64-unit resolution, so
they are stored at that resolution; `full()` returns all eleven channels at 32 units.

Wall rule: a ray that ends on a wall types the cell just behind the wall face. "Behind" is taken
along the wall's normal where neighbouring rays show the wall's direction (a step along a grazing
ray would land in the next cell along the wall), and along the ray otherwise. For the same reason
free space stops a fixed distance short of the wall, measured at right angles to it.

Door rule: a ray that ends on a door sets the door bit of the cell just behind the door face. A
ray that passes through a cell clears that cell's door bits, because rays only pass through a door
that is open. Within one update the pass-through writes come first and the hits second, so a door
that is closed now always ends the update marked. The bit therefore reads "closed door last seen
here": it disappears when the door is seen open and comes back when it is seen closed again.
Doors and walls vote against each other in a cell, so a hit that lands one cell to the side
cannot turn a wall into a door.

Bounds: writes outside the arrays are dropped, `crop()` pads with zeros, and a pose that is far
away or not finite never raises.

Purpose: remember walls, free space, doors, pickups, keys, enemies, the exit and the agent's own trail for a whole episode.
Shapes: update(pose (x, y, theta), obs, prev_action (9,), exit_obs (3,) or None, step) -> None; crop(pose) -> float32 (11, 64, 64), index [channel, x cell, y cell], agent in cell (32, 32).
Why: an allocentric grid written by fixed geometry is exact where odometry is exact, costs about 0.1 ms per step, and gives the planner a picture that does not rotate when the agent turns.
"""

from __future__ import annotations

import math

import numpy as np

from . import spec
from .odometry import COLLINEAR_TOL, COS, HEADINGS, RAY_OFFSETS, SIN, heading_index, ray_ranges, wall_normals

# ---- write rules -----------------------------------------------------------------------------------
SAMPLE_STEP = 16.0            # units between free-space samples along a ray (half a stored cell)
FREE_MARGIN = 4.0             # free space stops this far in front of a wall (at right angles to it)
HIT_PAST = 6.0                # the typed cell is the one this far behind the wall face (along its normal)
HIT_PAST_RAY = 2.0            # ... or this far along the ray where the wall's direction is not known
WALL_WEIGHT = 3               # evidence added by a wall hit (a pass-through removes 1)
STAND_WEIGHT = 4              # evidence removed from the cell the agent stands in
EXIT_GAIN = 0.5               # exit belief added per sighting, times the detector's confidence
EXIT_MIN_CONFIDENCE = 0.5     # sightings below this confidence are ignored
RECENCY_STEPS = 128.0         # the breadcrumb trail fades to 0 over this many decision steps
ENEMY_STEPS = 64.0            # an enemy sighting fades over this many steps
USED_STEPS = 64.0             # a USE press fades over this many steps (a door has closed again by then)

FLAG_DOOR, FLAG_LOCKED, FLAG_ITEM, FLAG_KEY, FLAG_VISITED = 1, 2, 4, 8, 16
_FLAG_CHANNELS = (spec.CH_DOOR, spec.CH_LOCKED, spec.CH_ITEM, spec.CH_KEY, spec.CH_VISITED)   # bit k -> channel
STAMP_VISIT, STAMP_ENEMY, STAMP_USED = 0, 1, 2
NEVER = -1.0e6                # stamp of a cell where the event has never happened

# ---- geometry ------------------------------------------------------------------------------------------
N = spec.MAP_SIZE                                  # 32-unit cells per side (256)
HALF = N // 2
INV_RES = 1.0 / spec.MAP_RES
CROP_CELL = spec.MAP_RES * spec.CROP_MULT          # 64 units
WINDOW = spec.CROP_SIZE * spec.CROP_MULT           # 32-unit cells per side of the crop window (128)
CROP_CENTER = spec.CROP_SIZE // 2                  # crop index of the agent's cell on both axes (32)
NC = N // spec.CROP_MULT + 1                       # 64-unit cells per side (129): cell k is index k + NC // 2
REACH_CELLS = int(spec.RAY_RANGE * INV_RES) + 1    # no ray sample is further than this many cells away

_RADII = np.arange(SAMPLE_STEP / 2, spec.RAY_RANGE, SAMPLE_STEP)                         # (64,) 8, 24, ..., 1016
# Sample offsets in cells for each of the 1024 ray directions: (1024, 64) each, 0.5 MB together.
_OX = (COS[:, None] * _RADII[None, :] * INV_RES).astype(np.float32)
_OY = (SIN[:, None] * _RADII[None, :] * INV_RES).astype(np.float32)
_COS_CELL = COS * INV_RES                          # (1024,) ray direction scaled to cells per unit
_SIN_CELL = SIN * INV_RES
_DECAY = (1.0 / np.array([RECENCY_STEPS, ENEMY_STEPS, USED_STEPS], np.float32))[:, None, None]
_NOT_DOOR = np.uint8(~(FLAG_DOOR | FLAG_LOCKED) & 0xFF)
_ENTITY_FLAG = np.zeros(spec.NUM_ENTITY_TYPES, np.uint8)
_ENTITY_FLAG[[spec.ENT_HEALTH, spec.ENT_ARMOR, spec.ENT_AMMO, spec.ENT_WEAPON]] = FLAG_ITEM
_ENTITY_FLAG[spec.ENT_KEY] = FLAG_KEY
_FINE_TO_COARSE = (np.arange(N) + 1) // spec.CROP_MULT        # 32-unit cell index -> 64-unit cell index


def crop_origin(pose) -> tuple[int, int]:
    """Start-frame cell (64-unit cells, cell k spans [64k - 32, 64k + 32)) shown at crop index (0, 0)."""
    return (int(math.floor((pose[0] + CROP_CELL / 2) / CROP_CELL)) - CROP_CENTER,
            int(math.floor((pose[1] + CROP_CELL / 2) / CROP_CELL)) - CROP_CENTER)


class MemoryMap:
    """Episode memory in the start frame. `reset()` before every map, `update()` once per observation."""

    def __init__(self):
        self.occ = np.zeros((N, N), np.int32)
        self.flags = np.zeros((N, N), np.uint8)
        self.stamp = np.full((3, NC, NC), NEVER, np.float32)
        self.exit = np.zeros((NC, NC), np.float32)
        self._occ_flat = self.occ.reshape(-1)            # views used by the scatter writes
        self._flags_flat = self.flags.reshape(-1)
        self._doors = False
        self._step = 0

    def reset(self) -> None:
        self.occ.fill(0)
        self.flags.fill(0)
        self.stamp.fill(NEVER)
        self.exit.fill(0.0)
        self._doors = False          # no door seen yet: lets update() skip the door bookkeeping
        self._step = 0

    # ------------------------------------------------------------------------------------------
    def update(self, pose, obs, prev_action=None, exit_obs=None, step=0) -> None:
        """Write one observation taken at `pose` = (x, y, theta) in the start frame.

        prev_action: the buttons that produced `obs` (USE marks the cell straight ahead).
        exit_obs: None or (confidence in [0, 1], forward / 1024, left / 1024) from the exit detector.
        step: decision step of this observation (the time base of the decaying channels).
        """
        self._step = step = int(step)
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)):
            return
        cx, cy = x * INV_RES + HALF, y * INV_RES + HALF          # position in 32-unit cells, array coordinates
        if not (-REACH_CELLS - 8 < cx < N + REACH_CELLS + 8 and -REACH_CELLS - 8 < cy < N + REACH_CELLS + 8):
            return                                               # nothing in view can land in the arrays
        inside = REACH_CELLS < cx < N - REACH_CELLS - 1 and REACH_CELLS < cy < N - REACH_CELLS - 1
        occ, flags = self._occ_flat, self._flags_flat
        rays = obs["rays"]
        hit = np.asarray(rays["hit_type"])
        r = ray_ranges(rays["distance"])
        rows = (RAY_OFFSETS + heading_index(theta)) % HEADINGS   # direction of every ray, heading units

        # ---- geometry of the hits: wall points and, where known, the direction of the wall ----------
        static = (hit >= spec.HIT_WALL) & (hit <= spec.HIT_LOCKED)
        ux, uy = _COS_CELL[rows], _SIN_CELL[rows]                # ray directions, cells per unit
        px, py = ux * r, uy * r                                  # hit points relative to the agent, cells
        nx, ny, known = wall_normals(px, py, static, COLLINEAR_TOL * INV_RES)
        facing = (nx * ux + ny * uy) * spec.MAP_RES              # cosine between ray and wall normal

        # ---- free space: samples along every ray, stopping FREE_MARGIN short of the wall ------------
        # The margin is measured at right angles to the wall, so a ray that runs along a wall stops
        # early instead of brushing the wall's own cells.
        short = np.where(known, FREE_MARGIN / np.maximum(np.abs(facing), 1.0 / 16.0), FREE_MARGIN)
        limit = np.where(r >= spec.RAY_RANGE - 0.5, 2.0 * spec.RAY_RANGE, r - short)     # nothing hit: whole range
        keep = _RADII < limit[:, None]
        if inside:                                               # every sample is inside: truncation is floor
            idx = ((_OX[rows] + cx).astype(np.intp) * N + (_OY[rows] + cy).astype(np.intp))[keep]
        else:
            ix = np.floor(_OX[rows] + cx).astype(np.intp)
            iy = np.floor(_OY[rows] + cy).astype(np.intp)
            keep &= (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
            idx = (ix * N + iy)[keep]
        v = occ[idx]
        v -= 1
        v[v == 0] = -1                                           # never back to 0: observed stays observed
        occ[idx] = v
        if self._doors:
            flags[idx] &= _NOT_DOOR                              # a ray passed through: the door is open

        # ---- hits: type the cell just behind the wall face ----------------------------------------
        typed = np.flatnonzero(static)
        if len(typed):
            away = np.where(facing < 0.0, -HIT_PAST * INV_RES, HIT_PAST * INV_RES)
            sx = np.where(known, nx * away, ux * HIT_PAST_RAY)
            sy = np.where(known, ny * away, uy * HIT_PAST_RAY)
            ix = np.floor(px[typed] + sx[typed] + cx).astype(np.intp)
            iy = np.floor(py[typed] + sy[typed] + cy).astype(np.intp)
            kind = hit[typed]
            if not inside:
                ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
                ix, iy, kind = ix[ok], iy[ok], kind[ok]
            idx = ix * N + iy
            is_wall = kind == spec.HIT_WALL
            wall = idx[is_wall]
            v = occ[wall]
            v += WALL_WEIGHT
            v[v == 0] = WALL_WEIGHT
            occ[wall] = v
            if self._doors or len(wall) < len(idx):
                # Doors and walls vote against each other, so a hit that lands one cell to the side
                # (the pose is an estimate) cannot turn a wall into a door or a door into a wall.
                self._doors = True
                flags[wall[v > 0]] &= _NOT_DOOR
                door = idx[~is_wall]
                v = occ[door]
                v = np.where(v > 0, np.maximum(v - WALL_WEIGHT, 0), v)
                occ[door] = v
                agreed = v <= 0
                bit = np.where(kind[~is_wall] == spec.HIT_DOOR, FLAG_DOOR, FLAG_LOCKED).astype(np.uint8)
                flags[door[agreed]] |= bit[agreed]

        c, s = math.cos(theta), math.sin(theta)
        # ---- the agent's own cell ---------------------------------------------------------------------
        ia, ja = int(math.floor(cx)), int(math.floor(cy))
        if 0 <= ia < N and 0 <= ja < N:
            self.occ[ia, ja] = min(int(self.occ[ia, ja]), 0) - STAND_WEIGHT
            # Pickups within reach of the body are taken (or will be marked again when seen again).
            block = self.flags[max(ia - 1, 0):ia + 2, max(ja - 1, 0):ja + 2]
            block &= ~(FLAG_ITEM | FLAG_KEY) & 0xFF
            self.flags[ia, ja] |= FLAG_VISITED
            ic, jc = (ia + 1) // 2, (ja + 1) // 2
            self.stamp[STAMP_VISIT, ic, jc] = step
            self.exit[ic, jc] = 0.0                              # standing on the pad would have ended the episode

        # ---- entities ---------------------------------------------------------------------------------
        ents = obs["visible_entities"]
        seen = np.flatnonzero(np.asarray(ents["mask"]))
        if len(seen):
            feats = np.nan_to_num(np.asarray(ents["features"], np.float64)[seen], nan=0.0, posinf=0.0, neginf=0.0)
            etype = np.clip(np.asarray(ents["entity_type"])[seen], 0, spec.NUM_ENTITY_TYPES - 1)
            fwd = feats[:, 0] * (feats[:, 2] * spec.RAY_RANGE * INV_RES)       # cells forward
            left = feats[:, 0] * (feats[:, 1] * spec.RAY_RANGE * INV_RES)      # cells to the left
            ex, ey = cx + c * fwd - s * left, cy + s * fwd + c * left
            ok = (ex >= 0) & (ex < N) & (ey >= 0) & (ey < N)                   # also drops NaN
            ix, iy, etype = ex[ok].astype(np.intp), ey[ok].astype(np.intp), etype[ok]
            enemy = (etype >= spec.ENT_MELEE) & (etype <= spec.ENT_PROJ_ENEMY)
            self.stamp[STAMP_ENEMY, (ix[enemy] + 1) // 2, (iy[enemy] + 1) // 2] = step
            bits = _ENTITY_FLAG[etype]
            pick = bits != 0
            self.flags[ix[pick], iy[pick]] |= bits[pick]

        # ---- exit sighting ----------------------------------------------------------------------------
        if exit_obs is not None:
            conf = float(exit_obs[0])
            if conf >= EXIT_MIN_CONFIDENCE:
                fwd, left = float(exit_obs[1]) * spec.RAY_RANGE * INV_RES, float(exit_obs[2]) * spec.RAY_RANGE * INV_RES
                ex, ey = cx + c * fwd - s * left, cy + s * fwd + c * left
                if 0 <= ex < N and 0 <= ey < N:                                # also drops NaN
                    i, j = (int(ex) + 1) // 2, (int(ey) + 1) // 2
                    self.exit[i, j] = min(1.0, float(self.exit[i, j]) + EXIT_GAIN * min(conf, 1.0))

        # ---- USE: the cell straight ahead, just past whatever is within reach --------------------------
        if prev_action is not None and prev_action[7]:
            d = min(float(r[0]) + HIT_PAST, spec.USE_REACH) * INV_RES
            ex, ey = cx + c * d, cy + s * d
            if 0 <= ex < N and 0 <= ey < N:
                self.stamp[STAMP_USED, (int(ex) + 1) // 2, (int(ey) + 1) // 2] = step

    # ------------------------------------------------------------------------------------------
    def crop(self, pose) -> np.ndarray:
        """Network view: float32 (MAP_CHANNELS, 64, 64), 64 units per cell, axes of the start frame.

        Index [channel, i, j]: i grows with x (forward at the start), j with y (left at the
        start). The agent's cell is (CROP_CENTER, CROP_CENTER) = (32, 32); crop cell (i, j) is the
        start-frame cell `crop_origin(pose) + (i, j)`. Each crop cell pools 2 x 2 cells of `occ`
        and `flags` (free, wall and every flag: on if on in any of the four).
        """
        out = np.zeros((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.float32)
        x, y = float(pose[0]), float(pose[1])
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 1e7 or abs(y) > 1e7:
            return out
        oi, oj = crop_origin((x, y))
        i0, j0 = spec.CROP_MULT * oi + HALF - 1, spec.CROP_MULT * oj + HALF - 1      # window start, 32-unit cells
        ci, cj = oi + NC // 2, oj + NC // 2                                          # window start, 64-unit cells
        if i0 >= N or j0 >= N or i0 + WINDOW <= 0 or j0 + WINDOW <= 0:
            return out
        if 0 <= i0 and i0 + WINDOW <= N and 0 <= j0 and j0 + WINDOW <= N:
            occ = np.ascontiguousarray(self.occ[i0:i0 + WINDOW, j0:j0 + WINDOW])
            flags = np.ascontiguousarray(self.flags[i0:i0 + WINDOW, j0:j0 + WINDOW])
            stamp = self.stamp[:, ci:ci + spec.CROP_SIZE, cj:cj + spec.CROP_SIZE]
            exit_ = self.exit[ci:ci + spec.CROP_SIZE, cj:cj + spec.CROP_SIZE]
        else:
            occ, flags = _window(self.occ, i0, j0, WINDOW, 0), _window(self.flags, i0, j0, WINDOW, 0)
            stamp = _window(self.stamp, ci, cj, spec.CROP_SIZE, NEVER)
            exit_ = _window(self.exit, ci, cj, spec.CROP_SIZE, 0.0)
        np.less(_pool(occ, np.minimum), 0, out=out[spec.CH_FREE])
        np.greater(_pool(occ, np.maximum), 0, out=out[spec.CH_WALL])
        bits = _pool(flags, np.bitwise_or)
        for k, ch in enumerate(_FLAG_CHANNELS):
            np.bitwise_and(bits >> k, 1, out=out[ch], casting="unsafe")
        fade = stamp * _DECAY                                    # (3, 64, 64): step of the latest event, scaled
        fade += 1.0 - self._step * _DECAY                        # 1 at the event, 0 after its fade time
        np.clip(fade, 0.0, 1.0, out=fade)
        out[spec.CH_RECENCY], out[spec.CH_ENEMY], out[spec.CH_USED] = fade[STAMP_VISIT], fade[STAMP_ENEMY], fade[STAMP_USED]
        out[spec.CH_EXIT] = exit_
        return out

    def full(self) -> np.ndarray:
        """The whole map as float32 (MAP_CHANNELS, 256, 256) at 32 units per cell (tests, tools, demo)."""
        out = np.zeros((spec.MAP_CHANNELS, N, N), np.float32)
        out[spec.CH_FREE] = self.occ < 0
        out[spec.CH_WALL] = self.occ > 0
        for k, ch in enumerate(_FLAG_CHANNELS):
            out[ch] = (self.flags >> k) & 1
        fade = np.clip(self.stamp * _DECAY + (1.0 - self._step * _DECAY), 0.0, 1.0)
        fine = fade[:, _FINE_TO_COARSE][:, :, _FINE_TO_COARSE]
        out[spec.CH_RECENCY], out[spec.CH_ENEMY], out[spec.CH_USED] = fine[STAMP_VISIT], fine[STAMP_ENEMY], fine[STAMP_USED]
        out[spec.CH_EXIT] = self.exit[_FINE_TO_COARSE][:, _FINE_TO_COARSE]
        return out


def _pool(win, op):
    """Pool 2 x 2 blocks of a contiguous (128, 128) window with `op` -> (64, 64)."""
    half, m = spec.CROP_SIZE, spec.CROP_MULT
    rows = win.reshape(half, m, WINDOW)
    cols = op(rows[:, 0, :], rows[:, 1, :]).reshape(half, half, m)
    return op(cols[:, :, 0], cols[:, :, 1])


def _window(arr, i0, j0, size, fill):
    """A size x size copy of the last two axes of `arr` starting at (i0, j0), padded with `fill`."""
    n = arr.shape[-1]
    out = np.full(arr.shape[:-2] + (size, size), fill, arr.dtype)
    a0, a1 = max(i0, 0), min(i0 + size, n)
    b0, b1 = max(j0, 0), min(j0 + size, n)
    if a0 < a1 and b0 < b1:
        out[..., a0 - i0:a1 - i0, b0 - j0:b1 - j0] = arr[..., a0:a1, b0:b1]
    return out
