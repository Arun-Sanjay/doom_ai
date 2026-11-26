"""Training-time scaffolding. Never imported by the agent in submission/.

Fair navigation teacher: the hand-written explorer whose labels the student network is trained
to imitate on navigation steps. It contains graph search, which is why it lives under training/.

Fairness. `Teacher.label` is a function of what the student network receives at the same step and
of nothing else: the memory-map crop (`MemoryMap.crop`), the pose estimate, the current observation
(rays, entities, vitals), the encoder scalars and the previous action. It never sees the true
position, the map file or any other ground truth. The state it keeps between calls is derived from
those same inputs and from the actions that were actually taken, so it stays valid when the student
drives: the plan made at the last manager step, the turn-key hold counter, which locked doors were
tried with the keys now held, and which floor cells the camera has already looked at.

Decision, once per manager step (every `MANAGER_PERIOD` steps) and whenever the plan breaks. First
match wins:

1. fight        an enemy entity is visible: the label names the enemy's cell and carries no
                controller labels (combat is learned by reinforcement); navigation continues below.
2. go_exit      the exit is on the map and reachable, and the steps left are at most
                LEAVE_FACTOR x the travel estimate + LEAVE_MARGIN.
3. get_key      a key is on the map, reachable and its cell not yet visited.
4. open_locked  a key is held and a locked door is known, reachable and not given up. A locked
                door is given up after LOCK_TRIES presses that did not open it, until the number
                of keys held changes.
5. get_pickup   health or bullets are low and an item is known, or an unvisited item lies within
                a few steps.
6. the nearest of: a frontier cell (explore), a room never entered (explore), a doorway not yet
   walked through, closed or open (open_door) and, while no exit is known, a dead end the camera
   has not looked into or any floor within a few cells that it has not looked at (sweep). The
   current choice is kept until it is done or another is less than half as far. If the way to the
   choice leads through a closed door, that door is the target first (open_door). A sweep target
   is done as soon as the camera has looked at it; the teacher turns to it from wherever it has
   a straight clear line and does not walk into it.
7. sweep        no exit known: the nearest floor cell neither visited nor looked at, then every
                unvisited cell, dead ends first.
8. go_exit      nothing else remains.
9. explore      the least recently visited reachable cell, so the agent keeps moving.

Until the teacher decides to leave, a cell where the exit is believed is kept out of every route.
The exit pad is a walk-over cell that only the camera can find, so while no exit is known the
teacher also looks before it steps: the view leads the body along the route, and a floor cell that
has been neither visited nor looked at is faced first and entered afterwards (LOOK_PATIENCE steps
at most). A dead end is therefore inspected from outside, and the map is not ended by accident
before it has been explored.

Between decisions the teacher follows the stored field of travel steps to the target from
whatever cell it is in (see `steer.py` for the keys). At a closed door on its way it comes to
rest in the cell in front of it, faces it, presses USE once the door is straight ahead within
reach and holds it until the door no longer reads as a door; it never lets USE go down twice
within USE_REPRESS_STEPS, because a second press reverses a rising door. Attack and weapon are
never labelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec

from . import steer
from .grid import CELL, CENTER, EXIT_SEEN, EXIT_SURE, INF, N, OWN, STEPS_PER_CELL, PlanGrid, cell_of

LEAVE_FACTOR = 1.5            # leave for the exit when steps left <= LEAVE_FACTOR * travel + LEAVE_MARGIN
LEAVE_MARGIN = 20.0
DOOR_STEPS = 13.0             # planning cost of a closed door: stop, press, wait for it to rise, start again (measured)
USE_REPRESS_STEPS = 8         # USE may go down again this many steps after it last went down
USE_PRESS_RANGE = 60.0        # the first press waits until the door face is this close (USE reaches 64)
USE_FACING_SINE = math.sin(math.radians(6.0))
LOCK_TRIES = 2                # presses without effect before a locked door is given up
LOCK_VERDICT_STEPS = 7        # a door that will open no longer reads as a door this many steps after the press
LOOK = 5                      # cells of the route the steering looks ahead
LEAD = 3                      # the view aims at the route cell this many cells ahead
PRETURN_RANGE = 100.0         # start facing a door this far from the cell in front of it
ITEM_IN_WAY = 48.0            # a pickup this close straight ahead, while facing a door, is stepped past
LOOK_PATIENCE = 16            # steps spent facing an unseen cell before it is entered anyway
LOOKS = 3                     # steps on which the camera must have looked at a cell before it counts as seen:
                              # the exit observations used in training miss one detection in five
NEAR_UNSEEN_STEPS = 2.0 * STEPS_PER_CELL  # floor this close that the camera has not looked at is looked at before moving on
PROBE_PATIENCE = 12           # steps spent pushing into a doubtful cell before it is taken for a wall
BLOCKED_RANGE = 40.0          # a closed door this close straight ahead is in the way even if the map shows floor
BLOCKED_SINE = math.sin(math.radians(35.0))
BLOCKED_PATIENCE = 24         # steps spent on such a door before it is left alone
NEAR_ITEM_STEPS = 3.0 * STEPS_PER_CELL    # an unvisited item this close is collected on the way
LOW_HEALTH, LOW_BULLETS = 50.0, 15.0
VALUE_CAP = 255.0             # value_field entries are travel steps, capped here (also "no route")
HYSTERESIS = 0.5              # a new exploration target must be less than this share as far as the current one
DEAD_END_WEIGHT = 0.5         # a dead end counts as this much nearer than it is: alcoves and corridor ends are where
                              # exit pads tend to be, and a look into one is quick

VIEW_RAYS = np.array([57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4, 5, 6, 7])      # rays inside the camera's view
VIEW_RANGE = 600.0            # the exit detector's range is 640
VIEW_RADII = np.arange(24.0, VIEW_RANGE, 32.0)
VIEW_CENTRAL = 20.0           # a cell counts as looked at when a view ray passes this close to its centre
GAZE_RANGE = VIEW_RANGE - 60.0          # a sweep target is looked at from where it stands only within this distance
_VIEW_COS = np.cos(VIEW_RAYS * (2.0 * np.pi / spec.NUM_RAYS))
_VIEW_SIN = np.sin(VIEW_RAYS * (2.0 * np.pi / spec.NUM_RAYS))
HALF = 2 * N                  # the teacher's own start-frame grids hold cells -HALF .. HALF: cell k is index k + HALF
ABS = 2 * HALF + 1
SOLID_HITS = (spec.HIT_WALL, spec.HIT_DOOR, spec.HIT_LOCKED)
THING_SLACK = 40.0            # a view ray that ends on a pickup or a monster still shows the floor this far behind it
_I_USE_AGE = spec.SCALAR_INDEX["use_age"]
_AGE_SCALE = 16.0             # the encoder's use_age is steps / 16, clipped to 1

EXPLORE, SWEEP, GET_KEY, OPEN_DOOR, OPEN_LOCKED, GO_EXIT, FIGHT, GET_PICKUP = (
    spec.SUBGOAL_INDEX[n] for n in ("explore", "sweep", "get_key", "open_door", "open_locked", "go_exit", "fight", "get_pickup"))
K_FRONTIER, K_DOORWAY, K_ROOM, K_DEAD_END, K_UNSEEN = range(5)      # kinds of exploration target
KIND_SUBGOAL = (EXPLORE, OPEN_DOOR, EXPLORE, SWEEP, SWEEP)


@dataclass
class TeacherLabel:
    """What the teacher would do at one step.

    subgoal      index into spec.SUBGOALS
    target       crop cell index i * CROP_SIZE + j of the target (the agent's own cell if none)
    heads        (6,) int64 action heads the teacher would take (attack and weapon always 0)
    head_mask    (6,) bool; False = no label for this head on this step
    value_field  (64, 64) float32 travel steps from every crop cell to the target through
                 known-passable cells, capped at VALUE_CAP, in the frame of the current crop
    field_valid  False when there is no target (then value_field is all VALUE_CAP)
    """

    subgoal: int
    target: int
    heads: np.ndarray
    head_mask: np.ndarray
    value_field: np.ndarray
    field_valid: bool


class _Plan:
    """The outcome of one decision: sub-goal, target and the field that leads to it."""

    __slots__ = ("subgoal", "target", "target_cell", "ox", "oy", "link", "plain", "value", "step", "valid", "travel",
                 "probe", "enter", "arrived", "_shift", "_shifted")

    def __init__(self, subgoal, target, ox, oy, link, plain, value, step, valid, travel, probe=False, enter=False):
        self.probe = probe                                          # the target is a doubtful cell to be tried
        self.enter = enter or probe                                 # a sweep target that has to be stepped on, not looked at
        self.arrived = 0                                            # steps spent in the target cell (go_exit only)
        self.subgoal, self.target = subgoal, target                 # target: flat index in the frame of (ox, oy)
        self.target_cell = (target // N + ox, target % N + oy)      # start-frame cell
        self.ox, self.oy = ox, oy
        self.link, self.plain, self.value = link, plain, value
        self.step, self.valid, self.travel = step, valid, travel
        self._shift, self._shifted = (0, 0), value

    def field_in(self, ox: int, oy: int) -> np.ndarray:
        """The value field re-indexed for a crop whose origin is (ox, oy)."""
        di, dj = ox - self.ox, oy - self.oy
        if (di, dj) != self._shift:
            out = np.full((N, N), VALUE_CAP, np.float32)
            i0, i1 = max(0, -di), min(N, N - di)
            j0, j1 = max(0, -dj), min(N, N - dj)
            if i0 < i1 and j0 < j1:
                out[i0:i1, j0:j1] = self.value[i0 + di:i1 + di, j0 + dj:j1 + dj]
            self._shift, self._shifted = (di, dj), out
        return self._shifted


_MASK_NAV = np.array([True, True, True, False, True, False])
_MASK_NONE = np.zeros(6, bool)
_NO_FIELD = np.full((N, N), VALUE_CAP, np.float32)
_NO_FIELD.setflags(write=False)


class Teacher:
    """See the module docstring. `reset()` before every map, `label()` once per observation."""

    def __init__(self, leave_factor: float = LEAVE_FACTOR, leave_margin: float = LEAVE_MARGIN,
                 lock_tries: int = LOCK_TRIES, door_steps: float = DOOR_STEPS, pickups: bool = True,
                 pad_floor: float = EXIT_SURE, look_first: bool = True, looks: int = LOOKS):
        self.leave_factor, self.leave_margin = float(leave_factor), float(leave_margin)
        self.lock_tries, self.door_steps, self.pickups = int(lock_tries), float(door_steps), bool(pickups)
        self.pad_floor, self.look_first, self.looks = float(pad_floor), bool(look_first), int(looks)
        self._look_count = np.zeros((ABS, ABS), np.uint16)
        self._looked = np.zeros((ABS, ABS), bool)
        self._doorways = np.zeros((ABS, ABS), bool)
        self.reset()

    def reset(self) -> None:
        self.plan: _Plan | None = None
        self.failed: dict = {}          # locked-door cell -> keys held when it was given up
        self.presses: dict = {}         # locked-door cell -> (keys held, presses without effect)
        self._facing = None             # locked-door cell straight ahead within reach at the last observation
        self._watch = None              # locked-door cell whose last press is still being judged
        self._held = 0                  # tics a turn key has been down without a break
        self._keys = 0
        self._choice = None             # (kind, start-frame cell) of the exploration target in force
        self._look_count.fill(0)        # per start-frame cell: steps on which the camera looked at it ...
        self._looked.fill(False)        # ... and whether that makes `looks` or more
        self._doorways.fill(False)      # start-frame cells where a door has been seen, open or not
        self._peek = None               # [cell, steps]: the unseen cell being faced before it is entered
        self._waived: set = set()       # unseen cells that are entered anyway (patience ran out)
        self._item_kind: dict = {}      # cell -> entity type of the pickup last seen there
        self._key_cells: set = set()    # cells where a key was seen and has not been walked over since
        self._probe = None              # [cell, steps]: the doubtful cell being tried
        self._probed: set = set()       # doubtful cells that were tried (entered, or found solid)
        self._probe_target = -1         # set by _choose when its target is a doubtful cell
        self._enter_target = -1         # ... or an unvisited cell that is to be stepped on
        self._enter_cells: set = set()  # sweep targets that could not be seen from outside: step on them
        self._blocked = None            # [cell, steps]: an unmapped closed door being opened
        self._left_alone: set = set()   # such doors that did not open
        self.plan_route = (0, 0, 0)     # cells, closed doors and turns on the route of the last decision
        self.exit_known = False
        self.errors = 0
        self.last_error = ""
        self.searches = 0
        self.replans = 0
        self.info: dict = {}            # what the last label was based on (debug views and tests)

    # ------------------------------------------------------------------------------------------
    def label(self, obs, crop, pose, scal, step, budget_steps, prev_action) -> TeacherLabel:
        """The teacher's choice for the observation `obs` taken at decision step `step`.

        crop: `MemoryMap.crop(pose)` after the map was updated with `obs`; pose: the odometry
        estimate (x, y, theta); scal: the encoder's scalar vector for this step; budget_steps: the
        assumed time budget in decision steps; prev_action: the (9,) buttons that led to `obs`
        (None at step 0). Never raises: on an internal error it counts it in `errors`, drops its
        plan and returns a label without controller labels.
        """
        try:
            return self._label(obs, crop, pose, scal, int(step), float(budget_steps), prev_action)
        except Exception as e:  # noqa: BLE001 - a label must always come back; tests assert errors == 0
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            self.plan = None
            self.info = {"mode": "error"}
            return TeacherLabel(EXPLORE, OWN, np.zeros(6, np.int64), _MASK_NONE.copy(), _NO_FIELD, False)

    def _label(self, obs, crop, pose, scal, step, budget, prev_action) -> TeacherLabel:
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)) or abs(x) > 1e6 or abs(y) > 1e6:
            raise ValueError("pose estimate is not usable")
        cx, cy = cell_of(x, y)
        ox, oy = cx - CENTER, cy - CENTER
        rays = obs["rays"]
        ray_dist, ray_hit = rays["distance"], rays["hit_type"]
        dist0, hit0 = float(ray_dist[0]) * spec.RAY_RANGE, int(ray_hit[0])
        vitals = obs["vitals"]
        keys = int(np.count_nonzero(np.asarray(vitals[8:14]) > 0.5))
        age = float(scal[_I_USE_AGE]) * _AGE_SCALE
        prev_use = prev_action is not None and bool(prev_action[7])
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # ---- state kept from the inputs and the actions actually taken ------------------------
        self._held = self._held + spec.FRAME_SKIP if prev_action is not None and (prev_action[4] or prev_action[5]) else 0
        self._mark_looked(ray_dist, ray_hit, x, y, cos_t, sin_t)
        changed = self._judge_locks(crop, ox, oy, keys, age, prev_use)
        gate_ahead = (hit0 == spec.HIT_LOCKED or hit0 == spec.HIT_DOOR) and dist0 <= spec.USE_REACH + 2.0
        self._facing = None
        if gate_ahead and hit0 == spec.HIT_LOCKED:
            reach = dist0 + 8.0
            self._facing = cell_of(x + reach * cos_t, y + reach * sin_t)
        if keys != self._keys:
            self._keys, changed = keys, True
            self._key_cells = {c for c in self._key_cells if abs(c[0] - cx) > 1 or abs(c[1] - cy) > 1}   # the one just taken
        self._key_cells.discard((cx, cy))
        enemy = self._scan_entities(obs, x, y, cos_t, sin_t, ox, oy)

        # ---- decide ---------------------------------------------------------------------------
        plan = self.plan
        if plan is None or changed or step % spec.MANAGER_PERIOD == 0 or self._stale(plan, crop, cx, cy, ox, oy):
            if plan is not None and step % spec.MANAGER_PERIOD != 0:
                self.replans += 1
            plan = self._decide(crop, vitals, ox, oy, step, budget, keys)

        # ---- act ------------------------------------------------------------------------------
        vf, vr = float(vitals[14]), float(vitals[15])
        if not (abs(vf) < 100.0 and abs(vr) < 100.0):
            vf = vr = 0.0
        vx, vy = vf * cos_t + vr * sin_t, vf * sin_t - vr * cos_t
        move, strafe, turn, use = self._steer(plan, crop, x, y, cos_t, sin_t, vx, vy, cx, cy, ox, oy,
                                              step, age, prev_use, gate_ahead, dist0, hit0)
        heads = np.array([move, strafe, turn, 0, use, 0], np.int64)

        # ---- label ----------------------------------------------------------------------------
        ti = min(max(plan.target_cell[0] - ox, 0), N - 1)
        tj = min(max(plan.target_cell[1] - oy, 0), N - 1)
        self.info.update(step=step, subgoal=spec.SUBGOALS[plan.subgoal], target=plan.target_cell, cell=(cx, cy),
                         heads=heads.tolist(), fight=enemy is not None, keys=keys)
        if enemy is not None:
            return TeacherLabel(FIGHT, enemy, heads, _MASK_NONE.copy(), _NO_FIELD, False)
        return TeacherLabel(plan.subgoal, ti * N + tj, heads, _MASK_NAV.copy(), plan.field_in(ox, oy), plan.valid)

    # ------------------------------------------------------------------------- bookkeeping
    def _mark_looked(self, ray_dist, ray_hit, x, y, cos_t, sin_t) -> None:
        """Mark the cells the camera can see now: on a view ray, before its hit, near the cell centre.

        A cell counts as looked at once this has happened on `looks` different steps, so a missed
        detection or two do not hide an exit pad.
        """
        if not (abs(x) < (HALF - 12) * CELL and abs(y) < (HALF - 12) * CELL):
            return
        reach = np.asarray(ray_dist, np.float64)[VIEW_RAYS] * spec.RAY_RANGE
        solid = np.isin(np.asarray(ray_hit)[VIEW_RAYS], SOLID_HITS)
        reach = np.minimum(np.where(solid, reach - 8.0, reach + THING_SLACK), VIEW_RANGE)
        dx = cos_t * _VIEW_COS - sin_t * _VIEW_SIN
        dy = sin_t * _VIEW_COS + cos_t * _VIEW_SIN
        px = x + dx[:, None] * VIEW_RADII[None, :]
        py = y + dy[:, None] * VIEW_RADII[None, :]
        ci = np.floor((px + CELL / 2) / CELL)
        cj = np.floor((py + CELL / 2) / CELL)
        ok = ((VIEW_RADII[None, :] < reach[:, None]) & (np.abs(px - ci * CELL) <= VIEW_CENTRAL)
              & (np.abs(py - cj * CELL) <= VIEW_CENTRAL))
        flat = ((ci[ok] + HALF) * ABS + (cj[ok] + HALF)).astype(np.intp)
        count = self._look_count.reshape(-1)
        count[flat] += 1                                       # (a cell sampled twice in one step still counts once)
        self._looked.reshape(-1)[flat[count[flat] >= self.looks]] = True

    def _window(self, grid: np.ndarray, ox: int, oy: int):
        """The part of one of the teacher's start-frame grids that a crop with origin (ox, oy) shows
        (a view, index [i, j] like the crop), or None when the crop reaches outside the grid."""
        i0, j0 = ox + HALF, oy + HALF
        if 0 <= i0 and i0 + N <= ABS and 0 <= j0 and j0 + N <= ABS:
            return grid[i0:i0 + N, j0:j0 + N]
        return None

    def _judge_locks(self, crop, ox, oy, keys, age, prev_use) -> bool:
        """Follow USE presses on locked doors; True when a door has just been given up."""
        if prev_use and age < 0.5 and self._facing is not None:
            self._watch = self._facing            # USE went down last step, facing this locked door
        cell = self._watch
        if cell is None:
            return False
        i, j = cell[0] - ox, cell[1] - oy
        still_locked = 0 <= i < N and 0 <= j < N and crop[spec.CH_LOCKED, i, j] > 0.5
        if not still_locked:                       # it opened (or was never a locked door)
            self._watch = None
            self.presses.pop(cell, None)
            return False
        if age < LOCK_VERDICT_STEPS:
            return False
        self._watch = None
        held, count = self.presses.get(cell, (keys, 0))
        count = count + 1 if held == keys else 1
        self.presses[cell] = (keys, count)
        if count >= self.lock_tries:
            self.failed[cell] = keys
            return True
        return False

    def _locked_ok(self, crop, ox, oy, keys):
        """Which locked-door cells may be planned through: none without a key, else all not given up."""
        if keys <= 0:
            return None
        ok = np.ones((N, N), bool)
        for cell, held in self.failed.items():
            i, j = cell[0] - ox, cell[1] - oy
            if held == keys and 0 <= i < N and 0 <= j < N:
                ok[i, j] = False
        return ok

    def _scan_entities(self, obs, x, y, cos_t, sin_t, ox, oy):
        """Remember what kind of pickup lies in which cell; returns the crop index of the nearest
        visible enemy, or None."""
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]) != 0
        if not mask.any():
            return None
        kind = np.asarray(ents["entity_type"])
        feats = np.asarray(ents["features"], np.float64)
        enemy, enemy_d = None, INF
        for k in np.flatnonzero(mask).tolist():
            t = int(kind[k])
            is_enemy = spec.ENT_MELEE <= t <= spec.ENT_PROJ_ENEMY
            if not (is_enemy or spec.ENT_HEALTH <= t <= spec.ENT_KEY):
                continue
            d = float(feats[k, 0]) * spec.RAY_RANGE
            fwd, left = d * float(feats[k, 2]), d * float(feats[k, 1])
            if not (math.isfinite(fwd) and math.isfinite(left)) or d > 8.0 * spec.RAY_RANGE:
                continue
            cell = cell_of(x + cos_t * fwd - sin_t * left, y + sin_t * fwd + cos_t * left)
            if is_enemy:
                if d < enemy_d:
                    enemy, enemy_d = cell, d
            elif t == spec.ENT_KEY:
                self._key_cells.add(cell)
            else:
                self._item_kind[cell] = t
        if enemy is None:
            return None
        return min(max(enemy[0] - ox, 0), N - 1) * N + min(max(enemy[1] - oy, 0), N - 1)

    # ---------------------------------------------------------------------------- deciding
    def _stale(self, plan: _Plan, crop, cx, cy, ox, oy) -> bool:
        """Does the plan need to be remade before the next manager step?"""
        if (cx, cy) == plan.target_cell:
            if plan.probe:
                self._probed.add(plan.target_cell)
            if plan.subgoal == GO_EXIT:                        # on the pad: the level ends as the body crosses its edge;
                plan.arrived += 1                              # if it does not, this was not the exit
                return plan.arrived > 3
            return plan.valid                                  # arrived
        q = (cx - plan.ox) * N + (cy - plan.oy)
        if not (0 <= cx - plan.ox < N and 0 <= cy - plan.oy < N) or plan.link[q] < 0:
            return True                                        # standing off the field
        sub = plan.subgoal
        ti, tj = plan.target_cell[0] - ox, plan.target_cell[1] - oy
        if 0 <= ti < N and 0 <= tj < N:
            if sub == GET_KEY and crop[spec.CH_KEY, ti, tj] < 0.5 and plan.target_cell not in self._key_cells:
                return True
            if sub == GET_PICKUP and crop[spec.CH_ITEM, ti, tj] < 0.5:
                return True
            if sub == GO_EXIT:
                if crop[spec.CH_EXIT, ti, tj] < 0.25:
                    return True
            elif crop[spec.CH_EXIT, ti, tj] >= self.pad_floor:
                return True                                    # the target turned out to be the exit pad
            if sub == SWEEP and not plan.enter and abs(plan.target_cell[0]) < HALF and abs(plan.target_cell[1]) < HALF \
                    and self._looked[plan.target_cell[0] + HALF, plan.target_cell[1] + HALF]:
                return True                                    # looked at: nothing there
        if plan.probe and plan.target_cell in self._probed:
            return True
        di, dj = ox - plan.ox, oy - plan.oy
        c = q
        for _ in range(LOOK):                                  # a wall or the exit pad has appeared on the route
            c = plan.link[c]
            if c < 0:
                break
            i, j = c // N - di, c % N - dj
            if crop[spec.CH_WALL, i, j] > 0.5 or (crop[spec.CH_EXIT, i, j] >= self.pad_floor and c != plan.target):
                return True
        return False

    def _decide(self, crop, vitals, ox, oy, step, budget, keys) -> _Plan:
        grid = PlanGrid(crop, self._locked_ok(crop, ox, oy, keys), self.door_steps, self.pad_floor)
        doorways = self._window(self._doorways, ox, oy)
        if doorways is not None:
            doorways |= grid.door | grid.locked
        dist, back = grid.from_agent()
        self.searches += 1
        self._probe_target = self._enter_target = -1
        subgoal, target = self._choose(grid, dist, back, vitals, ox, oy, step, budget, keys)
        valid = target != OWN
        if valid:
            field, link = grid.to_target(target)
            self.searches += 1
            value = np.minimum(np.array(field, np.float32).reshape(N, N), VALUE_CAP)
        else:
            link, value = [-1] * (N * N), _NO_FIELD
        self.plan = _Plan(subgoal, target, ox, oy, link, grid.plain_l, value, step, valid, dist[target],
                          probe=target == self._probe_target, enter=target == self._enter_target)
        cells = gates = turns = 0                              # shape of the route, for the logs
        c, last = OWN, 0
        while valid and link[c] >= 0 and cells < 4 * N:
            n = link[c]
            cells += 1
            gates += grid.extra_l[n] > 0.0
            turns += last != 0 and n - c != last
            c, last = n, n - c
        self.plan_route = (cells, gates, turns)
        self.info = {"decided": step, "travel": dist[target]}
        return self.plan

    def _choose(self, g: PlanGrid, dist, back, vitals, ox, oy, step, budget, keys) -> tuple[int, int]:
        """(sub-goal, flat target index) by the rules of the module docstring."""
        def nearest(mask):
            best, best_d = -1, INF
            for k in np.flatnonzero(mask).tolist():
                if dist[k] < best_d and k != OWN:
                    best, best_d = k, dist[k]
            return best, best_d

        def first_gate(t):
            """The first closed door on the route from the agent to t (-1 if none)."""
            found, c = -1, t
            while c >= 0 and c != OWN:
                if extra[c] > 0.0:
                    found = c
                c = back[c]
            return found

        extra = g.extra_l
        # ---- exit: leave now? ------------------------------------------------------------------
        # A pad seen twice in the same cell is the exit; one seen once may be a false detection and
        # is only headed for when there is nothing better.
        exit_cell, exit_d = nearest(g.pad & g.pad_sure)
        self.exit_known = exit_cell >= 0
        if exit_cell < 0:
            exit_cell, exit_d = nearest(g.pad_seen & (g.pad | g.plain))
        if exit_cell >= 0 and budget - step <= self.leave_factor * exit_d + self.leave_margin:
            self._choice = None
            return GO_EXIT, exit_cell
        # ---- keys and locked doors --------------------------------------------------------------
        # The teacher keeps its own list of the keys it has seen: the map's key flag is wiped when the
        # agent passes within a cell and a half of a key, also when it did not pick it up.
        key = g.key & ~g.visited
        for kx, ky in self._key_cells:
            if 0 < kx - ox < N - 1 and 0 < ky - oy < N - 1:
                key[kx - ox, ky - oy] = True
        t, _ = nearest(key & g.plain)
        if t >= 0:
            self._choice = None
            return GET_KEY, t
        if keys > 0:
            t, _ = nearest(g.locked & g.gate)
            if t >= 0:
                self._choice = None
                return OPEN_LOCKED, t
        # ---- pickups ----------------------------------------------------------------------------
        if self.pickups:
            health, armor, bullets = float(vitals[0]), float(vitals[1]), float(vitals[4])
            best, best_d = -1, INF
            for k in np.flatnonzero(g.item & ~g.visited & g.plain).tolist():
                kind = self._item_kind.get((k // N + ox, k % N + oy), spec.ENT_AMMO)
                if kind == spec.ENT_HEALTH:                    # a health item is left lying at full health
                    wanted, needed = health < 100.0, health < LOW_HEALTH
                elif kind == spec.ENT_ARMOR:
                    wanted, needed = armor < 100.0, False
                else:
                    wanted, needed = True, bullets < LOW_BULLETS
                if wanted and (needed or dist[k] <= NEAR_ITEM_STEPS) and dist[k] < best_d and k != OWN:
                    best, best_d = k, dist[k]
            if best >= 0:
                self._choice = None
                return GET_PICKUP, best
        # ---- exploration: frontier cells, doorways, rooms never entered, dead ends not looked into ----
        looked = self._window(self._looked, ox, oy)
        doorway = self._window(self._doorways, ox, oy)
        if looked is None:
            looked, doorway = np.zeros((N, N), bool), g.door
        unseen = g.plain & ~g.visited & ~looked
        options = [(K_FRONTIER, g.frontier() & ~g.visited), (K_DOORWAY, doorway & ~g.visited & g.passable),
                   (K_ROOM, g.unvisited_rooms(doorway))]
        if not self.exit_known:
            options.append((K_DEAD_END, g.dead_ends() & unseen))
            options.append((K_UNSEEN, unseen))
        best_kind, best, best_d = K_FRONTIER, -1, INF
        keep = None
        for kind, mask in options:
            t, d = nearest(mask)
            if kind == K_UNSEEN and d > NEAR_UNSEEN_STEPS:
                continue
            if kind == K_DEAD_END:
                d *= DEAD_END_WEIGHT
            if d < best_d:
                best_kind, best, best_d = kind, t, d
            if self._choice is not None and self._choice[0] == kind:      # is the current choice still open?
                ci, cj = self._choice[1][0] - ox, self._choice[1][1] - oy
                if 0 <= ci < N and 0 <= cj < N and mask[ci, cj] and dist[ci * N + cj] < INF:
                    keep = (kind, ci * N + cj, dist[ci * N + cj] * (DEAD_END_WEIGHT if kind == K_DEAD_END else 1.0))
        if best >= 0:
            if keep is not None and best_d >= HYSTERESIS * keep[2]:
                best_kind, best = keep[0], keep[1]
            self._choice = (best_kind, (best // N + ox, best % N + oy))
            gate = first_gate(best)
            if gate >= 0 and gate != best:
                return OPEN_DOOR, gate
            if self._choice[1] in self._enter_cells:
                self._enter_target = best
            return KIND_SUBGOAL[best_kind], best
        self._choice = None
        # ---- nothing left to explore ------------------------------------------------------------
        if not self.exit_known:
            t, _ = nearest(unseen)
            if t < 0:
                # A cell with both floor and wall evidence (a stray wall point in an alcove, say) could
                # hide the exit pad: step into each such cell that borders known floor, once.
                best_d = INF
                for k in np.flatnonzero(g.free & g.wall & ~g.door & ~g.locked).tolist():
                    if (k // N + ox, k % N + oy) in self._probed or not 0 < k // N < N - 1 or not 0 < k % N < N - 1:
                        continue
                    d = min(dist[n] for n in (k - N, k + N, k - 1, k + 1) if g.plain_l[n]) if any(
                        g.plain_l[n] for n in (k - N, k + N, k - 1, k + 1)) else INF
                    if d < best_d:
                        t, best_d = k, d
                if t >= 0:
                    self._probe_target = t
                    return SWEEP, t
            if t < 0:                                          # looked everywhere: now step on every cell
                rest = g.plain & ~g.visited
                t, _ = nearest(rest & g.dead_ends())
                if t < 0:
                    t, _ = nearest(rest)
                self._enter_target = t
            elif (t // N + ox, t % N + oy) in self._enter_cells:
                self._enter_target = t
            if t >= 0:
                gate = first_gate(t)
                return (OPEN_DOOR, gate) if gate >= 0 else (SWEEP, t)
        if exit_cell >= 0:
            return GO_EXIT, exit_cell
        # ---- keep moving: the reachable cell visited longest ago ----------------------------------
        reach = np.array(dist, np.float64).reshape(N, N)
        score = np.where(g.plain & (reach < INF), g.recency - 1e-3 * np.minimum(reach, 500.0), np.inf)
        score[CENTER, CENTER] = np.inf
        t = int(np.argmin(score))
        if np.isfinite(score.reshape(-1)[t]):
            gate = first_gate(t)
            return (OPEN_DOOR, gate) if gate >= 0 else (EXPLORE, t)
        return EXPLORE, OWN

    # ---------------------------------------------------------------------------- steering
    def _steer(self, plan: _Plan, crop, x, y, cos_t, sin_t, vx, vy, cx, cy, ox, oy, step, age, prev_use,
               gate_ahead, dist0, hit0) -> tuple[int, int, int, int]:
        """(move, strafe, turn, use) heads for this step."""
        tic = step * spec.FRAME_SKIP
        held = self._held
        px, py = cx * CELL, cy * CELL                          # centre of the cell the agent stands in
        qi, qj = cx - plan.ox, cy - plan.oy
        info = self.info
        info.pop("waypoint", None)
        if not (plan.valid and 0 <= qi < N and 0 <= qj < N):
            info["mode"] = "idle"
            move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, px, py, True)
            return move, strafe, steer.NONE, 0
        q = qi * N + qj
        link = plan.link
        cells = [q]
        c = q
        for _ in range(LOOK):
            c = link[c]
            if c < 0:
                break
            cells.append(c)
        info["route"] = [(c // N + plan.ox, c % N + plan.oy) for c in cells]
        if len(cells) == 1:                                    # in the target cell: settle on its centre
            info["mode"] = "arrive"
            move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, px, py, True)
            return move, strafe, steer.NONE, 0

        # ---- what stops the route ahead: a closed door, or a cell the camera has not looked at --------
        di, dj = ox - plan.ox, oy - plan.oy
        guard = self.look_first and not self.exit_known and abs(cx) < HALF - LOOK - 1 and abs(cy) < HALF - LOOK - 1
        gx, gy = float(cells[1] // N - qi), float(cells[1] % N - qj)           # direction of the next cell
        # The map keeps a door marked until rays have crossed the whole cell. Looking straight at the
        # next cell, a ray that runs on past its middle shows that nothing closes it now.
        open_ahead = (dist0 > 1.25 * CELL - ((x - px) * gx + (y - py) * gy)
                      and steer.facing(cos_t, sin_t, gx, gy, USE_FACING_SINE))
        gate_k = look_k = 0
        for k in range(1, len(cells)):
            i, j = cells[k] // N - di, cells[k] % N - dj
            if crop[spec.CH_DOOR, i, j] > 0.5 or crop[spec.CH_LOCKED, i, j] > 0.5:
                if k == 1 and open_ahead:
                    continue
                gate_k = k
                break
            if guard and crop[spec.CH_VISITED, i, j] < 0.5 and not self._looked[i + ox + HALF, j + oy + HALF] \
                    and (i + ox, j + oy) not in self._waived and not (plan.enter and cells[k] == plan.target):
                look_k = k
                break

        if gate_k == 1:                                        # the door routine
            faces = steer.facing(cos_t, sin_t, gx, gy, USE_FACING_SINE)
            creep = faces and hit0 == spec.HIT_ITEM and dist0 < ITEM_IN_WAY     # a pickup hides the door: step past it
            move, strafe = steer.wait_at_door(x, y, cos_t, sin_t, vx, vy, px, py, gx, gy, creep)
            turn = steer.turn_head(cos_t, sin_t, gx, gy, tic, held)
            if prev_use:
                use = int(gate_ahead and age < USE_REPRESS_STEPS)
            else:
                use = int(gate_ahead and dist0 <= USE_PRESS_RANGE and age >= USE_REPRESS_STEPS and faces)
            info["mode"] = "door"
            return move, strafe, turn, use
        if gate_ahead and dist0 <= BLOCKED_RANGE and steer.facing(cos_t, sin_t, gx, gy, BLOCKED_SINE):
            # A closed door is right in front, in the direction of travel, although the map shows floor
            # there (the door was drawn a cell off, or has closed unseen): stand still and open it.
            # Not a locked door without a key or one already given up, and not for ever.
            beyond = cell_of(x + (dist0 + 8.0) * cos_t, y + (dist0 + 8.0) * sin_t)
            ahead = (cells[1] // N + plan.ox, cells[1] % N + plan.oy)
            usable = hit0 == spec.HIT_DOOR or (self._keys > 0 and self.failed.get(beyond) != self._keys)
            if usable and beyond in ((cx, cy), ahead) and beyond not in self._left_alone:
                if self._blocked is not None and self._blocked[0] == beyond:
                    self._blocked[1] += 1
                else:
                    self._blocked = [beyond, 1]
                if self._blocked[1] > BLOCKED_PATIENCE:
                    self._left_alone.add(beyond)
                else:
                    move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, x, y, True)
                    use = int(age < USE_REPRESS_STEPS) if prev_use else int(age >= USE_REPRESS_STEPS)
                    info["mode"] = "blocked"
                    return move, strafe, steer.NONE, use
        if plan.subgoal == SWEEP and not plan.enter:
            # A sweep target only has to be looked at: turn to it from here if it is within the camera's
            # range and the line to it is clear. If that does not get it seen, walk into it after all.
            tx, ty = plan.target_cell[0] * CELL, plan.target_cell[1] * CELL
            if math.hypot(tx - x, ty - y) <= GAZE_RANGE and \
                    steer.swept_clear(plan.plain, plan.ox, plan.oy, x, y, tx, ty, (q, plan.target)):
                if self._peek is not None and self._peek[0] == plan.target_cell:
                    self._peek[1] += 1
                else:
                    self._peek = [plan.target_cell, 1]
                if self._peek[1] > LOOK_PATIENCE:
                    self._enter_cells.add(plan.target_cell)
                    plan.enter = True
                else:
                    move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, x, y, True)
                    info["mode"] = "look"
                    return move, strafe, steer.turn_head(cos_t, sin_t, tx - x, ty - y, tic, held), 0
        if plan.probe and len(cells) == 2:                     # pushing into a doubtful cell: not for ever
            if self._probe is not None and self._probe[0] == plan.target_cell:
                self._probe[1] += 1
            else:
                self._probe = [plan.target_cell, 1]
            if self._probe[1] > PROBE_PATIENCE:
                self._probed.add(plan.target_cell)

        if look_k == 1:                                        # look into the next cell before stepping into it
            cell = (cells[1] // N + plan.ox, cells[1] % N + plan.oy)
            if self._peek is not None and self._peek[0] == cell:
                self._peek[1] += 1
            else:
                self._peek = [cell, 1]
            if self._peek[1] > LOOK_PATIENCE:
                self._waived.add(cell)                         # it cannot be seen from here: go in
            else:
                move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, x, y, True)        # come to rest
                info["mode"] = "look"
                return move, strafe, steer.turn_head(cos_t, sin_t, cell[0] * CELL - x, cell[1] * CELL - y, tic, held), 0

        block_k = gate_k or look_k
        limit = block_k - 1 if block_k else len(cells) - 1
        # The view leads the body: it aims a few cells ahead along the route, at the cell that has
        # to be looked at, or at a door that is about to be reached.
        lead = cells[look_k] if look_k > 1 else cells[min(LEAD, len(cells) - 1)]
        fx, fy = (lead // N + plan.ox) * CELL - x, (lead % N + plan.oy) * CELL - y
        if gate_k:
            ax, ay = (cells[gate_k - 1] // N + plan.ox) * CELL, (cells[gate_k - 1] % N + plan.oy) * CELL
            if math.hypot(ax - x, ay - y) < PRETURN_RANGE:
                fx, fy = float(cells[gate_k] // N - cells[gate_k - 1] // N), float(cells[gate_k] % N - cells[gate_k - 1] % N)
        turn = steer.turn_head(cos_t, sin_t, fx, fy, tic, held)

        allow = set(cells[:limit + 1])                         # the route itself is passable (an opened door too)
        allow.add(plan.target)
        for k in range(limit, 0, -1):
            wx, wy = (cells[k] // N + plan.ox) * CELL, (cells[k] % N + plan.oy) * CELL
            if steer.swept_clear(plan.plain, plan.ox, plan.oy, x, y, wx, wy, allow):
                stop = k < limit or bool(block_k)
                move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, wx, wy, stop)
                info["mode"], info["waypoint"] = ("slow" if stop else "run"), (wx, wy)
                return move, strafe, turn, 0
        # No cell of the route can be reached in a straight line: get onto the line to the next cell first.
        ax, ay = float(cells[1] // N - qi), float(cells[1] % N - qj)
        norm = math.hypot(ax, ay)
        move, strafe = steer.hold_line(x, y, cos_t, sin_t, vx, vy, px, py, ax / norm, ay / norm)
        info["mode"] = "line"
        return move, strafe, turn, 0
