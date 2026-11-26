"""Training-time scaffolding. Never imported by the agent in submission/.

Fair navigation teacher: the hand-written explorer whose labels the student network is trained
to imitate on navigation steps. It contains graph search, which is why it lives under training/.

Fairness. `Teacher.label` is a function of what the student network receives at the same step and
of nothing else: the memory-map crop (`MemoryMap.crop`), the pose estimate, the current observation
(rays, entities, vitals), the encoder scalars and the previous action. It never sees the true
position, the map file or any other ground truth. The state it keeps between calls is derived from
those same inputs and from the actions that were actually taken, so it stays valid when the student
drives: the plan made at the last manager step, the turn-key hold counter, which locked doors were
tried with the keys now held, which floor cells the camera has already looked at, which cells the
agent has stood in, and which single exit sightings were looked at again and not repeated.

Three rules keep that state honest when somebody else holds the controls:

* A cell counts as looked at only on steps on which the exit sensor would have reported a pad
  there: its centre in front, within 40 degrees of the view axis, within range, and the view ray
  towards it not cut short (`_mark_looked`).
* Patience counters (steps spent facing a cell that will not come into view, pushing into a
  doubtful cell, opening a door the map does not show) advance only on steps whose executed
  action was the one the previous label asked for. A driver that ignores the labels runs no
  patience out.
* What a patience running out decides (enter the cell unseen, step on it, leave the door alone)
  is forgotten again after GIVE_UP_STEPS steps.

A body that was pushed for STALL_STEPS executed steps while the observation reports no speed is
caught on a wall corner; for that step the teacher asks for the centre line of its route instead
of the waypoint, which takes the body off the corner. In front of a door the forward ray is the
better measurement: when it reads the door further away than the pose estimate implies, the body
is placed by the ray, so it cannot come to rest just out of USE reach.

Grid fit. The teacher plans on the 64-unit cells of the start frame, which fit the walls only when
the episode starts at a cell centre facing along an axis. It checks this on the first observation
(`_check_grid`); if the walls are elsewhere, `grid_fits` is False for the episode and every label
carries an empty `head_mask` and no value field, so nothing is distilled from such a map.

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
   has not looked into, any floor within a few cells that it has not looked at, or a cell with a
   single exit sighting that has not been repeated (sweep). The current choice is kept until it
   is done or another is less than half as far. If the way to the choice leads through a closed
   door, that door is the target first (open_door). A sweep target is done as soon as the camera
   has looked at it; the teacher turns to it from wherever it has a straight clear line and does
   not walk into it.
7. sweep        no exit known: the nearest floor cell neither visited nor looked at, then every
                unvisited cell, dead ends first.
8. go_exit      nothing else remains.
9. explore      the least recently visited reachable cell, so the agent keeps moving.

Exit belief. The exit enters the map only through the exit detector's sightings, which can be
missed, misplaced or false. A cell counts as the exit once it was sighted and there are two
sightings in it and its eight neighbours together. A single sighting is looked at again from
within CHECK_RANGE: a real pad is reported again at once, and a sighting that CHECK_LOOKS close
looks do not repeat is dropped. A cell in which the agent has stood (well inside it) is never the
exit, whatever the detector says afterwards: the episode would have ended there.

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
VIEW_RANGE = 600.0            # how far the camera is trusted to show an exit pad (the exit labels reach 640)
VIEW_TAN = math.tan(math.radians(40.0))   # a cell is looked at when its centre is this close to the view axis: the exit
                              # labels report a pad whose centre is within 42 degrees of it, and nothing wider
VIEW_MIN = 16.0               # ... and at least this far away (the labels' nearest range)
VIEW_SOLID_SLACK = 8.0        # the view ray towards a cell centre may end on a wall or door this far short of it
GAZE_MARGIN = 60.0            # a sweep target is looked at from where the agent stands only this far inside the range
_VIEW_COS = np.cos(VIEW_RAYS * (2.0 * np.pi / spec.NUM_RAYS))
_VIEW_SIN = np.sin(VIEW_RAYS * (2.0 * np.pi / spec.NUM_RAYS))
CHECK_RANGE = 256.0           # an exit sighting that was not repeated is looked at again from within this distance
CHECK_LOOKS = 6               # ... and dropped when that many close looks bring no second sighting (a detector that
                              # misses one frame in five misses six in a row once in 15,000 times)
CHECK_WEIGHT = 0.5            # such a cell counts as this much nearer than it is: it may be the exit
STOOD_INSIDE = 20.0           # the body within this of a cell's centre on both axes has stood in the cell: not the exit
HALF = 2 * N                  # the teacher's own start-frame grids hold cells -HALF .. HALF: cell k is index k + HALF
ABS = 2 * HALF + 1
SOLID_HITS = (spec.HIT_WALL, spec.HIT_DOOR, spec.HIT_LOCKED)
THING_SLACK = 40.0            # a view ray that ends on a pickup or a monster still shows the floor this far behind it
GRID_TOLERANCE = 0.5          # a wall hit of the first observation this far from every line x or y = 64k + 32 of the
GRID_MISSES = 2               # start frame is off the grid; this many such hits and the grid does not fit the map
_RAY_ANGLES = np.arange(spec.NUM_RAYS) * (2.0 * np.pi / spec.NUM_RAYS)
DOOR_GAP = 4.0                # the forward ray reads a door this much further away than the pose estimate implies:
                              # the estimate is off, and the body is placed in front of the door by the ray
STALL_SPEED = 0.25            # reported speed (units per tic, either axis) under which a body that is being pushed
STALL_STEPS = 6               # is not moving; after this many such steps in a row it is caught on a wall corner
GIVE_UP_STEPS = 96            # a cell or door given up on (entered unseen, stepped on, left alone) is forgotten again
                              # after this many steps, so no give-up outlasts the situation that caused it
_I_USE_AGE = spec.SCALAR_INDEX["use_age"]
_I_STEP, _I_BUDGET, _I_PREV = (spec.SCALAR_INDEX[n] for n in ("step_frac", "budget_frac", "prev_fwd"))
_AGE_SCALE = 16.0             # the encoder's use_age is steps / 16, clipped to 1

EXPLORE, SWEEP, GET_KEY, OPEN_DOOR, OPEN_LOCKED, GO_EXIT, FIGHT, GET_PICKUP = (
    spec.SUBGOAL_INDEX[n] for n in ("explore", "sweep", "get_key", "open_door", "open_locked", "go_exit", "fight", "get_pickup"))
K_FRONTIER, K_DOORWAY, K_ROOM, K_DEAD_END, K_UNSEEN, K_CHECK = range(6)      # kinds of exploration target
KIND_SUBGOAL = (EXPLORE, OPEN_DOOR, EXPLORE, SWEEP, SWEEP, SWEEP)
KIND_WEIGHT = (1.0, 1.0, 1.0, DEAD_END_WEIGHT, 1.0, CHECK_WEIGHT)


def _nav_heads(buttons) -> tuple[int, int, int, int]:
    """(move, strafe, turn, use) head values of a (9,) button array, as `spec.buttons_to_heads` gives them."""
    b = [bool(v) for v in buttons[:8]]
    return ((2 if b[1] else 0) if not b[0] else (0 if b[1] else 1), (2 if b[3] else 0) if not b[2] else (0 if b[3] else 1),
            (2 if b[5] else 0) if not b[4] else (0 if b[5] else 1), int(b[7]))


@dataclass
class TeacherLabel:
    """What the teacher would do at one step.

    subgoal      index into spec.SUBGOALS
    target       crop cell index i * CROP_SIZE + j of the target (the agent's own cell if none)
    heads        (6,) int64 action heads the teacher would take (attack and weapon always 0)
    head_mask    (6,) bool; False = no label for this head on this step (all False on fight steps
                 and for the whole episode when the teacher's grid does not fit the map)
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
                 "probe", "enter", "check", "arrived", "_shift", "_shifted")

    def __init__(self, subgoal, target, ox, oy, link, plain, value, step, valid, travel, probe=False, enter=False,
                 check=False):
        self.probe = probe                                          # the target is a doubtful cell to be tried
        self.enter = enter or probe                                 # a sweep target that has to be stepped on, not looked at
        self.check = check                                          # the target is an unconfirmed exit sighting to look at again
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
_NO_CELLS = np.zeros(0, np.intp)


class Teacher:
    """See the module docstring. `reset()` before every map, `label()` once per observation."""

    def __init__(self, leave_factor: float = LEAVE_FACTOR, leave_margin: float = LEAVE_MARGIN,
                 lock_tries: int = LOCK_TRIES, door_steps: float = DOOR_STEPS, pickups: bool = True,
                 pad_floor: float = EXIT_SURE, look_first: bool = True, looks: int = LOOKS,
                 view_range: float = VIEW_RANGE):
        self.leave_factor, self.leave_margin = float(leave_factor), float(leave_margin)
        self.lock_tries, self.door_steps, self.pickups = int(lock_tries), float(door_steps), bool(pickups)
        self.pad_floor, self.look_first, self.looks = float(pad_floor), bool(look_first), int(looks)
        # Set `view_range` to the distance at which the exit detector in use still finds the pad
        # reliably: floor further away than this is not taken as inspected.
        self.view_range = min(max(float(view_range), 2.0 * CELL), spec.RAY_RANGE)
        self.gaze_range = self.view_range - GAZE_MARGIN
        # Offsets of the cells that can lie within view range of a body anywhere in the middle cell.
        reach = int(math.ceil(self.view_range / CELL)) + 1
        span = np.arange(-reach, reach + 1)
        gi, gj = (g.reshape(-1) for g in np.meshgrid(span, span, indexing="ij"))
        self._win_reach = reach
        self._win_x, self._win_y = gi * CELL, gj * CELL          # centre of each window cell from the middle cell's centre
        self._win_flat = (gi * ABS + gj).astype(np.intp)         # ... and its index from the middle cell's index
        self._look_count = np.zeros((ABS, ABS), np.uint16)
        self._looked = np.zeros((ABS, ABS), bool)
        self._doorways = np.zeros((ABS, ABS), bool)
        self._stood = np.zeros((ABS, ABS), bool)
        self._dismissed = np.zeros((ABS, ABS), bool)
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
        self._stood.fill(False)         # cells the body has been well inside: the episode went on, so not the exit
        self._dismissed.fill(False)     # cells with one exit sighting that close looks did not repeat
        self._checks: dict = {}         # such a cell -> close looks so far
        self._near_now = _NO_CELLS      # flat indices of the cells the camera looked at from close by on this step
        self._check_target = -1         # set by _choose when its target is such a cell
        self._asked = None              # (move, strafe, turn, use) of the last label ...
        self._followed = False          # ... and whether the action executed after it was exactly that
        self._stall = 0                 # executed steps in a row that pushed the body and did not move it
        self.unstuck = 0                # steps on which the body was taken off a wall corner, or moved up to a door
                                        # that was out of reach from where it had come to rest (see _steer)
        self._patience: dict = {}       # routine -> [cell, steps]: executed steps spent on looking at an unseen cell
                                        # ("look"), pushing into a doubtful one ("probe"), opening an unmapped door ("door")
        self._waived: dict = {}         # unseen cells that are entered anyway (patience ran out) -> step of that decision
        self._item_kind: dict = {}      # cell -> entity type of the pickup last seen there
        self._key_cells: set = set()    # cells where a key was seen and has not been walked over since
        self._probed: set = set()       # doubtful cells that were tried (entered, or found solid)
        self._probe_target = -1         # set by _choose when its target is a doubtful cell
        self._enter_target = -1         # ... or an unvisited cell that is to be stepped on
        self._enter_cells: dict = {}    # sweep targets that could not be seen from outside: step on them -> step decided
        self._left_alone: dict = {}     # unmapped closed doors that did not open -> step at which they were left
        self.grid_fits = True           # False: the walls do not lie on the 64-unit cells of the start frame (see _check_grid)
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
            self._asked = None
            self.info = {"mode": "error"}
            return TeacherLabel(EXPLORE, OWN, np.zeros(6, np.int64), _MASK_NONE.copy(), _NO_FIELD, False)

    def label_from_scalars(self, obs, crop, pose, scal) -> TeacherLabel:
        """`label()` with the step, the time budget and the previous action read back from the
        encoder's scalar vector (`step_frac`, `budget_frac`, `prev_*`), for a rollout loop that
        has the network's inputs at hand and nothing else. The budget field is clipped by the
        encoder at 4 x 1050 steps."""
        scal = np.asarray(scal)
        step = int(round(float(scal[_I_STEP]) * spec.STEP_NORM))
        prev = (scal[_I_PREV:_I_PREV + spec.NUM_BUTTONS] > 0.5).astype(np.int8) if step > 0 else None
        return self.label(obs, crop, pose, scal, step, float(scal[_I_BUDGET]) * spec.STEP_NORM, prev)

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
        # Patience counters measure what was tried, so they run only on steps whose executed action
        # was the one the last label asked for (always, when the teacher drives; when the student
        # drives and does something else, nothing has been tried yet).
        self._followed = (self._asked is not None and prev_action is not None
                          and _nav_heads(prev_action) == self._asked)
        if step == 0:
            self._check_grid(ray_dist, ray_hit, x, y, theta)
        self._mark_looked(ray_dist, ray_hit, x, y, cos_t, sin_t)
        if abs(cx) < HALF and abs(cy) < HALF and abs(x - cx * CELL) <= STOOD_INSIDE and abs(y - cy * CELL) <= STOOD_INSIDE:
            self._stood[cx + HALF, cy + HALF] = True
        self._judge_sightings(crop, ox, oy)
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
        pushed = self._followed and (self._asked[0] != 0 or self._asked[1] != 0)       # the last label, and it was executed
        self._stall = self._stall + 1 if pushed and abs(vf) < STALL_SPEED and abs(vr) < STALL_SPEED else 0
        vx, vy = vf * cos_t + vr * sin_t, vf * sin_t - vr * cos_t
        move, strafe, turn, use = self._steer(plan, crop, x, y, cos_t, sin_t, vx, vy, cx, cy, ox, oy,
                                              step, age, prev_use, gate_ahead, dist0, hit0)
        heads = np.array([move, strafe, turn, 0, use, 0], np.int64)
        self._asked = (int(move), int(strafe), int(turn), int(use))

        # ---- label ----------------------------------------------------------------------------
        ti = min(max(plan.target_cell[0] - ox, 0), N - 1)
        tj = min(max(plan.target_cell[1] - oy, 0), N - 1)
        self.info.update(step=step, subgoal=spec.SUBGOALS[plan.subgoal], target=plan.target_cell, cell=(cx, cy),
                         heads=heads.tolist(), fight=enemy is not None, keys=keys, grid_fits=self.grid_fits)
        if enemy is not None:
            return TeacherLabel(FIGHT, enemy, heads, _MASK_NONE.copy(), _NO_FIELD, False)
        if not self.grid_fits:                                 # the plan was made on cells that straddle the walls
            return TeacherLabel(plan.subgoal, ti * N + tj, heads, _MASK_NONE.copy(), _NO_FIELD, False)
        return TeacherLabel(plan.subgoal, ti * N + tj, heads, _MASK_NAV.copy(), plan.field_in(ox, oy), plan.valid)

    # ------------------------------------------------------------------------- bookkeeping
    def _check_grid(self, ray_dist, ray_hit, x, y, theta) -> None:
        """Do the walls of the first observation lie on the 64-unit cells of the start frame?

        The teacher plans on those cells. In a grid layout entered at a cell centre and facing
        along an axis every wall face lies on a line x = 64k + 32 or y = 64k + 32 of the start
        frame, and the end points of the wall rays show it (they are off by under 0.001 units).
        A start that is shifted inside its cell or turned by another angle puts the walls
        elsewhere; the cells then straddle them, corridors read as blocked and the teacher's
        choices are not worth imitating. `grid_fits` is set False for the episode and every label
        comes back with an empty `head_mask` and no value field.
        """
        wall = np.asarray(ray_hit) == spec.HIT_WALL
        reach = np.asarray(ray_dist, np.float64)[wall] * spec.RAY_RANGE
        angle = theta + _RAY_ANGLES[wall]
        off_x = np.abs((x + reach * np.cos(angle)) % CELL - CELL / 2)
        off_y = np.abs((y + reach * np.sin(angle)) % CELL - CELL / 2)
        self.grid_fits = int(np.count_nonzero(np.minimum(off_x, off_y) > GRID_TOLERANCE)) < GRID_MISSES

    def _mark_looked(self, ray_dist, ray_hit, x, y, cos_t, sin_t) -> None:
        """Count one look at every cell in which the exit sensor would report a pad right now.

        The sensor reports a pad when the pad's centre is inside the camera's view, within range
        and in line of sight. The same test is applied to the centre of every cell around the
        agent: it lies in front, within VIEW_TAN of the view axis (dot and cross products of the
        heading and the offset), between VIEW_MIN and `view_range` away, and the view ray whose
        direction is nearest to it (largest dot product) runs at least as far as the centre. A
        cell counts as looked at once this has held on `looks` different steps, so a missed
        detection or two do not hide an exit pad.
        """
        self._near_now = _NO_CELLS
        edge = (HALF - self._win_reach - 1) * CELL
        if not (abs(x) < edge and abs(y) < edge):
            return
        reach = np.asarray(ray_dist, np.float64)[VIEW_RAYS] * spec.RAY_RANGE
        solid = np.isin(np.asarray(ray_hit)[VIEW_RAYS], SOLID_HITS)
        reach = np.where(solid, reach + VIEW_SOLID_SLACK, reach + THING_SLACK)
        cx, cy = cell_of(x, y)
        dx = self._win_x + (cx * CELL - x)
        dy = self._win_y + (cy * CELL - y)
        fwd = dx * cos_t + dy * sin_t
        left = dy * cos_t - dx * sin_t
        cone = np.flatnonzero((fwd > 0.0) & (np.abs(left) <= fwd * VIEW_TAN))
        fwd, left = fwd[cone], left[cone]
        dist = np.hypot(fwd, left)
        ray = np.argmax(fwd[:, None] * _VIEW_COS + left[:, None] * _VIEW_SIN, axis=1)
        ok = (dist >= VIEW_MIN) & (dist <= self.view_range) & (reach[ray] >= dist)
        flat = self._win_flat[cone[ok]] + ((cx + HALF) * ABS + (cy + HALF))
        count = self._look_count.reshape(-1)
        count[flat] += 1
        self._looked.reshape(-1)[flat[count[flat] >= self.looks]] = True
        self._near_now = flat[dist[ok] <= CHECK_RANGE]

    def _spent(self, routine: str, cell) -> int:
        """Executed steps spent so far on `routine` for `cell`, this step included.

        The count goes up only when the action executed on the previous step was the one the
        teacher asked for, so a driver that ignores the labels does not run a patience out.
        """
        record = self._patience.get(routine)
        if record is None or record[0] != cell:
            record = self._patience[routine] = [cell, 0]
        record[1] += self._followed
        return record[1]

    @staticmethod
    def _given_up(book: dict, cell, step: int) -> bool:
        """Is `cell` in a give-up book (cell -> step of the decision) and not yet forgotten?"""
        since = book.get(cell)
        if since is None:
            return False
        if step - since > GIVE_UP_STEPS or step < since:
            del book[cell]
            return False
        return True

    def _unconfirmed(self, crop, i: int, j: int, ox: int, oy: int) -> bool:
        """Is crop cell (i, j) an exit sighting that still wants a second look: seen once, not
        confirmed, not stood in and not dropped?"""
        if not (0 < i < N - 1 and 0 < j < N - 1) or crop[spec.CH_EXIT, i, j] < EXIT_SEEN:
            return False
        a, b = i + ox + HALF, j + oy + HALF
        if not (0 <= a < ABS and 0 <= b < ABS) or self._stood[a, b] or self._dismissed[a, b]:
            return False
        return float(crop[spec.CH_EXIT, i - 1:i + 2, j - 1:j + 2].sum()) < EXIT_SURE

    def _judge_sightings(self, crop, ox, oy) -> None:
        """Count the close looks at every unconfirmed exit sighting and drop the ones that stay single.

        A real pad looked at from within CHECK_RANGE is reported again within a step or two, so a
        sighting that CHECK_LOOKS close looks do not repeat was a false detection.
        """
        seen = np.argwhere(crop[spec.CH_EXIT] >= EXIT_SEEN)
        if not len(seen):
            return
        near = self._near_now
        for i, j in seen.tolist():
            if not self._unconfirmed(crop, i, j, ox, oy):
                continue
            a, b = i + ox + HALF, j + oy + HALF
            if len(near) and bool((near == a * ABS + b).any()):
                cell = (i + ox, j + oy)
                looks = self._checks.get(cell, 0) + 1
                self._checks[cell] = looks
                if looks >= CHECK_LOOKS:
                    self._dismissed[a, b] = True

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
                plan.arrived += self._followed                 # if it does not, this was not the exit
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
            elif crop[spec.CH_EXIT, ti, tj] >= self.pad_floor and not self._stood_in(plan.target_cell):
                return True                                    # the target turned out to be the exit pad
            if plan.check:
                if not self._unconfirmed(crop, ti, tj, ox, oy):
                    return True                                # confirmed, or dropped
            elif sub == SWEEP and not plan.enter and abs(plan.target_cell[0]) < HALF and abs(plan.target_cell[1]) < HALF \
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
            if crop[spec.CH_WALL, i, j] > 0.5 or (crop[spec.CH_EXIT, i, j] >= self.pad_floor and c != plan.target
                                                  and not self._stood_in((i + ox, j + oy))):
                return True
        return False

    def _stood_in(self, cell) -> bool:
        a, b = cell[0] + HALF, cell[1] + HALF
        return 0 <= a < ABS and 0 <= b < ABS and bool(self._stood[a, b])

    def _decide(self, crop, vitals, ox, oy, step, budget, keys) -> _Plan:
        grid = PlanGrid(crop, self._locked_ok(crop, ox, oy, keys), self.door_steps, self.pad_floor,
                        self._window(self._stood, ox, oy))
        doorways = self._window(self._doorways, ox, oy)
        if doorways is not None:
            doorways |= grid.door | grid.locked
        dist, back = grid.from_agent()
        self.searches += 1
        self._probe_target = self._enter_target = self._check_target = -1
        subgoal, target = self._choose(grid, dist, back, vitals, ox, oy, step, budget, keys)
        valid = target != OWN
        if valid:
            field, link = grid.to_target(target)
            self.searches += 1
            value = np.minimum(np.array(field, np.float32).reshape(N, N), VALUE_CAP)
        else:
            link, value = [-1] * (N * N), _NO_FIELD
        self.plan = _Plan(subgoal, target, ox, oy, link, grid.plain_l, value, step, valid, dist[target],
                          probe=target == self._probe_target, enter=target == self._enter_target,
                          check=target == self._check_target)
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
        single = g.pad_seen & ~g.pad_sure & (g.pad | g.plain)      # sightings that were not repeated ...
        dropped = self._window(self._dismissed, ox, oy)
        maybe = single & ~dropped if dropped is not None else single   # ... and have not been looked at closely yet
        if exit_cell < 0:
            exit_cell, exit_d = nearest(maybe)
        if exit_cell < 0:
            exit_cell, exit_d = nearest(single)
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
            options.append((K_CHECK, maybe))
        best_kind, best, best_d = K_FRONTIER, -1, INF
        keep = None
        for kind, mask in options:
            t, d = nearest(mask)
            if kind == K_UNSEEN and d > NEAR_UNSEEN_STEPS:
                continue
            d *= KIND_WEIGHT[kind]
            if d < best_d:
                best_kind, best, best_d = kind, t, d
            if self._choice is not None and self._choice[0] == kind:      # is the current choice still open?
                ci, cj = self._choice[1][0] - ox, self._choice[1][1] - oy
                if 0 <= ci < N and 0 <= cj < N and mask[ci, cj] and dist[ci * N + cj] < INF:
                    keep = (kind, ci * N + cj, dist[ci * N + cj] * KIND_WEIGHT[kind])
        if best >= 0:
            if keep is not None and best_d >= HYSTERESIS * keep[2]:
                best_kind, best = keep[0], keep[1]
            self._choice = (best_kind, (best // N + ox, best % N + oy))
            gate = first_gate(best)
            if gate >= 0 and gate != best:
                return OPEN_DOOR, gate
            if best_kind == K_CHECK:
                self._check_target = best
            elif self._given_up(self._enter_cells, self._choice[1], step):
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
            elif self._given_up(self._enter_cells, (t // N + ox, t % N + oy), step):
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
                    and not self._given_up(self._waived, (i + ox, j + oy), step) \
                    and not (plan.enter and cells[k] == plan.target):
                look_k = k
                break

        if gate_k == 1:                                        # the door routine
            faces = steer.facing(cos_t, sin_t, gx, gy, USE_FACING_SINE)
            creep = faces and hit0 == spec.HIT_ITEM and dist0 < ITEM_IN_WAY     # a pickup hides the door: step past it
            # Facing the door, the forward ray measures the distance to its face. If that is more than
            # DOOR_GAP beyond what the pose estimate implies, the estimate is off along the door's axis
            # and the body, at rest where the map says the door is opened from, can be out of USE reach
            # for ever. The body is then placed by the ray's reading instead.
            far = None
            if faces and not prev_use and (hit0 == spec.HIT_DOOR or hit0 == spec.HIT_LOCKED):
                gap = dist0 - (CELL / 2 - ((x - px) * gx + (y - py) * gy))
                if DOOR_GAP < gap <= CELL / 2:
                    far = dist0
                    self.unstuck += dist0 > USE_PRESS_RANGE and math.hypot(vx, vy) < STALL_SPEED
            move, strafe = steer.wait_at_door(x, y, cos_t, sin_t, vx, vy, px, py, gx, gy, creep, far)
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
            if usable and beyond in ((cx, cy), ahead) and not self._given_up(self._left_alone, beyond, step):
                if self._spent("door", beyond) > BLOCKED_PATIENCE:
                    self._left_alone[beyond] = step
                    del self._patience["door"]
                else:
                    move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, x, y, True)
                    use = int(age < USE_REPRESS_STEPS) if prev_use else int(age >= USE_REPRESS_STEPS)
                    info["mode"] = "blocked"
                    return move, strafe, steer.NONE, use
        if plan.subgoal == SWEEP and not plan.enter:
            # A sweep target only has to be looked at: turn to it from here if it is within the camera's
            # range and the line to it is clear. If that does not get it seen, walk into it after all.
            tx, ty = plan.target_cell[0] * CELL, plan.target_cell[1] * CELL
            if math.hypot(tx - x, ty - y) <= (CHECK_RANGE - GAZE_MARGIN if plan.check else self.gaze_range) and \
                    steer.swept_clear(plan.plain, plan.ox, plan.oy, x, y, tx, ty, (q, plan.target)):
                if self._spent("look", plan.target_cell) > LOOK_PATIENCE:
                    self._enter_cells[plan.target_cell] = step
                    del self._patience["look"]
                    plan.enter = True
                else:
                    move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, x, y, True)
                    info["mode"] = "look"
                    return move, strafe, steer.turn_head(cos_t, sin_t, tx - x, ty - y, tic, held), 0
        if plan.probe and len(cells) == 2:                     # pushing into a doubtful cell: not for ever
            if self._spent("probe", plan.target_cell) > PROBE_PATIENCE:
                self._probed.add(plan.target_cell)

        if look_k == 1:                                        # look into the next cell before stepping into it
            cell = (cells[1] // N + plan.ox, cells[1] % N + plan.oy)
            if self._spent("look", cell) > LOOK_PATIENCE:
                self._waived[cell] = step                      # it cannot be seen from here: go in
                del self._patience["look"]
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
        # The keys push in eight directions only, so the body does not follow the straight line that
        # was tested below exactly, and now and then it ends up a unit short of clearing a wall
        # corner, pushing against it for good. A body that has been pushed for STALL_STEPS executed
        # steps without moving is put back on the centre line of the route first.
        stalled = self._stall >= STALL_STEPS
        self.unstuck += stalled
        for k in (() if stalled else range(limit, 0, -1)):
            wx, wy = (cells[k] // N + plan.ox) * CELL, (cells[k] % N + plan.oy) * CELL
            if steer.swept_clear(plan.plain, plan.ox, plan.oy, x, y, wx, wy, allow):
                stop = k < limit or bool(block_k)
                if not stop and cells[k] == plan.target and plan.subgoal != GO_EXIT:
                    # The route ends here. Run through only if the floor goes on in the direction of
                    # travel; what lies past a frontier cell is not known, and a body that arrives at
                    # full speed in front of a door or a wall cannot stop any more.
                    beyond = 2 * cells[k] - cells[k - 1]
                    stop = not (0 <= beyond < N * N and plan.plain[beyond])
                move, strafe = steer.drive(x, y, cos_t, sin_t, vx, vy, wx, wy, stop)
                info["mode"], info["waypoint"] = ("slow" if stop else "run"), (wx, wy)
                return move, strafe, turn, 0
        # No cell of the route can be reached in a straight line: get onto the line to the next cell first.
        ax, ay = float(cells[1] // N - qi), float(cells[1] % N - qj)
        norm = math.hypot(ax, ay)
        move, strafe = steer.hold_line(x, y, cos_t, sin_t, vx, vy, px, py, ax / norm, ay / norm)
        info["mode"] = "line"
        return move, strafe, turn, 0
