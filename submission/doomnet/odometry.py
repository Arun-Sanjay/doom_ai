"""Odometry: the agent's pose in the start frame, from its own actions and its own observations.

The engine never tells the agent where it is. This module integrates the pose from three things
the agent does have: the buttons it pressed, the ego velocity in `vitals`, and the 64 range rays.

Heading
    Exact. A turn key turns 320/65536 of a circle per tic for the first five tics it is held
    and 640/65536 per tic afterwards, and the first seven tics of an episode ignore input. Both
    rates are whole multiples of 1/1024 of a circle, so the heading is kept as an integer.

Position, free motion
    Each of the four tics of a step: add thrust along the heading, move by the momentum, multiply
    the momentum by the friction factor. The momentum is re-read from `vitals` after every step,
    so nothing accumulates. If the momentum predicted this way equals the reported one, nothing
    disturbed the step and the predicted displacement is used as it is.

Position, disturbed steps
    A wall, a monster or a hit changes the momentum in the middle of a step, and the prediction
    no longer matches the report. The displacement of such a step is measured from the rays
    instead: walls do not move, so every wall point of the new scan must lie on the straight wall
    that the previous scan saw at that place. Each matched ray gives one linear equation
    `n . t = b` in the displacement t (n = wall normal); all of them are solved together by gated
    least squares (a 2x2 system, closed form). A new point is only ever paired with a previous
    point of its own kind (wall with wall, door with door) whose wall runs in the same direction
    as the new scan's own wall at that point. A surface that has just come into view (the room
    behind a door that is opening, the far side of a corner) has no such partner and is left out
    instead of being matched to an unrelated wall.

    Where the walls in view constrain one direction only (a long corridor), the other component
    comes from a model of the engine's wall slide: for each tic at which the body could first have
    touched the wall, the model is fitted to the measured approach and to the reported momentum,
    and the tic that explains both is used. If no tic does, the momentum lost along the wall is
    spread over the step (one impulse if the measured direction can time it, else evenly).
    A single ray that does see along the corridor (its far end) is used instead of these models
    when what it measures lies between "did not move" and the free prediction.
    A monster standing in the way in such a corridor shows in the rays that end on it: the ray
    that comes closest gives its centre and so the room in front of it. A blocked tic leaves no
    momentum, so the reported momentum tells how many free tics followed the block; before the
    block the body used up at most that room, and no model may carry it further. If the gap to
    the monster closed by clearly less than that, the monster backed away and the body is taken
    to have followed it. How far a monster moves within one step is otherwise unknown, and is
    the main source of the drift that is left.

Position, wedged steps
    A body that runs squarely into the end of a wall can be stopped without any change of
    momentum. A step whose prediction ends with a wall point inside the body's reach is therefore
    checked against the rays as well, and measured if they disagree.

There is no search and no inverse trigonometry here: correspondences are nearest neighbours
between two 64-point scans, and everything else is closed-form arithmetic.

Failures
    `update` does not raise on anything an observation can hold. The heading and the free-motion
    prediction need only the buttons. Everything that reads the observation runs under one guard:
    if it raises, or a measured displacement is not a finite number, the step keeps the model's
    answer (the prediction, corrected by the reported momentum if that was readable), the stored
    scan is dropped and the step is counted in `Odometry.errors`. The counter is 0 on every map
    the tests run, so a real fault shows there instead of clearing the agent's state mid-map.
    `errors` and `last_error` are for the current map and start again at `reset()`.
    `errors_total` and `last_error_total` are for the whole life of the object and are what a
    training log or an evaluation report should read: they need no bookkeeping around `reset()`.
    So that a failure is seen even where nobody reads those fields, the 1st, 10th, 100th, ...
    failure of the process (`Odometry.process_errors`, all objects together) is printed to
    stderr as one line.

Purpose: turn (previous action, observation) into the pose (x, y, theta) in the start frame.
Shapes: prev_action (9,), obs["vitals"] (17,), obs["rays"]["distance"] (64,), obs["rays"]["hit_type"] (64,) -> x, y, theta floats.
Why: the heading follows exactly from the buttons, free motion from the known friction model, and the rays measure the few steps the model cannot predict, so drift stays at about one unit per episode on grid maps.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from . import spec

# ---- engine constants (measured) ---------------------------------------------------------------------
FRICTION = 0.90625            # momentum multiplier after every tic's move
THRUST_FWD = 0.78125          # momentum added per tic by forward / backward
THRUST_SIDE = 0.75            # momentum added per tic by strafe
DEAD_TICS = 7                 # tics at the start of an episode that ignore input
STOP_SPEED = 0.0625           # with no move key, momentum below this on both axes is set to zero
HEADINGS = 1024               # heading units per circle: one unit = 64/65536 of a circle
TURN_SLOW, TURN_FAST = 5, 10  # heading units per tic (320 and 640 of 65536 per circle)
TURN_RAMP_TICS = 6            # the fast rate applies from the sixth consecutive tic of turning
RAY_STEP = HEADINGS // spec.NUM_RAYS          # heading units between neighbouring rays (16)
SLIDE_FUDGE = 1.0 / 32.0      # the engine stops this share of a blocked move short of the wall
STOP_KEEP = 0.95              # share of a move that survives when a slide is cut short (measured: 0.94 to 0.97)

# ---- tolerances ------------------------------------------------------------------------------------
FREE_TOL = 0.02               # units/tic: a momentum mismatch below this is a free (undisturbed) step
MAX_SPEED = 128.0             # units/tic: reported velocities beyond this are treated as corrupt
CHECK_RADIUS = 24.0           # a free step that ends with a wall point this close to the centre is checked
                              # (the body is a 32-unit square: a wall point within 16 * sqrt(2) can block it)
MAX_OUTLIERS = 0.15           # a checked step is confirmed when at most this share of its rays disagree
CONFIRM_TOL = 0.25            # units: a measurement this close to the free prediction keeps the prediction
MIN_WALL_RANGE = 15.0         # units: a solid wall is never nearer than the body's half-width (16); a
                              # nearer "wall" is a ledge the player is stepping over, and is ignored
COLLINEAR_TOL = 0.02          # units: a wall point may be this far off the chord of its two neighbours
PAIR_MIN_FACING = 0.5         # two neighbouring rays form a wall only if it faces the viewer this much (cosine)
PAIR_MIN_COS = 0.9            # a new wall point pairs with a previous one only if their wall directions agree this well
GATES = (24.0, 6.0, 0.75)     # units: residual gates of the three least-squares passes
GATE_SLACK = 3.0              # units added to the physical bound on the first gate
MIN_SUPPORT = 2.0             # a direction is measured when the rays' squared normals along it sum to this
WEAK_SUPPORT = 0.9            # ... and one ray's worth is enough if what it says is physically possible:
WEAK_SLACK = 1.0              # between "did not move" and the free prediction, give or take this many units
MIN_INLIERS = 0.85            # share of matched rays that must agree before one start point is trusted
SLIDE_MOMENTUM_TOL = 0.05     # units/tic and units: scales of the mismatch between the wall-slide model
SLIDE_APPROACH_TOL = 0.02     # and the reported momentum / the measured approach to the wall
SLIDE_ACCEPT = 40.0           # the model is used when the squared, scaled mismatch is below this
STOP_PENALTY = 4.5            # ... and a cut-short slide must fit this many times better than a plain one
IMPULSE_MIN = 0.5             # units/tic: smallest measured momentum change that can time an impulse
MAIN_WALL_COS = 0.985         # rays within 10 degrees of the main wall's normal define that wall
BODY_HALF = 16.0              # units: half-width of the player's body
MONSTER_HALF = 20.0           # units: half-width assumed for a monster (most are 20; the widest common one is 30)
BODY_NEAR = 24.0              # units: a monster with less room than this in front of it is "in the way"
BODY_SLACK = 1.0              # units: a blocked body stops a little short of the monster
BODY_RETREAT = 3.0            # units: the gap to a monster closing this much less than the body advanced means
BODY_RETREAT_SQUARE = 1.0     # it backed away; less when the monster stands squarely ahead in both scans
BODY_SQUARE_COS = 0.98        # ... that is, within about one ray of the direction of travel
BODY_MATCH = 0.05             # units/tic: how well the reported momentum must fit a "stopped by a monster" history

_F = FRICTION
# Displacement caused by a constant force that changed the final momentum by one unit, for a step
# with 4 live tics: (4 + 3F + 2F^2 + F^3) / (F + F^2 + F^3 + F^4).
FORCE_GAIN = (4 + 3 * _F + 2 * _F**2 + _F**3) / (_F + _F**2 + _F**3 + _F**4)

_ANGLES = np.arange(HEADINGS) * (2.0 * np.pi / HEADINGS)
COS = np.cos(_ANGLES)                         # (1024,) float64, heading unit -> cosine
SIN = np.sin(_ANGLES)
_COS_L, _SIN_L = COS.tolist(), SIN.tolist()   # Python floats for the scalar path
RAY_OFFSETS = np.arange(spec.NUM_RAYS) * RAY_STEP          # heading units of ray k relative to the heading
_NEXT = (np.arange(spec.NUM_RAYS) + 1) % spec.NUM_RAYS
_PREV = (np.arange(spec.NUM_RAYS) - 1) % spec.NUM_RAYS
_THIS_FILE = (lambda: 0).__code__.co_filename               # as tracebacks name this file


def heading_index(theta: float) -> int:
    """Heading in radians -> nearest heading unit in [0, 1024)."""
    return int(round(theta * (HEADINGS / (2.0 * math.pi)))) % HEADINGS


def ray_ranges(distance) -> np.ndarray:
    """Ray distances of an observation in map units, float64 (64,); values that are not finite read as "nothing hit"."""
    r = np.nan_to_num(np.asarray(distance, np.float64), nan=1.0, posinf=1.0, neginf=0.0)
    return np.clip(r, 0.0, 1.0, out=r) * spec.RAY_RANGE


class Odometry:
    """Pose in the start frame (x forward, y left, theta counter-clockwise, all zero at the first observation).

    Call `reset()` before every map and `update(prev_action, obs)` once per decision step with the
    action that produced `obs` (not for the first observation of an episode, which has no action).
    """

    process_errors = 0                # failed measurements of every object of this process (kept on the class)
    ANNOUNCE_AT = (1, 10, 100, 1000, 10_000, 100_000, 1_000_000)        # which of them are printed to stderr

    def __init__(self):
        self.errors_total = 0         # failed measurements since this object was built: `reset()` keeps it
        self.last_error_total = ""    # the last of them, with the step of its map: "step 212: TypeError at line ..."
        self.reset()

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0              # radians in (-pi, pi]
        self.step = 0                 # decision steps integrated so far
        self.heading = 0              # integer heading, units of 2*pi/1024
        self._tic = 0
        self._turn_held = 0
        self._mx = 0.0                # momentum in the start frame, units per tic
        self._my = 0.0
        self._scan = None             # previous scan, see _make_scan
        self.disturbed_steps = 0      # steps whose displacement was measured instead of predicted
        self.last_mode = "free"       # how the last step's displacement was obtained (for tests and logs):
                                      # free | rays | rays+slide | rays+body | rays+model | model | error
        self.errors = 0               # steps of this map whose measurement failed and kept the model's answer (0 in
        self.last_error = ""          # normal use), and what the last such failure was; totals: see __init__

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta

    # ------------------------------------------------------------------------------------------
    def update(self, prev_action, obs) -> None:
        """Advance the pose by one decision step (4 tics) of `prev_action`, then correct it from `obs`."""
        a = prev_action
        if a is None:
            fwd = side = turn = 0
            turning = False
        else:
            fwd = int(a[0]) - int(a[1])
            side = int(a[3]) - int(a[2])
            left, right = int(a[4]), int(a[5])
            turn = left - right
            turning = bool(left or right)

        # ---- free-motion model: four tics of turn, thrust, move, friction -------------------------
        h, tic, held = self.heading, self._tic, self._turn_held
        m0x, m0y = self._mx, self._my
        mx, my = m0x, m0y
        dx = dy = 0.0
        ax, ay = [0.0] * 4, [0.0] * 4          # thrust of each tic, kept for the wall-slide model
        tfwd, tside = THRUST_FWD * fwd, THRUST_SIDE * side
        moving = bool(fwd or side)
        for i in range(spec.FRAME_SKIP):
            tic += 1
            held = held + 1 if turning else 0
            live = tic > DEAD_TICS
            if live:
                if turn:
                    h = (h + (TURN_SLOW if held < TURN_RAMP_TICS else TURN_FAST) * turn) % HEADINGS
                if moving:
                    c, s = _COS_L[h], _SIN_L[h]
                    ax[i] = tfwd * c + tside * s
                    ay[i] = tfwd * s - tside * c
                    mx += ax[i]
                    my += ay[i]
            dx += mx
            dy += my
            if -STOP_SPEED < mx < STOP_SPEED and -STOP_SPEED < my < STOP_SPEED and not (live and moving):
                mx = my = 0.0
            else:
                mx *= FRICTION
                my *= FRICTION
        self.heading, self._tic, self._turn_held = h, tic, held
        self.theta = (h if h <= HEADINGS // 2 else h - HEADINGS) * (2.0 * math.pi / HEADINGS)
        self.step += 1
        self.last_mode = "free"

        # ---- correction from the observation ------------------------------------------------------
        # (tx, ty), (ux, uy) and `new` are the step's displacement, the momentum after it and the scan
        # to keep. They start as the pure prediction and are replaced as the observation is read, so
        # whatever fails below leaves the best answer reached so far in them.
        tx, ty, ux, uy, new = dx, dy, mx, my, None
        try:
            vitals = obs["vitals"]
            vf, vr = float(vitals[14]), float(vitals[15])
            if abs(vf) <= MAX_SPEED and abs(vr) <= MAX_SPEED:           # else NaN, inf or absurd: keep the prediction
                c, s = _COS_L[h], _SIN_L[h]
                ox, oy = vf * c + vr * s, vf * s - vr * c               # reported momentum in the start frame
                rx, ry = ox - mx, oy - my                               # mismatch with the prediction
                # The model's answer given the reported momentum: the mismatch acted as a constant
                # force over the step. On a free step that is the small remainder of thrust the model
                # placed a hair off (the engine rounds the thrust direction to 1/8192 of a circle).
                tx, ty, ux, uy = dx + FORCE_GAIN * rx, dy + FORCE_GAIN * ry, ox, oy
                rays = obs["rays"]
                scan, fresh = self._scan, _make_scan(ray_ranges(rays["distance"]), np.asarray(rays["hit_type"]), h)
                if rx * rx + ry * ry < FREE_TOL * FREE_TOL:
                    # Free step. Wedged? Only possible if it ends with a wall of the previous scan within reach.
                    if scan is not None and scan[3] < CHECK_RADIUS + abs(tx) + abs(ty):
                        ex, ey = scan[4] - tx, scan[5] - ty
                        if float((ex * ex + ey * ey).min()) < CHECK_RADIUS * CHECK_RADIUS:
                            match = _ScanMatch(scan, fresh)
                            if not match.empty and match.outliers(tx, ty) > MAX_OUTLIERS * match.count:
                                sx, sy = self._measure(match, tx, ty, 0.0, 0.0, m0x, m0y, ax, ay, ox, oy, True)
                                if not math.isfinite(sx + sy):
                                    raise FloatingPointError("measured displacement is not finite")
                                if (sx - tx) ** 2 + (sy - ty) ** 2 < CONFIRM_TOL * CONFIRM_TOL:
                                    self.last_mode = "free"
                                else:
                                    tx, ty = sx, sy
                                    self.disturbed_steps += 1
                elif scan is None:
                    self.last_mode = "model"
                    self.disturbed_steps += 1
                else:
                    sx, sy = self._measure(_ScanMatch(scan, fresh), dx, dy, rx, ry, m0x, m0y, ax, ay, ox, oy, False)
                    if not math.isfinite(sx + sy):
                        raise FloatingPointError("measured displacement is not finite")
                    tx, ty = sx, sy
                    self.disturbed_steps += 1
                new = fresh
        except Exception as exc:  # noqa: BLE001 - last line of defence: the pose must survive any observation
            self.errors += 1
            self.errors_total += 1
            Odometry.process_errors += 1
            self.last_mode = "error"
            new = None
            try:                                                # the note: writing it cannot cost the step
                tb, line = exc.__traceback__, 0
                for _ in range(8):                              # the deepest line of this file on the way to the fault
                    if tb is not None:                          # (at most five of its frames lie between here and there)
                        if tb.tb_frame.f_code.co_filename == _THIS_FILE:
                            line = tb.tb_lineno
                        tb = tb.tb_next
                self.last_error = f"{type(exc).__name__} at line {line}: {exc}"
            except Exception:  # noqa: BLE001 - an exception that cannot be described keeps its name
                self.last_error = type(exc).__name__
            self.last_error_total = f"step {self.step}: {self.last_error}"
            if Odometry.process_errors in Odometry.ANNOUNCE_AT:
                try:
                    print(f"[doomnet] odometry: measurement failure {Odometry.process_errors} of this process, at step {self.step} "
                          f"of its map: {self.last_error}. The step kept the motion model's answer and the map goes on; failures "
                          "are counted in Odometry.errors_total and printed at the 1st, 10th, 100th, ...", file=sys.stderr, flush=True)
                except Exception:  # noqa: BLE001 - a log line that cannot be written changes nothing
                    pass
        self.x += tx
        self.y += ty
        self._mx, self._my = ux, uy
        self._scan = new

    # ------------------------------------------------------------------------------------------
    def _measure(self, match, dx, dy, rx, ry, m0x, m0y, ax, ay, ox, oy, momentum_ok):
        """Displacement of a step measured from the rays, with models for what the rays cannot see.

        (dx, dy) is the free-motion prediction, (rx, ry) the momentum mismatch; momentum_ok says
        the momentum matched and the step is here because the rays disagreed. Returns (tx, ty).
        """
        # Prior: the disturbance acted as a constant force over the step (exact for steady sliding
        # along a wall and for pushing into one).
        px, py = dx + FORCE_GAIN * rx, dy + FORCE_GAIN * ry
        if match.empty:
            self.last_mode = "model"
            return px, py
        # No step can end further from the free prediction than its own length plus what an impulse
        # of the size of the mismatch can add: that bounds the first, widest gate.
        reach = math.hypot(dx, dy) + math.hypot(rx, ry) / (1.0 - FRICTION) + GATE_SLACK
        # The true displacement lies somewhere between "did not move" and the free prediction: the
        # prior is tried first, the two ends only if too few rays agree with it.
        sol = match.solve(px, py, reach)
        if sol[0] < MIN_INLIERS * match.count or (not momentum_ok and _weak_support(sol) < MIN_SUPPORT):
            for sx, sy in ((0.0, 0.0),) if momentum_ok else ((0.0, 0.0), (dx, dy)):
                other = match.solve(sx, sy, reach)
                if other[0] > sol[0] + 0.5:
                    sol = other
        _, a11, a12, a22, g1, g2, njx, njy, b, w = sol
        # Eigen-decomposition of the 2x2 normal matrix: l1 >= l2, e1 the best-measured direction.
        half_tr = 0.5 * (a11 + a22)
        det = a11 * a22 - a12 * a12
        disc = math.sqrt(max(half_tr * half_tr - det, 0.0))
        l1, l2 = half_tr + disc, half_tr - disc
        if l2 >= MIN_SUPPORT:
            self.last_mode = "rays"
            return (a22 * g1 - a12 * g2) / det, (a11 * g2 - a12 * g1) / det
        if l1 < MIN_SUPPORT:
            self.last_mode = "model"
            return px, py
        # Walls in view constrain one direction only (e1): take that component from the rays.
        if abs(a12) > 1e-9:
            e1x, e1y = l1 - a22, a12
        elif a11 >= a22:
            e1x, e1y = 1.0, 0.0
        else:
            e1x, e1y = 0.0, 1.0
        norm = math.hypot(e1x, e1y)
        e1x, e1y = e1x / norm, e1y / norm
        if l2 >= WEAK_SUPPORT:
            # One ray or so sees along the other direction (the far end of a corridor). It is used
            # if what it says is physically possible: nothing here pushes the body, so along that
            # direction it moved between "not at all" and the free prediction.
            tx, ty = (a22 * g1 - a12 * g2) / det, (a11 * g2 - a12 * g1) / det
            t2, f2 = e1x * ty - e1y * tx, e1x * dy - e1y * dx
            if min(f2, 0.0) - WEAK_SLACK <= t2 <= max(f2, 0.0) + WEAK_SLACK:
                self.last_mode = "rays"
                return tx, ty
        # A stray ray on some other wall tilts that eigenvector a little, which matters to the slide
        # model below: rebuild e1 and the measured component from the rays of the main wall alone.
        along = njx * e1x + njy * e1y
        w = w * (np.abs(along) > MAIN_WALL_COS) * np.sign(along)
        count = float(np.abs(w).sum())
        if count >= MIN_SUPPORT:
            e1x, e1y = float(w @ njx), float(w @ njy)
            norm = math.hypot(e1x, e1y)
            e1x, e1y = e1x / norm, e1y / norm
            t1 = float(w @ b) / count
        else:
            t1 = (e1x * g1 + e1y * g2) / l1
        e2x, e2y = -e1y, e1x
        r1, r2 = e1x * rx + e1y * ry, e2x * rx + e2y * ry
        free2, o2 = e2x * dx + e2y * dy, e2x * ox + e2y * oy
        if abs(r2) > BODY_MATCH:
            sign = 1.0 if free2 > 0.0 else -1.0
            room_now, square_now = _body_room(*match.new, sign * e2x, sign * e2y)
            if room_now < BODY_NEAR:
                # A monster right ahead along e2: it may have been in the way for part of the step.
                scan = self._scan
                room, square = _body_room(scan[0], scan[8], scan[2], sign * e2x, sign * e2y)
                t2 = _body_block(sign * (e2x * m0x + e2y * m0y), [sign * (e2x * ax[i] + e2y * ay[i]) for i in range(4)],
                                 sign * o2, room - BODY_SLACK, room - room_now,
                                 BODY_RETREAT_SQUARE if square and square_now else BODY_RETREAT)
                if t2 is not None:
                    self.last_mode = "rays+body"
                    return e1x * t1 + e2x * sign * t2, e1y * t1 + e2y * sign * t2
            elif abs(o2) < BODY_MATCH and abs(e2x * m0x + e2y * m0y) < BODY_MATCH:
                # No speed along e2 before or after the step although the thrust should have built
                # some, and nothing in view to give way: the body was held in place for the whole step.
                self.last_mode = "rays+body"
                return e1x * t1, e1y * t1
        # Otherwise: the wall-slide model if it explains the step, else an impulse, else the prior.
        free1 = e1x * dx + e1y * dy
        over = free1 - t1                                       # how far the free motion overshot along e1
        t2, cost, plain = None, SLIDE_ACCEPT, False
        # The wall is on the side the free motion overshot; when the overshoot is tiny, try both sides.
        for sign in ((1.0, -1.0) if abs(over) < 0.5 else ((1.0,) if over > 0.0 else (-1.0,))):
            along, c, stopped = _wall_slide(sign * (e1x * m0x + e1y * m0y), sign * (e2x * m0x + e2y * m0y),
                                            [sign * (e1x * ax[i] + e1y * ay[i]) for i in range(4)],
                                            [sign * (e2x * ax[i] + e2y * ay[i]) for i in range(4)],
                                            sign * t1, sign * (e1x * ox + e1y * oy), sign * (e2x * ox + e2y * oy))
            if c < cost:
                t2, cost, plain = sign * along, c, not stopped
        if t2 is not None:
            self.last_mode = "rays+slide"
            if plain:                                           # the slide explains all of the momentum:
                return e1x * t1 + e2x * t2, e1y * t1 + e2y * t2   # nothing else was in the way
        else:
            self.last_mode = "rays+model"
            gain = -over / r1 if abs(r1) > IMPULSE_MIN else -1.0
            if 0.0 <= gain <= 1.0 / (1.0 - FRICTION):
                t2 = free2 + gain * r2                          # one impulse at one moment of the step
            else:
                t2 = _even_loss(e2x * m0x + e2y * m0y, [e2x * ax[i] + e2y * ay[i] for i in range(4)], o2)
                if t2 is None:
                    t2 = free2 + FORCE_GAIN * r2                # a constant force over the step
        # The momentum was lost to something other than a plain slide. Whatever the model says, the
        # body cannot have gone through a monster that the previous scan saw in the way: the room
        # in front of that monster bounds the step.
        sign = 1.0 if t2 > 0.0 else -1.0
        scan = self._scan
        room = max(_body_room(scan[0], scan[8], scan[2], sign * e2x, sign * e2y)[0] - BODY_SLACK, 0.0)
        if sign * t2 > room:
            t2 = sign * room
        return e1x * t1 + e2x * t2, e1y * t1 + e2y * t2


def _weak_support(sol) -> float:
    """Smaller eigenvalue of the normal matrix of a solve: how well the worst direction is measured."""
    a11, a12, a22 = sol[1:4]
    half_tr = 0.5 * (a11 + a22)
    return half_tr - math.sqrt(max(half_tr * half_tr - (a11 * a22 - a12 * a12), 0.0))


# ---- wall-slide model ------------------------------------------------------------------------------
def _slide_run(mn, mt, an, at, j, f, stop=-1):
    """Four tics in a wall's frame (n into the wall, t along it) with first contact at tic j.

    (mn, mt): momentum at the start; an, at: thrust of each tic; f: share of tic j's move made
    before touching. Engine rule for a blocked move w: the body stops SLIDE_FUDGE of the move short
    of the wall, keeps the part of the remaining (1 - f) that runs along the wall, and its momentum
    becomes that remainder.

    stop: tic at which the slide itself is cut short by the end of another wall across the path
    (-1: never). The engine then still lets the move along the wall through, but the momentum
    along the wall is lost and the momentum into the wall is kept instead.

    Returns (dn, dt, un, ut, wn_j): displacement, final momentum and the normal move of tic j, or
    None if the body is not pressed against the wall at tic j or at the stop tic.
    """
    un, ut, dn, dt, gap, wn_j = mn, mt, 0.0, 0.0, 0.0, 0.0
    for i in range(4):
        wn, wt = un + an[i], ut + at[i]
        if i < j or (i > j and wn <= gap):                      # free tic (before contact, or inside the gap)
            if i == stop:
                return None
            dn += wn
            dt += wt
            if i > j:
                gap -= wn
            un, ut = FRICTION * wn, FRICTION * wt
            continue
        if wn <= 0.0:
            return None
        if i == j:
            wn_j, fi = wn, f
        else:
            fi = gap / wn
        if i == stop:
            dt += STOP_KEEP * wt
            un, ut = STOP_KEEP * FRICTION * wn, 0.0
            continue
        if fi > SLIDE_FUDGE:
            dn += (fi - SLIDE_FUDGE) * wn
            dt += (1.0 - SLIDE_FUDGE) * wt
            gap = SLIDE_FUDGE * wn
        else:
            dt += (1.0 - fi) * wt
            gap = fi * wn
        un, ut = 0.0, (1.0 - fi) * FRICTION * wt
    return dn, dt, un, ut, wn_j


def _wall_slide(mn, mt, an, at, dn_seen, on, ot):
    """Displacement along a wall during a step that ran into it: (dt, cost, stopped), dt None if nothing fits.

    Wall frame as in `_slide_run`; dn_seen is the displacement towards the wall measured from the
    rays, (on, ot) the momentum reported after the step. Every first-contact tic is tried, with
    and without a later stop. The contact share f is taken once from dn_seen and once from the
    reported momentum along the wall (the second is sharper when the body drifts into the wall
    slowly); the candidate that best reproduces both the measured approach and the reported
    momentum wins. A candidate for which `_slide_run` reports "not pressed" is not a possible
    history of the step and is skipped.
    """
    best, best_cost, stopped = None, SLIDE_ACCEPT, False
    for j in range(4):
        for stop in (-1, *range(j, 4)):
            base = _slide_run(mn, mt, an, at, j, SLIDE_FUDGE, stop)
            if base is None:
                continue
            dn0, _, _, ut0, wn_j = base
            if stop == j:                                       # the share plays no part in this case
                shares = [SLIDE_FUDGE]
            else:
                shares = []
                f = SLIDE_FUDGE + (dn_seen - dn0) / wn_j        # dn is linear in f above the fudge share
                if -0.1 <= f <= 1.1:
                    shares.append(min(max(f, 0.0), 1.0))
                half = _slide_run(mn, mt, an, at, j, 0.5) if stop < 0 else None
                if half is not None:
                    slope = half[3] - ut0                       # so is the final momentum along the wall
                    if abs(slope) > 1e-9:
                        f = SLIDE_FUDGE + (ot - ut0) * (0.5 - SLIDE_FUDGE) / slope
                        if -0.02 <= f <= 1.02:
                            shares.append(min(max(f, 0.0), 1.0))
            for f in shares:
                run = _slide_run(mn, mt, an, at, j, f, stop)
                if run is None:                                 # a share below the fudge share leaves a smaller gap, and
                    continue                                    # the body is then not pressed at the stop tic: no such step
                dn, dt, un, ut, _ = run
                cost = ((un - on) ** 2 + (ut - ot) ** 2) / SLIDE_MOMENTUM_TOL ** 2 + ((dn - dn_seen) / SLIDE_APPROACH_TOL) ** 2
                if stop >= 0:
                    cost *= STOP_PENALTY
                if cost < best_cost:
                    best, best_cost, stopped = dt, cost, stop >= 0
    return best, best_cost, stopped


def _body_block(mt, at, ot, room, closed, retreat):
    """Displacement along a corridor in a step during which a monster in the way stopped the body.

    mt: momentum towards the monster at the start, at: thrust of each tic, ot: momentum reported
    after the step, room: free distance to the monster before the step, closed: by how much that
    distance shrank during the step, retreat: how much less than the body's own advance it must
    have shrunk before the monster is taken to have backed away. A blocked tic leaves the body where it is with no momentum
    in that direction, so the reported momentum tells how many free tics followed the last
    blocked one, and the displacement of those tics is known exactly. Before the last blocked
    tic the body moved freely until it had used up the room in front of the monster (none at all
    if it stood against it already). If the distance shrank by clearly less than that, the
    monster backed away and the body followed it up to the last blocked tic. Returns None if the
    reported momentum fits no such history.
    """
    for last in range(3, -1, -1):                               # last blocked tic
        u = after = 0.0
        for i in range(last + 1, 4):
            w = u + at[i]
            after += w
            u = FRICTION * w
        if abs(u - ot) < BODY_MATCH:
            u = mt
            free = 0.0
            for i in range(last):
                w = u + at[i]
                free += w
                u = FRICTION * w
            free = max(free, 0.0)
            before = min(free, max(room, 0.0))
            if closed < before + after - retreat:
                before = free
            return before + after
    return None


def _even_loss(mt, at, ot):
    """Displacement along a wall when the momentum along it shrank by the same factor at every tic.

    mt: momentum along the wall at the start, at: thrust of each tic, ot: momentum reported after
    the step. Used when the body brushed a wall more than once in a step: the losses are then
    spread over the step rather than concentrated at one tic. Returns None if the reported
    momentum is not a shrunken version of the free one.
    """
    def run(g):
        u, d = mt, 0.0
        for a in at:
            w = u + a
            d += w
            u = g * FRICTION * w
        return u, d

    free = run(1.0)[0]
    if ot * free < 0.0 or abs(ot) > abs(free) or abs(free) < 1e-6:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(16):                                         # bisection on the factor
        g = 0.5 * (lo + hi)
        if abs(run(g)[0]) < abs(ot):
            lo = g
        else:
            hi = g
    return run(0.5 * (lo + hi))[1]


# ---- scans -------------------------------------------------------------------------------------------
def _make_scan(r, hit, h):
    """Keep what the next step needs of a scan with ranges r (map units) and hit types.

    Returns [range, static, heading, nearest wall range, static wall x, static wall y, wall x, wall y, hit,
    normals]. Wall points are relative to the scan's own origin, in the start-frame orientation.
    The last entry is filled in by `_scan_normals` the first time a match needs it.
    """
    static = (hit >= spec.HIT_WALL) & (hit <= spec.HIT_LOCKED) & (r >= MIN_WALL_RANGE)
    rows = (RAY_OFFSETS + h) % HEADINGS
    wx, wy = r * COS[rows], r * SIN[rows]
    sx, sy = wx[static], wy[static]
    near = float(r[static].min()) if len(sx) else float("inf")
    return [r, static, h, near, sx, sy, wx, wy, hit, None]


def _scan_normals(scan):
    """(nx, ny, known) of a scan's wall points, computed once per scan and kept in the scan."""
    if scan[9] is None:
        scan[9] = wall_normals(scan[6], scan[7], scan[1])
    return scan[9]


def _body_room(r, hit, h, ux, uy) -> tuple[float, bool]:
    """Room in front of the nearest monster in the way: how far the body can move along the unit
    direction (ux, uy) before the two bodies touch (inf if no monster of the scan is in the way),
    and whether that monster stands squarely ahead (the room is then known to a fraction of a unit).

    The ray that comes closest to a monster points at its centre, MONSTER_HALF behind the hit.
    The engine blocks a move when the centres are closer than the sum of the half-widths on both
    axes, so a monster is in the way if its centre lies within that sum to the side of the path.
    """
    e = np.where(hit == spec.HIT_ENEMY, r, np.inf)
    k = np.flatnonzero((e < 6.0 * BODY_NEAR) & (e <= e[_PREV]) & (e <= e[_NEXT]))
    if len(k):
        rows = (RAY_OFFSETS[k] + h) % HEADINGS
        d = r[k] + MONSTER_HALF
        cosine = COS[rows] * ux + SIN[rows] * uy
        ahead = np.flatnonzero((cosine > 0.0) & (np.abs(d * (SIN[rows] * ux - COS[rows] * uy)) < BODY_HALF + MONSTER_HALF))
        if len(ahead):
            j = ahead[np.argmin(d[ahead] * cosine[ahead])]
            return float(d[j] * cosine[j]) - (BODY_HALF + MONSTER_HALF), bool(cosine[j] > BODY_SQUARE_COS)
    return float("inf"), False


def wall_normals(px, py, static, tol=COLLINEAR_TOL, kind=None):
    """Unit normals of the straight walls that the points of one scan lie on.

    px, py: (64,) wall points in ray order, relative to the scan origin; static: (64,) bool, True
    where the ray ended on a wall or door; tol: how far a point may lie off the chord of its two
    neighbours, in the units of px and py. A point gets a normal when it and its two neighbours are
    collinear (the point is then the middle of a straight run of three), when it is the end of
    such a run (it takes the normal of the run's middle ray), or when it and one neighbour form an
    isolated pair that belongs to no such run (a wall seen by exactly two rays, such as the far
    end of a corridor).

    kind: optional (64,) labels of the surfaces. When given, a point takes a normal from a
    neighbour only if both have the same label: a door ray that ends on the corner between the
    door's face and a side wall is collinear with the side wall's points, but it is not part of
    that wall.

    Returns (nx, ny, known): known is False where no wall direction is known. The sign of a normal
    is arbitrary.
    """
    pxp, pyp = px[_PREV], py[_PREV]
    cx, cy = px[_NEXT] - pxp, py[_NEXT] - pyp
    clen = np.sqrt(cx * cx + cy * cy) + 1e-9
    cross = cx * (py - pyp) - cy * (px - pxp)
    mid = static & static[_NEXT] & static[_PREV] & (np.abs(cross) <= tol * clen)
    nx, ny = cy / clen, -cx / clen
    # Ends of a run take the normal of the run's middle ray.
    end_a, end_b = mid[_NEXT] & ~mid, mid[_PREV] & ~mid & ~mid[_NEXT]
    link = static & static[_NEXT]                        # rays k and k + 1 may share a wall
    if kind is not None:
        link &= kind == kind[_NEXT]
        end_a &= link
        end_b = mid[_PREV] & ~mid & ~end_a & link[_PREV]
    nx = np.where(end_a, nx[_NEXT], np.where(end_b, nx[_PREV], nx))
    ny = np.where(end_a, ny[_NEXT], np.where(end_b, ny[_PREV], ny))
    known = mid | end_a | end_b
    # Isolated pairs (k, k+1): the chord between them is the wall, unless it runs nearly along the
    # line of sight (then the two rays ended on different walls, one behind the other).
    pair = link & ~known & ~known[_NEXT]
    if pair.any():
        qx, qy = px[_NEXT] - px, py[_NEXT] - py
        qlen = np.sqrt(qx * qx + qy * qy) + 1e-9
        pnx, pny = qy / qlen, -qx / qlen
        across = pnx * px + pny * py                     # distance of the chord's line from the scan origin
        pair &= across * across >= PAIR_MIN_FACING ** 2 * np.maximum(px * px + py * py, px[_NEXT] ** 2 + py[_NEXT] ** 2)
        second = pair[_PREV] & ~pair
        nx = np.where(pair, pnx, np.where(second, pnx[_PREV], nx))
        ny = np.where(pair, pny, np.where(second, pny[_PREV], ny))
        known = known | pair | second
    return nx, ny, known


class _ScanMatch:
    """Displacement between the previous scan and the new one.

    Every static ray of the new scan is paired with the nearest compatible wall point of the
    previous scan and must lie on that point's wall line: n . (t + q - p) = 0, one linear
    equation in t. Compatible means: the same kind of surface (wall with wall, door with door),
    and, where the new scan shows the direction of its own wall at that ray, the same direction
    within PAIR_MIN_COS. New rays with no compatible partner are surfaces the previous scan did
    not see; they take no part in the match.
    """

    def __init__(self, prev, new):
        nx, ny, known = _scan_normals(prev)
        idx = np.flatnonzero(known)
        self.lx, self.ly, self.nx, self.ny = prev[6][idx], prev[7][idx], nx[idx], ny[idx]     # previous walls
        self.new = (new[0], new[8], new[2])                                                   # the new scan as observed
        keep = np.flatnonzero(new[1])
        mx, my, mknown = _scan_normals(new)
        mx, my = mx[keep], my[keep]
        clash = (new[8][keep] != spec.HIT_WALL)[:, None] != (prev[8][idx] != spec.HIT_WALL)[None, :]
        clash |= mknown[keep][:, None] & (np.abs(mx[:, None] * self.nx[None, :] + my[:, None] * self.ny[None, :]) < PAIR_MIN_COS)
        keep = keep[~clash.all(axis=1)] if clash.size else keep[:0]
        self._far = clash[~clash.all(axis=1)] * 1.0e12 if clash.size else None                # added to the squared distance
        self.qx, self.qy = new[6][keep], new[7][keep]                                         # new wall points
        self.count = len(keep)
        self.empty = len(keep) == 0

    def _pair(self, tx, ty):
        """Residual of every new wall point against its nearest compatible previous wall, at displacement (tx, ty).

        Returns (njx, njy, b, residual) with residual = n . t - b.
        """
        lx, ly, qx, qy = self.lx, self.ly, self.qx, self.qy
        ex = (qx + tx)[:, None] - lx[None, :]
        ey = (qy + ty)[:, None] - ly[None, :]
        j = np.argmin(ex * ex + ey * ey + self._far, axis=1)
        njx, njy = self.nx[j], self.ny[j]
        b = njx * (lx[j] - qx) + njy * (ly[j] - qy)
        return njx, njy, b, njx * tx + njy * ty - b

    def outliers(self, tx, ty) -> int:
        """Number of new wall points that are off their previous wall at displacement (tx, ty)."""
        return int(np.count_nonzero(np.abs(self._pair(tx, ty)[3]) >= GATES[2]))

    def solve(self, tx, ty, reach):
        """Gated least squares from the start point (tx, ty); `reach` caps the first gate.

        Returns (inliers, a11, a12, a22, g1, g2, njx, njy, b, w): the number of rays within the
        last gate, the normal equations A t = g built from them, and per ray the wall normal, the
        right-hand side and the 0/1 weight of that last pass.
        """
        a11 = a12 = a22 = g1 = g2 = 0.0
        njx = njy = b = w = None
        for gate in (min(GATES[0], reach), GATES[1], GATES[2]):
            njx, njy, b, res = self._pair(tx, ty)
            w = (np.abs(res) < gate).astype(np.float64)
            wx, wy = w * njx, w * njy
            a11, a12, a22 = float(wx @ njx), float(wx @ njy), float(wy @ njy)
            g1, g2 = float(wx @ b), float(wy @ b)
            # Ridge towards the current estimate: only well-supported directions move.
            b1, b2 = g1 + tx, g2 + ty
            d11, d22 = a11 + 1.0, a22 + 1.0
            det = d11 * d22 - a12 * a12
            tx, ty = (d22 * b1 - a12 * b2) / det, (d11 * b2 - a12 * b1) / det
        return a11 + a22, a11, a12, a22, g1, g2, njx, njy, b, w
