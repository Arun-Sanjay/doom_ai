"""Training-time scaffolding. Never imported by the agent in submission/.

Steering of the navigation teacher: from the pose estimate, the reported velocity and a point to
reach, the move, strafe and turn choices of one decision step.

The player is a point mass with friction: every tic a held move key adds a fixed thrust, the body
moves by its momentum and the momentum is multiplied by the friction factor. Movement does not
depend on where the player looks, so the teacher steers the body with the four move keys and the
view with the two turn keys, separately:

* body: the wanted push is worked out in the frame of the line to the waypoint (along it and
  across it) and then pressed as the nearest of the eight directions the keys can produce, found
  from dot products with the heading. Along the line the push is "full ahead" unless the body has
  to come to rest at the waypoint; then, and across the line always, a one-step look-ahead on the
  point-mass model decides between push, coast and brake.
* view: the heading error is read from the cross and dot products of the heading and the wanted
  direction, and a turn key is pressed only while one more step of turning would shrink the error.
  A turn adds 7 degrees on its first step and 14 per step later, so this stops a fast turn early
  and finishes with single slow steps instead of swinging past the target.
"""

from __future__ import annotations

import math

from training import paths  # noqa: F401
from doomnet import spec

from .grid import CELL, N

FRICTION = 0.90625            # momentum kept per tic
THRUST = 0.75                 # momentum added per tic by a move key (forward gives 0.78, strafe 0.75)
TICS = spec.FRAME_SKIP        # tics per decision step
DEAD_TICS = 7                 # tics at the start of an episode that ignore the keys
TURN_SLOW, TURN_FAST = 5, 10  # heading units (1/1024 of a circle) per tic, slow for the first 5 tics of a turn
TURN_RAMP_TICS = 6
KEY_SECTOR = 0.38             # a key is pressed when the push has more than this share along its direction (sin 22.5)
SETTLE_DISTANCE = 12.0        # within this of the stop point ...
SETTLE_SPEED = 1.2            # ... and slower than this (units per tic): hands off, friction does the rest.
                              # A key held for one step shifts the place where the body comes to rest by about
                              # 32 units whatever is pressed afterwards, so smaller errors cannot be corrected
                              # at rest; they are corrected on the move, by steering.
STOP_SLACK = 3.0              # allowed overshoot of a stop point when deciding to keep pushing
LINE_BAND = 16.0              # sideways error that is left alone: half of what one key step shifts the body by
DRIFT_SPEED = 2.5             # sideways speed (units per tic) above which a body that is under way is braked
DOOR_BACK = 24.0              # a door is opened from no further back than this from the middle of the cell before it
DOOR_FRONT = 8.0              # ... and no further forward than this: a body that leans on the door while it rises
                              # makes the pose estimate jump (measured: 9 units in one step)
DOOR_REST = CELL / 2 + 12.0   # distance from the door face at which a body placed by the forward ray comes to rest
BODY = 16.5                   # half-width kept free around the body (the body itself is 16): with LINE_BAND this
                              # leaves no sideways error that is neither corrected nor passable in a 64-unit corridor
SAMPLE = 16.0                 # spacing of the samples of a swept segment

NONE, FIRST, SECOND = 0, 1, 2          # head values: move none/forward/back, strafe none/left/right, turn none/left/right


def advance(v: float, push: int) -> tuple[float, float]:
    """One decision step along one axis: (distance covered, velocity afterwards)."""
    d = 0.0
    a = push * THRUST
    for _ in range(TICS):
        v += a
        d += v
        v *= FRICTION
    return d, v


def stop_distance(v: float) -> float:
    """Distance still covered when braking as hard as the keys allow from velocity v (>= 0)."""
    d = 0.0
    while v > 0.0:
        v -= THRUST
        if v <= 0.0:
            break
        d += v
        v *= FRICTION
    return d


def axis_push(d: float, v: float) -> int:
    """Push (-1, 0 or +1) along one axis that brings the body to rest at signed distance d.

    v is the signed velocity along the same axis in units per tic. One step of look-ahead: push
    if the body can still stop in time after another step of pushing, else coast if it can stop
    after a step of coasting, else brake.
    """
    if d < 0.0:
        return -axis_push(-d, -v)
    if d <= SETTLE_DISTANCE and -SETTLE_SPEED <= v <= SETTLE_SPEED:
        return 0
    moved, after = advance(v, 1)
    if stop_distance(after) <= d - moved + STOP_SLACK:
        return 1
    moved, after = advance(v, 0)
    if stop_distance(after) <= d - moved + STOP_SLACK:
        return 0
    return -1


def keys_for(push_x: float, push_y: float, cos_t: float, sin_t: float) -> tuple[int, int]:
    """(move head, strafe head) that push the body along the start-frame vector (push_x, push_y)."""
    norm = math.hypot(push_x, push_y)
    if norm < 1e-9:
        return NONE, NONE
    ahead = (push_x * cos_t + push_y * sin_t) / norm          # dot with the heading
    left = (-push_x * sin_t + push_y * cos_t) / norm          # dot with the heading turned 90 degrees left
    move = FIRST if ahead > KEY_SECTOR else SECOND if ahead < -KEY_SECTOR else NONE
    strafe = FIRST if left > KEY_SECTOR else SECOND if left < -KEY_SECTOR else NONE
    return move, strafe


def drive(x, y, cos_t, sin_t, vx, vy, wx, wy, stop: bool) -> tuple[int, int]:
    """Move and strafe heads that take the body towards the point (wx, wy).

    (x, y) position and (vx, vy) velocity in the start frame. With `stop` the body comes to rest
    at the point; without it the body runs through at full speed. Sideways drift relative to the
    line to the point is always braked.
    """
    dx, dy = wx - x, wy - y
    dist = math.hypot(dx, dy)
    if dist < SETTLE_DISTANCE:
        if not stop:
            return keys_for(dx if dist > 1e-6 else cos_t, dy if dist > 1e-6 else sin_t, cos_t, sin_t)
        return keys_for(float(axis_push(dx, vx)), float(axis_push(dy, vy)), cos_t, sin_t)
    ax, ay = dx / dist, dy / dist
    along = vx * ax + vy * ay
    across = -vx * ay + vy * ax
    pa = axis_push(dist, along) if stop else 1
    pn = 0 if -DRIFT_SPEED <= across <= DRIFT_SPEED else (-1 if across > 0.0 else 1)
    return keys_for(pa * ax - pn * ay, pa * ay + pn * ax, cos_t, sin_t)


def hold_line(x, y, cos_t, sin_t, vx, vy, px, py, ax, ay) -> tuple[int, int]:
    """Move and strafe heads that bring the body onto the line through (px, py) with unit direction
    (ax, ay) and to rest along it: used before a turn into a narrow passage, where advancing off
    the centre line would clip the corner."""
    ex, ey = x - px, y - py
    offset = -ex * ay + ey * ax
    along = vx * ax + vy * ay
    across = -vx * ay + vy * ax
    pa = axis_push(0.0, along)
    pn = axis_push(-offset, across) if abs(offset) > LINE_BAND else axis_push(0.0, across)
    return keys_for(pa * ax - pn * ay, pa * ay + pn * ax, cos_t, sin_t)


def wait_at_door(x, y, cos_t, sin_t, vx, vy, px, py, gx, gy, creep: bool, door_ahead: float | None = None) -> tuple[int, int]:
    """Move and strafe heads for waiting in the cell in front of a door.

    (px, py) is the centre of that cell and (gx, gy) the unit vector from it to the door. The
    body only has to be somewhere the door can be opened from and walked through: within
    LINE_BAND of the door's axis, not further back than DOOR_BACK from the cell centre and not
    further forward than DOOR_FRONT (clear of the door itself).
    Inside that zone it just comes to rest; outside it is brought back in. `creep` pushes it
    towards the door (a pickup lying between the body and the door hides the door from the ray
    that has to reach it; a step forward puts the pickup behind). `door_ahead` is the distance the
    forward ray reads to the door face, given when it disagrees with the pose estimate: a body
    that is inside the zone by its pose estimate is then brought to rest DOOR_REST from the face
    the ray sees.
    """
    ex, ey = x - px, y - py
    along, lateral = ex * gx + ey * gy, -ex * gy + ey * gx
    v_along, v_lateral = vx * gx + vy * gy, -vx * gy + vy * gx
    if creep:
        pa = 1
    elif along < -DOOR_BACK or along > DOOR_FRONT:
        pa = axis_push(-along, v_along)
    elif door_ahead is not None:
        pa = axis_push(door_ahead - DOOR_REST, v_along)
    else:
        pa = axis_push(0.0, v_along)
    pn = axis_push(-lateral, v_lateral) if abs(lateral) > LINE_BAND else axis_push(0.0, v_lateral)
    return keys_for(pa * gx - pn * gy, pa * gy + pn * gx, cos_t, sin_t)


def turn_increment(tic: int, held: int) -> int:
    """Heading units one more step of turning would add, `tic` tics into the episode with a turn
    key held for the last `held` tics."""
    total = 0
    for _ in range(TICS):
        tic += 1
        held += 1
        if tic > DEAD_TICS:
            total += TURN_SLOW if held < TURN_RAMP_TICS else TURN_FAST
    return total


def turn_head(cos_t, sin_t, dx, dy, tic: int, held: int) -> int:
    """Turn head (0 none, 1 left, 2 right) that swings the heading towards the direction (dx, dy).

    cross > 0 means the direction is to the left. A key is pressed only if the heading error is
    more than half of what one more step of turning adds, so the error never grows.
    """
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return NONE
    cross = (cos_t * dy - sin_t * dx) / norm
    dot = (cos_t * dx + sin_t * dy) / norm
    half = math.sin(max(turn_increment(tic, held), TURN_SLOW * TICS) * math.pi / 1024.0) + 1e-4
    if dot > 0.0 and -half <= cross <= half:                  # (an error of exactly half a step would swing forever)
        return NONE
    return FIRST if cross > 0.0 else SECOND


def facing(cos_t, sin_t, dx, dy, sine_limit: float) -> bool:
    """True when the heading is within asin(sine_limit) of the direction (dx, dy)."""
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return True
    return cos_t * dx + sin_t * dy > 0.0 and abs(cos_t * dy - sin_t * dx) <= sine_limit * norm


def swept_clear(plain: list, ox: int, oy: int, x, y, wx, wy, allow) -> bool:
    """Can the body slide from (x, y) straight to (wx, wy) over plain floor only?

    `plain` is the flat cell list of a planning grid whose index (0, 0) is start-frame cell
    (ox, oy). The body is taken as a square of half-width BODY and tested every SAMPLE units;
    cells in `allow` (the route being followed) count as clear whatever the list says.
    """
    dx, dy = wx - x, wy - y
    dist = math.hypot(dx, dy)
    n = int(dist / SAMPLE) + 1
    half = CELL / 2
    for k in range(1, n + 1):
        t = k / n
        px, py = x + dx * t, y + dy * t
        i0 = int((px - BODY + half) // CELL) - ox
        i1 = int((px + BODY + half) // CELL) - ox
        j0 = int((py - BODY + half) // CELL) - oy
        j1 = int((py + BODY + half) // CELL) - oy
        if i0 < 0 or j0 < 0 or i1 >= N or j1 >= N:
            return False
        for c in (i0 * N + j0, i0 * N + j1, i1 * N + j0, i1 * N + j1):
            if not plain[c] and c not in allow:
                return False
    return True
