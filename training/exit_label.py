"""Exit observations computed from ground truth, used only while training.

Training-time scaffolding. Never imported by the agent in submission/.

The exit of a grid layout is a floor pad that the range sensors cannot see; at evaluation the agent
finds it with a small CNN on the RGB frame (doomnet.model.ExitDetector). During training the same
signal is produced here from the true pose and map geometry, with noise, so the policy can be
trained without rendering through the detector on every step. The detector itself is trained on
frames labelled by `exit_observation`, so both sides share one definition of "the exit is visible".
"""

from __future__ import annotations

import math

import numpy as np
from doom_env.sensors import _segments_blocked

MAX_RANGE = 640.0                      # beyond this the pad is under about one pixel row tall
AMBIGUOUS_RANGE = 900.0
TAN_VISIBLE = math.tan(math.radians(42.0))   # pad centre must be inside the 90 degree view with a margin
TAN_FRAME = math.tan(math.radians(46.0))     # anything inside this may put pad pixels in the frame
MIN_RANGE = 16.0


def _ego(px, py, heading, pts):
    """World points (N, 2) -> forward and left coordinates in the player's frame."""
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = pts[:, 0] - px, pts[:, 1] - py
    return dx * c + dy * s, -dx * s + dy * c


def _clear(env, px, py, pts):
    """Line of sight from the player to each point, against the lines that block sight right now."""
    geo = env._geometry
    idx = np.flatnonzero(geo.sight_block)
    blocked = _segments_blocked(np.array([px, py], np.float64), np.asarray(pts, np.float64), geo.p1[idx], geo.p2[idx])
    return ~blocked


def exit_observation(env, meta) -> tuple[bool, float, float, bool]:
    """(visible, x_forward, y_left, ambiguous) for the nearest visible exit pad.

    visible: the pad centre is inside the field of view (with a margin), within MAX_RANGE and in
    line of sight. ambiguous: not visible by that rule, but part of a pad may still show in the
    frame (a corner in view, or the centre in view between MAX_RANGE and AMBIGUOUS_RANGE); detector
    training ignores such frames, policy training treats them as not visible.
    """
    pads = [e for e in meta.exits if e.sector >= 0]
    if not pads:
        return False, 0.0, 0.0, False
    st = env.privileged_state()
    px, py, heading = st["x"], st["y"], math.radians(st["angle"])
    centres = np.array([e.centre for e in pads], np.float64).reshape(-1, 2)
    fwd, left = _ego(px, py, heading, centres)
    dist = np.hypot(fwd, left)
    in_view = (fwd > 0) & (np.abs(left) <= fwd * TAN_VISIBLE) & (dist >= MIN_RANGE)
    clear = _clear(env, px, py, centres)
    ok = in_view & clear & (dist <= MAX_RANGE)
    if ok.any():
        k = int(np.argmin(np.where(ok, dist, np.inf)))
        return True, float(fwd[k]), float(left[k]), False
    ambiguous = bool((in_view & clear & (dist <= AMBIGUOUS_RANGE)).any())
    if not ambiguous:
        for e in pads:
            corners = np.unique(np.asarray(e.polygon, np.float64).reshape(-1, 2), axis=0)
            cf, cl = _ego(px, py, heading, corners)
            cd = np.hypot(cf, cl)
            near = (cf > 0) & (np.abs(cl) <= cf * TAN_FRAME) & (cd <= AMBIGUOUS_RANGE)
            if near.any() and _clear(env, px, py, corners[near]).any():
                ambiguous = True
                break
    return False, 0.0, 0.0, ambiguous


def noisy(exit_obs, rng, p_drop: float = 0.2, sigma_rel: float = 0.05, sigma_abs: float = 8.0,
          p_false: float = 0.005) -> tuple[bool, float, float]:
    """Degrade a true exit observation the way an imperfect detector would: missed detections,
    position error growing with distance, and occasional false detections."""
    visible, x, y = bool(exit_obs[0]), float(exit_obs[1]), float(exit_obs[2])
    if visible:
        if rng.random() < p_drop:
            return False, 0.0, 0.0
        sigma = sigma_abs + sigma_rel * math.hypot(x, y)
        return True, x + float(rng.normal(0.0, sigma)), y + float(rng.normal(0.0, sigma))
    if rng.random() < p_false:
        d = float(rng.uniform(100.0, MAX_RANGE))
        t = float(rng.uniform(-TAN_VISIBLE, TAN_VISIBLE))
        return True, d, d * t
    return False, 0.0, 0.0
