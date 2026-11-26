"""Training-time scaffolding. Never imported by the agent in submission/.

Text views of what the teacher sees and intends: the memory-map crop around the agent with the
route, the waypoint and the target drawn in. Used to read failures without a viewer.

Legend (x grows to the right, y grows upwards, as on a map of the start frame):
    #  wall            .  free floor       ,  floor the agent has stood on     (blank) never observed
    %  floor and wall evidence in the same cell
    D  closed door     L  locked door      k  key      +  pickup      E  exit believed here
    o  route           *  target           @  agent (then < > ^ v for its heading)
"""

from __future__ import annotations

import math

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec

from .grid import CENTER, N, cell_of


def render(crop: np.ndarray, pose=None, info: dict | None = None, radius: int | None = None) -> str:
    """The crop as text. `info` is `Teacher.info` (route and target are drawn when present)."""
    info = info or {}
    if radius is None:                                    # fit the observed part of the map
        seen = np.argwhere((crop[spec.CH_FREE] > 0.5) | (crop[spec.CH_WALL] > 0.5) | (crop[spec.CH_DOOR] > 0.5)
                           | (crop[spec.CH_LOCKED] > 0.5))
        radius = int(np.abs(seen - CENTER).max()) + 1 if len(seen) else 4
    radius = min(radius, CENTER - 1)
    ox = oy = None
    if pose is not None:
        cx, cy = cell_of(float(pose[0]), float(pose[1]))
        ox, oy = cx - CENTER, cy - CENTER
    route = set()
    target = None
    if ox is not None:
        route = {(a - ox, b - oy) for a, b in info.get("route", ())}
        if info.get("target") is not None:
            target = (info["target"][0] - ox, info["target"][1] - oy)
    rows = []
    for j in range(CENTER + radius, CENTER - radius - 1, -1):
        line = []
        for i in range(CENTER - radius, CENTER + radius + 1):
            free, wall = crop[spec.CH_FREE, i, j] > 0.5, crop[spec.CH_WALL, i, j] > 0.5
            ch = "%" if free and wall else "#" if wall else "." if free else " "
            if ch == "." and crop[spec.CH_VISITED, i, j] > 0.5:
                ch = ","
            if crop[spec.CH_ITEM, i, j] > 0.5:
                ch = "+"
            if crop[spec.CH_KEY, i, j] > 0.5:
                ch = "k"
            if crop[spec.CH_DOOR, i, j] > 0.5:
                ch = "D"
            if crop[spec.CH_LOCKED, i, j] > 0.5:
                ch = "L"
            if crop[spec.CH_EXIT, i, j] >= 0.45:
                ch = "E"
            if (i, j) in route and ch in ".,":
                ch = "o"
            if (i, j) == target:
                ch = "*"
            if i == CENTER and j == CENTER:
                ch = "@"
                if pose is not None:
                    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
                    ch = (">" if c > 0 else "<") if abs(c) >= abs(s) else ("^" if s > 0 else "v")
            line.append(ch)
        rows.append("".join(line).rstrip())
    return "\n".join(rows)


def describe(state: dict) -> str:
    """One line from `TeacherPolicy.debug_state()`."""
    pose = state.get("pose", (0.0, 0.0, 0.0))
    return (f"step {state.get('step')} of budget {state.get('budget', 0):.0f}: {state.get('subgoal')} -> "
            f"{state.get('target')} ({state.get('mode')}), at cell {state.get('cell')} "
            f"pose ({pose[0]:.0f}, {pose[1]:.0f}, {math.degrees(pose[2]):.0f} deg), heads {state.get('heads')}, "
            f"keys {state.get('keys')}, exit known {state.get('exit_known')}, given-up locks {len(state.get('failed_locks', {}))}")


def frame(policy) -> str:
    """Status line plus map for a `TeacherPolicy` (after at least one `act`)."""
    state = policy.debug_state()
    if policy.crop is None:
        return describe(state)
    return describe(state) + "\n" + render(policy.crop, state.get("pose"), state)


__all__ = ["render", "describe", "frame", "N"]
