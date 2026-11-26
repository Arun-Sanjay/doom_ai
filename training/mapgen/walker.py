"""Reference walker for generated maps. Training-time scaffolding, never imported by the agent.

`walk(env, grid)` drives the player along the route found by `solver.solve`, using the true position
and heading from `env.privileged_state()`. It exists to test the generator inside the engine: a map
counts as verified when the walker picks up the keys, opens the doors and reaches the exit pad
within the tick cap. It is not a policy and nothing in `submission/` uses it.

Control: the walker aims at the end of the current straight stretch of the route and presses the
four move buttons so that the point where it would coast to a stop lands on the target (the engine
keeps about 9.7 tics' worth of momentum). When it is off the centre line of the stretch it returns
to the line before it goes on, so it never cuts across a wall corner. It turns only to face a
closed door, which it opens with one USE press from the middle of the cell in front of it.
"""

from __future__ import annotations

import math

import numpy as np

from . import udmf
from .layouts import strip_monsters  # noqa: F401  (kept under this name for the tests, which walk bare maps)
from .solver import solve

FORWARD, BACK, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, USE = 0, 1, 2, 3, 4, 5, 7
ARRIVE = 18.0            # map units from a cell centre that count as "there"
COAST = 9.67             # tics of momentum: friction keeps 29/32 of the speed every tic
DEADBAND = 10.0          # one 4-tic press moves the stop point by about 20 units: do not chase less
DOOR_OPEN = 56.0         # opening height the player fits through
OFF_LINE = 20.0          # allowed distance from the centre line of a stretch
# Heading in degrees (0 = east, counter-clockwise) of a step from one grid cell to its neighbour,
# keyed by (columns, rows) moved. Row 0 is the north row, so one row up is a step north.
STEP_HEADING = {(1, 0): 0.0, (0, -1): 90.0, (-1, 0): 180.0, (0, 1): -90.0}


def _run_end(rows, here, route: list) -> int:
    """Index of the last route cell that lies straight ahead of `here` (the cell just reached).
    A run never continues into a door cell: the walker stops in front of a door and opens it first."""
    end = 0
    if rows[route[0][1]][route[0][0]] not in udmf.DOOR_CHARS:
        step = (route[0][0] - here[0], route[0][1] - here[1])
        while (end + 1 < len(route)
               and (route[end + 1][0] - route[end][0], route[end + 1][1] - route[end][1]) == step
               and rows[route[end + 1][1]][route[end + 1][0]] not in udmf.DOOR_CHARS):
            end += 1
    return end


def _move_to(action: np.ndarray, st: dict, obs: dict, tx: float, ty: float, axis=(1.0, 0.0)) -> None:
    """Press move buttons so the point where the player would coast to a stop approaches (tx, ty).

    The error is split along `axis` (the direction of the current stretch, a unit vector in map
    axes) and across it, each with its own deadband, so a long way to go never hides a sideways
    error. Once the player is more than OFF_LINE from the line it only brakes along the axis. The
    wanted push is then turned into the nearest of the eight directions the buttons can produce.
    """
    heading = math.radians(st["angle"])
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    vf, vr = float(obs["vitals"][14]), float(obs["vitals"][15])      # forward and right speed, units per tic
    vx, vy = vf * cos_h + vr * sin_h, vf * sin_h - vr * cos_h
    ax, ay = axis
    dx, dy = tx - st["x"], ty - st["y"]
    along, across = dx * ax + dy * ay, -dx * ay + dy * ax
    if abs(across) > OFF_LINE:
        along = 0.0
    along -= COAST * (vx * ax + vy * ay)
    across -= COAST * (-vx * ay + vy * ax)
    push_a = (along > DEADBAND) - (along < -DEADBAND)
    push_n = (across > DEADBAND) - (across < -DEADBAND)
    wx, wy = push_a * ax - push_n * ay, push_a * ay + push_n * ax        # wanted push in map axes
    ahead, left = wx * cos_h + wy * sin_h, -wx * sin_h + wy * cos_h
    action[FORWARD], action[BACK] = ahead > 0.38, ahead < -0.38
    action[LEFT], action[RIGHT] = left > 0.38, left < -0.38


def walk(env, grid: str, max_steps: int | None = None) -> dict:
    """Reset `env` (built from `grid`) and follow the solver's route.
    Returns {"exit_reached", "steps", "route_cells", "optimal_steps", "keys"}."""
    rows = udmf.parse_grid(grid)
    height = len(rows)
    sol = solve(grid)
    if not sol.path:
        raise ValueError("the grid has no exit that the solver can reach")
    sector = udmf.assign_sectors(rows)[0]
    obs, info = env.reset(seed=0)
    here = sol.path[0]                 # last route cell reached
    route = sol.path[1:]
    steps, pressed_at, done = 0, -100, False
    limit = max_steps if max_steps is not None else env.episode_ticks // 4
    while route and steps < limit and not done:
        st = env.privileged_state()
        end = _run_end(rows, here, route)
        col, row = route[end]
        closed = False
        if rows[row][col] in udmf.DOOR_CHARS:
            sid = sector[row][col]
            closed = st["sector_ceiling"][sid] - st["sector_floor"][sid] < DOOR_OPEN
        action = np.zeros(9, np.int8)
        if closed:                      # wait in the middle of the cell in front of the door, facing it
            wx, wy = udmf.cell_centre(here[0], here[1], height)
            facing = STEP_HEADING[(col - here[0], row - here[1])]     # a closed door is always the next route cell
            turn = (facing - st["angle"] + 180.0) % 360.0 - 180.0
            if turn > 5.0:
                action[TURN_LEFT] = 1
            elif turn < -5.0:
                action[TURN_RIGHT] = 1
            _move_to(action, st, obs, wx, wy)
            if abs(turn) <= 12.0 and math.hypot(wx - st["x"], wy - st["y"]) < ARRIVE and steps - pressed_at > 30:
                action[USE] = 1
                pressed_at = steps
        else:
            tx, ty = udmf.cell_centre(col, row, height)
            if math.hypot(tx - st["x"], ty - st["y"]) < ARRIVE and rows[row][col] != "E":
                here = route[end]
                del route[: end + 1]
                continue
            _move_to(action, st, obs, tx, ty, axis=(route[0][0] - here[0], here[1] - route[0][1]))
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    return {
        "exit_reached": bool(info.get("exit_reached")),
        "steps": steps,
        "route_cells": sol.path_cells,
        "optimal_steps": sol.optimal_steps,
        "keys": [float(v) for v in obs["vitals"][8:14]],
    }
