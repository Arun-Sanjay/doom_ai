"""Solvability check for grid maps. Training-time scaffolding, never imported by the agent.

`solve(grid)` runs a breadth-first search over (cell, keys held): the player walks between
4-connected open cells, a key cell adds its colour to the set of keys, a locked door cell can only
be entered with its colour, and the exit pad ends the search branch. The generator uses the result
to reject unsolvable maps and to choose a tick cap that leaves several times the shortest route.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from . import udmf

KEY_BIT = {"red": 1, "blue": 2, "yellow": 4}
# Cost of a route in decision steps (4 tics each). Measured with the reference walker of
# `walker.py` on 120 generated maps: steps = 1.91 * cells + 3.2 * corners + 18.7 * doors + 4
# (mean absolute error 9 steps); the constants below round that up.
STEPS_PER_CELL = 2      # walking covers about half a 64-unit cell per decision step
STEPS_PER_CORNER = 4    # momentum has to be turned around a corner
STEPS_PER_DOOR = 20     # stop, face the door, press USE, wait for it to rise, start again
Cell = tuple[int, int]  # (col, row)


@dataclass
class Solution:
    """Result of `solve`. `path` is the shortest route from the start to an exit pad ([] if none)."""

    start: Cell
    has_exit: bool
    exit_reachable: bool
    path: list[Cell] = field(default_factory=list)
    doors_on_path: int = 0
    keys_needed: bool = False       # no route to the exit without picking up a key
    open_cells: int = 0
    reachable_cells: int = 0

    @property
    def all_reachable(self) -> bool:
        """Every open cell (floor, things, doors, pads) can be reached from the start."""
        return self.reachable_cells == self.open_cells

    @property
    def solvable(self) -> bool:
        """The exit can be reached (maps without an exit only need every cell reachable)."""
        return self.all_reachable and (self.exit_reachable or not self.has_exit)

    @property
    def path_cells(self) -> int:
        return max(0, len(self.path) - 1)

    @property
    def corners(self) -> int:
        """Changes of direction along the route."""
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.path, self.path[1:])]
        return sum(s != t for s, t in zip(steps, steps[1:]))

    @property
    def optimal_steps(self) -> int:
        """Decision steps the shortest route takes at walking speed, corners and doors included."""
        return (STEPS_PER_CELL * self.path_cells + STEPS_PER_CORNER * self.corners
                + STEPS_PER_DOOR * self.doors_on_path)


def _search(rows: list[str], start: Cell, pick_keys: bool):
    """BFS over (cell, key mask). Returns (first exit state or None, parent links)."""
    h, w = len(rows), len(rows[0])
    first = (start, 0)
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {first: None}
    queue = deque([first])
    goal = None
    while queue:
        state = queue.popleft()
        (c, r), mask = state
        if rows[r][c] == "E":
            goal = goal or state
            continue                                   # the level ends on the pad
        for nc, nr in ((c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1)):
            if not (0 <= nc < w and 0 <= nr < h) or rows[nr][nc] == udmf.WALL:
                continue
            sym = udmf.LEGEND[rows[nr][nc]]
            if sym.kind == "door" and sym.lock and not mask & KEY_BIT[sym.color]:
                continue
            new_mask = mask | KEY_BIT[sym.color] if pick_keys and sym.group == "key" else mask
            nxt = ((nc, nr), new_mask)
            if nxt not in parent:
                parent[nxt] = state
                queue.append(nxt)
    return goal, parent


def solve(grid: str) -> Solution:
    """Search the grid over (cell, keys held) and describe the shortest route to the exit."""
    rows = udmf.parse_grid(grid)
    start = next((c, r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "P")
    goal, parent = _search(rows, start, pick_keys=True)
    sol = Solution(
        start=start,
        has_exit=any("E" in row for row in rows),
        exit_reachable=goal is not None,
        open_cells=sum(len(row) - row.count(udmf.WALL) for row in rows),
        reachable_cells=len({cell for cell, _ in parent}),
    )
    if goal is not None:
        state = goal
        while state is not None:
            sol.path.append(state[0])
            state = parent[state]
        sol.path.reverse()
        sol.doors_on_path = sum(rows[r][c] in udmf.DOOR_CHARS for c, r in sol.path)
        sol.keys_needed = _search(rows, start, pick_keys=False)[0] is None
    return sol
