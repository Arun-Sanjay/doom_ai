"""Training-time scaffolding. Never imported by the agent in submission/.

Planning grid of the navigation teacher: the cell classes it reads from the memory-map crop and
the shortest-path search it runs over them.

The grid is the crop itself: 64 x 64 cells of 64 map units, index [i, j] with i along the start
frame's x axis and j along its y axis, the agent in cell (32, 32). A flat cell index is
`i * 64 + j`, the same number the manager's target pointer uses. Nothing here knows the true map:
every array is computed from the eleven channels of `MemoryMap.crop`.

Cell classes
    plain    observed free floor with nothing special on it: `CH_FREE` and not `CH_WALL`, no closed
             door, no exit pad that is being kept clear
    door     a closed ordinary door was last seen here (`CH_DOOR`): can be crossed at extra cost
    locked   a closed locked door was last seen here (`CH_LOCKED`): can be crossed only while the
             caller says so (a key is held and the door has not been given up)
    pad      the exit is believed here (`CH_EXIT`): can be entered but never crossed, so no route
             to anything else runs over it
    unknown  none of free, wall, door, locked door: never observed

Moves are 8-connected. A diagonal move needs the two cells beside it and the cell it ends in to be
plain, so no route cuts a wall corner, squeezes past a door frame or brushes the exit pad.
Costs are travel time in decision steps: 2.4 per cell (measured: the teacher's own routes take
2.4 steps per cell and 13 per closed door), 2.4 * sqrt(2) per diagonal, plus 13 per closed door.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from training import paths  # noqa: F401  (puts submission/ on sys.path)
from doomnet import spec

N = spec.CROP_SIZE                         # cells per side of the planning grid (64)
CELLS = N * N
CENTER = N // 2                            # the agent's cell on both axes
OWN = CENTER * N + CENTER                  # flat index of the agent's cell
CELL = spec.MAP_RES * spec.CROP_MULT       # map units per cell (64)
INF = float("inf")

STEPS_PER_CELL = 2.4                       # decision steps per cell of route: 1.9 at full speed, 2.4 measured over
                                           # whole routes of the teacher (295 legs), which slow down at corners
STEPS_DIAGONAL = STEPS_PER_CELL * math.sqrt(2.0)
EXIT_SEEN = 0.45                           # CH_EXIT at or above this: at least one sighting
EXIT_SURE = 0.95                           # ... at least two sightings, in the cell and its eight neighbours together


def cell_of(x: float, y: float) -> tuple[int, int]:
    """Start-frame cell that contains a start-frame position (cell k spans [64k - 32, 64k + 32))."""
    return int(math.floor((x + CELL / 2) / CELL)), int(math.floor((y + CELL / 2) / CELL))


def crop_origin(x: float, y: float) -> tuple[int, int]:
    """Start-frame cell shown at crop index (0, 0) when the agent stands at (x, y)."""
    cx, cy = cell_of(x, y)
    return cx - CENTER, cy - CENTER


def cell_centre(cx: int, cy: int) -> tuple[float, float]:
    return cx * CELL, cy * CELL


class PlanGrid:
    """Cell classes of one crop, as boolean arrays (N, N) and as the flat lists the search reads.

    `locked_ok` is a boolean array (N, N) saying which locked-door cells may be crossed now.
    `pad_floor` is the exit belief from which a floor cell is kept clear as a pad (EXIT_SEEN: one
    sighting is enough; EXIT_SURE: confirmed by a second sighting). The outermost ring of the grid is never
    passable, so the search can step to a neighbour by adding to the flat index without a bounds
    check; maps are at most 28 cells across and the agent sits at 32, so nothing real is lost.
    """

    def __init__(self, crop: np.ndarray, locked_ok: np.ndarray | None = None, door_steps: float = 13.0,
                 pad_floor: float = EXIT_SEEN):
        free = crop[spec.CH_FREE] > 0.5
        wall = crop[spec.CH_WALL] > 0.5
        locked = (crop[spec.CH_LOCKED] > 0.5) & ~wall
        door = (crop[spec.CH_DOOR] > 0.5) & ~wall & ~locked
        self.free, self.wall, self.door, self.locked = free, wall, door, locked
        self.visited = crop[spec.CH_VISITED] > 0.5
        self.unknown = ~(free | wall | door | locked)
        belief = crop[spec.CH_EXIT]
        self.pad_seen = belief >= EXIT_SEEN
        # Confirmed: seen here, and at least twice in this cell and the eight around it together
        # (a detector places a far pad a cell off now and then; false detections do not cluster).
        rows = belief.copy()
        rows[1:, :] += belief[:-1, :]
        rows[:-1, :] += belief[1:, :]
        around = rows.copy()
        around[:, 1:] += rows[:, :-1]
        around[:, :-1] += rows[:, 1:]
        self.pad_sure = self.pad_seen & (around >= EXIT_SURE)
        self.key = crop[spec.CH_KEY] > 0.5
        self.item = crop[spec.CH_ITEM] > 0.5
        self.recency = crop[spec.CH_RECENCY]

        floor = free & ~wall & ~door & ~locked
        floor[CENTER, CENTER] = True                      # the agent is standing there
        pad = (self.pad_sure if pad_floor >= EXIT_SURE else self.pad_seen) & floor
        pad[CENTER, CENTER] = False
        self.pad = pad                                    # pads that are kept clear
        plain = floor & ~pad
        gate = door | (locked & locked_ok if locked_ok is not None else np.zeros_like(locked))
        for a in (plain, gate, pad):
            a[0, :] = a[-1, :] = False
            a[:, 0] = a[:, -1] = False
        self.plain, self.gate = plain, gate
        self.passable = plain | gate

        # ---- flat lists for the search ------------------------------------------------------
        self.plain_l = plain.reshape(-1).tolist()
        extra = np.where(gate, float(door_steps), 0.0)
        self.extra_l = extra.reshape(-1).tolist()                                   # cost of opening the cell
        self.enter_l = np.where(plain | gate | pad, STEPS_PER_CELL + extra, 0.0).reshape(-1).tolist()
        self.leave_l = np.where(plain | gate, STEPS_PER_CELL, 0.0).reshape(-1).tolist()
        self.pad_l = pad.reshape(-1).tolist()
        self._zeros = [0.0] * CELLS
        self._false = [False] * CELLS

    # ------------------------------------------------------------------------------------------
    def from_agent(self) -> tuple[list, list]:
        """Travel steps from the agent's cell to every cell: (dist, link), link[n] = the cell before n.

        Pads can be reached but not crossed. Door cells cost their opening time when entered.
        """
        return search(self.enter_l, self._zeros, self.plain_l, self.pad_l, OWN)

    def to_target(self, target: int) -> tuple[list, list]:
        """Travel steps from every cell to `target`: (field, link), link[c] = the next cell on the way.

        The same costs as `from_agent`, searched backwards: the opening time of a door cell is
        charged to the move that enters it. Pads other than the target are never crossed.
        """
        return search(self.leave_l, self.extra_l, self.plain_l, self._false, target)

    def frontier(self) -> np.ndarray:
        """Plain cells with a 4-neighbour that has never been observed."""
        u = self.unknown
        beside = np.zeros_like(u)
        beside[1:, :] |= u[:-1, :]
        beside[:-1, :] |= u[1:, :]
        beside[:, 1:] |= u[:, :-1]
        beside[:, :-1] |= u[:, 1:]
        return self.plain & beside

    def dead_ends(self) -> np.ndarray:
        """Plain cells closed in on three sides by observed walls (corridor ends and alcoves)."""
        w = self.wall & ~self.gate & ~self.plain
        count = np.zeros(w.shape, np.int8)
        count[1:, :] += w[:-1, :]
        count[:-1, :] += w[1:, :]
        count[:, 1:] += w[:, :-1]
        count[:, :-1] += w[:, 1:]
        return self.plain & (count == 3)

    def unvisited_rooms(self, doorway: np.ndarray | None = None) -> np.ndarray:
        """Plain cells of 4-connected floor regions in which the agent has never stood, as a mask.

        A region stops at walls, pads and doors. `doorway` marks cells known to hold a door even
        while it stands open, so a room that was only looked into through its door is found here.
        """
        room = self.plain & ~doorway if doorway is not None else self.plain
        room_l = room.reshape(-1).tolist()
        seen = room & self.visited
        if room[CENTER, CENTER]:
            seen[CENTER, CENTER] = True
        mark = seen.reshape(-1).tolist()
        stack = np.flatnonzero(seen).tolist()
        while stack:                                   # spread from every visited cell over the room floor
            c = stack.pop()
            for n in (c - N, c + N, c - 1, c + 1):
                if room_l[n] and not mark[n]:
                    mark[n] = True
                    stack.append(n)
        return room & ~np.array(mark, bool).reshape(N, N)


def search(step_cost: list, entry_extra: list, plain: list, stop: list, source: int) -> tuple[list, list]:
    """Uniform-cost search from `source` over the flat grid.

    step_cost[n]   cost of an orthogonal move that ends in n; 0.0 = n cannot be moved into
    entry_extra[c] added once to every move that starts from c (the backward search uses it to
                   charge a door's opening time to the moves that enter the door)
    plain[c]       a diagonal move needs its start, its end and the two cells beside it plain
    stop[c]        cells that end a route: reached, but never expanded (unless c is the source)

    Returns (dist, link): dist[n] the cost of the cheapest route (INF if there is none) and
    link[n] the cell that route came from (-1 for the source and for unreached cells).
    """
    dist = [INF] * CELLS
    link = [-1] * CELLS
    dist[source] = 0.0
    heap = [(0.0, source)]
    pop, push = heapq.heappop, heapq.heappush
    diagonal = STEPS_DIAGONAL
    while heap:
        d, c = pop(heap)
        if d > dist[c] or (stop[c] and c != source):
            continue
        d += entry_extra[c]
        a, b, l, r = c - N, c + N, c - 1, c + 1
        e = step_cost[a]
        if e:
            nd = d + e
            if nd < dist[a]:
                dist[a] = nd
                link[a] = c
                push(heap, (nd, a))
        e = step_cost[b]
        if e:
            nd = d + e
            if nd < dist[b]:
                dist[b] = nd
                link[b] = c
                push(heap, (nd, b))
        e = step_cost[l]
        if e:
            nd = d + e
            if nd < dist[l]:
                dist[l] = nd
                link[l] = c
                push(heap, (nd, l))
        e = step_cost[r]
        if e:
            nd = d + e
            if nd < dist[r]:
                dist[r] = nd
                link[r] = c
                push(heap, (nd, r))
        if not plain[c]:
            continue
        nd = d + diagonal
        if plain[a]:
            if plain[l] and plain[a - 1] and nd < dist[a - 1]:
                dist[a - 1] = nd
                link[a - 1] = c
                push(heap, (nd, a - 1))
            if plain[r] and plain[a + 1] and nd < dist[a + 1]:
                dist[a + 1] = nd
                link[a + 1] = c
                push(heap, (nd, a + 1))
        if plain[b]:
            if plain[l] and plain[b - 1] and nd < dist[b - 1]:
                dist[b - 1] = nd
                link[b - 1] = c
                push(heap, (nd, b - 1))
            if plain[r] and plain[b + 1] and nd < dist[b + 1]:
                dist[b + 1] = nd
                link[b + 1] = c
                push(heap, (nd, b + 1))
    return dist, link
