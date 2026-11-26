"""Seeded procedural grid layouts for tiers 0-4. Training-time scaffolding, never imported by the agent.

`generate(tier, seed)` returns a map as plain data::

    {"grid": "#####\\n#P.E#\\n#####", "tier": 2, "seed": 7, "style": "house", "episode_ticks": 2800,
     "options": {...udmf.build_wad options...}, "counts": {...}}

Tiers follow docs/EVALUATION.md, plus a tier 0 for the start of the curriculum:

  0  navigation      monster-free maps with an exit: corridors, rooms, doors, dead ends; one map
                     in three also has one key and its locked door on the way to the exit
  1  combat arena    open rooms with cover and monsters, about half with an exit
  2  exploration     many rooms, corridors, doors and dead ends, light opposition
  3  keys and locks  key -> locked door -> exit, the door usually met before its key
  4  gauntlet        two or three locks of different colours, ambushes, scarce pickups

Keys and locked doors can be learned before any combat on the tier 0 maps that have them, and
`without_monsters` turns any map into a monster-free copy with the same layout and tick cap (used to
train and to score navigation apart from combat).

A map is built in passes. A *carver* (strip, arena, house, scatter, maze, cave) cuts floor out of
solid wall. Generic passes then place the start, the exit pad (in an alcove, at a corridor end,
against a wall, in a corner or in open floor), locked doors on chokepoints of the route with their
keys hidden in the part of the map that is reachable before the lock, ordinary doors on room
entrances, and finally monsters and pickups. Every draft is checked with `solver.solve` (a search
over cell and keys held): every open cell must be reachable, the exit must be reachable, locks must
really be needed, and the tick cap must be at least `MIN_BUDGET` times the shortest route at
walking speed. Drafts that fail are discarded and the next attempt of the same seed is tried, so
`generate` is a pure function of (tier, seed).
"""

from __future__ import annotations

import random
from collections import deque

from . import udmf
from .solver import Solution, solve

WALL, FLOOR, START, EXIT, DOOR = "#", ".", "P", "E", "D"
MIN_TICKS, MAX_TICKS, TICK_STEP = 2100, 4200, 350
MIN_BUDGET, MAX_BUDGET = 3.0, 4.5   # the tick cap is at least this many times the shortest route
MIN_LOCK_ZONE = 8            # open cells in front of a locked door, and between two locks
MIRROR_CHANCE = 0.12         # share of maps with a mirror-symmetric plan
GUARD_CHANCE = 0.35          # chance that a key or the exit has a monster next to it (tiers 3 and 4)
FRAME_SKIP = 4
MAX_ATTEMPTS = 1000
ROOM_KINDS = ("floor", "thing", "start")
DOOR_FOR = {"red": "R", "blue": "B", "yellow": "Y"}
CARD_FOR = {"red": "r", "blue": "b", "yellow": "y"}
SKULL_FOR = {"red": "1", "blue": "2", "yellow": "3"}
EXIT_STYLES = ("alcove", "end", "wall", "corner", "open")

MONSTER_HP = {"z": 20, "s": 30, "c": 70, "i": 60, "d": 150, "g": 150, "l": 100, "o": 400, "k": 500}
WIDE_MONSTERS = "dgok"       # radius 24-31: placed in rooms, never in 1-wide passages
MONSTER_MIX = {
    1: (("z", 22), ("s", 14), ("i", 24), ("c", 7), ("d", 11), ("g", 4), ("l", 6), ("o", 7), ("k", 5)),
    2: (("z", 45), ("i", 30), ("s", 15), ("d", 10)),
    3: (("z", 32), ("i", 28), ("s", 15), ("d", 10), ("c", 6), ("l", 5), ("g", 4)),
    4: (("z", 18), ("s", 15), ("i", 20), ("c", 12), ("d", 10), ("g", 6), ("l", 6), ("o", 7), ("k", 6)),
}
MONSTER_HP_PER_CELL = {1: 8, 2: 3, 3: 4, 4: 8}   # cap on total monster health per open cell
MIN_MONSTERS = {0: 0, 1: 2, 2: 0, 3: 1, 4: 4}
MONSTER_FREE = {2: 0.35}                         # share of the maps of a tier that get no monster at all
# Build options that only concern monsters; a monster-free copy of a map drops them.
MONSTER_OPTIONS = ("ambush", "thing_angles")
# Rough damage a pickup lets the player deal, in pistol bullets (about 10 damage each).
AMMO_VALUE = {"n": 10, "x": 50, "t": 28, "f": 56, "j": 20}
START_BULLETS = 50

# Every KEY_STAGE_EVERY-th tier 0 seed gives a map with one key and one locked door of its colour on
# the route to the exit (`has_key_stage`). The choice depends on the seed alone, so the other tier 0
# seeds give exactly the maps they would give without this.
KEY_STAGE_EVERY = 3
KEY_STAGE_LOCKS = ((1, 1),)

# `locks` is ((number of locks, weight), ...), drawn once per draft.
TIERS = {
    0: dict(name="navigation", size=(8, 18), exit_chance=1.0, locks=((0, 1),), ammo=0.0,
            doors=(0.0, 0.3, 0.6, 0.9, 1.0, 1.0),
            styles=(("strip", 3), ("arena", 2), ("house", 5), ("scatter", 4), ("maze", 4), ("cave", 1)),
            min_open=6, min_route=3, ticks=((2100, 5), (2450, 2), (2800, 2), (3500, 1))),
    1: dict(name="combat arena", size=(8, 18), exit_chance=0.5, locks=((0, 1),), ammo=2.2,
            doors=(0.0, 0.0, 0.3, 0.7),
            styles=(("arena", 12), ("strip", 2), ("house", 2), ("scatter", 2), ("cave", 2)),
            min_open=10, min_route=3, ticks=((2100, 6), (2450, 2), (2800, 2), (3500, 1))),
    2: dict(name="exploration", size=(12, 24), exit_chance=1.0, locks=((0, 1),), ammo=2.0,
            doors=(0.0, 0.3, 0.6, 0.9, 0.9, 1.0),
            styles=(("house", 7), ("scatter", 6), ("maze", 5), ("cave", 1)),
            min_open=60, min_route=14, ticks=((2100, 2), (2450, 2), (2800, 3), (3150, 2), (3500, 2), (4200, 1))),
    3: dict(name="keys and locks", size=(12, 26), exit_chance=1.0, locks=((1, 8), (2, 2)), ammo=2.0,
            doors=(0.0, 0.3, 0.6, 0.9, 0.9, 1.0),
            styles=(("house", 7), ("scatter", 6), ("maze", 5)),
            min_open=60, min_route=18, ticks=((2450, 1), (2800, 2), (3150, 2), (3500, 3), (3850, 2), (4200, 2))),
    4: dict(name="gauntlet", size=(16, 28), exit_chance=1.0, locks=((2, 6), (3, 4)), ammo=1.2,
            doors=(0.3, 0.5, 0.7, 0.9, 1.0),
            styles=(("house", 6), ("scatter", 6), ("maze", 4)),
            min_open=110, min_route=30, ticks=((2800, 1), (3150, 1), (3500, 2), (3850, 2), (4200, 4))),
}


class Reject(Exception):
    """A draft could not be finished; `generate` moves on to the next attempt of the same seed."""


def pick(rng: random.Random, weighted):
    """Draw one value from ((value, weight), ...)."""
    values, weights = zip(*weighted)
    return rng.choices(values, weights)[0]


# ---- grid ------------------------------------------------------------------------------------------
class Grid:
    """Mutable character grid addressed as (col, row); row 0 is the top row. Starts as solid wall."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.rows = [[WALL] * w for _ in range(h)]

    def at(self, c: int, r: int) -> str:
        return self.rows[r][c] if 0 <= c < self.w and 0 <= r < self.h else WALL

    def put(self, c: int, r: int, ch: str) -> None:
        self.rows[r][c] = ch

    def interior(self, c: int, r: int) -> bool:
        return 1 <= c < self.w - 1 and 1 <= r < self.h - 1

    def carve(self, x0: int, y0: int, x1: int, y1: int, ch: str = FLOOR) -> None:
        for r in range(y0, y1 + 1):
            for c in range(x0, x1 + 1):
                self.rows[r][c] = ch

    def cells(self, chars: str = FLOOR) -> list[tuple[int, int]]:
        return [(c, r) for r in range(self.h) for c in range(self.w) if self.rows[r][c] in chars]

    def open_cells(self) -> list[tuple[int, int]]:
        return [(c, r) for r in range(self.h) for c in range(self.w) if self.rows[r][c] != WALL]

    def text(self) -> str:
        return "\n".join("".join(row) for row in self.rows)

    @staticmethod
    def around(c: int, r: int):
        return ((c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1))

    def exits_of(self, c: int, r: int) -> list[tuple[int, int]]:
        """Open 4-neighbours of a cell."""
        return [n for n in self.around(c, r) if self.at(*n) != WALL]

    def is_room_cell(self, c: int, r: int) -> bool:
        """Open cell that belongs to a room sector (floor, start or a thing; not a door or a pad)."""
        ch = self.at(c, r)
        return ch != WALL and udmf.LEGEND[ch].kind in ROOM_KINDS

    def roomy(self, c: int, r: int) -> bool:
        """Part of a 2x2 block of open cells: a room or a wide corridor, not a 1-wide passage."""
        return any(
            all(self.at(c + dc + i, r + dr + j) != WALL for i in (0, 1) for j in (0, 1))
            for dc in (-1, 0) for dr in (-1, 0)
        )

    def solid(self, c: int, r: int) -> bool:
        """Wall cell that is part of a longer wall, not a free-standing pillar."""
        return self.at(c, r) == WALL and (not self.interior(c, r) or any(self.at(*n) == WALL for n in self.around(c, r)))

    def doorway(self, c: int, r: int) -> str | None:
        """'h' or 'v' if this plain floor cell is a 1-wide passage: solid wall on two opposite sides
        and room cells on the other two. Such a cell can be turned into a door."""
        if self.at(c, r) != FLOOR:
            return None
        if self.solid(c, r - 1) and self.solid(c, r + 1) and self.is_room_cell(c - 1, r) and self.is_room_cell(c + 1, r):
            return "h"
        if self.solid(c - 1, r) and self.solid(c + 1, r) and self.is_room_cell(c, r - 1) and self.is_room_cell(c, r + 1):
            return "v"
        return None

    def threshold(self, c: int, r: int) -> bool:
        """Doorway cell at which the passage ends on at least one side (room entrance, bend, junction)."""
        kind = self.doorway(c, r)
        if kind is None:
            return False
        ends = ((c - 1, r), (c + 1, r)) if kind == "h" else ((c, r - 1), (c, r + 1))
        return any(self.doorway(*n) != kind for n in ends)

    def dist(self, src, blocked=()) -> dict:
        """Walking distance in cells from `src` to every reachable open cell (doors count as open).
        An exit pad is entered but never crossed, because the level ends there."""
        dist = {src: 0}
        queue = deque([src])
        while queue:
            cell = queue.popleft()
            if self.at(*cell) == EXIT and cell != src:
                continue
            for n in self.around(*cell):
                if n not in dist and n not in blocked and self.at(*n) != WALL:
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        return dist

    def path(self, src, dst) -> list[tuple[int, int]]:
        """One shortest walk from src to dst (inclusive), [] if there is none."""
        dist = self.dist(src)
        if dst not in dist:
            return []
        out = [dst]
        while out[-1] != src:
            cell = out[-1]
            out.append(next(n for n in self.around(*cell)
                            if dist.get(n) == dist[cell] - 1 and (self.at(*n) != EXIT or n == src)))
        return out[::-1]

    def connected(self, start) -> bool:
        """Every open cell can be reached from `start`."""
        return len(self.dist(start)) == len(self.open_cells())

    def keep_largest_region(self) -> None:
        """Fill every open region except the largest one."""
        best, seen = {}, set()
        for cell in self.open_cells():
            if cell not in seen:
                region = self.dist(cell)
                seen |= set(region)
                if len(region) > len(best):
                    best = region
        for c, r in self.open_cells():
            if (c, r) not in best:
                self.put(c, r, WALL)

    def trimmed(self) -> "Grid":
        """Crop to the open cells plus a one-cell wall border."""
        cells = self.open_cells()
        c0, c1 = min(c for c, _ in cells) - 1, max(c for c, _ in cells) + 1
        r0, r1 = min(r for _, r in cells) - 1, max(r for _, r in cells) + 1
        g = Grid(c1 - c0 + 1, r1 - r0 + 1)
        g.rows = [self.rows[r][c0:c1 + 1] for r in range(r0, r1 + 1)]
        return g

    def transformed(self, rng: random.Random) -> "Grid":
        """Random mirror and transpose, so no carver has a preferred orientation."""
        rows = self.rows
        if rng.random() < 0.5:
            rows = [row[::-1] for row in rows]
        if rng.random() < 0.5:
            rows = rows[::-1]
        if rng.random() < 0.3:
            rows = [list(col) for col in zip(*rows)]
        g = Grid(len(rows[0]), len(rows))
        g.rows = [list(row) for row in rows]
        return g


# ---- carving helpers -------------------------------------------------------------------------------
def grow_stub(g: Grid, rng: random.Random, base, lengths=(2, 5)) -> list[tuple[int, int]]:
    """Carve a dead-end 1-wide corridor out of solid wall, starting next to open cell `base`.
    The corridor never breaks through into other floor. Returns its cells ([] if no room)."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rng.shuffle(dirs)
    want = rng.randint(*lengths)
    for dc, dr in dirs:
        cells, (x, y) = [], (base[0] + dc, base[1] + dr)
        while (len(cells) < want and g.interior(x, y) and g.at(x, y) == WALL
               and all(g.at(x + i * dr + j * dc, y + i * dc + j * dr) == WALL for i in (-1, 1) for j in (0, 1))
               and g.at(x + dc, y + dr) == WALL):
            cells.append((x, y))
            x, y = x + dc, y + dr
        if len(cells) >= lengths[0]:
            for c, r in cells:
                g.put(c, r, FLOOR)
            return cells
    return []


def add_stubs(g: Grid, rng: random.Random, count: int, lengths=(2, 5)) -> int:
    """Add up to `count` dead-end corridors at random places."""
    floor = g.cells(FLOOR)
    rng.shuffle(floor)
    made = 0
    for base in floor:
        if made >= count:
            break
        made += bool(grow_stub(g, rng, base, lengths))
    return made


def add_gates(g: Grid, rng: random.Random, count: int) -> int:
    """Narrow up to `count` wide passages to a single cell by building a wall across them.

    A passage is a straight line of 2 to 4 floor cells with wall at both ends and floor on both
    sides of every cell. One cell stays open and becomes a doorway that can carry a door or a lock.
    """
    spots = []
    for c, r in g.cells(FLOOR):
        for dc, dr in ((1, 0), (0, 1)):
            if g.at(c - dc, r - dr) != WALL:
                continue                                   # not the first cell of a line
            line, (x, y) = [], (c, r)
            while g.at(x, y) == FLOOR and g.at(x - dr, y - dc) == FLOOR and g.at(x + dr, y + dc) == FLOOR and len(line) < 5:
                line.append((x, y))
                x, y = x + dc, y + dr
            if 2 <= len(line) <= 4 and g.at(x, y) == WALL:
                spots.append(line)
    rng.shuffle(spots)
    made = 0
    for line in spots:
        if made >= count:
            break
        near = {(x + i, y + j) for x, y in line for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)} - set(line)
        (x0, y0), (x1, y1) = line[0], line[-1]
        ends = {(2 * x0 - line[1][0], 2 * y0 - line[1][1]), (2 * x1 - line[-2][0], 2 * y1 - line[-2][1])}
        if any(g.at(*cell) != FLOOR for cell in line) or any(g.doorway(*cell) for cell in near - ends):
            continue                                       # changed meanwhile, or too close to another doorway
        keep = rng.choice(line)
        for cell in line:
            if cell != keep:
                g.put(*cell, WALL)
        made += 1
    return made


def add_pillars(g: Grid, rng: random.Random, rects, chance: float) -> None:
    """Put cover into rooms of at least 5x5 cells: a lattice of pillars, a solid block, a walled
    vault with one entrance, or scattered pillars. The outer ring of a room always stays free and
    pillars never touch, so a room stays connected and its entrances stay open."""
    for x0, y0, x1, y1 in rects:
        if min(x1 - x0, y1 - y0) < 4 or rng.random() >= chance:
            continue
        kind = rng.choice(("lattice", "lattice", "block", "vault", "scatter", "scatter"))
        if kind == "vault" and min(x1 - x0, y1 - y0) >= 6:     # a walled room inside the room, one way in
            ax, ay = rng.randint(2, (x1 - x0 - 2) // 2), rng.randint(2, (y1 - y0 - 2) // 2)
            bx0, by0, bx1, by1 = x0 + ax, y0 + ay, x1 - ax, y1 - ay
            g.carve(bx0, by0, bx1, by1, WALL)
            g.carve(bx0 + 1, by0 + 1, bx1 - 1, by1 - 1)
            g.put(*rng.choice([(c, r) for c in range(bx0 + 1, bx1) for r in (by0, by1)]
                              + [(c, r) for r in range(by0 + 1, by1) for c in (bx0, bx1)]), FLOOR)
        elif kind == "lattice":
            px, py = rng.choice((2, 2, 3)), rng.choice((2, 2, 3))
            for r in range(y0 + 1, y1, py):
                for c in range(x0 + 1, x1, px):
                    g.put(c, r, WALL)
        elif kind == "block" and min(x1 - x0, y1 - y0) >= 5:
            ax, ay = rng.randint(2, (x1 - x0) // 2), rng.randint(2, (y1 - y0) // 2)
            g.carve(x0 + ax, y0 + ay, x1 - ax, y1 - ay, WALL)
        else:
            for _ in range(rng.randint(1, max(1, (x1 - x0) * (y1 - y0) // 10))):
                c, r = rng.randint(x0 + 1, x1 - 1), rng.randint(y0 + 1, y1 - 1)
                if all(g.at(c + i, r + j) != WALL for i in (-1, 0, 1) for j in (-1, 0, 1)):
                    g.put(c, r, WALL)


def connect_rooms(g: Grid, rng: random.Random, rects, max_len: int, loops: int,
                  widen: float = 0.0, merge: float = 0.0) -> None:
    """Carve the rooms `rects` and join them with straight passages through at most `max_len` wall cells.

    A random spanning tree over the rooms keeps everything reachable and `loops` extra passages add
    cycles. `widen` is the chance that a passage becomes 2 or 3 cells wide; `merge` the chance that
    the whole wall between two neighbouring rooms is removed, which gives L- and T-shaped rooms.
    """
    label: dict[tuple[int, int], int] = {}
    for i, (x0, y0, x1, y1) in enumerate(rects):
        g.carve(x0, y0, x1, y1)
        for r in range(y0, y1 + 1):
            for c in range(x0, x1 + 1):
                label[(c, r)] = i
    passages: dict[tuple[int, int], list] = {}
    for (c, r), a in label.items():
        for dc, dr in ((1, 0), (0, 1)):                 # east and south: every passage is found once
            run, (x, y) = [], (c + dc, r + dr)
            while (len(run) < max_len and g.interior(x, y) and g.at(x, y) == WALL
                   and g.at(x + dr, y + dc) == WALL and g.at(x - dr, y - dc) == WALL):
                run.append((x, y))
                x, y = x + dc, y + dr
            b = label.get((x, y))
            if run and b is not None and b != a:
                passages.setdefault((min(a, b), max(a, b)), []).append((dc, dr, tuple(run)))
    edges = sorted(passages)
    rng.shuffle(edges)
    root = list(range(len(rects)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    chosen, spare = [], []
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            root[ra] = rb
            chosen.append((a, b))
        else:
            spare.append((a, b))
    for edge in chosen + spare[:loops]:
        runs = passages[edge]
        dc, dr, run = rng.choice(runs)
        opened = [run]
        if len(run) == 1 and rng.random() < merge:
            opened = [other for _, _, other in runs if len(other) == 1]
        elif rng.random() < widen:
            known = {other for _, _, other in runs}
            for side in rng.sample((-1, 1), rng.choice((1, 1, 2))):
                shifted = tuple((x + side * dr, y + side * dc) for x, y in run)
                if shifted in known:
                    opened.append(shifted)
        for cells in opened:
            for c, r in cells:
                g.put(c, r, FLOOR)


def link_regions(g: Grid, rng: random.Random, max_links: int = 40) -> None:
    """Join separate open regions with 1-wide L-shaped corridors until one region is left."""
    for _ in range(max_links):
        open_cells = g.open_cells()
        if not open_cells:
            raise Reject("no floor left")
        main = g.dist(open_cells[0])
        rest = [cell for cell in open_cells if cell not in main]
        if not rest:
            return
        b = rng.choice(rest)
        a = min(main, key=lambda cell: abs(cell[0] - b[0]) + abs(cell[1] - b[1]) + rng.random())
        corner = (b[0], a[1]) if rng.random() < 0.5 else (a[0], b[1])
        for (x0, y0), (x1, y1) in ((a, corner), (corner, b)):
            for c in range(min(x0, x1), max(x0, x1) + 1):
                for r in range(min(y0, y1), max(y0, y1) + 1):
                    if g.at(c, r) == WALL:
                        g.put(c, r, FLOOR)
    g.keep_largest_region()


def mirror(g: Grid, rng: random.Random) -> None:
    """Reflect the left half onto the right half, then join the halves: a symmetric plan of the kind
    people draw by hand (a hub with matching wings). If the left half is solid, the right half is
    the one that is kept."""
    half = g.w // 2
    if all(ch == WALL for row in g.rows for ch in row[:half]):
        for row in g.rows:
            row.reverse()
    for row in g.rows:
        row[g.w - half:] = row[:half][::-1]
    link_regions(g, rng)


# ---- carvers ---------------------------------------------------------------------------------------
def carve_strip(rng: random.Random, w: int, h: int) -> Grid:
    """One corridor, 1 to 3 cells wide, cut into chambers by walls with a single opening. The start is
    at one end and the exit, if the tier has one, at the other (the shape of the reference test maps)."""
    length, width = rng.randint(5, max(6, w)), rng.choice((1, 1, 1, 2, 3))
    g = Grid(length + 2, width + 2)
    g.carve(1, 1, length, width)
    x = rng.randint(3, 5)
    while x <= length - 2:
        g.carve(x, 1, x, width, WALL)
        g.put(x, rng.randint(1, width), DOOR if rng.random() < 0.7 else FLOOR)
        x += rng.randint(3, 6)
    g.put(1, rng.randint(1, width), START)
    g.put(length, rng.randint(1, width), EXIT)
    if rng.random() < 0.5:                         # the start does not always face down the corridor
        g.rows = [row[::-1] for row in g.rows]
    if rng.random() < 0.3:
        g.rows = [list(col) for col in zip(*g.rows)]
        g.w, g.h = g.h, g.w
    return g


def carve_arena(rng: random.Random, w: int, h: int) -> Grid:
    """One large room in one of several shapes, with pillars for cover and a few side closets."""
    g = Grid(w, h)
    m = rng.choice((1, 1, 1, 2, 3)) if min(w, h) >= 10 else 1
    x0, y0, x1, y1 = m, m, w - 1 - m, h - 1 - m
    g.carve(x0, y0, x1, y1)
    rw, rh = x1 - x0 + 1, y1 - y0 + 1
    shape = rng.choice(("rect", "rect", "cut", "cross", "ring", "split", "ell"))
    if shape == "cut" and min(rw, rh) >= 5:          # clipped corners
        k = rng.randint(1, max(1, min(rw, rh) // 3))
        for r in range(y0, y1 + 1):
            for c in range(x0, x1 + 1):
                if min(c - x0, x1 - c) + min(r - y0, y1 - r) < k:
                    g.put(c, r, WALL)
    elif shape == "cross" and min(rw, rh) >= 6:
        a, b = rng.randint(1, rw // 3), rng.randint(1, rh // 3)
        for cx in (x0, x1 - a + 1):
            for cy in (y0, y1 - b + 1):
                g.carve(cx, cy, cx + a - 1, cy + b - 1, WALL)
    elif shape == "ring" and min(rw, rh) >= 6:
        t = rng.randint(2, max(2, min(rw, rh) // 2 - 1))
        if x0 + t <= x1 - t and y0 + t <= y1 - t:
            g.carve(x0 + t, y0 + t, x1 - t, y1 - t, WALL)
    elif shape == "split" and rw >= 7:               # a wall across the room with one or two gaps
        x = rng.randint(x0 + 3, x1 - 3)
        g.carve(x, y0, x, y1, WALL)
        for _ in range(rng.choice((1, 2))):
            r = rng.randint(y0, y1)
            g.carve(x, r, x, min(y1, r + rng.choice((0, 1, 2))))
    elif shape == "ell" and min(rw, rh) >= 5:
        g.carve(x0, y0, x0 + rng.randint(1, rw // 2) - 1, y0 + rng.randint(1, rh // 2) - 1, WALL)
    cover = rng.choice(("none", "lattice", "random", "random", "bars"))
    if cover == "lattice":
        px, py = rng.choice((2, 3, 4)), rng.choice((2, 3, 4))
        for r in range(y0 + 1, y1, py):
            for c in range(x0 + 1, x1, px):
                if all(g.at(c + i, r + j) != WALL for i in (-1, 0, 1) for j in (-1, 0, 1)):
                    g.put(c, r, WALL)
    elif cover in ("random", "bars"):
        for _ in range(rng.randint(1, max(1, rw * rh // 14))):
            c, r = rng.randint(x0 + 1, max(x0 + 1, x1 - 1)), rng.randint(y0 + 1, max(y0 + 1, y1 - 1))
            dc, dr = rng.choice(((1, 0), (0, 1)))
            cells = [(c + i * dc, r + i * dr) for i in range(rng.randint(2, 4) if cover == "bars" else 1)]
            ring = {(x + i, y + j) for x, y in cells for i in (-1, 0, 1) for j in (-1, 0, 1)}
            if all(g.at(*cell) == FLOOR for cell in ring):
                for cell in cells:
                    g.put(*cell, WALL)
    g.keep_largest_region()
    add_stubs(g, rng, rng.choice((0, 0, 1, 2, 3)), lengths=(1, 3))
    return g


def carve_house(rng: random.Random, w: int, h: int) -> Grid:
    """Rooms that tile the map and share walls, like a floor plan drawn by hand. Walls are 1 or 2
    cells thick; neighbouring rooms are joined by single openings, wide gaps or removed walls."""
    g = Grid(w, h)
    wall = 2 if rng.random() < 0.2 else 1
    margin = 2 if rng.random() < 0.35 else 1         # a thick outer wall leaves room for alcoves
    min_side = rng.choice((2, 2, 3, 3, 4))
    max_area = rng.choice((12, 20, 30, 42, 64))
    rects: list[tuple[int, int, int, int]] = []

    def split(x0: int, y0: int, x1: int, y1: int) -> None:
        rw, rh = x1 - x0 + 1, y1 - y0 + 1
        can_x, can_y = rw >= 2 * min_side + wall, rh >= 2 * min_side + wall
        if not (can_x or can_y) or (rw * rh <= max_area and rng.random() < 0.7):
            rects.append((x0, y0, x1, y1))
        elif can_x and (not can_y or rng.random() < rw / (rw + rh)):
            k = rng.randint(x0 + min_side, x1 - min_side - wall + 1)
            split(x0, y0, k - 1, y1)
            split(k + wall, y0, x1, y1)
        else:
            k = rng.randint(y0 + min_side, y1 - min_side - wall + 1)
            split(x0, y0, x1, k - 1)
            split(x0, k + wall, x1, y1)

    split(margin, margin, w - 1 - margin, h - 1 - margin)
    if len(rects) > 4 and rng.random() < 0.4:        # leave a few plots solid
        rects = [rect for rect in rects if rng.random() > 0.15]
    connect_rooms(g, rng, rects, max_len=wall, loops=rng.choice((0, 0, 1, 1, 2, 3)),
                  widen=rng.choice((0.0, 0.0, 0.2, 0.5)), merge=rng.choice((0.0, 0.0, 0.15, 0.3)))
    g.keep_largest_region()
    add_pillars(g, rng, rects, chance=rng.choice((0.0, 0.2, 0.5)))
    g.keep_largest_region()
    return g


def carve_scatter(rng: random.Random, w: int, h: int) -> Grid:
    """Separate rooms of different sizes joined by straight and L-shaped corridors, 1 to 3 cells
    wide, with dead-end branches."""
    g = Grid(w, h)
    rects: list[tuple[int, int, int, int]] = []
    gap = rng.choice((1, 1, 2, 2, 3))
    biggest = rng.choice((4, 5, 6, 7, 8))
    target = max(2, w * h // rng.choice((30, 45, 60)))
    for _ in range(rng.randint(40, 160)):
        rw, rh = rng.randint(2, biggest), rng.randint(2, biggest)
        if rw + 2 > w - 2 or rh + 2 > h - 2:
            continue
        x0, y0 = rng.randint(1, w - 1 - rw), rng.randint(1, h - 1 - rh)
        x1, y1 = x0 + rw - 1, y0 + rh - 1
        if all(x1 + gap < a0 or a1 + gap < x0 or y1 + gap < b0 or b1 + gap < y0 for a0, b0, a1, b1 in rects):
            rects.append((x0, y0, x1, y1))
        if len(rects) >= target:
            break
    if len(rects) < 2:
        raise Reject("no space for rooms")
    connect_rooms(g, rng, rects, max_len=rng.choice((4, 6, 9)), loops=rng.choice((0, 0, 1, 2)),
                  widen=rng.choice((0.0, 0.0, 0.3, 0.7)))
    link_regions(g, rng)
    add_pillars(g, rng, rects, chance=rng.choice((0.0, 0.3, 0.6)))
    g.keep_largest_region()
    add_gates(g, rng, rng.choice((0, 0, 2, 4)))
    add_stubs(g, rng, rng.choice((0, 1, 2, 3, 5)))
    return g


def carve_maze(rng: random.Random, w: int, h: int) -> Grid:
    """Winding corridors 1 to 3 cells wide on a lattice: a random spanning tree with optional loops,
    thinned dead ends and a few open rooms."""
    width = rng.choice((1, 1, 1, 2, 2, 3))
    step = width + 1
    nx, ny = (w - 1) // step, (h - 1) // step
    if nx < 2 or ny < 2:
        raise Reject("too small for a maze")
    g = Grid(1 + nx * step, 1 + ny * step)
    branch = rng.choice((0.0, 0.05, 0.2, 0.5))       # 0 = long corridors, higher = many short branches
    first = (rng.randrange(nx), rng.randrange(ny))
    alive, active, links = {first}, [first], set()

    def lattice(i: int, j: int):
        return [(a, b) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if 0 <= a < nx and 0 <= b < ny]

    while active:
        k = len(active) - 1 if rng.random() >= branch else rng.randrange(len(active))
        fresh = [n for n in lattice(*active[k]) if n not in alive]
        if not fresh:
            active.pop(k)
            continue
        nxt = rng.choice(fresh)
        links.add((min(active[k], nxt), max(active[k], nxt)))
        alive.add(nxt)
        active.append(nxt)
    walls = sorted({(min(a, b), max(a, b)) for a in alive for b in lattice(*a)} - links)
    rng.shuffle(walls)
    links |= set(walls[: int(len(walls) * rng.choice((0.0, 0.0, 0.04, 0.1, 0.2)))])
    rooms = []                                         # blocks of lattice nodes opened into rooms
    for _ in range(rng.choice((0, 0, 1, 2, 3))):
        a, b = rng.randint(2, 3), rng.randint(2, 3)
        if a <= nx and b <= ny:
            i, j = rng.randint(0, nx - a), rng.randint(0, ny - b)
            rooms.append((i, j, i + a - 1, j + b - 1))
    kept = {(i, j) for i0, j0, i1, j1 in rooms for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}
    kept.add(first)
    for _ in range(rng.choice((0, 0, 1, 2, 4))):      # thin out dead ends
        degree: dict = {}
        for a, b in links:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        for node in sorted(alive):
            if degree.get(node, 0) == 1 and node not in kept and rng.random() < 0.7 and len(alive) > 6:
                alive.discard(node)
                links = {link for link in links if node not in link}
    for i, j in alive:
        g.carve(1 + i * step, 1 + j * step, i * step + width, j * step + width)
    for (i, j), (a, b) in links:
        g.carve(1 + i * step, 1 + j * step, a * step + width, b * step + width)
    for i0, j0, i1, j1 in rooms:
        g.carve(1 + i0 * step, 1 + j0 * step, i1 * step + width, j1 * step + width)
    if width > 1:                                      # wide corridors get a few narrow doorways
        add_gates(g, rng, rng.choice((0, 2, 4, 8)))
    return g


def carve_cave(rng: random.Random, w: int, h: int) -> Grid:
    """Irregular open space from a cellular automaton (walls grow where most neighbours are wall)."""
    fill = rng.uniform(0.36, 0.46)
    solid = [[not (0 < c < w - 1 and 0 < r < h - 1) or rng.random() < fill for c in range(w)] for r in range(h)]
    for _ in range(rng.choice((2, 3, 4))):
        solid = [[not (0 < c < w - 1 and 0 < r < h - 1)
                  or sum(solid[r + j][c + i] for i in (-1, 0, 1) for j in (-1, 0, 1)) >= 5
                  for c in range(w)] for r in range(h)]
    g = Grid(w, h)
    for r in range(h):
        for c in range(w):
            if not solid[r][c]:
                g.put(c, r, FLOOR)
    if not g.open_cells():
        raise Reject("cave closed up")
    g.keep_largest_region()
    if len(g.open_cells()) < max(12, w * h // 6):
        raise Reject("cave too small")
    add_gates(g, rng, rng.choice((0, 1, 3)))
    add_stubs(g, rng, rng.choice((0, 0, 1, 2)))
    return g


CARVERS = {"strip": carve_strip, "arena": carve_arena, "house": carve_house,
           "scatter": carve_scatter, "maze": carve_maze, "cave": carve_cave}


# ---- generic passes --------------------------------------------------------------------------------
def place_start(g: Grid, rng: random.Random) -> tuple[int, int]:
    """Choose the player start: usually inside a room, sometimes at the end of a corridor."""
    placed = g.cells(START)
    if placed:
        return placed[0]
    floor = g.cells(FLOOR)
    rooms = [cell for cell in floor if g.roomy(*cell)]
    ends = [cell for cell in floor if len(g.exits_of(*cell)) == 1]
    cell = rng.choice(ends if ends and rng.random() < 0.25 else rooms or floor)
    g.put(*cell, START)
    return cell


def _exit_candidates(g: Grid, style: str, dist: dict, need: int) -> list[tuple[int, int]]:
    """Cells where a pad of the given style can go, at least `need` cells of walking from the start."""
    out = []
    if style == "alcove":                              # a niche cut into the wall of a room
        for r in range(1, g.h - 1):
            for c in range(1, g.w - 1):
                if g.at(c, r) == WALL:
                    near = g.exits_of(c, r)
                    if len(near) == 1 and g.at(*near[0]) == FLOOR and g.roomy(*near[0]) and dist.get(near[0], -1) + 1 >= need:
                        out.append((c, r))
        return out
    for c, r in g.cells(FLOOR):
        if dist.get((c, r), -1) < need:
            continue
        near = g.exits_of(c, r)
        if style == "end":
            ok = len(near) == 1
        elif style == "open":
            ok = all(g.at(c + i, r + j) != WALL for i in (-1, 0, 1) for j in (-1, 0, 1))
        elif style == "wall":
            ok = len(near) == 3 and g.roomy(c, r)
        else:                                          # corner of a room
            ok = len(near) == 2 and near[0][0] != near[1][0] and near[0][1] != near[1][1] and g.roomy(c, r)
        if ok:
            out.append((c, r))
    return out


def place_exit(g: Grid, rng: random.Random, start, style: str, far: float) -> tuple[tuple[int, int], str]:
    """Put the exit pad down in the wanted style, far from the start. Falls back to the other styles.
    Returns (cell, style used). The pad never cuts any part of the map off."""
    placed = g.cells(EXIT)
    if placed:
        return placed[0], "end"
    dist = g.dist(start)
    need = max(3, int(far * max(dist.values())))
    others = [s for s in EXIT_STYLES if s != style]
    rng.shuffle(others)
    for attempt in [style] + others:
        cands = _exit_candidates(g, attempt, dist, need)
        if attempt == "end" and not cands:             # no dead end yet: grow one
            bases = [cell for cell in g.cells(FLOOR) if dist.get(cell, -1) >= need - 2]
            rng.shuffle(bases)
            for base in bases[:30]:
                stub = grow_stub(g, rng, base, lengths=(2, 4))
                if stub:
                    cands = [stub[-1]]
                    break
        rng.shuffle(cands)
        for cell in cands[:12]:
            old = g.at(*cell)
            g.put(*cell, EXIT)
            if g.connected(start):
                return cell, attempt
            g.put(*cell, old)
    raise Reject("no place for the exit")


def place_locks(g: Grid, rng: random.Random, start, exit_cell, count: int) -> list[dict]:
    """Turn up to `count` chokepoints of the route into locked doors and hide a key for each.

    A chokepoint is a 1-wide passage cell that every route from the start to the exit crosses. The
    key of a lock is placed in the part of the map that can be reached without opening that lock,
    preferably past the previous lock (so locks must be opened in order), in a dead end, and
    farther from the entrance of that part than the locked door is (so the door is met first).
    """
    route = g.path(start, exit_cell)
    size = {}                                           # chokepoint -> cells reachable while it is shut
    for cell in route[2:-2]:
        if g.doorway(*cell):
            zone = g.dist(start, blocked={cell})
            if exit_cell not in zone and len(zone) >= MIN_LOCK_ZONE:
                size[cell] = len(zone)
    chosen: list[tuple[int, int]] = []
    for cell in rng.sample(sorted(size), len(size)):   # every lock opens up a new part of the map
        if len(chosen) < count and all(abs(size[cell] - size[other]) >= MIN_LOCK_ZONE for other in chosen):
            chosen.append(cell)
    chosen.sort(key=route.index)
    colours = rng.sample(("red", "blue", "yellow"), len(chosen))
    locks, before = [], set()
    for j, (cell, colour) in enumerate(zip(chosen, colours)):
        zone = g.dist(start, blocked={cell})                       # reachable while this lock is shut
        entry = start if j == 0 else chosen[j - 1]
        from_entry = g.dist(entry, blocked={cell})
        fresh = [c for c in zone if c not in before] if before and rng.random() < 0.75 else list(zone)
        spots = [c for c in fresh if g.at(*c) == FLOOR and c not in chosen and c in from_entry]
        if not spots:
            spots = [c for c in zone if g.at(*c) == FLOOR and c not in chosen and c in from_entry]
        if not spots:
            raise Reject("no place for a key")
        door_at = min(from_entry[n] for n in g.around(*cell) if n in from_entry) + 1
        beyond = [c for c in spots if from_entry[c] > door_at]
        pool = beyond if beyond and rng.random() < 0.85 else spots
        pool.sort(key=lambda c: (len(g.exits_of(*c)) != 1, -from_entry[c], c))
        key_cell = rng.choice(pool[: max(1, len(pool) // 4)])
        g.put(*key_cell, (SKULL_FOR if rng.random() < 0.25 else CARD_FOR)[colour])
        g.put(*cell, DOOR_FOR[colour])
        locks.append({"door": cell, "key": key_cell, "colour": colour, "door_first": from_entry[key_cell] > door_at})
        before = set(zone)
    return locks


def place_doors(g: Grid, rng: random.Random, density: float) -> int:
    """Turn room entrances (threshold cells) into ordinary doors with probability `density`,
    up to one door per ten open cells."""
    cands = [cell for cell in g.cells(FLOOR) if g.threshold(*cell)]
    rng.shuffle(cands)
    made, most = 0, max(2, len(g.open_cells()) // 10)
    for cell in cands:
        if made < most and rng.random() < density and g.doorway(*cell):   # re-checked: a neighbour may be a door now
            g.put(*cell, DOOR)
            made += 1
    return made


def place_monsters(g: Grid, rng: random.Random, tier: int, start) -> list[tuple[int, int]]:
    """Scatter monsters away from the start. In tiers 3 and 4 some guard a key or the exit, and in
    tier 4 some wait beside doors. A share `MONSTER_FREE` of the tier 2 maps gets none. Returns the
    cells whose monster gets the ambush flag."""
    area = len(g.open_cells())
    if tier == 1:
        count = rng.randint(2, max(3, min(12, area // 9)))
    elif tier == 2:
        count = 0 if rng.random() < MONSTER_FREE[2] else rng.randint(1, min(6, 2 + area // 60))
    elif tier == 3:
        count = rng.randint(1, min(8, 3 + area // 50))
    else:
        count = rng.randint(4, min(16, 6 + area // 30))
    dist = g.dist(start)
    near = 4 if area > 40 else 3
    ambush: list[tuple[int, int]] = []
    budget = [MONSTER_HP_PER_CELL[tier] * area]          # total monster health the map may hold

    def put(cell, flag_chance: float) -> None:
        kinds = [(m, wgt) for m, wgt in MONSTER_MIX[tier]
                 if (m not in WIDE_MONSTERS or g.roomy(*cell)) and (MONSTER_HP[m] <= budget[0] or m == "z")]
        kind = pick(rng, kinds)
        budget[0] -= MONSTER_HP[kind]
        g.put(*cell, kind)
        if rng.random() < flag_chance:
            ambush.append(cell)

    if tier == 4:                                       # beside the far side of a door: unseen until you step in
        for door in g.cells("".join(udmf.DOOR_CHARS)):
            sides = [n for n in g.around(*door) if n in dist and g.is_room_cell(*n)]
            if len(sides) != 2 or count <= 0 or rng.random() >= 0.45:
                continue
            inner = max(sides, key=dist.get)
            dc, dr = inner[0] - door[0], inner[1] - door[1]
            beside = [(inner[0] + s * dr, inner[1] + s * dc) for s in (-1, 1)]
            beside = [s for s in beside if g.at(*s) == FLOOR and dist.get(s, 0) >= near]
            ahead = (inner[0] + dc, inner[1] + dr)
            if beside or (g.at(*ahead) == FLOOR and dist.get(ahead, 0) >= near):
                put(rng.choice(beside) if beside else ahead, 0.6)
                count -= 1
    if tier >= 3:                                       # a guard next to a key or the exit
        for prize in g.cells("".join(udmf.KEY_CHARS) + EXIT):
            posts = [n for n in g.exits_of(*prize) if g.at(*n) == FLOOR and dist.get(n, 0) >= near]
            if posts and count > 0 and rng.random() < GUARD_CHANCE:
                put(rng.choice(posts), 0.2)
                count -= 1
    spots = [cell for cell in g.cells(FLOOR) if dist.get(cell, 0) >= near]
    rooms = [cell for cell in spots if g.roomy(*cell)]
    rng.shuffle(spots)
    rng.shuffle(rooms)
    for _ in range(count):
        pool = spots if spots and rng.random() < 0.1 else rooms or spots
        while pool and g.at(*pool[-1]) != FLOOR:
            pool.pop()
        if pool:
            put(pool.pop(), {3: 0.1, 4: 0.3}.get(tier, 0.0))
    return ambush


def place_pickups(g: Grid, rng: random.Random, tier: int) -> None:
    """Add ammunition and weapons in proportion to the monsters' total health, and some health.
    Tier 4 gets the least; tier 0 gets a few harmless pickups so items appear in navigation maps."""
    floor = g.cells(FLOOR)
    rng.shuffle(floor)
    monsters = [g.at(*cell) for cell in g.cells("".join(udmf.MONSTER_CHARS))]
    wanted: list[str] = []
    if monsters:
        need = TIERS[tier]["ammo"] * sum(MONSTER_HP[m] for m in monsters) / 10 - START_BULLETS
        have, gun = 0.0, False
        while have < need and len(wanted) < 12:
            if not gun and need >= 30 and rng.random() < 0.7:
                item, gun = rng.choice("ffj"), True
            elif need - have >= 40:
                item = "x" if not gun or rng.random() < 0.5 else "t"
            else:
                item = "t" if gun and rng.random() < 0.4 else "n"
            wanted.append(item)
            have += AMMO_VALUE[item]
        if tier < 4 and rng.random() < 0.3:
            wanted.append(rng.choice("nnxt"))
        health = {1: rng.randint(0, 1 + len(monsters) // 2), 2: rng.randint(0, 2),
                  3: rng.randint(0, 3), 4: rng.randint(0, 2)}[tier]
        wanted += rng.choices("hma", (6, 3, 1), k=health)
    elif rng.random() < 0.35:
        wanted += rng.choices("hmanxtfj", k=rng.randint(1, 3))
    for item in wanted:
        if floor:
            g.put(*floor.pop(), item)


# ---- assembly --------------------------------------------------------------------------------------
def count_rooms(g: Grid) -> int:
    """Number of separate open areas at least 2 cells wide (rooms and wide corridors)."""
    seen, rooms = set(), 0
    for cell in g.open_cells():
        if cell in seen or not g.roomy(*cell):
            continue
        rooms += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            for n in g.exits_of(*stack.pop()):
                if n not in seen and g.roomy(*n):
                    seen.add(n)
                    stack.append(n)
    return rooms


def describe(grid: str, options: dict | None = None) -> dict:
    """Counts for the manifest: size, rooms, doors, locks, keys, monsters, pickups, dead ends."""
    rows = udmf.parse_grid(grid)
    g = Grid(len(rows[0]), len(rows))
    g.rows = [list(row) for row in rows]
    flat = "".join(rows)
    sectors = udmf.assign_sectors(rows)[1]
    return {
        "width": g.w, "height": g.h,
        "open_cells": len(flat) - flat.count(WALL),
        "rooms": count_rooms(g),
        "sectors": len(sectors),
        "doors": flat.count(DOOR),
        "locks": sum(flat.count(ch) for ch in udmf.LOCKED_DOOR_CHARS),
        "keys": sum(flat.count(ch) for ch in udmf.KEY_CHARS),
        "monsters": sum(flat.count(ch) for ch in udmf.MONSTER_CHARS),
        "pickups": sum(flat.count(ch) for ch in udmf.PICKUP_CHARS),
        "exits": flat.count(EXIT),
        "dead_ends": sum(len(g.exits_of(*cell)) == 1 for cell in g.open_cells()),
        "ambush": len((options or {}).get("ambush", ())),
    }


def strip_monsters(grid: str) -> str:
    """The same grid with every monster replaced by floor."""
    return "".join(FLOOR if ch in udmf.MONSTER_CHARS else ch for ch in grid)


def without_monsters(m: dict) -> dict:
    """Monster-free copy of a map (a `generate` result, a manifest row or a suite row).

    Every monster cell becomes floor, the build options that only concern monsters are dropped and
    the counts are taken again. Walls, doors, keys, pickups, the route and the tick cap stay those
    of the original, so the copy asks for the same navigation without any combat. Such copies are
    what the navigation stage of training uses and what the `*_nav_*` suites score.
    """
    grid = strip_monsters(m["grid"])
    options = {k: v for k, v in m["options"].items() if k not in MONSTER_OPTIONS}
    return {**m, "grid": grid, "options": options, "counts": {**m["counts"], **describe(grid, options)}}


def tick_cap(rng: random.Random, tier: int, sol: Solution) -> int:
    """Tick cap between 2100 and 4200 in steps of 350: a tier-typical value, raised until it is a
    random factor between MIN_BUDGET and MAX_BUDGET times the shortest route at walking speed.
    Raises Reject when 4200 tics are less than MIN_BUDGET times the route."""
    cap = pick(rng, TIERS[tier]["ticks"])
    if sol.has_exit:
        route = FRAME_SKIP * sol.optimal_steps
        if MIN_BUDGET * route > MAX_TICKS:
            raise Reject("route too long for the largest tick cap")
        need = rng.uniform(MIN_BUDGET, MAX_BUDGET) * route
        cap = max(cap, MIN_TICKS + TICK_STEP * max(0, -(-(int(need) - MIN_TICKS) // TICK_STEP)))
    return min(cap, MAX_TICKS)


def draft(tier: int, rng: random.Random, lock_counts=None) -> dict:
    """One attempt at a map of the given tier. Raises Reject if the attempt does not work out.
    `lock_counts` replaces the tier's own ((number of locks, weight), ...) when given."""
    spec = TIERS[tier]
    lo, hi = spec["size"]
    w = rng.randint(lo, hi)
    h = rng.randint(max(lo, w // 2), w) if rng.random() < 0.6 else rng.randint(lo, hi)
    style = pick(rng, spec["styles"])
    g = CARVERS[style](rng, w, h)
    if style != "strip":
        if rng.random() < MIRROR_CHANCE:
            mirror(g, rng)
        g = g.transformed(rng)
    if len(g.open_cells()) < spec["min_open"]:
        raise Reject("too little floor")
    start = place_start(g, rng)
    has_exit = rng.random() < spec["exit_chance"]
    locks: list[dict] = []
    exit_style = None
    if has_exit:
        exit_cell, exit_style = place_exit(g, rng, start, rng.choice(EXIT_STYLES),
                                           far=rng.choice((0.3, 0.5, 0.5, 0.7, 0.9)))
        want = pick(rng, lock_counts or spec["locks"])
        if want:
            locks = place_locks(g, rng, start, exit_cell, want)
            if len(locks) < min(want, 2 if tier == 4 else 1):
                raise Reject("not enough chokepoints for the locks")
    else:
        for cell in g.cells(EXIT):                       # a strip without an exit
            g.put(*cell, FLOOR)
    if style != "strip":
        place_doors(g, rng, density=rng.choice(spec["doors"]))
    g = g.trimmed()                                      # cell coordinates are final from here on
    start = g.cells(START)[0]
    ambush = place_monsters(g, rng, tier, start) if tier > 0 else []
    place_pickups(g, rng, tier)

    grid = g.text()
    sol = solve(grid)
    if not sol.solvable:
        raise Reject("not solvable")
    if locks and not sol.keys_needed:
        raise Reject("locks can be bypassed")
    if has_exit and sol.path_cells < spec["min_route"]:
        raise Reject("exit too close to the start")
    options = {"ambush": [f"{c},{r}" for c, r in sorted(ambush)]} if ambush else {}
    counts = describe(grid, options)
    if tier >= 2 and counts["rooms"] + counts["doors"] + counts["dead_ends"] < 3:
        raise Reject("one plain room is not an exploration map")
    if counts["monsters"] < MIN_MONSTERS[tier]:
        raise Reject("no room for the monsters")
    counts.update(route_cells=sol.path_cells, route_corners=sol.corners, route_doors=sol.doors_on_path,
                  optimal_steps=sol.optimal_steps,
                  doors_met_before_key=sum(lock["door_first"] for lock in locks))
    return {"grid": grid, "tier": tier, "style": style, "exit_style": exit_style,
            "episode_ticks": tick_cap(rng, tier, sol), "options": options, "counts": counts}


def has_key_stage(tier: int, seed: int) -> bool:
    """Whether map (tier, seed) is a tier 0 map with a key and a locked door: one tier 0 seed in
    `KEY_STAGE_EVERY` (seeds 2, 5, 8, ...)."""
    return tier == 0 and seed % KEY_STAGE_EVERY == KEY_STAGE_EVERY - 1


def generate(tier: int, seed: int) -> dict:
    """The map of (tier, seed) in the reference dialect: grid, tick cap, build options and counts.
    Deterministic: the same arguments always give the same map."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier}")
    lock_counts = KEY_STAGE_LOCKS if has_key_stage(tier, seed) else None
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(f"mapgen-v1/{tier}/{seed}/{attempt}")
        try:
            out = draft(tier, rng, lock_counts)
        except Reject:
            continue
        out["seed"] = seed
        out["attempts"] = attempt + 1
        return out
    raise RuntimeError(f"no valid map for tier {tier}, seed {seed} in {MAX_ATTEMPTS} attempts")
