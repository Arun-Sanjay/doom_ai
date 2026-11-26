"""ASCII grid -> UDMF PWAD compiler. Training-time scaffolding, never imported by the agent.

A map is drawn as rows of characters, one character per 64-unit cell::

    ##########
    #P..D...E#
    ##########

`build_wad(grid)` returns the bytes of a PWAD holding `MAP01`, `TEXTMAP`, `ENDMAP` (UDMF, namespace
"zdoom", no node lumps: the engine builds nodes on load). With no options the output is the grid
dialect of the reference maps in `tests/fixtures/` and `maps/two_rooms.wad`, byte for byte.

Geometry rules
--------------
* Cell (col, row) of a grid with H rows covers x in [64*col, 64*col + 64] and
  y in [64*(H-1-row), 64*(H-row)]; row 0 is the top (north) row. Things sit at cell centres.
* Sectors are numbered in row-major order of their first cell. Every 4-connected region of open
  floor is one sector; every door cell and every exit cell is a sector of its own.
* Cells are visited in row-major order and their four edges in the order west, east, north, south.
  An edge to a wall is a one-sided line. An edge between two sectors is emitted once, by the cell of
  the lower sector index, as a two-sided line whose front side is the room and whose back side is
  the door or exit (door specials with tag 0 act on the back sector).
* Unit edges on the same grid line that share side and sectors are merged into the longest possible
  lines. Groups are written in order of first appearance, the runs inside a group by ascending
  coordinate. Vertices are numbered by first appearance in the line list, sidedefs follow the lines
  (front, then back), things follow the row-major scan.

The legend (`LEGEND`, also `legend_table()` as plain data) is a superset of the reference symbols
``# . P E D R B r z i c h``.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CELL = 64                      # map units per grid cell
CEILING_HEIGHT = 128
DEFAULT_LIGHT = 192
DEFAULT_DOOR_DELAY = 150       # tics an ordinary door stays open before it closes again
DOOR_SPEED = 16

# (floor flat, wall texture), chosen by sector index modulo the list length. Rooms use both; an exit
# cell uses the wall texture of its own index and the exit flat as floor. The reference maps show
# the first three floors and the four walls; the fourth floor is our choice.
DEFAULT_THEMES: tuple[tuple[str, str], ...] = (
    ("FLOOR0_1", "STARTAN2"),
    ("FLOOR4_8", "STONE2"),
    ("FLAT14", "BROWN1"),
    ("FLAT1", "GRAY1"),
)
DEFAULT_CEILING = "CEIL1_1"
DOOR_FLOOR = "FLAT5_4"
DOOR_TRACK = "DOORTRAK"
EXIT_FLOOR = "GATE3"

# ZDoom lock numbers that accept the card or the skull of one colour.
LOCK_RED, LOCK_BLUE, LOCK_YELLOW = 129, 130, 131
DOOR_FACE = {0: "BIGDOOR2", LOCK_RED: "DOORRED", LOCK_BLUE: "DOORBLU", LOCK_YELLOW: "DOORYEL"}


@dataclass(frozen=True)
class Symbol:
    """One legend entry.

    kind: "wall", "floor", "start", "exit", "door" or "thing".
    thing: Doom editor number placed at the cell centre (None for wall, floor, exit and doors).
    lock: ZDoom lock number of a locked door (0 = ordinary door), None for everything else.
    group: "monster", "key", "health", "armor", "ammo" or "weapon" for things.
    color: "red", "blue" or "yellow" for keys and locked doors.
    """

    char: str
    kind: str
    name: str
    thing: int | None = None
    lock: int | None = None
    group: str | None = None
    color: str | None = None


_SYMBOLS = (
    # structure (reference symbols)
    Symbol("#", "wall", "wall"),
    Symbol(".", "floor", "floor"),
    Symbol("P", "start", "player start", thing=1),
    Symbol("E", "exit", "exit pad"),
    Symbol("D", "door", "door", lock=0),
    Symbol("R", "door", "red locked door", lock=LOCK_RED, color="red"),
    Symbol("B", "door", "blue locked door", lock=LOCK_BLUE, color="blue"),
    Symbol("Y", "door", "yellow locked door", lock=LOCK_YELLOW, color="yellow"),
    # keys: the lower-case letter opens the upper-case door; digits are the skull keys
    Symbol("r", "thing", "red keycard", thing=13, group="key", color="red"),
    Symbol("b", "thing", "blue keycard", thing=5, group="key", color="blue"),
    Symbol("y", "thing", "yellow keycard", thing=6, group="key", color="yellow"),
    Symbol("1", "thing", "red skull key", thing=38, group="key", color="red"),
    Symbol("2", "thing", "blue skull key", thing=40, group="key", color="blue"),
    Symbol("3", "thing", "yellow skull key", thing=39, group="key", color="yellow"),
    # monsters
    Symbol("z", "thing", "zombieman", thing=3004, group="monster"),
    Symbol("s", "thing", "shotgun guy", thing=9, group="monster"),
    Symbol("c", "thing", "chaingunner", thing=65, group="monster"),
    Symbol("i", "thing", "imp", thing=3001, group="monster"),
    Symbol("d", "thing", "demon", thing=3002, group="monster"),
    Symbol("g", "thing", "spectre", thing=58, group="monster"),
    Symbol("l", "thing", "lost soul", thing=3006, group="monster"),
    Symbol("o", "thing", "cacodemon", thing=3005, group="monster"),
    Symbol("k", "thing", "hell knight", thing=69, group="monster"),
    # pickups
    Symbol("h", "thing", "stimpack", thing=2011, group="health"),
    Symbol("m", "thing", "medikit", thing=2012, group="health"),
    Symbol("a", "thing", "green armor", thing=2018, group="armor"),
    Symbol("n", "thing", "ammo clip", thing=2007, group="ammo"),
    Symbol("x", "thing", "box of bullets", thing=2048, group="ammo"),
    Symbol("t", "thing", "shotgun shells", thing=2008, group="ammo"),
    Symbol("f", "thing", "shotgun", thing=2001, group="weapon"),
    Symbol("j", "thing", "chaingun", thing=2002, group="weapon"),
)
LEGEND: dict[str, Symbol] = {s.char: s for s in _SYMBOLS}

WALL = "#"
DOOR_CHARS = frozenset(c for c, s in LEGEND.items() if s.kind == "door")
LOCKED_DOOR_CHARS = frozenset(c for c, s in LEGEND.items() if s.kind == "door" and s.lock)
KEY_CHARS = frozenset(c for c, s in LEGEND.items() if s.group == "key")
MONSTER_CHARS = frozenset(c for c, s in LEGEND.items() if s.group == "monster")
PICKUP_CHARS = frozenset(c for c, s in LEGEND.items() if s.group in ("health", "armor", "ammo", "weapon"))
KEY_FOR_DOOR = {"R": "red", "B": "blue", "Y": "yellow"}


def legend_table() -> list[dict]:
    """The legend as plain data (JSON-serialisable), one dict per symbol."""
    return [asdict(s) for s in _SYMBOLS]


# ---- grid helpers -----------------------------------------------------------------------------------
def parse_grid(grid: str) -> list[str]:
    """Split a grid string into rows of equal width; short rows are padded with wall."""
    rows = grid.split("\n")
    while rows and rows[-1] == "":
        rows.pop()
    if not rows:
        raise ValueError("empty grid")
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, WALL) for r in rows]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch not in LEGEND:
                raise ValueError(f"unknown symbol {ch!r} at column {c}, row {r}")
    if sum(row.count("P") for row in rows) != 1:
        raise ValueError("a grid needs exactly one player start 'P'")
    return rows


def cell_centre(col: int, row: int, height: int) -> tuple[float, float]:
    """Map coordinates of the centre of cell (col, row) in a grid with `height` rows."""
    return CELL * col + CELL / 2, CELL * (height - 1 - row) + CELL / 2


def _cell_key(key) -> tuple[int, int]:
    """Accept (col, row) or the string "col,row" (the JSON form used by the suite files)."""
    if isinstance(key, str):
        c, r = key.split(",")
        return int(c), int(r)
    c, r = key
    return int(c), int(r)


def _num(v: float) -> str:
    return repr(float(v))


# ---- sectors ----------------------------------------------------------------------------------------
def assign_sectors(rows: Sequence[str]) -> tuple[list[list[int]], list[tuple[str, str]]]:
    """Sector index per cell (-1 for wall) and, per sector, (kind, symbol of its first cell).

    kind is "room", "door" or "exit". Regions are 4-connected and numbered in row-major order.
    """
    h, w = len(rows), len(rows[0])
    sector = [[-1] * w for _ in range(h)]
    kinds: list[tuple[str, str]] = []
    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch == WALL or sector[r][c] >= 0:
                continue
            kind = LEGEND[ch].kind
            index = len(kinds)
            if kind in ("door", "exit"):
                sector[r][c] = index
                kinds.append((kind, ch))
                continue
            kinds.append(("room", ch))
            stack = [(c, r)]
            sector[r][c] = index
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h and sector[ny][nx] < 0:
                        if rows[ny][nx] != WALL and LEGEND[rows[ny][nx]].kind not in ("door", "exit"):
                            sector[ny][nx] = index
                            stack.append((nx, ny))
    return sector, kinds


# Edge sides in scan order with the neighbour offset (dcol, drow) and the opposite side.
_SIDES = (("W", -1, 0, "E"), ("E", 1, 0, "W"), ("N", 0, -1, "S"), ("S", 0, 1, "N"))


def _edge_slot(side: str, col: int, row: int, height: int) -> tuple[int, int]:
    """(grid line, position along it) of one side of cell (col, row), in map cells with y up."""
    y = height - 1 - row
    if side == "W":
        return col, y
    if side == "E":
        return col + 1, y
    if side == "N":
        return y + 1, col
    return y, col


def _segment(side: str, line: int, lo: int, hi: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """End points of the merged run [lo, hi] on a grid line. The front (room) side is on the right."""
    a, b, k = CELL * lo, CELL * (hi + 1), CELL * line
    if side == "W":
        return (k, a), (k, b)      # north
    if side == "E":
        return (k, b), (k, a)      # south
    if side == "N":
        return (a, k), (b, k)      # east
    return (b, k), (a, k)          # west


def build_textmap(
    grid: str,
    *,
    start_angle: int = 0,
    start_offset: tuple[float, float] = (0.0, 0.0),
    thing_angles: Mapping | None = None,
    ambush: Iterable = (),
    light: int | Sequence[int] = DEFAULT_LIGHT,
    themes: Sequence[Sequence[str]] | None = None,
    ceiling: str = DEFAULT_CEILING,
    exit_floor: str = EXIT_FLOOR,
    exit_wall: str | None = None,
    door_delay: int = DEFAULT_DOOR_DELAY,
) -> str:
    """Compile a grid to UDMF text. See `build_wad` for the options."""
    rows = parse_grid(grid)
    h, w = len(rows), len(rows[0])
    sector, kinds = assign_sectors(rows)
    themes = [tuple(t) for t in (themes or DEFAULT_THEMES)]
    lights = [int(light)] if isinstance(light, (int, float)) else [int(v) for v in light]
    angles = {_cell_key(k): int(v) for k, v in (thing_angles or {}).items()}
    deaf = {_cell_key(k) for k in ambush}

    # ---- unit edges, grouped for merging ------------------------------------------------------
    groups: dict[tuple[str, int, int, int], list[int]] = {}
    for r in range(h):
        for c in range(w):
            s = sector[r][c]
            if s < 0:
                continue
            for side, dc, dr, opposite in _SIDES:
                nc, nr = c + dc, r + dr
                t = sector[nr][nc] if 0 <= nc < w and 0 <= nr < h else -1
                if t < 0:
                    line, pos = _edge_slot(side, c, r, h)
                    groups.setdefault((side, line, s, -1), []).append(pos)
                elif t > s:
                    here, there = kinds[s][0], kinds[t][0]
                    if here != "room" and there != "room" and not (here == there == "exit"):
                        raise ValueError(f"door cell at column {c}, row {r} touches another door or an exit")
                    if here == "room" or there != "room":      # this cell is the front (room) side
                        line, pos = _edge_slot(side, c, r, h)
                        groups.setdefault((side, line, s, t), []).append(pos)
                    else:                                       # the neighbour is the room: flip the line
                        line, pos = _edge_slot(opposite, nc, nr, h)
                        groups.setdefault((opposite, line, t, s), []).append(pos)

    # ---- merged lines -------------------------------------------------------------------------
    lines: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []   # (p1, p2, front, back)
    for (side, line, front, back), slots in groups.items():
        slots.sort()
        lo = prev = slots[0]
        for pos in slots[1:] + [None]:
            if pos is not None and pos == prev + 1:
                prev = pos
                continue
            p1, p2 = _segment(side, line, lo, prev)
            lines.append((p1, p2, front, back))
            if pos is not None:
                lo = prev = pos

    # ---- text -----------------------------------------------------------------------------------
    vertex_index: dict[tuple[int, int], int] = {}
    for p1, p2, _, _ in lines:
        for p in (p1, p2):
            vertex_index.setdefault(p, len(vertex_index))
    out = ['namespace = "zdoom";']
    for (x, y) in vertex_index:
        out.append(f"vertex {{ x = {_num(x)}; y = {_num(y)}; }}")

    walls: list[str] = []          # wall texture per sector
    for i, (kind, ch) in enumerate(kinds):
        if kind == "room":
            floor, wall = themes[i % len(themes)]
            top = CEILING_HEIGHT
        elif kind == "door":
            floor, wall, top = DOOR_FLOOR, DOOR_TRACK, 0
        else:
            floor, wall, top = exit_floor, exit_wall or themes[i % len(themes)][1], CEILING_HEIGHT
        walls.append(wall)
        out.append(f'sector {{ texturefloor = "{floor}"; textureceiling = "{ceiling}"; heightfloor = 0; '
                   f"heightceiling = {top}; lightlevel = {lights[i % len(lights)]}; }}")

    def side_text(sec: int, top: str = "-", middle: str = "-") -> str:
        return (f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{middle}"; '
                'texturebottom = "-"; }')

    sides: list[str] = []
    linedefs: list[str] = []
    for p1, p2, front, back in lines:
        head = f"linedef {{ v1 = {vertex_index[p1]}; v2 = {vertex_index[p2]}; sidefront = {len(sides)}; "
        if back < 0:
            sides.append(side_text(front, middle=walls[front]))
            linedefs.append(head + "blocking = true; dontpegbottom = true; }")
            continue
        head += f"sideback = {len(sides) + 1}; twosided = true; "
        kind, ch = kinds[back]
        if kind == "door":
            lock = LEGEND[ch].lock
            sides.append(side_text(front, top=DOOR_FACE[lock]))
            if lock:
                action = (f"special = 13; arg0 = 0; arg1 = {DOOR_SPEED}; arg2 = 0; arg3 = {lock}; "
                          "playeruse = true; monsteruse = false; repeatspecial = true; }")
            else:
                action = (f"special = 12; arg0 = 0; arg1 = {DOOR_SPEED}; arg2 = {int(door_delay)}; "
                          "playeruse = true; monsteruse = true; repeatspecial = true; }")
        else:
            sides.append(side_text(front))
            # A line between two exit cells needs no action; every room-to-exit line ends the level.
            action = "special = 243; playercross = true; }" if kinds[front][0] == "room" else "}"
        sides.append(side_text(back))
        linedefs.append(head + action)
    out += sides + linedefs

    for r in range(h):
        for c in range(w):
            sym = LEGEND[rows[r][c]]
            if sym.thing is None:
                continue
            x, y = cell_centre(c, r, h)
            if sym.kind == "start":
                x, y, angle = x + float(start_offset[0]), y + float(start_offset[1]), int(start_angle)
            else:
                angle = angles.get((c, r), 180)
            flags = "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; "
            if (c, r) in deaf and sym.group == "monster":
                flags += "ambush = true; "
            out.append(f"thing {{ x = {_num(x)}; y = {_num(y)}; angle = {angle}; type = {sym.thing}; {flags}}}")
    return "\n".join(out) + "\n"


def build_wad(grid: str, **opts) -> bytes:
    """Compile an ASCII grid into the bytes of a UDMF PWAD (lumps MAP01, TEXTMAP, ENDMAP).

    Options, all off by default (the default output matches the reference maps):
      start_angle     player start angle in degrees (0 = east, counter-clockwise).
      start_offset    (dx, dy) in map units added to the player start; keep within +-16 so the
                      player (radius 16) stays inside its cell.
      thing_angles    {(col, row) or "col,row": degrees} for individual things (default 180).
      ambush          cells whose monster gets the ambush flag (it ignores sound until it sees the player).
      light           sector light level, one int or a list indexed by sector number (cycled).
      themes          list of (floor flat, wall texture) for rooms, indexed by sector number (cycled).
      ceiling         ceiling flat of every sector.
      exit_floor      floor flat of the exit cell (GATE3 unless overridden).
      exit_wall       wall texture of the exit cell (default: the theme wall of its sector number).
      door_delay      tics an ordinary door waits before closing (locked doors always stay open).
    """
    text = build_textmap(grid, **opts).encode("ascii")
    directory = b"".join(
        struct.pack("<ii8s", offset, size, name.ljust(8, b"\0"))
        for offset, size, name in ((12, 0, b"MAP01"), (12, len(text), b"TEXTMAP"), (12 + len(text), 0, b"ENDMAP"))
    )
    return b"PWAD" + struct.pack("<ii", 3, 12 + len(text)) + text + directory


def write_wad(grid: str, path, **opts) -> Path:
    """Compile `grid` and write the PWAD to `path` (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_wad(grid, **opts))
    return path


def read_textmap(wad: bytes | str | Path) -> str:
    """Return the TEXTMAP lump of a PWAD given as bytes or as a path."""
    raw = wad if isinstance(wad, (bytes, bytearray)) else Path(wad).read_bytes()
    count, offset = struct.unpack_from("<ii", raw, 4)
    for i in range(count):
        pos, size, name = struct.unpack_from("<ii8s", raw, offset + 16 * i)
        if name.rstrip(b"\0") == b"TEXTMAP":
            return raw[pos:pos + size].decode("ascii")
    raise ValueError("no TEXTMAP lump")
