"""Static, privileged facts about one map, read straight from its WAD.

Training-time scaffolding, never imported by the agent. The scorer, the reward code and the
evaluation tools use it to know what a map contains: where the player starts, where the exits
are, which sectors are doors and which key each needs, what things are placed at skill 3, the
outline of every sector, and (for maps drawn on a 64-unit grid) the cell grid itself.

    meta = load_meta("maps/two_rooms.wad")          # MAP01 by default
    meta.player_start        # (x, y, angle in degrees)
    meta.exits               # [ExitRegion(sector, centre, polygon, lines, trigger)]
    meta.doors               # [Door(sector, lock, colour, centre, polygon, delay_tics)]
    meta.things              # {kind: [Thing(cls, kind, type, x, y, angle, ambush)]} at skill 3
    meta.monsters, meta.keys # shortcuts; meta.things_by_class() gives positions per engine class
    meta.sector_polygons     # {sector index: (K, 4) [x1, y1, x2, y2]}
    meta.grid                # CellGrid for 64-unit grid maps, else None

Supported formats are the ones `doom_env.mapinfo` reads: UDMF (the grid layouts and the
reference maps), Doom binary (Freedoom) and Hexen binary. Door sectors and locks are taken from
`doom_env.mapinfo` so they always agree with what the wrapper reports in rays.

Run `python -m training.mapmeta WAD[:MAP]` to print a summary and the recovered grid.
"""

from __future__ import annotations

import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from doom_env.mapinfo import DOOM_NAMESPACES, load_mapinfo, parse_udmf, udmf_namespace
from doom_env.wad import list_maps, map_lumps, read_wad

MAX_SECTORS = 2048   # ViZDoom reports at most this many sectors; larger maps are skipped
CELL = 64.0          # side of one grid cell in map units
SKILL = 3            # the wrapper always plays "Hurt me plenty"

# Key order of the wrapper's vitals / privileged_state()["keys"].
KEY_CLASSES = ("BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull")
KEY_COLOUR = {"BlueCard": "blue", "YellowCard": "yellow", "RedCard": "red",
              "BlueSkull": "blue", "YellowSkull": "yellow", "RedSkull": "red"}

# Lock number -> colour. doom_env.mapinfo reports binary-format locks as 1/2/3 (a colour), 100
# (any key) and 101 (all keys), and UDMF locks as the raw ZDoom lock number: 1-6 one card or
# skull, 129-131 the card or skull of one colour (what the grid layouts use), 100/228 any key,
# 101/133/134/229 every colour. Locks are handled at the level of colours here.
LOCK_COLOUR = {1: "red", 2: "blue", 3: "yellow", 4: "red", 5: "blue", 6: "yellow",
               129: "red", 130: "blue", 131: "yellow", 100: "any", 228: "any",
               101: "all", 133: "all", 134: "all", 229: "all"}

# Line specials that end the level. Doom numbering: switch (11, 51), walk-over (52, 124),
# gunfire (197, 198). Hexen/ZDoom numbering: Exit_Normal 243, Exit_Secret 244,
# Teleport_NewMap 74, Teleport_EndGame 75.
DOOM_EXIT_SPECIALS = {11: "use", 51: "use", 52: "cross", 124: "cross", 197: "shoot", 198: "shoot"}
HEXEN_EXIT_SPECIALS = (243, 244, 74, 75)

# Editor number -> (engine class name, kind). Kinds: start, monster, key, health, armor, ammo,
# weapon, powerup, barrel. Anything not listed is scenery and gets kind "other".
THING_TABLE: dict[int, tuple[str, str]] = {
    1: ("Player1Start", "start"), 2: ("Player2Start", "start"), 3: ("Player3Start", "start"),
    4: ("Player4Start", "start"), 11: ("DeathmatchStart", "start"),
    3004: ("Zombieman", "monster"), 9: ("ShotgunGuy", "monster"), 65: ("ChaingunGuy", "monster"),
    3001: ("DoomImp", "monster"), 3002: ("Demon", "monster"), 58: ("Spectre", "monster"),
    3006: ("LostSoul", "monster"), 3005: ("Cacodemon", "monster"), 69: ("HellKnight", "monster"),
    3003: ("BaronOfHell", "monster"), 68: ("Arachnotron", "monster"), 71: ("PainElemental", "monster"),
    66: ("Revenant", "monster"), 67: ("Fatso", "monster"), 64: ("Archvile", "monster"),
    7: ("SpiderMastermind", "monster"), 16: ("Cyberdemon", "monster"), 84: ("WolfensteinSS", "monster"),
    72: ("CommanderKeen", "monster"), 88: ("BossBrain", "monster"),
    5: ("BlueCard", "key"), 6: ("YellowCard", "key"), 13: ("RedCard", "key"),
    40: ("BlueSkull", "key"), 39: ("YellowSkull", "key"), 38: ("RedSkull", "key"),
    2011: ("Stimpack", "health"), 2012: ("Medikit", "health"), 2014: ("HealthBonus", "health"),
    2013: ("Soulsphere", "health"),
    2018: ("GreenArmor", "armor"), 2019: ("BlueArmor", "armor"), 2015: ("ArmorBonus", "armor"),
    2007: ("Clip", "ammo"), 2048: ("ClipBox", "ammo"), 2008: ("Shell", "ammo"), 2049: ("ShellBox", "ammo"),
    2010: ("RocketAmmo", "ammo"), 2046: ("RocketBox", "ammo"), 2047: ("Cell", "ammo"), 17: ("CellPack", "ammo"),
    8: ("Backpack", "ammo"),
    2001: ("Shotgun", "weapon"), 82: ("SuperShotgun", "weapon"), 2002: ("Chaingun", "weapon"),
    2003: ("RocketLauncher", "weapon"), 2004: ("PlasmaRifle", "weapon"), 2006: ("BFG9000", "weapon"),
    2005: ("Chainsaw", "weapon"),
    83: ("Megasphere", "powerup"), 2022: ("InvulnerabilitySphere", "powerup"), 2023: ("Berserk", "powerup"),
    2024: ("BlurSphere", "powerup"), 2025: ("RadSuit", "powerup"), 2026: ("Allmap", "powerup"),
    2045: ("Infrared", "powerup"),
    2035: ("ExplosiveBarrel", "barrel"),
}
# ViZDoom object category -> our kind, for the cross-check against the engine's object list.
ENGINE_KIND = {"Monster": "monster", "Key": "key", "Health": "health", "Armor": "armor", "Ammo": "ammo",
               "Weapon": "weapon", "Powerup": "powerup", "Hazard": "barrel"}
CENSUS_KINDS = tuple(ENGINE_KIND.values())
PICKUP_KINDS = ("health", "armor", "ammo", "weapon", "powerup")

# Symbols of the recovered ASCII grid. Terrain and the lower-case things of the base legend keep
# their meaning; the rest is this module's own notation, used for display only.
TERRAIN_SYMBOL = {"wall": "#", "floor": ".", "door": "D", "red": "R", "blue": "B", "yellow": "Y", "locked": "L", "exit": "E"}
THING_SYMBOL = {"Player1Start": "P", "RedCard": "r", "BlueCard": "b", "YellowCard": "y", "RedSkull": "r",
                "BlueSkull": "b", "YellowSkull": "y", "Zombieman": "z", "DoomImp": "i", "ChaingunGuy": "c",
                "Stimpack": "h"}
KIND_SYMBOL = {"monster": "m", "key": "k", "health": "+", "armor": "+", "ammo": "+", "weapon": "+", "powerup": "+",
               "barrel": "o"}


class MapTooLarge(ValueError):
    """The map has more sectors than ViZDoom reports; callers skip such maps."""


# ------------------------------------------------------------------------------------ records
@dataclass(frozen=True)
class Thing:
    """One placed thing that exists at skill 3 in single player."""

    cls: str      # engine class name ("Zombieman", "RedCard", ...); "Thing<number>" when not in THING_TABLE
    kind: str     # start, monster, key, health, armor, ammo, weapon, powerup, barrel or other
    type: int     # editor number
    x: float
    y: float
    angle: float  # degrees, 0 = east, counter-clockwise
    ambush: bool = False


@dataclass(frozen=True)
class ExitRegion:
    """A place that ends the level.

    `sector` is the sector enclosed by exit lines (the exit pad of a grid layout), or -1 when the
    exit is only a line (a switch or a walk-over line of a classic map); then `centre` is the
    midpoint of that line and `polygon` is the line itself.
    """

    sector: int
    centre: tuple[float, float]
    polygon: np.ndarray   # (K, 4) [x1, y1, x2, y2] outline of the region
    lines: np.ndarray     # (M, 4) the lines carrying the exit special
    trigger: str          # "cross", "use" or "shoot"


@dataclass(frozen=True)
class Door:
    """A door sector. `lock` is 0 for an ordinary door, otherwise the lock number."""

    sector: int
    lock: int
    colour: str                    # "" (no lock), "red", "blue", "yellow", "any", "all" or "other"
    centre: tuple[float, float]
    polygon: np.ndarray            # (K, 4) outline
    delay_tics: int | None = None  # tics the door stays open (UDMF only; 0 = stays open; None = unknown)

    @property
    def locked(self) -> bool:
        return self.lock > 0


def lock_colour(lock: int) -> str:
    """Colour name of a lock number ("" for an ordinary door, "other" for an unknown lock)."""
    return "" if lock <= 0 else LOCK_COLOUR.get(int(lock), "other")


def lock_satisfied(lock: int, keys_held) -> bool:
    """Would a player holding `keys_held` (six 0/1 flags in KEY_CLASSES order) pass this lock?

    Works on colours: a red lock accepts the red card or the red skull.
    """
    colour = lock_colour(lock)
    if not colour:
        return True
    held = {KEY_COLOUR[name] for name, flag in zip(KEY_CLASSES, keys_held) if flag}
    if colour == "any":
        return bool(held)
    if colour == "all":
        return held == {"red", "blue", "yellow"}
    return colour in held


# ---------------------------------------------------------------------------- sector lookup
class SectorLocator:
    """Finds the sector that contains a point.

    Built from the `(sector id, (K, 4) segments)` list of `env.sector_polygons()` (or from
    `MapMeta.sector_polygons`). A point is inside a sector when a ray from it crosses the
    sector's outline an odd number of times, which also handles rooms with pillars and sectors
    nested inside other sectors.
    """

    def __init__(self, polygons):
        items = sorted(polygons.items()) if isinstance(polygons, dict) else list(polygons)
        items = [(int(sid), np.asarray(seg, np.float64).reshape(-1, 4)) for sid, seg in items]
        items = [(sid, seg) for sid, seg in items if len(seg)]
        self.ids = np.array([sid for sid, _ in items], np.int64)
        counts = [len(seg) for _, seg in items]
        stops = np.cumsum(counts) if counts else np.zeros(0, np.int64)
        self._span = {sid: (int(stop - n), int(stop)) for (sid, _), n, stop in zip(items, counts, stops)}
        seg = np.concatenate([s for _, s in items]) if items else np.zeros((0, 4))
        self.segments = seg
        self.owner = np.repeat(np.arange(len(items)), counts)
        self._x1, self._y1, self._y2 = seg[:, 0].copy(), seg[:, 1].copy(), seg[:, 3].copy()
        dy = seg[:, 3] - seg[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            self._slope = np.where(dy != 0.0, (seg[:, 2] - seg[:, 0]) / dy, 0.0)  # dx/dy; horizontal lines never cross

    def __len__(self) -> int:
        return len(self.ids)

    def _crossings(self, x: float, y: float, a: int, b: int) -> np.ndarray:
        """Which of segments a..b a ray from (x, y) towards +x crosses."""
        y1, y2 = self._y1[a:b], self._y2[a:b]
        straddle = (y1 > y) != (y2 > y)
        return straddle & (x < self._x1[a:b] + (y - y1) * self._slope[a:b])

    def contains(self, sector: int, x: float, y: float) -> bool:
        span = self._span.get(int(sector))
        return bool(span) and bool(np.count_nonzero(self._crossings(x, y, *span)) & 1)

    def locate(self, x: float, y: float, hint: int = -1) -> int:
        """Sector id containing (x, y), or -1 if it is inside none. `hint` is tried first."""
        if hint >= 0 and self.contains(hint, x, y):
            return int(hint)
        if not len(self.ids):
            return -1
        odd = np.bincount(self.owner[self._crossings(x, y, 0, len(self.owner))], minlength=len(self.ids)) & 1
        hit = np.flatnonzero(odd)
        return int(self.ids[hit[0]]) if len(hit) else -1

    def locate_many(self, points) -> np.ndarray:
        """(N, 2) points -> (N,) sector ids (-1 = none). Used to rasterise grid maps."""
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        out = np.full(len(pts), -1, np.int64)
        px, py = pts[:, 0:1], pts[:, 1:2]
        for sid, (a, b) in self._span.items():
            y1, y2 = self._y1[None, a:b], self._y2[None, a:b]
            cross = ((y1 > py) != (y2 > py)) & (px < self._x1[None, a:b] + (py - y1) * self._slope[None, a:b])
            inside = (cross.sum(axis=1) & 1).astype(bool)
            out[inside & (out < 0)] = sid
        return out

    def distance(self, sector: int, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest point of the sector's outline (inf if unknown)."""
        span = self._span.get(int(sector))
        if not span:
            return float("inf")
        seg = self.segments[span[0]:span[1]]
        return float(point_segment_distance(x, y, seg).min())


def point_segment_distance(x: float, y: float, segments: np.ndarray) -> np.ndarray:
    """Distance from one point to each of (K, 4) segments."""
    p1, d = segments[:, 0:2], segments[:, 2:4] - segments[:, 0:2]
    w = np.array([x, y]) - p1
    length2 = np.maximum((d * d).sum(axis=1), 1e-12)
    t = np.clip((w * d).sum(axis=1) / length2, 0.0, 1.0)
    return np.hypot(*(w - t[:, None] * d).T)


def polygon_centre(segments: np.ndarray) -> tuple[float, float]:
    """Length-weighted mean of the segment midpoints: the centre of a rectangle or regular outline."""
    seg = np.asarray(segments, np.float64).reshape(-1, 4)
    if not len(seg):
        return (0.0, 0.0)
    mid = 0.5 * (seg[:, 0:2] + seg[:, 2:4])
    length = np.maximum(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]), 1e-9)
    c = (mid * length[:, None]).sum(axis=0) / length.sum()
    return (float(c[0]), float(c[1]))


# -------------------------------------------------------------------------------- cell grid
@dataclass(frozen=True)
class CellGrid:
    """The 64-unit cell grid of a grid layout, recovered from the geometry.

    Row 0 is the top row (largest y), as in the ASCII source of a grid layout. The grid is the
    bounding box of the geometry plus a one-cell wall border, so a source grid with exactly one
    border ring is recovered at its original size.
    """

    x0: float              # map x of the left edge of column 0
    y0: float              # map y of the bottom edge of the last row
    sector: np.ndarray     # (H, W) int32: sector index under each cell centre, -1 = solid
    terrain: tuple         # H strings of W symbols from TERRAIN_SYMBOL
    cell: float = CELL

    @property
    def shape(self) -> tuple[int, int]:
        return self.sector.shape

    @property
    def open_cells(self) -> int:
        """Cells the player can stand in (floor, door and exit cells)."""
        return int((self.sector >= 0).sum())

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """(row, col) of a map position; may lie outside the grid."""
        h = self.sector.shape[0]
        return h - 1 - int((y - self.y0) // self.cell), int((x - self.x0) // self.cell)

    def centre(self, row: int, col: int) -> tuple[float, float]:
        h = self.sector.shape[0]
        return self.x0 + (col + 0.5) * self.cell, self.y0 + (h - 1 - row + 0.5) * self.cell

    def sector_at(self, x: float, y: float) -> int:
        """Sector under a map position (-1 for solid cells and positions off the grid)."""
        row, col = self.cell_of(x, y)
        h, w = self.sector.shape
        return int(self.sector[row, col]) if 0 <= row < h and 0 <= col < w else -1

    def ascii(self, things=()) -> str:
        """The grid as text, optionally with things drawn on their cells."""
        rows = [list(r) for r in self.terrain]
        h, w = self.sector.shape
        for th in things:
            symbol = THING_SYMBOL.get(th.cls) or KIND_SYMBOL.get(th.kind)
            row, col = self.cell_of(th.x, th.y)
            if symbol and 0 <= row < h and 0 <= col < w and rows[row][col] == TERRAIN_SYMBOL["floor"]:
                rows[row][col] = symbol
        return "\n".join("".join(r) for r in rows)


def _recover_grid(polygons: dict[int, np.ndarray], doors: list[Door], exits: list[ExitRegion]) -> CellGrid | None:
    """Rasterise sector outlines into cells; None unless every vertex lies on the 64-unit grid."""
    if not polygons:
        return None
    seg = np.concatenate([np.asarray(s, np.float64).reshape(-1, 4) for s in polygons.values()])
    if not len(seg) or np.any(np.mod(seg, CELL) != 0.0):
        return None
    if np.any((seg[:, 0] != seg[:, 2]) & (seg[:, 1] != seg[:, 3])):
        return None  # a diagonal wall: not a cell layout
    x0, y0 = seg[:, [0, 2]].min() - CELL, seg[:, [1, 3]].min() - CELL
    w = int(round((seg[:, [0, 2]].max() - x0) / CELL)) + 1
    h = int(round((seg[:, [1, 3]].max() - y0) / CELL)) + 1
    if w * h > 512 * 512:
        return None
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    centres = np.stack([x0 + (cols + 0.5) * CELL, y0 + (h - 1 - rows + 0.5) * CELL], axis=-1)
    sector = SectorLocator(polygons).locate_many(centres.reshape(-1, 2)).reshape(h, w).astype(np.int32)
    symbol = np.full((h, w), TERRAIN_SYMBOL["floor"], dtype="<U1")
    symbol[sector < 0] = TERRAIN_SYMBOL["wall"]
    for door in doors:
        name = "door" if not door.locked else (door.colour if door.colour in TERRAIN_SYMBOL else "locked")
        symbol[sector == door.sector] = TERRAIN_SYMBOL[name]
    for ex in exits:
        if ex.sector >= 0:
            symbol[sector == ex.sector] = TERRAIN_SYMBOL["exit"]
    return CellGrid(float(x0), float(y0), sector, tuple("".join(r) for r in symbol))


# --------------------------------------------------------------------------------- MapMeta
@dataclass
class MapMeta:
    """Everything static about one map. Instances are shared through a cache: treat them as read-only."""

    wad: str
    map_name: str
    format: str                               # "udmf", "udmf-doom", "hexen" or "doom"
    num_sectors: int
    player_start: tuple[float, float, float]  # x, y, angle in degrees
    exits: list[ExitRegion]
    doors: list[Door]
    things: dict[str, list[Thing]]            # kind -> things placed at skill 3, single player
    sector_polygons: dict[int, np.ndarray]    # sector index -> (K, 4) [x1, y1, x2, y2]
    grid: CellGrid | None = None
    _locator: SectorLocator | None = field(default=None, repr=False, compare=False)

    # ---- things
    def of_kind(self, *kinds: str) -> list[Thing]:
        return [t for kind in kinds for t in self.things.get(kind, ())]

    @property
    def all_things(self) -> list[Thing]:
        return [t for group in self.things.values() for t in group]

    @property
    def monsters(self) -> list[Thing]:
        return self.of_kind("monster")

    @property
    def keys(self) -> list[Thing]:
        return self.of_kind("key")

    @property
    def pickups(self) -> list[Thing]:
        """Health, armor, ammo, weapons and powerups (keys are listed separately)."""
        return self.of_kind(*PICKUP_KINDS)

    def things_by_class(self) -> dict[str, np.ndarray]:
        """Engine class name -> (N, 2) positions."""
        out: dict[str, list] = {}
        for t in self.all_things:
            out.setdefault(t.cls, []).append((t.x, t.y))
        return {k: np.array(v, np.float64) for k, v in out.items()}

    def census(self) -> dict[str, Counter]:
        """kind -> Counter of class names, for the kinds the engine's object list can confirm."""
        return {kind: Counter(t.cls for t in self.things.get(kind, ())) for kind in CENSUS_KINDS}

    # ---- totals used by the scorer
    @property
    def monsters_total(self) -> int:
        return len(self.monsters)

    @property
    def key_classes(self) -> tuple[str, ...]:
        """Distinct key classes on the map, in the wrapper's key order."""
        present = {t.cls for t in self.keys}
        return tuple(k for k in KEY_CLASSES if k in present)

    @property
    def keys_total(self) -> int:
        return len(self.key_classes)

    @property
    def doors_total(self) -> int:
        return sum(1 for d in self.doors if not d.locked)

    @property
    def locked_total(self) -> int:
        return sum(1 for d in self.doors if d.locked)

    @property
    def has_exit(self) -> bool:
        return bool(self.exits)

    @property
    def exit_centres(self) -> np.ndarray:
        """(E, 2) centres of the exit regions."""
        return np.array([e.centre for e in self.exits], np.float64).reshape(-1, 2)

    @property
    def exit_sectors(self) -> tuple[int, ...]:
        return tuple(e.sector for e in self.exits if e.sector >= 0)

    @property
    def door_lock(self) -> np.ndarray:
        """(num_sectors,) lock per sector, -1 = not a door: the layout of privileged_state()["door_lock"]."""
        out = np.full(self.num_sectors, -1, np.int64)
        for d in self.doors:
            out[d.sector] = d.lock
        return out

    @property
    def locator(self) -> SectorLocator:
        if self._locator is None:
            self._locator = SectorLocator(self.sector_polygons)
        return self._locator

    def sector_at(self, x: float, y: float, hint: int = -1) -> int:
        return self.locator.locate(x, y, hint)

    def summary(self) -> dict:
        counts = {kind: sum(c.values()) for kind, c in self.census().items()}
        return {
            "wad": self.wad, "map": self.map_name, "format": self.format, "sectors": self.num_sectors,
            "polygons": len(self.sector_polygons), "player_start": self.player_start,
            "exits": len(self.exits), "exit_sectors": list(self.exit_sectors),
            "doors": self.doors_total, "locked_doors": self.locked_total,
            "locks": sorted(d.colour for d in self.doors if d.locked), "keys": list(self.key_classes),
            "counts": counts, "grid": None if self.grid is None else self.grid.shape,
        }


# ---------------------------------------------------------------------------------- parsing
@dataclass
class _Raw:
    """Format-independent view of the lumps: every line with its two sectors and special, and the things."""

    p1: np.ndarray        # (N, 2)
    p2: np.ndarray        # (N, 2)
    front: np.ndarray     # (N,) sector of the front side, -1 = none
    back: np.ndarray      # (N,) sector of the back side, -1 = one-sided
    exit_trigger: list    # per line: "" or "cross" / "use" / "shoot" when the line ends the level
    things: list          # [Thing] at skill 3, single player
    door_delay: dict      # sector -> tics a door stays open (UDMF only)


def _thing(type_: int, x: float, y: float, angle: float, ambush: bool) -> Thing:
    cls, kind = THING_TABLE.get(int(type_), (f"Thing{int(type_)}", "other"))
    return Thing(cls, kind, int(type_), float(x), float(y), float(angle), bool(ambush))


def _raw_udmf(text: str) -> _Raw:
    blocks = parse_udmf(text)
    doom_numbering = udmf_namespace(text) in DOOM_NAMESPACES
    verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in blocks.get("vertex", [])]
    sides = [int(s.get("sector", -1)) for s in blocks.get("sidedef", [])]

    def side_sector(idx) -> int:
        return sides[idx] if isinstance(idx, int) and 0 <= idx < len(sides) else -1

    rows, triggers, delay = [], [], {}
    for line in blocks.get("linedef", []):
        try:
            a, b = verts[int(line["v1"])], verts[int(line["v2"])]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        special = int(line.get("special", 0) or 0)
        front, back = side_sector(line.get("sidefront", -1)), side_sector(line.get("sideback", -1))
        rows.append((a, b, front, back))
        if doom_numbering:
            triggers.append(DOOM_EXIT_SPECIALS.get(special, ""))
        elif special in HEXEN_EXIT_SPECIALS:
            triggers.append("use" if line.get("playeruse") else "shoot" if line.get("impact") else "cross")
        else:
            triggers.append("")
        if not doom_numbering and special in (12, 13) and not int(line.get("arg0", 0) or 0) and back >= 0:
            delay[back] = int(line.get("arg2", 0) or 0)  # Door_Raise / Door_LockedRaise: arg2 = delay
    things = [
        _thing(t.get("type", 0), t.get("x", 0.0), t.get("y", 0.0), t.get("angle", 0), t.get("ambush", False))
        for t in blocks.get("thing", [])
        if t.get(f"skill{SKILL}", False) and t.get("single", False)
    ]
    return _pack(rows, triggers, things, delay)


def _raw_binary(lumps: dict[str, bytes], hexen: bool) -> _Raw:
    vdata = lumps.get("VERTEXES", b"")
    verts = [struct.unpack_from("<hh", vdata, i * 4) for i in range(len(vdata) // 4)]
    sdata = lumps.get("SIDEDEFS", b"")
    sides = [struct.unpack_from("<h", sdata, i * 30 + 28)[0] for i in range(len(sdata) // 30)]

    def side_sector(idx: int) -> int:
        return sides[idx] if idx != 0xFFFF and idx < len(sides) else -1

    ldata = lumps.get("LINEDEFS", b"")
    size = 16 if hexen else 14
    rows, triggers = [], []
    for i in range(len(ldata) // size):
        v1, v2, flags = struct.unpack_from("<HHH", ldata, i * size)
        front, back = struct.unpack_from("<HH", ldata, i * size + size - 4)
        if v1 >= len(verts) or v2 >= len(verts):
            continue
        rows.append((verts[v1], verts[v2], side_sector(front), side_sector(back)))
        if hexen:
            special = ldata[i * size + 6]
            activation = (flags >> 10) & 0x7  # 0 cross, 1 use, 3 impact, ...
            triggers.append(("use" if activation == 1 else "shoot" if activation == 3 else "cross")
                            if special in HEXEN_EXIT_SPECIALS else "")
        else:
            triggers.append(DOOM_EXIT_SPECIALS.get(struct.unpack_from("<H", ldata, i * size + 6)[0], ""))

    tdata = lumps.get("THINGS", b"")
    things = []
    if hexen:  # tid, x, y, z, angle, type, flags, special, args[5]
        for i in range(len(tdata) // 20):
            _tid, x, y, _z, angle, type_, flags = struct.unpack_from("<hhhhhhH", tdata, i * 20)
            if flags & 0x0002 and flags & 0x0100:  # medium skill, appears in single player
                things.append(_thing(type_, x, y, angle, flags & 0x0008))
    else:      # x, y, angle, type, flags
        for i in range(len(tdata) // 10):
            x, y, angle, type_, flags = struct.unpack_from("<hhhhH", tdata, i * 10)
            if flags & 0x0002 and not flags & 0x0010:  # medium skill, not multiplayer-only
                things.append(_thing(type_, x, y, angle, flags & 0x0008))
    return _pack(rows, triggers, things, {})


def _pack(rows, triggers, things, delay) -> _Raw:
    n = len(rows)
    raw = _Raw(np.zeros((n, 2)), np.zeros((n, 2)), np.full(n, -1, np.int64), np.full(n, -1, np.int64),
               triggers, things, delay)
    for i, (a, b, front, back) in enumerate(rows):
        raw.p1[i], raw.p2[i], raw.front[i], raw.back[i] = a, b, front, back
    return raw


def _find_exits(raw: _Raw, polygons: dict[int, np.ndarray], start_sector: int) -> list[ExitRegion]:
    """Group exit lines into regions.

    A sector that can only be entered across a line carrying an exit special is an exit region
    (the exit pad of a grid layout): every two-sided line on its border is an exit line or leads
    to another such sector (pads that touch). The sector the player starts in is never one, which
    settles the case of a start room whose only opening is the exit. Exit lines that enclose no
    such sector (switches, a walk-over line across a corridor) are reported one by one with the
    line's midpoint as the centre.
    """
    is_exit = np.array([bool(t) for t in raw.exit_trigger], bool)
    if not is_exit.any():
        return []
    seg = np.concatenate([raw.p1, raw.p2], axis=1)
    two_sided = (raw.front >= 0) & (raw.back >= 0) & (raw.front != raw.back)
    touching = is_exit & two_sided
    regions = {int(s) for s in np.concatenate([raw.front[touching], raw.back[touching]])} - {start_sector}
    borders = {sec: np.flatnonzero(two_sided & ((raw.front == sec) | (raw.back == sec))) for sec in regions}
    shrinking = True
    while shrinking:   # drop sectors with an ordinary opening to anything that is not itself a region
        shrinking = False
        for sec in sorted(regions):
            idx = borders[sec]
            beyond = np.where(raw.front[idx] == sec, raw.back[idx], raw.front[idx])
            if not np.all(is_exit[idx] | np.isin(beyond, sorted(regions))):
                regions.discard(sec)
                shrinking = True
    exits: list[ExitRegion] = []
    used = np.zeros(len(seg), bool)
    for sec in sorted(regions):
        idx = borders[sec][is_exit[borders[sec]]]
        polygon = polygons.get(sec, seg[idx])
        exits.append(ExitRegion(sec, polygon_centre(polygon), polygon, seg[idx].copy(), raw.exit_trigger[idx[0]]))
        used[idx] = True
    for i in np.flatnonzero(is_exit & ~used):
        mid = (float(seg[i, 0] + seg[i, 2]) / 2.0, float(seg[i, 1] + seg[i, 3]) / 2.0)
        exits.append(ExitRegion(-1, mid, seg[i:i + 1].copy(), seg[i:i + 1].copy(), raw.exit_trigger[i]))
    return exits


_CACHE: dict[tuple, MapMeta] = {}


def load_meta(wad, map_name: str = "MAP01") -> MapMeta:
    """Read the static facts of one map. Raises MapTooLarge above 2,048 sectors.

    Results are cached per (file, map, modification time), so calling this once per episode is
    cheap; the returned object is shared and must not be modified.
    """
    path = Path(wad).resolve()
    name = map_name.upper()
    stat = path.stat()
    key = (str(path), name, stat.st_mtime_ns, stat.st_size)
    if key in _CACHE:
        return _CACHE[key]

    info = load_mapinfo(path, name)
    if info.num_sectors > MAX_SECTORS:
        raise MapTooLarge(f"{path.name}:{name} has {info.num_sectors} sectors (limit {MAX_SECTORS})")
    lumps = map_lumps(read_wad(path), name)
    if "TEXTMAP" in lumps:
        raw = _raw_udmf(lumps["TEXTMAP"].decode("utf-8", "replace"))
    else:
        raw = _raw_binary(lumps, hexen="BEHAVIOR" in lumps)

    polygons = dict(sorted(info.sector_lines().items()))
    doors = [
        Door(sec, int(lock), lock_colour(lock), polygon_centre(polygons.get(sec, np.zeros((0, 4)))),
             polygons.get(sec, np.zeros((0, 4))), raw.door_delay.get(sec))
        for sec, lock in sorted(info.door_sectors.items())
    ]
    starts = [t for t in raw.things if t.type == 1]
    start = (starts[-1].x, starts[-1].y, starts[-1].angle) if starts else (0.0, 0.0, 0.0)  # the engine uses the last one
    locator = SectorLocator(polygons)
    exits = _find_exits(raw, polygons, locator.locate(start[0], start[1]) if starts else -1)
    things: dict[str, list[Thing]] = {}
    for t in raw.things:
        things.setdefault(t.kind, []).append(t)
    meta = MapMeta(str(path), name, info.format, info.num_sectors, start, exits, doors, things, polygons,
                   _recover_grid(polygons, doors, exits), locator)
    if len(_CACHE) > 4096:
        _CACHE.clear()
    _CACHE[key] = meta
    return meta


def wad_maps(wad) -> list[str]:
    """Names of the maps inside a WAD."""
    return list_maps(read_wad(wad))


# ------------------------------------------------------------------- engine cross-check
def engine_census(env) -> dict[str, Counter]:
    """kind -> Counter of class names from the engine's object list.

    Call right after `env.reset()`. This reaches the game object directly (`env._game`), which
    only training code may do; it is the ground truth for how many monsters and items the level
    really spawned.
    """
    out = {kind: Counter() for kind in CENSUS_KINDS}
    state = env._game.get_state()
    for obj in (state.objects if state is not None else []):
        kind = ENGINE_KIND.get(obj.category)
        if kind is not None:
            out[kind][obj.name] += 1
    return out


def cross_check(meta: MapMeta, env) -> list[str]:
    """Differences between the WAD's things at skill 3 and what the engine spawned (empty = agree).

    Also compares the player start with the engine's position. Call right after `env.reset()`.
    """
    problems = []
    wad, engine = meta.census(), engine_census(env)
    for kind in CENSUS_KINDS:
        if wad[kind] != engine[kind]:
            diff = {c: (wad[kind].get(c, 0), engine[kind].get(c, 0))
                    for c in sorted(set(wad[kind]) | set(engine[kind])) if wad[kind].get(c, 0) != engine[kind].get(c, 0)}
            problems.append(f"{kind}: (wad, engine) counts differ for {diff}")
    st = env.privileged_state()
    if abs(st["x"] - meta.player_start[0]) > 1.0 or abs(st["y"] - meta.player_start[1]) > 1.0:
        problems.append(f"player start: wad {meta.player_start[:2]} vs engine ({st['x']:.1f}, {st['y']:.1f})")
    return problems


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    for spec in args:
        wad, _, name = spec.partition(":")
        for map_name in ([name] if name else wad_maps(wad)):
            try:
                meta = load_meta(wad, map_name)
            except MapTooLarge as e:
                print(f"skipped: {e}")
                continue
            for k, v in meta.summary().items():
                print(f"{k:>13}: {v}")
            if meta.grid is not None:
                print(meta.grid.ascii(meta.all_things))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
