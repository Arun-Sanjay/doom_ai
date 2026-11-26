"""Texture splits and build options for the exit-detector maps.

Training-time scaffolding. Never imported by the agent in submission/.

The detector must find the exit pad, not memorise the room around it. Its maps are therefore built
from the map generator's grid layouts with randomised looks, and the textures are split by name
into a part that training may use and a part that only appears in the "unseen" evaluation set:

* flats (floors and ceilings share one pool of names) and wall textures are split separately;
* every texture of the reference dialect stays on the training side, because the detector has to
  work on plain reference maps first of all;
* red-tinted textures are split as their own group, so both sides contain red floors and red walls
  (the detector must not fire on a red floor it has seen, nor on one it has never seen).

"Red-tinted" is measured, not guessed: `red_share(name)` reads the texture from the IWAD and
returns the share of its pixels that are strongly red (R > 90 and R > 1.6 * max(G, B)); a texture
with a share of at least 0.25 is red-tinted. The exit pad flat itself measures 0.58. The lists
below are the result of that rule and a test recomputes them.

`detector_options` draws `udmf.build_wad` options from one side of the split, and `add_distractors`
edits a grid so that things which look like the pad but are not (red keys, red doors, imps that
throw fireballs) show up far more often than in a normal map.
"""

from __future__ import annotations

import os
import random
import struct
from functools import lru_cache

from training.mapgen import randomize, udmf

RED_SHARE_MIN = 0.25
RED_FLOORS = ("FLAT5_3", "FLOOR1_6", "FLOOR1_7", "FLOOR6_1")
RED_WALLS = ("PANRED", "REDWALL", "ROCKRED1", "SKIN2", "SKINFACE", "SP_HOT1")

# Textures the compiler uses without being asked: the reference themes, ceiling and door floor.
REFERENCE_FLATS = frozenset([f for f, _ in udmf.DEFAULT_THEMES] + [udmf.DEFAULT_CEILING, udmf.DOOR_FLOOR])
REFERENCE_WALLS = frozenset(w for _, w in udmf.DEFAULT_THEMES)

HELD_OUT_SHARE = 0.25
SPLIT_SEED = "exitdet-texture-split-v1"
LIGHTS = (128, 144, 160, 176, 192, 192, 192, 208, 224, 240, 255)     # the generator's range; below 128 a far pad renders black


def _split(names, red, keep, tag: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(training names, held-out names): a seeded shuffle of each group (red, not red), the first
    HELD_OUT_SHARE of each group held out. Names in `keep` always stay on the training side."""
    train, held = set(keep) & set(names), set()
    for group_name, group in (("red", [n for n in names if n in red]), ("plain", [n for n in names if n not in red])):
        free = sorted(set(group) - set(keep))
        random.Random(f"{SPLIT_SEED}/{tag}/{group_name}").shuffle(free)
        cut = max(1, round(HELD_OUT_SHARE * len(free))) if free else 0
        held |= set(free[:cut])
        train |= set(free[cut:])
    return tuple(sorted(train)), tuple(sorted(held))


_FLATS = sorted(set(randomize.FLOORS) | set(randomize.CEILINGS))
TRAIN_FLATS, HELD_FLATS = _split(_FLATS, RED_FLOORS, REFERENCE_FLATS, "flats")
TRAIN_WALLS, HELD_WALLS = _split(sorted(randomize.WALLS), RED_WALLS, REFERENCE_WALLS, "walls")

SPLITS = {
    "train": {
        "floors": tuple(f for f in randomize.FLOORS if f in TRAIN_FLATS),
        "ceilings": tuple(c for c in randomize.CEILINGS if c in TRAIN_FLATS),
        "walls": TRAIN_WALLS,
    },
    "heldout": {
        "floors": tuple(f for f in randomize.FLOORS if f in HELD_FLATS),
        "ceilings": tuple(c for c in randomize.CEILINGS if c in HELD_FLATS),
        "walls": HELD_WALLS,
    },
}
for _side in SPLITS.values():
    _side["red_floors"] = tuple(f for f in _side["floors"] if f in RED_FLOORS)
    _side["red_walls"] = tuple(w for w in _side["walls"] if w in RED_WALLS)


def is_red_floor(name: str) -> bool:
    return name in RED_FLOORS


def is_red_wall(name: str) -> bool:
    return name in RED_WALLS


# ---- build options ------------------------------------------------------------------------------
def detector_options(rng: random.Random, grid: str, side: str, base: dict | None = None,
                     red_boost: float = 0.3) -> dict:
    """`udmf.build_wad` options with looks drawn from one side of the texture split.

    side "train": some options are left at the reference value now and then, so reference looks and
    random looks mix inside one map. side "heldout": every room floor, room wall, ceiling and the
    wall around the pad comes from the held-out textures, so nothing around the pad has been seen in
    training (door faces and door tracks are fixed by the compiler and are the same everywhere).
    With probability `red_boost` one or two rooms get a red-tinted floor or wall.
    """
    pool = SPLITS[side]
    strict = side != "train"
    opts = dict(base or {})
    rows = udmf.parse_grid(grid)
    sectors = len(udmf.assign_sectors(rows)[1])
    monsters = [(c, r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in udmf.MONSTER_CHARS]

    opts["start_angle"] = rng.randrange(360)
    if monsters and rng.random() < 0.8:
        opts["thing_angles"] = {f"{c},{r}": 45 * rng.randrange(8) for c, r in monsters}

    roll = rng.random()
    if roll < 0.45:
        opts["light"] = rng.choice(LIGHTS)
    elif roll < 0.9:
        opts["light"] = [rng.choice(LIGHTS) for _ in range(min(sectors, 8))]

    themes = [[rng.choice(pool["floors"]), rng.choice(pool["walls"])] for _ in range(rng.randint(1, 8))]
    if rng.random() < red_boost:
        for theme in rng.sample(themes, k=min(len(themes), rng.randint(1, 2))):
            if pool["red_floors"] and rng.random() < 0.7:
                theme[0] = rng.choice(pool["red_floors"])
            elif pool["red_walls"]:
                theme[1] = rng.choice(pool["red_walls"])
    if strict or rng.random() < 0.9:
        opts["themes"] = themes
    if strict or rng.random() < 0.6:
        opts["ceiling"] = rng.choice(pool["ceilings"])
    if strict or rng.random() < 0.4:
        opts["exit_wall"] = rng.choice(pool["walls"])
    return opts


def option_textures(options: dict) -> dict:
    """Texture names a set of build options puts on room floors, walls, the ceiling and the pad's wall."""
    themes = options.get("themes") or udmf.DEFAULT_THEMES
    walls = {t[1] for t in themes}
    if options.get("exit_wall"):
        walls.add(options["exit_wall"])
    return {"floors": sorted({t[0] for t in themes}), "walls": sorted(walls),
            "ceiling": options.get("ceiling", udmf.DEFAULT_CEILING)}


# ---- grid edits ---------------------------------------------------------------------------------
def add_distractors(rng: random.Random, grid: str, tier: int) -> tuple[str, dict]:
    """Edit a grid so more red things that are not the exit are in it. Returns (grid, what was done).

    * one to three extra red keycards or red skull keys on plain floor, half of the time within
      three cells of the pad (a far red key is the closest look-alike of a far pad);
    * now and then ordinary doors become red locked doors (same cell, red-barred face);
    * in tiers with monsters, one to three extra imps (they throw red fireballs);
    * rarely the pad itself is turned into plain floor: the same place without an exit.
    Nothing here needs the map to stay solvable; the frames are taken from teleported poses.
    """
    rows = [list(r) for r in udmf.parse_grid(grid)]
    done = {"red_keys": 0, "red_doors": 0, "imps": 0, "pad_removed": 0}

    def cells(ch: str) -> list[tuple[int, int]]:
        return [(c, r) for r, row in enumerate(rows) for c, v in enumerate(row) if v == ch]

    pads, floor = cells("E"), cells(".")
    if pads and rng.random() < 0.06:
        for c, r in pads:
            rows[r][c] = "."
        done["pad_removed"] = len(pads)
        pads = []
    for _ in range(rng.choice((0, 1, 1, 2, 3))):
        near = [(c, r) for c, r in floor if any(abs(c - pc) + abs(r - pr) <= 3 for pc, pr in pads)]
        pick = rng.choice(near if near and rng.random() < 0.5 else floor) if floor else None
        if pick is None:
            break
        floor.remove(pick)
        rows[pick[1]][pick[0]] = rng.choice(("r", "r", "1"))
        done["red_keys"] += 1
    if rng.random() < 0.35:
        for c, r in cells("D"):
            if rng.random() < 0.5:
                rows[r][c] = "R"
                done["red_doors"] += 1
    if tier >= 1 and rng.random() < 0.55:
        start = cells("P")[0]
        far = [(c, r) for c, r in floor if abs(c - start[0]) + abs(r - start[1]) >= 3]
        for _ in range(rng.randint(1, 3)):
            if not far:
                break
            pick = rng.choice(far)
            far.remove(pick)
            floor.remove(pick)
            rows[pick[1]][pick[0]] = "i"
            done["imps"] += 1
    return "\n".join("".join(r) for r in rows), done


# ---- measuring redness from the IWAD ---------------------------------------------------------------
@lru_cache(maxsize=1)
def _iwad() -> dict:
    import numpy as np
    import vizdoom
    from doom_env.wad import read_wad

    lumps = {lump.name: lump.data for lump in read_wad(os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"))}
    palette = np.frombuffer(lumps["PLAYPAL"][:768], np.uint8).reshape(256, 3).astype(np.float64)
    names = lumps["PNAMES"]
    patches = [names[4 + 8 * i:12 + 8 * i].rstrip(b"\0").decode("ascii").upper()
               for i in range(struct.unpack_from("<i", names, 0)[0])]
    textures = {}
    for lump in ("TEXTURE1", "TEXTURE2"):
        data = lumps.get(lump)
        if not data:
            continue
        for i in range(struct.unpack_from("<i", data, 0)[0]):
            off = struct.unpack_from("<i", data, 4 + 4 * i)[0]
            width, height = struct.unpack_from("<hh", data, off + 12)
            count = struct.unpack_from("<h", data, off + 20)[0]
            parts = [struct.unpack_from("<hhh", data, off + 22 + 10 * k) for k in range(count)]
            textures[data[off:off + 8].rstrip(b"\0").decode("ascii").upper()] = (width, height, parts)
    return {"lumps": lumps, "palette": palette, "patches": patches, "textures": textures}


def _patch(data: bytes):
    """A picture lump (column format) as an (H, W) array of palette indices, -1 where transparent."""
    import numpy as np

    width, height = struct.unpack_from("<hh", data, 0)
    img = np.full((height, width), -1, np.int32)
    for x in range(width):
        off = struct.unpack_from("<i", data, 8 + 4 * x)[0]
        while data[off] != 255:
            top, length = data[off], data[off + 1]
            img[top:top + length, x] = np.frombuffer(data[off + 3:off + 3 + length], np.uint8)[:max(0, height - top)]
            off += length + 4
    return img


def texture_pixels(name: str):
    """RGB pixels (N, 3) of a flat or a wall texture of the IWAD (walls composed from their patches)."""
    import numpy as np

    iwad = _iwad()
    name = name.upper()
    if name in iwad["textures"]:
        width, height, parts = iwad["textures"][name]
        img = np.full((height, width), -1, np.int32)
        for ox, oy, index in parts:
            patch = _patch(iwad["lumps"][iwad["patches"][index]])
            y0, y1 = max(oy, 0), min(oy + patch.shape[0], height)
            x0, x1 = max(ox, 0), min(ox + patch.shape[1], width)
            if y0 >= y1 or x0 >= x1:
                continue
            part = patch[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
            view = img[y0:y1, x0:x1]
            view[part >= 0] = part[part >= 0]
        return iwad["palette"][img[img >= 0]]
    return iwad["palette"][np.frombuffer(iwad["lumps"][name][:4096], np.uint8)]


def red_share(name: str) -> float:
    """Share of a texture's pixels that are strongly red."""
    rgb = texture_pixels(name)
    r, other = rgb[:, 0], rgb[:, 1:].max(axis=1)
    return float(((r > 90.0) & (r > 1.6 * other)).mean())
