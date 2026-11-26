"""Domain randomisation of the build options. Training-time scaffolding, never imported by the agent.

`random_options(rng, grid, base)` returns `udmf.build_wad` options that change how a grid looks and
where the agent starts, without changing the grid itself:

* start pose: the player is rotated (a multiple of 90 degrees or any angle) and shifted by up to 16
  units inside its cell, so walls do not line up with the axes or the 64-unit cells of the agent's
  start frame (`random_options` draws it for the first training set; `random_start` draws it with
  explicit probabilities, see below);
* things: random facing for monsters and a few extra ambush flags;
* looks: sector light, room floor and wall textures, ceiling, the wall around the exit pad;
* doors: now and then a different delay before an ordinary door closes.

A start is "on the grid" when the player stands on a cell centre and faces along an axis: the walls
then lie on the 64-unit cells of the agent's start frame, as they do in every reference map. A
quarter turn keeps that, any other heading or an offset breaks it. `random_start(rng, quarter,
arbitrary, offset)` draws the start pose alone, each kind with its own probability, so that off-grid
starts can be kept to a small share of a set; `start_on_grid(options)` tells which maps they are.

The exit pad keeps its floor (GATE3) in every map: it is the one visual cue for the exit.
All texture names exist in the Freedoom IWAD shipped with ViZDoom (checked by the tests). Animated
and liquid flats, sky and the teleport-pad flats that resemble the exit pad are left out.
"""

from __future__ import annotations

import random

from . import udmf

FLOORS = tuple("""
FLOOR0_1 FLOOR0_2 FLOOR0_3 FLOOR0_5 FLOOR0_6 FLOOR0_7 FLOOR1_1 FLOOR1_6 FLOOR1_7 FLOOR3_3 FLOOR4_1 FLOOR4_5
FLOOR4_6 FLOOR4_8 FLOOR5_1 FLOOR5_2 FLOOR5_3 FLOOR5_4 FLOOR6_1 FLOOR6_2 FLOOR7_1 FLOOR7_2 FLAT1 FLAT1_1 FLAT1_2
FLAT1_3 FLAT2 FLAT3 FLAT4 FLAT5 FLAT5_1 FLAT5_2 FLAT5_3 FLAT5_4 FLAT5_5 FLAT5_6 FLAT5_7 FLAT5_8 FLAT8 FLAT9
FLAT10 FLAT14 FLAT17 FLAT18 FLAT19 FLAT20 FLAT22 FLAT23 CEIL1_1 CEIL3_1 CEIL3_2 CEIL3_3 CEIL3_4 CEIL3_5 CEIL3_6
CEIL4_1 CEIL4_2 CEIL4_3 CEIL5_1 CEIL5_2 MFLR8_1 MFLR8_2 MFLR8_3 MFLR8_4 GRASS1 GRASS2 GRNROCK RROCK09 RROCK10
RROCK11 RROCK12 RROCK13 RROCK14 RROCK15 RROCK16 RROCK17 RROCK18 RROCK19 RROCK20 SLIME13 SLIME14 SLIME15 SLIME16
DEM1_1 DEM1_2 DEM1_3 DEM1_4 DEM1_5 DEM1_6 CRATOP1 CRATOP2 COMP01 STEP1 STEP2
""".split())

WALLS = tuple("""
STARTAN2 STARTAN3 STARG1 STARG3 STARGR1 STARGR2 STARBR2 STONE STONE2 STONE3 STONE4 STONE6 BROWN1 BROWN96 BROWN144
BROWNGRN BROWNHUG BROWNPIP BRONZE1 BRONZE3 GRAY1 GRAY4 GRAY5 GRAY7 GRAYBIG GRAYTALL GRAYVINE BRICK1 BRICK2 BRICK4
BRICK5 BRICK6 BRICK7 BRICK10 BRICK12 BIGBRIK1 BIGBRIK2 METAL METAL1 METAL2 METAL5 METAL6 TEKWALL1 TEKWALL4
TEKWALL6 TEKGREN1 TEKGREN2 TEKBRON1 WOOD1 WOOD3 WOOD5 WOOD9 WOOD12 WOODVERT MARBLE1 MARBLE2 MARBLE3 MARBGRAY
GSTONE1 GSTVINE1 ROCK1 ROCK2 ROCK3 ROCK4 ROCK5 ROCKRED1 STUCCO STUCCO1 STUCCO3 PANEL1 PANEL4 PANEL6 PANBLUE PANRED
SLADWALL ICKWALL1 ICKWALL2 ICKWALL3 CEMENT1 CEMENT6 CEMENT7 CEMENT9 COMPBLUE COMPTALL COMPSPAN SHAWN2 SILVER1
SILVER2 SPACEW2 SPACEW3 SPACEW4 PIPE2 PIPE4 PIPEWAL1 ZIMMER1 ZIMMER2 ZIMMER5 TANROCK2 TANROCK4 TANROCK5 BSTONE1
BSTONE2 ASHWALL2 ASHWALL3 ASHWALL4 MODWALL1 MODWALL2 CRATE1 CRATE2 CRATWIDE REDWALL SP_HOT1 SKIN2 SKINFACE
SKSNAKE1 BLAKWAL1 BLAKWAL2 NUKEDGE1 LITE3 LITE5 LITEBLU4 SUPPORT3 EXITSTON
""".split())

CEILINGS = tuple("""
CEIL1_1 CEIL1_2 CEIL1_3 CEIL3_1 CEIL3_2 CEIL3_5 CEIL4_1 CEIL4_2 CEIL5_1 CEIL5_2 FLAT1 FLAT2 FLAT5_4 FLAT5_5 FLAT18
FLAT19 FLAT20 FLAT22 FLAT23 FLOOR4_8 FLOOR5_2 FLOOR7_1 MFLR8_1 RROCK09 TLITE6_4 TLITE6_5 GRNLITE1 CRATOP2 DEM1_5
SLIME14
""".split())

LIGHTS = (128, 144, 160, 176, 192, 192, 208, 224, 255)
DOOR_DELAYS = (105, 120, 175, 200, 250)
MAX_START_OFFSET = 16       # the player has radius 16 and a cell is 64 wide


# Probabilities of `random_start` for a training set: (quarter turn, arbitrary heading, sub-cell offset).
START_DEFAULTS = (0.5, 0.05, 0.05)
QUARTER_TURNS = (90, 180, 270)
OFF_AXIS_ANGLES = tuple(a for a in range(360) if a % 90)


def random_start(rng: random.Random, quarter: float = START_DEFAULTS[0], arbitrary: float = START_DEFAULTS[1],
                 offset: float = START_DEFAULTS[2]) -> dict:
    """Start-pose build options, each kind drawn with its own probability.

    quarter    the player faces north, west or south instead of east; the start stays on the grid.
    arbitrary  the player faces a whole-degree heading that is not a multiple of 90: off the grid.
    offset     the player is shifted by up to 16 units on each axis, never by (0, 0): off the grid.

    The heading is one draw (quarter turn, arbitrary, or east as stored: `quarter + arbitrary` may
    not exceed 1) and the offset is independent of it, so a share
    `1 - (1 - arbitrary) * (1 - offset)` of the maps starts off the grid.
    """
    for name, p in (("quarter", quarter), ("arbitrary", arbitrary), ("offset", offset)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name}: a probability is between 0 and 1, got {p}")
    if quarter + arbitrary > 1.0 + 1e-9:
        raise ValueError("quarter + arbitrary may not exceed 1: they are two outcomes of one draw")
    opts: dict = {}
    roll = rng.random()
    if roll < quarter:
        opts["start_angle"] = rng.choice(QUARTER_TURNS)
    elif roll < quarter + arbitrary:
        opts["start_angle"] = rng.choice(OFF_AXIS_ANGLES)
    if rng.random() < offset:
        shift = [0, 0]
        while not any(shift):
            shift = [rng.randint(-MAX_START_OFFSET, MAX_START_OFFSET), rng.randint(-MAX_START_OFFSET, MAX_START_OFFSET)]
        opts["start_offset"] = shift
    return opts


def start_on_grid(options: dict | None) -> bool:
    """True when the build options leave the player start on a cell centre, facing along an axis."""
    options = options or {}
    offset = options.get("start_offset") or (0, 0)
    turn = float(options.get("start_angle") or 0) % 90.0
    return all(abs(float(v)) < 1e-6 for v in offset) and min(turn, 90.0 - turn) < 1e-6


def random_options(rng: random.Random, grid: str, base: dict | None = None, start: bool = True) -> dict:
    """Randomised build options for `grid`, merged over `base` (the options the layout came with).

    With `start` (the default, and what the first training set was built with) the start pose is
    part of the draw: a quarter turn on 35% of the maps, any heading on 50%, an offset on 60%, which
    leaves about one map in five on the grid. With `start=False` the start pose is left alone, for
    callers that set it with `random_start`.
    """
    opts = dict(base or {})
    rows = udmf.parse_grid(grid)
    sectors = len(udmf.assign_sectors(rows)[1])
    monsters = [(c, r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in udmf.MONSTER_CHARS]

    if start:
        roll = rng.random()
        if roll < 0.35:
            opts["start_angle"] = rng.choice((90, 180, 270))
        elif roll < 0.85:
            opts["start_angle"] = rng.randrange(360)
        if rng.random() < 0.6:
            opts["start_offset"] = [rng.randint(-MAX_START_OFFSET, MAX_START_OFFSET),
                                    rng.randint(-MAX_START_OFFSET, MAX_START_OFFSET)]

    if monsters and rng.random() < 0.8:
        opts["thing_angles"] = {f"{c},{r}": 45 * rng.randrange(8) for c, r in monsters}
    deaf = set(opts.get("ambush", ())) | {f"{c},{r}" for c, r in monsters if rng.random() < 0.15}
    if deaf:
        opts["ambush"] = sorted(deaf, key=lambda key: tuple(int(v) for v in key.split(",")))

    roll = rng.random()
    if roll < 0.4:
        opts["light"] = rng.choice(LIGHTS)
    elif roll < 0.8:
        opts["light"] = [rng.choice(LIGHTS) for _ in range(min(sectors, 8))]
    if rng.random() < 0.85:
        opts["themes"] = [[rng.choice(FLOORS), rng.choice(WALLS)] for _ in range(rng.randint(1, 8))]
    if rng.random() < 0.5:
        opts["ceiling"] = rng.choice(CEILINGS)
    if rng.random() < 0.4:
        opts["exit_wall"] = rng.choice(WALLS)
    if rng.random() < 0.25:
        opts["door_delay"] = rng.choice(DOOR_DELAYS)
    return opts
