# Map generator

Training-time scaffolding: nothing here is imported by the agent in `submission/`.

The agent is trained on procedural grid layouts. A layout is an ASCII grid with one character per
64-unit cell; `udmf.py` compiles it to a UDMF map that ViZDoom loads through `doom_env`.

```
##########        python -m training.mapgen.cli --out data/maps/train --n 300 \
#P..D...E#            --tiers 0,1,2,3,4 --seed 0 --randomize 0.5
##########
```

| File | What it does |
|---|---|
| `udmf.py` | `build_wad(grid, **options)`: grid to PWAD (`MAP01`, `TEXTMAP`, `ENDMAP`). The legend. |
| `solver.py` | `solve(grid)`: breadth-first search over (cell, keys held). Solvability and route cost. |
| `layouts.py` | `generate(tier, seed)`: seeded layouts for tiers 0-4 with a tick cap and counts. |
| `randomize.py` | `random_options(rng, grid)`: start pose, thing facing, ambush, light, textures, door delay. |
| `walker.py` | Reference walker that follows the solver's route in the engine. Used by the tests only. |
| `cli.py` | Training sets with a manifest, frozen suites, per-tier statistics. |

## Compiler

Every connected region of open floor is one sector; each door cell and each exit cell is its own
sector. Wall edges on the same grid line are merged into long one-sided lines. A door cell has floor
and ceiling at height 0 and opens with the engine's door action (special 12, or 13 with a lock
number for a locked door); an exit cell is ringed by walk-over exit lines (special 243). With no
options the compiler reproduces the reference maps byte for byte: the 13 grids of `tests/fixtures/`
and `maps/two_rooms.wad` (14 of 14 files identical).

Options, all off by default: `start_angle`, `start_offset`, `thing_angles`, `ambush`, `light`,
`themes` (floor and wall per sector), `ceiling`, `exit_floor`, `exit_wall`, `door_delay`.

### Legend

| Symbol | Meaning | Engine number |
|---|---|---|
| `#` `.` | wall, floor | |
| `P` | player start (angle 0 = east) | thing 1 |
| `E` | exit pad | line special 243 |
| `D` | door (closes again after 150 tics) | line special 12 |
| `R` `B` `Y` | red, blue, yellow locked door (stays open) | special 13, locks 129, 130, 131 |
| `r` `b` `y` | red, blue, yellow keycard | things 13, 5, 6 |
| `1` `2` `3` | red, blue, yellow skull key | things 38, 40, 39 |
| `z` `s` `c` | zombieman, shotgun guy, chaingunner | things 3004, 9, 65 |
| `i` `d` `g` | imp, demon, spectre | things 3001, 3002, 58 |
| `l` `o` `k` | lost soul, cacodemon, hell knight | things 3006, 3005, 69 |
| `h` `m` `a` | stimpack, medikit, green armor | things 2011, 2012, 2018 |
| `n` `x` `t` | ammo clip, box of bullets, shotgun shells | things 2007, 2048, 2008 |
| `f` `j` | shotgun, chaingun | things 2001, 2002 |

`udmf.LEGEND` and `udmf.legend_table()` hold the same information as data.

## Layouts

| Tier | Content |
|---|---|
| 0 navigation | Monster-free maps with an exit: corridors, rooms, doors, dead ends. Start of the curriculum. |
| 1 combat arena | Open rooms with cover and 2-12 monsters; about half have an exit. |
| 2 exploration | Several rooms, corridors, doors and dead ends; light opposition. |
| 3 keys and locks | One or two locked doors on the way to the exit; the door is usually met before its key. |
| 4 gauntlet | Two or three locks of different colours, monsters beside doors with the ambush flag, few pickups. |

A map is built in passes. One of six carvers cuts the floor plan: `strip` (a corridor cut into
chambers), `arena` (one large room in several shapes, with pillars), `house` (rooms that share walls),
`scatter` (separate rooms joined by straight and L-shaped corridors 1 to 3 cells wide), `maze` (winding
corridors with optional loops and rooms) and `cave` (irregular open space). About one map in eight is
mirrored into a symmetric plan. Generic passes then place the start, the exit pad (alcove, corridor
end, against a wall, corner or open floor), locks on chokepoints of the route with keys in the part
of the map reachable before each lock, ordinary doors on room entrances, monsters and pickups.

Every draft is checked by the solver: all open cells reachable, exit reachable, locks really needed.
The tick cap (2100 to 4200 in steps of 350) is at least three times the cost of the shortest route,
where the route cost model (2 steps per cell, 4 per corner, 20 per door) was fitted to the reference
walker in the engine. Drafts that fail are discarded and the next attempt of the same seed is used,
so `generate(tier, seed)` is a pure function.

## Randomisation

`--randomize P` builds a fraction P of the maps with random options: the player start is rotated
(a multiple of 90 degrees or any angle) and shifted by up to 16 units, so walls do not line up with
the agent's start frame; monsters face random directions and some get the ambush flag; sector light,
room floors and walls, the ceiling and the wall around the exit pad change (94 floors, 123 walls,
30 ceilings, all present in the Freedoom IWAD); ordinary doors sometimes use another delay. The exit
pad always keeps its floor.

## Manifest and suites

`manifest.jsonl` has one row per map: `name`, `path`, `file`, `tier`, `seed`, `episode_ticks`,
`style`, `exit_style`, `randomized`, `grid`, `options`, `counts`. `path` is written as given with
`--out`, so it is relative to the directory the command ran in (the repository root).

`training/suites/` holds the frozen evaluation suites as grids; see the README there.

## Checks

`training/tests/test_mapgen.py` covers: byte identity and identical observations against the
reference maps, every legend thing in the engine, lock colours, options, the solver, 50 maps per
tier (structure, engine load, random play without warnings, reference walker to the exit within 60%
of the tick cap) and the frozen suites.
