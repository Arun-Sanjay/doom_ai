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
| `layouts.py` | `generate(tier, seed)`: seeded layouts for tiers 0-4 with a tick cap and counts. `generate(tier, seed, "wide")`: the same passes under wider limits (tiers 1-4). |
| `randomize.py` | `random_options(rng, grid)`: start pose, thing facing, ambush, light, textures, door delay. `random_start(rng, quarter, arbitrary, offset)`: the start pose alone. `start_on_grid(options)`. |
| `walker.py` | Reference walker that follows the solver's route in the engine. Used by the tests and `--verify` only. |
| `cli.py` | Training sets with a manifest, monster-free copies, the wide profile, extending a set, frozen suites, per-tier statistics, engine check. |

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
| 0 navigation | Monster-free maps with an exit: corridors, rooms, doors, dead ends. One map in three also has one key and the locked door of its colour on the way to the exit. Start of the curriculum. |
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

Keys and locked doors without combat. The first stage of the curriculum has no monsters, and it
should already teach "fetch the key, come back to the door". Whether tier 0 seed `s` is such a map
depends on the seed alone (`layouts.has_key_stage`: every third seed); every draft of that seed
must then place exactly one lock on a chokepoint of the route, with its key in the part of the map
reachable before it. The other tier 0 seeds, and every seed of tiers 1 to 4, give the
same map as before this was added. In the training set 100 of the 300 tier 0 maps have a key and a
lock; the door is met before its key on 61 of them.

Any map can also be written without its monsters: `--strip-monsters` (training sets, `--from-suite`
and `--freeze`) turns every monster cell into floor and drops the ambush flags and monster facings,
keeping walls, doors, keys, pickups, start pose, looks and tick cap. For example

    python -m training.mapgen.cli --out data/maps/train_nav --n 300 --tiers 2,3,4 --seed 0 --randomize 0.5 --strip-monsters

writes the tier 2 to 4 training maps as pure navigation tasks, under the same names in a directory
of their own.

## The wide profile

`--profile wide` (in code `layouts.generate(tier, seed, "wide")`) is an explicit option. Without it
every command gives exactly the maps it gave before the option existed (a pinned digest in the tests
guards this). With it, the maps the command generates are held to the wider limits of
`layouts.WIDE`; the profile has tiers 1 to 4.

| Limit | Base | Wide |
|---|---|---|
| Tick cap over route cost, tiers 2-4 | at least 3 | 1.5 to 3: the cap is one of the 350-tic steps in that range, each equally likely |
| Route cost, tiers 2-4 | at most 350 decision steps (1400 tics) | 175 to 700 steps (700 to 2800 tics) |
| Most monsters, tiers 2 / 3 / 4 | 6 / 8 / 16 | 12 / 12 / 20 |
| Most monsters by open cells, tiers 2 / 3 / 4 | 2 + cells/60, 3 + cells/50, 6 + cells/30 | 4 + cells/30, 4 + cells/35, 7 + cells/24 |
| Number of monsters, tiers 2-4 | one draw up to the limit | the larger of two draws up to the limit |
| Monster health per open cell, tiers 1-4 | 8 / 3 / 4 / 8 | 8 / 6 / 6 / 10 (raised by the factor of the monster limit) |
| Side of the plan, tier 1 | 8 to 18 cells | 12 to 26 cells; no finished grid is longer than 26 |
| Side of the plan, tiers 2 / 3 / 4 | 12-24 / 12-26 / 16-28 | 16-26 / 16-26 / 18-28 |
| Exit, tiers 2-4 | at least 0.3 to 0.9 of the longest walk from the start | at least 0.7 to 0.9 |
| Key of a lock | one of the quarter of cells farthest from where its part of the map is entered, dead ends first | one of the eighth of cells farthest from its door, past the door when there are such cells |
| Locked doors | anywhere on the route | in the first 80% of the route, so the exit is far past the last one |

The passes, the carvers, the legend and the checks of a draft are those of the base profile: every
open cell reachable, the exit reachable, every lock needed, the route found by the solver, and the
health limit still decides which monsters a map may hold. Tier 1 keeps the base tick cap rule
(its routes are a few cells long and half of the arenas have no exit), and the tier 2 share of
monster-free maps stays. Map (tier, seed) of the wide profile is another map than (tier, seed) of
the base profile: layout, looks and start pose are drawn from random streams of their own.

With a cap of 1.5 to 3 times the route there is no time to walk the whole map: the reference
walker, which knows the route, needs up to 71% of the cap on these maps (at most 34% on base maps).
Tests that walk wide maps therefore allow 85% of the cap where the base tests allow 60%.

### The third training set, `data/maps/train_v3`

The 3,000 maps of `data/maps/train_v2` plus 1,500 maps of the wide profile, written from the
repository root with exactly this command:

    python -m training.mapgen.cli --out data/maps/train_v3 --extend data/maps/train_v2/manifest.jsonl --profile wide --n 150,350,500,500 --tiers 1,2,3,4 --seed 7000000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05

`--extend` copies the WAD files of `train_v2` byte for byte and its manifest rows unchanged apart
from `path`; nothing of that set is generated again. `--n` gives the number of new maps for each
tier of `--tiers`: 150, 350, 500 and 500 in tiers 1 to 4, from seeds 7,000,000 to 7,000,499, with
the looks and start settings of `train_v2`. Every row has a `profile` field, "base" for the first
3,000 rows and "wide" for the last 1,500. No seed and no grid is shared with `data/maps/train`,
`train_v2` or a frozen suite. 1,362 of the 1,500 wide maps start on the grid (135, 320, 452 and 455
in tiers 1 to 4), which makes 4,056 of all 4,500; 724 wide maps have random looks.

| Tier | Profile | Maps | Route cells | Route cost in steps | Cap over route cost | Monsters | Longest side | Key to its door, cells | Last key to exit, cells |
|---|---|---|---|---|---|---|---|---|---|
| 1 | base | 600 | 3 / 10 / 35 | 6 / 31 / 122 | 4.30 / 24.58 / 87.50 | 2 / 4.5 / 12 | 6 / 13.1 / 19 | - | - |
| 1 | wide | 150 | 3 / 18 / 49 | 10 / 55 / 204 | 3.43 / 14.69 / 52.50 | 2 / 6.4 / 12 | 7 / 19.2 / 26 | - | - |
| 2 | base | 600 | 14 / 25 / 76 | 32 / 113 / 344 | 3.03 / 8.27 / 27.34 | 0 / 1.7 / 6 | 11 / 19.2 / 24 | - | - |
| 2 | wide | 350 | 25 / 48 / 134 | 176 / 251 / 680 | 1.52 / 2.53 / 2.99 | 0 / 4.3 / 12 | 15 / 22.2 / 26 | - | - |
| 3 | base | 600 | 18 / 43 / 98 | 68 / 206 / 350 | 3.00 / 5.00 / 14.58 | 1 / 3.3 / 8 | 12 / 20.0 / 26 | 1 / 13 / 54 | 3 / 24 / 68 |
| 3 | wide | 500 | 28 / 63 / 137 | 176 / 312 / 694 | 1.50 / 2.35 / 2.99 | 1 / 6.1 / 12 | 14 / 21.5 / 26 | 3 / 22 / 77 | 7 / 41 / 112 |
| 4 | base | 600 | 30 / 59 / 102 | 126 / 272 / 350 | 3.00 / 3.87 / 7.72 | 4 / 7.9 / 16 | 14 / 22.3 / 28 | 1 / 12 / 80 | 3 / 20 / 63 |
| 4 | wide | 500 | 37 / 84 / 150 | 178 / 429 / 698 | 1.50 / 2.02 / 2.99 | 4 / 11.8 / 20 | 16 / 23.7 / 28 | 3 / 19 / 106 | 9 / 33 / 84 |

Minimum / mean / maximum; route figures over the maps with an exit. The two key columns are counted along
the solver's route. Among the wide maps of tiers 2, 3 and 4 the tick cap is under twice the route cost on 34,
132 and 273, the route costs more than the 350 steps the base rule allows on 34, 153 and 357, and 120, 83
and 57 hold more monsters than the base limit of their tier; 46 of the 150 wide arenas are longer than 22
cells. 121 of the 350 wide tier 2 maps have no monster.

Check it with

    python -m training.mapgen.cli --stats data/maps/train_v3/manifest.jsonl
    python -W error -m training.mapgen.cli --verify data/maps/train_v3/manifest.jsonl --profile wide --workers 3

`--profile` with `--stats` or `--verify` takes the rows of that profile only. As written,
all 1,500 wide maps load in `doom_env` and play without warnings, and the reference walker reaches
the exit on all 1,428 that have one, using 41% of the tick cap on average and at most 71% (the other 72
are tier 1 arenas without an exit).

The navigation teacher (`TeacherPolicy` with the noisy exit source, run through `training/runner.py`,
monsters removed, the true cap as its budget) is measured on a fixed sample of on-grid maps, 40 from
each of tiers 2, 3 and 4, with the script that sits next to the manifest:

    python data/maps/train_v3/teacher_check.py data/maps/train_v3/manifest.jsonl 40 data/maps/train_v3/teacher_wide_on_grid.csv wide
    python data/maps/train_v3/teacher_check.py data/maps/train_v3/manifest.jsonl 40 data/maps/train_v3/teacher_base_on_grid.csv base

On the 120 wide maps:

| Tier | Exit reached | Failures that are timeouts |
|---|---|---|
| 2 | 27 of 40 (67.5%) | 13 of 13 |
| 3 | 35 of 40 (87.5%) | 5 of 5 |
| 4 | 33 of 40 (82.5%) | 7 of 7 |
| 2-4 | 95 of 120 (79.2%) | 25 of 25 |

Every failure is an episode that ran to the tick cap: no death, no error, and the teacher's grid check
fitted on all 120 maps. The teacher's own causes: exit never in view before the cap 15, key not collected 6
(on one of them it never left the start area), left for the exit too late 3, locked door not opened 1. By
tightness it reaches the exit on 34 of 44 maps with a cap under 2 times the route cost, 17 of 22 from 2 to
2.5 and 44 of 54 above. On the 120 base maps of the second command it reaches the exit on 119. The
outputs of these checks are next to the manifest: `verify_wide.txt`, `teacher_wide_on_grid.csv` and
`teacher_base_on_grid.csv`, each with a `.txt` summary.

### The wide suites

Two frozen suites of the wide profile, reference dialect, from reserved seeds that no training set
can use, written once with

    python -m training.mapgen.cli --freeze training/suites/heldout_wide_v1.jsonl --profile wide --n 50 --tiers 1,2,3,4 --seed 9500000
    python -m training.mapgen.cli --freeze training/suites/dev_wide_v1.jsonl --profile wide --n 15 --tiers 1,2,3,4 --seed 8500000

`heldout_wide_v1.jsonl` has 50 maps for each of tiers 1 to 4 and `dev_wide_v1.jsonl` 15. Each
command prints the entry for `training/suites/checksums.json`; both entries were added there next
to the existing ones. Rebuild the WADs with

    python -m training.mapgen.cli --from-suite training/suites/dev_wide_v1.jsonl --out data/maps/dev_wide
    python -m training.mapgen.cli --from-suite training/suites/heldout_wide_v1.jsonl --out data/maps/heldout_wide

and check them in the engine with

    python -W error -m training.mapgen.cli --verify training/suites/heldout_wide_v1.jsonl --workers 3
    python -W error -m training.mapgen.cli --verify training/suites/dev_wide_v1.jsonl --workers 3

As frozen, all 260 maps load and play without warnings, and the reference walker reaches the exit on all
231 that have one (176 of the 200 held-out maps, 55 of the 60 development maps), using at most 66% of the
tick cap.

## The reference profiles

The reference maps (the 13 grids of `tests/fixtures/` and `maps/two_rooms.wad`) use twelve symbols:
`# . P E D R B r z i c h`. Their only monsters are the zombieman, the imp and the chaingunner, their
only pickup is the stimpack, and they show no weapon, ammunition or armour. `--profile reference`
and `--profile reference_wide` (in code `layouts.REFERENCE`) hold generated maps to that:

| | `reference` | `reference_wide` |
|---|---|---|
| Sizes, tick cap, exit and key distances | base profile | wide profile (tiers 2-4: cap 1.5 to 3 times the route cost) |
| Monsters | `z`, `i`, `c` only | `z`, `i`, `c` only, more chaingunners |
| Number of monsters, tiers 1 / 2 / 3 / 4 | 1-12 / 0-6 / 1-8 / 4-16 | 4-16 / 3-14 / 4-16 / 8-24, the larger of two draws |
| Monster-free maps | 35% of tier 2 | none |
| Pickups | stimpacks only | stimpacks only |
| Exit | every map, the tier 1 arenas included | every map |

Keys, locked doors (three colours, cards and skulls), ordinary doors, carvers, the solver checks and
the randomised looks are those of the other profiles. Both have tiers 1 to 4 and random streams of
their own; `layouts.PROFILES` and the maps of "base" and "wide" are unchanged (pinned digests in
`training/tests/test_mapgen_reference.py`).

`data/maps/train_ref_v1` (3,000 maps: 450, 750, 750 and 1,050 in tiers 1 to 4, half of every tier
from each profile, seeds 7,200,000 to 7,200,524 and 7,300,000 to 7,300,524) and the two frozen
suites `dev_ref_v1` and `heldout_ref_v1` were written with

    python -m training.mapgen.cli --out PART --profile reference --n 225,375,375,525 --tiers 1,2,3,4 --seed 7200000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
    python -m training.mapgen.cli --out data/maps/train_ref_v1 --extend PART/manifest.jsonl --profile reference_wide --n 225,375,375,525 --tiers 1,2,3,4 --seed 7300000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
    python -m training.mapgen.cli --freeze training/suites/dev_ref_v1.jsonl --profile reference --n 10 --tiers 1,2,3,4 --seed 8700000 --mix reference_wide:8750000
    python -m training.mapgen.cli --freeze training/suites/heldout_ref_v1.jsonl --profile reference --n 30 --tiers 1,2,3,4 --seed 9700000 --mix reference_wide:9750000

where PART is any directory outside the set. `--mix PROFILE:SEED` gives every tier of a frozen suite
`--n` more maps of a second profile after those of `--profile`, and the suite rows then carry a
`profile` field. As written, the solver finds every one of the 3,320 maps solvable with its exit
reachable; in the engine all 320 suite maps and every tenth map of the training set load and play
without warnings, and the reference walker reaches the exit on each of them (at most 65% of the
tick cap).

## Randomisation

`--randomize P` builds a fraction P of the maps with random options: the player start is rotated
(a multiple of 90 degrees or any angle) and shifted by up to 16 units, so walls do not line up with
the agent's start frame; monsters face random directions and some get the ambush flag; sector light,
room floors and walls, the ceiling and the wall around the exit pad change (94 floors, 123 walls,
30 ceilings, all present in the Freedoom IWAD); ordinary doors sometimes use another delay. The exit
pad always keeps its floor.

### Start pose and the 64-unit grid

A start is on the grid when the player stands on a cell centre and faces along an axis. The walls
then lie on the 64-unit cells of the agent's start frame, as in every reference map, and the
navigation teacher (`training/teacher`) can plan and give labels. A quarter turn keeps the start
on the grid; any other heading, or an offset inside the cell, takes it off.

With `--randomize P` alone the start pose is part of the random options: on the maps of the
fraction P, a quarter turn on 35%, any heading on 50% and an offset on 60%. That leaves 919 of the
1,500 maps of `data/maps/train` on the grid (61%); the other 581 cannot be labelled by the teacher.

The start pose therefore has a control of its own. Any of the three flags below gives it a separate
draw on every map of the set (also on the maps whose looks stay in the reference dialect), from a
random stream of its own, and the random options then no longer touch it:

| Flag | Meaning | On the grid | Default once any flag is given |
|---|---|---|---|
| `--start-quarter P` | the player faces north, west or south instead of east | yes | 0.5 |
| `--start-any P` | a whole-degree heading that is not a multiple of 90 | no | 0.05 |
| `--start-offset P` | a shift of up to 16 units on each axis, never (0, 0) | no | 0.05 |

The heading is one draw (quarter turn, arbitrary, or east), so `--start-quarter` plus `--start-any`
may not exceed 1; the offset is independent. A share `1 - (1 - any) * (1 - offset)` of the maps
starts off the grid: 9.75% with the defaults. Thing facing, ambush flags, light, textures and door
delay are randomised as before, on the fraction `--randomize`. A layout never depends on these
flags: map (tier, seed) has the same grid and tick cap in every set.

### The second training set, `data/maps/train_v2`

Written from the repository root with exactly this command:

    python -m training.mapgen.cli --out data/maps/train_v2 --n 600 --tiers 0,1,2,3,4 --seed 6000000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05

3,000 maps, 600 per tier, from seeds 6,000,000 to 6,000,601. No seed is shared with
`data/maps/train` (0 to 300), with the maps the exit detector is trained and measured on
(1,000,000 to about 5,000,500) or with the frozen suites (8,000,000 upwards), and no grid is in a
suite. Tier 0 keeps its rule of a key and a locked door without monsters on every third seed
(200 of 600). 2,694 maps start on the grid (89.8%): 549, 547, 527, 533 and 538 in tiers 0 to 4.
1,445 maps start with a quarter turn (1,369 of them on the grid), 155 have an arbitrary heading
and 154 an offset (3 have both); 1,492 maps have random looks.

Check it with

    python -m training.mapgen.cli --stats data/maps/train_v2/manifest.jsonl
    python -m training.mapgen.cli --verify data/maps/train_v2/manifest.jsonl --workers 3

As written, all 3,000 maps load in `doom_env` and play without warnings, and the reference walker
reaches the exit on all 2,717 maps that have one, using at most 33% of the tick cap (the other 283
are tier 1 arenas without an exit). The navigation teacher (`TeacherPolicy`, noisy exit source,
monsters removed, the true cap as its budget) reaches the exit on 150 of 150 sampled on-grid maps
of tiers 0, 2, 3 and 4, and on 11 of 40 sampled off-grid maps, where it gives no labels. The outputs
of these checks are next to the manifest: `verify.txt`, `teacher_on_grid.csv`, `teacher_off_grid.csv`.

## Manifest and suites

`manifest.jsonl` has one row per map: `name`, `path`, `file`, `tier`, `seed`, `episode_ticks`,
`style`, `exit_style`, `randomized`, `on_grid`, `grid`, `options`, `counts`; a set written with
`--profile` or `--extend` also has `profile` ("base" or "wide") after `on_grid`. `path` is written as
given with `--out`, so it is relative to the directory the command ran in (the repository root).
`on_grid` is a boolean: true when the start is on the grid (see above), which is where teacher
labels apply; `randomize.start_on_grid(row["options"])` gives the same answer for a manifest written
before the field existed, such as `data/maps/train/manifest.jsonl`.

`training/suites/` holds the frozen evaluation suites as grids; see the README there. The training
set is written with the command at the top of this file. `--stats FILE` prints its per-tier table
and `--verify FILE --workers 3` loads every map in the engine and sends the reference walker to the
exit on a monster-free copy.

## Checks

`training/tests/test_mapgen.py` covers: byte identity and identical observations against the
reference maps, every legend thing in the engine, lock colours, options, the solver, 50 maps per
tier (structure, engine load, random play without warnings, reference walker to the exit within 60%
of the tick cap), the tier 0 key maps (one key, one locked door that the route needs, no monster,
finished by the walker without a shot), monster-free copies, the start control (its probabilities,
that it leaves layouts and the first set's options alone, the `on_grid` field), the wide profile (the
base profile unchanged, the wider limits kept, longer walks from key to door and from the last key to
the exit than in the base profile, the reference walker to the exit within 85% of the tick cap,
`--extend` and the `profile` field), and the frozen suites, the two wide ones included.
