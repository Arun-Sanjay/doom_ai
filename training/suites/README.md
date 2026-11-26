# Frozen evaluation suites

| File | Maps | Seeds | Use |
|---|---|---|---|
| `heldout_v1.jsonl` | 100 per tier 1-4 | 9,000,000 upwards | Scoring checkpoints. Never used for training or tuning. |
| `dev_v1.jsonl` | 20 per tier 1-4 | 8,000,000 upwards | Quick checks during development. |
| `heldout_nav_v1.jsonl` | 100 per tier 2-4 | the maps of `heldout_v1` | The same layouts and tick caps without monsters: navigation scored apart from combat. |
| `dev_nav_v1.jsonl` | 20 per tier 2-4 | the maps of `dev_v1` | The same for quick checks. |

Each line is one map: `name`, `tier`, `seed`, `episode_ticks`, `style`, `exit_style`, `grid` (the ASCII
layout), `options` (keyword arguments for `training.mapgen.udmf.build_wad`; only ambush flags are set,
everything else is the reference dialect) and `counts`.

The suites store grids, not seeds, so later changes to the layout code cannot alter them. They were
written once with

    python -m training.mapgen.cli --freeze training/suites/heldout_v1.jsonl --n 100 --tiers 1,2,3,4 --seed 9000000
    python -m training.mapgen.cli --freeze training/suites/dev_v1.jsonl --n 20 --tiers 1,2,3,4 --seed 8000000

and are never regenerated: `--freeze` refuses to overwrite a file, training sets refuse seeds from
8,000,000 upwards and skip any grid that appears in a suite, and `checksums.json` holds the SHA-256 of
each suite file and of the WADs built from it (checked by `training/tests/test_mapgen.py`).

The two navigation suites add no new layouts. They are the tier 2, 3 and 4 rows of the suites above
with every monster cell turned into floor and the ambush flags dropped; walls, doors, keys, pickups
and tick caps are unchanged and the rows are numbered alike, so `heldout_nav_v1_t3_004` is
`heldout_v1_t3_004` without its monsters. The reference walker, which never fires, finishes every one of them,
which is not true of the full suites, so the pair of scores separates navigation from combat. They
were written once with

    python -m training.mapgen.cli --freeze training/suites/heldout_nav_v1.jsonl --from-suite training/suites/heldout_v1.jsonl --tiers 2,3,4 --strip-monsters
    python -m training.mapgen.cli --freeze training/suites/dev_nav_v1.jsonl --from-suite training/suites/dev_v1.jsonl --tiers 2,3,4 --strip-monsters

Rebuild the WADs with

    python -m training.mapgen.cli --from-suite training/suites/heldout_v1.jsonl --out data/maps/heldout_v1

All 480 maps load in `doom_env`, and on every map with an exit the reference walker
(`training/mapgen/walker.py`, monsters removed) reaches the pad in at most 34% of the tick cap.
