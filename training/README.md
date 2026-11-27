# Training pipeline

Everything under `training/` is training-time code: it produces the two weights files in
`submission/weights/` and measures them. Nothing here is imported by the agent in `submission/`.
Training runs on the official `doom_env` wrapper with the vanilla engine; the workers build
exactly the network inputs the agent builds at evaluation, with the agent's own modules.

Run every command from the repository root in the project environment (`uv run python ...` or
`.venv/bin/python ...`). Generated maps and datasets go to `data/`, run directories to `runs/`
(neither is tracked). The logs that the documents quote are kept in `training/logs/`.

```
 1 maps ──> 2 teacher ──┐
        └─> 3 exit detector ──┐
        └─────────────────────┴─> 4 PPO: combat alone, then the unified loop ──> 5 evaluation and promotion
```

| Stage | Code | Produces | Tracked logs |
|---|---|---|---|
| 1. Maps | `mapgen/`, `suites/` | training maps with a manifest; frozen held-out suites | `suites/checksums.json` |
| 2. Navigation teacher | `teacher/` | labels inside the PPO workers; nothing shipped | `logs/teacher_gate.csv`, `logs/teacher_failures.json` |
| 3. Exit detector | `exit_label.py`, `exitdet/` | `submission/weights/exit_detector.pt` | `logs/exit_detector*.csv`, `logs/exit_detector.json` |
| 4. PPO with curiosity and curriculum | `rl/`, `configs/` | checkpoints; `submission/weights/policy.pt` through `rl/export.py`, from one checkpoint or from a weight average of two (`rl/average.py`) | `logs/unified_full_2_train_log.csv` (phase two of the run behind the shipped policy), `logs/manager_probe.txt`, `logs/smoke_arena.*`, `logs/unified_smoke.*`, `logs/unified_combat_keep.csv` |
| 5. Evaluation and promotion | `scorer.py`, `runner.py`, `evaluate.py`, `promote.py`, `final_pick.py`, `mapmeta.py`, `demo.py`, `bench/`, `ops/` | scores of every checkpoint on the frozen suites; the ship gate; the final-pick report | `logs/results.csv`, `logs/final_pick_report_*.txt`, `logs/final_pick_0618.txt`, `logs/final_pick_0630_with_blend.txt`, `logs/final_pick_three_seeds_*.txt`, `logs/failure_breakdown_main_0135.txt`, `logs/latency.txt`, `logs/fresh_clone_validation.txt` |

What is learned by which stage: the exit detector by supervised learning in stage 3; everything
in `PolicyNet` by stage 4. Combat (the attack and weapon heads, and every button while an enemy
is visible) is learned by PPO alone. Navigation is learned by PPO together with distillation from
the stage 2 teacher, whose weight is stepped down only while the held-out exit rate of the policy
alone holds. In the runs behind the shipped policy that weight reached zero 44.2 M steps into
phase two; every later step, of the main run and of the branch that left it at 125.3 M steps, is
PPO alone ("What was run", below).

## 1. Maps

Files: `mapgen/udmf.py` (ASCII grid to UDMF map, the legend), `mapgen/layouts.py` (seeded layouts
per tier), `mapgen/solver.py` (solvability and route cost), `mapgen/randomize.py` (start pose,
looks, monsters), `mapgen/walker.py` (reference walker for checking maps in the engine),
`mapgen/cli.py`, `mapgen/authored.py`, `suites/`. Details: [mapgen/README.md](mapgen/README.md),
[suites/README.md](suites/README.md).

A map is a grid of 64-unit cells drawn as text. The generator cuts a floor plan in one of six
styles (corridor of chambers, arena, rooms sharing walls, scattered rooms with corridors, maze,
cave), then places the start, the exit pad, locks on chokepoints with their keys in the part of the
map reachable before them, doors, monsters and pickups. Tiers: 0 monster-free navigation (one
map in three with a key and its locked door), 1 combat arenas, 2 multi-room exploration, 3 keys
and locked doors, 4 gauntlet. Every draft is checked by a search over (cell, keys held): all open
cells reachable, exit reachable, locks really needed; the tick cap (2,100 to 4,200) is at least
three times the cost of the shortest route.

Randomisation (`--randomize P`, on a fraction P of the maps): start heading (quarter turns or any
angle) and a start offset of up to 16 units, so walls do not line up with the agent's start frame;
monster facing and extra ambush flags; sector light, floor, wall and ceiling textures; door delay.
The exit pad keeps its floor texture.

The wide profile (`--profile wide`, tiers 1 to 4) holds maps to wider limits: longer routes, more
monsters, and a tick cap of 1.5 to 3 times the route cost where the base profile has at least 3
([mapgen/README.md](mapgen/README.md), "The wide profile"). The `reference` profiles restrict the
legend to the monsters and pickups of the template's reference maps (zombieman, imp, chaingunner,
stimpack).

Three training sets were used. `train` (1,500 maps) for the combat run and phase one; `train_v2`
(3,000 maps, 90% of them with a start on the teacher's grid) for the first 5.1 M steps of phase
two; `train_v3` (`train_v2` plus 1,500 wide maps, 4,500 in all, 4,056 with teacher labels) for the
rest of phase two. The commands for the last two are in `mapgen/README.md`. A fourth set,
`train_mix_ref50`, is the map set of the branch of phase two: all 3,000 maps of `train_ref_v1`
(the two reference profiles, 1,500 maps each) plus 3,000 maps of `train_v3` chosen at random, as
many per tier (450, 750, 750 and 1,050 in tiers 1 to 4, no tier 0). `rl/mix_manifests.py` writes
such a manifest; the commands are in the top-level README (step 1b).

```bash
# the first training set: 300 maps per tier, half of them randomised (1,500 maps, 743 randomised)
python -m training.mapgen.cli --out data/maps/train --n 300 --tiers 0,1,2,3,4 --seed 0 --randomize 0.5
python -m training.mapgen.cli --stats data/maps/train/manifest.jsonl          # per-tier table
python -m training.mapgen.cli --verify data/maps/train/manifest.jsonl --workers 3   # loads every map in the engine

# the frozen suites rebuilt as WADs (held-out check inside training, and scoring)
python -m training.mapgen.cli --from-suite training/suites/dev_v1.jsonl     --out data/maps/dev
python -m training.mapgen.cli --from-suite training/suites/dev_nav_v1.jsonl --out data/maps/dev_nav
```

Held-out layouts. `suites/heldout_v1.jsonl` (100 maps per tier 1 to 4) and `suites/dev_v1.jsonl`
(20 per tier) come from seeds that training sets refuse, and are stored as grids so later changes
to the generator cannot alter them. `heldout_nav_v1` and `dev_nav_v1` are the tier 2 to 4 maps of
those suites with the monsters removed, to score navigation apart from combat. `checksums.json`
pins every generated suite file and the WADs built from it. `heldout_wide_v1` (50 per tier) and `dev_wide_v1`
(15 per tier) are the same for the wide profile, `heldout_ref_v1` (60 per tier) and `dev_ref_v1`
(20 per tier) for the reference profiles. `mapgen/authored.py` builds validation suites from
layouts drawn by a language model and filtered by the generator's own checks: `authored_v1`
(59 maps) and `authored_hard_v1` (60 maps drawn to exceed the generator's size and route limits);
these two are not in `checksums.json`. The base, navigation and reference suites are described in
[suites/README.md](suites/README.md), the wide ones in [mapgen/README.md](mapgen/README.md), the
authored ones in the docstring of `mapgen/authored.py`.

Logs: the generator writes `manifest.jsonl` next to the maps (one row per map: tier, seed, tick
cap, grid, options, counts). Tests: `tests/test_mapgen.py`, `tests/test_authored.py`.

## 2. Navigation teacher (training-time scaffolding)

Files: `teacher/core.py` (decision rules, door routine), `teacher/grid.py` (planning grid read
from the memory-map crop, shortest-path search), `teacher/steer.py` (move, strafe and turn keys
from dot and cross products), `teacher/policy.py` (standalone driver for evaluation and the
viewer), `teacher/gate.py` (evaluation on monster-free suites), `teacher/debug.py`.

The teacher is a hand-written explorer. It contains graph search, which is why it lives here and
not in `submission/`. Its label at a step is a function of what the student network receives at
that step and of nothing else: the memory-map crop, the pose estimate, the observation (rays,
entities, vitals), the encoder scalars and the previous buttons. It never reads the true position,
the map file or any engine state; a test replays an episode through a fresh teacher that is given
no environment object and requires identical labels (`tests/test_teacher.py`).

Per step it supplies: a sub-goal and a target cell (the manager's labels), the move, strafe, turn
and USE heads (the controller's labels) and a field of travel steps to the target (the label of the
value-propagation block). It supplies **no** label for the attack and weapon heads, no controller
label on any step with a visible enemy, and no label at all on maps whose start is off the 64-unit
grid (581 of the 1,500 maps of `train`, 444 of the 4,500 of `train_v3`), where its planning grid
does not fit. Those steps and maps are trained by reinforcement alone.

```bash
# exit rate, coverage, door reversals and stuck rate on monster-free rebuilds of a suite
python -m training.teacher.gate --tag dev_clean                                   # dev_v1, exact exit observation
python -m training.teacher.gate --tag dev_noisy --exit noisy --seeds 0,1,2        # with missed, displaced and false exit sightings
python -m training.teacher.gate --tag heldout_nav_clean --suite training/suites/heldout_nav_v1.jsonl
python -m training.teacher.gate --tag train_keep --suite data/maps/train/manifest.jsonl --tiers 0,2,3,4 --start keep
```

Logs: `logs/teacher_gate.csv` (one row per tier and run), `logs/teacher_failures.json` (failed
episodes with their cause); per-episode rows in `runs/teacher/<tag>.csv`. Measured on the
monster-free held-out layouts (`heldout_nav_v1`, 100 per tier, true tick cap as budget, start on
the grid): exit rate 1.00 / 0.98 / 0.99 on tiers 2 / 3 / 4, sectors visited 0.98, no door
reversal. On the training maps with their stored start poses: 0.99 on the 746 maps whose start is
on the grid and 0.29 on the 454 that are off it, which is why those get no labels. With the
monsters left in and nothing to fight with, the teacher reaches the exit on about a third of the
development maps: navigation alone is not enough, and combat has to come from reinforcement.

## 3. Exit detector

Files: `exit_label.py` (exit observation from true pose and geometry, and a noise model),
`exitdet/collect.py` (labelled frames), `exitdet/textures.py` (texture split between training and
"unseen" evaluation), `exitdet/data.py`, `exitdet/augment.py`, `exitdet/train.py`,
`exitdet/evaluate.py` (frame tables), `exitdet/episodes.py` (whole episodes),
`exitdet/fallback.py` (what the detector is worth to navigation).

The exit pad is not in the rays or the entity list; it shows only in the RGB frame. The detector
(`doomnet.model.ExitDetector`) outputs a visibility logit and the pad's position in the agent's
frame. It is trained on frames labelled by geometry: maps of all tiers with randomised textures
and light, poses from scripted wandering and from random cells and headings, extra red look-alikes
(red keys, red doors, imps and their fireballs), and frames from whole episodes that include
fights, corpses and blood. The scripted wanderer in `collect.py` walks, opens doors and shoots; it
exists to produce frames and is not a source of policy labels. Every batch is augmented (flip,
damage and pickup tints, gain, brightness, contrast, gamma, saturation, hue, noise, blur). The loss
is binary cross-entropy on visibility plus a Huber loss on position.

```bash
python -m training.exitdet.collect --sets train,extra,val,calib,heldout,unseen,weapons --workers 3
python -m training.exitdet.collect --sets play,play2,play_calib,play_heldout --miner EARLIER_WEIGHTS.pt --workers 3
python -m training.exitdet.train --tag NAME --save        # trains on train,extra,play,play2; --save publishes weights and tables
python -m training.exitdet.evaluate --weights submission/weights/exit_detector.pt
python -m training.exitdet.episodes --suite training/suites/dev_v1.jsonl --workers 3     # whole episodes
python -m training.exitdet.fallback --suite training/suites/heldout_v1.jsonl --workers 3 # teacher with label / detector / no exit sensor
```

Logs: `logs/exit_detector.csv` and `.json` (every set and subset: precision, recall by distance
band, position error, false detections by look-alike, latency), `logs/exit_detector_train.csv`
(training curve), `logs/exit_detector_data.csv` (frame sets), `logs/exit_detector_episodes.csv`
(whole episodes), `logs/exit_detector_robustness.csv` (fixed colour, blur and noise changes),
and `logs/exit_detector_fallback.csv`, which `fallback.py` writes.

Measured for the weights with sha256 prefix `8985406ed0a6` (235,520 training frames from 3,680
maps), at confidence 0.5, on maps never trained on:

| Set | Frames | Precision | Recall 0-400 units | Recall 400-640 | Median position error 0-400 |
|---|---|---|---|---|---|
| held-out maps | 20,480 | 0.9991 | 0.936 | 0.787 | 6.3 units |
| unseen textures | 20,480 | 0.9979 | 0.940 | 0.723 | 6.8 units |
| weapon varied | 12,800 | 0.9993 | 0.941 | 0.710 | 6.3 units |
| real play with fights | 12,800 | 0.9964 | 0.911 | 0.719 | 6.8 units |

Recall within 400 units is 0.95 to 0.97 with no monster in view and 0.81 to 0.89 with one: a
monster or corpse standing on the pad hides it, and the label counts such a pad as visible. One
forward pass takes 0.5 ms on one laptop thread. The detector recognises the red pad of the grid
layouts and its two near-identical variants; on a pad drawn with any other floor texture recall
within 400 units is 0.00 to 0.14 (`padflats` rows of the log). In that case the agent can still end the map by walking onto the
pad: the teacher with no exit sensor at all reaches the exit on 0.75 / 0.85 / 0.91 of the
monster-free tier 2 / 3 / 4 held-out layouts, against 1.00 / 0.98 / 0.99 with the detector.

While the policy trains, the exit sighting given to the front end is computed from geometry
(`exit_label.exit_observation`) and degraded by `exit_label.noisy` (missed sightings, position
noise that grows with distance, occasional false sightings), so that no frame has to be rendered.
`vec.exit_source = "detector"` runs the real detector in the workers instead. The shipped agent
always uses the detector. Every run behind the shipped policy trained on the geometric sighting
with the noise model (`exit_source: "oracle"`, `exit_noise: true` in the configs stored with the
checkpoints); the detector was never in the training loop.

## 4. PPO with curiosity and curriculum

Files (details and every setting in [rl/README.md](rl/README.md)):

| File | Role |
|---|---|
| `rl/vecenv.py` | worker processes: engine, the agent's own odometry, memory map and encoder, reward, labels; inputs returned through shared memory |
| `rl/rollout.py` | the collection loop: one batched forward pass per step, teacher mixing, held-out exit rate |
| `rl/buffer.py` | rollout storage, GAE for two reward streams, chunks for truncated backpropagation through time |
| `rl/ppo.py` | masked factored action distribution, the loss, schedules, the distillation schedule |
| `rl/reward.py` | extrinsic reward shaped like the score |
| `rl/intrinsic.py` | curiosity reward from the agent's own memory map |
| `rl/curriculum.py` | stages over map tiers, then adaptive sampling |
| `rl/train.py` | entry point, CSV log, checkpoints |
| `rl/export.py`, `rl/provenance.py` | write a checkpoint in the agent's format; tie every checkpoint to the code that trained it |
| `rl/average.py`, `rl/mix_manifests.py` | weighted average of the network weights of checkpoints of one architecture; the map set of the branch (all of one manifest plus as many maps of another, tier by tier) |
| `rl/play.py`, `rl/eval_alone.py`, `rl/agree.py`, `rl/bench.py`, `rl/select_maps.py` | play a checkpoint, exit rate alone, agreement with the teacher, throughput, map filters |
| `configs/` | `combat_1.json`, `unified_full.json`, `unified_nav.json`, `unified_smoke.json`, each unified config with a `.txt` that explains every setting, and `LAUNCH.txt` |

**The implementation.** Plain torch, no RL library. One decision is six categorical heads (move,
strafe, turn, attack, use, weapon); every 8 steps the manager also chooses a sub-goal and a target
cell. The loss is

    L = L_pi + vf_coef L_v + vf_int_coef L_v_int - entropy bonuses + w_distill (L_distill + field_coef L_field)

with a clipped surrogate `L_pi` over the joint log-probability of the six heads and, on manager
steps, the manager's two choices; clipped value losses for two value heads (extrinsic and
intrinsic return, each normalised by running statistics); an entropy bonus per head; advantages
from two GAE streams, `A_ext + 0.3 A_int`; recurrent state stored at the start of every 16-step
chunk. The USE and ATTACK masks of `doomnet.spec.action_masks` are applied to the logits in the
rollout and in the loss alike.

**Reward shaping** (`rl/reward.py`). Computed in the workers from privileged state, which the
agent never sees. Each scoring category is paid as its events happen, so the return of an episode
is proportional to the local scorer's composite: reaching the exit; health and time left at the
exit; each key, door and locked door; each sector entered for the first time; each kill; minus
damage taken, death (more when earlier), shots that hit nothing, and a door closing again because
USE was pressed twice. Categories a map does not have are dropped and the rest rescaled.
`hit_bonus` (a bonus per shot that hits a monster, not part of the score) is 0 by default and 0.05
in `combat_1.json` and the unified configs.

**Curiosity** (`rl/intrinsic.py`). Two episodic terms from the agent's own memory-map crop and
pose estimate, no privileged state: 0.01 for every 64-unit cell observed for the first time (free
or wall channel on), and 0.02/√n when the agent enters a cell for the n-th time. The second is paid
on entry, so standing still earns nothing and pacing between two cells decays. Every run behind
the shipped policy had it on: over iterations 451 to 899 of phase two the curiosity return is about
3.8 per episode beside a task return of 8.4 (columns `return_int` and `return_ext` of
`logs/unified_full_2_train_log.csv`). There is no ablation: `rl/configs/curiosity_on.json` and
`curiosity_off.json` exist, but the only pair of runs made with them lasted five minutes on a
laptop (about 200,000 steps each), too short to show a difference, so no gain from curiosity is claimed.

**Curriculum** (`rl/curriculum.py`). Fixed stages, each with sampling weights per tier, then
adaptive sampling of tiers 1 to 4 with weight (1 − recent exit rate) + a floor. `unified_full.json`:
navigation (70% tier 0, 30% arenas) for 15 M steps, doors for 25 M, keys for 40 M, gauntlet for
60 M, arenas in every stage so that combat is not forgotten. `unified_full_2.json` (phase two):
all tiers from the first step, weighted towards tier 4, then adaptive sampling after 30 M steps.
The stage name is written to every log row.

**Time budget.** The tick cap of a map is not observable. The policy takes an assumed budget as an
input: in training the true cap ±10%, or with probability 0.25 the evaluation default; at
evaluation the constant 525 decisions, the shortest cap. The unified configs make the cap terminal
(`bootstrap_on_truncation: false`), so running out of time costs the exit reward and when to leave
is learned from reward.

**4a. Combat by PPO alone.** Tier 1 arenas, extrinsic reward and curiosity, no teacher, from random
initialisation. The current code also gives a fresh network an offset on the output bias of the
weapon head (`head_bias`, an ordinary trained parameter afterwards), because a near-uniform weapon
head presses "next weapon" on half of all steps and the pistol is then almost never ready; that was
found in the arena check of `logs/smoke_arena.txt`. The combat run behind the shipped policy was
started before this option existed (the config stored in its checkpoints has no such entry), so we
do not claim it for that run.

```bash
python -u -m training.rl.train --maps data/maps/train/manifest.jsonl --config training/configs/combat_1.json \
    --workers 96 --envs-per-worker 12 --out runs/combat_1
```

**4b. The unified loop: the student drives, the teacher labels.** Started from the combat
checkpoint. Every worker runs the stage 2 teacher next to the agent's front end and stores its
labels with each observation. On a step with controller labels the teacher's buttons are executed
with probability β (1.0 falling linearly to 0.1 over the first 30 M steps), and on a manager step
the same coin forces the manager's sub-goal and target to the teacher's; forced steps are left out
of `L_pi` and kept in the value and distillation terms. Steps with a visible enemy and maps
without labels are never mixed. `L_distill` is the cross-entropy to the labels (four controller
heads, sub-goal, target); `L_field` regresses the network's value field on 0.95 ^ (teacher's
travel steps to the target).

The distillation weight is not annealed on a timer. Every 150 iterations the policy alone plays 384
held-out episodes with the agent's decoding and the evaluation budget (`--heldout`). The weight
steps down one level (1, 0.5, 0.25, 0.1, 0.03, 0) only if that exit rate is at least `min_rate` and
has not fallen by more than `tolerance` (0.06) since the last step down; two bad reports in a row
step it back up. Mixing stops when the weight reaches 0.

```bash
# phase one
python -u -m training.rl.train --maps data/maps/train/manifest.jsonl --config training/configs/unified_full.json \
    --init runs/combat_1/latest.pt --heldout data/maps/dev/manifest.jsonl \
    --workers 128 --envs-per-worker 12 --out runs/unified_full_1

# phase two, from the last checkpoint of phase one, on the larger map set (training/configs/unified_full_2.txt)
python -u -m training.rl.train --maps data/maps/train_v3/manifest.jsonl --config training/configs/unified_full_2.json \
    --init runs/unified_full_1/latest.pt --heldout data/maps/dev/manifest.jsonl \
    --workers 128 --envs-per-worker 12 --out runs/unified_full_2

# 25-minute laptop check of the same loop (training/configs/unified_smoke.txt)
python -m training.rl.select_maps --maps data/maps/train/manifest.jsonl --out data/maps/unified_smoke_train.jsonl \
    --tiers 0,2 --max-monsters 0 --on-grid
python -u -m training.rl.train --maps data/maps/unified_smoke_train.jsonl \
    --config training/configs/unified_smoke.json --out runs/rl/unified_smoke --max-minutes 25
```

The preflight, the unattended restart loop, the memory needed and the log columns to watch are in
`configs/LAUNCH.txt`.

Logs: every run writes `train_log.csv` (one row per iteration: steps, stage, reward terms, exit
rate per tier, kills, deaths, door reversals, losses, KL, β, agreement with the teacher,
`heldout_exit_rate`, `distill_weight`, throughput, failure counters), `config.json` and
checkpoints into its `--out` directory. Curated copies in `logs/`: `smoke_arena.csv` and `.txt`
(PPO from random initialisation on single-room arenas, 10 minutes), `unified_smoke.csv` and `.txt`
(the unified loop from random initialisation on monster-free maps, 20 minutes, with the
agreement and exit-rate measurements), `unified_combat_keep.csv` (the unified loop started from a
combat checkpoint), and `unified_full_2_train_log.csv`, the log of phase two of the run behind the
shipped policy.

**What was run for the shipped policy.** Three runs in a row on rented machines, each started from
the last checkpoint of the one before, and a branch of the third on a second machine. The step
counts of the first two are the counters stored in their last checkpoints; their `train_log.csv`
files were not brought back from the machines; the branch's log is not in this repository either.

| Run | Config | Maps | Steps | Teacher |
|---|---|---|---|---|
| `combat_1` | `configs/combat_1.json` | tier 1 arenas of `train` | 118.8 M (1,900 iterations) | none: PPO alone from random weights |
| `unified_full_1`, phase one | `configs/unified_full.json` | `train`, all tiers by curriculum (it ended in the keys stage) | 49.0 M (498 iterations) | β 1 → 0.1 over 30 M steps; weight 1 → 0.5 → 0.25 → 0.1 at held-out exit rates 0.45, 0.50, 0.63 |
| `unified_full_2`, phase two | `configs/unified_full_2.json` | `train_v2` for 5.1 M steps, then `train_v3` | which checkpoints are in the shipped weights is stated in the top-level README; the run was scored at 172.0 M (5:25 AM, tag `main_0525_dev`) and at 196.6 M (iteration 2,000, 6:08 AM, `main_final_M` in the final-pick logs) | β 0.15 → 0.1; weight 0.25, then 0.1 at 14.7 M, 0.03 at 29.5 M, **0 at 44.2 M**; β is 0 from the next iteration on |
| `clone_mix_full`, a branch of phase two | `configs/clone_mix_full.json` | `train_mix_ref50`: 3,000 organizer-style maps and 3,000 maps of `train_v3` | resumed from a copy of the main run's checkpoint at 125.3 M (iteration 1,275), same step counter; scored at 164.7 M (iteration 1,675, 6:08 AM, `clone_final_M`), 39.3 M steps of its own | none: the weight was 0 before the branch started |

What `logs/unified_full_2_train_log.csv` shows (iterations 1 to 899 of phase two, 88.4 M steps,
three hours of wall-clock time; the columns are named in brackets):

- *The gate* (`heldout_exit_rate`, `distill_weight`). Held-out exit rate of the policy alone: 0.67
  at 14.7 M steps, 0.72 at 29.5 M, 0.77 at 44.2 M, where the weight reached zero, then 0.83 at
  59.0 M and 0.85 at 73.7 M with no teacher in the loss or in the loop.
- *The manager after the teacher* (`ent_subgoal`, `ent_target`, `agree_subgoal`, `agree_target`).
  At iteration 450 the sub-goal head has an entropy of 0.68 nats and agrees with the teacher's
  sub-goal on 0.84 of the labelled manager steps; the target pointer has 1.56 nats and agrees on
  0.63. Fifty iterations later the figures are 1.92 and 0.31, 3.76 and 0.46; from iteration 700
  on they stay near 2.03 (the maximum for 8 sub-goals is 2.08) and 0.15, 7.7 (the maximum for
  4,096 cells is 8.32) and 0.03 to 0.07. With no label and no forced choice left, nothing but the
  PPO gradient and the entropy bonus acts on these two heads, and they drifted to near-uniform
  outputs. Scores rose over the same steps (the gate above; `logs/results.csv`, tags
  `main_0135_dev` to `main_0305_dev`). `logs/manager_probe.txt` repeats the measurement on the
  promoted checkpoints with recorded inputs, and shows how far the controller's outputs move when
  the manager's choice is swapped. The consequence for what the agent can be said to do is stated
  in `docs/ARCHITECTURE.md`, section 6e.
- *Agreement of the buttons* (`agree`). The controller's agreement with the teacher's move, strafe,
  turn and USE labels goes from 0.77 at iteration 450 to 0.59 at iteration 899: the policy leaves
  the teacher's way of moving while its exit rate rises.

**Weight average** (`rl/average.py`). The weighted mean of the network tensors of checkpoints of one
architecture: every tensor of the state dict is summed in float64 with the given weights and stored
as float32. Iteration, step counter, the front-end entry of the provenance record and the code
fingerprint come from the last file; the network entry of the record is computed again for the
averaged network; optimiser, normalisation, curriculum and distillation state are dropped, so
training cannot resume from the result. Checkpoints of another architecture are refused. The agent
shipped at 6:42 AM is the equal-weight average of the branch at iteration 1,675 and the main run at
iteration 2,000 (the command is step 4e of the top-level README, with the decoding setting it
stores in the exported file). Run on those two checkpoints the tool gives the state dict of the
policy file shipped at 6:42 AM (sha256 prefix `4dd361277fb1`) bit for bit, all 73 tensors. A weight
average is not a trained network: what was and was not measured on it is in the top-level README
under "Known limits".

Throughput. Training machine, 128 worker processes × 12 environments with the learner on one GPU:
8,000 to 12,000 environment steps per second in phase two (`env_steps_per_s`: median 7,975 up to
iteration 760, while a second run shared the machine, and 11,762 after). On a 15-core laptop under
load: about 1,100 for the whole loop with 3 workers and the learner on the laptop GPU, where the
acting forward pass and the update are the limits (`rl/README.md`, "Throughput").

Tests: `tests/test_rl.py`, `tests/test_unified.py`, `tests/test_average.py`.

## 5. Evaluation and promotion

Files: `mapmeta.py` (static facts of a map), `scorer.py` (per-episode metrics and the composite),
`runner.py` (episodes in worker processes, with the official time limits), `evaluate.py` (suites,
results log, comparison), `promote.py` (score a checkpoint as a candidate agent, ship gate,
decision rule), `ops/remote.py` (run an evaluation on a remote Linux machine and pull the rows
back), `demo.py` (viewer), `bench/agent_latency.py`, `bench/model_latency.py`.

The local scorer measures the categories of `docs/EVALUATION.md` from privileged state: exit,
progression (keys, doors, locked doors), exploration (sectors visited) and combat (kills, minus
damage and early death), plus diagnostics that the score does not contain: stuck rate (moved less
than 8 units over 10 decisions), door reversals, exit seen but not reached, and the drift of the
agent's pose estimate. The category weights are our estimate and are reported with two alternative
weightings, so a ranking that depends on the guess is visible.

```bash
# export a checkpoint in the agent's format and check it in a real DoomAgent
python -m training.rl.export runs/unified_full_1/latest.pt --dest runs/unified_full_1/policy.pt \
    --episode data/maps/dev/dev_v1_t2_000.wad

# score the agent in submission/ on the frozen held-out suite under the official time limits
python -m training.evaluate --policy training.runner:AgentPolicy --suite training/suites/heldout_v1.jsonl --tag NAME
python -m training.evaluate --table                 # latest result of every tag
python -m training.evaluate --compare OLD NEW       # exit status 0 only if NEW beats OLD

# the same for a checkpoint that is not in submission/ yet: build a candidate agent around it and score that
python -m training.promote score runs/unified_full_1/latest.pt --tag NAME --suite heldout
python -m training.promote check runs/promote/NAME  # ship gate: size, provenance, validator, tests, latency
python -m training.promote compare OLD NEW          # the decision rule

# look at episodes: video and contact sheets of frame, memory map, sub-goal, target, pose drift
python -m training.demo --policy training.runner:AgentPolicy --episodes runs/eval/NAME.csv --failed --video none --out runs/demo/NAME_failures

# latency of the whole agent on one thread, and the official validator
python training/bench/agent_latency.py
python validate.py
```

Promotion rule during the run: a policy file was copied to `submission/weights/policy.pt` only when the
comparison said it was better than the file already there, that is when its tier-weighted score on the frozen
suite is higher by more than one composite point and by more than two standard errors of the
paired difference; `promote compare` gives a tie to the checkpoint that has had more reinforcement
learning. `promote check` is the ship gate for a candidate, nine checks: the directory is small;
its policy file loads and gives the outputs its provenance record stores; `validate.py` passes
without the untrained switch; the rule lint, fuzz, agent and door test files pass (four checks);
the mean `act()` time on one thread is under 20 ms; and every file of the candidate but the policy
file is identical to the working tree's `submission/`. Neither command writes into
`submission/weights/`. `rl/export.py --check` verifies the shipped file against the current code.

How it was used. During the run a checkpoint was pulled about once an hour and scored on `dev_v1`
and `dev_wide_v1` (later also `dev_ref_v1`) against the shipped agent. Up to 5:30 AM four promotions
followed the first shipped agent (`unified_it425`, 11:23 PM): `main_0035` (dev 68.8 against 57.7,
wide 49.9 against 33.7), `main_0135` (dev 77.0 against 68.8, wide 58.4 against 49.9), `main_0305`
(dev 77.8 against 77.0, a tie; wide 69.9 against 58.4, a win beyond noise) and, at 5:28 AM,
`main_0525` (172.0 M steps; dev 89.9 against 77.8, a win beyond noise; wide 64.4 against 69.9,
lower but inside the noise margin). Two checkpoints in between were scored and not promoted:
`main_0355` (dev 75.5, wide 62.8, both below the shipped agent) and `main_0445` (dev 80.5 against
77.8, wide 62.3 against 69.9: a gain on one suite and a loss on the other, each inside its noise
margin). At 5:37 AM the same checkpoint was shipped again with another decoding setting stored in
its policy file (`main_0525_M`: move, strafe, turn, USE and weapon take their most likely option).
The agent shipped in the end is named in the top-level README. Side runs are in
`logs/results.csv` as well: a wider network (`wide_0035`, `configs/unified_wide.json`, stopped), a
copy of phase two with other reward settings (`copy_*`,
`configs/unified_full_2b.json`, stopped), and clones of phase two on map sets that mix in the
reference profiles (`clone_*`, `configs/clone_ref.json`; `mix_*`, `configs/clone_mix_full.json`,
started from the checkpoint at 125 M steps).

The final choice (`final_pick.py`) is made on four held-out suites: `heldout_v1`,
`heldout_wide_v1`, `authored_v1` and `authored_hard_v1`, with weights 0.35, 0.2, 0.2 and 0.25.
`score` plays an agent on all four through the real agent; `report` prints every suite and tier
with standard errors, the split by monsters, and for every pair of agents the paired differences
and the decision: B replaces A only if its weighted headline is better by more than noise and it
is not worse by more than noise on any single suite. The reports made up to 5:37 AM are
`logs/final_pick_report_*.txt`; `logs/failure_breakdown_main_0135.txt` is a breakdown of the
episodes one of them is made from (deaths by monster, timeouts by cause).

The last comparison (6:08 to 6:42 AM) had four challengers against the holder `main_0525_M`: the
main run at 196.6 M steps, the branch at 164.7 M, the equal-weight average of those two
(`blend_final_M`) and the average of the main run's last nine checkpoints. The organizer-style
suite `heldout_ref_v1` was scored beside the four suites; it is not part of the rule. On one
engine seed the report's rule said that the average of the two branches replaces the holder
(+1.86 on the headline, margin 1.82). The two generated suites and the organizer-style suite were
then played on two more engine seeds for the two branches and their average. On three seeds none
of the three is ahead of the holder's headline by more than noise (the average: +0.70 ± 0.74; the
nine-checkpoint average was level on its one seed), so the rule alone keeps the holder. The
average was shipped at 6:42 AM by a decision of the team lead, because it is not worse by more
than noise on any suite and is ahead on the organizer-style suite by +6.64 ± 1.30. The top-level
README states that decision with all its numbers ("How the shipped agent was chosen"). Logs:
`logs/final_pick_0618.txt` (three challengers, one seed), `logs/final_pick_0630_with_blend.txt`
(with the average of the two branches, one seed), `logs/final_pick_three_seeds_shipped_main_clone.txt`
and `logs/final_pick_three_seeds_shipped_blend_main.txt` (three seeds). `final_pick score` plays
the generated suites on one engine seed; the further seeds and the split of the organizer-style
suite into its two halves were run by scripts outside the repository through
`training.promote.score_local`, and `promote score --seeds` plays any seeds from the command line.

Logs: `logs/results.csv` (one row per tier and one for all tiers, per evaluation tag; the tags of
the baselines and of every agent that was shipped are listed in the table of the top-level
README); per-episode rows in `runs/eval/<tag>.csv` (not tracked).

Tests: `tests/test_scorer.py`, `tests/test_promote.py`, `tests/test_final_pick.py`, `tests/test_demo.py`, `tests/test_agent.py`,
`tests/test_doors.py`, `tests/test_fuzz.py`, `tests/test_rules.py`.
