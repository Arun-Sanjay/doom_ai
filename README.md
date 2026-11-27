# DoomNet

Our entry for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**, built from the official
starter template. The agent plays unseen levels through the official `doom_env` wrapper: it has to
survive fights, cross multi-room layouts, find keycards, open locked doors and reach the exit.

**How the agent decides.** Every action comes from trained neural network weights. The policy is
trained by reinforcement learning: our own recurrent PPO (`training/rl/`), with reward shaped like the
evaluation categories, a curiosity reward and a curriculum. Combat is learned by PPO alone, from
random weights. Navigation is kickstarted by a scripted teacher that exists only at training time
(`training/teacher/`, never imported by the agent). Its weight in the loss is stepped down as the
policy's own held-out exit rate rises; it reached zero 44.2 M steps into the second training phase,
and the shipped weights are the equal-weight average of two branches of that phase, with 152.4 M and 120.4 M further steps of reinforcement learning alone behind them (the first 81.1 M shared).
There is no rule tree, no path search and no aiming maths in `submission/`; the only
hand-written code that touches action choice is two action masks computed from observation fields
(`submission/doomnet/spec.py`), and a lint test enforces this on every push.

**What is not claimed.** The network has a manager that chooses a sub-goal and a target cell. In the
shipped weights its choices carry little information: the sub-goal is close to a uniform draw and the target is
spread over hundreds of map cells. They drifted there after the teacher was switched off, while the
scores kept rising. The measurement is under [Known limits](#known-limits).

**Techniques used to train the shipped agent.** Every row but the last is in the code path that
produced the committed weights; whether the last one is too is stated under
[Held-out suites](#held-out-suites) ("Shipped weights"). The last column is where to read it.

| Technique | What we did | Where |
|---|---|---|
| Recurrent PPO | Our own PPO, no SB3: GRU controller, truncated back-propagation through time over 16-step chunks, clipped policy and value losses, GAE | `training/rl/ppo.py`, `training/rl/buffer.py` |
| Actor-critic | One network: six factored action heads and the manager's two heads (actor); two value heads (critic), one for the task reward and one for the curiosity reward | `submission/doomnet/model.py`, `training/rl/ppo.py` |
| Reward shaping | Terms that mirror the evaluation categories (exit, efficiency, keys and doors, sectors explored, kills) plus small shaping terms: bonus per hit, cost per missed shot, damage, death and door-reversal penalties | `training/rl/reward.py` |
| Curriculum learning | Staged by tier: combat arenas first; then navigation, doors, keys and the full gauntlet in turn, each stage with its own tier mix; after the first 30 M steps of the last phase, adaptive sampling that favours the tier with the lowest recent exit rate | `training/rl/curriculum.py`, `training/configs/` |
| Curiosity | **On. Count-based and map novelty, not RND.** Two episodic terms from the agent's own memory map: a bonus for each map cell observed for the first time, and 1/sqrt(n) on the n-th entry into a cell. Own value head, weight 0.3 on the advantage; it contributes about 3.7 of return per episode beside 8.6 of task return. We ran no ablation without it, so we claim no measured gain from it | `training/rl/intrinsic.py` |
| Imitation, then PPO alone | An imitation term inside PPO, not a separate cloning stage. Combat: PPO alone from random weights. Navigation: the student drives the engine and a scripted teacher labels the states it reaches; the labels enter as a distillation loss beside the PPO loss. Its weight stepped down 0.25, 0.1, 0.03, 0 as the policy's held-out exit rate rose and reached 0 at 44 M steps of the last phase; the shipped weights are the equal-weight average of two branches of that phase, with 152.4 M and 120.4 M further steps of reinforcement learning alone behind them (the first 81.1 M shared) | `training/rl/train.py`, `training/rl/ppo.py`, `training/teacher/` |
| Action masking | Two masks from observation fields, identical in training and inference: USE only with a door ray within reach; ATTACK only with an enemy in view and a gun selected | `submission/doomnet/spec.py` |
| Ego-centric occupancy grid | An 11-channel 64 x 64 grid centred on the agent (free, wall, door, locked door, visited, recency, enemy, item, key, exit, used), cut from a map the agent builds from its own actions and observations. Agent-centred, but kept in the start frame's orientation: it does not rotate with the heading | `submission/doomnet/memory.py`, `submission/doomnet/encode.py` |
| Vectorised environments | 1,536 engine instances in 128 worker processes with batched policy inference on the GPU, 8,000 to 12,000 environment steps per second | `training/rl/vecenv.py`, `training/rl/rollout.py` |
| Procedural maps | Our own generator in the organizers' map format: layouts for every tier, randomised sizes, textures, starts, monsters and pickups, and a solver that proves each map can be finished. 4,500 maps in the main run of the last training phase, 6,000 in its branch (next row); held-out suites frozen as grids | `training/mapgen/`, `training/suites/` |
| Branch on organizer-style maps, weight averaging | A branch of the last phase: the main run's checkpoint at 125.3 M steps, continued with the same settings (PPO alone) on a map set that is half organizer-style (only zombieman, imp and chaingunner, only stimpacks, an exit on every map) and half the main run's own maps, tier by tier. The weights of a checkpoint of each branch can then be averaged, tensor by tensor, into one network of the same architecture. The agent shipped at 6:42 AM is such an average | `training/configs/clone_mix_full.json`, `training/rl/mix_manifests.py`, `training/rl/average.py` |

What is in this repository:

- **The agent** (`submission/`): a policy network (sensors, manager, value propagation, recurrent
  controller; 1,307,153 parameters) that reads a spatial memory the agent builds from its own
  actions and observations, plus a small CNN that finds the exit pad in the RGB frame (416,867
  parameters). Numpy and torch only, CPU only.
- **The training pipeline** (`training/`): a procedural map generator, our own recurrent PPO with
  reward shaping, a curiosity reward and a curriculum, a scripted navigation teacher used as
  training-time scaffolding, the exit detector's data collection and training, a local scorer, the
  promotion and final-pick tools and a viewer.
- **The logs** (`training/logs/`): every scored checkpoint, the training log of the second phase
  (its first 88.4 M steps), the held-out reports, latency and validation records. The numbers below
  name their log; the step counts of the first two runs and the gate history of phase one are read
  from their last checkpoints, which are not in the repository.
- **The documents**: the 2-page brief ([docs/brief/brief.pdf](docs/brief/brief.pdf), source
  [docs/brief/brief.md](docs/brief/brief.md)), the module-by-module description
  ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) and the pipeline in stage order
  ([training/README.md](training/README.md)).

## Results

All scores come from the local scorer (`training/scorer.py`), which measures the four categories of
[docs/EVALUATION.md](docs/EVALUATION.md) on frozen suites of generated maps that training never saw
(`training/suites/`, stored as grids). A composite of 100 is a perfect run. The scorer's category
weights are our own estimate, so the numbers rank our agents against each other and are not the
official score.

### The ladder

Development suite `dev_v1` (20 maps per tier). One row per baseline and per agent that was shipped
up to 5:37 AM, in order, then the agent shipped in the end; each row is in
`training/logs/results.csv` under the tag in its first column. Tier
columns are composites, "all" is tier-weighted (weights 1, 2, 3, 4) with its standard error, the
exit rate is over maps that have an exit, kills are kills over monsters. The teacher weight is the
weight of the distillation term in the loss at that checkpoint. Phase-two rows include the combat
run and all of phase one.

| Agent (tag) | Training behind it | Teacher weight | Tier 1 | Tier 2 | Tier 3 | Tier 4 | All | Exit rate | Kills | Deaths |
|---|---|---|---|---|---|---|---|---|---|---|
| Random buttons (`random_dev`) | none | | 23.7 | 4.0 | −0.6 | −0.7 | 2.7 ± 0.7 | 0.00 | 0.13 | 0.64 |
| Combat only (`combat_only_it1900`) | PPO alone from random weights, 118.8 M steps | none | 89.2 | 63.2 | 27.4 | 14.7 | 35.6 ± 3.0 | 0.30 | 0.51 | 0.13 |
| First unified agent, 11:23 PM (`unified_it425`) | + phase one, 41.8 M | 0.25 | 92.8 | 76.4 | 62.8 | 35.7 | 57.7 ± 4.3 | 0.61 | 0.63 | 0.19 |
| Promoted 12:38 AM (`main_0035`) | phase two, 19.7 M | 0.1 | 87.2 | 84.7 | 81.8 | 46.6 | 68.8 ± 3.9 | 0.71 | 0.72 | 0.18 |
| Promoted 1:38 AM (`main_0135_dev`) | phase two, 49.2 M | 0 | 91.2 | 86.8 | 85.7 | 62.1 | 77.0 ± 3.9 | 0.83 | 0.75 | 0.13 |
| Promoted 3:08 AM (`main_0305_dev`) | phase two, 98.3 M | 0 | 83.2 | 89.0 | 92.9 | 59.5 | 77.8 ± 3.8 | 0.82 | 0.78 | 0.14 |
| Shipped 5:37 AM (`main_0525_M`) | phase two, 172.0 M | 0 | 87.2 | 92.1 | 86.8 | 71.8 | 81.9 ± 3.6 | 0.86 | 0.80 | 0.10 |
| **Shipped**, chosen 6:42 AM (`blend_final_M`) | phase two: average of the main run at 196.6 M and its branch at 164.7 M | 0 | 93.0 | 90.1 | 92.6 | 83.1 | 88.3 ± 2.5 | 0.92 | 0.83 | 0.04 |

Twenty maps per tier is a small sample: a single tier moves by several points between two
checkpoints of the same run (compare tier 1 and tier 4 of the last rows), which is why a promotion
is decided on paired differences and on a second suite, not on one column (see
[training/README.md](training/README.md), stage 5).

### Held-out suites

The final choice between finished agents is made on four held-out suites
(`python -m training.final_pick`): `heldout_v1` (400 generated maps), `heldout_wide_v1` (200 maps
of the wide profile: longer routes, tighter tick caps), `authored_v1` (59 layouts drawn by a
language model and filtered by the generator's checks, five engine seeds each) and
`authored_hard_v1` (60 more, drawn to exceed the generator's size and route limits). The generated
suites come from seeds that training sets refuse; the authored ones have a different author than
the generator that made the training maps. A suite's score is the mean over its maps weighted by tier,
with a map that has no hell knight or cacodemon counted three times; the headline combines the
suites with weights 0.35, 0.2, 0.2 and 0.25 (640 counted maps; tier 1 maps without an exit are
left out). Reports: `training/logs/final_pick_report_*.txt` and, for the last comparison,
`training/logs/final_pick_0618.txt`, `final_pick_0630_with_blend.txt` and the two
`final_pick_three_seeds_*.txt`. These suites are held out of the training data, not out of our
sight: the failure breakdown of the 1:38 AM agent on them is the source of the limits listed below.

A fifth suite, `heldout_ref_v1` (`training/suites/heldout_ref_v1.jsonl`, 240 generated maps from
seeds no training set uses), is in the style of the maps the organizers published with the
template (`maps/`, `tests/fixtures/`): zombieman, imp and
chaingunner are the only monsters, stimpacks the only pickup, and every map has an exit. Half of
each tier has the base sizes (the normal half), half the long routes and crowded rooms of the wide
profile (the dense half). Its score is the tier-weighted mean over its maps. It is not part of the
headline and not part of the decision rule. The maps of the official evaluation are hidden: that
they resemble the published ones is our assumption, and this suite is our generator's imitation
of them, not a sample.

Seeds: engine seeds per map on the two generated suites and on the organizer-style suite; a map
counts once, as the mean over its seeds. The two authored suites are always played on five
seeds. For the first three agents the tracked logs hold no score on the organizer-style suite.

| Agent | Seeds | Headline | `heldout_v1` | `heldout_wide_v1` | `authored_v1` | `authored_hard_v1` | Organizer-style `heldout_ref_v1` |
|---|---|---|---|---|---|---|---|
| Promoted 12:38 AM | 1 | 61.4 ± 1.2 | 78.0 ± 1.5 | 50.3 ± 2.6 | 74.6 ± 3.0 | 36.5 ± 3.1 | |
| Promoted 1:38 AM | 1 | 72.2 ± 1.2 | 83.6 ± 1.3 | 66.2 ± 2.5 | 82.4 ± 2.4 | 52.8 ± 3.4 | |
| Promoted 3:08 AM | 1 | 75.5 ± 1.1 | 85.6 ± 1.2 | 69.4 ± 2.5 | 87.2 ± 2.5 | 56.7 ± 3.0 | |
| Shipped 5:37 AM (`main_0525_M`) | 1 | 78.4 ± 1.2 | 84.5 ± 1.3 | 77.1 ± 2.2 | 86.7 ± 1.9 | 64.4 ± 3.5 | 65.9 ± 2.5 |
| the same agent | 3 | 78.9 ± 1.1 | 86.8 ± 0.9 | 75.5 ± 1.7 | 86.7 ± 1.9 | 64.4 ± 3.5 | 68.3 ± 2.0 |
| **Shipped** (`blend_final_M`) | 3 | 79.6 ± 1.1 | 88.5 ± 0.7 | 75.2 ± 1.7 | 87.2 ± 2.5 | 64.6 ± 3.4 | 75.0 ± 1.8 |

The organizer-style suite by half (120 maps each), on the seeds of the rows above:

| Agent | Seeds | Normal half | Dense half | Both |
|---|---|---|---|---|
| Shipped 5:37 AM (`main_0525_M`) | 3 | 84.4 ± 1.8 | 52.3 ± 2.5 | 68.3 ± 2.0 |
| **Shipped** (`blend_final_M`) | 3 | 88.2 ± 1.4 | 61.8 ± 2.5 | 75.0 ± 1.8 |

The last row of both tables rests on 3,115 held-out episodes played through the real
agent with the shipped decoding: three engine seeds on the two generated suites and five on the two authored ones,
and the organizer-style suite on the seeds named in its row.

Shipped weights: `submission/weights/policy.pt`, sha256 prefix `4dd361277fb1` (the agent
prints it at start). It is the equal-weight average of the network weights (all 73 tensors, 1,307,153 values) of two checkpoints of the same network, made by `training/rl/average.py`: the main run of phase two at 196.6 M steps, and a branch that left the main run at 125.3 M steps and trained on to 164.7 M on a map set that is half organizer-style (39.3 M steps of its own). The step counter stored in the file is the main run's.
Behind the two checkpoints are 404 M environment steps: combat 118.8 M, phase one 49.0 M, the main run of phase two 196.6 M and the branch's own 39.3 M. Decoding setting stored with the policy file:
move, strafe, turn, USE and weapon take their most likely option, attack is sampled, and so are the manager's choices.
The other weights file is `submission/weights/exit_detector.pt`, sha256 prefix `8985406ed0a6`.

### How the shipped agent was chosen

The rule was written down before the last comparison and is the one `python -m training.final_pick
report` applies (`decide` in `training/final_pick.py`): a challenger replaces the holder only if its
weighted headline is higher by more than noise and it is not worse by more than noise on any of
the four suites. Noise is the larger of one composite point and two standard errors of the paired,
map-by-map difference.

At 6:42 AM the holder was `main_0525_M` (the main run at 172.0 M steps of phase two, shipped at 5:37 AM). On three engine seeds on the two generated suites and five on the two authored ones, the weighted headline of `blend_final_M` is 79.61 ± 1.07 against 78.91 ± 1.06, a paired difference of +0.70 ± 0.74 with a noise margin of 1.49: level, so by the rule alone the holder would have stayed. No suite is worse by more than noise; better by more than noise: `heldout_v1` (+1.69 ± 0.75). On the organizer-style suite, which is not part of the rule, it scores 74.98 against 68.35: +6.64 ± 1.30, better by more than noise (normal half 88.17 against 84.42, +3.75 ± 1.26, better by more than noise; dense half 61.80 against 52.27, +9.52 ± 2.25, better by more than noise). The team lead decided to ship `blend_final_M`: it is not worse by more than noise on any suite and it is clearly better on the maps most like the organizers' own. That is a decision outside the written rule, not a win under it. On the first engine seed alone the headline difference had been +1.86 ± 0.91, just beyond its margin of 1.82; the two further seeds on the generated suites gave +0.12 ± 0.80 on their own. The other candidates of that comparison, against the same holder: the main run alone at 196.6 M steps, headline 78.98 (level) and 71.10 on the organizer-style suite; the branch alone at 164.7 M steps, 78.34 (level, and worse by more than noise on `authored_hard_v1`, −5.71 ± 2.01) and 73.18; the average of the main run's last nine checkpoints, 79.02 on one seed (level). The same averaged weights with the default decoding (USE and weapon only) were level with the shipped setting on one seed (shipped minus default +0.82 ± 0.82 on the headline). Reports: `training/logs/final_pick_0618.txt`, `final_pick_0630_with_blend.txt`, `final_pick_three_seeds_shipped_main_clone.txt` and `final_pick_three_seeds_shipped_blend_main.txt`.

### What the training log shows

`training/logs/unified_full_2_train_log.csv` is the log of the second phase (one row per iteration of
98,304 environment steps; this copy ends at iteration 899, 88.4 M steps). Every 150 iterations the
policy alone, with no teacher in the loop, plays 384 episodes on the development maps; the teacher's
weight steps down only when that exit rate holds.

| Steps into phase two | Teacher weight from here on | Held-out exit rate of the policy alone |
|---|---|---|
| 14.7 M | 0.1 | 0.67 |
| 29.5 M | 0.03 | 0.72 |
| 44.2 M | **0** (and no teacher action is executed any more) | 0.77 |
| 59.0 M | 0 | 0.83 |
| 73.7 M | 0 | 0.85 |

Phase one had stepped the weight from 1 to 0.1 in the same way, at held-out exit rates of 0.45, 0.50
and 0.63 (the gate history stored in its last checkpoint; `training/configs/unified_full_2.txt`
quotes the last step). The second phase started again at 0.25 on a new map set.

The run went on past the end of this log copy. Its later gate reports, read from the history stored
in the run's checkpoints (not in the repository), stay between 0.83 and 0.90: 0.83 at 88.5 M steps,
0.84 at 103.2 M, 0.88 at 118.0 M, 0.86 at 132.7 M, 0.87 at 147.5 M, 0.89 at 162.2 M, 0.90 at
176.9 M and 0.88 at 191.7 M. The branch that left the run at 125.3 M steps (`clone_mix_full`,
see [Reproducing the training](#reproducing-the-training)) reports 0.87, 0.86 and 0.89 at 132.7 M,
147.5 M and 162.2 M of its own counter, read from its checkpoint at 164.7 M in the same way.

### Component measurements

| What | Result | Where |
|---|---|---|
| Pose estimate in scored play | mean final position error 0.32 map units over the 80 `dev_v1` episodes of the 3:08 AM agent and 0.20 over the 60 `dev_wide_v1` episodes (one grid cell is 64 units); heading error zero | columns `odom_err_final` and `odom_heading_err_max` of `training/logs/results.csv`, tags `main_0305_dev` and `main_0305_wide`; regression tests in `training/tests/test_memory.py` |
| Exit detector | precision 0.998 to 0.999; recall 0.94 within 400 units and 0.72 to 0.79 at 400 to 640 units, on held-out maps and on maps whose textures were never seen in training (20,480 frames each); 0.91 within 400 units in frames from real play with fights | `training/logs/exit_detector.csv` |
| Navigation teacher (training only) | reaches the exit on 1.00 / 0.98 / 0.99 of the tier 2 / 3 / 4 held-out layouts with the monsters removed (100 each, true tick cap as its budget) | `training/logs/teacher_gate.csv` |
| Time per decision | 2.4 ms median in the Docker image on one CPU; see [Hardware and latency](#hardware-and-latency) | `training/logs/fresh_clone_validation.txt`, `training/logs/latency.txt` |

Two short learning checks of the training loop are logged with their conditions: PPO from random
initialisation on single-room arenas (`training/logs/smoke_arena.txt`) and the unified loop on
monster-free maps (`training/logs/unified_smoke.txt`).

### Known limits

- **The shipped weights are an average that was never trained as one network.** They are the mean of two branches' weights: no gradient step was taken at these values, and the mean of two networks that play well need not play well. The two branches share their training up to 125.3 M steps of phase two and differ by the 71.3 M and 39.3 M steps after it; we did not test averages of networks further apart. Measured on the averaged weights, through the real agent: 3,115 held-out episodes with the shipped decoding (the tables above), the 80 development episodes, the nine-check ship gate and the manager's entropies in the next item. Not measured: any other weights than 0.5 and 0.5, a failure breakdown of its episodes (the one below is of the 1:38 AM agent), and the official maps and scorer.
- **The manager's choices are close to random in the shipped weights.** The manager and the
  value-propagation block were trained with the teacher's labels. After the teacher's weight
  reached zero (iteration 450 of phase two), the entropy of the sub-goal head rose from 0.68 to
  2.03 nats (a uniform choice over 8 sub-goals has 2.08), the entropy of the target pointer from
  1.56 to about 7.7 (uniform over 4,096 cells: 8.32), and agreement with the teacher's sub-goal and
  target fell from 0.84 and 0.63 to 0.15 and 0.03 (columns `ent_subgoal`, `ent_target`,
  `agree_subgoal`, `agree_target` of the training log, iterations 450 to 899). Over the same steps
  the held-out exit rate rose from 0.77 to 0.85. On the weights promoted at 3:08 AM the two
  entropies are 2.01 and 7.44 on recorded inputs, and on two later checkpoints of the run (120.4 M
  and 149.9 M steps) 2.00 and 7.32, 2.02 and 7.06 (`training/logs/manager_probe.txt`; a small probe, 10
  manager steps of two recorded episodes). The shipped policy file is not one of the checkpoints in
  that log; the same measurement on it gives 2.04 and 6.74 nats, the
  second being the entropy of an even choice among about 850 of the 4,096 cells
  (`python docs/brief/fill_final.py --manager-probe` computes both from the file). So the
  hierarchy is in the architecture and was trained, but in the shipped weights the sub-goal is a
  near-uniform draw and the target is spread over hundreds of cells, and the controller acts from
  its own sensors, its recurrent state and the patch of the value field it reads. The probe log
  also shows, for the checkpoints it lists, that the controller still responds to the sub-goal
  embedding it is given (swapping it moves the move and strafe probabilities by about 0.10 in total
  variation; swapping the target by 0.03 to 0.04). We have not measured what the value field still
  contributes, and we do not claim that the shipped agent plans through sub-goals.
- **Dense organizer-style maps.** The dense half of the organizer-style suite is the hardest set
  we score: the table under [Held-out suites](#held-out-suites) has the shipped agent's composite
  on it beside the normal half.
- **Heavy monsters and chaingunners.** On the four held-out suites the agent of 1:38 AM had a
  death rate of 0.41 on maps that hold a hell knight or a cacodemon, against 0.05 on the other
  maps. Among those other maps the death rate was 0.005 without a chaingunner, 0.15 with one and
  0.50 with two or more (`training/logs/failure_breakdown_main_0135.txt`; 1,195 episodes; not
  repeated for later agents).
- **Timeouts are search failures.** In the same episodes, of the 127 timeouts on maps without a heavy
  monster, 51 never had the exit in view after the gates were done, 43 never picked up a key and 27
  held the key and never opened the locked door; 6 saw the exit and did not reach it. None of them
  was standing still (stuck rate above 0.1 in no episode).
- **Ammunition.** About 9 shots per kill in tier 4 and 5.5 in tier 1 (agent of 12:38 AM, same file);
  10 of 70 tier 4 episodes ran out of bullets.
- **Exit detector.** It is trained on the red exit pad of the grid layouts: on a pad drawn with
  another floor texture recall within 400 units is 0.00 to 0.14 (`padflats` rows of
  `training/logs/exit_detector.csv`) and the agent has to walk onto the pad. Recall within 400
  units falls to 0.81 to 0.89 with a monster in view.
- **Other.** A switch-style exit cannot be pressed under the USE mask. The time-budget input is the
  constant 525 decisions at evaluation, the shortest tick cap. The category weights of the local
  scorer are a guess at the official ones.

## Quickstart

Set up the environment (Python 3.13, [uv](https://docs.astral.sh/uv/)):

```bash
git lfs install && git lfs pull   # the weights are Git LFS objects
uv sync                           # installs vizdoom, gymnasium, torch and the official wrapper (env/) in editable mode
uv run python validate.py         # the official validator: interface, a full episode, latency, memory
```

Or use the official image, which is what the scorer runs:

```bash
docker build -t devforge-doom .                      # or: docker pull <official image tag>
python validate.py --docker --image devforge-doom    # same checks, sandboxed like the scorer
```

**Platforms:** Linux (x86-64/ARM), macOS on Apple Silicon (macOS 15+) and Windows (x86-64) all
work with `uv sync` and `validate.py`. Intel Macs have no `vizdoom` 1.3.3 wheel: use Docker or a
Linux machine. Evaluation runs on Linux, so `validate.py --docker` is the closest check.

### Run the agent

Every command below is run from the repository root inside the project environment
(`uv run python ...`, or `.venv/bin/python ...`).

```bash
# one episode with a video and contact sheets of frame, memory map, sub-goal and target (CPU, no window)
python -m training.demo --policy training.runner:AgentPolicy --map maps/two_rooms.wad --out runs/demo/agent

# the same on a layout drawn as text: edit the grid, run again
python -m training.demo --policy training.runner:AgentPolicy --grid my_layout.txt --out runs/demo/edit

# score the agent on the frozen held-out suite under the official time limits, then print the table
python -m training.evaluate --policy training.runner:AgentPolicy --suite training/suites/heldout_v1.jsonl --tag NAME
python -m training.evaluate --table

# time per decision on one thread, with a per-stage breakdown
python training/bench/agent_latency.py

# our tests (rule lint, agent, fuzz, doors, memory, networks, PPO, teacher, map generator, scorer)
python -m pytest training/tests -q -m "not slow"
```

The agent refuses to start when a weights file is missing or is a Git LFS pointer. Tests and
benchmarks that only exercise the code set `DOOMNET_ALLOW_UNTRAINED=1`, which builds a missing
network with its initial weights; the scorer does not set it.

In Python:

```python
from doom_env import make_env, observation_space, action_space
from submission.agent import DoomAgent

env = make_env("maps/two_rooms.wad")
agent = DoomAgent(observation_space, action_space, "cpu")
obs, info = env.reset(seed=0)
agent.reset()
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(agent.act(obs))
    done = terminated or truncated
```

## Repository layout

```
submission/agent.py        DoomAgent, the evaluation entry point: one fixed pipeline per decision
submission/doomnet/        spec.py      frames, input layouts, map channels, action heads, the two action masks
                           odometry.py  pose from the agent's own buttons, velocity and rays
                           memory.py    the spatial memory map and the crop the networks read
                           encode.py    observation -> fixed-size network inputs
                           model.py     PolicyNet (manager, value propagation, controller) and ExitDetector
submission/weights/        policy.pt, exit_detector.pt (Git LFS)
submission/README.md       what is learned and what is hand-written, file by file, and how it is checked
training/mapgen/           procedural generator of grid layouts, randomisation, solvability check, suites
training/suites/           frozen held-out and development suites, stored as grids
training/teacher/          scripted navigation teacher (training-time scaffolding, uses search)
training/exitdet/          frame collection, training and evaluation of the exit detector
training/rl/               recurrent PPO: workers, buffer, loss, reward, curiosity, curriculum, export;
                           average.py (weight average of checkpoints), mix_manifests.py (map set of the branch)
training/configs/          configs of the combat run, the unified runs and the branches, with notes and the launch steps
training/scorer.py, runner.py, evaluate.py, mapmeta.py     local scorer, episode runner, suite evaluation
training/promote.py, ops/  score a checkpoint as a candidate agent, ship gate, remote evaluation runs
training/final_pick.py     the report for the final choice: four held-out suites, paired verdicts, decision rule
training/demo.py           headless viewer: video and contact sheets
training/bench/            latency benchmarks of the networks and of the whole agent
training/tests/            tests of all of the above; test_rules.py is the rule-compliance lint
training/logs/             curated logs of the runs quoted in the documents (results.csv: every scored checkpoint)
docs/                      contract, rules, submission guide, evaluation details (template), ARCHITECTURE.md
docs/brief/                brief.md and brief.pdf, build_brief.py (renders it, checks the 2 pages), fill_final.py
                           (writes the final agent's facts into the brief and this file; final_values.json holds them)
env/, maps/, validate.py, tests/, docker/, Dockerfile       the official template, unchanged
```

Generated maps, datasets and run directories go to `data/` and `runs/`, which are not tracked.

## Reproducing the training

The stages, their files and their logs are described in [training/README.md](training/README.md);
the settings of each run are explained next to its config in `training/configs/`. The shipped policy
comes out of three runs in a row, combat (118.8 M steps), phase one of the unified loop (49.0 M) and
phase two, and of a branch of phase two; which checkpoints are in it is stated under Results and
again below the commands. In short:

```bash
# 1. Maps. train: 300 per tier 0-4 (combat run and phase one). train_v2: 600 per tier with 90% of the
#    starts on the teacher's grid. train_v3: train_v2 plus 1,500 maps of the wide profile (phase two).
#    And the frozen development suite rebuilt as WADs for the held-out check inside training.
python -m training.mapgen.cli --out data/maps/train --n 300 --tiers 0,1,2,3,4 --seed 0 --randomize 0.5
python -m training.mapgen.cli --out data/maps/train_v2 --n 600 --tiers 0,1,2,3,4 --seed 6000000 --randomize 0.5 \
    --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
python -m training.mapgen.cli --out data/maps/train_v3 --extend data/maps/train_v2/manifest.jsonl --profile wide \
    --n 150,350,500,500 --tiers 1,2,3,4 --seed 7000000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
python -m training.mapgen.cli --from-suite training/suites/dev_v1.jsonl --out data/maps/dev

# 1b. Organizer-style maps (profiles reference and reference_wide: zombieman, imp, chaingunner, stimpacks, an exit
#     on every map), and the map set of the branch (step 4d): all 3,000 of them plus 3,000 maps of train_v3, as many
#     per tier (450, 750, 750, 1,050 in tiers 1 to 4; none of tier 0)
python -m training.mapgen.cli --out data/maps/ref_part --profile reference --n 225,375,375,525 --tiers 1,2,3,4 \
    --seed 7200000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
python -m training.mapgen.cli --out data/maps/train_ref_v1 --extend data/maps/ref_part/manifest.jsonl --profile reference_wide \
    --n 225,375,375,525 --tiers 1,2,3,4 --seed 7300000 --randomize 0.5 --start-quarter 0.5 --start-any 0.05 --start-offset 0.05
python -m training.rl.mix_manifests data/maps/train_mix_ref50/manifest.jsonl data/maps/train_ref_v1/manifest.jsonl \
    data/maps/train_v3/manifest.jsonl --seed 4321

# 2. Navigation teacher: check it on monster-free held-out layouts (it is used inside steps 4b and 4c)
python -m training.teacher.gate --tag heldout_nav_clean --suite training/suites/heldout_nav_v1.jsonl

# 3. Exit detector: collect labelled frames, train, evaluate. The "play" sets are frames from whole
#    episodes, part of them picked with an earlier detector (--miner); see training/README.md
python -m training.exitdet.collect --sets train,extra,val,calib,heldout,unseen,weapons --workers 3
python -m training.exitdet.collect --sets play,play2,play_calib,play_heldout --miner EARLIER_WEIGHTS.pt --workers 3
python -m training.exitdet.train --tag NAME --save      # trains on train,extra,play,play2; --save publishes weights and tables
python -m training.exitdet.evaluate --weights submission/weights/exit_detector.pt

# 4a. Combat by PPO alone, from random initialisation, on the tier 1 arenas
python -u -m training.rl.train --maps data/maps/train/manifest.jsonl --config training/configs/combat_1.json \
    --workers 96 --envs-per-worker 12 --out runs/combat_1

# 4b. Phase one of the unified run, continued from the combat checkpoint: PPO with curiosity and
#     curriculum on all tiers, plus distillation from the teacher with a weight gated on held-out exit rate
python -u -m training.rl.train --maps data/maps/train/manifest.jsonl --config training/configs/unified_full.json \
    --init runs/combat_1/latest.pt --heldout data/maps/dev/manifest.jsonl \
    --workers 128 --envs-per-worker 12 --out runs/unified_full_1

# 4c. Phase two, continued from phase one on the larger map set: the teacher's weight goes 0.25, 0.1,
#     0.03, 0 as the held-out exit rate rises; after that the run is PPO alone
python -u -m training.rl.train --maps data/maps/train_v3/manifest.jsonl --config training/configs/unified_full_2.json \
    --init runs/unified_full_1/latest.pt --heldout data/maps/dev/manifest.jsonl \
    --workers 128 --envs-per-worker 12 --out runs/unified_full_2

# 4d. A branch of phase two: the main run's checkpoint at iteration 1,275 (125.3 M steps), continued with the
#     same settings on the mixed map set of step 1b. The teacher's weight is zero from the start of the branch,
#     so every step of it is PPO alone. Its step counter goes on from the main run's
mkdir -p runs/clone_mix_full && cp runs/unified_full_2/ckpt_001275.pt runs/clone_mix_full/latest.pt
python -u -m training.rl.train --maps data/maps/train_mix_ref50/manifest.jsonl --config training/configs/clone_mix_full.json \
    --resume runs/clone_mix_full/latest.pt --heldout data/maps/dev/manifest.jsonl --out runs/clone_mix_full

# 4e. Weight average of one checkpoint of each branch (float64 sum of every tensor, stored as float32; iteration
#     and step counter are the last file's), exported as the agent's weights file with the decoding setting in
#     its config: move, strafe, turn, USE and weapon take their most likely option. The files and weights written
#     here are those of the agent shipped at 6:42 AM
python -m training.rl.average runs/blend/blend.pt runs/clone_mix_full/ckpt_001675.pt runs/unified_full_2/ckpt_002000.pt \
    --weights 0.5,0.5 --policy-file runs/blend/policy.pt --greedy-heads move,strafe,turn,use,weapon

# 5. A checkpoint (or an exported weights file such as runs/blend/policy.pt) becomes a candidate agent, is scored
#    through the real agent and compared with the shipped one; the ship gate must pass before its policy file is
#    copied into submission/weights/
python -m training.promote score runs/unified_full_2/latest.pt --tag NEW_dev --suite dev
python -m training.promote score runs/unified_full_2/latest.pt --tag NEW_wide --suite training/suites/dev_wide_v1.jsonl
python -m training.promote compare OLD_dev NEW_dev      # exit status 0 only if NEW replaces OLD: better by more than
                                                        # noise, or a tie within one point with more RL behind NEW
python -m training.promote check runs/promote/NEW_dev   # ship gate: 9 checks

# 6. The final choice between finished agents, on the four held-out suites (one engine seed on the two generated
#    suites, five on the authored ones), and the organizer-style suite beside them
python -m training.final_pick score runs/promote/OLD_dev --tag OLD
python -m training.final_pick score runs/promote/NEW_dev --tag NEW
python -m training.final_pick report OLD NEW
python -m training.promote score --candidate runs/promote/NEW_dev --tag NEW_ref --suite training/suites/heldout_ref_v1.jsonl \
    --seeds 0,1,2
```

The shipped weights are the output of step 4e with these arguments: `--weights 0.5,0.5` for `runs/clone_mix_full/ckpt_001675.pt runs/unified_full_2/ckpt_002000.pt`, that is iteration 1,675 of the branch (164.7 M steps) and iteration 2,000 of the main run (196.6 M), the branch named first, so that the file carries the main run's counters. `training.rl.average` gives the shipped tensors exactly: its average of these two checkpoints equals the state dict of `submission/weights/policy.pt` in all 73 tensors, bit for bit (compared by `docs/brief/fill_final.py --draft` when this text was written). The file's sha256 is not reproduced, because the file also stores the name and the hash of the averaged checkpoint and a provenance record with the time it was written.

What differed in the runs behind the shipped weights: phase two played its first 5.1 M steps
on `train_v2` and was then resumed (`--resume`) on `train_v3`; the runs were stopped and restarted a
few times on rented machines, each time from their own last checkpoint. The branch ran on a second
machine with 64 workers of 24 environments (the same 1,536 environments) and was restarted several
times in its first 2.5 M steps while that layout was being chosen. Its map set was written on that
machine by a scratch script with the logic of `training/rl/mix_manifests.py` and the same seed;
compared with a copy of that file, the tracked script's output holds the same 6,000 maps, the
3,000 of `train_v3` in another order within each tier. In the last comparison the two further engine
seeds on the generated suites, the organizer-style table by half and the storing of the decoding
setting were run by scripts outside the repository that call `training.promote.score_local`,
`training.final_pick` and the agent's own file layout; their output is in the logs named above,
and `training.rl.average --greedy-heads` is the tracked form of the last of the three.

`--workers` and `--envs-per-worker` are sized for a Linux machine with about 128 cores and a GPU
for the learner; the same commands run on a laptop with smaller numbers (the learner uses CUDA,
else MPS, else the CPU; rollout workers always run on the CPU). A 25-minute laptop version of
step 4b is `training/configs/unified_smoke.json`. The full launch procedure, with the lines a
healthy run prints and the log columns to watch, is `training/configs/LAUNCH.txt`.

## The three deliverable areas

### Sensory and state representation

*"Your encoders and preprocessing on top of the official observation dictionary."*

The agent uses the rays, the entity list, the vitals and the RGB frame; it does not use the depth
buffer. `submission/doomnet/encode.py` rescales one observation into fixed-size arrays: 64 rays × 7
features (distance, hit type one-hot), up to 32 entities × 17 features (distance, sine and cosine
of the bearing, relative velocity, type one-hot) with a padding mask, and 44 scalars (vitals,
selected weapon, ammunition, keys, velocity, heading, step and assumed time budget, previous
buttons, steps since USE was pressed, the exit sighting). The observation has no map coordinates,
so `odometry.py` integrates the agent's pose from its own buttons, the velocity in `vitals` and the
rays. The exit pad is not part of the rays or the entity list; `model.ExitDetector`, a small CNN,
finds it in the RGB frame. All layouts and constants are in `spec.py`.

### Model architecture and memory

*"Your custom neural network topology, specifically addressing how the agent maintains spatial
memory across rooms and handles variable numbers of enemies."*

`submission/doomnet/model.py` holds `PolicyNet`, three learned levels. A **manager** reads the
memory map through a convolutional tower and, every 8 decisions, chooses one of 8 sub-goals and
points at a target cell of the map. A **value-propagation block**, one learned Bellman backup
applied 12 times, turns that target into a value field over the map. A **controller** (circular
convolution over the rays, attention over the entities, a GRU) outputs six action heads. The
controller receives the manager's decision only as a sub-goal embedding and a 9×9 patch of the
value field around the agent; no code acts on the sub-goal or the target.

*What the shipped weights do with it.* The three levels were trained together, the upper two with
the teacher's labels as long as its weight was above zero. Since then the manager's two heads have
drifted to near-uniform outputs ([Known limits](#known-limits)), so in the shipped weights the
sub-goal and the target are close to random draws and the buttons come from the controller's own
sensors, its recurrent state and the value-field patch.

*Spatial memory across rooms.* `memory.py` keeps a 256×256 grid of 32-unit cells in the frame of
the agent's starting pose for the whole map, with 11 channels: free, wall, door, locked door,
visited, recency, enemy, item, key, exit, USE pressed. A locked door seen before its key, a key
seen behind a door and the exit pad stay marked after the agent has left the room. The networks
read a 64×64 crop at 64 units per cell, centred on the agent and not rotated with it. The write
path is hand-written path integration and ray carving: it is not learned and not differentiable.
The learned parts are the exit detector, which writes the exit channel, and the tower, manager and
value propagation, which read the map; the GRU holds what the map does not.

*Variable numbers of enemies.* The entity list is treated as a set of 0 to 32 elements: every
entity is embedded by the same layer, four attention heads whose queries are computed from the
agent's own scalars (health, ammunition, weapon) pool them, a constant "nothing" slot takes the
weight when the view is empty, and padding rows are masked out. Enemy sightings are also written
into the memory map (a channel that fades over 64 decisions), and enemies appear in the rays.

### Training and exploration pipeline

*"Your reinforcement learning implementation, custom reward shaping, and exploration mechanisms
that drive the agent to navigate unfamiliar rooms rather than getting stuck in corners."*

- **Reinforcement learning** (`training/rl/`, plain torch, no RL library): worker processes that
  run the engine and the agent's own front end, a rollout buffer with GAE on two reward streams,
  truncated backpropagation through time, a clipped surrogate loss over the joint probability of
  the six action heads and the manager's two choices, two value heads, per-head entropy bonuses.
- **Reward shaping** (`training/rl/reward.py`): the scoring categories paid out as their events
  happen (exit, health and time left at the exit, keys, doors, locked doors, sectors entered,
  kills), minus damage, death, shots that hit nothing and door reversals. It is computed in the
  workers from privileged state, which the agent never sees. Our configs add a small bonus per
  shot that hits a monster, which is not part of the score.
- **Exploration** (`training/rl/intrinsic.py`): a curiosity reward for every cell newly observed in
  the agent's own memory map, plus a bonus of 1/√n on the n-th entry into a cell. Standing still
  earns nothing and pacing between two cells decays. It has its own value head and advantage
  stream and reads no privileged state. The map itself carries a visited channel and a fading
  recency trail, and the local scorer reports a stuck rate (`training/scorer.py`).
- **Curriculum** (`training/rl/curriculum.py`): stages over map tiers (navigation, doors, keys,
  gauntlet, with arenas kept in every stage), then sampling weighted towards the tier with the
  lowest recent exit rate.
- **Teacher** (`training/teacher/`): navigation is started by distillation from a scripted
  explorer, described under rule compliance below; its weight reached zero 44.2 M steps into phase
  two and the run, and the branch that later left it, have been PPO alone since. Combat is learned
  by PPO alone.
- **Maps** (`training/mapgen/`): a procedural generator of grid layouts with randomised textures,
  light levels, start heading and offset, and monster facing and ambush flags, in a base and a
  wide profile; 4,500 maps in the main run of phase two. Two more profiles keep to the monsters and
  pickups of the organizers' maps; the branch of phase two trained on 3,000 of those and 3,000 of
  the 4,500. Checkpoints are scored on frozen layouts from seeds never trained on.
- **Evaluation** (`training/evaluate.py`, `promote.py`, `final_pick.py`): during the run a checkpoint
  replaced the shipped one only if it won the paired comparison on a development suite by more
  than noise (one composite point and two standard errors of the map-by-map difference), or tied
  it within one point with more reinforcement learning behind it, and passed a ship gate of nine
  checks: size, provenance of the weights, the official validator, four test files (rule lint,
  fuzz, agent, doors), latency, and a tree identical to the one that was scored. The last choice
  was made on the held-out suites; its rule and what was decided are under
  [How the shipped agent was chosen](#how-the-shipped-agent-was-chosen).
- **Weight averaging** (`training/rl/average.py`): the weighted mean of the tensors of checkpoints
  of one architecture, with a fresh provenance record. Whether the shipped weights are such an
  average is stated under [Held-out suites](#held-out-suites).

## Rule compliance

The rules are in [docs/RULES.md](docs/RULES.md). How this repository meets them:

- **Decisions come from trained parameters.** Every button is the output of a head of `PolicyNet`.
  The only hand-written code that touches action choice is two masks computed from observation
  fields (`spec.action_masks`): USE can be chosen only when the ray straight ahead ends on a door
  or locked door within USE reach, and ATTACK only when an enemy is visible and the selected
  weapon is not the melee slot. A mask removes an option that would do nothing; the network
  chooses among the rest. The same function is applied to the logits in training and in the agent.
- **No pathfinding, no rule trees, no aimbot in the agent.** `submission/` contains no search over
  the map, no inverse trigonometry, no scripted behaviour and no access to the engine or to
  privileged state. `training/tests/test_rules.py` reads every file there and fails on banned
  techniques, imports other than numpy, torch and the standard library, GPU use, and the code
  shapes a hand-written search takes (`while` loops, `break`, recursion).
  `training/tests/test_fuzz.py` runs the agent behind the scorer's protocol on malformed input.
  Both run on every push (`.github/workflows/ci.yml`).
- **What is hand-written in the agent, stated plainly.** Odometry, the memory map's write path,
  the encoder and the two masks are hand-written arithmetic on observation fields and on the
  agent's own past actions. `agent.py` also has a time governor, a compute schedule driven only
  by the clock, and an exception handler so that `act()` never raises. None of these reads an
  observation to choose a button. [submission/README.md](submission/README.md) labels every module.
- **Training-time scaffolding, disclosed.** The navigation teacher (`training/teacher/`) is a
  scripted explorer that uses shortest-path search. It exists only in training, is never imported
  by the agent, and decides only from what the student network receives at the same step (memory
  crop, pose estimate, observation, encoder scalars, previous buttons). It never labels the attack
  or weapon heads and gives no button labels while an enemy is visible. Its weight in the loss is
  stepped down only while the held-out exit rate of the policy alone holds; it has been zero since
  44.2 M steps into phase two, and from that step on no teacher action is executed and no label
  enters the loss (`training/logs/unified_full_2_train_log.csv`, columns `distill_weight` and
  `beta`). The branch of phase two started after that step and never had a teacher. The exit detector's
  frame collector (`training/exitdet/collect.py`) uses scripted wandering and shooting to produce
  frames; it produces no policy labels. The map generator's solvability check and reference walker
  use search and true positions and are used on maps only.
- **No off-the-shelf RL.** The PPO implementation is ours; no RL library is imported anywhere.
- **Evaluation conditions.** The agent uses only the observation it is given, needs no network and
  writes nothing. `env/` and `validate.py` are the template's files, unchanged.

## Hardware and latency

Evaluation gives each agent 2 GB of VRAM, 8 GB of RAM, one CPU thread and about 40 ms per decision
on average ([docs/EVALUATION.md](docs/EVALUATION.md#hardware)). The agent runs on the CPU with one
thread whatever `device` it is given and never initialises CUDA, so it uses no GPU memory.

| Measurement | Result | Conditions and log |
|---|---|---|
| Full `act()` in the official validator, Docker image, one CPU | median 2.4 ms, p99 6.8 ms, max 7 ms (first call 7 ms) | one episode of 45 decisions on `maps/two_rooms.wad`; fresh clone of commit 49633a4, the commit that shipped the blended weights, 27 Nov 6:45 AM (the same check on commit 0bf38b2 at 4:38 AM, with the 3:08 AM weights: 2.4 / 6.5 / 7 ms); `python validate.py --docker --no-gpu` with the template's image built locally (linux/arm64), agent pinned to one CPU; `training/logs/fresh_clone_validation.txt` |
| The same clone, natively on the Mac | median 1.1 ms | `python validate.py`; same file |
| Full `act()`, benchmark on the Mac, one thread | mean 1.05 ms, p95 1.99 ms, max 5.8 ms | `training/bench/agent_latency.py`: 3 maps × 700 decisions, best of 3 repeats; `training/logs/latency.txt`, section 1 |
| Full `act()`, GitHub Actions runner, x86-64 | mean 3.98 ms, p95 7.27 ms, max 10.6 ms | the same benchmark in the CI job; `training/logs/latency.txt`, section 2 |
| Mean `act()` time while scoring | 1.5 to 2.2 ms per scoring run, at most 14 ms for a single call | columns `act_ms_mean` and `act_ms_max` of `training/logs/results.csv`, the runs tagged `main_*` (22 by 5:26 AM), scored on the Mac |
| Building the agent | 1.7 s in the image | limit 60 s; all warm-up happens here |
| Peak RAM in `validate.py --docker` | 798 MB (794 MB at 4:38 AM) | limit 8,192 MB |

The two benchmark rows were measured on 26 Nov, before trained weights existed, with the policy
network at its initial weights; the architecture has not changed since, and the cost of a pass does
not depend on the values of the weights. Per decision on the Mac, by stage (`latency.txt`): exit
detector 0.37 ms, policy network 0.38 ms (a decision with a manager pass, one in 8, costs 2.0 ms
in all; the others 0.9 ms), memory update and crop 0.18 ms, odometry 0.06 ms, encoder 0.04 ms,
decoding 0.01 ms.

Training throughput. On the training machine (128 worker processes × 12 environments, learner on
one GPU): 8,000 to 12,000 environment steps per second in phase two, about 8,000 while a second run
shared the machine and about 12,000 alone (column `env_steps_per_s` of
`training/logs/unified_full_2_train_log.csv`: median 7,975 up to iteration 760 and 11,762 after).
On a 15-core laptop under load: about 1,100 steps per second for the whole PPO loop with 3 workers
and the learner on the laptop GPU (`training/rl/README.md`, "Throughput").

## Template documentation

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules: what's allowed, what gets you disqualified |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts, replays |
