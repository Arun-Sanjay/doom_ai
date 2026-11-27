# DoomNet: a neural DOOM agent with a self-built spatial memory

Yugaantar 2026, DevForge Track 1 "Can It Play DOOM?" · project brief · team Arun-Sanjay (solo entry) · https://github.com/Arun-Sanjay/doom_ai

## USP

- **Every action is a network output, and the training is reinforcement learning.** Recurrent PPO written in plain torch (`training/rl/`, no RL library). Combat is learned from random weights by PPO alone (118.8 M steps). Navigation is started with a scripted teacher that exists only in training and sees only what the network sees. Its loss weight is stepped down as the policy's own held-out exit rate rises and reached **zero** 44.2 M steps into the second phase; the shipped checkpoint has had 127.8 M further steps of reinforcement learning alone.
- **A spatial memory built from the agent's own actions.** The wrapper gives no coordinates. The agent integrates its pose from its buttons, velocity and rays, writes what it sees into an 11-channel grid in the frame of its starting pose, and keeps it for the whole map. The exit pad is absent from rays and entities, so a small CNN finds it in the RGB frame and writes it into the grid.
- **A custom network, small and CPU-only.** A 1,307,153-parameter policy (circular ray convolution, entity attention, a manager, a value-propagation block, a GRU controller with six factored heads) and a 416,867-parameter exit detector. `submission/` has no rule tree, no path search and no aiming maths. One decision takes 2.4 ms (median) on one CPU inside the template's Docker image, against a budget of 40 ms.
- **Measured, with the limits written down.** Frozen held-out suites, promotion only on a paired win beyond noise and a nine-check ship gate, validation from a fresh clone. The last section states the limits, including a manager whose choices have become close to uniform.

## Neural architecture

```
 OBSERVATION (doom_env)         HAND-WRITTEN ENCODING (not learned)       LEARNED (1.72 M parameters)

 buttons pressed, velocity,     ┌──────────────────────────────┐
 64 rays ────────────────────>  │ ODOMETRY   pose (x, y, angle)│
                                └───────────────┬──────────────┘
                                                │ pose
                                ┌───────────────┴──────────────┐   exit     ┌───────────────────────┐
 rays, entities ─────────────>  │ MEMORY MAP   256×256 cells   │ <────────  │ EXIT DETECTOR   CNN   │ <── RGB 120×160
                                │ of 32 units, 11 channels     │   sighting └───────────────────────┘
                                └───────────────┬──────────────┘
                                  crop 11×64×64 │ 64 units per cell, agent at the centre, not rotated
                                                v
                          ┌────────────────────────────────────────────────────────────────────────┐
 every 8 decisions        │ MAP TOWER ──> MANAGER   sub-goal (1 of 8) and target cell (1 of 4,096) │
                          │           ──> VALUE PROPAGATION   one conv cell × 12 ──> value field   │
                          └────────────────────────────────────┬───────────────────────────────────┘
                                sub-goal embedding + 9×9 patch │ of the value field, in the agent's frame
 rays 64×7 ──── circular conv ──┐                              v
 entities 32×17 ─── attention ──┼──> fusion ──> GRU 256 ──> CONTROLLER  move · strafe · turn · attack · use · weapon
 scalars 44 ─── MLP ────────────┘    every decision            ^ two masks from observation fields: USE, ATTACK
```

**Sensors** (`encode.py`, `spec.py`). Rays become 64×7 (distance, hit type one-hot), entities up to 32×17 (distance, sine and cosine of the bearing, relative velocity, type one-hot) with a padding mask, and 44 scalars carry vitals, weapon, ammunition, keys, velocity, heading, time budget, previous buttons and the exit sighting. **Controller.** A circular convolution over the rays (ray 63 is next to ray 0). The entities are a set: four attention heads with queries computed from the agent's own scalars, a constant "nothing" slot for an empty view and a set maximum, so 0 to 32 enemies, projectiles and pickups are handled in any order. A GRU (256) feeds six categorical heads (3, 3, 3, 2, 2, 2 options) and two value heads. **Manager.** A three-scale convolutional tower over the map crop, a categorical over 8 sub-goals, then a pointer: one softmax over the 4,096 crop cells. **Value propagation.** V ← max over moves a of (r_a + γ_a · P_a ∗ V) on a 32×32 grid with 8 value channels; the rewards r_a, the 3×3 transition kernels P_a (softmax) and the discounts γ_a (sigmoid) are all learned, so 12 applications are a contraction (14,404 parameters). The controller receives the levels above it only as a sub-goal embedding and a 9×9 patch of the value field; no code acts on the sub-goal or the target. **Exit detector.** Five convolutions and two linear layers read image rows 48 to 119 and output a visibility logit and the pad position.

## Spatial memory design

- **Pose** (`odometry.py`). Heading is exact: turn rates are whole multiples of 1/1024 of a circle, so it is kept as an integer. Position follows the engine's thrust and friction model; a step disturbed by a wall or a monster is measured by matching the wall points of two consecutive ray scans (closed-form least squares, no search). In scored play of the 3:08 AM agent the mean final position error is 0.32 map units over 80 episodes (a grid cell is 64) and the heading error is zero (`training/logs/results.csv`, columns `odom_err_*`).
- **Map** (`memory.py`). 256×256 cells of 32 units centred on the start pose, cleared in `reset()`. Channels: free, wall, door, locked door, visited, recency, enemy, item, key, exit, USE pressed. Free and wall are vote counts (a ray passing through removes 1, a wall hit adds 3), so one misplaced hit is outvoted. Recency, enemy and USE decay from time stamps.
- **Across rooms.** Walls, doors, keys, items and the exit stay in the map for the whole episode, so a locked door seen before its key is still marked when the key is found. The network reads a 64×64 crop at 64 units per cell (4,096 units across), centred on the agent and aligned with the start frame, so cells keep their place when the agent turns.
- **What is learned.** The write path is hand-written path integration and ray carving: not learned, not differentiable. Learned: the exit detector that writes the exit channel, the networks that read the map, and the GRU, which holds what the map does not.

## Training and exploration pipeline

1. **Maps** (`training/mapgen`). A procedural generator of grid layouts: six floor-plan styles, tiers 0 to 4, a base and a wide profile (longer routes, tighter tick caps), randomised textures, light, start pose and monster facing; every map is checked solvable. The second phase trained on 4,500 maps. Evaluation suites come from seeds never trained on and are frozen as stored grids.
2. **Teacher, training only** (`training/teacher`). A scripted explorer that runs a shortest-path search on the agent's own memory crop. It decides from the network's inputs alone, never from engine state, and is never imported by the agent. It never labels attack or weapon, and no button while an enemy is visible.
3. **Exit detector** (`training/exitdet`). Supervised on 235,520 frames labelled by map geometry, with colour, tint, noise and flip augmentation and red look-alikes as hard negatives.
4. **PPO** (`training/rl`). Workers run the agent's own odometry, memory and encoder; GAE on two reward streams, truncated backpropagation through time, a clipped surrogate on the joint log-probability of the six heads and the manager's two choices, entropy per head. **Reward shaping** (`reward.py`) pays the scoring categories as events happen (exit, health and time left, keys, doors, sectors, kills) minus damage, death, wasted shots and door reversals. **Curiosity** (`intrinsic.py`): +0.01 for every 64-unit cell newly observed in the agent's own memory map and +0.02/√n on the n-th entry into a cell, with its own value head; standing still earns nothing and pacing between two cells decays. No privileged state is used. **Curriculum** (`curriculum.py`): stages over tiers, then tiers sampled by 1 − recent exit rate.
5. **What was run.** Combat: PPO alone from random weights on arenas, 118.8 M steps. Phase one (49.0 M steps, from the combat checkpoint): the teacher labels the states the student reaches and its buttons are executed with a probability β that falls from 1 to 0.1, fights are always the student's, and the loss adds cross-entropy to the labels times a weight w. Every 150 iterations the policy alone plays 384 held-out episodes, and w steps down (1, 0.5, 0.25, 0.1, 0.03, 0) only if that exit rate holds. Phase two (`unified_full_2`): w was 0.25, then 0.1 at 14.7 M steps, 0.03 at 29.5 M and **0 at 44.2 M**, with held-out exit rates 0.67, 0.72 and 0.77 at those steps. With the teacher off the rate went on to 0.83 (59.0 M) and 0.85 (73.7 M) (`training/logs/unified_full_2_train_log.csv`).
6. **Evaluation and promotion** (`evaluate.py`, `promote.py`, `final_pick.py`). A local scorer with the four scoring categories. A checkpoint is promoted only if it beats the shipped one, map by map, by more than one point and two standard errors on `dev_v1` or its wide twin (3:08 AM: a tie on `dev_v1`, a win on the wide suite), and passes the ship gate (size, provenance, official validator, four test files, latency, identical tree).

## Results

**Cost** (official validator on a fresh clone of commit 0bf38b2, template's Docker image built locally, pinned to one CPU, one 46-step episode; `training/logs/fresh_clone_validation.txt`): 2.4 ms median, 6.5 ms p99 and 7 ms maximum per decision (1.1 ms median natively on the laptop); peak RAM 794 MB of 8,192 MB; no GPU memory. **Exit detector** (`exit_detector.csv`; 20,480 frames each on held-out maps and on unseen textures): precision 0.998 to 0.999, recall 0.94 within 400 units, 0.72 to 0.79 at 400 to 640. **Training throughput**: 8,000 to 12,000 environment steps per second (128 worker processes × 12 environments, one GPU).

**The ladder** on the development suite `dev_v1` (20 unseen maps per tier; `training/logs/results.csv`). Tier columns: the local scorer's composite (100 is a perfect run). All: tier-weighted, with its standard error. Exit: exit rate on maps with an exit. w: the teacher's loss weight at that checkpoint. Held-out: weighted score on four held-out suites (640 maps; `training/logs/final_pick_report_*.txt`). Phase-two rows include the combat run and phase one. The scorer's category weights are our estimate: the numbers rank our own agents and are not the official score.

| Agent | Training behind it | w | Tier 1 | Tier 2 | Tier 3 | Tier 4 | All | Exit | Held-out |
|---|---|---|---|---|---|---|---|---|---|
| Random buttons | none | | 23.7 | 4.0 | −0.6 | −0.7 | 2.7 ± 0.7 | 0.00 | |
| Combat only | PPO alone, 118.8 M | none | 89.2 | 63.2 | 27.4 | 14.7 | 35.6 ± 3.0 | 0.30 | |
| Unified, 11:23 PM | + phase one, 41.8 M | 0.25 | 92.8 | 76.4 | 62.8 | 35.7 | 57.7 ± 4.3 | 0.61 | |
| Promoted 12:38 AM | phase two, 19.7 M | 0.1 | 87.2 | 84.7 | 81.8 | 46.6 | 68.8 ± 3.9 | 0.71 | 61.4 ± 1.2 |
| Promoted 1:38 AM | phase two, 49.2 M | 0 | 91.2 | 86.8 | 85.7 | 62.1 | 77.0 ± 3.9 | 0.83 | 72.2 ± 1.2 |
| Promoted 3:08 AM | phase two, 98.3 M | 0 | 83.2 | 89.0 | 92.9 | 59.5 | 77.8 ± 3.8 | 0.82 | 75.5 ± 1.1 |
| **Shipped**, 5:37 AM | phase two, 172.0 M | 0 | 87.2 | 92.1 | 86.8 | 71.8 | 81.9 ± 3.6 | 0.86 | 78.4 ± 1.1 |

Shipped policy file: main_0525_M, sha256 3f90be8826ef, 340 M environment steps in all. Decoding: move, strafe, turn, USE and weapon take their most likely option, attack is sampled, and so are the manager's choices.

## Rule compliance and limits

- **Decisions.** Every button comes from a network head. The only hand-written code that touches action choice is two masks computed from observation fields (`spec.action_masks`): USE only when the ray straight ahead ends on a door within USE reach, ATTACK only when an enemy is visible and the weapon is not the melee slot. The same function masks the logits in training.
- **Agent folder.** `submission/` uses numpy and torch on the CPU and has no search over the map, no inverse trigonometry and no engine access; a lint (`training/tests/test_rules.py`) and a fuzz test enforce this on every push. Not learned there: odometry, memory writes, encoder, the two masks, a compute governor driven only by the clock, an exception handler. Scripted code exists only in `training/`: the teacher, and a wanderer that collects frames for the exit detector.
- **Limit: the manager.** The hierarchy is in the network and was trained with the teacher's labels. After the teacher's weight reached zero the manager's two heads drifted to near-uniform outputs while the scores kept rising: sub-goal entropy 0.68 → 2.03 nats (maximum 2.08), target entropy 1.56 → 7.7 (maximum 8.32), agreement with the teacher's target 0.63 → 0.03 (same log, 44 M to 88 M steps; on the 3:08 AM weights the entropies are 2.01 and 7.44, `manager_probe.txt`). In the shipped weights the manager's choices are therefore close to random; the controller acts from its sensors, its recurrent state and the value-field patch. We do not claim that the shipped agent plans through sub-goals.
- **Other measured limits** (1:38 AM agent, 1,195 held-out episodes, `failure_breakdown_main_0135.txt`). Death rate: 0.41 on maps with a hell knight or cacodemon against 0.05 on the others; among the others 0.15 with one chaingunner on the map, 0.50 with two or more, 0.005 with none. Timeouts are search failures, not standing still: exit never seen (51 of 127), key never picked up (43), locked door never opened with the key held (27). About 9 shots per kill in tier 4 (12:38 AM agent). The detector knows the red exit pad of the grid layouts only; a switch-style exit cannot be pressed under the USE mask. The time-budget input is the constant 525 decisions.
