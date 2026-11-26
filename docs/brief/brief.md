# DoomNet: a hierarchical policy over a self-built spatial memory

Yugaantar 2026, DevForge Track 1 "Can It Play DOOM?" · project brief · team [fill: team name] · [fill: public repository URL]

## USP

- **Three learned levels with one narrow goal channel.** A manager chooses one of 8 sub-goals and points at a cell of the agent's own map every 8 decisions. A value-propagation block, one learned Bellman backup applied 12 times, turns that target into a value field over the map. A recurrent controller presses the buttons. The controller receives the manager's decision only as a sub-goal embedding and a 9×9 patch of the value field; no code acts on the sub-goal or the target (`submission/doomnet/model.py`).
- **A spatial memory built from the agent's own actions.** The wrapper gives no coordinates. The agent integrates its pose from its buttons, velocity and rays, writes what it sees into an 11-channel grid in the frame of its starting pose, and keeps it for the whole map. The exit pad is absent from rays and entities, so a small CNN finds it in the RGB frame and writes it into the grid.
- **Reinforcement learning written from scratch, with a disclosed teacher.** Recurrent PPO in plain torch (no RL library), curiosity computed from the memory map, a curriculum over map tiers. Combat is learned by PPO alone. Navigation is started by a scripted teacher that sees only what the network sees; its loss weight is stepped down only while the held-out exit rate of the policy alone holds.
- **Small and CPU-only.** 1.72 M parameters. One decision takes 2.3 ms on one laptop core (budget: 40 ms). CUDA is never initialised.

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

**Sensory and state representation** (`encode.py`, `spec.py`). Rays become 64×7 (distance, hit type one-hot). Entities become up to 32×17 (distance, sine and cosine of the bearing, relative velocity, type one-hot) with a padding mask. 44 scalars carry vitals, selected weapon, ammunition, keys, velocity, heading, the step and an assumed time budget, the previous buttons, the steps since USE was pressed, and the exit sighting. **Controller.** A circular convolution over the rays (ray 63 is next to ray 0). The entities are a set: four attention heads whose queries are computed from the agent's own scalars, a constant "nothing" slot for an empty view, and a set maximum, so any number of enemies, projectiles and pickups from 0 to 32 is handled in any order. A GRU (256) feeds six categorical heads (3, 3, 3, 2, 2, 2 options) and two value heads. **Manager.** A three-scale convolutional tower over the map crop; a categorical over explore, sweep, get_key, open_door, open_locked, go_exit, fight, get_pickup; then a pointer, one softmax over all 4,096 crop cells conditioned on the sub-goal. **Value propagation.** V ← max over moves a of (r_a + γ_a · P_a ∗ V) on a 32×32 grid with 8 value channels: rewards r_a, 3×3 transition kernels P_a (softmax-normalised) and discounts γ_a (sigmoid) are all learned, so 12 applications are a contraction and cannot diverge. It is seeded with the target cell and conditioned on the sub-goal (14,404 parameters). Tower and propagation run once per 8 decisions and are cached. **Exit detector.** Five convolutions and two linear layers read image rows 48 to 119 and output a visibility logit and the pad position (forward, left).

## Spatial memory design

- **Pose** (`odometry.py`). Heading is exact: turn rates are whole multiples of 1/1024 of a circle, so it is an integer. Position follows the engine's thrust and friction model; a step disturbed by a wall or a monster is measured instead by matching the wall points of two consecutive ray scans (closed-form least squares, no search). Measured on 320 episodes over 80 generated maps with monsters: final position error median 0.04, 90th percentile 0.27, worst 12.0 map units (a grid cell is 64); heading error at most 1.5e-9 rad.
- **Map** (`memory.py`). 256×256 cells of 32 units centred on the start pose, cleared in `reset()`. Channels: free, wall, door, locked door, visited, recency, enemy, item, key, exit, USE pressed. Free and wall are vote counts (a ray passing through removes 1, a wall hit adds 3), so one misplaced hit is outvoted. A door bit means "closed door last seen here". Recency, enemy and USE decay from time stamps. Each exit sighting adds 0.5 × confidence to the exit belief of its cell.
- **Across rooms.** Walls, free space, doors, keys, items and the exit stay in the map for the whole episode, so a locked door seen before its key is still marked when the key is found. The network reads a 64×64 crop at 64 units per cell (4,096 units across) centred on the agent and aligned with the start frame, so cells keep their place when the agent turns.
- **What is learned.** The write path is hand-written path integration and ray carving. It is not learned and not differentiable. Learned parts: the exit detector that writes the exit channel; the tower, manager and value propagation that read the map; the controller's GRU, which holds what the map does not. No frontier channel is coded: the tower has to find unexplored space from the free and wall channels.

## Training and exploration pipeline

1. **Maps** (`training/mapgen`). A procedural generator of grid layouts: six floor-plan styles, tier 0 (monster-free navigation) and tiers 1 to 4. 1,500 training maps, half with randomised textures, light, start heading and offset, monster facing and ambush flags. Every map is checked solvable. Held-out suites (100 maps per tier) come from seeds never trained on and are frozen as stored grids.
2. **Teacher, training only** (`training/teacher`). A scripted explorer that runs a shortest-path search on the agent's memory crop. It decides from exactly the network's inputs (crop, pose estimate, observation, scalars, previous buttons), never from engine state, and is never imported by the agent. It labels sub-goal, target cell, move, strafe, turn, USE and a travel-step field. It never labels attack or weapon, labels no button while an enemy is visible, and nothing on maps whose start is off the 64-unit grid (39% of training maps).
3. **Exit detector** (`training/exitdet`). Supervised on 235,520 frames labelled by map geometry, with colour, tint, noise and flip augmentation and red look-alikes (keys, doors, fireballs, blood) as hard negatives. While the policy trains, the exit sighting is computed from map geometry with a noise model (misses, displaced and false sightings); the shipped agent uses the detector.
4. **PPO** (`training/rl`, plain torch). Workers run the agent's own odometry, memory and encoder; the learner runs one batched forward pass per step. Rollout buffer, GAE on two reward streams, truncated backpropagation in chunks of 16, clipped surrogate on the joint log-probability of the six heads and the manager's two choices, two clipped value heads, entropy per head. The two masks are applied in the rollout and in the loss. **Reward shaping** (`reward.py`) pays the scoring categories as events happen: exit, health and time left at the exit, keys, doors, locked doors, sectors entered, kills, minus damage, death, wasted shots and door reversals (our configs add 0.05 per shot that hits a monster, which is not part of the score). **Curiosity** (`intrinsic.py`): +0.01 for every 64-unit cell newly observed in the agent's own memory map and +0.02/√n on the n-th entry into a cell, with its own value head and advantage stream (weight 0.3). Standing still earns nothing and pacing between two cells decays, which is what drives the agent out of corners. No privileged state is used. **Curriculum** (`curriculum.py`): navigation, doors, keys, gauntlet, arenas in every stage, then tiers sampled by 1 − recent exit rate.
5. **One loop.** Combat starts from random initialisation (plus an offset on the weapon head's bias, so the pistol stays selected) on arenas with PPO alone. The unified run continues from that checkpoint: on labelled steps the teacher's buttons are executed with probability β (1 → 0.1 over 30 M steps), fights are always the student's, and the loss adds cross-entropy to the labels and a regression of the value field on the teacher's travel steps, both times a weight w. Every 150 iterations the policy alone plays 384 held-out episodes; w steps down (1, 0.5, 0.25, 0.1, 0.03, 0) only if that exit rate is at least 0.3 and has not fallen by more than 0.06 since the last step. The tick cap is not observable: the policy is given an assumed budget, the cap is terminal in training, and when to leave is learned from reward.
6. **Evaluation and promotion** (`training/evaluate.py`). A local scorer with the four scoring categories, plus a stuck rate (moved under 8 units in 10 decisions) and door reversals. Its `--compare` check passes a new checkpoint only if it is better by more than one point and two standard errors on the frozen suite.

## Results

| Measured | Value | Conditions |
|---|---|---|
| Parameters | 1,307,153 policy + 416,867 exit detector | `PolicyNet()` defaults, K = 12 |
| Time per decision | 2.3 ms mean, 4.1 ms p95; x86: [fill: mean ms on the CI runner] | full `act()`, one thread, laptop under load; 6.8 to 8.3 ms in the official image on one arm64 CPU |
| Exit detector | precision 0.998 to 0.999; recall 0.94 within 400 units, 0.72 to 0.79 at 400 to 640 | 20,480 frames each on held-out maps and on maps with unseen textures; 0.91 within 400 in real play with fights |
| Teacher, exit rate | tier 2: 1.00, tier 3: 0.98, tier 4: 0.99 | held-out layouts without monsters, 100 per tier; the same with the trained detector as its only exit sensor |
| Throughput | workers 4,929 steps/s; whole loop 1,100 steps/s on a laptop; [fill: steps/s of the final run] | 10 worker processes; 3 workers with the learner on the laptop GPU |

Held-out suite `heldout_v1` (100 maps per tier), played through `training/evaluate.py`. Tier columns are the local scorer's composite score (100 is a perfect run); the others are fractions over all tiers.

| Rung of the ladder | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Exit | Progression | Exploration | Combat |
|---|---|---|---|---|---|---|---|---|
| Random buttons | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: rate] | [fill: rate] | [fill: rate] |
| PPO combat policy alone | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: rate] | [fill: rate] | [fill: rate] |
| Unified loop, shipped checkpoint | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: rate] | [fill: rate] | [fill: rate] |
| [fill: further rung, if scored] | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: rate] | [fill: rate] | [fill: rate] |

| Comparison | Exit rate | Sectors visited | Stuck rate | Conditions |
|---|---|---|---|---|
| Curiosity on / off | [fill: on / off] | [fill: on / off] | [fill: on / off] | [fill: maps and steps of the pair of runs] |
| Teacher / student | 0.99 / [fill: student] | 0.98 / [fill: student] | 0.05 / [fill: student] | `heldout_nav_v1`, 300 layouts without monsters |

Final training run: [fill: environment steps, hours, final distillation weight w and β].

## Rule compliance and limits

- **Decisions.** Every button comes from a network head. The only hand-written code that touches action choice is two masks computed from observation fields (`spec.action_masks`): USE can be chosen only when the ray straight ahead ends on a door within USE reach, ATTACK only when an enemy is visible and the selected weapon is not the melee slot. USE and weapon take the most likely option; the other heads and the manager are sampled as in training.
- **Agent folder.** `submission/` uses numpy and torch on the CPU and contains no search over the map, no inverse trigonometry and no engine access. A lint (`training/tests/test_rules.py`) and a fuzz test enforce this on every push. Not learned there: odometry, memory writes, encoder, the masks, a compute governor driven only by the clock, and an exception handler.
- **Scripted code in `training/`.** The navigation teacher uses search. The frame collector of the exit detector has a scripted wanderer that shoots; it produces frames, never policy labels.
- **Limits.** The detector is trained on the red exit pad of the grid layouts: on a pad drawn with another floor texture it finds almost nothing, and the agent then has to walk onto the pad (the teacher without any exit sensor still exits 0.75, 0.85 and 0.91 of tiers 2 to 4). Detector recall within 400 units falls to 0.81 to 0.89 with a monster in view. The teacher's grid needs a start on the 64-unit grid; other maps are trained by reward alone. A monster pushing the agent along a corridor moved the position estimate by up to 45 units in a stress test. A switch-style exit cannot be pressed under the USE mask. At evaluation the time-budget input is the constant 525 decisions, the shortest tick cap.
