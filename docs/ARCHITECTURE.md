# Architecture

One section per module group of the agent, in the order `DoomAgent.act()` uses them. Each section
gives the tensor shapes, how the module works and the reason for each design choice. The text
follows the module docstrings under `submission/doomnet/`, which are the reference; where this file
and a docstring disagree, the docstring describes the code.

This file describes what the code computes. What the trained, shipped weights make of it is a
separate question with a measured answer, and for the manager that answer limits what can be
claimed: in the shipped weights its sub-goal is close to a uniform draw and its target is spread
over hundreds of map cells. Section 6e states this with the numbers; read it together with 6b to 6d.

```
 observation ──> 1 odometry ──> 2 exit detector ──> 3 memory map ──> 4 encoder, masks ──> 5 PolicyNet ──> 9 buttons
                 pose             RGB -> sighting     write, crop       fixed-size inputs     manager (1 step in 8),
                                                                                              value propagation, controller
```

| Module | File | Learned? | Parameters |
|---|---|---|---|
| Shared layout and action masks | `submission/doomnet/spec.py` | no (the masks are the only hand-written code that touches action choice) | 0 |
| Odometry | `submission/doomnet/odometry.py` | no | 0 |
| Spatial memory | `submission/doomnet/memory.py` | write path no; read by learned modules | 0 |
| Encoder | `submission/doomnet/encode.py` | no | 0 |
| Exit detector | `submission/doomnet/model.py`, `ExitDetector` | yes, supervised | 416,867 |
| Policy | `submission/doomnet/model.py`, `PolicyNet` | yes: PPO, plus distillation from a training-only teacher until its weight reached zero | 1,307,153 |
| Assembly | `submission/agent.py` | no (pipeline, decoding, time governor, exception handler) | 0 |

Parameters of `PolicyNet` by part: ray encoder 155,008; entity encoder and attention queries
17,792; scalar encoder and fusion 134,208; map tower 314,448; manager heads and pointer 88,620;
value propagation 14,404; goal read and GRU 473,728; action heads 47,119; value heads 61,826.

---

## 1. Shared layout: frames, inputs, heads and masks (`spec.py`)

**Frames.** The ego frame has x forward and y to the left, angles counter-clockwise, as in the
wrapper (ray k points at angle k · 2π/64). The start frame is the ego frame at the first
observation of an episode. A pose is (x, y, θ) in the start frame, integrated from the agent's own
actions and cleared before every map. The agent never receives map coordinates from the engine.

**Network inputs for one decision** (no batch dimension):

| Input | Shape | Content |
|---|---|---|
| rays | (64, 7) | distance / 1024, hit type one-hot (none, wall, door, locked door, enemy, item) |
| ents | (32, 17) | distance / 1024, sine and cosine of the bearing, relative velocity forward and right / 10, type one-hot (12 types) |
| ent_mask | (32,) bool | True where a row holds a real entity |
| scal | (44,) | health, armor; selected weapon slot one-hot (8); selected ammunition and four ammunition pools; six key flags; velocity (3); step, assumed budget, budget left; sine and cosine of the heading; offset inside the current 64-unit cell (2); previous buttons (9); steps since USE went down; exit sighting (flag, forward, left) |
| map | (11, 64, 64) | crop of the memory map, 64 units per cell, centred on the agent, axes of the start frame |
| rgb | (3, 120, 160) uint8 | the frame, for the exit detector only |

**Actions.** Six categorical heads: move (none, forward, back), strafe (none, left, right), turn
(none, left, right), attack, use, weapon (off, on). `heads_to_buttons` turns one choice per head
into the wrapper's nine buttons.

*Why factored heads:* the nine buttons have 2⁹ joint settings, more than half of them
contradictory (forward and back together). Six small heads cover the 216 that are not with 15
logits, and training gives each head its own entropy bonus and its own distillation label.

**Sub-goals.** `SUBGOALS = (explore, sweep, get_key, open_door, open_locked, go_exit, fight,
get_pickup)`. The manager chooses one and points at a target cell every `MANAGER_PERIOD = 8`
decisions. No code acts on the choice: it reaches the controller only as an embedding and as the
target of the value-propagation block (`agent.debug_state()` reports it to the viewer, and that is
the only place it is read). The names are the labels the training-time teacher used; in the shipped
weights the choice is close to uniform (section 6e), so a sub-goal shown by the viewer says little
about what the agent is doing.

**Action masks** (`action_masks`), the only hand-written code that touches action choice:

- `use_ok`: the ray straight ahead ends on a door or locked door within the engine's USE reach
  (64 units).
- `attack_ok`: at least one enemy entity is visible and the selected weapon is not the melee slot.

A False sets the "on" logit of that head to −∞. *Why:* a mask removes an option that would do
nothing; the network chooses among the rest.
*Why the melee clause:* a melee hit turns the player towards its target with no turn key pressed,
which would silently break the heading estimate that the memory map is built on. The same function
is applied to the logits in training (`training/rl/vecenv.py`, `training/rl/ppo.py`) and in the
agent.

**Time budget.** The tick cap of a map (2,100 to 4,200 tics) is not observable. The policy takes
an assumed budget as an input (`budget_frac`, `budget_left`): in training the true cap with noise,
at evaluation the constant `DEFAULT_BUDGET_STEPS = 525`, the shortest cap. *Why:* when to stop
exploring and leave is then decided by the network from an explicit input and learned from reward,
not coded.

---

## 2. Odometry (`odometry.py`) — hand-written

**Purpose.** Turn (previous buttons, new observation) into the pose (x, y, θ) in the start frame.

**Shapes.** `prev_action (9,)`, `vitals (17,)`, ray `distance (64,)` and `hit_type (64,)` → three
floats.

**Heading.** Exact. A turn key turns 320/65536 of a circle per tic for the first five tics it is
held and 640/65536 afterwards, and the first seven tics of an episode ignore input. Both rates are
whole multiples of 1/1024 of a circle, so the heading is kept as an integer.

**Position, free motion.** For each of the four tics of a decision: add thrust along the heading,
move by the momentum, multiply the momentum by the friction factor. The momentum is read again
from `vitals` after every step, so nothing accumulates. If the predicted momentum equals the
reported one, nothing disturbed the step and the predicted displacement is used.

**Position, disturbed steps.** A wall, a monster or a hit changes the momentum inside a step. The
displacement of such a step is measured from the rays: walls do not move, so every wall point of
the new scan must lie on the straight wall the previous scan saw there. Each matched ray gives one
linear equation n · t = b in the displacement t (n is the wall's normal); all are solved together
by gated least squares, a 2 × 2 system in closed form. A point is only paired with a previous
point of its own kind (wall with wall, door with door) whose wall runs the same way, so a surface
that has just come into view (the room behind an opening door) is left out. In a long corridor,
where the walls constrain one direction only, the other component comes from a model of the
engine's wall slide, or from a single ray that sees the corridor's end.

**No search, no inverse trigonometry.** Correspondences are nearest neighbours between two
64-point scans; everything else is closed-form arithmetic.

**Why this design.** The heading follows exactly from the buttons, free motion from the known
friction model, and the rays measure the few steps the model cannot predict, so drift stays small
(see Measured). That matters because the memory map is written with this pose: an error in the
pose becomes an error in the map.

**Measured.** The local scorer compares the agent's pose estimate with the true pose in every
scored episode (`training/logs/results.csv`, columns `odom_err_final`, `odom_err_max`,
`odom_heading_err_max`). For the agent promoted at 3:08 AM: mean final position error 0.32 map
units over the 80 `dev_v1` episodes and 0.20 over the 60 `dev_wide_v1` episodes (one grid cell is
64 units), and a heading error of zero in every episode (tags `main_0305_dev`, `main_0305_wide`).
An earlier drift script (26 Nov, 320 episodes on 80 generated maps with monsters; its output is not
in the repository) gave a median of 0.04, a 90th percentile of 0.27 and a worst case of 12.0 units.
Known weak spots: a monster that pushes the agent along a one-cell corridor (how far the monster
moves within a step cannot be observed; up to 45 units in a stress test of that script), and
standing on a pickup that cannot be taken. Cost: 0.06 ms per decision (`training/logs/latency.txt`).

---

## 3. Spatial memory (`memory.py`) — hand-written write path, learned readers

**Purpose.** Remember walls, free space, doors, pickups, keys, enemies, the exit and the agent's
own trail for a whole episode.

**Shapes.** `update(pose, obs, prev_action (9,), exit_obs (3,) or None, step)` → None;
`crop(pose)` → float32 `(11, 64, 64)`, index [channel, x cell, y cell], agent in cell (32, 32).

**Stored state** (start pose at the centre):

| Array | Shape | Content |
|---|---|---|
| `occ` | 256 × 256 cells of 32 units, int | occupancy votes: a ray passing through a cell removes 1, a ray ending on a wall adds 3; negative = free, positive = wall, 0 = never observed |
| `flags` | 256 × 256, bits | closed door last seen here; locked door last seen here; pickup seen here; key seen here; the agent has stood here |
| `stamp` | 3 × 129 × 129 cells of 64 units | decision step of the last visit, the last enemy sighting, the last USE press |
| `exit` | 129 × 129 cells of 64 units | belief that the exit pad is here: each sighting adds 0.5 × the detector's confidence; sightings under confidence 0.5 are ignored |

**Channels of the crop:** free, wall, door, locked door, visited, recency (1 after a visit, fading
over 128 decisions), enemy (fading over 64), item, key, exit, USE pressed (fading over 64; tells a
tried door from an untried one). Each crop cell pools 2 × 2 stored cells (on if any of the four is
on). The crop is 4,096 units across.

**Write rules.**

- *Walls.* A ray that ends on a wall types the cell just behind the wall face, along the wall's
  normal where neighbouring rays show the wall's direction. Free space stops a fixed distance short
  of the wall. *Why votes instead of bits:* the pose is an estimate, so a wall point lands in the
  wrong cell now and then; with votes it is outvoted by the rays that keep passing through.
- *Doors.* A ray that ends on a door sets the door bit behind the door face; a ray that passes
  through a cell clears its door bits, because rays only pass through an open door. The bit reads
  "closed door last seen here": it goes when the door is seen open and comes back when it is seen
  closed. Doors and walls vote against each other in a cell, which keeps door bits off the wall
  cells beside a door.
- *Entities.* Enemy sightings stamp their cell; pickups and keys set bits; the agent's own cell is
  marked visited; a USE press stamps the cell straight ahead.
- *Exit.* Written only from the exit detector's output, the one learned writer.
- *Bounds.* Writes outside the arrays are dropped, crops past the edge are zero-padded, and a pose
  that is far away or not finite never raises.

**Why this design.**

- *Allocentric, not rotated with the agent.* A cell keeps its place in the crop when the agent
  turns, so the tower and the value propagation see a picture that only translates. Rotation is
  applied once, where the controller reads its 9 × 9 patch.
- *Written by fixed geometry.* The grid is exact where the odometry is exact and costs about
  0.2 ms per update. The write path is hand-written path integration and ray carving. It is not
  learned and it is not differentiable; we do not describe it otherwise.
- *What is learned around it.* The exit detector fills the exit channel. The map tower, the
  manager and the value-propagation block read the crop. No frontier channel is coded: unexplored
  space has to be found by the tower from the free and wall channels. The GRU holds what the map
  does not.
- *Two resolutions.* Occupancy and flags are stored at 32 units. The stamps and the exit belief
  are written at single points and only ever read at the crop's resolution, so they are stored at
  64 units. The network view at 64 units keeps the crop at 64 × 64 while covering 4,096 units.

---

## 4. Encoder (`encode.py`) — hand-written

**Purpose.** Turn the wrapper's observation dict, the pose estimate and the previous action into
the fixed-size arrays of section 1.

**Shapes.** obs, pose, `prev_action (9,)`, `exit_obs (3,)` or None → `rays (64, 7)`,
`ents (32, 17)`, `ent_mask (32,)`, `scal (44,)`, all float32 except the mask.

Every value is a rescaled copy of an observation field, of the agent's own previous action or of
its own pose estimate. The only state is the USE timer: the number of decisions since USE last went
from released to pressed. *Why the timer:* a second press while a door is still moving reverses
it, so the network is told how long ago it pressed and does not have to keep that in the GRU alone.

**Why this design.** Fixed scales and one-hot types keep every input of order 1 and finite
whatever the engine reports, so the networks never see raw units or out-of-range values. Entity
bearings are passed on as the sine and cosine that the wrapper reports.

---

## 5. Exit detector (`model.py`, `ExitDetector`) — learned

**Purpose.** Find the exit pad in the RGB frame. The pad is a floor texture: it is not in the rays
and not in the entity list.

**Shapes.** `rgb (B, 3, 120, 160)` uint8 or float → `(B, 3)`: a visibility logit, and the pad
position in the agent's frame as forward / 1024 and left / 1024. The agent applies a sigmoid to the
logit and passes (confidence, forward, left) to the memory map and the encoder.

**Layers.** Four 3 × 3 convolutions (16, 32, 48, 64 channels) with 2 × 2 max pooling after the
first three, a 1 × 1 convolution to 16 channels, then two linear layers (128, 3).

**Why this design.**

- The floor can only appear below the horizon (image row 60), so the network reads rows 48 to 119.
- A distant pad is one or two pixels tall. The first convolution therefore runs at full resolution
  and every reduction is a max pool, which keeps the response of a thin sliver.
- The last layers keep the 9 × 20 layout (flatten, not global pooling), because image position is
  what encodes distance and bearing.
- A separate small network, not pixels into the policy: the detector is trained once by
  supervised learning, and the policy can be trained on an exit sighting computed from geometry
  without running a CNN on every frame of every environment.

Training and measurements: `training/README.md`, stage 3.

---

## 6. Policy network (`model.py`, `PolicyNet`) — learned

**Shapes.** `rays (B, 64, 7)`, `ents (B, 32, 17)`, `ent_mask (B, 32)`, `scal (B, 44)`,
`map (B, 11, 64, 64)` → six logit tensors sized (3, 3, 3, 2, 2, 2), `value (B,)`,
`value_int (B,)`, `subgoal (B,)`, `target (B,)`, `subgoal_logits (B, 8)`,
`target_logits (B, 4096)`, `value_field (B, 1, 64, 64)`, `ent_attn (B, 32)`.

### 6a. Sensors (every decision)

| Part | Layers | Output |
|---|---|---|
| Rays | circular pad, three 1-D convolutions (32, 64, 64 channels; strides 1, 2, 2), linear | (B, 128) |
| Scalars | linear, ReLU | (B, 64) |
| Entities | shared linear per entity → (B, 32, 64); four attention heads; set maximum | (B, 256) + (B, 64) |
| Fusion | linear over the concatenation, ReLU | (B, 256) |

*Why a circular convolution:* the 64 rays cover the full circle, so ray 63 is the neighbour of
ray 0; with circular padding a doorway looks the same wherever it is around the agent.

*Why attention over a set (variable numbers of enemies):* the wrapper lists 0 to 32 entities in no
fixed order. Every entity goes through the same embedding. Four attention heads pool them, with
queries computed from the agent's own scalars (health, ammunition, weapon), so what it attends to
can depend on its condition. A constant "nothing" slot takes the weight when no entity is visible
and padding rows get none. The maximum over the set of entity embeddings is concatenated to the
four pooled vectors. The mean attention per entity is returned as `ent_attn`.

### 6b. Manager and map tower (first decision of a map, then every 8 decisions)

**Map tower.** `map (M, 11, 64, 64)` plus two coordinate channels → cell features
`(M, 32, 32, 32)`, one vector per 2 × 2 block of crop cells, and a summary vector `(M, 128)`. A
three-scale convolutional pyramid (32 × 32, 16 × 16, 8 × 8) with a top-down pass, so every cell's
features carry context from the whole crop. Both outputs are normalised per sample (group norm,
layer norm). *Why:* their scale then cannot drift while the weights change, which keeps the
pointer, the value propagation and the manager's logits on a steady input.

**Sub-goal.** From the fused sensors, the GRU state and the tower's summary: a categorical over the
8 sub-goals.

**Target pointer.** One softmax over all 4,096 crop cells, conditioned on the chosen sub-goal:
every cell is scored from its own tower features, shifted by the manager's context and the
sub-goal embedding. *Why a pointer:* the target is a cell of the agent's own map, whatever is
there, and the same choice seeds the value propagation.

### 6c. Value propagation (same decisions as the manager)

**Shapes.** cell features `(M, 32, 32, 32)`, sub-goal embedding `(M, 32)`, target `(M,)` → value
field `(M, 1, 64, 64)`.

One small convolutional cell applied K = 12 times on a 32 × 32 grid, in the style of value
iteration networks. The cell is a Bellman backup whose parts are all learned:

    V ← max over moves a of ( r_a + γ_a · (P_a ∗ V) )

- V: 8 value channels.
- r_a: "reward" maps computed once per call from the tower's cell features, a one-hot of the chosen
  target cell, every cell's offset from the target, and a projection of the sub-goal embedding.
- P_a: 3 × 3 kernels, one per move (4) and value channel, normalised with a softmax so that each is
  a probability distribution: a learned transition.
- γ_a: learned discounts, kept inside (0, 1) by a sigmoid.

**Why this design.**

- *Contraction.* Every P_a is a distribution and every γ_a is below one, so one application is a
  contraction: K applications cannot blow up whatever the weights are. A free convolutional cell
  let the field grow by many orders of magnitude in learning runs.
- *Max, not mean.* The max keeps the best continuation, so a value can travel one grid cell per
  application without being blurred away.
- *Nothing about walls is coded.* Which cells are free and how values may flow is decided by the
  learned r_a and P_a. This is a learned module with trained weights, not a search; the lint
  cannot tell the two apart, which is why it is stated here.
- *Seeded by the chosen cell.* The field is a function of the choice in force, which makes forced
  re-evaluation in training and the sub-goal clamp well defined.
- *Supervision.* While the teacher's loss weight is above zero, the field is regressed on
  0.95 ^ (the teacher's travel steps to the target), which has the same form as a discounted
  value. Once the weight is zero nothing supervises the field directly: it is trained only through
  the controller's use of the patch it reads.
- *Cost.* Tower and propagation run on one decision in 8 and their output is held in the state.

### 6d. Goal bottleneck, GRU and heads (every decision)

**What the controller receives from the levels above it** (with `strict_goal_bottleneck=True`,
the default): the embedding of the sub-goal in force `(B, 32)`, and a 9 × 9 patch of the value
field read around its own position and rotated into its own frame, as differences to the value at
its own cell times a fixed gain, mapped to `(B, 64)`. It never sees the map, the map features or
the target index. Rays, entities and scalars reach it directly, so fighting and local movement do
not depend on the manager.

*Why a bottleneck:* the goal reaches the controller through one narrow, inspectable channel. No
code acts on the sub-goal or the target, so clamping the sub-goal (`PolicyNet.clamp_subgoal`, for
tests and the viewer) changes behaviour only because the learned controller is conditioned on it.

*Why differences times a fixed gain:* the controller needs the shape of the field around itself,
not its level. Dividing each patch by its own size silenced the channel for some seeds early in
training; with a constant gain nothing else in the field can attenuate it.

*Reading a held field.* Between manager steps the crop is re-centred on the agent while the held
value field stays in the frame of the step that produced it. The controller therefore counts the
whole cells it has crossed since then, from the sub-cell offset and the velocity scalars, and reads
the field at its current position. The patch is rotated with the sine and cosine of the heading;
no angle is computed.

**GRU.** `GRUCell(256 + 96 → 256)` over fused sensors and goal features.

**Heads.** Actor: linear (128) and linear to 15 logits, split (3, 3, 3, 2, 2, 2). Critic: two
values, for the extrinsic and the intrinsic return; the critic also reads the tower's summary
vector (the bottleneck applies to the action heads only).

**State carried between decisions:** GRU state `(B, 256)`; the manager's sub-goal and target; the
held value field `(B, 1, 64, 64)` and map summary `(B, 128)`; the cell counter and the step count
since the manager last ran.

**Why this design overall.** The costly map tower and value propagation run once per 8 decisions
and are cached, the per-decision controller is a few dozen small operators, and the three levels
are separate enough to be supervised and inspected one by one: sub-goal and target by the
manager's labels and by PPO, the field by a regression target, the buttons by PPO. That
inspectability is what showed the result of section 6e.

### 6e. What the shipped weights do with the hierarchy (measured)

The manager and the value-propagation block were trained with the teacher's labels: a sub-goal and
a target cell for the manager, a travel-step field for the propagation. Those labels entered the
loss with a weight that was stepped down as the policy's own held-out exit rate rose, and that
weight reached zero 44.2 M steps into the second training phase (iteration 450). From that step
on the run is reinforcement learning alone, and the two manager heads are trained only by the PPO
gradient and their entropy bonus.

What happened to them, from `training/logs/unified_full_2_train_log.csv` (statistics over whole
rollouts, about 98,000 decisions per iteration):

| Iteration of phase two | Steps | Sub-goal entropy (max 2.08 nats) | Target entropy (max 8.32) | Agreement with the teacher: sub-goal | target | Held-out exit rate at the last report |
|---|---|---|---|---|---|---|
| 450 (teacher weight reaches 0) | 44.2 M | 0.68 | 1.56 | 0.84 | 0.63 | 0.77 |
| 500 | 49.2 M | 1.92 | 3.76 | 0.31 | 0.46 | 0.77 |
| 600 | 59.0 M | 2.01 | 6.98 | 0.15 | 0.21 | 0.83 |
| 750 | 73.7 M | 2.03 | 7.73 | 0.15 | 0.05 | 0.85 |
| 899 (end of the log copy) | 88.4 M | 2.03 | 7.65 | 0.15 | 0.03 | 0.85 |

On recorded inputs (`training/logs/manager_probe.txt`: 80 decisions of the provenance probe, 10 of
them manager steps, so a small sample) the
checkpoint promoted at 3:08 AM (iteration 1,000) has a sub-goal entropy of 2.01 nats with a mean
top probability of 0.19 (a uniform choice has 0.125), and a target entropy of 7.44 nats with a
mean top probability of 0.005; the checkpoints at iterations 1,225 and 1,525 are the same within
0.4 nats. The first shipped agent, trained with the teacher throughout, has 0.15 and 0.62 nats.
"The shipped weights" below means the policy file the top-level README names as shipped: a
checkpoint of this lineage from iteration 1,000 on, or a weight average of such checkpoints of the
run and of its branch (the README says which, and how the average was made). The probe log covers
the three checkpoints listed here. For the shipped file itself the README gives the same two
entropies under "Known limits"; `python docs/brief/fill_final.py --manager-probe` computes them
from the file on the same recorded inputs.

What follows, and what does not:

- In the shipped weights the manager's sub-goal and target are close to random draws (the agent
  samples them, section 7). The hierarchy is in the architecture and was trained with the
  teacher; it is not what steers the shipped agent. We do not claim that the agent plans through
  sub-goals or that it walks towards a target cell it has chosen.
- The buttons come from the controller: its rays, entities and scalars, its recurrent state, and
  the 9 × 9 patch of the value field. The field is still computed from the map tower's features
  on every manager step, now seeded by a near-random target and conditioned on a near-random
  sub-goal. The memory map therefore reaches the action heads only through that patch (the
  bottleneck of 6d), and how much it carries in the shipped weights has not been measured.
- The controller is still conditioned on what it is given. In the same probe, replacing the
  manager's sub-goal by another one moves the move and strafe probabilities by about 0.10 in total
  variation, and replacing the target by 0.03 to 0.04. With a near-uniform sub-goal this is
  variation in the controller's behaviour from one 8-decision period to the next, not a plan.
- Scores rose while this happened: the held-out exit rate in the table, and the development and
  held-out composites of the agents promoted after iteration 450 (top-level README, Results). We
  cannot say whether the same network with a working manager would score higher. We can say that
  the shipped agent's score is earned by the recurrent controller, its sensors and the value-field
  patch.

---

## 7. Assembly (`submission/agent.py`)

**Purpose.** Turn one observation dict into one `(9,)` button array, within the time budget,
without ever raising.

**Pipeline, identical on every decision:** odometry (previous action, new observation) → exit
detector → memory update → encoder and masks → crop (rounded to uint8, as the training workers
store it) → `PolicyNet` → decoding. *Why one fixed pipeline:* it is the same one the training
workers run, so the networks meet at evaluation exactly the inputs they were trained on; a test
compares the two paths value by value.

**Decoding.** Each head is decoded in one of two fixed ways: by its most likely option, or by a
draw from its distribution, as in training. Which heads take the most likely option is a setting
stored with the weights (`greedy_heads` in the config of `weights/policy.pt`). A file without that
entry gets the default: USE and weapon take their most likely option, because a second USE press
on a moving door reverses it, and move, strafe, turn and attack are sampled. The agent prints the
setting in force when it starts, and the top-level README names the setting of the shipped file.
The setting never looks at an observation. The manager's sub-goal and target are always sampled.
The generators belong to the agent and are re-seeded in `reset()`, so a map always plays the same
way.

**Not learned in this file, and none of it reads an observation to choose a button:**

- the two masks of section 1, applied to the logits before decoding;
- a time governor driven only by the clock: if the mean `act()` time on the current map passes
  24 ms the exit detector rests on odd decisions, and past 32 ms the policy network does too and
  the previous action is repeated (still under the masks). In the scoring runs of the agent in
  `training/logs/results.csv` (49 by 5:26 AM) the mean `act()` time was 1.5 to 6.6 ms, far below
  the first threshold;
- an exception handler: a decision on which an internal error occurred answers with no buttons,
  the agent's state is cleared, and the network carries on at the next decision.

**Start-up.** Inference runs on the CPU with one thread whatever the `device` argument says, and
CUDA is never initialised. Weights are loaded and every code path is run a few times in
`__init__`, so the first decision of a map costs the same as any other. The agent refuses to start
when a weights file is missing, is a Git LFS pointer, does not fit its network, or holds a
non-finite value.

**Cost per decision** (`training/logs/latency.txt`, one laptop thread): exit detector 0.37 ms,
policy network 0.38 ms, memory update and crop 0.18 ms, odometry 0.06 ms, encoder 0.04 ms,
decoding 0.01 ms; 1.05 ms on average against a budget of 40 ms (2.0 ms on the one decision in 8
that runs the manager, 0.9 ms on the others). In the official validator on a fresh clone, inside
the template's Docker image on one CPU: median 2.4 ms, maximum 7 ms over one episode of 46 decisions
(`training/logs/fresh_clone_validation.txt`).

---

## Training-side counterparts

The modules above are trained and exercised by code under `training/`, described in
[training/README.md](../training/README.md): the map generator, the scripted navigation teacher
(training only, never imported by the agent), the exit detector's data and training, the recurrent
PPO with reward shaping, curiosity and curriculum, and the local scorer.
