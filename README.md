# DoomNet

Our entry for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**, built from the official
starter template. The agent plays unseen levels through the official `doom_env` wrapper: it has to
survive fights, cross multi-room layouts, find keycards, open locked doors and reach the exit.

**How the agent decides.** Every action comes from trained neural network weights. The policy is
trained by reinforcement learning: our own recurrent PPO (`training/rl/`), with reward shaped like the
evaluation categories, a curiosity reward and a curriculum. Combat is learned by PPO alone, from
random weights. Navigation is kickstarted by a scripted teacher that exists only at training time
(`training/teacher/`, never imported by the agent); its weight in the loss is stepped down as the
policy's own held-out exit rate rises. There is no rule tree, no path search and no aiming maths in
`submission/`; the only hand-written code that touches action choice is two action masks computed from
observation fields (`submission/doomnet/spec.py`), and a lint test enforces this on every push.

What is in this repository:

- **The agent** (`submission/`): a three-level policy network (manager, value propagation,
  controller) that reads a spatial memory the agent builds from its own actions and observations,
  plus a small CNN that finds the exit pad in the RGB frame. 1.72 M parameters, numpy and torch
  only, CPU only.
- **The training pipeline** (`training/`): a procedural map generator, our own recurrent PPO with
  reward shaping, a curiosity reward and a curriculum, a scripted navigation teacher used as
  training-time scaffolding, the exit detector's data collection and training, a local scorer and a
  viewer.
- **The documents**: the 2-page brief ([docs/brief/brief.pdf](docs/brief/brief.pdf), source
  [docs/brief/brief.md](docs/brief/brief.md)), the module-by-module description
  ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) and the pipeline in stage order
  ([training/README.md](training/README.md)).

## Results

Held-out suite `training/suites/heldout_v1.jsonl`: 100 generated layouts per tier from seeds never
used in training, stored as grids. Scores come from the local scorer (`training/scorer.py`), which
measures the four categories of [docs/EVALUATION.md](docs/EVALUATION.md); its category weights are
our own estimate, so the numbers rank our checkpoints against each other and are not the official
score.

| Agent | Tier 1 score | Tier 2 score | Tier 3 score | Tier 4 score | Exit rate | Kills | Deaths |
|---|---|---|---|---|---|---|---|
| Random buttons | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: fraction] | [fill: rate] |
| PPO combat policy alone | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: fraction] | [fill: rate] |
| Shipped agent | [fill: score] | [fill: score] | [fill: score] | [fill: score] | [fill: rate] | [fill: fraction] | [fill: rate] |

Shipped weights: `submission/weights/policy.pt` from [fill: training run, environment steps, sha256
prefix printed by the agent at start] and `submission/weights/exit_detector.pt`. The table is
written by `python -m training.evaluate --table` from `training/logs/results.csv`.

Component measurements that exist today, each with its log:

| What | Result | Where |
|---|---|---|
| Pose estimate | final position error median 0.04, 90th percentile 0.27, worst 12.0 map units over 320 episodes on 80 generated maps with monsters (one grid cell is 64 units); heading error at most 1.5e-9 rad | measured on 26 Nov with a drift script on a snapshot of the training maps; the regression tests are in `training/tests/test_memory.py` |
| Exit detector | precision 0.998 to 0.999; recall 0.94 within 400 units and 0.72 to 0.79 at 400 to 640 units, on held-out maps and on maps whose textures were never seen in training (20,480 frames each); 0.91 within 400 units in frames from real play with fights | `training/logs/exit_detector.csv` |
| Navigation teacher (training only) | reaches the exit on 1.00 / 0.98 / 0.99 of the tier 2 / 3 / 4 held-out layouts with the monsters removed (100 each, true tick cap as its budget) | `training/logs/teacher_gate.csv` |

Two short learning checks of the training loop are logged with their conditions: PPO from random
initialisation on single-room arenas (`training/logs/smoke_arena.txt`) and the unified loop on
monster-free maps (`training/logs/unified_smoke.txt`).

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
training/rl/               recurrent PPO: workers, buffer, loss, reward, curiosity, curriculum, export
training/configs/          configs of the combat run and the unified runs, with notes and the launch steps
training/scorer.py, runner.py, evaluate.py, mapmeta.py     local scorer, episode runner, suite evaluation
training/promote.py, ops/  score a checkpoint as a candidate agent, ship gate, remote evaluation runs
training/demo.py           headless viewer: video and contact sheets
training/bench/            latency benchmarks of the networks and of the whole agent
training/tests/            tests of all of the above; test_rules.py is the rule-compliance lint
training/logs/             curated logs of the runs quoted in the documents
docs/                      contract, rules, submission guide, evaluation details (template), ARCHITECTURE.md, brief/
env/, maps/, validate.py, tests/, docker/, Dockerfile       the official template, unchanged
```

Generated maps, datasets and run directories go to `data/` and `runs/`, which are not tracked.

## Reproducing the training

The stages, their files and their logs are described in [training/README.md](training/README.md);
the settings of each run are explained next to its config in `training/configs/`. In short:

```bash
# 1. Maps: 300 per tier 0-4, half of them with randomised looks, start pose and monsters;
#    and the frozen development suites rebuilt as WADs for the held-out check inside training
python -m training.mapgen.cli --out data/maps/train --n 300 --tiers 0,1,2,3,4 --seed 0 --randomize 0.5
python -m training.mapgen.cli --from-suite training/suites/dev_v1.jsonl     --out data/maps/dev
python -m training.mapgen.cli --from-suite training/suites/dev_nav_v1.jsonl --out data/maps/dev_nav

# 2. Navigation teacher: check it on monster-free held-out layouts (it is used inside step 4)
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

# 4b. The unified run on all tiers, continued from the combat checkpoint: PPO with curiosity and
#     curriculum, plus distillation from the teacher with a weight gated on held-out exit rate
python -u -m training.rl.train --maps data/maps/train/manifest.jsonl --config training/configs/unified_full.json \
    --init runs/combat_1/latest.pt --heldout data/maps/dev/manifest.jsonl \
    --workers 96 --envs-per-worker 12 --out runs/unified_full_1

# 5. Export in the agent's format, score on the held-out suite, promote only if better
python -m training.rl.export runs/unified_full_1/latest.pt --dest runs/unified_full_1/policy.pt \
    --episode data/maps/dev/dev_v1_t2_000.wad
python -m training.evaluate --policy training.runner:AgentPolicy --suite training/suites/heldout_v1.jsonl --tag NEW
python -m training.evaluate --compare OLD NEW      # exit status 0 only if NEW beats OLD
```

`--workers` and `--envs-per-worker` are sized for a 96-core Linux machine with a GPU for the
learner; the same commands run on a laptop with smaller numbers (the learner uses CUDA, else MPS,
else the CPU; rollout workers always run on the CPU). A 25-minute laptop version of step 4b is
`training/configs/unified_smoke.json`. The full launch procedure, with the lines a healthy run
prints and the log columns to watch, is `training/configs/LAUNCH.txt`.

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
  explorer, described under rule compliance below. Combat is learned by PPO alone.
- **Maps** (`training/mapgen/`): a procedural generator of grid layouts with randomised textures,
  light levels, start heading and offset, and monster facing and ambush flags; checkpoints are
  scored on frozen layouts from seeds never trained on.

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
  stepped down only while the held-out exit rate of the policy alone holds. The exit detector's
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

| Measurement | Result | Conditions |
|---|---|---|
| Full `act()`, macOS arm64, one thread | mean 2.3 ms, p95 4.1 ms, p99 6.4 ms, max 9.5 ms | 3 maps × 700 decisions, best of 3 repeats, machine under load (load average 46 on 15 cores); policy with initial weights, trained exit detector |
| Full `act()`, official image, linux/arm64, `--cpus 1` | mean 6.8 ms, p95 14.1 ms, max 20.9 ms | best of 3 repeats (repeat means 6.8 to 8.3 ms), host under load |
| Full `act()`, x86-64 CI runner | [fill: mean, p95, max in ms] | `training/bench/agent_latency.py` in the CI job |
| Building the agent | 0.9 s natively, 4.2 to 5.4 s in the image | limit 60 s; all warm-up happens here |
| Peak RAM in `validate.py --docker` | 792 MB | limit 8 GB |

Per decision on the Mac, by stage (microseconds): odometry 123, exit detector 788, memory update
205, map crop 148, encoder and masks 72, policy network 915 (322 from the manager steps, which run
once in 8 decisions and cost about 2.5 ms each; 593 from the others), decoding 42. The trained
policy has the same architecture as the one timed, so its cost per decision is the same.

Training throughput, measured on a 15-core laptop under load: 4,929 environment steps per second
from 10 worker processes with the agent's front end and no network; about 1,100 steps per second
for the whole PPO loop with 3 workers and the learner on the laptop GPU
(`training/rl/README.md`, "Throughput"). On the training machine: [fill: environment steps per
second and total steps of the final run].

## Template documentation

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules: what's allowed, what gets you disqualified |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts, replays |
