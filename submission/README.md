# `submission/`: the agent that is evaluated

The scorer imports `agent.py` and calls `DoomAgent.act(observation)` once per decision. Everything
that call runs is in this folder. It uses numpy and torch only and runs on the CPU.

## Modules

Listed in the order `act()` uses them. Each is labelled **observation encoding**, **action mask** or
**learned**.

| Module | Label | What it does |
|---|---|---|
| `doomnet/spec.py` | observation encoding, action mask | Shared constants: coordinate frames, the layout of every network input, the memory-map channels, the sub-goal names, and the six action heads with their conversion to the nine buttons. Also `action_masks()`, the two action masks described below. |
| `doomnet/odometry.py` | observation encoding | Estimates the agent's pose (x, y, heading) relative to where the episode started, from the buttons it pressed, the velocity in `vitals` and the rays. The engine never reports a position. |
| `doomnet/memory.py` | observation encoding | Spatial memory: a top-down grid in the start frame (256 × 256 cells of 32 map units) that holds what the rays, the entity list and the exit detector have shown. It is written with fixed geometry from the pose estimate and the observation. `crop()` returns the view the networks read: 64 × 64 cells of 64 units centred on the agent, with 11 channels (free, wall, door, locked door, visited, recency, enemy, item, key, exit, USE pressed). |
| `doomnet/encode.py` | observation encoding | Rescales one observation into the arrays the policy takes: 64 rays × 7 features, up to 32 entities × 17 features with a padding mask, and 44 scalars (vitals, heading, time budget, previous action, steps since USE, the exit detector's output). |
| `doomnet/model.py` | learned | `ExitDetector`: a CNN that finds the exit pad in the RGB frame; the pad is not part of the rays or the entity list. `PolicyNet`: three learned levels. A manager picks one of 8 sub-goals and points at a target cell of the map crop, every 8 decisions. A value-propagation block, one small convolutional cell applied several times, turns that target into a value field over the map. A controller (circular convolution over the rays, attention over the entities, a GRU) outputs six action heads: move, strafe, turn, attack, use, weapon. |
| `agent.py` | learned | `DoomAgent`, the entry point the scorer calls. `__init__` loads the weights of `PolicyNet` and `ExitDetector` and runs every code path a few times, so that the first decision of a map costs the same as any other. Each `act()` runs odometry, exit detector, memory map, encoder with the two action masks, and `PolicyNet`, in that order, and turns the six head outputs into the nine buttons. `reset()` clears the pose, the map, the recurrent state and the generators. The `device` argument is ignored: inference runs on the CPU with one thread. |

`weights/` is the folder for the trained parameters (Git LFS): `weights/policy.pt` and
`weights/exit_detector.pt`. If one of them is missing, or is a Git LFS pointer in place of the
file, `DoomAgent.__init__` raises. The one exception is the environment variable
`DOOMNET_ALLOW_UNTRAINED=1`, which tests and benchmarks set to run the networks with their initial
weights; the scorer does not set it. On start the agent prints each weights file with the first
12 hex digits of its sha256.
`requirements.txt` lists no packages: the agent needs nothing beyond the official image.

## What is hand-written and what is learned

- **Action choice.** Every button comes from the six heads of `PolicyNet`;
  `spec.heads_to_buttons()` turns one choice per head into the nine buttons. The only hand-written
  code that touches action choice is the two masks in `spec.py`. USE can be chosen only when the
  ray straight ahead ends on a door or locked door within USE reach. ATTACK can be chosen only
  when an enemy is visible and the selected weapon is not slot 1 (fist or chainsaw): a melee hit
  turns the player towards its target with no turn key pressed, which would break the heading
  estimate that the memory map is built on. A mask removes an option; the network chooses among
  the rest. The same function is applied to the logits during training
  (`training/rl/vecenv.py`) and in `agent.py`.
- **Sub-goals.** No code acts on the manager's sub-goal or target. They reach the controller only
  as a learned embedding and through the value field. `agent.py` reports them through
  `debug_state()` for the viewer, and that is the only place they are read.
- **Memory map.** Its write path is hand-written and not learned. `memory.py` projects rays,
  entities, the agent's own trail and its USE presses into the grid with fixed geometry. The
  learned parts around it are the exit detector, whose output fills the exit channel, and the
  networks that read the map.
- **Odometry and encoder.** Hand-written arithmetic on observation fields and on the agent's own
  past actions. Neither is learned and neither chooses an action.
- **`agent.py`.** The USE and weapon heads take their most likely option (`GREEDY_HEADS`): a
  second USE press on a moving door would reverse it. The move, strafe, turn and attack heads are
  sampled, as they were in training, and so is the manager's choice of sub-goal and target. The
  generators are re-seeded in `reset()`, so a map always plays the same way. Besides applying the
  masks, two things in the file are not learned, and neither reads an observation to choose a
  button. A time governor follows the mean `act()` time on the current map: above
  `GOVERNOR_DETECTOR_MS` the exit detector rests on every second decision, and above
  `GOVERNOR_NETWORK_MS` the policy network does too, with the previous action repeated in between
  (still under the masks); odometry, memory map and encoder run on every decision. An exception
  handler makes sure `act()` never raises: a decision on which an internal error occurred answers
  with no buttons, the agent's state is cleared, and the network carries on at the next decision.
- **Not in this folder:** path planning or graph search over the map, inverse-trigonometric aiming,
  scripted behaviours, and any access to the engine or to privileged state.

## Training-only code

Nothing under `training/` is imported by the agent. `training/` holds the code used only to
produce the weights: the map generator (`training/mapgen/`), the teacher that supplies
demonstration labels (`training/teacher/`), the reward shaping, intrinsic curiosity reward,
curriculum and PPO loop (`training/rl/`), the data collection and training of the exit detector
(`training/exitdet/`), and the scorer, runner, evaluation and viewer scripts. Privileged engine
state is used there for reward, labels and scoring, never here.

## How this is checked

`training/tests/test_rules.py` reads every file in this folder and fails if one of them

- names a banned technique (the inverse trigonometric functions, the classic graph searches and
  the containers they are built on), in code, comments or prose;
- imports anything other than numpy, torch and the standard library; the engine, the wrapper,
  network and process modules and `training` are banned by name;
- names a GPU: `.cuda`, a device number, or a string such as `"cuda"` used for anything but a
  comparison;
- has a `while` loop, leaves a loop with `break`, takes items from the front of a list, or has a
  function that calls itself. A search over a map written by hand takes one of these shapes
  whatever it is called, and nothing in this folder needs them.

Two things are beyond a check of the source text. It cannot follow a device that arrives in a
variable, such as the constructor's `device` argument. That is tested at run time instead:
`training/tests/test_fuzz.py` builds the agent with `device="cuda"` and fails if the agent so much
as tries to initialise CUDA. And it cannot tell a learned convolution applied K times, which is
what the value-propagation block in `doomnet/model.py` is, from the same loop with weights set by
hand. That is settled by reading the block: its weights are trained parameters.

The same test file checks this README against the code: every module has a row with one of the
three labels, each part of `agent.py` that the text names (the modules it runs, the weights it
loads, the time governor, the exception handler) must be found in that file, and so must every
constant and weights file cited by name. Had `agent.py` still been the template's random
baseline, the check would have required the README to say so.

`training/tests/test_fuzz.py` runs `DoomAgent` in its own process behind the scorer's protocol and
feeds it malformed observations, 1,200 decisions without `reset()` and resets in mid-episode. Every
answer must be a valid action, and the mean and the 99th percentile of `act()` time must stay
within the budget. Both files run on every push (`.github/workflows/ci.yml`), followed by the
official validator, run exactly as the scorer runs the agent, and by the agent's latency benchmark
when that file is present.
