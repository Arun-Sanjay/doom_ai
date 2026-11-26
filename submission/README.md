# `submission/`: the agent that is evaluated

The scorer imports `agent.py` and calls `DoomAgent.act(observation)` once per decision. Everything
that call runs is in this folder. It uses numpy and torch only and runs on the CPU.

> **Status of `agent.py`.** The file in the tree is still the template's placeholder: it presses
> random buttons and loads no weights. The `agent.py` row of the table and the `agent.py` paragraph
> further down describe the assembled agent that replaces it. `training/tests/test_rules.py`
> compares this note with the file and fails when the two disagree: the note cannot be removed
> before `agent.py` loads the networks, and it cannot stay once it does.

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
| `agent.py` | learned | `DoomAgent`, the entry point. Loads the weights from `weights/`, then on every decision runs odometry, exit detector, memory map, encoder and `PolicyNet` in that order and converts the head outputs into the nine buttons. It ignores the `device` argument. |

`weights/` is the folder for the trained parameters (Git LFS). `requirements.txt` lists no packages:
the agent needs nothing beyond the official image.

## What is hand-written and what is learned

- **Action choice.** Every button comes from the heads of `PolicyNet`. The only hand-written code
  that touches action choice is the two masks in `spec.py`: USE can be chosen only when the ray
  straight ahead ends on a door or locked door within USE reach, and ATTACK only when an enemy is
  visible. A mask removes an option that would have no effect; the network chooses among the
  rest. The same function is applied to the logits during training.
- **Sub-goals.** No code reads the manager's sub-goal or target. They reach the controller only as
  a learned embedding and through the value field.
- **Memory map.** Its write path is hand-written and not learned. `memory.py` projects rays,
  entities, the agent's own trail and its USE presses into the grid with fixed geometry. The
  learned parts around it are the exit detector, whose output fills the exit channel, and the
  networks that read the map.
- **Odometry and encoder.** Hand-written arithmetic on observation fields and on the agent's own
  past actions. Neither is learned and neither chooses an action.
- **`agent.py`.** The USE and weapon heads take their most likely option, and the other heads are
  sampled from a generator that is re-seeded in `reset()`. Two safeguards never read the
  observation. A time governor follows the running mean of `act()` time on the current map: past
  a first fixed threshold the exit detector runs on every second decision, and past a second one
  the networks do too, with the previous action repeated in between. An exception handler returns
  no buttons for a decision on which an internal error occurred and clears the agent's state.
- **Not in this folder:** path planning or graph search over the map, inverse-trigonometric aiming,
  scripted behaviours, and any access to the engine or to privileged state.

## Training-only code

Nothing under `training/` is imported by the agent. That folder holds the code used only to
produce the weights: the map generator (`training/mapgen/`), the teacher that supplies
demonstration labels (`training/teacher/`), the reward shaping, intrinsic curiosity reward,
curriculum and PPO loop (`training/rl/`), and the scorer, runner and evaluation scripts. Privileged
engine state is used there for reward and scoring, never here.

## How this is checked

`training/tests/test_rules.py` reads every file in this folder and fails if one of them

- names a banned technique (the inverse tangent, the classic graph searches and the containers
  they are built on), in code, comments or prose;
- imports anything other than numpy, torch and the standard library; the engine, the wrapper,
  network and process modules and `training` are banned by name;
- names a GPU: `.cuda`, a device number, or a string such as `"cuda"` used for anything but a
  comparison;
- has an open-ended `while` loop, takes items from the front of a list, or has a function that
  calls itself. A search over a map written by hand takes one of these shapes whatever it is
  called, and nothing in this folder needs them.

Two things are beyond a check of the source text. It cannot follow a device that arrives in a
variable, such as the constructor's `device` argument. That is tested at run time instead:
`training/tests/test_fuzz.py` builds the agent with `device="cuda"` and fails if the agent so much
as tries to initialise CUDA. And it cannot tell a learned convolution applied K times, which is
what the value-propagation block in `doomnet/model.py` is, from the same loop with weights set by
hand. That is settled by reading the block: its weights are trained parameters.

The same test file checks this README against `agent.py`: the label of every module, the status
note at the top, and the parts of `agent.py` that the text above names (the modules it runs, the
weights it loads, the time governor, the exception handler).

`training/tests/test_fuzz.py` runs `DoomAgent` in its own process behind the scorer's protocol and
feeds it malformed observations, 1,200 decisions without `reset()` and resets in mid-episode. Every
answer must be a valid action, and the mean and the 99th percentile of `act()` time must stay
within the budget. Both files run on every push (`.github/workflows/ci.yml`), followed by the
official validator, run exactly as the scorer runs the agent, and by a latency measurement.
