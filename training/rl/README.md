# training/rl: recurrent PPO from scratch

Training-time code. Nothing here is imported by the agent under `submission/`.
Plain torch and numpy; no RL library.

| File | What it does |
|---|---|
| `vecenv.py` | Worker processes own the `doom_env` environments and run the agent's own front end (odometry, memory map, encoder) per environment. Encoded network inputs come back through shared memory, map crops as uint8. The learner runs one batched forward pass per step. Environments are reused across episodes of a map; new maps are built in a background thread and drawn from the curriculum. |
| `rollout.py` | `Collector`: the per-step loop (observations in, forward, sample, actions out), state resets at episode starts, bootstrap values at the tick cap. `heldout_exit_rate`: one episode per held-out map with the agent's decoding. |
| `buffer.py` | Rollout storage, GAE for two reward streams, fixed-length chunks with stored initial recurrent states for truncated BPTT. |
| `ppo.py` | Masked factored action distribution, the loss, the learner, the learning-rate and entropy schedules, and the distillation schedule gated on held-out exit rate. |
| `reward.py` | Extrinsic reward from privileged state, shaped like the evaluation score. All coefficients in `RewardCoef`. |
| `intrinsic.py` | Curiosity from the agent's own memory map: newly observed cells plus an episodic visit-count bonus. No privileged state. |
| `curriculum.py` | Stages over map tiers, then sampling weighted to the tier with the lowest exit rate. |
| `train.py` | Entry point, CSV log, checkpoints, device selection (CUDA, else MPS, else CPU). |
| `deps.py` | The one place that imports from other areas (networks, front end, map metadata, environments, manifest). |
| `play.py` | Plays whole episodes with a checkpoint or with the uniform-random policy and prints mean kills, exits, deaths and cells observed: the baseline and the check of a trained policy. |
| `export.py` | Writes a checkpoint's network with the agent's own `save_weights` (`submission/weights/policy.pt`) and verifies it: read back with the agent's reader and compared with the checkpoint's provenance record, and with `--episode WAD` loaded by a real `DoomAgent` that plays one episode. `--check` verifies an existing weights file against the current code. |
| `provenance.py`, `probe_episodes.npz` | Ties a checkpoint to the code that trained it: a record of the source files' sha256 and of what the network and the front end gave on two recorded probe episodes, written into every checkpoint and export and compared on every load (see Checkpoints). |
| `eval_alone.py` | Exit rate of a checkpoint driving alone (no teacher), every map of a manifest played equally often, with the agent's decoding. |
| `agree.py` | Agreement of a checkpoint with the teacher's labels while the teacher drives (`--beta 1`) or the policy drives (`--beta 0`), and how much of it is lost when the manager's target is mirrored or its sub-goal changed. |
| `smoke_arenas.py` | Writes tiny single-room arenas (chosen monsters, no exit) with the map generator, for quick learning checks. |
| `bench.py` | Throughput: environment steps per second of the workers at several worker counts, learner samples per second per device. |
| `configs/` | The configs of the runs summarised in `training/logs/` (`smoke_arena.json`, `curiosity_on.json`, `curiosity_off.json`). The configs of the unified loop, with a text file explaining each, are in `training/configs/`. |
| `select_maps.py` | Writes a sub-manifest of maps matching filters on the generator's counts (for example single-room arenas without an exit). |
| `standins.py`, `standin_model.py` | Small stand-ins and toy environments used by the tests. |

## Policy and loss

One decision is six categorical heads (move, strafe, turn, attack, use, weapon). Every
`MANAGER_PERIOD` steps, starting at the episode start, the manager also chooses a sub-goal and a
target cell, which are held in the recurrent state until the next manager step.

    log pi(a|s) = sum_h log pi_h(a_h|s) + [manager step] (log pi_sub(g|s) + log pi_tgt(c|s))

USE and ATTACK are masked from observation fields (`doomnet.spec.action_masks`): the "on" logit is
-inf where the mask is False, in the rollout and in the loss alike.

    L = L_pi + vf_coef L_v + vf_int_coef L_v_int - ent_scale sum_k ent_coef_k H_k
        + w_distill (L_distill + field_coef L_field)

    ratio     = exp(log pi_new - log pi_old)
    A         = normalised (A_ext + int_coef A_int)             two GAE streams
    L_pi      = -mean(min(ratio A, clip(ratio, 1 - clip, 1 + clip) A))    over the steps the policy chose itself
    L_v       = 0.5 mean(max((V - R)^2, (V_old + clip(V - V_old, +-value_clip) - R)^2))
    L_v_int   = the same for the intrinsic value head
    H_k       = mean entropy of head k (six heads; sub-goal and target over manager steps)
    L_distill = cross-entropy to teacher labels: each controller head where that head is labelled,
                sub-goal and target on labelled manager steps
    L_field   = squared error between the value field and field_gamma ** (teacher's travel steps to
                the target), on manager steps where the target in force is the teacher's

Values and returns of each stream are in units of that stream's running return mean and standard
deviation. Episodes end by termination (exit or death: no bootstrap) or truncation (tick cap:
bootstrap with the value of the final observation; `bootstrap_on_truncation=False` makes the cap
terminal instead). The intrinsic stream is episodic by default.

## One loop: the student drives, the teacher labels

With `vec.teacher = "real"` every rollout worker runs the navigation teacher (`training/teacher`)
next to the agent's front end. Right after the front end has processed an observation the worker
calls `Teacher.label(obs, crop, pose, scal, step, budget, previous buttons)` (exactly the network's
inputs plus the buttons that were executed) and stores with that observation:

| Field | Meaning |
|---|---|
| `lab_heads`, `lab_head_mask` | the teacher's six head choices and which of them are labels: move, strafe, turn and USE on navigation steps; none while an enemy is visible; attack and weapon never |
| `lab_subgoal`, `lab_target`, `lab_mgr` | the manager labels (used on manager steps) |
| `lab_field`, `lab_field_ok` | on manager steps, the teacher's travel steps from every crop cell to the target |

Maps whose start is off the 64-unit grid (`start_offset`, or a `start_angle` that is not a multiple
of 90 in the manifest's options) get no labels at all: the teacher's planning grid does not fit them.

Mixing. On a step with controller labels the teacher's action is executed with probability `beta`
(`teach.beta_start` falling linearly to `teach.beta_floor` over `teach.beta_steps` environment
steps); on a manager step the same coin forces the manager's sub-goal and target to the teacher's
through the network's `forced` argument, so the value field and the controller are conditioned on
the forced choice. The buffer stores what was executed, the policy's own log-probability of it and
a `forced` flag; forced steps are left out of `L_pi` and kept in the value, distillation and field
terms. Fight steps and unlabelled maps are never mixed. The teacher's bookkeeping always follows
the executed buttons.

Log columns of the loop: `beta`, `forced` (share of steps executed by the teacher), `labelled`,
`loss_distill_heads` / `_subgoal` / `_target`, `loss_field`, `field_rows`, `agree` (share of labelled
head choices where the policy's most likely option is the teacher's), `agree_all` (all labelled
heads of a step at once), `agree_move` / `_strafe` / `_turn` / `_use`, `agree_subgoal`,
`agree_target`, `target_cells_off`, `exit_rate_labelled` / `_unlabelled`, `heldout_exit_rate`,
`distill_weight`, and the failure counters `env_resets`, `odom_errors`, `teacher_errors`.

Failures that do not stop a run: an odometry update that raises keeps the last pose estimate
(`odom_errors`); a failed held-out evaluation holds the distillation weight; and a failed engine or
worker is replaced, in three layers (`vecenv.py`):

* A row whose engine process is gone, whose step raises, or whose engine call does not return
  within `vec.step_timeout` (5 s) is recorded as cut short (a truncation with no reward, bootstrapped
  from its last observation) and moved to a freshly built map; a hung engine is killed. Each worker
  knows the process id of every engine it started and checks it before every step, because a call
  into a dead engine never returns (measured on macOS): engine calls are made from daemon threads,
  and a thread that is stuck is abandoned and replaced.
* A worker process that dies, reports an error or does not answer within `vec.worker_timeout`
  (30 s) is killed together with its engines; all its rows are cut short and a new worker starts
  them on fresh maps.
* Only when a worker cannot be restarted, or workers were replaced `5 x workers` times, does the
  run end with an error (after writing `latest.pt`).

Every restarted row counts in `env_resets` (the first 20 messages are printed) and every replaced
worker in `worker_respawns`.

    # 25-minute check on a laptop (see training/configs/unified_smoke.txt and training/logs/unified_smoke.txt)
    python -u -m training.rl.train --maps data/maps/unified_smoke_train.jsonl \
        --config training/configs/unified_smoke.json --out runs/rl/unified_smoke --max-minutes 25
    python -m training.rl.eval_alone --checkpoint runs/rl/unified_smoke/latest.pt --maps data/maps/dev_nav/manifest.jsonl

    # the full run: training/configs/LAUNCH.txt

## Running

    # toy task, a few seconds: checks the whole loop on a new machine and prints the learner device
    python -m training.rl.train --maps toy:bandit --out runs/toy --workers 2 --envs-per-worker 8 \
        --rollout 32 --total-steps 20000 --policy standin --frontend standin

    # generated maps
    python -m training.mapgen.cli --out data/maps/train --n 200 --tiers 0,1,2,3,4 --seed 0
    python -m training.rl.train --maps data/maps/train/manifest.jsonl --out runs/NAME \
        --workers 30 --envs-per-worker 16

    python -m training.rl.train --print-config            # every setting with its default
    python -m training.rl.vecenv --maps data/maps/train/manifest.jsonl --workers 3   # throughput
    python -m training.rl.bench --maps data/maps/train/manifest.jsonl --workers 4,8,10 --devices mps,cpu

    # learning smoke test: the single-room tier 1 arenas without an exit, 10 minutes of wall-clock time
    # (SIGTERM or Ctrl-C also end a run with a final checkpoint)
    python -m training.rl.select_maps --maps data/maps/train/manifest.jsonl --out data/maps/smoke_arena.jsonl \
        --tier 1 --exits 0 --max-doors 0 --max-rooms 1
    python -m training.rl.train --maps data/maps/smoke_arena.jsonl --config training/rl/configs/smoke_arena.json \
        --out runs/rl/smoke_arena --max-minutes 10

    python -m training.rl.play --maps data/maps/smoke_arena.jsonl --random                 # random-policy baseline
    python -m training.rl.play --maps data/maps/smoke_arena.jsonl --checkpoint runs/rl/smoke_arena/latest.pt
    python -m training.rl.export runs/rl/smoke_arena/latest.pt                             # -> submission/weights/policy.pt

## Settings worth knowing

* `ppo.target_kl_unlabelled` and `ppo.lr_warmup` slow the first updates of a run that starts
  from a trained checkpoint: an update ends as soon as the policy has moved by more than 1.5 times
  the limit (KL of the joint log-probability, manager choices included) on the steps it chose
  itself that carry no controller label (fights, unlabelled maps), and the learning rate rises
  linearly over the first `lr_warmup` updates. Both are off by default and off in the unified
  configs: from a combat checkpoint the first update moves those steps by KL 6, yet the fighting
  was intact four minutes later without any limit (`training/logs/unified_smoke.txt`). Log columns
  `kl_unlabelled` and `minibatches`.
* `agree` in the training log is measured on the trajectories of the run, which are driven more
  and more by the policy as `beta` falls; the teacher's labels in states the policy got itself
  into are harder to predict, so the column falls with `beta` although imitation improves. Use
  `python -m training.rl.agree --beta 1` for the number on teacher-driven states.

* `head_bias` (default `{"weapon": [2.5, -2.5]}`; `{}` turns it off) adds a starting offset to a
  head's output bias. The network starts near uniform, which means "next weapon" on half of all steps; the
  pistol is then hardly ever ready (0.3 hits per episode on the arenas against 3.9 when the key is
  never pressed), and nearly every kill of an untrained policy comes from monsters fighting each
  other. With the offset the first shots land and the kill and hit terms have something to credit.
  The bias remains an ordinary trained parameter.
* `vec.reward.hit_bonus` pays per shot that hits a monster (engine hit counter). It is off by
  default because it is not part of the score; the smoke test uses 0.2.
* `ppo.minibatch_chunks` times `ppo.epochs` sets the number of gradient steps per rollout. With
  the default 256 chunks a rollout of 96 environments gives only 2 minibatches per epoch; the
  smoke test uses 48 chunks (8 minibatches per epoch, 24 gradient steps per rollout).

## Checkpoints

`train.py` writes `ckpt_NNNNNN.pt` and `latest.pt`: a plain dict with `"format": "doomnet-policy-v1"`,
`"model"` (the network's state dict), `"policy_config"` (the constructor arguments of `PolicyNet`),
and what training needs to resume (optimiser, schedules, curriculum, return statistics, the full
config). `export.py` keeps only what the agent needs and writes it with the agent's own
`save_weights`:

    {"state_dict": state_dict, "config": {"policy_kwargs": {...PolicyNet arguments...}, "format": "doomnet-policy-v1",
                                          "source": ..., "source_sha256": ..., "iteration": ..., "env_steps": ...}}

`--init` and `load_policy` read both layouts.

### A checkpoint is tied to the code that trained it

A state dict records parameters, not the function they parameterise. If the forward pass of
`PolicyNet` or the front end (odometry, memory map, encoder) is edited without renaming a
parameter, an older checkpoint still loads with `strict=True` and then plays differently from what
was trained, with no error anywhere. So every checkpoint carries a record under `"provenance"`,
and every export the same record as nested lists under `config["provenance"]`:

| Entry | Content |
|---|---|
| `code` | sha256 of `submission/doomnet/{model,encode,memory,odometry,spec}.py` as imported by the run |
| `network` | the network's outputs on the probe episodes when the file was written: (2, 40, 31) = head logits, both values, sub-goal logits, the log-probability of four target cells and two statistics of the value field, with the manager's choices forced so that every number is continuous in the weights |
| `frontend` | (80, 16): per probe step, sums of the ray, entity and scalar inputs and the set cells per map channel that the rollout workers' front end produced from the recorded raw observations (reported by the workers when they start) |
| `probe_file` | sha256 of `training/rl/probe_episodes.npz`: 40 decision steps on each of two small maps (an arena with monsters; a door, a locked door, a key and an exit pad), raw observations and the network inputs made of them |

`load_policy` (used by `play`, `eval_alone`, `export`), `--init`, `--resume` and `export.verify`
rebuild the network, recompute both entries with the code as it is now and compare:

* network or front end differs: `provenance.CheckpointMismatch`, the file is refused. Tolerance
  5e-4 on network outputs (the same file gives differences of 6e-7 between thread counts and 0
  between MPS-trained and CPU weights; a 10 % change of one constant in the forward pass of an
  untrained network gives 0.1).
* no record (a file from before records existed): a warning on load, and `export` refuses it.
* only the source text changed and the outputs still match: one line of note.

Overrides for deliberate migrations: `--allow-mismatch` (`train`, `play`, `eval_alone`, `export`),
`--allow-unverified` (`export`), or `DOOMRL_ALLOW_MISMATCH=1` for any other caller of `load_policy`.

    python -m training.rl.provenance runs/NAME/latest.pt submission/weights/policy.pt   # OK / MISMATCH / UNVERIFIED per file
    python -m training.rl.export --check                                                # the shipped weights against the current code

`train.py` takes the source fingerprint when it starts and warns once if a source file changes
on disk during the run; a worker that is replaced after such an edit is reported, because it
imports the edited front end. Do not edit `submission/doomnet/*` while a run whose checkpoints
will be used is in progress, and run `export --check` after any edit there.

The learner device is printed at start and written to every row of `train_log.csv`, next to
environment steps per second and learner samples per second. Rollout workers always run on the CPU.

## Throughput: plan with about 1,000 environment steps per second on the laptop

Measured on the 15-core Mac while it was shared with other jobs (load average 55 to 95), learner on
MPS, real network and front end, `training/rl/configs/smoke_arena.json`:

| Quantity | 3 workers x 16 environments | 6 workers x 16 environments |
|---|---|---|
| whole loop, environment steps/s | 1,074 (mean of 204 iterations, 21:02-21:12) | 1,085 (105 iterations, 19:24-19:34) |
| collection only, steps/s | 1,735 | 2,272 |
| learner, samples/s (768 samples per minibatch, 3 epochs) | 8,515 | 6,271 |
| share of an iteration spent in the PPO update | 38 % | 52 % |
| workers alone, random actions, no network (`training.rl.vecenv`) | 1,480 | not measured |

Doubling the workers did not raise the whole-loop number: the machine has no free cores for them,
and every environment step costs three sample passes of the update (3 epochs), which the workers
sit out. The SPEC target of 5,000 steps/s on 10 cores was not reached in any measurement of the
whole loop; the workers' own ceiling at 10 workers was 4,929 steps/s in one earlier run and has
not been repeated on a quiet machine. Budget training time on this machine from 1,000 steps/s.

What limits it, measured (3 workers x 16 environments, two groups of 24 rows stepped alternately):

* One collection step of a group takes 18 to 20 ms: 11 to 12 ms in the acting forward pass on
  MPS, 2 ms copying the rows' inputs to the device, 4 ms waiting for the workers (`collect_wait_frac`
  0.16 to 0.32 in the log). So at this size the forward pass is the limit, not the workers. The
  same pass timed outside the training loop takes 4.1 ms when all rows share their manager steps
  and 6.2 ms when one row in eight runs the manager on every pass, as in a rollout. The remaining
  difference was not broken down; inside the loop the pass runs while the workers and the other
  jobs on the machine compete for cores and for the GPU.
* Rows are nearly free in a pass, so more rows per pass help: 3 workers x 32 environments
  (48 rows per pass, 15.6 ms) collected 1,594 steps/s against 1,329 for 3 x 16 in the same
  profile (+20 %). On a machine with a CUDA GPU and free cores use many rows per pass
  (`--workers` up to the free cores, `--envs-per-worker` 16 to 32).
* An acting copy of the network on CPU threads would be slower, so there is none: 11 to 14 ms
  per pass of 24 rows on the CPU (the same with 1, 2 or 4 threads) against 4 to 6 ms on MPS.
* The update costs `ppo.epochs` sample passes per environment step at 7,800 samples/s (768
  samples per minibatch; 9,300 at 1,536, 9,600 at 3,072, 10,700 at 6,144). With 3 epochs that is
  0.35 ms per environment step, 38 % of the loop at 3 workers and 52 % at 6, and it caps the
  whole loop near 2,900 steps/s however fast collection is. Larger minibatches buy at most 37 %
  and cost gradient steps per rollout (see Settings worth knowing); `epochs: 2` cuts the update's
  time by a third.

Every log row carries `update_frac`, `collect_wait_frac` and `collect_forward_frac`, and the
progress line prints the first two, so a run on another machine shows at once which part limits
it: `collect_wait_frac` near 1 means the workers do (add workers), a low value means the
learner's part of each step does (add rows per pass), `update_frac` near one half means the
update does.

Tests: `python -m pytest training/tests/test_rl.py -q`.
