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
| `standins.py`, `standin_model.py` | Small stand-ins and toy environments used by the tests. |

## Policy and loss

One decision is six categorical heads (move, strafe, turn, attack, use, weapon). Every
`MANAGER_PERIOD` steps, starting at the episode start, the manager also chooses a sub-goal and a
target cell, which are held in the recurrent state until the next manager step.

    log pi(a|s) = sum_h log pi_h(a_h|s) + [manager step] (log pi_sub(g|s) + log pi_tgt(c|s))

USE and ATTACK are masked from observation fields (`doomnet.spec.action_masks`): the "on" logit is
-inf where the mask is False, in the rollout and in the loss alike.

    L = L_pi + vf_coef L_v + vf_int_coef L_v_int - ent_scale sum_k ent_coef_k H_k + w_distill L_distill

    ratio     = exp(log pi_new - log pi_old)
    A         = normalised (A_ext + int_coef A_int)             two GAE streams
    L_pi      = -mean(min(ratio A, clip(ratio, 1 - clip, 1 + clip) A))
    L_v       = 0.5 mean(max((V - R)^2, (V_old + clip(V - V_old, +-value_clip) - R)^2))
    L_v_int   = the same for the intrinsic value head
    H_k       = mean entropy of head k (six heads; sub-goal and target over manager steps)
    L_distill = cross-entropy to teacher labels (six heads, sub-goal, target) where a label mask is set

Values and returns of each stream are in units of that stream's running return mean and standard
deviation. Episodes end by termination (exit or death: no bootstrap) or truncation (tick cap:
bootstrap with the value of the final observation; `bootstrap_on_truncation=False` makes the cap
terminal instead). The intrinsic stream is episodic by default.

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

The learner device is printed at start and written to every row of `train_log.csv`, next to
environment steps per second and learner samples per second. Rollout workers always run on the CPU.

Tests: `python -m pytest training/tests/test_rl.py -q`.
