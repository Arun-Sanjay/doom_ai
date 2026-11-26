"""Agreement of a checkpoint with the navigation teacher on labelled steps.

    python -m training.rl.agree --checkpoint runs/NAME/latest.pt --maps data/maps/dev_nav/manifest.jsonl \
        [--tiers 2] [--beta 1.0] [--steps 256] [--workers 6 --envs-per-worker 10] [--json out.json]

Training-time scaffolding, never imported by the agent.

The same collector as in training plays ``--steps`` steps in every environment with the teacher's
action executed with probability ``--beta`` (1.0: the teacher drives and the states are the ones a
perfect imitator would visit; 0.0: the policy drives and the states are its own). Nothing is
learned. Reported, over the labelled steps: the share of labelled head choices on which the
policy's most likely option is the teacher's (``agree``, and per head), the share of steps on which
every labelled head agrees (``agree_all``), sub-goal and target agreement on manager steps, and the
cross-entropies of the distillation term.

It also measures how much the controller depends on the manager: the same steps are replayed with
the manager's target replaced by its mirror image through the crop centre (``agree_mirrored_target``)
and with another sub-goal (``agree_other_subgoal``). A controller that follows the manager loses
agreement with the teacher under both; one that steers by its sensors alone does not notice.
"""

from __future__ import annotations

import argparse
import json

import torch

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import deps
from training.rl.buffer import RolloutBuffer
from training.rl.ppo import PPO, PPOConfig, ppo_loss, unroll
from training.rl.rollout import Collector
from training.rl.vecenv import VecConfig, VecEnv

KEYS = ("agree", "agree_all", "agree_move", "agree_strafe", "agree_turn", "agree_use", "agree_subgoal", "agree_target",
        "target_cells_off", "loss_distill_heads", "loss_distill_subgoal", "loss_distill_target", "loss_field", "labelled", "forced")


@torch.no_grad()
def measure(policy, entries: list, device, workers: int = 6, envs_per_worker: int = 10, steps: int = 256, beta: float = 1.0,
            seed: int = 0, vec_cfg: VecConfig | None = None) -> dict:
    """Play ``steps`` steps per environment and compare the policy with the teacher's labels (see the module docstring)."""
    cfg = PPOConfig()
    steps = max(cfg.chunk, steps // cfg.chunk * cfg.chunk)
    vec_cfg = vec_cfg or VecConfig(teacher="real", seed=seed)
    with VecEnv(entries, workers, envs_per_worker, vec_cfg) as vec:
        buf = RolloutBuffer(steps, vec.n, cfg.chunk, device)
        ppo = PPO(policy, cfg, device, seed=seed)
        collector = Collector(vec, policy, buf, device, seed, False)
        collector.beta = beta
        episodes = [info for info in collector.collect() if not info.get("crashed")]
        buf.finish(cfg.gamma, cfg.lam, cfg.gamma_int, 0.0)
        sums, weights = {}, {}
        for batch in buf.minibatches(max(buf.num_chunks() // 8, 1), torch.Generator().manual_seed(seed)):
            _, stats = ppo_loss(unroll(policy, ppo.layout, batch), batch, cfg, 1.0, 1.0)
            n = batch["step"].numel()
            for k in KEYS:
                if k in stats:
                    sums[k] = sums.get(k, 0.0) + stats[k] * n
                    weights[k] = weights.get(k, 0) + n
            # The same steps with the manager's choice replaced: does the controller notice?
            size, target = spec.CROP_SIZE, batch["target"]
            mirrored = (size - 1 - target // size) * size + (size - 1 - target % size)
            swapped = {"agree_mirrored_target": dict(batch, target=mirrored),
                       "agree_other_subgoal": dict(batch, subgoal=(batch["subgoal"] + spec.NUM_SUBGOALS // 2) % spec.NUM_SUBGOALS)}
            for name, other in swapped.items():
                _, changed = ppo_loss(unroll(policy, ppo.layout, other), other, cfg, 1.0, 1.0)
                if "agree" in changed:
                    sums[name] = sums.get(name, 0.0) + changed["agree"] * n
                    weights[name] = weights.get(name, 0) + n
    out = {k: sums[k] / weights[k] for k in sums}
    out.update(beta=beta, steps=steps * vec.n, episodes_finished=len(episodes),
               exit_rate_of_finished=(sum(e["exit"] for e in episodes) / len(episodes)) if episodes else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Agreement of a checkpoint with the teacher's labels")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--tiers", help="comma-separated tiers to keep")
    ap.add_argument("--beta", type=float, default=1.0, help="probability that the teacher's action is executed (1 = teacher drives)")
    ap.add_argument("--steps", type=int, default=256, help="steps per environment")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--envs-per-worker", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the result to this file")
    args = ap.parse_args()

    from training.rl.train import load_policy, pick_device

    entries = deps.load_manifest(args.maps)
    if args.tiers:
        keep = {int(t) for t in args.tiers.split(",")}
        entries = [e for e in entries if e.tier in keep]
    device = pick_device(args.device)
    policy, payload = load_policy(args.checkpoint, device)
    policy.eval()
    torch.manual_seed(args.seed)
    out = measure(policy, entries, device, args.workers, args.envs_per_worker, args.steps, args.beta, args.seed)
    out.update(checkpoint=args.checkpoint, env_steps=int(payload.get("env_steps", 0)), suite=args.maps)
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
