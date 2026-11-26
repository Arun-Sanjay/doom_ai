"""Play whole episodes with a checkpoint or with the uniform-random policy and print the means.

    python -m training.rl.play --maps MANIFEST --random                       # random-policy baseline
    python -m training.rl.play --maps MANIFEST --checkpoint runs/NAME/latest.pt
           [--tiers 1] [--episodes-per-row 4] [--workers 3 --envs-per-worker 16] [--no-intrinsic]

Training-time scaffolding, never imported by the agent.

Both policies act through the same vector environment, masks and sampling as training, so their
numbers are directly comparable with each other and with the ``kills`` column of the training log.
The first ``--episodes-per-row`` episodes of every environment row are counted.
"""

from __future__ import annotations

import argparse
import json

from training import paths  # noqa: F401
from training.rl import deps
from training.rl.intrinsic import IntrinsicCoef
from training.rl.vecenv import VecConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean kills, exits, deaths and cells observed per episode of a policy")
    ap.add_argument("--maps", required=True, help="manifest.jsonl, a WAD or a directory of WADs")
    ap.add_argument("--tiers", help="comma-separated tiers to keep, e.g. 1 or 0,1")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--checkpoint", help="checkpoint written by training.rl.train")
    who.add_argument("--random", action="store_true", help="uniform over every head's allowed options")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--episodes-per-row", type=int, default=2)
    ap.add_argument("--episodes-per-map", type=int, default=1, help="episodes before a row moves to another map")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-intrinsic", action="store_true", help="switch the curiosity bookkeeping's reward off")
    ap.add_argument("--json", help="also write the summary to this file")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="play a checkpoint although the current code no longer computes what it recorded")
    args = ap.parse_args()

    import torch
    from training.rl.rollout import play_episodes, summarise_episodes
    from training.rl.train import load_policy, pick_device

    entries = deps.load_manifest(args.maps)
    if args.tiers:
        keep = {int(t) for t in args.tiers.split(",")}
        entries = [e for e in entries if e.tier in keep]
    cfg = VecConfig(episodes_per_map=args.episodes_per_map, intrinsic=IntrinsicCoef(enabled=not args.no_intrinsic))
    policy, device = None, torch.device("cpu")
    if args.checkpoint:
        device = pick_device(args.device)
        policy, _ = load_policy(args.checkpoint, device, check="warn" if args.allow_mismatch else "strict")
        policy.eval()
    results = play_episodes(policy, entries, device, args.workers, args.envs_per_worker, cfg, args.seed, args.episodes_per_row)
    out = summarise_episodes(results)
    out["policy"] = args.checkpoint or "uniform random"
    out["maps_available"] = len(entries)
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
