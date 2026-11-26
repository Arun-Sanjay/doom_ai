"""Exit rate of a checkpoint driving alone, with every map of a manifest played the same number of times.

    python -m training.rl.eval_alone --checkpoint runs/NAME/latest.pt --maps data/maps/dev_nav/manifest.jsonl \
        [--tiers 2] [--episodes 1] [--workers 6] [--seed 0] [--json out.json]

Training-time scaffolding, never imported by the agent.

No teacher and no mixing: the network chooses every action. Decoding and budget follow the agent
at evaluation: USE and weapon by most likely option, the other heads and the manager's choices
sampled, the assumed time budget fixed at its evaluation default. Row r of the vector environment
plays map r for ``--episodes`` episodes (a new engine seed each time), so every map counts equally.
The exit observation is the training one (ground truth with the detector-like noise model).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import deps
from training.rl.vecenv import VecConfig


AGENT_GREEDY = (spec.HEAD_USE, spec.HEAD_WEAPON)       # the agent's decoding (submission/agent.py GREEDY_HEADS)


def evaluate(policy, entries: list, device, workers: int = 6, episodes: int = 1, seed: int = 0, vec_cfg: VecConfig | None = None,
             greedy_heads: tuple = AGENT_GREEDY) -> dict:
    """Play ``episodes`` episodes of every map with ``policy`` alone. Returns totals and per-tier exit rates.

    ``greedy_heads``: heads decoded by most likely option (the others are sampled); the default is the agent's decoding.
    """
    from training.rl.rollout import play_episodes

    n_workers = max(1, min(workers, len(entries)))
    per_worker = math.ceil(len(entries) / n_workers)
    cfg = dataclasses.replace(vec_cfg or VecConfig(), pin_maps=True, p_default_budget=1.0, teacher="none", episodes_per_map=10**9)
    started = time.perf_counter()
    results = play_episodes(policy, entries, device, workers and n_workers, per_worker, cfg, seed, episodes,
                            tuple(greedy_heads), max_steps=episodes * 1100)
    results = [r for r in results if r["row"] < len(entries)]          # rows beyond the last map repeat the first ones
    tiers = {}
    for tier in sorted({r["tier"] for r in results}):
        part = [r for r in results if r["tier"] == tier and r.get("has_exit", True)]
        tiers[str(tier)] = {"episodes": len(part), "exits": int(sum(r["exit"] for r in part)),
                            "exit_rate": float(np.mean([r["exit"] for r in part])) if part else None,
                            "timed_out": int(sum(r["truncated"] for r in part)), "dead": int(sum(r["dead"] for r in part)),
                            "mean_steps": float(np.mean([r["len"] for r in part])) if part else None,
                            "door_reversals": float(np.mean([r.get("reversals", 0) for r in part])) if part else None}
    with_exit = [r for r in results if r.get("has_exit", True)]
    return {"maps": len({r["map"] for r in results}), "episodes": len(with_exit), "exits": int(sum(r["exit"] for r in with_exit)),
            "exit_rate": float(np.mean([r["exit"] for r in with_exit])) if with_exit else None, "tiers": tiers,
            "expected_episodes": len(entries) * episodes, "seconds": time.perf_counter() - started,
            "exit_by_map": {str(m): [int(r["exit"]) for r in with_exit if r["map"] == m] for m in sorted({r["map"] for r in with_exit})}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Exit rate of a checkpoint driving alone, every map played equally often")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--tiers", help="comma-separated tiers to keep")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per map")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the result to this file")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="play a checkpoint although the current code no longer computes what it recorded")
    ap.add_argument("--greedy", default="use,weapon",
                    help="heads decoded by most likely option, e.g. move,strafe,turn,use,weapon (default: the agent's decoding)")
    args = ap.parse_args()

    import torch
    from training.rl.train import load_policy, pick_device

    entries = deps.load_manifest(args.maps)
    if args.tiers:
        keep = {int(t) for t in args.tiers.split(",")}
        entries = [e for e in entries if e.tier in keep]
    entries = [dataclasses.replace(e, index=i) for i, e in enumerate(entries)]
    device = pick_device(args.device)
    policy, payload = load_policy(args.checkpoint, device, check="warn" if args.allow_mismatch else "strict")
    policy.eval()
    torch.manual_seed(args.seed)
    greedy = tuple(spec.HEAD_NAMES.index(name) for name in args.greedy.split(",") if name)
    out = evaluate(policy, entries, device, args.workers, args.episodes, args.seed, greedy_heads=greedy)
    out.update(greedy_heads=args.greedy, checkpoint=args.checkpoint, env_steps=int(payload.get("env_steps", 0)), suite=args.maps, seed=args.seed)
    by_map = out.pop("exit_by_map")
    print(json.dumps(out, indent=1))
    out["exit_by_map"] = by_map
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
