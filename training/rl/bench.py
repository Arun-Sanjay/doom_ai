"""Throughput of the two halves of training: rollout workers and the learner.

    python -m training.rl.bench --maps data/maps/train/manifest.jsonl --workers 4,8,10 --devices mps,cpu

Training-time scaffolding, never imported by the agent.

* **Environment steps per second**: ``vecenv.benchmark`` with random actions and no network, the
  real odometry / memory / encoder running in the workers. This is the workers' ceiling.
* **Learner samples per second**: full PPO updates (gather, sequence-mode forward, loss, backward,
  Adam step) of the real network on a synthetic rollout at the configured minibatch size, on each
  device asked for. One sample is one environment step of one chunk; with ``epochs`` passes per
  rollout the learner must process ``epochs`` samples for every environment step collected.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import deps
from training.rl.buffer import RolloutBuffer, StateLayout
from training.rl.ppo import PPO, PPOConfig
from training.rl.vecenv import OBS_FIELDS, VecConfig, benchmark


def synthetic_rollout(policy, steps: int, n_envs: int, chunk: int, device, seed: int = 0) -> RolloutBuffer:
    """A rollout buffer filled with random but valid data (episodes of 128 steps, nothing masked)."""
    g = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer(steps, n_envs, chunk, device)
    layout = StateLayout(policy, device)
    rows = slice(0, n_envs)
    for t in range(steps):
        arrays = {}
        for k, (shape, dtype) in OBS_FIELDS.items():
            x = torch.rand((n_envs, *shape), generator=g)
            if k == "map":
                x = (x < 0.2).to(torch.uint8) * 255
            elif k in ("ent_mask", "lab_mask"):
                x = x < (0.2 if k == "ent_mask" else 0.0)
            elif k == "masks":
                x = torch.ones((n_envs, *shape), dtype=torch.bool)
            elif k == "step":
                x = torch.full((n_envs,), t % 128, dtype=torch.int32)
            elif k.startswith("lab_"):
                x = torch.zeros((n_envs, *shape))
            arrays[k] = x.numpy().astype(dtype)
        buf.put_obs(t, rows, arrays)
        if t % chunk == 0:
            buf.put_state(t, rows, layout.to_rows(policy.initial_state(n_envs, device)))
        heads = torch.stack([torch.randint(0, n, (n_envs,), generator=g) for n in spec.HEAD_SIZES], -1)
        buf.put_action(t, rows, heads, torch.randint(0, spec.NUM_SUBGOALS, (n_envs,), generator=g),
                       torch.randint(0, spec.CROP_SIZE ** 2, (n_envs,), generator=g),
                       -6.0 + torch.rand(n_envs, generator=g), torch.randn(n_envs, generator=g), torch.randn(n_envs, generator=g))
        done = torch.full((n_envs,), t % 128 == 127)
        buf.put_outcome(t, rows, torch.randn(n_envs, generator=g) * 0.1, torch.rand(n_envs, generator=g) * 0.01,
                        done, torch.zeros(n_envs, dtype=torch.bool), torch.zeros(n_envs), torch.zeros(n_envs))
    return buf


def learner_benchmark(device: str, cfg: PPOConfig | None = None, steps: int = 64, n_envs: int = 128, updates: int = 3,
                      threads: int = 0, policy_kwargs: dict | None = None) -> dict:
    """Samples per second of whole PPO updates of the real network on ``device``."""
    cfg = cfg or PPOConfig()
    dev = torch.device(device)
    if threads > 0:
        torch.set_num_threads(threads)
    torch.manual_seed(0)
    policy = deps.make_policy("real", **(policy_kwargs or {})).to(dev)
    buf = synthetic_rollout(policy, steps, n_envs, cfg.chunk, dev)
    ppo = PPO(policy, cfg, dev, total_updates=100)
    ppo.update(buf)                                        # warm-up: allocations, kernel compilation
    samples, start = 0, time.perf_counter()
    for _ in range(updates):
        samples += ppo.update(buf)["samples"]
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"device": device, "threads": torch.get_num_threads() if dev.type == "cpu" else 0,
            "minibatch_samples": cfg.minibatch_chunks * cfg.chunk, "samples_per_s": samples / elapsed,
            "ms_per_minibatch": 1000.0 * elapsed / (updates * cfg.epochs * -(-buf.num_chunks() // cfg.minibatch_chunks))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Environment steps/s of the workers and learner samples/s per device")
    ap.add_argument("--maps", help="manifest for the environment benchmark (omit to skip it)")
    ap.add_argument("--workers", default="", help="comma-separated worker counts, e.g. 4,8,10")
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200, help="measured steps per environment row")
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--devices", default="", help="comma-separated learner devices, e.g. mps,cpu or cuda")
    ap.add_argument("--minibatch-chunks", type=int, default=PPOConfig().minibatch_chunks)
    ap.add_argument("--cpu-threads", type=int, default=0, help="torch threads for the CPU learner (0 = torch default)")
    ap.add_argument("--updates", type=int, default=3, help="measured PPO updates per device (after one warm-up update)")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()
    results = {"env": [], "learner": []}
    if args.maps and args.workers:
        entries = deps.load_manifest(args.maps)
        for w in (int(x) for x in args.workers.split(",")):
            out = benchmark(entries, w, args.envs_per_worker, args.steps, cfg=VecConfig(frontend="real"), groups=args.groups)
            results["env"].append(out)
            print(f"workers {w:3d} x {args.envs_per_worker} envs: {out['steps_per_s']:8.0f} env steps/s "
                  f"({out['per_worker']:.0f} per worker), front end {out['frontend']}, {out['episodes']} episodes", flush=True)
    for device in (d for d in args.devices.split(",") if d):
        out = learner_benchmark(device, PPOConfig(minibatch_chunks=args.minibatch_chunks), updates=args.updates, threads=args.cpu_threads)
        results["learner"].append(out)
        print(f"learner on {device}: {out['samples_per_s']:8.0f} samples/s at {out['minibatch_samples']} samples per minibatch "
              f"({out['ms_per_minibatch']:.0f} ms each)" + (f", {out['threads']} threads" if device == "cpu" else ""), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
