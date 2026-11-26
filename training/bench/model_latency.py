"""Latency and size of the agent's networks (PolicyNet + ExitDetector), measured the way they run at evaluation.

    .venv/bin/python training/bench/model_latency.py                 # full report, about a minute
    .venv/bin/python training/bench/model_latency.py --quick         # short smoke run
    .venv/bin/python training/bench/model_latency.py --train mps     # also learner throughput at batch 4096
    .venv/bin/python training/bench/model_latency.py --train-only --train cuda cpu   # learner throughput alone

Inference numbers are batch 1, one thread (torch.set_num_threads(1)), torch.inference_mode(), with the
manager running on the first step of an episode and then every MANAGER_PERIOD steps. The budget is
BUDGET_MS mean per step for both networks together, averaged over manager and non-manager steps.
The machine is shared, so every figure is taken over several repeats: the table shows the repeat
with the lowest mean and the spread of the means says how noisy the run was.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, so the script also runs by path

from training import paths  # noqa: E402,F401  (puts submission/ on sys.path)

import torch  # noqa: E402

from doomnet import spec  # noqa: E402
from doomnet.model import ExitDetector, PolicyNet  # noqa: E402

BUDGET_MS = 5.5
EPISODE_STEPS = 525          # shortest tick cap: 2100 tics / 4


def make_inputs(batch: int, seq: int | None = None, device="cpu", seed: int = 0, uint8_map: bool = True) -> dict:
    """Random but well-formed network inputs (unit headings, sub-cell offsets in range, 0-12 visible entities)."""
    g = torch.Generator().manual_seed(seed)
    lead = (batch,) if seq is None else (batch, seq)
    hit = torch.randint(0, spec.NUM_HIT_TYPES, lead + (spec.NUM_RAYS,), generator=g)
    rays = torch.cat([torch.rand(lead + (spec.NUM_RAYS, 1), generator=g),
                      torch.nn.functional.one_hot(hit, spec.NUM_HIT_TYPES).float()], -1)
    count = torch.randint(0, 13, lead + (1,), generator=g)
    ent_mask = torch.arange(spec.MAX_ENTITIES).expand(lead + (spec.MAX_ENTITIES,)) < count
    ents = torch.randn(lead + (spec.MAX_ENTITIES, spec.ENT_FEATS), generator=g) * ent_mask[..., None]
    scal = torch.rand(lead + (spec.SCALAR_DIM,), generator=g)
    theta = torch.rand(lead, generator=g) * 2 * torch.pi
    idx = spec.SCALAR_INDEX
    scal[..., idx["sin_theta"]], scal[..., idx["cos_theta"]] = torch.sin(theta), torch.cos(theta)
    scal[..., idx["subcell_x"]] = torch.rand(lead, generator=g) - 0.5
    scal[..., idx["subcell_y"]] = torch.rand(lead, generator=g) - 0.5
    scal[..., idx["vel_forward"]] = torch.rand(lead, generator=g) * 0.8
    scal[..., idx["vel_right"]] = 0.0
    crop = torch.rand(lead + (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), generator=g) < 0.3
    crop = (crop * 255).to(torch.uint8) if uint8_map else crop.float()
    out = {"rays": rays, "ents": ents, "ent_mask": ent_mask, "scal": scal, "map": crop,
           "masks": torch.rand(lead + (2,), generator=g) < 0.5,
           "rgb": torch.randint(0, 256, lead + (3, spec.SCREEN_H, spec.SCREEN_W), generator=g, dtype=torch.uint8)}
    return {k: v.to(device) for k, v in out.items()}


def count_parameters(net: PolicyNet, det: ExitDetector) -> dict:
    """Parameters per component."""
    groups = {
        "rays (circular conv)": [net.ray_net],
        "entities (attention)": [net.ent_fc, net.ent_query],
        "scalars + fusion": [net.scal_fc, net.fuse],
        "map tower": [net.tower],
        "manager heads": [net.manager_fc, net.subgoal_head, net.subgoal_emb, net.pointer_fc, net.pointer_out],
        "value propagation": [net.vp],
        "goal read + GRU": [net.goal_fc, net.gru] + ([] if net.strict else [net.local_fc]),
        "action heads": [net.actor],
        "value heads": [net.critic],
    }
    counts = {name: sum(p.numel() for m in mods for p in m.parameters()) for name, mods in groups.items()}
    counts["PolicyNet total"] = sum(p.numel() for p in net.parameters())
    assert counts["PolicyNet total"] == sum(v for k, v in counts.items() if k != "PolicyNet total")
    counts["ExitDetector"] = sum(p.numel() for p in det.parameters())
    counts["both networks"] = counts["PolicyNet total"] + counts["ExitDetector"]
    return counts


def _stats(ms: list[float]) -> dict:
    t = torch.tensor(ms, dtype=torch.float64)
    return {"mean": float(t.mean()), "p95": float(t.quantile(0.95)), "max": float(t.max()), "n": len(ms)}


def time_steps(net: PolicyNet, det: ExitDetector, steps: int, pool: int = 32) -> dict:
    """One run of `steps` decision steps: the exit detector, then the policy, as the agent calls them."""
    data = [make_inputs(1, seed=s) for s in range(pool)]
    clock = time.perf_counter
    total, manager, other, det_ms, pol_ms = [], [], [], [], []
    with torch.inference_mode():
        state = net.initial_state(1, "cpu")
        for i in range(60):                                        # warm-up, covering manager steps
            d = data[i % pool]
            det(d["rgb"])
            _, state = net(d["rays"], d["ents"], d["ent_mask"], d["scal"], d["map"], state, d["masks"])
        for i in range(steps):
            if i % EPISODE_STEPS == 0:
                state = net.initial_state(1, "cpu")                # new episode: the manager runs on its first step
            d = data[i % pool]
            is_manager = bool(state["since"][0] >= spec.MANAGER_PERIOD)
            t0 = clock()
            det(d["rgb"])
            t1 = clock()
            _, state = net(d["rays"], d["ents"], d["ent_mask"], d["scal"], d["map"], state, d["masks"])
            t2 = clock()
            total.append((t2 - t0) * 1e3)
            det_ms.append((t1 - t0) * 1e3)
            pol_ms.append((t2 - t1) * 1e3)
            (manager if is_manager else other).append((t2 - t1) * 1e3)
    return {"step (both networks)": _stats(total), "ExitDetector": _stats(det_ms), "PolicyNet, all steps": _stats(pol_ms),
            "PolicyNet, manager steps": _stats(manager), "PolicyNet, other steps": _stats(other)}


def repeat_steps(net, det, steps: int, repeats: int) -> tuple[dict, list[float]]:
    """Best repeat (lowest mean step time) and the mean of every repeat."""
    runs = [time_steps(net, det, steps) for _ in range(repeats)]
    means = [r["step (both networks)"]["mean"] for r in runs]
    return runs[means.index(min(means))], means


def _best_us(fn, loops: int, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(max(10, loops // 10)):
            fn()
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(loops):
                fn()
            best = min(best, (time.perf_counter() - t0) / loops)
    return best * 1e6


def time_components(net: PolicyNet, det: ExitDetector, loops: int, repeats: int) -> dict:
    """Each stage timed on its own (microseconds, lowest of `repeats` loops). Stages marked * run on manager steps only."""
    d = make_inputs(1, seed=1)
    with torch.inference_mode():
        state = net.initial_state(1, "cpu")
        out, state = net(d["rays"], d["ents"], d["ent_mask"], d["scal"], d["map"], state, d["masks"])
        scal = d["scal"]
        fused, _ = net._perceive(d["rays"], d["ents"], d["ent_mask"], scal)
        feat, vec = net.tower(d["map"].float() / 255)
        ctx, _ = net._subgoal_logits(fused, state["h"], vec)
        emb = net.subgoal_emb(out["subgoal"])
        goal = net._goal_features(out["subgoal"], out["value_field"], state["cell"], scal, d["map"], vec)
        x = torch.cat([fused, goal], 1)

    def manager_heads():
        c, sl = net._subgoal_logits(fused, state["h"], vec)
        sg = net._choose(sl, None, True)
        net._choose(net._target_logits(feat, c, net.subgoal_emb(sg)), None, True)

    return {
        "ExitDetector": _best_us(lambda: det(d["rgb"]), loops, repeats),
        "sensors: rays, entities, scalars, fusion": _best_us(lambda: net._perceive(d["rays"], d["ents"], d["ent_mask"], scal), loops, repeats),
        "schedule and cell counter": _best_us(lambda: net._advance(scal, state["since"], state["cell"], state["sub"], state["move"], None), loops, repeats),
        "goal read: value-field patch + sub-goal embedding": _best_us(
            lambda: net._goal_features(out["subgoal"], out["value_field"], state["cell"], scal, d["map"], vec), loops, repeats),
        "GRU": _best_us(lambda: net.gru(x, state["h"]), loops, repeats),
        "action and value heads (with masks)": _best_us(lambda: net._heads(state["h"], goal, vec, d["masks"]), loops, repeats),
        "* map tower": _best_us(lambda: net.tower(d["map"].float() * (1 / 255)), loops, repeats),
        "* manager heads: sub-goal, pointer, two samples": _best_us(manager_heads, loops, repeats),
        f"* value propagation (K={net.vp.iters})": _best_us(lambda: net.vp(feat, emb, out["target"]), loops, repeats),
    }


def train_throughput(device: str, batch: int = 256, seq: int = 16, iters: int = 6, threads: int = 3, **net_kwargs) -> dict:
    """Learner-side cost: forward_sequence with stored manager choices, loss, backward, at batch*seq samples."""
    torch.set_num_threads(threads)
    dev = torch.device(device)
    net = PolicyNet(**net_kwargs).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    d = make_inputs(batch, seq, device=dev, seed=3)
    forced = {"subgoal": torch.randint(0, spec.NUM_SUBGOALS, (batch, seq), device=dev),
              "target": torch.randint(0, spec.CROP_SIZE ** 2, (batch, seq), device=dev)}
    labels = [torch.randint(0, n, (batch * seq,), device=dev) for n in spec.HEAD_SIZES]
    state = net.initial_state(batch, dev)

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    def one_update():
        out, _ = net.forward_sequence(d["rays"], d["ents"], d["ent_mask"], d["scal"], d["map"], state, forced=forced)
        loss = sum(torch.nn.functional.cross_entropy(l.flatten(0, 1), y) for l, y in zip(out["logits"], labels))
        loss = loss + out["value"].pow(2).mean() + out["value_int"].pow(2).mean()
        run = out["target_logits"].abs().sum(-1) > 0
        loss = loss + (torch.logsumexp(out["target_logits"], -1) * run).sum() / run.sum().clamp(min=1)
        loss = loss + (torch.logsumexp(out["subgoal_logits"], -1) * run).sum() / run.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss)

    one_update()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one_update()
        sync()
        times.append(time.perf_counter() - t0)
    torch.set_num_threads(1)
    best = min(times)
    return {"device": device, "samples": batch * seq, "seconds_per_update": best, "samples_per_second": batch * seq / best,
            "threads": threads if dev.type == "cpu" else None}


def inference_report(net: PolicyNet, det: ExitDetector, args, report: dict) -> None:
    """Step latency, per-component times and the variants: prints them and fills `report`."""
    best, means = repeat_steps(net, det, args.steps, args.repeats)
    report["steps"], report["repeat_means_ms"] = best, means
    print(f"\nStep latency in ms ({args.steps} steps per repeat, {args.repeats} repeats, best repeat shown)")
    print(f"  {'':28s} {'mean':>8s} {'p95':>8s} {'max':>8s} {'steps':>7s}")
    for name, s in best.items():
        print(f"  {name:28s} {s['mean']:8.3f} {s['p95']:8.3f} {s['max']:8.3f} {s['n']:7d}")
    mean = best["step (both networks)"]["mean"]
    print(f"  mean over repeats: lowest {min(means):.3f}, median {sorted(means)[len(means) // 2]:.3f}, highest {max(means):.3f} ms "
          f"(spread {100 * (max(means) - min(means)) / min(means):.0f}%)")
    print(f"  budget {BUDGET_MS} ms mean per step: {'PASS' if mean <= BUDGET_MS else 'FAIL'} ({100 * mean / BUDGET_MS:.0f}% used)")

    report["components_us"] = time_components(net, det, 300 if args.quick else 2000, 3 if args.quick else 5)
    print("\nPer-component time in microseconds (each stage alone; * = manager steps only)")
    for name, us in report["components_us"].items():
        print(f"  {name:52s} {us:9.1f}")

    report["variants"] = {}
    variants = [(f"K={k}", dict(vp_iters=k)) for k in args.k] + [("loosened bottleneck", dict(strict_goal_bottleneck=False))]
    print("\nVariants (best repeat, ms; last column: the value propagation alone, lowest of several loops)")
    print(f"  {'':22s} {'step mean':>10s} {'step p95':>10s} {'step max':>10s} {'manager step':>13s} {'other step':>11s} {'propagation':>12s}")
    feat, emb, target = torch.randn(1, 32, 32, 32), torch.randn(1, PolicyNet.EMB_DIM), torch.tensor([2080])
    for name, kwargs in variants:
        torch.manual_seed(0)
        variant = PolicyNet(**kwargs).eval()
        b, _ = repeat_steps(variant, det, max(args.steps // 2, 420), max(args.repeats // 2, 2))
        row = {"step_mean": b["step (both networks)"]["mean"], "step_p95": b["step (both networks)"]["p95"],
               "step_max": b["step (both networks)"]["max"], "manager_step": b["PolicyNet, manager steps"]["mean"],
               "other_step": b["PolicyNet, other steps"]["mean"],
               "propagation": _best_us(lambda: variant.vp(feat, emb, target), 300 if args.quick else 1000, 5) / 1e3}
        report["variants"][name] = row
        print(f"  {name:22s} {row['step_mean']:10.3f} {row['step_p95']:10.3f} {row['step_max']:10.3f} {row['manager_step']:13.3f} "
              f"{row['other_step']:11.3f} {row['propagation']:12.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2100, help="decision steps per repeat (default: four shortest episodes)")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--k", type=int, nargs="*", default=[1, 4, 8, 12], help="value-propagation iterations to compare")
    ap.add_argument("--train", nargs="*", default=[], metavar="DEVICE", help="also measure learner throughput on these devices (cpu, mps, cuda)")
    ap.add_argument("--train-only", action="store_true", help="skip the inference measurements")
    ap.add_argument("--quick", action="store_true", help="short run for a smoke check")
    ap.add_argument("--json", type=Path, default=None, help="also write every number to this file")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.repeats = 420, 2

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    net, det = PolicyNet().eval(), ExitDetector().eval()
    report: dict = {"machine": platform.processor() or platform.machine(), "torch": torch.__version__,
                    "threads": torch.get_num_threads(), "budget_ms": BUDGET_MS, "config": net.config}
    print(f"torch {torch.__version__}, {report['machine']}, {torch.get_num_threads()} thread, batch 1, inference mode")
    print(f"PolicyNet config: {net.config}")

    print("\nParameters")
    report["parameters"] = count_parameters(net, det)
    for name, n in report["parameters"].items():
        print(f"  {name:28s} {n:>10,d}")

    if not args.train_only:
        inference_report(net, det, args, report)

    if args.train:
        print("\nLearner throughput: forward_sequence + backward + Adam step, 256 sequences x 16 steps = 4096 samples")
        report["train"] = []
        for device in args.train:
            r = train_throughput(device)
            report["train"].append(r)
            note = f" ({r['threads']} threads)" if r["threads"] else ""
            print(f"  {device:6s} {r['seconds_per_update'] * 1e3:9.1f} ms per update  {r['samples_per_second']:10,.0f} samples/s{note}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
