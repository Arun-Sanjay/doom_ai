"""Exit detector on whole episodes: what ends up in the agent's exit map.

Training-time scaffolding. Never imported by the agent in submission/.

    python -m training.exitdet.episodes --suite training/suites/dev_v1.jsonl --workers 3

The frame sets of `collect.py` are built from teleported poses with look-alikes made common on
purpose. This module measures the same detector the way the agent uses it: every map of a frozen
suite is played from its real start by a scripted wanderer (the collector's `Wander`: walk where
the range rays show room, open doors, shoot now and then), the detector runs on every frame at
batch 1, and each sighting (confidence >= 0.5) is written into a 64-unit grid exactly as the
spatial memory does it (0.5 x confidence per sighting, capped at 1), using the true pose so that
the result measures the detector and not the odometry.

Per frame: precision, recall by distance band and false detections per 1,000 negative frames, as
in `evaluate.py`, but on the frame mix of real play.

Per episode:
    pad_in_view        the label called a pad visible within 400 units on at least 3 steps
    found              ... and the strongest cell of the exit map is a pad cell or touches one
    steps_to_find      steps from the first such step to the first sighting placed on the pad
    false_cells        cells with any belief (one sighting adds 0.25 to 0.5) that neither are a pad
                       cell nor touch one
    false_strong       the same with belief >= 0.75 (two or more confident sightings on one cell)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

from training import paths
from training.exit_label import exit_observation
from training.exitdet import collect as C
from training.exitdet.data import TAGS, has_tag

OUT_DIR = paths.ROOT / "data" / "exitdet" / "episodes"
LOG_CSV = paths.ROOT / "training" / "logs" / "exit_detector_episodes.csv"
CELL = 64.0
EXIT_GAIN = 0.5                    # belief added per sighting x confidence (doomnet.memory.EXIT_GAIN)
MIN_CONFIDENCE = 0.5               # sightings below this are ignored (doomnet.memory.EXIT_MIN_CONFIDENCE)
NEAR = 400.0
STRONG = 0.75                      # belief that takes two or more confident sightings on one cell


def play(job: dict) -> dict:
    """Worker entry: one episode. Never raises: an error is reported in the result."""
    try:
        return _play(job)
    except Exception:  # noqa: BLE001 - one broken map must not stop the run
        return {"name": job["name"], "error": traceback.format_exc()}


def _play(job: dict) -> dict:
    import torch
    from doom_env import make_env
    from training.exitdet.evaluate import load_detector
    from training.mapgen import udmf
    from training.mapmeta import load_meta

    torch.set_num_threads(1)
    model, _ = load_detector(job["weights"])
    wad = udmf.write_wad(job["grid"], job["wad"], **job["options"])
    meta = load_meta(wad)
    env = make_env(str(wad), episode_ticks=int(job["episode_ticks"]))
    rng = np.random.default_rng(job["seed"])
    rec = {k: [] for k in ("logit", "pred", "visible", "ambiguous", "pos", "tags", "pose")}
    kept: list = []
    try:
        obs, _ = env.reset(seed=int(job["seed"]))
        scene = C.Scene(env, meta, job)
        wander = C.Wander(rng)
        st = env.privileged_state()
        health, flash, fired = st["health"], 0.0, False
        ended = "time"
        for _ in range(int(job["episode_ticks"]) // 4 + 2):
            with torch.inference_mode():
                out = model(torch.from_numpy(np.ascontiguousarray(obs["rgb"])).permute(2, 0, 1)[None])[0].numpy()
            visible, x_fwd, y_left, ambiguous = exit_observation(env, meta)
            rec["logit"].append(float(out[0]))
            rec["pred"].append((float(out[1]) * 1024.0, float(out[2]) * 1024.0))
            rec["visible"].append(visible)
            rec["ambiguous"].append(ambiguous)
            rec["pos"].append((x_fwd, y_left))
            rec["tags"].append(scene.tags(st, obs, flash, fired))
            rec["pose"].append((st["x"], st["y"], math.radians(st["angle"])))
            if out[0] >= 0.0 and not visible and not ambiguous and len(kept) < int(job.get("keep", 0)):
                kept.append((len(rec["logit"]) - 1, obs["rgb"].copy()))      # false detections, to look at
            action = wander.act(obs, st)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                ended = "terminated" if terminated else "time"
                break
            st = env.privileged_state()
            flash = min(100.0, max(0.0, flash + max(0.0, health - st["health"]) - 4.0))
            health, fired = st["health"], bool(action[C.ATTACK])
    finally:
        env.close()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(name=job["name"], tier=job["tier"], error="", ended=ended, pads=scene.pads, kept=kept)
    return out


def exit_map(ep: dict, shift: float = 0.0) -> dict:
    """Write an episode's sightings into a 64-unit grid the way the spatial memory does; see the module docstring.

    `shift` is subtracted from every logit first: what the episode would have been with a stricter (positive
    shift) or looser decision threshold."""
    pads = {(int(p[0] // CELL), int(p[1] // CELL)) for p in ep["pads"]}

    def on_pad(cell) -> bool:
        return any(abs(cell[0] - p[0]) <= 1 and abs(cell[1] - p[1]) <= 1 for p in pads)

    belief: dict = {}
    conf = 1.0 / (1.0 + np.exp(-(ep["logit"] - shift)))
    dist = np.hypot(ep["pos"][:, 0], ep["pos"][:, 1])
    near = ep["visible"] & (dist < NEAR)
    first_view = int(np.flatnonzero(near)[0]) if near.any() else -1
    first_hit = -1
    for t in np.flatnonzero(conf >= MIN_CONFIDENCE):
        x, y, heading = ep["pose"][t]
        fwd, left = ep["pred"][t]
        c, s = math.cos(heading), math.sin(heading)
        cell = (int((x + fwd * c - left * s) // CELL), int((y + fwd * s + left * c) // CELL))
        belief[cell] = min(1.0, belief.get(cell, 0.0) + EXIT_GAIN * min(float(conf[t]), 1.0))
        if first_hit < 0 and first_view >= 0 and t >= first_view and on_pad(cell):
            first_hit = int(t)
    best = max(belief, key=belief.get) if belief else None
    false_cells = [v for cell, v in belief.items() if not on_pad(cell)]
    return {"pad_in_view": int(near.sum()) >= 3, "found": bool(best is not None and on_pad(best)),
            "exact": bool(best is not None and best in pads),
            "steps_to_find": (first_hit - first_view) if first_hit >= 0 else -1,
            "sightings": int((conf >= MIN_CONFIDENCE).sum()),
            "false_cells": len(false_cells), "false_strong": int(sum(v >= STRONG for v in false_cells))}


def summarise(episodes: list[dict], shift: float = 0.0) -> list[dict]:
    """One row per tier and one for all tiers: the frame metrics and the episode metrics (`shift`: see `exit_map`)."""
    rows = []
    tiers = sorted({ep["tier"] for ep in episodes})
    for label, group in [("all", episodes)] + [(f"tier {t}", [ep for ep in episodes if ep["tier"] == t]) for t in tiers]:
        logit = np.concatenate([ep["logit"] for ep in group]) - shift
        vis = np.concatenate([ep["visible"] for ep in group])
        amb = np.concatenate([ep["ambiguous"] for ep in group])
        pos = np.concatenate([ep["pos"] for ep in group])
        pred = np.concatenate([ep["pred"] for ep in group])
        tags = np.concatenate([ep["tags"] for ep in group])
        fired, neg, dist = logit >= 0.0, ~vis & ~amb, np.hypot(pos[:, 0], pos[:, 1])
        tp, fp = int((fired & vis).sum()), int((fired & neg).sum())

        def rate(a, b) -> float:
            return float((a & b).sum() / b.sum()) if b.any() else float("nan")

        maps = [exit_map(ep, shift) for ep in group]
        seen = [m for m in maps if m["pad_in_view"]]
        delays = [m["steps_to_find"] for m in seen if m["steps_to_find"] >= 0]
        err = np.hypot(pred[:, 0] - pos[:, 0], pred[:, 1] - pos[:, 1])
        hit = fired & vis & (dist < NEAR)
        row = {"subset": label, "episodes": len(group), "frames": int(len(logit)), "visible_frames": int(vis.sum()),
               "negative_frames": int(neg.sum()), "ambiguous_frames": int(amb.sum()),
               "precision": tp / (tp + fp) if tp + fp else float("nan"), "false_positives": fp,
               "fp_per_1000": 1000.0 * rate(fired, neg),
               "recall_0_200": rate(fired, vis & (dist < 200)), "recall_200_400": rate(fired, vis & (dist >= 200) & (dist < 400)),
               "recall_0_400": rate(fired, vis & (dist < 400)), "recall_400_640": rate(fired, vis & (dist >= 400)),
               "n_0_400": int((vis & (dist < 400)).sum()), "n_400_640": int((vis & (dist >= 400)).sum()),
               "pos_err_median_0_400": float(np.median(err[hit])) if hit.any() else float("nan"),
               "pos_err_p90_0_400": float(np.percentile(err[hit], 90)) if hit.any() else float("nan"),
               "episodes_pad_in_view": len(seen), "episodes_found": sum(m["found"] for m in seen),
               "episodes_found_exact_cell": sum(m["exact"] for m in seen),
               "steps_to_find_median": float(np.median(delays)) if delays else float("nan"),
               "steps_to_find_p90": float(np.percentile(delays, 90)) if delays else float("nan"),
               "episodes_with_false_cell": sum(m["false_cells"] > 0 for m in maps),
               "false_cells": sum(m["false_cells"] for m in maps), "false_strong_cells": sum(m["false_strong"] for m in maps)}
        for tag in TAGS:
            group_neg = neg & has_tag(tags, tag)
            row[f"n_neg_{tag}"] = int(group_neg.sum())
            row[f"fp_per_1000_{tag}"] = 1000.0 * rate(fired, group_neg)
        rows.append(row)
    return rows


def run(suites, weights, out_dir=OUT_DIR, workers: int = 3, seeds=(0, 1), limit: int | None = None, log=print) -> dict:
    """Play every map of the suites once per seed. Returns {"rows": summary rows, "episodes": per-episode results}."""
    import multiprocessing as mp

    from doom_env.limits import THREAD_ENV

    out_dir = Path(out_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    jobs = []
    for suite in suites:
        rows = [json.loads(line) for line in Path(suite).read_text().splitlines() if line.strip()]
        for row in rows[:limit]:
            for seed in seeds:
                jobs.append({"name": f"{row['name']}_s{seed}", "tier": int(row["tier"]), "grid": row["grid"],
                             "options": row.get("options") or {}, "episode_ticks": int(row["episode_ticks"]),
                             "seed": int(seed), "weights": str(weights), "wad": str(out_dir / "maps" / f"{row['name']}.wad")})
    began = time.perf_counter()
    if workers <= 0:
        results = [play(job) for job in jobs]
    else:
        saved = {k: os.environ.get(k) for k in THREAD_ENV}
        os.environ.update(THREAD_ENV)
        try:
            with mp.get_context("spawn").Pool(workers) as pool:
                # one map's episodes stay in one chunk: they write the same WAD file
                results = list(pool.imap(play, jobs, chunksize=len(seeds)))
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    errors = [r for r in results if r["error"]]
    for r in errors[:3]:
        log(f"error in {r['name']}: {r['error'][-600:]}")
    episodes = [r for r in results if not r["error"]]
    rows = summarise(episodes)
    log(f"{len(episodes)} episodes ({len(errors)} errors), {rows[0]['frames']} frames in {time.perf_counter() - began:.0f} s")
    return {"rows": rows, "episodes": episodes, "errors": len(errors)}


def write_csv(rows: list[dict], path=LOG_CSV, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**(extra or {}), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}} for r in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def summary_lines(rows: list[dict]) -> list[str]:
    return [f"{r['subset']:>7}: {r['episodes']:3d} episodes {r['frames']:6d} frames  precision {r['precision']:.4f}  FP/1000 "
            f"{r['fp_per_1000']:.2f} ({r['false_positives']})  recall 0-400 {r['recall_0_400']:.3f} (n={r['n_0_400']})  400-640 "
            f"{r['recall_400_640']:.3f} (n={r['n_400_640']})  pos err {r['pos_err_median_0_400']:.1f}/{r['pos_err_p90_0_400']:.1f}  "
            f"found {r['episodes_found']}/{r['episodes_pad_in_view']} (exact cell {r['episodes_found_exact_cell']})  steps to find "
            f"{r['steps_to_find_median']:.0f}/{r['steps_to_find_p90']:.0f}  episodes with a false cell {r['episodes_with_false_cell']} "
            f"(cells {r['false_cells']}, strong {r['false_strong_cells']})" for r in rows]


def main(argv=None) -> int:
    from training.exitdet.evaluate import WEIGHTS, file_sha256

    ap = argparse.ArgumentParser(prog="python -m training.exitdet.episodes", description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--suite", default=str(paths.ROOT / "training" / "suites" / "dev_v1.jsonl"), help="comma-separated suite files")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--limit", type=int, default=None, help="first N maps of each suite")
    ap.add_argument("--csv", default="", help="write the table here (default: print only)")
    ap.add_argument("--shifts", default="", help="also print the all-tiers row for these threshold shifts (logits), e.g. 1,2,3")
    args = ap.parse_args(argv)
    result = run([s for s in args.suite.split(",") if s], args.weights, args.out, args.workers,
                 [int(s) for s in args.seeds.split(",")], args.limit)
    print("\n".join(summary_lines(result["rows"])))
    for shift in [float(v) for v in args.shifts.split(",") if v]:
        print(f"threshold raised by {shift:+.1f}: " + summary_lines(summarise(result["episodes"], shift)[:1])[0])
    r = result["rows"][0]
    print("false detections per 1,000 negative frames by what is in view: "
          + ", ".join(f"{t} {r['fp_per_1000_' + t]:.1f} (n={r['n_neg_' + t]})" for t in TAGS))
    if args.csv:
        extra = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "weights_sha256": file_sha256(args.weights)[:12],
                 "suite": ",".join(Path(s).stem for s in args.suite.split(",") if s), "seeds": args.seeds.replace(",", " ")}
        print("wrote", write_csv(result["rows"], args.csv, extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
