"""Evaluation of the exit detector: detection, position and latency tables.

Training-time scaffolding. Never imported by the agent in submission/.

    python -m training.exitdet.evaluate --weights submission/weights/exit_detector.pt

A frame counts as a detection when sigmoid(visibility logit) >= 0.5, which is the confidence from
which the agent's memory map and encoder count a sighting. Frames the label code marks as ambiguous
(part of a pad may be in the frame without the pad centre being visible within 640 units) are
neither positives nor negatives; how often the detector fires on them is reported separately.

    precision        detections on visible-pad frames / all detections on visible and negative frames
    recall           by true distance of the pad centre: 0-200, 200-400, 400-640 units, and 0-400.
                     `recall_*` uses every frame the geometric label calls visible. `recall_drawn_*`
                     keeps only frames in which the renderer drew pad pixels (`data.drawn`): beyond
                     about 450 units the pad is under one pixel row and is often not drawn.
    fp_per_1000_*    false detections per 1,000 negative frames, over all negatives and over the
                     negatives that show each kind of hard negative (red key, red door, fireball...)
    pos_err_*        distance in map units between the predicted and the true pad centre, over
                     detected visible frames: median and 90th percentile
    latency          one frame, batch 1, one CPU thread, through the same forward the agent calls
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from training import paths
from doomnet.model import ExitDetector
from training.exitdet.data import KINDS, SCREEN_H, SCREEN_W, TAGS, drawn, has_tag, load_set

WEIGHTS = paths.SUBMISSION / "weights" / "exit_detector.pt"
LOG_CSV = paths.ROOT / "training" / "logs" / "exit_detector.csv"
POS_SCALE = 1024.0                 # the detector's position outputs are map units / 1024
BANDS = (("0_200", 0.0, 200.0), ("200_400", 200.0, 400.0), ("0_400", 0.0, 400.0), ("400_640", 400.0, 1e9))
HARD_NEGATIVES = TAGS              # every tag gets a false-positive row


def load_detector(path=WEIGHTS) -> tuple[ExitDetector, dict]:
    """The detector with the weights of `path` ({"state_dict", "config"} or a bare state dict) and its config."""
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model = ExitDetector()
    model.load_state_dict(state)
    return model.eval(), (blob.get("config", {}) if isinstance(blob, dict) else {})


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def predict(model: ExitDetector, rgb: np.ndarray, top: int = 0, device: str = "cpu", batch: int = 1024):
    """Run the detector over uint8 frames (N, 120 - top, 160, 3); rows above `top` are not stored
    (the detector does not read them) and are filled with zeros. Returns (logit (N,), position (N, 2) in map units)."""
    model = model.to(device).eval()
    logits, pos = [], []
    for i in range(0, len(rgb), batch):
        x = torch.from_numpy(rgb[i:i + batch]).to(device).permute(0, 3, 1, 2)
        if top:
            x = torch.nn.functional.pad(x, (0, 0, top, 0))
        out = model(x).float().cpu()
        logits.append(out[:, 0])
        pos.append(out[:, 1:3] * POS_SCALE)
    return torch.cat(logits).numpy(), torch.cat(pos).numpy()


def detection_metrics(data: dict, logit: np.ndarray, pos: np.ndarray, select: np.ndarray | None = None,
                      threshold_logit: float = 0.0) -> dict:
    """The detection and position metrics of the module docstring over the frames in `select` (all by default)."""
    n = len(logit)
    select = np.ones(n, bool) if select is None else np.asarray(select, bool)
    vis = data["visible"] & select
    neg = ~data["visible"] & ~data["ambiguous"] & select
    amb = data["ambiguous"] & select
    seen = drawn(data) & select
    fired = logit >= threshold_logit
    dist = np.hypot(data["pos"][:, 0], data["pos"][:, 1])
    tp, fp = int((fired & vis).sum()), int((fired & neg).sum())

    def rate(a: np.ndarray, b: np.ndarray) -> float:
        return float((a & b).sum() / b.sum()) if b.any() else float("nan")

    out = {"frames": int(select.sum()), "visible": int(vis.sum()), "negative": int(neg.sum()), "ambiguous": int(amb.sum()),
           "detections": tp + fp, "true_positives": tp, "false_positives": fp,
           "precision": tp / (tp + fp) if tp + fp else float("nan"),
           "recall": rate(fired, vis), "recall_drawn": rate(fired, seen)}
    for name, lo, hi in BANDS:
        band = (dist >= lo) & (dist < hi)
        out[f"n_{name}"] = int((vis & band).sum())
        out[f"recall_{name}"] = rate(fired, vis & band)
        out[f"n_drawn_{name}"] = int((seen & band).sum())
        out[f"recall_drawn_{name}"] = rate(fired, seen & band)
    out["fp_per_1000_all"] = 1000.0 * rate(fired, neg)
    for tag in HARD_NEGATIVES:
        group = neg & has_tag(data["tags"], tag)
        out[f"n_neg_{tag}"] = int(group.sum())
        out[f"fp_per_1000_{tag}"] = 1000.0 * rate(fired, group)
    for i, kind in enumerate(KINDS):
        group = neg & (data["kind"] == i)
        out[f"n_neg_kind_{kind}"] = int(group.sum())
        out[f"fp_per_1000_kind_{kind}"] = 1000.0 * rate(fired, group)
    plain = neg & (data["tags"] == 0)
    out["n_neg_untagged"] = int(plain.sum())
    out["fp_per_1000_untagged"] = 1000.0 * rate(fired, plain)
    out["fires_on_ambiguous"] = rate(fired, amb)

    err = np.hypot(pos[:, 0] - data["pos"][:, 0], pos[:, 1] - data["pos"][:, 1])
    for name, lo, hi in BANDS + (("all", 0.0, 1e9),):
        hit = fired & vis & (dist >= lo) & (dist < hi)
        out[f"pos_err_median_{name}"] = float(np.median(err[hit])) if hit.any() else float("nan")
        out[f"pos_err_p90_{name}"] = float(np.percentile(err[hit], 90)) if hit.any() else float("nan")
    hit = fired & vis & (dist < 400.0)
    out["pos_err_within_32_0_400"] = float((err[hit] <= 32.0).mean()) if hit.any() else float("nan")
    out["pos_err_within_64_0_400"] = float((err[hit] <= 64.0).mean()) if hit.any() else float("nan")
    return out


def latency(model: ExitDetector, frames: int = 3000, warmup: int = 200, seed: int = 0) -> dict:
    """Per-frame time of the detector's forward at batch 1 on one CPU thread, uint8 input as the agent passes it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = model.to("cpu").eval()
        gen = torch.Generator().manual_seed(seed)
        pool = torch.randint(0, 256, (64, 1, 3, SCREEN_H, SCREEN_W), generator=gen, dtype=torch.uint8)
        times = np.empty(frames)
        with torch.inference_mode():
            for i in range(warmup):
                model(pool[i % 64])
            for i in range(frames):
                x = pool[i % 64]
                t0 = time.perf_counter()
                model(x)
                times[i] = time.perf_counter() - t0
    finally:
        torch.set_num_threads(threads)
    ms = times * 1000.0
    return {"latency_mean_ms": float(ms.mean()), "latency_p50_ms": float(np.percentile(ms, 50)),
            "latency_p95_ms": float(np.percentile(ms, 95)), "latency_max_ms": float(ms.max()), "latency_frames": frames}


def subsets(data: dict) -> dict[str, np.ndarray]:
    """Frame selections reported for a set: all frames, and the maps in the reference look and in a random look."""
    look = np.array([m.get("look", "") for m in data["maps"]])
    by_frame = look[data["map_id"]] if len(look) else np.array([""] * len(data["visible"]))
    out = {"all": np.ones(len(data["visible"]), bool)}
    for name in ("reference", "random"):
        if (by_frame == name).any():
            out[f"{name} look"] = by_frame == name
    return out


def evaluate_sets(model: ExitDetector, data_dir, sets=("heldout", "unseen"), device: str = "cpu", top: int | None = None,
                  max_frames: int | None = None) -> list[dict]:
    """One row of metrics per (set, subset)."""
    top = ExitDetector.TOP if top is None else top
    rows = []
    for name in sets:
        data = load_set(Path(data_dir) / name, top=top, max_frames=max_frames)
        logit, pos = predict(model, data["rgb"], top=top, device=device)
        for subset, select in subsets(data).items():
            rows.append({"set": name, "subset": subset, **detection_metrics(data, logit, pos, select)})
    model.to("cpu")
    return rows


def gate(rows: list[dict], lat: dict) -> dict:
    """The four gate numbers for every (set, subset) row and whether each is met."""
    out = {}
    for r in rows:
        checks = {"precision >= 0.98": r["precision"] >= 0.98, "recall within 400 >= 0.90": r["recall_0_400"] >= 0.90,
                  "median position error within 400 <= 32": r["pos_err_median_0_400"] <= 32.0,
                  "latency <= 1.5 ms": lat["latency_mean_ms"] <= 1.5}
        out[f"{r['set']}/{r['subset']}"] = {k: bool(v) for k, v in checks.items()}
    return out


def write_csv(rows: list[dict], path=LOG_CSV, extra: dict | None = None) -> Path:
    """Write the table: one line per (set, subset), every metric a column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**(extra or {}), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}} for r in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def summary_lines(rows: list[dict], lat: dict | None = None) -> list[str]:
    out = []
    for r in rows:
        out.append(
            f"{r['set']:>8} {r['subset']:<14} frames {r['frames']:6d}  precision {r['precision']:.4f}  recall 0-200 "
            f"{r['recall_0_200']:.3f}  200-400 {r['recall_200_400']:.3f}  0-400 {r['recall_0_400']:.3f}  400-640 "
            f"{r['recall_400_640']:.3f} (drawn {r['recall_drawn_400_640']:.3f})  FP/1000 {r['fp_per_1000_all']:.2f}  "
            f"pos err 0-400 median {r['pos_err_median_0_400']:.1f} p90 {r['pos_err_p90_0_400']:.1f}")
    if lat:
        out.append(f"latency, one thread, batch 1: mean {lat['latency_mean_ms']:.3f} ms, p50 {lat['latency_p50_ms']:.3f}, "
                   f"p95 {lat['latency_p95_ms']:.3f}, max {lat['latency_max_ms']:.3f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.exitdet.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--data", default=str(paths.ROOT / "data" / "exitdet"))
    ap.add_argument("--sets", default="heldout,unseen")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default="", help="write the table here (default: print only)")
    args = ap.parse_args(argv)
    model, config = load_detector(args.weights)
    rows = evaluate_sets(model, args.data, [s for s in args.sets.split(",") if s], args.device)
    lat = latency(model)
    print("\n".join(summary_lines(rows, lat)))
    for r in rows:
        print(f"{r['set']}/{r['subset']}: false positives per 1,000 negative frames by hard negative: "
              + ", ".join(f"{t} {r['fp_per_1000_' + t]:.1f} (n={r['n_neg_' + t]})" for t in HARD_NEGATIVES))
    print(json.dumps(gate(rows, lat), indent=1))
    if args.csv:
        extra = {"weights_sha256": file_sha256(args.weights)[:12], **{k: round(v, 4) for k, v in lat.items()}}
        print("wrote", write_csv(rows, args.csv, extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
