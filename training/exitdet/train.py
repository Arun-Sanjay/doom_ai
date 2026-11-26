"""Train the exit detector (doomnet.model.ExitDetector) on the collected frames.

Training-time scaffolding. Never imported by the agent in submission/.

    python -m training.exitdet.train --tag v1 --save

Targets. The visibility target of a frame is 1 when the label says the pad is visible and the
renderer drew it (`data.drawn`: pad-coloured pixels where the pad must be). Frames the label marks
ambiguous, and frames labelled visible in which nothing of the pad was drawn (too far for one pixel
row, faded to black in a dark room, or below the frame), take no part in the visibility loss: the
detector is never told that a pad it cannot see is there, nor that a pad it can partly see is not.
The position target (forward and left offset of the pad centre / 1024) is used on positive frames
and, at half weight, on ambiguous frames in which the pad is drawn and its centre is in line of
sight (a pad between 640 and 900 units, or one at the edge of the view), so that a detection on
such a frame still points at the pad.

Loss. Binary cross-entropy on the visibility logit plus a Huber loss on the two position outputs.
Every batch goes through `augment.augment` (flip, simulated damage and pickup tints, gain,
brightness, contrast, gamma, saturation, hue, noise, slight blur). AdamW with a warm-up and a
cosine schedule; the weights that are kept are an exponential moving average of the trained ones.

Decision threshold. The agent counts a sighting from confidence 0.5. After training the threshold
that gives the wanted precision on the validation maps is found and folded into the bias of the
visibility output, so that "sigmoid(logit) >= 0.5" is that operating point and nothing downstream
needs to know a threshold.

Outputs. runs/exitdet/TAG/: exit_detector.pt ({"state_dict", "config"}), train.csv (one row per
epoch), table.csv and report.json (held-out evaluation). With --save the weights are also written
to submission/weights/exit_detector.pt and the tables to training/logs/exit_detector*.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training import paths
from doomnet.model import ExitDetector
from training.exitdet import evaluate as ev
from training.exitdet.augment import AugmentConfig, augment
from training.exitdet.data import describe, drawn, load_set

RUNS = paths.ROOT / "runs" / "exitdet"
LOGS = paths.ROOT / "training" / "logs"
# Ambiguous frames that still get a position target: pad drawn, centre in line of sight and in front.
AMBIGUOUS_MIN_PX = 3
AMBIGUOUS_TAN = math.tan(math.radians(50.0))
AMBIGUOUS_RANGE = 900.0
AMBIGUOUS_POS_WEIGHT = 0.5


@dataclass
class TrainConfig:
    epochs: int = 24
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    pos_weight: float = 10.0         # weight of the position loss against the visibility loss
    huber_beta: float = 0.02         # 20 map units: quadratic below, linear above
    far_boost: float = 1.0           # extra visibility weight of a positive at 640 units (0 at the camera)
    hard_negative_weight: float = 1.0
    ema: float = 0.999
    seed: int = 0
    precision_target: float = 0.995  # on the validation maps, for the decision threshold
    min_recall_0_400: float = 0.93   # the threshold is lowered again if recall within 400 units falls below this


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def targets(data: dict) -> dict:
    """Per-frame training targets: visibility (1 / 0), whether the frame counts for the visibility
    loss, position / 1024 and its weight, and whether the frame shows a tagged hard negative."""
    positive = drawn(data)
    valid = positive | (~data["visible"] & ~data["ambiguous"])
    fwd, left = data["pad_ego"][:, 0], data["pad_ego"][:, 1]
    edge = (data["ambiguous"] & data["pad_los"] & (data["pad_px"] >= AMBIGUOUS_MIN_PX) & (fwd > 16.0)
            & (np.abs(left) <= fwd * AMBIGUOUS_TAN) & (data["pad_dist"] <= AMBIGUOUS_RANGE))
    pos = np.where(positive[:, None], data["pos"], np.where(edge[:, None], data["pad_ego"], 0.0))
    return {"y": positive.astype(np.float32), "valid": valid.astype(np.float32),
            "pos": (pos / ev.POS_SCALE).astype(np.float32),
            "pos_w": (positive + AMBIGUOUS_POS_WEIGHT * edge).astype(np.float32),
            "hard": (data["tags"] != 0).astype(np.float32)}


def loss_terms(out: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, pos: torch.Tensor, cfg: TrainConfig,
               hard: torch.Tensor | None = None, pos_w: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Total loss and its two parts for one batch of detector outputs (B, 3).

    y: visibility target; valid: 1 where the frame counts for the visibility loss; pos: position
    target / 1024; pos_w: weight of the position loss per frame (default: the positives).
    """
    pos_w = y if pos_w is None else pos_w
    weight = valid.clone()
    if cfg.far_boost:
        weight = weight * (1.0 + cfg.far_boost * y * torch.hypot(pos[:, 0], pos[:, 1]) / 0.625)
    if hard is not None and cfg.hard_negative_weight != 1.0:
        weight = weight * (1.0 + (cfg.hard_negative_weight - 1.0) * hard * (1.0 - y))
    bce = F.binary_cross_entropy_with_logits(out[:, 0], y, reduction="none")
    vis_loss = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    per_frame = F.smooth_l1_loss(out[:, 1:3], pos, beta=cfg.huber_beta, reduction="none").sum(1)
    pos_loss = (per_frame * pos_w).sum() / pos_w.sum().clamp_min(1.0)
    return vis_loss + cfg.pos_weight * pos_loss, {"vis": float(vis_loss.detach()), "pos": float(pos_loss.detach())}


def forward_cropped(model: ExitDetector, x: torch.Tensor) -> torch.Tensor:
    """The detector's forward on frames that already start at row ExitDetector.TOP: float (B, 3, 120 - TOP, 160).

    Same layers and same result as `model(rgb)`; the only difference is that the 2 x 2 max pooling
    is the backend's own operator. The class pools with strided slices, which is the fastest form
    for one CPU thread at batch 1 but several times slower to differentiate on a GPU (measured on
    MPS at batch 256: 317 ms against 41 ms for the backward pass). `check_forward` verifies the two
    forwards agree before any training step, so a change to the class cannot go unnoticed.
    """
    x = F.max_pool2d(torch.relu(model.conv1(x)), 2)
    x = F.max_pool2d(torch.relu(model.conv2(x)), 2)
    x = F.max_pool2d(torch.relu(model.conv3(x)), 2)
    x = torch.relu(model.squeeze(torch.relu(model.conv4(x)))).flatten(1)
    return model.fc2(torch.relu(model.fc1(x)))


@torch.no_grad()
def check_forward(model: ExitDetector, tolerance: float = 1e-5) -> float:
    """Largest difference between `forward_cropped` and the class's own forward on random frames."""
    device = next(model.parameters()).device
    frames = torch.randint(0, 256, (4, 3, 120, 160), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    own = model(frames.to(device))
    ours = forward_cropped(model, frames.to(device)[:, :, ExitDetector.TOP:].float() / 255.0)
    diff = float((own - ours).abs().max())
    if not diff <= tolerance:
        raise RuntimeError(f"training forward differs from ExitDetector.forward by {diff}: the class has changed")
    return diff


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t))))


def choose_threshold(logit: np.ndarray, data: dict, precision_target: float, min_recall: float) -> dict:
    """Lowest threshold (as a logit) whose precision on these frames reaches the target.

    If recall within 400 units at that threshold is below `min_recall`, the threshold is lowered to
    the highest one that keeps that recall, and the precision actually reached is reported.
    """
    vis = data["visible"]
    neg = ~data["visible"] & ~data["ambiguous"]
    near = vis & (np.hypot(data["pos"][:, 0], data["pos"][:, 1]) < 400.0)
    order = np.argsort(-logit)
    tp, fp = np.cumsum(vis[order]), np.cumsum(neg[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall_near = np.cumsum(near[order]) / max(1, int(near.sum()))
    ok = np.flatnonzero(precision >= precision_target)
    k = int(ok[-1]) if len(ok) else 0                       # the longest prefix that still meets the target
    if recall_near[k] < min_recall:
        k = max(k, int(np.searchsorted(recall_near, min_recall)))
        k = min(k, len(order) - 1)
    return {"threshold_logit": float(logit[order[k]]), "precision": float(precision[k]),
            "recall_0_400": float(recall_near[k]), "detections": int(k + 1)}


def train(cfg: TrainConfig, data_dir, out_dir, device: str = "auto", max_train_frames: int | None = None,
          aug: AugmentConfig | None = None, log=print) -> dict:
    """Train, calibrate and save. Returns {"weights": path, "config": dict, "history": rows}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)
    aug = aug or AugmentConfig()
    top = ExitDetector.TOP
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    began = time.perf_counter()
    train_set = load_set(Path(data_dir) / "train", top=top, max_frames=max_train_frames)
    val_set = load_set(Path(data_dir) / "val", top=top)
    stats = describe(train_set)
    log(f"learner device: {device}; train {stats['frames']} frames from {stats['maps']} maps "
        f"({stats['visible_share']:.1%} visible, {stats['rendered']} drawn, {stats['ambiguous']} ambiguous); "
        f"val {len(val_set['visible'])} frames; loaded in {time.perf_counter() - began:.0f} s")
    rgb = torch.from_numpy(train_set["rgb"])
    tgt = {k: torch.from_numpy(v) for k, v in targets(train_set).items()}
    n = len(rgb)

    model = ExitDetector().to(device)
    check_forward(model)
    decay = [p for name, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for name, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    steps_per_epoch = n // cfg.batch
    total = steps_per_epoch * cfg.epochs
    ema_model = ExitDetector().to(device)
    history, step = [], 0
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(n, generator=gen)
        sums = {"loss": 0.0, "vis": 0.0, "pos": 0.0}
        t0 = time.perf_counter()
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch:(b + 1) * cfg.batch]
            x = rgb[idx].to(device)
            y, valid, pos, hard, pos_w = (tgt[k][idx].to(device) for k in ("y", "valid", "pos", "hard", "pos_w"))
            x, pos = augment(x, pos, gen, aug)
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total, cfg)
            out = forward_cropped(model, x)
            loss, parts = loss_terms(out, y, valid, pos, cfg, hard, pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                d = min(cfg.ema, (1.0 + step) / (10.0 + step))
                for k, v in model.state_dict().items():
                    shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            sums["loss"] += float(loss.detach())
            sums["vis"] += parts["vis"]
            sums["pos"] += parts["pos"]
            step += 1
        ema_model.load_state_dict(shadow)
        logit, pos_pred = ev.predict(ema_model, val_set["rgb"], top=top, device=device)
        m = ev.detection_metrics(val_set, logit, pos_pred)
        row = {"epoch": epoch + 1, "steps": step, "lr": lr_at(step - 1, total, cfg),
               "train_loss": sums["loss"] / steps_per_epoch, "train_vis_loss": sums["vis"] / steps_per_epoch,
               "train_pos_loss": sums["pos"] / steps_per_epoch,
               "val_precision": m["precision"], "val_recall_0_200": m["recall_0_200"], "val_recall_200_400": m["recall_200_400"],
               "val_recall_0_400": m["recall_0_400"], "val_recall_400_640": m["recall_400_640"],
               "val_fp_per_1000": m["fp_per_1000_all"], "val_pos_err_median_0_400": m["pos_err_median_0_400"],
               "val_pos_err_p90_0_400": m["pos_err_p90_0_400"], "epoch_s": time.perf_counter() - t0,
               "frames_per_s": steps_per_epoch * cfg.batch / (time.perf_counter() - t0), "device": device}
        history.append(row)
        log(f"epoch {row['epoch']:2d}  loss {row['train_loss']:.4f} (vis {row['train_vis_loss']:.4f}, pos "
            f"{row['train_pos_loss']:.5f})  val precision {m['precision']:.4f}  recall 0-200 {m['recall_0_200']:.3f} "
            f"200-400 {m['recall_200_400']:.3f} 400-640 {m['recall_400_640']:.3f}  FP/1000 {m['fp_per_1000_all']:.2f}  "
            f"pos err {m['pos_err_median_0_400']:.1f}/{m['pos_err_p90_0_400']:.1f}  {row['frames_per_s']:.0f} frames/s")
    with (out_dir / "train.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows([{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in history])

    # ---- decision threshold on maps not trained on (val, plus calib when collected), folded into the bias
    ema_model.load_state_dict(shadow)
    held = [val_set] + ([load_set(Path(data_dir) / "calib", top=top)] if (Path(data_dir) / "calib").is_dir() else [])
    logit = np.concatenate([ev.predict(ema_model, part["rgb"], top=top, device=device)[0] for part in held])
    merged = {k: np.concatenate([part[k] for part in held]) for k in ("visible", "ambiguous", "pos")}
    np.savez(out_dir / "threshold_scores.npz", logit=logit, **merged)
    pick = choose_threshold(logit, merged, cfg.precision_target, cfg.min_recall_0_400)
    state = {k: v.detach().to("cpu").clone() for k, v in shadow.items()}
    state["fc2.bias"][0] -= pick["threshold_logit"]
    log(f"threshold on {len(logit)} frames not trained on: logit {pick['threshold_logit']:+.3f} (probability {1 / (1 + math.exp(-pick['threshold_logit'])):.3f}) "
        f"-> precision {pick['precision']:.4f}, recall within 400 {pick['recall_0_400']:.4f}; folded into fc2.bias[0]")

    config = {"arch": "doomnet.model.ExitDetector", "input": "rgb (B,3,120,160) uint8, or float in [0,1]",
              "rows_read": [top, 120], "outputs": ["visibility logit", "forward / 1024", "left / 1024"],
              "position_scale": ev.POS_SCALE, "sighting_from_confidence": 0.5,
              "bias_shift_logit": -pick["threshold_logit"], "threshold_chosen_on": "val" + ("+calib" if len(held) > 1 else ""),
              "threshold_frames": int(len(logit)),
              "val_precision_at_threshold": pick["precision"], "val_recall_0_400_at_threshold": pick["recall_0_400"],
              "train_frames": int(n), "train_maps": stats["maps"], "label_max_range": 640.0,
              "parameters": int(sum(p.numel() for p in model.parameters())),
              "train": asdict(cfg), "augment": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(aug).items()},
              "learner_device": device, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "train_seconds": round(time.perf_counter() - began, 1)}
    weights = out_dir / "exit_detector.pt"
    torch.save({"state_dict": state, "config": config}, str(weights))
    (out_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    return {"weights": weights, "config": config, "history": history}


def report(weights, data_dir, out_dir, device: str = "auto", sets=("val", "heldout", "unseen"), log=print) -> dict:
    """Evaluate saved weights on the evaluation sets; writes table.csv and report.json into `out_dir`."""
    out_dir = Path(out_dir)
    model, config = ev.load_detector(weights)
    rows = ev.evaluate_sets(model, data_dir, sets, pick_device(device))
    lat = ev.latency(model)
    extra = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "weights_sha256": ev.file_sha256(weights)[:12],
             **{k: round(v, 4) for k, v in lat.items()}}
    ev.write_csv(rows, out_dir / "table.csv", extra)
    result = {"weights": str(weights), "weights_sha256": ev.file_sha256(weights), "config": config, "latency": lat,
              "gate": ev.gate([r for r in rows if r["set"] != "val"], lat), "rows": rows}
    (out_dir / "report.json").write_text(json.dumps(result, indent=1) + "\n")
    for line in ev.summary_lines(rows, lat):
        log(line)
    for r in rows:
        if r["subset"] == "all":
            log(f"{r['set']}: FP per 1,000 negatives: " + ", ".join(
                f"{t} {r['fp_per_1000_' + t]:.1f} (n={r['n_neg_' + t]})" for t in ev.HARD_NEGATIVES))
    return result


def publish(run_dir, log=print) -> None:
    """Copy a run's weights into submission/weights/ and its tables into training/logs/."""
    run_dir = Path(run_dir)
    (paths.SUBMISSION / "weights").mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "exit_detector.pt", ev.WEIGHTS)
    shutil.copyfile(run_dir / "table.csv", LOGS / "exit_detector.csv")
    shutil.copyfile(run_dir / "train.csv", LOGS / "exit_detector_train.csv")
    shutil.copyfile(run_dir / "report.json", LOGS / "exit_detector.json")
    log(f"published {ev.WEIGHTS} (sha256 {ev.file_sha256(ev.WEIGHTS)[:12]}) and training/logs/exit_detector.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.exitdet.train", description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(paths.ROOT / "data" / "exitdet"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--device", default="auto", help="auto = cuda, else mps, else cpu")
    ap.add_argument("--max-train-frames", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="publish the weights and tables after the evaluation")
    ap.add_argument("--threads", type=int, default=3)
    for name, default in asdict(TrainConfig()).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = TrainConfig(**{name: getattr(args, name) for name in asdict(TrainConfig())})
    run_dir = RUNS / args.tag

    def log(*parts) -> None:
        print(*parts, flush=True)

    result = train(cfg, args.data, run_dir, args.device, args.max_train_frames, log=log)
    report(result["weights"], args.data, run_dir, args.device, log=log)
    if args.save:
        publish(run_dir, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
