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
                     `recall_clear_*` also leaves out frames in which a monster or a corpse stands
                     in front of most of the pad (`data.clear`; the label's line of sight is cut by
                     walls only): the pad is in clear view. It is empty (n_clear = 0) in sets
                     collected before the cover was recorded (`cover_recorded` = 0).
                     `recall_covered_0_400` is the rest: pads drawn but mostly behind a sprite.
                     `recall_0_400_lo95 / _hi95` and the same for `recall_clear_0_400`: a 95%
                     interval from resampling whole maps (frames of one map are not independent).
    fp_per_1000_*    false detections per 1,000 negative frames, over all negatives and over the
                     negatives that show each kind of hard negative (red key, red door, fireball...)
    *_pad_at_edge    negatives by the label in which a patch of pad is drawn all the same (see
                     `data.pad_at_edge`): how many there are and how many false detections fall on
                     them. They count against `precision`; `precision_without_pad_at_edge` leaves them out.
    pos_err_*        distance in map units between the predicted and the true pad centre, over
                     detected visible frames: median and 90th percentile
    latency          one frame, batch 1, one CPU thread, through the same forward the agent calls

Rows. Every set is reported as a whole and split by what makes detection harder, so that a pooled
figure cannot hide a weak case: by look (reference / random textures), by map tier (tiers 1, 3 and
4 have monsters), by whether a monster is in view, by whether the red damage tint is on the screen,
by weapon held, and, in sets built with other pad flats, by the flat the pad is drawn with.

Gate. Each row gets `gate_recall_0_400` (recall within 400 units by the label, which lets a pad
count as visible when a monster stands on it) and `gate_recall_clear_0_400` (the same over pads in
clear view): "pass", "FAIL", or "not judged" when the row has fewer than MIN_GATE_FRAMES such
frames or the set did not record the cover.

`--robustness SET` evaluates one set again under fixed changes of the picture (brightness, gamma,
contrast, saturation, hue, channel gain, noise, blur, damage tint), some beyond the training range.
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
from training.exitdet.data import (COVER_MAX, KINDS, SCREEN_H, SCREEN_W, TAGS, cover_known, drawn, has_tag, load_set,
                                   pad_at_edge)

WEIGHTS = paths.SUBMISSION / "weights" / "exit_detector.pt"
LOG_CSV = paths.ROOT / "training" / "logs" / "exit_detector.csv"
POS_SCALE = 1024.0                 # the detector's position outputs are map units / 1024
BANDS = (("0_200", 0.0, 200.0), ("200_400", 200.0, 400.0), ("0_400", 0.0, 400.0), ("400_640", 400.0, 1e9))
HARD_NEGATIVES = TAGS              # every tag gets a false-positive row
WEAPON_SLOTS = {1: "fist", 2: "pistol", 3: "shotgun", 4: "chaingun"}
MIN_GATE_FRAMES = 100              # a row with fewer positives within 400 units is reported, not judged
GATE_RECALL = 0.90
BOOTSTRAP_DRAWS = 1000


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


def map_interval(fired: np.ndarray, group: np.ndarray, map_id: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = 0) -> tuple[float, float]:
    """95% interval of the share of `group` frames that fired, resampling whole maps with replacement.

    Frames of one map share its pad, its textures and its monsters, so they are not independent;
    resampling frames would give an interval that is too narrow.
    """
    idx = np.flatnonzero(group)
    if not len(idx):
        return float("nan"), float("nan")
    _, inverse = np.unique(np.asarray(map_id)[idx], return_inverse=True)
    n = np.bincount(inverse).astype(np.float64)
    hits = np.bincount(inverse, weights=fired[idx].astype(np.float64))
    pick = np.random.default_rng(seed).integers(0, len(n), (draws, len(n)))
    share = hits[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(share, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(recall: float, n: int, floor: float = GATE_RECALL) -> str:
    """The recall gate for one row of the table: "pass", "FAIL", or "not judged" (too few frames)."""
    if n < MIN_GATE_FRAMES or not np.isfinite(recall):
        return "not judged"
    return "pass" if recall >= floor else "FAIL"


def detection_metrics(data: dict, logit: np.ndarray, pos: np.ndarray, select: np.ndarray | None = None,
                      threshold_logit: float = 0.0) -> dict:
    """The detection and position metrics of the module docstring over the frames in `select` (all by default)."""
    n = len(logit)
    select = np.ones(n, bool) if select is None else np.asarray(select, bool)
    vis = data["visible"] & select
    neg = ~data["visible"] & ~data["ambiguous"] & select
    amb = data["ambiguous"] & select
    seen = drawn(data) & select
    known = cover_known(data)                          # False in sets collected before the cover was recorded
    open_view = known & (data["pad_cover"] < COVER_MAX)     # no monster or corpse in front of most of the pad
    covered = known & (data["pad_cover"] >= COVER_MAX)
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
        out[f"n_clear_{name}"] = int((seen & open_view & band).sum())
        out[f"recall_clear_{name}"] = rate(fired, seen & open_view & band)
    near = dist < 400.0
    out["cover_recorded"] = float(known[select].mean()) if select.any() else float("nan")
    out["n_covered_0_400"] = int((seen & covered & near).sum())
    out["recall_covered_0_400"] = rate(fired, seen & covered & near)
    out["recall_0_400_lo95"], out["recall_0_400_hi95"] = map_interval(fired, vis & near, data["map_id"])
    out["recall_clear_0_400_lo95"], out["recall_clear_0_400_hi95"] = map_interval(fired, seen & open_view & near, data["map_id"])
    out["gate_recall_0_400"] = verdict(out["recall_0_400"], out["n_0_400"])
    out["gate_recall_clear_0_400"] = verdict(out["recall_clear_0_400"], out["n_clear_0_400"])
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
    edge = pad_at_edge(data) & select                  # negatives by the label in which a pad is drawn all the same
    out["n_neg_pad_at_edge"] = int(edge.sum())
    out["false_positives_pad_at_edge"] = int((fired & edge).sum())
    out["precision_without_pad_at_edge"] = tp / (tp + fp - out["false_positives_pad_at_edge"]) if tp else float("nan")

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
    """Frame selections reported for a set: all frames; the maps in the reference look and in a
    random look; the maps of each tier; frames with and without a monster in view; frames with and
    without the red damage tint; (where the collector recorded it) the frames by the weapon the
    player holds; and the maps of each pad flat, where maps name one. A set in which every map
    names one (`padflats`) is reported as a whole and by flat only: pooled over flats it is a
    mixture that describes nothing."""
    n = len(data["visible"])
    out = {"all": np.ones(n, bool)}

    def per_map(key, default):
        values = np.array([key(m) for m in data["maps"]], dtype=object)
        return values[data["map_id"]] if len(values) else np.full(n, default, dtype=object)

    flat = per_map(lambda m: (m.get("options") or {}).get("exit_floor", ""), "")
    if n and (flat != "").all():                            # a pad-flat set: a mixture of flats, only the flats mean anything
        return out | {f"pad flat {name}": flat == name for name in sorted(set(flat.tolist()))}
    look = per_map(lambda m: m.get("look", ""), "")
    for name in ("reference", "random"):
        if (look == name).any():
            out[f"{name} look"] = look == name
    tier = per_map(lambda m: int(m.get("tier", -1)), -1)
    for t in sorted(set(tier.tolist()) - {-1}):
        out[f"tier {t}"] = tier == t
    monster, hurt = has_tag(data["tags"], "monster"), has_tag(data["tags"], "hurt")
    if monster.any():
        out["monster in view"], out["no monster in view"] = monster, ~monster
    if hurt.any():
        out["damage tint"], out["no damage tint"] = hurt, ~hurt
    for slot, name in WEAPON_SLOTS.items():                 # only in sets collected with the weapon recorded
        if (data["weapon"] == slot).any():
            out[f"holding {name}"] = data["weapon"] == slot
    for name in sorted(set(flat.tolist()) - {""}):
        out[f"pad flat {name}"] = flat == name
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


def perturbations() -> dict:
    """Named changes of a float frame batch (B, 3, H, W) in [0, 1], one at a time and at a fixed strength.

    Most sit at the edge of what training draws at random (`augment.AugmentConfig`); "hue +-0.08" and
    "red x0.8" go beyond it. They answer one question: does the detector still find the pad when its
    pixels are not the ones it was trained on?
    """
    import torch.nn.functional as F

    from training.exitdet import augment as A

    def colour(saturation: float, hue: float):
        def apply(x):
            m = A.colour_matrices(torch.tensor([saturation]), torch.tensor([hue]))[0].to(x.device)
            return torch.einsum("ij,bjhw->bihw", m, x)
        return apply

    def gain(r: float, g: float, b: float):
        return lambda x: x * torch.tensor([r, g, b], device=x.device).view(1, 3, 1, 1)

    def contrast(c: float):
        return lambda x: (x - x.mean(dim=(1, 2, 3), keepdim=True)) * c + x.mean(dim=(1, 2, 3), keepdim=True)

    def blur(alpha: float):
        def apply(x):
            kernel = A._BLUR.to(x.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            return x * (1 - alpha) + alpha * F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
        return apply

    def noise(sigma: float):
        return lambda x: x + sigma * torch.randn(x.shape, generator=torch.Generator().manual_seed(0)).to(x.device)

    return {"none": lambda x: x,
            "brightness x0.7": lambda x: x * 0.7, "brightness x1.3": lambda x: x * 1.3,
            "gamma 0.7": lambda x: x.clamp_min(1e-4) ** 0.7, "gamma 1.4": lambda x: x.clamp_min(1e-4) ** 1.4,
            "contrast x0.75": contrast(0.75), "contrast x1.3": contrast(1.3),
            "saturation x0.6": colour(0.6, 0.0), "saturation x1.4": colour(1.4, 0.0),
            "hue +0.04 turn": colour(1.0, 0.04), "hue -0.04 turn": colour(1.0, -0.04),
            "hue +0.08 turn": colour(1.0, 0.08), "hue -0.08 turn": colour(1.0, -0.08),
            "red x0.9": gain(0.9, 1.0, 1.0), "red x0.8": gain(0.8, 1.0, 1.0), "green and blue x1.1": gain(1.0, 1.1, 1.1),
            "noise sigma 0.03": noise(0.03), "blur 0.6": blur(0.6),
            **{f"damage tint {a}": (lambda x, a=a: x * (1.0 - a) + a * torch.tensor([1.0, 0.0, 0.0], device=x.device).view(1, 3, 1, 1))
               for a in (0.3, 0.5, 0.7)}}


@torch.no_grad()
def robustness(model: ExitDetector, data_dir, set_name: str = "heldout", device: str = "cpu", batch: int = 1024,
               max_frames: int | None = None) -> list[dict]:
    """The detection metrics of one set under each of `perturbations()`: one row per perturbation."""
    top = ExitDetector.TOP
    data = load_set(Path(data_dir) / set_name, top=top, max_frames=max_frames)
    model = model.to(device).eval()
    rows = []
    for name, change in perturbations().items():
        logits, pos = [], []
        for i in range(0, len(data["rgb"]), batch):
            x = torch.from_numpy(data["rgb"][i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.pad(change(x).clamp(0.0, 1.0), (0, 0, top, 0))
            out = model(x).float().cpu()
            logits.append(out[:, 0])
            pos.append(out[:, 1:3] * POS_SCALE)
        m = detection_metrics(data, torch.cat(logits).numpy(), torch.cat(pos).numpy())
        rows.append({"set": set_name, "perturbation": name,
                     **{k: m[k] for k in ("frames", "precision", "recall_0_200", "recall_200_400", "recall_0_400", "recall_400_640",
                                          "n_clear_0_400", "recall_clear_0_400",
                                          "fp_per_1000_all", "false_positives", "pos_err_median_0_400", "pos_err_p90_0_400")}})
    model.to("cpu")
    return rows


def gate(rows: list[dict], lat: dict) -> dict:
    """The gate numbers for every (set, subset) row and whether each is met.

    True / False, or None where the row has too few frames to judge (fewer than MIN_GATE_FRAMES
    positives within 400 units, no detection at all, or a set that did not record the cover).
    Recall is judged twice: by the label as it is (a pad counts as visible when only walls are
    considered, so a pad under a monster is a miss) and over pads in clear view.
    """
    def judged(value: float, ok: bool, n: int = MIN_GATE_FRAMES):
        return bool(ok) if np.isfinite(value) and n >= MIN_GATE_FRAMES else None

    passed = {"pass": True, "FAIL": False, "not judged": None}
    out = {}
    for r in rows:
        out[f"{r['set']}/{r['subset']}"] = {
            "precision >= 0.98": judged(r["precision"], r["precision"] >= 0.98, r["detections"]),
            "recall within 400 >= 0.90": passed[r["gate_recall_0_400"]],
            "recall within 400, pad in clear view >= 0.90": passed[r["gate_recall_clear_0_400"]],
            "median position error within 400 <= 32": judged(r["pos_err_median_0_400"], r["pos_err_median_0_400"] <= 32.0,
                                                             r["n_0_400"]),
            "latency <= 1.5 ms": bool(lat["latency_mean_ms"] <= 1.5)}
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


COMPARE_SUBSETS = ("all", "tier 1", "tier 3", "tier 4", "monster in view", "no monster in view", "damage tint", "no damage tint")
COMPARE_COLUMNS = ("frames", "precision", "false_positives", "fp_per_1000_all", "n_0_400", "recall_0_400", "recall_0_400_lo95",
                   "recall_0_400_hi95", "n_clear_0_400", "recall_clear_0_400", "n_covered_0_400", "recall_covered_0_400",
                   "recall_400_640", "pos_err_median_0_400")


def compare_tables(tables: dict, sets=None, subsets=COMPARE_SUBSETS, columns=COMPARE_COLUMNS) -> list[dict]:
    """Rows of several evaluation tables side by side: {label: table.csv} -> one row per (set, subset, label).

    Only (set, subset) pairs present in every table are kept, so two detectors are compared on the same frames."""
    loaded = {}
    for label, path in tables.items():
        with Path(path).open() as f:
            loaded[label] = {(r["set"], r["subset"]): r for r in csv.DictReader(f)}
    first = next(iter(loaded.values()))
    out = []
    for key, row in first.items():
        if key[1] not in subsets or (sets and key[0] not in sets) or not all(key in t for t in loaded.values()):
            continue
        for label, table in loaded.items():
            out.append({"set": key[0], "subset": key[1], "detector": label, "weights_sha256": table[key].get("weights_sha256", ""),
                        **{c: table[key].get(c, "") for c in columns}})
    return out


def summary_lines(rows: list[dict], lat: dict | None = None) -> list[str]:
    out = []
    for r in rows:
        out.append(
            f"{r['set']:>12} {r['subset']:<18} frames {r['frames']:6d}  precision {r['precision']:.4f}  recall 0-200 "
            f"{r['recall_0_200']:.3f}  200-400 {r['recall_200_400']:.3f}  0-400 {r['recall_0_400']:.3f} "
            f"[{r['recall_0_400_lo95']:.3f}, {r['recall_0_400_hi95']:.3f}] (n={r['n_0_400']}, {r['gate_recall_0_400']}); "
            f"clear view {r['recall_clear_0_400']:.3f} (n={r['n_clear_0_400']}, {r['gate_recall_clear_0_400']}); "
            f"covered {r['recall_covered_0_400']:.3f} (n={r['n_covered_0_400']})  400-640 "
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
    ap.add_argument("--sets", default="heldout,unseen,heldout2,unseen2", help="those that exist on disk")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default="", help="write the table here (default: print only)")
    ap.add_argument("--robustness", default="", help="instead: evaluate this set under fixed colour changes, blur and noise")
    ap.add_argument("--compare", nargs="+", default=[], metavar="LABEL=TABLE.csv",
                    help="instead: put rows of tables written earlier side by side (with --csv), e.g. two detectors")
    args = ap.parse_args(argv)
    if args.compare:
        rows = compare_tables(dict(item.split("=", 1) for item in args.compare), [s for s in args.sets.split(",") if s])
        for r in rows:
            print(f"{r['set']:>12} {r['subset']:<18} {r['detector']:<34} precision {r['precision']}  FP/1000 {r['fp_per_1000_all']}  "
                  f"recall 0-400 {r['recall_0_400']} (n={r['n_0_400']})  clear view {r['recall_clear_0_400']} (n={r['n_clear_0_400']})")
        if args.csv:
            print("wrote", write_csv(rows, args.csv))
        return 0
    model, config = load_detector(args.weights)
    if args.robustness:
        rows = robustness(model, args.data, args.robustness, args.device)
        for r in rows:
            print(f"{r['perturbation']:>20}: precision {r['precision']:.4f}  recall 0-200 {r['recall_0_200']:.3f}  200-400 "
                  f"{r['recall_200_400']:.3f}  0-400 {r['recall_0_400']:.3f} (clear view {r['recall_clear_0_400']:.3f})  "
                  f"400-640 {r['recall_400_640']:.3f}  FP/1000 "
                  f"{r['fp_per_1000_all']:.2f}  pos err {r['pos_err_median_0_400']:.1f}/{r['pos_err_p90_0_400']:.1f}")
        if args.csv:
            print("wrote", write_csv(rows, args.csv, {"weights_sha256": file_sha256(args.weights)[:12]}))
        return 0
    rows = evaluate_sets(model, args.data, [s for s in args.sets.split(",") if s and any((Path(args.data) / s).glob("shard_*.npz"))],
                         args.device)
    lat = latency(model)
    print("\n".join(summary_lines(rows, lat)))
    for r in rows:
        if r["subset"] == "all":
            print(f"{r['set']}: false positives per 1,000 negative frames by hard negative: "
                  + ", ".join(f"{t} {r['fp_per_1000_' + t]:.1f} (n={r['n_neg_' + t]})" for t in HARD_NEGATIVES))
    print(json.dumps(gate(rows, lat), indent=1))
    if args.csv:
        extra = {"weights_sha256": file_sha256(args.weights)[:12], **{k: round(v, 4) for k, v in lat.items()}}
        print("wrote", write_csv(rows, args.csv, extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
