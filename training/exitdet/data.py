"""Frame sets of the exit detector on disk: tags, shard writer and loader.

Training-time scaffolding. Never imported by the agent in submission/.

A set is a directory:

    maps/NAME.wad      the compiled maps the frames were taken on
    maps.jsonl         one row per map: name, tier, seed, grid, options, look, edits, frame counts
    shard_000.npy      uint8 (N, 120, 160, 3) frames exactly as the agent receives them
    shard_000.npz      per-frame arrays, see FIELDS
    summary.json       totals written by the collector

Per-frame fields:
    visible    bool     label: the pad centre is in view, within 640 units and in line of sight
    ambiguous  bool     part of a pad may be in the frame without meeting that rule (ignored by the loss)
    pos        float32  (2,) forward and left offset of the pad centre in map units (0 when not visible)
    pad_dist   float32  distance to the nearest pad centre whether visible or not (-1: the map has no pad)
    pad_ego    float32  (2,) forward and left offset of that pad centre, whether visible or not
    pad_los    bool     the line from the player to that pad centre is clear
    pad_px     uint16   pad-coloured pixels inside the screen area a pad in front of the player can cover
    tags       uint32   bit set of TAGS: what else is in the frame
    flash      uint8    strength of the red damage tint (the engine's damage counter, 0-100)
    weapon     uint8    selected weapon slot (1 fist, 2 pistol, 3 shotgun, 4 chaingun)
    kind       uint8    index into KINDS: how the pose was produced
    map_id     int32    row of maps.jsonl
    step       int16    decision step inside its episode
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCREEN_H, SCREEN_W = 120, 160

# What else is in a frame. The first ten are the hard negatives of the evaluation table.
TAGS = ("red_key", "other_key", "red_door", "other_door", "imp_fireball", "other_projectile", "blood", "monster",
        "red_floor", "red_wall", "hurt", "firing")
TAG_BIT = {name: 1 << i for i, name in enumerate(TAGS)}
KINDS = ("exit_view", "approach", "spin", "thing_view", "wander")
KIND_INDEX = {name: i for i, name in enumerate(KINDS)}

FIELDS = {"visible": np.bool_, "ambiguous": np.bool_, "pos": np.float32, "pad_dist": np.float32, "pad_ego": np.float32,
          "pad_los": np.bool_, "pad_px": np.uint16, "tags": np.uint32, "flash": np.uint8, "weapon": np.uint8,
          "kind": np.uint8, "map_id": np.int32, "step": np.int16}
PAIRS = ("pos", "pad_ego")       # fields with two values per frame

# A frame labelled visible counts as "drawn" when enough pad-coloured pixels are where the pad must
# be. Beyond about 450 units the pad covers less than one pixel row and the renderer often draws
# none of it; in a dark room it fades to black long before that; nearer than about 50 units it lies
# below the bottom of the frame. A far pad needs RENDERED_MIN_PX pixels. A near pad covers hundreds,
# and the pistol in front of it has up to four red pixels of its own, so the bar rises with the
# area the pad should cover (a tenth of it, at most RENDERED_NEAR_PX).
RENDERED_MIN_PX = 3
RENDERED_NEAR_PX = 12
PAD_AREA = 80.0 * 64.0 * 96.0 * 41.0 * 64.0      # pixels a pad covers at distance d: PAD_AREA / d**3
HURT_MIN_FLASH = 6               # damage counter from which the red tint is plain to see


def empty_fields(n: int = 0) -> dict:
    return {k: np.zeros((n, 2) if k in PAIRS else (n,), dt) for k, dt in FIELDS.items()}


def has_tag(tags: np.ndarray, name: str) -> np.ndarray:
    return (np.asarray(tags) & TAG_BIT[name]) != 0


def drawn(data: dict) -> np.ndarray:
    """Per frame: is the pad the label calls visible actually in the picture? (False where not visible.)"""
    dist = np.maximum(np.hypot(data["pos"][:, 0], data["pos"][:, 1]), 1.0)
    need = np.clip(0.1 * PAD_AREA / dist**3, RENDERED_MIN_PX, RENDERED_NEAR_PX)
    return np.asarray(data["visible"], bool) & (data["pad_px"] >= need)


class ShardWriter:
    """Appends frames to numbered shards of at most `shard_frames` frames."""

    def __init__(self, directory, shard_frames: int = 4096):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in list(self.dir.glob("shard_*.npy")) + list(self.dir.glob("shard_*.npz")):
            old.unlink()
        self.shard_frames = int(shard_frames)
        self.frames: list[np.ndarray] = []
        self.fields: list[dict] = []
        self.pending = 0
        self.shards = 0
        self.total = 0

    def add(self, rgb: np.ndarray, fields: dict) -> None:
        if not len(rgb):
            return
        assert rgb.dtype == np.uint8 and rgb.shape[1:] == (SCREEN_H, SCREEN_W, 3), rgb.shape
        self.frames.append(rgb)
        self.fields.append({k: np.asarray(fields[k], dt) for k, dt in FIELDS.items()})
        self.pending += len(rgb)
        self.total += len(rgb)
        while self.pending >= self.shard_frames:
            self._flush(self.shard_frames)

    def _flush(self, n: int) -> None:
        rgb = np.concatenate(self.frames)
        fields = {k: np.concatenate([f[k] for f in self.fields]) for k in FIELDS}
        np.save(self.dir / f"shard_{self.shards:03d}.npy", rgb[:n])
        np.savez(self.dir / f"shard_{self.shards:03d}.npz", **{k: v[:n] for k, v in fields.items()})
        self.shards += 1
        self.frames = [rgb[n:]] if len(rgb) > n else []
        self.fields = [{k: v[n:] for k, v in fields.items()}] if len(rgb) > n else []
        self.pending = len(rgb) - n

    def close(self) -> None:
        if self.pending:
            self._flush(self.pending)


def set_frames(directory) -> int:
    """Number of frames stored in a set."""
    return sum(len(np.load(p)["visible"]) for p in sorted(Path(directory).glob("shard_*.npz")))


def load_set(directory, top: int = 0, max_frames: int | None = None, with_rgb: bool = True) -> dict:
    """A whole set in memory: the per-frame fields plus `rgb` uint8 (N, 120 - top, 160, 3).

    `top` drops image rows the detector does not read (it reads rows from ExitDetector.TOP on), which
    keeps a 100,000-frame set at 3.5 GB. `maps` holds the rows of maps.jsonl.
    """
    directory = Path(directory)
    shards = sorted(directory.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no frames in {directory}: run python -m training.exitdet.collect first")
    fields, counts = [], []
    for p in shards:
        with np.load(p) as z:
            fields.append({k: z[k] for k in FIELDS})
        counts.append(len(fields[-1]["visible"]))
    total = sum(counts) if max_frames is None else min(sum(counts), int(max_frames))
    out = {k: np.concatenate([f[k] for f in fields])[:total] for k in FIELDS}
    if with_rgb:
        rgb = np.empty((total, SCREEN_H - top, SCREEN_W, 3), np.uint8)
        at = 0
        for p, n in zip(shards, counts):
            if at >= total:
                break
            n = min(n, total - at)
            rgb[at:at + n] = np.load(p.with_suffix(".npy"), mmap_mode="r")[:n, top:]
            at += n
        out["rgb"] = rgb
    maps_file = directory / "maps.jsonl"
    out["maps"] = [json.loads(line) for line in maps_file.read_text().splitlines() if line.strip()] if maps_file.exists() else []
    return out


def describe(data: dict) -> dict:
    """Counts that characterise a loaded set: frames, positives by distance band, ambiguous, tags, kinds."""
    vis, amb = data["visible"], data["ambiguous"]
    dist = np.hypot(data["pos"][:, 0], data["pos"][:, 1])
    n = len(vis)
    seen = drawn(data)
    out = {"frames": int(n), "visible": int(vis.sum()), "ambiguous": int(amb.sum()),
           "rendered": int(seen.sum()), "rendered_0_200": int((seen & (dist < 200)).sum()),
           "rendered_200_400": int((seen & (dist >= 200) & (dist < 400)).sum()),
           "rendered_400_640": int((seen & (dist >= 400)).sum()),
           "negative": int((~vis & ~amb).sum()), "visible_share": float(vis.mean()) if n else 0.0,
           "visible_0_200": int((vis & (dist < 200)).sum()), "visible_200_400": int((vis & (dist >= 200) & (dist < 400)).sum()),
           "visible_400_640": int((vis & (dist >= 400)).sum()),
           "maps": int(len(np.unique(data["map_id"])))}
    neg = ~vis & ~amb
    for name in TAGS:
        out[f"tag_{name}"] = int(has_tag(data["tags"], name).sum())
        out[f"negative_{name}"] = int((has_tag(data["tags"], name) & neg).sum())
    for i, name in enumerate(KINDS):
        out[f"kind_{name}"] = int((data["kind"] == i).sum())
    return out
