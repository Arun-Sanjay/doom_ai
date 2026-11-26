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
    pad_cover  uint8    percent of that pad's square hidden behind the sprite of a monster or a corpse
    tags       uint32   bit set of TAGS: what else is in the frame
    flash      uint8    strength of the red damage tint (the engine's damage counter, 0-100)
    weapon     uint8    selected weapon slot (1 fist, 2 pistol, 3 shotgun, 4 chaingun)
    kind       uint8    index into KINDS: how the pose was produced
    map_id     int32    row of maps.jsonl
    step       int16    decision step inside its episode

`load_set` adds one more per-frame array that is not stored: `cover_known` (bool), True where the
frame's shard recorded `pad_cover`. The first sets were collected before it existed; for their
frames nothing can be said about a monster in front of the pad.
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
KINDS = ("exit_view", "approach", "spin", "thing_view", "wander", "play")
KIND_INDEX = {name: i for i, name in enumerate(KINDS)}

FIELDS = {"visible": np.bool_, "ambiguous": np.bool_, "pos": np.float32, "pad_dist": np.float32, "pad_ego": np.float32,
          "pad_los": np.bool_, "pad_px": np.uint16, "pad_cover": np.uint8, "tags": np.uint32, "flash": np.uint8,
          "weapon": np.uint8, "kind": np.uint8, "map_id": np.int32, "step": np.int16}
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
# The label's line of sight is cut by walls and doors only. A monster standing in front of the pad,
# or a corpse lying on it, hides it just as well, and its red pixels then pass for pad pixels. A
# pad hidden by this share or more is neither a positive nor a negative for training.
COVER_MAX = 40


def empty_fields(n: int = 0) -> dict:
    return {k: np.zeros((n, 2) if k in PAIRS else (n,), dt) for k, dt in FIELDS.items()}


def has_tag(tags: np.ndarray, name: str) -> np.ndarray:
    return (np.asarray(tags) & TAG_BIT[name]) != 0


def drawn(data: dict) -> np.ndarray:
    """Per frame: is the pad the label calls visible actually in the picture? (False where not visible.)"""
    dist = np.maximum(np.hypot(data["pos"][:, 0], data["pos"][:, 1]), 1.0)
    need = np.clip(0.1 * PAD_AREA / dist**3, RENDERED_MIN_PX, RENDERED_NEAR_PX)
    return np.asarray(data["visible"], bool) & (data["pad_px"] >= need)


def clear(data: dict) -> np.ndarray:
    """Per frame: the pad is visible by the label, drawn, and not hidden behind a monster or a corpse.

    Sets collected before `pad_cover` was recorded count every drawn pad as clear."""
    return drawn(data) & (data["pad_cover"] < COVER_MAX)


def cover_known(data: dict) -> np.ndarray:
    """Per frame: did the collector record `pad_cover`? (True for data that was not loaded from shards.)"""
    known = data.get("cover_known")
    return np.ones(len(data["visible"]), bool) if known is None else np.asarray(known, bool)


def pad_at_edge(data: dict) -> np.ndarray:
    """Per frame: labelled neither visible nor ambiguous, yet a clear patch of pad colour is where the pad must be.

    The label's "ambiguous" test looks along the lines to the pad's corners, and the corners of a pad
    in an alcove lie on its walls, so a pad that pokes into the edge of the frame is sometimes
    labelled plainly "not visible". Such frames are left out of the visibility loss (the detector is
    not told that a pad it can see is absent); the evaluation still counts them as negatives and
    reports how many of the false detections fall on them. Where the collector stored it, the line to
    the pad centre must be clear; in older shards the frame must show nothing else that is red and
    tagged (a damage tint, blood, a red key, door, floor or wall also put red pixels there).
    """
    base = (~np.asarray(data["visible"], bool) & ~np.asarray(data["ambiguous"], bool) & (data["pad_px"] >= RENDERED_NEAR_PX)
            & (data["pad_dist"] > 0.0) & (data["pad_dist"] <= 900.0))
    recorded = data["weapon"] > 0                      # shards written since pad_los exists
    return base & np.where(recorded, data["pad_los"], data["tags"] == 0)


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


def _complete(fields: dict) -> dict:
    """Fill the fields a shard written by an earlier collector does not have.

    `pad_ego`, `pad_los`, `weapon` and `pad_cover` were added after the first sets were collected
    (`pad_cover` is then 0: nothing is known to hide the pad). For a frame
    labelled visible the nearest pad is the labelled one (so `pad_ego` = `pos`, line of sight clear);
    for the others the offset is unknown and left at zero with no line of sight, which only means
    such a frame never gets the optional position target of an ambiguous frame. Weapon 0 = unknown.
    """
    n = len(fields["visible"])
    vis = np.asarray(fields["visible"], bool)
    if "pad_ego" not in fields:
        fields["pad_ego"] = np.where(vis[:, None], fields["pos"], 0.0).astype(np.float32)
    if "pad_los" not in fields:
        fields["pad_los"] = vis.copy()
    if "weapon" not in fields:
        fields["weapon"] = np.zeros(n, np.uint8)
    if "pad_cover" not in fields:
        fields["pad_cover"] = np.zeros(n, np.uint8)
    return {k: np.asarray(fields[k], dt) for k, dt in FIELDS.items()}


def set_frames(directory) -> int:
    """Number of frames stored in a set."""
    return sum(len(np.load(p)["visible"]) for p in sorted(Path(directory).glob("shard_*.npz")))


def load_set(directory, top: int = 0, max_frames: int | None = None, with_rgb: bool = True, into=None) -> dict:
    """A whole set in memory: the per-frame fields plus `rgb` uint8 (N, 120 - top, 160, 3).

    `top` drops image rows the detector does not read (it reads rows from ExitDetector.TOP on), which
    keeps a 100,000-frame set at 3.5 GB. `maps` holds the rows of maps.jsonl. `into`: an array of the
    right shape to read the frames into (a slice of a larger one, when several sets are joined).
    """
    directory = Path(directory)
    shards = sorted(directory.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no frames in {directory}: run python -m training.exitdet.collect first")
    fields, counts, known = [], [], []
    for p in shards:
        with np.load(p) as z:
            had_cover = "pad_cover" in z.files
            fields.append(_complete({k: z[k] for k in z.files if k in FIELDS}))
        counts.append(len(fields[-1]["visible"]))
        known.append(np.full(counts[-1], had_cover, bool))
    total = sum(counts) if max_frames is None else min(sum(counts), int(max_frames))
    out = {k: np.concatenate([f[k] for f in fields])[:total] for k in FIELDS}
    out["cover_known"] = np.concatenate(known)[:total]
    if with_rgb:
        rgb = np.empty((total, SCREEN_H - top, SCREEN_W, 3), np.uint8) if into is None else into
        assert rgb.shape == (total, SCREEN_H - top, SCREEN_W, 3), (rgb.shape, total)
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
