"""The single import site for everything the RL core takes from other areas of the repository.

Training-time scaffolding, never imported by the agent.

| What                          | Real module                               | Stand-in (``kind="standin"``)      |
|-------------------------------|-------------------------------------------|------------------------------------|
| Odometry, MemoryMap, Encoder  | ``doomnet.odometry / memory / encode``    | ``training.rl.standins.Stub*``     |
| PolicyNet                     | ``doomnet.model``                         | ``training.rl.standin_model``      |
| static map facts              | ``training.mapmeta.load_meta``            | ``training.rl.standins.basic_meta``|
| environments                  | ``doom_env.make_env``                     | ``training.rl.standins.TOY_ENVS``  |
| map manifest                  | ``manifest.jsonl`` from the map generator | a ``toy:NAME`` string              |
| exit observation (training)   | ``training.exit_label``                   | ``ray_exit_observation`` below     |
| teacher labels                | wave 2 (``training.teacher``)             | the toy environments' own labels   |

Every other file in ``training/rl`` goes through the functions below, so wiring in a real module
(or changing how it is called) is a change to this file only. ``kind="auto"`` uses the real module
when it is importable and the stand-in otherwise. Nothing here imports torch at module level:
rollout workers import this file and must stay light.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training import paths
from doomnet.spec import NUM_RAYS, RAY_RANGE
from training.rl import standins

# ---------------------------------------------------------------------------------------------
# B. odometry -> memory map -> encoder (the per-environment front end)
# ---------------------------------------------------------------------------------------------


def frontend_classes(kind: str = "auto"):
    """(Odometry, MemoryMap, Encoder, name). The three real modules are used together or not at all."""
    if kind in ("auto", "real"):
        try:
            from doomnet.encode import Encoder
            from doomnet.memory import MemoryMap
            from doomnet.odometry import Odometry
            return Odometry, MemoryMap, Encoder, "doomnet"
        except ImportError:
            if kind == "real":
                raise
    return standins.StubOdometry, standins.StubMemoryMap, standins.StubEncoder, "stand-in"


class FrontEnd:
    """Odometry -> memory map -> encoder for one environment: the pipeline the agent runs.

    ``observe`` is called once per observation. It returns the encoder's dict (rays, ents,
    ent_mask, scal), the float32 map crop (MAP_CHANNELS, 64, 64) and the pose estimate
    (x, y, theta) in the start frame.
    """

    def __init__(self, kind: str = "auto"):
        odometry, memory, encoder, self.name = frontend_classes(kind)
        self.odom, self.mem, self.enc = odometry(), memory(), encoder()

    def reset(self) -> None:
        self.odom.reset()
        self.mem.reset()
        self.enc.reset()

    def observe(self, obs: dict, prev_action, step: int, budget_steps: float, exit_obs):
        if step > 0:
            self.odom.update(prev_action, obs)
        else:
            prev_action = None                      # the first observation of an episode has no action
        pose = (self.odom.x, self.odom.y, self.odom.theta)
        self.mem.update(pose, obs, prev_action, exit_obs, step)
        crop = self.mem.crop(pose)
        feats = self.enc.encode(obs, pose, prev_action, step, budget_steps, exit_obs)
        return feats, crop, pose


# ---------------------------------------------------------------------------------------------
# D. policy network
# ---------------------------------------------------------------------------------------------


def make_policy(kind: str = "auto", **kwargs):
    """Build the policy network (``doomnet.model.PolicyNet``) or the stand-in with the same signature."""
    if kind in ("auto", "real"):
        try:
            from doomnet.model import PolicyNet
            return PolicyNet(**kwargs)
        except ImportError:
            if kind == "real":
                raise
    from training.rl.standin_model import StandInPolicy
    return StandInPolicy(**kwargs)


# ---------------------------------------------------------------------------------------------
# C. static map facts for the reward and the exit oracle
# ---------------------------------------------------------------------------------------------
_warned_meta = False


def map_facts(wad: str) -> standins.BasicMeta:
    """Keys and monsters present at skill 3 and exit centres (map units) of the map in ``wad``."""
    global _warned_meta
    try:
        from training.mapmeta import load_meta
    except ImportError:
        return standins.basic_meta(wad)
    try:
        return _adapt_meta(load_meta(wad))
    except Exception as exc:  # an interface mismatch must not stop a training run
        if not _warned_meta:
            warnings.warn(f"training.mapmeta could not be adapted ({exc!r}); using the basic UDMF reader")
            _warned_meta = True
        return standins.basic_meta(wad)


def _adapt_meta(meta) -> standins.BasicMeta:
    """Map ``training.mapmeta.MapMeta`` onto the three facts used here (the full object is kept too)."""
    centres = np.asarray(meta.exit_centres, np.float64).reshape(-1, 2)
    return standins.BasicMeta(int(meta.keys_total), int(meta.monsters_total), centres, meta=meta)


# ---------------------------------------------------------------------------------------------
# Exit observation while training (at evaluation it comes from the detector network)
# ---------------------------------------------------------------------------------------------


def ray_exit_observation(obs: dict, priv: dict, centres: np.ndarray, max_range: float):
    """Fallback exit observation from ground truth: (1, forward / 1024, left / 1024) or None.

    Reports the nearest exit pad that is inside the 90 degree field of view, within ``max_range``
    and not hidden: the ray pointing at it must reach the pad.
    """
    if len(centres) == 0 or "angle" not in priv:
        return None
    heading = math.radians(priv["angle"])
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = centres[:, 0] - priv["x"], centres[:, 1] - priv["y"]
    fwd, left = dx * c + dy * s, -dx * s + dy * c
    dist = np.hypot(fwd, left)
    reach = np.asarray(obs["rays"]["distance"], np.float64) * RAY_RANGE
    for k in np.argsort(dist):
        if dist[k] > max_range:
            break
        if fwd[k] <= 0.0 or abs(left[k]) > fwd[k]:
            continue
        ray = int(round(math.atan2(left[k], fwd[k]) / (2.0 * math.pi / NUM_RAYS))) % NUM_RAYS
        if reach[ray] >= dist[k] - 16.0:
            return np.array([1.0, fwd[k] / RAY_RANGE, left[k] / RAY_RANGE], np.float32)
    return None


def exit_observation(env, facts, obs: dict, priv: dict, rng=None, max_range: float = 640.0):
    """(confidence, forward / 1024, left / 1024) of a visible exit pad, or None.

    Uses ``training.exit_label`` (the definition the detector is trained against) when it and the
    map metadata are available, degraded by its noise model when ``rng`` is given; otherwise the
    ray-based fallback above. Training only: the agent gets these numbers from its detector network.
    """
    meta = getattr(facts, "meta", None)
    if meta is not None and getattr(env, "_geometry", None) is not None:
        try:
            from training import exit_label
        except ImportError:
            exit_label = None
        if exit_label is not None:
            seen = exit_label.exit_observation(env, meta)[:3]
            if rng is not None:
                seen = exit_label.noisy(seen, rng)
            if not seen[0]:
                return None
            return np.array([1.0, seen[1] / RAY_RANGE, seen[2] / RAY_RANGE], np.float32)
    return ray_exit_observation(obs, priv, facts.exit_centres, max_range)


def make_exit_detector(weights: str):
    """rgb (120, 160, 3) uint8 -> (confidence, forward / 1024, left / 1024) from ``doomnet.model.ExitDetector``.

    For the training phase that runs the real detector in the loop (``exit_source="detector"``).
    It runs on the CPU, one frame per call on one thread, as the agent does at evaluation. This is
    the only path on which a rollout worker loads torch.
    """
    import torch
    from doomnet.model import ExitDetector

    torch.set_num_threads(1)
    net = ExitDetector()
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
    net.eval()

    def detect(rgb) -> np.ndarray:
        with torch.inference_mode():
            out = net(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0))[0]
            out[0] = torch.sigmoid(out[0])
        return out.numpy().astype(np.float32)

    return detect


# ---------------------------------------------------------------------------------------------
# Environments and the map manifest
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MapEntry:
    """One line of a map manifest."""

    path: str                 # absolute WAD path, or "toy:NAME"
    tier: int = 0
    episode_ticks: int = 2100
    index: int = 0

    @property
    def is_toy(self) -> bool:
        return self.path.startswith("toy:")


def load_manifest(source: str, default_tier: int = 0, default_ticks: int = 2100) -> list[MapEntry]:
    """Read maps from a ``manifest.jsonl``, a single WAD, a directory of WADs or a ``toy:NAME`` string.

    Manifest lines are JSON objects with at least ``path`` (relative paths are resolved against
    the manifest's directory, then the repository root), ``tier`` and ``episode_ticks``.
    """
    source = str(source)
    if source.startswith("toy:"):
        if source not in standins.TOY_ENVS:
            raise ValueError(f"unknown toy environment {source!r}; choose from {sorted(standins.TOY_ENVS)}")
        return [MapEntry(source, default_tier, default_ticks if default_ticks != 2100 else 128, 0)]
    p = Path(source)
    if p.is_dir():
        return [MapEntry(str(w.resolve()), default_tier, default_ticks, i) for i, w in enumerate(sorted(p.glob("*.wad")))]
    if p.suffix.lower() == ".wad":
        return [MapEntry(str(p.resolve()), default_tier, default_ticks, 0)]
    entries = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = Path(row.get("path") or row["wad"])
        if not raw.is_absolute():
            raw = next((base / raw for base in (p.parent, paths.ROOT, Path.cwd()) if (base / raw).exists()), p.parent / raw)
        entries.append(MapEntry(str(raw.resolve()), int(row.get("tier", default_tier)),
                                int(row.get("episode_ticks", default_ticks)), len(entries)))
    if not entries:
        raise ValueError(f"no maps in {source}")
    return entries


def make_env(entry: MapEntry, render_buffers: bool = False):
    """Construct the environment of a manifest entry (slow for real maps: about 0.15 s each)."""
    if entry.is_toy:
        return standins.TOY_ENVS[entry.path](entry.episode_ticks)
    from doom_env import make_env as make_doom_env
    return make_doom_env(entry.path, episode_ticks=entry.episode_ticks, render_buffers=render_buffers)


def hit_count(env):
    """Engine counter of the player's shots that hit a monster, or None when there is no engine.

    ``privileged_state()`` reports kills but not hits; the ammo term of the reward needs hits.
    """
    game = getattr(env, "_game", None)
    if game is None:
        return None
    import vizdoom as vd
    return float(game.get_game_variable(vd.GameVariable.HITCOUNT))


# ---------------------------------------------------------------------------------------------
# Teacher labels. Until a teacher exists only the toy environments provide labels, and the label
# mask of every real step is zero.
# ---------------------------------------------------------------------------------------------


def make_teacher():
    """A teacher for one environment, or None while there is none.

    Expected interface: ``reset()`` at every episode start and
    ``label(frontend, obs, step) -> (head labels (6,), sub-goal index, target cell index, mask)``,
    reading only the agent's own memory (``frontend.mem``) and pose estimate (``frontend.odom``).
    """
    return None


def teacher_label(teacher, env, frontend: FrontEnd, obs: dict, step: int):
    """(head labels (6,), sub-goal index, target cell index, mask) or None when there is no label."""
    if teacher is not None:
        return teacher.label(frontend, obs, step)
    label = getattr(env, "teacher_label", None)
    return label() if label is not None else None
