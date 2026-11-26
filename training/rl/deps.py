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
| teacher labels                | ``training.teacher.Teacher``              | the toy environments' own labels   |

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
        self.odom_errors = 0            # odometry updates that raised (the pose estimate was kept), since construction
        self.last_odom_error = ""

    def reset(self) -> None:
        self.odom.reset()
        self.mem.reset()
        self.enc.reset()

    def observe(self, obs: dict, prev_action, step: int, budget_steps: float, exit_obs):
        if step > 0:
            try:
                self.odom.update(prev_action, obs)
            except Exception as exc:                # noqa: BLE001 - one bad step must not end a training run:
                self.odom_errors += 1               # keep the last pose estimate and carry on
                self.last_odom_error = f"{type(exc).__name__}: {exc}"
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


def set_head_bias(policy, biases: dict) -> None:
    """Add a starting offset to the output biases of chosen controller heads.

    biases: head name (``spec.HEAD_NAMES``) -> one offset per option of that head, for example
    ``{"weapon": [1.5, -1.5]}``. The network's own initialisation makes every head near uniform,
    so an untrained policy presses "next weapon" on half of all steps; the weapon is then almost
    never ready to fire and shots that could earn reward are rare. An offset is only a starting
    point: the bias stays an ordinary trained parameter. Applies to ``PolicyNet``, whose last actor
    layer holds the six heads side by side; a network without that layer (the stand-in used by
    the toy tests) is left unchanged.
    """
    actor = getattr(policy, "actor", None)
    if not biases or actor is None:
        return
    import torch
    from doomnet import spec

    layer = actor[-1]
    if getattr(layer, "out_features", None) != sum(spec.HEAD_SIZES):
        raise ValueError("the policy's last actor layer does not hold the six controller heads")
    offsets = np.cumsum((0,) + tuple(spec.HEAD_SIZES))
    with torch.no_grad():
        for name, values in biases.items():
            head = spec.HEAD_NAMES.index(name)
            if len(values) != spec.HEAD_SIZES[head]:
                raise ValueError(f"head {name!r} has {spec.HEAD_SIZES[head]} options, got {len(values)} offsets")
            layer.bias[offsets[head]:offsets[head + 1]] += torch.tensor(values, dtype=layer.bias.dtype, device=layer.bias.device)


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


_RAY_ANGLES = np.arange(NUM_RAYS) * (2.0 * math.pi / NUM_RAYS)      # ray k points at angle k * 2 pi / 64, left positive
_RAY_COS, _RAY_SIN = np.cos(_RAY_ANGLES), np.sin(_RAY_ANGLES)


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
        ray = int(np.argmax(_RAY_COS * fwd[k] + _RAY_SIN * left[k]))      # the ray pointing most nearly at the pad
        if reach[ray] >= dist[k] - 16.0:
            return np.array([1.0, fwd[k] / RAY_RANGE, left[k] / RAY_RANGE], np.float32)
    return None


_exit_label = None            # training.exit_label, imported on first use (False when it is missing)


def _exit_label_module():
    global _exit_label
    if _exit_label is None:
        try:
            from training import exit_label
            _exit_label = exit_label
        except ImportError:
            _exit_label = False
    return _exit_label


def _pad_may_be_visible(label, pads: np.ndarray, priv: dict) -> bool:
    """Cheap necessary condition for ``exit_label.exit_observation`` to report a visible pad.

    True when some pad centre is in front of the player, inside the label's view cone and range
    (each with a small margin, so this never rejects a pad the label would accept). Most steps
    fail it, and the label's line-of-sight test over the level geometry is then skipped.
    """
    heading = math.radians(priv["angle"])
    c, s = math.cos(heading), math.sin(heading)
    px, py = priv["x"], priv["y"]
    far, near, tan = (label.MAX_RANGE + 1.0) ** 2, max(label.MIN_RANGE - 1.0, 0.0) ** 2, label.TAN_VISIBLE * 1.001
    for ex, ey in pads:
        dx, dy = ex - px, ey - py
        fwd, left = dx * c + dy * s, dy * c - dx * s
        if fwd > 0.0 and abs(left) <= fwd * tan + 1e-6 and near <= fwd * fwd + left * left <= far:
            return True
    return False


def exit_observation(env, facts, obs: dict, priv: dict, rng=None, max_range: float = 640.0):
    """(confidence, forward / 1024, left / 1024) of a visible exit pad, or None.

    Uses ``training.exit_label`` (the definition the detector is trained against) when it and the
    map metadata are available, degraded by its noise model when ``rng`` is given; otherwise the
    ray-based fallback above. Training only: the agent gets these numbers from its detector network.
    """
    meta = getattr(facts, "meta", None)
    label = _exit_label_module() if (meta is not None and getattr(env, "_geometry", None) is not None) else False
    if label:
        pads = getattr(facts, "pads", None)
        if pads is None:                                # centres of the exit pads, cached on the map's facts
            pads = facts.pads = [tuple(map(float, e.centre)) for e in meta.exits if e.sector >= 0]
        seen = (False, 0.0, 0.0)
        if pads and "angle" in priv and _pad_may_be_visible(label, pads, priv):
            seen = label.exit_observation(env, meta)[:3]
        if rng is not None:
            seen = label.noisy(seen, rng)
        if not seen[0]:
            return None
        return np.array([1.0, seen[1] / RAY_RANGE, seen[2] / RAY_RANGE], np.float32)
    return ray_exit_observation(obs, priv, facts.exit_centres, max_range)


def detector_state_dict(payload) -> dict:
    """The state dict inside a loaded exit-detector file.

    Accepted layouts: ``{"state_dict", "config"}`` (the committed ``submission/weights/exit_detector.pt``,
    written by the agent's ``save_weights``), ``{"model": ...}`` (older training files) and a bare
    state dict.
    """
    if isinstance(payload, dict):
        for key in ("state_dict", "model"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


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
    try:
        payload = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:                               # noqa: BLE001 - an older file that holds more than tensors
        payload = torch.load(weights, map_location="cpu", weights_only=False)
    net.load_state_dict(detector_state_dict(payload))
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
    teach: bool = True        # False: the navigation teacher gives no labels on this map (see start_on_grid)

    @property
    def is_toy(self) -> bool:
        return self.path.startswith("toy:")


def start_on_grid(options) -> bool:
    """True when a map's build options leave the player start on the 64-unit grid, facing along an axis.

    The navigation teacher plans on the 64-unit cells of the agent's start frame. A start that is
    moved inside its cell (``start_offset``) or turned by anything but a multiple of 90 degrees
    (``start_angle``) puts the walls out of line with those cells and the teacher's labels become
    wrong, so such maps are trained without labels.
    """
    options = options or {}
    offset = options.get("start_offset")
    if offset is not None and any(abs(float(v)) > 1e-6 for v in np.ravel(offset)):
        return False
    angle = options.get("start_angle")
    if angle is None:
        return True
    quarter = float(angle) % 90.0
    return min(quarter, 90.0 - quarter) < 1e-6


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
                                int(row.get("episode_ticks", default_ticks)), len(entries),
                                bool(row.get("teach", True)) and start_on_grid(row.get("options"))))
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
# Teacher labels: the navigation teacher of ``training.teacher`` on real maps, the toy
# environments' own labels otherwise.
# ---------------------------------------------------------------------------------------------
N_HEADS = 6                                         # len(spec.HEAD_SIZES)
FIELD_CAP = 255                                     # travel steps stored per cell: 255 = no route (the teacher's cap)
_ALL_HEADS = np.ones(N_HEADS, bool)
_NO_HEADS = np.zeros(N_HEADS, bool)


def make_teacher(kind: str = "real", **kwargs):
    """A navigation teacher for one environment, or None.

    kind: "real" = ``training.teacher.Teacher(**kwargs)``; "auto" = the same when the module can be
    imported, else None; "none" = no teacher. The teacher decides only from what the student network
    receives: ``reset()`` at every episode start, then ``teacher_label`` below once per observation.
    """
    if kind == "none":
        return None
    try:
        from training.teacher import Teacher
    except ImportError:
        if kind == "real":
            raise
        return None
    return Teacher(**kwargs)


def teacher_label(teacher, env, obs: dict, crop, pose, scal, step: int, budget_steps: float, prev_action):
    """One step's labels, or None when there are none.

    Returns (heads (6,), head_mask (6,) bool, sub-goal index, target cell index, manager_ok, field):

    * ``head_mask`` says which controller heads carry a label. The navigation teacher never labels
      attack and weapon, and labels no head at all while an enemy is visible (sub-goal "fight":
      fighting is learned from reward alone).
    * ``manager_ok``: the sub-goal and target are labels (False after an internal error of the teacher).
    * ``field``: (64, 64) travel steps from every crop cell to the target, capped at ``FIELD_CAP``,
      in the frame of the current crop; None when the teacher has no valid field.

    The teacher is called with exactly what its interface specifies: the observation, the memory
    crop and the pose estimate after the front end processed that observation, the encoder's scalar
    vector, the step index, the assumed budget and the buttons actually executed on the previous
    step (None at step 0). With no teacher, a toy environment's own ``teacher_label()`` is used.
    """
    if teacher is None:
        own = getattr(env, "teacher_label", None)
        if own is None:
            return None
        heads, subgoal, target, ok = own()
        if not ok:
            return None
        return np.asarray(heads), _ALL_HEADS, int(subgoal), int(target), True, None
    errors = getattr(teacher, "errors", 0)
    label = teacher.label(obs, crop, pose, scal, step, budget_steps, prev_action)
    if getattr(teacher, "errors", 0) != errors:     # the teacher caught an internal error: nothing to learn from
        return None
    head_mask = np.asarray(label.head_mask, bool)
    return (np.asarray(label.heads), head_mask if head_mask.shape == (N_HEADS,) else _NO_HEADS, int(label.subgoal),
            int(label.target), True, label.value_field if label.field_valid else None)


# ---------------------------------------------------------------------------------------------
# Forcing the manager's choice on some rows only
# ---------------------------------------------------------------------------------------------


def enable_partial_forcing(policy) -> bool:
    """Let ``forced`` entries of -1 mean "this row chooses for itself". Returns True when supported.

    ``PolicyNet.forward(forced=...)`` replaces the manager's choice on every manager row of the
    batch. The training loop forces the teacher's sub-goal and target on some rows (with
    probability beta) and lets the others sample, in one forward pass. For that the network's
    choice function is wrapped on this one object: rows whose forced value is negative keep the
    network's own choice. Everything downstream (the value field and the controller) is then
    conditioned on the choice in force, forced or not, exactly as in an unwrapped pass. A network
    that handles -1 itself says so with a ``partial_forcing`` attribute; one with neither is left
    alone and the collector falls back to two passes.
    """
    if getattr(policy, "partial_forcing", False):
        return True
    inner = getattr(policy, "_choose", None)
    if inner is None:
        return False
    import torch

    def choose(logits, forced, sample):
        if forced is None:
            return inner(logits, None, sample)
        forced = forced.long()
        keep = forced >= 0
        if bool(keep.all()):
            return forced
        return torch.where(keep, forced, inner(logits, None, sample))

    policy._choose = choose
    policy.partial_forcing = True
    return True
