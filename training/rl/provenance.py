"""Ties a checkpoint to the code that trained it.

Training-time scaffolding, never imported by the agent.

A state dict records parameters, not the function they parameterise. When the forward pass of the
network (or the front end that produces its inputs) is edited without renaming a parameter, an
older checkpoint still loads without error and then plays differently from what was trained.
So every checkpoint and every exported weights file carries a *provenance record*:

    "code"       sha256 of the source files that define the function (``SOURCE_FILES``)
    "probe_file" sha256 of the probe episodes the two entries below were computed on
    "network"    (episodes, steps, NETWORK_FEATS) what the network answered on the probe episodes
    "frontend"   (episodes * steps, FRONTEND_FEATS) a summary of what odometry -> memory map ->
                 encoder produced from the probe episodes' raw observations (None for the stand-in)

The probe episodes (``probe_episodes.npz``, next to this file) are two short recordings from real
maps: the raw observations and executed buttons of 40 decision steps each, and the network inputs
the front end made of them. Loading a checkpoint recomputes both entries with the code as it is
now and compares them with the stored ones (``compare``). A difference means the file no longer
describes the same policy, and loading is refused unless the caller asks for it (``enforce``).

    python -m training.rl.provenance FILE [FILE ...]     # check checkpoints or exported weights
    python -m training.rl.provenance --record            # write the probe episodes (once; see record_probe)

No torch at module level: rollout workers import this file for the front-end summary.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from training import paths
from doomnet import spec
from training.rl import deps

VERSION = 1
# The files whose text defines what a state dict means: the network, and the code that turns
# observations into its inputs.
SOURCE_FILES = ("model.py", "encode.py", "memory.py", "odometry.py", "spec.py")
SOURCE_DIR = paths.ROOT / "submission" / "doomnet"
PROBE_FILE = Path(__file__).resolve().parent / "probe_episodes.npz"
ALLOW_ENV = "DOOMRL_ALLOW_MISMATCH"            # set to 1 to load mismatching files anyway (deliberate migrations)

PROBE_STEPS = 40
# Manager choices forced on the probe's manager steps, so that no discrete choice of the network
# (which could flip on a rounding difference between machines) feeds into the compared numbers.
_PROBE_TARGETS = ((32, 36), (27, 32), (35, 29), (38, 38), (30, 39))      # (row, column) of the crop
_PROBE_CELLS = (0, 32 * 64 + 32, 20 * 64 + 45, 4095)                     # cells whose target log-probability is recorded
NETWORK_FEATS = sum(spec.HEAD_SIZES) + 2 + spec.NUM_SUBGOALS + len(_PROBE_CELLS) + 2
FRONTEND_FEATS = 5 + spec.MAP_CHANNELS
# Tolerances. Network outputs are recomputed on the CPU in float32; the same file on another machine
# or BLAS differs by about 1e-6 (measured: 6e-7 between 1 and 8 threads). Map sums are in cells (a cell is stored as 0..255).
NET_ATOL, NET_RTOL = 5e-4, 5e-4
FRONT_ATOL, FRONT_RTOL, FRONT_MAP_ATOL = 1e-3, 1e-4, 0.05


class CheckpointMismatch(RuntimeError):
    """A checkpoint or weights file does not describe the same policy under the current code."""


# ---------------------------------------------------------------------------------------------
# Source fingerprint and probe episodes
# ---------------------------------------------------------------------------------------------
def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_fingerprint() -> dict:
    """sha256 of each file in ``SOURCE_FILES`` as it is on disk now ("" for a missing file)."""
    return {name: (_sha256(SOURCE_DIR / name) if (SOURCE_DIR / name).is_file() else "") for name in SOURCE_FILES}


@functools.lru_cache(maxsize=1)
def probe() -> dict:
    """The probe episodes: a dict of arrays shaped (episodes, PROBE_STEPS, ...), plus "sha256" of the file."""
    with np.load(PROBE_FILE) as data:
        out = {k: data[k] for k in data.files}
    out["sha256"] = _sha256(PROBE_FILE)
    return out


def _observation(p: dict, e: int, t: int) -> dict:
    """The raw observation of probe step (e, t) in the wrapper's layout (without the frame buffers)."""
    return {"vitals": p["vitals"][e, t],
            "rays": {"distance": p["distance"][e, t], "hit_type": p["hit_type"][e, t]},
            "visible_entities": {"features": p["features"][e, t], "entity_type": p["entity_type"][e, t],
                                 "mask": p["entity_mask"][e, t]}}


def encode_step(frontend, obs: dict, prev_action, step: int, budget: float, exit_obs) -> dict:
    """One observation through the front end, in the layout of the rollout workers' rows (map crop as uint8)."""
    feats, crop, _ = frontend.observe(obs, prev_action, step, budget, exit_obs)
    rays, ents = obs["rays"], obs["visible_entities"]
    return {"rays": feats["rays"].copy(), "ents": feats["ents"].copy(), "ent_mask": np.asarray(feats["ent_mask"], bool).copy(),
            "scal": feats["scal"].copy(), "map": np.clip(np.rint(crop * 255.0), 0.0, 255.0).astype(np.uint8),
            "masks": np.array(spec.action_masks(rays["distance"], rays["hit_type"], ents["entity_type"], ents["mask"],
                                                obs["vitals"][2]), bool)}


def replay_frontend(kind: str = "real") -> dict | None:
    """Run the current front end over the probe's raw observations; None when ``kind`` gives the stand-in.

    Returns the network inputs it produces, each shaped (episodes, PROBE_STEPS, ...).
    """
    if deps.frontend_classes(kind)[3] != "doomnet":
        return None
    p = probe()
    episodes = []
    for e in range(p["vitals"].shape[0]):
        frontend = deps.FrontEnd(kind)
        frontend.reset()
        steps = []
        for t in range(p["vitals"].shape[1]):
            exit_obs = None if np.isnan(p["exit_obs"][e, t, 0]) else p["exit_obs"][e, t]
            steps.append(encode_step(frontend, _observation(p, e, t), p["action"][e, t - 1] if t else None, t,
                                     float(p["budget"]), exit_obs))
        episodes.append({k: np.stack([s[k] for s in steps]) for k in steps[0]})
    return {k: np.stack([ep[k] for ep in episodes]) for k in episodes[0]}


_SCAL_WEIGHTS = np.stack([np.cos(np.arange(spec.SCALAR_DIM)), np.sin(1.7 * np.arange(spec.SCALAR_DIM))]).astype(np.float64)


def summarise_inputs(inputs: dict) -> np.ndarray:
    """(episodes * steps, FRONTEND_FEATS) float32: per step, sums of rays, entities and the entity mask,
    two fixed projections of the scalar vector, and the number of set cells per map channel."""
    n = inputs["scal"].shape[0] * inputs["scal"].shape[1]
    flat = lambda k: inputs[k].reshape(n, -1).astype(np.float64)
    maps = inputs["map"].reshape(n, spec.MAP_CHANNELS, -1).astype(np.float64).sum(-1) / 255.0
    return np.concatenate([flat("rays").sum(1, keepdims=True), flat("ents").sum(1, keepdims=True),
                           flat("ent_mask").sum(1, keepdims=True), flat("scal") @ _SCAL_WEIGHTS.T, maps], 1).astype(np.float32)


@functools.lru_cache(maxsize=4)
def frontend_signature(kind: str = "real"):
    """Summary of what the front end loaded in this process makes of the probe episodes (None for the stand-in).

    Computed once per process: it describes the code as imported, whatever happens to the files later.
    """
    inputs = replay_frontend(kind)
    return None if inputs is None else summarise_inputs(inputs)


# ---------------------------------------------------------------------------------------------
# What the network answers on the probe
# ---------------------------------------------------------------------------------------------
def network_signature(policy) -> "torch.Tensor":
    """(episodes, PROBE_STEPS, NETWORK_FEATS) float32: the network's outputs on the probe episodes.

    A copy of the network is rebuilt on the CPU from its constructor arguments and state dict (the
    way ``load_policy`` and the agent rebuild it) and run step by step from the initial state with
    the manager schedule of an episode. The manager's choices are forced to fixed values, so every
    recorded number is a continuous function of the weights. Per step: the logits of the six heads
    (unmasked), both values, the sub-goal logits, the log-probability of four fixed target cells,
    and the mean and standard deviation of the value field.
    """
    import torch

    p = probe()
    kind = "real" if type(policy).__name__ == "PolicyNet" else "standin"
    net = deps.make_policy(kind, **dict(getattr(policy, "config", {}) or {}))
    net.load_state_dict({k: v.detach().to("cpu") for k, v in policy.state_dict().items()})
    net.eval()
    inputs = {k: torch.from_numpy(p[k]) for k in ("rays", "ents", "ent_mask", "scal", "map")}
    episodes, steps = p["scal"].shape[:2]
    cells = torch.tensor(_PROBE_CELLS)
    rows = []
    with torch.no_grad():
        state = net.initial_state(episodes, torch.device("cpu"))
        for t in range(steps):
            m = t // spec.MANAGER_PERIOD
            row, col = _PROBE_TARGETS[m % len(_PROBE_TARGETS)]
            forced = {"subgoal": torch.tensor([(m + 3 * e) % spec.NUM_SUBGOALS for e in range(episodes)]),
                      "target": torch.full((episodes,), row * spec.CROP_SIZE + col)}
            run = torch.full((episodes,), t % spec.MANAGER_PERIOD == 0)
            out, state = net(inputs["rays"][:, t], inputs["ents"][:, t], inputs["ent_mask"][:, t], inputs["scal"][:, t],
                             inputs["map"][:, t], state=state, masks=None, run_manager=run, forced=forced, sample=False)
            field_ = out["value_field"].flatten(1).float()
            rows.append(torch.cat([torch.cat(list(out["logits"]), 1).float(), out["value"][:, None].float(),
                                   out["value_int"][:, None].float(), out["subgoal_logits"].float(),
                                   torch.log_softmax(out["target_logits"].float(), 1)[:, cells],
                                   field_.mean(1, keepdim=True), field_.std(1, keepdim=True)], 1))
    return torch.stack(rows, 1)


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
def stamp(policy, frontend="auto", code: dict | None = None) -> dict:
    """The provenance record of ``policy`` as it is now (plain containers and tensors only).

    frontend: the front-end summary of the processes that produced the training data (from the
    rollout workers), None when they ran the stand-in, or "auto" to compute it in this process.
    code: the source fingerprint taken when the code was imported (default: the files as they are now).
    Never raises: a part that cannot be computed is stored as None with the reason under "error".
    """
    import torch

    record = {"version": VERSION, "code": dict(code or {}), "probe_file": "", "network": None, "frontend": None,
              "written": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": str(torch.__version__),
              "platform": f"{sys.platform}-{platform.machine()}", "error": ""}
    try:
        record["code"] = record["code"] or code_fingerprint()
        record["probe_file"] = probe()["sha256"]
        record["network"] = network_signature(policy)
        if isinstance(frontend, str):
            frontend = frontend_signature("auto" if frontend == "auto" else frontend)
        if frontend is not None:
            record["frontend"] = torch.as_tensor(np.asarray(frontend, np.float32))
    except Exception as exc:                            # noqa: BLE001 - a checkpoint must still be written
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return record


def to_json(record: dict | None) -> dict | None:
    """The record with its tensors as nested lists (the agent's weights file keeps a JSON-able config)."""
    if record is None:
        return None
    out = dict(record)
    for key in ("network", "frontend"):
        if out.get(key) is not None:
            out[key] = [[float(f"{v:.7g}") for v in row] for row in _host(out[key]).reshape(-1, np.shape(out[key])[-1])]
    return out


def _host(values) -> np.ndarray:
    """A stored signature (tensor or nested lists) as a float64 array in host memory.

    A checkpoint is loaded with ``map_location=`` the learner's device, which moves the record's
    tensors to the GPU along with the weights; numpy cannot read them there.
    """
    if hasattr(values, "detach"):
        values = values.detach().to("cpu")
    return np.asarray(values, np.float64)


def record_of(payload) -> dict | None:
    """The provenance record inside a loaded checkpoint or exported weights file, or None."""
    if not isinstance(payload, dict):
        return None
    for holder in (payload, payload.get("config") if isinstance(payload.get("config"), dict) else {}):
        if isinstance(holder.get("provenance"), dict):
            return holder["provenance"]
    return None


@dataclass
class Verdict:
    """Result of ``compare``. status: "ok", "unverified" (nothing to compare with) or "mismatch"."""

    status: str = "ok"
    problems: list = field(default_factory=list)       # what differs (status "mismatch") or is missing ("unverified")
    notes: list = field(default_factory=list)          # harmless differences worth one line


def _worst(now: np.ndarray, then: np.ndarray, atol, rtol) -> tuple[float, int]:
    """Largest |now - then| in units of its tolerance, and how many entries exceed it."""
    excess = np.abs(now - then) / (atol + rtol * np.abs(then))
    return float(excess.max()), int((excess > 1.0).sum())


def compare(record: dict | None, policy, frontend: bool = True) -> Verdict:
    """Compare a stored record with what ``policy`` and the current front end give on the probe now."""
    if not isinstance(record, dict) or record.get("network") is None:
        why = (record or {}).get("error") if isinstance(record, dict) else ""
        return Verdict("unverified", ["no provenance record" + (f" ({why})" if why else "")
                                      + ": nothing ties this file to the current model code"])
    try:
        p = probe()
    except OSError as exc:
        return Verdict("unverified", [f"the probe episodes cannot be read ({exc})"])
    if record.get("probe_file") != p["sha256"]:
        return Verdict("unverified", ["the record was computed on other probe episodes than training/rl/probe_episodes.npz"])
    verdict = Verdict()
    then = _host(record["network"]).reshape(-1, NETWORK_FEATS)
    now = network_signature(policy).double().numpy().reshape(-1, NETWORK_FEATS)
    worst, count = _worst(now, then, NET_ATOL, NET_RTOL)
    if worst > 1.0:
        diff = np.abs(now - then)
        verdict.problems.append(
            f"the network computes a different function than when the file was written: {count} of {now.size} probe outputs "
            f"differ, largest difference {diff.max():.4g} (logits up to {diff[:, :sum(spec.HEAD_SIZES)].max():.4g}, "
            f"values up to {diff[:, sum(spec.HEAD_SIZES):sum(spec.HEAD_SIZES) + 2].max():.4g}; tolerance {NET_ATOL:g})")
    if frontend and record.get("frontend") is not None:
        current = frontend_signature("auto")
        if current is None:
            verdict.notes.append("the agent's front end is not importable here: its part of the record was not checked")
        else:
            then_f = _host(record["frontend"]).reshape(-1, FRONTEND_FEATS)
            now_f = current.astype(np.float64)
            atol = np.array([FRONT_ATOL] * 5 + [FRONT_MAP_ATOL] * spec.MAP_CHANNELS)
            rtol = np.array([FRONT_RTOL] * 5 + [0.0] * spec.MAP_CHANNELS)
            worst_f, count_f = _worst(now_f, then_f, atol, rtol)
            if worst_f > 1.0:
                bad = sorted({int(i) for i in np.nonzero((np.abs(now_f - then_f) > atol + rtol * np.abs(then_f)).any(0))[0]})
                names = ["rays", "entities", "entity mask", "scalars", "scalars"] + [f"map channel {c}" for c in range(spec.MAP_CHANNELS)]
                verdict.problems.append(
                    f"odometry / memory map / encoder give other network inputs than when the file was written: {count_f} of "
                    f"{now_f.size} probe summaries differ ({', '.join(dict.fromkeys(names[i] for i in bad))})")
    changed = [name for name, digest in code_fingerprint().items() if (record.get("code") or {}).get(name) != digest]
    if verdict.problems:
        verdict.status = "mismatch"
        if changed:
            verdict.problems.append("source files changed since the file was written: " + ", ".join(changed))
    elif changed:
        verdict.notes.append("source files changed since the file was written (" + ", ".join(changed)
                             + "), but the probe outputs still match")
    return verdict


def allowed() -> bool:
    return os.environ.get(ALLOW_ENV, "") not in ("", "0")


def enforce(verdict: Verdict, path, check: str = "strict") -> Verdict:
    """Act on a verdict. check: "strict", "require", "warn" or "off".

    strict   a mismatch raises ``CheckpointMismatch``; a file without a record loads with a warning
    require  as strict, and a file without a usable record raises too (used before exporting)
    warn     everything is printed, nothing raises (the caller passed --allow-mismatch, or DOOMRL_ALLOW_MISMATCH=1)
    off      no output
    """
    if check == "off":
        return verdict
    if check not in ("strict", "require", "warn"):
        raise ValueError(f"unknown check mode {check!r}")
    lenient = check == "warn" or allowed()
    for note in verdict.notes:
        print(f"note: {path}: {note}", flush=True)
    if verdict.status == "mismatch":
        text = f"{path} does not match the current code:\n  - " + "\n  - ".join(verdict.problems)
        if not lenient:
            raise CheckpointMismatch(
                text + "\nThe file would load without error and then play differently from what was trained. Check out the "
                f"code it was written with, retrain, or (for a deliberate migration) pass --allow-mismatch / set {ALLOW_ENV}=1.")
        print(f"WARNING: {text}\n  loaded anyway because the mismatch was explicitly allowed", flush=True)
    elif verdict.status == "unverified":
        text = f"{path}: " + "; ".join(verdict.problems)
        if check == "require" and not lenient:
            raise CheckpointMismatch(text + f". Refusing; pass --allow-unverified / set {ALLOW_ENV}=1 to use it anyway.")
        print(f"WARNING: {text}", flush=True)
    return verdict


# ---------------------------------------------------------------------------------------------
# Recording the probe episodes (done once; the file is tracked)
# ---------------------------------------------------------------------------------------------
PROBE_GRIDS = (
    # a combat arena: monsters, health, ammo
    "##########\n#P......z#\n#..h.....#\n#....i...#\n#.n.....z#\n##########",
    # navigation: an exit pad in view, a door, a locked door, a key and a pickup
    "############\n#P.....D.r.#\n#.....E#...#\n####R#######\n#......h...#\n############",
)


def record_probe(path=PROBE_FILE, steps: int = PROBE_STEPS, seed: int = 0) -> dict:
    """Play ``steps`` decision steps on each of ``PROBE_GRIDS`` and store what the front end saw and produced.

    The buttons are drawn at random with a bias to walking forward and obey the action masks. The
    exit observation is the training oracle without noise. The stored network inputs are those of
    the front end at recording time; they are the fixed inputs of every later network comparison.
    """
    from training.mapgen import udmf

    rng = np.random.default_rng(seed)
    budget = float(spec.DEFAULT_BUDGET_STEPS)
    raw_keys = ("vitals", "distance", "hit_type", "features", "entity_type", "entity_mask", "action", "exit_obs")
    episodes = []
    with tempfile.TemporaryDirectory() as folder:
        for index, grid in enumerate(PROBE_GRIDS):
            wad = str(udmf.write_wad(grid, Path(folder) / f"probe{index}.wad"))
            entry = deps.MapEntry(wad, tier=1, episode_ticks=2100)
            env, facts = deps.make_env(entry), deps.map_facts(wad)
            try:
                obs, _ = env.reset(seed=seed + index)
                frontend = deps.FrontEnd("real")
                frontend.reset()
                rows, action = {k: [] for k in raw_keys}, None
                encoded = []
                for t in range(steps):
                    exit_obs = deps.exit_observation(env, facts, obs, env.privileged_state(), None)
                    step = encode_step(frontend, obs, action, t, budget, exit_obs)
                    encoded.append(step)
                    heads = [rng.choice(3, p=[0.2, 0.7, 0.1]), rng.choice(3, p=[0.7, 0.15, 0.15]), rng.choice(3, p=[0.5, 0.25, 0.25]),
                             int(step["masks"][1] and rng.random() < 0.6), int(step["masks"][0]), int(rng.random() < 0.02)]
                    action = spec.heads_to_buttons(heads)
                    rows["vitals"].append(np.asarray(obs["vitals"], np.float32))
                    rows["distance"].append(np.asarray(obs["rays"]["distance"], np.float32))
                    rows["hit_type"].append(np.asarray(obs["rays"]["hit_type"], np.int64))
                    rows["features"].append(np.asarray(obs["visible_entities"]["features"], np.float32))
                    rows["entity_type"].append(np.asarray(obs["visible_entities"]["entity_type"], np.int64))
                    rows["entity_mask"].append(np.asarray(obs["visible_entities"]["mask"], np.int8))
                    rows["action"].append(action)
                    rows["exit_obs"].append(np.full(3, np.nan, np.float32) if exit_obs is None else np.asarray(exit_obs, np.float32))
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        raise RuntimeError(f"probe episode {index} ended after {t + 1} steps; choose another seed")
            finally:
                env.close()
            episode = {k: np.stack(v) for k, v in rows.items()}
            episode.update({k: np.stack([s[k] for s in encoded]) for k in encoded[0]})
            episodes.append(episode)
    data = {k: np.stack([ep[k] for ep in episodes]) for k in episodes[0]}
    data["budget"] = np.float32(budget)
    np.savez_compressed(path, **data)
    probe.cache_clear()
    frontend_signature.cache_clear()
    replayed = replay_frontend("real")                   # the stored raw observations must be all the front end needs
    for k, v in replayed.items():
        if not np.array_equal(v, data[k]):
            raise RuntimeError(f"replaying the stored observations does not reproduce the recorded {k!r}")
    return {"path": str(path), "sha256": _sha256(path), "bytes": Path(path).stat().st_size,
            "episodes": len(episodes), "steps": steps}


def check_file(path, policy_loader=None) -> Verdict:
    """Compare one checkpoint or exported weights file with the current code (no exception on a mismatch)."""
    from training.rl.train import load_policy

    policy, payload = (policy_loader or load_policy)(str(path), "cpu", check="off")
    return compare(record_of(payload), policy)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check that checkpoints or exported weights still match the current model code")
    ap.add_argument("files", nargs="*", help="checkpoints of training.rl.train or weights files of training.rl.export")
    ap.add_argument("--record", action="store_true", help="record the probe episodes (refuses to replace an existing file)")
    ap.add_argument("--force", action="store_true", help="with --record: replace the file; every earlier record becomes unverifiable")
    args = ap.parse_args()
    if args.record:
        if PROBE_FILE.exists() and not args.force:
            ap.error(f"{PROBE_FILE} exists; records of existing checkpoints refer to it (--force replaces it)")
        print(record_probe())
        return
    if not args.files:
        ap.error("give at least one file, or --record")
    worst = 0
    for path in args.files:
        verdict = check_file(path)
        print(f"{verdict.status.upper():10s} {path}")
        for line in verdict.problems + verdict.notes:
            print(f"           {line}")
        worst = max(worst, {"ok": 0, "unverified": 2, "mismatch": 1}[verdict.status])
    sys.exit(worst)


if __name__ == "__main__":
    main()
