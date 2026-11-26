"""Training entry point: recurrent PPO in the vector environment.

    python -m training.rl.train --maps data/maps/train/manifest.jsonl --out runs/NAME
           [--config cfg.json] [--workers 30 --envs-per-worker 16] [--device auto]
           [--init weights.pt | --resume runs/NAME/latest.pt] [--heldout heldout_manifest.jsonl]

Training-time scaffolding, never imported by the agent.

* **Learner device: CUDA when available, else MPS, else CPU.** It is printed at start, checked
  against the device the parameters actually sit on, and written to every log row. The rollout
  workers always run on the CPU; their number is a flag, so the same command runs on a laptop and
  on a many-core box with an NVIDIA GPU.
* One iteration = collect ``rollout`` steps from every environment (one batched forward pass per
  step on the learner device), then the PPO epochs, then one CSV row.
* ``--maps toy:bandit`` runs the same loop on a toy task in seconds: a quick check of a new machine.
* ``--print-config`` prints every setting with its default as JSON; ``--config`` takes a JSON file
  with the same nesting to override any of them. The resolved config is saved next to the log.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training import paths  # noqa: F401
from training.rl import deps, provenance
from training.rl.buffer import RolloutBuffer
from training.rl.curriculum import NUM_TIERS, Curriculum, CurriculumConfig
from training.rl.ppo import ENTROPY_NAMES, PPO, DistillSchedule, PPOConfig
from training.rl.reward import TERMS
from training.rl.rollout import Collector, heldout_exit_rate
from training.rl.vecenv import VecConfig, VecEnv


@dataclass
class TeachConfig:
    """Schedule of beta, the probability that the teacher's action is executed on a labelled step.

    beta falls linearly from ``beta_start`` to ``beta_floor`` over the first ``beta_steps``
    environment steps and stays at the floor afterwards. ``beta_steps = 0`` (the default) means no
    mixing at all: the policy always drives. Steps without controller labels are never mixed, and
    mixing stops altogether once the distillation weight has reached zero.
    """

    beta_start: float = 1.0
    beta_floor: float = 0.0
    beta_steps: int = 0

    def beta(self, env_steps: int) -> float:
        if self.beta_steps <= 0:
            return 0.0
        left = max(0.0, 1.0 - env_steps / self.beta_steps)
        return float(self.beta_floor + (self.beta_start - self.beta_floor) * left)


@dataclass
class TrainConfig:
    """Run settings; the nested configs hold the loss, the workers, the curriculum and the distillation schedule."""

    rollout: int = 64                    # steps per environment per iteration (a multiple of ppo.chunk)
    workers: int = 8                     # rollout worker processes (0 = step in this process)
    envs_per_worker: int = 16
    groups: int = 2                      # 2: workers step one half of their rows while the learner runs the other
    total_steps: int = 50_000_000        # environment steps
    seed: int = 0
    policy: str = "auto"                 # "auto" | "real" | "standin"
    policy_kwargs: dict = field(default_factory=dict)
    # Starting offsets of head output biases (deps.set_head_bias). A near-uniform weapon head presses
    # "next weapon" on half of all steps and the gun is then almost never ready, so the default makes
    # that 0.7 % at the start. {} leaves the network's own initialisation untouched.
    head_bias: dict = field(default_factory=lambda: {"weapon": [2.5, -2.5]})
    checkpoint_every: int = 25           # iterations
    heldout_every: int = 50              # iterations between held-out evaluations (needs --heldout)
    heldout_envs: int = 96               # one episode each
    heldout_workers: int = 4
    stats_window: int = 400              # episodes averaged in the log
    learner_threads: int = 0             # CPU threads for torch in the learner (0 = torch default)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    vec: VecConfig = field(default_factory=VecConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    distill: DistillSchedule = field(default_factory=DistillSchedule)
    teach: TeachConfig = field(default_factory=TeachConfig)


def apply_overrides(obj, overrides: dict, where: str = "config"):
    """Set fields of a (nested) dataclass from a dict with the same nesting; unknown keys are errors."""
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in overrides.items():
        if key not in names:
            raise KeyError(f"{where}: unknown setting {key!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            apply_overrides(current, value, f"{where}.{key}")
        else:
            setattr(obj, key, tuple(value) if isinstance(current, tuple) and isinstance(value, list) else value)
    return obj


def pick_device(name: str = "auto") -> torch.device:
    """CUDA when available, else MPS, else CPU (or exactly what was asked for)."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Device name with the GPU model and memory, for the start-up line and the log."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda ({props.name}, {props.total_memory / 2**30:.0f} GB)"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return f"cpu ({torch.get_num_threads()} threads)"


def device_memory_mb(device: torch.device) -> float:
    """Memory the learner holds on its device (0 on the CPU): evidence that the device is in use."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return 0.0


LOG_COLUMNS = (
    ["time", "iteration", "env_steps", "stage", "device", "env_steps_per_s", "collect_steps_per_s", "learner_samples_per_s",
     "episodes", "ep_len", "return_ext", "return_int", "exit_rate", "kills", "kill_frac", "death_rate", "door_reversals"]
    + [f"exit_rate_t{t}" for t in range(NUM_TIERS)]
    + [f"rew_{name}" for name in TERMS] + ["int_new_cells", "int_visit", "cells_seen"]
    + [f"ent_{name}" for name in ENTROPY_NAMES]
    + ["distill_weight", "loss_distill", "labelled", "heldout_exit_rate", "loss", "loss_pi", "loss_v", "loss_v_int",
       "kl", "clip_frac", "ev", "grad_norm", "lr", "ent_scale", "value_mean", "return_std", "device_mem_mb"]
    # The unified loop: teacher mixing, the parts of the distillation term, agreement with the teacher
    # (most likely option against the label, on labelled steps) and the failure counters of the run.
    + ["beta", "forced", "loss_distill_heads", "loss_distill_subgoal", "loss_distill_target", "loss_field", "field_rows",
       "agree", "agree_all", "agree_move", "agree_strafe", "agree_turn", "agree_use", "agree_subgoal", "agree_target",
       "target_cells_off", "exit_rate_labelled", "exit_rate_unlabelled", "heldout_episodes",
       "env_resets", "odom_errors", "teacher_errors", "kl_unlabelled", "minibatches"]
    # Where the time of an iteration goes: the share spent in the PPO update, and the shares of the
    # collection spent waiting for the workers and in the learner's own part of each step. Workers
    # replaced after a failure (VecEnv.respawns), total of this run.
    + ["update_frac", "collect_wait_frac", "collect_forward_frac", "worker_respawns"]
)


def episode_stats(recent) -> dict:
    """Means over the recent finished episodes (blank when there are none yet)."""
    if not recent:
        return {}
    mean = lambda xs: float(np.mean(xs)) if len(xs) else ""
    with_exit = [e for e in recent if e.get("has_exit", True)]
    with_monsters = [e for e in recent if e.get("monsters", 0) > 0]
    out = {"ep_len": mean([e["len"] for e in recent]), "return_ext": mean([e["ret"] for e in recent]),
           "return_int": mean([e["ret_int"] for e in recent]), "exit_rate": mean([e["exit"] for e in with_exit]),
           "kills": mean([e["kills"] for e in recent]),
           "kill_frac": mean([e["kills"] / e["monsters"] for e in with_monsters]),
           "death_rate": mean([e["dead"] for e in recent]),
           "door_reversals": mean([e.get("reversals", 0) for e in recent])}
    terms = np.mean([e["terms"] for e in recent], axis=0)
    out.update({f"rew_{name}": float(terms[i]) for i, name in enumerate(TERMS)})
    parts = np.mean([e["int_terms"] for e in recent], axis=0)
    out.update({"int_new_cells": float(parts[0]), "int_visit": float(parts[1]),
                "cells_seen": mean([e.get("cells_seen", 0) for e in recent])})   # memory-map cells observed per episode
    for flag, name in ((True, "exit_rate_labelled"), (False, "exit_rate_unlabelled")):   # episodes with / without teacher labels
        part = [e["exit"] for e in with_exit if (e.get("labelled", 0) > 0) == flag]
        if part:
            out[name] = float(np.mean(part))
    return out


CHECKPOINT_FORMAT = "doomnet-policy-v1"


def policy_config(policy) -> dict:
    """Constructor arguments of the network (``PolicyNet.config``), as a JSON-able dict."""
    return json.loads(json.dumps(dict(getattr(policy, "config", {}) or {})))


def save_checkpoint(path: Path, policy, ppo: PPO, curriculum: Curriculum, cfg: TrainConfig, iteration: int, env_steps: int,
                    frontend="auto", code: dict | None = None, also: Path | None = None) -> dict:
    """Write a checkpoint atomically (and a copy to ``also``). Returns its provenance record.

    Layout (a plain dict, loadable with ``torch.load(..., weights_only=True)``):

        "format"         "doomnet-policy-v1"
        "model"          the network's state dict: all the agent needs, with
        "policy_config"  the network's constructor arguments (``PolicyNet(**policy_config)``)
        "policy_kind"    "PolicyNet" or the name of the stand-in class
        "iteration", "env_steps", "updates", "optimizer", "curriculum", "distill", "norm", "norm_int"
                         what training needs to resume
        "config"         the full TrainConfig as nested JSON-able dicts
        "provenance"     what ties the file to the code that trained it (``training/rl/provenance.py``): sha256 of
                         the source files of the network and the front end, and what both gave on the probe
                         episodes when the file was written. ``load_policy`` recomputes and compares them.

    frontend: the front-end summary reported by the rollout workers (``VecEnv.frontend_signature``), None for the
    stand-in front end, "auto" to compute it in this process. code: the source fingerprint taken when the run
    started (default: the files as they are now).

    ``training/rl/export.py`` reduces this to the file the agent loads.
    """
    if isinstance(frontend, str) and deps.frontend_classes(cfg.vec.frontend)[3] != "doomnet":
        frontend = None
    record = provenance.stamp(policy, frontend, code)
    payload = {"format": CHECKPOINT_FORMAT, "model": policy.state_dict(), "policy_config": policy_config(policy),
               "policy_kind": type(policy).__name__, "optimizer": ppo.opt.state_dict(), "iteration": iteration,
               "env_steps": env_steps, "updates": ppo.updates, "curriculum": curriculum.state_dict(),
               "distill": cfg.distill.state_dict(),
               "norm": None if ppo.norm is None else ppo.norm.state_dict(),
               "norm_int": None if ppo.norm_int is None else ppo.norm_int.state_dict(),
               "config": json.loads(json.dumps(dataclasses.asdict(cfg), default=str)), "provenance": record}
    for target in (path, also):
        if target is not None:
            tmp = Path(target).with_suffix(".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, target)
    return record


def load_weights(policy, path: str, device, check: str = "strict") -> dict:
    """Load network weights from a checkpoint of this script or a bare state dict. Returns the payload.

    The file's provenance record is compared with the current code as in ``load_policy``.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload
    if isinstance(payload, dict):
        state = next((payload[k] for k in ("model", "state_dict") if isinstance(payload.get(k), dict)), payload)
    policy.load_state_dict(state)
    if check != "off":
        provenance.enforce(provenance.compare(provenance.record_of(payload), policy), path, check)
    return payload if isinstance(payload, dict) else {}


def load_policy(path: str, device="cpu", check: str = "strict"):
    """Rebuild the network of a checkpoint (or of an exported weights file). Returns (policy, payload).

    A state dict loads without error into any code with the same parameter names, whatever the
    forward pass has become. So the file's provenance record is recomputed here with the current
    code and compared (``training/rl/provenance.py``). check: "strict" (default) raises
    ``provenance.CheckpointMismatch`` when the network or the front end no longer gives what the
    file recorded and warns when the file carries no record; "require" also refuses a file
    without a record; "warn" only prints; "off" skips the comparison.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" not in payload and isinstance(payload.get("state_dict"), dict):
        # The agent's weights file ({"state_dict", "config": {"policy_kwargs": ...}}, written by export.py).
        payload = dict(payload, model=payload["state_dict"], policy_config=dict(payload.get("config", {}).get("policy_kwargs", {})))
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a checkpoint of training.rl.train")
    kwargs = payload.get("policy_config")
    if kwargs is None:                                   # exported file, or a checkpoint from before the key existed
        kwargs = payload["config"] if "format" in payload and "optimizer" not in payload else payload.get("config", {}).get("policy_kwargs", {})
    kind = "standin" if payload.get("policy_kind", "PolicyNet") != "PolicyNet" else "real"
    policy = deps.make_policy(kind, **kwargs).to(device)
    policy.load_state_dict(payload["model"])
    if check != "off":
        provenance.enforce(provenance.compare(provenance.record_of(payload), policy), path, check)
    return policy, payload


def train(cfg: TrainConfig, maps: str, out: str, device_name: str = "auto", init: str | None = None,
          resume: str | None = None, heldout: str | None = None, max_iterations: int | None = None,
          max_minutes: float | None = None, check: str = "strict") -> dict:
    """Run training. Returns the last log row (used by tests).

    ``check`` is how ``--init`` and ``--resume`` files are compared with the current code (see ``load_policy``).

    Training stops at ``cfg.total_steps`` environment steps, after ``max_iterations`` iterations
    or after ``max_minutes`` of wall-clock time from the first rollout, whichever comes first; the
    schedules always span ``cfg.total_steps``.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.learner_threads > 0:
        torch.set_num_threads(cfg.learner_threads)
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device_text = describe_device(device)
    print(f"learner device: {device_text}", flush=True)
    code = provenance.code_fingerprint()                   # the sources as they are when this run imports them

    entries = deps.load_manifest(maps)
    tiers = sorted({e.tier for e in entries})
    curriculum = Curriculum(cfg.curriculum, tiers)
    policy = deps.make_policy(cfg.policy, **cfg.policy_kwargs).to(device)
    deps.set_head_bias(policy, cfg.head_bias)              # a fresh network only: --init and --resume overwrite it
    param_devices = {p.device.type for p in policy.parameters()}
    if param_devices != {device.type}:
        raise RuntimeError(f"policy parameters are on {param_devices}, expected {device.type}")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {type(policy).__name__}, {n_params / 1e6:.2f} M parameters, all on {device.type}", flush=True)

    n_envs = max(cfg.workers, 1) * cfg.envs_per_worker
    per_iteration = cfg.rollout * n_envs
    total_iterations = max(cfg.total_steps // per_iteration, 1)
    ppo = PPO(policy, cfg.ppo, device, total_iterations, cfg.seed)
    iteration, env_steps = 0, 0
    if init:
        state = load_weights(policy, init, device, check)
        for norm, key in ((ppo.norm, "norm"), (ppo.norm_int, "norm_int")):     # the value heads were trained in these units
            if norm is not None and state.get(key):
                norm.load_state_dict(state[key])
        print(f"initial weights from {init}", flush=True)
    if resume:
        state = load_weights(policy, resume, device, check)
        ppo.opt.load_state_dict(state["optimizer"])
        iteration, env_steps, ppo.updates = state["iteration"], state["env_steps"], state["updates"]
        curriculum.load_state_dict(state["curriculum"])
        cfg.distill.load_state_dict(state["distill"])
        for norm, key in ((ppo.norm, "norm"), (ppo.norm_int, "norm_int")):
            if norm is not None and state.get(key):
                norm.load_state_dict(state[key])
        print(f"resumed from {resume} at iteration {iteration}, {env_steps} steps", flush=True)

    cfg.vec.seed = cfg.seed
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    vec = VecEnv(entries, cfg.workers, cfg.envs_per_worker, cfg.vec, cfg.groups, curriculum.tier_probs())
    print(f"rollout: {n_envs} environments on {cfg.workers} CPU worker processes, front end: {vec.frontend_name}, "
          f"{len(entries)} maps in tiers {tiers}, {per_iteration} steps per iteration", flush=True)
    buf = RolloutBuffer(cfg.rollout, n_envs, cfg.ppo.chunk, device)
    collector = Collector(vec, policy, buf, device, cfg.seed, cfg.ppo.bootstrap_on_truncation)

    def checkpoint(path: Path, also: Path | None = None) -> None:
        """Write a checkpoint stamped with the code this run imported and the front end its workers run."""
        save_checkpoint(path, policy, ppo, curriculum, cfg, iteration, env_steps, vec.frontend_signature, code, also)
        edited = [name for name, digest in provenance.code_fingerprint().items() if digest != code[name]]
        if edited and not checkpoint.warned:
            checkpoint.warned = True
            print(f"WARNING: {', '.join(edited)} changed on disk after this run started. Its checkpoints describe the code "
                  "that was imported at the start; they will be refused by code that computes something else.", flush=True)

    checkpoint.warned = False
    heldout_entries = deps.load_manifest(heldout) if heldout else None

    log_path = out_dir / "train_log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS, extrasaction="ignore")
    if new_log:
        writer.writeheader()
    recent: deque = deque(maxlen=cfg.stats_window)
    heldout_rate, heldout_episodes, row = "", "", {}
    counters = {"env_resets": 0, "odom_errors": 0, "teacher_errors": 0}      # totals of this run
    if cfg.vec.teacher != "none":
        taught = sum(e.teach for e in entries)
        print(f"teacher: {cfg.vec.teacher}; labels on {taught} of {len(entries)} maps (the others start off the 64-unit grid); "
              f"beta {cfg.teach.beta_start:g} -> {cfg.teach.beta_floor:g} over {cfg.teach.beta_steps} steps", flush=True)
    started = time.perf_counter()
    try:
        while env_steps < cfg.total_steps and (max_iterations is None or iteration < max_iterations):
            if max_minutes is not None and time.perf_counter() - started >= 60.0 * max_minutes:
                print(f"stopping after {max_minutes:g} minutes of wall-clock time", flush=True)
                break
            t0 = time.perf_counter()
            # Teacher mixing ends with the distillation: once its weight is zero the policy always drives.
            collector.beta = beta = cfg.teach.beta(env_steps) if cfg.distill.weight > 0.0 else 0.0
            reports = collector.collect()
            t1 = time.perf_counter()
            infos = [info for info in reports if not info.get("crashed")]
            for report in reports:
                if report.get("crashed"):                # a row that failed and was restarted on a fresh map
                    counters["env_resets"] += 1
                    if counters["env_resets"] <= 20 or counters["env_resets"] % 100 == 0:
                        print(f"environment reset #{counters['env_resets']}: row {report['row']}, tier {report.get('tier')}, "
                              f"map {report.get('map')}, step {report.get('len')}: {report.get('error')}", flush=True)
            for info in infos:
                curriculum.record(info)
                recent.append(info)
                counters["odom_errors"] += int(info.get("odom_errors", 0))
                counters["teacher_errors"] += int(info.get("teacher_errors", 0))
            stats = ppo.update(buf, cfg.distill.weight)
            t2 = time.perf_counter()
            iteration += 1
            env_steps += per_iteration
            if curriculum.advance(env_steps):
                print(f"curriculum: stage -> {curriculum.stage} at {env_steps} steps", flush=True)
            vec.set_tier_probs(curriculum.tier_probs())

            if heldout_entries and iteration % cfg.heldout_every == 0:
                try:
                    result = heldout_exit_rate(policy, heldout_entries, device, cfg.heldout_workers,
                                               max(cfg.heldout_envs // max(cfg.heldout_workers, 1), 1), cfg.vec, cfg.seed + iteration)
                except Exception as exc:                 # noqa: BLE001 - a failed measurement holds the weight; training goes on
                    print(f"held-out evaluation failed ({type(exc).__name__}: {exc}); distillation weight held", flush=True)
                    result = {"exit_rate": "", "episodes": 0}
                heldout_rate, heldout_episodes = result["exit_rate"], result["episodes"]
                weight = cfg.distill.report(heldout_rate if result["episodes"] else None)
                print(f"held-out: exit rate {heldout_rate if heldout_rate == '' else format(heldout_rate, '.3f')} over "
                      f"{result['episodes']} episodes -> distillation weight {weight}", flush=True)

            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "iteration": iteration, "env_steps": env_steps,
                   "stage": curriculum.stage, "device": device_text, "env_steps_per_s": per_iteration / (t2 - t0),
                   "collect_steps_per_s": per_iteration / (t1 - t0), "learner_samples_per_s": stats["samples"] / (t2 - t1),
                   "episodes": len(infos), "distill_weight": cfg.distill.weight, "heldout_exit_rate": heldout_rate,
                   "heldout_episodes": heldout_episodes, "beta": beta, **counters,
                   "update_frac": (t2 - t1) / (t2 - t0), "collect_wait_frac": collector.wait_s / (t1 - t0),
                   "collect_forward_frac": collector.forward_s / (t1 - t0), "worker_respawns": vec.respawns,
                   "value_mean": float(buf.value.mean()), "return_std": "" if ppo.norm is None else ppo.norm.std,
                   "device_mem_mb": device_memory_mb(device)}
            row.update({f"exit_rate_t{t}": ("" if curriculum.exit_rate(t) is None else curriculum.exit_rate(t)) for t in range(NUM_TIERS)})
            row.update(episode_stats(recent))
            row.update(stats)
            writer.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in row.items()})
            log_file.flush()
            if iteration == 1 or iteration % 10 == 0:
                show = lambda k, fmt: format(row[k], fmt) if isinstance(row.get(k), (int, float)) else "-"
                print(f"it {iteration} steps {env_steps} stage {curriculum.stage} | {row['env_steps_per_s']:.0f} env steps/s "
                      f"(collect {row['collect_steps_per_s']:.0f}, of it {100 * row['collect_wait_frac']:.0f} % waiting for workers; "
                      f"learner {row['learner_samples_per_s']:.0f} samples/s on {device.type}, update {100 * row['update_frac']:.0f} % "
                      f"of the time, {row['device_mem_mb']:.0f} MB) | return {show('return_ext', '.2f')} exit {show('exit_rate', '.2f')} "
                      f"kills {show('kills', '.2f')} | kl {stats['kl']:.4f} ev {stats['ev']:.2f} distill {cfg.distill.weight}"
                      + (f" | beta {beta:.2f} forced {show('forced', '.2f')} agree {show('agree', '.3f')} (all heads {show('agree_all', '.3f')}) "
                         f"sub-goal {show('agree_subgoal', '.2f')} target {show('agree_target', '.2f')} "
                         f"CE {show('loss_distill', '.3f')} field {show('loss_field', '.4f')}" if cfg.vec.teacher != "none" else "")
                      + (f" | resets {counters['env_resets']} odometry errors {counters['odom_errors']}"
                         if counters["env_resets"] or counters["odom_errors"] else ""), flush=True)
            if iteration % cfg.checkpoint_every == 0:
                checkpoint(out_dir / f"ckpt_{iteration:06d}.pt", also=out_dir / "latest.pt")
    except KeyboardInterrupt:
        print("interrupted: saving a checkpoint", flush=True)
    finally:
        checkpoint(out_dir / "latest.pt")
        log_file.close()
        vec.close()
    return row


def _stop_cleanly_on_signals() -> None:
    """Make SIGTERM and SIGINT end the run through the normal shutdown path (checkpoint, workers closed).

    A job started in the background with ``nohup ... &`` has SIGINT ignored, so without this the
    only way to stop it would be a kill that skips the final checkpoint.
    """
    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recurrent PPO training in the vector environment")
    ap.add_argument("--maps", help="manifest.jsonl, a WAD, a directory of WADs, or toy:bandit")
    ap.add_argument("--out", help="run directory (log, config and checkpoints)")
    ap.add_argument("--config", help="JSON file overriding TrainConfig fields (nested like --print-config)")
    ap.add_argument("--device", default="auto", help="auto (CUDA, else MPS, else CPU), cuda, mps or cpu")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--envs-per-worker", type=int)
    ap.add_argument("--rollout", type=int)
    ap.add_argument("--total-steps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--policy", choices=("auto", "real", "standin"))
    ap.add_argument("--frontend", choices=("auto", "real", "standin"))
    ap.add_argument("--init", help="initial network weights (a checkpoint or a bare state dict)")
    ap.add_argument("--resume", help="checkpoint of this script to continue from")
    ap.add_argument("--heldout", help="held-out manifest for the exit-rate check that gates the distillation weight")
    ap.add_argument("--max-iterations", type=int)
    ap.add_argument("--max-minutes", type=float, help="stop after this much wall-clock time (schedules still span --total-steps)")
    ap.add_argument("--print-config", action="store_true", help="print every setting with its default and exit")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="load an --init / --resume file although the current code no longer computes what it recorded")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):            # lines appear at once when the output is redirected to a file
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    cfg = TrainConfig()
    if args.config:
        apply_overrides(cfg, json.loads(Path(args.config).read_text()))
    for name in ("workers", "envs_per_worker", "rollout", "total_steps", "seed", "policy"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if args.frontend:
        cfg.vec.frontend = args.frontend
    if args.print_config:
        print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
        return
    if not args.maps or not args.out:
        ap.error("--maps and --out are required")
    _stop_cleanly_on_signals()
    train(cfg, args.maps, args.out, args.device, args.init, args.resume, args.heldout, args.max_iterations, args.max_minutes,
          "warn" if args.allow_mismatch else "strict")


if __name__ == "__main__":
    main()
