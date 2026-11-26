"""Export a training checkpoint as the weights file the agent loads.

    python -m training.rl.export runs/NAME/ckpt_000100.pt            # -> submission/weights/policy.pt
    python -m training.rl.export runs/NAME/latest.pt --dest some/dir/policy.pt --episode data/maps/train/t0_0000004.wad

Training-time scaffolding, never imported by the agent.

The file is written by the agent's own ``save_weights`` (``submission/agent.py``), so it is exactly
the layout ``DoomAgent`` reads with ``torch.load(..., weights_only=True)``:

    "state_dict"  state dict of ``doomnet.model.PolicyNet``, float32 tensors on the CPU
    "config"      {"policy_kwargs": the constructor arguments of the trained network (hidden, vp_iters,
                   strict_goal_bottleneck, vp_channels, vp_moves),
                   "format", "source", "source_sha256", "iteration", "env_steps": where it came from,
                   "provenance": the checkpoint's record of the code that trained it, as nested lists
                                 (``training/rl/provenance.py``)}

Optimiser state, schedules and the training config stay in the checkpoint.

A state dict loads into any code that kept the parameter names, whatever the forward pass has
become, so three checks stand between a checkpoint and the agent's weights folder:

1. the checkpoint must carry a provenance record and the current code must still compute what it
   recorded, for the network and for odometry / memory map / encoder (``load_policy`` with
   ``check="require"`` from the command line; ``--allow-mismatch`` and ``--allow-unverified`` override,
   for deliberate migrations; ``export()`` called from code refuses a mismatch and warns about a missing record);
2. the written file is read back with the agent's ``read_weights`` into a fresh network
   (``strict=True``) and must give the recorded probe outputs again (``verify``);
3. with ``--episode WAD`` (or ``verify_in_agent``) it is loaded by a real ``DoomAgent`` that is not
   allowed to fall back to untrained weights, whose own network must give the recorded outputs,
   and that plays one episode of that map.

``python -m training.rl.export --check [FILE]`` runs check 2 on an existing weights file (default:
``submission/weights/policy.pt``) and exits non-zero unless it matches the code as it is now: run
it after any edit under ``submission/doomnet`` and before submitting.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import torch

from training import paths
from doomnet import spec
from training.rl import deps, provenance
from training.rl.train import CHECKPOINT_FORMAT, load_policy

DEFAULT_DEST = paths.ROOT / "submission" / "weights" / "policy.pt"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _agent_module():
    """``submission/agent.py`` (``training.paths`` has put ``submission/`` on the import path)."""
    import agent
    return agent


def export(checkpoint: str, dest=DEFAULT_DEST, episode_wad: str | None = None, check: str = "strict") -> dict:
    """Write the agent's weights file from ``checkpoint``.

    Returns {"path", "sha256", "parameters", "config", "provenance"} and, when ``episode_wad`` names
    a map, the result of ``verify_in_agent`` under "agent". Nothing is written to ``dest`` unless
    every check passes. check: "strict" (the default of this function) refuses a checkpoint whose
    provenance record no longer matches the current code and accepts one without a record with a
    warning ("provenance": "unverified" in the result); "require" (the default of the command
    line) refuses a missing record too; "warn" only prints.
    """
    policy, payload = load_policy(checkpoint, "cpu", check=check)
    if type(policy).__name__ != "PolicyNet":
        raise ValueError(f"{checkpoint} holds a {type(policy).__name__}, not the agent's PolicyNet")
    config = {"format": CHECKPOINT_FORMAT, "source": Path(checkpoint).name, "source_sha256": sha256(checkpoint),
              "iteration": int(payload.get("iteration", 0)), "env_steps": int(payload.get("env_steps", 0))}
    record = provenance.to_json(provenance.record_of(payload))
    if record is not None:
        config["provenance"] = record
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        _agent_module().save_weights(tmp, policy, config)       # adds config["policy_kwargs"] = policy.config
        status = verify(tmp, "strict" if check == "require" and record is None else check).status
        played = verify_in_agent(tmp, episode_wad, check=check) if episode_wad else None
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    out = {"path": str(dest), "sha256": sha256(dest), "parameters": sum(p.numel() for p in policy.parameters()),
           "config": dict(policy.config), "provenance": status}
    if played is not None:
        out["agent"] = played
    return out


def verify(path, check: str = "require") -> provenance.Verdict:
    """Read ``path`` with the agent's own reader into a fresh network and compare it with its provenance record.

    Raises on an unreadable file, a wrong layout, non-finite outputs, and (``provenance.CheckpointMismatch``)
    when the network or the front end no longer gives what the file recorded on the probe episodes,
    or when the file has no record (``check="require"``). Returns the verdict of the comparison.
    """
    state, config, status = _agent_module().read_weights(path)
    if status != "ok":
        raise ValueError(f"{path}: weights file is {status}")
    if config.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unexpected format {config.get('format')!r}")
    net = deps.make_policy("real", **config["policy_kwargs"])
    net.load_state_dict(state, strict=True)
    net.eval()
    verdict = provenance.enforce(provenance.compare(config.get("provenance"), net), path, check)
    with torch.inference_mode():
        out, _ = net(torch.zeros(1, spec.NUM_RAYS, spec.RAY_FEATS), torch.zeros(1, spec.MAX_ENTITIES, spec.ENT_FEATS),
                     torch.zeros(1, spec.MAX_ENTITIES, dtype=torch.bool), torch.zeros(1, spec.SCALAR_DIM),
                     torch.zeros(1, spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE),
                     masks=torch.ones(1, 2, dtype=torch.bool), sample=False)
    if not all(bool(torch.isfinite(lg).all()) for lg in out["logits"]):
        raise ValueError(f"{path}: the exported network returns non-finite logits")
    return verdict


def verify_in_agent(path, wad: str | None = None, episode_ticks: int = 2100, seed: int = 0, check: str = "strict") -> dict:
    """Load ``path`` as ``policy.pt`` in a real ``DoomAgent`` and, with ``wad``, play one episode of that map.

    The agent is pointed at a temporary weights folder that holds ``path`` and the committed exit
    detector, with the untrained-weights switch removed from the environment: it starts only if
    the file is one it accepts. Returns the weights line the agent reports and the episode's
    outcome; raises if the agent refuses to start, reports another architecture than the one
    stored, holds a network that does not give the file's recorded probe outputs, or answers any
    decision of the episode with its error handler.
    """
    agent = _agent_module()
    _, config, _ = agent.read_weights(path)
    saved_dir, saved_env = agent.WEIGHTS_DIR, os.environ.pop(agent.ALLOW_UNTRAINED_ENV, None)
    folder = Path(tempfile.mkdtemp(prefix="doomnet_export_"))
    try:
        shutil.copyfile(path, folder / agent.POLICY_FILE)
        detector = saved_dir / agent.EXIT_DETECTOR_FILE
        if not detector.is_file():
            raise FileNotFoundError(f"{detector} is missing: the agent cannot start without its exit detector")
        shutil.copyfile(detector, folder / agent.EXIT_DETECTOR_FILE)
        agent.WEIGHTS_DIR = folder
        from doom_env import action_space, make_env, observation_space
        player = agent.DoomAgent(observation_space(), action_space(), "cpu")
        if dict(player.policy.config) != dict(config["policy_kwargs"]):
            raise ValueError(f"{path}: the agent built PolicyNet({player.policy.config}), the file stores {config['policy_kwargs']}")
        in_agent = provenance.enforce(provenance.compare(config.get("provenance"), player.policy, frontend=False),
                                      f"{path} (as loaded by DoomAgent)", check)
        out = {"policy_sha256": config["sha256"], "policy_kwargs": dict(player.policy.config), "warmup_failures": player.warmup_failures,
               "provenance": in_agent.status}
        if wad:
            env = make_env(str(wad), episode_ticks=episode_ticks, render_buffers=True)
            try:
                obs, _ = env.reset(seed=seed)
                player.reset()
                steps, done, pressed = 0, False, 0
                while not done:
                    action = player.act(obs)
                    pressed += int(action.any())
                    obs, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    done = bool(terminated or truncated)
                state = env.privileged_state()
            finally:
                env.close()
            out.update(map=Path(wad).name, steps=steps, steps_with_buttons=pressed, exit=bool(state.get("exit_reached", False)),
                       dead=bool(state.get("dead", False)), failures=int(player.failures))
            if player.failures:
                raise RuntimeError(f"{path}: the agent's error handler answered {player.failures} of {steps} decisions")
        return out
    finally:
        agent.WEIGHTS_DIR = saved_dir
        if saved_env is not None:
            os.environ[agent.ALLOW_UNTRAINED_ENV] = saved_env
        shutil.rmtree(folder, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy a checkpoint's network to submission/weights/policy.pt")
    ap.add_argument("checkpoint", nargs="?", help="a checkpoint written by training.rl.train")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="output file (default: submission/weights/policy.pt)")
    ap.add_argument("--episode", metavar="WAD", help="also load the file in a DoomAgent and play one episode of this map")
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--check", nargs="?", const=str(DEFAULT_DEST), metavar="FILE",
                    help="do not export: check that an exported file (default: submission/weights/policy.pt) still "
                         "matches the current code; exit status 0 only if it does")
    ap.add_argument("--allow-unverified", action="store_true", help="export a checkpoint that carries no provenance record")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="export although the current code no longer computes what the checkpoint recorded")
    args = ap.parse_args()
    if args.check:
        try:
            verdict = verify(args.check, "require")
        except (provenance.CheckpointMismatch, ValueError) as exc:
            raise SystemExit(f"CHECK FAILED: {exc}")
        print(f"{args.check}: {verdict.status}; the network and the front end give what the file recorded when it was written")
        return
    if not args.checkpoint:
        ap.error("give a checkpoint to export, or --check")
    out = export(args.checkpoint, args.dest, args.episode,
                 "warn" if args.allow_mismatch else "strict" if args.allow_unverified else "require")
    print(f"wrote {out['path']}: {out['parameters']} parameters, PolicyNet({out['config']}), sha256 {out['sha256'][:12]}, "
          f"provenance {out['provenance']}")
    if "agent" in out:
        print(f"DoomAgent loaded it without the untrained switch and played one episode: {out['agent']}")


if __name__ == "__main__":
    main()
