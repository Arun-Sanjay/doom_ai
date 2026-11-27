"""Weighted average of the network weights of checkpoints of the same architecture.

    python -m training.rl.average OUT.pt CKPT_A CKPT_B ... [--weights w1,w2,...]
                                  [--policy-file POLICY.pt [--greedy-heads move,strafe,turn,use,weapon]]

Training-time scaffolding, never imported by the agent.

OUT.pt is a file in the checkpoint layout of `training/rl/train.py` that holds a network and no
training state. `training.rl.export` and `training.promote score` take it like any checkpoint.

What is averaged
    Every tensor of `payload["model"]`, the state dict of the network (for `PolicyNet`: 73 float32
    tensors, 1,307,153 values, the two value heads included). For each tensor the terms
    `w_i * tensor_i` are added in float64 in the order the files are given and the sum is stored
    as float32. The weights must be positive and are divided by their sum first; without
    `--weights` every file has the same weight.

What is taken from the last file
    `policy_config`, `policy_kind`, `iteration`, `env_steps`, and from its provenance record the
    front-end entry and the fingerprint of the source files. So the step counter of OUT.pt is the
    last file's counter and says nothing about the other files: when two branches of a run are
    averaged, name the branch whose counter the file should carry last. `averaged_from` lists
    every file with its sha256, iteration, step counter and weight.

What is computed again
    The network entry of the provenance record (`training/rl/provenance.py`): the average is a
    new function of the same code, so it gets the outputs it gives on the probe episodes itself.

What is dropped
    `optimizer`, `norm`, `norm_int`, `curriculum`, `distill`, `updates` and `config`. They belong
    to one training run each and are never read by the agent. Training cannot resume from OUT.pt.

Refused: files that are not checkpoints, another `policy_kind` or `policy_config` than the first
file's, a state dict with other tensor names, shapes or types, a tensor that is not floating
point, a weight that is not positive, a result that is not finite. An average of networks that
were not trained from a common start is still written; whether it plays is a matter for
`training.promote` and `training.final_pick`.

With `--policy-file` the average is also exported as the weights file the agent loads
(`training.rl.export`, with its provenance checks). `--greedy-heads` stores the decoding setting in
that file's config: the heads named take their most likely option, the others are sampled, as
`submission/agent.py` describes under `GREEDY_KEY`.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

KEPT_FROM_LAST = ("policy_config", "policy_kind", "iteration", "env_steps")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalised(weights, count: int) -> list[float]:
    """`count` positive weights that sum to one; equal weights when none are given."""
    weights = [1.0] * count if weights is None else [float(w) for w in weights]
    if len(weights) != count:
        raise ValueError(f"{len(weights)} weight(s) for {count} file(s)")
    if not all(w > 0.0 and w < float("inf") for w in weights):
        raise ValueError(f"every weight must be positive and finite: {weights}")
    total = sum(weights)
    return [w / total for w in weights]


def average_state(states: list[dict], weights=None) -> dict:
    """The weighted average of state dicts with the same tensors. Sums in float64, returns float32.

    Raises ValueError when the dicts do not describe the same architecture (names, order, shapes,
    types) or hold a tensor that cannot be averaged.
    """
    if not states:
        raise ValueError("no state dict given")
    weights = normalised(weights, len(states))
    first = states[0]
    for number, state in enumerate(states[1:], 2):
        if list(state) != list(first):
            other = sorted(set(state) ^ set(first))
            raise ValueError(f"file {number} holds other tensors than file 1" + (f": {other[:6]}" if other else " (another order)"))
    out = {}
    for name, tensor in first.items():
        if not tensor.is_floating_point():
            raise ValueError(f"{name} is {tensor.dtype}: only floating-point tensors can be averaged")
        for number, state in enumerate(states[1:], 2):
            if state[name].shape != tensor.shape or state[name].dtype != tensor.dtype:
                raise ValueError(f"{name}: file {number} has {tuple(state[name].shape)} {state[name].dtype}, "
                                 f"file 1 has {tuple(tensor.shape)} {tensor.dtype}")
        mean = sum(w * state[name].double() for w, state in zip(weights, states)).float()
        if not bool(torch.isfinite(mean).all()):
            raise ValueError(f"{name}: the average is not finite")
        out[name] = mean
    return out


def merge(payloads: list[dict], weights=None, names=None) -> dict:
    """The averaged checkpoint of loaded checkpoints, without a provenance record.

    names: what to call the files in `averaged_from` and in error messages (default: their position).
    Raises ValueError when the files are not checkpoints of one architecture.
    """
    names = [f"file {k}" for k in range(1, len(payloads) + 1)] if names is None else [str(n) for n in names]
    for name, payload in zip(names, payloads):
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict) or any(
                key not in payload for key in KEPT_FROM_LAST):
            raise ValueError(f"{name} is not a checkpoint of training.rl.train")
    first, last = payloads[0], payloads[-1]
    for name, payload in zip(names[1:], payloads[1:]):
        for key in ("policy_kind", "policy_config"):
            if payload.get(key) != first.get(key):
                raise ValueError(f"{name} has another architecture than {names[0]}: {key} {payload.get(key)} "
                                 f"against {first.get(key)}")
    out = {"format": last.get("format"), "model": average_state([p["model"] for p in payloads], weights)}
    out.update({key: last[key] for key in KEPT_FROM_LAST})
    out["averaged_from"] = [{"file": name, "iteration": int(p["iteration"]), "env_steps": int(p["env_steps"]), "weight": w}
                            for name, p, w in zip(names, payloads, normalised(weights, len(payloads)))]
    return out


def average_files(files: list[str], weights=None, check: str = "strict") -> dict:
    """Load checkpoints, average them (`merge`) and stamp the result with a fresh provenance record.

    check: how each file's own provenance record is compared with the current code when it is
    loaded, as in `training.rl.train.load_policy`.
    """
    from training.rl import deps, provenance
    from training.rl.train import load_policy

    payloads = [load_policy(str(f), "cpu", check=check)[1] for f in files]
    out = merge(payloads, weights, [str(f) for f in files])
    for entry, f in zip(out["averaged_from"], files):
        entry["sha256"] = sha256(f)
    kind = "real" if out["policy_kind"] == "PolicyNet" else "standin"
    net = deps.make_policy(kind, **out["policy_config"])
    net.load_state_dict(out["model"], strict=True)
    net.eval()
    last = provenance.record_of(payloads[-1]) or {}
    record = provenance.stamp(net, frontend=last.get("frontend"), code=dict(last.get("code") or {}))
    if record["error"]:
        raise RuntimeError(f"no provenance record for the average: {record['error']}")
    out["provenance"] = record
    return out


def store_decoding(policy_file, heads: list[str]) -> list[str]:
    """Write `heads` as the decoding setting of an exported weights file; state dict and the rest of the config are kept."""
    from training import paths  # noqa: F401 - puts submission/ on the import path
    import agent

    blob = torch.load(str(policy_file), map_location="cpu", weights_only=True)
    if set(blob) != {"state_dict", "config"}:
        raise ValueError(f"{policy_file} is not in the agent's layout: {sorted(blob)}")
    names = [name.strip().lower() for name in heads if name.strip()]
    agent.head_indices(names, "--greedy-heads")                       # raises ValueError on a name the agent would refuse
    config = dict(blob["config"])
    config[agent.GREEDY_KEY] = names
    torch.save({"state_dict": blob["state_dict"], "config": config}, str(policy_file))
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Weighted average of the network weights of checkpoints of one architecture")
    ap.add_argument("out", help="the averaged checkpoint to write")
    ap.add_argument("checkpoints", nargs="+", help="checkpoints of training.rl.train; step counter and iteration are the last one's")
    ap.add_argument("--weights", help="comma-separated positive weights, one per checkpoint (default: equal)")
    ap.add_argument("--policy-file", help="also export the average as the weights file the agent loads")
    ap.add_argument("--greedy-heads", help="with --policy-file: comma-separated heads decoded by their most likely option, "
                                           "stored in the file's config")
    ap.add_argument("--allow-unverified", action="store_true", help="accept checkpoints that carry no provenance record")
    args = ap.parse_args(argv)
    if args.greedy_heads and not args.policy_file:
        ap.error("--greedy-heads needs --policy-file")
    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    try:
        payload = average_files(args.checkpoints, weights, check="strict" if args.allow_unverified else "require")
    except ValueError as error:
        raise SystemExit(f"refused: {error}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    values = sum(v.numel() for v in payload["model"].values())
    print(f"wrote {out}: {len(payload['model'])} tensors, {values} values, iteration {payload['iteration']}, "
          f"{payload['env_steps']} env steps (the last file's counters), sha256 {sha256(out)[:12]}")
    for entry in payload["averaged_from"]:
        print(f"  weight {entry['weight']:.4f}  {entry['file']}  iteration {entry['iteration']}, {entry['env_steps']} env steps, "
              f"sha256 {entry['sha256'][:12]}")
    if args.policy_file:
        from training.rl import export

        done = export.export(str(out), args.policy_file, check="strict" if args.allow_unverified else "require")
        note = ""
        if args.greedy_heads:
            names = store_decoding(args.policy_file, args.greedy_heads.split(","))
            export.verify(args.policy_file, "strict" if args.allow_unverified else "require")
            note = f", most likely option for {', '.join(names)}"
        print(f"wrote {args.policy_file}: provenance {done['provenance']}{note}, sha256 {sha256(args.policy_file)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
