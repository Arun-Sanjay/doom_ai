"""Recurrent PPO for a factored, hierarchical policy. Plain torch, no RL library.

Training-time scaffolding, never imported by the agent.

**Policy.** One decision is six categorical controller heads (move, strafe, turn, attack, use,
weapon). On manager steps (every ``MANAGER_PERIOD`` steps from the episode start) it also holds
the manager's sub-goal and target cell. The joint log-probability of a step is

    log pi(a | s) = sum_h log pi_h(a_h | s)  +  [manager step] * (log pi_sub(g | s) + log pi_tgt(c | s))

USE and ATTACK are masked from observation fields: where the mask is False the "on" logit is -inf,
in the rollout and in the loss alike, so a masked choice has probability exactly zero in both.

**Loss** (minimised; means are over all samples of a minibatch unless stated):

    L = L_pi + vf_coef * L_v + vf_int_coef * L_v_int - sum_k ent_coef_k * ent_scale * H_k + w_distill * L_distill

    ratio     = exp(log pi_new - log pi_old)
    A         = normalised (A_ext + int_coef * A_int)                       (two GAE streams, see buffer.py)
    L_pi      = -mean(min(ratio * A, clip(ratio, 1 - clip, 1 + clip) * A))
    L_v       = 0.5 * mean(max((V - R)^2, (V_old + clip(V - V_old, +-value_clip) - R)^2))
    L_v_int   = the same for the intrinsic value head and the intrinsic return
                (V and R of each stream in units of that stream's running return mean and std)
    H_k       = mean entropy of head k (six controller heads; sub-goal and target over manager steps)
    L_distill = sum_h d_h * CE(pi_h, teacher a_h) + d_sub * CE(pi_sub, teacher g) + d_tgt * CE(pi_tgt, teacher c)
                over steps whose label mask is set (manager terms over labelled manager steps)

Gradients are clipped by global norm. The learning rate and the entropy scale follow linear
schedules; ``w_distill`` comes from a ``DistillSchedule`` that only steps down while the held-out
exit rate holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from training import paths  # noqa: F401
from doomnet import spec
from training.rl.buffer import RolloutBuffer, RunningNorm, StateLayout

ENTROPY_NAMES = spec.HEAD_NAMES + ("subgoal", "target")


@dataclass
class PPOConfig:
    """Every hyper-parameter of the loss and the update."""

    gamma: float = 0.997                 # extrinsic discount (episodes last 525 to 1050 steps)
    gamma_int: float = 0.99              # intrinsic discount
    lam: float = 0.95                    # GAE lambda, both streams
    int_coef: float = 0.3                # weight of the intrinsic advantage stream
    episodic_intrinsic: bool = True      # False: intrinsic returns run across episode ends
    bootstrap_on_truncation: bool = True  # False: treat the tick cap as a terminal state
    normalise_adv: bool = True
    clip: float = 0.2                    # surrogate clip
    value_clip: float | None = 0.2       # clip on the value update, in normalised return units; None disables
    normalise_returns: bool = True       # value heads regress returns normalised by running statistics
    vf_coef: float = 0.5
    vf_int_coef: float = 0.5
    # Entropy bonus per head: move, strafe, turn, attack, use, weapon, then manager sub-goal and target.
    ent_coef: tuple = (0.01, 0.01, 0.01, 0.01, 0.003, 0.003, 0.01, 0.001)
    ent_final: float = 0.3               # entropy scale at the end of training (linear from 1)
    lr: float = 3e-4
    lr_final: float = 0.1                # learning-rate factor at the end of training (linear from 1)
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    epochs: int = 3
    chunk: int = 16                      # truncated-BPTT length
    minibatch_chunks: int = 256          # chunks per minibatch (samples = this * chunk)
    target_kl: float | None = None       # stop the epochs of an update early above 1.5 * this
    # Distillation weights per controller head, then manager sub-goal and target. The target is a
    # 4096-way choice (cross-entropy ln 4096 = 8.3 at the start), so it is weighted down to the
    # scale of the other terms.
    distill_coef: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25)


# ---------------------------------------------------------------------------------------------
# Factored categorical distribution with observation masks
# ---------------------------------------------------------------------------------------------
def apply_masks(logits: list, masks) -> list:
    """Set the "on" logit of the USE and ATTACK heads to -inf where the observation forbids it.

    masks: (B, 2) bool = (use_ok, attack_ok), the output of ``spec.action_masks``. Idempotent, so
    it is safe on logits the network has already masked.
    """
    if masks is None:
        return list(logits)
    out = list(logits)
    for head, col in ((spec.HEAD_USE, 0), (spec.HEAD_ATTACK, 1)):
        off, on = out[head][..., 0], out[head][..., 1]
        out[head] = torch.stack([off, on.masked_fill(~masks[..., col], float("-inf"))], -1)
    return out


def log_softmax(logits):
    return F.log_softmax(logits, dim=-1)


def entropy(logp):
    """Entropy of a categorical given its log-probabilities; zero-probability entries contribute 0."""
    p = logp.exp()
    return -(p * logp.masked_fill(p == 0, 0.0)).sum(-1)


def log_prob_tables(out: dict, masks, run_manager):
    """Masked log-probabilities of every option of every head, and the manager's term.

    Returns (six (B, n) tables for the controller heads, sub-goal table (B, 8), target table
    (B, 4096), manager (B,)), where ``manager`` is log pi_sub(g) + log pi_tgt(c) for the choice in
    force on manager steps and 0 on the other steps.
    """
    logps = [log_softmax(lg) for lg in apply_masks(out["logits"], masks)]
    sub_lp, tgt_lp = log_softmax(out["subgoal_logits"]), log_softmax(out["target_logits"])
    manager = (sub_lp.gather(-1, out["subgoal"].unsqueeze(-1)) + tgt_lp.gather(-1, out["target"].unsqueeze(-1))).squeeze(-1)
    return logps, sub_lp, tgt_lp, torch.where(run_manager, manager, torch.zeros_like(manager))


def joint_log_prob(head_logp: list, actions, manager):
    """sum_h log pi_h(a_h) + the manager's term. head_logp: six (B, n) tables; actions: (B, 6) long."""
    return sum(lp.gather(-1, actions[..., h:h + 1]).squeeze(-1) for h, lp in enumerate(head_logp)) + manager


def evaluate(out: dict, masks, actions, run_manager) -> dict:
    """Log-probability and entropies of one step under the policy output ``out``.

    out: the network's output dict; ``out["subgoal"]`` / ``out["target"]`` are the manager choice
    in force. actions: (B, 6) long. run_manager: (B,) bool. Returns per-sample tensors:
    ``logp`` (joint), ``entropy`` (B, 8) in the order of ``ENTROPY_NAMES``, and the per-head
    log-probability tables (for the distillation term).
    """
    logps, sub_lp, tgt_lp, manager = log_prob_tables(out, masks, run_manager)
    ent = torch.stack([entropy(lp) for lp in logps] + [entropy(sub_lp), entropy(tgt_lp)], -1)
    return {"logp": joint_log_prob(logps, actions, manager), "entropy": ent, "head_logp": logps,
            "subgoal_logp": sub_lp, "target_logp": tgt_lp}


def step_summary(out: dict, masks, run_manager):
    """Everything the collector needs from one forward pass as one (B, 20) tensor: one device transfer.

    Columns: the 15 masked head log-probabilities, the manager's log-probability term, the two
    values, and the sub-goal and target in force (as floats; both are exact).
    """
    logps, _, _, manager = log_prob_tables(out, masks, run_manager)
    tail = torch.stack([manager, out["value"], out["value_int"], out["subgoal"].to(manager.dtype),
                        out["target"].to(manager.dtype)], -1)
    return torch.cat(logps + [tail], -1)


def sample_from_summary(summary, generator=None, greedy_heads: tuple = ()):
    """Sample the six heads from a ``step_summary`` on the CPU.

    Returns (heads (B, 6) long, joint logp (B,), value, value_int, subgoal long, target long).
    Heads listed in ``greedy_heads`` take their most likely option instead of a sample.
    """
    n = sum(spec.HEAD_SIZES)
    tables = list(summary[:, :n].split(spec.HEAD_SIZES, -1))
    heads = torch.stack([torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1) for lp in tables], -1)
    for h in greedy_heads:
        heads[:, h] = tables[h].argmax(-1)
    logp = joint_log_prob(tables, heads, summary[:, n])
    return heads, logp, summary[:, n + 1], summary[:, n + 2], summary[:, n + 3].long(), summary[:, n + 4].long()


_BUTTON_TABLES = None


def heads_to_buttons(heads: np.ndarray) -> np.ndarray:
    """(B, 6) head indices -> (B, 9) int8 buttons, using ``spec.heads_to_buttons`` for the mapping."""
    global _BUTTON_TABLES
    if _BUTTON_TABLES is None:
        tables = []
        for h, size in enumerate(spec.HEAD_SIZES):
            rows = []
            for v in range(size):
                one = np.zeros(len(spec.HEAD_SIZES), np.int64)
                one[h] = v
                rows.append(spec.heads_to_buttons(one))
            tables.append(np.stack(rows))
        _BUTTON_TABLES = tables
    return sum(table[heads[:, h]] for h, table in enumerate(_BUTTON_TABLES)).astype(np.int8)


# ---------------------------------------------------------------------------------------------
# Replaying chunks (truncated BPTT)
# ---------------------------------------------------------------------------------------------
_REPLAY_KEYS = ("value", "value_int", "subgoal", "target", "subgoal_logits", "target_logits")


def unroll(policy, layout: StateLayout, batch: dict, by_steps: bool = False) -> dict:
    """Re-evaluate a minibatch of chunks ``(B, L, ...)`` with the current parameters.

    Each chunk starts from its stored recurrent state; the state is reset wherever an episode
    starts inside the chunk, and the manager's stored choices are forced, so the controller is
    evaluated under the sub-goal and target it actually had. The network's sequence mode is used
    when it has one (it batches everything but the GRU over all B * L steps); otherwise, or with
    ``by_steps=True``, the single-step forward is called L times. Returns flattened ``(B * L, ...)``
    tensors: logp, entropy, value, value_int and the log-probability tables.
    """
    B, L = batch["step"].shape
    start = batch["step"] == 0
    run_manager = batch["step"] % spec.MANAGER_PERIOD == 0
    forced = {"subgoal": batch["subgoal"], "target": batch["target"]}
    state = layout.from_rows(batch["state"])
    sequence = None if by_steps else getattr(policy, "forward_sequence", None)
    if sequence is not None:
        out, _ = sequence(batch["rays"], batch["ents"], batch["ent_mask"], batch["scal"], batch["map"], state=state,
                          masks=batch["masks"], run_manager=run_manager, forced=forced, sample=False, reset=start)
        flat = {k: out[k].flatten(0, 1) for k in _REPLAY_KEYS}
        flat["logits"] = [lg.flatten(0, 1) for lg in out["logits"]]
    else:
        initial = policy.initial_state(B, batch["step"].device)
        outs = []
        for l in range(L):
            state = layout.reset(state, initial, start[:, l])
            out, state = policy(batch["rays"][:, l], batch["ents"][:, l], batch["ent_mask"][:, l], batch["scal"][:, l],
                                batch["map"][:, l], state=state, masks=batch["masks"][:, l], run_manager=run_manager[:, l],
                                forced={k: v[:, l] for k, v in forced.items()}, sample=False)
            outs.append(out)
        flat = {k: torch.stack([o[k] for o in outs], 1).flatten(0, 1) for k in _REPLAY_KEYS}
        flat["logits"] = [torch.stack([o["logits"][h] for o in outs], 1).flatten(0, 1) for h in range(len(spec.HEAD_SIZES))]
    new = evaluate(flat, batch["masks"].flatten(0, 1), batch["actions"].flatten(0, 1), run_manager.flatten())
    new["value"], new["value_int"] = flat["value"], flat["value_int"]
    return new


def _masked_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _cross_entropy(logp_table, labels, mask):
    """Mean -log p(label) over ``mask``; labels the action mask forbids (probability zero) are skipped."""
    lp = logp_table.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ok = mask & torch.isfinite(lp)
    return _masked_mean(-lp.masked_fill(~ok, 0.0), ok)


def ppo_loss(new: dict, batch: dict, cfg: PPOConfig, ent_scale: float = 1.0, distill_weight: float = 0.0):
    """The loss of the module docstring for one minibatch. Returns (scalar loss, dict of float stats)."""
    flat = lambda k: batch[k].flatten(0, 1)
    adv, ret, ret_int = flat("adv"), flat("ret"), flat("ret_int")
    old_logp, old_v, old_v_int = flat("logp"), flat("value"), flat("value_int")
    manager = flat("step") % spec.MANAGER_PERIOD == 0

    log_ratio = new["logp"] - old_logp
    ratio = log_ratio.exp()
    pi_loss = -torch.min(ratio * adv, ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv).mean()

    def value_loss(v, v_old, target):
        err = (v - target) ** 2
        if cfg.value_clip is not None:
            clipped = v_old + (v - v_old).clamp(-cfg.value_clip, cfg.value_clip)
            err = torch.max(err, (clipped - target) ** 2)
        return 0.5 * err.mean()

    v_loss = value_loss(new["value"], old_v, ret)
    v_int_loss = value_loss(new["value_int"], old_v_int, ret_int)

    ent = new["entropy"]                                             # (S, 8)
    n_heads = len(spec.HEAD_SIZES)
    head_ent = ent[:, :n_heads].mean(0)
    mgr_ent = torch.stack([_masked_mean(ent[:, n_heads], manager), _masked_mean(ent[:, n_heads + 1], manager)])
    all_ent = torch.cat([head_ent, mgr_ent])
    ent_bonus = (all_ent * torch.tensor(cfg.ent_coef, device=ent.device, dtype=ent.dtype)).sum() * ent_scale

    labelled = flat("lab_mask")
    distill = torch.zeros((), device=ent.device)
    if distill_weight > 0 and bool(labelled.any()):
        lab_heads = flat("lab_heads").long()
        d = cfg.distill_coef
        for h in range(n_heads):
            distill = distill + d[h] * _cross_entropy(new["head_logp"][h], lab_heads[:, h], labelled)
        both = labelled & manager
        distill = distill + d[n_heads] * _cross_entropy(new["subgoal_logp"], flat("lab_subgoal").long(), both)
        distill = distill + d[n_heads + 1] * _cross_entropy(new["target_logp"], flat("lab_target").long(), both)

    loss = pi_loss + cfg.vf_coef * v_loss + cfg.vf_int_coef * v_int_loss - ent_bonus + distill_weight * distill

    with torch.no_grad():
        stats = {"loss": loss.item(), "loss_pi": pi_loss.item(), "loss_v": v_loss.item(), "loss_v_int": v_int_loss.item(),
                 "loss_distill": distill.item(), "kl": (ratio - 1.0 - log_ratio).mean().item(),
                 "clip_frac": ((ratio - 1.0).abs() > cfg.clip).float().mean().item(),
                 "ev": (1.0 - (ret - new["value"]).var() / ret.var().clamp(min=1e-8)).item(),
                 "labelled": labelled.float().mean().item()}
        stats.update({f"ent_{name}": all_ent[i].item() for i, name in enumerate(ENTROPY_NAMES)})
    return loss, stats


# ---------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------
def linear(start: float, end: float, progress: float) -> float:
    """Linear interpolation from ``start`` to ``end`` as progress goes from 0 to 1."""
    return start + (end - start) * min(max(progress, 0.0), 1.0)


@dataclass
class DistillSchedule:
    """Weight of the distillation term, lowered only while the held-out exit rate holds.

    The weight moves along ``levels``. ``report(exit_rate)`` is called with a fresh held-out exit
    rate (the trainer calls it after each held-out evaluation):

    * if the rate has not fallen by more than ``tolerance`` below the rate recorded at the last
      step down, the weight steps down one level and that rate becomes the new reference;
    * otherwise the weight holds; after ``patience`` such reports in a row it steps back up one
      level (the reference is kept, so the rate has to recover before the next step down).

    With no reports the weight never changes: it is not annealed on a timer.
    """

    levels: tuple = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
    tolerance: float = 0.03
    patience: int = 2
    index: int = 0
    reference: float | None = None
    bad_reports: int = 0
    history: list = field(default_factory=list)

    @property
    def weight(self) -> float:
        return float(self.levels[self.index])

    def report(self, exit_rate: float | None) -> float:
        """Take one held-out exit rate (None = no measurement: hold) and return the new weight."""
        if exit_rate is None:
            return self.weight
        if self.reference is None or exit_rate >= self.reference - self.tolerance:
            self.bad_reports = 0
            self.reference = float(exit_rate)
            self.index = min(self.index + 1, len(self.levels) - 1)
        else:
            self.bad_reports += 1
            if self.bad_reports >= self.patience:
                self.bad_reports = 0
                self.index = max(self.index - 1, 0)
        self.history.append((float(exit_rate), self.weight))
        return self.weight

    def state_dict(self) -> dict:
        return {"index": self.index, "reference": self.reference, "bad_reports": self.bad_reports, "history": list(self.history)}

    def load_state_dict(self, state: dict) -> None:
        self.index, self.reference = int(state["index"]), state["reference"]
        self.bad_reports, self.history = int(state["bad_reports"]), list(state.get("history", []))


# ---------------------------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------------------------
class PPO:
    """Owns the optimiser and runs the epochs of one update over a filled rollout buffer."""

    def __init__(self, policy, cfg: PPOConfig, device, total_updates: int = 1, seed: int = 0):
        self.policy, self.cfg, self.device = policy, cfg, torch.device(device)
        self.layout = StateLayout(policy, self.device)
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.total_updates = max(int(total_updates), 1)
        self.updates = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.norm = RunningNorm() if cfg.normalise_returns else None          # extrinsic returns
        self.norm_int = RunningNorm() if cfg.normalise_returns else None      # intrinsic returns

    @property
    def progress(self) -> float:
        return min(self.updates / self.total_updates, 1.0)

    @property
    def lr(self) -> float:
        return self.cfg.lr * linear(1.0, self.cfg.lr_final, self.progress)

    @property
    def ent_scale(self) -> float:
        return linear(1.0, self.cfg.ent_final, self.progress)

    def update(self, buf: RolloutBuffer, distill_weight: float = 0.0) -> dict:
        """Compute returns for the rollout in ``buf`` and run the PPO epochs. Returns mean stats."""
        cfg = self.cfg
        buf.finish(cfg.gamma, cfg.lam, cfg.gamma_int, cfg.int_coef, cfg.episodic_intrinsic,
                   cfg.bootstrap_on_truncation, cfg.normalise_adv, self.norm, self.norm_int)
        for group in self.opt.param_groups:
            group["lr"] = self.lr
        params = [p for p in self.policy.parameters() if p.requires_grad]
        sums: dict[str, float] = {}
        n_batches, samples, stop = 0, 0, False
        for _ in range(cfg.epochs):
            for batch in buf.minibatches(cfg.minibatch_chunks, self.generator):
                new = unroll(self.policy, self.layout, batch)
                loss, stats = ppo_loss(new, batch, cfg, self.ent_scale, distill_weight)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
                self.opt.step()
                for k, v in stats.items():
                    sums[k] = sums.get(k, 0.0) + v
                n_batches += 1
                samples += batch["step"].numel()
                if cfg.target_kl is not None and stats["kl"] > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.updates += 1
        out = {k: v / max(n_batches, 1) for k, v in sums.items()}
        out.update({"lr": self.lr, "ent_scale": self.ent_scale, "samples": samples, "minibatches": n_batches})
        return out
