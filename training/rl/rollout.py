"""Collector: runs the policy in the vector environment and fills the rollout buffer.

Training-time scaffolding, never imported by the agent.

Per step and per group of rows the learner does one batched forward pass:

1. copy the rows' encoded observations from shared memory into the buffer (network inputs go
   straight to the learner device);
2. reset the recurrent state and the manager's held choice of rows whose episode just started;
3. forward: the network samples the manager's sub-goal and target on manager steps and returns
   logits for the six controller heads; their masked log-probabilities come back in one transfer
   and the heads are sampled on the CPU;
4. send the actions; the workers step while the learner stores what it did.

When an episode was cut by the tick cap, the value of its final observation is computed with one
extra small forward pass and stored as the bootstrap value of that step. Device calls per step are
kept to a minimum (inputs in, one summary out): on a GPU the cost of a step is mostly per call,
not per row.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from training import paths  # noqa: F401
from doomnet import spec
from training.rl.buffer import NETWORK_INPUTS, RolloutBuffer, StateLayout
from training.rl.ppo import heads_to_buttons, sample_from_summary, step_summary
from training.rl.vecenv import VecConfig, VecEnv


def _flags(masks, step, device):
    """(masks (B, 2), run_manager (B,), episode start (B,)) on the device, sent as one tensor."""
    flags = torch.stack([masks[:, 0], masks[:, 1], step % spec.MANAGER_PERIOD == 0, step == 0], -1).to(device)
    return flags[:, :2], flags[:, 2], flags[:, 3]


def _forward(policy, obs: dict, masks, run_manager, state, sample: bool):
    return policy(obs["rays"], obs["ents"], obs["ent_mask"], obs["scal"], obs["map"], state=state, masks=masks,
                  run_manager=run_manager, sample=sample)


class Collector:
    """Holds the recurrent state of every row between rollouts."""

    def __init__(self, vec: VecEnv, policy, buf: RolloutBuffer, device, seed: int = 0, bootstrap_on_truncation: bool = True):
        self.vec, self.policy, self.buf = vec, policy, buf
        self.device = torch.device(device)
        self.layout = StateLayout(policy, self.device)
        self.bootstrap_on_truncation = bootstrap_on_truncation
        self.initial = [policy.initial_state(vec.group_size, self.device) for _ in range(vec.groups)]
        self.state = [policy.initial_state(vec.group_size, self.device) for _ in range(vec.groups)]
        self.pending = [False] * vec.groups
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> list:
        """Fill the buffer with ``buf.T`` steps of every row. Returns the summaries of finished episodes."""
        infos = []
        for t in range(self.buf.T):
            for g in range(self.vec.groups):
                if self.pending[g]:
                    infos += self._finish(t - 1, g)
                self._begin(t, g)
        for g in range(self.vec.groups):
            infos += self._finish(self.buf.T - 1, g)
            self._last_values(g)
        return infos

    def _inputs(self, obs: dict) -> dict:
        return {k: (obs[k] if obs[k].device == self.device else obs[k].to(self.device)) for k in NETWORK_INPUTS}

    def _begin(self, t: int, g: int) -> None:
        """Observation of step t for group g -> forward -> actions sent to the workers."""
        rows = self.vec.rows(g)
        obs = self.buf.put_obs(t, rows, self.vec.arrays)
        masks, run_manager, start = _flags(obs["masks"], obs["step"], self.device)
        state = self.layout.reset(self.state[g], self.initial[g], start)
        if t % self.buf.L == 0:
            self.buf.put_state(t, rows, self.layout.to_rows(state))
        out, self.state[g] = _forward(self.policy, self._inputs(obs), masks, run_manager, state, sample=True)
        heads, logp, value, value_int, subgoal, target = sample_from_summary(
            step_summary(out, masks, run_manager).cpu(), self.generator)
        self.vec.send(g, heads_to_buttons(heads.numpy()))
        self.buf.put_action(t, rows, heads, subgoal, target, logp, value, value_int)
        self.pending[g] = True

    def _finish(self, t: int, g: int) -> list:
        """Wait for the step of group g, store its outcome and the bootstrap values of truncated rows."""
        rows = self.vec.rows(g)
        infos = self.vec.recv(g)
        a = self.vec.arrays
        truncated = torch.from_numpy(a["truncated"][rows].copy())
        boot = torch.zeros(self.vec.group_size)
        boot_int = torch.zeros(self.vec.group_size)
        if self.bootstrap_on_truncation and bool(truncated.any()):
            sel = torch.nonzero(truncated).squeeze(1)
            idx = sel.numpy() + rows.start
            obs = {k: torch.from_numpy(a["fin_" + k][idx]).to(self.device) for k in NETWORK_INPUTS}
            masks, run_manager, _ = _flags(torch.from_numpy(a["fin_masks"][idx]), torch.from_numpy(a["fin_step"][idx]), self.device)
            out, _ = _forward(self.policy, obs, masks, run_manager, self.layout.take(self.state[g], sel.to(self.device)), sample=False)
            values = torch.stack([out["value"], out["value_int"]], -1).cpu()
            boot[sel], boot_int[sel] = values[:, 0], values[:, 1]
        self.buf.put_outcome(t, rows, torch.from_numpy(a["reward"][rows].copy()), torch.from_numpy(a["reward_int"][rows].copy()),
                             torch.from_numpy(a["terminated"][rows].copy()), truncated, boot, boot_int)
        self.pending[g] = False
        return infos

    def _last_values(self, g: int) -> None:
        """Value of the observation after the last stored step (the workers are idle here)."""
        rows = self.vec.rows(g)
        a = self.vec.arrays
        obs = {k: torch.from_numpy(a[k][rows].copy()).to(self.device) for k in NETWORK_INPUTS}
        masks, run_manager, start = _flags(torch.from_numpy(a["masks"][rows].copy()), torch.from_numpy(a["step"][rows].copy()), self.device)
        state = self.layout.reset(self.state[g], self.initial[g], start)
        out, _ = _forward(self.policy, obs, masks, run_manager, state, sample=False)
        values = torch.stack([out["value"], out["value_int"]], -1).cpu()
        self.buf.last_value[rows], self.buf.last_value_int[rows] = values[:, 0], values[:, 1]


@torch.no_grad()
def heldout_exit_rate(policy, entries: list, device, workers: int, envs_per_worker: int, vec_cfg: VecConfig | None = None,
                      seed: int = 0, max_steps: int = 1100) -> dict:
    """Exit rate of the current policy on held-out maps, one episode per environment row.

    Only the first episode of every row is counted, so quick episodes are not over-represented.
    Decoding and budget follow the agent at evaluation: USE and weapon heads by most-likely choice,
    the other heads sampled, and the assumed time budget fixed at its evaluation default. This is
    the measurement the distillation schedule is gated on.
    """
    device = torch.device(device)
    cfg = dataclasses.replace(vec_cfg or VecConfig(), seed=seed, p_default_budget=1.0)
    layout = StateLayout(policy, device)
    generator = torch.Generator().manual_seed(seed)
    first: dict[int, dict] = {}
    with VecEnv(entries, workers, envs_per_worker, cfg) as vec:
        a = vec.arrays
        initial = policy.initial_state(vec.n, device)
        state = policy.initial_state(vec.n, device)
        for _ in range(max_steps):
            obs = {k: torch.from_numpy(a[k].copy()).to(device) for k in NETWORK_INPUTS}
            masks, run_manager, start = _flags(torch.from_numpy(a["masks"].copy()), torch.from_numpy(a["step"].copy()), device)
            out, state = _forward(policy, obs, masks, run_manager, layout.reset(state, initial, start), sample=True)
            heads = sample_from_summary(step_summary(out, masks, run_manager).cpu(), generator,
                                        greedy_heads=(spec.HEAD_USE, spec.HEAD_WEAPON))[0]
            vec.send(0, heads_to_buttons(heads.numpy()))
            for info in vec.recv(0):
                first.setdefault(info["row"], info)
            if len(first) == vec.n:
                break
    results = list(first.values())
    with_exit = [r for r in results if r.get("has_exit", True)]
    return {"exit_rate": float(np.mean([r["exit"] for r in with_exit])) if with_exit else 0.0,
            "episodes": len(results), "return": float(np.mean([r["ret"] for r in results])) if results else 0.0}
