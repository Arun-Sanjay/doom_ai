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

Teacher mixing. The workers attach the teacher's labels to every observation they can. On a step
that carries controller labels a coin with probability ``beta`` decides whether the teacher's
action is executed instead of the policy's sample (labelled heads only; the unlabelled heads keep
the sample). On a manager step the same coin decides whether the manager's sub-goal and target
are forced to the teacher's, through the network's ``forced`` argument, so that the value field
and the controller are conditioned on the forced choice. Steps without controller labels (an
enemy is visible, or the map has no labels) are never mixed. The buffer stores what was executed,
the policy's log-probability of it and the ``forced`` flag.

When an episode was cut by the tick cap, the value of its final observation is computed with one
extra small forward pass and stored as the bootstrap value of that step. Device calls per step are
kept to a minimum (inputs in, one summary out): on a GPU the cost of a step is mostly per call,
not per row.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import deps
from training.rl.buffer import NETWORK_INPUTS, RolloutBuffer, StateLayout
from training.rl.ppo import heads_to_buttons, sample_from_summary, step_summary
from training.rl.vecenv import VecConfig, VecEnv


def _flags(masks, step, device):
    """(masks (B, 2), run_manager (B,), episode start (B,)) on the device, sent as one tensor."""
    flags = torch.stack([masks[:, 0], masks[:, 1], step % spec.MANAGER_PERIOD == 0, step == 0], -1).to(device)
    return flags[:, :2], flags[:, 2], flags[:, 3]


def _forward(policy, obs: dict, masks, run_manager, state, sample: bool, forced=None):
    return policy(obs["rays"], obs["ents"], obs["ent_mask"], obs["scal"], obs["map"], state=state, masks=masks,
                  run_manager=run_manager, forced=forced, sample=sample)


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
        self.beta = 0.0                 # probability of executing the teacher's action on a labelled step (set by the trainer)
        # Where the last collect() spent its time: in the learner's part of a step (inputs, forward pass,
        # sampling, storing) and waiting for the workers. Mostly waiting = the workers are the limit.
        self.forward_s = self.wait_s = 0.0
        self.partial_forcing = deps.enable_partial_forcing(policy)

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self) -> list:
        """Fill the buffer with ``buf.T`` steps of every row. Returns the summaries of finished episodes."""
        infos = []
        self.forward_s = self.wait_s = 0.0
        for t in range(self.buf.T):
            for g in range(self.vec.groups):
                if self.pending[g]:
                    infos += self._finish(t - 1, g)
                started = time.perf_counter()
                self._begin(t, g)
                self.forward_s += time.perf_counter() - started
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
        # Teacher mixing: one coin per row, used only where the step carries controller labels.
        force = override = forced = None
        if self.beta > 0.0 and bool(obs["lab_mask"].any()):
            force = obs["lab_mask"] & (torch.rand(obs["lab_mask"].shape[0], generator=self.generator) < self.beta)
            override = (obs["lab_heads"], obs["lab_head_mask"] & force[:, None])
            force_mgr = force & obs["lab_mgr"] & (obs["step"] % spec.MANAGER_PERIOD == 0)
            if bool(force_mgr.any()):                # -1 = the manager chooses for itself on that row
                minus = torch.full_like(force_mgr, -1, dtype=torch.long)
                forced = {"subgoal": torch.where(force_mgr, obs["lab_subgoal"].long(), minus).to(self.device),
                          "target": torch.where(force_mgr, obs["lab_target"].long(), minus).to(self.device)}
        inputs = self._inputs(obs)
        if forced is not None and not self.partial_forcing:
            # The network cannot force some rows only: take its own choices first, then run the
            # pass again from the same state with every row forced (own choice or the teacher's).
            own, _ = _forward(self.policy, inputs, masks, run_manager, state, sample=True)
            forced = {k: torch.where(v >= 0, v, own[k]) for k, v in forced.items()}
        out, self.state[g] = _forward(self.policy, inputs, masks, run_manager, state, sample=True, forced=forced)
        heads, logp, value, value_int, subgoal, target = sample_from_summary(
            step_summary(out, masks, run_manager).cpu(), self.generator, override=override)
        self.vec.send(g, heads_to_buttons(heads.numpy()))
        self.buf.put_action(t, rows, heads, subgoal, target, logp, value, value_int, force)
        self.pending[g] = True

    def _finish(self, t: int, g: int) -> list:
        """Wait for the step of group g, store its outcome and the bootstrap values of truncated rows."""
        rows = self.vec.rows(g)
        started = time.perf_counter()
        infos = self.vec.recv(g)
        self.wait_s += time.perf_counter() - started
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


def uniform_summary(masks) -> torch.Tensor:
    """The ``step_summary`` of a policy that is uniform over every head's allowed options.

    masks: (B, 2) bool = (use_ok, attack_ok). This is the random-policy baseline: it obeys the same
    observation masks as the network, so it differs from an untrained network only by the
    network's small initial biases.
    """
    b = masks.shape[0]
    tables = []
    for head, size in enumerate(spec.HEAD_SIZES):
        table = torch.full((b, size), -float(np.log(size)))
        col = {spec.HEAD_USE: 0, spec.HEAD_ATTACK: 1}.get(head)
        if col is not None:
            blocked = ~masks[:, col]
            table[blocked, 0], table[blocked, 1] = 0.0, float("-inf")
        tables.append(table)
    return torch.cat(tables + [torch.zeros(b, 5)], -1)


@torch.no_grad()
def play_episodes(policy, entries: list, device, workers: int, envs_per_worker: int, vec_cfg: VecConfig | None = None,
                  seed: int = 0, episodes_per_row: int = 1, greedy_heads: tuple = (), max_steps: int | None = None) -> list:
    """Play whole episodes with ``policy`` (None = uniform random under the masks); return their summaries.

    The first ``episodes_per_row`` episodes of every environment row are counted, so quick
    episodes are not over-represented the way they would be in a fixed number of steps. Heads in
    ``greedy_heads`` take their most likely option; the others are sampled, as in training.
    """
    device = torch.device(device)
    cfg = dataclasses.replace(vec_cfg or VecConfig(), seed=seed)
    generator = torch.Generator().manual_seed(seed)
    counted: dict[int, list] = {}
    max_steps = max_steps or episodes_per_row * 1100
    with VecEnv(entries, workers, envs_per_worker, cfg) as vec:
        a = vec.arrays
        if policy is not None:
            layout = StateLayout(policy, device)
            initial = policy.initial_state(vec.n, device)
            state = policy.initial_state(vec.n, device)
        for _ in range(max_steps):
            cpu_masks, step = torch.from_numpy(a["masks"].copy()), torch.from_numpy(a["step"].copy())
            if policy is None:
                summary = uniform_summary(cpu_masks)
            else:
                obs = {k: torch.from_numpy(a[k].copy()).to(device) for k in NETWORK_INPUTS}
                masks, run_manager, start = _flags(cpu_masks, step, device)
                out, state = _forward(policy, obs, masks, run_manager, layout.reset(state, initial, start), sample=True)
                summary = step_summary(out, masks, run_manager).cpu()
            heads = sample_from_summary(summary, generator, greedy_heads=greedy_heads)[0]
            vec.send(0, heads_to_buttons(heads.numpy()))
            for info in vec.recv(0):
                if info.get("crashed"):             # a row restarted after a failure: not an episode
                    continue
                row = counted.setdefault(info["row"], [])
                if len(row) < episodes_per_row:
                    row.append(info)
            if len(counted) == vec.n and all(len(row) >= episodes_per_row for row in counted.values()):
                break
    return [info for row in counted.values() for info in row]


def summarise_episodes(results: list) -> dict:
    """Means over episode summaries: the numbers the smoke tests and baselines report."""
    if not results:
        return {"episodes": 0}
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    sem = lambda xs: float(np.std(xs) / np.sqrt(len(xs))) if len(xs) else float("nan")
    with_exit = [r for r in results if r.get("has_exit", True)]
    with_monsters = [r for r in results if r.get("monsters", 0) > 0]
    kills = [r["kills"] for r in results]
    return {"episodes": len(results), "maps": len({r["map"] for r in results}),
            "kills": mean(kills), "kills_sem": sem(kills),
            "kill_frac": mean([r["kills"] / r["monsters"] for r in with_monsters]),
            "death_rate": mean([r["dead"] for r in results]),
            "exit_rate": mean([r["exit"] for r in with_exit]),
            "return": mean([r["ret"] for r in results]), "return_int": mean([r["ret_int"] for r in results]),
            "ep_len": mean([r["len"] for r in results]),
            "cells_seen": mean([r["cells_seen"] for r in results]), "cells_seen_sem": sem([r["cells_seen"] for r in results]),
            "sectors": mean([r.get("sectors", 0) for r in results])}


def heldout_exit_rate(policy, entries: list, device, workers: int, envs_per_worker: int, vec_cfg: VecConfig | None = None,
                      seed: int = 0, max_steps: int = 1100) -> dict:
    """Exit rate of the current policy on held-out maps, one episode per environment row.

    Only the first episode of every row is counted, so quick episodes are not over-represented.
    Decoding and budget follow the agent at evaluation: USE and weapon heads by most-likely choice,
    the other heads sampled, and the assumed time budget fixed at its evaluation default. This is
    the measurement the distillation schedule is gated on.
    """
    cfg = dataclasses.replace(vec_cfg or VecConfig(), p_default_budget=1.0, teacher="none")    # the policy drives alone
    results = play_episodes(policy, entries, device, workers, envs_per_worker, cfg, seed, 1,
                            (spec.HEAD_USE, spec.HEAD_WEAPON), max_steps)
    with_exit = [r for r in results if r.get("has_exit", True)]
    return {"exit_rate": float(np.mean([r["exit"] for r in with_exit])) if with_exit else 0.0,
            "episodes": len(results), "return": float(np.mean([r["ret"] for r in results])) if results else 0.0}
