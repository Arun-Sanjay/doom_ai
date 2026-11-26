"""Rollout storage, generalised advantage estimation and chunking for truncated BPTT.

Training-time scaffolding, never imported by the agent. Plain torch, no RL library.

A rollout is ``T`` steps of ``N`` environments, stored time-major as ``(T, N, ...)`` tensors on
one device. Episodes start and end anywhere inside a rollout:

* ``step[t, n]`` is the index of the observation inside its episode; ``step == 0`` marks an
  episode start (the recurrent state and the manager's held choice are reset there) and
  ``step % MANAGER_PERIOD == 0`` marks a manager step.
* ``terminated[t, n]`` (exit or death) and ``truncated[t, n]`` (tick cap) describe what happened
  after the action of step t. A truncated step also stores ``bootstrap[t, n]``, the value of the
  final observation of that episode, because the next stored observation belongs to a new episode.

For the update the rollout is cut into chunks of ``chunk`` consecutive steps per environment. The
recurrent state in force at the first step of every chunk is stored, so a chunk can be replayed
from it; gradients do not flow across chunk boundaries (truncated backpropagation through time).
"""

from __future__ import annotations

import torch

from training import paths  # noqa: F401
from doomnet import spec
from training.rl.vecenv import OBS_FIELDS


def compute_gae(rewards, values, last_value, terminated, truncated, bootstrap, gamma: float, lam: float,
                episodic: bool = True):
    """Generalised advantage estimation over a rollout with episode ends inside it.

    All inputs are ``(T, N)`` except ``last_value`` ``(N,)``, the value of the observation after
    the last step. With V' the value of the next state:

        V'_t    = 0              if the episode terminated at t (exit or death: nothing follows)
                = bootstrap_t    if it was truncated at t (the tick cap is not a property of the state)
                = V_{t+1}        otherwise (last_value after the final step)
        delta_t = r_t + gamma * V'_t - V_t
        A_t     = delta_t + gamma * lam * A_{t+1}, with A_{t+1} taken as 0 across an episode end

    ``episodic=False`` ignores episode ends altogether (V'_t = V_{t+1} always): the non-episodic
    treatment of an intrinsic reward stream. Returns (advantages, returns = advantages + values).
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    done = terminated | truncated
    running = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        v_next = last_value if t == T - 1 else values[t + 1]
        if episodic:
            v_next = torch.where(truncated[t], bootstrap[t], v_next)
            v_next = torch.where(terminated[t], torch.zeros_like(v_next), v_next)
            running = running * (~done[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * v_next - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv, adv + values


class RunningNorm:
    """Running mean and standard deviation of a return stream (debiased exponential averages).

    The value heads predict returns in normalised units, ``(R - mean) / std``, so the value loss
    stays of order one whatever the reward scale and does not swamp the policy gradient under
    global-norm clipping. Predictions are mapped back to return units before GAE.
    """

    def __init__(self, beta: float = 0.98):
        self.beta, self.m1, self.m2, self.w = beta, 0.0, 0.0, 0.0

    @property
    def mean(self) -> float:
        return self.m1 / self.w if self.w > 0 else 0.0

    @property
    def std(self) -> float:
        if self.w <= 0:
            return 1.0
        return max(self.m2 / self.w - self.mean ** 2, 1e-6) ** 0.5

    def update(self, x) -> None:
        self.m1 = self.beta * self.m1 + (1 - self.beta) * float(x.mean())
        self.m2 = self.beta * self.m2 + (1 - self.beta) * float((x * x).mean())
        self.w = self.beta * self.w + (1 - self.beta)

    def normalise(self, x):
        return (x - self.mean) / self.std

    def denormalise(self, x):
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"m1": self.m1, "m2": self.m2, "w": self.w}

    def load_state_dict(self, state: dict) -> None:
        self.m1, self.m2, self.w = float(state["m1"]), float(state["m2"]), float(state["w"])


class StateLayout:
    """Where the batch axis sits in each tensor of a policy's recurrent-state dict.

    The policy's state is a dict of tensors (GRU state plus the manager's held outputs). The batch
    axis of each entry is found by comparing ``initial_state`` at two batch sizes, so nothing here
    depends on the layout the network chose.
    """

    def __init__(self, policy, device):
        a, b = policy.initial_state(2, device), policy.initial_state(3, device)
        self.axis = {}
        for k in a:
            if not torch.is_tensor(a[k]):
                raise TypeError(f"recurrent state entry {k!r} is not a tensor")
            diff = [i for i, (x, y) in enumerate(zip(a[k].shape, b[k].shape)) if x != y]
            if len(diff) != 1:
                raise ValueError(f"cannot find the batch axis of recurrent state entry {k!r}: {tuple(a[k].shape)}")
            self.axis[k] = diff[0]

    def to_rows(self, state: dict) -> dict:
        """Model layout -> batch-first (B, ...)."""
        return {k: v.movedim(self.axis[k], 0) for k, v in state.items()}

    def from_rows(self, rows: dict) -> dict:
        """Batch-first -> model layout."""
        return {k: v.movedim(0, self.axis[k]) for k, v in rows.items()}

    def take(self, state: dict, index) -> dict:
        """Select batch rows ``index`` (a 1-D long tensor)."""
        return {k: v.index_select(self.axis[k], index) for k, v in state.items()}

    def reset(self, state: dict, initial: dict, mask) -> dict:
        """Replace the rows where ``mask`` (B,) is True by the initial state."""
        out = {}
        for k, v in state.items():
            shape = [1] * v.dim()
            shape[self.axis[k]] = -1
            out[k] = torch.where(mask.view(shape), initial[k], v)
        return out

    @staticmethod
    def detach(state: dict) -> dict:
        return {k: v.detach() for k, v in state.items()}


NETWORK_INPUTS = ("rays", "ents", "ent_mask", "scal", "map")      # stored on the learner device
_TRAIN_FIELDS = ("actions", "subgoal", "target", "logp", "value", "value_int", "adv", "ret", "ret_int", "forced")


class RolloutBuffer:
    """Fixed-size storage for one rollout and its minibatches of chunks.

    The network inputs (rays, entities, scalars, map crops) are written straight to the learner
    device, where the forward pass reads them. Everything else is a few numbers per step (masks,
    step index, labels, actions, log-probs, values, rewards, episode ends): it stays on the CPU
    while the rollout is collected, so the per-step bookkeeping costs no device calls, and is moved
    to the device once, in ``finish()``.
    """

    def __init__(self, steps: int, n_envs: int, chunk: int, device):
        if steps % chunk:
            raise ValueError(f"rollout length {steps} must be a multiple of the chunk length {chunk}")
        self.T, self.N, self.L = steps, n_envs, chunk
        self.device = torch.device(device)

        def z(*shape, dtype=torch.float32, device="cpu"):
            return torch.zeros((steps, n_envs, *shape), dtype=dtype, device=device)

        # Observations, in the dtypes the workers produce (map crops stay uint8 until used).
        self.obs = {k: z(*shape, dtype=torch.from_numpy(_np_zero(dtype)).dtype,
                         device=self.device if k in NETWORK_INPUTS else "cpu")
                    for k, (shape, dtype) in OBS_FIELDS.items()}
        # What the policy did.
        self.actions = z(len(spec.HEAD_SIZES), dtype=torch.long)
        self.subgoal = z(dtype=torch.long)
        self.target = z(dtype=torch.long)
        self.logp = z()
        self.value = z()
        self.value_int = z()
        # True where the executed action (and, on a manager step, the manager's choice) was the teacher's
        # and not the policy's own sample. ``logp`` is always the policy's log-probability of what was executed.
        self.forced = z(dtype=torch.bool)
        # What happened.
        self.reward = z()
        self.reward_int = z()
        self.terminated = z(dtype=torch.bool)
        self.truncated = z(dtype=torch.bool)
        self.bootstrap = z()
        self.bootstrap_int = z()
        self.last_value = torch.zeros(n_envs)
        self.last_value_int = torch.zeros(n_envs)
        # Filled by finish().
        self.adv = z()
        self.ret = z()
        self.ret_int = z()
        self.state: dict[str, torch.Tensor] = {}       # batch-first initial states, (T / L, N, ...), on the device
        self._train: dict[str, torch.Tensor] = {}      # device copies of the small fields, made by finish()
        self._arange = torch.arange(chunk, device=self.device)

    # ------------------------------------------------------------------------------- writing
    def put_obs(self, t: int, rows: slice, arrays: dict) -> dict:
        """Copy the observation rows of the vector environment into step t; return the stored tensors.

        The returned dict holds views of the storage: network inputs on the device, the rest on the CPU.
        """
        out = {}
        for k in OBS_FIELDS:
            dst = self.obs[k][t, rows]
            dst.copy_(torch.from_numpy(arrays[k][rows]))
            out[k] = dst
        return out

    def put_state(self, t: int, rows: slice, state_rows: dict) -> None:
        """Store the batch-first recurrent state in force at step t (t must be a chunk start)."""
        c = t // self.L
        for k, v in state_rows.items():
            if k not in self.state:
                self.state[k] = torch.zeros((self.T // self.L, self.N, *v.shape[1:]), dtype=v.dtype, device=self.device)
            self.state[k][c, rows] = v.detach().to(self.device)

    def put_action(self, t: int, rows: slice, actions, subgoal, target, logp, value, value_int, forced=None) -> None:
        self.forced[t, rows] = False if forced is None else forced
        self.actions[t, rows] = actions
        self.subgoal[t, rows] = subgoal
        self.target[t, rows] = target
        self.logp[t, rows] = logp
        self.value[t, rows] = value
        self.value_int[t, rows] = value_int

    def put_outcome(self, t: int, rows: slice, reward, reward_int, terminated, truncated, bootstrap, bootstrap_int) -> None:
        self.reward[t, rows] = reward
        self.reward_int[t, rows] = reward_int
        self.terminated[t, rows] = terminated
        self.truncated[t, rows] = truncated
        self.bootstrap[t, rows] = bootstrap
        self.bootstrap_int[t, rows] = bootstrap_int

    # ------------------------------------------------------------------------------- returns
    def finish(self, gamma: float, lam: float, gamma_int: float, int_coef: float, episodic_intrinsic: bool = True,
               bootstrap_on_truncation: bool = True, normalise: bool = True,
               norm: RunningNorm | None = None, norm_int: RunningNorm | None = None) -> None:
        """Compute both advantage streams and returns, then the combined advantage used by the loss.

        adv = A_ext + int_coef * A_int, normalised to zero mean and unit variance over the rollout.
        With ``bootstrap_on_truncation=False`` the tick cap is treated as a terminal state.
        ``norm`` / ``norm_int`` hold the running statistics of the two return streams: stored value
        predictions are mapped to return units with them, the statistics are then updated with this
        rollout's returns, and ``ret`` / ``ret_int`` are left in normalised units (the value heads'
        regression targets). Finally the small per-step fields are copied to the device for the update.
        """
        term, trunc = self.terminated, self.truncated
        if not bootstrap_on_truncation:
            term, trunc = term | trunc, torch.zeros_like(trunc)
        real = (lambda x: x) if norm is None else norm.denormalise
        real_int = (lambda x: x) if norm_int is None else norm_int.denormalise
        adv_ext, ret = compute_gae(self.reward, real(self.value), real(self.last_value), term, trunc,
                                   real(self.bootstrap), gamma, lam)
        adv_int, ret_int = compute_gae(self.reward_int, real_int(self.value_int), real_int(self.last_value_int), term, trunc,
                                       real_int(self.bootstrap_int), gamma_int, lam, episodic=episodic_intrinsic)
        self.ret_raw, self.ret_int_raw = ret, ret_int           # return units, for logging
        if norm is not None:
            norm.update(ret)
            ret = norm.normalise(ret)
        if norm_int is not None:
            norm_int.update(ret_int)
            ret_int = norm_int.normalise(ret_int)
        self.ret, self.ret_int = ret, ret_int
        adv = adv_ext + int_coef * adv_int
        if normalise:
            own = adv[~self.forced]                 # the policy term uses only the steps the policy chose itself
            if own.numel() < 2:
                own = adv
            adv = (adv - own.mean()) / (own.std() + 1e-8)
        self.adv = adv
        self._train = {k: v.to(self.device) for k, v in self.obs.items() if k not in NETWORK_INPUTS}
        self._train.update({k: getattr(self, k).to(self.device) for k in _TRAIN_FIELDS})

    # ------------------------------------------------------------------------------- reading
    def num_chunks(self) -> int:
        return (self.T // self.L) * self.N

    def minibatches(self, chunks_per_batch: int, generator=None):
        """Yield minibatches of whole chunks in random order (call ``finish()`` first).

        Each batch is a dict of ``(B, L, ...)`` tensors on the device (B chunks of L consecutive
        steps) plus ``state``, the batch-first recurrent state at the first step of each chunk.
        """
        perm = torch.randperm(self.num_chunks(), generator=generator)
        for i in range(0, len(perm), chunks_per_batch):
            idx = perm[i:i + chunks_per_batch].to(self.device)
            c, n = idx // self.N, idx % self.N
            tt = c[:, None] * self.L + self._arange[None, :]          # (B, L) time index of every sample
            nn = n[:, None].expand_as(tt)
            batch = {k: self.obs[k][tt, nn] for k in NETWORK_INPUTS}
            batch.update({k: v[tt, nn] for k, v in self._train.items()})
            batch["state"] = {k: v[c, n] for k, v in self.state.items()}
            yield batch


def _np_zero(dtype):
    import numpy as np
    return np.zeros(1, dtype)
