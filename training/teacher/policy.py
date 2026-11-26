"""Training-time scaffolding. Never imported by the agent in submission/.

`TeacherPolicy`: the navigation teacher as a stand-alone policy for the episode runner, used to
measure the teacher itself and to record demonstrations.

It runs the agent's own front end (`Odometry` -> `MemoryMap` -> `Encoder`, the same classes the
agent in submission/ uses) and hands exactly their outputs to `Teacher.label`:

    exit_obs = sensor.read()                                   # see ExitSensor
    odometry.update(prev_action, obs)                          # not at step 0
    memory.update(pose, obs, prev_action, exit_obs, step)
    crop = memory.crop(pose)
    scal = encoder.encode(obs, pose, prev_action, step, budget_steps, exit_obs)["scal"]
    label = teacher.label(obs, crop, pose, scal, step, budget_steps, prev_action)

The environment object is touched in one place only, `ExitSensor`: the exit pad cannot be seen by
the range sensors, the agent finds it with a small CNN on the camera frame, and while that network
is being trained the same three numbers (seen, forward, left) come from `training.exit_label`.
Nothing else about the environment reaches the teacher.
"""

from __future__ import annotations

import time

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec
from doomnet.encode import Encoder
from doomnet.memory import MemoryMap
from doomnet.odometry import Odometry

from .core import FIGHT, GO_EXIT, Teacher, TeacherLabel

EXIT_SOURCES = ("label", "noisy", "none")


class ExitSensor:
    """Stand-in for the exit detector: (confidence, forward / 1024, left / 1024) or None.

    This class is the only code of the teacher that holds the environment. `source`:
      "label"  the exit observation of `training.exit_label` (pad centre in view, in line of
               sight and within 640 units)
      "noisy"  the same, degraded by `training.exit_label.noisy` (20% misses, position noise,
               rare false detections): the condition the policy is trained under
      "none"   never reports an exit
    `max_range` drops sightings further away than that many map units (a detector with a shorter
    reach than the labels' 640 units). `noise` overrides the noise levels of the "noisy" source.
    """

    def __init__(self, source: str = "label", max_range: float | None = None, noise: dict | None = None):
        if source not in EXIT_SOURCES:
            raise ValueError(f"exit_source must be one of {EXIT_SOURCES}, not {source!r}")
        self.source = source
        self.max_range = None if max_range is None else float(max_range)
        self.noise = dict(noise or {})        # keyword arguments of exit_label.noisy (p_drop, sigma_rel, sigma_abs, p_false)
        self._env = self._meta = None
        self._rng = np.random.default_rng(0)

    def attach(self, env, meta) -> None:
        self._env, self._meta = env, meta

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng([int(seed), 0x7EAC])

    def read(self):
        if self.source == "none" or self._env is None:
            return None
        from training import exit_label

        seen = exit_label.exit_observation(self._env, self._meta)[:3]
        if seen[0] and self.max_range is not None and seen[1] * seen[1] + seen[2] * seen[2] > self.max_range ** 2:
            seen = (False, 0.0, 0.0)
        if self.source == "noisy":
            seen = exit_label.noisy(seen, self._rng, **self.noise)
        if not seen[0]:
            return None
        return np.array([1.0, seen[1] / spec.RAY_RANGE, seen[2] / spec.RAY_RANGE], np.float32)


class TeacherPolicy:
    """The teacher behind the runner's policy interface (`attach`, `seed`, `reset`, `act`).

    combat_policy  optional policy (`reset()`, `act(obs) -> (9,)`) that takes the controls on
                   steps where an enemy is visible; without one the teacher keeps navigating.
    exit_source    "label", "noisy" or "none" (see ExitSensor).
    budget_steps   the assumed time budget in decision steps handed to the encoder and the
                   teacher; `set_budget()` overrides it for the coming episode (the gate
                   evaluation sets the true cap that way). Default: spec.DEFAULT_BUDGET_STEPS.
    exit_range     drop exit observations from further away than this (see ExitSensor); pair it
                   with the teacher's `view_range`.
    exit_noise     noise levels of the "noisy" exit source, e.g. {"p_false": 0.03} (see ExitSensor).
    slip           stand-in for a student that drives: on each step, with this probability, a
                   run of random actions starts (mean length `slip_steps`) and is executed instead
                   of the teacher's. The random actions pass through `spec.action_masks` as the
                   student's do, and press USE with probability `slip_use` when it is allowed. The
                   teacher keeps labelling throughout and sees the executed action as its
                   previous action, which is exactly its situation in the training loop.
    Other keyword arguments go to `Teacher`.
    """

    def __init__(self, combat_policy=None, exit_source: str = "label", budget_steps: float | None = None,
                 slip: float = 0.0, slip_steps: float = 4.0, slip_use: float = 0.3, exit_range: float | None = None,
                 exit_noise: dict | None = None, **teacher):
        self.combat_policy = combat_policy
        self.sensor = ExitSensor(exit_source, exit_range, exit_noise)
        self.default_budget = float(budget_steps if budget_steps is not None else spec.DEFAULT_BUDGET_STEPS)
        self.slip, self.slip_steps, self.slip_use = float(slip), max(1.0, float(slip_steps)), float(slip_use)
        self._slip_rng = np.random.default_rng(0)
        self.odom, self.mem, self.enc = Odometry(), MemoryMap(), Encoder()
        self.teacher = Teacher(**teacher)
        self.reset()

    # ---- runner hooks ----------------------------------------------------------------------------
    def attach(self, env, meta) -> None:
        """Only to obtain exit observations (see ExitSensor)."""
        self.sensor.attach(env, meta)

    def seed(self, seed: int) -> None:
        self.sensor.seed(seed)
        self._slip_rng = np.random.default_rng([int(seed), 0x511B])

    def set_budget(self, steps: float) -> None:
        """Assumed time budget, in decision steps, for the episode that is starting."""
        self.budget = float(steps)

    def reset(self) -> None:
        self.odom.reset()
        self.mem.reset()
        self.enc.reset()
        self.teacher.reset()
        if self.combat_policy is not None:
            self.combat_policy.reset()
        self.budget = self.default_budget
        self.step = 0
        self.prev_action = None
        self.crop = None
        self.exit_obs = None
        self.label: TeacherLabel | None = None
        self.label_seconds = 0.0
        self.front_seconds = 0.0
        self.exit_step = -1                 # first step at which the teacher had a confirmed exit on its map
        self.leave_step = -1                # first step on which the teacher headed for the exit
        self.leave_travel = float("nan")    # travel estimate (decision steps) made the last time it turned for the exit,
        self.leave_last = -1                # the step of that decision
        self._heading_out = False
        self.leave_route = (0, 0, 0)        # and the cells, closed doors and turns of its route
        self.counts = np.zeros(spec.NUM_SUBGOALS, np.int64)
        self.modes: dict = {}               # steering mode -> steps
        self.slipped = 0                    # steps on which a random action was executed instead of the teacher's
        self.use_masked = 0                 # steps on which the teacher asked for USE while the action mask forbids it
        self.labelled = 0                   # steps whose label carried controller labels (none on fight steps and
                                            # on maps the teacher's grid does not fit)
        self._slip_left = 0

    def act(self, obs) -> np.ndarray:
        step, prev = self.step, self.prev_action
        self.exit_obs = exit_obs = self.sensor.read()
        t0 = time.perf_counter()
        if step > 0:
            self.odom.update(prev, obs)
        pose = (self.odom.x, self.odom.y, self.odom.theta)
        self.mem.update(pose, obs, prev, exit_obs, step)
        self.crop = crop = self.mem.crop(pose)
        scal = self.enc.encode(obs, pose, prev, step, self.budget, exit_obs)["scal"]
        t1 = time.perf_counter()
        self.label = label = self.teacher.label(obs, crop, pose, scal, step, self.budget, prev)
        t2 = time.perf_counter()
        self.front_seconds += t1 - t0
        self.label_seconds += t2 - t1
        self.counts[label.subgoal] += 1
        self.labelled += int(label.head_mask.any())
        mode = self.teacher.info.get("mode", "")
        self.modes[mode] = self.modes.get(mode, 0) + 1
        if self.exit_step < 0 and self.teacher.exit_known:
            self.exit_step = step
        plan = self.teacher.plan
        heading_out = plan is not None and plan.subgoal == GO_EXIT
        if heading_out and not self._heading_out:              # turned for the exit on this step
            if self.leave_step < 0:
                self.leave_step = step
            self.leave_last, self.leave_travel, self.leave_route = step, float(plan.travel), self.teacher.plan_route
        self._heading_out = heading_out
        if label.subgoal == FIGHT and self.combat_policy is not None:
            action = np.asarray(self.combat_policy.act(obs)).astype(np.int8)
        else:
            action = spec.heads_to_buttons(label.heads)
        if label.heads[spec.HEAD_USE] or self.slip > 0.0:
            ents = obs["visible_entities"]
            use_ok, _ = spec.action_masks(obs["rays"]["distance"], obs["rays"]["hit_type"], ents["entity_type"],
                                          ents["mask"], obs["vitals"][2])
            self.use_masked += int(bool(label.heads[spec.HEAD_USE]) and not use_ok)
            if self.slip > 0.0:
                action = self._slipped(action, use_ok)
        self.prev_action = action
        self.step = step + 1
        return action

    def _slipped(self, action: np.ndarray, use_ok: bool) -> np.ndarray:
        """The teacher's action, or a random one while a run of random actions lasts (see `slip`)."""
        rng = self._slip_rng
        if self._slip_left <= 0 and rng.random() < self.slip:
            self._slip_left = int(rng.geometric(1.0 / self.slip_steps))
        if self._slip_left <= 0:
            return action
        self._slip_left -= 1
        self.slipped += 1
        heads = np.array([rng.integers(3), rng.integers(3), rng.integers(3), 0,
                          int(use_ok and rng.random() < self.slip_use), 0], np.int64)
        return spec.heads_to_buttons(heads)

    # ---- reports ---------------------------------------------------------------------------------
    def pose_estimate(self):
        return self.odom.x, self.odom.y, self.odom.theta

    def exit_believed(self) -> bool:
        return self.exit_step >= 0

    def debug_state(self) -> dict:
        """What the last action was based on: the teacher's notes, the label and the timings."""
        t = self.teacher
        out = dict(t.info)
        out.update(step=self.step - 1, budget=self.budget, pose=self.pose_estimate(), exit_step=self.exit_step,
                   leave_step=self.leave_step, leave_travel=self.leave_travel, leave_last=self.leave_last,
                   leave_route=self.leave_route,
                   exit_known=t.exit_known, grid_fits=t.grid_fits, errors=t.errors, last_error=t.last_error, searches=t.searches,
                   replans=t.replans, failed_locks=dict(t.failed),
                   label_ms=1000.0 * self.label_seconds / max(1, self.step),
                   front_ms=1000.0 * self.front_seconds / max(1, self.step),
                   subgoal_steps={n: int(c) for n, c in zip(spec.SUBGOALS, self.counts)}, mode_steps=dict(self.modes),
                   slipped=self.slipped, use_masked=self.use_masked, labelled=self.labelled, unstuck=t.unstuck)
        return out
