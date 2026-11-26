"""Curriculum over map tiers.

Training-time scaffolding, never imported by the agent.

Tiers: 0 = monster-free navigation, 1 = combat arenas, 2 = multi-room exploration with doors,
3 = key -> locked door -> exit, 4 = gauntlet. Training moves through fixed stages and then
samples adaptively:

    nav         tier 0 only
    arena       mostly tier 1, some tier 0
    keys_doors  tiers 2 and 3 (doors, keys, locked doors), with some of the earlier tiers
    gauntlet    mostly tier 4, with the earlier tiers
    adaptive    tiers 1-4 weighted towards the tier with the lowest recent exit rate

A stage ends when the recent success rate on its focus tiers reaches its threshold, or after its
budget of environment steps, whichever comes first (the budget keeps a run moving when a stage is
harder than expected). "Success" is reaching the exit; on a map without an exit it is killing
every monster and surviving. The current stage name is written to every row of the training log,
and ``tier_probs()`` is what the rollout workers sample maps from.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

NUM_TIERS = 5


@dataclass(frozen=True)
class Stage:
    """One stage of the fixed part of the curriculum."""

    name: str
    weights: tuple            # sampling weight per tier 0..4
    focus: tuple              # tiers whose success rate decides when the stage ends
    threshold: float          # recent success rate on the focus tiers that ends the stage
    max_steps: int            # environment steps after which the stage ends regardless


DEFAULT_STAGES = (
    Stage("nav", (1.0, 0.0, 0.0, 0.0, 0.0), (0,), 0.70, 3_000_000),
    Stage("arena", (0.2, 0.8, 0.0, 0.0, 0.0), (1,), 0.60, 6_000_000),
    Stage("keys_doors", (0.1, 0.15, 0.3, 0.45, 0.0), (2, 3), 0.50, 15_000_000),
    Stage("gauntlet", (0.05, 0.15, 0.15, 0.2, 0.45), (4,), 0.40, 15_000_000),
)
ADAPTIVE = "adaptive"


@dataclass
class CurriculumConfig:
    """Stages and the statistics that move training through them."""

    stages: tuple = DEFAULT_STAGES
    window: int = 200             # episodes per tier in the "recent" statistics
    min_episodes: int = 100       # episodes on the focus tiers before a threshold can end a stage
    adaptive_tiers: tuple = (1, 2, 3, 4)
    adaptive_floor: float = 0.1   # added to (1 - exit rate) so no tier's weight reaches zero
    start_stage: int = 0          # skip earlier stages (len(stages) = adaptive from the start)


class Curriculum:
    """Tracks recent results per tier and decides which tiers the workers sample."""

    def __init__(self, cfg: CurriculumConfig | None = None, available_tiers=None):
        self.cfg = cfg or CurriculumConfig()
        # Stages read back from a JSON config arrive as dicts.
        self.cfg.stages = tuple(s if isinstance(s, Stage) else Stage(s["name"], tuple(s["weights"]), tuple(s["focus"]),
                                                                    float(s["threshold"]), int(s["max_steps"]))
                                for s in self.cfg.stages)
        self.available = None if available_tiers is None else set(int(t) for t in available_tiers)
        self.stage_index = min(self.cfg.start_stage, len(self.cfg.stages))
        self.stage_start = 0
        self.success = [deque(maxlen=self.cfg.window) for _ in range(NUM_TIERS)]
        self.exits = [deque(maxlen=self.cfg.window) for _ in range(NUM_TIERS)]
        self._skip_unavailable(0)

    # ------------------------------------------------------------------------------------------
    @property
    def stage(self) -> str:
        stages = self.cfg.stages
        return stages[self.stage_index].name if self.stage_index < len(stages) else ADAPTIVE

    def record(self, info: dict) -> None:
        """Take the summary of one finished episode (see ``vecenv.EnvSlot.step``)."""
        tier = int(info.get("tier", 0))
        if not 0 <= tier < NUM_TIERS:
            return
        if info.get("has_exit", True):
            ok = bool(info.get("exit", False))
            self.exits[tier].append(ok)
        else:
            ok = (not info.get("dead", False)) and info.get("kills", 0) >= info.get("monsters", 1) > 0
        self.success[tier].append(bool(ok))

    def exit_rate(self, tier: int) -> float | None:
        """Recent exit rate on maps of ``tier`` that have an exit; None before any such episode."""
        d = self.exits[tier]
        return float(np.mean(d)) if d else None

    def success_rate(self, tiers) -> tuple[float, int]:
        """(recent success rate, episodes counted) pooled over ``tiers``."""
        results = [x for t in tiers for x in self.success[t]]
        return (float(np.mean(results)) if results else 0.0), len(results)

    def advance(self, env_steps: int) -> bool:
        """Move to the next stage when the current one is complete. Returns True if the stage changed."""
        stages = self.cfg.stages
        if self.stage_index >= len(stages):
            return False
        stage = stages[self.stage_index]
        rate, n = self.success_rate(stage.focus)
        passed = n >= self.cfg.min_episodes and rate >= stage.threshold
        if not passed and env_steps - self.stage_start < stage.max_steps:
            return False
        self.stage_index += 1
        self.stage_start = env_steps
        self._skip_unavailable(env_steps)
        return True

    def _skip_unavailable(self, env_steps: int) -> None:
        """Skip stages none of whose focus tiers has maps in the manifest."""
        stages = self.cfg.stages
        while (self.available is not None and self.stage_index < len(stages)
               and not self.available.intersection(stages[self.stage_index].focus)):
            self.stage_index += 1
            self.stage_start = env_steps

    def tier_probs(self) -> np.ndarray:
        """(5,) sampling weights per tier for the current stage, restricted to tiers that have maps."""
        stages = self.cfg.stages
        if self.stage_index < len(stages):
            w = np.array(stages[self.stage_index].weights, np.float64)
        else:
            w = np.zeros(NUM_TIERS)
            for t in self.cfg.adaptive_tiers:
                rate = self.exit_rate(t)
                w[t] = (1.0 - (0.0 if rate is None else rate)) + self.cfg.adaptive_floor
        if self.available is not None:
            w = np.array([w[t] if t in self.available else 0.0 for t in range(NUM_TIERS)])
            if w.sum() <= 0.0:                         # nothing the stage wants is available: use what is
                w = np.array([1.0 if t in self.available else 0.0 for t in range(NUM_TIERS)])
        return w / w.sum() if w.sum() > 0 else np.full(NUM_TIERS, 1.0 / NUM_TIERS)

    # ------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"stage_index": self.stage_index, "stage_start": self.stage_start,
                "success": [list(d) for d in self.success], "exits": [list(d) for d in self.exits]}

    def load_state_dict(self, state: dict) -> None:
        self.stage_index, self.stage_start = int(state["stage_index"]), int(state["stage_start"])
        for t in range(NUM_TIERS):
            self.success[t].clear()
            self.success[t].extend(state["success"][t])
            self.exits[t].clear()
            self.exits[t].extend(state["exits"][t])
