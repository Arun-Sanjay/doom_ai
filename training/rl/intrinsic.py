"""Curiosity reward computed from the agent's own memory only.

Training-time scaffolding, never imported by the agent. The inputs are the memory-map crop the
network sees and the agent's pose estimate; no privileged state is read, so the same quantity can
be shown next to the agent at demonstration time.

Two terms, both episodic (cleared by ``reset``):

* **new cells**: ``new_cell`` for every 64-unit cell that is observed for the first time, where
  "observed" means the crop's ``CH_FREE`` or ``CH_WALL`` channel is on. The crop is centred on the
  agent and moves with it, so observed cells are accumulated in start-frame cell coordinates
  (the crop window is placed using the agent's cell from the pose estimate).
* **visit count**: ``visit / sqrt(n)`` when the agent enters a cell for the n-th time. The bonus is
  paid on entry, so standing still earns nothing and pacing between two cells decays.

The first observation of an episode seeds both sets and earns nothing (no action produced it).
Seeing nothing new while staying in the same cell gives exactly zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec

CELL = spec.MAP_RES * spec.CROP_MULT            # 64 map units per crop cell
SIDE = spec.MAP_SIZE // spec.CROP_MULT          # crop cells per side of the whole stored map (128)
HALF_CROP = spec.CROP_SIZE // 2                 # the agent's cell sits at this index of the crop
# Crop layout assumed here: crop[channel, i, j] is the cell at (agent cell x - 32 + i, agent cell y - 32 + j).


@dataclass
class IntrinsicCoef:
    """Coefficients of the intrinsic reward."""

    new_cell: float = 0.01      # per cell observed for the first time this episode
    visit: float = 0.02         # times 1/sqrt(n) on the n-th entry into a cell
    enabled: bool = True        # False gives zero reward (the curiosity ablation)


def agent_cell(pose) -> tuple[int, int]:
    """Start-frame crop cell (64 units) containing the pose: cell k spans [64k - 32, 64k + 32)."""
    return (int(math.floor((float(pose[0]) + CELL / 2) / CELL)), int(math.floor((float(pose[1]) + CELL / 2) / CELL)))


class IntrinsicReward:
    """Per-episode novelty bookkeeping for one environment."""

    def __init__(self, coef: IntrinsicCoef | None = None):
        self.coef = coef or IntrinsicCoef()
        self.seen = np.zeros((SIDE, SIDE), bool)
        self.visits = np.zeros((SIDE, SIDE), np.int32)
        self.reset()

    def reset(self) -> None:
        self.seen[:] = False
        self.visits[:] = 0
        self.cell = None
        self.n_seen = 0

    def step(self, crop: np.ndarray, pose) -> tuple[float, float, float]:
        """Return (reward, new-cell part, visit part) for the observation that produced ``crop``."""
        first = self.cell is None
        kx, ky = agent_cell(pose)

        # Newly observed cells: place the crop window in start-frame cell coordinates, clip to the array.
        observed = (crop[spec.CH_FREE] > 0) | (crop[spec.CH_WALL] > 0)
        i0, j0 = kx - HALF_CROP + SIDE // 2, ky - HALF_CROP + SIDE // 2
        a0, a1 = max(i0, 0), min(i0 + spec.CROP_SIZE, SIDE)
        b0, b1 = max(j0, 0), min(j0 + spec.CROP_SIZE, SIDE)
        n_new = 0
        if a0 < a1 and b0 < b1:
            window = self.seen[a0:a1, b0:b1]
            fresh = observed[a0 - i0:a1 - i0, b0 - j0:b1 - j0] & ~window
            n_new = int(np.count_nonzero(fresh))
            if n_new:
                window |= fresh
                self.n_seen += n_new

        # Visit count, paid when the agent's cell changes.
        bonus = 0.0
        if (kx, ky) != self.cell:
            self.cell = (kx, ky)
            ci, cj = kx + SIDE // 2, ky + SIDE // 2
            if 0 <= ci < SIDE and 0 <= cj < SIDE:
                self.visits[ci, cj] += 1
                bonus = 1.0 / math.sqrt(self.visits[ci, cj])

        if first or not self.coef.enabled:
            return 0.0, 0.0, 0.0
        r_new, r_visit = self.coef.new_cell * n_new, self.coef.visit * bonus
        return r_new + r_visit, r_new, r_visit
