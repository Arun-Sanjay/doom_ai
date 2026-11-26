"""Extrinsic reward shaped like the evaluation score.

Training-time scaffolding, never imported by the agent. The reward is computed in the rollout
workers from ``env.privileged_state()`` (ground truth the agent never sees) and the static facts of
the map. It mirrors the scoring categories of docs/EVALUATION.md with the local scorer's weights:

    composite = 0.45 exit + 0.05 efficiency + 0.20 progression + 0.15 exploration + 0.15 combat

Each category is paid out as its events happen, so with the default coefficients the undiscounted
return of an episode equals ``scale`` times the map's composite score (less the two small shaping
costs). Categories a map does not have are dropped and the other weights rescaled, as the scorer
does. The terms, in the order of ``TERMS``:

    exit         w_exit * scale                                 when the exit is reached
    exit_health  w_eff * scale * 0.5 * health / 100             at the exit
    exit_time    w_eff * scale * 0.5 * (1 - tic / cap)          at the exit
    key          w_prog * scale / (n_parts * keys on the map)   per key picked up
    door         w_prog * scale / (n_parts * doors)             first time a door is 56 units open
    locked_door  w_prog * scale / (n_parts * locked doors)      same, for doors that need a key
    sector       w_expl * scale / sectors                       first time a sector is entered
    kill         w_combat * scale / monsters                    per kill
    damage       -w_combat * scale * 0.2 / 100                  per point of damage, first 100 points
    death        -w_combat * scale * 0.3 * (1 - tic / cap)      on death
    ammo         -ammo_cost                                     per shot fired that hit no monster
    reversal     -reversal_cost                                 a door closing again within 6 steps
                                                                of starting to rise (USE pressed twice)
    hit          +hit_bonus (0 by default)                      per shot that hits a monster; optional
                                                                shaping for combat runs, not in the score

``n_parts`` is the number of progression parts present on the map (keys, doors, locked doors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec

TERMS = ("exit", "exit_health", "exit_time", "key", "door", "locked_door", "sector", "kill",
         "damage", "death", "ammo", "reversal", "hit")
T = {name: i for i, name in enumerate(TERMS)}


@dataclass
class RewardCoef:
    """Every coefficient of the extrinsic reward."""

    scale: float = 10.0             # return of a perfect episode
    w_exit: float = 0.45            # category weights of the composite score
    w_efficiency: float = 0.05
    w_progress: float = 0.20
    w_explore: float = 0.15
    w_combat: float = 0.15
    renormalise: bool = True        # drop categories absent from a map and rescale the rest to sum to 1
    damage_frac: float = 0.2        # combat loses this fraction of its weight at 100 points of damage
    death_frac: float = 0.3         # ... and this fraction for dying at tic 0, scaled by time left
    ammo_cost: float = 0.002        # per shot fired that hit no monster
    reversal_cost: float = 0.05     # per door reversal
    hit_bonus: float = 0.0          # optional shaping per shot that hits a monster (off: not in the score)
    fist_cost: float = 0.0          # per decision on the melee slot while bullets are held (off by default); booked under "ammo"
    door_open_height: float = 56.0  # a door counts as opened when its sector opening reaches this
    reversal_window: int = 6        # decision steps after a door starts to rise
    unknown_monsters: int = 8       # totals assumed when the map's things cannot be read
    unknown_keys: int = 1


def fist_charge(vitals, fist_cost: float) -> float:
    """Cost of this decision for standing on the melee slot while a loaded gun is held.

    vitals: the observation's vitals (index 2 selected weapon slot, index 4 bullets). The pistol is
    always owned, so bullets in reserve mean a loaded gun is one weapon press away. Returns
    ``fist_cost`` when slot 1 is selected and bullets are held, and 0 otherwise: in particular 0
    when the agent is out of bullets and the engine itself has put the fist up.
    """
    if vitals is None or fist_cost <= 0.0:
        return 0.0
    on_melee = int(round(float(vitals[2]))) == spec.MELEE_WEAPON_SLOT
    return float(fist_cost) if (on_melee and float(vitals[4]) > 0.0) else 0.0


class SectorIndex:
    """Which sector contains a point, by even-odd crossing counts over the sectors' boundary lines."""

    def __init__(self, polygons):
        """polygons: ``env.sector_polygons()``, a list of (sector id, (K, 4) [x1, y1, x2, y2])."""
        polygons = [(sid, np.asarray(seg, np.float64).reshape(-1, 4)) for sid, seg in polygons]
        polygons = [(sid, seg) for sid, seg in polygons if len(seg)]
        self.n = len(polygons)
        if self.n == 0:
            self.x1 = self.y1 = self.y2 = self.slope = np.zeros(0)
            self.owner = np.zeros(0, np.int64)
            return
        seg = np.concatenate([s for _, s in polygons])
        self.owner = np.concatenate([np.full(len(s), k, np.int64) for k, (_, s) in enumerate(polygons)])
        self.x1, self.y1, self.y2 = seg[:, 0], seg[:, 1], seg[:, 3]
        dy = self.y2 - self.y1
        self.slope = np.divide(seg[:, 2] - seg[:, 0], dy, out=np.zeros_like(dy), where=dy != 0)

    def locate(self, x: float, y: float) -> int:
        """Index (into the polygon list) of the sector containing (x, y), or -1."""
        if self.n == 0:
            return -1
        cross = ((self.y1 > y) != (self.y2 > y)) & (x < self.x1 + (y - self.y1) * self.slope)
        inside = np.flatnonzero(np.bincount(self.owner[cross], minlength=self.n) & 1)
        return int(inside[0]) if len(inside) else -1


class RewardTracker:
    """Per-episode bookkeeping that turns successive privileged states into reward terms."""

    def __init__(self, coef: RewardCoef, n_keys: int, n_monsters: int, has_exit: bool, known: bool = True):
        self.coef = coef
        self.n_keys = n_keys if known else coef.unknown_keys
        self.n_monsters = n_monsters if known else coef.unknown_monsters
        self.has_exit = has_exit if known else True
        self.sectors: SectorIndex | None = None

    # ------------------------------------------------------------------------------------------
    def reset(self, env, priv: dict, ammo: float | None = None) -> None:
        """Start an episode. ``priv`` is ``env.privileged_state()`` right after ``env.reset()``."""
        c = self.coef
        if self.sectors is None:                     # static per map; the environment is reused
            self.sectors = SectorIndex(env.sector_polygons())
        lock = priv.get("door_lock")
        lock = np.zeros(0, np.int64) if lock is None else np.asarray(lock)
        self.door_ids = np.flatnonzero(lock >= 0)
        self.is_locked = lock[self.door_ids] > 0
        n_doors, n_locked = int((~self.is_locked).sum()), int(self.is_locked.sum())

        # Category weights for this map (absent categories dropped, the rest rescaled).
        parts = [n for n in (self.n_keys, n_doors, n_locked) if n > 0]
        weights = {"exit": c.w_exit if self.has_exit else 0.0,
                   "efficiency": c.w_efficiency if self.has_exit else 0.0,
                   "progress": c.w_progress if parts else 0.0,
                   "explore": c.w_explore,
                   "combat": c.w_combat if self.n_monsters > 0 else 0.0}
        total = sum(weights.values())
        k = c.scale / total if (c.renormalise and total > 0) else c.scale
        self.r_exit = k * weights["exit"]
        self.r_eff = k * weights["efficiency"]
        per_part = k * weights["progress"] / max(len(parts), 1)
        self.r_key = per_part / max(self.n_keys, 1)
        self.r_door = per_part / max(n_doors, 1)
        self.r_locked = per_part / max(n_locked, 1)
        self.r_sector = k * weights["explore"] / max(self.sectors.n, 1)
        self.r_combat = k * weights["combat"]
        self.r_kill = self.r_combat / max(self.n_monsters, 1)

        self.step_idx = 0
        self.keys = np.asarray(priv.get("keys", np.zeros(6)), np.float64).copy()
        self.kills = int(priv.get("kills", 0))
        self.damage = float(priv.get("damage_taken", 0.0))
        self.hits = priv.get("hits")
        self.ammo = ammo
        self.opening = self._opening(priv)
        self.opened = self.opening >= c.door_open_height
        self.rising = np.zeros(len(self.door_ids), bool)
        self.rise_start = np.full(len(self.door_ids), -10**9, np.int64)
        self.rise_counted = np.zeros(len(self.door_ids), bool)
        self.visited = np.zeros(self.sectors.n, bool)
        here = self.sectors.locate(float(priv.get("x", 0.0)), float(priv.get("y", 0.0)))
        if here >= 0:
            self.visited[here] = True
        self.done = False
        self.reversals = 0

    def _opening(self, priv: dict) -> np.ndarray:
        if len(self.door_ids) == 0:
            return np.zeros(0)
        return (np.asarray(priv["sector_ceiling"], np.float64)[self.door_ids]
                - np.asarray(priv["sector_floor"], np.float64)[self.door_ids])

    # ------------------------------------------------------------------------------------------
    def step(self, priv: dict, action=None, ammo: float | None = None, vitals=None) -> tuple[float, np.ndarray]:
        """Reward of the transition that led to ``priv``. Returns (total, per-term array)."""
        c = self.coef
        terms = np.zeros(len(TERMS), np.float64)
        self.step_idx += 1
        # Time left as a fraction of the tick cap. Tics are counted here: the engine's own episode
        # time reads 0 once the level has been exited, which is exactly when it is needed.
        left = 1.0 - min(1.0, self.step_idx * spec.FRAME_SKIP / max(priv["episode_ticks"], 1))

        if priv.get("exit_reached") and not self.done:
            terms[T["exit"]] = self.r_exit
            terms[T["exit_health"]] = self.r_eff * 0.5 * min(max(priv["health"], 0.0), 100.0) / 100.0
            terms[T["exit_time"]] = self.r_eff * 0.5 * left

        keys = np.asarray(priv.get("keys", self.keys), np.float64)
        terms[T["key"]] = self.r_key * float(((keys > 0.5) & (self.keys <= 0.5)).sum())
        self.keys = keys.copy()

        if len(self.door_ids):
            opening = self._opening(priv)
            now_open = (opening >= c.door_open_height) & ~self.opened
            terms[T["door"]] = self.r_door * float((now_open & ~self.is_locked).sum())
            terms[T["locked_door"]] = self.r_locked * float((now_open & self.is_locked).sum())
            self.opened |= now_open
            up, down = opening > self.opening + 1e-6, opening < self.opening - 1e-6
            started = up & ~self.rising
            self.rise_start[started] = self.step_idx
            self.rise_counted[started] = False
            reversed_ = down & ~self.rise_counted & (self.step_idx - self.rise_start <= c.reversal_window)
            n_rev = int(reversed_.sum())
            self.rise_counted |= reversed_
            self.rising = np.where(up, True, np.where(down, False, self.rising))
            self.opening = opening
            self.reversals += n_rev
            terms[T["reversal"]] = -c.reversal_cost * n_rev

        here = self.sectors.locate(float(priv.get("x", 0.0)), float(priv.get("y", 0.0)))
        if here >= 0 and not self.visited[here]:
            self.visited[here] = True
            terms[T["sector"]] = self.r_sector

        kills = int(priv.get("kills", self.kills))
        terms[T["kill"]] = self.r_kill * max(kills - self.kills, 0)
        self.kills = kills

        damage = float(priv.get("damage_taken", self.damage))
        terms[T["damage"]] = -self.r_combat * c.damage_frac * (min(1.0, damage / 100.0) - min(1.0, self.damage / 100.0))
        self.damage = damage

        if priv.get("dead") and not self.done:
            terms[T["death"]] = -self.r_combat * c.death_frac * left

        # Ammo: a shot was fired (ammunition went down) and no monster was hit this step.
        hits = priv.get("hits")
        new_hits = 0.0 if (hits is None or self.hits is None) else max(float(hits) - float(self.hits), 0.0)
        fired = ammo is not None and self.ammo is not None and ammo < self.ammo - 1e-6
        if fired and new_hits == 0 and terms[T["kill"]] == 0:
            terms[T["ammo"]] = -c.ammo_cost
        terms[T["hit"]] = c.hit_bonus * new_hits
        terms[T["ammo"]] -= fist_charge(vitals, c.fist_cost)
        self.hits, self.ammo = hits, ammo

        self.done = bool(priv.get("finished") or priv.get("exit_reached") or priv.get("dead"))
        return float(terms.sum()), terms

    def summary(self) -> dict:
        """Raw episode facts for logging."""
        return {"sectors": int(self.visited.sum()), "sectors_total": int(self.sectors.n),
                "doors_opened": int(self.opened.sum()), "doors_total": int(len(self.door_ids)),
                "reversals": int(self.reversals)}
