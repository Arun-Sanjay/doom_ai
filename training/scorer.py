"""Local episode scorer: the metrics the automated score is built from, plus diagnostics.

Training-time scaffolding, never imported by the agent. `docs/EVALUATION.md` names what is
rewarded (exit, progression, exploration, combat) but not the weights, so this module measures
every raw quantity and combines them with weights kept in one place (`WEIGHTS`).

    env.reset(seed=0)
    scorer = EpisodeScorer(env, load_meta(wad))
    while not done:
        action = policy.act(obs)
        scorer.agent_report(pose=policy.pose_estimate())     # optional
        obs, _, terminated, truncated, _ = env.step(action)
        scorer.step()
    row = scorer.result()

The scorer reads the environment only through `env.privileged_state()` and
`env.sector_polygons()`; totals that are not observable there (monsters, keys, exits) come from
`MapMeta`.

Composite score of one episode (100 is a perfect run; the combat penalties can take it below 0):

    100 * (0.45*exit + 0.05*eff + 0.20*prog + 0.15*expl + 0.15*combat)
    eff    = exit * (0.5 * health/100 + 0.5 * (1 - tic/cap))
    prog   = mean of the available fractions: keys picked, doors opened, locked doors opened
    expl   = sectors visited / sectors on the map
    combat = kills/monsters - 0.2*min(1, damage/100) - 0.3*dead*(1 - tic/cap), never below -0.5

A category the map does not have (no exit, nothing to progress through, no monsters) is dropped
and the remaining weights are rescaled to sum to one.
"""

from __future__ import annotations

import math

import numpy as np

from training.mapmeta import CELL, MapMeta, SectorLocator

FRAME_SKIP = 4               # game tics per decision step
DOOR_OPEN_UNITS = 56.0       # a door counts as opened once its opening reaches the player's height
STUCK_WINDOW_STEPS = 10      # stuck = moved less than STUCK_UNITS over this many steps
STUCK_UNITS = 8.0
AT_DOOR_UNITS = 96.0         # "at a closed door": this close to the outline of a door that is not open
EXIT_VISIBLE_RANGE = 512.0   # beyond this an exit pad is under a pixel tall in the 160x120 frame
HALF_FOV_COS = math.cos(math.radians(45.0))

# Category weights. "default" is the working estimate of the hidden weighting; the other two
# bracket it and are reported next to it so a ranking that depends on the guess is visible.
WEIGHTS = {
    "default":    {"exit": 0.45, "eff": 0.05, "prog": 0.20, "expl": 0.15, "combat": 0.15},
    "exit_heavy": {"exit": 0.60, "eff": 0.05, "prog": 0.15, "expl": 0.10, "combat": 0.10},
    "balanced":   {"exit": 0.30, "eff": 0.05, "prog": 0.25, "expl": 0.20, "combat": 0.20},
}
COMPONENTS = ("exit", "eff", "prog", "expl", "combat")
TIER_WEIGHTS = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}   # harder tiers carry more weight
NAN = float("nan")

# Every key of EpisodeScorer.result(), in order.
RESULT_FIELDS = (
    "episode_ticks", "steps", "tic",
    "has_exit", "exit", "dead", "timeout", "death_tic", "health", "damage_taken",
    "keys_picked", "keys_total", "doors_opened", "doors_total", "locked_opened", "locked_total",
    "sectors_visited", "sectors_total", "cells_visited", "cells_total", "kills", "monsters_total",
    "stuck_rate", "stuck_at_door_rate", "door_reversals",
    "exit_believed_tic", "exit_seen_timeout", "exit_visible_tic", "exit_visible_timeout",
    "exit_dist_min", "exit_dist_final",
    "odom_err_final", "odom_err_mean", "odom_err_max", "odom_heading_err_max",
    "c_exit", "c_eff", "c_prog", "c_expl", "c_combat",
    "score", "score_exit_heavy", "score_balanced",
)


# ------------------------------------------------------------------------------- composite
def components(r: dict) -> dict[str, float]:
    """The five score components of one episode. NaN marks a category the map does not have."""
    cap = max(1.0, float(r["episode_ticks"]))
    time_left = 1.0 - min(1.0, float(r["tic"]) / cap)
    out = dict.fromkeys(COMPONENTS, NAN)
    if r["has_exit"]:
        health = min(1.0, max(0.0, float(r["health"])) / 100.0)
        out["exit"] = float(r["exit"])
        out["eff"] = float(r["exit"]) * (0.5 * health + 0.5 * time_left)
    progress = ((r["keys_picked"], r["keys_total"]), (r["doors_opened"], r["doors_total"]),
                (r["locked_opened"], r["locked_total"]))
    parts = [got / total for got, total in progress if total > 0]
    if parts:
        out["prog"] = float(sum(parts) / len(parts))
    if r["sectors_total"] > 0:
        out["expl"] = r["sectors_visited"] / r["sectors_total"]
    if r["monsters_total"] > 0:
        combat = (min(1.0, r["kills"] / r["monsters_total"]) - 0.2 * min(1.0, float(r["damage_taken"]) / 100.0)
                  - 0.3 * float(r["dead"]) * time_left)
        out["combat"] = max(-0.5, combat)
    return out


def composite(comp: dict[str, float], weights: dict[str, float] | str = "default") -> float:
    """Weighted sum of the components present, rescaled so the weights in use sum to one, times 100."""
    w = WEIGHTS[weights] if isinstance(weights, str) else weights
    total = used = 0.0
    for name, weight in w.items():
        value = comp.get(name, NAN)
        if value == value:  # not NaN
            total += weight * value
            used += weight
    return 100.0 * total / used if used > 0 else 0.0


# ----------------------------------------------------------------------------- line of sight
class _Sight:
    """Line of sight over the sector outlines: a wall always blocks, a two-sided line blocks
    while the opening between its sectors is closed (a shut door)."""

    def __init__(self, locator: SectorLocator):
        seg = locator.segments
        owner = locator.ids[locator.owner] if len(seg) else np.zeros(0, np.int64)
        # Put both copies of a two-sided line in the same endpoint order, then pair them up.
        swap = (seg[:, 0] > seg[:, 2]) | ((seg[:, 0] == seg[:, 2]) & (seg[:, 1] > seg[:, 3]))
        canon = np.where(swap[:, None], seg[:, [2, 3, 0, 1]], seg)
        uniq, inverse, counts = np.unique(canon, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(inverse, kind="stable")
        first = np.searchsorted(inverse[order], np.arange(len(uniq)))
        self.p1, self.d = uniq[:, 0:2], uniq[:, 2:4] - uniq[:, 0:2]
        self.one_sided = counts < 2
        self.sec_a = owner[order[first]] if len(uniq) else owner
        self.sec_b = owner[order[np.minimum(first + 1, max(len(order) - 1, 0))]] if len(uniq) else owner

    def clear(self, floor: np.ndarray, ceil: np.ndarray, ox: float, oy: float, tx: float, ty: float) -> bool:
        """True if nothing blocks the straight line from (ox, oy) to (tx, ty)."""
        if not len(self.p1):
            return True
        a, b = self.sec_a, self.sec_b
        opening = np.minimum(ceil[a], ceil[b]) - np.maximum(floor[a], floor[b])
        blocking = self.one_sided | (opening <= 0.0)
        rx, ry = tx - ox, ty - oy
        wx, wy = self.p1[:, 0] - ox, self.p1[:, 1] - oy
        dx, dy = self.d[:, 0], self.d[:, 1]
        denom = rx * dy - ry * dx
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (wx * dy - wy * dx) / denom   # position along the sight line, 0..1
            u = (wx * ry - wy * rx) / denom   # position along the wall, 0..1
        hit = blocking & (np.abs(denom) > 1e-9) & (t > 1e-6) & (t < 1.0 - 1e-6) & (u >= 0.0) & (u <= 1.0)
        return not bool(hit.any())


# ---------------------------------------------------------------------------- episode scorer
class EpisodeScorer:
    """Follows one episode and reports its metrics.

    Create it after `env.reset()`, call `step()` after every `env.step()`, then `result()`.
    `agent_report()` is optional and carries what only the agent knows (its pose estimate and
    whether it believes it has seen an exit).

    Definitions that need a choice:
      * tic: game tics elapsed, `privileged_state()["tic"] - 1`. The engine reports 0 on the step
        that reaches the exit, so that step is counted as a full 4 tics.
      * sectors visited: sectors whose outline contained the player after some step, plus the
        exit region when the exit is reached (the exit line is its border, and the engine stops
        reporting positions on that step).
      * cells visited: distinct 64-unit squares the player's position fell in (plus the square of
        the exit region, likewise).
      * a door is opened once its opening reaches 56 units, whoever opened it.
      * stuck step: the player is less than 8 units from where it was 10 steps earlier. Such a
        step counts towards `stuck_rate`, unless the player is within 96 units of a door that is
        not open, or was at one of its last 10 stuck steps (the wait is still inside the window
        just after the door opens): then it counts towards `stuck_at_door_rate`, because waiting
        for a door is not being stuck.
      * door reversal: a door's opening shrinks on the step right after it grew, i.e. the door
        turned round while rising because USE was pressed again. Left alone, a door pauses at the
        top before closing, so this never fires on its own. It covers the first 6 steps of the
        rise (while rays still report a door, the only time the USE mask allows a press) and
        the rest of the rise as well.
      * exit visible: an exit centre within 512 units, inside the 90-degree field of view and in
        line of sight. `exit_visible_timeout` marks episodes that saw the exit this way and still
        ran out of time; `exit_seen_timeout` is the same from the agent's own report.
    """

    def __init__(self, env, meta: MapMeta, monsters_total: int | None = None):
        self.env, self.meta = env, meta
        st = env.privileged_state()
        self.cap = int(st["episode_ticks"])
        self.locator = SectorLocator(env.sector_polygons())
        self.monsters_total = meta.monsters_total if monsters_total is None else int(monsters_total)

        lock = st["door_lock"]
        self._door_ids = np.flatnonzero(lock >= 0)
        self._door_locked = lock[self._door_ids] > 0
        self._opening = self._door_opening(st)
        self._door_max = self._opening.copy()
        self._rising = np.zeros(len(self._door_ids), bool)
        self.door_reversals = 0

        self.steps = 0
        self.tic = 0
        self._raw_tic = int(st["tic"])
        self._start = (float(st["x"]), float(st["y"]), math.radians(float(st["angle"])))
        self._xs, self._ys = [float(st["x"])], [float(st["y"])]
        self._sector = self.locator.locate(st["x"], st["y"])
        self._sectors = {self._sector} - {-1}
        self._cells = {(int(st["x"] // CELL), int(st["y"] // CELL))}
        self.stuck_steps = 0
        self.stuck_at_door_steps = 0
        self._door_wait_step = -STUCK_WINDOW_STEPS   # last step spent standing at a closed door
        self.death_tic = -1

        self._exits = [(float(x), float(y)) for x, y in meta.exit_centres]
        self._sight = _Sight(self.locator) if self._exits else None
        self.exit_dist_min = self.exit_dist_final = NAN
        self.exit_visible_tic = -1
        self.exit_believed_tic = -1
        self._exit_hook = False
        self._odom_n = 0
        self._odom_sum = self._odom_max = self._odom_last = self._odom_heading_max = 0.0

        self._state = st
        self._watch_exit(st)

    # ---- per step
    def step(self) -> None:
        """Call once after every `env.step()`."""
        st = self.env.privileged_state()
        self._state = st
        self.steps += 1
        raw = int(st["tic"])
        self.tic = min(self.cap, raw - 1 if raw > self._raw_tic else self.tic + FRAME_SKIP)
        self._raw_tic = max(raw, self._raw_tic)
        if st["dead"] and self.death_tic < 0:
            self.death_tic = self.tic

        x, y = float(st["x"]), float(st["y"])
        sector = self.locator.locate(x, y, self._sector)
        if sector >= 0:
            self._sector = sector
            self._sectors.add(sector)
        self._cells.add((int(x // CELL), int(y // CELL)))

        if len(self._door_ids):
            opening = self._door_opening(st)
            delta = opening - self._opening
            self.door_reversals += int(np.count_nonzero((delta < 0.0) & self._rising))
            self._rising = delta > 0.0
            np.maximum(self._door_max, opening, out=self._door_max)
            self._opening = opening

        self._xs.append(x)
        self._ys.append(y)
        if self.steps >= STUCK_WINDOW_STEPS:
            k = self.steps - STUCK_WINDOW_STEPS
            if math.hypot(x - self._xs[k], y - self._ys[k]) < STUCK_UNITS:
                if self._at_closed_door(x, y):
                    self._door_wait_step = self.steps
                if self.steps - self._door_wait_step < STUCK_WINDOW_STEPS:
                    self.stuck_at_door_steps += 1
                else:
                    self.stuck_steps += 1
        self._watch_exit(st)

    def agent_report(self, pose=None, exit_believed=None) -> None:
        """What the agent says about the observation it has just acted on.

        Call it after `policy.act(obs)` and before `env.step()`, so it is compared with the state
        the agent saw. `pose` is the agent's `(x, y, theta)` in its start frame (x forward, y
        left, theta counter-clockwise in radians); `exit_believed` is whether it currently
        believes it knows where an exit is. Either may be None. A pose that is not three finite
        numbers raises ValueError and leaves the scorer as it was.
        """
        st = self._state
        if pose is not None:
            px, py, ptheta = float(pose[0]), float(pose[1]), float(pose[2])
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(ptheta)):
                raise ValueError(f"pose must be three finite numbers (x, y, theta), got ({px}, {py}, {ptheta})")
            x0, y0, a0 = self._start
            dx, dy = float(st["x"]) - x0, float(st["y"]) - y0
            c, s = math.cos(a0), math.sin(a0)
            err = math.hypot(px - (dx * c + dy * s), py - (-dx * s + dy * c))
            turn = ptheta - (math.radians(float(st["angle"])) - a0)
            heading = abs(math.degrees((turn + math.pi) % (2.0 * math.pi) - math.pi))
            self._odom_n += 1
            self._odom_sum += err
            self._odom_last = err
            self._odom_max = max(self._odom_max, err)
            self._odom_heading_max = max(self._odom_heading_max, heading)
        if exit_believed is not None:
            believed = bool(exit_believed)   # before anything is changed: an array of several values raises here
            self._exit_hook = True
            if believed and self.exit_believed_tic < 0:
                self.exit_believed_tic = self.tic

    # ---- helpers
    def _door_opening(self, st: dict) -> np.ndarray:
        return (st["sector_ceiling"][self._door_ids] - st["sector_floor"][self._door_ids]).astype(np.float64)

    def _at_closed_door(self, x: float, y: float) -> bool:
        for k in np.flatnonzero(self._opening < DOOR_OPEN_UNITS):
            if self.locator.distance(int(self._door_ids[k]), x, y) <= AT_DOOR_UNITS:
                return True
        return False

    def _watch_exit(self, st: dict) -> None:
        """Track the distance to the nearest exit and the first tic an exit is in view."""
        if not self._exits:
            return
        x, y = float(st["x"]), float(st["y"])
        heading = math.radians(float(st["angle"]))
        fx, fy = math.cos(heading), math.sin(heading)
        nearest = math.inf
        for ex, ey in self._exits:
            dist = math.hypot(ex - x, ey - y)
            nearest = min(nearest, dist)
            if (self.exit_visible_tic < 0 and dist <= EXIT_VISIBLE_RANGE
                    and (ex - x) * fx + (ey - y) * fy >= HALF_FOV_COS * dist
                    and self._sight.clear(st["sector_floor"], st["sector_ceiling"], x, y, ex, ey)):
                self.exit_visible_tic = self.tic
        self.exit_dist_final = nearest
        self.exit_dist_min = nearest if self.exit_dist_min != self.exit_dist_min else min(self.exit_dist_min, nearest)

    # ---- result
    def result(self) -> dict:
        """Metrics of the episode so far (normally called once it has ended); keys are RESULT_FIELDS."""
        st = self._state
        exit_reached = bool(st["exit_reached"])
        dead = bool(st["dead"])
        timeout = bool(st["timeout"]) and not dead and not exit_reached
        has_exit = bool(self._exits) or exit_reached
        sectors, cells = set(self._sectors), set(self._cells)
        if exit_reached and self.meta.exit_sectors:   # the level ended on the border of an exit region: count it
            x, y = float(st["x"]), float(st["y"])
            regions = [e for e in self.meta.exits if e.sector >= 0]
            pad = min(regions, key=lambda e: math.hypot(e.centre[0] - x, e.centre[1] - y))
            sectors.add(pad.sector)
            cells.add((int(pad.centre[0] // CELL), int(pad.centre[1] // CELL)))
        opened = self._door_max >= DOOR_OPEN_UNITS
        steps = max(1, self.steps)
        odom = self._odom_n > 0
        r = {
            "episode_ticks": self.cap, "steps": self.steps, "tic": int(self.tic),
            "has_exit": int(has_exit), "exit": int(exit_reached), "dead": int(dead), "timeout": int(timeout),
            "death_tic": int(self.death_tic), "health": max(0.0, float(st["health"])),
            "damage_taken": float(st["damage_taken"]),
            "keys_picked": int(np.count_nonzero(st["keys"])), "keys_total": self.meta.keys_total,
            "doors_opened": int(np.count_nonzero(opened & ~self._door_locked)),
            "doors_total": int(np.count_nonzero(~self._door_locked)),
            "locked_opened": int(np.count_nonzero(opened & self._door_locked)),
            "locked_total": int(np.count_nonzero(self._door_locked)),
            "sectors_visited": len(sectors), "sectors_total": len(self.locator),
            "cells_visited": len(cells),
            "cells_total": self.meta.grid.open_cells if self.meta.grid is not None else NAN,
            "kills": int(st["kills"]), "monsters_total": max(self.monsters_total, int(st["kills"])),
            "stuck_rate": self.stuck_steps / steps, "stuck_at_door_rate": self.stuck_at_door_steps / steps,
            "door_reversals": int(self.door_reversals),
            "exit_believed_tic": int(self.exit_believed_tic) if self._exit_hook else NAN,
            "exit_seen_timeout": float(self.exit_believed_tic >= 0 and timeout) if self._exit_hook else NAN,
            "exit_visible_tic": int(self.exit_visible_tic) if self._exits else NAN,
            "exit_visible_timeout": float(self.exit_visible_tic >= 0 and timeout) if self._exits else NAN,
            "exit_dist_min": self.exit_dist_min, "exit_dist_final": self.exit_dist_final,
            "odom_err_final": self._odom_last if odom else NAN,
            "odom_err_mean": self._odom_sum / self._odom_n if odom else NAN,
            "odom_err_max": self._odom_max if odom else NAN,
            "odom_heading_err_max": self._odom_heading_max if odom else NAN,
        }
        if r["keys_picked"] > r["keys_total"]:   # a key the WAD does not list (spawned later)
            r["keys_total"] = r["keys_picked"]
        comp = components(r)
        r.update({f"c_{name}": comp[name] for name in COMPONENTS})
        r["score"] = composite(comp, "default")
        r["score_exit_heavy"] = composite(comp, "exit_heavy")
        r["score_balanced"] = composite(comp, "balanced")
        return r


def empty_result(episode_ticks: int = 0, has_exit: bool = True) -> dict:
    """A row for an episode that could not be played: score 0, no exit, every measurement NaN."""
    r = dict.fromkeys(RESULT_FIELDS, NAN)
    r.update(episode_ticks=int(episode_ticks), steps=0, tic=0, has_exit=int(has_exit), exit=0, dead=0, timeout=0,
             score=0.0, score_exit_heavy=0.0, score_balanced=0.0)
    return r


# ------------------------------------------------------------------------------- aggregation
# Columns of summarise(): what evaluate.py writes per tier.
SUMMARY_FIELDS = (
    "episodes", "maps", "errors", "agent_errors", "hook_errors",
    "score", "score_se", "score_exit_heavy", "score_balanced",
    "exit_rate", "exit_rate_se", "death_rate", "timeout_rate", "tic_frac", "health", "damage_taken",
    "keys_frac", "doors_frac", "locked_frac", "sectors_frac", "kills_frac", "cells_visited", "cells_frac",
    "stuck_rate", "stuck_at_door_rate", "door_reversals",
    "exit_seen_timeout", "exit_visible_timeout", "exit_dist_min",
    "odom_err_final", "odom_err_max", "odom_heading_err_max",
    "c_exit", "c_eff", "c_prog", "c_expl", "c_combat",
    "steps", "act_ms_mean", "act_ms_max", "wall_s",
)


def _column(rows, key: str) -> np.ndarray:
    """One field of every row as floats; missing or empty values become NaN."""
    return np.array([float(v) if (v := r.get(key)) not in (None, "") else NAN for r in rows], np.float64)


def _mean(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else NAN


def _peak(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.max()) if len(values) else NAN


def _sem(values: np.ndarray) -> float:
    """Standard error of the mean (0 for fewer than two values)."""
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def _pooled(rows, part: str, total: str) -> float:
    """Sum of `part` over sum of `total`, over the rows where both are known."""
    a, b = _column(rows, part), _column(rows, total)
    known = ~np.isnan(a) & ~np.isnan(b)
    return float(a[known].sum() / b[known].sum()) if b[known].sum() > 0 else NAN


def summarise(rows) -> dict:
    """Aggregate episode rows into one flat dict with the keys of SUMMARY_FIELDS.

    Scores and the exit rate are over all episodes, and an episode that could not be played
    (`error` set) counts as score 0 and no exit. The exit rate only counts maps that have an
    exit. `errors`, `agent_errors` and `hook_errors` count the episodes with `error`,
    `agent_error` (the agent failed or ran out of time and stood still from then on) and
    `hook_error` (a telemetry hook failed; the score is not affected) set.
    Every other column is over the episodes that were played: plain means, except the
    `*_frac` columns, which pool counts (all keys picked over all keys present, and so on), and
    `odom_heading_err_max` and `act_ms_max`, which are maxima. A column with nothing to average
    (no map with a key, no agent report, ...) is NaN.
    """
    rows = list(rows)
    played = [r for r in rows if not r.get("error")]
    score = _column(rows, "score")
    exits = _column([r for r in rows if r.get("has_exit")], "exit")
    exit_rate = _mean(exits)
    out = {
        "episodes": len(rows), "maps": len({r.get("map_id", i) for i, r in enumerate(rows)}),
        "errors": len(rows) - len(played), "agent_errors": sum(1 for r in rows if r.get("agent_error")),
        "hook_errors": sum(1 for r in rows if r.get("hook_error")),
        "score": _mean(score), "score_se": _sem(score) if len(rows) else NAN,
        "score_exit_heavy": _mean(_column(rows, "score_exit_heavy")),
        "score_balanced": _mean(_column(rows, "score_balanced")),
        "exit_rate": exit_rate,
        "exit_rate_se": math.sqrt(exit_rate * (1.0 - exit_rate) / len(exits)) if len(exits) else NAN,
        "death_rate": _mean(_column(played, "dead")), "timeout_rate": _mean(_column(played, "timeout")),
        "tic_frac": _mean(_column(played, "tic") / np.maximum(_column(played, "episode_ticks"), 1.0)),
        "keys_frac": _pooled(played, "keys_picked", "keys_total"),
        "doors_frac": _pooled(played, "doors_opened", "doors_total"),
        "locked_frac": _pooled(played, "locked_opened", "locked_total"),
        "sectors_frac": _pooled(played, "sectors_visited", "sectors_total"),
        "kills_frac": _pooled(played, "kills", "monsters_total"),
        "cells_frac": _pooled(played, "cells_visited", "cells_total"),
        "odom_heading_err_max": _peak(_column(played, "odom_heading_err_max")),
        "act_ms_max": _peak(_column(played, "act_ms_max")),
        "wall_s": float(np.nansum(_column(rows, "wall_s"))),
    }
    for key in SUMMARY_FIELDS:   # everything else is the mean of the per-episode field of the same name
        if key not in out:
            out[key] = _mean(_column(played, key))
    return {key: out[key] for key in SUMMARY_FIELDS}


def suite_score(results, key: str = "score") -> dict:
    """Leaderboard-style score of a set of episode rows (each with a `tier`).

    Returns {"score", "score_se", "episodes", "errors", "tiers": {tier: summarise(rows of tier)}}.
    `score` is the tier-weighted mean over episodes, sum(w * score) / sum(w) with w = 1, 2, 3, 4
    for tiers 1-4: the "weighted sum over all maps" of docs/EVALUATION.md divided by the total
    weight. Tiers outside 1-4 (the tier 0 curriculum maps, untiered maps) are summarised but
    carry no weight; if no episode has a weighted tier the plain mean is returned instead.
    `score_se` is the standard error from the spread inside each tier (a tier with a single
    episode adds none). Each tier's `exit_rate` and `exit_rate_se` cover the episodes of that
    tier whose map has an exit.
    """
    rows = list(results)
    by_tier: dict[int, list] = {}
    for r in rows:
        by_tier.setdefault(int(r.get("tier", 0) or 0), []).append(r)
    tiers = {t: summarise(by_tier[t]) for t in sorted(by_tier)}
    weighted = [t for t in by_tier if TIER_WEIGHTS.get(t, 0.0) > 0.0]
    weight_of = (lambda t: TIER_WEIGHTS[t]) if weighted else (lambda t: 1.0)
    total = num = var = 0.0
    for t in (weighted or list(by_tier)):
        values = _column(by_tier[t], key)
        w = weight_of(t)
        total += w * len(values)
        num += w * float(values.sum())
        var += w * w * len(values) * (float(values.var(ddof=1)) if len(values) > 1 else 0.0)
    return {
        "score": num / total if total else NAN, "score_se": math.sqrt(var) / total if total else NAN,
        "episodes": len(rows), "errors": sum(1 for r in rows if r.get("error")), "tiers": tiers,
    }
