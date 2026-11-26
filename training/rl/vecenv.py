"""Vector environment: worker processes own the environments and return batched network inputs.

Training-time scaffolding, never imported by the agent. No torch in this file: workers stay light.

Layout
------
* ``N = workers * envs_per_worker`` environment rows. Every array below has ``N`` as its first
  dimension and lives in one block of shared memory, so nothing but a few bytes of control
  messages crosses a pipe per step.
* Each worker owns a slice of the rows. For each row it runs the environment and the agent's own
  front end (odometry -> memory map -> encoder), exactly as the agent will at evaluation, and
  writes the encoded network inputs into the row. Map crops are stored as uint8 (0..255).
* The learner reads all rows, runs **one batched forward pass**, writes the actions, and signals
  the workers. Workers never run the network.

Episodes
--------
* When an episode ends the worker writes the encoded *final* observation into the ``fin_*`` rows
  (needed to bootstrap the value on truncation), starts the next episode at once, and writes the
  new episode's first observation into the ordinary rows with ``step = 0``.
* Constructing an environment takes about 0.15 s, an episode takes about that long too. So an
  environment is reused for ``episodes_per_map`` episodes of its map (each with a new engine seed:
  the engine is deterministic for a seed). When a row is due for a new map it takes one that a
  background thread has already built from the curriculum's current tier weights; if none is ready
  the row keeps its map for one more episode. Stepping never waits for a construction.
* ``groups=2`` splits every worker's rows into two halves that are stepped alternately, so workers
  step one half while the learner runs the network on the other.

Exit observation: the exit pad is invisible to the range sensors, so the front end takes an "exit
observation" as a fourth input. While training it comes from ground truth with a detector-like
noise model (``exit_source="oracle"``); for the last phase of training the real detector network
can be run on the rendered frame instead (``exit_source="detector"``), as the agent does.

Rows of group g are the contiguous slice ``VecEnv.rows(g)``.
"""

from __future__ import annotations

import atexit
import contextlib
import mmap
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import deps
from training.rl.intrinsic import IntrinsicCoef, IntrinsicReward
from training.rl.reward import TERMS, RewardCoef, RewardTracker

NUM_TIERS = 5                      # map tiers 0..4 (0 = navigation only, used by the curriculum)

# Network inputs and teacher labels for one observation: name -> (shape per row, dtype).
OBS_FIELDS = {
    "rays": ((spec.NUM_RAYS, spec.RAY_FEATS), np.float32),
    "ents": ((spec.MAX_ENTITIES, spec.ENT_FEATS), np.float32),
    "ent_mask": ((spec.MAX_ENTITIES,), np.bool_),
    "scal": ((spec.SCALAR_DIM,), np.float32),
    "map": ((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.uint8),
    "masks": ((2,), np.bool_),                       # (use_ok, attack_ok) from spec.action_masks
    "step": ((), np.int32),                          # index of the observation inside its episode
    "lab_heads": ((len(spec.HEAD_SIZES),), np.int8),  # teacher labels, valid where lab_mask is set
    "lab_subgoal": ((), np.int8),
    "lab_target": ((), np.int16),
    "lab_mask": ((), np.bool_),
}
FINAL_FIELDS = ("rays", "ents", "ent_mask", "scal", "map", "masks", "step")   # stored again as fin_*
STEP_FIELDS = {
    "action": ((spec.NUM_BUTTONS,), np.int8),        # written by the learner
    "reward": ((), np.float32),                      # written by the workers
    "reward_int": ((), np.float32),
    "terminated": ((), np.bool_),
    "truncated": ((), np.bool_),
}


@dataclass
class VecConfig:
    """Settings of the rollout workers."""

    frontend: str = "auto"            # "auto" | "real" | "standin": which odometry / memory / encoder
    exit_source: str = "oracle"       # "oracle": from ground truth; "detector": the CNN on the frame; "none"
    exit_weights: str = ""            # weights file of the exit detector (exit_source="detector")
    exit_noise: bool = True           # degrade the oracle like an imperfect detector (misses, position error)
    exit_range: float = 640.0         # range of the fallback oracle (used when training.exit_label is missing)
    episodes_per_map: int = 8         # episodes before a row asks for a new map
    spare_envs: int = 1               # environments each worker keeps pre-built for rotation
    step_threads: int = -1            # threads per worker for the engine calls; -1 = automatic (see Worker)
    budget_noise: float = 0.1         # assumed budget = true cap * (1 +- this), uniform
    p_default_budget: float = 0.25    # ... or, with this probability, the evaluation default
    seed: int = 0
    reward: RewardCoef = field(default_factory=RewardCoef)
    intrinsic: IntrinsicCoef = field(default_factory=IntrinsicCoef)


# ---------------------------------------------------------------------------------------------
# Shared arrays
# ---------------------------------------------------------------------------------------------
def _field_specs(n: int) -> dict:
    specs = {k: ((n, *shape), dtype) for k, (shape, dtype) in OBS_FIELDS.items()}
    specs.update({"fin_" + k: ((n, *OBS_FIELDS[k][0]), OBS_FIELDS[k][1]) for k in FINAL_FIELDS})
    specs.update({k: ((n, *shape), dtype) for k, (shape, dtype) in STEP_FIELDS.items()})
    specs["tier_probs"] = ((NUM_TIERS,), np.float64)     # curriculum weights, written by the learner
    return specs


def _shm_has_room(size: int) -> bool:
    """True when POSIX shared memory can hold ``size`` bytes (containers often cap /dev/shm at 64 MB)."""
    try:
        st = os.statvfs("/dev/shm")
    except (OSError, AttributeError):
        return True                                      # no /dev/shm to run out of (macOS)
    return st.f_bavail * st.f_frsize > 1.25 * size


class SharedArrays:
    """Named numpy arrays for N environment rows in one block of memory shared between processes.

    The block is POSIX shared memory when there is room for it, otherwise a memory-mapped file in
    the temporary directory (same semantics; avoids the bus error a small /dev/shm would cause).
    ``name`` identifies the block to worker processes: "shm:<name>" or "file:<path>".
    """

    def __init__(self, n: int, name: str | None = None, backing: str = "auto"):
        specs = _field_specs(n)
        offsets, total = {}, 0
        for k, (shape, dtype) in specs.items():
            total = -(-total // 64) * 64                 # 64-byte alignment
            offsets[k] = total
            total += int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.owner, self.n = name is None, n
        self.shm = self.map = self.path = None
        if self.owner and (backing == "shm" or (backing == "auto" and _shm_has_room(total))):
            self.shm = shared_memory.SharedMemory(create=True, size=total)
            self.name = "shm:" + self.shm.name
        elif self.owner:
            fd, self.path = tempfile.mkstemp(prefix="doomrl_", suffix=".bin")
            os.ftruncate(fd, total)
            self.map = mmap.mmap(fd, total)
            os.close(fd)
            self.name = "file:" + self.path
        elif name.startswith("shm:"):
            self.shm, self.name = shared_memory.SharedMemory(name=name[4:], track=False), name
        else:
            with open(name[5:], "r+b") as f:
                self.map = mmap.mmap(f.fileno(), total)
            self.name = name
        buffer = self.shm.buf if self.shm is not None else self.map
        self.arrays = {k: np.ndarray(shape, dtype, buffer=buffer, offset=offsets[k]) for k, (shape, dtype) in specs.items()}
        if self.owner:
            for a in self.arrays.values():
                a[...] = 0

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def close(self) -> None:
        self.arrays = {}
        try:
            if self.shm is not None:
                self.shm.close()
                if self.owner:
                    self.shm.unlink()
            if self.map is not None:
                self.map.close()
        except (BufferError, FileNotFoundError):
            pass
        if self.path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


# ---------------------------------------------------------------------------------------------
# One environment row
# ---------------------------------------------------------------------------------------------
def _total_ammo(obs: dict) -> float:
    return float(np.asarray(obs["vitals"])[4:8].sum())


class EnvSlot:
    """One environment row: the environment, the agent's front end, rewards and episode statistics."""

    def __init__(self, row: int, arrays: SharedArrays, cfg: VecConfig, rng: np.random.Generator, detector=None):
        self.row, self.a, self.cfg, self.rng = row, arrays, cfg, rng
        self.frontend = deps.FrontEnd(cfg.frontend)
        self.intrinsic = IntrinsicReward(cfg.intrinsic)
        self.teacher = deps.make_teacher()
        self.detector = detector
        self.env = None
        self._scratch = np.zeros(OBS_FIELDS["map"][0], np.float32)

    def adopt(self, entry: deps.MapEntry, env, facts) -> None:
        """Take over a constructed environment for the next ``episodes_per_map`` episodes."""
        self.entry, self.env, self.facts = entry, env, facts
        self.native = bool(getattr(env, "native_reward", False))
        self.has_exit = len(facts.exit_centres) > 0 if facts.known else True
        self.reward = None if self.native else RewardTracker(
            self.cfg.reward, facts.n_keys, facts.n_monsters, self.has_exit, facts.known)
        self.episodes_left = self.cfg.episodes_per_map

    def _priv(self) -> dict:
        priv = self.env.privileged_state()
        hits = deps.hit_count(self.env)
        if hits is not None:
            priv["hits"] = hits
        return priv

    def begin_episode(self) -> None:
        """Reset the environment with a fresh seed and write the first observation (step 0)."""
        cfg, rng = self.cfg, self.rng
        obs, _ = self.env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        self.frontend.reset()
        self.intrinsic.reset()
        if self.teacher is not None:
            self.teacher.reset()
        self.t = 0
        if rng.random() < cfg.p_default_budget:
            self.budget = float(spec.DEFAULT_BUDGET_STEPS)
        else:
            cap = self.entry.episode_ticks / spec.FRAME_SKIP
            self.budget = cap * (1.0 + cfg.budget_noise * rng.uniform(-1.0, 1.0))
        priv = self._priv()
        if self.reward is not None:
            self.reward.reset(self.env, priv, _total_ammo(obs))
        self.ret = self.ret_int = 0.0
        self.terms = np.zeros(len(TERMS))
        self.int_terms = np.zeros(2)
        self._observe(obs, None, priv, final=False)

    def _observe(self, obs: dict, action, priv: dict, final: bool):
        """Run the front end on ``obs`` and write its row; returns the intrinsic reward parts."""
        cfg, a, row = self.cfg, self.a, self.row
        exit_obs = None
        if cfg.exit_source == "oracle":
            exit_obs = deps.exit_observation(self.env, self.facts, obs, priv, self.rng if cfg.exit_noise else None, cfg.exit_range)
        elif self.detector is not None and "rgb" in obs:
            exit_obs = self.detector(obs["rgb"])
        feats, crop, pose = self.frontend.observe(obs, action, self.t, self.budget, exit_obs)
        gains = self.intrinsic.step(crop, pose)
        p = "fin_" if final else ""
        a[p + "rays"][row] = feats["rays"]
        a[p + "ents"][row] = feats["ents"]
        a[p + "ent_mask"][row] = feats["ent_mask"]
        a[p + "scal"][row] = feats["scal"]
        np.multiply(crop, 255.0, out=self._scratch)
        np.rint(self._scratch, out=self._scratch)
        np.clip(self._scratch, 0.0, 255.0, out=self._scratch)
        a[p + "map"][row] = self._scratch
        rays, ents = obs["rays"], obs["visible_entities"]
        a[p + "masks"][row] = spec.action_masks(rays["distance"], rays["hit_type"], ents["entity_type"], ents["mask"], obs["vitals"][2])
        a[p + "step"][row] = self.t
        if not final:
            label = deps.teacher_label(self.teacher, self.env, self.frontend, obs, self.t)
            if label is None:
                a["lab_mask"][row] = False
            else:
                a["lab_heads"][row], a["lab_subgoal"][row], a["lab_target"][row], a["lab_mask"][row] = label
        return gains

    def step(self, action: np.ndarray, result=None):
        """Apply one action. Returns the episode summary when the episode ended, else None.

        ``result`` is the tuple returned by ``env.step(action)`` when the caller has already made
        that call (the worker makes the engine calls of all its rows from a thread pool).
        """
        a, row = self.a, self.row
        obs, env_reward, terminated, truncated, _ = self.env.step(action) if result is None else result
        self.t += 1
        priv = self._priv()
        if self.native:
            reward = float(env_reward)
        else:
            reward, terms = self.reward.step(priv, action, _total_ammo(obs))
            self.terms += terms
        done = bool(terminated or truncated)
        reward_int, r_new, r_visit = self._observe(obs, action, priv, final=done)
        a["reward"][row], a["reward_int"][row] = reward, reward_int
        a["terminated"][row], a["truncated"][row] = terminated, truncated
        self.ret += reward
        self.ret_int += reward_int
        self.int_terms += (r_new, r_visit)
        if not done:
            return None
        info = {"row": self.row, "tier": self.entry.tier, "map": self.entry.index, "len": self.t, "ret": self.ret,
                "ret_int": self.ret_int, "int_terms": self.int_terms.tolist(), "terms": self.terms.tolist(),
                "exit": bool(priv.get("exit_reached", False)), "dead": bool(priv.get("dead", False)),
                "truncated": bool(truncated), "has_exit": self.has_exit, "kills": int(priv.get("kills", 0)),
                "monsters": int(self.facts.n_monsters), "health": float(priv.get("health", 0.0)),
                "cells_seen": int(self.intrinsic.n_seen)}
        if self.reward is not None:
            info.update(self.reward.summary())
        return info


# ---------------------------------------------------------------------------------------------
# Worker: a set of rows, stepped in this process
# ---------------------------------------------------------------------------------------------
class _Builder(threading.Thread):
    """Builds environments for upcoming map rotations off the stepping thread.

    Engine start-up waits on the engine process and releases the interpreter lock, so stepping
    continues at nearly full speed while a replacement is prepared. Old environments are closed
    here as well.
    """

    def __init__(self, worker: "Worker", spares: int, seed):
        super().__init__(daemon=True)
        self.worker = worker
        self.ready: queue.Queue = queue.Queue(maxsize=spares)
        self.trash: queue.Queue = queue.Queue()
        self.rng = np.random.default_rng(seed)
        self.stop = threading.Event()
        self.failures = 0

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.trash.get_nowait().close()
                continue
            except queue.Empty:
                pass
            if self.ready.full():
                time.sleep(0.01)
                continue
            try:
                self.ready.put(self.worker.build(self.worker.draw(self.rng)))
            except Exception:                      # keep stepping on the old maps; report the count
                self.failures += 1
                time.sleep(0.5)

    def take(self):
        try:
            return self.ready.get_nowait()
        except queue.Empty:
            return None


class Worker:
    """Steps a set of rows. Runs inside a worker process, or in the caller's process (serial mode)."""

    def __init__(self, wid: int, arrays: SharedArrays, group_rows: list, entries: list, cfg: VecConfig):
        self.a, self.cfg = arrays, cfg
        self.rng = np.random.default_rng([cfg.seed, wid])
        self.by_tier: dict[int, list] = {}
        for e in entries:
            self.by_tier.setdefault(e.tier, []).append(e)
        self.rotate = len(entries) > 1
        if cfg.exit_source not in ("oracle", "detector", "none"):
            raise ValueError(f"unknown exit_source {cfg.exit_source!r}")
        detector = deps.make_exit_detector(cfg.exit_weights) if cfg.exit_source == "detector" else None
        self.groups = [[EnvSlot(row, arrays, cfg, self.rng, detector) for row in rows] for rows in group_rows]
        self.slots = [slot for group in self.groups for slot in group]
        self.pool = self.builder = None
        try:
            for slot in self.slots:
                slot.adopt(*self.build(self.draw(self.rng)))
                slot.begin_episode()
        except BaseException:
            self.close()                           # do not leave engines of the rows already built
            raise
        self.frontend_name = self.groups[0][0].frontend.name
        # Engine calls. On Linux a worker calls its engines one after another: each answers in a
        # fraction of a millisecond. On macOS the engine's inter-process wait falls back to polling
        # with sleeps once several engines share a process, and a call can take milliseconds; made
        # from a thread pool those waits overlap (measured: 3 workers x 16 rows, 690 -> 2750 steps/s
        # on macOS; on Linux threads only add contention, 3650 -> 1640 steps/s).
        per_group = max(len(rows) for rows in group_rows)
        threads = (per_group if sys.platform == "darwin" else 0) if cfg.step_threads < 0 else min(cfg.step_threads, per_group)
        self.pool = ThreadPoolExecutor(threads) if threads > 1 and not all(e.is_toy for e in entries) else None
        if self.rotate and cfg.spare_envs > 0 and not all(e.is_toy for e in entries):
            self.builder = _Builder(self, cfg.spare_envs, [cfg.seed, wid, 1])
            self.builder.start()

    def draw(self, rng: np.random.Generator) -> deps.MapEntry:
        """Draw the next map: a tier from the curriculum's weights, then a map of that tier."""
        tiers = sorted(self.by_tier)
        probs = np.array(self.a["tier_probs"], np.float64)
        w = np.array([probs[t] if 0 <= t < NUM_TIERS else 0.0 for t in tiers])
        if not np.isfinite(w).all() or w.sum() <= 0.0:
            w = np.ones(len(tiers))
        maps = self.by_tier[tiers[int(rng.choice(len(tiers), p=w / w.sum()))]]
        return maps[int(rng.integers(len(maps)))]

    def build(self, entry: deps.MapEntry, attempts: int = 3):
        """Construct the environment of ``entry`` (retried: engines started at once can fail)."""
        for attempt in range(attempts):
            try:
                env = deps.make_env(entry, render_buffers=self.cfg.exit_source == "detector")   # the detector needs frames
                break
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.2 * (attempt + 1))
        facts = deps.standins.BasicMeta() if entry.is_toy else deps.map_facts(entry.path)
        return entry, env, facts

    def step(self, group: int) -> list:
        """Step every row of ``group`` with the actions in the shared array; return episode summaries."""
        actions = self.a["action"]
        slots = self.groups[group]
        results = [None] * len(slots)
        if self.pool is not None:                  # engine calls overlap; they wait outside the interpreter lock
            results = list(self.pool.map(lambda slot: slot.env.step(actions[slot.row]), slots))
        infos = []
        for slot, result in zip(slots, results):
            info = slot.step(actions[slot.row], result)
            if info is not None:
                infos.append(info)
                self._next_map(slot)
                slot.begin_episode()
        return infos

    def _next_map(self, slot: EnvSlot) -> None:
        """At an episode end: move the row to a new map when it is due and one is ready."""
        slot.episodes_left -= 1
        if slot.episodes_left > 0 or not self.rotate:
            return
        if self.builder is None:
            new = self.build(self.draw(self.rng))
        elif slot.episodes_left > min(s.episodes_left for s in self.slots):
            return                                  # a row that has waited longer takes the next one
        else:
            new = self.builder.take()
        if new is None:
            return                                  # nothing built yet: keep the map one more episode
        old = slot.env
        slot.adopt(*new)
        if self.builder is not None:
            self.builder.trash.put(old)
        else:
            old.close()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        if self.builder is not None:
            self.builder.stop.set()
            self.builder.join(timeout=5.0)
            for q in (self.builder.ready, self.builder.trash):
                while not q.empty():
                    item = q.get_nowait()
                    (item[1] if isinstance(item, tuple) else item).close()
        for slot in self.slots:
            if slot.env is not None:
                slot.env.close()
                slot.env = None


def _worker_main(wid: int, conn, shm_name: str, n: int, group_rows: list, entries: list, cfg: VecConfig) -> None:
    """Process entry point: build the rows, then serve step requests until told to close."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # the learner shuts workers down
    scratch = tempfile.mkdtemp(prefix="doomrl_")           # engines write their config file to the cwd
    os.chdir(scratch)
    arrays = worker = None
    try:
        arrays = SharedArrays(n, shm_name)
        worker = Worker(wid, arrays, group_rows, entries, cfg)
        conn.send(("ready", worker.frontend_name))
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                conn.send(("ok", worker.step(arg)))
            else:
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if worker is not None:
            worker.close()
        if arrays is not None:
            arrays.close()
        os.chdir(tempfile.gettempdir())
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Learner side
# ---------------------------------------------------------------------------------------------
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@contextlib.contextmanager
def _light_workers():
    """Start worker processes with one numeric thread each and without re-importing the main script.

    A spawned process normally imports the parent's main module first. For the training script that
    would load torch into every worker (about a second and 200 MB each) although workers never use
    it, so the main module is hidden while the workers start. It is left alone when this file is
    the main module, because the worker entry point then lives there.
    """
    saved_env = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    main = sys.modules.get("__main__")
    hide = main is not None and _worker_main.__module__ != "__main__"
    saved_main = (getattr(main, "__spec__", None), getattr(main, "__file__", None)) if hide else None
    if hide:
        main.__spec__ = None
        if saved_main[1] is not None:
            del main.__file__
    try:
        yield
    finally:
        if hide:
            main.__spec__ = saved_main[0]
            if saved_main[1] is not None:
                main.__file__ = saved_main[1]
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class VecEnv:
    """N environment rows stepped by worker processes (``workers=0``: in this process).

    Usage per group g:  read ``arrays`` rows ``rows(g)`` -> ``send(g, buttons)`` -> ``recv(g)``.
    After ``recv`` the rows hold the next observation, ``reward``, ``reward_int``, ``terminated``
    and ``truncated`` describe the step just taken, and ``fin_*`` hold the final observation of
    every row whose episode ended.
    """

    def __init__(self, entries: list, workers: int, envs_per_worker: int, cfg: VecConfig | None = None,
                 groups: int = 1, tier_probs=None, timeout: float = 300.0):
        if envs_per_worker % groups:
            raise ValueError("envs_per_worker must be a multiple of groups")
        self.cfg = cfg or VecConfig()
        self.groups, self.timeout = groups, timeout
        n_workers = max(workers, 1)
        self.n = n_workers * envs_per_worker
        per = envs_per_worker // groups                    # rows per worker and group
        self.group_size = n_workers * per
        self.arrays = SharedArrays(self.n)
        self.set_tier_probs(np.ones(NUM_TIERS) if tier_probs is None else tier_probs)
        rows = [[[g * self.group_size + w * per + j for j in range(per)] for g in range(groups)] for w in range(n_workers)]
        self.local, self.procs, self.conns = None, [], []
        self.closed = False
        atexit.register(self.close)                        # never leave engines or shared memory behind
        if workers == 0:
            self.local = Worker(0, self.arrays, rows[0], entries, self.cfg)
            self.frontend_name = self.local.frontend_name
            return
        ctx = mp.get_context("spawn")
        with _light_workers():
            for w in range(n_workers):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(w, child, self.arrays.name, self.n, rows[w], entries, self.cfg),
                                   daemon=True)
                proc.start()
                child.close()
                self.procs.append(proc)
                self.conns.append(parent)
        try:
            self.frontend_name = [self._reply(w) for w in range(n_workers)][0]
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------------------------------
    def rows(self, group: int = 0) -> slice:
        return slice(group * self.group_size, (group + 1) * self.group_size)

    def set_tier_probs(self, probs) -> None:
        """Publish the curriculum's tier weights; workers read them when they draw the next map."""
        p = np.zeros(NUM_TIERS)
        probs = np.asarray(probs, np.float64)
        p[:len(probs)] = probs[:NUM_TIERS]
        self.arrays["tier_probs"][:] = p

    def send(self, group: int, buttons: np.ndarray) -> None:
        """Write the actions of ``group`` (group_size, 9) and start stepping its rows."""
        self.arrays["action"][self.rows(group)] = buttons
        self._pending = group
        for conn in self.conns:
            conn.send(("step", group))

    def recv(self, group: int) -> list:
        """Wait for the step of ``group`` to finish; return the summaries of episodes that ended."""
        if self.local is not None:
            return self.local.step(group)
        infos = []
        for w in range(len(self.conns)):
            infos.extend(self._reply(w))
        return infos

    def _reply(self, w: int):
        conn, deadline = self.conns[w], time.monotonic() + self.timeout
        while not conn.poll(0.5):
            if not self.procs[w].is_alive():
                raise RuntimeError(f"rollout worker {w} died")
            if time.monotonic() > deadline:
                raise TimeoutError(f"rollout worker {w} did not answer within {self.timeout:.0f} s")
        kind, payload = conn.recv()
        if kind == "error":
            raise RuntimeError(f"rollout worker {w} failed:\n{payload}")
        return payload

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.close)
        if self.local is not None:
            self.local.close()
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=20.0)
            if proc.is_alive():
                proc.terminate()
        for conn in self.conns:
            conn.close()
        self.arrays.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------------------------
def benchmark(entries: list, workers: int, envs_per_worker: int, steps: int = 300, warmup: int = 30,
              cfg: VecConfig | None = None, groups: int = 1, seed: int = 0) -> dict:
    """Environment steps per second with random actions and no network (the workers' ceiling)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    with VecEnv(entries, workers, envs_per_worker, cfg, groups) as vec:
        build = time.perf_counter() - t0
        episodes = 0
        for i in range(warmup + steps):
            if i == warmup:
                start = time.perf_counter()
                episodes = 0
            for g in range(groups):
                buttons = (rng.random((vec.group_size, spec.NUM_BUTTONS)) < 0.3).astype(np.int8)
                buttons[:, 0] |= rng.random(vec.group_size) < 0.5       # mostly moving forward
                buttons[:, 7:] = 0                                       # leave doors and weapons alone
                vec.send(g, buttons)
            for g in range(groups):
                episodes += len(vec.recv(g))
        elapsed = time.perf_counter() - start
        return {"frontend": vec.frontend_name, "envs": vec.n, "workers": workers, "build_s": build,
                "steps_per_s": steps * vec.n / elapsed, "per_worker": steps * vec.n / elapsed / max(workers, 1),
                "episodes": episodes}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Vector-environment throughput with random actions")
    ap.add_argument("--maps", required=True, help="manifest.jsonl, a WAD, a directory of WADs or toy:NAME")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--frontend", default="auto", choices=("auto", "real", "standin"))
    ap.add_argument("--step-threads", type=int, default=-1, help="-1 = automatic")
    args = ap.parse_args()
    out = benchmark(deps.load_manifest(args.maps), args.workers, args.envs_per_worker, args.steps,
                    cfg=VecConfig(frontend=args.frontend, step_threads=args.step_threads), groups=args.groups)
    print(f"front end {out['frontend']}: {out['envs']} envs on {out['workers']} workers, built in {out['build_s']:.1f} s; "
          f"{out['steps_per_s']:.0f} env steps/s ({out['per_worker']:.0f} per worker), {out['episodes']} episodes finished")


if __name__ == "__main__":
    main()
