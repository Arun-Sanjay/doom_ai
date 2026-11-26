"""Evaluation entry point: `DoomAgent` assembles the modules of `doomnet/` into one decision per call.

Every decision runs the same five stages, in this order:

1. odometry        the previous action and the new observation advance the pose estimate
2. exit detector   a small CNN looks for the exit pad in the RGB frame (the pad is not in the rays)
3. memory map      rays, entities and the exit sighting are written into the start-frame map
4. encoder, masks  the observation becomes the fixed-size network inputs; `spec.action_masks`
                   says whether USE and ATTACK can have any effect right now
5. PolicyNet       reads those inputs and the map crop around the agent: manager (on decision
                   steps 0, MANAGER_PERIOD, 2 * MANAGER_PERIOD, ... counted from `reset()`),
                   value propagation, controller; the six head outputs become the nine buttons

Every button comes from a head of the network. Each head is decoded in one of two fixed ways:
by its most likely option, or by a draw from its distribution, as when the policy was trained.
Which heads take the most likely option is a decoding setting stored with the weights:
`config["greedy_heads"]` of `weights/policy.pt`, a list of names from `spec.HEAD_NAMES`. A file
without that entry gets GREEDY_HEADS: USE and weapon (a second USE press on a moving door would
reverse it), with move, strafe, turn and attack sampled. DOOMNET_GREEDY_HEADS (comma-separated
head names, "all" or "none") overrides the file; it is for evaluation runs that compare
settings and is never set by the scorer. The setting is read once in `__init__` and never looks
at an observation. The manager's choice of sub-goal and target is sampled. Sampling uses
generators that belong to the agent and are re-seeded in `reset()`, so a map always plays the
same way. The time-budget input is `spec.DEFAULT_BUDGET_STEPS` unless DOOMNET_BUDGET_STEPS
overrides it (used for tuning).

Three things here are not learned and none of them reads an observation to choose a button:

* the two action masks of `spec.action_masks`, applied to the logits before decoding;
* the time governor, a compute schedule driven only by the clock. It reads the mean `act()` time
  of the last GOVERNOR_WINDOW decisions, and nothing before GOVERNOR_MIN_DECISIONS decisions of
  the map were timed, so one slow call does not engage it. Level 1, recent mean above
  GOVERNOR_DETECTOR_MS: the exit detector rests on odd decisions. Level 2, recent mean above
  GOVERNOR_NETWORK_MS: the policy network rests on odd decisions too and the previous action is
  repeated. Level 2 also engages when the map's total would pass GOVERNOR_BUDGET_SHARE of the
  scorer's budget (BUDGET_MS_PER_STEP times the decision steps of the tick cap, of which there
  are at least BUDGET_MIN_STEPS) if the remaining decisions cost what the recent ones did.
  Odometry, memory and the encoder run on every decision at every level;
* the failure handling: `act()` never raises, and a fault costs as little of the episode as is
  safe. A pose update that raises is caught where it is called: the pose stays where it was,
  the failure is counted (`odometry_failures`) and the decision goes on as usual. Anything else
  that fails is caught around the whole decision: that decision answers with no buttons and the
  recurrent state of the network is cleared, while the memory map, the pose and the step count
  are kept. Only when FULL_CLEAR_AFTER decisions in a row fail is everything cleared, because a
  fault that repeats may live in what was kept.

Inference runs on the CPU with one thread whatever the `device` argument says. Everything slow
(loading the weights, the first passes through every network and every numpy path) happens in
`__init__`, so the first `act()` of a map costs the same as any other.

Weights: `weights/policy.pt` and `weights/exit_detector.pt`. `save_weights()` writes
`{"state_dict": ..., "config": {...}}` (a plain state dict plus a small JSON-able config; for the
policy `config["policy_kwargs"]` holds the constructor arguments of `PolicyNet`). The file that
`training/rl/export.py` writes (`{"format", "model", "config": constructor arguments, "meta"}`)
and a training checkpoint load as well; `constructor_arguments()` lists where each layout keeps
the arguments, so the agent always rebuilds the architecture that was trained. `__init__` raises,
and the agent does not start, when

* a file is missing or is a Git LFS pointer, unless DOOMNET_ALLOW_UNTRAINED=1 is set, which builds
  that network with its initial weights (tests and benchmarks only);
* a file does not fit its network, or holds a NaN or an infinite value anywhere;
* the decoding setting, in the policy file or in DOOMNET_GREEDY_HEADS, is not a list of head names;
* every warm-up decision failed, which means every decision of a map would fail too.

`pose_estimate()`, `exit_believed()` and `debug_state()` report what the agent holds to the
evaluation runner and the viewer (the failure counters and the governor's numbers are in
`debug_state()`). Nothing reads them to choose an action.

Purpose: turn one observation dict into one (9,) button array, within the time budget, without ever raising.
Shapes: observation (docs/CONTRACT.md) -> rays (1,64,7), ents (1,32,17), ent_mask (1,32), scal (1,44), map (1,11,64,64) uint8, rgb (1,3,120,160) uint8 -> 15 logits -> (9,) int8 of 0/1.
Why: one fixed pipeline, the same as the training workers run, so the networks meet at evaluation exactly the inputs they were trained on.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:          # so that `doomnet` is importable however this file was loaded
    sys.path.insert(0, str(SUBMISSION_DIR))

from doomnet import spec  # noqa: E402
from doomnet.encode import Encoder  # noqa: E402
from doomnet.memory import EXIT_MIN_CONFIDENCE, MemoryMap, crop_origin  # noqa: E402
from doomnet.model import ExitDetector, PolicyNet  # noqa: E402
from doomnet.odometry import Odometry  # noqa: E402

# ---- time governor (a compute schedule; its only inputs are the clock and the number of decisions) ---
GOVERNOR_DETECTOR_MS = 24.0      # recent mean act() time above this: the exit detector runs on even decisions only
GOVERNOR_NETWORK_MS = 32.0       # above this the policy network does too, and the previous action is held
GOVERNOR_WINDOW = 50             # "recent" is the mean over this many decisions
GOVERNOR_MIN_DECISIONS = 50      # no level engages before this many decisions of the map were timed
BUDGET_MS_PER_STEP = 40.0        # the scorer's limit per map: this (doom_env.limits.AGENT_MS_PER_STEP) times the
BUDGET_MIN_STEPS = spec.DEFAULT_BUDGET_STEPS         # decision steps of the tick cap: at least 525 steps, so 21 s
GOVERNOR_BUDGET_SHARE = 0.8      # level 2 also engages when the projected total of the map passes this share of
                                 # that budget (the same share as GOVERNOR_NETWORK_MS of BUDGET_MS_PER_STEP)
# The design target is 8 ms per decision.

# ---- failure handling --------------------------------------------------------------------------------
FULL_CLEAR_AFTER = 3             # this many failed decisions in a row clear the map and the pose as well

# ---- weights -----------------------------------------------------------------------------------------
POLICY_FILE = "policy.pt"
EXIT_DETECTOR_FILE = "exit_detector.pt"
ALLOW_UNTRAINED_ENV = "DOOMNET_ALLOW_UNTRAINED"      # "1": build a network with initial weights when its file is absent
BUDGET_ENV = "DOOMNET_BUDGET_STEPS"                  # overrides spec.DEFAULT_BUDGET_STEPS (used for tuning)
GREEDY_ENV = "DOOMNET_GREEDY_HEADS"                  # overrides the stored decoding setting (evaluation runs only)
GREEDY_KEY = "greedy_heads"                          # entry of the policy file's config: a list of head names
LFS_POINTER_PREFIX = b"version https://git-lfs"      # first bytes of a Git LFS pointer file
UNTRAINED_SEED = 0                                   # initial weights are drawn with this seed, so tests repeat

# ---- decoding ----------------------------------------------------------------------------------------
GREEDY_HEADS = (spec.HEAD_USE, spec.HEAD_WEAPON)     # decoded by most likely option when the policy file names no
                                                     # setting of its own (GREEDY_KEY); the other heads are sampled
SAMPLE_MANAGER = True                                # the manager's sub-goal and target are sampled, as in training
DECODE_SEED = 0                                      # both private generators restart from this in reset()
_HEAD_SLICES = tuple(slice(sum(spec.HEAD_SIZES[:k]), sum(spec.HEAD_SIZES[:k + 1])) for k in range(len(spec.HEAD_SIZES)))
NUM_LOGITS = sum(spec.HEAD_SIZES)

# ---- warm-up -----------------------------------------------------------------------------------------
WARMUP_EPISODES = 4                                  # synthetic episodes played in __init__
WARMUP_STEPS = 2 * spec.MANAGER_PERIOD + 2           # decisions per episode: manager and other steps, twice

# Stage names reported to `stage_hook` (the latency benchmark's per-stage breakdown).
STAGES = ("odometry", "exit_detector", "memory_update", "crop", "encode", "policy_manager", "policy_other", "decode")

_SCREEN_SHAPE = (spec.SCREEN_H, spec.SCREEN_W, 3)
_USE, _ATTACK = 7, 6                                 # button indices (wrapper order)


# ---------------------------------------------------------------------------------------------
# Weight files
# ---------------------------------------------------------------------------------------------
def save_weights(path, net: torch.nn.Module, config: dict | None = None) -> str:
    """Write `net` in the format `DoomAgent` loads and return the file's sha256.

    The file is `{"state_dict": CPU tensors, "config": dict}`. `config` must be JSON-able; for a
    `PolicyNet` its constructor arguments are added as `config["policy_kwargs"]`, so the agent
    rebuilds the same architecture. `config["greedy_heads"]`, when given, is the decoding setting
    the agent will use with these weights (a list of names from `spec.HEAD_NAMES`: the heads
    decoded by their most likely option); a list the agent would refuse raises ValueError here.
    Called by training code; the agent itself never writes.
    """
    config = dict(config or {})
    if isinstance(net, PolicyNet):
        config.setdefault("policy_kwargs", dict(net.config))
    if config.get(GREEDY_KEY) is not None:
        head_indices(config[GREEDY_KEY], f'config["{GREEDY_KEY}"]')
    json.dumps(config)                               # raises TypeError when the config is not JSON-able
    state = {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": config}, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constructor_arguments(blob: dict) -> dict | None:
    """The constructor arguments of the network stored in a loaded weights file, or None when it records none.

    Three layouts keep them in three places, tried in this order:

    1. a training checkpoint: top-level "policy_config", taken from the network when it was saved;
    2. `save_weights()`: `config["policy_kwargs"]` (also the place an older checkpoint has them,
       inside its training configuration);
    3. `training/rl/export.py`: `{"format", "model", "config", "meta"}` with no "optimizer" entry,
       where "config" is the arguments themselves (all of it but the decoding setting, GREEDY_KEY,
       which is not an argument of the network).

    The training code reads its own files by the same rule (`training.rl.train.load_policy`).
    """
    config = blob.get("config") if isinstance(blob.get("config"), dict) else {}
    if isinstance(blob.get("policy_config"), dict):
        return dict(blob["policy_config"])
    if isinstance(config.get("policy_kwargs"), dict):
        return dict(config["policy_kwargs"])
    if "format" in blob and "optimizer" not in blob and isinstance(blob.get("config"), dict):
        return {key: value for key, value in config.items() if key != GREEDY_KEY}
    return None


def read_weights(path) -> tuple[dict | None, dict, str]:
    """(state dict, config, status) of one weights file.

    status is "ok", "missing" or "pointer" (a Git LFS pointer file stands where the weights should
    be); in the last two cases the state dict is None. A file that is there but cannot be read
    raises. Accepted layouts: `{"state_dict", "config"}` (what `save_weights` writes), the export
    file or a training checkpoint with the network under "model", or a bare state dict.

    The returned config is the file's "config" plus two entries: "sha256" of the file and, when
    the file records constructor arguments anywhere (`constructor_arguments`), those arguments
    under "policy_kwargs", whichever layout the file has.
    """
    path = Path(path)
    if not path.is_file():
        return None, {}, "missing"
    data = path.read_bytes()
    if data[:64].startswith(LFS_POINTER_PREFIX):
        return None, {}, "pointer"
    blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise TypeError(f"{path} holds a {type(blob).__name__}, not a state dict")
    config = blob["config"] if isinstance(blob.get("config"), dict) else {}
    wrapped = [blob[key] for key in ("state_dict", "model") if isinstance(blob.get(key), dict)]
    state = wrapped[0] if wrapped else blob
    config = dict(config, sha256=hashlib.sha256(data).hexdigest())
    arguments = constructor_arguments(blob)
    if arguments is not None:
        config["policy_kwargs"] = arguments
    return state, config, "ok"


def non_finite_entries(net: torch.nn.Module) -> list[str]:
    """Names of the tensors in the state dict of `net` (parameters and saved buffers) that hold a NaN or an infinite value."""
    return [name for name, tensor in net.state_dict().items()
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]


def _initial_network(factory, kwargs: dict) -> torch.nn.Module:
    """`factory(**kwargs)` with its initial weights drawn from a fixed seed; the global generator is left as it was."""
    saved = torch.default_generator.get_state()
    torch.default_generator.manual_seed(UNTRAINED_SEED)
    try:
        return factory(**kwargs)
    finally:
        torch.default_generator.set_state(saved)


def load_network(file_name: str, factory, kwargs_key: str | None = None) -> tuple[torch.nn.Module, str]:
    """`load_network_and_config` without the config: (network, one-line description)."""
    return load_network_and_config(file_name, factory, kwargs_key)[:2]


def load_network_and_config(file_name: str, factory, kwargs_key: str | None = None) -> tuple[torch.nn.Module, str, dict]:
    """Build one network and load `WEIGHTS_DIR / file_name` into it.

    Returns (network, one-line description, the file's config as `read_weights` gives it; empty
    when the network was built without a file).

    `kwargs_key` names the config entry that holds the constructor arguments (see `read_weights`);
    None builds the network with its defaults.

    Raises RuntimeError when the file is missing or is a Git LFS pointer, unless
    DOOMNET_ALLOW_UNTRAINED=1, in which case the network keeps its initial weights. A file that is
    there always has to be usable: it raises when the stored arguments or tensors do not fit the
    network, and when any stored value is NaN or infinite (a diverged training run), because such
    a network answers every decision with an error.
    """
    path = WEIGHTS_DIR / file_name
    state, config, status = read_weights(path)
    if state is None:
        problem = ("is missing" if status == "missing" else
                   "is a Git LFS pointer, not the weights themselves (run `git lfs pull`)")
        if os.environ.get(ALLOW_UNTRAINED_ENV) != "1":
            raise RuntimeError(
                f"DoomAgent cannot start: the weights file {path} {problem}. The agent only plays with trained weights. "
                f"Set {ALLOW_UNTRAINED_ENV}=1 to run it with untrained networks (tests and benchmarks only).")
        return _initial_network(factory, {}).eval().requires_grad_(False), f"{file_name} UNTRAINED ({status}, initial weights)", {}
    kwargs = dict(config.get(kwargs_key) or {}) if kwargs_key else {}
    try:
        net = _initial_network(factory, kwargs)
    except TypeError as error:                       # arguments of some other network class
        raise RuntimeError(f"DoomAgent cannot start: the weights file {path} stores constructor arguments "
                           f"{kwargs} that {factory.__name__} does not accept ({error}).") from error
    net.load_state_dict(state)                       # strict: a file for another architecture raises here
    broken = non_finite_entries(net)
    if broken:
        raise RuntimeError(
            f"DoomAgent cannot start: the weights file {path} holds NaN or infinite values in {len(broken)} of its "
            f"tensors ({', '.join(broken[:4])}{', ...' if len(broken) > 4 else ''}). Export a healthy checkpoint.")
    return net.eval().requires_grad_(False), f"{file_name} sha256 {config['sha256'][:12]}", config


# ---------------------------------------------------------------------------------------------
# Decoding setting
# ---------------------------------------------------------------------------------------------
def head_indices(names, where: str) -> tuple[int, ...]:
    """Indices into `spec.HEAD_NAMES` for a list of head names: ascending, each head once.

    `where` says where the list came from, for the message. Raises ValueError when `names` is not
    a list or tuple of names from `spec.HEAD_NAMES` (capitals and spaces around a name are ignored).
    """
    if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{where} must be a list of head names from {list(spec.HEAD_NAMES)}, not {names!r}")
    cleaned = [name.strip().lower() for name in names]
    unknown = [name for name in cleaned if name not in spec.HEAD_NAMES]
    if unknown:
        raise ValueError(f"{where} names {unknown}, which {'is' if len(unknown) == 1 else 'are'} not among the "
                         f"heads {list(spec.HEAD_NAMES)}")
    return tuple(sorted({spec.HEAD_NAMES.index(name) for name in cleaned}))


def greedy_heads_setting(config: dict) -> tuple[tuple[int, ...], str]:
    """(heads decoded by their most likely option, where that was read from) for a policy file's config.

    In this order: DOOMNET_GREEDY_HEADS when it is set and not empty (comma-separated head names,
    or "all", or "none"; for evaluation runs that compare settings), `config["greedy_heads"]` of
    the policy file, GREEDY_HEADS. The other heads are sampled. A value that cannot be read
    raises RuntimeError, so a misspelt setting stops the agent at start-up, where the ship gate
    and whoever started the run see it, and never plays a map with some other decoding.
    """
    heads, source = tuple(GREEDY_HEADS), "default"
    try:
        if config.get(GREEDY_KEY) is not None:
            source = f'{POLICY_FILE} config["{GREEDY_KEY}"]'
            heads = head_indices(config[GREEDY_KEY], source)
        override = os.environ.get(GREEDY_ENV, "").strip()
        if override:
            source = f"{GREEDY_ENV}={override}"
            word = override.lower()
            names = (list(spec.HEAD_NAMES) if word == "all" else [] if word == "none" else
                     [name for name in override.split(",") if name.strip()])
            heads = head_indices(names, source)
    except ValueError as error:
        raise RuntimeError(f"DoomAgent cannot start: {error}.") from error
    return heads, source


# ---------------------------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------------------------
def _index_array(values, top: int) -> np.ndarray:
    """An index field (hit type, entity type) as int64, whatever numeric dtype it arrived in."""
    a = np.asarray(values)
    if a.dtype == np.int64:
        return a
    return np.clip(np.nan_to_num(a.astype(np.float64), nan=0.0), -1, top).astype(np.int64)


def _contract_dtypes(observation: dict) -> dict:
    """A shallow copy of the observation whose index fields have the contract's dtypes.

    The modules index tables with the hit types and entity types, which needs integers; the
    protocol can deliver any numeric dtype. Values are not changed (each module bounds its own
    inputs) and the caller's arrays are not written to.
    """
    rays, ents = observation["rays"], observation["visible_entities"]
    return {
        "rgb": observation.get("rgb"),
        "vitals": observation["vitals"],
        "rays": {"distance": rays["distance"], "hit_type": _index_array(rays["hit_type"], spec.NUM_HIT_TYPES)},
        "visible_entities": {"features": ents["features"],
                             "entity_type": _index_array(ents["entity_type"], spec.NUM_ENTITY_TYPES),
                             "mask": np.asarray(ents["mask"]) != 0},
    }


def _selected_weapon(vitals) -> float:
    """vitals[2] for `spec.action_masks`. A value that is not a number reads as the melee slot, so ATTACK stays masked."""
    weapon = float(vitals[2])
    return weapon if math.isfinite(weapon) else float(spec.MELEE_WEAPON_SLOT)


def _warmup_observation(rng: np.random.Generator, busy: bool) -> dict:
    """A made-up observation with the wrapper's keys, shapes and dtypes; used only to warm the code paths up.

    busy=False: a plain room, nothing visible, standing still. busy=True: random ranges with every
    hit type, a door within USE reach straight ahead, up to 32 entities of every type, motion.
    """
    distance = np.full(spec.NUM_RAYS, 0.25, np.float32)
    hit_type = np.full(spec.NUM_RAYS, spec.HIT_WALL, np.int64)
    features = np.zeros((spec.MAX_ENTITIES, 5), np.float32)
    entity_type = np.zeros(spec.MAX_ENTITIES, np.int64)
    mask = np.zeros(spec.MAX_ENTITIES, np.int8)
    vitals = np.zeros(spec.NUM_VITALS, np.float32)
    vitals[0], vitals[2], vitals[3], vitals[4] = 100.0, 2.0, 50.0, 50.0
    rgb = np.zeros(_SCREEN_SHAPE, np.uint8)
    if busy:
        distance = rng.uniform(0.03, 1.0, spec.NUM_RAYS).astype(np.float32)
        hit_type = rng.integers(0, spec.NUM_HIT_TYPES, spec.NUM_RAYS)
        distance[hit_type == spec.HIT_NONE] = 1.0
        distance[0], hit_type[0] = 0.03, spec.HIT_DOOR
        n = int(rng.integers(1, spec.MAX_ENTITIES + 1))
        angle = rng.uniform(-np.pi / 4, np.pi / 4, n)
        features[:n] = np.column_stack([rng.uniform(0.03, 3.0, n), np.sin(angle), np.cos(angle), rng.normal(0.0, 3.0, (n, 2))])
        entity_type[:n] = rng.integers(1, spec.NUM_ENTITY_TYPES, n)
        mask[:n] = 1
        vitals[14:17] = rng.normal(0.0, 4.0, 3)
        rgb = rng.integers(0, 256, _SCREEN_SHAPE, dtype=np.uint8)
    return {"rgb": rgb, "depth": np.zeros(_SCREEN_SHAPE[:2], np.uint8), "vitals": vitals,
            "rays": {"distance": distance, "hit_type": hit_type},
            "visible_entities": {"features": features, "entity_type": entity_type, "mask": mask}}


# ---------------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------------
class DoomAgent:
    """The submitted agent (docs/CONTRACT.md section 6): `__init__` once, `reset()` before every map, `act()` per decision."""

    def __init__(self, observation_space, action_space, device: str):
        """Load the weights and warm every code path up. `device` is ignored: inference runs on the CPU."""
        started = time.perf_counter()
        self.observation_space = observation_space
        self.action_space = action_space
        self.requested_device = device
        self.device = "cpu"
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:                         # torch allows this once per process; a second agent finds it set
            pass
        torch.set_flush_denormal(True)               # tiny numbers are treated as zero instead of slowing the CPU down

        self.policy, policy_line, policy_config = load_network_and_config(POLICY_FILE, PolicyNet, "policy_kwargs")
        self.exit_detector, detector_line = load_network(EXIT_DETECTOR_FILE, ExitDetector)
        print(f"[doomnet] weights: {policy_line} | {detector_line}", flush=True)
        print(f"[doomnet] device argument {device!r} ignored: inference runs on the CPU with one thread", flush=True)

        # Decoding setting: fixed here for the life of the agent, whatever it observes later.
        self.greedy_heads, self.greedy_source = greedy_heads_setting(policy_config)
        self._greedy = tuple(k in self.greedy_heads for k in range(len(spec.HEAD_SIZES)))
        names = [[name for name, greedy in zip(spec.HEAD_NAMES, self._greedy) if greedy == wanted] for wanted in (True, False)]
        print(f"[doomnet] decoding: most likely option for {', '.join(names[0]) or 'no head'}; "
              f"sampled: {', '.join(names[1]) or 'no head'} ({self.greedy_source})", flush=True)

        self.budget_steps = float(spec.DEFAULT_BUDGET_STEPS)
        override = os.environ.get(BUDGET_ENV)
        if override:
            try:
                self.budget_steps = float(override)
            except ValueError:
                print(f"[doomnet] {BUDGET_ENV}={override!r} is not a number; using {self.budget_steps:g}", flush=True)
            if not (math.isfinite(self.budget_steps) and self.budget_steps > 0.0):
                self.budget_steps = float(spec.DEFAULT_BUDGET_STEPS)
            print(f"[doomnet] time budget input: {self.budget_steps:g} decision steps", flush=True)

        self.odometry, self.memory, self.encoder = Odometry(), MemoryMap(), Encoder()
        self.governor_ms = (GOVERNOR_DETECTOR_MS, GOVERNOR_NETWORK_MS)
        self.governor_min = GOVERNOR_MIN_DECISIONS   # decisions timed on a map before the governor may engage
        self.stage_hook = None                       # optional callable(stage name), set by the latency benchmark
        self.failures = 0                            # decisions that failed and were answered with no buttons
        self.odometry_failures = 0                   # pose updates that raised; the pose was kept and the decision made
        self.odometry_last_failure = ""              # what the last of them was
        self._failure_logged = self._odometry_logged = False
        self._rng = np.random.Generator(np.random.PCG64(DECODE_SEED))     # controller heads
        self._manager_rng = torch.Generator()                             # manager's sub-goal and target
        self.policy.generator = self._manager_rng
        self._map_float = np.zeros((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.float32)
        self._map_bytes = np.zeros((1, spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.uint8)
        self.reset()
        warmup_decisions = self._warm_up()
        self.warmup_failures, self.failures, self._failure_logged = self.failures, 0, False
        self.warmup_odometry_failures, self.odometry_failures, self._odometry_logged = self.odometry_failures, 0, False
        if self.warmup_failures >= warmup_decisions:
            # Nothing the agent holds can produce an action: on a map it would answer every decision
            # with no buttons. Refusing to start scores the same and cannot pass unnoticed.
            raise RuntimeError(
                f"DoomAgent cannot start: all {warmup_decisions} warm-up decisions failed (first traceback above), "
                "so every decision of a map would answer with no buttons. Check the weights files.")
        if self.warmup_failures:
            print(f"[doomnet] WARNING: {self.warmup_failures} of {warmup_decisions} warm-up decisions failed "
                  "(see the traceback above)", flush=True)
        if self.warmup_odometry_failures:
            print(f"[doomnet] WARNING: the pose update failed on {self.warmup_odometry_failures} of {warmup_decisions} "
                  "warm-up decisions (see the traceback above); the agent plays, but without a pose its map is of no use",
                  flush=True)
        self.reset()
        print(f"[doomnet] ready in {time.perf_counter() - started:.2f} s", flush=True)

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Before every map: clear the pose, the memory map, the recurrent state, the clock and the generators."""
        self._clear_state()
        self._step = 0                               # decision steps since reset()
        self._acts = 0                               # act() calls timed on this map
        self._act_seconds = 0.0                      # their total
        self._recent = [0.0] * GOVERNOR_WINDOW       # the last GOVERNOR_WINDOW of them, seconds; slot = call number % window
        self._failed_in_a_row = 0                    # failed decisions since the last one that worked
        self.governor_level = 0
        self._exit_seen = False
        self._rng = np.random.Generator(np.random.PCG64(DECODE_SEED))
        self._manager_rng.manual_seed(DECODE_SEED)

    def _clear_state(self) -> None:
        """Everything that describes the episode so far (also used when decisions keep failing)."""
        self.odometry.reset()
        self.memory.reset()
        self.encoder.reset()
        self._pose = (0.0, 0.0, 0.0)                 # the pose the last decision used: stands in when the odometry's cannot be read
        self._fresh = True                           # no previous action: the next observation starts the pose estimate
        self._clear_recurrent_state()
        self._prev_action = np.zeros(spec.NUM_BUTTONS, np.int8)
        self._out = None                             # network output of the last pass (for debug_state)
        self._target_origin = (0, 0)                 # crop origin at the manager step that chose the target
        self.last_inputs = None                      # what the network was given on its last pass (tests, viewer)

    def _clear_recurrent_state(self) -> None:
        """The network's own state: what a failed decision may have left half-written or not finite."""
        self._state = self.policy.initial_state(1)
        self._need_manager = True                    # the next network pass must run the manager

    # ------------------------------------------------------------------------------------------
    def act(self, observation: dict) -> np.ndarray:
        """One decision: (9,) int8 array of 0/1 in the wrapper's button order. Never raises."""
        started = self._now()
        try:
            action = self._decide(observation)
            self._failed_in_a_row = 0
        except Exception:  # noqa: BLE001 - whatever went wrong, the scorer must still get a valid action
            action = self._recover(observation)
        self._prev_action = action
        self._step += 1
        seconds = self._now() - started
        self._recent[self._acts % GOVERNOR_WINDOW] = seconds
        self._acts += 1
        self._act_seconds += seconds
        return action.copy()

    @staticmethod
    def _now() -> float:
        """The clock of the time governor, in seconds."""
        return time.perf_counter()

    def _recent_seconds(self) -> float:
        """Mean act() time of the last GOVERNOR_WINDOW decisions of this map (of all of them while there are fewer)."""
        return sum(self._recent) / max(1, min(self._acts, GOVERNOR_WINDOW))

    def _governor(self) -> int:
        """0, 1 or 2. Reads the times of the act() calls made on this map and their number, nothing else.

        Recent mean: level 1 above governor_ms[0], level 2 above governor_ms[1]. Nothing engages
        before `governor_min` decisions were timed, so that one slow call among the first of a map
        (a busy machine) cannot change the episode: to lift the mean of GOVERNOR_WINDOW decisions
        over a threshold on its own it would have to take about a second.

        Budget: the scorer stops an agent whose total time on a map passes BUDGET_MS_PER_STEP times
        the decision steps of the map's tick cap. The cap is not known here; it is at least
        BUDGET_MIN_STEPS steps, and once the map has lasted that long it is at least one step more
        than were played. Level 2 engages when the total so far plus the remaining steps up to that
        earliest possible end, at the recent mean, passes GOVERNOR_BUDGET_SHARE of the budget of a
        map that ends there. That catches a slow stretch early in a map that the recent mean has
        already forgotten.
        """
        acts = self._acts
        if acts == 0 or acts < self.governor_min:
            return 0
        recent = self._recent_seconds()
        recent_ms = 1000.0 * recent
        if recent_ms > self.governor_ms[1]:
            return 2
        end = max(BUDGET_MIN_STEPS, acts + 1)                            # the earliest step count the map can end at
        if self._act_seconds + recent * (end - acts) > GOVERNOR_BUDGET_SHARE * BUDGET_MS_PER_STEP / 1000.0 * end:
            return 2
        return 1 if recent_ms > self.governor_ms[0] else 0

    def _decide(self, observation: dict) -> np.ndarray:
        """The pipeline of the module docstring for one observation."""
        self._pose_current = False                   # True once the pose has taken the step that led to this observation
        hook = self.stage_hook
        step = self._step
        level = self.governor_level = self._governor()
        odd = step % 2 == 1
        look = not (level >= 1 and odd)                                  # level 1: the exit detector rests on odd steps
        hold = level >= 2 and odd and not self._need_manager             # level 2: so does the policy network
        obs = _contract_dtypes(observation)
        prev = None if self._fresh else self._prev_action

        # 1. odometry: the action that produced this observation, then the observation itself.
        # A pose update that raises must not cost the episode its map: the pose stays, the rest goes on.
        if prev is None:
            pose = (self.odometry.x, self.odometry.y, self.odometry.theta)
        else:
            try:
                self.odometry.update(prev, obs)
                pose = (self.odometry.x, self.odometry.y, self.odometry.theta)
            except Exception:  # noqa: BLE001 - counted and reported; see _odometry_failed
                pose = self._odometry_failed()
        self._pose_current = True
        if not math.isfinite(pose[0] + pose[1] + pose[2]):               # nothing can be placed on the map from such a pose,
            raise FloatingPointError("the odometry holds a pose that is not finite")      # and only a full clear resets it
        self._pose = pose
        if hook:
            hook("odometry")

        # 2. exit detector
        exit_obs = self._detect(obs["rgb"]) if look else None
        if exit_obs is not None and exit_obs[0] >= EXIT_MIN_CONFIDENCE:
            self._exit_seen = True
        if hook:
            hook("exit_detector")

        # 3. memory map
        self.memory.update(pose, obs, prev, exit_obs, step)
        self._fresh = False
        if hook:
            hook("memory_update")

        # 4. encoder and action masks
        feats = self.encoder.encode(obs, pose, prev, step, self.budget_steps, exit_obs)
        rays, ents = obs["rays"], obs["visible_entities"]
        use_ok, attack_ok = spec.action_masks(rays["distance"], rays["hit_type"], ents["entity_type"], ents["mask"],
                                              _selected_weapon(obs["vitals"]))
        if hook:
            hook("encode")

        if hold:                                     # governor level 2: repeat the previous action, masks still applied
            action = self._prev_action.copy()
            action[_USE] &= use_ok
            action[_ATTACK] &= attack_ok
            return action

        crop = self.memory.crop(pose)
        np.multiply(crop, 255.0, out=self._map_float)                    # the same rounding to bytes as in training
        np.rint(self._map_float, out=self._map_float)
        np.clip(self._map_float, 0.0, 255.0, out=self._map_float)
        self._map_bytes[0] = self._map_float
        if hook:
            hook("crop")

        # 5. policy network
        run_manager = self._need_manager or step % spec.MANAGER_PERIOD == 0
        inputs = {"rays": feats["rays"], "ents": feats["ents"], "ent_mask": feats["ent_mask"], "scal": feats["scal"],
                  "map": self._map_bytes[0], "masks": (use_ok, attack_ok), "run_manager": run_manager, "exit_obs": exit_obs}
        with torch.inference_mode():
            out, self._state = self.policy(
                torch.from_numpy(feats["rays"]).unsqueeze(0), torch.from_numpy(feats["ents"]).unsqueeze(0),
                torch.from_numpy(feats["ent_mask"]).unsqueeze(0), torch.from_numpy(feats["scal"]).unsqueeze(0),
                torch.from_numpy(self._map_bytes), state=self._state, masks=torch.tensor([[use_ok, attack_ok]]),
                run_manager=run_manager, sample=SAMPLE_MANAGER)
            logits = torch.cat(out["logits"], 1)[0].numpy()
        self._need_manager = False
        self._out, self.last_inputs = out, inputs
        if run_manager:
            self._target_origin = crop_origin(pose)
        if hook:
            hook("policy_manager" if run_manager else "policy_other")

        action = self._decode(logits)
        if hook:
            hook("decode")
        return action

    def _detect(self, rgb):
        """Exit detector on one frame: float32 (confidence in [0, 1], forward / 1024, left / 1024), or None without a frame."""
        frame = np.asarray(rgb) if rgb is not None else None
        if frame is None or frame.shape != _SCREEN_SHAPE:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(np.nan_to_num(frame.astype(np.float32)), 0.0, 255.0).astype(np.uint8)
        if not (frame.flags.c_contiguous and frame.flags.writeable):
            frame = np.array(frame)
        with torch.inference_mode():
            out = self.exit_detector(torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0))[0]
            out[0] = torch.sigmoid(out[0])           # the networks output a logit; memory and encoder take a confidence
        return out.numpy()

    def _decode(self, logits: np.ndarray) -> np.ndarray:
        """15 masked logits -> (9,) buttons.

        The heads of the decoding setting (`self.greedy_heads`, fixed in `__init__`) take their
        most likely option. The others are sampled with the Gumbel-max identity: the largest of
        (logits + Gumbel noise) is a draw from softmax(logits). A masked option has logit -inf and
        can never be the largest. The noise is drawn for all 15 logits whatever the setting, so
        the generator advances the same way under every setting.
        """
        if np.isnan(logits).any():
            raise FloatingPointError("the policy network produced NaN logits")
        noisy = logits + self._rng.gumbel(size=NUM_LOGITS)
        greedy = self._greedy
        heads = [int(np.argmax((logits if greedy[k] else noisy)[s])) for k, s in enumerate(_HEAD_SLICES)]
        return spec.heads_to_buttons(heads)

    def _odometry_failed(self) -> tuple[float, float, float]:
        """After an exception from `Odometry.update`: count it, log the first one, return the pose to go on with.

        That pose is the one the odometry still holds (a step that did not complete has not moved
        it), or the pose of the previous decision if that cannot be read or is not finite. The
        odometry is not reset: its next step starts from where it stands.
        """
        self.odometry_failures += 1
        pose = self._pose
        try:
            held = (float(self.odometry.x), float(self.odometry.y), float(self.odometry.theta))
            if math.isfinite(held[0] + held[1] + held[2]):
                pose = held
        except Exception:  # noqa: BLE001 - the previous decision's pose stands
            pass
        try:
            kind, error = sys.exc_info()[:2]
            self.odometry_last_failure = f"{getattr(kind, '__name__', kind)}: {error}"
            if not self._odometry_logged:
                self._odometry_logged = True
                print(f"[doomnet] the pose update failed at decision {self._step}; the pose stays where it was and the "
                      "decision goes on. Later failures are counted but not printed.\n" + traceback.format_exc(),
                      file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - a log that cannot be written changes nothing
            pass
        return pose

    def _recover(self, observation=None) -> np.ndarray:
        """After an exception inside a decision: log the first one, answer with no buttons, keep what is sound.

        Kept: the memory map, the pose, the encoder's counters and the step count. A decision that
        stopped half way cannot leave them unusable. Every write into the map is one whole array
        statement of bounded values, so the map holds part of one observation at worst. The pose
        update of the decision has returned before any later stage runs. If the decision failed
        before it (an observation that could not be read at all), the update is made here with the
        observation as it came: the odometry takes the heading and the model's displacement from
        the buttons alone, so the pose does not fall a step behind the player. And the player
        really does stand through a decision with no buttons, which is what the next pose update
        is told.
        Cleared: the recurrent state of the network, the one thing a failed pass can leave not
        finite or out of step; the next pass starts with the manager.
        After FULL_CLEAR_AFTER failed decisions in a row everything is cleared, as `reset()` does
        (the step count and the clock go on): a fault that repeats may come from what was kept,
        and then keeping it would cost every remaining decision of the map.
        """
        self.failures += 1
        self._failed_in_a_row += 1
        everything = self._failed_in_a_row >= FULL_CLEAR_AFTER
        try:
            if not self._failure_logged:
                self._failure_logged = True
                print(f"[doomnet] act() failed at decision {self._step}; this decision answers with no buttons and the "
                      f"network's recurrent state is cleared; the map and the pose are kept unless {FULL_CLEAR_AFTER} "
                      "decisions in a row fail. Later failures are counted but not printed.\n" + traceback.format_exc(),
                      file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - a log that cannot be written changes nothing
            pass
        try:
            if everything:
                self._clear_state()
            else:
                self._clear_recurrent_state()
                if not (self._pose_current or self._fresh):
                    try:
                        self.odometry.update(self._prev_action, observation)
                    except Exception:  # noqa: BLE001 - counted and reported; see _odometry_failed
                        self._odometry_failed()
                self._fresh = False                  # an action was answered: the next pose update integrates it
        except Exception:  # noqa: BLE001 - nothing that goes wrong in here may reach the caller
            try:                                     # a module that cannot even be reset is replaced
                self.odometry, self.memory, self.encoder = Odometry(), MemoryMap(), Encoder()
                self._clear_state()
            except Exception:  # noqa: BLE001
                self._fresh = self._need_manager = True
        return np.zeros(spec.NUM_BUTTONS, np.int8)

    # ------------------------------------------------------------------------------------------
    def _warm_up(self) -> int:
        """Run every network and every numpy path a few times, so no decision of a map pays a first-use cost.

        Synthetic episodes go through `act()` itself: quiet and busy observations, manager and
        other steps, and both levels of the time governor. The paths of the memory map that only
        a pose near or beyond the edge of the stored area reaches, and an exit sighting, are
        called directly. Returns the number of `act()` calls made; `__init__` compares it with
        `self.failures` to tell an agent that works from one that cannot decide at all.
        """
        rng = np.random.default_rng(0)
        governor_ms, governor_min = self.governor_ms, self.governor_min
        for episode in range(WARMUP_EPISODES):
            self.reset()
            governed = episode == WARMUP_EPISODES - 1                    # the last episode runs at level 2 from its second decision
            self.governor_ms, self.governor_min = ((-1.0, -1.0), 1) if governed else (governor_ms, governor_min)
            for step in range(WARMUP_STEPS):
                self.act(_warmup_observation(rng, busy=episode > 0))
        self.governor_ms, self.governor_min = governor_ms, governor_min
        decisions = WARMUP_EPISODES * WARMUP_STEPS
        if self.failures >= decisions:               # nothing works; __init__ reports it, the rest would only fail again
            return decisions
        obs = _contract_dtypes(_warmup_observation(rng, busy=True))
        sighting = np.array([1.0, 0.2, 0.05], np.float32)
        action = np.ones(spec.NUM_BUTTONS, np.int8)
        for x in (0.0, 3900.0, 10000.0):
            pose = (x, -x, 0.5)
            self.memory.update(pose, obs, action, sighting, 1)
            self.memory.crop(pose)
            self.encoder.encode(obs, pose, action, 1, self.budget_steps, sighting)
        return decisions

    # ------------------------------------------------------------------------------------------
    # Telemetry for the runner and the viewer. None of it is used to choose an action.
    def pose_estimate(self) -> tuple[float, float, float]:
        """(x, y, theta) in the start frame: the odometry's current estimate."""
        return float(self.odometry.x), float(self.odometry.y), float(self.odometry.theta)

    def exit_believed(self) -> bool:
        """True once the exit detector has reported a sighting on this map."""
        return bool(self._exit_seen)

    def debug_state(self) -> dict:
        """What the agent currently holds: pose, the manager's sub-goal and target, exit belief, governor, failures.

        "subgoal" is a name from spec.SUBGOALS; "target" is the manager's crop cell index
        (i * CROP_SIZE + j) in the crop whose origin was "target_origin"; "target_cell" is the same
        cell in the start frame (cell k spans [64k - 32, 64k + 32)). All four are None before the
        first decision of a map.

        "act_ms_mean" is over the whole map, "act_ms_recent" over the governor's window.
        "failures": decisions answered with no buttons since the agent was built.
        "odometry_failures": pose updates that raised since then and were survived with the pose
        kept; "odometry_last_failure" names the last one. "odometry_errors" and
        "odometry_last_error" are the odometry module's own count and note for the current map
        (measurements that failed inside it, where it kept its model's answer), and
        "odometry_errors_total" and "odometry_last_error_total" the same for the life of the
        odometry object, the warm-up included; each is None when the module has no such field.
        "greedy_heads": the heads decoded by their most likely option.
        """
        module = {name: getattr(self.odometry, name, None) for name in ("errors", "last_error", "errors_total", "last_error_total")}
        state = {"pose": self.pose_estimate(), "subgoal": None, "subgoal_index": None, "target": None,
                 "target_origin": None, "target_cell": None, "exit_seen": bool(self._exit_seen), "step": int(self._step),
                 "governor_level": int(self.governor_level), "failures": int(self.failures),
                 "act_ms_mean": 1000.0 * self._act_seconds / max(1, self._acts),
                 "act_ms_recent": 1000.0 * self._recent_seconds(),
                 "odometry_failures": int(self.odometry_failures), "odometry_last_failure": str(self.odometry_last_failure),
                 "odometry_errors": None if module["errors"] is None else int(module["errors"]),
                 "odometry_last_error": None if module["last_error"] is None else str(module["last_error"]),
                 "odometry_errors_total": None if module["errors_total"] is None else int(module["errors_total"]),
                 "odometry_last_error_total": None if module["last_error_total"] is None else str(module["last_error_total"]),
                 "greedy_heads": [spec.HEAD_NAMES[k] for k in self.greedy_heads]}
        if self._out is not None:
            subgoal, target = int(self._out["subgoal"][0]), int(self._out["target"][0])
            i, j = divmod(target, spec.CROP_SIZE)
            origin = self._target_origin
            state.update(subgoal=spec.SUBGOALS[subgoal], subgoal_index=subgoal, target=target,
                         target_origin=origin, target_cell=(origin[0] + i, origin[1] + j))
        return state
