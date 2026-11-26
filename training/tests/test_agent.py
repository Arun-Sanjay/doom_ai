"""Tests of the assembled agent, submission/agent.py::DoomAgent, built in this process.

test_fuzz.py runs the agent behind the scorer's protocol and feeds it malformed observations. This
file looks inside it instead: the weight files, the order of the stages, that the networks get the
same inputs as in training, decoding (and the decoding setting stored with the weights) and masks,
what a failure at each stage costs, the time governor, the time-budget input and the telemetry hooks.

Unless a test says otherwise the agent is built with DOOMNET_ALLOW_UNTRAINED=1 and an empty weights
folder, so every network has its initial weights and the tests do not depend on what is committed
under submission/weights. The files that are committed there are checked by two tests of their own.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import agent as agent_module
from doom_env import limits
from doom_env import spaces as S
from doomnet import spec
from doomnet.model import ExitDetector, PolicyNet
from doomnet.odometry import Odometry

ROOT = Path(__file__).resolve().parents[2]
TWO_ROOMS = ROOT / "maps" / "two_rooms.wad"
COMMITTED_WEIGHTS = agent_module.WEIGHTS_DIR         # submission/weights, before the fixture below swaps it
ZEROS = np.zeros(9, np.int8)
SUBMODULE_CALLS = ("odometry.update", "exit_detector.forward", "memory.update", "encoder.encode", "memory.crop", "policy.forward")


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def torch_threads():
    """The agent sets torch to one thread for the whole process; give the other test files their threads back."""
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module", autouse=True)
def empty_weights_folder(tmp_path_factory):
    """Point the agent at a folder without weights for the whole file (see the module docstring)."""
    agent_module.WEIGHTS_DIR = tmp_path_factory.mktemp("no-weights")
    yield
    agent_module.WEIGHTS_DIR = COMMITTED_WEIGHTS


@pytest.fixture(autouse=True)
def untrained(monkeypatch):
    """No weights are needed for these tests; the budget and decoding overrides must not leak in from the shell."""
    monkeypatch.setenv(agent_module.ALLOW_UNTRAINED_ENV, "1")
    monkeypatch.delenv(agent_module.BUDGET_ENV, raising=False)
    monkeypatch.delenv(agent_module.GREEDY_ENV, raising=False)


def build(device: str = "cpu"):
    return agent_module.DoomAgent(S.observation_space(), S.action_space(), device)


@pytest.fixture(scope="module")
def shared():
    """One agent for the tests that do not change it (each of them calls reset() first)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(agent_module.ALLOW_UNTRAINED_ENV, "1")
        patch.delenv(agent_module.GREEDY_ENV, raising=False)
        return build()


@pytest.fixture(scope="module")
def env():
    from doom_env import make_env

    environment = make_env(str(TWO_ROOMS))
    yield environment
    environment.close()


def blank(door: bool = False, enemy: bool = False, weapon: float = 2.0) -> dict:
    """A quiet observation: walls 256 units away all round, optionally a door within reach and an enemy in view."""
    distance = np.full(S.NUM_RAYS, 0.25, np.float32)
    hit_type = np.full(S.NUM_RAYS, spec.HIT_WALL, np.int64)
    features = np.zeros((S.MAX_ENTITIES, 5), np.float32)
    entity_type = np.zeros(S.MAX_ENTITIES, np.int64)
    mask = np.zeros(S.MAX_ENTITIES, np.int8)
    if door:
        distance[0], hit_type[0] = 0.03, spec.HIT_DOOR
    if enemy:
        features[0], entity_type[0], mask[0] = (0.2, 0.0, 1.0, 0.0, 0.0), spec.ENT_HITSCAN, 1
    vitals = np.zeros(len(S.VITALS_FIELDS), np.float32)
    vitals[0], vitals[2], vitals[3], vitals[4] = 100.0, weapon, 50.0, 50.0
    return {"rgb": np.zeros((S.SCREEN_HEIGHT, S.SCREEN_WIDTH, 3), np.uint8),
            "depth": np.zeros((S.SCREEN_HEIGHT, S.SCREEN_WIDTH), np.uint8), "vitals": vitals,
            "rays": {"distance": distance, "hit_type": hit_type},
            "visible_entities": {"features": features, "entity_type": entity_type, "mask": mask}}


def masks_of(obs: dict) -> tuple[bool, bool]:
    rays, ents = obs["rays"], obs["visible_entities"]
    return spec.action_masks(rays["distance"], rays["hit_type"], ents["entity_type"], ents["mask"], obs["vitals"][2])


def valid(action) -> bool:
    return isinstance(action, np.ndarray) and action.shape == (9,) and action.dtype == np.int8 and set(action.tolist()) <= {0, 1}


def play(agent, env, steps: int, seed: int) -> list[tuple[dict, np.ndarray]]:
    """`steps` decisions on the engine with the agent's own actions: [(observation, action), ...]."""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    out = []
    for _ in range(steps):
        action = agent.act(obs)
        out.append((obs, action))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return out


def press_everything(agent) -> None:
    """Make the "on" options of the attack, use and weapon heads overwhelmingly likely."""
    bias = agent.policy.actor[-1].bias
    with torch.no_grad():
        for head in (spec.HEAD_ATTACK, spec.HEAD_USE, spec.HEAD_WEAPON):
            first = sum(spec.HEAD_SIZES[:head])
            bias[first], bias[first + 1] = -30.0, 30.0


def spy(agent, monkeypatch) -> list[str]:
    """Record every call of the six sub-module entry points, in order."""
    calls: list[str] = []
    for path in SUBMODULE_CALLS:
        owner_name, method = path.split(".")
        owner = getattr(agent, owner_name)
        real = getattr(owner, method)

        def wrapped(*args, _real=real, _path=path, **kwargs):
            calls.append(_path)
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, method, wrapped, raising=False)
    return calls


# ---------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------
def test_missing_weights_raise_with_a_clear_message(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    with pytest.raises(RuntimeError, match=r"policy\.pt is missing.*DOOMNET_ALLOW_UNTRAINED=1"):
        build()
    agent_module.save_weights(tmp_path / "policy.pt", PolicyNet())
    with pytest.raises(RuntimeError, match=r"exit_detector\.pt is missing"):
        build()


@pytest.mark.parametrize("value", ["0", "true", ""])
def test_only_the_value_1_allows_untrained_networks(monkeypatch, tmp_path, value):
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setenv(agent_module.ALLOW_UNTRAINED_ENV, value)
    with pytest.raises(RuntimeError, match="is missing"):
        build()


def test_git_lfs_pointer_raises(monkeypatch, tmp_path):
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 5300000\n"
    (tmp_path / "policy.pt").write_bytes(pointer)
    (tmp_path / "exit_detector.pt").write_bytes(pointer)
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    with pytest.raises(RuntimeError, match=r"policy\.pt is a Git LFS pointer"):
        build()
    monkeypatch.setenv(agent_module.ALLOW_UNTRAINED_ENV, "1")
    assert build().act(blank()).shape == (9,)        # the switch covers a pointer file as well


def test_untrained_networks_are_the_same_every_time(monkeypatch, tmp_path, capsys):
    """Initial weights come from a fixed seed, and building the agent leaves the global generators alone."""
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    torch.manual_seed(123)
    np.random.seed(123)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
    first, second = build(), build()
    assert "policy.pt UNTRAINED (missing, initial weights) | exit_detector.pt UNTRAINED (missing, initial weights)" in capsys.readouterr().out
    assert torch.equal(torch.get_rng_state(), before[0]) and np.array_equal(np.random.get_state()[1], before[1])
    for net in ("policy", "exit_detector"):
        for a, b in zip(getattr(first, net).parameters(), getattr(second, net).parameters()):
            assert torch.equal(a, b)
    assert first.failures == 0 and first.warmup_failures == 0


def test_saved_weights_load_and_their_hashes_are_printed(monkeypatch, tmp_path, capsys):
    """save_weights() writes a plain state dict plus a JSON-able config, and the agent rebuilds the same networks."""
    torch.manual_seed(5)
    policy, detector = PolicyNet(hidden=128, vp_iters=4), ExitDetector()
    sha_p = agent_module.save_weights(tmp_path / "policy.pt", policy, {"note": "test", "updates": 3})
    sha_d = agent_module.save_weights(tmp_path / "exit_detector.pt", detector)
    for name, sha in (("policy.pt", sha_p), ("exit_detector.pt", sha_d)):
        assert sha == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        blob = torch.load(tmp_path / name, map_location="cpu", weights_only=True)     # nothing but tensors and plain data
        assert set(blob) == {"state_dict", "config"} and json.loads(json.dumps(blob["config"])) == blob["config"]
    assert torch.load(tmp_path / "policy.pt", weights_only=True)["config"] == {
        "note": "test", "updates": 3, "policy_kwargs": policy.config}

    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    agent = build()
    assert f"[doomnet] weights: policy.pt sha256 {sha_p[:12]} | exit_detector.pt sha256 {sha_d[:12]}" in capsys.readouterr().out
    assert agent.policy.config == policy.config and agent.policy.hidden == 128
    for mine, saved in ((agent.policy, policy), (agent.exit_detector, detector)):
        assert not mine.training and not any(p.requires_grad for p in mine.parameters())
        for (name, a), (_, b) in zip(mine.state_dict().items(), saved.state_dict().items()):
            assert torch.equal(a, b), name
    assert valid(agent.act(blank())) and agent.failures == 0


def test_training_checkpoint_layout_loads(monkeypatch, tmp_path):
    """A checkpoint with the network under "model" and the constructor arguments in config["policy_kwargs"]."""
    policy = PolicyNet(vp_iters=2)
    torch.save({"model": policy.state_dict(), "iteration": 7, "config": {"policy_kwargs": {"vp_iters": 2}, "seed": 0}},
               tmp_path / "policy.pt")
    torch.save(ExitDetector().state_dict(), tmp_path / "exit_detector.pt")            # a bare state dict
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    agent = build()
    assert agent.policy.config["vp_iters"] == 2
    assert all(torch.equal(a, b) for a, b in zip(agent.policy.parameters(), policy.parameters()))


def export_layout(net) -> dict:
    """The file training/rl/export.py writes: its "config" entry is the constructor arguments themselves."""
    return {"format": "doomnet-policy-v1", "model": net.state_dict(), "config": dict(net.config),
            "meta": {"source": "ckpt_000100.pt", "iteration": 100, "env_steps": 1000}}


def checkpoint_layout(net, policy_kwargs: dict) -> dict:
    """A checkpoint of training/rl/train.py: the arguments under "policy_config", the training configuration under "config"."""
    return {"format": "doomnet-policy-v1", "model": net.state_dict(), "policy_config": dict(net.config),
            "policy_kind": "PolicyNet", "optimizer": {}, "iteration": 3,
            "config": {"policy": "auto", "policy_kwargs": dict(policy_kwargs), "seed": 0}}


WEIGHTS_LAYOUTS = {
    "save_weights": lambda net, kwargs: {"state_dict": net.state_dict(), "config": {"policy_kwargs": dict(net.config)}},
    "export": lambda net, kwargs: export_layout(net),
    "checkpoint": checkpoint_layout,
    "checkpoint with an empty training configuration": lambda net, kwargs: checkpoint_layout(net, {}),
    "checkpoint without policy_config": lambda net, kwargs: {"model": net.state_dict(), "optimizer": {},
                                                             "config": {"policy_kwargs": dict(kwargs), "seed": 0}},
}


@pytest.mark.parametrize("kwargs", [{}, {"vp_iters": 3}, {"hidden": 128}, {"strict_goal_bottleneck": False}], ids=str)
@pytest.mark.parametrize("layout", list(WEIGHTS_LAYOUTS))
def test_every_weights_layout_rebuilds_the_trained_architecture(monkeypatch, tmp_path, layout, kwargs):
    """Whichever file ends up as policy.pt, the agent builds the network that was trained, not the default one."""
    torch.manual_seed(11)
    policy = PolicyNet(**kwargs)
    torch.save(WEIGHTS_LAYOUTS[layout](policy, kwargs), tmp_path / "policy.pt")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    net, _ = agent_module.load_network("policy.pt", PolicyNet, "policy_kwargs")
    assert net.config == policy.config
    saved = policy.state_dict()
    assert all(torch.equal(tensor, saved[name]) for name, tensor in net.state_dict().items())


def test_where_the_constructor_arguments_are_read_from():
    find, arguments = agent_module.constructor_arguments, {"vp_iters": 3}
    assert find({"state_dict": {}, "config": {"policy_kwargs": arguments, "note": "x"}}) == arguments
    assert find({"format": "doomnet-policy-v1", "model": {}, "config": arguments, "meta": {}}) == arguments
    # what was read from the network itself wins over the training configuration
    assert find({"format": "doomnet-policy-v1", "model": {}, "optimizer": {}, "policy_config": arguments,
                 "config": {"policy_kwargs": {"vp_iters": 5}}}) == arguments
    # a checkpoint's training configuration is never mistaken for constructor arguments
    assert find({"format": "doomnet-policy-v1", "model": {}, "optimizer": {}, "config": {"seed": 0}}) is None
    assert find({"state_dict": {}, "config": {}}) is None and find({"weight": torch.zeros(1)}) is None


def test_agent_runs_the_planner_depth_stored_in_the_export_file(monkeypatch, tmp_path):
    """A policy trained with 3 value-propagation iterations must not be played with the default 12."""
    policy = PolicyNet(vp_iters=3)
    torch.save(export_layout(policy), tmp_path / "policy.pt")
    agent_module.save_weights(tmp_path / "exit_detector.pt", ExitDetector())
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    agent = build()
    assert agent.policy.config["vp_iters"] == 3 and agent.policy.vp.iters == 3
    assert valid(agent.act(blank())) and agent.failures == 0 and agent.warmup_failures == 0


def test_the_file_written_by_the_export_script_loads(monkeypatch, tmp_path):
    """training/rl/export.py itself, from a checkpoint to the file the agent reads."""
    export = pytest.importorskip("training.rl.export")
    torch.manual_seed(12)
    policy = PolicyNet(hidden=128, vp_iters=3)
    torch.save(checkpoint_layout(policy, {}), tmp_path / "ckpt_000003.pt")
    weights = tmp_path / "weights"
    written = export.export(str(tmp_path / "ckpt_000003.pt"), weights / "policy.pt")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", weights)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    net, line = agent_module.load_network("policy.pt", PolicyNet, "policy_kwargs")
    assert net.config == policy.config and line == f"policy.pt sha256 {written['sha256'][:12]}"
    saved = policy.state_dict()
    assert all(torch.equal(tensor, saved[name]) for name, tensor in net.state_dict().items())


def test_arguments_of_another_network_class_raise_with_a_clear_message(monkeypatch, tmp_path):
    torch.save({"format": "doomnet-policy-v1", "model": PolicyNet().state_dict(), "config": {"width": 64}}, tmp_path / "policy.pt")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match=r"policy\.pt stores constructor arguments .*PolicyNet does not accept"):
        build()


@pytest.mark.parametrize("allow_untrained", [False, True])
@pytest.mark.parametrize("file_name, tensor, value", [
    ("policy.pt", "actor.2.bias", float("nan")),                 # the action heads
    ("policy.pt", "gru.weight_hh", float("nan")),                # the recurrent state
    ("policy.pt", "tower.stem.weight", float("inf")),            # the map tower
    ("policy.pt", "manager_fc.weight", float("-inf")),           # the manager
    ("exit_detector.pt", "conv1.weight", float("nan")),
])
def test_weights_with_a_nan_or_an_infinite_value_raise(monkeypatch, tmp_path, file_name, tensor, value, allow_untrained):
    """One unusable number in either file stops the agent at start-up, with the file and the tensor named."""
    nets = {"policy.pt": PolicyNet(), "exit_detector.pt": ExitDetector()}
    with torch.no_grad():
        nets[file_name].state_dict()[tensor].view(-1)[0] = value
    for name, net in nets.items():
        agent_module.save_weights(tmp_path / name, net)
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    if not allow_untrained:
        monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    with pytest.raises(RuntimeError, match=rf"{file_name} holds NaN or infinite values in 1 of its tensors \({tensor}\)"):
        build()
    healthy = type(nets[file_name])()
    assert agent_module.non_finite_entries(healthy) == []
    agent_module.save_weights(tmp_path / file_name, healthy)
    agent_module.load_network(file_name, type(healthy), "policy_kwargs" if file_name == "policy.pt" else None)


def test_agent_refuses_to_start_when_every_warm_up_decision_fails(monkeypatch, tmp_path, capsys):
    """Finite weights that overflow to NaN logits: the files load, yet no decision can be made."""
    policy = PolicyNet()
    with torch.no_grad():
        weight = policy.actor[-1].weight
        weight.fill_(3e38)
        weight[:, ::2] = -3e38
    assert agent_module.non_finite_entries(policy) == []
    agent_module.save_weights(tmp_path / "policy.pt", policy)
    agent_module.save_weights(tmp_path / "exit_detector.pt", ExitDetector())
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    decisions = agent_module.WARMUP_EPISODES * agent_module.WARMUP_STEPS
    with pytest.raises(RuntimeError, match=f"all {decisions} warm-up decisions failed"):
        build()
    assert "NaN logits" in capsys.readouterr().err


def test_a_pipeline_that_always_fails_stops_the_agent_even_when_untrained_is_allowed(monkeypatch):
    monkeypatch.setattr(agent_module.Encoder, "encode", lambda self, *args, **kwargs: 1 / 0)
    with pytest.raises(RuntimeError, match="warm-up decisions failed"):
        build()


def test_a_few_failed_warm_up_decisions_only_warn(monkeypatch, capsys):
    real, calls = agent_module.Encoder.encode, []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) <= 3:
            raise ZeroDivisionError("first calls only")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(agent_module.Encoder, "encode", flaky)
    agent = build()
    decisions = agent_module.WARMUP_EPISODES * agent_module.WARMUP_STEPS
    assert f"WARNING: 3 of {decisions} warm-up decisions failed" in capsys.readouterr().out
    assert agent.warmup_failures == 3 and agent.failures == 0
    assert valid(agent.act(blank())) and agent.failures == 0


def test_a_broken_weights_file_raises_even_when_untrained_is_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    (tmp_path / "policy.pt").write_bytes(b"not a checkpoint at all")
    with pytest.raises(Exception):  # noqa: B017 - whatever torch raises for an unreadable file
        build()
    torch.save(PolicyNet(hidden=128).state_dict(), tmp_path / "policy.pt")            # another architecture, no config
    with pytest.raises(RuntimeError, match="size mismatch"):
        build()


def test_save_weights_rejects_a_config_that_is_not_plain_data(tmp_path):
    with pytest.raises(TypeError):
        agent_module.save_weights(tmp_path / "policy.pt", PolicyNet(), {"device": torch.device("cpu")})
    assert not (tmp_path / "policy.pt").exists()


def test_weight_file_names():
    assert (agent_module.POLICY_FILE, agent_module.EXIT_DETECTOR_FILE) == ("policy.pt", "exit_detector.pt")
    assert COMMITTED_WEIGHTS == ROOT / "submission" / "weights"


@pytest.mark.parametrize("file_name, factory, kwargs_key", [(agent_module.POLICY_FILE, PolicyNet, "policy_kwargs"),
                                                           (agent_module.EXIT_DETECTOR_FILE, ExitDetector, None)])
def test_each_committed_weights_file_loads(monkeypatch, file_name, factory, kwargs_key):
    """Whatever is in submission/weights right now must be a real file that fits its network."""
    if not (COMMITTED_WEIGHTS / file_name).is_file():
        pytest.skip(f"submission/weights/{file_name} does not exist yet")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", COMMITTED_WEIGHTS)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    net, line = agent_module.load_network(file_name, factory, kwargs_key)             # raises on a pointer or a misfit
    sha = hashlib.sha256((COMMITTED_WEIGHTS / file_name).read_bytes()).hexdigest()
    assert line == f"{file_name} sha256 {sha[:12]}"
    assert all(bool(torch.isfinite(p).all()) for p in net.parameters())


def test_agent_plays_with_the_committed_weights(monkeypatch, env, capsys):
    """The submission as it stands: no untrained switch, both files present, a clean run on a real map."""
    missing = [n for n in (agent_module.POLICY_FILE, agent_module.EXIT_DETECTOR_FILE) if not (COMMITTED_WEIGHTS / n).is_file()]
    if missing:
        pytest.skip(f"not in submission/weights yet: {', '.join(missing)}")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", COMMITTED_WEIGHTS)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    agent = build()
    assert "UNTRAINED" not in capsys.readouterr().out and agent.warmup_failures == 0
    steps = play(agent, env, 120, seed=0)
    assert all(valid(action) for _, action in steps) and any(action.any() for _, action in steps)
    assert agent.failures == 0


# ---------------------------------------------------------------------------------------------
# CPU only, one thread
# ---------------------------------------------------------------------------------------------
def test_cpu_only_whatever_the_device_argument(capsys):
    agent = build("cuda")
    assert "device argument 'cuda' ignored" in capsys.readouterr().out
    assert agent.device == "cpu" and not torch.cuda.is_initialized()
    assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
    for net in (agent.policy, agent.exit_detector):
        assert all(p.device.type == "cpu" for p in net.parameters())
    assert valid(agent.act(blank())) and agent.failures == 0


# ---------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------
def test_stage_order_on_every_decision(shared, env, monkeypatch):
    """Odometry (previous action, new observation), exit detector, memory update, encoder, crop, network."""
    calls = spy(shared, monkeypatch)
    seen_by_odometry = []
    real_update = shared.odometry.update
    monkeypatch.setattr(shared.odometry, "update", lambda prev, obs: (seen_by_odometry.append(np.array(prev)), real_update(prev, obs))[1])
    stages: list[str] = []
    shared.stage_hook = stages.append
    try:
        steps = play(shared, env, 20, seed=1)
    finally:
        shared.stage_hook = None
    order = [c for c in SUBMODULE_CALLS if c != "odometry.update"]
    assert calls == order + (["odometry.update"] + order) * 19          # no odometry step for the first observation
    assert all(np.array_equal(seen, action) for seen, (_, action) in zip(seen_by_odometry, steps))
    per_step = ["odometry", "exit_detector", "memory_update", "encode", "crop", "policy", "decode"]
    assert [s.replace("_manager", "").replace("_other", "") for s in stages] == per_step * 20
    manager = [i for i, s in enumerate(s for s in stages if s.startswith("policy")) if s == "policy_manager"]
    assert manager == [0, 8, 16]                                         # counted in decision steps from reset()
    assert set(stages) <= set(agent_module.STAGES)


def test_network_inputs_equal_the_training_front_end(shared, env):
    """The training workers run training.rl.deps.FrontEnd; the agent must hand its networks the same arrays."""
    from training.rl.deps import FrontEnd

    front = FrontEnd("real")
    front.reset()
    obs, _ = env.reset(seed=3)
    shared.reset()
    prev = None
    for t in range(48):
        action = shared.act(obs)
        given = shared.last_inputs
        feats, crop, pose = front.observe(obs, prev, t, float(spec.DEFAULT_BUDGET_STEPS), given["exit_obs"])
        assert pose == (shared.odometry.x, shared.odometry.y, shared.odometry.theta)
        for key in ("rays", "ents", "ent_mask", "scal"):
            assert np.array_equal(given[key], feats[key]), (t, key)
        assert given["map"].dtype == np.uint8
        assert np.array_equal(given["map"], np.clip(np.rint(crop * 255.0), 0, 255).astype(np.uint8)), t
        assert given["masks"] == masks_of(obs)
        assert given["run_manager"] == (t % spec.MANAGER_PERIOD == 0)
        assert 0.0 <= given["exit_obs"][0] <= 1.0                        # a confidence, not the detector's logit
        prev = action
        obs, _, terminated, truncated, _ = env.step(action)
        assert not (terminated or truncated)
    assert shared.failures == 0


def test_valid_actions_and_no_internal_errors_on_a_real_map(shared, env):
    steps = play(shared, env, 120, seed=0)
    assert all(valid(action) for _, action in steps)
    assert sum(int(action.any()) for _, action in steps) > 60
    assert shared.failures == 0 and shared.governor_level == 0
    assert shared.governor_ms == (agent_module.GOVERNOR_DETECTOR_MS, agent_module.GOVERNOR_NETWORK_MS) == (24.0, 32.0)


def test_a_map_plays_the_same_way_every_time(shared, env):
    """reset() re-seeds the agent's generators: same map and engine seed, same actions; a second agent agrees."""
    first = play(shared, env, 64, seed=5)
    again = play(shared, env, 64, seed=5)
    other = play(build(), env, 64, seed=5)
    assert len({tuple(a.tolist()) for _, a in first}) > 4                # the heads really are sampled
    for (_, a), (_, b), (_, c) in zip(first, again, other):
        assert np.array_equal(a, b) and np.array_equal(a, c)


def test_sampling_uses_private_generators(shared):
    torch.manual_seed(9)
    np.random.seed(9)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
    shared.reset()
    for _ in range(20):
        shared.act(blank(enemy=True))
    assert torch.equal(torch.get_rng_state(), before[0]) and np.array_equal(np.random.get_state()[1], before[1])
    assert shared.policy.generator is shared._manager_rng


def test_reset_clears_the_episode(shared, env):
    play(shared, env, 30, seed=2)
    assert shared.debug_state()["step"] == 30 and shared.pose_estimate() != (0.0, 0.0, 0.0)
    shared.reset()
    state = shared.debug_state()
    assert state["pose"] == (0.0, 0.0, 0.0) and state["step"] == 0 and state["subgoal"] is None and not state["exit_seen"]
    assert not shared.memory.full().any()
    assert float(shared._state["h"].abs().sum()) == 0.0


# ---------------------------------------------------------------------------------------------
# Decoding and masks
# ---------------------------------------------------------------------------------------------
def test_decode_use_and_weapon_by_most_likely_choice_others_sampled(shared):
    shared.reset()
    logits = np.zeros(agent_module.NUM_LOGITS, np.float32)
    logits[0:3] = (0.0, np.log(3.0), -np.inf)        # move: fwd three times as likely as none, back impossible
    logits[11:13] = (0.0, 0.05)                      # use: "on" barely more likely
    logits[13:15] = (0.05, 0.0)                      # weapon: "off" barely more likely
    actions = np.array([shared._decode(logits) for _ in range(4000)])
    assert actions[:, 7].all() and not actions[:, 8].any()               # most likely choice, every time
    assert not actions[:, 1].any()                                       # a -inf option is never drawn
    assert abs(actions[:, 0].mean() - 0.75) < 0.03                       # sampled with the softmax probabilities
    assert abs(actions[:, 6].mean() - 0.5) < 0.04                        # attack is sampled too
    for pair in ((2, 3), (4, 5)):                                        # strafe and turn: three options each
        assert abs((actions[:, pair[0]] | actions[:, pair[1]]).mean() - 2 / 3) < 0.04
    with pytest.raises(FloatingPointError):
        shared._decode(np.full(agent_module.NUM_LOGITS, np.nan, np.float32))


def decode_as_before(rng, logits) -> np.ndarray:
    """The agent's decoding before the setting existed: USE and weapon by most likely option, the other heads sampled."""
    noisy = logits + rng.gumbel(size=agent_module.NUM_LOGITS)
    heads = [int(np.argmax((logits if k in (spec.HEAD_USE, spec.HEAD_WEAPON) else noisy)[s]))
             for k, s in enumerate(agent_module._HEAD_SLICES)]
    return spec.heads_to_buttons(heads)


def test_default_decoding_setting_is_use_and_weapon_exactly_as_before(shared):
    assert agent_module.GREEDY_HEADS == (spec.HEAD_USE, spec.HEAD_WEAPON)
    assert (agent_module.GREEDY_KEY, agent_module.GREEDY_ENV) == ("greedy_heads", "DOOMNET_GREEDY_HEADS")
    assert shared.greedy_heads == (spec.HEAD_USE, spec.HEAD_WEAPON) and shared.greedy_source == "default"
    assert shared.debug_state()["greedy_heads"] == ["use", "weapon"]
    shared.reset()
    rng, source = np.random.Generator(np.random.PCG64(agent_module.DECODE_SEED)), np.random.default_rng(1)
    for _ in range(500):                                                 # the same draws from the same generator state
        logits = source.normal(0.0, 2.0, agent_module.NUM_LOGITS).astype(np.float32)
        logits[source.integers(0, agent_module.NUM_LOGITS)] = -np.inf
        assert np.array_equal(shared._decode(logits), decode_as_before(rng, logits))


def weights_with_setting(folder: Path, greedy_heads) -> None:
    """Both weights files in `folder`, the policy's config carrying this decoding setting."""
    agent_module.save_weights(folder / "policy.pt", PolicyNet(), {"greedy_heads": greedy_heads})
    agent_module.save_weights(folder / "exit_detector.pt", ExitDetector())


def test_decoding_setting_is_read_from_the_policy_file(monkeypatch, tmp_path, capsys):
    stored = ["move", "strafe", "turn", "use", "weapon"]
    weights_with_setting(tmp_path, stored)
    assert torch.load(tmp_path / "policy.pt", weights_only=True)["config"]["greedy_heads"] == stored
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.delenv(agent_module.ALLOW_UNTRAINED_ENV)
    agent = build()
    assert ('[doomnet] decoding: most likely option for move, strafe, turn, use, weapon; sampled: attack '
            '(policy.pt config["greedy_heads"])') in capsys.readouterr().out
    assert agent.greedy_heads == (0, 1, 2, 4, 5) and agent.debug_state()["greedy_heads"] == stored
    agent.reset()
    logits = np.zeros(agent_module.NUM_LOGITS, np.float32)
    logits[0:3] = (0.0, 0.05, -0.05)                 # move: forward barely most likely
    logits[3:6] = (0.05, 0.0, 0.0)                   # strafe: none
    logits[6:9] = (0.0, 0.0, 0.05)                   # turn: right
    logits[11:13] = (0.0, 0.05)                      # use: on
    logits[13:15] = (0.05, 0.0)                      # weapon: off
    actions = np.array([agent._decode(logits) for _ in range(2000)])
    assert (actions[:, [0, 1, 2, 3, 4, 5, 7, 8]] == (1, 0, 0, 0, 0, 1, 1, 0)).all()       # most likely choice, every time
    assert abs(actions[:, 6].mean() - 0.5) < 0.05                        # attack is not in the list: still sampled
    assert valid(agent.act(blank())) and agent.failures == 0 and agent.warmup_failures == 0


def test_decoding_setting_in_an_export_layout_file_is_not_a_constructor_argument(monkeypatch, tmp_path):
    """Where a file's "config" is the constructor arguments themselves, the setting sits among them and is taken out."""
    policy = PolicyNet(vp_iters=3)
    blob = export_layout(policy)
    blob["config"]["greedy_heads"] = ["use"]
    assert agent_module.constructor_arguments(blob) == policy.config
    torch.save(blob, tmp_path / "policy.pt")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    agent = build()
    assert agent.policy.config == policy.config and agent.greedy_heads == (spec.HEAD_USE,)


def test_greedy_heads_setting_precedence_and_values(monkeypatch):
    setting, every = agent_module.greedy_heads_setting, tuple(range(len(spec.HEAD_NAMES)))
    in_file = 'policy.pt config["greedy_heads"]'
    assert setting({}) == setting({"greedy_heads": None}) == ((spec.HEAD_USE, spec.HEAD_WEAPON), "default")
    assert setting({"greedy_heads": []}) == ((), in_file)                # an empty list is a setting: sample every head
    assert setting({"greedy_heads": ["weapon", "use", "use"]}) == ((spec.HEAD_USE, spec.HEAD_WEAPON), in_file)
    assert setting({"greedy_heads": list(spec.HEAD_NAMES)}) == (every, in_file)
    assert setting({"greedy_heads": ("turn",), "policy_kwargs": {}}) == ((spec.HEAD_TURN,), in_file)
    for value, heads in (("all", every), ("ALL", every), ("none", ()), ("use,weapon,attack", (3, 4, 5)),
                         (" move , Turn ", (0, 2)), ("use,", (4,))):
        monkeypatch.setenv(agent_module.GREEDY_ENV, value)               # the override wins over the file
        assert setting({"greedy_heads": ["attack"]}) == (heads, f"DOOMNET_GREEDY_HEADS={value.strip()}")
        assert setting({})[0] == heads
    for empty in ("", "   "):                                            # set but empty: no override
        monkeypatch.setenv(agent_module.GREEDY_ENV, empty)
        assert setting({"greedy_heads": ["attack"]}) == ((spec.HEAD_ATTACK,), in_file)


@pytest.mark.parametrize("stored", ["use", ["use", "jump"], [4, 5], {"use": True}, "all", 5], ids=str)
def test_a_stored_decoding_setting_that_cannot_be_read_stops_the_agent(monkeypatch, tmp_path, stored):
    """Refused when the file is written, and at start-up if such a file exists anyway: never played with another decoding."""
    with pytest.raises(ValueError, match=r'config\["greedy_heads"\]'):
        agent_module.save_weights(tmp_path / "policy.pt", PolicyNet(), {"greedy_heads": stored})
    assert not (tmp_path / "policy.pt").exists()
    policy = PolicyNet()
    torch.save({"state_dict": policy.state_dict(), "config": {"policy_kwargs": dict(policy.config), "greedy_heads": stored}},
               tmp_path / "policy.pt")
    monkeypatch.setattr(agent_module, "WEIGHTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match=r'DoomAgent cannot start: policy\.pt config\["greedy_heads"\]'):
        build()


@pytest.mark.parametrize("value", ["jump", "use,jump", "all,use", "use weapon"])
def test_a_misspelt_decoding_override_stops_the_agent(monkeypatch, value):
    monkeypatch.setenv(agent_module.GREEDY_ENV, value)
    with pytest.raises(RuntimeError, match="DoomAgent cannot start: DOOMNET_GREEDY_HEADS="):
        build()


def test_override_takes_every_head_by_most_likely_choice_and_a_map_still_repeats(shared, monkeypatch, env, capsys):
    monkeypatch.setenv(agent_module.GREEDY_ENV, "all")
    agent = build()
    assert ("[doomnet] decoding: most likely option for move, strafe, turn, attack, use, weapon; sampled: no head "
            "(DOOMNET_GREEDY_HEADS=all)") in capsys.readouterr().out
    assert agent.debug_state()["greedy_heads"] == list(spec.HEAD_NAMES)
    agent.reset()
    shared.reset()
    source = np.random.default_rng(2)
    for _ in range(200):
        logits = source.normal(0.0, 2.0, agent_module.NUM_LOGITS).astype(np.float32)
        expected = spec.heads_to_buttons([int(np.argmax(logits[s])) for s in agent_module._HEAD_SLICES])
        assert np.array_equal(agent._decode(logits), expected)
        shared._decode(logits)
    # The noise is drawn whatever the setting, so the private generator stands where it does under the default.
    assert agent._rng.bit_generator.state == shared._rng.bit_generator.state
    first = play(agent, env, 64, seed=5)
    again = play(agent, env, 64, seed=5)
    other = play(build(), env, 64, seed=5)
    assert len({tuple(a.tolist()) for _, a in first}) > 1
    for (_, a), (_, b), (_, c) in zip(first, again, other):
        assert np.array_equal(a, b) and np.array_equal(a, c)
    assert agent.failures == 0


def test_masks_are_applied_before_decoding():
    """With a network that wants ATTACK, USE and the weapon key on every step, the masks decide the first two."""
    agent = build()
    press_everything(agent)
    cases = [(blank(), (0, 0)), (blank(door=True), (0, 1)), (blank(enemy=True), (1, 0)),
             (blank(door=True, enemy=True), (1, 1)),
             (blank(door=True, enemy=True, weapon=1.0), (0, 1)),          # fist selected: no ATTACK
             (blank(enemy=True, weapon=float("nan")), (0, 0))]            # unreadable weapon slot: no ATTACK
    agent.reset()
    for _ in range(5):
        for obs, (attack, use) in cases:
            action = agent.act(obs)
            assert (action[6], action[7], action[8]) == (attack, use, 1), (masks_of(blank()), action)
    assert agent.failures == 0


# ---------------------------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------------------------
# What the agent keeps when something inside a decision fails, stage by stage:
#
#   pose update raises                  nothing is lost: the pose stays, the decision is made as usual
#   any later stage raises, once        that decision answers with no buttons and the recurrent state is
#                                       cleared; the map, the pose and the step count are kept
#   the observation cannot be read      the same, and the pose still takes the step of the last action
#   FULL_CLEAR_AFTER failures in a row  everything is cleared, as before a map: the fault may live in what was kept
class Episode:
    """An agent on the engine with a twin odometry that is told what the agent's own odometry is told.

    While nothing fails the two poses are equal. After a failure they are equal only if the agent
    kept its pose and went on integrating from it, so the twin shows that a kept pose is still the
    right one and not merely a pose that is not zero.
    """

    def __init__(self, agent, env, seed: int, steps: int):
        self.agent, self.env, self.twin, self.prev = agent, env, Odometry(), None
        self.obs, _ = env.reset(seed=seed)
        agent.reset()
        self.actions = [self.step() for _ in range(steps)]

    def step(self, twin_sees: bool = True) -> np.ndarray:
        """One decision. twin_sees=False: the agent's odometry is known not to run on this step."""
        if self.prev is not None and twin_sees:
            self.twin.update(self.prev, agent_module._contract_dtypes(self.obs))
        self.prev = self.agent.act(self.obs)
        self.obs = self.env.step(self.prev)[0]
        return self.prev

    def free_cells(self) -> np.ndarray:
        return self.agent.memory.full()[spec.CH_FREE] > 0

    def visited_cells(self) -> np.ndarray:
        return self.agent.memory.full()[spec.CH_VISITED] > 0


def fail_once(monkeypatch, agent, where: str) -> None:
    """The next call of this sub-module entry point raises; the calls after it work."""
    owner = getattr(agent, where.split(".")[0])
    real = getattr(owner, where.split(".")[1])
    armed = [True]

    def fails_once(*args, **kwargs):
        if armed[0]:
            armed[0] = False
            raise RuntimeError("injected failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, where.split(".")[1], fails_once, raising=False)


@pytest.mark.parametrize("where", SUBMODULE_CALLS)
def test_one_failure_at_any_stage_keeps_the_map_and_the_pose(env, monkeypatch, capsys, where):
    agent = build()
    capsys.readouterr()
    episode = Episode(agent, env, seed=4, steps=30)
    free, visited, pose = episode.free_cells(), episode.visited_cells(), agent.pose_estimate()
    assert free.sum() > 20 and visited.sum() > 1 and pose[:2] != (0.0, 0.0)      # there is something to lose
    assert pose == episode.twin.pose and agent.failures == 0

    calls = spy(agent, monkeypatch)
    fail_once(monkeypatch, agent, where)
    in_odometry = where == "odometry.update"
    action = episode.step(twin_sees=not in_odometry)                     # must not raise
    err = capsys.readouterr().err
    state = agent.debug_state()
    assert valid(action) and "injected failure" in err and state["step"] == 31
    if in_odometry:                                                      # the decision is made, with the pose it had
        assert agent.failures == 0 and agent.odometry_failures == 1 and "policy.forward" in calls
        assert agent.pose_estimate() == pose
        assert err.count("the pose update failed") == 1 and "act() failed" not in err
        assert state["odometry_failures"] == 1 and state["odometry_last_failure"] == "RuntimeError: injected failure"
        assert agent.last_inputs["map"][spec.CH_FREE].any()              # the network was shown the map it had built
    else:                                                                # one decision without buttons, nothing else
        assert np.array_equal(action, ZEROS) and agent.failures == 1 and agent.odometry_failures == 0
        assert err.count("act() failed") == 1 and "the pose update failed" not in err
        assert float(agent._state["h"].abs().sum()) == 0.0               # the recurrent state is the one thing cleared
        assert state["failures"] == 1 and state["odometry_failures"] == 0
    assert agent.pose_estimate() == episode.twin.pose and agent.pose_estimate()[:2] != (0.0, 0.0)
    assert (episode.free_cells() & free).sum() == free.sum()             # every cell known to be free still is
    assert (episode.visited_cells() & visited).sum() == visited.sum()

    actions = []
    for i in range(40):
        actions.append(episode.step())
        assert agent.pose_estimate() == episode.twin.pose, i             # integrating on from the pose that was kept
        if i == 0:
            assert agent.last_inputs["map"][spec.CH_FREE].any()          # the map goes back into the network at once
            assert in_odometry or agent.last_inputs["run_manager"]       # a cleared recurrent state starts with the manager
    assert all(valid(a) for a in actions) and sum(int(a.any()) for a in actions) > 15
    assert agent.failures == (0 if in_odometry else 1) and agent.odometry_failures == (1 if in_odometry else 0)
    assert agent.debug_state()["step"] == 71 and agent.pose_estimate() != pose
    # 40 decisions later the map from before the failure is still there. After a failure in a later stage the pose
    # never lost a step, so all of it is. A pose update that failed before it began lost one step for good (here
    # 28 units and 7 degrees: the twin missed the same step), and new scans disagree with a few old cells.
    assert (episode.free_cells() & free).sum() >= (0.85 if in_odometry else 1.0) * free.sum()
    assert (episode.visited_cells() & visited).sum() == visited.sum()
    assert agent.odometry.errors == 0


def test_a_pose_update_that_raises_after_moving_the_pose_loses_nothing(env, monkeypatch):
    """The agent goes on with what the odometry holds: a step that was integrated before the fault is not thrown away."""
    agent = build()
    episode = Episode(agent, env, seed=4, steps=30)
    free, real = episode.free_cells(), agent.odometry.update
    armed = [True]

    def raises_at_the_end(prev, obs):
        real(prev, obs)
        if armed[0]:
            armed[0] = False
            raise RuntimeError("after the step was integrated")

    monkeypatch.setattr(agent.odometry, "update", raises_at_the_end)
    for _ in range(41):
        episode.step()                                                   # the twin sees every step, the failed one too
        assert agent.pose_estimate() == episode.twin.pose
    assert agent.odometry_failures == 1 and agent.failures == 0
    assert (episode.free_cells() & free).sum() == free.sum()


def test_a_pose_update_that_always_fails_costs_no_decision_and_does_not_wipe_the_map(env, monkeypatch, capsys):
    agent = build()
    capsys.readouterr()
    episode = Episode(agent, env, seed=4, steps=30)
    free, pose = episode.free_cells(), agent.pose_estimate()
    monkeypatch.setattr(agent.odometry, "update", lambda prev, obs: 1 / 0)
    actions = [episode.step(twin_sees=False) for _ in range(40)]
    assert all(valid(a) for a in actions) and sum(int(a.any()) for a in actions) > 15
    assert agent.failures == 0 and agent.odometry_failures == 40
    assert agent.pose_estimate() == pose == episode.twin.pose            # frozen where it was, never back to the origin
    # The map is still written on every decision, now from a pose that no longer follows the player, so new scans
    # contradict a few old cells (3 of 118 here after 40 decisions). It is never emptied.
    assert (episode.free_cells() & free).sum() >= 0.9 * free.sum() and agent.last_inputs["map"][spec.CH_FREE].any()
    err = capsys.readouterr().err
    assert err.count("the pose update failed") == 1 and "ZeroDivisionError" in err and "act() failed" not in err
    assert agent.debug_state()["odometry_last_failure"] == "ZeroDivisionError: division by zero"
    agent.reset()                                                        # the counters are for the agent's whole life
    assert agent.odometry_failures == 40 and agent.pose_estimate() == (0.0, 0.0, 0.0)


def test_a_pose_update_that_fails_with_a_dead_log_or_an_unreadable_pose(monkeypatch):
    """Nothing the odometry does can reach the caller: not a dead stderr, not a pose that cannot be read afterwards."""
    agent = build()
    agent.reset()
    for _ in range(3):
        agent.act(blank())

    class Broken:
        def write(self, text):
            raise OSError("stderr is closed")

        def flush(self):
            raise OSError("stderr is closed")

    def breaks_the_pose(prev, obs):
        agent.odometry.x = float("nan")
        raise RuntimeError("left a pose that is not a number")

    before = agent._pose
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", Broken())
        patch.setattr(agent.odometry, "update", breaks_the_pose)
        action = agent.act(blank())
    assert valid(action) and agent.failures == 0 and agent.odometry_failures == 1
    assert agent._pose == before and np.isfinite(before).all()           # the pose of the decision before stands
    assert not np.isnan(agent.last_inputs["scal"]).any()
    # The odometry now holds a pose that is not a number, and no step can repair it. That is an internal error on
    # every decision, so the third one clears everything, the odometry included, and the agent plays on.
    actions = [agent.act(blank()) for _ in range(40)]
    assert all(valid(a) for a in actions) and agent.failures == agent_module.FULL_CLEAR_AFTER
    assert not any(a.any() for a in actions[:3]) and sum(int(a.any()) for a in actions) > 5
    assert np.isfinite(agent.pose_estimate()).all() and agent.odometry_failures == 1


def test_an_odometry_that_fails_in_warm_up_is_reported_and_the_agent_still_plays(monkeypatch, capsys):
    monkeypatch.setattr(agent_module.Odometry, "update", lambda self, prev, obs: 1 / 0)
    agent = build()
    decisions = agent_module.WARMUP_EPISODES * agent_module.WARMUP_STEPS
    updates = decisions - agent_module.WARMUP_EPISODES                   # the first observation of an episode has no action
    out = capsys.readouterr()
    assert f"WARNING: the pose update failed on {updates} of {decisions} warm-up decisions" in out.out
    assert out.err.count("the pose update failed") == 1
    assert (agent.warmup_odometry_failures, agent.odometry_failures, agent.warmup_failures, agent.failures) == (updates, 0, 0, 0)
    agent.reset()
    assert sum(int(agent.act(blank()).any()) for _ in range(30)) > 5 and agent.failures == 0 and agent.odometry_failures == 29


def test_debug_state_reports_the_odometry_modules_own_counter(monkeypatch):
    agent = build()
    agent.reset()
    agent.act(blank())
    state = agent.debug_state()
    assert (state["odometry_failures"], state["odometry_last_failure"]) == (0, "")
    assert (state["odometry_errors"], state["odometry_last_error"]) == (0, "")
    agent.odometry.update(ZEROS, {"vitals": None})                       # caught inside the module, which keeps its model's answer
    state = agent.debug_state()
    assert state["odometry_errors"] == 1 and state["odometry_last_error"].startswith("TypeError at line ")
    assert state["odometry_failures"] == 0 and json.loads(json.dumps(state))["odometry_errors"] == 1
    # The module's counters for its whole life, if it has them (they outlast reset(); the per-map ones do not).
    totals = (getattr(agent.odometry, "errors_total", None), getattr(agent.odometry, "last_error_total", None))
    assert (state["odometry_errors_total"], state["odometry_last_error_total"]) == totals
    agent.reset()
    state = agent.debug_state()
    assert (state["odometry_errors"], state["odometry_last_error"]) == (0, "")
    assert (state["odometry_errors_total"], state["odometry_last_error_total"]) == totals
    monkeypatch.setattr(agent, "odometry", types.SimpleNamespace(x=1.0, y=2.0, theta=0.5))    # a module without the fields
    state = agent.debug_state()
    assert [state[key] for key in ("odometry_errors", "odometry_last_error", "odometry_errors_total",
                                   "odometry_last_error_total")] == [None] * 4 and state["pose"] == (1.0, 2.0, 0.5)


@pytest.mark.parametrize("observation", [None, {}, "observation"])
def test_an_unreadable_observation_keeps_the_map_and_the_pose_in_step(env, observation):
    """The player moved on the action before; the pose takes that step from the buttons although nothing could be read."""
    agent = build()
    episode = Episode(agent, env, seed=4, steps=30)
    free = episode.free_cells()
    real, episode.obs = episode.obs, observation
    episode.twin.update(episode.prev, agent_module._contract_dtypes(real))   # the twin is shown what the agent was denied
    episode.prev = agent.act(observation)
    episode.obs = env.step(episode.prev)[0]
    assert np.array_equal(episode.prev, ZEROS) and agent.failures == 1 and agent.odometry_failures == 0
    assert (episode.free_cells() & free).sum() == free.sum()
    state = agent.debug_state()
    assert state["odometry_errors"] == 1 and state["odometry_last_error"]    # counted by the module: it had only the buttons
    x, y, theta = agent.pose_estimate()
    assert theta == episode.twin.theta                                       # the heading needs the buttons alone: exact
    assert abs(x - episode.twin.x) < 4.0 and abs(y - episode.twin.y) < 4.0, (agent.pose_estimate(), episode.twin.pose)
    for _ in range(40):
        episode.step()
        assert agent.pose_estimate()[2] == episode.twin.theta
    x, y, _ = agent.pose_estimate()
    assert abs(x - episode.twin.x) < 4.0 and abs(y - episode.twin.y) < 4.0, (agent.pose_estimate(), episode.twin.pose)
    assert agent.failures == 1 and agent.debug_state()["step"] == 71


def test_failures_in_a_row_clear_everything_at_the_third(env, monkeypatch):
    """One or two failed decisions keep the map; a fault that repeats gets the full clear the handler always gave."""
    assert agent_module.FULL_CLEAR_AFTER == 3
    agent = build()
    episode = Episode(agent, env, seed=4, steps=30)
    free = episode.free_cells()
    with monkeypatch.context() as patch:
        patch.setattr(agent.encoder, "encode", lambda *a, **k: 1 / 0)
        for failed in (1, 2):
            assert np.array_equal(episode.step(), ZEROS) and agent.failures == failed
            assert (episode.free_cells() & free).sum() == free.sum() and agent.pose_estimate() == episode.twin.pose
        assert np.array_equal(episode.step(twin_sees=False), ZEROS) and agent.failures == 3
        assert not agent.memory.full().any() and agent.pose_estimate() == (0.0, 0.0, 0.0)
        assert np.array_equal(episode.step(twin_sees=False), ZEROS) and agent.failures == 4
    actions = [episode.step(twin_sees=False) for _ in range(40)]
    assert agent.failures == 4 and sum(int(a.any()) for a in actions) > 15 and agent.debug_state()["step"] == 74
    assert agent.memory.full()[spec.CH_FREE].any() and agent.odometry_failures == 0

    episode = Episode(agent, env, seed=4, steps=30)                      # two failures, one good decision, two more:
    free = episode.free_cells()                                          # never three in a row, so the map stays
    for _ in range(2):
        for _ in range(2):
            fail_once(monkeypatch, agent, "encoder.encode")
            assert np.array_equal(episode.step(), ZEROS)
        episode.step()
    assert agent.failures == 8 and (episode.free_cells() & free).sum() == free.sum()
    assert agent.pose_estimate() == episode.twin.pose


def test_a_fault_that_lives_in_the_kept_state_is_cleared_with_it(env, monkeypatch):
    """Why the handler does not keep the map for ever: a map that makes a stage fail would cost every later decision."""
    agent = build()
    episode = Episode(agent, env, seed=4, steps=30)
    agent.memory.exit[3, 3] = 7.0                                        # stands for kept state that a stage cannot digest
    real_crop = agent.memory.crop

    def crop(pose):
        if float(agent.memory.exit.max()) > 1.0:
            raise ValueError("the map holds a value no write can produce")
        return real_crop(pose)

    monkeypatch.setattr(agent.memory, "crop", crop)
    actions = [episode.step(twin_sees=False) for _ in range(40)]
    assert agent.failures == agent_module.FULL_CLEAR_AFTER               # and not 40
    assert all(not a.any() for a in actions[:3]) and sum(int(a.any()) for a in actions) > 15


def test_a_failure_on_every_step_is_logged_once_and_never_raised(monkeypatch, capsys):
    agent = build()
    capsys.readouterr()
    agent.reset()
    with monkeypatch.context() as patch:
        patch.setattr(agent.encoder, "encode", lambda *a, **k: 1 / 0)
        actions = [agent.act(blank()) for _ in range(30)]
    assert all(valid(a) and not a.any() for a in actions) and agent.failures == 30
    err = capsys.readouterr().err
    assert err.count("act() failed") == 1 and "ZeroDivisionError" in err
    assert sum(int(agent.act(blank()).any()) for _ in range(30)) > 5      # the network is back as soon as the fault is gone
    assert agent.failures == 30


def test_nan_from_the_network_does_not_stick(capsys):
    """NaN logits are an internal error: the recurrent state is cleared, so the next decision is clean."""
    agent = build()
    agent.reset()
    for _ in range(5):
        agent.act(blank())
    bias = agent.policy.actor[-1].bias
    saved = bias.clone()
    with torch.no_grad():
        bias.fill_(float("nan"))
    assert not agent.act(blank()).any() and agent.failures == 1
    assert float(agent._state["h"].abs().sum()) == 0.0 and agent._need_manager
    with torch.no_grad():
        bias.copy_(saved)
    actions = [agent.act(blank()) for _ in range(30)]
    assert agent.failures == 1 and sum(int(a.any()) for a in actions) > 5
    assert "NaN logits" in capsys.readouterr().err


def test_recovery_survives_a_module_that_cannot_be_reset_and_a_dead_log(monkeypatch):
    """The handler itself must not raise: not when a module's reset() fails as well, not when stderr is gone."""
    agent = build()
    agent.reset()

    class Broken:
        def write(self, text):
            raise OSError("stderr is closed")

        def flush(self):
            raise OSError("stderr is closed")

    broken_memory = agent.memory
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", Broken())
        patch.setattr(broken_memory, "update", lambda *a, **k: 1 / 0)
        patch.setattr(broken_memory, "reset", lambda: 1 / 0)
        for failed in range(1, agent_module.FULL_CLEAR_AFTER + 1):
            assert agent.memory is broken_memory                         # a dead log alone costs the map nothing
            action = agent.act(blank())
            assert valid(action) and not action.any() and agent.failures == failed
    assert agent.memory is not broken_memory                             # the full clear could not reset it: replaced
    assert sum(int(agent.act(blank()).any()) for _ in range(30)) > 5 and agent.failures == agent_module.FULL_CLEAR_AFTER


@pytest.mark.parametrize("observation", [None, {}, {"rays": {}}, "observation", 3])
def test_garbage_instead_of_an_observation_never_raises(shared, observation):
    shared.reset()
    action = shared.act(observation)
    assert valid(action) and not action.any()
    assert valid(shared.act(blank()))
    for _ in range(agent_module.FULL_CLEAR_AFTER + 2):                   # also in the middle of a map, and several in a row
        action = shared.act(observation)
        assert valid(action) and not action.any()
    assert valid(shared.act(blank())) and shared.odometry_failures == 0
    shared.failures = 0


# ---------------------------------------------------------------------------------------------
# Time governor
# ---------------------------------------------------------------------------------------------
class ScriptedClock:
    """Stands in for the agent's clock: every act() appears to take `cost_ms`."""

    def __init__(self):
        self.now, self.cost_ms, self.reads = 0.0, 1.0, 0

    def __call__(self) -> float:
        self.reads += 1
        if self.reads % 2 == 0:                      # act() reads the clock when it starts and when it ends
            self.now += self.cost_ms / 1000.0
        return self.now


# 2 ms, then a long stretch at 30 ms (between the two thresholds), one at 36 ms (over both), then 2 ms again.
SUSTAINED = [2.0] * 60 + [30.0] * 80 + [36.0] * 80 + [2.0] * 120


def clocked_run(agent, monkeypatch, costs, observations=lambda step: blank()) -> list[dict]:
    """One map on a scripted clock, with the agent's own thresholds; one record per decision.

    "level" is the governor level the decision ran at, which the agent worked out from the cost of
    the decisions before it.
    """
    clock = ScriptedClock()
    with monkeypatch.context() as patch:
        patch.setattr(agent, "_now", clock)
        calls = spy(agent, patch)
        agent.reset()
        rows = []
        for step, cost in enumerate(costs):
            clock.cost_ms = cost
            del calls[:]
            obs = observations(step)
            action = agent.act(obs)
            rows.append({"step": step, "level": agent.governor_level, "calls": list(calls), "action": action, "obs": obs,
                         "recent_ms": agent.debug_state()["act_ms_recent"]})
    return rows


def test_governor_constants_match_the_scorers_budget():
    assert (agent_module.GOVERNOR_DETECTOR_MS, agent_module.GOVERNOR_NETWORK_MS) == (24.0, 32.0)
    assert agent_module.GOVERNOR_WINDOW == 50 and agent_module.GOVERNOR_MIN_DECISIONS == 50
    assert agent_module.BUDGET_MS_PER_STEP == limits.AGENT_MS_PER_STEP == 40.0
    assert agent_module.BUDGET_MIN_STEPS == 525                          # a 2,100-tick map, the shortest there is
    assert limits.agent_budget_s(2100) == agent_module.BUDGET_MS_PER_STEP / 1000.0 * agent_module.BUDGET_MIN_STEPS == 21.0
    assert agent_module.GOVERNOR_BUDGET_SHARE == agent_module.GOVERNOR_NETWORK_MS / agent_module.BUDGET_MS_PER_STEP


@pytest.mark.parametrize("step, cost_ms", [(3, 200.0), (0, 35.0), (0, 150.0), (0, 1000.0), (49, 200.0), (120, 500.0)])
def test_one_slow_call_does_not_engage_the_governor(monkeypatch, step, cost_ms):
    """A single slow decision (a busy machine) changes nothing: same levels, same passes, same actions as without it."""
    agent = build()
    costs = [2.0] * 200
    plain = clocked_run(agent, monkeypatch, costs)
    costs[step] = cost_ms
    slow = clocked_run(agent, monkeypatch, costs)
    assert [r["level"] for r in slow] == [0] * 200
    for row in slow:
        assert "exit_detector.forward" in row["calls"] and "policy.forward" in row["calls"], row["step"]
    assert all(np.array_equal(a["action"], b["action"]) for a, b in zip(plain, slow))
    assert agent.failures == 0 and agent.debug_state()["act_ms_mean"] == pytest.approx((398.0 + cost_ms) / 200)


def test_governor_levels_engage_under_sustained_load_and_release(monkeypatch):
    """Sustained 30 ms engages level 1, sustained 36 ms level 2, and both release when the load is gone."""
    agent = build()
    press_everything(agent)
    rows = clocked_run(agent, monkeypatch, SUSTAINED, lambda step: blank(door=step % 2 == 0, enemy=step % 2 == 0))
    levels = [r["level"] for r in rows]
    changes = [levels[0]] + [b for a, b in zip(levels, levels[1:]) if a != b]
    assert changes == [0, 1, 2, 1, 0], changes                           # both levels engage, then both release
    # The mean of the last 50 decisions passes 24 ms after 40 decisions at 30 ms, and 32 ms after 17 more at 36 ms.
    assert levels.index(1) == 60 + 40 and levels.index(2) == 140 + 17
    assert levels[220:].index(1) == 6 and levels[220:].index(0) == 18    # released as the window fills with 2 ms
    assert rows[99]["recent_ms"] > 24.0 > rows[98]["recent_ms"]          # "recent_ms" is read after the decision

    for prev, row in zip(rows, rows[1:]):
        odd, level, calls = row["step"] % 2 == 1, row["level"], row["calls"]
        assert ("exit_detector.forward" in calls) == (not (level >= 1 and odd)), row["step"]
        assert ("policy.forward" in calls) == (not (level >= 2 and odd)), row["step"]
        assert ("memory.crop" in calls) == ("policy.forward" in calls)
        for always in ("odometry.update", "memory.update", "encoder.encode"):
            assert always in calls, (row["step"], always)                # the pose and the memory never skip a step
        if level >= 2 and odd:                                           # held: the previous action, masks still applied
            expected = prev["action"].copy()
            use_ok, attack_ok = masks_of(row["obs"])
            expected[7] &= use_ok
            expected[6] &= attack_ok
            assert np.array_equal(row["action"], expected), row["step"]
            assert prev["action"][6] == 1 and prev["action"][7] == 1 and row["action"][6] == 0 and row["action"][7] == 0
        assert valid(row["action"])
    assert sum(r["level"] == 1 for r in rows) > 30 and sum(r["level"] == 2 for r in rows) > 30
    assert agent.failures == 0

    agent.reset()                                                        # a new map starts with a clean clock
    assert agent.debug_state()["act_ms_recent"] == 0.0 and agent.debug_state()["act_ms_mean"] == 0.0
    agent.act(blank())
    assert agent.governor_level == 0


def test_governor_waits_for_its_minimum_number_of_decisions(monkeypatch):
    """Even far over both thresholds from the first decision on, nothing engages before 50 decisions were timed."""
    agent = build()
    levels = [r["level"] for r in clocked_run(agent, monkeypatch, [100.0] * 70)]
    assert levels == [0] * 50 + [2] * 20


def test_governor_engages_level_2_when_the_map_would_run_out_of_budget(monkeypatch):
    """A slow start that the recent mean has forgotten: 100 decisions at 100 ms have used 10 s of the 21 s."""
    agent = build()
    rows = clocked_run(agent, monkeypatch, [100.0] * 100 + [22.0] * 800)
    levels = [r["level"] for r in rows]
    # From decision 150 on the last 50 decisions took 22 ms each: under both thresholds. But 10 s + 22 ms for each of
    # the 425 decisions that the shortest map still has makes 19.35 s, over 0.8 x 21 s = 16.8 s.
    assert all(r["recent_ms"] == pytest.approx(22.0) for r in rows[151:])
    assert levels[50:771] == [2] * 721
    # Past 525 decisions the map can end at any step, so the total so far is compared with the budget of a map that
    # ends at the next one: 7.8 s + 22 ms x n falls under 0.8 x 40 ms x n at n = 779.
    assert levels[790:] == [0] * 110 and 2 not in levels[780:]
    for row in rows[150:771]:                                            # level 2 does what it does under the other rule
        assert ("policy.forward" in row["calls"]) == (row["step"] % 2 == 0)

    quick = [r["level"] for r in clocked_run(agent, monkeypatch, [100.0] * 100 + [2.0] * 200)]
    assert quick[50:100] == [2] * 50 and quick[150:] == [0] * 150        # 10 s + 2 ms x 425 is no danger: released

    steady = [r["level"] for r in clocked_run(agent, monkeypatch, [30.0] * 700)]
    assert steady == [0] * 50 + [1] * 650                                # 30 ms x 525 = 15.75 s: level 1, never level 2


def test_governor_is_a_schedule_of_the_clock_alone(monkeypatch):
    """Two different observation streams under the same clock get the same levels and the same skipped passes."""
    agent = build()
    rng = np.random.default_rng(0)
    quiet = clocked_run(agent, monkeypatch, SUSTAINED)
    busy = clocked_run(agent, monkeypatch, SUSTAINED, lambda step: agent_module._warmup_observation(rng, busy=True))
    assert [r["level"] for r in quiet] == [r["level"] for r in busy] and {r["level"] for r in quiet} == {0, 1, 2}
    networks = ("exit_detector.forward", "policy.forward")
    assert [[c for c in r["calls"] if c in networks] for r in quiet] == [[c for c in r["calls"] if c in networks] for r in busy]


def test_governor_on_the_real_clock(shared):
    """Thresholds below any real act() time engage both levels once the minimum is met; the real thresholds release them."""
    shared.reset()
    shared.governor_ms = (1e-9, 2e-9)
    try:
        for _ in range(6):
            assert valid(shared.act(blank()))
        assert shared.governor_level == 0                                # 6 decisions: under the minimum of 50
        shared.governor_min = 1
        shared.act(blank())
        assert shared.governor_level == 2
        shared.governor_ms = (1e-9, 1e9)
        shared.act(blank())
        assert shared.governor_level == 1
    finally:
        shared.governor_ms = (agent_module.GOVERNOR_DETECTOR_MS, agent_module.GOVERNOR_NETWORK_MS)
        shared.governor_min = agent_module.GOVERNOR_MIN_DECISIONS
    shared.act(blank())
    assert shared.governor_level == 0 and shared.failures == 0
    fresh = build()
    assert (fresh.governor_ms, fresh.governor_min) == ((24.0, 32.0), 50)             # the warm-up puts its own values back


# ---------------------------------------------------------------------------------------------
# Time-budget input
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value, steps", [(None, 525.0), ("300", 300.0), ("787.5", 787.5), ("soon", 525.0), ("-5", 525.0), ("nan", 525.0)])
def test_budget_input_and_its_override(monkeypatch, value, steps):
    assert spec.DEFAULT_BUDGET_STEPS == 525
    if value is not None:
        monkeypatch.setenv(agent_module.BUDGET_ENV, value)
    agent = build()
    assert agent.budget_steps == steps
    agent.act(blank())
    agent.act(blank())
    scal = agent.last_inputs["scal"]
    assert scal[spec.SCALAR_INDEX["budget_frac"]] == pytest.approx(steps / spec.STEP_NORM)
    assert scal[spec.SCALAR_INDEX["budget_left"]] == pytest.approx((steps - 1) / spec.STEP_NORM)
    assert scal[spec.SCALAR_INDEX["step_frac"]] == pytest.approx(1 / spec.STEP_NORM)


# ---------------------------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------------------------
def test_pose_estimate_and_debug_state(shared, env):
    play(shared, env, 20, seed=6)
    pose = shared.pose_estimate()
    assert len(pose) == 3 and all(type(v) is float for v in pose)
    assert pose == (shared.odometry.x, shared.odometry.y, shared.odometry.theta)
    state = shared.debug_state()
    json.dumps(state)                                                    # plain data
    assert state["pose"] == pose and state["subgoal"] in spec.SUBGOALS and state["exit_seen"] in (True, False)
    assert spec.SUBGOALS[state["subgoal_index"]] == state["subgoal"]
    i, j = divmod(state["target"], spec.CROP_SIZE)
    assert 0 <= state["target"] < spec.CROP_SIZE ** 2
    assert state["target_cell"] == (state["target_origin"][0] + i, state["target_origin"][1] + j)
    assert shared.exit_believed() is state["exit_seen"]
    assert hasattr(shared, "memory") and hasattr(shared, "odometry")     # the viewer reads the agent's own map


def test_exit_sighting_reaches_memory_encoder_and_telemetry(monkeypatch):
    """The detector's logit becomes a confidence; a sighting marks the map, sets the scalars and latches the belief."""
    agent = build()
    agent.reset()
    agent.act(blank())
    assert not agent.exit_believed() or agent.last_inputs["exit_obs"][0] >= 0.5
    agent.reset()
    monkeypatch.setattr(agent.exit_detector, "forward", lambda rgb: torch.tensor([[5.0, 0.25, 0.0]]))
    agent.act(blank())
    seen = agent.last_inputs["exit_obs"]
    assert seen.dtype == np.float32 and seen[0] == pytest.approx(1 / (1 + np.exp(-5.0)), abs=1e-6) and seen[1] == 0.25
    assert agent.exit_believed() and agent.debug_state()["exit_seen"]
    scal = agent.last_inputs["scal"]
    assert scal[spec.SCALAR_INDEX["exit_visible"]] == 1.0 and scal[spec.SCALAR_INDEX["exit_x"]] == 0.25
    centre = spec.CROP_SIZE // 2
    assert agent.last_inputs["map"][spec.CH_EXIT, centre + 4, centre] > 0          # 256 units ahead = 4 crop cells
    monkeypatch.setattr(agent.exit_detector, "forward", lambda rgb: torch.tensor([[-5.0, 0.25, 0.0]]))
    agent.act(blank())
    assert agent.exit_believed()                                         # the belief stays for the rest of the map
    agent.reset()
    assert not agent.exit_believed()


def test_runner_and_scorer_accept_the_agent():
    """Through training.runner, as evaluation runs it: no agent or hook error, time limits on, tight odometry."""
    from training import runner

    rows = runner.run_episodes("training.runner:AgentPolicy", [{"wad": str(TWO_ROOMS), "episode_ticks": 800}],
                               workers=0, seeds=[0], time_limits=True)
    row = rows[0]
    assert row["agent_error"] == "" and row["hook_error"] == "" and row["error"] == ""
    assert row["steps"] == 200 or row["dead"] or row["exit"]
    assert row["odom_err_max"] < 16.0 and row["odom_heading_err_max"] < 1e-3
    assert row["act_ms_mean"] < 40.0


# ---------------------------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------------------------
FIRST_ACT_SCRIPT = """
import json, sys, time
sys.path[:0] = [{submission!r}, {tests!r}, {root!r}]
import numpy as np
import agent as A
from doom_env import spaces as S
from test_fuzz import ordinary
agent = A.DoomAgent(S.observation_space(), S.action_space(), "cpu")
rng = np.random.default_rng(0)
agent.reset()
ms = []
for i in range(8 * 12 + 1):
    obs = ordinary(rng)
    t = time.perf_counter()
    agent.act(obs)
    ms.append(1000 * (time.perf_counter() - t))
print("@@" + json.dumps({{"first": ms[0], "manager": sorted(ms[8::8])[6], "failures": agent.failures}}))
"""


def test_first_act_costs_no_more_than_any_other_manager_step(capsys):
    """In a fresh interpreter: the first act() after __init__ against the median of later manager steps.

    The machine is shared, so the comparison is repeated up to three times and has a generous margin;
    training/bench/agent_latency.py reports the actual numbers.
    """
    script = FIRST_ACT_SCRIPT.format(submission=str(ROOT / "submission"), tests=str(ROOT / "training" / "tests"), root=str(ROOT))
    tries = []
    for _ in range(3):
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                              env={**os.environ, "DOOMNET_ALLOW_UNTRAINED": "1"})
        assert done.returncode == 0, done.stderr[-2000:]
        tries.append(json.loads(done.stdout.split("@@")[-1]))
        assert tries[-1]["failures"] == 0
        if tries[-1]["first"] <= max(3.0 * tries[-1]["manager"], tries[-1]["manager"] + 3.0):
            break
    with capsys.disabled():
        print(f"\n[agent] first act() {tries[-1]['first']:.3f} ms, median manager step {tries[-1]['manager']:.3f} ms")
    assert tries[-1]["first"] <= max(3.0 * tries[-1]["manager"], tries[-1]["manager"] + 3.0), tries
