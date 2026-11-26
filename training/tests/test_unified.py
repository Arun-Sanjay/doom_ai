"""Tests of the unified loop: teacher labels in the rollout workers, teacher mixing, the loss terms
that use the labels, the failure handling of the workers and the run configs.

The toy tests use the stand-in network and environments; tests that need the agent's own modules,
the teacher or the engine build their maps with the map generator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training import paths
from doomnet import spec
from training.rl import deps, standins
from training.rl.buffer import RolloutBuffer
from training.rl.ppo import PPO, PPOConfig, field_loss, heads_to_buttons, ppo_loss, sample_from_summary, unroll
from training.rl.rollout import Collector
from training.rl.vecenv import VecConfig, VecEnv

NAV_HEADS = [True, True, True, False, True, False]
CONFIGS = paths.ROOT / "training" / "configs"


def _wad(tmp_path, grid: str, name: str = "map", **options) -> str:
    from training.mapgen import udmf
    return str(udmf.write_wad(grid, tmp_path / f"{name}.wad", **options))


def _toy_setup(maps="toy:bandit", envs=8, steps=32, groups=1, seed=0, **ppo_kwargs):
    torch.manual_seed(seed)
    cfg = PPOConfig(**{"chunk": 8, "minibatch_chunks": 16, "epochs": 4, "lr": 2e-3, "gamma": 0.9, "lam": 0.9,
                       "int_coef": 0.0, **ppo_kwargs})
    vec = VecEnv(deps.load_manifest(maps), 0, envs, VecConfig(frontend="standin", seed=seed), groups)
    net = deps.make_policy("standin")
    buf = RolloutBuffer(steps, vec.n, cfg.chunk, "cpu")
    ppo = PPO(net, cfg, "cpu", total_updates=1000, seed=seed)
    return vec, net, buf, ppo, Collector(vec, net, buf, "cpu", seed, cfg.bootstrap_on_truncation)


# ---------------------------------------------------------------------------------------------
# Which maps are labelled
# ---------------------------------------------------------------------------------------------
def test_maps_with_a_start_off_the_grid_are_not_labelled(tmp_path):
    assert deps.start_on_grid(None) and deps.start_on_grid({}) and deps.start_on_grid({"light": 128})
    assert all(deps.start_on_grid({"start_angle": a}) for a in (0, 90, 180, 270, 360, -90))
    assert not any(deps.start_on_grid({"start_angle": a}) for a in (54, 1, 89.5, 135))
    assert deps.start_on_grid({"start_offset": [0, 0]}) and not deps.start_on_grid({"start_offset": [-14, 16]})
    assert not deps.start_on_grid({"start_angle": 90, "start_offset": [0, 3]})
    rows = [{"path": "a.wad", "tier": 0, "episode_ticks": 2100, "options": {}},
            {"path": "b.wad", "tier": 2, "episode_ticks": 2100, "options": {"start_angle": 180}},
            {"path": "c.wad", "tier": 2, "episode_ticks": 2100, "options": {"start_angle": 54}},
            {"path": "d.wad", "tier": 3, "episode_ticks": 2100, "options": {"start_offset": [5, 0]}},
            {"path": "e.wad", "tier": 3, "episode_ticks": 2100}]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [e.teach for e in deps.load_manifest(str(manifest))] == [True, True, False, False, True]


# ---------------------------------------------------------------------------------------------
# Teacher labels in the workers (real front end, real teacher, real engine)
# ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_teacher_labels_follow_the_executed_actions_to_the_exit(tmp_path):
    """Driving with the labels of each observation reaches the exit: the labels belong to the
    observation they are stored with and the teacher's own bookkeeping follows what was executed."""
    pytest.importorskip("training.teacher")
    grid = "#########\n#P..#...#\n#...D...#\n#...#..E#\n#########"
    entries = [deps.MapEntry(_wad(tmp_path, grid), tier=0, episode_ticks=2100)]
    with VecEnv(entries, 0, 2, VecConfig(frontend="real", teacher="real", exit_noise=False)) as vec:
        a = vec.arrays
        finished, uses, fields = [], 0, 0
        for _ in range(500):
            assert a["lab_mask"].all() and a["lab_mgr"].all()
            assert (a["lab_head_mask"] == NAV_HEADS).all()                 # attack and weapon are never labelled
            assert not a["lab_heads"][:, [spec.HEAD_ATTACK, spec.HEAD_WEAPON]].any()
            assert not (a["lab_heads"][:, spec.HEAD_USE].astype(bool) & ~a["masks"][:, 0]).any()   # no USE label the mask forbids
            manager = a["step"] % spec.MANAGER_PERIOD == 0
            assert not (a["lab_field_ok"] & ~manager).any()                # the field is stored on manager steps only
            for row in np.flatnonzero(a["lab_field_ok"]):
                steps = a["lab_field"][row]
                ti, tj = divmod(int(a["lab_target"][row]), spec.CROP_SIZE)
                assert steps[ti, tj] == 0 and steps.max() == deps.FIELD_CAP   # zero at the target, the cap where there is no route
                assert steps[spec.CROP_SIZE // 2, spec.CROP_SIZE // 2] < deps.FIELD_CAP      # the agent's own cell has a route
                fields += 1
            uses += int(a["lab_heads"][:, spec.HEAD_USE].sum())
            vec.send(0, heads_to_buttons(a["lab_heads"].astype(np.int64)))
            finished += vec.recv(0)
            if len(finished) >= 2:
                break
        assert len(finished) >= 2 and all(info["exit"] for info in finished)
        assert all(info["labelled"] == info["len"] and info["teacher_errors"] == 0 for info in finished)
        assert uses > 0 and fields > 0                                     # it opened the door on the way


@pytest.mark.slow
def test_no_controller_labels_while_an_enemy_is_visible(tmp_path):
    pytest.importorskip("training.teacher")
    entries = [deps.MapEntry(_wad(tmp_path, "########\n#P...z.#\n########"), tier=1, episode_ticks=400)]
    with VecEnv(entries, 0, 1, VecConfig(frontend="real", teacher="real")) as vec:
        a = vec.arrays
        assert a["masks"][0, 1]                                            # an enemy is in view: ATTACK is allowed
        assert not a["lab_mask"][0] and not a["lab_head_mask"][0].any()    # ... and no controller head is labelled
        assert a["lab_mgr"][0] and a["lab_subgoal"][0] == spec.SUBGOAL_INDEX["fight"] and not a["lab_field_ok"][0]


# ---------------------------------------------------------------------------------------------
# Teacher mixing in the collector
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_mixing_executes_the_teacher_with_probability_beta_and_replays_exactly(beta):
    vec, net, buf, ppo, col = _toy_setup("toy:episodic", envs=8, steps=64, groups=2)
    try:
        col.beta = beta
        col.collect()
        forced, labels = buf.forced, buf.obs["lab_heads"].long()
        assert abs(float(forced.float().mean()) - beta) < 0.08             # every toy step is labelled
        assert bool((buf.actions[forced] == labels[forced]).all())         # the teacher's action was executed there
        if beta < 1.0:
            own = ~forced                                                  # elsewhere the policy's own sample: not always the label
            assert float((buf.actions[own] == labels[own]).all(-1).float().mean()) < 0.5
        manager = buf.obs["step"] % spec.MANAGER_PERIOD == 0
        both = forced & manager                                            # the toy's manager labels are sub-goal 0, target 0
        assert bool((buf.subgoal[both] == 0).all()) and bool((buf.target[both] == 0).all())
        if beta == 0.5:
            free = ~forced & manager
            assert bool((buf.target[free] != 0).any()) and bool(both.any())
        # Manager choices change only on manager steps, forced or not.
        held = buf.subgoal[1:] == buf.subgoal[:-1]
        assert bool((held | manager[1:]).all())
        # The stored log-probability is the policy's own log-probability of what was executed.
        buf.finish(0.9, 0.9, 0.9, 0.0)
        with torch.no_grad():
            for batch in buf.minibatches(8, torch.Generator().manual_seed(0)):
                new = unroll(net, ppo.layout, batch)
                assert torch.allclose(new["logp"], batch["logp"].flatten(), atol=1e-4)
                assert torch.isfinite(batch["logp"]).all()
    finally:
        vec.close()


def test_teacher_action_the_mask_forbids_is_not_executed():
    b = 4
    tables = [torch.log_softmax(torch.zeros(b, n), -1) for n in spec.HEAD_SIZES]
    tables[spec.HEAD_USE] = torch.tensor([[0.0, float("-inf")]]).repeat(b, 1)       # USE is masked off on every row
    summary = torch.cat(tables + [torch.zeros(b, 5)], -1)
    labels = torch.tensor([[1, 2, 1, 0, 1, 0]]).repeat(b, 1)
    mask = torch.tensor([NAV_HEADS]).repeat(b, 1)
    mask[0] = False                                                        # row 0 is not forced
    heads, logp, *_ = sample_from_summary(summary, torch.Generator().manual_seed(0), override=(labels, mask))
    assert bool((heads[1:, :3] == labels[1:, :3]).all()) and bool((heads[:, spec.HEAD_USE] == 0).all())
    assert torch.isfinite(logp).all()


def test_forced_steps_leave_the_policy_term_and_stay_in_the_others():
    vec, net, buf, ppo, col = _toy_setup("toy:bandit", envs=8, steps=16, value_clip=None)
    try:
        col.beta = 0.5
        col.collect()
        buf.finish(0.9, 0.9, 0.9, 0.0, norm=ppo.norm, norm_int=ppo.norm_int)
        batch = next(buf.minibatches(buf.num_chunks(), torch.Generator().manual_seed(0)))
        forced = batch["forced"].flatten()
        assert 0 < int(forced.sum()) < forced.numel()
        with torch.no_grad():
            new = unroll(net, ppo.layout, batch)
            _, base = ppo_loss(new, batch, ppo.cfg, 1.0, distill_weight=1.0)
            changed = dict(batch, adv=torch.where(batch["forced"], batch["adv"] + 100.0, batch["adv"]),
                           logp=torch.where(batch["forced"], batch["logp"] - 50.0, batch["logp"]))
            _, other = ppo_loss(new, changed, ppo.cfg, 1.0, distill_weight=1.0)
            assert other["loss_pi"] == pytest.approx(base["loss_pi"], abs=1e-6)      # nothing of a forced step enters it
            assert other["kl"] == pytest.approx(base["kl"], abs=1e-7) and base["forced"] == pytest.approx(float(forced.float().mean()))
            moved = dict(batch, ret=torch.where(batch["forced"], batch["ret"] + 3.0, batch["ret"]))
            _, shifted = ppo_loss(new, moved, ppo.cfg, 1.0, distill_weight=1.0)
            assert shifted["loss_v"] > base["loss_v"] + 0.1                          # the value term keeps the forced steps
            everything = dict(batch, forced=torch.ones_like(batch["forced"]))
            _, none_own = ppo_loss(new, everything, ppo.cfg, 1.0, distill_weight=1.0)
            assert none_own["loss_pi"] == 0.0 and none_own["loss_distill"] == pytest.approx(base["loss_distill"])
            assert 0.0 <= base["agree"] <= 1.0 and 0.0 <= base["agree_all"] <= base["agree"]
    finally:
        vec.close()


def test_update_stops_when_unlabelled_steps_move_too_far_and_warms_the_learning_rate_up():
    """Distillation must not rewrite the policy on steps the teacher has no label for (fights)."""
    vec, net, buf, ppo, col = _toy_setup("toy:bandit", envs=8, steps=32, lr=2e-2, epochs=6, minibatch_chunks=4)
    try:
        col.collect()
        half = torch.zeros_like(buf.obs["lab_mask"])
        half[:, :4] = True                                                 # rows 4-7 carry no labels: the steps to protect
        for key in ("lab_mask", "lab_mgr"):
            buf.obs[key] &= half
        buf.obs["lab_head_mask"] &= half[..., None]
        free = ppo.update(buf, distill_weight=1.0)
        assert free["minibatches"] == 48 and free["kl_unlabelled"] > 0.01  # every epoch ran and the unlabelled steps moved
        col.collect()
        for key in ("lab_mask", "lab_mgr"):
            buf.obs[key] &= half
        buf.obs["lab_head_mask"] &= half[..., None]
        ppo.cfg.target_kl_unlabelled = 0.002
        held = ppo.update(buf, distill_weight=1.0)
        assert held["minibatches"] < 48                                    # the update ended early
        ppo.cfg.target_kl_unlabelled, ppo.cfg.lr_warmup, ppo.updates = None, 10, 0
        rates = []
        for ppo.updates in (0, 4, 9, 50):
            rates.append(ppo.lr / (ppo.cfg.lr * (1.0 + (ppo.cfg.lr_final - 1.0) * ppo.progress)))
        assert rates == pytest.approx([0.1, 0.5, 1.0, 1.0])
    finally:
        vec.close()


@pytest.mark.slow
def test_mixing_with_distillation_learns_the_labels_on_the_toy_task():
    vec, net, buf, ppo, col = _toy_setup("toy:bandit", envs=8, steps=32, lr=4e-3, distill_coef=(1, 1, 1, 1, 1, 1, 0, 0))
    try:
        col.beta = 0.7
        for _ in range(20):
            col.collect()
            stats = ppo.update(buf, distill_weight=1.0)
            if stats["agree_turn"] > 0.97:
                break
        assert stats["agree_turn"] > 0.9 and 0.5 < stats["forced"] < 0.9
    finally:
        vec.close()


# ---------------------------------------------------------------------------------------------
# The real network: forcing some rows only, and the value-field term
# ---------------------------------------------------------------------------------------------
def _inputs(b, g):
    return (torch.rand(b, spec.NUM_RAYS, spec.RAY_FEATS, generator=g), torch.rand(b, spec.MAX_ENTITIES, spec.ENT_FEATS, generator=g),
            torch.ones(b, spec.MAX_ENTITIES, dtype=torch.bool), torch.rand(b, spec.SCALAR_DIM, generator=g),
            (torch.rand(b, spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE, generator=g) < 0.3).to(torch.uint8) * 255)


def test_real_network_forces_the_manager_on_some_rows_only():
    pytest.importorskip("doomnet.model")
    torch.manual_seed(0)
    net = deps.make_policy("real", vp_iters=3)
    assert deps.enable_partial_forcing(net) and deps.enable_partial_forcing(net)      # idempotent
    b = 6
    args = _inputs(b, torch.Generator().manual_seed(1))
    run = torch.tensor([True, True, True, True, False, False])
    sub = torch.tensor([5, -1, 2, -1, 5, -1])
    tgt = torch.tensor([100, -1, 4000, -1, 100, -1])
    with torch.no_grad():
        state = net.initial_state(b)
        out, new_state = net(*args, state=state, run_manager=run, forced={"subgoal": sub, "target": tgt}, sample=True)
        assert out["subgoal"][[0, 2]].tolist() == [5, 2] and out["target"][[0, 2]].tolist() == [100, 4000]
        assert bool((out["subgoal"] >= 0).all()) and bool((out["target"] >= 0).all())
        assert out["subgoal"][4:].tolist() == [0, 0] and out["target"][4:].tolist() == [0, 0]   # no manager step: the held (initial) choice
        # The same choices forced on every row give the same pass: what conditions the controller is the choice in force.
        again, _ = net(*args, state=state, run_manager=run, forced={"subgoal": out["subgoal"], "target": out["target"]}, sample=True)
        assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(out["logits"], again["logits"]))
        assert torch.allclose(out["value_field"], again["value_field"], atol=1e-5)
        # Another forced target changes the value field of that row and of no other.
        moved = tgt.clone()
        moved[0] = 2080
        other, _ = net(*args, state=state, run_manager=run, sample=True,
                       forced={"subgoal": out["subgoal"], "target": torch.where(moved >= 0, moved, out["target"])})
        delta = (other["value_field"] - out["value_field"]).abs().flatten(1).max(1).values
        assert delta[0] > 1e-4 and bool((delta[1:] < 1e-6).all())


@pytest.mark.slow
def test_real_loop_with_mixing_replays_exactly(tmp_path):
    """Real maps, front end, teacher and network: with the teacher's action and manager choice executed
    on about half of the steps, replaying the stored chunks reproduces the stored log-probabilities."""
    pytest.importorskip("training.teacher")
    torch.manual_seed(0)
    grid = "###########\n#P..#.....#\n#...D.....#\n#...#....E#\n###########"
    entries = [deps.MapEntry(_wad(tmp_path, grid), tier=0, episode_ticks=2100)]
    cfg = PPOConfig(chunk=8)
    net = deps.make_policy("real", vp_iters=3)
    with VecEnv(entries, 0, 4, VecConfig(frontend="real", teacher="real"), groups=2) as vec:
        buf = RolloutBuffer(48, vec.n, cfg.chunk, "cpu")
        ppo = PPO(net, cfg, "cpu")
        col = Collector(vec, net, buf, "cpu", 0, False)
        assert col.partial_forcing
        col.beta = 0.5
        col.collect()
        forced, manager = buf.forced, buf.obs["step"] % spec.MANAGER_PERIOD == 0
        assert 0.3 < float(forced.float().mean()) < 0.7 and bool(buf.obs["lab_mask"].all())
        both = forced & manager
        assert bool(both.any()) and bool((~forced & manager).any())
        assert bool((buf.subgoal[both] == buf.obs["lab_subgoal"][both].long()).all())          # the teacher's choice is in force
        assert bool((buf.target[both] == buf.obs["lab_target"][both].long()).all())
        labelled = buf.obs["lab_head_mask"] & forced[..., None]
        allowed = ~((buf.obs["lab_heads"][..., spec.HEAD_USE] == 1) & ~buf.obs["masks"][..., 0])
        nav = [spec.HEAD_MOVE, spec.HEAD_STRAFE, spec.HEAD_TURN]
        assert bool((buf.actions[..., nav][labelled[..., nav]] == buf.obs["lab_heads"].long()[..., nav][labelled[..., nav]]).all())
        use = labelled[..., spec.HEAD_USE] & allowed
        assert bool((buf.actions[..., spec.HEAD_USE][use] == buf.obs["lab_heads"].long()[..., spec.HEAD_USE][use]).all())
        buf.finish(0.99, 0.95, 0.99, 0.0)
        with torch.no_grad():
            for batch in buf.minibatches(8, torch.Generator().manual_seed(0)):
                new = unroll(net, ppo.layout, batch)
                assert torch.allclose(new["logp"], batch["logp"].flatten(), atol=2e-3)
                assert torch.allclose(new["value"], batch["value"].flatten(), atol=2e-3)
                _, stats = ppo_loss(new, batch, cfg, 1.0, 1.0)
                assert abs(stats["kl"]) < 1e-4 and stats["loss_field"] > 0.0 and 0.0 < stats["field_rows"] <= 1.0


def test_field_loss_targets_and_masks():
    steps = torch.full((2, 64, 64), 255.0)
    steps[0, 10, 10:20] = torch.arange(10.0)
    steps[1, 30, 30] = 0.0
    perfect = torch.where(steps < 255, 0.95 ** steps, torch.zeros_like(steps)).unsqueeze(1)
    assert float(field_loss(perfect, steps, 0.95)) == pytest.approx(0.0, abs=1e-9)
    assert float(field_loss(perfect + 0.1, steps, 0.95)) == pytest.approx(0.02, rel=1e-3)     # 0.1 ** 2 over each group of cells
    assert float(field_loss(torch.zeros_like(perfect), steps, 0.95)) > 0.1                    # the few cells with a route count


def test_field_term_reaches_the_value_propagation_only_where_the_teacher_field_applies():
    pytest.importorskip("doomnet.model")
    torch.manual_seed(0)
    net = deps.make_policy("real", vp_iters=3)
    cfg = PPOConfig(chunk=8, field_coef=1.0)
    ppo = PPO(net, cfg, "cpu")
    B, L = 3, 8
    g = torch.Generator().manual_seed(2)
    rays, ents, ent_mask, scal, maps = _inputs(B * L, g)
    step = torch.arange(L).repeat(B, 1)                                    # one manager step per chunk, at its start
    target = torch.tensor([700, 800, 900])[:, None].repeat(1, L)
    field = torch.full((B, L, 64, 64), 255, dtype=torch.uint8)
    field[:, :, 20:30, 20:30] = 12
    batch = {"rays": rays.view(B, L, *rays.shape[1:]), "ents": ents.view(B, L, *ents.shape[1:]), "ent_mask": ent_mask.view(B, L, -1),
             "scal": scal.view(B, L, -1), "map": maps.view(B, L, *maps.shape[1:]), "masks": torch.ones(B, L, 2, dtype=torch.bool),
             "step": step, "actions": torch.zeros(B, L, 6, dtype=torch.long), "subgoal": torch.zeros(B, L, dtype=torch.long),
             "target": target, "logp": torch.zeros(B, L), "value": torch.zeros(B, L), "value_int": torch.zeros(B, L),
             "adv": torch.zeros(B, L), "ret": torch.zeros(B, L), "ret_int": torch.zeros(B, L), "forced": torch.ones(B, L, dtype=torch.bool),
             "lab_mask": torch.zeros(B, L, dtype=torch.bool), "lab_head_mask": torch.zeros(B, L, 6, dtype=torch.bool),
             "lab_heads": torch.zeros(B, L, 6, dtype=torch.int8), "lab_mgr": torch.zeros(B, L, dtype=torch.bool),
             "lab_subgoal": torch.zeros(B, L, dtype=torch.int8),
             "lab_target": torch.tensor([700, 800, 5])[:, None].repeat(1, L).to(torch.int16),    # row 2: another target in force
             "lab_field": field, "lab_field_ok": torch.tensor([True, False, True])[:, None].repeat(1, L),
             "state": ppo.layout.to_rows(net.initial_state(B))}
    new = unroll(net, ppo.layout, batch)
    _, stats = ppo_loss(new, batch, cfg, 1.0, distill_weight=1.0)
    assert stats["loss_field"] > 0.0 and stats["field_rows"] == pytest.approx(1 / 3)          # only row 0 qualifies
    solo = field_loss(new["value_field"].view(B, L, 1, 64, 64)[0, :1], field[0, :1], cfg.field_gamma)
    assert stats["loss_field"] == pytest.approx(solo.item(), rel=1e-5)
    net.zero_grad()
    solo.backward()
    assert float(net.vp.inject.weight.grad.abs().sum()) > 0.0              # the term trains the value propagation
    _, off = ppo_loss(unroll(net, ppo.layout, batch), batch, PPOConfig(chunk=8, field_coef=0.0), 1.0, distill_weight=1.0)
    assert off["loss_field"] == 0.0


# ---------------------------------------------------------------------------------------------
# Robustness of the workers
# ---------------------------------------------------------------------------------------------
def test_front_end_keeps_the_pose_when_odometry_raises():
    front = deps.FrontEnd("standin")
    obs = standins.blank_observation()
    front.reset()
    front.observe(obs, None, 0, 525.0, None)
    front.odom.x, front.odom.y, front.odom.theta = 3.0, 4.0, 0.5

    def broken(prev_action, obs):
        raise TypeError("cannot unpack non-iterable NoneType object")

    front.odom.update = broken
    feats, crop, pose = front.observe(obs, np.zeros(9, np.int8), 1, 525.0, None)
    assert pose == (3.0, 4.0, 0.5) and front.odom_errors == 1 and "NoneType" in front.last_odom_error
    assert feats["scal"].shape == (spec.SCALAR_DIM,) and crop.shape == (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE)
    front.observe(obs, np.zeros(9, np.int8), 2, 525.0, None)
    assert front.odom_errors == 2


class _Flaky(standins.ToyBandit):
    """A toy environment whose engine "dies": every call after ``fail_at`` steps raises."""

    built = 0
    fail_at = 5

    def __init__(self, ticks):
        super().__init__(ticks)
        type(self).built += 1
        self.serial = type(self).built
        self.calls = 0

    def step(self, action):
        self.calls += 1
        if self.serial == 1 and self.calls >= self.fail_at:                # the first environment built is the broken one
            raise RuntimeError("engine closed unexpectedly")
        return super().step(action)

    def reset(self, *, seed=None, options=None):
        if self.serial == 1 and self.calls >= self.fail_at:
            raise RuntimeError("engine closed unexpectedly")
        return super().reset(seed=seed, options=options)


def test_a_failing_environment_is_replaced_and_the_others_carry_on(monkeypatch):
    _Flaky.built = 0
    monkeypatch.setitem(standins.TOY_ENVS, "toy:flaky", _Flaky)
    entries = [deps.MapEntry("toy:flaky", 0, 4000, 0)]
    with VecEnv(entries, 0, 3, VecConfig(frontend="standin")) as vec:
        a, reports = vec.arrays, []
        for i in range(12):
            before = a["step"].copy()
            vec.send(0, np.zeros((3, spec.NUM_BUTTONS), np.int8))
            new = vec.recv(0)
            reports += new
            if any(r.get("crashed") for r in new):
                assert a["truncated"][0] and not a["terminated"][0] and a["reward"][0] == 0.0   # the episode was cut short
                assert a["fin_step"][0] == before[0] and a["step"][0] == 0                     # ... and a new one has begun
                assert (a["step"][1:] == before[1:] + 1).all() and not a["truncated"][1:].any()  # the other rows just stepped
        crashed = [r for r in reports if r.get("crashed")]
        assert len(crashed) == 1 and crashed[0]["row"] == 0 and "engine closed" in crashed[0]["error"]
        assert _Flaky.built == 4 and a["step"][0] > 0                      # one replacement was built and is stepping


def test_collector_and_trainer_count_environment_resets(tmp_path, monkeypatch):
    from training.rl.train import TrainConfig, apply_overrides, train
    _Flaky.built = 0
    monkeypatch.setitem(standins.TOY_ENVS, "toy:flaky", _Flaky)
    cfg = apply_overrides(TrainConfig(), {"workers": 0, "envs_per_worker": 4, "groups": 2, "rollout": 16, "policy": "standin",
                                          "ppo": {"chunk": 8, "minibatch_chunks": 4, "epochs": 1}, "vec": {"frontend": "standin"},
                                          "teach": {"beta_start": 1.0, "beta_floor": 0.5, "beta_steps": 128}})
    row = train(cfg, "toy:flaky", str(tmp_path), device_name="cpu", max_iterations=3)
    rows = list(csv.DictReader((tmp_path / "train_log.csv").open()))
    assert [r["env_resets"] for r in rows] == ["1", "1", "1"] and row["iteration"] == 3
    assert [float(r["beta"]) for r in rows] == [1.0, 0.75, 0.5]            # 64 steps per iteration, floor after 128
    assert float(rows[0]["forced"]) > 0.9 and 0.3 < float(rows[2]["forced"]) < 0.7
    assert all(r[k] != "" for r in rows for k in ("agree", "agree_all", "loss_distill_heads", "loss_distill_subgoal",
                                                  "loss_distill_target", "loss_field", "odom_errors", "teacher_errors"))


@pytest.mark.slow
def test_a_dead_engine_is_replaced(tmp_path):
    pytest.importorskip("doom_env")
    entries = [deps.MapEntry(_wad(tmp_path, "#######\n#P....#\n#.....#\n#######"), tier=0, episode_ticks=2100)]
    with VecEnv(entries, 0, 2, VecConfig(frontend="real")) as vec:
        noop = np.zeros((2, spec.NUM_BUTTONS), np.int8)
        vec.send(0, noop)
        assert vec.recv(0) == []
        vec.local.slots[1].env._game.close()                              # the engine of row 1 is gone
        vec.send(0, noop)
        reports = vec.recv(0)
        assert len(reports) == 1 and reports[0]["crashed"] and reports[0]["row"] == 1
        assert vec.arrays["step"][1] == 0 and vec.arrays["step"][0] == 2
        for _ in range(3):                                                 # the replacement steps like any other row
            vec.send(0, noop)
            assert vec.recv(0) == []
        assert vec.arrays["step"].tolist() == [5, 3]


def test_exit_detector_loader_reads_every_layout(tmp_path):
    pytest.importorskip("doomnet.model")
    from doomnet.model import ExitDetector
    torch.manual_seed(0)
    net = ExitDetector()
    state = net.state_dict()
    layouts = {"committed": {"state_dict": state, "config": {"note": "written by save_weights"}}, "older": {"model": state}, "bare": state}
    frame = np.random.default_rng(0).integers(0, 255, (spec.SCREEN_H, spec.SCREEN_W, 3), dtype=np.uint8)
    answers = []
    for name, payload in layouts.items():
        path = tmp_path / f"{name}.pt"
        torch.save(payload, path)
        answers.append(deps.make_exit_detector(str(path))(frame))
    assert all(a.shape == (3,) and np.allclose(a, answers[0]) for a in answers) and 0.0 <= answers[0][0] <= 1.0
    committed = paths.ROOT / "submission" / "weights" / "exit_detector.pt"
    if committed.is_file() and committed.stat().st_size > 1024:
        assert deps.make_exit_detector(str(committed))(frame).shape == (3,)


# ---------------------------------------------------------------------------------------------
# Schedules and configs
# ---------------------------------------------------------------------------------------------
def test_beta_schedule():
    from training.rl.train import TeachConfig
    assert TeachConfig().beta(0) == 0.0 and TeachConfig().beta(10**9) == 0.0          # the default: the policy always drives
    t = TeachConfig(beta_start=1.0, beta_floor=0.2, beta_steps=1000)
    assert t.beta(0) == 1.0 and t.beta(500) == pytest.approx(0.6) and t.beta(1000) == pytest.approx(0.2) and t.beta(10**7) == pytest.approx(0.2)


def test_distillation_weight_holds_below_the_minimum_rate():
    from training.rl.ppo import DistillSchedule
    d = DistillSchedule(min_rate=0.5, tolerance=0.06)
    # A policy that does not exit yet "holds" its rate trivially: the weight must stay where it is.
    assert [d.report(r) for r in (0.0, 0.1, 0.45)] == [1.0, 1.0, 1.0] and d.reference is None
    assert [d.report(r) for r in (0.55, 0.6)] == [0.5, 0.25]             # learned, and holding: step down
    assert [d.report(r) for r in (0.5, 0.4)] == [0.25, 0.5]              # two reports below the reference: step back up
    assert d.report(0.62) == 0.25 and len(d.history) == 8
    plain = DistillSchedule()                                            # the default has no minimum
    assert plain.report(0.0) == 0.5


def test_pinned_maps_give_every_map_its_own_row():
    entries = [deps.MapEntry("toy:bandit", tier=t, episode_ticks=64, index=i) for i, t in enumerate((0, 2, 3))]
    with VecEnv(entries, 0, 6, VecConfig(frontend="standin", pin_maps=True, episodes_per_map=1)) as vec:
        seen = {}
        for _ in range(40):                                              # 16-step episodes: two per row
            vec.send(0, np.zeros((6, spec.NUM_BUTTONS), np.int8))
            for info in vec.recv(0):
                seen.setdefault(info["row"], []).append(info["map"])
        assert seen == {row: [row % 3, row % 3] for row in range(6)}     # row r stays on map r modulo the number of maps


def test_heldout_callback_runs_and_drives_the_distillation_weight(tmp_path):
    from training.rl.train import TrainConfig, apply_overrides, train
    cfg = apply_overrides(TrainConfig(), {"workers": 0, "envs_per_worker": 4, "groups": 2, "rollout": 16, "policy": "standin",
                                          "heldout_every": 1, "heldout_envs": 4, "heldout_workers": 0,
                                          "ppo": {"chunk": 8, "minibatch_chunks": 4, "epochs": 1}, "vec": {"frontend": "standin"},
                                          "distill": {"levels": [1.0, 0.5, 0.25], "tolerance": 0.5}})
    train(cfg, "toy:bandit", str(tmp_path), device_name="cpu", heldout="toy:bandit", max_iterations=2)
    rows = list(csv.DictReader((tmp_path / "train_log.csv").open()))
    assert [r["heldout_episodes"] for r in rows] == ["4", "4"] and all(r["heldout_exit_rate"] != "" for r in rows)
    assert [float(r["distill_weight"]) for r in rows] == [0.5, 0.25]       # each report within tolerance steps the weight down
    assert len(cfg.distill.history) == 2


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.json")))
def test_run_configs_apply_cleanly(name):
    from training.rl.curriculum import Curriculum
    from training.rl.train import TrainConfig, apply_overrides
    cfg = apply_overrides(TrainConfig(), json.loads((CONFIGS / name).read_text()))     # an unknown setting raises
    stage = Curriculum(cfg.curriculum, [0, 1, 2, 3, 4]).stage
    assert isinstance(stage, str) and cfg.rollout % cfg.ppo.chunk == 0 and cfg.envs_per_worker % cfg.groups == 0
    assert len(cfg.ppo.distill_coef) == 8 and len(cfg.ppo.ent_coef) == 8
    if name.startswith("unified"):
        assert cfg.vec.teacher == "real" and cfg.ppo.bootstrap_on_truncation is False
        assert cfg.teach.beta_steps > 0 and 0.0 <= cfg.teach.beta_floor < cfg.teach.beta_start <= 1.0
        assert (CONFIGS / name).with_suffix(".txt").is_file()              # the settings are explained next to the file
        for text in (CONFIGS / name).with_suffix(".txt").read_text(), (CONFIGS / "LAUNCH.txt").read_text():
            assert "atan" + "2" not in text and "arctan" + "2" not in text


# ---------------------------------------------------------------------------------------------
# Export into the agent
# ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_exported_weights_start_a_real_agent_and_play_an_episode(tmp_path, monkeypatch):
    pytest.importorskip("doom_env")
    from training.rl.curriculum import Curriculum
    from training.rl.export import export
    from training.rl.train import TrainConfig, save_checkpoint
    if not (paths.ROOT / "submission" / "weights" / "exit_detector.pt").is_file():
        pytest.skip("no exit detector committed")
    monkeypatch.setenv("DOOMNET_ALLOW_UNTRAINED", "1")                     # export must not rely on it, and must restore it
    threads = torch.get_num_threads()
    try:
        torch.manual_seed(0)
        net = deps.make_policy("real", vp_iters=4)                         # not the default: the agent must rebuild this one
        cfg = TrainConfig()
        ckpt = tmp_path / "ckpt.pt"
        save_checkpoint(ckpt, net, PPO(net, cfg.ppo, "cpu"), Curriculum(cfg.curriculum, [0]), cfg, iteration=3, env_steps=99)
        wad = _wad(tmp_path, "#######\n#P...E#\n#######")
        out = export(str(ckpt), tmp_path / "weights" / "policy.pt", episode_wad=wad)
        played = out["agent"]
        assert played["policy_kwargs"]["vp_iters"] == 4 and played["failures"] == 0 and played["warmup_failures"] == 0
        assert played["steps"] > 0 and played["steps_with_buttons"] > 0
        import os
        assert os.environ.get("DOOMNET_ALLOW_UNTRAINED") == "1"
        import agent
        assert agent.WEIGHTS_DIR == paths.ROOT / "submission" / "weights" and not list(Path(tmp_path / "weights").glob("*.tmp"))
    finally:
        torch.set_num_threads(threads)
