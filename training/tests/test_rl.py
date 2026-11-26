"""Tests of the reinforcement-learning core (training/rl).

Stand-in tests use a small network with the policy's forward signature, toy environments with the
doom_env interface and stub odometry / memory / encoder, so they do not depend on other areas.
Tests marked "real" use the agent's own modules and are skipped while those are missing.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import deps, standins
from training.rl.buffer import RolloutBuffer, RunningNorm, StateLayout, compute_gae
from training.rl.curriculum import Curriculum, CurriculumConfig
from training.rl.intrinsic import IntrinsicCoef, IntrinsicReward
from training.rl.ppo import (PPO, DistillSchedule, PPOConfig, apply_masks, entropy, evaluate, heads_to_buttons,
                             log_softmax, ppo_loss, sample_from_summary, step_summary, unroll)
from training.rl.reward import TERMS, RewardCoef, RewardTracker, SectorIndex
from training.rl.rollout import Collector, heldout_exit_rate
from training.rl.vecenv import OBS_FIELDS, VecConfig, VecEnv

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
NOOP = np.zeros(9, np.int8)
TERM = {name: i for i, name in enumerate(TERMS)}


def make_setup(maps="toy:bandit", envs=8, steps=32, groups=1, workers=0, policy="standin", frontend="standin",
               seed=0, **ppo_kwargs):
    """Vector environment + policy + buffer + learner + collector on the CPU."""
    torch.manual_seed(seed)
    cfg = PPOConfig(**{"chunk": 8, "minibatch_chunks": 16, "epochs": 4, "lr": 2e-3, "gamma": 0.9, "lam": 0.9,
                       "int_coef": 0.0, **ppo_kwargs})
    entries = maps if isinstance(maps, list) else deps.load_manifest(maps)
    vec = VecEnv(entries, workers, envs, VecConfig(frontend=frontend, seed=seed), groups)
    net = deps.make_policy(policy)
    buf = RolloutBuffer(steps, vec.n, cfg.chunk, "cpu")
    ppo = PPO(net, cfg, "cpu", total_updates=1000, seed=seed)
    return vec, net, buf, ppo, Collector(vec, net, buf, "cpu", seed, cfg.bootstrap_on_truncation)


# ---------------------------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------------------------
def test_gae_matches_hand_calculation():
    """Three environments over three steps: no episode end, a termination and a truncation at t = 1."""
    gamma, lam = 0.9, 0.5
    r = torch.tensor([[1.0, 2.0, 3.0]] * 3).T                    # (T=3, N=3)
    v = torch.tensor([[0.5, 1.0, 1.5]] * 3).T
    last = torch.tensor([2.0, 2.0, 2.0])
    term = torch.zeros(3, 3, dtype=torch.bool)
    trunc = torch.zeros(3, 3, dtype=torch.bool)
    boot = torch.zeros(3, 3)
    term[1, 1] = True                                             # env 1: exit or death after step 1
    trunc[1, 2], boot[1, 2] = True, 4.0                           # env 2: tick cap after step 1, V(final obs) = 4
    adv, ret = compute_gae(r, v, last, term, trunc, boot, gamma, lam)
    # Step 2 is the same everywhere: delta = 3 + 0.9 * 2.0 - 1.5 = 3.3.
    # env 0: A1 = (2 + 0.9 * 1.5 - 1.0) + 0.45 * 3.3 = 3.835;  A0 = (1 + 0.9 * 1.0 - 0.5) + 0.45 * 3.835 = 3.12575
    # env 1: A1 = 2 + 0 - 1.0 = 1.0 (no bootstrap, no carry);  A0 = 1.4 + 0.45 * 1.0 = 1.85
    # env 2: A1 = 2 + 0.9 * 4.0 - 1.0 = 4.6 (bootstrap, no carry);  A0 = 1.4 + 0.45 * 4.6 = 3.47
    expected = torch.tensor([[3.12575, 1.85, 3.47], [3.835, 1.0, 4.6], [3.3, 3.3, 3.3]])
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(ret, expected + v, atol=1e-6)
    # Non-episodic stream: episode ends are ignored, so every environment looks like env 0.
    adv_ne, _ = compute_gae(r, v, last, term, trunc, boot, gamma, lam, episodic=False)
    assert torch.allclose(adv_ne, expected[:, :1].expand(3, 3), atol=1e-6)


def test_truncation_as_terminal_and_two_streams():
    """finish(): the tick cap can be made terminal, and the two streams combine with int_coef."""
    buf = RolloutBuffer(2, 1, 2, "cpu")
    buf.reward[:, 0] = torch.tensor([1.0, 1.0])
    buf.reward_int[:, 0] = torch.tensor([0.5, 0.0])
    buf.truncated[1, 0], buf.bootstrap[1, 0] = True, 10.0
    buf.finish(1.0, 1.0, 1.0, int_coef=2.0, normalise=False)
    assert torch.allclose(buf.ret[:, 0], torch.tensor([12.0, 11.0]))             # bootstrapped at the cap
    assert torch.allclose(buf.adv[:, 0], torch.tensor([12.0 + 2.0 * 0.5, 11.0]))
    buf.finish(1.0, 1.0, 1.0, int_coef=2.0, normalise=False, bootstrap_on_truncation=False)
    assert torch.allclose(buf.ret[:, 0], torch.tensor([2.0, 1.0]))               # cap treated as terminal


def test_running_norm_round_trip():
    norm = RunningNorm()
    x = torch.tensor([1.0, 3.0, 5.0, 7.0])
    norm.update(x)
    assert norm.mean == pytest.approx(4.0) and norm.std == pytest.approx(float(x.std(unbiased=False)))
    assert torch.allclose(norm.denormalise(norm.normalise(x)), x, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Masked, factored log-probability
# ---------------------------------------------------------------------------------------------
def _fake_output(batch, generator):
    logits = [torch.randn(batch, n, generator=generator, requires_grad=True) for n in spec.HEAD_SIZES]
    return {"logits": logits,
            "subgoal_logits": torch.randn(batch, spec.NUM_SUBGOALS, generator=generator),
            "target_logits": torch.randn(batch, spec.CROP_SIZE ** 2, generator=generator),
            "subgoal": torch.randint(0, spec.NUM_SUBGOALS, (batch,), generator=generator),
            "target": torch.randint(0, spec.CROP_SIZE ** 2, (batch,), generator=generator),
            "value": torch.randn(batch, generator=generator), "value_int": torch.randn(batch, generator=generator)}


def sample_heads(logits, generator):
    """One sample per head from (masked) logits, (B, 6)."""
    return torch.stack([torch.multinomial(log_softmax(lg).exp(), 1, generator=generator).squeeze(-1) for lg in logits], -1)


def test_masked_log_prob():
    g = torch.Generator().manual_seed(0)
    out = _fake_output(4, g)
    masks = torch.tensor([[False, True], [True, False], [False, False], [True, True]])     # (use_ok, attack_ok)
    run_manager = torch.tensor([True, False, True, False])

    masked = apply_masks(out["logits"], masks)
    probs = [log_softmax(lg).exp() for lg in masked]
    assert probs[spec.HEAD_USE][0, 1] == 0 and probs[spec.HEAD_USE][2, 1] == 0            # exactly zero
    assert probs[spec.HEAD_ATTACK][1, 1] == 0 and probs[spec.HEAD_ATTACK][2, 1] == 0
    assert probs[spec.HEAD_USE][1, 1] > 0 and probs[spec.HEAD_ATTACK][0, 1] > 0           # allowed rows untouched
    assert all(torch.equal(a, b) for a, b in zip(apply_masks(masked, masks), masked))       # idempotent

    # A masked option is never sampled.
    many = sample_heads([lg.detach().repeat(500, 1) for lg in masked], g).view(500, 4, 6)
    assert many[:, 0, spec.HEAD_USE].sum() == 0 and many[:, 2, spec.HEAD_ATTACK].sum() == 0
    assert many[:, 3, spec.HEAD_USE].sum() > 0

    actions = sample_heads([lg.detach() for lg in masked], g)
    ev = evaluate(out, masks, actions, run_manager)

    # The collector's path: one packed summary per forward pass, sampled on the CPU.
    with torch.no_grad():
        summary = step_summary(out, masks, run_manager)
    assert summary.shape == (4, sum(spec.HEAD_SIZES) + 5)
    heads, logp, value, value_int, subgoal, target = sample_from_summary(summary, g)
    assert torch.equal(subgoal, out["subgoal"]) and torch.equal(target, out["target"])
    assert torch.equal(value, out["value"]) and torch.equal(value_int, out["value_int"])
    assert torch.allclose(logp, evaluate(out, masks, heads, run_manager)["logp"], atol=1e-5)
    assert heads[0, spec.HEAD_USE] == 0 and heads[2, spec.HEAD_ATTACK] == 0 and heads[2, spec.HEAD_USE] == 0
    greedy = sample_from_summary(summary, g, greedy_heads=(spec.HEAD_MOVE,))[0]
    assert torch.equal(greedy[:, spec.HEAD_MOVE], out["logits"][spec.HEAD_MOVE].argmax(-1))
    # Joint log-prob by hand: six heads, plus the manager's two choices on manager steps only.
    by_hand = torch.zeros(4)
    for h, lg in enumerate(out["logits"]):
        lg = lg.detach().clone()
        if h == spec.HEAD_USE:
            lg[~masks[:, 0], 1] = -math.inf
        if h == spec.HEAD_ATTACK:
            lg[~masks[:, 1], 1] = -math.inf
        by_hand += torch.log_softmax(lg, -1)[torch.arange(4), actions[:, h]]
    manager = (torch.log_softmax(out["subgoal_logits"], -1)[torch.arange(4), out["subgoal"]]
               + torch.log_softmax(out["target_logits"], -1)[torch.arange(4), out["target"]])
    by_hand += torch.where(run_manager, manager, torch.zeros(4))
    assert torch.allclose(ev["logp"], by_hand, atol=1e-5)
    assert torch.isfinite(ev["logp"]).all()

    # A head with one option left has zero entropy and contributes log 1 = 0; gradients stay finite.
    assert ev["entropy"][0, spec.HEAD_USE] == 0 and ev["entropy"][2, spec.HEAD_ATTACK] == 0
    assert ev["entropy"][3, spec.HEAD_USE] > 0
    (ev["logp"].sum() + ev["entropy"].sum()).backward()
    assert all(torch.isfinite(lg.grad).all() for lg in out["logits"])
    assert out["logits"][spec.HEAD_USE].grad[0].abs().sum() == 0                            # nothing to learn there


def test_entropy_matches_definition():
    lp = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0], [0.0, -math.inf, 0.0]]), -1)
    assert entropy(lp)[1] == pytest.approx(math.log(2.0))
    p = lp[0].exp()
    assert entropy(lp)[0] == pytest.approx(float(-(p * p.log()).sum()))


def test_heads_to_buttons_matches_spec():
    rng = np.random.default_rng(0)
    heads = np.stack([rng.integers(0, n, 64) for n in spec.HEAD_SIZES], -1)
    buttons = heads_to_buttons(heads)
    assert buttons.dtype == np.int8 and buttons.shape == (64, 9)
    for h, b in zip(heads, buttons):
        assert np.array_equal(b, spec.heads_to_buttons(h))


# ---------------------------------------------------------------------------------------------
# Rollout, buffer and loss on the stand-ins
# ---------------------------------------------------------------------------------------------
def test_rollout_replay_is_exact():
    """Replaying stored chunks with unchanged weights reproduces the rollout's log-probs and values.

    This covers the stored initial states, state resets at episode starts inside a chunk, the
    forced manager choices and the masks: any mismatch would make the first PPO ratio differ from 1.
    """
    vec, net, buf, ppo, col = make_setup("toy:episodic", envs=6, steps=32, groups=2)
    try:
        col.collect()
        col.collect()                                   # second rollout starts mid-episode
        assert bool(buf.terminated.any()) and bool((buf.obs["step"][1:] == 0).any())
        buf.finish(0.9, 0.9, 0.9, 0.0)
        with torch.no_grad():
            for batch in buf.minibatches(8, torch.Generator().manual_seed(0)):
                new = unroll(net, ppo.layout, batch)
                assert torch.allclose(new["logp"], batch["logp"].flatten(), atol=1e-4)
                assert torch.allclose(new["value"], batch["value"].flatten(), atol=1e-4)
                assert torch.allclose(new["value_int"], batch["value_int"].flatten(), atol=1e-4)
        # Manager choices are held between manager steps and may change only on them (or at a reset).
        held = buf.subgoal[1:] == buf.subgoal[:-1]
        manager_step = buf.obs["step"][1:] % spec.MANAGER_PERIOD == 0
        assert bool((held | manager_step).all())
    finally:
        vec.close()


def test_truncation_stores_bootstrap_value():
    vec, net, buf, ppo, col = make_setup("toy:bandit", envs=4, steps=40)       # 32-step episodes, cap only
    try:
        col.collect()
        assert bool(buf.truncated[31].all()) and not bool(buf.terminated.any())
        assert bool((buf.bootstrap[31] != 0).all()) and bool((buf.bootstrap[:31] == 0).all())
        assert bool((buf.obs["step"][32] == 0).all())                         # the next stored step starts an episode
    finally:
        vec.close()


def test_loss_decreases_on_fixed_batch():
    vec, net, buf, ppo, col = make_setup("toy:bandit", envs=8, steps=16, value_clip=None)
    try:
        col.collect()
        buf.finish(0.9, 0.9, 0.9, 0.0, norm=ppo.norm, norm_int=ppo.norm_int)
        batch = next(buf.minibatches(buf.num_chunks(), torch.Generator().manual_seed(0)))
        losses, distill = [], []
        for _ in range(25):
            loss, stats = ppo_loss(unroll(net, ppo.layout, batch), batch, ppo.cfg, 1.0, distill_weight=1.0)
            ppo.opt.zero_grad()
            loss.backward()
            ppo.opt.step()
            losses.append(loss.item())
            distill.append(stats["loss_distill"])
        assert losses[-1] < losses[0] - 0.05
        # The toy environment supplies labels for every step: action heads, sub-goal and target cell.
        # At the start the term is about 6 small cross-entropies + ln 8 + 0.25 * ln 4096.
        assert stats["labelled"] == 1.0 and distill[0] > 5.0 and distill[-1] < 0.5 * distill[0]
        # Without labels (the state of things until a teacher exists) the term is exactly zero.
        batch["lab_mask"] = torch.zeros_like(batch["lab_mask"])
        _, stats = ppo_loss(unroll(net, ppo.layout, batch), batch, ppo.cfg, 1.0, distill_weight=1.0)
        assert stats["loss_distill"] == 0.0 and stats["labelled"] == 0.0
    finally:
        vec.close()


def test_standin_learns_two_armed_bandit():
    """Pure RL (no labels) on the fixed two-armed bandit: the paying arm is chosen almost always."""
    vec, net, buf, ppo, col = make_setup("toy:fixed", envs=8, steps=32)
    try:
        rewards = []
        for _ in range(8):
            col.collect()
            rewards.append(float(buf.reward.mean()))
            ppo.update(buf, distill_weight=0.0)
        assert rewards[0] < 0.6                    # a fresh policy picks the paying arm about one time in three
        assert rewards[-1] > 0.9
    finally:
        vec.close()


@pytest.mark.slow
@pytest.mark.parametrize("task,threshold", [("toy:bandit", 0.9), ("toy:recall", 0.85)])
def test_standin_learns_contextual_and_memory_tasks(task, threshold):
    """The arm to pull is shown in the observation every step (bandit) or only at step 0 (recall).

    The second needs the recurrent state to carry the cue across chunk boundaries.
    """
    vec, net, buf, ppo, col = make_setup(task, envs=16, steps=32, groups=2)
    try:
        best = 0.0
        for _ in range(40):
            col.collect()
            best = max(best, float(buf.reward.mean()))
            ppo.update(buf, distill_weight=0.0)
            if best > threshold:
                break
        assert best > threshold
    finally:
        vec.close()


@pytest.mark.slow
def test_distillation_alone_learns_the_labels():
    """With rewards switched off the policy still learns the task from the teacher labels.

    Advantages are left unnormalised here: with no reward they are noise around zero, and
    normalising them would blow that noise up to unit scale.
    """
    vec, net, buf, ppo, col = make_setup("toy:bandit", envs=8, steps=32, normalise_adv=False, lr=4e-3,
                                         distill_coef=(1, 1, 1, 1, 1, 1, 0, 0))       # the six action heads
    try:
        for _ in range(16):
            col.collect()
            match = float((buf.actions[..., spec.HEAD_TURN] == buf.obs["lab_heads"][..., spec.HEAD_TURN]).float().mean())
            buf.reward.zero_()
            ppo.update(buf, distill_weight=1.0)
            if match > 0.95:
                break
        assert match > 0.9
    finally:
        vec.close()


def test_state_layout_finds_batch_axis():
    class Odd(torch.nn.Module):
        def initial_state(self, batch, device):
            return {"h": torch.ones(1, batch, 5), "goal": torch.full((batch,), 7, dtype=torch.long)}

    layout = StateLayout(Odd(), "cpu")
    assert layout.axis == {"h": 1, "goal": 0}
    init = Odd().initial_state(3, "cpu")
    state = {"h": torch.zeros(1, 3, 5), "goal": torch.zeros(3, dtype=torch.long)}
    out = layout.reset(state, init, torch.tensor([True, False, True]))
    assert out["h"][0, :, 0].tolist() == [1.0, 0.0, 1.0] and out["goal"].tolist() == [7, 0, 7]
    rows = layout.to_rows(out)
    assert rows["h"].shape == (3, 1, 5)
    assert torch.equal(layout.from_rows(rows)["h"], out["h"])
    assert layout.take(out, torch.tensor([2]))["h"].shape == (1, 1, 5)


# ---------------------------------------------------------------------------------------------
# Distillation schedule
# ---------------------------------------------------------------------------------------------
def test_distill_schedule_steps_down_only_while_exit_rate_holds():
    s = DistillSchedule(levels=(1.0, 0.5, 0.25, 0.0), tolerance=0.03, patience=2)
    assert s.weight == 1.0
    assert s.report(None) == 1.0                    # no measurement: hold
    assert s.report(0.60) == 0.5                    # first measurement becomes the reference
    assert s.report(0.58) == 0.25                   # within tolerance of 0.60: step down, reference 0.58
    assert s.report(0.50) == 0.25                   # fell by more than the tolerance: hold
    assert s.report(0.51) == 0.5                    # still down after `patience` reports: step back up
    assert s.report(0.54) == 0.5                    # reference is still 0.58: not recovered, hold
    assert s.report(0.57) == 0.25                   # recovered: step down again
    assert s.report(0.60) == 0.0
    assert s.report(0.70) == 0.0                    # stays at the last level
    restored = DistillSchedule(levels=(1.0, 0.5, 0.25, 0.0))
    restored.load_state_dict(s.state_dict())
    assert restored.weight == 0.0 and restored.reference == pytest.approx(0.70)


# ---------------------------------------------------------------------------------------------
# Intrinsic reward
# ---------------------------------------------------------------------------------------------
def _crop_with(cells):
    crop = np.zeros((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.float32)
    for ch, i, j in cells:
        crop[ch, i, j] = 1.0
    return crop


def test_intrinsic_zero_when_nothing_new():
    coef = IntrinsicCoef(new_cell=0.01, visit=0.02)
    cur = IntrinsicReward(coef)
    crop = _crop_with([(spec.CH_FREE, 32, 32), (spec.CH_FREE, 33, 32), (spec.CH_WALL, 34, 32)])
    assert cur.step(crop, (0.0, 0.0, 0.0))[0] == 0.0             # first observation seeds the memory
    for _ in range(5):                                            # same cells, same place: exactly zero
        assert cur.step(crop, (5.0, -3.0, 0.3)) == (0.0, 0.0, 0.0)
    # Two more cells observed: paid once, then zero again.
    more = _crop_with([(spec.CH_FREE, 32, 32), (spec.CH_FREE, 33, 32), (spec.CH_WALL, 34, 32),
                       (spec.CH_FREE, 32, 33), (spec.CH_WALL, 32, 34)])
    assert cur.step(more, (0.0, 0.0, 0.0))[0] == pytest.approx(0.02)
    assert cur.step(more, (0.0, 0.0, 0.0))[0] == 0.0
    # Other channels (items, recency, ...) are not "newly observed space".
    other = more.copy()
    other[spec.CH_ITEM, 10, 10] = other[spec.CH_RECENCY, 11, 11] = 1.0
    assert cur.step(other, (0.0, 0.0, 0.0))[0] == 0.0


def test_intrinsic_follows_the_moving_crop_and_counts_visits():
    coef = IntrinsicCoef(new_cell=0.01, visit=0.02)
    cur = IntrinsicReward(coef)
    # The same three world cells seen from the start cell and from one cell further along +x:
    # the crop is centred on the agent, so the pattern shifts by one index and nothing is new.
    at_start = _crop_with([(spec.CH_FREE, 32, 32), (spec.CH_FREE, 33, 32), (spec.CH_WALL, 34, 32)])
    one_ahead = _crop_with([(spec.CH_FREE, 31, 32), (spec.CH_FREE, 32, 32), (spec.CH_WALL, 33, 32)])
    cur.step(at_start, (0.0, 0.0, 0.0))
    total, new, visit = cur.step(one_ahead, (64.0, 0.0, 0.0))
    assert new == 0.0 and visit == pytest.approx(0.02)            # first entry into the new cell: 1/sqrt(1)
    assert cur.step(one_ahead, (70.0, 4.0, 0.0))[0] == 0.0        # staying in it earns nothing
    assert cur.step(at_start, (0.0, 0.0, 0.0))[2] == pytest.approx(0.02 / math.sqrt(2))   # second entry of the start cell
    assert cur.step(one_ahead, (64.0, 0.0, 0.0))[2] == pytest.approx(0.02 / math.sqrt(2))
    # Far outside the stored map: no error, no reward for cells that cannot be stored.
    assert cur.step(at_start, (1e6, -1e6, 0.0))[1] == 0.0
    cur.reset()
    assert cur.step(at_start, (0.0, 0.0, 0.0))[0] == 0.0 and cur.n_seen == 3
    off = IntrinsicReward(IntrinsicCoef(enabled=False))
    off.step(at_start, (0.0, 0.0, 0.0))
    assert off.step(one_ahead, (64.0, 0.0, 0.0))[0] == 0.0


def test_stub_memory_crop_is_centred_on_the_agent():
    """The crop layout the intrinsic reward assumes: [channel, x, y], agent cell at index 32, +x = higher index."""
    mem = standins.StubMemoryMap()
    obs = standins.blank_observation()
    obs["rays"]["distance"][:] = 0.05
    obs["rays"]["distance"][0] = 150.0 / spec.RAY_RANGE          # a wall 150 units straight ahead (+x)
    obs["rays"]["hit_type"][0] = spec.HIT_WALL
    mem.update((0.0, 0.0, 0.0), obs)
    crop = mem.crop((0.0, 0.0, 0.0))
    assert crop.shape == (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE) and crop.dtype == np.float32
    assert crop[spec.CH_VISITED, 32, 32] == 1.0
    assert crop[spec.CH_WALL, 34, 32] == 1.0                      # 151 units ahead is crop cell +2 in x
    moved = mem.crop((64.0, 0.0, 0.0))
    assert moved[spec.CH_WALL, 33, 32] == 1.0
    assert mem.crop((10_000.0, -10_000.0, 0.0)).sum() == 0.0      # far outside: zeros, no error


# ---------------------------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------------------------
def test_curriculum_stages_and_adaptive_sampling():
    cur = Curriculum(CurriculumConfig(min_episodes=10, window=50), available_tiers=[0, 1, 2, 3, 4])
    assert cur.stage == "nav" and cur.tier_probs().tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    for _ in range(9):
        cur.record({"tier": 0, "has_exit": True, "exit": True})
    assert not cur.advance(1000)                                  # not enough episodes yet
    cur.record({"tier": 0, "has_exit": True, "exit": True})
    assert cur.advance(2000) and cur.stage == "arena"
    assert cur.tier_probs()[1] == pytest.approx(0.8)
    assert not cur.advance(3000)
    assert cur.advance(2000 + 6_000_000) and cur.stage == "keys_doors"       # step budget ends a stage too
    for _ in range(20):                                           # arenas without an exit: success = all kills, alive
        cur.record({"tier": 1, "has_exit": False, "kills": 3, "monsters": 3, "dead": False})
    assert cur.success_rate((1,)) == (1.0, 20) and cur.exit_rate(1) is None
    cur.stage_index = len(cur.cfg.stages)
    assert cur.stage == "adaptive"
    for tier, rate in ((1, 0.9), (2, 0.6), (3, 0.2), (4, 0.5)):
        for k in range(20):
            cur.record({"tier": tier, "has_exit": True, "exit": k < rate * 20})
    p = cur.tier_probs()
    assert p[0] == 0.0 and p.sum() == pytest.approx(1.0)
    assert p[3] > p[4] > p[2] > p[1]                              # lowest exit rate sampled most
    restored = Curriculum(CurriculumConfig(min_episodes=10, window=50), available_tiers=[0, 1, 2, 3, 4])
    restored.load_state_dict(cur.state_dict())
    assert restored.stage == "adaptive" and np.allclose(restored.tier_probs(), p)


def test_curriculum_skips_tiers_without_maps():
    cur = Curriculum(available_tiers=[2])
    assert cur.stage == "keys_doors"
    assert cur.tier_probs().tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert Curriculum(available_tiers=[7]).tier_probs().sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# Reward and exit oracle on real maps
# ---------------------------------------------------------------------------------------------
def _reference_map(grid):
    from fixtures import fixture_wad
    return fixture_wad(grid)


def _tracker(env, wad, **coef):
    facts = deps.map_facts(wad)
    return RewardTracker(RewardCoef(**coef), facts.n_keys, facts.n_monsters, len(facts.exit_centres) > 0, facts.known), facts


def test_sector_index_even_odd():
    square = np.array([[0, 0, 4, 0], [4, 0, 4, 4], [4, 4, 0, 4], [0, 4, 0, 0]], float)
    hole = np.array([[1, 1, 2, 1], [2, 1, 2, 2], [2, 2, 1, 2], [1, 2, 1, 1]], float)
    index = SectorIndex([(10, np.concatenate([square, hole])), (11, hole)])
    assert index.locate(3.0, 3.0) == 0 and index.locate(1.5, 1.5) == 1 and index.locate(9.0, 9.0) == -1


def test_reward_walk_through_door_to_exit():
    """Door corridor: the return adds up to the composite score (x 10) of a clean run."""
    from doom_env import make_env
    wad = _reference_map("##########\n#P..D...E#\n##########")
    env = make_env(wad, episode_ticks=1000, render_buffers=False)
    try:
        tracker, facts = _tracker(env, wad)
        assert facts.n_monsters == 0 and facts.n_keys == 0 and len(facts.exit_centres) == 1
        obs, _ = env.reset(seed=0)
        tracker.reset(env, env.privileged_state())
        total, terms, pressed, steps = 0.0, np.zeros(len(TERMS)), False, 0
        for _ in range(240):
            near_door = obs["rays"]["hit_type"][0] == spec.HIT_DOOR and obs["rays"]["distance"][0] * 1024 < 60
            action = USE if (near_door and not pressed) else FWD
            pressed = pressed or bool(near_door)
            obs, _, term, trunc, _ = env.step(action)
            r, t = tracker.step(env.privileged_state(), action)
            total += r
            terms += t
            steps += 1
            if term or trunc:
                break
        priv = env.privileged_state()
        assert priv["exit_reached"] and 20 < steps < 200
        tic = steps * spec.FRAME_SKIP        # the engine's own clock reads 0 after an exit, so tics are counted
        # No monsters: combat is dropped, the other weights are rescaled by 1 / 0.85.
        k = 10.0 / 0.85
        assert terms[TERM["exit"]] == pytest.approx(0.45 * k)
        assert terms[TERM["door"]] == pytest.approx(0.20 * k)                  # the only progression part, one door
        assert terms[TERM["exit_health"]] == pytest.approx(0.05 * k * 0.5)
        assert terms[TERM["exit_time"]] == pytest.approx(0.05 * k * 0.5 * (1 - tic / 1000), abs=1e-6)
        assert 0.0 < terms[TERM["exit_time"]] < 0.05 * k * 0.5 * 0.95         # some of the time bonus was spent
        n = tracker.sectors.n
        assert tracker.summary()["sectors"] == n and terms[TERM["sector"]] == pytest.approx(0.15 * k * (n - 1) / n)
        assert terms[TERM["reversal"]] == 0 and terms[TERM["kill"]] == 0 and terms[TERM["death"]] == 0
        assert total == pytest.approx(terms.sum())
        assert total == pytest.approx(10.0 - 0.15 * k / n - 0.05 * k * 0.5 * tic / 1000, abs=1e-6)
    finally:
        env.close()


def test_reward_counts_one_door_reversal():
    from doom_env import make_env
    wad = _reference_map("##########\n#P..D...E#\n##########")
    env = make_env(wad, episode_ticks=2000, render_buffers=False)
    try:
        tracker, _ = _tracker(env, wad)
        obs, _ = env.reset(seed=0)
        tracker.reset(env, env.privileged_state())
        while not (obs["rays"]["hit_type"][0] == spec.HIT_DOOR and obs["rays"]["distance"][0] * 1024 < 60):
            obs, *_ = env.step(FWD)
            tracker.step(env.privileged_state(), FWD)
        reversal = 0.0
        for action in [USE, NOOP, NOOP, USE] + [NOOP] * 30:      # second press while the door is rising
            env.step(action)
            reversal += tracker.step(env.privileged_state(), action)[1][TERM["reversal"]]
        assert reversal == pytest.approx(-RewardCoef().reversal_cost)
        assert tracker.summary()["reversals"] == 1
    finally:
        env.close()


def test_reward_key_locked_door_and_death_terms():
    from doom_env import make_env
    wad = _reference_map("############\n#P.r..R...E#\n############")
    env = make_env(wad, episode_ticks=1000, render_buffers=False)
    try:
        tracker, facts = _tracker(env, wad)
        assert facts.n_keys == 1
        obs, _ = env.reset(seed=0)
        tracker.reset(env, env.privileged_state())
        terms, last_x = np.zeros(len(TERMS)), None
        for i in range(240):
            x = env.privileged_state()["x"]
            action = USE if (last_x is not None and abs(x - last_x) < 0.5 and i % 6 == 0) else FWD
            last_x = x
            _, _, term, trunc, _ = env.step(action)
            terms += tracker.step(env.privileged_state(), action)[1]
            if term or trunc:
                break
        k = 10.0 / 0.85
        assert terms[TERM["key"]] == pytest.approx(0.20 * k / 2)          # two progression parts: key, locked door
        assert terms[TERM["locked_door"]] == pytest.approx(0.20 * k / 2)
        assert terms[TERM["door"]] == 0 and terms[TERM["exit"]] == pytest.approx(0.45 * k)
    finally:
        env.close()

    wad = _reference_map("##########\n#P......c#\n##########")                # stand still in front of a chaingunner
    env = make_env(wad, episode_ticks=2000, render_buffers=False)
    try:
        tracker, facts = _tracker(env, wad)
        assert facts.n_monsters == 1 and len(facts.exit_centres) == 0
        env.reset(seed=0)
        tracker.reset(env, env.privileged_state())
        terms, steps = np.zeros(len(TERMS)), 0
        for _ in range(500):
            _, _, term, trunc, _ = env.step(NOOP)
            priv = env.privileged_state()
            terms += tracker.step(priv, NOOP)[1]
            steps += 1
            if term or trunc:
                break
        assert priv["dead"] and abs(priv["tic"] - steps * spec.FRAME_SKIP) <= 4
        k = 10.0 / 0.30                                                    # only exploration and combat are present
        assert terms[TERM["death"]] == pytest.approx(-0.15 * k * 0.3 * (1 - steps * spec.FRAME_SKIP / 2000), abs=1e-6)
        assert terms[TERM["damage"]] == pytest.approx(-0.15 * k * 0.2 * min(1.0, priv["damage_taken"] / 100), abs=1e-6)
        assert terms[TERM["exit"]] == 0 and terms[TERM["kill"]] == 0
    finally:
        env.close()


def test_reward_combat_terms_use_the_engine_hit_counter():
    """Shooting a zombieman: kill reward once; a shot costs ammo only when it hits nothing."""
    from doom_env import make_env
    wad = _reference_map("#####\n#P.z#\n#####")
    env = make_env(wad, episode_ticks=2000, render_buffers=False)
    attack = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], np.int8)
    try:
        tracker, facts = _tracker(env, wad)
        assert facts.n_monsters == 1
        obs, _ = env.reset(seed=3)

        def priv_now():
            priv = env.privileged_state()
            priv["hits"] = deps.hit_count(env)
            return priv

        def ammo(o):
            return float(o["vitals"][4:8].sum())

        tracker.reset(env, priv_now(), ammo(obs))
        terms, shots, hits, last_ammo = np.zeros(len(TERMS)), 0, 0, ammo(obs)
        misses_after_kill = 0
        for step in range(120):
            was_alive = env.privileged_state()["kills"] == 0
            obs, _, term, trunc, _ = env.step(attack)
            priv = priv_now()
            before = tracker.hits
            terms += tracker.step(priv, attack, ammo(obs))[1]
            fired = ammo(obs) < last_ammo
            last_ammo = ammo(obs)
            shots += fired
            hit_now = priv["hits"] > before
            hits += hit_now
            if fired and not hit_now and not was_alive:
                misses_after_kill += 1
            if term or trunc or (priv["kills"] == 1 and misses_after_kill >= 3):
                break
        assert priv["kills"] == 1 and hits >= 1 and shots > hits
        k = 10.0 / 0.30                                      # exploration and combat only
        assert terms[TERM["kill"]] == pytest.approx(0.15 * k)
        assert misses_after_kill >= 3
        assert terms[TERM["ammo"]] == pytest.approx(-RewardCoef().ammo_cost * (shots - hits))
        assert terms[TERM["hit"]] == 0.0                     # the optional hit bonus is off by default
    finally:
        env.close()


def test_exit_oracle_sees_only_unobstructed_exits_in_view():
    """Both forms of the training-time exit observation: the ray-based fallback and the shared label module."""
    from doom_env import make_env
    wad = _reference_map("#######\n#P...E#\n#######")
    env = make_env(wad, episode_ticks=200, render_buffers=False)
    try:
        facts = deps.map_facts(wad)
        centres = facts.exit_centres
        obs, _ = env.reset(seed=0)
        priv = env.privileged_state()
        seen = deps.ray_exit_observation(obs, priv, centres, 640.0)
        assert seen is not None and seen[0] == 1.0
        assert seen[1] * 1024 == pytest.approx(centres[0, 0] - priv["x"], abs=1.0) and abs(seen[2]) < 0.01
        assert deps.ray_exit_observation(obs, priv, centres, 100.0) is None                       # out of range
        assert deps.ray_exit_observation(obs, dict(priv, angle=priv["angle"] + 180.0), centres, 640.0) is None   # behind
        full = deps.exit_observation(env, facts, obs, priv)                                        # no noise
        assert full is not None and np.allclose(full, seen, atol=2e-3)
        drops = sum(deps.exit_observation(env, facts, obs, priv, np.random.default_rng(i)) is None for i in range(200))
        assert (0 < drops < 100) if facts.meta is not None else drops == 0                         # the noise model misses some
    finally:
        env.close()
    wad = _reference_map("##########\n#P..D...E#\n##########")
    env = make_env(wad, episode_ticks=200, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        facts, priv = deps.map_facts(wad), env.privileged_state()
        assert deps.ray_exit_observation(obs, priv, facts.exit_centres, 2000.0) is None           # behind the door
        assert deps.exit_observation(env, facts, obs, priv) is None
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------
# Vector environment
# ---------------------------------------------------------------------------------------------
def test_vecenv_rows_have_spec_shapes_on_a_real_map():
    entries = deps.load_manifest(str(paths.ROOT / "maps" / "two_rooms.wad"))
    rng = np.random.default_rng(0)
    with VecEnv(entries, 0, 4, VecConfig(frontend="standin"), groups=2) as vec:
        a = vec.arrays
        assert a["rays"].shape == (4, 64, 7) and a["ents"].shape == (4, 32, 17) and a["scal"].shape == (4, spec.SCALAR_DIM)
        assert a["map"].shape == (4, spec.MAP_CHANNELS, 64, 64) and a["map"].dtype == np.uint8
        assert a["masks"].dtype == np.bool_ and a["ent_mask"].dtype == np.bool_
        assert (a["step"] == 0).all() and a["map"][:, spec.CH_FREE].any() and not a["lab_mask"].any()
        seen0 = (a["map"][:, spec.CH_FREE] > 0).sum()
        for step in range(1, 31):
            for g in range(2):
                buttons = (rng.random((vec.group_size, 9)) < 0.3).astype(np.int8)
                buttons[:, 7:] = 0
                vec.send(g, buttons)
                assert vec.recv(g) == []
        assert (a["step"] == 30).all() and np.isfinite(a["rays"]).all() and np.isfinite(a["scal"]).all()
        assert (a["map"][:, spec.CH_FREE] > 0).sum() >= seen0
        assert a["reward_int"].min() >= 0.0 and not a["terminated"].any()
        assert set(np.unique(a["map"][:, spec.CH_FREE])) <= {0, 255}


def test_vecenv_episode_end_writes_final_row_and_restarts():
    entries = [deps.MapEntry(_reference_map("#######\n#P...E#\n#######"), tier=2, episode_ticks=400)]
    with VecEnv(entries, 0, 2, VecConfig(frontend="standin", exit_noise=False)) as vec:
        a = vec.arrays
        assert a["scal"][:, spec.SCALAR_INDEX["exit_visible"]].all()         # the oracle reports the pad ahead
        infos = []
        for _ in range(60):
            vec.send(0, np.tile(FWD, (2, 1)))
            infos = vec.recv(0)
            if infos:
                break
        assert len(infos) == 2 and all(i["exit"] and i["has_exit"] and not i["truncated"] for i in infos)
        assert a["terminated"].all() and not a["truncated"].any()
        assert (a["step"] == 0).all() and (a["fin_step"] == infos[0]["len"]).all()
        assert infos[0]["ret"] > 5.0 and a["reward"][0] > 4.0                 # exit reward on the last step
        assert infos[0]["tier"] == 2 and infos[0]["terms"][TERM["exit"]] > 0
    entries = [deps.MapEntry(_reference_map("#####\n#P..#\n#####"), tier=0, episode_ticks=40)]
    with VecEnv(entries, 0, 1, VecConfig(frontend="standin")) as vec:
        for _ in range(10):
            vec.send(0, NOOP[None])
            infos = vec.recv(0)
        assert infos and infos[0]["truncated"] and not infos[0]["has_exit"] and vec.arrays["truncated"].all()


def test_shared_arrays_fall_back_to_a_mapped_file():
    """When /dev/shm is too small the rows live in a memory-mapped file; other processes attach by name."""
    from training.rl.vecenv import SharedArrays
    for backing in ("shm", "file"):
        owner = SharedArrays(3, backing=backing)
        assert owner.name.startswith(backing + ":")
        other = SharedArrays(3, owner.name)
        owner["scal"][1, 5] = 0.25
        other["action"][2] = 1
        assert other["scal"][1, 5] == 0.25 and owner["action"][2].all() and other["map"].shape == (3, spec.MAP_CHANNELS, 64, 64)
        path = owner.path
        other.close()
        owner.close()
        assert path is None or not __import__("os").path.exists(path)


def test_vecenv_workers_match_serial():
    """One worker process gives the same rows as stepping in-process (same seeds, same actions)."""
    entries = deps.load_manifest("toy:episodic")
    rng = np.random.default_rng(1)
    actions = (rng.random((40, 4, 9)) < 0.4).astype(np.int8)
    snapshots = []
    for workers in (0, 1):
        with VecEnv(entries, workers, 4, VecConfig(frontend="standin", seed=3)) as vec:
            episodes = 0
            for t in range(40):
                vec.send(0, actions[t])
                episodes += len(vec.recv(0))
            snapshots.append({k: vec.arrays[k].copy() for k in list(OBS_FIELDS) + ["reward", "terminated", "truncated"]})
            snapshots[-1]["episodes"] = episodes
    assert snapshots[0]["episodes"] == snapshots[1]["episodes"] > 0
    for k in snapshots[0]:
        assert np.array_equal(snapshots[0][k], snapshots[1][k]), k


@pytest.mark.slow
def test_vecenv_rotates_maps_from_curriculum_weights():
    """Rows move to maps built in the background, drawn from the tier weights of the curriculum.

    Episodes here last 10 steps (a few milliseconds), far shorter than building an environment, so
    most episode ends find no replacement ready and the row keeps its map: stepping never waits.
    """
    import time
    grids = ["#######\n#P...E#\n#######", "#####\n#P..#\n#####", "####\n#P.#\n####"]
    entries = [deps.MapEntry(_reference_map(g), tier=i, episode_ticks=40, index=i) for i, g in enumerate(grids)]
    cfg = VecConfig(frontend="standin", episodes_per_map=1, spare_envs=1)
    with VecEnv(entries, 1, 4, cfg, tier_probs=[0, 0, 1, 0, 0]) as vec:
        seen = set()
        for _ in range(60):
            vec.send(0, np.tile(NOOP, (4, 1)))
            seen |= {i["map"] for i in vec.recv(0)}
        assert seen == {2}                                   # only the tier with weight
        vec.set_tier_probs([1, 0, 0, 0, 0])
        current, slowest, deadline = {}, 0.0, time.monotonic() + 30.0
        while time.monotonic() < deadline and not (len(current) == 4 and set(current.values()) == {0}):
            t0 = time.perf_counter()
            vec.send(0, np.tile(NOOP, (4, 1)))
            for info in vec.recv(0):
                current[info["row"]] = info["map"]
            slowest = max(slowest, time.perf_counter() - t0)
            time.sleep(0.002)
        assert len(current) == 4 and set(current.values()) == {0}      # every row has moved to the new tier
        assert slowest < 0.1                                           # and no step waited for a construction


def test_vecenv_runs_the_exit_detector_in_the_loop(tmp_path):
    """exit_source="detector": frames are rendered and the detector's output feeds memory and encoder."""
    model = pytest.importorskip("doomnet.model")
    torch.manual_seed(0)
    weights = tmp_path / "exit_detector.pt"
    torch.save(model.ExitDetector().state_dict(), weights)
    entries = [deps.MapEntry(_reference_map("#######\n#P...E#\n#######"), tier=2, episode_ticks=400)]
    cfg = VecConfig(frontend="standin", exit_source="detector", exit_weights=str(weights))
    with VecEnv(entries, 0, 2, cfg) as vec:
        slot = vec.local.slots[0]
        assert slot.env.render_buffers and slot.detector is not None
        seen = slot.detector(np.zeros((spec.SCREEN_H, spec.SCREEN_W, 3), np.uint8))
        assert seen.shape == (3,) and 0.0 <= seen[0] <= 1.0
        for _ in range(5):
            vec.send(0, np.tile(FWD, (2, 1)))
            vec.recv(0)
        assert np.isfinite(vec.arrays["scal"]).all() and (vec.arrays["step"] == 5).all()
    with pytest.raises(ValueError):
        VecEnv(entries, 0, 1, VecConfig(frontend="standin", exit_source="camera"))


def test_teacher_hook_supplies_labels():
    """A teacher object per environment is reset at episode starts and labels every observation."""
    class Teacher:
        resets = 0

        def reset(self):
            Teacher.resets += 1

        def label(self, frontend, obs, step):
            assert hasattr(frontend, "mem") and hasattr(frontend, "odom")
            return np.array([1, 0, 2, 0, 0, 0]), 3, 100 + step, True

    original = deps.make_teacher
    deps.make_teacher = Teacher
    try:
        entries = [deps.MapEntry(_reference_map("#####\n#P..#\n#####"), tier=0, episode_ticks=40)]
        with VecEnv(entries, 0, 2, VecConfig(frontend="standin")) as vec:
            a = vec.arrays
            assert a["lab_mask"].all() and (a["lab_subgoal"] == 3).all() and (a["lab_target"] == 100).all()
            for _ in range(12):                                    # episodes last 10 steps
                vec.send(0, np.tile(NOOP, (2, 1)))
                vec.recv(0)
            assert (a["lab_target"] == 102).all() and a["lab_heads"][0].tolist() == [1, 0, 2, 0, 0, 0]
            assert Teacher.resets == 4                             # two rows, two episodes each
    finally:
        deps.make_teacher = original


def test_heldout_exit_rate_counts_first_episodes():
    entries = [deps.MapEntry(_reference_map("#######\n#P...E#\n#######"), tier=2, episode_ticks=200)]
    torch.manual_seed(0)
    net = deps.make_policy("standin")
    out = heldout_exit_rate(net, entries, "cpu", 0, 3, VecConfig(frontend="standin"), seed=1)
    assert out["episodes"] == 3 and 0.0 <= out["exit_rate"] <= 1.0


# ---------------------------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------------------------
def test_pick_device_prefers_cuda_then_mps_then_cpu():
    from training.rl.train import pick_device
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert pick_device("auto").type == expected
    assert pick_device("cpu").type == "cpu"


def test_train_script_writes_log_and_checkpoint(tmp_path):
    from training.rl.train import LOG_COLUMNS, TrainConfig, apply_overrides, train
    cfg = apply_overrides(TrainConfig(), {"workers": 0, "envs_per_worker": 4, "groups": 2, "rollout": 16, "policy": "standin",
                                          "ppo": {"chunk": 8, "minibatch_chunks": 4, "epochs": 1}, "vec": {"frontend": "standin"}})
    row = train(cfg, "toy:bandit", str(tmp_path), device_name="cpu", max_iterations=3)
    rows = list(csv.DictReader((tmp_path / "train_log.csv").open()))
    assert len(rows) == 3 and list(rows[0]) == LOG_COLUMNS
    assert rows[-1]["device"].startswith("cpu") and rows[-1]["stage"] == "nav" and rows[-1]["env_steps"] == "192"
    assert float(rows[-1]["env_steps_per_s"]) > 0 and float(rows[-1]["learner_samples_per_s"]) > 0
    assert all(rows[-1][f"ent_{name}"] != "" for name in spec.HEAD_NAMES) and rows[-1]["distill_weight"] == "1"
    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["iteration"] == 3 and "model" in state and state["curriculum"]["stage_index"] == 0
    with pytest.raises(KeyError):
        apply_overrides(TrainConfig(), {"ppo": {"no_such_setting": 1}})
    # The config saved next to the log can be passed back with --config.
    saved = json.loads((tmp_path / "config.json").read_text())
    again = apply_overrides(TrainConfig(), saved)
    assert again.ppo.chunk == 8 and again.vec.frontend == "standin" and again.ppo.ent_coef == cfg.ppo.ent_coef
    assert Curriculum(again.curriculum, [0, 1]).stage == "nav" and again.vec.reward.scale == 10.0
    resumed = train(cfg, "toy:bandit", str(tmp_path), device_name="cpu", resume=str(tmp_path / "latest.pt"), max_iterations=4)
    assert resumed["iteration"] == 4 and len(list(csv.DictReader((tmp_path / "train_log.csv").open()))) == 4


# ---------------------------------------------------------------------------------------------
# The agent's own modules (skipped until they exist)
# ---------------------------------------------------------------------------------------------
def _real_frontend():
    pytest.importorskip("doomnet.odometry")
    pytest.importorskip("doomnet.memory")
    pytest.importorskip("doomnet.encode")


def test_real_frontend_rows_and_crop_layout():
    """The real odometry / memory / encoder fill the rows, and the crop has the layout curiosity assumes."""
    _real_frontend()
    entries = deps.load_manifest(str(paths.ROOT / "maps" / "two_rooms.wad"))
    with VecEnv(entries, 0, 2, VecConfig(frontend="real")) as vec:
        assert vec.frontend_name == "doomnet"
        for _ in range(20):
            vec.send(0, np.tile(FWD, (2, 1)))
            vec.recv(0)
        a = vec.arrays
        assert np.isfinite(a["rays"]).all() and np.isfinite(a["scal"]).all() and a["map"][:, spec.CH_FREE].any()
    front = deps.FrontEnd("real")
    front.reset()
    obs = standins.blank_observation()
    obs["rays"]["distance"][:] = 0.05
    obs["rays"]["distance"][0] = 150.0 / spec.RAY_RANGE
    obs["rays"]["hit_type"][:] = spec.HIT_WALL
    feats, crop, pose = front.observe(obs, None, 0, 525.0, None)
    assert crop.shape == (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE)
    assert crop[spec.CH_WALL, 34, 32] > 0                       # wall 150 units ahead (+x): two cells up the first axis
    assert feats["rays"].shape == (64, 7) and feats["scal"].shape == (spec.SCALAR_DIM,)


def test_real_policy_replay_is_exact():
    """The same replay check as above with the agent's network (forced manager choices, state resets)."""
    pytest.importorskip("doomnet.model")
    vec, net, buf, ppo, col = make_setup("toy:episodic", envs=4, steps=16, groups=2, policy="real")
    try:
        col.collect()
        col.collect()
        buf.finish(0.9, 0.9, 0.9, 0.0)
        assert hasattr(net, "forward_sequence")
        with torch.no_grad():
            for batch in buf.minibatches(4, torch.Generator().manual_seed(0)):
                for by_steps in (False, True):          # the network's sequence mode, then step by step
                    new = unroll(net, ppo.layout, batch, by_steps=by_steps)
                    assert torch.allclose(new["logp"], batch["logp"].flatten(), atol=1e-3)
                    assert torch.allclose(new["value"], batch["value"].flatten(), atol=1e-3)
                    assert torch.allclose(new["value_int"], batch["value_int"].flatten(), atol=1e-3)
        stats = ppo.update(buf)
        assert np.isfinite(stats["loss"])
    finally:
        vec.close()
