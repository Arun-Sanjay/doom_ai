"""Tests for submission/doomnet/model.py: PolicyNet (manager, value propagation, controller) and ExitDetector.

Run: .venv/bin/python -m pytest training/tests/test_model.py -q
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from doomnet import model, spec
from doomnet.model import CENTER, EgoPatch, ExitDetector, PolicyNet

IDX = spec.SCALAR_INDEX
PERIOD = spec.MANAGER_PERIOD


@pytest.fixture(autouse=True, scope="module")
def _two_threads():
    """The machine is shared: keep this module to two threads and restore the setting afterwards."""
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def net() -> PolicyNet:
    torch.manual_seed(0)
    return PolicyNet().eval()


def make_inputs(batch: int, seq: int | None = None, seed: int = 0, uint8_map: bool = False) -> dict:
    """Random, well-formed inputs: unit headings, sub-cell offsets in range, some entity slots padded."""
    g = torch.Generator().manual_seed(seed)
    lead = (batch,) if seq is None else (batch, seq)
    scal = torch.randn(lead + (spec.SCALAR_DIM,), generator=g) * 0.3
    theta = torch.rand(lead, generator=g) * 2 * math.pi
    scal[..., IDX["sin_theta"]], scal[..., IDX["cos_theta"]] = torch.sin(theta), torch.cos(theta)
    scal[..., IDX["subcell_x"]] = torch.rand(lead, generator=g) - 0.5
    scal[..., IDX["subcell_y"]] = torch.rand(lead, generator=g) - 0.5
    crop = torch.rand(lead + (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), generator=g)
    if uint8_map:
        crop = (crop * 255).to(torch.uint8)
    return {"rays": torch.rand(lead + (spec.NUM_RAYS, spec.RAY_FEATS), generator=g),
            "ents": torch.randn(lead + (spec.MAX_ENTITIES, spec.ENT_FEATS), generator=g),
            "ent_mask": torch.rand(lead + (spec.MAX_ENTITIES,), generator=g) < 0.3,
            "scal": scal, "map": crop}


def step(net, d, state=None, **kw):
    return net(d["rays"], d["ents"], d["ent_mask"], d["scal"], d["map"], state, **kw)


def sequence(net, d, state=None, **kw):
    return net.forward_sequence(d["rays"], d["ents"], d["ent_mask"], d["scal"], d["map"], state, **kw)


def at(d: dict, t: int) -> dict:
    return {k: v[:, t] for k, v in d.items()}


def assert_outputs_close(a: dict, b: dict, atol: float = 2e-4) -> None:
    assert a.keys() == b.keys()
    for k in a:
        if k == "logits":
            for x, y in zip(a[k], b[k]):
                assert torch.allclose(x, y, atol=atol), k
        elif a[k].dtype == torch.long:
            assert torch.equal(a[k], b[k]), k
        else:
            assert torch.allclose(a[k], b[k], atol=atol), (k, float((a[k] - b[k]).abs().max()))


# ------------------------------------------------------------------------------------------ size and shapes

def test_parameter_count_is_between_one_and_two_million(net):
    total = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in ExitDetector().parameters())
    assert 1_000_000 <= total <= 2_000_000, total


@pytest.mark.parametrize("batch", [1, 5])
def test_step_shapes_and_dtypes(net, batch):
    d = make_inputs(batch)
    with torch.inference_mode():
        out, state = step(net, d, net.initial_state(batch, "cpu"), masks=torch.ones(batch, 2, dtype=torch.bool))
    assert [tuple(l.shape) for l in out["logits"]] == [(batch, n) for n in spec.HEAD_SIZES]
    assert out["value"].shape == out["value_int"].shape == (batch,)
    assert out["subgoal"].shape == out["target"].shape == (batch,)
    assert out["subgoal"].dtype == out["target"].dtype == torch.long
    assert int(out["subgoal"].max()) < spec.NUM_SUBGOALS and int(out["target"].max()) < spec.CROP_SIZE ** 2
    assert out["subgoal_logits"].shape == (batch, spec.NUM_SUBGOALS)
    assert out["target_logits"].shape == (batch, spec.CROP_SIZE ** 2)
    assert out["value_field"].shape == (batch, 1, spec.CROP_SIZE, spec.CROP_SIZE)
    assert out["ent_attn"].shape == (batch, spec.MAX_ENTITIES)
    assert all(torch.isfinite(v).all() for k, v in out.items() if k != "logits")
    fresh = net.initial_state(batch, "cpu")
    assert state.keys() == fresh.keys()
    for k in fresh:
        assert state[k].shape == fresh[k].shape and state[k].dtype == fresh[k].dtype, k


def test_uint8_map_is_scaled_like_a_float_map(net):
    d = make_inputs(2, uint8_map=True)
    with torch.inference_mode():
        a, _ = step(net, d, sample=False)
        b, _ = step(net, {**d, "map": d["map"].float() / 255}, sample=False)
    assert_outputs_close(a, b, atol=1e-4)


def test_state_none_means_initial_state(net):
    d = make_inputs(2)
    with torch.inference_mode():
        a, _ = step(net, d, None, sample=False)
        b, _ = step(net, d, net.initial_state(2, "cpu"), sample=False)
    assert_outputs_close(a, b, atol=0.0)


def test_config_rebuilds_the_same_network(net):
    clone = PolicyNet(**net.config).eval()
    clone.load_state_dict(net.state_dict())
    d = make_inputs(2)
    with torch.inference_mode():
        a, _ = step(net, d, sample=False)
        b, _ = step(clone, d, sample=False)
    assert_outputs_close(a, b, atol=0.0)


# ------------------------------------------------------------------------------------------ state and schedule

def test_manager_runs_on_first_step_then_every_period(net):
    state = net.initial_state(2, "cpu")
    chosen = []
    with torch.inference_mode():
        for t in range(2 * PERIOD + 1):
            before = state
            out, state = step(net, make_inputs(2, seed=t), state)
            ran = t % PERIOD == 0
            assert bool((out["target_logits"].abs().sum(1) > 0).all()) == ran, t
            assert bool((out["subgoal_logits"].abs().sum(1) > 0).all()) == ran, t
            assert state["since"].tolist() == [t % PERIOD + 1] * 2
            if not ran:                                   # the manager's output is held unchanged
                assert torch.equal(out["subgoal"], before["subgoal"]) and torch.equal(out["target"], before["target"])
                assert torch.equal(out["value_field"], before["value_field"])
                assert torch.equal(state["map_vec"], before["map_vec"])
            chosen.append(out["value_field"])
    assert not torch.equal(chosen[0], chosen[PERIOD])     # a new field at the next manager step
    assert torch.equal(chosen[PERIOD], chosen[2 * PERIOD - 1])


def test_explicit_schedule_overrides_the_counter(net):
    d = make_inputs(3)
    with torch.inference_mode():
        _, state = step(net, d, sample=False)
        run = torch.tensor([True, False, True])
        out, new = step(net, make_inputs(3, seed=1), state, run_manager=run, sample=False)
    assert (out["target_logits"].abs().sum(1) > 0).tolist() == run.tolist()
    assert new["since"].tolist() == [1, 2, 1]
    with torch.inference_mode():
        every, _ = step(net, make_inputs(3, seed=1), state, run_manager=True, sample=False)   # one bool for all rows
        never, _ = step(net, make_inputs(3, seed=1), state, run_manager=False, sample=False)
    assert bool((every["target_logits"].abs().sum(1) > 0).all()) and float(never["target_logits"].abs().sum()) == 0
    assert torch.equal(new["value_field"][1], state["value_field"][1])
    assert not torch.equal(new["value_field"][0], state["value_field"][0])


def test_rows_of_a_batch_do_not_interact(net):
    """A batch with only some rows on a manager step equals the same rows computed one at a time."""
    d0, d1 = make_inputs(4, seed=0), make_inputs(4, seed=1)
    run = torch.tensor([True, False, False, True])
    with torch.inference_mode():
        _, state = step(net, d0, sample=False)
        out, new = step(net, d1, state, run_manager=run, sample=False)
        for i in range(4):
            one = {k: v[i:i + 1] for k, v in d1.items()}
            out_i, new_i = step(net, one, {k: v[i:i + 1] for k, v in state.items()}, run_manager=run[i:i + 1], sample=False)
            assert_outputs_close({k: (tuple(l[i:i + 1] for l in v) if k == "logits" else v[i:i + 1]) for k, v in out.items()}, out_i)
            for k in new:
                assert torch.allclose(new[k][i:i + 1].float(), new_i[k].float(), atol=2e-4), k


def test_reset_state_restores_selected_rows(net):
    with torch.inference_mode():
        _, state = step(net, make_inputs(3))
        _, state = step(net, make_inputs(3, seed=1), state)
        reset = net.reset_state(state, torch.tensor([False, True, False]))
    fresh = net.initial_state(3, "cpu")
    for k in state:
        assert torch.equal(reset[k][1], fresh[k][1]), k
        assert torch.equal(reset[k][0], state[k][0]) and torch.equal(reset[k][2], state[k][2]), k


def test_most_likely_choice_is_deterministic_and_sampling_follows_the_generator(net):
    d = make_inputs(4)
    with torch.inference_mode():
        a, _ = step(net, d, sample=False)
        b, _ = step(net, d, sample=False)
        assert torch.equal(a["subgoal"], a["subgoal_logits"].argmax(1)) and torch.equal(a["target"], a["target_logits"].argmax(1))
        assert_outputs_close(a, b, atol=0.0)
        draws = []
        for _ in range(2):
            net.generator = torch.Generator().manual_seed(7)
            draws.append(step(net, d, sample=True)[0])
        net.generator = None
    assert torch.equal(draws[0]["target"], draws[1]["target"]) and torch.equal(draws[0]["subgoal"], draws[1]["subgoal"])
    assert not torch.equal(draws[0]["target"], a["target"])       # 4096 near-uniform cells: a sample is not the arg-max


def test_forced_choices_are_used_and_the_target_moves_the_value_field(net):
    d = make_inputs(2)
    forced = {"subgoal": torch.tensor([3, 5]), "target": torch.tensor([10 * 64 + 20, 40 * 64 + 7])}
    with torch.inference_mode():
        a, _ = step(net, d, forced=forced)
        b, _ = step(net, d, forced={**forced, "target": torch.tensor([50 * 64 + 50, 40 * 64 + 7])})
        c, _ = step(net, d, forced={"target": forced["target"]}, sample=False)      # only the target forced
    assert torch.equal(a["subgoal"], forced["subgoal"]) and torch.equal(a["target"], forced["target"])
    assert not torch.allclose(a["value_field"][0], b["value_field"][0])
    assert torch.allclose(a["value_field"][1], b["value_field"][1])
    assert torch.equal(c["target"], forced["target"]) and torch.equal(c["subgoal"], c["subgoal_logits"].argmax(1))


# ------------------------------------------------------------------------------------------ sensors

def test_masks_give_exactly_zero_probability(net):
    d = make_inputs(4)
    masks = torch.tensor([[True, True], [False, True], [True, False], [False, False]])      # (use_ok, attack_ok)
    with torch.inference_mode():
        free, _ = step(net, d, sample=False)
        out, _ = step(net, d, masks=masks, sample=False)
    use, attack = out["logits"][spec.HEAD_USE], out["logits"][spec.HEAD_ATTACK]
    p_use, p_attack = torch.softmax(use, 1), torch.softmax(attack, 1)
    assert p_use[:, 1].tolist()[1] == 0.0 and p_use[:, 1].tolist()[3] == 0.0
    assert p_attack[:, 1].tolist()[2] == 0.0 and p_attack[:, 1].tolist()[3] == 0.0
    assert p_use[1, 0] == 1.0 and p_attack[2, 0] == 1.0
    assert torch.equal(use[[0, 2]], free["logits"][spec.HEAD_USE][[0, 2]])                 # allowed rows are untouched
    assert torch.equal(attack[[0, 1]], free["logits"][spec.HEAD_ATTACK][[0, 1]])
    for h in (spec.HEAD_MOVE, spec.HEAD_STRAFE, spec.HEAD_TURN, spec.HEAD_WEAPON):
        assert torch.equal(out["logits"][h], free["logits"][h])
    assert torch.isfinite(F.log_softmax(use, 1)[:, 0]).all()                               # the "off" option always remains


def test_attention_ignores_padding(net):
    d = make_inputs(3)
    d["ent_mask"] = torch.zeros(3, spec.MAX_ENTITIES, dtype=torch.bool)
    d["ent_mask"][0, :5] = True
    d["ent_mask"][1, :1] = True                                                            # row 2: nothing visible
    other = {**d, "ents": torch.where(d["ent_mask"][:, :, None], d["ents"], torch.randn_like(d["ents"]) * 50)}
    with torch.inference_mode():
        a, _ = step(net, d, sample=False)
        b, _ = step(net, other, sample=False)
    assert_outputs_close(a, b, atol=0.0)
    attn = a["ent_attn"]
    assert torch.equal(attn[~d["ent_mask"]], torch.zeros(int((~d["ent_mask"]).sum())))      # exactly zero on padding
    assert (attn[d["ent_mask"]] > 0).all() and (attn.sum(1) <= 1 + 1e-6).all()
    assert attn[2].sum() == 0 and all(torch.isfinite(l).all() for l in a["logits"])


def test_attention_depends_on_visible_entities(net):
    d = make_inputs(1)
    d["ent_mask"][0, :3] = True
    moved = {**d, "ents": d["ents"].clone()}
    moved["ents"][0, 1] += 1.0
    with torch.inference_mode():
        a, _ = step(net, d, sample=False)
        b, _ = step(net, moved, sample=False)
    assert not torch.allclose(a["ent_attn"], b["ent_attn"])
    assert not torch.allclose(a["logits"][spec.HEAD_TURN], b["logits"][spec.HEAD_TURN], atol=1e-7)


def test_ray_convolution_is_circular(net):
    """Turning the ray fan by 4 rays shifts the 16 convolution outputs by one position, wrapping around."""
    rays = torch.rand(2, spec.NUM_RAYS, spec.RAY_FEATS)
    conv = net.ray_net[:6]
    with torch.inference_mode():
        a = conv(F.pad(rays.transpose(1, 2), (5, 5), mode="circular"))
        b = conv(F.pad(rays.roll(4, 1).transpose(1, 2), (5, 5), mode="circular"))
    assert a.shape == (2, 64, 16)
    assert torch.allclose(a.roll(1, 2), b, atol=1e-5)


def test_bad_scalars_do_not_poison_the_state(net):
    d = make_inputs(2)
    d["scal"][0, 0], d["scal"][0, 3], d["scal"][1, IDX["vel_forward"]] = float("nan"), float("inf"), -float("inf")
    d["ents"][0, 0, 3] = float("nan")
    with torch.inference_mode():
        out, state = step(net, d, sample=False)
        out, state = step(net, make_inputs(2, seed=1), state, sample=False)
    assert all(torch.isfinite(l).all() for l in out["logits"]) and torch.isfinite(out["value"]).all()
    assert all(torch.isfinite(v).all() for v in state.values())


# ------------------------------------------------------------------------------------------ ego frame and cell counter

@pytest.mark.parametrize("theta_deg", [0.0, 90.0, 180.0, -37.0])
def test_ego_patch_is_rotated_into_the_agent_frame(theta_deg):
    """On a field equal to a x + b y, the sample `f` cells ahead and `l` cells left must read the rotated offset."""
    ego = EgoPatch(4)
    i = torch.arange(spec.CROP_SIZE, dtype=torch.float32)
    a, b = 1.0, 0.25
    field = (a * i[:, None] + b * i[None, :])[None, None]                                  # x along dim -2, y along dim -1
    theta = math.radians(theta_deg)
    sin, cos = torch.tensor([[math.sin(theta)]]), torch.tensor([[math.cos(theta)]])
    pos = torch.tensor([[CENTER + 2.3, CENTER - 1.4]])
    patch = ego(field, pos, sin, cos)[0, 0]
    assert patch.shape == (9, 9)
    for r, f in enumerate(range(4, -5, -1)):                                               # rows: ahead -> behind
        for c, l in enumerate(range(4, -5, -1)):                                           # columns: left -> right
            x = pos[0, 0] + f * math.cos(theta) - l * math.sin(theta)
            y = pos[0, 1] + f * math.sin(theta) + l * math.cos(theta)
            assert abs(float(patch[r, c]) - float(a * x + b * y)) < 1e-3, (r, c)
    assert abs(float(patch[4, 4]) - float(a * pos[0, 0] + b * pos[0, 1])) < 1e-4          # the centre is the agent itself


def test_ego_patch_is_differentiable_with_respect_to_the_field():
    ego = EgoPatch(4)
    field = torch.zeros(1, 1, 64, 64, requires_grad=True)
    ego(field, torch.tensor([[CENTER + 0.2, CENTER - 0.3]]), torch.tensor([[0.6]]), torch.tensor([[0.8]])).sum().backward()
    assert float(field.grad.sum()) == pytest.approx(81.0, abs=1e-3) and int((field.grad != 0).sum()) > 81


def _walk(steps: int, seed: int):
    """A physically plausible walk: (x, y, heading, forward velocity, rightward velocity) per decision step.

    The velocity changes gradually, as with the engine's friction, and the position advances by the
    mean of the velocity before and after the step plus an error of up to 6 units per axis (the
    pose is an estimate). Every few steps the agent runs into something and stops dead.
    """
    g = torch.Generator().manual_seed(seed)
    rand = lambda: float(torch.rand(1, generator=g))                                       # noqa: E731
    x = y = theta = fwd = right = wx = wy = 0.0
    rows = []
    for t in range(steps):
        theta += (rand() - 0.5) * 0.6
        if t % 8 == 0:
            goal_fwd, goal_right = rand() * 14.0 - 2.0, (rand() - 0.5) * 16.0              # up to running and strafing speed
        fwd, right = 0.7 * fwd + 0.3 * goal_fwd, 0.7 * right + 0.3 * goal_right
        if t % 37 == 36:
            fwd = right = 0.0                                                              # blocked: the velocity drops to zero
        left = -right
        nwx, nwy = fwd * math.cos(theta) - left * math.sin(theta), fwd * math.sin(theta) + left * math.cos(theta)
        x += spec.FRAME_SKIP * 0.5 * (wx + nwx) + (rand() - 0.5) * 12.0
        y += spec.FRAME_SKIP * 0.5 * (wy + nwy) + (rand() - 0.5) * 12.0
        wx, wy = nwx, nwy
        rows.append((x, y, theta, fwd, right))
    return rows


def test_cell_counter_follows_the_agent_between_manager_steps(net):
    cell_size = spec.MAP_RES * spec.CROP_MULT
    state = net.initial_state(1, "cpu")
    anchor, crossings = None, 0
    with torch.inference_mode():
        for t, (x, y, theta, fwd, right) in enumerate(_walk(480, seed=3)):
            ix, iy = math.floor(x / cell_size + 0.5), math.floor(y / cell_size + 0.5)    # crop cell: [64k - 32, 64k + 32)
            d = make_inputs(1, seed=t)
            d["scal"][0, IDX["sin_theta"]], d["scal"][0, IDX["cos_theta"]] = math.sin(theta), math.cos(theta)
            d["scal"][0, IDX["subcell_x"]], d["scal"][0, IDX["subcell_y"]] = x / cell_size - ix, y / cell_size - iy
            d["scal"][0, IDX["vel_forward"]], d["scal"][0, IDX["vel_right"]] = fwd / 10.0, right / 10.0
            _, state = step(net, d, state)
            if t % PERIOD == 0:
                anchor = (ix, iy)
            assert state["cell"][0].tolist() == [ix - anchor[0], iy - anchor[1]], t
            crossings += int(t % PERIOD != 0 and (ix, iy) != anchor)
    assert crossings > 150                                                                 # the walk really crossed cells


def test_controller_reads_the_field_at_its_current_position(net):
    """Walk forward one whole cell after a manager step: the patch must shift by one cell on the held field."""
    d = make_inputs(1)
    for k in ("subcell_x", "subcell_y", "sin_theta", "vel_right"):
        d["scal"][0, IDX[k]] = 0.0
    d["scal"][0, IDX["cos_theta"]] = 1.0
    d["scal"][0, IDX["vel_forward"]] = 32.0 / spec.FRAME_SKIP / 10.0                       # half a cell per step, facing +x
    half, whole = {k: v.clone() for k, v in d.items()}, {k: v.clone() for k, v in d.items()}
    half["scal"][0, IDX["subcell_x"]] = -0.5                                               # on the edge of the next cell
    with torch.inference_mode():
        out, state = step(net, d, sample=False)
        _, state = step(net, half, state)
        assert state["cell"][0].tolist() == [1.0, 0.0] and state["sub"][0].tolist() == [-0.5, 0.0]
        _, state = step(net, whole, state)
        assert state["cell"][0].tolist() == [1.0, 0.0]
        field = out["value_field"]
        got = net.ego(field, CENTER + state["cell"] + state["sub"], torch.zeros(1, 1), torch.ones(1, 1))[0, 0]
    expected = field[0, 0, CENTER + 1 - 4:CENTER + 1 + 5, CENTER - 4:CENTER + 5].flip(0, 1)  # ahead/left first
    assert torch.allclose(got, expected, atol=1e-3)


def test_goal_read_is_linear_in_the_value_field(net):
    """Nothing else in the field may shrink the part of the patch that depends on the target.

    A per-sample normalisation of the patch (dividing by its own size) fails this: a pattern that
    all rows share then squeezes the differences between rows, which silenced the goal channel
    early in training for some seeds. The read is differences to the agent's own cell times a constant.
    """
    g = torch.Generator().manual_seed(4)
    fields = torch.randn(6, 1, spec.CROP_SIZE, spec.CROP_SIZE, generator=g) * 0.1         # the part that differs between rows
    common = (torch.randn(1, 1, spec.CROP_SIZE, spec.CROP_SIZE, generator=g) * 30.0).expand_as(fields)   # shared, 300 times larger
    pos = CENTER + torch.rand(6, 2, generator=g) - 0.5
    theta = torch.rand(6, 1, generator=g) * 2 * math.pi
    read = lambda f: net._goal_patch(f, pos, torch.sin(theta), torch.cos(theta))           # noqa: E731
    with torch.inference_mode():
        alone = read(fields)
        assert alone.shape == (6, 81) and float(alone[:, 40].abs().max()) == 0.0           # differences to the agent's own cell
        assert torch.allclose(read(fields + common) - read(common), alone, atol=2e-3)      # the shared part does not attenuate it
        assert torch.allclose(read(3.0 * fields), 3.0 * alone, atol=1e-4)                  # no normalisation by the patch's own size
        assert torch.allclose(read(fields + 7.0), alone, atol=1e-3)                        # the level of the field carries nothing
        plain = net.ego(fields, pos, torch.sin(theta), torch.cos(theta)).flatten(1)
        assert torch.allclose(alone, (plain - plain[:, 40:41]) * model.GOAL_GAIN, atol=1e-6)


def test_goal_channel_stats_show_a_dead_channel(net):
    """The numbers a training log needs to notice that the value field no longer reaches the controller."""
    assert PolicyNet().goal_channel_stats() == {"goal_feat_std": 0.0, "goal_feat_alive": 0.0}   # before any forward pass
    d = make_inputs(6)
    with torch.inference_mode():
        _, state = step(net, d, sample=False)
        live = net.goal_channel_stats()
        flat = {**state, "value_field": torch.zeros_like(state["value_field"])}            # a field with no shape at all
        step(net, d, flat, run_manager=False)
        dead = net.goal_channel_stats()
        step(net, {k: v[:1] for k, v in d.items()}, sample=False)
        single = net.goal_channel_stats()
        sequence(net, make_inputs(2, 5, seed=3), sample=False)
        seq = net.goal_channel_stats()
    assert live["goal_feat_std"] > 0.05 and live["goal_feat_alive"] > 0.5
    assert dead == {"goal_feat_std": 0.0, "goal_feat_alive": 0.0}
    assert single["goal_feat_std"] == 0.0 and single["goal_feat_alive"] > 0.2             # one row: no spread to measure
    assert seq["goal_feat_std"] > 0.05
    assert "_goal_feat" not in net.state_dict()                                            # diagnostics only, never saved


# ------------------------------------------------------------------------------------------ hierarchy and bottleneck

def test_clamping_the_subgoal_changes_the_controller_logits(net):
    d = make_inputs(2)
    try:
        results = []
        for index in (0, 5):
            net.clamp_subgoal(index)
            with torch.inference_mode():
                out, state = step(net, d, sample=False)
                out2, _ = step(net, make_inputs(2, seed=1), state, sample=False)           # a non-manager step
            assert out["subgoal"].tolist() == [index, index] and out2["subgoal"].tolist() == [index, index]
            results.append((out, out2))
        for a, b in zip(results[0], results[1]):
            assert any(float((x - y).abs().max()) > 1e-5 for x, y in zip(a["logits"], b["logits"]))
            assert not torch.allclose(a["value_field"], b["value_field"])                  # the value propagation is conditioned too
        net.clamp_subgoal(None)
        with torch.inference_mode():
            free, _ = step(net, d, sample=False)
        assert torch.equal(free["subgoal"], free["subgoal_logits"].argmax(1))
        with pytest.raises(ValueError):
            net.clamp_subgoal(spec.NUM_SUBGOALS)
    finally:
        net.clamp_subgoal(None)


def test_clamp_takes_effect_between_manager_steps(net):
    d = make_inputs(1)
    try:
        with torch.inference_mode():
            out, state = step(net, d, sample=False)
            other = (int(out["subgoal"]) + 1) % spec.NUM_SUBGOALS
            free, _ = step(net, make_inputs(1, seed=1), state, sample=False)
            net.clamp_subgoal(other)
            held, _ = step(net, make_inputs(1, seed=1), state, sample=False)
        assert int(held["subgoal"]) == other != int(free["subgoal"])
        assert any(float((x - y).abs().max()) > 1e-5 for x, y in zip(free["logits"], held["logits"]))
    finally:
        net.clamp_subgoal(None)


def test_strict_bottleneck_controller_sees_only_subgoal_and_value_field(net):
    """Off the manager schedule the controller's logits may not depend on the map, the target or the map summary."""
    assert net.strict
    with torch.inference_mode():
        _, state = step(net, make_inputs(2), sample=False)
        d = make_inputs(2, seed=1)
        base, _ = step(net, d, state, sample=False)

        other_map, _ = step(net, {**d, "map": torch.rand_like(d["map"])}, state, sample=False)
        hidden = {**state, "target": (state["target"] + 77) % spec.CROP_SIZE ** 2, "map_vec": state["map_vec"] + 1.0}
        other_hidden, _ = step(net, d, hidden, sample=False)
        for variant in (other_map, other_hidden):
            for x, y in zip(base["logits"], variant["logits"]):
                assert torch.equal(x, y)
        assert not torch.allclose(base["value"], other_hidden["value"])                    # the critic does read the map summary

        other_field, _ = step(net, d, {**state, "value_field": state["value_field"].flip(2)}, sample=False)
        other_goal, _ = step(net, d, {**state, "subgoal": (state["subgoal"] + 1) % spec.NUM_SUBGOALS}, sample=False)
        for variant in (other_field, other_goal):
            assert any(float((x - y).abs().max()) > 1e-5 for x, y in zip(base["logits"], variant["logits"]))


def test_loosened_bottleneck_lets_the_controller_read_the_map():
    torch.manual_seed(0)
    loose = PolicyNet(strict_goal_bottleneck=False).eval()
    with torch.inference_mode():
        _, state = step(loose, make_inputs(2), sample=False)
        d = make_inputs(2, seed=1)
        base, _ = step(loose, d, state, sample=False)
        other, _ = step(loose, {**d, "map": torch.rand_like(d["map"])}, state, sample=False)
        assert any(float((x - y).abs().max()) > 1e-5 for x, y in zip(base["logits"], other["logits"]))
        seq = make_inputs(3, 10, seed=2, uint8_map=True)
        ref, ref_state = sequence(loose, seq, sample=False)
        fast, fast_state = sequence(loose, seq, forced={"subgoal": ref["subgoal"], "target": ref["target"]})
    assert_outputs_close(ref, fast)
    for k in ref_state:
        assert torch.allclose(ref_state[k].float(), fast_state[k].float(), atol=2e-4), k


@pytest.mark.parametrize("iters", [0, 1, 4, 12])
def test_value_propagation_applies_its_cell_k_times(iters):
    torch.manual_seed(0)
    small = PolicyNet(vp_iters=iters).eval()
    calls, best = [], small.vp._best
    small.vp._best = lambda q: (calls.append(1), best(q))[1]                 # one max per backup, plus the initial one
    with torch.inference_mode():
        out, _ = step(small, make_inputs(1), sample=False)
    assert len(calls) == iters + 1 and out["value_field"].shape == (1, 1, 64, 64)


def test_value_propagation_is_a_contraction_for_any_weights():
    """Transition kernels are distributions and discounts are below one, so values stay within |r| / (1 - gamma)."""
    torch.manual_seed(0)
    vp = model.ValuePropagation(iters=40)
    with torch.no_grad():
        vp.transition.normal_(0, 30.0)                                       # hostile weights
        vp.discount.fill_(3.0)                                               # gamma = 0.953
        vp.inject.weight.normal_(0, 1.0)
    w, gamma = vp.kernel().detach(), torch.sigmoid(vp.discount).detach()
    assert (w >= 0).all() and torch.allclose(w.sum((1, 2, 3)), gamma, atol=1e-5) and float(gamma.max()) < 1
    feat, emb, target = torch.randn(3, 32, 32, 32), torch.randn(3, 32), torch.tensor([0, 2080, 4095])
    captured = {}
    vp.inject.register_forward_hook(lambda m, i, o: captured.__setitem__("r", o))
    best = vp._best
    vp._best = lambda q: captured.setdefault("v", []).append(best(q)) or captured["v"][-1]
    with torch.inference_mode():
        field = vp(feat, emb, target)
        r_max = float((captured["r"] + vp.goal(emb)[:, :, None, None]).abs().max())
    bound = r_max / (1 - float(gamma.max()))
    assert all(float(v.abs().max()) <= bound * (1 + 1e-5) for v in captured["v"])
    assert torch.isfinite(field).all() and field.shape == (3, 1, 64, 64)


def test_value_field_starts_as_a_function_of_the_target(net):
    """Before any learning the field around the agent must already tell where the target is."""
    d = make_inputs(4)
    d["map"] = (d["map"] < 0.03).float()                                     # sparse clutter
    with torch.inference_mode():
        fields = [step(net, d, forced={"subgoal": torch.zeros(4, dtype=torch.long), "target": torch.full((4,), t)})[0]["value_field"]
                  for t in ((CENTER + 6) * 64 + CENTER, (CENTER - 6) * 64 + CENTER)]
    centre = slice(CENTER - 4, CENTER + 5)
    a, b = fields[0][:, 0, centre, centre], fields[1][:, 0, centre, centre]
    a, b = a - a.mean((1, 2), keepdim=True), b - b.mean((1, 2), keepdim=True)
    assert float((a - b).abs().mean()) > 0.5 * float(a.abs().mean())          # the target moves the local shape of the field


def test_gradients_reach_every_level(net):
    """The controller's loss must train the value propagation, the tower and the sub-goal embedding."""
    d = make_inputs(4, 6, seed=5)
    forced = {"subgoal": torch.randint(0, spec.NUM_SUBGOALS, (4, 6)), "target": torch.randint(0, 4096, (4, 6))}
    net.zero_grad()
    out, _ = sequence(net, d, forced=forced)
    sum(l.logsumexp(-1).sum() + l[..., 0].sum() for l in out["logits"]).backward()
    for name in ("vp.transition", "vp.discount", "vp.inject.weight", "vp.read.weight", "tower.stem.weight", "subgoal_emb.weight",
                 "goal_fc.weight", "gru.weight_hh", "ray_net.0.weight", "ent_query.weight"):
        grad = net.get_parameter(name).grad
        assert grad is not None and float(grad.abs().sum()) > 0, name
    assert net.pointer_out.weight.grad is None or float(net.pointer_out.weight.grad.abs().sum()) == 0   # not through the pointer
    net.zero_grad()
    out, _ = sequence(net, d, forced=forced)
    (out["target_logits"].logsumexp(-1).sum() + out["subgoal_logits"].logsumexp(-1).sum()).backward()
    for name in ("pointer_out.weight", "pointer_fc.weight", "subgoal_head.weight", "manager_fc.weight", "tower.conv8.weight"):
        assert float(net.get_parameter(name).grad.abs().sum()) > 0, name
    net.zero_grad()


# ------------------------------------------------------------------------------------------ sequence mode

def _resets(batch: int, seq: int) -> torch.Tensor:
    reset = torch.zeros(batch, seq, dtype=torch.bool)
    reset[0, 0] = True
    reset[1, 5] = True
    if seq > 10:
        reset[2, 9] = reset[2, 10] = True                                     # two episodes of one step each
    return reset


def test_sequence_without_forced_choices_is_the_step_loop(net):
    batch, seq = 3, 19
    d, reset = make_inputs(batch, seq, seed=2), _resets(batch, 19)
    masks = torch.rand(batch, seq, 2) < 0.5
    with torch.inference_mode():
        out, final = sequence(net, d, masks=masks, reset=reset, sample=False)
        state = net.initial_state(batch, "cpu")
        for t in range(seq):
            state = net.reset_state(state, reset[:, t])
            o, state = step(net, at(d, t), state, masks=masks[:, t], sample=False)
            assert_outputs_close({k: (tuple(l[:, t] for l in v) if k == "logits" else v[:, t]) for k, v in out.items()}, o, atol=1e-6)
            if t == 5:
                assert bool(o["target_logits"][1].abs().sum() > 0)                         # the reset restarted the schedule
    for k in final:
        assert torch.allclose(final[k].float(), state[k].float(), atol=1e-6), k


@pytest.mark.parametrize("explicit_schedule", [False, True])
def test_sequence_with_forced_choices_equals_step_mode(net, explicit_schedule):
    """The batched learning path against step mode: outputs, manager schedule, resets and final state."""
    batch, seq = 4, 27
    d, reset = make_inputs(batch, seq, seed=4, uint8_map=True), _resets(batch, 27)
    masks = torch.rand(batch, seq, 2) < 0.5
    run = None
    if explicit_schedule:                                                                  # an irregular schedule
        run = torch.rand(batch, seq, generator=torch.Generator().manual_seed(1)) < 0.2
        run[:, 0] = True
    with torch.inference_mode():
        state = net.initial_state(batch, "cpu")
        steps = []
        for t in range(seq):                                                               # rollout: sampled manager choices
            state = net.reset_state(state, reset[:, t])
            o, state = step(net, at(d, t), state, masks=masks[:, t], run_manager=None if run is None else run[:, t] | reset[:, t])
            steps.append(o)
        forced = {"subgoal": torch.stack([o["subgoal"] for o in steps], 1), "target": torch.stack([o["target"] for o in steps], 1)}
        out, final = sequence(net, d, masks=masks, run_manager=run, forced=forced, reset=reset)
    ran = torch.stack([o["target_logits"].abs().sum(1) > 0 for o in steps], 1)
    assert torch.equal(out["target_logits"].abs().sum(2) > 0, ran)                         # same manager schedule
    if not explicit_schedule:
        assert ran[3].tolist() == [t % PERIOD == 0 for t in range(seq)] and bool(ran[1, 5]) and bool(ran[1, 13])
    for t, o in enumerate(steps):
        assert_outputs_close({k: (tuple(l[:, t] for l in v) if k == "logits" else v[:, t]) for k, v in out.items()}, o)
    for k in final:
        assert torch.allclose(final[k].float(), state[k].float(), atol=2e-4), k


def test_sequence_can_be_split_into_chunks(net):
    """Truncated back-propagation: two consecutive chunks with the carried state equal one long sequence."""
    batch, seq, cut = 3, 22, 11                                                            # the cut is not on a manager step
    d, reset = make_inputs(batch, seq, seed=6), _resets(batch, 22)
    with torch.inference_mode():
        ref, _ = sequence(net, d, reset=reset, sample=True)
        forced = {"subgoal": ref["subgoal"], "target": ref["target"]}
        whole, whole_state = sequence(net, d, forced=forced, reset=reset)
        head, mid_state = sequence(net, {k: v[:, :cut] for k, v in d.items()}, forced={k: v[:, :cut] for k, v in forced.items()},
                                   reset=reset[:, :cut])
        tail, tail_state = sequence(net, {k: v[:, cut:] for k, v in d.items()}, mid_state,
                                    forced={k: v[:, cut:] for k, v in forced.items()}, reset=reset[:, cut:])
    assert_outputs_close(whole, ref)
    joined = {k: (tuple(torch.cat([a, b], 1) for a, b in zip(head[k], tail[k])) if k == "logits" else torch.cat([head[k], tail[k]], 1))
              for k in head}
    assert_outputs_close(whole, joined)
    for k in whole_state:
        assert torch.allclose(whole_state[k].float(), tail_state[k].float(), atol=2e-4), k


def test_sequence_with_no_manager_step_keeps_the_held_output(net):
    d = make_inputs(2, 3, seed=8)
    with torch.inference_mode():
        _, state = step(net, make_inputs(2), sample=False)
        forced = {"subgoal": torch.zeros(2, 3, dtype=torch.long), "target": torch.zeros(2, 3, dtype=torch.long)}
        out, final = sequence(net, d, state, forced=forced)
    assert float(out["target_logits"].abs().sum()) == 0
    assert torch.equal(out["subgoal"], state["subgoal"][:, None].expand(2, 3))
    assert torch.equal(final["value_field"], state["value_field"]) and final["since"].tolist() == [4, 4]


# ------------------------------------------------------------------------------------------ learning

def test_tiny_overfit_policy_memorises_a_batch():
    """Sequence mode can drive the loss of all six heads and both manager heads towards zero on one batch."""
    torch.manual_seed(1)
    small = PolicyNet()
    batch, seq = 2, 8
    d = make_inputs(batch, seq, seed=9)
    g = torch.Generator().manual_seed(2)
    labels = [torch.randint(0, n, (batch, seq), generator=g) for n in spec.HEAD_SIZES]
    forced = {"subgoal": torch.randint(0, spec.NUM_SUBGOALS, (batch, seq), generator=g),
              "target": torch.randint(0, 4096, (batch, seq), generator=g)}
    returns = torch.randn(batch, seq, generator=g)
    opt = torch.optim.Adam(small.parameters(), lr=1e-3)

    def loss_fn():
        out, _ = sequence(small, d, forced=forced)
        run = out["target_logits"].abs().sum(-1) > 0
        loss = sum(F.cross_entropy(l.flatten(0, 1), y.flatten()) for l, y in zip(out["logits"], labels))
        loss = loss + F.cross_entropy(out["subgoal_logits"][run], forced["subgoal"][run])
        loss = loss + F.cross_entropy(out["target_logits"][run], forced["target"][run])
        return loss + F.mse_loss(out["value"], returns) + F.mse_loss(out["value_int"], -returns)

    first = float(loss_fn().detach())
    for _ in range(80):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    last = float(loss_fn().detach())
    assert first > 10.0 and last < 0.05 * first, (first, last)


def test_controller_learns_to_follow_the_target_through_the_value_field():
    """Rows differ only in the manager's target, so the correct button can only be read off the value field."""
    torch.manual_seed(2)
    small = PolicyNet()
    offsets = [(3, 0), (-3, 0), (0, 3), (0, -3)] * 2                                       # target ahead, behind, left, right
    batch, seq = len(offsets), 4
    d = {k: v[:1].expand(batch, *v.shape[1:]).clone() for k, v in make_inputs(1, seq, seed=3).items()}
    for k, value in (("sin_theta", 0.0), ("cos_theta", 1.0), ("subcell_x", 0.0), ("subcell_y", 0.0), ("vel_forward", 0.0), ("vel_right", 0.0)):
        d["scal"][..., IDX[k]] = value                                                     # facing +x, standing still
    target = torch.tensor([(CENTER + i) * spec.CROP_SIZE + CENTER + j for i, j in offsets])
    forced = {"subgoal": torch.zeros(batch, seq, dtype=torch.long), "target": target[:, None].expand(batch, seq)}
    move = torch.tensor([1 if i > 0 else 2 if i < 0 else 0 for i, _ in offsets])[:, None].expand(batch, seq)      # fwd / back
    strafe = torch.tensor([1 if j > 0 else 2 if j < 0 else 0 for _, j in offsets])[:, None].expand(batch, seq)    # left / right
    opt = torch.optim.Adam(small.parameters(), lr=2e-3)
    for _ in range(40):
        out, _ = sequence(small, d, forced=forced)
        loss = F.cross_entropy(out["logits"][spec.HEAD_MOVE].flatten(0, 1), move.flatten()) \
            + F.cross_entropy(out["logits"][spec.HEAD_STRAFE].flatten(0, 1), strafe.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.inference_mode():
        out, _ = sequence(small, d, forced=forced)
    assert torch.equal(out["logits"][spec.HEAD_MOVE].argmax(-1), move)
    assert torch.equal(out["logits"][spec.HEAD_STRAFE].argmax(-1), strafe)
    assert float(loss.detach()) < 0.2


def _open_floor(n: int) -> torch.Tensor:
    crop = torch.zeros(n, spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE)
    crop[:, spec.CH_FREE, CENTER - 14:CENTER + 15, CENTER - 14:CENTER + 15] = 1.0
    return crop


def _detour_batch(n: int, g: torch.Generator):
    """Half the rows: open floor, target to the left or right. Other half: a wall bar across the way to the
    target with a single gap on the left or on the right. Label: strafe towards the target, or towards the gap."""
    half = n // 2
    side = torch.randint(0, 2, (n,), generator=g) * 2 - 1                                  # +1 = left (+y)
    crop = _open_floor(n)
    ahead = torch.randint(2, 6, (half,), generator=g)
    gap = torch.randint(2, 9, (half,), generator=g)
    for k in range(half):                                                                  # rows [0, half): the wall
        i, j = CENTER + int(ahead[k]), CENTER + int(side[k]) * int(gap[k])
        crop[k, spec.CH_WALL, i, CENTER - 10:CENTER + 11] = 1.0
        crop[k, spec.CH_FREE, i, CENTER - 10:CENTER + 11] = 0.0
        crop[k, spec.CH_WALL, i, j], crop[k, spec.CH_FREE, i, j] = 0.0, 1.0
    behind_wall = (CENTER + ahead + torch.randint(2, 6, (half,), generator=g)) * spec.CROP_SIZE \
        + CENTER + torch.randint(-1, 2, (half,), generator=g)                              # target straight ahead: no hint
    open_side = side[half:]
    in_the_open = (CENTER + torch.randint(0, 7, (n - half,), generator=g)) * spec.CROP_SIZE \
        + CENTER + open_side * torch.randint(2, 9, (n - half,), generator=g)
    scal = torch.zeros(n, spec.SCALAR_DIM)
    scal[:, IDX["cos_theta"]] = 1.0                                                        # facing +x
    d = {"rays": torch.zeros(n, spec.NUM_RAYS, spec.RAY_FEATS), "ents": torch.zeros(n, spec.MAX_ENTITIES, spec.ENT_FEATS),
         "ent_mask": torch.zeros(n, spec.MAX_ENTITIES, dtype=torch.bool), "scal": scal, "map": crop}
    forced = {"subgoal": torch.zeros(n, dtype=torch.long), "target": torch.cat([behind_wall, in_the_open])}
    return d, forced, torch.where(side > 0, 1, 2)                                          # strafe left / right


def _detour_scores(small: PolicyNet) -> tuple[float, float, float]:
    """Accuracy on held-out rows: open floor, wall, and the wall rows again with the wall removed from the map."""
    d, forced, label = _detour_batch(512, torch.Generator().manual_seed(99))
    with torch.inference_mode():
        out, _ = step(small, d, forced=forced)
        right = out["logits"][spec.HEAD_STRAFE].argmax(1) == label
        blind, _ = step(small, {**d, "map": _open_floor(512)}, forced=forced)               # same targets, wall removed
        blind_right = blind["logits"][spec.HEAD_STRAFE].argmax(1) == label
    return float(right[256:].float().mean()), float(right[:256].float().mean()), float(blind_right[:256].float().mean())


def _detour_training(seed: int, lr: float):
    """A fresh default network and a function that trains it on the detour task for a number of updates of 32 rows."""
    torch.manual_seed(seed)
    small = PolicyNet()
    assert small.strict
    opt = torch.optim.Adam(small.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    def train(updates: int) -> None:
        for _ in range(updates):
            d, forced, label = _detour_batch(32, g)
            out, _ = step(small, d, forced=forced)
            loss = F.cross_entropy(out["logits"][spec.HEAD_STRAFE], label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(small.parameters(), 5.0)
            opt.step()

    return small, train


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 3])
def test_value_field_learns_a_detour_around_a_wall(seed):
    """Under the strict bottleneck the side of the gap can only reach the controller through the value field.

    The sensors are blank and the target lies straight ahead behind the wall, so chance is 50%.
    Open-floor rows teach the controller to follow the field; the wall rows then force the learned
    value propagation to shape the field from the map. Run for three seeds at the learning rate
    training uses (3e-4). The update at which the wall is learned varies between seeds, so a
    seed that has not met the thresholds after 220 updates gets 110 more.
    """
    small, train = _detour_training(seed, lr=3e-4)
    history = []
    for updates in (220, 110):
        train(updates)
        open_floor, wall, wall_removed = _detour_scores(small)
        history.append((open_floor, wall, wall_removed))
        # open floor: follows the target; wall: goes for the gap; and it read that from the map
        if open_floor > 0.95 and wall > 0.9 and wall_removed < 0.7:
            break
    else:
        pytest.fail(f"seed {seed}: (open floor, wall, wall removed) after 220 and 330 updates = {history}")


@pytest.mark.slow
def test_goal_channel_survives_a_high_learning_rate():
    """At learning rate 1e-3 the value field must still reach the controller.

    With the patch divided by its own size, this seed stayed at chance on every row of the detour
    task, open floor included, however long it trained: a pattern shared by all rows grew in the
    field and the division squeezed out the part that depends on the target. The open-floor rows
    need nothing but that part, so they are the check; goal_channel_stats() must agree.
    """
    small, train = _detour_training(seed=1, lr=1e-3)
    train(220)
    open_floor, _, _ = _detour_scores(small)
    stats = small.goal_channel_stats()                                                     # of the last scoring pass
    assert open_floor > 0.95, (open_floor, stats)
    assert stats["goal_feat_std"] > 0.1 and stats["goal_feat_alive"] > 0.5, stats


# ------------------------------------------------------------------------------------------ exit detector

def _pad_frames(n: int, seed: int = 0):
    """Synthetic frames: half show a red rectangle on the lower half, with a label for where it is."""
    g = torch.Generator().manual_seed(seed)
    rgb = torch.randint(60, 120, (n, 3, spec.SCREEN_H, spec.SCREEN_W), generator=g, dtype=torch.uint8)
    labels = torch.zeros(n, 3)
    for k in range(0, n, 2):
        row = int(torch.randint(62, 110, (1,), generator=g))
        col = int(torch.randint(5, 140, (1,), generator=g))
        height = 1 + (row - 60) // 8                                                       # thin when near the horizon
        rgb[k, 0, row:row + height, col:col + 14] = 230
        rgb[k, 1:, row:row + height, col:col + 14] = 20
        forward = 3.2 / (row - 59)                                             # nearer pads sit lower in the image
        labels[k] = torch.tensor([1.0, forward, forward * (80 - col - 7) / 80.0])
    return rgb, labels


def test_exit_detector_shapes_and_input_types():
    torch.manual_seed(0)
    det = ExitDetector().eval()
    rgb, _ = _pad_frames(4)
    with torch.inference_mode():
        a = det(rgb)
        b = det(rgb.float() / 255)
        one = det(rgb[2:3])
        hwc = det(rgb.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2))                 # a permuted view, as the agent passes it
    assert a.shape == (4, 3) and a.dtype == torch.float32
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a[2:3], one, atol=1e-5) and torch.allclose(a, hwc, atol=1e-6)


def test_exit_detector_ignores_rows_above_its_window():
    det = ExitDetector().eval()
    rgb, _ = _pad_frames(2)
    other = rgb.clone()
    other[:, :, :ExitDetector.TOP] = 255 - other[:, :, :ExitDetector.TOP]
    with torch.inference_mode():
        assert torch.equal(det(rgb), det(other))
    assert ExitDetector.TOP < spec.SCREEN_H // 2                                           # the horizon row is inside the window


def test_max_pool_matches_torch():
    x = torch.randn(2, 5, 12, 20)
    assert torch.equal(model.max_pool_2x2(x), F.max_pool2d(x, 2))


def test_tiny_overfit_exit_detector():
    torch.manual_seed(0)
    det = ExitDetector()
    rgb, labels = _pad_frames(8)
    opt = torch.optim.Adam(det.parameters(), lr=2e-3)

    def loss_fn():
        out = det(rgb)
        seen = labels[:, 0] > 0
        return F.binary_cross_entropy_with_logits(out[:, 0], labels[:, 0]) + F.mse_loss(out[seen, 1:], labels[seen, 1:])

    first = float(loss_fn().detach())
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    with torch.inference_mode():
        out = det(rgb)
    assert float(loss_fn().detach()) < 0.1 * first
    assert torch.equal(out[:, 0] > 0, labels[:, 0] > 0)


# ------------------------------------------------------------------------------------------ rules, devices, speed

def test_model_source_respects_the_submission_rules():
    source = Path(model.__file__).read_text().lower()
    for token in ("atan2", "arctan2", "heapq", "deque", "astar", "dijkstra", "bfs", "flood", ".cuda(", "torch.cuda", 'to("cuda"'):
        assert token not in source, token
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            assert line.split()[1].split(".")[0] in ("__future__", "math", "torch", ""), line     # "" = relative import of spec
    assert Path(model.__file__).read_text().split('"""')[1].strip().splitlines()[-3].startswith("Purpose:")


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_mps_matches_cpu(net):
    """The learner's device must compute the same function, forward and backward, without a CPU fallback."""
    d = make_inputs(3, 12, seed=11, uint8_map=True)
    forced = {"subgoal": torch.randint(0, spec.NUM_SUBGOALS, (3, 12)), "target": torch.randint(0, 4096, (3, 12))}
    reset = _resets(3, 12)
    with torch.inference_mode():
        cpu, _ = sequence(net, d, forced=forced, reset=reset)
    gpu_net = PolicyNet(**net.config).to("mps")
    gpu_net.load_state_dict(net.state_dict())
    to = lambda x: {k: v.to("mps") for k, v in x.items()}                                   # noqa: E731
    out, _ = sequence(gpu_net, to(d), forced=to(forced), reset=reset.to("mps"))
    sum(l.sum() for l in out["logits"]).backward()
    assert float(gpu_net.vp.transition.grad.abs().sum()) > 0
    with torch.inference_mode():
        step_out, _ = step(gpu_net, to(at(d, 0)))                                           # sampling on the device
    assert step_out["target"].device.type == "mps"
    moved = {k: (tuple(l.detach().cpu() for l in v) if k == "logits" else v.detach().cpu()) for k, v in out.items()}
    assert_outputs_close(cpu, moved, atol=2e-3)
    det, rgb = ExitDetector(), _pad_frames(2)[0]
    with torch.inference_mode():
        ref = det(rgb)
        assert torch.allclose(ref, det.to("mps")(rgb.to("mps")).cpu(), atol=1e-3)


@pytest.mark.slow
def test_step_latency_is_inside_the_budget():
    """Both networks, batch 1, one thread: mean per step over manager and non-manager steps (best of 3 runs)."""
    from training.bench import model_latency as bench
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        with bench.ThreadMeter() as meter:
            best, means = bench.repeat_steps(PolicyNet().eval(), ExitDetector().eval(), steps=420, repeats=3)
    finally:
        torch.set_num_threads(before)
    assert meter.result["single_threaded"], meter.result                                   # otherwise the figure means nothing
    assert min(means) <= bench.BUDGET_MS, means
    assert best["PolicyNet, manager steps"]["n"] == 53 and best["PolicyNet, other steps"]["n"] == 367


def test_bench_pins_the_one_thread_variables_where_unset():
    from training.bench import model_latency as bench
    env = {"OMP_NUM_THREADS": "4", "HOME": "/x"}
    assert bench.pin_one_thread_env(env) == {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    assert env == {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "HOME": "/x"}
    before = {name: os.environ.get(name) for name in bench.ONE_THREAD_ENV}
    importlib.reload(bench)                                                                # importing it must not touch the environment
    assert before == {name: os.environ.get(name) for name in bench.ONE_THREAD_ENV}


def test_bench_thread_verdict():
    """The verdict on recorded cases: one working thread passes, a pool behind a one-thread torch setting does not."""
    from training.bench import model_latency as bench
    one = bench.judge_threads(wall_s=6.0, cpu_s=5.9, grew=[5.9, 0.0, 0.01])
    assert one["single_threaded"] and one["os_threads"] == 3 and one["busy_threads"] == 1
    # a 15-thread pool under a one-CPU quota: CPU over wall stays at 1.0, only the per-thread times show it
    pool = bench.judge_threads(wall_s=29.0, cpu_s=28.1, grew=[8.5] + [1.4] * 14 + [0.0] * 14)
    assert not pool["single_threaded"] and pool["os_threads"] == 29 and pool["busy_threads"] == 15
    assert not bench.judge_threads(wall_s=2.0, cpu_s=5.0)["single_threaded"]               # no per-thread times: CPU over wall decides
    assert bench.judge_threads(wall_s=2.0, cpu_s=1.2)["single_threaded"]                   # a loaded machine: less than one CPU
    with bench.ThreadMeter() as meter:
        sum(i * i for i in range(200_000))
    assert meter.result["single_threaded"] and meter.result["cpu_over_wall"] <= 1.1
    assert {"quota_cpus", "periods", "throttled_periods"} <= meter.result.keys()


def test_bench_script_runs_single_threaded():
    """Run as a script with the thread variables unset: it sets them before torch loads and reports one working thread."""
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    script = Path(__file__).resolve().parents[1] / "bench" / "model_latency.py"
    done = subprocess.run([sys.executable, str(script), "--steps", "105", "--repeats", "1", "--steps-only"],
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in done.stdout
    verdict = [line for line in done.stdout.splitlines() if line.strip().startswith("single-threaded")]
    assert len(verdict) == 1 and verdict[0].split()[-1] == "yes", done.stdout[-2000:]
    assert "NOT VALID" not in done.stdout


def test_bench_helpers_run(net):
    from training.bench import model_latency as bench
    counts = bench.count_parameters(net, ExitDetector())
    assert counts["both networks"] == counts["PolicyNet total"] + counts["ExitDetector"]
    d = bench.make_inputs(2, 3)
    assert d["map"].dtype == torch.uint8 and d["rgb"].shape == (2, 3, 3, 120, 160)
    stats = bench.time_steps(net, ExitDetector().eval(), steps=20)
    assert stats["step (both networks)"]["n"] == 20 and stats["PolicyNet, manager steps"]["n"] == 3
