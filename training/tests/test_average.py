"""Tests of the two tools behind a blended agent: training/rl/average.py and training/rl/mix_manifests.py.

No network of the agent is built and no map is played: the state dicts are tiny synthetic ones and
the manifests are made-up rows in a temporary directory. Covered: the weighted average itself
(float64 sum, float32 result, weights divided by their sum, equal weights by default), what is
taken from the last file and what is dropped, the refusals (another architecture, other tensor
names or shapes, an integer tensor, a weight that is not positive, a file that is not a
checkpoint), and the manifest mix (all of the first set, the same count per tier from the second,
a tier the first set lacks left out, the same draw for the same seed, rows written back unchanged).
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch

from training.rl import average, mix_manifests

CONFIG = {"hidden": 4, "vp_iters": 2}


def state(seed: int, hidden: int = 4) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"body.weight": torch.randn(hidden, 3, generator=g), "body.bias": torch.randn(hidden, generator=g),
            "head.weight": torch.randn(2, hidden, generator=g)}


def checkpoint(seed: int, iteration: int, config: dict | None = None, **extra) -> dict:
    return {"format": "doomnet-policy-v1", "model": state(seed), "policy_config": dict(config or CONFIG),
            "policy_kind": "PolicyNet", "iteration": iteration, "env_steps": iteration * 1000,
            "optimizer": {"state": "of one run"}, "curriculum": {"stage_index": 1}, "distill": {"index": 3}, **extra}


# ------------------------------------------------------------------------------------- the average
def test_equal_weights_give_the_mean_in_float32():
    a, b = state(1), state(2)
    out = average.average_state([a, b])
    assert list(out) == list(a)
    for name in a:
        assert out[name].dtype == torch.float32 and out[name].shape == a[name].shape
        assert torch.equal(out[name], (0.5 * a[name].double() + 0.5 * b[name].double()).float())
    assert not torch.equal(out["body.weight"], a["body.weight"]) and not torch.equal(out["body.weight"], b["body.weight"])


def test_weights_are_divided_by_their_sum_and_summed_in_float64():
    a, b, c = state(1), state(2), state(3)
    out = average.average_state([a, b, c], [1, 1, 2])
    for name in a:
        expected = (0.25 * a[name].double() + 0.25 * b[name].double() + 0.5 * c[name].double()).float()
        assert torch.equal(out[name], expected)
    assert average.normalised([3, 1], 2) == [0.75, 0.25]
    assert average.normalised(None, 4) == [0.25] * 4
    # the sum is not taken in float32: over 20,000 values a float32 sum of three thirds rounds differently somewhere
    g = torch.Generator().manual_seed(0)
    wide = [{"w": torch.randn(20000, generator=g)} for _ in range(3)]
    third = torch.tensor(1.0 / 3.0)
    in_float32 = third * wide[0]["w"] + third * wide[1]["w"] + third * wide[2]["w"]
    in_float64 = sum((1.0 / 3.0) * t["w"].double() for t in wide).float()
    got = average.average_state(wide)["w"]
    assert torch.equal(got, in_float64) and not torch.equal(got, in_float32)


def test_one_file_with_weight_one_is_a_copy():
    a = state(5)
    out = average.average_state([a], [7.0])
    assert all(torch.equal(out[name], a[name]) for name in a)


def test_merge_takes_counters_from_the_last_file_and_drops_training_state():
    old, new = checkpoint(1, 1675), checkpoint(2, 2000)
    out = average.merge([old, new], [0.5, 0.5], ["branch.pt", "main.pt"])
    assert out["iteration"] == 2000 and out["env_steps"] == 2_000_000 and out["policy_config"] == CONFIG
    assert out["policy_kind"] == "PolicyNet" and out["format"] == "doomnet-policy-v1"
    assert not {"optimizer", "curriculum", "distill", "norm", "config"} & set(out)
    assert out["averaged_from"] == [{"file": "branch.pt", "iteration": 1675, "env_steps": 1_675_000, "weight": 0.5},
                                    {"file": "main.pt", "iteration": 2000, "env_steps": 2_000_000, "weight": 0.5}]
    assert torch.equal(out["model"]["head.weight"],
                       (0.5 * old["model"]["head.weight"].double() + 0.5 * new["model"]["head.weight"].double()).float())
    swapped = average.merge([new, old], [0.5, 0.5])
    assert swapped["iteration"] == 1675                              # the order decides the counters ...
    assert all(torch.equal(swapped["model"][k], out["model"][k]) for k in out["model"])     # ... not a two-file average


# -------------------------------------------------------------------------------------- refusals
def test_another_architecture_is_refused():
    with pytest.raises(ValueError, match="another architecture"):
        average.merge([checkpoint(1, 10), checkpoint(2, 20, config={"hidden": 8, "vp_iters": 2})])
    with pytest.raises(ValueError, match="another architecture"):
        average.merge([checkpoint(1, 10), checkpoint(2, 20, policy_kind="StandInPolicy")])


def test_other_tensors_shapes_and_types_are_refused():
    a = state(1)
    with pytest.raises(ValueError, match="other tensors"):
        average.average_state([a, {k: v for k, v in state(2).items() if k != "head.weight"}])
    with pytest.raises(ValueError, match="other tensors"):
        average.average_state([a, dict(state(2), extra=torch.zeros(1))])
    with pytest.raises(ValueError, match="body.weight"):
        average.average_state([a, state(2, hidden=8)])
    with pytest.raises(ValueError, match="body.bias"):
        average.average_state([a, dict(state(2), **{"body.bias": state(2)["body.bias"].double()})])
    counted = dict(a, steps=torch.tensor([3]))
    with pytest.raises(ValueError, match="floating-point"):
        average.average_state([counted, dict(counted)])
    with pytest.raises(ValueError, match="not finite"):
        average.average_state([a, dict(state(2), **{"body.bias": torch.full((4,), float("nan"))})])


def test_bad_weights_and_files_that_are_not_checkpoints_are_refused():
    a, b = state(1), state(2)
    for weights in ([1.0], [1.0, 0.0], [1.0, -1.0], [1.0, float("inf")], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            average.average_state([a, b], weights)
    with pytest.raises(ValueError, match="no state dict"):
        average.average_state([])
    exported = {"state_dict": state(1), "config": {"policy_kwargs": CONFIG}}
    with pytest.raises(ValueError, match="not a checkpoint"):
        average.merge([checkpoint(1, 10), exported])
    with pytest.raises(ValueError, match="not a checkpoint"):
        average.merge([checkpoint(1, 10), {"model": state(2), "policy_config": CONFIG, "policy_kind": "PolicyNet"}])


def test_command_line_refuses_a_decoding_setting_without_a_policy_file(tmp_path):
    with pytest.raises(SystemExit):
        average.main([str(tmp_path / "out.pt"), str(tmp_path / "a.pt"), "--greedy-heads", "move"])
    assert not (tmp_path / "out.pt").exists()


# ---------------------------------------------------------------------------------- manifest mix
def rows(prefix: str, per_tier: dict) -> list[dict]:
    return [{"name": f"{prefix}_t{tier}_{k:04d}", "path": f"data/maps/{prefix}/t{tier}_{k:04d}.wad", "tier": tier,
             "episode_ticks": 2100, "note": "café"} for tier, count in per_tier.items() for k in range(count)]


def write(path, items: list[dict]) -> str:
    path.write_text("".join(json.dumps(item, ensure_ascii=False, separators=(", ", ": ")) + "\n" for item in items), encoding="utf-8")
    return str(path)


def test_mix_takes_all_of_the_first_set_and_the_same_count_per_tier_from_the_second():
    first = rows("ref", {1: 3, 2: 5, 3: 5, 4: 7})
    second = rows("main", {0: 6, 1: 8, 2: 9, 3: 11, 4: 11})
    out = mix_manifests.mix(first, second, seed=4321)
    assert out[:len(first)] == first and len(out) == 2 * len(first)
    added = out[len(first):]
    assert Counter(r["tier"] for r in added) == Counter(r["tier"] for r in first) == {1: 3, 2: 5, 3: 5, 4: 7}
    assert all(r["name"].startswith("main_") for r in added) and len({r["name"] for r in added}) == len(added)
    assert not any(r["tier"] == 0 for r in out)                      # a tier the first set lacks is left out
    assert [second.index(r) for r in added] == sorted(second.index(r) for r in added)       # kept in the second set's order
    assert mix_manifests.mix(first, second, seed=4321) == out
    assert mix_manifests.mix(first, second, seed=1) != out


def test_mix_refuses_a_short_tier_and_a_repeated_map():
    first = rows("ref", {1: 3, 2: 5})
    with pytest.raises(ValueError, match="tier 2"):
        mix_manifests.mix(first, rows("main", {1: 8, 2: 4}))
    with pytest.raises(ValueError, match="twice"):
        mix_manifests.mix(first, rows("ref", {1: 3, 2: 5}))


def test_command_line_writes_the_rows_unchanged(tmp_path, capsys):
    first, second = rows("ref", {1: 2, 4: 3}), rows("main", {0: 2, 1: 4, 2: 4, 4: 5})
    a, b, out = write(tmp_path / "a.jsonl", first), write(tmp_path / "b.jsonl", second), tmp_path / "mix" / "manifest.jsonl"
    assert mix_manifests.main([str(out), a, b, "--seed", "4321"]) == 0
    written = out.read_text(encoding="utf-8").splitlines()
    source = set((tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()) | set(
        (tmp_path / "b.jsonl").read_text(encoding="utf-8").splitlines())
    assert len(written) == 10 and set(written) <= source and written[:5] == (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert Counter(json.loads(line)["tier"] for line in written) == {1: 4, 4: 6}
    assert "10 maps" in capsys.readouterr().out
    (tmp_path / "bad.jsonl").write_text('{"name": "x"}\n')
    with pytest.raises(SystemExit, match="no tier"):
        mix_manifests.main([str(tmp_path / "never.jsonl"), str(tmp_path / "bad.jsonl"), b])
    assert not (tmp_path / "never.jsonl").exists()
