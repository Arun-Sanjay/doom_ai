"""Door test: a policy must walk a corridor with three doors without ever reversing one.

A door that is rising turns round and closes again if USE is pressed a second time during its
first steps, and the agent then stands in front of a shut door. This file compiles a corridor
with three ordinary doors between the start and the exit, plays a policy on it through
`training.runner` (so the episode is scored by `training.scorer`), and fails if

* the scorer counts any door reversal, or
* the policy does not pass all three doors (the exit lies behind the third) within the tick cap.

The test is parametrised over policy specs (`"module:factory"`, as the runner takes them):

* `HoldUsePolicy`, a scripted reference that lives in this file. It verifies the harness: it holds
  USE exactly while `spec.action_masks` allows it, which is what the agent's USE head does when it
  wants the door open, and it must pass.
* `training.runner:AgentPolicy`, the submitted agent. It is skipped until trained weights exist.
* any further specs listed in the environment variable DOOMNET_DOOR_POLICIES (separated by `;`).

A second scripted policy, `DoublePressPolicy`, presses USE, lets go and presses again. The harness
must catch it whenever the second press comes while the door still reads as a door (measured here:
released for 1 to 5 steps); so must a policy that never moves.

The scripted policies are test code: hand-written rules are not allowed in the agent, and nothing
under submission/ imports this file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from training import paths  # noqa: F401  (puts submission/ on sys.path)
from doomnet import spec
from training import runner
from training.mapgen import udmf

ROOT = Path(__file__).resolve().parents[2]
CORRIDOR = "\n".join([
    "#################",
    "#P..D...D...D..E#",
    "#################",
])
DOORS = CORRIDOR.count("D")
TICK_CAP = 2100                                      # the shortest cap of any evaluation map
SHORT_CAP = 400                                      # for policies that are meant to fail (the reference needs 220 tics)
SEEDS = (0, 1)
WEIGHTS = ROOT / "submission" / "weights" / "policy.pt"


# ---------------------------------------------------------------------------------------------
# Scripted reference policies (test code only)
# ---------------------------------------------------------------------------------------------
def use_allowed(obs: dict) -> bool:
    """The agent's own USE mask: a door straight ahead within USE reach."""
    rays, ents = obs["rays"], obs["visible_entities"]
    return spec.action_masks(rays["distance"], rays["hit_type"], ents["entity_type"], ents["mask"], obs["vitals"][2])[0]


class HoldUsePolicy:
    """Walks forward and keeps USE down for as long as the mask allows it. Holding USE never reverses a door."""

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        action = np.zeros(9, np.int8)
        action[0] = 1
        action[7] = use_allowed(obs)
        return action


class DoublePressPolicy:
    """Walks forward, presses USE at a door, lets go for `gap` steps and presses again: the door reverses."""

    def __init__(self, gap: int = 2):
        self.gap = int(gap)
        self.since_press = None                      # steps since the first press at the current door

    def reset(self) -> None:
        self.since_press = None

    def act(self, obs: dict) -> np.ndarray:
        action = np.zeros(9, np.int8)
        action[0] = 1
        if not use_allowed(obs):
            self.since_press = None
        elif self.since_press is None:
            action[7], self.since_press = 1, 0
        else:
            self.since_press += 1
            action[7] = int(self.since_press == self.gap + 1)
        return action


class StandStillPolicy:
    """Never presses anything."""

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        return np.zeros(9, np.int8)


# ---------------------------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def corridor(tmp_path_factory) -> dict:
    """The three-door corridor, compiled by the map generator, as a runner map entry."""
    wad = udmf.write_wad(CORRIDOR, tmp_path_factory.mktemp("doors") / "three_doors.wad")
    return {"id": "three_doors", "wad": str(wad), "tier": 0, "episode_ticks": TICK_CAP}


def play(policy_spec, corridor: dict, seeds=SEEDS, ticks: int = TICK_CAP) -> list[dict]:
    """One scored episode per seed, in this process. `ticks` shortens the cap for policies that are meant to fail."""
    return runner.run_episodes(policy_spec, [dict(corridor, episode_ticks=ticks)], workers=0, seeds=seeds)


def door_problems(row: dict) -> list[str]:
    """What is wrong with one episode on the corridor, as text; empty when the policy passed."""
    problems = []
    for key in ("error", "agent_error"):
        if row[key]:
            problems.append(f"{key}: {row[key]}")
    if row["doors_total"] != DOORS:
        problems.append(f"the map has {row['doors_total']} doors, not {DOORS}")
    if row["door_reversals"]:
        problems.append(f"{row['door_reversals']} door reversal(s): USE was pressed again while a door was rising")
    if row["doors_opened"] < DOORS or not row["exit"]:
        problems.append(f"passed {row['doors_opened']} of {DOORS} doors and {'reached' if row['exit'] else 'did not reach'} "
                        f"the exit behind the third within {row['episode_ticks']} tics (stopped at tic {row['tic']})")
    return problems


def agent_weights_present() -> bool:
    return WEIGHTS.is_file() and not WEIGHTS.read_bytes()[:64].startswith(b"version https://git-lfs")


REFERENCE = f"{__name__}:HoldUsePolicy"
AGENT = "training.runner:AgentPolicy"
EXTRA = [s.strip() for s in os.environ.get("DOOMNET_DOOR_POLICIES", "").split(";") if s.strip()]


@pytest.mark.parametrize("policy_spec", [REFERENCE, AGENT, *EXTRA])
def test_three_doors_no_reversal(policy_spec, corridor, capsys):
    """The gate: no door reversal, all three doors passed, on every seed."""
    if policy_spec == AGENT and not agent_weights_present():
        pytest.skip("submission/weights/policy.pt does not exist yet: the trained agent is tested once it does")
    rows = play(policy_spec, corridor)
    assert len(rows) == len(SEEDS)
    with capsys.disabled():
        print(f"\n[doors] {policy_spec}: " + "; ".join(
            f"seed {r['seed']}: {r['doors_opened']}/{r['doors_total']} doors, {r['door_reversals']} reversals, "
            f"exit {r['exit']} at tic {r['tic']}" for r in rows))
    failures = {r["seed"]: door_problems(r) for r in rows if door_problems(r)}
    assert not failures, f"{policy_spec} fails the door test: {failures}"


# ---------------------------------------------------------------------------------------------
# The harness itself
# ---------------------------------------------------------------------------------------------
def test_corridor_has_three_doors_between_start_and_exit(corridor):
    row = play(f"{__name__}:StandStillPolicy", corridor, seeds=[0], ticks=40)[0]
    assert (row["doors_total"], row["locked_total"], row["has_exit"], row["monsters_total"]) == (DOORS, 0, 1, 0) and DOORS == 3
    line = CORRIDOR.split("\n")[1]
    assert line.index("P") < line.index("D") and line.rindex("D") < line.index("E")
    assert set(CORRIDOR.split("\n")[0]) == set(CORRIDOR.split("\n")[2]) == {"#"}       # no way round a door


def test_reference_policy_holds_use_and_passes(corridor):
    """The scripted reference keeps USE down for several steps in a row at every door and still reverses none."""
    presses = []

    class Recording(HoldUsePolicy):
        def act(self, obs):
            action = super().act(obs)
            presses.append(int(action[7]))
            return action

    row = runner.run_episodes(Recording(), [corridor], workers=0, seeds=[0])[0]
    assert door_problems(row) == []
    assert (row["doors_opened"], row["door_reversals"], row["exit"]) == (3, 0, 1) and row["tic"] < TICK_CAP
    runs = "".join(map(str, presses)).split("0")
    assert sum(1 for r in runs if r) == DOORS and all(len(r) >= 3 for r in runs if r)    # three holds, none a single tap


def test_double_press_is_caught(corridor):
    """Two presses a few steps apart reverse the door; the harness must say so."""
    rows = play(f"{__name__}:DoublePressPolicy", corridor, ticks=SHORT_CAP)
    for row in rows:
        problems = door_problems(row)
        assert row["door_reversals"] >= 1
        assert any("door reversal" in p for p in problems), problems
    with pytest.raises(AssertionError, match="fails the door test"):
        failures = {r["seed"]: door_problems(r) for r in rows if door_problems(r)}
        assert not failures, f"DoublePressPolicy fails the door test: {failures}"


@pytest.mark.parametrize("gap", [1, 2, 3, 4, 5])
def test_any_second_press_while_the_door_still_reads_as_a_door_reverses(corridor, gap):
    """USE released for 1 to 5 steps and pressed again: the door turns round every time."""
    row = play((f"{__name__}:DoublePressPolicy", {"gap": gap}), corridor, seeds=[0], ticks=SHORT_CAP)[0]
    assert row["door_reversals"] >= 1, (gap, row["doors_opened"])
    assert door_problems(row)


@pytest.mark.parametrize("gap", [0, 6])
def test_no_reversal_without_a_second_press(corridor, gap):
    """gap 0 is two steps of USE in a row, which is holding; after 6 steps the rays no longer show a
    door, the mask forbids USE and the second press never happens. Both pass."""
    row = play((f"{__name__}:DoublePressPolicy", {"gap": gap}), corridor, seeds=[0])[0]
    assert door_problems(row) == [], gap


def test_not_passing_the_doors_is_caught(corridor):
    """A policy that reverses nothing but does not get through fails as well."""
    row = play(f"{__name__}:StandStillPolicy", corridor, seeds=[0], ticks=SHORT_CAP)[0]
    problems = door_problems(row)
    assert row["door_reversals"] == 0 and row["doors_opened"] == 0 and not row["exit"]
    assert len(problems) == 1 and "passed 0 of 3 doors" in problems[0]
