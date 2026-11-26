"""Tests of the final-pick report (training/final_pick.py) on synthetic per-episode rows.

No episode is played: suites are small files of made-up grids in a temporary directory and the
per-episode files are written by hand. Covered: reading heavy / reference-only / exit from a grid
with the legend's symbols, the tier, light-map and suite weights of the headline, tier 1 counted
on exit maps only, averaging over seeds before averaging over maps, the paired verdicts with the
noise rule of `training.evaluate`, the decision rule (a challenger that wins the headline but is
clearly worse on one suite does not replace the holder), and what `score` hands to
`training.promote.score_local` and where its rows go.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest

from training import evaluate, final_pick as fp, promote
from training.mapgen.udmf import LEGEND
from training.scorer import suite_score as scorer_suite_score

COLUMNS = ("map_id", "tier", "seed", "score", "has_exit", "exit", "dead", "timeout", "kills", "monsters_total",
           "door_reversals", "agent_error", "error")
GRIDS = {"heavy": "#######\n#P.zkE#\n#######", "caco": "#######\n#P.o.E#\n#######", "ref": "#######\n#P.zicE#\n#######",
         "mixed": "#######\n#P.zsE#\n#######", "empty": "#####\n#P.E#\n#####", "noexit": "#####\n#Pz.#\n#####"}


def grid_row(name: str, tier: int, kind: str) -> dict:
    return {"name": name, "tier": tier, "episode_ticks": 2100, "grid": GRIDS[kind], "options": {}}


def write_suite(directory: Path, name: str, maps: list[dict], seeds=(0,), weight: float = 1.0) -> fp.Suite:
    path = directory / f"{name}.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in maps))
    return fp.Suite(name, str(path), tuple(seeds), weight)


def episode(map_id: str, tier: int, seed: int = 0, score: float = 50.0, **more) -> dict:
    row = {"map_id": map_id, "tier": tier, "seed": seed, "score": score, "has_exit": 1, "exit": 0, "dead": 0, "timeout": 0,
           "kills": 0, "monsters_total": 0, "door_reversals": 0, "agent_error": "", "error": ""}
    row.update(more)
    return row


def write_rows(runs_dir: Path, agent: str, suite: fp.Suite, rows: list[dict]) -> None:
    path = runs_dir / agent / f"{suite.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def four_suites(directory: Path, per_tier: int = 6) -> tuple:
    """Four suites shaped like the real ones (weights, one seed or five), `per_tier` maps in each of tiers 1-4."""
    suites = []
    for real in fp.SUITES:
        maps = [grid_row(f"{real.name}_t{tier}_{k:02d}", tier, "heavy" if tier == 4 and k % 2 else "ref")
                for tier in (1, 2, 3, 4) for k in range(per_tier)]
        suites.append(write_suite(directory, real.name, maps, real.seeds, real.weight))
    return tuple(suites)


def play(runs_dir: Path, agent: str, suites, score_of, seed: int = 1) -> None:
    """Rows for every map and seed of every suite; `score_of(suite name, map id, noise)` gives the composite."""
    for suite in suites:
        rng = random.Random(f"{seed}-{suite.name}")
        rows = [episode(m.map_id, m.tier, s, score_of(suite.name, m.map_id, rng.uniform(-1.0, 1.0)))
                for m in fp.load_maps(suite).values() for s in suite.seeds]
        write_rows(runs_dir, agent, suite, rows)


def base_score(suite: str, map_id: str, noise: float) -> float:
    """A composite that depends on the map only, so that two agents built from it are paired exactly."""
    return 30.0 + (sum(map(ord, map_id)) % 40)


# -------------------------------------------------------------------------------------- splits
def test_the_monster_symbols_come_from_the_legend():
    assert {LEGEND[c].name for c in fp.HEAVY_CHARS} == {"hell knight", "cacodemon"}
    assert {LEGEND[c].name for c in fp.REFERENCE_CHARS} == {"zombieman", "imp", "chaingunner"}
    assert fp.HEAVY_CHARS <= fp.MONSTER_CHARS and fp.REFERENCE_CHARS <= fp.MONSTER_CHARS
    assert not fp.HEAVY_CHARS & fp.REFERENCE_CHARS


@pytest.mark.parametrize("kind, has_exit, heavy, reference_only, monsters", [
    ("heavy", True, True, False, 2),      # a zombieman and a hell knight
    ("caco", True, True, False, 1),
    ("ref", True, False, True, 3),        # zombieman, imp and chaingunner
    ("mixed", True, False, False, 2),     # a shotgun guy: not heavy, and not a reference monster either
    ("empty", True, False, False, 0),     # no monsters: without a heavy one, but not `reference monsters only`
    ("noexit", False, False, True, 1),
])
def test_classify_reads_the_grid(kind, has_exit, heavy, reference_only, monsters):
    assert fp.classify(GRIDS[kind]) == {"has_exit": has_exit, "heavy": heavy, "reference_only": reference_only,
                                        "monsters": monsters}


def test_tier_one_counts_only_with_an_exit(tmp_path):
    suite = write_suite(tmp_path, "s", [grid_row("a", 1, "ref"), grid_row("b", 1, "noexit"), grid_row("c", 2, "noexit"),
                                        grid_row("d", 0, "ref")])
    maps = fp.load_maps(suite)
    assert [maps[k].counted for k in "abcd"] == [True, False, True, False]   # tier 0 carries no weight at all
    assert fp.map_weight(maps["b"]) == 0.0 and fp.map_weight(maps["a"]) == 1.0 * fp.LIGHT_MAP_WEIGHT


def test_the_real_suites_are_the_four_held_out_ones():
    assert [(s.name, len(fp.load_maps(s)), s.seeds) for s in fp.SUITES] == [
        ("heldout_v1", 400, (0,)), ("heldout_wide_v1", 200, (0,)),
        ("authored_v1", 59, (0, 1, 2, 3, 4)), ("authored_hard_v1", 60, (0, 1, 2, 3, 4))]
    assert [s.weight for s in fp.SUITES] == [0.35, 0.20, 0.20, 0.25]
    assert fp.LIGHT_MAP_WEIGHT == 3.0


# ------------------------------------------------------------------------------------- weights
def test_suite_score_weights_tiers_and_light_maps(tmp_path):
    suite = write_suite(tmp_path, "s", [grid_row("t4k", 4, "heavy"), grid_row("t4z", 4, "ref"), grid_row("t1z", 1, "ref"),
                                        grid_row("t1n", 1, "noexit")])
    maps = fp.load_maps(suite)
    values = {"t4k": 10.0, "t4z": 50.0, "t1z": 100.0, "t1n": 0.0}
    weighted, _, n = fp.suite_score(values, maps)
    assert n == 3                                                    # the tier 1 map without an exit is left out
    assert weighted == pytest.approx((4 * 10 + 12 * 50 + 3 * 100) / (4 + 12 + 3))
    plain, _, _ = fp.suite_score(values, maps, light_weight=1.0)
    assert plain == pytest.approx((4 * 10 + 4 * 50 + 1 * 100) / 9)


def test_headline_uses_the_suite_weights():
    per_suite = {"heldout_v1": (40.0, 1.0), "heldout_wide_v1": (50.0, 2.0), "authored_v1": (60.0, 3.0),
                 "authored_hard_v1": (20.0, 4.0)}
    value, se = fp.headline(per_suite)
    assert value == pytest.approx(0.35 * 40 + 0.20 * 50 + 0.20 * 60 + 0.25 * 20)
    assert se == pytest.approx(math.sqrt(0.35 ** 2 + (0.2 * 2) ** 2 + (0.2 * 3) ** 2 + 1.0))
    del per_suite["authored_v1"]
    assert math.isnan(fp.headline(per_suite)[0])                     # no headline without all four suites


def test_weighted_mean_matches_the_scorer_and_the_plain_standard_error():
    rng = random.Random(3)
    rows = [{"tier": t, "score": rng.uniform(0, 100)} for t in (1, 2, 3, 4) for _ in range(7)]
    theirs = scorer_suite_score(rows)
    mean, se, n = fp.weighted_mean([r["score"] for r in rows], [float(r["tier"]) for r in rows], [r["tier"] for r in rows])
    assert (mean, se, n) == (pytest.approx(theirs["score"]), pytest.approx(theirs["score_se"]), 28)
    values = [3.0, 5.0, 10.0]
    mean, se, _ = fp.weighted_mean(values)
    spread = math.sqrt(sum((v - 6.0) ** 2 for v in values) / 2)
    assert (mean, se) == (pytest.approx(6.0), pytest.approx(spread / math.sqrt(3)))
    assert fp.weighted_mean([float("nan"), 4.0])[::2] == (4.0, 1)     # an unknown value is left out


# ------------------------------------------------------------------------------ seed averaging
def test_seeds_are_averaged_per_map_before_maps_are_averaged(tmp_path):
    suite = write_suite(tmp_path, "s", [grid_row("a", 2, "ref"), grid_row("b", 2, "ref")], seeds=(0, 1, 2, 3, 4))
    rows = [episode("a", 2, s, 100.0 if s else 0.0, exit=int(s > 0), dead=int(s == 0), kills=3 if s else 0, monsters_total=3)
            for s in range(5)]
    rows += [episode("b", 2, s, 50.0, timeout=1, kills=1, monsters_total=3, door_reversals=2) for s in range(5)]
    write_rows(tmp_path, "agent", suite, rows)
    run = fp.load_run("agent", (suite,), runs_dir=tmp_path)
    assert run.complete and not run.notes
    maps = fp.load_maps(suite)
    means = {k: fp.map_means(seeds) for k, seeds in run.episodes["s"].items()}
    assert means["a"]["score"] == pytest.approx(80.0) and means["b"]["score"] == pytest.approx(50.0)
    stats = fp.cell_stats(means, maps, ["a", "b"])
    assert stats["maps"] == 2
    # two maps of 80 and 50: the standard error is the one between maps (15), not the one of ten episodes (about 10.7)
    assert stats["score"] == (pytest.approx(65.0), pytest.approx(15.0), 2)
    assert stats["exit"][:2] == (pytest.approx(0.4), pytest.approx(0.4))
    assert stats["dead"][0] == pytest.approx(0.1) and stats["timeout"][0] == pytest.approx(0.5)
    assert stats["door_reversals"][0] == pytest.approx(1.0)
    assert stats["kills"][0] == pytest.approx((2.4 + 1.0) / 6.0)       # all kills over all monsters


def test_rows_that_do_not_belong_and_missing_rows_are_reported(tmp_path):
    suite = write_suite(tmp_path, "s", [grid_row("a", 2, "ref"), grid_row("b", 2, "ref")], seeds=(0, 1))
    write_rows(tmp_path, "agent", suite, [episode("a", 2, 0), episode("a", 2, 1), episode("a", 2, 7), episode("zz", 2, 0),
                                          episode("b", 2, 0, 0.0, error="engine crashed")])
    run = fp.load_run("agent", (suite,), runs_dir=tmp_path)
    assert not run.complete
    assert sorted(run.episodes["s"]["a"]) == [0, 1] and "zz" not in run.episodes["s"]
    text = " | ".join(run.notes)
    assert "3 of 4 episodes" in text and "other seeds" in text and "not in" in text and "could not be played" in text
    assert not fp.load_run("nobody", (suite,), runs_dir=tmp_path).complete


# ------------------------------------------------------------------------------ paired verdicts
@pytest.mark.parametrize("delta, se, expected", [
    (1.5, 0.1, fp.BETTER),        # over one point and over two standard errors
    (0.9, 0.0, fp.LEVEL),         # under one point is level whatever the statistics say
    (1.5, 1.0, fp.LEVEL),         # over one point but inside two standard errors
    (-3.0, 1.0, fp.WORSE),
    (-1.9, 1.0, fp.LEVEL),
    (float("nan"), 1.0, fp.UNKNOWN),
])
def test_verdict_uses_the_noise_rule_of_evaluate(delta, se, expected):
    v = fp.verdict(delta, se)
    assert v["verdict"] == expected
    if expected != fp.UNKNOWN:
        assert v["margin"] == max(evaluate.NOISE_POINTS, evaluate.NOISE_SIGMAS * se)


def test_paired_difference_is_taken_on_shared_maps_and_seeds(tmp_path):
    suite = write_suite(tmp_path, "s", [grid_row("a", 2, "ref"), grid_row("b", 2, "ref"), grid_row("c", 2, "ref")], seeds=(0, 1))
    write_rows(tmp_path, "A", suite, [episode("a", 2, 0, 10.0), episode("a", 2, 1, 30.0), episode("b", 2, 0, 50.0),
                                      episode("b", 2, 1, 50.0), episode("c", 2, 0, 0.0)])
    write_rows(tmp_path, "B", suite, [episode("a", 2, 0, 20.0), episode("a", 2, 1, 90.0), episode("b", 2, 0, 56.0)])
    a, b = (fp.load_run(name, (suite,), runs_dir=tmp_path) for name in "AB")
    assert fp.paired_differences(a, b, "s") == {"a": pytest.approx(35.0), "b": pytest.approx(6.0)}   # b: seed 0 only


def test_paired_verdicts_better_worse_and_level(tmp_path):
    suites = four_suites(tmp_path)
    runs = tmp_path / "runs"
    play(runs, "A", suites, base_score)
    play(runs, "up", suites, lambda s, m, noise: base_score(s, m, 0) + 5.0 + noise)
    play(runs, "down", suites, lambda s, m, noise: base_score(s, m, 0) - 5.0 + noise)
    play(runs, "same", suites, lambda s, m, noise: base_score(s, m, 0) + 0.5)
    play(runs, "noisy", suites, lambda s, m, noise: base_score(s, m, 0) + 2.0 + 40.0 * noise)
    maps = {s.name: fp.load_maps(s) for s in suites}
    load = lambda name: fp.load_run(name, suites, maps, runs)  # noqa: E731
    a = load("A")
    up = fp.paired(a, load("up"), maps, suites)
    assert up["headline"]["verdict"] == fp.BETTER and up["headline"]["delta"] == pytest.approx(5.0, abs=0.5)
    assert all(v["verdict"] == fp.BETTER for v in up["suites"].values())
    assert up["maps"] == {s.name: 24 for s in suites}
    down = fp.paired(a, load("down"), maps, suites)
    assert down["headline"]["verdict"] == fp.WORSE and all(v["verdict"] == fp.WORSE for v in down["suites"].values())
    same = fp.paired(a, load("same"), maps, suites)
    assert same["headline"]["delta"] == pytest.approx(0.5) and same["headline"]["se"] == pytest.approx(0.0)
    assert same["headline"]["verdict"] == fp.LEVEL                   # half a point is under the one-point floor
    noisy = fp.paired(a, load("noisy"), maps, suites)
    assert noisy["headline"]["se"] > 1.0 and noisy["headline"]["verdict"] == fp.LEVEL
    # the paired headline difference is the difference of the two headlines when both played everything
    scores_a, scores_b = fp.agent_scores(a, maps, suites), fp.agent_scores(load("up"), maps, suites)
    assert up["headline"]["delta"] == pytest.approx(scores_b["headline"][0] - scores_a["headline"][0])


# ------------------------------------------------------------------------------- decision rule
def v(kind: str, delta: float = 0.0) -> dict:
    return {"delta": delta, "se": 0.5, "margin": 1.0, "verdict": kind}


@pytest.mark.parametrize("head, suites, complete, replace", [
    (v(fp.BETTER, 3.0), [fp.BETTER, fp.LEVEL, fp.LEVEL, fp.BETTER], True, True),
    (v(fp.BETTER, 3.0), [fp.BETTER, fp.BETTER, fp.WORSE, fp.BETTER], True, False),   # clearly worse on one suite
    (v(fp.LEVEL, 0.8), [fp.BETTER, fp.LEVEL, fp.LEVEL, fp.LEVEL], True, False),      # the headline has to win
    (v(fp.WORSE, -3.0), [fp.LEVEL] * 4, True, False),
    (v(fp.BETTER, 3.0), [fp.LEVEL] * 4, False, False),                              # not scored on everything
    (v(fp.BETTER, 3.0), [fp.LEVEL, fp.LEVEL, fp.LEVEL, fp.UNKNOWN], True, False),
    (v(fp.UNKNOWN, float("nan")), [fp.LEVEL] * 4, True, False),
])
def test_decide(head, suites, complete, replace):
    per_suite = {s.name: v(kind, -2.0 if kind == fp.WORSE else 0.0) for s, kind in zip(fp.SUITES, suites)}
    decision, why = fp.decide(head, per_suite, complete)
    assert decision is replace and why
    if fp.WORSE in suites and head["verdict"] == fp.BETTER and complete:
        assert "authored_v1" in why


def test_winning_the_headline_but_clearly_worse_on_one_suite_does_not_replace(tmp_path):
    suites = four_suites(tmp_path)
    runs = tmp_path / "runs"
    play(runs, "holder", suites, base_score)
    # +10 everywhere except authored_v1 (weight 0.20), where the challenger loses 8 points on every map
    play(runs, "lopsided", suites, lambda s, m, noise: base_score(s, m, 0) + (-8.0 if s == "authored_v1" else 10.0) + noise)
    play(runs, "sound", suites, lambda s, m, noise: base_score(s, m, 0) + (0.3 if s == "authored_v1" else 10.0) + noise)
    said = []
    numbers = fp.report(["holder", "lopsided"], suites, runs, echo=said.append)
    result = numbers["pairs"][("holder", "lopsided")]
    assert result["headline"]["verdict"] == fp.BETTER and result["headline"]["delta"] == pytest.approx(6.4, abs=0.5)
    assert result["suites"]["authored_v1"]["verdict"] == fp.WORSE
    assert numbers["decisions"] == [{"a": "holder", "b": "lopsided", "replace": False, "why": numbers["decisions"][0]["why"]}]
    assert "worse by more than noise on authored_v1" in numbers["decisions"][0]["why"]
    text = "\n".join(said)
    assert "DECISION (A = holder, B = lopsided): KEEP holder" in text and "FINAL PICK: holder" in text
    assert numbers["file"] == runs / "report_holder_vs_lopsided.txt" and "FINAL PICK: holder" in numbers["file"].read_text()

    numbers = fp.report(["holder", "sound"], suites, runs, echo=lambda line: None)
    assert numbers["pairs"][("holder", "sound")]["suites"]["authored_v1"]["verdict"] == fp.LEVEL
    assert numbers["decisions"][0]["replace"] is True

    # three names: the holder changes hands, and the next one challenges the new holder
    lines, numbers = fp.build_report(["holder", "sound", "lopsided"], suites, runs)
    assert [(d["a"], d["b"], d["replace"]) for d in numbers["decisions"]] == [("holder", "sound", True), ("sound", "lopsided", False)]
    assert lines.count("FINAL PICK: sound") == 1


def test_an_incomplete_challenger_does_not_replace(tmp_path):
    suites = four_suites(tmp_path)
    runs = tmp_path / "runs"
    play(runs, "holder", suites, base_score)
    play(runs, "partial", suites, lambda s, m, noise: base_score(s, m, 0) + 20.0)
    (runs / "partial" / "authored_hard_v1.csv").unlink()
    lines, numbers = fp.build_report(["holder", "partial"], suites, runs)
    assert numbers["decisions"][0]["replace"] is False
    assert math.isnan(numbers["weighted"][1]["headline"][0])
    assert any("authored_hard_v1: no rows" in line for line in lines)


# -------------------------------------------------------------------------------------- report
def test_report_prints_the_splits_the_counts_and_tier_one_on_exit_maps(tmp_path):
    maps = ([grid_row(f"t1e{k}", 1, "ref") for k in range(3)] + [grid_row(f"t1n{k}", 1, "noexit") for k in range(2)]
            + [grid_row(f"t4k{k}", 4, "heavy") for k in range(12)] + [grid_row(f"t4s{k}", 4, "mixed") for k in range(11)]
            + [grid_row("t4z", 4, "ref")])
    suite = write_suite(tmp_path, "only", maps, weight=1.0)
    rows = []
    for m in fp.load_maps(suite).values():
        score = 0.0 if not m.has_exit else 20.0 if m.heavy else 80.0
        rows.append(episode(m.map_id, m.tier, 0, score, has_exit=int(m.has_exit), exit=int(score > 50), dead=int(m.heavy),
                            kills=1, monsters_total=m.monsters))
    write_rows(tmp_path / "runs", "agent", suite, rows)
    lines, numbers = fp.build_report(["agent"], (suite,), tmp_path / "runs")
    counts = numbers["counts"]
    assert counts[("only", "1")] == {"maps": 5, "exit": 3, "no exit": 2, "counted": 3, "heavy": 0, "no heavy": 3,
                                     "ref only": 3, "no monst": 0}
    assert counts[("only", "4")] == {"maps": 24, "exit": 24, "no exit": 0, "counted": 24, "heavy": 12, "no heavy": 12,
                                     "ref only": 1, "no monst": 0}
    by_label = {tuple(line.split()[:3]): line.split() for line in lines if line.startswith("only ")}
    assert by_label[("only", "1", "exit")][4:7] == ["3*", "agent", "80.00"]        # tier 1 on exit maps, a thin cell
    assert by_label[("only", "1", "all")][4:7] == ["5*", "agent", "48.00"]         # ... and on all tier 1 maps
    assert by_label[("only", "4", "heavy")][3:6] == ["12", "agent", "20.00"]
    assert by_label[("only", "4", "no")][4:7] == ["12", "agent", "80.00"]
    assert by_label[("only", "4", "ref")][4:7] == ["1*", "agent", "80.00"]
    # headline: light maps three times a heavy map inside tier 4, tiers 1 and 4 weighted 1 and 4, no-exit maps left out
    weighted = (3 * 3 * 80 + 12 * 4 * 20 + 12 * 12 * 80) / (9 + 48 + 144)
    plain = (3 * 80 + 12 * 4 * 20 + 12 * 4 * 80) / (3 + 96)
    assert numbers["weighted"][0]["headline"][0] == pytest.approx(weighted)
    assert numbers["plain"][0]["headline"][0] == pytest.approx(plain)
    assert any(line.startswith("headline") and f"{weighted:.2f}" in line and f"{plain:.2f}" in line for line in lines)
    assert any("fewer than 10 maps" in line and "tier 1: 3 counted" in line for line in lines)
    assert not any("DECISION" in line for line in lines)                            # one agent: nothing to decide


# --------------------------------------------------------------------------------------- score
def test_score_plays_every_suite_through_promote_and_logs_only_on_request(tmp_path, monkeypatch):
    calls, logged = [], []

    def fake_score_local(candidate, tag, suite, **kw):
        calls.append((Path(candidate), tag, suite, kw["workers"], tuple(kw["seeds"]), Path(kw["out"]), Path(kw["episodes_dir"])))
        return [{"tag": tag, "tier": "1", "episodes": 2}, {"tag": tag, "tier": "all", "episodes": 2 * len(kw["seeds"])}]

    home = tmp_path / "cand"
    (home / "submission" / "weights").mkdir(parents=True)
    (home / "submission" / "agent.py").write_text("")
    (home / "submission" / "weights" / "policy.pt").write_bytes(b"policy")
    monkeypatch.setattr(promote, "score_local", fake_score_local)
    monkeypatch.setattr(promote, "build_candidate", lambda *a, **k: pytest.fail("a candidate directory is played as it is"))
    monkeypatch.setattr(evaluate, "append_results", lambda path, rows: logged.append((Path(path), rows)))
    facts = fp.score(home, "name", workers=4, runs_dir=tmp_path / "final", echo=lambda line: None)
    target = (tmp_path / "final").resolve() / "name"
    assert calls == [(home, s.name, s.file, 4, s.seeds, target / "results.csv", target) for s in fp.SUITES]
    assert logged == []                                               # nothing reaches training/logs/results.csv
    assert facts["episodes"] == 2 + 2 + 10 + 10 and facts["workers"] == 4 and facts["partial"] is False
    assert json.loads((target / "score.json").read_text())["suites"]["authored_v1"]["seeds"] == [0, 1, 2, 3, 4]

    fp.score(home, "name", workers=4, runs_dir=tmp_path / "final", log=True, limit=1, echo=lambda line: None)
    assert [path for path, _ in logged] == [evaluate.RESULTS_CSV] * 4
    assert {row["tag"] for _, rows in logged for row in rows} == {f"name:{s.name}" for s in fp.SUITES}
    assert json.loads((target / "score.json").read_text())["partial"] is True

    built = []
    monkeypatch.setattr(promote, "build_candidate", lambda source, name, check, echo=print: built.append((Path(source), name, check)))
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"x")
    assert fp.candidate(weights, "fresh", "strict") == promote.CANDIDATES / "fresh"
    assert built == [(weights, "fresh", "strict")]
    with pytest.raises(SystemExit):
        fp.candidate(tmp_path, "fresh")                               # a directory without an agent in it
