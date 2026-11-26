"""Tests of the checkpoint promotion pipeline (training/promote.py, training/ops/remote.py).

Fast tests cover the decision rule (with the one-point tie band and the check of `--more-rl`
against the candidates' records), the comparison on synthetic results, how a candidate directory
is built and what the export refuses, the comparison of a candidate with the working tree, merging
rows of a remote run, and the guards of the remote helper (no network is used: the ssh call is
replaced). The slow test runs the whole ship gate on a small candidate, fresh and then stale.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from training import evaluate, paths, promote
from training.ops import remote
from training.runner import ROW_FIELDS
from doomnet.model import PolicyNet

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def torch_threads():
    """Building a candidate starts a DoomAgent, which sets torch to one thread for the whole process."""
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def checkpoint(path: Path, record: bool, seed: int = 5) -> Path:
    """A small checkpoint in the layout of training/rl/train.py, with or without a provenance record."""
    from training.rl import provenance

    torch.manual_seed(seed)
    net = PolicyNet(hidden=64, vp_iters=2)
    payload = {"format": "doomnet-policy-v1", "model": net.state_dict(), "policy_config": dict(net.config),
               "policy_kind": "PolicyNet", "optimizer": {}, "iteration": 7, "env_steps": 1234,
               "config": {"policy": "auto", "policy_kwargs": {}, "seed": 0}}
    if record:
        payload["provenance"] = provenance.stamp(net)
        assert payload["provenance"]["network"] is not None, payload["provenance"]["error"]
    torch.save(payload, path)
    return path


# ---------------------------------------------------------------------------------- decision rule
def verdict(better=False, worse=False, notes=(), delta=0.0, margin=1.0):
    return {"better": better, "worse": worse, "delta": delta, "se": margin / 2.0, "margin": margin, "notes": list(notes)}


@pytest.mark.parametrize("v, more_rl, replace", [
    (verdict(better=True), None, True),            # better by more than noise: B replaces A whatever else
    (verdict(better=True), "a", True),
    (verdict(worse=True), "b", False),             # worse by more than noise: never
    (verdict(), "b", True),                        # a tie goes to the one with more reinforcement learning
    (verdict(), "a", False),
    (verdict(), None, False),                      # a tie with nothing stated keeps what is shipped
    (verdict(better=True, notes=["the two results come from different suites or seeds"]), "b", False),
    # Inside the noise margin (5.53 on dev_v1 tonight) but not a tie: the plan's tie is one composite point.
    (verdict(delta=-2.30, margin=5.53), "b", False),
    (verdict(delta=-1.01, margin=5.53), "b", False),
    (verdict(delta=-1.00, margin=5.53), "b", True),
    (verdict(delta=-0.54, margin=4.52), "b", True),
    (verdict(delta=+2.30, margin=5.53), "b", True),     # measured higher, not by more than noise: still a tie for B
    (verdict(delta=+2.30, margin=5.53), "a", False),
    (verdict(delta=+2.30, margin=5.53), None, False),
    (verdict(delta=float("nan"), margin=5.53), "b", False),
])
def test_decision_rule(v, more_rl, replace):
    got, why = promote.decide(v, more_rl)
    assert got is replace and why


def test_a_candidate_measured_lower_is_never_promoted_by_the_tie_rule():
    """Whatever the noise margin, `--more-rl b` promotes only from one point below A upwards."""
    for margin in (1.0, 4.52, 5.53, 8.3, 50.0):
        for tenths in range(-600, 601):
            delta = tenths / 10.0
            v = verdict(better=delta > margin, worse=delta < -margin, delta=delta, margin=margin)
            got, why = promote.decide(v, "b")
            assert got is (delta >= -1.0), (margin, delta, why)
            if -margin <= delta < -1.0:
                assert "LOWER" in why and f"{-delta:.2f}" in why
            assert promote.decide(v, None)[0] is (delta > margin) and promote.decide(v, "a")[0] is (delta > margin)


def log_result(tmp_path: Path, tag: str, scores: dict, suite_sha: str = "abc") -> None:
    """Log one synthetic evaluation: `scores` maps (map id, tier) to the episode's score."""
    rows = []
    for (map_id, tier), score in scores.items():
        row = {k: 0 for k in ROW_FIELDS}
        row.update(map_id=map_id, tier=tier, seed=0, wad="", map="MAP01", policy="p", score=score, score_exit_heavy=score,
                   score_balanced=score, has_exit=1, episode_ticks=2100, steps=10, agent_error="", hook_error="", error="",
                   census_ok=1)
        rows.append(row)
    info = {"stamp": f"2026-01-01T00:00:0{len(tag) % 10}", "tag": tag, "commit": "x", "suite": "s.jsonl", "suite_sha": suite_sha,
            "policy": "p", "seeds": "0", "time_scale": 1.0}
    evaluate.append_results(tmp_path / "results.csv", evaluate.tier_rows(rows, info))
    evaluate.write_episodes(tmp_path / "eval" / f"{tag}.csv", rows)


def test_compare_applies_the_rule_to_logged_results(tmp_path, capsys):
    base = {(f"m{i}", 1 + i % 4): 20.0 + i for i in range(40)}
    log_result(tmp_path, "shipped", base)
    log_result(tmp_path, "clearly_better", {k: v + 5.0 + (i % 3) for i, (k, v) in enumerate(base.items())})
    log_result(tmp_path, "same", {k: v + (0.2 if i % 2 else -0.2) for i, (k, v) in enumerate(base.items())})
    log_result(tmp_path, "worse", {k: v - 6.0 - (i % 3) for i, (k, v) in enumerate(base.items())})
    log_result(tmp_path, "elsewhere", {k: v + 9.0 for k, v in base.items()}, suite_sha="other")
    # 2.5 points lower on average, with a spread inside every tier that keeps it within the noise margin: not a tie.
    log_result(tmp_path, "a_bit_lower", {k: v - 2.5 + (12.0 if (i // 4) % 2 else -12.0) for i, (k, v) in enumerate(base.items())})
    run = lambda b, more=None, **kw: promote.compare("shipped", b, more, tmp_path / "results.csv", tmp_path / "eval",  # noqa: E731
                                                     candidates=tmp_path / "cands", **kw)
    assert run("clearly_better")["replace"] is True
    assert run("same")["replace"] is False
    assert run("same", "b")["replace"] is True
    assert run("same", "a")["replace"] is False
    assert run("worse", "b")["replace"] is False
    assert run("elsewhere", "b")["replace"] is False            # another suite decides nothing
    out = capsys.readouterr().out
    assert "DECISION: REPLACE shipped WITH clearly_better" in out and "DECISION: KEEP shipped" in out
    assert "PROMOTE:" not in out and "\nKEEP " not in out          # one verdict line per comparison: the DECISION
    lower = run("a_bit_lower", "b")
    assert not lower["better"] and not lower["worse"] and lower["delta"] == pytest.approx(-2.5) and lower["margin"] > 2.5
    assert lower["replace"] is False                            # within noise, measured lower: more RL does not buy it
    out = capsys.readouterr().out
    assert "DECISION: KEEP shipped: B measured LOWER by 2.50 points" in out and "B measured LOWER than A" in out
    common = ["--out", str(tmp_path / "results.csv"), "--episodes-dir", str(tmp_path / "eval")]
    assert promote.main(["compare", "shipped", "same", "--more-rl", "b", *common]) == 0
    assert promote.main(["compare", "shipped", "same", *common]) == 1
    assert promote.main(["compare", "shipped", "a_bit_lower", "--more-rl", "b", *common]) == 1


def test_more_rl_is_read_from_the_candidates_records_and_a_contradicting_flag_is_not_used(tmp_path, capsys):
    base = {(f"m{i}", 1 + i % 4): 20.0 + i for i in range(40)}
    log_result(tmp_path, "early", base)
    log_result(tmp_path, "later", {k: v + (0.2 if i % 2 else -0.2) for i, (k, v) in enumerate(base.items())})
    log_result(tmp_path, "norecord", {k: v + (0.1 if i % 2 else -0.1) for i, (k, v) in enumerate(base.items())})
    for tag, iteration, steps in (("early", 1800, 111_411_200), ("later", 1900, 118_784_000)):
        (tmp_path / "cands" / tag).mkdir(parents=True)
        (tmp_path / "cands" / tag / "candidate.json").write_text(json.dumps({"name": tag, "iteration": iteration, "env_steps": steps}))
    assert promote.rl_on_record("early", tmp_path / "cands") == {"iteration": 1800, "env_steps": 111_411_200, "source": ""}
    assert promote.rl_on_record("norecord", tmp_path / "cands") is None
    run = lambda a, b, more=None, **kw: promote.compare(a, b, more, tmp_path / "results.csv", tmp_path / "eval",  # noqa: E731
                                                        candidates=tmp_path / "cands", **kw)
    got = run("early", "later")                                 # nothing stated: the records decide the tie
    assert got["replace"] is True and got["more_rl"] == "b" and got["more_rl_recorded"] == "b"
    got = run("later", "early")
    assert got["replace"] is False and got["more_rl"] == "a"
    capsys.readouterr()
    got = run("later", "early", "b")                            # the stated flag says the opposite of the records
    assert got["replace"] is False and got["more_rl"] == "a" and "contradicts" in capsys.readouterr().out
    got = run("later", "early", "b", trust_more_rl=True)        # ... and stands only when that is asked for
    assert got["replace"] is True and "taken as stated although the records say otherwise" in capsys.readouterr().out
    assert run("early", "later", "a")["replace"] is False       # a contradiction never promotes
    got = run("early", "early", "b")                            # the same checkpoint twice: no one has more
    assert got["replace"] is False and got["more_rl"] is None and got["more_rl_recorded"] == "same"
    capsys.readouterr()
    got = run("early", "norecord", "b")                         # one record missing: the flag is all there is, and it says so
    assert got["replace"] is True and got["more_rl_recorded"] is None
    assert "taken as stated: no candidate.json with an iteration for norecord" in capsys.readouterr().out
    assert run("early", "norecord")["replace"] is False
    common = ["--out", str(tmp_path / "results.csv"), "--episodes-dir", str(tmp_path / "eval")]
    assert promote.main(["compare", "early", "norecord", "--more-rl", "b", "--trust-more-rl", *common]) == 0


# -------------------------------------------------------------------------------------- candidate
def test_candidate_is_built_from_the_working_tree_and_never_touches_the_shipped_weights(tmp_path):
    from training.rl import provenance

    shipped = promote.SHIPPED_POLICY
    before = (shipped.stat().st_mtime_ns, shipped.stat().st_size) if shipped.is_file() else None
    source = checkpoint(tmp_path / "ckpt_000007.pt", record=True)
    facts = promote.build_candidate(source, "with record", "require", root=tmp_path / "cands")
    home = tmp_path / "cands" / "with_record"
    sub = home / "submission"
    assert facts["provenance"] == "ok" and facts["provenance_in_agent"] == "ok" and facts["warmup_failures"] == 0
    assert facts["iteration"] == 7 and facts["env_steps"] == 1234 and facts["copy_matches_working_tree"] is True
    assert json.loads((home / "candidate.json").read_text())["policy_sha256"] == promote.sha256(sub / "weights" / "policy.pt")
    for name in ("agent.py", "README.md", "doomnet/spec.py", "doomnet/model.py", "weights/exit_detector.pt", "weights/policy.pt"):
        assert (sub / name).is_file(), name
    assert (sub / "agent.py").read_bytes() == (paths.SUBMISSION / "agent.py").read_bytes()
    assert not list(sub.rglob("__pycache__"))
    assert promote.submission_dir(home) == sub and promote.submission_dir(sub) == sub
    after = (shipped.stat().st_mtime_ns, shipped.stat().st_size) if shipped.is_file() else None
    assert after == before                                      # submission/weights/policy.pt is not this tool's to write

    # The exported file is itself a valid source: it is verified and copied byte for byte.
    assert promote.is_exported_weights(sub / "weights" / "policy.pt") and not promote.is_exported_weights(source)
    again = promote.build_candidate(sub / "weights" / "policy.pt", "again", "require", root=tmp_path / "cands")
    assert again["policy_sha256"] == facts["policy_sha256"] and again["source_kind"] == "exported weights"
    assert provenance.record_of(torch.load(sub / "weights" / "policy.pt", weights_only=True)) is not None


def test_a_checkpoint_without_a_record_is_refused_unless_explicitly_allowed(tmp_path, capsys, monkeypatch):
    from training.rl import provenance

    source = checkpoint(tmp_path / "ckpt_old.pt", record=False)
    with pytest.raises(provenance.CheckpointMismatch, match="no provenance record"):
        promote.build_candidate(source, "old", "require", root=tmp_path / "cands")
    assert not (tmp_path / "cands" / "old").exists()               # a refused export leaves nothing behind
    monkeypatch.setattr(promote, "CANDIDATES", tmp_path / "cli")
    assert promote.main(["score", str(source), "--tag", "t-refused", "--build-only"]) == 3
    assert "REFUSED" in capsys.readouterr().err and not (tmp_path / "cli" / "t-refused").exists()
    assert promote.main(["score", str(source), "--tag", "t-allowed", "--build-only", "--allow-unverified"]) == 0
    assert (tmp_path / "cli" / "t-allowed" / "submission" / "weights" / "policy.pt").is_file()
    facts = promote.build_candidate(source, "old", "strict", root=tmp_path / "cands")
    assert facts["provenance"] == "unverified" and facts["provenance_check"] == "strict"
    assert (tmp_path / "cands" / "old" / "submission" / "weights" / "policy.pt").is_file()


def test_a_record_that_the_weights_no_longer_reproduce_is_refused(tmp_path):
    from training.rl import provenance

    source = checkpoint(tmp_path / "ckpt.pt", record=True)
    payload = torch.load(source, weights_only=False)
    torch.manual_seed(99)
    payload["model"] = PolicyNet(hidden=64, vp_iters=2).state_dict()      # other weights under the old record
    torch.save(payload, source)
    with pytest.raises(provenance.CheckpointMismatch, match="different function"):
        promote.build_candidate(source, "swapped", "require", root=tmp_path / "cands")
    with pytest.raises(provenance.CheckpointMismatch):
        promote.build_candidate(source, "swapped", "strict", root=tmp_path / "cands")      # --allow-unverified is not enough
    assert not (tmp_path / "cands" / "swapped").exists()
    assert promote.build_candidate(source, "swapped", "warn", root=tmp_path / "cands")["provenance"] == "mismatch"


def test_agent_switches_are_not_inherited(monkeypatch):
    monkeypatch.setenv("DOOMNET_ALLOW_UNTRAINED", "1")
    monkeypatch.setenv("DOOMNET_BUDGET_STEPS", "99")
    env = promote.clean_environment({"X": "1"})
    assert "DOOMNET_ALLOW_UNTRAINED" not in env and "DOOMNET_BUDGET_STEPS" not in env and env["X"] == "1"
    said = []
    promote.drop_agent_switches(said.append)
    import os
    assert "DOOMNET_ALLOW_UNTRAINED" not in os.environ and "DOOMNET_BUDGET_STEPS" not in os.environ and len(said) == 2


def test_suite_names():
    assert promote.resolve_suite("dev").endswith("dev_v1.jsonl") and promote.resolve_suite("heldout").endswith("heldout_v1.jsonl")
    assert promote.resolve_suite("some/file.jsonl") == "some/file.jsonl"
    for name in promote.SUITES.values():
        assert (ROOT / name).is_file(), name


def test_slow_decisions_are_pointed_out(tmp_path):
    (tmp_path / "agent.py").write_text("GOVERNOR_DETECTOR_MS = 24.0      # comment\n")
    episodes = tmp_path / "e.csv"
    episodes.write_text("map_id,seed,act_ms_max\nm0,0,5.1\nm1,0,37.1\nm2,0,23.9\n")
    note = promote.governor_note(episodes, tmp_path)
    assert "1 of 3 episodes" in note and "m1" in note and "24 ms" in note
    episodes.write_text("map_id,seed,act_ms_max\nm0,0,5.1\n")
    assert promote.governor_note(episodes, tmp_path) == "" and promote.governor_note(tmp_path / "absent.csv", tmp_path) == ""


def test_differing_episodes_ignores_timing_only(tmp_path):
    head = "map_id,seed,steps,dead,act_ms_mean,act_ms_max,wall_s,wad,policy\n"
    (tmp_path / "a.csv").write_text(head + "m0,0,525,0,3.0,12.4,1.0,/x/a.wad,p1\nm1,0,100,0,3.0,12.4,1.0,/x/b.wad,p1\n")
    (tmp_path / "b.csv").write_text(head + "m0,0,106,1,4.4,37.1,0.4,/y/a.wad,p2\nm1,0,100,0,9.0,50.0,2.0,/y/b.wad,p2\n")
    assert promote.differing_episodes(tmp_path / "a.csv", tmp_path / "b.csv") == (2, 1, ["m0 seed 0"])
    assert promote.differing_episodes(tmp_path / "a.csv", tmp_path / "a.csv") == (2, 0, [])


# ------------------------------------------------------------------------------------ remote rows
def test_rows_of_a_remote_run_are_appended_with_local_commit_and_time(tmp_path):
    remote_results = tmp_path / "remote_results.csv"
    info = {"stamp": "2026-11-26T16:39:03+00:00", "tag": "cand", "commit": "unknown", "suite": "dev_v1.jsonl", "suite_sha": "abc",
            "policy": "p", "seeds": "0", "time_scale": 1.0}
    older = dict(info, stamp="2026-11-26T10:00:00+00:00")
    rows = [dict(older, tier="all", episodes=3, score=1.0), dict(info, tier="1", episodes=2, score=50.0),
            dict(info, tier="all", episodes=2, score=50.0), dict(info, tag="someone_else", tier="all", episodes=2, score=9.0)]
    evaluate.append_results(remote_results, rows)
    episodes = tmp_path / "cand_remote.csv"
    episodes.write_text("map_id,seed,tier,score\nm0,0,1,50\n")
    merged = promote.merge_remote_rows(remote_results, episodes, "cand", tmp_path / "results.csv", tmp_path / "eval")
    assert [r["tier"] for r in merged] == ["1", "all"]              # the latest run of this tag only
    with open(tmp_path / "results.csv", newline="") as f:
        logged = list(csv.DictReader(f))
    assert len(logged) == 2 and {r["tag"] for r in logged} == {"cand"} and logged[0]["commit"] == evaluate.git_commit()
    from datetime import datetime
    assert datetime.fromisoformat(logged[0]["stamp"]) == datetime.fromisoformat(info["stamp"])       # the same instant
    assert (tmp_path / "eval" / "cand.csv").read_text() == episodes.read_text()
    with pytest.raises(SystemExit):
        promote.merge_remote_rows(remote_results, episodes, "absent", tmp_path / "results.csv", tmp_path / "eval")


# ---------------------------------------------------------------------------------- remote helper
class FakeBox(remote.Box):
    """A Box whose ssh and rsync calls are recorded and answered from a table."""

    def __init__(self, answers):
        super().__init__("user@host", 2222, "/work/eval/doom_ai")
        self.answers, self.commands, self.synced = answers, [], []

    def run(self, command, check=True, timeout=120.0):
        self.commands.append(command)
        text = next((out for key, out in self.answers.items() if key in command), "")
        return subprocess.CompletedProcess([], 0, stdout=text, stderr="")

    def _rsync(self, args, timeout=900.0):
        self.synced.append(args)


def test_remote_settings_and_session_names(monkeypatch):
    monkeypatch.delenv("DOOM_BOX", raising=False)
    monkeypatch.delenv("DOOM_BOX_DIR", raising=False)
    with pytest.raises(remote.RemoteError, match="no remote machine configured"):
        remote.Box.from_settings()
    with pytest.raises(remote.RemoteError, match="absolute path"):
        remote.Box.from_settings("u@h:22", "/")
    box = remote.Box.from_settings("u@h:2200", "/work/eval/doom_ai/")
    assert (box.target, box.port, box.directory) == ("u@h", 2200, "/work/eval/doom_ai")
    assert remote.session_name("combat only/v2") == "eval-combat_only_v2"
    for banned in (".git", ".venv", "runs", "data", ".env"):
        assert banned in remote.EXCLUDES


def test_sync_refuses_a_directory_it_did_not_create():
    foreign = FakeBox({"echo ours": "foreign\n"})
    with pytest.raises(remote.RemoteError, match="refusing to sync"):
        foreign.sync_code()
    assert foreign.synced == []
    ours = FakeBox({"echo ours": "ours\n"})
    ours.sync_code()
    args = ours.synced[0]
    assert "--delete" in args and args[-1] == "user@host:/work/eval/doom_ai/"
    assert all(args[i - 1] == "--exclude" for i, a in enumerate(args) if a in remote.EXCLUDES)


def test_a_job_never_reuses_or_stops_a_running_session():
    busy = FakeBox({"tmux ls": "combat\neval-cand\n"})
    with pytest.raises(remote.RemoteError, match="already exists"):
        busy.run_job("eval-cand", "true", "runs/x/log.txt", "runs/x/status")
    with pytest.raises(remote.RemoteError, match="session names start with"):
        busy.run_job("combat", "true", "runs/x/log.txt", "runs/x/status")
    assert not any("kill" in c for c in busy.commands)
    free = FakeBox({"tmux ls": "combat\n", "echo running": "0\nline one\n"})
    assert free.run_job("eval-cand", "true", "runs/x/log.txt", "runs/x/status", poll_s=0.0, echo=lambda *_: None) == 0
    started = next(c for c in free.commands if "new-session" in c)
    assert "-s eval-cand" in started and "kill" not in started


# ------------------------------------------------------------------------------------------- gate
def test_the_working_copy_points_the_tests_at_the_candidate(tmp_path):
    sub = tmp_path / "cand" / "submission"
    (sub / "doomnet").mkdir(parents=True)
    (sub / "agent.py").write_text("MARK = 'candidate'\n")
    shadow = promote.shadow_repo(sub, tmp_path / "shadow")
    out = subprocess.run([sys.executable, "-c", "from training import paths; import agent; print(paths.SUBMISSION); print(agent.MARK)"],
                         cwd=shadow, capture_output=True, text=True, env=promote.clean_environment())
    assert out.returncode == 0, out.stderr
    lines = out.stdout.split()
    assert Path(lines[0]).resolve() == (shadow / "submission").resolve() and lines[1] == "candidate"
    assert (shadow / "validate.py").is_file() and (shadow / "training" / "tests" / "test_rules.py").is_file()
    assert not (shadow / "runs").exists() and not (shadow / ".venv").exists() and not (shadow / ".git").exists()


def frozen_tree(tmp_path: Path) -> Path:
    """A copy of the working tree's submission directory that stays as it is while a test runs."""
    tree = tmp_path / "tree" / "submission"
    promote.shutil.copytree(paths.SUBMISSION, tree, ignore=promote.shutil.ignore_patterns("__pycache__", "policy.pt"))
    return tree


def test_a_candidate_that_differs_from_the_working_tree_is_stale(tmp_path):
    tree = frozen_tree(tmp_path)
    home = tmp_path / "cand"
    sub = home / "submission"
    promote.shutil.copytree(tree, sub)
    (sub / "weights" / "policy.pt").write_bytes(b"policy")
    (sub / "doomnet" / "__pycache__").mkdir()
    (sub / "doomnet" / "__pycache__" / "spec.cpython-312.pyc").write_bytes(b"x")       # never counts
    before = promote.snapshot(sub)
    assert "weights/policy.pt" in before and "agent.py" in before and not any("__pycache__" in name for name in before)
    assert promote.stale_files(before, promote.snapshot(tree)) == []                    # the policy file is left aside

    ok, detail, notes = promote.freshness(sub, before, tree)
    assert ok and "identical" in detail and any("no candidate.json" in n for n in notes)
    record = {"name": "cand", "source": "runs/x/ckpt.pt", "provenance_check": "strict",
              "policy_sha256": before["weights/policy.pt"], "agent_sha256": before["agent.py"],
              "exit_detector_sha256": before["weights/exit_detector.pt"],
              "code": {"spec.py": before["doomnet/spec.py"], "odometry.py": before["doomnet/odometry.py"]}}
    (home / "candidate.json").write_text(json.dumps(record))
    ok, detail, notes = promote.freshness(sub, before, tree)
    assert ok and not notes and "is the file `score` built" in detail
    assert promote.stale_note(home, tree) == ""
    assert promote.rebuild_command(record) == "python -m training.promote score runs/x/ckpt.pt --tag cand --allow-unverified"
    assert promote.rebuild_command(dict(record, provenance="ok")) == "python -m training.promote score runs/x/ckpt.pt --tag cand"
    assert promote.rebuild_command(None) == "" and promote.rebuild_command({"name": "cand"}) == ""

    # One line of the working tree's agent.py changes after the candidate was built (the reviewer's case).
    (tree / "agent.py").write_text((tree / "agent.py").read_text() + "\n# edited after the candidate was built\n")
    ok, detail, _ = promote.freshness(sub, before, tree)
    assert not ok and "stale: agent.py differs from " in detail and "doomnet" not in detail
    assert "agent.py" in promote.stale_note(home, tree)
    (tree / "doomnet" / "extra.py").write_text("X = 1\n")
    (sub / "doomnet" / "left_over.py").write_text("X = 1\n")
    ok, detail, _ = promote.freshness(sub, promote.snapshot(sub), tree)
    assert not ok and "doomnet/extra.py (only in the working tree)" in detail
    assert "doomnet/left_over.py (only in the candidate)" in detail

    # The tree is as the candidate again, but the candidate's policy is no longer the one that was built and scored.
    (sub / "doomnet" / "left_over.py").unlink()
    promote.shutil.rmtree(tree.parent)
    promote.shutil.copytree(sub, tree, ignore=promote.shutil.ignore_patterns("__pycache__", "policy.pt"))
    before = promote.snapshot(sub)
    assert promote.freshness(sub, before, tree)[0] is True
    (sub / "weights" / "policy.pt").write_bytes(b"another policy")
    ok, detail, _ = promote.freshness(sub, promote.snapshot(sub), tree)
    assert not ok and "not the files `score` built (candidate.json): weights/policy.pt" in detail
    ok, detail, _ = promote.freshness(sub, before, tree)                                 # ... and it changed under the gate
    assert not ok and "changed while it was being checked: weights/policy.pt" in detail

    assert promote.freshness(tree, promote.snapshot(tree), tree)[0] is True              # the tree itself is never stale


def test_check_fails_with_reasons_when_the_policy_is_missing(tmp_path):
    tree = frozen_tree(tmp_path)
    sub = tmp_path / "cand" / "submission"
    promote.shutil.copytree(tree, sub)
    lines = []
    result = promote.check(tmp_path / "cand", tests=(), latency_steps=20, tree=tree, echo=lines.append)
    assert result["passed"] is False and lines[-1].startswith("FAIL ")
    assert any(r.startswith("weights:") for r in result["reasons"]) and any(r.startswith("validator:") for r in result["reasons"])
    assert result["steps"]["tree"][0] is True and not any(r.startswith("tree:") for r in result["reasons"])
    assert not any("to ship it" in line for line in lines)


@pytest.mark.slow
def test_the_gate_runs_every_step_on_a_sound_candidate_and_prints_the_copy_command(tmp_path):
    """Size, provenance, validator and latency pass for a freshly exported candidate. The rule lint runs on
    the candidate's copy of the working tree's code, so the verdict follows it: PASS with the copy
    command when the lint is clean, FAIL naming the lint when the tree currently breaks a rule."""
    source = checkpoint(tmp_path / "ckpt.pt", record=True)
    promote.build_candidate(source, "sound", "require", root=tmp_path / "cands")
    # The tree the candidate is compared with is frozen here: other people edit submission/ while this runs.
    tree = tmp_path / "tree" / "submission"
    promote.shutil.copytree(tmp_path / "cands" / "sound" / "submission", tree, ignore=promote.shutil.ignore_patterns("policy.pt"))
    lines = []
    result = promote.check(tmp_path / "cands" / "sound", tests=("test_rules.py",), latency_steps=60, tree=tree, echo=lines.append)
    steps = result["steps"]
    assert list(steps) == ["size", "provenance", "validator", "test_rules.py", "latency", "tree"]
    for name in ("size", "provenance", "validator", "latency", "tree"):
        assert steps[name][0], (name, steps[name][1])
    assert "passed" in steps["test_rules.py"][1]                      # the pytest summary line was found
    assert result["passed"] is steps["test_rules.py"][0]
    if result["passed"]:
        assert lines[-2].startswith("PASS ") and lines[-1].startswith("to ship it: cp ")
        assert lines[-1].endswith("submission/weights/policy.pt")
    else:
        assert lines[-1].startswith("FAIL ") and "test_rules.py" in lines[-1] and not any("to ship it" in x for x in lines)

    # One line of agent.py changes in the tree after the candidate was built: every other step still
    # passes, and the verdict is FAIL with the file named and no copy command.
    (tree / "agent.py").write_text((tree / "agent.py").read_text() + "\n# edited after the candidate was built\n")
    lines = []
    result = promote.check(tmp_path / "cands" / "sound", tests=(), latency_steps=60, tree=tree, echo=lines.append)
    assert [name for name, (ok, _) in result["steps"].items() if not ok] == ["tree"]
    assert result["passed"] is False and lines[-2].startswith("FAIL ") and "stale: agent.py differs" in lines[-2]
    assert lines[-1].startswith("no copy command") and "python -m training.promote score " in lines[-1]
    assert not any("to ship it" in x for x in lines)
