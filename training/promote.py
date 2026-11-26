"""Checkpoint evaluation and promotion: is this agent better than the one on main, and is it safe to ship?

Training-time scaffolding, never imported by the agent.

    python -m training.promote score CHECKPOINT --tag NAME [--suite dev|heldout|FILE] [--workers 3]
                                     [--remote] [--twice] [--allow-unverified | --allow-mismatch]
    python -m training.promote check runs/promote/NAME
    python -m training.promote compare TAG_A TAG_B [--more-rl a|b] [--trust-more-rl]

score
    Builds a candidate under `runs/promote/NAME/submission/`: `agent.py`, `doomnet/` and the exit
    detector from the working tree, plus `weights/policy.pt` made from CHECKPOINT by the project's
    own export (`training/rl/export.py`, which compares the checkpoint's provenance record with the
    current code and reads the written file back). CHECKPOINT may also be a weights file that was
    exported earlier; it is then verified and copied unchanged. The candidate is scored through the
    real agent, `training.runner:AgentPolicy` loaded from that directory, with the official time
    limits, on a frozen suite with fixed seeds (`dev_v1` by default, `heldout_v1` with
    `--suite heldout`). Per-tier rows are appended to `training/logs/results.csv` by
    `training.evaluate` and the per-episode rows go to `runs/eval/NAME.csv`.
    With `--remote` the candidate is still built and checked here, then the working tree and the
    candidate are sent to the evaluation machine (`training/ops/remote.py`), played there with
    many workers inside a tmux session, and the rows are pulled back and appended here.
    `--twice` plays the suite a second time and reports how many episodes came out differently:
    the agent is deterministic except for its time governor, so that number is the run-to-run
    noise of the machine at its current load.

check
    The ship gate for a candidate directory (or any directory with a `submission/` in it). A copy
    of the repository's test and validation code is placed around the candidate, so everything
    below runs against the candidate's files and nothing else:
      size        the directory is small (default limit 50 MB)
      provenance  `weights/policy.pt` loads and gives the outputs its provenance record stores
      validator   `validate.py` passes with DOOMNET_ALLOW_UNTRAINED unset
      tests       test_rules.py, test_fuzz.py, test_agent.py and test_doors.py pass
      latency     mean `act()` time on one thread is under the budget (default 20 ms)
      tree        every file of the candidate except `weights/policy.pt` is identical to the working
                  tree's `submission/` as it is when the other steps have finished, the policy file
                  is the one `score` built (`candidate.json`), and nothing in the candidate changed
                  while it was being checked
    Prints one line per step and then one PASS or FAIL line with the reasons. Exit status 0 on PASS.
    The last step is what makes the printed copy command safe: what was scored and gated is the
    candidate's own copy of `agent.py`, `doomnet/` and the exit detector, and what ships is the
    working tree plus the copied policy file. When the two differ the verdict is FAIL with the
    files named, no copy command is printed, and the candidate has to be built and scored again.

compare
    `training.evaluate.compare` on two logged tags, plus the decision rule: B replaces A only if it
    is better by more than noise (one composite point and two standard errors of the paired
    difference). A tie goes to the one that has had more reinforcement learning, and to A when that
    is not known. A tie is what the plan says it is: B is not better by more than noise and is at
    most one composite point below A. A candidate that measured lower than that is never promoted,
    however much training it has had, even when the loss is inside the noise margin.
    Which one has had more reinforcement learning is read from the `candidate.json` files of the two
    tags (environment steps, then iteration) when both exist. `--more-rl a|b` states it when they do
    not; where it contradicts the records the tie stays with A unless `--trust-more-rl` is given.
    Exit status 0 only when B should replace A.

Nothing here writes into `submission/weights/`. When a candidate passes `check`, the command that
copies its policy file there is printed for whoever decides to ship it. That command is valid for
the working tree as it was when `check` finished: run `check` again if `submission/` changes first.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from training import paths

CANDIDATES = paths.ROOT / "runs" / "promote"
SHIPPED_POLICY = paths.ROOT / "submission" / "weights" / "policy.pt"
SUITES = {"dev": "training/suites/dev_v1.jsonl", "heldout": "training/suites/heldout_v1.jsonl",
          "dev_nav": "training/suites/dev_nav_v1.jsonl", "heldout_nav": "training/suites/heldout_nav_v1.jsonl"}
AGENT_POLICY = "training.runner:AgentPolicy"
GATE_TESTS = ("test_rules.py", "test_fuzz.py", "test_agent.py", "test_doors.py")
# What the test and validation code needs around a submission directory (copied by `shadow_repo`).
SHADOW_PARTS = ("training", "tests", "env", "maps", "docs", "docker", "validate.py", "pyproject.toml", "README.md",
                "Dockerfile", ".gitignore", ".gitattributes", ".github")
SHADOW_SKIP = ("__pycache__", ".pytest_cache", "*.pyc", "_vizdoom.ini")
LATENCY_BUDGET_MS = 20.0     # mean act() time on one thread: the project's binding number (the scorer allows 40 ms)
SIZE_LIMIT_MB = 50.0         # submission directory with weights; the repository limit is 1 GB
# Environment switches that change how the agent plays or loads: never inherited by a scoring or gate run.
AGENT_SWITCHES = ("DOOMNET_ALLOW_UNTRAINED", "DOOMNET_BUDGET_STEPS", "DOOMRL_ALLOW_MISMATCH", "DOOMNET_FUZZ_BUDGET_MS",
                  "DOOMNET_DOOR_POLICIES", "DOOMNET_GREEDY_HEADS")
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
POLICY_NAME = "weights/policy.pt"
# Files that never count when a candidate's submission directory is compared with the working tree's.
SNAPSHOT_SKIP = ("__pycache__", ".pytest_cache", "*.pyc", "*.tmp", ".gitkeep", ".DS_Store", "_vizdoom.ini")
# candidate.json key -> the file of the candidate whose sha256 it holds (the copies of doomnet/ are under "code").
RECORDED_FILES = (("policy_sha256", POLICY_NAME), ("agent_sha256", "agent.py"), ("exit_detector_sha256", "weights/exit_detector.pt"))
# Columns of the short table printed after a scoring run: (column of results.csv, heading, decimals).
REPORT_COLUMNS = (("score", "score", 2), ("score_se", "+-", 2), ("kills_frac", "kills", 3), ("death_rate", "deaths", 3),
                  ("exit_rate", "exits", 3), ("c_combat", "combat", 3), ("door_reversals", "rev", 2),
                  ("act_ms_mean", "ms", 2), ("agent_errors", "agent", 0), ("errors", "err", 0))


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_environment(extra: dict | None = None) -> dict:
    """A copy of the environment without the switches that change the agent's behaviour, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if k not in AGENT_SWITCHES and k != "PYTEST_ADDOPTS"}
    env.update(extra or {})
    return env


def drop_agent_switches(echo=print) -> None:
    """Remove those switches from this process, so worker processes started later do not inherit them."""
    for name in AGENT_SWITCHES:
        if os.environ.pop(name, None) is not None:
            echo(f"note: {name} was set in the environment and is ignored for this run")


def resolve_suite(name: str) -> str:
    return SUITES.get(name, name)


def submission_dir(directory) -> Path:
    """`directory` itself when it is a submission directory, else its `submission/` sub-directory."""
    directory = Path(directory)
    if (directory / "agent.py").is_file():
        return directory
    if (directory / "submission" / "agent.py").is_file():
        return directory / "submission"
    raise SystemExit(f"{directory} is neither a submission directory nor a directory that holds one (no agent.py found)")


def _relative(path: Path) -> str:
    """Path relative to the repository root when it is inside it (the same text on every machine), else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(paths.ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _as_committed(relative: str) -> bool | None:
    """True when the file is tracked and identical to the last commit, False when not, None outside git. Read-only."""
    try:
        out = subprocess.run(["git", "--no-optional-locks", "-C", str(paths.ROOT), "status", "--porcelain", "--", relative],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None if out.returncode != 0 else not out.stdout.strip()


# ------------------------------------------------------------------------------------- snapshot
def snapshot(directory) -> dict[str, str]:
    """{path relative to `directory`: sha256} of every file of a submission directory that counts."""
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        parts = path.relative_to(directory).parts
        if path.is_file() and not any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in SNAPSHOT_SKIP):
            files["/".join(parts)] = sha256(path)
    return files


def stale_files(candidate: dict, tree: dict) -> list[str]:
    """Where two snapshots differ, the policy file left aside: what was gated against what would ship."""
    out = []
    for name in sorted((set(candidate) | set(tree)) - {POLICY_NAME}):
        if name not in tree:
            out.append(f"{name} (only in the candidate)")
        elif name not in candidate:
            out.append(f"{name} (only in the working tree)")
        elif candidate[name] != tree[name]:
            out.append(name)
    return out


def read_record(home) -> dict | None:
    """The facts `build_candidate` wrote beside a candidate's submission directory, or None."""
    try:
        facts = json.loads((Path(home) / "candidate.json").read_text())
    except (OSError, ValueError):
        return None
    return facts if isinstance(facts, dict) else None


def record_mismatches(record: dict, files: dict) -> list[str]:
    """Files of a candidate whose sha256 is no longer the one written down when `score` built it."""
    expected = {name: record.get(key) for key, name in RECORDED_FILES}
    expected.update({f"doomnet/{name}": digest for name, digest in (record.get("code") or record.get("code_now") or {}).items()})
    return sorted(name for name, digest in expected.items() if digest and files.get(name) != digest)


def _listed(names: list[str], most: int = 6) -> str:
    return ", ".join(names[:most]) + (f" and {len(names) - most} more" if len(names) > most else "")


def rebuild_command(record: dict | None) -> str:
    """The command that builds and scores a candidate again from the checkpoint it was made of, or ""."""
    if not record or not record.get("source") or not record.get("name"):
        return ""
    flag = "" if record.get("provenance") == "ok" else {
        "strict": " --allow-unverified", "warn": " --allow-mismatch"}.get(record.get("provenance_check"), "")
    return f"python -m training.promote score {record['source']} --tag {record['name']}{flag}"


def freshness(sub: Path, before: dict, tree=None) -> tuple[bool, str, list[str]]:
    """(ok, detail, notes): is what was just gated the same thing that would ship?

    sub     the candidate's submission directory.
    before  its `snapshot` taken before the gate's other steps ran.
    tree    the submission directory that ships (the working tree's, read now).
    Three things can be wrong: a file other than the policy differs from the working tree (the
    candidate is stale: it was scored and gated with code that is not the code that would ship),
    the policy or another recorded file is not the one `score` built, or the candidate changed
    while the checks were running.
    """
    sub, tree = Path(sub).resolve(), Path(tree or paths.SUBMISSION).resolve()
    if sub == tree:
        return True, "this is the working tree's own submission directory", []
    now, record, problems, notes = snapshot(sub), read_record(sub.parent), [], []
    where = "the working tree's submission/" if tree == Path(paths.SUBMISSION).resolve() else f"{_relative(tree)}/"
    moved = sorted(name for name in set(before) | set(now) if before.get(name) != now.get(name))
    if moved:
        problems.append(f"changed while it was being checked: {_listed(moved)}")
    differ = stale_files(now, snapshot(tree))
    if differ:
        problems.append(f"stale: {_listed(differ)} differ{'s' if len(differ) == 1 else ''} from {where}: "
                        "what was scored and gated is not what would ship; build and score it again")
    if record is None:
        notes.append("no candidate.json beside it: nothing ties this policy file to a build by `score` or to a logged score")
    else:
        changed = record_mismatches(record, now)
        if changed:
            problems.append(f"not the files `score` built (candidate.json): {_listed(changed)}")
    if problems:
        return False, "; ".join(problems), notes
    digest = now.get(POLICY_NAME, "")[:12]
    return True, (f"{len(now) - (POLICY_NAME in now)} files identical to {where} at {datetime.now():%H:%M:%S}"
                  + (f"; policy.pt {digest} is the file `score` built" if record is not None and digest else "")
                  + "; check again if that directory changes before the copy"), notes


def stale_note(candidate, tree=None) -> str:
    """One line for a scoring run when the candidate no longer matches the working tree, else ""."""
    sub = submission_dir(candidate).resolve()
    tree = Path(tree or paths.SUBMISSION).resolve()
    differ = [] if sub == tree else stale_files(snapshot(sub), snapshot(tree))
    if not differ:
        return ""
    return (f"note: {_relative(sub)} differs from the working tree in {_listed(differ)}: this score is for the candidate's "
            "own copy of the code, and `check` will refuse to ship it until it is built again from the current tree")


# ------------------------------------------------------------------------------------ candidate
def is_exported_weights(path) -> bool:
    """True for a file in the agent's own layout ({"state_dict", "config"}), False for a training checkpoint."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and "model" not in payload and isinstance(payload.get("state_dict"), dict)


def build_candidate(source, name: str, check: str = "require", root=None, echo=print) -> dict:
    """Make `root/name/submission/` from the working tree and the policy in `source`. Returns the candidate's facts.

    source  a checkpoint of training/rl, or a weights file exported earlier.
    check   how the provenance record is treated, as in `training.rl.export`: "require" refuses a
            file without a record or with one the current code no longer reproduces; "strict"
            accepts a missing record with a warning (`--allow-unverified`); "warn" only prints
            (`--allow-mismatch`).
    The facts are also written to `root/name/candidate.json`. Raises when the export refuses.
    """
    from training.rl import export, provenance

    source = Path(source)
    if not source.is_file():
        raise SystemExit(f"checkpoint not found: {source}")
    home = Path(root or CANDIDATES) / _safe(name)
    target, building = home / "submission", home / "submission.building"
    if building.exists():
        shutil.rmtree(building)
    shutil.copytree(paths.SUBMISSION, building, ignore=shutil.ignore_patterns("__pycache__", "policy.pt", "*.tmp", ".gitkeep"))
    try:      # built aside and moved into place at the end: a refused export leaves an earlier candidate as it was
        if not (building / "weights" / "exit_detector.pt").is_file():
            raise SystemExit(f"{paths.SUBMISSION / 'weights' / 'exit_detector.pt'} is missing: the agent cannot start without it")
        policy_file = building / "weights" / "policy.pt"
        exported = is_exported_weights(source)
        if exported:      # already in the agent's layout: verify it against the current code and keep its bytes
            status = export.verify(source, check).status
            shutil.copyfile(source, policy_file)
            details = {}
        else:
            out = export.export(str(source), policy_file, check=check)
            status, details = out["provenance"], {"parameters": out["parameters"], "policy_kwargs": out["config"]}
        in_agent = export.verify_in_agent(policy_file, check="warn" if check == "warn" else "strict")
        if target.exists():
            shutil.rmtree(target)
        os.replace(building, target)
    finally:
        shutil.rmtree(building, ignore_errors=True)
        if home.is_dir() and not any(home.iterdir()):
            home.rmdir()
    detector, policy_file = target / "weights" / "exit_detector.pt", target / "weights" / "policy.pt"
    import agent as agent_module

    _, config, _ = agent_module.read_weights(policy_file)
    record = provenance.record_of({"config": config}) or {}
    now, then = provenance.code_fingerprint(), dict(record.get("code") or {})
    copied = {f: sha256(target / "doomnet" / f) for f in provenance.SOURCE_FILES if (target / "doomnet" / f).is_file()}
    facts = {
        "name": name, "built": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source), "source_sha256": sha256(source), "source_kind": "exported weights" if exported else "checkpoint",
        "iteration": int(config.get("iteration", 0) or 0), "env_steps": int(config.get("env_steps", 0) or 0),
        "policy_sha256": sha256(policy_file), "exit_detector_sha256": sha256(detector),
        "agent_sha256": sha256(target / "agent.py"),
        "provenance": status, "provenance_check": check, "provenance_in_agent": in_agent.get("provenance"),
        "warmup_failures": in_agent.get("warmup_failures"),
        "record_written": record.get("written", ""), "record_platform": record.get("platform", ""),
        "code": copied, "code_changed_since_record": sorted(f for f in copied if then and then.get(f) != copied[f]),
        "copy_matches_working_tree": copied == now, "exit_detector_as_committed": _as_committed("submission/weights/exit_detector.pt"),
        **details,
    }
    (home / "candidate.json").write_text(json.dumps(facts, indent=1) + "\n")
    if copied != now:
        echo("WARNING: files under submission/doomnet changed while the candidate was being built; its provenance was checked "
             f"against the working tree, not against its own copy. `check {_relative(home)}` checks it against the copy.")
    if facts["exit_detector_as_committed"] is False:
        echo("note: submission/weights/exit_detector.pt differs from the committed file (or is not committed); "
             "the candidate carries the working tree's file")
    echo(f"candidate {_relative(home)}: policy.pt sha256 {facts['policy_sha256'][:12]} from {source.name} "
         f"(iteration {facts['iteration']}, {facts['env_steps']} env steps), provenance {status}"
         + ("" if status == "ok" else f" [accepted with check={check}]"))
    return facts


# ---------------------------------------------------------------------------------------- score
def report_rows(rows: list[dict], echo=print) -> None:
    """Kills, deaths, exits and composite per tier, from results.csv rows (dicts of numbers or CSV text)."""
    echo(f"{'tag':<22} {'suite':<16} {'tier':>4} {'n':>4} " + " ".join(f"{h:>7}" for _, h, _ in REPORT_COLUMNS))
    for r in rows:
        cells = []
        for key, _, decimals in REPORT_COLUMNS:
            value = r.get(key)
            cells.append(f"{'-':>7}" if value in ("", None) or float(value) != float(value) else f"{float(value):>7.{decimals}f}")
        echo(f"{str(r['tag'])[:22]:<22} {str(r.get('suite', ''))[:16]:<16} {str(r['tier']):>4} {int(float(r['episodes'])):>4} "
             + " ".join(cells))


def governor_note(episodes_file, submission: Path) -> str:
    """One line when a slow decision may have changed how some episodes were played, else "".

    The agent's time governor rests the exit detector, and above a second threshold the network,
    while the mean `act()` time of the current map is over a limit. Early in a map that mean is
    over few decisions, so a single slow call there (a busy machine) changes the next decisions and
    with them the rest of the episode: such an episode need not repeat on another run.
    """
    text = (Path(submission) / "agent.py").read_text() if (Path(submission) / "agent.py").is_file() else ""
    found = re.search(r"^GOVERNOR_DETECTOR_MS\s*=\s*([0-9.]+)", text, re.M)
    limit_ms = float(found.group(1)) if found else 24.0
    if not Path(episodes_file).is_file():
        return ""
    with open(episodes_file, newline="") as f:
        rows = list(csv.DictReader(f))
    slow = [r["map_id"] for r in rows if float(r.get("act_ms_max") or 0.0) > limit_ms]
    if not slow:
        return ""
    return (f"note: in {len(slow)} of {len(rows)} episodes one act() took over {limit_ms:g} ms (first: {slow[0]}). If that was "
            "among the first decisions of the map the time governor changed the next ones, and the episode may not repeat; "
            "score again on a quieter machine before reading a small difference")


def differing_episodes(file_a, file_b) -> tuple[int, int, list[str]]:
    """(episodes in both files, how many differ in anything but timing, the first few that do)."""
    from training.runner import TIMING_FIELDS

    def load(path):
        with open(path, newline="") as f:
            return {(r["map_id"], r["seed"]): r for r in csv.DictReader(f)}

    a, b = load(file_a), load(file_b)
    shared = sorted(set(a) & set(b))
    skip = set(TIMING_FIELDS) | {"wad", "policy"}
    differ = [f"{k[0]} seed {k[1]}" for k in shared if any(a[k][f] != b[k].get(f) for f in a[k] if f not in skip)]
    return len(shared), len(differ), differ[:5]


def score_local(candidate: Path, tag: str, suite: str, *, workers: int = 3, seeds=(0,), tiers=None, limit=None,
                time_scale: float = 1.0, out=None, episodes_dir=None, twice: bool = False, echo=print) -> list[dict]:
    """Play `suite` with the agent in `candidate` and append the rows. Returns the rows as written.

    twice  play everything a second time (nothing of it is logged) and say how many episodes came
           out differently. The agent re-seeds itself at every map, so none should; the ones that
           do were changed by the clock (see `governor_note`), and their number is the run-to-run
           noise of this machine at its current load.
    """
    from training import evaluate

    sub = submission_dir(candidate)
    if not (sub / "weights" / "policy.pt").is_file():
        raise SystemExit(f"{sub / 'weights' / 'policy.pt'} is missing: build the candidate with `score CHECKPOINT` first")
    drop_agent_switches(echo)
    stale = stale_note(sub)
    if stale:
        echo(stale)
    policy = (AGENT_POLICY, {"repo": _relative(sub.parent)})
    os.chdir(paths.ROOT)                # the candidate is named relative to the repository root
    rows = evaluate.evaluate(policy, resolve_suite(suite), tag, workers=workers, seeds=seeds, tiers=tiers, limit=limit,
                             time_limits=True, time_scale=time_scale,
                             out=out or evaluate.RESULTS_CSV, episodes_dir=episodes_dir or evaluate.EPISODES_DIR)
    echo("")
    report_rows(rows, echo)
    first = Path(episodes_dir or evaluate.EPISODES_DIR) / f"{_safe(tag)}.csv"
    note = governor_note(first, sub)
    if note:
        echo(note)
    if twice:
        with tempfile.TemporaryDirectory(prefix="promote-twice-") as scratch:
            evaluate.evaluate(policy, resolve_suite(suite), tag, workers=workers, seeds=seeds, tiers=tiers, limit=limit,
                              time_limits=True, time_scale=time_scale, out=Path(scratch) / "results.csv",
                              episodes_dir=scratch, quiet=True)
            shared, differ, which = differing_episodes(first, Path(scratch) / f"{_safe(tag)}.csv")
        echo(f"REPEAT: {differ} of {shared} episodes came out differently when played a second time"
             + (f" (first: {', '.join(which)}): the logged scores carry that much run-to-run noise" if differ else
                ": the logged scores repeat exactly on this machine"))
    return rows


def merge_remote_rows(results_file, episodes_file, tag: str, out=None, episodes_dir=None) -> list[dict]:
    """Append the rows of a remote run to the local results file and keep its per-episode file for `compare`."""
    from training import evaluate

    rows = [r for r in evaluate.read_results(results_file) if r.get("tag") == tag]
    if not rows:
        raise SystemExit(f"{results_file} holds no rows tagged {tag!r}")
    rows = [r for r in rows if r["stamp"] == rows[-1]["stamp"]]
    commit = evaluate.git_commit()      # the remote copy has no history; the code is the working tree that was sent
    for r in rows:
        r["commit"] = commit
        try:                            # the remote clock's time zone -> this machine's, like every other row
            r["stamp"] = datetime.fromisoformat(r["stamp"]).astimezone().isoformat(timespec="seconds")
        except ValueError:
            pass
    evaluate.append_results(out or evaluate.RESULTS_CSV, rows)
    target = Path(episodes_dir or evaluate.EPISODES_DIR) / f"{_safe(tag)}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(episodes_file, target)
    return rows


def score_remote(candidate: Path, tag: str, suite: str, box, *, workers: int = 64, seeds=(0,), tiers=None, limit=None,
                 time_scale: float = 1.0, out=None, episodes_dir=None, twice: bool = False, echo=print) -> list[dict]:
    """The same as `score_local`, played on the evaluation machine. Returns the rows as appended here."""
    from training.ops import remote

    home = Path(candidate).resolve()
    relative = _relative(home)
    if relative.startswith("/"):
        raise SystemExit(f"a candidate scored remotely must live inside the repository (under runs/promote), not at {home}")
    workers = max(1, min(int(workers), remote.MAX_REMOTE_WORKERS))
    suite = resolve_suite(suite)
    for old in ("remote_results.csv", "remote.txt"):
        (home / old).unlink(missing_ok=True)
    stale = stale_note(home)
    if stale:
        echo(stale)
    echo(f"sending the working tree and {relative} to {box.target}:{box.directory}")
    box.sync_code()
    box.push(relative)
    q = lambda text: "'" + str(text).replace("'", "'\\''") + "'"  # noqa: E731
    command = (f"{q(box.python)} -u -m training.promote score --candidate {q(relative)} --tag {q(tag)} --suite {q(suite)} "
               f"--workers {workers} --seeds {','.join(str(int(s)) for s in seeds)} --time-scale {time_scale:g} "
               f"--out {q(relative + '/remote_results.csv')} --episodes-dir {q(relative + '/remote_eval')}"
               + (f" --tiers {','.join(str(int(t)) for t in tiers)}" if tiers else "")
               + (f" --limit {int(limit)}" if limit is not None else "") + (" --twice" if twice else ""))
    status = box.run_job(remote.session_name(tag), command, f"{relative}/remote.txt", f"{relative}/remote.status", echo=echo)
    box.pull(f"{relative}/remote.txt", home / "remote.txt")
    for line in (home / "remote.txt").read_text(errors="replace").splitlines():
        if line.startswith("REPEAT: "):
            echo(line)
    if status != 0:
        raise SystemExit(f"the remote run failed with status {status}; its output is in {home / 'remote.txt'}")
    box.pull(f"{relative}/remote_results.csv", home / "remote_results.csv")
    box.pull(f"{relative}/remote_eval/{_safe(tag)}.csv", home / "remote_eval" / f"{_safe(tag)}.csv")
    rows = merge_remote_rows(home / "remote_results.csv", home / "remote_eval" / f"{_safe(tag)}.csv", tag, out, episodes_dir)
    echo("")
    report_rows(rows, echo)
    note = governor_note(home / "remote_eval" / f"{_safe(tag)}.csv", submission_dir(home))
    if note:
        echo(note)
    return rows


# ---------------------------------------------------------------------------------------- check
def shadow_repo(submission: Path, where: Path) -> Path:
    """A copy of the repository's test and validation code at `where`, with `submission` as its submission/.

    The tests and the validator find the agent relative to their own files, so inside this copy
    they exercise the candidate and cannot reach the working tree's `submission/`.
    """
    where = Path(where)
    ignore = shutil.ignore_patterns(*SHADOW_SKIP)
    for part in SHADOW_PARTS:
        src = paths.ROOT / part
        if src.is_dir():
            shutil.copytree(src, where / part, ignore=ignore)
        elif src.is_file():
            shutil.copy2(src, where / part)
    shutil.copytree(submission, where / "submission", ignore=ignore)
    return where


def _run(cmd: list[str], cwd: Path, env: dict, timeout: float) -> tuple[int, str]:
    """(exit status, combined output) of a command; status 124 when it ran out of time."""
    try:
        done = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return 124, text + f"\n[timed out after {timeout:.0f} s]"
    return done.returncode, done.stdout + done.stderr


def _last_line(text: str, pattern: str = "") -> str:
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in text.splitlines() if line.strip()]
    hits = [line for line in lines if re.search(pattern, line)] if pattern else lines
    return (hits or lines or [""])[-1][:300]


def directory_size(directory: Path) -> int:
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def check(directory, *, allow_unverified: bool = False, budget_ms: float = LATENCY_BUDGET_MS,
          size_limit_mb: float = SIZE_LIMIT_MB, tests=GATE_TESTS, fast: bool = False, latency_steps: int = 400,
          keep: bool = False, tree=None, echo=print) -> dict:
    """Run the ship gate on a candidate. Returns {"passed", "reasons", "steps": {name: (ok, detail)}, "line"}.

    allow_unverified  accept a policy file without a provenance record (reported in the verdict line).
    fast              leave out the tests marked slow (a quicker look; not the gate).
    tree              the submission directory the candidate's policy would be copied into
                      (default: the working tree's); the candidate must match it file for file.
    """
    sub = submission_dir(directory).resolve()
    tree = Path(tree or paths.SUBMISSION).resolve()
    before = snapshot(sub)
    python = sys.executable
    steps: dict[str, tuple[bool, str]] = {}

    def done(name: str, ok: bool, detail: str) -> None:
        steps[name] = (bool(ok), detail)
        echo(f"  {'ok  ' if ok else 'FAIL'} {name:<11} {detail}")

    echo(f"check {_relative(sub)}")
    policy_file = sub / "weights" / "policy.pt"
    size_mb = directory_size(sub) / 1e6
    done("size", size_mb <= size_limit_mb, f"{size_mb:.1f} MB (limit {size_limit_mb:g} MB)")
    if not policy_file.is_file():
        done("weights", False, "weights/policy.pt is missing")
    # The working copy lives in the system's temporary directory, outside this git repository: inside
    # it the validator would ask git about the copy's files and be told that all of them are ignored.
    shadow = Path(tempfile.mkdtemp(prefix=f"promote-check-{_safe(sub.parent.name)}-")).resolve()
    if any((parent / ".git").exists() for parent in (shadow, *shadow.parents)):
        echo(f"  note: the temporary directory {shadow} is inside a git repository; the validator's git checks may misfire")
    try:
        shadow_repo(sub, shadow)
        env = clean_environment({"PYTHONPATH": str(shadow / "env"), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        if policy_file.is_file():
            mode = "strict" if allow_unverified else "require"
            code = ("import sys\nfrom training.rl import export\n"
                    f"v = export.verify('submission/weights/policy.pt', {mode!r})\nprint('PROVENANCE', v.status)\n")
            status, text = _run([python, "-c", code], shadow, env, 300)
            verdict = _last_line(text, r"^PROVENANCE ").removeprefix("PROVENANCE ") if status == 0 else ""
            done("provenance", status == 0 and verdict in ("ok", "unverified"),
                 {"ok": "the policy file gives the outputs its record stores, under the candidate's own code",
                  "unverified": "NO RECORD in the policy file; accepted because --allow-unverified was given"}.get(
                     verdict, _last_line(text)))
        status, text = _run([python, "validate.py", "--repo", str(shadow), "--device", "cpu"], shadow, env, 900)
        crosses = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip()[:200] for line in text.splitlines() if "✗" in line]
        done("validator", status == 0, _last_line(text) if status == 0 else "; ".join(crosses[:3]) or _last_line(text))
        for name in tests:
            cmd = [python, "-m", "pytest", f"training/tests/{name}", "-q", "-p", "no:cacheprovider", "--no-header", "-rf"]
            status, text = _run(cmd + (["-m", "not slow"] if fast else []), shadow, env, 1800)
            summary = _last_line(text, r"\b(passed|failed|error|errors|skipped|no tests ran)\b")
            failed = [line.split(" - ")[0].removeprefix("FAILED ") for line in text.splitlines() if line.startswith("FAILED ")]
            done(name, status == 0, summary + ("" if status == 0 or not failed else " | " + "; ".join(failed[:4])))
        report = shadow / "latency.json"
        cmd = [python, "training/bench/agent_latency.py", "--steps", str(int(latency_steps)), "--repeats", "3", "--json", str(report)]
        status, text = _run(cmd, shadow, {**env, **THREAD_ENV}, 900)
        if report.is_file():
            result = json.loads(report.read_text())
            best = result["repeats"][result["best"]]
            mean, failures = best["all"]["mean"], sum(r["failures"] for r in result["repeats"])
            load = (result.get("load_average") or [float("nan")])[0]
            done("latency", mean <= budget_ms and not failures,
                 f"mean {mean:.2f} ms, p95 {best['all']['p95']:.2f}, max {best['all']['max']:.2f} on one thread "
                 f"(budget {budget_ms:g} ms; {result['machine']}, load {load:.1f})"
                 + (f"; {failures} decisions raised inside the agent" if failures else ""))
        else:
            done("latency", False, "the benchmark did not finish: " + _last_line(text))
    finally:
        if keep:
            echo(f"  kept the working copy at {shadow}")
        else:
            shutil.rmtree(shadow, ignore_errors=True)
    # Last, after everything that takes time: is what was gated above the thing that would ship?
    fresh, detail, tree_notes = freshness(sub, before, tree)
    done("tree", fresh, detail)
    reasons = [f"{name}: {detail}" for name, (ok, detail) in steps.items() if not ok]
    passed = not reasons
    notes = [detail for name, (ok, detail) in steps.items() if ok and name == "provenance" and "NO RECORD" in detail]
    notes += tree_notes
    notes += ["slow tests were left out (--fast): not the full gate"] if fast else []
    line = (f"PASS {_relative(sub)}: {len(steps)} checks passed" + ("".join(f"; NOTE {n}" for n in notes))
            if passed else f"FAIL {_relative(sub)}: " + "; ".join(reasons))
    echo(line)
    if not fresh:
        again = rebuild_command(read_record(sub.parent))
        echo("no copy command: copying only the policy file would ship a combination that was neither scored nor gated"
             + (f". To build and score it again from the current tree: {again}" if again else ""))
    elif passed and sub != tree:
        echo(f"to ship it: cp {_relative(policy_file)} {_relative(tree / POLICY_NAME)}")
    elif passed:
        echo("nothing to copy: this is the working tree's own submission directory")
    return {"passed": passed, "reasons": reasons, "steps": steps, "line": line}


# -------------------------------------------------------------------------------------- compare
def rl_on_record(tag: str, root=None) -> dict | None:
    """{"iteration", "env_steps", "source"} of the policy behind a tag, from its candidate.json; None when unknown."""
    record = read_record(Path(root or CANDIDATES) / _safe(tag))
    if record is None:
        return None
    try:
        amount = {"iteration": int(record.get("iteration") or 0), "env_steps": int(record.get("env_steps") or 0),
                  "source": str(record.get("source", ""))}
    except (TypeError, ValueError):
        return None
    return amount if amount["iteration"] > 0 or amount["env_steps"] > 0 else None


def more_rl_on_record(a: dict | None, b: dict | None) -> str | None:
    """"a", "b" or "same" by environment steps and then iteration; None unless both are on record."""
    if a is None or b is None:
        return None
    key_a, key_b = (a["env_steps"], a["iteration"]), (b["env_steps"], b["iteration"])
    return "same" if key_a == key_b else "b" if key_b > key_a else "a"


def decide(verdict: dict, more_rl: str | None = None, unknown: str = "") -> tuple[bool, str]:
    """(B replaces A?, why) from the verdict of `training.evaluate.compare`.

    B replaces A when it is better by more than noise. Otherwise B replaces A only in a tie that
    goes its way: B is not worse by more than noise, B measured no more than the tie band
    (`evaluate.NOISE_POINTS`, the plan's one composite point) below A, and B is the one with more
    reinforcement learning (`more_rl == "b"`). The noise margin is two standard errors and several
    points wide on the frozen suites, so without the tie band a candidate measured well below the
    shipped agent would replace it, and repeated tie promotions could only lower the shipped score.
    A comparison across different suites or seeds decides nothing: A stays.
    `unknown` replaces the reason given when a tie is left with A because `more_rl` is None.
    """
    from training import evaluate

    band = float(evaluate.NOISE_POINTS)
    delta = float(verdict.get("delta", float("nan")))
    if any("different suites or seeds" in note for note in verdict.get("notes", ())):
        return False, "the two results come from different suites or seeds: score both on the same suite first"
    if verdict["better"]:
        return True, "B is better by more than noise"
    if verdict["worse"]:
        return False, "B is worse by more than noise"
    if delta != delta:
        return False, "the difference could not be computed"
    if delta < -band:
        return False, (f"B measured LOWER by {-delta:.2f} points. That is inside the noise margin, but it is not a tie "
                       f"(a tie is within {band:g} point): more reinforcement learning does not make up for it")
    measured = f"B measured {delta:+.2f}, not better by more than noise and not below the tie band of -{band:g}"
    if more_rl == "b":
        return True, f"a tie ({measured}), and B has had more reinforcement learning"
    if more_rl == "a":
        return False, f"a tie ({measured}), and A has had more reinforcement learning"
    return False, f"a tie ({measured}), and " + (
        unknown or "it is not known which one has had more reinforcement learning (--more-rl a|b)")


def compare(tag_a: str, tag_b: str, more_rl: str | None = None, out=None, episodes_dir=None, echo=print,
            candidates=None, trust_more_rl: bool = False) -> dict:
    """Print the comparison and the decision. Returns the verdict with "replace", "why" and "more_rl" added.

    more_rl        "a" or "b" as stated by the caller, or None.
    candidates     where the candidate directories of the two tags are looked for (default runs/promote).
    trust_more_rl  let the stated `more_rl` stand where the candidates' records say otherwise.
    Which one has had more reinforcement learning is taken from the two candidate.json files when
    both exist; a stated value that contradicts them is not used unless `trust_more_rl` is set.
    """
    from training import evaluate

    out = out or evaluate.RESULTS_CSV
    verdict = evaluate.compare(tag_a, tag_b, out, episodes_dir or evaluate.EPISODES_DIR, quiet=True)
    amount_a, amount_b = rl_on_record(tag_a, candidates), rl_on_record(tag_b, candidates)
    recorded, unknown, rl_notes = more_rl_on_record(amount_a, amount_b), "", []
    effective = more_rl
    if recorded is not None:
        facts = "; ".join(f"{tag}: {amount['env_steps']:,} env steps, iteration {amount['iteration']}"
                          for tag, amount in ((tag_a, amount_a), (tag_b, amount_b)))
        which = {"a": f"more for {tag_a} (A)", "b": f"more for {tag_b} (B)", "same": "the same for both"}[recorded]
        rl_notes.append(f"reinforcement learning on record (candidate.json): {facts}: {which}")
        if more_rl is None:
            effective = recorded if recorded in ("a", "b") else None
            unknown = "" if effective else "the records show the same amount of reinforcement learning for both"
        elif more_rl != recorded and trust_more_rl:
            rl_notes.append(f"--more-rl {more_rl} is taken as stated although the records say otherwise (--trust-more-rl)")
        elif more_rl != recorded:
            effective = recorded if recorded == "a" else None
            unknown = (f"--more-rl {more_rl} contradicts the candidates' records, which show {which} "
                       "(--trust-more-rl overrules them)")
            rl_notes.append(f"--more-rl {more_rl} contradicts those records and is not used")
    elif more_rl is not None:
        missing = [tag for tag, amount in ((tag_a, amount_a), (tag_b, amount_b)) if amount is None]
        rl_notes.append(f"--more-rl {more_rl} is taken as stated: no candidate.json with an iteration for {' and '.join(missing)}")
    replace, why = decide(verdict, effective, unknown)
    results = evaluate.read_results(out)
    a, b = evaluate._latest(results, tag_a), evaluate._latest(results, tag_b)
    evaluate.print_rows([a[t] for t in sorted(a)] + [b[t] for t in sorted(b)])
    for note in verdict["notes"] + rl_notes:
        echo(f"note: {note}")
    delta = verdict["delta"]
    side = "LOWER than" if delta < 0 else "higher than" if delta > 0 else "the same as"
    echo(f"{tag_b} - {tag_a}: {delta:+.2f} +- {verdict['se']:.2f} composite points: B measured {side} A "
         f"({'paired' if verdict['paired'] else 'unpaired'}, n = {verdict['n']}); more than {verdict['margin']:.2f} either way "
         f"is a real difference; a tie needs B no more than {float(evaluate.NOISE_POINTS):g} below A")
    echo("with the other weightings: " + ", ".join(f"{k.removeprefix('score_')} {v:+.2f}" for k, v in verdict["other"].items()))
    echo(f"DECISION: {'REPLACE ' + tag_a + ' WITH ' + tag_b if replace else 'KEEP ' + tag_a}: {why} "
         f"(B - A = {delta:+.2f}, margin {verdict['margin']:.2f})")
    return dict(verdict, replace=replace, why=why, more_rl=effective, more_rl_recorded=recorded)


# ------------------------------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.promote", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="build a candidate from a checkpoint and score it through the real agent")
    s.add_argument("checkpoint", nargs="?", help="a checkpoint of training/rl, or an exported weights file")
    s.add_argument("--candidate", help="score this existing candidate directory instead of building one")
    s.add_argument("--tag", required=True, help="name of the result in results.csv, and of the candidate directory")
    s.add_argument("--suite", default="dev", help="dev (default), heldout, dev_nav, heldout_nav, or a suite file")
    s.add_argument("--workers", type=int, help="worker processes (default 3 here, 64 with --remote)")
    s.add_argument("--seeds", default="0", help="comma-separated engine seeds")
    s.add_argument("--tiers", help="comma-separated tiers to play (default: all)")
    s.add_argument("--limit", type=int, help="play only the first N maps of every tier")
    s.add_argument("--time-scale", type=float, default=1.0, help="multiply measured agent time by this before the limits apply")
    s.add_argument("--allow-unverified", action="store_true", help="accept a checkpoint that carries no provenance record")
    s.add_argument("--allow-mismatch", action="store_true",
                   help="accept a checkpoint whose record the current code no longer reproduces (measurements only)")
    s.add_argument("--twice", action="store_true",
                   help="play everything a second time and report how many episodes differ (run-to-run noise)")
    s.add_argument("--build-only", action="store_true", help="build and verify the candidate, do not play")
    s.add_argument("--remote", action="store_true", help="play on the evaluation machine (see training/ops/remote.py)")
    s.add_argument("--out", help="results file (default training/logs/results.csv)")
    s.add_argument("--episodes-dir", help="where per-episode rows are kept (default runs/eval)")
    from training.ops import remote as remote_module

    remote_module.add_arguments(s)

    c = sub.add_parser("check", help="the ship gate for a candidate directory")
    c.add_argument("directory", help="a candidate (runs/promote/NAME) or any directory with a submission/ in it")
    c.add_argument("--allow-unverified", action="store_true", help="accept a policy file without a provenance record")
    c.add_argument("--budget-ms", type=float, default=LATENCY_BUDGET_MS, help="limit on the mean act() time on one thread")
    c.add_argument("--size-limit-mb", type=float, default=SIZE_LIMIT_MB)
    c.add_argument("--fast", action="store_true", help="leave out the tests marked slow (not the full gate)")
    c.add_argument("--keep", action="store_true", help="keep the working copy the checks ran in")

    m = sub.add_parser("compare", help="noise-aware comparison of two tags and the decision rule")
    m.add_argument("tag_a", help="the result that is shipped now")
    m.add_argument("tag_b", help="the result of the candidate")
    m.add_argument("--more-rl", choices=("a", "b"),
                   help="which of the two has had more reinforcement learning (decides ties); read from the candidates' "
                        "candidate.json files when both exist, and checked against them when given")
    m.add_argument("--trust-more-rl", action="store_true", help="let --more-rl stand where those records say otherwise")
    m.add_argument("--out", help="results file (default training/logs/results.csv)")
    m.add_argument("--episodes-dir", help="where per-episode rows are kept (default runs/eval)")
    args = ap.parse_args(argv)

    if args.command == "compare":
        return 0 if compare(args.tag_a, args.tag_b, args.more_rl, args.out, args.episodes_dir,
                            trust_more_rl=args.trust_more_rl)["replace"] else 1
    if args.command == "check":
        return 0 if check(args.directory, allow_unverified=args.allow_unverified, budget_ms=args.budget_ms,
                          size_limit_mb=args.size_limit_mb, fast=args.fast, keep=args.keep)["passed"] else 1

    if bool(args.checkpoint) == bool(args.candidate):
        ap.error("give a checkpoint to build a candidate from, or --candidate DIR to score an existing one")
    started = time.perf_counter()
    if args.candidate:
        home = Path(args.candidate)
    else:
        mode = "warn" if args.allow_mismatch else "strict" if args.allow_unverified else "require"
        try:
            build_candidate(args.checkpoint, args.tag, mode)
        except Exception as exc:  # noqa: BLE001 - the export's refusal is the answer; show it without a traceback
            if type(exc).__name__ not in ("CheckpointMismatch", "ValueError", "RuntimeError", "FileNotFoundError"):
                raise
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 3
        home = CANDIDATES / _safe(args.tag)
    if args.build_only:
        return 0
    seeds = [int(x) for x in args.seeds.split(",")]
    tiers = [int(t) for t in args.tiers.split(",")] if args.tiers else None
    common = dict(seeds=seeds, tiers=tiers, limit=args.limit, time_scale=args.time_scale, out=args.out,
                  episodes_dir=args.episodes_dir, twice=args.twice)
    if args.remote:
        try:
            box = remote_module.Box.from_settings(args.box, args.box_dir, args.box_python)
            rows = score_remote(home, args.tag, args.suite, box, workers=args.workers or remote_module.MAX_REMOTE_WORKERS, **common)
        except remote_module.RemoteError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        rows = score_local(home, args.tag, args.suite, workers=args.workers or 3, **common)
    overall = rows[-1]
    failed = int(float(overall.get("agent_errors") or 0)) + int(float(overall.get("errors") or 0))
    print(f"scored {args.tag} in {time.perf_counter() - started:.0f} s"
          + (f"; WARNING: {failed} episodes had the agent fail or could not be played" if failed else ""))
    print(f"next: python -m training.promote check {_relative(home)}    and    python -m training.promote compare SHIPPED_TAG {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
