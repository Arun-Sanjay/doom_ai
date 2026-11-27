"""Fill the facts of the final agent into the judged documents.

`docs/brief/brief.md` and `README.md` are final except for tokens of the form {{FINAL_NAME}}, which
stand for facts that exist only once the final agent has been chosen: its numbers, and the sentences
that say what its weights are and how it was chosen. This script replaces them from a JSON file and
refuses to write anything unless every token gets a value.

    python docs/brief/fill_final.py --tokens [--from DIR]      # every token: where it is used, what it must contain
    python docs/brief/fill_final.py --check                    # exit status 1 if a token is still in a document
    python docs/brief/fill_final.py --draft --dev-tag TAG --report FILE --report-name NAME --holder NAME \\
        --picked-at "6:42 AM" [--main-ckpt FILE | --main-steps N] [--branch-ckpt FILE | --branch-steps N] \\
        [--blend-weights MAIN,BRANCH] [--decision TEXT] [--decision-short TEXT] [--choice-extra TEXT] [--gate-passed]
                                                               # writes docs/brief/final_values.json from the logs
    python docs/brief/fill_final.py docs/brief/final_values.json --from DIR --build
                                                               # fills both documents from the token sources in DIR
                                                               # (DIR/README.md, DIR/brief.md), then rebuilds brief.pdf
    python docs/brief/fill_final.py docs/brief/final_values.json --from DIR --verify
                                                               # exit status 1 unless the documents are exactly the
                                                               # token sources filled with that file; writes nothing
    python docs/brief/fill_final.py --manager-probe [FILE]     # the two manager entropies of a policy file

Without `--from` the documents themselves are the token sources and are filled in place, which
works once. With `--from DIR` the sources stay as they are, so the documents can be filled again
when another agent is chosen.

The values file is one JSON object, token name (without braces) -> text. Keys that start with "_"
are notes and are ignored.

`--draft` reads what can be read from files, so that numbers are copied by code and not by hand:

- the per-tier rows of `--dev-tag` in training/logs/results.csv (it must be a run on dev_v1);
- from a final-pick report (`--report`; a file that holds a `python -m training.final_pick report`
  with the agent called `--report-name` and the holder it challenged, `--holder`): the HEADLINE
  block, the paired differences of the two, the DECISION line, the number of engine seeds, and,
  when the file has them, the organizer-style table and its paired rows (heldout_ref_v1);
- the shipped policy file: its sha256, its decoding setting, the step counter stored in it, and the
  entropies of the manager's two heads on the recorded inputs of the provenance probe (this part
  needs torch and the repository's own modules);
- what the weights are made of. A single checkpoint of the main run of phase two: nothing to give,
  the step counter of the policy file is used (or `--main-steps`, `--main-ckpt`). A weight average
  of a checkpoint of the main run and one of its branch (training/configs/clone_mix_full.json):
  the two step counts from `--main-steps` and `--branch-steps`, or read from the two checkpoints
  with `--main-ckpt` and `--branch-ckpt`, and `--blend-weights MAIN,BRANCH` (default 0.5,0.5).
  Given the two checkpoint files, the draft averages them again with `training.rl.average` and
  stops unless the result is the shipped state dict, tensor for tensor.
  `--lineage FILE` reads the same from a JSON file with the keys main_steps, branch_steps and
  blend_weights. The sentences that describe the weights are composed from these numbers.

Whatever it cannot read is written as "TODO: ..." and has to be typed in. `--decision` is the
sentence that says what was decided and why; without it the draft writes what the rule said.
Read the draft before filling.

Filling fails, and changes no file, when a token used in a document has no value, a value is empty,
still says TODO, spans several lines or holds a "|" (it would break a table), the file has a key
that is not a token, or anything shaped like a token would be left behind. With `--build` the
filled brief is first built aside by build_brief.py --strict, which fails on more than two pages or
on a token left over: a brief that fails changes no file either. Then the documents are written
and brief.pdf is rebuilt; the exit status is that of the build.

Standard library only (torch is imported by `--draft` and `--manager-probe` alone, and only if it is installed).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCUMENTS = (HERE / "brief.md", ROOT / "README.md")
VALUES = HERE / "final_values.json"
RESULTS = ROOT / "training" / "logs" / "results.csv"
POLICY = ROOT / "submission" / "weights" / "policy.pt"

TOKEN = re.compile(r"\{\{(FINAL_[A-Z0-9_]+)\}\}")
LEFTOVER = re.compile(r"\{\{\s*FINAL[^}]*\}*")        # anything that still looks like a token, well-formed or not

# Steps of the runs before the second phase (step counters stored in their last checkpoints), the
# second-phase step at which the teacher's weight reached zero (training/logs/unified_full_2_train_log.csv),
# the size of one iteration, and the second-phase step at which the branch of
# training/configs/clone_mix_full.json left the main run (iteration 1,275).
COMBAT_STEPS = 118_784_000
PHASE_ONE_STEPS = 48_955_392
TEACHER_OFF_STEP = 44_236_800
STEPS_PER_ITERATION = 98_304
BRANCH_SPLIT_STEP = 125_337_600
REF_MAPS = 240                                          # maps of training/suites/heldout_ref_v1.jsonl

# name -> what it must contain (with an example of the form)
TOKENS = {
    "FINAL_TEAM_NAME": "team name as registered with the organizers (brief title line)",
    "FINAL_AGENT": "name of the final agent as it is tagged in the logs, e.g. blend_final_M",
    "FINAL_PICKED_AT": "clock time of the final choice, e.g. 6:42 AM",
    "FINAL_POLICY_SHA": "first 12 hex digits of the sha256 of submission/weights/policy.pt (the agent prints them at start)",
    "FINAL_DECODING": "decoding setting of the shipped policy file as a clause, e.g. USE and weapon take their most likely "
                      "option, the other heads and the manager's choices are sampled",
    "FINAL_RL_ALONE": "clause: how many steps of reinforcement learning alone are behind the shipped weights after the "
                      "teacher's weight reached zero at 44.2 M; for an average, the count of each branch and what they share",
    "FINAL_LADDER_TRAINING": "ladder cell 'training behind it', e.g. phase two, 196.6 M; or the two checkpoints of an average",
    "FINAL_LADDER_SHORT": "the same cell for the brief's narrower table, at most about 30 characters",
    "FINAL_WEIGHTS_ARE": "README: what the shipped policy file is (a checkpoint, or an average of which checkpoints made how), no final stop",
    "FINAL_TOTAL_STEPS": "README: sentence without final stop on all environment steps behind the shipped weights",
    "FINAL_WEIGHTS_SHORT": "brief: what the shipped policy file is, in one clause",
    "FINAL_REPRO": "README: which reproduction steps give the shipped weights, with iterations and weights, no final stop",
    "FINAL_CHOICE": "README: the choice in sentences: the holder, the numbers of both, what the rule said, what was decided and why",
    "FINAL_CHOICE_SHORT": "brief: the same in two or three sentences",
    "FINAL_LIMIT_WEIGHTS": "README, known limits: a bullet's text (bold title first) on what the shipped weights are, what was "
                           "measured on them and what was not",
    "FINAL_LIMIT_WEIGHTS_SHORT": "brief: the same bullet, short",
    "FINAL_DEV_T1": "dev_v1 composite of tier 1, one decimal (training/logs/results.csv)",
    "FINAL_DEV_T2": "dev_v1 composite of tier 2",
    "FINAL_DEV_T3": "dev_v1 composite of tier 3",
    "FINAL_DEV_T4": "dev_v1 composite of tier 4",
    "FINAL_DEV_ALL": "dev_v1 tier-weighted composite with its standard error, e.g. 77.8 ± 3.8",
    "FINAL_DEV_EXIT": "dev_v1 exit rate on maps with an exit, two decimals, e.g. 0.82",
    "FINAL_DEV_KILLS": "dev_v1 kills over monsters, two decimals, e.g. 0.78",
    "FINAL_DEV_DEATHS": "dev_v1 death rate, two decimals, e.g. 0.14",
    "FINAL_SEEDS": "engine seeds behind the final agent's held-out numbers, in words, e.g. three engine seeds on the two "
                   "generated suites and five on the two authored ones",
    "FINAL_SEEDS_N": "engine seeds on the generated suites as a number, e.g. 3",
    "FINAL_EPISODES": "held-out episodes behind the final agent's held-out numbers, organizer-style suite included, e.g. 3,115",
    "FINAL_HEADLINE": "weighted headline of the final-pick report with its standard error, e.g. 75.5 ± 1.1",
    "FINAL_HELDOUT_V1": "weighted score on heldout_v1 in the same report, e.g. 85.6 ± 1.2",
    "FINAL_HELDOUT_WIDE": "weighted score on heldout_wide_v1",
    "FINAL_AUTHORED": "weighted score on authored_v1",
    "FINAL_AUTHORED_HARD": "weighted score on authored_hard_v1",
    "FINAL_REF": "tier-weighted score on the organizer-style suite heldout_ref_v1, all 240 maps, e.g. 75.0 ± 1.8",
    "FINAL_REF_NORMAL": "the same on its normal half (120 maps)",
    "FINAL_REF_DENSE": "the same on its dense half (120 maps)",
    "FINAL_MGR_SUBGOAL_ENT": "entropy of the manager's sub-goal head in the shipped weights on the probe inputs, nats, e.g. 2.04",
    "FINAL_MGR_TARGET_ENT": "entropy of the manager's target pointer in the shipped weights on the probe inputs, nats, e.g. 6.74",
    "FINAL_MGR_TARGET_CELLS": "that target entropy as an even spread over so many cells (e to its power), e.g. about 850",
}
REPORT_ROWS = {"headline": "FINAL_HEADLINE", "heldout_v1": "FINAL_HELDOUT_V1", "heldout_wide_v1": "FINAL_HELDOUT_WIDE",
               "authored_v1": "FINAL_AUTHORED", "authored_hard_v1": "FINAL_AUTHORED_HARD"}
REF_ROWS = {"both": "FINAL_REF", "normal": "FINAL_REF_NORMAL", "dense": "FINAL_REF_DENSE"}
SUITES = ("heldout_v1", "heldout_wide_v1", "authored_v1", "authored_hard_v1")
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
# What a holder is, for the sentence that names it (agents that have held the place; --holder-is overrides).
HOLDERS = {
    "main_0525_M": "the main run at 172.0 M steps of phase two, shipped at 5:37 AM",
    "blend_final_M": "the equal-weight average of the main run at 196.6 M steps and its branch at 164.7 M, shipped at 6:42 AM "
                     "by a decision outside the rule",
}
HOLDERS_SHORT = {"main_0525_M": "the 5:37 AM agent", "blend_final_M": "the average of two branches shipped at 6:42 AM"}


def sources(source_dir: Path | None) -> dict[Path, Path]:
    """document -> the file its text with tokens is read from (the document itself without a source directory)."""
    return {path: (Path(source_dir) / path.name if source_dir else path) for path in DOCUMENTS}


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def uses(source_dir: Path | None = None) -> dict[str, list[str]]:
    """token name -> the places it appears, as 'path:line'."""
    found: dict[str, list[str]] = {}
    for path in sources(source_dir).values():
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for name in TOKEN.findall(line):
                found.setdefault(name, []).append(f"{_shown(path)}:{number}")
    return found


def leftovers() -> list[str]:
    hits = []
    for path in DOCUMENTS:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            hits += [f"{path.relative_to(ROOT).as_posix()}:{number}: {m.group(0)}" for m in LEFTOVER.finditer(line)]
    return hits


def show_tokens(source_dir: Path | None) -> int:
    used = uses(source_dir)
    for name, what in TOKENS.items():
        print(f"{{{{{name}}}}}\n    {what}\n    used in: {', '.join(used.get(name, [])) or 'no document'}")
    unknown = sorted(set(used) - set(TOKENS))
    for name in unknown:
        print(f"{{{{{name}}}}}\n    NOT IN THE LIST OF THIS SCRIPT; used in: {', '.join(used[name])}")
    unused = sorted(set(TOKENS) - set(used)) if used else []
    print(f"{len(used)} token(s) in the documents, {len(TOKENS)} known"
          + (f"; known but not used: {', '.join(unused)}" if unused else ""))
    return 1 if unknown else 0


def check() -> int:
    hits = leftovers()
    for hit in hits:
        print(hit)
    print(f"{len(hits)} token(s) left in {', '.join(p.name for p in DOCUMENTS)}")
    return 1 if hits else 0


# ---------------------------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------------------------
def filled_texts(values_file: Path, source_dir: Path | None) -> tuple[dict[Path, str], dict[Path, int], list[str]]:
    """(document -> filled text, document -> tokens replaced, problems). Nothing is written."""
    try:
        raw = json.loads(values_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {}, {}, [f"cannot read {values_file}: {error}"]
    if not isinstance(raw, dict):
        return {}, {}, [f"{values_file} must hold one JSON object"]
    missing = [str(src) for src in sources(source_dir).values() if not src.is_file()]
    if missing:
        return {}, {}, [f"token source not found: {', '.join(missing)}"]
    values = {k: v for k, v in raw.items() if not k.startswith("_")}
    used = uses(source_dir)
    problems = [f"no value for {{{{{name}}}}} (used in {', '.join(where)})" for name, where in sorted(used.items())
                if name not in values]
    problems += [f"'{name}' in {values_file.name} is not a token of this script" for name in sorted(set(values) - set(TOKENS))]
    for name, value in sorted(values.items()):
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name}: the value must be a non-empty string")
        elif "TODO" in value or "{{" in value or "|" in value or "\n" in value:
            problems.append(f"{name}: the value still says TODO, or holds '{{{{', '|' or a line break: {value[:60]!r}")
    if not used:
        problems.append("no token is left in the documents: they are already filled (fill from the token sources: --from DIR)")
    if problems:
        return {}, {}, problems
    filled, counts = {}, {}
    for path, src in sources(source_dir).items():
        text, counts[path] = TOKEN.subn(lambda m: values[m.group(1)].strip(), src.read_text(encoding="utf-8"))
        rest = LEFTOVER.findall(text)
        if rest:
            return {}, {}, [f"{path.name} would still hold {rest[:5]}"]
        filled[path] = text
    return filled, counts, []


def brief_fits(text: str) -> tuple[bool, str]:
    """(would build_brief.py --strict accept this brief?, its output). The trial PDF goes to a temporary folder."""
    with tempfile.TemporaryDirectory(prefix="brief_trial_") as tmp:
        trial = Path(tmp) / "brief.md"
        trial.write_text(text, encoding="utf-8")
        done = subprocess.run([sys.executable, str(HERE / "build_brief.py"), "--strict", "--source", str(trial),
                               "--out", str(Path(tmp) / "brief.pdf")], capture_output=True, text=True, check=False)
    return done.returncode == 0, (done.stdout + done.stderr).strip()


def fill(values_file: Path, build: bool, source_dir: Path | None = None) -> int:
    filled, counts, problems = filled_texts(values_file, source_dir)
    if not problems and build:            # the brief is built aside first: a brief that is too long changes no file
        fits, output = brief_fits(filled[HERE / "brief.md"])
        if not fits:
            problems = ["the filled brief does not pass build_brief.py --strict:"] + ["  " + line for line in output.split("\n")]
    if problems:
        print("FAILED, nothing was written:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    for path, text in filled.items():
        path.write_text(text, encoding="utf-8")
        origin = f" (from {_shown(sources(source_dir)[path])})" if source_dir else ""
        print(f"{path.relative_to(ROOT).as_posix()}: {counts[path]} token(s) replaced{origin}", flush=True)
    if build:
        return subprocess.run([sys.executable, str(HERE / "build_brief.py"), "--strict"], check=False).returncode
    print("now rebuild the PDF: python docs/brief/build_brief.py --strict")
    return 0


def verify(values_file: Path, source_dir: Path | None) -> int:
    """Exit status 0 only if the documents are exactly the token sources filled with `values_file`."""
    if source_dir is None:
        print("FAILED: --verify needs --from DIR, the folder of the token sources", file=sys.stderr)
        return 1
    filled, counts, problems = filled_texts(values_file, source_dir)
    if problems:
        print("FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    status = 0
    for path, text in filled.items():
        now = path.read_text(encoding="utf-8")
        name = path.relative_to(ROOT).as_posix()
        if now == text:
            print(f"{name}: identical to {_shown(sources(source_dir)[path])} filled with {values_file.name} ({counts[path]} tokens)")
            continue
        status = 1
        diff = list(difflib.unified_diff(text.split("\n"), now.split("\n"), "sources filled", name, lineterm="", n=0))
        print(f"{name}: DIFFERS from {_shown(sources(source_dir)[path])} filled with {values_file.name}:\n" + "\n".join(diff[:40]))
    return status


# ---------------------------------------------------------------------------------------------
# Draft of the values file: numbers from the logs
# ---------------------------------------------------------------------------------------------
def rounded(value: float, digits: int = 1) -> str:
    """Rounded half up on the decimal text, so that 75.25 reads 75.3 and not what its binary form happens to give.

    A report number is itself rounded to two decimals, so a 5 in its second decimal is a tie that this
    function cannot settle: 75.25 may stand for 75.2494, which is 75.2. `--draft` lists such values (`is_tie`).
    """
    return str(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def is_tie(value: float) -> bool:
    """Does the two-decimal text of a report number end in 5? Its one-decimal form then needs the unrounded number."""
    return f"{float(value):.2f}".endswith("5")


def pm(value: float, se: float, digits: int = 1) -> str:
    return f"{rounded(value, digits)} ± {rounded(se, digits)}"


def dev_values(tag: str) -> dict[str, str]:
    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["tag"] == tag]
    if not rows:
        raise SystemExit(f"no rows tagged '{tag}' in {RESULTS}")
    last = max(r["stamp"] for r in rows)
    rows = {r["tier"]: r for r in rows if r["stamp"] == last}
    suites = {r["suite"] for r in rows.values()}
    if suites != {"dev_v1.jsonl"} or set(rows) != {"1", "2", "3", "4", "all"}:
        raise SystemExit(f"tag '{tag}' is not a complete run on dev_v1.jsonl (suites {sorted(suites)}, tiers {sorted(rows)})")
    out = {f"FINAL_DEV_T{t}": f"{float(rows[t]['score']):.1f}" for t in "1234"}
    every = rows["all"]
    out["FINAL_DEV_ALL"] = pm(float(every["score"]), float(every["score_se"]))
    out["FINAL_DEV_EXIT"] = f"{float(every['exit_rate']):.2f}"
    out["FINAL_DEV_KILLS"] = f"{float(every['kills_frac']):.2f}"
    out["FINAL_DEV_DEATHS"] = f"{float(every['death_rate']):.2f}"
    out["_dev_episodes"] = str(int(every["episodes"]))
    return out


def _number(text: str) -> float:
    return float(text.replace("+-", "").replace("±", ""))


def headline_rows(text: str, names: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """agent -> {row of the first HEADLINE block: (weighted score, standard error)}."""
    out: dict[str, dict] = {name: {} for name in names}
    label, inside = "", False
    for line in text.split("\n"):
        if line.startswith("HEADLINE"):
            inside = True
            continue
        if inside and not line.strip():
            break
        parts = line.split()
        if not inside or not parts:
            continue
        if parts[0] in REPORT_ROWS:              # the first agent's line of a row carries the row's name
            label = parts[0]
        for name in names:
            if name in parts and label in REPORT_ROWS:
                at = parts.index(name)
                out[name][label] = (float(parts[at + 1]), _number(parts[at + 2]))
    return out


PAIR_ROW = re.compile(r"^(.*?)\s+(\d+)\s+([+-]\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S.*?)\s*$")


def paired_rows(text: str, agent: str, holder: str) -> dict[str, dict]:
    """row label -> {"delta", "se", "margin", "verdict"} of the block `PAIRED  B = agent  minus  A = holder`."""
    start = re.search(rf"^PAIRED\s+B = {re.escape(agent)}\s+minus\s+A = {re.escape(holder)}\b.*$", text, re.M)
    out = {}
    if not start:
        return out
    for line in text[start.end():].split("\n")[1:]:
        if not line.strip():
            break
        m = PAIR_ROW.match(line)
        if m:
            out[m.group(1).strip()] = {"delta": float(m.group(3)), "se": float(m.group(4)), "margin": float(m.group(5)),
                                       "verdict": m.group(6)}
    return out


def decision_line(text: str, agent: str, holder: str) -> str:
    m = re.search(rf"^DECISION \(A = {re.escape(holder)}, B = {re.escape(agent)}\): (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def ref_rows(text: str, names: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """agent -> {"normal" | "dense" | "both": (composite, standard error)} from the `all` rows of the first organizer-style table."""
    out: dict[str, dict] = {name: {} for name in names}
    start = re.search(r"^ORGANIZER-STYLE MAPS.*$", text, re.M)
    if not start:
        return out
    tier = half = ""
    for line in text[start.end():].split("\n")[1:]:
        if not line.strip():
            break
        parts = line.split()
        if not line.startswith(" ") and len(parts) >= 6 and parts[2].isdigit():
            tier, half, parts = parts[0], parts[1], parts[3:]
        if tier == "all" and len(parts) >= 3 and parts[0] in out:
            out[parts[0]][half] = (float(parts[1]), _number(parts[2]))
    return out


def ref_paired(text: str, agent: str, holder: str) -> dict[str, dict]:
    """half -> {"delta", "se", "margin", "verdict"} of `PAIRED on heldout_ref_v1: B minus A = holder` for B = agent."""
    start = re.search(rf"^PAIRED on heldout_ref_v1: B minus A = {re.escape(holder)}\b.*$", text, re.M)
    out = {}
    if not start:
        return out
    for line in text[start.end():].split("\n")[1:]:
        if not line.strip() or line.startswith(("PAIRED", "=")):
            break
        m = PAIR_ROW.match(line)
        if m and m.group(1).split()[:1] == [agent]:
            out[m.group(1).split()[1]] = {"delta": float(m.group(3)), "se": float(m.group(4)), "margin": float(m.group(5)),
                                          "verdict": m.group(6)}
    return out


def suite_seeds(text: str) -> dict[str, tuple[int, int]]:
    """suite -> (maps, seeds) from the report's `suites (weight, maps x seeds)` line."""
    line = re.search(r"^suites \(weight, maps x seeds\): (.*)$", text, re.M)
    return {m.group(1): (int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"(\w+) \([\d.]+, (\d+) x (\d+)\)", line.group(1))} if line else {}


def read_report(report: Path, agent: str, holder: str | None, ref_seeds: int | None) -> dict:
    """Everything the draft takes from a final-pick report, as numbers."""
    text = report.read_text(encoding="utf-8")
    names = [agent] + ([holder] if holder else [])
    head = headline_rows(text, names)
    missing = sorted(set(REPORT_ROWS) - set(head[agent]))
    if missing:
        raise SystemExit(f"{report}: no HEADLINE rows for '{agent}' ({', '.join(missing)} not found)")
    if holder and set(REPORT_ROWS) - set(head[holder]):
        raise SystemExit(f"{report}: no HEADLINE rows for the holder '{holder}'")
    seeds = suite_seeds(text)
    if set(seeds) != set(SUITES):
        raise SystemExit(f"{report}: no `suites (weight, maps x seeds)` line for the four suites")
    generated = {seeds["heldout_v1"][1], seeds["heldout_wide_v1"][1]}
    drawn = {seeds["authored_v1"][1], seeds["authored_hard_v1"][1]}
    if len(generated) != 1 or len(drawn) != 1:
        raise SystemExit(f"{report}: the suites of one kind do not have the same number of seeds: {seeds}")
    ref = ref_rows(text, names)
    out = {"head": head, "seeds": seeds, "generated_seeds": generated.pop(), "drawn_seeds": drawn.pop(),
           "ref": ref if set(ref[agent]) == set(REF_ROWS) else None}
    out["ref_seeds"] = (ref_seeds or out["generated_seeds"]) if out["ref"] else 0
    out["episodes"] = sum(maps * n for maps, n in seeds.values()) + REF_MAPS * out["ref_seeds"]
    if holder:
        out["pairs"] = paired_rows(text, agent, holder)
        if "headline (weighted)" not in out["pairs"]:
            raise SystemExit(f"{report}: no block `PAIRED  B = {agent}  minus  A = {holder}`")
        out["decision"] = decision_line(text, agent, holder)
        out["ref_pairs"] = ref_paired(text, agent, holder) if out["ref"] and set(ref[holder]) == set(REF_ROWS) else {}
    return out


def decoding_clause(greedy) -> str:
    order = ("move", "strafe", "turn", "attack", "use", "weapon")
    greedy = ["use", "weapon"] if greedy is None else list(greedy)       # the agent's default (agent.GREEDY_HEADS)
    shown = {"use": "USE"}
    likely = [shown.get(h, h) for h in order if h in greedy]
    drawn = [shown.get(h, h) for h in order if h not in greedy]

    def listed(names: list[str]) -> str:
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    if likely and drawn:
        return (f"{listed(likely)} take their most likely option, {listed(drawn)} "
                f"{'is' if len(drawn) == 1 else 'are'} sampled, and so are the manager's choices")
    if likely:
        return "all six heads take their most likely option, the manager's choices are sampled"
    return "all six heads and the manager's choices are sampled"


def manager_probe(policy: Path) -> tuple[float, float]:
    """Mean entropy (nats) of the manager's sub-goal head and target pointer on the provenance probe's recorded inputs.

    The method of part (a) of training/logs/manager_probe.txt: the network of `policy` is run on the two recorded
    episodes of training/rl/probe_episodes.npz with its own recurrent state and nothing forced, and the entropies
    are averaged over the manager steps (10 of the 80 decisions). A small sample; it shows whether the two heads
    are near uniform (ln 8 = 2.08 and ln 4096 = 8.32), not more.
    """
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from training import paths  # noqa: F401 - puts submission/ on the import path
    from doomnet import spec
    from training.rl import deps, provenance

    blob = torch.load(str(policy), map_location="cpu", weights_only=True)
    net = deps.make_policy("real", **blob["config"]["policy_kwargs"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    recorded = provenance.probe()
    inp = {k: torch.from_numpy(recorded[k]) for k in ("rays", "ents", "ent_mask", "scal", "map", "masks")}
    episodes, steps = recorded["scal"].shape[:2]
    torch.manual_seed(0)
    subgoal, target = [], []
    with torch.no_grad():
        state = net.initial_state(episodes, torch.device("cpu"))
        for t in range(steps):
            manager_step = t % spec.MANAGER_PERIOD == 0
            out, state = net(inp["rays"][:, t], inp["ents"][:, t], inp["ent_mask"][:, t], inp["scal"][:, t], inp["map"][:, t],
                             state=state, masks=inp["masks"][:, t], run_manager=torch.full((episodes,), manager_step),
                             forced=None, sample=False)
            if manager_step:
                for logits, kept in ((out["subgoal_logits"], subgoal), (out["target_logits"], target)):
                    logp = torch.log_softmax(logits.float(), 1)
                    kept.append(-(logp.exp() * logp).sum(1))
    return float(torch.stack(subgoal).mean()), float(torch.stack(target).mean())


def policy_values(policy: Path) -> dict[str, str]:
    """sha256, decoding and manager entropies of the policy file; "_env_steps" is the step counter stored in it."""
    out = {"FINAL_POLICY_SHA": hashlib.sha256(policy.read_bytes()).hexdigest()[:12]}
    try:
        import torch
        config = torch.load(policy, map_location="cpu", weights_only=False)["config"]
        steps = int(config["env_steps"])
    except Exception as error:  # noqa: BLE001 - the draft then says what has to be typed in
        print(f"note: step count and decoding not read from {policy.name} ({type(error).__name__}: {error})")
        return out
    out["_env_steps"] = str(steps)
    out["FINAL_DECODING"] = decoding_clause(config.get("greedy_heads"))
    out["_policy_file"] = (f"{policy.name}: source {config.get('source')}, iteration {config.get('iteration')}, "
                           f"env_steps {steps}, greedy_heads {config.get('greedy_heads')}")
    try:
        subgoal, target = manager_probe(policy)
    except Exception as error:  # noqa: BLE001
        print(f"note: manager entropies not computed for {policy.name} ({type(error).__name__}: {error})")
        return out
    out["FINAL_MGR_SUBGOAL_ENT"] = f"{subgoal:.2f}"
    out["FINAL_MGR_TARGET_ENT"] = f"{target:.2f}"
    cells = math.exp(target)
    out["FINAL_MGR_TARGET_CELLS"] = "about " + f"{float(f'{cells:.2g}'):,.0f}"
    out["_manager_probe"] = f"sub-goal entropy {subgoal:.3f}, target entropy {target:.3f} nats (exp = {cells:.0f} cells)"
    return out


# ---------------------------------------------------------------------------------------------
# Draft of the values file: the sentences on what the weights are and how they were chosen
# ---------------------------------------------------------------------------------------------
def million(steps: int, digits: int = 1) -> str:
    return f"{steps / 1e6:.{digits}f}"


def checkpoint_steps(path: str) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["env_steps"])


def average_matches(policy: Path, files: list[str], weights: list[float]) -> tuple[int, int]:
    """(tensors of the shipped state dict that `training.rl.average` reproduces bit for bit from `files`, tensors in all)."""
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from training.rl import average

    states = [torch.load(f, map_location="cpu", weights_only=False)["model"] for f in files]
    mean = average.average_state(states, weights)
    shipped = torch.load(str(policy), map_location="cpu", weights_only=True)["state_dict"]
    same = sum(int(k in mean and mean[k].dtype == v.dtype and torch.equal(mean[k], v)) for k, v in shipped.items())
    return (same if list(mean) == list(shipped) else 0), len(shipped)


def lineage(args: argparse.Namespace, policy_steps: int | None) -> dict | None:
    """{"main": steps or None, "branch": steps or None, "weights": (main, branch)}: what the shipped weights are made of."""
    given = {"main_steps": args.main_steps, "branch_steps": args.branch_steps, "blend_weights": args.blend_weights}
    if args.lineage:
        stored = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
        for key in given:
            stored_key = next((k for k in (key, "_" + key) if stored.get(k) is not None), None)
            if given[key] is None and stored_key:
                given[key] = stored[stored_key]
    main = checkpoint_steps(args.main_ckpt) if args.main_ckpt else given["main_steps"]
    branch = checkpoint_steps(args.branch_ckpt) if args.branch_ckpt else given["branch_steps"]
    main, branch = (None if v is None else int(v) for v in (main, branch))
    if main is None and branch is None:
        main = policy_steps                               # a single checkpoint of the main run: the file's own counter
    if main is None and branch is None:
        return None
    weights = given["blend_weights"]
    if isinstance(weights, str):
        weights = [float(w) for w in weights.split(",")]
    weights = [0.5, 0.5] if weights is None else [float(w) for w in weights]
    if len(weights) != 2 or min(weights) <= 0:
        raise SystemExit(f"--blend-weights takes two positive numbers, MAIN,BRANCH: {weights}")
    for name, steps in (("main", main), ("branch", branch)):
        if steps is not None and (steps % STEPS_PER_ITERATION or steps <= TEACHER_OFF_STEP):
            raise SystemExit(f"{name} steps {steps}: not a whole number of iterations of {STEPS_PER_ITERATION} steps after "
                             f"the teacher's weight reached zero; give environment steps, e.g. 196608000")
    if branch is not None and branch <= BRANCH_SPLIT_STEP:
        raise SystemExit(f"branch steps {branch}: the branch left the main run at {BRANCH_SPLIT_STEP}")
    if main is not None and branch is not None and policy_steps not in (None, main, branch):
        raise SystemExit(f"the policy file stores {policy_steps} steps, which is neither the main run's {main} nor the "
                         f"branch's {branch}: these are not the checkpoints the shipped file was averaged from")
    if main is not None and branch is None and policy_steps not in (None, main):
        raise SystemExit(f"the policy file stores {policy_steps} steps, not {main}")
    made = {"main": main, "branch": branch, "weights": tuple(w / sum(weights) for w in weights), "last": policy_steps,
            "verified": 0}
    if main is not None and branch is not None and args.main_ckpt and args.branch_ckpt and Path(args.policy).exists():
        # The two files are averaged again here and compared with the shipped state dict, tensor by tensor.
        main_last = policy_steps in (None, main)
        files = [args.branch_ckpt, args.main_ckpt] if main_last else [args.main_ckpt, args.branch_ckpt]
        order = [weights[1], weights[0]] if main_last else [weights[0], weights[1]]
        same, total = average_matches(Path(args.policy), files, order)
        if same != total:
            raise SystemExit(f"the average of {files[0]} and {files[1]} with weights {order} is not the shipped state dict: "
                             f"{same} of {total} tensors are identical. Wrong files or weights?")
        made["verified"] = total
    return made


def weights_sentences(made: dict) -> dict[str, str]:
    """The tokens that say what the shipped weights are, for a single checkpoint or for an average of two branches."""
    main, branch = made["main"], made["branch"]
    before = COMBAT_STEPS + PHASE_ONE_STEPS
    if main is not None and branch is not None:
        w_main, w_branch = made["weights"]
        equal = abs(w_main - w_branch) < 1e-9
        average = "the equal-weight average" if equal else f"the weighted average (weights {w_main:g} and {w_branch:g})"
        own, shared = branch - BRANCH_SPLIT_STEP, BRANCH_SPLIT_STEP - TEACHER_OFF_STEP
        total = before + main + own
        last = "main run's" if made.get("last") in (None, main) else "branch's"
        order = ("the branch named first, so that the file carries the main run's counters" if last == "main run's"
                 else "the main run named first, so that the file carries the branch's counters")
        files = (f"runs/clone_mix_full/ckpt_{branch // STEPS_PER_ITERATION:06d}.pt runs/unified_full_2/ckpt_{main // STEPS_PER_ITERATION:06d}.pt"
                 if last == "main run's" else
                 f"runs/unified_full_2/ckpt_{main // STEPS_PER_ITERATION:06d}.pt runs/clone_mix_full/ckpt_{branch // STEPS_PER_ITERATION:06d}.pt")
        listed = f"{w_branch:g},{w_main:g}" if last == "main run's" else f"{w_main:g},{w_branch:g}"
        return {
            "FINAL_RL_ALONE": (f"the shipped weights are {average} of two branches of that phase, with {million(main - TEACHER_OFF_STEP)} M "
                               f"and {million(branch - TEACHER_OFF_STEP)} M further steps of reinforcement learning alone behind "
                               f"them (the first {million(shared)} M shared)"),
            "FINAL_LADDER_TRAINING": f"phase two: average of the main run at {million(main)} M and its branch at {million(branch)} M",
            "FINAL_LADDER_SHORT": f"average: {million(main)} M, branch {million(branch)} M",
            "FINAL_WEIGHTS_ARE": (f"{average} of the network weights (all 73 tensors, 1,307,153 values) of two checkpoints of the same "
                                  f"network, made by `training/rl/average.py`: the main run of phase two at {million(main)} M steps, and a "
                                  f"branch that left the main run at {million(BRANCH_SPLIT_STEP)} M steps and trained on to "
                                  f"{million(branch)} M on a map set that is half organizer-style ({million(own)} M steps of its own). "
                                  f"The step counter stored in the file is the {last}"),
            "FINAL_TOTAL_STEPS": (f"Behind the two checkpoints are {million(total, 0)} M environment steps: combat {million(COMBAT_STEPS)} M, "
                                  f"phase one {million(PHASE_ONE_STEPS)} M, the main run of phase two {million(main)} M and the "
                                  f"branch's own {million(own)} M"),
            "FINAL_WEIGHTS_SHORT": (f"{average} of two checkpoints of the same network, the main run at {million(main)} M steps of "
                                    f"phase two and its organizer-style branch at {million(branch)} M"),
            "FINAL_REPRO": (f"The shipped weights are the output of step 4e with these arguments: `--weights {listed}` for `{files}`, "
                            f"that is iteration {branch // STEPS_PER_ITERATION:,} of the branch ({million(branch)} M steps) and "
                            f"iteration {main // STEPS_PER_ITERATION:,} of the main run ({million(main)} M), {order}"
                            + (f". `training.rl.average` gives the shipped tensors exactly: its average of these two checkpoints "
                               f"equals the state dict of `submission/weights/policy.pt` in all {made['verified']} tensors, bit for "
                               "bit (compared by `docs/brief/fill_final.py --draft` when this text was written). The file's sha256 "
                               "is not reproduced, because the file also stores the name and the hash of the averaged checkpoint "
                               "and a provenance record with the time it was written" if made.get("verified") else "")),
            "_kind": "average",
        }
    if main is not None:
        return {
            "FINAL_RL_ALONE": f"the shipped checkpoint has had {million(main - TEACHER_OFF_STEP)} M further steps of reinforcement learning alone",
            "FINAL_LADDER_TRAINING": f"phase two, {million(main)} M",
            "FINAL_LADDER_SHORT": f"phase two, {million(main)} M",
            "FINAL_WEIGHTS_ARE": f"the main run's checkpoint at {million(main)} M steps of phase two (iteration {main // STEPS_PER_ITERATION:,})",
            "FINAL_TOTAL_STEPS": (f"Behind it are {million(before + main, 0)} M environment steps: combat {million(COMBAT_STEPS)} M, "
                                  f"phase one {million(PHASE_ONE_STEPS)} M and phase two {million(main)} M"),
            "FINAL_WEIGHTS_SHORT": f"the main run's checkpoint at {million(main)} M steps of phase two",
            "FINAL_REPRO": (f"The shipped weights are the export of iteration {main // STEPS_PER_ITERATION:,} of step 4c "
                            f"({million(main)} M steps) with the decoding setting of step 4e; steps 4d and 4e made the weight "
                            f"average that was shipped before it"),
            "_kind": "single",
        }
    own = branch - BRANCH_SPLIT_STEP
    return {
        "FINAL_RL_ALONE": (f"the shipped checkpoint, from a branch of that phase, has had {million(branch - TEACHER_OFF_STEP)} M "
                           f"further steps of reinforcement learning alone"),
        "FINAL_LADDER_TRAINING": f"phase two, branch at {million(branch)} M",
        "FINAL_LADDER_SHORT": f"branch, {million(branch)} M",
        "FINAL_WEIGHTS_ARE": (f"the checkpoint at {million(branch)} M steps (iteration {branch // STEPS_PER_ITERATION:,}) of the branch that "
                              f"left the main run of phase two at {million(BRANCH_SPLIT_STEP)} M steps and trained on a map set that is "
                              f"half organizer-style"),
        "FINAL_TOTAL_STEPS": (f"Behind it are {million(before + branch, 0)} M environment steps: combat {million(COMBAT_STEPS)} M, "
                              f"phase one {million(PHASE_ONE_STEPS)} M and phase two {million(branch)} M, the last {million(own)} M "
                              f"of them on the branch"),
        "FINAL_WEIGHTS_SHORT": f"the checkpoint at {million(branch)} M steps of the organizer-style branch of phase two",
        "FINAL_REPRO": (f"The shipped weights are the export of iteration {branch // STEPS_PER_ITERATION:,} of step 4d "
                        f"({million(branch)} M steps) with the decoding setting of step 4e"),
        "_kind": "single",
    }


def seeds_words(found: dict) -> str:
    generated, drawn = found["generated_seeds"], found["drawn_seeds"]
    one = f"{COUNT_WORDS.get(generated, generated)} engine seed{'s' if generated != 1 else ''} on the two generated suites"
    return f"{one} and {COUNT_WORDS.get(drawn, drawn)} on the two authored ones"


def signed(value: float, se: float) -> str:
    return f"{value:+.2f} ± {se:.2f}".replace("-", "−")


def verdict_words(pair: dict) -> str:
    return {"level": "level", "better by more than noise": "better by more than noise",
            "worse by more than noise": "worse by more than noise"}.get(pair["verdict"], pair["verdict"])


def choice_sentences(args: argparse.Namespace, agent: str, found: dict, kind: str) -> dict[str, str]:
    """FINAL_CHOICE and FINAL_CHOICE_SHORT from the report's numbers and the decision that was taken."""
    holder = args.holder
    what = args.holder_is or HOLDERS.get(holder, "")
    head_a, head_b = found["head"][holder]["headline"], found["head"][agent]["headline"]
    pair = found["pairs"]["headline (weighted)"]
    absent = [s for s in SUITES if f"{s} (weighted)" not in found["pairs"]]
    if absent:
        raise SystemExit(f"the report's paired block of `{agent}` against `{holder}` has no row for {', '.join(absent)}")
    better = [s for s in SUITES if found["pairs"][f"{s} (weighted)"]["verdict"] == "better by more than noise"]
    worse = [s for s in SUITES if found["pairs"][f"{s} (weighted)"]["verdict"] == "worse by more than noise"]
    # The rule of training.final_pick.decide, applied to the report's own paired verdicts.
    replace = pair["verdict"] == "better by more than noise" and not worse
    if found["decision"] and found["decision"].startswith("REPLACE") != replace:
        raise SystemExit(f"the report's DECISION line ({found['decision'][:60]}...) contradicts its paired rows")
    rule = ("and no suite is worse by more than noise, so the rule says replace" if replace else
            "but a suite is worse by more than noise, so by the rule alone the holder would have stayed" if worse
            and pair["verdict"] == "better by more than noise" else "so by the rule alone the holder would have stayed")

    def named(suites: list[str]) -> str:
        return " and ".join(f"`{s}` ({signed(found['pairs'][s + ' (weighted)']['delta'], found['pairs'][s + ' (weighted)']['se'])})"
                            for s in suites)

    per_suite = ("No suite is worse by more than noise" if not worse else f"Worse by more than noise: {named(worse)}")
    per_suite += f"; better by more than noise: {named(better)}" if better else "; none is better by more than noise either"
    long = [f"At {args.picked_at} the holder was `{holder}`" + (f" ({what})" if what else "") + ".",
            f"On {seeds_words(found)}, the weighted headline of `{agent}` is {pm(head_b[0], head_b[1], 2)} against "
            f"{pm(head_a[0], head_a[1], 2)}, a paired difference of {signed(pair['delta'], pair['se'])} with a noise margin of "
            f"{pair['margin']:.2f}: {verdict_words(pair)}, {rule}.",
            per_suite + "."]
    short = [f"Chosen at {args.picked_at} against {args.holder_short or HOLDERS_SHORT.get(holder, f'`{holder}`')}: paired "
             f"held-out difference {signed(pair['delta'], pair['se'])}, "
             f"{verdict_words(pair)}" + ("" if replace else ", so the rule alone kept the holder")]
    if found.get("ref_pairs"):
        ref_a, ref_b, both = found["ref"][holder], found["ref"][agent], found["ref_pairs"]["both"]
        halves = "; ".join(f"{half} half {ref_b[half][0]:.2f} against {ref_a[half][0]:.2f}, "
                           f"{signed(found['ref_pairs'][half]['delta'], found['ref_pairs'][half]['se'])}, "
                           f"{verdict_words(found['ref_pairs'][half])}" for half in ("normal", "dense") if half in found["ref_pairs"])
        long.append(f"On the organizer-style suite, which is not part of the rule, it scores {ref_b['both'][0]:.2f} against "
                    f"{ref_a['both'][0]:.2f}: {signed(both['delta'], both['se'])}, {verdict_words(both)} ({halves}).")
        short[0] += f"; Org. {signed(both['delta'], both['se'])}, {verdict_words(both)}"
    decision = args.decision or ("It replaced the holder under the rule." if replace else
                                 "TODO: the rule kept the holder; say what was decided and why")
    long.append(decision)
    short[0] += "."
    short.append(args.decision_short or decision)
    if args.choice_extra:
        long.append(args.choice_extra)
    return {"FINAL_CHOICE": " ".join(long), "FINAL_CHOICE_SHORT": " ".join(short)}


def limit_sentences(args: argparse.Namespace, made: dict | None, found: dict | None, dev_episodes: str) -> dict[str, str]:
    """FINAL_LIMIT_WEIGHTS and its short form: what the shipped weights are, what was measured on them and what was not."""
    episodes = f"{found['episodes']:,}" if found else "TODO"
    measured = [f"{episodes} held-out episodes with the shipped decoding (the tables above)"]
    if dev_episodes:
        measured.append(f"the {dev_episodes} development episodes")
    if args.gate_passed:
        measured.append("the nine-check ship gate")
    measured.append("the manager's entropies in the next item")
    listed = ", ".join(measured[:-1]) + " and " + measured[-1]
    not_measured = ("a failure breakdown of its episodes (the one below is of the 1:38 AM agent), and the official maps and scorer")
    if made and made["main"] is not None and made["branch"] is not None:
        w_main, w_branch = made["weights"]
        apart = f"{million(made['main'] - BRANCH_SPLIT_STEP)} M and {million(made['branch'] - BRANCH_SPLIT_STEP)} M"
        long = ("**The shipped weights are an average that was never trained as one network.** They are the mean of two "
                "branches' weights: no gradient step was taken at these values, and the mean of two networks that play well "
                f"need not play well. The two branches share their training up to {million(BRANCH_SPLIT_STEP)} M steps of phase two "
                f"and differ by the {apart} steps after it; we did not test averages of networks further apart. Measured on the "
                f"averaged weights, through the real agent: {listed}. Not measured: any other weights than {w_main:g} and "
                f"{w_branch:g}, {not_measured}")
        short = ("**Limit: the shipped weights are an average** of two branches' weights and were never trained as one network. "
                 f"Measured on them: {episodes} held-out episodes" + (", the ship gate" if args.gate_passed else "")
                 + f", the manager's entropies. Not measured: other weights than {w_main:g} and {w_branch:g}, a failure "
                 "breakdown, the official maps")
    else:
        long = ("**What was measured on the shipped weights.** They are one checkpoint of one run. Measured on them, through "
                f"the real agent: {listed}. Not measured: {not_measured}")
        short = (f"**Shipped weights.** One checkpoint; measured on it: {episodes} held-out episodes"
                 + (", the ship gate" if args.gate_passed else "") + ", the manager's entropies. Not measured: a failure "
                 "breakdown, the official maps")
    return {"FINAL_LIMIT_WEIGHTS": long, "FINAL_LIMIT_WEIGHTS_SHORT": short}


def draft(args: argparse.Namespace) -> int:
    values = {name: f"TODO: {what}" for name, what in TOKENS.items()}
    notes = {"_how": "values for docs/brief/fill_final.py; replace every TODO, check the rest against the logs"}
    keep = Path(args.keep)
    if keep.is_file():                                   # what no log holds is taken from the values file before this one
        earlier = json.loads(keep.read_text(encoding="utf-8"))
        if isinstance(earlier.get("FINAL_TEAM_NAME"), str):
            values["FINAL_TEAM_NAME"] = earlier["FINAL_TEAM_NAME"]
    if args.team:
        values["FINAL_TEAM_NAME"] = args.team
    if args.picked_at:
        values["FINAL_PICKED_AT"] = args.picked_at
    dev_episodes = ""
    if args.dev_tag:
        found_dev = dev_values(args.dev_tag)
        dev_episodes = found_dev.pop("_dev_episodes")
        values.update(found_dev)
        notes["_dev_tag"] = args.dev_tag
        values["FINAL_AGENT"] = re.sub(r"_dev$", "", args.dev_tag)
    found = None
    if args.report:
        if not args.report_name:
            raise SystemExit("--report needs --report-name, the agent's name inside the report")
        found = read_report(Path(args.report), args.report_name, args.holder, args.ref_seeds)
        for label, token in REPORT_ROWS.items():
            values[token] = pm(*found["head"][args.report_name][label])
        if found["ref"]:
            for half, token in REF_ROWS.items():
                values[token] = pm(*found["ref"][args.report_name][half])
        values["FINAL_SEEDS"] = seeds_words(found)
        values["FINAL_SEEDS_N"] = str(found["generated_seeds"])
        values["FINAL_EPISODES"] = f"{found['episodes']:,}"
        notes["_report"] = (f"{args.report} ({args.report_name}" + (f" against {args.holder}" if args.holder else "")
                            + f"; organizer-style suite on {found['ref_seeds']} seed(s))")
        if args.holder:
            notes["_rule"] = found["decision"] or "no DECISION line for this pair in the report"
        read_pairs = [(token, found["head"][args.report_name][label]) for label, token in REPORT_ROWS.items()]
        read_pairs += [(token, found["ref"][args.report_name][half]) for half, token in REF_ROWS.items()] if found["ref"] else []
        ties = [f"{token} ({value:.2f} +- {se:.2f})" for token, (value, se) in read_pairs if is_tie(value) or is_tie(se)]
        if ties:                                         # one decimal of a two-decimal number that ends in 5: see `rounded`
            notes["_rounding"] = ("the report's second decimal is a 5 in these and was rounded up; correct the value by hand "
                                  "if the unrounded number is below the tie: " + ", ".join(ties))
            print("note: a 5 in the report's second decimal was rounded up, check against the unrounded scores: " + ", ".join(ties))
        values["FINAL_AGENT"] = re.sub(r"^shipped_", "", args.report_name)
    if args.agent:
        values["FINAL_AGENT"] = args.agent
    policy, policy_steps, made = Path(args.policy), None, None
    if policy.exists():
        read = policy_values(policy)
        policy_steps = int(read.pop("_env_steps")) if "_env_steps" in read else None
        notes.update({k: v for k, v in read.items() if k.startswith("_")})
        values.update({k: v for k, v in read.items() if not k.startswith("_")})
    made = lineage(args, policy_steps)
    kind = "single"
    if made:
        sentences = weights_sentences(made)
        kind = sentences.pop("_kind")
        values.update(sentences)
        notes["_lineage"] = (f"main_steps {made['main']}, branch_steps {made['branch']}, blend_weights (main, branch) "
                             f"{list(made['weights'])}, step counter in the policy file {made['last']}"
                             + (f"; average of the two checkpoint files compared with the shipped state dict: "
                                f"{made['verified']} of {made['verified']} tensors identical" if made["verified"] else
                                "; the average was not compared with the shipped file (no checkpoint files given)"
                                if made["main"] is not None and made["branch"] is not None else ""))
    if found and args.holder:
        if not args.picked_at:
            raise SystemExit("--holder needs --picked-at, the clock time of the choice")
        values.update(choice_sentences(args, args.report_name, found, kind))
    values.update(limit_sentences(args, made, found, dev_episodes))
    out = Path(args.out)
    out.write_text(json.dumps({**notes, **values}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    todo = [name for name, value in values.items() if "TODO" in value]
    print(f"{out}: {len(values) - len(todo)} value(s) read from files, {len(todo)} to type in"
          + (f": {', '.join(todo)}" if todo else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("values", nargs="?", help="JSON file of token values; fills the documents")
    ap.add_argument("--from", dest="source_dir", metavar="DIR",
                    help="folder with the token sources README.md and brief.md; the documents are written from them")
    ap.add_argument("--build", action="store_true", help="after filling, rebuild brief.pdf with build_brief.py --strict")
    ap.add_argument("--verify", action="store_true",
                    help="write nothing: exit status 0 only if the documents equal the sources of --from filled with the values")
    ap.add_argument("--tokens", action="store_true", help="list every token with its meaning and where it is used")
    ap.add_argument("--check", action="store_true", help="exit status 1 if a token is left in a document")
    ap.add_argument("--manager-probe", nargs="?", const=str(POLICY), metavar="FILE",
                    help="print the manager's two entropies for a policy file (default: the shipped one)")
    ap.add_argument("--draft", action="store_true", help="write a values file prefilled from the logs")
    ap.add_argument("--dev-tag", help="--draft: tag of the final agent's dev_v1 rows in training/logs/results.csv")
    ap.add_argument("--report", help="--draft: final-pick report that includes the final agent")
    ap.add_argument("--report-name", help="--draft: the final agent's name in that report, e.g. blend_final_M")
    ap.add_argument("--holder", help="--draft: the agent it challenged, by its name in that report")
    ap.add_argument("--holder-is", help="--draft: what the holder is, in words (known holders have a default)")
    ap.add_argument("--holder-short", help="--draft: the holder in a few words, for the brief (known holders have a default)")
    ap.add_argument("--agent", help="--draft: the final agent's name in the documents (default: taken from --report-name)")
    ap.add_argument("--picked-at", help="--draft: clock time of the choice, e.g. '6:42 AM'")
    ap.add_argument("--team", help="--draft: team name (default: the one in the values file named by --keep)")
    ap.add_argument("--main-steps", type=int, help="--draft: second-phase environment steps of the main run's checkpoint")
    ap.add_argument("--branch-steps", type=int, help="--draft: step counter of the branch's checkpoint, when one is in the weights")
    ap.add_argument("--main-ckpt", help="--draft: read --main-steps from this checkpoint")
    ap.add_argument("--branch-ckpt", help="--draft: read --branch-steps from this checkpoint")
    ap.add_argument("--blend-weights", help="--draft: MAIN,BRANCH weights of an average (default 0.5,0.5)")
    ap.add_argument("--lineage", help="--draft: JSON file with main_steps, branch_steps, blend_weights (arguments win)")
    ap.add_argument("--decision", help="--draft: the sentence that says what was decided and why")
    ap.add_argument("--decision-short", help="--draft: the same for the brief (default: --decision)")
    ap.add_argument("--choice-extra", help="--draft: sentences added to the README's account of the choice")
    ap.add_argument("--gate-passed", action="store_true", help="--draft: the final agent passed the nine-check ship gate")
    ap.add_argument("--ref-seeds", type=int, help="--draft: engine seeds of the organizer-style table in the report "
                                                 "(default: as many as the generated suites have)")
    ap.add_argument("--policy", default=str(POLICY), help="--draft: the shipped policy file (default submission/weights/policy.pt)")
    ap.add_argument("--keep", default=str(VALUES), help="--draft: earlier values file the team name is taken from")
    ap.add_argument("--out", default=str(VALUES), help="--draft: where the values file is written")
    args = ap.parse_args()
    source_dir = Path(args.source_dir) if args.source_dir else None
    if args.tokens:
        return show_tokens(source_dir)
    if args.check:
        return check()
    if args.manager_probe:
        subgoal, target = manager_probe(Path(args.manager_probe))
        sha = hashlib.sha256(Path(args.manager_probe).read_bytes()).hexdigest()[:12]
        print(f"{args.manager_probe} (sha256 {sha}): sub-goal entropy {subgoal:.3f} nats (uniform over 8: {math.log(8):.3f}), "
              f"target entropy {target:.3f} nats (uniform over 4,096: {math.log(4096):.3f}; exp = {math.exp(target):.0f} cells); "
              f"2 recorded episodes, 10 manager steps")
        return 0
    if args.draft:
        return draft(args)
    if not args.values:
        ap.error("give a values file, or one of --tokens, --check, --draft, --manager-probe")
    if args.verify:
        return verify(Path(args.values), source_dir)
    return fill(Path(args.values), args.build, source_dir)


if __name__ == "__main__":
    sys.exit(main())
