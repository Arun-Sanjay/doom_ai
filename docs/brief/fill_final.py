"""Fill the numbers of the final agent into the judged documents.

`docs/brief/brief.md` and `README.md` are final except for tokens of the form {{FINAL_NAME}}, which
stand for facts that exist only once the final agent has been chosen. This script replaces them
from a JSON file and refuses to write anything unless every token gets a value.

    python docs/brief/fill_final.py --tokens                   # every token: where it is used, what it must contain
    python docs/brief/fill_final.py --check                    # exit status 1 if a token is still in a document
    python docs/brief/fill_final.py --draft --dev-tag TAG --report FILE --report-name NAME
                                                               # writes docs/brief/final_values.json from the logs
    python docs/brief/fill_final.py docs/brief/final_values.json --build
                                                               # fills both documents, then rebuilds brief.pdf

The values file is one JSON object, token name (without braces) -> text. Keys that start with "_"
are notes and are ignored.

`--draft` reads what can be read from files, so that numbers are copied by code and not by hand:
the per-tier rows of `--dev-tag` in training/logs/results.csv (it must be a run on dev_v1), the
HEADLINE block of a final-pick report for the agent called `--report-name` in it, and the shipped
policy file (its sha256, and from its stored config the step count and the decoding setting; this
part needs torch). Whatever it cannot read is written as "TODO: ..." and has to be typed in.
Read the draft before filling: the step counts assume a checkpoint of the second-phase run.

Filling fails, and changes no file, when a token used in a document has no value, a value is empty,
still says TODO, spans several lines or holds a "|" (it would break a table), the file has a key
that is not a token, or anything shaped like a token would be left behind. With `--build` the brief
is rebuilt by build_brief.py --strict, which fails on more than two pages or on a token left over;
the exit status is then that of the build.

Standard library only (torch is imported by `--draft` alone, and only if it is installed).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCUMENTS = (HERE / "brief.md", ROOT / "README.md")
VALUES = HERE / "final_values.json"
RESULTS = ROOT / "training" / "logs" / "results.csv"
POLICY = ROOT / "submission" / "weights" / "policy.pt"

TOKEN = re.compile(r"\{\{(FINAL_[A-Z0-9_]+)\}\}")
LEFTOVER = re.compile(r"\{\{\s*FINAL[^}]*\}*")        # anything that still looks like a token, well-formed or not

# Steps of the runs before the second phase (step counters stored in their last checkpoints) and the
# second-phase step at which the teacher's weight reached zero (training/logs/unified_full_2_train_log.csv).
COMBAT_STEPS = 118_784_000
PHASE_ONE_STEPS = 48_955_392
TEACHER_OFF_STEP = 44_236_800

# name -> what it must contain (with an example of the form)
TOKENS = {
    "FINAL_TEAM_NAME": "team name as registered with the organizers (brief title line)",
    "FINAL_AGENT": "name of the final agent as it is tagged in the logs, e.g. main_0305",
    "FINAL_PICKED_AT": "clock time of the final choice, e.g. 6:40 AM",
    "FINAL_POLICY_SHA": "first 12 hex digits of the sha256 of submission/weights/policy.pt (the agent prints them at start)",
    "FINAL_PHASE2_STEPS": "second-phase environment steps of the shipped checkpoint in millions, one decimal, e.g. 98.3",
    "FINAL_TOTAL_STEPS": "all environment steps behind the shipped policy in millions (combat 118.8 + phase one 49.0 + phase two), e.g. 266",
    "FINAL_RL_ONLY_STEPS": "second-phase steps after the teacher's weight reached zero at 44.2 M, in millions, e.g. 54.1",
    "FINAL_DECODING": "decoding setting of the shipped policy file as a clause, e.g. USE and weapon take their most likely "
                      "option, the other heads and the manager's choices are sampled",
    "FINAL_DEV_T1": "dev_v1 composite of tier 1, one decimal (training/logs/results.csv)",
    "FINAL_DEV_T2": "dev_v1 composite of tier 2",
    "FINAL_DEV_T3": "dev_v1 composite of tier 3",
    "FINAL_DEV_T4": "dev_v1 composite of tier 4",
    "FINAL_DEV_ALL": "dev_v1 tier-weighted composite with its standard error, e.g. 77.8 ± 3.8",
    "FINAL_DEV_EXIT": "dev_v1 exit rate on maps with an exit, two decimals, e.g. 0.82",
    "FINAL_DEV_KILLS": "dev_v1 kills over monsters, two decimals, e.g. 0.78",
    "FINAL_DEV_DEATHS": "dev_v1 death rate, two decimals, e.g. 0.14",
    "FINAL_HEADLINE": "weighted headline of the final-pick report with its standard error, e.g. 75.5 ± 1.1",
    "FINAL_HELDOUT_V1": "weighted score on heldout_v1 in the same report, e.g. 85.6 ± 1.2",
    "FINAL_HELDOUT_WIDE": "weighted score on heldout_wide_v1",
    "FINAL_AUTHORED": "weighted score on authored_v1",
    "FINAL_AUTHORED_HARD": "weighted score on authored_hard_v1",
}
REPORT_ROWS = {"headline": "FINAL_HEADLINE", "heldout_v1": "FINAL_HELDOUT_V1", "heldout_wide_v1": "FINAL_HELDOUT_WIDE",
               "authored_v1": "FINAL_AUTHORED", "authored_hard_v1": "FINAL_AUTHORED_HARD"}


def uses() -> dict[str, list[str]]:
    """token name -> the places it appears, as 'path:line'."""
    found: dict[str, list[str]] = {}
    for path in DOCUMENTS:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for name in TOKEN.findall(line):
                found.setdefault(name, []).append(f"{path.relative_to(ROOT).as_posix()}:{number}")
    return found


def leftovers() -> list[str]:
    hits = []
    for path in DOCUMENTS:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            hits += [f"{path.relative_to(ROOT).as_posix()}:{number}: {m.group(0)}" for m in LEFTOVER.finditer(line)]
    return hits


def show_tokens() -> int:
    used = uses()
    for name, what in TOKENS.items():
        print(f"{{{{{name}}}}}\n    {what}\n    used in: {', '.join(used.get(name, [])) or 'no document'}")
    unknown = sorted(set(used) - set(TOKENS))
    for name in unknown:
        print(f"{{{{{name}}}}}\n    NOT IN THE LIST OF THIS SCRIPT; used in: {', '.join(used[name])}")
    print(f"{len(used)} token(s) in the documents, {len(TOKENS)} known")
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
def fill(values_file: Path, build: bool) -> int:
    try:
        raw = json.loads(values_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"FAILED, nothing was written: cannot read {values_file}: {error}", file=sys.stderr)
        return 1
    if not isinstance(raw, dict):
        print(f"FAILED, nothing was written: {values_file} must hold one JSON object", file=sys.stderr)
        return 1
    values = {k: v for k, v in raw.items() if not k.startswith("_")}
    used = uses()
    problems = [f"no value for {{{{{name}}}}} (used in {', '.join(where)})" for name, where in sorted(used.items())
                if name not in values]
    problems += [f"'{name}' in {values_file.name} is not a token of this script" for name in sorted(set(values) - set(TOKENS))]
    for name, value in sorted(values.items()):
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name}: the value must be a non-empty string")
        elif "TODO" in value or "{{" in value or "|" in value or "\n" in value:
            problems.append(f"{name}: the value still says TODO, or holds '{{{{', '|' or a line break: {value[:60]!r}")
    if not used:
        problems.append("no token is left in the documents: they are already filled (restore them from git to fill again)")
    if problems:
        print("FAILED, nothing was written:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    filled, counts = {}, {}
    for path in DOCUMENTS:
        text, counts[path] = TOKEN.subn(lambda m: values[m.group(1)].strip(), path.read_text(encoding="utf-8"))
        rest = LEFTOVER.findall(text)
        if rest:
            print(f"FAILED, nothing was written: {path.name} would still hold {rest[:5]}", file=sys.stderr)
            return 1
        filled[path] = text
    for path, text in filled.items():
        path.write_text(text, encoding="utf-8")
        print(f"{path.relative_to(ROOT).as_posix()}: {counts[path]} token(s) replaced", flush=True)
    if build:
        return subprocess.run([sys.executable, str(HERE / "build_brief.py"), "--strict"], check=False).returncode
    print("now rebuild the PDF: python docs/brief/build_brief.py --strict")
    return 0


# ---------------------------------------------------------------------------------------------
# Draft of the values file from the logs
# ---------------------------------------------------------------------------------------------
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
    out["FINAL_DEV_ALL"] = f"{float(every['score']):.1f} ± {float(every['score_se']):.1f}"
    out["FINAL_DEV_EXIT"] = f"{float(every['exit_rate']):.2f}"
    out["FINAL_DEV_KILLS"] = f"{float(every['kills_frac']):.2f}"
    out["FINAL_DEV_DEATHS"] = f"{float(every['death_rate']):.2f}"
    return out


def report_values(report: Path, name: str) -> dict[str, str]:
    """The weighted column of the HEADLINE block of a final-pick report, for the agent called `name`."""
    out, label, inside = {}, "", False
    for line in report.read_text(encoding="utf-8").split("\n"):
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
        if name not in parts:
            continue
        at = parts.index(name)
        if label in REPORT_ROWS:
            out[REPORT_ROWS[label]] = f"{float(parts[at + 1]):.1f} ± {float(parts[at + 2].lstrip('+-')):.1f}"
    missing = sorted(set(REPORT_ROWS.values()) - set(out))
    if missing:
        raise SystemExit(f"{report}: no HEADLINE rows for '{name}' ({', '.join(missing)} not found)")
    return out


def policy_values(policy: Path) -> dict[str, str]:
    out = {"FINAL_POLICY_SHA": hashlib.sha256(policy.read_bytes()).hexdigest()[:12]}
    try:
        import torch
        config = torch.load(policy, map_location="cpu", weights_only=False)["config"]
        steps = int(config["env_steps"])
    except Exception as error:  # noqa: BLE001 - the draft then says what has to be typed in
        print(f"note: step count and decoding not read from {policy.name} ({type(error).__name__}: {error})")
        return out
    out["FINAL_PHASE2_STEPS"] = f"{steps / 1e6:.1f}"
    out["FINAL_TOTAL_STEPS"] = f"{(COMBAT_STEPS + PHASE_ONE_STEPS + steps) / 1e6:.0f}"
    out["FINAL_RL_ONLY_STEPS"] = f"{(steps - TEACHER_OFF_STEP) / 1e6:.1f}"
    order = ("move", "strafe", "turn", "attack", "use", "weapon")
    greedy = config.get("greedy_heads")
    greedy = ["use", "weapon"] if greedy is None else list(greedy)       # the agent's default (agent.GREEDY_HEADS)
    shown = {"use": "USE"}
    likely = [shown.get(h, h) for h in order if h in greedy]
    drawn = [shown.get(h, h) for h in order if h not in greedy]

    def listed(names: list[str]) -> str:
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    if likely and drawn:
        out["FINAL_DECODING"] = (f"{listed(likely)} take their most likely option, {listed(drawn)} "
                                 f"{'is' if len(drawn) == 1 else 'are'} sampled, and so are the manager's choices")
    elif likely:
        out["FINAL_DECODING"] = "all six heads take their most likely option, the manager's choices are sampled"
    else:
        out["FINAL_DECODING"] = "all six heads and the manager's choices are sampled"
    out["_policy_file"] = (f"{policy.name}: source {config.get('source')}, iteration {config.get('iteration')}, "
                           f"env_steps {steps}, greedy_heads {config.get('greedy_heads')}")
    return out


def draft(args: argparse.Namespace) -> int:
    values = {name: f"TODO: {what}" for name, what in TOKENS.items()}
    notes = {"_how": "values for docs/brief/fill_final.py; replace every TODO, check the rest against the logs"}
    if args.dev_tag:
        values.update(dev_values(args.dev_tag))
        notes["_dev_tag"] = args.dev_tag
        values["FINAL_AGENT"] = re.sub(r"_dev$", "", args.dev_tag)
    if args.report:
        if not args.report_name:
            raise SystemExit("--report needs --report-name, the agent's name inside the report")
        values.update(report_values(Path(args.report), args.report_name))
        notes["_report"] = f"{args.report} ({args.report_name})"
        values["FINAL_AGENT"] = re.sub(r"^shipped_", "", args.report_name)
    policy = Path(args.policy)
    if policy.exists():
        found = policy_values(policy)
        notes.update({k: v for k, v in found.items() if k.startswith("_")})
        values.update({k: v for k, v in found.items() if not k.startswith("_")})
    out = Path(args.out)
    out.write_text(json.dumps({**notes, **values}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    todo = [name for name, value in values.items() if value.startswith("TODO")]
    print(f"{out}: {len(values) - len(todo)} value(s) read from files, {len(todo)} to type in"
          + (f": {', '.join(todo)}" if todo else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("values", nargs="?", help="JSON file of token values; fills the documents")
    ap.add_argument("--build", action="store_true", help="after filling, rebuild brief.pdf with build_brief.py --strict")
    ap.add_argument("--tokens", action="store_true", help="list every token with its meaning and where it is used")
    ap.add_argument("--check", action="store_true", help="exit status 1 if a token is left in a document")
    ap.add_argument("--draft", action="store_true", help="write a values file prefilled from the logs")
    ap.add_argument("--dev-tag", help="--draft: tag of the final agent's dev_v1 rows in training/logs/results.csv")
    ap.add_argument("--report", help="--draft: final-pick report that includes the final agent")
    ap.add_argument("--report-name", help="--draft: the final agent's name in that report, e.g. shipped_main_0305")
    ap.add_argument("--policy", default=str(POLICY), help="--draft: the shipped policy file (default submission/weights/policy.pt)")
    ap.add_argument("--out", default=str(VALUES), help="--draft: where the values file is written")
    args = ap.parse_args()
    if args.tokens:
        return show_tokens()
    if args.check:
        return check()
    if args.draft:
        return draft(args)
    if not args.values:
        ap.error("give a values file, or one of --tokens, --check, --draft")
    return fill(Path(args.values), args.build)


if __name__ == "__main__":
    sys.exit(main())
