"""Compliance lint for the evaluated code, plus a secret scan of the repository.

docs/RULES.md allows observation encoding, action masks and learned modules in the final agent. It
forbids classical path planning, trigonometric aimbots, stock RL-library policies and any attempt to
reach the engine, the host or the network. Everything the scorer runs lives under submission/, so
that folder is checked statically:

1. Names (test_submission_text_has_no_banned_names). Every text file under submission/ is read line
   by line, comments, docstrings and Markdown included. No line may contain the name of a banned
   technique (BANNED_NAMES): the inverse trigonometric functions, queue containers and the classic
   graph searches under their usual names. The first eight of the list (atan2, arctan2, heapq,
   deque, astar, dijkstra, bfs, flood) count wherever they appear, also inside a longer word, so a
   harmless word that happens to contain one has to be renamed.
2. Imports and devices (test_submission_code_has_no_banned_constructs). Every Python file under
   submission/ is parsed. Imports from outside the standard library are limited to numpy and torch.
   The engine, the wrapper, network and process modules, stock RL libraries and this repository's
   training/ package are banned (BANNED_MODULES), and so is naming a GPU: `.cuda`, a device ordinal,
   or a string such as "cuda" used for anything but a comparison. Because the code is parsed, a
   comment that merely mentions one of these does not count. The angle of a complex number
   (`.angle()`, `.phase()`) is rejected here too: it is the inverse tangent without its name.
3. Search structure (same test). Names can be avoided, so the parsed code is also checked for the
   shapes a hand-written search over a map takes: an open-ended loop (take a cell, add its
   neighbours, go on until none is left), written as `while` or as a `for` loop left by `break`;
   a list used as a queue (`.pop(0)`, `.popleft()`); and a function that calls itself. Code that
   encodes observations and runs networks needs none of them, so none is allowed: every loop
   under submission/ runs a number of passes that is fixed before it starts.
   Limit of this check: a `for` loop with a fixed count that updates a whole array on every pass is
   not flagged. The learned value-propagation block in doomnet/model.py has exactly that shape (one
   convolution cell applied K times), and no static check can tell trained weights from a kernel
   set by hand. Whether such a loop is learned is settled by reading it; README.md says which
   parts are. A loop left by `return` and two functions that call each other are not followed
   either.
4. README (test_readme_labels_every_module, test_readme_agrees_with_agent_py). Every module has a
   row in submission/README.md labelled "observation encoding", "action mask" or "learned", and
   what the README says about agent.py is compared with the file. While agent.py is still the
   template's random baseline the README must say so (a status note and the label "placeholder")
   and must not credit it with parts it does not have; once agent.py imports the networks the note
   and that label must go, every claim listed in README_CLAIMS needs its counterpart in the code,
   and every constant and weights file the README cites by name must exist there.
5. Secrets (test_no_key_like_token_in_repo_files and ..._in_git_history). Every file git would
   commit, and every commit already made, is searched for words shaped like an API key.

Each hit is reported as path:line. The second half of the file proves that the checks bite: banned
lines are planted in temporary folders (never in submission/) and must be found at the right line,
and look-alikes that are allowed must pass.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission"


class Hit(NamedTuple):
    path: str  # relative to the repository root
    line: int  # 1-based
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.what}"


def fail_if(hits: list[Hit], title: str) -> None:
    """Fail the test with one path:line per hit."""
    if hits:
        report = "\n".join(str(h) for h in sorted(set(hits)))
        print(report)
        pytest.fail(f"{title}: {len(set(hits))} hit(s)\n{report}", pytrace=False)


# ---------------------------------------------------------------------------------------------
# Which files are read
# ---------------------------------------------------------------------------------------------
def submission_files(folder: Path) -> list[Path]:
    """Every file under `folder` except Python's bytecode cache."""
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"))


def read_text(path: Path) -> str | None:
    """The file's text, or None for a binary file (a NUL byte in the first 8 kB, the test git uses).

    Weights are binary and are skipped: random bytes would produce chance matches.
    """
    with open(path, "rb") as f:
        head = f.read(8192)
        if b"\0" in head:
            return None
        return (head + f.read()).decode("utf-8", errors="replace")


def binary_files(folder: Path) -> list[Path]:
    """The files under `folder` that the text scan skips."""
    return [p for p in submission_files(folder) if read_text(p) is None]


# ---------------------------------------------------------------------------------------------
# 1. Banned names in text
# ---------------------------------------------------------------------------------------------
ANYWHERE, WORD, AS_WRITTEN = "anywhere", "word", "as written"

# (name reported, its spelling as a regular expression, how it is matched, why it is banned)
# ANYWHERE:   the spelling counts wherever it appears, also inside a longer word, in any case.
# WORD:       the spelling must start a word and the word must end with it, in any case. `dfs2`,
#             `run_dfs` and `DfsOrder` count; `pdfs`, the hyperbolic `atanh` and `asinh`, the word
#             `increasing` and the constant `A_START` do not.
# AS_WRITTEN: the expression is used as it stands.
BANNED_NAMES = (
    # --- the eight tokens of the build spec: a file that contains one of them fails
    ("atan2", "atan2", ANYWHERE, "two-argument inverse tangent, the rule book's aimbot example"),
    ("arctan2", "arctan2", ANYWHERE, "two-argument inverse tangent, the rule book's aimbot example"),
    ("heapq", "heapq", ANYWHERE, "priority queue, the core of shortest-path search"),
    ("deque", "deque", ANYWHERE, "double-ended queue, the core of breadth-first search"),
    ("astar", "astar", ANYWHERE, "A-star search"),
    ("dijkstra", "d(ij|ji)(ks|sk)tra", ANYWHERE, "shortest-path search; the usual misspellings count too"),
    ("bfs", "bfs", ANYWHERE, "breadth-first search"),
    ("flood", "flood", ANYWHERE, "region filling over a map"),
    # --- the same techniques under other names (stricter than the spec asks)
    ("a-star", "a[-_]star", WORD, "A-star search"),
    ("A*", r"(?<![\w.)\]*])A\*(?=\s|$|[,;:]|\.(?:\s|$))", AS_WRITTEN, "A-star search; a product such as A * B or A*B is not matched"),
    ("dfs", "dfs", WORD, "depth-first search"),
    ("atan", "atan", WORD, "inverse tangent"),
    ("arctan", "arctan", WORD, "inverse tangent"),
    ("asin", "asin", WORD, "inverse sine: with the sine of a bearing among the entity features it gives the bearing, as the inverse tangent does"),
    ("arcsin", "arcsin", WORD, "inverse sine, see 'asin'"),
    ("acos", "acos", WORD, "inverse cosine, see 'asin'"),
    ("arccos", "arccos", WORD, "inverse cosine, see 'asin'"),
    ("cmath", "cmath", WORD, "complex-number module; the phase of x + iy is the two-argument inverse tangent"),
    ("queue", "queue", ANYWHERE, "queue container, the worklist of a graph search"),
    ("frontier", "frontier", ANYWHERE, "frontier of a graph search or of hand-coded exploration"),
    ("wavefront", "wave[-_ ]?front", ANYWHERE, "wavefront expansion over a map"),
    ("pathfind", "path[-_ ]?find", ANYWHERE, "path finding"),
    ("path planner", "path_?plan", ANYWHERE, "path planner as an identifier; the two words in a sentence are not matched"),
    ("shortest path", "shortest[-_ ]?path", ANYWHERE, "shortest-path search"),
    ("breadth-first", "breadth[-_ ]?first", ANYWHERE, "breadth-first search"),
    ("depth-first", "depth[-_]?first", ANYWHERE, "depth-first search; 'depth first' with a space is left to sentences about the depth image"),
    ("open_set", "open_?set", WORD, "open set of a best-first search, as an identifier"),
    ("came_from", "came_?from", WORD, "parent links of a graph search, as an identifier"),
)

WORD_START = r"(?<![A-Za-z])"             # no letter before the name
WORD_END = r"(?![a-z]|[A-Z](?![a-z]))"    # no lowercase letter after it, and no capital unless that capital starts a new CamelCase word
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")  # "MapSearch" is also read as the two words "Map Search"


def _pattern(spelling: str, how: str) -> re.Pattern:
    """The regular expression for one banned name (see ANYWHERE, WORD and AS_WRITTEN above)."""
    if how == AS_WRITTEN:
        return re.compile(spelling)
    body = f"(?i:{spelling})"
    return re.compile(body if how == ANYWHERE else WORD_START + body + WORD_END)


NAME_RULES = [(name, why, _pattern(spelling, how)) for name, spelling, how, why in BANNED_NAMES]


def scan_names(folder: Path) -> list[Hit]:
    """Banned technique names in the text of every file under `folder`."""
    hits = []
    for path in submission_files(folder):
        text = read_text(path)
        if text is None:
            continue
        rel = path.relative_to(folder.parent).as_posix()
        for number, line in enumerate(text.split("\n"), 1):
            forms = (line, _CAMEL.sub(" ", line))
            for name, why, pattern in NAME_RULES:
                if any(pattern.search(form) for form in forms):
                    hits.append(Hit(rel, number, f"banned name '{name}' ({why})"))
    return hits


# ---------------------------------------------------------------------------------------------
# 2 and 3. Banned imports, process calls, GPU use and search structure in Python code
# ---------------------------------------------------------------------------------------------
ALLOWED_PACKAGES = {"numpy", "torch"}  # the only imports allowed from outside the standard library

BANNED_MODULES = {
    # --- the list in the build spec
    "vizdoom": "the game engine",
    "doom_env": "the environment wrapper; the agent only ever receives observations",
    "socket": "network access",
    "subprocess": "starts other processes",
    "requests": "network access",
    "urllib": "network access",
    "stable_baselines3": "stock RL library",
    "training": "training-time code (teacher, map generator, reward) is never imported by the agent",
    # --- the same capabilities elsewhere in the standard library (stricter than the spec asks)
    "queue": "queue containers",
    "_socket": "network access",
    "http": "network access",
    "ftplib": "network access",
    "smtplib": "network access",
    "imaplib": "network access",
    "poplib": "network access",
    "socketserver": "network access",
    "ssl": "network access",
    "xmlrpc": "network access",
    "asyncio": "network access",
    "multiprocessing": "starts other processes",
    "concurrent": "starts other processes or threads",
    "_posixsubprocess": "starts other processes",
    "pty": "starts other processes",
    "runpy": "runs a module chosen at run time, which cannot be checked",
    "ctypes": "native code outside numpy and torch",
}

# "cuda", "cuda:0", "mps", and the templates a device name is built from: "cuda:%d", "cuda:{}", the "cuda:" of an f-string
GPU_DEVICE = re.compile(r"(cuda|mps)(:(\d+|%[di]|\{\w*\})?)?", re.IGNORECASE)

WHILE_LOOP = ("has a `while` loop. An open-ended loop is how a search over a map is written by hand (take a cell, add its "
              "neighbours, go on until none is left). Use array operations or a `for` loop with a fixed count")
BREAK = ("leaves a loop with `break`. A loop that runs until a condition is met is a `while` loop in other words "
         "(see the rule on `while`). Give the loop a fixed count, or use any() or next() to pick an item")
FRONT_OF_LIST = "takes items from the front of a list ({}), which makes the list a queue: the worklist of a breadth-first search"
CALLS_ITSELF = "function '{}' calls itself. Recursion is how a region fill or a depth-first search is written by hand"
COMPLEX_ANGLE = "calls .{}(), the angle of a complex number, which is the two-argument inverse tangent under another name"
TORCH_INTERFACES = {"multiprocessing": "starts other processes", "hub": "downloads files", "distributed": "network access"}


def _starts_process(name: str) -> bool:
    """Functions of the os module that run another program."""
    return name in ("system", "popen", "fork", "forkpty", "startfile") or name.startswith(("exec", "spawn", "posix_spawn"))


def _import_problem(module: str, names: tuple[str, ...], local: set[str]) -> str | None:
    """Why `import module` or `from module import names` is not allowed under submission/, if it is not."""
    parts = module.split(".")
    top = parts[0]
    if top in BANNED_MODULES:
        return f"imports '{module}' ({BANNED_MODULES[top]})"
    if top == "torch" and ("cuda" in parts or "cuda" in names):
        return "imports torch's GPU interface (the agent is CPU only)"
    if top == "torch":
        for part in (*parts, *names):
            if part in TORCH_INTERFACES:
                return f"imports torch's '{part}' interface ({TORCH_INTERFACES[part]})"
    if top == "os" and any(_starts_process(n) for n in names):
        return "imports a function of the os module that starts another process"
    if top in ALLOWED_PACKAGES or top in local or top in sys.stdlib_module_names:
        return None
    return f"imports '{module}' (only numpy and torch may be imported from outside the standard library)"


def _compared(node: ast.AST):
    """The constants that `node` only compares with something: operands of ==, != and `in`, the
    arguments of .startswith() and .endswith(), and the values of `case` patterns. Reading the
    `device` argument this way is allowed."""
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith", "endswith"):
        operands = node.args
    elif isinstance(node, ast.MatchValue):
        operands = [node.value]
    else:
        return
    for operand in operands:
        for item in (operand.elts if isinstance(operand, (ast.Tuple, ast.List, ast.Set)) else [operand]):
            if isinstance(item, ast.Constant):
                yield item


def _problems(node: ast.AST, local: set[str], compared: set[int], os_names: set[str]):
    """Rule violations of one syntax-tree node, as messages. `os_names`: the names the file gives the os module."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield _import_problem(alias.name, (), local)
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:  # relative imports stay inside submission/
            yield _import_problem(node.module, tuple(a.name for a in node.names), local)
    elif isinstance(node, ast.While):
        yield WHILE_LOOP
    elif isinstance(node, ast.Break):
        yield BREAK
    elif isinstance(node, ast.Attribute):
        if node.attr == "cuda":
            yield "uses '.cuda' (the agent is CPU only)"
        elif node.attr == "accelerator":
            yield "uses '.accelerator', which selects a GPU when there is one (the agent is CPU only)"
        elif node.attr in TORCH_INTERFACES:
            yield f"uses '.{node.attr}' ({TORCH_INTERFACES[node.attr]})"
        elif isinstance(node.value, ast.Name) and node.value.id in os_names and _starts_process(node.attr):
            yield f"uses os.{node.attr}, which starts another process"
    elif isinstance(node, ast.Constant):
        # x.to("cuda"), torch.device("cuda"), f(device="cuda"), device = "cuda", "cuda" if flag else "cpu", f"cuda:{i}"
        if isinstance(node.value, str) and GPU_DEVICE.fullmatch(node.value) and id(node) not in compared:
            yield f"names the GPU device '{node.value}' (the agent is CPU only; such a string may only be compared with)"
    elif isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if isinstance(func, ast.Name) and name in ("eval", "exec"):  # the builtins, not a method such as net.eval()
            yield f"calls {name}(), which runs code built at run time that cannot be checked"
        if name in ("__import__", "import_module"):
            target = node.args[0] if node.args else None
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                yield _import_problem(target.value, (), local)
            else:
                yield "imports a module whose name is computed at run time, which cannot be checked"
        if isinstance(func, ast.Attribute):
            first = node.args[0] if node.args else None
            if name == "pop" and isinstance(first, ast.Constant) and first.value == 0 and not isinstance(first.value, bool):
                yield FRONT_OF_LIST.format(".pop(0)")
            if name in ("popleft", "appendleft"):
                yield FRONT_OF_LIST.format(f".{name}()")
            if name in ("angle", "phase"):
                yield COMPLEX_ANGLE.format(name)
        if name in ("to", "device"):  # x.to(0) and torch.device(0) mean the first GPU
            for arg in node.args:
                if isinstance(arg, ast.Constant) and type(arg.value) is int:
                    yield f"selects a device by its number ({name}({arg.value})), which means a GPU (the agent is CPU only)"


def _recursive_calls(tree: ast.AST):
    """(line, message) for every call a function makes to itself: f() inside f, self.f() or cls.f() inside method f.

    A method that calls a module-level function of the same name, super().f() or another object's
    f() is not recursion and is not reported.
    """
    for parent in ast.walk(tree):
        for func in ast.iter_child_nodes(parent):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owner = parent.name if isinstance(parent, ast.ClassDef) else None
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                by_name = owner is None and isinstance(f, ast.Name) and f.id == func.name
                on_self = (owner is not None and isinstance(f, ast.Attribute) and f.attr == func.name
                           and isinstance(f.value, ast.Name) and f.value.id in ("self", "cls", owner))
                if by_name or on_self:
                    yield node.lineno, CALLS_ITSELF.format(func.name)


def scan_python(folder: Path) -> list[Hit]:
    """Banned imports, process calls, GPU use and search structure in every Python file under `folder`."""
    local = {p.stem for p in folder.glob("*.py")} | {p.name for p in folder.iterdir() if p.is_dir()} | {folder.name}
    hits = []
    for path in submission_files(folder):
        if path.suffix != ".py":
            continue
        rel = path.relative_to(folder.parent).as_posix()
        try:
            tree = ast.parse(path.read_bytes(), filename=rel)
        except (SyntaxError, ValueError) as e:
            hits.append(Hit(rel, getattr(e, "lineno", None) or 1, f"cannot be parsed, so it cannot be checked ({e})"))
            continue
        compared = {id(constant) for node in ast.walk(tree) for constant in _compared(node)}
        os_names = {"os"} | {alias.asname for node in ast.walk(tree) if isinstance(node, ast.Import)
                             for alias in node.names if alias.name == "os" and alias.asname}
        for node in ast.walk(tree):
            hits += [Hit(rel, node.lineno, what) for what in _problems(node, local, compared, os_names) if what]
        hits += [Hit(rel, line, what) for line, what in _recursive_calls(tree)]
    return hits


# ---------------------------------------------------------------------------------------------
# 4. What README.md says about agent.py
# ---------------------------------------------------------------------------------------------
LABELS = {"observation encoding", "action mask", "learned"}
PLACEHOLDER = "placeholder"  # the label of agent.py's row, and only of that row, while agent.py is the template's random baseline
README_ROW = re.compile(r"^\| `([^`]+\.py)` \| ([^|]+) \|", re.MULTILINE)  # | `doomnet/model.py` | learned | ...

# README.md carries a block quote that starts with these words for as long as agent.py is the
# template's random baseline. It says that agent.py does not run the modules of doomnet/ yet.
STATUS_NOTE = "Status of `agent.py`"

# (words in README.md, the fact about agent.py that must hold when the words are there, what is missing if it does not)
# Checked once agent.py imports the networks. The first five also describe modules of doomnet/, which
# exist before the agent is assembled; the last three can only be about agent.py (AGENT_ONLY).
README_CLAIMS = (
    ("`PolicyNet`", "networks", "does not import doomnet.model"),
    ("exit detector", "networks", "does not import doomnet.model"),
    ("odometry", "odometry", "does not import doomnet.odometry"),
    ("memory map", "memory", "does not import doomnet.memory"),
    ("encoder", "encoder", "does not import doomnet.encode"),
    ("loads the weights", "weights", "never calls load_state_dict() or torch.load()"),
    ("time governor", "clock", "never reads a clock (time.perf_counter() or the like)"),
    ("exception handler", "handler", "has no try/except in DoomAgent.act"),
)
AGENT_ONLY = ("loads the weights", "time governor", "exception handler")
# Names the README cites in code quotes must exist in the code, so that a renamed constant or
# weights file cannot leave a stale sentence behind.
CITED_CONSTANT = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)(?:=[^`]*)?`")  # `GOVERNOR_NETWORK_MS`, `DOOMNET_ALLOW_UNTRAINED=1`
CITED_WEIGHTS = re.compile(r"`weights/([\w.-]+)`")                             # `weights/policy.pt`

CLOCKS = {"perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "process_time", "process_time_ns", "time", "time_ns"}


def agent_facts(agent_py: Path) -> dict[str, bool]:
    """What can be read off agent.py without running it; the keys are the facts named in README_CLAIMS."""
    tree = ast.parse(agent_py.read_bytes())
    imports, calls = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports |= {node.module} | {f"{node.module}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Call):
            calls.add(node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", ""))
    act = [f for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) and cls.name == "DoomAgent"
           for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "act"]

    def uses(module: str) -> bool:
        return any(name == module or name.startswith(module + ".") for name in imports)

    return {
        "networks": uses("doomnet.model"),
        "odometry": uses("doomnet.odometry"),
        "memory": uses("doomnet.memory"),
        "encoder": uses("doomnet.encode"),
        "weights": bool(calls & {"load_state_dict", "load"}),
        "clock": bool(calls & CLOCKS),
        "handler": any(isinstance(node, ast.Try) and node.handlers for f in act for node in ast.walk(f)),
    }


def readme_problems(folder: Path) -> list[Hit]:
    """Where README.md and agent.py under `folder` disagree about what the agent does.

    Two states are consistent. Baseline: agent.py does not import doomnet.model, and the README has
    the status note, labels the agent.py row "placeholder" and claims none of AGENT_ONLY. Assembled:
    agent.py imports doomnet.model, the note and the placeholder label are gone, every claim of
    README_CLAIMS that the README makes holds for the file, and the constants and weights files
    cited in code quotes exist in the code.
    """
    readme = folder / "README.md"
    rel = readme.relative_to(folder.parent).as_posix()
    text = readme.read_text()
    lines = text.split("\n")

    def line_of(words: str) -> int:
        """1-based line where `words` start (any case, line breaks between them allowed), or 0."""
        found = re.search(r"\s+".join(re.escape(w) for w in words.split()), text, re.IGNORECASE)
        return text.count("\n", 0, found.start()) + 1 if found else 0

    facts = agent_facts(folder / "agent.py")
    note = line_of(STATUS_NOTE)
    row = next((i for i, line in enumerate(lines, 1) if line.startswith("| `agent.py` |")), 0)
    labels = {label.strip() for label in lines[row - 1].split("|")[2].split(",")} if row else set()
    hits = []
    if not facts["networks"]:  # agent.py is still the template's random baseline
        if not note:
            hits.append(Hit(rel, row or 1, "agent.py does not run the networks of doomnet/model.py yet and README.md does not say so: "
                                           f"it needs the block quote that starts with '> **{STATUS_NOTE}.**' until agent.py is replaced"))
        if row and labels != {PLACEHOLDER}:
            hits.append(Hit(rel, row, f"agent.py is still the template's random baseline, but its row is labelled {sorted(labels)}: "
                                      f"label it '{PLACEHOLDER}' and describe the file as it is"))
        hits += [Hit(rel, line_of(words), f"README.md says '{words}', but agent.py is still the template's random baseline")
                 for words in AGENT_ONLY if line_of(words)]
        return hits
    if note:
        last = note
        while last < len(lines) and lines[last].startswith(">"):  # the block quote runs on while lines start with ">"
            last += 1
        hits.append(Hit(rel, note, f"agent.py now runs the networks, so the note that calls it a baseline is wrong: "
                                   f"delete lines {note} to {last} of README.md, then write what agent.py does from the file itself"))
    if PLACEHOLDER in labels:
        hits.append(Hit(rel, row, f"agent.py now runs the networks, so its row cannot be labelled '{PLACEHOLDER}': label it "
                                  "'learned' and say what the file does (what it loads, what it runs per decision, how heads become buttons)"))
    for words, fact, missing in README_CLAIMS:
        line = line_of(words)
        if line and not facts[fact]:
            hits.append(Hit(rel, line, f"README.md says '{words}', but agent.py {missing}"))
    agent_source = (folder / "agent.py").read_text()
    sources = "\n".join(p.read_text() for p in submission_files(folder) if p.suffix == ".py")
    for number, line in enumerate(lines, 1):
        for name in CITED_CONSTANT.findall(line):
            if not re.search(rf"\b{name}\b", sources):
                hits.append(Hit(rel, number, f"README.md cites '{name}', which no Python file under {folder.name}/ contains"))
        for name in CITED_WEIGHTS.findall(line):
            if f'"{name}"' not in agent_source and f"'{name}'" not in agent_source:
                hits.append(Hit(rel, number, f"README.md cites the weights file '{name}', which agent.py does not name"))
    return hits


# ---------------------------------------------------------------------------------------------
# 5. Key-like tokens
# ---------------------------------------------------------------------------------------------
KEY_PREFIX = "cr" + "_"  # in two pieces, so that this file passes the scan it defines
# A word that starts with the prefix and goes on with letters or digits. "incr_x" and "scr_w" are
# not hits: there the prefix does not start the word.
KEY_TOKEN = re.compile(r"\b" + KEY_PREFIX + r"[A-Za-z0-9]+")
# History cannot be edited by renaming a variable, so there only a long token counts.
KEY_IN_HISTORY = r"(^|[^A-Za-z0-9_])" + KEY_PREFIX + r"[A-Za-z0-9]{16}"


def _git(root: Path, *args: str) -> str | None:
    """Output of a read-only git command, or None when `root` is not a git checkout."""
    git = shutil.which("git")
    if not git or not (root / ".git").exists():
        return None
    out = subprocess.run([git, "-C", str(root), *args], capture_output=True, timeout=120)
    return out.stdout.decode("utf-8", "surrogateescape") if out.returncode == 0 else None


def repo_files(root: Path) -> list[Path]:
    """Every file git would commit from `root`: tracked files plus new files that are not ignored.

    Ignored files (a local .env, run outputs, the virtual environment) are never opened. Without git
    (for example an unpacked archive) every file outside the usual scratch folders is taken.
    """
    listing = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if listing is not None:
        return sorted({root / name for name in listing.split("\0") if name and (root / name).is_file()})
    scratch = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "runs"}
    return sorted(p for p in root.rglob("*") if p.is_file() and not scratch.intersection(p.relative_to(root).parts))


def scan_keys(files: list[Path], root: Path) -> list[Hit]:
    """Key-like tokens in text files. The token itself is not printed, only its first characters."""
    hits = []
    for path in files:
        text = read_text(path)
        if text is None:
            continue
        rel = path.relative_to(root).as_posix()
        for number, line in enumerate(text.split("\n"), 1):
            for token in KEY_TOKEN.findall(line):
                hits.append(Hit(rel, number, f"key-like token '{token[:5]}...' ({len(token)} characters)"))
    return hits


# ---------------------------------------------------------------------------------------------
# The checks on this repository
# ---------------------------------------------------------------------------------------------
def test_scan_covers_the_agent_code(capsys):
    """The lint is not vacuous: it reads the entry point and the shared spec, and skips no Python file."""
    files = submission_files(SUBMISSION)
    skipped = binary_files(SUBMISSION)
    assert SUBMISSION / "agent.py" in files and SUBMISSION / "doomnet" / "spec.py" in files
    assert not [p for p in skipped if p.suffix == ".py"], "a Python file under submission/ is binary"
    with capsys.disabled():
        names = ", ".join(p.relative_to(ROOT).as_posix() for p in skipped) or "none"
        print(f"\n[rules] {len(files) - len(skipped)} text files under submission/ scanned; binary files skipped: {names}")


def test_submission_text_has_no_banned_names():
    fail_if(scan_names(SUBMISSION), "banned names under submission/")


def test_submission_code_has_no_banned_constructs():
    fail_if(scan_python(SUBMISSION), "banned imports, process calls, GPU use or search loops under submission/")


def test_readme_labels_every_module():
    """submission/README.md has one table row per module, with nothing but the three labels."""
    rows = {path: {label.strip() for label in labels.split(",")}
            for path, labels in README_ROW.findall((SUBMISSION / "README.md").read_text())}
    modules = {p.relative_to(SUBMISSION).as_posix() for p in submission_files(SUBMISSION)
               if p.suffix == ".py" and p.name != "__init__.py"}
    assert set(rows) == modules, (f"submission/README.md must have one table row per module: "
                                  f"not listed {sorted(modules - set(rows))}, listed but absent {sorted(set(rows) - modules)}")
    for path, labels in rows.items():
        if path == "agent.py" and labels == {PLACEHOLDER}:
            continue  # allowed only while agent.py is the template's baseline; test_readme_agrees_with_agent_py decides that
        assert labels and labels <= LABELS, f"submission/README.md labels {path} as {sorted(labels)}; allowed: {sorted(LABELS)}"


def test_readme_agrees_with_agent_py():
    """The README neither describes an agent that is not in the tree nor calls the assembled agent a baseline."""
    fail_if(readme_problems(SUBMISSION), "submission/README.md and submission/agent.py disagree")


def test_no_key_like_token_in_repo_files():
    files = repo_files(ROOT)
    assert Path(__file__).resolve() in files, "the secret scan does not even cover this file"
    fail_if(scan_keys(files, ROOT),
            "key-like tokens in files git would commit (remove the key and rotate it; if it is an "
            f"ordinary identifier, rename it so that no word starts with '{KEY_PREFIX}')")


def test_no_key_like_token_in_git_history():
    """A key that was committed and deleted again is still public once the repository is public."""
    log = _git(ROOT, "log", "--all", "--date=short", "--format=%h %ad %s", "-G", KEY_IN_HISTORY)
    if log is None:
        pytest.skip("not a git checkout (or git log failed), so there is no history to search")
    commits = log.strip().splitlines()
    if commits:
        pytest.fail("these commits add or remove a line with a key-like token; rotate the key and rewrite "
                    "the history before the repository is made public:\n" + "\n".join(commits), pytrace=False)


# ---------------------------------------------------------------------------------------------
# Proof that the checks bite. Nothing below touches submission/: every banned line is planted in
# a temporary folder laid out like the repository.
# ---------------------------------------------------------------------------------------------
PLANT_LINE = 5  # every planted line is line 5 of its file


def plant(folder: Path, name: str, line: str) -> Path:
    """Write `line` as line PLANT_LINE of folder/name; the lines around it are harmless."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    harmless = ["import math", "import numpy as np", "import torch", "x = np.zeros(3)"] if name.endswith(".py") else ["notes", "", "", ""]
    path.write_text("\n".join([*harmless, line, "y = 1" if name.endswith(".py") else "end"]) + "\n")
    return path


NAME_PLANTS = [
    # (file under submission/, planted line, banned name that must be reported)
    # --- the eight tokens of the build spec
    ("doomnet/aim.py", "yaw = math.atan2(dy, dx)", "atan2"),
    ("doomnet/aim.py", "yaw = np.arctan2(dy, dx)", "arctan2"),
    ("doomnet/aim.py", "from math import atan2 as angle_of", "atan2"),
    ("doomnet/plan.py", "import heapq", "heapq"),
    ("doomnet/plan.py", "from collections import deque", "deque"),
    ("doomnet/plan.py", "def astar_search(grid): pass", "astar"),
    ("doomnet/plan.py", "class AStarPlanner: pass", "astar"),
    ("doomnet/plan.py", "path = self.run_Astar(grid)", "astar"),
    ("doomnet/plan.py", "# Dijkstra over the grid", "dijkstra"),
    ("doomnet/plan.py", "order = bfs(grid, start)", "bfs"),
    ("doomnet/plan.py", "class GridBFS: pass", "bfs"),
    ("doomnet/plan.py", "seen = BFS_ORDER[0]", "bfs"),
    ("doomnet/plan.py", "mask = flood_fill(free, seed)", "flood"),
    ("README.md", "The planner runs a flood fill from the exit.", "flood"),
    ("weights/config.json", '{"planner": "dijkstra"}', "dijkstra"),
    # --- the spec's tokens inside longer words: a digit after them, a word before or after them
    ("doomnet/plan.py", "def bfs4(grid): pass", "bfs"),
    ("doomnet/plan.py", "def bfs2d(grid): pass", "bfs"),
    ("doomnet/plan.py", "def gridbfs(grid): pass", "bfs"),
    ("doomnet/plan.py", "def bfsorder(grid): pass", "bfs"),
    ("doomnet/plan.py", "def astar2(grid): pass", "astar"),
    ("doomnet/plan.py", "def gridastar(grid): pass", "astar"),
    ("doomnet/plan.py", "def astarsearch(grid): pass", "astar"),
    ("doomnet/plan.py", "def djikstra(grid): pass", "dijkstra"),
    ("doomnet/plan.py", "def dijsktra(grid): pass", "dijkstra"),
    ("doomnet/plan.py", "obfs_len, subfs = 1, 2", "bfs"),
    ("doomnet/plan.py", "datastart = 0", "astar"),
    # --- the same techniques under other names
    ("doomnet/plan.py", "a_star = search(grid)", "a-star"),
    ("doomnet/plan.py", "def a_star2(grid): pass", "a-star"),
    ("doomnet/plan.py", "# A-star over the grid", "a-star"),
    ("doomnet/plan.py", "# A* over the grid", "A*"),
    ("README.md", "The route to the exit comes from A*.", "A*"),
    ("doomnet/plan.py", "stack = dfs(root)", "dfs"),
    ("doomnet/plan.py", "def dfs2(root): pass", "dfs"),
    ("doomnet/plan.py", "order = runDfs(root)", "dfs"),
    ("doomnet/aim.py", "yaw = math.atan(dy / dx)", "atan"),
    ("doomnet/aim.py", "yaw = np.arctan(dy / dx)", "arctan"),
    ("doomnet/aim.py", "yaw = np.arcsin(sin_rel)", "arcsin"),
    ("doomnet/aim.py", "yaw = math.asin(sin_rel)", "asin"),
    ("doomnet/aim.py", "yaw = np.arccos(cos_rel)", "arccos"),
    ("doomnet/aim.py", "yaw = torch.acos(cos_rel)", "acos"),
    ("doomnet/aim.py", "import cmath", "cmath"),
    ("doomnet/plan.py", "queue = [start]", "queue"),
    ("doomnet/plan.py", "enqueue(cell)", "queue"),
    ("doomnet/plan.py", "frontier = [goal]", "frontier"),
    ("doomnet/plan.py", "def wavefront(grid): pass", "wavefront"),
    ("doomnet/plan.py", "def pathfind(grid): pass", "pathfind"),
    ("README.md", "Path-finding runs on the memory map.", "pathfind"),
    ("doomnet/plan.py", "class PathPlanner: pass", "path planner"),
    ("doomnet/plan.py", "route = path_planner.run(grid)", "path planner"),
    ("doomnet/plan.py", "def shortest_path(grid): pass", "shortest path"),
    ("doomnet/plan.py", "def breadth_first(grid): pass", "breadth-first"),
    ("doomnet/plan.py", "# breadth-first search over the map", "breadth-first"),
    ("doomnet/plan.py", "class DepthFirstWalk: pass", "depth-first"),
    ("doomnet/plan.py", "open_set = {start}", "open_set"),
    ("doomnet/plan.py", "came_from = {}", "came_from"),
    ("doomnet/plan.py", "parent = self.cameFrom[cell]", "came_from"),
]


@pytest.mark.parametrize("name,line,banned", NAME_PLANTS, ids=[f"{i}: {line}" for i, (_, line, _) in enumerate(NAME_PLANTS)])
def test_planted_name_is_found(tmp_path, name, line, banned):
    plant(tmp_path / "submission", name, line)
    hits = scan_names(tmp_path / "submission")
    assert hits and {(h.path, h.line) for h in hits} == {(f"submission/{name}", PLANT_LINE)}, hits
    assert any(f"'{banned}'" in h.what for h in hits), hits
    with pytest.raises(pytest.fail.Exception, match=rf"submission/{re.escape(name)}:{PLANT_LINE}: "):
        fail_if(hits, "planted")


CODE_PLANTS = [
    # (planted line, part of the message that must be reported)
    ("import vizdoom", "'vizdoom'"),
    ("import vizdoom as vzd", "'vizdoom'"),
    ("from doom_env import make_env", "'doom_env'"),
    ("from doom_env.wrapper import DoomEnv", "'doom_env.wrapper'"),
    ("import socket", "'socket'"),
    ("import os, subprocess", "'subprocess'"),
    ("import requests", "'requests'"),
    ("import urllib.parse", "'urllib.parse'"),
    ("from urllib.request import urlopen", "'urllib.request'"),
    ("from stable_baselines3 import PPO", "'stable_baselines3'"),
    ("import training", "'training'"),
    ("from training.teacher import Teacher", "'training.teacher'"),
    ("from training import paths", "'training'"),
    ("import queue", "'queue'"),
    ("import _socket", "'_socket'"),
    ("import _posixsubprocess", "'_posixsubprocess'"),
    ("from concurrent.futures import ProcessPoolExecutor", "'concurrent.futures'"),
    ("import pty", "'pty'"),
    ("import imaplib", "'imaplib'"),
    ("import runpy", "'runpy'"),
    ("import scipy.ndimage", "only numpy and torch"),
    ("import gymnasium", "only numpy and torch"),
    ("mod = __import__('socket')", "'socket'"),
    ("mod = importlib.import_module('training.rl.reward')", "'training.rl.reward'"),
    ("mod = importlib.import_module(name)", "computed at run time"),
    ("exec('import ' + name)", "calls exec()"),
    ("value = eval(text)", "calls eval()"),
    # --- GPU use
    ("ok = torch.cuda.is_available()", "'.cuda'"),
    ("net = net.cuda()", "'.cuda'"),
    ('x = x.to("cuda")', "GPU device 'cuda'"),
    ("x = x.to('cuda:0')", "GPU device 'cuda:0'"),
    ('x = x.to(device="cuda")', "GPU device 'cuda'"),
    ('dev = torch.device("cuda")', "GPU device 'cuda'"),
    ('z = torch.zeros(3, device="cuda")', "GPU device 'cuda'"),
    ('w = torch.load(path, map_location="cuda")', "GPU device 'cuda'"),
    ('x = x.to("mps")', "GPU device 'mps'"),
    ('x = x.to("cuda" if flag else "cpu")', "GPU device 'cuda'"),
    ('device = "cuda"', "GPU device 'cuda'"),
    ("x = x.to(f'cuda:{index}')", "GPU device 'cuda:'"),
    ("x = x.to('cuda:%d' % index)", "GPU device 'cuda:%d'"),
    ("torch.set_default_device('cuda')", "GPU device 'cuda'"),
    ("w = torch.load(path, 'cuda')", "GPU device 'cuda'"),
    ("ok = getattr(torch, 'cuda').is_available()", "GPU device 'cuda'"),
    ("x = x.to(0)", "by its number"),
    ("dev = torch.device(0)", "by its number"),
    ("dev = torch.accelerator.current_accelerator()", "'.accelerator'"),
    ("from torch import cuda", "GPU interface"),
    ("import torch.cuda", "GPU interface"),
    ("os.system('ls')", "os.system"),
    ("import os as o; o.system('ls')", "os.system"),
    ("ctx = torch.multiprocessing.get_context('spawn')", "'.multiprocessing'"),
    ("import torch.multiprocessing as mp", "'multiprocessing' interface"),
    ("from torch import hub", "'hub' interface"),
    ("w = torch.hub.load_state_dict_from_url(url)", "'.hub'"),
    # --- the angle of a complex number: the banned inverse tangent without its name
    ("yaw = np.angle(dx + 1j * dy)", ".angle()"),
    ("yaw = torch.angle(torch.complex(dx, dy))", ".angle()"),
    ("yaw = phases.phase(complex(dx, dy))", ".phase()"),
    ("from os import popen", "starts another process"),
    ("def broken(:", "cannot be parsed"),
    # --- search structure: none of these lines contains a banned name
    ("while todo: todo = step(todo)", "`while` loop"),
    ("while True: pass", "`while` loop"),
    ("for cell in cells: break", "`break`"),
    ("cell = cells.pop(0)", ".pop(0)"),
    ("cell = cells.popleft()", ".popleft()"),
    ("def fill(grid, r, c): return fill(grid, r + 1, c)", "'fill' calls itself"),
]


@pytest.mark.parametrize("line,expected", CODE_PLANTS, ids=[line for line, _ in CODE_PLANTS])
def test_planted_code_is_found(tmp_path, line, expected):
    plant(tmp_path / "submission", "doomnet/extra.py", line)
    hits = scan_python(tmp_path / "submission")
    assert hits and {(h.path, h.line) for h in hits} == {("submission/doomnet/extra.py", PLANT_LINE)}, hits
    assert any(expected in h.what for h in hits), hits


# Hand-written searches spelt with neutral names: no banned word in any of them, so only the shape
# of the code can give them away. Each entry: (what it is, source, the message parts that must be reported).
SEARCHES = [
    ("level-by-level walk with a list as worklist", '''
def reachable(grid, start):
    todo = [start]
    seen = {start}
    while todo:
        r, c = todo.pop(0)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if grid[nxt] == 0 and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen
''', ["`while` loop", ".pop(0)"]),
    ("distance field grown ring by ring, then the step that lowers it", '''
def distance_field(free, goal):
    dist = np.full(free.shape, -1)
    dist[goal] = 0
    ring = [goal]
    while ring:
        nxt_ring = []
        for r, c in ring:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if free[p] and dist[p] < 0:
                    dist[p] = dist[r, c] + 1
                    nxt_ring.append(p)
        ring = nxt_ring
    return dist
''', ["`while` loop"]),
    ("walk with a stack", '''
def region(grid, start):
    stack = [start]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        stack.extend(neighbours(grid, cell))
    return seen
''', ["`while` loop"]),
    ("cheapest-first search with a set and min()", '''
def cheapest(cost, start, goal):
    best = {start: 0.0}
    pending = {start}
    parent = {}
    while pending:
        cur = min(pending, key=best.get)
        pending.discard(cur)
        if cur == goal:
            break
        for n in neighbours(cost, cur):
            g = best[cur] + cost[n]
            if g < best.get(n, 1e9):
                best[n], parent[n] = g, cur
                pending.add(n)
    return parent
''', ["`while` loop"]),
    ("recursive region fill", '''
def fill(grid, r, c, seen):
    if (r, c) in seen or grid[r][c]:
        return
    seen.add((r, c))
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        fill(grid, r + dr, c + dc, seen)
''', ["'fill' calls itself"]),
    ("recursive walk written as a method", '''
class Walker:
    def visit(self, cell, seen):
        seen.add(cell)
        for n in self.neighbours(cell):
            if n not in seen:
                self.visit(n, seen)
''', ["'visit' calls itself"]),
    ("relaxation of a whole array until nothing changes", '''
def spread(free, goal):
    dist = np.full(free.shape, np.inf)
    dist[goal] = 0
    for _ in range(free.size):
        best = np.minimum.reduce([np.roll(dist, s, a) for s in (1, -1) for a in (0, 1)]) + 1
        new = np.where(free, np.minimum(dist, best), np.inf)
        if (new == dist).all():
            break
        dist = new
    return dist
''', ["`break`"]),
    ("sorted list as a priority list", '''
import bisect

def search(cost, start):
    pending = [(0.0, start)]
    done = set()
    while pending:
        g, cur = pending.pop(0)
        if cur in done:
            continue
        done.add(cur)
        for n in neighbours(cost, cur):
            bisect.insort(pending, (g + cost[n], n))
    return done
''', ["`while` loop", ".pop(0)"]),
]


@pytest.mark.parametrize("label,source,expected", SEARCHES, ids=[label for label, _, _ in SEARCHES])
def test_search_without_a_banned_name_is_found_by_its_shape(tmp_path, label, source, expected):
    folder = tmp_path / "submission"
    (folder / "doomnet").mkdir(parents=True)
    (folder / "doomnet" / "helper.py").write_text("import numpy as np\n" + source)
    assert scan_names(folder) == [], "the example is meant to contain no banned name"
    hits = scan_python(folder)
    assert hits and all(h.path == "submission/doomnet/helper.py" for h in hits), hits
    for part in expected:
        assert any(part in h.what for h in hits), (part, hits)


ALLOWED_CODE = '''"""Prose is not code: import socket, torch.cuda, x.to("cuda"), from training import paths.

No path planning and no search here. The exit estimate came from the detector; the depth image is
read depth first, then colour; a door can be an open set piece. A loop that runs while a door moves.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import time
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import agent
from doomnet import spec
from doomnet.spec import action_masks
from . import spec as same_package
from .model import PolicyNet

meta_start, a_start, beta_start, data_start, pdfs, restart = 1, 2, 3, 4, 5, 6   # short names inside longer words
A_START, A_END, DATA_START, ATANH_CLIP = 0.1, 0.9, 0, 0.999                     # upper-case constants
tanh_inverse = torch.atanh(torch.zeros(1)) + np.arctanh(0.0)                    # hyperbolic, not the banned one
sinh_inverse = torch.asinh(torch.zeros(1)) + np.arccosh(1.0)                    # hyperbolic as well
increasing, basin, angle, phase = 1, 2, 3, 4                                    # words and plain names, not calls
incr_count = scr_w = 0
A = B = np.eye(2)
product = A * B + A*B + A**2 + np.stack([A, *[B]])                              # a matrix named A, not the search
open_settings, reopen_set, welcome_from = 1, 2, 3


def encode(x):
    return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()                     # the parent's method, not this one
        nn.Module.__init__(self)
        self.cell = nn.Conv2d(1, 1, 3, padding=1)
        self.cache = {}

    def encode(self, x):
        return encode(x)                       # the module-level function of the same name

    def reset(self):
        self.cache.pop("state", None)          # removes a key
        self.cell.reset_parameters()           # another object's method

    def forward(self, x, device="cpu"):
        if self.training:                      # an attribute named training is not the training package
            x = x + 1
        if str(device).startswith("cuda") or device == "cuda:0" or device in ("cuda", "mps"):
            print("device argument ignored: this agent runs on the CPU")   # reading the argument is fine, using it is not
        assert device in ("cuda", "cpu")
        y = x.to("cpu").to(torch.float32).to(torch.device("cpu")).to(x.device)
        for _ in range(8):                     # a fixed number of passes of a learned cell: value propagation
            y = self.cell(y)
        last = [1, 2, 3].pop()                 # the end of a list, not its front
        second = [1, 2, 3].pop(1)
        return torch.zeros(3, device="cpu") + y + last + second


net = Net().eval()                             # the method, not the builtin
started = time.perf_counter()


allow_untrained = os.environ.get("DOOMNET_ALLOW_UNTRAINED") == "1"
weights = Path(__file__).resolve().parent / "weights"
digest = hashlib.sha256(b"").hexdigest()[:12]
'''


def test_allowed_look_alikes_pass(tmp_path):
    """Words that only resemble a banned name, prose about banned modules, CPU device calls and fixed-count loops are fine."""
    folder = tmp_path / "submission"
    (folder / "doomnet").mkdir(parents=True)
    (folder / "agent.py").write_text("import numpy as np\n")
    (folder / "doomnet" / "spec.py").write_text("X = 1\n")
    (folder / "doomnet" / "model.py").write_text(ALLOWED_CODE)
    (folder / "README.md").write_text("No path planning, no trigonometric aiming, no graph search. Runs on the CPU, never on cuda.\n"
                                      "The arctangent is not used. The matrices *A* and *B* are rotations; y = A * x.\n")
    (folder / "weights").mkdir()
    (folder / "weights" / "policy.pt").write_bytes(b"PK\x03\x04\0\0" + b"bfs flood atan2 queue" * 50)  # binary: skipped
    assert scan_names(folder) == []
    assert scan_python(folder) == []
    assert [p.name for p in binary_files(folder)] == ["policy.pt"]


# ---- README.md against agent.py -----------------------------------------------------------------
TEMPLATE_AGENT = '''import numpy as np


class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        self.rng = np.random.default_rng(0)

    def reset(self):
        pass

    def act(self, observation):
        return self.rng.integers(0, 2, 9, dtype=np.int8)
'''

ASSEMBLED_AGENT = '''import time

import numpy as np
import torch

from doomnet import spec
from doomnet.encode import Encoder
from doomnet.memory import MemoryMap
from doomnet.model import ExitDetector, PolicyNet
from doomnet.odometry import Odometry


GOVERNOR_NETWORK_MS = 32.0
POLICY_FILE = "policy.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        self.policy = PolicyNet()
        self.policy.load_state_dict(torch.load("weights/" + POLICY_FILE, map_location="cpu"))

    def reset(self):
        pass

    def act(self, observation):
        start = time.perf_counter()
        try:
            action = self.step(observation)
        except Exception:
            action = np.zeros(spec.NUM_BUTTONS, np.int8)
        self.spent = time.perf_counter() - start
        return action
'''

NOTE = f"> **{STATUS_NOTE}.** The file in the tree is still the template's baseline.\n> It presses random buttons.\n"

# Two READMEs written for these tests, so that they do not depend on which of the two states the
# real README is in. Lines 3 and 4 of the first are the status note; the table rows follow.
BASELINE_README = f"""# The agent

{NOTE}
| Module | Label | What it does |
|---|---|---|
| `doomnet/odometry.py` | observation encoding | Odometry: the pose estimate. |
| `doomnet/memory.py` | observation encoding | The memory map. |
| `doomnet/encode.py` | observation encoding | The encoder of one observation. |
| `doomnet/model.py` | learned | `PolicyNet` and the exit detector. |
| `agent.py` | placeholder | Random buttons; no file is read from `weights/`. |
"""

ASSEMBLED_README = """# The agent

| Module | Label | What it does |
|---|---|---|
| `doomnet/odometry.py` | observation encoding | Odometry: the pose estimate. |
| `doomnet/memory.py` | observation encoding | The memory map. |
| `doomnet/encode.py` | observation encoding | The encoder of one observation. |
| `doomnet/model.py` | learned | `PolicyNet` and the exit detector. |
| `agent.py` | learned | Loads the weights, then runs the modules above on every decision. |

A time governor follows the running mean of `act()` time (`GOVERNOR_NETWORK_MS`). An exception
handler returns no buttons for a decision on which an internal error occurred.
The parameters are in `weights/policy.pt`.
"""


def readme_case(tmp_path: Path, readme: str, agent: str) -> list[Hit]:
    """readme_problems() for a folder that holds exactly this README.md and this agent.py."""
    folder = tmp_path / "submission"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(readme)
    (folder / "agent.py").write_text(agent)
    return readme_problems(folder)


def line_with(readme: str, words: str) -> int:
    """1-based number of the first line of `readme` that contains `words`."""
    return next(i for i, line in enumerate(readme.split("\n"), 1) if words in line)


def test_the_two_consistent_states_pass(tmp_path):
    """A README that calls the baseline a baseline, and one that describes an agent that has every part it names."""
    assert readme_case(tmp_path / "baseline", BASELINE_README, TEMPLATE_AGENT) == []
    assert readme_case(tmp_path / "assembled", ASSEMBLED_README, ASSEMBLED_AGENT) == []


def test_readme_must_not_describe_an_agent_that_is_not_there(tmp_path):
    """While agent.py is the template's baseline: the README of the assembled agent is rejected, at the
    lines that make the claims, and so is each single step towards it."""
    row = line_with(ASSEMBLED_README, "| `agent.py` |")
    hits = readme_case(tmp_path / "all", ASSEMBLED_README, TEMPLATE_AGENT)
    assert {h.path for h in hits} == {"submission/README.md"}
    found = {(h.line, h.what.split(":")[0].split(",")[0]) for h in hits}
    assert found == {
        (row, "agent.py does not run the networks of doomnet/model.py yet and README.md does not say so"),
        (row, "agent.py is still the template's random baseline"),
        (row, "README.md says 'loads the weights'"),
        (line_with(ASSEMBLED_README, "time governor"), "README.md says 'time governor'"),
        (line_with(ASSEMBLED_README, "An exception"), "README.md says 'exception handler'"),
    }, hits
    with pytest.raises(pytest.fail.Exception, match=rf"submission/README\.md:{row}: agent\.py does not run the networks"):
        fail_if(hits, "planted")

    no_note = BASELINE_README.replace(NOTE, "")
    hits = readme_case(tmp_path / "no_note", no_note, TEMPLATE_AGENT)
    assert len(hits) == 1 and "does not say so" in hits[0].what and "`agent.py`" in no_note.split("\n")[hits[0].line - 1], hits

    relabelled = BASELINE_README.replace("| `agent.py` | placeholder |", "| `agent.py` | learned |")
    hits = readme_case(tmp_path / "label", relabelled, TEMPLATE_AGENT)
    assert [(h.line, "label it 'placeholder'" in h.what) for h in hits] == [(line_with(relabelled, "| `agent.py` |"), True)], hits

    for words in AGENT_ONLY:
        claimed = BASELINE_README + f"\nThe agent has this part: {words}.\n"
        hits = readme_case(tmp_path / words.replace(" ", "_"), claimed, TEMPLATE_AGENT)
        assert [(h.line, f"says '{words}'" in h.what) for h in hits] == [(line_with(claimed, "The agent has this part"), True)], hits


def test_readme_must_not_call_the_assembled_agent_a_baseline(tmp_path):
    """The opposite direction: once agent.py runs the networks, the note and the placeholder label must go."""
    hits = readme_case(tmp_path / "stale", BASELINE_README, ASSEMBLED_AGENT)
    row = line_with(BASELINE_README, "| `agent.py` |")
    assert [(h.path, h.line) for h in sorted(hits)] == [("submission/README.md", 3), ("submission/README.md", row)], hits
    assert "delete lines 3 to 4" in sorted(hits)[0].what and "label it 'learned'" in sorted(hits)[1].what

    with_note = ASSEMBLED_README.replace("# The agent\n\n", "# The agent\n\n" + NOTE + "\n")
    hits = readme_case(tmp_path / "note", with_note, ASSEMBLED_AGENT)
    assert [(h.line, "delete lines 3 to 4" in h.what) for h in hits] == [(3, True)], hits


def test_readme_claims_are_checked_against_the_assembled_agent(tmp_path):
    """Each part taken out of the assembled agent is reported at the README line that claims it."""
    removed = {
        "time governor": ASSEMBLED_AGENT.replace("time.perf_counter()", "0.0"),
        "exception handler": ASSEMBLED_AGENT.replace("        try:\n            action = self.step(observation)\n        except Exception:\n"
                                                     "            action = np.zeros(spec.NUM_BUTTONS, np.int8)\n",
                                                     "        action = self.step(observation)\n"),
        "loads the weights": ASSEMBLED_AGENT.replace('        self.policy.load_state_dict(torch.load("weights/" + POLICY_FILE, map_location="cpu"))\n', ""),
        "odometry": ASSEMBLED_AGENT.replace("from doomnet.odometry import Odometry\n", ""),
        "memory map": ASSEMBLED_AGENT.replace("from doomnet.memory import MemoryMap\n", ""),
        "encoder": ASSEMBLED_AGENT.replace("from doomnet.encode import Encoder\n", ""),
    }
    for words, agent in removed.items():
        assert agent != ASSEMBLED_AGENT, words
        ast.parse(agent)
        hits = readme_case(tmp_path / words.replace(" ", "_"), ASSEMBLED_README, agent)
        assert len(hits) == 1 and f"says '{words}'" in hits[0].what, (words, hits)
        assert words.split()[0].lower() in ASSEMBLED_README.split("\n")[hits[0].line - 1].lower(), (words, hits)
    renamed = {
        "cites 'GOVERNOR_NETWORK_MS'": ASSEMBLED_AGENT.replace("GOVERNOR_NETWORK_MS", "SLOW_MS"),
        "cites the weights file 'policy.pt'": ASSEMBLED_AGENT.replace('"policy.pt"', '"net.pt"'),
    }
    for i, (message, agent) in enumerate(renamed.items()):
        assert agent != ASSEMBLED_AGENT, message
        hits = readme_case(tmp_path / f"renamed{i}", ASSEMBLED_README, agent)
        assert len(hits) == 1 and message in hits[0].what, (message, hits)
        assert message.split("'")[1] in ASSEMBLED_README.split("\n")[hits[0].line - 1], (message, hits)


def test_the_real_readme_is_in_the_state_of_the_real_agent():
    """Whichever state agent.py is in, the real README must be in the same one and say so in its table."""
    facts = agent_facts(SUBMISSION / "agent.py")
    text = (SUBMISSION / "README.md").read_text()
    rows = dict(README_ROW.findall(text))
    assert "agent.py" in rows, "submission/README.md has no table row for agent.py"
    labels = {label.strip() for label in rows["agent.py"].split(",")}
    if facts["networks"]:
        assert "learned" in labels and STATUS_NOTE not in text, f"agent.py runs the networks; its row is labelled {sorted(labels)}"
        for words in AGENT_ONLY:  # the three things a reader needs to know about the assembled agent
            assert re.search(r"\s+".join(words.split()), text, re.IGNORECASE), f"submission/README.md does not say '{words}'"
    else:
        assert labels == {PLACEHOLDER} and STATUS_NOTE in text, f"agent.py is the template's baseline; its row is labelled {sorted(labels)}"


# ---- secrets ------------------------------------------------------------------------------------
FAKE_KEY = KEY_PREFIX + "0123456789abcdef" * 4  # shaped like a key, obviously not a real one


def test_planted_key_is_found_and_not_printed(tmp_path):
    leak = plant(tmp_path, "training/notes.txt", f"export API_KEY={FAKE_KEY}")
    plant(tmp_path, "training/short.py", f'token = "{KEY_PREFIX}abc123"')
    plant(tmp_path, "training/fine.py", "incr_count = scr_w = xcr_1 = 0  # the prefix does not start these words")
    (tmp_path / "weights.bin").write_bytes(b"\0\0" + FAKE_KEY.encode())  # binary: skipped
    hits = scan_keys(repo_files(tmp_path), tmp_path)
    assert [(h.path, h.line) for h in hits] == [("training/notes.txt", PLANT_LINE), ("training/short.py", PLANT_LINE)]
    assert FAKE_KEY not in str(hits) and f"({len(FAKE_KEY)} characters)" in hits[0].what
    assert leak.read_text().count(FAKE_KEY) == 1
    with pytest.raises(pytest.fail.Exception, match=rf"training/notes\.txt:{PLANT_LINE}: key-like token"):
        fail_if(hits, "planted")


def test_key_scan_follows_git_ignore_rules(tmp_path):
    """New files that are not ignored are scanned before they are ever committed; ignored files are not opened."""
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text(".env\nruns/\n")
    (tmp_path / ".env").write_text(f"API_KEY={FAKE_KEY}\n")
    plant(tmp_path, "runs/train.txt", f"key {FAKE_KEY}")
    plant(tmp_path, "training/config.py", f'KEY = "{FAKE_KEY}"')
    files = repo_files(tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in files) == [".gitignore", "training/config.py"]
    assert [(h.path, h.line) for h in scan_keys(files, tmp_path)] == [("training/config.py", PLANT_LINE)]


def test_the_checks_fail_under_pytest_when_a_banned_line_is_added(tmp_path):
    """End to end: a copy of this file, run by pytest on a copy of the layout, passes when the code is
    clean and fails, naming file and line, once banned lines are added."""
    (tmp_path / "training" / "tests").mkdir(parents=True)
    copy = shutil.copy(__file__, tmp_path / "training" / "tests" / "test_rules.py")
    (tmp_path / "submission" / "doomnet").mkdir(parents=True)
    (tmp_path / "submission" / "agent.py").write_text("import numpy as np\n")
    (tmp_path / "submission" / "doomnet" / "spec.py").write_text("import numpy as np\n")
    checks = [f"{copy}::{name}" for name in ("test_scan_covers_the_agent_code", "test_submission_text_has_no_banned_names",
                                             "test_submission_code_has_no_banned_constructs", "test_no_key_like_token_in_repo_files")]

    def run() -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}  # the inner run uses only the options below
        return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *checks],
                              cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)

    clean = run()
    assert clean.returncode == 0, clean.stdout + clean.stderr

    plant(tmp_path / "submission", "doomnet/planner.py", "order = gridbfs(grid)")
    plant(tmp_path / "submission", "doomnet/net.py", "import vizdoom")
    plant(tmp_path / "submission", "doomnet/walk.py", "while todo: todo = step(todo)")
    plant(tmp_path, "training/run.sh", f"export API_KEY={FAKE_KEY}")
    dirty = run()
    print(dirty.stdout)
    assert dirty.returncode == 1, dirty.stdout + dirty.stderr
    assert "3 failed, 1 passed" in dirty.stdout
    for expected in (f"submission/doomnet/planner.py:{PLANT_LINE}: banned name 'bfs'",
                     f"submission/doomnet/net.py:{PLANT_LINE}: imports 'vizdoom'",
                     f"submission/doomnet/walk.py:{PLANT_LINE}: has a `while` loop",
                     f"training/run.sh:{PLANT_LINE}: key-like token"):
        assert expected in dirty.stdout, dirty.stdout
    assert FAKE_KEY not in dirty.stdout
