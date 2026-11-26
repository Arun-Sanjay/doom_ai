"""Authored validation maps: grid layouts drawn by language models, filtered by the generator's checks.

Training-time scaffolding, never imported by the agent.

The frozen suites `heldout_v1` and `dev_v1` come from the procedural generator, so they share its
habits with the training maps. This script builds a suite from a different author. It asks chat
models to draw ASCII maps for tiers 1 to 4 in the generator's legend, keeps only the maps that pass
the checks a generated map has to pass, drops near-duplicates and writes
`training/suites/authored_v1.jsonl` in the frozen-suite format that `training.evaluate` reads::

    python -m training.mapgen.authored --per-tier 40      # needs CREO_API_KEY and CREO_BASE_URL
    python -m training.mapgen.authored --dry-run          # canned replies, no network
    python -m training.mapgen.authored --list-models      # ids offered by the endpoint

Requests
--------
OpenAI-compatible chat completions with the standard library only: POST {CREO_BASE_URL}/chat/completions
with a bearer key, body {model, messages, temperature, max_tokens}. Both values are read from the
environment and from nowhere else. Neither of them, no header and no request body is ever printed
or written; a failed request shows its status code and at most 200 characters of the response.
`--max-requests` is a hard cap on the chat requests of one run. Before the first request the
checks are tried on hand-drawn maps (`self_check`), so a broken installation costs no request.
`--summary-json FILE` adds the summary of the run to a JSON list (a log for `training/logs/`) and
`--save-replies DIR` keeps the text of every reply (for `runs/`); requests are never stored.

Every request asks for several maps of one tier and carries the legend (`udmf.LEGEND`), the rules
of a valid map, what the tier tests (docs/EVALUATION.md), the size range and content of that tier
(`layouts.TIERS`), two tiny hand-drawn examples of the format, and a design brief. The brief
changes from request to request (plan style, size, exit position, a tier-specific twist), so the
layouts differ from each other and from the procedural ones. Models and temperatures are used in turn.

Filter
------
Maps are read from the fenced code blocks of a reply. Rows are tidied (indentation, one blank
between symbols, short rows padded with wall) and then checked, in this order:

  form    only legend symbols; exactly one start; at most one exit; enclosed by wall; no larger
          than the tier's size range; every door in a one-cell gap of a wall; compiles with
          `udmf.build_wad`; solvable with `solver.solve` (every open cell reachable, every key
          reachable before its door)
  tier    floor area of the tier; an exit in tiers 2 to 4; no locks in tiers 1 and 2, one in tier 3,
          two colours in tier 4, and locks that cannot be bypassed; exit far enough from the
          start; more than one plain room; monsters within the tier's range
  time    the tick cap is chosen by `layouts.tick_cap`, which refuses a route too long for 4200 tics
  copies  a map is dropped if another accepted map, a map of the frozen suites, a map of a file
          named with --exclude (the training manifest) or an example of the prompt has the same
          walls, doors, start and exit up to rotation and mirror (things and lock colours ignored)
  engine  `cli.verify_map`: the map loads in `doom_env`, random play raises no warning and, on a
          copy without monsters, the reference walker reaches the exit within 60% of the tick cap
          (skipped with --no-engine-check)

Every rejection is counted by reason and the counts are printed at the end.

Suite rows
----------
`name, tier, seed, episode_ticks, style, exit_style, grid, options, counts` as in the frozen suites
(so `training.evaluate` and `training.mapgen.cli --stats / --verify / --from-suite` read the file),
plus `source`, `model`, `temperature`, `brief`, `hash` (SHA-1 of the grid) and `layout` (the
near-duplicate key). `style` is the name of the plan style asked for, `options` is empty (the
reference dialect), and `seed` is 32 bits of the grid hash: an authored map has no generator seed.
An existing suite file is never overwritten; `--resume` adds maps to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import cli, layouts, udmf
from .solver import solve

ENV_KEY, ENV_URL = "CREO_API_KEY", "CREO_BASE_URL"     # the only places the key and the base URL come from
DEFAULT_MODEL = "@cf/openai/gpt-oss-120b"
DEFAULT_OUT = cli.SUITE_DIR / "authored_v1.jsonl"
CLIENT_NAME = "ascii-map-client/1.0"                   # sent as User-Agent
ERROR_CHARS = 200            # most characters of a response body that are ever shown
REPLY_BYTES = 4 << 20        # most bytes read from one response
MODEL_STRIKES = 3            # failed requests in a row after which a model is no longer asked
SIZE_SLACK = 4               # cells by which a side may exceed the tier's size range
WALKER_SHARE = 0.6           # share of the tick cap the reference walker may use (the bound of the generator's tests)
THROTTLE_WAVES = 5           # waves in a row that may be answered with HTTP 429 before the run stops
WALK_CELLS, WALK_DOORS = 70, 7    # route length the prompt asks for: within the largest tick cap at walking speed
WALL = udmf.WALL
TIERS = (1, 2, 3, 4)

# What each tier tests: the rows of the "Hidden map suite" table in docs/EVALUATION.md.
TIER_TESTS = {
    1: ("Combat & Survival", "Aiming, dodging projectiles, managing health/ammo"),
    2: ("Multi-Room Exploration", "Partial observability, spatial memory, avoiding dead-end traps, doors"),
    3: ("Long-Horizon Progression", "Finding keycards, remembering locked doors, backtracking to the exit"),
    4: ("Full Gauntlet", "All of the above with ambushes, scarce resources and multiple locks"),
}
MIN_LOCKS = {1: 0, 2: 0, 3: 1, 4: 2}             # locked doors a tier needs, as in layouts.draft
MAX_MONSTERS = {1: 12, 2: 6, 3: 8, 4: 16}        # the most layouts.place_monsters puts on a map of the tier

# What a map of each tier contains, in the terms of layouts.TIERS (the numbers are filled in from there).
TIER_CONTENT = {
    1: "An arena fight: one room or a few open rooms with pillars or wall stubs for cover, {monsters} monsters of "
       "mixed kinds placed away from P, and a little health and ammunition. No keys and no locked doors. "
       "At least {min_open} cells that are not wall.",
    2: "Several rooms joined by corridors, with ordinary doors (D), branches and dead ends, so that E cannot be seen "
       "from P and the way has to be searched for. Light opposition: {monsters} monsters, mostly z and i. No keys and "
       "no locked doors. At least {min_open} cells that are not wall, and E at least {min_route} cells of walking from P.",
    3: "One or two locked doors on the only way to E, each with its key somewhere else on the map, so that the player "
       "has to find the key, remember the door and come back. There is no way around a locked door: without the "
       "key E cannot be reached. {monsters} monsters and a few pickups. At least {min_open} cells that are not wall, "
       "and a walk of at least {min_route} cells from P to E.",
    4: "Two or three locked doors of different colours that have to be opened one after another on the way to E, "
       "monsters waiting right behind doors and around corners ({monsters} in total, some of them strong), and no "
       "more than three pickups. There is no way around a locked door: without the keys E cannot be reached. "
       "At least {min_open} cells that are not wall, and a walk of at least {min_route} cells from P to E.",
}

SYSTEM = ("You design levels for a first-person shooter whose maps are drawn as ASCII grids. "
          "You follow the format rules exactly, because a map that breaks one of them is thrown away.")

RULES = "\n".join((
    "Rules. A map that breaks any of them is thrown away:",
    "1. One character per cell, and only characters from the legend. No spaces, no row numbers, no comments inside a map.",
    "2. The map is a rectangle: every row has the same number of characters.",
    "3. It is fully enclosed: the first row, the last row, the first column and the last column are all '#'.",
    "4. Exactly one P. {exit_rule}",
    "5. The player walks from a cell to the cell above, below, left or right of it, never diagonally. "
    "Every cell that is not '#' must be reachable from P that way.",
    "6. A locked door (R, B, Y) can only be passed with a key of its colour (r or 1 opens R, b or 2 opens B, "
    "y or 3 opens Y), so every key must be reachable before the door it opens.",
    "7. A door (D, R, B, Y) fills a one-cell gap in a wall: '#' on two opposite sides of it and floor on the other two. "
    "A door never touches another door or the exit pad.",
    "8. A cell is 64 map units wide and the player needs one full cell, so every corridor is at least one cell wide. "
    "There are no thinner gaps.",
    "9. The level ends as soon as the player steps on E, so nothing may lie behind E.",
    "10. Size: {lo} to {hi} cells on each side, the outer wall included. The complete walk from P past every key that "
    "is needed to E, backtracking included, is at most about {walk} cells and passes through doors at most {doors} times.",
))

# Two tiny examples of the format, drawn by hand for the prompt (never taken from the generator).
EXAMPLES = (
    "#########\n"
    "#P..#...#\n"
    "#...D.E.#\n"
    "#...#...#\n"
    "#########",
    "#############\n"
    "#P....#.....#\n"
    "#.....R..z..#\n"
    "###.###....E#\n"
    "#.....#######\n"
    "#..r..#######\n"
    "#############",
)

# Plan styles: (name, tiers it suits, what the designer is asked for).
STYLES = (
    ("hub_and_spokes", (2, 3, 4),
     "a central hub room with four or five corridors leading out of it to separate rooms, like the spokes of a wheel"),
    ("spiral", (2, 3, 4),
     "a spiral: one corridor winds inwards around the map to a room in the middle, with a few niches and side rooms off it"),
    ("symmetric_wings", (1, 2, 3, 4), "a symmetric plan: a central hall with matching left and right wings"),
    ("corridor_with_side_rooms", (2, 3, 4),
     "one long corridor, straight or with a single bend, with rooms opening off both of its sides"),
    ("loop_with_shortcuts", (1, 2, 3, 4),
     "a ring corridor around a solid core, with one or two shortcuts cutting across and rooms hanging off the ring"),
    ("arena_with_pillars", (1,), "one big arena with pillars for cover, in a regular pattern or scattered"),
    ("courtyard", (1, 2), "an open courtyard with a ring of small rooms or niches around it"),
    ("twin_arenas", (1,), "two arenas of different shape joined by a short narrow passage"),
    ("cross_hall", (1, 2), "a plus-shaped hall with an alcove or a small room at the end of every arm"),
    ("corridor_maze", (2, 3, 4), "a maze of corridors one cell wide with several dead ends and two or three small rooms"),
    ("room_block", (2, 3, 4),
     "a block of square rooms that share walls, three by three or so, joined through some of the walls while other "
     "walls stay solid"),
    ("zigzag", (2, 3, 4), "rooms stacked in rows and joined at alternating ends, so that the way through snakes back and forth"),
    ("fortress", (3, 4), "a fortress: an outer ring of rooms and corridors around an inner keep that holds the exit"),
    ("caves", (1, 2, 3), "cave-like rooms with ragged walls, joined by short crooked passages"),
    ("two_districts", (3, 4),
     "two districts of different character, one of small rooms and one of wide halls, joined by a single passage"),
)
EXIT_SPOTS = ("in an alcove off a room", "at the far end of a corridor", "in the middle of an open room",
              "in a corner of the last room", "against a wall of a large room")
TWISTS = {
    1: ("Mostly enemies that shoot (z, s, c).", "Mostly enemies that charge (d, g, l).",
        "Enemies that throw fireballs (i, o, k) among weaker ones.", "A mix of weak and strong enemies.",
        "Include a shotgun or a chaingun and some ammunition.", "Very little cover.", "Plenty of cover."),
    2: ("Use many ordinary doors.", "Use no doors at all, only openings.", "Add several dead-end corridors.",
        "Give one room pillars.", "Make two different ways lead to the exit.", "Keep every corridor one cell wide.",
        "Use corridors two cells wide."),
    3: ("Let the player pass the locked door before finding its key.", "Put the key at the end of a dead-end branch.",
        "Use a skull key (a digit) instead of a keycard.", "Use two locked doors, the second key behind the first door.",
        "Put the key behind an ordinary door.", "Guard the key with a monster."),
    4: ("Use three locked doors of three colours, opened one after another.",
        "Use two locked doors, the second key behind the first door.",
        "Place monsters just inside doors and around corners.", "Put only one or two small pickups on the whole map.",
        "Let the player see a locked door long before its key.", "Guard every key with a monster."),
}


# ---- prompt -----------------------------------------------------------------------------------------
def legend_text() -> str:
    """The generator's legend as prompt text, one symbol per line, grouped by what the symbol is."""
    groups = (("Structure", lambda s: s["kind"] in ("wall", "floor", "start", "exit")),
              ("Doors", lambda s: s["kind"] == "door"),
              ("Keys", lambda s: s["group"] == "key"),
              ("Monsters", lambda s: s["group"] == "monster"),
              ("Pickups", lambda s: s["group"] in ("health", "armor", "ammo", "weapon")))
    table = udmf.legend_table()
    out = []
    for title, wanted in groups:
        out.append(f"{title}:")
        out += [f"  {s['char']}  {s['name']}" for s in table if wanted(s)]
    return "\n".join(out)


def tier_text(tier: int) -> str:
    """What the tier tests (docs/EVALUATION.md) and what a map of it contains (layouts.TIERS)."""
    spec = layouts.TIERS[tier]
    name, skills = TIER_TESTS[tier]
    content = TIER_CONTENT[tier].format(monsters=f"{layouts.MIN_MONSTERS[tier]} to {MAX_MONSTERS[tier]}",
                                        min_open=spec["min_open"], min_route=spec["min_route"])
    return f"What tier {tier} tests: {name}. {skills}.\nWhat a tier {tier} map contains: {content}"


def build_messages(tier: int, brief: str, count: int, with_exit: bool = True) -> list[dict]:
    """The chat messages of one request: legend, rules, tier, design brief, two examples, answer format."""
    lo, hi = layouts.TIERS[tier]["size"]
    exit_rule = "Exactly one E." if with_exit else "No E: these maps have no exit pad."
    examples = "\n".join(f"```\n{grid}\n```" for grid in EXAMPLES)
    user = "\n\n".join((
        f"Draw {count} different maps for tier {tier} of a first-person shooter. A map is an ASCII grid.",
        "Legend, one character per square cell:\n" + legend_text(),
        RULES.format(exit_rule=exit_rule, lo=lo, hi=hi, walk=WALK_CELLS, doors=WALK_DOORS),
        tier_text(tier),
        f"Design brief for this request: {brief}",
        "Two tiny examples of the format. Do not copy them: your maps must be larger and follow the brief.\n" + examples,
        "Before you answer, check every map: all rows equally long, the outer wall closed, one P, every cell that is "
        "not wall reachable from P, every door between two walls, every key reachable before its door.",
        f"Answer with {count} maps. Put each map in its own fenced code block (a line of three backticks before and "
        f"after it) with nothing but the rows of the map inside. Make the {count} maps clearly different from each "
        "other. No explanations.",
    ))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass
class Request:
    """One chat request. Only model, messages, temperature and max_tokens are sent."""

    index: int                 # 1-based number of the request in this run
    tier: int
    model: str
    temperature: float
    style: str                 # name of the plan style
    brief: str                 # the design brief as given to the model
    messages: list = field(repr=False, default_factory=list)
    max_tokens: int = 8000


def style_order(tier: int, seed: int) -> list[tuple[str, str]]:
    """The plan styles that suit a tier, in the order this seed uses them."""
    styles = [(name, text) for name, tiers, text in STYLES if tier in tiers]
    random.Random(f"authored-styles-v1/{seed}/{tier}").shuffle(styles)
    return styles


def make_request(index: int, tier: int, turn: int, model: str, temperature: float, *, seed: int = 0,
                 count: int = 5, max_tokens: int = 8000) -> Request:
    """Request number `turn` (0-based) for a tier. The brief is a pure function of (seed, tier, turn)."""
    rng = random.Random(f"authored-brief-v1/{seed}/{tier}/{turn}")
    spec = layouts.TIERS[tier]
    lo, hi = spec["size"]
    styles = style_order(tier, seed)
    style, text = styles[turn % len(styles)]
    # The small half of the tier's range (the larger the grid, the more often a drawn row comes out
    # a cell short), but tall enough for the tier's floor area when half of the interior is open.
    width = rng.randint(lo + 2, min(hi - 4, lo + 6))
    low = min(hi, max(lo, -(-2 * spec["min_open"] // (width - 2)) + 2))
    height = rng.randint(low, min(max(low, width), low + 4))
    if rng.random() < 0.3:
        width, height = height, width
    with_exit = tier > 1 or turn % 2 == 0          # about half of the arenas have an exit
    parts = [f"{text[0].upper()}{text[1:]}.", f"Aim for about {width} columns by {height} rows."]
    if with_exit:
        parts.append(f"Put the exit {rng.choice(EXIT_SPOTS)}.")
    parts += rng.sample(TWISTS[tier], 2)
    brief = " ".join(parts)
    return Request(index=index, tier=tier, model=model, temperature=temperature, style=style, brief=brief,
                   messages=build_messages(tier, brief, count, with_exit), max_tokens=max_tokens)


# ---- reading replies --------------------------------------------------------------------------------
_FENCE = re.compile(r"(`{3,}|~{3,})(.*)$")
_THINKING = re.compile(r"<think>.*?(</think>|\Z)", re.S)      # reasoning some servers leave in the content


def _is_row(line: str) -> bool:
    """Is this line of a fenced block a row of a map? Every row of an enclosed map starts and ends
    with '#', so a line without '#' is not one (blank, a caption, a rule). A line with '#' is taken
    as a caption only if it is written in words: several blank-separated pieces, one of them longer
    than a character and not made of legend symbols ("# Map 2", "Map #2: spiral")."""
    words = line.split()
    return WALL in line and not (len(words) > 1 and any(len(w) > 1 and set(w) - set(udmf.LEGEND) for w in words))


def _maps_of_block(lines: list[str]) -> list[list[str]]:
    """Split the lines of one fenced block into maps: a line that is not a row ends the map."""
    maps, rows = [], []
    for line in lines:
        if _is_row(line):
            rows.append(line)
        elif rows:
            maps.append(rows)
            rows = []
    return maps + ([rows] if rows else [])


def split_reply(text: str) -> list[list[str]]:
    """The candidate maps of a reply, as lists of raw lines.

    Maps are taken from fenced code blocks (three or more backticks or tildes, with or without a
    language tag). A block the reply never closes still counts, and so does one that is ended by
    the opening fence of the next block. Text outside the blocks is ignored, and so is anything
    between <think> tags (drafts are not answers).
    """
    maps: list[list[str]] = []
    block, fence = None, ""
    for line in _THINKING.sub("", str(text)).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        mark = _FENCE.match(line.strip())
        if block is None:
            if mark:
                block, fence = [], mark.group(1)[0]
        elif mark and mark.group(1)[0] == fence:
            maps += _maps_of_block(block)
            block = [] if mark.group(2).strip() else None
        else:
            block.append(line)
    return maps + (_maps_of_block(block) if block else [])


def tidy(lines: list[str]) -> tuple[list[str], bool]:
    """Raw lines of one map -> rows of equal width, and whether short rows had to be padded.

    Trailing blanks and common indentation are removed; if every row has one blank between its
    symbols ("# . P #") the blanks are dropped; blanks left of the outline and the missing ends of
    short rows become wall, because nothing lies outside the outline of a map.
    """
    rows = [line.expandtabs(4).rstrip() for line in lines]
    indent = min(len(row) - len(row.lstrip(" ")) for row in rows)
    rows = [row[indent:] for row in rows]
    if any(len(row) > 1 for row in rows) and all(row[1::2].strip(" ") == "" and " " not in row[::2] for row in rows):
        rows = [row[::2] for row in rows]
    rows = [WALL * (len(row) - len(row.lstrip(" "))) + row.lstrip(" ") for row in rows]
    width = max(len(row) for row in rows)
    return [row.ljust(width, WALL) for row in rows], any(len(row) < width for row in rows)


# ---- checks -----------------------------------------------------------------------------------------
class Rejected(Exception):
    """A returned map failed a check. `reason` is the text counted in the summary."""

    def __init__(self, reason: str, symbols=()):
        super().__init__(reason)
        self.reason, self.symbols = reason, tuple(symbols)


def is_doorway(g: layouts.Grid, c: int, r: int) -> bool:
    """A door cell fills a one-cell gap in a wall: wall on two opposite sides, and on the other two
    a room cell (floor, a thing or the start; not another door and not the exit pad)."""
    return ((g.at(c - 1, r) == WALL and g.at(c + 1, r) == WALL and g.is_room_cell(c, r - 1) and g.is_room_cell(c, r + 1))
            or (g.at(c, r - 1) == WALL and g.at(c, r + 1) == WALL and g.is_room_cell(c - 1, r) and g.is_room_cell(c + 1, r)))


def doors_met_before_key(g: layouts.Grid, start) -> int:
    """Locked doors the player comes to before the nearest key of their colour, by walking distance
    from the start with that door shut and every other door open."""
    met = 0
    for door in g.cells("".join(sorted(udmf.LOCKED_DOOR_CHARS))):
        colour = udmf.LEGEND[g.at(*door)].color
        dist = g.dist(start, blocked={door})
        doorstep = [dist[n] for n in g.around(*door) if n in dist]
        keys = [d for cell, d in dist.items() if g.at(*cell) in udmf.KEY_CHARS and udmf.LEGEND[g.at(*cell)].color == colour]
        met += bool(doorstep and keys and min(keys) > min(doorstep) + 1)
    return met


def layout_key(grid: str) -> str:
    """Hash that is the same for near-duplicates.

    Two maps share a key when their walls, doors, start and exit coincide after some rotation or
    mirror. Things (monsters, keys, pickups) are left out and a locked door counts as one symbol
    whatever its colour, so "the same grid rotated or mirrored" and "the same grid with other
    things on it" both give the same key.
    """
    def plain(ch: str) -> str:
        sym = udmf.LEGEND[ch]
        if sym.kind == "door":
            return "L" if sym.lock else "D"
        return "." if sym.kind in ("floor", "thing") else ch
    rows = ["".join(plain(ch) for ch in row) for row in udmf.parse_grid(grid)]
    forms = []
    for _ in range(4):
        rows = ["".join(col) for col in zip(*rows[::-1])]          # quarter turn
        forms += ["\n".join(rows), "\n".join(row[::-1] for row in rows)]
    return hashlib.sha1(min(forms).encode()).hexdigest()


def examine(lines: list[str], tier: int, seed: int = 0) -> dict:
    """Run one returned map through every check of the module docstring except the search for copies.

    Returns {"grid", "episode_ticks", "counts", "hash", "layout", "padded"}; raises Rejected with
    the reason. The grid is cropped to its open cells plus one cell of wall, as generated maps are.
    """
    rows, padded = tidy(lines)
    # ---- form
    unknown = sorted({ch for row in rows for ch in row} - set(udmf.LEGEND))
    if unknown:
        raise Rejected("unknown symbol", unknown)
    starts = sum(row.count("P") for row in rows)
    if starts != 1:
        raise Rejected("more than one player start" if starts else "no player start")
    if sum(row.count("E") for row in rows) > 1:
        raise Rejected("more than one exit")
    if set(rows[0] + rows[-1]) != {WALL} or any(row[0] != WALL or row[-1] != WALL for row in rows):
        raise Rejected("not enclosed by walls")
    g = layouts.Grid(len(rows[0]), len(rows))
    g.rows = [list(row) for row in rows]
    g = g.trimmed()
    spec = layouts.TIERS[tier]
    if max(g.w, g.h) > spec["size"][1] + SIZE_SLACK:       # checked before the grid is compiled and searched
        raise Rejected("larger than the size range")
    if any(not is_doorway(g, c, r) for c, r in g.cells("".join(sorted(udmf.DOOR_CHARS)))):
        raise Rejected("door not in a doorway")
    grid = g.text()
    try:
        udmf.build_wad(grid)
    except ValueError:
        raise Rejected("does not compile") from None
    sol = solve(grid)
    start = g.cells("P")[0]
    if not sol.all_reachable:
        locks_only = len(g.dist(start)) == len(g.open_cells())      # everything reachable once locks are ignored
        raise Rejected("key not reachable before its door" if locks_only else "unreachable cells")
    # ---- tier
    counts = layouts.describe(grid, {})
    if counts["open_cells"] < spec["min_open"]:
        raise Rejected("too little floor")
    if tier >= 2 and not sol.has_exit:
        raise Rejected("no exit")
    colours = {udmf.LEGEND[ch].color for ch in grid if ch in udmf.LOCKED_DOOR_CHARS}
    if MIN_LOCKS[tier] == 0 and (counts["locks"] or counts["keys"]):
        raise Rejected("keys or locked doors in a tier without them")
    if counts["locks"] < MIN_LOCKS[tier] or len(colours) < min(2, MIN_LOCKS[tier]):
        raise Rejected("too few locked doors")
    if counts["locks"] and not sol.keys_needed:
        raise Rejected("locks can be bypassed")
    if sol.has_exit and sol.path_cells < spec["min_route"]:
        raise Rejected("exit too close to the start")
    if tier >= 2 and counts["rooms"] + counts["doors"] + counts["dead_ends"] < 3:
        raise Rejected("one plain room is not an exploration map")
    if counts["monsters"] < layouts.MIN_MONSTERS[tier]:
        raise Rejected("too few monsters")
    if counts["monsters"] > MAX_MONSTERS[tier]:
        raise Rejected("too many monsters")
    # ---- time
    digest = cli.grid_hash(grid)
    try:
        ticks = layouts.tick_cap(random.Random(f"authored-ticks-v1/{seed}/{digest}"), tier, sol)
    except layouts.Reject:
        raise Rejected("route too long for the largest tick cap") from None
    counts.update(route_cells=sol.path_cells, route_corners=sol.corners, route_doors=sol.doors_on_path,
                  optimal_steps=sol.optimal_steps, doors_met_before_key=doors_met_before_key(g, start))
    return {"grid": grid, "episode_ticks": ticks, "counts": counts, "hash": digest, "layout": layout_key(grid),
            "padded": padded}


def engine_check(row: dict) -> None:
    """The generator's engine test for one suite row (`cli.verify_map`). Raises Rejected if the
    map does not load in `doom_env` with the sectors the grid has, if 25 random steps raise a
    warning, or if the reference walker, on a copy without monsters, does not reach the exit
    within WALKER_SHARE of the tick cap. An error is tried a second time before it counts: on a
    busy machine the engine now and then fails to start."""
    try:
        result = cli.verify_map(row)
    except Exception:
        try:
            result = cli.verify_map(row)
        except Exception:
            raise Rejected("fails in the engine") from None
    if result["reached"] is False:
        raise Rejected("reference walker does not reach the exit")
    if result["cap_used"] is not None and result["cap_used"] > WALKER_SHARE:
        raise Rejected("reference walker needs too much of the tick cap")


def suite_row(found: dict, name: str, request: Request) -> dict:
    """The suite line of an accepted map: the frozen-suite keys, then where the map came from."""
    row = dict.fromkeys(cli.SUITE_KEYS)          # the keys of the frozen suites, in their order
    row.update(name=name, tier=request.tier, seed=int(found["hash"][:8], 16), episode_ticks=found["episode_ticks"],
               style=request.style, exit_style=None, grid=found["grid"], options={}, counts=found["counts"])
    row.update(source="authored", model=request.model, temperature=request.temperature, brief=request.brief,
               hash=found["hash"], layout=found["layout"])
    return row


def known_layouts(paths) -> set[str]:
    """Near-duplicate keys of every grid in the given suite or manifest files."""
    keys = set()
    for path in paths:
        for row in cli.read_rows(path):
            grid = row.get("grid")
            if isinstance(grid, list):
                grid = "\n".join(grid)
            try:
                keys.add(layout_key(grid))
            except (ValueError, AttributeError, TypeError):      # no grid, or not a grid of this legend: nothing to compare
                continue
    return keys


# ---- clients ----------------------------------------------------------------------------------------
@dataclass
class Reply:
    """What came back for one request. `error` is ready to print: cut to ERROR_CHARS, values hidden."""

    text: str = ""
    error: str = ""
    status: int = 0                # HTTP status, 0 when there was no response
    cut_off: bool = False          # the model stopped at max_tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    exhausted: bool = False        # dry run only: no canned reply is left for this tier


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is reported as an error, so the bearer key is never sent on to another address."""

    def redirect_request(self, *args, **kwargs):
        return None


class HttpClient:
    """Chat completions over HTTP(S) with the standard library.

    The key and the base URL are read from the environment here. They are held privately and are
    never printed, logged or written: every piece of text that leaves this class passes `_show`.
    """

    def __init__(self, timeout: float = 240.0):
        values = {name: os.environ.get(name, "").strip() for name in (ENV_KEY, ENV_URL)}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise SystemExit(f"{' and '.join(missing)} must be set in the environment "
                             "(or use --dry-run, which needs neither)")
        if not values[ENV_URL].lower().startswith(("https://", "http://")):
            raise SystemExit(f"{ENV_URL} must be an http:// or https:// address")
        for name, value in values.items():      # a blank or control character would end up in a header or request line
            if not re.fullmatch(r"[\x21-\x7e]+", value):
                raise SystemExit(f"{name} must be printable ASCII without blanks")
        base = re.sub(r"/chat/completions$", "", values[ENV_URL].rstrip("/")).rstrip("/")
        self._key, self._base, self.timeout = values[ENV_KEY], base, float(timeout)
        parts = urllib.parse.urlsplit(self._base)
        self._hidden = sorted({v for v in (self._key, self._base, parts.netloc, parts.hostname or "") if len(v) >= 4},
                              key=len, reverse=True)
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:
        return "HttpClient()"

    def _show(self, text) -> str:
        """Text that may be printed: the key and the address blanked, one line, at most ERROR_CHARS."""
        text = str(text)
        for value in self._hidden:
            text = text.replace(value, "[hidden]")
        return " ".join(text.split())[:ERROR_CHARS]

    def _call(self, path: str, body: dict | None = None) -> tuple[int, bytes, str]:
        """(status, response bytes, error). GET without a body, POST with a JSON body. Never raises."""
        try:
            request = urllib.request.Request(self._base + path, data=None if body is None else json.dumps(body).encode(),
                                             method="GET" if body is None else "POST")
            request.add_header("Authorization", "Bearer " + self._key)
            request.add_header("Accept", "application/json")
            request.add_header("User-Agent", CLIENT_NAME)
            if body is not None:
                request.add_header("Content-Type", "application/json")
            with self._opener.open(request, timeout=self.timeout) as response:
                return int(getattr(response, "status", 200) or 200), response.read(REPLY_BYTES), ""
        except urllib.error.HTTPError as e:
            try:
                detail = e.read(65536).decode("utf-8", "replace")
            except Exception:
                detail = ""
            return int(e.code), b"", f"HTTP {int(e.code)}: {self._show(detail)}".rstrip(": ")
        except urllib.error.URLError as e:     # no route, refused, TLS, a timeout while connecting
            return 0, b"", f"no response: {self._show(e.reason)}".rstrip(": ")
        except OSError as e:                   # a timeout or a dropped connection while reading
            return 0, b"", f"no response: {type(e).__name__} {self._show(e)}".rstrip(": ")
        except Exception as e:                 # anything else: its message could quote the request, so only its name
            return 0, b"", f"no response: {type(e).__name__}"

    def chat(self, request: Request) -> Reply:
        """POST one chat completion and return the text of the first choice."""
        status, raw, error = self._call("/chat/completions", {
            "model": request.model, "messages": request.messages,
            "temperature": request.temperature, "max_tokens": request.max_tokens})
        if error:
            return Reply(status=status, error=error)
        try:
            payload = json.loads(raw)
        except ValueError:
            return Reply(status=status, error="the response is not JSON")
        try:
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
            if isinstance(content, list):          # some servers send the text in parts
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            return Reply(status=status, text=str(content), cut_off=choice.get("finish_reason") == "length",
                         prompt_tokens=int(usage.get("prompt_tokens") or 0),
                         completion_tokens=int(usage.get("completion_tokens") or 0))
        except (LookupError, TypeError, AttributeError, ValueError):
            inner = payload.get("error") if isinstance(payload, dict) else None
            return Reply(status=status, error="the response is not a chat completion"
                         + (f": {self._show(json.dumps(inner))}" if inner else ""))

    def models(self) -> list[str]:
        """Ids listed by GET {base}/models."""
        status, raw, error = self._call("/models")
        if error:
            raise SystemExit(f"could not list the models: {error}")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise SystemExit("could not list the models: the response is not JSON") from None
        items = payload.get("data", payload.get("models", payload.get("result", []))) if isinstance(payload, dict) else payload
        ids = [str(item.get("id") or item.get("name") or "") if isinstance(item, dict) else str(item)
               for item in (items if isinstance(items, list) else [])]
        return [i for i in ids if i]


# Hand-drawn maps for the dry run: two per tier that pass every check ...
_PILLAR_HALL = """
#############
#P....#....z#
#.....#.....#
#..#.....#..#
#.....i.....#
#..#.....#..#
#.....#....h#
#z....#....E#
#############"""
_CROSS_PIT = """
###########
####...####
####.s.####
#....#....#
#.i..P..z.#
#....#....#
####.n.####
####...####
###########"""
_SIDE_ROOMS = """
###################
#...#...#...#.....#
#.P.#.h.#...#..z..#
#...#...#...#.....#
##D###D###.###D####
#.................#
####D####.####D####
#.....#...#.......#
#..i..#...#...E...#
#.....#...#.......#
###################"""
_RING = """
###############
#P............#
#.#####.#####.#
#.#...#.#...#.#
#.#.h.D.#.z.#.#
#.#...#.#...#.#
#.#####.##D##.#
#.............#
#.###.#####.###
#.#...#...#...#
#.#.#.#.E.#.#.#
#...#.....#.#.#
###############"""
_RED_ERRAND = """
#################
#P....#.........#
#.....R....z....#
#.....#.......E.#
###D###.........#
#.....###########
#.....#...#.....#
#..h..D...D..i..#
#.....#...#....r#
#.....#...#.....#
#################"""
_BLUE_WINGS = """
###################
#.....#.....#.....#
#..s..#..P..#..E..#
#.....#.....#.....#
###D#####.#####B###
#.................#
#.###.#######.###.#
#.#...#.....#...#.#
#.#.2.#..z..#.h.#.#
#.#...D.....D...#.#
#.....#.....#.....#
###################"""
_TWO_KEYS = """
#####################
#P....#.....#.......#
#.....#..i..#...c...#
#.....R.....#.......#
#.....#.....Y....E..#
###D###..z..#.......#
#.....#.....#########
#..z..###D###.......#
#.....#.....#...s...#
#..r..#..d..D....3..#
#.....#.....#.......#
#####################"""
_FOUR_BANDS = """
#################
#...#...........#
#.b.D....P...z..#
#...#...........#
###############B#
#...............#
#..s.....#...i..#
#........#......#
#D###############
#.....#.........#
#..c..D....y....#
#.....#.........#
###############Y#
#...............#
#.E......d....n.#
#...............#
#################"""
HAND_DRAWN = {1: (_PILLAR_HALL, _CROSS_PIT), 2: (_SIDE_ROOMS, _RING), 3: (_RED_ERRAND, _BLUE_WINGS),
              4: (_TWO_KEYS, _FOUR_BANDS)}
# ... and maps that each break one rule.
_ODD_SYMBOLS = """
###########
#P...X...z#
#..~~~....#
#...i....E#
###########"""
_SEALED_EXIT = """
###############
#P....#.......#
#.....#...E...#
#..z..#.......#
#.....#########
#.............#
###############"""
_DOOR_IN_FLOOR = """
#############
#P..........#
#.....D.....#
#..........E#
#############"""
_KEY_BEHIND_ITS_DOOR = """
###############
#P....#.......#
#.....R..r....#
#..z..#.....E.#
###############"""
_TWO_STARTS = """
#############
#P....#....P#
#.....D.....#
#..z..#...E.#
#############"""
_OPEN_EDGE = """
#############
#P....#.....#
#.....D..z..#
#.....#....E.
#############"""
_TINY_LOCK = """
###########
#P..R....E#
#.r.#..z..#
###########"""


def _fenced(grid: str, tag: str = "") -> str:
    return f"```{tag}\n{grid.strip()}\n```"


def canned_replies() -> dict[int, list[str]]:
    """Replies of the dry run, per tier, written the way chat models answer: captions between the
    blocks, language tags, and the usual slips (short rows, blanks between symbols, two maps in
    one block, an answer in prose)."""
    def rows(grid: str) -> list[str]:
        return grid.strip().split("\n")

    ring = rows(_RING)
    ragged = "\n".join([ring[0]] + [row.rstrip(WALL) for row in ring[1:-1]] + [ring[-1]])     # right-hand wall left off
    mirrored = "\n".join(row[::-1] for row in rows(_SIDE_ROOMS)).replace("z", ".").replace("i", "s").replace("h", "m")
    spaced = "\n".join(" ".join(row) for row in rows(_BLUE_WINGS))
    bands = [list(row) for row in rows(_FOUR_BANDS)]
    bands[4][1] = bands[12][1] = "."                                                           # a way round both locks
    bypass = "\n".join("".join(row) for row in bands)
    return {
        1: ["Here are two arenas.\n\nMap 1: pillared hall\n" + _fenced(_PILLAR_HALL, "text")
            + "\n\nMap 2: acid hall\n" + _fenced(_ODD_SYMBOLS),
            "I would rather describe the next arenas in words. The first is a round hall with four pillars and a "
            "ring of zombiemen; the second is a long hall split by a low wall, with imps on the far side.",
            "One more, without an exit:\n" + _fenced(_CROSS_PIT)],
        2: ["Map A (corridor with side rooms)\n" + _fenced(_SIDE_ROOMS) + "\nMap B (sealed vault)\n"
            + _fenced(_SEALED_EXIT) + "\nMap C (free-standing door)\n" + _fenced(_DOOR_IN_FLOOR),
            _fenced(mirrored, "ascii") + "\n\n" + _fenced(ragged, "ascii")],
        3: [_fenced(_RED_ERRAND) + "\n\n" + _fenced(_KEY_BEHIND_ITS_DOOR),
            "Second map, drawn with spacing for readability:\n~~~\n" + spaced + "\n~~~"],
        4: [_fenced(_TWO_KEYS, "text") + "\n" + _fenced(_TWO_STARTS, "text") + "\n" + _fenced(_OPEN_EDGE, "text"),
            "```\n" + _FOUR_BANDS.strip() + "\n\nbonus, a small one:\n" + _TINY_LOCK.strip() + "\n```\n\n"
            "And a variant with shortcuts:\n" + _fenced(bypass)],
    }


class CannedClient:
    """Dry run: answers every request with the next canned reply of its tier. No network."""

    MODELS = ("dry-run-model-a", "dry-run-model-b")

    def __init__(self):
        self._replies = canned_replies()
        self._served: Counter = Counter()

    def chat(self, request: Request) -> Reply:
        replies = self._replies.get(request.tier, [])
        turn = self._served[request.tier]
        self._served[request.tier] += 1
        return Reply(text=replies[turn], status=200) if turn < len(replies) else Reply(exhausted=True)

    def models(self) -> list[str]:
        return list(self.MODELS)


def self_check(engine: bool = True, echo=print) -> None:
    """Try the checks on the hand-drawn maps before any request is sent.

    If the generator, the solver or the engine cannot be used here, the run ends at once instead
    of after the requests have been spent. A hand-drawn map that is merely rejected (the tier's
    requirements have moved since it was drawn) is reported and the run goes on."""
    passed = []
    for tier, grids in HAND_DRAWN.items():
        for grid in grids:
            try:
                passed.append((tier, examine(grid.strip().split("\n"), tier)))
            except Rejected as e:
                echo(f"note: a hand-drawn tier {tier} map no longer passes the checks ({e.reason})")
            except Exception as e:
                raise SystemExit("self-check failed, no request was sent: the checks do not run here "
                                 f"({type(e).__name__}: {e})") from None
    with_exit = [(tier, found) for tier, found in passed if found["counts"]["exits"]]
    if engine and with_exit:
        tier, found = with_exit[-1]
        request = Request(index=0, tier=tier, model="", temperature=0.0, style="", brief="")
        try:
            engine_check(suite_row(found, "self_check", request))
        except Rejected as e:
            raise SystemExit("self-check failed, no request was sent: a hand-drawn map does not pass the engine "
                             f"check here ({e.reason}). --no-engine-check runs without it") from None


# ---- collecting -------------------------------------------------------------------------------------
@dataclass
class Stats:
    """Counts of one run, printed as the summary."""

    target: int = 0
    cap: int = 0
    requests: int = 0
    failed: int = 0                # requests without a usable response
    no_map: int = 0                # replies without a map in a fenced block
    cut_off: int = 0               # replies that stopped at max_tokens
    returned: int = 0              # maps found in the replies
    accepted: int = 0
    surplus: int = 0               # good maps beyond the target of their tier
    padded: int = 0                # accepted maps whose short rows were padded
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stopped: str = ""              # why the run ended early
    rejected: Counter = field(default_factory=Counter)      # reason -> maps
    symbols: Counter = field(default_factory=Counter)       # unknown symbol -> maps that used it
    tiers: dict = field(default_factory=dict)               # tier -> Counter(requests, returned, accepted)
    models: dict = field(default_factory=dict)              # model -> Counter(requests, returned, accepted)
    total: Counter = field(default_factory=Counter)         # tier -> maps in the suite, earlier runs included
    notes: list = field(default_factory=list)

    def count(self, request: Request, what: str, n: int = 1) -> None:
        self.tiers.setdefault(request.tier, Counter())[what] += n
        self.models.setdefault(request.model, Counter())[what] += n

    def as_dict(self) -> dict:
        out = {k: getattr(self, k) for k in ("target", "cap", "requests", "failed", "no_map", "cut_off", "returned",
                                             "accepted", "surplus", "padded", "prompt_tokens", "completion_tokens",
                                             "stopped", "notes")}
        out["rejected"] = dict(self.rejected.most_common())
        out["unknown_symbols"] = dict(self.symbols.most_common())
        out["tiers"] = {str(t): {**dict(c), "total": self.total[t]} for t, c in sorted(self.tiers.items())}
        out["models"] = {m: dict(c) for m, c in self.models.items()}
        return out

    def summary(self) -> str:
        lines = [f"requests made: {self.requests} of at most {self.cap} ({self.failed} failed, "
                 f"{self.no_map} without a map in a fenced block, {self.cut_off} cut off by --max-tokens)"]
        if self.prompt_tokens or self.completion_tokens:
            lines.append(f"tokens reported by the endpoint: {self.prompt_tokens} prompt, {self.completion_tokens} completion")
        lines.append(f"maps returned: {self.returned} | accepted {self.accepted} | rejected {sum(self.rejected.values())}"
                     f" | beyond the tier target {self.surplus} | accepted after padding short rows {self.padded}")
        lines.append(f"{'tier':<6}{'requests':>9}{'returned':>9}{'accepted':>9}{'in suite':>10}")
        for tier in sorted(set(self.tiers) | set(self.total)):
            c = self.tiers.get(tier, Counter())
            share = f"{self.total[tier]}/{self.target}"
            lines.append(f"{tier:<6}{c['requests']:>9}{c['returned']:>9}{c['accepted']:>9}{share:>10}")
        wide = max([5] + [len(m) for m in self.models])
        lines.append(f"{'model':<{wide}}{'requests':>10}{'returned':>9}{'accepted':>9}")
        for model, c in self.models.items():
            lines.append(f"{model:<{wide}}{c['requests']:>10}{c['returned']:>9}{c['accepted']:>9}")
        lines.append("rejected, by reason:" if self.rejected else "rejected: none")
        lines += [f"{n:>6}  {reason}" for reason, n in self.rejected.most_common()]
        if self.symbols:
            lines.append("unknown symbols (maps using them): "
                         + ", ".join(f"{ch!r} {n}" for ch, n in self.symbols.most_common(12)))
        lines += [f"note: {note}" for note in self.notes]
        if self.stopped:
            lines.append(f"stopped early: {self.stopped}")
        return "\n".join(lines)


def _next_index(rows: list[dict], stem: str, tier: int) -> int:
    """First free number for a map name of this tier (names are <stem>_t<tier>_<number>)."""
    pattern = re.compile(rf"{re.escape(stem)}_t{tier}_(\d+)")
    used = [int(m.group(1)) for row in rows if (m := pattern.fullmatch(str(row.get("name", ""))))]
    return max(used, default=-1) + 1


def collect(client, *, tiers=TIERS, per_tier: int = 40, models=(DEFAULT_MODEL,), temperatures=(0.7, 1.0),
            max_requests: int = 150, maps_per_request: int = 5, max_tokens: int = 8000, seed: int = 0,
            patience: int = 8, concurrency: int = 1, stem: str = "authored_v1", rows=(), taken=(),
            verify=None, echo=print, sleep=time.sleep, after_wave=None, keep_reply=None) -> tuple[list[dict], Stats]:
    """Ask for maps until every tier has `per_tier` accepted ones or `max_requests` requests are spent.

    `rows` are the maps already in the suite (kept, counted towards the targets and never repeated)
    and `taken` holds further near-duplicate keys to avoid. Requests go out in waves of up to
    `concurrency`; each request goes to the tier that is furthest from its target (of several, the
    one asked least so far), so the tiers fill evenly and a run that ends at the cap is balanced.
    Replies are handled in the order the requests were planned, so a run is reproducible for given replies.
    A tier is given up after `patience` requests in a row without an accepted map, and a model after
    MODEL_STRIKES failed requests in a row. An HTTP 429 is not held against the model: the run
    waits, goes on one request at a time and stops after THROTTLE_WAVES such waves in a row.
    `verify(row)`, if given, is the last check of a map and raises Rejected (see `engine_check`).
    Returns (all suite rows, statistics).
    """
    rows = list(rows)
    taken = set(taken) | {row.get("layout") or layout_key(row["grid"]) for row in rows}
    taken |= {layout_key(grid) for grid in EXAMPLES}       # an example of the prompt handed back is not a new map
    stats = Stats(target=per_tier, cap=max_requests)
    stats.total.update({t: sum(row["tier"] == t for row in rows) for t in tiers})
    index = {t: _next_index(rows, stem, t) for t in tiers}
    turns: Counter = Counter()         # requests planned per tier in this run: the number of the next brief
    barren: Counter = Counter()        # requests in a row without an accepted map, per tier
    strikes: Counter = Counter()       # failed requests in a row, per model
    closed: dict[int, str] = {}        # tier -> why it gets no more requests
    live = list(models)
    width, throttled_waves = max(1, concurrency), 0

    def wanting() -> list[int]:
        return [t for t in tiers if t not in closed and stats.total[t] < per_tier]

    def expected(tier: int) -> float:
        c = stats.tiers.get(tier, Counter())
        return max(0.25, c["accepted"] / c["requests"]) if c["requests"] >= 2 else max(0.25, 0.3 * maps_per_request)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while stats.requests < max_requests and live and wanting() and not stats.stopped:
            wave: list[Request] = []
            planned: Counter = Counter()
            for _ in range(min(width, max_requests - stats.requests)):
                open_now = [t for t in wanting() if stats.total[t] + planned[t] * expected(t) < per_tier]
                if not open_now:
                    break
                tier = min(open_now, key=lambda t: ((stats.total[t] + planned[t] * expected(t)) / per_tier, turns[t], t))
                turn = turns[tier]
                wave.append(make_request(stats.requests + len(wave) + 1, tier, turn, live[(turn + tier) % len(live)],
                                         temperatures[(turn // len(live)) % len(temperatures)], seed=seed,
                                         count=maps_per_request, max_tokens=max_tokens))
                turns[tier] += 1
                planned[tier] += 1
            replies = list(pool.map(client.chat, wave)) if len(wave) > 1 else [client.chat(wave[0])]
            throttled = False
            for request, reply in zip(wave, replies):
                tier, model = request.tier, request.model
                if reply.exhausted:
                    closed[tier] = "no canned reply left"
                    continue
                stats.requests += 1
                stats.count(request, "requests")
                if keep_reply is not None:
                    keep_reply(request, reply)
                if reply.error:
                    stats.failed += 1
                    echo(f"[{request.index:>3}] tier {tier} {model}: {reply.error}")
                    if reply.status == 429:
                        throttled = True
                        continue
                    strikes[model] += 1
                    if reply.status in (401, 403):
                        stats.stopped = f"the endpoint refused the request (HTTP {reply.status}); check {ENV_KEY} and {ENV_URL}"
                    elif strikes[model] >= MODEL_STRIKES and model in live:
                        live.remove(model)
                        stats.notes.append(f"{model} was dropped after {MODEL_STRIKES} failed requests in a row")
                    continue
                strikes[model] = 0
                stats.prompt_tokens += reply.prompt_tokens
                stats.completion_tokens += reply.completion_tokens
                stats.cut_off += bool(reply.cut_off)
                candidates = split_reply(reply.text)
                stats.no_map += not candidates
                got = 0
                for lines in candidates:
                    stats.returned += 1
                    stats.count(request, "returned")
                    try:
                        found = examine(lines, tier, seed)
                        if found["layout"] in taken:
                            raise Rejected("near-duplicate")
                    except Rejected as e:
                        stats.rejected[e.reason] += 1
                        stats.symbols.update(e.symbols)
                        continue
                    except Exception as e:      # text no check foresaw must not end a run that has requests under way
                        stats.rejected[f"could not be checked ({type(e).__name__})"] += 1
                        continue
                    if stats.total[tier] >= per_tier:
                        stats.surplus += 1
                        continue
                    row = suite_row(found, f"{stem}_t{tier}_{index[tier]:03d}", request)
                    if verify is not None:
                        try:
                            verify(row)
                        except Rejected as e:
                            stats.rejected[e.reason] += 1
                            continue
                    taken.add(found["layout"])
                    rows.append(row)
                    index[tier] += 1
                    got += 1
                    stats.total[tier] += 1
                    stats.accepted += 1
                    stats.padded += bool(found["padded"])
                    stats.count(request, "accepted")
                barren[tier] = 0 if got else barren[tier] + 1
                if barren[tier] >= patience and stats.total[tier] < per_tier:
                    closed.setdefault(tier, f"{patience} requests in a row without an accepted map")
                echo(f"[{request.index:>3}] tier {tier} {model} t={request.temperature:g} {request.style}: "
                     f"{len(candidates)} maps, {got} accepted (tier {tier}: {stats.total[tier]}/{per_tier})")
            if after_wave is not None:
                after_wave(rows)
            throttled_waves = throttled_waves + 1 if throttled else 0
            if throttled_waves >= THROTTLE_WAVES:
                stats.stopped = stats.stopped or "the endpoint kept answering HTTP 429 (too many requests)"
            elif throttled and not stats.stopped:
                width = 1
                sleep(min(120.0, 15.0 * 2 ** (throttled_waves - 1)))
    capped = "the request cap was reached"
    short = {tier: closed.get(tier) or stats.stopped or ("no model left to ask" if not live else capped)
             for tier in tiers if stats.total[tier] < per_tier}
    stats.notes += [f"tier {tier} has {stats.total[tier]} of {per_tier} maps: {why}" for tier, why in short.items()]
    if any(why == capped or "in a row" in why for why in short.values()):
        stats.notes.append("to add the missing maps, run again with --resume and another --seed (other briefs)")
    return rows, stats


# ---- files and command line -------------------------------------------------------------------------
def write_suite(path, rows: list[dict]) -> str:
    """Write the suite (one JSON object per line, tier by tier) and return the SHA-256 of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row) for row in sorted(rows, key=lambda row: (row["tier"], row["name"]))) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return hashlib.sha256(text.encode()).hexdigest()


def _numbers(text: str, kind=int) -> list:
    """Comma-separated numbers of a flag; [] if a piece is not a number."""
    try:
        return [kind(part) for part in str(text).split(",") if part.strip()]
    except ValueError:
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.mapgen.authored", description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=DEFAULT_MODEL, help="comma-separated model ids, asked in turn")
    ap.add_argument("--per-tier", type=int, default=40, metavar="N", help="accepted maps wanted per tier")
    ap.add_argument("--tiers", default="1,2,3,4", help="comma-separated tiers (1 to 4)")
    ap.add_argument("--max-requests", type=int, default=150, help="hard cap on the chat requests of this run")
    ap.add_argument("--out", help="suite file to write (default training/suites/authored_v1.jsonl; "
                                  "a dry run writes a file only when --out is given)")
    ap.add_argument("--dry-run", action="store_true", help="canned replies instead of the network")
    ap.add_argument("--seed", type=int, default=0, help="seed of the design briefs and of the tick caps")
    ap.add_argument("--temperature", default="0.7,1.0", help="comma-separated sampling temperatures, used in turn")
    ap.add_argument("--list-models", action="store_true", help=f"print the model ids of GET {{{ENV_URL}}}/models and exit")
    ap.add_argument("--maps-per-request", type=int, default=5, help="maps asked for in one request")
    ap.add_argument("--max-tokens", type=int, default=8000, help="max_tokens of a request")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds to wait for one response")
    ap.add_argument("--concurrency", type=int, default=3, help="requests sent at the same time")
    ap.add_argument("--patience", type=int, default=8,
                    help="give a tier up after this many requests in a row without an accepted map")
    ap.add_argument("--resume", action="store_true", help="add to an existing suite file instead of refusing to touch it")
    ap.add_argument("--no-engine-check", action="store_true",
                    help="accept maps without loading them in the engine and walking them with the reference walker")
    ap.add_argument("--exclude", action="append", default=[], metavar="FILE",
                    help="another suite or manifest whose layouts must not be repeated (the frozen suites always count)")
    ap.add_argument("--summary-json", metavar="FILE",
                    help="also add the summary of this run to a JSON list in FILE (for training/logs/)")
    ap.add_argument("--save-replies", metavar="DIR", help="keep the text of every reply here (for runs/)")
    args = ap.parse_args(argv)

    tiers = _numbers(args.tiers)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    temperatures = _numbers(args.temperature, float)
    if not tiers or any(t not in TIERS for t in tiers):
        ap.error("--tiers takes tiers 1 to 4")
    if not models or not temperatures:
        ap.error("--models and --temperature need at least one value each")
    if args.per_tier < 1 or args.max_requests < 0 or args.maps_per_request < 1 or args.concurrency < 1:
        ap.error("--per-tier, --maps-per-request and --concurrency must be at least 1 and --max-requests at least 0")

    if args.list_models:
        listed = (CannedClient() if args.dry_run else HttpClient(args.timeout)).models()
        print("\n".join(listed) if listed else "the endpoint listed no model ids")
        return 0 if listed else 1

    out = Path(args.out) if args.out else (None if args.dry_run else DEFAULT_OUT)
    rows: list[dict] = []
    if out is not None and out.exists():
        if not args.resume:
            raise SystemExit(f"{out} exists: a suite is written once. Pass --resume to add maps to it, or choose another --out")
        rows = cli.read_rows(out)
    client = CannedClient() if args.dry_run else HttpClient(args.timeout)
    self_check(engine=not args.no_engine_check)
    others = [p for p in sorted(cli.SUITE_DIR.glob("*.jsonl")) if out is None or p.resolve() != out.resolve()]
    others += [Path(p) for p in args.exclude]
    taken = known_layouts(others)
    print(f"{len(taken)} layouts are already in use and will not be repeated ({', '.join(p.name for p in others) or 'no files'})")

    stamp = datetime.now().strftime("%m%d-%H%M%S")      # one prefix per run: a resumed run keeps the earlier replies

    def keep_reply(request: Request, reply: Reply) -> None:
        folder = Path(args.save_replies)
        folder.mkdir(parents=True, exist_ok=True)
        head = (f"request {request.index} | tier {request.tier} | model {request.model} | temperature {request.temperature:g}"
                f" | style {request.style}\nbrief: {request.brief}\n")
        (folder / f"{stamp}_{request.index:03d}_t{request.tier}.txt").write_text(
            head + "\n" + (reply.text or reply.error) + "\n", encoding="utf-8")

    saved = [len(rows)]

    def save(current: list[dict]) -> None:      # after every wave, so an interrupted run keeps its maps
        if out is not None and len(current) != saved[0]:
            write_suite(out, current)
            saved[0] = len(current)

    if args.dry_run:
        print("dry run: canned replies, no network request is made")
    else:
        print(f"asking {', '.join(models)} for {args.per_tier} maps per tier (tiers {', '.join(map(str, tiers))}): "
              f"at most {args.max_requests} requests, {args.concurrency} at a time, {args.maps_per_request} maps each")
    started = time.perf_counter()
    try:
        rows, stats = collect(
            client, tiers=tiers, per_tier=args.per_tier, models=models, temperatures=temperatures,
            max_requests=args.max_requests, maps_per_request=args.maps_per_request, max_tokens=args.max_tokens,
            seed=args.seed, patience=args.patience, concurrency=1 if args.dry_run else args.concurrency,
            stem=(out or DEFAULT_OUT).stem, rows=rows, taken=taken, after_wave=save,
            verify=None if args.no_engine_check else engine_check,
            keep_reply=keep_reply if args.save_replies else None)
    except KeyboardInterrupt:
        raise SystemExit(f"interrupted; maps accepted so far are in {out}" if out is not None else "interrupted") from None
    print(stats.summary())
    digest = ""
    if out is not None and rows:
        digest = write_suite(out, rows)
        print(f"wrote {len(rows)} maps to {out} (sha256 {digest})")
    elif out is None:
        print("nothing written (a dry run writes a suite only when --out is given)")
    else:
        print(f"no map was accepted: {out} was not written")
    if args.summary_json:
        report = {"finished": datetime.now().astimezone().isoformat(timespec="seconds"), "dry_run": bool(args.dry_run),
                  "suite": out.name if out is not None else None, "suite_sha256": digest, "maps": len(rows),
                  "seconds": round(time.perf_counter() - started, 1), "seed": args.seed, "asked_tiers": tiers,
                  "asked_models": models, "temperatures": temperatures, "maps_per_request": args.maps_per_request,
                  "max_tokens": args.max_tokens, "engine_check": not args.no_engine_check, **stats.as_dict()}
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        earlier = []                                  # the file is a list with one entry per run, oldest first
        if path.is_file():
            try:
                earlier = json.loads(path.read_text())
            except ValueError:
                earlier = []
        path.write_text(json.dumps((earlier if isinstance(earlier, list) else [earlier]) + [report], indent=2) + "\n")
    return 1 if stats.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
