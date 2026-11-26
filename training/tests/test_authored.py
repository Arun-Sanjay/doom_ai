"""Tests for the authored validation maps (training/mapgen/authored.py).

Everything here runs without the network: an autouse fixture makes any attempt to open a URL or a
socket fail the test, and the key and base URL are only ever set to made-up values.
"""

import ast
import email.message
import io
import json
import random
import re
import socket
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path

import pytest

from training.mapgen import authored, cli, layouts, udmf
from training.mapgen.solver import solve

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "training" / "suites"
MODULE = ROOT / "training" / "mapgen" / "authored.py"
KEY = "unit-test-" + "0123456789abcdef" * 2          # made up; stands in for the real key
REAL_OPEN = urllib.request.OpenerDirector.open       # kept so one test can run urllib's own logic on a fake handler
BASE = "https://llm.invalid/api/v1"
GOOD = authored.HAND_DRAWN                            # the hand-drawn maps of the dry run, by tier


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail the test if anything tries to reach the network, even where the code under test turns
    the refusal into an error message. The real key and address are removed from the environment."""
    attempts = []

    def refuse_open(self, *args, **kwargs):
        attempts.append("urllib opener")
        raise AssertionError("the tests must not touch the network")

    def refuse_connect(self, *args, **kwargs):
        attempts.append("socket")
        raise AssertionError("the tests must not touch the network")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refuse_open)
    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.delenv(authored.ENV_KEY, raising=False)
    monkeypatch.delenv(authored.ENV_URL, raising=False)
    yield attempts
    assert not attempts, f"network access attempted through: {attempts}"


# ---- helpers ----------------------------------------------------------------------------------------
def lines(grid: str) -> list[str]:
    return grid.strip("\n").split("\n")


def put(grid: str, *edits) -> str:
    """Copy of a grid with (col, row, symbol) edits applied."""
    rows = [list(row) for row in lines(grid)]
    for c, r, ch in edits:
        rows[r][c] = ch
    return "\n".join("".join(row) for row in rows)


def box(w: int, h: int, *things) -> str:
    """A walled room of w x h cells with (col, row, symbol) things in it."""
    rows = ["#" * w] + ["#" + "." * (w - 2) + "#" for _ in range(h - 2)] + ["#" * w]
    return put("\n".join(rows), *things)


def snake(w: int = 21, h: int = 21) -> str:
    """One corridor that runs back and forth over the whole map, with a door near the start."""
    rows = [["#"] * w for _ in range(h)]
    for i, r in enumerate(range(1, h - 1, 2)):
        rows[r][1:w - 1] = "." * (w - 2)
        if r + 1 < h - 1:
            rows[r + 1][w - 2 if i % 2 == 0 else 1] = "."
    rows[1][1], rows[1][5], rows[h - 2][1] = "P", "D", "E"
    return "\n".join("".join(row) for row in rows)


def arena(k: int) -> str:
    """A valid tier 1 map; every k gives another layout."""
    return box(8 + k, 7, (1, 1, "P"), (5, 4, "z"), (6, 2, "i"))


def fence(*grids: str) -> str:
    return "\n\n".join(f"```\n{grid.strip()}\n```" for grid in grids)


def reason(grid: str, tier: int) -> str:
    with pytest.raises(authored.Rejected) as caught:
        authored.examine(lines(grid), tier)
    return caught.value.reason


class Scripted:
    """Stands in for a chat client: answers from a function and records every request."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def chat(self, request):
        self.seen.append(request)
        out = self.answer(request)
        return out if isinstance(out, authored.Reply) else authored.Reply(text=out, status=200)


def quiet(*args, **kwargs):
    pass


# ---- prompt -----------------------------------------------------------------------------------------
def user_text(tier: int, turn: int = 0, **kw) -> str:
    request = authored.make_request(1, tier, turn, "some-model", 0.7, **kw)
    assert [m["role"] for m in request.messages] == ["system", "user"]
    return request.messages[1]["content"]


@pytest.mark.parametrize("tier", authored.TIERS)
def test_prompt_carries_the_generators_legend(tier):
    text = user_text(tier)
    assert len(udmf.LEGEND) == 31
    for char, symbol in udmf.LEGEND.items():
        assert f"\n  {char}  {symbol.name}\n" in text, (char, symbol.name)
    legend = authored.legend_text()
    assert legend in text and len(legend.split("\n")) == len(udmf.LEGEND) + 5      # one line per symbol, five headings


@pytest.mark.parametrize("tier", authored.TIERS)
def test_prompt_states_the_rules_the_size_and_the_tier(tier):
    text = user_text(tier)
    lo, hi = layouts.TIERS[tier]["size"]
    for phrase in ("every row has the same number of characters", "fully enclosed", "Exactly one P",
                   "Every cell that is not '#' must be reachable from P",
                   "every key must be reachable before the door it opens",
                   "A cell is 64 map units wide", "every corridor is at least one cell wide",
                   f"Size: {lo} to {hi} cells on each side", "each map in its own fenced code block"):
        assert phrase in text, phrase
    name, skills = authored.TIER_TESTS[tier]
    assert f"What tier {tier} tests: {name}. {skills}." in text
    assert f"At least {layouts.TIERS[tier]['min_open']} cells that are not wall" in text
    assert f"{layouts.MIN_MONSTERS[tier]} to {authored.MAX_MONSTERS[tier]}" in text
    assert "Exactly one E." in text and "No E:" not in text
    if tier == 1:                                           # every second arena request asks for maps without an exit
        assert "No E: these maps have no exit pad." in user_text(1, 1) and "Exactly one E." not in user_text(1, 1)
        assert "Put the exit" in user_text(1, 0) and "Put the exit" not in user_text(1, 1)
    else:
        assert "Exactly one E." in user_text(tier, 1)
    assert f"Draw 5 different maps for tier {tier}" in text and "Draw 3 different" in user_text(tier, count=3)


def test_tier_descriptions_are_those_of_the_evaluation_doc():
    doc = (ROOT / "docs" / "EVALUATION.md").read_text()
    for tier, (name, skills) in authored.TIER_TESTS.items():
        assert f"| {tier} | {name} | {skills} |" in doc
    assert set(authored.TIER_TESTS) == set(authored.TIERS) == {1, 2, 3, 4}


def test_tier_numbers_follow_the_generator():
    assert authored.MIN_LOCKS == {1: 0, 2: 0, 3: 1, 4: 2}
    for tier in authored.TIERS:
        most_locks = max(n for n, _ in layouts.TIERS[tier]["locks"])
        assert (authored.MIN_LOCKS[tier] > 0) == (most_locks > 0)
        assert layouts.MIN_MONSTERS[tier] <= authored.MAX_MONSTERS[tier]
        for seed in range(8000300, 8000340):               # generated maps stay inside the range the prompt gives
            assert layouts.generate(tier, seed)["counts"]["monsters"] <= authored.MAX_MONSTERS[tier]


def test_prompt_examples_are_the_two_hand_drawn_ones_and_valid():
    text = user_text(3)
    assert [tuple(m) for m in authored.split_reply(text)] == [tuple(lines(g)) for g in authored.EXAMPLES]
    assert len(authored.EXAMPLES) == 2
    for grid in authored.EXAMPLES:
        rows = udmf.parse_grid(grid)
        assert len({len(row) for row in lines(grid)}) == 1 and len(rows[0]) <= 13 and len(rows) <= 7     # tiny
        assert set(rows[0] + rows[-1]) == {"#"} and all(row[0] == row[-1] == "#" for row in rows)
        g = layouts.Grid(len(rows[0]), len(rows))
        g.rows = [list(row) for row in rows]
        assert all(authored.is_doorway(g, c, r) for c, r in g.cells("DRBY"))
        sol = solve(grid)
        assert sol.solvable and sol.has_exit
        assert udmf.build_wad(grid)[:4] == b"PWAD"
    assert solve(authored.EXAMPLES[1]).keys_needed


def test_prompt_holds_no_generated_map():
    """The examples are not layouts of the generator: no stored grid appears in any prompt."""
    files = sorted(SUITES.glob("*.jsonl")) + [ROOT / "data" / "maps" / "train" / "manifest.jsonl"]
    grids = [row["grid"] for path in files if path.is_file() for row in cli.read_rows(path)]
    assert len(grids) >= 480
    known = {authored.layout_key(grid) for grid in grids}
    assert not {authored.layout_key(grid) for grid in authored.EXAMPLES} & known
    prompts = [user_text(tier, turn) for tier in authored.TIERS for turn in range(3)]
    for prompt in prompts:
        assert len(authored.split_reply(prompt)) == 2
        assert not any(grid in prompt for grid in grids)
    for seed in range(8000400, 8000410):                   # and fresh generator output is not there either
        for tier in authored.TIERS:
            assert layouts.generate(tier, seed)["grid"] not in prompts[0]


@pytest.mark.parametrize("tier", authored.TIERS)
def test_briefs_vary_across_requests(tier):
    styles = authored.style_order(tier, seed=0)
    names = [name for name, _ in styles]
    assert len(set(names)) == len(names) >= 6
    requests = [authored.make_request(i + 1, tier, i, "m", 0.7) for i in range(3 * len(styles))]
    assert [r.style for r in requests[:len(styles)]] == names             # every style before any repeats
    assert len({r.brief for r in requests}) == len(requests)              # and no brief twice
    assert all(r.brief in r.messages[1]["content"] for r in requests)
    assert sorted(names) == sorted(n for n, _ in authored.style_order(tier, seed=1))
    again = authored.make_request(9, tier, 4, "other", 1.0)
    assert again.brief == requests[4].brief and again.model == "other"    # a pure function of (seed, tier, turn)
    assert authored.make_request(1, tier, 4, "m", 0.7, seed=5).brief != requests[4].brief
    lo, hi = layouts.TIERS[tier]["size"]
    for r in requests:
        w, h = (int(v) for v in re.search(r"about (\d+) columns by (\d+) rows", r.brief).groups())
        assert lo <= min(w, h) and max(w, h) <= hi
        assert (w - 2) * (h - 2) >= 2 * layouts.TIERS[tier]["min_open"]   # room for the floor area the tier needs


def test_styles_cover_the_named_plans():
    names = {name for name, _, _ in authored.STYLES}
    assert {"hub_and_spokes", "spiral", "symmetric_wings", "corridor_with_side_rooms", "loop_with_shortcuts",
            "arena_with_pillars"} <= names
    assert all(set(tiers) <= set(authored.TIERS) and text for _, tiers, text in authored.STYLES)


# ---- reading replies --------------------------------------------------------------------------------
A, B = "#####\n#P.E#\n#####", "######\n#P..z#\n######"


def test_split_reply_reads_fenced_blocks():
    reply = f"Sure!\n\nMap 1\n```text\n{A}\n```\nnot a map: #P#\n\n~~~ascii\r\n{B}\r\n~~~\r\nBye."
    assert authored.split_reply(reply) == [lines(A), lines(B)]
    assert authored.split_reply("Here is a map without a fence:\n" + A) == []
    assert authored.split_reply("") == [] and authored.split_reply("```\nno map here\n```") == []
    assert authored.split_reply(f"  ```\n{A}\n  ```") == [lines(A)]                      # indented fences
    assert authored.split_reply(f"````\n{A}\n````") == [lines(A)]                        # longer fences
    assert authored.split_reply(f"```\n{A}") == [lines(A)]                               # never closed
    assert authored.split_reply(f"```map\n{A}\n```map\n{B}\n```") == [lines(A), lines(B)]   # closed by the next opener
    drafts = f"<think>first try:\n```\n{B}\n```\nno, too small</think>\n```\n{A}\n```"
    assert authored.split_reply(drafts) == [lines(A)]                                     # reasoning is not an answer
    assert authored.split_reply(f"<think>never finished\n```\n{B}\n```") == []


def test_one_block_may_hold_several_maps():
    block = f"```\nMap 1: the first\n{A}\n\n# Map 2\n{B}\nMap #3: again\n{A}\n```"
    assert authored.split_reply(block) == [lines(A), lines(B), lines(A)]
    assert authored._is_row("#..zz..#") and authored._is_row("# . P . #") and authored._is_row("#.. ..#")
    assert authored._is_row("#P...X...z#")                    # a wrong symbol stays in the map and is reported
    assert not authored._is_row("") and not authored._is_row("hall") and not authored._is_row("## Map 2: spiral")


def test_known_layouts_skips_what_it_cannot_read(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (
        {"grid": A}, {"grid": lines(B)}, {"path": "somewhere.wad"}, {"grid": "#P?#"}, {"grid": "####"}, {"grid": 7})) + "\n")
    assert authored.known_layouts([path]) == {authored.layout_key(A), authored.layout_key(B)}


def test_tidy_pads_short_rows_with_wall():
    rows, padded = authored.tidy(["#####", "#P.", "#..E#", "####"])
    assert rows == ["#####", "#P.##", "#..E#", "#####"] and padded
    assert authored.tidy(lines(A)) == (lines(A), False)
    assert authored.tidy(["   #####  ", "   #P.E#", "   #####\t"]) == (lines(A), False)     # indentation, trailing blanks
    assert authored.tidy(["\t#####", "\t#P.E#", "\t#####"]) == (lines(A), False)
    assert authored.tidy(["# # # # #", "# P . E #", "# # # # #"]) == (lines(A), False)      # one blank between symbols
    assert authored.tidy(["  ###", "#####", "#P.E#", "#####"])[0] == ["#####", "#####", "#P.E#", "#####"]   # left of the outline
    assert authored.tidy(["#####", "#P E#", "#####"])[0][1] == "#P E#"                      # a blank inside stays ...
    assert reason("#####\n#P E#\n#####", 1) == "unknown symbol"                             # ... and is rejected


# ---- checks -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("tier,which", [(t, i) for t in authored.TIERS for i in (0, 1)])
def test_hand_drawn_maps_pass_every_check(tier, which):
    grid = GOOD[tier][which].strip()
    found = authored.examine(lines(grid), tier, seed=3)
    assert found["grid"] == grid and not found["padded"]
    assert found["hash"] == cli.grid_hash(grid) and found["layout"] == authored.layout_key(grid)
    sol = solve(grid)
    spec = layouts.TIERS[tier]
    assert sol.solvable and found["counts"]["open_cells"] >= spec["min_open"]
    # the tick cap is the generator's: same function, a private stream per grid, feasible at walking speed
    rng = random.Random(f"authored-ticks-v1/3/{found['hash']}")
    assert found["episode_ticks"] == layouts.tick_cap(rng, tier, sol)
    assert found["episode_ticks"] in range(layouts.MIN_TICKS, layouts.MAX_TICKS + 1, layouts.TICK_STEP)
    assert found["episode_ticks"] >= layouts.MIN_BUDGET * layouts.FRAME_SKIP * sol.optimal_steps
    assert found["episode_ticks"] == authored.examine(lines(grid), tier, seed=3)["episode_ticks"]
    # counts as the generator writes them
    counts = dict(found["counts"])
    route = {k: counts.pop(k) for k in ("route_cells", "route_corners", "route_doors", "optimal_steps", "doors_met_before_key")}
    assert counts == layouts.describe(grid, {})
    assert route["route_cells"] == sol.path_cells and route["optimal_steps"] == sol.optimal_steps
    assert set(found["counts"]) == set(layouts.generate(tier, 8000500)["counts"])
    if tier >= 3:
        assert sol.keys_needed and counts["locks"] >= authored.MIN_LOCKS[tier]
        assert sol.path_cells >= spec["min_route"]


def test_door_met_before_its_key_is_counted():
    g = layouts.Grid(1, 1)
    for grid, met in ((authored._RED_ERRAND, 1), (authored._BLUE_WINGS, 1), (authored._SIDE_ROOMS, 0),
                      ("###########\n#P.r..R..E#\n###########", 0)):
        g.rows = [list(row) for row in lines(grid)]
        g.w, g.h = len(g.rows[0]), len(g.rows)
        assert authored.doors_met_before_key(g, g.cells("P")[0]) == met, grid


REJECTIONS = [
    # ---- form
    ("unknown symbol", 1, authored._ODD_SYMBOLS),
    ("unknown symbol", 2, put(authored._SIDE_ROOMS, (5, 5, "e"))),               # lower-case e is not the exit
    ("no player start", 1, "#####\n#.z.#\n#####"),
    ("more than one player start", 4, authored._TWO_STARTS),
    ("more than one exit", 1, box(9, 6, (1, 1, "P"), (7, 1, "E"), (7, 4, "E"), (4, 2, "z"), (4, 3, "z"))),
    ("not enclosed by walls", 4, authored._OPEN_EDGE),
    ("not enclosed by walls", 1, "\n".join(lines(arena(2))[:-1])),                # bottom wall missing
    ("not enclosed by walls", 1, put(arena(2), (0, 3, "."))),
    ("door not in a doorway", 2, authored._DOOR_IN_FLOOR),
    ("door not in a doorway", 1, "########\n#P.DD.z#\n########"),                 # a door beside a door
    ("door not in a doorway", 1, "#######\n#P.DEz#\n#######"),                    # a door against the exit pad
    ("door not in a doorway", 1, "######\n#Pz.D#\n######"),                       # a door with nothing behind it
    ("unreachable cells", 2, authored._SEALED_EXIT),
    ("unreachable cells", 1, "#######\n#P.E.z#\n#######"),                        # floor behind the exit pad
    ("key not reachable before its door", 3, authored._KEY_BEHIND_ITS_DOOR),
    ("key not reachable before its door", 3, authored._RED_ERRAND.replace("r", ".")),   # no key at all
    ("key not reachable before its door", 3, authored._RED_ERRAND.replace("r", "b")),   # a key of another colour
    # ---- tier
    ("larger than the size range", 1, box(24, 6, (1, 1, "P"), (20, 2, "z"), (20, 3, "z"))),
    ("too little floor", 2, authored.EXAMPLES[0]),
    ("too little floor", 4, authored._TINY_LOCK),
    ("no exit", 2, authored._SIDE_ROOMS.replace("E", ".")),
    ("keys or locked doors in a tier without them", 2, authored._RED_ERRAND),
    ("keys or locked doors in a tier without them", 2, put(authored._SIDE_ROOMS, (9, 5, "y"))),
    ("too few locked doors", 3, authored._SIDE_ROOMS),
    ("too few locked doors", 4, authored._RED_ERRAND),                            # one lock where two are needed
    ("too few locked doors", 4, authored._TWO_KEYS.replace("Y", "R").replace("3", "1")),    # two locks, one colour
    ("locks can be bypassed", 4, put(authored._FOUR_BANDS, (1, 4, "."), (1, 12, "."))),
    ("exit too close to the start", 2, put(authored._SIDE_ROOMS.replace("E", "."), (3, 2, "E"))),
    ("one plain room is not an exploration map", 2, box(14, 9, (1, 1, "P"), (12, 7, "E"))),
    ("too few monsters", 1, box(8, 6, (1, 1, "P"), (5, 3, "z"))),
    ("too many monsters", 2, put(authored._SIDE_ROOMS, *[(c, 5, "z") for c in range(3, 13, 2)])),
    # ---- time
    ("route too long for the largest tick cap", 2, snake()),
]


@pytest.mark.parametrize("expected,tier,grid", REJECTIONS, ids=[f"{r[0]}-{i}" for i, r in enumerate(REJECTIONS)])
def test_examine_rejects(expected, tier, grid):
    assert reason(grid, tier) == expected


def test_a_grid_the_compiler_refuses_is_rejected(monkeypatch):
    def refuse(grid, **options):
        raise ValueError("door cell at column 3, row 2 touches another door or an exit")
    monkeypatch.setattr(udmf, "build_wad", refuse)
    assert reason(arena(1), 1) == "does not compile"


def test_rejection_cases_fail_for_the_stated_reason_only():
    """Each case above differs from an accepted map by the one thing it is rejected for."""
    assert authored.examine(lines(arena(2)), 1)["counts"]["monsters"] == 2
    assert authored.examine(lines(box(8, 6, (1, 1, "P"), (5, 3, "z"), (5, 2, "z"))), 1)
    assert authored.examine(lines(box(22, 6, (1, 1, "P"), (18, 2, "z"), (18, 3, "z"))), 1)        # within the slack
    assert authored.examine(lines(put(authored._SIDE_ROOMS, *[(c, 5, "z") for c in range(3, 11, 2)])), 2)   # six monsters
    halves = put(box(14, 9, (1, 1, "P"), (12, 7, "E")), *[(6, r, "#") for r in range(1, 8)], (6, 4, "D"))
    assert authored.examine(lines(halves), 2)["counts"]["rooms"] == 2                             # the room, divided
    short = snake(21, 9)                                                                          # the same plan, shorter
    assert authored.examine(lines(short), 2)["episode_ticks"] <= layouts.MAX_TICKS
    assert solve(snake()).solvable and layouts.FRAME_SKIP * solve(snake()).optimal_steps * layouts.MIN_BUDGET > layouts.MAX_TICKS
    with pytest.raises(authored.Rejected) as caught:
        authored.examine(lines(authored._ODD_SYMBOLS), 1)
    assert caught.value.symbols == ("X", "~")


def test_examine_crops_spare_wall_like_the_generator():
    grid = GOOD[2][0].strip()
    wide = "\n".join(["#" * 25] * 2 + ["###" + row + "###" for row in lines(grid)] + ["#" * 25])
    assert authored.examine(lines(wide), 2)["grid"] == grid
    ragged = [row.rstrip("#") if 0 < i < 10 else row for i, row in enumerate(lines(grid))]
    found = authored.examine(ragged, 2)
    assert found["grid"] == grid and found["padded"]
    spaced = [" ".join(row) for row in lines(grid)]
    assert authored.examine(spaced, 2)["grid"] == grid


def mangle(rng: random.Random, grid: str) -> list[str]:
    """A grid with the slips a drawn map tends to have: cells dropped, added or changed, rows lost,
    doubled, cut short or indented, blanks between symbols."""
    alphabet = "".join(udmf.LEGEND) + "#####....." + " \t|-+XSeO0?"
    rows = grid.split("\n")
    for _ in range(rng.randint(1, 4)):
        kind, r = rng.randrange(9), rng.randrange(len(rows))
        c = rng.randrange(len(rows[r]) + 1)
        if kind == 0:
            rows[r] = rows[r][:c] + rows[r][c + 1:]
        elif kind == 1:
            rows[r] = rows[r][:c] + rng.choice(alphabet) + rows[r][c:]
        elif kind in (2, 3):
            rows[r] = rows[r][:c] + rng.choice(alphabet) + rows[r][c + 1:]
        elif kind == 4 and len(rows) > 1:
            del rows[r]
        elif kind == 5:
            rows.insert(r, rows[r])
        elif kind == 6:
            rows[r] = rows[r].rstrip("#")
        elif kind == 7:
            rows = [" ".join(row) for row in rows]
        else:
            rows[r] = "   " + rows[r]
    return rows


def test_mangled_maps_never_break_the_checks_and_what_passes_is_a_valid_map():
    rng = random.Random(11)
    sources = cli.read_rows(SUITES / "dev_v1.jsonl")
    reasons, accepted = set(), 0
    for _ in range(1200):
        source = rng.choice(sources)
        tier = rng.choice(authored.TIERS) if rng.random() < 0.15 else source["tier"]
        reply = "Map:\n```\n" + "\n".join(mangle(rng, source["grid"])) + "\n```\n"
        for candidate in authored.split_reply(reply):
            try:
                found = authored.examine(candidate, tier)             # anything but Rejected fails the test
            except authored.Rejected as e:
                reasons.add(e.reason)
                continue
            accepted += 1
            rows = udmf.parse_grid(found["grid"])                     # one start, legend symbols only
            assert len({len(row) for row in found["grid"].split("\n")}) == 1
            assert set(rows[0] + rows[-1]) == {"#"} and all(row[0] == row[-1] == "#" for row in rows)
            sol = solve(found["grid"])
            assert sol.solvable and found["grid"].count("E") <= 1 and (sol.has_exit or tier == 1)
            assert udmf.build_wad(found["grid"])[:4] == b"PWAD"
            assert layouts.MIN_TICKS <= found["episode_ticks"] <= layouts.MAX_TICKS
            assert found["episode_ticks"] >= layouts.MIN_BUDGET * layouts.FRAME_SKIP * sol.optimal_steps
            assert found["counts"]["open_cells"] >= layouts.TIERS[tier]["min_open"]
            assert found["counts"]["locks"] >= authored.MIN_LOCKS[tier] and (sol.keys_needed or not found["counts"]["locks"])
    assert accepted >= 100 and len(reasons) >= 10, (accepted, sorted(reasons))
    for _ in range(300):                                              # and plain noise is only ever rejected
        noise = ["".join(rng.choice("#.PEDRz \t~`x1") for _ in range(rng.randint(1, 12))) for _ in range(rng.randint(1, 8))]
        for candidate in authored.split_reply("```\n" + "\n".join(noise) + "\n```"):
            try:
                authored.examine(candidate, rng.choice(authored.TIERS))
            except authored.Rejected:
                pass


def test_a_check_that_breaks_costs_one_map_not_the_run(monkeypatch):
    real = authored.examine

    def flaky(lines_, tier, seed=0):
        if len(lines_[0]) == 9:                                        # arena(1) is nine cells wide
            raise IndexError("list index out of range")
        return real(lines_, tier, seed)
    monkeypatch.setattr(authored, "examine", flaky)
    client = Scripted(lambda request: fence(arena(0), arena(1), arena(2)))
    rows, stats = authored.collect(client, tiers=[1], per_tier=2, max_requests=1, echo=quiet)
    assert [row["grid"] for row in rows] == [arena(0), arena(2)]
    assert stats.rejected == {"could not be checked (IndexError)": 1}


def test_self_check_stops_a_broken_installation_before_any_request(monkeypatch, tmp_path, capsys):
    said = []
    authored.self_check(engine=False, echo=said.append)                # passes as things are, without a word
    assert said == []
    client = Scripted(lambda request: "")
    out = tmp_path / "x.jsonl"

    def broken(rng, tier, sol):
        raise TypeError("tick_cap() takes 2 positional arguments but 3 were given")
    monkeypatch.setattr(authored, "CannedClient", lambda: client)
    monkeypatch.setattr(layouts, "tick_cap", broken)                   # the generator changed under the script
    with pytest.raises(SystemExit, match="self-check failed, no request was sent: the checks do not run here"):
        authored.main(["--dry-run", "--no-engine-check", "--out", str(out)])
    monkeypatch.undo()

    monkeypatch.setattr(authored, "CannedClient", lambda: client)
    monkeypatch.setattr(cli, "verify_map", lambda row: {"name": row["name"], "reached": False, "cap_used": None})
    with pytest.raises(SystemExit, match="does not pass the engine check here .reference walker does not reach the exit"):
        authored.main(["--dry-run", "--out", str(out)])                # the engine cannot confirm a known good map
    assert not client.seen and not out.exists()
    monkeypatch.undo()

    monkeypatch.setitem(layouts.TIERS, 4, {**layouts.TIERS[4], "min_open": 500})     # a requirement moved: no stop
    authored.self_check(engine=False, echo=said.append)
    assert said == ["note: a hand-drawn tier 4 map no longer passes the checks (too little floor)"] * 2
    capsys.readouterr()


# ---- near-duplicates --------------------------------------------------------------------------------
def turns_and_mirrors(grid: str) -> list[str]:
    rows, out = lines(grid), []
    for _ in range(4):
        rows = ["".join(col) for col in zip(*rows[::-1])]
        out += ["\n".join(rows), "\n".join(row[::-1] for row in rows), "\n".join(rows[::-1])]
    return out


@pytest.mark.parametrize("tier", authored.TIERS)
def test_layout_key_ignores_rotation_mirror_and_things(tier):
    grid = GOOD[tier][0].strip()
    key = authored.layout_key(grid)
    variants = turns_and_mirrors(grid)
    assert len(set(variants)) == 8
    assert {authored.layout_key(v) for v in variants} == {key}
    bare = "".join("." if udmf.LEGEND.get(ch) and udmf.LEGEND[ch].kind == "thing" else ch for ch in grid)
    assert bare != grid and authored.layout_key(bare) == key                       # identical after removing things
    other_things = grid.replace("z", "c").replace("h", "m").replace("i", "d")
    assert authored.layout_key(other_things) == key
    assert authored.layout_key(grid.translate(str.maketrans("RBYrby123", "BYRbyr231"))) == key     # lock colours


def test_layout_key_tells_different_plans_apart():
    keys = {authored.layout_key(g.strip()) for pair in GOOD.values() for g in pair}
    assert len(keys) == 8
    grid = GOOD[2][0].strip()
    assert authored.layout_key(put(grid, (9, 5, "#"))) != authored.layout_key(grid)             # one wall more
    assert authored.layout_key(put(grid, (2, 2, "."), (1, 1, "P"))) != authored.layout_key(grid)   # the start moved
    assert authored.layout_key(put(grid, (2, 4, "."))) != authored.layout_key(grid)             # a door removed
    assert authored.layout_key(GOOD[3][0].replace("R", "D")) != authored.layout_key(GOOD[3][0].strip())  # lock -> door


def test_copies_are_dropped_within_a_run_and_against_the_frozen_suites():
    grid = GOOD[2][0].strip()
    mirrored = "\n".join(row[::-1] for row in lines(grid)).replace("z", ".")
    client = Scripted(lambda request: fence(grid, mirrored, grid.replace("h", "m")))
    rows, stats = authored.collect(client, tiers=[2], per_tier=3, max_requests=1, echo=quiet)
    assert [row["grid"] for row in rows] == [grid] and stats.rejected == {"near-duplicate": 2}

    frozen = next(row["grid"] for row in cli.read_rows(SUITES / "dev_v1.jsonl") if row["tier"] == 2)
    assert authored.examine(lines(frozen), 2)                                     # a generated map passes the checks,
    taken = authored.known_layouts([SUITES / "dev_v1.jsonl", SUITES / "heldout_v1.jsonl"])
    assert len(taken) >= 470
    turned = "\n".join(lines(frozen)[::-1])
    client = Scripted(lambda request: fence(frozen, turned, grid))
    rows, stats = authored.collect(client, tiers=[2], per_tier=3, max_requests=1, taken=taken, echo=quiet)
    assert [row["grid"] for row in rows] == [grid] and stats.rejected == {"near-duplicate": 2}   # but is not taken again


def test_an_example_of_the_prompt_handed_back_is_not_accepted():
    echoed = put(authored.EXAMPLES[0], (6, 1, "z"), (6, 3, "i"))            # the first example with monsters added
    assert authored.examine(lines(echoed), 1)                                # a valid arena by the checks,
    client = Scripted(lambda request: fence(echoed, arena(0)))
    rows, stats = authored.collect(client, tiers=[1], per_tier=2, max_requests=1, echo=quiet)
    assert [row["grid"] for row in rows] == [arena(0)] and stats.rejected == {"near-duplicate": 1}   # but not a new one


def test_generated_maps_pass_the_same_checks():
    """The filter is the generator's own standard: maps of the frozen suite pass it in their tier."""
    rows = cli.read_rows(SUITES / "dev_v1.jsonl")
    failed = []
    for row in rows:
        try:
            found = authored.examine(lines(row["grid"]), row["tier"])
        except authored.Rejected as e:
            failed.append((row["name"], e.reason))
            continue
        assert found["grid"] == row["grid"]
        assert {k: v for k, v in found["counts"].items() if k not in ("ambush", "doors_met_before_key")} == \
               {k: v for k, v in row["counts"].items() if k not in ("ambush", "doors_met_before_key")}
    assert not failed, failed


# ---- collecting -------------------------------------------------------------------------------------
def test_quota_surplus_and_names():
    served = iter(range(100))
    client = Scripted(lambda request: fence(arena(next(served)), arena(next(served)), arena(next(served))))
    rows, stats = authored.collect(client, tiers=[1], per_tier=4, max_requests=10, stem="mysuite", echo=quiet)
    assert len(client.seen) == 2 and stats.requests == 2                 # stops as soon as the tier is full
    assert [row["name"] for row in rows] == [f"mysuite_t1_00{i}" for i in range(4)]
    assert (stats.returned, stats.accepted, stats.surplus) == (6, 4, 2) and not stats.rejected and not stats.notes
    assert stats.total == {1: 4}
    row = rows[0]
    assert tuple(row)[:len(cli.SUITE_KEYS)] == cli.SUITE_KEYS
    assert row["options"] == {} and row["source"] == "authored" and row["model"] == authored.DEFAULT_MODEL
    assert row["brief"] == client.seen[0].brief and row["style"] == client.seen[0].style
    assert row["hash"] == cli.grid_hash(row["grid"]) and row["seed"] == int(row["hash"][:8], 16)
    assert row["layout"] == authored.layout_key(row["grid"]) and row["temperature"] == 0.7


@pytest.mark.parametrize("concurrency", [1, 3, 8])
def test_max_requests_is_a_hard_cap(concurrency):
    client = Scripted(lambda request: fence(authored._SEALED_EXIT))      # never an accepted map
    rows, stats = authored.collect(client, per_tier=40, max_requests=7, concurrency=concurrency, patience=50, echo=quiet)
    assert len(client.seen) == 7 == stats.requests and not rows
    assert stats.rejected == {"unreachable cells": 7}
    assert sorted(r.index for r in client.seen) == list(range(1, 8))
    assert any("the request cap was reached" in note for note in stats.notes)
    none = Scripted(lambda request: "")
    assert authored.collect(none, max_requests=0, echo=quiet)[1].requests == 0 and not none.seen


def test_requests_go_to_the_tier_furthest_from_its_target():
    served = iter(range(1000))

    def answer(request):                                                  # tier 1 fills quickly, the others never
        return fence(arena(next(served)), arena(next(served))) if request.tier == 1 else "no map"
    client = Scripted(answer)
    rows, stats = authored.collect(client, per_tier=4, max_requests=14, patience=3, echo=quiet)
    asked = [r.tier for r in client.seen]
    assert asked[:4] == [1, 2, 3, 4]
    assert asked.count(1) == 2 and {asked.count(t) for t in (2, 3, 4)} == {3}       # patience ends the barren tiers
    assert stats.total == {1: 4, 2: 0, 3: 0, 4: 0} and stats.no_map == 9
    assert sum("3 requests in a row without an accepted map" in note for note in stats.notes) == 3


def test_models_and_temperatures_take_turns():
    served = iter(range(1000))
    client = Scripted(lambda request: fence(arena(next(served))))
    rows, stats = authored.collect(client, tiers=[1], per_tier=8, models=["m-a", "m-b"], temperatures=[0.2, 0.9],
                                   max_requests=8, echo=quiet)
    used = [(r.model, r.temperature) for r in client.seen]
    assert len(used) == 8 and all(used.count(pair) == 2 for pair in set(used)) and len(set(used)) == 4
    assert {m: c["accepted"] for m, c in stats.models.items()} == {"m-a": 4, "m-b": 4}
    assert {row["model"] for row in rows} == {"m-a", "m-b"} and {row["temperature"] for row in rows} == {0.2, 0.9}
    assert all(r.max_tokens == 8000 and "Draw 5 different maps" in r.messages[1]["content"] for r in client.seen)


def test_a_failing_model_is_dropped_and_the_other_goes_on():
    served = iter(range(1000))

    def answer(request):
        if request.model == "broken":
            return authored.Reply(status=404, error="HTTP 404: no such model")
        return fence(arena(next(served)))
    client = Scripted(answer)
    rows, stats = authored.collect(client, tiers=[1], per_tier=6, models=["broken", "fine"], max_requests=30, echo=quiet)
    assert [r.model for r in client.seen].count("broken") == authored.MODEL_STRIKES
    assert len(rows) == 6 and stats.failed == authored.MODEL_STRIKES and not stats.stopped
    assert any("broken was dropped" in note for note in stats.notes)
    alone = Scripted(lambda request: authored.Reply(status=500, error="HTTP 500"))
    rows, stats = authored.collect(alone, tiers=[1, 2], per_tier=2, max_requests=30, echo=quiet)
    assert len(alone.seen) == authored.MODEL_STRIKES and not rows
    assert sum("no model left to ask" in note for note in stats.notes) == 2          # one note per unfinished tier


def test_refused_credentials_stop_the_run():
    client = Scripted(lambda request: authored.Reply(status=401, error="HTTP 401: unauthorized"))
    said = []
    rows, stats = authored.collect(client, per_tier=5, max_requests=50, echo=said.append)
    assert len(client.seen) == 1 and not rows and "HTTP 401" in stats.stopped
    assert authored.ENV_KEY in stats.stopped and said == [f"[  1] tier 1 {authored.DEFAULT_MODEL}: HTTP 401: unauthorized"]


def test_rate_limit_backs_off_and_then_stops():
    waits = []
    client = Scripted(lambda request: authored.Reply(status=429, error="HTTP 429: slow down"))
    rows, stats = authored.collect(client, per_tier=5, max_requests=50, concurrency=3, echo=quiet, sleep=waits.append)
    assert waits == [15.0, 30.0, 60.0, 120.0] and "429" in stats.stopped
    assert len(client.seen) == 3 + authored.THROTTLE_WAVES - 1            # one wave of three, then one at a time
    assert not stats.notes or all("dropped" not in note for note in stats.notes)     # not held against the model

    served = iter(range(1000))
    turn = iter(range(1000))
    client = Scripted(lambda request: authored.Reply(status=429, error="HTTP 429") if next(turn) == 0
                      else fence(arena(next(served))))
    rows, stats = authored.collect(client, tiers=[1], per_tier=3, max_requests=50, echo=quiet, sleep=waits.append)
    assert len(rows) == 3 and stats.failed == 1 and not stats.stopped and waits[-1] == 15.0


def test_engine_verdict_is_the_last_check(monkeypatch):
    verdicts = iter([{"reached": True, "cap_used": 0.2}, {"reached": False, "cap_used": None},
                     {"reached": True, "cap_used": 0.9}, RuntimeError("sector count differs"), RuntimeError("again"),
                     OSError("the engine did not start"), {"reached": None, "cap_used": None}])

    def verify_map(row):
        verdict = next(verdicts)
        if isinstance(verdict, Exception):
            raise verdict
        return {"name": row["name"], **verdict}
    monkeypatch.setattr(cli, "verify_map", verify_map)
    client = Scripted(lambda request: fence(*[arena(k) for k in range(5)]))
    rows, stats = authored.collect(client, tiers=[1], per_tier=5, max_requests=1, verify=authored.engine_check, echo=quiet)
    assert [row["grid"] for row in rows] == [arena(0), arena(4)]      # the last one passed at the second attempt
    assert next(verdicts, None) is None
    assert [row["name"] for row in rows] == ["authored_v1_t1_000", "authored_v1_t1_001"]
    assert stats.rejected == {"reference walker does not reach the exit": 1, "fails in the engine": 1,
                              "reference walker needs too much of the tick cap": 1}


# ---- dry run ----------------------------------------------------------------------------------------
DRY_REASONS = {"unknown symbol": 1, "unreachable cells": 1, "door not in a doorway": 1, "near-duplicate": 1,
               "key not reachable before its door": 1, "more than one player start": 1, "not enclosed by walls": 1,
               "too little floor": 1, "locks can be bypassed": 1}


def dry_run(tmp_path, capsys, *extra):
    out = tmp_path / "suite" / "authored_test.jsonl"
    code = authored.main(["--dry-run", "--per-tier", "2", "--out", str(out), "--no-engine-check",
                          "--summary-json", str(tmp_path / "summary.json"), *extra])
    return code, out, capsys.readouterr().out


def test_canned_replies_hold_the_cases_the_dry_run_is_for():
    replies = authored.canned_replies()
    assert set(replies) == set(authored.TIERS)
    flat = [reply for tier in authored.TIERS for reply in replies[tier]]
    assert sum(not authored.split_reply(reply) for reply in flat) == 1          # one answer without a fenced block
    maps = [(tier, m) for tier in authored.TIERS for reply in replies[tier] for m in authored.split_reply(reply)]
    assert len(maps) == 17
    assert sum(len({len(row) for row in m}) > 1 for _, m in maps) == 1          # one ragged map
    assert sum(any(ch not in udmf.LEGEND and ch != " " for row in m for ch in row) for _, m in maps) == 1
    verdicts = []
    for tier, m in maps:
        try:
            authored.examine(m, tier)
            verdicts.append("ok")
        except authored.Rejected as e:
            verdicts.append(e.reason)
    assert verdicts.count("ok") == 9                                              # eight maps and one mirrored copy
    assert {"unknown symbol", "unreachable cells", "key not reachable before its door"} <= set(verdicts)


def test_dry_run_end_to_end(tmp_path, capsys):
    code, out, printed = dry_run(tmp_path, capsys)
    assert code == 0
    rows = cli.read_rows(out)
    assert [row["name"] for row in rows] == [f"authored_test_t{t}_00{i}" for t in authored.TIERS for i in (0, 1)]
    assert [row["grid"] for row in rows] == [g.strip() for t in authored.TIERS for g in GOOD[t]]
    for row in rows:
        assert set(cli.SUITE_KEYS) < set(row) and set(row) - set(cli.SUITE_KEYS) == {
            "source", "model", "temperature", "brief", "hash", "layout"}
        assert row["model"] == authored.DEFAULT_MODEL and row["brief"] and len(row["hash"]) == 40
        assert solve(row["grid"]).solvable and isinstance(row["grid"], str)
    # the summary: requests, maps, per tier, per model, reasons
    assert "dry run: canned replies, no network request is made" in printed
    assert "requests made: 9 of at most 150 (0 failed, 1 without a map in a fenced block, 0 cut off" in printed
    assert ("maps returned: 17 | accepted 8 | rejected 9 | beyond the tier target 0 | "
            "accepted after padding short rows 1") in printed
    table = [line.split() for line in printed.splitlines()]
    assert ["1", "3", "3", "2", "2/2"] in table and ["2", "2", "5", "2", "2/2"] in table
    assert ["3", "2", "3", "2", "2/2"] in table and ["4", "2", "6", "2", "2/2"] in table
    assert [authored.DEFAULT_MODEL, "9", "17", "8"] in table
    for why in DRY_REASONS:
        assert ["1", *why.split()] in table, why
    assert "unknown symbols (maps using them): 'X' 1, '~' 1" in printed
    digest = re.search(r"wrote 8 maps to .*authored_test\.jsonl \(sha256 ([0-9a-f]{64})\)", printed).group(1)
    import hashlib
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    (report,) = json.loads((tmp_path / "summary.json").read_text())
    assert report["dry_run"] is True and report["suite"] == "authored_test.jsonl" and report["suite_sha256"] == digest
    assert report["asked_tiers"] == [1, 2, 3, 4] and report["asked_models"] == [authored.DEFAULT_MODEL]
    assert report["engine_check"] is False and report["target"] == 2 and report["cap"] == 150 and report["seed"] == 0
    assert (report["requests"], report["returned"], report["accepted"], report["no_map"], report["padded"]) == (9, 17, 8, 1, 1)
    assert report["rejected"] == DRY_REASONS and report["unknown_symbols"] == {"X": 1, "~": 1}
    assert report["tiers"] == {str(t): {"requests": 3 if t == 1 else 2, "returned": {1: 3, 2: 5, 3: 3, 4: 6}[t],
                                        "accepted": 2, "total": 2} for t in authored.TIERS}
    assert report["models"] == {authored.DEFAULT_MODEL: {"requests": 9, "returned": 17, "accepted": 8}}
    assert not list(out.parent.glob("*.tmp"))


def test_dry_run_is_reproducible_and_needs_no_environment(tmp_path, capsys, monkeypatch):
    first = dry_run(tmp_path / "a", capsys)
    second = dry_run(tmp_path / "b", capsys)
    assert first[1].read_bytes() == second[1].read_bytes()
    other = dry_run(tmp_path / "c", capsys, "--seed", "7", "--models", "m-one,m-two", "--temperature", "0.3")
    rows = cli.read_rows(other[1])
    assert [row["grid"] for row in rows] == [row["grid"] for row in cli.read_rows(first[1])]
    assert {row["model"] for row in rows} == {"m-one", "m-two"} and {row["temperature"] for row in rows} == {0.3}
    assert [row["brief"] for row in rows] != [row["brief"] for row in cli.read_rows(first[1])]


def test_dry_run_with_the_default_target_uses_every_canned_reply_once(capsys):
    real = authored.DEFAULT_OUT
    before = (real.stat().st_mtime_ns, real.read_bytes()) if real.exists() else None
    listing = sorted(p.name for p in real.parent.iterdir())
    assert authored.main(["--dry-run", "--no-engine-check"]) == 0
    assert ((real.stat().st_mtime_ns, real.read_bytes()) if real.exists() else None) == before      # the real suite is untouched
    assert sorted(p.name for p in real.parent.iterdir()) == listing
    printed = capsys.readouterr().out
    assert "requests made: 9 of at most 150" in printed and "accepted 8" in printed
    assert printed.count("of 40 maps: no canned reply left") == 4
    assert "run again with --resume" not in printed                      # more requests would not help a dry run
    assert "nothing written (a dry run writes a suite only when --out is given)" in printed
    assert real == SUITES / "authored_v1.jsonl"


def test_exclude_keeps_out_the_layouts_of_another_file(tmp_path, capsys):
    other = tmp_path / "other.jsonl"
    turned = "\n".join(lines(GOOD[3][0])[::-1])                               # upside down, as a list of rows
    other.write_text(json.dumps({"grid": GOOD[1][1].strip()}) + "\n" + json.dumps({"grid": lines(turned)}) + "\n")
    code, out, printed = dry_run(tmp_path, capsys, "--exclude", str(other))
    rows = cli.read_rows(out)
    assert code == 0 and len(rows) == 6 and "rejected 11" in printed
    assert GOOD[1][1].strip() not in [row["grid"] for row in rows] and GOOD[3][0].strip() not in [row["grid"] for row in rows]
    assert ["3", "near-duplicate"] in [line.split() for line in printed.splitlines()]


def test_dry_run_respects_tiers_and_the_request_cap(tmp_path, capsys):
    code, out, printed = dry_run(tmp_path, capsys, "--tiers", "3,4", "--max-requests", "3")
    assert code == 0 and "requests made: 3 of at most 3" in printed
    rows = cli.read_rows(out)
    assert {row["tier"] for row in rows} == {3, 4} and len(rows) == 3
    assert "the request cap was reached" in printed and "run again with --resume and another --seed" in printed
    for bad in (["--tiers", "0,1"], ["--tiers", "5"], ["--tiers", "two"], ["--per-tier", "0"], ["--models", ","],
                ["--temperature", ""], ["--temperature", "warm"], ["--concurrency", "0"], ["--max-requests", "-1"]):
        with pytest.raises(SystemExit):
            authored.main(["--dry-run", *bad])
    capsys.readouterr()


def test_a_suite_file_is_never_overwritten_but_can_be_resumed(tmp_path, capsys):
    out = tmp_path / "authored_test.jsonl"
    assert authored.main(["--dry-run", "--no-engine-check", "--per-tier", "1", "--out", str(out)]) == 0
    first = out.read_text()
    assert len(first.splitlines()) == 4
    with pytest.raises(SystemExit, match="exists: a suite is written once"):
        authored.main(["--dry-run", "--no-engine-check", "--per-tier", "2", "--out", str(out)])
    assert out.read_text() == first
    log = tmp_path / "log.json"
    again = ["--dry-run", "--no-engine-check", "--per-tier", "2", "--out", str(out), "--resume", "--summary-json", str(log)]
    assert authored.main(again) == 0
    assert authored.main(again) == 0                                            # nothing is missing any more
    runs = json.loads(log.read_text())
    assert [(run["requests"], run["accepted"], run["maps"]) for run in runs] == [(9, 4, 8), (0, 0, 8)]
    rows = cli.read_rows(out)
    assert len(rows) == 8 and len({row["name"] for row in rows}) == 8 and len({row["layout"] for row in rows}) == 8
    kept = [json.loads(line) for line in first.splitlines()]
    assert [row for row in rows if row["name"].endswith("_000")] == kept       # earlier maps are kept as they were
    printed = capsys.readouterr().out
    assert printed.count("near-duplicate") >= 1                                 # the canned maps came round again


def test_suite_is_read_by_evaluate_and_by_the_generator_cli(tmp_path, capsys):
    from training import evaluate

    code, out, _ = dry_run(tmp_path, capsys)
    rows = cli.read_rows(out)
    loaded = evaluate.load_suite(out)
    assert [m["id"] for m in loaded] == [row["name"] for row in rows]
    assert [(m["tier"], m["episode_ticks"], m["grid"], m["options"], m["wad"]) for m in loaded] == \
           [(row["tier"], row["episode_ticks"], row["grid"], {}, None) for row in rows]
    maps = evaluate.suite_maps(out, maps_dir=tmp_path / "wads")
    assert [(m["id"], m["tier"], m["episode_ticks"]) for m in maps] == [(r["name"], r["tier"], r["episode_ticks"]) for r in rows]
    for m, row in zip(maps, rows):
        assert Path(m["wad"]).read_bytes() == udmf.build_wad(row["grid"])
    cli.main(["--from-suite", str(out), "--out", str(tmp_path / "rebuilt")])
    assert len(list((tmp_path / "rebuilt").glob("*.wad"))) == 8
    stats = cli.tier_stats(rows)
    assert all(f"tier {t}: 2 maps" in stats for t in authored.TIERS)
    capsys.readouterr()


@pytest.mark.slow
def test_dry_run_maps_play_in_the_engine(tmp_path, capsys):
    """With the engine check on (the default), every accepted map loads in doom_env, plays without
    warnings and is walked to its exit by the reference walker."""
    out = tmp_path / "authored_test.jsonl"
    assert authored.main(["--dry-run", "--per-tier", "2", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "accepted 8 | rejected 9" in printed and "engine" not in printed and "walker" not in printed
    rows = cli.read_rows(out)
    results = [cli.verify_map(row) for row in rows]
    assert [r["reached"] for r in results] == [True, None] + [True] * 6        # the second arena has no exit
    assert max(r["cap_used"] for r in results if r["cap_used"] is not None) <= authored.WALKER_SHARE
    from training.mapmeta import load_meta
    for i, row in enumerate(rows):
        meta = load_meta(str(udmf.write_wad(row["grid"], tmp_path / f"{i}.wad")))
        assert len(meta.exits) == row["counts"]["exits"] and len(meta.monsters) == row["counts"]["monsters"]
        assert len(meta.keys) == row["counts"]["keys"] and meta.locked_total == row["counts"]["locks"]


# ---- client -----------------------------------------------------------------------------------------
class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def completion(text: str, finish: str = "stop", usage=(900, 2100)) -> bytes:
    return json.dumps({"id": "x", "object": "chat.completion",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}).encode()


@pytest.fixture
def endpoint(monkeypatch):
    """A stand-in for the endpoint: records what the opener is asked to send and answers from `reply`."""
    monkeypatch.setenv(authored.ENV_KEY, KEY)
    monkeypatch.setenv(authored.ENV_URL, BASE + "/")

    class Endpoint:
        calls: list = []
        reply = staticmethod(lambda request: completion("```\n" + A + "\n```"))

    def fake_open(self, request, data=None, timeout=None):
        Endpoint.calls.append({"url": request.full_url, "method": request.get_method(), "timeout": timeout,
                               "headers": {k.lower(): v for k, v in request.header_items()},
                               "body": json.loads(request.data) if request.data else None})
        answer = Endpoint.reply(request)
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
    Endpoint.calls = []
    return Endpoint


def http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(BASE + "/chat/completions", code, "error", {}, io.BytesIO(body.encode()))


def test_missing_environment_names_the_variable_and_shows_no_value(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        authored.HttpClient()
    assert authored.ENV_KEY in str(stop.value) and authored.ENV_URL in str(stop.value)
    monkeypatch.setenv(authored.ENV_URL, BASE)
    with pytest.raises(SystemExit) as stop:
        authored.HttpClient()
    assert authored.ENV_KEY in str(stop.value) and authored.ENV_URL not in str(stop.value)
    assert "llm.invalid" not in str(stop.value)
    monkeypatch.delenv(authored.ENV_URL)
    monkeypatch.setenv(authored.ENV_KEY, KEY)
    with pytest.raises(SystemExit) as stop:
        authored.main(["--per-tier", "1", "--out", "unused.jsonl"])
    assert authored.ENV_URL in str(stop.value) and KEY not in str(stop.value) and authored.ENV_KEY not in str(stop.value)
    with pytest.raises(SystemExit) as stop:
        authored.main(["--list-models"])
    assert authored.ENV_URL in str(stop.value) and KEY not in str(stop.value)
    for name, value in ((authored.ENV_URL, "ftp://files.invalid/v1"), (authored.ENV_URL, "llm.invalid/v1"),
                        (authored.ENV_URL, "https://llm.invalid/v 1"), (authored.ENV_KEY, "two words\nand a line")):
        monkeypatch.setenv(authored.ENV_URL, BASE)
        monkeypatch.setenv(authored.ENV_KEY, KEY)
        monkeypatch.setenv(name, value)
        with pytest.raises(SystemExit) as stop:
            authored.HttpClient()
        assert name in str(stop.value) and value not in str(stop.value) and "invalid" not in str(stop.value)
    assert not Path("unused.jsonl").exists()
    printed = capsys.readouterr()
    assert KEY not in printed.out + printed.err


def test_chat_request_is_an_openai_style_post(endpoint):
    client = authored.HttpClient(timeout=12)
    request = authored.make_request(1, 2, 0, "@cf/openai/gpt-oss-120b", 0.7, max_tokens=1234)
    reply = client.chat(request)
    assert (reply.text, reply.error, reply.status, reply.cut_off) == ("```\n" + A + "\n```", "", 200, False)
    assert (reply.prompt_tokens, reply.completion_tokens) == (900, 2100)
    (call,) = endpoint.calls
    assert call["url"] == BASE + "/chat/completions" and call["method"] == "POST" and call["timeout"] == 12
    assert call["headers"]["authorization"] == "Bearer " + KEY
    assert call["headers"]["content-type"] == "application/json"
    assert set(call["body"]) == {"model", "messages", "temperature", "max_tokens"}
    assert call["body"] == {"model": "@cf/openai/gpt-oss-120b", "messages": request.messages, "temperature": 0.7,
                            "max_tokens": 1234}
    assert KEY not in repr(client) and BASE not in repr(client) and KEY not in repr(request) and KEY not in repr(reply)
    assert KEY not in json.dumps(call["body"]) and "llm.invalid" not in json.dumps(call["body"])


def test_base_url_forms(endpoint, monkeypatch):
    for value in (BASE, BASE + "/", BASE + "/chat/completions", BASE + "/chat/completions/"):
        monkeypatch.setenv(authored.ENV_URL, value)
        authored.HttpClient().chat(authored.make_request(1, 1, 0, "m", 0.7))
    assert {call["url"] for call in endpoint.calls} == {BASE + "/chat/completions"}


def test_reply_shapes(endpoint):
    client = authored.HttpClient()
    request = authored.make_request(1, 1, 0, "m", 1.0)
    endpoint.reply = staticmethod(lambda r: completion("cut", finish="length"))
    assert client.chat(request).cut_off
    endpoint.reply = staticmethod(lambda r: json.dumps({"choices": [{"message": {"content": [
        {"type": "text", "text": "part one, "}, {"type": "text", "text": "part two"}]}}]}).encode())
    assert client.chat(request).text == "part one, part two"
    endpoint.reply = staticmethod(lambda r: json.dumps({"choices": [{"message": {"content": None, "reasoning_content": "..."},
                                                                     "finish_reason": "length"}]}).encode())
    empty = client.chat(request)
    assert empty.text == "" and empty.cut_off and not empty.error          # thinking used up the tokens
    endpoint.reply = staticmethod(lambda r: b"<html>gateway</html>")
    assert client.chat(request).error == "the response is not JSON"
    endpoint.reply = staticmethod(lambda r: json.dumps({"error": {"message": f"bad key {KEY} at {BASE}"}}).encode())
    wrong = client.chat(request).error
    assert wrong.startswith("the response is not a chat completion: ") and KEY not in wrong and "llm.invalid" not in wrong
    endpoint.reply = staticmethod(lambda r: json.dumps({"choices": []}).encode())
    assert client.chat(request).error == "the response is not a chat completion"


def test_http_error_shows_the_status_and_200_characters_at_most(endpoint, capsys):
    body = json.dumps({"error": {"message": f"Incorrect key provided: {KEY}. Sent to {BASE}/chat/completions. " + "x" * 2000}})
    endpoint.reply = staticmethod(lambda r: http_error(500, body))
    client = authored.HttpClient()
    reply = client.chat(authored.make_request(1, 1, 0, "m", 0.7))
    assert reply.status == 500 and reply.text == "" and reply.error.startswith("HTTP 500: ")
    assert len(reply.error) <= len("HTTP 500: ") + authored.ERROR_CHARS == 210
    assert "Incorrect key provided: [hidden]" in reply.error
    assert KEY not in reply.error and "llm.invalid" not in reply.error and BASE not in reply.error
    assert client.chat(authored.make_request(1, 1, 0, "m", 0.7)).error == reply.error

    endpoint.reply = staticmethod(lambda r: http_error(503, ""))
    assert client.chat(authored.make_request(1, 1, 0, "m", 0.7)).error == "HTTP 503"
    endpoint.reply = staticmethod(lambda r: urllib.error.URLError(OSError(61, "Connection refused")))
    assert client.chat(authored.make_request(1, 1, 0, "m", 0.7)).error == "no response: [Errno 61] Connection refused"
    endpoint.reply = staticmethod(lambda r: TimeoutError("timed out"))
    assert client.chat(authored.make_request(1, 1, 0, "m", 0.7)).error == "no response: TimeoutError timed out"
    endpoint.reply = staticmethod(lambda r: ValueError(f"Invalid header value b'Bearer {KEY}'"))
    assert client.chat(authored.make_request(1, 1, 0, "m", 0.7)).error == "no response: ValueError"

    # ... and through a whole run: the status and the excerpt are printed, the key and the address never
    endpoint.reply = staticmethod(lambda r: http_error(500, body))
    rows, stats = authored.collect(client, tiers=[1], per_tier=1, max_requests=9)
    printed = capsys.readouterr()
    assert stats.failed == authored.MODEL_STRIKES and "HTTP 500: " in printed.out
    for text in (printed.out, printed.err, stats.summary(), json.dumps(stats.as_dict())):
        assert KEY not in text and "llm.invalid" not in text and "Bearer" not in text and "uthorization" not in text


class InMemoryHTTPS(urllib.request.HTTPSHandler):
    """Answers https requests from a script, so urllib's own opener logic (headers, error
    handling, redirects) runs for real while no socket is ever opened."""

    script: list = []          # (status, headers, body) per request, in order
    seen: list = []

    def https_open(self, req):
        InMemoryHTTPS.seen.append((req.full_url, req.get_method(), {k.lower(): v for k, v in req.header_items()}, req.data))
        code, headers, body = InMemoryHTTPS.script.pop(0)
        message = email.message.Message()
        for name, value in headers.items():
            message[name] = value
        response = urllib.response.addinfourl(io.BytesIO(body), message, req.full_url, code)
        response.msg = "scripted"
        return response


def test_through_urllibs_own_opener_without_a_socket(monkeypatch, capsys):
    """The client on urllib's real OpenerDirector with an in-memory https handler: what is sent,
    how errors come back, and that a redirect is refused instead of followed with the key."""
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", REAL_OPEN)      # the socket guard stays in place
    monkeypatch.setenv(authored.ENV_KEY, KEY)
    monkeypatch.setenv(authored.ENV_URL, BASE)
    client = authored.HttpClient(timeout=5)
    assert any(isinstance(h, authored._NoRedirect) for h in client._opener.handlers)
    assert not any(type(h) is urllib.request.HTTPRedirectHandler for h in client._opener.handlers)
    client._opener = urllib.request.build_opener(authored._NoRedirect, InMemoryHTTPS)
    request = authored.make_request(1, 2, 0, "some/model", 0.7, max_tokens=99)
    elsewhere = "https://elsewhere.invalid/v1/chat/completions"
    InMemoryHTTPS.seen = []
    InMemoryHTTPS.script = [
        (200, {"Content-Type": "application/json"}, completion("```\n" + A + "\n```")),
        (429, {"Content-Type": "application/json"}, json.dumps({"error": f"slow down, {KEY}"}).encode()),
        (302, {"Location": elsewhere}, b"moved"),
        (200, {"Content-Type": "application/json"}, json.dumps({"data": [{"id": "one"}, {"id": "two"}]}).encode()),
    ]
    ok = client.chat(request)
    assert (ok.status, ok.error, authored.split_reply(ok.text)) == (200, "", [lines(A)])
    url, method, headers, data = InMemoryHTTPS.seen[0]
    assert (url, method) == (BASE + "/chat/completions", "POST")
    assert headers["authorization"] == "Bearer " + KEY and headers["content-type"] == "application/json"
    assert headers["user-agent"] == authored.CLIENT_NAME and "python" not in headers["user-agent"].lower()
    assert json.loads(data) == {"model": "some/model", "messages": request.messages, "temperature": 0.7, "max_tokens": 99}

    limited = client.chat(request)
    assert limited.status == 429 and limited.error == 'HTTP 429: {"error": "slow down, [hidden]"}'
    moved = client.chat(request)
    assert moved.status == 302 and moved.error.startswith("HTTP 302") and len(InMemoryHTTPS.seen) == 3     # not followed
    assert client.models() == ["one", "two"] and InMemoryHTTPS.seen[3][:2] == (BASE + "/models", "GET")
    assert not InMemoryHTTPS.script

    # the control: urllib's default redirect handling would have gone on to the other address with the key
    InMemoryHTTPS.seen = []
    InMemoryHTTPS.script = [(302, {"Location": elsewhere}, b"moved"), (200, {}, completion("x"))]
    client._opener = urllib.request.build_opener(InMemoryHTTPS)
    assert client.chat(request).error == ""
    assert InMemoryHTTPS.seen[1][0] == elsewhere and InMemoryHTTPS.seen[1][2]["authorization"] == "Bearer " + KEY
    printed = capsys.readouterr()
    assert KEY not in printed.out + printed.err


def test_list_models_prints_the_ids(endpoint, capsys):
    endpoint.reply = staticmethod(lambda r: json.dumps({"object": "list", "data": [
        {"id": "@cf/openai/gpt-oss-120b"}, {"id": "@cf/meta/llama-4"}, {"name": "named-only"}, "bare-id"]}).encode())
    assert authored.main(["--list-models"]) == 0
    printed = capsys.readouterr()
    assert printed.out.split() == ["@cf/openai/gpt-oss-120b", "@cf/meta/llama-4", "named-only", "bare-id"]
    (call,) = endpoint.calls
    assert call["url"] == BASE + "/models" and call["method"] == "GET" and call["body"] is None
    assert call["headers"]["authorization"] == "Bearer " + KEY
    assert KEY not in printed.out + printed.err and "llm.invalid" not in printed.out + printed.err

    endpoint.reply = staticmethod(lambda r: http_error(403, f"forbidden for {KEY}"))
    with pytest.raises(SystemExit) as stop:
        authored.main(["--list-models"])
    assert "HTTP 403" in str(stop.value) and KEY not in str(stop.value)
    assert authored.main(["--list-models", "--dry-run"]) == 0
    assert capsys.readouterr().out.split() == list(authored.CannedClient.MODELS)
    assert len(endpoint.calls) == 2                                      # the dry run asked nobody
    endpoint.reply = staticmethod(lambda r: json.dumps({"data": {"unexpected": "shape"}}).encode())
    assert authored.main(["--list-models"]) == 1 and "listed no model ids" in capsys.readouterr().out


def test_real_run_against_a_stand_in_endpoint_leaves_no_secret_anywhere(endpoint, tmp_path, capsys):
    """The whole command without --dry-run: requests are built and sent (to the stand-in), replies
    parsed, files written. Neither the key nor the address ends up in any output."""
    canned = authored.canned_replies()
    served = {tier: iter(replies) for tier, replies in canned.items()}

    def reply(request):
        body = json.loads(request.data)
        tier = int(re.search(r"maps for tier (\d)", body["messages"][1]["content"]).group(1))
        return completion(next(served[tier], "nothing more"), usage=(1000, 3000))
    endpoint.reply = staticmethod(reply)
    out = tmp_path / "authored_test.jsonl"
    code = authored.main(["--per-tier", "2", "--out", str(out), "--concurrency", "2", "--no-engine-check",
                          "--models", "model-a,model-b", "--summary-json", str(tmp_path / "logs" / "summary.json"),
                          "--save-replies", str(tmp_path / "replies"), "--max-requests", "20", "--patience", "2"])
    printed = capsys.readouterr()
    assert code == 0
    rows = cli.read_rows(out)
    assert sorted(row["grid"] for row in rows) == sorted(g.strip() for t in authored.TIERS for g in GOOD[t])
    assert {row["model"] for row in rows} == {"model-a", "model-b"}
    assert all(call["url"] == BASE + "/chat/completions" and call["method"] == "POST" for call in endpoint.calls)
    assert all(set(call["body"]) == {"model", "messages", "temperature", "max_tokens"} for call in endpoint.calls)
    assert 9 <= len(endpoint.calls) <= 20
    assert f"requests made: {len(endpoint.calls)} of at most 20" in printed.out
    sent = len(endpoint.calls)
    assert f"tokens reported by the endpoint: {1000 * sent} prompt, {3000 * sent} completion" in printed.out
    saved = sorted((tmp_path / "replies").glob("*.txt"))
    assert len(saved) == len(endpoint.calls) and "brief: " in saved[0].read_text()
    written = [out, tmp_path / "logs" / "summary.json", *saved]
    for text in [printed.out, printed.err] + [path.read_text() for path in written]:
        assert KEY not in text and "llm.invalid" not in text and "Bearer" not in text and "uthorization" not in text
    assert "Draw 5 different maps" not in "".join(path.read_text() for path in saved)      # replies only, no request


# ---- the file itself --------------------------------------------------------------------------------
def test_module_uses_the_standard_library_only():
    tree = ast.parse(MODULE.read_text())
    top = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add(node.module.split(".")[0])
    import sys
    assert top <= set(sys.stdlib_module_names), top - set(sys.stdlib_module_names)
    relative = {(node.level, node.module) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level}
    assert relative == {(1, None), (1, "solver")}                         # the generator package, nothing else
    assert "urllib" in top and not top & {"requests", "httpx", "http", "subprocess", "socket"}


def test_files_keep_the_repository_rules():
    banned = ["a" + "tan2", "arc" + "tan2", "reverse" + " engineer", "reverse-" + "engineer", "organi" + "zer",
              "organi" + "ser"]
    key_like = re.compile(r"\b" + "cr" + "_" + r"[A-Za-z0-9]+")
    dot_file = re.compile(r"\.e" + r"nv\b")                              # the file that holds the real key
    for path in (MODULE, Path(__file__)):
        text = path.read_text()
        low = text.lower()
        assert not [word for word in banned if word in low], path.name
        assert not key_like.findall(text) and not dot_file.findall(low), path.name
    assert "fix" + "ture" not in MODULE.read_text().lower()                # the maps to compare with are "reference maps"
    source = MODULE.read_text()
    assert source.count("os.environ") == 1 and "getenv" not in source      # one place reads the environment
    assert "Training-time scaffolding, never imported by the agent." in source.split('"""')[1]
    assert "print(self._key" not in source and "print(request" not in source
