"""Tests for map metadata, the episode scorer, the runner and evaluate.py.

The scorer tests replay short button scripts on the small grid-layout maps under
`tests/fixtures/` and check the result against numbers worked out by hand from the layout.
"""

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from doom_env import limits as L
from doom_env import make_env
from doom_env.wad import Lump, read_wad, write_wad
from fixtures import FIXTURES, fixture_wad
from training import evaluate as E
from training import mapmeta as M
from training import runner as R
from training import scorer as S

ROOT = Path(__file__).resolve().parents[2]
TWO_ROOMS = str(ROOT / "maps" / "two_rooms.wad")

# The layouts of the grid maps used below (the WAD of each is tests/fixtures/<sha1 of the text>.wad).
EXIT = "#######\n#P...E#\n#######"
KEY = "############\n#P.r..R...E#\n############"
GUNNER = "##########\n#P......c#\n##########"
DOOR = "##########\n#P..D...E#\n##########"
ROOM = "#####\n#P..#\n#####"
BLUE = "#######\n#P..B.E#\n#######"
SIDE_DOOR = "#########\n#P.....z#\n#..D....#\n#########"
ALL_GRIDS = (
    EXIT, KEY, GUNNER, DOOR, ROOM, BLUE, SIDE_DOOR, "####\n#P.#\n####", "#####\n#P.z#\n#####",
    "#########\n#P..h..z#\n#########", "###########\n#z.P..D..z#\n###########", "##########\n#P......i#\n##########",
    "\n".join(["#" * 13] * 3 + ["####" + row for row in SIDE_DOOR.splitlines()]),
)


def play(grid_or_wad, script, ticks=2100, seed=0, report=None):
    """Replay a button script (see runner.ScriptPolicy) and return (result row, scorer)."""
    wad = fixture_wad(grid_or_wad) if "\n" in grid_or_wad else grid_or_wad
    env = make_env(wad, episode_ticks=ticks, render_buffers=False)
    try:
        obs, _ = env.reset(seed=seed)
        scorer = S.EpisodeScorer(env, M.load_meta(wad))
        policy = R.ScriptPolicy(script)
        for _ in range(len(policy.actions)):
            if report is not None:
                scorer.agent_report(**report(env))
            _, _, terminated, truncated, _ = env.step(policy.act(obs))
            scorer.step()
            if terminated or truncated:
                break
        return scorer.result(), scorer
    finally:
        env.close()


def variant(tmp_path, grid, old, new, name="variant.wad"):
    """Copy a fixture WAD with one piece of its TEXTMAP replaced (for example the start angle)."""
    out = []
    for lump in read_wad(fixture_wad(grid)):
        if lump.name == "TEXTMAP":
            text = lump.data.decode()
            assert old in text
            lump = Lump(lump.name, text.replace(old, new).encode())
        out.append(lump)
    path = tmp_path / name
    write_wad(path, out)
    return str(path)


def one_ring(grid):
    """A grid with ragged rows padded and all-wall border rows/columns reduced to a single ring."""
    rows = grid.splitlines()
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, "#") for r in rows]
    while len(rows) > 2 and set(rows[1]) == {"#"}:
        rows.pop(0)
    while len(rows) > 2 and set(rows[-2]) == {"#"}:
        rows.pop()
    while all(r[1] == "#" for r in rows):
        rows = [r[1:] for r in rows]
    while all(r[-2] == "#" for r in rows):
        rows = [r[:-1] for r in rows]
    return "\n".join(rows)


def strip_timing(rows):
    """Rows without the wall-clock fields, in a form where NaN compares equal to NaN."""
    return [{k: repr(v) for k, v in r.items() if k not in R.TIMING_FIELDS} for r in rows]


# =================================================================================== mapmeta
@pytest.mark.parametrize("grid", ALL_GRIDS, ids=lambda g: hashlib.sha1(g.encode()).hexdigest()[:8])
def test_grid_is_recovered_from_the_wad(grid):
    meta = M.load_meta(fixture_wad(grid))
    assert meta.grid is not None and meta.format == "udmf"
    assert meta.grid.ascii(meta.all_things) == one_ring(grid)
    # every thing sits at the centre of the cell that shows its symbol
    for thing in meta.all_things:
        row, col = meta.grid.cell_of(thing.x, thing.y)
        assert meta.grid.centre(row, col) == (thing.x, thing.y)
    assert meta.grid.open_cells == sum(ch not in "#\n" for ch in grid)


def test_meta_of_key_and_locked_door_map():
    meta = M.load_meta(fixture_wad(KEY))
    assert meta.player_start == (96.0, 96.0, 0.0)
    assert [(e.sector, e.centre, e.trigger) for e in meta.exits] == [(3, (672.0, 96.0), "cross")]
    assert meta.exits[0].polygon.shape == (4, 4) and meta.exits[0].lines.tolist() == [[640.0, 128.0, 640.0, 64.0]]
    assert [(d.sector, d.lock, d.colour, d.centre, d.delay_tics) for d in meta.doors] == [(1, 129, "red", (416.0, 96.0), 0)]
    assert meta.key_classes == ("RedCard",) and meta.keys_total == 1
    assert [(t.cls, t.x, t.y) for t in meta.keys] == [("RedCard", 224.0, 96.0)]
    assert (meta.doors_total, meta.locked_total, meta.monsters_total, meta.has_exit) == (0, 1, 0, True)
    assert meta.door_lock.tolist() == [-1, 129, -1, -1]
    assert sorted(meta.sector_polygons) == [0, 1, 2, 3]
    assert meta.things_by_class()["RedCard"].tolist() == [[224.0, 96.0]]


def test_meta_of_two_rooms():
    meta = M.load_meta(TWO_ROOMS)
    assert meta.player_start == (160.0, 416.0, 0.0)
    assert [(e.sector, e.centre, len(e.lines)) for e in meta.exits] == [(3, (672.0, 96.0), 3)]
    assert [(d.sector, d.lock, d.delay_tics) for d in meta.doors] == [(2, 0, 150)]
    assert {k: sum(c.values()) for k, c in meta.census().items() if c} == {"monster": 1, "health": 1}
    assert meta.grid.shape == (9, 14)
    assert meta.grid.ascii(meta.all_things).splitlines()[3] == "#......D.....#"
    assert meta.grid.sector_at(160.0, 416.0) == 0 and meta.grid.sector_at(672.0, 96.0) == 3
    assert meta.grid.sector_at(10.0, 10.0) == -1 and meta.grid.sector_at(-500.0, 9999.0) == -1


def test_start_room_whose_only_opening_is_the_exit():
    """Both sides of the single exit line are "enclosed"; the side the player starts in is not the exit."""
    meta = M.load_meta(fixture_wad(EXIT))
    assert [(e.sector, e.centre) for e in meta.exits] == [(1, (352.0, 96.0))]


@pytest.mark.parametrize("grid", [KEY, DOOR, SIDE_DOOR, GUNNER, "#########\n#P..h..z#\n#########"])
def test_wad_things_match_the_engine(grid):
    wad = fixture_wad(grid)
    env = make_env(wad, episode_ticks=100, render_buffers=False)
    try:
        env.reset(seed=0)
        meta = M.load_meta(wad)
        assert M.cross_check(meta, env) == []
        assert M.engine_census(env) == meta.census()
        assert [sid for sid, _ in env.sector_polygons()] == list(meta.sector_polygons)
        assert meta.door_lock.tolist() == env.privileged_state()["door_lock"].tolist()
    finally:
        env.close()


def test_freedoom_maps():
    """Doom-format maps: all 32 load, are not grid maps, and (bar the boss map) have an exit."""
    import vizdoom

    freedoom = os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad")
    names = M.wad_maps(freedoom)
    assert len(names) == 32
    for name in names:
        meta = M.load_meta(freedoom, name)
        assert meta.format == "doom" and meta.grid is None and meta.num_sectors <= M.MAX_SECTORS
        assert meta.sector_at(*meta.player_start[:2]) >= 0
        assert meta.has_exit == (name != "MAP30")
        assert all(d.colour in ("red", "blue", "yellow") for d in meta.doors if d.locked)
    # exits of a classic map are lines (a switch here); the engine agrees on every monster and item
    meta = M.load_meta(freedoom, "MAP01")
    assert [(e.sector, e.trigger) for e in meta.exits] == [(-1, "use")]
    assert meta.monsters_total == 18 and meta.census()["weapon"] == {"Shotgun": 1, "Chainsaw": 1}
    env = make_env(freedoom, "MAP01", episode_ticks=100, render_buffers=False)
    try:
        env.reset(seed=0)
        assert M.cross_check(meta, env) == []
        result = S.EpisodeScorer(env, meta).result()
        assert (result["sectors_total"], result["monsters_total"], result["has_exit"]) == (211, 18, 1)
        assert math.isnan(result["cells_total"])
    finally:
        env.close()


def test_maps_over_the_sector_limit_are_refused(monkeypatch):
    monkeypatch.setattr(M, "MAX_SECTORS", 3)
    M._CACHE.clear()
    with pytest.raises(M.MapTooLarge, match="4 sectors"):
        M.load_meta(fixture_wad(KEY))
    M._CACHE.clear()
    rows = R.run_episodes("training.runner:RandomPolicy", [fixture_wad(KEY)], workers=0)
    assert rows[0]["error"].startswith("skipped:") and rows[0]["score"] == 0.0


def test_locks():
    assert [M.lock_colour(n) for n in (0, 1, 2, 3, 129, 130, 131, 100, 101, 77)] == \
        ["", "red", "blue", "yellow", "red", "blue", "yellow", "any", "all", "other"]
    none, red_card, red_skull = [0] * 6, [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]   # blue, yellow, red cards, then skulls
    assert M.lock_satisfied(0, none) and not M.lock_satisfied(129, none)
    assert M.lock_satisfied(129, red_card) and M.lock_satisfied(129, red_skull) and not M.lock_satisfied(130, red_card)
    assert M.lock_satisfied(100, red_skull) and not M.lock_satisfied(100, none)
    assert not M.lock_satisfied(101, red_card) and M.lock_satisfied(101, [1, 1, 1, 0, 0, 0])


def test_sector_locator_with_a_pillar_and_a_nested_sector():
    def box(x0, y0, x1, y1):
        return np.array([[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]], float)

    pillar, pool = box(100, 100, 150, 150), box(300, 300, 400, 400)
    room = np.concatenate([box(0, 0, 500, 500), pillar, pool])   # the room's outline includes both holes
    loc = M.SectorLocator([(7, room), (9, pool)])
    assert loc.locate(50, 50) == 7 and loc.locate(125, 125) == -1 and loc.locate(350, 350) == 9
    assert loc.locate(350, 350, hint=7) == 9 and loc.locate(50, 50, hint=9) == 7 and loc.locate(600, 50) == -1
    assert loc.locate_many([[50, 50], [125, 125], [350, 350], [-5, 0]]).tolist() == [7, -1, 9, -1]
    assert loc.distance(9, 250, 350) == pytest.approx(50.0) and loc.distance(123, 0, 0) == math.inf
    assert len(loc) == 2 and len(M.SectorLocator([])) == 0 and M.SectorLocator([]).locate(0, 0) == -1


def test_mapmeta_command_line(capsys):
    assert M.main([fixture_wad(KEY)]) == 0
    out = capsys.readouterr().out
    assert "#P.r..R...E#" in out and "locked_doors: 1" in out


# ==================================================================================== scorer
def test_walk_to_the_exit():
    """#P...E#: five open cells, two sectors, nothing to collect or fight."""
    r, _ = play(EXIT, "F*100")
    assert (r["exit"], r["dead"], r["timeout"], r["has_exit"]) == (1, 0, 0, 1)
    assert r["steps"] == 12 and r["tic"] == 48            # 11 steps of 4 tics, then the step that ends the level
    assert (r["sectors_visited"], r["sectors_total"]) == (2, 2) and (r["cells_visited"], r["cells_total"]) == (5, 5)
    assert r["health"] == 100.0 and r["damage_taken"] == 0.0 and r["death_tic"] == -1
    assert (r["keys_total"], r["doors_total"], r["locked_total"], r["monsters_total"]) == (0, 0, 0, 0)
    assert r["stuck_rate"] == 0.0 and r["door_reversals"] == 0
    assert r["exit_visible_tic"] == 0 and r["exit_dist_min"] < 32.0   # in view from the start; ends on the pad
    eff = 0.5 * 1.0 + 0.5 * (1 - 48 / 2100)
    assert r["c_exit"] == 1.0 and r["c_eff"] == pytest.approx(eff) and r["c_expl"] == 1.0
    assert math.isnan(r["c_prog"]) and math.isnan(r["c_combat"])   # absent categories are dropped
    assert r["score"] == pytest.approx(100 * (0.45 + 0.05 * eff + 0.15) / 0.65)
    assert r["score_exit_heavy"] == pytest.approx(100 * (0.60 + 0.05 * eff + 0.10) / 0.75)
    assert r["score_balanced"] == pytest.approx(100 * (0.30 + 0.05 * eff + 0.20) / 0.55)
    assert list(r) == list(S.RESULT_FIELDS)


def test_key_then_locked_door():
    """#P.r..R...E#: walk over the red card, open the red door once, walk to the exit."""
    r, _ = play(KEY, "F*34 U N*8 F*60")
    assert r["exit"] == 1 and (r["keys_picked"], r["keys_total"]) == (1, 1)
    assert (r["locked_opened"], r["locked_total"]) == (1, 1) and (r["doors_opened"], r["doors_total"]) == (0, 0)
    assert (r["sectors_visited"], r["sectors_total"]) == (4, 4) and (r["cells_visited"], r["cells_total"]) == (10, 10)
    assert r["door_reversals"] == 0
    assert r["stuck_rate"] == 0.0 and r["stuck_at_door_rate"] > 0.3   # the wait at the door is not "stuck"
    assert r["exit_visible_tic"] == 4 * 35                             # hidden until the door starts to rise (step 35)
    eff = 0.5 + 0.5 * (1 - r["tic"] / 2100)
    assert r["c_prog"] == 1.0 and r["c_eff"] == pytest.approx(eff)
    assert r["score"] == pytest.approx(100 * (0.45 + 0.05 * eff + 0.20 + 0.15) / 0.85)
    # stopping short of the key: nothing picked up, nothing opened, one sector of four
    r, _ = play(KEY, "F*4 N*10")
    assert (r["keys_picked"], r["locked_opened"], r["exit"], r["sectors_visited"]) == (0, 0, 0, 1)
    assert r["c_prog"] == 0.0 and r["score"] == pytest.approx(100 * (0.15 * 0.25) / 0.85)


def test_locked_door_without_its_key():
    """#P..B.E#: the blue door cannot be opened, so progression and the exit stay at zero."""
    r, _ = play(BLUE, "F*20 U N*8 F*20 U N*10")
    assert (r["exit"], r["locked_opened"], r["locked_total"], r["keys_total"]) == (0, 0, 1, 0)
    assert (r["sectors_visited"], r["sectors_total"]) == (1, 4) and r["exit_visible_tic"] == -1
    assert r["c_prog"] == 0.0 and r["score"] == pytest.approx(100 * (0.15 * 0.25) / 0.85)


def test_stand_still_and_die_to_the_chaingunner():
    """#P......c#: no exit, one monster; the player is shot dead well before the tick cap."""
    r, _ = play(GUNNER, "N*600", seed=0)
    assert (r["dead"], r["exit"], r["timeout"], r["has_exit"]) == (1, 0, 0, 0)
    assert (r["kills"], r["monsters_total"]) == (0, 1) and r["damage_taken"] >= 100.0 and r["health"] == 0.0
    assert r["steps"] == 65 and r["tic"] == r["death_tic"] == 259
    combat = 0.0 - 0.2 * 1.0 - 0.3 * (1 - 259 / 2100)
    assert r["c_combat"] == pytest.approx(combat) and math.isnan(r["c_exit"]) and math.isnan(r["c_eff"])
    assert r["score"] == pytest.approx(100 * (0.15 * 1.0 + 0.15 * combat) / 0.30)   # one sector, already visited
    assert math.isnan(r["exit_visible_tic"]) and math.isnan(r["exit_dist_min"])


def test_door_pressed_twice_is_one_reversal():
    """#P..D...E#: USE, two idle steps, USE again while the door is still rising."""
    r, scorer = play(DOOR, "F*12 N*6 U N*2 U N*30")
    assert r["door_reversals"] == 1
    assert (r["doors_opened"], r["doors_total"]) == (0, 1)     # it reached 24 units, not the 56 that count
    assert scorer._door_max.tolist() == [24.0] and r["c_prog"] == 0.0
    # pressed again 6 steps after the first press: the door has just reached 56 units, then turns round
    r, _ = play(DOOR, "F*12 N*6 U N*5 U N*30")
    assert r["door_reversals"] == 1 and r["doors_opened"] == 0
    r, _ = play(DOOR, "F*12 N*6 U N*6 U N*30")
    assert r["door_reversals"] == 1 and r["doors_opened"] == 1


def test_door_left_alone_or_held_is_no_reversal():
    # one press, then the whole cycle: up in 16 steps, open for 150 tics, down again
    r, scorer = play(DOOR, "F*12 N*6 U N*100")
    assert r["door_reversals"] == 0 and (r["doors_opened"], r["doors_total"]) == (1, 1)
    assert scorer._door_max.tolist() == [124.0] and scorer._opening.tolist() == [0.0]
    assert r["c_prog"] == 1.0 and r["score"] == pytest.approx(100 * (0.20 * 1.0 + 0.15 * 0.25) / 0.85)
    # holding USE down is a single press
    r, _ = play(DOOR, "F*12 N*6 U*12 N*10")
    assert r["door_reversals"] == 0 and r["doors_opened"] == 1


def test_stuck_rate():
    r, _ = play(ROOM, "N*50")
    assert r["stuck_rate"] == pytest.approx(41 / 50)        # steps 10..50: no movement over the last 10 steps
    r, _ = play(ROOM, "B*60")
    assert r["stuck_rate"] > 0.75                           # backing into the wall
    r, _ = play(EXIT, "F*9")
    assert r["stuck_rate"] == 0.0
    # standing in front of a closed door is waiting, not stuck ...
    r, _ = play(DOOR, "F*12 N*40")
    assert r["stuck_rate"] == 0.0 and r["stuck_at_door_rate"] == pytest.approx(34 / 52)
    # ... but standing at the start, 160 units from it, is
    r, _ = play(DOOR, "N*40")
    assert r["stuck_rate"] == pytest.approx(31 / 40) and r["stuck_at_door_rate"] == 0.0


def test_timeout_with_the_exit_in_view():
    r, _ = play(EXIT, "N*30", ticks=80)
    assert (r["timeout"], r["exit"], r["steps"], r["tic"]) == (1, 0, 20, 80)
    assert r["exit_visible_tic"] == 0 and r["exit_visible_timeout"] == 1.0
    assert math.isnan(r["exit_seen_timeout"]) and math.isnan(r["exit_believed_tic"])   # the agent never reported
    assert r["score"] == pytest.approx(100 * (0.15 * 0.5) / 0.65)


def test_exit_behind_the_player_or_a_closed_door_is_not_visible(tmp_path):
    back = variant(tmp_path, EXIT, "angle = 0; type = 1;", "angle = 180; type = 1;")
    r, _ = play(back, "N*20")
    assert r["exit_visible_tic"] == -1 and r["exit_visible_timeout"] == 0.0
    r, _ = play(back, "L*40")                               # turning round brings it into view
    assert r["exit_visible_tic"] > 0
    r, _ = play(DOOR, "F*12 N*20")
    assert r["exit_visible_tic"] == -1


def test_agent_report_of_the_exit():
    seen_from = 5
    calls = []

    def report(env):
        calls.append(1)
        return {"exit_believed": len(calls) > seen_from}

    r, _ = play(EXIT, "N*30", ticks=80, report=report)
    assert r["exit_believed_tic"] == 4 * seen_from and r["exit_seen_timeout"] == 1.0
    r, _ = play(EXIT, "N*30", ticks=80, report=lambda env: {"exit_believed": False})
    assert r["exit_believed_tic"] == -1 and r["exit_seen_timeout"] == 0.0
    calls.clear()
    r, _ = play(EXIT, "F*100", report=report)               # believed and reached: not a timeout
    assert r["exit"] == 1 and r["exit_seen_timeout"] == 0.0 and r["exit_believed_tic"] == 4 * seen_from


def test_odometry_drift_in_the_start_frame(tmp_path):
    # Facing east at the start: forward is +x on the map, so (x - 96, 0, 0) is the exact pose.
    exact = lambda env: {"pose": (env.privileged_state()["x"] - 96.0, 0.0, 0.0)}
    r, _ = play(EXIT, "F*8", report=exact)
    assert r["odom_err_max"] == pytest.approx(0.0, abs=1e-9) and r["odom_heading_err_max"] == pytest.approx(0.0, abs=1e-9)
    r, _ = play(EXIT, "F*8", report=lambda env: {"pose": (env.privileged_state()["x"] - 96.0, 3.0, 0.1)})
    assert r["odom_err_final"] == r["odom_err_mean"] == r["odom_err_max"] == pytest.approx(3.0)
    assert r["odom_heading_err_max"] == pytest.approx(math.degrees(0.1))
    # no report at all: the fields are empty
    r, _ = play(EXIT, "F*8")
    assert math.isnan(r["odom_err_final"]) and math.isnan(r["odom_heading_err_max"])
    # Facing north at the start: a strafe to the right moves east on the map, which is -y (y is left).
    north = variant(tmp_path, EXIT, "angle = 0; type = 1;", "angle = 90; type = 1;")
    r, scorer = play(north, ">*8", report=lambda env: {"pose": (0.0, -(env.privileged_state()["x"] - 96.0), 0.0)})
    assert scorer._xs[-1] > 96.0 + 32.0 and r["odom_err_max"] == pytest.approx(0.0, abs=1e-6)
    # Turning left is counter-clockwise: theta grows with the engine angle.
    r, _ = play(EXIT, "L*10", report=lambda env: {"pose": (0.0, 0.0, math.radians(env.privileged_state()["angle"]))})
    assert r["odom_heading_err_max"] == pytest.approx(0.0, abs=1e-6)


def test_components_and_weights_by_hand():
    base = dict(episode_ticks=2000, tic=500, has_exit=1, exit=1, health=60.0, dead=0, damage_taken=40.0,
                keys_picked=1, keys_total=2, doors_opened=3, doors_total=4, locked_opened=0, locked_total=0,
                sectors_visited=3, sectors_total=12, kills=2, monsters_total=5)
    c = S.components(base)
    assert c == pytest.approx({"exit": 1.0, "eff": 0.5 * 0.6 + 0.5 * 0.75, "prog": (0.5 + 0.75) / 2, "expl": 0.25,
                               "combat": 0.4 - 0.2 * 0.4})
    assert S.composite(c) == pytest.approx(100 * (0.45 + 0.05 * 0.675 + 0.20 * 0.625 + 0.15 * 0.25 + 0.15 * 0.32))
    assert sum(S.WEIGHTS["default"].values()) == pytest.approx(1.0)
    for name in ("exit_heavy", "balanced"):
        assert sum(S.WEIGHTS[name].values()) == pytest.approx(1.0) and set(S.WEIGHTS[name]) == set(S.COMPONENTS)
    # dying early with nothing killed hits the floor of -0.5
    dead = S.components({**base, "exit": 0, "dead": 1, "tic": 0, "kills": 0, "damage_taken": 300.0, "health": -20.0})
    assert dead["combat"] == -0.5 and dead["exit"] == 0.0 and dead["eff"] == 0.0
    # health above 100 and time past the cap are clipped
    assert S.components({**base, "health": 200.0, "tic": 9999})["eff"] == pytest.approx(0.5)
    # a map with no exit, nothing to progress through and no monsters is scored on exploration alone
    bare = S.components({**base, "has_exit": 0, "keys_total": 0, "doors_total": 0, "monsters_total": 0})
    assert [k for k, v in bare.items() if not math.isnan(v)] == ["expl"] and S.composite(bare) == pytest.approx(25.0)
    assert S.composite({}) == 0.0


def test_suite_score_weights_tiers_and_reports_standard_errors():
    def row(tier, score, exit_=0, has_exit=1, error=""):
        return {"tier": tier, "score": score, "score_exit_heavy": score, "score_balanced": score, "exit": exit_,
                "has_exit": has_exit, "error": error, "map_id": f"m{tier}-{score}"}

    rows = [row(1, 10.0, 1), row(1, 30.0, 0), row(2, 50.0, 1), row(2, 70.0, 1), row(4, 100.0, 1), row(0, 99.0, 0),
            row(1, 0.0, 0, has_exit=0)]
    total = S.suite_score(rows)
    # weights 1, 2, 4 per episode; tier 0 is summarised but carries no weight
    weighted = (1 * (10 + 30 + 0) + 2 * (50 + 70) + 4 * 100) / (1 * 3 + 2 * 2 + 4 * 1)
    assert total["score"] == pytest.approx(weighted) and total["episodes"] == 7 and sorted(total["tiers"]) == [0, 1, 2, 4]
    var1, var2 = np.var([10, 30, 0], ddof=1), np.var([50, 70], ddof=1)
    assert total["score_se"] == pytest.approx(math.sqrt(1 * 3 * var1 + 4 * 2 * var2) / 11)
    t1 = total["tiers"][1]
    assert t1["episodes"] == 3 and t1["score"] == pytest.approx(40 / 3)
    assert t1["exit_rate"] == pytest.approx(0.5) and t1["exit_rate_se"] == pytest.approx(math.sqrt(0.25 / 2))  # 2 maps have an exit
    assert total["tiers"][2]["exit_rate"] == 1.0 and total["tiers"][2]["exit_rate_se"] == 0.0
    assert list(t1) == list(S.SUMMARY_FIELDS)
    # untiered maps only: the plain mean
    assert S.suite_score([row(0, 20.0), row(0, 40.0)])["score"] == pytest.approx(30.0)
    # an episode that could not be played counts as zero
    failed = S.suite_score([row(3, 80.0, 1), {**row(3, 0.0), **S.empty_result(2100), "tier": 3, "error": "worker died"}])
    assert failed["score"] == pytest.approx(40.0) and failed["errors"] == 1 and failed["tiers"][3]["exit_rate"] == 0.5
    assert S.suite_score([])["episodes"] == 0
    # episodes in which the agent or one of its telemetry hooks failed are counted, each kind on its own
    counted = S.summarise([row(2, 10.0), {**row(2, 5.0), "agent_error": "step 3: ValueError: boom"},
                           {**row(2, 90.0, 1), "hook_error": "step 0: pose_estimate(): KeyError: 'x'"},
                           {**row(2, 0.0), "agent_error": "reset: timeout: reset() took longer than 10 s", "error": "void"}])
    assert (counted["episodes"], counted["errors"], counted["agent_errors"], counted["hook_errors"]) == (4, 1, 2, 1)


# ==================================================================================== runner
class CrashOnSeed:
    """Walks forward; kills its whole process at step 3 of the episodes with the given seed."""

    def __init__(self, seed=1, mode="exit", marker=None):
        self.bad_seed, self.mode, self.marker, self.current, self.t = seed, mode, marker, None, 0

    def seed(self, seed):
        self.current = seed

    def reset(self):
        self.t = 0
        if self.mode == "reset" and self.current == self.bad_seed:
            raise RuntimeError("reset failed")

    def act(self, obs):
        self.t += 1
        if self.t == 3 and self.current == self.bad_seed:
            if self.mode == "exit" and not (self.marker and os.path.exists(self.marker)):
                if self.marker:
                    Path(self.marker).touch()
                os._exit(7)
            if self.mode == "hang":
                if self.marker:
                    with open(self.marker, "a") as f:
                        f.write("hang\n")
                time.sleep(3600)
            if self.mode == "raise":
                raise ValueError("boom")
            if self.mode == "bad":
                return np.full(9, 2)
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)


def broken_factory():
    raise RuntimeError("weights missing")


class CountingRecorder:
    def begin(self, task, env, meta, policy, obs):
        self.steps, self.first_obs_ok = 0, obs["rays"]["distance"].shape == (64,)

    def step(self, t, obs, action, next_obs, reward, done, env, scorer):
        assert t == self.steps and action.shape == (9,) and scorer.steps == t + 1
        self.steps += 1

    def end(self, row):
        return {"rec_steps": self.steps, "rec_ok": int(self.first_obs_ok)}


RUN_MAPS = [
    {"wad": fixture_wad(KEY), "tier": 3, "id": "key", "episode_ticks": 400},
    {"wad": TWO_ROOMS, "tier": 2, "episode_ticks": 400},
    fixture_wad(GUNNER),
]


def test_runner_rows_are_deterministic_for_fixed_seeds():
    spec = ("training.runner:RandomPolicy", {"p": 0.4})
    pooled = R.run_episodes(spec, RUN_MAPS, workers=2, seeds=[0, 1], render_buffers=False)
    again = R.run_episodes(spec, RUN_MAPS, workers=3, seeds=[0, 1], render_buffers=False)
    inline = R.run_episodes(spec, RUN_MAPS, workers=0, seeds=[0, 1], render_buffers=False)
    assert strip_timing(pooled) == strip_timing(again) == strip_timing(inline)
    # rows come back in input order and carry the map id, tier, seed and every metric
    assert [(r["map_id"], r["tier"], r["seed"]) for r in pooled] == [
        ("key", 3, 0), ("key", 3, 1), ("two_rooms", 2, 0), ("two_rooms", 2, 1),
        (Path(fixture_wad(GUNNER)).stem, 0, 0), (Path(fixture_wad(GUNNER)).stem, 0, 1)]
    assert all(list(r) == list(R.ROW_FIELDS) for r in pooled)
    assert all(r["error"] == "" and r["agent_error"] == "" and r["census_ok"] == 1 for r in pooled)
    assert all(r["policy"] == 'training.runner:RandomPolicy{"p": 0.4}' for r in pooled)
    assert pooled[0]["episode_ticks"] == 400 and pooled[0]["steps"] == 100 and pooled[4]["episode_ticks"] == 2100
    # another seed is another episode
    assert strip_timing(pooled[0:1]) != [{**strip_timing(pooled[1:2])[0], "seed": "0"}]


def test_runner_scripted_policies_inline():
    rows = R.run_episodes("training.runner:CorridorPolicy", [fixture_wad(KEY), fixture_wad(DOOR)], workers=0)
    assert [(r["exit"], r["door_reversals"], r["stuck_rate"]) for r in rows] == [(1, 0, 0.0), (1, 0, 0.0)]
    assert rows[0]["keys_picked"] == 1 and rows[0]["locked_opened"] == 1 and rows[1]["doors_opened"] == 1
    assert rows[0]["score"] > 99.0 and rows[1]["score"] > 99.0
    # a ready policy object and a recorder object are accepted when everything runs in this process
    recorder = CountingRecorder()
    row = R.run_episodes(R.ScriptPolicy("F*100"), [fixture_wad(EXIT)], workers=0, record=recorder)[0]
    assert row["exit"] == 1 and row["rec_steps"] == row["steps"] == 12 and row["rec_ok"] == 1
    assert row["policy"] == "ScriptPolicy"
    with pytest.raises(TypeError, match="workers=0"):
        R.run_episodes(R.ScriptPolicy("F"), [fixture_wad(EXIT)], workers=1)


class PrivilegedWalker:
    """A training-only policy: it reads the true position and the exit through attach()."""

    def __init__(self):
        self.calls = []

    def attach(self, env, meta):
        self.calls.append("attach")
        self.env, self.meta = env, meta

    def seed(self, seed):
        self.calls.append(f"seed {seed}")

    def reset(self):
        self.calls.append("reset")
        self.goal_x = self.meta.exits[0].centre[0]     # attach() has already run

    def act(self, obs):
        action = np.zeros(9, np.int8)
        action[0] = self.env.privileged_state()["x"] < self.goal_x
        return action


def test_optional_policy_hooks_are_called_in_order():
    policy = PrivilegedWalker()
    rows = R.run_episodes(policy, [fixture_wad(EXIT)], workers=0, seeds=[3, 4])
    assert policy.calls == ["attach", "seed 3", "reset", "attach", "seed 4", "reset"]
    assert [r["exit"] for r in rows] == [1, 1] and [r["seed"] for r in rows] == [3, 4]


def test_policy_specs():
    assert R.parse_spec("training.runner:RandomPolicy") == ("training.runner:RandomPolicy", {})
    assert R.parse_spec(("a.b:C", {"p": 0.3})) == R.parse_spec({"factory": "a.b:C", "kwargs": {"p": 0.3}}) == ("a.b:C", {"p": 0.3})
    name = R.spec_name(("training.runner:RandomPolicy", {"p": 0.3, "attack": False}))
    assert name == 'training.runner:RandomPolicy{"attack": false, "p": 0.3}'
    assert R.parse_spec(name) == ("training.runner:RandomPolicy", {"attack": False, "p": 0.3})   # round trip
    policy = R.build(name)
    assert isinstance(policy, R.RandomPolicy) and policy.p == 0.3 and policy.attack is False
    policy.seed(5)
    policy.reset()
    first = [policy.act(None).tolist() for _ in range(4)]
    policy.reset()
    assert first == [policy.act(None).tolist() for _ in range(4)] and all(a[6] == 0 for a in first)
    with pytest.raises(ValueError, match="module:factory"):
        R.parse_spec("no_colon_here")


def test_script_policy_tokens():
    policy = R.ScriptPolicy("F*2 FU <A N R*1 w")
    assert [a.tolist() for a in policy.actions] == [
        [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0], [0] * 9, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]]
    policy.reset()
    assert [policy.act(None).tolist() for _ in range(8)][-1] == [0] * 9      # stands still once the script ends


def test_a_failing_agent_stands_still_and_is_still_scored():
    for mode, message in (("raise", "ValueError: boom"), ("bad", "invalid_action")):
        rows = R.run_episodes(CrashOnSeed(seed=0, mode=mode), [{"wad": fixture_wad(EXIT), "episode_ticks": 200}],
                              workers=0, seeds=[0, 1])
        failed, fine = rows
        assert message in failed["agent_error"] and failed["agent_error"].startswith("step 2:")
        assert failed["error"] == "" and failed["steps"] == 50 and failed["exit"] == 0 and failed["timeout"] == 1
        assert 0.0 < failed["score"] < 20.0
        assert fine["agent_error"] == "" and fine["exit"] == 1     # the next map starts with a working agent


# ---- time limits
FORWARD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
SHORT = {"wad": fixture_wad(EXIT), "episode_ticks": 60}      # 15 decision steps: a compute budget of 0.6 s
LONG = {"wad": fixture_wad(EXIT), "episode_ticks": 2100}     # 525 decision steps: 21 s, so the per-call limits bind first


class Clock:
    """Stands in for the runner's `time` module: a clock that only moves when a policy says how long it took."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class Costly:
    """Walks forward. Call n of act() takes costs[n] seconds on the clock (the last cost repeats), reset() takes reset_s."""

    def __init__(self, clock, costs=(0.0,), reset_s=0.0):
        self.clock, self.costs, self.reset_s, self.calls = clock, list(costs), reset_s, 0

    def reset(self):
        self.calls = 0
        self.clock.now += self.reset_s

    def act(self, obs):
        self.clock.now += self.costs[min(self.calls, len(self.costs) - 1)]
        self.calls += 1
        return FORWARD


@pytest.fixture
def arena():
    """arena(policy, map, seed=0, **run_episode options) -> row, on environments that stay open for the whole test."""
    envs = {}

    def play(policy, entry, seed=0, **options):
        task = {**R.normalise_map(entry), "seed": seed}
        key = (task["wad"], task["episode_ticks"])
        if key not in envs:
            envs[key] = (make_env(task["wad"], episode_ticks=task["episode_ticks"], render_buffers=False),
                         M.load_meta(task["wad"]))
        return R.run_episode(policy, *envs[key], task, **options)

    yield play
    for env, _ in envs.values():
        env.close()


@pytest.fixture
def timed(arena, monkeypatch):
    """timed(map, costs=..., reset_s=..., time_scale=1.0) -> (row, policy): the official limits on a clock the policy moves."""
    clock = Clock()
    monkeypatch.setattr(R, "time", clock)

    def play(entry, costs=(0.0,), reset_s=0.0, time_scale=1.0):
        clock.now = 0.0
        policy = Costly(clock, costs, reset_s)
        return arena(policy, entry, time_scale=time_scale), policy

    return play


def test_the_compute_budget_of_a_map_stops_a_slow_agent(timed, arena):
    """40 ms x 15 decision steps = 0.6 s. At 70 ms a call, eight calls fit and the ninth has 0.04 s left."""
    assert L.agent_budget_s(60) == 0.6 and L.agent_budget_s(2100) == 21.0
    row, policy = timed(SHORT, costs=[0.07])
    assert row["agent_error"] == ("step 8: timeout: act() ran past the rest of the agent's compute budget for this map "
                                  "(0.6 s = 40 ms x 15 decision steps)")
    assert policy.calls == 9 and R.is_timeout(row) and row["error"] == ""
    assert (row["exit"], row["timeout"], row["steps"]) == (0, 1, 15)
    assert row["act_ms_mean"] == pytest.approx(70.0) and row["act_ms_max"] == pytest.approx(70.0)
    # the ninth action came too late and was not used: the player ends where eight steps forward take it
    eight = arena(R.ScriptPolicy("F*8"), SHORT)
    assert row["exit_dist_final"] == eight["exit_dist_final"] and row["score"] == eight["score"]
    # without the limits (the runner's default, for training-only policies) the same agent walks out
    free, policy = timed(SHORT, costs=[0.07], time_scale=None)
    assert (free["exit"], free["steps"], free["agent_error"], policy.calls) == (1, 12, "", 12) and not R.is_timeout(free)
    # a call that uses up exactly what is left is still in time; the next one is not made at all
    row, policy = timed(SHORT, costs=[0.6, 0.0])
    assert row["agent_error"] == ("step 1: timeout: the agent's compute budget for this map (0.6 s = 40 ms x 15 decision "
                                  "steps) is used up") and policy.calls == 1 and row["exit"] == 0
    # a fast agent is not touched, and the timing columns report what was measured
    row, policy = timed(SHORT, costs=[0.03])
    assert (row["exit"], row["steps"], row["agent_error"]) == (1, 12, "") and row["act_ms_mean"] == pytest.approx(30.0)


def test_the_limits_on_single_calls(timed, arena):
    """10 s for reset() and for the first act() of a map, 5 s for every other act()."""
    row, policy = timed(LONG, costs=[9.0, 0.0, 0.0, 4.9, 0.0])
    assert (row["exit"], row["agent_error"], policy.calls) == (1, "", 12)
    row, policy = timed(LONG, costs=[9.0, 0.0, 0.0, 6.0, 0.0])
    assert row["agent_error"] == "step 3: timeout: act() took longer than 5 s" and policy.calls == 4
    assert (row["exit"], row["timeout"], row["steps"]) == (0, 1, 525) and row["act_ms_max"] == pytest.approx(9000.0)
    row, policy = timed(LONG, costs=[11.0, 0.0])
    assert row["agent_error"] == "step 0: timeout: the first act() of the map took longer than 10 s" and policy.calls == 1
    assert (row["exit"], row["steps"], row["error"]) == (0, 525, "")
    # reset(): inside the limit the map is played, beyond it the map is void, as in the evaluation (0 points)
    row, policy = timed(LONG, reset_s=9.0)
    assert (row["exit"], row["agent_error"], row["error"]) == (1, "", "")
    row, policy = timed(LONG, reset_s=11.0)
    assert row["agent_error"] == "reset: timeout: reset() took longer than 10 s" and R.is_timeout(row)
    assert row["error"] == "the map is void (0 points): timeout: reset() took longer than 10 s" and policy.calls == 0
    assert (row["score"], row["exit"], row["has_exit"], row["steps"]) == (0.0, 0, 1, 0) and list(row) == list(R.ROW_FIELDS)
    # an exception in reset() is not a timeout: the agent stops, stands still and is scored (as before)
    row = arena(CrashOnSeed(seed=0, mode="reset"), SHORT, time_scale=1.0)
    assert row["agent_error"] == "reset: RuntimeError: reset failed" and row["error"] == "" and row["steps"] == 15


def test_time_scale_judges_the_agent_as_a_slower_machine_would(timed):
    """30 ms a call is inside the 0.6 s budget here (12 calls to the exit: 0.36 s) and outside it at a third of the speed."""
    row, _ = timed(SHORT, costs=[0.03], time_scale=1.0)
    assert (row["exit"], row["agent_error"]) == (1, "")
    row, policy = timed(SHORT, costs=[0.03], time_scale=3.0)       # 90 ms a call: six fit, the seventh has 0.06 s left
    assert row["agent_error"] == ("step 6: timeout: act() ran past the rest of the agent's compute budget for this map "
                                  "(0.6 s = 40 ms x 15 decision steps) (measured time x 3)")
    assert policy.calls == 7 and row["exit"] == 0 and row["act_ms_mean"] == pytest.approx(30.0)   # reported unscaled
    row, _ = timed(LONG, costs=[0.0, 2.0, 0.0], time_scale=3.0)
    assert row["agent_error"] == "step 1: timeout: act() took longer than 5 s (measured time x 3)"
    row, _ = timed(LONG, reset_s=6.0, time_scale=0.5)              # a machine twice as fast: 3 s
    assert row["error"] == "" and row["exit"] == 1
    row, _ = timed(LONG, reset_s=6.0, time_scale=2.0)
    assert row["agent_error"] == "reset: timeout: reset() took longer than 10 s (measured time x 2)" and row["score"] == 0.0
    with pytest.raises(ValueError, match="time_scale must be positive"):
        R.run_episodes(R.ScriptPolicy("F"), [SHORT], workers=0, time_limits=True, time_scale=0.0)


def official_loop(costs, ticks):
    """The agent-time bookkeeping of the loop in validate.py, literally: (step, limit) of the stop, or None.

    `costs[n]` is how long the agent takes over call n. The official loop waits for a reply for
    min(limit, remaining) seconds; "budget" means the rest of the map's budget was the shorter wait.
    """
    budget = L.agent_budget_s(ticks)
    lat = []
    for steps in range(math.ceil(ticks / 4)):
        limit = L.FIRST_ACT_TIMEOUT_S if steps == 0 else L.ACT_TIMEOUT_S
        remaining = budget - sum(lat)
        if remaining <= 0:
            return steps, "used up"
        if costs[steps] > min(limit, remaining):
            return steps, "budget" if remaining < limit else "call"
        lat.append(costs[steps])
    return None


def test_the_runner_stops_the_agent_where_the_official_loop_does(timed):
    """Random call times on a map without an exit (300 decision steps, budget 12 s), against validate.py's bookkeeping."""
    entry = {"wad": fixture_wad(ROOM), "episode_ticks": 1200}
    text = {"budget": "act() ran past the rest of the agent's compute budget for this map (12 s = 40 ms x 300 decision steps)",
            "call": "act() took longer than 5 s", "first": "the first act() of the map took longer than 10 s"}
    seen = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        kind = seed % 5
        costs = rng.uniform(0.0, (0.07, 0.09, 0.02, 0.02, 0.02)[kind], 300)   # 0: inside the budget, 1: beyond it near the end
        if kind == 2:                                                          # one call of 5.5 to 7 s
            costs[rng.integers(1, 100)] = rng.uniform(5.5, 7.0)
        elif kind == 3:                                                        # a first call of more than 10 s
            costs[0] = rng.uniform(10.2, 11.0)
        elif kind == 4:                                                        # 8 to 9.5 s at first, then one of 4 s
            costs[0], costs[rng.integers(1, 100)] = rng.uniform(8.0, 9.5), 4.0
        expected = official_loop(costs, 1200)
        row, policy = timed(entry, costs=list(costs))
        if expected is None:
            assert (row["agent_error"], policy.calls, row["steps"]) == ("", 300, 300), seed
        else:
            step, limit = expected
            limit = "first" if (limit, step) == ("call", 0) else limit
            assert row["agent_error"] == f"step {step}: timeout: {text[limit]}", seed
            assert policy.calls == step + 1 and row["steps"] == 300, seed
        seen.append(None if expected is None else limit)
    assert {None, "budget", "call", "first"} <= set(seen), seen


def test_how_long_the_pool_waits_for_an_episode():
    """Without time limits 300 s; with them the reset limit, the map's compute budget and a 60 s margin."""
    assert R.patience_s({"episode_ticks": 2100}) == 300.0
    assert R.patience_s({"episode_ticks": 2100}, time_scale=1.0) == pytest.approx(10.0 + 21.0 + 60.0)
    assert R.patience_s({"episode_ticks": 4200}, time_scale=2.0) == pytest.approx((10.0 + 42.0) / 2.0 + 60.0)
    assert R.patience_s({"episode_ticks": 2100}, time_scale=1.0, fixed=4.0) == 4.0


class Sleeper:
    """Walks forward; in the episodes with seed `seed`, call number `at` of act() sleeps for `secs` seconds.

    `log` gets one line per policy built and per sleep started. A `stubborn` sleeper swallows whatever is raised
    inside its sleep, the way a call stuck in native code never sees the runner's interrupt.
    """

    def __init__(self, seed=1, at=3, secs=3600.0, log=None, stubborn=False):
        self.bad_seed, self.at, self.secs, self.log, self.stubborn = seed, at, secs, log, stubborn
        self.current, self.t = None, 0
        self.note("built")

    def note(self, text):
        if self.log:
            with open(self.log, "a") as f:
                f.write(text + "\n")

    def seed(self, seed):
        self.current = seed

    def reset(self):
        self.t = 0

    def act(self, obs):
        self.t += 1
        if self.t == self.at and self.current == self.bad_seed:
            self.note("sleep")
            until = time.perf_counter() + self.secs
            while time.perf_counter() < until:
                try:
                    time.sleep(until - time.perf_counter())
                except BaseException:
                    if not self.stubborn:
                        raise
        return FORWARD


def test_a_slow_call_is_judged_when_it_returns():
    """In the calling process nothing can end a call early: a 0.3 s call counted 20-fold is 6 s, over the 5 s limit."""
    rows = R.run_episodes(Sleeper(seed=0, secs=0.3), [LONG], workers=0, seeds=[0, 1], render_buffers=False,
                          time_limits=True, time_scale=20.0)
    assert rows[0]["agent_error"] == "step 2: timeout: act() took longer than 5 s (measured time x 20)"
    assert (rows[0]["exit"], rows[0]["steps"]) == (0, 525) and 300.0 <= rows[0]["act_ms_max"] < 3000.0
    assert (rows[1]["exit"], rows[1]["agent_error"]) == (1, "")          # the next map starts with a working agent


def test_a_call_that_never_returns_is_ended_at_its_limit_in_a_worker(tmp_path):
    """The policy sleeps for an hour in its third call. The worker ends the call at the limit (5 s, here counted
    10-fold: 0.5 s), the player stands still for the rest of the map and the episode is scored."""
    log = tmp_path / "log"
    started = time.perf_counter()
    rows = R.run_episodes((f"{__name__}:Sleeper", {"log": str(log)}), [LONG], workers=1, seeds=[0, 1, 2],
                          render_buffers=False, time_limits=True, time_scale=10.0)
    assert time.perf_counter() - started < 30.0
    assert [(r["exit"], r["steps"], r["error"]) for r in rows] == [(1, 12, ""), (0, 525, ""), (1, 12, "")]
    assert rows[1]["agent_error"] == "step 2: timeout: act() took longer than 5 s (measured time x 10)"
    assert 500.0 <= rows[1]["act_ms_max"] < 3000.0 and rows[1]["score"] == rows[1]["score"] and rows[1]["timeout"] == 1
    # the interrupted policy is not used again: the worker builds a fresh one for its next map
    assert log.read_text().splitlines() == ["built", "sleep", "built"]
    # the same episode comes out the same in the calling process, where the call is only measured (0.6 s, counted 6 s)
    inline = R.run_episodes(Sleeper(secs=0.6), [LONG], workers=0, seeds=[1], render_buffers=False, time_limits=True,
                            time_scale=10.0)
    assert strip_timing(inline) == [{**strip_timing(rows[1:2])[0], "policy": "'Sleeper'"}]


@pytest.mark.slow
def test_a_call_that_cannot_be_interrupted_is_left_to_the_pool(tmp_path):
    log = tmp_path / "log"
    started = time.perf_counter()
    rows = R.run_episodes((f"{__name__}:Sleeper", {"log": str(log), "stubborn": True}), [LONG], workers=1, seeds=[0, 1, 2],
                          render_buffers=False, time_limits=True, time_scale=10.0, episode_timeout_s=3.0)
    assert 3.0 < time.perf_counter() - started < 30.0
    assert [r["exit"] for r in rows] == [1, 0, 1] and rows[1]["score"] == 0.0 and rows[1]["agent_error"] == ""
    assert "timed out" in rows[1]["error"] and "within 3 s" in rows[1]["error"]
    assert log.read_text().splitlines() == ["built", "sleep", "built"]    # one attempt, then a new worker for the rest


# ---- telemetry hooks
class Telemetry:
    """Walks forward and reports a pose and an exit belief; from call `bad_at` on, hook `bad` misbehaves (`how`)."""

    def __init__(self, bad=None, bad_at=3, how="raise"):
        self.bad, self.bad_at, self.how, self.t = bad, bad_at, how, 0
        self.asked = {"pose_estimate": 0, "exit_believed": 0}

    def reset(self):
        self.t = 0

    def act(self, obs):
        self.t += 1
        return FORWARD

    def _report(self, name, good):
        self.asked[name] += 1
        if name != self.bad or self.t < self.bad_at:
            return good
        if self.how == "raise":
            raise RuntimeError("telemetry bug")
        return {"garbage": "garbage", "nan": (float("nan"), 0.0, 0.0), "short": (1.0, 2.0), "array": np.zeros(3)}[self.how]

    def pose_estimate(self):
        return self._report("pose_estimate", (0.0, 0.0, 0.0))

    def exit_believed(self):
        return self._report("exit_believed", self.t >= 2)


def test_a_failing_telemetry_hook_does_not_stop_the_agent(arena):
    """The official scorer never calls these hooks, so a bug in one must not change the episode."""
    def play(**kwargs):
        policy = Telemetry(**kwargs)
        return arena(policy, fixture_wad(EXIT)), policy

    good, policy = play()
    assert (good["exit"], good["steps"], good["agent_error"], good["hook_error"]) == (1, 12, "", "")
    assert policy.asked == {"pose_estimate": 12, "exit_believed": 12}
    assert good["exit_believed_tic"] == 4 and good["odom_err_final"] > 100.0     # it reports (0, 0, 0) all the way

    row, policy = play(bad="exit_believed")                                    # raises from its third call on
    assert row["hook_error"] == "step 2: exit_believed(): RuntimeError: telemetry bug" and row["agent_error"] == ""
    assert policy.asked == {"pose_estimate": 12, "exit_believed": 3}            # not asked again; the other hook goes on
    assert strip_timing([row]) == strip_timing([{**good, "hook_error": row["hook_error"]}])

    row, policy = play(bad="pose_estimate", bad_at=4, how="garbage")
    assert row["hook_error"] == "step 3: pose_estimate(): ValueError: could not convert string to float: 'g'"
    assert (row["exit"], row["steps"], row["agent_error"]) == (1, 12, "") and policy.asked == {"pose_estimate": 4, "exit_believed": 12}
    assert row["exit_believed_tic"] == 4 and 0.0 < row["odom_err_max"] < good["odom_err_max"]    # three poses were compared
    for how, message in (("nan", "three finite numbers"), ("short", "IndexError")):
        row, policy = play(bad="pose_estimate", bad_at=1, how=how)
        assert row["hook_error"].startswith("step 0: pose_estimate(): ") and message in row["hook_error"]
        assert (row["exit"], row["agent_error"]) == (1, "") and math.isnan(row["odom_err_final"])
    row, policy = play(bad="exit_believed", bad_at=1, how="array")
    assert "exit_believed(): ValueError" in row["hook_error"] and row["exit"] == 1 and math.isnan(row["exit_believed_tic"])
    # both hooks failing: the first message is kept and neither is called again
    class Both(Telemetry):
        def pose_estimate(self):
            self.asked["pose_estimate"] += 1
            raise KeyError("pose")

    policy = Both(bad="exit_believed", bad_at=1)
    row = arena(policy, fixture_wad(EXIT))
    assert row["hook_error"] == "step 0: pose_estimate(): KeyError: 'pose'" and row["exit"] == 1
    assert policy.asked == {"pose_estimate": 1, "exit_believed": 1}
    # a hook is not asked once the agent itself has stopped
    policy = CrashOnSeed(seed=0, mode="raise")
    policy.pose_estimate = lambda: (_ for _ in ()).throw(AssertionError("asked after the agent stopped"))
    policy.t_at_report = []
    policy.exit_believed = lambda: policy.t_at_report.append(policy.t) or False
    row = arena(policy, SHORT)
    assert row["agent_error"] == "step 2: ValueError: boom" and policy.t_at_report == [1, 2]
    assert row["hook_error"].startswith("step 0: pose_estimate(): AssertionError")


def test_a_crashed_worker_costs_one_episode_not_the_run(tmp_path):
    maps = [{"wad": fixture_wad(EXIT), "episode_ticks": 200}, {"wad": fixture_wad(ROOM), "episode_ticks": 80}]
    spec = (f"{__name__}:CrashOnSeed", {"seed": 1})
    started = time.perf_counter()
    rows = R.run_episodes(spec, maps, workers=2, seeds=[0, 1, 2], render_buffers=False)
    assert time.perf_counter() - started < 60.0
    assert [bool(r["error"]) for r in rows] == [False, True, False, False, True, False]
    assert "exit code 7" in rows[1]["error"] and rows[1]["score"] == 0.0 and rows[1]["exit"] == 0
    assert rows[1]["has_exit"] == 1 and rows[4]["has_exit"] == 0 and math.isnan(rows[1]["stuck_rate"])
    assert [r["exit"] for r in rows] == [1, 0, 1, 0, 0, 0] and [r["seed"] for r in rows] == [0, 1, 2, 0, 1, 2]
    assert S.suite_score(rows)["errors"] == 2
    # a crash that does not repeat is retried and leaves no trace
    marker = tmp_path / "crashed-once"
    rows = R.run_episodes((f"{__name__}:CrashOnSeed", {"seed": 1, "marker": str(marker)}), maps, workers=2,
                          seeds=[0, 1, 2], render_buffers=False)
    assert marker.exists() and all(r["error"] == "" for r in rows) and [r["exit"] for r in rows] == [1, 1, 1, 0, 0, 0]


def test_a_policy_that_cannot_be_built_fails_fast():
    with pytest.raises(RuntimeError, match="weights missing"):
        R.run_episodes(f"{__name__}:broken_factory", [fixture_wad(EXIT)], workers=2)
    with pytest.raises(ValueError, match="module:factory"):
        R.run_episodes("not-a-spec", [fixture_wad(EXIT)], workers=1)


@pytest.mark.slow
def test_a_hung_worker_is_stopped_at_the_timeout(tmp_path):
    maps = [{"wad": fixture_wad(EXIT), "episode_ticks": 200}]
    started = time.perf_counter()
    rows = R.run_episodes((f"{__name__}:CrashOnSeed", {"seed": 1, "mode": "hang", "marker": str(tmp_path / "hangs")}), maps,
                          workers=2, seeds=[0, 1, 2], render_buffers=False, episode_timeout_s=3.0)
    assert time.perf_counter() - started < 30.0
    assert "timed out" in rows[1]["error"] and "within 3 s" in rows[1]["error"] and rows[1]["score"] == 0.0
    assert rows[0]["exit"] == rows[2]["exit"] == 1
    # what hung will hang again: unlike a crash, a timeout is not retried (retries is 1 here)
    assert (tmp_path / "hangs").read_text() == "hang\n"


def test_recorder_in_worker_processes():
    rows = R.run_episodes("training.runner:RandomPolicy", [{"wad": fixture_wad(ROOM), "episode_ticks": 120}], workers=1,
                          seeds=[0, 1], record=f"{__name__}:CountingRecorder", render_buffers=False)
    assert [(r["rec_steps"], r["rec_ok"]) for r in rows] == [(30, 1), (30, 1)]


def test_agent_policy_loads_a_submission(tmp_path):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "agent.py").write_text(
        "import numpy as np\n"
        "class DoomAgent:\n"
        "    def __init__(self, observation_space, action_space, device):\n"
        "        self.n, self.device, self.resets = action_space.shape[0], device, 0\n"
        "    def reset(self):\n"
        "        self.resets += 1\n"
        "    def act(self, obs):\n"
        "        a = np.zeros(self.n, np.int64); a[0] = 1\n"
        "        return a\n"
        "    def pose_estimate(self):\n"
        "        return (0.0, 0.0, 0.0)\n")
    policy = R.AgentPolicy(device="cpu", repo=str(tmp_path))
    assert not hasattr(policy, "exit_believed") and policy.pose_estimate() == (0.0, 0.0, 0.0)
    row = R.run_episodes(policy, [fixture_wad(EXIT)], workers=0)[0]
    assert row["exit"] == 1 and policy.agent.resets == 1 and policy.agent.device == "cpu"
    assert row["odom_err_final"] > 100.0 and row["odom_heading_err_max"] == 0.0   # it never updates its pose
    assert math.isnan(row["exit_seen_timeout"])


def test_map_entries_and_manifests(tmp_path):
    entry = R.normalise_map("maps/two_rooms.wad")      # relative to the repository root, wherever we are run from
    assert Path(entry.pop("wad")).samefile(TWO_ROOMS) and entry == {"id": "two_rooms", "map": "MAP01", "tier": 0, "episode_ticks": 2100}
    assert R.normalise_map("some/dir/x.wad:map05")["map"] == "MAP05" and R.normalise_map("x.wad:MAP05")["id"] == "x:MAP05"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": TWO_ROOMS, "tier": 2, "seed": 9, "episode_ticks": 2800, "grid": "x"}) + "\n"
                        + json.dumps({"path": "tests/fixtures/" + Path(fixture_wad(KEY)).name, "tier": 3}) + "\n")
    maps = R.load_maps(manifest)
    assert [(m["id"], m["tier"], m["episode_ticks"]) for m in maps] == [("two_rooms", 2, 2800), (Path(fixture_wad(KEY)).stem, 3, 2100)]
    assert Path(maps[1]["wad"]).samefile(fixture_wad(KEY)) and Path(maps[1]["wad"]).is_absolute()
    assert len(R.load_maps(FIXTURES)) == 13 and R.load_maps(TWO_ROOMS, tier=1)[0]["tier"] == 1
    tasks = R.episode_tasks(maps, 2)
    assert [(t["id"], t["seed"]) for t in tasks] == [("two_rooms", 0), ("two_rooms", 1), (maps[1]["id"], 0), (maps[1]["id"], 1)]
    # a manifest that was moved together with its WADs is found by file name; a missing WAD is named
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "m.wad").write_bytes(Path(fixture_wad(EXIT)).read_bytes())
    (moved / "manifest.jsonl").write_text(json.dumps({"name": "first", "path": "/built/somewhere/else/m.wad", "tier": 1}) + "\n")
    entry = R.load_maps(moved / "manifest.jsonl")[0]
    assert Path(entry["wad"]).samefile(moved / "m.wad") and (entry["id"], entry["tier"]) == ("first", 1)
    (moved / "manifest.jsonl").write_text(json.dumps({"path": "gone.wad"}) + "\n")
    with pytest.raises(FileNotFoundError, match="line 1: WAD not found: gone.wad"):
        R.load_maps(moved / "manifest.jsonl")


def test_runner_command_line_and_benchmark(capsys):
    assert R.main(["--policy", "training.runner:CorridorPolicy", "--maps", fixture_wad(KEY), fixture_wad(DOOR),
                   "--workers", "0", "--seeds", "0,1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 5 and "exit 1" in out[0] and out[-1].startswith("score 99.") and out[-1].endswith("errors 0")
    stats = R.bench([{"wad": fixture_wad(ROOM), "episode_ticks": 200}], worker_counts=(1,), seeds=2)
    assert stats[0]["episodes"] == 2 and stats[0]["steps"] == 100 and stats[0]["errors"] == 0
    assert stats[0]["steps_per_s"] > 0 and stats[0]["in_episode_steps_per_s"] > stats[0]["steps_per_s"]
    assert len(R.default_bench_maps()) == 14


# ================================================================================== evaluate
def fixture_builder(grid, **options):
    """Stands in for the map generator: the test maps already exist as WADs named after their grid."""
    return Path(fixture_wad(grid)).read_bytes()


def write_suite(tmp_path, name="mini.jsonl"):
    rows = [
        {"grid": EXIT, "tier": 1, "episode_ticks": 400, "options": {}},
        {"grid": GUNNER, "tier": 1, "episode_ticks": 800, "options": {}},
        {"grid": DOOR, "tier": 2, "episode_ticks": 600, "options": {}},
        {"grid": KEY.splitlines(), "tier": 3, "episode_ticks": 800},            # rows may be a list of lines
        {"path": TWO_ROOMS, "tier": 2, "episode_ticks": 400, "id": "two-rooms"},  # ... or point at a WAD
    ]
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def run_eval(tmp_path, policy, tag, **kw):
    kw.setdefault("quiet", True)
    return E.evaluate(policy, write_suite(tmp_path), tag, workers=0, out=tmp_path / "results.csv",
                      episodes_dir=tmp_path / "eval", maps_dir=tmp_path / "maps", render_buffers=False,
                      builder=fixture_builder, **kw)


def test_evaluate_writes_one_row_per_tier(tmp_path):
    summary = run_eval(tmp_path, "training.runner:CorridorPolicy", "corridor")
    assert [r["tier"] for r in summary] == ["1", "2", "3", "all"]
    with open(tmp_path / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        header = f.readline().strip().split(",")
    assert header == list(E.RESULT_COLUMNS) and len(rows) == 4
    assert b"\r" not in (tmp_path / "results.csv").read_bytes() + (tmp_path / "eval" / "corridor.csv").read_bytes()
    assert header[:9] == ["stamp", "tag", "commit", "suite", "suite_sha", "policy", "seeds", "time_scale", "tier"]
    by_tier = {r["tier"]: r for r in rows}
    assert {r["time_scale"] for r in rows} == {"1"}        # the official time limits were applied, at this machine's speed
    assert by_tier["all"]["agent_errors"] == "0" and by_tier["all"]["hook_errors"] == "0"
    assert {r["tag"] for r in rows} == {"corridor"} and {r["suite"] for r in rows} == {"mini.jsonl"}
    assert by_tier["1"]["episodes"] == "2" and by_tier["1"]["exit_rate"] == "1"      # one of the two tier-1 maps has an exit
    assert by_tier["3"]["exit_rate"] == "1" and by_tier["3"]["keys_frac"] == "1" and by_tier["3"]["locked_frac"] == "1"
    assert by_tier["2"]["episodes"] == "2" and by_tier["2"]["exit_rate"] == "0.5"    # the corridor walker fails two_rooms
    assert by_tier["all"]["episodes"] == "5" and by_tier["all"]["errors"] == "0" and by_tier["all"]["maps"] == "5"
    assert rows[0]["policy"] == "training.runner:CorridorPolicy" and rows[0]["seeds"] == "0"
    assert len(rows[0]["suite_sha"]) == 12 and rows[0]["commit"] and rows[0]["stamp"].startswith("20")
    # the `all` row is the tier-weighted score of the per-episode rows
    episodes = list(csv.DictReader(open(tmp_path / "eval" / "corridor.csv", newline="")))
    assert [e["map_id"] for e in episodes] == ["mini-0000", "mini-0001", "mini-0002", "mini-0003", "two-rooms"]
    weights = {"1": 1.0, "2": 2.0, "3": 3.0}
    expected = sum(weights[e["tier"]] * float(e["score"]) for e in episodes) / sum(weights[e["tier"]] for e in episodes)
    assert float(by_tier["all"]["score"]) == pytest.approx(expected, abs=1e-3)
    assert sorted(p.name for p in (tmp_path / "maps" / "mini").iterdir()) == ["0000.wad", "0001.wad", "0002.wad", "0003.wad"]

    # a second evaluation appends below the same header; a subset is labelled as one
    run_eval(tmp_path, "training.runner:RandomPolicy", "random", tiers=[2, 3], limit=1, seeds=[0, 1])
    text = (tmp_path / "results.csv").read_text().splitlines()
    assert len(text) == 1 + 4 + 3 and text[0].split(",") == list(E.RESULT_COLUMNS)
    last = E.read_results(tmp_path / "results.csv")[-1]
    assert (last["tag"], last["tier"], last["episodes"], last["seeds"]) == ("random", "all", "4", "0 1")
    assert last["suite"] == "mini.jsonl[tiers=2,3;limit=1]"
    # the time limits a result was measured under are part of it
    run_eval(tmp_path, "training.runner:CorridorPolicy", "slower", tiers=[3], time_scale=2.5)
    run_eval(tmp_path, "training.runner:CorridorPolicy", "unlimited", tiers=[3], time_limits=False)
    scales = {r["tag"]: r["time_scale"] for r in E.read_results(tmp_path / "results.csv")}
    assert scales == {"corridor": "1", "random": "1", "slower": "2.5", "unlimited": ""}


def test_evaluate_shows_agent_failures(tmp_path, capsys):
    """An agent that raises on every map: counted in the table, named in a WARNING, shown while the run is going."""
    summary = run_eval(tmp_path, CrashOnSeed(seed=0, mode="raise"), "broken", quiet=False)
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[0].split()[-3:] == ["agent", "hook", "err"]
    assert [line.split()[-3:] for line in lines[1:5]] == [["2", "0", "0"], ["2", "0", "0"], ["1", "0", "0"], ["5", "0", "0"]]
    assert lines[5:] == ["WARNING: the agent failed in 5 of 5 episodes and stood still from then on (0 stopped by a time "
                         "limit, 5 by an error or an invalid action); first: mini-0000 seed 0: step 2: ValueError: boom"]
    assert err.count("the agent failed: ") == 3 and "the agent failed: mini-0000 seed 0: step 2: ValueError: boom" in err
    assert [(r["tier"], r["agent_errors"], r["errors"]) for r in summary] == [("1", 2, 0), ("2", 2, 0), ("3", 1, 0), ("all", 5, 0)]
    logged = E.read_results(tmp_path / "results.csv")
    assert [r["agent_errors"] for r in logged] == ["2", "2", "1", "5"]
    # a clean run prints the table and nothing else; a quiet one prints nothing at all
    run_eval(tmp_path, "training.runner:CorridorPolicy", "fine", quiet=False)
    out, err = capsys.readouterr()
    assert len(out.splitlines()) == 5 and "WARNING" not in out and "failed" not in err
    assert out.splitlines()[-1].split()[-3:] == ["0", "0", "0"]
    run_eval(tmp_path, CrashOnSeed(seed=0, mode="raise"), "silent", tiers=[3])
    assert capsys.readouterr() == ("", "")
    # the comparison names the failures of either run, and a row logged before the counts existed still prints
    E.compare("fine", "broken", tmp_path / "results.csv", tmp_path / "eval")
    out = capsys.readouterr().out
    assert "note: broken: 5 of 5 episodes had the agent fail or run out of time and stand still" in out
    E.print_rows([{"tag": "old", "tier": "all", "episodes": "3", "score": "12.5", "errors": "1"}])
    assert capsys.readouterr().out.splitlines()[1].split()[-3:] == ["-", "-", "1"]


def test_evaluate_applies_the_time_limits_and_reports_hooks(tmp_path, capsys, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(R, "time", clock)
    suite = tmp_path / "short.jsonl"
    suite.write_text(json.dumps({"path": fixture_wad(EXIT), "tier": 2, "episode_ticks": 60, "id": "short"}) + "\n"
                     + json.dumps({"path": fixture_wad(EXIT), "tier": 2, "episode_ticks": 2100, "id": "long"}) + "\n")
    kw = dict(workers=0, out=tmp_path / "r.csv", episodes_dir=tmp_path / "eval", render_buffers=False)
    # 70 ms a call: over the budget of the 60-tick map (stopped at its ninth call), 4% of the budget of the long one
    summary = E.evaluate(Costly(clock, [0.07]), suite, "slow", **kw)
    out = capsys.readouterr().out
    assert summary[-1]["agent_errors"] == 1 and summary[-1]["exit_rate"] == 0.5
    assert ("WARNING: the agent failed in 1 of 2 episodes and stood still from then on (1 stopped by a time limit, 0 by "
            "an error or an invalid action); first: short seed 0: step 8: timeout: act() ran past the rest of") in out
    # without the limits it walks out of both, and the report says what the evaluation would have done
    summary = E.evaluate(Costly(clock, [0.07]), suite, "slow-unlimited", time_limits=False, **kw)
    out = capsys.readouterr().out
    assert summary[-1]["agent_errors"] == 0 and summary[-1]["exit_rate"] == 1.0
    assert "WARNING: the time limits were not applied, and in 1 of 2 episodes act() took longer than they allow" in out
    # 20 ms a call leaves room here (12 calls: 40% of the short map's budget) and little at twice the cost (80%)
    E.evaluate(Costly(clock, [0.02]), suite, "ok", **kw)
    assert "WARNING" not in capsys.readouterr().out
    E.evaluate(Costly(clock, [0.02]), suite, "tight", time_scale=2.0, **kw)
    out = capsys.readouterr().out
    assert "WARNING: little headroom: act() used 80% of the map's compute budget on short seed 0 (mean 20.0 ms, counted x 2;" in out
    # a failing telemetry hook is reported and costs nothing
    summary = E.evaluate(Telemetry(bad="pose_estimate"), suite, "hook", **kw)
    out, err = capsys.readouterr()
    assert (summary[-1]["hook_errors"], summary[-1]["agent_errors"], summary[-1]["exit_rate"]) == (2, 0, 1.0)
    assert "WARNING: a telemetry hook of the policy failed in 2 of 2 episodes" in out
    assert "first: short seed 0: step 2: pose_estimate(): RuntimeError: telemetry bug" in out and err.count("a hook failed: ") == 2
    assert out.splitlines()[2].split()[-3:] == ["0", "2", "0"]
    # a map that is void because reset() ran out of time counts as score 0 and is reported as such
    summary = E.evaluate(Costly(clock, reset_s=11.0), suite, "void", **kw)
    out = capsys.readouterr().out
    assert (summary[-1]["errors"], summary[-1]["agent_errors"], summary[-1]["score"]) == (2, 2, 0.0)
    assert "WARNING: 2 of 2 episodes could not be played or are void, and count as score 0; first: short seed 0: the map is void" in out


def test_evaluate_command_line_strict(tmp_path, capsys):
    suite = tmp_path / "one.jsonl"
    suite.write_text(json.dumps({"path": fixture_wad(EXIT), "tier": 2, "episode_ticks": 200}) + "\n")
    args = ["--suite", str(suite), "--workers", "0", "--no-render", "--out", str(tmp_path / "r.csv"),
            "--episodes-dir", str(tmp_path / "eval"), "--maps-dir", str(tmp_path / "maps")]
    broken = ["--policy", f"{__name__}:CrashOnSeed", "--kw", "seed=0", "--kw", "mode=raise"]
    assert E.main([*broken, "--tag", "broken", *args]) == 0             # measured and logged ...
    assert "WARNING: the agent failed in 1 of 1 episodes" in capsys.readouterr().out
    assert E.main([*broken, "--tag", "broken", "--strict", *args]) == 2  # ... and an error for scripts that ask
    assert E.main(["--policy", "training.runner:CorridorPolicy", "--tag", "fine", "--strict", *args]) == 0
    assert E.main(["--policy", "training.runner:CorridorPolicy", "--tag", "free", "--no-time-limits", "--time-scale", "3", *args]) == 0
    logged = {r["tag"]: r["time_scale"] for r in E.read_results(tmp_path / "r.csv")}
    assert logged == {"broken": "1", "fine": "1", "free": ""}


def test_compare_is_promote_only_if_better(tmp_path, capsys):
    run_eval(tmp_path, ("training.runner:ScriptPolicy", {"script": "N"}), "idle")
    run_eval(tmp_path, "training.runner:CorridorPolicy", "corridor")
    args = ["--out", str(tmp_path / "results.csv"), "--episodes-dir", str(tmp_path / "eval")]
    better = E.compare("idle", "corridor", tmp_path / "results.csv", tmp_path / "eval", quiet=True)
    assert better["paired"] and better["n"] == 5 and better["delta"] > 30.0
    assert set(better["other"]) == {"score_exit_heavy", "score_balanced"} and all(v > 20.0 for v in better["other"].values())
    assert better["better"] == (better["delta"] > max(1.0, 2.0 * better["se"]))
    assert E.main(["--compare", "idle", "corridor", *args]) == (0 if better["better"] else 1)
    # the same result twice is a tie, and a worse one is never promoted
    same = E.compare("corridor", "corridor", tmp_path / "results.csv", tmp_path / "eval", quiet=True)
    assert same["delta"] == 0.0 and not same["better"] and same["margin"] == 1.0
    assert E.main(["--compare", "corridor", "corridor", *args]) == 1
    assert E.main(["--compare", "corridor", "idle", *args]) == 1
    out = capsys.readouterr().out
    assert "KEEP corridor" in out and "paired, n = 5" in out and "with the other weightings: exit_heavy" in out
    # without the per-episode files it falls back to the logged scores and their standard errors
    loose = E.compare("idle", "corridor", tmp_path / "results.csv", tmp_path / "nowhere", quiet=True)
    assert not loose["paired"] and loose["delta"] == pytest.approx(better["delta"], abs=1e-3) and loose["se"] > 0.0
    with pytest.raises(SystemExit, match="no result logged under tag 'missing'"):
        E.compare("idle", "missing", tmp_path / "results.csv", tmp_path / "eval", quiet=True)
    assert E.main(["--table", *args]) == 0
    assert capsys.readouterr().out.splitlines()[1].startswith("corridor")       # best first


def test_compare_threshold_on_synthetic_results(tmp_path):
    def log(tag, scores):
        rows = [{"map_id": f"m{i}", "tier": 2, "seed": 0, "score": s, "score_exit_heavy": s, "score_balanced": s,
                 "exit": 0, "has_exit": 1, "error": ""} for i, s in enumerate(scores)]
        E.write_episodes(tmp_path / "eval" / f"{tag}.csv", rows)
        E.append_results(tmp_path / "r.csv", E.tier_rows(rows, {"stamp": f"2026-11-26T20:00:0{len(tag)}", "tag": tag,
                                                              "suite": "s", "suite_sha": "x", "seeds": "0"}))

    base = [20.0, 40.0, 60.0, 80.0] * 10
    log("a", base)
    log("b0", [s + 0.5 for s in base])                           # uniformly +0.5: below one composite point
    log("b2", [s + 2.0 for s in base])                           # uniformly +2: beyond the point, no spread at all
    log("noisy", [s + (9.0 if i % 2 else -5.0) for i, s in enumerate(base)])   # +2 on average, but it is noise
    cmp = lambda b: E.compare("a", b, tmp_path / "r.csv", tmp_path / "eval", quiet=True)
    assert not cmp("b0")["better"] and cmp("b0")["delta"] == pytest.approx(0.5)
    assert cmp("b2")["better"] and cmp("b2")["se"] == pytest.approx(0.0, abs=1e-9)
    assert cmp("b2")["other"] == pytest.approx({"score_exit_heavy": 2.0, "score_balanced": 2.0})
    noisy = cmp("noisy")
    assert noisy["delta"] == pytest.approx(2.0) and noisy["se"] == pytest.approx(np.std([9.0, -5.0] * 20, ddof=1) / math.sqrt(40))
    assert not noisy["better"] and noisy["margin"] == pytest.approx(2.0 * noisy["se"])
    assert E.compare("b2", "a", tmp_path / "r.csv", tmp_path / "eval", quiet=True)["worse"]


def test_results_file_keeps_old_rows_when_columns_change(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("stamp,tag,old_metric,score\n2026-11-26T18:00:00,early,7,12.5\n")
    E.append_results(path, [{"stamp": "2026-11-26T19:00:00", "tag": "late", "tier": "all", "score": 50.123456,
                             "exit_rate": float("nan"), "episodes": 3}])
    rows = E.read_results(path)
    header = path.read_text().splitlines()[0].split(",")
    assert header == list(E.RESULT_COLUMNS) + ["old_metric"]
    assert (rows[0]["tag"], rows[0]["old_metric"], rows[0]["score"], rows[0]["tier"]) == ("early", "7", "12.5", "")
    assert (rows[1]["tag"], rows[1]["score"], rows[1]["exit_rate"], rows[1]["episodes"]) == ("late", "50.1235", "", "3")


def test_manifest_rows_play_the_wad_that_is_already_built(tmp_path):
    """A generator manifest has both the grid and the path of the WAD made from it: the WAD is used."""
    def no_builder(grid, **options):
        raise AssertionError("the WAD exists and must not be rebuilt")

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"name": "t3_0000007", "path": fixture_wad(KEY), "grid": KEY, "tier": 3,
                                    "episode_ticks": 2800, "options": {}}) + "\n")
    maps = E.suite_maps(manifest, tmp_path / "maps", builder=no_builder)
    assert [(m["id"], m["tier"], m["episode_ticks"]) for m in maps] == [("t3_0000007", 3, 2800)]
    assert Path(maps[0]["wad"]).samefile(fixture_wad(KEY)) and not (tmp_path / "maps").exists()
    # the same row without a usable path falls back to the grid
    manifest.write_text(json.dumps({"name": "t3_0000007", "path": "nowhere/x.wad", "grid": KEY, "tier": 3}) + "\n")
    maps = E.suite_maps(manifest, tmp_path / "maps", builder=fixture_builder)
    assert Path(maps[0]["wad"]).read_bytes() == Path(fixture_wad(KEY)).read_bytes()
    manifest.write_text(json.dumps({"path": "nowhere/x.wad", "tier": 3}) + "\n")
    with pytest.raises(SystemExit, match="has no grid and its WAD was not found"):
        E.suite_maps(manifest, tmp_path / "maps")


def test_suite_of_grids_needs_the_map_generator(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "training.mapgen.udmf", None)   # as if the generator were not there
    with pytest.raises(SystemExit, match="needs the map generator"):
        E.suite_maps(write_suite(tmp_path), tmp_path / "maps")
    with pytest.raises(SystemExit, match="suite file not found"):
        E.suite_maps(tmp_path / "absent.jsonl", tmp_path / "maps")


def test_suite_built_by_the_map_generator(tmp_path):
    """With the real generator: grids and options go through build_wad and the maps play."""
    pytest.importorskip("training.mapgen.udmf")
    suite = tmp_path / "gen.jsonl"
    suite.write_text(json.dumps({"grid": KEY, "tier": 3, "episode_ticks": 800, "options": {}}) + "\n"
                     + json.dumps({"grid": DOOR, "tier": 2, "episode_ticks": 600, "options": {"door_delay": 70}}) + "\n")
    maps = E.suite_maps(suite, tmp_path / "maps")
    built = M.load_meta(maps[0]["wad"])
    reference = M.load_meta(fixture_wad(KEY))
    assert built.grid.ascii(built.all_things) == reference.grid.ascii(reference.all_things)
    assert [(d.lock, d.centre) for d in built.doors] == [(d.lock, d.centre) for d in reference.doors]
    assert M.load_meta(maps[1]["wad"]).doors[0].delay_tics == 70          # the option reached the WAD
    rows = R.run_episodes("training.runner:CorridorPolicy", maps, workers=0)
    assert [(r["map_id"], r["tier"], r["exit"], r["door_reversals"]) for r in rows] == [("gen-0000", 3, 1, 0), ("gen-0001", 2, 1, 0)]
    stamp = os.stat(maps[0]["wad"]).st_mtime_ns
    assert E.suite_maps(suite, tmp_path / "maps")[0]["wad"] == maps[0]["wad"] and os.stat(maps[0]["wad"]).st_mtime_ns == stamp
