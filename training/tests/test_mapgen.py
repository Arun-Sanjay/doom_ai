"""Tests for the map generator: grid compiler, solver, layouts (base and wide profile), command line and frozen suites."""

import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from doom_env import make_env
from doom_env import spaces as S
from doom_env.wad import read_wad
from fixtures import fixture_wad
from training.mapgen import cli, layouts, randomize, udmf, walker
from training.mapgen.solver import solve

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "training" / "suites"

_G = "#########\n#P.....z#\n#..D....#\n#########"
REFERENCE_GRIDS = [
    "#####\n#P.z#\n#####",
    "##########\n#P..D...E#\n##########",
    "#######\n#P..B.E#\n#######",
    "#########\n#P..h..z#\n#########",
    "###########\n#z.P..D..z#\n###########",
    "############\n#P.r..R...E#\n############",
    "#####\n#P..#\n#####",
    _G,
    "\n".join(["#" * 13] * 3 + ["####" + row for row in _G.splitlines()]),
    "####\n#P.#\n####",
    "##########\n#P......i#\n##########",
    "##########\n#P......c#\n##########",
    "#######\n#P...E#\n#######",
]
TWO_ROOMS = "\n".join([
    "##############",
    "#......##....#",
    "#.P....##..z.#",
    "#......D.....#",
    "#..h...##....#",
    "#......####..#",
    "########.....#",
    "########..E..#",
    "##############",
])
FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)


def build(tmp_path, grid, name="m.wad", **opts):
    return str(udmf.write_wad(grid, tmp_path / name, **opts))


# ---- compiler: the reference gate ---------------------------------------------------------------------
REFERENCE_MAPS = [(g, None) for g in REFERENCE_GRIDS] + [(TWO_ROOMS, ROOT / "maps" / "two_rooms.wad")]


@pytest.mark.parametrize("index", range(len(REFERENCE_MAPS)))
def test_reference_maps_compile_byte_identical(index):
    """All 13 reference grids and maps/two_rooms.wad: same TEXTMAP and same file, byte for byte."""
    grid, path = REFERENCE_MAPS[index]
    reference = Path(path or fixture_wad(grid)).read_bytes()
    ours = udmf.build_wad(grid)
    assert udmf.read_textmap(ours) == udmf.read_textmap(reference)
    assert ours == reference


def _polygons(env):
    return sorted((sid, sorted(map(tuple, segs.tolist()))) for sid, segs in env.sector_polygons())


@pytest.mark.parametrize("index", range(len(REFERENCE_GRIDS)))
def test_reference_gate_same_observations(tmp_path, index):
    """The regenerated WAD behaves exactly like the reference map: rays, vitals and visible entities
    over 160 steps of the same seeded random actions, door sectors, sector count and sector polygons."""
    grid = REFERENCE_GRIDS[index]
    a = make_env(fixture_wad(grid), episode_ticks=2100)
    b = make_env(build(tmp_path, grid), episode_ticks=2100)
    rng = np.random.default_rng(index)
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert a.mapinfo.door_sectors == b.mapinfo.door_sectors
    assert a.mapinfo.num_sectors == b.mapinfo.num_sectors
    assert _polygons(a) == _polygons(b)
    steps = episodes = 0
    while steps < 160:
        for key in ("distance", "hit_type"):
            np.testing.assert_array_equal(oa["rays"][key], ob["rays"][key])
        np.testing.assert_array_equal(oa["vitals"], ob["vitals"])
        for key in ("features", "entity_type", "mask"):
            np.testing.assert_array_equal(oa["visible_entities"][key], ob["visible_entities"][key])
        action = rng.integers(0, 2, 9).astype(np.int8)
        oa, ra, ta, tra, _ = a.step(action)
        ob, rb, tb, trb, _ = b.step(action)
        assert (ra, ta, tra) == (rb, tb, trb)
        steps += 1
        if ta or tra:
            episodes += 1
            oa, _ = a.reset(seed=100 + episodes)
            ob, _ = b.reset(seed=100 + episodes)
    a.close()
    b.close()


# ---- compiler: legend and options -------------------------------------------------------------------
def test_legend_keeps_the_reference_symbols():
    expect = {"#": ("wall", None), ".": ("floor", None), "P": ("start", 1), "E": ("exit", None),
              "D": ("door", None), "R": ("door", None), "B": ("door", None), "r": ("thing", 13),
              "z": ("thing", 3004), "i": ("thing", 3001), "c": ("thing", 65), "h": ("thing", 2011)}
    for ch, (kind, thing) in expect.items():
        assert (udmf.LEGEND[ch].kind, udmf.LEGEND[ch].thing) == (kind, thing)
    assert (udmf.LEGEND["R"].lock, udmf.LEGEND["B"].lock, udmf.LEGEND["Y"].lock, udmf.LEGEND["D"].lock) == (129, 130, 131, 0)
    things = {s.name: s.thing for s in udmf.LEGEND.values() if s.thing}
    assert things == {
        "player start": 1, "red keycard": 13, "blue keycard": 5, "yellow keycard": 6, "red skull key": 38,
        "blue skull key": 40, "yellow skull key": 39, "zombieman": 3004, "shotgun guy": 9, "chaingunner": 65,
        "imp": 3001, "demon": 3002, "spectre": 58, "lost soul": 3006, "cacodemon": 3005, "hell knight": 69,
        "stimpack": 2011, "medikit": 2012, "green armor": 2018, "ammo clip": 2007, "box of bullets": 2048,
        "shotgun shells": 2008, "shotgun": 2001, "chaingun": 2002,
    }
    table = udmf.legend_table()
    assert json.loads(json.dumps(table)) == table and len(table) == len(udmf.LEGEND)


ENTITY_CLASS = {
    "z": "enemy_hitscan", "s": "enemy_hitscan", "c": "enemy_hitscan", "i": "enemy_projectile", "o": "enemy_projectile",
    "k": "enemy_projectile", "d": "enemy_melee", "g": "enemy_melee", "l": "enemy_melee",
    "h": "health", "m": "health", "a": "armor", "n": "ammo", "x": "ammo", "t": "ammo", "f": "weapon", "j": "weapon",
    "r": "key", "b": "key", "y": "key", "1": "key", "2": "key", "3": "key",
}
KEY_FLAG = {"r": "key_red_card", "b": "key_blue_card", "y": "key_yellow_card",
            "1": "key_red_skull", "2": "key_blue_skull", "3": "key_yellow_skull"}


def test_every_thing_loads_and_is_classified(tmp_path):
    """Each legend thing appears in the engine with a sensible entity class and ray type; walking over
    a key sets exactly its own vitals flag."""
    assert set(ENTITY_CLASS) == {ch for ch, s in udmf.LEGEND.items() if s.kind == "thing"}
    for ch, want in ENTITY_CLASS.items():
        env = make_env(build(tmp_path, f"#########\n#P..{ch}...#\n#########", f"thing_{ord(ch)}.wad"), episode_ticks=400)
        obs, _ = env.reset(seed=0)
        ents = obs["visible_entities"]
        assert ents["mask"].sum() == 1, ch
        assert S.ENTITY_TYPES[ents["entity_type"][0]] == want, ch
        assert S.RAY_HIT_TYPES[obs["rays"]["hit_type"][0]] == ("enemy" if want.startswith("enemy") else "item"), ch
        if ch in KEY_FLAG:
            for _ in range(20):
                obs, *_ = env.step(FWD)
            held = {S.VITALS_FIELDS[i] for i in range(8, 14) if obs["vitals"][i] == 1}
            assert held == {KEY_FLAG[ch]}, ch
        env.close()


@pytest.mark.parametrize("key,door,opens", [("r", "R", True), ("1", "R", True), ("b", "B", True), ("2", "B", True),
                                           ("y", "Y", True), ("3", "Y", True), ("b", "R", False), (".", "Y", False)])
def test_locked_doors_need_their_colour(tmp_path, key, door, opens):
    grid = f"############\n#P.{key}..{door}...E#\n############"
    env = make_env(build(tmp_path, grid), episode_ticks=1200)
    assert env.mapinfo.door_sectors == {1: udmf.LEGEND[door].lock}
    result = walker.walk(env, grid) if opens else None
    if opens:
        assert result["exit_reached"]
    else:                                   # the solver refuses, and the engine keeps the door shut
        assert not solve(grid).exit_reachable
        obs, _ = env.reset(seed=0)
        use = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
        for i in range(80):
            obs, *_ = env.step(use if i % 8 == 0 else FWD)
        assert env.privileged_state()["x"] < 64 * 6 and S.RAY_HIT_TYPES[obs["rays"]["hit_type"][0]] == "locked_door"
    env.close()


def test_start_pose_and_thing_options(tmp_path):
    grid = "#######\n#.....#\n#.P.z.#\n#.....#\n#######"
    opts = dict(start_angle=137, start_offset=(-12, 9.5), thing_angles={"4,2": 45}, ambush=[(4, 2)])
    text = udmf.build_textmap(grid, **opts)
    assert "thing { x = 148.0; y = 169.5; angle = 137; type = 1;" in text
    assert "angle = 45; type = 3004;" in text and text.count("ambush = true;") == 1
    assert "ambush" not in udmf.build_textmap(grid) and "angle = 180; type = 3004;" in udmf.build_textmap(grid)
    env = make_env(build(tmp_path, grid, **opts), episode_ticks=100)
    env.reset(seed=0)
    st = env.privileged_state()
    assert (st["x"], st["y"]) == pytest.approx((148.0, 169.5)) and st["angle"] == pytest.approx(137, abs=0.2)
    env.close()


def test_look_and_door_options(tmp_path):
    grid = "###########\n#P..D..E..#\n###########"
    base = udmf.build_textmap(grid)
    assert base.count("arg2 = 150;") == 2 and 'texturefloor = "GATE3"' in base and "lightlevel = 192" in base
    text = udmf.build_textmap(grid, light=[160, 255], themes=[["FLAT1", "BRICK1"], ["FLOOR5_1", "METAL1"]],
                              ceiling="FLAT20", exit_wall="WOOD1", door_delay=70)
    assert text.count("arg2 = 70;") == 2 and "arg2 = 150;" not in text
    assert text.count('textureceiling = "FLAT20"') == text.count("sector {") == 5
    assert 'texturefloor = "GATE3"' in text                     # the exit pad keeps its floor
    assert "BRICK1" in text and "WOOD1" in text and "STARTAN2" not in text
    assert text.count("lightlevel = 160") == 3 and text.count("lightlevel = 255") == 2
    assert 'texturefloor = "REDPAD"' in udmf.build_textmap(grid, exit_floor="REDPAD")
    # the randomised build still plays: the reference walker opens the door and reaches the pad
    env = make_env(build(tmp_path, grid, light=[160, 255], themes=[["FLAT1", "BRICK1"]], door_delay=70), episode_ticks=800)
    assert walker.walk(env, grid)["exit_reached"]
    env.close()


@pytest.mark.parametrize("grid", [
    "#####\n#E.P#\n#####",                                         # the pad is sector 0, west of the start
    "###\n#E#\n#.#\n#P#\n###",                                     # pad to the north
    "###\n#P#\n#.#\n#E#\n###",                                     # pad to the south
    "#####\n#...#\n#.E.#\n#..P#\n#####",                           # pad in open floor
    "###\n#E#\n#.#\n#D#\n#.#\n#P#\n###",                           # door on a north-south passage
    "########\n#E.D..P#\n#####.##\n#####R##\n#r......#\n########",   # door to the west, lock to the south
    TWO_ROOMS,
])
def test_doors_and_pads_work_in_every_direction(tmp_path, grid):
    env = make_env(build(tmp_path, walker.strip_monsters(grid)), episode_ticks=1400)
    result = walker.walk(env, walker.strip_monsters(grid))
    env.close()
    assert result["exit_reached"] and result["steps"] <= 2 * result["optimal_steps"] + 10


def test_walker_needs_a_reachable_exit(tmp_path):
    grid = "#####\n#P.z#\n#####"
    env = make_env(build(tmp_path, grid), episode_ticks=100)
    with pytest.raises(ValueError):
        walker.walk(env, grid)
    env.close()


def test_bad_grids_are_rejected():
    for grid in ("###\n#.#\n###", "#PP#", "#P?#", "#P.DD.#", "#P.DE#", ""):
        with pytest.raises(ValueError):
            udmf.build_wad(grid)
    assert udmf.build_wad("#P.EE#")                             # two pads side by side are fine
    assert udmf.build_wad("#P.E#\n") == udmf.build_wad("#P.E#")   # a trailing newline is ignored
    assert udmf.build_wad("#####\n#P.E") == udmf.build_wad("#####\n#P.E#")  # short rows end in wall


def test_randomisation_textures_exist_in_the_iwad():
    import vizdoom
    lumps = read_wad(os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"))
    names = [lump.name for lump in lumps]
    flats = set(names[names.index("F_START"):names.index("F_END")])
    walls = set()
    for lump in lumps:
        if lump.name in ("TEXTURE1", "TEXTURE2"):
            count = struct.unpack_from("<i", lump.data, 0)[0]
            for offset in struct.unpack_from(f"<{count}i", lump.data, 4):
                walls.add(lump.data[offset:offset + 8].rstrip(b"\0").decode())
    assert set(randomize.FLOORS) | set(randomize.CEILINGS) | {udmf.EXIT_FLOOR, udmf.DOOR_FLOOR, udmf.DEFAULT_CEILING} <= flats
    assert set(randomize.WALLS) | set(udmf.DOOR_FACE.values()) | {udmf.DOOR_TRACK} <= walls
    for floor, wall in udmf.DEFAULT_THEMES:
        assert floor in flats and wall in walls
    assert udmf.EXIT_FLOOR not in randomize.FLOORS and not any(f.startswith("GATE") for f in randomize.FLOORS)


def test_random_options_are_valid_and_reproducible(tmp_path):
    m = layouts.generate(4, 3)
    a = randomize.random_options(random.Random(5), m["grid"], m["options"])
    assert a == randomize.random_options(random.Random(5), m["grid"], m["options"])
    assert set(m["options"].get("ambush", [])) <= set(a.get("ambush", []))
    seen = set()
    for seed in range(40):
        opts = randomize.random_options(random.Random(seed), m["grid"], m["options"])
        seen |= set(opts)
        assert json.loads(json.dumps(opts)) == opts
        assert all(abs(v) <= 16 for v in opts.get("start_offset", (0, 0)))
        assert "exit_floor" not in opts
        udmf.build_textmap(m["grid"], **opts)
    assert seen == {"start_angle", "start_offset", "thing_angles", "ambush", "light", "themes", "ceiling", "exit_wall", "door_delay"}
    seen = set()
    for seed in range(40):                                        # the looks alone: the start pose is left to random_start
        seen |= set(randomize.random_options(random.Random(seed), m["grid"], m["options"], start=False))
    assert seen == {"thing_angles", "ambush", "light", "themes", "ceiling", "exit_wall", "door_delay"}


def test_first_training_set_keeps_its_options():
    """Without a start control `make_map` draws what it drew for data/maps/train (digest taken
    before the start control was added)."""
    options = [cli.make_map(tier, seed, 0.5)["options"] for tier in range(5) for seed in range(12)]
    assert hashlib.sha1(json.dumps(options, sort_keys=True).encode()).hexdigest() == "0d481bb1d951193a2e0b052bacdbc09d2e312735"


def test_start_pose_has_its_own_probabilities():
    on_grid = randomize.start_on_grid
    assert on_grid(None) and on_grid({}) and on_grid({"start_angle": 270}) and on_grid({"start_offset": [0, 0]})
    assert not on_grid({"start_angle": 45}) and not on_grid({"start_offset": [0, 3]}) and not on_grid({"start_angle": 90, "start_offset": (-1, 0)})
    rng, n = random.Random(7), 4000
    draws = [randomize.random_start(rng) for _ in range(n)]
    quarter = sum(d.get("start_angle") in (90, 180, 270) for d in draws)
    turned = sum(d.get("start_angle", 0) % 90 != 0 for d in draws)
    shifted = sum("start_offset" in d for d in draws)
    assert randomize.START_DEFAULTS == (0.5, 0.05, 0.05)
    assert abs(quarter / n - 0.5) < 0.03 and abs(turned / n - 0.05) < 0.015 and abs(shifted / n - 0.05) < 0.015
    assert 0.06 < sum(not on_grid(d) for d in draws) / n < 0.14                 # the robustness slice stays small
    for d in draws:
        assert set(d) <= {"start_angle", "start_offset"} and json.loads(json.dumps(d)) == d
        assert on_grid(d) == (d.get("start_angle", 0) % 90 == 0 and "start_offset" not in d)
        assert "start_offset" not in d or (any(d["start_offset"]) and all(abs(v) <= 16 for v in d["start_offset"]))
    assert all(randomize.random_start(rng, 0, 0, 0) == {} for _ in range(50))
    assert all(randomize.random_start(rng, 1, 0, 0)["start_angle"] in (90, 180, 270) for _ in range(50))
    assert not any(on_grid(randomize.random_start(rng, 0, 1, 0)) or on_grid(randomize.random_start(rng, 0, 0, 1)) for _ in range(200))
    for bad in ((0.7, 0.4, 0.0), (-0.1, 0.0, 0.0), (0.0, 0.0, 1.5)):
        with pytest.raises(ValueError):
            randomize.random_start(rng, *bad)


def test_start_control_leaves_the_map_and_sets_the_start_on_every_map():
    start = randomize.START_DEFAULTS
    off = 0
    for tier in (0, 3):
        for seed in range(150):
            plain, m = cli.make_map(tier, seed), cli.make_map(tier, seed, start=start)
            assert m == cli.make_map(tier, seed, start=start)
            assert (m["grid"], m["episode_ticks"], m["counts"], m["randomized"]) == (plain["grid"], plain["episode_ticks"], plain["counts"], False)
            assert set(m["options"]) <= {"ambush", "start_angle", "start_offset"}          # reference looks, own start
            assert {k: v for k, v in m["options"].items() if not k.startswith("start_")} == plain["options"]
            looks = cli.make_map(tier, seed, 1.0, start=start)                            # same start under random looks
            assert {k: looks["options"].get(k) for k in ("start_angle", "start_offset")} == \
                {k: m["options"].get(k) for k in ("start_angle", "start_offset")}
            assert looks["grid"] == m["grid"] and looks["randomized"]
            assert cli.make_map(tier, seed, 1.0, strip=True, start=start) == layouts.without_monsters(looks)
            off += not randomize.start_on_grid(m["options"])
    assert 12 <= off <= 50                                                                # about 10% of 300
    legacy = [cli.make_map(2, seed, 1.0) for seed in range(100)]
    assert sum(not randomize.start_on_grid(m["options"]) for m in legacy) > 60            # the first set: mostly off the grid


# ---- solver -----------------------------------------------------------------------------------------
def test_solver_on_hand_made_grids():
    plain = solve("#######\n#P...E#\n#######")
    assert plain.solvable and plain.path_cells == 4 and plain.doors_on_path == 0 and not plain.keys_needed
    assert plain.optimal_steps == 8
    back = solve("#########\n#r.P.R.E#\n#########")                    # walk away for the key, then back
    assert back.solvable and back.keys_needed and back.path_cells == 8 and back.doors_on_path == 1
    assert back.corners == 1 and back.optimal_steps == 2 * 8 + 4 + 20
    assert not solve("#######\n#P.R.E#\n#######").solvable               # no key
    assert not solve("#########\n#P.R.E.r#\n#########").solvable           # key behind its own door
    assert not solve("########\n#P.B.E##\n#.r#####\n########").solvable    # wrong colour
    skull = solve("########\n#P.B.E##\n#.2#####\n########")
    assert skull.solvable and skull.keys_needed
    nested = solve("###########\n#P.R.b.B.E#\n#r#########\n###########")   # red key, red door, blue key, blue door
    assert nested.solvable and nested.keys_needed and nested.doors_on_path == 2 and nested.path_cells == 10
    sealed = solve("#########\n#P..E#..#\n#########")                     # a pocket nobody can reach
    assert sealed.exit_reachable and not sealed.all_reachable and not sealed.solvable
    assert not solve("#######\n#P.E..#\n#######").solvable               # the pad cuts the corridor
    arena = solve("#####\n#P.z#\n#...#\n#####")                          # no exit: only reachability counts
    assert arena.solvable and not arena.has_exit and arena.optimal_steps == 0


# ---- layouts ----------------------------------------------------------------------------------------
def test_generate_is_deterministic_and_varied():
    for tier in layouts.TIERS:
        assert layouts.generate(tier, 5) == layouts.generate(tier, 5)
        grids = {layouts.generate(tier, seed)["grid"] for seed in range(12)}
        assert len(grids) >= 11
    with pytest.raises(ValueError):
        layouts.generate(9, 0)


def test_generation_does_not_depend_on_the_hash_seed():
    code = ("import hashlib, sys; sys.path.insert(0, '.'); from training.mapgen import cli; "
            "print(hashlib.sha1(repr([cli.make_map(t, s, 0.5) for t in range(5) for s in range(6)]).encode()).hexdigest())")
    digests = set()
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                             env={**os.environ, "PYTHONHASHSEED": hash_seed})
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def check_map(m: dict, bare: bool = False, profile: str = "base") -> None:
    """Everything a generated map promises, checked without the engine. `bare` is set for a
    monster-free copy: a map written with --strip-monsters or a row of a navigation suite.
    `profile` is the layout profile the map was generated with (it decides the tick cap rule)."""
    tier, counts, grid = m["tier"], m["counts"], m["grid"]
    sol = solve(grid)
    assert sol.solvable and sol.all_reachable
    rows = grid.split("\n")
    assert 3 <= len(rows) <= 28 and 3 <= len(rows[0]) <= 28 and len(set(map(len, rows))) == 1
    assert grid.count("P") == 1 and counts["exits"] == grid.count("E") <= 1
    assert m["episode_ticks"] in range(2100, 4201, 350)
    if counts["exits"]:
        assert sol.exit_reachable and counts["optimal_steps"] == sol.optimal_steps
        if profile == "wide" and tier >= 2:                    # a long route and little time beside it
            least, most = layouts.WIDE["budget"]
            assert (least, most) == (1.5, 3.0)
            assert least * 4 * sol.optimal_steps <= m["episode_ticks"] <= most * 4 * sol.optimal_steps
        else:
            assert m["episode_ticks"] >= layouts.MIN_BUDGET * 4 * sol.optimal_steps
    assert counts["keys"] == counts["locks"]
    assert counts["monsters"] == sum(grid.count(ch) for ch in udmf.MONSTER_CHARS)
    if tier == 0:                                               # navigation: never a monster, at most one lock
        assert counts["monsters"] == 0 and counts["locks"] <= 1 and counts["exits"] == 1
        assert sol.keys_needed == (counts["locks"] == 1)
    if tier == 1:
        assert counts["locks"] == 0 and (bare or counts["monsters"] >= 2)
    if tier == 2:
        assert counts["exits"] == 1 and counts["locks"] == 0
    if tier in (3, 4):
        assert counts["exits"] == 1 and sol.keys_needed
        assert counts["locks"] >= (1 if tier == 3 else 2) and (bare or counts["monsters"] >= (1 if tier == 3 else 4))
        doors = [ch for ch in grid if ch in udmf.LOCKED_DOOR_CHARS]
        assert len(set(doors)) == len(doors)                   # every lock has its own colour
    if bare:
        assert counts["monsters"] == 0 and not set(m["options"]) & set(layouts.MONSTER_OPTIONS)
    assert counts["ambush"] == len(m["options"].get("ambush", [])) <= counts["monsters"]
    for key in m["options"].get("ambush", []):
        c, r = (int(v) for v in key.split(","))
        assert rows[r][c] in udmf.MONSTER_CHARS
    udmf.build_wad(grid, **m["options"])


def test_each_tier_generates_solvable_maps():
    """50 maps per tier pass every structural check (the engine checks are below)."""
    for tier in layouts.TIERS:
        maps = [layouts.generate(tier, seed) for seed in range(50)]
        for m in maps:
            check_map(m)
        exits = sum(m["counts"]["exits"] for m in maps)
        assert exits == 50 if tier != 1 else 15 <= exits <= 35     # about half of the arenas have an exit
        assert len({m["style"] for m in maps}) >= 3
        if tier >= 2:
            assert any(m["counts"]["doors"] == 0 for m in maps) and any(m["counts"]["doors"] >= 6 for m in maps)
            assert any(m["counts"]["dead_ends"] >= 3 for m in maps)
    tier4 = [layouts.generate(4, seed) for seed in range(50)]
    assert sum(m["counts"]["ambush"] > 0 for m in tier4) >= 25
    tier3 = [layouts.generate(3, seed) for seed in range(50)]
    locks = sum(m["counts"]["locks"] for m in tier3)
    assert sum(m["counts"]["doors_met_before_key"] for m in tier3) >= 0.6 * locks   # the door is usually met first


def test_a_third_of_tier_0_has_a_key_and_a_locked_door():
    """Keys and locked doors without combat: one tier 0 map in three has exactly one key and one locked
    door of its colour that the route to the exit must pass, and still no monster."""
    maps = [layouts.generate(0, seed) for seed in range(300)]
    locked = [m for m in maps if m["counts"]["locks"]]
    assert len(locked) == 100 and [m["seed"] for m in locked] == list(range(2, 300, 3))
    assert [bool(m["counts"]["locks"]) for m in maps] == [layouts.has_key_stage(0, seed) for seed in range(300)]
    assert not any(layouts.has_key_stage(tier, seed) for tier in (1, 2, 3, 4) for seed in range(50))
    for m in locked:
        grid = m["grid"]
        doors = [ch for ch in grid if ch in udmf.LOCKED_DOOR_CHARS]
        keys = [ch for ch in grid if ch in udmf.KEY_CHARS]
        assert len(doors) == len(keys) == 1 and grid.count("E") == 1 and not set(grid) & udmf.MONSTER_CHARS
        assert {"R": "r1", "B": "b2", "Y": "y3"}[doors[0]].count(keys[0]) == 1        # the key fits the door
        sol = solve(grid)
        assert sol.keys_needed and sol.exit_reachable                    # the door is on every route to the exit
        assert not solve(grid.replace(keys[0], ".")).exit_reachable      # and without the key it stays shut
        check_map(m)
    assert not any(m["counts"]["monsters"] for m in maps)
    assert len({m["style"] for m in locked}) >= 4 and len({m["exit_style"] for m in locked}) == 5
    assert {ch for m in locked for ch in m["grid"] if ch in udmf.LOCKED_DOOR_CHARS} == set("RBY")
    assert sum(m["counts"]["doors_met_before_key"] for m in locked) >= 0.4 * len(locked)
    # A tier 0 map without a lock is the map the tier gives when no seed is a key map.
    for m in [m for m in maps if not m["counts"]["locks"]][:25]:
        plain = layouts.draft(0, random.Random(f"mapgen-v1/0/{m['seed']}/{m['attempts'] - 1}"))
        assert plain["grid"] == m["grid"] and plain["episode_ticks"] == m["episode_ticks"]


def test_tiers_1_to_4_still_generate_the_frozen_maps():
    """The tier 0 key maps left the other tiers alone: `generate` still gives the maps that were
    frozen into the development suite, and tier 3 and tier 4 keep their monsters."""
    rows = cli.read_rows(SUITES / "dev_v1.jsonl")
    for row in rows[::4]:
        m = layouts.generate(row["tier"], row["seed"])
        assert (m["grid"], m["episode_ticks"], m["options"]) == (row["grid"], row["episode_ticks"], row["options"])
    for tier in (1, 3, 4):
        assert all(layouts.generate(tier, seed)["counts"]["monsters"] >= layouts.MIN_MONSTERS[tier] >= 1 for seed in range(30))


# ---- the wide profile ---------------------------------------------------------------------------------
def test_base_profile_is_what_it_was_before_the_wide_option():
    """`generate` without a profile gives the maps it gave before the option existed (digest taken
    from the code as it was), and naming the base profile changes nothing."""
    maps = [layouts.generate(tier, seed) for tier in range(5) for seed in range(16)]
    assert hashlib.sha1(json.dumps(maps, sort_keys=True).encode()).hexdigest() == "766dd2a082e428438edd656faeb43f18f51b25cd"
    assert not any("profile" in m for m in maps)
    for tier, seed in ((0, 2), (1, 3), (2, 5), (3, 7), (4, 11)):
        assert layouts.generate(tier, seed, "base") == layouts.generate(tier, seed)
        assert cli.make_map(tier, seed, 0.5, profile="base") == cli.make_map(tier, seed, 0.5)
    assert layouts.PROFILES == ("base", "wide")
    for tier, profile in ((0, "wide"), (5, "wide"), (2, "huge")):
        with pytest.raises(ValueError):
            layouts.generate(tier, 0, profile)


def _monster_health(grid: str, kinds: str) -> int:
    return sum(layouts.MONSTER_HP[ch] for ch in grid if ch in kinds)


def _key_walks(m: dict) -> tuple[list[int], int]:
    """Along the solver's route: cells from every key to its door, and from the last key to the exit."""
    route = solve(m["grid"]).path
    rows = m["grid"].split("\n")
    seen = [rows[r][c] for c, r in route]
    to_door, last = [], 0
    for door, keys in (("R", "r1"), ("B", "b2"), ("Y", "y3")):
        if door in seen:
            at = max(i for i, ch in enumerate(seen[:seen.index(door)]) if ch in keys)
            to_door.append(seen.index(door) - at)
            last = max(last, at)
    return to_door, len(seen) - 1 - last


def test_wide_profile_keeps_the_rules_inside_wider_limits():
    """Tiers 1 to 4 of the wide profile: solvable, every lock needed, sizes and monster numbers inside
    the wider limits, monster health per open cell inside the relaxed cap, and in tiers 2 to 4 a tick
    cap of 1.5 to 3 times the cost of the route. The population really uses the room it was given."""
    wide = layouts.WIDE
    assert {t: wide["monsters"][t][0] for t in (2, 3, 4)} == {2: 12, 3: 12, 4: 20}
    assert {t: wide["hp_per_cell"][t] / layouts.MONSTER_HP_PER_CELL[t] for t in (2, 3, 4)} == \
        {t: wide["monsters"][t][0] / layouts.MONSTER_COUNT[t][0] for t in (2, 3, 4)}       # relaxed in proportion
    assert wide["hp_per_cell"][1] == layouts.MONSTER_HP_PER_CELL[1] and wide["size"][1][1] == wide["max_side"][1] == 26
    maps = {tier: [layouts.generate(tier, seed, "wide") for seed in range(60)] for tier in (1, 2, 3, 4)}
    for tier, part in maps.items():
        assert part[5] == layouts.generate(tier, 5, "wide") and part[5]["grid"] != layouts.generate(tier, 5)["grid"]
        assert len({m["grid"] for m in part}) == 60
        for m in part:
            check_map(m, profile="wide")
            counts = m["counts"]
            assert m["profile"] == "wide" and max(counts["width"], counts["height"]) <= wide["max_side"][tier]
            assert counts["monsters"] <= (12 if tier == 1 else wide["monsters"][tier][0])
            # every monster but the zombieman is paid for out of the health the map may hold
            assert _monster_health(m["grid"], "scidglok") <= wide["hp_per_cell"][tier] * counts["open_cells"]
            assert set(m["options"]) <= {"ambush"}
    assert max(max(m["counts"]["width"], m["counts"]["height"]) for m in maps[1]) > 22           # larger arenas
    assert max(m["counts"]["monsters"] for m in maps[2]) > 6 and max(m["counts"]["monsters"] for m in maps[3]) > 8
    assert max(m["counts"]["monsters"] for m in maps[4]) > 16
    assert 15 <= sum(m["counts"]["exits"] for m in maps[1]) <= 45                                 # arenas: about half
    assert sum(m["counts"]["monsters"] == 0 for m in maps[2]) >= 10                              # still a monster-free share
    for tier in (2, 3, 4):
        base = [layouts.generate(tier, seed) for seed in range(60)]
        budgets = [m["episode_ticks"] / (4 * m["counts"]["optimal_steps"]) for m in maps[tier]]
        assert min(budgets) < 1.8 and max(budgets) > 2.7 and all(m["counts"]["optimal_steps"] >= 175 for m in maps[tier])
        mean = lambda part, key: sum(m["counts"][key] for m in part) / len(part)             # noqa: E731
        assert mean(maps[tier], "route_cells") > 1.3 * mean(base, "route_cells")
        assert mean(maps[tier], "monsters") > 1.3 * mean(base, "monsters")
        if tier >= 3:                                              # long backtracking: key far from its door, exit far from the last key
            far, near = [_key_walks(m) for m in maps[tier]], [_key_walks(m) for m in base]
            to_door = lambda walks: sum(sum(w[0]) for w in walks) / sum(len(w[0]) for w in walks)   # noqa: E731
            to_exit = lambda walks: sum(w[1] for w in walks) / len(walks)                             # noqa: E731
            assert to_door(far) > 1.3 * to_door(near) and to_exit(far) > 1.3 * to_exit(near)
            locks = sum(m["counts"]["locks"] for m in maps[tier])
            assert sum(m["counts"]["doors_met_before_key"] for m in maps[tier]) >= 0.6 * locks


def test_wide_maps_get_looks_and_start_of_their_own():
    start = randomize.START_DEFAULTS
    for tier in (1, 3):
        for seed in range(20):
            plain = cli.make_map(tier, seed, profile="wide")
            m = cli.make_map(tier, seed, 1.0, start=start, profile="wide")
            assert m == cli.make_map(tier, seed, 1.0, start=start, profile="wide") and m["randomized"] and m["profile"] == "wide"
            assert (m["grid"], m["episode_ticks"]) == (plain["grid"], plain["episode_ticks"]) and plain["name"] == f"t{tier}_{seed:07d}"
            assert cli.make_map(tier, seed, 1.0, strip=True, start=start, profile="wide") == layouts.without_monsters(m)
            check_map(m, profile="wide")


@pytest.mark.parametrize("tier", [1, 2, 3, 4])
def test_wide_maps_play_in_the_engine(tmp_path, tier):
    """Wide maps load, step without warnings, and the reference walker reaches the exit inside the
    (tight) tick cap."""
    for seed in (0, 1):
        _engine_check(tmp_path, cli.make_map(tier, seed, randomize=float(seed), profile="wide"), steps=40, within=0.85)


def test_without_monsters_keeps_everything_but_the_monsters():
    """A monster-free copy has the same walls, doors, keys, pickups, start, looks, route and tick cap."""
    monster_types = {udmf.LEGEND[ch].thing for ch in udmf.MONSTER_CHARS}

    def thing_type(line: str) -> int:
        return int(line.split("type = ")[1].split(";")[0])

    assert layouts.strip_monsters("#P.zscidglok.hrE#") == "#P...........hrE#" and walker.strip_monsters is layouts.strip_monsters
    stripped = 0
    for tier in (1, 2, 3, 4):
        for seed in range(6):
            m = cli.make_map(tier, seed, randomize=1.0)
            before = json.dumps(m, sort_keys=True)
            bare = layouts.without_monsters(m)
            assert json.dumps(m, sort_keys=True) == before                              # the original is left alone
            assert bare == cli.make_map(tier, seed, randomize=1.0, strip=True) and list(bare) == list(m)
            assert all(a == b or (a in udmf.MONSTER_CHARS and b == ".") for a, b in zip(m["grid"], bare["grid"]))
            assert len(bare["grid"]) == len(m["grid"]) and not set(bare["grid"]) & udmf.MONSTER_CHARS
            assert bare["options"] == {k: v for k, v in m["options"].items() if k not in ("ambush", "thing_angles")}
            assert bare["counts"] == {**m["counts"], "monsters": 0, "ambush": 0}
            assert {k: v for k, v in bare.items() if k not in ("grid", "options", "counts")} == \
                   {k: v for k, v in m.items() if k not in ("grid", "options", "counts")}
            check_map(bare, bare=True)
            assert layouts.without_monsters(bare) == bare
            # compiled: the same vertices, sectors and lines, and the same things minus the monsters
            full_text = udmf.build_textmap(m["grid"], **m["options"]).splitlines()
            bare_text = udmf.build_textmap(bare["grid"], **bare["options"]).splitlines()
            assert [ln for ln in full_text if not ln.startswith("thing")] == [ln for ln in bare_text if not ln.startswith("thing")]
            assert [ln for ln in full_text if ln.startswith("thing") and thing_type(ln) not in monster_types] == \
                   [ln for ln in bare_text if ln.startswith("thing")]
            stripped += m["counts"]["monsters"]
    assert stripped > 50


@pytest.mark.parametrize("tier", [0, 3])
def test_lock_maps_without_monsters_are_finished_without_a_shot(tmp_path, tier):
    """Tier 0 maps with a lock exactly as generated, and tier 3 maps written with --strip-monsters:
    the reference walker (it never fires) fetches the key, opens the locked door and reaches the pad alive."""
    maps = [m for m in (cli.make_map(tier, seed, strip=tier > 0) for seed in range(30))
            if m["counts"]["locks"] and not m["counts"]["monsters"]][:3]
    assert len(maps) == 3
    for m in maps:
        assert not set(m["grid"]) & udmf.MONSTER_CHARS and solve(m["grid"]).keys_needed
        env = make_env(build(tmp_path, m["grid"], f"{tier}_{m['seed']}.wad", **m["options"]),
                       episode_ticks=m["episode_ticks"], render_buffers=False)
        result = walker.walk(env, m["grid"])
        dead = env.privileged_state()["dead"]
        env.close()
        assert result["exit_reached"] and not dead and sum(result["keys"]) == m["counts"]["keys"] >= 1
        assert 4 * result["steps"] <= 0.6 * m["episode_ticks"]


def test_style_variety():
    """Exit placements, corridor widths, loops and pillars all occur."""
    maps = [layouts.generate(tier, seed) for tier in (0, 2, 3) for seed in range(60)]
    assert {m["exit_style"] for m in maps} == set(layouts.EXIT_STYLES)
    assert {m["style"] for m in maps} >= {"strip", "arena", "house", "scatter", "maze", "cave"}
    sizes = [max(m["counts"]["width"], m["counts"]["height"]) for m in maps]
    assert min(sizes) <= 10 and max(sizes) >= 24


def _play(path: str, ticks: int, steps: int, seed: int) -> None:
    """Load a map and step it with random actions; any Python warning is an error."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = make_env(path, episode_ticks=ticks)
        obs, _ = env.reset(seed=seed)
        assert env._geometry.source == "wad"
        for _ in range(steps):
            obs, _, terminated, truncated, _ = env.step(rng.integers(0, 2, 9).astype(np.int8))
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        env.close()


def _engine_check(tmp_path, m: dict, steps: int, within: float = 0.6) -> None:
    """The map loads with the sectors and doors the grid describes, the player stands on its start
    cell, random play raises no warning, and (monsters removed) the reference walker reaches the exit
    well inside the tick cap: in at most the share `within` of it."""
    rows = m["grid"].split("\n")
    path = build(tmp_path, m["grid"], f"{m['tier']}_{m['seed']}.wad", **m["options"])
    env = make_env(path, episode_ticks=m["episode_ticks"], render_buffers=False)
    env.reset(seed=0)
    st = env.privileged_state()
    row = next(r for r, line in enumerate(rows) if "P" in line)
    offset = m["options"].get("start_offset", (0, 0))
    x, y = udmf.cell_centre(rows[row].index("P"), row, len(rows))
    assert (st["x"], st["y"]) == pytest.approx((x + offset[0], y + offset[1]))
    assert len(st["sector_floor"]) == m["counts"]["sectors"]
    locks = sorted(lock for lock in env.mapinfo.door_sectors.values() if lock)
    assert len(env.mapinfo.door_sectors) == m["counts"]["doors"] + m["counts"]["locks"] and len(locks) == m["counts"]["locks"]
    env.close()
    _play(path, m["episode_ticks"], steps, seed=m["seed"])
    if m["counts"]["exits"]:
        bare = layouts.without_monsters(m)
        env = make_env(build(tmp_path, bare["grid"], "bare.wad", **bare["options"]),
                       episode_ticks=m["episode_ticks"], render_buffers=False)
        result = walker.walk(env, bare["grid"])
        env.close()
        assert result["exit_reached"], (m["tier"], m["seed"])
        assert sum(result["keys"]) == m["counts"]["keys"]            # every key on the route was picked up
        assert 4 * result["steps"] <= within * m["episode_ticks"]


@pytest.mark.parametrize("tier", sorted(layouts.TIERS))
def test_generated_maps_play_in_the_engine(tmp_path, tier):
    for seed in (0, 1):
        _engine_check(tmp_path, cli.make_map(tier, seed, randomize=float(seed)), steps=40)


@pytest.mark.slow
@pytest.mark.parametrize("tier", sorted(layouts.TIERS))
def test_fifty_maps_per_tier_play_in_the_engine(tmp_path, tier):
    """Each tier: 50 maps (half of them randomised) load in doom_env, step without warnings and are
    walked to the exit by the reference walker within 60% of the tick cap."""
    for seed in range(100, 150):
        _engine_check(tmp_path, cli.make_map(tier, seed, randomize=0.5), steps=60)


# ---- command line -----------------------------------------------------------------------------------
def test_cli_writes_wads_and_manifest(tmp_path, capsys):
    args = ["--n", "6", "--tiers", "0,1,2,3,4", "--seed", "40", "--randomize", "0.5"]
    cli.main(["--out", str(tmp_path / "a")] + args)
    cli.main(["--out", str(tmp_path / "b")] + args)
    rows = cli.read_rows(tmp_path / "a" / "manifest.jsonl")
    assert len(rows) == 30 and [row["tier"] for row in rows] == sorted(row["tier"] for row in rows)
    assert len({row["grid"] for row in rows}) == 30
    assert 5 <= sum(row["randomized"] for row in rows) <= 25
    for row, other in zip(rows, cli.read_rows(tmp_path / "b" / "manifest.jsonl")):
        assert set(row) == {"name", "path", "file", "tier", "seed", "episode_ticks", "style", "exit_style",
                            "randomized", "on_grid", "grid", "options", "counts"}
        assert row["on_grid"] is randomize.start_on_grid(row["options"])
        wad = Path(row["path"])
        assert wad.name == row["file"] and wad.read_bytes() == udmf.build_wad(row["grid"], **row["options"])
        assert wad.read_bytes() == (tmp_path / "b" / other["file"]).read_bytes()     # reproducible
        assert {k: v for k, v in row.items() if k != "path"} == {k: v for k, v in other.items() if k != "path"}
        check_map(row)
        assert row["randomized"] or set(row["options"]) <= {"ambush"}
    assert "tier 4: 6 maps" in capsys.readouterr().out
    cli.main(["--stats", str(tmp_path / "a" / "manifest.jsonl")])
    assert "tick caps" in capsys.readouterr().out
    assert cli.verify(rows[::7])                                   # engine check of a few rows
    assert "reference walker reached it on" in capsys.readouterr().out
    broken = dict(rows[0], grid=rows[0]["grid"].replace("E", "."), counts=dict(rows[0]["counts"], sectors=99))
    with pytest.raises(RuntimeError):
        cli.verify_map(broken)


def test_cli_start_flags_keep_most_starts_on_the_grid(tmp_path, capsys):
    base = ["--n", "40", "--tiers", "0,2", "--seed", "700", "--randomize", "0.5"]
    cli.main(["--out", str(tmp_path / "v2"), *base, "--start-quarter", "0.5", "--start-any", "0.05", "--start-offset", "0.05"])
    assert "start on the grid" in capsys.readouterr().out
    rows = cli.read_rows(tmp_path / "v2" / "manifest.jsonl")
    assert len(rows) == 80 and 60 <= sum(row["on_grid"] for row in rows) < 80
    assert 20 <= sum(row["options"].get("start_angle") in (90, 180, 270) for row in rows) <= 60
    assert 20 <= sum(row["randomized"] for row in rows) <= 60
    for row in rows:
        assert row["on_grid"] is randomize.start_on_grid(row["options"])
        assert row["randomized"] or set(row["options"]) <= {"ambush", "start_angle", "start_offset"}
        assert Path(row["path"]).read_bytes() == udmf.build_wad(row["grid"], **row["options"])
        check_map(row)
    assert cli.verify([row for row in rows if not row["on_grid"]][:3] + rows[:2])
    cli.main(["--out", str(tmp_path / "default"), *base, "--start-any", "0.05"])       # the flags left out take the defaults
    assert cli.read_rows(tmp_path / "default" / "manifest.jsonl") == [
        dict(row, path=row["path"].replace("/v2/", "/default/")) for row in rows]
    cli.main(["--out", str(tmp_path / "on"), *base, "--start-quarter", "1", "--start-any", "0", "--start-offset", "0"])
    on = cli.read_rows(tmp_path / "on" / "manifest.jsonl")
    assert all(row["on_grid"] and row["options"]["start_angle"] in (90, 180, 270) for row in on)
    assert [row["grid"] for row in on] == [row["grid"] for row in rows]                # the start flags never change a layout
    cli.main(["--out", str(tmp_path / "off"), *base, "--start-quarter", "0", "--start-any", "0", "--start-offset", "1"])
    assert not any(row["on_grid"] for row in cli.read_rows(tmp_path / "off" / "manifest.jsonl"))
    cli.main(["--out", str(tmp_path / "old"), *base])                                  # no flag: the first set's start draw
    old = cli.read_rows(tmp_path / "old" / "manifest.jsonl")
    assert all(row["on_grid"] for row in old if not row["randomized"]) and sum(not row["on_grid"] for row in old) > 16
    for bad in (["--start-quarter", "0.8", "--start-any", "0.5"], ["--start-offset", "2"],
                ["--start-any", "0.1", "--from-suite", str(SUITES / "dev_v1.jsonl")]):
        with pytest.raises(SystemExit):
            cli.main(["--out", str(tmp_path / "bad"), *base, *bad])
    assert not (tmp_path / "bad").exists()


def test_cli_wide_profile_extends_a_set(tmp_path, capsys):
    """--extend copies a set as it is, --profile wide adds wide maps (one --n per tier), and every
    row names its profile; a set written without these flags has no such field."""
    cli.main(["--out", str(tmp_path / "v2"), "--n", "4", "--tiers", "0,2,3", "--seed", "500", "--randomize", "0.5", "--start-quarter", "0.5"])
    old = cli.read_rows(tmp_path / "v2" / "manifest.jsonl")
    assert not any("profile" in row for row in old)
    capsys.readouterr()
    wide_args = ["--profile", "wide", "--n", "2,3,2,3", "--tiers", "1,2,3,4", "--seed", "900", "--randomize", "0.5", "--start-quarter", "0.5"]
    cli.main(["--out", str(tmp_path / "v3"), "--extend", str(tmp_path / "v2" / "manifest.jsonl"), *wide_args])
    out = capsys.readouterr().out
    assert "after 12 copied" in out and "profile base" in out and "profile wide" in out and "tick cap over route cost" in out
    rows = cli.read_rows(tmp_path / "v3" / "manifest.jsonl")
    assert [row["profile"] for row in rows] == ["base"] * 12 + ["wide"] * 10
    assert [row["tier"] for row in rows[12:]] == [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
    assert all(tuple(row) == cli.ROW_KEYS for row in rows) and len({row["name"] for row in rows}) == 22
    for row, source in zip(rows, old):                             # copied: the same row and the same bytes in a new place
        assert {k: v for k, v in row.items() if k not in ("path", "profile")} == {k: v for k, v in source.items() if k != "path"}
        assert Path(row["path"]).parent == tmp_path / "v3" and Path(row["path"]).read_bytes() == Path(source["path"]).read_bytes()
    for row in rows[12:]:
        assert Path(row["path"]).read_bytes() == udmf.build_wad(row["grid"], **row["options"])
        assert row["on_grid"] is randomize.start_on_grid(row["options"]) and 900 <= row["seed"] < 910
        assert row["grid"] == layouts.generate(row["tier"], row["seed"], "wide")["grid"]
        check_map(row, profile="wide")
    assert cli.read_rows(tmp_path / "v2" / "manifest.jsonl") == old                   # the extended set is left alone
    cli.main(["--stats", str(tmp_path / "v3" / "manifest.jsonl"), "--profile", "wide"])
    out = capsys.readouterr().out
    assert "tier 4: 3 maps" in out and "tier 0" not in out and "profile" not in out
    assert cli.verify([row for row in rows[12:] if row["tier"] >= 2][::3])           # the walker finishes inside the tight cap
    cli.main(["--out", str(tmp_path / "w"), *wide_args])                                # a wide set on its own
    alone = cli.read_rows(tmp_path / "w" / "manifest.jsonl")
    assert [{k: v for k, v in row.items() if k != "path"} for row in alone] == [{k: v for k, v in row.items() if k != "path"} for row in rows[12:]]
    for bad in (["--out", str(tmp_path / "bad"), "--profile", "wide", "--n", "1,2", "--tiers", "1,2,3,4"],
                ["--out", str(tmp_path / "bad"), "--profile", "wide", "--n", "1", "--tiers", "0,1"],
                ["--out", str(tmp_path / "bad"), "--profile", "wide", "--from-suite", str(SUITES / "dev_v1.jsonl")],
                ["--out", str(tmp_path / "bad"), "--extend", str(tmp_path / "v2" / "manifest.jsonl"), "--profile", "wide",
                 "--n", "2", "--tiers", "2,3", "--seed", "500"],                           # names of the extended set
                ["--out", str(tmp_path / "w"), "--extend", str(tmp_path / "w" / "manifest.jsonl"), "--n", "1"]):
        with pytest.raises(SystemExit):
            cli.main(bad)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError):
        cli.make_set([1, 2], [1, 2, 3], seed=0, randomize=0.0, exclude=set())


def test_cli_keeps_training_and_suite_seeds_apart(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--out", str(tmp_path), "--n", "2", "--seed", str(cli.RESERVED_SEED)])
    with pytest.raises(FileExistsError):
        cli.freeze_suite(SUITES / "heldout_v1.jsonl", 1, [1], cli.RESERVED_SEED)
    with pytest.raises(ValueError):
        cli.freeze_suite(tmp_path / "new.jsonl", 1, [1], seed=0)
    frozen = cli.frozen_grids()
    train = cli.make_set(40, [1, 2, 3, 4], seed=0, randomize=0.0, exclude=frozen)
    assert not {cli.grid_hash(m["grid"]) for m in train} & frozen


# ---- frozen suites ----------------------------------------------------------------------------------
SUITE_SIZES = {"heldout_v1": 100, "dev_v1": 20}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("name", sorted(SUITE_SIZES))
def test_suites_are_frozen_and_rebuild_byte_stable(tmp_path, name):
    """The suite file and the WADs rebuilt from it match the recorded checksums, twice in a row."""
    path = SUITES / f"{name}.jsonl"
    rows = cli.read_rows(path)
    recorded = json.loads((SUITES / "checksums.json").read_text())[path.name]
    assert _sha256(path.read_bytes()) == recorded["file_sha256"]
    assert len(rows) == 4 * SUITE_SIZES[name] and len({row["name"] for row in rows}) == len(rows)
    for tier in (1, 2, 3, 4):
        assert sum(row["tier"] == tier for row in rows) == SUITE_SIZES[name]
    digests = []
    for run in ("first", "second"):
        cli.main(["--from-suite", str(path), "--out", str(tmp_path / run)])
        manifest = cli.read_rows(tmp_path / run / "manifest.jsonl")
        assert [row["name"] for row in manifest] == [row["name"] for row in rows]
        digest = hashlib.sha256()
        for row in manifest:
            digest.update(Path(row["path"]).read_bytes())
        digests.append(digest.hexdigest())
    assert digests[0] == digests[1] == recorded["wads_sha256"]
    for row in rows:
        assert row["seed"] >= cli.RESERVED_SEED and set(row["options"]) <= {"ambush"}   # reference dialect
        assert set(cli.SUITE_KEYS) == set(row)
        check_map(row)


def test_suites_do_not_overlap():
    grids = [{row["grid"] for row in cli.read_rows(SUITES / f"{name}.jsonl")} for name in sorted(SUITE_SIZES)]
    assert len(grids[0]) == 80 and len(grids[1]) == 400 and not grids[0] & grids[1]


NAV_SUITES = {"heldout_nav_v1": "heldout_v1", "dev_nav_v1": "dev_v1"}


@pytest.mark.parametrize("name", sorted(NAV_SUITES))
def test_navigation_suites_are_the_frozen_maps_without_monsters(name):
    """A navigation suite is tiers 2-4 of its full suite, row for row, with only the monsters removed,
    and it matches its recorded checksums."""
    path, full_name = SUITES / f"{name}.jsonl", NAV_SUITES[name]
    assert cli.suite_checksums(path) == json.loads((SUITES / "checksums.json").read_text())[path.name]
    rows = cli.read_rows(path)
    source = [row for row in cli.read_rows(SUITES / f"{full_name}.jsonl") if row["tier"] in (2, 3, 4)]
    assert len(rows) == len(source) == 3 * SUITE_SIZES[full_name]
    for row, full in zip(rows, source):
        assert row == {**layouts.without_monsters(full), "name": full["name"].replace(full_name, name)}
        assert row["episode_ticks"] == full["episode_ticks"] and row["counts"]["locks"] == full["counts"]["locks"]
        check_map(row, bare=True)


WIDE_SUITES = {"heldout_wide_v1": 50, "dev_wide_v1": 15}


@pytest.mark.parametrize("name", sorted(WIDE_SUITES))
def test_wide_suites_are_frozen(name):
    """The two suites of the wide profile: recorded checksums, tiers 1 to 4, reserved seeds, the
    reference dialect, no grid shared with another suite, and rows the wide profile still generates."""
    path = SUITES / f"{name}.jsonl"
    rows = cli.read_rows(path)
    assert cli.suite_checksums(path) == json.loads((SUITES / "checksums.json").read_text())[path.name]
    assert len(rows) == 4 * WIDE_SUITES[name] and [row["name"] for row in rows] == \
        [f"{name}_t{tier}_{i:03d}" for tier in (1, 2, 3, 4) for i in range(WIDE_SUITES[name])]
    for row in rows:
        assert row["seed"] >= cli.RESERVED_SEED and set(row["options"]) <= {"ambush"} and set(cli.SUITE_KEYS) == set(row)
        check_map(row, profile="wide")
    for row in rows[::7]:
        m = layouts.generate(row["tier"], row["seed"], "wide")
        assert (m["grid"], m["episode_ticks"], m["options"], m["counts"]) == (row["grid"], row["episode_ticks"], row["options"], row["counts"])
    others = set()
    for other in sorted(SUITES.glob("*.jsonl")):
        if other != path:
            others |= {row["grid"] for row in cli.read_rows(other)}
    assert len({row["grid"] for row in rows}) == len(rows) and not {row["grid"] for row in rows} & others
    with pytest.raises(FileExistsError):
        cli.freeze_suite(path, 1, [1], cli.RESERVED_SEED, profile="wide")


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(WIDE_SUITES))
def test_wide_suite_maps_play_in_the_engine(tmp_path, name):
    rows = cli.read_rows(SUITES / f"{name}.jsonl")
    for row in rows[:: max(1, len(rows) // 40)]:
        _engine_check(tmp_path, row, steps=40, within=0.85)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SUITE_SIZES))
def test_suite_maps_play_in_the_engine(tmp_path, name):
    rows = cli.read_rows(SUITES / f"{name}.jsonl")
    for row in rows[:: max(1, len(rows) // 40)]:
        _engine_check(tmp_path, row, steps=40)
