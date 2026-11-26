"""Tests of the reference profiles of the map generator (`layouts.REFERENCE`): maps held to the
legend of the reference maps in maps/ and tests/fixtures/. Run: .venv/bin/python -m pytest training/tests/test_mapgen_reference.py -q"""

import hashlib
import json
import re
import statistics
from pathlib import Path

import pytest

from training.mapgen import cli, layouts, udmf
from training.mapgen.solver import solve

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "training" / "suites"
REFERENCE_WADS = sorted((ROOT / "tests" / "fixtures").glob("*.wad")) + [ROOT / "maps" / "two_rooms.wad"]
MONSTERS, PICKUPS = set("zic"), set("h")
# Structure, keys and locked doors as every profile places them, and the monsters and pickup above.
ALLOWED = set("#.PED") | set(udmf.LOCKED_DOOR_CHARS) | set(udmf.KEY_CHARS) | MONSTERS | PICKUPS
REF_SUITES = {"dev_ref_v1": 10, "heldout_ref_v1": 30}     # maps of each profile in every tier


def _things(path: Path) -> set[int]:
    return {int(n) for n in re.findall(r"thing\s*(?://[^\n]*)?\s*\{[^}]*?type\s*=\s*(\d+)", udmf.read_textmap(path))}


def _maps(profile: str, tier: int, n: int = 40) -> list[dict]:
    return [layouts.generate(tier, seed, profile) for seed in range(n)]


def test_reference_maps_show_three_monsters_and_one_pickup():
    """What the 13 fixtures and maps/two_rooms.wad contain is what the reference profiles may place."""
    assert len(REFERENCE_WADS) == 14
    seen = set().union(*(_things(path) for path in REFERENCE_WADS))
    assert seen == {1, 13, 3004, 3001, 65, 2011}       # start, red keycard, zombieman, imp, chaingunner, stimpack
    number = {s.thing: ch for ch, s in udmf.LEGEND.items() if s.thing}
    assert {number[n] for n in seen if udmf.LEGEND[number[n]].group == "monster"} == MONSTERS
    assert {number[n] for n in seen if number[n] in udmf.PICKUP_CHARS} == PICKUPS
    for name in layouts.REFERENCE_PROFILES:
        ref = layouts.REFERENCE[name]
        assert set(ref["pickups"]) == PICKUPS and sorted(ref["mix"]) == [1, 2, 3, 4]
        assert all({kind for kind, _ in ref["mix"][tier]} == MONSTERS for tier in ref["mix"])


@pytest.mark.parametrize("profile", layouts.REFERENCE_PROFILES)
def test_reference_profiles_keep_to_the_legend(profile):
    """Tiers 1 to 4: only zombieman, imp and chaingunner, only stimpacks, one reachable exit on
    every map (the arenas included), solvable with every lock needed, and a pure function of the seed."""
    ref = layouts.REFERENCE[profile]
    for tier in (1, 2, 3, 4):
        maps = _maps(profile, tier)
        kinds = set()
        for m in maps:
            grid, counts = m["grid"], m["counts"]
            assert set(grid) - {"\n"} <= ALLOWED and m["profile"] == profile and m["tier"] == tier
            assert {ch for ch in grid if ch in udmf.MONSTER_CHARS} <= MONSTERS
            assert {ch for ch in grid if ch in udmf.PICKUP_CHARS} <= PICKUPS
            kinds |= {ch for ch in grid if ch in MONSTERS}
            sol = solve(grid)
            assert grid.count("E") == counts["exits"] == 1 and grid.count("P") == 1
            assert sol.solvable and sol.all_reachable and sol.exit_reachable and counts["optimal_steps"] == sol.optimal_steps
            assert counts["keys"] == counts["locks"] and bool(counts["locks"]) == bool(sol.keys_needed)
            assert counts["locks"] >= {1: 0, 2: 0, 3: 1, 4: 2}[tier] and (tier > 2 or counts["locks"] == 0)
            least = 0 if tier in ref["free"] else ref["least"][tier]
            assert least <= counts["monsters"] <= ref["monsters"][tier][0]
            assert m["episode_ticks"] in range(2100, 4201, 350)
            assert max(counts["width"], counts["height"]) <= 28
        assert kinds == MONSTERS                                   # all three kinds in every tier
        again = layouts.generate(tier, 7, profile)
        assert again == maps[7] and again["grid"] != layouts.generate(tier, 7)["grid"]
    for tier in (0, 5):
        with pytest.raises(ValueError):
            layouts.generate(tier, 0, profile)


def test_reference_wide_has_long_routes_and_crowded_rooms():
    """`reference_wide` against `reference`: more monsters in every tier, more chaingunners and imps
    in tier 4, and in tiers 2 to 4 a longer route under the tight tick cap of the wide profile."""
    least, most = layouts.WIDE["budget"]
    for tier in (1, 2, 3, 4):
        normal, hard = _maps("reference", tier), _maps("reference_wide", tier)
        count = lambda maps: [m["counts"]["monsters"] for m in maps]
        assert statistics.mean(count(hard)) >= 1.8 * statistics.mean(count(normal)) and max(count(hard)) > max(count(normal))
        assert min(count(hard)) >= layouts.REFERENCE["reference_wide"]["least"][tier]
        for m in normal:
            assert m["episode_ticks"] >= layouts.MIN_BUDGET * 4 * m["counts"]["optimal_steps"]
        if tier >= 2:
            route = lambda maps: statistics.mean(m["counts"]["route_cells"] for m in maps)
            assert route(hard) > 1.3 * route(normal)
            for m in hard:
                assert least * 4 * m["counts"]["optimal_steps"] <= m["episode_ticks"] <= most * 4 * m["counts"]["optimal_steps"]
    hard = "".join(m["grid"] for m in _maps("reference_wide", 4))
    assert max(m["counts"]["monsters"] for m in _maps("reference_wide", 4)) >= 20
    assert hard.count("c") + hard.count("i") >= 4 * hard.count("z") and hard.count("c") >= 200
    assert statistics.mean(m["counts"]["locks"] for m in _maps("reference_wide", 4)) >= 2


def test_old_profiles_are_what_they_were_before_the_reference_profiles():
    """The base and wide profiles give the maps they gave before the reference profiles existed
    (digests taken from the code as it was), and their names are still the tuple `PROFILES`."""
    base = [layouts.generate(tier, seed) for tier in range(5) for seed in range(16)]
    assert hashlib.sha1(json.dumps(base, sort_keys=True).encode()).hexdigest() == "766dd2a082e428438edd656faeb43f18f51b25cd"
    wide = [layouts.generate(tier, seed, "wide") for tier in (1, 2, 3, 4) for seed in range(8)]
    assert hashlib.sha1(json.dumps(wide, sort_keys=True).encode()).hexdigest() == "5e4e825a30ada2cbb9c76e1ea644ce25e71bff23"
    made = [cli.make_map(tier, seed, 0.5, start=(0.5, 0.05, 0.05), profile=profile)
            for profile in ("base", "wide") for tier in (1, 2, 3, 4) for seed in range(100, 104)]
    assert hashlib.sha1(json.dumps(made, sort_keys=True).encode()).hexdigest() == "61a50a9569370d402d6a3b1e7f572b268e52040f"
    assert layouts.PROFILES == ("base", "wide") and layouts.ALL_PROFILES == ("base", "wide", "reference", "reference_wide")
    assert not any("profile" in m for m in base) and {m["profile"] for m in wide} == {"wide"}
    with pytest.raises(ValueError):
        layouts.generate(2, 0, "reference_huge")


def test_cli_reference_profiles_write_sets_and_mixed_suites(tmp_path, capsys):
    """--profile reference and reference_wide write a set (the second extending the first), and
    --mix freezes a suite with both profiles in every tier, named in its rows."""
    flags = ["--n", "2,3,2,3", "--tiers", "1,2,3,4", "--randomize", "0.5", "--start-quarter", "0.5"]
    cli.main(["--out", str(tmp_path / "part"), "--profile", "reference", "--seed", "300", *flags])
    cli.main(["--out", str(tmp_path / "set"), "--extend", str(tmp_path / "part" / "manifest.jsonl"),
              "--profile", "reference_wide", "--seed", "700", *flags])
    out = capsys.readouterr().out
    assert "profile reference\n" in out and "profile reference_wide\n" in out
    rows = cli.read_rows(tmp_path / "set" / "manifest.jsonl")
    assert [row["profile"] for row in rows] == ["reference"] * 10 + ["reference_wide"] * 10
    assert [row["tier"] for row in rows] == [1, 1, 2, 2, 2, 3, 3, 4, 4, 4] * 2 and len({row["name"] for row in rows}) == 20
    for row in rows:
        assert tuple(row) == cli.ROW_KEYS and row["counts"]["exits"] == 1
        assert Path(row["path"]).read_bytes() == udmf.build_wad(row["grid"], **row["options"])
        assert row["grid"] == layouts.generate(row["tier"], row["seed"], row["profile"])["grid"]
        assert {ch for ch in row["grid"] if ch in udmf.MONSTER_CHARS} <= MONSTERS
    suite = tmp_path / "mixed_v0.jsonl"
    cli.main(["--freeze", str(suite), "--profile", "reference", "--n", "2", "--tiers", "1,4",
              "--seed", str(cli.RESERVED_SEED + 990_000), "--mix", f"reference_wide:{cli.RESERVED_SEED + 995_000}"])
    frozen = cli.read_rows(suite)
    assert [(row["tier"], row["profile"]) for row in frozen] == \
        [(tier, profile) for tier in (1, 4) for profile in ("reference", "reference_wide") for _ in range(2)]
    assert [row["name"] for row in frozen] == [f"mixed_v0_t{tier}_{i:03d}" for tier in (1, 4) for i in range(4)]
    assert all(tuple(row) == cli.SUITE_KEYS + ("profile",) and set(row["options"]) <= {"ambush"} for row in frozen)
    for bad in (["--out", str(tmp_path / "bad"), "--profile", "reference", "--n", "1", "--tiers", "0,1"],
                ["--out", str(tmp_path / "bad"), "--profile", "reference", "--n", "1", "--mix", "reference_wide:9990000"],
                ["--freeze", str(tmp_path / "bad.jsonl"), "--n", "1", "--tiers", "1", "--seed", "9990000", "--mix", "huge:9991000"],
                ["--freeze", str(tmp_path / "bad.jsonl"), "--n", "1", "--tiers", "1", "--seed", "9990000", "--mix", "reference"]):
        with pytest.raises(SystemExit):
            cli.main(bad)
    with pytest.raises(FileExistsError):
        cli.freeze_mixed_suite(suite, 1, [1], (("reference", cli.RESERVED_SEED), ("reference_wide", cli.RESERVED_SEED)))
    with pytest.raises(ValueError):
        cli.freeze_mixed_suite(tmp_path / "low.jsonl", 1, [1], (("reference", cli.RESERVED_SEED), ("reference_wide", 5)))
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.jsonl").exists() and not (tmp_path / "low.jsonl").exists()


@pytest.mark.parametrize("name", sorted(REF_SUITES))
def test_reference_suites_are_frozen(name):
    """The two reference suites: recorded checksums, tiers 1 to 4 with the first half of every tier
    from `reference` and the second from `reference_wide`, reserved seeds, the reference dialect,
    three monsters, stimpacks, an exit everywhere, no grid shared with another suite, and rows the
    profiles still generate."""
    path, n = SUITES / f"{name}.jsonl", REF_SUITES[name]
    rows = cli.read_rows(path)
    assert cli.suite_checksums(path) == json.loads((SUITES / "checksums.json").read_text())[path.name]
    assert len(rows) == 8 * n and [row["name"] for row in rows] == [f"{name}_t{tier}_{i:03d}" for tier in (1, 2, 3, 4) for i in range(2 * n)]
    assert [row["profile"] for row in rows] == (["reference"] * n + ["reference_wide"] * n) * 4
    for row in rows:
        assert row["seed"] >= cli.RESERVED_SEED and set(row["options"]) <= {"ambush"}
        assert tuple(row) == cli.SUITE_KEYS + ("profile",)
        assert set(row["grid"]) - {"\n"} <= ALLOWED and row["counts"]["exits"] == row["grid"].count("E") == 1
        sol = solve(row["grid"])
        assert sol.solvable and sol.exit_reachable and bool(row["counts"]["locks"]) == bool(sol.keys_needed)
    for row in rows[::7]:
        m = layouts.generate(row["tier"], row["seed"], row["profile"])
        assert (m["grid"], m["episode_ticks"], m["options"], m["counts"]) == (row["grid"], row["episode_ticks"], row["options"], row["counts"])
    others = set()
    for other in sorted(SUITES.glob("*.jsonl")):
        if other != path:
            others |= {row["grid"] for row in cli.read_rows(other)}
    assert len({row["grid"] for row in rows}) == len(rows) and not {row["grid"] for row in rows} & others
    for tier in (1, 2, 3, 4):
        part = [row["counts"]["monsters"] for row in rows if row["tier"] == tier]
        assert statistics.mean(part[n:]) > 1.8 * statistics.mean(part[:n])          # the hard half is the crowded one
    with pytest.raises(FileExistsError):
        cli.freeze_mixed_suite(path, 1, [1], (("reference", cli.RESERVED_SEED), ("reference_wide", cli.RESERVED_SEED)))


def test_reference_maps_play_in_the_engine():
    """Two maps of each profile and tier from the development suite load in `doom_env`, play without
    warnings, and the reference walker reaches the exit on the copy without monsters."""
    rows = cli.read_rows(SUITES / "dev_ref_v1.jsonl")
    assert cli.verify(rows[::5])
