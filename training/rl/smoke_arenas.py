"""Tiny single-room combat arenas for the learning smoke test.

    python -m training.rl.smoke_arenas --out data/maps/smoke_imps --n 12 --seed 0 [--monsters i --count 4,6]

Training-time scaffolding, never imported by the agent.

Each map is one rectangular room of 7 to 10 cells a side with the player start, a few monsters,
a box of bullets and an ammo clip, no exit and no doors, compiled with the map generator's own
``training.mapgen.udmf.write_wad``. With no exit the only rewards are the combat terms, so mean
kills per episode is a clean signal of whether the policy-gradient loop learns.

The default monster is the imp (``i``): it takes about six pistol hits, so a kill needs sustained
aim and firing, and imps cannot hurt each other, so no kill comes from monsters fighting among
themselves. ``--monsters zi`` mixes zombiemen and imps (first letter most frequent). Writes the
WADs and a ``manifest.jsonl`` (tier 1, 2100 tics) next to them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training import paths  # noqa: F401
from training.mapgen import udmf


def tiny_arena(rng: np.random.Generator, monsters: str = "i", count: tuple = (4, 6)) -> tuple[str, dict]:
    """One arena as an ASCII grid plus its counts.

    monsters: legend letters to draw from; the first is used with probability 0.7 when there are
    several. count: inclusive range of the number of monsters.
    """
    w, h = int(rng.integers(7, 11)), int(rng.integers(7, 11))
    cells = [["#"] * (w + 2)] + [["#"] + ["."] * w + ["#"] for _ in range(h)] + [["#"] * (w + 2)]
    px, py = int(rng.integers(1, w + 1)), int(rng.integers(1, h + 1))
    cells[py][px] = "P"
    free = [(x, y) for y in range(1, h + 1) for x in range(1, w + 1) if max(abs(x - px), abs(y - py)) >= 3]
    order = rng.permutation(len(free))
    n_monsters = int(rng.integers(count[0], count[1] + 1))
    for k in order[:n_monsters]:
        x, y = free[k]
        rest = monsters[1:]
        cells[y][x] = monsters[0] if (not rest or rng.random() < 0.7) else rest[int(rng.integers(len(rest)))]
    for k, pickup in zip(order[n_monsters:n_monsters + 2], "xn"):       # a box of bullets and a clip
        x, y = free[k]
        cells[y][x] = pickup
    grid = "\n".join("".join(row) for row in cells)
    counts = {"width": w + 2, "height": h + 2, "open_cells": w * h, "rooms": 1, "doors": 0, "exits": 0,
              "monsters": n_monsters, "pickups": 2}
    return grid, counts


def write_set(out: str, n: int = 12, seed: int = 0, episode_ticks: int = 2100, monsters: str = "i",
              count: tuple = (4, 6)) -> list:
    """Write ``n`` arenas and their manifest into ``out``; returns the manifest rows."""
    for ch in monsters:
        if ch not in udmf.MONSTER_CHARS:
            raise ValueError(f"{ch!r} is not a monster in the map legend")
    rng = np.random.default_rng(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        grid, counts = tiny_arena(rng, monsters, count)
        name = f"tiny_{seed}_{i:03d}"
        udmf.write_wad(grid, out_dir / f"{name}.wad")
        rows.append({"name": name, "path": str((out_dir / f"{name}.wad").resolve()), "tier": 1, "seed": seed,
                     "episode_ticks": episode_ticks, "style": "tiny_arena", "grid": grid, "options": {}, "counts": counts})
    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Write tiny single-room combat arenas and their manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--monsters", default="i", help="legend letters of the monsters to place (default: imps)")
    ap.add_argument("--count", default="4,6", help="inclusive range of monsters per arena")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.count.split(","))
    rows = write_set(args.out, args.n, args.seed, args.episode_ticks, args.monsters, (lo, hi))
    print(f"{len(rows)} arenas -> {args.out}/manifest.jsonl")
    print(rows[0]["grid"])


if __name__ == "__main__":
    main()
