"""What the exit detector is worth to navigation, and what happens when it says nothing.

Training-time scaffolding. Never imported by the agent in submission/.

    python -m training.exitdet.fallback --suite training/suites/heldout_v1.jsonl --workers 3

The frame tables of `evaluate.py` say how often the detector sees a pad. This module measures what
depends on it: the exit rate of the navigation teacher (`training.teacher`) on the monster-free
rebuilds of a frozen suite, tiers 2 to 4, under three sources of exit observations:

    label      the geometric exit observation of `training.exit_label` (a perfect detector)
    detector   the trained detector on the rendered frame, exactly as the agent calls it: uint8
               frame in, sigmoid of the visibility output as the confidence, a sighting from 0.5
    none       no exit observation at all: the detector silent on every frame, which is what a
               pad drawn with another flat than the one it was trained on amounts to (see the
               `padflats` rows of the evaluation table)

Without an exit in its map the teacher can only sweep the cells it has not stood on, because
walking onto the pad still ends the map. The `none` rows are therefore the floor under the exit
rate when detection fails completely, and the gap between `detector` and `label` is what the
detector's misses and false sightings cost.

The teacher does not fight, so the maps are played without monsters; the time budget handed to it
is the true cap of each map, and again the shortest cap (525 decision steps) for every map, which
is the condition of the final evaluation. One row per (source, budget, tier) goes to
training/logs/exit_detector_fallback.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from training import paths
from doomnet import spec
from training.exitdet import evaluate as ev

LOG_CSV = paths.ROOT / "training" / "logs" / "exit_detector_fallback.csv"
MAPS_DIR = paths.ROOT / "data" / "exitdet" / "fallback_maps"
SOURCES = ("label", "detector", "none")
POLICY = "training.exitdet.fallback:teacher_with"
COLUMNS = ("tier", "episodes", "exit_rate", "sectors_frac", "exit_seen_timeout", "stuck_rate", "steps_to_exit",
           "steps_frac_of_cap", "door_reversals", "agent_errors")


class DetectorSensor:
    """Drop-in for the teacher's exit sensor: the trained detector on the frame of the current step."""

    def __init__(self, weights=None):
        import torch

        self._torch = torch
        torch.set_num_threads(1)
        self.model = ev.load_detector(weights or ev.WEIGHTS)[0]
        self.frame = None
        self.sightings = 0

    def attach(self, env, meta) -> None:       # the detector needs neither
        pass

    def seed(self, seed: int) -> None:
        self.sightings = 0

    def read(self):
        torch = self._torch
        if self.frame is None:
            return None
        frame = np.ascontiguousarray(self.frame, np.uint8)
        with torch.inference_mode():
            out = self.model(torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0))[0]
            out[0] = torch.sigmoid(out[0])
        self.sightings += int(float(out[0]) >= 0.5)
        return out.numpy().astype(np.float32)


def teacher_with(exit_source: str = "label", weights: str = "", **kwargs):
    """Policy factory for the runner: the teacher with one of SOURCES as its exit observations."""
    from training.teacher.policy import TeacherPolicy

    if exit_source != "detector":
        return TeacherPolicy(exit_source=exit_source, **kwargs)

    class DetectorTeacher(TeacherPolicy):
        def act(self, obs):
            self.sensor.frame = obs.get("rgb")
            return super().act(obs)

    policy = DetectorTeacher(exit_source="none", **kwargs)
    policy.sensor = DetectorSensor(weights or None)
    return policy


def run(suite, tiers=(2, 3, 4), sources=SOURCES, budgets=("cap", float(spec.DEFAULT_BUDGET_STEPS)), workers: int = 3,
        limit: int | None = None, weights: str = "", log=print) -> list[dict]:
    """Play the monster-free suite once per (budget, source); returns one row per tier and for all tiers."""
    from training.runner import run_episodes
    from training.teacher import gate

    maps = gate.bare_maps(suite, tiers, limit, "aligned", out_dir=MAPS_DIR)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sha = ev.file_sha256(weights or ev.WEIGHTS)[:12]
    out = []
    for budget in budgets:
        for source in sources:
            began = time.perf_counter()
            rows = run_episodes((POLICY, {"exit_source": source, "weights": str(weights or "")}), maps, workers=workers,
                                seeds=[0], render_buffers=source == "detector",
                                record=(gate.RECORDER, {"dump": "", "budget": budget}))
            for r in gate.gate_table(rows):
                out.append({"stamp": stamp, "weights_sha256": sha if source == "detector" else "", "suite": Path(suite).name,
                            "maps": "monster-free rebuilds", "exit_source": source, "budget": str(budget),
                            **{k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in COLUMNS}})
            pooled = out[-1]
            log(f"{source:>8}, budget {budget}: exit rate {pooled['exit_rate']:.3f} over {pooled['episodes']} maps, sectors "
                f"{pooled['sectors_frac']:.3f}, steps to exit {pooled['steps_to_exit']:.0f}  ({time.perf_counter() - began:.0f} s)")
    return out


def write_csv(rows: list[dict], path=LOG_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.exitdet.fallback", description=__doc__.split("\n")[0])
    ap.add_argument("--suite", default="training/suites/heldout_v1.jsonl")
    ap.add_argument("--tiers", default="2,3,4")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first N maps of every tier only")
    ap.add_argument("--weights", default="", help="detector weights (default: the published ones)")
    ap.add_argument("--csv", default=str(LOG_CSV), help="where the table goes ('' = print only)")
    args = ap.parse_args(argv)
    rows = run(args.suite, [int(t) for t in args.tiers.split(",")], [s for s in args.sources.split(",") if s],
               workers=args.workers, limit=args.limit, weights=args.weights)
    if args.csv:
        print("wrote", write_csv(rows, args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
