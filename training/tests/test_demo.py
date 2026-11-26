"""Tests for the debug viewer (training/demo.py).

The fast tests draw panels from made-up state and play short episodes (40 to 60 decision steps) on
a reference map and on a generated map. The slow ones time a full 525-step episode and view the
submission's agent and the navigation teacher when those exist:

    .venv/bin/python -m pytest training/tests/test_demo.py -m slow -s
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import pkgutil
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from doomnet import spec  # noqa: E402
from doomnet.memory import MemoryMap, crop_origin  # noqa: E402
from doomnet.odometry import Odometry  # noqa: E402
from fixtures import fixture_wad  # noqa: E402
from training import demo as D  # noqa: E402
from training import evaluate as E  # noqa: E402
from training import runner as R  # noqa: E402
from training.mapgen import layouts, udmf  # noqa: E402
from training.mapmeta import load_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TWO_ROOMS = str(ROOT / "maps" / "two_rooms.wad")
EXIT = "#######\n#P...E#\n#######"
KEY = "############\n#P.r..R...E#\n############"
GUNNER = "##########\n#P......c#\n##########"
C = D.CENTER


def has_ffmpeg() -> bool:
    exe = shutil.which("ffmpeg")
    return bool(exe) and D.find_encoder(exe) is not None


def trace_rows(summary) -> list[dict]:
    with open(summary["trace"], newline="") as f:
        return list(csv.DictReader(f))


def check_outputs(summary, video=True) -> None:
    """Every promised file is there and is what it says it is."""
    out = Path(summary["json"]).parent
    steps = summary["row"]["steps"]
    assert json.loads((out / "episode.json").read_text())["row"]["steps"] == steps
    assert len(trace_rows(summary)) == steps
    assert [Path(p).name for p in summary["sheets"]] == ["sheet_even.png", "sheet_final.png"]
    for sheet in summary["sheets"]:
        with Image.open(sheet) as im:
            assert im.size == (D.SHEET_COLS * D.PANEL_W + (D.SHEET_COLS + 1) * D.SHEET_GAP,
                               D.SHEET_HEADER_H + D.SHEET_ROWS * D.PANEL_H + (D.SHEET_ROWS + 1) * D.SHEET_GAP)
    assert summary["panels"] == steps + 1                       # one per step and the outcome frame
    if not video:
        assert summary["video"] is None and not list(out.glob("video.*"))
        return
    hold = max(1, round(D.END_HOLD_S * summary["fps"]))
    path = Path(summary["video"])
    assert path.is_file() and path.stat().st_size > 1000
    if has_ffmpeg():
        assert path.name == "video.mp4" and path.read_bytes()[4:8] == b"ftyp"
        assert summary["video_frames"] == steps + hold
    else:
        assert path.name == "video.gif"
        with Image.open(path) as im:
            assert im.size == (D.PANEL_W, D.PANEL_H) and im.n_frames == summary["video_frames"]


def sheet_panel(path, k: int) -> Image.Image:
    """Panel k (row-major) cut out of a contact sheet."""
    row, col = divmod(k, D.SHEET_COLS)
    x0 = D.SHEET_GAP + col * (D.PANEL_W + D.SHEET_GAP)
    y0 = D.SHEET_HEADER_H + D.SHEET_GAP + row * (D.PANEL_H + D.SHEET_GAP)
    with Image.open(path) as im:
        return im.convert("RGB").crop((x0, y0, x0 + D.PANEL_W, y0 + D.PANEL_H))


def frame_state(**over) -> dict:
    """A complete panel state with plain values, for the drawing tests."""
    f = {"step": 7, "steps_cap": 525, "tic": 28, "ticks": 2100, "rgb": np.zeros((spec.SCREEN_H, spec.SCREEN_W, 3), np.uint8),
         "crop": np.zeros((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.float32), "view": D.View.centred(1),
         "frame_pose": (0.0, 0.0, 0.0), "est": (0.0, 0.0, 0.0), "true": (0.0, 0.0, 0.0), "drift": 0.0, "heading_err": 0.0,
         "pose_source": "shadow", "map_source": "shadow memory", "action": np.zeros(9, np.int8), "masks": (False, False),
         "totals": {}, "subgoal": None, "exit_believed": None, "note": "", "title": "test", "error": "",
         "target_text": None, "target_box": None}
    f.update(over)
    return f


class HookedPolicy:
    """A scripted policy with every hook the viewer reads: its own odometry and memory map, a pose
    estimate, an exit belief and a debug state with a sub-goal, a target cell and extra values."""

    def __init__(self, script: str = "F*20 L*6 F*14", memory_as: str = "attribute", pose_offset: float = 0.0, debug=None):
        self.debug = debug
        self.script = R.ScriptPolicy(script)
        self.odometry = Odometry()
        store = MemoryMap()
        if memory_as == "attribute":
            self.memory = store
        elif memory_as == "method":
            self.memory_map = lambda: store
        else:                                   # an attribute with a name the viewer does not know
            self.spatial = store
        self._store, self.pose_offset = store, float(pose_offset)
        self.seen = []

    def reset(self) -> None:
        self.script.reset()
        self.odometry.reset()
        self._store.reset()
        self.prev, self.t, self.seen = None, 0, []

    def act(self, obs) -> np.ndarray:
        if self.prev is not None:
            self.odometry.update(self.prev, obs)
        self._store.update(self.odometry.pose, obs, self.prev, None, self.t)
        self.seen.append(np.array(obs["rgb"]))
        self.prev = self.script.act(obs)
        self.t += 1
        return self.prev

    def pose_estimate(self):
        x, y, theta = self.odometry.pose
        return x + self.pose_offset, y, theta

    def exit_believed(self) -> bool:
        return self.t > 10

    def debug_state(self) -> dict:
        if self.debug is not None:
            return self.debug(self)
        return {"subgoal": spec.SUBGOAL_INDEX["open_door"], "target": (C + 3) * spec.CROP_SIZE + (C - 1),
                "target_origin": crop_origin(self.pose_estimate()),      # chosen anew at every step, in the current crop
                "use_age": 3, "note": "hello", "heads": np.arange(6), "value_field": np.zeros((64, 64), np.float32)}


class Budgeted(R.ScriptPolicy):
    """A script with the teacher's `set_budget(steps)` hook; the budget in force shows in its debug state."""

    def __init__(self, script: str = "F*4", delay: float = 0.0):
        super().__init__(script)
        self.budget, self.delay, self.calls = 525.0, float(delay), []

    def reset(self) -> None:
        super().reset()
        self.budget = 525.0

    def set_budget(self, steps: float) -> None:
        self.budget = float(steps)
        self.calls.append(self.budget)

    def act(self, obs) -> np.ndarray:
        if self.delay:
            time.sleep(self.delay)
        return super().act(obs)

    def debug_state(self) -> dict:
        return {"budget": self.budget}


CORRIDOR = "#" * 22 + "\n#P" + "." * 19 + "#\n" + "#" * 22
FAR_CELL, OTHER_CELL = (12, 3), (4, -3)                 # start-frame cells beside the corridor, clear of the agent's marker


class Manager:
    """Walks forward and reports a manager's target the way the submission's agent holds it: a cell is
    chosen every `period` steps (and at the steps in `extra`) as an index into the crop of that step,
    and the index is then reported unchanged until the next choice.

    The cell is FAR_CELL (OTHER_CELL at the steps in `extra`), or `ahead` cells in front of the agent's
    own cell, which gives the same index at every choice. `report` says what goes with the index:
    nothing ("index"), the crop origin of the choice ("origin") or the start-frame cell ("cell").
    `cells[t]` and `poses[t]` keep the cell meant and the pose estimate at step t.
    """

    def __init__(self, report: str = "index", period: int = spec.MANAGER_PERIOD, ahead: int | None = None, extra=(),
                 pair: bool = False):
        self.report, self.period, self.ahead, self.extra, self.pair = report, int(period), ahead, tuple(extra), pair
        self.odometry, self.memory = Odometry(), MemoryMap()

    def reset(self) -> None:
        self.odometry.reset()
        self.memory.reset()
        self.prev, self.t, self.cells, self.poses = None, 0, [], []
        self.origin, self.cell = (-C, -C), (0, 0)

    def act(self, obs) -> np.ndarray:
        if self.prev is not None:
            self.odometry.update(self.prev, obs)
        pose = tuple(float(v) for v in self.odometry.pose)
        self.memory.update(pose, obs, self.prev, None, self.t)
        if self.t % self.period == 0 or self.t in self.extra:
            self.origin = crop_origin(pose)
            if self.ahead is not None:
                self.cell = (self.origin[0] + C + self.ahead, self.origin[1] + C + 3)
            else:
                self.cell = OTHER_CELL if self.t in self.extra else FAR_CELL
        self.cells.append(self.cell)
        self.poses.append(pose)
        self.prev = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
        self.t += 1
        return self.prev

    def pose_estimate(self):
        return self.odometry.pose

    def debug_state(self) -> dict:
        i, j = self.cell[0] - self.origin[0], self.cell[1] - self.origin[1]
        state = {"subgoal": "explore", "target": (i, j) if self.pair else i * spec.CROP_SIZE + j}
        if self.report == "origin":
            state["target_origin"] = self.origin
        elif self.report == "cell":
            state["target_cell"] = self.cell
        return state


class Wrapper:
    """Keeps the real policy one level down in `.agent`, the way runner.AgentPolicy does."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self) -> None:
        self.agent.reset()

    def act(self, obs):
        return self.agent.act(obs)


class Failing:
    """Walks forward, then raises (or returns a bad action) at step `at`."""

    def __init__(self, at: int = 5, bad_action: bool = False):
        self.at, self.bad_action, self.t = at, bad_action, 0

    def reset(self) -> None:
        self.t = 0

    def act(self, obs):
        self.t += 1
        if self.t > self.at:
            if self.bad_action:
                return np.full(9, 2, np.int8)
            raise RuntimeError("policy broke")
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)


# ------------------------------------------------------------------------------------- drawing
def test_layout_fits_a_video_frame():
    assert D.PANEL_W % 2 == 0 and D.PANEL_H % 2 == 0            # H.264 with 4:2:0 colour needs even sides
    assert D.MAP_PX == spec.CROP_SIZE * 4 and D.RGB_H + D.BAR_H == D.MAP_PX
    assert D.FONT.getbbox("M") == (0, 0, D.CHAR_W, 11)          # the built-in bitmap font, not a font file


def test_map_image_colours_and_orientation():
    crop = np.zeros((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.float32)
    cells = {"free": (C, C), "wall": (C + 8, C), "mixed": (C + 8, C + 1), "door": (C, C + 8), "locked": (C, C - 8),
             "key": (C - 8, C), "item": (C - 8, C + 1), "enemy": (C + 4, C + 4), "exit": (C - 4, C - 4)}
    for name in ("free", "mixed", "key", "item", "enemy", "exit", "door", "locked"):
        crop[spec.CH_FREE][cells[name]] = 1.0
    crop[spec.CH_WALL][cells["wall"]] = crop[spec.CH_WALL][cells["mixed"]] = 1.0
    for name, ch in (("door", spec.CH_DOOR), ("locked", spec.CH_LOCKED), ("key", spec.CH_KEY), ("item", spec.CH_ITEM),
                     ("enemy", spec.CH_ENEMY), ("exit", spec.CH_EXIT)):
        crop[ch][cells[name]] = 1.0
    colours = {"free": D.C_FREE, "wall": D.C_WALL, "mixed": D.C_MIXED, "door": D.C_DOOR, "locked": D.C_LOCKED, "key": D.C_KEY,
               "item": D.C_ITEM, "enemy": D.C_ENEMY, "exit": D.C_EXIT}
    for zoom in D.ZOOMS:
        view = D.View.centred(zoom)
        img = D.map_image(crop, view)
        assert img.shape == (D.MAP_PX, D.MAP_PX, 3) and img.dtype == np.uint8
        n, px, lo = view.cells, view.px, view.low((0, 0))[0]
        assert n * px == D.MAP_PX and view.margin == 0 and lo + n // 2 == C and view.low((7, -3)) == (lo, lo)
        assert tuple(img[1, 1]) == D.C_UNKNOWN
        for name, (i, j) in cells.items():
            if not (lo <= i < lo + n and lo <= j < lo + n):
                continue
            # x grows to the right, y grows upwards; a corner pixel of the cell holds the fill colour
            x, y = (i - lo) * px + 1, (lo + n - 1 - j) * px + 1
            assert tuple(int(v) for v in img[y, x]) == colours[name], (zoom, name)
    # the same picture from the uint8 form the training workers use
    assert np.array_equal(D.map_image((crop * 255).astype(np.uint8)), D.map_image(crop))


def test_fixed_view_does_not_move_with_the_agent():
    """The fitted picture shows the same start-frame cells whatever crop it is cut from."""
    memory = MemoryMap()
    memory.occ[100:140, 110:150] = -1                           # a block of free space ...
    memory.occ[98:100, 110:150] = 3                             # ... with a wall along one side
    memory.flags[120, 130] = 1                                  # and a door
    view = D.View(26, (-18, -12))
    assert view.px == 9 and view.margin == 11 and "26 x 26" in view.describe()
    pictures = []
    for pose in ((0.0, 0.0, 0.0), (300.0, -200.0, 1.0), (-512.0, 640.0, 2.0)):
        origin = crop_origin(pose)
        assert view.low(origin) == (-18 - origin[0], -12 - origin[1])
        pictures.append(D.map_image(memory.crop(pose), view, origin))
        frame = D._MapFrame(pose, view)
        assert frame.point(64.0, 128.0) == D._MapFrame((0.0, 0.0, 0.0), view).point(64.0, 128.0)
        assert frame.cell_box(1 - origin[0], 2 - origin[1]) == D._MapFrame((0.0, 0.0, 0.0), view).cell_box(1 + C, 2 + C)
    assert np.array_equal(pictures[0], pictures[1]) and np.array_equal(pictures[0], pictures[2])
    seen = {tuple(int(v) for v in c) for c in pictures[0].reshape(-1, 3)}
    assert {D.C_FREE, D.C_WALL, D.C_DOOR, D.C_UNKNOWN, D.C_BG} <= seen
    # start-frame cell (0, 0) sits 18 cells right of and 12 cells above the picture's bottom-left corner
    x0, y0, x1, y1 = D._MapFrame((0.0, 0.0, 0.0), view).cell_box(C, C)
    assert (x0, y1) == (11 + 18 * 9, 11 + (26 - 12) * 9 - 1) and (x1 - x0, y1 - y0) == (8, 8)
    # a crop that does not reach the window leaves it unknown instead of failing
    far = D.map_image(memory.crop((5000.0, 5000.0, 0.0)), view, crop_origin((5000.0, 5000.0, 0.0)))
    lines = {tuple(int(k * v) for v in D.C_UNKNOWN) for k in (0.8, 0.8 * 0.8)}        # cell borders and their crossings
    assert {tuple(int(v) for v in c) for c in far.reshape(-1, 3)} == {D.C_UNKNOWN, D.C_BG} | lines


def test_map_image_trail_and_use_marks():
    crop = np.zeros((spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE), np.float32)
    crop[spec.CH_FREE, C - 2:C + 3, C] = 1.0
    crop[spec.CH_VISITED, C - 2:C + 1, C] = 1.0
    crop[spec.CH_RECENCY, C, C] = 1.0                           # just visited
    crop[spec.CH_USED, C + 2, C] = 1.0
    img = D.map_image(crop, D.View.centred(2))
    view, px, lo = 32, 8, C - 16

    def pixel(i, j, dx, dy):
        return tuple(int(v) for v in img[(lo + view - 1 - j) * px + dy, (i - lo) * px + dx])

    assert pixel(C, C, px // 2, px // 2) == D.C_TRAIL           # dot in the middle of the cell, bright when recent
    assert pixel(C - 2, C, px // 2, px // 2) == D.C_VISITED     # ... and dim for an old visit
    assert pixel(C - 2, C, 1, 1) == D.C_FREE                    # the cell itself keeps its colour
    assert pixel(C + 1, C, px // 2, px // 2) == D.C_FREE        # not visited: no dot
    assert pixel(C + 2, C, 0, 3) == D.C_USED                    # frame of the cell USE was pressed at
    assert pixel(C + 2, C, px // 2, px // 2) == D.C_FREE


def test_markers_are_drawn_where_the_poses_are():
    pose = (100.0, -40.0, 0.0)                                  # 100 units ahead of the start, 40 to its right
    f = frame_state(frame_pose=pose, est=pose, true=(100.0 + 3 * 64.0, -40.0, 0.0), view=D.View.centred(2))
    frame = D._MapFrame(pose, D.View.centred(2))
    assert frame.origin == crop_origin(pose)
    box = frame.cell_box(C + 5, C + 2)
    img = D.compose(frame_state(**{**f, "target_box": box}))
    assert img.size == (D.PANEL_W, D.PANEL_H)
    ex, ey = frame.point(pose[0], pose[1])
    tx, ty = frame.point(*f["true"][:2])
    assert abs(tx - ex - 3 * frame.px) < 1e-6 and ty == ey      # 3 cells further along x: 3 cells to the right
    assert img.getpixel((D.RGB_W + int(tx), int(ty))) == D.C_TRUE
    r = min(6, max(3, frame.px // 2))
    assert img.getpixel((D.RGB_W + int(round(ex)) - r, int(round(ey)))) == D.C_AGENT   # the ring around the estimate
    assert img.getpixel((D.RGB_W + box[0] - 1, box[1] - 1)) == D.C_TARGET
    # the estimate sits in the middle cell of the picture, wherever it is inside that cell
    view, px, lo = 32, 8, C - 16
    assert (C - lo) * px <= ex < (C - lo + 1) * px and (lo + view - 1 - C) * px <= ey < (lo + view - C) * px
    # facing +y (left at the start) draws the heading line upwards
    up = D.compose(frame_state(frame_pose=(0.0, 0.0, 0.0), est=(0.0, 0.0, math.pi / 2), true=None))
    x0, y0 = D._MapFrame((0.0, 0.0, 0.0), D.View.centred(1)).point(0.0, 0.0)
    assert up.getpixel((D.RGB_W + int(x0), int(y0) - 6)) == D.C_AGENT
    # a marker far outside the map picture leaves the frame next to it untouched
    far = D.compose(frame_state(true=(-100000.0, 0.0, 0.0), target_box=(-5000, 100, -4990, 110)))
    assert far.getpixel((D.RGB_W - 1, 126)) == (0, 0, 0)


def test_buttons_show_presses_and_masks():
    action = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0], np.int8)
    img = D.compose(frame_state(action=action, masks=(True, False)))
    slot = D.RGB_W / 9

    def box_colours(k):
        x = int(round(k * slot)) + 1
        return img.getpixel((x, D.RGB_H + 2)), img.getpixel((x + 2, D.RGB_H + 4))     # outline, fill

    assert box_colours(0)[1] == (236, 236, 236)                 # forward: pressed
    assert box_colours(1)[1] == (34, 34, 40)                    # back: not pressed
    assert box_colours(7) == (D.C_GOOD, (236, 236, 236))        # USE pressed while the mask allows it
    assert box_colours(6)[1] == D.C_WARN                        # ATTACK pressed with no enemy in view
    assert D.compose(frame_state(action=None, masks=None)).size == (D.PANEL_W, D.PANEL_H)


def test_text_lines_fit_the_panel_at_extreme_values():
    totals = {"sectors_visited": 27, "sectors_total": 27, "keys_picked": 3, "keys_total": 3, "doors_opened": 12,
              "doors_total": 12, "locked_opened": 3, "locked_total": 3, "kills": 12, "monsters_total": 12, "health": 200.0,
              "damage_taken": 1234.0, "cells_visited": 456, "cells_total": 456, "has_exit": 1, "exit_visible_tic": 4200,
              "exit_dist_final": 1234.0, "stuck_rate": 0.99, "stuck_at_door_rate": 0.99, "door_reversals": 12, "score": -12.3}
    f = frame_state(step=1050, steps_cap=1050, tic=4200, ticks=4200, subgoal="open_locked", target_text="~(63,63)=self",
                    exit_believed=True, est=(-1234.5, -1234.5, -3.1), true=(-1234.5, -1234.5, -3.1), drift=1234.5,
                    heading_err=-179.9, pose_source="policy", totals=totals)
    lines = ["".join(text for text, _ in line) for line in D._info_lines(f)]
    assert len(lines) == D.INFO_LINES and all(len(line) <= D.MAX_CHARS for line in lines), [len(line) for line in lines]
    assert "open_locked" in lines[0] and "27/27" in lines[2] and "reversals 12" in lines[3]
    # missing totals, NaN and text the bitmap font cannot draw must not raise
    odd = frame_state(totals={"health": float("nan"), "cells_total": float("nan"), "has_exit": 0},
                      note="café ☃\n\ttab " + "x" * 400, error="RuntimeError: ☃", subgoal="☃" * 30)
    assert D.compose(odd).size == (D.PANEL_W, D.PANEL_H)
    assert D.compose(frame_state(est=None, true=None, banner=("TIME OUT", D.C_WARN))).size == (D.PANEL_W, D.PANEL_H)


def test_even_sampler_is_bounded_and_even():
    for n in (1, 5, 16, 17, 31, 32, 33, 100, 525, 1050, 4000):
        sampler = D.EvenSampler(16)
        for k in range(n):
            sampler.add(k, k)
            assert len(sampler.items) < 32
        picked = [k for k, _ in sampler.pick()]
        assert len(picked) == min(n, 16) and picked[0] == 0 and picked == sorted(set(picked))
        gaps = np.diff(picked)
        if n > 16:
            assert picked[-1] >= n - 2 * sampler.stride and gaps.max() <= 2 * gaps.min()   # spread over the whole episode
        else:
            assert picked == list(range(n))


def test_contact_sheet_layout():
    panels = [Image.new("RGB", (D.PANEL_W, D.PANEL_H), (k * 10, 0, 0)) for k in range(5)]
    sheet = D.contact_sheet(panels, ["first line", "second line"])
    for k in range(5):
        row, col = divmod(k, D.SHEET_COLS)
        x = D.SHEET_GAP + col * (D.PANEL_W + D.SHEET_GAP)
        y = D.SHEET_HEADER_H + D.SHEET_GAP + row * (D.PANEL_H + D.SHEET_GAP)
        assert sheet.getpixel((x, y)) == (k * 10, 0, 0) and sheet.getpixel((x + D.PANEL_W - 1, y + D.PANEL_H - 1)) == (k * 10, 0, 0)


# ---------------------------------------------------------------------------- reading a policy
def test_read_debug_accepts_many_forms():
    torch = pytest.importorskip("torch")
    d = D.read_debug({"subgoal": 3, "target": 5 * 64 + 7, "use_age": 2, "flag": True, "big": np.zeros((64, 64)), "note": "n"})
    assert d["subgoal"] == spec.SUBGOALS[3] and d["target"] == (5, 7)
    assert d["note"] == "n  use_age=2  flag=1"                 # small values are printed, arrays are not
    d = D.read_debug({"subgoal": torch.tensor([5]), "target": torch.tensor(130), "target_origin": (-30, -31),
                      "target_xy": np.array([64.0, -128.0]), "pose": torch.tensor([1.0, 2.0, 0.5])})
    assert d["subgoal"] == "go_exit" and d["target"] == (2, 2) and d["target_origin"] == (-30.0, -31.0)
    assert d["target_xy"] == (64.0, -128.0) and d["pose"] == (1.0, 2.0, 0.5)
    assert D.read_debug({"subgoal": "sweep", "target": (40, 41)})["target"] == (40, 41)
    assert D.read_debug({"subgoal": 99})["subgoal"] == "#99"
    crop = np.zeros((spec.MAP_CHANNELS, 64, 64), np.uint8)
    assert D.read_debug({"crop": crop})["crop"] is not None and D.read_debug({"crop": np.zeros((3, 3))})["crop"] is None
    for junk in (None, {}, "text", 7, {"target": "x", "pose": "y", "target_xy": 3, "subgoal": [1, 2]}, {"pose": (1.0, float("nan"), 0.0)}):
        d = D.read_debug(junk)
        assert d["target"] is None and d["pose"] is None and d["crop"] is None

    class Label:                                                # an object with attributes instead of a dict
        def __init__(self):
            self.subgoal, self.target, self.field_valid = 1, C * 64 + C, False

    d = D.read_debug(Label())
    assert d["subgoal"] == "sweep" and d["target"] == (C, C) and "field_valid=0" in d["note"]
    # the teacher's form: its own start-frame cell, the target as a start-frame cell, the route and the waypoint
    d = D.read_debug({"subgoal": "open_door", "target": (12, 7), "cell": (12, 6), "route": [(12, 6), (12, 7)], "waypoint": (768.0, 448.0),
                      "mode": "door", "step": 9, "zeta": 1, "heads": [0, 0, 0, 0, 1, 0], "failed_locks": {}, "subgoal_steps": {"explore": 4}})
    assert d["target"] is None and d["target_cell"] == (12.0, 7.0) and d["route"] == [(12, 6), (12, 7)] and d["waypoint"] == (768.0, 448.0)
    assert d["note"] == "mode=door  cell=(12,6)  heads=(0,0,0,0,1,0)  zeta=1"       # the most telling values first; no step, no dicts
    assert D.read_debug({"target_cell": (3, -2), "target": 5})["target_cell"] == (3.0, -2.0)
    alone = D.read_debug({"target_cell": (3, -2)})
    assert alone["target_cell"] == (3.0, -2.0) and alone["target"] is None
    assert D.read_debug({"route": "nonsense", "waypoint": None})["route"] is None


def test_find_memory_and_hooks():
    for how in ("attribute", "method", "unnamed"):
        policy = HookedPolicy(memory_as=how)
        for holder in (policy, D.Probe(policy), Wrapper(policy), D.Probe(Wrapper(policy)), R.ScriptPolicy("N")):
            found = D.find_memory(holder)
            assert (found is None) if isinstance(holder, R.ScriptPolicy) else (found is policy._store), how
    assert D._hook(D.Probe(Wrapper(HookedPolicy())), "debug_state") is not None
    assert D._hook(R.RandomPolicy(), "pose_estimate") is None
    probe = D.Probe(R.RandomPolicy())
    assert hasattr(probe, "seed") and not hasattr(probe, "attach") and not hasattr(probe, "pose_estimate")


# ------------------------------------------------------------------------------------ episodes
def test_random_policy_on_a_reference_map(tmp_path):
    """The headline use: a policy with no hooks at all, on a reference map, MP4 (or GIF) and sheets."""
    summary = D.render_episode("training.runner:RandomPolicy", TWO_ROOMS, tmp_path / "out", seed=3, ticks=200)
    row = summary["row"]
    assert row["steps"] == 50 and summary["warnings"] == [] and summary["error"] == ""
    check_outputs(summary)
    assert summary["pose_source"] == "shadow" and summary["map_source"].startswith("shadow memory")
    # two_rooms is 14 x 9 cells with the start in column 2, 6 rows up: the picture is 16 cells wide and fixed on the map
    assert summary["view_cells"] == 16 and summary["view_corner"] == [-3, -9] and "fixed on the map" in summary["view"]
    # the viewer changes nothing about the episode: the row is the runner's own
    plain = R.run_episodes("training.runner:RandomPolicy", [{"wad": TWO_ROOMS, "episode_ticks": 200}], workers=0, seeds=[3])[0]
    assert {k: v for k, v in row.items() if k not in R.TIMING_FIELDS} == pytest.approx(
        {k: v for k, v in plain.items() if k not in R.TIMING_FIELDS}, nan_ok=True)
    rows = trace_rows(summary)
    assert [int(r["step"]) for r in rows] == list(range(50)) and rows[0]["pose_source"] == "shadow"
    assert all(float(r["drift"]) < 2.0 and abs(float(r["heading_err_deg"])) < 0.01 for r in rows)   # the shadow odometry holds
    assert (float(rows[0]["world_x"]), float(rows[0]["world_y"])) == load_meta(TWO_ROOMS).player_start[:2]
    # the map picture of the last panel shows what the rays have seen: free space and walls
    panel = np.asarray(sheet_panel(summary["sheets"][1], D.SHEET_PANELS - 1))[:D.MAP_PX, D.RGB_W:]
    seen = {tuple(c) for c in panel.reshape(-1, 3)}
    assert {D.C_WALL, D.C_AGENT, D.C_TRUE} <= seen and len(seen & {D.C_FREE, D.C_VISITED, D.C_TRAIL}) >= 2


def test_generated_map_from_a_grid_file(tmp_path, capsys):
    """--grid compiles an ASCII grid with the map generator first; here a generated tier 3 layout."""
    layout = layouts.generate(3, 4242)
    grid_file = tmp_path / "layout.txt"
    grid_file.write_text(layout["grid"].replace("\n", "\r\n") + "\r\n\r\n")       # an editor's line ends and trailing blank lines
    out = tmp_path / "out"
    code = D.main(["--policy", "training.runner:RandomPolicy", "--kw", "p=0.4", "--grid", str(grid_file), "--out", str(out),
                   "--seed", "5", "--ticks", "240", "--video", "none"])
    printed = capsys.readouterr().out
    assert code == 0 and "sheet_final.png" in printed and "WARNING" not in printed
    assert udmf.read_textmap(out / "map.wad") == udmf.build_textmap(layout["grid"])
    summary = json.loads((out / "episode.json").read_text())
    assert summary["row"]["steps"] <= 60 and summary["policy"] == 'training.runner:RandomPolicy{"p": 0.4}'
    assert summary["map"]["id"] == "layout" and summary["video"] is None and summary["warnings"] == []
    assert len(summary["sheets"]) == 2 and all(Path(p).is_file() for p in summary["sheets"])
    rows, cols = len(layout["grid"].split("\n")), len(layout["grid"].split("\n")[0])
    assert summary["view_cells"] == max(max(rows, cols) + 2, D.FIT_MIN_CELLS) and summary["view_corner"] is not None


def test_grid_sources(tmp_path):
    suite = ROOT / "training" / "suites" / "dev_v1.jsonl"
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    pick = next(r for r in rows if r["tier"] == 3)
    got = D.read_grid(f"{suite}:{pick['name']}")
    assert got["grid"] == pick["grid"] and got["episode_ticks"] == pick["episode_ticks"] and got["tier"] == 3
    assert D.read_grid(f"training/suites/dev_v1.jsonl:{rows.index(pick)}")["grid"] == pick["grid"]    # by line index, repo-relative
    assert D.read_grid(str(suite))["grid"] == rows[0]["grid"]
    bare = D.read_grid(f"{suite}:{pick['name']}", no_monsters=True)
    assert not set(bare["grid"]) & udmf.MONSTER_CHARS and len(bare["grid"]) == len(pick["grid"])
    assert "ambush" not in bare["options"] and "thing_angles" not in bare["options"]
    for bad in (f"{suite}:no_such_map", str(tmp_path / "missing.txt")):
        with pytest.raises(SystemExit):
            D.read_grid(bad)
    wrong = tmp_path / "wrong.txt"
    wrong.write_text("#####\n#P?E#\n#####\n")
    with pytest.raises(SystemExit, match="unknown symbol"):
        D.read_grid(str(wrong))


def test_command_line_needs_exactly_one_map_source(tmp_path):
    for argv in (["--out", str(tmp_path)], ["--out", str(tmp_path), "--map", TWO_ROOMS, "--grid", "x.txt"],
                 ["--out", str(tmp_path), "--map", str(tmp_path / "none.wad")], ["--out", str(tmp_path), "--map", TWO_ROOMS, "--no-monsters"]):
        with pytest.raises(SystemExit) as e:
            D.main(argv)
        assert e.value.code == 2


def test_policy_hooks_are_shown(tmp_path):
    """pose_estimate(), debug_state(), exit_believed() and the policy's own memory map are what the panels show."""
    policy = HookedPolicy()
    summary = D.render_episode(policy, fixture_wad(KEY), tmp_path / "out", ticks=160, video="none")
    row = summary["row"]
    assert summary["warnings"] == [] and row["steps"] == 40
    assert summary["pose_source"] == "policy" and summary["map_source"] == "policy memory"
    assert row["odom_err_max"] < 1.0 and row["exit_believed_tic"] == 40          # the runner got the same hooks
    rows = trace_rows(summary)
    assert all(r["subgoal"] == "open_door" and (r["target_i"], r["target_j"]) == (str(C + 3), str(C - 1)) for r in rows)
    assert all(r["target_ref"] == "origin" for r in rows)
    assert all(r["note"] == "hello  heads=(0,1,2,3,4,5)  use_age=3" and r["acted"] == "1" for r in rows)
    assert max(float(r["drift"]) for r in rows) == pytest.approx(row["odom_err_max"], abs=0.006)
    assert float(rows[-1]["est_x"]) > 100.0 and rows[-1]["pose_source"] == "policy"
    # each panel shows the frame the policy was given at that step (the sheet holds the last 16 steps)
    for k in range(D.SHEET_PANELS):
        shown = np.asarray(sheet_panel(summary["sheets"][1], k))[:D.RGB_H:D.SCALE, :D.RGB_W:D.SCALE]
        assert np.array_equal(shown, policy.seen[40 - D.SHEET_PANELS + k]), k
    # ... and the target box sits 3 cells ahead and 1 to the right of the estimate's cell
    last = sheet_panel(summary["sheets"][1], D.SHEET_PANELS - 1)
    frame = D._MapFrame(policy.pose_estimate(), D.View(summary["view_cells"], summary["view_corner"]))
    box = frame.cell_box(C + 3, C - 1)
    assert last.getpixel((D.RGB_W + box[0] - 1, box[1] - 1)) == D.C_TARGET

    # a wrong pose estimate shows as drift; the memory is found one level down and under any name
    off = HookedPolicy(memory_as="unnamed", pose_offset=50.0)
    summary = D.render_episode(Wrapper(off), fixture_wad(KEY), tmp_path / "off", ticks=80, video="none")
    assert summary["map_source"] == "policy memory" and summary["warnings"] == []
    assert all(float(r["drift"]) == pytest.approx(50.0, abs=1.0) for r in trace_rows(summary))


def test_target_forms_follow_the_agent(tmp_path):
    """A target given as a start-frame point, or as a cell of an earlier crop, stays on its place while the agent moves."""
    def own_column(est_x: float) -> int:                         # crop column of start-frame cell 5 for an agent at est_x
        return 5 - (math.floor((est_x + 32.0) / 64.0) - C)

    forms = {"point": lambda policy: {"subgoal": "go_exit", "target_xy": (5 * 64.0 + 10.0, -20.0)},
             "held": lambda policy: {"subgoal": 5, "target": (C + 5) * spec.CROP_SIZE + C, "target_origin": (-C, -C)}}
    for name, debug in forms.items():
        summary = D.render_episode(HookedPolicy("F*30", debug=debug), fixture_wad(KEY), tmp_path / name, ticks=120, video="none")
        rows = trace_rows(summary)
        assert summary["warnings"] == [] and all(r["subgoal"] == "go_exit" for r in rows)
        assert [int(r["target_i"]) for r in rows] == [own_column(float(r["est_x"])) for r in rows], name
        assert all(r["target_j"] == str(C) for r in rows)
        assert int(rows[0]["target_i"]) == C + 5 and int(rows[-1]["target_i"]) < C + 5      # the agent has walked towards it
    # hooks that raise are reported once and replaced by the shadow's view
    def broken(policy):
        raise KeyError("no debug today")

    policy = HookedPolicy("F*10", debug=broken)
    policy.pose_estimate = lambda: (_ for _ in ()).throw(RuntimeError("no pose"))
    summary = D.render_episode(policy, fixture_wad(KEY), tmp_path / "broken", ticks=40, video="none")
    assert len(summary["warnings"]) == 2 and summary["row"]["steps"] == 10
    assert any("debug_state()" in w and "no debug today" in w for w in summary["warnings"])
    assert any("pose_estimate()" in w and "no pose" in w for w in summary["warnings"])
    assert summary["pose_source"] == "policy odometry" and summary["map_source"] == "policy memory"
    row = summary["row"]                                        # the runner asks for the pose too and notes the failure; the policy plays on
    assert row["agent_error"] == "" and "pose_estimate()" in row["hook_error"] and "no pose" in row["hook_error"]
    assert all(r["acted"] == "1" for r in trace_rows(summary)) and float(trace_rows(summary)[-1]["est_x"]) > 100.0


def target_boxes(sheet, steps: int) -> list:
    """Bounding box (x0, y0, x1, y1) of the target-coloured pixels in the map picture of each panel of the final sheet."""
    out = []
    for k in range(min(steps, D.SHEET_PANELS)):
        picture = np.asarray(sheet_panel(sheet, k))[:D.MAP_PX, D.RGB_W:D.RGB_W + D.MAP_PX]
        ys, xs = np.nonzero((picture == D.C_TARGET).all(-1))
        out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None)
    return out


HELD_CASES = {
    # the submission agent's form before it reported an origin: a fixed cell, chosen every MANAGER_PERIOD steps, index held
    "index": (dict(report="index"), "held"),
    "index as a pair": (dict(report="index", pair=True), "held"),
    # every choice gives the same index (a cell 5 ahead of the agent), so only the manager period tells the choices apart
    "same index at every choice": (dict(report="index", ahead=5), "held"),
    # a choice made between two manager periods shows as a changed index
    "choice out of step": (dict(report="index", extra=(11, 21)), "held"),
    "origin": (dict(report="origin"), "origin"),
    "origin, other period": (dict(report="origin", period=13, ahead=5), "origin"),
    "cell": (dict(report="cell", period=13, extra=(3,)), "cell"),
}


@pytest.mark.parametrize("case", list(HELD_CASES))
def test_held_target_stays_on_its_cell(tmp_path, case):
    """A manager's target is an index into the crop of the step that chose it and is held while the agent walks on.

    The box must stay on the cell that was meant, with the crop origin reported or without it; without
    it the viewer says once that the place is inferred and marks the target.
    """
    kwargs, ref = HELD_CASES[case]
    policy = Manager(**kwargs)
    wad = udmf.write_wad(CORRIDOR, tmp_path / "corridor.wad")
    summary = D.render_episode(Wrapper(policy), wad, tmp_path / "out", ticks=128, video="none")
    rows = trace_rows(summary)
    assert len(rows) == 32 and float(rows[-1]["est_x"]) > 850.0          # 14 cells walked: four manager periods
    assert len({crop_origin(p) for p in policy.poses[8:16]}) >= 4        # the crop moves on while the index is held
    origins = [crop_origin(pose) for pose in policy.poses]
    shown = [(int(r["target_i"]) + o[0], int(r["target_j"]) + o[1]) for r, o in zip(rows, origins)]
    assert shown == policy.cells, case
    assert all(r["target_ref"] == ref for r in rows)
    if ref == "held":
        assert len(summary["warnings"]) == 1 and "target_origin" in summary["warnings"][0] and "guess" in summary["warnings"][0]
    else:
        assert summary["warnings"] == []
    # the picture: in each of the last 16 panels the box is drawn around that cell and nowhere else
    view = D.View(summary["view_cells"], summary["view_corner"])
    boxes = target_boxes(summary["sheets"][1], len(rows))
    for k, found in enumerate(boxes):
        t = len(rows) - len(boxes) + k
        frame = D._MapFrame(policy.poses[t], view)
        x0, y0, x1, y1 = frame.cell_box(policy.cells[t][0] - frame.origin[0], policy.cells[t][1] - frame.origin[1])
        assert found == (x0 - 1, y0 - 1, x1 + 1, y1 + 1), (case, t)


def test_inferred_target_is_marked_on_the_panel():
    lines = D._info_lines(frame_state(target_text="~(39,35)"))
    assert "target ~(39,35)" in "".join(text for text, _ in lines[0])
    rec = D.Recorder("unused", D.Probe(R.ScriptPolicy("F")))
    own = [(-C, -C), (-C + 1, -C), (-C + 2, -C), (-C + 3, -C)]

    def origin_at(t, target, k):                                # what _panel_state does with a target that has no origin
        rec._held = rec._held_target(t, target, own[k])
        return rec._held[2]

    assert origin_at(8, (40, 32), 0) == own[0]                  # first report: the current crop
    assert origin_at(9, (40, 32), 1) == own[0]                  # held: still the crop of step 8
    assert origin_at(10, (41, 32), 2) == own[2]                 # the index changed: a new choice
    assert origin_at(11, (41, 32), 3) == own[2]
    assert origin_at(13, (41, 32), 3) == own[3]                 # a step without this report in between: start again
    rec._held = None
    assert origin_at(15, (41, 32), 1) == own[1] and origin_at(16, (41, 32), 2) == own[2]   # a manager period begins at 16
    assert len(rec.warnings) == 1                               # said once


def test_teacher_style_report_is_drawn(tmp_path):
    """A start-frame target cell with a route and a waypoint: the box, the line and the cross land on those cells."""
    def report(policy):
        x, y, _ = policy.odometry.pose
        own = (math.floor((x + 32.0) / 64.0), math.floor((y + 32.0) / 64.0))
        return {"subgoal": "explore", "cell": own, "target": (6, 0), "route": [own, (own[0] + 1, 0), (6, 0)],
                "waypoint": (6 * 64.0, 0.0), "mode": "run"}

    policy = HookedPolicy("F*12", debug=report)
    summary = D.render_episode(policy, fixture_wad(KEY), tmp_path / "out", ticks=48, video="none")
    rows = trace_rows(summary)
    assert summary["warnings"] == [] and all(r["note"].startswith("mode=run  cell=(") for r in rows)
    own_last = math.floor((float(rows[-1]["est_x"]) + 32.0) / 64.0)
    assert (int(rows[-1]["target_i"]), int(rows[-1]["target_j"])) == (C + 6 - own_last, C)
    last = sheet_panel(summary["sheets"][1], min(len(rows), D.SHEET_PANELS) - 1)
    frame = D._MapFrame(policy.pose_estimate(), D.View(summary["view_cells"], summary["view_corner"]))
    origin = crop_origin(policy.pose_estimate())
    box = frame.cell_box(6 - origin[0], 0 - origin[1])
    assert last.getpixel((D.RGB_W + box[0] - 1, box[1] - 1)) == D.C_TARGET                       # corner of the target box
    wx, wy = frame.point(6 * 64.0, 0.0)
    assert last.getpixel((D.RGB_W + int(wx) + 3, int(wy))) == D.C_TARGET                         # arm of the waypoint cross
    mid = frame.point((own_last + 1.5) * 64.0, 0.0)                                              # on the route, between two cells
    assert last.getpixel((D.RGB_W + int(mid[0]), int(mid[1]))) == D.C_ROUTE


def test_scripted_run_to_the_exit(tmp_path):
    """Key, locked door, exit: the running totals follow the episode and the exit shows in the shadow map."""
    summary = D.render_episode(("training.runner:ScriptPolicy", {"script": "F*34 U N*8 F*60"}), fixture_wad(KEY),
                               tmp_path / "out", video="none")
    row = summary["row"]
    assert row["exit"] == 1 and row["keys_picked"] == 1 and row["locked_opened"] == 1 and D.outcome_banner(row)[0] == "EXIT REACHED"
    rows = trace_rows(summary)
    assert len(rows) == row["steps"] and [r["keys_picked"] for r in rows][0] == "0" and rows[-1]["keys_picked"] == "1"
    assert rows[0]["locked_opened"] == "0" and rows[-1]["locked_opened"] == "1"
    assert rows[0]["action"] == "F........" and rows[34]["action"] == ".......U." and rows[35]["action"] == "........."
    assert rows[34]["use_ok"] == "1" and rows[0]["use_ok"] == "0"               # facing the locked door within reach
    assert rows[0]["exit_label_visible"] == "0" and rows[-1]["exit_label_visible"] == "1"
    assert float(rows[-1]["exit_dist"]) < float(rows[0]["exit_dist"])
    panel = np.asarray(sheet_panel(summary["sheets"][1], D.SHEET_PANELS - 1))[:D.MAP_PX, D.RGB_W:]
    seen = {tuple(c) for c in panel.reshape(-1, 3)}
    assert D.C_EXIT in seen and D.C_LOCKED not in seen          # the pad is in the map; the opened door no longer reads as one
    # earlier panels: the key lies ahead at the start, the locked door comes into view once the key is out of the way
    early = [{tuple(c) for c in np.asarray(sheet_panel(summary["sheets"][0], k))[:D.MAP_PX, D.RGB_W:].reshape(-1, 3)}
             for k in range(D.SHEET_PANELS)]
    assert D.C_KEY in early[0] and D.C_LOCKED not in early[0] and any(D.C_LOCKED in colours for colours in early)
    assert D.C_KEY not in early[-1]                             # picked up: the key is gone from the map
    # without labels the shadow map has no exit
    summary = D.render_episode(("training.runner:ScriptPolicy", {"script": "F*34 U N*8 F*60"}), fixture_wad(KEY),
                               tmp_path / "plain", video="none", exit_source="none")
    panel = np.asarray(sheet_panel(summary["sheets"][1], D.SHEET_PANELS - 1))[:D.MAP_PX, D.RGB_W:]
    assert D.C_EXIT not in {tuple(c) for c in panel.reshape(-1, 3)} and summary["map_source"] == "shadow memory"


def test_death_and_enemies_are_shown(tmp_path):
    """Standing still in front of a chaingunner: the enemy is on the map from the first panel, the episode ends in death."""
    summary = D.render_episode(("training.runner:ScriptPolicy", {"script": "N"}), fixture_wad(GUNNER), tmp_path / "out", video="none")
    row = summary["row"]
    assert row["dead"] == 1 and row["health"] == 0 and D.outcome_banner(row) == ("DIED", D.C_BAD) and summary["warnings"] == []
    rows = trace_rows(summary)
    assert rows[0]["attack_ok"] == "1" and rows[0]["health"] == "100" and float(rows[-1]["damage_taken"]) > 0
    first = np.asarray(sheet_panel(summary["sheets"][0], 0))[:D.MAP_PX, D.RGB_W:]
    assert D.C_ENEMY in {tuple(int(v) for v in c) for c in first.reshape(-1, 3)}
    attack = sheet_panel(summary["sheets"][0], 0).getpixel((int(round(6 * D.RGB_W / 9)) + 1, D.RGB_H + 2))
    assert attack == D.C_GOOD                                   # the ATK button is framed: an enemy is in view


@pytest.mark.parametrize("bad_action", [False, True])
def test_a_failing_policy_is_still_recorded(tmp_path, bad_action):
    summary = D.render_episode(Failing(at=5, bad_action=bad_action), fixture_wad(EXIT), tmp_path / "out", ticks=80, video="none")
    row = summary["row"]
    assert row["steps"] == 20 and row["agent_error"].startswith("step 5:") and summary["warnings"] == []
    check_outputs(summary, video=False)
    rows = trace_rows(summary)
    assert [r["acted"] for r in rows] == ["1"] * 5 + ["0"] * 15
    assert rows[4]["action"] == "F........" and rows[5]["action"] == "........."      # the player stands still afterwards
    assert ("policy broke" in rows[5]["note"]) != bad_action and ("invalid action" in rows[5]["note"]) == bad_action


def test_gif_when_ffmpeg_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(D.shutil, "which", lambda name: None)
    summary = D.render_episode("training.runner:RandomPolicy", fixture_wad(EXIT), tmp_path / "out", ticks=48, fps=10)
    assert Path(summary["video"]).name == "video.gif" and summary["row"]["steps"] == 12
    assert len(summary["warnings"]) == 1 and "no ffmpeg on PATH" in summary["warnings"][0] and "GIF" in summary["warnings"][0]
    with Image.open(summary["video"]) as im:
        assert im.size == (D.PANEL_W, D.PANEL_H) and im.n_frames == 13 == summary["video_frames"]
        durations = []
        for k in range(im.n_frames):
            im.seek(k)
            durations.append(im.info["duration"])
    assert durations[:12] == [100] * 12 and durations[12] == round(D.END_HOLD_S * 10) * 100      # the outcome is held
    assert len(summary["sheets"]) == 2


def test_gif_writer_bounds_its_memory(tmp_path):
    writer = D.GifWriter(tmp_path / "long.gif", fps=10.0, max_steps=1050, max_frames=100)
    assert writer.stride == 11
    frame = Image.new("RGB", (32, 20), (10, 20, 30))
    for _ in range(1050):
        writer.write(frame)
    writer.write(frame, repeat=15)                              # the outcome frame is always kept
    assert writer.frames == 97 and len(writer._images) == 97
    writer.close()
    with Image.open(tmp_path / "long.gif") as im:
        assert im.n_frames >= 1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_an_ffmpeg_that_fails_costs_only_the_video(tmp_path, monkeypatch):
    """ffmpeg dying in the middle of the episode must not lose the contact sheets or the result."""
    monkeypatch.setattr(D, "find_encoder", lambda exe: ("broken", ["-c:v", "no_such_encoder"]))
    summary = D.render_episode("training.runner:RandomPolicy", fixture_wad(EXIT), tmp_path / "out", ticks=400)
    assert summary["video"] is None and summary["row"]["steps"] == 100
    assert len(summary["warnings"]) == 1 and summary["warnings"][0].startswith("video")
    assert len(summary["sheets"]) == 2 and len(trace_rows(summary)) == 100


def test_encoder_trial_and_fallback_order(monkeypatch, tmp_path):
    D._ENCODER_CACHE.clear()
    assert D.find_encoder(str(tmp_path / "no_such_ffmpeg")) is None        # cannot even be started
    D._ENCODER_CACHE.clear()
    writer, note = D.open_video(tmp_path, "none", 15.0, 100)
    assert writer is None and note == ""
    writer, note = D.open_video(tmp_path, "gif", 15.0, 100)
    assert isinstance(writer, D.GifWriter) and note == ""
    if shutil.which("ffmpeg") and D.find_encoder(shutil.which("ffmpeg")):
        name, args = D.find_encoder(shutil.which("ffmpeg"))
        assert name in [n for n, _ in D.ENCODERS]
        writer, note = D.open_video(tmp_path, "auto", 15.0, 100)
        assert isinstance(writer, D.FfmpegWriter) and name in note
        writer.abort()


def test_view_fits_the_map(tmp_path):
    def chosen(wad, zoom="fit", name="MAP01"):
        meta = load_meta(wad, name)
        recorder = D.Recorder(tmp_path, D.Probe(R.RandomPolicy()), zoom=zoom)
        x, y, angle = meta.player_start
        return meta, recorder, recorder._choose_view(meta, (x, y, math.radians(angle)))

    meta, _, view = chosen(fixture_wad(KEY))                    # 12 x 3 cells, the start in column 1 of the middle row
    assert (view.cells, view.corner) == (14, (-2, -6)) and view.px == 18
    meta, _, view = chosen(fixture_wad("#####\n#P..#\n#####"))     # tiny: the picture never shows fewer than 12 cells a side
    assert view.cells == D.FIT_MIN_CELLS and view.corner == (-4, -5)
    for angle in (0, 90, 180, 270, 37):                         # wherever the start frame points, the whole map is inside
        wad = udmf.write_wad(layouts.generate(4, 77)["grid"], tmp_path / f"big{angle}.wad", start_angle=angle)
        meta, recorder, view = chosen(wad)
        assert view.corner is not None and recorder.warnings == []
        rows, cols = meta.grid.shape
        assert view.cells == max(rows, cols) + 2 if angle % 90 == 0 else view.cells > max(rows, cols)
        sx, sy, a0 = meta.player_start[0], meta.player_start[1], math.radians(angle)
        frame = D._MapFrame((0.0, 0.0, 0.0), view)
        for r in range(rows):
            for c in range(cols):
                wx, wy = meta.grid.centre(r, c)
                fwd = (wx - sx) * math.cos(a0) + (wy - sy) * math.sin(a0)
                left = -(wx - sx) * math.sin(a0) + (wy - sy) * math.cos(a0)
                px, py = frame.point(fwd, left)
                assert view.margin <= px <= D.MAP_PX - view.margin and view.margin <= py <= D.MAP_PX - view.margin, (angle, r, c)
    assert chosen(fixture_wad(KEY), zoom=4)[2].corner is None and chosen(fixture_wad(KEY), zoom=4)[2].cells == 16
    assert chosen(fixture_wad(KEY), zoom=1)[2].cells == spec.CROP_SIZE
    freedoom = Path(pytest.importorskip("vizdoom").__file__).parent / "freedoom2.wad"
    meta, recorder, view = chosen(freedoom)                     # not a grid layout: fitted to the outline of its sectors
    assert meta.grid is None and view.corner is not None and 40 <= view.cells <= spec.CROP_SIZE and recorder.warnings == []
    wide = SimpleNamespace(grid=None, player_start=(0.0, 0.0, 0.0), sector_polygons={0: np.array([[0.0, 0.0, 5000.0, 300.0]])})
    view = recorder._choose_view(wide, (0.0, 0.0, 0.0))         # wider than one crop: follow the agent instead
    assert view.corner is None and view.cells == spec.CROP_SIZE and recorder.warnings == []
    view = recorder._choose_view(SimpleNamespace(grid=None, sector_polygons={}), (0.0, 0.0, 0.0))
    assert view.corner is None and len(recorder.warnings) == 1  # no outline at all: said once, then the same fallback


def test_rerun_replaces_old_outputs(tmp_path):
    out = tmp_path / "out"
    D.render_episode("training.runner:RandomPolicy", fixture_wad(EXIT), out, ticks=40, video="gif", sheets=2)
    assert (out / "video.gif").is_file()
    summary = D.render_episode("training.runner:RandomPolicy", fixture_wad(EXIT), out, ticks=40, video="none")
    assert not (out / "video.gif").exists() and sorted(p.name for p in out.iterdir()) == [
        "episode.json", "sheet_even.png", "sheet_final.png", "trace.csv"]
    assert summary["row"]["steps"] == 10


def test_select_rows():
    def row(**over):
        base = {"wad": "a.wad", "tier": "2", "has_exit": "1", "exit": "1", "dead": "0", "agent_error": "", "error": "",
                "door_reversals": "0", "stuck_rate": "0.05", "map_id": "m"}
        return {**base, **over}

    rows = [row(), row(exit="0"), row(dead="1", exit="0", tier="1"), row(agent_error="step 3: boom"), row(has_exit="0", exit="0"),
            row(door_reversals="2", tier="3"), row(error="worker died", exit="0"), row(wad="", exit="0"), row(stuck_rate="")]
    assert [D.is_failure(r) for r in rows[:6]] == [False, True, True, True, False, False]
    assert D.select_rows(rows, failed=True) == rows[1:4]         # rows that could not be played have nothing to replay
    assert D.select_rows(rows, where=["door_reversals>0"]) == [rows[5]]
    assert D.select_rows(rows, where=["tier == 2", "exit==0"]) == [rows[1], rows[4]]
    assert D.select_rows(rows, where=["stuck_rate<0.1"], limit=2) == rows[:2] and len(D.select_rows(rows, where=["stuck_rate<0.1"])) == 6
    assert D.select_rows(rows, where=["map_id==m", "agent_error!="]) == [rows[3]]  # "is not empty": an empty value never matches
    assert D.select_rows(rows, where=["no_such_column==1"]) == []
    assert D.select_rows([dict(r, tier=int(r["tier"])) for r in rows], where=["tier>=3"])[0]["door_reversals"] == "2"
    with pytest.raises(SystemExit, match="KEY OP VALUE"):
        D.select_rows(rows, where=["tier is 3"])


def test_replaying_scored_episodes(tmp_path, capsys):
    """--episodes replays what training.evaluate scored: here the failures of a script that never presses USE."""
    maps = [{"wad": fixture_wad(EXIT), "id": "open_way", "tier": 2, "episode_ticks": 120},
            {"wad": fixture_wad(KEY), "id": "locked_way", "tier": 3, "episode_ticks": 120}]
    spec_text = 'training.runner:ScriptPolicy{"script": "F*100"}'
    logged = R.run_episodes(spec_text, maps, workers=0, seeds=[0, 1])
    assert [r["exit"] for r in logged] == [1, 1, 0, 0]
    E.write_episodes(tmp_path / "eval.csv", logged)
    out = tmp_path / "out"
    code = D.main(["--episodes", str(tmp_path / "eval.csv"), "--failed", "--video", "none", "--out", str(out)])     # the logged policy
    printed = capsys.readouterr().out
    assert code == 0 and "2 of 4 episodes" in printed and "differs" not in printed
    with open(out / "index.csv", newline="") as f:
        index = list(csv.DictReader(f))
    assert [(r["map_id"], r["seed"], r["outcome"], r["same_as_logged"]) for r in index] == [
        ("locked_way", "0", "TIME OUT", "1"), ("locked_way", "1", "TIME OUT", "1")]
    assert index[0]["keys_picked"] == "1" and index[0]["locked_opened"] == "0" and index[0]["tier"] == "3"
    for r in index:
        assert Path(r["sheet_final"]).is_file() and (Path(r["dir"]) / "trace.csv").is_file() and r["video"] == ""
    # a replay that ends differently from the logged row is flagged; a map that is gone does not stop the rest
    changed = [dict(logged[2], steps=7), dict(logged[3], wad=str(tmp_path / "gone.wad")), dict(logged[0])]
    summaries = D.render_rows(R.ScriptPolicy("F*100"), changed, tmp_path / "again", video="none")
    assert [s["same_as_logged"] for s in summaries] == [0, "", 1] and summaries[1]["row"] is None
    assert D.main(["--policy", spec_text, "--episodes", str(tmp_path / "eval.csv"), "--where", "tier==9", "--out", str(out)]) == 0
    for argv in (["--out", str(out), "--map", TWO_ROOMS, "--failed"], ["--out", str(out), "--episodes", str(tmp_path / "none.csv")]):
        with pytest.raises(SystemExit):
            D.main(argv)


def test_time_budget_is_handed_to_the_policy(tmp_path):
    """Policies with set_budget(steps) get the budget asked for, after their reset() and before the first act()."""
    room = fixture_wad("#####\n#P..#\n#####")
    for budget, expected in (("cap", 30.0), (77, 77.0), (None, 525.0)):
        policy = Budgeted()
        summary = D.render_episode(policy, room, tmp_path / str(budget), ticks=120, video="none", budget=budget)
        assert policy.calls == ([] if budget is None else [expected]) and summary["warnings"] == []
        assert all(r["note"] == f"budget={expected:g}" for r in trace_rows(summary))
    summary = D.render_episode(R.ScriptPolicy("F"), room, tmp_path / "plain", ticks=40, video="none", budget="cap")
    assert len(summary["warnings"]) == 1 and "set_budget" in summary["warnings"][0]
    # rows that carry their budget (the teacher's gate logs one) and their policy are replayed with both
    logged = R.run_episodes("test_demo:Budgeted", [{"wad": room, "id": "room", "episode_ticks": 40}], workers=0, seeds=[0, 1])
    E.write_episodes(tmp_path / "gate.csv", [dict(logged[0], budget_steps=12.5), dict(logged[1], budget_steps="")])
    assert D.main(["--episodes", str(tmp_path / "gate.csv"), "--video", "none", "--out", str(tmp_path / "replayed")]) == 0
    notes = [trace_rows({"trace": str(tmp_path / "replayed" / f"room_seed{k}" / "trace.csv")})[0]["note"] for k in (0, 1)]
    assert notes == ["budget=12.5", "budget=525"]
    with open(tmp_path / "replayed" / "index.csv", newline="") as f:
        assert [r["same_as_logged"] for r in csv.DictReader(f)] == ["1", "1"]
    assert D.main(["--episodes", str(tmp_path / "gate.csv"), "--budget", "9", "--limit", "1", "--video", "none",
                   "--out", str(tmp_path / "forced")]) == 0
    assert trace_rows({"trace": str(tmp_path / "forced" / "room_seed0" / "trace.csv")})[0]["note"] == "budget=9"
    E.write_episodes(tmp_path / "anonymous.csv", [dict(logged[0], policy="")])
    with pytest.raises(SystemExit):                              # no policy in the rows and none given
        D.main(["--episodes", str(tmp_path / "anonymous.csv"), "--video", "none", "--out", str(tmp_path / "x")])


def test_time_limits_can_be_applied(tmp_path):
    """With the evaluation's limits on, an agent that uses up the map's compute budget is stopped, and the panels say so."""
    room = fixture_wad("#####\n#P..#\n#####")
    try:                                                         # 6 decision steps: 0.24 s of agent time; act() takes 0.1 s
        summary = D.render_episode(Budgeted("F*6", delay=0.1), room, tmp_path / "limited", ticks=24, video="none", time_scale=1.0)
    except TypeError:
        pytest.skip("this runner has no time limits")
    row = summary["row"]
    if row is None:
        pytest.skip(f"this runner has no time limits ({summary['error']})")
    assert ": timeout: " in row["agent_error"] and row["steps"] == 6
    rows = trace_rows(summary)
    assert [r["acted"] for r in rows][:2] == ["1", "1"] and rows[-1]["acted"] == "0" and rows[-1]["action"] == "........."
    assert "no longer calls this policy" in rows[-1]["note"]
    free = D.render_episode(Budgeted("F*6", delay=0.1), room, tmp_path / "free", ticks=24, video="none")
    assert free["row"]["agent_error"] == "" and all(r["acted"] == "1" for r in trace_rows(free))


def test_replay_of_a_trace(tmp_path, capsys):
    """--replay presses the recorded buttons again: the same trajectory without the policy that chose them."""
    first = D.render_episode(HookedPolicy("F*9 L*4 F*6 <*3 U A N*2 R*5 B*4 W"), fixture_wad(KEY), tmp_path / "first", seed=4, ticks=160,
                             video="none", map_id="keymap", tier=3)
    script = D.script_from_trace(first["trace"])
    assert script.split()[:10] == ["F"] * 9 + ["L"] and script.split()[21:26] == ["<", "U", "A", "N", "N"] and len(script.split()) == 40
    code = D.main(["--replay", first["trace"], "--out", str(tmp_path / "again"), "--video", "none"])
    assert code == 0 and "WARNING" not in capsys.readouterr().out
    again = json.loads((tmp_path / "again" / "episode.json").read_text())
    assert again["map"] == first["map"] and again["policy"] == f"replay of {first['trace']}" and again["pose_source"] == "shadow"
    same = ("steps", "tic", "health", "keys_picked", "sectors_visited", "cells_visited", "exit_dist_final", "stuck_rate", "score")
    assert {k: again["row"][k] for k in same} == {k: first["row"][k] for k in same}
    a, b = trace_rows(first), trace_rows({"trace": str(tmp_path / "again" / "trace.csv")})
    for column in ("action", "world_x", "world_y", "true_heading_deg", "use_ok", "est_x", "est_y"):   # the shadow odometry agrees too
        assert [r[column] for r in a] == [r[column] for r in b], column
    # on another map, or with the seed given, the buttons are simply played there
    room = fixture_wad("#####\n#P..#\n#####")
    assert D.main(["--replay", first["trace"], "--map", room, "--seed", "1", "--ticks", "60", "--out", str(tmp_path / "other"),
                   "--video", "none"]) == 0
    other = json.loads((tmp_path / "other" / "episode.json").read_text())
    assert other["map"]["seed"] == 1 and other["row"]["steps"] == 15 and other["map"]["wad"] == room
    assert [r["action"] for r in trace_rows({"trace": str(tmp_path / "other" / "trace.csv")})] == [r["action"] for r in a[:15]]
    lonely = tmp_path / "lonely" / "trace.csv"
    lonely.parent.mkdir()
    lonely.write_text(Path(first["trace"]).read_text())
    for argv in (["--replay", str(lonely), "--out", str(tmp_path / "x")],
                 ["--replay", str(tmp_path / "none.csv"), "--map", TWO_ROOMS, "--out", str(tmp_path / "x")],
                 ["--replay", first["trace"], "--episodes", "e.csv", "--out", str(tmp_path / "x")]):
        with pytest.raises(SystemExit):
            D.main(argv)


# ---------------------------------------------------------------------------------------- slow
@pytest.mark.slow
def test_time_to_render_a_525_step_episode(tmp_path):
    """The number asked for: wall time for a full 2100-tic episode, video and sheets included."""
    layout = layouts.generate(0, 31)                            # monster-free, so the episode runs to the cap
    wad = udmf.write_wad(layout["grid"], tmp_path / "map.wad")
    for name, policy in (("random", ("training.runner:RandomPolicy", {"attack": False, "p": 0.3})), ("hooked", HookedPolicy("F*8 L*3 " * 60))):
        summary = D.render_episode(policy, wad, tmp_path / name, ticks=2100)
        t = summary["timing"]
        print(f"\n{name}: {summary['row']['steps']} steps in {t['total_s']:.2f} s ({t['ms_per_step']:.2f} ms per step); viewer "
              f"{t['viewer_s']:.2f} s, policy {t['policy_s']:.2f} s, video close {t['video_close_s']:.2f} s, sheets {t['sheets_s']:.2f} s; "
              f"{Path(summary['video']).name} {Path(summary['video']).stat().st_size / 1e6:.1f} MB")
        assert summary["row"]["steps"] == 525 and summary["warnings"] == []
        check_outputs(summary)
        assert t["total_s"] < 60.0


@pytest.mark.slow
def test_long_episode_with_several_sheets(tmp_path):
    layout = layouts.generate(0, 31)
    wad = udmf.write_wad(layout["grid"], tmp_path / "map.wad")
    turning = ("training.runner:ScriptPolicy", {"script": "L*2000"})          # stays in its cell, so the episode runs to the cap
    summary = D.render_episode(turning, wad, tmp_path / "out", ticks=4200, sheets=2, video="gif", fps=8.75)
    assert summary["row"]["steps"] == 1050 and summary["warnings"] == []
    assert [Path(p).name for p in summary["sheets"]] == ["sheet_even_1.png", "sheet_even_2.png", "sheet_final.png"]
    with Image.open(summary["video"]) as im:
        assert im.n_frames == summary["video_frames"] <= 242


@pytest.mark.slow
def test_views_the_submission_agent(tmp_path, monkeypatch):
    """runner.AgentPolicy wraps submission/agent.py; whatever hooks that agent has, an episode can be viewed."""
    monkeypatch.setenv("DOOMNET_ALLOW_UNTRAINED", "1")
    try:
        policy = R.AgentPolicy()
    except Exception as e:  # noqa: BLE001 - the agent is another area's file
        pytest.skip(f"the submission agent could not be built: {type(e).__name__}: {e}")
    for name, wad in (("reference", fixture_wad(KEY)), ("two_rooms", TWO_ROOMS)):
        summary = D.render_episode(policy, wad, tmp_path / name, ticks=400)
        print(f"\nagent on {name}: pose {summary['pose_source']}, map {summary['map_source']}, warnings {summary['warnings']}, "
              f"agent_error {summary['row']['agent_error']!r}")
        assert summary["row"]["steps"] >= 1 and summary["row"]["error"] == ""
        check_outputs(summary)


def find_teacher_policy():
    try:
        package = importlib.import_module("training.teacher")
    except ImportError:
        return None
    if hasattr(package, "TeacherPolicy"):
        return package.TeacherPolicy
    for info in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"training.teacher.{info.name}")
        except Exception:  # noqa: BLE001 - a module of another area that is still being written
            continue
        if hasattr(module, "TeacherPolicy"):
            return module.TeacherPolicy
    return None


@pytest.mark.slow
def test_views_the_teacher(tmp_path):
    teacher = find_teacher_policy()
    if teacher is None:
        pytest.skip("training.teacher has no TeacherPolicy yet")
    layout = layouts.generate(0, 31)
    wad = udmf.write_wad(layout["grid"], tmp_path / "map.wad")
    summary = D.render_episode(teacher(), wad, tmp_path / "out", ticks=layout["episode_ticks"])
    print(f"\nteacher: {D.outcome_banner(summary['row'])[0]} after {summary['row']['steps']} steps, pose {summary['pose_source']}, "
          f"map {summary['map_source']}, warnings {summary['warnings']}, agent_error {summary['row']['agent_error']!r}")
    assert summary["row"]["error"] == ""
    check_outputs(summary)
