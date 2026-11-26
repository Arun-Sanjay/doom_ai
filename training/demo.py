"""Debug viewer: plays one episode headless and writes a video and contact sheets of what the agent
saw, what it remembers and what it did.

Training-time scaffolding, never imported by the agent. CPU only, no window, no network.

    python -m training.demo --policy training.runner:RandomPolicy --map maps/two_rooms.wad --out runs/demo/random
    python -m training.demo --policy 'training.runner:ScriptPolicy{"script": "F*40 U N*8 F*40"}' \\
        --grid my_layout.txt --out runs/demo/edit --seed 3 --ticks 2800
    python -m training.demo --policy SPEC --grid training/suites/dev_v1.jsonl:dev_v1_t3_004 --no-monsters --out runs/demo/t3
    python -m training.demo --policy SPEC --episodes runs/eval/TAG.csv --failed --video none --out runs/demo/TAG_failures
    python -m training.demo --replay runs/demo/NAME/trace.csv --out runs/demo/NAME_again

`--episodes` replays episodes that `training.evaluate` (or the teacher's gate) has scored: all of
them, the failed ones, or those matching `--where 'KEY OP VALUE'`; one directory per episode plus
an `index.csv`. Without `--policy` each episode is replayed with the policy it was logged with.
`--replay` presses the buttons recorded in a trace again, on the same map with the same seed, so a
trajectory stays reproducible after the policy that produced it has changed.

Outputs, all inside the `--out` directory:

    video.mp4        one frame per decision step, then the outcome held for a moment. Frames are piped
                     to the `ffmpeg` binary on PATH; without a working ffmpeg an animated `video.gif`
                     is written with Pillow instead.
    sheet_even.png   contact sheet: 4 x 4 panels sampled evenly over the episode
    sheet_final.png  contact sheet: the final 16 decision steps
    trace.csv        one row per decision step: action, sub-goal, target, poses, drift, running totals
                     (target_i, target_j: the target as a cell of the current crop; target_ref: how
                     it was reported: xy, cell, origin, or held when its crop had to be inferred)
    episode.json     the runner's result row, the output paths, timings and any warnings
    map.wad          the compiled map, when the map was given as a grid

One panel shows one decision step, as the agent met it: the frame it was given (left, with the
buttons it then pressed underneath) and its memory-map crop (right). The crop is drawn top-down in
the agent's start frame, x (forward at the start) to the right and y (left at the start) up, one
square per 64-unit cell, coloured by channel (the legend is on every panel). By default the
picture is fixed on the whole map and as large as the map allows; the cells are still those of
the crop, which is centred on the agent, and `--zoom 1` shows the crop exactly as the network
gets it. On the picture: the pose estimate (ring and heading line), the true position brought into
the same frame (magenta dot, so drift shows as the dot leaving the ring), and the target cell if
the policy reports one (yellow box, with the planned route when that is reported too; a `~` before
the target in the text strip means its place was inferred, see "target" below). The text
strip gives the step and tic, the sub-goal name, the pose estimate (x, y, heading in degrees)
against the true pose with the drift, and the scorer's running totals (sectors, keys, doors,
locked doors, kills, health, damage, cells, distance to the exit, stuck rate, door reversals,
composite score). All of it describes the moment of the frame: the totals are those before the
action shown is applied. The true pose, the totals and the outline used to place the fitted
picture are ground truth read by the viewer; the agent sees none of them.

Policies
--------
Anything `training.runner` accepts: a `"module:factory"` spec, optionally followed by a JSON
object of keyword arguments. The episode itself is played by `runner.run_episode`, so the call
order, the handling of a policy that raises and the result row are exactly the runner's. The
viewer reads the following from the policy when it is there (also one level down, in `.agent`,
which is where `runner.AgentPolicy` keeps the submission's agent):

    pose_estimate()   -> (x, y, theta) in the start frame
    exit_believed()   -> bool
    memory            the policy's `MemoryMap` (attribute or method; `memory_map` and `mem` also work)
    odometry          the policy's `Odometry` (any attribute with x, y and theta), read when there
                      is no pose_estimate()
    set_budget(steps) called after reset() with the time budget given by `--budget` (decision steps,
                      or `cap` for the true cap) or logged with the episode; the teacher has this hook
    debug_state()     -> dict, every key optional:
        "subgoal"        index into spec.SUBGOALS, or a name
        "target"         crop cell index i * CROP_SIZE + j, or (i, j), in the crop the target was chosen in
        "target_origin"  `memory.crop_origin(pose)` of that crop. A manager's choice is held for
                         several steps while the crop moves with the agent, so the index alone does
                         not say which cell is meant. Without this key (and without "target_cell")
                         the viewer takes the crop of the step at which the reported index last
                         changed or a manager period began (step % MANAGER_PERIOD == 0, the schedule
                         of PolicyNet), marks the target with `~` and warns once: that is exact for
                         a choice held on that schedule and a guess for anything else
        "target_cell"    the target as a start-frame cell (cx, cy); cell k spans [64k - 32, 64k + 32);
                         wins over "target"
        "target_xy"      the target as (x, y) in start-frame map units; wins over the other forms
        "cell"           the agent's own start-frame cell. When it is given, a pair in "target" is
                         read as a start-frame cell as well (this is how the teacher reports)
        "route"          start-frame cells of the planned route, drawn as a line
        "waypoint"       (x, y) in start-frame map units that the controller steers at, drawn as a cross
        "crop"           the (MAP_CHANNELS, 64, 64) crop the networks were given this step
        "pose"           (x, y, theta), used when there is no pose_estimate()
        "note"           free text; every other small value is printed as key=value on the debug line

Whatever the policy does not expose is taken from a shadow `Odometry` and `MemoryMap` that the
viewer runs itself on the same observations and the actions actually executed, so a random policy
or a script can be viewed too. The shadow map's exit channel is written from ground-truth exit
labels (`training.exit_label`), since no policy code is involved; the panel says which map and
which pose it shows ("policy" or "shadow").

`render_episode()` is the same thing as a function; `render_rows()` replays a list of result rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training import paths  # noqa: F401  (puts submission/ on sys.path; must come before doomnet)
from doom_env.remote import validate_action
from doomnet import spec
from doomnet.memory import MemoryMap, crop_origin
from doomnet.odometry import Odometry
from training import runner
from training.mapmeta import CELL, MapTooLarge, load_meta
from training.scorer import EpisodeScorer

# ---- panel layout (pixels) -------------------------------------------------------------------------
SCALE = 2                                             # the 160 x 120 frame is shown at twice its size
RGB_W, RGB_H = spec.SCREEN_W * SCALE, spec.SCREEN_H * SCALE
MAP_PX = 256                                          # side of the memory-map picture
BAR_H = MAP_PX - RGB_H                                # strip of buttons under the frame
PANEL_W = RGB_W + MAP_PX
LINE_H = 12
INFO_LINES, LEGEND_LINES = 5, 2
TEXT_H = (INFO_LINES + LEGEND_LINES) * LINE_H + 4
PANEL_H = MAP_PX + TEXT_H                             # 576 x 344: both even, as H.264 needs
SHEET_COLS, SHEET_ROWS = 4, 4
SHEET_PANELS = SHEET_COLS * SHEET_ROWS
SHEET_GAP = 4
SHEET_HEADER_H = 2 * LINE_H + 8
ZOOMS = (1, 2, 4)                                     # agent-centred views: 64, 32 or 16 cells per side
FIT_MARGIN = 1                                        # cells of space around the map in the fitted view
FIT_MIN_CELLS = 12                                    # ... which never shows fewer cells than this per side
DEFAULT_POLICY = "training.runner:AgentPolicy"         # the submission's agent, loaded as the scorer loads it
DEFAULT_FPS = 15.0                                    # one decision step is 4 tics: 8.75 frames per second is real time
END_HOLD_S = 1.5                                      # how long the outcome frame stays up

# Pillow's built-in bitmap font: 6 x 11 pixel characters, no font file involved.
FONT = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)()
CHAR_W = 6
MAX_CHARS = (PANEL_W - 6) // CHAR_W                   # characters that fit on one text line of a panel

# ---- colours ---------------------------------------------------------------------------------------
C_BG = (12, 12, 16)
C_UNKNOWN = (26, 26, 32)
C_FREE = (84, 88, 98)
C_MIXED = (150, 138, 118)       # a 64-unit cell that holds both free space and wall
C_WALL = (226, 226, 216)
C_DOOR = (255, 140, 0)
C_LOCKED = (232, 50, 205)
C_KEY = (255, 238, 60)
C_ITEM = (60, 200, 90)
C_ENEMY = (255, 48, 40)
C_EXIT = (0, 236, 255)
C_VISITED = (52, 92, 170)       # trail dot of a cell visited long ago
C_TRAIL = (160, 208, 255)       # ... of a cell visited just now
C_USED = (255, 255, 255)
C_AGENT = (255, 255, 255)
C_TRUE = (255, 0, 255)
C_TARGET = (255, 255, 0)
C_ROUTE = (190, 190, 0)         # planned route, or the straight line to the target when no route is reported
C_TEXT = (214, 214, 214)
C_DIM = (140, 140, 148)
C_GOOD = (90, 220, 110)
C_WARN = (255, 170, 40)
C_BAD = (255, 80, 70)
C_HILITE = (255, 238, 60)
LEGEND_CELLS = (("unknown", C_UNKNOWN), ("free", C_FREE), ("wall", C_WALL), ("wall+free", C_MIXED), ("door", C_DOOR),
                ("locked", C_LOCKED), ("key", C_KEY), ("item", C_ITEM), ("enemy", C_ENEMY), ("exit", C_EXIT))

BUTTON_LABELS = ("FWD", "BACK", "<STR", "STR>", "<TURN", "TURN>", "ATK", "USE", "WPN")
BUTTON_LETTERS = "FB<>LRAUW"
CENTER = spec.CROP_SIZE // 2
CROP_CELL = spec.MAP_RES * spec.CROP_MULT
STEP_TICS = spec.FRAME_SKIP
NAN = float("nan")

NOTE_ORDER = ("mode", "cell", "heads", "use_age", "keys", "fight", "travel", "decided")     # debug values shown first
TRACE_FIELDS = ("step", "tic", "action", "acted", "subgoal", "target_i", "target_j", "target_ref", "est_x", "est_y", "est_heading_deg",
                "true_x", "true_y", "true_heading_deg", "world_x", "world_y", "drift", "heading_err_deg", "pose_source",
                "map_source", "use_ok", "attack_ok", "exit_label_visible", "health", "damage_taken", "kills", "keys_picked",
                "doors_opened", "locked_opened", "sectors_visited", "cells_visited", "exit_dist", "stuck_rate",
                "door_reversals", "score", "note")


# ------------------------------------------------------------------------------------ small helpers
def _ascii(text) -> str:
    """Text the bitmap font can draw: ASCII on one line (runs of spaces are kept, they align columns)."""
    return re.sub(r"[\t\r\n\f\v]+", " ", str(text)).encode("ascii", "replace").decode("ascii")


def _plain(value):
    """A debug value as plain Python: tensors and arrays become lists or scalars, anything else is kept."""
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(value, attr) and not isinstance(value, np.ndarray):
            try:
                value = getattr(value, attr)()
            except Exception:  # noqa: BLE001 - not a tensor after all
                break
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _short(value) -> str | None:
    """Compact text of a small debug value; None for values too large to print (arrays, nested data)."""
    value = _plain(value)
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, str):
        return value if len(value) <= 40 else value[:37] + "..."
    if value is None:
        return "-"
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) <= 9:
        parts = [_short(v) for v in value]
        if all(p is not None and len(p) <= 8 for p in parts):
            return "(" + ",".join(parts) + ")"
    return None


def _pose(value):
    """(x, y, theta) as three floats, or None if `value` is not a finite pose."""
    try:
        value = _plain(value)
        x, y, theta = float(value[0]), float(value[1]), float(value[2])
    except Exception:  # noqa: BLE001 - whatever a policy returned
        return None
    return (x, y, theta) if math.isfinite(x) and math.isfinite(y) and math.isfinite(theta) else None


def _wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _json_safe(value):
    """NaN and infinity are not JSON: write them as null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ratio(row: dict, part: str, total: str) -> str:
    a, b = row.get(part, NAN), row.get(total, NAN)
    left = "-" if a != a else str(int(a))
    return left if b != b else f"{left}/{int(b)}"


# ---------------------------------------------------------------------------------- map picture
class View:
    """Which cells the map picture shows, and how large.

    `corner` None: the `cells` x `cells` cells around the agent's own cell, so the picture is the
    crop itself (cells = 64) or its middle part, and scrolls as the agent crosses cells.
    `corner` (cx, cy): the picture is fixed in the start frame, with start-frame cell (cx, cy) at
    its bottom-left. The cells are still read from the crop of the moment, which reaches 32 cells
    from the agent in every direction; what lies outside it is drawn as unknown.
    """

    def __init__(self, cells: int, corner: tuple[int, int] | None = None):
        self.cells = max(1, int(cells))
        self.corner = None if corner is None else (int(corner[0]), int(corner[1]))
        self.px = max(1, MAP_PX // self.cells)            # pixels per cell
        self.margin = (MAP_PX - self.cells * self.px) // 2

    @classmethod
    def centred(cls, zoom: int = 1) -> "View":
        """The middle 64 / zoom cells of the crop, around the agent (zoom 1, 2 or 4)."""
        return cls(spec.CROP_SIZE // int(zoom))

    def low(self, origin) -> tuple[int, int]:
        """Crop index of the picture's bottom-left cell, for a crop whose index (0, 0) is start-frame cell `origin`."""
        if self.corner is None:
            return CENTER - self.cells // 2, CENTER - self.cells // 2
        return self.corner[0] - origin[0], self.corner[1] - origin[1]

    def describe(self) -> str:
        if self.corner is None:
            return f"{self.cells} x {self.cells} cells around the agent"
        return f"fixed on the map, {self.cells} x {self.cells} cells"


def _upscale(a: np.ndarray, px: int) -> np.ndarray:
    return np.repeat(np.repeat(a, px, axis=0), px, axis=1)


def _to_image_axes(a: np.ndarray) -> np.ndarray:
    """[x cell, y cell, ...] -> [row, column, ...] with x to the right and y up."""
    return np.swapaxes(a, 0, 1)[::-1]


def map_image(crop, view: View | None = None, origin=(-CENTER, -CENTER)) -> np.ndarray:
    """Colour picture (MAP_PX, MAP_PX, 3) uint8 of a memory-map crop (MAP_CHANNELS, 64, 64).

    `view` chooses the cells shown (default: the whole crop); `origin` is the start-frame cell
    at crop index (0, 0), needed only for a view that is fixed in the start frame.
    Each cell is filled by what it is known to hold, later entries covering earlier ones: unknown,
    free, wall (a separate colour where the cell is also free), item, key, door, locked door, then
    enemy and exit blended in by their strength. A dot in the middle of a cell marks a visit, the
    brighter the more recent; a white frame marks a recent USE press.
    """
    view = view or View.centred(1)
    c = np.asarray(crop)
    c = c.astype(np.float32) / 255.0 if c.dtype == np.uint8 else np.nan_to_num(c.astype(np.float32))
    n, px = view.cells, view.px
    lo_i, lo_j = view.low(origin)
    w = np.zeros((c.shape[0], n, n), np.float32)          # the window; cells outside the crop stay unknown
    i0, i1, j0, j1 = max(lo_i, 0), min(lo_i + n, c.shape[1]), max(lo_j, 0), min(lo_j + n, c.shape[2])
    if i0 < i1 and j0 < j1:
        w[:, i0 - lo_i:i1 - lo_i, j0 - lo_j:j1 - lo_j] = c[:, i0:i1, j0:j1]
    free, wall = w[spec.CH_FREE] > 0.5, w[spec.CH_WALL] > 0.5
    col = np.empty((n, n, 3), np.float32)
    col[:] = C_UNKNOWN
    col[free] = C_FREE
    col[wall & free] = C_MIXED
    col[wall & ~free] = C_WALL
    for ch, colour in ((spec.CH_ITEM, C_ITEM), (spec.CH_KEY, C_KEY), (spec.CH_DOOR, C_DOOR), (spec.CH_LOCKED, C_LOCKED)):
        col[w[ch] > 0.5] = colour
    for ch, colour in ((spec.CH_ENEMY, C_ENEMY), (spec.CH_EXIT, C_EXIT)):
        v = np.clip(w[ch], 0.0, 1.0)
        alpha = np.where(v > 0.0, 0.35 + 0.65 * v, 0.0)[..., None]
        col = col * (1.0 - alpha) + np.asarray(colour, np.float32) * alpha
    big = _upscale(_to_image_axes(col), px)
    if px >= 7:                                           # cell borders, where there is room for them
        big[::px, :] *= 0.8
        big[:, ::px] *= 0.8
    q = max(1, px // 4)
    inner = np.zeros((px, px), bool)
    inner[q:px - q, q:px - q] = True
    recency = np.clip(w[spec.CH_RECENCY], 0.0, 1.0)
    visited = (w[spec.CH_VISITED] > 0.5) | (recency > 0.0)
    dots = _upscale(_to_image_axes(visited), px) & np.tile(inner, (n, n))
    dot_col = np.asarray(C_VISITED, np.float32) + (np.asarray(C_TRAIL, np.float32) - np.asarray(C_VISITED, np.float32)) * recency[..., None]
    big[dots] = _upscale(_to_image_axes(dot_col), px)[dots]
    used = np.clip(w[spec.CH_USED], 0.0, 1.0)
    if used.any() and px >= 3:
        frame = np.ones((px, px), bool)
        frame[1:px - 1, 1:px - 1] = False
        ring = _upscale(_to_image_axes(used > 0.0), px) & np.tile(frame, (n, n))
        alpha = _upscale(_to_image_axes(0.3 + 0.7 * used), px)[ring][:, None]
        big[ring] = big[ring] * (1.0 - alpha) + np.asarray(C_USED, np.float32) * alpha
    out = np.empty((MAP_PX, MAP_PX, 3), np.uint8)
    out[:] = C_BG
    m = view.margin
    out[m:m + n * px, m:m + n * px] = np.clip(big, 0.0, 255.0).astype(np.uint8)
    return out


class _MapFrame:
    """Start-frame map units and cells of a crop -> pixels of the map picture, for one pose and view."""

    def __init__(self, pose, view: View):
        self.view, self.px = view, view.px
        self.origin = crop_origin(pose) if pose is not None else (-CENTER, -CENTER)
        self.lo = view.low(self.origin)

    def point(self, x: float, y: float) -> tuple[float, float]:
        ci = x / CROP_CELL + 0.5 - self.origin[0]
        cj = y / CROP_CELL + 0.5 - self.origin[1]
        return self.view.margin + (ci - self.lo[0]) * self.px, self.view.margin + (self.lo[1] + self.view.cells - cj) * self.px

    def cell_box(self, i: int, j: int) -> tuple[int, int, int, int]:
        """Pixel box of crop cell (i, j)."""
        x0 = self.view.margin + (i - self.lo[0]) * self.px
        y0 = self.view.margin + (self.lo[1] + self.view.cells - 1 - j) * self.px
        return x0, y0, x0 + self.px - 1, y0 + self.px - 1


# -------------------------------------------------------------------------------------- drawing
def _segments(draw, x: int, y: int, parts, limit: int) -> int:
    """Draw (text, colour) pieces left to right from x, cut at pixel `limit`; returns where it stopped."""
    for text, colour in parts:
        text = _ascii(text)
        room = (limit - x) // CHAR_W
        if room <= 0:
            break
        text = text[:room]
        draw.text((x, y), text, fill=colour, font=FONT)
        x += len(text) * CHAR_W
    return x


def _big_text(img: Image.Image, xy, text: str, scale: int, colour) -> None:
    """Text drawn `scale` times larger (the bitmap font has one size)."""
    text = _ascii(text)
    mask = Image.new("L", (max(1, len(text) * CHAR_W), 11), 0)
    ImageDraw.Draw(mask).text((0, 0), text, fill=255, font=FONT)
    mask = mask.resize((mask.width * scale, mask.height * scale), Image.NEAREST)
    img.paste(Image.new("RGB", mask.size, colour), (int(xy[0]), int(xy[1])), mask)


def _draw_buttons(draw, action, masks) -> None:
    """The nine buttons under the frame: lit when pressed; ATK and USE are framed green while the
    action mask of spec.action_masks allows them and lit orange when pressed without effect."""
    slot = RGB_W / len(BUTTON_LABELS)
    allowed = {6: bool(masks[1]), 7: bool(masks[0])} if masks is not None else {}
    for k, label in enumerate(BUTTON_LABELS):
        x0, x1 = int(round(k * slot)) + 1, int(round((k + 1) * slot)) - 2
        y0, y1 = RGB_H + 2, RGB_H + BAR_H - 2
        pressed = action is not None and bool(action[k])
        fill = ((C_WARN if allowed.get(k) is False else (236, 236, 236)) if pressed else (34, 34, 40))
        outline = C_GOOD if allowed.get(k) else (70, 70, 78)
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline)
        tx = x0 + ((x1 - x0 + 1) - len(label) * CHAR_W) // 2
        draw.text((tx, y0 + 1), label, fill=(0, 0, 0) if pressed else C_DIM, font=FONT)


def _draw_markers(draw, f: dict) -> None:
    """Target cell, true position and pose estimate, drawn on the map picture (its own pixel frame)."""
    est = f.get("est")
    if est is None:
        return
    mf = _MapFrame(f.get("frame_pose") or est, f["view"])
    ax, ay = mf.point(est[0], est[1])
    box, route = f.get("target_box"), f.get("route_px")
    if route:                                             # the planned route, from the estimate through the cell centres
        draw.line([(ax, ay), *route[1:]] if len(route) > 1 else [(ax, ay), route[0]], fill=C_ROUTE, width=1)
    elif box is not None:                                 # no route known: a straight line to the target
        draw.line((ax, ay, (box[0] + box[2] + 1) / 2.0, (box[1] + box[3] + 1) / 2.0), fill=C_ROUTE, width=1)
    if box is not None:
        draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), outline=C_TARGET)
    way = f.get("waypoint_px")
    if way is not None:
        draw.line((way[0] - 3, way[1], way[0] + 3, way[1]), fill=C_TARGET)
        draw.line((way[0], way[1] - 3, way[0], way[1] + 3), fill=C_TARGET)
    r = min(6, max(3, mf.px // 2))
    hx, hy = math.cos(est[2]), -math.sin(est[2])
    reach = min(20.0, max(8.0, 1.6 * mf.px))
    draw.line((ax, ay, ax + hx * reach, ay + hy * reach), fill=(0, 0, 0), width=3)
    draw.line((ax, ay, ax + hx * reach, ay + hy * reach), fill=C_AGENT, width=1)
    draw.ellipse((ax - r - 1, ay - r - 1, ax + r + 1, ay + r + 1), outline=(0, 0, 0))
    draw.ellipse((ax - r, ay - r, ax + r, ay + r), outline=C_AGENT)
    true = f.get("true")
    if true is not None:
        tx, ty = mf.point(true[0], true[1])
        draw.rectangle((tx - 1, ty - 1, tx + 1, ty + 1), fill=C_TRUE)


def _legend(draw, f: dict, y: int) -> None:
    x = 4
    for name, colour in LEGEND_CELLS:
        draw.rectangle((x, y + 2, x + 7, y + 9), fill=colour, outline=(70, 70, 78))
        x = _segments(draw, x + 10, y, [(name, C_DIM)], PANEL_W) + 7
    y += LINE_H
    x = 4
    draw.rectangle((x, y + 2, x + 7, y + 9), fill=C_FREE)
    draw.rectangle((x + 2, y + 4, x + 5, y + 7), fill=C_TRAIL)
    x = _segments(draw, x + 10, y, [("visited", C_DIM)], PANEL_W) + 7
    draw.rectangle((x, y + 2, x + 7, y + 9), fill=C_FREE, outline=C_USED)
    x = _segments(draw, x + 10, y, [("USE tried", C_DIM)], PANEL_W) + 7
    draw.ellipse((x, y + 2, x + 7, y + 9), outline=C_AGENT)
    draw.line((x + 4, y + 5, x + 10, y + 5), fill=C_AGENT)
    x = _segments(draw, x + 13, y, [("pose estimate", C_DIM)], PANEL_W) + 7
    draw.rectangle((x + 2, y + 4, x + 4, y + 6), fill=C_TRUE)
    x = _segments(draw, x + 9, y, [("true position", C_DIM)], PANEL_W) + 7
    draw.rectangle((x, y + 2, x + 7, y + 9), outline=C_TARGET)
    draw.line((x - 4, y + 9, x, y + 6), fill=C_ROUTE)
    x = _segments(draw, x + 10, y, [("target, route", C_DIM)], PANEL_W) + 7
    _segments(draw, x, y, [("| " + f.get("title", ""), C_DIM)], PANEL_W - 2)


def _info_lines(f: dict) -> list[list]:
    """The five text lines of a panel as lists of (text, colour) pieces.

    Fields have fixed widths, so the columns stand still from frame to frame, and every line fits
    the panel (MAX_CHARS) at the largest values a 4200-tic map can produce.
    """
    tot = f.get("totals") or {}
    sep = (" | ", C_DIM)

    def num(key, fmt, width):
        value = tot.get(key, NAN)
        return f"{'-' if value is None or value != value else format(value, fmt):>{width}}"

    target = f.get("target_text") or "-"                  # at most 13 characters: "~(63,63)=self"
    line1 = [(f"step {f['step']:4d}/{f['steps_cap']:<4d} tic {f['tic']:4d}/{f['ticks']:<4d}", C_TEXT), sep,
             ("subgoal ", C_DIM), (f"{(f.get('subgoal') or '-')[:11]:<11}", C_HILITE), sep,
             ("target ", C_DIM), (f"{target:<13}", C_TEXT)]
    if f.get("exit_believed") is not None:
        line1 += [sep, ("exit believed ", C_DIM), ("yes" if f["exit_believed"] else "no", C_GOOD if f["exit_believed"] else C_TEXT)]

    def pose_text(pose):
        return f"({pose[0]:7.1f},{pose[1]:7.1f},{_wrap_deg(math.degrees(pose[2])):6.1f})" if pose else f"({'-':>7},{'-':>7},{'-':>6})"

    line2 = [("est ", C_DIM), (pose_text(f.get("est")), C_TEXT), (" true ", C_DIM), (pose_text(f.get("true")), C_TEXT)]
    drift, turn = f.get("drift", NAN), f.get("heading_err", NAN)
    if drift == drift:
        colour = C_BAD if drift > 32.0 or abs(turn) > 2.0 else C_WARN if drift > 16.0 else C_GOOD
        line2 += [sep, ("drift ", C_DIM), (f"{min(drift, 9999.9):6.1f} u {min(abs(turn), 180.0):5.1f} deg", colour)]
    line2 += [(f" ({f.get('pose_source', '-')})", C_DIM)]

    health = tot.get("health", NAN)
    line3 = [("sectors ", C_DIM), (f"{_ratio(tot, 'sectors_visited', 'sectors_total'):>5}", C_TEXT),
             (" keys ", C_DIM), (f"{_ratio(tot, 'keys_picked', 'keys_total'):>3}", C_TEXT),
             (" doors ", C_DIM), (f"{_ratio(tot, 'doors_opened', 'doors_total'):>5}", C_TEXT),
             (" locked ", C_DIM), (f"{_ratio(tot, 'locked_opened', 'locked_total'):>3}", C_TEXT),
             (" kills ", C_DIM), (f"{_ratio(tot, 'kills', 'monsters_total'):>5}", C_TEXT),
             (" health ", C_DIM), (num("health", ".0f", 3), C_BAD if health == health and health < 30.0 else C_TEXT),
             (" dmg ", C_DIM), (num("damage_taken", ".0f", 4), C_TEXT),
             (" cells ", C_DIM), (f"{_ratio(tot, 'cells_visited', 'cells_total'):>7}", C_TEXT)]

    if not tot.get("has_exit"):
        line4 = [("exit ", C_DIM), (f"{'none on this map':<29}", C_TEXT)]
    else:
        seen = tot.get("exit_visible_tic", NAN)
        in_view = f"in view since tic {int(seen)}" if seen == seen and seen >= 0 else "not yet in view"
        line4 = [("exit ", C_DIM), (num("exit_dist_final", ".0f", 4) + " u", C_TEXT), (f", {in_view:<21}", C_DIM)]
    stuck = tot.get("stuck_rate", NAN)
    reversals = tot.get("door_reversals", 0) or 0
    line4 += [sep, ("stuck ", C_DIM), (num("stuck_rate", ".2f", 4), C_WARN if stuck == stuck and stuck > 0.2 else C_TEXT),
              (" +door ", C_DIM), (num("stuck_at_door_rate", ".2f", 4), C_TEXT),
              sep, ("reversals ", C_DIM), (f"{int(reversals):<2d}", C_BAD if reversals else C_TEXT),
              sep, ("score ", C_DIM), (num("score", ".1f", 5), C_TEXT)]

    line5 = [(f"map: {f.get('map_source', '-')}", C_DIM)]
    if f.get("error"):
        line5 += [sep, (f["error"], C_BAD)]
    if f.get("note"):
        line5 += [sep, (f["note"], C_TEXT)]
    return [line1, line2, line3, line4, line5]


def compose(f: dict) -> Image.Image:
    """One panel (PANEL_W x PANEL_H) from the state of one decision step.

    Keys of `f`: rgb (120, 160, 3) uint8; crop (MAP_CHANNELS, 64, 64); view; frame_pose, the pose
    the crop is centred on; est and true, the pose estimate and the true pose (x, y, theta) in the
    start frame, or None; action (9,) or None; masks (use_ok, attack_ok) or None; the text fields
    used by `_info_lines`; optional target_box, route_px and waypoint_px (pixels of the map
    picture), title and banner.
    """
    img = Image.new("RGB", (PANEL_W, PANEL_H), C_BG)
    rgb = np.ascontiguousarray(f["rgb"], np.uint8)
    img.paste(Image.fromarray(rgb).resize((RGB_W, RGB_H), Image.NEAREST), (0, 0))
    map_img = Image.fromarray(map_image(f["crop"], f["view"], crop_origin(f["frame_pose"]) if f.get("frame_pose") else (-CENTER, -CENTER)))
    _draw_markers(ImageDraw.Draw(map_img), f)             # drawn before pasting, so nothing spills out of the map
    img.paste(map_img, (RGB_W, 0))
    draw = ImageDraw.Draw(img)
    _draw_buttons(draw, f.get("action"), f.get("masks"))
    y = MAP_PX + 2
    for line in _info_lines(f):
        _segments(draw, 4, y, line, PANEL_W - 2)
        y += LINE_H
    _legend(draw, f, y)
    banner = f.get("banner")
    if banner:
        text, colour = banner
        draw.rectangle((0, 0, RGB_W - 1, 43), fill=(0, 0, 0))
        _big_text(img, (8, 5), text, 3, colour)
    return img


def fallback_panel(rgb, message: str) -> Image.Image:
    """The bare frame with an error line, for a step whose panel could not be drawn."""
    img = Image.new("RGB", (PANEL_W, PANEL_H), C_BG)
    try:
        img.paste(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)).resize((RGB_W, RGB_H), Image.NEAREST), (0, 0))
    except Exception:  # noqa: BLE001
        pass
    _segments(ImageDraw.Draw(img), 4, MAP_PX + 2, [("viewer error: " + message, C_BAD)], PANEL_W - 2)
    return img


def contact_sheet(panels: list, header: list[str]) -> Image.Image:
    """Up to 16 panels in a 4 x 4 grid under a two-line header."""
    width = SHEET_COLS * PANEL_W + (SHEET_COLS + 1) * SHEET_GAP
    height = SHEET_HEADER_H + SHEET_ROWS * PANEL_H + (SHEET_ROWS + 1) * SHEET_GAP
    sheet = Image.new("RGB", (width, height), (40, 40, 46))
    draw = ImageDraw.Draw(sheet)
    for k, line in enumerate(header[:2]):
        _segments(draw, SHEET_GAP + 2, 4 + k * LINE_H, [(line, C_TEXT)], width - 4)
    for k, panel in enumerate(panels[:SHEET_PANELS]):
        row, col = divmod(k, SHEET_COLS)
        sheet.paste(panel, (SHEET_GAP + col * (PANEL_W + SHEET_GAP), SHEET_HEADER_H + SHEET_GAP + row * (PANEL_H + SHEET_GAP)))
    return sheet


class EvenSampler:
    """Keeps about `count` items spread evenly over a stream of unknown length, in bounded memory.

    Item k is kept when k is a multiple of the stride; whenever 2 * count items are held, every
    second one is dropped and the stride doubles. `pick()` returns `count` of what is left.
    """

    def __init__(self, count: int):
        self.count, self.stride, self.items = int(count), 1, []

    def add(self, index: int, item) -> None:
        if index % self.stride == 0:
            self.items.append((index, item))
            if len(self.items) >= 2 * self.count:
                self.items = self.items[::2]
                self.stride *= 2

    def pick(self) -> list:
        n = len(self.items)
        if n <= self.count:
            return list(self.items)
        chosen = sorted({int(round(k * (n - 1) / (self.count - 1))) for k in range(self.count)})
        return [self.items[k] for k in chosen]


# ---------------------------------------------------------------------------------------- video
class VideoError(RuntimeError):
    pass


# Encoders in order of preference, with their quality settings. The first one that the installed
# ffmpeg lists and that passes a three-frame trial is used.
ENCODERS = (
    ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]),
    ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-b:v", "3M"]),
    ("libopenh264", ["-c:v", "libopenh264", "-b:v", "3M"]),
    ("mpeg4", ["-c:v", "mpeg4", "-q:v", "3"]),
)
_ENCODER_CACHE: dict[str, tuple | None] = {}


class FfmpegWriter:
    """Pipes raw RGB frames into ffmpeg, which writes an MP4."""

    kind = "mp4"

    def __init__(self, exe: str, path, size: tuple[int, int], fps: float, encoder_args: list[str]):
        self.path = Path(path)
        self.frames = 0
        self._log = tempfile.TemporaryFile()
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-", "-an", *encoder_args,
               "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)

    def _stderr(self) -> str:
        try:
            self._log.seek(0)
            return self._log.read().decode("utf-8", "replace").strip()[-600:]
        except (OSError, ValueError):
            return ""

    def write(self, img: Image.Image, repeat: int = 1) -> None:
        data = img.tobytes()
        try:
            for _ in range(repeat):
                self._proc.stdin.write(data)
        except (BrokenPipeError, OSError, ValueError) as e:
            raise VideoError(f"ffmpeg stopped accepting frames: {self._stderr() or e}") from None
        self.frames += repeat

    def close(self, timeout: float = 180.0) -> None:
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            code = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            code = -9
        message = self._stderr()
        self._log.close()
        if code != 0 or not self.path.is_file() or self.path.stat().st_size == 0:
            raise VideoError(f"ffmpeg failed (exit code {code}): {message}")

    def abort(self) -> None:
        try:
            self._proc.kill()
            self._proc.wait(timeout=10)
            self._log.close()
        except Exception:  # noqa: BLE001
            pass


class GifWriter:
    """Animated GIF through Pillow: the fallback when no ffmpeg can be used.

    Frames are held in memory until `close()`, so at most `max_frames` are kept: with a longer
    episode only every n-th step is stored and shown n times as long.
    """

    kind = "gif"

    def __init__(self, path, fps: float, max_steps: int, max_frames: int = 240):
        self.path = Path(path)
        self.fps = float(fps)
        self.stride = max(1, math.ceil(max_steps / max_frames))
        self.frames = 0
        self._seen = 0
        self._images: list[Image.Image] = []
        self._durations: list[int] = []

    def write(self, img: Image.Image, repeat: int = 1) -> None:
        keep = self._seen % self.stride == 0 or repeat > 1
        self._seen += 1
        if not keep:
            return
        self._images.append(img.quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
        self._durations.append(max(20, int(round(1000.0 * (self.stride if repeat == 1 else repeat) / self.fps))))
        self.frames += 1

    def close(self) -> None:
        if not self._images:
            raise VideoError("no frames to write")
        self._images[0].save(self.path, save_all=True, append_images=self._images[1:], duration=self._durations,
                             loop=0, optimize=False)
        self._images = []

    def abort(self) -> None:
        self._images = []


def find_encoder(exe: str) -> tuple[str, list[str]] | None:
    """The first encoder of ENCODERS that this ffmpeg has and that encodes a three-frame trial clip."""
    if exe in _ENCODER_CACHE:
        return _ENCODER_CACHE[exe]
    found = None
    try:
        listing = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    with tempfile.TemporaryDirectory() as tmp:
        for name, args in ENCODERS:
            if not re.search(rf"\s{re.escape(name)}\s", listing):
                continue
            try:
                trial = FfmpegWriter(exe, Path(tmp) / f"{name}.mp4", (64, 48), DEFAULT_FPS, args)
                trial.write(Image.new("RGB", (64, 48), (40, 80, 120)), repeat=3)
                trial.close(timeout=30.0)
            except (VideoError, OSError):
                continue
            found = (name, args)
            break
    _ENCODER_CACHE[exe] = found
    return found


def open_video(out_dir: Path, mode: str, fps: float, max_steps: int) -> tuple[object | None, str]:
    """(writer or None, note). mode: "auto" (MP4, or GIF if ffmpeg cannot be used), "mp4", "gif", "none"."""
    if mode == "none":
        return None, ""
    note = ""
    if mode in ("auto", "mp4"):
        exe = shutil.which("ffmpeg")
        encoder = find_encoder(exe) if exe else None
        if encoder is not None:
            try:
                return FfmpegWriter(exe, out_dir / "video.mp4", (PANEL_W, PANEL_H), fps, encoder[1]), f"ffmpeg {encoder[0]}"
            except OSError as e:
                note = f"ffmpeg could not be started ({e}); "
        else:
            note = ("no ffmpeg on PATH; " if not exe else "the ffmpeg on PATH has no usable H.264 or MPEG-4 encoder; ")
        note += "writing an animated GIF instead"
    return GifWriter(out_dir / "video.gif", fps, max_steps), note


# ------------------------------------------------------------------------- reading the policy
class Probe:
    """Wraps a policy for one episode: passes everything through and notes whether `act` ran and failed.

    Attribute access is forwarded, so the runner sees exactly the hooks the policy has.
    """

    def __init__(self, policy):
        self._policy = policy
        self.calls = 0
        self.error = ""

    def act(self, obs):
        self.calls += 1
        try:
            action = self._policy.act(obs)
        except Exception as e:  # noqa: BLE001 - noted for the picture, handled by the runner
            self.error = self.error or f"act raised {type(e).__name__}: {e}"
            raise
        try:
            validate_action(action)
        except Exception as e:  # noqa: BLE001
            self.error = self.error or f"invalid action: {e}"
        return action

    def __getattr__(self, name):
        return getattr(self.__dict__["_policy"], name)


def _layers(policy) -> list:
    """The policy and what it wraps (runner.AgentPolicy keeps the submission's agent in `.agent`)."""
    if isinstance(policy, Probe):
        policy = policy.__dict__["_policy"]
    out = [policy]
    for name in ("agent", "policy"):
        inner = getattr(policy, name, None)
        plain = isinstance(inner, (str, bytes, int, float, dict, list, tuple, np.ndarray))
        if inner is not None and not plain and all(inner is not seen for seen in out):
            out.append(inner)
    return out


def _hook(policy, name: str):
    for obj in _layers(policy):
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None


def _is_memory(obj) -> bool:
    return obj is not None and callable(getattr(obj, "crop", None)) and not isinstance(obj, np.ndarray)


def find_memory(policy):
    """The policy's memory map (an object with `crop(pose)`), or None.

    Looked up as the attribute or zero-argument method `memory`, `memory_map` or `mem`, then as any
    attribute that looks like a `MemoryMap`, on the policy and on the agent it wraps.
    """
    for obj in _layers(policy):
        for name in ("memory", "memory_map", "mem"):
            value = getattr(obj, name, None)
            if callable(value) and not _is_memory(value):
                try:
                    value = value()
                except Exception:  # noqa: BLE001 - a method that wants arguments
                    value = None
            if _is_memory(value):
                return value
        try:
            members = list(vars(obj).values())
        except TypeError:
            members = []
        for value in members:
            if _is_memory(value) and hasattr(value, "occ"):
                return value
    return None


def find_pose(policy):
    """(x, y, theta) from an odometry object the policy holds, for policies without `pose_estimate()`.

    Tried: the attributes `odometry` and `odom`, then any attribute with numeric `x`, `y` and `theta`.
    """
    def read(obj):
        try:
            return _pose((obj.x, obj.y, obj.theta))
        except Exception:  # noqa: BLE001 - not an odometry
            return None

    for obj in _layers(policy):
        for name in ("odometry", "odom"):
            pose = read(getattr(obj, name, None))
            if pose is not None:
                return pose
        try:
            members = list(vars(obj).values())
        except TypeError:
            members = []
        for value in members:
            if hasattr(value, "theta") and hasattr(value, "x") and not isinstance(value, (np.ndarray, dict)):
                pose = read(value)
                if pose is not None:
                    return pose
    return None


def read_debug(state) -> dict:
    """Normalise whatever `debug_state()` returned; see the module docstring for the keys."""
    out = {"subgoal": None, "target": None, "target_origin": None, "target_cell": None, "target_xy": None, "route": None,
           "waypoint": None, "crop": None, "pose": None, "note": ""}
    if state is None:
        return out
    if not isinstance(state, Mapping):
        state = dict(vars(state)) if hasattr(state, "__dict__") else {"note": state}
    state = dict(state)

    def take(*names):
        for name in names:
            if name in state:
                return state.pop(name)
        return None

    sub = _plain(take("subgoal", "sub_goal", "subgoal_name", "subgoal_index", "mode"))
    if isinstance(sub, str):
        out["subgoal"] = sub
    elif sub is not None:
        try:
            out["subgoal"] = spec.SUBGOALS[int(sub)] if 0 <= int(sub) < spec.NUM_SUBGOALS else f"#{int(sub)}"
        except (TypeError, ValueError):
            out["subgoal"] = _short(sub)
    target = _plain(take("target", "target_index"))
    if target is not None:
        try:
            if isinstance(target, (list, tuple, np.ndarray)):
                out["target"] = (int(target[0]), int(target[1]))
            elif int(target) >= 0:
                out["target"] = divmod(int(target), spec.CROP_SIZE)
        except (TypeError, ValueError, IndexError):
            pass
    for key, names in (("target_origin", ("target_origin",)), ("target_xy", ("target_xy", "target_pos")),
                       ("target_cell", ("target_cell",)), ("waypoint", ("waypoint",))):
        value = _plain(take(*names))
        try:
            if value is not None:
                out[key] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError, IndexError):
            pass
    own = _plain(state.get("cell"))
    if out["target_cell"] is None and isinstance(target, (list, tuple, np.ndarray)) and out["target"] is not None \
            and isinstance(own, (list, tuple, np.ndarray)) and len(own) == 2:
        out["target_cell"], out["target"] = (float(out["target"][0]), float(out["target"][1])), None
    route = take("route", "path")
    try:
        cells = [(int(c[0]), int(c[1])) for c in route] if route is not None else []
        out["route"] = cells[:4096] or None
    except (TypeError, ValueError, IndexError):
        pass
    crop = take("crop", "map_crop")
    if crop is not None:
        try:
            crop = np.asarray(_plain(crop))
            if crop.shape == (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE):
                out["crop"] = crop
        except Exception:  # noqa: BLE001
            pass
    out["pose"] = _pose(take("pose", "pose_estimate"))
    notes = []
    note = take("note", "text")
    if note:
        notes.append(str(note))
    first = {name: k for k, name in enumerate(NOTE_ORDER)}    # these come first: the line is cut at the panel's edge
    for key, value in sorted(state.items(), key=lambda item: first.get(item[0], len(first))):
        text = None if key == "step" else _short(value)       # the step is on the panel already
        if text is not None:
            notes.append(f"{key}={text}")
    out["note"] = "  ".join(notes)
    return out


# ------------------------------------------------------------------------------------- recorder
class Recorder:
    """Recorder hook for `runner.run_episode`: draws one panel per decision step.

    begin() and step() snapshot what belongs to an observation when it arrives (frame, action
    masks, true pose, running totals), because the policy acts on it one call later; the panel of
    step t is drawn in step(t), after the policy has acted on observation t and before it sees the
    next one, so its pose estimate, memory and debug state are the ones behind action t.
    """

    def __init__(self, out_dir, probe: Probe, *, video: str = "auto", fps: float = DEFAULT_FPS, zoom="fit",
                 exit_source: str = "label", title: str = "", sheets: int = 1, budget=None):
        self.out_dir = Path(out_dir)
        self.probe = probe
        self.budget = budget            # None, "cap" or decision steps: handed to the policy's set_budget() at the start
        self.video_mode, self.fps, self.zoom_arg, self.exit_source, self.title = video, float(fps), zoom, exit_source, title
        self.warnings: list[str] = []
        self.writer = None
        self.video_note = ""
        self.even = EvenSampler(SHEET_PANELS * max(1, int(sheets)))
        self.last = deque(maxlen=SHEET_PANELS)
        self.frames = 0                 # panels drawn: one per decision step, plus the outcome frame
        self.video_frames = 0           # frames handed to the video writer (the outcome frame is repeated)
        self.render_s = 0.0
        self.pose_source = self.map_source = "shadow"
        self.view = View.centred(1)
        self._trace_file = None
        self._trace = None
        self._snap: dict | None = None
        self._final: dict | None = None
        self.row: dict | None = None
        self.odometry, self.memory = Odometry(), MemoryMap()
        self._exit_label = None
        self._held = None               # (step, reported crop cell, crop origin) of a target reported without its origin

    # ---- helpers
    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _choose_view(self, meta, start) -> View:
        """The view asked for: 1, 2 or 4 centre the picture on the agent; "fit" fixes it on the whole map.

        The fitted view needs the outline of the map in the agent's start frame, which the viewer
        (not the agent) takes from the WAD and the true start pose `start` = (x, y, angle in
        radians). A map that does not fit into one crop (more than 64 cells across) is shown
        around the agent instead.
        """
        if self.zoom_arg not in ("fit", "auto", None):
            return View.centred(int(self.zoom_arg))
        try:
            if meta.grid is not None:
                rows, cols = meta.grid.shape
                x0, y0 = meta.grid.x0, meta.grid.y0
                x1, y1 = x0 + cols * CELL, y0 + rows * CELL
            else:
                seg = np.concatenate([np.asarray(s, np.float64).reshape(-1, 4) for s in meta.sector_polygons.values()])
                x0, x1, y0, y1 = seg[:, [0, 2]].min(), seg[:, [0, 2]].max(), seg[:, [1, 3]].min(), seg[:, [1, 3]].max()
            sx, sy, a0 = start
            c, s = math.cos(a0), math.sin(a0)
            fwd = [(x - sx) * c + (y - sy) * s for x in (x0, x1) for y in (y0, y1)]       # the four corners, start frame
            left = [-(x - sx) * s + (y - sy) * c for x in (x0, x1) for y in (y0, y1)]
            # a little inside the outline, so an edge that lies exactly on a cell border does not add a cell
            i0, i1 = math.floor((min(fwd) + 1.0) / CROP_CELL + 0.5), math.floor((max(fwd) - 1.0) / CROP_CELL + 0.5)
            j0, j1 = math.floor((min(left) + 1.0) / CROP_CELL + 0.5), math.floor((max(left) - 1.0) / CROP_CELL + 0.5)
            wide, high = i1 - i0 + 1, j1 - j0 + 1
            side = max(max(wide, high) + 2 * FIT_MARGIN, FIT_MIN_CELLS)
            if side > spec.CROP_SIZE:
                return View.centred(1)
            return View(side, (i0 - (side - wide) // 2, j0 - (side - high) // 2))
        except Exception as e:  # noqa: BLE001
            self.warn(f"the map outline could not be read ({type(e).__name__}: {e}); the picture follows the agent")
            return View.centred(1)

    def _exit_obs(self, env, meta):
        """Ground-truth exit observation for the shadow map as (confidence, forward / 1024, left / 1024)."""
        if self.exit_source != "label" or self._exit_label is False:
            return None, False
        try:
            if self._exit_label is None:
                from training import exit_label

                self._exit_label = exit_label
            visible, fwd, left = self._exit_label.exit_observation(env, meta)[:3]
        except Exception as e:  # noqa: BLE001 - the viewer goes on without exit labels
            self._exit_label = False
            self.warn(f"exit labels unavailable ({type(e).__name__}: {e}); the shadow map has no exit channel")
            return None, False
        return (1.0 if visible else 0.0, fwd / spec.RAY_RANGE, left / spec.RAY_RANGE), bool(visible)

    def _true_pose(self, st: dict):
        """True pose in the agent's start frame, from the engine's position and angle."""
        x0, y0, a0 = self._start
        dx, dy = float(st["x"]) - x0, float(st["y"]) - y0
        c, s = math.cos(a0), math.sin(a0)
        return dx * c + dy * s, -dx * s + dy * c, math.radians(float(st["angle"])) - a0

    def _snapshot(self, obs, env, totals: dict, exit_visible: bool) -> dict:
        """What belongs to an observation, copied while it is the current one."""
        st = env.privileged_state()
        rays, ents = obs["rays"], obs["visible_entities"]
        try:
            masks = spec.action_masks(rays["distance"], rays["hit_type"], ents["entity_type"], ents["mask"], obs["vitals"][2])
        except Exception:  # noqa: BLE001
            masks = None
        return {"rgb": np.array(obs["rgb"], np.uint8), "masks": masks, "true": self._true_pose(st),
                "world": (float(st["x"]), float(st["y"])), "totals": totals, "exit_visible": exit_visible}

    # ---- runner hooks
    def begin(self, task, env, meta, policy, obs) -> None:
        t0 = time.perf_counter()
        self.env, self.meta, self.task = env, meta, task
        self.ticks = int(task["episode_ticks"])
        self.steps_cap = math.ceil(self.ticks / STEP_TICS)
        if self.budget is not None:                       # after the policy's reset(), before its first act()
            fn = self._ask("looking for set_budget()", _hook, self.probe, "set_budget")
            if fn is None:
                self.warn("a time budget was given, but this policy has no set_budget(steps)")
            else:
                self._ask("set_budget()", fn, self.ticks / STEP_TICS if self.budget == "cap" else float(self.budget))
        st = env.privileged_state()
        self._start = (float(st["x"]), float(st["y"]), math.radians(float(st["angle"])))
        self.view = self._choose_view(meta, self._start)
        self._held = None
        self.odometry.reset()
        self.memory.reset()
        visible = False
        try:
            exit_obs, visible = self._exit_obs(env, meta)
            self.memory.update(self.odometry.pose, obs, None, exit_obs, 0)
        except Exception as e:  # noqa: BLE001
            self.warn(f"shadow memory failed on the first observation: {type(e).__name__}: {e}")
        try:
            totals = EpisodeScorer(env, meta).result()
        except Exception as e:  # noqa: BLE001
            totals = {}
            self.warn(f"initial totals unavailable ({type(e).__name__}: {e})")
        self._snap = self._snapshot(obs, env, totals, visible)
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.writer, self.video_note = open_video(self.out_dir, self.video_mode, self.fps, self.steps_cap)
            if self.video_note and "instead" in self.video_note:
                self.warn(self.video_note)
        except Exception as e:  # noqa: BLE001
            self.writer = None
            self.warn(f"no video: {type(e).__name__}: {e}")
        try:
            self._trace_file = open(self.out_dir / "trace.csv", "w", newline="")
            self._trace = csv.DictWriter(self._trace_file, fieldnames=TRACE_FIELDS, lineterminator="\n")
            self._trace.writeheader()
        except OSError as e:
            self._trace = None
            self.warn(f"no trace: {e}")
        self.render_s += time.perf_counter() - t0

    def _ask(self, what: str, fn, *args):
        """Call into the policy; a failure is a warning and counts as "not there"."""
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - whatever the policy raises
            self.warn(f"{what} failed: {type(e).__name__}: {e}")
            return None

    def _policy_state(self, t: int) -> dict:
        """Pose, crop, sub-goal and target for the panel of step t: the policy's where it offers them, else the shadow's."""
        acted = self.probe.calls > t and not self.probe.error
        debug = read_debug(None)
        fn = self._ask("looking for debug_state()", _hook, self.probe, "debug_state")
        if fn is not None and acted:                      # after a failure the debug state would be a stale one
            debug = self._ask("reading debug_state()", lambda: read_debug(fn())) or debug
        pose, source = None, "shadow"
        fn = self._ask("looking for pose_estimate()", _hook, self.probe, "pose_estimate")
        if fn is not None:
            pose = self._ask("pose_estimate()", lambda: _pose(fn()))
        pose = pose or debug["pose"]
        if pose is not None:
            source = "policy"
        else:
            pose = self._ask("looking for the policy's odometry", find_pose, self.probe)
            if pose is not None:
                source = "policy odometry"
            else:
                pose = _pose(self.odometry.pose) or (0.0, 0.0, 0.0)
        crop, map_source, frame_pose = debug["crop"], "policy crop", pose
        if crop is None:
            memory = self._ask("looking for the policy's memory map", find_memory, self.probe)
            if memory is not None:
                def draw():
                    out = np.asarray(memory.crop(pose))
                    if out.shape != (spec.MAP_CHANNELS, spec.CROP_SIZE, spec.CROP_SIZE):
                        raise ValueError(f"crop has shape {out.shape}")
                    return out

                crop, map_source = self._ask("drawing the policy's memory map (the shadow map is shown instead)", draw), "policy memory"
        if crop is None:                                  # the shadow map is centred on the shadow pose
            frame_pose = _pose(self.odometry.pose) or (0.0, 0.0, 0.0)
            crop = self.memory.crop(frame_pose)
            map_source = "shadow memory" + (" (exit from labels)" if self._exit_label else "")
        believed = None
        fn = self._ask("looking for exit_believed()", _hook, self.probe, "exit_believed")
        if fn is not None:
            answer = self._ask("exit_believed()", fn)
            believed = None if answer is None else bool(answer)
        return {"debug": debug, "pose": pose, "pose_source": source, "crop": crop, "map_source": map_source,
                "frame_pose": frame_pose, "exit_believed": believed, "acted": acted}

    def _held_target(self, t: int, target: tuple[int, int], own: tuple[int, int]) -> tuple:
        """(step, target, crop origin) for a crop cell that was reported without the origin of its crop.

        A manager's target is an index into the crop of the step that chose it and is then held
        while the crop moves on with the agent (PolicyNet chooses at the first step of an episode
        and then every MANAGER_PERIOD steps). Read against the current crop, a held index would
        slide along with the agent, by about one cell every two steps of walking. The viewer cannot
        see when the choice was made, so it keeps the origin of the step at which the reported
        index last changed or a manager period began, and takes the current one only then (or when
        the previous step reported no such target). `own` is the origin of the current crop.
        """
        last = self._held
        if last is not None and last[0] == t - 1 and last[1] == target and t % spec.MANAGER_PERIOD != 0:
            origin = last[2]
        else:
            origin = own
        self.warn("debug_state() gives 'target' as a crop index without 'target_origin' or 'target_cell': it is read against "
                  f"the crop of the step where the index last changed or a manager period began (every {spec.MANAGER_PERIOD} "
                  "steps) and marked '~'; this is a guess unless the choice is held on exactly that schedule")
        return t, target, origin

    def _panel_state(self, t: int, action, snap: dict) -> dict:
        ps = self._policy_state(t)
        debug, pose, true = ps["debug"], ps["pose"], snap["true"]
        self.pose_source, self.map_source = ps["pose_source"], ps["map_source"]
        f = {"step": t, "steps_cap": self.steps_cap, "tic": int(snap["totals"].get("tic", t * STEP_TICS)), "ticks": self.ticks,
             "rgb": snap["rgb"], "crop": ps["crop"], "view": self.view, "frame_pose": ps["frame_pose"], "est": pose, "true": true,
             "drift": math.hypot(pose[0] - true[0], pose[1] - true[1]), "heading_err": _wrap_deg(math.degrees(pose[2] - true[2])),
             "pose_source": ps["pose_source"], "map_source": ps["map_source"], "action": action, "masks": snap["masks"],
             "totals": snap["totals"], "subgoal": debug["subgoal"], "exit_believed": ps["exit_believed"], "note": debug["note"],
             "title": self.title, "acted": ps["acted"], "world": snap["world"],
             "error": self.probe.error or ("" if ps["acted"] else "the runner no longer calls this policy (agent error)"),
             "exit_visible": snap["exit_visible"], "target_ij": None, "target_text": None, "target_box": None}
        # However the target is reported (a point, a start-frame cell, a cell of a crop given by
        # its origin, or a cell of a crop whose origin the viewer has to infer), it becomes a
        # start-frame cell.
        own, cell, ref, held = crop_origin(pose), None, "", None
        if debug["target_xy"] is not None:
            cell = (int(math.floor(debug["target_xy"][0] / CROP_CELL + 0.5)), int(math.floor(debug["target_xy"][1] / CROP_CELL + 0.5)))
            ref = "xy"
        elif debug["target_cell"] is not None:
            cell, ref = (int(round(debug["target_cell"][0])), int(round(debug["target_cell"][1]))), "cell"
        elif debug["target"] is not None:
            base, ref = debug["target_origin"], "origin"
            if base is None:
                held = self._held_target(t, debug["target"], own)
                base, ref = held[2], "held"
            cell = (debug["target"][0] + int(round(base[0])), debug["target"][1] + int(round(base[1])))
        self._held = held                                 # forgotten as soon as the policy reports the target otherwise
        f["target_ref"] = ref
        shown = _MapFrame(ps["frame_pose"], self.view)
        if cell is not None:
            f["target_ij"] = (cell[0] - own[0], cell[1] - own[1])
            f["target_text"] = f"{'~' if ref == 'held' else ''}({f['target_ij'][0]},{f['target_ij'][1]})"
            if f["target_ij"] == (CENTER, CENTER):
                f["target_text"] += "=self"
            else:
                f["target_box"] = shown.cell_box(cell[0] - shown.origin[0], cell[1] - shown.origin[1])
        if debug["route"]:
            f["route_px"] = [shown.point(cx * CROP_CELL, cy * CROP_CELL) for cx, cy in debug["route"]]
        if debug["waypoint"] is not None:
            f["waypoint_px"] = shown.point(*debug["waypoint"])
        return f

    def _emit(self, img: Image.Image, t: int | None, repeat: int = 1) -> None:
        if t is not None:
            self.even.add(t, img)
            self.last.append((t, img))
        if self.writer is not None:
            try:
                self.writer.write(img, repeat)
                self.video_frames = self.writer.frames
            except VideoError as e:
                self.warn(f"video abandoned: {e}")
                self.writer.abort()
                self.writer = None
        self.frames += 1

    def _trace_row(self, f: dict) -> None:
        if self._trace is None:
            return
        tot, est, true = f["totals"], f["est"], f["true"]
        action = f["action"]
        target = f["target_ij"] or ("", "")
        masks = f["masks"] or ("", "")
        row = {
            "step": f["step"], "tic": f["tic"],
            "action": "".join(BUTTON_LETTERS[k] if action[k] else "." for k in range(spec.NUM_BUTTONS)) if action is not None else "",
            "acted": int(f["acted"]), "subgoal": f["subgoal"] or "", "target_i": target[0], "target_j": target[1],
            "target_ref": f.get("target_ref", ""),
            "est_x": f"{est[0]:.2f}", "est_y": f"{est[1]:.2f}", "est_heading_deg": f"{_wrap_deg(math.degrees(est[2])):.2f}",
            "true_x": f"{true[0]:.2f}", "true_y": f"{true[1]:.2f}", "true_heading_deg": f"{_wrap_deg(math.degrees(true[2])):.2f}",
            "world_x": f"{f['world'][0]:.2f}", "world_y": f"{f['world'][1]:.2f}",
            "drift": f"{f['drift']:.2f}", "heading_err_deg": f"{f['heading_err']:.3f}",
            "pose_source": f["pose_source"], "map_source": f["map_source"],
            "use_ok": "" if masks[0] == "" else int(masks[0]), "attack_ok": "" if masks[1] == "" else int(masks[1]),
            "exit_label_visible": int(f["exit_visible"]), "note": _ascii(" | ".join(x for x in (f["error"], f["note"]) if x)),
        }
        for key in ("health", "damage_taken", "kills", "keys_picked", "doors_opened", "locked_opened", "sectors_visited",
                    "cells_visited", "door_reversals"):
            value = tot.get(key)
            row[key] = "" if value is None or value != value else (int(value) if float(value) == int(value) else f"{value:.2f}")
        for key, name in (("exit_dist_final", "exit_dist"), ("stuck_rate", "stuck_rate"), ("score", "score")):
            value = tot.get(key, NAN)
            row[name] = "" if value != value else f"{value:.3f}"
        self._trace.writerow(row)

    def step(self, t, obs, action, next_obs, reward, done, env, scorer) -> None:
        t0 = time.perf_counter()
        snap = self._snap
        if snap is None:                                  # begin() failed: nothing to draw from
            self.warn("the viewer could not start; no panels were drawn")
            return
        if t == 0 and snap["totals"]:
            snap["totals"]["monsters_total"] = max(int(scorer.monsters_total), int(snap["totals"].get("kills", 0)))
        try:
            f = self._panel_state(t, action, snap)
            img = compose(f)
            self._final = f
            try:
                self._trace_row(f)
            except Exception as e:  # noqa: BLE001
                self.warn(f"trace row failed: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - one bad panel must not end the episode
            self.warn(f"panel of step {t} could not be drawn: {type(e).__name__}: {e}")
            img = fallback_panel(snap["rgb"], f"{type(e).__name__}: {e}")
        self._emit(img, t)

        # Move on to the next observation: shadow odometry and memory, then its snapshot.
        visible = False
        if not done:
            try:
                self.odometry.update(action, next_obs)
                exit_obs, visible = self._exit_obs(env, self.meta)
                self.memory.update(self.odometry.pose, next_obs, action, exit_obs, t + 1)
            except Exception as e:  # noqa: BLE001
                self.warn(f"shadow odometry or memory failed at step {t + 1}: {type(e).__name__}: {e}")
        try:
            totals = scorer.result()
        except Exception as e:  # noqa: BLE001
            totals = dict(snap["totals"])
            self.warn(f"scorer totals unavailable: {type(e).__name__}: {e}")
        try:
            self._snap = self._snapshot(obs if done else next_obs, env, totals, visible)
        except Exception as e:  # noqa: BLE001
            self._snap = {**snap, "totals": totals}
            self.warn(f"snapshot failed at step {t + 1}: {type(e).__name__}: {e}")
        self.render_s += time.perf_counter() - t0

    def end(self, row: dict):
        """Outcome frame: the last picture with the final totals and a banner, held for END_HOLD_S."""
        t0 = time.perf_counter()
        self.row = row
        try:
            f = dict(self._final) if self._final else None
            if f is not None:
                f.update(step=int(row["steps"]), tic=int(row["tic"]), totals=dict(row), action=None, masks=None)
                f["banner"] = outcome_banner(row)
                if row.get("agent_error") or row.get("hook_error"):
                    f["error"] = "; ".join(f"{kind} error at {row[key]}" for kind, key in (("agent", "agent_error"), ("hook", "hook_error"))
                                           if row.get(key))
                self._emit(compose(f), None, repeat=max(1, int(round(END_HOLD_S * self.fps))))
        except Exception as e:  # noqa: BLE001
            self.warn(f"outcome frame could not be drawn: {type(e).__name__}: {e}")
        self.render_s += time.perf_counter() - t0
        return None

    # ---- after the episode
    def header(self, what: str) -> list[str]:
        row = self.row or {}
        task = getattr(self, "task", {})
        line1 = (f"{self.title} | tick cap {task.get('episode_ticks', '-')} | {what} | "
                 f"pose: {self.pose_source}, map: {self.map_source}")
        if not row:
            return [line1, "the episode did not finish"]
        drift = row.get("odom_err_final", NAN)
        line2 = (f"result: {outcome_banner(row)[0]} after {row['steps']} steps (tic {row['tic']}), score {row['score']:.1f} | "
                 f"sectors {_ratio(row, 'sectors_visited', 'sectors_total')} keys {_ratio(row, 'keys_picked', 'keys_total')} "
                 f"doors {_ratio(row, 'doors_opened', 'doors_total')} locked {_ratio(row, 'locked_opened', 'locked_total')} "
                 f"kills {_ratio(row, 'kills', 'monsters_total')} health {row['health']:.0f} damage {row['damage_taken']:.0f} | "
                 f"stuck {row['stuck_rate']:.2f} at door {row['stuck_at_door_rate']:.2f} door reversals {row['door_reversals']}")
        if drift == drift:
            line2 += f" | policy drift final {drift:.1f} mean {row['odom_err_mean']:.1f} max {row['odom_err_max']:.1f} u"
        if row.get("agent_error"):
            line2 += f" | AGENT ERROR at {row['agent_error']}"
        if row.get("hook_error"):
            line2 += f" | hook error at {row['hook_error']}"
        return [line1, line2]

    def finish(self) -> dict:
        """Close the video and the trace and write the contact sheets. Safe to call after a failed episode."""
        out = {"video": None, "sheets": [], "trace": None}
        t0 = time.perf_counter()
        if self._trace_file is not None:
            self._trace_file.close()
            out["trace"] = str(self.out_dir / "trace.csv")
        if self.writer is not None:
            try:
                self.writer.close()
                out["video"] = str(self.writer.path)
            except Exception as e:  # noqa: BLE001
                self.warn(f"video not written: {e}")
        self.encode_close_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        steps = self.last[-1][0] + 1 if self.last else 0
        try:
            if self.even.items:
                picked = self.even.pick()
                pages = [picked[k:k + SHEET_PANELS] for k in range(0, len(picked), SHEET_PANELS)]
                for n, page in enumerate(pages):
                    name = "sheet_even.png" if len(pages) == 1 else f"sheet_even_{n + 1}.png"
                    what = f"{len(page)} of {steps} steps, sampled evenly" + (f" (sheet {n + 1} of {len(pages)})" if len(pages) > 1 else "")
                    contact_sheet([img for _, img in page], self.header(what)).save(self.out_dir / name, compress_level=3)
                    out["sheets"].append(str(self.out_dir / name))
            if self.last:
                what = f"the final {len(self.last)} of {steps} steps"
                contact_sheet([img for _, img in self.last], self.header(what)).save(self.out_dir / "sheet_final.png", compress_level=3)
                out["sheets"].append(str(self.out_dir / "sheet_final.png"))
        except Exception as e:  # noqa: BLE001
            self.warn(f"contact sheets not written: {type(e).__name__}: {e}")
        self.sheets_s = time.perf_counter() - t0
        return out


def outcome_banner(row: dict) -> tuple[str, tuple]:
    if row.get("exit"):
        return "EXIT REACHED", C_GOOD
    if row.get("dead"):
        return "DIED", C_BAD
    if row.get("timeout"):
        return "TIME OUT", C_WARN
    return "ENDED", C_TEXT


# ----------------------------------------------------------------------------------- map sources
def read_grid(source: str, no_monsters: bool = False) -> dict:
    """A grid to play, from `FILE` or `FILE:NAME`.

    FILE is a text file holding one ASCII grid (the legend of `training.mapgen.udmf`), or a
    JSON-lines suite or manifest of the map generator, or a JSON file with one such row. NAME
    picks the row of a suite by its `name` or by its line index (default: the first row).
    Returns {"grid", "options", "episode_ticks" (None if the file gives none), "tier", "name"}.
    """
    from training.mapgen import udmf

    path, _, name = str(source), "", ""
    if not Path(path).is_file() and ":" in path:
        path, _, name = path.rpartition(":")
    file = Path(path)
    if not file.is_file() and not file.is_absolute() and (paths.ROOT / file).is_file():
        file = paths.ROOT / file
    if not file.is_file():
        raise SystemExit(f"grid file not found: {path}")
    text = file.read_text()
    out = {"grid": None, "options": {}, "episode_ticks": None, "tier": 0, "name": file.stem}
    if file.suffix.lower() in (".jsonl", ".json") or text.lstrip().startswith("{"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()] if file.suffix.lower() != ".json" else [json.loads(text)]
        if not rows:
            raise SystemExit(f"{file} has no maps")
        row = rows[0]
        if name:
            by_name = [r for r in rows if str(r.get("name") or r.get("id") or "") == name]
            if by_name:
                row = by_name[0]
            elif name.isdigit() and int(name) < len(rows):
                row = rows[int(name)]
            else:
                raise SystemExit(f"{file} has no map named {name!r} (and no line with that index)")
        grid = row.get("grid")
        if isinstance(grid, list):
            grid = "\n".join(grid)
        if not grid:
            raise SystemExit(f"{file}: the chosen row has no grid")
        out.update(grid=grid, options=dict(row.get("options") or {}), tier=int(row.get("tier", 0) or 0),
                   episode_ticks=row.get("episode_ticks") or row.get("ticks"), name=str(row.get("name") or file.stem))
    else:
        if name:
            raise SystemExit(f"{file} is a plain grid file: it has no map names to choose from")
        lines = [line.rstrip() for line in text.replace("\r", "").split("\n")]
        while lines and not lines[-1]:
            lines.pop()
        while lines and not lines[0]:
            lines.pop(0)
        out["grid"] = "\n".join(lines)
    if no_monsters:
        out["grid"] = "".join("." if ch in udmf.MONSTER_CHARS else ch for ch in out["grid"])
        out["options"] = {k: v for k, v in out["options"].items() if k not in ("ambush", "thing_angles")}
    try:
        udmf.parse_grid(out["grid"])
    except ValueError as e:
        raise SystemExit(f"{file}: {e}") from None
    return out


# ------------------------------------------------------------------------------------ one episode
def render_episode(policy, wad, out, *, map_name: str = "MAP01", seed: int = 0, ticks: int = runner.DEFAULT_TICKS,
                   tier: int = 0, map_id: str | None = None, video: str = "auto", fps: float = DEFAULT_FPS, zoom="fit",
                   exit_source: str = "label", sheets: int = 1, quiet: bool = True, policy_name: str | None = None,
                   budget=None, time_scale: float | None = None) -> dict:
    """Play one episode and write the video, the contact sheets, the trace and episode.json into `out`.

    policy       a runner spec ("module:factory", optionally with JSON keyword arguments, or a
                 (target, kwargs) pair) or a ready policy object.
    wad, map_name, seed, ticks   the map, the engine seed and the tick cap.
    video        "auto" (MP4 through ffmpeg, else GIF), "mp4", "gif" or "none" (sheets only).
    fps          video frames per second; one frame is one decision step (8.75 is real time).
    zoom         "fit": the map picture is fixed on the whole map, as large as it fits; 1, 2 or 4:
                 it shows the 64, 32 or 16 cells around the agent and scrolls with it.
    exit_source  "label" writes ground-truth exit sightings into the shadow map, "none" leaves it empty.
    sheets       number of evenly sampled contact sheets (16 panels each).
    policy_name  the name written to the row and the pictures when `policy` is a ready object.
    budget       for policies with `set_budget(steps)` (the teacher): the assumed time budget in
                 decision steps, or "cap" for the true cap of this episode; None leaves the policy's own.
    time_scale   None: no time limits. A number applies the runner's official time limits to the
                 measured act() time multiplied by it (1.0 = this machine is the evaluation machine).
    Returns {"row", "video", "sheets", "trace", "json", "panels", "timing", "warnings", ...}; `row` is
    the result row of `runner.run_episode` (None if the episode could not be played).
    """
    from doom_env import make_env

    started = time.perf_counter()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("video.mp4", "video.gif", "sheet_final.png", "trace.csv", "episode.json", *(p.name for p in out_dir.glob("sheet_even*.png"))):
        (out_dir / stale).unlink(missing_ok=True)       # a rerun must not leave an older run's files behind
    task = {**runner.normalise_map({"wad": str(wad), "map": map_name, "tier": tier, "episode_ticks": ticks, "id": map_id}),
            "seed": int(seed)}
    policy_name = policy_name or runner.spec_name(policy)
    built = policy if hasattr(policy, "act") else runner.build(policy)
    probe = Probe(built)
    head, brace, rest = policy_name.partition("{")      # "module:factory{kwargs}" -> "factory{kwargs}"
    title = f"{head.rpartition(':')[2]}{brace}{rest} on {task['id']} seed {task['seed']}"
    recorder = Recorder(out_dir, probe, video=video, fps=fps, zoom=zoom, exit_source=exit_source, title=title, sheets=sheets,
                        budget=budget)
    meta = load_meta(task["wad"], task["map"])
    env = make_env(task["wad"], task["map"], episode_ticks=task["episode_ticks"], render_buffers=True)
    row, error = None, ""
    try:
        limits = {} if time_scale is None else {"time_scale": float(time_scale)}
        row = runner.run_episode(probe, env, meta, task, recorder, policy_name, **limits)
    except Exception as e:  # noqa: BLE001 - keep whatever was recorded
        error = f"{type(e).__name__}: {e}"
        recorder.warn(f"episode stopped early: {error}")
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        outputs = recorder.finish()
    total = time.perf_counter() - started
    steps = int(row["steps"]) if row else recorder.frames
    summary = {
        "policy": policy_name, "map": task, "row": row, "error": error,
        "video": outputs["video"], "video_encoder": recorder.video_note, "sheets": outputs["sheets"], "trace": outputs["trace"],
        "json": str(out_dir / "episode.json"), "panels": recorder.frames, "video_frames": recorder.video_frames, "steps": steps,
        "view": recorder.view.describe(), "view_cells": recorder.view.cells,
        "view_corner": None if recorder.view.corner is None else list(recorder.view.corner),
        "fps": float(fps),
        "pose_source": recorder.pose_source, "map_source": recorder.map_source,
        "timing": {"total_s": total, "viewer_s": recorder.render_s, "video_close_s": getattr(recorder, "encode_close_s", 0.0),
                   "sheets_s": getattr(recorder, "sheets_s", 0.0),
                   "policy_s": (row["act_ms_mean"] * steps / 1000.0) if row and row["act_ms_mean"] == row["act_ms_mean"] else 0.0,
                   "ms_per_step": 1000.0 * total / max(1, steps)},
        "warnings": list(recorder.warnings),
    }
    (out_dir / "episode.json").write_text(json.dumps(_json_safe(summary), indent=1) + "\n")
    if not quiet:
        print_summary(summary)
    return summary


# -------------------------------------------------------------------------------- many episodes
INDEX_FIELDS = ("map_id", "seed", "tier", "outcome", "steps", "tic", "episode_ticks", "score", "exit", "dead", "keys_picked",
                "keys_total", "doors_opened", "doors_total", "locked_opened", "locked_total", "sectors_visited", "sectors_total",
                "kills", "monsters_total", "health", "stuck_rate", "stuck_at_door_rate", "door_reversals", "exit_visible_tic",
                "exit_dist_final", "odom_err_max", "agent_error", "hook_error", "same_as_logged", "warnings", "dir", "sheet_final", "video")
_CONDITION = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.*?)\s*$")
_COMPARE = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def is_failure(row: dict) -> bool:
    """An episode worth looking at: the agent failed, the player died, or a map with an exit was not finished."""
    def flag(key):
        return str(row.get(key, "")).strip() not in ("", "0", "0.0", "False")

    return flag("agent_error") or flag("dead") or (flag("has_exit") and not flag("exit"))


def select_rows(rows: list[dict], failed: bool = False, where=(), limit: int | None = None) -> list[dict]:
    """Filter result rows (dicts as the runner returns them, or as read from a per-episode CSV).

    failed  keep only `is_failure` rows.
    where   conditions `KEY OP VALUE` with OP one of == != >= <= > <, all of which must hold, for
            example "door_reversals>0" or "tier==3". Values are compared as numbers when both
            sides are numbers, else as text; a row without the key (or with an empty value) fails.
    limit   keep the first N of what is left.
    """
    tests = []
    for text in where:
        m = _CONDITION.match(text)
        if not m:
            raise SystemExit(f"cannot read the condition {text!r}: write KEY OP VALUE, for example door_reversals>0")
        tests.append(m.groups())

    def holds(row) -> bool:
        for key, op, want in tests:
            have = row.get(key)
            if have is None or str(have).strip() == "":
                return False
            try:
                a, b = float(have), float(want)
            except (TypeError, ValueError):
                a, b = str(have).strip(), want.strip("'\"")
            if a != a or not _COMPARE[op](a, b):
                return False
        return True

    kept = [r for r in rows if r.get("wad") and not r.get("error") and (not failed or is_failure(r)) and holds(r)]
    return kept[:limit] if limit else kept


def read_episode_rows(path) -> list[dict]:
    """Rows of a per-episode CSV as `training.evaluate` writes them to runs/eval/<tag>.csv."""
    file = Path(path)
    if not file.is_file() and not file.is_absolute() and (paths.ROOT / file).is_file():
        file = paths.ROOT / file
    if not file.is_file():
        raise SystemExit(f"episode file not found: {path}")
    with open(file, newline="") as f:
        return list(csv.DictReader(f))


def script_from_trace(path) -> str:
    """The buttons recorded in a `trace.csv`, as a `runner.ScriptPolicy` script (one token per decision step).

    Playing that script on the same map with the same engine seed repeats the episode without the
    policy that produced it, which keeps a trajectory reproducible after the policy has changed.
    """
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"trace not found: {path}")
    with open(file, newline="") as f:
        actions = [row.get("action") or "" for row in csv.DictReader(f)]
    if not actions:
        raise SystemExit(f"{path} holds no steps")
    return " ".join(a.replace(".", "") or "N" for a in actions)


def render_rows(policy, rows: list[dict], out, *, quiet: bool = True, **options) -> list[dict]:
    """Replay the episodes of result rows and write one viewer directory per episode under `out`.

    Each row needs `wad` and may carry `map`, `seed`, `episode_ticks`, `tier` and `map_id`, as
    the rows of `runner.run_episodes` and of a per-episode CSV do. `policy` is a runner spec or a
    ready object; None takes the spec each row was logged with (its `policy` column). A policy is
    built once per spec and reused. A row with a `budget_steps` value (the teacher's gate writes
    one) has it handed to the policy's `set_budget()`, as in the run that logged it. `options`
    are passed on to `render_episode` (video, fps, zoom, ...). Also writes `out/index.csv`: one
    line per episode with its outcome, the main totals and the output paths; `same_as_logged`
    is 0 when the replay ended differently from the row it came from (another outcome or step
    count), which means the policy or the map has changed since.
    Returns the summaries of `render_episode`, each with `dir` added.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    built: dict[str, object] = {}
    summaries, used = [], set()
    for row in rows:
        entry = runner.normalise_map({"wad": row["wad"], "map": row.get("map") or "MAP01", "id": row.get("map_id") or row.get("id"),
                                      "tier": int(float(row.get("tier") or 0)),
                                      "episode_ticks": int(float(row.get("episode_ticks") or runner.DEFAULT_TICKS))})
        seed = int(float(row.get("seed") or 0))
        folder = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{entry['id']}_seed{seed}")
        while folder in used:
            folder += "_again"
        used.add(folder)
        try:
            spec_ = policy if policy is not None else str(row.get("policy") or "")
            name = runner.spec_name(spec_)
            if hasattr(spec_, "act"):
                player = spec_
            else:
                if name not in built:
                    built[name] = runner.build(spec_)
                player = built[name]
            extra = dict(options)
            if str(row.get("budget_steps") or "").strip() and "budget" not in extra:
                extra["budget"] = float(row["budget_steps"])
            summary = render_episode(player, entry["wad"], out / folder, map_name=entry["map"], seed=seed, ticks=entry["episode_ticks"],
                                     tier=entry["tier"], map_id=entry["id"], policy_name=name, **extra)
        except Exception as e:  # noqa: BLE001 - one map that cannot be played must not stop the others
            summary = {"row": None, "error": f"{type(e).__name__}: {e}", "warnings": [], "sheets": [], "video": None,
                       "map": {**entry, "seed": seed}}
        summary["dir"] = str(out / folder)
        new = summary["row"]
        same = ""
        if new and str(row.get("steps", "")).strip() != "":
            same = int(int(float(row["steps"])) == new["steps"] and int(float(row.get("exit") or 0)) == new["exit"]
                       and int(float(row.get("dead") or 0)) == new["dead"])
        summary["same_as_logged"] = same
        summaries.append(summary)
        if not quiet:
            if new:
                print(f"{entry['id']:>24} seed {seed}: {outcome_banner(new)[0]:<12} {new['steps']:4d} steps, score {new['score']:6.1f}"
                      + ("" if same != 0 else "  [differs from the logged episode]")
                      + (f"  [agent error: {new['agent_error']}]" if new["agent_error"] else "") + f"  -> {summary['dir']}", flush=True)
            else:
                print(f"{entry['id']:>24} seed {seed}: could not be played: {summary['error']}", flush=True)
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            new = summary["row"] or {}
            final = [p for p in summary["sheets"] if p.endswith("sheet_final.png")]
            line = {k: ("" if v != v else round(v, 3)) if isinstance(v, float) else v for k, v in new.items()}
            line.update(map_id=summary["map"]["id"], seed=summary["map"]["seed"], tier=summary["map"]["tier"],
                        outcome=outcome_banner(new)[0] if new else "NOT PLAYED: " + summary.get("error", ""),
                        same_as_logged=summary["same_as_logged"], warnings=" | ".join(summary["warnings"]), dir=summary["dir"],
                        sheet_final=final[0] if final else "", video=summary["video"] or "")
            writer.writerow(line)
    return summaries


def print_summary(s: dict) -> None:
    row, t = s["row"], s["timing"]
    if row:
        print(f"{outcome_banner(row)[0]} after {row['steps']} steps (tic {row['tic']} of {row['episode_ticks']}), score {row['score']:.1f}: "
              f"sectors {_ratio(row, 'sectors_visited', 'sectors_total')}, keys {_ratio(row, 'keys_picked', 'keys_total')}, "
              f"doors {_ratio(row, 'doors_opened', 'doors_total')}, locked {_ratio(row, 'locked_opened', 'locked_total')}, "
              f"kills {_ratio(row, 'kills', 'monsters_total')}, health {row['health']:.0f}, door reversals {row['door_reversals']}")
        if row["odom_err_final"] == row["odom_err_final"]:
            print(f"policy pose drift: final {row['odom_err_final']:.1f}, mean {row['odom_err_mean']:.1f}, max {row['odom_err_max']:.1f} units")
        if row["agent_error"]:
            print(f"AGENT ERROR at {row['agent_error']} (the player stood still from then on)")
        if row.get("hook_error"):
            print(f"hook error at {row['hook_error']} (that report was not asked for again)")
    else:
        print(f"the episode could not be played: {s['error']}")
    print(f"pose shown: {s['pose_source']}; map shown: {s['map_source']}; map picture: {s['view']}")
    how = f"  ({s['video_encoder'] or 'Pillow'}, {s['video_frames']} frames at {s['fps']:g} fps)" if s["video"] else ""
    print(f"video:  {s['video'] or 'none'}{how}")
    for sheet in s["sheets"]:
        print(f"sheet:  {sheet}")
    print(f"trace:  {s['trace']}\njson:   {s['json']}")
    print(f"time:   {t['total_s']:.2f} s for {s['steps']} steps ({t['ms_per_step']:.1f} ms per step): viewer {t['viewer_s']:.2f} s "
          f"(shadow map, drawing, piping frames), policy {t['policy_s']:.2f} s, video close {t['video_close_s']:.2f} s, sheets {t['sheets_s']:.2f} s")
    for w in s["warnings"]:
        print(f"WARNING: {w}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.demo", description=__doc__.split("\n\n")[0].replace("\n", " "))
    ap.add_argument("--policy", help="module:factory, optionally followed by a JSON object of keyword arguments (default: "
                                     f"{DEFAULT_POLICY}, the submission's agent; with --episodes, the policy each row was logged with)")
    ap.add_argument("--kw", action="append", default=[], metavar="KEY=VALUE", help="policy keyword argument (JSON value)")
    ap.add_argument("--map", help="WAD[:MAP] to play")
    ap.add_argument("--grid", metavar="FILE[:NAME]",
                    help="compile an ASCII grid first: a text file with one grid, or a suite/manifest .jsonl with a map name or line index")
    ap.add_argument("--no-monsters", action="store_true", help="with --grid: replace every monster by floor")
    ap.add_argument("--episodes", metavar="CSV",
                    help="replay episodes listed in a per-episode file of training.evaluate (runs/eval/TAG.csv) or of the "
                         "teacher's gate (runs/teacher/TAG.csv), one directory each")
    ap.add_argument("--failed", action="store_true",
                    help="with --episodes: only episodes with an agent error, a death, or an exit that was not reached")
    ap.add_argument("--where", action="append", default=[], metavar="'KEY OP VALUE'",
                    help="with --episodes: only rows where this holds, for example 'door_reversals>0' or 'tier==3' (repeatable)")
    ap.add_argument("--limit", type=int, default=16, help="with --episodes: replay at most this many (default %(default)s; 0 = all)")
    ap.add_argument("--replay", metavar="TRACE",
                    help="instead of --policy: press the buttons recorded in a trace.csv of an earlier run; without --map or "
                         "--grid the map, seed and tick cap are taken from the episode.json next to it")
    ap.add_argument("--out", required=True, help="output directory, for example runs/demo/NAME")
    ap.add_argument("--seed", type=int, help="engine seed (default 0)")
    ap.add_argument("--ticks", type=int, help=f"tick cap (default: the one stored with the grid, else {runner.DEFAULT_TICKS})")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="video frames per second; 8.75 is real time (default %(default)s)")
    ap.add_argument("--zoom", default="fit", choices=("fit", *(str(z) for z in ZOOMS)),
                    help="map picture: fit = fixed on the whole map, as large as it fits (default); "
                         "1, 2, 4 = the 64, 32 or 16 cells around the agent")
    ap.add_argument("--video", default="auto", choices=("auto", "mp4", "gif", "none"),
                    help="auto: MP4 through ffmpeg, or a GIF if ffmpeg cannot be used; none: contact sheets only")
    ap.add_argument("--sheets", type=int, default=1, help="number of evenly sampled contact sheets (default 1)")
    ap.add_argument("--exit-source", default="label", choices=("label", "none"),
                    help="exit channel of the shadow map: ground-truth labels (default) or nothing")
    ap.add_argument("--time-limits", action="store_true",
                    help="apply the evaluation's time limits to act() (see training.runner); off by default")
    ap.add_argument("--budget", metavar="STEPS|cap",
                    help="for policies with set_budget(steps), such as the teacher: the assumed time budget in decision steps, "
                         "or 'cap' for the true cap (with --episodes the default is each row's budget_steps, if it has one)")
    args = ap.parse_args(argv)
    sources = sum(bool(x) for x in (args.map, args.grid, args.episodes))
    if sources > 1 or (sources == 0 and not args.replay) or (args.replay and args.episodes):
        ap.error("give exactly one of --map, --grid and --episodes (--replay alone replays on the map of its own run)")
    if args.no_monsters and not args.grid:
        ap.error("--no-monsters needs --grid (a WAD is played as it is)")
    if (args.failed or args.where) and not args.episodes:
        ap.error("--failed and --where need --episodes")

    # One compute thread, as in the evaluation sandbox (several jobs share this machine).
    import os

    from doom_env.limits import THREAD_ENV

    for key, value in THREAD_ENV.items():
        os.environ.setdefault(key, value)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)

    options = dict(video=args.video, fps=args.fps, zoom=args.zoom if args.zoom == "fit" else int(args.zoom),
                   exit_source=args.exit_source, sheets=args.sheets)
    if args.time_limits:
        options["time_scale"] = 1.0
    if args.budget:
        try:
            options["budget"] = "cap" if args.budget == "cap" else float(args.budget)
        except ValueError:
            ap.error("--budget takes a number of decision steps or 'cap'")
    out_dir = Path(args.out)
    seed, ticks, tier, map_id = args.seed, args.ticks, 0, None
    if args.replay:
        policy = runner.ScriptPolicy(script_from_trace(args.replay))
        options["policy_name"] = f"replay of {args.replay}"
        if not (args.map or args.grid):                   # the map, seed and tick cap of the run that wrote the trace
            try:
                logged = json.loads((Path(args.replay).parent / "episode.json").read_text())["map"]
                args.map = f"{logged['wad']}:{logged['map']}"
            except (OSError, ValueError, KeyError, TypeError):
                ap.error(f"no usable episode.json next to {args.replay}: give the map with --map or --grid")
            seed = logged.get("seed", 0) if seed is None else seed
            ticks, tier, map_id = ticks or logged.get("episode_ticks"), int(logged.get("tier") or 0), logged.get("id")
    elif args.policy or not args.episodes:
        target, kwargs = runner.parse_spec(args.policy or DEFAULT_POLICY)
        for item in args.kw:
            key, _, value = item.partition("=")
            try:
                kwargs[key] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key] = value
        policy = (target, kwargs) if kwargs else target
    else:
        policy = None                                     # --episodes without --policy: the one each row was logged with
        if args.kw:
            ap.error("--kw needs --policy")
    seed = int(seed or 0)

    if args.episodes:
        rows = read_episode_rows(args.episodes)
        chosen = select_rows(rows, args.failed, args.where, args.limit)
        print(f"{len(chosen)} of {len(rows)} episodes of {args.episodes} to replay")
        if not chosen:
            return 0
        if policy is None and not all(":" in str(r.get("policy") or "") for r in chosen):
            ap.error(f"{args.episodes} does not say which policy played its episodes: give it with --policy")
        started = time.perf_counter()
        summaries = render_rows(policy, chosen, out_dir, quiet=False, **options)
        played = [s for s in summaries if s["row"] is not None]
        print(f"{len(played)} episodes written under {out_dir} in {time.perf_counter() - started:.1f} s; index: {out_dir / 'index.csv'}")
        return 0 if len(played) == len(summaries) else 1

    if args.grid:
        from training.mapgen import udmf

        grid = read_grid(args.grid, args.no_monsters)
        wad, map_name = udmf.write_wad(grid["grid"], out_dir / "map.wad", **grid["options"]), "MAP01"
        ticks = ticks or grid["episode_ticks"]
        tier, map_id = grid["tier"], grid["name"]
        print(f"compiled {args.grid} to {wad}")
    else:
        entry = runner.normalise_map(args.map)
        wad, map_name = entry["wad"], entry["map"]
        if not Path(wad).is_file():
            ap.error(f"WAD not found: {wad}")
    try:
        summary = render_episode(policy, wad, out_dir, map_name=map_name, seed=seed, ticks=int(ticks or runner.DEFAULT_TICKS),
                                 tier=tier, map_id=map_id, quiet=False, **options)
    except MapTooLarge as e:
        print(f"cannot play this map: {e}")
        return 1
    return 0 if summary["row"] is not None and (summary["video"] or args.video == "none") and summary["sheets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
