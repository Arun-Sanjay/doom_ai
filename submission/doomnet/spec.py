"""Shared constants: frames, observation encoding, memory-map layout, action heads and action masks.

Frames
------
* ego frame: x = forward, y = left, angles counter-clockwise positive (radians). This matches the
  wrapper: ray k points at angle k * 2*pi/64, entity rel_angle is positive to the left.
* start frame: the ego frame at the first observation of the episode. Pose = (x, y, theta) in the
  start frame, integrated from the agent's own actions and reset before every map. The agent never
  sees map coordinates from the engine.
"""

from __future__ import annotations

import numpy as np

# ---- wrapper contract (mirrors doom_env.spaces; duplicated so submission/ does not import it) ----
FRAME_SKIP = 4
NUM_RAYS = 64
RAY_RANGE = 1024.0
NUM_HIT_TYPES = 6          # none, wall, door, locked_door, enemy, item
HIT_NONE, HIT_WALL, HIT_DOOR, HIT_LOCKED, HIT_ENEMY, HIT_ITEM = range(6)
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12      # pad, enemy_melee, enemy_hitscan, enemy_projectile, projectile, health, armor, ammo, weapon, key, barrel, other
(ENT_PAD, ENT_MELEE, ENT_HITSCAN, ENT_PROJ_ENEMY, ENT_PROJECTILE, ENT_HEALTH, ENT_ARMOR, ENT_AMMO,
 ENT_WEAPON, ENT_KEY, ENT_BARREL, ENT_OTHER) = range(12)
NUM_VITALS = 17
SCREEN_H, SCREEN_W = 120, 160

# ---- spatial memory -----------------------------------------------------------------------------
MAP_RES = 32.0             # map units per stored cell
MAP_SIZE = 256             # stored cells per side (8192 units); the start pose sits at the centre
# Stored cell index along an axis: i = floor(coord / MAP_RES) + MAP_SIZE // 2. Writes outside the
# array are dropped and crops that reach past the edge are zero-padded: no index may ever raise.
(CH_FREE,       # observed traversable (a ray passed through)
 CH_WALL,       # a ray ended on a wall here
 CH_DOOR,       # closed ordinary door last seen here
 CH_LOCKED,     # locked door last seen here
 CH_VISITED,    # the agent has stood here
 CH_RECENCY,    # 1 right after a visit, decaying to 0 (breadcrumb trail)
 CH_ENEMY,      # enemy last seen here (decays)
 CH_ITEM,       # health / armor / ammo / weapon seen here
 CH_KEY,        # key seen here
 CH_EXIT,       # exit pad believed here (written from the exit detector's output)
 CH_USED,       # USE was pressed facing this cell (decays); tells tried doors from untried ones
 ) = range(11)
MAP_CHANNELS = 11
# Network view of the memory: allocentric (axes of the start frame, not rotated with the agent),
# centred on the agent's cell, CROP_MULT x CROP_MULT stored cells pooled into one crop cell.
CROP_MULT = 2              # 64 units per crop cell
CROP_SIZE = 64             # crop cells per side (4096 units across)
# Pooling groups are aligned so a crop cell spans [64k - 32, 64k + 32) in the start frame.

# ---- hierarchy ----------------------------------------------------------------------------------
# The manager head chooses one sub-goal and points at a target cell of the crop. Nothing in the
# code reads the choice: it reaches the controller only as an embedding and as the target of the
# value-propagation block.
SUBGOALS = ("explore", "sweep", "get_key", "open_door", "open_locked", "go_exit", "fight", "get_pickup")
NUM_SUBGOALS = len(SUBGOALS)
SUBGOAL_INDEX = {n: i for i, n in enumerate(SUBGOALS)}
MANAGER_PERIOD = 8         # decision steps between manager updates

# ---- network inputs (one decision step, no batch dimension) ---------------------------------------
RAY_FEATS = 1 + NUM_HIT_TYPES                 # distance, one-hot hit type            -> (64, 7)
ENT_FEATS = 5 + NUM_ENTITY_TYPES              # 5 geometric features, one-hot type    -> (32, 17)
SCALAR_FIELDS = (
    "health", "armor",                                           # / 100
    "weapon_0", "weapon_1", "weapon_2", "weapon_3", "weapon_4", "weapon_5", "weapon_6", "weapon_7",  # one-hot selected weapon slot
    "sel_ammo", "ammo_bullets", "ammo_shells", "ammo_rockets", "ammo_cells",   # / 50, clipped to [0, 4]
    "key_blue_card", "key_yellow_card", "key_red_card", "key_blue_skull", "key_yellow_skull", "key_red_skull",
    "vel_forward", "vel_right", "vel_up",                        # / 10
    "step_frac",                                                 # decision step / 1050
    "budget_frac",                                               # assumed time budget (decision steps) / 1050
    "budget_left",                                               # (budget - step) / 1050, clipped to [-1, 1]
    "sin_theta", "cos_theta",                                    # heading in the start frame
    "subcell_x", "subcell_y",                                    # offset inside the current 64-unit crop cell, in [-0.5, 0.5)
    "prev_fwd", "prev_back", "prev_left", "prev_right", "prev_tl", "prev_tr", "prev_attack", "prev_use", "prev_weapon",
    "use_age",                                                   # decision steps since USE last went down / 16, clipped to [0, 1]
    "exit_visible", "exit_x", "exit_y",                          # exit observation this step: flag, forward / 1024, left / 1024
)
SCALAR_DIM = len(SCALAR_FIELDS)               # 44
SCALAR_INDEX = {n: i for i, n in enumerate(SCALAR_FIELDS)}
STEP_NORM = 1050.0
# The tick cap of a map (2100-4200 tics) is not observable. The policy is conditioned on an assumed
# budget: during training it is the true cap with noise, at evaluation the shortest possible cap.
DEFAULT_BUDGET_STEPS = 525

# ---- actions ----------------------------------------------------------------------------------------
# Wrapper buttons: 0 fwd, 1 back, 2 strafe-left, 3 strafe-right, 4 turn-left, 5 turn-right, 6 attack, 7 use, 8 next weapon
NUM_BUTTONS = 9
HEAD_NAMES = ("move", "strafe", "turn", "attack", "use", "weapon")
HEAD_SIZES = (3, 3, 3, 2, 2, 2)   # move: none/fwd/back; strafe: none/left/right; turn: none/left/right; others: off/on
HEAD_MOVE, HEAD_STRAFE, HEAD_TURN, HEAD_ATTACK, HEAD_USE, HEAD_WEAPON = range(6)

_PAIR = np.array([[0, 0], [1, 0], [0, 1]], np.int8)      # head index -> (positive button, negative button)
_PAIR_INV = np.array([[0, 2], [1, 0]], np.int64)         # (positive, negative) -> head index; both pressed cancel


def heads_to_buttons(heads) -> np.ndarray:
    """(6,) head indices -> (9,) int8 button array."""
    h = np.asarray(heads).astype(np.int64)
    b = np.zeros(NUM_BUTTONS, np.int8)
    b[0:2], b[2:4], b[4:6] = _PAIR[h[0]], _PAIR[h[1]], _PAIR[h[2]]
    b[6:9] = h[3:6]
    return b


def buttons_to_heads(buttons) -> np.ndarray:
    """(9,) buttons -> (6,) head indices."""
    b = np.asarray(buttons).astype(np.int64)
    return np.array([_PAIR_INV[b[0], b[1]], _PAIR_INV[b[2], b[3]], _PAIR_INV[b[4], b[5]], b[6], b[7], b[8]], np.int64)


# ---- action masks from observation fields (docs/RULES.md section 1) ---------------------------------
USE_REACH = 64.0           # the engine's USE trace goes this far, straight ahead
MELEE_WEAPON_SLOT = 1      # fist and chainsaw share weapon slot 1


def action_masks(ray_distance, ray_hit_type, entity_type, entity_mask, selected_weapon=None) -> tuple[bool, bool]:
    """Which of USE and ATTACK can have any effect right now.

    use_ok: the ray straight ahead ends on a door or locked door within USE reach.
    attack_ok: at least one enemy is currently visible and the selected weapon (vitals[2]) is not
    the melee slot. A melee hit turns the player towards its target without any turn key, which
    would silently break the pose estimate that the spatial memory is built on.
    A mask removes an option that would do nothing; the network chooses among what is left.
    The same function is applied to the logits in training and at inference.
    """
    use_ok = bool((ray_hit_type[0] == HIT_DOOR or ray_hit_type[0] == HIT_LOCKED)
                  and float(ray_distance[0]) * RAY_RANGE <= USE_REACH + 2.0)
    t = np.asarray(entity_type)[np.asarray(entity_mask).astype(bool)]
    attack_ok = bool(((t >= ENT_MELEE) & (t <= ENT_PROJ_ENEMY)).any())
    if selected_weapon is not None and int(round(float(selected_weapon))) == MELEE_WEAPON_SLOT:
        attack_ok = False
    return use_ok, attack_ok
