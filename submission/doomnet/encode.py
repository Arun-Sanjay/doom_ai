"""Encoder: one observation -> the fixed-size arrays the networks take.

Every value is a rescaled copy of an observation field, of the agent's own previous action or of
its own pose estimate. Nothing here is learned and nothing here chooses an action.

rays (64, 7)   distance / 1024, then the hit type one-hot (none, wall, door, locked door, enemy, item).
ents (32, 17)  distance / 1024 (at most 4), sin and cos of the bearing, relative velocity forward and
               right / 10, then the entity type one-hot. Padding rows are all zero.
ent_mask (32,) True where a row holds a real entity.
scal (44,)     the fields of `spec.SCALAR_FIELDS`, in that order.

The only state is the USE timer: the number of decision steps since USE last went from released
to pressed (0 on the observation that follows the press), which a door needs because a second
press while it is still moving reverses it.

Purpose: turn the wrapper's observation dict, the pose estimate and the previous action into network inputs.
Shapes: obs dict, pose (x, y, theta), prev_action (9,), exit_obs (3,) or None -> rays (64, 7) f32, ents (32, 17) f32, ent_mask (32,) bool, scal (44,) f32.
Why: fixed scales and one-hot types keep every input of order 1 and finite whatever the engine reports, so the networks never see raw units or out-of-range values.
"""

from __future__ import annotations

import math

import numpy as np

from . import spec

USE_AGE_STEPS = 16.0          # the USE timer saturates after this many decision steps
VELOCITY_SCALE = 0.1          # units per tic -> network units
MAX_ENTITY_DISTANCE = 4.0     # entity distances are clipped here (4096 units)
CLIP = 4.0                    # bound on every rescaled scalar that has no natural bound
CROP_CELL = spec.MAP_RES * spec.CROP_MULT

_I = spec.SCALAR_INDEX
_RAY_ONE_HOT = np.eye(spec.NUM_HIT_TYPES, dtype=np.float32)
_ENT_ONE_HOT = np.eye(spec.NUM_ENTITY_TYPES, dtype=np.float32)
# vitals -> scalars: one multiply and one clip for the whole vector.
_VITAL_SCALE = np.ones(spec.NUM_VITALS, np.float32)
_VITAL_SCALE[0:2] = 1.0 / 100.0                     # health, armor
_VITAL_SCALE[3:8] = 1.0 / 50.0                      # selected ammo and the four ammo pools
_VITAL_SCALE[14:17] = VELOCITY_SCALE                # velocity forward, right, up
_VITAL_LOW = np.zeros(spec.NUM_VITALS, np.float32)
_VITAL_LOW[14:17] = -CLIP
_VITAL_HIGH = np.full(spec.NUM_VITALS, CLIP, np.float32)
_VITAL_HIGH[8:14] = 1.0                             # key flags
_ENT_SCALE = np.array([1.0, 1.0, 1.0, VELOCITY_SCALE, VELOCITY_SCALE], np.float32)
_ENT_LOW = np.array([0.0, -1.0, -1.0, -CLIP, -CLIP], np.float32)
_ENT_HIGH = np.array([MAX_ENTITY_DISTANCE, 1.0, 1.0, CLIP, CLIP], np.float32)


class Encoder:
    """Observation -> network inputs. `reset()` before every map, `encode()` once per observation."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._use_age = USE_AGE_STEPS        # decision steps since USE last went down (saturated: never)
        self._use_down = False               # USE state in the previous action

    def encode(self, obs, pose, prev_action, step, budget_steps, exit_obs) -> dict:
        """Encode one observation.

        pose: (x, y, theta) in the start frame. prev_action: the (9,) buttons that produced `obs`
        (None for the first observation). step: decision step index. budget_steps: the assumed
        time budget in decision steps. exit_obs: None or (confidence, forward / 1024, left / 1024).
        """
        # ---- rays --------------------------------------------------------------------------------
        ray_obs = obs["rays"]
        rays = np.empty((spec.NUM_RAYS, spec.RAY_FEATS), np.float32)
        rays[:, 0] = ray_obs["distance"]
        rays[:, 1:] = _RAY_ONE_HOT[np.clip(ray_obs["hit_type"], 0, spec.NUM_HIT_TYPES - 1)]
        np.nan_to_num(rays[:, 0], copy=False, nan=1.0, posinf=1.0, neginf=0.0)
        np.clip(rays[:, 0], 0.0, 1.0, out=rays[:, 0])

        # ---- entities ----------------------------------------------------------------------------
        ent_obs = obs["visible_entities"]
        ent_mask = np.asarray(ent_obs["mask"]) != 0
        ents = np.zeros((spec.MAX_ENTITIES, spec.ENT_FEATS), np.float32)
        if ent_mask.any():
            feats = np.nan_to_num(np.asarray(ent_obs["features"], np.float32), nan=0.0, posinf=CLIP / VELOCITY_SCALE,
                                  neginf=-CLIP / VELOCITY_SCALE)
            feats *= _ENT_SCALE
            np.clip(feats, _ENT_LOW, _ENT_HIGH, out=ents[:, :5])
            ents[:, 5:] = _ENT_ONE_HOT[np.clip(ent_obs["entity_type"], 0, spec.NUM_ENTITY_TYPES - 1)]
            ents[~ent_mask] = 0.0

        # ---- scalars -----------------------------------------------------------------------------
        scal = np.zeros(spec.SCALAR_DIM, np.float32)
        v = np.nan_to_num(np.asarray(obs["vitals"], np.float32), nan=0.0, posinf=CLIP * 100.0, neginf=-CLIP * 100.0)
        slot = int(min(max(float(v[2]), 0.0), 7.0))
        v *= _VITAL_SCALE
        np.clip(v, _VITAL_LOW, _VITAL_HIGH, out=v)
        scal[_I["health"]:_I["health"] + 2] = v[0:2]
        scal[_I["weapon_0"] + slot] = 1.0
        scal[_I["sel_ammo"]:_I["sel_ammo"] + 5] = v[3:8]
        scal[_I["key_blue_card"]:_I["key_blue_card"] + 6] = v[8:14]
        scal[_I["vel_forward"]:_I["vel_forward"] + 3] = v[14:17]

        step, budget = float(step), float(budget_steps)
        step = step if math.isfinite(step) else 0.0
        budget = budget if math.isfinite(budget) else float(spec.DEFAULT_BUDGET_STEPS)
        scal[_I["step_frac"]] = min(max(step / spec.STEP_NORM, 0.0), CLIP)
        scal[_I["budget_frac"]] = min(max(budget / spec.STEP_NORM, 0.0), CLIP)
        scal[_I["budget_left"]] = min(max((budget - step) / spec.STEP_NORM, -1.0), 1.0)

        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        if math.isfinite(theta):
            scal[_I["sin_theta"]], scal[_I["cos_theta"]] = math.sin(theta), math.cos(theta)
        else:
            scal[_I["cos_theta"]] = 1.0
        if math.isfinite(x) and math.isfinite(y):
            # Offset inside the 64-unit crop cell, in [-0.5, 0.5): 0 at the cell centre.
            scal[_I["subcell_x"]] = (x + CROP_CELL / 2) % CROP_CELL / CROP_CELL - 0.5
            scal[_I["subcell_y"]] = (y + CROP_CELL / 2) % CROP_CELL / CROP_CELL - 0.5

        use_down = False
        if prev_action is not None:
            prev = np.asarray(prev_action, np.float32)[:spec.NUM_BUTTONS] != 0
            scal[_I["prev_fwd"]:_I["prev_fwd"] + len(prev)] = prev
            use_down = bool(prev[7]) if len(prev) > 7 else False
        if use_down and not self._use_down:
            self._use_age = 0.0
        else:
            self._use_age = min(self._use_age + 1.0, USE_AGE_STEPS)
        self._use_down = use_down
        scal[_I["use_age"]] = self._use_age / USE_AGE_STEPS

        if exit_obs is not None:
            conf, fwd, left = float(exit_obs[0]), float(exit_obs[1]), float(exit_obs[2])
            if conf >= 0.5 and math.isfinite(fwd) and math.isfinite(left):
                scal[_I["exit_visible"]] = 1.0
                scal[_I["exit_x"]] = min(max(fwd, -CLIP), CLIP)
                scal[_I["exit_y"]] = min(max(left, -CLIP), CLIP)
        return {"rays": rays, "ents": ents, "ent_mask": ent_mask, "scal": scal}
