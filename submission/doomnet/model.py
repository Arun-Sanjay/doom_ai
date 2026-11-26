"""Neural networks of the agent: a three-level policy and the exit detector.

PolicyNet maps one decision step of encoded observations to button logits through three learned
levels:

1. Manager (every MANAGER_PERIOD steps and at episode start). A convolutional tower reads the
   allocentric memory-map crop. From the tower, the fused sensor features and the GRU state the
   manager chooses a sub-goal (a categorical over spec.SUBGOALS) and then a target cell (a pointer:
   one softmax over all 64 x 64 crop cells, conditioned on the chosen sub-goal).
2. Value propagation (same steps). One small convolutional cell is applied K times on a 32 x 32
   version of the crop, in the style of value iteration networks: a Bellman backup with learned
   rewards, transitions and discounts, V <- max_a (r_a + gamma_a * (P_a * V)). It is seeded with
   the chosen target cell and conditioned on the sub-goal embedding, and it outputs a value field
   over the crop.
3. Controller (every step). A circular convolution over the 64 rays, attention over the visible
   entities, an MLP over the scalars, a GRU, and six factored action heads.

Goal bottleneck. With strict_goal_bottleneck=True the controller receives exactly two things from
the levels above it: the embedding of the sub-goal in force, and a 9 x 9 patch of the value field
read around its own position and rotated into its own frame (as differences to the value at its
own cell, times a fixed gain). It never sees the map, the map features or the target index. Rays,
entities and scalars reach it directly, so fighting and local movement do not depend on the
manager. No code reads the sub-goal or the target: clamping the sub-goal (clamp_subgoal) changes
behaviour only because the learned controller is conditioned on it.

Between manager steps the manager's output is held in the state. The crop is re-centred on the
agent every step while the held value field stays in the frame of the step that produced it, so
the controller keeps count of the whole cells it has crossed since then (from the sub-cell offset
and the velocity scalars) and reads the field at its current position.

ExitDetector is a separate small CNN that finds the exit pad in the RGB frame.

Array conventions used here (the only place they matter is EgoPatch and the cell counter):
map[b, c, i, j] has i along the start frame's x axis (forward at the start) and j along its y axis
(left at the start), both increasing with the coordinate; the agent's cell is index CROP_SIZE // 2
on both axes. A target index is i * CROP_SIZE + j.

Purpose: PolicyNet (manager -> value propagation -> controller) and ExitDetector, CPU inference at batch 1 and batched learning on any device.
Tensor shapes: rays (B,64,7), ents (B,32,17), ent_mask (B,32), scal (B,44), map (B,11,64,64) -> 6 logit tensors sized HEAD_SIZES, value (B,), value_field (B,1,64,64); rgb (B,3,120,160) -> (B,3).
Why this design: the costly map tower and value propagation run once per 8 steps and are cached, the per-step controller is a few dozen small operators, and the goal reaches the controller through one narrow, inspectable channel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import spec

CELLS = spec.CROP_SIZE * spec.CROP_SIZE          # 4096 pointer targets
CENTER = spec.CROP_SIZE // 2                     # crop index of the agent's own cell
PATCH_RADIUS = 4                                 # the controller reads the value field +-4 cells around itself
LOCAL_RADIUS = 7                                 # raw-map window that contains every rotation of that patch
GOAL_GAIN = 5.0                                  # fixed scale of that patch: about unit size for a freshly initialised field
NEG_INF = float("-inf")

_S = spec.SCALAR_INDEX
_VEL_F, _VEL_R = _S["vel_forward"], _S["vel_right"]
_SIN, _COS = _S["sin_theta"], _S["cos_theta"]
_SUB = slice(_S["subcell_x"], _S["subcell_y"] + 1)
# velocity scalar (map units per tic / 10) -> crop cells moved in one decision step
_VEL_TO_CELLS = 10.0 * spec.FRAME_SKIP / (spec.MAP_RES * spec.CROP_MULT)
# Cap on the move expected from the velocity, in cells. A step covers at most about 0.75 of a cell;
# with the cap every such move stays within half a cell of the expectation, and a velocity that
# over-predicts (the agent pressed against a wall) cannot produce a crossing that did not happen.
_MAX_EXPECTED_MOVE = 0.35


def _finite(x: torch.Tensor, bound: float) -> torch.Tensor:
    """Replace NaN by 0 and limit magnitudes, so one bad observation cannot poison the GRU state."""
    return torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)


def _as_unit_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 images and map crops are scaled to [0, 1]; float inputs are taken as already scaled."""
    return x if x.is_floating_point() else x.float() * (1.0 / 255.0)


def _put(old: torch.Tensor, rows: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    """Copy of `old` with the given batch rows replaced by `new` (rows=None means every row)."""
    return new if rows is None else old.index_copy(0, rows, new.to(old.dtype))


def max_pool_2x2(x: torch.Tensor) -> torch.Tensor:
    """2 x 2 max pooling written with slices (same result as F.max_pool2d, several times faster on one CPU thread)."""
    top = torch.maximum(x[..., 0::2, 0::2], x[..., 0::2, 1::2])
    return torch.maximum(top, torch.maximum(x[..., 1::2, 0::2], x[..., 1::2, 1::2]))


class EgoPatch(nn.Module):
    """Fixed, parameter-free and differentiable: read a P x P patch of an allocentric field in the agent's frame.

    Sample (r, c) lies `fwd[r]` cells ahead of the agent and `left[c]` cells to its left; rows run
    from ahead to behind and columns from left to right, like a top-down view with the agent facing
    up. The rotation uses sin(theta) and cos(theta) directly; no angle is ever computed.
    """

    def __init__(self, radius: int = PATCH_RADIUS):
        super().__init__()
        self.size = 2 * radius + 1
        steps = torch.arange(radius, -radius - 1, -1, dtype=torch.float32)
        fwd, left = torch.meshgrid(steps, steps, indexing="ij")
        self.register_buffer("fwd", fwd.reshape(1, -1), persistent=False)       # (1, P*P)
        self.register_buffer("left", left.reshape(1, -1), persistent=False)

    def forward(self, field: torch.Tensor, pos: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """field (N,C,S,S); pos (N,2) agent position in field index units (x, y); sin, cos (N,1) -> (N,C,P,P)."""
        x = pos[:, 0:1] + cos * self.fwd - sin * self.left                      # (N, P*P) index along the x axis
        y = pos[:, 1:2] + sin * self.fwd + cos * self.left
        # grid_sample addresses (last dim, second-to-last dim) = (y, x) and wants coordinates in [-1, 1]
        grid = torch.stack([y, x], -1) * (2.0 / (field.shape[-1] - 1)) - 1.0
        grid = grid.view(-1, self.size, self.size, 2).to(field.dtype)
        return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


class MapTower(nn.Module):
    """Manager's view of the memory-map crop: a three-scale convolutional pyramid with a top-down pass.

    map (M,11,64,64) -> cell features (M,32,32,32), one vector per 2 x 2 block of crop cells with
    context from the whole crop, and a summary vector (M,128). Both outputs are normalised per
    sample (group norm, layer norm; batch size does not matter): their scale then cannot drift
    while the weights change, which keeps the pointer, the value propagation and the manager's
    logits on a steady input.
    """

    def __init__(self, channels: int = 32, vec_dim: int = 128):
        super().__init__()
        c1, c2, c3 = channels, 2 * channels, 3 * channels
        self.stem = nn.Conv2d(spec.MAP_CHANNELS + 2, c1, 2, stride=2)           # 64 -> 32: one output per 2 x 2 block
        self.conv32 = nn.Conv2d(c1, c1, 3, padding=1)
        self.down16 = nn.Conv2d(c1, c2, 2, stride=2)
        self.conv16 = nn.Conv2d(c2, c2, 3, padding=1)
        self.down8 = nn.Conv2d(c2, c3, 2, stride=2)
        self.conv8 = nn.Conv2d(c3, c3, 3, padding=1)
        self.lat16 = nn.Conv2d(c3, c2, 1)                                       # top-down: 8 -> 16
        self.lat32 = nn.Conv2d(c2, c1, 1)                                       # top-down: 16 -> 32
        self.out32 = nn.Conv2d(c1, c1, 3, padding=1)
        self.norm32 = nn.GroupNorm(4, c1)                                       # steady feature scale for pointer and value propagation
        self.vec_conv = nn.Conv2d(c3, 16, 1)
        self.vec_fc = nn.Linear(16 * 8 * 8, vec_dim)
        self.vec_norm = nn.LayerNorm(vec_dim)
        # offset of every crop cell from the agent's cell, in units of half a crop: the tower knows what is near
        offs = (torch.arange(spec.CROP_SIZE, dtype=torch.float32) - CENTER) / CENTER
        ii, jj = torch.meshgrid(offs, offs, indexing="ij")
        self.register_buffer("coords", torch.stack([ii, jj])[None], persistent=False)   # (1,2,64,64)

    def forward(self, map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([map, self.coords.expand(map.shape[0], -1, -1, -1).to(map.dtype)], 1)
        f32 = torch.relu(self.conv32(torch.relu(self.stem(x))))
        f16 = torch.relu(self.conv16(torch.relu(self.down16(f32))))
        f8 = torch.relu(self.conv8(torch.relu(self.down8(f16))))
        f16 = f16 + F.interpolate(self.lat16(f8), scale_factor=2.0, mode="nearest")
        f32 = torch.relu(self.norm32(self.out32(f32 + F.interpolate(self.lat32(f16), scale_factor=2.0, mode="nearest"))))
        vec = torch.relu(self.vec_norm(self.vec_fc(torch.relu(self.vec_conv(f8)).flatten(1))))
        return f32, vec


class ValuePropagation(nn.Module):
    """Learned value propagation: one small convolutional cell applied K times, as in value iteration networks.

    The cell is a Bellman backup whose parts are all learned:

        V <- max over moves a of ( r_a + gamma_a * (P_a * V) )

    V     `channels` value maps on a 32 x 32 grid (128 map units per grid cell).
    r_a   "reward" maps, computed once per call: a convolution over the tower's cell features, the
          target seed (a one-hot of the chosen target cell) and the offset of every grid cell from
          the target, plus a projection of the sub-goal embedding.
    P_a   3 x 3 kernels over all value channels, one per move and value channel, normalised with
          a softmax so that each is a probability distribution: a learned transition.
    gamma_a  learned discounts, kept inside (0, 1) by a sigmoid.

    Since every P_a is a distribution and every gamma_a is below one, one application is a
    contraction: K applications cannot blow up whatever the weights are, and the values stay within
    |r| / (1 - gamma). The max keeps the best continuation instead of an average, so a value can
    travel one grid cell per application without being blurred away. Nothing here encodes which
    cells are free or how to walk: where values may flow is decided by the learned r_a and P_a.
    The read-out restores crop resolution: four values per grid cell, one per crop cell of its
    2 x 2 block.
    """

    SEED_CHANNELS = 4          # one-hot of the target cell, one channel per position inside a 2 x 2 block
    OFFSET_CHANNELS = 2        # offset of each grid cell from the target cell, along both crop axes
    # Initial values only; every one of these quantities is learned afterwards.
    FEATURE_GAIN = 0.3         # gain of the map features inside r_a (the target terms start at gain 1)
    TAP_STD = 2.0              # spread of the transition logits: every move starts on a few random taps
    OTHER_CHANNEL_LOGIT = -3.0 # a move starts mostly inside its own value channel
    STAY_LOGIT = 2.0           # move 0 of every channel starts with a preference for the cell's own value
    DISCOUNT = 0.9             # gamma

    def __init__(self, in_channels: int = 32, emb_dim: int = 32, channels: int = 8, moves: int = 4, iters: int = 12):
        super().__init__()
        self.in_channels, self.channels, self.moves, self.iters = in_channels, channels, moves, iters
        self.inject = nn.Conv2d(in_channels + self.SEED_CHANNELS + self.OFFSET_CHANNELS, channels * moves, 3, padding=1)
        self.goal = nn.Linear(emb_dim, channels * moves)
        self.transition = nn.Parameter(torch.zeros(channels * moves, channels * 9))      # logits of P_a
        self.discount = nn.Parameter(torch.zeros(channels * moves))                      # logits of gamma_a
        self.read = nn.Conv2d(channels, 4, 1)
        centres = torch.arange(spec.CROP_SIZE // 2, dtype=torch.float32) * 2.0 + 0.5   # crop index of each grid-cell centre
        ii, jj = torch.meshgrid(centres, centres, indexing="ij")
        self.register_buffer("centres", torch.stack([ii, jj])[None], persistent=False)  # (1,2,32,32)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start of learning: random transport, no knowledge of the map.

        Each move begins concentrated on a few random taps of its own channel, so values are
        carried in assorted directions from the first update on (uniform kernels would only blur
        the field). The map features enter r_a with a small gain: the first value fields are
        dominated by the target (a cone around the target cell), which lets the controller learn
        to head for the target before the map has learned to bend the field.
        """
        with torch.no_grad():
            for m in (self.inject, self.goal, self.read):
                m.bias.zero_()
            nn.init.orthogonal_(self.inject.weight[:, :self.in_channels], self.FEATURE_GAIN)
            nn.init.orthogonal_(self.inject.weight[:, self.in_channels:])
            nn.init.orthogonal_(self.goal.weight, 0.1)
            nn.init.orthogonal_(self.read.weight[:1])
            self.read.weight.copy_(self.read.weight[:1].repeat(4, 1, 1, 1))      # the 4 crop cells of a block start equal: a smooth field
            nn.init.normal_(self.transition, std=self.TAP_STD)
            own = torch.eye(self.channels).repeat(self.moves, 1).repeat_interleave(9, 1)    # 1 on a row's own channel
            self.transition += (1.0 - own) * self.OTHER_CHANNEL_LOGIT
            for c in range(self.channels):                                       # row c = move 0 of channel c; tap 4 = centre
                self.transition[c, c * 9 + 4] += self.STAY_LOGIT
            self.discount.fill_(math.log(self.DISCOUNT / (1.0 - self.DISCOUNT)))

    def kernel(self) -> torch.Tensor:
        """gamma_a * P_a as a convolution weight (moves * channels, channels, 3, 3)."""
        p = torch.softmax(self.transition, 1) * torch.sigmoid(self.discount)[:, None]
        return p.view(self.channels * self.moves, self.channels, 3, 3)

    def _best(self, q: torch.Tensor) -> torch.Tensor:
        return q.unflatten(1, (self.moves, self.channels)).amax(1)

    def forward(self, feat: torch.Tensor, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """feat (M,32,32,32), emb (M,32), target (M,) long cell index -> value field (M,1,64,64)."""
        m, size = feat.shape[0], spec.CROP_SIZE
        seed = feat.new_zeros(m, CELLS).scatter_(1, target[:, None], 1.0).view(m, 1, size, size)
        seed = F.pixel_unshuffle(seed, 2)                                        # (M,4,32,32)
        cell_ij = torch.stack([torch.div(target, size, rounding_mode="floor"), target % size], 1)
        rel = (self.centres - cell_ij.to(feat.dtype)[:, :, None, None]) * (1.0 / CENTER)     # (M,2,32,32)
        r = self.inject(torch.cat([feat, seed, rel], 1)) + self.goal(emb)[:, :, None, None]
        w = self.kernel().to(r.dtype)
        v = self._best(r)
        for _ in range(self.iters):
            v = self._best(F.conv2d(v, w, padding=1) + r)
        return F.pixel_shuffle(self.read(v), 2)


class PolicyNet(nn.Module):
    """Manager, value propagation and controller (see the module docstring).

    Arguments: hidden = GRU width; vp_iters = K, the number of applications of the value-propagation
    cell; strict_goal_bottleneck = keep the controller blind to the map except through the value
    field (False also gives it a rotated 9 x 9 read of the raw crop and the tower's summary vector).
    """

    ATTN_HEADS = 4
    ENT_DIM = 64
    EMB_DIM = 32
    VEC_DIM = 128

    def __init__(self, hidden: int = 256, vp_iters: int = 12, strict_goal_bottleneck: bool = True,
                 vp_channels: int = 8, vp_moves: int = 4):
        super().__init__()
        self.config = dict(hidden=hidden, vp_iters=vp_iters, strict_goal_bottleneck=strict_goal_bottleneck,
                           vp_channels=vp_channels, vp_moves=vp_moves)
        self.hidden, self.strict = hidden, strict_goal_bottleneck
        self.generator: torch.Generator | None = None    # optional private generator for the manager's sampling
        self._clamp: int | None = None
        self._goal_feat: torch.Tensor | None = None      # value-field features of the last forward pass (diagnostics only)

        # ---- controller, part 1: sensors -> fused features (every step)
        # One circular pad of 5, then three unpadded convolutions: the same result as circular padding
        # at every layer, with one pad operator. 16 output positions, 4 rays apart.
        self.ray_net = nn.Sequential(
            nn.Conv1d(spec.RAY_FEATS, 32, 5), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU())
        self.scal_fc = nn.Linear(spec.SCALAR_DIM, 64)
        self.ent_fc = nn.Linear(spec.ENT_FEATS, self.ENT_DIM)
        self.ent_query = nn.Linear(64, self.ATTN_HEADS * self.ENT_DIM)
        fused = 256
        self.fuse = nn.Linear(128 + self.ATTN_HEADS * self.ENT_DIM + self.ENT_DIM + 64, fused)

        # ---- manager and value propagation (manager steps only)
        self.tower = MapTower(32, self.VEC_DIM)
        self.manager_fc = nn.Linear(fused + hidden + self.VEC_DIM, 128)
        self.subgoal_head = nn.Linear(128, spec.NUM_SUBGOALS)
        self.subgoal_emb = nn.Embedding(spec.NUM_SUBGOALS, self.EMB_DIM)
        self.pointer_fc = nn.Linear(128 + self.EMB_DIM, 32)
        self.pointer_out = nn.Conv2d(32, 4, 1)
        self.vp = ValuePropagation(32, self.EMB_DIM, vp_channels, vp_moves, vp_iters)

        # ---- controller, part 2: goal read, GRU, heads (every step)
        self.ego = EgoPatch(PATCH_RADIUS)
        self.goal_fc = nn.Linear(self.ego.size ** 2, 64)
        goal_dim = self.EMB_DIM + 64
        if not self.strict:
            self.local_fc = nn.Linear(spec.MAP_CHANNELS * self.ego.size ** 2, 64)
            goal_dim += 64 + self.VEC_DIM
        self.gru = nn.GRUCell(fused + goal_dim, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden + goal_dim, 128), nn.ReLU(), nn.Linear(128, sum(spec.HEAD_SIZES)))
        critic_in = hidden + goal_dim + (self.VEC_DIM if self.strict else 0)     # the critic may always see the map summary
        self.critic = nn.Sequential(nn.Linear(critic_in, 128), nn.ReLU(), nn.Linear(128, 2))
        self.register_buffer("on_column", torch.tensor([[False, True]]), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal weights (gain sqrt 2 before ReLU), zero biases, near-uniform initial policies."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)
        for m in (self.actor[-1], self.subgoal_head, self.pointer_out):          # policy outputs start near uniform
            nn.init.orthogonal_(m.weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        self.vp.reset_parameters()                                               # the value propagation has its own start

    # ------------------------------------------------------------------------------------ state
    def initial_state(self, batch: int, device=None) -> dict:
        """State at the start of an episode. Every entry has a leading batch dimension.

        h (B,hidden) GRU state; subgoal, target (B,) long and value_field (B,1,64,64), map_vec (B,128):
        the manager output in force; cell (B,2) whole cells crossed since the manager last ran, sub
        (B,2) and move (B,2) the previous step's sub-cell offset and velocity; since (B,) long steps
        since the manager last ran (starts at MANAGER_PERIOD so the first step is a manager step).
        """
        size = spec.CROP_SIZE
        return {
            "h": torch.zeros(batch, self.hidden, device=device),
            "subgoal": torch.zeros(batch, dtype=torch.long, device=device),
            "target": torch.zeros(batch, dtype=torch.long, device=device),
            "value_field": torch.zeros(batch, 1, size, size, device=device),
            "map_vec": torch.zeros(batch, self.VEC_DIM, device=device),
            "cell": torch.zeros(batch, 2, device=device),
            "sub": torch.zeros(batch, 2, device=device),
            "move": torch.zeros(batch, 2, device=device),
            "since": torch.full((batch,), spec.MANAGER_PERIOD, dtype=torch.long, device=device),
        }

    def reset_state(self, state: dict, mask: torch.Tensor) -> dict:
        """Copy of `state` with the rows where mask (B,) is True set back to the initial state."""
        fresh = self.initial_state(mask.shape[0], mask.device)
        return {k: torch.where(mask.view(-1, *([1] * (v.dim() - 1))), fresh[k].to(v.dtype), v) for k, v in state.items()}

    def clamp_subgoal(self, index: int | None) -> None:
        """Hold the sub-goal at `index` whatever the manager chooses (None releases it). For tests and the demo."""
        if index is not None and not 0 <= int(index) < spec.NUM_SUBGOALS:
            raise ValueError(f"sub-goal index {index} out of range")
        self._clamp = None if index is None else int(index)

    def _clamped(self, subgoal: torch.Tensor) -> torch.Tensor:
        return subgoal if self._clamp is None else torch.full_like(subgoal, self._clamp)

    # ------------------------------------------------------------------------------------ level 3a: sensors
    def _perceive(self, rays, ents, ent_mask, scal) -> tuple[torch.Tensor, torch.Tensor]:
        """rays (N,64,7), ents (N,32,17), ent_mask (N,32), scal (N,44) -> fused (N,256), ent_attn (N,32)."""
        n = scal.shape[0]
        r = self.ray_net(F.pad(rays.transpose(1, 2), (5, 5), mode="circular"))   # (N,128)
        s = torch.relu(self.scal_fc(scal))                                       # (N,64)
        # Entities: a set of up to 32. Four attention heads whose queries come from the agent's own
        # scalars (health, ammunition, weapon), so what it attends to can depend on its condition.
        # A constant "nothing" slot takes the weight when no entity is visible; padding gets none.
        e = torch.relu(self.ent_fc(_finite(ents, 64.0)))                         # (N,32,64)
        pad = ~ent_mask.bool()[:, :, None]                                       # (N,32,1)
        query = self.ent_query(s).view(n, self.ATTN_HEADS, self.ENT_DIM)         # (N,4,64)
        score = torch.bmm(e, query.transpose(1, 2)) * (self.ENT_DIM ** -0.5)     # (N,32,4)
        score = torch.cat([score.masked_fill(pad, NEG_INF), score.new_zeros(n, 1, self.ATTN_HEADS)], 1)
        attn = torch.softmax(score, 1)[:, :spec.MAX_ENTITIES]                    # (N,32,4), zero on padding
        focus = torch.bmm(attn.transpose(1, 2), e).flatten(1)                    # (N,256) one weighted mean per head
        strongest = e.masked_fill(pad, 0.0).amax(1)                              # (N,64) set maximum
        fused = torch.relu(self.fuse(torch.cat([r, focus, strongest, s], 1)))
        return fused, attn.mean(2)

    # ------------------------------------------------------------------------------------ levels 1 and 2
    def _choose(self, logits: torch.Tensor, forced: torch.Tensor | None, sample: bool) -> torch.Tensor:
        if forced is not None:
            return forced.long()
        if sample:
            return torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=self.generator).squeeze(1)
        return logits.argmax(1)

    def _subgoal_logits(self, fused, h, vec) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.relu(self.manager_fc(torch.cat([fused, h, vec], 1)))         # (M,128)
        return ctx, self.subgoal_head(ctx)

    def _target_logits(self, feat, ctx, emb) -> torch.Tensor:
        """Pointer: every crop cell is scored from its own tower features, shifted by the manager's context."""
        shift = self.pointer_fc(torch.cat([ctx, emb], 1))[:, :, None, None]      # (M,32,1,1)
        return F.pixel_shuffle(self.pointer_out(torch.relu(feat + shift)), 2).flatten(1)     # (M,4096)

    def _manage(self, fused, h, map, forced_subgoal, forced_target, sample: bool) -> dict:
        """The manager and value propagation for the rows on which they run."""
        feat, vec = self.tower(_as_unit_float(map))
        ctx, subgoal_logits = self._subgoal_logits(fused, h, vec)
        subgoal = self._clamped(self._choose(subgoal_logits, forced_subgoal, sample))
        emb = self.subgoal_emb(subgoal)
        target_logits = self._target_logits(feat, ctx, emb)
        target = self._choose(target_logits, forced_target, sample)
        return {"subgoal": subgoal, "target": target, "subgoal_logits": subgoal_logits, "target_logits": target_logits,
                "value_field": self.vp(feat, emb, target), "map_vec": vec}

    # ------------------------------------------------------------------------------------ bookkeeping
    def _advance(self, scal, since, cell, sub_prev, move_prev, run_manager, reset=None):
        """Manager schedule and the count of whole cells crossed since the manager last ran.

        The sub-cell offset wraps when the agent enters the next cell. The velocity scalars say
        roughly how far it moved during the step (mean of the velocity before and after it), which
        tells a wrap from a move inside the cell:
        cells crossed = round(expected move - observed change of the offset).
        Returns run (N,) bool, since (N,) long, cell (N,2), sub (N,2), move (N,2).
        """
        if reset is not None:
            since = since.masked_fill(reset, spec.MANAGER_PERIOD)
        if run_manager is None:
            run = since >= spec.MANAGER_PERIOD
        else:                                                                    # a tensor (N,) or one bool for all rows
            run = torch.as_tensor(run_manager, dtype=torch.bool, device=since.device).expand(since.shape[0])
            run = run if reset is None else run | reset
        sub = scal[:, _SUB]
        sin, cos, fwd, left = scal[:, _SIN], scal[:, _COS], scal[:, _VEL_F], -scal[:, _VEL_R]
        move = torch.stack([fwd * cos - left * sin, fwd * sin + left * cos], 1) * _VEL_TO_CELLS   # velocity now, start frame
        expected = ((move + move_prev) * 0.5).clamp(-_MAX_EXPECTED_MOVE, _MAX_EXPECTED_MOVE)
        crossed = torch.round(expected - (sub - sub_prev))
        cell = (cell + crossed).masked_fill(run[:, None], 0.0)
        since = (since + 1).masked_fill(run, 1)
        return run, since, cell, sub, move

    # ------------------------------------------------------------------------------------ level 3b: controller
    def _goal_patch(self, value_field, pos, sin, cos) -> torch.Tensor:
        """The controller's read of the value field: value_field (N,1,64,64), pos (N,2) -> (N,81).

        The controller reads the shape of the field around itself, not its level: differences to the
        value at its own position (the middle sample), times the constant GOAL_GAIN. The read is
        linear in the field on purpose. Dividing each patch by its own size instead (a per-sample
        normalisation) makes the part of the patch that depends on the target shrink whenever
        another part of the field grows, and early in training that silenced the whole channel
        for some seeds; with a constant gain nothing else in the field can attenuate it.
        """
        patch = self.ego(value_field, pos, sin, cos).flatten(1)
        mid = patch.shape[1] // 2
        return (patch - patch[:, mid:mid + 1]) * GOAL_GAIN

    def goal_channel_stats(self) -> dict:
        """Health of the goal channel on the most recent forward pass, for training logs.

        "goal_feat_std": standard deviation of each of the 64 value-field features across the rows
        of the batch, averaged over the features. Close to zero means the controller received the
        same thing whatever the manager chose (a dead goal channel). "goal_feat_alive": fraction of
        the 64 features that were non-zero on at least one row. Both are 0.0 before any forward
        pass, and the first is 0.0 for a single row.
        """
        feat = self._goal_feat
        if feat is None or feat.shape[0] == 0:
            return {"goal_feat_std": 0.0, "goal_feat_alive": 0.0}
        spread = float(feat.float().std(0).mean()) if feat.shape[0] > 1 else 0.0
        return {"goal_feat_std": spread, "goal_feat_alive": float((feat > 0).any(0).float().mean())}

    def _goal_features(self, subgoal, value_field, cell, scal, map, map_vec) -> torch.Tensor:
        """Everything the controller receives from the manager: sub-goal embedding and value-field patch."""
        sub, sin, cos = scal[:, _SUB], scal[:, _SIN:_SIN + 1], scal[:, _COS:_COS + 1]
        feat = torch.relu(self.goal_fc(self._goal_patch(value_field, CENTER + cell + sub, sin, cos)))
        self._goal_feat = feat.detach()                                          # kept for goal_channel_stats(); costs nothing
        parts = [self.subgoal_emb(subgoal), feat]
        if not self.strict:                                                      # loosened: also a direct local map read
            lo, hi = CENTER - LOCAL_RADIUS, CENTER + LOCAL_RADIUS + 1
            window = _as_unit_float(map[:, :, lo:hi, lo:hi])
            local = self.ego(window, LOCAL_RADIUS + sub, sin, cos).flatten(1)
            parts += [torch.relu(self.local_fc(local)), map_vec]
        return torch.cat(parts, 1)

    def _heads(self, h, goal, map_vec, masks):
        x = torch.cat([h, goal], 1)
        logits = list(self.actor(x).split(spec.HEAD_SIZES, 1))
        values = self.critic(torch.cat([x, map_vec], 1) if self.strict else x)
        if masks is not None:                                                    # a False removes the "on" option
            blocked = ~masks.bool()
            logits[spec.HEAD_USE] = logits[spec.HEAD_USE].masked_fill(blocked[:, 0:1] & self.on_column, NEG_INF)
            logits[spec.HEAD_ATTACK] = logits[spec.HEAD_ATTACK].masked_fill(blocked[:, 1:2] & self.on_column, NEG_INF)
        return tuple(logits), values[:, 0], values[:, 1]

    # ------------------------------------------------------------------------------------ one step
    def forward(self, rays, ents, ent_mask, scal, map, state=None, masks=None, run_manager=None, forced=None,
                sample: bool = True):
        """One decision step for a batch.

        rays (B,64,7), ents (B,32,17), ent_mask (B,32) bool, scal (B,44), map (B,11,64,64) float in
        [0,1] or uint8 in [0,255]; state from initial_state() or the previous step.
        masks (B,2) bool = (use_ok, attack_ok): False sets the "on" logit of that head to -inf.
        run_manager (B,) bool (or one bool): rows on which the manager runs; None follows the state's
        own counter (first step of an episode, then every MANAGER_PERIOD steps).
        forced {"subgoal": (B,), "target": (B,)}: use these manager choices instead of choosing
        (either key may be omitted); otherwise the choice is sampled (sample=True) or most likely.
        Returns (out, new_state). out: "logits" tuple of six (B, n) tensors, "value", "value_int" (B,),
        "subgoal", "target" (B,) long in force this step, "subgoal_logits" (B,8) and "target_logits"
        (B,4096) (zeros on rows where the manager did not run), "value_field" (B,1,64,64) in the
        frame of the manager step that produced it, "ent_attn" (B,32).
        """
        b = scal.shape[0]
        if state is None:
            state = self.initial_state(b, scal.device)
        scal = _finite(scal, 10.0)
        fused, ent_attn = self._perceive(rays, ents, ent_mask, scal)
        run, since, cell, sub, move = self._advance(scal, state["since"], state["cell"], state["sub"], state["move"], run_manager)

        held = {k: state[k] for k in ("subgoal", "target", "value_field", "map_vec")}
        subgoal_logits = fused.new_zeros(b, spec.NUM_SUBGOALS)
        target_logits = fused.new_zeros(b, CELLS)
        n_run = int(run.sum())
        if n_run:
            rows = None if n_run == b else run.nonzero().squeeze(1)
            take = (lambda x: x) if rows is None else (lambda x: x.index_select(0, rows))
            forced = forced or {}
            fs, ft = forced.get("subgoal"), forced.get("target")
            m = self._manage(take(fused), take(state["h"]), take(map), None if fs is None else take(fs),
                             None if ft is None else take(ft), sample)
            held = {k: _put(v, rows, m[k]) for k, v in held.items()}
            subgoal_logits = _put(subgoal_logits, rows, m["subgoal_logits"])
            target_logits = _put(target_logits, rows, m["target_logits"])

        subgoal = self._clamped(held["subgoal"])
        goal = self._goal_features(subgoal, held["value_field"], cell, scal, map, held["map_vec"])
        h = self.gru(torch.cat([fused, goal], 1), state["h"])
        logits, value, value_int = self._heads(h, goal, held["map_vec"], masks)

        out = {"logits": logits, "value": value, "value_int": value_int, "subgoal": subgoal, "target": held["target"],
               "subgoal_logits": subgoal_logits, "target_logits": target_logits, "value_field": held["value_field"],
               "ent_attn": ent_attn}
        new_state = {"h": h, "subgoal": subgoal, "target": held["target"], "value_field": held["value_field"],
                     "map_vec": held["map_vec"], "cell": cell, "sub": sub, "move": move, "since": since}
        return out, new_state

    # ------------------------------------------------------------------------------------ sequences
    def forward_sequence(self, rays, ents, ent_mask, scal, map, state=None, masks=None, run_manager=None, forced=None,
                         sample: bool = True, reset=None):
        """The same computation as forward() for T consecutive steps: every tensor is (B,T,...).

        reset (B,T) bool marks the first step of an episode: that row's state is set back to the
        initial state before the step, and the manager runs on it. Returns (out, final_state) with
        every entry of out shaped (B,T,...).

        With both manager choices forced (learning from stored rollouts) nothing but the GRU depends
        on the previous step, so everything else is computed in a few large batches: sensors for
        all B*T steps, tower and value propagation for the manager steps only. Without forced
        choices the manager must choose inside the recurrence and the steps are simply run in order.
        """
        b, t_len = scal.shape[:2]
        if state is None:
            state = self.initial_state(b, scal.device)
        if reset is None:
            reset = torch.zeros(b, t_len, dtype=torch.bool, device=scal.device)
        reset = reset.bool()
        if forced is None or "subgoal" not in forced or "target" not in forced:
            return self._sequence_by_steps(rays, ents, ent_mask, scal, map, state, masks, run_manager, forced, sample, reset)

        n = b * t_len
        flat = lambda x: x.reshape(n, *x.shape[2:])                              # noqa: E731
        scal_f = _finite(flat(scal), 10.0)
        fused, ent_attn = self._perceive(flat(rays), flat(ents), flat(ent_mask), scal_f)

        # 1. Schedule and cell counters for every step (cheap, no gradient).
        with torch.no_grad():
            since, cell, sub, move = state["since"], state["cell"], state["sub"], state["move"]
            scal_t = scal_f.view(b, t_len, -1)
            runs, cells = [], []
            for t in range(t_len):
                run, since, cell, sub, move = self._advance(scal_t[:, t], since, cell, sub, move,
                                                            None if run_manager is None else run_manager[:, t], reset[:, t])
                runs.append(run)
                cells.append(cell)
            run = torch.stack(runs, 1)                                           # (B,T)
            cells = torch.stack(cells, 1).reshape(n, 2)
            # Which manager output is in force at every step: row k < B of the pools below is the
            # incoming state of sequence k, row B + m is the m-th manager step of this call.
            rows = run.reshape(n).nonzero().squeeze(1)
            rank = run.reshape(n).long().cumsum(0).view(b, t_len) - 1
            own = torch.arange(b, device=scal.device)[:, None]
            pick = torch.where(run, rank + b, own).cummax(1).values.reshape(n)

        # 2. Tower and value propagation on the manager steps only.
        pools = {k: state[k] for k in ("subgoal", "target", "value_field", "map_vec")}
        if rows.numel():
            feat, vec = self.tower(_as_unit_float(flat(map).index_select(0, rows)))
            sub_m = self._clamped(flat(forced["subgoal"]).index_select(0, rows).long())
            tgt_m = flat(forced["target"]).index_select(0, rows).long()
            emb_m = self.subgoal_emb(sub_m)
            new = {"subgoal": sub_m, "target": tgt_m, "value_field": self.vp(feat, emb_m, tgt_m), "map_vec": vec}
            pools = {k: torch.cat([v, new[k].to(v.dtype)]) for k, v in pools.items()}
        held = {k: v.index_select(0, pick) for k, v in pools.items()}            # (B*T, ...)

        # 3. Controller inputs for every step, then the recurrence, then the heads.
        subgoal = self._clamped(held["subgoal"])
        goal = self._goal_features(subgoal, held["value_field"], cells, scal_f, flat(map), held["map_vec"])
        x = torch.cat([fused, goal], 1).view(b, t_len, -1)
        h = state["h"]
        before, after = [], []
        for t in range(t_len):
            h = h.masked_fill(reset[:, t, None], 0.0)
            before.append(h)
            h = self.gru(x[:, t], h)
            after.append(h)
        logits, value, value_int = self._heads(torch.stack(after, 1).reshape(n, -1), goal, held["map_vec"],
                                               None if masks is None else flat(masks))

        # 4. Manager logits on the manager steps (they read the GRU state from before the step).
        subgoal_logits = fused.new_zeros(n, spec.NUM_SUBGOALS)
        target_logits = fused.new_zeros(n, CELLS)
        if rows.numel():
            h_before = torch.stack(before, 1).reshape(n, -1).index_select(0, rows)
            ctx, sl = self._subgoal_logits(fused.index_select(0, rows), h_before, vec)
            subgoal_logits = subgoal_logits.index_copy(0, rows, sl.to(subgoal_logits.dtype))
            target_logits = target_logits.index_copy(0, rows, self._target_logits(feat, ctx, emb_m).to(target_logits.dtype))

        seq = lambda x: x.view(b, t_len, *x.shape[1:])                           # noqa: E731
        out = {"logits": tuple(seq(l) for l in logits), "value": seq(value), "value_int": seq(value_int),
               "subgoal": seq(subgoal), "target": seq(held["target"]), "subgoal_logits": seq(subgoal_logits),
               "target_logits": seq(target_logits), "value_field": seq(held["value_field"]), "ent_attn": seq(ent_attn)}
        new_state = {"h": h, "subgoal": out["subgoal"][:, -1], "target": out["target"][:, -1],
                     "value_field": out["value_field"][:, -1], "map_vec": seq(held["map_vec"])[:, -1],
                     "cell": cell, "sub": sub, "move": move, "since": since}
        return out, new_state

    def _sequence_by_steps(self, rays, ents, ent_mask, scal, map, state, masks, run_manager, forced, sample, reset):
        """Reference form of forward_sequence: forward() once per step, with the resets applied in between."""
        outs = []
        for t in range(scal.shape[1]):
            state = self.reset_state(state, reset[:, t])
            run = None if run_manager is None else run_manager[:, t].bool() | reset[:, t]
            step_forced = None if forced is None else {k: v[:, t] for k, v in forced.items()}
            out, state = self.forward(rays[:, t], ents[:, t], ent_mask[:, t], scal[:, t], map[:, t], state,
                                      None if masks is None else masks[:, t], run, step_forced, sample)
            outs.append(out)
        stacked = {k: torch.stack([o[k] for o in outs], 1) for k in outs[0] if k != "logits"}
        stacked["logits"] = tuple(torch.stack([o["logits"][i] for o in outs], 1) for i in range(len(spec.HEAD_SIZES)))
        return stacked, state


class ExitDetector(nn.Module):
    """Finds the exit pad (a red floor square) in the 120 x 160 RGB frame.

    Output (B,3): logit that a pad is visible, and its position in the agent's frame as
    forward / 1024 and left / 1024 (map units). The floor can only appear below the horizon (image
    row 60), so the network reads rows 48-119. A distant pad is one or two pixels tall just under
    the horizon: the first convolution therefore runs at full resolution and every reduction is a
    max pool, which keeps the response of a thin sliver. The last layers keep the 9 x 20 layout
    (flatten, not global pooling) because image position is what encodes distance and bearing.
    """

    TOP = 48                                             # first image row that is read

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)      # 72 x 160
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)     # 36 x 80
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)     # 18 x 40
        self.conv4 = nn.Conv2d(48, 64, 3, padding=1)     # 9 x 20
        self.squeeze = nn.Conv2d(64, 16, 1)
        rows, cols = (spec.SCREEN_H - self.TOP) // 8, spec.SCREEN_W // 8
        self.fc1 = nn.Linear(16 * rows * cols, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (B,3,120,160), uint8 in [0,255] or float in [0,1] -> (B,3)."""
        x = _as_unit_float(rgb[:, :, self.TOP:])
        x = max_pool_2x2(torch.relu(self.conv1(x)))
        x = max_pool_2x2(torch.relu(self.conv2(x)))
        x = max_pool_2x2(torch.relu(self.conv3(x)))
        x = torch.relu(self.squeeze(torch.relu(self.conv4(x)))).flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))
