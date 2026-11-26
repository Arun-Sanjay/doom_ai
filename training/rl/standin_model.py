"""A small network with the forward signature of the agent's policy network.

Training-time scaffolding, never imported by the agent. It exists so the rollout loop, the buffer
and the loss can be tested end to end without the real network: same inputs, same output keys,
same recurrent-state contract (a dict holding the GRU state and the manager's held choice).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training import paths  # noqa: F401
from doomnet import spec

N_CELLS = spec.CROP_SIZE * spec.CROP_SIZE


class StandInPolicy(nn.Module):
    """Encoders -> GRU cell -> manager (sub-goal, target cell) -> controller heads and two values."""

    partial_forcing = True          # ``forced`` entries of -1 are rows that choose for themselves

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.ray = nn.Linear(spec.NUM_RAYS * spec.RAY_FEATS, 64)
        self.ent = nn.Linear(spec.ENT_FEATS, 16)
        self.scal = nn.Linear(spec.SCALAR_DIM, 32)
        self.map = nn.Sequential(
            nn.Conv2d(spec.MAP_CHANNELS, 8, 5, stride=4, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(8 * 16, 32))
        self.fuse = nn.Linear(64 + 16 + 32 + 32, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.sub_head = nn.Linear(hidden, spec.NUM_SUBGOALS)
        self.row_head = nn.Linear(hidden, spec.CROP_SIZE)
        self.col_head = nn.Linear(hidden, spec.CROP_SIZE)
        self.sub_emb = nn.Embedding(spec.NUM_SUBGOALS, 16)
        self.tgt_emb = nn.Linear(2, 16)
        self.ctrl = nn.Linear(hidden + 32, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in spec.HEAD_SIZES])
        self.v_ext = nn.Linear(hidden, 1)
        self.v_int = nn.Linear(hidden, 1)

    def initial_state(self, batch: int, device) -> dict:
        return {"h": torch.zeros(batch, self.hidden, device=device),
                "subgoal": torch.zeros(batch, dtype=torch.long, device=device),
                "target": torch.zeros(batch, dtype=torch.long, device=device)}

    def forward(self, rays, ents, ent_mask, scal, map, state, masks=None, run_manager=None, forced=None, sample=True):
        B = rays.shape[0]
        if not map.is_floating_point():                 # uint8 crops (0..255) as stored by the rollout workers
            map = map.float() * (1.0 / 255.0)
        m = ent_mask.unsqueeze(-1).float()
        ent_feat = (F.relu(self.ent(ents)) * m).sum(1) / m.sum(1).clamp(min=1.0)
        x = torch.cat([F.relu(self.ray(rays.flatten(1))), ent_feat, F.relu(self.scal(scal)), F.relu(self.map(map))], -1)
        h = self.gru(F.relu(self.fuse(x)), state["h"])

        sub_logits = self.sub_head(h)
        tgt_logits = (self.row_head(h)[:, :, None] + self.col_head(h)[:, None, :]).flatten(1)
        if run_manager is None:
            run_manager = torch.ones(B, dtype=torch.bool, device=rays.device)
        if sample:
            new_sub = torch.multinomial(F.softmax(sub_logits, -1), 1).squeeze(1)
            new_tgt = torch.multinomial(F.softmax(tgt_logits, -1), 1).squeeze(1)
        else:
            new_sub, new_tgt = sub_logits.argmax(-1), tgt_logits.argmax(-1)
        if forced is not None:                          # a negative entry leaves that row its own choice
            new_sub = torch.where(forced["subgoal"] >= 0, forced["subgoal"].long(), new_sub)
            new_tgt = torch.where(forced["target"] >= 0, forced["target"].long(), new_tgt)
        sub = torch.where(run_manager, new_sub, state["subgoal"])
        tgt = torch.where(run_manager, new_tgt, state["target"])

        cell = torch.stack([tgt // spec.CROP_SIZE, tgt % spec.CROP_SIZE], -1).float() / spec.CROP_SIZE - 0.5
        goal = torch.cat([self.sub_emb(sub), self.tgt_emb(cell)], -1)
        c = F.relu(self.ctrl(torch.cat([h, goal], -1)))
        logits = [head(c) for head in self.heads]
        if masks is not None:
            neg = float("-inf")
            use, attack = logits[spec.HEAD_USE], logits[spec.HEAD_ATTACK]
            logits[spec.HEAD_USE] = torch.stack([use[:, 0], use[:, 1].masked_fill(~masks[:, 0], neg)], -1)
            logits[spec.HEAD_ATTACK] = torch.stack([attack[:, 0], attack[:, 1].masked_fill(~masks[:, 1], neg)], -1)
        out = {"logits": logits, "value": self.v_ext(c).squeeze(-1), "value_int": self.v_int(c).squeeze(-1),
               "subgoal": sub, "target": tgt, "subgoal_logits": sub_logits, "target_logits": tgt_logits,
               "value_field": torch.zeros(B, 1, spec.CROP_SIZE, spec.CROP_SIZE, device=rays.device),
               "ent_attn": torch.zeros(B, spec.MAX_ENTITIES, device=rays.device)}
        return out, {"h": h, "subgoal": sub, "target": tgt}
