"""Batch augmentation for the exit detector, applied on the learner's device to every batch.

Training-time scaffolding. Never imported by the agent in submission/.

The pad is a small red patch; a detector trained on raw frames could key on exact pixel values of
the pad and of the floors it happened to see. Every training batch is therefore re-coloured at
random, each frame with its own parameters:

    flip          mirror left-right (the label's left offset changes sign)
    flash         the engine's screen tints, simulated: red after damage, yellow after a pickup
                  (a blend of the whole frame towards one colour, as the engine does it)
    gain          one factor per colour channel
    brightness    one factor for the frame, plus a small offset
    contrast      scale about the frame's mean
    gamma         power curve
    saturation    scale the distance from grey
    hue           rotate colours about the grey axis
    noise         Gaussian, per pixel
    blur          blend with a 3 x 3 binomial blur (slight: a far pad is one pixel row tall)

`augment(rgb, pos, gen)` takes uint8 frames (B, H, W, 3) and returns float frames (B, 3, H, W) in
[0, 1] with the mirrored positions. Parameters are drawn on the CPU from `gen`, so a seed fixes
them; only the pixel noise comes from the device's own generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_RGB_TO_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]])
_YIQ_TO_RGB = torch.linalg.inv(_RGB_TO_YIQ)
_DAMAGE_TINT = torch.tensor([1.0, 0.0, 0.0])
_PICKUP_TINT = torch.tensor([0.8431, 0.7333, 0.2706])
_BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


@dataclass
class AugmentConfig:
    flip: float = 0.5
    damage_flash: float = 0.10           # share of frames that get a red tint
    damage_alpha: tuple = (0.05, 0.75)
    pickup_flash: float = 0.03
    pickup_alpha: tuple = (0.03, 0.25)
    gain: float = 0.10                   # per-channel factor in [1 - gain, 1 + gain]
    brightness: tuple = (0.70, 1.30)
    offset: float = 0.04
    contrast: tuple = (0.75, 1.30)
    gamma: float = 0.35                  # exponent = exp(U(-gamma, gamma))
    saturation: tuple = (0.60, 1.40)
    hue: float = 0.04                    # turns of the colour circle, either way (0.04 = 14 degrees)
    noise: float = 0.03                  # largest standard deviation of the pixel noise
    blur: float = 0.15                   # share of frames that are blurred
    blur_alpha: tuple = (0.2, 0.6)


def _uniform(gen, n: int, lo: float, hi: float) -> torch.Tensor:
    return torch.rand(n, generator=gen) * (hi - lo) + lo


def colour_matrices(saturation: torch.Tensor, hue_turns: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) matrices that scale saturation and rotate hue (grey stays grey)."""
    angle = hue_turns * (2.0 * math.pi)
    cos, sin = torch.cos(angle) * saturation, torch.sin(angle) * saturation
    rot = torch.zeros(len(angle), 3, 3)
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
    return _YIQ_TO_RGB @ rot @ _RGB_TO_YIQ


def augment(rgb: torch.Tensor, pos: torch.Tensor, gen: torch.Generator | None = None,
            cfg: AugmentConfig | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 (B, H, W, 3) frames and (B, 2) positions -> float (B, 3, H, W) in [0, 1] and mirrored positions."""
    cfg = cfg or AugmentConfig()
    dev, b = rgb.device, rgb.shape[0]

    def put(t: torch.Tensor) -> torch.Tensor:
        return t.to(dev).view(b, *([1] * (4 - 1)))

    x = rgb.permute(0, 3, 1, 2).float() * (1.0 / 255.0)

    flip = torch.rand(b, generator=gen) < cfg.flip
    flip_dev = flip.to(dev)
    x = torch.where(flip_dev.view(b, 1, 1, 1), x.flip(3), x)
    pos = torch.where(flip_dev.view(b, 1), pos * torch.tensor([1.0, -1.0], device=dev), pos)

    # screen tints: frame * (1 - a) + a * colour
    for share, (lo, hi), colour in ((cfg.damage_flash, cfg.damage_alpha, _DAMAGE_TINT),
                                    (cfg.pickup_flash, cfg.pickup_alpha, _PICKUP_TINT)):
        alpha = _uniform(gen, b, lo, hi) * (torch.rand(b, generator=gen) < share)
        alpha = put(alpha)
        x = x * (1.0 - alpha) + alpha * colour.to(dev).view(1, 3, 1, 1)

    gain = (torch.rand(b, 3, generator=gen) * 2.0 - 1.0) * cfg.gain + 1.0
    gain = gain * _uniform(gen, b, *cfg.brightness).view(b, 1)
    x = x * gain.to(dev).view(b, 3, 1, 1) + put(_uniform(gen, b, -cfg.offset, cfg.offset))

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * put(_uniform(gen, b, *cfg.contrast)) + mean

    x = x.clamp(1e-4, 1.0) ** put(torch.exp(_uniform(gen, b, -cfg.gamma, cfg.gamma)))

    matrix = colour_matrices(_uniform(gen, b, *cfg.saturation), _uniform(gen, b, -cfg.hue, cfg.hue)).to(dev)
    x = torch.einsum("bij,bjhw->bihw", matrix, x)

    sigma = put(_uniform(gen, b, 0.0, cfg.noise) * (torch.rand(b, generator=gen) < 0.7))
    x = x + torch.randn_like(x) * sigma

    blur = _uniform(gen, b, *cfg.blur_alpha) * (torch.rand(b, generator=gen) < cfg.blur)
    if bool((blur > 0).any()):
        kernel = _BLUR.to(dev).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        soft = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
        alpha = put(blur)
        x = x * (1.0 - alpha) + soft * alpha
    return x.clamp(0.0, 1.0), pos


def plain(rgb: torch.Tensor) -> torch.Tensor:
    """uint8 (B, H, W, 3) -> float (B, 3, H, W) in [0, 1], no augmentation."""
    return rgb.permute(0, 3, 1, 2).float() * (1.0 / 255.0)
