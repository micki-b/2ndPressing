"""ClickNet: strided, dilated, causal TCN click detector (spec section 3).

Only Conv1d, ReLU, add, permute and reshape -- trivially hand-portable to Rust.
``logit[n]`` is the prediction for target sample ``n - D``; it depends only on
input samples ``<= 4*floor(n/4) + 3`` (causal per 4-sample frame).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pressing_tools.nn.features import N_FEATURES


@dataclass
class NetConfig:
    C: int = 16
    blocks: int = 8          # dilations 1, 2, 4, ... 2**(blocks-1) frames
    stem_k: int = 8
    stride: int = 4
    D: int = 512             # output delay (lookahead) in samples

    def to_dict(self) -> dict:
        return asdict(self)


def receptive_field(cfg: NetConfig) -> int:
    """Input samples that can influence one logit."""
    dil = sum(2 ** i for i in range(cfg.blocks))
    return cfg.stem_k + cfg.stride * 2 * dil


def param_count_formula(cfg: NetConfig) -> int:
    C, k = cfg.C, cfg.stem_k
    stem = N_FEATURES * C * k + C
    block = 3 * C * C + C + C * C + C
    head = C * C + C + C * cfg.stride + cfg.stride
    return stem + cfg.blocks * block + head


def macs_per_sample(cfg: NetConfig) -> float:
    """Multiply-accumulates per input sample per channel."""
    C, s = cfg.C, cfg.stride
    stem = N_FEATURES * cfg.stem_k * C / s
    blocks = cfg.blocks * (3 * C * C + C * C) / s
    head = (C * C + C * s) / s
    return stem + blocks + head


class ResBlock(nn.Module):
    def __init__(self, C: int, dilation: int) -> None:
        super().__init__()
        self.pad = 2 * dilation
        self.dil = nn.Conv1d(C, C, 3, dilation=dilation)
        self.mix = nn.Conv1d(C, C, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dil(F.pad(x, (self.pad, 0))))
        return F.relu(x + self.mix(h))


class ClickNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or NetConfig()
        self.stem = nn.Conv1d(N_FEATURES, cfg.C, cfg.stem_k, stride=cfg.stride)
        self.blocks = nn.ModuleList([ResBlock(cfg.C, 2 ** i) for i in range(cfg.blocks)])
        self.head1 = nn.Conv1d(cfg.C, cfg.C, 1)
        self.head2 = nn.Conv1d(cfg.C, cfg.stride, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (N, 3, T), T a multiple of stride -> logits (N, T)."""
        cfg = self.cfg
        h = F.relu(self.stem(F.pad(feat, (cfg.stem_k - cfg.stride, 0))))
        for b in self.blocks:
            h = b(h)
        y = self.head2(F.relu(self.head1(h)))          # (N, stride, F)
        N, S, Fr = y.shape
        return y.permute(0, 2, 1).reshape(N, Fr * S)
