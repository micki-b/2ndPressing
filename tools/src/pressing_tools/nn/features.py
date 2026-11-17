"""Causal input features for the neural click detector (spec section 2).

Per channel at 48 kHz::

    d1[n] = x[n] - x[n-1]            (x[-1] = 0)
    d2[n] = d1[n] - d1[n-1]
    l_k   = ln(|d_k| + EPS)
    m_k   = mean of l_k over the last NW samples (expanding window at start)
    s_k   = max(exp(m_k), FLOOR)
    g_k   = asinh(d_k / s_k / ASINH_SCALE)

The scale is a causal geometric mean of |d_k| over ~43 ms, so the features
are gain-invariant (away from the floor) and need no lookahead. Absolute
level is deliberately not an input.

Model input per channel (weights shared between L and R):
    L: [g1_L, g2_L, g1_R]      R: [g1_R, g2_R, g1_L]

Three implementations that must agree: ``features_np`` (float64 reference),
``features_torch`` (float32 training on device) and ``FeatureState``
(streaming, any block size).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6
FLOOR = 1e-5
NW = 2048
ASINH_SCALE = 8.0
N_FEATURES = 3


# MARK: - Reference (numpy, float64)


def _diffs(x: np.ndarray, prev_x: float = 0.0, prev_d1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.diff(x, prepend=prev_x)
    d2 = np.diff(d1, prepend=prev_d1)
    return d1, d2


def _g(d: np.ndarray, m: np.ndarray) -> np.ndarray:
    s = np.maximum(np.exp(m), FLOOR)
    return np.arcsinh(d / s / ASINH_SCALE)


def _window_mean(l: np.ndarray) -> np.ndarray:
    n = len(l)
    c = np.concatenate([[0.0], np.cumsum(l)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - NW + 1)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


def channel_gs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(g1, g2) for one channel, float64."""
    x = np.asarray(x, np.float64)
    d1, d2 = _diffs(x)
    g1 = _g(d1, _window_mean(np.log(np.abs(d1) + EPS)))
    g2 = _g(d2, _window_mean(np.log(np.abs(d2) + EPS)))
    return g1, g2


def assemble(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """g1, g2: (2, L) -> model input (2, 3, L)."""
    return np.stack([np.stack([g1[0], g2[0], g1[1]]),
                     np.stack([g1[1], g2[1], g1[0]])])


def features_np(x2: np.ndarray) -> np.ndarray:
    """x2: (2, L) stereo (or (1, L) mono) -> (2, 3, L) float64."""
    x2 = np.atleast_2d(np.asarray(x2, np.float64))
    if x2.shape[0] == 1:
        x2 = np.repeat(x2, 2, axis=0)
    gs = [channel_gs(x2[c]) for c in range(2)]
    return assemble(np.stack([g[0] for g in gs]), np.stack([g[1] for g in gs]))


# MARK: - Training (torch, float32)


def features_torch(x):
    """x: (B, 2, T) float tensor -> (2B, 3, T); sequence order b0L, b0R, b1L, ..."""
    import torch

    B, C, T = x.shape
    assert C == 2
    x = x.float()
    zero = torch.zeros_like(x[..., :1])
    d1 = torch.diff(x, dim=-1, prepend=zero)
    d2 = torch.diff(d1, dim=-1, prepend=zero)
    idx = torch.arange(T, device=x.device)
    count = (torch.clamp(idx + 1, max=NW)).to(x.dtype)

    def g(d):
        l = torch.log(d.abs() + EPS)
        # Centre before the cumulative sum so float32 stays accurate over long T.
        mu = l.mean(dim=-1, keepdim=True)
        c = torch.cumsum(l - mu, dim=-1)
        c = torch.cat([torch.zeros_like(c[..., :1]), c], dim=-1)
        lo = torch.clamp(idx - NW + 1, min=0)
        m = (c[..., idx + 1] - c[..., lo]) / count + mu
        s = torch.clamp(torch.exp(m), min=FLOOR)
        return torch.asinh(d / s / ASINH_SCALE)

    g1, g2 = g(d1), g(d2)                          # (B, 2, T)
    left = torch.stack([g1[:, 0], g2[:, 0], g1[:, 1]], dim=1)
    right = torch.stack([g1[:, 1], g2[:, 1], g1[:, 0]], dim=1)
    return torch.stack([left, right], dim=1).reshape(2 * B, N_FEATURES, T)


# MARK: - Streaming


class FeatureState:
    """Streaming feature computation; process() accepts any block length."""

    def __init__(self) -> None:
        self.prev_x = np.zeros(2)
        self.prev_d1 = np.zeros(2)
        self.hist = [[np.zeros(0), np.zeros(0)] for _ in range(2)]   # last <= NW-1 l values per (ch, k)

    def _gk(self, ch: int, k: int, d: np.ndarray) -> np.ndarray:
        l = np.log(np.abs(d) + EPS)
        h = self.hist[ch][k]
        ext = np.concatenate([h, l])
        c = np.concatenate([[0.0], np.cumsum(ext)])
        pos = np.arange(len(h), len(ext))                 # indices of new samples in ext
        lo = np.maximum(0, pos - NW + 1)
        m = (c[pos + 1] - c[lo]) / (pos + 1 - lo)
        self.hist[ch][k] = ext[-(NW - 1):]
        return _g(d, m)

    def process(self, block2: np.ndarray) -> np.ndarray:
        block2 = np.atleast_2d(np.asarray(block2, np.float64))
        if block2.shape[0] == 1:
            block2 = np.repeat(block2, 2, axis=0)
        g1 = np.zeros_like(block2)
        g2 = np.zeros_like(block2)
        for ch in range(2):
            d1, d2 = _diffs(block2[ch], self.prev_x[ch], self.prev_d1[ch])
            if len(d1):
                self.prev_x[ch], self.prev_d1[ch] = block2[ch, -1], d1[-1]
            g1[ch], g2[ch] = self._gk(ch, 0, d1), self._gk(ch, 1, d2)
        return assemble(g1, g2)
