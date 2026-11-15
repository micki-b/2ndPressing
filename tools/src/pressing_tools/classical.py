"""Classical click detection and repair (the baseline to beat).

Detection (after Vaseghi & Godsill): fit a short-term autoregressive (AR)
model per block; clicks are samples the model cannot predict. The forward
prediction error is passed through the time-reversed AR filter (a matched
filter), which concentrates each click's energy back onto its true position,
then normalised by the block's robust error level.

Repair: least-squares AR interpolation (LSAR, Janssen et al.). Unknown
samples are chosen to minimise the AR prediction-error energy across the gap,
using an AR model fitted to the clean audio on both sides.

All functions operate on a single channel (1-D float arrays).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz


# MARK: - Configuration


@dataclass
class ClassicalConfig:
    order: int = 32            # AR order for detection
    block: int = 2048          # samples per AR block (detection)
    threshold: float = 4.0     # score threshold (robust sigmas) that triggers a region
    extend_threshold: float = 2.0  # hysteresis: grow a triggered region while score > this
    merge: int = 8             # merge hot samples closer than this
    pre: int = 2               # widen regions before ...
    post: int = 16             # ... and after (click tails decay slowly)
    max_region: int = 4800     # don't touch anything longer (0.1 s at 48 kHz)
    min_contrast: float = 16.0 # reject regions whose curvature contrast is below this (tuned on val)
    contrast_context: int = 150
    repair_order: int = 32     # AR order for interpolation
    repair_context: int = 1024 # samples each side used to fit the repair model


# MARK: - AR helpers


def autocorr(x: np.ndarray, p: int) -> np.ndarray:
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    return np.fft.irfft(f * np.conj(f), nfft)[:p + 1]


def ar_from_autocorr(r: np.ndarray, p: int) -> np.ndarray:
    """Return the prediction-error filter A = [1, a1..ap] (e = A * x)."""
    if r[0] <= 1e-20:
        return np.r_[1.0, np.zeros(p)]
    r = r.copy()
    r[0] *= 1.0 + 1e-6  # white-noise correction keeps the Toeplitz system well conditioned
    try:
        a = solve_toeplitz(r[:p], -r[1:p + 1])
    except np.linalg.LinAlgError:
        return np.r_[1.0, np.zeros(p)]
    return np.r_[1.0, a]


# MARK: - Detection


def scores(x: np.ndarray, cfg: ClassicalConfig) -> np.ndarray:
    """Per-sample click score in robust sigmas (higher = more click-like)."""
    x = np.asarray(x, np.float64)
    n, p, B = len(x), cfg.order, cfg.block
    e = np.zeros(n)
    filters = []
    for b0 in range(0, n, B):
        b1 = min(b0 + B, n)
        lo, hi = max(0, b0 - B // 2), min(n, b1 + B // 2)
        seg = x[lo:hi] * np.hanning(hi - lo)
        A = ar_from_autocorr(autocorr(seg, p), p)
        h = max(0, b0 - p)
        e[b0:b1] = signal.lfilter(A, 1.0, x[h:b1])[b0 - h:]
        filters.append((b0, b1, A))

    out = np.zeros(n)
    for b0, b1, A in filters:
        hi = min(n, b1 + p)
        # matched filter: d[n] = sum_k A[k] e[n + k]
        d = signal.lfilter(A, 1.0, e[b0:hi][::-1])[::-1][:b1 - b0]
        sigma = 1.4826 * np.median(np.abs(e[b0:b1])) + 1e-12
        out[b0:b1] = np.abs(d) / (sigma * np.linalg.norm(A))
    out[:p] = 0.0  # no prediction history yet: errors there are meaningless
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def contrast(x: np.ndarray, a: int, b: int, ctx: int = 150, guard: int = 20) -> float:
    """Peak curvature |x''| inside [a, b) over the median curvature around it.

    Clicks are far sharper than their surroundings (median ~25x on the
    library); musical attacks that fool the AR detector are not (~7x)."""
    lo, hi = max(1, a - ctx), min(len(x) - 1, b + ctx)
    d = np.abs(x[lo + 1:hi + 1] - 2 * x[lo:hi] + x[lo - 1:hi - 1])
    ia, ib = a - lo, b - lo
    around = np.r_[d[:max(1, ia - guard)], d[min(len(d) - 1, ib + guard):]]
    return float(d[ia:ib].max() / (np.median(around) + 1e-12)) if ib > ia else 0.0


def regions_from_scores(s: np.ndarray, cfg: ClassicalConfig,
                        threshold: float | None = None,
                        x: np.ndarray | None = None) -> np.ndarray:
    """Hysteresis regions: triggered where s > threshold, grown to the
    surrounding run where s > extend_threshold (a click's decaying
    tail is more predictable than its onset and scores lower), then padded.
    If x is given and cfg.min_contrast > 0, regions that are not sharp
    relative to their surroundings (musical attacks) are dropped."""
    thr = cfg.threshold if threshold is None else threshold
    mask = np.zeros(len(s), bool)
    if not (s > thr).any():
        return mask
    weak = s > min(cfg.extend_threshold, thr)
    # merge weak runs separated by small gaps before deciding which are triggered
    idx = np.flatnonzero(weak)
    splits = np.flatnonzero(np.diff(idx) > cfg.merge) + 1
    for g in np.split(idx, splits):
        a, b = int(g[0]), int(g[-1]) + 1
        if s[a:b].max() <= thr:
            continue
        a, b = max(0, a - cfg.pre), min(len(s), b + cfg.post)
        if b - a > cfg.max_region:
            continue
        if x is not None and cfg.min_contrast > 0 and \
                contrast(x, a, b, cfg.contrast_context) < cfg.min_contrast:
            continue
        mask[a:b] = True
    return mask


def detect(x: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    cfg = cfg or ClassicalConfig()
    return regions_from_scores(scores(x, cfg), cfg, x=x)


# MARK: - Repair (LSAR)


def lsar_fill(x: np.ndarray, mask: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """Replace masked samples by least-squares AR interpolation."""
    cfg = cfg or ClassicalConfig()
    y = np.asarray(x, np.float64).copy()
    n, p, ctx = len(y), cfg.repair_order, cfg.repair_context
    rs = runs(mask)
    # Group runs closer than 2p: they share one solve.
    groups: list[list[tuple[int, int]]] = []
    for r in rs:
        if groups and r[0] - groups[-1][-1][1] < 2 * p:
            groups[-1].append(r)
        else:
            groups.append([r])

    for g in groups:
        u0, u1 = g[0][0], g[-1][1]
        if u0 < p or u1 + p > n:
            continue  # not enough history at the edges; leave untouched
        left = y[max(0, u0 - ctx):u0]          # already repaired where needed
        right = y[u1:min(n, u1 + ctx)]
        r = np.zeros(p + 1)
        for piece in (left, right):
            if len(piece) > p:
                r += autocorr(piece * np.hanning(len(piece)), p)
        A = ar_from_autocorr(r, p)

        c0, c1 = u0 - p, u1 + p
        seg = y[c0:c1]
        unknown = mask[c0:c1].copy()
        N = c1 - c0
        M = np.zeros((N - p, N))
        Ar = A[::-1]
        for i in range(N - p):
            M[i, i:i + p + 1] = Ar
        Mu, Mk = M[:, unknown], M[:, ~unknown]
        xu, *_ = np.linalg.lstsq(Mu, -Mk @ seg[~unknown], rcond=None)
        seg[unknown] = xu
    return y


def declick(x: np.ndarray, cfg: ClassicalConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detect and repair; returns (repaired, mask)."""
    cfg = cfg or ClassicalConfig()
    mask = detect(x, cfg)
    return lsar_fill(x, mask, cfg), mask
