"""Rhythm protector: spare detections that sit on a periodic musical grid.

Idea
----
Hand-claps, snare/hi-hat hits and similar percussion make the click
detector fire again and again *on a beat grid*: the detection train of a
clapping song autocorrelates at the beat period (0.2-0.9 s). Real vinyl
defects arrive at random times (or, for a scratch, once per turntable
rotation: 1.8 s at 33 1/3, 1.333 s at 45, 0.769 s at 78 RPM).

For every detected region we look at the *event train* -- the peak times of
all detections in both channels -- in a window from ``past`` seconds before
to ``future`` seconds after the region, and ask: is there a period P such
that the grid ``t + k*P`` (k != 0, t = this region's peak) is occupied by
other events much more often than the phase-shifted grids
``t + (k + 1/4, 1/2, 3/4) * P``? The test statistic is

    S(P) = (on - off) / sqrt(on + off + 1)

(``on`` = occupied grid points, ``off`` = mean occupied shifted points, each
within ``tol`` seconds), maximised over P in [0.15, 0.9] s excluding the
rotation periods (+-2.5 %). The on-vs-off contrast makes the test
independent of event density, so dense random crackle in a quiet passage
(about as many off-grid as on-grid hits) is not mistaken for rhythm, and a
rotation-periodic scratch fills at most every m-th point of a P = R/m grid.
A region is protected (not repaired) when max S(P) > ``theta``.

Safety veto for rotation-periodic defects: if events in the same channel sit
at t - R' and t - 2R' (or t + R', or t - R' and t + R') for some R' within
+-2 % of a rotation period, with the two spacings equal to within
``rot_precision`` seconds (turntable speed is constant to far better than a
human or a drum part repeats), the region is treated as a click and repaired
regardless of the rhythm score.

The event train is built from the detector's logits at a fixed internal
threshold (``event_thr``), not the user's sensitivity, so a gentler user
threshold does not weaken the rhythm evidence.

Streaming (Rust) form: keep a ring buffer of event timestamps covering
``past + future`` seconds (a few hundred u32s at most); a region is released
``future`` seconds after its peak, when its whole window is known. Cost per
region: |P grid| x |k| x 4 binary searches in a sorted list (~10^4 cheap ops).
"""

from __future__ import annotations

import numpy as np

from pressing_tools import classical

ROTATION_PERIODS = (1.8, 60 / 45, 60 / 78)      # 33 1/3, 45, 78 RPM


def _event_times(s2: np.ndarray, x2: np.ndarray, thr: float) -> list[np.ndarray]:
    """Per channel: sample index of the logit peak of each hysteresis region."""
    from pressing_tools.protect_eval import regions_for
    regs = regions_for(x2, s2, thr)
    return [np.array([a + int(np.argmax(s2[c, a:b])) for a, b in regs[c]], np.int64) for c in range(2)]


class Protector:
    name = "rhythm"

    def __init__(self, past: float = 3.0, future: float = 1.0, tol: float = 0.008,
                 p_min: float = 0.15, p_max: float = 0.9, n_periods: int = 120,
                 theta: float = 2.5, min_points: int = 4, event_thr: float | None = 2.75,
                 rot_tol: float = 0.025, rot_precision: float = 0.0005, veto: bool = True):
        self.past, self.future, self.tol = past, future, tol
        self.theta, self.min_points, self.event_thr = theta, min_points, event_thr
        self.rot_precision, self.veto = rot_precision, veto
        P = np.exp(np.linspace(np.log(p_min), np.log(p_max), n_periods))
        keep = np.array([all(abs(p - r) / r >= rot_tol for r in ROTATION_PERIODS) for p in P])
        self.P = P[keep]
        kmax = int(past / self.P.min()) + 1
        fmax = int(future / self.P.min()) + 1
        k = np.arange(-kmax, fmax + 1)
        self.k = k[k != 0]
        self.T = self.P[:, None] * self.k[None, :]                     # (nP, nk) offsets in s
        self.valid = (self.T >= -past) & (self.T <= future)
        self.n = self.valid.sum(1)
        self.offsets = (0.25, 0.5, 0.75)

    @property
    def lookahead_s(self) -> float:
        return self.future + self.tol

    # MARK: - Scores

    def _hits(self, ev: np.ndarray, T: np.ndarray) -> np.ndarray:
        if len(ev) == 0:
            return np.zeros(T.shape, bool)
        i = np.searchsorted(ev, T)
        d = np.minimum(np.abs(ev[np.clip(i, 0, len(ev) - 1)] - T),
                       np.abs(ev[np.clip(i - 1, 0, len(ev) - 1)] - T))
        return (d < self.tol) & self.valid

    def rhythm_score(self, ev: np.ndarray) -> float:
        """ev: sorted event times relative to the region peak (s), self excluded."""
        on = self._hits(ev, self.T).sum(1)
        off = np.mean([self._hits(ev, self.T + o * self.P[:, None]).sum(1) for o in self.offsets], 0)
        s = (on - off) / np.sqrt(on + off + 1.0)
        s[self.n < self.min_points] = -np.inf
        return float(s.max())

    def rotation_periodic(self, ev_same: np.ndarray, sr: int) -> bool:
        """ev_same: event times (s) relative to t in the same channel. True if two of
        them form an exactly evenly spaced triple with t at a rotation period."""
        for R in ROTATION_PERIODS:
            lo, hi = 0.98 * R, 1.02 * R
            back = ev_same[(ev_same <= -lo) & (ev_same >= -hi)]
            fwd = ev_same[(ev_same >= lo) & (ev_same <= hi)]
            back2 = ev_same[(ev_same <= -2 * lo) & (ev_same >= -2 * hi)]
            for e1 in back:                       # t-R', t-2R'
                if np.any(np.abs(back2 - 2 * e1) < self.rot_precision):
                    return True
                if np.any(np.abs(fwd + e1) < self.rot_precision):   # t-R', t+R'
                    return True
        return False

    # MARK: - Protector API

    def keep(self, x2, s2, regions, sr):
        thr = self.event_thr
        if thr is None:
            peaks = [np.array([a + int(np.argmax(s2[c, a:b])) for a, b in regions[c]], np.int64)
                     for c in range(2)]
        else:
            peaks = _event_times(s2, x2, thr)
        ch = [p / sr for p in peaks]
        allt = np.sort(np.concatenate(ch))
        out = []
        for c in range(2):
            res = []
            for a, b in regions[c]:
                t = (a + int(np.argmax(s2[c, a:b]))) / sr
                lo = np.searchsorted(allt, t - self.past - self.tol)
                hi = np.searchsorted(allt, t + self.future + self.tol, side="right")
                ev = allt[lo:hi] - t
                ev = ev[np.abs(ev) > 0.002]
                protect = self.rhythm_score(ev) > self.theta
                if protect and self.veto:
                    same = ch[c][(ch[c] > t - 2 * 1.02 * max(ROTATION_PERIODS) - 0.01)
                                 & (ch[c] < t + 1.02 * max(ROTATION_PERIODS) + 0.01)] - t
                    same = same[np.abs(same) > 0.002]
                    # only past events and those within the lookahead are available
                    same = same[same <= self.future + self.tol]
                    protect = not self.rotation_periodic(same, sr)
                res.append(not protect)
            out.append(res)
        return out
