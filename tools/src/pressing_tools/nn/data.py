"""Training crops for the neural click detector (spec section 5).

``CropSampler`` is an IterableDataset that yields *whole batches* from each
DataLoader worker (use ``batch_size=None``). A batch holds 128 stereo crops of
T = 12,288 samples; targets are scored on [S0, S1) = [3584, 11776) so every
scored sample has its full network and normaliser history inside the crop,
and the last D = 512 samples supply the lookahead.

Per channel the sampler builds (target coordinates = crop sample index):
    target  uint8    1 = click
    weight  float32  0 = ignored (don't care / outside scored region)
    neg_ok  bool     safe negatives for the chunk-max term
plus an event table (seq, a, b, w) of positives fully inside [S0, S1) for the
multiple-instance term. Sequence index = 2 * crop + channel.

Prep tables (data/v1/nn/<split>/, see nn/prep.py): events, hardneg, suspected,
bank, bank_meta, sides.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from pressing_tools import classical
from pressing_tools.memmap import open_split

T_CROP = 12288
S0, S1 = 3584, 11776
WIN_LEN = 96000
LEVEL_EDGES = (-45.0, -30.0, -18.0)


# MARK: - Configuration


@dataclass
class SamplerConfig:
    batch: int = 128
    frac_event: float = 0.4
    frac_hardneg: float = 0.3          # remainder uniform
    transplant_p: float = 0.5          # per uniform / hard-negative crop
    transplant_max: int = 3
    contrast_lo: float = 5.0
    contrast_hi: float = 120.0
    contrast_positive: float = 8.0     # audibility rule
    attack_p: float = 0.5              # first click on the attack itself (hard-neg crops)
    loud_p: float = 0.5                # restrict to loud positions (> -30 dBFS) half the time
    polarity_p: float = 0.5
    gain_db: tuple[float, float] = (-20.0, 6.0)
    mono_p: float = 0.05
    pre_ignore: int = 3                # ignore before each label-1 run
    post_ignore: int = 6               # ... and after (VinylStudio extents truncate tails)
    label2_dilate: int = 8
    suspected_dilate: int = 16
    neg_guard: int = 16
    event_margin: int = 64


# MARK: - Small helpers


def dilate(m: np.ndarray, k: int) -> np.ndarray:
    """Binary dilation by k samples each side (1-D or along the last axis)."""
    if k <= 0:
        return m.copy()
    c = np.cumsum(np.pad(m.astype(np.int32), [(0, 0)] * (m.ndim - 1) + [(k + 1, k)]), axis=-1)
    return (c[..., 2 * k + 1:] - c[..., :-(2 * k + 1)]) > 0


def runs(m: np.ndarray) -> list[tuple[int, int]]:
    return classical.runs(m)


def level_bin(db: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(LEVEL_EDGES), db)


def _inv_sqrt_count(keys: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return 1.0 / np.sqrt(cnt[inv])


def _rows_by_window(table: np.ndarray, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """Sort a table by window; return (sorted table, offsets (n_windows+1,))."""
    order = np.argsort(table["win"], kind="stable")
    t = table[order]
    off = np.searchsorted(t["win"], np.arange(n_windows + 1))
    return t, off


def _curv(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x)
    d[1:-1] = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    return d


# MARK: - Sampler


class CropSampler(IterableDataset):
    def __init__(self, data: Path, split: str = "train", cfg: SamplerConfig | None = None,
                 seed: int = 0, transplant: bool = True) -> None:
        super().__init__()
        self.data, self.split = Path(data), split
        self.cfg = cfg or SamplerConfig()
        self.seed = seed
        self.transplant = transplant
        self._ready = False

    # Lazy so the object pickles cheaply into DataLoader workers.
    def _load(self) -> None:
        if self._ready:
            return
        mm = open_split(self.data, self.split)
        self.audio, self.label = mm["audio"], mm["label"]
        self.n_windows = self.audio.shape[0]
        nd = self.data / "nn" / self.split
        ev = np.load(nd / "events.npy")
        hn = np.load(nd / "hardneg.npy")
        sp = np.load(nd / "suspected.npy")
        self.bank = np.load(nd / "bank.npy")
        self.bank_meta = np.load(nd / "bank_meta.npy")
        self.events = ev
        self.hardneg = hn
        # Sampling distributions (spec 5): event P ~ n_side^-1/2 * n_levelbin^-1/2.
        w_ev = _inv_sqrt_count(ev["side"]) * _inv_sqrt_count(level_bin(ev["level_db"]))
        self.ev_cdf = np.cumsum(w_ev) / w_ev.sum()
        w_hn = _inv_sqrt_count(hn["side"]) if len(hn) else np.ones(0)
        self.hn_cdf = np.cumsum(w_hn) / max(w_hn.sum(), 1e-12)
        self.ev_by_win, self.ev_off = _rows_by_window(ev, self.n_windows)
        self.sp_by_win, self.sp_off = _rows_by_window(sp, self.n_windows)
        # Templates are (raw - VS)[a-8 : a+120]; a neighbouring click in the same
        # channel can fall inside that span. Keep only the template's own run so
        # the clean gap between two clicks is never labelled as a click.
        k = np.arange(self.bank.shape[1])
        own = (k >= 8) & (k < 8 + self.bank_meta["length"].astype(np.int64)[:, None])
        self.bank = np.where(own, self.bank, 0.0).astype(np.float32)
        bk = self.bank.astype(np.float64)
        self.bank_peak = (np.abs(bk[:, 2:] - 2 * bk[:, 1:-1] + bk[:, :-2]).max(axis=1)
                          if len(bk) else np.zeros(0))
        self._ready = True

    # MARK: Crop selection

    def _place(self, rng, a: int, b: int) -> int:
        """Crop start so that [a, b) lands uniformly inside the scored region."""
        m = self.cfg.event_margin
        lo, hi = a - (S1 - m - (b - a)), a - (S0 + m)
        st = int(rng.integers(min(lo, hi), max(lo, hi) + 1))
        return int(np.clip(st, 0, WIN_LEN - T_CROP))

    @staticmethod
    def _draw(cdf: np.ndarray, rng) -> int:
        # cdf[-1] can round to just below 1.0; clamp so searchsorted never runs off the end.
        return min(int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right")), len(cdf) - 1)

    def _fits(self, st: int, a: int, b: int) -> bool:
        return S0 <= a - st and b - st <= S1

    def _pick(self, rng) -> tuple[str, int, int, np.ndarray | None]:
        u = rng.random()
        if u < self.cfg.frac_event and len(self.events):
            for _ in range(8):          # events near window edges can't be centred; re-pick
                r = self.events[self._draw(self.ev_cdf, rng)]
                st = self._place(rng, int(r["a"]), int(r["b"]))
                if self._fits(st, int(r["a"]), int(r["b"])):
                    break
            return "event", int(r["win"]), st, None
        if u < self.cfg.frac_event + self.cfg.frac_hardneg and len(self.hardneg):
            for _ in range(8):
                r = self.hardneg[self._draw(self.hn_cdf, rng)]
                st = self._place(rng, int(r["a"]), int(r["b"]))
                if self._fits(st, int(r["a"]), int(r["b"])):
                    break
            return "hardneg", int(r["win"]), st, r
        w = int(rng.integers(self.n_windows))
        return "uniform", w, int(rng.integers(0, WIN_LEN - T_CROP + 1)), None

    # MARK: Transplant

    def _insert(self, rng, x: np.ndarray, ch: int, pos: int, tmpl_idx: int, scale: float,
                clicks: list) -> bool:
        """Add bank template (and stereo partner) at pos; record clicks; False if rejected."""
        adds = [(ch, pos, self.bank[tmpl_idx])]
        meta = self.bank_meta[tmpl_idx]
        if meta["partner"] >= 0:
            adds.append((1 - ch, pos + int(meta["partner_offset"]), self.bank[int(meta["partner"])]))
        staged = []
        for c, p, tm in adds:
            lo = p - 8                                     # template index 8 == run start
            a, b = max(lo, 0), min(lo + len(tm), T_CROP)
            if b <= a:
                continue
            click = np.zeros(T_CROP, np.float32)
            click[a:b] = scale * tm[a - lo:b - lo]
            if np.abs(x[c, a:b] + click[a:b]).max() > 0.999:
                return False
            staged.append((c, click))
        for c, click in staged:
            x[c] += click
            clicks.append((c, click))
        return True

    def _transplant(self, rng, x: np.ndarray, blocked: np.ndarray, attack: tuple[int, int] | None,
                    clicks: list) -> None:
        cfg = self.cfg
        n = int(rng.integers(1, cfg.transplant_max + 1))
        for k in range(n):
            ch = int(rng.integers(2))
            if k == 0 and attack is not None and rng.random() < cfg.attack_p:
                ch, pos = attack[0], attack[1] + int(rng.integers(-200, 201))
            else:
                free = np.flatnonzero(~blocked[ch, S0:S1 - 128]) + S0
                if len(free) == 0:
                    continue
                if rng.random() < cfg.loud_p:
                    e = np.concatenate([[0.0], np.cumsum(x[ch].astype(np.float64) ** 2)])
                    lo, hi = np.clip(free - 1024, 0, T_CROP), np.clip(free + 1024, 0, T_CROP)
                    rms_db = 10 * np.log10((e[hi] - e[lo]) / np.maximum(hi - lo, 1) + 1e-20)
                    loud = free[rms_db > -30]
                    if len(loud):
                        free = loud
                pos = int(rng.choice(free))
            if not (S0 <= pos < S1 - 128):
                continue
            t = int(rng.integers(len(self.bank)))
            L = int(self.bank_meta[t]["length"])
            host = _curv(x[ch, max(0, pos - 160):min(T_CROP, pos + L + 160)])
            off = pos - max(0, pos - 160)
            around = np.r_[host[:max(1, off - 20)], host[off + L + 20:]]
            c_star = math.exp(rng.uniform(math.log(cfg.contrast_lo), math.log(cfg.contrast_hi)))
            scale = c_star * (np.median(around) + 1e-9) / (self.bank_peak[t] + 1e-12)
            scale *= rng.choice([-1.0, 1.0])
            self._insert(rng, x, ch, pos, t, float(scale), clicks)

    # MARK: Maps

    def _maps(self, x: np.ndarray, lab: np.ndarray, w: int, st: int, clicks: list,
              mono: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        cfg = self.cfg
        target = np.zeros((2, T_CROP), np.uint8)
        weight = np.ones((2, T_CROP), np.float32)
        ignore = np.zeros((2, T_CROP), bool)
        pos_w = np.zeros((2, T_CROP), np.float32)
        events = []

        # Real label-1 runs with their confidence weights.
        ev = self.ev_by_win[self.ev_off[w]:self.ev_off[w + 1]]
        wmap = {(int(r["ch"]), int(r["a"])): float(r["w_evt"]) for r in ev}
        for ch in range(2):
            src = 0 if mono else ch
            l1 = lab[ch] == 1
            for a, b in runs(l1):
                we = wmap.get((src, a + st), 1.0)
                pos_w[ch, a:b] = we
                ignore[ch, max(0, a - cfg.pre_ignore):a] = True
                ignore[ch, b:b + cfg.post_ignore] = True
                if S0 <= a and b <= S1:
                    events.append((ch, a, b, we))
            ignore[ch] |= dilate(lab[ch] == 2, cfg.label2_dilate)
            sp = self.sp_by_win[self.sp_off[w]:self.sp_off[w + 1]]
            for r in sp[sp["ch"] == src]:
                a, b = int(r["a"]) - st, int(r["b"]) - st
                if b > 0 and a < T_CROP:
                    ignore[ch, max(0, a - cfg.suspected_dilate):min(T_CROP, b + cfg.suspected_dilate)] = True

        # Transplanted clicks: positive if audible (contrast >= 8), else ignored.
        for ch, click in clicks:
            nz = np.flatnonzero(np.abs(click) > 0.05 * np.abs(click).max())
            if len(nz) == 0:
                continue
            a, b = int(nz[0]), int(nz[-1]) + 1
            if classical.contrast(x[ch], a, b, 150) >= cfg.contrast_positive:
                pos_w[ch, a:b] = 1.0
                if S0 <= a and b <= S1:
                    events.append((ch, a, b, 1.0))
            else:
                ignore[ch, max(0, a - 4):b + 4] = True

        positive = pos_w > 0
        weight[ignore] = 0.0
        target[positive] = 1
        weight[positive] = pos_w[positive]
        scored = np.zeros(T_CROP, bool)
        scored[S0:S1] = True
        weight[:, ~scored] = 0.0
        bad = dilate(positive | ignore, cfg.neg_guard)
        neg_ok = (weight > 0) & (target == 0) & ~bad
        return target, weight, neg_ok, events

    # MARK: Crop

    def crop(self, rng) -> tuple:
        cfg = self.cfg
        kind, w, st, hn_row = self._pick(rng)
        x = np.asarray(self.audio[w, :, st:st + T_CROP], np.float32).copy()
        lab = np.asarray(self.label[w, :, st:st + T_CROP]).copy()
        mono = rng.random() < cfg.mono_p
        if mono:
            x[1], lab[1] = x[0], lab[0]
        if rng.random() < cfg.polarity_p:
            x = -x
        x = np.clip(x * 10 ** (rng.uniform(*cfg.gain_db) / 20), -1.0, 1.0).astype(np.float32)

        clicks: list = []
        if self.transplant and kind != "event" and rng.random() < cfg.transplant_p:
            blocked = np.zeros((2, T_CROP), bool)
            sp = self.sp_by_win[self.sp_off[w]:self.sp_off[w + 1]]
            for ch in range(2):
                src = 0 if mono else ch
                b = lab[ch] != 0
                for r in sp[sp["ch"] == src]:
                    a0, b0 = int(r["a"]) - st, int(r["b"]) - st
                    b[max(0, a0):max(0, min(T_CROP, b0))] = True
                blocked[ch] = dilate(b, 64)
            attack = None
            if hn_row is not None:
                attack = (int(hn_row["ch"]) if not mono else 0,
                          (int(hn_row["a"]) + int(hn_row["b"])) // 2 - st)
            self._transplant(rng, x, blocked, attack, clicks)
            if mono:   # keep the duplicated channel identical
                x[1] = x[0]
                clicks = [(0, c) for ch, c in clicks if ch == 0]
                clicks += [(1, c) for ch, c in clicks]

        target, weight, neg_ok, events = self._maps(x, lab, w, st, clicks, mono)
        return x, target, weight, neg_ok, events, kind

    def batch(self, rng) -> dict:
        B = self.cfg.batch
        xs, ts, ws, ns, evs, kinds = [], [], [], [], [], []
        for i in range(B):
            x, t, w, n, ev, kind = self.crop(rng)
            xs.append(x); ts.append(t); ws.append(w); ns.append(n); kinds.append(kind)
            evs += [(2 * i + ch, a, b, we) for ch, a, b, we in ev]
        ev = np.array(evs, np.float32).reshape(-1, 4)
        return {
            "audio": torch.from_numpy(np.stack(xs)),                         # (B, 2, T)
            "target": torch.from_numpy(np.concatenate(ts)),                  # (2B, T)
            "weight": torch.from_numpy(np.concatenate(ws)),
            "neg_ok": torch.from_numpy(np.concatenate(ns)),
            "ev_idx": torch.from_numpy(ev[:, :3].astype(np.int64)),          # (E, 3) seq, a, b
            "ev_w": torch.from_numpy(ev[:, 3].copy()),
            "n_transplant_crops": sum(k != "event" for k in kinds),
        }

    def __iter__(self):
        self._load()
        info = get_worker_info()
        wid = info.id if info else 0
        step = 0
        while True:
            rng = np.random.default_rng([self.seed, wid, step])
            yield self.batch(rng)
            step += 1
