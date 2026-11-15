"""One-time training tables for the neural click detector (spec section 4).

Reads the memmapped split ``<data>/mm/<split>`` and writes ``<data>/nn/<split>/``::

    sides.json      sorted unique shard names; side id = position in this list
    events.npy      one row per label-1 run (EVENT_DTYPE)
    hardneg.npy     classical gate-off regions away from labels (REGION_DTYPE)
    suspected.npy   ... of those, the sharp, strong ones: likely clicks
                    VinylStudio missed (ignored in training)
    bank.npy        float32 (M, 128) residual templates (raw - VS)[a-8 : a+120]
    bank_meta.npy   BANK_META_DTYPE, one row per template
    summary.json    counts and the pre-training sanity gate

VS is the VinylStudio-repaired audio (raw with label-1 samples replaced by
``repl``), so the residual ``raw - VS`` is exactly 0 outside label-1 runs.
All per-window windows of context ([a-1024, b+1024) etc.) are clipped to the
window. Work is split into chunks of windows over a process pool; each
worker opens the memmaps itself.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pressing_tools import classical
from pressing_tools.memmap import open_split


# MARK: - Formats and constants


EVENT_DTYPE = np.dtype([("win", "<i4"), ("ch", "i1"), ("a", "<i4"), ("b", "<i4"), ("side", "<i4"),
                        ("level_db", "<f4"), ("w_evt", "<f4"), ("partner", "<i4")])
REGION_DTYPE = np.dtype([("win", "<i4"), ("ch", "i1"), ("a", "<i4"), ("b", "<i4"),
                         ("peak", "<f4"), ("contrast", "<f4"), ("side", "<i4")])
BANK_META_DTYPE = np.dtype([("event", "<i4"), ("length", "<i2"), ("partner", "<i4"),
                            ("partner_offset", "<i2"), ("level_db", "<f4")])

NO_PARTNER = np.iinfo(np.int32).min   # events.partner sentinel
SR = 48000
TOL = 16                 # region-to-label slack (same as evaluate.TOL)
SUSPECT_LABEL_GAP = 64   # suspected regions need no label within this many samples
EDGE = 2400              # regions must lie inside [EDGE, L - EDGE)
PARTNER_SLACK = 8        # stereo partner starts within +-this many samples
LEVEL_CTX = 1024         # level_db context each side
TEMPLATE_PRE, TEMPLATE_LEN = 8, 128
BANK_MAX_LEN = 96
# Clicks are sharp, not loud: peak |raw - VS| is only ~1x the local full-band
# RMS (median), so amplitude ratios misjudge them (the spec's 4x-RMS gate kept
# 6% of events and floored 72% of weights). Curvature contrast -- the same
# measure as classical.contrast and the transplant audibility rule -- is ~23x
# median for label-1 runs.
BANK_MIN_CONTRAST = 8.0      # same as the transplant audibility rule
W_EVT_REF = 2.0              # w_evt = clip(log2(contrast / W_EVT_REF) / 3, 0.3, 1): 1.0 at 16, 0.67 at 8
LEVEL_BINS = [("<-45", -math.inf, -45.0), ("-45..-30", -45.0, -30.0),
              ("-30..-18", -30.0, -18.0), (">-18", -18.0, math.inf)]
GATE_FRACTION = 0.4


@dataclass
class PrepConfig:
    threshold: float = 4.0        # classical trigger threshold (gate off)
    suspect_peak: float = 8.0     # suspected: peak score >  this ...
    suspect_contrast: float = 20.0  # ... and contrast > this (spec: tighten to 30 if gate trips)
    contrast_ctx: int = 150


# MARK: - Per-window helpers


def event_contrast(x: np.ndarray, a: int, b: int, ctx: int = 150) -> float:
    """classical.contrast, safe at window edges."""
    a, b = max(1, a), min(len(x) - 2, b)
    return classical.contrast(x, a, b, ctx) if b > a else 0.0


def w_evt_from_contrast(c: float) -> float:
    return min(1.0, max(0.3, math.log2(c / W_EVT_REF) / 3)) if c > 0 else 0.3


def _rms(x: np.ndarray, lo: int, hi: int) -> float:
    lo, hi = max(0, lo), min(len(x), hi)
    return float(np.sqrt(np.mean(x[lo:hi] ** 2))) if hi > lo else 0.0


def _near(cs: np.ndarray, lo: int, hi: int) -> bool:
    """Any nonzero label in [lo, hi) (clipped)? ``cs`` = r_[0, cumsum(label != 0)]."""
    lo, hi = max(0, lo), min(len(cs) - 1, hi)
    return hi > lo and cs[hi] > cs[lo]


def _partner(a: int, b: int, other: list[tuple[int, int]]) -> int:
    """Offset (start - a) of the overlapping other-channel run whose start is
    closest to a within +-PARTNER_SLACK, else NO_PARTNER."""
    best = None
    for a2, b2 in other:
        if abs(a2 - a) <= PARTNER_SLACK and a2 < b and b2 > a:
            if best is None or abs(a2 - a) < abs(best - a):
                best = a2
    return NO_PARTNER if best is None else best - a


def window_vs(raw: np.ndarray, lab: np.ndarray, repl: np.ndarray) -> np.ndarray:
    """VinylStudio-repaired audio for one (C, L) window; repl is in np.nonzero order."""
    vs = raw.copy()
    m1 = lab == 1
    if int(m1.sum()) != len(repl):
        raise ValueError(f"repl length {len(repl)} != label-1 count {int(m1.sum())}")
    vs[m1] = repl
    return vs


def process_window(raw: np.ndarray, lab: np.ndarray, vs: np.ndarray, win: int, side: int,
                   cfg: PrepConfig) -> tuple[list, list, list, list, list]:
    """All tables for one stereo window (float64 raw / vs, uint8 labels).

    Returns (events, hardneg, suspected, bank templates, bank (local event, length) rows);
    the bank rows index into this window's events list."""
    C, L = raw.shape
    res = raw - vs
    runs1 = [classical.runs(lab[c] == 1) for c in range(C)]
    events, tpls, bank_rows = [], [], []
    for c in range(C):
        other = runs1[1 - c] if C == 2 else []
        for a, b in runs1[c]:
            peak = float(np.abs(res[c, a:b]).max())
            level = _rms(vs[c], a - LEVEL_CTX, b + LEVEL_CTX)
            level_db = 20 * math.log10(max(level, 1e-10))
            con = event_contrast(raw[c], a, b, cfg.contrast_ctx)
            w_evt = w_evt_from_contrast(con)
            events.append((win, c, a, b, side, level_db, w_evt, _partner(a, b, other)))
            if b - a <= BANK_MAX_LEN and con >= BANK_MIN_CONTRAST and peak > 0:
                t = np.zeros(TEMPLATE_LEN, np.float32)
                lo, hi = a - TEMPLATE_PRE, a - TEMPLATE_PRE + TEMPLATE_LEN
                t[max(0, lo) - lo:min(L, hi) - lo] = res[c, max(0, lo):min(L, hi)]
                tpls.append(t)
                bank_rows.append(len(events) - 1)

    hard, sus = [], []
    ccfg = classical.ClassicalConfig(min_contrast=0.0)
    for c in range(C):
        x = raw[c]
        cs = np.r_[0, np.cumsum(lab[c] != 0)]
        s = classical.scores(x, ccfg)
        mask = classical.regions_from_scores(s, ccfg, threshold=cfg.threshold, x=None)
        for a, b in classical.runs(mask):
            if a < EDGE or b > L - EDGE or _near(cs, a - TOL, b + TOL):
                continue
            peak = float(s[a:b].max())
            con = classical.contrast(x, a, b, cfg.contrast_ctx)
            row = (win, c, a, b, peak, con, side)
            if peak > cfg.suspect_peak and con > cfg.suspect_contrast and \
                    not _near(cs, a - SUSPECT_LABEL_GAP, b + SUSPECT_LABEL_GAP):
                sus.append(row)
            else:
                hard.append(row)
    return events, hard, sus, tpls, bank_rows


# MARK: - Chunk worker


@functools.lru_cache(maxsize=2)
def _open(root: str, split: str) -> dict:
    return open_split(Path(root), split)


def process_chunk(root: str, split: str, w0: int, w1: int, sides: list[int], cfg: dict) -> dict:
    """Windows [w0, w1) -> chunk-local tables (event / bank indices local to the chunk)."""
    mm = _open(root, split)
    pc = PrepConfig(**cfg)
    ev, hard, sus, tpls, brow = [], [], [], [], []
    for i, w in enumerate(range(w0, w1)):
        raw = np.asarray(mm["audio"][w], np.float64)
        lab = np.asarray(mm["label"][w])
        repl = np.asarray(mm["repl"][mm["repl_off"][w]:mm["repl_off"][w + 1]], np.float64)
        e, h, s, t, br = process_window(raw, lab, window_vs(raw, lab, repl), w, sides[i], pc)
        brow += [len(ev) + j for j in br]
        ev += e; hard += h; sus += s; tpls += t
    events = np.array(ev, EVENT_DTYPE) if ev else np.zeros(0, EVENT_DTYPE)
    return {
        "events": events,
        "hardneg": np.array(hard, REGION_DTYPE) if hard else np.zeros(0, REGION_DTYPE),
        "suspected": np.array(sus, REGION_DTYPE) if sus else np.zeros(0, REGION_DTYPE),
        "bank": np.stack(tpls) if tpls else np.zeros((0, TEMPLATE_LEN), np.float32),
        "bank_event": np.array(brow, np.int64),
    }


# MARK: - Assembly


def bank_meta(events: np.ndarray, bank_event: np.ndarray) -> np.ndarray:
    """Metadata rows for the bank; partner = bank row of the referenced partner event."""
    meta = np.zeros(len(bank_event), BANK_META_DTYPE)
    ev = events[bank_event]
    meta["event"] = bank_event
    meta["length"] = ev["b"] - ev["a"]
    meta["level_db"] = ev["level_db"]
    has = ev["partner"] != NO_PARTNER
    meta["partner_offset"] = np.where(has, ev["partner"], 0)
    key_to_bank = {(int(e["win"]), int(e["ch"]), int(e["a"])): m for m, e in enumerate(ev)}
    meta["partner"] = [
        key_to_bank.get((int(e["win"]), 1 - int(e["ch"]), int(e["a"]) + int(e["partner"])), -1)
        if h else -1 for e, h in zip(ev, has)]
    return meta


def _bins(level_db: np.ndarray) -> dict:
    return {name: int(((level_db >= lo) & (level_db < hi)).sum()) for name, lo, hi in LEVEL_BINS}


def summarize(split: str, n_win: int, L: int, C: int, events: np.ndarray, hard: np.ndarray,
              sus: np.ndarray, meta: np.ndarray, cfg: PrepConfig, wall_s: float) -> dict:
    ch_min = n_win * C * L / SR / 60
    lens = (events["b"] - events["a"]).astype(np.float64)
    frac = len(sus) / max(1, len(sus) + len(hard))
    pct = (lambda q: float(np.percentile(lens, q))) if len(lens) else (lambda q: float("nan"))
    return {
        "split": split,
        "windows": n_win,
        "channel_minutes": ch_min,
        "events": {
            "count": len(events),
            "level_bins": _bins(events["level_db"]),
            "len_p50": pct(50), "len_p90": pct(90),
            "len_mean": float(lens.mean()) if len(lens) else float("nan"),
            "len_max": int(lens.max()) if len(lens) else 0,
            "w_evt_mean": float(events["w_evt"].mean()) if len(events) else float("nan"),
            "w_evt_floor_frac": float((events["w_evt"] <= 0.3 + 1e-6).mean()) if len(events) else float("nan"),
            "with_partner_frac": float((events["partner"] != NO_PARTNER).mean()) if len(events) else float("nan"),
        },
        "bank": {
            "count": len(meta),
            "level_bins": _bins(meta["level_db"]),
            "paired_frac": float((meta["partner"] >= 0).mean()) if len(meta) else float("nan"),
            "mbytes": len(meta) * TEMPLATE_LEN * 4 / 1e6,
        },
        "suspected": {"count": len(sus), "per_channel_minute": len(sus) / max(ch_min, 1e-9)},
        "hardneg": {"count": len(hard), "per_channel_minute": len(hard) / max(ch_min, 1e-9)},
        "suspected_fraction": frac,
        "gate": {"limit": GATE_FRACTION, "tripped": frac > GATE_FRACTION,
                 "advice": ("suspected > 40% of gate-off false alarms: rerun with --suspect-contrast 30"
                            if frac > GATE_FRACTION else "ok")},
        "config": asdict(cfg),
        "wall_s": wall_s,
    }


def format_summary(s: dict) -> str:
    e, b = s["events"], s["bank"]
    lines = [
        f"== nn prep {s['split']}: {s['windows']} windows, {s['channel_minutes']:.1f} channel-minutes, "
        f"{s['wall_s']:.0f} s",
        f"events     {e['count']:>8}  len p50 {e['len_p50']:.0f} p90 {e['len_p90']:.0f} "
        f"mean {e['len_mean']:.1f} max {e['len_max']}  w_evt mean {e['w_evt_mean']:.2f} "
        f"(at floor {e['w_evt_floor_frac']:.1%})  stereo-paired {e['with_partner_frac']:.1%}",
        f"bank       {b['count']:>8}  ({b['mbytes']:.0f} MB, partner in bank {b['paired_frac']:.1%})",
        "level bin (dBFS)   " + "  ".join(f"{k:>9}" for k in e["level_bins"]),
        "  events           " + "  ".join(f"{v:>9}" for v in e["level_bins"].values()),
        "  bank             " + "  ".join(f"{v:>9}" for v in b["level_bins"].values()),
        f"suspected  {s['suspected']['count']:>8}  ({s['suspected']['per_channel_minute']:.2f} / channel-minute)",
        f"hardneg    {s['hardneg']['count']:>8}  ({s['hardneg']['per_channel_minute']:.2f} / channel-minute)",
        f"suspected / (suspected + hardneg) = {s['suspected_fraction']:.1%}  "
        f"gate (> {s['gate']['limit']:.0%}): {'TRIPPED' if s['gate']['tripped'] else 'ok'}"
        + (f" -- {s['gate']['advice']}" if s["gate"]["tripped"] else ""),
    ]
    return "\n".join(lines)


def prep_split(root: Path, split: str, out: Path | None = None, jobs: int = 16, chunk: int = 32,
               cfg: PrepConfig | None = None, progress: bool = False) -> dict:
    cfg = cfg or PrepConfig()
    t0 = time.time()
    mm = open_split(root, split)
    n_win, C, L = mm["audio"].shape
    sides = sorted({r["shard"] for r in mm["index"]})
    side_of = {s: i for i, s in enumerate(sides)}
    win_side = [side_of[r["shard"]] for r in mm["index"]]
    del mm
    bounds = [(w0, min(n_win, w0 + chunk)) for w0 in range(0, n_win, chunk)]
    args = [(str(root), split, w0, w1, win_side[w0:w1], asdict(cfg)) for w0, w1 in bounds]
    parts: list[dict | None] = [None] * len(args)
    if jobs <= 1:
        for i, a in enumerate(args):
            parts[i] = process_chunk(*a)
    else:
        with ProcessPoolExecutor(jobs) as ex:
            futs = {ex.submit(process_chunk, *a): i for i, a in enumerate(args)}
            for k, f in enumerate(as_completed(futs), 1):
                parts[futs[f]] = f.result()
                if progress and (k % max(1, len(args) // 20) == 0 or k == len(args)):
                    print(f"  {split}: {k}/{len(args)} chunks, {time.time() - t0:.0f} s", flush=True)

    # Chunk-local -> global indices (parts are in window order).
    ev_off = np.cumsum([0] + [len(p["events"]) for p in parts])
    events = np.concatenate([p["events"] for p in parts])
    hard = np.concatenate([p["hardneg"] for p in parts])
    sus = np.concatenate([p["suspected"] for p in parts])
    bank = np.concatenate([p["bank"] for p in parts]).astype(np.float32)
    bank_event = np.concatenate([p["bank_event"] + ev_off[i] for i, p in enumerate(parts)]).astype(np.int64)
    meta = bank_meta(events, bank_event)

    out = out or root / "nn" / split
    out.mkdir(parents=True, exist_ok=True)
    (out / "sides.json").write_text(json.dumps(sides))
    np.save(out / "events.npy", events)
    np.save(out / "hardneg.npy", hard)
    np.save(out / "suspected.npy", sus)
    np.save(out / "bank.npy", bank)
    np.save(out / "bank_meta.npy", meta)
    summary = summarize(split, n_win, L, C, events, hard, sus, meta, cfg, time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# MARK: - CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-nn-prep", description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, default=None, help="default <data>/nn/<split>")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32, help="windows per worker task")
    ap.add_argument("--suspect-contrast", type=float, default=PrepConfig.suspect_contrast)
    ap.add_argument("--suspect-peak", type=float, default=PrepConfig.suspect_peak)
    a = ap.parse_args(argv)
    cfg = PrepConfig(suspect_peak=a.suspect_peak, suspect_contrast=a.suspect_contrast)
    s = prep_split(a.data, a.split, a.out, a.jobs, a.chunk, cfg, progress=True)
    print(format_summary(s), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
