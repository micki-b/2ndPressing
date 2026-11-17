"""Synthetic click benchmark with exact truth (spec section 11).

Hosts are windows of one split (``data/v1/mm/<split>``); click templates come
only from that split's own bank (``data/v1/nn/<split>/bank.npy`` +
``bank_meta.npy``, written by ``pressing-nn-prep``). If the prep bank is
missing, a bank is built on the fly from the split's memmap with the same rule
(label-1 runs of length <= 96 whose residual peak is >= 4x the local RMS of
the VinylStudio-repaired audio) and the log says so.

Per level bin (< -45, -45..-30, -30..-18, > -18 dBFS local RMS over +-1024
samples of the VinylStudio-repaired host) up to 400 windows get about 3 clicks
per channel-second at positions in that bin: >= 256 samples from any original
nonzero label (either channel), >= 512 samples apart, outside EDGE. Each click
is scaled to a target contrast c* (log-uniform in [8, 100]) using the
``classical.contrast`` definition (context 150, guard 20), with a random sign;
its stereo partner (if the template has one in the bank) goes into the other
channel at ``partner_offset`` with the same scale. Insertions that would make
|x| > 0.999 are rejected and retried elsewhere with the same c* and sign (so
the realised c* stays log-uniform in loud bins); a c* that clips on
``clip_retries`` placements is dropped. synth.json records clip rejections,
drops and realised c* quantiles against the log-uniform expectation.

Output, one directory per bin, in the normal dataset shard format::

    data/v1/synth/<split>_lvl<k>/<source shard>.npz
        audio  f16 (W, 2, L)  host + injected clicks
        label  u8  (W, 2, L)  original labels 1/2 -> 2 (don't care);
                              injected clicks with measured contrast >= 8 -> 1
                              over their support, weaker ones -> 2
        repl   f16 (K,)       clean host samples at label-1 positions
                              (np.nonzero order), so dataset.repaired()
                              rebuilds the host exactly at every label-1 sample
        starts, sr_native, sr as in the source shard
        src_window i8 (W,)    global memmap window of each host
        clicks                one row per injected click (CLICK_DTYPE)
    data/v1/synth/manifest.jsonl   one line per shard; ``shard`` is relative
                                   to data/v1/synth/ (e.g. "val_lvl2/X.npz")

The host is the raw window (its own clicks stay in and are don't-care), so the
unchanged harness (``pressing-eval --data ../data/v1 --split synth/val_lvl2``)
reports recall against exact truth, spike ratio and repair SNR vs the host.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pressing_tools import classical
from pressing_tools.dataset import LABEL_CLEAN, LABEL_DETECTED, LABEL_REPAIRED
from pressing_tools.evaluate import EDGE
from pressing_tools.memmap import open_split

# MARK: - Constants and formats

SR = 48000
LEVEL_EDGES = (-45.0, -30.0, -18.0)          # bin k: LEVEL_EDGES[k-1] <= dB < LEVEL_EDGES[k]
LEVEL_NAMES = ("<-45", "-45..-30", "-30..-18", ">-18")
LEVEL_HALF = 1024                            # level context each side
TEMPLATE_LEN = 128
PRE_ROLL = 8                                 # template sample index of the event start a
BANK_MAX_LEN = 96
BANK_MIN_RATIO = 4.0
BANK_RMS_CTX = 256
PARTNER_SLACK = 8
NO_PARTNER = np.iinfo(np.int32).min

BANK_META_DTYPE = np.dtype([("event", "<i4"), ("length", "<i2"), ("partner", "<i4"),
                            ("partner_offset", "<i2"), ("level_db", "<f4")])
CLICK_DTYPE = np.dtype([("win", "<i4"), ("ch", "i1"), ("a", "<i4"), ("b", "<i4"), ("pos", "<i4"),
                        ("bank", "<i4"), ("partner_of", "<i4"), ("scale", "<f4"),
                        ("c_target", "<f4"), ("contrast", "<f4"), ("level_db", "<f4"),
                        ("label", "u1")])


@dataclass
class SynthConfig:
    windows: int = 400            # hosts per level bin (at most)
    rate: float = 3.0             # injected clicks per channel-second (partners count)
    min_gap: int = 512            # injected clicks at least this far apart (per channel)
    label_clear: int = 256        # template extent at least this far from any original label
    c_min: float = 8.0            # target contrast range (log-uniform)
    c_max: float = 100.0
    label_contrast: float = 8.0   # measured contrast >= this -> label 1, else 2
    support_frac: float = 0.05    # click support: |click| > this fraction of its peak
    clip: float = 0.999           # reject insertions that would exceed this
    clip_retries: int = 50        # placements tried for one (c*, sign) before it is dropped
    ctx: int = 150                # classical.contrast context ...
    guard: int = 20               # ... and guard
    min_candidates: int = 2048    # a window qualifies for a bin if a channel has this many positions in it
    attempts: int = 400           # placement attempts per window
    seed: int = 0


def level_bin(db: np.ndarray | float) -> np.ndarray | int:
    return np.digitize(db, LEVEL_EDGES)


# MARK: - Window access


def window_arrays(mm: dict, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(raw, VinylStudio-repaired, label) for memmap window w; float64, float64, uint8."""
    raw = np.asarray(mm["audio"][w], np.float64)
    lab = np.asarray(mm["label"][w])
    vs = raw.copy()
    o0, o1 = int(mm["repl_off"][w]), int(mm["repl_off"][w + 1])
    vs[lab == LABEL_REPAIRED] = np.asarray(mm["repl"][o0:o1], np.float64)
    return raw, vs, lab


def level_db_curve(v: np.ndarray, half: int = LEVEL_HALF) -> np.ndarray:
    """20*log10 RMS of v over [n - half, n + half) clipped to the window, for every n."""
    n = len(v)
    cs = np.r_[0.0, np.cumsum(v * v)]
    i = np.arange(n)
    lo, hi = np.maximum(0, i - half), np.minimum(n, i + half)
    return 10 * np.log10((cs[hi] - cs[lo]) / (hi - lo) + 1e-24)


def _rms(v: np.ndarray, lo: int, hi: int) -> float:
    seg = v[max(0, lo):min(len(v), hi)]
    return float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0


def placeable(lab: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """Positions p (template start a) whose template [p-8, p+120) lies inside
    [EDGE, L-EDGE) and is >= label_clear from any nonzero label in any channel."""
    L = lab.shape[-1]
    cs = np.r_[0, np.cumsum((lab != LABEL_CLEAN).any(0))]
    p = np.arange(L)
    t0, t1 = p - PRE_ROLL, p - PRE_ROLL + TEMPLATE_LEN
    lo, hi = np.clip(t0 - cfg.label_clear, 0, L), np.clip(t1 + cfg.label_clear, 0, L)
    return (cs[hi] - cs[lo] == 0) & (t0 >= EDGE) & (t1 <= L - EDGE)


# MARK: - Template bank


def _partner_offset(a: int, b: int, other: list[tuple[int, int]]) -> int:
    best = NO_PARTNER
    for a2, b2 in other:
        if a2 < b and b2 > a and abs(a2 - a) <= PARTNER_SLACK and \
                (best == NO_PARTNER or abs(a2 - a) < abs(best)):
            best = a2 - a
    return best


def build_bank(mm: dict, windows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Minimal on-the-fly bank with the prep rule (bank.npy / bank_meta.npy formats).

    meta.event is the event's position in (window, channel, start) scan order,
    which is events.npy row order when prep scans the same way."""
    temps, meta = [], []
    ev = 0
    windows = np.arange(len(mm["audio"])) if windows is None else windows
    for w in windows:
        raw, vs, lab = window_arrays(mm, int(w))
        C, L = raw.shape
        r = raw - vs
        runs = [classical.runs(lab[c] == LABEL_REPAIRED) for c in range(C)]
        row_of: dict[tuple[int, int], int] = {}
        pending = []
        for c in range(C):
            for a, b in runs[c]:
                off = _partner_offset(a, b, runs[1 - c]) if C == 2 else NO_PARTNER
                peak = float(np.abs(r[c, a:b]).max())
                if b - a <= BANK_MAX_LEN and peak >= BANK_MIN_RATIO * _rms(vs[c], a - BANK_RMS_CTX, b + BANK_RMS_CTX):
                    t = np.zeros(TEMPLATE_LEN, np.float32)
                    s0, s1 = max(0, a - PRE_ROLL), min(L, a - PRE_ROLL + TEMPLATE_LEN)
                    t[s0 - (a - PRE_ROLL):s1 - (a - PRE_ROLL)] = r[c, s0:s1]
                    row_of[(c, a)] = len(temps)
                    temps.append(t)
                    lvl = 20 * math.log10(_rms(vs[c], a - LEVEL_HALF, b + LEVEL_HALF) + 1e-12)
                    pending.append((c, a, off))
                    meta.append((ev, b - a, -1, 0 if off == NO_PARTNER else off, lvl))
                ev += 1
        for c, a, off in pending:
            if off != NO_PARTNER:
                i = row_of[(c, a)]
                ev_, n_, _, po, lv = meta[i]
                meta[i] = (ev_, n_, row_of.get((1 - c, a + off), -1), po, lv)
    bank = np.stack(temps) if temps else np.zeros((0, TEMPLATE_LEN), np.float32)
    return bank, np.array(meta, BANK_META_DTYPE)


def load_bank(data: Path, split: str, mm: dict, log=print) -> tuple[np.ndarray, np.ndarray, str]:
    d = data / "nn" / split
    if (d / "bank.npy").exists() and (d / "bank_meta.npy").exists():
        bank = np.load(d / "bank.npy").astype(np.float32)
        meta = np.load(d / "bank_meta.npy")
        source = f"prep bank {d / 'bank.npy'}"
    else:
        t0 = time.time()
        bank, meta = build_bank(mm)
        source = f"built on the fly from mm/{split} ({d / 'bank.npy'} missing; {time.time() - t0:.0f} s)"
    if len(bank) != len(meta):
        raise ValueError(f"bank has {len(bank)} rows but bank_meta {len(meta)}")
    log(f"bank: {len(bank)} templates, {int((meta['partner'] >= 0).sum())} with a stereo partner -- {source}")
    return bank, meta, source


def template_support(t: np.ndarray, frac: float) -> tuple[int, int] | None:
    """[first, last+1) where |t| > frac * peak, or None for an all-zero template."""
    pk = float(np.abs(t).max())
    if pk <= 0:
        return None
    idx = np.flatnonzero(np.abs(t) > frac * pk)
    return int(idx[0]), int(idx[-1]) + 1


def peak_curvature(t: np.ndarray, s0: int, s1: int) -> float:
    """max |t''| over [s0, s1) for the template placed in silence."""
    tp = np.r_[0.0, np.asarray(t, np.float64), 0.0]
    d = np.abs(tp[2:] - 2 * tp[1:-1] + tp[:-2])
    return float(d[s0:s1].max())


def median_curvature_around(x: np.ndarray, a: int, b: int, ctx: int = 150, guard: int = 20) -> float:
    """The denominator of classical.contrast(x, a, b, ctx, guard), without the 1e-12."""
    lo, hi = max(1, a - ctx), min(len(x) - 1, b + ctx)
    d = np.abs(x[lo + 1:hi + 1] - 2 * x[lo:hi] + x[lo - 1:hi - 1])
    ia, ib = a - lo, b - lo
    around = np.r_[d[:max(1, ia - guard)], d[min(len(d) - 1, ib + guard):]]
    return float(np.median(around))


# MARK: - Injection


def _f16(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float16).astype(np.float64)


def inject_window(raw: np.ndarray, vs: np.ndarray, lab: np.ndarray, k: int, bank: np.ndarray,
                  meta: np.ndarray, rng: np.random.Generator, cfg: SynthConfig,
                  stats: dict | None = None) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Inject clicks at level-bin-k positions of one window.

    The target contrast c* and sign are drawn once per click and kept until
    that click is placed: a placement that would clip (|x| > cfg.clip) retries
    another position/template with the same c*, and only after
    cfg.clip_retries clip rejections is the click dropped. Redrawing c* on a
    clip rejection would bias the loud bins against high-contrast clicks.
    If ``stats`` is given, 'clip_rejects' and 'dropped' are added to it.

    Returns (audio float64 (C, L) exactly representable in float16, label uint8
    (C, L), click rows (CLICK_DTYPE fields without 'win'; partner_of indexes
    this list, -1 for primaries))."""
    C, L = raw.shape
    y = _f16(raw)
    out_lab = np.where(lab != LABEL_CLEAN, LABEL_DETECTED, LABEL_CLEAN).astype(np.uint8)
    base = placeable(lab, cfg)
    lvl = np.stack([level_db_curve(vs[c]) for c in range(C)])
    inbin = base[None] & (level_bin(lvl) == k)
    free = np.repeat(base[None], C, 0)
    target = int(round(cfg.rate * (L - 2 * EDGE) / SR))
    count = np.zeros(C, int)
    exhausted = np.zeros(C, bool)
    clicks: list[tuple] = []
    log_c = (math.log(cfg.c_min), math.log(cfg.c_max))
    clip_rejects = dropped = 0
    pending: tuple[float, float] | None = None   # (c*, sign) of the click being placed
    clip_fails = 0

    for _ in range(cfg.attempts):
        open_ch = [c for c in range(C) if count[c] < target and not exhausted[c]]
        if not open_ch:
            break
        if pending is None:
            pending = (float(math.exp(rng.uniform(*log_c))), float(rng.choice([-1.0, 1.0])))
            clip_fails = 0
        c_star, sign = pending
        c = min(open_ch, key=lambda ch: (count[ch], rng.random()))
        cand = np.flatnonzero(inbin[c] & free[c])
        if not len(cand):
            exhausted[c] = True
            continue
        p = int(rng.choice(cand))
        row = int(rng.integers(len(bank)))
        parts = [(c, p, row)]
        pr = int(meta["partner"][row])
        if C == 2 and 0 <= pr < len(bank):
            q = p + int(meta["partner_offset"][row])
            if not (0 <= q < L and free[1 - c, q]):
                continue
            parts.append((1 - c, q, pr))

        sup = [template_support(bank[r], cfg.support_frac) for _, _, r in parts]
        if sup[0] is None or any(s is None for s in sup):
            continue
        s0, s1 = sup[0]
        a0, b0 = p - PRE_ROLL + s0, p - PRE_ROLL + s1
        med = median_curvature_around(y[c], a0, b0, cfg.ctx, cfg.guard)
        pc = peak_curvature(bank[row], s0, s1)
        if med <= 0 or pc <= 0:
            continue
        scale = sign * c_star * med / pc

        new = []
        for ch, pos, r in parts:
            t0 = pos - PRE_ROLL
            seg = y[ch, t0:t0 + TEMPLATE_LEN] + scale * bank[r].astype(np.float64)
            if np.abs(seg).max() > cfg.clip:
                break
            new.append((ch, t0, _f16(seg)))
        if len(new) != len(parts):
            clip_rejects += 1
            clip_fails += 1
            if clip_fails >= cfg.clip_retries:
                dropped += 1
                pending = None
            continue
        pending = None
        for ch, t0, seg in new:
            y[ch, t0:t0 + TEMPLATE_LEN] = seg
        first = len(clicks)
        for i, ((ch, pos, r), (u0, u1)) in enumerate(zip(parts, sup)):
            a, b = pos - PRE_ROLL + u0, pos - PRE_ROLL + u1
            cm = classical.contrast(y[ch], a, b, cfg.ctx, cfg.guard)
            lbl = LABEL_REPAIRED if cm >= cfg.label_contrast else LABEL_DETECTED
            out_lab[ch, a:b] = lbl
            clicks.append((ch, a, b, pos, r, -1 if i == 0 else first, scale,
                           c_star if i == 0 else np.nan, cm, float(lvl[ch, pos]), lbl))
            free[ch, max(0, pos - cfg.min_gap + 1):pos + cfg.min_gap] = False
            count[ch] += 1
    if pending is not None and clip_fails:
        dropped += 1          # window ran out of room/attempts while this c* kept clipping
    if stats is not None:
        stats["clip_rejects"] = stats.get("clip_rejects", 0) + clip_rejects
        stats["dropped"] = stats.get("dropped", 0) + dropped
    return y, out_lab, clicks


# MARK: - Host selection


def bin_candidate_counts(mm: dict, cfg: SynthConfig) -> np.ndarray:
    """(N, 4): per window, the max over channels of placeable positions in each level bin."""
    N = len(mm["audio"])
    out = np.zeros((N, len(LEVEL_NAMES)), np.int64)
    for w in range(N):
        _, vs, lab = window_arrays(mm, w)
        base = placeable(lab, cfg)
        for c in range(vs.shape[0]):
            k = level_bin(level_db_curve(vs[c]))[base]
            out[w] = np.maximum(out[w], np.bincount(k, minlength=len(LEVEL_NAMES)))
    return out


def select_hosts(counts: np.ndarray, k: int, cfg: SynthConfig) -> np.ndarray:
    elig = np.flatnonzero(counts[:, k] >= cfg.min_candidates)
    rng = np.random.default_rng([cfg.seed, 1000 + k])
    n = min(cfg.windows, len(elig))
    return np.sort(rng.choice(elig, n, replace=False)) if n else elig


# MARK: - Writing


def _read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _source_starts(data: Path, split: str, shard: str) -> np.ndarray | None:
    p = data / split / shard
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["starts"]


def c_target_summary(c: np.ndarray, cfg: SynthConfig) -> dict:
    """Realised primary c* quantiles and tail fractions next to their log-uniform expectation."""
    c = np.asarray(c, np.float64)
    c = c[np.isfinite(c)]
    span = math.log(cfg.c_max / cfg.c_min)
    exp_q = {q: cfg.c_min * math.exp(q * span) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out: dict = {"n": int(len(c))}
    for q, e in exp_q.items():
        out[f"q{int(q * 100)}"] = float(np.quantile(c, q)) if len(c) else float("nan")
        out[f"q{int(q * 100)}_expected"] = e
    for t in (50.0, 80.0):
        out[f"frac_gt{int(t)}"] = float((c > t).mean()) if len(c) else float("nan")
        out[f"frac_gt{int(t)}_expected"] = math.log(cfg.c_max / t) / span
    return out


def synthesize_bin(data: Path, split: str, mm: dict, k: int, hosts: np.ndarray, bank: np.ndarray,
                   meta: np.ndarray, cfg: SynthConfig, src_manifest: dict, out_root: Path) -> dict:
    """Write <out_root>/<split>_lvl<k>/*.npz; return stats and manifest lines."""
    name = f"{split}_lvl{k}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npz"):
        old.unlink()
    by_shard: dict[str, list[int]] = defaultdict(list)
    for w in hosts:
        by_shard[mm["index"][int(w)]["shard"]].append(int(w))

    lines, all_clicks = [], []
    inj: dict[str, int] = {"clip_rejects": 0, "dropped": 0}
    for shard in sorted(by_shard):
        ws = by_shard[shard]
        audio, label, repl, rows = [], [], [], []
        for j, w in enumerate(ws):
            raw, vs, lab = window_arrays(mm, w)
            rng = np.random.default_rng([cfg.seed, k, w])
            y, ol, cl = inject_window(raw, vs, lab, k, bank, meta, rng, cfg, inj)
            audio.append(y.astype(np.float16))
            label.append(ol)
            repl.append(_f16(raw)[ol == LABEL_REPAIRED].astype(np.float16))
            base = len(rows)                  # partner_of -> row in this shard's clicks
            rows += [(j, *r[:5], r[5] + base if r[5] >= 0 else -1, *r[6:]) for r in cl]
        src = src_manifest.get(f"{split}/{shard}", {})
        starts = _source_starts(data, split, shard)
        win_in_src = [mm["index"][w]["window"] for w in ws]
        starts = (starts[win_in_src] if starts is not None else np.zeros(len(ws))).astype(np.int64)
        clicks = np.array(rows, CLICK_DTYPE)
        np.savez_compressed(out_dir / shard, audio=np.stack(audio), label=np.stack(label),
                            repl=np.concatenate(repl), starts=starts,
                            sr_native=int(src.get("sr_native", SR)), sr=SR,
                            src_window=np.array(ws, np.int64), clicks=clicks)
        all_clicks.append(clicks)
        album = mm["index"][ws[0]].get("album", "|")
        artist, _, alb = album.partition("|")
        lines.append({"side": src.get("side", shard), "artist": src.get("artist", artist),
                      "album": src.get("album", alb), "split": name, "shard": f"{name}/{shard}",
                      "source_shard": f"{split}/{shard}", "level_bin": LEVEL_NAMES[k],
                      "windows": len(ws), "sr_native": int(src.get("sr_native", SR))})

    cl = np.concatenate(all_clicks) if all_clicks else np.zeros(0, CLICK_DTYPE)
    L = mm["audio"].shape[-1]
    ch_s = len(hosts) * mm["audio"].shape[1] * (L - 2 * EDGE) / SR
    stats = {"bin": k, "level": LEVEL_NAMES[k], "windows": int(len(hosts)), "shards": len(lines),
             "clicks": int(len(cl)), "label1": int((cl["label"] == LABEL_REPAIRED).sum()),
             "label2": int((cl["label"] == LABEL_DETECTED).sum()),
             "partners": int((cl["partner_of"] >= 0).sum()),
             "clicks_per_channel_second": float(len(cl) / ch_s) if ch_s else 0.0,
             "contrast_median": float(np.median(cl["contrast"])) if len(cl) else float("nan"),
             "level_db_median": float(np.median(cl["level_db"])) if len(cl) else float("nan"),
             "clip_rejects": inj["clip_rejects"], "dropped": inj["dropped"],
             "c_target": c_target_summary(cl["c_target"][cl["partner_of"] < 0], cfg)}
    (out_dir / "synth.json").write_text(json.dumps({"stats": stats, "config": asdict(cfg)}, indent=2))
    return {"stats": stats, "manifest": lines}


def run(data: Path, split: str, cfg: SynthConfig, out_root: Path | None = None, log=print) -> list[dict]:
    out_root = out_root or data / "synth"
    mm = open_split(data, split)
    src_manifest = {m["shard"]: m for m in _read_manifest(data / "manifest.jsonl")}
    bank, meta, source = load_bank(data, split, mm, log)
    if not len(bank):
        raise SystemExit(f"no templates in the {split} bank")
    t0 = time.time()
    counts = bin_candidate_counts(mm, cfg)
    log(f"scanned {len(counts)} {split} windows for level-bin positions in {time.time() - t0:.0f} s")

    results = []
    for k in range(len(LEVEL_NAMES)):
        hosts = select_hosts(counts, k, cfg)
        r = synthesize_bin(data, split, mm, k, hosts, bank, meta, cfg, src_manifest, out_root)
        r["stats"]["bank_source"] = source
        results.append(r)
        s = r["stats"]
        log(f"{split}_lvl{k} {s['level']:>9}: {s['windows']:4d} windows ({int((counts[:, k] >= cfg.min_candidates).sum())} "
            f"eligible), {s['clicks']:5d} clicks ({s['clicks_per_channel_second']:.2f}/ch-s, "
            f"{s['partners']} partners), label1 {s['label1']}, label2 {s['label2']}, "
            f"contrast median {s['contrast_median']:.1f}, level median {s['level_db_median']:.1f} dB")
        ct = s["c_target"]
        log(f"  c* median {ct['q50']:.1f} (exp {ct['q50_expected']:.1f}), P(c*>80) {ct['frac_gt80']:.3f} "
            f"(exp {ct['frac_gt80_expected']:.3f}); clip rejects {s['clip_rejects']}, dropped {s['dropped']}")

    mpath = out_root / "manifest.jsonl"
    keep = [m for m in _read_manifest(mpath) if not str(m.get("split", "")).startswith(f"{split}_lvl")]
    lines = keep + [m for r in results for m in r["manifest"]]
    mpath.write_text("".join(json.dumps(m) + "\n" for m in lines))
    log(f"wrote {sum(len(r['manifest']) for r in results)} shards under {out_root}; manifest {mpath}")
    return results


# MARK: - CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-nn-synth", description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=SynthConfig.windows, help="hosts per level bin")
    ap.add_argument("--rate", type=float, default=SynthConfig.rate, help="clicks per channel-second")
    ap.add_argument("--out", type=Path, default=None, help="default <data>/synth")
    a = ap.parse_args(argv)
    cfg = SynthConfig(windows=a.windows, rate=a.rate, seed=a.seed)
    run(a.data, a.split, cfg, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
