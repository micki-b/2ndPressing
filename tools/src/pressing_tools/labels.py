"""Extract click labels from (original, de-clicked) recording pairs.

The de-clicked versions in the library were made from the same capture as the
originals but may also have had gain, EQ, noise reduction or resampling
applied. To isolate *only* the declicking edits we:

1. Bring both files to the same sample rate.
2. Estimate the time offset (and check for drift) by cross-correlation.
3. Per block, fit a short two-sided FIR filter mapping original -> de-clicked.
   This absorbs gain, gentle EQ, polarity and fractional-sample offsets. The
   fit is robust: samples with large residuals (i.e. the clicks themselves)
   are excluded and the filter is refit.
4. The residual (de-clicked minus filtered original) is then small broadband
   noise (e.g. hiss removed by noise reduction) plus large, sparse impulses
   where clicks were removed. Impulses well above the local residual floor
   become labelled click regions.

Outputs per pair: labels.csv (channel, start, end, peak), report.json (fit
quality, lag, drift, click statistics) and optionally residual.wav for
listening.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import maximum_filter1d


# MARK: - Configuration


@dataclass
class LabelConfig:
    fir_half_taps: int = 32          # FIR spans +/- this many samples
    block_seconds: float = 1.0       # block length for the per-block FIR fit
    robust_iters: int = 2            # refits after excluding outliers
    outlier_k: float = 5.0           # MAD multiple treated as outlier in the fit
    detect_k: float = 8.0            # MAD multiple for click detection
    outlier_rel: float = 0.3         # fit outlier must also be >= this x local peak of de-clicked
    detect_rel: float = 0.15         # click must also be >= this x local peak of de-clicked
    floor_dbfs: float = -72.0        # never flag residuals below this level
    subblock: int = 1024             # samples per local noise-floor estimate
    pad_ms: float = 0.25             # widen each detected region on both sides
    merge_gap_ms: float = 1.0        # merge regions closer than this
    max_lag_seconds: float = 2.0     # search range for the offset
    long_filter: bool = True         # first apply a long global filter (rumble HPF etc.)
    long_nperseg: int = 32768        # FFT size for the global filter estimate
    drift_tolerance: int = 2         # samples; more than this flags the pair


# MARK: - Data types


@dataclass
class ClickRegion:
    channel: int
    start: int   # first sample (inclusive), in original-file coordinates
    end: int     # last sample (exclusive)
    peak: float  # peak absolute residual inside the region


@dataclass
class PairReport:
    original: str
    declicked: str
    sample_rate: int
    channels: int
    frames: int
    declicked_rate: int
    lag_samples: int
    lag_by_position: list[int]
    drift_samples: int
    polarity: int
    residual_floor_db: float         # rms of non-click residual relative to signal
    click_count: int
    clicked_fraction: float          # fraction of samples inside click regions
    click_length_ms_percentiles: dict[str, float] = field(default_factory=dict)
    clicks_per_minute: float = 0.0
    usable: bool = True
    warnings: list[str] = field(default_factory=list)


# MARK: - Loading and alignment


def load(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = math.gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def estimate_lags(ref: np.ndarray, other: np.ndarray, sr: int, max_lag_s: float,
                  win_s: float = 10.0, positions: int = 5) -> tuple[list[int], list[int]]:
    """Return (lags, polarities) at several positions along the track.

    Convention: other[n] ~= ref[n - lag].
    """
    max_lag = int(max_lag_s * sr)
    win = int(min(win_s * sr, max(len(ref) - 2 * max_lag, sr // 2)))
    usable = len(ref) - win - 2 * max_lag
    if usable <= 0:
        max_lag = max((len(ref) - win) // 4, 0)
        usable = max(len(ref) - win - 2 * max_lag, 1)
    starts = np.linspace(max_lag, max_lag + usable - 1, positions).astype(int)

    lags, pols = [], []
    for p in starts:
        ref_seg = ref[p:p + win]
        lo, hi = p - max_lag, p + win + max_lag
        pad_lo, pad_hi = max(0, -lo), max(0, hi - len(other))
        other_seg = other[max(lo, 0):min(hi, len(other))]
        other_seg = np.pad(other_seg, (pad_lo, pad_hi))
        if not np.any(ref_seg) or not np.any(other_seg):
            continue
        c = signal.correlate(other_seg, ref_seg, mode="valid", method="fft")
        i = int(np.argmax(np.abs(c)))
        lags.append(i - max_lag)
        pols.append(1 if c[i] >= 0 else -1)
    if not lags:
        raise ValueError("Could not estimate offset: one of the files is silent")
    return lags, pols


def shift(x: np.ndarray, lag: int, length: int) -> np.ndarray:
    """Return y with y[n] = x[n + lag], zero-padded, of the given length."""
    y = np.zeros((length,) + x.shape[1:], dtype=x.dtype)
    src_lo, src_hi = max(lag, 0), min(lag + length, len(x))
    if src_hi > src_lo:
        y[src_lo - lag:src_hi - lag] = x[src_lo:src_hi]
    return y


# MARK: - Robust FIR matching


def _fit_fir(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xm, ym = X[mask], y[mask]
    A = Xm.T @ Xm
    b = Xm.T @ ym
    ridge = 1e-9 * (np.trace(A) + 1e-12)
    return np.linalg.solve(A + ridge * np.eye(A.shape[0]), b)


def local_peak(x: np.ndarray, half: int) -> np.ndarray:
    """Max |x| within +/- half samples of each sample."""
    return maximum_filter1d(np.abs(x), size=2 * half + 1, mode="nearest")


def is_impulsive(r: np.ndarray, peak_ref: np.ndarray, k: float, rel: float, sub: int,
                 smooth: bool) -> np.ndarray:
    """Residual samples that look like a removed click rather than a fit error.

    A removed click leaves a residual that is large relative to the de-clicked
    signal around it (the click is absent there). A filter-mismatch error is a
    small fraction of the local signal. Requiring both an outlier vs. the local residual
    floor and a large fraction of the local original peak separates the two.
    """
    return ((np.abs(r) > k * local_sigma(r, sub, smooth=smooth))
            & (np.abs(r) > rel * peak_ref))


def global_filter(x: np.ndarray, y: np.ndarray, nperseg: int) -> np.ndarray:
    """Apply to x the long linear filter that best maps x -> y.

    Estimated from Welch cross-spectra over the whole signal (H = Pxy / Pxx),
    so a few click spans barely influence it. Needed because real exports can
    include rumble high-pass filters whose impulse responses last tens of
    milliseconds -- far beyond the short per-block FIR.
    """
    x = x.astype(np.float64)
    _, pxy = signal.csd(x, y, nperseg=nperseg)        # conj(X) * Y
    _, pxx = signal.welch(x, nperseg=nperseg)
    h = np.fft.irfft(pxy / (pxx + 1e-30), n=nperseg)
    h = np.roll(h, nperseg // 2) * signal.windows.tukey(nperseg, 0.25)
    return signal.fftconvolve(x, h, mode="full")[nperseg // 2: nperseg // 2 + len(x)]


def match_residual(x: np.ndarray, y: np.ndarray, cfg: LabelConfig, sr: int) -> np.ndarray:
    """Residual y - H(x): optional long global filter, then a per-block robust
    short FIR fit for what remains. x, y are 1-D float."""
    if cfg.long_filter and len(x) >= 4 * cfg.long_nperseg:
        x = global_filter(x, y.astype(np.float64), cfg.long_nperseg)
    c = cfg.fir_half_taps
    taps = 2 * c + 1
    xp = np.pad(x.astype(np.float64), (c, c))
    y = y.astype(np.float64)
    block = max(int(cfg.block_seconds * sr), 8 * taps)
    resid = np.zeros_like(y)
    dilate = np.ones(taps, dtype=bool)
    peak_y = local_peak(y, c)

    for b0 in range(0, len(y), block):
        b1 = min(b0 + block, len(y))
        if b1 - b0 < 4 * taps:  # tiny tail: reuse previous filter
            b0_fit = max(0, b1 - block)
        else:
            b0_fit = b0
        X = sliding_window_view(xp[b0_fit:b1 + 2 * c], taps)
        yb = y[b0_fit:b1]
        if not np.any(X) or not np.any(yb):
            resid[b0:b1] = yb[b0 - b0_fit:]
            continue
        mask = np.ones(len(yb), dtype=bool)
        h = _fit_fir(X, yb, mask)
        for _ in range(cfg.robust_iters):
            r = yb - X @ h
            # Loud transients (drums) have proportionally larger fit error and
            # must stay in the fit, or the filter never learns the high
            # frequencies they carry -- so only exclude *impulsive* outliers.
            bad = is_impulsive(r, peak_y[b0_fit:b1], cfg.outlier_k, cfg.outlier_rel,
                               cfg.subblock, smooth=False)
            bad = np.convolve(bad, dilate, mode="same") > 0
            if bad.mean() > 0.5:  # fit is hopeless here; don't exclude everything
                break
            mask = ~bad
            h = _fit_fir(X, yb, mask)
        resid[b0:b1] = (yb - X @ h)[b0 - b0_fit:]
    return resid


# MARK: - Detection


def local_sigma(r: np.ndarray, sub: int, smooth: bool = True) -> np.ndarray:
    """Per-sample robust noise floor: MAD per sub-block, optionally median-smoothed."""
    n_sub = max(1, math.ceil(len(r) / sub))
    padded = np.pad(np.abs(r), (0, n_sub * sub - len(r)))
    mads = 1.4826 * np.median(padded.reshape(n_sub, sub), axis=1)
    if smooth and n_sub >= 9:
        mads = signal.medfilt(mads, 9)
    return np.repeat(mads, sub)[:len(r)] + 1e-12


def detect_regions(r: np.ndarray, y: np.ndarray, valid: np.ndarray, channel: int,
                   cfg: LabelConfig, sr: int) -> list[ClickRegion]:
    """Find click regions in residual r; y is the aligned de-clicked channel, valid masks
    samples where both files have real (non-padded) content."""
    floor = 10 ** (cfg.floor_dbfs / 20)
    hot = (is_impulsive(r, local_peak(y, cfg.fir_half_taps), cfg.detect_k, cfg.detect_rel,
                        cfg.subblock, smooth=True)
           & (np.abs(r) > floor) & valid)
    if not hot.any():
        return []
    pad = max(1, int(cfg.pad_ms * 1e-3 * sr))
    merge = int(cfg.merge_gap_ms * 1e-3 * sr)
    idx = np.flatnonzero(hot)
    # Split where the gap between consecutive hot samples exceeds merge + 2*pad.
    splits = np.flatnonzero(np.diff(idx) > merge + 2 * pad) + 1
    regions = []
    for group in np.split(idx, splits):
        s = max(0, int(group[0]) - pad)
        e = min(len(r), int(group[-1]) + pad + 1)
        regions.append(ClickRegion(channel, s, e, float(np.max(np.abs(r[s:e])))))
    return regions


# MARK: - Pair processing


def process_pair(orig_path: Path, clean_path: Path, out_dir: Path | None = None,
                 cfg: LabelConfig | None = None, write_residual: bool = False
                 ) -> tuple[PairReport, list[ClickRegion]]:
    cfg = cfg or LabelConfig()
    orig, sr = load(orig_path)
    clean, sr_clean = load(clean_path)
    warnings: list[str] = []

    if sr_clean != sr:
        warnings.append(f"declicked file resampled {sr_clean} -> {sr} Hz for comparison")
        clean = resample(clean, sr_clean, sr)
    if clean.shape[1] != orig.shape[1]:
        warnings.append(f"channel mismatch {orig.shape[1]} vs {clean.shape[1]}; using mono mix")
        orig = orig.mean(axis=1, keepdims=True)
        clean = clean.mean(axis=1, keepdims=True)

    lags, pols = estimate_lags(orig.mean(axis=1), clean.mean(axis=1), sr, cfg.max_lag_seconds)
    lag = int(np.median(lags))
    drift = int(max(lags) - min(lags))
    polarity = int(np.sign(np.sum(pols)) or 1)
    if drift > cfg.drift_tolerance:
        warnings.append(f"offset drifts by {drift} samples along the track "
                        "(clock drift or edits); labels may be unreliable")

    aligned = shift(clean, lag, len(orig))
    # Samples where the shifted de-clicked file has real content, minus the FIR
    # edge region -- padding there would otherwise look like clicks.
    valid = np.zeros(len(orig), dtype=bool)
    edge = cfg.fir_half_taps
    lo, hi = max(0, -lag) + edge, min(len(orig), len(clean) - lag) - edge
    if hi > lo:
        valid[lo:hi] = True
    regions: list[ClickRegion] = []
    resid = np.zeros_like(orig, dtype=np.float64)
    for ch in range(orig.shape[1]):
        resid[:, ch] = match_residual(orig[:, ch], aligned[:, ch], cfg, sr)
        regions.extend(detect_regions(resid[:, ch], aligned[:, ch], valid, ch, cfg, sr))

    # Quality: residual outside clicks relative to signal level.
    in_click = np.zeros(orig.shape, dtype=bool)
    for reg in regions:
        in_click[reg.start:reg.end, reg.channel] = True
    sig_rms = float(np.sqrt(np.mean(orig.astype(np.float64) ** 2)) + 1e-12)
    keep = ~in_click & valid[:, None]
    floor_rms = float(np.sqrt(np.mean(resid[keep] ** 2)) + 1e-12)
    floor_db = 20 * math.log10(floor_rms / sig_rms)
    if floor_db > -20:
        warnings.append(f"residual floor {floor_db:.1f} dB re signal: processing "
                        "differs too much from a linear filter (heavy NR, edits or "
                        "different transfer)")

    lengths_ms = np.array([(r.end - r.start) / sr * 1e3 for r in regions])
    minutes = len(orig) / sr / 60
    report = PairReport(
        original=str(orig_path), declicked=str(clean_path),
        sample_rate=sr, channels=orig.shape[1], frames=len(orig),
        declicked_rate=sr_clean, lag_samples=lag, lag_by_position=lags,
        drift_samples=drift, polarity=polarity,
        residual_floor_db=round(floor_db, 2),
        click_count=len(regions),
        clicked_fraction=float(in_click.mean()),
        click_length_ms_percentiles=(
            {p: round(float(np.percentile(lengths_ms, int(p[1:]))), 3)
             for p in ("p50", "p90", "p99")} if len(regions) else {}),
        clicks_per_minute=round(len(regions) / minutes, 2) if minutes else 0.0,
        usable=drift <= cfg.drift_tolerance and floor_db <= -20,
        warnings=warnings,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["channel", "start", "end", "peak"])
            for reg in regions:
                w.writerow([reg.channel, reg.start, reg.end, f"{reg.peak:.6f}"])
        (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2))
        if write_residual:
            sf.write(str(out_dir / "residual.wav"), resid.astype(np.float32), sr,
                     subtype="FLOAT")
    return report, regions


# MARK: - CLI


def _pair_out_dir(root: Path, orig: Path) -> Path:
    return root / orig.stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-labels", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pair", help="process one original/de-clicked pair")
    p.add_argument("original", type=Path)
    p.add_argument("declicked", type=Path)

    b = sub.add_parser("batch", help="process pairs listed in a CSV (original,declicked)")
    b.add_argument("pairs_csv", type=Path)

    for s in (p, b):
        s.add_argument("-o", "--out", type=Path, default=Path("out/labels"))
        s.add_argument("--residual", action="store_true", help="also write residual.wav")
        s.add_argument("--detect-k", type=float, default=LabelConfig.detect_k)

    args = ap.parse_args(argv)
    cfg = LabelConfig(detect_k=args.detect_k)

    if args.cmd == "pair":
        pairs = [(args.original, args.declicked)]
    else:
        with open(args.pairs_csv, newline="") as f:
            pairs = [(Path(r[0]), Path(r[1])) for r in csv.reader(f)
                     if r and not r[0].startswith("#")]

    summary = []
    for orig, clean in pairs:
        try:
            rep, _ = process_pair(orig, clean, _pair_out_dir(args.out, orig), cfg,
                                  args.residual)
        except Exception as e:  # keep going through a batch
            print(f"FAIL  {orig.name}: {e}", file=sys.stderr)
            continue
        flag = "ok  " if rep.usable else "WARN"
        print(f"{flag}  {orig.name}: lag={rep.lag_samples} drift={rep.drift_samples} "
              f"floor={rep.residual_floor_db} dB clicks={rep.click_count} "
              f"({rep.clicks_per_minute}/min)")
        for w in rep.warnings:
            print(f"      - {w}")
        summary.append(asdict(rep))

    if len(summary) > 1:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
