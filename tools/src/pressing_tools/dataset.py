"""Build the click-detection training set from raw sides + .crd labels.

Each side yields fixed-length windows at a canonical sample rate. Per window
we store (shard ``<split>/<side>.npz``)::

    audio   float16 (W, 2, L)   raw side audio (the clicks are still in it)
    label   uint8   (W, 2, L)   0 clean, 1 repaired by VinylStudio,
                                2 detected by VinylStudio but left alone
    repl    float16 (K,)        VinylStudio's replacement samples at every
                                label==1 position, in np.nonzero order --
                                see ``repaired()`` to rebuild the target
    starts  int64   (W,)        window start in *native* side samples
    sr_native, sr               sample rates

Splits are assigned per album (hash of artist+album), so every side of a
test album is unseen during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from pressing_tools import paths
from pressing_tools import library as lib
from pressing_tools.crd import PAYLOAD_KINDS, CrdFormatError, read_crd

LABEL_CLEAN, LABEL_REPAIRED, LABEL_DETECTED = 0, 1, 2


# MARK: - Configuration


@dataclass
class DatasetConfig:
    sr: int = 48000                  # canonical sample rate
    window_s: float = 2.0
    seconds_per_window: float = 30.0  # density: one window per this much audio
    val_frac: float = 0.05
    test_frac: float = 0.05
    resample_margin: int = 2048      # native samples of context for resampling
    seed: int = 1


def split_for(side: lib.Side, cfg: DatasetConfig) -> str:
    h = int(hashlib.sha1("|".join(side.key).encode()).hexdigest()[:8], 16) / 2**32
    if h < cfg.test_frac:
        return "test"
    if h < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


# MARK: - Streaming decode


def iter_blocks(path: Path, channels: int, block_frames: int = 1 << 20):
    """Yield (start_frame, float32 block (n, channels)) for the whole file."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
           "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = 4 * channels
    pos = 0
    try:
        while True:
            buf = proc.stdout.read(block_frames * frame_bytes)
            if not buf:
                break
            usable = len(buf) - len(buf) % frame_bytes
            block = np.frombuffer(buf[:usable], "<f4").reshape(-1, channels)
            yield pos, block
            pos += len(block)
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed decoding {path}")


def gather(path: Path, channels: int, spans: list[tuple[int, int]]) -> list[np.ndarray]:
    """Extract [s, e) frame spans (sorted by s, may overlap) in one decode pass."""
    out = [np.zeros((e - s, channels), np.float32) for s, e in spans]
    for b0, block in iter_blocks(path, channels):
        b1 = b0 + len(block)
        for i, (s, e) in enumerate(spans):
            if e <= b0:
                continue
            if s >= b1:
                break
            lo, hi = max(s, b0), min(e, b1)
            out[i][lo - s:hi - s] = block[lo - b0:hi - b0]
    return out


# MARK: - Per-side build


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = math.gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g, axis=0)


def build_side(side: lib.Side, out_root: Path, cfg: DatasetConfig) -> dict:
    info = lib.probe(side.path)
    sr0, nch = info["sr"], info["channels"]
    n_native = int(round(info["duration"] * sr0))
    split = split_for(side, cfg)
    out = out_root / split / f"{side.path.stem}.npz"
    meta = {"side": side.path.name, "artist": side.artist, "album": side.album,
            "split": split, "sr_native": sr0, "seconds": round(info["duration"], 1),
            "shard": str(out.relative_to(out_root))}

    _, recs = read_crd(side.crd)
    meta["records"] = {str(k): sum(r.kind == k for r in recs) for k in sorted({r.kind for r in recs})}

    L0 = int(math.ceil(cfg.window_s * sr0))                 # native window length
    L = int(round(cfg.window_s * cfg.sr))                   # canonical window length
    m = cfg.resample_margin if sr0 != cfg.sr else 0
    count = max(1, int(info["duration"] / cfg.seconds_per_window))
    rng = np.random.default_rng([cfg.seed, int(hashlib.sha1(side.path.name.encode()).hexdigest()[:8], 16)])
    starts = np.sort(rng.integers(m, max(m + 1, n_native - L0 - m), count))

    spans = [(int(s) - m, int(s) + L0 + m) for s in starts]
    chunks = gather(side.path, nch, spans)

    # Label and paste replacements at native rate, then resample both.
    by_ch = [sorted((r for r in recs if r.channel == c and r.length > 0), key=lambda r: r.start)
             for c in range(nch)]
    rec_starts = [np.array([r.start for r in rs], np.int64) for rs in by_ch]
    max_len = [max((r.length for r in rs), default=0) for rs in by_ch]
    audio = np.zeros((count, nch, L), np.float16)
    label = np.zeros((count, nch, L), np.uint8)
    repl_parts = []
    ratio = cfg.sr / sr0
    for w, ((a0, a1), chunk) in enumerate(zip(spans, chunks)):
        target = chunk.copy()
        repaired_mask = np.zeros((L, nch), bool)
        detected_mask = np.zeros((L, nch), bool)
        for c in range(nch):
            # records overlapping [a0, a1): start in [a0 - max_len, a1)
            i0 = int(np.searchsorted(rec_starts[c], a0 - max_len[c]))
            i1 = int(np.searchsorted(rec_starts[c], a1))
            for r in by_ch[c][i0:i1]:
                if r.end <= a0:
                    continue
                if r.kind in PAYLOAD_KINDS:
                    lo, hi = max(r.start, a0), min(r.end, a1)
                    target[lo - a0:hi - a0, c] = r.replacement[lo - r.start:hi - r.start] / 32768.0
                # label span: native [start, end) relative to window (sans margin) -> canonical
                ls = max(math.floor((r.start - (a0 + m)) * ratio), 0)
                le = min(math.ceil((r.end - (a0 + m)) * ratio), L)
                if le > ls:
                    mask = repaired_mask if r.kind in PAYLOAD_KINDS else detected_mask
                    mask[ls:le, c] = True
        # A repair wins over a detection where they overlap.
        lab = np.where(repaired_mask, LABEL_REPAIRED,
                       np.where(detected_mask, LABEL_DETECTED, LABEL_CLEAN)).astype(np.uint8)
        x = _resample(chunk, sr0, cfg.sr)
        t = _resample(target, sr0, cfg.sr)
        off = int(round(m * ratio))
        x, t = x[off:off + L], t[off:off + L]
        audio[w] = x.T.astype(np.float16)
        label[w] = lab.T
        repl_parts.append(t.T[lab.T == LABEL_REPAIRED].astype(np.float16))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, audio=audio, label=label, repl=np.concatenate(repl_parts),
                        starts=starts, sr_native=sr0, sr=cfg.sr)
    tmp.rename(out)
    meta.update(windows=count,
                frac_repaired=float(np.mean(label == LABEL_REPAIRED)),
                frac_detected=float(np.mean(label == LABEL_DETECTED)))
    return meta


# MARK: - Loading helpers


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def repaired(shard: dict[str, np.ndarray]) -> np.ndarray:
    """VinylStudio-repaired audio for every window (float32, W x C x L)."""
    t = shard["audio"].astype(np.float32)
    t[shard["label"] == LABEL_REPAIRED] = shard["repl"].astype(np.float32)
    return t


# MARK: - CLI


def _job(side_path: str, out_root: str, cfg: dict) -> dict:
    p = Path(side_path)
    artist, album, side_name = p.stem.split("~", 2)
    side = lib.Side(p, artist.replace("_", " "), album.replace("_", " "), side_name)
    try:
        return build_side(side, Path(out_root), DatasetConfig(**cfg))
    except (CrdFormatError, RuntimeError) as e:
        return {"side": p.name, "error": str(e)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-dataset", description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=None, help="VinylStudio captures (default: paths.raw_dir())")
    ap.add_argument("--out", type=Path, default=Path("../data/v1"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sides (testing)")
    ap.add_argument("--seconds-per-window", type=float, default=DatasetConfig.seconds_per_window)
    ap.add_argument("--window", type=float, default=DatasetConfig.window_s)
    a = ap.parse_args(argv)
    cfg = DatasetConfig(window_s=a.window, seconds_per_window=a.seconds_per_window)

    sides = [s for s in lib.find_sides(a.raw or paths.raw_dir()) if s.crd.exists()]
    if a.limit:
        sides = sides[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    manifest = a.out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["side"] for l in manifest.read_text().splitlines() if l.strip()}
    todo = [s for s in sides if s.path.name not in done]
    print(f"{len(sides)} sides, {len(done)} already built, {len(todo)} to do", flush=True)

    with ProcessPoolExecutor(a.jobs) as ex, open(manifest, "a") as mf:
        futs = {ex.submit(_job, str(s.path), str(a.out), asdict(cfg)): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            meta = f.result()
            if "error" not in meta:
                mf.write(json.dumps(meta) + "\n")
                mf.flush()
            status = meta.get("error") or (f"{meta['split']:5s} {meta['windows']:4d} win  "
                                           f"rep {meta['frac_repaired']:.3%} det {meta['frac_detected']:.3%}")
            print(f"[{i}/{len(todo)}] {meta['side'][:60]:60s} {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
