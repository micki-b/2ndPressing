"""Parity fixtures for the Rust percussion protection (rhythm protector).

For each case writes <out>/<case>/ with raw little-endian arrays + manifest:
    audio.bin            f32 (2, L)
    scores.bin           f32 (2, L) target-aligned logits (Python/ONNX reference)
    mask_off.bin         u8  (2, L) regions without protection
    mask_<mode>.bin      u8  (2, L) kept regions with protection (standard, extended)
    repaired_<mode>.bin  f32 (2, L) full offline output with protection
manifest.json: samples, sample_rate, threshold, modes {name: future_s}.

Cases: 'synthetic' (generated clapping track: crackle intro, claps on a
0.2 s grid with tails, random clicks, a 1.8 s rotation-periodic scratch; no
library audio, safe to commit) and optionally real 60 s long-form val
segments (git-ignored output).

usage: export_protect_fixtures.py OUT_DIR [--longform N] [--synthetic-only]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pressing_tools import classical
from pressing_tools.protect.rhythm import Protector
from pressing_tools.protect_eval import DATA, _method, regions_for

SR = 48000
THR = 2.75
MODES = {"standard": 1.0, "extended": 2.0}


def synthetic_track(seconds: float = 24.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    from scipy import signal
    b, a = signal.butter(2, [80, 5000], btype="band", fs=SR)
    music = 0.05 * signal.lfilter(b, a, rng.standard_normal(n))
    music += sum(amp * np.sin(2 * np.pi * f * t) for f, amp in ((110, .08), (165, .05), (220, .04), (440, .02)))
    env = np.where(t < 6.0, 0.03, 1.0)                       # quiet intro, then loud song
    x = np.stack([music * env, np.roll(music, 37) * env])
    # Surface crackle in the intro (random, quiet).
    for p in rng.integers(SR // 10, 6 * SR, 120):
        c = int(rng.integers(2)); L = int(rng.integers(8, 30)); k = np.arange(L)
        x[c, p:p + L] += rng.uniform(.005, .02) * rng.choice([-1, 1]) * np.exp(-k / (L / 3))
    # Claps: noise bursts with ~25 ms decay on a 0.2 s grid (+-2 ms jitter), both channels.
    bc, ac = signal.butter(2, [800, 6000], btype="band", fs=SR)
    for p in np.arange(int(6.2 * SR), int(21 * SR), int(0.2 * SR)):
        p = int(p + rng.integers(-96, 97)); L = 2400; k = np.arange(L)
        burst = signal.lfilter(bc, ac, rng.standard_normal(L)) * np.exp(-k / 1200) * 0.5
        burst[:6] += rng.choice([-1, 1]) * np.array([0.6, -0.45, 0.3, -0.18, 0.1, -0.05])  # sharp attack
        x[0, p:p + L] += burst
        x[1, p:p + L] += 0.8 * burst
    # Random clicks everywhere (one channel each).
    for p in rng.integers(SR // 2, n - SR // 2, 60):
        c = int(rng.integers(2)); L = int(rng.integers(10, 40)); k = np.arange(L)
        x[c, p:p + L] += rng.uniform(.1, .4) * rng.choice([-1, 1]) * np.exp(-k / (L / 3))
    # Rotation-periodic scratch (33 1/3 RPM), left channel, from 7 s.
    for p in np.arange(int(7 * SR), n - SR, int(1.8 * SR)):
        L = 24; k = np.arange(L)
        x[0, p:p + L] += 0.35 * np.exp(-k / 6)
    return np.clip(x, -0.99, 0.99)


def write_case(out: Path, x2: np.ndarray) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = _method()
    s2 = m.scores_stereo(x2)
    regions = regions_for(x2, s2, THR)
    L = x2.shape[1]
    arrays = {"audio": x2.astype(np.float32), "scores": s2.astype(np.float32)}
    off = np.zeros((2, L), np.uint8)
    for c in range(2):
        for a, b in regions[c]:
            off[c, a:b] = 1
    arrays["mask_off"] = off
    stats = {"regions": int(sum(len(r) for r in regions))}
    for mode, future in MODES.items():
        keep = Protector(future=future).keep(x2, s2, regions, SR)
        mk = np.zeros((2, L), np.uint8)
        rep = np.zeros((2, L), np.float32)
        for c in range(2):
            for (a, b), k in zip(regions[c], keep[c]):
                if k:
                    mk[c, a:b] = 1
            rep[c] = m.repair(x2[c], mk[c].astype(bool))
        arrays[f"mask_{mode}"] = mk
        arrays[f"repaired_{mode}"] = rep
        stats[f"protected_{mode}"] = int(sum(not k for kc in keep for k in kc))
    for k, v in arrays.items():
        v.astype(v.dtype.newbyteorder("<")).tofile(out / f"{k}.bin")
    manifest = {"samples": L, "channels": 2, "sample_rate": SR, "threshold": THR, "modes": MODES,
                "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()},
                "edge_ignore": 2400, **stats}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def decision_cases(x2: np.ndarray, name: str) -> dict:
    """Event-level parity case (no audio): the rhythm protector's inputs and
    decisions, for a compact committed Rust test."""
    from pressing_tools.protect.rhythm import _event_times
    s2 = _method().scores_stereo(x2)
    regions = regions_for(x2, s2, THR)
    base = Protector()
    lvl = base._levels(x2, SR)
    ev = _event_times(s2, x2, base.event_thr)
    case = {"name": name, "sample_rate": SR,
            "events": [[[int(t), float(lvl[c][t])] for t in ev[c]] for c in range(2)],
            "regions": [], "modes": {}}
    for c in range(2):
        for a, b in regions[c]:
            pk = a + int(np.argmax(s2[c, a:b]))
            case["regions"].append({"c": c, "a": int(a), "b": int(b), "peak": int(pk),
                                    "level": float(lvl[c][pk])})
    for mode, future in MODES.items():
        raw = Protector(future=future, cluster_s=0.0, veto=False).keep(x2, s2, regions, SR)
        fin = Protector(future=future).keep(x2, s2, regions, SR)
        case["modes"][mode] = {"future_s": future,
                               "rhythmic": [not k for c in range(2) for k in raw[c]],
                               "keep": [k for c in range(2) for k in fin[c]]}
    return case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--longform", type=int, default=0, help="also export N real long-form val segments")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--mirage", type=Path, default=None, help="also export this file (e.g. the Mirage sample)")
    ap.add_argument("--decisions", type=Path, default=None,
                    help="write the compact event-level JSON (synthetic + first long-form segment) here")
    a = ap.parse_args()
    syn = synthetic_track()
    if a.decisions:
        cases = [decision_cases(syn, "synthetic")]
        a.decisions.parent.mkdir(parents=True, exist_ok=True)
        a.decisions.write_text(json.dumps(cases))
        print("decisions:", a.decisions, f"{a.decisions.stat().st_size / 1e3:.0f} kB")
    m = write_case(a.out / "synthetic", syn)
    print("synthetic:", {k: v for k, v in m.items() if k in ("regions", "protected_standard", "protected_extended")})
    if a.mirage:
        from pressing_tools import library as lib
        m = write_case(a.out / "mirage", lib.decode(a.mirage).T.astype(np.float64))
        print("mirage", {k: v for k, v in m.items() if k in ("regions", "protected_standard", "protected_extended")})
    files = sorted((DATA / "longform" / "val").glob("*.npz"))
    for f in files[:a.longform]:
        x2 = np.load(f)["audio"].astype(np.float64)
        m = write_case(a.out / f.stem, x2)
        print(f.stem, {k: v for k, v in m.items() if k in ("regions", "protected_standard", "protected_extended")})


if __name__ == "__main__":
    main()
