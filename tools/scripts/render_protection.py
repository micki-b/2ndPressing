"""Render a file through the Python reference pipeline with/without a protector.

Writes <out>/<stem>_<variant>_{declicked,clicks-only}.wav (float32 WAV) for
variant in (standard, protected), plus a small CSV of repaired regions per
second. Uses the selected R2 model and plugin-equivalent threshold.

usage: render_protection.py IN.m4a OUT_DIR [--threshold 2.75]
       [--protector pressing_tools.protect.rhythm:Protector]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from pressing_tools import classical, library as lib
from pressing_tools.protect_eval import _method, load_protector, regions_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--threshold", type=float, default=2.75)
    ap.add_argument("--protector", default="pressing_tools.protect.rhythm:Protector")
    a = ap.parse_args()
    sr = lib.probe(a.inp)["sr"]
    x2 = lib.decode(a.inp).T.astype(np.float64)
    m = _method()
    s2 = m.scores_stereo(x2)
    regions = regions_for(x2, s2, a.threshold)
    a.out.mkdir(parents=True, exist_ok=True)
    per_sec = {}
    for variant, prot in (("standard", load_protector(None)), ("protected", load_protector(a.protector))):
        keep = prot.keep(x2, s2, regions, sr)
        y = np.zeros_like(x2)
        counts = np.zeros(int(x2.shape[1] / sr) + 1, int)
        for c in range(2):
            mask = np.zeros(x2.shape[1], bool)
            for (r0, r1), k in zip(regions[c], keep[c]):
                if k:
                    mask[r0:r1] = True
                    counts[r0 // sr] += 1
            y[c] = m.repair(x2[c], mask)
        per_sec[variant] = counts
        stem = a.inp.stem
        sf.write(a.out / f"{stem}_{variant}_declicked.wav", y.T.astype(np.float32), sr, subtype="FLOAT")
        sf.write(a.out / f"{stem}_{variant}_clicks-only.wav", (x2 - y).T.astype(np.float32), sr, subtype="FLOAT")
    with open(a.out / f"{a.inp.stem}_regions_per_second.csv", "w") as f:
        f.write("second,standard,protected\n")
        for i, (s, p) in enumerate(zip(per_sec["standard"], per_sec["protected"])):
            f.write(f"{i},{s},{p}\n")
    print("regions repaired  standard:", int(per_sec["standard"].sum()),
          " protected:", int(per_sec["protected"].sum()))
    print("per second (standard):  ", " ".join(f"{v:3d}" for v in per_sec["standard"]))
    print("per second (protected): ", " ".join(f"{v:3d}" for v in per_sec["protected"]))


if __name__ == "__main__":
    main()
