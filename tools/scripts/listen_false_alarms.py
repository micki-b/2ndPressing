"""Listening audit: random false alarms of the neural method on a split.

Writes, for N random false-alarm regions (predicted regions touching no label):
    fa_raw.wav       1.5 s clips of the raw audio, separated by 0.6 s of silence
    fa_repaired.wav  the same clips after our repair
    fa_index.csv     clip number, start time in the files, album/side, window,
                     channel, region position and length, logit peak

usage: listen_false_alarms.py MODEL.onnx THRESHOLD [--split test] [--n 50]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from pressing_tools import classical
from pressing_tools.dataset import load_shard
from pressing_tools.evaluate import EDGE, TOL, _dilate
from pressing_tools.nn.method import NeuralConfig, NeuralMethod

SR = 48000
HALF = int(0.75 * SR)
GAP = np.zeros((2, int(0.6 * SR)), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("threshold", type=float)
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/listen_fa"))
    a = ap.parse_args()

    method = NeuralMethod(NeuralConfig(model=a.model, threshold=a.threshold, min_contrast=0.0))
    manifest = {json.loads(l)["shard"].split("/")[-1]: json.loads(l)
                for l in (a.data / "manifest.jsonl").read_text().splitlines() if l.strip()}
    cands = []
    for shard in sorted((a.data / a.split).glob("*.npz")):
        sh = load_shard(shard)
        audio, label = sh["audio"].astype(np.float64), sh["label"]
        for w in range(audio.shape[0]):
            s2 = method.scores_stereo(audio[w])
            for c in range(2):
                m = method.mask(audio[w, c], s2[c], a.threshold)
                near = _dilate(label[w, c] != 0, TOL)
                for r0, r1 in classical.runs(m):
                    if r0 - HALF >= EDGE // 4 and r1 + HALF <= audio.shape[2] and not near[r0:r1].any():
                        cands.append((shard.name, w, c, r0, r1, float(s2[c, r0:r1].max())))
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(cands), min(a.n, len(cands)), replace=False))
    a.out.mkdir(parents=True, exist_ok=True)
    raws, fixed, rows, t = [], [], [], 0.0
    cache: dict = {}
    for i, k in enumerate(pick):
        name, w, c, r0, r1, peak = cands[k]
        if name not in cache:
            cache.clear()
            cache[name] = load_shard(a.data / a.split / name)["audio"].astype(np.float64)
        x = cache[name][w]
        s2 = method.scores_stereo(x)
        y = x.copy()
        for ch in range(2):
            y[ch] = method.repair(x[ch], method.mask(x[ch], s2[ch], a.threshold))
        mid = (r0 + r1) // 2
        raws += [x[:, mid - HALF:mid + HALF].astype(np.float32), GAP]
        fixed += [y[:, mid - HALF:mid + HALF].astype(np.float32), GAP]
        meta = manifest.get(name, {})
        rows.append({"clip": i + 1, "time_in_file_s": round(t + 0.75, 2),
                     "album": f"{meta.get('artist', '')} - {meta.get('album', '')}", "shard": name,
                     "window": w, "channel": "LR"[c], "region_start": r0, "length": r1 - r0,
                     "logit_peak": round(peak, 2)})
        t += 1.5 + 0.6
    sf.write(a.out / "fa_raw.wav", np.concatenate(raws, axis=1).T, SR, subtype="FLOAT")
    sf.write(a.out / "fa_repaired.wav", np.concatenate(fixed, axis=1).T, SR, subtype="FLOAT")
    with open(a.out / "fa_index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(cands)} false-alarm regions found; wrote {len(rows)} clips to {a.out}")


if __name__ == "__main__":
    main()
