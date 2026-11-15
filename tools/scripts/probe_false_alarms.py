"""Characterise classical false alarms vs. labelled clicks on a few val shards."""
import sys
from pathlib import Path
import numpy as np
from pressing_tools import classical
from pressing_tools.dataset import load_shard
from pressing_tools.evaluate import EDGE, TOL, _dilate, spike_ratio

shards = sorted(Path("../data/v1/val").glob("*.npz"))[: int(sys.argv[1]) if len(sys.argv) > 1 else 4]
cfg = classical.ClassicalConfig()
rows = {"false": [], "l1": [], "l2": []}
for p in shards:
    sh = load_shard(p); A = sh["audio"].astype(np.float64); Lb = sh["label"]
    for w in range(A.shape[0]):
        S = [classical.scores(A[w, c], cfg) for c in range(2)]
        for c in range(2):
            x, lab = A[w, c], Lb[w, c]
            s = S[c]; m = classical.regions_from_scores(s, cfg)
            near = _dilate(lab != 0, TOL)
            for a, b in classical.runs(m):
                if a < EDGE or b > len(x) - EDGE: continue
                lvl = 20 * np.log10(np.sqrt(np.mean(x[a - 2400:b + 2400] ** 2)) + 1e-9)
                s2 = S[1 - c][a:b].max()   # does the other channel also spike here?
                rec = (spike_ratio(x, a, b), lvl, s[a:b].max(), s2, b - a)
                if None in rec: continue
                rows["l1" if (lab[max(0, a - TOL):b + TOL] == 1).any() else ("false" if not near[a:b].any() else "l2")].append(rec)
for k, v in rows.items():
    v = np.array(v)
    print(f"{k:5s} n={len(v):5d}  spike ratio p25/50/75 {np.percentile(v[:,0],25):5.1f} {np.median(v[:,0]):5.1f} {np.percentile(v[:,0],75):5.1f}"
          f" | level dBFS p50 {np.median(v[:,1]):6.1f} | score p50 {np.median(v[:,2]):5.1f} | other-ch score>8 {np.mean(v[:,3]>8):5.1%} | len p50 {np.median(v[:,4]):.0f}")
f = np.array(rows["false"]); l = np.array(rows["l1"])
for lo, hi in ((-90, -40), (-40, -30), (-30, -20), (-20, 0)):
    print(f"  level {lo:>4}..{hi:>4} dBFS: false {np.sum((f[:,1]>=lo)&(f[:,1]<hi)):5d}   labelled {np.sum((l[:,1]>=lo)&(l[:,1]<hi)):5d}")
