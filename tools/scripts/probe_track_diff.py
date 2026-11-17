"""Compare one de-clicked track with its raw side segment, sample by sample."""
import sys
from pathlib import Path
import numpy as np
from pressing_tools import library as lib
from pressing_tools.crd import read_crd

side, track, offset = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
t = lib.decode(track).astype(np.float64)
for off in (offset - 1, offset, offset + 1):
    s = lib.decode(side, start=off, frames=len(t)).astype(np.float64)
    n = min(len(s), len(t)); d = t[:n] - s[:n]
    print(f"offset {off}: exact-equal samples {np.mean(d == 0):.4%}  rms diff {20*np.log10(np.sqrt(np.mean(d**2))+1e-20):.1f} dBFS  "
          f"gain(lsq) {np.sum(t[:n]*s[:n])/np.sum(s[:n]**2):.5f}")
s = lib.decode(side, start=offset, frames=len(t)).astype(np.float64)
d = t - s
_, recs = read_crd(side.with_name(side.name + ".crd"))
inside = np.zeros(d.shape, bool)
recs = [r for r in recs if r.start >= offset and r.end <= offset + len(t)]
for r in recs: inside[r.start - offset:r.end - offset, r.channel] = True
print("records in track:", len(recs), " fraction of samples inside records:", f"{inside.mean():.4%}")
print("outside records: exact-equal", f"{np.mean(d[~inside] == 0):.4%}", " max|diff|", np.abs(d[~inside]).max())
print("inside records : exact-equal", f"{np.mean(d[inside] == 0):.4%}")
nz = np.argwhere((d != 0) & ~inside)
print("first nonzero diffs outside records (track sample, ch, diff*2^23):", [(int(i), int(c), round(d[i, c] * 2**23)) for i, c in nz[:8]])
