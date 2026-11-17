"""Show raw vs exported samples around 'unexplained' residual impulses."""
import sys
from pathlib import Path
import numpy as np
from pressing_tools import library as lib
from pressing_tools.labels import LabelConfig, global_filter

side, track, offset = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
positions = [tuple(map(int, p.split(":"))) for p in sys.argv[4:]]   # ch:start
t = lib.decode(track).astype(np.float64)
raw = lib.decode(side, start=offset, frames=len(t)).astype(np.float64)
cfg = LabelConfig()
for ch, s in positions:
    hx = global_filter(raw[:, ch], t[:, ch], cfg.long_nperseg)
    sl = slice(s - 4, s + 30)
    r = t[sl, ch] - hx[sl]
    i = int(np.argmax(np.abs(r))) + s - 4
    w = slice(i - 6, i + 7)
    print(f"ch{ch} @ {i} (side sample {offset + i}, {i / 48000:.2f}s)")
    print("   track   :", np.round(t[w, ch] * 1000, 1).tolist())
    print("   H(raw)  :", np.round(hx[w] * 1000, 1).tolist())
    print("   2nd diff: track", round(float(np.max(np.abs(np.diff(t[w, ch], 2)))) * 1000, 1),
          " raw", round(float(np.max(np.abs(np.diff(hx[w], 2)))) * 1000, 1))
