"""Estimate the transfer function and coherence between a raw side segment and the exported track."""
import sys
from pathlib import Path
import numpy as np
from scipy import signal
from pressing_tools import library as lib

side, track, offset = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sr = lib.probe(side)["sr"]
t = lib.decode(track).astype(np.float64)
raw = lib.decode(side, start=offset, frames=len(t)).astype(np.float64)
n = min(len(t), len(raw)); t, raw = t[:n], raw[:n]
print("DC  raw:", raw.mean(0).round(5), " track:", t.mean(0).round(5))
print("RMS raw:", np.sqrt((raw**2).mean(0)).round(4), " track:", np.sqrt((t**2).mean(0)).round(4))
for a, b, lbl in ((0, 0, "L->L"), (1, 1, "R->R"), (0, 1, "L->R"), (1, 0, "R->L")):
    f, Pxy = signal.csd(raw[:, a], t[:, b], sr, nperseg=8192)
    _, Pxx = signal.welch(raw[:, a], sr, nperseg=8192)
    _, C = signal.coherence(raw[:, a], t[:, b], sr, nperseg=8192)
    H = Pxy / Pxx
    pick = [10, 20, 40, 100, 300, 1000, 3000, 8000, 12000, 16000, 20000]
    idx = [int(np.argmin(np.abs(f - p))) for p in pick]
    print(lbl, "coherence:", " ".join(f"{p}Hz={C[i]:.3f}" for p, i in zip(pick, idx)))
    if a == b:
        print("     |H| dB :", " ".join(f"{p}Hz={20*np.log10(abs(H[i])):+.1f}" for p, i in zip(pick, idx)))
        print("     phase deg:", " ".join(f"{p}Hz={np.degrees(np.angle(H[i])):+.0f}" for p, i in zip(pick, idx)))
# drift: local lag in 10 s windows
for k in range(0, n - sr * 12, sr * 40):
    a = raw[k:k + sr * 10, 0]; b = t[k:k + sr * 10, 0]
    c = signal.correlate(b, a, mode="full", method="fft"); lag = int(np.argmax(np.abs(c))) - (len(a) - 1)
    print(f"t={k/sr:6.1f}s local lag {lag}")
