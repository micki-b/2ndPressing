"""Parity fixtures for the Rust runtime (raw little-endian arrays + manifest.json).

From an export dir's golden.npz (val windows) this writes, per window w and
channel c (all float32 unless noted, shape (W, 2, L)):
    audio.bin      raw input
    features.bin   (W, 2, 3, L) model input (features_np, float64 -> f32)
    scores.bin     target-aligned logits (offline ONNX/torch reference)
    masks.bin      uint8 region masks (harness post-processing at the export threshold)
    repaired.bin   full offline pipeline output (NeuralMethod mask + LSAR repair)
plus manifest.json with shapes, dtypes, threshold, post and repair settings.

usage: export_rust_fixtures.py EXPORT_DIR OUT_DIR [--windows N] [--samples L]
       export_rust_fixtures.py EXPORT_DIR OUT_DIR --audio FILE --offset S --samples L [--source TEXT]

With --audio, the input is one stereo window of FILE (decoded to 48 kHz,
starting at S seconds) instead of golden.npz's validation windows. The
committed fixtures use this with a public-domain 78, so no library audio is
in the repository; --source records where the audio came from.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pressing_tools.library import decode
from pressing_tools.nn.features import features_np
from pressing_tools.nn.method import NeuralConfig, NeuralMethod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--samples", type=int, default=0, help="truncate windows to this many samples")
    ap.add_argument("--audio", type=Path, help="take one window from this file instead of golden.npz")
    ap.add_argument("--offset", type=float, default=0.0, help="start of the --audio window, seconds")
    ap.add_argument("--source", default="", help="provenance note for manifest.json")
    a = ap.parse_args()

    w = json.loads((a.export_dir / "weights.json").read_text())
    post = w["post"]
    thr = float(post["threshold"])
    if a.audio:
        if not a.samples:
            ap.error("--audio needs --samples")
        x = decode(a.audio, sr=48000)                      # (frames, channels)
        if x.shape[1] != 2:
            ap.error("--audio must be stereo")
        start = round(a.offset * 48000)
        audio = x[start:start + a.samples].T[None].astype(np.float32)   # (1, 2, L)
        if audio.shape[-1] != a.samples:
            ap.error("--offset + --samples runs past the end of the file")
    else:
        g = np.load(a.export_dir / "golden.npz")
        audio = g["audio"][:a.windows].astype(np.float32)
        if a.samples:
            audio = audio[..., :a.samples]
    W, C, L = audio.shape
    method = NeuralMethod(NeuralConfig(model=str(a.export_dir / "model.onnx"), threshold=thr,
                                       ext_drop=post["ext_drop"], merge=post["merge"], pre=post["pre"],
                                       post=post["post"], max_region=post["max_region"],
                                       min_contrast=post["min_contrast"],
                                       contrast_context=post["contrast_context"]))
    feats = np.zeros((W, C, 3, L), np.float32)
    scores = np.zeros((W, C, L), np.float32)
    masks = np.zeros((W, C, L), np.uint8)
    rep = np.zeros((W, C, L), np.float32)
    for i in range(W):
        x = audio[i].astype(np.float64)
        feats[i] = features_np(x).astype(np.float32)
        s = method.scores_stereo(x)
        scores[i] = s
        for c in range(C):
            m = method.mask(x[c], s[c], thr)
            masks[i, c] = m
            rep[i, c] = method.repair(x[c], m)
    a.out.mkdir(parents=True, exist_ok=True)
    arrays = {"audio": audio, "features": feats, "scores": scores, "masks": masks, "repaired": rep}
    for k, v in arrays.items():
        v.astype(v.dtype.newbyteorder("<")).tofile(a.out / f"{k}.bin")
    manifest = {"windows": W, "channels": C, "samples": L, "sample_rate": 48000,
                "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()},
                "threshold": thr, "post": post,
                "repair": {"order": method.rep_cfg.repair_order, "context": method.rep_cfg.repair_context,
                           "context_right": method.rep_cfg.repair_context_right,
                           "max_group": method.rep_cfg.max_group},
                "D": int(w["D"]), "edge_ignore": 2400,
                "source": a.source or "golden.npz validation windows",
                "notes": "Compare masks/repaired on [edge_ignore, samples - edge_ignore); the offline "
                         "reference zero-pads features at the window end. Unmasked samples of repaired "
                         "equal audio exactly."}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {W} windows x {L} samples to {a.out}; flagged samples {int(masks.sum())}, "
          f"regions per window-channel ~{sum(len(np.flatnonzero(np.diff(np.r_[0, masks[i, c], 0]) == 1)) for i in range(W) for c in range(C)) / (W * C):.1f}")


if __name__ == "__main__":
    main()
