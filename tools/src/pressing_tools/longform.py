"""Long-form evaluation set: contiguous minutes of raw audio with .crd labels.

The training windows are 2 s long, too short to judge methods that use
seconds of context (e.g. rhythm-aware percussion protection). This builds,
per side of a split (48 kHz sides only), K contiguous segments::

    data/v1/longform/<split>/<side stem>_s<k>.npz
        audio  float32 (2, N)    raw side audio
        label  uint8   (2, N)    0 clean, 1 repaired by VinylStudio (crd kind 1/2),
                                 2 detected but left alone (kind 11/12)
        start  int64             segment start in side samples
        side, artist, album

    python -m pressing_tools.longform --split val --segments 3 --seconds 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pressing_tools import library as lib
from pressing_tools import paths
from pressing_tools.crd import PAYLOAD_KINDS, read_crd
from pressing_tools.dataset import gather


def _sides_for(data: Path, split: str) -> list[dict]:
    rows = [json.loads(l) for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["split"] == split and r["sr_native"] == 48000]


def build_side(meta: dict, out_dir: str, segments: int, seconds: float, seed: int) -> list[str]:
    path = paths.raw_dir() / meta["side"]
    info = lib.probe(path)
    n = int(round(info["duration"] * 48000))
    N = int(seconds * 48000)
    margin = 48000 * 5
    if n < N + 2 * margin:
        return []
    rng = np.random.default_rng([seed, int(hashlib.sha1(meta["side"].encode()).hexdigest()[:8], 16)])
    # Non-overlapping segments, uniformly placed.
    slots = (n - 2 * margin) // N
    picks = sorted(rng.choice(slots, min(segments, slots), replace=False))
    spans = [(margin + int(k) * N, margin + int(k) * N + N) for k in picks]
    chunks = gather(path, info["channels"], spans)
    _, recs = read_crd(path.with_name(path.name + ".crd"))
    out = []
    for k, ((s0, s1), chunk) in enumerate(zip(spans, chunks)):
        lab = np.zeros((2, N), np.uint8)
        for r in recs:
            if r.end <= s0 or r.start >= s1 or r.length == 0:
                continue
            a, b = max(r.start, s0) - s0, min(r.end, s1) - s0
            val = 1 if r.kind in PAYLOAD_KINDS else 2
            seg = lab[r.channel, a:b]
            np.maximum(seg, 2, out=seg) if val == 2 else seg.fill(1)
        # A repair wins over a detection; re-apply repairs in case a detection overwrote one.
        for r in recs:
            if r.kind in PAYLOAD_KINDS and r.end > s0 and r.start < s1:
                lab[r.channel, max(r.start, s0) - s0:min(r.end, s1) - s0] = 1
        f = Path(out_dir) / f"{Path(meta['side']).stem}_s{k}.npz"
        np.savez(f, audio=chunk.T.astype(np.float32), label=lab, start=np.int64(s0),
                 side=meta["side"], artist=meta["artist"], album=meta["album"])
        out.append(f.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-longform", description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    out = a.data / "longform" / a.split
    out.mkdir(parents=True, exist_ok=True)
    sides = _sides_for(a.data, a.split)
    with ProcessPoolExecutor(a.jobs) as ex:
        done = list(ex.map(build_side, sides, [str(out)] * len(sides), [a.segments] * len(sides),
                           [a.seconds] * len(sides), [a.seed] * len(sides)))
    files = [f for d in done for f in d]
    print(f"{a.split}: {len(sides)} sides at 48 kHz -> {len(files)} segments of {a.seconds:.0f} s in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
