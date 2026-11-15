"""Consolidate a dataset split into memory-mapped arrays for fast random access.

The npz shards are compressed, which is fine for evaluation but far too slow
for training, where random crops are drawn constantly. This writes, per split::

    <root>/mm/<split>/audio.npy   float16 (N, 2, L)
    <root>/mm/<split>/label.npy   uint8   (N, 2, L)
    <root>/mm/<split>/repl.npy    float16 (K,)    VinylStudio replacement samples
    <root>/mm/<split>/repl_off.npy int64  (N+1,)  window w's samples are
                                                   repl[repl_off[w]:repl_off[w+1]]
    <root>/mm/<split>/index.json  per-window shard name / window / album

Open with ``open_split`` (read-only memmaps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pressing_tools.dataset import load_shard


def build(root: Path, split: str) -> Path:
    shards = sorted((root / split).glob("*.npz"))
    manifest = {json.loads(l)["shard"]: json.loads(l)
                for l in (root / "manifest.jsonl").read_text().splitlines() if l.strip()}
    # First pass: sizes (shapes only need the label array; cheap enough).
    sizes, L, C = [], None, None
    for s in shards:
        with np.load(s) as z:
            n, C, L = z["label"].shape
            sizes.append((n, len(z["repl"])))
    N, K = sum(n for n, _ in sizes), sum(k for _, k in sizes)
    out = root / "mm" / split
    out.mkdir(parents=True, exist_ok=True)
    audio = np.lib.format.open_memmap(out / "audio.npy", "w+", np.float16, (N, C, L))
    label = np.lib.format.open_memmap(out / "label.npy", "w+", np.uint8, (N, C, L))
    repl = np.lib.format.open_memmap(out / "repl.npy", "w+", np.float16, (K,))
    repl_off = np.zeros(N + 1, np.int64)
    index = []
    w0 = k0 = 0
    for s, (n, k) in zip(shards, sizes):
        sh = load_shard(s)
        audio[w0:w0 + n] = sh["audio"]
        label[w0:w0 + n] = sh["label"]
        repl[k0:k0 + k] = sh["repl"]
        per_win = (sh["label"] == 1).reshape(n, -1).sum(1)
        repl_off[w0 + 1:w0 + n + 1] = k0 + np.cumsum(per_win)
        meta = manifest.get(f"{split}/{s.name}", {})
        index += [{"shard": s.name, "window": i, "album": f"{meta.get('artist', '')}|{meta.get('album', '')}"}
                  for i in range(n)]
        w0, k0 = w0 + n, k0 + k
    audio.flush(); label.flush(); repl.flush()
    np.save(out / "repl_off.npy", repl_off)
    (out / "index.json").write_text(json.dumps(index))
    return out


def open_split(root: Path, split: str) -> dict:
    d = root / "mm" / split
    return {
        "audio": np.load(d / "audio.npy", mmap_mode="r"),
        "label": np.load(d / "label.npy", mmap_mode="r"),
        "repl": np.load(d / "repl.npy", mmap_mode="r"),
        "repl_off": np.load(d / "repl_off.npy"),
        "index": json.loads((d / "index.json").read_text()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-memmap", description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--splits", default="val,test,train")
    a = ap.parse_args(argv)
    for split in a.splits.split(","):
        out = build(a.data, split)
        print(f"{split}: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
