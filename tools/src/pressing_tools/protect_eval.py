"""Benchmark for percussion-protection methods on long-form audio.

A *protector* decides, for each detected region, whether to repair it
(True) or leave it alone because it is music (False)::

    class Protector:
        name: str
        lookahead_s: float      # future audio it needs beyond the base detector
        def keep(self, x2, s2, regions, sr) -> list[list[bool]]:
            # x2: (2, N) float64 audio, s2: (2, N) target-aligned logits,
            # regions: per channel list of (a, b) from NeuralMethod.mask
            ...

Load one with ``module:Class`` (e.g. ``pressing_tools.protect.rhythm:Protector``).

Benchmarks (all at 48 kHz, regions from the selected R2 model):
  longform   data/v1/longform/<split>  real .crd labels (1 repaired, 2 detected-only)
  synth      the same audio + ~1 real click shape per channel-second injected at
             random places (including on beats); original labels become
             don't-care, so recall = clicks a protector would wrongly spare
  mirage     test/Fleetwood_Mac~Mirage_sample.m4a with VinylStudio's labels for that
             span: recall in the quiet transition (4-13 s), false alarms in the
             clapping song (14-27 s)

    python -m pressing_tools.protect_eval [--protector mod:Class] [--split val]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pressing_tools import classical, paths
from pressing_tools.evaluate import Counts, detection_counts

ROOT = Path(__file__).resolve().parents[3]          # 2ndPressing/
DATA = ROOT / "data" / "v1"
MODEL = ROOT / "tools" / "out" / "nn" / "r2_c24_d512" / "export" / "model.onnx"
MIRAGE = ROOT / "test" / "Fleetwood_Mac~Mirage_sample.m4a"
THRESHOLDS = (2.75, 3.4, 4.05)          # plugin sensitivity 50 / 40 / 30 %
SR = 48000


# MARK: - Shared helpers


class KeepAll:
    name = "baseline"
    lookahead_s = 0.0

    def keep(self, x2, s2, regions, sr):
        return [[True] * len(r) for r in regions]


def load_protector(spec: str | None):
    if not spec:
        return KeepAll()
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def _method():
    from pressing_tools.nn.method import NeuralConfig, NeuralMethod
    return NeuralMethod(NeuralConfig(model=str(MODEL), min_contrast=0.0))


def scores_cached(x2: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    s = _method().scores_stereo(x2).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, s)
    return s


def regions_for(x2, s2, thr) -> list[list[tuple[int, int]]]:
    m = _method()
    return [classical.runs(m.mask(x2[c], s2[c].astype(np.float64), thr)) for c in range(2)]


def counts_with(x2, s2, lab, protector, thr) -> Counts:
    regions = regions_for(x2, s2, thr)
    keep = protector.keep(x2, s2, regions, SR)
    tot = Counts()
    for c in range(2):
        m = np.zeros(x2.shape[1], bool)
        for (a, b), k in zip(regions[c], keep[c]):
            if k:
                m[a:b] = True
        tot.add(detection_counts(lab[c], m))
    return tot


# MARK: - Synthetic injection


def inject(x2: np.ndarray, lab: np.ndarray, bank: np.ndarray, meta: np.ndarray, seed: int):
    """~1 click per channel-second at random positions >= 256 from labels;
    contrast-scaled (log-uniform 8..100) like nn/synth.py."""
    rng = np.random.default_rng(seed)
    x = x2.copy()
    out = np.where(lab > 0, 2, 0).astype(np.uint8)
    near = np.zeros_like(lab, bool)
    for c in range(2):
        idx = np.flatnonzero(lab[c] > 0)
        if len(idx):
            m = np.zeros(lab.shape[1] + 1, np.int32)
            np.add.at(m, np.clip(idx - 256, 0, None), 1)
            np.add.at(m, np.clip(idx + 257, None, lab.shape[1]), -1)
            near[c] = np.cumsum(m)[:-1] > 0
    N = x.shape[1]
    for c in range(2):
        for pos in np.sort(rng.choice(np.arange(4800, N - 4800), int(N / SR), replace=False)):
            k = int(rng.integers(len(bank)))
            L = int(meta[k]["length"])
            if near[c, pos - 64:pos + L + 64].any() or out[c, pos - 600:pos + L + 600].any():
                continue
            t = bank[k][8:8 + L].astype(np.float64)
            dt = np.abs(np.diff(t, 2)).max() + 1e-12
            ctx = np.abs(np.diff(x[c, pos - 170:pos + L + 170], 2))
            med = np.median(np.r_[ctx[:150], ctx[-150:]]) + 1e-12
            cstar = math.exp(rng.uniform(math.log(8), math.log(100)))
            click = t * cstar * med / dt * rng.choice([-1, 1])
            if np.abs(x[c, pos:pos + L] + click).max() > 0.999:
                continue
            x[c, pos:pos + L] += click
            out[c, pos:pos + L] = 1
    return x, out


# MARK: - Benchmarks


def _seg_job(path: str, spec: str | None, synth: bool) -> dict:
    protector = load_protector(spec)
    z = np.load(path)
    x2, lab = z["audio"].astype(np.float64), z["label"]
    tag = "synth" if synth else "real"
    if synth:
        nd = DATA / "nn" / Path(path).parent.name
        x2, lab = inject(x2, lab, np.load(nd / "bank.npy"), np.load(nd / "bank_meta.npy"),
                         seed=int(hashlib.sha1(Path(path).stem.encode()).hexdigest()[:8], 16))
    s2 = scores_cached(x2, Path(path).parent.parent / f"{Path(path).parent.name}_scores_{tag}" /
                       (Path(path).stem + ".npy"))
    return {str(t): asdict(counts_with(x2, s2, lab, protector, t)) for t in THRESHOLDS}


def longform(split: str, spec: str | None, synth: bool, jobs: int) -> dict:
    files = sorted(str(f) for f in (DATA / "longform" / split).glob("*.npz"))
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_seg_job, files, [spec] * len(files), [synth] * len(files)))
    out = {}
    for t in THRESHOLDS:
        tot = Counts()
        for r in res:
            tot.add(Counts(**r[str(t)]))
        s = tot.summary()
        out[str(t)] = {k: s[k] for k in ("recall_repaired", "recall_detected_only",
                                         "false_alarms_per_min", "clean_touched_pct")}
    return out


def mirage(spec: str | None) -> dict:
    from pressing_tools import library as lib
    from pressing_tools.crd import PAYLOAD_KINDS, read_crd
    protector = load_protector(spec)
    x2 = lib.decode(MIRAGE).T.astype(np.float64)
    N = x2.shape[1]
    cache = DATA / "longform" / "mirage.json"
    mirage_side = paths.get("protect_side")
    if cache.exists():
        off = json.loads(cache.read_text())["offset"]
    else:
        off = lib.locate(MIRAGE, mirage_side).offset
        cache.write_text(json.dumps({"offset": off}))
    lab = np.zeros((2, N), np.uint8)
    _, recs = read_crd(mirage_side.with_name(mirage_side.name + ".crd"))
    for r in recs:
        if off <= r.start < off + N and r.length:
            a, b = r.start - off, min(r.end - off, N)
            if r.kind in PAYLOAD_KINDS:
                lab[r.channel, a:b] = 1
            elif not lab[r.channel, a:b].any():
                lab[r.channel, a:b] = 2
    s2 = scores_cached(x2, DATA / "longform" / "mirage_scores.npy")
    out = {}
    for t in THRESHOLDS:
        regions = regions_for(x2, s2, t)
        keep = protector.keep(x2, s2, regions, SR)
        res = {}
        for name, lo, hi in (("transition_4_13s", 4, 13), ("clap_song_14_27s", 14, 27)):
            tot = Counts()
            kept_regions = 0
            for c in range(2):
                m = np.zeros(N, bool)
                for (a, b), k in zip(regions[c], keep[c]):
                    if k and lo * SR <= a < hi * SR:
                        m[a:b] = True
                        kept_regions += 1
                sl = slice(lo * SR, hi * SR)
                tot.add(detection_counts(lab[c, sl], m[sl]))
            sm = tot.summary()
            res[name] = {"repaired_regions": kept_regions, "recall_repaired": sm["recall_repaired"],
                         "false_alarms_per_min": sm["false_alarms_per_min"]}
        out[str(t)] = res
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="protect-eval", description=__doc__.split("\n")[0])
    ap.add_argument("--protector", default=None, help="module:Class (default: no protection)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    p = load_protector(a.protector)
    res = {"protector": a.protector or "baseline", "lookahead_s": p.lookahead_s,
           "longform": longform(a.split, a.protector, False, a.jobs),
           "synth": longform(a.split, a.protector, True, a.jobs),
           "mirage": mirage(a.protector)}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
