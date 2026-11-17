"""Evaluation harness: score any detector + repairer on a dataset split.

A *method* provides three steps per channel (1-D float arrays at 48 kHz):

    scores(x)            -> per-sample score (higher = more click-like)
    mask(x, scores, thr) -> boolean region mask at a threshold
    repair(x, mask)      -> repaired signal

A method with ``stereo = True`` instead provides ``scores_stereo(x2)``
((2, L) -> (2, L)), called once per window; mask and repair stay per channel.
An optional ``default_thresholds`` attribute sets the sweep when
``--thresholds`` is not given.

Detection is reported over a sweep of thresholds (event recall vs. false
alarms per minute); repair quality at the method's default threshold. Labels:
1 = repaired by VinylStudio, 2 = detected but left alone (reported separately,
neither rewarded nor penalised). "False alarms" are predicted regions touching
no label -- note that this also counts real clicks VinylStudio didn't flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from pressing_tools import classical
from pressing_tools.dataset import LABEL_CLEAN, LABEL_DETECTED, LABEL_REPAIRED, load_shard, repaired

CLASSICAL_THRESHOLDS = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0]
EDGE = 2400          # samples ignored at each window edge (filter warm-up)
TOL = 16             # samples of slack when matching predictions to labels
SR = 48000


# MARK: - Methods


class ClassicalMethod:
    name = "classical"

    def __init__(self, cfg: classical.ClassicalConfig | None = None):
        self.cfg = cfg or classical.ClassicalConfig()

    @property
    def default_threshold(self) -> float:
        return self.cfg.threshold

    def scores(self, x: np.ndarray) -> np.ndarray:
        return classical.scores(x, self.cfg)

    def mask(self, x: np.ndarray, s: np.ndarray, thr: float) -> np.ndarray:
        return classical.regions_from_scores(s, self.cfg, thr, x=x)

    def repair(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return classical.lsar_fill(x, m, self.cfg)


def make_method(name: str, overrides: dict):
    if name == "classical":
        return ClassicalMethod(replace(classical.ClassicalConfig(), **overrides))
    if name == "neural":
        from pressing_tools.nn.method import NeuralConfig, NeuralMethod
        return NeuralMethod(NeuralConfig(**overrides))
    raise ValueError(f"unknown method {name}")


def method_scores(method, x2: np.ndarray) -> np.ndarray:
    """Per-channel scores for one window x2 (C, L): one stereo call, or one call per channel."""
    if getattr(method, "stereo", False):
        return np.asarray(method.scores_stereo(x2))
    return np.stack([method.scores(x2[c]) for c in range(x2.shape[0])])


def parse_value(v: str):
    """--set values: int, then float, then str (so paths work)."""
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


# MARK: - Metric helpers


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or not m.any():
        return m
    return np.convolve(m.astype(np.int8), np.ones(2 * k + 1, np.int8), mode="same") > 0


def _any_in(cs: np.ndarray, s: int, e: int) -> bool:
    return cs[e] - cs[s] > 0


def _d2(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[1:-1] = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    return y


def spike_ratio(x: np.ndarray, s: int, e: int, ctx: int = 150) -> float | None:
    """Peak curvature inside [s, e) over the median curvature around it."""
    lo, hi = s - ctx, e + ctx
    if lo < 2 or hi > len(x) - 2:
        return None
    d = _d2(x[lo:hi])
    inner = d[s - lo:e - lo]
    around = np.r_[d[:max(1, s - lo - 20)], d[e - lo + 20:]]
    return float(inner.max() / (np.median(around) + 1e-12))


def _snr(ref: np.ndarray, est: np.ndarray) -> float:
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-20) / (np.sum((ref - est) ** 2) + 1e-20)))


@dataclass
class Counts:
    seconds: float = 0.0
    l1_events: int = 0
    l1_found: int = 0
    l2_events: int = 0
    l2_found: int = 0
    pred_regions: int = 0
    pred_on_l1: int = 0
    pred_on_l2_only: int = 0
    pred_false: int = 0
    l1_samples: int = 0
    l1_samples_covered: int = 0
    clean_samples: int = 0
    clean_samples_touched: int = 0

    def add(self, o: "Counts") -> None:
        for k, v in asdict(o).items():
            setattr(self, k, getattr(self, k) + v)

    def summary(self) -> dict:
        minutes = self.seconds / 60
        return {
            "recall_repaired": self.l1_found / max(1, self.l1_events),
            "recall_detected_only": self.l2_found / max(1, self.l2_events),
            "false_alarms_per_min": self.pred_false / max(minutes, 1e-9),
            "precision_vs_any_label": (self.pred_on_l1 + self.pred_on_l2_only) / max(1, self.pred_regions),
            "repaired_samples_covered": self.l1_samples_covered / max(1, self.l1_samples),
            "clean_touched_pct": 100 * self.clean_samples_touched / max(1, self.clean_samples),
            "events": {"repaired": self.l1_events, "detected_only": self.l2_events,
                       "predicted": self.pred_regions},
        }


def detection_counts(lab: np.ndarray, m: np.ndarray) -> Counts:
    """lab: labels for one channel window; m: predicted mask."""
    n = len(lab)
    valid = np.zeros(n, bool)
    valid[EDGE:n - EDGE] = True
    c = Counts(seconds=(n - 2 * EDGE) / SR)
    m = m & valid
    cs_pred = np.r_[0, np.cumsum(_dilate(m, TOL))]
    l1, l2 = lab == LABEL_REPAIRED, lab == LABEL_DETECTED
    for s, e in classical.runs(l1):
        if valid[s]:
            c.l1_events += 1
            c.l1_found += _any_in(cs_pred, s, e)
    for s, e in classical.runs(l2):
        if valid[s]:
            c.l2_events += 1
            c.l2_found += _any_in(cs_pred, s, e)
    cs1, cs2 = np.r_[0, np.cumsum(_dilate(l1, TOL))], np.r_[0, np.cumsum(_dilate(l2, TOL))]
    for s, e in classical.runs(m):
        c.pred_regions += 1
        if _any_in(cs1, s, e):
            c.pred_on_l1 += 1
        elif _any_in(cs2, s, e):
            c.pred_on_l2_only += 1
        else:
            c.pred_false += 1
    c.l1_samples = int((l1 & valid).sum())
    c.l1_samples_covered = int((l1 & m).sum())
    clean = (lab == LABEL_CLEAN) & ~_dilate(lab != LABEL_CLEAN, TOL) & valid
    c.clean_samples = int(clean.sum())
    c.clean_samples_touched = int((clean & m).sum())
    return c


# MARK: - Per-shard evaluation


def evaluate_shard(path: str, method_name: str, overrides: dict, thresholds: list[float],
                   gap_trials: int, seed: int) -> dict:
    method = make_method(method_name, overrides)
    shard = load_shard(Path(path))
    audio = shard["audio"].astype(np.float64)
    label = shard["label"]
    vs = repaired(shard).astype(np.float64)
    W, C, L = audio.shape
    rng = np.random.default_rng([seed, int(hashlib.sha1(Path(path).name.encode()).hexdigest()[:8], 16)])

    per_thr = {t: Counts() for t in thresholds}
    rep = {"spike_raw": [], "spike_vs": [], "spike_ours": [], "agree_snr": [],
           "false_change_db": [], "gap": []}
    for w in range(W):
        sw = method_scores(method, audio[w])
        for c in range(C):
            x, lab, s = audio[w, c], label[w, c], sw[c]
            for t in thresholds:
                per_thr[t].add(detection_counts(lab, method.mask(x, s, t)))

            m = method.mask(x, s, method.default_threshold)
            y = method.repair(x, m)
            for a, b in classical.runs(lab == LABEL_REPAIRED):
                if a < EDGE or b > L - EDGE or not m[a:b].any():
                    continue
                r = [spike_ratio(v, a, b) for v in (x, vs[w, c], y)]
                if None not in r:
                    rep["spike_raw"].append(r[0]); rep["spike_vs"].append(r[1]); rep["spike_ours"].append(r[2])
                    rep["agree_snr"].append(_snr(vs[w, c, a:b], y[a:b]))
            near_label = _dilate(lab != LABEL_CLEAN, TOL)
            for a, b in classical.runs(m):
                if a >= EDGE and b <= L - EDGE and not near_label[a:b].any():
                    # how much did we change clean audio, relative to the local level?
                    lvl = np.sqrt(np.mean(x[max(0, a - 256):b + 256] ** 2)) + 1e-12
                    chg = np.sqrt(np.mean((y[a:b] - x[a:b]) ** 2))
                    rep["false_change_db"].append(float(20 * np.log10(chg / lvl + 1e-12)))

            # Synthetic gaps in clean audio: ground truth is known here.
            l1_lengths = [b - a for a, b in classical.runs(lab == LABEL_REPAIRED)] or [18]
            clean_ok = (lab == LABEL_CLEAN) & ~_dilate(lab != LABEL_CLEAN, 256)
            for _ in range(gap_trials):
                g = int(rng.choice(l1_lengths))
                pos = int(rng.integers(EDGE, L - EDGE - g))
                if not clean_ok[pos - 64:pos + g + 64].all():
                    continue
                gm = np.zeros(L, bool); gm[pos:pos + g] = True
                lsar = method.repair(x, gm)[pos:pos + g]
                lin = np.linspace(x[pos - 1], x[pos + g], g + 2)[1:-1]
                rep["gap"].append((g, _snr(x[pos:pos + g], lsar), _snr(x[pos:pos + g], lin)))
    return {"counts": {t: asdict(v) for t, v in per_thr.items()}, "repair": rep,
            "shard": Path(path).name, "repaired_events": int(sum(len(classical.runs(label[w, c] == 1))
                                                              for w in range(W) for c in range(C)))}


# MARK: - Listening examples


def write_listening(path: Path, method, out_dir: Path, windows: int = 2) -> list[str]:
    shard = load_shard(path)
    audio = shard["audio"].astype(np.float64)
    vs = repaired(shard)
    counts = [(int((shard["label"][w] == LABEL_REPAIRED).sum()), w) for w in range(audio.shape[0])]
    written = []
    for _, w in sorted(counts, reverse=True)[:windows]:
        sw = method_scores(method, audio[w])
        ours = np.stack([method.repair(audio[w, c], method.mask(audio[w, c], sw[c], method.default_threshold))
                         for c in range(audio.shape[1])])
        stem = f"{path.stem[:50]}_w{w}"
        for tag, data in (("1_raw", audio[w]), ("2_vinylstudio", vs[w]), ("3_ours", ours)):
            f = out_dir / f"{stem}_{tag}.wav"
            sf.write(f, data.T.astype(np.float32), SR, subtype="FLOAT")
            written.append(f.name)
    return written


# MARK: - Aggregation and CLI


def load_albums(data: Path) -> dict[str, str]:
    """Manifest shard path ("val/X.npz") and bare file name -> "artist|album"."""
    out: dict[str, str] = {}
    mf = data / "manifest.jsonl"
    if not mf.exists():
        return out
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if "shard" not in m:
            continue
        album = f"{m.get('artist', '')}|{m.get('album', '')}"
        out[m["shard"]] = album
        out.setdefault(Path(m["shard"]).name, album)
    return out


def album_for(albums: dict[str, str], split: str, shard: str) -> str:
    """Album key for a shard; shards outside the manifest get their own key ("|<stem>")."""
    return albums.get(f"{split}/{shard}") or albums.get(shard) or f"|{Path(shard).stem}"


def _plain(c: dict) -> dict:
    """numpy scalars (from cumsum comparisons) -> JSON-serialisable Python numbers."""
    return {k: v.item() if hasattr(v, "item") else v for k, v in c.items()}


def shard_records(results: list[dict], albums: dict[str, str], split: str) -> list[dict]:
    return [{"shard": r["shard"], "album": album_for(albums, split, r["shard"]),
             "counts": {str(float(t)): _plain(c) for t, c in r["counts"].items()}} for r in results]


def _safe(v) -> str:
    """Filesystem-safe rendering of an override value; paths become a readable tail plus a hash."""
    s = str(v)
    if re.fullmatch(r"[A-Za-z0-9.=+-]{1,40}", s):
        return s
    tail = re.sub(r"[^A-Za-z0-9.=+-]+", "_", "_".join(Path(s).parts[-3:])).strip("_")
    return f"{tail[-32:]}-{hashlib.sha1(s.encode()).hexdigest()[:8]}"


def make_tag(method: str, split: str, overrides: dict) -> str:
    split_tag = re.sub(r"[^A-Za-z0-9.=+-]+", "-", split.strip("/"))
    return f"{method}_{split_tag}" + "".join(f"_{k}{_safe(v)}" for k, v in overrides.items())


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def aggregate(results: list[dict], thresholds: list[float]) -> dict:
    out = {"detection": {}, "repair": {}}
    for t in thresholds:
        tot = Counts()
        for r in results:
            tot.add(Counts(**r["counts"][t]))
        out["detection"][str(t)] = tot.summary()
    rep = {k: sum((r["repair"][k] for r in results), []) for k in results[0]["repair"]}
    gap = np.array(rep["gap"]) if rep["gap"] else np.zeros((0, 3))
    bins = [(1, 12), (12, 24), (24, 48), (48, 10_000)]
    out["repair"] = {
        "real_clicks_repaired": len(rep["spike_ours"]),
        "spike_ratio_median": {"raw": _pct(rep["spike_raw"], 50), "vinylstudio": _pct(rep["spike_vs"], 50),
                               "ours": _pct(rep["spike_ours"], 50)},
        "spike_ratio_p90": {"raw": _pct(rep["spike_raw"], 90), "vinylstudio": _pct(rep["spike_vs"], 90),
                            "ours": _pct(rep["spike_ours"], 90)},
        "agreement_with_vinylstudio_snr_db_median": _pct(rep["agree_snr"], 50),
        "false_alarm_change_db_median": _pct(rep["false_change_db"], 50),
        "false_alarm_change_db_p90": _pct(rep["false_change_db"], 90),
        "synthetic_gaps": {
            f"{a}-{b}": {"n": int(((gap[:, 0] >= a) & (gap[:, 0] < b)).sum()),
                         "lsar_snr_db_median": _pct(gap[(gap[:, 0] >= a) & (gap[:, 0] < b), 1], 50),
                         "linear_snr_db_median": _pct(gap[(gap[:, 0] >= a) & (gap[:, 0] < b), 2], 50)}
            for a, b in bins},
    }
    return out


def print_report(agg: dict, method: str, split: str, default_thr: float) -> None:
    print(f"\n=== {method} on {split} ===")
    print(f"{'thr':>6} | {'recall rep':>10} {'recall det':>10} | {'false/min':>9} {'prec(any)':>9} "
          f"| {'rep cov':>7} {'clean touched':>13}")
    for t, d in agg["detection"].items():
        star = "*" if float(t) == default_thr else " "
        print(f"{t:>5}{star} | {d['recall_repaired']:10.1%} {d['recall_detected_only']:10.1%} | "
              f"{d['false_alarms_per_min']:9.1f} {d['precision_vs_any_label']:9.1%} | "
              f"{d['repaired_samples_covered']:7.1%} {d['clean_touched_pct']:12.3f}%")
    r = agg["repair"]
    print(f"\nRepair at threshold {default_thr} ({r['real_clicks_repaired']} real clicks):")
    print(f"  spike ratio median  raw {r['spike_ratio_median']['raw']:.1f}  "
          f"VinylStudio {r['spike_ratio_median']['vinylstudio']:.1f}  ours {r['spike_ratio_median']['ours']:.1f}")
    print(f"  spike ratio p90     raw {r['spike_ratio_p90']['raw']:.1f}  "
          f"VinylStudio {r['spike_ratio_p90']['vinylstudio']:.1f}  ours {r['spike_ratio_p90']['ours']:.1f}")
    print(f"  agreement with VinylStudio repair: {r['agreement_with_vinylstudio_snr_db_median']:.1f} dB SNR (median)")
    print(f"  change made at false alarms: median {r['false_alarm_change_db_median']:.1f} dB, "
          f"p90 {r['false_alarm_change_db_p90']:.1f} dB re local level")
    print("  synthetic gaps in clean music (SNR vs truth, median):")
    for k, v in r["synthetic_gaps"].items():
        print(f"    {k:>9} samples  n={v['n']:5d}  LSAR {v['lsar_snr_db_median']:6.1f} dB   "
              f"linear {v['linear_snr_db_median']:6.1f} dB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-eval", description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--split", default="val", help="any subdirectory of --data, e.g. val or synth/val_lvl2")
    ap.add_argument("--method", default="classical")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="method config override, e.g. --set threshold=10")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated sweep (default: the method's own list)")
    ap.add_argument("--gap-trials", type=int, default=3, help="synthetic gaps per channel-window")
    ap.add_argument("--limit", type=int, default=0, help="only the first N shards")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--listen", type=int, default=0, help="write listening examples from N shards")
    ap.add_argument("--out", type=Path, default=Path("out/eval"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
    method = make_method(a.method, overrides)
    if a.thresholds:
        base = [float(t) for t in a.thresholds.split(",")]
    else:
        base = [float(t) for t in getattr(method, "default_thresholds", CLASSICAL_THRESHOLDS)]
    thresholds = sorted(set(base) | {float(method.default_threshold)})
    split_dir = a.data / a.split
    if not split_dir.is_dir():
        ap.error(f"{split_dir} is not a directory")
    shards = sorted(split_dir.glob("*.npz"))
    if not shards:
        ap.error(f"no .npz shards in {split_dir}")
    if a.limit:
        shards = shards[:a.limit]
    print(f"{len(shards)} shards from {a.split}; thresholds {thresholds}", flush=True)

    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(evaluate_shard, [str(s) for s in shards], [a.method] * len(shards),
                              [overrides] * len(shards), [thresholds] * len(shards),
                              [a.gap_trials] * len(shards), [a.seed] * len(shards)))
    agg = aggregate(results, thresholds)
    agg["config"] = {"method": a.method, "overrides": overrides, "split": a.split,
                     "shards": len(shards)}
    a.out.mkdir(parents=True, exist_ok=True)
    tag = make_tag(a.method, a.split, overrides)
    (a.out / f"{tag}.json").write_text(json.dumps(agg, indent=2))
    (a.out / f"{tag}.shards.json").write_text(
        json.dumps(shard_records(results, load_albums(a.data), a.split), indent=1))
    print_report(agg, a.method, a.split, float(method.default_threshold))

    if a.listen:
        ld = a.out / "listen"
        ld.mkdir(parents=True, exist_ok=True)
        best = sorted(results, key=lambda r: -r["repaired_events"])[:a.listen]
        for r in best:
            write_listening(split_dir / r["shard"], method, ld)
        print(f"\nListening examples written to {ld}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
