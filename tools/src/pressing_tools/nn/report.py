"""Pareto curves, matched-FA recall and paired album bootstrap (spec section 10).

Input: one or more ``pressing-eval`` aggregate JSONs (``<tag>.json``) and,
for the bootstrap, their per-shard files ``<tag>.shards.json`` (list of
{"shard", "album", "counts": {str(threshold): asdict(Counts)}}).

Per method:
- the Pareto envelope of label-1 recall vs false alarms/min over thresholds
  (points no other threshold beats on both axes);
- recall at 25/50/100 FA/min, linearly interpolated in log(FA) between
  envelope points (target below the lowest FA: the lowest-FA point; above the
  highest: the highest-FA point);
- FA/min at a matched recall (default 0.74), interpolated in log(FA) vs recall.

The first JSON is the baseline (e.g. classical-stream). For every other
method, a paired album bootstrap (2000 resamples of albums with replacement,
fixed seed) gives the 95% CI of (method - baseline) recall@50 FA/min: each
resample re-aggregates both methods' Counts per threshold from the per-shard
counts of the drawn albums and recomputes each method's recall@50 FA/min.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FA_FLOOR = 1e-3            # FA/min below this is treated as this in log(FA)
FIELDS = ("seconds", "l1_events", "l1_found", "l2_events", "l2_found", "pred_false")


# MARK: - Curves


@dataclass
class Point:
    threshold: float
    fa_per_min: float
    recall: float           # label 1 (repaired)
    recall_l2: float        # label 2 (detected only)


def point_from_totals(thr: float, tot: dict | np.ndarray) -> Point:
    """tot: Counts-like dict, or an array ordered as FIELDS."""
    v = dict(zip(FIELDS, tot)) if isinstance(tot, np.ndarray) else tot
    return Point(float(thr), float(v["pred_false"] / max(v["seconds"] / 60, 1e-9)),
                 float(v["l1_found"] / max(1, v["l1_events"])),
                 float(v["l2_found"] / max(1, v["l2_events"])))


def points_from_aggregate(agg: dict) -> list[Point]:
    return [Point(float(t), d["false_alarms_per_min"], d["recall_repaired"], d["recall_detected_only"])
            for t, d in agg["detection"].items()]


def pareto(points: list[Point]) -> list[Point]:
    """Non-dominated points, FA ascending and recall strictly increasing."""
    env: list[Point] = []
    # A threshold with no false alarms and no hits predicts (almost) nothing; as an
    # anchor at FA=0 it would make log-FA interpolation flatter than the data.
    points = [p for p in points if not (p.fa_per_min == 0 and p.recall == 0)]
    for p in sorted(points, key=lambda p: (p.fa_per_min, -p.recall)):
        if not env or p.recall > env[-1].recall:
            env.append(p)
    return env


def _lfa(fa: float) -> float:
    return math.log(max(fa, FA_FLOOR))


def recall_at_fa(env: list[Point], target: float) -> dict:
    """Recall (and label-2 recall) on the envelope at `target` FA/min."""
    if not env:
        return {"recall": float("nan"), "recall_l2": float("nan"), "note": "empty"}
    lt = _lfa(target)
    if lt < _lfa(env[0].fa_per_min):
        p = env[0]
        return {"recall": p.recall, "recall_l2": p.recall_l2, "note": f"FA target not reached; lowest-FA point thr {p.threshold:g}"}
    for p0, p1 in zip(env, env[1:]):
        l0, l1 = _lfa(p0.fa_per_min), _lfa(p1.fa_per_min)
        if l0 <= lt < l1:
            f = (lt - l0) / (l1 - l0)
            return {"recall": p0.recall + f * (p1.recall - p0.recall),
                    "recall_l2": p0.recall_l2 + f * (p1.recall_l2 - p0.recall_l2),
                    "note": f"between thr {p0.threshold:g} and {p1.threshold:g}"}
    p = env[-1]
    return {"recall": p.recall, "recall_l2": p.recall_l2, "note": f"highest-FA point thr {p.threshold:g}"}


def fa_at_recall(env: list[Point], target: float) -> dict:
    """FA/min on the envelope at a matched label-1 recall (None if never reached)."""
    if not env or target > env[-1].recall:
        return {"fa_per_min": None, "note": "recall not reached"}
    if target <= env[0].recall:
        return {"fa_per_min": env[0].fa_per_min, "note": f"lowest-FA point thr {env[0].threshold:g}"}
    for p0, p1 in zip(env, env[1:]):
        if p0.recall < target <= p1.recall:
            f = (target - p0.recall) / (p1.recall - p0.recall)
            l0, l1 = _lfa(p0.fa_per_min), _lfa(p1.fa_per_min)
            return {"fa_per_min": math.exp(l0 + f * (l1 - l0)),
                    "note": f"between thr {p0.threshold:g} and {p1.threshold:g}"}
    raise AssertionError("unreachable")


# MARK: - Album bootstrap


def shards_path(agg_path: Path) -> Path:
    return agg_path.with_name(agg_path.name.removesuffix(".json") + ".shards.json")


def album_tensor(entries: list[dict], albums: list[str]) -> tuple[list[float], np.ndarray]:
    """(thresholds, array (n_albums, n_thresholds, len(FIELDS))) summed per album."""
    thr_keys = sorted(entries[0]["counts"], key=float)
    pos = {a: i for i, a in enumerate(albums)}
    M = np.zeros((len(albums), len(thr_keys), len(FIELDS)))
    for e in entries:
        for j, t in enumerate(thr_keys):
            c = e["counts"][t]
            M[pos[e["album"]], j] += [c[f] for f in FIELDS]
    return [float(t) for t in thr_keys], M


def recall_at_fa_totals(thresholds: list[float], tot: np.ndarray, target: float) -> float:
    return recall_at_fa(pareto([point_from_totals(t, tot[j]) for j, t in enumerate(thresholds)]),
                        target)["recall"]


def paired_bootstrap(entries_a: list[dict], entries_b: list[dict], target_fa: float = 50.0,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI of recall@target_fa(a) - recall@target_fa(b), resampling albums.

    Only shards present in both methods are used (pairing)."""
    sa, sb = {e["shard"]: e for e in entries_a}, {e["shard"]: e for e in entries_b}
    common = sorted(set(sa) & set(sb))
    if not common:
        raise ValueError("no shards in common between the two methods")
    ea, eb = [sa[s] for s in common], [sb[s] for s in common]
    albums = sorted({e["album"] for e in ea})
    ta, Ma = album_tensor(ea, albums)
    tb, Mb = album_tensor(eb, albums)
    point_a = recall_at_fa_totals(ta, Ma.sum(0), target_fa)
    point_b = recall_at_fa_totals(tb, Mb.sum(0), target_fa)

    rng = np.random.default_rng(seed)
    n = len(albums)
    idx = rng.integers(n, size=(n_boot, n))
    W = np.zeros((n_boot, n))
    np.add.at(W, (np.arange(n_boot)[:, None], idx), 1.0)
    Ta, Tb = np.einsum("ba,atf->btf", W, Ma), np.einsum("ba,atf->btf", W, Mb)
    diffs = np.array([recall_at_fa_totals(ta, Ta[b], target_fa) - recall_at_fa_totals(tb, Tb[b], target_fa)
                      for b in range(n_boot)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"target_fa_per_min": target_fa, "diff": point_a - point_b, "recall_a": point_a,
            "recall_b": point_b, "ci95": [float(lo), float(hi)], "albums": n, "shards": len(common),
            "dropped_shards": len(set(sa) ^ set(sb)), "resamples": n_boot, "seed": seed,
            "excludes_zero": bool(lo > 0 or hi < 0)}


# MARK: - Report


def method_report(agg_path: Path, fa_targets: list[float], match_recall: float) -> dict:
    agg = json.loads(agg_path.read_text())
    sp = shards_path(agg_path)
    shards = json.loads(sp.read_text()) if sp.exists() else None
    pts = points_from_aggregate(agg)
    env = pareto(pts)
    return {"name": agg_path.name.removesuffix(".json"), "path": str(agg_path),
            "config": agg.get("config", {}), "points": [asdict(p) for p in pts],
            "envelope": [asdict(p) for p in env],
            "recall_at_fa": {str(f): recall_at_fa(env, f) for f in fa_targets},
            "fa_at_recall": {str(match_recall): fa_at_recall(env, match_recall)},
            "_shards": shards}


def format_method(m: dict) -> str:
    lines = [f"\n=== {m['name']} ===", f"{'thr':>7} {'FA/min':>9} {'recall':>8} {'recall l2':>9}   (Pareto envelope)"]
    for p in m["envelope"]:
        lines.append(f"{p['threshold']:7g} {p['fa_per_min']:9.2f} {p['recall']:8.1%} {p['recall_l2']:9.1%}")
    for f, r in m["recall_at_fa"].items():
        lines.append(f"  recall @ {float(f):g} FA/min: {r['recall']:.1%} (label 2 {r['recall_l2']:.1%}; {r['note']})")
    for rc, r in m["fa_at_recall"].items():
        fa = "not reached" if r["fa_per_min"] is None else f"{r['fa_per_min']:.2f}"
        lines.append(f"  FA/min @ recall {float(rc):.0%}: {fa} ({r['note']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-nn-report", description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="+", type=Path,
                    help="pressing-eval aggregate JSONs; the first is the bootstrap baseline")
    ap.add_argument("--fa", default="25,50,100", help="FA/min targets for recall")
    ap.add_argument("--match-recall", type=float, default=0.74)
    ap.add_argument("--boot", type=int, default=2000, help="album bootstrap resamples (0 = skip)")
    ap.add_argument("--boot-fa", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    a = ap.parse_args(argv)

    fa_targets = [float(f) for f in a.fa.split(",")]
    methods = [method_report(p, fa_targets, a.match_recall) for p in a.results]
    for m in methods:
        print(format_method(m))

    comparisons = []
    base = methods[0]
    if a.boot > 0 and len(methods) > 1:
        print(f"\n=== paired album bootstrap: recall @ {a.boot_fa:g} FA/min, method - {base['name']} ===")
    for m in methods[1:] if a.boot > 0 else []:
        if base["_shards"] is None or m["_shards"] is None:
            missing = [x["name"] for x in (base, m) if x["_shards"] is None]
            print(f"  {m['name']}: skipped, no .shards.json for {', '.join(missing)}")
            continue
        b = paired_bootstrap(m["_shards"], base["_shards"], a.boot_fa, a.boot, a.seed)
        b.update(method=m["name"], baseline=base["name"])
        comparisons.append(b)
        print(f"  {m['name']}: {b['diff'] * 100:+.1f} points (95% CI {b['ci95'][0] * 100:+.1f} .. "
              f"{b['ci95'][1] * 100:+.1f}; {b['albums']} albums, {b['shards']} shards"
              f"{', excludes 0' if b['excludes_zero'] else ''})")

    if a.out:
        for m in methods:
            m.pop("_shards")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"methods": methods, "bootstrap": comparisons}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
