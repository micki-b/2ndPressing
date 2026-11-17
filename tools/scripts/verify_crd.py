"""Check which .crd repair records were applied in exported (iTunes) tracks.

For each record inside a track we measure how far the de-clicked track
departs from a filter-matched raw side at that span (z = peak residual /
local residual floor). Applied repairs give large z; untouched spans give z
like random control spans. A second pass pastes the stored kind-1/2
replacement samples into the raw audio first: if the export used exactly those
samples, their z collapses back to the control level. Impulses remaining after
pasting that no record explains are counted as "unexplained" edits.

usage:
  verify_crd.py one SIDE TRACK OFFSET
  verify_crd.py sample N [--tracks-per-album K] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pressing_tools import library as lib
from pressing_tools.crd import PAYLOAD_KINDS, read_crd
from pressing_tools.labels import LabelConfig, detect_regions, local_sigma, match_residual

RAW = Path("/Volumes/Media RAID/Music/VinylStudio/Michael")
ITUNES = Path("/Volumes/Media RAID/Music/iTunes/Media/Music")
Z_APPLIED = 8.0


# MARK: - Single track


def _span_z(resid, sigma, ch, s, e) -> float:
    return float(np.max(np.abs(resid[s:e, ch])) / np.median(sigma[s:e, ch]))


def verify(side: Path, track: Path, offset: int) -> dict:
    sr = lib.probe(side)["sr"]
    t = lib.decode(track).astype(np.float64)
    raw = lib.decode(side, start=offset, frames=len(t)).astype(np.float64)
    n = min(len(t), len(raw))
    t, raw = t[:n], raw[:n]
    nch = t.shape[1]
    _, recs = read_crd(side.with_name(side.name + ".crd"))
    recs = [r for r in recs if r.start >= offset and r.end <= offset + n and r.length > 0]

    pasted = raw.copy()
    for r in recs:
        if r.kind in PAYLOAD_KINDS:
            pasted[r.start - offset:r.end - offset, r.channel] = r.replacement / 32768.0

    cfg = LabelConfig()
    res, floors = {}, {}
    for name, src in (("raw", raw), ("pasted", pasted)):
        resid = np.stack([match_residual(src[:, c], t[:, c], cfg, sr) for c in range(nch)], 1)
        sigma = np.stack([local_sigma(resid[:, c], cfg.subblock) for c in range(nch)], 1)
        res[name] = (resid, sigma)
        floors[name] = float(20 * np.log10(np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(t ** 2))))

    covered = np.zeros(t.shape, bool)
    for r in recs:
        covered[max(0, r.start - offset - 64):r.end - offset + 64, r.channel] = True

    rng = np.random.default_rng(0)
    z = defaultdict(list)
    for r in recs:
        s, e = r.start - offset, r.end - offset
        z[str(r.kind)].append((_span_z(*res["raw"], r.channel, s, e),
                               _span_z(*res["pasted"], r.channel, s, e)))
        for _ in range(20):
            cs = int(rng.integers(64, n - r.length - 64))
            if not covered[cs:cs + r.length, r.channel].any():
                z["control"].append((_span_z(*res["raw"], r.channel, cs, cs + r.length),) * 2)
                break

    # Impulsive edits left after pasting that no record explains.
    valid = np.ones(n, bool)
    valid[:cfg.long_nperseg // 8] = valid[-cfg.long_nperseg // 8:] = False  # filter edges
    # Spans where the export sits at full scale are export clipping (gain
    # applied after repair), not declicking; count them separately.
    unexplained, clipped = [], 0
    for c in range(nch):
        for reg in detect_regions(res["pasted"][0][:, c], t[:, c], valid, c, cfg, sr):
            if covered[reg.start:reg.end, c].any():
                continue
            if np.max(np.abs(t[reg.start:reg.end, c])) >= 0.999:
                clipped += 1
            else:
                unexplained.append((c, reg.start, reg.end - reg.start, round(reg.peak, 5)))

    return {
        "side": side.name, "track": track.name, "offset": offset, "sr": sr,
        "seconds": round(n / sr, 1), "floor_raw_db": round(floors["raw"], 1),
        "floor_pasted_db": round(floors["pasted"], 1),
        "kinds": {k: {"n": len(v),
                      "applied_raw": float(np.mean(np.array(v)[:, 0] > Z_APPLIED)),
                      "applied_pasted": float(np.mean(np.array(v)[:, 1] > Z_APPLIED)),
                      "z_raw_p50": float(np.median(np.array(v)[:, 0]))}
                  for k, v in z.items()},
        "unexplained": len(unexplained), "unexplained_examples": unexplained[:5],
        "clipped_spans": clipped,
        "export_peak": float(np.max(np.abs(t))),
    }


def print_one(r: dict) -> None:
    print(f"== {r['side'][:60]} | {r['track'][:40]} @ {r['offset']} ({r['sr']} Hz, {r['seconds']} s)")
    print(f"   floor raw {r['floor_raw_db']} dB, pasted {r['floor_pasted_db']} dB | "
          f"clipped spans {r['clipped_spans']} | unexplained {r['unexplained']} {r['unexplained_examples'][:3]}")
    for k in sorted(r["kinds"]):
        v = r["kinds"][k]
        print(f"   kind {k:>7}: n={v['n']:>5}  changed(raw) {v['applied_raw']:6.1%}  "
              f"changed(after paste) {v['applied_pasted']:6.1%}  z50 {v['z_raw_p50']:.1f}")


# MARK: - Library sample


def _album_job(album_dir: str, sides: list[str], k: int, seed: int) -> list[dict]:
    album = lib.Album(Path(album_dir), [s for s in lib.find_sides(RAW) if str(s.path) in sides])
    tracks = album.tracks
    random.Random(seed).shuffle(tracks)
    out = []
    coarse = {s.path: lib.decode(s.path, sr=2000, mono=True)[:, 0] for s in album.sides}
    for t in tracks:
        if len(out) >= k:
            break
        if lib.probe(t)["codec"] != "alac":
            continue
        best = max((lib.locate(t, s.path, coarse[s.path]) for s in album.sides),
                   key=lambda p: p.score)
        if not best.same_rate:
            out.append({"track": t.name, "skipped": "sample rate differs from side"})
            continue
        try:
            out.append(verify(best.side, t, best.offset) | {"locate_score": round(best.score, 3)})
        except Exception as e:  # keep the sample going
            out.append({"track": t.name, "error": str(e)[:200]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one"); one.add_argument("side", type=Path)
    one.add_argument("track", type=Path); one.add_argument("offset", type=int)
    smp = sub.add_parser("sample"); smp.add_argument("n", type=int)
    smp.add_argument("--tracks-per-album", type=int, default=2)
    smp.add_argument("--seed", type=int, default=1)
    smp.add_argument("--jobs", type=int, default=4)
    smp.add_argument("--out", type=Path, default=Path("out/verify_crd.json"))
    a = ap.parse_args()

    if a.cmd == "one":
        print_one(verify(a.side, a.track, a.offset))
        return

    albums, _ = lib.match_albums(RAW, ITUNES)
    albums = [x for x in albums if all(s.crd.exists() for s in x.sides)]
    random.Random(a.seed).shuffle(albums)
    albums = albums[:a.n]
    results = []
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(_album_job, str(x.dir), [str(s.path) for s in x.sides],
                          a.tracks_per_album, a.seed) for x in albums]
        for f in futs:
            for r in f.result():
                results.append(r)
                if "kinds" in r:
                    print_one(r)
                else:
                    print("!!", r)
                sys.stdout.flush()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
