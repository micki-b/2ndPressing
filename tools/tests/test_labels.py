"""Synthetic end-to-end test for click-label extraction.

Builds an 'original' (music + hiss + clicks) and a 'de-clicked' version that
mimics the real library: clicks removed, plus gain, EQ, hiss reduction and a
time offset. Then checks the extracted labels against the known click
positions.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from pressing_tools.labels import process_pair

SR = 44100


def _music(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    tones = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3))
                for f, a in [(110, .2), (220, .12), (330, .08), (523, .06), (1760, .02)])
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
    # Band-limited noise stands in for dense instrumentation.
    b, a = signal.butter(4, [200, 4000], btype="band", fs=SR)
    texture = signal.lfilter(b, a, rng.standard_normal(n)) * 0.05
    # Drum-like transients that must NOT be labelled (present in both versions).
    drums = np.zeros(n)
    for pos in rng.integers(SR, n - SR, 20):
        k = np.arange(2000)
        drums[pos:pos + 2000] += 0.3 * np.exp(-k / 300) * rng.standard_normal(2000)
    return tones * env + texture + drums


def _clicks(n: int, rng: np.random.Generator, count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    x = np.zeros(n)
    spans = []
    positions = np.sort(rng.choice(np.arange(SR // 2, n - SR // 2, 600), count, replace=False))
    for pos in positions:
        length = int(rng.integers(4, 80))
        k = np.arange(length)
        amp = rng.uniform(0.05, 0.6) * rng.choice([-1, 1])
        x[pos:pos + length] += amp * np.exp(-k / (length / 4)) * np.cos(k * rng.uniform(0.3, 2.0))
        spans.append((int(pos), int(pos + length)))
    return x, spans


def test_recovers_clicks_through_gain_eq_offset_and_nr(tmp_path: Path):
    rng = np.random.default_rng(7)
    n = SR * 20
    music = _music(n, rng)
    hiss = rng.standard_normal(n) * 0.003
    clicks, spans = _clicks(n, rng, 60)

    original = music + hiss + clicks
    # "De-clicked" product: clicks gone, hiss reduced, gentle shelf EQ, gain, offset.
    b, a = signal.butter(1, 3000, btype="high", fs=SR)
    processed = music + 0.3 * hiss
    processed = 0.8 * (processed + 0.25 * signal.lfilter(b, a, processed))
    lag = 123
    declicked = np.concatenate([np.zeros(lag), processed])[:n]

    op, dp = tmp_path / "orig.wav", tmp_path / "clean.wav"
    sf.write(op, np.stack([original, original * 0.9], 1).astype(np.float32), SR, subtype="FLOAT")
    sf.write(dp, np.stack([declicked, declicked * 0.9], 1).astype(np.float32), SR, subtype="FLOAT")

    report, regions = process_pair(op, dp, tmp_path / "out")

    assert report.lag_samples == lag
    assert report.drift_samples == 0
    assert report.usable, report.warnings

    ch0 = [r for r in regions if r.channel == 0]
    hit = sum(any(r.start <= s < r.end or s <= r.start < e for r in ch0) for s, e in spans)
    recall = hit / len(spans)
    false_pos = sum(not any(r.start < e and s < r.end for s, e in spans) for r in ch0)

    assert recall >= 0.95, f"recall {recall:.2f}"
    assert false_pos <= 2, f"{false_pos} false positives"
    assert (tmp_path / "out" / "labels.csv").exists()
    assert (tmp_path / "out" / "report.json").exists()
