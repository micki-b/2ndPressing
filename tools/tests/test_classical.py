import numpy as np
from scipy import signal

from pressing_tools.classical import ClassicalConfig, declick, detect, lsar_fill, runs

SR = 48000


def _snr(ref, est):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2))


def _music(n, rng):
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f, a in
            [(196, .2), (293.7, .1), (440, .08), (659, .05), (1318, .02), (2637, .01)])
    b, a = signal.butter(2, [100, 6000], btype="band", fs=SR)
    return x + signal.lfilter(b, a, rng.standard_normal(n)) * 0.005


def _gap(x, length, at=20000):
    mask = np.zeros(len(x), bool)
    mask[at:at + length] = True
    return mask


def test_lsar_is_near_exact_on_pure_tones():
    t = np.arange(SR) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(196, .2), (440, .1), (1318, .03)])
    # Long gaps are limited by estimating the AR model from finite context.
    for length, floor in ((8, 60), (32, 60), (96, 18)):
        mask = _gap(x, length)
        y = lsar_fill(x, mask)
        assert _snr(x[mask], y[mask]) > floor, length
        assert np.array_equal(y[~mask], x[~mask])   # nothing else touched


def test_lsar_on_noisy_music_degrades_gracefully():
    # The noise component is unpredictable, so long gaps can't be recovered
    # exactly; real clicks are mostly 10-30 samples at 48 kHz.
    x = _music(SR, np.random.default_rng(0))
    for length, floor in ((8, 25), (32, 12)):
        mask = _gap(x, length)
        assert _snr(x[mask], lsar_fill(x, mask)[mask]) > floor, length


def test_detects_clicks_without_flagging_drums():
    rng = np.random.default_rng(1)
    n = SR * 4
    x = _music(n, rng)
    # Drum-like transients: must not be flagged.
    for pos in rng.integers(SR // 2, n - SR // 2, 6):
        k = np.arange(12000)
        env = np.minimum(k / 96, 1.0) * np.exp(-k / 1500)   # 2 ms attack, decays to ~0
        b, a = signal.butter(2, 150, fs=SR)
        x[pos:pos + 12000] += 3.2 * env * signal.lfilter(b, a, rng.standard_normal(12000))
    clean = x.copy()
    spans = []
    for pos in range(SR // 4, n - SR // 4, SR // 8):
        length = int(rng.integers(3, 40))
        k = np.arange(length)
        x[pos:pos + length] += rng.uniform(0.05, 0.3) * rng.choice([-1, 1]) * np.exp(-k / (length / 3))
        spans.append((pos, pos + length))

    mask = detect(x)
    found = sum(mask[s:e].any() for s, e in spans)
    assert found / len(spans) > 0.95
    clicks = np.zeros(n, bool)
    for s, e in spans:
        clicks[max(0, s - 16):e + 16] = True
    false_regions = [r for r in runs(mask) if not clicks[r[0]:r[1]].any()]
    assert len(false_regions) <= 2, false_regions[:5]

    y, _ = declick(x)
    inner = np.zeros(n, bool)
    inner[SR // 8:-SR // 8] = True
    # A perfect mask gains ~16 dB here (noise inside gaps is unpredictable);
    # the detector's own mask should get most of the way.
    assert _snr(clean[inner], y[inner]) > _snr(clean[inner], x[inner]) + 8
