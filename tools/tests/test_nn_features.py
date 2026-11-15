import numpy as np
import pytest

from pressing_tools.nn.features import FLOOR, FeatureState, features_np, features_torch

torch = pytest.importorskip("torch")


def _signal(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 48000
    x = np.stack([0.2 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(n),
                  0.1 * np.sin(2 * np.pi * 330 * t) + 0.01 * rng.standard_normal(n)])
    x[0, 5000:5020] += 0.3          # a click
    x[1, 12000:12010] -= 0.2
    return x


def test_layout_and_cross_channel():
    x = _signal()
    f = features_np(x)
    assert f.shape == (2, 3, x.shape[1])
    assert np.array_equal(f[0, 2], f[1, 0]) and np.array_equal(f[1, 2], f[0, 0])
    assert np.abs(f[0, 0, 5000:5021]).max() > 1.0 > np.abs(f[0, 0, 1000:4000]).max() * 0.5


def test_torch_matches_numpy():
    x = _signal(12288)
    ref = features_np(x)
    got = features_torch(torch.tensor(x[None], dtype=torch.float32)).numpy()
    assert got.shape == (2, 3, 12288)
    assert np.max(np.abs(got - ref)) < 1e-4


def test_streaming_matches_offline_any_block_size():
    x = _signal(30000)
    ref = features_np(x)
    rng = np.random.default_rng(1)
    st, out, i = FeatureState(), [], 0
    while i < x.shape[1]:
        n = int(rng.integers(1, 4097))
        out.append(st.process(x[:, i:i + n]))
        i += n
    got = np.concatenate(out, axis=-1)
    assert np.max(np.abs(got - ref)) < 1e-9


def test_gain_invariance_and_silence():
    x = _signal()
    f1, f10 = features_np(x), features_np(10 * x)
    diff = np.abs(f1 - f10)[:, :, 100:]   # EPS breaks exactness only where |d| ~ 1e-6
    assert diff.max() < 5e-3 and np.median(diff) < 1e-4
    assert not np.any(features_np(np.zeros((2, 5000))))
    assert FLOOR > 0
