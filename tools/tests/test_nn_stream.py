"""Streaming reference vs offline torch vs ORT (spec section 12)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from pressing_tools.classical import ClassicalConfig, regions_from_scores  # noqa: E402
from pressing_tools.nn.export import (DEFAULT_DATA, export_random, ort_logits,  # noqa: E402
                                      ort_session, synthetic_audio, torch_logits)
from pressing_tools.nn.features import features_np  # noqa: E402
from pressing_tools.nn.model import NetConfig  # noqa: E402
from pressing_tools.nn.stream import StreamingDetector, StreamingNet  # noqa: E402

SECONDS = 60
THRESHOLD = 3.0


# MARK: - Fixtures


def _audio() -> tuple[np.ndarray, str]:
    """(2, SECONDS*48000) float32: concatenated val windows if present, else noise + impulses."""
    n_win = SECONDS // 2
    try:
        from pressing_tools.memmap import open_split

        mm = open_split(DEFAULT_DATA, "val")
        a = mm["audio"]
        if a.shape[0] >= n_win:
            idx = np.linspace(0, a.shape[0] - 1, n_win).astype(int)
            return np.concatenate(list(np.asarray(a[idx], np.float32)), axis=-1), "val"
    except (FileNotFoundError, OSError):
        pass
    x = synthetic_audio(n_win, seed=1)
    return np.concatenate(list(x), axis=-1), "synthetic"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    net, summary = export_random(out, NetConfig(), seed=3, post={"threshold": THRESHOLD},
                                 n_golden=2, n_check=2, verbose=False)
    x, source = _audio()
    feat = features_np(x).astype(np.float32)                   # (2, 3, L)
    L = feat.shape[-1] - feat.shape[-1] % net.cfg.stride
    x, feat = x[:, :L], feat[..., :L]
    offline = torch_logits(net, feat)
    return {"net": net, "out": out, "summary": summary, "x": x, "feat": feat,
            "offline": offline, "source": source}


def _stream(net, x, seed=0, lo=1, hi=512, mult=4):
    """Feed x through a fresh StreamingDetector in random block sizes mult*[lo, hi]."""
    rng = np.random.default_rng(seed)
    det = StreamingDetector(StreamingNet.from_torch(net))
    outs, i = [], 0
    while i < x.shape[1]:
        n = mult * int(rng.integers(lo, hi + 1))
        outs.append(det.process(x[:, i:i + n]))
        i += n
    return np.concatenate(outs, axis=-1)


def _masks(s, x):
    cfg = ClassicalConfig(threshold=THRESHOLD, extend_threshold=THRESHOLD - 2.0, merge=8, pre=2,
                          post=8, max_region=480, min_contrast=0.0)
    return np.stack([regions_from_scores(s[c].astype(np.float64), cfg, THRESHOLD, x=x[c])
                     for c in range(2)])


# MARK: - Equivalence


def test_residual_branches_are_live(setup):
    net = setup["net"]
    assert all(b.mix.weight.abs().max() > 0 for b in net.blocks)


def test_stream_vs_torch_vs_ort(setup):
    net, x, feat, offline = setup["net"], setup["x"], setup["feat"], setup["offline"]
    streamed = _stream(net, x, seed=0)                      # blocks 4..2048, multiples of 4
    ort = ort_logits(ort_session(setup["out"] / "model.onnx"), feat)
    assert streamed.shape == offline.shape == ort.shape == x.shape
    d_stream = np.abs(streamed - offline).max()
    d_ort = np.abs(ort - offline).max()
    print(f"{setup['source']} {x.shape[1] / 48000:.0f} s: stream {d_stream:.2e}, ort {d_ort:.2e}")
    assert d_stream < 1e-4 and d_ort < 1e-4
    ref = _masks(offline, x)
    assert ref.any(), "threshold never crossed; the comparison would be vacuous"
    assert np.array_equal(_masks(streamed, x), ref)
    assert np.array_equal(_masks(ort, x), ref)


def test_any_block_size_detector(setup):
    """Arbitrary (non-multiple-of-4) raw blocks: output covers complete frames only."""
    net, x, offline = setup["net"], setup["x"][:, :48000], setup["offline"]
    streamed = _stream(net, x, seed=1, lo=1, hi=700, mult=1)
    assert streamed.shape == (2, 48000)
    assert np.abs(streamed - offline[:, :48000]).max() < 1e-4


def test_start_state_is_zero_padding(setup):
    """A single block from a fresh state equals the offline pass (zero history)."""
    net, feat, offline = setup["net"], setup["feat"], setup["offline"]
    sn = StreamingNet.from_torch(net)
    got = sn.process(feat[..., :4096])
    assert np.abs(got - offline[:, :4096]).max() < 1e-4
    sn.reset()
    assert np.array_equal(sn.process(feat[..., :4096]), got)
    assert sn.state_floats() == 12 + 2 * 255 * 16


# MARK: - Export artefacts


def test_weights_roundtrip(setup):
    net, out, feat = setup["net"], setup["out"], setup["feat"][..., :8192]
    hdr = json.loads((out / "weights.json").read_text())
    assert hdr["net_config"] == net.cfg.to_dict() and hdr["D"] == net.cfg.D
    assert set(hdr["features"]) >= {"EPS", "FLOOR", "NW", "ASINH_SCALE"}
    assert hdr["post"]["threshold"] == THRESHOLD
    end = 0
    for layer in hdr["layers"]:
        assert layer["offset"] == end
        end += layer["nbytes"]
    assert end == (out / "weights.bin").stat().st_size == hdr["total_bytes"]
    a = StreamingNet.from_export(out).process(feat)
    b = StreamingNet.from_torch(net).process(feat)
    assert np.array_equal(a, b)


def test_onnx_ops_and_golden(setup):
    s = setup["summary"]
    assert set(s["ops"]) <= {"Conv", "Relu", "Add", "Transpose", "Flatten", "Reshape"}
    assert s["ort_vs_torch_max_abs"] < 1e-4
    meta = ort_session(setup["out"] / "model.onnx").get_modelmeta().custom_metadata_map
    assert int(meta["D"]) == setup["net"].cfg.D
    g = np.load(setup["out"] / "golden.npz")
    B, _, L = g["audio"].shape
    assert g["features"].shape == (B, 2, 3, L) and g["logits"].shape == (B, 2, L)
    assert g["masks"].shape == (B, 2, L) and g["masks"].dtype == bool
    np.testing.assert_allclose(features_np(g["audio"][0]), g["features"][0], atol=1e-6)
    lg = torch_logits(setup["net"], g["features"][0])
    np.testing.assert_allclose(lg, g["logits"][0], atol=1e-5)
    # scores are the D-shifted logits (section 9); away from the right edge they match
    D = int(g["D"])
    np.testing.assert_allclose(g["scores"][0][:, :L - D - 4], g["logits"][0][:, D:L - 4], atol=1e-4)


# MARK: - CLI on a train-format checkpoint


def test_cli_train_checkpoint_and_post(tmp_path):
    """Checkpoints as nn/train.py writes them (numpy scalars in 'curve', empty
    'post') load, and --post / --post-json land fully merged in weights.json and
    golden.npz, with golden masks at the given threshold."""
    from pressing_tools.nn.export import POST_DEFAULTS, main, random_model

    net = random_model(NetConfig(), seed=5)
    ck = tmp_path / "best.pt"
    torch.save({"model": net.state_dict(), "net_config": net.cfg.to_dict(), "post": {"merge": 6},
                "step": 10, "metric": np.float64(0.5),
                "curve": {"recall_repaired": [np.float64(0.9)], "tp": np.int64(3)}}, ck)
    (tmp_path / "chosen.json").write_text(json.dumps({"min_contrast": 8, "pre": 3}))
    out = tmp_path / "exp"
    args = ["--ckpt", str(ck), "--out", str(out), "--data", str(tmp_path / "none"),
            "--golden", "2", "--check", "2", "--post-json", str(tmp_path / "chosen.json"),
            "--post", "threshold=2.5", "--post", "pre=4"]
    assert main(args) == 0
    want = {**POST_DEFAULTS, "merge": 6, "min_contrast": 8.0, "pre": 4, "threshold": 2.5}
    hdr = json.loads((out / "weights.json").read_text())
    assert hdr["post"] == want
    g = np.load(out / "golden.npz")
    assert json.loads(str(g["post"])) == want and float(g["threshold"]) == 2.5
    cfg = ClassicalConfig(threshold=2.5, extend_threshold=0.5, merge=6, pre=4, post=8,
                          max_region=480, min_contrast=8.0, contrast_context=150)
    for b in range(2):
        for c in range(2):
            m = regions_from_scores(g["scores"][b, c].astype(np.float64), cfg, 2.5,
                                    x=g["audio"][b, c].astype(np.float64))
            assert np.array_equal(m, g["masks"][b, c])
    # bad keys / values are rejected before anything is written
    assert main(["--ckpt", str(ck), "--out", str(tmp_path / "x"), "--post", "thresh=2"]) == 2
    assert main(["--ckpt", str(ck), "--out", str(tmp_path / "x"), "--post", "merge=2.5"]) == 2
    assert not (tmp_path / "x").exists()
