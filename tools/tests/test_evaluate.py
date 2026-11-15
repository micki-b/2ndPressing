import numpy as np

from pressing_tools.evaluate import EDGE, detection_counts


def test_detection_counts_events_false_alarms_and_dont_care():
    n = 2 * EDGE + 10_000
    lab = np.zeros(n, np.uint8)
    lab[EDGE + 1000:EDGE + 1020] = 1      # repaired click: found
    lab[EDGE + 3000:EDGE + 3020] = 1      # repaired click: missed
    lab[EDGE + 5000:EDGE + 5030] = 2      # detected-only: found (neutral, not a false alarm)
    lab[10:30] = 1                        # inside the ignored edge: not counted
    m = np.zeros(n, bool)
    m[EDGE + 1005:EDGE + 1015] = True
    m[EDGE + 5000:EDGE + 5010] = True
    m[EDGE + 8000:EDGE + 8010] = True     # false alarm
    c = detection_counts(lab, m)
    assert (c.l1_events, c.l1_found) == (2, 1)
    assert (c.l2_events, c.l2_found) == (1, 1)
    assert (c.pred_regions, c.pred_on_l1, c.pred_on_l2_only, c.pred_false) == (3, 1, 1, 1)
    assert c.clean_samples_touched == 10
    s = c.summary()
    assert s["recall_repaired"] == 0.5 and s["precision_vs_any_label"] == 2 / 3


# MARK: - Stereo methods through evaluate_shard

import json  # noqa: E402

import pytest  # noqa: E402

from pressing_tools import evaluate  # noqa: E402
from pressing_tools.classical import lsar_fill  # noqa: E402


class DummyStereo:
    """Score = |x| of the channel minus |x| of the other channel; counts calls per process."""
    name = "dummy"
    stereo = True
    default_threshold = 0.5
    default_thresholds = [0.02, 0.5]
    calls = 0

    def scores_stereo(self, x2):
        DummyStereo.calls += 1
        assert x2.shape[0] == 2
        a = np.abs(x2)
        return np.stack([a[0] - 0.5 * a[1], a[1] - 0.5 * a[0]])

    def scores(self, x):
        raise AssertionError("stereo methods must not be scored per channel")

    def mask(self, x, s, thr):
        return s > thr

    def repair(self, x, m):
        return lsar_fill(x, m)


def _tiny_shard(path, W=2, L=2 * EDGE + 9600, seed=0):
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal((W, 2, L)) * 0.01).astype(np.float32)
    clean = audio.copy()
    label = np.zeros((W, 2, L), np.uint8)
    for w in range(W):
        for c, at in ((0, EDGE + 1000), (0, EDGE + 5000), (1, EDGE + 3000)):
            audio[w, c, at:at + 10] += 0.9
            label[w, c, at:at + 10] = 1
        label[w, 1, EDGE + 7000:EDGE + 7010] = 2          # detected-only, not a click in the audio
    np.savez_compressed(path, audio=audio.astype(np.float16), label=label,
                        repl=clean[label == 1].astype(np.float16),
                        starts=np.arange(W) * L, sr_native=48000, sr=48000)
    return W


def test_evaluate_shard_stereo_path(tmp_path, monkeypatch):
    p = tmp_path / "Artist~Album~side1.npz"
    W = _tiny_shard(p)
    monkeypatch.setattr(evaluate, "make_method", lambda name, ov: DummyStereo())
    DummyStereo.calls = 0
    r = evaluate.evaluate_shard(str(p), "dummy", {}, [0.02, 0.5], gap_trials=1, seed=0)
    assert DummyStereo.calls == W                          # once per window, not per channel
    lo, hi = r["counts"][0.02], r["counts"][0.5]
    assert lo["l1_events"] == hi["l1_events"] == 3 * W
    assert lo["l1_found"] == hi["l1_found"] == 3 * W
    assert lo["l2_events"] == hi["l2_events"] == W and hi["l2_found"] == 0
    assert hi["pred_false"] == 0 and lo["pred_false"] > 0   # noise trips the low threshold only
    assert r["repaired_events"] == 3 * W
    assert len(r["repair"]["spike_ours"]) == 3 * W
    assert np.median(r["repair"]["spike_ours"]) < np.median(r["repair"]["spike_raw"])

    # per-shard records: album from the manifest when present, own key otherwise
    albums = {"val/Artist~Album~side1.npz": "Artist|Album"}
    rec = evaluate.shard_records([r], albums, "val")
    assert rec == [{"shard": p.name, "album": "Artist|Album",
                    "counts": {"0.02": r["counts"][0.02], "0.5": r["counts"][0.5]}}]
    assert evaluate.shard_records([r], {}, "synth/val_lvl2")[0]["album"] == "|Artist~Album~side1"
    json.dumps(rec)


def test_cli_parsing_and_tags():
    assert [evaluate.parse_value(v) for v in ("3", "-1", "2.5", "out/nn/r/model.onnx")] == \
        [3, -1, 2.5, "out/nn/r/model.onnx"]
    assert evaluate.make_tag("classical", "val", {"threshold": 10, "min_contrast": 8.0}) == \
        "classical_val_threshold10_min_contrast8.0"
    tag = evaluate.make_tag("neural", "synth/val_lvl2", {"model": "out/nn/r/export/model.onnx"})
    assert "/" not in tag and tag.startswith("neural_synth-val-lvl2_model")


# MARK: - NeuralMethod with a tiny exported ClickNet


def _export(tmp_path, name, D=None):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from pressing_tools.nn.model import ClickNet, NetConfig

    torch.manual_seed(1)
    net = ClickNet(NetConfig(C=4, blocks=3)).eval()
    with torch.no_grad():
        for prm in net.parameters():
            prm.normal_(0, 0.3)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, (torch.zeros(2, 3, 256),), str(path), opset_version=17, dynamo=False,
                      input_names=["feat"], output_names=["logit"],
                      dynamic_axes={"feat": {0: "N", 2: "T"}, "logit": {0: "N", 1: "T"}})
    if D is not None:
        m = onnx.load(str(path))
        e = m.metadata_props.add()
        e.key, e.value = "D", str(D)
        onnx.save(m, str(path))
    return net, path


def test_neural_scores_stereo_alignment(tmp_path):
    import torch

    from pressing_tools.nn.features import features_np
    from pressing_tools.nn.method import NeuralConfig, NeuralMethod

    D = 64
    net, path = _export(tmp_path, "a/model.onnx", D=D)
    method = evaluate.make_method("neural", {"model": str(path)})
    assert isinstance(method, NeuralMethod) and method.stereo and method.D == D
    assert method.default_thresholds == [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    rng = np.random.default_rng(0)
    L = 4000
    x2 = rng.standard_normal((2, L)) * 0.05
    x2[0, 1500:1510] += 0.8
    s = method.scores_stereo(x2)
    assert s.shape == (2, L) and np.isfinite(s).all()
    with torch.no_grad():
        z = net(torch.from_numpy(features_np(x2).astype(np.float32))).numpy()   # (2, L), no padding
    # logit[n] predicts sample n - D: the method's score at t is torch's logit at t + D
    np.testing.assert_allclose(s[:, :L - D], z[:, D:], atol=1e-4)

    # odd lengths are padded to a multiple of 4 and trimmed back; the last full
    # frame of the shorter input ends at L - 4, so compare up to there
    s_odd = method.scores_stereo(x2[:, :L - 3])
    assert s_odd.shape == (2, L - 3)
    np.testing.assert_allclose(s_odd[:, :L - 4 - D], z[:, D:L - 4], atol=1e-4)

    # mask / repair plumbing: hysteresis in logit units, unflagged samples untouched
    m = method.mask(x2[0], s[0], float(np.quantile(s[0], 0.999)))
    y = method.repair(x2[0], m)
    assert np.array_equal(y[~m], x2[0][~m])

    # D fallbacks: sibling weights.json, then the config field
    _, p2 = _export(tmp_path, "b/model.onnx")
    (p2.parent / "weights.json").write_text(json.dumps({"net": {"C": 4, "D": 256}}))
    assert NeuralMethod(NeuralConfig(model=str(p2))).D == 256
    _, p3 = _export(tmp_path, "c/model.onnx")
    assert NeuralMethod(NeuralConfig(model=str(p3), D=128)).D == 128


def _write_weights(net, export_dir, D):
    """weights.bin + weights.json in the layout StreamingNet.from_export reads."""
    layers, blobs, off = [], [], 0
    for name, t in net.state_dict().items():
        a = t.detach().numpy().astype("<f4")
        layers.append({"name": name, "shape": list(a.shape), "offset": off})
        blobs.append(a.tobytes())
        off += a.nbytes
    (export_dir / "weights.bin").write_bytes(b"".join(blobs))
    cfg = dict(net.cfg.to_dict(), D=D)
    (export_dir / "weights.json").write_text(json.dumps({"layers": layers, "net_config": cfg, "D": D}))


def test_neural_stream_matches_onnx(tmp_path):
    pytest.importorskip("pressing_tools.nn.stream")
    from pressing_tools.nn.method import NeuralConfig, NeuralMethod

    D = 64
    net, path = _export(tmp_path, "s/model.onnx", D=D)
    _write_weights(net, path.parent, D)
    rng = np.random.default_rng(1)
    x2 = rng.standard_normal((2, 3001)) * 0.05
    x2[1, 2000:2008] -= 0.7
    off = NeuralMethod(NeuralConfig(model=str(path))).scores_stereo(x2)
    on = NeuralMethod(NeuralConfig(model=str(path), stream=1)).scores_stereo(x2)
    assert on.shape == off.shape == (2, 3001)
    np.testing.assert_allclose(on, off, atol=1e-4)


def test_neural_stream_flag_without_stream_module(tmp_path, monkeypatch):
    import sys

    from pressing_tools.nn.method import NeuralConfig, NeuralMethod

    _, path = _export(tmp_path, "model.onnx", D=64)
    monkeypatch.setitem(sys.modules, "pressing_tools.nn.stream", None)   # import now fails
    with pytest.raises(RuntimeError, match="nn.stream"):
        NeuralMethod(NeuralConfig(model=str(path), stream=1))
