"""Neural click detector as an ``evaluate.py`` method (spec section 8).

Scores are ClickNet logits from an exported ONNX model, aligned to the input
(``logit[n]`` predicts target sample ``n - D``, so the first D outputs are
dropped and D zeros are appended to the features). Regions come from
``classical.regions_from_scores`` with additive logit-space hysteresis and
repair is ``classical.lsar_fill`` with streaming-sized context.

The output delay ``D`` is read, in order of preference, from the ONNX
model's ``metadata_props`` key ``"D"``, from a sibling ``weights.json``, or
from ``NeuralConfig.D``.

With ``stream=1`` the logits come from ``nn/stream.py`` instead of
onnxruntime: ``StreamingNet.from_export`` loads ``weights.bin`` +
``weights.json`` from the model's directory, and ``StreamingDetector`` is fed
the audio in blocks of 512, followed by the same D zero feature frames the
offline path appends, so both paths see identical inputs.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pressing_tools import classical
from pressing_tools.nn.features import features_np

STREAM_BLOCK = 512
DEFAULT_THRESHOLDS = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


# MARK: - Configuration


@dataclass
class NeuralConfig:
    model: str = ""            # path to .onnx
    threshold: float = 3.0     # logit; final value chosen on val
    ext_drop: float = 2.0      # extend_threshold = threshold - ext_drop (logit units)
    merge: int = 8
    pre: int = 2
    post: int = 8
    max_region: int = 480      # 10 ms; longer = pop, passed through (out of scope)
    min_contrast: float = 0.0  # optional gate, swept on val
    contrast_context: int = 150
    repair_order: int = 32
    repair_context: int = 1024
    repair_context_right: int = 512
    max_group: int = 480
    stream: int = 0            # 1 = compute logits with nn/stream.py (block 512) instead of ORT
    D: int = 512               # output delay fallback when neither the model nor weights.json says


# MARK: - Model loading (cached per process)


_SESSIONS: dict[tuple[str, float], object] = {}
_STREAM_NETS: dict[tuple[str, float], object] = {}


def _session(path: str):
    """One onnxruntime session per (model file, mtime) per process."""
    p = os.path.abspath(path)
    key = (p, os.path.getmtime(p))
    sess = _SESSIONS.get(key)
    if sess is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(p, sess_options=opts, providers=["CPUExecutionProvider"])
        _SESSIONS[key] = sess
    return sess


def _find_d(obj) -> int | None:
    """``D`` at the top level of a JSON header, else in any nested dict (e.g. the NetConfig)."""
    if not isinstance(obj, dict):
        return None
    if "D" in obj:
        return int(obj["D"])
    for v in obj.values():
        d = _find_d(v)
        if d is not None:
            return d
    return None


def resolve_delay(model: str, fallback: int) -> int:
    meta = _session(model).get_modelmeta().custom_metadata_map
    if "D" in meta:
        return int(meta["D"])
    wj = Path(model).with_name("weights.json")
    if wj.exists():
        d = _find_d(json.loads(wj.read_text()))
        if d is not None:
            return d
    return int(fallback)


# MARK: - Method


class NeuralMethod:
    name = "neural"
    stereo = True
    default_thresholds = DEFAULT_THRESHOLDS

    def __init__(self, cfg: NeuralConfig | None = None):
        self.cfg = cfg = cfg or NeuralConfig()
        if not cfg.model:
            raise ValueError("NeuralConfig.model must be set, e.g. --set model=out/nn/<run>/export/model.onnx")
        if not Path(cfg.model).exists():
            raise FileNotFoundError(f"model not found: {cfg.model}")
        self.D = resolve_delay(cfg.model, cfg.D)
        self.det_cfg = classical.ClassicalConfig(
            threshold=cfg.threshold, extend_threshold=cfg.threshold - cfg.ext_drop,
            merge=cfg.merge, pre=cfg.pre, post=cfg.post, max_region=cfg.max_region,
            min_contrast=cfg.min_contrast, contrast_context=cfg.contrast_context)
        self.rep_cfg = classical.ClassicalConfig(
            repair_order=cfg.repair_order, repair_context=cfg.repair_context,
            repair_context_right=cfg.repair_context_right, max_group=cfg.max_group)
        self._stream = _stream_module() if cfg.stream else None

    @property
    def default_threshold(self) -> float:
        return self.cfg.threshold

    # MARK: Scores

    def _padded_features(self, x2: np.ndarray) -> tuple[np.ndarray, int]:
        f = features_np(x2).astype(np.float32)          # (2, 3, L)
        L = f.shape[-1]
        T = -(-(L + self.D) // 4) * 4
        return np.pad(f, ((0, 0), (0, 0), (0, T - L))), L

    def scores_stereo(self, x2: np.ndarray) -> np.ndarray:
        """x2: (2, L) (or (1, L) mono) -> (2, L) logits aligned to the input samples."""
        if self.cfg.stream:
            return self._scores_stream(x2)
        f, L = self._padded_features(x2)
        sess = _session(self.cfg.model)
        name = sess.get_inputs()[0].name
        logit = sess.run(None, {name: f})[0]
        return np.asarray(logit[:, self.D:self.D + L], np.float64)

    def _scores_stream(self, x2: np.ndarray) -> np.ndarray:
        """Same logits via the streaming reference, fed STREAM_BLOCK audio samples at a time."""
        x2 = np.atleast_2d(np.asarray(x2, np.float64))
        L = x2.shape[-1]
        net = self._stream_net()
        net.reset()
        det = self._stream.StreamingDetector(net)
        out = [det.process(x2[:, i:i + STREAM_BLOCK]) for i in range(0, L, STREAM_BLOCK)]
        # Flush: the held-back partial frame plus D zero feature samples, as offline.
        T = -(-(L + self.D) // 4) * 4
        tail = det.pending
        tail = np.concatenate([tail, np.zeros((2, tail.shape[1], T - L), tail.dtype)], axis=-1)
        out += [net.process(tail[..., i:i + STREAM_BLOCK]) for i in range(0, tail.shape[-1], STREAM_BLOCK)]
        logit = np.concatenate(out, axis=-1)
        assert logit.shape[-1] == T, (logit.shape, T)
        return np.asarray(logit[:, self.D:self.D + L], np.float64)

    def _stream_net(self):
        export_dir = Path(self.cfg.model).resolve().parent
        wj = export_dir / "weights.json"
        if not wj.exists():
            raise FileNotFoundError(f"stream=1 needs weights.bin + weights.json next to the model: {wj}")
        key = (str(export_dir), os.path.getmtime(wj))
        net = _STREAM_NETS.get(key)
        if net is None:
            net = _STREAM_NETS[key] = self._stream.StreamingNet.from_export(export_dir)
        return net

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Mono fallback: the channel is its own 'other' channel."""
        return self.scores_stereo(np.asarray(x)[None])[0]

    # MARK: Regions and repair

    def mask(self, x: np.ndarray, s: np.ndarray, thr: float) -> np.ndarray:
        cfg = replace(self.det_cfg, threshold=thr, extend_threshold=thr - self.cfg.ext_drop)
        return classical.regions_from_scores(s, cfg, thr, x=x)

    def repair(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return classical.lsar_fill(x, m, self.rep_cfg)


def _stream_module():
    try:
        stream = importlib.import_module("pressing_tools.nn.stream")
    except ImportError as e:
        raise RuntimeError("stream=1 needs pressing_tools.nn.stream (StreamingDetector), "
                           "which is not available in this checkout") from e
    for name in ("StreamingDetector", "StreamingNet"):
        if not hasattr(stream, name):
            raise RuntimeError(f"pressing_tools.nn.stream has no {name} class")
    return stream
