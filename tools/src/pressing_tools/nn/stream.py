"""Numpy float32 streaming reference of ClickNet (spec section 12).

This is the specification the Rust port follows. Every layer keeps a fixed-size
ring buffer of its own past input, so a stream fed in blocks of any length that
is a multiple of ``stride`` produces exactly the logits of one offline pass
over the concatenated input (up to float32 summation order):

    stem      last stem_k - stride = 4 input samples x 3 features
    block d   last 2d frames x C              (d = 1, 2, ..., 128)

That is 12 + 2 * 255 * 16 = 8,172 floats per sequence at C=16. The start-up
state is all zeros, which is exactly the offline causal zero padding.

Layer arithmetic (per sequence, frame j covers input samples 4j .. 4j+3):

    h0[j]  = ReLU(b + sum_k W[:, :, k] @ f[4j - 4 + k])          k = 0..7
    hi[j]  = ReLU(h[j] + M @ ReLU(b + W0 @ h[j-2d] + W1 @ h[j-d] + W2 @ h[j]) + m)
    y[j]   = H2 @ ReLU(H1 @ h[j] + b1) + b2                       (stride, )
    logit[4j + r] = y[j][r]

``StreamingDetector`` couples ``FeatureState`` (float64) to ``StreamingNet``
and takes raw stereo audio blocks of any length.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pressing_tools.nn.features import N_FEATURES, FeatureState

DTYPE = np.float32


# MARK: - Ring buffer


class _Ring:
    """Fixed-size ring of the last ``size`` columns of an (S, K, ·) stream."""

    def __init__(self, S: int, K: int, size: int) -> None:
        self.buf = np.zeros((S, K, size), DTYPE)
        self.size = size
        self.ptr = 0                                # next write position == oldest column

    def history(self) -> np.ndarray:
        """Stored columns in chronological order (oldest first)."""
        return self.buf[..., (self.ptr + np.arange(self.size)) % self.size]

    def push(self, x: np.ndarray) -> None:
        n = x.shape[-1]
        if n >= self.size:
            self.buf[...] = x[..., n - self.size:]
            self.ptr = 0
        elif n:
            self.buf[..., (self.ptr + np.arange(n)) % self.size] = x
            self.ptr = (self.ptr + n) % self.size

    def reset(self) -> None:
        self.buf[...] = 0
        self.ptr = 0


# MARK: - Streaming network


class StreamingNet:
    """ClickNet as a block-streaming float32 computation.

    ``weights`` maps the ClickNet ``state_dict`` names (``stem.weight``,
    ``blocks.{i}.dil.weight``, ``blocks.{i}.mix.weight``, ``head1.weight``,
    ``head2.weight`` and their ``.bias``) to arrays. ``n_seq`` independent
    sequences are processed side by side (2 for stereo: L and R).
    """

    def __init__(self, weights: dict[str, np.ndarray], cfg, n_seq: int = 2) -> None:
        self.cfg = cfg
        self.n_seq = n_seq
        w = {k: np.ascontiguousarray(np.asarray(v, DTYPE)) for k, v in weights.items()}
        C, s, k = cfg.C, cfg.stride, cfg.stem_k
        assert w["stem.weight"].shape == (C, N_FEATURES, k), w["stem.weight"].shape
        # Stem as a (C, 3*k) matrix applied to im2col columns ordered (feature, tap).
        self.stem_w = w["stem.weight"].reshape(C, N_FEATURES * k)
        self.stem_b = w["stem.bias"][None, :, None]
        self.blocks = []
        for i in range(cfg.blocks):
            p = f"blocks.{i}."
            self.blocks.append({
                "d": 2 ** i,
                "taps": [w[p + "dil.weight"][:, :, t] for t in range(3)],   # (C, C) each
                "b": w[p + "dil.bias"][None, :, None],
                "mw": w[p + "mix.weight"][:, :, 0],
                "mb": w[p + "mix.bias"][None, :, None],
            })
        self.h1_w, self.h1_b = w["head1.weight"][:, :, 0], w["head1.bias"][None, :, None]
        self.h2_w, self.h2_b = w["head2.weight"][:, :, 0], w["head2.bias"][None, :, None]
        self.stem_ring = _Ring(n_seq, N_FEATURES, k - s)
        self.rings = [_Ring(n_seq, C, 2 * b["d"]) for b in self.blocks]

    # MARK: - Constructors

    @classmethod
    def from_torch(cls, net, n_seq: int = 2) -> "StreamingNet":
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return cls(sd, net.cfg, n_seq)

    @classmethod
    def from_export(cls, export_dir: Path, n_seq: int = 2) -> "StreamingNet":
        """Load ``weights.bin`` + ``weights.json`` written by ``nn/export.py``."""
        from pressing_tools.nn.model import NetConfig

        export_dir = Path(export_dir)
        hdr = json.loads((export_dir / "weights.json").read_text())
        raw = (export_dir / "weights.bin").read_bytes()
        weights = {}
        for layer in hdr["layers"]:
            n = int(np.prod(layer["shape"]))
            a = np.frombuffer(raw, "<f4", count=n, offset=layer["offset"])
            weights[layer["name"]] = a.reshape(layer["shape"])
        return cls(weights, NetConfig(**hdr["net_config"]), n_seq)

    # MARK: - State

    def reset(self) -> None:
        self.stem_ring.reset()
        for r in self.rings:
            r.reset()

    def state_floats(self) -> int:
        """Ring-buffer floats per sequence."""
        return self.stem_ring.buf[0].size + sum(r.buf[0].size for r in self.rings)

    # MARK: - Processing

    def process(self, feat_block: np.ndarray) -> np.ndarray:
        """feat_block: (n_seq, 3, n), n a multiple of stride -> logits (n_seq, n) float32.

        ``logit[n]`` is the prediction for target sample ``n - D`` of the stream.
        """
        cfg = self.cfg
        s, k = cfg.stride, cfg.stem_k
        x = np.asarray(feat_block, DTYPE)
        S, nf, n = x.shape
        assert S == self.n_seq and nf == N_FEATURES, x.shape
        assert n % s == 0, f"block length {n} is not a multiple of {s}"
        if n == 0:
            return np.zeros((S, 0), DTYPE)
        Fr = n // s

        # Stem: frame j sees ext[4j .. 4j+7], ext = [last 4 samples, block].
        ext = np.concatenate([self.stem_ring.history(), x], axis=-1)
        self.stem_ring.push(x)
        win = np.lib.stride_tricks.sliding_window_view(ext, k, axis=-1)[:, :, ::s]   # (S, 3, F, k)
        cols = win.transpose(0, 2, 1, 3).reshape(S, Fr, N_FEATURES * k)
        h = np.maximum(np.matmul(self.stem_w, cols.transpose(0, 2, 1)) + self.stem_b, 0)

        # Residual blocks: causal dilated conv over [last 2d frames, block].
        for blk, ring in zip(self.blocks, self.rings):
            d = blk["d"]
            hx = np.concatenate([ring.history(), h], axis=-1)        # (S, C, 2d + F)
            ring.push(h)
            w0, w1, w2 = blk["taps"]
            u = (np.matmul(w0, hx[..., 0:Fr]) + np.matmul(w1, hx[..., d:d + Fr])
                 + np.matmul(w2, hx[..., 2 * d:2 * d + Fr]) + blk["b"])
            u = np.maximum(u, 0)
            h = np.maximum(h + np.matmul(blk["mw"], u) + blk["mb"], 0)

        # Head: 1x1 convs, then sub-pixel interleave logit[4j + r] = y[r, j].
        y = np.matmul(self.h2_w, np.maximum(np.matmul(self.h1_w, h) + self.h1_b, 0)) + self.h2_b
        return np.ascontiguousarray(y.transpose(0, 2, 1)).reshape(S, Fr * s)


# MARK: - Detector (features + network)


class StreamingDetector:
    """Raw stereo audio in, logits out: ``FeatureState`` then ``StreamingNet``.

    ``process`` accepts blocks of any length. Features for a trailing partial
    frame are held back until the frame completes, so the returned logits
    cover all *complete* frames received so far: the output length is a
    multiple of ``stride`` and can differ from the input length. Feeding
    multiples of ``stride`` returns exactly as many logits as samples.
    """

    def __init__(self, net: StreamingNet) -> None:
        assert net.n_seq == 2
        self.net = net
        self.features = FeatureState()
        self.pending = np.zeros((2, N_FEATURES, 0), DTYPE)

    def reset(self) -> None:
        self.net.reset()
        self.features = FeatureState()
        self.pending = np.zeros((2, N_FEATURES, 0), DTYPE)

    def process(self, block2: np.ndarray) -> np.ndarray:
        """block2: (2, n) or (1, n) audio -> (2, m) float32 logits."""
        f = self.features.process(block2).astype(DTYPE)
        f = np.concatenate([self.pending, f], axis=-1)
        m = f.shape[-1] - f.shape[-1] % self.net.cfg.stride
        self.pending = f[..., m:]
        return self.net.process(f[..., :m])


def stream_logits(net: StreamingNet, x2: np.ndarray, block: int = 512) -> np.ndarray:
    """Convenience: run a whole (2, L) signal through a fresh detector in fixed blocks."""
    net.reset()
    det = StreamingDetector(net)
    out = [det.process(x2[:, i:i + block]) for i in range(0, x2.shape[1], block)]
    return np.concatenate(out, axis=-1) if out else np.zeros((2, 0), DTYPE)
