"""Export a trained ClickNet for ORT and the Rust runtime (spec section 12).

Writes to ``--out``:

    model.onnx     opset 17, input ``feat`` (N, 3, T), output ``logit`` (N, T),
                   N and T dynamic (T a multiple of stride). After a small
                   cleanup pass (constant folding, Pad fused into Conv pads)
                   the graph has only Conv, Relu, Add, Transpose and Flatten.
                   metadata_props: D, stride, receptive_field, net_config.
    weights.bin    raw little-endian float32, layers back to back
    weights.json   layer names / shapes / byte offsets, NetConfig, feature
                   constants, D, post-processing
    golden.npz     val windows for the Rust parity test: raw audio, features,
                   logits, plus aligned scores and masks at the chosen threshold,
                   and the effective post dict as a JSON string (``post``)

Usage::

    uv run python -m pressing_tools.nn.export --ckpt out/nn/<run>/best.pt --out out/nn/<run>/export

Checkpoint format: ``torch.save({'model': state_dict, 'net_config': NetConfig.to_dict(),
'post': {...}})``. ``post`` is optional (NeuralConfig field names, section 8); the
values chosen on val (section 10 step 4) can be given with ``--post threshold=2.5
--post min_contrast=8`` or ``--post-json chosen.json``, merged over the checkpoint's.
weights.json and golden.npz record the effective (fully merged) post dict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pressing_tools.nn import features as feat_mod
from pressing_tools.nn.features import features_np
from pressing_tools.nn.model import ClickNet, NetConfig, receptive_field

DEFAULT_DATA = Path(__file__).resolve().parents[4] / "data" / "v1"
TOLERANCE = 1e-4
WEIGHTS_FORMAT = "clicknet-v1-weights"

# Section 8 defaults for the post-processing fields used to build golden masks.
POST_DEFAULTS = {"threshold": 3.0, "ext_drop": 2.0, "merge": 8, "pre": 2, "post": 8,
                 "max_region": 480, "min_contrast": 0.0, "contrast_context": 150}


# MARK: - Loading


def load_checkpoint(path: Path) -> tuple[ClickNet, dict]:
    """-> (eval-mode ClickNet on CPU, post dict).

    Loaded with ``weights_only=False``: checkpoints are our own artefacts, and
    nn/train.py stores extras (``curve``) holding numpy scalars that the
    weights-only unpickler rejects."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ClickNet(NetConfig(**ck["net_config"]))
    net.load_state_dict(ck["model"])
    return net.eval(), dict(ck.get("post") or {})


def random_model(cfg: NetConfig | None = None, seed: int = 0, gain: float = 1.0) -> ClickNet:
    """Randomly initialised ClickNet with non-zero residual branches (for tests).

    Every weight is N(0, gain^2 / fan_in) and every bias N(0, 0.1^2), so the
    activations stay O(1) through the residual stack.
    """
    g = torch.Generator().manual_seed(seed)
    net = ClickNet(cfg or NetConfig()).eval()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("weight"):
                fan_in = p.shape[1] * p.shape[2]
                p.copy_(torch.randn(p.shape, generator=g) * gain / fan_in ** 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return net


# MARK: - ONNX


class _ExportNet(nn.Module):
    """ClickNet.forward with the final interleave written as transpose + flatten,
    which traces without shape arithmetic. Checked against ClickNet.forward via ORT."""

    def __init__(self, net: ClickNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        n, cfg = self.net, self.net.cfg
        h = F.relu(n.stem(F.pad(feat, (cfg.stem_k - cfg.stride, 0))))
        for b in n.blocks:
            h = b(h)
        y = n.head2(F.relu(n.head1(h)))                  # (N, stride, F)
        return y.transpose(1, 2).flatten(1)


def _simplify(model):
    """Fold constant subgraphs, fuse zero Pad into the following Conv's pads, and
    give every layer its own initializer (the exporter de-duplicates identical
    tensors, e.g. zero-initialised residual branches, through Identity nodes)."""
    import onnx
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    const: dict[str, np.ndarray] = {}
    nodes = []
    for node in g.node:
        ins = [i for i in node.input if i]
        if node.op_type == "Identity" and ins[0] in inits:
            t = onnx.TensorProto()
            t.CopyFrom(inits[ins[0]])
            t.name = node.output[0]
            g.initializer.append(t)
            inits[t.name] = t
            continue
        if all(i in const for i in ins):           # includes Constant (no inputs)
            outs = ReferenceEvaluator(node).run(None, {i: const[i] for i in ins})
            const.update(zip(node.output, (np.asarray(o) for o in outs)))
            continue
        nodes.append(node)

    def value(name):
        if name in const:
            return const[name]
        return numpy_helper.to_array(inits[name]) if name in inits else None

    consumers: dict[str, list] = {}
    for node in nodes:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    drop = set()
    for node in nodes:
        if node.op_type != "Pad":
            continue
        mode = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "mode"), b"constant")
        pads = value(node.input[1])
        fill = value(node.input[2]) if len(node.input) > 2 and node.input[2] else np.zeros(())
        users = consumers.get(node.output[0], [])
        if mode != b"constant" or pads is None or fill is None or np.any(fill != 0) \
                or len(users) != 1 or users[0].op_type != "Conv" or users[0].input[0] != node.output[0]:
            continue
        r = len(pads) // 2
        if np.any(pads[:2]) or np.any(pads[r:r + 2]):    # only spatial padding can fuse
            continue
        conv = users[0]
        old = next((a for a in conv.attribute if a.name == "pads"), None)
        cur = list(helper.get_attribute_value(old)) if old is not None else [0] * (2 * (r - 2))
        new = [int(a) + int(b) for a, b in zip(cur, list(pads[2:r]) + list(pads[r + 2:]))]
        if old is not None:
            conv.attribute.remove(old)
        conv.attribute.append(helper.make_attribute("pads", new))
        conv.input[0] = node.input[0]
        drop.add(id(node))
    nodes = [n for n in nodes if id(n) not in drop]

    used = {i for n in nodes for i in n.input} | {o.name for o in g.output}
    for name in sorted(used):
        if name in const:
            g.initializer.append(numpy_helper.from_array(const[name], name))
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def export_onnx(net: ClickNet, path: Path) -> list[str]:
    """Write model.onnx; returns the sorted list of op types in the graph."""
    import warnings

    import onnx

    net = net.eval()
    T = 64 * net.cfg.stride
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(_ExportNet(net), (torch.zeros(2, 3, T),), str(path),
                          opset_version=17, dynamo=False, do_constant_folding=True,
                          input_names=["feat"], output_names=["logit"],
                          dynamic_axes={"feat": {0: "N", 2: "T"}, "logit": {0: "N", 1: "T"}})
    model = _simplify(onnx.load(str(path)))
    cfg = net.cfg
    onnx.helper.set_model_props(model, {"D": str(cfg.D), "stride": str(cfg.stride),
                                        "receptive_field": str(receptive_field(cfg)),
                                        "net_config": json.dumps(cfg.to_dict())})
    onnx.save(model, str(path))
    return sorted({n.op_type for n in model.graph.node})


def ort_session(path: Path, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(sess, feat: np.ndarray) -> np.ndarray:
    """feat: (N, 3, T) -> (N, T) float32."""
    return sess.run(["logit"], {"feat": np.ascontiguousarray(feat, np.float32)})[0]


def torch_logits(net: ClickNet, feat: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.from_numpy(np.ascontiguousarray(feat, np.float32))).numpy()


# MARK: - Raw weights


def write_weights(net: ClickNet, out: Path, post: dict | None = None) -> dict:
    """weights.bin (raw f32 LE, layers back to back) + weights.json header."""
    cfg = net.cfg
    layers, chunks, off = [], [], 0
    for name, t in net.state_dict().items():
        a = np.ascontiguousarray(t.detach().cpu().numpy(), "<f4")
        layers.append({"name": name, "shape": list(a.shape), "offset": off, "nbytes": a.nbytes})
        chunks.append(a.tobytes())
        off += a.nbytes
    (out / "weights.bin").write_bytes(b"".join(chunks))
    hdr = {
        "format": WEIGHTS_FORMAT,
        "dtype": "f32le",
        "total_bytes": off,
        "layers": layers,
        "net_config": cfg.to_dict(),
        "D": cfg.D,
        "receptive_field": receptive_field(cfg),
        "features": {"EPS": feat_mod.EPS, "FLOOR": feat_mod.FLOOR, "NW": feat_mod.NW,
                     "ASINH_SCALE": feat_mod.ASINH_SCALE, "N_FEATURES": feat_mod.N_FEATURES},
        "post": effective_post(post),
        "layout": {
            "conv_weight": "[out, in, k] row-major; bias [out]",
            "stem": f"Conv1d(3->C, k={cfg.stem_k}, stride={cfg.stride}), left pad "
                    f"{cfg.stem_k - cfg.stride}, ReLU",
            "blocks": "y = ReLU(x + mix(ReLU(dil(x)))); dil k=3 dilation d=2^i frames, left pad 2d",
            "head": "head2(ReLU(head1(h))) -> (stride, F); logit[stride*j + r] = y[r, j]",
            "input": "per channel [g1_self, g2_self, g1_other]; logit[n] predicts target n - D",
        },
    }
    (out / "weights.json").write_text(json.dumps(hdr, indent=1))
    return hdr


# MARK: - Golden vectors and check


def val_windows(data: Path, n: int, pool: int = 64) -> tuple[np.ndarray, np.ndarray] | None:
    """Up to n val windows as float32 (n, 2, L) plus their indices, or None if no data.

    Deterministic: from ``pool`` evenly spaced windows, the n with the most
    label-1 samples (so golden masks are non-trivial), in index order."""
    try:
        from pressing_tools.memmap import open_split

        mm = open_split(data, "val")
    except (FileNotFoundError, OSError):
        return None
    N = mm["audio"].shape[0]
    if N == 0:
        return None
    cand = np.unique(np.linspace(0, N - 1, min(pool, N)).astype(int))
    clicks = np.array([(mm["label"][i] == 1).sum() for i in cand])
    idx = np.sort(cand[np.argsort(-clicks, kind="stable")[:n]])
    return np.asarray(mm["audio"][idx], np.float32), idx


def _cast(key: str, v):
    """Cast to the type of POST_DEFAULTS[key]; int fields reject non-integral values."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"post {key}: expected a number, got {v!r}")
    if isinstance(POST_DEFAULTS[key], int):
        if float(v) != int(v):
            raise ValueError(f"post {key}: expected an integer, got {v!r}")
        return int(v)
    return float(v)


def effective_post(post: dict | None) -> dict:
    """``post`` merged over POST_DEFAULTS, each value cast to its default's type.
    Exactly the parameters ``mask_config`` uses; unknown keys are kept as given."""
    post = dict(post or {})
    out = {k: _cast(k, post.get(k, d)) for k, d in POST_DEFAULTS.items()}
    out.update({k: v for k, v in post.items() if k not in POST_DEFAULTS})
    return out


def parse_post(items: list[str]) -> dict:
    """``["threshold=2.5", "merge=8"]`` -> dict; values parsed as JSON numbers.
    Keys must be post-processing fields (POST_DEFAULTS)."""
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        k = k.strip()
        if not sep or k not in POST_DEFAULTS:
            raise ValueError(f"--post expects key=value with key in {sorted(POST_DEFAULTS)}: {it!r}")
        try:
            out[k] = _cast(k, json.loads(v))
        except ValueError as e:
            raise ValueError(f"--post {k}: bad value {v!r} ({e})") from e
    return out


def mask_config(post: dict):
    from pressing_tools.classical import ClassicalConfig

    p = effective_post(post)
    thr = float(p["threshold"])
    return ClassicalConfig(threshold=thr, extend_threshold=thr - float(p["ext_drop"]),
                           merge=int(p["merge"]), pre=int(p["pre"]), post=int(p["post"]),
                           max_region=int(p["max_region"]), min_contrast=float(p["min_contrast"]),
                           contrast_context=int(p["contrast_context"])), thr


def aligned_scores(net: ClickNet, feat: np.ndarray, L: int) -> np.ndarray:
    """Section 9 scoring: features padded right with D zeros to a multiple of
    stride, then ``logits[:, D:D+L]`` (score for target sample t)."""
    D, s = net.cfg.D, net.cfg.stride
    T = -(-(L + D) // s) * s
    f = np.zeros(feat.shape[:-1] + (T,), np.float32)
    f[..., :L] = feat
    return torch_logits(net, f)[:, D:D + L]


def write_golden(net: ClickNet, audio: np.ndarray, post: dict, path: Path,
                 index: np.ndarray | None = None) -> None:
    from pressing_tools.classical import regions_from_scores

    cfg, thr = mask_config(post)
    B, _, L = audio.shape
    feats = np.stack([features_np(a) for a in audio]).astype(np.float32)        # (B, 2, 3, L)
    logits = torch_logits(net, feats.reshape(2 * B, 3, L)).reshape(B, 2, L)
    scores = np.stack([aligned_scores(net, f, L) for f in feats])               # (B, 2, L)
    masks = np.stack([[regions_from_scores(scores[b, c].astype(np.float64), cfg, thr,
                                           x=audio[b, c].astype(np.float64))
                       for c in range(2)] for b in range(B)])
    np.savez_compressed(path, audio=audio.astype(np.float32), features=feats, logits=logits,
                        scores=scores, masks=masks, threshold=np.float32(thr),
                        post=np.asarray(json.dumps(effective_post(post), sort_keys=True)),
                        D=np.int64(net.cfg.D),
                        window_index=np.asarray(index if index is not None else np.arange(B)))


def check_ort(net: ClickNet, sess, audio: np.ndarray) -> float:
    """max |ORT - torch| logit over (B, 2, L) audio windows."""
    worst = 0.0
    for a in audio:
        f = features_np(a).astype(np.float32)
        L = f.shape[-1] - f.shape[-1] % net.cfg.stride
        f = f[..., :L]
        worst = max(worst, float(np.abs(ort_logits(sess, f) - torch_logits(net, f)).max()))
    return worst


# MARK: - Export


def export(net: ClickNet, out: Path, post: dict | None = None, data: Path | None = DEFAULT_DATA,
           n_golden: int = 8, n_check: int = 20, seed: int = 0, verbose: bool = True) -> dict:
    """Write model.onnx, weights.bin/json and golden.npz; returns a summary dict.

    Uses val windows from ``data`` when present, otherwise seeded noise with
    injected impulses. Raises if ORT and torch disagree by >= 1e-4.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    post = dict(post or {})
    net = net.eval()
    ops = export_onnx(net, out / "model.onnx")
    write_weights(net, out, post)

    got = val_windows(data, max(n_golden, n_check)) if data is not None else None
    source = "val"
    if got is None:
        source = "synthetic"
        audio = synthetic_audio(max(n_golden, n_check), seed=seed)
        index = np.arange(len(audio))
    else:
        audio, index = got
    sess = ort_session(out / "model.onnx")
    dmax = check_ort(net, sess, audio[:n_check])
    write_golden(net, audio[:n_golden], post, out / "golden.npz", index[:n_golden])
    summary = {"ops": ops, "ort_vs_torch_max_abs": dmax, "check_windows": int(min(n_check, len(audio))),
               "golden_windows": int(min(n_golden, len(audio))), "source": source, "out": str(out)}
    if verbose:
        print(f"ops: {', '.join(ops)}")
        print(f"ORT vs torch over {summary['check_windows']} {source} windows: "
              f"max |dlogit| = {dmax:.3e} (tolerance {TOLERANCE:g})")
    if not dmax < TOLERANCE:
        raise RuntimeError(f"ORT vs torch max |dlogit| {dmax:.3e} >= {TOLERANCE:g}")
    return summary


def export_random(out: Path, cfg: NetConfig | None = None, seed: int = 0, post: dict | None = None,
                  data: Path | None = None, **kw) -> tuple[ClickNet, dict]:
    """Export a randomly initialised model (tests). Synthetic audio unless ``data`` is given."""
    net = random_model(cfg, seed)
    return net, export(net, out, post, data=data, seed=seed, **kw)


def synthetic_audio(n: int, L: int = 96000, seed: int = 0) -> np.ndarray:
    """(n, 2, L) float32 pink-ish noise with injected impulses of varied size."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, 2, L)), axis=-1)
    x -= np.mean(x, axis=-1, keepdims=True)
    x = x / (np.abs(x).max(axis=-1, keepdims=True) + 1e-9) * 0.2
    x += 0.003 * rng.standard_normal(x.shape)
    for w in range(n):
        for _ in range(rng.integers(5, 30)):
            c, t = rng.integers(0, 2), rng.integers(0, L - 64)
            ln = rng.integers(1, 40)
            x[w, c, t:t + ln] += rng.choice([-1, 1]) * 10 ** rng.uniform(-2.5, -0.5) \
                * np.exp(-np.arange(ln) / max(1, ln / 4))
    return x.astype(np.float32)


# MARK: - CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-nn-export", description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True, help=".pt checkpoint")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset root (uses mm/val)")
    ap.add_argument("--golden", type=int, default=8, help="golden val windows")
    ap.add_argument("--check", type=int, default=20, help="val windows for the ORT check")
    ap.add_argument("--post", action="append", default=[], metavar="KEY=VALUE",
                    help="post-processing chosen on val, merged over the checkpoint's "
                         f"(repeatable; keys: {', '.join(POST_DEFAULTS)})")
    ap.add_argument("--post-json", type=Path, help="JSON file of post-processing fields, "
                    "merged over the checkpoint's and under --post")
    a = ap.parse_args(argv)
    net, post = load_checkpoint(a.ckpt)
    try:
        if a.post_json is not None:
            extra = json.loads(a.post_json.read_text())
            if not isinstance(extra, dict):
                raise ValueError(f"--post-json must hold a JSON object: {a.post_json}")
            post.update(parse_post([f"{k}={json.dumps(v)}" for k, v in extra.items()]))
        post.update(parse_post(a.post))
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        export(net, a.out, post, data=a.data, n_golden=a.golden, n_check=a.check)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {a.out}/model.onnx, weights.bin, weights.json, golden.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
