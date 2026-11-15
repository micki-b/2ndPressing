"""Crop sampler + loss on a tiny hand-built split (no real data needed)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pressing_tools.nn.data import S0, S1, T_CROP, CropSampler, SamplerConfig, dilate  # noqa: E402
from pressing_tools.nn.loss import click_loss  # noqa: E402

EV_DT = [("win", "<i4"), ("ch", "i1"), ("a", "<i4"), ("b", "<i4"), ("side", "<i4"),
         ("level_db", "<f4"), ("w_evt", "<f4"), ("partner", "<i4")]
HN_DT = [("win", "<i4"), ("ch", "i1"), ("a", "<i4"), ("b", "<i4"), ("peak", "<f4"),
         ("contrast", "<f4"), ("side", "<i4")]
BM_DT = [("event", "<i4"), ("length", "<i2"), ("partner", "<i4"), ("partner_offset", "<i2"),
         ("level_db", "<f4")]
L = 96000


def _split(tmp_path, n_win=3):
    rng = np.random.default_rng(0)
    mm = tmp_path / "mm" / "train"; mm.mkdir(parents=True)
    nd = tmp_path / "nn" / "train"; nd.mkdir(parents=True)
    t = np.arange(L) / 48000
    audio = np.stack([np.stack([0.1 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 330 * t)])
                      + 0.003 * rng.standard_normal((2, L)) for _ in range(n_win)]).astype(np.float16)
    label = np.zeros((n_win, 2, L), np.uint8)
    events, hard, susp = [], [], []
    for w in range(n_win):
        for k, a in enumerate(range(10000, 86000, 6000)):
            label[w, 0, a:a + 20] = 1
            audio[w, 0, a:a + 20] += np.float16(0.2)
            events.append((w, 0, a, a + 20, w, -40.0, 0.8, -2147483648))
            if k % 3 == 0:
                label[w, 1, a + 3000:a + 3030] = 2
        hard.append((w, 1, 20000, 20040, 9.0, 6.0, w))
        susp.append((w, 1, 45000, 45020, 12.0, 30.0, w))
    np.save(mm / "audio.npy", audio); np.save(mm / "label.npy", label)
    np.save(mm / "repl.npy", np.zeros(int((label == 1).sum()), np.float16))
    np.save(mm / "repl_off.npy", np.zeros(n_win + 1, np.int64))
    (mm / "index.json").write_text(json.dumps([{"shard": f"s{w}.npz", "window": 0, "album": "a|b"}
                                               for w in range(n_win)]))
    ev = np.array(events, EV_DT)
    np.save(nd / "events.npy", ev)
    np.save(nd / "hardneg.npy", np.array(hard, HN_DT))
    np.save(nd / "suspected.npy", np.array(susp, HN_DT))
    k = np.arange(128) - 8
    tmpl = np.where((k >= 0) & (k < 20), np.exp(-np.clip(k, 0, None) / 5.0), 0.0).astype(np.float32)
    np.save(nd / "bank.npy", np.stack([tmpl, -tmpl]))
    np.save(nd / "bank_meta.npy", np.array([(0, 20, 1, 2, -40.0), (1, 20, 0, -2, -40.0)], BM_DT))
    (nd / "sides.json").write_text(json.dumps([f"s{w}.npz" for w in range(n_win)]))
    return tmp_path


def test_maps_rules_and_batch_shapes(tmp_path):
    root = _split(tmp_path)
    s = CropSampler(root, "train", SamplerConfig(batch=16, transplant_p=1.0), seed=1)
    s._load()
    rng = np.random.default_rng(3)
    saw_transplant = False
    for _ in range(40):
        x, tgt, wt, neg, events, kind = s.crop(rng)
        assert x.shape == (2, T_CROP) and tgt.shape == (2, T_CROP)
        assert not wt[:, :S0].any() and not wt[:, S1:].any()           # outside scored region
        sc = tgt[:, S0:S1] == 1
        assert (wt[:, S0:S1][sc] > 0).all()                            # scored positives carry weight
        assert not (neg & (tgt == 1)).any()
        assert not (neg & dilate(tgt == 1, 16)).any()                   # guard band around positives
        for ch, a, b, w in events:
            assert S0 <= a < b <= S1 and (tgt[ch, a:b] == 1).all()
        saw_transplant |= kind != "event" and len(events) > 0
    b = s.batch(rng)
    assert b["audio"].shape == (16, 2, T_CROP) and b["target"].shape == (32, T_CROP)
    assert b["ev_idx"].shape[1] == 3 and (b["ev_idx"][:, 0] < 32).all()
    assert saw_transplant


def test_suspected_and_label2_are_ignored(tmp_path):
    root = _split(tmp_path)
    s = CropSampler(root, "train", SamplerConfig(transplant_p=0.0, polarity_p=0.0, mono_p=0.0), seed=0)
    s._load()
    st = 38000                       # suspected at 45000 and a label-2 run at 49000 both land in [S0, S1)
    lab = np.asarray(s.label[0, :, st:st + T_CROP]).copy()
    x = np.asarray(s.audio[0, :, st:st + T_CROP], np.float32).copy()
    tgt, wt, neg, ev = s._maps(x, lab, 0, st, [], False)
    a = 45000 - st                                       # suspected region in ch1
    assert (wt[1, a - 16:a + 20 + 16] == 0).all()
    l2 = np.flatnonzero(lab[1] == 2)
    assert len(l2) and (wt[1, l2] == 0).all() and S0 <= l2[0] < S1
    l1 = np.flatnonzero(lab[0] == 1)
    assert (tgt[0, l1] == 1).all() and np.allclose(wt[0, l1[(l1 >= S0) & (l1 < S1)]], 0.8)


def test_loss_runs_and_prefers_correct_logits(tmp_path):
    root = _split(tmp_path)
    s = CropSampler(root, "train", SamplerConfig(batch=4, transplant_p=0.0), seed=0)
    s._load()
    b = s.batch(np.random.default_rng(0))
    D = 512
    tgt = b["target"].float()
    good = torch.full((8, T_CROP), -6.0)
    good[:, D:] = torch.where(tgt[:, :T_CROP - D] > 0, 6.0, -6.0)
    bad = -good
    lg, parts_g = click_loss(good, b, D)
    lb, _ = click_loss(bad, b, D)
    assert lg < lb and parts_g["pos"] < 0.01 and parts_g["neg"] < 0.01


def test_templates_are_masked_to_their_own_run(tmp_path):
    root = _split(tmp_path)
    nd = root / "nn" / "train"
    bank = np.load(nd / "bank.npy")
    bank[0, 60:70] = 0.5                 # a neighbouring click inside the 128-sample span
    np.save(nd / "bank.npy", bank)
    s = CropSampler(root, "train")
    s._load()
    L0 = int(s.bank_meta[0]["length"])
    assert not s.bank[0, 8 + L0:].any() and not s.bank[0, :8].any()
    assert s.bank[0, 8:8 + L0].any()
