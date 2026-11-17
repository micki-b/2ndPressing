import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from pressing_tools import classical, memmap
from pressing_tools.dataset import load_shard, repaired
from pressing_tools.evaluate import EDGE, Counts
from pressing_tools.nn import report, synth

L = 48000
LEVELS = [-52.0, -38.0, -24.0, -12.0]   # host level of fixture window w is LEVELS[w % 4]


# MARK: - Tiny synthetic split


def _host(rng, n, level_db):
    """Low-passed noise at a given RMS level (dBFS), music-like curvature."""
    x = np.convolve(rng.standard_normal(n + 64), np.hanning(16), mode="same")[32:32 + n]
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (level_db / 20)


def _make_split(root, split="val", n_shards=2, windows=4, seed=0):
    """Shards with labelled clicks (stereo pairs), memmapped, plus manifest."""
    rng = np.random.default_rng(seed)
    (root / split).mkdir(parents=True)
    levels = LEVELS
    manifest = []
    for s in range(n_shards):
        audio = np.zeros((windows, 2, L), np.float16)
        label = np.zeros((windows, 2, L), np.uint8)
        repl = []
        for w in range(windows):
            lvl = levels[(s * windows + w) % len(levels)]
            vs = np.stack([_host(rng, L, lvl), _host(rng, L, lvl)]).astype(np.float16).astype(np.float64)
            raw = vs.copy()
            lab = np.zeros((2, L), np.uint8)
            for a in (5000, 21000, 37000):           # paired label-1 clicks: residual 20x local RMS
                amp = 20 * 10 ** (lvl / 20)
                for c, off in ((0, 0), (1, 3)):
                    n = 12
                    raw[c, a + off:a + off + n] += amp * np.exp(-np.arange(n) / 3.0) * (-1) ** c
                    lab[c, a + off:a + off + n] = 1
            lab[0, 30000:30020] = 2                  # detected-not-repaired
            raw = raw.astype(np.float16).astype(np.float64)
            audio[w], label[w] = raw, lab
            repl.append(vs[lab == 1].astype(np.float16))
        name = f"Artist{s}~Album{s}~side1.npz"
        np.savez_compressed(root / split / name, audio=audio, label=label, repl=np.concatenate(repl),
                            starts=np.arange(windows) * 1000 + s, sr_native=44100, sr=48000)
        manifest.append({"side": name.replace(".npz", ".m4a"), "artist": f"Artist {s}",
                         "album": f"Album {s}", "split": split, "sr_native": 44100,
                         "shard": f"{split}/{name}"})
    (root / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    memmap.build(root, split)
    return root


@pytest.fixture(scope="module")
def split_root(tmp_path_factory):
    return _make_split(tmp_path_factory.mktemp("data"))


CFG = synth.SynthConfig(windows=2, min_candidates=512, seed=3)


# MARK: - Bank


def test_on_the_fly_bank_rule_and_partners(split_root):
    mm = memmap.open_split(split_root, "val")
    logs = []
    bank, meta, source = synth.load_bank(split_root, "val", mm, log=logs.append)
    assert "on the fly" in source and "on the fly" in logs[0]
    assert bank.shape == (2 * 4 * 3 * 2, synth.TEMPLATE_LEN) and bank.dtype == np.float32
    assert meta.dtype == synth.BANK_META_DTYPE
    assert (meta["length"] == 12).all()
    # pre-roll: residual starts at template index 8 and is zero before it
    assert np.all(bank[:, :synth.PRE_ROLL] == 0) and np.all(bank[:, synth.PRE_ROLL] != 0)
    # stereo pairs reference each other with antisymmetric offsets
    assert (meta["partner"] >= 0).all()
    p = meta["partner"]
    assert np.array_equal(p[p], np.arange(len(meta)))
    assert np.array_equal(meta["partner_offset"][p], -meta["partner_offset"])
    assert set(np.abs(meta["partner_offset"]).tolist()) == {3}
    # level_db = 20 log10 RMS(VS over [a-1024, b+1024)): the host level of the event's window.
    # Every window holds 6 bank events (3 per channel) and window w has host level LEVELS[w % 4].
    expect = np.array(LEVELS)[(meta["event"] // 6) % len(LEVELS)]
    assert np.all(np.abs(meta["level_db"] - expect) < 1.5), meta["level_db"] - expect


def test_prep_bank_is_preferred_when_present(split_root, tmp_path):
    mm = memmap.open_split(split_root, "val")
    bank, meta = synth.build_bank(mm, np.array([0]))
    d = tmp_path / "nn" / "val"
    d.mkdir(parents=True)
    np.save(d / "bank.npy", bank)
    np.save(d / "bank_meta.npy", meta)
    logs = []
    b2, m2, source = synth.load_bank(tmp_path, "val", mm, log=logs.append)
    assert source.startswith("prep bank") and len(b2) == len(bank) == 6
    assert np.array_equal(m2, meta)


# MARK: - Injection


def test_contrast_scaling_hits_target():
    rng = np.random.default_rng(1)
    host = np.stack([_host(rng, L, -30.0), _host(rng, L, -30.0)]).astype(np.float16).astype(np.float64)
    lab = np.zeros((2, L), np.uint8)
    t = np.zeros((1, synth.TEMPLATE_LEN), np.float32)
    t[0, 8:20] = np.exp(-np.arange(12) / 3.0)
    meta = np.array([(0, 12, -1, 0, -30.0)], synth.BANK_META_DTYPE)
    cfg = synth.SynthConfig(rate=6.0)
    y, out, clicks = synth.inject_window(host, host, lab, 2, t, meta, np.random.default_rng(2), cfg)
    assert len(clicks) >= 10
    ratio = np.array([c[8] / c[7] for c in clicks])          # measured / target contrast
    assert 0.7 < np.median(ratio) < 1.3
    assert np.all(np.abs(y) <= cfg.clip)
    for ch, a, b, pos, *_, cm, lvl, lbl in clicks:
        assert -30 <= lvl < -18
        assert lbl == (1 if cm >= 8 else 2) and (out[ch, a:b] == lbl).all()
        assert EDGE <= pos - synth.PRE_ROLL and pos - synth.PRE_ROLL + synth.TEMPLATE_LEN <= L - EDGE
    for ch in (0, 1):
        pos = np.sort([c[3] for c in clicks if c[0] == ch])
        assert np.all(np.diff(pos) >= cfg.min_gap)


def test_clip_rejection_keeps_target_contrast_log_uniform():
    """On a loud host high-c* clicks often clip; retrying them (same c*) must keep the
    realised c* log-uniform, while redrawing c* on each rejection (clip_retries=1) skews it low."""
    rng = np.random.default_rng(1)
    host = np.stack([_host(rng, L, -12.0), _host(rng, L, -12.0)]).astype(np.float16).astype(np.float64)
    lab = np.zeros((2, L), np.uint8)
    t = np.zeros((1, synth.TEMPLATE_LEN), np.float32)
    t[0, 8:20] = np.exp(-np.arange(12) / 3.0)
    meta = np.array([(0, 12, -1, 0, -12.0)], synth.BANK_META_DTYPE)
    mid = (math.log(8) + math.log(100)) / 2                   # mean of log c* if log-uniform

    def realised(retries):
        cfg = synth.SynthConfig(clip_retries=retries)
        cs, st = [], {}
        for s in range(100):
            _, _, cl = synth.inject_window(host, host, lab, 3, t, meta, np.random.default_rng(s), cfg, st)
            cs += [c[7] for c in cl]
        return np.array(cs), st, cfg

    cs, st, cfg = realised(50)
    assert len(cs) >= 500 and st["clip_rejects"] > 50 and st["dropped"] == 0
    assert abs(np.log(cs).mean() - mid) < 0.08
    summ = synth.c_target_summary(cs, cfg)
    assert summ["frac_gt80"] > 0.065 and summ["frac_gt80_expected"] == pytest.approx(math.log(100 / 80) / math.log(100 / 8))
    biased, st1, _ = realised(1)
    assert st1["dropped"] > 0 and np.log(biased).mean() - mid < -0.08


def test_run_writes_shards_with_exact_truth(split_root, tmp_path):
    out = tmp_path / "synth"
    logs = []
    res = synth.run(split_root, "val", CFG, out, log=logs.append)
    assert any("on the fly" in l for l in logs)
    mm = memmap.open_split(split_root, "val")
    lines = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert {m["split"] for m in lines} == {f"val_lvl{k}" for k in range(4)}
    total_l1 = 0
    for m in lines:
        assert m["album"].startswith("Album") and m["artist"].startswith("Artist")
        sh = load_shard(out / m["shard"])
        assert set(sh) >= {"audio", "label", "repl", "starts", "sr_native", "sr"}
        assert sh["audio"].dtype == np.float16 and sh["label"].dtype == np.uint8
        assert int(sh["sr_native"]) == 44100 and int(sh["sr"]) == 48000
        hosts = np.stack([np.asarray(mm["audio"][w]) for w in sh["src_window"]])
        orig = np.stack([np.asarray(mm["label"][w]) for w in sh["src_window"]])
        s_no = int(m["source_shard"].split("Artist")[1][0])
        assert np.array_equal(sh["starts"], [mm["index"][w]["window"] * 1000 + s_no for w in sh["src_window"]])
        # original labels become don't-care; repaired() rebuilds the host at every label-1 sample
        assert (sh["label"][orig != 0] == 2).all()
        rep = repaired(sh)
        l1 = sh["label"] == 1
        assert np.array_equal(rep[l1], hosts[l1].astype(np.float32))
        cl = sh["clicks"]
        total_l1 += int((cl["label"] == 1).sum())
        for r in cl:
            x = sh["audio"][r["win"], r["ch"]].astype(np.float64)
            assert classical.contrast(x, r["a"], r["b"], 150) == pytest.approx(r["contrast"], rel=1e-6)
            assert (r["contrast"] >= 8) == (r["label"] == 1)
            assert (sh["label"][r["win"], r["ch"], r["a"]:r["b"]] == r["label"]).all()
            near = orig[r["win"]].any(0)[max(0, r["a"] - 256):r["b"] + 256]
            assert not near.any()
        assert np.all(cl["partner_of"][cl["partner_of"] >= 0] < len(cl))
        # the injected audio differs from the host only near injected clicks
        diff = sh["audio"] != hosts
        allowed = np.zeros_like(diff)
        for r in cl:
            allowed[r["win"], r["ch"], r["pos"] - 8:r["pos"] + 120] = True
        assert not (diff & ~allowed).any()
        assert np.abs(sh["audio"][allowed].astype(np.float32)).max() <= 0.999
    assert total_l1 > 0
    assert all(r["stats"]["clicks"] > 0 for r in res)
    for r in res:                                  # injection diagnostics land in synth.json
        js = json.loads((out / f"val_lvl{r['stats']['bin']}" / "synth.json").read_text())["stats"]
        assert {"clip_rejects", "dropped"} <= set(js) and js["c_target"]["n"] > 0
        assert js["c_target"]["q50_expected"] == pytest.approx(math.sqrt(8 * 100))
    # deterministic, and a rerun replaces this split's manifest entries
    synth.run(split_root, "val", CFG, out, log=lambda *_: None)
    again = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert again == lines
    a = load_shard(out / lines[0]["shard"])
    assert np.array_equal(a["audio"], load_shard(out / lines[0]["shard"])["audio"])


def test_cli_entry(split_root, tmp_path, capsys):
    assert synth.main(["--data", str(split_root), "--split", "val", "--windows", "1",
                       "--out", str(tmp_path / "s")]) == 0
    assert "val_lvl3" in capsys.readouterr().out


# MARK: - Report: curves


def _P(t, fa, r, r2=0.0):
    return report.Point(t, fa, r, r2)


def test_pareto_and_interpolation():
    pts = [_P(8, 10, 0.5), _P(6, 100, 0.8), _P(7, 50, 0.4), _P(4, 1000, 0.8), _P(10, 1, 0.2)]
    env = report.pareto(pts)
    assert [p.threshold for p in env] == [10, 8, 6]
    r = report.recall_at_fa(env, math.sqrt(10 * 100))       # halfway in log FA
    assert r["recall"] == pytest.approx(0.65)
    assert report.recall_at_fa(env, 0.1)["recall"] == 0.2     # target never reached -> lowest-FA point
    assert report.recall_at_fa(env, 500)["recall"] == 0.8
    assert report.recall_at_fa(env, 10)["recall"] == 0.5
    f = report.fa_at_recall(env, 0.65)["fa_per_min"]
    assert f == pytest.approx(math.sqrt(1000))
    assert report.fa_at_recall(env, 0.9)["fa_per_min"] is None
    assert report.fa_at_recall(env, 0.1)["fa_per_min"] == 1


def test_zero_fa_points_use_floor():
    env = report.pareto([_P(10, 0.0, 0.3), _P(5, 10.0, 0.6)])
    assert report.recall_at_fa(env, 50)["recall"] == 0.6
    assert 0.3 < report.recall_at_fa(env, 1.0)["recall"] < 0.6


# MARK: - Report: bootstrap and CLI


THRS = [4.0, 8.0, 16.0]


def _entries(n_albums=8, extra_found=0, seed=0):
    """Per-shard Counts at three thresholds; extra_found adds found l1 at every threshold."""
    rng = np.random.default_rng(seed)
    out = []
    for a in range(n_albums):
        for s in range(2):
            counts = {}
            for j, t in enumerate(THRS):
                ev = 100
                found = int(rng.integers(60, 80)) - 15 * j
                counts[str(t)] = asdict(Counts(seconds=600.0, l1_events=ev, l1_found=min(ev, found + extra_found),
                                               l2_events=20, l2_found=10 - j, pred_false=[800, 450, 100][j]))
            out.append({"shard": f"a{a}_s{s}.npz", "album": f"artist|album{a}", "counts": counts})
    return out


def test_paired_bootstrap():
    base = _entries()
    same = report.paired_bootstrap(base, base, 50.0, n_boot=200, seed=0)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0] and not same["excludes_zero"]
    better = report.paired_bootstrap(_entries(extra_found=5), base, 50.0, n_boot=200, seed=0)
    assert better["diff"] == pytest.approx(0.05) and better["excludes_zero"]
    assert better["ci95"][0] > 0 and better["albums"] == 8 and better["shards"] == 16
    again = report.paired_bootstrap(_entries(extra_found=5), base, 50.0, n_boot=200, seed=0)
    assert again["ci95"] == better["ci95"]
    noisy = report.paired_bootstrap(_entries(seed=1), base, 50.0, n_boot=200, seed=0)
    assert noisy["ci95"][0] < noisy["diff"] < noisy["ci95"][1]


def _write_eval(path, entries):
    det = {}
    for t in THRS:
        tot = Counts()
        for e in entries:
            tot.add(Counts(**e["counts"][str(t)]))
        det[str(t)] = tot.summary()
    path.write_text(json.dumps({"detection": det, "repair": {}, "config": {"method": "x"}}))
    report.shards_path(path).write_text(json.dumps(entries))


def test_report_cli(tmp_path, capsys):
    base, neural = tmp_path / "classical_val.json", tmp_path / "neural_val.json"
    _write_eval(base, _entries())
    _write_eval(neural, _entries(extra_found=5))
    out = tmp_path / "rep.json"
    assert report.main([str(base), str(neural), "--boot", "100", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "recall @ 50 FA/min" in text and "neural_val: +5.0 points" in text
    rep = json.loads(out.read_text())
    assert [m["name"] for m in rep["methods"]] == ["classical_val", "neural_val"]
    b = rep["bootstrap"][0]
    assert b["method"] == "neural_val" and b["baseline"] == "classical_val" and b["excludes_zero"]
    # aggregate-based point estimate agrees with the per-shard re-aggregation
    m0 = rep["methods"][0]["recall_at_fa"]["50.0"]["recall"]
    assert m0 == pytest.approx(b["recall_b"])
