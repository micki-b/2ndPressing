import json
import math

import numpy as np
import pytest

from pressing_tools.nn import prep
from pressing_tools.nn.prep import NO_PARTNER, PrepConfig, prep_split

L = 96000


# MARK: - Synthetic split


def _click(n=12, amp=0.3):
    return amp * np.exp(-np.arange(n) / 3.0)


def _make_split(root, split="val"):
    """3 windows, 2 sides. Returns (raw, vs, label) as written (float16-exact)."""
    rng = np.random.default_rng(0)
    t = np.arange(L) / 48000
    W = 3
    vs = np.stack([np.stack([0.05 * np.sin(2 * np.pi * (200 + 30 * w + 7 * c) * t)
                             + 1e-3 * rng.standard_normal(L) for c in range(2)]) for w in range(W)])
    vs = vs.astype(np.float16).astype(np.float64)
    raw = vs.copy()
    lab = np.zeros((W, 2, L), np.uint8)

    # window 0: a stereo click pair (ch1 starts 3 later), both repaired
    raw[0, 0, 10000:10012] += _click()
    lab[0, 0, 10000:10020] = 1
    raw[0, 1, 10003:10013] += _click(10, -0.2)
    lab[0, 1, 10003:10015] = 1
    # window 0 ch0: a long repair (> 96 samples) -> event, not bank
    raw[0, 0, 40000:40150] += 0.05 * np.hanning(150)
    lab[0, 0, 40000:40150] = 1
    # window 0 ch1: a "repair" that changes nothing -> w_evt floor, not bank
    lab[0, 1, 60000:60010] = 1
    # window 0 ch0: run whose ch1 neighbour starts 20 later -> no partner
    raw[0, 0, 70000:70012] += _click()
    lab[0, 0, 70000:70012] = 1
    raw[0, 1, 70020:70030] += _click(10)
    lab[0, 1, 70020:70030] = 1

    # window 1: unlabelled sharp click -> suspected
    raw[1, 0, 30000:30003] += np.array([0.3, -0.2, 0.1])
    # window 1: unlabelled sharp click ~36-70 samples after a label-2 run -> hardneg
    lab[1, 0, 49880:49930] = 2
    raw[1, 0, 50000:50003] += np.array([0.3, -0.2, 0.1])
    # window 1: unlabelled click right next to a label -> excluded (within TOL)
    lab[1, 1, 20000:20004] = 2
    raw[1, 1, 20002:20005] += np.array([0.3, -0.2, 0.1])
    # window 1: unlabelled click in the edge zone -> excluded
    raw[1, 1, 1000:1003] += np.array([0.3, -0.2, 0.1])
    # window 2: clean

    raw = raw.astype(np.float16).astype(np.float64)
    vs = np.where(lab == 1, vs, raw)  # outside label-1, VS == raw
    d = root / "mm" / split
    d.mkdir(parents=True)
    np.save(d / "audio.npy", raw.astype(np.float16))
    np.save(d / "label.npy", lab)
    reps = [vs[w][lab[w] == 1] for w in range(W)]
    np.save(d / "repl.npy", np.concatenate(reps).astype(np.float16))
    np.save(d / "repl_off.npy", np.r_[0, np.cumsum([len(r) for r in reps])].astype(np.int64))
    shards = ["b_side.npz", "a_side.npz", "a_side.npz"]
    (d / "index.json").write_text(json.dumps(
        [{"shard": s, "window": i, "album": "x|y"} for i, s in enumerate(shards)]))
    return raw, vs, lab


@pytest.fixture(scope="module")
def prepped(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    raw, vs, lab = _make_split(root)
    summary = prep_split(root, "val", jobs=1, chunk=2)
    out = root / "nn" / "val"
    tables = {k: np.load(out / f"{k}.npy") for k in ["events", "hardneg", "suspected", "bank", "bank_meta"]}
    return root, out, summary, tables, raw, vs, lab


def _find(events, win, ch, a):
    i = np.flatnonzero((events["win"] == win) & (events["ch"] == ch) & (events["a"] == a))
    assert len(i) == 1
    return int(i[0])


# MARK: - Tests


def test_formats_and_sides(prepped):
    root, out, summary, t, *_ = prepped
    assert t["events"].dtype == prep.EVENT_DTYPE
    assert t["hardneg"].dtype == prep.REGION_DTYPE and t["suspected"].dtype == prep.REGION_DTYPE
    assert t["bank_meta"].dtype == prep.BANK_META_DTYPE
    assert t["bank"].dtype == np.float32 and t["bank"].shape[1] == 128
    assert len(t["bank"]) == len(t["bank_meta"])
    assert json.loads((out / "sides.json").read_text()) == ["a_side.npz", "b_side.npz"]
    ev = t["events"]
    assert set(ev[ev["win"] == 0]["side"]) == {1}
    assert json.loads((out / "summary.json").read_text())["events"]["count"] == len(ev)


def test_events_partner_level_weight(prepped):
    _, _, _, t, raw, vs, lab = prepped
    ev = t["events"]
    assert len(ev) == 6
    i0, i1 = _find(ev, 0, 0, 10000), _find(ev, 0, 1, 10003)
    assert (ev[i0]["b"], ev[i1]["b"]) == (10020, 10015)
    assert ev[i0]["partner"] == 3 and ev[i1]["partner"] == -3
    assert ev[_find(ev, 0, 0, 70000)]["partner"] == NO_PARTNER
    assert ev[_find(ev, 0, 0, 40000)]["partner"] == NO_PARTNER
    # level_db over [a-1024, b+1024) of VS
    e = ev[i0]
    seg = vs[0, 0, 10000 - 1024:10020 + 1024]
    assert e["level_db"] == pytest.approx(20 * math.log10(np.sqrt(np.mean(seg ** 2))), abs=1e-3)
    # w_evt from curvature contrast; the no-op repair sits at the 0.3 floor
    con = prep.event_contrast(raw[0, 0], 10000, 10020)
    assert e["w_evt"] == pytest.approx(prep.w_evt_from_contrast(con), abs=1e-6)
    assert ev[_find(ev, 0, 1, 60000)]["w_evt"] == pytest.approx(0.3)


def test_bank(prepped):
    _, _, _, t, raw, vs, lab = prepped
    ev, bank, meta = t["events"], t["bank"], t["bank_meta"]
    banked = {int(m) for m in meta["event"]}
    assert _find(ev, 0, 0, 40000) not in banked      # too long
    assert _find(ev, 0, 1, 60000) not in banked      # residual at zero
    i0, i1 = _find(ev, 0, 0, 10000), _find(ev, 0, 1, 10003)
    assert {i0, i1, _find(ev, 0, 0, 70000), _find(ev, 0, 1, 70020)} == banked
    m0 = int(np.flatnonzero(meta["event"] == i0)[0])
    m1 = int(np.flatnonzero(meta["event"] == i1)[0])
    np.testing.assert_allclose(bank[m0], (raw[0, 0] - vs[0, 0])[10000 - 8:10000 + 120], atol=1e-7)
    assert bank[m0][:8].tolist() == [0.0] * 8 and np.abs(bank[m0][8:20]).max() > 0.1
    assert (meta[m0]["partner"], meta[m0]["partner_offset"], meta[m0]["length"]) == (m1, 3, 20)
    assert (meta[m1]["partner"], meta[m1]["partner_offset"]) == (m0, -3)
    m2 = int(np.flatnonzero(meta["event"] == _find(ev, 0, 0, 70000))[0])
    assert (meta[m2]["partner"], meta[m2]["partner_offset"]) == (-1, 0)
    assert meta[m0]["level_db"] == ev[i0]["level_db"]


def test_suspected_and_hardneg(prepped):
    _, _, _, t, raw, vs, lab = prepped
    sus, hard = t["suspected"], t["hardneg"]
    assert any(r["win"] == 1 and r["ch"] == 0 and r["a"] <= 30000 < r["b"] for r in sus)
    assert any(r["win"] == 1 and r["ch"] == 0 and r["a"] <= 50000 < r["b"] for r in hard)
    assert not any(r["win"] == 1 and r["ch"] == 0 and r["a"] <= 50000 < r["b"] for r in sus)
    for r in sus:
        assert r["peak"] > 8 and r["contrast"] > 20
    for tab in (sus, hard):
        for r in tab:
            assert r["a"] >= 2400 and r["b"] <= L - 2400
            assert not (lab[r["win"], r["ch"], max(0, r["a"] - 16):r["b"] + 16] != 0).any()
        assert not any(r["win"] == 1 and r["ch"] == 1 and r["a"] < 2400 + 5000 for r in tab)
        # the labelled clicks of window 0 never show up as regions
        assert not any(r["win"] == 0 and 9900 < r["a"] < 10100 for r in tab)


def test_parallel_matches_serial(prepped, tmp_path):
    root, out, *_ = prepped
    prep_split(root, "val", out=tmp_path, jobs=2, chunk=1, cfg=PrepConfig())
    for k in ["events", "hardneg", "suspected", "bank", "bank_meta"]:
        assert np.array_equal(np.load(tmp_path / f"{k}.npy"), np.load(out / f"{k}.npy")), k


# MARK: - Boundary tests (process_window directly)


def _quiet_window(seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(L) / 48000
    x = 0.05 * np.sin(2 * np.pi * 220 * t) + 1e-3 * rng.standard_normal(L)
    return np.stack([x, x[::-1].copy()])


def _regions(raw, lab):
    _, hard, sus, _, _ = prep.process_window(raw, lab, raw, 0, 0, PrepConfig())
    return [(r[1], r[2], r[3]) for r in hard + sus]


def test_tol_excludes_regions_near_labels():
    raw = _quiet_window()
    raw[0, 50000:50003] += np.array([0.3, -0.2, 0.1])
    lab = np.zeros((2, L), np.uint8)
    hit = [(a, b) for c, a, b in _regions(raw, lab) if c == 0 and a <= 50000 < b]
    assert len(hit) == 1
    a, b = hit[0]

    def kept(lo, hi):
        lab = np.zeros((2, L), np.uint8)
        lab[0, lo:hi] = 2           # label 2: raw (and hence the classical region) unchanged
        return (0, a, b) in _regions(raw, lab)

    # right side: [a - TOL, b + TOL) must be label-free
    assert not kept(b + 5, b + 25)            # 5-sample gap, does not touch the region
    assert not kept(b + 16 - 1, b + 16 + 10)
    assert kept(b + 16, b + 16 + 10)
    assert kept(b + 40, b + 60)
    # left side
    assert not kept(a - 25, a - 5)
    assert not kept(a - 16 - 10, a - 16 + 1)
    assert kept(a - 16 - 10, a - 16)


def _contrast_case(scale, a=50000, n=12):
    """One label-1 run on ch0: a decaying click of the given amplitude on a quiet host."""
    vs = _quiet_window()
    raw = vs.copy()
    raw[0, a:a + n] += scale * np.exp(-np.arange(n) / 3.0) * np.cos(np.arange(n) * 1.3)
    lab = np.zeros((2, L), np.uint8)
    lab[0, a:a + n] = 1
    events, _, _, tpls, bank_rows = prep.process_window(raw, lab, vs, 0, 0, PrepConfig())
    assert len(events) == 1
    return events[0], bool(bank_rows), prep.event_contrast(raw[0], a, a + n)


def test_bank_gate_and_w_evt_follow_curvature_contrast():
    seen_reject = seen_accept = False
    for scale in (2e-4, 5e-4, 1e-3, 3e-3, 1e-2, 0.1):
        e, banked, con = _contrast_case(scale)
        assert banked == (con >= prep.BANK_MIN_CONTRAST), (scale, con)
        assert e[6] == pytest.approx(min(1.0, max(0.3, math.log2(con / prep.W_EVT_REF) / 3)), abs=1e-6)
        seen_reject |= not banked
        seen_accept |= banked
    assert seen_reject and seen_accept          # the sweep straddles the gate
    assert prep.w_evt_from_contrast(16.0) == pytest.approx(1.0)
    assert prep.w_evt_from_contrast(8.0) == pytest.approx(2 / 3)
    assert prep.w_evt_from_contrast(2.0) == pytest.approx(0.3)
