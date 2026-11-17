"""End-to-end test of dataset building on a synthetic side + .crd file."""

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pressing_tools import library as lib
from pressing_tools.dataset import (LABEL_CLEAN, LABEL_DETECTED, LABEL_REPAIRED, DatasetConfig,
                                    _resample, build_side, load_shard, repaired)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

SR = 44100  # non-canonical on purpose: exercises resampling + label mapping


def _crd(channels: list[list[bytes]]) -> bytes:
    body = b"".join(b"".join(recs) + b"\x00\x00" for recs in channels)
    return b"CRDF" + struct.pack("<I", len(channels)) + body


def _rep(start: int, samples: np.ndarray) -> bytes:
    return (struct.pack("<HHIIH", 2, 0, start, start + len(samples), 0)
            + np.round(samples * 32768).astype("<i2").tobytes())


def _det(start: int, end: int) -> bytes:
    return struct.pack("<HHIIH", 12, 0, start, end, 40)


def test_build_side_labels_and_repairs(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = SR * 6
    t = np.arange(n) / SR
    clean = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)], 1)
    raw = clean.copy()
    # Clicks every 0.25 s on ch0; every other one on ch1 is "detected only".
    rep0, rep1, det1 = [], [], []
    for k, pos in enumerate(range(SR // 2, n - SR // 2, SR // 4)):
        raw[pos:pos + 20, 0] += 0.5
        rep0.append(_rep(pos, clean[pos:pos + 20, 0]))
        if k % 2:
            det1.append(_det(pos + 5, pos + 40))
        else:
            raw[pos:pos + 10, 1] -= 0.4
            rep1.append(_rep(pos, clean[pos:pos + 10, 1]))
            det1.append(_det(pos + 2, pos + 30))    # overlaps the repair: repair wins

    side_path = tmp_path / "Artist~Album_[LP]~side1.flac"
    sf.write(side_path, raw.astype(np.float32), SR, subtype="PCM_24")
    (tmp_path / (side_path.name + ".crd")).write_bytes(
        _crd([rep0, sorted(rep1 + det1, key=lambda b: struct.unpack_from("<I", b, 4)[0])]))

    side = lib.find_sides(tmp_path)[0]
    cfg = DatasetConfig(seconds_per_window=1.0, window_s=1.0)
    meta = build_side(side, tmp_path / "ds", cfg)
    shard = load_shard(tmp_path / "ds" / meta["shard"])

    W = meta["windows"]
    assert shard["audio"].shape == (W, 2, 48000) and shard["label"].shape == (W, 2, 48000)
    assert int(shard["sr_native"]) == SR and int(shard["sr"]) == 48000
    lab = shard["label"]
    assert set(np.unique(lab)) <= {LABEL_CLEAN, LABEL_REPAIRED, LABEL_DETECTED}

    # Every labelled-repaired run on ch0 is ~20 native samples (~22 canonical).
    for w in range(W):
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, lab[w, 0] == LABEL_REPAIRED, 0])))[::2]
        assert all(20 <= r <= 24 for r in runs[(runs > 0)]), runs

    # Ch1: overlapping repair beats detection -- the click centre is labelled repaired.
    assert (lab[:, 1] == LABEL_DETECTED).any() and (lab[:, 1] == LABEL_REPAIRED).any()

    # Repaired audio ~ clean signal; raw audio ~ raw. Compare in canonical rate.
    fixed = repaired(shard)
    m = cfg.resample_margin
    off = round(m * 48000 / SR)
    for w in range(W):
        s = int(shard["starts"][w])
        ref = _resample(raw[s - m:s + SR + m], SR, 48000)[off:off + 48000].T
        # Window audio equals the raw signal resampled (float16 + 24-bit precision).
        assert np.max(np.abs(shard["audio"][w].astype(np.float32) - ref)) < 2e-3
    # At repaired positions the repaired audio has no +0.5 step.
    rep = lab == LABEL_REPAIRED
    assert np.max(np.abs(fixed[rep])) < 0.36 and np.max(shard["audio"][:, 0][rep[:, 0]]) > 0.5
