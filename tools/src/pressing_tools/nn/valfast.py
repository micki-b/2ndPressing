"""Fast in-training validation (spec section 7).

1. Model over every val window (full 2 s, both channels) on device; target-
   aligned logits (N, 2, L) go to a float16 memmap.
2. A process pool turns logits into regions with the *harness* code
   (classical.regions_from_scores + evaluate.detection_counts) over a sweep of
   logit thresholds, and aggregates Counts.
3. Selection metric = mean label-1 recall at 25 / 50 / 100 false alarms per
   minute, interpolated linearly in log(FA).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pressing_tools import classical
from pressing_tools.evaluate import Counts, detection_counts
from pressing_tools.memmap import open_split
from pressing_tools.nn.features import features_torch

THRESHOLDS = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
FA_TARGETS = (25.0, 50.0, 100.0)


# MARK: - Logits


@torch.no_grad()
def window_logits(model, audio: torch.Tensor, D: int) -> torch.Tensor:
    """audio (B, 2, L) -> target-aligned logits (B, 2, L). Features are padded
    right with D zeros (plus to a multiple of 4), as NeuralMethod does."""
    B, _, L = audio.shape
    f = features_torch(audio)
    pad = D + (-(L + D)) % 4
    out = model(F.pad(f, (0, pad)))[:, D:D + L]
    return out.reshape(B, 2, L)


def compute_val_logits(model, data: Path, D: int, out_path: Path, device, batch: int = 32) -> np.ndarray:
    mm = open_split(data, "val")
    audio = mm["audio"]
    N, C, L = audio.shape
    out = np.lib.format.open_memmap(out_path, "w+", np.float16, (N, C, L))
    model.eval()
    for i in range(0, N, batch):
        x = torch.from_numpy(np.asarray(audio[i:i + batch], np.float32)).to(device)
        out[i:i + batch] = window_logits(model, x, D).float().cpu().numpy().astype(np.float16)
    out.flush()
    return out


# MARK: - Counting


def post_config(thr: float) -> classical.ClassicalConfig:
    return classical.ClassicalConfig(threshold=thr, extend_threshold=thr - 2.0, merge=8, pre=2,
                                     post=8, max_region=480, min_contrast=0.0)


def _count_chunk(logits_path: str, data: str, w0: int, w1: int, thresholds: list[float]) -> dict:
    lg = np.load(logits_path, mmap_mode="r")
    lab = open_split(Path(data), "val")["label"]
    tot = {t: Counts() for t in thresholds}
    for w in range(w0, w1):
        for c in range(lg.shape[1]):
            s = np.asarray(lg[w, c], np.float64)
            lb = np.asarray(lab[w, c])
            for t in thresholds:
                m = classical.regions_from_scores(s, post_config(t), t)
                tot[t].add(detection_counts(lb, m))
    return {t: asdict(c) for t, c in tot.items()}


def sweep(logits_path: Path, data: Path, thresholds=THRESHOLDS, jobs: int = 8) -> dict[float, dict]:
    n = np.load(logits_path, mmap_mode="r").shape[0]
    bounds = np.linspace(0, n, jobs * 4 + 1).astype(int)
    tot = {t: Counts() for t in thresholds}
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(_count_chunk, str(logits_path), str(data), int(a), int(b), list(thresholds))
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        for f in futs:
            for t, c in f.result().items():
                tot[t].add(Counts(**c))
    return {t: c.summary() for t, c in tot.items()}


# MARK: - Selection metric


def recall_at_fa(curve: dict[float, dict], fa_target: float, key: str = "recall_repaired") -> float:
    """Recall on the (FA, recall) Pareto envelope at fa_target -- report.py's
    implementation, so in-training numbers match the final report."""
    from pressing_tools.nn import report
    env = report.pareto(report.points_from_aggregate({"detection": curve}))
    r = report.recall_at_fa(env, fa_target)
    return float(r["recall" if key == "recall_repaired" else "recall_l2"])


def selection_metric(curve: dict[float, dict]) -> float:
    return float(np.mean([recall_at_fa(curve, f) for f in FA_TARGETS]))


# MARK: - Alignment check


def alignment_profile(logits: np.ndarray, data: Path, max_onsets: int = 5000) -> np.ndarray:
    """Mean logit at offsets -64..+64 around val label-1 onsets."""
    lab = open_split(data, "val")["label"]
    prof, n = np.zeros(129), 0
    for w in range(lab.shape[0]):
        for c in range(lab.shape[1]):
            l1 = np.asarray(lab[w, c]) == 1
            for a, _ in classical.runs(l1):
                if 2400 <= a < lab.shape[2] - 2400:
                    prof += np.asarray(logits[w, c, a - 64:a + 65], np.float64)
                    n += 1
        if n >= max_onsets:
            break
    return prof / max(n, 1)


def check_alignment(logits: np.ndarray, data: Path) -> tuple[bool, int, np.ndarray]:
    prof = alignment_profile(logits, data)
    peak = int(np.argmax(prof)) - 64
    return 0 <= peak <= 30, peak, prof
