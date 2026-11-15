"""Loss for the click detector (spec section 6).

    L_sample = sum w * BCE(z, y; pos_weight) / sum w
    L_pos    = mean_e w_e * BCE(max z[a_e-4 : b_e+4], 1)     (one confident sample per click)
    L_neg    = mean_r BCE(max_{neg_ok} z[chunk_r], 0)          (32 chunks of 256 per sequence)
    L        = L_sample + 0.5 L_pos + 0.5 L_neg

``logits`` are network outputs (N, T); the prediction for target sample t is
``logits[:, t + D]``. The D shift is applied here, exactly once.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from pressing_tools.nn.data import S0, S1

MIL_PAD = 4
MIL_WIDTH = 256
CHUNK = 256
CHUNK_MIN = 128


def target_logits(logits: torch.Tensor, D: int) -> torch.Tensor:
    """z[:, t] = logits[:, t + D] for t in [0, T - D)."""
    return logits[:, D:]


def click_loss(logits: torch.Tensor, batch: dict, D: int, pos_weight: float = 4.0,
               w_pos: float = 0.5, w_neg: float = 0.5) -> tuple[torch.Tensor, dict]:
    z_all = target_logits(logits, D)                 # (N, T - D)
    assert z_all.shape[1] >= S1, "crop too short for scored region + lookahead"
    z = z_all[:, S0:S1]
    y = batch["target"][:, S0:S1].float()
    w = batch["weight"][:, S0:S1].float()
    pw = torch.tensor(pos_weight, device=z.device)
    bce = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
    l_sample = (bce * w).sum() / w.sum().clamp_min(1.0)

    # Multiple-instance positive term: max over [a-4, b+4) per event.
    ev, ev_w = batch["ev_idx"], batch["ev_w"]
    if ev.shape[0]:
        seq, a, b = ev[:, 0], ev[:, 1] - MIL_PAD, ev[:, 2] + MIL_PAD
        offs = torch.arange(MIL_WIDTH, device=z.device)
        idx = a[:, None] + offs[None, :]
        valid = (idx < b[:, None]) & (idx >= 0) & (idx < z_all.shape[1])
        g = z_all[seq[:, None].expand_as(idx), idx.clamp(0, z_all.shape[1] - 1)]
        m = torch.where(valid, g, torch.full_like(g, -1e4)).amax(dim=1)
        l_pos = (ev_w * F.softplus(-m)).sum() / ev_w.sum().clamp_min(1e-6)
    else:
        l_pos = z.new_zeros(())

    # Chunk-max negative term over safe negatives only.
    ok = batch["neg_ok"][:, S0:S1]
    N, S = z.shape
    zc = z.reshape(N, S // CHUNK, CHUNK)
    okc = ok.reshape(N, S // CHUNK, CHUNK)
    use = okc.sum(dim=-1) >= CHUNK_MIN
    mx = torch.where(okc, zc, torch.full_like(zc, -1e4)).amax(dim=-1)
    l_neg = F.softplus(mx[use]).mean() if use.any() else z.new_zeros(())

    loss = l_sample + w_pos * l_pos + w_neg * l_neg
    return loss, {"loss": float(loss.detach()), "sample": float(l_sample.detach()),
                  "pos": float(l_pos.detach()), "neg": float(l_neg.detach())}
