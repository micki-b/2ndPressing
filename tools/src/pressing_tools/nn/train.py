"""Train the neural click detector (spec section 7).

    python -m pressing_tools.nn.train --run c16_d512 --C 16 --D 512

AdamW + warmup/cosine, EMA weights (used for val, checkpoints, export),
periodic val sweep with the harness post-processing, best-by-selection-metric
checkpointing and early stopping. Outputs in out/nn/<run>/.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pressing_tools.nn import valfast
from pressing_tools.nn.data import CropSampler, SamplerConfig
from pressing_tools.nn.features import features_torch
from pressing_tools.nn.loss import click_loss
from pressing_tools.nn.model import ClickNet, NetConfig, macs_per_sample


# MARK: - Helpers


def lr_lambda(warmup: int, steps: int, min_ratio: float):
    def f(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        u = min(1.0, (s - warmup) / max(1, steps - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * u))
    return f


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1.0 - decay)


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def save(path: Path, ema, cfg: NetConfig, step: int, metric: float, curve: dict | None) -> None:
    # Plain Python types only, so the checkpoint loads with torch.load(weights_only=True).
    curve = json.loads(json.dumps(curve)) if curve is not None else None
    torch.save({"model": ema.state_dict(), "net_config": cfg.to_dict(), "post": {},
                "step": int(step), "metric": float(metric), "curve": curve}, path)


# MARK: - Main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pressing-nn-train", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", type=Path, default=Path("../data/v1"))
    ap.add_argument("--out", type=Path, default=Path("out/nn"))
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-steps", type=int, default=10000, help="no early stop before this")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-jobs", type=int, default=8)
    ap.add_argument("--no-transplant", action="store_true")
    ap.add_argument("--train-split", default="train", help="split to sample crops from (smoke tests)")
    ap.add_argument("--skip-align", action="store_true", help="skip the first-eval alignment gate (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(argv)

    run_dir = a.out / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    cfg = NetConfig(C=a.C, D=a.D)
    model = ClickNet(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "data": str(a.data), "out": str(a.out),
                                                     "net": cfg.to_dict(), "params": n_params,
                                                     "macs_per_sample": macs_per_sample(cfg)}, indent=2))
    print(f"[{a.run}] params {n_params}, {macs_per_sample(cfg):.0f} MAC/sample/ch, device {device}", flush=True)

    decay = [p for p in model.parameters() if p.dim() > 1]
    no_decay = [p for p in model.parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, a.steps, 0.01))

    ds = CropSampler(a.data, a.train_split, SamplerConfig(batch=a.batch), seed=a.seed,
                     transplant=not a.no_transplant)
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)

    metrics_f = open(run_dir / "metrics.jsonl", "a")
    best, since_best, t0, t_log = -1.0, 0, time.time(), time.time()
    acc = {"loss": 0.0, "sample": 0.0, "pos": 0.0, "neg": 0.0}
    n_acc = 0
    model.train()
    for step, batch in enumerate(dl, start=1):
        batch = to_device(batch, device)
        logits = model(features_torch(batch["audio"]))
        loss, parts = click_loss(logits, batch, cfg.D)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_update(ema, model, min(a.ema, (1 + step) / (10 + step)))   # warm-up: early EMA tracks the model
        for k in acc:
            acc[k] += parts[k]
        n_acc += 1

        if step % 100 == 0:
            dt = time.time() - t_log
            rec = {"step": step, "lr": sched.get_last_lr()[0], "steps_per_s": 100 / dt,
                   "crops_per_s": 100 * a.batch / dt, **{k: v / n_acc for k, v in acc.items()}}
            print(json.dumps(rec), flush=True)
            metrics_f.write(json.dumps({"type": "train", **rec}) + "\n"); metrics_f.flush()
            acc = {k: 0.0 for k in acc}; n_acc = 0; t_log = time.time()

        if step % a.eval_every == 0 or step == a.steps:
            te = time.time()
            lg_path = run_dir / "val_logits.npy"
            lg = valfast.compute_val_logits(ema, a.data, cfg.D, lg_path, device)
            if step == a.eval_every and not a.skip_align:
                ok, peak, prof = valfast.check_alignment(lg, a.data)
                print(f"[align] mean-logit peak at offset {peak} (want 0..30): {'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    (run_dir / "alignment_fail.json").write_text(json.dumps(prof.tolist()))
                    raise SystemExit("alignment check failed: off-by-D bug?")
            curve = valfast.sweep(lg_path, a.data, jobs=a.eval_jobs)
            metric = valfast.selection_metric(curve)
            r50 = valfast.recall_at_fa(curve, 50.0)
            r50_l2 = valfast.recall_at_fa(curve, 50.0, key="recall_detected_only")
            rec = {"type": "val", "step": step, "metric": metric, "recall@25": valfast.recall_at_fa(curve, 25.0),
                   "recall@50": r50, "recall@100": valfast.recall_at_fa(curve, 100.0),
                   "recall_l2@50": r50_l2, "eval_s": time.time() - te,
                   "curve": {str(t): {k: v for k, v in d.items() if k != "events"} for t, d in curve.items()}}
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            print(f"[val {step}] metric {metric:.4f}  recall@25/50/100 {rec['recall@25']:.3f}/"
                  f"{r50:.3f}/{rec['recall@100']:.3f}  label2@50 {r50_l2:.3f}  ({rec['eval_s']:.0f}s)", flush=True)
            save(run_dir / "last.pt", ema, cfg, step, metric, rec["curve"])
            save(run_dir / f"step{step:06d}.pt", ema, cfg, step, metric, rec["curve"])   # ~40 KB each
            if metric > best:
                best, since_best = metric, 0
                save(run_dir / "best.pt", ema, cfg, step, metric, rec["curve"])
            elif step >= a.min_steps:
                since_best += 1
                if step >= a.min_steps and since_best >= a.patience:
                    print(f"early stop at step {step}; best {best:.4f}", flush=True)
                    break
            model.train()
        if step >= a.steps:
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min; best metric {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
