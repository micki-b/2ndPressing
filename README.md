# 2ndPressing

Near-real-time removal of clicks and pops from LP transfers, on modest hardware,
across Linux / macOS / Windows on x86 and ARM. Open source.

## Approach (hybrid)

1. **Detect**: a small causal neural network flags damaged samples.
2. **Repair**: flagged spans only are filled by autoregressive (LSAR) interpolation.
3. **Pass through**: every other sample is untouched (bit-exact).

Training data comes from a personal library of VinylStudio transfers. Each raw
side has a `.crd` sidecar recording every repair VinylStudio made (verified
against the exported de-clicked tracks), which gives sample-accurate labels.
Audio never goes in this repo (`data/` is git-ignored).

## Layout

- `tools/` — Python (uv): data preparation, label extraction, training.
  - `pressing_tools/crd.py` — reader for VinylStudio `.crd` click-repair files
    (reverse-engineered; format notes in the module docstring).
  - `pressing_tools/labels.py` — derives click labels from an original /
    de-clicked pair (aligns, matches gain/EQ, diffs, detects impulses).
  - `pressing_tools/library.py` — matches raw sides to iTunes albums, decodes
    via ffmpeg, locates exported tracks inside sides.
  - `pressing_tools/dataset.py` — builds the training set (see below).
  - `pressing_tools/classical.py` — baseline: AR matched-filter detector +
    curvature-contrast gate + LSAR interpolation.
  - `pressing_tools/evaluate.py` — evaluation harness (`pressing-eval`).
  - `pressing_tools/nn/` — neural detector (ClickNet): features, model, prep
    tables, crop sampler, loss, training, streaming reference, ONNX export,
    synthetic benchmark, reports. Design: `docs/neural-detector-v1.md`.
  - `scripts/verify_crd.py` — checks `.crd` records against exported tracks.
- `runtime/` — Rust real-time engine (planned: nih-plug VST3/CLAP + standalone).

## Tools quick start

```sh
cd tools
uv sync
uv run pytest
uv run pressing-labels pair original.wav declicked.wav -o out/labels --residual
```

## Training set

```sh
uv run pressing-dataset --out ../data/v1 --jobs 8      # resumable
```

Per raw side, random 2 s windows (one per 30 s of audio) resampled to 48 kHz,
stored as `data/v1/<train|val|test>/<side>.npz`:

| array | shape | meaning |
|---|---|---|
| `audio` | (W, 2, L) float16 | raw audio, clicks included |
| `label` | (W, 2, L) uint8 | 0 clean · 1 repaired by VinylStudio · 2 detected, not repaired |
| `repl` | (K,) float16 | VinylStudio's replacement samples at every label 1 position |
| `starts` | (W,) int64 | window start in native side samples |

`dataset.repaired(shard)` rebuilds the VinylStudio-repaired audio. Splits are
per album, so test albums are never seen in training. Label 2 spans are nearly
as impulsive as label 1 (quieter clicks or real transients), so treat them as
"don't care" in training and evaluate on them separately.

## Evaluation

```sh
uv run pressing-eval --split val                       # tune on val
uv run pressing-eval --split test --listen 3           # final numbers + A/B wavs
uv run pressing-eval --split val --set min_contrast=12 --thresholds 3,4,6
```

Detection: event recall on VinylStudio repairs (label 1) and detected-only
spots (label 2, reported but not scored), false alarms per minute (regions
touching no label; this includes genuine clicks VinylStudio didn't mark), and
% of clean audio touched. Repair: curvature "spike ratio" left at real clicks
vs. VinylStudio's own repair, and SNR on synthetic gaps cut into clean music
(where the truth is known) vs. linear interpolation. `--listen N` writes
raw / VinylStudio / ours WAVs to `out/eval/listen/`.

### Classical baseline (test split, 24 unseen albums, 2026-11-14)

Defaults: threshold 4, min_contrast 16 (tuned on val).

| | value |
|---|---|
| VinylStudio repairs found | 59.9 % |
| detected-only spots found | 56.1 % |
| false alarms | 49.7 / min |
| clean audio touched | 0.13 % |
| spike ratio left at clicks (median) | raw 26.6 · VinylStudio 5.1 · ours 1.1 |
| synthetic 12–24-sample gaps, SNR | LSAR 10.9 dB · linear 7.0 dB |
| speed (Python, 1 core, stereo) | ~105× real time |

On val the same settings found 74 % at 101 false alarms/min: albums vary a lot.
The contrast gate is what makes this usable: without it, ~73 % recall costs
~390 false alarms/min (musical attacks in loud passages).

## Neural detector v1 (ClickNet)

Tiny causal TCN on gain-invariant derivative features, stereo-aware, 10.7 ms
lookahead; LSAR repair unchanged. Design in `docs/neural-detector-v1.md`.

```sh
uv run pressing-nn-prep --split train            # tables (also val, test)
uv run pressing-nn-train --run r2 --C 24 --D 512 # ~1 h on an M5 Pro (MPS)
uv run pressing-nn-export --ckpt out/nn/r2/best.pt --out out/nn/r2/export
uv run pressing-eval --method neural --split test \
    --set model=out/nn/r2/export/model.onnx --set threshold=2.75
```

Selected model **R2** (C=24, 20,116 params, 80 KB, 4,920 MAC/sample/channel),
logit threshold 2.75, no contrast gate, post 8 — chosen on val.

### Test split (24 unseen albums), operating points chosen on val

| | classical-stream | **neural R2** |
|---|---|---|
| VinylStudio repairs found | 59.9 % | **76.5 %** |
| detected-only (masked) clicks found | 56.0 % | **68.9 %** |
| false alarms | 49.6 / min | **27.5 / min** |
| clean audio touched | 0.131 % | **0.030 %** |
| recall @ 50 FA/min | 60.0 % | **82.4 %** (+22.4 pts, album bootstrap 95 % CI +19.5 … +28.8) |
| FA/min at 60 % recall | 49.9 | **6.0** |
| spike ratio left (median; VinylStudio 5.1) | 1.2 | 1.4 |

Synthetic test clicks (exact truth, by host loudness <−45 / −45…−30 / −30…−18 /
>−18 dBFS): neural 88.2 / 87.5 / 90.7 / 90.5 % vs classical 75.8 / 73.0 / 73.4 /
73.9 %, with fewer false alarms in every band.

Ablations (val): C=16 is ~1 point behind C=24; halving lookahead (D=256) costs
~4 points and hurts masked clicks; training without click transplants scores
slightly higher on VinylStudio's own labels but its recall falls with loudness
on synthetic clicks (79 → 72 %), i.e. it learns VinylStudio's "leave loud
clicks" habit — transplants remove that bias. Streaming implementation
reproduces offline harness counts exactly.

Listening: `tools/out/eval_final/neural/listen/` (raw / VinylStudio / ours) and
`tools/out/eval_final/listen_false_alarms/` (50 random test false alarms, raw
and repaired, with an index CSV).
