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
