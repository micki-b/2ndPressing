# 2ndPressing

Near-real-time removal of clicks and pops from LP transfers, on modest hardware,
across Linux / macOS / Windows on x86 and ARM. Open source.

## Approach (hybrid)

1. **Detect**: a small causal neural network flags damaged samples.
2. **Repair**: flagged spans only are filled by autoregressive (LSAR) interpolation.
3. **Pass through**: every other sample is untouched (bit-exact).

Training data comes from a personal library of VinylStudio transfers and their
de-clicked counterparts. Audio never goes in this repo (`data/` is git-ignored).

## Layout

- `tools/` — Python (uv): data preparation, label extraction, training.
  - `pressing_tools/crd.py` — reader for VinylStudio `.crd` click-repair files
    (reverse-engineered; format notes in the module docstring).
  - `pressing_tools/labels.py` — derives click labels from an original /
    de-clicked pair (aligns, matches gain/EQ, diffs, detects impulses).
- `runtime/` — Rust real-time engine (planned: nih-plug VST3/CLAP + standalone).

## Tools quick start

```sh
cd tools
uv sync
uv run pytest
uv run pressing-labels pair original.wav declicked.wav -o out/labels --residual
```
