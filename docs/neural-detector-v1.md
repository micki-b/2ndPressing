# 2ndPressing neural click detector v1 ("ClickNet-Nano v1"): implementation spec

## 0. Decisions in brief

- **Base design:** ClickNet-Nano, which both judges ranked first. It is a strided, dilated, causal TCN built only from conv, ReLU and add. One set of weights runs per channel and also sees the other channel's features. Output is a per-sample logit with a fixed output delay `D`.
- **What changes from Nano:**
  - The per-hop running-median gain is replaced by a causal geometric-mean normaliser. It needs no lookahead and does not depend on hop phase, which fixes the latency error both judges found.
  - The context is longer: 8 blocks with dilations up to 128 give a receptive field of 2048 samples, of which 1535 are past.
  - The default lookahead is `D = 512` (10.7 ms). `D = 256` is run as an ablation.
  - BatchNorm is removed. Residual branches are zero-initialised and EMA weights are used, so there is nothing to fold at export.
- **Grafts:**
  - Suspected missed clicks are ignored rather than treated as negatives (from proposals 2 and 4). This matters because Nano's chunk-max negative loss would otherwise train away real clicks VinylStudio never marked.
  - Clicks are transplanted onto or near mined musical attacks (proposal 4).
  - Stereo-paired templates keep the real channel coincidence (proposal 4).
  - Real positives are weighted by confidence, `w_evt` (proposal 4).
  - Synthetic clicks below an audibility threshold are ignored (proposal 4, with the threshold raised as judge 1 asked).
  - A split-clean synthetic benchmark with exact truth, per level bin, measures loud-click recall (proposals 2 and 4).
  - The full val evaluation is also run through the streaming implementation (proposal 2).
  - Classical is re-baselined under the same `max_region` and streaming repair context, and both methods are compared with paired album-bootstrap CIs at matched false alarms (all judges).
- **Not in v1:** see section 15.

Before writing this I measured four things on this machine:
- **Parameter counts:** obtained by building the model in PyTorch.
- **MPS training step:** C=16 takes **0.075 s** and C=24 **0.117 s** for 256 sequences × 10,240 samples (forward, backward and AdamW, random data).
- **ORT single-thread:** a 2 s stereo window takes **12.4 ms** at C=16 (RTF 0.0062, about 34 GMAC/s effective) and **23.5 ms** at C=24 (RTF 0.0117).
- **ONNX vs torch:** max |diff| 7e-8.
- **Train label counts:** estimated from 2,000 random train windows. About **230k label-1 runs** (judge 2 counted about 270k; the sample-fraction estimate is about 250k), about **150k label-2 runs**, and label-1 lengths of p50 19, p90 34, mean 22.8, max 219.

Nothing was written to the repo.

---

## 1. Files

New package `tools/src/pressing_tools/nn/`:

| file | purpose | CLI |
|---|---|---|
| `nn/__init__.py` | empty | |
| `nn/features.py` | Feature front-end. `features_np` (float64 reference), `features_torch` (training, on device) and `FeatureState` (streaming). Defines the constants. | |
| `nn/model.py` | `NetConfig`, `ClickNet` (torch), and the `receptive_field()` and `macs_per_sample()` helpers | |
| `nn/prep.py` | One-time tables: events, hard negatives, suspected clicks, template bank | `pressing-nn-prep` |
| `nn/data.py` | `CropSampler` (an IterableDataset that yields whole batches), transplant code, target and weight maps | |
| `nn/loss.py` | Sample BCE, event MIL term and chunk-max negative term | |
| `nn/train.py` | Training loop, EMA, checkpointing, early stop | `pressing-nn-train` |
| `nn/valfast.py` | In-training val sweep: logits on MPS, then `regions_from_scores` + `detection_counts` in a process pool | |
| `nn/export.py` | ONNX, raw weights + JSON, and golden vectors for Rust | `pressing-nn-export` |
| `nn/stream.py` | Numpy streaming reference (ring buffers). This is the spec the Rust port follows. | |
| `nn/method.py` | `NeuralConfig` and `NeuralMethod` for `evaluate.py` | |
| `nn/synth.py` | Synthetic click benchmark from a split's own click bank, one directory per level bin | `pressing-nn-synth` |
| `nn/report.py` | Pareto curves, matched-FA interpolation, paired album bootstrap | `pressing-nn-report` |

Changes to existing files:
- **`classical.py`:** add `repair_context_right: int = 0` to `ClassicalConfig` (0 means use `repair_context`), and use it for the `right` slice in `lsar_fill`. Add `max_group: int = 0` (0 means unlimited): when a group's span `u1 - u0` would exceed it, start a new group. Defaults leave current behaviour unchanged.
- **`evaluate.py`:** see section 9.
- **`pyproject.toml`:** add the script entries above. Torch, onnx and onnxruntime are already in the `train` group.
- **`memmap.py`:** commit the untracked module as is. `data/v1/mm/{train,val,test}` already exist (train audio 16.9 GB, labels 8.4 GB).

New tests:
- `tests/test_nn_features.py`
- `tests/test_nn_model.py`
- `tests/test_nn_data.py`
- `tests/test_nn_stream.py`
- an extension of `tests/test_evaluate.py`

Run artifacts go to `tools/out/nn/<run>/` (already git-ignored). Prep tables go to `data/v1/nn/<split>/`.

---

## 2. Input front-end (`nn/features.py`)

These are per channel, at 48 kHz, in float64 for the reference and float32 for torch and Rust. **All of it is causal: no lookahead.**

```
d1[n] = x[n] - x[n-1]                  (x[-1] = 0)
d2[n] = d1[n] - d1[n-1]                (causal curvature, centred at n-1)
for k in {1, 2}:
    l_k[n] = ln(|d_k[n]| + EPS)                                  EPS   = 1e-6
    m_k[n] = mean(l_k[max(0, n-NW+1) .. n])                       NW    = 2048, expanding window at start
    s_k[n] = max(exp(m_k[n]), FLOOR)                              FLOOR = 1e-5
    g_k[n] = asinh(d_k[n] / s_k[n] / 8)
```

- The scale is the geometric mean of |d| over the last 43 ms. Averaging in the log domain makes it robust to the click itself: a 25-sample click at 30× shifts it by about 0.04 nat.
- The newest input sample is `D` ahead of the target, so the scale window relative to the target is `[t-1535, t+512]`.
- On noise, `|d|/s` is about 1-2 and maps to roughly 0.2 after asinh. A 25× click maps to about 2, and a 1e5 pop to about 10.
- **Absolute level is not an input.** That keeps the network from learning VinylStudio's "loud means leave it" policy.

**Model input per channel** (3 features, with weights shared between L and R):
- `self = [g1_self, g2_self, g1_other]`
- L uses `[g1_L, g2_L, g1_R]` and R uses `[g1_R, g2_R, g1_L]`. Each channel's `g1` is computed once and reused for the other channel.
- Mono input uses other = self.

**Implementations:**
- `features_np(x2) -> (2, 3, L)`: float64, using cumsum for the window means.
- `features_torch(x: (B, 2, T)) -> (2B, 3, T)`: float32 on MPS, cumsum over T = 12,288. The float32 cumsum error is about 1e-5 in log terms, which is negligible.
- `FeatureState.process(block2) -> (2, 3, n)`: rings of `l_k` with **float64 running sums**, for any block size.

**Cost in Rust:** 2 ln, 2 exp and 2 asinh per sample per channel, under 1% of a Pi 5 core.

**Tests** (`test_nn_features.py`):
- torch vs numpy: max abs < 1e-4
- streaming at random block sizes 1-4096 vs offline: < 1e-9 (float64)
- gain invariance: ×10 gives the same output to 1e-6 wherever s > FLOOR
- digital silence gives an all-zero output

---

## 3. Architecture (`nn/model.py`)

```
NetConfig: C=16, blocks=8 (dilations 1,2,4,...,128 frames), stem_k=8, stride=4, D=512
```

Notation is N sequences (2 per stereo window) and T input samples, with T a multiple of 4 and F = T/4.

| # | layer | output shape | notes |
|---|---|---|---|
| in | features | (N, 3, T) | |
| 1 | CausalConv1d(3→C, k=8, stride=4), left pad 4, ReLU | (N, C, F) | frame j sees samples [4j−4, 4j+3] |
| 2–9 | ResBlock(d ∈ {1,2,…,128}): `y = ReLU(x + Conv1x1(C→C)(ReLU(CausalDilConv(C→C, k=3, d)(x))))` | (N, C, F) | left pad 2d; last 1×1 is zero-initialised; no norm layers |
| 10 | Conv1x1(C→C), ReLU | (N, C, F) | |
| 11 | Conv1x1(C→4) | (N, 4, F) | sub-pixel head |
| out | permute + reshape | (N, T) | `logit[4j+r] = y[:, r, j]` |

**Output semantics:** `logit[n]` is the prediction for **target sample `n − D`**. It becomes available once input sample `4⌊n/4⌋+3` has arrived.

**Receptive field:**
- 8 + 4·2·(1+…+128) = **2048 samples (42.7 ms)**.
- With D=512, each decision sees about **1535 samples of past and 512-515 of future**.
- With D=256 it is 1791 and 256.
- The classical contrast context of ±150 is covered many times over.

**Parameter and compute formulas:**
- Parameters (C): 33C² + 46C + 4.
- MACs per sample per channel: 8.25C² + 7C. That is 6C for the stem, C² per block, and C²/4 + C for the head.

| C | params (measured) | MAC/sample/ch | MMAC/s stereo 48 kHz | weights fp32 |
|---|---|---|---|---|
| 8 | 2,484 | 584 | 56.1 | 9.9 KB |
| **16 (v1 default)** | **9,188** | **2,224** | **213.5** | **36.8 KB** |
| 24 (step-up) | 20,116 | 4,920 | 472.3 | 80.5 KB |
| 32 | 35,268 | 8,672 | 832.5 | 141 KB |

**Runtime estimate for C=16** (Pi 5, Cortex-A76 at 2.4 GHz, fp32 NEON peak 19.2 GMAC/s):
- Hand-written kernels at a realistic 2-5 GMAC/s give **4-11% of one core**.
- The C=24 step-up is 9-24% of one core.
- LSAR plus the front-end add under 2%.
- These are estimates. The Pi benchmark comes later (section 15).

**`test_nn_model.py`:**
- **Causality:** perturbing input index k changes no logit at n < 4⌊k/4⌋.
- **Receptive field:** no logit changes at n ≥ k + 2048.
- The parameter count matches the formula, and `macs_per_sample()` matches the table.

---

## 4. One-time prep (`pressing-nn-prep --split train|val|test`, about 15 min with 16 processes)

This reads `data/v1/mm/<split>` and uses `index.json` for side and album IDs. It writes `data/v1/nn/<split>/`:

1. **`events.npy`**, one row per label-1 run: `win, ch, a, b, side_id, level_db, w_evt, partner_offset`.
   - `level_db`: 20·log10 RMS of the VinylStudio-repaired audio over ±1024 samples.
   - `w_evt = clip(log2(peak|raw−VS| / RMS(VS over ±256)) / 3, 0.3, 1)`. This down-weights "repairs" whose residual sits at the noise floor.
   - `partner_offset`: the offset of an overlapping label-1 run in the other channel within ±8 samples, or the sentinel `INT32_MIN`.
2. **Classical pass with the gate off.** Run `classical.scores` and `regions_from_scores(threshold=4, min_contrast=0)` on every channel-window: about 15 CPU-minutes, about 1 min on 16 processes. Each region not within TOL=16 of any label goes to one of two tables:
   - **`suspected.npy`** if peak score > 8 **and** `classical.contrast` > 20 and there is no label within ±64. These are likely clicks VinylStudio missed and will be **ignored** in training.
   - **`hardneg.npy`** otherwise. These are mostly musical attacks. Each row stores `win, ch, a, b, peak, contrast, side_id`.
3. **`bank.npy`** (M, 128) float32 and **`bank_meta.npy`**. Residual templates `raw − VS` over each label-1 run of length 96 or less, with 8 samples of pre-roll.
   - Keep a template only if peak|r| ≥ 4 × RMS of VS over ±256.
   - Metadata: length, the partner template index (for stereo pairs), `level_db`.
   - Expected size about 150-200k templates, about 100 MB. Banks are built per split and **never cross splits**.

**Sanity printout (a gate before training):**
- counts per level bin (< −45, −45..−30, −30..−18, > −18 dBFS)
- suspected and hard negatives per channel-minute

If suspected is more than 40% of the classical gate-off false alarms, tighten to contrast > 30 before training.

---

## 5. Data loader (`nn/data.py`)

**Crop geometry.** Each crop is a stereo slice of length **T = 12,288** from one 2 s window, with start in [0, 96000 − T]:
- Features are computed over all 12,288 samples on device.
- Targets are scored on `[3584, 11776)`, which is 8,192 samples per sequence. Every scored target then has its full 1535-sample network history and about 2048 samples of normaliser history inside the crop.
- The last 512 samples supply the D lookahead.
- In memory each crop is two contiguous 24.6 KB reads from the (N, 2, L) memmap, plus 24 KB of labels.

**Batch:** 128 stereo crops, giving 256 sequences.
- **40% event-centred:** pick a label-1 event with P ∝ n_side^(−1/2) · n_levelbin^(−1/2). The first factor keeps crackly sides from dominating; the second upweights the rare loud label-1 events by about 3-4×. Place the event uniformly inside the scored region, with at least 64 samples of margin.
- **30% hard-negative-centred:** pick a `hardneg` row with P ∝ n_side^(−1/2) and place it the same way.
- **30% uniform:** uniform window and uniform offset.

**Transplant** (in 50% of uniform and hard-negative crops, 1-3 clicks, train bank only):
- **Position:**
  - In hard-negative crops, the first click is placed with probability 0.5 at the attack peak offset by U(−200, 200) samples. Without this, the net could learn "attack nearby means no click".
  - Otherwise the position is uniform over scored positions with no label or suspected region within ±64. In half of the draws it is restricted to positions with local RMS (±1024) > −30 dBFS, if any exist.
- **Scale:** draw the target contrast c* log-uniformly in [5, 120]. Set `scale = c* · median_curv(host) / peak_curv(template)`, using the `classical.contrast` definition (ctx 150, guard 20). Random sign. Reject the click if |x| would exceed 0.999.
- **Stereo:** if the template has a partner, insert it into the other channel at its stored offset with the **same scale**. This keeps the real 48% coincidence and the inter-channel ratio.
- **Label:** the support is first-to-last sample with |click| > 5% of its peak. Measure `classical.contrast` on the augmented channel:
  - **≥ 8:** positive, weight 1.
  - **< 8:** ignored over the support ±4. That is the audibility rule; the threshold sits above the ~7× attack range, as judge 1 asked.

**Whole-crop augmentation:**
- polarity flip, p = 0.5
- gain U(−20, +6) dB with clip, which exercises the normaliser floor
- mono duplication, p = 0.05: channel B is replaced by A, and B's labels and tables are copied from A

There is no L/R swap, because both orderings are already trained per crop. There is no time reversal and no EQ.

**Target and weight maps** (per channel, uint8 target and float16 weight; weight 0 means ignored):

| region | target | weight |
|---|---|---|
| label-1 run | 1 | `w_evt` |
| 3 samples before and 6 after each label-1 run (VinylStudio extents truncate tails) | — | 0 |
| label-2 run dilated ±8 (don't care) | — | 0 |
| suspected region dilated ±16 | — | 0 |
| transplant with contrast ≥ 8 | 1 | 1 |
| transplant with contrast < 8, support ±4 | — | 0 |
| outside the scored region | — | 0 |
| everything else | 0 | 1 |

**Extra per-crop outputs:**
- **Event table** (seq, a, b, w): real label-1 and positive transplants fully inside the scored region.
- **`neg_ok` mask:** samples whose weight is 1 with target 0 and with no non-zero target and no ignore within ±16.

**Worker set-up:**
- An `IterableDataset` builds **whole batches inside each worker** (DataLoader `batch_size=None`, 8 workers, `persistent_workers`, `prefetch_factor=4`).
- Seed with (seed, worker_id, step).
- The required throughput is about 1,100 crops/s, i.e. a budget of about 7 ms per crop per worker.

**`test_nn_data.py`:**
- the map rules on a hand-built window
- a transplant into silence gives contrast far above 8, and a transplant into loud noise at c*=5 is ignored
- the partner is placed in the other channel
- scored-region indices are correct

---

## 6. Loss (`nn/loss.py`)

Notation: `z = logits[:, D + t]` for scored targets t (the training loop applies the `D` shift exactly once).

```
L_sample = Σ w · BCEwithLogits(z, y, pos_weight=4) / Σ w
L_pos    = mean_e  w_e · BCE( max(z[a_e−4 : b_e+4]), 1 )        # MIL: one confident sample per click
L_neg    = mean_r  BCE( max_{neg_ok}(z[chunk_r]), 0 )            # 32 chunks of 256 per sequence; chunk used if ≥128 neg_ok samples
L        = L_sample + 0.5·L_pos + 0.5·L_neg
```

- **MIL max:** implemented as a padded gather `(E, 256)` with masking. That is MPS-friendly with no scatter.
- **L_neg excludes suspected and label-2 areas** through `neg_ok`. This is the fix for Nano's chunk-max term suppressing real unmarked clicks.
- **Imbalance:** about 0.25% positive samples after sampling. `pos_weight=4`, no focal loss.

---

## 7. Training (`pressing-nn-train --run c16_d512 --C 16 --D 512`)

- **Precision and device:** fp32 on MPS (CPU fallback with 16 threads).
- **Optimiser:** AdamW, peak lr 3e-3, betas (0.9, 0.99), weight decay 1e-4 on conv weights only, grad-clip 1.0.
- **Schedule:** 500 linear warmup steps, then cosine to 3e-5 at **30k steps**.
- **EMA** of weights at 0.999. EMA weights are used for val, checkpoints and export.
- **Volume:** about 63 G scored channel-samples in total, about 7.5 passes over train-equivalent, with clicks oversampled.
- **Step cost:** measured 0.075 s at T=10,240. That is about 0.09 s at T=12,288, plus features and loss, so about 0.12 s per step: **about 60 min of steps for C=16 and about 90 min for C=24.**

**Validation (`valfast.py`, every 2,000 steps, about 1 min):**
1. EMA model over all 1,879 val windows (full 2 s, both channels, about 3 s on MPS).
2. Logits go to a float16 memmap in the run directory.
3. A pool of 8 workers runs `regions_from_scores` (logit thresholds {−1, 0, …, 8}, `extend = thr − 2`, merge 8, pre 2, post 8, max_region 480, min_contrast 0) and then `evaluate.detection_counts`.
4. **Selection metric:** the mean of label-1 recall at 25, 50 and 100 FA/min, interpolated linearly in log FA. If a target FA is never reached, use the recall at the lowest-FA point. Label-2 recall at 50 FA/min is also logged.
5. **Alignment check (first eval, automatic):** the mean logit at offsets −64..+64 around val label-1 onsets must peak within [0, +30]. If not, abort: it indicates an off-by-D bug.

**Stopping:**
- Keep `best.pt` by the selection metric.
- Stop early if there is no improvement in 4 consecutive evals after step 10k; otherwise run to 30k.
- Write `metrics.jsonl` (loss terms, selection metric, curve) and `last.pt`.

---

## 8. Post-processing (`nn/method.py`, reusing `classical.regions_from_scores` unchanged)

```python
@dataclass
class NeuralConfig:
    model: str = ""            # path to .onnx
    threshold: float = 3.0     # logit; final value chosen on val
    ext_drop: float = 2.0      # extend_threshold = threshold - ext_drop (logit units)
    merge: int = 8; pre: int = 2; post: int = 8
    max_region: int = 480      # 10 ms; longer = pop, passed through (out of scope)
    min_contrast: float = 0.0  # optional gate, swept on val
    contrast_context: int = 150
    repair_order: int = 32; repair_context: int = 1024
    repair_context_right: int = 512; max_group: int = 480
    stream: int = 0            # 1 = compute logits with nn/stream.py (block 512) instead of ORT
```

- **`mask`:** build `ClassicalConfig(threshold, extend_threshold = thr − ext_drop, merge, pre, post, max_region, min_contrast, contrast_context)` and call `regions_from_scores(s, cfg, thr, x=x)`. Logit-space hysteresis is additive.
- **`repair`:** `classical.lsar_fill` with the repair fields. Unflagged samples are bit-exact.

---

## 9. Harness integration (`evaluate.py`)

1. `make_method("neural", overrides)` returns `NeuralMethod(NeuralConfig(**overrides))`.
2. **`--set` parsing:** try int, then float, then str, so `--set model=out/nn/.../model.onnx` works.
3. **Stereo path.** A method may define `stereo = True` and `scores_stereo(x2: (2, L)) -> (2, L)`. `evaluate_shard` (and `write_listening`) call it once per window, then loop channels exactly as now for `mask`, `repair` and `detection_counts`. `ClassicalMethod` is untouched.
4. **`NeuralMethod.scores_stereo`:**
   - `f = features_np(x2)`, padded right with D zeros and rounded up to a multiple of 4.
   - Run ORT with a cached session per process, `intra_op_num_threads=1`.
   - Return `logits[:, D:D+L]`.
   - The right edge and the start-up both lie inside EDGE = 2400 (network history 1535 is less than 2400; the normaliser is expanding).
   - Measured cost: about 12 ms per window, so val needs about 25 s of compute in total over 12 jobs.
5. **Default thresholds:** a method attribute `default_thresholds`. Neural uses `-1,0,1,2,3,4,5,6,7,8`; classical keeps its current list.
6. **`--split`:** accept any subdirectory of `--data` (e.g. `synth/val_lvl2`).
7. **Per-shard output:** also write `<tag>.shards.json` with per-shard `Counts` per threshold, for album bootstrap in `report.py`.

The test extends `test_evaluate.py` with a dummy stereo method.

---

## 10. Threshold and post-processing selection (val only)

1. **Re-baseline classical as "classical-stream":**
   - `--set max_region=480 --set repair_context_right=512 --set max_group=480`
   - min_contrast in {0, 8, 12, 16, 24}, thresholds 3-24
   - That is 5 runs of about 1 min each.
   - Its own detector still uses 1024 samples of future; only post-processing and repair are matched. The published defaults are also re-run for continuity.
2. **Neural first stage:** sweep min_contrast in {0, 4, 8, 12, 16} at post=8, with the logit-threshold sweep inside each run.
3. **Neural second stage:** at the best gate, run post in {4, 16}. Accept a smaller post only if the median spike ratio is ≤ 1.5 and `repaired_samples_covered` does not drop.
4. **Operating point:**
   - Take the highest val label-1 recall subject to **FA ≤ 50/min** and clean touched ≤ 0.13%.
   - If moving one threshold step changes recall by more than 3 points (a steep, noisy region), take the more conservative neighbour.
   - Write the chosen `threshold`, `min_contrast` and `post` into the export JSON.
5. **Also report:**
   - recall at 25, 50 and 100 FA/min on both methods' Pareto envelopes
   - FA/min at matched recall of 74% (val)
   - `pressing-nn-report` also produces paired album-bootstrap 95% CIs (2,000 resamples over albums) of the neural − classical-stream recall difference at 50 FA/min

**Go/no-go on val (decided now):**
- **Go if all of these hold:**
  - recall@50 FA/min is at least 5 points above classical-stream, with the bootstrap CI excluding 0
  - label-2 recall at the operating point is not below classical-stream's
  - synthetic-benchmark recall in the two loud bins is at least classical-stream's
- **If not:** train C=24 (section 13) before changing anything else.
- **Test** is run **once** on the frozen candidate, with the README-style table next to classical and classical-stream.

---

## 11. Synthetic benchmark (`pressing-nn-synth --split val|test`)

- **Hosts:** windows of the split itself, using **only that split's own bank**.
- **Per level bin** (< −45, −45..−30, −30..−18, > −18 dBFS):
  - Select up to 400 windows.
  - Inject about 3 clicks per channel-second at positions in that bin: at least 256 samples from any original label, at least 512 apart, outside EDGE.
  - c* is log-uniform in [8, 100], with partners and random sign.
- **Output** goes to `data/v1/synth/<split>_lvl<k>/*.npz` in the normal shard format:
  - original labels 1 and 2 become 2 (don't care)
  - injected clicks with measured contrast ≥ 8 become 1, otherwise 2
  - `repl` holds the clean host samples
- **What the unchanged harness then gives per level bin:**
  - recall against exact truth
  - spike ratio
  - `agreement_with_vinylstudio_snr` = repair SNR against the true host
- This is the direct check that the quiet/loud labelling bias was removed. It complements the real-label metrics and is not the headline.

---

## 12. Streaming, export and equivalence

**Streaming reference (`nn/stream.py`, numpy float32):**
- `FeatureState` plus `StreamingNet`.
- Per-layer ring buffers:
  - stem: last 4 input samples × 3 features
  - block d: last 2d frames × C
  - in total (2·255 frames × 16 + 12) ≈ 8.2k floats per channel
- `process(feat_block)` accepts any block length that is a multiple of 4. The host re-blocks to 128.
- Start-up state is all zeros, which matches offline causal zero-padding exactly.

**Export (`pressing-nn-export --ckpt best.pt --out out/nn/<run>/export/`):**
- **`model.onnx`:** opset 17, `torch.onnx.export(dynamo=False)`. Input `feat` (N, 3, T) with dynamic N and T; output `logit` (N, T). Only Conv, ReLU, Add, Transpose and Reshape.
- **`weights.bin` + `weights.json`:** raw little-endian f32 plus a JSON header with layer names, shapes, offsets, `NetConfig`, the feature constants (EPS, FLOOR, NW, asinh scale 8), `D`, and the chosen post-processing. The Rust runtime embeds this with `include_bytes!`.
- **`golden.npz`:** 8 val windows of raw float32 audio, the features, the logits and the masks at the chosen threshold, for the Rust parity test.
- **Export-time check:** ORT vs torch over 20 val windows, max |Δlogit| < 1e-4 (measured 7e-8).

**`test_nn_stream.py`** (random-weight model, so no checkpoint is needed):
- Input is 60 s of concatenated val audio (or noise with injected impulses if no data is present), fed at random block sizes in multiples of 4 from 4 to 2048.
- `StreamingNet ∘ FeatureState` vs offline torch vs ORT: **max |Δlogit| < 1e-4**.
- **Identical masks** at threshold 3.

**Harness-level check:** `pressing-eval --method neural --set stream=1` on val must reproduce the offline counts exactly at every threshold. It is a one-off run of about 5 min.

**Latency (48 kHz).** Defaults: host-side block H = 128, D = 512, merge 8, post 8, LSAR right context 512, max_region 480, max_group 480. Let `e` be the last sample of a region above the extend threshold. The region becomes final, and the samples it needs are all present, when input reaches:
- hysteresis needs logits up to e+8, which needs input up to **e + 8 + D + 3 = e+523**
- contrast gate: b + 151 = e+159
- LSAR: b + 512 = e+520

These overlap, so the region is ready at **e+523**. The earliest sample of a group is at most 480 before its end, so the worst-case wait is 480 + 523 − 8 ≈ 995 samples, plus up to 127 for block quantisation, giving **≈1,122 samples**.

**The engine reports a fixed latency of 1,536 samples (32.0 ms)** and copies unflagged samples from the delay line bit-exactly. The D=1024 variant would need 2,048 (42.7 ms). Both are within the 20-50 ms budget.

---

## 13. Run plan and time budget (M5 Pro)

| item | wall time |
|---|---|
| Implement modules and tests (sections 1-12) | ~1.5 working days |
| `pressing-nn-prep` train, val, test | ~15 min |
| `pressing-nn-synth` val and test | ~10 min |
| Smoke run (2k steps, alignment check, loader throughput ≥ 1,100 crops/s) | ~10 min |
| **R1** C=16, D=512 (main) | ~1.3 h (60 min of steps + 15 val passes) |
| R2 C=24, D=512 (step-up; run regardless, for the size curve) | ~1.8 h |
| R3 C=16, D=256 (latency ablation) | ~1.3 h |
| R4 C=16, D=512, no transplant (checks de-biasing) | ~1.3 h |
| classical-stream re-baseline (val) | ~10 min |
| Neural val selection (about 7 harness runs per candidate) | ~25 min per candidate |
| Streaming harness check (`stream=1`) | ~5 min |
| Test (once) + synthetic test + report | ~15 min |
| Listening audit: 50 random false alarms at the operating point (`--listen`) | ~1 h by ear |

- R1 alone gives a usable answer in about 2.5 h of compute including evaluation.
- R2-R4 run back-to-back overnight (about 4.5 h).
- MPS jobs run sequentially; harness and CPU work can overlap with training.

---

## 14. Expected outcome (estimates, not measurements)

- **Val:** classical is at 74% at 101 FA/min. Neural at matched FA: about 80-86% at 100 FA/min, and about +5-12 points at 50 FA/min.
- **Test:** classical is at 59.9% at 49.7 FA/min. Neural: about 66-75% at about 50 FA/min.
- **Label 2:** about 56% → 65-80%, mainly from transplants and the level-free inputs.
- **Repair:** spike ratio about 1.1-1.5 (same LSAR). Clean audio touched no worse than 0.13%.
- **Floor on measured FA:** a share of measured false alarms are real unmarked clicks. The listening audit estimates that share.

---

## 15. Deferred (not in v1)

| idea | reason |
|---|---|
| Rust runtime, Pi 5 and x86 benchmarks, nih-plug | Next milestone. v1 delivers the weights, JSON, golden vectors and the numpy streaming spec it needs. |
| D=1024, C=8/32, extra dilation-256 block or second stride level | Only if R1-R3 show the model is limited by context or capacity |
| Logistic regression or MLP on classical candidate features (ARC control) | First thing to run if the neural gain is under 5 points |
| ARC candidate re-scorer, U-Net, two heads, StereoMix | Higher risk and more knobs; not needed for v1 |
| AR-residual score as a 4th input | Ablation if the model fails across albums |
| Label-2 soft-positive, teacher phase, round-2 self-labelling, model-based hard-negative re-mining | Confirmation-bias risk; ablate after v1 |
| Platt calibration, gentle/normal/aggressive presets | Product feature, after the operating point is stable |
| Cross-channel mask coupling, boundary snap, sub-sample refinement layer | Only if spike ratio or coverage regress |
| "Harmful FA" triage metric | Partly grades itself with the contrast cue; the listening audit is used instead |
| EQ, rumble, noise augmentation; crackle bursts; label-2 LSAR click bank; labels v1.1 rebuild | Extra pipeline; revisit for domain shift |
| Stateful ONNX step graph, int8 | The runtime is hand-written; offline ONNX is the reference |
| 44.1 / 96 kHz runtime handling | Runtime milestone; decide between resampling the detection path and running at native rate |
| Pops and regions > 480 samples | Out of scope; passed through untouched |

---

## 16. Errors in the proposals fixed here

1. **Normaliser lookahead.** The Nano median gain and the proposal-4 envelope both added lookahead on top of D. Replaced by a causal geometric mean, and latency recomputed (section 12).
2. **Latency arithmetic.** Detector lookahead and LSAR right context overlap rather than add, and `max_group` bounds dense crackle. The result is ≈1,122 samples, reported as a fixed 1,536.
3. **Unmarked clicks trained as negatives** by Nano's `L_event_neg`. Suspected regions are now ignored and excluded from `neg_ok`.
4. **Transplant label threshold.** Positives start at contrast 8, above the ~7× attack range. Weaker transplants are ignored rather than labelled positive.
5. **Template count.** About 230-250k label-1 runs in train (measured sample), not "~1M". The filtered bank is about 150-200k.
6. **`max_region` 4800 → 480 and right context 512.** Classical is re-baselined under the same settings ("classical-stream").
7. **Parameters and MACs** recomputed for the 8-block variant and checked against an instantiated model. C=8 is 584 MAC/sample (Nano's 7-block figure was about 520).
8. **Unsupported claims** ("halves FAs", "3-5× fewer FAs") dropped. Gains are stated as estimates with a pre-registered go/no-go rule.