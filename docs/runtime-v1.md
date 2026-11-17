# 2ndPressing Rust runtime v1: spec

The Rust runtime must reproduce the Python reference pipeline exactly, apart from float rounding. The reference is `NeuralMethod` in `tools/src/pressing_tools/nn/method.py`, which uses `classical.regions_from_scores` and `classical.lsar_fill` from `classical.py`. It streams with a fixed latency, runs in real time on modest CPUs, and builds on x86_64 and aarch64 for Linux, macOS and Windows.

## Workspace (`runtime/`)

| crate | kind | purpose |
|---|---|---|
| `pressing-core` | lib | Model loading, features, streaming network, region finder, LSAR, `Declicker` engine. Its only dependencies are `serde` and `serde_json`. Real-time safe: nothing is allocated in `process`. |
| `pressing-cli` | bin `pressing` | `declick` (files, including the library's ALAC `.m4a`), `bench`, `parity` |
| `pressing-plugin` | cdylib + bin | nih-plug VST3 and CLAP plugin plus standalone app |
| `xtask` | bin | `cargo xtask bundle pressing-plugin --release` (nih-plug bundler) |

The model ships in `runtime/models/clicknet-r2/weights.{json,bin}` (R2, C=24, D=512, threshold 2.75) and is embedded with `include_bytes!`.

Parity fixtures:
- `crates/pressing-core/tests/fixtures/`: 1 window of 48,000 samples, committed.
- `runtime/fixtures-large/`: 8 windows of 96,000 samples, git-ignored. Regenerate with `tools/scripts/export_rust_fixtures.py`. Tests skip if it is absent.

The fixture format is raw little-endian arrays plus `manifest.json`: `audio` f32 (W,2,L), `features` f32 (W,2,3,L), `scores` f32 (W,2,L) (target-aligned logits), `masks` u8, and `repaired` f32 (the full offline pipeline output).

## 1. Features (`features.rs`): `tools/src/pressing_tools/nn/features.py`

Per channel, causal:
- `d1[n] = x[n] − x[n−1]`, with `x[−1] = 0`
- `d2[n] = d1[n] − d1[n−1]`, with `d1[−1] = 0`
- `l_k = ln(|d_k| + EPS)`
- `m_k[n]` = mean of `l_k` over the last `min(n+1, NW)` samples
- `s_k = max(exp(m_k), FLOOR)`
- `g_k = asinh(d_k / s_k / ASINH_SCALE)`

Constants come from `weights.json` → `features`: EPS 1e-6, FLOOR 1e-5, NW 2048, ASINH_SCALE 8.

Model input:
- L uses `[g1_L, g2_L, g1_R]`
- R uses `[g1_R, g2_R, g1_L]`

**Running sums must be f64.** Keep a ring of the last NW `l_k` values (f64, or f32 values with an f64 sum). The streaming reference keeps an exact windowed mean. Expected parity against the fixture features is max |Δ| < 1e-4.

## 2. Network (`net.rs`): `tools/src/pressing_tools/nn/stream.py`

Layout is in `weights.json` → `layout` and `layers` (name, shape, byte offset). Weights are `[out, in, k]` row-major.

Per sequence (one per channel, same weights), frame j covers input samples 4j..4j+3:

```
h0[j] = ReLU(b + Σ_k W[:, :, k] · f[4j − 4 + k])        k = 0..7  (f[<0] = 0)
blocks i = 0..7, d = 2^i:
  u[j]  = ReLU(b_dil + W0·h[j−2d] + W1·h[j−d] + W2·h[j])   (h[<0] = 0)
  h'[j] = ReLU(h[j] + M·u[j] + m)
y[j]  = H2 · ReLU(H1·h[j] + b1) + b2          (4 values)
logit[4j + r] = y[j][r]
```

- `logit[n]` predicts **target sample n − D** (D = 512).
- Each block keeps a ring buffer of its last 2d input frames. Start-up state is all zeros.
- Expected parity: scores within 1e-3 of the fixture (target-aligned: `scores[t] = logit[t + D]`). In practice about 1e-5.

## 3. Region finder (`post.rs`): `classical.regions_from_scores`

One per channel. It consumes target-aligned scores `s[t]` for t = 0, 1, 2, … in order.

- `weak(t) = s[t] > min(threshold − ext_drop, threshold)`
- Weak samples group together while consecutive weak indices differ by ≤ `merge`.
  - A group (first weak `a`, last weak `e`) is **finalised** at the first t with `t − e > merge`.
  - It is **triggered** if `max s[a..=e] > threshold`.
  - It becomes the region `[max(0, a − pre), e + 1 + post)` and is dropped if its length exceeds `max_region`.
  - If `min_contrast > 0`, it is dropped when `contrast(x, a', b', contrast_context) < min_contrast`. Use the same formula as `classical.contrast`, including its edge clamping.
- Regions come out in increasing order. Overlapping or touching regions **union** into one run, exactly like `mask[a:b] = True` followed by `runs(mask)`.
- **Frontier.**
  - `F = min(t_next, a_open) − pre`, where `t_next` is the next unseen target index and `a_open` is the first weak index of the open group.
  - An open group that is already **doomed** does not hold back F. A group is doomed when `(e + 1 + post) − max(0, a − pre) > max_region`; it can only grow, so it can never produce a region.
  - No future region can start before F. A run `[u0, u1)` is **complete** when `u1 < F`.

## 4. LSAR repair (`lsar.rs`): `classical.lsar_fill` (repair settings in the fixture manifest)

Settings: `order p = 32`, `context = 1024`, `context_right = 512`, `max_group = 480`.

**Grouping.** Take complete runs in order. A run joins the current group if `r0 − group_end < 2p` **and** `r1 − group_u0 ≤ max_group`; otherwise it starts a new group. A group is **closed** at the earliest of:
- the next complete run arrives and does not join;
- `F ≥ group_end + 2p`;
- `F > group_u0 + max_group`.

**Solving a closed group `[u0, u1)`.** Skip it if `u0 < p` (stream start).
- `left = y[max(0, u0−context) .. u0]`, which is already repaired.
- `right = y[u1 .. u1 + context_right]`, which is still unrepaired.
- `r = Σ autocorr(piece · hann(len(piece)), p)` over each piece with `len > p`.
  - `hann(N)[n] = 0.5 − 0.5·cos(2πn/(N−1))` (numpy `hanning`).
  - `autocorr[k] = Σ_n v[n]·v[n+k]` for k = 0..p.
- Prediction-error filter `A = [1, a]`:
  - set `r[0] *= 1 + 1e-6`, then solve the Toeplitz system with Levinson;
  - use `A = [1, 0…]` if `r[0] ≤ 1e-20`;
  - fall back to the same if the solve fails.
- Segment `c0 = u0 − p`, `c1 = u1 + p`.
  - Unknowns are the masked samples in `[u0, u1)`. Masked samples in `[c0, u0)` belong to earlier groups and count as known.
  - Rows `i = 0..N−p`: `e_i = Σ_k A[p−k]·y[c0 + i + k]` over the band.
  - Minimise ‖e‖² over the unknowns via the normal equations, solved by Cholesky in f64. numpy uses `lstsq`, but the results agree to float rounding when the system has full rank.
- Write the solution into y.
- **Unmasked samples are never modified (bit-exact).**
- Work in f64 internally. Preallocate for up to `max_group` unknowns and `max_group + 2p` columns.

## 5. Engine (`engine.rs`): `Declicker`

- **Input.** Stereo f32 at any block size, processed sample by sample internally.
- **Pipeline.** Detector → per-channel region finder → runs → LSAR groups. Repair happens in a **delay line** of the working signal: input is written to a ring, repairs are written in place, and output is read `latency` samples behind input.
- **Latency.** Fixed at **1536 samples at 48 kHz**. Worst case from the analysis in section 6 is ~1,486. The `latency_samples()` value is reported to hosts.
- **Deadline guard.** If a group's first sample is about to leave the delay line before it can be solved:
  - solve it with whatever right context exists, provided at least `p` samples are present;
  - otherwise leave it unrepaired;
  - either way, increment `stats.deadline_misses`.

  The tests require 0 misses on the fixtures.
- **Output modes:**
  - `Declick`: the repaired, delayed signal;
  - `ClicksOnly`: delayed input minus repaired output, for monitoring what was removed;
  - `Bypass`: delayed input, with latency preserved.
- **Threshold.** Taken from the model's post config. `set_threshold` changes it at run time, which acts as the sensitivity control.
- **Sample rates.**
  - 44.1/48 kHz: native, run directly (the model was trained at 48 kHz).
  - ≥ 88.2 kHz (**experimental**): `factor = round(sr / 48000)`. Detection runs on a per-channel low-passed, decimated signal (linear-phase FIR, delay compensated). Regions map back ×factor. LSAR runs at the native rate with `order p = 32` and contexts ×factor. Latency becomes `1536·factor` plus the FIR delay.
- **`process_offline(model, sr, l, r)`.** Latency-compensated whole-buffer processing: feed the input, then `latency` zeros, and drop the first `latency` output samples. Used by the CLI and the tests.
- **`ScoreSource` trait.** Lets the tests drive the engine with fixture scores instead of the network. Masks must then match the fixture **exactly** on `[2400, L−2400)`, and repaired samples must agree within 1e-5. With the real network, masks may differ only where a score sits within ~1e-3 of a threshold, so the test reports the number of differing regions and allows ≤ 1% of regions.

## 6. Latency analysis

Assume `p = 32`, `pre = 2`, `post = 8`, `merge = 8`, `D = 512` and stride 4.
- **When a group can close.** A group starting at u0 closes once `F ≥ min(u1 + 2p, u0 + max_group + 1) ≤ u0 + 481`.
- **When F gets there.** F reaches X once `t_next ≥ X + pre` and no non-doomed open group starts before `X + pre`. A non-doomed open group finalises or becomes doomed within `max_region + merge` of its start.
- **Logit timing.** The logit for target t is available when input reaches `t + D + 3`.
- **Worst case.** These combine to about `u0 + 481 + 2 + 480 + 9 + 515 ≈ u0 + 1487`.
- **Right context.** `u1 + context_right ≤ u0 + 992`, which is not the binding constraint.
- **Conclusion.** 1536 is enough, and the deadline guard covers anything missed.

## 7. Performance and portability

- **Budget.** Total network MACs per stereo second are about 472 M at C=24. Target **< 5% of one core** on Apple M-series or modern x86, and **< 25% of a Raspberry Pi 5 core**.
- **Implementation.** Plain Rust; let auto-vectorisation work on contiguous `[f32; C]` arrays, no `unsafe` SIMD in v1.
- **Benchmark.** `pressing bench` reports the real-time factor single-threaded.
- **CI-style checks.** `cargo test`, `cargo clippy -- -D warnings`, and `cargo check` / `cargo build` for these targets:
  - aarch64-apple-darwin and x86_64-apple-darwin
  - x86_64-unknown-linux-gnu and aarch64-unknown-linux-gnu (`check` at least)
  - x86_64-pc-windows-msvc (`check` at least)
