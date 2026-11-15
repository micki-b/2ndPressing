# 2ndPressing runtime (Rust)

Near-real-time LP click and pop removal: the ClickNet R2 detector plus LSAR
repair, ported from the Python reference (`tools/src/pressing_tools/nn/method.py`).
The spec is `docs/runtime-v1.md`.

| crate | what |
|---|---|
| `pressing-core` | Model loading (R2 embedded with `include_bytes!`), features, streaming network, region finder, LSAR, `Declicker` engine. Depends only on `serde`/`serde_json`. No allocation in `process`. |
| `pressing-cli` | `pressing` binary: `declick`, `bench`, `parity` |
| `pressing-plugin` | nih-plug VST3 + CLAP plugin; optional standalone app (`--features standalone`) |
| `xtask` | `cargo xtask bundle pressing-plugin --release` |

## Build

Requires Rust 1.88 or newer (`rust-version` in `Cargo.toml`). Measured with 1.100.0.

```sh
cd runtime
cargo build --release                      # core + CLI + plugin library
cargo xtask bundle pressing-plugin --release   # VST3 + CLAP bundles in target/bundled/
cargo build -p pressing-plugin --release --features standalone   # target/release/pressing-standalone
```

On Linux the standalone app (but not the VST3/CLAP bundle) needs the ALSA and
JACK development packages, for example `libasound2-dev libjack-jackd2-dev` on
Debian/Ubuntu (plus X11 dev headers for the nih-plug standalone).

## Test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace            # debug: about 100 s, most of it the large fixtures
cargo test --workspace --release  # a few seconds
cargo test --workspace --release -- --ignored --nocapture   # micro-bench + CLI large parity
```

Parity fixtures:

- `crates/pressing-core/tests/fixtures/` (1 window of 48,000 samples) is committed.
- `fixtures-large/` (8 windows of 96,000 samples) is git-ignored. Regenerate it with
  `tools/scripts/export_rust_fixtures.py EXPORT_DIR runtime/fixtures-large --windows 8`.
  Tests that need it print `fixtures-large absent; skipping` and pass when it is missing.

What the parity tests check against the Python reference:

| test | compares | measured (large fixture) |
|---|---|---|
| `dsp_parity` | features, streaming logits | features max 5.96e-8 (tol 1e-4); logits max 1.53e-5 (tol 1e-3) |
| `engine_parity` | engine driven by fixture scores: masks, repaired | 0 mask mismatches; repaired max 1.49e-8 (tol 1e-5); 0 deadline misses |
| `network_pipeline` | full pipeline with the real network (`Declicker::new`) | 0 of 378 runs differ (tol 1%); 0 mask samples differ; repaired max 1.49e-8; 0 deadline misses |

Masks and repaired audio are compared on `[2400, L-2400)`, because the offline
reference zero-pads features at the window end. Unmasked samples are checked to
be bit-exact over the full length.

## CLI

```sh
pressing declick IN OUT.wav [--threshold 2.75] [--mode declick|clicks-only|bypass]
                 [--protection off|standard|extended] [--report] [--float]
pressing bench [--seconds 20] [--block 512] [--rate 48000] [--protection off|standard|extended]
pressing parity DIR        # e.g. crates/pressing-core/tests/fixtures or fixtures-large
```

- Input: WAV, AIFF, FLAC, or ALAC/AAC in `.m4a`. AAC encoder priming and padding
  are trimmed using iTunSMPB or the MP4 edit list.
- Output: WAV at the same rate and channel count, latency compensated.
  - Integer sources up to 24 bits are written as 24-bit integer (rounded, no dither).
  - Float sources and integer sources wider than 24 bits are written as 32-bit float.
  - `--float` forces 32-bit float.
- `--threshold`: lower values are more sensitive.
- `--mode clicks-only` writes what was removed.
- `--protection` (default `extended`): percussion protection, see below. The CLI
  compensates for its latency, so the setting only affects the result.
- `--report` lists every detected region.
- The CLI refuses to overwrite its input, including through hard links and symlinks.
- Channels: more than 2 channels are processed as independent stereo pairs; mono
  (or an odd last channel) is processed as duplicated stereo.

## Plugin

Parameters:

- **Bypass**: the host bypass. It crossfades over 10 ms and keeps the latency.
- **Sensitivity**: 0 to 100 % in 5 % steps, shown as just the percentage (hosts'
  generic editors have a narrow value box). 50 % is the model threshold 2.75,
  0 % is threshold 6.0 (gentle) and 100 % is threshold 1.0 (aggressive).
- **Monitor**: `Declick` or `Clicks only`.
- **Percussion protection** (default `Extended (~2 s)`, preferred in listening
  tests on the start of a clapping song): leaves claps, snares and
  hi-hats alone when they sit on a beat grid. A defect that repeats at exactly
  one turntable rotation (1.8 s at 33⅓, 1.333 s at 45, 0.769 s at 78 RPM) is
  still repaired.
  - `Off (32 ms)`: no protection; use it for live monitoring.
  - `Standard (~1 s)`: 1 s look-ahead.
  - `Extended (~2 s)`: 2 s look-ahead; also protects the first second or so of a
    rhythmic passage, at about 1 point of click recall.
  - Changing it changes the reported latency. The plugin switches to a processor
    built for that mode, which starts from silence for the length of its delay.
    During playback nih-plug asks the host to restart the plugin and reports the
    new latency on re-activation (in Logic, via the Audio Unit wrapper).

  Measured in Python on 54 held-out minutes (Standard): VinylStudio repairs
  found 91.0 → 89.4 %, real click shapes injected at random times 88.0 → 87.9 %;
  in the clapping song of the test sample, 290 → 55 regions repaired. The Rust
  engine matches the Python reference exactly on all parity fixtures.

Copy the bundles from `target/bundled/` to:

| OS | VST3 | CLAP |
|---|---|---|
| macOS | `~/Library/Audio/Plug-Ins/VST3/` (or `/Library/...`) | `~/Library/Audio/Plug-Ins/CLAP/` (or `/Library/...`) |
| Windows | `C:\Program Files\Common Files\VST3\` | `C:\Program Files\Common Files\CLAP\` |
| Linux | `~/.vst3/` (or `/usr/lib/vst3/`) | `~/.clap/` (or `/usr/lib/clap/`) |

The macOS bundles are ad-hoc signed and thin (arm64 when built on Apple Silicon).
For a universal binary, build with nih-plug's `bundle-universal`, which is untested here.
Only the macOS arm64 bundle has been built and validated. On Windows and Linux,
`cargo check` passes, but no bundle has been built or loaded in a host.

## Audio Unit (Logic Pro, GarageBand, other AU hosts)

Logic only loads Audio Units, so `au/` builds an AUv2 wrapper with
[clap-wrapper](https://github.com/free-audio/clap-wrapper) (0.17.0, MIT). The
`.component` is a thin shim that loads `2ndPressing Declicker.clap` at runtime,
so **both** must be installed. Requires CMake (`brew install cmake`).

```sh
cargo xtask bundle pressing-plugin --release
cmake -S au -B au/build -DCMAKE_BUILD_TYPE=Release && cmake --build au/build
cp -R "target/bundled/2ndPressing Declicker.clap"   ~/Library/Audio/Plug-Ins/CLAP/
cp -R "au/build/2ndPressing Declicker.component"    ~/Library/Audio/Plug-Ins/Components/
killall -9 AudioComponentRegistrar; auval -v aufx Dclk SnPr   # expect: AU VALIDATION SUCCEEDED
```

Type `aufx`, subtype `Dclk`, manufacturer `SnPr`. After rebuilding the CLAP,
copy it again; the component does not need rebuilding.

## Latency

- **44.1 and 48 kHz:** 1536 samples, about 32 ms at 48 kHz, with percussion
  protection off. Standard adds `ceil(1.018·sr) + 2048` (52,448 samples in
  total at 48 kHz, 1.09 s); Extended adds `ceil(2.018·sr) + 2048` (100,448,
  2.09 s). The plugin reports it to the host and the CLI compensates for it.
- **88.2 kHz and above (experimental):** `1536*factor + 32*factor`, where `factor = round(sr/48000)`.
  At 88.2 and 96 kHz that is 3136 samples.
- **Deadline misses:** a group that cannot be solved before it leaves the delay
  line is counted in `Stats::deadline_misses`. None occur on the fixtures.

## Measured performance

These numbers are from a release build running single-threaded on an Apple M5 Pro,
using `pressing bench` (20 s of synthetic stereo, block 512):

| rate | engine ns / stereo sample | x real time | % of one core | of which: network | features |
|---|---|---|---|---|---|
| 44.1 kHz | 482 | 47x | 2.1 % | 412 ns | 54 ns |
| 48 kHz | 489 | 43x | 2.3 % | 415 ns | 55 ns |
| 96 kHz (decimated detection) | 317 per 96 kHz sample | 33x | 3.0 % | | |
| 48 kHz, protection Standard | 525 | 40x | 2.5 % | | |
| 48 kHz, protection Extended | 528 | 40x | 2.5 % | | |

- **LSAR solve:** about 26 us for an 8-sample gap and about 208 us for a 480-sample
  gap. At the measured click rate that is about 0.02 % of a core.
- **x86_64 (Rosetta 2 on the same Mac):** the engine took 587 ns per stereo sample
  (2.8 % of a core) and parity passed. Emulated x86 timing is not representative
  of native x86 hardware.
- **Not measured:** native x86, Linux, Windows and the Raspberry Pi 5. The spec's
  Pi 5 target is under 25 % of a core.

## Limitations

- **Sample rates.** The model was trained at 48 kHz. 44.1 kHz runs natively without
  resampling.
  - The 88.2 kHz and above path, which decimates for detection, has only been tested
    for timing and mapping. Its detection quality is unvalidated.
  - Other rates, such as 22.05, 32 or 64 kHz, run without any adaptation. The plugin
    logs a warning but nothing appears in its UI.
- **Long chained runs.** A run of chained regions longer than about 1019 samples at
  48 kHz (latency minus 517) cannot be solved in one piece within the latency.
  - Its final prefix is extrapolated from left context, which counts as a deadline
    miss.
  - Here the output differs from the Python reference, which solves the run in one
    piece.
- **CLI memory.** The CLI decodes the whole file into memory. A 25-minute side at
  96 kHz stereo needs roughly 2.3 GB.
- **CLI pipes.** Piping `pressing` output into `head` ends with a broken-pipe panic.
- **AAC edit lists.** Only single-edit lists are applied; other edit-list layouts
  are ignored with a warning.
- **Percussion protection.** Only the rhythm (beat-grid) cue is used. Percussion
  that is not regular, and the first ~1 s of a rhythmic passage in Standard mode,
  can still be repaired. Tuned and verified at 48 kHz; at other rates the same
  time constants are used.
- **clap-validator, latency changes.** With the protection parameter, three tests
  (`param-set-events`, `param-set-no-cookies`, `param-fuzz-sample-accurate`)
  report `clap_host_latency::changed` called outside `activate`. nih-plug requests
  a restart while the host is processing but calls `latency changed` directly
  when the plugin is active and not processing (a documented nih-plug
  simplification). Hosts generally accept this; the Audio Unit route restarts
  the plugin and reports the latency inside activation.
- **clap-validator, state.** Before the protection parameter: 30 of 44 tests
  passed, 4 failed and 10 were skipped.
  - The failures are `state-invalid-random` (it crashes in nih-plug's state loader
    on corrupt data) and three `state-reproducibility-*` tests.
  - nih-plug's own `gain` example, built at the pinned revision, fails the same
    four tests in the same way (checked independently twice), so they come from
    nih-plug and this validator version, not from this plugin. Restoring sessions
    in a real DAW has not been tested yet.
- **Standalone app.** It does not compensate for latency (an upstream nih-plug
  limitation).
- **Allocation checks.** "No allocation in `process`" is enforced by
  `assert_no_alloc` tests in debug builds only. In release those tests show as
  ignored.
