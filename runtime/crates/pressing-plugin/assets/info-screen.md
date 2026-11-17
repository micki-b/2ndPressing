# Info screen, layout A: text to edit

Edit any text below; keep the headings (`##`, `###`) so each piece lands in the
right place. Length guides in *italics* are what fits without crowding at the
plugin's 520 × 470 window. The sheet itself scrolls, so its sections can grow;
only the header, readouts and status band have hard limits. Superscript
footnote markers are written as [1] … [5].

## Header band

- Section label (after the "2ndPressing DECLICKER" wordmark): INFO
- Back key: CONTROLS

## Readouts (three amber displays across the top)

*Number: up to 6 characters. Caption: up to about 34 characters, shown in small capitals.*

1. Number: >76%
   Caption: Clicks found · vs 60% classical
2. Number: 0.030 %
   Caption: Clean audio touched · vs 0.131%
3. Number: 43 ×
   Caption: Real time · 2.3% of one core

## Data sheet (scrolling)

### More music. Fewer clicks.

A compact neural network finds clicks and pops, then rebuilds each flagged passage from the music on either side. Everything else passes through untouched, bit for bit.  Built by a music lover with an extensively recorded LP collection who strives to build efficient and simple solutions for sensible people.

### Measured on 24 albums it never trained on [1]

| What matters for your transfer | ClickNet R2 | Classical |
|---|---:|---:|
| Reference clicks found | 76.5 % | 59.9 % |
| False alarms per minute | 27.5 | 49.6 |
| Reference-clean audio touched | 0.030 % | 0.131 % |

### Choose your balance [3]

At 60 % click detection: about **8× fewer false alarms** (6.0 vs 49.9 per minute). At 50 false alarms per minute: **82.4 %** of clicks found vs 60.0 %, a gain of 22.4 points (95 % CI +19.5 to +28.8).

### Loud or quiet passages [4]

*Shown as paired bars (ClickNet R2 above, classical below) with these values.*

| Background loudness | ClickNet R2 | Classical |
|---|---:|---:|
| < −45 dBFS | 88.2 % | 75.8 % |
| −45 … −30 dBFS | 87.5 % | 73.0 % |
| −30 … −18 dBFS | 90.7 % | 73.4 % |
| > −18 dBFS | 90.5 % | 73.9 % |

### Percussion protection

Claps and drums that repeat on a beat are left alone; a scratch that repeats exactly once per rotation (1.8 s at 33⅓, 1.333 s at 45, 0.769 s at 78 RPM) is still repaired. Cost on held-out long-form audio: VinylStudio repairs found 91.0 → 89.4 %.

### Engine [5]

| | |
|---|---|
| High Efficiency Model | 20,116 parameters · only ≈ 80 KB |
| Detector look-ahead | 10.7 ms |
| Latency at 48 kHz (reported to host) | 32 ms · 1.09 s · 2.09 s (protection Off · Standard · Extended) |
| Throughput, one Apple M5 Pro core | 43× real time |
| Unflagged samples [2] | bit-exact |

### Platforms and formats

*Lamp: green = validated, amber = partly checked, off = not yet validated.*

| Lamp | Platform | Status |
|---|---|---|
| green | macOS, Apple Silicon | built and validated |
| amber | macOS x86_64 (Rosetta 2) | parity tested |
| amber | Windows, Linux | compile-checked |
| off | Raspberry Pi 5, other ARM | not yet validated |
| | VST3 · CLAP · Audio Unit (AUv2) · file CLI | 44.1 / 48 kHz |

### Notes

[1] ClickNet R2 at threshold 2.75 vs the classical streaming baseline (threshold 4, contrast gate 16); 10,690 VinylStudio-repaired and 8,680 detected-only events in sampled windows from 24 unseen albums. False alarms touch no label; unlabelled audio can contain genuine clicks. Not a comparison with VinylStudio or commercial products.

[2] Within the engine, after latency; decoding and format conversion can change sample representation.

[3] Interpolated from threshold sweeps; album bootstrap, 2,000 resamples; not the default operating point.

[4] Synthetic clicks with known positions, by background loudness.

[5] Release build, 20 s synthetic stereo, 512-sample blocks, one thread, protection off.

Results vary by recording.

## Status band (bottom strip)

*Each side: up to about 40 characters.*

- Left: CLICKNET R2 · 48 kHz MODEL
- Right: Results vary by recording
