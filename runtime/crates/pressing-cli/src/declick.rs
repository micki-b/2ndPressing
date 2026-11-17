//! `pressing declick`: decode a file, run the offline (latency-compensated)
//! declicker, write a WAV.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};
use pressing_core::{Declicker, Model, OutputMode, Protection, Stats};

use crate::audio::{self, OutFormat};

/// Options of the `declick` subcommand.
pub struct DeclickArgs {
    /// Input file (WAV, AIFF, FLAC, ALAC/AAC .m4a).
    pub input: PathBuf,
    /// Output WAV path.
    pub output: PathBuf,
    /// Trigger threshold (logits); `None` = model default.
    pub threshold: Option<f32>,
    /// Engine output mode.
    pub mode: OutputMode,
    /// Percussion protection mode.
    pub protection: Protection,
    /// Print the list of detected regions.
    pub report: bool,
    /// Write 32-bit float instead of 24-bit integer.
    pub float: bool,
    /// Decode at most this many seconds of input.
    pub max_seconds: Option<f64>,
}

/// One detected region in the output timeline.
struct RegionInfo {
    channel: usize,
    start: usize,
    len: usize,
}

/// Run the `declick` subcommand.
pub fn run(args: &DeclickArgs) -> Result<()> {
    if same_file(&args.input, &args.output) {
        bail!("output path is the input file; refusing to overwrite the source");
    }

    // ---- Decode ----
    let t0 = Instant::now();
    if let Some(s) = args.max_seconds {
        if !(s.is_finite() && s > 0.0) {
            bail!("--max-seconds must be > 0");
        }
    }
    let dec = audio::decode(&args.input, args.max_seconds)?;
    let t_decode = t0.elapsed();
    let frames = dec.frames();
    let rate = dec.sample_rate;
    let nch = dec.channels.len();
    let seconds = frames as f64 / f64::from(rate);
    println!(
        "input:   {} ({}, {} ch, {} Hz, {}{}, {:.2} s{})",
        args.input.display(),
        dec.codec,
        nch,
        rate,
        dec.bits_per_sample
            .map_or_else(|| "?".to_string(), |b| b.to_string()),
        if dec.integer {
            "-bit int"
        } else {
            "-bit float"
        },
        seconds,
        if dec.truncated { ", truncated" } else { "" },
    );
    if let Some(t) = dec.trim.filter(|_| dec.trimmed != (0, 0)) {
        println!(
            "gapless: {} (skip {}, keep {}): dropped {} leading and {} trailing frames",
            t.source,
            t.skip,
            t.keep.map_or_else(|| "all".to_string(), |k| k.to_string()),
            dec.trimmed.0,
            dec.trimmed.1
        );
    }
    if dec.bad_packets > 0 {
        println!(
            "decode:  {} undecodable packets replaced by {} frames of silence",
            dec.bad_packets, dec.concealed_frames
        );
    }
    if rate < 44_100 {
        eprintln!("warning: {rate} Hz is below the model's 44.1/48 kHz design rate");
    } else if rate >= 88_200 {
        eprintln!("note: {rate} Hz uses the experimental decimated detection path");
    }

    // ---- Process ----
    let model = Model::embedded();
    let mut eng = Declicker::new_with(&model, rate as f32, args.protection);
    if let Some(t) = args.threshold {
        if !t.is_finite() {
            bail!("--threshold must be finite");
        }
        eng.set_threshold(t);
    }
    eng.set_mode(args.mode);
    let latency = eng.latency_samples();

    let t1 = Instant::now();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(nch);
    let mut stats = Stats::default();
    let mut regions = Vec::new();
    // Channels are processed in pairs; an odd last channel (or mono) is
    // duplicated to stereo and only its left output is kept.
    let mut c = 0;
    while c < nch {
        let l = &dec.channels[c];
        let r = if c + 1 < nch { &dec.channels[c + 1] } else { l };
        let o = eng.process_offline_with_masks(l, r);
        let s = eng.stats();
        let pair = c + 1 < nch;
        if pair {
            add_stats(&mut stats, s);
        } else {
            // Mono / odd channel: the two copies are processed identically,
            // count one of them.
            stats.regions += s.regions / 2;
            stats.samples_repaired += s.samples_repaired / 2;
            stats.deadline_misses += s.deadline_misses.div_ceil(2);
            stats.solve_failures += s.solve_failures.div_ceil(2);
        }
        if args.report {
            regions.extend(runs(&o.masks[0], c));
            if pair {
                regions.extend(runs(&o.masks[1], c + 1));
            }
        }
        out.push(o.left);
        if pair {
            out.push(o.right);
        }
        c += 2;
    }
    let t_proc = t1.elapsed();
    drop(dec.channels);

    // ---- Write ----
    // Integer sources of up to 24 bits round-trip exactly through f32 and
    // 24-bit output. Wider integer sources (32-bit PCM) are written as float:
    // f32 keeps 24 significant bits at every level, whereas 24-bit integer
    // output would drop the low 8 bits of every sample.
    let wide_int = dec.integer && dec.bits_per_sample.is_some_and(|b| b > 24);
    if wide_int && !args.float {
        eprintln!(
            "note: {}-bit integer source: writing 32-bit float (processing is 32-bit float, so \
             samples above 2^24 LSB are rounded to 24 significant bits)",
            dec.bits_per_sample.unwrap_or(32)
        );
    }
    let fmt = if args.float || !dec.integer || wide_int {
        OutFormat::Float32
    } else {
        OutFormat::Int24
    };
    let t2 = Instant::now();
    let refs: Vec<&[f32]> = out.iter().map(Vec::as_slice).collect();
    let clamped = audio::write_wav(&args.output, &refs, rate, fmt)?;
    let t_write = t2.elapsed();

    // ---- Report ----
    let total = (frames * nch) as f64;
    println!(
        "output:  {} ({} ch, {} Hz, {}, mode {:?})",
        args.output.display(),
        nch,
        rate,
        match fmt {
            OutFormat::Int24 => "24-bit int, no dither",
            OutFormat::Float32 => "32-bit float",
        },
        args.mode
    );
    println!(
        "engine:  threshold {:.3}, percussion protection {:?}, latency {} samples ({:.1} ms, compensated)",
        args.threshold.unwrap_or(model.post.threshold),
        args.protection,
        latency,
        latency as f64 * 1000.0 / f64::from(rate)
    );
    println!(
        "stats:   regions {}, samples repaired {} ({:.4}% of samples), deadline misses {}, solve failures {}",
        stats.regions,
        stats.samples_repaired,
        100.0 * stats.samples_repaired as f64 / total.max(1.0),
        stats.deadline_misses,
        stats.solve_failures
    );
    if seconds > 0.0 {
        println!(
            "rate:    {:.1} regions/min",
            stats.regions as f64 * 60.0 / seconds
        );
    }
    if clamped > 0 {
        println!("clamped: {clamped} samples at 24-bit full scale");
    }
    let proc = t_proc.as_secs_f64();
    println!(
        "time:    decode {:.2} s, process {:.2} s ({:.1}x real time, {:.0} ns/frame), write {:.2} s",
        t_decode.as_secs_f64(),
        proc,
        seconds / proc.max(1e-12),
        proc * 1e9 / (frames.max(1) as f64),
        t_write.as_secs_f64()
    );
    if cfg!(debug_assertions) {
        eprintln!("note: debug build; use --release for meaningful timings");
    }

    if args.report {
        print_report(&regions, rate);
    }
    Ok(())
}

fn add_stats(acc: &mut Stats, s: Stats) {
    acc.regions += s.regions;
    acc.samples_repaired += s.samples_repaired;
    acc.deadline_misses += s.deadline_misses;
    acc.solve_failures += s.solve_failures;
}

/// Runs of 1s in a mask.
fn runs(mask: &[u8], channel: usize) -> Vec<RegionInfo> {
    let mut v = Vec::new();
    let mut i = 0;
    while i < mask.len() {
        if mask[i] != 0 {
            let s = i;
            while i < mask.len() && mask[i] != 0 {
                i += 1;
            }
            v.push(RegionInfo {
                channel,
                start: s,
                len: i - s,
            });
        } else {
            i += 1;
        }
    }
    v
}

fn print_report(regions: &[RegionInfo], rate: u32) {
    let mut sorted: Vec<&RegionInfo> = regions.iter().collect();
    sorted.sort_by_key(|r| (r.start, r.channel));
    println!("report:  {} masked runs (output timeline)", sorted.len());
    if !sorted.is_empty() {
        let mut lens: Vec<usize> = sorted.iter().map(|r| r.len).collect();
        lens.sort_unstable();
        let q = |p: f64| lens[((lens.len() - 1) as f64 * p).round() as usize];
        println!(
            "  run length (samples): min {}, median {}, p90 {}, max {}",
            lens[0],
            q(0.5),
            q(0.9),
            lens[lens.len() - 1]
        );
    }
    println!(
        "  {:>12}  {:>3}  {:>10}  {:>6}",
        "time", "ch", "sample", "len"
    );
    for r in sorted {
        let t = r.start as f64 / f64::from(rate);
        let (m, s) = ((t / 60.0).floor(), t % 60.0);
        println!(
            "  {:>5}:{:06.3}  {:>3}  {:>10}  {:>6}",
            m as u64, s, r.channel, r.start, r.len
        );
    }
}

/// `true` if both paths name the same existing file: same device and inode
/// (or file index on Windows), which also catches hard links and symlinks,
/// with a canonical-path comparison as a fallback.
fn same_file(a: &Path, b: &Path) -> bool {
    if let Ok(same) = same_file::is_same_file(a, b) {
        if same {
            return true;
        }
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}
