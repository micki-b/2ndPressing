//! `pressing bench`: single-thread real-time factor of the engine and its
//! parts on synthetic music-like material with injected clicks.

use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use pressing_core::features::FeatureState;
use pressing_core::lsar::{LsarConfig, LsarSolver};
use pressing_core::net::{Detector, StreamingNet};
use pressing_core::{Declicker, Model, Protection, ScoreSource};

use crate::synth::{music_with_clicks, Rng};

/// Options of the `bench` subcommand.
pub struct BenchArgs {
    /// Seconds of audio processed in each timed run.
    pub seconds: f64,
    /// Block size passed to `Declicker::process`.
    pub block: usize,
    /// Sample rate.
    pub rate: u32,
    /// Percussion protection mode of the engine run.
    pub protection: Protection,
}

/// Lowest `--rate` accepted (below it the synthetic material is meaningless).
const MIN_RATE: u32 = 8_000;
/// Highest `--rate` accepted.
const MAX_RATE: u32 = 768_000;

/// Prints one timing line; `frames` are stereo frames at `rate`.
fn line(name: &str, elapsed: Duration, frames: usize, rate: f64) {
    let s = elapsed.as_secs_f64();
    let ns = s * 1e9 / frames.max(1) as f64;
    let audio_s = frames as f64 / rate;
    let rt = audio_s / s.max(1e-12);
    println!(
        "{name:<10} {ns:9.1} ns/stereo sample  {rt:9.1}x real time  {:7.3}% of one core",
        100.0 / rt
    );
}

/// Run the `bench` subcommand.
pub fn run(args: &BenchArgs) -> Result<()> {
    if !(args.seconds.is_finite() && args.seconds > 0.0) {
        bail!("--seconds must be > 0");
    }
    if args.block == 0 {
        bail!("--block must be > 0");
    }
    if !(MIN_RATE..=MAX_RATE).contains(&args.rate) {
        bail!("--rate must be between {MIN_RATE} and {MAX_RATE} Hz");
    }
    let rate = f64::from(args.rate);
    let model = Model::embedded();
    let sig = music_with_clicks(args.seconds, args.rate, 4.0, 0x2bd_9e55);
    let n = sig.left.len();
    println!(
        "bench: {:.1} s synthetic stereo at {} Hz, block {}, {} injected clicks, single thread{}",
        args.seconds,
        args.rate,
        args.block,
        sig.clicks.len(),
        if cfg!(debug_assertions) {
            " (DEBUG BUILD: use --release)"
        } else {
            ""
        }
    );

    // ---- Engine ----
    let mut eng = Declicker::new_with(&model, args.rate as f32, args.protection);
    let (mut l, mut r) = (sig.left.clone(), sig.right.clone());
    // Warm-up (caches, branch predictors) on up to 1 s, then reset.
    let warm = n.min(args.rate as usize);
    for (bl, br) in l[..warm]
        .chunks_mut(args.block)
        .zip(r[..warm].chunks_mut(args.block))
    {
        eng.process(bl, br);
    }
    eng.reset();
    let (mut l, mut r) = (sig.left.clone(), sig.right.clone());
    let t0 = Instant::now();
    for (bl, br) in l.chunks_mut(args.block).zip(r.chunks_mut(args.block)) {
        eng.process(bl, br);
    }
    let el = t0.elapsed();
    black_box((&l, &r));
    line("engine", el, n, rate);
    let st = eng.stats();
    println!(
        "           latency {} samples, regions {}, samples repaired {}, deadline misses {}, solve failures {}",
        eng.latency_samples(),
        st.regions,
        st.samples_repaired,
        st.deadline_misses,
        st.solve_failures
    );

    // ---- Detector parts (run on every sample at the detection rate) ----
    let factor = pressing_core::engine::rate_factor(args.rate as f32);
    if factor > 1 {
        println!(
            "           (detector below runs on the native {} Hz stream; the engine decimates by {factor})",
            args.rate
        );
    }
    let mut fs = FeatureState::new(&model.features);
    let t0 = Instant::now();
    let feats: Vec<[[f32; 3]; 2]> = sig
        .left
        .iter()
        .zip(&sig.right)
        .map(|(&a, &b)| fs.push(a, b))
        .collect();
    line("features", t0.elapsed(), n, rate);

    let mut net = StreamingNet::new(&model);
    let mut acc = 0.0f32;
    let t0 = Instant::now();
    for f in &feats {
        if let Some(y) = net.push(*f) {
            acc += y[0][0] + y[1][3];
        }
    }
    line("network", t0.elapsed(), n, rate);
    black_box(acc);

    let mut det = Detector::new(&model);
    let t0 = Instant::now();
    for (&a, &b) in sig.left.iter().zip(&sig.right) {
        if let Some(y) = det.push(a, b) {
            acc += y[0][1];
        }
    }
    line("detector", t0.elapsed(), n, rate);
    black_box(acc);

    // ---- LSAR ----
    bench_lsar(st.regions as f64 / args.seconds);
    Ok(())
}

/// Times single LSAR solves on an AR(2)-like signal for several gap lengths.
fn bench_lsar(runs_per_sec: f64) {
    let cfg = LsarConfig::default();
    let p = cfg.order;
    let mut solver = LsarSolver::new(&cfg);
    let mut rng = Rng(7);
    let mut sizes = vec![8usize, 24, 96];
    sizes.push(cfg.max_group);
    println!(
        "lsar:      order {p}, context {}/{} (left/right)",
        cfg.context, cfg.context_right
    );
    for gap in sizes {
        let u0 = cfg.context;
        let u1 = u0 + gap;
        let len = u1 + cfg.context_right.max(p);
        // Resonant AR(2) signal.
        let mut y = vec![0.0f64; len];
        for i in 2..len {
            y[i] = 1.8 * y[i - 1] - 0.9 * y[i - 2] + 0.01 * rng.range(-1.0, 1.0);
        }
        let mut unknown = vec![false; len];
        unknown[u0..u1].fill(true);
        let base = y.clone();
        // Repeat for ~0.25 s (at least 3 solves); the copy is included but
        // is negligible next to the solve.
        let mut ok = true;
        let mut reps = 0u32;
        let t0 = Instant::now();
        while reps < 3 || (t0.elapsed() < Duration::from_millis(250) && reps < 100_000) {
            y.copy_from_slice(&base);
            ok &= solver.try_fill(&mut y, &unknown, u0, u1);
            reps += 1;
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        black_box(&y);
        println!(
            "           gap {gap:>4}: {us:9.2} us/solve{}",
            if ok { "" } else { "  (solve FAILED)" }
        );
        if gap == 24 && runs_per_sec > 0.0 {
            println!(
                "           at the measured {runs_per_sec:.1} runs/s with 24-sample gaps: {:.4}% of one core",
                runs_per_sec * us * 1e-4
            );
        }
    }
}
