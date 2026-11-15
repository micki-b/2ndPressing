//! Engine parity against the Python reference fixtures, plus unit tests of the
//! region finder, the LSAR solver and the engine's latency/modes.
//!
//! The engine is driven by `FixtureScores` (fixture logits replayed with the
//! model delay D) so these tests do not depend on the network.

use pressing_core::engine::LATENCY_48K;
use pressing_core::fixtures::Fixtures;
use pressing_core::lsar::{add_windowed_autocorr, ar_from_autocorr, LsarConfig, LsarSolver};
use pressing_core::post::{contrast, RegionFinder};
use pressing_core::{
    Declicker, FeatureConfig, Model, NetConfig, OutputMode, PostConfig, ScoreSource,
};

// ---- Helpers ----

/// Score that can never trigger or extend a region.
const QUIET: f32 = -100.0;

/// Replays target-aligned fixture scores as a streaming source with delay D:
/// `logit[n] = scores[n - D]` for `D <= n < D + L`, `QUIET` otherwise, emitted
/// in frames of 4 on pushes `n = 4j + 3` exactly like the network.
struct FixtureScores {
    scores: [Vec<f32>; 2],
    delay: usize,
    n: usize,
}

impl FixtureScores {
    fn new(left: Vec<f32>, right: Vec<f32>, delay: usize) -> Self {
        FixtureScores {
            scores: [left, right],
            delay,
            n: 0,
        }
    }

    fn logit(&self, c: usize, m: usize) -> f32 {
        m.checked_sub(self.delay)
            .and_then(|t| self.scores[c].get(t).copied())
            .unwrap_or(QUIET)
    }
}

impl ScoreSource for FixtureScores {
    fn push(&mut self, _l: f32, _r: f32) -> Option<[[f32; 4]; 2]> {
        let n = self.n;
        self.n += 1;
        if n % 4 != 3 {
            return None;
        }
        let mut out = [[0.0; 4]; 2];
        for (c, o) in out.iter_mut().enumerate() {
            for (k, v) in o.iter_mut().enumerate() {
                *v = self.logit(c, n - 3 + k);
            }
        }
        Some(out)
    }
    fn delay(&self) -> usize {
        self.delay
    }
    fn reset(&mut self) {
        self.n = 0;
    }
}

fn post_from(v: &serde_json::Value) -> PostConfig {
    serde_json::from_value(v.clone()).expect("post config")
}

/// A model carrying only configs (no tensors): enough for score-source engines.
fn config_model(post: PostConfig) -> Model {
    Model {
        net: NetConfig {
            c: 24,
            blocks: 8,
            stem_k: 8,
            stride: 4,
            d: 512,
        },
        features: FeatureConfig {
            eps: 1e-6,
            floor: 1e-5,
            nw: 2048,
            asinh_scale: 8.0,
        },
        post,
        tensors: Vec::new(),
    }
}

fn default_post() -> PostConfig {
    PostConfig {
        threshold: 2.75,
        ext_drop: 2.0,
        merge: 8,
        pre: 2,
        post: 8,
        max_region: 480,
        min_contrast: 0.0,
        contrast_context: 150,
    }
}

/// splitmix64: deterministic test randomness without dependencies.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---- Fixture parity ----

struct ParityReport {
    windows: usize,
    mask_mismatches: usize,
    max_abs_err: f64,
    regions: u64,
    repaired: u64,
}

fn run_fixture_parity(fx: &Fixtures) -> ParityReport {
    let audio = fx.f32s("audio");
    let scores = fx.f32s("scores");
    let masks = fx.u8s("masks");
    let reference = fx.f32s("repaired");
    let post = post_from(&fx.manifest["post"]);
    let d = fx.manifest["D"].as_u64().expect("D") as usize;
    let edge = fx.manifest["edge_ignore"].as_u64().unwrap_or(2400) as usize;
    let rep = &fx.manifest["repair"];
    assert_eq!(
        LsarConfig {
            order: rep["order"].as_u64().unwrap() as usize,
            context: rep["context"].as_u64().unwrap() as usize,
            context_right: rep["context_right"].as_u64().unwrap() as usize,
            max_group: rep["max_group"].as_u64().unwrap() as usize,
        },
        LsarConfig::default(),
        "engine uses the default repair settings"
    );
    let model = config_model(post);
    let l = fx.samples;
    let mut report = ParityReport {
        windows: fx.windows,
        mask_mismatches: 0,
        max_abs_err: 0.0,
        regions: 0,
        repaired: 0,
    };

    for w in 0..fx.windows {
        let ch = |a: &[f32], c: usize| a[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l].to_vec();
        let src = FixtureScores::new(ch(&scores, 0), ch(&scores, 1), d);
        let mut eng = Declicker::with_score_source(&model, Box::new(src));
        assert_eq!(eng.latency_samples(), LATENCY_48K);
        let (xl, xr) = (ch(&audio, 0), ch(&audio, 1));
        let out = eng.process_offline_with_masks(&xl, &xr);
        let stats = eng.stats();
        assert_eq!(stats.deadline_misses, 0, "window {w}: deadline misses");
        report.regions += stats.regions;
        report.repaired += stats.samples_repaired;

        for (c, (y, x)) in [(&out.left, &xl), (&out.right, &xr)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(y.len(), l);
            let m_ref = &masks[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l];
            let r_ref = &reference[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l];
            for i in 0..l {
                // Unmasked samples are never modified (bit-exact), anywhere.
                if out.masks[c][i] == 0 {
                    assert_eq!(
                        y[i].to_bits(),
                        x[i].to_bits(),
                        "w{w} c{c} unmasked sample {i} modified"
                    );
                }
                if i < edge || i >= l - edge {
                    continue;
                }
                if out.masks[c][i] != m_ref[i] {
                    report.mask_mismatches += 1;
                }
                let err = (f64::from(y[i]) - f64::from(r_ref[i])).abs();
                report.max_abs_err = report.max_abs_err.max(err);
                assert!(
                    err <= 1e-5,
                    "w{w} c{c} sample {i}: {} vs reference {}",
                    y[i],
                    r_ref[i]
                );
            }
        }
    }
    report
}

fn check_parity(fx: &Fixtures, name: &str) {
    let r = run_fixture_parity(fx);
    println!(
        "{name}: {} windows, {} runs, {} samples repaired, mask mismatches {}, max |err| {:.3e}",
        r.windows, r.regions, r.repaired, r.mask_mismatches, r.max_abs_err
    );
    assert_eq!(r.mask_mismatches, 0, "{name}: masks must match exactly");
    assert!(r.regions > 0, "{name}: fixture should contain clicks");
}

#[test]
fn parity_small_fixture() {
    let fx = Fixtures::small().expect("committed small fixtures");
    check_parity(&fx, "small");
}

#[test]
fn parity_large_fixture() {
    let Some(fx) = Fixtures::large() else {
        eprintln!("fixtures-large absent; skipping");
        return;
    };
    check_parity(&fx, "large");
}

#[test]
fn clicks_only_is_delayed_input_minus_declick() {
    let fx = Fixtures::small().expect("committed small fixtures");
    let (audio, scores) = (fx.f32s("audio"), fx.f32s("scores"));
    let l = fx.samples;
    let d = fx.manifest["D"].as_u64().unwrap() as usize;
    let model = config_model(post_from(&fx.manifest["post"]));
    let run = |mode: OutputMode| {
        let src = FixtureScores::new(scores[..l].to_vec(), scores[l..2 * l].to_vec(), d);
        let mut eng = Declicker::with_score_source(&model, Box::new(src));
        eng.set_mode(mode);
        eng.process_offline(&audio[..l], &audio[l..2 * l])
    };
    let (dl, dr) = run(OutputMode::Declick);
    let (cl, cr) = run(OutputMode::ClicksOnly);
    let (bl, br) = run(OutputMode::Bypass);
    let mut nonzero = 0;
    for i in 0..l {
        assert_eq!(cl[i].to_bits(), (audio[i] - dl[i]).to_bits());
        assert_eq!(cr[i].to_bits(), (audio[l + i] - dr[i]).to_bits());
        assert_eq!(bl[i].to_bits(), audio[i].to_bits());
        assert_eq!(br[i].to_bits(), audio[l + i].to_bits());
        nonzero += usize::from(cl[i] != 0.0) + usize::from(cr[i] != 0.0);
    }
    assert!(nonzero > 0, "something was removed");
}

#[test]
fn random_blocks_match_offline() {
    let fx = Fixtures::small().expect("committed small fixtures");
    let (audio, scores) = (fx.f32s("audio"), fx.f32s("scores"));
    let l = fx.samples;
    let d = fx.manifest["D"].as_u64().unwrap() as usize;
    let model = config_model(post_from(&fx.manifest["post"]));
    let src = || {
        Box::new(FixtureScores::new(
            scores[..l].to_vec(),
            scores[l..2 * l].to_vec(),
            d,
        ))
    };

    let mut eng = Declicker::with_score_source(&model, src());
    let (ol, or) = eng.process_offline(&audio[..l], &audio[l..2 * l]);
    let lat = eng.latency_samples();

    let mut eng = Declicker::with_score_source(&model, src());
    let mut left: Vec<f32> = audio[..l]
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0.0, lat))
        .collect();
    let mut right: Vec<f32> = audio[l..2 * l]
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0.0, lat))
        .collect();
    let mut rng = Rng(7);
    let mut i = 0;
    while i < left.len() {
        let b = [1, 3, 64, 511, 512, 1000, 4096][rng.below(7)].min(left.len() - i);
        let b = if rng.below(4) == 0 {
            rng.below(b) + 1
        } else {
            b
        };
        eng.process(&mut left[i..i + b], &mut right[i..i + b]);
        i += b;
    }
    for k in 0..l {
        assert_eq!(left[lat + k].to_bits(), ol[k].to_bits(), "left {k}");
        assert_eq!(right[lat + k].to_bits(), or[k].to_bits(), "right {k}");
    }
    assert_eq!(eng.stats().deadline_misses, 0);
}

/// A source that never detects anything.
struct Silent(usize);

impl ScoreSource for Silent {
    fn push(&mut self, _l: f32, _r: f32) -> Option<[[f32; 4]; 2]> {
        self.0 += 1;
        self.0.is_multiple_of(4).then_some([[QUIET; 4]; 2])
    }
    fn delay(&self) -> usize {
        512
    }
    fn reset(&mut self) {
        self.0 = 0;
    }
}

#[test]
fn bypass_impulse_latency() {
    let model = config_model(default_post());
    for mode in [OutputMode::Bypass, OutputMode::Declick] {
        let mut eng = Declicker::with_score_source(&model, Box::new(Silent(0)));
        eng.set_mode(mode);
        let lat = eng.latency_samples();
        assert_eq!(lat, 1536);
        let k = 777;
        let mut l = vec![0.0f32; k + lat + 100];
        let mut r = l.clone();
        l[k] = 1.0;
        r[k] = -0.5;
        eng.process(&mut l, &mut r);
        for (i, (&a, &b)) in l.iter().zip(&r).enumerate() {
            let (ea, eb) = if i == k + lat {
                (1.0, -0.5)
            } else {
                (0.0, 0.0)
            };
            assert_eq!((a, b), (ea, eb), "mode {mode:?} sample {i}");
        }
    }
}

#[test]
fn high_rate_path_latency_and_bypass() {
    let model = config_model(default_post());
    let mut eng = Declicker::with_score_source_at(&model, 96_000.0, Box::new(Silent(0)));
    let lat = eng.latency_samples();
    assert_eq!(lat, 1536 * 2 + 64);
    let mut rng = Rng(3);
    let x: Vec<f32> = (0..20_000).map(|_| rng.normal() as f32 * 0.1).collect();
    let (yl, yr) = eng.process_offline(&x, &x);
    assert_eq!(yl, x, "nothing detected: output equals input");
    assert_eq!(yr, x);
}

/// Noise plus sharp clicks at `clicks`; scores spike at every position in
/// `clicks` and `soft` (the latter sit on smooth bursts: low contrast).
fn synthetic_case(
    rng: &mut Rng,
    n: usize,
    clicks: &[usize],
    soft: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut x: Vec<f32> = (0..n)
        .map(|i| (0.2 * (i as f64 * 0.013).sin() + 0.002 * rng.normal()) as f32)
        .collect();
    let mut s = vec![-3.0f32; n];
    for &k in clicks {
        x[k] += 0.5;
        x[k + 1] -= 0.3;
        s[k..k + 3].copy_from_slice(&[4.0, 3.0, 1.0]);
    }
    for &k in soft {
        for (j, v) in x[k - 40..k + 40].iter_mut().enumerate() {
            *v += 0.05 * (j as f32 * 0.3).sin();
        }
        s[k..k + 3].copy_from_slice(&[4.0, 3.0, 1.0]);
    }
    (x, s)
}

#[test]
fn contrast_gate_matches_reference() {
    let post = PostConfig {
        min_contrast: 16.0,
        ..default_post()
    };
    let model = config_model(post.clone());
    let mut rng = Rng(9);
    let n = 30_000;
    let clicks = [3000, 3020, 7000, 12_345, 20_000];
    let soft = [5000, 9000, 15_000, 25_000];
    let (x, s) = synthetic_case(&mut rng, n, &clicks, &soft);
    let src = FixtureScores::new(s.clone(), s.clone(), 512);
    let mut eng = Declicker::with_score_source(&model, Box::new(src));
    let out = eng.process_offline_with_masks(&x, &x);
    let want = regions_from_scores_ref(&s, &post, Some(&x));
    assert_eq!(out.masks[0], want);
    assert_eq!(out.masks[1], want);
    // The gate kept every click and rejected every soft burst.
    assert!(clicks.iter().all(|&k| want[k] == 1));
    assert!(soft.iter().all(|&k| want[k] == 0));
    assert_eq!(eng.stats().deadline_misses, 0);
}

#[test]
fn high_rate_path_detects_and_repairs() {
    // 96 kHz: the source sees the decimated signal; a score spike at
    // detection target t must give the native region [2(t - pre), 2(t + 1 + post)).
    let model = config_model(default_post());
    let mut rng = Rng(21);
    let n = 40_000;
    let targets = [4000usize, 9000, 15_000];
    let mut scores = vec![-3.0f32; n / 2];
    for &t in &targets {
        scores[t] = 5.0;
    }
    let x: Vec<f32> = (0..n)
        .map(|i| (0.3 * (i as f64 * 0.02).sin() + 0.001 * rng.normal()) as f32)
        .collect();
    let mut xc = x.clone();
    for &t in &targets {
        xc[2 * t] += 0.4;
    }
    let src = FixtureScores::new(scores.clone(), scores, 512);
    let mut eng = Declicker::with_score_source_at(&model, 96_000.0, Box::new(src));
    let out = eng.process_offline_with_masks(&xc, &xc);
    let mut want = vec![0u8; n];
    for &t in &targets {
        want[2 * (t - 2)..2 * (t + 9)].fill(1);
    }
    assert_eq!(out.masks[0], want);
    assert_eq!(eng.stats().deadline_misses, 0);
    assert_eq!(eng.stats().regions, 2 * targets.len() as u64);
    let max_err = (0..n)
        .map(|i| (out.left[i] - x[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.01, "repair error {max_err}");
}

/// Scores for `groups` weak groups (469 weak samples each, one trigger) chained
/// so their padded regions overlap: one run of about `479 * groups` samples
/// starting at `start - 2`. Returns (clean, clicked, scores).
fn chained_run_case(start: usize, groups: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(77 + groups as u64);
    let tau = 2.0 * std::f64::consts::PI;
    let clean: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / 48_000.0;
            (0.2 * (tau * 440.0 * t).sin()
                + 0.1 * (tau * 1250.0 * t + 1.0).sin()
                + 0.001 * rng.normal()) as f32
        })
        .collect();
    let mut s = vec![-3.0f32; n];
    let mut a = start;
    for _ in 0..groups {
        let e = a + 468;
        s[a..=e].fill(1.0);
        s[a + 200] = 4.0;
        a = e + 10; // gap of 9 > merge: separate groups, overlapping regions
    }
    let run_end = a - 10 + 9;
    let mut x = clean.clone();
    for k in (start..run_end).step_by(37) {
        x[k] += 0.5;
        x[k + 1] -= 0.35;
    }
    (clean, x, s)
}

#[test]
fn long_chained_runs_are_repaired() {
    // A run longer than the latency can cover (see the engine module docs):
    // before the fix the deadline guard abandoned the whole run (max error
    // ~0.5 per click); now the final prefix is extrapolated at the deadline
    // and the rest is solved as its own run.
    let model = config_model(default_post());
    for groups in [1usize, 2, 3, 4, 6] {
        let start = 5000;
        let n = start + 479 * groups + 4000;
        let (clean, x, s) = chained_run_case(start, groups, n);
        let want = regions_from_scores_ref(&s, &default_post(), None);
        let run: Vec<usize> = (0..n).filter(|&i| want[i] == 1).collect();
        assert_eq!(
            run.len(),
            run[run.len() - 1] - run[0] + 1,
            "one contiguous run"
        );
        let src = FixtureScores::new(s.clone(), s.clone(), 512);
        let mut eng = Declicker::with_score_source(&model, Box::new(src));
        let out = eng.process_offline_with_masks(&x, &x);
        let st = eng.stats();
        assert_eq!(out.masks[0], want, "groups {groups}: mask");
        assert_eq!(st.solve_failures, 0, "groups {groups}");
        let mut max_err = 0.0f32;
        for i in 0..n {
            if want[i] == 0 {
                assert_eq!(out.left[i].to_bits(), x[i].to_bits(), "unmasked {i}");
            } else {
                max_err = max_err.max((out.left[i] - clean[i]).abs());
            }
        }
        // What `classical.lsar_fill` does: one solve over the whole run.
        let (u0, u1) = (run[0], run[run.len() - 1] + 1);
        let mut y: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let unknown: Vec<bool> = want.iter().map(|&m| m == 1).collect();
        let mut solver = LsarSolver::with_max_span(&LsarConfig::default(), u1 - u0);
        assert!(solver.try_fill(&mut y, &unknown, u0, u1));
        let one_piece_err = (u0..u1)
            .map(|i| (y[i] as f32 - clean[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "chained run of {} samples ({groups} regions): misses {}, max |err| vs clean \
             {max_err:.3e} (one-piece reference solve {one_piece_err:.3e})",
            run.len(),
            st.deadline_misses
        );
        assert_eq!(st.samples_repaired, 2 * run.len() as u64, "groups {groups}");
        if run.len() < LATENCY_48K - 517 {
            // In time: exactly the reference's one-piece solve.
            assert_eq!(st.deadline_misses, 0, "groups {groups}");
            for i in u0..u1 {
                let d = (f64::from(out.left[i]) - y[i]).abs();
                assert!(d <= 1e-5, "groups {groups} sample {i}: {d}");
            }
        } else {
            // Too long for the latency: repaired in pieces (one miss per
            // channel per piece), close to the one-piece solve. Unrepaired,
            // the 0.5 clicks would remain.
            assert!(st.deadline_misses > 0, "groups {groups}: guard must engage");
            assert!(
                max_err <= 2.0 * one_piece_err + 0.01 && max_err < 0.25,
                "groups {groups}: repair error {max_err} vs one-piece {one_piece_err}"
            );
        }
    }
}

#[test]
fn non_finite_input_does_not_block_repair() {
    // NaN/inf near a click: masked samples (including a NaN click sample) are
    // repaired, unmasked non-finite samples pass through bit-exactly, and no
    // solve fails (non-finite context is read as 0 inside the solve).
    let model = config_model(default_post());
    let mut rng = Rng(5);
    let n = 20_000;
    let (mut x, s) = synthetic_case(&mut rng, n, &[8000, 12_000], &[]);
    x[7900] = f32::NAN; // left context, unmasked
    x[8001] = f32::NAN; // inside the region [7998, 8011)
    x[8030] = f32::INFINITY; // right context, unmasked
    x[11_990] = f32::NEG_INFINITY; // left context of the second click, unmasked
    let want = regions_from_scores_ref(&s, &default_post(), None);
    let src = FixtureScores::new(s.clone(), s, 512);
    let mut eng = Declicker::with_score_source(&model, Box::new(src));
    let out = eng.process_offline_with_masks(&x, &x);
    let st = eng.stats();
    assert_eq!(out.masks[0], want);
    assert_eq!(st.solve_failures, 0);
    assert_eq!(st.deadline_misses, 0);
    assert_eq!(st.samples_repaired, 2 * 2 * 13);
    for i in 0..n {
        if want[i] == 1 {
            assert!(out.left[i].is_finite(), "masked sample {i} not repaired");
            assert!(out.left[i].abs() < 0.3, "sample {i}: {}", out.left[i]);
        } else {
            assert_eq!(out.left[i].to_bits(), x[i].to_bits(), "unmasked {i}");
        }
    }
}

#[test]
fn contrast_gate_random_matches_reference() {
    // Random scores (many regions, some near the stream start to exercise
    // the edge clamping) on noisy audio with random spikes; the engine's
    // contrast-gated masks must equal the reference with the independent
    // numpy-semantics contrast port.
    let mut rng = Rng(1234);
    let mut runs = 0usize;
    for min_contrast in [4.0f32, 16.0, 30.0] {
        for _ in 0..3 {
            let post = PostConfig {
                min_contrast,
                ..default_post()
            };
            let model = config_model(post.clone());
            let n = 12_000;
            let mut s = random_scores(&mut rng, n);
            // The reference clips at len(s) while the engine sees the zero
            // tail: keep the last regions away from the end.
            s[n - 1000..].fill(QUIET);
            let mut x: Vec<f32> = (0..n)
                .map(|i| (0.1 * (i as f64 * 0.01).sin() + 0.003 * rng.normal()) as f32)
                .collect();
            for _ in 0..200 {
                let k = rng.below(n);
                x[k] += (rng.normal() * 0.2) as f32;
            }
            let src = FixtureScores::new(s.clone(), s.clone(), 512);
            let mut eng = Declicker::with_score_source(&model, Box::new(src));
            let out = eng.process_offline_with_masks(&x, &x);
            let want = regions_from_scores_ref(&s, &post, Some(&x));
            assert_eq!(out.masks[0], want, "min_contrast {min_contrast}");
            assert_eq!(out.masks[1], want, "min_contrast {min_contrast}");
            runs += (1..n).filter(|&i| want[i] == 1 && want[i - 1] == 0).count();
            runs += usize::from(want[0] == 1);
        }
    }
    eprintln!("contrast-gated random engine cases: {runs} runs, 0 mask mismatches");
    assert!(runs > 100, "random cases should keep regions ({runs})");
}

// ---- RegionFinder vs regions_from_scores ----

/// Direct port of `classical.regions_from_scores` (contrast gate if `x`).
fn regions_from_scores_ref(s: &[f32], cfg: &PostConfig, x: Option<&[f32]>) -> Vec<u8> {
    let n = s.len();
    let mut mask = vec![0u8; n];
    let thr = f64::from(cfg.threshold);
    if !s.iter().any(|&v| f64::from(v) > thr) {
        return mask;
    }
    let weak_thr = (thr - f64::from(cfg.ext_drop)).min(thr);
    let idx: Vec<usize> = (0..n).filter(|&i| f64::from(s[i]) > weak_thr).collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &i in &idx {
        match groups.last_mut() {
            Some(g) if i - *g.last().unwrap() <= cfg.merge => g.push(i),
            _ => groups.push(vec![i]),
        }
    }
    for g in groups {
        let (a, b) = (g[0], g[g.len() - 1] + 1);
        if s[a..b].iter().all(|&v| f64::from(v) <= thr) {
            continue;
        }
        let (a, b) = (a.saturating_sub(cfg.pre), (b + cfg.post).min(n));
        if b - a > cfg.max_region {
            continue;
        }
        if let Some(x) = x {
            let c = contrast_numpy(x, a as isize, b as isize, cfg.contrast_context as isize);
            if cfg.min_contrast > 0.0 && c < f64::from(cfg.min_contrast) {
                continue;
            }
        }
        mask[a..b].fill(1);
    }
    mask
}

/// Python slice `start:stop` on a sequence of length `len` (negative indices
/// count from the end; out-of-range bounds clamp), as a range.
fn py_slice(len: usize, start: isize, stop: isize) -> std::ops::Range<usize> {
    let n = len as isize;
    let norm = |i: isize| (if i < 0 { i + n } else { i }).clamp(0, n) as usize;
    let (a, b) = (norm(start), norm(stop));
    a..b.max(a)
}

/// Independent port of `classical.contrast` (guard 20) with numpy slicing
/// semantics, on the float64 signal (`method.py` passes float64 audio).
/// Written from the Python source, not from `post.rs`, to serve as an oracle.
fn contrast_numpy(x: &[f32], a: isize, b: isize, ctx: isize) -> f64 {
    let guard = 20isize;
    let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let n = x.len() as isize;
    let (a, b) = (a.max(1), b.min(n - 1));
    if b <= a {
        return 0.0;
    }
    let (lo, hi) = ((a - ctx).max(1), (b + ctx).min(n - 1));
    // d = abs(x[lo+1:hi+1] - 2 * x[lo:hi] + x[lo-1:hi-1])
    let (s1, s0, sm) = (
        py_slice(x.len(), lo + 1, hi + 1),
        py_slice(x.len(), lo, hi),
        py_slice(x.len(), lo - 1, hi - 1),
    );
    assert!(
        s1.len() == s0.len() && s0.len() == sm.len(),
        "numpy would not broadcast"
    );
    let d: Vec<f64> = s1
        .zip(s0)
        .zip(sm)
        .map(|((i1, i0), im)| (x[i1] - 2.0 * x[i0] + x[im]).abs())
        .collect();
    let (ia, ib) = (a - lo, b - lo);
    // around = np.r_[d[:max(0, ia - guard)], d[ib + guard:]]
    let mut around: Vec<f64> = d[py_slice(d.len(), 0, (ia - guard).max(0))].to_vec();
    around.extend_from_slice(&d[py_slice(d.len(), ib + guard, d.len() as isize)]);
    let reference = if around.is_empty() {
        0.0
    } else {
        around.sort_by(f64::total_cmp);
        let m = around.len();
        if m % 2 == 1 {
            around[m / 2]
        } else {
            (around[m / 2 - 1] + around[m / 2]) / 2.0
        }
    };
    let inside = &d[py_slice(d.len(), ia, ib)];
    let peak = inside.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    peak / (reference + 1e-12)
}

fn random_scores(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        // Background, weak bursts, strong spikes, long weak stretches.
        let kind = rng.below(10);
        let len = match kind {
            0..=4 => rng.below(40) + 1,
            5..=7 => rng.below(12) + 1,
            8 => rng.below(4) + 1,
            _ => rng.below(80) + 1,
        };
        for v in s.iter_mut().skip(i).take(len) {
            *v = match kind {
                0..=4 => (rng.uniform() * 2.0 - 3.0) as f32,
                5..=7 => (rng.uniform() * 2.5 + 0.5) as f32,
                8 => (rng.uniform() * 4.0 + 2.0) as f32,
                _ => (rng.uniform() * 2.0 + 0.6) as f32,
            };
        }
        i += len;
    }
    s
}

#[test]
fn region_finder_matches_reference_on_random_scores() {
    let mut rng = Rng(42);
    let mut total_regions = 0usize;
    for case in 0..4000 {
        let cfg = PostConfig {
            threshold: [2.75, 2.0, 3.5, 1.0][rng.below(4)],
            ext_drop: [2.0, 1.0, 0.0, -1.0][rng.below(4)],
            merge: rng.below(12),
            pre: rng.below(5),
            post: rng.below(12),
            max_region: [480, 60, 25, 12, 5][rng.below(5)],
            min_contrast: 0.0,
            contrast_context: 150,
        };
        let n = rng.below(600) + 1;
        let mut s = random_scores(&mut rng, n);
        // Pad so every group finalises and no region is clipped at the end
        // (the reference clips `b` to len(s); a stream has no end).
        s.extend(std::iter::repeat_n(QUIET, cfg.merge + cfg.post + 2));
        let want = regions_from_scores_ref(&s, &cfg, None);

        let mut f = RegionFinder::new(&cfg);
        let mut got = vec![0u8; s.len()];
        let mut max_frontier = 0u64;
        let mut last_start = 0u64;
        for (t, &v) in s.iter().enumerate() {
            if let Some(r) = f.push(t as u64, v) {
                assert!(
                    r.start >= max_frontier,
                    "case {case}: region {r:?} before frontier {max_frontier}"
                );
                assert!(r.start >= last_start, "case {case}: regions out of order");
                last_start = r.start;
                got[r.start as usize..(r.end as usize).min(s.len())].fill(1);
                total_regions += 1;
            }
            let fr = f.frontier();
            assert!(fr >= max_frontier, "case {case}: frontier went backwards");
            assert!(fr <= t as u64 + 1, "case {case}: frontier beyond t_next");
            max_frontier = fr;
        }
        assert_eq!(got, want, "case {case}: cfg {cfg:?}");
    }
    assert!(
        total_regions > 1000,
        "random cases should produce regions ({total_regions})"
    );
}

#[test]
fn region_finder_doomed_group_releases_frontier() {
    let cfg = PostConfig {
        max_region: 20,
        ..default_post()
    };
    let mut f = RegionFinder::new(&cfg);
    // A long weak stretch: once its padded span exceeds 20 it is doomed.
    for t in 0..100u64 {
        assert!(f.push(t, 1.0).is_none());
        let doomed = t + 1 + 8 > 20;
        assert_eq!(f.frontier(), if doomed { t + 1 - 2 } else { 0 }, "t {t}");
    }
}

#[test]
fn region_finder_set_threshold() {
    let mut f = RegionFinder::new(&default_post());
    f.set_threshold(10.0);
    let mut s = [QUIET; 50];
    s[20] = 5.0;
    let got: Vec<_> = s
        .iter()
        .enumerate()
        .filter_map(|(t, &v)| f.push(t as u64, v))
        .collect();
    assert!(got.is_empty());
    f.reset();
    f.set_threshold(4.0);
    let got: Vec<_> = s
        .iter()
        .enumerate()
        .filter_map(|(t, &v)| f.push(t as u64, v))
        .collect();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].start, got[0].end), (18, 29));
}

#[test]
fn contrast_matches_hand_computation() {
    // Alternating 0.25, 0: |x''| = 0.5 everywhere, so the reference median is
    // 0.5. A spike of 3 at 100 raises |x''| at 99..=101 to 3.5, 6.5, 3.5.
    let mut x: Vec<f32> = (0..300)
        .map(|i| if i % 2 == 0 { 0.25 } else { 0.0 })
        .collect();
    x[100] += 3.0;
    let c = contrast(&x, 98, 103, 50);
    assert!((f64::from(c) - 6.5 / (0.5 + 1e-12)).abs() < 1e-4, "{c}");
}

#[test]
fn contrast_matches_numpy_port_on_random_cases() {
    // post::contrast / contrast_with_scratch against the independent port,
    // including a or b beyond the signal, b <= a, tiny signals, ctx 0..200
    // and flat stretches (zero median).
    let mut rng = Rng(2024);
    let mut scratch = Vec::new();
    let mut nonzero = 0usize;
    for case in 0..20_000 {
        let n = rng.below(400);
        let flat = rng.below(5) == 0;
        let mut x: Vec<f32> = (0..n)
            .map(|_| {
                if flat {
                    0.25
                } else {
                    (rng.normal() * 0.01) as f32
                }
            })
            .collect();
        for _ in 0..rng.below(4) {
            if n > 0 {
                let k = rng.below(n);
                x[k] += (rng.normal() * 0.5) as f32;
            }
        }
        let a = rng.below(n + 60);
        let b = if rng.below(10) == 0 {
            rng.below(n + 60)
        } else {
            a + rng.below(60)
        };
        let ctx = rng.below(201);
        let want = contrast_numpy(&x, a as isize, b as isize, ctx as isize);
        let got = pressing_core::post::contrast_with_scratch(&x, a, b, ctx, &mut scratch);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "case {case}: n {n} a {a} b {b} ctx {ctx}: {got} vs {want}"
        );
        assert_eq!(contrast(&x, a, b, ctx).to_bits(), (want as f32).to_bits());
        nonzero += usize::from(want != 0.0);
    }
    assert!(
        nonzero > 10_000,
        "most cases should be non-trivial ({nonzero})"
    );
}

// ---- LSAR ----

/// Dense reference for one group: builds M and solves the least-squares
/// problem by Householder QR (like numpy lstsq on a full-rank system).
fn lsar_dense_ref(y: &mut [f64], unknown: &[bool], u0: usize, u1: usize, cfg: &LsarConfig) {
    let p = cfg.order;
    let mut r = vec![0.0; p + 1];
    let mut scratch = Vec::new();
    let left = y[u0.saturating_sub(cfg.context)..u0].to_vec();
    let right = y[u1..(u1 + cfg.context_right).min(y.len())].to_vec();
    for piece in [&left, &right] {
        if piece.len() > p {
            add_windowed_autocorr(piece, &mut r, &mut scratch);
        }
    }
    let mut a = vec![0.0; p + 1];
    let mut tmp = vec![0.0; p + 1];
    ar_from_autocorr(&r, &mut a, &mut tmp);
    let (c0, c1) = (u0 - p, u1 + p);
    let n = c1 - c0;
    let mut unk: Vec<bool> = unknown[c0..c1].to_vec();
    unk[..p].fill(false);
    let cols: Vec<usize> = (0..n).filter(|&c| unk[c]).collect();
    let rows = n - p;
    // Mu (rows x U) and rhs = -Mk yk.
    let mut m = vec![vec![0.0; cols.len()]; rows];
    let mut rhs = vec![0.0; rows];
    for i in 0..rows {
        for k in 0..=p {
            let c = i + k;
            let coef = a[p - k];
            if unk[c] {
                let j = cols.iter().position(|&q| q == c).unwrap();
                m[i][j] = coef;
            } else {
                rhs[i] -= coef * y[c0 + c];
            }
        }
    }
    // Householder QR least squares.
    let u = cols.len();
    for j in 0..u {
        let norm = (j..rows).map(|i| m[i][j] * m[i][j]).sum::<f64>().sqrt();
        let alpha = if m[j][j] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (j..rows).map(|i| m[i][j]).collect();
        v[0] -= alpha;
        let vn = v.iter().map(|x| x * x).sum::<f64>();
        if vn == 0.0 {
            continue;
        }
        for col in j..u {
            let dot: f64 = (j..rows).map(|i| v[i - j] * m[i][col]).sum();
            for i in j..rows {
                m[i][col] -= 2.0 * dot / vn * v[i - j];
            }
        }
        let dot: f64 = (j..rows).map(|i| v[i - j] * rhs[i]).sum();
        for i in j..rows {
            rhs[i] -= 2.0 * dot / vn * v[i - j];
        }
    }
    let mut x = vec![0.0; u];
    for j in (0..u).rev() {
        let s = rhs[j] - (j + 1..u).map(|k| m[j][k] * x[k]).sum::<f64>();
        x[j] = s / m[j][j];
    }
    for (j, &c) in cols.iter().enumerate() {
        y[c0 + c] = x[j];
    }
}

fn ar_signal(rng: &mut Rng, n: usize) -> Vec<f64> {
    // Two resonances plus a little noise: a reasonable audio stand-in.
    (0..n)
        .map(|i| {
            let t = i as f64;
            0.4 * (t * 0.031).sin() + 0.2 * (t * 0.17 + 1.0).sin() + 0.01 * rng.normal()
        })
        .collect()
}

#[test]
fn lsar_matches_dense_reference() {
    let cfg = LsarConfig::default();
    let mut rng = Rng(11);
    let mut solver = LsarSolver::new(&cfg);
    for case in 0..40 {
        let n = 4000;
        let y0 = ar_signal(&mut rng, n);
        let u0 = 1200 + rng.below(400);
        let span = rng.below(440) + 1;
        let u1 = u0 + span;
        let mut unknown = vec![false; n];
        // Random runs inside the group, plus masked samples just before u0
        // (earlier group: known) and in [u1, u1 + p) (solved, as reference).
        let mut i = u0;
        while i < u1 {
            let run = (rng.below(20) + 1).min(u1 - i);
            unknown[i..i + run].fill(true);
            i += run + rng.below(30) + 1;
        }
        unknown[u0] = true;
        unknown[u1 - 1] = true;
        if case % 2 == 0 {
            unknown[u0 - 5..u0 - 2].fill(true);
            unknown[u1 + 3..u1 + 9].fill(true);
        }
        let mut want = y0.clone();
        lsar_dense_ref(&mut want, &unknown, u0, u1, &cfg);
        let mut got = y0.clone();
        assert!(solver.try_fill(&mut got, &unknown, u0, u1), "case {case}");
        let mut max_err = 0.0f64;
        for k in 0..n {
            if unknown[k] && k >= u0 {
                max_err = max_err.max((got[k] - want[k]).abs());
            } else {
                assert_eq!(
                    got[k].to_bits(),
                    y0[k].to_bits(),
                    "case {case}: known sample {k} modified"
                );
            }
        }
        assert!(max_err < 1e-8, "case {case}: max err {max_err:e}");
    }
}

#[test]
fn lsar_pure_tone_gap_snr() {
    let cfg = LsarConfig::default();
    let mut solver = LsarSolver::new(&cfg);
    for (freq, phase) in [(440.0, 0.3), (1000.0, 0.0), (3150.0, 1.1), (9000.0, 2.0)] {
        let n = 3000;
        let w = 2.0 * std::f64::consts::PI * freq / 48_000.0;
        let x: Vec<f64> = (0..n).map(|i| 0.5 * (w * i as f64 + phase).sin()).collect();
        let (u0, u1) = (1500, 1532);
        let mut unknown = vec![false; n];
        unknown[u0..u1].fill(true);
        let mut y = x.clone();
        for v in &mut y[u0..u1] {
            *v = 0.0;
        }
        solver.fill(&mut y, &unknown, u0, u1);
        let sig: f64 = x[u0..u1].iter().map(|v| v * v).sum();
        let err: f64 = (u0..u1).map(|i| (y[i] - x[i]).powi(2)).sum();
        let snr = 10.0 * (sig / err.max(1e-300)).log10();
        println!("pure tone {freq} Hz, 32-sample gap: SNR {snr:.1} dB");
        assert!(snr > 60.0, "{freq} Hz: SNR {snr:.1} dB");
    }
}

#[test]
fn lsar_hand_computed_cases() {
    // Case 1: contexts too short for an AR fit -> A = [1, 0, ...]; every
    // unknown appears alone in one row, so the solution is exactly 0.
    let cfg = LsarConfig {
        order: 4,
        context: 4,
        context_right: 4,
        max_group: 16,
    };
    let mut solver = LsarSolver::new(&cfg);
    let mut y: Vec<f64> = (0..20).map(|i| 1.0 + i as f64).collect();
    let mut unknown = vec![false; 20];
    unknown[8..11].fill(true);
    solver.fill(&mut y, &unknown, 8, 11);
    for (i, v) in y.iter().enumerate() {
        let want = if (8..11).contains(&i) {
            0.0
        } else {
            1.0 + i as f64
        };
        assert_eq!(*v, want, "sample {i}");
    }

    // Case 2: order 1 on a constant c with 4-sample contexts. hann(4) =
    // [0, .75, .75, 0], so each piece gives r0 = 1.125 c², r1 = 0.5625 c².
    // a1 = -r1 / (r0 (1 + 1e-6)) = -0.5 / (1 + 1e-6). One unknown x between
    // neighbours c: minimise (x + a c)² + (c + a x)² -> x = -2 a c / (1 + a²).
    let cfg = LsarConfig {
        order: 1,
        context: 4,
        context_right: 4,
        max_group: 8,
    };
    let mut solver = LsarSolver::new(&cfg);
    let c = 0.3;
    let mut y = vec![c; 9];
    let mut unknown = vec![false; 9];
    unknown[4] = true;
    y[4] = 123.0;
    solver.fill(&mut y, &unknown, 4, 5);
    let a = -0.5 / (1.0 + 1e-6);
    let want = -2.0 * a * c / (1.0 + a * a);
    assert!((y[4] - want).abs() < 1e-14, "{} vs {}", y[4], want);
    assert_eq!(y[3], c);
    assert_eq!(y[5], c);
}

#[test]
fn ar_from_autocorr_solves_yule_walker() {
    let mut rng = Rng(5);
    let p = 12;
    let x = ar_signal(&mut rng, 2048);
    let mut r = vec![0.0; p + 1];
    let mut scratch = Vec::new();
    add_windowed_autocorr(&x, &mut r, &mut scratch);
    // Direct autocorrelation of the windowed signal.
    let m = x.len();
    let v: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            s * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64).cos())
        })
        .collect();
    for (k, &rk) in r.iter().enumerate() {
        let direct: f64 = (0..m - k).map(|i| v[i] * v[i + k]).sum();
        assert!(
            (rk - direct).abs() <= 1e-10 * direct.abs().max(1.0),
            "lag {k}"
        );
    }
    let mut a = vec![0.0; p + 1];
    let mut tmp = vec![0.0; p + 1];
    assert!(ar_from_autocorr(&r, &mut a, &mut tmp));
    // Residual of T a = -r[1..] with T[i][j] = r'[|i-j|], r'[0] = r[0](1+1e-6).
    let rr = |k: usize| if k == 0 { r[0] * (1.0 + 1e-6) } else { r[k] };
    for i in 1..=p {
        let lhs: f64 = (1..=p).map(|j| a[j] * rr(i.abs_diff(j))).sum();
        assert!((lhs + r[i]).abs() < 1e-9 * r[0], "row {i}");
    }
    // Fallbacks.
    let zero = vec![0.0; p + 1];
    assert!(!ar_from_autocorr(&zero, &mut a, &mut tmp));
    assert_eq!(a[0], 1.0);
    assert!(a[1..].iter().all(|&v| v == 0.0));
}
