//! `pressing parity DIR`: fixture parity checks (spec sections 1-5) against
//! the Python reference fixtures.
//!
//! Checks and tolerances (the same as the `pressing-core` tests):
//! - features: max |diff| < 1e-4;
//! - scores (features + network): max |diff| < 1e-3 on target-aligned scores;
//! - engine driven by the fixture scores: masks equal on
//!   `[edge_ignore, L - edge_ignore)`, repaired max |diff| <= 1e-5 there,
//!   unmasked samples bit-exact everywhere, no deadline misses;
//! - engine driven by the network: differing runs <= 1% of reference runs,
//!   no deadline misses (repaired error is reported where the runs agree).

use std::path::Path;

use anyhow::{bail, Context, Result};
use pressing_core::engine::OfflineOutput;
use pressing_core::features::FeatureState;
use pressing_core::fixtures::Fixtures;
use pressing_core::lsar::LsarConfig;
use pressing_core::net::Detector;
use pressing_core::{Declicker, Model, PostConfig, ScoreSource};

/// Score that can never trigger or extend a region.
const QUIET: f32 = -100.0;

/// Replays target-aligned fixture scores with the model delay `D`, emitted in
/// frames of 4 on pushes `n = 4j + 3` exactly like the network.
struct FixtureScores {
    scores: [Vec<f32>; 2],
    delay: usize,
    n: usize,
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
                *v = (n - 3 + k)
                    .checked_sub(self.delay)
                    .and_then(|t| self.scores[c].get(t).copied())
                    .unwrap_or(QUIET);
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

/// Max/mean absolute error accumulator.
#[derive(Default)]
struct Err {
    max: f64,
    sum: f64,
    n: u64,
    non_finite: u64,
}

impl Err {
    fn add(&mut self, got: f32, want: f32) {
        let e = (f64::from(got) - f64::from(want)).abs();
        if !e.is_finite() {
            self.non_finite += 1;
            return;
        }
        self.max = self.max.max(e);
        self.sum += e;
        self.n += 1;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
    fn ok(&self, tol: f64, inclusive: bool) -> bool {
        self.non_finite == 0
            && self.n > 0
            && if inclusive {
                self.max <= tol
            } else {
                self.max < tol
            }
    }
}

/// Collects check results and prints them.
#[derive(Default)]
struct Checks {
    failed: Vec<String>,
}

impl Checks {
    fn report(&mut self, name: &str, pass: bool, detail: String) {
        println!(
            "{:<5} {name:<16} {detail}",
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            self.failed.push(name.to_string());
        }
    }
}

/// Runs of 1s in `mask[lo..hi]` as `(start, end)`.
fn runs(mask: &[u8], lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut i = lo;
    while i < hi {
        if mask[i] != 0 {
            let s = i;
            while i < hi && mask[i] != 0 {
                i += 1;
            }
            v.push((s, i));
        } else {
            i += 1;
        }
    }
    v
}

/// Run all parity checks on the fixtures in `dir`; fails if any check fails.
pub fn run(dir: &Path) -> Result<()> {
    let fx = Fixtures::open(dir)
        .with_context(|| format!("no readable manifest.json in {}", dir.display()))?;
    let model = Model::embedded();
    let d = model.net.d;
    let (w_n, l) = (fx.windows, fx.samples);
    if w_n == 0 || fx.channels != 2 || l <= d {
        bail!(
            "degenerate manifest: windows {w_n}, channels {}, samples {l}",
            fx.channels
        );
    }
    if let Some(sr) = fx.manifest["sample_rate"].as_u64() {
        if sr != 48_000 {
            bail!("fixtures are at {sr} Hz; parity is defined at 48 kHz");
        }
    }
    if let Some(fd) = fx.manifest["D"].as_u64() {
        if fd as usize != d {
            bail!("fixture D = {fd} but the embedded model has D = {d}");
        }
    }
    let edge = fx.manifest["edge_ignore"].as_u64().unwrap_or(2400) as usize;
    let load = |name: &str, per: usize| -> Result<Vec<f32>> {
        let p = fx.dir.join(format!("{name}.bin"));
        let len = std::fs::metadata(&p)
            .with_context(|| format!("missing {}", p.display()))?
            .len() as usize;
        if len != 4 * w_n * per {
            bail!("{name}.bin: {len} bytes, expected {}", 4 * w_n * per);
        }
        Ok(fx.f32s(name))
    };
    let audio = load("audio", 2 * l)?;
    let feats = load("features", 2 * 3 * l)?;
    let scores = load("scores", 2 * l)?;
    let reference = load("repaired", 2 * l)?;
    let masks_path = fx.dir.join("masks.bin");
    let masks =
        std::fs::read(&masks_path).with_context(|| format!("missing {}", masks_path.display()))?;
    if masks.len() != w_n * 2 * l {
        bail!("masks.bin: {} bytes, expected {}", masks.len(), w_n * 2 * l);
    }
    check_repair_config(&fx.manifest)?;
    let post: PostConfig = match fx.manifest.get("post") {
        Some(v) => serde_json::from_value(v.clone()).context("manifest post")?,
        None => model.post.clone(),
    };
    println!(
        "fixtures: {} ({w_n} windows x 2 ch x {l} samples, edge_ignore {edge}, threshold {})",
        dir.display(),
        post.threshold
    );
    let ch = |a: &[f32], w: usize, c: usize| a[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l].to_vec();
    let mut checks = Checks::default();

    // ---- 1. Features ----
    let mut fs = FeatureState::new(&model.features);
    let mut e = Err::default();
    for w in 0..w_n {
        fs.reset();
        for n in 0..l {
            let g = fs.push(audio[fx.idx(w, 0, n)], audio[fx.idx(w, 1, n)]);
            for (c, gc) in g.iter().enumerate() {
                for (f, &v) in gc.iter().enumerate() {
                    e.add(v, feats[((w * 2 + c) * 3 + f) * l + n]);
                }
            }
        }
    }
    checks.report(
        "features",
        e.ok(1e-4, false),
        format!(
            "max |diff| {:.3e} (tol < 1e-4), mean {:.3e}, n {}",
            e.max,
            e.mean(),
            e.n
        ),
    );

    // ---- 2. Scores (features + network) ----
    let mut det = Detector::new(&model);
    let mut e = Err::default();
    for w in 0..w_n {
        det.reset();
        let mut lg: [Vec<f32>; 2] = [Vec::with_capacity(l), Vec::with_capacity(l)];
        for n in 0..l {
            if let Some(y) = det.push(audio[fx.idx(w, 0, n)], audio[fx.idx(w, 1, n)]) {
                lg[0].extend_from_slice(&y[0]);
                lg[1].extend_from_slice(&y[1]);
            }
        }
        for (c, lgc) in lg.iter().enumerate() {
            for t in 0..lgc.len().saturating_sub(d) {
                e.add(lgc[t + d], scores[fx.idx(w, c, t)]);
            }
        }
    }
    checks.report(
        "scores",
        e.ok(1e-3, false),
        format!(
            "max |diff| {:.3e} (tol < 1e-3), mean {:.3e}, n {}",
            e.max,
            e.mean(),
            e.n
        ),
    );

    // ---- 3-5. Engine with fixture scores: masks + repaired ----
    let mut cfg_model = model.clone();
    cfg_model.post = post.clone();
    let mut mask_mismatch = 0u64;
    let mut mask_compared = 0u64;
    let mut unmasked_modified = 0u64;
    let mut misses = 0u64;
    let mut e = Err::default();
    let (mut n_runs, mut n_rep) = (0u64, 0u64);
    for w in 0..w_n {
        let src = FixtureScores {
            scores: [ch(&scores, w, 0), ch(&scores, w, 1)],
            delay: d,
            n: 0,
        };
        let mut eng = Declicker::with_score_source(&cfg_model, Box::new(src));
        let (xl, xr) = (ch(&audio, w, 0), ch(&audio, w, 1));
        let out = eng.process_offline_with_masks(&xl, &xr);
        let st = eng.stats();
        misses += st.deadline_misses;
        n_runs += st.regions;
        n_rep += st.samples_repaired;
        compare_engine(
            &fx,
            w,
            &out,
            &[&xl, &xr],
            &masks,
            &reference,
            edge,
            &mut e,
            &mut unmasked_modified,
            None,
            Some((&mut mask_mismatch, &mut mask_compared)),
        );
    }
    checks.report(
        "masks",
        mask_mismatch == 0 && mask_compared > 0,
        format!("{mask_mismatch} mismatched of {mask_compared} compared samples; {n_runs} runs over the full windows"),
    );
    checks.report(
        "repaired",
        e.ok(1e-5, true) && unmasked_modified == 0 && misses == 0,
        format!(
            "max |diff| {:.3e} (tol <= 1e-5), mean {:.3e}; unmasked modified {unmasked_modified}; deadline misses {misses}; {n_rep} samples repaired",
            e.max,
            e.mean()
        ),
    );

    // ---- Engine with the network (end to end) ----
    let mut eng = Declicker::new(&cfg_model, 48_000.0);
    let (mut ref_runs, mut diff_runs, mut misses) = (0usize, 0usize, 0u64);
    let mut e = Err::default();
    let mut unmasked_modified = 0u64;
    for w in 0..w_n {
        let (xl, xr) = (ch(&audio, w, 0), ch(&audio, w, 1));
        let out = eng.process_offline_with_masks(&xl, &xr);
        misses += eng.stats().deadline_misses;
        for c in 0..2 {
            let m_ref = &masks[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l];
            let a = runs(m_ref, edge, l - edge.min(l));
            let b = runs(&out.masks[c], edge, l - edge.min(l));
            ref_runs += a.len();
            diff_runs += a.iter().filter(|r| !b.contains(r)).count();
            diff_runs += b.iter().filter(|r| !a.contains(r)).count();
        }
        // Repaired error only inside runs that agree exactly: exclude the
        // union of both masks around every disagreement.
        let mut exclude = [vec![false; l], vec![false; l]];
        for (c, ex) in exclude.iter_mut().enumerate() {
            let (m, m_ref) = (&out.masks[c], &masks[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l]);
            let busy = |k: usize| m[k] != 0 || m_ref[k] != 0;
            let mut i = 0;
            while i < l {
                if m[i] != m_ref[i] {
                    let (mut s, mut t) = (i, i);
                    while s > 0 && busy(s - 1) {
                        s -= 1;
                    }
                    while t < l && busy(t) {
                        t += 1;
                    }
                    ex[s..t].fill(true);
                    i = t.max(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        compare_engine(
            &fx,
            w,
            &out,
            &[&xl, &xr],
            &masks,
            &reference,
            edge,
            &mut e,
            &mut unmasked_modified,
            Some(&exclude),
            None,
        );
    }
    let frac = diff_runs as f64 / ref_runs.max(1) as f64;
    // With no reference runs in the compared range there is nothing to
    // compare: fail rather than pass vacuously.
    checks.report(
        "end-to-end",
        ref_runs > 0 && frac <= 0.01 && misses == 0 && unmasked_modified == 0,
        format!(
            "differing runs {diff_runs} of {ref_runs} reference ({:.2}%, tol <= 1%); deadline misses {misses}; repaired max |diff| {:.3e} where runs agree{}",
            100.0 * frac,
            e.max,
            if ref_runs == 0 {
                " (no reference runs in the compared range)"
            } else {
                ""
            }
        ),
    );

    if checks.failed.is_empty() {
        println!("parity: all checks passed");
        Ok(())
    } else {
        bail!("parity failed: {}", checks.failed.join(", "))
    }
}

/// The engine always repairs with [`LsarConfig::default`]; fixtures generated
/// with other repair settings cannot be compared, so refuse them. A manifest
/// without a `repair` entry is taken to use the defaults.
fn check_repair_config(manifest: &serde_json::Value) -> Result<()> {
    let Some(rep) = manifest.get("repair") else {
        return Ok(());
    };
    let field = |k: &str| -> Result<usize> {
        rep[k]
            .as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("manifest repair.{k} missing or not an integer"))
    };
    let got = LsarConfig {
        order: field("order")?,
        context: field("context")?,
        context_right: field("context_right")?,
        max_group: field("max_group")?,
    };
    let want = LsarConfig::default();
    if got != want {
        bail!(
            "fixtures use repair settings {got:?} but the engine uses {want:?}; regenerate the fixtures with the default repair settings"
        );
    }
    Ok(())
}

/// Compares engine output against the reference. Samples whose output mask is
/// 0 must equal the input bit for bit (everywhere). Inside the compared range,
/// samples flagged in `exclude` are skipped; the others are compared with the
/// reference repaired signal and, if `mismatch` is given as `(mismatched,
/// compared)` counters, the masks are compared too.
#[allow(clippy::too_many_arguments)]
fn compare_engine(
    fx: &Fixtures,
    w: usize,
    out: &OfflineOutput,
    input: &[&Vec<f32>; 2],
    masks: &[u8],
    reference: &[f32],
    edge: usize,
    e: &mut Err,
    unmasked_modified: &mut u64,
    exclude: Option<&[Vec<bool>; 2]>,
    mut mismatch: Option<(&mut u64, &mut u64)>,
) {
    let l = fx.samples;
    for (c, y) in [&out.left, &out.right].into_iter().enumerate() {
        let base = fx.idx(w, c, 0);
        for i in 0..l.min(y.len()) {
            let m = out.masks[c][i];
            if m == 0 && y[i].to_bits() != input[c][i].to_bits() {
                *unmasked_modified += 1;
            }
            if i < edge || i + edge >= l || exclude.is_some_and(|x| x[c][i]) {
                continue;
            }
            if let Some((mm, n)) = mismatch.as_mut() {
                **n += 1;
                if m != masks[base + i] {
                    **mm += 1;
                }
            }
            e.add(y[i], reference[base + i]);
        }
    }
}
