//! Full-pipeline parity with the real network: `Declicker::new` with the
//! embedded R2 model (features + streaming ClickNet + region finder + LSAR)
//! against the Python offline reference masks and repaired audio.
//!
//! Spec section 5: with the real network, masks may differ only where a score
//! sits within ~1e-3 of a threshold, so differing runs must be <= 1% of the
//! reference runs. Compared on `[edge_ignore, L - edge_ignore)` because the
//! offline reference zero-pads features at the window end.

use pressing_core::fixtures::Fixtures;
use pressing_core::{Declicker, Model};

/// Runs of 1s in `mask[lo..hi]` as `(start, end)` (runs crossing `lo`/`hi`
/// are clipped identically for both masks).
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

#[derive(Default)]
struct Report {
    ref_runs: usize,
    eng_runs: usize,
    differing_runs: usize,
    /// Samples in the compared range whose mask bits differ.
    mask_sample_mismatches: usize,
    /// max |repaired - reference| over samples where both masks agree.
    max_err_mask_agree: f64,
    /// Same, but only inside runs that agree exactly (or outside any run).
    max_err_run_agree: f64,
    compared: usize,
    deadline_misses: u64,
    solve_failures: u64,
}

fn run_network_parity(fx: &Fixtures, name: &str) -> Report {
    let model = Model::embedded();
    // The fixtures were generated with the same post settings as the model.
    let p = &fx.manifest["post"];
    let post = &model.post;
    assert_eq!(p["threshold"].as_f64().unwrap() as f32, post.threshold);
    assert_eq!(p["ext_drop"].as_f64().unwrap() as f32, post.ext_drop);
    assert_eq!(p["merge"].as_u64().unwrap() as usize, post.merge);
    assert_eq!(p["pre"].as_u64().unwrap() as usize, post.pre);
    assert_eq!(p["post"].as_u64().unwrap() as usize, post.post);
    assert_eq!(p["max_region"].as_u64().unwrap() as usize, post.max_region);
    assert_eq!(
        p["min_contrast"].as_f64().unwrap() as f32,
        post.min_contrast
    );
    assert_eq!(fx.manifest["D"].as_u64().unwrap() as usize, model.net.d);
    assert_eq!(fx.manifest["sample_rate"].as_u64(), Some(48_000));

    let audio = fx.f32s("audio");
    let masks = fx.u8s("masks");
    let reference = fx.f32s("repaired");
    let l = fx.samples;
    let edge = fx.manifest["edge_ignore"].as_u64().unwrap_or(2400) as usize;
    assert!(
        fx.windows > 0 && fx.channels == 2 && l > 2 * edge,
        "{name}: fixture shape"
    );
    assert_eq!(audio.len(), fx.windows * 2 * l);
    assert_eq!(masks.len(), audio.len());
    assert_eq!(reference.len(), audio.len());

    let mut eng = Declicker::new(&model, 48_000.0);
    let mut rep = Report::default();
    for w in 0..fx.windows {
        let ch = |a: &[f32], c: usize| a[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l].to_vec();
        let (xl, xr) = (ch(&audio, 0), ch(&audio, 1));
        // process_offline resets the engine (and its stats) first.
        let out = eng.process_offline_with_masks(&xl, &xr);
        let st = eng.stats();
        rep.deadline_misses += st.deadline_misses;
        rep.solve_failures += st.solve_failures;

        for (c, (y, x)) in [(&out.left, &xl), (&out.right, &xr)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(y.len(), l);
            let m_eng = &out.masks[c];
            let m_ref = &masks[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l];
            let r_ref = &reference[fx.idx(w, c, 0)..fx.idx(w, c, 0) + l];

            let a = runs(m_ref, edge, l - edge);
            let b = runs(m_eng, edge, l - edge);
            let diff = a.iter().filter(|r| !b.contains(r)).count()
                + b.iter().filter(|r| !a.contains(r)).count();

            // Exclude every run (in either mask) that touches a disagreement.
            let mut excluded = vec![false; l];
            let busy = |k: usize| m_eng[k] != 0 || m_ref[k] != 0;
            let mut i = 0;
            while i < l {
                if m_eng[i] != m_ref[i] {
                    let (mut s, mut t) = (i, i);
                    while s > 0 && busy(s - 1) {
                        s -= 1;
                    }
                    while t < l && busy(t) {
                        t += 1;
                    }
                    excluded[s..t].fill(true);
                    i = t.max(i + 1);
                } else {
                    i += 1;
                }
            }

            let (mut e_mask, mut e_run, mut mism) = (0.0f64, 0.0f64, 0usize);
            for i in 0..l {
                if m_eng[i] == 0 {
                    assert_eq!(
                        y[i].to_bits(),
                        x[i].to_bits(),
                        "{name} w{w} c{c}: unmasked sample {i} modified"
                    );
                }
                if i < edge || i >= l - edge {
                    continue;
                }
                rep.compared += 1;
                if m_eng[i] != m_ref[i] {
                    mism += 1;
                    continue;
                }
                let err = (f64::from(y[i]) - f64::from(r_ref[i])).abs();
                e_mask = e_mask.max(err);
                if !excluded[i] {
                    e_run = e_run.max(err);
                }
            }
            println!(
                "{name} w{w} c{c}: ref runs {:3}, engine runs {:3}, differing {}, \
                 mask samples differing {mism}, max |err| masks-agree {e_mask:.3e}, \
                 runs-agree {e_run:.3e}",
                a.len(),
                b.len(),
                diff
            );
            rep.ref_runs += a.len();
            rep.eng_runs += b.len();
            rep.differing_runs += diff;
            rep.mask_sample_mismatches += mism;
            rep.max_err_mask_agree = rep.max_err_mask_agree.max(e_mask);
            rep.max_err_run_agree = rep.max_err_run_agree.max(e_run);
        }
        println!(
            "{name} w{w}: deadline_misses {}, solve_failures {}, stats.regions {}",
            st.deadline_misses, st.solve_failures, st.regions
        );
    }
    println!(
        "{name} TOTAL: {} windows, ref runs {}, engine runs {}, differing runs {} ({:.2}%), \
         mask samples differing {} of {}, max |err| where masks agree {:.3e}, \
         where runs agree {:.3e}, deadline_misses {}, solve_failures {}",
        fx.windows,
        rep.ref_runs,
        rep.eng_runs,
        rep.differing_runs,
        100.0 * rep.differing_runs as f64 / rep.ref_runs.max(1) as f64,
        rep.mask_sample_mismatches,
        rep.compared,
        rep.max_err_mask_agree,
        rep.max_err_run_agree,
        rep.deadline_misses,
        rep.solve_failures
    );
    rep
}

fn check(rep: &Report, name: &str) {
    assert!(rep.ref_runs > 0, "{name}: fixture should contain clicks");
    assert!(
        rep.differing_runs as f64 <= 0.01 * rep.ref_runs as f64,
        "{name}: {} differing runs of {} reference (> 1%)",
        rep.differing_runs,
        rep.ref_runs
    );
    assert!(
        rep.max_err_run_agree <= 1e-5,
        "{name}: repaired error {:.3e} inside agreeing runs",
        rep.max_err_run_agree
    );
    assert_eq!(rep.deadline_misses, 0, "{name}: deadline misses");
    assert_eq!(rep.solve_failures, 0, "{name}: solve failures");
}

#[test]
fn network_pipeline_small_fixture() {
    let fx = Fixtures::small().expect("committed small fixtures");
    let rep = run_network_parity(&fx, "small");
    check(&rep, "small");
}

#[test]
fn network_pipeline_large_fixture() {
    let Some(fx) = Fixtures::large() else {
        eprintln!("fixtures-large absent; skipping");
        return;
    };
    let rep = run_network_parity(&fx, "large");
    check(&rep, "large");
}
