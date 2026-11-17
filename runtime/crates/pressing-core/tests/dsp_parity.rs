//! Parity of model loading, features (spec section 1) and the streaming network
//! (spec section 2) against the Python fixtures. Run with `-- --nocapture` to
//! see the measured errors; the benchmark is `#[ignore]`d (run it in release).

use pressing_core::engine::ScoreSource;
use pressing_core::features::FeatureState;
use pressing_core::fixtures::Fixtures;
use pressing_core::model::{Model, ModelError, EMBEDDED_BIN, EMBEDDED_JSON};
use pressing_core::net::{Detector, StreamingNet};

// ---- Helpers ----

/// Small fixtures always, large ones when present.
///
/// Every set is checked to be non-degenerate (stereo, at least one window, more
/// than `D` samples), and the committed small set must have its known shape, so
/// a renamed/missing manifest key (which `Fixtures::open` reads as 0) fails
/// loudly instead of making the parity tests compare nothing.
fn fixture_sets() -> Vec<(&'static str, Fixtures)> {
    let d = Model::embedded().net.d;
    let small = Fixtures::small().expect("committed fixtures in tests/fixtures");
    assert_eq!(
        (small.windows, small.channels, small.samples),
        (1, 2, 48_000),
        "committed small fixture manifest shape"
    );
    let mut v = vec![("small", small)];
    match Fixtures::large() {
        Some(f) => v.push(("large", f)),
        None => eprintln!("fixtures-large absent: skipping large set"),
    }
    for (set, fx) in &v {
        assert!(
            fx.windows > 0 && fx.channels == 2 && fx.samples > d,
            "fixtures[{set}] degenerate manifest: windows {}, channels {}, samples {} (D = {d})",
            fx.windows,
            fx.channels,
            fx.samples
        );
    }
    v
}

/// Loads a fixture array and checks it has exactly `per_window * windows` values.
fn load(fx: &Fixtures, set: &str, name: &str, per_window: usize) -> Vec<f32> {
    let a = fx.f32s(name);
    assert_eq!(
        a.len(),
        fx.windows * per_window,
        "fixtures[{set}] `{name}` length vs manifest"
    );
    a
}

/// Number of target-aligned scores compared per fixture set.
fn expected_scores(fx: &Fixtures, d: usize) -> usize {
    fx.windows * fx.channels * (fx.samples - fx.samples % 4 - d)
}

/// Index into a (W, 2, 3, L) feature array.
fn fidx(fx: &Fixtures, w: usize, c: usize, f: usize, n: usize) -> usize {
    ((w * fx.channels + c) * 3 + f) * fx.samples + n
}

#[derive(Default)]
struct ErrStats {
    max: f64,
    sum: f64,
    n: usize,
    at: (usize, usize, usize),
}

impl ErrStats {
    fn add(&mut self, got: f32, want: f32, at: (usize, usize, usize)) {
        let e = (got as f64 - want as f64).abs();
        assert!(
            e.is_finite(),
            "non-finite value at {at:?}: got {got}, want {want}"
        );
        if e > self.max {
            self.max = e;
            self.at = at;
        }
        self.sum += e;
        self.n += 1;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
}

/// Streams one fixture window through `push` and returns per-channel logits.
fn stream_window(
    len: usize,
    mut push: impl FnMut(usize) -> Option<[[f32; 4]; 2]>,
) -> [Vec<f32>; 2] {
    let mut out = [Vec::with_capacity(len), Vec::with_capacity(len)];
    for n in 0..len {
        if let Some(y) = push(n) {
            for (o, yc) in out.iter_mut().zip(y) {
                o.extend_from_slice(&yc);
            }
        }
    }
    out
}

/// Compares target-aligned scores (`scores[t] == logit[t + D]`) for t in [0, L - D).
fn compare_scores(
    fx: &Fixtures,
    w: usize,
    logits: &[Vec<f32>; 2],
    d: usize,
    scores: &[f32],
    st: &mut ErrStats,
) {
    let l = fx.samples;
    for (c, lg) in logits.iter().enumerate() {
        assert_eq!(lg.len(), l - l % 4, "logit count");
        for t in 0..l.saturating_sub(d).min(lg.len() - d) {
            st.add(lg[t + d], scores[fx.idx(w, c, t)], (w, c, t));
        }
    }
}

// ---- 1. Model loading ----

#[test]
fn model_loads_with_expected_tensors() {
    let m = Model::embedded();
    assert_eq!(
        (m.net.c, m.net.blocks, m.net.stem_k, m.net.stride, m.net.d),
        (24, 8, 8, 4, 512)
    );
    assert_eq!(m.post.threshold, 2.75);
    assert_eq!(m.features.nw, 2048);
    assert_eq!(m.tensors.len(), 2 + 4 * 8 + 4);
    let shape = |n: &str| {
        m.tensors
            .iter()
            .find(|t| t.0 == n)
            .map(|t| t.1.clone())
            .expect(n)
    };
    assert_eq!(shape("stem.weight"), vec![24, 3, 8]);
    assert_eq!(shape("stem.bias"), vec![24]);
    for i in 0..8 {
        assert_eq!(shape(&format!("blocks.{i}.dil.weight")), vec![24, 24, 3]);
        assert_eq!(shape(&format!("blocks.{i}.dil.bias")), vec![24]);
        assert_eq!(shape(&format!("blocks.{i}.mix.weight")), vec![24, 24, 1]);
        assert_eq!(shape(&format!("blocks.{i}.mix.bias")), vec![24]);
    }
    assert_eq!(shape("head1.weight"), vec![24, 24, 1]);
    assert_eq!(shape("head2.weight"), vec![4, 24, 1]);
    assert_eq!(shape("head2.bias"), vec![4]);
    for (name, shape, data) in &m.tensors {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
    }
    // Same count as Python's StreamingNet.state_floats: stem ring 3*(k-s) plus
    // 24 channels x 2*2^i frames for 8 blocks.
    let net = StreamingNet::new(&m);
    assert_eq!(net.state_floats(), 3 * (8 - 4) + 24 * 2 * 255);
    let total: usize = m.tensors.iter().map(|t| t.2.len() * 4).sum();
    assert_eq!(total, EMBEDDED_BIN.len());
    // First stem weight read straight from the file.
    let b = &EMBEDDED_BIN[..4];
    assert_eq!(
        m.tensor("stem.weight").unwrap()[0],
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    );
}

fn expect_layout_err(json: &str, bin: &[u8], needle: &str) {
    match Model::from_json_and_bin(json, bin) {
        Err(e @ (ModelError::Layout(_) | ModelError::Json(_))) => {
            let msg = e.to_string();
            assert!(
                msg.contains(needle),
                "error `{msg}` does not mention `{needle}`"
            );
        }
        Ok(_) => panic!("expected an error mentioning `{needle}`"),
    }
}

#[test]
fn model_rejects_bad_inputs() {
    let mut v: serde_json::Value = serde_json::from_str(EMBEDDED_JSON).unwrap();
    expect_layout_err("{", EMBEDDED_BIN, "weights.json");
    // Truncated binary.
    let short = &EMBEDDED_BIN[..EMBEDDED_BIN.len() - 4];
    expect_layout_err(EMBEDDED_JSON, short, "total_bytes");
    v.as_object_mut().unwrap().remove("total_bytes");
    expect_layout_err(&v.to_string(), short, "exceed weights.bin length");
    // Wrong shape for a block weight.
    let mut bad = v.clone();
    bad["layers"][4]["shape"] = serde_json::json!([24, 24, 3]);
    expect_layout_err(&bad.to_string(), EMBEDDED_BIN, "blocks.0.mix.weight");
    // Config mismatch: C=16 no longer matches the tensors.
    let mut bad = v.clone();
    bad["net_config"]["C"] = serde_json::json!(16);
    expect_layout_err(&bad.to_string(), EMBEDDED_BIN, "expected [16, 3, 8]");
    // Missing and unexpected tensors.
    let mut bad = v.clone();
    bad["layers"].as_array_mut().unwrap().pop();
    expect_layout_err(
        &bad.to_string(),
        EMBEDDED_BIN,
        "missing tensor `head2.bias`",
    );
    let mut bad = v.clone();
    bad["layers"][0]["name"] = serde_json::json!("stem.w");
    expect_layout_err(&bad.to_string(), EMBEDDED_BIN, "unexpected tensor `stem.w`");
    // Unsupported stride.
    let mut bad = v.clone();
    bad["net_config"]["stride"] = serde_json::json!(2);
    expect_layout_err(&bad.to_string(), EMBEDDED_BIN, "stride");
}

/// Feature/post constants that would make the features NaN/inf, divide by zero,
/// or allocate absurdly are rejected at load time.
#[test]
fn model_rejects_bad_feature_and_post_constants() {
    let v: serde_json::Value = serde_json::from_str(EMBEDDED_JSON).unwrap();
    let cases: [(&str, &str, serde_json::Value, &str); 9] = [
        ("features", "NW", serde_json::json!(0), "NW must be in"),
        (
            "features",
            "NW",
            serde_json::json!(1u64 << 21),
            "NW must be in",
        ),
        (
            "features",
            "EPS",
            serde_json::json!(0.0),
            "EPS must be finite and > 0",
        ),
        (
            "features",
            "EPS",
            serde_json::json!(-1.0),
            "EPS must be finite and > 0",
        ),
        ("features", "FLOOR", serde_json::json!(0.0), "FLOOR must be"),
        (
            "features",
            "ASINH_SCALE",
            serde_json::json!(0.0),
            "ASINH_SCALE must be",
        ),
        (
            "features",
            "ASINH_SCALE",
            serde_json::json!(-8.0),
            "ASINH_SCALE must be",
        ),
        (
            "post",
            "threshold",
            serde_json::json!(1e39),
            "post.threshold must be finite",
        ),
        (
            "post",
            "ext_drop",
            serde_json::json!(-1e39),
            "post.ext_drop must be finite",
        ),
    ];
    for (section, key, val, needle) in cases {
        let mut bad = v.clone();
        bad[section][key] = val;
        expect_layout_err(&bad.to_string(), EMBEDDED_BIN, needle);
    }
    // The boundary values themselves are accepted.
    let mut ok = v.clone();
    ok["features"]["NW"] = serde_json::json!(1);
    let m = Model::from_json_and_bin(&ok.to_string(), EMBEDDED_BIN).expect("NW = 1 is valid");
    let mut fs = FeatureState::new(&m.features);
    let g = fs.push(0.5, -0.25);
    assert!(g.iter().flatten().all(|x| x.is_finite()));
}

/// A hand-built invalid config is refused by `FeatureState::new` instead of being
/// silently clamped (NW = 0 used to behave like NW = 1).
#[test]
#[should_panic(expected = "invalid feature config")]
fn feature_state_rejects_zero_window() {
    let mut cfg = Model::embedded().features;
    cfg.nw = 0;
    let _ = FeatureState::new(&cfg);
}

// ---- 2. Features ----

#[test]
fn features_match_fixtures() {
    let model = Model::embedded();
    for (set, fx) in fixture_sets() {
        let audio = load(&fx, set, "audio", 2 * fx.samples);
        let feats = load(&fx, set, "features", 2 * 3 * fx.samples);
        let mut st = ErrStats::default();
        let mut fs = FeatureState::new(&model.features);
        for w in 0..fx.windows {
            fs.reset();
            for n in 0..fx.samples {
                let g = fs.push(audio[fx.idx(w, 0, n)], audio[fx.idx(w, 1, n)]);
                for (c, gc) in g.iter().enumerate() {
                    for (f, &v) in gc.iter().enumerate() {
                        st.add(v, feats[fidx(&fx, w, c, f, n)], (w, c * 3 + f, n));
                    }
                }
            }
        }
        println!(
            "features[{set}]: {} values, max |diff| = {:.3e} at (w, c*3+f, n) = {:?}, mean = {:.3e}",
            st.n,
            st.max,
            st.at,
            st.mean()
        );
        assert_eq!(
            st.n,
            fx.windows * 2 * 3 * fx.samples,
            "features[{set}] count"
        );
        assert!(
            st.max < 1e-4,
            "features[{set}] max |diff| {:.3e} >= 1e-4",
            st.max
        );
    }
}

// ---- 3. Streaming network ----

#[test]
fn detector_logits_match_fixture_scores() {
    let model = Model::embedded();
    let d = model.net.d;
    let mut det = Detector::new(&model);
    for (set, fx) in fixture_sets() {
        let audio = load(&fx, set, "audio", 2 * fx.samples);
        let scores = load(&fx, set, "scores", 2 * fx.samples);
        let mut st = ErrStats::default();
        for w in 0..fx.windows {
            det.reset();
            let lg = stream_window(fx.samples, |n| {
                det.push(audio[fx.idx(w, 0, n)], audio[fx.idx(w, 1, n)])
            });
            compare_scores(&fx, w, &lg, d, &scores, &mut st);
        }
        println!(
            "detector[{set}]: {} scores, max |diff| = {:.3e} at (w, c, t) = {:?}, mean = {:.3e}",
            st.n,
            st.max,
            st.at,
            st.mean()
        );
        assert_eq!(st.n, expected_scores(&fx, d), "{set} score count");
        assert!(
            st.max < 1e-3,
            "detector[{set}] max |diff| {:.3e} >= 1e-3",
            st.max
        );
    }
}

/// The network alone, fed the fixture features (isolates section 2 from section 1).
#[test]
fn net_on_fixture_features_matches_scores() {
    let model = Model::embedded();
    let d = model.net.d;
    let mut net = StreamingNet::new(&model);
    for (set, fx) in fixture_sets() {
        let feats = load(&fx, set, "features", 2 * 3 * fx.samples);
        let scores = load(&fx, set, "scores", 2 * fx.samples);
        let mut st = ErrStats::default();
        for w in 0..fx.windows {
            net.reset();
            let lg = stream_window(fx.samples, |n| {
                let f = |c| std::array::from_fn(|k| feats[fidx(&fx, w, c, k, n)]);
                net.push([f(0), f(1)])
            });
            compare_scores(&fx, w, &lg, d, &scores, &mut st);
        }
        println!(
            "net-only[{set}]: {} scores, max |diff| = {:.3e} at (w, c, t) = {:?}, mean = {:.3e}",
            st.n,
            st.max,
            st.at,
            st.mean()
        );
        assert_eq!(st.n, expected_scores(&fx, d), "{set} score count");
        assert!(
            st.max < 1e-3,
            "net-only[{set}] max |diff| {:.3e} >= 1e-3",
            st.max
        );
    }
}

/// After `reset`, the detector reproduces a fresh detector bit for bit.
#[test]
fn reset_restores_start_state() {
    let model = Model::embedded();
    let fx = Fixtures::small().expect("committed fixtures");
    let audio = fx.f32s("audio");
    let n = 6000;
    let run = |det: &mut Detector| {
        stream_window(n, |i| {
            det.push(audio[fx.idx(0, 0, i)], audio[fx.idx(0, 1, i)])
        })
    };
    let mut fresh = Detector::new(&model);
    let want = run(&mut fresh);
    let mut used = Detector::new(&model);
    // Leave it mid-frame with non-trivial history.
    for i in 0..4097 {
        used.push(audio[fx.idx(0, 1, i)], audio[fx.idx(0, 0, i)]);
    }
    used.reset();
    assert_eq!(run(&mut used), want);
}

// ---- 4. Micro-benchmark ----

/// `cargo test -p pressing-core --release --test dsp_parity -- --ignored --nocapture bench`
#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn bench_detector_ns_per_stereo_sample() {
    use std::hint::black_box;
    use std::time::Instant;

    let model = Model::embedded();
    let fx = Fixtures::small().expect("committed fixtures");
    let audio = fx.f32s("audio");
    let (l, r) = (
        &audio[fx.idx(0, 0, 0)..][..fx.samples],
        &audio[fx.idx(0, 1, 0)..][..fx.samples],
    );
    let seconds = 20;
    let total = 48_000 * seconds;

    let mut det = Detector::new(&model);
    let mut fs = FeatureState::new(&model.features);
    let mut net = StreamingNet::new(&model);
    let feat0: Vec<[[f32; 3]; 2]> = l.iter().zip(r).map(|(&a, &b)| fs.push(a, b)).collect();

    let time = |name: &str, f: &mut dyn FnMut(usize) -> f32| {
        let mut acc = 0.0;
        for i in 0..48_000 {
            acc += f(i % l.len()); // warm-up
        }
        let t0 = Instant::now();
        for i in 0..total {
            acc += f(i % l.len());
        }
        let ns = t0.elapsed().as_nanos() as f64 / total as f64;
        black_box(acc);
        println!(
            "bench {name}: {ns:.1} ns per stereo sample, {:.3}% of one core at 48 kHz (real-time factor {:.0}x)",
            ns * 48_000.0 / 1e9 * 100.0,
            1e9 / (ns * 48_000.0)
        );
    };
    time("features", &mut |i| fs.push(l[i], r[i])[0][0]);
    time("net", &mut |i| net.push(feat0[i]).map_or(0.0, |y| y[0][0]));
    time("detector", &mut |i| {
        det.push(l[i], r[i]).map_or(0.0, |y| y[1][3])
    });
}
