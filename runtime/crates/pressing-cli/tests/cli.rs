//! End-to-end tests of the `pressing` binary: synthetic WAVs with clicks are
//! written to a temp dir, processed, and read back.

use std::path::{Path, PathBuf};

mod common;

use common::{click_error, ok, pressing, read_wav, s, signal, tmp, write_wav};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../pressing-core/tests/fixtures")
}

// ---- declick ----

#[test]
fn declick_stereo_reduces_clicks_and_preserves_format() {
    let dir = tmp();
    let (inp, out) = (dir.path().join("in.wav"), dir.path().join("out.wav"));
    let rate = 48_000;
    let (clean, clicked, starts) = signal(rate, 1.5, 8);
    write_wav(&inp, &[&clicked[0], &clicked[1]], rate, 24, false);

    let stdout = ok(&["declick", s(&inp), s(&out)]);
    assert!(stdout.contains("deadline misses 0"), "{stdout}");

    let (spec, y) = read_wav(&out);
    assert_eq!(spec.sample_rate, rate);
    assert_eq!(spec.channels, 2);
    assert_eq!(
        (spec.bits_per_sample, spec.sample_format),
        (24, hound::SampleFormat::Int)
    );
    for c in 0..2 {
        assert_eq!(y[c].len(), clicked[c].len(), "length preserved");
        let before = click_error(&clicked[c], &clean[c], &starts);
        let after = click_error(&y[c], &clean[c], &starts);
        println!("ch{c}: click error energy {before:.4e} -> {after:.4e}");
        assert!(
            after < 0.1 * before,
            "ch{c}: clicks not reduced enough ({before:.4e} -> {after:.4e})"
        );
        // Most samples pass through bit for bit (24-bit round trip).
        let same = y[c].iter().zip(&clicked[c]).filter(|(a, b)| a == b).count();
        assert!(
            same as f64 > 0.95 * y[c].len() as f64,
            "ch{c}: only {same} of {} samples unchanged",
            y[c].len()
        );
    }
}

#[test]
fn bypass_is_bit_exact_and_clicks_only_is_the_difference() {
    let dir = tmp();
    let inp = dir.path().join("in.wav");
    let (_, clicked, _) = signal(48_000, 1.0, 4);
    write_wav(&inp, &[&clicked[0], &clicked[1]], 48_000, 24, false);

    let byp = dir.path().join("bypass.wav");
    ok(&["declick", s(&inp), s(&byp), "--mode", "bypass"]);
    let (_, y) = read_wav(&byp);
    assert_eq!(y[0], clicked[0]);
    assert_eq!(y[1], clicked[1]);

    let dec = dir.path().join("declick.wav");
    let clk = dir.path().join("clicks.wav");
    ok(&["declick", s(&inp), s(&dec), "--float"]);
    ok(&[
        "declick",
        s(&inp),
        s(&clk),
        "--mode",
        "clicks-only",
        "--float",
    ]);
    let (spec, d) = read_wav(&dec);
    assert_eq!(spec.sample_format, hound::SampleFormat::Float);
    let (_, k) = read_wav(&clk);
    let mut removed = 0.0f64;
    for c in 0..2 {
        for i in 0..d[c].len() {
            let want = clicked[c][i] - d[c][i];
            assert!((k[c][i] - want).abs() < 1e-6, "clicks-only mismatch at {i}");
            removed += f64::from(k[c][i]).abs();
        }
    }
    assert!(removed > 0.0, "clicks-only output is silent");
}

#[test]
fn declick_mono_16bit_44k1_writes_mono() {
    let dir = tmp();
    let (inp, out) = (dir.path().join("mono.wav"), dir.path().join("out.wav"));
    let rate = 44_100;
    let (clean, clicked, starts) = signal(rate, 1.0, 5);
    write_wav(&inp, &[&clicked[0]], rate, 16, false);

    let stdout = ok(&["declick", s(&inp), s(&out), "--report"]);
    assert!(stdout.contains("report:"), "{stdout}");
    let (spec, y) = read_wav(&out);
    assert_eq!((spec.channels, spec.sample_rate), (1, rate));
    assert_eq!(
        spec.bits_per_sample, 24,
        "integer sources are written as 24-bit"
    );
    assert_eq!(y[0].len(), clicked[0].len());
    // The input was quantised to 16 bits; compare against the 16-bit version.
    let q16 = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .map(|&x| ((f64::from(x) * 32_768.0).round() / 32_768.0) as f32)
            .collect()
    };
    let (x16, c16) = (q16(&clicked[0]), q16(&clean[0]));
    let before = click_error(&x16, &c16, &starts);
    let after = click_error(&y[0], &c16, &starts);
    println!("mono: click error energy {before:.4e} -> {after:.4e}");
    assert!(after < 0.1 * before, "{before:.4e} -> {after:.4e}");
}

#[test]
fn declick_96k_preserves_rate_and_length() {
    let dir = tmp();
    let (inp, out) = (dir.path().join("in.wav"), dir.path().join("out.wav"));
    let rate = 96_000;
    let (_, clicked, _) = signal(rate, 0.75, 3);
    write_wav(&inp, &[&clicked[0], &clicked[1]], rate, 24, false);
    let stdout = ok(&["declick", s(&inp), s(&out)]);
    assert!(stdout.contains("latency 3136 samples"), "{stdout}");
    let (spec, y) = read_wav(&out);
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (2, rate, 24)
    );
    assert_eq!(y[0].len(), clicked[0].len());
    assert_eq!(y[1].len(), clicked[1].len());
}

#[test]
fn max_seconds_and_threshold_flags() {
    let dir = tmp();
    let (inp, out) = (dir.path().join("in.wav"), dir.path().join("out.wav"));
    let (_, clicked, _) = signal(48_000, 1.0, 4);
    write_wav(&inp, &[&clicked[0], &clicked[1]], 48_000, 32, true);
    ok(&[
        "declick",
        s(&inp),
        s(&out),
        "--max-seconds",
        "0.5",
        "--threshold",
        "100",
    ]);
    let (spec, y) = read_wav(&out);
    assert_eq!(
        spec.sample_format,
        hound::SampleFormat::Float,
        "float source -> float output"
    );
    assert_eq!(y[0].len(), 24_000);
    // Threshold 100 never triggers: output equals the input prefix.
    assert_eq!(y[0], clicked[0][..24_000]);
}

#[test]
fn refuses_to_overwrite_input_and_rejects_garbage() {
    let dir = tmp();
    let inp = dir.path().join("in.wav");
    let (_, clicked, _) = signal(48_000, 0.5, 1);
    write_wav(&inp, &[&clicked[0], &clicked[1]], 48_000, 24, false);
    let before = std::fs::read(&inp).unwrap();
    let out = pressing(&["declick", s(&inp), s(&inp)]);
    assert!(!out.status.success());
    assert_eq!(std::fs::read(&inp).unwrap(), before, "input untouched");

    let junk = dir.path().join("junk.wav");
    std::fs::write(&junk, b"not audio at all").unwrap();
    let out = pressing(&["declick", s(&junk), s(&dir.path().join("o.wav"))]);
    assert!(!out.status.success());
}

// ---- parity / bench ----

#[test]
fn parity_on_committed_fixtures_passes() {
    let stdout = ok(&["parity", s(&fixtures_dir())]);
    println!("{stdout}");
    assert!(stdout.contains("parity: all checks passed"), "{stdout}");
    assert!(!stdout.contains("FAIL"), "{stdout}");
}

/// About 60 s in a debug build (under 1 s in release):
/// `cargo test -p pressing-cli --release -- --ignored`.
#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn parity_on_large_fixtures_passes_if_present() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures-large");
    if !dir.join("manifest.json").exists() {
        eprintln!("fixtures-large absent; skipping");
        return;
    }
    let stdout = ok(&["parity", s(&dir)]);
    println!("{stdout}");
    assert!(stdout.contains("parity: all checks passed"), "{stdout}");
}

#[test]
fn parity_fails_cleanly_on_missing_dir() {
    let dir = tmp();
    let out = pressing(&["parity", s(dir.path())]);
    assert!(!out.status.success());
}

#[test]
fn bench_runs() {
    let stdout = ok(&["bench", "--seconds", "0.25", "--block", "128"]);
    for key in ["engine", "network", "detector", "us/solve", "x real time"] {
        assert!(stdout.contains(key), "missing `{key}` in:\n{stdout}");
    }
}

// ---- Review regressions ----

#[test]
fn refuses_to_overwrite_input_through_a_hard_link() {
    let dir = tmp();
    let (inp, link) = (dir.path().join("in.wav"), dir.path().join("link.wav"));
    let (_, clicked, _) = signal(48_000, 0.5, 1);
    write_wav(&inp, &[&clicked[0], &clicked[1]], 48_000, 24, false);
    std::fs::hard_link(&inp, &link).unwrap();
    let before = std::fs::read(&inp).unwrap();
    let out = pressing(&["declick", s(&inp), s(&link)]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("refusing to overwrite"));
    assert_eq!(std::fs::read(&inp).unwrap(), before, "input untouched");
}

#[test]
fn bench_rejects_tiny_rates_instead_of_hanging() {
    let out = pressing(&["bench", "--seconds", "1", "--rate", "1"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--rate"));
}

/// Copy of the committed fixtures with `edit` applied to the manifest.
fn fixture_copy(edit: impl FnOnce(&mut serde_json::Value)) -> tempfile::TempDir {
    let dir = tmp();
    for e in std::fs::read_dir(fixtures_dir()).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dir.path().join(e.file_name())).unwrap();
    }
    let mp = dir.path().join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&mp).unwrap()).unwrap();
    edit(&mut m);
    std::fs::write(&mp, m.to_string()).unwrap();
    dir
}

fn parity_fails(dir: &Path, needle: &str) -> String {
    let out = pressing(&["parity", s(dir)]);
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "parity passed:\n{all}");
    assert!(!all.contains("panicked"), "{all}");
    assert!(all.contains(needle), "missing `{needle}` in:\n{all}");
    all
}

#[test]
fn parity_rejects_non_default_repair_settings() {
    let dir = fixture_copy(|m| {
        m["repair"]["order"] = 16.into();
        m["repair"]["context"] = 256.into();
    });
    parity_fails(dir.path(), "repair settings");
}

#[test]
fn parity_reports_missing_masks_without_panicking() {
    let dir = fixture_copy(|_| {});
    std::fs::remove_file(dir.path().join("masks.bin")).unwrap();
    parity_fails(dir.path(), "masks.bin");
}

#[test]
fn parity_fails_when_nothing_is_compared() {
    let dir = fixture_copy(|m| m["edge_ignore"] = 30_000.into());
    let all = parity_fails(dir.path(), "FAIL  masks");
    assert!(all.contains("of 0 compared samples"), "{all}");
    assert!(all.contains("FAIL  end-to-end"), "{all}");
}
