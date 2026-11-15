//! Input-format tests of `pressing declick`: ALAC/M4A (bit depth from the
//! magic cookie, gapless trimming from iTunSMPB and edit lists, corrupt
//! packets), AIFF, 32-bit integer WAV, and (when `afconvert`/`ffmpeg` are
//! installed) real FLAC, ALAC and AAC encodes.
//!
//! The ALAC and AIFF inputs are generated here (no audio files are committed).

use std::path::Path;
use std::process::Command;

mod common;

use common::{
    click_error, f32_to_int, have_tool, int_to_f32, ok, pressing, read_wav, s, signal, tmp,
    write_aiff, write_alac_m4a, write_wav, M4aOpts,
};

/// Stereo test material (clicked version) as `bits`-bit integers.
fn source(rate: u32, seconds: f64, bits: u32) -> Vec<Vec<i32>> {
    let n = (seconds * f64::from(rate)).round() as usize;
    let (_, clicked, _) = signal(rate, seconds.max(1.0), 3);
    clicked.iter().map(|c| f32_to_int(&c[..n], bits)).collect()
}

/// Runs `declick --mode bypass` and returns (stdout, stderr, output spec,
/// output channels).
fn bypass(inp: &Path, extra: &[&str]) -> (String, String, hound::WavSpec, Vec<Vec<f32>>) {
    let out = inp.with_extension("out.wav");
    let mut args = vec!["declick", s(inp), s(&out), "--mode", "bypass"];
    args.extend_from_slice(extra);
    let o = pressing(&args);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    );
    assert!(o.status.success(), "{args:?}\n{stdout}\n{stderr}");
    let (spec, y) = read_wav(&out);
    (stdout, stderr, spec, y)
}

// ---- ALAC in M4A (generated) ----

#[test]
fn alac24_stereo_bypass_is_bit_exact() {
    let dir = tmp();
    let p = dir.path().join("a.m4a");
    let src = source(48_000, 0.25, 24); // 12000 frames: 2 full packets + partial
    write_alac_m4a(&p, &src, 48_000, 24, 4096, &M4aOpts::default());
    let (stdout, _, spec, y) = bypass(&p, &[]);
    assert!(
        stdout.contains("(alac, 2 ch, 48000 Hz, 24-bit int"),
        "{stdout}"
    );
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (2, 48_000, 24)
    );
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    for c in 0..2 {
        assert_eq!(y[c], int_to_f32(&src[c], 24), "channel {c}");
    }
}

#[test]
fn alac16_mono_44k1_bypass_is_bit_exact_and_mono() {
    let dir = tmp();
    let p = dir.path().join("m.m4a");
    let src = vec![source(44_100, 0.2, 16).swap_remove(0)];
    write_alac_m4a(&p, &src, 44_100, 16, 4096, &M4aOpts::default());
    let (stdout, _, spec, y) = bypass(&p, &[]);
    assert!(
        stdout.contains("(alac, 1 ch, 44100 Hz, 16-bit int"),
        "{stdout}"
    );
    assert_eq!((spec.channels, spec.sample_rate), (1, 44_100));
    assert_eq!(y[0], int_to_f32(&src[0], 16));
}

#[test]
fn alac_itunsmpb_trims_delay_and_padding() {
    let dir = tmp();
    let p = dir.path().join("g.m4a");
    let src = source(48_000, 0.25, 24);
    let (skip, keep) = (2112usize, 8000usize);
    let smpb = format!(" 00000000 {skip:08X} 00000000 {keep:016X} 00000000 00000000");
    let opts = M4aOpts {
        itunsmpb: Some(smpb),
        ..Default::default()
    };
    write_alac_m4a(&p, &src, 48_000, 24, 4096, &opts);
    let (stdout, _, _, y) = bypass(&p, &[]);
    assert!(
        stdout.contains("gapless: iTunSMPB (skip 2112, keep 8000)"),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!(
            "dropped 2112 leading and {} trailing",
            12_000 - skip - keep
        )),
        "{stdout}"
    );
    for c in 0..2 {
        assert_eq!(y[c], int_to_f32(&src[c][skip..skip + keep], 24));
    }

    // --max-seconds counts from the first kept frame.
    let (_, _, _, y) = bypass(&p, &["--max-seconds", "0.1"]);
    assert_eq!(y[0], int_to_f32(&src[0][skip..skip + 4800], 24));
}

#[test]
fn alac_edit_list_trims_start_and_end() {
    let dir = tmp();
    let p = dir.path().join("e.m4a");
    let src = source(48_000, 0.25, 24);
    let (skip, keep) = (1024usize, 10_000usize);
    let opts = M4aOpts {
        elst: Some((keep as u32, skip as i32)),
        ..Default::default()
    };
    write_alac_m4a(&p, &src, 48_000, 24, 4096, &opts);
    let (stdout, _, _, y) = bypass(&p, &[]);
    assert!(
        stdout.contains("gapless: edit list (skip 1024, keep 10000)"),
        "{stdout}"
    );
    for c in 0..2 {
        assert_eq!(y[c], int_to_f32(&src[c][skip..skip + keep], 24));
    }
}

#[test]
fn alac_noop_gapless_info_changes_nothing() {
    // What Apple's encoder writes for ALAC: no delay, the exact frame count,
    // and padding relative to the packet-rounded length (must not be cut).
    let dir = tmp();
    let p = dir.path().join("n.m4a");
    let src = source(48_000, 0.25, 24);
    let opts = M4aOpts {
        itunsmpb: Some(" 00000000 00000000 00000900 0000000000002EE0".into()),
        elst: Some((12_000, 0)),
        ..Default::default()
    };
    write_alac_m4a(&p, &src, 48_000, 24, 4096, &opts);
    let (stdout, _, _, y) = bypass(&p, &[]);
    assert!(!stdout.contains("gapless:"), "{stdout}");
    assert_eq!(y[1], int_to_f32(&src[1], 24));
}

#[test]
fn corrupt_packet_is_replaced_by_silence_of_its_duration() {
    let dir = tmp();
    let p = dir.path().join("c.m4a");
    let src = source(48_000, 0.25, 24);
    let opts = M4aOpts {
        corrupt_packet: Some(1),
        ..Default::default()
    };
    write_alac_m4a(&p, &src, 48_000, 24, 4096, &opts);
    let (stdout, stderr, _, y) = bypass(&p, &[]);
    assert!(
        stdout.contains("1 undecodable packets replaced by 4096 frames of silence"),
        "{stdout}"
    );
    assert!(stderr.contains("undecodable packet at 0.085 s"), "{stderr}");
    for c in 0..2 {
        let want = int_to_f32(&src[c], 24);
        assert_eq!(y[c].len(), want.len(), "timeline length preserved");
        assert_eq!(y[c][..4096], want[..4096]);
        assert!(y[c][4096..8192].iter().all(|&v| v == 0.0));
        assert_eq!(y[c][8192..], want[8192..], "later audio stays aligned");
    }
}

#[test]
fn declick_on_alac_reduces_clicks() {
    let dir = tmp();
    let (p, out) = (dir.path().join("k.m4a"), dir.path().join("k.wav"));
    let (clean, clicked, starts) = signal(48_000, 1.0, 6);
    let src: Vec<Vec<i32>> = clicked.iter().map(|c| f32_to_int(c, 24)).collect();
    write_alac_m4a(&p, &src, 48_000, 24, 4096, &M4aOpts::default());
    ok(&["declick", s(&p), s(&out)]);
    let (spec, y) = read_wav(&out);
    assert_eq!((spec.channels, spec.bits_per_sample), (2, 24));
    for c in 0..2 {
        assert_eq!(y[c].len(), clicked[c].len());
        let before = click_error(&clicked[c], &clean[c], &starts);
        let after = click_error(&y[c], &clean[c], &starts);
        assert!(after < 0.1 * before, "ch{c}: {before:.4e} -> {after:.4e}");
    }
}

// ---- AIFF and WAV ----

#[test]
fn aiff24_bypass_is_bit_exact() {
    let dir = tmp();
    let p = dir.path().join("a.aiff");
    let src = source(48_000, 0.2, 24);
    write_aiff(&p, &src, 48_000, 24);
    let (stdout, _, spec, y) = bypass(&p, &[]);
    assert!(stdout.contains("2 ch, 48000 Hz, 24-bit int"), "{stdout}");
    assert_eq!((spec.sample_rate, spec.bits_per_sample), (48_000, 24));
    for c in 0..2 {
        assert_eq!(y[c], int_to_f32(&src[c], 24));
    }
}

#[test]
fn int32_wav_is_written_as_float_not_truncated_to_24_bits() {
    let dir = tmp();
    let p = dir.path().join("i32.wav");
    // Small values with low bits set: exact in f32, lost in 24-bit output.
    let v: Vec<i32> = (0..4800).map(|i| (i * 37 % 20_001) - 10_000).collect();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&p, spec).unwrap();
    for &x in &v {
        w.write_sample(x).unwrap();
    }
    w.finalize().unwrap();
    let (_, stderr, spec, y) = bypass(&p, &[]);
    assert!(stderr.contains("32-bit integer source"), "{stderr}");
    assert_eq!(
        (spec.sample_format, spec.bits_per_sample),
        (hound::SampleFormat::Float, 32)
    );
    let back: Vec<i32> = y[0]
        .iter()
        .map(|&x| (f64::from(x) * 2_147_483_648.0) as i32)
        .collect();
    assert_eq!(back, v);
}

// ---- Real encoders (skipped if the tools are absent) ----

/// Encodes `wav` to `out` with `afconvert` (macOS) or `ffmpeg`; `false` if
/// neither tool can produce it.
fn encode(wav: &Path, out: &Path, afconvert: &[&str], ffmpeg: &[&str]) -> bool {
    let run = |tool: &str, args: Vec<&str>| {
        Command::new(tool)
            .args(args)
            .output()
            .is_ok_and(|o| o.status.success())
            && out.exists()
    };
    if !afconvert.is_empty() && have_tool("afconvert", "-h") {
        let mut a = afconvert.to_vec();
        a.extend([s(wav), s(out)]);
        if run("afconvert", a) {
            return true;
        }
    }
    if !ffmpeg.is_empty() && have_tool("ffmpeg", "-version") {
        let mut a = vec!["-v", "error", "-y", "-i", s(wav)];
        a.extend_from_slice(ffmpeg);
        a.push(s(out));
        if run("ffmpeg", a) {
            return true;
        }
    }
    false
}

fn source_wav(dir: &Path) -> (std::path::PathBuf, Vec<Vec<i32>>) {
    let p = dir.join("src.wav");
    let src = source(48_000, 1.0, 24);
    let f: Vec<Vec<f32>> = src.iter().map(|c| int_to_f32(c, 24)).collect();
    write_wav(&p, &[&f[0], &f[1]], 48_000, 24, false);
    (p, src)
}

#[test]
fn real_lossless_encodes_bypass_bit_exact_if_tools_present() {
    let dir = tmp();
    let (wav, src) = source_wav(dir.path());
    let cases: [(&str, &[&str], &[&str]); 3] = [
        ("alac.m4a", &["-f", "m4af", "-d", "alac"], &["-c:a", "alac"]),
        (
            "flac.flac",
            &["-f", "flac", "-d", "flac"],
            &["-c:a", "flac"],
        ),
        ("aiff.aiff", &["-f", "AIFF", "-d", "BEI24"], &[]),
    ];
    for (name, af, ff) in cases {
        let p = dir.path().join(name);
        if !encode(&wav, &p, af, ff) {
            eprintln!("{name}: no encoder available; skipping");
            continue;
        }
        let (stdout, _, spec, y) = bypass(&p, &[]);
        assert!(stdout.contains("24-bit int"), "{name}: {stdout}");
        assert_eq!(spec.bits_per_sample, 24, "{name}");
        for c in 0..2 {
            assert_eq!(y[c], int_to_f32(&src[c], 24), "{name} channel {c}");
        }
    }
}

/// Normalised correlation of `a[i]` with `b[i + lag]`.
fn corr(a: &[f32], b: &[f32], lag: isize) -> f64 {
    let n = a.len().min(b.len()) as isize;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for i in 0.max(-lag)..n.min(n - lag) {
        let (x, y) = (f64::from(a[i as usize]), f64::from(b[(i + lag) as usize]));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt().max(1e-30)
}

#[test]
fn real_aac_encodes_are_aligned_and_same_length_if_tools_present() {
    let dir = tmp();
    let (wav, src) = source_wav(dir.path());
    let want = int_to_f32(&src[0], 24);
    // Apple's encoder writes iTunSMPB; ffmpeg writes an edit list.
    let cases: [(&str, &[&str], &[&str]); 2] = [
        (
            "apple.m4a",
            &["-f", "m4af", "-d", "aac", "-b", "256000"],
            &[],
        ),
        ("ffmpeg.m4a", &[], &["-c:a", "aac", "-b:a", "256k"]),
    ];
    for (name, af, ff) in cases {
        let p = dir.path().join(name);
        if !encode(&wav, &p, af, ff) {
            eprintln!("{name}: no encoder available; skipping");
            continue;
        }
        let (stdout, stderr, spec, y) = bypass(&p, &[]);
        assert!(stdout.contains("gapless:"), "{name}: {stdout}");
        assert!(!stderr.contains("warning"), "{name}: {stderr}");
        assert_eq!(spec.sample_format, hound::SampleFormat::Float);
        assert_eq!(y[0].len(), want.len(), "{name}: length");
        let c0 = corr(&want, &y[0], 0);
        println!("{name}: corr at lag 0 = {c0:.5}");
        assert!(c0 > 0.98, "{name}: corr at lag 0 = {c0}");
        for lag in [-2112, -1024, -64, 64, 1024, 2112] {
            let c = corr(&want, &y[0], lag);
            assert!(c < c0, "{name}: lag {lag} correlates better ({c} >= {c0})");
        }
    }
}
