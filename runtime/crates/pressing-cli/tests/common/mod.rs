//! Helpers shared by the integration tests: running the binary, synthetic
//! signals, and minimal writers for WAV, AIFF and ALAC-in-M4A input files.

#![allow(dead_code)] // each test crate uses a subset

use std::path::Path;
use std::process::{Command, Output};

// ---- Running the binary ----

pub fn pressing(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pressing"))
        .args(args)
        .output()
        .expect("run pressing")
}

pub fn ok(args: &[&str]) -> String {
    let out = pressing(args);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "pressing {args:?} failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

pub fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 temp path")
}

/// splitmix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Clean music-like stereo (two decaying harmonic tones plus low hiss) and the
/// same signal with `n_clicks` decaying impulses (both channels), quantised to
/// 24 bits. Returns `(clean, clicked, click starts)`, planar.
pub fn signal(
    rate: u32,
    seconds: f64,
    n_clicks: usize,
) -> ([Vec<f32>; 2], [Vec<f32>; 2], Vec<usize>) {
    let n = (seconds * f64::from(rate)) as usize;
    let sr = f64::from(rate);
    let mut rng = Rng(42);
    let mut clean = [vec![0.0f64; n], vec![0.0f64; n]];
    let (mut hl, mut hr) = (0.0, 0.0);
    for i in 0..n {
        let t = i as f64 / sr;
        let note = |f: f64, period: f64| {
            let tn = t % period;
            let env = (tn * 100.0).min(1.0) * (-3.0 * tn).exp();
            (1..=4)
                .map(|h| (2.0 * std::f64::consts::PI * f * h as f64 * t).sin() / h as f64)
                .sum::<f64>()
                * env
        };
        hl = 0.9 * hl + 0.1 * (rng.uniform() * 2.0 - 1.0);
        hr = 0.9 * hr + 0.1 * (rng.uniform() * 2.0 - 1.0);
        let a = note(220.0, 0.7);
        let b = note(330.0, 0.45);
        clean[0][i] = 0.08 * a + 0.04 * b + 0.005 * hl;
        clean[1][i] = 0.04 * a + 0.08 * b + 0.005 * hr;
    }
    let mut clicked = clean.clone();
    let mut starts = Vec::new();
    for k in 0..n_clicks {
        // Evenly spread, away from the first 0.25 s (start-up) and the end.
        let pos = (0.25 * sr) as usize + k * (n - (0.35 * sr) as usize) / n_clicks;
        let amp = if k % 2 == 0 { 0.3 } else { -0.25 };
        let len = 6 + 3 * (k % 5);
        for j in 0..len {
            let v = amp * (-4.0 * j as f64 / len as f64).exp() * (0.9 * j as f64).cos();
            clicked[0][pos + j] += v;
            clicked[1][pos + j] += 0.8 * v;
        }
        starts.push(pos);
    }
    let q = |v: &Vec<f64>| -> Vec<f32> {
        v.iter()
            .map(|&x| ((x * 8_388_608.0).round() / 8_388_608.0) as f32)
            .collect()
    };
    (
        [q(&clean[0]), q(&clean[1])],
        [q(&clicked[0]), q(&clicked[1])],
        starts,
    )
}

pub fn write_wav(path: &Path, chans: &[&[f32]], rate: u32, bits: u16, float: bool) {
    let spec = hound::WavSpec {
        channels: chans.len() as u16,
        sample_rate: rate,
        bits_per_sample: bits,
        sample_format: if float {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create wav");
    let scale = f64::from(1u32 << (bits - 1));
    for i in 0..chans[0].len() {
        for c in chans {
            if float {
                w.write_sample(c[i]).unwrap();
            } else {
                let v = (f64::from(c[i]) * scale).round().clamp(-scale, scale - 1.0);
                w.write_sample(v as i32).unwrap();
            }
        }
    }
    w.finalize().unwrap();
}

/// Reads a WAV as planar f32 (integers scaled by `2^-(bits-1)`).
pub fn read_wav(path: &Path) -> (hound::WavSpec, Vec<Vec<f32>>) {
    let mut r = hound::WavReader::open(path).expect("open output wav");
    let spec = r.spec();
    let nch = usize::from(spec.channels);
    let inter: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = f64::from(1u32 << (spec.bits_per_sample - 1));
            r.samples::<i32>()
                .map(|v| (f64::from(v.unwrap()) / scale) as f32)
                .collect()
        }
    };
    let chans = (0..nch)
        .map(|c| inter.iter().skip(c).step_by(nch).copied().collect())
        .collect();
    (spec, chans)
}

/// Energy of `x - clean` in `[start - 16, start + 64)` around every click.
pub fn click_error(x: &[f32], clean: &[f32], starts: &[usize]) -> f64 {
    starts
        .iter()
        .flat_map(|&p| p - 16..p + 64)
        .map(|i| (f64::from(x[i]) - f64::from(clean[i])).powi(2))
        .sum()
}

pub fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir")
}

/// Integer samples of `bits` bits as f32 in [-1, 1) (`v / 2^(bits-1)`, exact
/// up to 24 bits).
pub fn int_to_f32(v: &[i32], bits: u32) -> Vec<f32> {
    let scale = f64::from(1u32 << (bits - 1));
    v.iter().map(|&x| (f64::from(x) / scale) as f32).collect()
}

/// `x` (in [-1, 1)) quantised to `bits`-bit integers.
pub fn f32_to_int(x: &[f32], bits: u32) -> Vec<i32> {
    let scale = f64::from(1u32 << (bits - 1));
    x.iter()
        .map(|&v| (f64::from(v) * scale).round().clamp(-scale, scale - 1.0) as i32)
        .collect()
}

// ---- AIFF writer ----

/// 80-bit IEEE extended encoding of a positive integer sample rate.
fn ext80(rate: u32) -> [u8; 10] {
    let e = 31 - rate.leading_zeros();
    let mut out = [0u8; 10];
    out[..2].copy_from_slice(&((16_383 + e) as u16).to_be_bytes());
    out[2..].copy_from_slice(&((u64::from(rate)) << (63 - e)).to_be_bytes());
    out
}

/// Writes big-endian integer PCM AIFF (`bits` = 16 or 24).
pub fn write_aiff(path: &Path, chans: &[Vec<i32>], rate: u32, bits: u16) {
    let (nch, n) = (chans.len(), chans[0].len());
    let bps = usize::from(bits / 8);
    let mut ssnd = vec![0u8; 8];
    for i in 0..n {
        for c in chans {
            ssnd.extend_from_slice(&c[i].to_be_bytes()[4 - bps..]);
        }
    }
    let mut comm = Vec::new();
    comm.extend_from_slice(&(nch as u16).to_be_bytes());
    comm.extend_from_slice(&(n as u32).to_be_bytes());
    comm.extend_from_slice(&bits.to_be_bytes());
    comm.extend_from_slice(&ext80(rate));
    let chunk = |id: &[u8; 4], body: &[u8]| {
        let mut v = id.to_vec();
        v.extend_from_slice(&(body.len() as u32).to_be_bytes());
        v.extend_from_slice(body);
        if body.len() % 2 == 1 {
            v.push(0);
        }
        v
    };
    let body = [
        b"AIFF".to_vec(),
        chunk(b"COMM", &comm),
        chunk(b"SSND", &ssnd),
    ]
    .concat();
    std::fs::write(path, chunk(b"FORM", &body)).expect("write aiff");
}

// ---- ALAC in M4A writer ----

/// Options of [`write_alac_m4a`].
#[derive(Default)]
pub struct M4aOpts {
    /// Edit list entry `(segment_duration, media_time)` (timescales = rate).
    pub elst: Option<(u32, i32)>,
    /// `iTunSMPB` tag value.
    pub itunsmpb: Option<String>,
    /// Index of a packet to corrupt (nonzero "unused" header bits, which
    /// symphonia rejects with a decode error).
    pub corrupt_packet: Option<usize>,
}

/// MSB-first bit writer.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn new() -> Self {
        Bits {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }
    fn put(&mut self, v: u32, bits: u32) {
        for k in (0..bits).rev() {
            self.acc = (self.acc << 1) | u64::from((v >> k) & 1);
            self.n += 1;
            if self.n == 8 {
                self.out.push(self.acc as u8);
                self.acc = 0;
                self.n = 0;
            }
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put(0, pad);
        }
        self.out
    }
}

fn atom(t: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    v.extend_from_slice(t);
    v.extend_from_slice(body);
    v
}

fn full_atom(t: &[u8; 4], flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = flags.to_be_bytes().to_vec(); // version 0 + flags
    b.extend_from_slice(body);
    atom(t, &b)
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// Writes an ALAC `.m4a` (mono or stereo, `bits` = 16 or 24) whose packets
/// are ALAC "escape" (uncompressed) frames of `frame_len` frames; the last
/// packet is a partial frame. Optional edit list, iTunSMPB tag and corrupted
/// packet.
pub fn write_alac_m4a(
    path: &Path,
    chans: &[Vec<i32>],
    rate: u32,
    bits: u8,
    frame_len: u32,
    opts: &M4aOpts,
) {
    let (nch, n) = (chans.len(), chans[0].len());
    assert!(nch == 1 || nch == 2);
    let mask = (1u64 << bits) - 1;

    // ---- Packets ----
    let mut packets = Vec::new();
    for (k, start) in (0..n).step_by(frame_len as usize).enumerate() {
        let len = (n - start).min(frame_len as usize);
        let partial = len < frame_len as usize;
        let mut b = Bits::new();
        b.put(if nch == 2 { 1 } else { 0 }, 3); // CPE / SCE
        b.put(0, 4); // instance tag
        b.put(u32::from(opts.corrupt_packet == Some(k)), 12); // unused, must be 0
        b.put(u32::from(partial), 1);
        b.put(0, 2); // bytes shifted
        b.put(1, 1); // escape: uncompressed
        if partial {
            b.put(len as u32, 32);
        }
        for i in start..start + len {
            for c in chans {
                b.put((i64::from(c[i]) as u64 & mask) as u32, u32::from(bits));
            }
        }
        b.put(7, 3); // END
        packets.push((len as u32, b.finish()));
    }

    // ---- moov ----
    let n32 = n as u32;
    let mut mvhd = vec![0u8; 8];
    mvhd.extend_from_slice(&be32(rate));
    mvhd.extend_from_slice(&be32(n32));
    mvhd.extend_from_slice(&be32(0x0001_0000));
    mvhd.extend_from_slice(&[1, 0]);
    mvhd.extend_from_slice(&[0; 10]);
    mvhd.extend_from_slice(&[0; 36]);
    mvhd.extend_from_slice(&[0; 24]);
    mvhd.extend_from_slice(&be32(2));
    let mvhd = full_atom(b"mvhd", 0, &mvhd);

    let mut tkhd = vec![0u8; 8];
    tkhd.extend_from_slice(&be32(1));
    tkhd.extend_from_slice(&[0; 4]);
    tkhd.extend_from_slice(&be32(n32));
    tkhd.extend_from_slice(&[0; 8]);
    tkhd.extend_from_slice(&[0, 0, 0, 0, 1, 0, 0, 0]);
    tkhd.extend_from_slice(&[0; 36]);
    tkhd.extend_from_slice(&[0; 8]);
    let tkhd = full_atom(b"tkhd", 7, &tkhd);

    let mut mdhd = vec![0u8; 8];
    mdhd.extend_from_slice(&be32(rate));
    mdhd.extend_from_slice(&be32(n32));
    mdhd.extend_from_slice(&[0x55, 0xc4, 0, 0]);
    let mdhd = full_atom(b"mdhd", 0, &mdhd);
    let mut hdlr = vec![0u8; 4];
    hdlr.extend_from_slice(b"soun");
    hdlr.extend_from_slice(&[0; 13]);
    let hdlr = full_atom(b"hdlr", 0, &hdlr);

    let mut cookie = Vec::new();
    cookie.extend_from_slice(&be32(frame_len));
    cookie.extend_from_slice(&[0, bits, 40, 10, 14, nch as u8]);
    cookie.extend_from_slice(&255u16.to_be_bytes());
    let max_bytes = packets.iter().map(|p| p.1.len()).max().unwrap_or(0) as u32;
    cookie.extend_from_slice(&be32(max_bytes));
    cookie.extend_from_slice(&be32(0));
    cookie.extend_from_slice(&be32(rate));
    let mut entry = vec![0u8; 6];
    entry.extend_from_slice(&[0, 1]); // data reference index
    entry.extend_from_slice(&[0; 8]); // version, revision, vendor
    entry.extend_from_slice(&(nch as u16).to_be_bytes());
    entry.extend_from_slice(&u16::from(bits).to_be_bytes());
    entry.extend_from_slice(&[0; 4]); // compression id, packet size
    entry.extend_from_slice(&be32(rate << 16)); // 16.16 (rates < 65536)
    entry.extend_from_slice(&full_atom(b"alac", 0, &cookie));
    let mut stsd = be32(1).to_vec();
    stsd.extend_from_slice(&atom(b"alac", &entry));
    let stsd = full_atom(b"stsd", 0, &stsd);

    let full = packets.iter().filter(|p| p.0 == frame_len).count() as u32;
    let mut stts = Vec::new();
    let mut runs = vec![(full, frame_len)];
    if let Some(last) = packets.last().filter(|p| p.0 != frame_len) {
        runs.push((1, last.0));
    }
    runs.retain(|r| r.0 > 0);
    stts.extend_from_slice(&be32(runs.len() as u32));
    for (count, d) in runs {
        stts.extend_from_slice(&be32(count));
        stts.extend_from_slice(&be32(d));
    }
    let stts = full_atom(b"stts", 0, &stts);
    let mut stsc = be32(1).to_vec();
    stsc.extend_from_slice(&be32(1));
    stsc.extend_from_slice(&be32(packets.len() as u32));
    stsc.extend_from_slice(&be32(1));
    let stsc = full_atom(b"stsc", 0, &stsc);
    let mut stsz = be32(0).to_vec();
    stsz.extend_from_slice(&be32(packets.len() as u32));
    for p in &packets {
        stsz.extend_from_slice(&be32(p.1.len() as u32));
    }
    let stsz = full_atom(b"stsz", 0, &stsz);

    let ftyp = atom(b"ftyp", b"M4A \0\0\0\0M4A mp42isom");
    let mdat_body: Vec<u8> = packets.iter().flat_map(|p| p.1.clone()).collect();
    let mdat = atom(b"mdat", &mdat_body);
    let mut stco = be32(1).to_vec();
    stco.extend_from_slice(&be32((ftyp.len() + 8) as u32));
    let stco = full_atom(b"stco", 0, &stco);

    let stbl = atom(b"stbl", &[stsd, stts, stsc, stsz, stco].concat());
    let smhd = full_atom(b"smhd", 0, &[0; 4]);
    let dref = full_atom(
        b"dref",
        0,
        &[be32(1).to_vec(), full_atom(b"url ", 1, &[])].concat(),
    );
    let minf = atom(b"minf", &[smhd, atom(b"dinf", &dref), stbl].concat());
    let mdia = atom(b"mdia", &[mdhd, hdlr, minf].concat());
    let mut trak = tkhd;
    if let Some((dur, time)) = opts.elst {
        let mut e = be32(1).to_vec();
        e.extend_from_slice(&be32(dur));
        e.extend_from_slice(&time.to_be_bytes());
        e.extend_from_slice(&[0, 1, 0, 0]);
        trak.extend_from_slice(&atom(b"edts", &full_atom(b"elst", 0, &e)));
    }
    trak.extend_from_slice(&mdia);
    let mut moov = [mvhd, atom(b"trak", &trak)].concat();
    if let Some(v) = &opts.itunsmpb {
        let mut data = be32(1).to_vec(); // UTF-8
        data.extend_from_slice(&be32(0));
        data.extend_from_slice(v.as_bytes());
        let freeform = atom(
            b"----",
            &[
                full_atom(b"mean", 0, b"com.apple.iTunes"),
                full_atom(b"name", 0, b"iTunSMPB"),
                atom(b"data", &data),
            ]
            .concat(),
        );
        let mut mhdlr = vec![0u8; 4];
        mhdlr.extend_from_slice(b"mdirappl");
        mhdlr.extend_from_slice(&[0; 9]);
        let meta = full_atom(
            b"meta",
            0,
            &[full_atom(b"hdlr", 0, &mhdlr), atom(b"ilst", &freeform)].concat(),
        );
        moov.extend_from_slice(&atom(b"udta", &meta));
    }
    let file = [ftyp, mdat, atom(b"moov", &moov)].concat();
    std::fs::write(path, file).expect("write m4a");
}

/// `true` if `tool` can be run (external encoders are optional).
pub fn have_tool(tool: &str, probe_arg: &str) -> bool {
    Command::new(tool)
        .arg(probe_arg)
        .output()
        .is_ok_and(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
}
