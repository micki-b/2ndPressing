//! Deterministic synthetic test material: music-like stereo with injected
//! clicks (used by `pressing bench`).

/// splitmix64: small deterministic PRNG (no dependencies).
pub struct Rng(pub u64);

impl Rng {
    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

/// Stereo test signal plus the injected click positions.
pub struct Synth {
    /// Left channel.
    pub left: Vec<f32>,
    /// Right channel.
    pub right: Vec<f32>,
    /// `(channel mask: 1 = L, 2 = R, 3 = both, start sample, length)`.
    pub clicks: Vec<(u8, usize, usize)>,
}

/// `seconds` of music-like stereo at `rate`: decaying harmonic notes (a new
/// note roughly every 0.25-0.6 s per voice, three voices) over low-passed noise
/// (surface hiss), peak around -10 dBFS, with about `clicks_per_sec` clicks
/// (decaying, ringing impulses of 2-40 samples, amplitude 0.05-0.5, 60% on both
/// channels).
pub fn music_with_clicks(seconds: f64, rate: u32, clicks_per_sec: f64, seed: u64) -> Synth {
    let n = (seconds * f64::from(rate)).round() as usize;
    let sr = f64::from(rate);
    let mut rng = Rng(seed);
    let mut left = vec![0.0f64; n];
    let mut right = vec![0.0f64; n];

    // ---- Voices ----
    for voice in 0..3 {
        let pan = 0.2 + 0.3 * voice as f64;
        let mut t0 = (rng.uniform() * 0.2 * sr) as usize;
        while t0 < n {
            // At least one sample so `t0` always advances (tiny rates).
            let dur = ((rng.range(0.25, 0.6) * sr) as usize).max(1);
            let midi = rng.range(40.0, 84.0).round();
            let f0 = 440.0 * 2f64.powf((midi - 69.0) / 12.0);
            let amp = rng.range(0.04, 0.09);
            let decay = rng.range(2.0, 8.0);
            let end = (t0 + dur).min(n);
            for i in t0..end {
                let t = (i - t0) as f64 / sr;
                let env = (t * 200.0).min(1.0) * (-decay * t).exp();
                let mut s = 0.0;
                for h in 1..=6 {
                    let fh = f0 * h as f64;
                    if fh < 0.45 * sr {
                        s += (2.0 * std::f64::consts::PI * fh * t).sin() / h as f64;
                    }
                }
                left[i] += amp * env * s * (1.0 - pan);
                right[i] += amp * env * s * pan;
            }
            t0 = end;
        }
    }

    // ---- Surface hiss (one-pole low-passed white noise) ----
    let (mut hl, mut hr) = (0.0, 0.0);
    for i in 0..n {
        hl = 0.9 * hl + 0.1 * rng.range(-1.0, 1.0);
        hr = 0.9 * hr + 0.1 * rng.range(-1.0, 1.0);
        left[i] += 0.01 * hl;
        right[i] += 0.01 * hr;
    }

    // ---- Clicks ----
    let mut clicks = Vec::new();
    let mean_gap = sr / clicks_per_sec.max(1e-9);
    let mut pos = (rng.range(0.2, 1.0) * mean_gap) as usize;
    while pos + 64 < n {
        let len = rng.range(2.0, 40.0) as usize;
        let amp = rng.range(0.05, 0.5) * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        let ring = rng.range(0.05, 0.45);
        let which: u8 = match rng.uniform() {
            u if u < 0.6 => 3,
            u if u < 0.8 => 1,
            _ => 2,
        };
        for k in 0..len {
            let x = k as f64;
            let v =
                amp * (-4.0 * x / len as f64).exp() * (2.0 * std::f64::consts::PI * ring * x).cos();
            if which & 1 != 0 {
                left[pos + k] += v;
            }
            if which & 2 != 0 {
                right[pos + k] += 0.8 * v;
            }
        }
        clicks.push((which, pos, len));
        pos += (rng.range(0.3, 1.7) * mean_gap) as usize + len;
    }

    Synth {
        left: left.iter().map(|&v| v as f32).collect(),
        right: right.iter().map(|&v| v as f32).collect(),
        clicks,
    }
}

#[cfg(test)]
mod tests {
    use super::music_with_clicks;

    #[test]
    fn tiny_rates_terminate() {
        for rate in [1, 2, 3, 100] {
            let s = music_with_clicks(5.0, rate, 4.0, 1);
            assert_eq!(s.left.len(), (5.0 * f64::from(rate)).round() as usize);
        }
    }
}
