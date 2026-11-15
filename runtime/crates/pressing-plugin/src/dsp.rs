//! Host-independent audio path of the plugin.
//!
//! [`Dsp`] wraps a [`Declicker`] that always runs in [`OutputMode::Declick`],
//! plus its own delay line of the raw input with the same length as the engine
//! latency. That gives both the repaired signal `y` and the delayed input `xd`
//! for every sample, so the output modes can be crossfaded instead of switched:
//!
//! - `Declick`: `y`
//! - `ClicksOnly`: `xd - y`
//! - `Bypass`: `xd`
//!
//! These are exactly the engine's own [`OutputMode`] outputs (the engine reads
//! its raw ring `latency` samples behind the input, as this delay line does),
//! so when no crossfade is running the output is bit-identical to calling
//! [`Declicker::set_mode`] with the same mode. Latency never changes with the
//! mode. Detection and repair keep running while bypassed, so turning the
//! plugin back on needs no warm-up.
//!
//! Nothing in [`Dsp::process`], [`Dsp::process_mono`], [`Dsp::set_output`],
//! [`Dsp::set_threshold`] or [`Dsp::reset`] allocates.

use pressing_core::{Declicker, Model, OutputMode, Protection, Stats};

/// Internal block size; host blocks of any length are split into chunks of at
/// most this many samples (scratch lives on the stack).
pub const CHUNK: usize = 256;

/// Crossfade time when the output mode changes (seconds).
pub const CROSSFADE_SECONDS: f32 = 0.010;

/// Mixing weights `[declick, clicks_only, bypass]` that produce `mode`.
fn weights(mode: OutputMode) -> [f32; 3] {
    match mode {
        OutputMode::Declick => [1.0, 0.0, 0.0],
        OutputMode::ClicksOnly => [0.0, 1.0, 0.0],
        OutputMode::Bypass => [0.0, 0.0, 1.0],
    }
}

/// Real-time audio processor used by the plugin (see the module docs).
pub struct Dsp {
    engine: Declicker,
    sample_rate: f32,
    latency: usize,
    /// Raw input delay lines (`latency` samples each).
    delay: [Vec<f32>; 2],
    pos: usize,
    /// Mode the output is heading to (or settled on when `ramp_left == 0`).
    target: OutputMode,
    /// Current mixing weights during a crossfade.
    gains: [f32; 3],
    /// Per-sample weight increments during a crossfade.
    step: [f32; 3],
    ramp_left: usize,
    ramp_len: usize,
    threshold: f32,
}

impl Dsp {
    /// Build the engine for `sample_rate` (allocates; call from
    /// `initialize`, never from the audio callback). Starts in
    /// [`OutputMode::Declick`] at the model's threshold.
    pub fn new(model: &Model, sample_rate: f32) -> Self {
        Self::new_with(model, sample_rate, Protection::Off)
    }

    /// [`Dsp::new`] with a percussion protection mode (adds ~1 s for
    /// Standard, ~2 s for Extended to the latency).
    pub fn new_with(model: &Model, sample_rate: f32, protection: Protection) -> Self {
        let mut engine = Declicker::new_with(model, sample_rate, protection);
        engine.set_mode(OutputMode::Declick);
        let latency = engine.latency_samples();
        let rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        // Bounded: rate is finite and positive here.
        let ramp_len = ((rate * CROSSFADE_SECONDS).round() as usize).max(1);
        Dsp {
            engine,
            sample_rate,
            latency,
            delay: [vec![0.0; latency], vec![0.0; latency]],
            pos: 0,
            target: OutputMode::Declick,
            gains: weights(OutputMode::Declick),
            step: [0.0; 3],
            ramp_left: 0,
            ramp_len,
            threshold: model.post.threshold,
        }
    }

    /// Sample rate this processor was built for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Percussion protection mode of this processor.
    pub fn protection(&self) -> Protection {
        self.engine.protection()
    }

    /// Fixed latency in samples (report to the host); independent of mode.
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// Current detection threshold (logit units).
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Engine counters since construction or the last [`Dsp::reset`].
    pub fn stats(&self) -> Stats {
        self.engine.stats()
    }

    /// Set the detection threshold; a no-op if unchanged.
    pub fn set_threshold(&mut self, threshold: f32) {
        if threshold != self.threshold {
            self.threshold = threshold;
            self.engine.set_threshold(threshold);
        }
    }

    /// Select the output mode; changes crossfade linearly over
    /// [`CROSSFADE_SECONDS`] (also when retargeted mid-fade).
    pub fn set_output(&mut self, mode: OutputMode) {
        if mode == self.target {
            return;
        }
        self.target = mode;
        let w = weights(mode);
        let n = self.ramp_len as f32;
        for k in 0..3 {
            self.step[k] = (w[k] - self.gains[k]) / n;
        }
        self.ramp_left = self.ramp_len;
    }

    /// Clear all signal state and engine stats; keeps threshold and mode
    /// (any running crossfade is completed instantly).
    pub fn reset(&mut self) {
        self.engine.reset();
        for d in &mut self.delay {
            d.fill(0.0);
        }
        self.pos = 0;
        self.gains = weights(self.target);
        self.ramp_left = 0;
    }

    /// Process a stereo block in place (any length; only the first
    /// `min(left.len(), right.len())` samples are processed).
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len().min(right.len());
        for (l, r) in left[..n]
            .chunks_mut(CHUNK)
            .zip(right[..n].chunks_mut(CHUNK))
        {
            self.process_chunk(l, r);
        }
    }

    /// Process a mono block in place: the signal is fed to both engine
    /// channels (the model's cross-channel feature then sees the same signal)
    /// and the left result is written back.
    pub fn process_mono(&mut self, x: &mut [f32]) {
        let mut copy = [0.0f32; CHUNK];
        for c in x.chunks_mut(CHUNK) {
            let r = &mut copy[..c.len()];
            r.copy_from_slice(c);
            self.process_chunk(c, r);
        }
    }

    fn process_chunk(&mut self, l: &mut [f32], r: &mut [f32]) {
        debug_assert!(l.len() == r.len() && l.len() <= CHUNK);
        let mut xd = [[0.0f32; CHUNK]; 2];
        let mut pos = self.pos;
        for (i, (&a, &b)) in l.iter().zip(r.iter()).enumerate() {
            xd[0][i] = std::mem::replace(&mut self.delay[0][pos], a);
            xd[1][i] = std::mem::replace(&mut self.delay[1][pos], b);
            pos += 1;
            if pos == self.latency {
                pos = 0;
            }
        }
        self.pos = pos;

        self.engine.process(l, r);

        for i in 0..l.len() {
            let (dl, dr) = (xd[0][i], xd[1][i]);
            if self.ramp_left == 0 {
                // Settled: exactly the engine's own mode outputs.
                match self.target {
                    OutputMode::Declick => {}
                    OutputMode::ClicksOnly => {
                        l[i] = dl - l[i];
                        r[i] = dr - r[i];
                    }
                    OutputMode::Bypass => {
                        l[i] = dl;
                        r[i] = dr;
                    }
                }
            } else {
                let [gd, gc, gb] = self.gains;
                l[i] = gd * l[i] + gc * (dl - l[i]) + gb * dl;
                r[i] = gd * r[i] + gc * (dr - r[i]) + gb * dr;
                self.ramp_left -= 1;
                if self.ramp_left == 0 {
                    self.gains = weights(self.target);
                } else {
                    for k in 0..3 {
                        self.gains[k] += self.step[k];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise with sparse impulsive clicks (LCG; no deps).
    fn test_signal(len: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut l = vec![0.0f32; len];
        let mut r = vec![0.0f32; len];
        let (mut pl, mut pr) = (0.0f32, 0.0f32);
        for i in 0..len {
            // Low-passed noise, roughly music-like spectrum.
            pl = 0.97 * pl + 0.05 * next();
            pr = 0.97 * pr + 0.05 * next();
            l[i] = pl;
            r[i] = pr;
        }
        let mut i = 3000;
        while i + 8 < len {
            for k in 0..4 {
                let a = 0.6 * (1.0 - k as f32 / 4.0);
                l[i + k] += a;
                r[i + k + 1] -= 0.8 * a;
            }
            i += 2311;
        }
        (l, r)
    }

    fn run_dsp(dsp: &mut Dsp, l: &[f32], r: &[f32], blocks: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let (mut ol, mut or) = (l.to_vec(), r.to_vec());
        let mut at = 0;
        let mut b = 0;
        while at < ol.len() {
            let n = blocks[b % blocks.len()].min(ol.len() - at);
            dsp.process(&mut ol[at..at + n], &mut or[at..at + n]);
            at += n;
            b += 1;
        }
        (ol, or)
    }

    fn run_engine(model: &Model, mode: OutputMode, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut e = Declicker::new(model, 48_000.0);
        e.set_mode(mode);
        let (mut ol, mut or) = (l.to_vec(), r.to_vec());
        e.process(&mut ol, &mut or);
        (ol, or)
    }

    #[test]
    fn settled_modes_match_engine_bit_exact() {
        let model = Model::embedded();
        let (l, r) = test_signal(20_000, 7);
        for mode in [
            OutputMode::Declick,
            OutputMode::ClicksOnly,
            OutputMode::Bypass,
        ] {
            let mut dsp = Dsp::new(&model, 48_000.0);
            dsp.set_output(mode);
            dsp.reset(); // completes the fade: settled from sample 0
            let got = run_dsp(&mut dsp, &l, &r, &[1, 37, 256, 1000, 64]);
            let want = run_engine(&model, mode, &l, &r);
            assert!(got.0 == want.0 && got.1 == want.1, "{mode:?} differs");
            if mode == OutputMode::Declick {
                assert!(dsp.stats().regions > 0, "test signal produced no regions");
            }
        }
    }

    #[test]
    fn crossfade_is_bounded_and_settles() {
        let model = Model::embedded();
        let (l, r) = test_signal(20_000, 11);
        let dec = run_engine(&model, OutputMode::Declick, &l, &r);
        let byp = run_engine(&model, OutputMode::Bypass, &l, &r);

        let mut dsp = Dsp::new(&model, 48_000.0);
        let (mut ol, mut or) = (l.clone(), r.clone());
        // The ramp's output window [switch, switch + 480) is input
        // [switch - 1536, ...), which contains the click at input 9933.
        let switch = 11_300;
        dsp.process(&mut ol[..switch], &mut or[..switch]);
        dsp.set_output(OutputMode::Bypass);
        dsp.process(&mut ol[switch..], &mut or[switch..]);

        let ramp = 480;
        assert!(
            (switch..switch + ramp).any(|i| dec.0[i] != byp.0[i]),
            "no repair inside the crossfade window; test is vacuous"
        );
        assert_eq!(&ol[..switch], &dec.0[..switch]);
        assert_eq!(&ol[switch + ramp..], &byp.0[switch + ramp..]);
        assert_eq!(&or[switch + ramp..], &byp.1[switch + ramp..]);
        for i in switch..switch + ramp {
            for (o, a, b) in [(ol[i], dec.0[i], byp.0[i]), (or[i], dec.1[i], byp.1[i])] {
                let (lo, hi) = (a.min(b), a.max(b));
                assert!(
                    o >= lo - 1e-6 && o <= hi + 1e-6,
                    "sample {i}: {o} not in [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn mono_matches_duplicated_stereo() {
        let model = Model::embedded();
        let (l, _) = test_signal(6_000, 3);
        let mut a = Dsp::new(&model, 48_000.0);
        let mut b = Dsp::new(&model, 48_000.0);
        let mut m = l.clone();
        for c in m.chunks_mut(300) {
            a.process_mono(c);
        }
        let (sl, sr) = run_dsp(&mut b, &l, &l, &[300]);
        assert_eq!(m, sl);
        assert_eq!(sl, sr);
    }

    #[test]
    fn latency_per_sample_rate() {
        let model = Model::embedded();
        for (sr, want) in [(44_100.0, 1536), (48_000.0, 1536)] {
            assert_eq!(Dsp::new(&model, sr).latency_samples(), want);
        }
        // Experimental decimated path: 1536 * factor + FIR delay (32 * factor).
        assert_eq!(Dsp::new(&model, 96_000.0).latency_samples(), 2 * 1536 + 64);
    }
}
