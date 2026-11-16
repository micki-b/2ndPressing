//! Hand-off from the input device's clock to the output device's clock.
//!
//! The input callback writes processed stereo frames into a ring buffer at
//! the input rate; the output callback reads them through [`Resampler`].
//!
//! - **Locked** (input and output are the same device at the same rate, so
//!   one clock): the step is exactly 1 and every output frame is an input
//!   frame, bit for bit (4-point Hermite interpolation at `t = 0` returns the
//!   sample itself).
//! - **Free-running** (two devices, or two rates): the step is
//!   `in_rate / out_rate`, trimmed by at most [`MAX_TRIM`] (±1000 ppm, under
//!   2 cents) by a proportional controller that holds the ring's fill at the
//!   target. That absorbs the drift between two crystal clocks and converts
//!   44.1 <-> 48 kHz.
//!
//! On an underrun the output goes silent and waits until the ring is back at
//! its target fill (counted in [`Resampler::underruns`]). If the fill runs
//! far above the target (e.g. the output stalled), the oldest frames are
//! dropped down to the target (counted in [`Resampler::resyncs`]).
//! Nothing here allocates.

/// Largest step trim of the drift controller (fraction of the nominal step).
pub const MAX_TRIM: f64 = 1.0e-3;
/// Controller gain: trim per second of fill error.
const KP: f64 = 0.02;
/// Fill above the target that triggers a resync (seconds).
const RESYNC_EXCESS_S: f64 = 0.25;

/// See the module docs.
pub struct Resampler {
    /// Four-frame window; output lies between `hist[1]` and `hist[2]`.
    hist: [[f32; 2]; 4],
    /// Fractional position in `[0, 1)`.
    t: f64,
    nominal: f64,
    step: f64,
    locked: bool,
    in_rate: f64,
    fill_ema: f64,
    primed: bool,
    pub underruns: u64,
    pub resyncs: u64,
}

impl Resampler {
    /// `locked`: input and output share a clock (same device, same rate).
    pub fn new(in_rate: f64, out_rate: f64, locked: bool) -> Self {
        let nominal = if locked { 1.0 } else { in_rate / out_rate };
        Resampler {
            hist: [[0.0; 2]; 4],
            t: 0.0,
            nominal,
            step: nominal,
            locked,
            in_rate,
            fill_ema: 0.0,
            primed: false,
            underruns: 0,
            resyncs: 0,
        }
    }

    /// Current step (input frames per output frame).
    pub fn step(&self) -> f64 {
        self.step
    }

    /// Call once per output block with the ring's fill and target, in input
    /// frames. `pop` removes one frame from the ring.
    pub fn begin_block(
        &mut self,
        fill: usize,
        target: f64,
        mut pop: impl FnMut() -> Option<[f32; 2]>,
    ) {
        let fill_f = fill as f64;
        if !self.primed {
            if fill_f >= target {
                self.primed = true;
                self.fill_ema = fill_f;
                self.step = self.nominal;
            }
            return;
        }
        let excess = RESYNC_EXCESS_S * self.in_rate;
        if fill_f > target + excess {
            let drop = (fill_f - target) as usize;
            for _ in 0..drop {
                if pop().is_none() {
                    break;
                }
            }
            self.resyncs += 1;
            self.fill_ema = target;
            return;
        }
        if self.locked {
            return;
        }
        self.fill_ema += 0.05 * (fill_f - self.fill_ema);
        let err_s = (self.fill_ema - target) / self.in_rate;
        let trim = (KP * err_s).clamp(-MAX_TRIM, MAX_TRIM);
        self.step = self.nominal * (1.0 + trim);
    }

    /// Next output frame (silence while not primed).
    pub fn next(&mut self, mut pop: impl FnMut() -> Option<[f32; 2]>) -> [f32; 2] {
        if !self.primed {
            return [0.0; 2];
        }
        let t = self.t as f32;
        let h = &self.hist;
        let mut out = [0.0f32; 2];
        for (c, o) in out.iter_mut().enumerate() {
            let (y0, y1, y2, y3) = (h[0][c], h[1][c], h[2][c], h[3][c]);
            let c1 = 0.5 * (y2 - y0);
            let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
            let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
            *o = ((c3 * t + c2) * t + c1) * t + y1;
        }
        self.t += self.step;
        while self.t >= 1.0 {
            match pop() {
                Some(f) => {
                    self.hist = [self.hist[1], self.hist[2], self.hist[3], f];
                    self.t -= 1.0;
                }
                None => {
                    self.primed = false;
                    self.underruns += 1;
                    self.t = 0.0;
                    break;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn frame(i: usize) -> [f32; 2] {
        let x = (i as f32 * 0.37).sin() * 0.5;
        [x, -x * 0.7]
    }

    #[test]
    fn locked_is_a_bit_exact_delay() {
        let mut q: VecDeque<[f32; 2]> = VecDeque::new();
        let mut rs = Resampler::new(48_000.0, 48_000.0, true);
        let mut produced = 0;
        let mut out = Vec::new();
        for _ in 0..200 {
            for _ in 0..256 {
                q.push_back(frame(produced));
                produced += 1;
            }
            rs.begin_block(q.len(), 512.0, || q.pop_front());
            for _ in 0..256 {
                out.push(rs.next(|| q.pop_front()));
            }
        }
        assert_eq!(rs.underruns, 0);
        // Skip the priming silence, then every frame must be an input frame
        // in order, unaltered.
        let first = out.iter().position(|f| *f != [0.0; 2]).unwrap();
        let start = (0..produced).find(|&i| frame(i) == out[first]).unwrap();
        for (k, f) in out[first..].iter().enumerate() {
            assert_eq!(*f, frame(start + k), "frame {k}");
        }
    }

    /// Input clock 300 ppm fast against the output, plus 44.1 -> 48 kHz:
    /// the fill must stay bounded with no underruns once running.
    #[test]
    fn free_running_tracks_drift_and_rate() {
        let in_rate = 44_100.0 * 1.0003;
        let out_rate = 48_000.0;
        let mut rs = Resampler::new(44_100.0, out_rate, false);
        let mut q: VecDeque<[f32; 2]> = VecDeque::new();
        let (mut produced, mut acc) = (0usize, 0.0f64);
        let target = 1024.0;
        let mut max_fill = 0usize;
        for block in 0..20_000 {
            // Input arrives in proportion to elapsed output time.
            acc += 512.0 * in_rate / out_rate;
            while acc >= 1.0 {
                q.push_back(frame(produced));
                produced += 1;
                acc -= 1.0;
            }
            rs.begin_block(q.len(), target, || q.pop_front());
            for _ in 0..512 {
                rs.next(|| q.pop_front());
            }
            if block > 2_000 {
                max_fill = max_fill.max(q.len());
            }
        }
        assert!(rs.underruns <= 1, "underruns {}", rs.underruns);
        assert_eq!(rs.resyncs, 0);
        assert!(max_fill < 3_000, "fill {max_fill}");
        assert!((rs.step() / (44_100.0 / 48_000.0) - 1.0 - 3.0e-4).abs() < 1.0e-4);
    }

    /// Hermite interpolation of a 1 kHz sine at 44.1 -> 48 kHz stays clean.
    #[test]
    fn interpolation_accuracy() {
        let f = 1000.0;
        let src: Vec<[f32; 2]> = (0..44_100)
            .map(|i| {
                let x = (2.0 * std::f64::consts::PI * f * i as f64 / 44_100.0).sin() as f32;
                [x, x]
            })
            .collect();
        let mut it = src.into_iter();
        let mut rs = Resampler::new(44_100.0, 48_000.0, false);
        rs.begin_block(10, 0.0, || None);
        let mut worst = 0.0f64;
        for k in 0..40_000usize {
            let y = rs.next(|| it.next())[0] as f64;
            // Output k sits at input position k * step - 3: the window
            // starts as three frames of silence before the first input.
            let pos = k as f64 * 44_100.0 / 48_000.0 - 3.0;
            if k > 10 {
                let want = (2.0 * std::f64::consts::PI * f * pos / 44_100.0).sin();
                worst = worst.max((y - want).abs());
            }
        }
        assert!(worst < 2.0e-3, "worst error {worst}");
    }
}
