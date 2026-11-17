//! Causal gain-invariant input features (spec section 1).
//!
//! Per channel: `d1[n] = x[n] - x[n-1]`, `d2[n] = d1[n] - d1[n-1]` (zero history),
//! `l_k = ln(|d_k| + EPS)`, `m_k[n]` = mean of `l_k` over the last `min(n+1, NW)`
//! samples, `s_k = max(exp(m_k), FLOOR)`, `g_k = asinh(d_k / s_k / ASINH_SCALE)`.
//! Everything is computed in f64; only the returned model input is f32.

use crate::model::FeatureConfig;

/// Exact mean over the last `min(n+1, NW)` pushed values.
///
/// Keeps a ring of the last NW values and an f64 running sum. To stop rounding
/// drift from accumulating over long streams, the sum is recomputed from the
/// ring every time the write position wraps (amortised O(1), no allocation).
struct WindowMean {
    ring: Vec<f64>,
    pos: usize,
    count: usize,
    sum: f64,
}

impl WindowMean {
    fn new(nw: usize) -> Self {
        assert!(nw >= 1, "window mean needs NW >= 1");
        WindowMean {
            ring: vec![0.0; nw],
            pos: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn push(&mut self, v: f64) -> f64 {
        let nw = self.ring.len();
        if self.count == nw {
            self.sum -= self.ring[self.pos];
        } else {
            self.count += 1;
        }
        self.ring[self.pos] = v;
        self.sum += v;
        self.pos += 1;
        if self.pos == nw {
            self.pos = 0;
            if self.count == nw {
                self.sum = self.ring.iter().sum();
            }
        }
        self.sum / self.count as f64
    }

    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.pos = 0;
        self.count = 0;
        self.sum = 0.0;
    }
}

/// Per-channel feature state: difference history and two window means.
struct ChannelFeatures {
    prev_x: f64,
    prev_d1: f64,
    m1: WindowMean,
    m2: WindowMean,
}

impl ChannelFeatures {
    fn new(nw: usize) -> Self {
        ChannelFeatures {
            prev_x: 0.0,
            prev_d1: 0.0,
            m1: WindowMean::new(nw),
            m2: WindowMean::new(nw),
        }
    }

    fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_d1 = 0.0;
        self.m1.reset();
        self.m2.reset();
    }
}

/// Streaming feature state for one stereo stream.
///
/// Allocates its rings (`2 channels x 2 orders x NW` f64) in [`FeatureState::new`];
/// [`FeatureState::push`] never allocates.
pub struct FeatureState {
    eps: f64,
    floor: f64,
    asinh_scale: f64,
    ch: [ChannelFeatures; 2],
}

impl FeatureState {
    /// Fresh state (zero history) for the given feature constants.
    ///
    /// # Panics
    /// If `cfg` fails [`FeatureConfig::validate`] (e.g. `NW == 0`, `EPS <= 0`).
    /// Configs from [`crate::model::Model::from_json_and_bin`] are already validated.
    pub fn new(cfg: &FeatureConfig) -> Self {
        if let Err(e) = cfg.validate() {
            panic!("invalid feature config: {e}");
        }
        FeatureState {
            eps: cfg.eps,
            floor: cfg.floor,
            asinh_scale: cfg.asinh_scale,
            ch: [ChannelFeatures::new(cfg.nw), ChannelFeatures::new(cfg.nw)],
        }
    }

    #[inline]
    fn g(&self, d: f64, m: f64) -> f64 {
        let s = m.exp().max(self.floor);
        (d / s / self.asinh_scale).asinh()
    }

    /// `(g1, g2)` for one channel and one new sample.
    #[inline]
    fn channel(&mut self, c: usize, x: f64) -> (f64, f64) {
        let eps = self.eps;
        let st = &mut self.ch[c];
        let d1 = x - st.prev_x;
        let d2 = d1 - st.prev_d1;
        st.prev_x = x;
        st.prev_d1 = d1;
        let m1 = st.m1.push((d1.abs() + eps).ln());
        let m2 = st.m2.push((d2.abs() + eps).ln());
        (self.g(d1, m1), self.g(d2, m2))
    }

    /// Push one stereo sample; returns model input per channel:
    /// `[[g1_L, g2_L, g1_R], [g1_R, g2_R, g1_L]]`.
    pub fn push(&mut self, l: f32, r: f32) -> [[f32; 3]; 2] {
        let (g1l, g2l) = self.channel(0, l as f64);
        let (g1r, g2r) = self.channel(1, r as f64);
        let (g1l, g2l, g1r, g2r) = (g1l as f32, g2l as f32, g1r as f32, g2r as f32);
        [[g1l, g2l, g1r], [g1r, g2r, g1l]]
    }

    /// Back to the start-of-stream state (zero history).
    pub fn reset(&mut self) {
        for c in &mut self.ch {
            c.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WindowMean;

    #[test]
    fn window_mean_matches_direct_mean() {
        let nw = 5;
        let mut w = WindowMean::new(nw);
        let vals: Vec<f64> = (0..37).map(|i| ((i * 7919) % 13) as f64 - 6.5).collect();
        for (n, &v) in vals.iter().enumerate() {
            let got = w.push(v);
            let lo = (n + 1).saturating_sub(nw);
            let want = vals[lo..=n].iter().sum::<f64>() / (n + 1 - lo) as f64;
            assert!((got - want).abs() < 1e-12, "n={n}: {got} vs {want}");
        }
    }
}
