//! Streaming hysteresis region finder (spec section 3).
//!
//! [`RegionFinder`] is the per-channel streaming equivalent of
//! `classical.regions_from_scores`: it consumes target-aligned scores one
//! sample at a time and emits each triggered, padded region as soon as its
//! weak group is finalised. [`RegionFinder::frontier`] tells the engine how far
//! the region mask is final.

use crate::model::PostConfig;

/// A finalised region `[start, end)` in target-sample coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// First sample of the region (`max(0, a - pre)`).
    pub start: u64,
    /// One past the last sample of the region (`e + 1 + post`).
    pub end: u64,
}

/// Per-channel streaming equivalent of `classical.regions_from_scores`.
///
/// Thresholds are compared in f64, like the reference (which compares
/// float64 scores against Python floats).
#[derive(Clone, Debug)]
pub struct RegionFinder {
    threshold: f64,
    weak: f64,
    ext_drop: f64,
    merge: u64,
    pre: u64,
    post: u64,
    max_region: u64,
    /// Next unseen target index.
    t_next: u64,
    /// Open weak group: first weak index, last weak index, peak score.
    open: Option<(u64, u64, f32)>,
}

impl RegionFinder {
    /// New finder with the thresholds and padding of `cfg`.
    pub fn new(cfg: &PostConfig) -> Self {
        let mut f = RegionFinder {
            threshold: 0.0,
            weak: 0.0,
            ext_drop: f64::from(cfg.ext_drop),
            merge: cfg.merge as u64,
            pre: cfg.pre as u64,
            post: cfg.post as u64,
            max_region: cfg.max_region as u64,
            t_next: 0,
            open: None,
        };
        f.set_threshold(cfg.threshold);
        f
    }

    /// Change the trigger threshold (the weak threshold follows at
    /// `threshold - ext_drop`). Takes effect from the next pushed score; an
    /// open group is judged against the threshold in force when it finalises.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = f64::from(threshold);
        self.weak = (self.threshold - self.ext_drop).min(self.threshold);
    }

    /// Consume the score for the next target index `t` (strictly increasing
    /// from 0). Returns a region finalised by this sample, if any (before any
    /// contrast gate, which the engine applies).
    pub fn push(&mut self, t: u64, s: f32) -> Option<Region> {
        debug_assert_eq!(t, self.t_next, "targets must be pushed in order");
        let mut out = None;
        if let Some((a, e, peak)) = self.open {
            if t - e > self.merge {
                self.open = None;
                if f64::from(peak) > self.threshold {
                    let start = a.saturating_sub(self.pre);
                    let end = e + 1 + self.post;
                    if end - start <= self.max_region {
                        out = Some(Region { start, end });
                    }
                }
            }
        }
        if f64::from(s) > self.weak {
            self.open = Some(match self.open {
                Some((a, _, peak)) => (a, t, if s > peak { s } else { peak }),
                None => (t, t, s),
            });
        }
        self.t_next = t + 1;
        out
    }

    /// Earliest target index at which any future region can start.
    ///
    /// `min(t_next, a_open) - pre` (saturating), where an open group that is
    /// already doomed (its padded span exceeds `max_region`; it can only grow)
    /// does not hold the frontier back.
    pub fn frontier(&self) -> u64 {
        let hold = match self.open {
            Some((a, e, _)) if !self.doomed(a, e) => a.min(self.t_next),
            _ => self.t_next,
        };
        hold.saturating_sub(self.pre)
    }

    /// Next target index this finder expects.
    pub fn t_next(&self) -> u64 {
        self.t_next
    }

    /// Forget all state (next expected target is 0 again).
    pub fn reset(&mut self) {
        self.t_next = 0;
        self.open = None;
    }

    fn doomed(&self, a: u64, e: u64) -> bool {
        (e + 1 + self.post) - a.saturating_sub(self.pre) > self.max_region
    }
}

// ---- Contrast gate ----

/// Guard band (samples) around the region excluded from the reference level.
const CONTRAST_GUARD: usize = 20;

/// `classical.contrast`: peak |x''| in [a, b) over median |x''| around it
/// (ctx each side, guard 20), with the same edge clamping.
///
/// Allocates a scratch buffer; the engine uses [`contrast_with_scratch`].
pub fn contrast(x: &[f32], a: usize, b: usize, ctx: usize) -> f32 {
    let mut scratch = Vec::new();
    contrast_with_scratch(x, a, b, ctx, &mut scratch) as f32
}

/// [`contrast`] in f64 (the reference compares the f64 value against
/// `min_contrast`), using `scratch` as working memory. Does not allocate when
/// `scratch.capacity() >= min(len(x), b - a + 2 ctx)`.
pub fn contrast_with_scratch(
    x: &[f32],
    a: usize,
    b: usize,
    ctx: usize,
    scratch: &mut Vec<f64>,
) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    // Curvature needs a neighbour on each side.
    let (a, b) = (a.max(1), b.min(n - 1));
    if b <= a {
        return 0.0;
    }
    let lo = a.saturating_sub(ctx).max(1);
    let hi = (b + ctx).min(n - 1);
    // d[i] = |x''| at sample lo + i, i in 0..hi-lo.
    let curv = |i: usize| -> f64 {
        let k = lo + i;
        (f64::from(x[k + 1]) - 2.0 * f64::from(x[k]) + f64::from(x[k - 1])).abs()
    };
    let (ia, ib) = (a - lo, b - lo);
    let peak = (ia..ib).map(curv).fold(f64::NEG_INFINITY, f64::max);

    scratch.clear();
    scratch.extend((0..ia.saturating_sub(CONTRAST_GUARD)).map(curv));
    scratch.extend((ib + CONTRAST_GUARD..hi - lo).map(curv));
    let reference = median_in_place(scratch);
    peak / (reference + 1e-12)
}

/// numpy-style median (mean of the two middle values for even sizes);
/// 0 for an empty slice. Reorders `v`.
fn median_in_place(v: &mut [f64]) -> f64 {
    let m = v.len();
    if m == 0 {
        return 0.0;
    }
    let (left, mid, _) = v.select_nth_unstable_by(m / 2, f64::total_cmp);
    let mid = *mid;
    if m % 2 == 1 {
        mid
    } else {
        let below = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (below + mid) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_matches_numpy() {
        assert_eq!(median_in_place(&mut []), 0.0);
        assert_eq!(median_in_place(&mut [3.0]), 3.0);
        assert_eq!(median_in_place(&mut [4.0, 1.0]), 2.5);
        assert_eq!(median_in_place(&mut [5.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_in_place(&mut [5.0, 1.0, 3.0, 2.0, 9.0]), 3.0);
    }

    #[test]
    fn contrast_hand_case() {
        // A single spike in a quiet ramp: x'' is 0 except around the spike.
        let mut x: Vec<f32> = (0..400).map(|i| i as f32 * 1e-3).collect();
        x[200] += 1.0;
        // Around the spike |x''| is 1, 2, 1; elsewhere ~0 (rounding only).
        let c = contrast(&x, 195, 205, 150);
        assert!(c > 1e6, "contrast {c}");
        // Edge clamping: b <= a after clamping returns 0.
        assert_eq!(contrast(&x, 399, 450, 150), 0.0);
        assert_eq!(contrast(&x, 0, 1, 150), 0.0);
    }
}
