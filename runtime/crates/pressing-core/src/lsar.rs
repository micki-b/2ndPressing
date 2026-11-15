//! Least-squares AR interpolation (spec section 4), f64, preallocated.
//!
//! Port of `classical.lsar_fill` for one group: an AR model is fitted to the
//! Hann-windowed left and right context, and the masked samples minimise the
//! prediction-error energy over the segment `[u0 - p, u1 + p)`. The normal
//! equations are banded (bandwidth `p` in unknown-index space) and solved by a
//! banded Cholesky factorisation in f64. All scratch space is allocated in
//! [`LsarSolver::new`] / [`LsarSolver::with_max_span`].

/// Repair settings (fixture manifest -> repair).
#[derive(Clone, Debug, PartialEq)]
pub struct LsarConfig {
    /// AR order `p`.
    pub order: usize,
    /// Left context length used for the AR fit.
    pub context: usize,
    /// Right context length used for the AR fit.
    pub context_right: usize,
    /// Maximum span `u1 - u0` of runs sharing one solve (0 = unlimited).
    pub max_group: usize,
}

impl Default for LsarConfig {
    fn default() -> Self {
        LsarConfig {
            order: 32,
            context: 1024,
            context_right: 512,
            max_group: 480,
        }
    }
}

// ---- AR model helpers ----

/// Add the autocorrelation of `piece · hann(len(piece))` at lags `0..r.len()`
/// to `r` (`r[k] += Σ_n v[n]·v[n+k]`). The window is numpy's `hanning`.
pub fn add_windowed_autocorr(piece: &[f64], r: &mut [f64], scratch: &mut Vec<f64>) {
    let m = piece.len();
    scratch.clear();
    if m == 1 {
        scratch.push(piece[0]);
    } else {
        let denom = (m - 1) as f64;
        scratch.extend(piece.iter().enumerate().map(|(i, &v)| {
            // numpy: n = arange(1 - M, M, 2); w = 0.5 + 0.5 cos(pi n / (M - 1))
            let k = (2 * i) as f64 + 1.0 - m as f64;
            v * (0.5 + 0.5 * (std::f64::consts::PI * k / denom).cos())
        }));
    }
    let v = scratch.as_slice();
    for (k, rk) in r.iter_mut().enumerate() {
        if k >= m {
            break;
        }
        *rk += v[..m - k]
            .iter()
            .zip(&v[k..])
            .map(|(a, b)| a * b)
            .sum::<f64>();
    }
}

/// `classical.ar_from_autocorr`: prediction-error filter `A = [1, a1..ap]`
/// written to `a` (`a.len() == r.len() == p + 1`).
///
/// `r[0] <= 1e-20` gives `A = [1, 0, ...]`; otherwise `r[0]` is scaled by
/// `1 + 1e-6` and the Toeplitz (Yule-Walker) system is solved by
/// Levinson-Durbin, falling back to `[1, 0, ...]` if a pivot vanishes or the
/// result is not finite. `tmp` must have at least `p + 1` elements. Returns
/// `false` when a fallback was used.
pub fn ar_from_autocorr(r: &[f64], a: &mut [f64], tmp: &mut [f64]) -> bool {
    let p = r.len() - 1;
    a.fill(0.0);
    a[0] = 1.0;
    if r[0] <= 1e-20 || !r[0].is_finite() {
        return false;
    }
    let mut err = r[0] * (1.0 + 1e-6);
    for k in 1..=p {
        if !err.is_finite() || err <= 0.0 {
            a.fill(0.0);
            a[0] = 1.0;
            return false;
        }
        let acc = r[k] + (1..k).map(|j| a[j] * r[k - j]).sum::<f64>();
        let kk = -acc / err;
        tmp[..k].copy_from_slice(&a[..k]);
        for j in 1..k {
            a[j] = tmp[j] + kk * tmp[k - j];
        }
        a[k] = kk;
        err *= 1.0 - kk * kk;
    }
    if a.iter().all(|v| v.is_finite()) {
        true
    } else {
        a.fill(0.0);
        a[0] = 1.0;
        false
    }
}

// ---- Solver ----

/// Reusable solver with scratch space sized at construction.
pub struct LsarSolver {
    cfg: LsarConfig,
    max_span: usize,
    /// Autocorrelation (p + 1).
    r: Vec<f64>,
    /// Prediction-error filter A (p + 1).
    a: Vec<f64>,
    /// Levinson scratch (p + 1).
    tmp: Vec<f64>,
    /// Windowed context piece.
    piece: Vec<f64>,
    /// Segment offsets of the unknowns.
    pos: Vec<usize>,
    /// Banded normal matrix / its Cholesky factor: row j, `band[j*(p+1) + k]`
    /// holds entry (j, j - k).
    band: Vec<f64>,
    /// Right-hand side, then the solution.
    rhs: Vec<f64>,
    /// Prediction error of the known samples per row.
    known: Vec<f64>,
}

impl LsarSolver {
    /// Solver for groups spanning up to `cfg.max_group` samples (at least
    /// `order`); use [`LsarSolver::with_max_span`] for longer groups.
    pub fn new(cfg: &LsarConfig) -> Self {
        Self::with_max_span(cfg, cfg.max_group.max(cfg.order))
    }

    /// Solver for groups with `u1 - u0 <= max_span`: room for
    /// `max_span + order` unknowns (the reference also solves masked samples
    /// in `[u1, u1 + p)`) and `max_span + 2 order` segment columns.
    pub fn with_max_span(cfg: &LsarConfig, max_span: usize) -> Self {
        let p = cfg.order;
        let u = max_span + p;
        LsarSolver {
            cfg: cfg.clone(),
            max_span,
            r: vec![0.0; p + 1],
            a: vec![0.0; p + 1],
            tmp: vec![0.0; p + 1],
            piece: Vec::with_capacity(cfg.context.max(cfg.context_right).max(1)),
            pos: Vec::with_capacity(u),
            band: vec![0.0; u * (p + 1)],
            rhs: vec![0.0; u],
            known: vec![0.0; u],
        }
    }

    /// The settings this solver was built with.
    pub fn config(&self) -> &LsarConfig {
        &self.cfg
    }

    /// Largest `u1 - u0` this solver accepts.
    pub fn max_span(&self) -> usize {
        self.max_span
    }

    /// Fill the unknown samples of one group in place.
    ///
    /// `y` is a contiguous working window, `u0..u1` the group span inside it
    /// and `unknown[i]` marks masked samples of `y`. As in the reference, the
    /// unknowns are the masked samples in `[u0, u1 + order)`; masked samples in
    /// `[u0 - order, u0)` belong to earlier groups and count as known.
    /// Requires `u0 >= order` and `u1 + order <= y.len()`; left context is
    /// `y[u0-context..u0]` (clipped), right context is `y[u1..u1+context_right]`
    /// (clipped to `y.len()`). Never touches samples with `unknown == false`.
    /// Must not allocate. Invalid or over-capacity input leaves `y` unchanged
    /// (see [`LsarSolver::try_fill`]).
    pub fn fill(&mut self, y: &mut [f64], unknown: &[bool], u0: usize, u1: usize) {
        let _ = self.try_fill(y, unknown, u0, u1);
    }

    /// [`LsarSolver::fill`], returning `false` (and leaving `y` unchanged)
    /// when the arguments violate the requirements, `u1 - u0 > max_span()`,
    /// or the solve fails (non-positive pivot or non-finite solution).
    pub fn try_fill(&mut self, y: &mut [f64], unknown: &[bool], u0: usize, u1: usize) -> bool {
        let p = self.cfg.order;
        if p == 0 || u0 < p || u1 < u0 || u1 + p > y.len() || unknown.len() < y.len() {
            return false;
        }
        if u1 - u0 > self.max_span {
            return false;
        }

        // AR model from the windowed context on both sides.
        self.r.fill(0.0);
        let left = &y[u0.saturating_sub(self.cfg.context)..u0];
        let right = &y[u1..(u1 + self.cfg.context_right).min(y.len())];
        for piece in [left, right] {
            if piece.len() > p {
                add_windowed_autocorr(piece, &mut self.r, &mut self.piece);
            }
        }
        ar_from_autocorr(&self.r, &mut self.a, &mut self.tmp);
        let a = &self.a;

        // Segment [c0, c1): N columns, N - p rows. Row i, column c holds
        // A[p - (c - i)] for 0 <= c - i <= p.
        let c0 = u0 - p;
        let n = u1 + p - c0;
        let rows = n - p;
        let seg = &y[c0..c0 + n];
        let unk = &unknown[c0..c0 + n];
        self.pos.clear();
        self.pos.extend((p..n).filter(|&c| unk[c]));
        let nu = self.pos.len();
        if nu == 0 {
            return true;
        }

        // Prediction error of the known samples (unknowns set to zero).
        for (i, e) in self.known[..rows].iter_mut().enumerate() {
            *e = (0..=p)
                .filter(|&k| !(i + k >= p && unk[i + k]))
                .map(|k| a[p - k] * seg[i + k])
                .sum();
        }

        // Normal equations (Mu^T Mu) x = -Mu^T (Mk yk), banded.
        let w = p + 1;
        for j in 0..nu {
            let cj = self.pos[j];
            let i_lo = cj.saturating_sub(p);
            let i_hi = cj.min(rows - 1);
            self.rhs[j] = -(i_lo..=i_hi)
                .map(|i| a[p + i - cj] * self.known[i])
                .sum::<f64>();
            let row = &mut self.band[j * w..(j + 1) * w];
            row.fill(0.0);
            for l in (j.saturating_sub(p)..=j).rev() {
                let cl = self.pos[l];
                if cj - cl > p {
                    break;
                }
                // Rows covering both columns: max(cj, cl) - p ..= min(cj, cl).
                let hi = cl.min(rows - 1);
                row[j - l] = (i_lo..=hi).map(|i| a[p + i - cj] * a[p + i - cl]).sum();
            }
        }

        // Banded Cholesky in place: band[j][k] = L(j, j - k).
        for j in 0..nu {
            let j_lo = j.saturating_sub(p);
            for l in j_lo..=j {
                let m_lo = j_lo.max(l.saturating_sub(p));
                let mut s = self.band[j * w + (j - l)];
                for m in m_lo..l {
                    s -= self.band[j * w + (j - m)] * self.band[l * w + (l - m)];
                }
                if l == j {
                    if !s.is_finite() || s <= 0.0 {
                        return false;
                    }
                    self.band[j * w] = s.sqrt();
                } else {
                    self.band[j * w + (j - l)] = s / self.band[l * w];
                }
            }
        }
        // Forward substitution L z = rhs.
        for j in 0..nu {
            let mut s = self.rhs[j];
            for m in j.saturating_sub(p)..j {
                s -= self.band[j * w + (j - m)] * self.rhs[m];
            }
            self.rhs[j] = s / self.band[j * w];
        }
        // Back substitution L^T x = z.
        for j in (0..nu).rev() {
            let mut s = self.rhs[j];
            for i in j + 1..(j + w).min(nu) {
                s -= self.band[i * w + (i - j)] * self.rhs[i];
            }
            self.rhs[j] = s / self.band[j * w];
        }
        if !self.rhs[..nu].iter().all(|v| v.is_finite()) {
            return false;
        }
        for (&c, &v) in self.pos.iter().zip(&self.rhs[..nu]) {
            y[c0 + c] = v;
        }
        true
    }
}
