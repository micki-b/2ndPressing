//! The streaming declicker (spec section 5).
//!
//! Per input sample: the score source (network or test fixture) emits logits
//! once per frame; each channel's [`RegionFinder`] turns them into regions,
//! which are marked in a mask ring and unioned into runs. Complete runs are
//! grouped exactly like `classical.lsar_fill`, and each closed group is solved
//! by [`LsarSolver`] in place in a delay line of the working signal. Output is
//! read `latency` samples behind the input.
//!
//! Timing (all indices are absolute input-sample indices):
//! - the region mask is final below the finder's frontier `F`;
//! - a run `[u0, u1)` is complete when `u1 < F`;
//! - a group is solved once it is closed, the mask is final through
//!   `u1 + p` (the reference also solves masked samples in `[u1, u1 + p)`) and
//!   `context_right` samples after `u1` have arrived;
//! - the deadline guard forces anything still pending when its first sample is
//!   about to leave the delay line.
//!
//! Limitation (long runs): spec section 6 assumes every group spans at most
//! `max_group`. A single run is its own group whatever its length, and
//! overlapping or touching regions can chain into runs longer than the latency
//! can cover (about `latency - 517` samples at 48 kHz). The deadline guard then
//! does not wait for the run to complete. It repairs the prefix whose mask is
//! already final, using left context only (the least-squares solution with no
//! right context is the AR forward extrapolation). The rest of the run then
//! continues as a run of its own, with that prefix as left context. The
//! reference (`classical.lsar_fill`) solves such a run in one piece, so the
//! repaired samples differ from it there. Each such event counts as a deadline
//! miss.

use crate::lsar::{LsarConfig, LsarSolver};
use crate::model::{Model, PostConfig};
use crate::post::{contrast_with_scratch, RegionFinder};

/// Anything that turns stereo samples into per-channel logits with a fixed
/// delay (the network, or fixture scores in tests).
pub trait ScoreSource: Send {
    /// Push one stereo sample; every `stride` samples returns 4 logits per
    /// channel; `logit[n]` predicts target `n - delay()`.
    ///
    /// The engine assumes stride 4 and that logits are returned on pushes
    /// `n = 4j + 3`, covering `logit[4j..4j+4]`.
    fn push(&mut self, l: f32, r: f32) -> Option<[[f32; 4]; 2]>;
    /// Output delay `D` in source samples.
    fn delay(&self) -> usize;
    /// Return to the start-up state.
    fn reset(&mut self);
}

/// What [`Declicker::process`] writes to its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Repaired, delayed signal.
    #[default]
    Declick,
    /// Delayed input minus output: what was removed.
    ClicksOnly,
    /// Delayed input (latency preserved).
    Bypass,
}

/// Running counters (cleared by [`Declicker::reset`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Complete click runs (unions of overlapping regions), both channels.
    pub regions: u64,
    /// Masked samples inside solved groups (`[u0, u1)`), both channels.
    pub samples_repaired: u64,
    /// Groups or runs whose first sample reached the output before they could
    /// be solved normally. They are solved early with partial right context;
    /// for a still-incomplete run, its final prefix is extrapolated from left
    /// context only; otherwise they are left unrepaired.
    pub deadline_misses: u64,
    /// Solves that could not run or failed, leaving their samples unrepaired:
    /// fewer than `order` samples of right context, a window that does not
    /// fit the delay line, or a numerical failure (non-positive Cholesky pivot
    /// or non-finite solution). Groups skipped at stream start (`u0 < order`,
    /// as in the reference) are not counted. Non-finite input samples are
    /// treated as 0 inside the solve window (they are never written back
    /// unless masked), so NaN/inf input does not by itself cause failures.
    pub solve_failures: u64,
}

/// Default latency at 48 kHz (spec section 6).
pub const LATENCY_48K: usize = 1536;

/// Largest detection decimation factor (`round(sr / 48000)`, i.e. 384 kHz).
/// Higher rates are clamped to it; see [`rate_factor`].
pub const MAX_RATE_FACTOR: usize = 8;

/// Detection decimation factor the engine uses for `sample_rate`.
///
/// `round(sample_rate / 48000)` clamped to `1..=MAX_RATE_FACTOR`. Rates that
/// are not finite or not positive fall back to native 48 kHz behaviour
/// (factor 1). Rates above ~408 kHz are clamped to [`MAX_RATE_FACTOR`]; the
/// detector then sees material above 48 kHz, which is unsupported (detection
/// quality is unvalidated), but memory and latency stay bounded.
pub fn rate_factor(sample_rate: f32) -> usize {
    if !(sample_rate.is_finite() && sample_rate > 0.0) {
        return 1;
    }
    let f = (f64::from(sample_rate) / 48_000.0).round();
    // `f` is finite and >= 0 here; the clamp bounds the cast.
    (f.min(MAX_RATE_FACTOR as f64) as usize).clamp(1, MAX_RATE_FACTOR)
}

/// Decimated-detection rates (`factor >= 2`) use a linear-phase FIR with
/// `FIR_HALF_PER_FACTOR * factor` taps on each side of the centre.
const FIR_HALF_PER_FACTOR: usize = 32;

/// Capacity of the closed-group queue per channel. Groups normally wait only
/// a few hundred samples; on overflow the oldest is forced (deadline miss).
const QUEUE_CAP: usize = 64;

// ---- Engine ----

/// Fixed-latency streaming click remover for stereo audio.
pub struct Declicker {
    source: Box<dyn ScoreSource>,
    /// Score-source delay `D` (source samples).
    delay: u64,
    post: PostConfig,
    /// LSAR settings at the native rate (contexts and max_group x factor).
    lsar: LsarConfig,
    /// Native samples per detection sample (1 at 44.1/48 kHz).
    factor: u64,
    decim: Option<Decimator>,
    latency: usize,
    /// Contrast context at the native rate.
    contrast_ctx: usize,
    ring_mask: usize,
    /// Native samples consumed.
    n: u64,
    /// Samples pushed to the score source.
    src_n: u64,
    chans: [Channel; 2],
    solver: LsarSolver,
    win: Vec<f64>,
    win_mask: Vec<bool>,
    cx: Vec<f32>,
    cscratch: Vec<f64>,
    mode: OutputMode,
    stats: Stats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    u0: u64,
    u1: u64,
}

/// Per-channel streaming state.
struct Channel {
    finder: RegionFinder,
    /// Raw input ring.
    raw: Vec<f32>,
    /// Working (repaired) signal ring, f64 like the reference.
    work: Vec<f64>,
    /// Region mask ring.
    mask: Vec<bool>,
    /// Run being built from overlapping/touching regions.
    pending: Option<Span>,
    /// The pending run missed its deadline: discard it when it completes.
    abandoned: bool,
    /// Open LSAR group.
    group: Option<Span>,
    /// Closed groups waiting for mask finality / right context.
    queue: SpanQueue,
    /// Native-rate frontier: the mask is final below it.
    frontier: u64,
}

impl Declicker {
    /// Engine with the network as score source.
    ///
    /// 44.1/48 kHz (and any rate below ~72 kHz) run natively. Rates with
    /// `factor = round(sr / 48000) >= 2` use the experimental decimated
    /// detection path (see [`Declicker::with_score_source_at`]). Invalid or
    /// extreme rates are handled as described in [`rate_factor`] and never
    /// panic.
    pub fn new(model: &Model, sample_rate: f32) -> Self {
        let source = Box::new(crate::net::Detector::new(model));
        Self::build(model, sample_rate, source)
    }

    /// Engine driven by an arbitrary score source at 48 kHz (tests).
    pub fn with_score_source(model: &Model, source: Box<dyn ScoreSource>) -> Self {
        Self::build(model, 48_000.0, source)
    }

    /// Engine driven by an arbitrary score source at `sample_rate`.
    ///
    /// Experimental high-rate path (`factor = round(sr / 48000) >= 2`): the
    /// source sees each channel low-passed (Blackman-windowed sinc, cutoff
    /// `0.45 / factor` cycles/sample, `64·factor + 1` taps, linear phase) and
    /// decimated by `factor`, delay compensated so detection sample `m`
    /// corresponds to native sample `m·factor`. Regions map back ×factor; the
    /// region finder keeps the model's 48 kHz settings. LSAR runs natively with
    /// the configured order and `context`, `context_right`, `max_group`
    /// ×factor; the contrast gate (if enabled) runs natively with its context
    /// ×factor. Latency is `1536·factor` plus the FIR delay `32·factor`.
    /// Limitations: the model never saw low-passed high-rate material, so
    /// detection quality there is unvalidated; regions are quantised to
    /// `factor` samples. The factor is computed by [`rate_factor`] (non-finite
    /// or non-positive rates act as 48 kHz; the factor is capped at
    /// [`MAX_RATE_FACTOR`]).
    pub fn with_score_source_at(
        model: &Model,
        sample_rate: f32,
        source: Box<dyn ScoreSource>,
    ) -> Self {
        Self::build(model, sample_rate, source)
    }

    fn build(model: &Model, sample_rate: f32, source: Box<dyn ScoreSource>) -> Self {
        let factor = rate_factor(sample_rate);
        let decim = (factor >= 2).then(|| Decimator::new(factor));
        let fir_delay = decim.as_ref().map_or(0, |d| d.delay);
        let latency = LATENCY_48K * factor + fir_delay;
        let post = model.post.clone();
        let base = LsarConfig::default();
        let lsar = LsarConfig {
            order: base.order,
            context: base.context * factor,
            context_right: base.context_right * factor,
            max_group: base.max_group * factor,
        };
        let contrast_ctx = post.contrast_context * factor;
        let p = lsar.order;
        let max_region = post.max_region * factor;

        // Rings hold the left context of anything that can still be solved
        // plus the contrast window of any region being finalised.
        let ring = (latency + lsar.context.max(p) + max_region + 2 * contrast_ctx + 256)
            .next_power_of_two();
        // A group solved before (or at) its deadline spans at most `latency`.
        let max_span = latency.max(lsar.max_group);
        let solver = LsarSolver::with_max_span(&lsar, max_span);
        let win_len = lsar.context.max(p) + max_span + lsar.context_right.max(p) + p;
        let contrast_len = max_region + 2 * contrast_ctx + 4;
        let chan = || Channel::new(&post, ring);
        Declicker {
            delay: source.delay() as u64,
            source,
            post: post.clone(),
            lsar,
            factor: factor as u64,
            decim,
            latency,
            contrast_ctx,
            ring_mask: ring - 1,
            n: 0,
            src_n: 0,
            chans: [chan(), chan()],
            solver,
            win: vec![0.0; win_len],
            win_mask: vec![false; win_len],
            cx: vec![0.0; contrast_len],
            cscratch: Vec::with_capacity(contrast_len),
            mode: OutputMode::default(),
            stats: Stats::default(),
        }
    }

    /// Fixed processing latency in samples at the engine's sample rate.
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// Change the trigger threshold (logit units; the sensitivity control).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.post.threshold = threshold;
        for ch in &mut self.chans {
            ch.finder.set_threshold(threshold);
        }
    }

    /// Select what [`Declicker::process`] outputs; detection and repair keep
    /// running in every mode.
    pub fn set_mode(&mut self, mode: OutputMode) {
        self.mode = mode;
    }

    /// Process in place; any block length; real-time safe (no allocation).
    /// Only the first `min(left.len(), right.len())` samples are processed.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let (out, _) = self.step(*l, *r);
            *l = out[0];
            *r = out[1];
        }
    }

    /// Clear all signal state and [`Stats`]; keeps threshold and mode.
    pub fn reset(&mut self) {
        self.source.reset();
        if let Some(d) = &mut self.decim {
            d.reset();
        }
        self.n = 0;
        self.src_n = 0;
        for ch in &mut self.chans {
            ch.reset();
        }
        self.stats = Stats::default();
    }

    /// Counters since construction or the last [`Declicker::reset`].
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Latency-compensated whole-buffer processing (CLI, tests).
    ///
    /// Resets the engine, feeds the input followed by `latency` zeros and
    /// drops the first `latency` output samples.
    pub fn process_offline(&mut self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let out = self.process_offline_with_masks(left, right);
        (out.left, out.right)
    }

    /// [`Declicker::process_offline`] that also returns the final region mask
    /// per channel (1 = sample inside a detected region), aligned with the
    /// output. The mask is read when each sample leaves the delay line, which
    /// is after it became final unless a deadline was missed. Intended for
    /// parity tests and diagnostics.
    pub fn process_offline_with_masks(&mut self, left: &[f32], right: &[f32]) -> OfflineOutput {
        self.reset();
        let len = left.len().min(right.len());
        let mut out = OfflineOutput {
            left: Vec::with_capacity(len),
            right: Vec::with_capacity(len),
            masks: [Vec::with_capacity(len), Vec::with_capacity(len)],
        };
        let zeros = std::iter::repeat_n((0.0, 0.0), self.latency);
        let input = left[..len]
            .iter()
            .copied()
            .zip(right[..len].iter().copied());
        for (i, (l, r)) in input.chain(zeros).enumerate() {
            let (y, m) = self.step(l, r);
            if i >= self.latency {
                out.left.push(y[0]);
                out.right.push(y[1]);
                out.masks[0].push(u8::from(m[0]));
                out.masks[1].push(u8::from(m[1]));
            }
        }
        out
    }

    // ---- Per-sample pipeline ----

    /// One stereo sample in, one delayed stereo sample (and its mask) out.
    fn step(&mut self, l: f32, r: f32) -> ([f32; 2], [bool; 2]) {
        let n = self.n;
        let slot = n as usize & self.ring_mask;
        for (ch, x) in self.chans.iter_mut().zip([l, r]) {
            ch.raw[slot] = x;
            ch.work[slot] = f64::from(x);
            ch.mask[slot] = ch.pending.is_some_and(|p| p.u0 <= n && n < p.u1);
        }

        let det = match &mut self.decim {
            Some(d) => d.push(l, r),
            None => Some((l, r)),
        };
        if let Some((dl, dr)) = det {
            let sn = self.src_n;
            self.src_n += 1;
            if let Some(logits) = self.source.push(dl, dr) {
                // logits cover source indices sn - 3 ..= sn.
                for k in 0..4u64 {
                    let Some(t) = (sn + k).checked_sub(3 + self.delay) else {
                        continue;
                    };
                    for (c, lg) in logits.iter().enumerate() {
                        self.on_score(c, t, lg[k as usize]);
                    }
                }
            }
        }

        for c in 0..2 {
            self.solve_ready(c);
        }

        let mut out = [0.0f32; 2];
        let mut mask = [false; 2];
        if let Some(o) = n.checked_sub(self.latency as u64) {
            let oslot = o as usize & self.ring_mask;
            for c in 0..2 {
                self.deadline(c, o);
                let ch = &self.chans[c];
                let x = ch.raw[oslot];
                let y = ch.work[oslot] as f32;
                out[c] = match self.mode {
                    OutputMode::Declick => y,
                    OutputMode::ClicksOnly => x - y,
                    OutputMode::Bypass => x,
                };
                mask[c] = ch.mask[oslot];
            }
        }
        self.n += 1;
        (out, mask)
    }

    /// Feed score `s` for detection target `t` to channel `c`.
    fn on_score(&mut self, c: usize, t: u64, s: f32) {
        let f = self.factor;
        if let Some(reg) = self.chans[c].finder.push(t, s) {
            let (a, b) = (reg.start * f, reg.end * f);
            if self.post.min_contrast <= 0.0
                || self.contrast(c, a, b) >= f64::from(self.post.min_contrast)
            {
                self.add_region(c, a, b);
            }
        }
        let frontier = self.chans[c].finder.frontier() * f;
        self.chans[c].frontier = frontier;
        if let Some(p) = self.chans[c].pending {
            if p.u1 < frontier {
                self.chans[c].pending = None;
                self.complete_run(c, p);
            }
        }
        self.check_close(c);
    }

    /// `classical.contrast` on the raw input around native region `[a, b)`.
    fn contrast(&mut self, c: usize, a: u64, b: u64) -> f64 {
        let ctx = self.contrast_ctx as u64;
        let avail = self.n + 1;
        let oldest = avail.saturating_sub(self.ring_mask as u64 + 1);
        // Window [w0, w1) so that the slice's edge clamping matches the
        // reference's whole-signal clamping (w0 = 0 or w0 = a - ctx - 1).
        let w0 = a.saturating_sub(ctx + 1).max(oldest);
        let w1 = (b + ctx + 1).min(avail).min(w0 + self.cx.len() as u64);
        let len = (w1 - w0) as usize;
        let ch = &self.chans[c];
        for (i, v) in self.cx[..len].iter_mut().enumerate() {
            *v = ch.raw[(w0 as usize + i) & self.ring_mask];
        }
        let (la, lb) = ((a.max(w0) - w0) as usize, (b.max(w0) - w0) as usize);
        contrast_with_scratch(
            &self.cx[..len],
            la,
            lb,
            self.contrast_ctx,
            &mut self.cscratch,
        )
    }

    /// Mark `[a, b)` in the mask and union it into the pending run.
    fn add_region(&mut self, c: usize, a: u64, b: u64) {
        let n = self.n;
        let ring_mask = self.ring_mask;
        let ch = &mut self.chans[c];
        // Samples not yet written are marked on arrival (see `step`).
        for i in a..b.min(n + 1) {
            ch.mask[i as usize & ring_mask] = true;
        }
        match ch.pending {
            Some(ref mut p) if a <= p.u1 => p.u1 = p.u1.max(b),
            Some(p) => {
                // Region starts are non-decreasing, so nothing can touch `p`.
                ch.pending = Some(Span { u0: a, u1: b });
                let abandoned = std::mem::replace(&mut ch.abandoned, false);
                self.finish_run(c, p, abandoned);
            }
            None => ch.pending = Some(Span { u0: a, u1: b }),
        }
    }

    /// The pending run became complete (`u1 < F`).
    fn complete_run(&mut self, c: usize, run: Span) {
        let abandoned = std::mem::replace(&mut self.chans[c].abandoned, false);
        self.finish_run(c, run, abandoned);
    }

    /// Group a complete run like `classical.lsar_fill`.
    fn finish_run(&mut self, c: usize, run: Span, abandoned: bool) {
        self.stats.regions += 1;
        if abandoned {
            return;
        }
        let p2 = 2 * self.lsar.order as u64;
        let max_group = self.lsar.max_group as u64;
        match self.chans[c].group {
            Some(ref mut g)
                if run.u0 - g.u1 < p2 && (max_group == 0 || run.u1 - g.u0 <= max_group) =>
            {
                g.u1 = run.u1;
            }
            prev => {
                self.chans[c].group = Some(run);
                if let Some(g) = prev {
                    self.enqueue(c, g);
                }
            }
        }
    }

    /// Close the open group once no future run can join it.
    fn check_close(&mut self, c: usize) {
        let ch = &self.chans[c];
        let Some(g) = ch.group else { return };
        // Earliest start of any run not yet complete.
        let f = ch.pending.map_or(ch.frontier, |p| p.u0.min(ch.frontier));
        let p2 = 2 * self.lsar.order as u64;
        let max_group = self.lsar.max_group as u64;
        if f >= g.u1 + p2 || (max_group > 0 && f > g.u0 + max_group) {
            self.chans[c].group = None;
            self.enqueue(c, g);
        }
    }

    fn enqueue(&mut self, c: usize, g: Span) {
        if self.chans[c].queue.is_full() {
            let head = self.chans[c].queue.pop().expect("full queue has a head");
            self.stats.deadline_misses += 1;
            self.solve(c, head);
        }
        self.chans[c].queue.push(g);
    }

    /// Solve queued groups whose mask and right context are final.
    fn solve_ready(&mut self, c: usize) {
        let p = self.lsar.order as u64;
        let need_right = self.lsar.context_right.max(self.lsar.order) as u64;
        while let Some(g) = self.chans[c].queue.front() {
            if self.chans[c].frontier >= g.u1 + p && self.n + 1 >= g.u1 + need_right {
                self.chans[c].queue.pop();
                self.solve(c, g);
            } else {
                break;
            }
        }
    }

    /// Deadline guard before native sample `o` leaves the delay line.
    fn deadline(&mut self, c: usize, o: u64) {
        while let Some(g) = self.chans[c].queue.front().filter(|g| g.u0 <= o) {
            self.chans[c].queue.pop();
            self.stats.deadline_misses += 1;
            self.solve(c, g);
        }
        if let Some(g) = self.chans[c].group.filter(|g| g.u0 <= o) {
            self.chans[c].group = None;
            self.stats.deadline_misses += 1;
            self.solve(c, g);
        }
        // A still-incomplete run reaching the output (only runs longer than
        // the latency can cover; see the module docs): repair the prefix whose
        // mask is final, `[u0, k)` with `k = min(F, u1)`, from left context
        // only, and carry on with `[k, u1)` as the pending run. If too little
        // is final (or the solve fails), leave the whole run unrepaired.
        let ch = &self.chans[c];
        if ch.abandoned {
            return;
        }
        let Some(run) = ch.pending.filter(|p| p.u0 <= o) else {
            return;
        };
        self.stats.deadline_misses += 1;
        let p = self.lsar.order as u64;
        let k = ch.frontier.min(run.u1).min(self.n + 1);
        if run.u0 >= p && k >= run.u0 + p && self.solve_prefix(c, run.u0, k) {
            self.chans[c].pending = Some(Span { u0: k, u1: run.u1 });
        } else {
            self.chans[c].abandoned = true;
        }
    }

    /// LSAR-solve group `g` in the delay line with the right context present
    /// now (needs at least `p` samples after `u1`; otherwise left unrepaired
    /// and counted in [`Stats::solve_failures`]). Groups with `u0 < p` are
    /// skipped (stream start, as the reference).
    fn solve(&mut self, c: usize, g: Span) {
        let p = self.lsar.order as u64;
        let right_end = (g.u1 + self.lsar.context_right as u64).min(self.n + 1);
        if g.u0 >= p && right_end < g.u1 + p {
            self.stats.solve_failures += 1;
            return;
        }
        self.solve_window(c, g.u0, g.u1, right_end, g.u1);
    }

    /// Deadline repair of the final prefix `[u0, k)` of an incomplete run with
    /// no right context: the window ends at `k`, so every masked sample in
    /// `[u0, k)` is an unknown and the least-squares solution is the AR
    /// forward extrapolation from the left context. Requires `k >= u0 + p`.
    fn solve_prefix(&mut self, c: usize, u0: u64, k: u64) -> bool {
        let p = self.lsar.order as u64;
        self.solve_window(c, u0, k - p, k, k)
    }

    /// Copy `[u0 - context, end)` of channel `c` into the solve window, solve
    /// the masked samples in `[u0, u1 + p)` (the solver's convention) and
    /// write them back. `samples_repaired` counts masked samples in
    /// `[u0, count_end)`. Non-finite samples are read as 0 (only masked ones
    /// are ever written back).
    fn solve_window(&mut self, c: usize, u0: u64, u1: u64, end: u64, count_end: u64) -> bool {
        let p = self.lsar.order as u64;
        if u0 < p {
            return true; // stream start: not enough history (as the reference)
        }
        let avail = self.n + 1;
        let oldest = avail.saturating_sub(self.ring_mask as u64 + 1);
        let w0 = u0
            .saturating_sub(self.lsar.context.max(self.lsar.order) as u64)
            .max(oldest);
        let len = end.saturating_sub(w0) as usize;
        if u0 < w0 + p || end > avail || u1 + p > end || len > self.win.len() {
            self.stats.solve_failures += 1;
            return false;
        }
        let ring_mask = self.ring_mask;
        let ch = &mut self.chans[c];
        for (i, (v, m)) in self.win[..len]
            .iter_mut()
            .zip(&mut self.win_mask[..len])
            .enumerate()
        {
            let s = (w0 as usize + i) & ring_mask;
            let w = ch.work[s];
            *v = if w.is_finite() { w } else { 0.0 };
            *m = ch.mask[s];
        }
        let (lu0, lu1) = ((u0 - w0) as usize, (u1 - w0) as usize);
        if !self
            .solver
            .try_fill(&mut self.win[..len], &self.win_mask[..len], lu0, lu1)
        {
            self.stats.solve_failures += 1;
            return false;
        }
        for i in lu0..lu1 + p as usize {
            if self.win_mask[i] {
                ch.work[(w0 as usize + i) & ring_mask] = self.win[i];
            }
        }
        let lc = (count_end - w0) as usize;
        self.stats.samples_repaired += self.win_mask[lu0..lc].iter().filter(|&&m| m).count() as u64;
        true
    }
}

/// Result of [`Declicker::process_offline_with_masks`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OfflineOutput {
    /// Left output (latency compensated).
    pub left: Vec<f32>,
    /// Right output (latency compensated).
    pub right: Vec<f32>,
    /// Region mask per channel (0/1), aligned with the output.
    pub masks: [Vec<u8>; 2],
}

impl Channel {
    fn new(post: &PostConfig, ring: usize) -> Self {
        Channel {
            finder: RegionFinder::new(post),
            raw: vec![0.0; ring],
            work: vec![0.0; ring],
            mask: vec![false; ring],
            pending: None,
            abandoned: false,
            group: None,
            queue: SpanQueue::new(QUEUE_CAP),
            frontier: 0,
        }
    }

    fn reset(&mut self) {
        self.finder.reset();
        self.raw.fill(0.0);
        self.work.fill(0.0);
        self.mask.fill(false);
        self.pending = None;
        self.abandoned = false;
        self.group = None;
        self.queue.clear();
        self.frontier = 0;
    }
}

// ---- Fixed-capacity FIFO ----

struct SpanQueue {
    buf: Vec<Span>,
    head: usize,
    len: usize,
}

impl SpanQueue {
    fn new(cap: usize) -> Self {
        SpanQueue {
            buf: vec![Span { u0: 0, u1: 0 }; cap],
            head: 0,
            len: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.len == self.buf.len()
    }

    fn front(&self) -> Option<Span> {
        (self.len > 0).then(|| self.buf[self.head])
    }

    fn push(&mut self, s: Span) {
        debug_assert!(!self.is_full());
        let i = (self.head + self.len) % self.buf.len();
        self.buf[i] = s;
        self.len += 1;
    }

    fn pop(&mut self) -> Option<Span> {
        let s = self.front()?;
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        Some(s)
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

// ---- Decimator (experimental high-rate detection) ----

/// Per-channel linear-phase low-pass + decimation for the detection path.
struct Decimator {
    factor: usize,
    /// FIR group delay (taps on each side of the centre).
    delay: usize,
    taps: Vec<f64>,
    hist: [Vec<f64>; 2],
    hist_mask: usize,
    n: usize,
}

impl Decimator {
    fn new(factor: usize) -> Self {
        let delay = FIR_HALF_PER_FACTOR * factor;
        let len = 2 * delay + 1;
        let fc = 0.45 / factor as f64; // cycles per native sample
        let pi = std::f64::consts::PI;
        let mut taps: Vec<f64> = (0..len)
            .map(|k| {
                let m = k as f64 - delay as f64;
                let sinc = if m == 0.0 {
                    1.0
                } else {
                    (2.0 * pi * fc * m).sin() / (2.0 * pi * fc * m)
                };
                let ph = 2.0 * pi * k as f64 / (len - 1) as f64;
                let blackman = 0.42 - 0.5 * ph.cos() + 0.08 * (2.0 * ph).cos();
                sinc * blackman
            })
            .collect();
        let sum: f64 = taps.iter().sum();
        taps.iter_mut().for_each(|t| *t /= sum);
        let size = len.next_power_of_two();
        Decimator {
            factor,
            delay,
            taps,
            hist: [vec![0.0; size], vec![0.0; size]],
            hist_mask: size - 1,
            n: 0,
        }
    }

    /// Push one native sample; returns detection sample `m` (centred on native
    /// sample `m·factor`) when native input reaches `m·factor + delay`.
    fn push(&mut self, l: f32, r: f32) -> Option<(f32, f32)> {
        let n = self.n;
        self.n += 1;
        let slot = n & self.hist_mask;
        self.hist[0][slot] = f64::from(l);
        self.hist[1][slot] = f64::from(r);
        if n < self.delay || !(n - self.delay).is_multiple_of(self.factor) {
            return None;
        }
        let mut y = [0.0f64; 2];
        for (yc, h) in y.iter_mut().zip(&self.hist) {
            *yc = self
                .taps
                .iter()
                .enumerate()
                .filter(|&(k, _)| k <= n)
                .map(|(k, t)| t * h[(n - k) & self.hist_mask])
                .sum();
        }
        Some((y[0] as f32, y[1] as f32))
    }

    fn reset(&mut self) {
        self.n = 0;
        for h in &mut self.hist {
            h.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FeatureConfig, NetConfig};

    /// A source that never detects anything.
    struct Silent(usize);

    impl ScoreSource for Silent {
        fn push(&mut self, _l: f32, _r: f32) -> Option<[[f32; 4]; 2]> {
            self.0 += 1;
            self.0.is_multiple_of(4).then_some([[-100.0; 4]; 2])
        }
        fn delay(&self) -> usize {
            512
        }
        fn reset(&mut self) {
            self.0 = 0;
        }
    }

    fn config_model() -> Model {
        Model {
            net: NetConfig {
                c: 24,
                blocks: 8,
                stem_k: 8,
                stride: 4,
                d: 512,
            },
            features: FeatureConfig {
                eps: 1e-6,
                floor: 1e-5,
                nw: 2048,
                asinh_scale: 8.0,
            },
            post: PostConfig {
                threshold: 2.75,
                ext_drop: 2.0,
                merge: 8,
                pre: 2,
                post: 8,
                max_region: 480,
                min_contrast: 0.0,
                contrast_context: 150,
            },
            tensors: Vec::new(),
        }
    }

    #[test]
    fn rate_factor_bounds() {
        let cases = [
            (48_000.0, 1),
            (44_100.0, 1),
            (71_999.0, 1),
            (72_000.0, 2),
            (88_200.0, 2),
            (96_000.0, 2),
            (176_400.0, 4),
            (192_000.0, 4),
            (384_000.0, 8),
            (1e12, MAX_RATE_FACTOR),
            (f32::MAX, MAX_RATE_FACTOR),
            (0.0, 1),
            (-48_000.0, 1),
            (f32::NAN, 1),
            (f32::INFINITY, 1),
            (f32::NEG_INFINITY, 1),
        ];
        for (sr, want) in cases {
            assert_eq!(rate_factor(sr), want, "sample rate {sr}");
        }
    }

    #[test]
    fn extreme_sample_rates_do_not_panic() {
        let model = config_model();
        let capped = LATENCY_48K * MAX_RATE_FACTOR + FIR_HALF_PER_FACTOR * MAX_RATE_FACTOR;
        for (sr, lat) in [
            (f32::INFINITY, LATENCY_48K),
            (f32::NAN, LATENCY_48K),
            (0.0, LATENCY_48K),
            (-48_000.0, LATENCY_48K),
            (1e12, capped),
            (f32::MAX, capped),
        ] {
            let mut eng = Declicker::with_score_source_at(&model, sr, Box::new(Silent(0)));
            assert_eq!(eng.latency_samples(), lat, "sample rate {sr}");
            let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
            let (yl, _) = eng.process_offline(&x, &x);
            assert_eq!(yl, x, "sample rate {sr}: nothing detected, output = input");
        }
    }

    #[test]
    fn failed_solves_are_counted() {
        let mut eng = Declicker::with_score_source(&config_model(), Box::new(Silent(0)));
        let mut l: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut r = l.clone();
        eng.process(&mut l, &mut r);
        let n = eng.n;
        assert_eq!(n, 3000);
        let mark = |eng: &mut Declicker, u0: u64, u1: u64| {
            for i in u0..u1 {
                eng.chans[0].mask[i as usize & eng.ring_mask] = true;
            }
        };

        // Fewer than p samples of right context: counted, left unrepaired.
        mark(&mut eng, n - 100, n - 10);
        eng.solve(
            0,
            Span {
                u0: n - 100,
                u1: n - 10,
            },
        );
        assert_eq!(eng.stats.solve_failures, 1);
        assert_eq!(eng.stats.samples_repaired, 0);

        // Stream start (u0 < p) is skipped like the reference: not a failure.
        eng.solve(0, Span { u0: 5, u1: 10 });
        assert_eq!(eng.stats.solve_failures, 1);

        // A normal solve succeeds and is not counted.
        mark(&mut eng, 1500, 1520);
        eng.solve(0, Span { u0: 1500, u1: 1520 });
        assert_eq!(eng.stats.solve_failures, 1);
        assert_eq!(eng.stats.samples_repaired, 20);

        // Longer than the solver capacity: try_fill refuses; counted.
        mark(&mut eng, 100, 2900);
        eng.solve(0, Span { u0: 100, u1: 2900 });
        assert_eq!(eng.stats.solve_failures, 2);
        assert_eq!(eng.stats.samples_repaired, 20);
    }
}
