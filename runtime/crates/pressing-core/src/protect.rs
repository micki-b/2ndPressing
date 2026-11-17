//! Percussion protection: leave detections that sit on a musical beat grid.
//!
//! Port of `tools/src/pressing_tools/protect/rhythm.py` (the reference; keep
//! the two in step). Hand-claps, snares and hi-hats make the click detector
//! fire again and again *on a beat grid*; vinyl defects arrive at random times
//! or, for a scratch, exactly once per rotation (1.8 s at 33⅓, 1.333 s at 45,
//! 0.769 s at 78 RPM).
//!
//! For a candidate run with peak time `t`, the *event train* (peaks of all
//! detections at a fixed internal threshold, both channels, within ±10 dB of
//! the candidate's 10 ms level) from `t - past` to `t + future` is tested for
//! a period `P` in 0.15–0.9 s (rotation periods ±2.5 % excluded):
//!
//! ```text
//! S(P) = (on - off) / sqrt(on + off + 1)
//! ```
//!
//! where `on` counts occupied grid points `t + kP` and `off` the mean over the
//! grids shifted by ¼, ½ and ¾ of `P`. The run is *rhythmic* when
//! `max S > theta`. Protection spreads to detections within ±10 ms (one clap
//! yields several detections). A protected run is still repaired if the same
//! channel has events exactly one (or two) rotation periods away (within
//! 0.5 ms): a scratch repeats with turntable precision.
//!
//! Streaming: a run is decided once the events up to `t + cluster + future +
//! tol` are known, so the engine adds about `future` seconds of latency.

use crate::model::PostConfig;
use crate::post::RegionFinder;

/// Rotation periods in seconds: 33⅓, 45 and 78 RPM.
pub const ROTATION_PERIODS: [f64; 3] = [1.8, 60.0 / 45.0, 60.0 / 78.0];

/// User-facing protection mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Protection {
    /// No protection; lowest latency (1536 samples at 48 kHz).
    Off,
    /// About 1 s of look-ahead.
    #[default]
    Standard,
    /// About 2 s of look-ahead: protects more of the start of a rhythmic
    /// passage, at ~1 point of click recall.
    Extended,
}

impl Protection {
    /// Look-ahead in seconds, or `None` when off.
    pub fn future_s(self) -> Option<f64> {
        match self {
            Protection::Off => None,
            Protection::Standard => Some(1.0),
            Protection::Extended => Some(2.0),
        }
    }

    /// Parse `off`, `standard` or `extended` (case-insensitive).
    pub fn parse(s: &str) -> Option<Protection> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" => Some(Protection::Off),
            "standard" | "on" => Some(Protection::Standard),
            "extended" => Some(Protection::Extended),
            _ => None,
        }
    }
}

/// Parameters of the rhythm protector (defaults = the Python reference).
#[derive(Clone, Debug, PartialEq)]
pub struct ProtectConfig {
    pub past: f64,
    pub future: f64,
    pub tol: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub n_periods: usize,
    pub theta: f64,
    pub min_points: usize,
    pub event_thr: f32,
    pub rot_tol: f64,
    pub rot_precision: f64,
    pub veto: bool,
    pub level_match_db: Option<f64>,
    pub level_win: f64,
    pub cluster_s: f64,
}

impl ProtectConfig {
    /// Reference defaults with the given look-ahead.
    pub fn with_future(future: f64) -> Self {
        ProtectConfig {
            past: 3.0,
            future,
            tol: 0.008,
            p_min: 0.15,
            p_max: 0.9,
            n_periods: 120,
            theta: 2.5,
            min_points: 4,
            event_thr: 2.75,
            rot_tol: 0.025,
            rot_precision: 0.0005,
            veto: true,
            level_match_db: Some(10.0),
            level_win: 0.010,
            cluster_s: 0.010,
        }
    }

    /// How far before a candidate the event train must be kept (rhythm
    /// window or rotation veto, whichever reaches further back).
    pub fn history_s(&self) -> f64 {
        (self.past + self.tol).max(2.0 * 1.02 * ROTATION_PERIODS[0] + 0.01)
    }

    /// Delay before a candidate can be decided, in seconds after its peak.
    pub fn decision_delay_s(&self) -> f64 {
        self.cluster_s + self.future + self.tol
    }
}

// ---- Beat-grid statistic ----

/// Precomputed period grid (`P`, `k`, offsets `T = P·k`, validity).
pub struct RhythmGrid {
    periods: Vec<f64>,
    nk: usize,
    offsets: Vec<f64>,
    valid: Vec<bool>,
    n_valid: Vec<usize>,
    tol: f64,
    min_points: usize,
}

/// numpy `linspace(start, stop, num)` (endpoint included, exact end).
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

impl RhythmGrid {
    pub fn new(cfg: &ProtectConfig) -> Self {
        let periods: Vec<f64> = linspace(cfg.p_min.ln(), cfg.p_max.ln(), cfg.n_periods)
            .into_iter()
            .map(f64::exp)
            .filter(|p| {
                ROTATION_PERIODS
                    .iter()
                    .all(|r| (p - r).abs() / r >= cfg.rot_tol)
            })
            .collect();
        let pmin = periods.iter().copied().fold(f64::INFINITY, f64::min);
        let kmax = (cfg.past / pmin) as i64 + 1;
        let fmax = (cfg.future / pmin) as i64 + 1;
        let ks: Vec<i64> = (-kmax..=fmax).filter(|&k| k != 0).collect();
        let nk = ks.len();
        let mut offsets = Vec::with_capacity(periods.len() * nk);
        let mut valid = Vec::with_capacity(periods.len() * nk);
        let mut n_valid = Vec::with_capacity(periods.len());
        for &p in &periods {
            let mut n = 0;
            for &k in &ks {
                let t = p * k as f64;
                let v = t >= -cfg.past && t <= cfg.future;
                offsets.push(t);
                valid.push(v);
                n += usize::from(v);
            }
            n_valid.push(n);
        }
        RhythmGrid {
            periods,
            nk,
            offsets,
            valid,
            n_valid,
            tol: cfg.tol,
            min_points: cfg.min_points,
        }
    }

    /// Is any event within `tol` of `x`? `ev` sorted ascending.
    fn hit(&self, ev: &[f64], x: f64) -> bool {
        if ev.is_empty() {
            return false;
        }
        let i = ev.partition_point(|&e| e < x);
        let a = ev[i.min(ev.len() - 1)];
        let b = ev[i.saturating_sub(1).min(ev.len() - 1)];
        (a - x).abs().min((b - x).abs()) < self.tol
    }

    /// `max_P S(P)` for event times relative to the candidate (seconds,
    /// sorted, candidate excluded). `-inf` when no period has enough points.
    pub fn score(&self, ev: &[f64]) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for (i, &p) in self.periods.iter().enumerate() {
            if self.n_valid[i] < self.min_points {
                continue;
            }
            let row = i * self.nk..(i + 1) * self.nk;
            let mut on = 0usize;
            let mut off = [0usize; 3];
            for (&t, &v) in self.offsets[row.clone()].iter().zip(&self.valid[row]) {
                if !v {
                    continue;
                }
                on += usize::from(self.hit(ev, t));
                for (o, frac) in off.iter_mut().zip([0.25, 0.5, 0.75]) {
                    *o += usize::from(self.hit(ev, t + frac * p));
                }
            }
            let off = (off[0] + off[1] + off[2]) as f64 / 3.0;
            let on = on as f64;
            let s = (on - off) / (on + off + 1.0).sqrt();
            if s > best {
                best = s;
            }
        }
        best
    }
}

/// Two other events at exactly one rotation (`t-R, t-2R` or `t-R, t+R`)?
/// `same`: event times relative to the candidate, same channel.
pub fn rotation_periodic(same: &[f64], precision: f64) -> bool {
    for r in ROTATION_PERIODS {
        let (lo, hi) = (0.98 * r, 1.02 * r);
        for &e1 in same.iter().filter(|&&e| e <= -lo && e >= -hi) {
            if same
                .iter()
                .filter(|&&e| e <= -2.0 * lo && e >= -2.0 * hi)
                .any(|&e2| (e2 - 2.0 * e1).abs() < precision)
            {
                return true;
            }
            if same
                .iter()
                .filter(|&&e| e >= lo && e <= hi)
                .any(|&f| (f + e1).abs() < precision)
            {
                return true;
            }
        }
    }
    false
}

// ---- Streaming state ----

/// A detection event: native sample index of its peak and its 10 ms level.
#[derive(Clone, Copy, Debug, Default)]
struct Event {
    sample: u64,
    level: f64,
}

/// An undecided candidate run (native coordinates).
#[derive(Clone, Copy, Debug, Default)]
pub struct Candidate {
    pub u0: u64,
    pub u1: u64,
    pub peak: u64,
    pub level: f64,
    rhythmic: Option<bool>,
}

/// Fixed-capacity ring (no allocation after construction).
struct Ring<T: Copy + Default> {
    buf: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Copy + Default> Ring<T> {
    fn new(cap: usize) -> Self {
        Ring {
            buf: vec![T::default(); cap.max(1)],
            head: 0,
            len: 0,
        }
    }
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, i: usize) -> T {
        self.buf[(self.head + i) % self.buf.len()]
    }
    fn get_mut(&mut self, i: usize) -> &mut T {
        let n = self.buf.len();
        &mut self.buf[(self.head + i) % n]
    }
    fn front(&self) -> Option<T> {
        (self.len > 0).then(|| self.get(0))
    }
    fn pop(&mut self) -> Option<T> {
        let v = self.front()?;
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        Some(v)
    }
    /// Push; returns the evicted oldest element if the ring was full.
    fn push(&mut self, v: T) -> Option<T> {
        let evicted = if self.len == self.buf.len() {
            self.pop()
        } else {
            None
        };
        let i = (self.head + self.len) % self.buf.len();
        self.buf[i] = v;
        self.len += 1;
        evicted
    }
    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

const EVENT_CAP: usize = 8192;
const QUEUE_CAP: usize = 4096;
const RECENT_CAP: usize = 256;
const SCORE_RING: usize = 1 << 14;

/// Streaming rhythm protector for one stereo stream (owned by the engine).
pub struct Protector {
    pub cfg: ProtectConfig,
    grid: RhythmGrid,
    sr: f64,
    factor: u64,
    level_h: u64,
    ev_finders: [RegionFinder; 2],
    /// Open event run and open candidate run (detection coordinates).
    ev_open: [Option<(u64, u64)>; 2],
    cand_open: [Option<(u64, u64)>; 2],
    scores: [Vec<f32>; 2],
    events: [Ring<Event>; 2],
    queue: [Ring<Candidate>; 2],
    /// Recently decided candidates: (peak, rhythmic), for clustering.
    recent: [Ring<(u64, bool)>; 2],
    scratch: Vec<f64>,
    same: Vec<f64>,
}

/// Decision for one candidate run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub channel: usize,
    pub u0: u64,
    pub u1: u64,
    pub keep: bool,
    pub forced: bool,
}

impl Protector {
    /// `sr`: native sample rate; `factor`: native samples per detection sample.
    pub fn new(cfg: ProtectConfig, post: &PostConfig, sr: f64, factor: u64) -> Self {
        let mut ev_post = post.clone();
        ev_post.threshold = cfg.event_thr;
        let finder = || {
            let mut f = RegionFinder::new(&ev_post);
            f.set_threshold(cfg.event_thr);
            f
        };
        Protector {
            grid: RhythmGrid::new(&cfg),
            level_h: (cfg.level_win * sr) as u64,
            sr,
            factor,
            ev_finders: [finder(), finder()],
            ev_open: [None; 2],
            cand_open: [None; 2],
            scores: [vec![0.0; SCORE_RING], vec![0.0; SCORE_RING]],
            events: [Ring::new(EVENT_CAP), Ring::new(EVENT_CAP)],
            queue: [Ring::new(QUEUE_CAP), Ring::new(QUEUE_CAP)],
            recent: [Ring::new(RECENT_CAP), Ring::new(RECENT_CAP)],
            scratch: Vec::with_capacity(2 * EVENT_CAP),
            same: Vec::with_capacity(EVENT_CAP),
            cfg,
        }
    }

    /// Extra latency (native samples) the engine must add.
    pub fn extra_latency(&self) -> usize {
        (self.cfg.decision_delay_s() * self.sr).ceil() as usize + 2048 * self.factor as usize
    }

    pub fn reset(&mut self) {
        for c in 0..2 {
            self.ev_finders[c].reset();
            self.ev_open[c] = None;
            self.cand_open[c] = None;
            self.scores[c].fill(0.0);
            self.events[c].clear();
            self.queue[c].clear();
            self.recent[c].clear();
        }
    }

    fn secs(&self, sample: u64) -> f64 {
        sample as f64 / self.sr
    }

    /// Peak (native) of detection-coordinate run `[a, b)`: first maximum score.
    fn peak(&self, c: usize, a: u64, b: u64) -> u64 {
        let lo = a.max(b.saturating_sub(SCORE_RING as u64 - 1));
        let mut best = lo;
        let mut bv = f32::NEG_INFINITY;
        for t in lo..b {
            let v = self.scores[c][t as usize & (SCORE_RING - 1)];
            if v > bv {
                bv = v;
                best = t;
            }
        }
        best * self.factor
    }

    /// Feed one score of channel `c` at detection target `t`. `user` is the
    /// region the engine's own finder emitted for this score (if any, after
    /// any contrast gate). `level(c, native_sample)` returns the 10 ms level
    /// in dB. Completed candidate runs enter the decision queue.
    pub fn on_score(
        &mut self,
        c: usize,
        t: u64,
        s: f32,
        user: Option<(u64, u64)>,
        user_frontier: u64,
        level: &dyn Fn(usize, u64, u64) -> f64,
    ) -> Option<Decision> {
        self.scores[c][t as usize & (SCORE_RING - 1)] = s;
        let mut forced = None;
        // Event train at the fixed internal threshold.
        if let Some(reg) = self.ev_finders[c].push(t, s) {
            match self.ev_open[c] {
                Some((_, ref mut b)) if reg.start <= *b => *b = (*b).max(reg.end),
                prev => {
                    self.ev_open[c] = Some((reg.start, reg.end));
                    if let Some((a, b)) = prev {
                        self.finish_event(c, a, b, level);
                    }
                }
            }
        }
        if let Some((a, b)) = self.ev_open[c] {
            if b < self.ev_finders[c].frontier() {
                self.ev_open[c] = None;
                self.finish_event(c, a, b, level);
            }
        }
        // Candidate runs (union of the engine's regions).
        if let Some((ra, rb)) = user {
            match self.cand_open[c] {
                Some((_, ref mut b)) if ra <= *b => *b = (*b).max(rb),
                prev => {
                    self.cand_open[c] = Some((ra, rb));
                    if let Some((a, b)) = prev {
                        forced = forced.or(self.finish_candidate(c, a, b, level));
                    }
                }
            }
        }
        if let Some((a, b)) = self.cand_open[c] {
            if b < user_frontier {
                self.cand_open[c] = None;
                forced = forced.or(self.finish_candidate(c, a, b, level));
            }
        }
        forced
    }

    fn finish_event(&mut self, c: usize, a: u64, b: u64, level: &dyn Fn(usize, u64, u64) -> f64) {
        let pk = self.peak(c, a, b);
        let lv = level(c, pk, self.level_h);
        self.events[c].push(Event {
            sample: pk,
            level: lv,
        });
    }

    /// Returns a forced decision if the queue overflowed.
    fn finish_candidate(
        &mut self,
        c: usize,
        a: u64,
        b: u64,
        level: &dyn Fn(usize, u64, u64) -> f64,
    ) -> Option<Decision> {
        let f = self.factor;
        let pk = self.peak(c, a, b);
        let cand = Candidate {
            u0: a * f,
            u1: b * f,
            peak: pk,
            level: level(c, pk, self.level_h),
            rhythmic: None,
        };
        let mut forced = None;
        if self.queue[c].len() == QUEUE_CAP {
            forced = self.decide_front(c, true);
        }
        self.queue[c].push(cand);
        forced
    }

    /// Native index below which every event (both channels) is known.
    fn events_known(&self) -> u64 {
        (0..2)
            .map(|c| {
                let fr = self.ev_finders[c].frontier();
                self.ev_open[c].map_or(fr, |(a, _)| a.min(fr)) * self.factor
            })
            .min()
            .unwrap_or(0)
    }

    /// Earliest native start of any candidate not yet decided (or `u64::MAX`).
    pub fn undecided_floor(&self, c: usize) -> u64 {
        let open = self.cand_open[c].map_or(u64::MAX, |(a, _)| a * self.factor);
        open.min(self.queue[c].front().map_or(u64::MAX, |q| q.u0))
    }

    /// Native start of the open (incomplete) candidate run of channel `c`,
    /// or `u64::MAX`: no complete candidate can start at or after it.
    pub fn undecided_open(&self, c: usize) -> u64 {
        self.cand_open[c].map_or(u64::MAX, |(a, _)| a * self.factor)
    }

    /// Earliest native sample of any undecided candidate in channel `c`
    /// (queue only), for the deadline guard.
    pub fn queue_front_u0(&self, c: usize) -> Option<u64> {
        self.queue[c].front().map(|q| q.u0)
    }

    fn rhythmic(&mut self, cand: &Candidate) -> bool {
        let t = self.secs(cand.peak);
        let (lo, hi) = (
            t - self.cfg.past - self.cfg.tol,
            t + self.cfg.future + self.cfg.tol,
        );
        self.scratch.clear();
        for ch in 0..2 {
            for i in 0..self.events[ch].len() {
                let e = self.events[ch].get(i);
                let te = self.secs(e.sample);
                if te < lo || te > hi {
                    continue;
                }
                let rel = te - t;
                if rel.abs() <= 0.002 {
                    continue;
                }
                if let Some(db) = self.cfg.level_match_db {
                    if (e.level - cand.level).abs() > db {
                        continue;
                    }
                }
                self.scratch.push(rel);
            }
        }
        self.scratch.sort_unstable_by(f64::total_cmp);
        self.grid.score(&self.scratch) > self.cfg.theta
    }

    fn vetoed(&mut self, c: usize, cand: &Candidate) -> bool {
        let t = self.secs(cand.peak);
        let r0 = ROTATION_PERIODS[0];
        let (lo, hi) = (t - 2.0 * 1.02 * r0 - 0.01, t + 1.02 * r0 + 0.01);
        self.same.clear();
        for i in 0..self.events[c].len() {
            let te = self.secs(self.events[c].get(i).sample);
            if te > lo && te < hi {
                let rel = te - t;
                if rel.abs() > 0.002 && rel <= self.cfg.future + self.cfg.tol {
                    self.same.push(rel);
                }
            }
        }
        rotation_periodic(&self.same, self.cfg.rot_precision)
    }

    /// Compute rhythm flags that have become computable, then decide queue
    /// fronts whose neighbourhood is known. Returns decisions in order via
    /// `out` (at most `out.len()`; the rest wait for the next call).
    pub fn decide(&mut self, user_known: [u64; 2], out: &mut Vec<Decision>) {
        let ev_known = self.events_known();
        let need_flag = (self.cfg.future + self.cfg.tol) * self.sr;
        for c in 0..2 {
            for i in 0..self.queue[c].len() {
                let q = self.queue[c].get(i);
                if q.rhythmic.is_some() {
                    continue;
                }
                if (q.peak as f64 + need_flag) < ev_known as f64 {
                    let r = self.rhythmic(&q);
                    self.queue[c].get_mut(i).rhythmic = Some(r);
                } else {
                    break; // later candidates have later peaks
                }
            }
        }
        let need_decide = self.cfg.decision_delay_s() * self.sr;
        let cl = (self.cfg.cluster_s * self.sr).ceil() as u64;
        let user_ok = user_known[0].min(user_known[1]);
        for c in 0..2 {
            while let Some(q) = self.queue[c].front() {
                let ready = q.rhythmic.is_some()
                    && (q.peak as f64 + need_decide) < ev_known as f64
                    && q.peak + cl < user_ok;
                if !ready || out.len() == out.capacity() {
                    break;
                }
                if let Some(d) = self.decide_front(c, false) {
                    out.push(d);
                }
            }
        }
        self.prune(ev_known);
    }

    /// Decide the oldest candidate of channel `c` with whatever is known now.
    pub fn decide_front(&mut self, c: usize, forced: bool) -> Option<Decision> {
        let mut q = self.queue[c].front()?;
        let rhythmic = match q.rhythmic {
            Some(r) => r,
            None => self.rhythmic(&q),
        };
        q.rhythmic = Some(rhythmic);
        let t = self.secs(q.peak);
        let mut protect = rhythmic;
        if !protect && self.cfg.cluster_s > 0.0 {
            let near = |p: u64, r: bool| r && (self.secs(p) - t).abs() <= self.cfg.cluster_s;
            'outer: for ch in 0..2 {
                for i in 0..self.queue[ch].len() {
                    let o = self.queue[ch].get(i);
                    if o.rhythmic.is_some_and(|r| near(o.peak, r)) {
                        protect = true;
                        break 'outer;
                    }
                }
                for i in 0..self.recent[ch].len() {
                    let (p, r) = self.recent[ch].get(i);
                    if near(p, r) {
                        protect = true;
                        break 'outer;
                    }
                }
            }
        }
        if protect && self.cfg.veto && self.vetoed(c, &q) {
            protect = false;
        }
        self.queue[c].pop();
        self.recent[c].push((q.peak, rhythmic));
        Some(Decision {
            channel: c,
            u0: q.u0,
            u1: q.u1,
            keep: !protect,
            forced,
        })
    }

    /// Drop events older than any undecided candidate can need.
    fn prune(&mut self, ev_known: u64) {
        let floor = (0..2)
            .filter_map(|c| self.queue[c].front().map(|q| q.peak))
            .min()
            .unwrap_or(ev_known);
        let keep_from = floor.saturating_sub(((self.cfg.history_s() + 0.05) * self.sr) as u64);
        for events in &mut self.events {
            while events.front().is_some_and(|e| e.sample < keep_from) {
                events.pop();
            }
        }
    }

    // ---- Test hooks (event-level parity) ----

    /// Insert an event directly (tests).
    #[doc(hidden)]
    pub fn test_push_event(&mut self, c: usize, sample: u64, level: f64) {
        self.events[c].push(Event { sample, level });
    }

    /// Rhythm flag for a candidate given the stored events (tests).
    #[doc(hidden)]
    pub fn test_rhythmic(&mut self, peak: u64, level: f64) -> bool {
        let q = Candidate {
            u0: peak,
            u1: peak + 1,
            peak,
            level,
            rhythmic: None,
        };
        self.rhythmic(&q)
    }

    /// Final keep decision for a candidate given precomputed rhythm flags of
    /// all candidates `(channel, peak, rhythmic)` (tests).
    #[doc(hidden)]
    pub fn test_keep(
        &mut self,
        c: usize,
        peak: u64,
        rhythmic: bool,
        all: &[(usize, u64, bool)],
    ) -> bool {
        let t = self.secs(peak);
        let mut protect = rhythmic
            || all
                .iter()
                .any(|&(_, p, r)| r && (self.secs(p) - t).abs() <= self.cfg.cluster_s);
        let q = Candidate {
            u0: peak,
            u1: peak + 1,
            peak,
            level: 0.0,
            rhythmic: Some(rhythmic),
        };
        if protect && self.cfg.veto && self.vetoed(c, &q) {
            protect = false;
        }
        !protect
    }
}
