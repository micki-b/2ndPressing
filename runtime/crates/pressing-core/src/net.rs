//! Streaming ClickNet inference (spec section 2).
//!
//! Per sequence (one per channel, shared weights), frame `j` covers input
//! samples `4j..4j+3`:
//!
//! ```text
//! h0[j] = ReLU(b + sum_k W[:, :, k] . f[4j - 4 + k])                  k = 0..7
//! u[j]  = ReLU(b_dil + W0.h[j-2d] + W1.h[j-d] + W2.h[j])              d = 2^i
//! h'[j] = ReLU(h[j] + M.u[j] + m)
//! y[j]  = H2.ReLU(H1.h[j] + b1) + b2;     logit[4j + r] = y[j][r]
//! ```
//!
//! Layout for auto-vectorisation: every dense layer stores its weights in
//! output-blocked order (`[out / LANES][in][LANES]`, outputs zero-padded to a
//! multiple of [`LANES`]), so the inner loop is a fixed-width multiply-add over
//! `LANES` contiguous outputs with the accumulator held in registers. Both
//! channels run through each layer together so every weight is loaded once per
//! frame. All state is allocated in [`StreamingNet::new`]; `push` never allocates.

use crate::model::{Model, N_FEATURES};

/// Output-block width of the dense kernels (two NEON / one AVX register).
pub const LANES: usize = 8;

/// Number of sequences processed side by side (stereo: L and R).
const SEQ: usize = 2;

fn round_up(n: usize) -> usize {
    n.div_ceil(LANES) * LANES
}

// ---- Dense kernel ----

/// `out = bias + W . x` with `W` stored output-blocked and zero-padded.
struct Dense {
    n_in: usize,
    /// Padded output count (multiple of LANES).
    n_out: usize,
    /// `[n_out / LANES][n_in][LANES]`.
    w: Vec<f32>,
    /// `[n_out]`, zero-padded.
    b: Vec<f32>,
}

impl Dense {
    /// `weight(o, i)` gives `W[o, i]` for `o < out`, `i < n_in` (padded inputs
    /// may simply return 0).
    fn new(out: usize, n_in: usize, bias: &[f32], weight: impl Fn(usize, usize) -> f32) -> Self {
        let n_out = round_up(out);
        let mut w = vec![0.0; n_out * n_in];
        for blk in 0..n_out / LANES {
            for i in 0..n_in {
                for l in 0..LANES {
                    let o = blk * LANES + l;
                    if o < out {
                        w[(blk * n_in + i) * LANES + l] = weight(o, i);
                    }
                }
            }
        }
        let mut b = vec![0.0; n_out];
        b[..out].copy_from_slice(&bias[..out]);
        Dense { n_in, n_out, w, b }
    }

    /// `out[s] = b + W . x[s]` for both sequences at once.
    #[inline]
    fn apply2(&self, x: [&[f32]; SEQ], out: [&mut [f32]; SEQ]) {
        let [x0, x1] = x;
        let [o0, o1] = out;
        let (x0, x1) = (&x0[..self.n_in], &x1[..self.n_in]);
        let (w, _) = self.w.as_chunks::<LANES>();
        let (bias, _) = self.b.as_chunks::<LANES>();
        let (y0, _) = o0[..self.n_out].as_chunks_mut::<LANES>();
        let (y1, _) = o1[..self.n_out].as_chunks_mut::<LANES>();
        let outs = y0.iter_mut().zip(y1.iter_mut());
        for ((wb, bias), (y0, y1)) in w.chunks_exact(self.n_in).zip(bias).zip(outs) {
            let mut a0 = *bias;
            let mut a1 = *bias;
            for ((w, &v0), &v1) in wb.iter().zip(x0).zip(x1) {
                for l in 0..LANES {
                    a0[l] += w[l] * v0;
                    a1[l] += w[l] * v1;
                }
            }
            *y0 = a0;
            *y1 = a1;
        }
    }
}

#[inline]
fn relu(v: &mut [f32]) {
    for x in v {
        *x = x.max(0.0);
    }
}

// ---- Weights ----

struct Block {
    /// Dilation in frames.
    d: usize,
    /// Offset of this block's ring in `Seq::rings` (floats).
    ring_off: usize,
    /// Input `[h[j-2d], h[j-d], h[j]]` (3 x CP) -> CP.
    dil: Dense,
    /// CP -> CP.
    mix: Dense,
}

/// Per-sequence state and scratch (all CP = padded C long unless noted).
struct Seq {
    /// Stem window, feature-major `[N_FEATURES][stem_k]`: the last `stem_k` inputs.
    win: Vec<f32>,
    /// Concatenated block input `[h[j-2d], h[j-d], h[j]]` (3 x CP).
    xcat: Vec<f32>,
    h: Vec<f32>,
    u: Vec<f32>,
    t: Vec<f32>,
    /// Head output (padded stride).
    y: Vec<f32>,
    /// All block rings back to back; block i holds `2 d_i` frames of CP floats.
    rings: Vec<f32>,
}

impl Seq {
    fn new(n_win: usize, cp: usize, y_len: usize, ring_len: usize) -> Self {
        Seq {
            win: vec![0.0; n_win],
            xcat: vec![0.0; 3 * cp],
            h: vec![0.0; cp],
            u: vec![0.0; cp],
            t: vec![0.0; cp],
            y: vec![0.0; y_len],
            rings: vec![0.0; ring_len],
        }
    }

    fn reset(&mut self) {
        for v in [
            &mut self.win,
            &mut self.xcat,
            &mut self.h,
            &mut self.u,
            &mut self.t,
            &mut self.y,
            &mut self.rings,
        ] {
            v.fill(0.0);
        }
    }
}

/// Streaming network for both channels (shared weights, one state per channel).
pub struct StreamingNet {
    cp: usize,
    stride: usize,
    stem_k: usize,
    stem: Dense,
    blocks: Vec<Block>,
    head1: Dense,
    head2: Dense,
    /// Next write slot (== oldest frame `j - 2d`) of each block ring; shared by both sequences.
    ring_pos: Vec<usize>,
    /// Samples of the current frame received so far.
    phase: usize,
    seqs: [Seq; SEQ],
}

impl StreamingNet {
    /// Build from a validated [`Model`] (see [`Model::from_json_and_bin`]).
    ///
    /// # Panics
    /// If a tensor is missing or `net.stride != 4` (cannot happen for a model
    /// returned by [`Model::from_json_and_bin`]).
    pub fn new(model: &Model) -> Self {
        let cfg = &model.net;
        assert_eq!(cfg.stride, 4, "StreamingNet emits 4 logits per frame");
        assert!(cfg.stem_k >= cfg.stride, "stem_k must be >= stride");
        let (c, k) = (cfg.c, cfg.stem_k);
        let cp = round_up(c);
        let t = |name: &str| {
            model
                .tensor(name)
                .unwrap_or_else(|| panic!("model has no tensor `{name}`"))
        };

        // stem.weight [C, 3, k] row-major == [C, 3k] with input index f*k + tap.
        let sw = t("stem.weight");
        let stem = Dense::new(c, N_FEATURES * k, t("stem.bias"), |o, i| {
            sw[o * N_FEATURES * k + i]
        });

        let mut blocks = Vec::with_capacity(cfg.blocks);
        let mut ring_off = 0;
        for i in 0..cfg.blocks {
            let dw = t(&format!("blocks.{i}.dil.weight"));
            let mw = t(&format!("blocks.{i}.mix.weight"));
            // dil.weight [C, C, 3]; input index tap*CP + ci.
            let dil = Dense::new(c, 3 * cp, t(&format!("blocks.{i}.dil.bias")), |o, idx| {
                let (tap, ci) = (idx / cp, idx % cp);
                if ci < c {
                    dw[(o * c + ci) * 3 + tap]
                } else {
                    0.0
                }
            });
            let mix = Dense::new(c, cp, t(&format!("blocks.{i}.mix.bias")), |o, ci| {
                if ci < c {
                    mw[o * c + ci]
                } else {
                    0.0
                }
            });
            let d = 1usize << i;
            blocks.push(Block {
                d,
                ring_off,
                dil,
                mix,
            });
            ring_off += 2 * d * cp;
        }
        let pointwise = |wname: &str, bname: &str, out: usize| {
            let w = t(wname);
            Dense::new(
                out,
                cp,
                t(bname),
                |o, ci| if ci < c { w[o * c + ci] } else { 0.0 },
            )
        };
        let head1 = pointwise("head1.weight", "head1.bias", c);
        let head2 = pointwise("head2.weight", "head2.bias", cfg.stride);
        let y_len = head2.n_out;
        let seq = || Seq::new(N_FEATURES * k, cp, y_len, ring_off);
        StreamingNet {
            cp,
            stride: cfg.stride,
            stem_k: k,
            stem,
            ring_pos: vec![0; blocks.len()],
            blocks,
            head1,
            head2,
            phase: 0,
            seqs: [seq(), seq()],
        }
    }

    /// Ring-buffer floats of streaming state per sequence, counted like Python's
    /// `StreamingNet.state_floats`: the stem history ring (`3 * (stem_k - stride)`)
    /// plus the dilated-block rings (`2 d_i` frames each, at the padded width CP,
    /// which equals C when C is a multiple of [`LANES`], as for R2). The `3 * stride` slots of the in-progress
    /// frame (also held in the stem window) are not included.
    pub fn state_floats(&self) -> usize {
        N_FEATURES * (self.stem_k - self.stride) + self.seqs[0].rings.len()
    }

    /// Push one feature vector per channel. Every `stride` (= 4) pushes returns
    /// the 4 logits of the completed frame per channel; `logit[n]` predicts
    /// target sample `n - D`.
    pub fn push(&mut self, feat: [[f32; 3]; 2]) -> Option<[[f32; 4]; 2]> {
        let (k, s) = (self.stem_k, self.stride);
        let slot = k - s + self.phase;
        for (seq, f) in self.seqs.iter_mut().zip(feat) {
            for (fi, &v) in f.iter().enumerate() {
                seq.win[fi * k + slot] = v;
            }
        }
        self.phase += 1;
        if self.phase < s {
            return None;
        }
        self.phase = 0;
        self.frame();
        let mut out = [[0.0; 4]; 2];
        for (o, seq) in out.iter_mut().zip(&mut self.seqs) {
            o.copy_from_slice(&seq.y[..4]);
            // Keep the last k - s inputs as the start of the next window.
            for fi in 0..N_FEATURES {
                seq.win.copy_within(fi * k + s..(fi + 1) * k, fi * k);
            }
        }
        Some(out)
    }

    /// Run one frame for both sequences; result in `seqs[*].y`.
    fn frame(&mut self) {
        let cp = self.cp;
        let [a, b] = &mut self.seqs;

        self.stem.apply2([&a.win, &b.win], [&mut a.h, &mut b.h]);
        relu(&mut a.h);
        relu(&mut b.h);

        for (blk, pos) in self.blocks.iter().zip(self.ring_pos.iter_mut()) {
            let size = 2 * blk.d;
            let p = *pos;
            let q = (p + blk.d) % size;
            for seq in [&mut *a, &mut *b] {
                let ring = &mut seq.rings[blk.ring_off..blk.ring_off + size * cp];
                seq.xcat[..cp].copy_from_slice(&ring[p * cp..(p + 1) * cp]);
                seq.xcat[cp..2 * cp].copy_from_slice(&ring[q * cp..(q + 1) * cp]);
                seq.xcat[2 * cp..].copy_from_slice(&seq.h);
                ring[p * cp..(p + 1) * cp].copy_from_slice(&seq.h);
            }
            *pos = (p + 1) % size;

            blk.dil.apply2([&a.xcat, &b.xcat], [&mut a.u, &mut b.u]);
            relu(&mut a.u);
            relu(&mut b.u);
            blk.mix.apply2([&a.u, &b.u], [&mut a.t, &mut b.t]);
            for seq in [&mut *a, &mut *b] {
                for (h, t) in seq.h.iter_mut().zip(&seq.t) {
                    *h = (*h + t).max(0.0);
                }
            }
        }

        self.head1.apply2([&a.h, &b.h], [&mut a.t, &mut b.t]);
        relu(&mut a.t);
        relu(&mut b.t);
        self.head2.apply2([&a.t, &b.t], [&mut a.y, &mut b.y]);
    }

    /// Back to the start-of-stream state (all zeros).
    pub fn reset(&mut self) {
        self.phase = 0;
        self.ring_pos.fill(0);
        for seq in &mut self.seqs {
            seq.reset();
        }
    }
}

/// Features + network: raw stereo samples in, logits out.
pub struct Detector {
    /// Causal input features (section 1).
    pub features: crate::features::FeatureState,
    /// Streaming network (section 2).
    pub net: StreamingNet,
    /// Model delay `D`: `logit[n]` predicts target sample `n - delay`.
    pub delay: usize,
}

impl Detector {
    /// Fresh detector (zero state) for `model`.
    pub fn new(model: &Model) -> Self {
        Detector {
            features: crate::features::FeatureState::new(&model.features),
            net: StreamingNet::new(model),
            delay: model.net.d,
        }
    }
}

impl crate::engine::ScoreSource for Detector {
    fn push(&mut self, l: f32, r: f32) -> Option<[[f32; 4]; 2]> {
        let f = self.features.push(l, r);
        self.net.push(f)
    }
    fn delay(&self) -> usize {
        self.delay
    }
    fn reset(&mut self) {
        self.features.reset();
        self.net.reset();
    }
}
