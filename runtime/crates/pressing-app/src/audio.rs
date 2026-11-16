//! Live audio: device discovery, native-format streams, channel routing.
//!
//! Every device is opened at its **native channel count** (a 6-channel
//! interface in multitrack mode, a 6-channel USB speaker) and the user picks
//! which channel pair (or single channel, as mono) carries the music. That
//! avoids the "exactly 2 channels" limit of nih-plug's generic standalone.
//!
//! Signal path:
//!
//! ```text
//! input callback  : pick pair -> Dsp (declick, in place) -> ring (stereo frames)
//! output callback : ring -> Resampler (clock hand-off) -> chosen output pair
//! ```
//!
//! The declicker runs at the **input** rate. When input and output are the
//! same device at the same rate the hand-off is bit-exact; otherwise it is
//! rate-matched (see [`crate::resample`]). Nothing in either callback
//! allocates or locks; the GUI reads counters and the published
//! [`UiStats`] to tell "no callbacks", "digital silence" and "running" apart.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use pressing_core::{Model, OutputMode};
use pressing_plugin::dsp::Dsp;
use pressing_plugin::editor::UiStats;
use pressing_plugin::{sensitivity_to_threshold, Monitor, PercussionProtection};

use crate::resample::Resampler;

/// Rates offered in the rate menu when a device supports them.
pub const COMMON_RATES: [u32; 6] = [44_100, 48_000, 88_200, 96_000, 176_400, 192_000];

/// Largest block processed at once inside the input callback (frames).
const CHUNK: usize = 1024;

// ---- Devices ----

/// One input or output device as the app shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    /// Native channel count (the most any of its configurations offers).
    pub channels: u16,
    /// Rates from [`COMMON_RATES`] the device supports.
    pub rates: Vec<u32>,
    /// The device's current/default rate.
    pub default_rate: u32,
}

impl DeviceInfo {
    pub fn supports(&self, rate: u32) -> bool {
        self.rates.contains(&rate)
    }
}

/// A channel pair (or one channel as mono when `left == right`), 0-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Route {
    pub left: u16,
    pub right: u16,
}

impl Route {
    pub const FIRST_PAIR: Route = Route { left: 0, right: 1 };

    pub fn is_mono(self) -> bool {
        self.left == self.right
    }

    /// "1–2" or "3 mono" (1-based).
    pub fn label(self) -> String {
        if self.is_mono() {
            format!("{} mono", self.left + 1)
        } else {
            format!("{}–{}", self.left + 1, self.right + 1)
        }
    }

    pub fn fits(self, channels: u16) -> bool {
        self.left < channels && self.right < channels
    }
}

/// Routes for a device with `channels` channels: adjacent pairs first, then
/// every channel as mono.
pub fn routes(channels: u16) -> Vec<Route> {
    let mut v: Vec<Route> = (0..channels / 2)
        .map(|i| Route {
            left: 2 * i,
            right: 2 * i + 1,
        })
        .collect();
    v.extend((0..channels).map(|c| Route { left: c, right: c }));
    v
}

fn describe(dev: &cpal::Device, input: bool) -> Option<DeviceInfo> {
    let name = dev.name().ok()?;
    let ranges: Vec<_> = if input {
        dev.supported_input_configs().ok()?.collect()
    } else {
        dev.supported_output_configs().ok()?.collect()
    };
    let channels = ranges.iter().map(|r| r.channels()).max()?;
    if channels == 0 {
        return None;
    }
    let rates: Vec<u32> = COMMON_RATES
        .into_iter()
        .filter(|&sr| {
            ranges
                .iter()
                .any(|r| r.min_sample_rate().0 <= sr && sr <= r.max_sample_rate().0)
        })
        .collect();
    let default_rate = if input {
        dev.default_input_config().ok().map(|c| c.sample_rate().0)
    } else {
        dev.default_output_config().ok().map(|c| c.sample_rate().0)
    }
    .or_else(|| rates.first().copied())
    .unwrap_or(48_000);
    Some(DeviceInfo {
        name,
        channels,
        rates,
        default_rate,
    })
}

/// All input devices, then all output devices.
pub fn scan() -> (Vec<DeviceInfo>, Vec<DeviceInfo>) {
    let host = cpal::default_host();
    let inputs = host
        .input_devices()
        .map(|it| it.filter_map(|d| describe(&d, true)).collect())
        .unwrap_or_default();
    let outputs = host
        .output_devices()
        .map(|it| it.filter_map(|d| describe(&d, false)).collect())
        .unwrap_or_default();
    (inputs, outputs)
}

/// Names of the system default input and output devices.
pub fn default_names() -> (Option<String>, Option<String>) {
    let host = cpal::default_host();
    (
        host.default_input_device().and_then(|d| d.name().ok()),
        host.default_output_device().and_then(|d| d.name().ok()),
    )
}

fn find(input: bool, name: &str) -> Option<cpal::Device> {
    let host = cpal::default_host();
    let mut it: Box<dyn Iterator<Item = cpal::Device>> = if input {
        Box::new(host.input_devices().ok()?)
    } else {
        Box::new(host.output_devices().ok()?)
    };
    it.find(|d| d.name().ok().as_deref() == Some(name))
}

/// Native configuration of `dev` at `rate`: most channels, preferring f32.
fn native_config(
    dev: &cpal::Device,
    input: bool,
    rate: u32,
) -> Result<(StreamConfig, SampleFormat), String> {
    let ranges: Vec<_> = if input {
        dev.supported_input_configs()
            .map_err(|e| e.to_string())?
            .collect()
    } else {
        dev.supported_output_configs()
            .map_err(|e| e.to_string())?
            .collect()
    };
    let pick = ranges
        .iter()
        .filter(|r| r.min_sample_rate().0 <= rate && rate <= r.max_sample_rate().0)
        .filter(|r| {
            matches!(
                r.sample_format(),
                SampleFormat::F32 | SampleFormat::I16 | SampleFormat::I32
            )
        })
        .max_by_key(|r| (r.channels(), r.sample_format() == SampleFormat::F32))
        .ok_or_else(|| format!("no usable format at {rate} Hz"))?;
    Ok((
        StreamConfig {
            channels: pick.channels(),
            sample_rate: cpal::SampleRate(rate),
            buffer_size: cpal::BufferSize::Default,
        },
        pick.sample_format(),
    ))
}

// ---- Shared state ----

/// State shared between the GUI and the audio callbacks (atomics only on
/// the audio side; the error slot is written from cpal's error callback).
pub struct Shared {
    /// Sensitivity in percent (`f32` bits).
    pub sensitivity: AtomicU32,
    /// [`PercussionProtection`] index.
    pub protection: AtomicU8,
    /// [`Monitor`] index.
    pub monitor: AtomicU8,
    pub bypass: AtomicBool,
    pub stats: Arc<UiStats>,
    /// Input / output callbacks since the session started.
    pub in_callbacks: AtomicU64,
    pub out_callbacks: AtomicU64,
    /// Largest input block seen (frames).
    pub in_block_max: AtomicU32,
    /// Peak |input| since the GUI last took it (`f32` bits; non-negative
    /// floats order like their bits, so `fetch_max` works).
    pub peak: AtomicU32,
    pub underruns: AtomicU64,
    pub overflows: AtomicU64,
    pub resyncs: AtomicU64,
    /// Output step trim in ppm (rate-matched sessions).
    pub trim_ppm: AtomicU32,
    pub error: Mutex<Option<String>>,
}

impl Shared {
    pub fn new(sensitivity: f32, p: PercussionProtection, m: Monitor) -> Self {
        Shared {
            sensitivity: AtomicU32::new(sensitivity.to_bits()),
            protection: AtomicU8::new(protection_index(p)),
            monitor: AtomicU8::new(monitor_index(m)),
            bypass: AtomicBool::new(false),
            stats: Arc::new(UiStats::default()),
            in_callbacks: AtomicU64::new(0),
            out_callbacks: AtomicU64::new(0),
            in_block_max: AtomicU32::new(0),
            peak: AtomicU32::new(0),
            underruns: AtomicU64::new(0),
            overflows: AtomicU64::new(0),
            resyncs: AtomicU64::new(0),
            trim_ppm: AtomicU32::new(0),
            error: Mutex::new(None),
        }
    }

    /// Zero the per-session counters (before a new session starts).
    fn reset_counters(&self) {
        for c in [
            &self.in_callbacks,
            &self.out_callbacks,
            &self.underruns,
            &self.overflows,
            &self.resyncs,
        ] {
            c.store(0, Relaxed);
        }
        self.in_block_max.store(0, Relaxed);
        self.peak.store(0, Relaxed);
        self.trim_ppm.store(0, Relaxed);
        self.stats.publish(0, 0, 0, 0, 0.0);
        if let Ok(mut e) = self.error.lock() {
            *e = None;
        }
    }

    pub fn set_error(&self, msg: String) {
        if let Ok(mut e) = self.error.lock() {
            e.get_or_insert(msg);
        }
    }
}

pub fn protection_index(p: PercussionProtection) -> u8 {
    match p {
        PercussionProtection::Off => 0,
        PercussionProtection::Standard => 1,
        PercussionProtection::Extended => 2,
    }
}

pub fn protection_from(i: u8) -> PercussionProtection {
    match i {
        0 => PercussionProtection::Off,
        1 => PercussionProtection::Standard,
        _ => PercussionProtection::Extended,
    }
}

pub fn monitor_index(m: Monitor) -> u8 {
    match m {
        Monitor::Declick => 0,
        Monitor::ClicksOnly => 1,
    }
}

pub fn monitor_from(i: u8) -> Monitor {
    if i == 1 {
        Monitor::ClicksOnly
    } else {
        Monitor::Declick
    }
}

// ---- Session ----

/// What to open.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionConfig {
    pub input: String,
    pub in_route: Route,
    pub output: String,
    pub out_route: Route,
    /// Input (and processing) rate.
    pub rate: u32,
}

/// Running streams (dropping it stops them).
pub struct Session {
    _input: cpal::Stream,
    _output: cpal::Stream,
    pub in_rate: u32,
    pub out_rate: u32,
    pub in_channels: u16,
    pub out_channels: u16,
    /// Same device and rate on both sides: bit-exact hand-off.
    pub locked: bool,
}

/// Open both devices natively and start processing.
pub fn start(cfg: &SessionConfig, model: &Model, shared: &Arc<Shared>) -> Result<Session, String> {
    shared.reset_counters();
    let in_dev = find(true, &cfg.input).ok_or_else(|| format!("{} is not connected", cfg.input))?;
    let out_dev =
        find(false, &cfg.output).ok_or_else(|| format!("{} is not connected", cfg.output))?;
    let (in_cfg, in_fmt) =
        native_config(&in_dev, true, cfg.rate).map_err(|e| format!("{}: {e}", cfg.input))?;
    if !cfg.in_route.fits(in_cfg.channels) {
        return Err(format!(
            "{} has {} input channels; choose another pair",
            cfg.input, in_cfg.channels
        ));
    }
    // Output: same rate if it can, else its own default rate (rate-matched).
    let out_info = describe(&out_dev, false).ok_or("output device vanished")?;
    let out_rate = if out_info.supports(cfg.rate) {
        cfg.rate
    } else {
        out_info.default_rate
    };
    let (out_cfg, out_fmt) =
        native_config(&out_dev, false, out_rate).map_err(|e| format!("{}: {e}", cfg.output))?;
    if !cfg.out_route.fits(out_cfg.channels) {
        return Err(format!(
            "{} has {} output channels; choose another pair",
            cfg.output, out_cfg.channels
        ));
    }
    let locked = cfg.input == cfg.output && out_rate == cfg.rate;

    // Ring: 1 s of stereo at the input rate.
    let (prod, cons) = rtrb::RingBuffer::<f32>::new(2 * cfg.rate as usize);

    // One processor per protection mode (allocation happens here).
    let sr = cfg.rate as f32;
    let dsps: Vec<Dsp> = [
        PercussionProtection::Off,
        PercussionProtection::Standard,
        PercussionProtection::Extended,
    ]
    .map(|p| Dsp::new_with(model, sr, p.engine()))
    .into();

    let input = InputSide {
        shared: shared.clone(),
        dsps,
        active: usize::MAX,
        model_threshold: model.post.threshold,
        nch: usize::from(in_cfg.channels),
        route: cfg.in_route,
        l: vec![0.0; CHUNK],
        r: vec![0.0; CHUNK],
        prod,
        samples_done: 0,
    };
    let output = OutputSide {
        shared: shared.clone(),
        cons,
        rs: Resampler::new(f64::from(cfg.rate), f64::from(out_rate), locked),
        nch: usize::from(out_cfg.channels),
        route: cfg.out_route,
        in_rate: f64::from(cfg.rate),
        ratio: f64::from(cfg.rate) / f64::from(out_rate),
    };

    let in_stream = build_input(&in_dev, &in_cfg, in_fmt, input, shared.clone())
        .map_err(|e| format!("{}: {e}", cfg.input))?;
    let out_stream = build_output(&out_dev, &out_cfg, out_fmt, output, shared.clone())
        .map_err(|e| format!("{}: {e}", cfg.output))?;
    in_stream
        .play()
        .map_err(|e| format!("{}: {e}", cfg.input))?;
    out_stream
        .play()
        .map_err(|e| format!("{}: {e}", cfg.output))?;
    Ok(Session {
        _input: in_stream,
        _output: out_stream,
        in_rate: cfg.rate,
        out_rate,
        in_channels: in_cfg.channels,
        out_channels: out_cfg.channels,
        locked,
    })
}

fn error_cb(shared: Arc<Shared>, side: &'static str) -> impl FnMut(cpal::StreamError) + Send {
    move |e| shared.set_error(format!("{side}: {e}"))
}

macro_rules! by_format {
    ($fmt:expr, $build:ident, $($args:expr),*) => {
        match $fmt {
            SampleFormat::F32 => $build::<f32>($($args),*),
            SampleFormat::I16 => $build::<i16>($($args),*),
            SampleFormat::I32 => $build::<i32>($($args),*),
            other => Err(cpal::BuildStreamError::BackendSpecific {
                err: cpal::BackendSpecificError {
                    description: format!("unsupported sample format {other}"),
                },
            }),
        }
    };
}

fn build_input(
    dev: &cpal::Device,
    cfg: &StreamConfig,
    fmt: SampleFormat,
    side: InputSide,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    fn typed<T>(
        dev: &cpal::Device,
        cfg: &StreamConfig,
        mut side: InputSide,
        shared: Arc<Shared>,
    ) -> Result<cpal::Stream, cpal::BuildStreamError>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        dev.build_input_stream(
            cfg,
            move |data: &[T], _: &cpal::InputCallbackInfo| side.run(data),
            error_cb(shared, "input"),
            None,
        )
    }
    by_format!(fmt, typed, dev, cfg, side, shared)
}

fn build_output(
    dev: &cpal::Device,
    cfg: &StreamConfig,
    fmt: SampleFormat,
    side: OutputSide,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    fn typed<T>(
        dev: &cpal::Device,
        cfg: &StreamConfig,
        mut side: OutputSide,
        shared: Arc<Shared>,
    ) -> Result<cpal::Stream, cpal::BuildStreamError>
    where
        T: SizedSample + FromSample<f32>,
    {
        dev.build_output_stream(
            cfg,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| side.run(data),
            error_cb(shared, "output"),
            None,
        )
    }
    by_format!(fmt, typed, dev, cfg, side, shared)
}

// ---- Input side ----

struct InputSide {
    shared: Arc<Shared>,
    dsps: Vec<Dsp>,
    active: usize,
    model_threshold: f32,
    nch: usize,
    route: Route,
    l: Vec<f32>,
    r: Vec<f32>,
    prod: rtrb::Producer<f32>,
    samples_done: u64,
}

impl InputSide {
    fn run<T>(&mut self, data: &[T])
    where
        T: Sample,
        f32: FromSample<T>,
    {
        let sh = &*self.shared;
        sh.in_callbacks.fetch_add(1, Relaxed);
        let frames = data.len() / self.nch.max(1);
        sh.in_block_max
            .fetch_max(u32::try_from(frames).unwrap_or(u32::MAX), Relaxed);

        // Controls (same logic as the plugin's process_channels).
        let sens = f32::from_bits(sh.sensitivity.load(Relaxed));
        let threshold = sensitivity_to_threshold(sens, self.model_threshold);
        let mode = if sh.bypass.load(Relaxed) {
            OutputMode::Bypass
        } else {
            match monitor_from(sh.monitor.load(Relaxed)) {
                Monitor::Declick => OutputMode::Declick,
                Monitor::ClicksOnly => OutputMode::ClicksOnly,
            }
        };
        let wanted = usize::from(sh.protection.load(Relaxed)).min(self.dsps.len() - 1);
        if wanted != self.active {
            self.active = wanted;
            let dsp = &mut self.dsps[wanted];
            dsp.set_threshold(threshold);
            dsp.set_output(mode);
            dsp.reset();
            self.samples_done = 0;
        }
        let dsp = &mut self.dsps[self.active];
        dsp.set_threshold(threshold);
        dsp.set_output(mode);

        let (a, b) = (usize::from(self.route.left), usize::from(self.route.right));
        let mono = self.route.is_mono();
        let mut peak = 0.0f32;
        let mut start = 0;
        while start < frames {
            let n = (frames - start).min(CHUNK);
            for i in 0..n {
                let f = &data[(start + i) * self.nch..];
                let (x, y) = (f32::from_sample(f[a]), f32::from_sample(f[b]));
                peak = peak.max(x.abs()).max(y.abs());
                self.l[i] = x;
                self.r[i] = y;
            }
            if mono {
                dsp.process_mono(&mut self.l[..n]);
                self.r[..n].copy_from_slice(&self.l[..n]);
            } else {
                dsp.process(&mut self.l[..n], &mut self.r[..n]);
            }
            for i in 0..n {
                if self.prod.slots() >= 2 {
                    let _ = self.prod.push(self.l[i]);
                    let _ = self.prod.push(self.r[i]);
                } else {
                    sh.overflows.fetch_add(1, Relaxed);
                }
            }
            start += n;
        }
        self.samples_done += frames as u64;
        if peak.is_finite() {
            sh.peak.fetch_max(peak.to_bits(), Relaxed);
        }
        let st = dsp.stats();
        sh.stats.publish(
            st.regions,
            st.samples_repaired,
            self.samples_done,
            dsp.latency_samples(),
            dsp.sample_rate(),
        );
    }
}

// ---- Output side ----

struct OutputSide {
    shared: Arc<Shared>,
    cons: rtrb::Consumer<f32>,
    rs: Resampler,
    nch: usize,
    route: Route,
    in_rate: f64,
    /// Input frames per output frame (nominal).
    ratio: f64,
}

impl OutputSide {
    fn run<T>(&mut self, data: &mut [T])
    where
        T: Sample + FromSample<f32>,
    {
        let sh = &*self.shared;
        sh.out_callbacks.fetch_add(1, Relaxed);
        let nch = self.nch.max(1);
        let frames = data.len() / nch;
        // Target fill: two of the larger block (in input frames) + 5 ms.
        let in_block = f64::from(sh.in_block_max.load(Relaxed));
        let out_block = frames as f64 * self.ratio;
        let target = 2.0 * in_block.max(out_block) + 0.005 * self.in_rate;

        // Ring fill in stereo frames.
        let fill = self.cons.slots() / 2;
        let cons = &mut self.cons;
        let mut pop = || {
            if cons.slots() >= 2 {
                Some([cons.pop().unwrap_or(0.0), cons.pop().unwrap_or(0.0)])
            } else {
                None
            }
        };
        self.rs.begin_block(fill, target, &mut pop);
        let (a, b) = (usize::from(self.route.left), usize::from(self.route.right));
        let mono = self.route.is_mono();
        for f in data.chunks_mut(nch) {
            let [l, r] = self.rs.next(&mut pop);
            for s in f.iter_mut() {
                *s = T::EQUILIBRIUM;
            }
            if mono {
                f[a] = T::from_sample(0.5 * (l + r));
            } else {
                f[a] = T::from_sample(l);
                f[b] = T::from_sample(r);
            }
        }
        sh.underruns.store(self.rs.underruns, Relaxed);
        sh.resyncs.store(self.rs.resyncs, Relaxed);
        let trim = (self.rs.step() / self.ratio - 1.0) * 1.0e6;
        sh.trim_ppm.store((trim as f32).to_bits(), Relaxed);
    }
}

/// One-paragraph summary of a running session's counters.
pub fn report(s: &Session, sh: &Shared) -> String {
    let stats = &sh.stats;
    let trim = f32::from_bits(sh.trim_ppm.load(Relaxed));
    let peak = f32::from_bits(sh.peak.load(Relaxed));
    format!(
        "in {} Hz {} ch, out {} Hz {} ch, {}; input callbacks {} (max block {}), output \
         callbacks {}; processed {} frames, {} clicks repaired; underruns {}, overflows {}, \
         resyncs {}; clock trim {trim:+.1} ppm; input peak {}{}",
        s.in_rate,
        s.in_channels,
        s.out_rate,
        s.out_channels,
        if s.locked { "locked (bit-exact)" } else { "rate-matched" },
        sh.in_callbacks.load(Relaxed),
        sh.in_block_max.load(Relaxed),
        sh.out_callbacks.load(Relaxed),
        stats.samples.load(Relaxed),
        stats.regions.load(Relaxed),
        sh.underruns.load(Relaxed),
        sh.overflows.load(Relaxed),
        sh.resyncs.load(Relaxed),
        if peak > 0.0 {
            format!("{:.1} dBFS", 20.0 * peak.log10())
        } else {
            "silence".into()
        },
        sh.error
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .map(|e| format!("; ERROR {e}"))
            .unwrap_or_default(),
    )
}

// ---- Probe ----

/// Open `name` natively for `seconds` and report what arrives (CLI
/// diagnostics: `2ndpressing --probe "H6"`).
pub fn probe(name: &str, seconds: f32) -> Result<String, String> {
    let dev = find(true, name).ok_or_else(|| format!("no input device named {name:?}"))?;
    let info = describe(&dev, true).ok_or("cannot read the device's formats")?;
    let (cfg, fmt) = native_config(&dev, true, info.default_rate)?;
    let calls = Arc::new(AtomicU64::new(0));
    let frames = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU32::new(0));
    let err: Arc<Mutex<Option<String>>> = Arc::default();
    let nch = usize::from(cfg.channels);
    let (c2, f2, p2, e2) = (calls.clone(), frames.clone(), peak.clone(), err.clone());
    let stream = dev
        .build_input_stream(
            &cfg,
            move |d: &[f32], _: &cpal::InputCallbackInfo| {
                c2.fetch_add(1, Relaxed);
                f2.fetch_add((d.len() / nch) as u64, Relaxed);
                let p = d.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                p2.fetch_max(p.to_bits(), Relaxed);
            },
            move |e| {
                if let Ok(mut g) = e2.lock() {
                    *g = Some(e.to_string());
                }
            },
            None,
        )
        .map_err(|e| format!("open failed: {e} (format {fmt})"))?;
    stream.play().map_err(|e| format!("start failed: {e}"))?;
    std::thread::sleep(std::time::Duration::from_secs_f32(seconds));
    drop(stream);
    let n = calls.load(Relaxed);
    let pk = f32::from_bits(peak.load(Relaxed));
    let db = if pk > 0.0 {
        format!("{:.1} dBFS", 20.0 * pk.log10())
    } else {
        "digital silence".into()
    };
    let mut s = format!(
        "{name}: {} ch @ {} Hz ({fmt}); {n} callbacks, {} frames in {seconds} s; peak {db}",
        cfg.channels,
        cfg.sample_rate.0,
        frames.load(Relaxed)
    );
    if let Ok(g) = err.lock() {
        if let Some(e) = g.as_ref() {
            s.push_str(&format!("; stream error: {e}"));
        }
    }
    if n == 0 {
        s.push_str(
            "\n  -> no audio callbacks at all: the device is busy (another app holds it) or \
             access to audio input is blocked (System Settings > Privacy & Security > Microphone).",
        );
    } else if pk == 0.0 {
        s.push_str(
            "\n  -> callbacks arrive but carry exact zeros: microphone access is probably denied \
             for the app that launched this (Terminal), or nothing is plugged in.",
        );
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_lists() {
        let r = routes(6);
        assert_eq!(r.len(), 3 + 6);
        assert_eq!(r[0], Route::FIRST_PAIR);
        assert_eq!(r[2].label(), "5–6");
        assert_eq!(r[3].label(), "1 mono");
        assert_eq!(routes(1), vec![Route { left: 0, right: 0 }]);
        assert!(Route::FIRST_PAIR.fits(2) && !Route::FIRST_PAIR.fits(1));
    }

    #[test]
    fn index_round_trips() {
        for p in [
            PercussionProtection::Off,
            PercussionProtection::Standard,
            PercussionProtection::Extended,
        ] {
            assert_eq!(protection_from(protection_index(p)), p);
        }
        for m in [Monitor::Declick, Monitor::ClicksOnly] {
            assert_eq!(monitor_from(monitor_index(m)), m);
        }
    }
}
