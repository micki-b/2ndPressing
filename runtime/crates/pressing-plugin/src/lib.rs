//! 2ndPressing Declicker: nih-plug VST3/CLAP plugin and standalone app
//! around [`pressing_core::Declicker`].
//!
//! Window: a custom editor (see [`editor`]); hosts can still show their
//! generic view. Parameters:
//! - **Bypass** (`bypass`): the host's bypass parameter. Bypassed output is
//!   the input delayed by the same latency; switching crossfades over
//!   [`dsp::CROSSFADE_SECONDS`] and never changes the reported latency.
//! - **Sensitivity** (`sensitivity`, 0..100 %, default 50 %): mapped to the
//!   detection threshold by [`sensitivity_to_threshold`]. 0 % is gentle
//!   (threshold [`THRESHOLD_GENTLE`]), 50 % is the model's tuned threshold
//!   (2.75 for ClickNet R2), 100 % is aggressive (threshold
//!   [`THRESHOLD_AGGRESSIVE`]); linear in threshold on each half.
//! - **Monitor** (`monitor`): `Declick` (repaired audio) or `Clicks only`
//!   (what was removed: delayed input minus repaired output).
//! - **Percussion protection** (`protection`, default Extended): leave claps
//!   and drums that sit on a beat grid alone (rotation-periodic scratches are
//!   still repaired). `Off` keeps the 32 ms latency for live monitoring;
//!   `Standard` adds ~1 s and `Extended` ~2 s of look-ahead (Extended also
//!   protects the start of a rhythmic passage). One processor per mode is
//!   built in `initialize`; switching restarts the new one from silence and
//!   reports the new latency to the host.
//!
//! Sample rates (see [`rate_support`]): 44.1 and 48 kHz are native (the
//! model was trained at 48 kHz). 88.2 kHz and up is the spec's experimental
//! range: detection runs on a decimated signal with `factor =
//! round(sr / 48000)` and latency grows to `1536 * factor` plus the FIR
//! delay. The engine picks its path from `round(sr / 48000)` alone, so other
//! rates still run but are outside the spec: below about 72 kHz (e.g. 22.05,
//! 32 or 64 kHz) the 48 kHz model runs undecimated on a different time
//! scale, and from 72 kHz up the decimated path is already used. The plugin
//! does not reject such rates (a failed `initialize` leaves most hosts with a
//! dead insert); it logs a warning via `nih_warn!` instead. Latency is
//! reported in `initialize`, which nih-plug calls again whenever the sample
//! rate changes.
//!
//! Cargo features: `standalone` (off by default) builds the
//! `pressing-standalone` binary and pulls in nih-plug's audio backends
//! (CoreAudio / WASAPI / ALSA + JACK). It is opt-in so that bundling the
//! VST3/CLAP plugin (`cargo xtask bundle pressing-plugin --release`) never
//! needs ALSA or JACK development files on Linux. Build the app with
//! `cargo build -p pressing-plugin --release --features standalone`.

pub mod dsp;
pub mod editor;
pub mod info;

use std::sync::Arc;

use nih_plug::prelude::*;
use pressing_core::{Model, OutputMode, Protection};

use crate::dsp::Dsp;

// ---- Sensitivity mapping ----

/// Threshold at 0 % sensitivity (fewest detections).
pub const THRESHOLD_GENTLE: f32 = 6.0;
/// Threshold at 100 % sensitivity (most detections).
pub const THRESHOLD_AGGRESSIVE: f32 = 1.0;
/// Sensitivity that maps to the model's own threshold.
pub const SENSITIVITY_DEFAULT: f32 = 50.0;

/// Sensitivity slider increment in percent (the knob snaps to this).
pub const SENSITIVITY_STEP: f32 = 5.0;

/// The model threshold, clamped strictly inside the knob's range so both
/// halves of the mapping stay monotonic.
fn centre_threshold(model_threshold: f32) -> f32 {
    if model_threshold.is_finite() {
        model_threshold.clamp(THRESHOLD_AGGRESSIVE + 1e-3, THRESHOLD_GENTLE - 1e-3)
    } else {
        0.5 * (THRESHOLD_AGGRESSIVE + THRESHOLD_GENTLE)
    }
}

/// Map a sensitivity in percent (clamped to 0..=100) to a detection
/// threshold in logit units. Piecewise linear: 0 % -> [`THRESHOLD_GENTLE`],
/// 50 % -> `model_threshold`, 100 % -> [`THRESHOLD_AGGRESSIVE`].
pub fn sensitivity_to_threshold(percent: f32, model_threshold: f32) -> f32 {
    let t0 = centre_threshold(model_threshold);
    let p = if percent.is_finite() {
        percent.clamp(0.0, 100.0)
    } else {
        SENSITIVITY_DEFAULT
    };
    if p <= 50.0 {
        THRESHOLD_GENTLE + (t0 - THRESHOLD_GENTLE) * (p / 50.0)
    } else {
        t0 + (THRESHOLD_AGGRESSIVE - t0) * ((p - 50.0) / 50.0)
    }
}

/// Inverse of [`sensitivity_to_threshold`] (threshold clamped to the knob's
/// range).
pub fn threshold_to_sensitivity(threshold: f32, model_threshold: f32) -> f32 {
    let t0 = centre_threshold(model_threshold);
    let t = threshold.clamp(THRESHOLD_AGGRESSIVE, THRESHOLD_GENTLE);
    if t >= t0 {
        50.0 * (THRESHOLD_GENTLE - t) / (THRESHOLD_GENTLE - t0)
    } else {
        50.0 + 50.0 * (t0 - t) / (t0 - THRESHOLD_AGGRESSIVE)
    }
}

// ---- Sample-rate support ----

/// How a host sample rate relates to the runtime spec (docs/runtime-v1.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateSupport {
    /// 44.1 or 48 kHz: detection runs directly at the host rate.
    Native,
    /// 88.2 kHz up to the engine's largest decimation factor: decimated
    /// detection with `factor = round(sr / 48000)`; quality unvalidated.
    Experimental,
    /// Anything else (including non-finite or non-positive rates). The
    /// engine still runs, with the path chosen by
    /// [`pressing_core::engine::rate_factor`], but behaviour is not covered
    /// by the spec.
    Unsupported,
}

/// Classify a host sample rate (tolerance 0.5 Hz on the native rates).
pub fn rate_support(sample_rate: f32) -> RateSupport {
    let sr = f64::from(sample_rate);
    if !(sr.is_finite() && sr > 0.0) {
        return RateSupport::Unsupported;
    }
    if (sr - 44_100.0).abs() < 0.5 || (sr - 48_000.0).abs() < 0.5 {
        return RateSupport::Native;
    }
    let max_factor = pressing_core::engine::MAX_RATE_FACTOR as f64;
    // Above (max + 0.5) * 48 kHz the engine clamps the decimation factor.
    if sr >= 88_200.0 - 0.5 && sr < (max_factor + 0.5) * 48_000.0 {
        RateSupport::Experimental
    } else {
        RateSupport::Unsupported
    }
}

/// Log how the engine will handle `sample_rate` (called when the engine is
/// rebuilt in `initialize`, never from `process`).
fn log_rate_support(sample_rate: f32) {
    let factor = pressing_core::engine::rate_factor(sample_rate);
    match rate_support(sample_rate) {
        RateSupport::Native => {}
        RateSupport::Experimental => nih_log!(
            "2ndPressing: {sample_rate} Hz uses the experimental decimated detection path \
             (factor {factor})"
        ),
        RateSupport::Unsupported if factor == 1 => nih_warn!(
            "2ndPressing: {sample_rate} Hz is not a supported sample rate (44.1/48 kHz native, \
             >= 88.2 kHz experimental); the 48 kHz model runs undecimated and detection \
             quality is unvalidated"
        ),
        RateSupport::Unsupported => nih_warn!(
            "2ndPressing: {sample_rate} Hz is not a supported sample rate (44.1/48 kHz native, \
             >= 88.2 kHz experimental); running the decimated path with factor {factor}, \
             detection quality is unvalidated"
        ),
    }
}

// ---- Parameters ----

/// What the plugin outputs when not bypassed.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monitor {
    /// Repaired audio.
    #[id = "declick"]
    #[name = "Declick"]
    Declick,
    /// Delayed input minus repaired output: only what was removed.
    #[id = "clicks"]
    #[name = "Clicks only"]
    ClicksOnly,
}

/// Percussion protection (user-facing names include the latency).
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PercussionProtection {
    /// No protection: lowest latency, for live monitoring.
    #[id = "off"]
    #[name = "Off (32 ms)"]
    Off,
    /// Claps and drums on a beat grid are left alone; ~1 s latency.
    #[id = "standard"]
    #[name = "Standard (~1 s)"]
    Standard,
    /// Like Standard with ~2 s look-ahead: also protects the start of a
    /// rhythmic passage, at ~1 point of click recall.
    #[id = "extended"]
    #[name = "Extended (~2 s)"]
    Extended,
}

impl PercussionProtection {
    /// Engine mode.
    pub fn engine(self) -> Protection {
        match self {
            PercussionProtection::Off => Protection::Off,
            PercussionProtection::Standard => Protection::Standard,
            PercussionProtection::Extended => Protection::Extended,
        }
    }

    /// Index into the plugin's per-mode processors.
    fn index(self) -> usize {
        self as usize
    }
}

/// Plugin parameters (see the crate docs for semantics).
#[derive(Params)]
pub struct PressingParams {
    /// Host bypass: delayed dry signal, same latency, crossfaded.
    #[id = "bypass"]
    pub bypass: BoolParam,
    /// Detection sensitivity in percent (see [`sensitivity_to_threshold`]).
    #[id = "sensitivity"]
    pub sensitivity: FloatParam,
    /// Output monitor mode.
    #[id = "monitor"]
    pub monitor: EnumParam<Monitor>,
    /// Percussion protection (changes the reported latency).
    #[id = "protection"]
    pub protection: EnumParam<PercussionProtection>,
    /// Window size/open state, saved with the project.
    // New key for the Instrument faceplate: windows saved at the old size are not restored.
    #[persist = "editor-state-b"]
    pub editor_state: Arc<nih_plug_egui::EguiState>,
}

impl PressingParams {
    /// The plugin's parameters. `_model_threshold` is kept for API stability;
    /// the Sensitivity display is the percentage only (see
    /// [`sensitivity_to_threshold`] for the mapping).
    pub fn new(_model_threshold: f32) -> Self {
        PressingParams {
            bypass: BoolParam::new("Bypass", false).make_bypass(),
            sensitivity: FloatParam::new(
                "Sensitivity",
                SENSITIVITY_DEFAULT,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_step_size(SENSITIVITY_STEP)
            // Short on purpose: hosts' generic editors (e.g. Logic's) give the
            // value a narrow fixed-width box. The threshold is an internal detail.
            .with_value_to_string(Arc::new(|v| format!("{v:.0} %")))
            .with_string_to_value(Arc::new(|s| {
                s.split_whitespace()
                    .next()?
                    .trim_end_matches('%')
                    .parse::<f32>()
                    .ok()
            })),
            monitor: EnumParam::new("Monitor", Monitor::Declick),
            protection: EnumParam::new("Percussion protection", PercussionProtection::Extended),
            editor_state: editor::default_state(),
        }
    }
}

// ---- Plugin ----

/// The 2ndPressing Declicker plugin.
pub struct SecondPressing {
    params: Arc<PressingParams>,
    model: Model,
    /// One processor per percussion protection mode, built in `initialize`
    /// (allocation happens there, never in `process`).
    dsps: Vec<Dsp>,
    /// Index of the processor in use.
    active: usize,
    /// Latency to report from the next `process` call after a mode switch.
    pending_latency: Option<u32>,
    /// Test hook: overrides the protection parameter (hosts never set it).
    protection_override: Option<PercussionProtection>,
    /// Live numbers for the window (atomics; written by `process`).
    ui_stats: Arc<editor::UiStats>,
    /// Input samples processed by the active engine since it (re)started
    /// (the clicks-per-minute meter's clock).
    samples_done: u64,
}

impl Default for SecondPressing {
    fn default() -> Self {
        let model = Model::embedded();
        SecondPressing {
            params: Arc::new(PressingParams::new(model.post.threshold)),
            model,
            dsps: Vec::new(),
            active: PercussionProtection::Extended.index(),
            pending_latency: None,
            protection_override: None,
            ui_stats: Arc::new(editor::UiStats::default()),
            samples_done: 0,
        }
    }
}

impl SecondPressing {
    fn output_mode(&self) -> OutputMode {
        if self.params.bypass.value() {
            OutputMode::Bypass
        } else {
            match self.params.monitor.value() {
                Monitor::Declick => OutputMode::Declick,
                Monitor::ClicksOnly => OutputMode::ClicksOnly,
            }
        }
    }

    fn threshold(&self) -> f32 {
        sensitivity_to_threshold(self.params.sensitivity.value(), self.model.post.threshold)
    }

    /// Latency the engine reports, or `None` before the first `initialize`.
    pub fn latency_samples(&self) -> Option<usize> {
        self.dsps.get(self.active).map(Dsp::latency_samples)
    }

    /// Latency change waiting to be reported to the host (set when the
    /// percussion protection mode switches; taken by `process`).
    pub fn take_pending_latency(&mut self) -> Option<u32> {
        self.pending_latency.take()
    }

    fn wanted(&self) -> usize {
        self.protection_override
            .unwrap_or_else(|| self.params.protection.value())
            .index()
    }

    /// Test hook: act as if the protection parameter were `p` (nih-plug
    /// parameters cannot be set outside a host). Takes effect on the next
    /// `process_channels`, exactly like a parameter change.
    #[doc(hidden)]
    pub fn set_protection_override(&mut self, p: Option<PercussionProtection>) {
        self.protection_override = p;
    }

    /// The whole body of [`Plugin::process`] apart from unwrapping the host
    /// buffer: reads the current parameters, applies them, and processes
    /// `channels` in place (two or more: first two as stereo; one: mono;
    /// none: no-op). Does nothing before `initialize`. Never allocates; it
    /// is public so tests can drive the real process path without a host.
    pub fn process_channels(&mut self, channels: &mut [&mut [f32]]) {
        let threshold = self.threshold();
        let mode = self.output_mode();
        let wanted = self.wanted();
        if wanted != self.active && wanted < self.dsps.len() {
            // Switch processors: the new one restarts from silence (its
            // delay line is empty) and the host is told the new latency.
            self.active = wanted;
            let dsp = &mut self.dsps[wanted];
            dsp.set_threshold(threshold);
            dsp.set_output(mode);
            dsp.reset();
            self.pending_latency = Some(u32::try_from(dsp.latency_samples()).unwrap_or(u32::MAX));
            self.samples_done = 0;
        }
        let Some(dsp) = self.dsps.get_mut(self.active) else {
            return;
        };
        dsp.set_threshold(threshold);
        dsp.set_output(mode);
        let n = match channels {
            [l, r, ..] => {
                dsp.process(l, r);
                l.len().min(r.len())
            }
            [m] => {
                dsp.process_mono(m);
                m.len()
            }
            [] => 0,
        };
        self.samples_done += n as u64;
        let st = dsp.stats();
        self.ui_stats.publish(
            st.regions,
            st.samples_repaired,
            self.samples_done,
            dsp.latency_samples(),
            dsp.sample_rate(),
        );
    }
}

impl Plugin for SecondPressing {
    const NAME: &'static str = "2ndPressing Declicker";
    const VENDOR: &'static str = "2ndPressing";
    const URL: &'static str = env!("CARGO_PKG_REPOSITORY");
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[],
            aux_output_ports: &[],
            names: PortNames::const_default(),
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.ui_stats.clone(),
            self.model.post.threshold,
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate;
        if self.dsps.first().is_none_or(|d| d.sample_rate() != sr) {
            log_rate_support(sr);
            self.dsps = [
                PercussionProtection::Off,
                PercussionProtection::Standard,
                PercussionProtection::Extended,
            ]
            .map(|p| Dsp::new_with(&self.model, sr, p.engine()))
            .into();
        }
        self.active = self.wanted();
        self.pending_latency = None;
        self.samples_done = 0;
        let threshold = self.threshold();
        let mode = self.output_mode();
        let Some(dsp) = self.dsps.get_mut(self.active) else {
            return false;
        };
        dsp.set_threshold(threshold);
        dsp.set_output(mode);
        dsp.reset();
        let latency = u32::try_from(dsp.latency_samples()).unwrap_or(u32::MAX);
        context.set_latency_samples(latency);
        true
    }

    fn reset(&mut self) {
        if let Some(dsp) = self.dsps.get_mut(self.active) {
            dsp.reset();
        }
        self.samples_done = 0;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.process_channels(buffer.as_slice());
        if let Some(latency) = self.take_pending_latency() {
            context.set_latency_samples(latency);
        }
        // Keep processing after the input stops: the delay line still holds
        // `latency` samples of output.
        ProcessStatus::Normal
    }
}

impl ClapPlugin for SecondPressing {
    const CLAP_ID: &'static str = "com.secondpressing.declicker";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Near-real-time LP click and pop removal (ClickNet detector + LSAR repair)");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for SecondPressing {
    const VST3_CLASS_ID: [u8; 16] = *b"2ndPressDeclick1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nih_export_clap!(SecondPressing);

/// The VST3 entry point. The allow is for a future-incompatibility warning
/// raised inside the upstream `vst3_com::vtable` macro expansion (not this
/// crate's code); scoped to this module so it hides nothing else.
#[allow(semicolon_in_expressions_from_non_local_macros)]
mod vst3_entry {
    use super::SecondPressing;
    use nih_plug::prelude::*;

    nih_export_vst3!(SecondPressing);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitivity_mapping() {
        let t0 = Model::embedded().post.threshold;
        assert_eq!(t0, 2.75);
        assert_eq!(sensitivity_to_threshold(0.0, t0), THRESHOLD_GENTLE);
        assert_eq!(sensitivity_to_threshold(50.0, t0), t0);
        assert_eq!(sensitivity_to_threshold(100.0, t0), THRESHOLD_AGGRESSIVE);
        assert_eq!(sensitivity_to_threshold(-5.0, t0), THRESHOLD_GENTLE);
        assert_eq!(sensitivity_to_threshold(f32::NAN, t0), t0);
        let mut prev = f32::INFINITY;
        for i in 0..=1000 {
            let p = i as f32 * 0.1;
            let t = sensitivity_to_threshold(p, t0);
            assert!(t <= prev, "not monotonic at {p}");
            prev = t;
            let back = threshold_to_sensitivity(t, t0);
            assert!((back - p).abs() < 1e-3, "{p} -> {t} -> {back}");
        }
    }

    #[test]
    fn rate_support_classes() {
        use RateSupport::*;
        for (sr, want) in [
            (44_100.0, Native),
            (48_000.0, Native),
            (88_200.0, Experimental),
            (96_000.0, Experimental),
            (192_000.0, Experimental),
            (384_000.0, Experimental),
            // Outside the spec: the engine still picks a path from
            // round(sr / 48000), so these must be flagged.
            (8_000.0, Unsupported),
            (22_050.0, Unsupported),
            (32_000.0, Unsupported),
            (64_000.0, Unsupported),
            (72_000.0, Unsupported),
            (88_000.0, Unsupported),
            (500_000.0, Unsupported),
            (0.0, Unsupported),
            (-48_000.0, Unsupported),
            (f32::NAN, Unsupported),
            (f32::INFINITY, Unsupported),
        ] {
            assert_eq!(rate_support(sr), want, "sr = {sr}");
        }
        // Experimental rates really take the decimated path; native ones
        // do not.
        assert_eq!(pressing_core::engine::rate_factor(44_100.0), 1);
        assert_eq!(pressing_core::engine::rate_factor(88_200.0), 2);
        // 72 kHz is unsupported yet already decimated (the doc fix).
        assert_eq!(pressing_core::engine::rate_factor(72_000.0), 2);
    }

    #[test]
    fn sensitivity_text_round_trip() {
        let params = PressingParams::new(2.75);
        let s = &params.sensitivity;
        let text = s.normalized_value_to_string(0.5, true);
        assert_eq!(text, "50 %");
        assert_eq!(s.string_to_normalized_value(&text), Some(0.5));
        assert_eq!(s.string_to_normalized_value("75%"), Some(0.75));
    }
}
