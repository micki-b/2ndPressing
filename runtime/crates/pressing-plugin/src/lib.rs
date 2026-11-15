//! 2ndPressing Declicker: nih-plug VST3/CLAP plugin and standalone app
//! around [`pressing_core::Declicker`].
//!
//! Parameters (generic host UI):
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

use std::sync::Arc;

use nih_plug::prelude::*;
use pressing_core::{Model, OutputMode};

use crate::dsp::Dsp;

// ---- Sensitivity mapping ----

/// Threshold at 0 % sensitivity (fewest detections).
pub const THRESHOLD_GENTLE: f32 = 6.0;
/// Threshold at 100 % sensitivity (most detections).
pub const THRESHOLD_AGGRESSIVE: f32 = 1.0;
/// Sensitivity that maps to the model's own threshold.
pub const SENSITIVITY_DEFAULT: f32 = 50.0;

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
}

impl PressingParams {
    /// Parameters whose sensitivity display is relative to `model_threshold`.
    pub fn new(model_threshold: f32) -> Self {
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
            .with_step_size(0.1)
            .with_value_to_string(Arc::new(move |v| {
                let t = sensitivity_to_threshold(v, model_threshold);
                format!("{v:.1} % (threshold {t:.2})")
            }))
            .with_string_to_value(Arc::new(|s| {
                s.split_whitespace()
                    .next()?
                    .trim_end_matches('%')
                    .parse::<f32>()
                    .ok()
            })),
            monitor: EnumParam::new("Monitor", Monitor::Declick),
        }
    }
}

// ---- Plugin ----

/// The 2ndPressing Declicker plugin.
pub struct SecondPressing {
    params: Arc<PressingParams>,
    model: Model,
    /// Built in `initialize` (allocation happens there, never in `process`).
    dsp: Option<Dsp>,
}

impl Default for SecondPressing {
    fn default() -> Self {
        let model = Model::embedded();
        SecondPressing {
            params: Arc::new(PressingParams::new(model.post.threshold)),
            model,
            dsp: None,
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
        self.dsp.as_ref().map(Dsp::latency_samples)
    }

    /// The whole body of [`Plugin::process`] apart from unwrapping the host
    /// buffer: reads the current parameters, applies them, and processes
    /// `channels` in place (two or more: first two as stereo; one: mono;
    /// none: no-op). Does nothing before `initialize`. Never allocates; it
    /// is public so tests can drive the real process path without a host.
    pub fn process_channels(&mut self, channels: &mut [&mut [f32]]) {
        let threshold = self.threshold();
        let mode = self.output_mode();
        let Some(dsp) = self.dsp.as_mut() else {
            return;
        };
        dsp.set_threshold(threshold);
        dsp.set_output(mode);
        match channels {
            [l, r, ..] => dsp.process(l, r),
            [m] => dsp.process_mono(m),
            [] => {}
        }
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate;
        if self.dsp.as_ref().is_none_or(|d| d.sample_rate() != sr) {
            log_rate_support(sr);
            self.dsp = Some(Dsp::new(&self.model, sr));
        }
        let threshold = self.threshold();
        let mode = self.output_mode();
        let Some(dsp) = self.dsp.as_mut() else {
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
        if let Some(dsp) = self.dsp.as_mut() {
            dsp.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.process_channels(buffer.as_slice());
        // Keep processing after the input stops: the delay line still holds
        // `latency` samples of output.
        ProcessStatus::Normal
    }
}

impl ClapPlugin for SecondPressing {
    const CLAP_ID: &'static str = "org.2ndpressing.declicker";
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
        assert_eq!(text, "50.0 % (threshold 2.75)");
        assert_eq!(s.string_to_normalized_value(&text), Some(0.5));
        assert_eq!(s.string_to_normalized_value("75%"), Some(0.75));
    }
}
