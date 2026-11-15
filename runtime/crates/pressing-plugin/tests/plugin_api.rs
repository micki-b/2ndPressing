//! The plugin through its nih-plug entry points (`Plugin::initialize` with a
//! stand-in host context, then the process path), without a real host.

mod common;

use pressing_core::{Model, OutputMode};
use pressing_plugin::dsp::Dsp;
use pressing_plugin::SecondPressing;

/// Latency is reported on every `initialize`, including after a sample-rate
/// change on the same instance, and matches the engine for that rate.
#[test]
fn initialize_reports_latency_on_every_rate_change() {
    let model = Model::embedded();
    for channels in [2u32, 1] {
        let mut plugin = SecondPressing::default();
        assert_eq!(plugin.latency_samples(), None);
        for (sr, want) in [
            (48_000.0f32, 1536u32),
            (44_100.0, 1536),
            (96_000.0, 2 * 1536 + 64),
            (32_000.0, 1536),
            (48_000.0, 1536),
            // Same rate twice: engine kept, latency still reported.
            (48_000.0, 1536),
        ] {
            let (ok, latency) = common::initialize(&mut plugin, channels, sr);
            assert!(ok, "initialize at {sr} Hz");
            assert_eq!(latency, Some(want), "reported latency at {sr} Hz");
            assert_eq!(
                Dsp::new(&model, sr).latency_samples(),
                want as usize,
                "engine latency at {sr} Hz"
            );
            assert_eq!(plugin.latency_samples(), Some(want as usize));
        }
    }
}

/// Before `initialize` the process path is a no-op (no engine yet).
#[test]
fn process_before_initialize_is_a_no_op() {
    let mut plugin = SecondPressing::default();
    let mut l = common::signal(4_000, 0.01, 500, 0.8);
    let want = l.clone();
    plugin.process_channels(&mut [&mut l]);
    assert_eq!(l, want);
}

/// With default parameters (not bypassed, Monitor = Declick, 50 % = the
/// model threshold) the plugin's process path is bit-identical to the
/// engine wrapper configured the same way, for stereo and mono layouts.
#[test]
fn default_params_match_dsp_bit_exact() {
    let model = Model::embedded();
    let len = 20_000;
    for sr in [44_100.0f32, 48_000.0] {
        let l0 = common::signal(len, 0.01, 1_333, 0.8);
        let r0 = common::signal(len, 0.017, 1_333, -0.6);

        let mut dsp = Dsp::new(&model, sr);
        dsp.set_threshold(model.post.threshold);
        dsp.set_output(OutputMode::Declick);
        let (mut dl, mut dr) = (l0.clone(), r0.clone());
        for (cl, cr) in dl.chunks_mut(300).zip(dr.chunks_mut(300)) {
            dsp.process(cl, cr);
        }
        assert!(dsp.stats().regions > 0, "signal has detections at {sr} Hz");

        let mut plugin = SecondPressing::default();
        assert!(common::initialize(&mut plugin, 2, sr).0);
        let (mut pl, mut pr) = (l0.clone(), r0.clone());
        for (cl, cr) in pl.chunks_mut(300).zip(pr.chunks_mut(300)) {
            plugin.process_channels(&mut [cl, cr]);
        }
        assert_eq!(pl, dl, "left at {sr} Hz");
        assert_eq!(pr, dr, "right at {sr} Hz");

        let mut dsp = Dsp::new(&model, sr);
        dsp.set_threshold(model.post.threshold);
        dsp.set_output(OutputMode::Declick);
        let mut dm = l0.clone();
        for c in dm.chunks_mut(300) {
            dsp.process_mono(c);
        }
        let mut plugin = SecondPressing::default();
        assert!(common::initialize(&mut plugin, 1, sr).0);
        let mut pm = l0.clone();
        for c in pm.chunks_mut(300) {
            plugin.process_channels(&mut [c]);
        }
        assert_eq!(pm, dm, "mono at {sr} Hz");
    }
}
