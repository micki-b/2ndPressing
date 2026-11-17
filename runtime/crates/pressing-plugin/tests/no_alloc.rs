//! The audio path must not allocate. In debug builds nih-plug's
//! `assert_process_allocs` feature installs `assert_no_alloc`'s global
//! allocator, so any allocation inside `assert_no_alloc(..)` aborts this test
//! binary. In release builds that allocator is compiled out, so these tests
//! are `#[ignore]`d there rather than passing without checking anything.

mod common;

use nih_plug::prelude::Plugin;
use pressing_core::{Model, OutputMode};
use pressing_plugin::dsp::Dsp;
use pressing_plugin::SecondPressing;

/// Native, experimental (decimated) and an out-of-spec rate.
const RATES: [f32; 4] = [44_100.0, 48_000.0, 96_000.0, 32_000.0];

#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "allocation guard only active in debug builds"
)]
fn dsp_process_does_not_allocate() {
    let model = Model::embedded();
    for sr in RATES {
        let mut dsp = Dsp::new(&model, sr);
        let len = 40_000;
        let mut l = common::signal(len, 0.01, 1_777, 0.8);
        let mut r = common::signal(len, 0.013, 1_777, -0.7);
        let mut m = l.clone();
        assert_no_alloc::assert_no_alloc(|| {
            let modes = [
                OutputMode::ClicksOnly,
                OutputMode::Bypass,
                OutputMode::Declick,
            ];
            for (k, (cl, cr)) in l.chunks_mut(1_000).zip(r.chunks_mut(1_000)).enumerate() {
                dsp.set_threshold(if k % 2 == 0 { 2.0 } else { 2.75 });
                dsp.set_output(modes[k % 3]);
                dsp.process(cl, cr);
            }
            for c in m.chunks_mut(512) {
                dsp.process_mono(c);
            }
            dsp.reset();
        });
        assert!(dsp.stats().regions == 0, "reset clears stats");
    }
}

/// Drives the plugin itself: `Plugin::initialize` (allowed to allocate),
/// then `SecondPressing::process_channels`, which is the entire body of
/// `Plugin::process` apart from `buffer.as_slice()`, under the guard.
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "allocation guard only active in debug builds"
)]
fn plugin_process_does_not_allocate() {
    let len = 30_000;
    for channels in [2u32, 1] {
        let mut plugin = SecondPressing::default();
        for sr in RATES {
            let (ok, latency) = common::initialize(&mut plugin, channels, sr);
            assert!(ok, "initialize at {sr} Hz");
            assert!(latency.is_some(), "latency reported at {sr} Hz");
            let mut l = common::signal(len, 0.01, 1_333, 0.8);
            let mut r = common::signal(len, 0.017, 1_333, -0.6);
            let before = l.clone();
            assert_no_alloc::assert_no_alloc(|| {
                // Irregular host block sizes, including 1 and > CHUNK.
                let mut start = 0;
                for (k, n) in [1usize, 37, 256, 1_000, 64, 513].iter().cycle().enumerate() {
                    if start >= len {
                        break;
                    }
                    let end = (start + n).min(len);
                    let (cl, cr) = (&mut l[start..end], &mut r[start..end]);
                    if channels == 2 {
                        plugin.process_channels(&mut [cl, cr]);
                    } else {
                        plugin.process_channels(&mut [cl]);
                    }
                    if k == 20 {
                        Plugin::reset(&mut plugin);
                    }
                    start = end;
                }
            });
            assert_ne!(l, before, "audio passed through the plugin at {sr} Hz");
        }
    }
}
