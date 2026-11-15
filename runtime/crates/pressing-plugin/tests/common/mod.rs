//! Host stand-ins shared by the integration tests.

use std::cell::Cell;

use nih_plug::prelude::*;
use pressing_plugin::SecondPressing;

/// Minimal [`InitContext`] that records the reported latency.
#[derive(Default)]
pub struct MockInit {
    pub latency: Cell<Option<u32>>,
}

impl InitContext<SecondPressing> for MockInit {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Clap
    }

    fn execute(&self, _task: ()) {}

    fn set_latency_samples(&self, samples: u32) {
        self.latency.set(Some(samples));
    }

    fn set_current_voice_capacity(&self, _capacity: u32) {}
}

/// The plugin's layout with `channels` in and out (1 = mono, 2 = stereo).
pub fn layout(channels: u32) -> &'static AudioIOLayout {
    SecondPressing::AUDIO_IO_LAYOUTS
        .iter()
        .find(|l| l.main_input_channels.map(NonZeroU32::get) == Some(channels))
        .expect("layout")
}

/// Call `Plugin::initialize` like a host would; returns its result and the
/// latency it reported.
pub fn initialize(
    plugin: &mut SecondPressing,
    channels: u32,
    sample_rate: f32,
) -> (bool, Option<u32>) {
    let mut ctx = MockInit::default();
    let cfg = BufferConfig {
        sample_rate,
        min_buffer_size: None,
        max_buffer_size: 1024,
        process_mode: ProcessMode::Realtime,
    };
    let ok = plugin.initialize(layout(channels), &cfg, &mut ctx);
    (ok, ctx.latency.get())
}

/// Test signal: a sine plus a click every `click_every` samples from 2000 on.
pub fn signal(len: usize, freq: f32, click_every: usize, click: f32) -> Vec<f32> {
    let mut x: Vec<f32> = (0..len).map(|i| (i as f32 * freq).sin() * 0.3).collect();
    for i in (2_000..len).step_by(click_every) {
        x[i] += click;
    }
    x
}
