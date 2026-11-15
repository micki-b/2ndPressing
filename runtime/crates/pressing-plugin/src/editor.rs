//! Custom plugin window (nih_plug_egui).
//!
//! Layout (fixed size [`WIDTH`] x [`HEIGHT`] logical pixels):
//! - Sensitivity: a wide slider whose value is drawn in a roomy box.
//! - Percussion protection: three buttons (Off / Standard / Extended), each
//!   with its latency underneath.
//! - Monitor (Declick / Clicks only) and Bypass.
//! - Clicks-per-minute meter: rate over the last 10 s of audio (log bar,
//!   coloured clean / moderate / heavy) and a 60 s history line.
//! - Status: clicks repaired since the engine started, latency, sample rate.
//!
//! Live numbers come from [`UiStats`], written by the audio thread with
//! relaxed atomic stores (no locks, no allocation in `process`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_egui::egui::{
    self, Align, Align2, Color32, FontId, Layout, Pos2, RichText, Sense, Stroke, Vec2,
};
use nih_plug_egui::{create_egui_editor, widgets, EguiState};

use crate::{Monitor, PercussionProtection, PressingParams};

/// Window size in logical pixels.
pub const WIDTH: u32 = 520;
pub const HEIGHT: u32 = 470;

const ACCENT: Color32 = Color32::from_rgb(0xE0, 0x9F, 0x3E); // warm amber, vinyl label
const DIM: Color32 = Color32::from_gray(150);

/// Numbers the audio thread publishes for the window.
#[derive(Default)]
pub struct UiStats {
    /// Click runs repaired since the active engine was (re)started.
    pub regions: AtomicU64,
    /// Samples repaired since then.
    pub samples_repaired: AtomicU64,
    /// Input samples processed since then (the meter's clock).
    pub samples: AtomicU64,
    /// Reported latency in samples.
    pub latency: AtomicU32,
    /// Sample rate (`f32` bits).
    pub sample_rate: AtomicU32,
}

impl UiStats {
    /// Publish from the audio thread (relaxed stores; never blocks).
    pub fn publish(
        &self,
        regions: u64,
        samples_repaired: u64,
        samples: u64,
        latency: usize,
        sample_rate: f32,
    ) {
        self.regions.store(regions, Ordering::Relaxed);
        self.samples_repaired
            .store(samples_repaired, Ordering::Relaxed);
        self.samples.store(samples, Ordering::Relaxed);
        self.latency.store(
            u32::try_from(latency).unwrap_or(u32::MAX),
            Ordering::Relaxed,
        );
        self.sample_rate
            .store(sample_rate.to_bits(), Ordering::Relaxed);
    }
}

// ---- Clicks-per-minute meter ----

/// Rate window in seconds of audio.
pub const METER_WINDOW_S: f64 = 10.0;
/// History length (one point per second of audio).
pub const METER_HISTORY: usize = 60;
/// Top of the log scale, clicks per minute.
pub const METER_MAX: f64 = 3000.0;

/// Clicks-per-minute from the published counters, in audio time (pausing or
/// faster-than-real-time rendering does not distort it). Counts click runs
/// in both channels. Lives on the GUI thread.
#[derive(Default)]
pub struct Meter {
    /// (samples, regions) snapshots covering the last [`METER_WINDOW_S`].
    snaps: VecDeque<(u64, u64)>,
    /// One rate per second of audio, newest last.
    history: VecDeque<f64>,
    next_history_at: u64,
    last_samples: u64,
}

impl Meter {
    /// Feed the latest counters; `sr` is the sample rate.
    pub fn update(&mut self, samples: u64, regions: u64, sr: f64) {
        if sr <= 0.0 {
            return;
        }
        if samples < self.last_samples {
            // The engine restarted (mode switch, reset, new sample rate).
            *self = Meter::default();
        }
        self.last_samples = samples;
        let quarter = (sr * 0.25) as u64;
        if self
            .snaps
            .back()
            .is_none_or(|&(s, _)| samples >= s + quarter.max(1))
        {
            self.snaps.push_back((samples, regions));
        }
        let window = (METER_WINDOW_S * sr) as u64;
        while self.snaps.len() > 2 && self.snaps[1].0 + window <= samples {
            self.snaps.pop_front();
        }
        if samples >= self.next_history_at {
            if let Some(r) = self.rate(sr) {
                self.history.push_back(r);
                if self.history.len() > METER_HISTORY {
                    self.history.pop_front();
                }
            }
            self.next_history_at = samples + sr as u64;
        }
    }

    /// Clicks per minute over the window, once at least 1 s has passed.
    pub fn rate(&self, sr: f64) -> Option<f64> {
        let (&(s0, r0), &(s1, r1)) = (self.snaps.front()?, self.snaps.back()?);
        let minutes = (s1.saturating_sub(s0)) as f64 / sr / 60.0;
        (minutes * 60.0 >= 1.0).then(|| r1.saturating_sub(r0) as f64 / minutes)
    }

    /// Rates of the last [`METER_HISTORY`] seconds, oldest first.
    pub fn history(&self) -> impl Iterator<Item = f64> + '_ {
        self.history.iter().copied()
    }
}

/// Position on the meter's log scale, 0..=1.
pub fn meter_fraction(rate: f64) -> f32 {
    ((1.0 + rate.max(0.0)).ln() / (1.0 + METER_MAX).ln()).clamp(0.0, 1.0) as f32
}

/// Zone colour and word for a rate.
pub fn meter_zone(rate: f64) -> (Color32, &'static str) {
    if rate < 100.0 {
        (Color32::from_rgb(0x5F, 0xB3, 0x6B), "clean")
    } else if rate < 600.0 {
        (ACCENT, "moderate")
    } else {
        (Color32::from_rgb(0xD9, 0x5B, 0x4A), "heavy")
    }
}

fn draw_meter(ui: &mut egui::Ui, meter: &Meter, sr: f64) {
    let rate = meter.rate(sr);
    ui.horizontal(|ui| {
        let (big, word, col) = match rate {
            Some(r) => {
                let (c, w) = meter_zone(r);
                (format!("{r:.0}"), w, c)
            }
            None => ("–".to_string(), "", DIM),
        };
        ui.label(RichText::new(big).size(26.0).strong().color(col));
        ui.label(RichText::new("clicks / min").color(DIM));
        ui.label(RichText::new(word).color(col));
    });
    // Bar on a log scale with ticks at 10, 100, 1000.
    let width = 460.0;
    let (bar, _) = ui.allocate_exact_size(Vec2::new(width, 16.0), Sense::hover());
    let painter = ui.painter_at(bar.expand(1.0));
    painter.rect_filled(bar, 3.0, Color32::from_gray(45));
    if let Some(r) = rate {
        let mut fill = bar;
        fill.set_width(bar.width() * meter_fraction(r));
        painter.rect_filled(fill, 3.0, meter_zone(r).0);
    }
    let (ticks, _) = ui.allocate_exact_size(Vec2::new(width, 12.0), Sense::hover());
    let tp = ui.painter_at(ticks);
    for t in [10.0, 100.0, 1000.0] {
        let x = bar.left() + bar.width() * meter_fraction(t);
        painter.line_segment(
            [Pos2::new(x, bar.top()), Pos2::new(x, bar.bottom())],
            Stroke::new(1.0_f32, Color32::from_gray(90)),
        );
        tp.text(
            Pos2::new(x, ticks.top()),
            Align2::CENTER_TOP,
            format!("{t:.0}"),
            FontId::proportional(10.0),
            DIM,
        );
    }
    // 60 s history line (same log scale).
    let (hist, _) = ui.allocate_exact_size(Vec2::new(width, 46.0), Sense::hover());
    let hp = ui.painter_at(hist);
    hp.rect_stroke(
        hist,
        3.0,
        Stroke::new(1.0_f32, Color32::from_gray(60)),
        egui::StrokeKind::Inside,
    );
    let pts: Vec<Pos2> = meter
        .history()
        .enumerate()
        .map(|(i, r)| {
            let x = hist.left() + hist.width() * (i as f32 / (METER_HISTORY - 1) as f32);
            let y = hist.bottom() - 3.0 - (hist.height() - 6.0) * meter_fraction(r);
            Pos2::new(x, y)
        })
        .collect();
    if pts.len() >= 2 {
        hp.add(egui::Shape::line(pts, Stroke::new(1.5_f32, ACCENT)));
    }
    hp.text(
        hist.left_top() + Vec2::new(4.0, 2.0),
        Align2::LEFT_TOP,
        "last 60 s",
        FontId::proportional(10.0),
        DIM,
    );
}

/// Default persisted editor state.
pub fn default_state() -> Arc<EguiState> {
    EguiState::from_size(WIDTH, HEIGHT)
}

/// Build the editor (called by the host when it opens the window).
pub fn create(params: Arc<PressingParams>, stats: Arc<UiStats>) -> Option<Box<dyn Editor>> {
    let state = params.editor_state.clone();
    create_egui_editor(
        state,
        Meter::default(),
        |ctx, _| {
            let mut style = (*ctx.style()).clone();
            style.spacing.item_spacing = Vec2::new(8.0, 8.0);
            style.spacing.button_padding = Vec2::new(10.0, 6.0);
            ctx.set_style(style);
        },
        move |ctx, setter, meter| {
            egui::CentralPanel::default().show(ctx, |ui| {
                draw(ui, setter, &params, &stats, meter);
            });
            // Keep the status line live (cheap: one small repaint per 250 ms).
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        },
    )
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title).strong().size(14.0));
}

fn draw(
    ui: &mut egui::Ui,
    setter: &ParamSetter,
    params: &PressingParams,
    stats: &UiStats,
    meter: &mut Meter,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("2ndPressing")
                .size(22.0)
                .strong()
                .color(ACCENT),
        );
        ui.label(RichText::new("Declicker").size(22.0));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let bypassed = params.bypass.value();
            let label = if bypassed { "Bypassed" } else { "Bypass" };
            let text = RichText::new(label).color(if bypassed { ACCENT } else { DIM });
            if ui.selectable_label(bypassed, text).clicked() {
                setter.begin_set_parameter(&params.bypass);
                setter.set_parameter(&params.bypass, !bypassed);
                setter.end_set_parameter(&params.bypass);
            }
        });
    });
    ui.separator();

    // Sensitivity: wide slider, value drawn inside it with plenty of room.
    section(ui, "Sensitivity");
    ui.horizontal(|ui| {
        ui.label(RichText::new("gentle").color(DIM));
        ui.add(widgets::ParamSlider::for_param(&params.sensitivity, setter).with_width(330.0));
        ui.label(RichText::new("aggressive").color(DIM));
    });

    // Percussion protection: one button per mode, latency underneath.
    section(ui, "Percussion protection");
    ui.horizontal(|ui| {
        let current = params.protection.value();
        for (mode, name, sub) in [
            (PercussionProtection::Off, "Off", "32 ms"),
            (PercussionProtection::Standard, "Standard", "≈ 1 s"),
            (PercussionProtection::Extended, "Extended", "≈ 2 s"),
        ] {
            let selected = current == mode;
            let text = RichText::new(format!("{name}\n{sub}")).size(13.0);
            let button = egui::Button::new(text)
                .selected(selected)
                .min_size(Vec2::new(150.0, 44.0));
            if ui.add(button).clicked() && !selected {
                setter.begin_set_parameter(&params.protection);
                setter.set_parameter(&params.protection, mode);
                setter.end_set_parameter(&params.protection);
            }
        }
    });
    ui.label(
        RichText::new("Leaves claps and drums on a beat alone. Off is for live monitoring.")
            .color(DIM)
            .size(11.0),
    );

    // Monitor.
    section(ui, "Listen to");
    ui.horizontal(|ui| {
        let current = params.monitor.value();
        for (mode, name) in [
            (Monitor::Declick, "Repaired audio"),
            (Monitor::ClicksOnly, "Clicks only"),
        ] {
            let selected = current == mode;
            let button = egui::Button::new(RichText::new(name).size(13.0))
                .selected(selected)
                .min_size(Vec2::new(150.0, 30.0));
            if ui.add(button).clicked() && !selected {
                setter.begin_set_parameter(&params.monitor);
                setter.set_parameter(&params.monitor, mode);
                setter.end_set_parameter(&params.monitor);
            }
        }
    });

    // Clicks-per-minute meter.
    let sr_meter = f64::from(f32::from_bits(stats.sample_rate.load(Ordering::Relaxed)));
    meter.update(
        stats.samples.load(Ordering::Relaxed),
        stats.regions.load(Ordering::Relaxed),
        sr_meter,
    );
    section(ui, "Clicks per minute");
    draw_meter(ui, meter, sr_meter);

    // Status line.
    ui.add_space(6.0);
    ui.separator();
    let regions = stats.regions.load(Ordering::Relaxed);
    let latency = stats.latency.load(Ordering::Relaxed);
    let sr = f32::from_bits(stats.sample_rate.load(Ordering::Relaxed));
    let status = if sr > 0.0 {
        format!(
            "Clicks repaired: {regions}    Latency: {:.0} ms    {:.1} kHz",
            f64::from(latency) * 1000.0 / f64::from(sr),
            sr / 1000.0
        )
    } else {
        "Waiting for audio…".to_string()
    };
    ui.label(RichText::new(status).color(DIM).size(12.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// Feed `seconds` of audio at `per_min` clicks per minute, 0.25 s steps.
    fn run(m: &mut Meter, start: (u64, u64), seconds: f64, per_min: f64) -> (u64, u64) {
        let (mut s, mut r) = start;
        let step = (SR * 0.25) as u64;
        let mut acc = 0.0;
        for _ in 0..(seconds * 4.0) as usize {
            s += step;
            acc += per_min / 60.0 * 0.25;
            while acc >= 1.0 {
                r += 1;
                acc -= 1.0;
            }
            m.update(s, r, SR);
        }
        (s, r)
    }

    #[test]
    fn rate_tracks_audio_time_over_ten_seconds() {
        let mut m = Meter::default();
        assert_eq!(m.rate(SR), None);
        let pos = run(&mut m, (0, 0), 20.0, 240.0);
        let r = m.rate(SR).unwrap();
        assert!((r - 240.0).abs() < 15.0, "{r}");
        // Rate change shows within the 10 s window.
        run(&mut m, pos, 12.0, 1200.0);
        let r = m.rate(SR).unwrap();
        assert!((r - 1200.0).abs() < 60.0, "{r}");
        assert!(m.history().count() >= 30 && m.history().count() <= METER_HISTORY);
    }

    #[test]
    fn engine_restart_clears_the_meter() {
        let mut m = Meter::default();
        run(&mut m, (0, 0), 15.0, 600.0);
        assert!(m.rate(SR).is_some());
        m.update(1_000, 0, SR); // counters went backwards: restarted engine
        assert_eq!(m.rate(SR), None);
        assert_eq!(m.history().count(), 0);
    }

    #[test]
    fn scale_and_zones() {
        assert_eq!(meter_fraction(0.0), 0.0);
        assert_eq!(meter_fraction(METER_MAX * 10.0), 1.0);
        assert!(meter_fraction(100.0) > meter_fraction(10.0));
        assert_eq!(meter_zone(50.0).1, "clean");
        assert_eq!(meter_zone(300.0).1, "moderate");
        assert_eq!(meter_zone(1500.0).1, "heavy");
    }
}
