//! Custom plugin window (nih_plug_egui).
//!
//! Layout (fixed size [`WIDTH`] x [`HEIGHT`] logical pixels):
//! - Sensitivity: a wide slider whose value is drawn in a roomy box.
//! - Percussion protection: three buttons (Off / Standard / Extended), each
//!   with its latency underneath.
//! - Monitor (Declick / Clicks only) and Bypass.
//! - Status: clicks repaired since the engine started, latency, sample rate.
//!
//! Live numbers come from [`UiStats`], written by the audio thread with
//! relaxed atomic stores (no locks, no allocation in `process`).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_egui::egui::{self, Align, Color32, Layout, RichText, Vec2};
use nih_plug_egui::{create_egui_editor, widgets, EguiState};

use crate::{Monitor, PercussionProtection, PressingParams};

/// Window size in logical pixels.
pub const WIDTH: u32 = 520;
pub const HEIGHT: u32 = 340;

const ACCENT: Color32 = Color32::from_rgb(0xE0, 0x9F, 0x3E); // warm amber, vinyl label
const DIM: Color32 = Color32::from_gray(150);

/// Numbers the audio thread publishes for the window.
#[derive(Default)]
pub struct UiStats {
    /// Click runs repaired since the active engine was (re)started.
    pub regions: AtomicU64,
    /// Samples repaired since then.
    pub samples_repaired: AtomicU64,
    /// Reported latency in samples.
    pub latency: AtomicU32,
    /// Sample rate (`f32` bits).
    pub sample_rate: AtomicU32,
}

impl UiStats {
    /// Publish from the audio thread (relaxed stores; never blocks).
    pub fn publish(&self, regions: u64, samples: u64, latency: usize, sample_rate: f32) {
        self.regions.store(regions, Ordering::Relaxed);
        self.samples_repaired.store(samples, Ordering::Relaxed);
        self.latency.store(
            u32::try_from(latency).unwrap_or(u32::MAX),
            Ordering::Relaxed,
        );
        self.sample_rate
            .store(sample_rate.to_bits(), Ordering::Relaxed);
    }
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
        (),
        |ctx, _| {
            let mut style = (*ctx.style()).clone();
            style.spacing.item_spacing = Vec2::new(8.0, 8.0);
            style.spacing.button_padding = Vec2::new(10.0, 6.0);
            ctx.set_style(style);
        },
        move |ctx, setter, _| {
            egui::CentralPanel::default().show(ctx, |ui| {
                draw(ui, setter, &params, &stats);
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

fn draw(ui: &mut egui::Ui, setter: &ParamSetter, params: &PressingParams, stats: &UiStats) {
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
