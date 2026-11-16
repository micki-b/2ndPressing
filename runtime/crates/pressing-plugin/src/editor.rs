//! Custom plugin window: the "Instrument" faceplate (design direction B).
//!
//! A warm grey panel in the manner of 1970s laboratory test gear, painted
//! directly with egui at its design size [`WIDTH`] x [`HEIGHT`]. The header
//! band carries the logo's cartridge icon (from `brand/export.py`):
//! - Sensitivity: amber readout (percent, plus the internal threshold in
//!   small print) above a knob with 21 detents (5 % steps). Drag vertically or
//!   scroll to change it; double-click returns to 50 %.
//! - Clicks per minute: amber readout and zone word, a 30-segment LED ladder
//!   on a log scale (10 / 100 / 1000), and a paper strip chart of the last
//!   60 s (same scale).
//! - Latching keys with lamps: Listen (Repaired / Clicks) and Bypass on the
//!   left under Sensitivity; Percussion protection (Off / Standard /
//!   Extended, each with its delay) on the right under the meter.
//! - Status band: latency, sample rate, clicks repaired.
//!
//! Live numbers come from [`UiStats`], written by the audio thread with
//! relaxed atomic stores (no locks, no allocation in `process`). Fonts: IBM
//! Plex Sans Condensed and IBM Plex Mono (SIL OFL 1.1, `assets/fonts`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_egui::egui::text::{LayoutJob, TextFormat};
use nih_plug_egui::egui::{
    self, Align2, Color32, ColorImage, CornerRadius, CursorIcon, FontData, FontDefinitions, FontFamily,
    FontId, Mesh, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2,
};
use nih_plug_egui::{create_egui_editor, EguiState};

use crate::{sensitivity_to_threshold, Monitor, PercussionProtection, PressingParams};

/// Where a double-click puts the Sensitivity knob.
pub const KNOB_RESET: f32 = 50.0;

/// Window size in logical pixels (the faceplate's design size).
pub const WIDTH: u32 = 520;
pub const HEIGHT: u32 = 470;

// ---- Palette (direction B) ----

pub const PANEL_TOP: Color32 = Color32::from_rgb(0xD4, 0xCF, 0xC3);
pub const PANEL_BOT: Color32 = Color32::from_rgb(0xC6, 0xC0, 0xB2);
pub const BAND: Color32 = Color32::from_rgb(0x3C, 0x3A, 0x36);
pub const BAND_TEXT: Color32 = Color32::from_rgb(0xEB, 0xE6, 0xDA);
pub const BAND_DIM: Color32 = Color32::from_rgb(0xA9, 0xA3, 0x95);
pub const LEGEND: Color32 = Color32::from_rgb(0x3C, 0x3A, 0x36);
pub const GLASS_TOP: Color32 = Color32::from_rgb(0x1B, 0x1A, 0x17);
pub const GLASS_BOT: Color32 = Color32::from_rgb(0x26, 0x24, 0x1F);
pub const AMBER: Color32 = Color32::from_rgb(0xF2, 0xA9, 0x3B);
pub const AMBER_DIM: Color32 = Color32::from_rgb(0xC9, 0x8A, 0x2E);
pub const AMBER_FAINT: Color32 = Color32::from_rgb(0x8C, 0x75, 0x47);
pub const KEY_TOP: Color32 = Color32::from_rgb(0xEC, 0xE8, 0xDE);
pub const KEY_BOT: Color32 = Color32::from_rgb(0xCF, 0xC9, 0xBB);
pub const KEY_ON_TOP: Color32 = Color32::from_rgb(0xD9, 0xD4, 0xC7);
pub const KEY_ON_BOT: Color32 = Color32::from_rgb(0xBD, 0xB6, 0xA6);
pub const KEY_EDGE: Color32 = Color32::from_rgb(0x8C, 0x85, 0x77);
pub const KEY_TEXT: Color32 = Color32::from_rgb(0x2C, 0x2A, 0x26);
pub const KEY_SUB: Color32 = Color32::from_rgb(0x6A, 0x64, 0x55);
pub const LAMP_OFF: Color32 = Color32::from_rgb(0x6D, 0x65, 0x53);
pub const RED: Color32 = Color32::from_rgb(0xE2, 0x55, 0x3F);
pub const GREEN: Color32 = Color32::from_rgb(0x7E, 0xC2, 0x7A);
pub const LED_OFF: Color32 = Color32::from_rgb(0x4A, 0x46, 0x3D);
pub const PAPER: Color32 = Color32::from_rgb(0xF4, 0xEC, 0xD4);
pub const PAPER_EDGE: Color32 = Color32::from_rgb(0xA8, 0x9F, 0x8A);
pub const GRID: Color32 = Color32::from_rgb(0xC9, 0xB9, 0x8F);
pub const INK: Color32 = Color32::from_rgb(0x2F, 0x3B, 0x6B);
pub const PAPER_LABEL: Color32 = Color32::from_rgb(0x8A, 0x7A, 0x52);
pub const FOOT: Color32 = Color32::from_rgb(0xBD, 0xB6, 0xA7);
pub const KNOB_EDGE: Color32 = Color32::from_rgb(0x34, 0x32, 0x2F);
pub const KNOB_MID: Color32 = Color32::from_rgb(0x4B, 0x49, 0x45);
pub const KNOB_HI: Color32 = Color32::from_rgb(0x7D, 0x7A, 0x73);
pub const KNOB_CAP: Color32 = Color32::from_rgb(0x55, 0x53, 0x4E);
pub const KNOB_LINE: Color32 = Color32::from_rgb(0xF0, 0xEB, 0xDF);

pub const COND: &str = "plex-cond";
pub const MONO: &str = "plex-mono";
pub const MONO_M: &str = "plex-mono-medium";
pub const SANS: &str = "plex-sans";
pub const SANS_B: &str = "plex-sans-semibold";

// ---- Shared numbers ----

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
        (GREEN, "clean")
    } else if rate < 600.0 {
        (AMBER, "moderate")
    } else {
        (RED, "heavy")
    }
}

// ---- Controls ----

/// What the faceplate reads and changes. The plugin implements it on top of
/// nih-plug's parameters (so hosts see gestures and automation); the
/// standalone app implements it on its own shared atomics.
pub trait Controls {
    /// Sensitivity in percent (0..=100).
    fn sensitivity(&self) -> f32;
    /// Start of a sensitivity gesture (knob drag).
    fn begin_sensitivity(&self) {}
    fn set_sensitivity(&self, percent: f32);
    /// End of a sensitivity gesture.
    fn end_sensitivity(&self) {}
    fn protection(&self) -> PercussionProtection;
    fn set_protection(&self, p: PercussionProtection);
    fn monitor(&self) -> Monitor;
    fn set_monitor(&self, m: Monitor);
    fn bypass(&self) -> bool;
    fn set_bypass(&self, on: bool);
}

/// [`Controls`] on the plugin's parameters.
struct HostControls<'a> {
    setter: &'a ParamSetter<'a>,
    params: &'a PressingParams,
}

impl Controls for HostControls<'_> {
    fn sensitivity(&self) -> f32 {
        self.params.sensitivity.value()
    }
    fn begin_sensitivity(&self) {
        self.setter.begin_set_parameter(&self.params.sensitivity);
    }
    fn set_sensitivity(&self, percent: f32) {
        self.setter.set_parameter(&self.params.sensitivity, percent);
    }
    fn end_sensitivity(&self) {
        self.setter.end_set_parameter(&self.params.sensitivity);
    }
    fn protection(&self) -> PercussionProtection {
        self.params.protection.value()
    }
    fn set_protection(&self, p: PercussionProtection) {
        self.setter.begin_set_parameter(&self.params.protection);
        self.setter.set_parameter(&self.params.protection, p);
        self.setter.end_set_parameter(&self.params.protection);
    }
    fn monitor(&self) -> Monitor {
        self.params.monitor.value()
    }
    fn set_monitor(&self, m: Monitor) {
        self.setter.begin_set_parameter(&self.params.monitor);
        self.setter.set_parameter(&self.params.monitor, m);
        self.setter.end_set_parameter(&self.params.monitor);
    }
    fn bypass(&self) -> bool {
        self.params.bypass.value()
    }
    fn set_bypass(&self, on: bool) {
        self.setter.begin_set_parameter(&self.params.bypass);
        self.setter.set_parameter(&self.params.bypass, on);
        self.setter.end_set_parameter(&self.params.bypass);
    }
}

// ---- Editor ----

/// The faceplate with its GUI-thread state, drawn at the top-left of the
/// `Ui` it is shown in ([`WIDTH`] x [`HEIGHT`]).
pub struct Panel {
    st: EditorState,
    model_threshold: f32,
}

impl Panel {
    /// `allowed` is shared with the audio side: the panel re-checks the
    /// license when it opens and sets it when a code is activated.
    pub fn new(model_threshold: f32, allowed: Arc<AtomicBool>) -> Self {
        let status = pressing_license::check();
        allowed.store(status.may_process(), Ordering::Relaxed);
        Panel {
            st: EditorState {
                license: LicenseUi {
                    status,
                    allowed,
                    msg: None,
                },
                ..EditorState::default()
            },
            model_threshold,
        }
    }

    /// Draw the faceplate (controls or info page) and handle its input.
    pub fn show(&mut self, ui: &mut egui::Ui, controls: &dyn Controls, stats: &UiStats) {
        Faceplate::new(ui).draw(ui, controls, stats, &mut self.st, self.model_threshold);
    }
}

/// The logo's cartridge (front view, no lettering) for the header band:
/// straight RGBA, `BAND_ICON_PX` square, exported by `brand/export.py`.
const BAND_ICON: &[u8] = include_bytes!("../assets/brand/cartridge-plain-102.rgba");
const BAND_ICON_PX: usize = 102;

/// GUI-thread state: the meter and the knob gesture.
#[derive(Default)]
struct EditorState {
    /// Header-band cartridge icon (uploaded on first draw).
    icon: Option<TextureHandle>,
    meter: Meter,
    /// Showing the info screen instead of the controls.
    info: bool,
    /// Showing the activation page instead of the controls.
    activation: bool,
    license: LicenseUi,
    /// Parsed `assets/info-screen.md` (filled on first use).
    doc: Option<crate::info::InfoDoc>,
    /// Sensitivity when the current drag started, and pixels dragged since.
    drag_start: f32,
    drag_px: f32,
    scroll_px: f32,
}

/// Activation state shown on the faceplate.
struct LicenseUi {
    status: pressing_license::Status,
    /// Shared with the audio side (false: audio passes through).
    allowed: Arc<AtomicBool>,
    /// Result of the last PASTE CODE: text, and whether it worked.
    msg: Option<(&'static str, bool)>,
}

impl Default for LicenseUi {
    fn default() -> Self {
        LicenseUi {
            status: pressing_license::Status::Trial {
                days_left: pressing_license::TRIAL_DAYS,
            },
            allowed: Arc::new(AtomicBool::new(true)),
            msg: None,
        }
    }
}

impl LicenseUi {
    /// PASTE CODE: read the clipboard, verify, save.
    fn paste(&mut self) {
        let text = arboard::Clipboard::new().and_then(|mut c| c.get_text());
        self.msg = Some(match text {
            Err(_) => ("Couldn't read the clipboard. Copy the code again.", false),
            Ok(text) => match pressing_license::activate(&text) {
                Ok(a) => {
                    self.status = pressing_license::Status::Registered(a);
                    self.allowed.store(true, Ordering::Relaxed);
                    ("Activated. Thank you for testing 2ndPressing!", true)
                }
                Err(e) => (e.message(), false),
            },
        });
    }
}

/// Open a web page in the default browser (best effort).
fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(windows)]
    let _ = std::process::Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

/// Default persisted editor state.
pub fn default_state() -> Arc<EguiState> {
    EguiState::from_size(WIDTH, HEIGHT)
}

/// The embedded IBM Plex fonts, with egui's own fonts as fallback.
pub fn fonts() -> FontDefinitions {
    let mut f = FontDefinitions::default();
    for (name, bytes) in [
        (
            COND,
            &include_bytes!("../assets/fonts/IBMPlexSansCondensed-SemiBold.ttf")[..],
        ),
        (
            MONO,
            &include_bytes!("../assets/fonts/IBMPlexMono-Regular.ttf")[..],
        ),
        (
            MONO_M,
            &include_bytes!("../assets/fonts/IBMPlexMono-Medium.ttf")[..],
        ),
        (
            SANS,
            &include_bytes!("../assets/fonts/IBMPlexSans-Regular.ttf")[..],
        ),
        (
            SANS_B,
            &include_bytes!("../assets/fonts/IBMPlexSans-SemiBold.ttf")[..],
        ),
    ] {
        f.font_data
            .insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
        // egui's bundled fonts as fallback for any glyph Plex lacks.
        let mut chain = vec![name.to_owned()];
        chain.extend(
            f.families
                .get(&FontFamily::Proportional)
                .cloned()
                .unwrap_or_default(),
        );
        f.families.insert(FontFamily::Name(name.into()), chain);
    }
    f
}

/// Build the editor (called by the host when it opens the window).
pub fn create(
    params: Arc<PressingParams>,
    stats: Arc<UiStats>,
    model_threshold: f32,
    allowed: Arc<AtomicBool>,
) -> Option<Box<dyn Editor>> {
    let state = params.editor_state.clone();
    create_egui_editor(
        state,
        Panel::new(model_threshold, allowed),
        |ctx, _| ctx.set_fonts(fonts()),
        move |ctx, setter, panel| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let controls = HostControls {
                        setter,
                        params: &params,
                    };
                    panel.show(ui, &controls, &stats);
                });
            // Keep readouts live (cheap: a small repaint every 250 ms).
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        },
    )
}

/// A font from the embedded set ([`COND`], [`MONO`], [`MONO_M`], [`SANS`], [`SANS_B`]).
pub fn font(family: &str, size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(family.into()))
}

/// The wordmark "2ndPressing" with "nd" as a moderate superscript (60 % size,
/// top-aligned). `left_bottom` is the bottom-left of the text; returns its width.
pub fn brand(p: &egui::Painter, left_bottom: Pos2, size: f32, color: Color32) -> f32 {
    let full = TextFormat {
        font_id: font(COND, size),
        color,
        valign: egui::Align::BOTTOM,
        ..Default::default()
    };
    let sup = TextFormat {
        font_id: font(COND, size * 0.6),
        color,
        valign: egui::Align::TOP,
        ..Default::default()
    };
    let mut job = LayoutJob::default();
    job.append("2", 0.0, full.clone());
    job.append("nd", 0.0, sup);
    job.append("Pressing", 0.0, full);
    let galley = p.layout_job(job);
    let size = galley.size();
    p.galley(left_bottom - Vec2::new(0.0, size.y), galley, color);
    size.x
}

/// Draw the logo's cartridge icon (front view) into `rect`.
fn band_icon(ui: &egui::Ui, pt: &egui::Painter, st: &mut EditorState, rect: Rect) {
    let tex = st.icon.get_or_insert_with(|| {
        ui.ctx().load_texture(
            "band-icon",
            ColorImage::from_rgba_unmultiplied([BAND_ICON_PX, BAND_ICON_PX], BAND_ICON),
            TextureOptions::LINEAR,
        )
    });
    pt.image(
        tex.id(),
        rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}

/// Vertical gradient rectangle.
pub fn vgrad(p: &egui::Painter, r: Rect, top: Color32, bot: Color32) {
    let mut m = Mesh::default();
    m.colored_vertex(r.left_top(), top);
    m.colored_vertex(r.right_top(), top);
    m.colored_vertex(r.right_bottom(), bot);
    m.colored_vertex(r.left_bottom(), bot);
    m.add_triangle(0, 1, 2);
    m.add_triangle(0, 2, 3);
    p.add(Shape::mesh(m));
}

fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Maps the 520 x 470 design coordinates onto the window.
struct Faceplate {
    o: Pos2,
}

impl Faceplate {
    fn new(ui: &egui::Ui) -> Self {
        Faceplate {
            o: ui.max_rect().min,
        }
    }
    fn p(&self, x: f32, y: f32) -> Pos2 {
        self.o + Vec2::new(x, y)
    }
    fn r(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::from_min_size(self.p(x, y), Vec2::new(w, h))
    }

    fn draw(
        &self,
        ui: &mut egui::Ui,
        c: &dyn Controls,
        stats: &UiStats,
        st: &mut EditorState,
        model_threshold: f32,
    ) {
        let painter = ui.painter().clone();
        let pt = &painter;

        // Panel and bands.
        vgrad(
            pt,
            self.r(0.0, 0.0, WIDTH as f32, HEIGHT as f32),
            PANEL_TOP,
            PANEL_BOT,
        );
        let sr = f64::from(f32::from_bits(stats.sample_rate.load(Ordering::Relaxed)));
        st.meter.update(
            stats.samples.load(Ordering::Relaxed),
            stats.regions.load(Ordering::Relaxed),
            sr,
        );
        if st.info {
            self.info_page(ui, pt, st);
            return;
        }
        if st.activation {
            self.activation_page(ui, pt, st);
            return;
        }
        pt.rect_filled(self.r(0.0, 0.0, WIDTH as f32, 54.0), 0.0, BAND);
        band_icon(ui, pt, st, self.r(18.0, 9.0, 36.0, 36.0));
        let brand_w = brand(pt, self.p(60.0, 38.0), 18.0, BAND_TEXT);
        pt.text(
            self.p(60.0 + brand_w + 8.0, 38.0),
            Align2::LEFT_BOTTOM,
            "DECLICKER",
            font(COND, 18.0),
            BAND_DIM,
        );
        pt.text(
            self.p(432.0, 36.0),
            Align2::RIGHT_BOTTOM,
            "CLICK REPAIR · STEREO",
            font(MONO, 9.5),
            BAND_DIM,
        );
        if self.small_key(ui, pt, "info-key", self.r(444.0, 14.0, 52.0, 26.0), "INFO") {
            st.info = true;
        }

        self.sensitivity(ui, pt, c, st, model_threshold);
        self.meter(pt, &st.meter, sr);
        self.keys(ui, pt, c);
        self.status(pt, stats, sr, st.license.status);
        self.license_tag(ui, pt, st);
    }

    // ---- Small band key (INFO / CONTROLS) ----

    fn small_key(
        &self,
        ui: &mut egui::Ui,
        pt: &egui::Painter,
        id: &str,
        rect: Rect,
        label: &str,
    ) -> bool {
        let resp = ui
            .interact(rect, ui.id().with(id), Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand);
        let down = resp.is_pointer_button_down_on();
        let cap = rect.translate(Vec2::new(0.0, if down { 1.0 } else { 0.0 }));
        pt.rect_filled(
            rect.translate(Vec2::new(0.0, 2.0)),
            3.0,
            Color32::from_black_alpha(90),
        );
        pt.rect_filled(cap, 3.0, KEY_BOT);
        vgrad(
            pt,
            cap.shrink(1.5),
            if down { KEY_ON_TOP } else { KEY_TOP },
            KEY_BOT,
        );
        pt.rect_stroke(
            cap,
            CornerRadius::same(3),
            Stroke::new(1.0_f32, KEY_EDGE),
            StrokeKind::Inside,
        );
        pt.text(
            cap.center(),
            Align2::CENTER_CENTER,
            label,
            font(COND, 10.0),
            KEY_TEXT,
        );
        resp.clicked()
    }

    // ---- Info screen (layout A) ----

    fn info_page(&self, ui: &mut egui::Ui, pt: &egui::Painter, st: &mut EditorState) {
        let doc = st
            .doc
            .get_or_insert_with(|| crate::info::parse(crate::info::INFO_MD))
            .clone();
        // Band: wordmark, section label, back key.
        pt.rect_filled(self.r(0.0, 0.0, WIDTH as f32, 46.0), 0.0, BAND);
        band_icon(ui, pt, st, self.r(16.0, 7.0, 32.0, 32.0));
        let bw = brand(pt, self.p(54.0, 33.0), 17.0, BAND_TEXT);
        let dw = pt
            .text(
                self.p(54.0 + bw + 8.0, 33.0),
                Align2::LEFT_BOTTOM,
                "DECLICKER",
                font(COND, 17.0),
                BAND_DIM,
            )
            .width();
        pt.text(
            self.p(54.0 + bw + 8.0 + dw + 12.0, 31.0),
            Align2::LEFT_BOTTOM,
            &doc.section_label,
            font(MONO_M, 10.0),
            AMBER,
        );
        let back = if doc.back_key.is_empty() {
            "CONTROLS"
        } else {
            doc.back_key.as_str()
        };
        if self.small_key(ui, pt, "back-key", self.r(416.0, 10.0, 88.0, 26.0), back) {
            st.info = false;
        }

        // Readouts.
        let n = doc.readouts.len().max(1) as f32;
        let gap = 8.0;
        let rw = (488.0 - gap * (n - 1.0)) / n;
        for (i, r) in doc.readouts.iter().enumerate() {
            let rect = self.r(16.0 + i as f32 * (rw + gap), 56.0, rw, 64.0);
            vgrad(pt, rect, GLASS_TOP, GLASS_BOT);
            let (num, unit) = crate::info::split_unit(&r.number);
            let nr = pt.text(
                rect.left_top() + Vec2::new(10.0, 6.0),
                Align2::LEFT_TOP,
                num,
                font(MONO_M, 25.0),
                AMBER,
            );
            pt.text(
                Pos2::new(nr.right() + 3.0, nr.bottom() - 4.0),
                Align2::LEFT_BOTTOM,
                unit,
                font(MONO, 11.0),
                AMBER_DIM,
            );
            let mut job = LayoutJob::simple(
                r.caption.to_uppercase(),
                font(MONO, 8.0),
                AMBER_FAINT,
                rw - 20.0,
            );
            job.wrap.max_rows = 2;
            let g = pt.layout_job(job);
            pt.galley(
                Pos2::new(rect.left() + 10.0, rect.bottom() - 6.0 - g.size().y),
                g,
                AMBER_FAINT,
            );
        }

        // Data sheet (scrolls).
        let paper = self.r(16.0, 128.0, 488.0, 304.0);
        pt.rect_filled(paper, 2.0, PAPER);
        pt.rect_stroke(
            paper,
            2.0,
            Stroke::new(1.0_f32, PAPER_EDGE),
            StrokeKind::Inside,
        );
        let style = crate::info::Style {
            text: Color32::from_rgb(0x2A, 0x26, 0x20),
            sub: KEY_SUB,
            heading: LEGEND,
            rule: Color32::from_rgba_unmultiplied(0x3C, 0x3A, 0x36, 46),
            ink: INK,
            grey_bar: Color32::from_rgb(0xA3, 0x9A, 0x86),
            track: Color32::from_rgba_unmultiplied(0x3C, 0x3A, 0x36, 30),
            lamp_green: GREEN,
            lamp_amber: AMBER,
            lamp_off: LAMP_OFF,
            sans: SANS,
            sans_bold: SANS_B,
            cond: COND,
            mono: MONO,
        };
        let inner = paper.shrink2(Vec2::new(12.0, 8.0));
        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(inner), |ui| {
            egui::ScrollArea::vertical()
                .id_salt("info-sheet")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.set_width(inner.width() - 12.0);
                    crate::info::sheet(ui, &doc, &style);
                });
        });

        // Status band.
        pt.rect_filled(self.r(0.0, 440.0, WIDTH as f32, 30.0), 0.0, FOOT);
        pt.text(
            self.p(16.0, 455.0),
            Align2::LEFT_CENTER,
            &doc.status_left,
            font(MONO, 9.0),
            LEGEND,
        );
        pt.text(
            self.p(504.0, 455.0),
            Align2::RIGHT_CENTER,
            &doc.status_right,
            font(MONO, 9.0),
            LEGEND,
        );
    }

    // ---- Sensitivity ----

    fn sensitivity(
        &self,
        ui: &mut egui::Ui,
        pt: &egui::Painter,
        c: &dyn Controls,
        st: &mut EditorState,
        model_threshold: f32,
    ) {
        let value = c.sensitivity();
        pt.text(
            self.p(26.0, 72.0),
            Align2::LEFT_TOP,
            "SENSITIVITY",
            font(COND, 10.0),
            LEGEND,
        );
        let glass = self.r(26.0, 90.0, 200.0, 56.0);
        vgrad(pt, glass, GLASS_TOP, GLASS_BOT);
        pt.text(
            self.p(200.0, 140.0),
            Align2::RIGHT_BOTTOM,
            format!("{value:.0}"),
            font(MONO_M, 36.0),
            AMBER,
        );
        pt.text(
            self.p(205.0, 135.0),
            Align2::LEFT_BOTTOM,
            "%",
            font(MONO, 13.0),
            AMBER_DIM,
        );
        let thr = sensitivity_to_threshold(value, model_threshold);
        pt.text(
            self.p(34.0, 96.0),
            Align2::LEFT_TOP,
            format!("THRESHOLD {thr:.2}"),
            font(MONO, 8.5),
            AMBER_FAINT,
        );

        // Scale: 21 detents over 270 degrees, numbers every 25 %.
        let ctr = self.p(126.0, 224.0);
        let dir = |deg: f32| {
            let a = (deg - 90.0).to_radians();
            Vec2::new(a.cos(), a.sin())
        };
        for i in 0..=20 {
            let deg = -135.0 + i as f32 * 13.5;
            let major = i % 5 == 0;
            let (r1, r2) = (46.0, if major { 55.0 } else { 51.0 });
            pt.line_segment(
                [ctr + dir(deg) * r1, ctr + dir(deg) * r2],
                Stroke::new(if major { 1.3_f32 } else { 0.8 }, LEGEND),
            );
            if major {
                pt.text(
                    ctr + dir(deg) * 66.0,
                    Align2::CENTER_CENTER,
                    format!("{}", i * 5),
                    font(MONO, 8.5),
                    LEGEND,
                );
            }
        }
        // Knob: shadow, shaded body, cap, pointer.
        pt.circle_filled(ctr + Vec2::new(2.0, 3.0), 40.0, Color32::from_black_alpha(70));
        pt.circle_filled(ctr, 40.0, KNOB_EDGE);
        for k in 0..10 {
            let t = k as f32 / 9.0;
            let col = lerp(KNOB_MID, KNOB_HI, t * 0.8);
            pt.circle_filled(ctr + Vec2::new(-5.0, -7.0) * t, 38.0 * (1.0 - t * 0.55), col);
        }
        pt.circle_filled(ctr, 30.0, KNOB_CAP);
        pt.circle_stroke(ctr, 30.0, Stroke::new(1.0_f32, KNOB_EDGE));
        let deg = -135.0 + 270.0 * (value / 100.0);
        pt.line_segment(
            [ctr + dir(deg) * 13.0, ctr + dir(deg) * 27.0],
            Stroke::new(2.6_f32, KNOB_LINE),
        );

        // Interaction: vertical drag (0.4 %/px), scroll, double-click reset.
        let resp = ui
            .interact(
                Rect::from_center_size(ctr, Vec2::splat(84.0)),
                ui.id().with("sens"),
                Sense::click_and_drag(),
            )
            .on_hover_cursor(CursorIcon::ResizeVertical);
        let snap = |v: f32| ((v / 5.0).round() * 5.0).clamp(0.0, 100.0);
        if resp.drag_started() {
            st.drag_start = value;
            st.drag_px = 0.0;
            c.begin_sensitivity();
        }
        if resp.dragged() {
            st.drag_px += resp.drag_delta().y;
            let v = snap(st.drag_start - st.drag_px * 0.4);
            if v != value {
                c.set_sensitivity(v);
            }
        }
        if resp.drag_stopped() {
            c.end_sensitivity();
        }
        if resp.double_clicked() {
            // Double-click returns to 50 % (the model's tuned threshold).
            c.begin_sensitivity();
            c.set_sensitivity(KNOB_RESET);
            c.end_sensitivity();
        }
        if resp.hovered() {
            st.scroll_px += ui.input(|i| i.smooth_scroll_delta.y);
            if st.scroll_px.abs() >= 24.0 {
                let v = snap(value + 5.0 * st.scroll_px.signum());
                st.scroll_px = 0.0;
                if v != value {
                    c.begin_sensitivity();
                    c.set_sensitivity(v);
                    c.end_sensitivity();
                }
            }
        }
    }

    // ---- Clicks per minute ----

    fn meter(&self, pt: &egui::Painter, meter: &Meter, sr: f64) {
        pt.text(
            self.p(262.0, 72.0),
            Align2::LEFT_TOP,
            "CLICKS PER MINUTE",
            font(COND, 10.0),
            LEGEND,
        );
        let glass = self.r(262.0, 90.0, 232.0, 56.0);
        vgrad(pt, glass, GLASS_TOP, GLASS_BOT);
        let rate = meter.rate(sr);
        let (num, word) = match rate {
            Some(r) => (format!("{r:.0}"), meter_zone(r).1.to_uppercase()),
            None => ("--".to_string(), "WAITING FOR AUDIO".to_string()),
        };
        pt.text(
            self.p(418.0, 140.0),
            Align2::RIGHT_BOTTOM,
            num,
            font(MONO_M, 36.0),
            AMBER,
        );
        pt.text(
            self.p(423.0, 135.0),
            Align2::LEFT_BOTTOM,
            "/min",
            font(MONO, 12.0),
            AMBER_DIM,
        );
        pt.text(
            self.p(270.0, 96.0),
            Align2::LEFT_TOP,
            word,
            font(MONO, 8.5),
            AMBER_FAINT,
        );

        // LED ladder: 30 segments on the log scale, zone colour per segment.
        let n = 30;
        let (x0, w, gap) = (262.0, 232.0, 2.0);
        let sw = (w - gap * (n - 1) as f32) / n as f32;
        let lit = rate.map_or(0, |r| (meter_fraction(r) * n as f32).round() as usize);
        for i in 0..n {
            let v = ((i as f64 + 0.5) / n as f64 * (1.0 + METER_MAX).ln()).exp() - 1.0;
            let col = if i < lit { meter_zone(v).0 } else { LED_OFF };
            pt.rect_filled(
                self.r(x0 + i as f32 * (sw + gap), 156.0, sw, 14.0),
                1.0,
                col,
            );
        }
        for v in [10.0, 100.0, 1000.0] {
            let x = x0 + w * meter_fraction(v);
            pt.line_segment(
                [self.p(x, 172.0), self.p(x, 177.0)],
                Stroke::new(1.0_f32, LEGEND),
            );
            pt.text(
                self.p(x, 179.0),
                Align2::CENTER_TOP,
                format!("{v:.0}"),
                font(MONO, 8.5),
                LEGEND,
            );
        }

        // Strip chart: paper, grid, ink trace of the last 60 s (log scale).
        let paper = self.r(262.0, 196.0, 232.0, 70.0);
        pt.rect_filled(paper, 2.0, PAPER);
        let mut x = paper.left() + 10.0;
        while x < paper.right() {
            pt.line_segment(
                [Pos2::new(x, paper.top()), Pos2::new(x, paper.bottom())],
                Stroke::new(0.5_f32, GRID),
            );
            x += 10.0;
        }
        let mut y = paper.top() + 8.0;
        while y < paper.bottom() {
            pt.line_segment(
                [Pos2::new(paper.left(), y), Pos2::new(paper.right(), y)],
                Stroke::new(0.5_f32, GRID),
            );
            y += 8.0;
        }
        pt.rect_stroke(
            paper,
            2.0,
            Stroke::new(1.0_f32, PAPER_EDGE),
            StrokeKind::Inside,
        );
        pt.text(
            self.p(268.0, 199.0),
            Align2::LEFT_TOP,
            "60 s chart",
            font(MONO, 8.0),
            PAPER_LABEL,
        );
        let pts: Vec<Pos2> = meter
            .history()
            .enumerate()
            .map(|(i, r)| {
                let px = paper.left()
                    + 4.0
                    + (paper.width() - 8.0) * (i as f32 / (METER_HISTORY - 1) as f32);
                let py = paper.bottom() - 4.0 - (paper.height() - 16.0) * meter_fraction(r);
                Pos2::new(px, py)
            })
            .collect();
        if pts.len() >= 2 {
            pt.add(Shape::line(pts, Stroke::new(1.2_f32, INK)));
        }
    }

    // ---- Keys ----

    #[allow(clippy::too_many_arguments)]
    fn key(
        &self,
        ui: &mut egui::Ui,
        pt: &egui::Painter,
        id: &str,
        rect: Rect,
        label: &str,
        sub: &str,
        on: bool,
        lamp_on: Color32,
    ) -> bool {
        let resp = ui
            .interact(rect, ui.id().with(id), Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand);
        let pressed = on || resp.is_pointer_button_down_on();
        pt.rect_filled(rect.translate(Vec2::new(0.0, 3.0)), 3.0, KEY_EDGE);
        let cap = rect.translate(Vec2::new(0.0, if pressed { 2.0 } else { 0.0 }));
        let (top, bot) = if pressed {
            (KEY_ON_TOP, KEY_ON_BOT)
        } else {
            (KEY_TOP, KEY_BOT)
        };
        pt.rect_filled(cap, 3.0, bot);
        vgrad(pt, cap.shrink(1.5), top, bot);
        pt.rect_stroke(
            cap,
            CornerRadius::same(3),
            Stroke::new(1.0_f32, KEY_EDGE),
            StrokeKind::Inside,
        );
        let lamp =
            Rect::from_center_size(cap.center_top() + Vec2::new(0.0, 9.0), Vec2::new(18.0, 4.0));
        pt.rect_filled(lamp, 1.0, if on { lamp_on } else { LAMP_OFF });
        let ty = if sub.is_empty() {
            cap.center().y + 4.0
        } else {
            cap.center().y + 2.0
        };
        pt.text(
            Pos2::new(cap.center().x, ty),
            Align2::CENTER_CENTER,
            label,
            font(COND, 11.0),
            KEY_TEXT,
        );
        if !sub.is_empty() {
            pt.text(
                Pos2::new(cap.center().x, cap.bottom() - 9.0),
                Align2::CENTER_CENTER,
                sub,
                font(MONO, 8.5),
                KEY_SUB,
            );
        }
        resp.clicked()
    }

    fn keys(&self, ui: &mut egui::Ui, pt: &egui::Painter, c: &dyn Controls) {
        // Keys sit on the two columns above them: Listen and Bypass span the
        // Sensitivity display (x 26..226, 200 px); Percussion protection
        // spans the meter (x 262..494, three keys filling its 232 px).
        const GAP: f32 = 5.0;
        const KEY_W: f32 = 74.0;
        const KEY_STEP: f32 = KEY_W + GAP;
        const LEFT: f32 = 26.0;
        const LEFT_W: f32 = 200.0;
        const LISTEN_W: f32 = (LEFT_W - GAP) / 2.0;
        const RIGHT: f32 = 262.0;

        pt.text(
            self.p(LEFT, 302.0),
            Align2::LEFT_TOP,
            "LISTEN",
            font(COND, 10.0),
            LEGEND,
        );
        let mon = c.monitor();
        for (i, (mode, label, sub)) in [
            (Monitor::Declick, "REPAIRED", "audio"),
            (Monitor::ClicksOnly, "CLICKS", "only"),
        ]
        .into_iter()
        .enumerate()
        {
            let r = self.r(LEFT + i as f32 * (LISTEN_W + GAP), 320.0, LISTEN_W, 46.0);
            if self.key(ui, pt, label, r, label, sub, mon == mode, AMBER) && mon != mode {
                c.set_monitor(mode);
            }
        }

        let bypassed = c.bypass();
        let r = self.r(LEFT, 376.0, LEFT_W, 40.0);
        if self.key(ui, pt, "BYPASS", r, "BYPASS", "", bypassed, RED) {
            c.set_bypass(!bypassed);
        }

        pt.text(
            self.p(RIGHT, 302.0),
            Align2::LEFT_TOP,
            "PERCUSSION PROTECTION",
            font(COND, 10.0),
            LEGEND,
        );
        let prot = c.protection();
        for (i, (mode, label, sub)) in [
            (PercussionProtection::Off, "OFF", "32 ms"),
            (PercussionProtection::Standard, "STANDARD", "≈ 1 s"),
            (PercussionProtection::Extended, "EXTENDED", "≈ 2 s"),
        ]
        .into_iter()
        .enumerate()
        {
            let r = self.r(RIGHT + i as f32 * KEY_STEP, 320.0, KEY_W, 46.0);
            if self.key(ui, pt, label, r, label, sub, prot == mode, AMBER) && prot != mode {
                c.set_protection(mode);
            }
        }
    }

    // ---- License tag (status band, right) ----

    /// Trial days left / trial ended / activated; opens the activation page.
    fn license_tag(&self, ui: &mut egui::Ui, pt: &egui::Painter, st: &mut EditorState) {
        use pressing_license::Status;
        let (text, color) = match st.license.status {
            Status::Registered(_) => ("ACTIVATED".to_string(), KEY_SUB),
            Status::Trial { days_left } => (
                format!(
                    "FREE TRIAL · {days_left} DAY{} LEFT  ›",
                    if days_left == 1 { "" } else { "S" }
                ),
                LEGEND,
            ),
            Status::Expired => ("TRIAL ENDED · ACTIVATE  ›".to_string(), RED),
        };
        let r = pt.text(
            self.p(494.0, 449.0),
            Align2::RIGHT_CENTER,
            text,
            font(MONO_M, 9.5),
            color,
        );
        let resp = ui
            .interact(r.expand(6.0), ui.id().with("license-tag"), Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand);
        if resp.hovered() {
            pt.line_segment(
                [r.left_bottom() + Vec2::new(0.0, 2.0), r.right_bottom() + Vec2::new(0.0, 2.0)],
                Stroke::new(1.0_f32, color),
            );
        }
        if resp.clicked() {
            st.license.msg = None;
            st.activation = true;
        }
    }

    // ---- Activation page ----

    fn activation_page(&self, ui: &mut egui::Ui, pt: &egui::Painter, st: &mut EditorState) {
        use pressing_license::Status;
        // Band (as on the info page).
        pt.rect_filled(self.r(0.0, 0.0, WIDTH as f32, 46.0), 0.0, BAND);
        band_icon(ui, pt, st, self.r(16.0, 7.0, 32.0, 32.0));
        let bw = brand(pt, self.p(54.0, 33.0), 17.0, BAND_TEXT);
        let dw = pt
            .text(
                self.p(54.0 + bw + 8.0, 33.0),
                Align2::LEFT_BOTTOM,
                "DECLICKER",
                font(COND, 17.0),
                BAND_DIM,
            )
            .width();
        pt.text(
            self.p(54.0 + bw + 8.0 + dw + 12.0, 31.0),
            Align2::LEFT_BOTTOM,
            "ACTIVATION",
            font(MONO_M, 10.0),
            AMBER,
        );
        if self.small_key(ui, pt, "act-back", self.r(416.0, 10.0, 88.0, 26.0), "CONTROLS") {
            st.activation = false;
        }

        // Readout.
        let glass = self.r(16.0, 58.0, 488.0, 78.0);
        vgrad(pt, glass, GLASS_TOP, GLASS_BOT);
        let (num, unit, caption) = match st.license.status {
            Status::Registered(a) => (
                "ACTIVE".to_string(),
                String::new(),
                format!("REGISTERED BETA USER #{} · CODE ISSUED {}", a.id, a.issued()),
            ),
            Status::Trial { days_left } => (
                days_left.to_string(),
                format!("DAY{} LEFT", if days_left == 1 { "" } else { "S" }),
                "FREE TRIAL · AFTER IT, A FREE ACTIVATION CODE IS NEEDED".to_string(),
            ),
            Status::Expired => (
                "ENDED".to_string(),
                String::new(),
                "FREE TRIAL ENDED · AUDIO PASSES THROUGH UNPROCESSED".to_string(),
            ),
        };
        let nr = pt.text(
            glass.left_top() + Vec2::new(14.0, 8.0),
            Align2::LEFT_TOP,
            num,
            font(MONO_M, 30.0),
            if st.license.status.may_process() { AMBER } else { RED },
        );
        pt.text(
            Pos2::new(nr.right() + 6.0, nr.bottom() - 5.0),
            Align2::LEFT_BOTTOM,
            unit,
            font(MONO, 12.0),
            AMBER_DIM,
        );
        pt.text(
            Pos2::new(glass.left() + 14.0, glass.bottom() - 10.0),
            Align2::LEFT_BOTTOM,
            caption,
            font(MONO, 8.5),
            AMBER_FAINT,
        );

        // Instructions on paper.
        let paper = self.r(16.0, 146.0, 488.0, 206.0);
        pt.rect_filled(paper, 2.0, PAPER);
        pt.rect_stroke(paper, 2.0, Stroke::new(1.0_f32, PAPER_EDGE), StrokeKind::Inside);
        let ink = Color32::from_rgb(0x2A, 0x26, 0x20);
        let x = paper.left() + 16.0;
        let w = paper.width() - 32.0;
        let mut y = paper.top() + 14.0;
        pt.text(Pos2::new(x, y), Align2::LEFT_TOP, "FREE ACTIVATION CODE", font(COND, 11.0), LEGEND);
        y += 22.0;
        let paras: &[&str] = if matches!(st.license.status, Status::Registered(_)) {
            &[
                "This computer is activated. The app and every plugin format on it share this activation.",
                "Lost your code? Register the same email again and the website shows it once more.",
            ]
        } else {
            &[
                "2ndPressing is free. Register your email on the website and you get an activation code right away: no payment, no account.",
                "Copy the whole code (it starts with 2NDP), then press PASTE CODE. One code activates the app and every plugin format on this computer.",
            ]
        };
        for p in paras {
            let job = LayoutJob::simple(p.to_string(), font(SANS, 12.5), ink, w);
            let g = pt.layout_job(job);
            let h = g.size().y;
            pt.galley(Pos2::new(x, y), g, ink);
            y += h + 10.0;
        }
        if let Some((msg, ok)) = st.license.msg {
            let color = if ok {
                Color32::from_rgb(0x2E, 0x6B, 0x2B)
            } else {
                Color32::from_rgb(0xB0, 0x3A, 0x26)
            };
            let job = LayoutJob::simple(msg.to_string(), font(SANS_B, 12.5), color, w);
            let g = pt.layout_job(job);
            pt.galley(Pos2::new(x, paper.bottom() - 14.0 - g.size().y), g, color);
        }

        // Keys.
        let registered = matches!(st.license.status, Status::Registered(_));
        if self.key(ui, pt, "act-paste", self.r(16.0, 366.0, 238.0, 46.0), "PASTE CODE", "from the clipboard", registered, GREEN) {
            st.license.paste();
        }
        if self.key(ui, pt, "act-web", self.r(266.0, 366.0, 238.0, 46.0), "GET A FREE CODE", "secondpressing.com", false, AMBER) {
            open_url(pressing_license::REGISTER_URL);
        }

        // Status band.
        pt.rect_filled(self.r(0.0, 440.0, WIDTH as f32, 30.0), 0.0, FOOT);
        pt.text(
            self.p(16.0, 455.0),
            Align2::LEFT_CENTER,
            "OFFLINE CHECK · 2NDPRESSING NEVER CONTACTS THE INTERNET",
            font(MONO, 9.0),
            LEGEND,
        );
    }

    // ---- Status band ----

    fn status(
        &self,
        pt: &egui::Painter,
        stats: &UiStats,
        sr: f64,
        license: pressing_license::Status,
    ) {
        pt.rect_filled(self.r(0.0, 428.0, WIDTH as f32, 42.0), 0.0, FOOT);
        let text = if !license.may_process() {
            "AUDIO PASSES THROUGH UNPROCESSED".to_string()
        } else if sr > 0.0 {
            let latency = f64::from(stats.latency.load(Ordering::Relaxed)) / sr;
            let lat = if latency < 1.0 {
                format!("{:.0} ms", latency * 1000.0)
            } else {
                format!("{latency:.2} s")
            };
            format!(
                "LATENCY {lat}    {:.1} kHz    {} repaired",
                sr / 1000.0,
                stats.regions.load(Ordering::Relaxed)
            )
        } else {
            "WAITING FOR AUDIO".to_string()
        };
        pt.text(
            self.p(26.0, 449.0),
            Align2::LEFT_CENTER,
            text,
            font(MONO, 9.5),
            LEGEND,
        );
    }
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
        m.update(1_000, 0, SR);
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

    #[test]
    fn double_click_resets_to_fifty_percent_the_default() {
        assert_eq!(KNOB_RESET, 50.0);
        assert_eq!(KNOB_RESET, crate::SENSITIVITY_DEFAULT);
        let p = PressingParams::new(2.75);
        assert_eq!(p.sensitivity.default_plain_value(), KNOB_RESET);
        assert!((sensitivity_to_threshold(KNOB_RESET, 2.75) - 2.75).abs() < 1e-6);
    }

    #[test]
    fn band_icon_has_the_right_size() {
        assert_eq!(BAND_ICON.len(), BAND_ICON_PX * BAND_ICON_PX * 4);
        // Not blank: the cartridge is opaque somewhere.
        assert!(BAND_ICON.chunks(4).any(|p| p[3] == 255));
    }

    #[test]
    fn embedded_fonts_load() {
        let f = fonts();
        for name in [COND, MONO, MONO_M] {
            assert!(f.font_data.contains_key(name), "{name}");
            assert!(
                f.families.contains_key(&FontFamily::Name(name.into())),
                "{name}"
            );
        }
    }
}
