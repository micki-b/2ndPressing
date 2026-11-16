//! The window: the plugin's Instrument faceplate on top, an audio-setup
//! strip in the same style below it (devices, channel pairs, rate, input
//! level, and a status line that says plainly what is wrong when audio does
//! not flow).

use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, CornerRadius, Pos2, Rect, RichText, Stroke, UiBuilder, Vec2,
};
use pressing_core::Model;
use pressing_plugin::editor::{
    self, font, vgrad, Controls, Panel, AMBER, AMBER_FAINT, COND, FOOT, GLASS_BOT, GLASS_TOP,
    GREEN, KEY_EDGE, KEY_ON_BOT, KEY_ON_TOP, KEY_TEXT, KEY_TOP, LAMP_OFF, LEGEND, MONO, PANEL_BOT,
    PANEL_TOP, PAPER, PAPER_EDGE, RED,
};
use pressing_plugin::{Monitor, PercussionProtection};

use crate::audio::{self, DeviceInfo, Route, Session, SessionConfig, Shared};
use crate::settings::{self, Settings};

/// Window/taskbar icon (the logo's app icon, 256 px straight RGBA, from
/// `brand/export.py`). The macOS bundle uses `macos/AppIcon.icns` instead.
const WINDOW_ICON: &[u8] = include_bytes!("../assets/window-icon-256.rgba");

/// Height of the audio-setup strip under the faceplate.
pub const STRIP_H: f32 = 112.0;

/// Open the window and run until it is closed.
pub fn run() -> Result<(), String> {
    let size = [
        editor::WIDTH as f32,
        editor::HEIGHT as f32 + STRIP_H,
    ];
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("2ndPressing Declicker")
            .with_icon(Arc::new(egui::IconData {
                rgba: WINDOW_ICON.to_vec(),
                width: 256,
                height: 256,
            }))
            .with_inner_size(size)
            .with_min_inner_size(size)
            .with_max_inner_size(size)
            .with_resizable(false),
        ..Default::default()
    };
    eframe::run_native(
        "2ndPressing",
        opts,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .map_err(|e| e.to_string())
}

// ---- Controls on the shared atomics ----

struct AppControls<'a>(&'a Shared);

impl Controls for AppControls<'_> {
    fn sensitivity(&self) -> f32 {
        f32::from_bits(self.0.sensitivity.load(Relaxed))
    }
    fn set_sensitivity(&self, percent: f32) {
        self.0
            .sensitivity
            .store(percent.clamp(0.0, 100.0).to_bits(), Relaxed);
    }
    fn protection(&self) -> PercussionProtection {
        audio::protection_from(self.0.protection.load(Relaxed))
    }
    fn set_protection(&self, p: PercussionProtection) {
        self.0.protection.store(audio::protection_index(p), Relaxed);
    }
    fn monitor(&self) -> Monitor {
        audio::monitor_from(self.0.monitor.load(Relaxed))
    }
    fn set_monitor(&self, m: Monitor) {
        self.0.monitor.store(audio::monitor_index(m), Relaxed);
    }
    fn bypass(&self) -> bool {
        self.0.bypass.load(Relaxed)
    }
    fn set_bypass(&self, on: bool) {
        self.0.bypass.store(on, Relaxed);
    }
}

// ---- App ----

struct App {
    panel: Panel,
    shared: Arc<Shared>,
    model: Model,
    settings: Settings,
    saved: Settings,
    inputs: Vec<DeviceInfo>,
    outputs: Vec<DeviceInfo>,
    session: Option<Session>,
    /// The config the session was started with (or failed to start with).
    config: Option<SessionConfig>,
    start_error: Option<String>,
    started: Instant,
    /// Input callback count and when it last changed.
    last_in: (u64, Instant),
    last_signal: Instant,
    level_db: f32,
    any_combo_open: bool,
    restart: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_fonts(editor::fonts());
        style(&cc.egui_ctx);
        let model = Model::embedded();
        let settings = settings::load();
        let shared = Arc::new(Shared::new(
            settings.sensitivity,
            audio::protection_from(settings.protection),
            audio::monitor_from(settings.monitor),
        ));
        let (inputs, outputs) = audio::scan();
        let now = Instant::now();
        App {
            panel: Panel::new(model.post.threshold),
            shared,
            model,
            saved: settings.clone(),
            settings,
            inputs,
            outputs,
            session: None,
            config: None,
            start_error: None,
            started: now,
            last_in: (0, now),
            last_signal: now,
            level_db: -120.0,
            any_combo_open: false,
            restart: true,
        }
    }

    fn input_info(&self) -> Option<&DeviceInfo> {
        let name = self.settings.input.as_ref()?;
        self.inputs.iter().find(|d| &d.name == name)
    }

    fn output_info(&self) -> Option<&DeviceInfo> {
        let name = self.settings.output.as_ref()?;
        self.outputs.iter().find(|d| &d.name == name)
    }

    /// Fill in missing choices from the system defaults, then (re)open.
    fn start(&mut self) {
        self.restart = false;
        // Release the devices first.
        self.session = None;
        let (def_in, def_out) = audio::default_names();
        if self.input_info().is_none() && self.settings.input.is_none() {
            self.settings.input = def_in.or_else(|| self.inputs.first().map(|d| d.name.clone()));
        }
        if self.output_info().is_none() && self.settings.output.is_none() {
            self.settings.output =
                def_out.or_else(|| self.outputs.first().map(|d| d.name.clone()));
        }
        self.started = Instant::now();
        self.last_in = (0, self.started);
        self.last_signal = self.started;
        let (Some(inp), Some(out)) = (self.input_info().cloned(), self.output_info().cloned())
        else {
            self.start_error = Some(match (&self.settings.input, self.input_info()) {
                (None, _) => "NO INPUT DEVICE FOUND".into(),
                (Some(n), None) => format!("{n} IS NOT CONNECTED"),
                _ => match &self.settings.output {
                    None => "NO OUTPUT DEVICE FOUND".into(),
                    Some(n) => format!("{n} IS NOT CONNECTED"),
                },
            });
            self.config = None;
            return;
        };
        self.settings.in_route = fit(self.settings.in_route, inp.channels);
        self.settings.out_route = fit(self.settings.out_route, out.channels);
        let rate = self
            .settings
            .rate
            .filter(|r| inp.supports(*r))
            .unwrap_or(inp.default_rate);
        let cfg = SessionConfig {
            input: inp.name,
            in_route: self.settings.in_route,
            output: out.name,
            out_route: self.settings.out_route,
            rate,
        };
        match audio::start(&cfg, &self.model, &self.shared) {
            Ok(s) => {
                self.session = Some(s);
                self.start_error = None;
            }
            Err(e) => self.start_error = Some(e),
        }
        self.config = Some(cfg);
    }

    fn rescan(&mut self) {
        let (i, o) = audio::scan();
        self.inputs = i;
        self.outputs = o;
    }

    /// Pull control values from the shared atomics and save on change.
    fn persist(&mut self) {
        let c = AppControls(&self.shared);
        self.settings.sensitivity = c.sensitivity();
        self.settings.protection = audio::protection_index(c.protection());
        self.settings.monitor = audio::monitor_index(c.monitor());
        if self.settings != self.saved {
            settings::save(&self.settings);
            self.saved = self.settings.clone();
        }
    }

    /// Lamp colour, short status, and a longer explanation for the tooltip.
    fn status(&mut self) -> (Color32, String, String) {
        let now = Instant::now();
        if let Some(e) = &self.start_error {
            return (RED, e.to_uppercase(), e.clone());
        }
        if let Some(e) = self.shared.error.lock().ok().and_then(|g| g.clone()) {
            return (
                RED,
                "AUDIO DEVICE ERROR — RESELECT OR RECONNECT".into(),
                e,
            );
        }
        let (Some(s), Some(cfg)) = (&self.session, &self.config) else {
            return (LAMP_OFF, "STOPPED".into(), String::new());
        };
        let n_in = self.shared.in_callbacks.load(Relaxed);
        if n_in != self.last_in.0 {
            self.last_in = (n_in, now);
        }
        let quiet_for = now - self.last_in.1;
        if quiet_for > Duration::from_millis(1500) {
            let short = if n_in == 0 {
                format!("NO AUDIO FROM {}", cfg.input.to_uppercase())
            } else {
                format!("{} STOPPED SENDING AUDIO", cfg.input.to_uppercase())
            };
            return (
                RED,
                short,
                format!(
                    "{} is open but delivers no audio. Usual causes: another app \
                     (e.g. VinylStudio) holds the device exclusively — quit it and reselect; \
                     or input access is blocked — System Settings › Privacy & Security › \
                     Microphone must allow 2ndPressing (or Terminal, when launched from it).",
                    cfg.input
                ),
            );
        }
        if self.shared.out_callbacks.load(Relaxed) == 0
            && now - self.started > Duration::from_millis(1500)
        {
            return (
                RED,
                format!("{} IS NOT PLAYING", cfg.output.to_uppercase()),
                "The output device accepted the stream but never asked for audio.".into(),
            );
        }
        if now - self.last_signal > Duration::from_secs(3) {
            return (
                AMBER,
                "INPUT IS DIGITAL SILENCE".into(),
                format!(
                    "{} sends exact zeros on channels {}. Check the channel pair, that the \
                     turntable is playing, and that microphone access is allowed (macOS \
                     delivers silence to apps without it).",
                    cfg.input,
                    cfg.in_route.label()
                ),
            );
        }
        let khz = |r: u32| format!("{:.1}", f64::from(r) / 1000.0);
        let path = if s.locked {
            format!("{} kHz · BIT-EXACT", khz(s.in_rate))
        } else if s.in_rate == s.out_rate {
            format!("{} kHz · CLOCK-MATCHED", khz(s.in_rate))
        } else {
            format!("{} → {} kHz", khz(s.in_rate), khz(s.out_rate))
        };
        let under = self.shared.underruns.load(Relaxed);
        let mut short = format!("RUNNING · {path}");
        if under > 0 {
            short.push_str(&format!(" · {under} DROPOUTS"));
        }
        let trim = f32::from_bits(self.shared.trim_ppm.load(Relaxed));
        let detail = format!(
            "{} ({} ch, {} kHz) channels {} → {} ({} ch, {} kHz) channels {}. {}{} \
             Output underruns: {under}; input overflows: {}; resyncs: {}.",
            cfg.input,
            s.in_channels,
            khz(s.in_rate),
            cfg.in_route.label(),
            cfg.output,
            s.out_channels,
            khz(s.out_rate),
            cfg.out_route.label(),
            if s.locked {
                "One device, one clock: output samples are the engine's samples, unaltered."
            } else {
                "Two clocks: the output is rate-matched to the input."
            },
            if s.locked {
                String::new()
            } else {
                format!(" Clock trim {trim:+.0} ppm.")
            },
            self.shared.overflows.load(Relaxed),
            self.shared.resyncs.load(Relaxed),
        );
        (GREEN, short, detail)
    }
}

/// Keep a route that fits, else the first pair (or channel 1 as mono).
fn fit(r: Route, channels: u16) -> Route {
    if r.fits(channels) {
        r
    } else if channels >= 2 {
        Route::FIRST_PAIR
    } else {
        Route { left: 0, right: 0 }
    }
}

fn style(ctx: &egui::Context) {
    let mut s = (*ctx.style()).clone();
    s.visuals = egui::Visuals::light();
    let v = &mut s.visuals;
    v.window_fill = PAPER;
    v.panel_fill = PANEL_TOP;
    v.window_stroke = Stroke::new(1.0_f32, PAPER_EDGE);
    v.selection.bg_fill = Color32::from_rgb(0xE9, 0xC9, 0x8A);
    v.selection.stroke = Stroke::new(1.0_f32, KEY_TEXT);
    v.override_text_color = Some(KEY_TEXT);
    let w = &mut v.widgets;
    for ws in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        ws.corner_radius = CornerRadius::same(3);
        ws.fg_stroke = Stroke::new(1.0_f32, KEY_TEXT);
        ws.bg_stroke = Stroke::new(1.0_f32, KEY_EDGE);
    }
    w.inactive.weak_bg_fill = KEY_TOP;
    w.inactive.bg_fill = KEY_TOP;
    w.hovered.weak_bg_fill = KEY_ON_TOP;
    w.hovered.bg_fill = KEY_ON_TOP;
    w.open.weak_bg_fill = KEY_ON_TOP;
    w.active.weak_bg_fill = KEY_ON_BOT;
    s.spacing.interact_size.y = 22.0;
    ctx.set_style(s);
}

fn item(text: impl Into<String>) -> RichText {
    RichText::new(text).font(font(MONO, 10.5)).color(KEY_TEXT)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.restart {
            self.start();
        }
        // Input level: take the peak since the last frame, decay the display.
        let pk = f32::from_bits(self.shared.peak.swap(0, Relaxed));
        if pk > 0.0 {
            self.last_signal = Instant::now();
        }
        let db = if pk > 0.0 { 20.0 * pk.log10() } else { -120.0 };
        self.level_db = db.max(self.level_db - 1.5);

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let o = ui.max_rect().min;
                let face = Rect::from_min_size(
                    o,
                    Vec2::new(editor::WIDTH as f32, editor::HEIGHT as f32),
                );
                ui.scope_builder(UiBuilder::new().max_rect(face), |ui| {
                    self.panel
                        .show(ui, &AppControls(&self.shared), &self.shared.stats);
                });
                self.strip(ui, o + Vec2::new(0.0, editor::HEIGHT as f32));
            });

        self.persist();
        if self.restart {
            ctx.request_repaint();
        }
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

impl App {
    fn strip(&mut self, ui: &mut egui::Ui, o: Pos2) {
        let p = |x: f32, y: f32| o + Vec2::new(x, y);
        let r = |x: f32, y: f32, w: f32, h: f32| Rect::from_min_size(p(x, y), Vec2::new(w, h));
        let pt = ui.painter().clone();
        let w = editor::WIDTH as f32;
        vgrad(&pt, r(0.0, 0.0, w, STRIP_H), PANEL_TOP, PANEL_BOT);
        pt.line_segment([p(0.0, 0.5), p(w, 0.5)], Stroke::new(1.0_f32, KEY_EDGE));
        pt.text(
            p(26.0, 10.0),
            Align2::LEFT_TOP,
            "AUDIO",
            font(COND, 10.0),
            LEGEND,
        );

        let mut open = false;
        let in_rows = 28.0;
        let out_rows = 56.0;
        for (y, label) in [(in_rows, "INPUT"), (out_rows, "OUTPUT")] {
            pt.text(
                p(26.0, y + 11.0),
                Align2::LEFT_CENTER,
                label,
                font(COND, 10.0),
                LEGEND,
            );
        }

        // Device menus.
        for input in [true, false] {
            let y = if input { in_rows } else { out_rows };
            let list = if input { &self.inputs } else { &self.outputs };
            let current = if input {
                self.settings.input.clone()
            } else {
                self.settings.output.clone()
            };
            let mut pick: Option<String> = None;
            let resp = ui.scope_builder(UiBuilder::new().max_rect(r(86.0, y, 212.0, 22.0)), |ui| {
                egui::ComboBox::from_id_salt(if input { "in-dev" } else { "out-dev" })
                    .width(212.0)
                    .height(320.0)
                    .selected_text(item(current.clone().unwrap_or_else(|| "—".into())))
                    .show_ui(ui, |ui| {
                        for d in list {
                            let sel = current.as_deref() == Some(d.name.as_str());
                            let text = format!("{}  ·  {} ch", d.name, d.channels);
                            if ui.selectable_label(sel, item(text)).clicked() && !sel {
                                pick = Some(d.name.clone());
                            }
                        }
                    })
            });
            open |= resp.inner.inner.is_some();
            if let Some(name) = pick {
                if input {
                    self.settings.input = Some(name);
                    self.settings.rate = None;
                } else {
                    self.settings.output = Some(name);
                }
                self.restart = true;
            }
        }

        // Channel-pair menus.
        for input in [true, false] {
            let y = if input { in_rows } else { out_rows };
            let info = if input {
                self.input_info()
            } else {
                self.output_info()
            };
            let channels = info.map(|d| d.channels).unwrap_or(2);
            let current = if input {
                self.settings.in_route
            } else {
                self.settings.out_route
            };
            let mut pick = None;
            let resp = ui.scope_builder(UiBuilder::new().max_rect(r(306.0, y, 84.0, 22.0)), |ui| {
                egui::ComboBox::from_id_salt(if input { "in-ch" } else { "out-ch" })
                    .width(84.0)
                    .height(320.0)
                    .selected_text(item(format!("CH {}", current.label())))
                    .show_ui(ui, |ui| {
                        for rt in audio::routes(channels) {
                            let sel = rt == current;
                            if ui.selectable_label(sel, item(rt.label())).clicked() && !sel {
                                pick = Some(rt);
                            }
                        }
                    })
            });
            open |= resp.inner.inner.is_some();
            if let Some(rt) = pick {
                if input {
                    self.settings.in_route = rt;
                } else {
                    self.settings.out_route = rt;
                }
                self.restart = true;
            }
        }

        // Rate menu (input/processing rate).
        let rates = self.input_info().map(|d| d.rates.clone()).unwrap_or_default();
        let now_rate = self.config.as_ref().map(|c| c.rate);
        let mut pick = None;
        let resp = ui.scope_builder(UiBuilder::new().max_rect(r(398.0, in_rows, 96.0, 22.0)), |ui| {
            egui::ComboBox::from_id_salt("rate")
                .width(96.0)
                .selected_text(item(match now_rate {
                    Some(sr) => format!("{:.1} kHz", f64::from(sr) / 1000.0),
                    None => "— kHz".into(),
                }))
                .show_ui(ui, |ui| {
                    for sr in rates {
                        let sel = Some(sr) == now_rate;
                        let mut t = format!("{:.1} kHz", f64::from(sr) / 1000.0);
                        if !matches!(sr, 44_100 | 48_000) {
                            t.push_str("  (exp.)");
                        }
                        if ui.selectable_label(sel, item(t)).clicked() && !sel {
                            pick = Some(sr);
                        }
                    }
                })
        });
        open |= resp.inner.inner.is_some();
        if let Some(sr) = pick {
            self.settings.rate = Some(sr);
            self.restart = true;
        }

        // Input level readout.
        let glass = r(398.0, out_rows, 96.0, 22.0);
        vgrad(&pt, glass, GLASS_TOP, GLASS_BOT);
        let lamp_on = self.level_db > -60.0;
        pt.circle_filled(
            glass.left_center() + Vec2::new(10.0, 0.0),
            3.5,
            if lamp_on { GREEN } else { LAMP_OFF },
        );
        let lvl = if self.level_db <= -119.0 {
            "SILENCE".to_string()
        } else {
            format!("{:.0} dB", self.level_db)
        };
        pt.text(
            glass.right_center() - Vec2::new(8.0, 0.0),
            Align2::RIGHT_CENTER,
            lvl,
            font(MONO, 10.0),
            if lamp_on { AMBER } else { AMBER_FAINT },
        );

        // Rescan devices when a device menu opens (hot-plugged hardware).
        if open && !self.any_combo_open {
            self.rescan();
        }
        self.any_combo_open = open;

        // Status line.
        let band = r(0.0, 86.0, w, STRIP_H - 86.0);
        pt.rect_filled(band, 0.0, FOOT);
        let (lamp, short, detail) = self.status();
        pt.circle_filled(p(30.0, 99.0), 4.0, lamp);
        pt.text(
            p(42.0, 99.0),
            Align2::LEFT_CENTER,
            short,
            font(MONO, 9.5),
            LEGEND,
        );
        let hover = ui.interact(band, ui.id().with("status"), egui::Sense::hover());
        if !detail.is_empty() {
            hover.on_hover_ui(|ui| {
                ui.set_max_width(360.0);
                ui.label(RichText::new(detail).font(font(MONO, 10.0)));
            });
        }
        // A failed start can be retried by clicking the status line.
        if self.session.is_none()
            && ui
                .interact(band, ui.id().with("retry"), egui::Sense::click())
                .clicked()
        {
            self.rescan();
            self.restart = true;
        }
    }
}
