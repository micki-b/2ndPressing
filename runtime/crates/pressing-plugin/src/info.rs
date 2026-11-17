//! Info screen (layout A, "spec sheet"): text from `assets/info-screen.md`.
//!
//! The Markdown file is the source of every word on the screen; edit it and
//! rebuild, no code changes needed. Structure (kept from the marketing
//! draft): `## Header band`, `## Readouts`, `## Data sheet` with one `###`
//! section per block, `## Status band`. Inside sections: paragraphs, pipe
//! tables, `**bold**` and footnote markers `[n]` (drawn as superscripts).
//! Lines wholly in `*italics*` are editing guides and are skipped.
//!
//! Special tables: a header starting with "Lamp" draws status lamps
//! (green / amber / off); a header starting with "Background loudness" draws
//! paired bars (first value column above, second below).

// ---- Document model and parser (GUI-free, unit tested) ----

/// The embedded info text.
pub const INFO_MD: &str = include_str!("../assets/info-screen.md");

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readout {
    pub number: String,
    pub caption: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Para(String),
    Table {
        header: Option<Vec<String>>,
        rows: Vec<Vec<String>>,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Section {
    pub title: String,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InfoDoc {
    pub section_label: String,
    pub back_key: String,
    pub readouts: Vec<Readout>,
    pub sections: Vec<Section>,
    pub status_left: String,
    pub status_right: String,
}

/// Inline pieces of a line of text.
#[derive(Clone, Debug, PartialEq)]
pub enum Span {
    Text(String),
    Bold(String),
    Sup(String),
}

fn after_colon(line: &str) -> String {
    line.rsplit_once(": ")
        .map_or(String::new(), |(_, v)| v.trim().to_string())
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cells(line: &str) -> Vec<String> {
    let t = line.trim().trim_start_matches('|').trim_end_matches('|');
    t.split('|').map(collapse).collect()
}

fn is_separator(line: &str) -> bool {
    let t = line.trim();
    // At least one '-': an all-blank row such as `| | |` is an empty header, not a separator.
    t.starts_with('|') && t.contains('-') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// Parse the info Markdown.
pub fn parse(md: &str) -> InfoDoc {
    #[derive(PartialEq)]
    enum Part {
        None,
        Header,
        Readouts,
        Sheet,
        Status,
    }
    let mut doc = InfoDoc::default();
    let mut part = Part::None;
    let mut para: Vec<String> = Vec::new();
    let mut table: Vec<String> = Vec::new();

    let flush = |doc: &mut InfoDoc, para: &mut Vec<String>, table: &mut Vec<String>| {
        let Some(sec) = doc.sections.last_mut() else {
            para.clear();
            table.clear();
            return;
        };
        if !para.is_empty() {
            sec.blocks.push(Block::Para(collapse(&para.join(" "))));
            para.clear();
        }
        if !table.is_empty() {
            let has_header = table.len() >= 2 && is_separator(&table[1]);
            let mut rows: Vec<Vec<String>> = table
                .iter()
                .filter(|l| !is_separator(l))
                .map(|l| cells(l))
                .collect();
            let header = if has_header && !rows.is_empty() {
                let h = rows.remove(0);
                (!h.iter().all(String::is_empty)).then_some(h)
            } else {
                None
            };
            sec.blocks.push(Block::Table { header, rows });
            table.clear();
        }
    };

    for raw in md.lines() {
        let line = raw.trim_end();
        let t = line.trim();
        if let Some(h) = t.strip_prefix("## ") {
            flush(&mut doc, &mut para, &mut table);
            part = match h.trim() {
                s if s.starts_with("Header") => Part::Header,
                s if s.starts_with("Readouts") => Part::Readouts,
                s if s.starts_with("Data sheet") => Part::Sheet,
                s if s.starts_with("Status") => Part::Status,
                _ => Part::None,
            };
            continue;
        }
        if t.starts_with("# ") {
            continue;
        }
        let guide = t.len() > 2 && t.starts_with('*') && t.ends_with('*') && !t.starts_with("**");
        match part {
            Part::None => {}
            Part::Header => {
                if t.starts_with("- Section label") {
                    doc.section_label = after_colon(t);
                } else if t.starts_with("- Back key") {
                    doc.back_key = after_colon(t);
                }
            }
            Part::Readouts => {
                if let Some(rest) = t.split_once(". Number:").map(|(_, r)| r) {
                    doc.readouts.push(Readout {
                        number: rest.trim().to_string(),
                        caption: String::new(),
                    });
                } else if let Some(c) = t.strip_prefix("Caption:") {
                    if let Some(r) = doc.readouts.last_mut() {
                        r.caption = c.trim().to_string();
                    }
                }
            }
            Part::Status => {
                if let Some(v) = t.strip_prefix("- Left:") {
                    doc.status_left = v.trim().to_string();
                } else if let Some(v) = t.strip_prefix("- Right:") {
                    doc.status_right = v.trim().to_string();
                }
            }
            Part::Sheet => {
                if let Some(title) = t.strip_prefix("### ") {
                    flush(&mut doc, &mut para, &mut table);
                    doc.sections.push(Section {
                        title: title.trim().to_string(),
                        blocks: Vec::new(),
                    });
                } else if t.is_empty() {
                    flush(&mut doc, &mut para, &mut table);
                } else if guide {
                    // editing guide, not shown
                } else if t.starts_with('|') {
                    if !para.is_empty() {
                        flush(&mut doc, &mut para, &mut Vec::new());
                    }
                    table.push(t.to_string());
                } else {
                    if !table.is_empty() {
                        flush(&mut doc, &mut Vec::new(), &mut table);
                    }
                    para.push(t.to_string());
                }
            }
        }
    }
    flush(&mut doc, &mut para, &mut table);
    doc
}

/// Split inline markup: `**bold**` and footnote markers `[n]`.
pub fn spans(s: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut rest = s;
    let push_text = |out: &mut Vec<Span>, text: &mut String| {
        if !text.is_empty() {
            out.push(Span::Text(std::mem::take(text)));
        }
    };
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("**") {
            if let Some((b, after)) = r.split_once("**") {
                push_text(&mut out, &mut text);
                out.push(Span::Bold(b.to_string()));
                rest = after;
                continue;
            }
        }
        if rest.starts_with('[') {
            if let Some(end) = rest.find(']') {
                let inner = &rest[1..end];
                if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit()) {
                    // Drop a space before the marker so it hugs the word.
                    if text.ends_with(' ') {
                        text.pop();
                    }
                    push_text(&mut out, &mut text);
                    out.push(Span::Sup(inner.to_string()));
                    rest = &rest[end + 1..];
                    continue;
                }
            }
        }
        let ch = rest.chars().next().unwrap_or(' ');
        text.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    push_text(&mut out, &mut text);
    out
}

/// Leading number part and trailing unit of a readout ("76.5 %" -> ("76.5", "%")).
pub fn split_unit(s: &str) -> (String, String) {
    let t = s.trim();
    match t.rfind(|c: char| c.is_ascii_digit()) {
        Some(i) => {
            let (a, b) = t.split_at(i + 1);
            (a.trim().to_string(), b.trim().to_string())
        }
        None => (t.to_string(), String::new()),
    }
}

/// First number in a cell ("88.2 %" -> 88.2).
pub fn number_in(s: &str) -> Option<f32> {
    let start = s.find(|c: char| c.is_ascii_digit())?;
    let rest = &s[start..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ---- Rendering (egui) ----

use nih_plug_egui::egui::text::{LayoutJob, TextFormat};
use nih_plug_egui::egui::{
    self, Align, Color32, FontFamily, FontId, Pos2, Rect, Sense, Stroke, Vec2,
};

/// Colours and fonts the renderer needs (from the faceplate palette).
pub struct Style {
    pub text: Color32,
    pub sub: Color32,
    pub heading: Color32,
    pub rule: Color32,
    pub ink: Color32,
    pub grey_bar: Color32,
    pub track: Color32,
    pub lamp_green: Color32,
    pub lamp_amber: Color32,
    pub lamp_off: Color32,
    pub sans: &'static str,
    pub sans_bold: &'static str,
    pub cond: &'static str,
    pub mono: &'static str,
}

fn fid(family: &str, size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(family.into()))
}

/// Build a wrapped layout job from inline spans.
#[allow(clippy::too_many_arguments)]
fn job(
    st: &Style,
    text: &str,
    family: &str,
    bold: &str,
    size: f32,
    color: Color32,
    wrap: f32,
    align: Align,
) -> LayoutJob {
    let mut j = LayoutJob::default();
    j.wrap.max_width = wrap;
    j.halign = align;
    let base = TextFormat {
        font_id: fid(family, size),
        color,
        valign: Align::BOTTOM,
        ..Default::default()
    };
    for sp in spans(text) {
        match sp {
            Span::Text(t) => j.append(&t, 0.0, base.clone()),
            Span::Bold(t) => j.append(
                &t,
                0.0,
                TextFormat {
                    font_id: fid(bold, size),
                    ..base.clone()
                },
            ),
            Span::Sup(n) => j.append(
                &n,
                1.0,
                TextFormat {
                    font_id: fid(st.mono, size * 0.62),
                    valign: Align::TOP,
                    ..base.clone()
                },
            ),
        }
    }
    j
}

/// Paint the scrolling data sheet into the current (scroll area) `ui`.
pub fn sheet(ui: &mut egui::Ui, doc: &InfoDoc, st: &Style) {
    let w = ui.available_width();
    for (si, sec) in doc.sections.iter().enumerate() {
        let notes = sec.title.eq_ignore_ascii_case("notes");
        // Heading.
        let mut title = job(
            st,
            &sec.title.to_uppercase(),
            st.cond,
            st.cond,
            10.0,
            st.heading,
            w,
            Align::LEFT,
        );
        title
            .sections
            .iter_mut()
            .for_each(|s| s.format.extra_letter_spacing = 0.8);
        ui.add_space(if si == 0 { 0.0 } else { 9.0 });
        paint_job(ui, title, w);
        ui.add_space(3.0);
        for b in &sec.blocks {
            match b {
                Block::Para(p) => {
                    let j = if notes {
                        job(st, p, st.mono, st.mono, 8.0, st.sub, w, Align::LEFT)
                    } else {
                        job(st, p, st.sans, st.sans_bold, 11.0, st.text, w, Align::LEFT)
                    };
                    paint_job(ui, j, w);
                    ui.add_space(4.0);
                }
                Block::Table { header, rows } => table(ui, header.as_deref(), rows, st, w),
            }
        }
    }
    ui.add_space(6.0);
}

fn paint_job(ui: &mut egui::Ui, j: LayoutJob, w: f32) {
    let galley = ui.fonts(|f| f.layout_job(j));
    let (rect, _) = ui.allocate_exact_size(Vec2::new(w, galley.size().y), Sense::hover());
    ui.painter().galley(rect.min, galley, Color32::PLACEHOLDER);
}

fn table(ui: &mut egui::Ui, header: Option<&[String]>, rows: &[Vec<String>], st: &Style, w: f32) {
    let first = header
        .and_then(|h| h.first())
        .map(String::as_str)
        .unwrap_or("");
    if first.eq_ignore_ascii_case("lamp") {
        return lamp_table(ui, rows, st, w);
    }
    if first.starts_with("Background loudness") {
        return bar_table(ui, header.unwrap_or(&[]), rows, st, w);
    }
    let ncols = rows
        .iter()
        .map(Vec::len)
        .chain(header.map(<[String]>::len))
        .max()
        .unwrap_or(1);
    let (lead, rest) = if ncols <= 2 {
        (w * 0.46, w * 0.54)
    } else {
        (w - 78.0 * (ncols - 1) as f32, 78.0)
    };
    let row = |ui: &mut egui::Ui, cells: &[String], head: bool| {
        let (size, family, bold, color) = if head {
            (8.5, st.cond, st.cond, st.sub)
        } else {
            (10.5, st.sans, st.sans_bold, st.text)
        };
        let mut galleys = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            let (cw, align, fam, b) = if i == 0 {
                (lead - 6.0, Align::LEFT, family, bold)
            } else if ncols <= 2 {
                (
                    rest,
                    Align::RIGHT,
                    if head { family } else { st.mono },
                    if head { bold } else { st.mono },
                )
            } else {
                (
                    rest,
                    Align::RIGHT,
                    if head { family } else { st.mono },
                    if head { bold } else { st.sans_bold },
                )
            };
            let text = if head { c.to_uppercase() } else { c.clone() };
            let mut j = job(
                st,
                &text,
                fam,
                b,
                if i == 0 { size } else { size - 0.5 },
                color,
                cw,
                align,
            );
            if head {
                j.sections
                    .iter_mut()
                    .for_each(|s| s.format.extra_letter_spacing = 0.6);
            }
            galleys.push((ui.fonts(|f| f.layout_job(j)), i));
        }
        let h = galleys.iter().map(|(g, _)| g.size().y).fold(0.0, f32::max) + 6.0;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(w, h), Sense::hover());
        let p = ui.painter();
        for (g, i) in galleys {
            let x = if i == 0 {
                rect.left()
            } else if ncols <= 2 {
                rect.right()
            } else {
                rect.left() + lead + rest * i as f32
            };
            // Right-aligned galleys are positioned by their right edge.
            let pos = Pos2::new(x, rect.top() + 3.0);
            p.galley(pos, g, Color32::PLACEHOLDER);
        }
        p.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(0.6_f32, st.rule),
        );
    };
    if let Some(h) = header {
        row(ui, h, true);
    }
    for r in rows {
        row(ui, r, false);
    }
    ui.add_space(4.0);
}

fn lamp_table(ui: &mut egui::Ui, rows: &[Vec<String>], st: &Style, w: f32) {
    for r in rows {
        let get = |i: usize| r.get(i).cloned().unwrap_or_default();
        let lamp = get(0).to_lowercase();
        let has_lamp = !lamp.is_empty();
        let x0 = if has_lamp { 16.0 } else { 0.0 };
        let name = ui.fonts(|f| {
            f.layout_job(job(
                st,
                &get(1),
                st.sans,
                st.sans_bold,
                10.5,
                st.text,
                w * 0.6 - x0,
                Align::LEFT,
            ))
        });
        let status = ui.fonts(|f| {
            f.layout_job(job(
                st,
                &get(2),
                st.mono,
                st.mono,
                10.0,
                st.text,
                w * 0.4,
                Align::RIGHT,
            ))
        });
        let h = name.size().y.max(status.size().y) + 6.0;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(w, h), Sense::hover());
        let p = ui.painter();
        if has_lamp {
            let col = match lamp.as_str() {
                "green" => st.lamp_green,
                "amber" => st.lamp_amber,
                _ => st.lamp_off,
            };
            let c = Pos2::new(rect.left() + 5.0, rect.top() + 3.0 + name.size().y * 0.5);
            p.circle_filled(c, 4.0, col);
            p.circle_stroke(c, 4.0, Stroke::new(0.6_f32, st.sub));
        }
        p.galley(
            Pos2::new(rect.left() + x0, rect.top() + 3.0),
            name,
            Color32::PLACEHOLDER,
        );
        p.galley(
            Pos2::new(rect.right(), rect.top() + 3.0),
            status,
            Color32::PLACEHOLDER,
        );
        p.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(0.6_f32, st.rule),
        );
    }
    ui.add_space(4.0);
}

fn bar_table(ui: &mut egui::Ui, header: &[String], rows: &[Vec<String>], st: &Style, w: f32) {
    // Legend from the header's value columns.
    let a = header.get(1).cloned().unwrap_or_default();
    let b = header.get(2).cloned().unwrap_or_default();
    let (lr, _) = ui.allocate_exact_size(Vec2::new(w, 14.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(
        Rect::from_min_size(lr.min + Vec2::new(0.0, 4.0), Vec2::new(14.0, 5.0)),
        1.0,
        st.ink,
    );
    p.text(
        lr.min + Vec2::new(18.0, 0.0),
        egui::Align2::LEFT_TOP,
        &a,
        fid(st.mono, 8.5),
        st.text,
    );
    let off = 30.0 + a.chars().count() as f32 * 5.3;
    p.rect_filled(
        Rect::from_min_size(lr.min + Vec2::new(off, 4.0), Vec2::new(14.0, 5.0)),
        1.0,
        st.grey_bar,
    );
    p.text(
        lr.min + Vec2::new(off + 18.0, 0.0),
        egui::Align2::LEFT_TOP,
        &b,
        fid(st.mono, 8.5),
        st.text,
    );

    let label_w = 96.0;
    let val_w = 48.0;
    let bar_w = w - label_w - val_w - 8.0;
    for r in rows {
        let v1 = r.get(1).and_then(|c| number_in(c)).unwrap_or(0.0);
        let v2 = r.get(2).and_then(|c| number_in(c)).unwrap_or(0.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 24.0), Sense::hover());
        let p = ui.painter();
        p.text(
            Pos2::new(rect.left(), rect.center().y),
            egui::Align2::LEFT_CENTER,
            r.first().map_or("", String::as_str),
            fid(st.mono, 8.5),
            st.text,
        );
        let x = rect.left() + label_w;
        for (k, (v, col)) in [(v1, st.ink), (v2, st.grey_bar)].into_iter().enumerate() {
            let y = rect.top() + 5.0 + k as f32 * 8.0;
            let track = Rect::from_min_size(Pos2::new(x, y), Vec2::new(bar_w, 6.0));
            p.rect_filled(track, 1.0, st.track);
            p.rect_filled(
                Rect::from_min_size(
                    track.min,
                    Vec2::new(bar_w * (v / 100.0).clamp(0.0, 1.0), 6.0),
                ),
                1.0,
                col,
            );
            p.text(
                Pos2::new(rect.right(), y + 3.0),
                egui::Align2::RIGHT_CENTER,
                r.get(k + 1).map_or("", String::as_str),
                fid(st.mono, 8.5),
                if k == 0 { st.text } else { st.sub },
            );
        }
        p.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(0.6_f32, st.rule),
        );
    }
    ui.add_space(4.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shipped_info_text() {
        let d = parse(INFO_MD);
        assert_eq!(d.section_label, "INFO");
        assert_eq!(d.back_key, "CONTROLS");
        assert_eq!(d.readouts.len(), 3);
        assert_eq!(d.readouts[0].number, ">76%");
        assert!(d.readouts[0].caption.starts_with("Clicks found"));
        assert!(!d.status_left.is_empty() && !d.status_right.is_empty());
        let titles: Vec<&str> = d.sections.iter().map(|s| s.title.as_str()).collect();
        assert_eq!(titles.first(), Some(&"More music. Fewer clicks."));
        assert_eq!(titles.last(), Some(&"Notes"));
        // No editing guides leak into the screen.
        for s in &d.sections {
            for b in &s.blocks {
                if let Block::Para(p) = b {
                    assert!(
                        !(p.starts_with('*') && p.ends_with('*') && !p.starts_with("**")),
                        "{p}"
                    );
                }
            }
        }
        // Special tables are recognised.
        let tables: Vec<&Option<Vec<String>>> = d
            .sections
            .iter()
            .flat_map(|s| &s.blocks)
            .filter_map(|b| {
                if let Block::Table { header, .. } = b {
                    Some(header)
                } else {
                    None
                }
            })
            .collect();
        assert!(tables
            .iter()
            .any(|h| h.as_ref().is_some_and(|h| h[0] == "Lamp")));
        assert!(tables.iter().any(|h| h
            .as_ref()
            .is_some_and(|h| h[0].starts_with("Background loudness"))));
        assert!(
            tables.iter().any(|h| h.is_none()),
            "engine table has an empty header"
        );
        // The engine row keeps its footnote marker.
        let engine = d
            .sections
            .iter()
            .find(|s| s.title.starts_with("Engine"))
            .unwrap();
        let Block::Table { rows, .. } = &engine.blocks[0] else {
            panic!()
        };
        assert!(rows.iter().any(|r| r[0].contains("[2]")));
        assert!(rows.iter().any(|r| r[0] == "High Efficiency Model"));
    }

    #[test]
    fn inline_markup() {
        assert_eq!(
            spans("about **8× fewer** clicks [3] here"),
            vec![
                Span::Text("about ".into()),
                Span::Bold("8× fewer".into()),
                Span::Text(" clicks".into()),
                Span::Sup("3".into()),
                Span::Text(" here".into()),
            ]
        );
        assert_eq!(spans("[x] stays"), vec![Span::Text("[x] stays".into())]);
    }

    #[test]
    fn readout_units_and_numbers() {
        assert_eq!(split_unit(">76%"), (">76".into(), "%".into()));
        assert_eq!(split_unit("0.030 %"), ("0.030".into(), "%".into()));
        assert_eq!(split_unit("43 ×"), ("43".into(), "×".into()));
        assert_eq!(number_in("88.2 %"), Some(88.2));
        assert_eq!(number_in("−45 … −30 dBFS"), Some(45.0));
    }
}
