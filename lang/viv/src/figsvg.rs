// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv plot`: a run's figure records drawn as SVG,
//! with no browser and no JavaScript.
//!
//! ```text
//! viv run prog.viv | viv plot -o fig.svg
//! viv plot run.out --chart 2 -o fig.svg
//! ```
//!
//! The same records always give the same bytes: every coordinate is written
//! with a fixed number of decimals by Rust's correctly rounded formatting, the
//! logarithms and powers come from Viv's own libm (`vmath`), text widths are a
//! fixed estimate per character (no font is measured), nothing is keyed by a
//! hash, and the clip-path ids are the chart's position. So an SVG can be
//! committed and compared byte for byte, like the records themselves
//! (tests/plot_svg.rs holds golden SVGs for probes of the corpus).
//!
//! What it draws: line, scatter and bar charts; `fill_between` bands and
//! `rangebars`; the marks (`hline`, `vline`, `xtick`, `ytick`, `label`,
//! `rect`, `segment`); log axes; series styles; `aspect("equal")`; a right y
//! axis (`yright`); grids of figures, with shared ranges. It follows the site's
//! renderer defaults in the
//! light palette. A heatmap (`plot2d`) is drawn as an embedded PNG with a
//! colorbar, under the figure's `origin`, `colormap` and `clim`; its contour
//! `levels` are not drawn. Animated heatmaps (`frames2d`), 3-D curves and
//! scenes, vector fields and animations are not drawn: each is listed under
//! the figure as "not drawn by viv plot".
//!
//! The grouping follows the spec's rules, simplified where drawing does not
//! depend on them: every series carries its own x, so a figure's plots are one
//! chart whatever their x.

use crate::mathtext::{self, Anchor, Label};
use crate::vmath;
use std::cell::RefCell;
use std::fmt::Write;

thread_local! {
    /// The labels whose math did not parse, since the last take_label_errors().
    static LABEL_ERRORS: RefCell<Vec<(String, mathtext::MathError)>> = const { RefCell::new(Vec::new()) };
}

/// A label laid out at a text size: its `$...$` parts set as math
/// (mathtext.rs). A label whose math has an error is drawn as its text, and
/// the error kept for `viv plot` to report.
fn lab(s: &str, size: f64) -> Label {
    match mathtext::label(s, size) {
        Ok(l) => l,
        Err(e) => {
            LABEL_ERRORS.with(|v| {
                let mut v = v.borrow_mut();
                if !v.iter().any(|(l, _)| l == s) {
                    v.push((s.to_string(), e));
                }
            });
            mathtext::plain(s, size)
        }
    }
}

/// The labels drawn since the last call whose math has an error, each with
/// where in the records it is: "line N: in the label "...": ... at column K".
pub fn take_label_errors(records: &str) -> Vec<String> {
    let errs = LABEL_ERRORS.with(|v| std::mem::take(&mut *v.borrow_mut()));
    errs.into_iter()
        .map(|(label, e)| {
            let line = records.lines().position(|l| l.contains(label.as_str())).map(|k| format!("line {}: ", k + 1)).unwrap_or_default();
            format!("{line}in the label \"{label}\": {e}")
        })
        .collect()
}

/// A label's width in characters of the tick font, for the margins.
fn label_chars(s: &str, size: f64) -> f64 {
    match mathtext::label(s, size) {
        Ok(l) if l.math => l.w / 6.4,
        _ => s.chars().count() as f64,
    }
}

// Light-theme series colors: the data palette's four (cyan, amber, indigo,
// green; data_palette.rs, generated from www/src/lib/dataPalette.mjs), then a
// fifth and sixth (www/src/styles/viv.css --s5, --s6), and the ink.
#[path = "data_palette.rs"]
mod data_palette;
const PALETTE: [&str; 6] = [data_palette::DATA_LIGHT[0], data_palette::DATA_LIGHT[1], data_palette::DATA_LIGHT[2], data_palette::DATA_LIGHT[3], "#be185d", "#64748b"];
/// The primary series (the first palette slot) is drawn heavier, and on the
/// site in dark mode with a soft glow (its class, `vp-primary`).
const PRIMARY_WIDTH: f64 = 2.4;
/// Bars: nearly opaque, slightly rounded, no outline.
const BAR_OPACITY: &str = "0.88";
const BAR_RX: &str = "1.5";
const BG: &str = "#ffffff";
const BORDER: &str = "#e0daea";
const AXIS: &str = "#c5bad6";
const GRID: &str = "#e0daea";
const TICK: &str = "#8d82a1";
const LABEL: &str = "#6b5f80";
const TITLE: &str = "#1a1226";
const MARK: &str = "#8d82a1";
const FONT: &str = "'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, sans-serif";

fn named_color(name: &str) -> Option<&'static str> {
    Some(match name {
        "red" => "#d62728",
        "blue" => "#1f77b4",
        "green" => "#2ca02c",
        "purple" => "#9467bd",
        "orange" => "#ff7f0e",
        "black" => "#333333",
        _ => return None,
    })
}

// ---- the records --------------------------------------------------------------

/// One number of a record: C's spellings, `nan`, `-nan`, `inf`, `-inf` included.
fn num(t: &str) -> f64 {
    match t.trim() {
        "nan" | "-nan" => f64::NAN,
        "inf" | "+inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        s => s.parse().unwrap_or(f64::NAN),
    }
}

/// A bracketed array field, `[1, 2.5, nan]`; `[]` is empty. None if not an array.
fn arr(field: &str) -> Option<Vec<f64>> {
    let t = field.trim();
    let inner = t.strip_prefix('[')?.strip_suffix(']')?;
    Some(if inner.trim().is_empty() { Vec::new() } else { inner.split(',').map(num).collect() })
}

fn nums(field: &str) -> Vec<f64> {
    field.split_whitespace().map(num).collect()
}

#[derive(Clone, Default)]
struct Style {
    line: Option<String>,
    color: Option<usize>,
    marker: Option<String>,
    width: Option<String>,
}

#[derive(Clone, Default)]
struct Fig {
    title: String,
    xlabel: String,
    ylabel: String,
    xlim: Option<(f64, f64)>,
    ylim: Option<(f64, f64)>,
    xlog: bool,
    ylog: bool,
    equal: bool,
    styles: Vec<(String, Style)>,
    yright: Option<(String, Vec<String>)>,
    /// the figure's style token (`contour`, `pixels`, `diagram`)
    token: String,
    /// in a grid: the tick labels a neighbor panel shows (x, y), and room for a right axis
    inner: (bool, bool),
    in_grid: bool,
    room_right: bool,
    /// heatmap settings: row 0 at the bottom, the colormap's name, its limits
    origin_lower: bool,
    colormap: String,
    clim: Option<(f64, f64)>,
}

impl Fig {
    fn style(&self, name: &str) -> Option<&Style> {
        self.styles.iter().rev().find(|(n, _)| n == name).map(|(_, s)| s)
    }
    fn right(&self, name: &str) -> bool {
        self.yright.as_ref().is_some_and(|(_, s)| s.iter().any(|n| n == name))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Line,
    Points,
    Bars,
    Fill,
    Range,
}

#[derive(Clone)]
struct Series {
    name: String,
    kind: Kind,
    xs: Vec<f64>,
    ys: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    color: String,
    open: bool,
}

#[derive(Clone)]
enum Mark {
    HLine(f64),
    VLine(f64),
    XTick(String, f64),
    YTick(String, f64),
    Label(String, f64, f64),
    Rect(f64, f64, f64, f64),
    Segment(f64, f64, f64, f64),
}

/// A diagram's mark: its id, its style
/// words, and what it is.
#[derive(Clone)]
struct DMark {
    id: String,
    style: String,
    what: DKind,
}

#[derive(Clone)]
enum DKind {
    Box(f64, f64, f64, f64),
    Arrow(Vec<f64>, Vec<f64>),
    Note(String, f64, f64),
}

#[derive(Clone, Default)]
struct Chart {
    fig: Option<Fig>,
    series: Vec<Series>,
    marks: Vec<Mark>,
    /// a diagram's boxes, arrows and notes, in record order
    diagram: Vec<DMark>,
    /// charts of a kind this renderer does not draw, met in this figure
    undrawn: Vec<&'static str>,
    /// a heatmap: rows, columns and the values row by row
    heat: Option<(usize, usize, Vec<f64>)>,
}

enum Entry {
    Chart(usize),
    Grid { title: String, rows: usize, cols: usize, share: String, panels: Vec<Vec<usize>> },
}

struct Parsed {
    charts: Vec<Chart>,
    entries: Vec<Entry>,
}

fn parse_mark(verb: &str, f: &[&str]) -> Option<Mark> {
    let n = |i: usize| f.get(i).map(|s| nums(s)).unwrap_or_default();
    let m = match verb {
        "hline" => Mark::HLine(*n(1).first()?),
        "vline" => Mark::VLine(*n(1).first()?),
        "xtick" => Mark::XTick(f.first()?.to_string(), *n(1).first()?),
        "ytick" => Mark::YTick(f.first()?.to_string(), *n(1).first()?),
        "label" => {
            let v = n(2);
            Mark::Label(f.get(1)?.to_string(), *v.first()?, *v.get(1)?)
        }
        "rect" => {
            let v = n(1);
            Mark::Rect(*v.first()?, *v.get(1)?, *v.get(2)?, *v.get(3)?)
        }
        "segment" => {
            let v = n(1);
            Mark::Segment(*v.first()?, *v.get(1)?, *v.get(2)?, *v.get(3)?)
        }
        _ => return None,
    };
    Some(m)
}

fn parse_dmark(verb: &str, f: &[&str]) -> Option<DMark> {
    let id = f.first()?.to_string();
    let what = match verb {
        "box" => {
            let v = nums(f.get(2)?);
            if v.len() != 4 || !v.iter().all(|x| x.is_finite()) {
                return None;
            }
            DKind::Box(v[0], v[1], v[2], v[3])
        }
        "arrow" => {
            let (xs, ys) = (arr(f.get(2)?)?, arr(f.get(3)?)?);
            if xs.len() != ys.len() || xs.len() < 2 {
                return None;
            }
            DKind::Arrow(xs, ys)
        }
        _ => {
            let v = nums(f.get(3)?);
            if v.len() != 2 || !v.iter().all(|x| x.is_finite()) {
                return None;
            }
            DKind::Note(f.get(1)?.to_string(), v[0], v[1])
        }
    };
    let style = if verb == "note" { f.get(2) } else { f.get(1) };
    Some(DMark { id, style: style.map(|s| s.trim().to_string()).unwrap_or_default(), what })
}

const STYLE_WORDS: [(&str, &[&str]); 4] = [
    ("line", &["solid", "dash", "dot", "noline"]),
    ("color", &["s1", "s2", "s3", "s4", "s5", "s6"]),
    ("marker", &["circle", "ring", "square", "cross", "nomarker"]),
    ("width", &["thin", "thick"]),
];

fn parse_style(words: &str) -> Style {
    let mut st = Style::default();
    for w in words.split_whitespace() {
        for (aspect, ws) in STYLE_WORDS {
            if ws.contains(&w) {
                match aspect {
                    "line" => st.line = Some(w.to_string()),
                    "color" => st.color = w[1..].parse::<usize>().ok().map(|k| k - 1),
                    "marker" => st.marker = Some(w.to_string()),
                    _ => st.width = Some(w.to_string()),
                }
            }
        }
    }
    st
}

fn parse(text: &str) -> Parsed {
    let mut p = Parsed { charts: Vec::new(), entries: Vec::new() };
    let mut cur: Option<usize> = None; // the chart the open figure draws into
    let mut in_anim = false;
    // the open grid (its index in entries) and whether a panel of it is open
    let mut grid: Option<usize> = None;
    let mut in_panel = false;

    // where a new chart goes: the open panel, or the top level
    fn place(p: &mut Parsed, grid: Option<usize>, in_panel: bool, k: usize) {
        if let (Some(g), true) = (grid, in_panel) {
            if let Entry::Grid { panels, .. } = &mut p.entries[g] {
                if let Some(last) = panels.last_mut() {
                    last.push(k);
                    return;
                }
            }
        }
        p.entries.push(Entry::Chart(k));
    }
    let new_chart = |p: &mut Parsed, fig: Option<Fig>, grid: Option<usize>, in_panel: bool| -> usize {
        p.charts.push(Chart { fig, ..Chart::default() });
        let k = p.charts.len() - 1;
        place(p, grid, in_panel, k);
        k
    };

    for line in text.lines() {
        let Some((verb, rest)) = line.split_once(':') else { continue };
        if verb.is_empty() || !verb.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
            continue;
        }
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        let f: Vec<&str> = rest.split(" ; ").collect();
        match verb {
            "grid" => {
                let v = nums(f.get(1).copied().unwrap_or(""));
                let (rows, cols) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                if rows >= 1.0 && cols >= 1.0 && rows * cols <= 64.0 {
                    p.entries.push(Entry::Grid { title: f[0].trim().to_string(), rows: rows as usize, cols: cols as usize, share: String::new(), panels: Vec::new() });
                    grid = Some(p.entries.len() - 1);
                    in_panel = false;
                    cur = None;
                }
            }
            "share" => {
                if let Some(Entry::Grid { share, .. }) = grid.map(|g| &mut p.entries[g]) {
                    *share = f[0].trim().to_string();
                }
            }
            "figure" => {
                let mut fig = Fig {
                    title: f.first().unwrap_or(&"").trim().to_string(),
                    xlabel: f.get(1).unwrap_or(&"").trim().to_string(),
                    ylabel: f.get(2).unwrap_or(&"").trim().to_string(),
                    ..Fig::default()
                };
                fig.token = f.get(4).map(|t| t.trim().to_string()).unwrap_or_default();
                if let Some(w) = f.get(3) {
                    let w = nums(w);
                    if w.len() == 4 && w.iter().all(|v| v.is_finite()) {
                        if w[0] != w[1] {
                            fig.xlim = Some((w[0], w[1]));
                        }
                        if w[2] != w[3] {
                            fig.ylim = Some((w[2], w[3]));
                        }
                    }
                }
                // inside a grid with a panel left, a figure opens the next panel
                in_panel = false;
                if let Some(Entry::Grid { rows, cols, panels, .. }) = grid.map(|g| &mut p.entries[g]) {
                    if panels.len() < *rows * *cols {
                        panels.push(Vec::new());
                        in_panel = true;
                    }
                }
                if !in_panel {
                    grid = None;
                }
                cur = Some(new_chart(&mut p, Some(fig), grid, in_panel));
                in_anim = false;
            }
            "xscale" | "yscale" | "aspect" | "style" | "yright" | "origin" | "colormap" | "clim" => {
                let Some(fig) = cur.and_then(|k| p.charts[k].fig.as_mut()) else { continue };
                match verb {
                    "xscale" => fig.xlog = f[0].trim() == "log",
                    "yscale" => fig.ylog = f[0].trim() == "log",
                    "aspect" => fig.equal = f[0].trim() == "equal",
                    "style" => fig.styles.push((f[0].to_string(), parse_style(f.get(1).unwrap_or(&"")))),
                    "origin" => fig.origin_lower = f[0].trim() == "lower",
                    "colormap" => fig.colormap = f[0].trim().to_string(),
                    "clim" => {
                        let v = nums(f[0]);
                        if v.len() == 2 && v.iter().all(|x| x.is_finite()) && v[0] != v[1] {
                            fig.clim = Some((v[0], v[1]));
                        }
                    }
                    _ => {
                        let label = f.get(1).unwrap_or(&"").trim().to_string();
                        let r = fig.yright.get_or_insert((String::new(), Vec::new()));
                        r.0 = label;
                        if !r.1.iter().any(|n| n == f[0]) {
                            r.1.push(f[0].to_string());
                        }
                    }
                }
            }
            "hline" | "vline" | "xtick" | "ytick" | "label" | "rect" | "segment" if !in_anim => {
                if let (Some(k), Some(m)) = (cur, parse_mark(verb, &f)) {
                    p.charts[k].marks.push(m);
                }
            }
            "box" | "arrow" | "note" if !in_anim => {
                if let (Some(k), Some(m)) = (cur, parse_dmark(verb, &f)) {
                    p.charts[k].diagram.push(m);
                }
            }
            "stage" => in_anim = true,
            "plot" | "bars" | "scatter" | "fill_between" | "rangebars" => {
                let s = match verb {
                    "scatter" => {
                        let (Some(xs), Some(ys)) = (f.get(2).and_then(|a| arr(a)), f.get(3).and_then(|a| arr(a))) else { continue };
                        let tok: Vec<&str> = f.get(1).unwrap_or(&"").split_whitespace().collect();
                        Series { name: f[0].trim().to_string(), kind: Kind::Points, xs, ys, lo: vec![], hi: vec![], color: tok.first().unwrap_or(&"").to_string(), open: tok.get(1) == Some(&"o") }
                    }
                    "fill_between" | "rangebars" => {
                        let (Some(xs), Some(lo), Some(hi)) = (f.get(1).and_then(|a| arr(a)), f.get(2).and_then(|a| arr(a)), f.get(3).and_then(|a| arr(a))) else { continue };
                        if lo.len() != xs.len() || hi.len() != xs.len() {
                            continue;
                        }
                        let kind = if verb == "fill_between" { Kind::Fill } else { Kind::Range };
                        Series { name: f[0].trim().to_string(), kind, ys: vec![f64::NAN; xs.len()], xs, lo, hi, color: String::new(), open: false }
                    }
                    _ => {
                        let (Some(xs), Some(ys)) = (f.get(1).and_then(|a| arr(a)), f.get(2).and_then(|a| arr(a))) else { continue };
                        let kind = if verb == "bars" { Kind::Bars } else { Kind::Line };
                        Series { name: f[0].trim().to_string(), kind, xs, ys, lo: vec![], hi: vec![], color: String::new(), open: false }
                    }
                };
                if s.xs.is_empty() || s.xs.len() != s.ys.len() {
                    continue;
                }
                // with no figure open, each record is its own chart
                let k = match cur {
                    Some(k) => k,
                    None => new_chart(&mut p, None, grid, in_panel),
                };
                p.charts[k].series.push(s);
            }
            "plot2d" if !in_anim && cur.is_some_and(|k| p.charts[k].heat.is_none() && p.charts[k].series.is_empty()) && heat_of(&f).is_some() => {
                let k = cur.unwrap_or_default();
                p.charts[k].heat = heat_of(&f);
            }
            "plot2d" | "frames2d" | "plot3d" | "quiver" | "scene" | "wire" | "verts" => {
                let what = match verb {
                    "plot2d" | "frames2d" => "a heatmap",
                    "plot3d" => "a 3-D curve",
                    "quiver" => "a vector field",
                    _ => "a 3-D scene",
                };
                let k = match cur {
                    Some(k) => k,
                    None => new_chart(&mut p, None, grid, in_panel),
                };
                if !p.charts[k].undrawn.contains(&what) {
                    p.charts[k].undrawn.push(what);
                }
            }
            _ => {}
        }
    }
    p
}

// ---- axes and ticks (www/src/lib/axisScale.js, line for line) ---------------

fn tx(log: bool, v: f64) -> f64 {
    if log {
        if v > 0.0 { vmath::log10(v) } else { f64::NAN }
    } else {
        v
    }
}

fn pow10(k: f64) -> f64 {
    vmath::pow(10.0, k)
}

fn nice_ticks(a: f64, b: f64, n: usize) -> (Vec<f64>, f64) {
    let span = if b - a != 0.0 { b - a } else { 1.0 };
    let step0 = span / (n.max(1) as f64);
    let mag = pow10(vmath::log10(step0).floor());
    let norm = step0 / mag;
    let step = (if norm >= 5.0 { 10.0 } else if norm >= 2.0 { 5.0 } else if norm >= 1.0 { 2.0 } else { 1.0 }) * mag;
    let mut out = Vec::new();
    let mut t = (a / step - 1e-9).ceil() * step;
    while t <= b + step * 1e-9 && out.len() < 1000 {
        out.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        t += step;
    }
    (out, step)
}

/// A tick label for ticks `step` apart (axisScale.js tickFormat).
fn tick_label(v: f64, step: f64) -> String {
    let decimals = (-(vmath::log10(step) + 1e-9).floor()).clamp(0.0, 8.0) as usize;
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if !(1e-4..1e5).contains(&a) {
        return format!("{v:.1e}").replace(".0e", "e");
    }
    format!("{v:.decimals$}").replace('-', "\u{2212}")
}

fn sup(k: i64) -> String {
    k.to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            _ => '⁹',
        })
        .collect()
}

fn num_label(d: i64, k: i64) -> String {
    if !(-4..5).contains(&k) {
        return format!("{d}e{k}");
    }
    if k >= 0 {
        return (d * 10i64.pow(k as u32)).to_string();
    }
    let digits = (-k) as usize;
    format!("{:.digits$}", d as f64 / pow10(-k as f64))
}

/// Ticks of an axis over the t-space range [lo, hi]: ([(t, label)], [minor t]).
fn axis_ticks(log: bool, lo: f64, hi: f64, n: usize) -> (Vec<(f64, String)>, Vec<f64>) {
    if !log {
        let (ticks, step) = nice_ticks(lo, hi, n);
        return (ticks.into_iter().map(|t| (t, tick_label(t, step))).collect(), Vec::new());
    }
    let k0 = (lo - 1e-9).ceil() as i64;
    let k1 = (hi + 1e-9).floor() as i64;
    if k1 - k0 + 1 >= 2 {
        let m = ((k1 - k0 + 1) as f64 / n.max(1) as f64).ceil().max(1.0) as i64;
        let mut ticks = Vec::new();
        let mut k = (k0 as f64 / m as f64).ceil() as i64 * m;
        while k <= k1 {
            ticks.push((k as f64, format!("10{}", sup(k))));
            k += m;
        }
        let mut minor = Vec::new();
        if m == 1 {
            for k in (lo.floor() as i64)..=k1 {
                for d in 2..=9 {
                    let t = k as f64 + vmath::log10(d as f64);
                    if t >= lo && t <= hi {
                        minor.push(t);
                    }
                }
            }
        }
        return (ticks, minor);
    }
    let within = |digits: &[i64]| {
        let mut out = Vec::new();
        for k in (lo.floor() as i64)..=(hi.ceil() as i64) {
            for &d in digits {
                let t = k as f64 + vmath::log10(d as f64);
                if t >= lo - 1e-9 && t <= hi + 1e-9 {
                    out.push((t, num_label(d, k)));
                }
            }
        }
        out
    };
    let mut ticks = within(&[1, 2, 5]);
    if ticks.len() < 3 {
        ticks = within(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
    (ticks, Vec::new())
}

/// aspect("equal") (axisScale.js equalAspect).
fn equal_aspect(r: (f64, f64, f64, f64), pw: f64, min_h: f64, max_h: f64) -> (f64, f64, f64, f64, f64) {
    let (mut xlo, mut xhi, mut ylo, mut yhi) = r;
    let xs = if xhi - xlo != 0.0 { xhi - xlo } else { 1.0 };
    let ys = if yhi - ylo != 0.0 { yhi - ylo } else { 1.0 };
    let mut ph = pw * ys / xs;
    if ph > max_h {
        ph = max_h;
        let (c, half) = ((xlo + xhi) / 2.0, ys * pw / max_h / 2.0);
        xlo = c - half;
        xhi = c + half;
    } else if ph < min_h {
        ph = min_h;
        let (c, half) = ((ylo + yhi) / 2.0, xs * min_h / pw / 2.0);
        ylo = c - half;
        yhi = c + half;
    }
    (xlo, xhi, ylo, yhi, ph)
}

// ---- drawing ------------------------------------------------------------------

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A number as the SVG writes it: two decimals, and never `-0.00`.
fn c(v: f64) -> String {
    let s = format!("{v:.2}");
    if s == "-0.00" { "0.00".to_string() } else { s }
}

fn clamp(v: f64) -> f64 {
    v.clamp(-1e4, 1e4)
}

fn marker_svg(out: &mut String, kind: &str, x: f64, y: f64, color: &str) {
    let r = 3.0;
    match kind {
        "dot" => { let _ = write!(out, r#"<circle cx="{}" cy="{}" r="2.6" fill="{color}" fill-opacity="0.85"/>"#, c(x), c(y)); }
        "ring" => { let _ = write!(out, r#"<circle cx="{}" cy="{}" r="3.0" fill="none" stroke="{color}" stroke-width="1.3"/>"#, c(x), c(y)); }
        "square" => {
            let a = r * 0.9;
            let _ = write!(out, r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{color}"/>"#, c(x - a), c(y - a), c(2.0 * a), c(2.0 * a));
        }
        "cross" => {
            let a = r * 0.95;
            let _ = write!(out, r#"<path d="M{} {}L{} {}M{} {}L{} {}" stroke="{color}" stroke-width="1.5" stroke-linecap="round" fill="none"/>"#,
                c(x - a), c(y - a), c(x + a), c(y + a), c(x - a), c(y + a), c(x + a), c(y - a));
        }
        _ => { let _ = write!(out, r#"<circle cx="{}" cy="{}" r="3.0" fill="{color}"/>"#, c(x), c(y)); }
    }
}

/// Each series' palette slot (seriesStyle.jsx paletteSlots): in order of first
/// appearance, one per label, skipping slots a style claims; None for a
/// series whose style sets its color.
fn palette_slots(series: &[Series], fig: &Fig) -> Vec<Option<usize>> {
    let claimed: Vec<usize> = fig.styles.iter().filter_map(|(_, s)| s.color).collect();
    let mut slots = Vec::new();
    let mut by_name: Vec<(String, Option<usize>)> = Vec::new();
    let mut next = 0usize;
    for s in series {
        if !s.name.is_empty() {
            if let Some((_, k)) = by_name.iter().find(|(n, _)| *n == s.name) {
                slots.push(*k);
                continue;
            }
        }
        let k = if fig.style(&s.name).is_some_and(|st| st.color.is_some()) {
            None
        } else {
            let mut tries = 0;
            while claimed.contains(&(next % PALETTE.len())) && tries < PALETTE.len() {
                next += 1;
                tries += 1;
            }
            next += 1;
            Some(next - 1)
        };
        if !s.name.is_empty() {
            by_name.push((s.name.clone(), k));
        }
        slots.push(k);
    }
    slots
}

enum Swatch {
    Line { line: bool, dash: Option<&'static str>, width: f64, marker: Option<String> },
    Bars,
    Fill,
    Range,
}

/// A semibold title is this much wider than the regular text sans_width measures;
/// a wrapped title's lines are this far apart.
const TITLE_BOLD: f64 = 1.06;
const TITLE_LINE: f64 = 18.0;

/// How much wider than its measured width a legend label may be drawn
/// (`frame`): 15 px on the home page for 12 here, in the site's own face.
const LEGEND_ROOM: f64 = 1.3;

/// The card around a chart: its title and legend above the plot, as on the site.
fn frame(width: f64, height: f64, body: &str, title: &str, legend: &[(String, String, Swatch)]) -> (String, f64) {
    let mut head = String::new();
    let mut top = 0.0;
    if !title.is_empty() {
        let l = lab(title, 14.0);
        let room = width - 28.0;
        if !l.math && l.w * TITLE_BOLD > room {
            // a title too long for the card wraps at spaces, as the site's HTML title does
            let mut lines: Vec<String> = Vec::new();
            for word in title.split(' ') {
                match lines.last_mut() {
                    Some(last) if mathtext::sans_width(&format!("{last} {word}"), 14.0) * TITLE_BOLD <= room => {
                        last.push(' ');
                        last.push_str(word);
                    }
                    _ => lines.push(word.to_string()),
                }
            }
            for (k, line) in lines.iter().enumerate() {
                let _ = write!(head, r#"<text x="14" y="{}" font-size="14" font-weight="600" fill="{TITLE}">{}</text>"#, c(22.0 + TITLE_LINE * k as f64), lab(line, 14.0).inner);
            }
            top = 30.0 + TITLE_LINE * (lines.len() as f64 - 1.0);
        }
        else {
            let _ = write!(head, r#"<text x="14" y="22" font-size="14" font-weight="600" fill="{TITLE}">{}</text>{}"#, l.inner, l.shapes_svg(14.0, 22.0, Anchor::Start, 0.0, TITLE));
            top = 30.0;
        }
    }
    if !legend.is_empty() {
        let mut x = 14.0;
        let mut y = top + 18.0;
        for (name, color, sw) in legend {
            let l = lab(name, 12.0);
            // the label's measured width with LEGEND_ROOM to spare: a page may set
            // the text in a wider face or raise its size (the home page's figures,
            // www/src/pages/index.astro .tab-fig, take 12 to 15 px), and the next
            // swatch must still clear the label
            let w = 17.0 + LEGEND_ROOM * if l.math { l.w } else { mathtext::sans_width(name, 12.0) };
            if x + w > width - 10.0 && x > 14.0 {
                x = 14.0;
                y += 18.0;
            }
            match sw {
                Swatch::Bars => { let _ = write!(head, r#"<rect x="{}" y="{}" width="11" height="10" rx="{BAR_RX}" fill="{color}" fill-opacity="{BAR_OPACITY}"/>"#, c(x), c(y - 9.0)); }
                Swatch::Fill => { let _ = write!(head, r#"<rect x="{}" y="{}" width="11" height="10" fill="{color}" fill-opacity="0.2"/>"#, c(x), c(y - 9.0)); }
                Swatch::Range => {
                    let _ = write!(head, r#"<path d="M{} {}H{}M{} {}V{}M{} {}H{}" stroke="{color}" stroke-width="1.3" fill="none"/>"#,
                        c(x + 2.0), c(y - 10.0), c(x + 9.0), c(x + 5.5), c(y - 10.0), c(y), c(x + 2.0), c(y), c(x + 9.0));
                }
                Swatch::Line { line, dash, width, marker } => {
                    if *line {
                        let dash_attr = dash.map(|d| format!(r#" stroke-dasharray="{d}""#)).unwrap_or_default();
                        let _ = write!(head, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{color}" stroke-width="{}"{dash_attr}/>"#,
                            c(x), c(x + 14.0), c(y - 4.0), c(y - 4.0), c(width.max(2.0)));
                    }
                    if let Some(m) = marker {
                        marker_svg(&mut head, if m == "dot" { "circle" } else { m }, x + 7.0, y - 4.0, color);
                    }
                }
            }
            let _ = write!(head, r#"<text x="{}" y="{}" font-size="12" fill="{LABEL}">{}</text>{}"#, c(x + 17.0), c(y), l.inner, l.shapes_svg(x + 17.0, y, Anchor::Start, 0.0, LABEL));
            x += w + 14.0;
        }
        top = y + 8.0;
    }
    let total = top + height;
    let mut svg = String::new();
    let _ = write!(svg, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" font-family="{}" role="img" aria-label="{}">"#,
        c(width), c(total), c(width), c(total), esc(FONT), esc(if title.is_empty() { "figure" } else { title }));
    let _ = write!(svg, r#"<rect x="0.5" y="0.5" width="{}" height="{}" rx="8" fill="{BG}" stroke="{BORDER}"/>"#, c(width - 1.0), c(total - 1.0));
    svg.push_str(&head);
    let _ = write!(svg, r#"<g transform="translate(0 {})">{body}</g></svg>"#, c(top));
    (svg, total)
}

fn finite(v: f64) -> bool {
    v.is_finite()
}

// ---- tick labels and bar groups (www/src/lib/axisScale.js does the same) ----------

/// Tick label text size, and the least room between two flat x tick labels.
const TICK_SIZE: f64 = 11.0;
const TICK_PAD: f64 = 6.0;
/// x tick labels that do not fit side by side are turned this many degrees ...
const TICK_TURN_DEG: f64 = 40.0;
/// ... and kept this far apart, across their baselines.
const TURN_ROW: f64 = 13.0;
/// Where a turned label's baseline ends, from its tick at the bottom of the plot.
const TURN_DX: f64 = 4.0;
const TURN_DY: f64 = 14.0;
/// How far below the plot a flat tick label reaches (baseline 17, descent 2).
const FLAT_DROP: f64 = 19.0;
/// The least spacing of the program's own y ticks, and the tallest plot that buys it.
const YTICK_ROW: f64 = 14.0;
const MAX_PLOT_H: f64 = 900.0;
/// A bar group fills this part of the space between categories; a bar in a
/// group, this part of its slot.
const GROUP_FILL: f64 = 0.8;
const BAR_FILL: f64 = 0.88;
/// A lone bar series: its bars fill this part of the space between them.
const SOLO_FILL: f64 = 0.86;

/// How the x tick labels are set: `turn`, turned TICK_TURN_DEG degrees, and
/// which of them are drawn (`show`).
struct XTickLayout {
    turn: bool,
    show: Vec<bool>,
}

/// Lays out x tick labels at pixel positions `px` with text widths `w`. Flat
/// when every neighbor pair fits; otherwise a category axis (the program's own
/// ticks) turns its labels, and a number axis drops every other label (then
/// two of three ...) until the rest fit.
fn x_tick_layout(px: &[f64], w: &[f64], categories: bool) -> XTickLayout {
    let n = px.len();
    let mut ord: Vec<usize> = (0..n).filter(|&i| finite(px[i])).collect();
    ord.sort_by(|a, b| px[*a].total_cmp(&px[*b]));
    let picked = |step: usize| -> Vec<usize> { ord.iter().copied().step_by(step).collect() };
    let flat_fits = |idx: &[usize]| idx.windows(2).all(|p| px[p[1]] - px[p[0]] >= (w[p[0]] + w[p[1]]) / 2.0 + TICK_PAD);
    let turn_fits = |idx: &[usize]| idx.windows(2).all(|p| (px[p[1]] - px[p[0]]) * vmath::sin(TICK_TURN_DEG.to_radians()) >= TURN_ROW);
    let shown = |idx: &[usize]| -> Vec<bool> { (0..n).map(|i| idx.contains(&i)).collect() };
    if flat_fits(&ord) {
        return XTickLayout { turn: false, show: shown(&ord) };
    }
    for step in 1..=ord.len().max(1) {
        let idx = picked(step);
        if categories && turn_fits(&idx) {
            return XTickLayout { turn: true, show: shown(&idx) };
        }
        if !categories && step > 1 && flat_fits(&idx) {
            return XTickLayout { turn: false, show: shown(&idx) };
        }
    }
    XTickLayout { turn: categories, show: shown(&picked(ord.len().max(1))) }
}

/// The extra room under the plot that turned labels of these widths need,
/// past a flat label's.
fn turned_drop(w: &[f64]) -> f64 {
    let (s, co) = (vmath::sin(TICK_TURN_DEG.to_radians()), vmath::cos(TICK_TURN_DEG.to_radians()));
    let widest = w.iter().copied().fold(0.0, f64::max);
    (TURN_DY + widest * s + 0.2 * TICK_SIZE * co - FLAT_DROP).max(0.0)
}

/// How far a turned label reaches left of its tick.
fn turned_reach(w: f64) -> f64 {
    w * vmath::cos(TICK_TURN_DEG.to_radians()) - TURN_DX
}

/// Where each bar series' bars stand, in pixels: (offset from its x, width).
/// Bar series that share an x stand side by side, a group centered on the x;
/// `px_per_x` turns x units into pixels.
fn bar_slots(series: &[Series], px_per_x: f64) -> Vec<(f64, f64)> {
    let bars: Vec<usize> = (0..series.len()).filter(|&k| series[k].kind == Kind::Bars).collect();
    let min_gap = |xs: &mut Vec<f64>| {
        xs.sort_by(|a, b| a.total_cmp(b));
        xs.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0).fold(f64::INFINITY, f64::min)
    };
    let fin_xs = |k: usize| -> Vec<f64> { series[k].xs.iter().copied().filter(|v| finite(*v)).collect() };
    let shared = bars.len() > 1 && bars.iter().enumerate().any(|(j, &a)| bars[j + 1..].iter().any(|&b| fin_xs(a).iter().any(|x| series[b].xs.contains(x))));
    let mut out = vec![(0.0, 0.0); series.len()];
    if shared {
        let mut all: Vec<f64> = bars.iter().flat_map(|&k| fin_xs(k)).collect();
        let gap = min_gap(&mut all);
        let group = if gap.is_finite() { gap * px_per_x * GROUP_FILL } else { 12.0 * bars.len() as f64 };
        let slot = group / bars.len() as f64;
        for (j, &k) in bars.iter().enumerate() {
            out[k] = ((j as f64 - (bars.len() as f64 - 1.0) / 2.0) * slot, (slot * BAR_FILL).max(1.5));
        }
        return out;
    }
    // bars that do not share an x: each as wide as the closest two of all of them allow
    let mut all: Vec<f64> = bars.iter().flat_map(|&k| fin_xs(k)).collect();
    let union_gap = min_gap(&mut all);
    for &k in &bars {
        let gap = if bars.len() > 1 { union_gap } else { min_gap(&mut fin_xs(k)) };
        out[k] = (0.0, if gap.is_finite() { (gap * px_per_x * SOLO_FILL).max(2.0) } else { 12.0 });
    }
    out
}

/// One chart as SVG at the given width; returns the SVG and its height.
fn render_chart(ch: &Chart, width: f64, id: usize) -> (String, f64) {
    let fig = ch.fig.clone().unwrap_or_default();
    let (xlog, ylog) = (fig.xlog, fig.ylog);
    // everything below is in t-space: log10 of the value on a log axis
    let series: Vec<Series> = ch
        .series
        .iter()
        .map(|s| {
            let right = fig.right(&s.name);
            let ty = |v: &f64| if right { *v } else { tx(ylog, *v) };
            Series {
                xs: s.xs.iter().map(|v| tx(xlog, *v)).collect(),
                ys: s.ys.iter().map(ty).collect(),
                lo: s.lo.iter().map(ty).collect(),
                hi: s.hi.iter().map(ty).collect(),
                ..s.clone()
            }
        })
        .collect();
    let right: Vec<bool> = series.iter().map(|s| fig.right(&s.name)).collect();
    let has_right = right.iter().any(|r| *r);
    let yvals = |s: &Series| -> Vec<f64> { if matches!(s.kind, Kind::Fill | Kind::Range) { s.lo.iter().chain(s.hi.iter()).copied().collect() } else { s.ys.clone() } };
    // what a log axis cannot show is said under the chart
    let mut hidden = Vec::new();
    let count = |vals: &mut dyn Iterator<Item = f64>| vals.filter(|v| finite(*v) && *v <= 0.0).count();
    if xlog {
        let n = count(&mut ch.series.iter().flat_map(|s| s.xs.iter().copied()));
        if n > 0 {
            hidden.push(format!("{n} x value{} ≤ 0", if n > 1 { "s" } else { "" }));
        }
    }
    if ylog {
        let n = count(&mut ch.series.iter().filter(|s| !fig.right(&s.name)).flat_map(|s| {
            if matches!(s.kind, Kind::Fill | Kind::Range) { s.lo.iter().chain(s.hi.iter()).copied().collect::<Vec<_>>() } else { s.ys.clone() }
        }));
        if n > 0 {
            hidden.push(format!("{n} y value{} ≤ 0", if n > 1 { "s" } else { "" }));
        }
    }
    let marks: Vec<Mark> = ch
        .marks
        .iter()
        .map(|m| match m {
            Mark::HLine(y) => Mark::HLine(tx(ylog, *y)),
            Mark::VLine(x) => Mark::VLine(tx(xlog, *x)),
            Mark::XTick(t, x) => Mark::XTick(t.clone(), tx(xlog, *x)),
            Mark::YTick(t, y) => Mark::YTick(t.clone(), tx(ylog, *y)),
            Mark::Label(t, x, y) => Mark::Label(t.clone(), tx(xlog, *x), tx(ylog, *y)),
            Mark::Segment(a, b, cc, d) => Mark::Segment(tx(xlog, *a), tx(ylog, *b), tx(xlog, *cc), tx(ylog, *d)),
            Mark::Rect(x, y, w, h) => {
                let (x0, x1) = (tx(xlog, x - w / 2.0), tx(xlog, x + w / 2.0));
                let (y0, y1) = (tx(ylog, y - h / 2.0), tx(ylog, y + h / 2.0));
                Mark::Rect((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0).abs(), (y1 - y0).abs())
            }
        })
        .collect();
    let (mut mx, mut my) = (Vec::new(), Vec::new());
    for m in &marks {
        match m {
            Mark::HLine(y) | Mark::YTick(_, y) => my.push(*y),
            Mark::VLine(x) | Mark::XTick(_, x) => mx.push(*x),
            Mark::Label(_, x, y) => {
                mx.push(*x);
                my.push(*y);
            }
            Mark::Rect(x, y, w, h) => {
                mx.extend([x - w / 2.0, x + w / 2.0]);
                my.extend([y - h / 2.0, y + h / 2.0]);
            }
            Mark::Segment(a, b, cc, d) => {
                mx.extend([*a, *cc]);
                my.extend([*b, *d]);
            }
        }
    }
    let lim = |l: Option<(f64, f64)>, log: bool| l.map(|(a, b)| (tx(log, a), tx(log, b))).filter(|(a, b)| finite(*a) && finite(*b));
    let (xlim, ylim) = (lim(fig.xlim, xlog), lim(fig.ylim, ylog));
    let extent = |vals: &mut dyn Iterator<Item = f64>| -> Option<(f64, f64)> {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in vals.filter(|v| finite(*v)) {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo.is_finite() { Some((lo, hi)) } else { None }
    };
    let has_bars = series.iter().any(|s| s.kind == Kind::Bars);
    let has_points = series.iter().any(|s| s.kind == Kind::Points);
    let (mut xlo, mut xhi) = match xlim {
        Some(l) => l,
        None => {
            let (a, b) = extent(&mut series.iter().flat_map(|s| s.xs.iter().copied()).chain(mx.iter().copied())).unwrap_or((0.0, 1.0));
            if has_points {
                let pad = (if b - a != 0.0 { b - a } else if b != 0.0 { b.abs() } else { 1.0 }) * 0.06;
                (a - pad, b + pad)
            } else {
                (a, b)
            }
        }
    };
    if has_bars && xlim.is_none() {
        let mut xs0: Vec<f64> = series.iter().find(|s| s.kind == Kind::Bars).map(|s| s.xs.iter().copied().filter(|v| finite(*v)).collect()).unwrap_or_default();
        xs0.sort_by(|a, b| a.total_cmp(b));
        let gap = xs0.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0).fold(f64::INFINITY, f64::min);
        if gap.is_finite() {
            xlo -= gap / 2.0;
            xhi += gap / 2.0;
        }
    }
    let (mut ylo, mut yhi) = match ylim {
        Some(l) => l,
        None => {
            let left: Vec<f64> = series.iter().zip(&right).filter(|(_, r)| !**r).flat_map(|(s, _)| yvals(s)).chain(my.iter().copied()).collect();
            let (mut a, b) = extent(&mut left.into_iter()).unwrap_or((0.0, 1.0));
            let on_zero = has_bars && !ylog;
            if on_zero && a > 0.0 {
                a = 0.0;
            }
            let pad = (if b - a != 0.0 { b - a } else if b != 0.0 { b.abs() } else { 1.0 }) * 0.07;
            (if on_zero && a == 0.0 { a } else { a - pad }, b + pad)
        }
    };
    let (mut y2lo, mut y2hi) = (0.0, 1.0);
    if has_right {
        let vals: Vec<f64> = series.iter().zip(&right).filter(|(_, r)| **r).flat_map(|(s, _)| yvals(s)).collect();
        let (a, b) = extent(&mut vals.into_iter()).unwrap_or((0.0, 1.0));
        let pad = (if b - a != 0.0 { b - a } else if b != 0.0 { b.abs() } else { 1.0 }) * 0.07;
        (y2lo, y2hi) = (a - pad, b + pad);
    }
    if fig.equal {
        let pw0 = width - 80.0;
        let r = equal_aspect((xlo, xhi, ylo, yhi), pw0, 160.0, pw0.min(520.0));
        (xlo, xhi, ylo, yhi) = (r.0, r.1, r.2, r.3);
    }

    let yticks_prog: Vec<(f64, String)> = marks.iter().filter_map(|m| if let Mark::YTick(t, y) = m { Some((*y, t.clone())) } else { None }).collect();
    let (yt, yminor) = if yticks_prog.is_empty() { axis_ticks(ylog, ylo, yhi, 5) } else { (yticks_prog.clone(), Vec::new()) };
    let xticks_prog: Vec<(f64, String)> = marks.iter().filter_map(|m| if let Mark::XTick(t, x) = m { Some((*x, t.clone())) } else { None }).collect();
    let (xt, xminor) = if xticks_prog.is_empty() { axis_ticks(xlog, xlo, xhi, if width < 420.0 { 4 } else { 7 }) } else { (xticks_prog.clone(), Vec::new()) };
    let y2t = if has_right { axis_ticks(false, y2lo, y2hi, 5).0 } else { Vec::new() };
    let widest = yt.iter().map(|(_, l)| label_chars(l, 11.0)).fold(1.0, f64::max);
    let widest2 = y2t.iter().map(|(_, l)| label_chars(l, 11.0)).fold(1.0, f64::max);
    // a tall math label moves out from the plot
    let xl = lab(&fig.xlabel, 11.5);
    let yl = lab(&fig.ylabel, 11.5);
    let below = if xl.math { (xl.d - 3.0).max(0.0) } else { 0.0 };
    let xl_room = below + if xl.math { (xl.h - 9.0).max(0.0) } else { 0.0 };
    let yl_room = if yl.math { (yl.h - 9.0).max(0.0) } else { 0.0 };
    let room = 16.0 + 8.0 * 6.4 + 18.0;
    let (inner_x, inner_y) = fig.inner;
    let mut ml = if inner_y { 12.0 } else { (16.0 + widest * 6.4 + if fig.ylabel.is_empty() { 0.0 } else { 18.0 + yl_room }).max(if fig.in_grid { room } else { 0.0 }) };
    let mr = (if has_right { 16.0 + widest2 * 6.4 + 18.0 } else { 14.0 }).max(if fig.in_grid && fig.room_right { room } else { 0.0 });
    let mt = 10.0;
    let mut mb = if inner_x { 10.0 } else if fig.xlabel.is_empty() { 26.0 } else { 44.0 + xl_room };
    let mut pw = (width - ml - mr).max(40.0);
    // x tick labels that would run into each other are turned (categories) or thinned (numbers)
    let xt_w: Vec<f64> = xt.iter().map(|(_, l)| lab(l, TICK_SIZE).w).collect();
    let xt_px = |ml: f64, pw: f64| -> Vec<f64> { xt.iter().map(|(t, _)| ml + (t - xlo) / if xhi - xlo != 0.0 { xhi - xlo } else { 1.0 } * pw).collect() };
    let xlay = x_tick_layout(&xt_px(ml, pw), &xt_w, !xticks_prog.is_empty());
    if xlay.turn && !inner_x {
        let shown_w: Vec<f64> = xt_w.iter().zip(&xlay.show).filter(|(_, s)| **s).map(|(w, _)| *w).collect();
        mb += turned_drop(&shown_w);
        // the first labels reach left past the plot: the margin grows to hold them
        let over = xt_px(ml, pw).iter().zip(&xt_w).zip(&xlay.show).filter(|(_, s)| **s).map(|((x, w), _)| turned_reach(*w) - (x - 4.0)).fold(0.0, f64::max);
        if over > 0.0 {
            ml += over;
            pw = (width - ml - mr).max(40.0);
        }
    }
    let square = !fig.equal && xlim.is_some() && ylim.is_some() && ((xhi - xlo) - (yhi - ylo)).abs() < 1e-9 * (if xhi - xlo != 0.0 { (xhi - xlo).abs() } else { 1.0 });
    let mut ph = if square { pw.min(420.0) } else { (pw * 0.52).clamp(200.0, 340.0).round() };
    if fig.equal {
        ph = (pw * (yhi - ylo) / if xhi - xlo != 0.0 { xhi - xlo } else { 1.0 }).round();
    }
    else if !square && yticks_prog.len() > 1 {
        // the program's own y ticks (categories) get a row each
        let mut ys: Vec<f64> = yticks_prog.iter().map(|(y, _)| *y).filter(|v| finite(*v)).collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        let gap = ys.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0).fold(f64::INFINITY, f64::min);
        if gap.is_finite() {
            ph = ph.max((YTICK_ROW * (yhi - ylo).abs() / gap).round()).min(MAX_PLOT_H.max(ph));
        }
    }
    let mut h = ph + mt + mb;
    let span = |a: f64, b: f64| if b - a != 0.0 { b - a } else { 1.0 };
    let xmap = |v: f64| ml + (v - xlo) / span(xlo, xhi) * pw;
    let ymap = |v: f64| mt + (1.0 - (v - ylo) / span(ylo, yhi)) * ph;
    let y2map = |v: f64| mt + (1.0 - (v - y2lo) / span(y2lo, y2hi)) * ph;
    let yof = |k: usize, v: f64| if right[k] { y2map(v) } else { ymap(v) };

    let mut g = String::new();
    for (t, label) in &yt {
        let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{GRID}"/>"#, c(ml), c(ml + pw), c(ymap(*t)), c(ymap(*t)));
        if !inner_y {
            let l = lab(label, 11.0);
            let _ = write!(g, r#"<text x="{}" y="{}" text-anchor="end" font-size="11" fill="{TICK}">{}</text>{}"#, c(ml - 8.0), c(ymap(*t) + 3.8), l.inner, l.shapes_svg(ml - 8.0, ymap(*t) + 3.8, Anchor::End, 0.0, TICK));
        }
    }
    for t in &yminor {
        let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#, c(ml - 3.0), c(ml), c(ymap(*t)), c(ymap(*t)));
    }
    let bottom = mt + ph;
    let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#, c(ml), c(ml + pw), c(bottom), c(bottom));
    for (k, (t, label)) in xt.iter().enumerate() {
        let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#, c(xmap(*t)), c(xmap(*t)), c(bottom), c(bottom + 4.0));
        if !inner_x && xlay.show[k] {
            let l = lab(label, TICK_SIZE);
            if xlay.turn {
                let (ax, ay) = (xmap(*t) + TURN_DX, bottom + TURN_DY);
                let _ = write!(g, r#"<text x="{}" y="{}" text-anchor="end" font-size="11" fill="{TICK}" transform="rotate({} {} {})">{}</text>{}"#,
                    c(ax), c(ay), c(-TICK_TURN_DEG), c(ax), c(ay), l.inner, l.shapes_svg(ax, ay, Anchor::End, -TICK_TURN_DEG, TICK));
            }
            else {
                let _ = write!(g, r#"<text x="{}" y="{}" text-anchor="middle" font-size="11" fill="{TICK}">{}</text>{}"#, c(xmap(*t)), c(bottom + 17.0), l.inner, l.shapes_svg(xmap(*t), bottom + 17.0, Anchor::Middle, 0.0, TICK));
            }
        }
    }
    for t in &xminor {
        let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#, c(xmap(*t)), c(xmap(*t)), c(bottom), c(bottom + 2.5));
    }
    if has_right {
        let xr = ml + pw;
        let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#, c(xr), c(xr), c(mt), c(bottom));
        for (t, label) in &y2t {
            let _ = write!(g, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#, c(xr), c(xr + 4.0), c(y2map(*t)), c(y2map(*t)));
            let l = lab(label, 11.0);
            let _ = write!(g, r#"<text x="{}" y="{}" font-size="11" fill="{TICK}">{}</text>{}"#, c(xr + 8.0), c(y2map(*t) + 3.8), l.inner, l.shapes_svg(xr + 8.0, y2map(*t) + 3.8, Anchor::Start, 0.0, TICK));
        }
        if let Some((label, _)) = &fig.yright {
            if !label.is_empty() {
                let cy = mt + ph / 2.0;
                let l = lab(label, 11.5);
                let _ = write!(g, r#"<text x="{}" y="{}" text-anchor="middle" font-size="11.5" font-weight="500" fill="{LABEL}" transform="rotate(90 {} {})">{}</text>{}"#,
                    c(width - 10.0), c(cy), c(width - 10.0), c(cy), l.inner, l.shapes_svg(width - 10.0, cy, Anchor::Middle, 90.0, LABEL));
            }
        }
    }
    if !fig.xlabel.is_empty() && !inner_x {
        let ly = h - 6.0 - below;
        let _ = write!(g, r#"<text x="{}" y="{}" text-anchor="middle" font-size="11.5" font-weight="500" fill="{LABEL}">{}</text>{}"#, c(ml + pw / 2.0), c(ly), xl.inner, xl.shapes_svg(ml + pw / 2.0, ly, Anchor::Middle, 0.0, LABEL));
    }
    if !fig.ylabel.is_empty() && !inner_y {
        let cy = mt + ph / 2.0;
        let lx = 12.0 + yl_room;
        let lxs = if yl_room == 0.0 { "12".to_string() } else { c(lx) };
        let _ = write!(g, r#"<text x="{lxs}" y="{}" text-anchor="middle" font-size="11.5" font-weight="500" fill="{LABEL}" transform="rotate(-90 {lxs} {})">{}</text>{}"#, c(cy), c(cy), yl.inner, yl.shapes_svg(lx, cy, Anchor::Middle, -90.0, LABEL));
    }
    let clip = format!("vp{id}");
    let _ = write!(g, r#"<defs><clipPath id="{clip}"><rect x="{}" y="{}" width="{}" height="{}"/></clipPath></defs>"#, c(ml), c(mt - 1.0), c(pw), c(ph + 2.0));

    let mut under = String::new(); // bands and shaded rects, under everything
    let mut body = String::new();
    for m in &marks {
        match m {
            Mark::Rect(x, y, w, hh) => {
                let (x0, x1, y0, y1) = (xmap(x - w / 2.0), xmap(x + w / 2.0), ymap(y + hh / 2.0), ymap(y - hh / 2.0));
                if [x0, x1, y0, y1].iter().all(|v| finite(*v)) {
                    let _ = write!(under, r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{MARK}" fill-opacity="0.14"/>"#, c(x0.min(x1)), c(y0.min(y1)), c((x1 - x0).abs()), c((y1 - y0).abs()));
                }
            }
            Mark::XTick(_, x) if finite(*x) => {
                let _ = write!(under, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{GRID}"/>"#, c(xmap(*x)), c(xmap(*x)), c(mt), c(bottom));
            }
            _ => {}
        }
    }
    let slots = palette_slots(&series, &fig);
    let slots_px = bar_slots(&series, pw / span(xlo, xhi));
    let mut legend: Vec<(String, String, Swatch)> = Vec::new();
    let sparse = series.iter().map(|s| s.xs.len()).max().unwrap_or(0) <= 24;
    for (k, s) in series.iter().enumerate() {
        let st = fig.style(&s.name).cloned().unwrap_or_default();
        let color = match (st.color, slots[k]) {
            (Some(slot), _) => PALETTE[slot % PALETTE.len()].to_string(),
            (None, slot) => match named_color(&s.color) {
                Some(n) => n.to_string(),
                None => PALETTE[slot.unwrap_or(0) % PALETTE.len()].to_string(),
            },
        };
        let line_word = st.line.clone().unwrap_or_else(|| if s.kind == Kind::Line { "solid".into() } else { "noline".into() });
        let dash = match line_word.as_str() {
            "dash" => Some("6 4"),
            "dot" => Some("1.5 3.5"),
            _ => None,
        };
        // the primary series: the first palette slot's color, a solid line
        let primary = color == PALETTE[0] && dash.is_none();
        let lw = match st.width.as_deref() {
            Some("thin") => 1.1,
            Some("thick") => 2.8,
            _ if primary => PRIMARY_WIDTH,
            _ => 1.8,
        };
        let marker: Option<String> = match st.marker.as_deref() {
            Some("nomarker") => None,
            Some(m) => Some(m.to_string()),
            None if s.kind == Kind::Points => Some(if s.open { "ring".into() } else { "dot".into() }),
            None if s.kind == Kind::Line && sparse => Some("circle".into()),
            None => None,
        };
        let interval = matches!(s.kind, Kind::Fill | Kind::Range);
        // one legend entry per label, drawn like the label's first series that is not an interval
        if !s.name.is_empty() && !legend.iter().any(|(n, _, _)| n.trim_end_matches(" (right axis)") == s.name) {
            let later_line = interval && series.iter().any(|t| t.name == s.name && !matches!(t.kind, Kind::Fill | Kind::Range));
            if !later_line {
                let sw = match s.kind {
                    Kind::Bars => Swatch::Bars,
                    Kind::Fill => Swatch::Fill,
                    Kind::Range => Swatch::Range,
                    _ => Swatch::Line { line: line_word != "noline", dash, width: lw, marker: marker.clone() },
                };
                let name = if right[k] { format!("{} (right axis)", s.name) } else { s.name.clone() };
                legend.push((name, color.clone(), sw));
            }
        }
        let ok = |i: usize| finite(s.xs[i]) && finite(s.lo[i]) && finite(s.hi[i]);
        match s.kind {
            Kind::Fill => {
                // the band from lo to hi; a gap splits it
                let mut d = String::new();
                let mut run: Vec<usize> = Vec::new();
                let flush = |run: &mut Vec<usize>, d: &mut String| {
                    if run.is_empty() {
                        return;
                    }
                    for (j, &i) in run.iter().enumerate() {
                        let _ = write!(d, "{}{} {}", if j == 0 { "M" } else { "L" }, c(xmap(s.xs[i])), c(clamp(yof(k, s.hi[i]))));
                    }
                    for &i in run.iter().rev() {
                        let _ = write!(d, "L{} {}", c(xmap(s.xs[i])), c(clamp(yof(k, s.lo[i]))));
                    }
                    d.push('Z');
                    run.clear();
                };
                for i in 0..s.xs.len() {
                    if ok(i) {
                        run.push(i);
                    } else {
                        flush(&mut run, &mut d);
                    }
                }
                flush(&mut run, &mut d);
                let _ = write!(under, r#"<path d="{d}" fill="{color}" fill-opacity="0.2" stroke="none"/>"#);
            }
            Kind::Range => {
                // a capped bar from lo to hi at each x
                let mut d = String::new();
                for i in 0..s.xs.len() {
                    if ok(i) {
                        let x = xmap(s.xs[i]);
                        let (a, b) = (clamp(yof(k, s.lo[i])), clamp(yof(k, s.hi[i])));
                        let _ = write!(d, "M{} {}V{}M{} {}H{}M{} {}H{}", c(x), c(a), c(b), c(x - 3.5), c(a), c(x + 3.5), c(x - 3.5), c(b), c(x + 3.5));
                    }
                }
                let _ = write!(body, r#"<path d="{d}" stroke="{color}" stroke-width="1.3" fill="none"/>"#);
            }
            Kind::Bars => {
                let (off, bw) = slots_px[k];
                // a bar stands on zero; on a log axis, on the bottom of the chart
                let y0 = if ylog { bottom } else { ymap(0.0f64.clamp(ylo.min(yhi), yhi.max(ylo))) };
                for (x, y) in s.xs.iter().zip(&s.ys) {
                    if finite(*x) && finite(*y) {
                        let yy = yof(k, *y);
                        let _ = write!(body, r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{BAR_RX}" fill="{color}" fill-opacity="{BAR_OPACITY}"/>"#,
                            c(xmap(*x) + off - bw / 2.0), c(y0.min(yy)), c(bw), c((y0 - yy).abs()));
                    }
                }
            }
            _ => {
                if line_word != "noline" {
                    // a nan or inf lifts the pen: the line has a gap there
                    let mut d = String::new();
                    let mut pen = false;
                    for (x, y) in s.xs.iter().zip(&s.ys) {
                        if !(finite(*x) && finite(*y)) {
                            pen = false;
                            continue;
                        }
                        let _ = write!(d, "{}{} {}", if pen { "L" } else { "M" }, c(xmap(*x)), c(clamp(yof(k, *y))));
                        pen = true;
                    }
                    let dash_attr = dash.map(|d| format!(r#" stroke-dasharray="{d}""#)).unwrap_or_default();
                    let _ = write!(body, r#"<path d="{d}"{} fill="none" stroke="{color}" stroke-width="{lw}"{dash_attr} stroke-linejoin="round" stroke-linecap="round"/>"#, if primary { r#" class="vp-primary""# } else { "" });
                }
                if let Some(m) = &marker {
                    for (x, y) in s.xs.iter().zip(&s.ys) {
                        if finite(*x) && finite(*y) {
                            marker_svg(&mut body, m, xmap(*x), yof(k, *y), &color);
                        }
                    }
                }
            }
        }
    }
    for m in &marks {
        match m {
            Mark::HLine(y) if finite(*y) => {
                let _ = write!(body, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{MARK}" stroke-width="1.2" stroke-dasharray="5 4"/>"#, c(ml), c(ml + pw), c(ymap(*y)), c(ymap(*y)));
            }
            Mark::VLine(x) if finite(*x) => {
                let _ = write!(body, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{MARK}" stroke-width="1.2" stroke-dasharray="5 4"/>"#, c(xmap(*x)), c(xmap(*x)), c(mt), c(bottom));
            }
            Mark::Segment(a, b, cc, d) if [a, b, cc, d].iter().all(|v| finite(**v)) => {
                let _ = write!(body, r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{MARK}" stroke-width="1.2" stroke-dasharray="5 4"/>"#, c(xmap(*a)), c(ymap(*b)), c(xmap(*cc)), c(ymap(*d)));
            }
            _ => {}
        }
    }
    let mut labels = String::new();
    for m in &marks {
        if let Mark::Label(t, x, y) = m {
            if finite(*x) && finite(*y) {
                let l = lab(t, 11.5);
                if l.math {
                    // centered on the point by its own height, as dominant-baseline cannot be for the shapes
                    let by = ymap(*y) + (l.h - l.d) / 2.0;
                    let _ = write!(labels, r#"<text x="{}" y="{}" text-anchor="middle" font-size="11.5" fill="{LABEL}">{}</text>{}"#, c(xmap(*x)), c(by), l.inner, l.shapes_svg(xmap(*x), by, Anchor::Middle, 0.0, LABEL));
                }
                else {
                    let _ = write!(labels, r#"<text x="{}" y="{}" text-anchor="middle" dominant-baseline="middle" font-size="11.5" fill="{LABEL}">{}</text>"#, c(xmap(*x)), c(ymap(*y)), l.inner);
                }
            }
        }
    }
    let mut all = g;
    let _ = write!(all, r#"<g clip-path="url(#{clip})">{under}{body}</g>{labels}"#);
    let mut notes = Vec::new();
    if !hidden.is_empty() {
        notes.push(format!("not drawn on the log axis: {}", hidden.join(", ")));
    }
    if !ch.undrawn.is_empty() {
        notes.push(format!("{} (not drawn by viv plot)", ch.undrawn.join(", ")));
    }
    for n in notes {
        let _ = write!(all, r#"<text x="{}" y="{}" font-size="11" fill="{TICK}">{}</text>"#, c(ml), c(h + 8.0), esc(&n));
        h += 16.0;
    }
    frame(width, h, &all, &fig.title, &legend)
}

// ---- diagrams ---------------------------------------------------------------------

/// Text sizes of a note, in the
/// diagram's units, and the fonts: a monospace character is 0.6 of its size.
const NOTE_SIZE: [(&str, f64); 3] = [("small", 11.0), ("normal", 13.0), ("large", 15.0)];
const MONO: &str = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
/// Space around a diagram, inside its card.
const DIAGRAM_MARGIN: f64 = 12.0;
/// An arrowhead's length and half width.
const HEAD_LEN: f64 = 7.0;
const HEAD_HALF: f64 = 3.2;

/// The color a style's color word names: a palette slot, `ink` or `muted`.
fn word_color(words: &[&str], default: &'static str) -> &'static str {
    for w in words {
        if let Some(k) = w.strip_prefix('s').and_then(|d| d.parse::<usize>().ok()).filter(|k| (1..=PALETTE.len()).contains(k)) {
            return PALETTE[k - 1];
        }
        match *w {
            "ink" => return TITLE,
            "muted" => return TICK,
            _ => {}
        }
    }
    default
}

fn has(words: &[&str], w: &str) -> bool {
    words.contains(&w)
}

fn dash_attr(words: &[&str]) -> &'static str {
    if has(words, "dash") {
        r#" stroke-dasharray="5 3""#
    } else if has(words, "dot") {
        r#" stroke-dasharray="1.5 2.5""#
    } else {
        ""
    }
}

/// A smooth path through the points (Catmull-Rom, as cubic Béziers), or
/// straight segments.
fn path_d(xs: &[f64], ys: &[f64], curved: bool) -> String {
    let n = xs.len();
    let mut d = format!("M{} {}", c(xs[0]), c(ys[0]));
    for i in 1..n {
        if curved && n > 2 {
            let (x0, y0) = if i >= 2 { (xs[i - 2], ys[i - 2]) } else { (xs[i - 1], ys[i - 1]) };
            let (x3, y3) = if i + 1 < n { (xs[i + 1], ys[i + 1]) } else { (xs[i], ys[i]) };
            let (x1, y1, x2, y2) = (xs[i - 1], ys[i - 1], xs[i], ys[i]);
            let _ = write!(d, "C{} {} {} {} {} {}", c(x1 + (x2 - x0) / 6.0), c(y1 + (y2 - y0) / 6.0), c(x2 - (x3 - x1) / 6.0), c(y2 - (y3 - y1) / 6.0), c(x2), c(y2));
        } else {
            let _ = write!(d, "L{} {}", c(xs[i]), c(ys[i]));
        }
    }
    d
}

/// A diagram: boxes, arrows and notes where the records put them, one unit
/// to a pixel, with no axes.
fn render_diagram(ch: &Chart) -> (String, f64) {
    let fig = ch.fig.clone().unwrap_or_default();
    // the window, or the marks' extent
    let (mut xlo, mut xhi, mut ylo, mut yhi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let mut take = |x: f64, y: f64| {
        xlo = xlo.min(x);
        xhi = xhi.max(x);
        ylo = ylo.min(y);
        yhi = yhi.max(y);
    };
    for m in &ch.diagram {
        match &m.what {
            DKind::Box(x0, y0, x1, y1) => {
                take(*x0, *y0);
                take(*x1, *y1);
            }
            DKind::Arrow(xs, ys) => xs.iter().zip(ys).filter(|(x, y)| x.is_finite() && y.is_finite()).for_each(|(x, y)| take(*x, *y)),
            DKind::Note(_, x, y) => take(*x, *y),
        }
    }
    if let Some((a, b)) = fig.xlim {
        xlo = a;
        xhi = b;
    }
    if let Some((a, b)) = fig.ylim {
        ylo = a;
        yhi = b;
    }
    if !(xlo.is_finite() && xhi.is_finite() && ylo.is_finite() && yhi.is_finite()) {
        xlo = 0.0;
        xhi = 1.0;
        ylo = 0.0;
        yhi = 1.0;
    }
    let px = |x: f64| DIAGRAM_MARGIN + (x - xlo);
    let py = |y: f64| DIAGRAM_MARGIN + (yhi - y);
    let mut body = String::new();
    for m in &ch.diagram {
        let words: Vec<&str> = m.style.split_whitespace().collect();
        let id = esc(&m.id);
        match &m.what {
            DKind::Box(x0, y0, x1, y1) => {
                let color = word_color(&words, TICK);
                let (fill, opacity) = if has(&words, "tint") {
                    (color, "0.07")
                } else if has(&words, "clear") {
                    ("none", "1")
                } else {
                    (BG, "1")
                };
                let stroke = if has(&words, "noline") { "none" } else { color };
                let sw = if has(&words, "thin") { "0.8" } else if has(&words, "thick") { "2" } else { "1.2" };
                let rx = if has(&words, "square") { "0" } else { "5" };
                if has(&words, "diamond") {
                    // the rhombus through the midpoints of the box's sides
                    let (l, r, t, b) = (px(x0.min(*x1)), px(x0.max(*x1)), py(y0.max(*y1)), py(y0.min(*y1)));
                    let (mx, my) = ((l + r) / 2.0, (t + b) / 2.0);
                    let _ = write!(body, r#"<path data-id="{id}" d="M{} {}L{} {}L{} {}L{} {}Z" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"{}/>"#,
                        c(mx), c(t), c(r), c(my), c(mx), c(b), c(l), c(my), dash_attr(&words));
                    continue;
                }
                let _ = write!(body, r#"<rect data-id="{id}" x="{}" y="{}" width="{}" height="{}" rx="{rx}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"{}/>"#,
                    c(px(x0.min(*x1))), c(py(y0.max(*y1))), c((x1 - x0).abs()), c((y1 - y0).abs()), dash_attr(&words));
            }
            DKind::Arrow(xs, ys) => {
                let pts: Vec<(f64, f64)> = xs.iter().zip(ys).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (px(*x), py(*y))).collect();
                if pts.len() < 2 {
                    continue;
                }
                let color = word_color(&words, TITLE);
                let sw = if has(&words, "thin") { 0.9 } else if has(&words, "thick") { 2.2 } else { 1.3 };
                let mut xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
                let mut ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
                let n = xs.len();
                let head = !has(&words, "nohead");
                // the head points along the last piece; the line stops at its base
                let (mut ux, mut uy) = (xs[n - 1] - xs[n - 2], ys[n - 1] - ys[n - 2]);
                let len = (ux * ux + uy * uy).sqrt();
                let (tipx, tipy) = (xs[n - 1], ys[n - 1]);
                if head && len > 0.0 {
                    ux /= len;
                    uy /= len;
                    let back = HEAD_LEN.min(len);
                    xs[n - 1] -= ux * back;
                    ys[n - 1] -= uy * back;
                }
                let _ = write!(body, r#"<path data-id="{id}" d="{}" fill="none" stroke="{color}" stroke-width="{}" stroke-linejoin="round"{}/>"#,
                    path_d(&xs, &ys, has(&words, "curved")), sw, dash_attr(&words));
                if head && len > 0.0 {
                    let (bx, by) = (tipx - ux * HEAD_LEN, tipy - uy * HEAD_LEN);
                    let _ = write!(body, r#"<path data-id="{id}" d="M{} {}L{} {}L{} {}Z" fill="{color}"/>"#,
                        c(tipx), c(tipy), c(bx - uy * HEAD_HALF), c(by + ux * HEAD_HALF), c(bx + uy * HEAD_HALF), c(by - ux * HEAD_HALF));
                }
            }
            DKind::Note(text, x, y) => {
                let color = word_color(&words, TITLE);
                let size = NOTE_SIZE.iter().find(|(w, _)| has(&words, w)).map(|(_, s)| *s).unwrap_or(NOTE_SIZE[0].1);
                let anchor = if has(&words, "left") { "start" } else if has(&words, "right") { "end" } else { "middle" };
                // the baseline, from the anchor point: the text's top, middle or bottom there
                let base = if has(&words, "top") { 0.78 * size } else if has(&words, "bottom") { -0.22 * size } else { 0.35 * size };
                let font = if has(&words, "mono") { format!(r#" font-family="{}""#, esc(MONO)) } else { String::new() };
                let weight = if has(&words, "bold") { r#" font-weight="600""# } else { "" };
                let _ = write!(body, r#"<text data-id="{id}" x="{}" y="{}" font-size="{}" text-anchor="{anchor}" fill="{color}"{font}{weight}>{}</text>"#,
                    c(px(*x)), c(py(*y) + base), size, esc(text));
            }
        }
    }
    let width = (xhi - xlo) + 2.0 * DIAGRAM_MARGIN;
    let height = (yhi - ylo) + 2.0 * DIAGRAM_MARGIN;
    frame(width, height, &body, &fig.title, &[])
}

/// A chart with nothing this renderer can draw but its notes.
// ---- heatmaps ----------------------------------------------------------------

/// `plot2d: label ; nx ny ; [z]`: (rows, columns, values) when the shape and the values agree.
fn heat_of(f: &[&str]) -> Option<(usize, usize, Vec<f64>)> {
    let shape = nums(f.get(1)?);
    let z = arr(f.get(2)?)?;
    let (nx, ny) = (*shape.first()? as usize, *shape.get(1)? as usize);
    (nx >= 1 && ny >= 1 && nx * ny == z.len()).then_some((nx, ny, z))
}

// the site's colormaps (www/src/components/VivCell.jsx): viridis at nine
// points, and ColorBrewer RdBu reversed for a signed field
const VIRIDIS: [[f64; 3]; 9] = [[68., 1., 84.], [71., 44., 122.], [59., 81., 139.], [44., 113., 142.], [33., 144., 141.], [39., 173., 129.], [92., 200., 99.], [170., 220., 50.], [253., 231., 37.]];
const DIVERGING: [[f64; 3]; 11] = [[5., 48., 97.], [33., 102., 172.], [67., 147., 195.], [146., 197., 222.], [209., 229., 240.], [247., 247., 247.], [253., 219., 199.], [244., 165., 130.], [214., 96., 77.], [178., 24., 43.], [103., 0., 31.]];

fn cmap(stops: &[[f64; 3]], t: f64) -> [u8; 3] {
    let x = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let i = (x.floor() as usize).min(stops.len() - 2);
    let f = x - i as f64;
    let (a, b) = (stops[i], stops[i + 1]);
    [0, 1, 2].map(|k| (a[k] + f * (b[k] - a[k])).round() as u8)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

/// An RGBA image as a PNG: rows of `w` pixels, unfiltered, in stored
/// (uncompressed) deflate blocks, so the bytes depend on the pixels alone.
fn png(w: usize, h: usize, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(h * (4 * w + 1));
    for row in rgba.chunks(4 * w) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (k, b) in blocks.iter().enumerate() {
        z.push(u8::from(k + 1 == blocks.len()));
        let n = b.len() as u16;
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(!n).to_le_bytes());
        z.extend_from_slice(b);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for g in bytes.chunks(3) {
        let n = (g[0] as u32) << 16 | (*g.get(1).unwrap_or(&0) as u32) << 8 | *g.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            s.push(if k <= g.len() { A[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

/// A heatmap: the values as a PNG (a cell that is not finite is left
/// transparent), axes in the figure's window or in cell coordinates, the marks
/// over it, and a colorbar at the right.
fn render_heat(ch: &Chart, width: f64, id: usize) -> (String, f64) {
    let fig = ch.fig.clone().unwrap_or_default();
    let Some((nx, ny, z)) = ch.heat.clone() else { return frame(width, 0.0, "", &fig.title, &[]) };
    let diverging = fig.colormap == "diverging";
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in z.iter().filter(|v| v.is_finite()) {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    if !lo.is_finite() {
        (lo, hi) = (0.0, 1.0);
    }
    let (cmin, cmax) = match fig.clim {
        Some(l) => l,
        None if diverging => {
            let m = lo.abs().max(hi.abs());
            let m = if m > 0.0 { m } else { 1.0 };
            (-m, m)
        }
        None => (lo, hi),
    };
    let stops: &[[f64; 3]] = if diverging { &DIVERGING } else { &VIRIDIS };
    let t = |v: f64| if cmax != cmin { (v - cmin) / (cmax - cmin) } else { 0.5 };
    // the picture: row 0 on top unless origin("lower")
    let mut rgba = Vec::with_capacity(nx * ny * 4);
    for r in 0..nx {
        let i = if fig.origin_lower { nx - 1 - r } else { r };
        for j in 0..ny {
            let v = z[i * ny + j];
            if v.is_finite() {
                rgba.extend_from_slice(&cmap(stops, t(v)));
                rgba.push(255);
            } else {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    let window = match (fig.xlim, fig.ylim) {
        (Some(x), Some(y)) => Some((x, y)),
        _ => None,
    };
    // a panel of a grid is narrow: smaller margins
    let (ml, mr, mt) = if fig.in_grid { (44.0, 62.0, 10.0) } else { (60.0, 86.0, 10.0) };
    let mb = if fig.xlabel.is_empty() { 30.0 } else { 46.0 };
    let avail = (width - ml - mr).max(60.0);
    // square cells, as tall as the column allows
    let cell = (avail / ny as f64).min(420.0 / nx as f64);
    let (pw, ph) = (cell * ny as f64, cell * nx as f64);
    let x0 = ml + (avail - pw) / 2.0;
    let (fx, fy): (Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>) = match window {
        Some(((xa, xb), (ya, yb))) => (Box::new(move |x| x0 + (x - xa) / (xb - xa) * pw), Box::new(move |y| mt + ph - (y - ya) / (yb - ya) * ph)),
        None => {
            let lower = fig.origin_lower;
            (Box::new(move |j| x0 + (j + 0.5) * cell), Box::new(move |i| if lower { mt + ph - (i + 0.5) * cell } else { mt + (i + 0.5) * cell }))
        }
    };
    let mut b = String::new();
    let _ = write!(b, r#"<image x="{}" y="{}" width="{}" height="{}" preserveAspectRatio="none" style="image-rendering:pixelated" href="data:image/png;base64,{}"/>"#,
        c(x0), c(mt), c(pw), c(ph), base64(&png(ny, nx, &rgba)));
    let _ = write!(b, r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{AXIS}"/>"#, c(x0), c(mt), c(pw), c(ph));
    // ticks: the program's own, or nice ones in the window or over the cell indices
    let prog = |want_x: bool| -> Vec<(f64, String)> {
        ch.marks.iter().filter_map(|m| match m {
            Mark::XTick(s, v) if want_x => Some((*v, s.clone())),
            Mark::YTick(s, v) if !want_x => Some((*v, s.clone())),
            _ => None,
        }).collect()
    };
    let auto = |a: f64, bnd: f64, n: usize, ints: bool| -> Vec<(f64, String)> {
        let (ts, step) = nice_ticks(a, bnd, n);
        if ints && step < 1.0 {
            let (ts, _) = nice_ticks(a, bnd, ((bnd - a) as usize).max(1));
            return ts.into_iter().filter(|v| v.fract() == 0.0).map(|v| (v, tick_label(v, 1.0))).collect();
        }
        ts.into_iter().map(|v| (v, tick_label(v, step))).collect()
    };
    let (xt, yt) = {
        let (px, py) = (prog(true), prog(false));
        let (xa, xb, ya, yb, ints) = match window {
            Some(((xa, xb), (ya, yb))) => (xa, xb, ya, yb, false),
            None => (0.0, ny as f64 - 1.0, 0.0, nx as f64 - 1.0, true),
        };
        (if px.is_empty() { auto(xa, xb, 6, ints) } else { px }, if py.is_empty() { auto(ya, yb, 5, ints) } else { py })
    };
    for (v, s) in &xt {
        let x = fx(*v);
        if x < x0 - 0.5 || x > x0 + pw + 0.5 {
            continue;
        }
        let l = lab(s, 11.0);
        let _ = write!(b, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/><text x="{}" y="{}" text-anchor="middle" font-size="11" fill="{TICK}">{}</text>{}"#,
            c(x), c(x), c(mt + ph), c(mt + ph + 4.0), c(x), c(mt + ph + 16.0), l.inner, l.shapes_svg(x, mt + ph + 16.0, Anchor::Middle, 0.0, TICK));
    }
    for (v, s) in &yt {
        let y = fy(*v);
        if y < mt - 0.5 || y > mt + ph + 0.5 {
            continue;
        }
        let l = lab(s, 11.0);
        let _ = write!(b, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/><text x="{}" y="{}" text-anchor="end" font-size="11" fill="{TICK}">{}</text>{}"#,
            c(x0 - 4.0), c(x0), c(y), c(y), c(x0 - 7.0), c(y + 3.5), l.inner, l.shapes_svg(x0 - 7.0, y + 3.5, Anchor::End, 0.0, TICK));
    }
    if !fig.xlabel.is_empty() {
        let l = lab(&fig.xlabel, 12.0);
        let _ = write!(b, r#"<text x="{}" y="{}" text-anchor="middle" font-size="12" fill="{LABEL}">{}</text>{}"#, c(x0 + pw / 2.0), c(mt + ph + 34.0), l.inner, l.shapes_svg(x0 + pw / 2.0, mt + ph + 34.0, Anchor::Middle, 0.0, LABEL));
    }
    if !fig.ylabel.is_empty() {
        let (lx, ly) = (x0 - ml + 14.0, mt + ph / 2.0);
        let l = lab(&fig.ylabel, 12.0);
        let _ = write!(b, r#"<text x="{}" y="{}" text-anchor="middle" font-size="12" fill="{LABEL}" transform="rotate(-90 {} {})">{}</text>{}"#, c(lx), c(ly), c(lx), c(ly), l.inner, l.shapes_svg(lx, ly, Anchor::Middle, -90.0, LABEL));
    }
    // marks over the picture, in its units, clipped to it
    let ink = if diverging { "#1e1e1e" } else { "#ffffff" };
    let mut over = String::new();
    for m in &ch.marks {
        match m {
            Mark::HLine(y) => { let _ = write!(over, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{ink}" stroke-width="1.3" stroke-dasharray="5 4"/>"#, c(x0), c(x0 + pw), c(fy(*y)), c(fy(*y))); }
            Mark::VLine(x) => { let _ = write!(over, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{ink}" stroke-width="1.3" stroke-dasharray="5 4"/>"#, c(fx(*x)), c(fx(*x)), c(mt), c(mt + ph)); }
            Mark::Segment(a, bb, cc, d) => { let _ = write!(over, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{ink}" stroke-width="1.5"/>"#, c(fx(*a)), c(fx(*cc)), c(fy(*bb)), c(fy(*d))); }
            Mark::Label(s, x, y) => {
                let l = lab(s, 12.0);
                let _ = write!(over, r#"<text x="{}" y="{}" font-size="12" fill="{ink}">{}</text>{}"#, c(fx(*x) + 4.0), c(fy(*y) - 4.0), l.inner, l.shapes_svg(fx(*x) + 4.0, fy(*y) - 4.0, Anchor::Start, 0.0, ink));
            }
            Mark::Rect(x, y, w, h) => {
                let (xa, xb, ya, yb) = (fx(x - w / 2.0), fx(x + w / 2.0), fy(y - h / 2.0), fy(y + h / 2.0));
                let _ = write!(over, r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{ink}" stroke-width="1.3"/>"#, c(xa.min(xb)), c(ya.min(yb)), c((xb - xa).abs()), c((yb - ya).abs()));
            }
            Mark::XTick(..) | Mark::YTick(..) => {}
        }
    }
    if !over.is_empty() {
        let clip = format!("vp{id}");
        let _ = write!(b, r#"<clipPath id="{clip}"><rect x="{}" y="{}" width="{}" height="{}"/></clipPath><g clip-path="url(#{clip})">{over}</g>"#, c(x0), c(mt), c(pw), c(ph));
    }
    // the colorbar: 128 steps of the map, low at the bottom
    let (bx, bw) = (x0 + pw + 16.0, 12.0);
    let mut bar = Vec::with_capacity(128 * 4);
    for k in 0..128 {
        bar.extend_from_slice(&cmap(stops, 1.0 - (k as f64 + 0.5) / 128.0));
        bar.push(255);
    }
    let _ = write!(b, r#"<image x="{}" y="{}" width="{}" height="{}" preserveAspectRatio="none" href="data:image/png;base64,{}"/><rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{AXIS}"/>"#,
        c(bx), c(mt), c(bw), c(ph), base64(&png(1, 128, &bar)), c(bx), c(mt), c(bw), c(ph));
    let (ts, step) = nice_ticks(cmin.min(cmax), cmin.max(cmax), 5);
    for v in ts {
        let y = mt + ph - t(v) * ph;
        let _ = write!(b, r#"<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/><text x="{}" y="{}" font-size="11" fill="{TICK}">{}</text>"#,
            c(bx + bw), c(bx + bw + 4.0), c(y), c(y), c(bx + bw + 7.0), c(y + 3.5), tick_label(v, step));
    }
    frame(width, mt + ph + mb, &b, &fig.title, &[])
}

fn render_note(ch: &Chart, width: f64) -> (String, f64) {
    let title = ch.fig.as_ref().map(|f| f.title.clone()).unwrap_or_default();
    let body = format!(r#"<text x="14" y="18" font-size="12" fill="{TICK}">{} (not drawn by viv plot)</text>"#, esc(&ch.undrawn.join(", ")));
    frame(width, 30.0, &body, &title, &[])
}

fn render_any(ch: &Chart, width: f64, id: usize) -> Option<(String, f64)> {
    if ch.fig.as_ref().is_some_and(|f| f.token == "diagram") {
        Some(render_diagram(ch))
    } else if ch.heat.is_some() {
        Some(render_heat(ch, width, id))
    } else if !ch.series.is_empty() {
        Some(render_chart(ch, width, id))
    } else if !ch.undrawn.is_empty() {
        Some(render_note(ch, width))
    } else {
        None
    }
}

/// The union of the charts' ranges on one axis, in data units (www/src/lib/figureGrid.js sharedRange).
fn shared_range(charts: &[&Chart], x_axis: bool) -> Option<(f64, f64)> {
    let (mut lo, mut hi, mut fixed, mut log) = (f64::INFINITY, f64::NEG_INFINITY, true, false);
    for ch in charts {
        let fig = ch.fig.clone().unwrap_or_default();
        let axis_log = if x_axis { fig.xlog } else { fig.ylog };
        log |= !x_axis && axis_log;
        let lim = if x_axis { fig.xlim } else { fig.ylim };
        let vals: Vec<f64> = match lim {
            Some((a, b)) => vec![a, b],
            None => {
                fixed = false;
                let mut v: Vec<f64> = Vec::new();
                for s in &ch.series {
                    if x_axis {
                        v.extend(&s.xs);
                    } else if !fig.right(&s.name) {
                        if matches!(s.kind, Kind::Fill | Kind::Range) {
                            v.extend(s.lo.iter().chain(&s.hi));
                        } else {
                            v.extend(&s.ys);
                        }
                    }
                }
                if !x_axis && ch.series.iter().any(|s| s.kind == Kind::Bars) {
                    v.push(0.0);
                }
                v
            }
        };
        for v in vals {
            if finite(v) && finite(tx(axis_log, v)) {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if !lo.is_finite() || hi == lo {
        return None;
    }
    if !x_axis && !fixed {
        if log {
            let (a, b) = (vmath::log10(lo), vmath::log10(hi));
            let pad = (b - a) * 0.07;
            return Some((pow10(a - pad), pow10(b + pad)));
        }
        let pad = (hi - lo) * 0.07;
        return Some((lo - pad, hi + pad));
    }
    Some((lo, hi))
}

fn render_grid(p: &Parsed, title: &str, cols: usize, share: &str, panels: &[Vec<usize>], width: f64, first_id: usize) -> (String, f64) {
    // each panel's charts, their figures told the shared ranges and which labels a neighbor shows
    let mut laid: Vec<Vec<Chart>> = panels.iter().map(|pp| pp.iter().map(|&k| p.charts[k].clone()).collect()).collect();
    for ch in laid.iter_mut().flatten() {
        if let Some(f) = ch.fig.as_mut() {
            f.in_grid = true;
        }
    }
    let n = laid.len();
    for (x_axis, word) in [(true, "x"), (false, "y")] {
        if !share.contains(word) {
            continue;
        }
        let groups = if x_axis { cols } else { n.div_ceil(cols) };
        for g in 0..groups {
            let members: Vec<usize> = (0..n).filter(|&k| if x_axis { k % cols == g } else { k / cols == g }).collect();
            let r = {
                let charts: Vec<&Chart> = members.iter().flat_map(|&k| laid[k].iter()).filter(|c| !c.series.is_empty()).collect();
                shared_range(&charts, x_axis)
            };
            if let Some(r) = r {
                for &k in &members {
                    for ch in laid[k].iter_mut().filter(|c| !c.series.is_empty()) {
                        if let Some(f) = ch.fig.as_mut() {
                            if x_axis { f.xlim = Some(r) } else { f.ylim = Some(r) }
                        }
                    }
                }
            }
        }
    }
    let rights: Vec<bool> = (0..cols).map(|col| (col..n).step_by(cols).any(|j| laid[j].iter().any(|c| c.fig.as_ref().is_some_and(|f| f.yright.is_some())))).collect();
    for (k, pp) in laid.iter_mut().enumerate() {
        for ch in pp.iter_mut() {
            if let Some(f) = ch.fig.as_mut() {
                f.inner = (share.contains('x') && k + cols < n, share.contains('y') && k % cols > 0);
                f.room_right = rights[k % cols];
            }
        }
    }
    let gap = 8.0;
    let pw = ((width - gap * (cols as f64 - 1.0)) / cols as f64).floor();
    let top = if title.is_empty() { 0.0 } else { 30.0 };
    let mut parts = String::new();
    let mut y = top;
    let mut id = first_id;
    for row in laid.chunks(cols) {
        let mut row_h: f64 = 0.0;
        for (j, pp) in row.iter().enumerate() {
            let mut yy = y;
            for ch in pp {
                if let Some((svg, hh)) = render_any(ch, pw, id) {
                    id += 1;
                    parts.push_str(&svg.replacen("<svg ", &format!(r#"<svg x="{}" y="{}" "#, c(j as f64 * (pw + gap)), c(yy)), 1));
                    yy += hh + 4.0;
                }
            }
            row_h = row_h.max(yy - y);
        }
        y += row_h + gap;
    }
    let mut svg = String::new();
    let _ = write!(svg, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" font-family="{}" role="img" aria-label="{}">"#,
        c(width), c(y), c(width), c(y), esc(FONT), esc(if title.is_empty() { "grid of figures" } else { title }));
    if !title.is_empty() {
        let l = lab(title, 14.0);
        let _ = write!(svg, r#"<text x="{}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="{TITLE}">{}</text>{}"#, c(width / 2.0), l.inner, l.shapes_svg(width / 2.0, 20.0, Anchor::Middle, 0.0, TITLE));
    }
    svg.push_str(&parts);
    svg.push_str("</svg>");
    (svg, y)
}

/// Every chart of a run's output, each as its own SVG document, in order: a
/// grid of figures is one. Charts with nothing to draw are left out.
pub fn render_all(text: &str, width: f64) -> Vec<String> {
    LABEL_ERRORS.with(|v| v.borrow_mut().clear());
    let p = parse(text);
    let mut out = Vec::new();
    let mut id = 0;
    for e in &p.entries {
        match e {
            Entry::Chart(k) => {
                if let Some((svg, _)) = render_any(&p.charts[*k], width, id) {
                    out.push(svg);
                    id += 1;
                }
            }
            Entry::Grid { title, cols, share, panels, .. } => {
                // a panel may be a diagram, which has marks and no series
                if panels.iter().flatten().any(|&k| !p.charts[k].series.is_empty() || !p.charts[k].undrawn.is_empty() || !p.charts[k].diagram.is_empty() || p.charts[k].heat.is_some()) {
                    out.push(render_grid(&p, title, *cols, share, panels, width, id).0);
                    id += 64;
                }
            }
        }
    }
    out
}

/// The whole output as one SVG: every chart, one under another.
pub fn render_page(text: &str, width: f64) -> String {
    let charts = render_all(text, width);
    let mut y = 0.0;
    let mut parts = String::new();
    // a diagram is drawn at its own size, which may be wider than the page
    let mut width = width;
    for svg in &charts {
        let w = svg
            .split_once(r#"width=""#)
            .and_then(|(_, r)| r.split_once('"'))
            .and_then(|(v, _)| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        width = width.max(w);
    }
    for svg in &charts {
        let h = svg
            .split_once(r#"height=""#)
            .and_then(|(_, r)| r.split_once('"'))
            .and_then(|(v, _)| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        parts.push_str(&svg.replacen("<svg ", &format!(r#"<svg x="0" y="{}" "#, c(y)), 1));
        y += h + 12.0;
    }
    let total = (y - 12.0).max(0.0);
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">{parts}</svg>"#,
        c(width), c(total), c(width), c(total)
    ) + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_labels_match_the_site() {
        assert_eq!(tick_label(0.5, 0.5), "0.5");
        assert_eq!(tick_label(-20.0, 20.0), "\u{2212}20");
        assert_eq!(tick_label(150000.0, 50000.0), "1.5e5");
        assert_eq!(tick_label(0.00001, 0.00001), "1e-5");
        let (t, minor) = axis_ticks(true, -3.0, -1.0, 5);
        assert_eq!(t.iter().map(|(_, l)| l.as_str()).collect::<Vec<_>>(), ["10⁻³", "10⁻²", "10⁻¹"]);
        assert_eq!(minor.len(), 16);
        let (t, _) = axis_ticks(true, vmath::log10(2.0), vmath::log10(8.0), 5);
        assert_eq!(t.iter().map(|(_, l)| l.as_str()).collect::<Vec<_>>(), ["2", "3", "4", "5", "6", "7", "8"]);
    }

    #[test]
    fn the_same_records_give_the_same_bytes() {
        let rec = "figure: t ; x ; y\nplot: a ; [0, 1, 2] ; [1, 4, 9]\nhline: h ; 2\n";
        assert_eq!(render_page(rec, 640.0), render_page(rec, 640.0));
        assert!(!render_page(rec, 640.0).contains("NaN"));
    }

    #[test]
    fn a_grid_of_diagrams_is_drawn() {
        // two diagram panels: marks only, no series
        let rec = concat!(
            "grid: two ways ; 1 2\n",
            "figure: left ;  ;  ; 0 100 0 80 ; diagram\n",
            "arrow: a ; s1 ; [10, 90] ; [10, 70]\n",
            "figure: right ;  ;  ; 0 100 0 80 ; diagram\n",
            "note: n ; θ₁ = 30° ; s2 left middle ; 20 40\n",
        );
        let all = render_all(rec, 640.0);
        assert_eq!(all.len(), 1);
        assert!(all[0].contains(r#"data-id="a""#) && all[0].contains("θ₁ = 30°"));
    }
}
