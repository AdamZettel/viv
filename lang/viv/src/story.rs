// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The story lens, milestone L5: one debug
//! run told in the intent's words, with the watched values filled in.
//!
//! A debug run of a program with a debug lens ends with one `__story__` line
//! (the interpreter's `render_output`, the C runtime's `viv_story_flush`):
//!
//! ```text
//! __story__ {"total":N,"events":[["call","scf"],["enter","scf","guess",1],...]}
//! ```
//!
//! The events, in the order they happened:
//!
//! | event | fields |
//! |---|---|
//! | `call`, `ret` | def |
//! | `enter`, `exit` | def, step id, nesting depth |
//! | `iter` | def, iteration number (from 1) |
//! | `iter_end` | def |
//! | `watch` | def, where (a step id or `iteration`), `{name: value}` |
//! | `hold` | def, where, passed, sentence, worst \|a - b\| or null |
//!
//! This module reads that line back into a tree -- calls hold steps and
//! iterations, steps and iterations hold what ran inside them -- and prints
//! it: one line per call and step, and one line per iteration with the
//! values watched at its end. Nothing here runs a program; `viv story` in
//! main.rs runs it and hands over the line.

use std::collections::HashMap;

/// A JSON value, numbers kept as written (so a value prints as recorded).
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => n.parse().ok(),
            Json::Str(s) => match s.as_str() {
                "nan" => Some(f64::NAN),
                "inf" => Some(f64::INFINITY),
                "-inf" => Some(f64::NEG_INFINITY),
                _ => None,
            },
            _ => None,
        }
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Obj(fs) = self { fs.iter().find(|(k, _)| k == key).map(|(_, v)| v) } else { None }
    }
    /// The value as JSON text.
    pub fn text(&self) -> String {
        match self {
            Json::Null => "null".into(),
            Json::Bool(b) => b.to_string(),
            Json::Num(n) => n.clone(),
            Json::Str(s) => crate::debug_lens::json_str(s),
            Json::Arr(xs) => format!("[{}]", xs.iter().map(Json::text).collect::<Vec<_>>().join(",")),
            Json::Obj(fs) => format!("{{{}}}", fs.iter().map(|(k, v)| format!("{}:{}", crate::debug_lens::json_str(k), v.text())).collect::<Vec<_>>().join(",")),
        }
    }
}

struct Reader<'a> {
    s: &'a [u8],
    i: usize,
}

impl Reader<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("the story is not valid JSON at byte {}: {what}", self.i))
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'n') => self.word("null", Json::Null),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut xs = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(xs));
                }
                loop {
                    xs.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(xs));
                        }
                        _ => return self.err("expected `,` or `]`"),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut fs = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(fs));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return self.err("expected `:`");
                    }
                    self.i += 1;
                    fs.push((k, self.value()?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(fs));
                        }
                        _ => return self.err("expected `,` or `}`"),
                    }
                }
            }
            Some(_) => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                }
                if self.i == start {
                    return self.err("unexpected character");
                }
                Ok(Json::Num(String::from_utf8_lossy(&self.s[start..self.i]).to_string()))
            }
        }
    }
    fn word(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            self.err("unexpected word")
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        while let Some(&c) = self.s.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return Ok(String::from_utf8_lossy(&out).to_string()),
                b'\\' => {
                    let e = self.s.get(self.i).copied();
                    self.i += 1;
                    match e {
                        Some(b'n') => out.push(b'\n'),
                        Some(b't') => out.push(b'\t'),
                        Some(b'r') => out.push(b'\r'),
                        Some(b'u') => {
                            let hex = std::str::from_utf8(self.s.get(self.i..self.i + 4).unwrap_or(b"")).unwrap_or("");
                            let ch = u32::from_str_radix(hex, 16).ok().and_then(char::from_u32).unwrap_or('\u{fffd}');
                            self.i += 4;
                            let mut b = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                        }
                        Some(o) => out.push(o),
                        None => return self.err("unterminated string"),
                    }
                }
                c => out.push(c),
            }
        }
        self.err("unterminated string")
    }
}

/// Parse JSON text.
pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut r = Reader { s: text.as_bytes(), i: 0 };
    let v = r.value()?;
    r.ws();
    if r.i != r.s.len() {
        return r.err("trailing text");
    }
    Ok(v)
}

/// One hold as recorded.
#[derive(Clone, Debug, PartialEq)]
pub struct HoldRec {
    pub def: String,
    pub at: String,
    pub ok: bool,
    pub sentence: String,
    pub worst: Option<Json>,
}

/// A node of the story: the run (`main`), a call, a step or an iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// "main", "call", "step" or "iteration"
    pub kind: &'static str,
    pub def: String,
    /// a step's id
    pub id: String,
    /// an iteration's number
    pub n: i64,
    /// the values watched here, in the order recorded
    pub watch: Vec<(String, Json)>,
    pub holds: Vec<HoldRec>,
    pub children: Vec<Node>,
}

impl Node {
    fn new(kind: &'static str, def: &str) -> Node {
        Node { kind, def: def.to_string(), id: String::new(), n: 0, watch: Vec::new(), holds: Vec::new(), children: Vec::new() }
    }
}

/// A story read back from its line.
#[derive(Clone, Debug, PartialEq)]
pub struct Story {
    /// the build that ran: "exact" (every engine's default), or "fast" for a
    /// fast build, whose numbers may differ from the
    /// exact run's within `viv check --fast`'s tolerance
    pub mode: String,
    /// events recorded in the run, and how many of them the line holds
    pub total: u64,
    pub kept: usize,
    pub root: Node,
}

impl Story {
    /// Holds checked and holds that failed, over the kept events.
    pub fn hold_counts(&self) -> (usize, usize) {
        fn go(n: &Node, c: &mut (usize, usize)) {
            c.0 += n.holds.len();
            c.1 += n.holds.iter().filter(|h| !h.ok).count();
            for k in &n.children {
                go(k, c);
            }
        }
        let mut c = (0, 0);
        go(&self.root, &mut c);
        c
    }
}

/// The `__story__` line of a run's output, without its prefix.
pub fn story_line(output: &str) -> Option<&str> {
    output.lines().find_map(|l| l.strip_prefix("__story__ "))
}

/// Read a story from its line (the part after `__story__ `).
pub fn parse(line: &str) -> Result<Story, String> {
    let v = parse_json(line)?;
    let total = v.get("total").and_then(Json::num).ok_or("the story has no total")? as u64;
    let Some(Json::Arr(events)) = v.get("events") else { return Err("the story has no events".into()) };
    let mut stack: Vec<Node> = vec![Node::new("main", crate::parser::MAIN_DEF)];
    // Close the innermost open node into its parent.
    fn close(stack: &mut Vec<Node>) {
        if stack.len() > 1 {
            let n = stack.pop().expect("more than the root");
            stack.last_mut().expect("the root").children.push(n);
        }
    }
    for ev in events {
        let Json::Arr(f) = ev else { return Err(format!("a story event is not a list: {}", ev.text())) };
        let kind = f.first().and_then(Json::str).unwrap_or("");
        let def = f.get(1).and_then(Json::str).unwrap_or("").to_string();
        match kind {
            "call" => stack.push(Node::new("call", &def)),
            "ret" => {
                // close up to and including the innermost call of this def
                if let Some(k) = stack.iter().rposition(|n| n.kind == "call" && n.def == def) {
                    while stack.len() > k {
                        close(&mut stack);
                    }
                }
            }
            "enter" => {
                let mut n = Node::new("step", &def);
                n.id = f.get(2).and_then(Json::str).unwrap_or("").to_string();
                stack.push(n);
            }
            "exit" => {
                let id = f.get(2).and_then(Json::str).unwrap_or("");
                if let Some(k) = stack.iter().rposition(|n| n.kind == "step" && n.def == def && n.id == id) {
                    while stack.len() > k {
                        close(&mut stack);
                    }
                }
            }
            "iter" => {
                let mut n = Node::new("iteration", &def);
                n.n = f.get(2).and_then(Json::num).unwrap_or(0.0) as i64;
                stack.push(n);
            }
            "iter_end" => {
                if let Some(k) = stack.iter().rposition(|n| n.kind == "iteration" && n.def == def) {
                    while stack.len() > k {
                        close(&mut stack);
                    }
                }
            }
            // a line-level recording's events (`crate::debug_lens::with_lines`)
            // belong to the explorer; the story is told by steps and iterations
            "watch" if f.get(2).and_then(Json::str).is_some_and(|at| at.starts_with("line:") || at.starts_with("br:")) => {}
            "watch" => {
                let Some(Json::Obj(vals)) = f.get(3) else { return Err(format!("a watch event without values: {}", ev.text())) };
                stack.last_mut().expect("the root").watch.extend(vals.iter().cloned());
            }
            "hold" => {
                let h = HoldRec {
                    def,
                    at: f.get(2).and_then(Json::str).unwrap_or("").to_string(),
                    ok: matches!(f.get(3), Some(Json::Bool(true))),
                    sentence: f.get(4).and_then(Json::str).unwrap_or("").to_string(),
                    worst: f.get(5).filter(|w| **w != Json::Null).cloned(),
                };
                stack.last_mut().expect("the root").holds.push(h);
            }
            other => return Err(format!("unknown story event `{other}`")),
        }
    }
    while stack.len() > 1 {
        close(&mut stack);
    }
    let mode = v.get("mode").and_then(Json::str).unwrap_or("exact").to_string();
    Ok(Story { mode, total, kept: events.len(), root: stack.pop().expect("the root") })
}

/// What the story says about each def and step, from the intent lens: a
/// def's `what`, and each step's text, keyed `def/id`.
#[derive(Clone, Debug, Default)]
pub struct Words {
    pub what: HashMap<String, String>,
    pub steps: HashMap<String, String>,
    /// placeholders, which the story can only meet as an error
    pub unwritten: Vec<String>,
}

/// A number as a person reads it: ten significant digits, trailing zeros off.
pub fn num_text(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    let trim = |s: String| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    };
    if !(-4..10).contains(&e) {
        let s = format!("{:.9e}", x);
        let (m, ex) = s.split_once('e').unwrap_or((&s, "0"));
        let ex: i32 = ex.parse().unwrap_or(0);
        format!("{}e{}{:02}", trim(m.to_string()), if ex < 0 { '-' } else { '+' }, ex.abs())
    } else {
        trim(format!("{:.*}", (9 - e).max(0) as usize, x))
    }
}

/// A watched value as a person reads it: numbers as numbers, an array as its
/// shape and norm (and its values, when it has at most four).
pub fn value_text(v: &Json) -> String {
    match v {
        Json::Num(_) | Json::Str(_) => v.num().map(num_text).unwrap_or_else(|| v.text()),
        Json::Bool(b) => b.to_string(),
        Json::Obj(_) => {
            let shape: Vec<String> = match v.get("shape") {
                Some(Json::Arr(d)) => d.iter().map(Json::text).collect(),
                _ => Vec::new(),
            };
            let shape = shape.join("x");
            if let Some(Json::Arr(vals)) = v.get("values") {
                let xs: Vec<f64> = vals.iter().filter_map(Json::num).collect();
                if xs.len() <= 4 {
                    return format!("[{}]", xs.iter().map(|x| num_text(*x)).collect::<Vec<_>>().join(", "));
                }
                let norm = xs.iter().map(|x| x * x).sum::<f64>().sqrt();
                return format!("{shape} |{}|", num_text(norm));
            }
            let norm = v.get("norm").and_then(Json::num).map(num_text).unwrap_or_default();
            format!("{shape} |{norm}|")
        }
        other => other.text(),
    }
}

fn watch_text(w: &[(String, Json)]) -> String {
    w.iter().map(|(k, v)| format!("{k} {}", value_text(v))).collect::<Vec<_>>().join("  ")
}

fn hold_text(h: &HoldRec) -> String {
    let mut s = format!("hold FAILED at {}: {}", h.at, h.sentence);
    if let Some(w) = &h.worst {
        s.push_str(&format!(" (worst {})", value_text(w)));
    }
    s
}

/// A def's `what` up to its first full stop or semicolon: one line beside
/// the call. The JSON form carries the whole of it.
fn first_clause(what: &str) -> String {
    let mut end = what.len();
    for pat in [". ", "; "] {
        if let Some(k) = what.find(pat) {
            end = end.min(k);
        }
    }
    what[..end].trim_end_matches('.').to_string()
}

/// Options for the text rendering.
#[derive(Clone, Copy, Debug, Default)]
pub struct Opts {
    /// show calls down to this many levels below the top level (1: only
    /// the calls the top level makes)
    pub depth: Option<usize>,
    /// show what ran inside each iteration, not only its line
    pub expand: bool,
}

/// The story as indented text.
pub fn render_text(story: &Story, words: &Words, opts: Opts) -> String {
    // (indent, label, detail)
    let mut lines: Vec<(usize, String, String)> = Vec::new();
    fn failed_holds(n: &Node, out: &mut Vec<HoldRec>) {
        out.extend(n.holds.iter().filter(|h| !h.ok).cloned());
        for c in &n.children {
            failed_holds(c, out);
        }
    }
    fn go(n: &Node, indent: usize, level: usize, words: &Words, opts: Opts, lines: &mut Vec<(usize, String, String)>) {
        let kids = |lines: &mut Vec<(usize, String, String)>, indent: usize| {
            for c in &n.children {
                go(c, indent, level, words, opts, lines);
            }
        };
        match n.kind {
            "main" => {
                lines.push((indent, n.def.clone(), watch_text(&n.watch)));
                for h in n.holds.iter().filter(|h| !h.ok) {
                    lines.push((indent + 1, hold_text(h), String::new()));
                }
                kids(lines, indent + 1);
            }
            "call" => {
                let level = level + 1;
                if opts.depth.is_some_and(|k| level > k) {
                    return;
                }
                let what = words.what.get(&n.def).map(|w| first_clause(w)).unwrap_or_default();
                lines.push((indent, n.def.clone(), what));
                for h in n.holds.iter().filter(|h| !h.ok) {
                    lines.push((indent + 1, hold_text(h), String::new()));
                }
                for c in &n.children {
                    go(c, indent + 1, level, words, opts, lines);
                }
            }
            "step" => {
                let key = format!("{}/{}", n.def, n.id);
                let label = words.steps.get(&key).filter(|t| !t.is_empty()).cloned().unwrap_or_else(|| n.id.clone());
                lines.push((indent, label, watch_text(&n.watch)));
                for h in n.holds.iter().filter(|h| !h.ok) {
                    lines.push((indent + 1, hold_text(h), String::new()));
                }
                kids(lines, indent + 1);
            }
            "iteration" => {
                lines.push((indent, format!("iteration {}", n.n), watch_text(&n.watch)));
                if opts.expand {
                    for h in n.holds.iter().filter(|h| !h.ok) {
                        lines.push((indent + 1, hold_text(h), String::new()));
                    }
                    kids(lines, indent + 1);
                } else {
                    // collapsed: the iteration's line, and any hold that failed inside it
                    let mut bad = Vec::new();
                    failed_holds(n, &mut bad);
                    for h in &bad {
                        lines.push((indent + 1, hold_text(h), String::new()));
                    }
                }
            }
            _ => {}
        }
    }
    go(&story.root, 0, 0, words, opts, &mut lines);
    // one column for the details; a label too long for it just runs past
    let width = lines.iter().filter(|(_, _, d)| !d.is_empty()).map(|(i, l, _)| 2 * i + l.chars().count()).filter(|w| *w <= 40).max().unwrap_or(0);
    let mut out = String::new();
    if story.mode != "exact" {
        out.push_str(&format!("(a {} build: its numbers may differ from the exact run's within the tolerance of `viv check --fast`)\n", story.mode));
    }
    for (i, l, d) in &lines {
        let left = format!("{}{}", "  ".repeat(*i), l);
        if d.is_empty() {
            out.push_str(left.trim_end());
        } else {
            let pad = width.saturating_sub(left.chars().count());
            out.push_str(&format!("{left}{}  {d}", " ".repeat(pad)));
        }
        out.push('\n');
    }
    let (checked, failed) = story.hold_counts();
    if checked > 0 {
        out.push_str(&format!("holds: {checked} checked, {failed} failed\n"));
    }
    if (story.kept as u64) < story.total {
        out.push_str(&format!("(the run recorded {} events; the story keeps the first {})\n", story.total, story.kept));
    }
    out
}

/// The story as JSON, with the intent's words filled in: what the viewer
/// and the explorer read.
pub fn render_json(story: &Story, words: &Words) -> String {
    fn node(n: &Node, words: &Words) -> String {
        let mut f: Vec<String> = vec![format!("\"kind\":\"{}\"", n.kind), format!("\"def\":{}", crate::debug_lens::json_str(&n.def))];
        match n.kind {
            "call" => f.push(format!("\"what\":{}", crate::debug_lens::json_str(words.what.get(&n.def).map(String::as_str).unwrap_or("")))),
            "step" => {
                f.push(format!("\"id\":{}", crate::debug_lens::json_str(&n.id)));
                let key = format!("{}/{}", n.def, n.id);
                f.push(format!("\"text\":{}", crate::debug_lens::json_str(words.steps.get(&key).map(String::as_str).unwrap_or(""))));
            }
            "iteration" => f.push(format!("\"n\":{}", n.n)),
            _ => {}
        }
        f.push(format!("\"watch\":{}", Json::Obj(n.watch.clone()).text()));
        let holds: Vec<String> = n
            .holds
            .iter()
            .map(|h| {
                format!(
                    "{{\"at\":{},\"ok\":{},\"sentence\":{},\"worst\":{}}}",
                    crate::debug_lens::json_str(&h.at),
                    h.ok,
                    crate::debug_lens::json_str(&h.sentence),
                    h.worst.as_ref().map(Json::text).unwrap_or_else(|| "null".into())
                )
            })
            .collect();
        f.push(format!("\"holds\":[{}]", holds.join(",")));
        f.push(format!("\"children\":[{}]", n.children.iter().map(|c| node(c, words)).collect::<Vec<_>>().join(",")));
        format!("{{{}}}", f.join(","))
    }
    let (checked, failed) = story.hold_counts();
    // an exact run's JSON is what it always was; a fast run's says so
    let mode = if story.mode == "exact" { String::new() } else { format!("\"mode\":{},", crate::debug_lens::json_str(&story.mode)) };
    format!(
        "{{{mode}\"total\":{},\"kept\":{},\"holds\":{{\"checked\":{checked},\"failed\":{failed}}},\"root\":{}}}",
        story.total,
        story.kept,
        node(&story.root, words)
    )
}
