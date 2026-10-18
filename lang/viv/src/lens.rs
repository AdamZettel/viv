// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Lenses, milestones L2-L4: the
//! intent lens and the tools that keep it true.
//!
//! - The `.intent` file format: a parser that keeps every line as written, so
//!   printing what it parsed gives back the same bytes, and a renderer that
//!   writes new entries in the canonical layout.
//! - `lens init`: a skeleton `.intent` with every def, step id, control word and
//!   current body hash.
//! - `view --lens code|intent|all`: the program without its intent lines, the
//!   intent without its code, or both interleaved.
//! - `lens check`: staleness by body hash, coverage, step and nesting agreement,
//!   one home per anchor, and numbers in prose that must cite a test.
//! - `lens stamp` and `lens move`: re-stamp a def after reviewing its entry, and
//!   move entries between the `.intent` file and inline `#|` comments.
//! - `explain`: the intent walked along the static call graph.
//!
//! None of this touches how a program runs. Everything here reads source text
//! and the IR's body hashes; nothing is emitted into the C or the interpreter.
//! The file system and git stay in main.rs: this module is handed the texts.

use std::collections::{HashMap, HashSet};

use knotkit::span::SourceId;

use crate::ast::{Expr, Item, Stmt};
use crate::parser::MAIN_DEF;

// ---------------------------------------------------------------------------
// The `.intent` file
// ---------------------------------------------------------------------------

/// The three prose fields of a def, in the order `#|`, `#||`, `#|||`.
pub const FIELDS: [&str; 3] = ["what", "how", "why"];

/// Every field a card can hold: the prose fields, and the two that say why
/// the def is needed: `entry <who calls
/// it>` declares a library entry point, reachable from outside the
/// repository's programs; `serves [test "..."]` names what the def exists
/// for beyond its callers. These two live only in the `.intent` file (an
/// entry point can also be tagged inline, `#~ entry`).
pub const CARD_FIELDS: [&str; 5] = ["what", "how", "why", "entry", "serves"];

/// Words that open a control line in an outline. `for`, `while`, `if` and
/// `else` are Viv keywords, so no step can be named after them; `repeat` is
/// not, and a step named `repeat` cannot be listed in an `.intent` file.
const CONTROL: [&str; 5] = ["repeat", "for", "while", "if", "else"];

/// Detail lines that only exist under a step (at the top of the outline these
/// words are step ids): its own why and how, a sentence about the recorded
/// run, and a question to answer before that sentence is shown.
const STEP_DETAILS: [&str; 4] = ["why", "how", "run", "predict"];

/// What an outline line can have indented under it.
const HOLDS_NOTHING: u8 = 0;
const HOLDS_ANY: u8 = 1;
/// a line anchor: equations and other detail lines about that one statement
const HOLDS_DETAILS: u8 = 2;

/// What one line of an `.intent` file is.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Blank,
    /// `module <name>`
    Module(String),
    /// an indented line of the module's description
    ModuleText(String),
    /// `<def>  @<hash>  by <who>`; the hash and the signature are optional
    Header { name: String, stamp: Option<String>, by: Option<String> },
    /// `what`, `how` or `why` and its text
    Field { label: String, text: String },
    /// a glossary line: a name in the def's scope and what it holds
    Gloss { name: String, text: String },
    /// a deeper-indented line continuing the field or glossary line above
    Cont(String),
    /// an outline line naming a step: `id  text`
    Step { id: String, text: String },
    /// an outline line opening a control structure, as written:
    /// `repeat until the density stops changing`
    Control(String),
    /// an outline line of prose that is not a step: `return the energy ...`
    Note(String),
    /// a line of detail under a step (or at the top of the outline, for the
    /// def's own statements): `why <text>`, `how <text>`, a line anchor
    /// `= <anchor>  <text>`, `> <text>`, another paragraph of the line
    /// above, `run <text>` (what the recorded run did, with `{name}` for a
    /// value from it), `predict <text>` (a question to answer before the
    /// `run` line is shown), or an equation `$$ <LaTeX> $$`, which may
    /// cite what checks it: `[hold "..."]`, `[test "..."]`. `label` is
    /// `why`, `how`, `run`, `predict`, `=<anchor>`, `>`, `$$` (the line
    /// holds the whole equation, dollars included) or `$$+` (a further line
    /// of an equation the `$$` line above left open).
    Detail { label: String, text: String },
    /// `path <id>  <title>`: a reading path through the program (the Read
    /// view's core path), a block of its own like `module`
    Path { id: String, title: String },
    /// a stop of a path: `<def>`, `<def>/<step>` or `<def>/<step>=<anchor>`
    /// (`<def>=<anchor>` for a statement outside every step), and a heading
    PathStop { target: String, text: String },
    /// a line under a stop: `> <prose>`, `figure <title>` (a figure the
    /// program draws, by its title), `run <text>` or `predict <text>`
    PathDetail { label: String, text: String },
}

/// One line: its bytes as written, its indentation, and what it is.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub raw: String,
    pub indent: usize,
    pub kind: Kind,
}

/// A parsed `.intent` file. It keeps every line as written, so `print` gives
/// back the text `parse` was handed, byte for byte.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct IntentFile {
    pub lines: Vec<Line>,
    pub final_newline: bool,
}

/// A problem at a line (1-based) of some file.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

/// The module's own entry: `module <name>` and its description.
#[derive(Clone, Debug, PartialEq)]
pub struct ModuleEntry {
    pub name: String,
    pub text: String,
    pub line: usize,
    /// `module <name>  documents every variable`: `lens check` requires a
    /// glossary line for every parameter and every variable of each def the
    /// file has an entry for
    pub every_var: bool,
}

/// What a module line may say after the module's name.
pub const EVERY_VARIABLE: &str = "documents every variable";

/// The details a path stop can have under it.
const PATH_DETAILS: [&str; 4] = [">", "figure", "run", "predict"];

/// A reading path: `path <id>  <title>` and its stops.
#[derive(Clone, Debug, PartialEq)]
pub struct PathBlock {
    pub id: String,
    pub title: String,
    pub line: usize,
    pub stops: Vec<PathStop>,
}

/// One stop of a path: a def's card, a step, or one statement.
#[derive(Clone, Debug, PartialEq)]
pub struct PathStop {
    /// as written: `lmder`, `lmder/radius`, `lmpar/lower=parl`
    pub target: String,
    pub def: String,
    pub step: Option<String>,
    pub anchor: Option<String>,
    /// the heading the path gives the stop (may be empty)
    pub text: String,
    pub line: usize,
    /// `>`, `figure`, `run`, `predict` lines, in file order
    pub details: Vec<Detail>,
}

/// A stop's target split into (def, step, anchor), or None if malformed.
pub fn split_stop(target: &str) -> Option<(String, Option<String>, Option<String>)> {
    let (head, anchor) = match target.split_once('=') {
        Some((h, a)) => (h, Some(a.to_string())),
        None => (target, None),
    };
    let (def, step) = match head.split_once('/') {
        Some((d, s)) => (d, Some(s.to_string())),
        None => (head, None),
    };
    if !is_def_name(def) || step.as_deref().is_some_and(|s| !is_ident(s)) {
        return None;
    }
    if let Some(a) = &anchor {
        let base = a.split('#').next().unwrap_or("");
        let base = base.strip_suffix("()").unwrap_or(base);
        if !is_ident(base) {
            return None;
        }
    }
    Some((def.to_string(), step, anchor))
}

/// A glossary key split into the variable and where this doc starts to hold:
/// `wa1` (the variable's doc), `wa1@trial` (from its first use in step
/// `trial` on), `temp@evaluate#2` (from the second statement of `evaluate`
/// that assigns it), `s2@#2` (from the second statement of the def that
/// assigns it). None if malformed.
pub fn split_doc_key(key: &str) -> Option<(String, Option<String>, Option<usize>)> {
    let (name, at) = match key.split_once('@') {
        Some((n, a)) => (n, Some(a)),
        None => (key, None),
    };
    if !is_ident(name) {
        return None;
    }
    let Some(at) = at else { return Some((name.to_string(), None, None)) };
    let (step, k) = match at.split_once('#') {
        Some((s, k)) => (s, Some(k.parse::<usize>().ok().filter(|&k| k >= 1)?)),
        None => (at, None),
    };
    if step.is_empty() {
        return k.map(|k| (name.to_string(), None, Some(k)));
    }
    if !is_ident(step) {
        return None;
    }
    Some((name.to_string(), Some(step.to_string()), k))
}

/// A text with the key it is filed under (a field label or a glossary name)
/// and the line it starts on.
#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub key: String,
    pub text: String,
    pub line: usize,
}

/// The control structures an outline can name. A step nested in one of these
/// in the code must sit under the matching control line in the intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctl {
    /// `while` or `for` in the code; `repeat`, `while` or `for` in the intent
    Loop,
    If,
    ElseIf,
    Else,
}

impl Ctl {
    fn sig(self) -> &'static str {
        match self {
            Ctl::Loop => "loop",
            Ctl::If => "if",
            Ctl::ElseIf => "else-if",
            Ctl::Else => "else",
        }
    }
    /// The control a written control line opens.
    fn of_line(text: &str) -> Ctl {
        let mut w = text.split_whitespace();
        match w.next() {
            Some("if") => Ctl::If,
            Some("else") => {
                if w.next() == Some("if") {
                    Ctl::ElseIf
                } else {
                    Ctl::Else
                }
            }
            _ => Ctl::Loop,
        }
    }
}

/// One node of an intent outline.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    Step(String),
    Control(Ctl),
    Note,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    /// the step's text, or the whole control or note line
    pub text: String,
    pub line: usize,
    pub children: Vec<Node>,
    /// a step's `why`, `how` and line anchors, in file order
    pub details: Vec<Detail>,
}

/// A step's `why` or `how`, or the explanation of one statement, anchored
/// by what it does rather than by its line number: `= x` is the statement that assigns `x`, `= f()` the one that
/// calls `f`, and `#k` after either picks the k-th such statement (from 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Detail {
    /// `why`, `how`, `run`, `predict`, `=` or `$$` (an equation)
    pub label: String,
    /// for `=`: the anchor as written (`x`, `f()`, `x#2`)
    pub anchor: String,
    /// paragraphs separated by `\n`; for `$$`, the equation's lines as
    /// written, dollars and citations included
    pub text: String,
    pub line: usize,
    /// the anchor of the `=` line this detail is indented under, or empty:
    /// an equation or a `run` line can belong to one statement
    pub under: String,
}

impl Detail {
    /// The anchor's parts: the name, whether it names a call, and which match (1-based).
    pub fn target(&self) -> (String, bool, usize) {
        let (base, k) = match self.anchor.split_once('#') {
            Some((b, k)) => (b, k.parse::<usize>().unwrap_or(1).max(1)),
            None => (self.anchor.as_str(), 1),
        };
        match base.strip_suffix("()") {
            Some(f) => (f.to_string(), true, k),
            None => (base.to_string(), false, k),
        }
    }
}

/// One def's block in an `.intent` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub stamp: Option<String>,
    pub by: Option<String>,
    pub line: usize,
    /// index range into `IntentFile::lines`: header to last line of the block
    pub first: usize,
    pub last: usize,
    pub fields: Vec<Text>,
    pub gloss: Vec<Text>,
    pub outline: Vec<Node>,
    /// line anchors at the top of the outline: the def's statements outside
    /// every step
    pub details: Vec<Detail>,
}

impl Entry {
    pub fn field(&self, label: &str) -> Option<&Text> {
        self.fields.iter().find(|t| t.key == label)
    }
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_') && !s.starts_with(|c: char| c.is_ascii_digit())
}

fn is_def_name(s: &str) -> bool {
    is_ident(s) || s == MAIN_DEF
}

fn is_hash(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// The first word of `s` and the rest, the rest without its leading space.
fn first_word(s: &str) -> (&str, &str) {
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], s[i..].trim_start()),
        None => (s, ""),
    }
}

impl IntentFile {
    /// Parse the text of an `.intent` file.
    ///
    /// The layout rules, which keep the file readable as prose:
    /// - an unindented line opens a block: `module <name>`, or a def's header,
    ///   `<def>  @<hash>`;
    /// - a def block's first indented lines are its card: `what`, `how` and
    ///   `why`, and glossary lines `name  description`; a line indented deeper
    ///   than the card continues the line above it;
    /// - a blank line inside the block ends the card, and the lines after it
    ///   are the outline: steps `id  text`, control lines (`repeat ...`,
    ///   `for ...`, `while ...`, `if ...`, `else ...`), and `return ...` notes,
    ///   nested by indentation. A block with no card starts with the blank line.
    pub fn parse(text: &str) -> Result<IntentFile, Vec<ParseError>> {
        let (body, final_newline) = match text.strip_suffix('\n') {
            Some(b) => (b, true),
            None => (text, false),
        };
        let raws: Vec<&str> = if text.is_empty() { Vec::new() } else { body.split('\n').collect() };
        let blank = |r: &str| r.trim().is_empty();
        // the indentation of the next non-blank line after each line, if any
        let mut next_indent: Vec<Option<usize>> = vec![None; raws.len()];
        let mut ahead = None;
        for i in (0..raws.len()).rev() {
            next_indent[i] = ahead;
            if !blank(raws[i]) {
                ahead = Some(raws[i].len() - raws[i].trim_start_matches(' ').len());
            }
        }
        #[derive(PartialEq, Clone, Copy)]
        enum State {
            Top,
            Module,
            Card,
            Outline,
            Path,
        }
        let mut state = State::Top;
        let mut card_indent: Option<usize> = None;
        // the indentation of each open outline node, and what it can hold under it
        let mut stack: Vec<(usize, u8)> = Vec::new();
        // the column of the block's outermost outline lines
        let mut outline_root: Option<usize> = None;
        let mut last_card_kind_ok = false;
        let mut seen_module = false;
        let mut lines = Vec::with_capacity(raws.len());
        let mut errs = Vec::new();
        // an equation whose `$$` line did not close it: (its line, its indent)
        let mut open_eq: Option<(usize, usize)> = None;
        // the column of a path's stops
        let mut path_indent: Option<usize> = None;
        for (i, raw) in raws.iter().enumerate() {
            let n = i + 1;
            let content = raw.trim_start_matches(' ');
            let indent = raw.len() - content.len();
            if content.starts_with('\t') {
                errs.push(ParseError { line: n, message: "indent with spaces, not tabs".into() });
            }
            let content = content.trim_end();
            // the lines of an open equation, up to the one with its closing `$$`
            if let Some((at, eq_indent)) = open_eq {
                if !content.is_empty() && indent >= eq_indent && state == State::Outline {
                    if content.contains("$$") {
                        open_eq = None;
                    }
                    lines.push(Line { raw: raw.to_string(), indent, kind: Kind::Detail { label: "$$+".into(), text: content.to_string() } });
                    continue;
                }
                errs.push(ParseError { line: at, message: "this equation's `$$` is never closed (continue it on lines indented at least as far, and end it with `$$`)".into() });
                open_eq = None;
            }
            let mut err = |m: String| errs.push(ParseError { line: n, message: m });
            let kind = if content.is_empty() {
                // a blank line stays inside a block when an indented line follows
                let inside = next_indent[i].is_some_and(|k| k > 0);
                match state {
                    State::Card | State::Outline if inside => {
                        if state == State::Card {
                            stack.clear();
                            outline_root = None;
                        }
                        state = State::Outline;
                    }
                    State::Module | State::Path if inside => {}
                    _ => state = State::Top,
                }
                Kind::Blank
            } else if indent == 0 {
                if let Some(rest) = content.strip_prefix("module ") {
                    if seen_module {
                        err("a second `module` line; a file holds one module".into());
                    }
                    let (_, flags) = first_word(rest.trim());
                    if !flags.is_empty() && flags.split_whitespace().collect::<Vec<_>>().join(" ") != EVERY_VARIABLE {
                        err(format!("after the module's name expected nothing or `{EVERY_VARIABLE}`, found `{flags}`"));
                    }
                    seen_module = true;
                    state = State::Module;
                    Kind::Module(rest.trim().to_string())
                } else if let Some((id, title)) = content.strip_prefix("path ").map(|r| first_word(r.trim())).filter(|(id, _)| is_ident(id) && *id != "by" && !id.starts_with('@')) {
                    state = State::Path;
                    path_indent = None;
                    Kind::Path { id: id.to_string(), title: title.to_string() }
                } else {
                    let mut toks = content.split_whitespace();
                    let name = toks.next().unwrap_or("").to_string();
                    if !is_def_name(&name) {
                        err(format!("expected a def name or `module <name>`, found `{name}`"));
                    }
                    let mut stamp = None;
                    let mut by = None;
                    let mut rest: Vec<&str> = toks.collect();
                    if let Some(first) = rest.first() {
                        if let Some(h) = first.strip_prefix('@') {
                            if !is_hash(h) {
                                err(format!("`@{h}` is not a body hash (8 hex digits, as `viv hash` prints)"));
                            }
                            stamp = Some(h.to_string());
                            rest.remove(0);
                        }
                    }
                    if !rest.is_empty() {
                        if rest[0] == "by" && rest.len() > 1 {
                            by = Some(rest[1..].join(" "));
                        } else {
                            err(format!("after `{name}` expected `@<hash>` and then `by <who>`, found `{}`", rest.join(" ")));
                        }
                    }
                    state = State::Card;
                    card_indent = None;
                    last_card_kind_ok = false;
                    stack.clear();
                    outline_root = None;
                    Kind::Header { name, stamp, by }
                }
            } else {
                match state {
                    State::Top => {
                        err("an indented line outside any block (a block starts with an unindented def name)".into());
                        Kind::Blank
                    }
                    State::Module => Kind::ModuleText(content.to_string()),
                    State::Path => {
                        let base = *path_indent.get_or_insert(indent);
                        let (word, rest) = first_word(content);
                        if indent == base {
                            if split_stop(word).is_none() {
                                err(format!("`{word}`: a path stop is a def, `def/step`, or `def/step=anchor`"));
                            }
                            Kind::PathStop { target: word.to_string(), text: rest.to_string() }
                        } else if indent > base {
                            if !PATH_DETAILS.contains(&word) {
                                err(format!("under a path stop expected `>`, `figure`, `run` or `predict`, found `{word}`"));
                            }
                            Kind::PathDetail { label: word.to_string(), text: rest.to_string() }
                        } else {
                            err(format!("this path line is indented {indent}, the stops above it {base}"));
                            Kind::Blank
                        }
                    }
                    State::Card => {
                        let base = *card_indent.get_or_insert(indent);
                        if indent > base {
                            if !last_card_kind_ok {
                                err("a continuation line with nothing above it to continue".into());
                            }
                            Kind::Cont(content.to_string())
                        } else if indent < base {
                            err(format!("this card line is indented {indent}, the lines above it {base}"));
                            Kind::Blank
                        } else {
                            let (word, rest) = first_word(content);
                            last_card_kind_ok = true;
                            if CARD_FIELDS.contains(&word) {
                                Kind::Field { label: word.to_string(), text: rest.to_string() }
                            } else if is_ident(word) || (word.contains('@') && split_doc_key(word).is_some()) {
                                Kind::Gloss { name: word.to_string(), text: rest.to_string() }
                            } else if word.contains('@') {
                                err(format!("`{word}`: a glossary line for a later use of a variable is `name@step`, `name@step#k` or `name@#k`"));
                                Kind::Blank
                            } else {
                                err(format!("expected `what`, `how`, `why` or a glossary name, found `{word}`"));
                                Kind::Blank
                            }
                        }
                    }
                    State::Outline => {
                        // siblings share a column; a child sits right of its parent
                        let mut popped_deeper = false;
                        while stack.last().is_some_and(|&(k, _)| k > indent) {
                            stack.pop();
                            popped_deeper = true;
                        }
                        match stack.last().copied() {
                            Some((k, _)) if k == indent => {
                                stack.pop();
                            }
                            Some((_, _)) if popped_deeper => {
                                err("this line does not line up with any line above it".into());
                            }
                            Some((_, HOLDS_NOTHING)) => {
                                err("only a step or a control line can have lines under it".into());
                            }
                            None if outline_root.is_some_and(|r| r != indent) => {
                                err("this line does not line up with any line above it".into());
                            }
                            _ => {}
                        }
                        let parent = stack.last().map(|&(_, h)| h);
                        outline_root.get_or_insert(indent);
                        let (word, rest) = first_word(content);
                        let under = !stack.is_empty();
                        let kind = if content.starts_with("$$") {
                            if content.matches("$$").count() < 2 {
                                open_eq = Some((n, indent));
                            }
                            Kind::Detail { label: "$$".into(), text: content.to_string() }
                        } else if word == "=" || word == ">" || (STEP_DETAILS.contains(&word) && under) {
                            if word == "=" {
                                let (anchor, text) = first_word(rest);
                                let base = anchor.split('#').next().unwrap_or("");
                                let base = base.strip_suffix("()").unwrap_or(base);
                                if !is_ident(base) {
                                    err(format!("`= {anchor}`: a line anchor is a variable the statement assigns (`= x`) or a def it calls (`= f()`), optionally `#k` for the k-th"));
                                }
                                Kind::Detail { label: format!("={anchor}"), text: text.to_string() }
                            } else {
                                Kind::Detail { label: word.to_string(), text: rest.to_string() }
                            }
                        } else if CONTROL.contains(&word) {
                            Kind::Control(content.to_string())
                        } else if word == "return" {
                            Kind::Note(content.to_string())
                        } else if is_ident(word) {
                            Kind::Step { id: word.to_string(), text: rest.to_string() }
                        } else {
                            err(format!("expected a step id, a control word or `return`, found `{word}`"));
                            Kind::Note(content.to_string())
                        };
                        let is_anchor = matches!(&kind, Kind::Detail { label, .. } if label.starts_with('='));
                        if parent == Some(HOLDS_DETAILS) && (is_anchor || !matches!(kind, Kind::Detail { .. })) {
                            err("only equation, `run`, `predict`, `why`, `how` and `>` lines can sit under a line anchor".into());
                        }
                        let holds = match &kind {
                            Kind::Detail { .. } if is_anchor => HOLDS_DETAILS,
                            Kind::Note(_) | Kind::Detail { .. } => HOLDS_NOTHING,
                            _ => HOLDS_ANY,
                        };
                        stack.push((indent, holds));
                        kind
                    }
                }
            };
            lines.push(Line { raw: raw.to_string(), indent, kind });
        }
        if let Some((at, _)) = open_eq {
            errs.push(ParseError { line: at, message: "this equation's `$$` is never closed (continue it on lines indented at least as far, and end it with `$$`)".into() });
        }
        if errs.is_empty() {
            Ok(IntentFile { lines, final_newline })
        } else {
            Err(errs)
        }
    }

    /// The text of the file: every line exactly as parsed or rendered.
    pub fn print(&self) -> String {
        let mut out = self.lines.iter().map(|l| l.raw.as_str()).collect::<Vec<_>>().join("\n");
        if self.final_newline {
            out.push('\n');
        }
        out
    }

    /// The module entry and the def entries, in file order.
    pub fn model(&self) -> (Option<ModuleEntry>, Vec<Entry>) {
        let mut module: Option<ModuleEntry> = None;
        let mut entries: Vec<Entry> = Vec::new();
        let mut in_module = false;
        // (indent, path of child indexes) of the open outline nodes
        let mut open: Vec<(usize, Vec<usize>)> = Vec::new();
        // where continuation lines go: a field or a glossary line
        let mut cont_target: Option<(bool, usize)> = None;
        // the last line anchor, while the lines after it are indented under it
        let mut last_anchor: Option<(usize, String)> = None;
        // after a path block, lines belong to no def until the next header
        let mut entries_closed = false;
        for (i, l) in self.lines.iter().enumerate() {
            let n = i + 1;
            match &l.kind {
                Kind::Blank => {
                    if in_module {
                        if let Some(m) = module.as_mut() {
                            if !m.text.is_empty() && !m.text.ends_with('\n') {
                                m.text.push('\n');
                            }
                        }
                    }
                }
                Kind::Module(rest) => {
                    in_module = true;
                    let (name, flags) = first_word(rest);
                    module = Some(ModuleEntry { name: name.to_string(), text: String::new(), line: n, every_var: !flags.is_empty() });
                }
                Kind::Path { .. } | Kind::PathStop { .. } | Kind::PathDetail { .. } => {
                    // a path is a block of its own: it belongs to no def
                    in_module = false;
                    entries_closed = true;
                }
                Kind::ModuleText(t) => {
                    if let Some(m) = module.as_mut() {
                        if !m.text.is_empty() && !m.text.ends_with('\n') {
                            m.text.push(' ');
                        }
                        m.text.push_str(t);
                    }
                }
                Kind::Header { name, stamp, by } => {
                    in_module = false;
                    entries_closed = false;
                    open.clear();
                    cont_target = None;
                    entries.push(Entry {
                        name: name.clone(),
                        stamp: stamp.clone(),
                        by: by.clone(),
                        line: n,
                        first: i,
                        last: i,
                        fields: Vec::new(),
                        gloss: Vec::new(),
                        outline: Vec::new(),
                        details: Vec::new(),
                    });
                }
                _ => {
                    if entries_closed {
                        continue;
                    }
                    let Some(e) = entries.last_mut() else { continue };
                    e.last = i;
                    match &l.kind {
                        Kind::Field { label, text } => {
                            e.fields.push(Text { key: label.clone(), text: text.clone(), line: n });
                            cont_target = Some((true, e.fields.len() - 1));
                        }
                        Kind::Gloss { name, text } => {
                            e.gloss.push(Text { key: name.clone(), text: text.clone(), line: n });
                            cont_target = Some((false, e.gloss.len() - 1));
                        }
                        Kind::Cont(t) => {
                            let target = match cont_target {
                                Some((true, k)) => e.fields.get_mut(k),
                                Some((false, k)) => e.gloss.get_mut(k),
                                None => None,
                            };
                            if let Some(tx) = target {
                                if !tx.text.is_empty() {
                                    tx.text.push(' ');
                                }
                                tx.text.push_str(t);
                            }
                        }
                        Kind::Step { .. } | Kind::Control(_) | Kind::Note(_) => {
                            last_anchor = None;
                            let node = match &l.kind {
                                Kind::Step { id, text } => Node { kind: NodeKind::Step(id.clone()), text: text.clone(), line: n, children: vec![], details: vec![] },
                                Kind::Control(t) => Node { kind: NodeKind::Control(Ctl::of_line(t)), text: t.clone(), line: n, children: vec![], details: vec![] },
                                Kind::Note(t) => Node { kind: NodeKind::Note, text: t.clone(), line: n, children: vec![], details: vec![] },
                                _ => unreachable!(),
                            };
                            while open.last().is_some_and(|(k, _)| *k >= l.indent) {
                                open.pop();
                            }
                            let path = match open.last() {
                                Some((_, p)) => p.clone(),
                                None => Vec::new(),
                            };
                            let list = node_list(&mut e.outline, &path);
                            list.push(node);
                            let mut p = path;
                            p.push(list.len() - 1);
                            open.push((l.indent, p));
                        }
                        Kind::Detail { label, text } => {
                            // it belongs to the innermost open step above it,
                            // or to the def when no step is open
                            while open.last().is_some_and(|(k, _)| *k >= l.indent) {
                                open.pop();
                            }
                            let mut step_path: Option<Vec<usize>> = None;
                            for (_, p) in open.iter().rev() {
                                let mut ns: &Vec<Node> = &e.outline;
                                let mut node: Option<&Node> = None;
                                for &k in p {
                                    node = ns.get(k);
                                    ns = match node {
                                        Some(nd) => &nd.children,
                                        None => break,
                                    };
                                }
                                if node.is_some_and(|nd| matches!(nd.kind, NodeKind::Step(_))) {
                                    step_path = Some(p.clone());
                                    break;
                                }
                            }
                            let target: &mut Node;
                            let list: &mut Vec<Detail> = match &step_path {
                                Some(p) => {
                                    let (last, up) = p.split_last().expect("a path to a node");
                                    target = &mut node_list(&mut e.outline, up)[*last];
                                    &mut target.details
                                }
                                None => &mut e.details,
                            };
                            if label == "$$+" {
                                if let Some(d) = list.last_mut() {
                                    d.text.push('\n');
                                    d.text.push_str(text);
                                }
                            } else if label == ">" {
                                // another paragraph of the prose above it: an
                                // equation in between is skipped, so the text
                                // is not typeset as part of the display
                                if let Some(d) = list.iter_mut().rev().find(|d| d.label != "$$") {
                                    d.text.push('\n');
                                    d.text.push_str(text);
                                }
                            } else {
                                let (lab, anchor) = match label.strip_prefix('=') {
                                    Some(a) => ("=".to_string(), a.to_string()),
                                    None => (label.clone(), String::new()),
                                };
                                let under = match &last_anchor {
                                    Some((k, a)) if l.indent > *k => a.clone(),
                                    _ => String::new(),
                                };
                                if lab == "=" {
                                    last_anchor = Some((l.indent, anchor.clone()));
                                } else if under.is_empty() {
                                    last_anchor = None;
                                }
                                list.push(Detail { label: lab, anchor, text: text.clone(), line: n, under });
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(m) = module.as_mut() {
            m.text = m.text.trim_end().to_string();
        }
        (module, entries)
    }

    /// The file's reading paths, in file order.
    pub fn paths(&self) -> Vec<PathBlock> {
        let mut out: Vec<PathBlock> = Vec::new();
        let mut inside = false;
        for (i, l) in self.lines.iter().enumerate() {
            let n = i + 1;
            match &l.kind {
                Kind::Path { id, title } => {
                    inside = true;
                    out.push(PathBlock { id: id.clone(), title: title.clone(), line: n, stops: Vec::new() });
                }
                Kind::PathStop { target, text } if inside => {
                    let (def, step, anchor) = split_stop(target).unwrap_or_default();
                    if let Some(p) = out.last_mut() {
                        p.stops.push(PathStop { target: target.clone(), def, step, anchor, text: text.clone(), line: n, details: Vec::new() });
                    }
                }
                Kind::PathDetail { label, text } if inside => {
                    if let Some(st) = out.last_mut().and_then(|p| p.stops.last_mut()) {
                        st.details.push(Detail { label: label.clone(), anchor: String::new(), text: text.clone(), line: n, under: String::new() });
                    }
                }
                Kind::Blank => {}
                _ => inside = false,
            }
        }
        out
    }
}

/// The child list at `path` in an outline.
fn node_list<'a>(root: &'a mut Vec<Node>, path: &[usize]) -> &'a mut Vec<Node> {
    let mut list = root;
    for &k in path {
        list = &mut list[k].children;
    }
    list
}

// ---------------------------------------------------------------------------
// Rendering entries in the canonical layout
// ---------------------------------------------------------------------------

/// The width prose is wrapped to when it is written out.
const WRAP: usize = 88;

/// Words of `text` in lines of at most `width` characters (a longer word
/// gets a line of its own).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    // a citation `[test "..."]` is kept on one line
    let mut atoms: Vec<String> = Vec::new();
    let mut open = false;
    for w in text.split_whitespace() {
        if open {
            let last = atoms.last_mut().expect("an open citation has a first word");
            last.push(' ');
            last.push_str(w);
        } else {
            atoms.push(w.to_string());
        }
        if w.contains("\"]") {
            open = false;
        } else if w.starts_with("[test") {
            open = true;
        }
    }
    for w in atoms.iter().map(String::as_str) {
        if !cur.is_empty() && cur.chars().count() + 1 + w.chars().count() > width {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

/// What a block is rendered from.
#[derive(Clone, Debug, Default)]
pub struct Block {
    pub name: String,
    pub stamp: Option<String>,
    pub by: Option<String>,
    pub fields: Vec<(String, String)>,
    pub gloss: Vec<(String, String)>,
    pub outline: Vec<Node>,
    /// line anchors outside every step
    pub details: Vec<Detail>,
}

impl Block {
    fn from_entry(e: &Entry) -> Block {
        Block {
            name: e.name.clone(),
            stamp: e.stamp.clone(),
            by: e.by.clone(),
            fields: e.fields.iter().map(|t| (t.key.clone(), t.text.clone())).collect(),
            gloss: e.gloss.iter().map(|t| (t.key.clone(), t.text.clone())).collect(),
            outline: e.outline.clone(),
            details: e.details.clone(),
        }
    }
    fn is_empty_of_text(&self) -> bool {
        fn node_has_text(n: &Node) -> bool {
            match n.kind {
                NodeKind::Step(_) => !n.text.is_empty(),
                NodeKind::Control(_) => !is_bare_control(&n.text),
                NodeKind::Note => true,
            }
        }
        fn any(ns: &[Node]) -> bool {
            ns.iter().any(|n| node_has_text(n) || any(&n.children))
        }
        self.fields.iter().all(|(_, t)| t.is_empty()) && self.gloss.is_empty() && !any(&self.outline) && self.details.is_empty()
    }
}

fn header_line(name: &str, stamp: Option<&str>, by: Option<&str>) -> String {
    let mut h = name.to_string();
    if let Some(s) = stamp {
        h.push_str(&format!("  @{s}"));
    }
    if let Some(b) = by {
        h.push_str(&format!("  by {b}"));
    }
    h
}

/// The lines of one block, in the canonical layout: the header, the card with
/// its labels padded to one column (prose wrapped under it), and after a blank
/// line the outline, each level indented two more and step ids padded to one
/// column. An empty text leaves no trailing space.
pub fn render_block(b: &Block, with_stamp: bool) -> Vec<String> {
    let mut out = vec![header_line(&b.name, if with_stamp { b.stamp.as_deref() } else { None }, if with_stamp { b.by.as_deref() } else { None })];
    let labels = b.fields.iter().map(|(k, _)| k.chars().count()).chain(b.gloss.iter().map(|(k, _)| k.chars().count())).max().unwrap_or(0);
    let col = 2 + labels + 2;
    for (k, t) in b.fields.iter().chain(b.gloss.iter()) {
        if t.is_empty() {
            out.push(format!("  {k}"));
            continue;
        }
        for (j, piece) in wrap(t, WRAP.saturating_sub(col).max(20)).into_iter().enumerate() {
            if j == 0 {
                out.push(format!("  {k:<w$}{piece}", w = labels + 2));
            } else {
                out.push(format!("{}{piece}", " ".repeat(col)));
            }
        }
    }
    if !b.outline.is_empty() || !b.details.is_empty() {
        out.push(String::new());
        let idw = max_step_id(&b.outline);
        render_details(&b.details, 2, &mut out);
        render_nodes(&b.outline, 2, idw, &mut out);
    }
    out
}

fn render_details(ds: &[Detail], indent: usize, out: &mut Vec<String>) {
    for d in ds {
        // a line that belongs to a line anchor sits under it
        let pad = " ".repeat(if d.under.is_empty() { indent } else { indent + 2 });
        if d.label == "$$" {
            for (k, part) in d.text.split('\n').enumerate() {
                out.push(if k == 0 { format!("{pad}{part}") } else { format!("{pad}  {part}") });
            }
            continue;
        }
        let mut paras = d.text.split('\n');
        let first = paras.next().unwrap_or("");
        let head = if d.label == "=" { format!("= {}", d.anchor) } else { d.label.clone() };
        out.push(if first.is_empty() { format!("{pad}{head}") } else { format!("{pad}{head}  {first}") });
        for p in paras {
            out.push(format!("{pad}>  {p}"));
        }
    }
}

fn max_step_id(ns: &[Node]) -> usize {
    ns.iter()
        .map(|n| {
            let own = if let NodeKind::Step(id) = &n.kind { id.chars().count() } else { 0 };
            own.max(max_step_id(&n.children))
        })
        .max()
        .unwrap_or(0)
}

fn render_nodes(ns: &[Node], indent: usize, idw: usize, out: &mut Vec<String>) {
    for n in ns {
        let pad = " ".repeat(indent);
        match &n.kind {
            NodeKind::Step(id) if n.text.is_empty() => out.push(format!("{pad}{id}")),
            NodeKind::Step(id) => out.push(format!("{pad}{id:<w$}{}", n.text, w = idw + 2)),
            NodeKind::Control(_) | NodeKind::Note => out.push(format!("{pad}{}", n.text)),
        }
        render_details(&n.details, indent + 2, out);
        render_nodes(&n.children, indent + 2, idw, out);
    }
}

/// The lines of a module block.
fn render_module(m: &ModuleEntry) -> Vec<String> {
    let mut out = vec![if m.every_var { format!("module {}  {EVERY_VARIABLE}", m.name) } else { format!("module {}", m.name) }];
    for (k, para) in m.text.split('\n').enumerate() {
        if k > 0 {
            out.push(String::new());
        }
        if !para.trim().is_empty() {
            for piece in wrap(para, WRAP - 2) {
                out.push(format!("  {piece}"));
            }
        }
    }
    out
}

/// A whole file from a module entry and blocks, blocks separated by one blank line.
pub fn render_file(module: Option<&ModuleEntry>, blocks: &[Block], with_stamp: bool) -> String {
    let mut parts: Vec<Vec<String>> = Vec::new();
    if let Some(m) = module {
        parts.push(render_module(m));
    }
    for b in blocks {
        parts.push(render_block(b, with_stamp));
    }
    let mut out = String::new();
    for (k, p) in parts.iter().enumerate() {
        if k > 0 {
            out.push('\n');
        }
        for l in p {
            out.push_str(l);
            out.push('\n');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The code side: each def's steps, control structure, calls and inline intent
// ---------------------------------------------------------------------------

/// A module handed to the lens tools: its name (`lib/hf/rhf`), the path it is
/// reported under, its source, and its `.intent` file if it has one.
#[derive(Clone, Debug, Default)]
pub struct ModuleText {
    pub name: String,
    pub path: String,
    pub src: String,
    pub intent_path: String,
    pub intent: Option<String>,
}

/// What the lens tools read: the module, every module its `#! use` lines reach
/// (each once, in the order first used), and its source with those lines
/// expanded, which is what the body hashes are computed from (the same text
/// `viv hash` reads).
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub root: ModuleText,
    pub used: Vec<ModuleText>,
    pub expanded: String,
    /// the tests of the module's companion test file (`<dir>/tests/<name>.viv`
    /// beside `<dir>/<name>.viv`): tests that need data or drivers a program
    /// using the module should not load. A citation may name them.
    pub companion_tests: Vec<String>,
}

/// One node of a def's code outline.
#[derive(Clone, Debug, PartialEq)]
pub enum OKind {
    Step(String),
    Control(Ctl),
    /// a `test "name"` block at the top level
    Test(String),
    /// a call to a def of the program; `value` when the def is named as a
    /// function value (a registry member) rather than called
    Call { name: String, value: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ONode {
    pub kind: OKind,
    /// a step's inline text; a control's generated word (`repeat`, `for each i`)
    pub text: String,
    /// raw source lines (1-based) of the node's first and last line
    pub line: usize,
    pub end: usize,
    pub children: Vec<ONode>,
}

/// A def as the lens tools see it.
#[derive(Clone, Debug)]
pub struct CodeDef {
    pub name: String,
    pub sig: String,
    /// raw source lines of `def` and of the closing brace; 0 for `<main>`
    pub line: usize,
    pub end: usize,
    /// the body hash `viv hash` prints, when the program builds
    pub hash: Option<String>,
    /// statements in the body, nested ones included, steps not counted
    pub stmts: usize,
    pub outline: Vec<ONode>,
    pub inline: Inline,
    /// a `placeholder def`: a signature with no body yet
    pub placeholder: bool,
}

impl CodeDef {
    /// Every step in the def, in source order, with its nesting depth.
    pub fn steps(&self) -> Vec<(&ONode, usize)> {
        fn go<'a>(ns: &'a [ONode], d: usize, out: &mut Vec<(&'a ONode, usize)>) {
            for n in ns {
                let d2 = if matches!(n.kind, OKind::Step(_)) {
                    out.push((n, d));
                    d + 1
                } else {
                    d
                };
                go(&n.children, d2, out);
            }
        }
        let mut out = Vec::new();
        go(&self.outline, 1, &mut out);
        out
    }
    fn step(&self, id: &str) -> Option<&ONode> {
        self.steps().into_iter().map(|(n, _)| n).find(|n| n.kind == OKind::Step(id.to_string()))
    }
}

/// A def's intent written in its source: `#|` `#||` `#|||` above it, `#:`
/// glossary lines above it or in its body, and a `#@ <hash>` stamp.
#[derive(Clone, Debug, Default)]
pub struct Inline {
    /// (label, text, raw line of its first line)
    pub fields: Vec<(String, String, usize)>,
    pub gloss: Vec<(String, String, usize)>,
    pub stamp: Option<(String, usize)>,
    /// the def's `#~` tags
    pub tags: Vec<String>,
    /// raw lines (1-based) holding the def's `#|`, `#:` and `#@` lines
    pub lines: Vec<usize>,
    /// a `#|` note directly above a control statement that holds steps, by the
    /// statement's raw line: (text, the raw lines of the note)
    pub controls: Vec<(usize, String, Vec<usize>)>,
}

impl Inline {
    pub fn field(&self, label: &str) -> Option<&(String, String, usize)> {
        self.fields.iter().find(|f| f.0 == label)
    }
    fn has_intent(&self, def: &CodeDef) -> bool {
        !self.fields.is_empty() || !self.gloss.is_empty() || def.steps().iter().any(|(n, _)| !n.text.is_empty())
    }
}

/// Raw line (1-based) of a byte offset.
struct LineIndex(Vec<usize>);

impl LineIndex {
    fn new(src: &str) -> LineIndex {
        let mut starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        LineIndex(starts)
    }
    fn line(&self, offset: u32) -> usize {
        match self.0.binary_search(&(offset as usize)) {
            Ok(k) => k + 1,
            Err(k) => k,
        }
    }
}

/// A control line that says nothing beyond its structure: a word `lens init`
/// generates (`repeat`, `for each i`, `if`, `else`, `else if`) or a bare
/// `while` or `for`. Such a line needs no inline home.
fn is_bare_control(text: &str) -> bool {
    let w: Vec<&str> = text.split_whitespace().collect();
    matches!(w.as_slice(), ["repeat"] | ["while"] | ["for"] | ["if"] | ["else"] | ["else", "if"])
        || matches!(w.as_slice(), ["for", "each", v] if is_ident(v))
}

/// The generated word for a control structure, when the intent has no text for it.
fn default_control_word(c: Ctl) -> &'static str {
    match c {
        Ctl::Loop => "repeat",
        Ctl::If => "if",
        Ctl::ElseIf => "else if",
        Ctl::Else => "else",
    }
}

/// Builds a def's outline from its statements. Steps are flat markers in the
/// AST (a step is a block, not a scope), so the open steps are a stack.
struct Outliner<'a> {
    defs: &'a HashSet<String>,
    li: &'a LineIndex,
}

impl Outliner<'_> {
    fn stmts(&self, body: &[Stmt]) -> Vec<ONode> {
        let mut root: Vec<ONode> = Vec::new();
        let mut open: Vec<ONode> = Vec::new();
        for s in body {
            let push = |open: &mut Vec<ONode>, root: &mut Vec<ONode>, n: ONode| match open.last_mut() {
                Some(top) => top.children.push(n),
                None => root.push(n),
            };
            match s {
                Stmt::StepEnter { id, text, span, .. } => {
                    let l = self.li.line(span.start);
                    open.push(ONode { kind: OKind::Step(id.clone()), text: text.clone(), line: l, end: l, children: vec![] });
                }
                Stmt::StepExit { span, .. } => {
                    if let Some(mut n) = open.pop() {
                        n.end = self.li.line(span.start);
                        push(&mut open, &mut root, n);
                    }
                }
                other => {
                    for n in self.stmt(other) {
                        push(&mut open, &mut root, n);
                    }
                }
            }
        }
        // a step left open cannot happen in a parsed program; keep its nodes anyway
        while let Some(n) = open.pop() {
            match open.last_mut() {
                Some(top) => top.children.push(n),
                None => root.push(n),
            }
        }
        root
    }

    fn stmt(&self, s: &Stmt) -> Vec<ONode> {
        let line = self.li.line(s.span().start);
        let end = self.li.line(s.span().end.saturating_sub(1));
        let mut calls = Vec::new();
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => {
                self.expr(value, line, &mut calls)
            }
            Stmt::SetIndex { base, indices, value, .. } => {
                self.expr(base, line, &mut calls);
                for i in indices {
                    self.expr(i, line, &mut calls);
                }
                self.expr(value, line, &mut calls);
            }
            Stmt::SetField { base, value, .. } => {
                self.expr(base, line, &mut calls);
                self.expr(value, line, &mut calls);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.expr(cond, line, &mut calls);
                let mut out = calls;
                out.push(ONode { kind: OKind::Control(Ctl::If), text: "if".into(), line, end, children: self.stmts(then_body) });
                self.else_chain(else_body, &mut out);
                return out;
            }
            Stmt::While { cond, body, .. } => {
                self.expr(cond, line, &mut calls);
                let mut out = calls;
                out.push(ONode { kind: OKind::Control(Ctl::Loop), text: "repeat".into(), line, end, children: self.stmts(body) });
                return out;
            }
            Stmt::For { var, bound, body, .. } => {
                self.expr(bound, line, &mut calls);
                let mut out = calls;
                out.push(ONode { kind: OKind::Control(Ctl::Loop), text: format!("for each {var}"), line, end, children: self.stmts(body) });
                return out;
            }
            Stmt::Test { name, body, .. } => {
                return vec![ONode { kind: OKind::Test(name.clone()), text: format!("test \"{name}\""), line, end, children: self.stmts(body) }];
            }
            _ => {}
        }
        calls
    }

    /// `else` bodies: an `else if` is an `If` alone in the else body.
    fn else_chain(&self, else_body: &[Stmt], out: &mut Vec<ONode>) {
        if else_body.is_empty() {
            return;
        }
        if let [Stmt::If { cond, then_body, else_body: rest, span }] = else_body {
            let line = self.li.line(span.start);
            let end = self.li.line(span.end.saturating_sub(1));
            // the condition's calls run only when this branch is reached
            let mut children = Vec::new();
            self.expr(cond, line, &mut children);
            children.extend(self.stmts(then_body));
            out.push(ONode { kind: OKind::Control(Ctl::ElseIf), text: "else if".into(), line, end, children });
            self.else_chain(rest, out);
            return;
        }
        let line = self.li.line(else_body[0].span().start);
        let end = self.li.line(else_body[else_body.len() - 1].span().end.saturating_sub(1));
        out.push(ONode { kind: OKind::Control(Ctl::Else), text: "else".into(), line, end, children: self.stmts(else_body) });
    }

    fn expr(&self, e: &Expr, line: usize, out: &mut Vec<ONode>) {
        let call = |name: &str, value: bool, out: &mut Vec<ONode>| {
            out.push(ONode { kind: OKind::Call { name: name.to_string(), value }, text: String::new(), line, end: line, children: vec![] });
        };
        match e {
            Expr::Call { callee, args, .. } => {
                for a in args {
                    self.expr(a, line, out);
                }
                if self.defs.contains(callee) {
                    call(callee, false, out);
                }
            }
            Expr::Var { name, .. } => {
                if self.defs.contains(name) {
                    call(name, true, out);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs, line, out);
                self.expr(rhs, line, out);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.expr(operand, line, out),
            Expr::Index { base, indices, .. } => {
                self.expr(base, line, out);
                for i in indices {
                    self.expr(i, line, out);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for el in elems {
                    self.expr(el, line, out);
                }
            }
            Expr::Lambda { body, .. } => self.expr(body, line, out),
            Expr::Field { base, .. } => self.expr(base, line, out),
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    self.expr(v, line, out);
                }
            }
            _ => {}
        }
    }
}

fn count_stmts(body: &[Stmt]) -> usize {
    body.iter()
        .map(|s| match s {
            Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => 0,
            Stmt::If { then_body, else_body, .. } => 1 + count_stmts(then_body) + count_stmts(else_body),
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => 1 + count_stmts(body),
            _ => 1,
        })
        .sum()
}

/// An annotation line's kind: 1-3 for `#|`..`#|||`, 4 for `#:`, 5 for `#@`,
/// 6 for the other annotations the doc scanner skips (`#~`, `#[`, `#]`).
fn annotation(line: &str) -> Option<u8> {
    let t = line.trim_start();
    if t.starts_with("#|||") {
        Some(3)
    } else if t.starts_with("#||") {
        Some(2)
    } else if t.starts_with("#|") {
        Some(1)
    } else if t.starts_with("#:") {
        Some(4)
    } else if t.starts_with("#@") {
        Some(5)
    } else if t.starts_with("#~") || t.starts_with("#[") || t.starts_with("#]") {
        Some(6)
    } else {
        None
    }
}

/// Which raw lines (0-based) are inside a `###` fence, fences included: prose,
/// not code. The lexer's rule: a line that is `###` opens a fence that runs to
/// the next such line.
fn fenced(lines: &[&str]) -> Vec<bool> {
    let mut out = vec![false; lines.len()];
    let mut inside = false;
    for (i, l) in lines.iter().enumerate() {
        let fence = l.trim() == "###";
        out[i] = inside || fence;
        if fence {
            inside = !inside;
        }
    }
    out
}

/// A parsed module: its defs (and `<main>`, when it has statements), the
/// names of its tests, and its `.intent` file.
#[derive(Clone, Debug)]
pub struct Mod {
    pub text: ModuleText,
    pub defs: Vec<CodeDef>,
    pub tests: Vec<String>,
    pub intent: Option<IntentFile>,
    pub intent_errors: Vec<ParseError>,
}

impl Mod {
    pub fn def(&self, name: &str) -> Option<&CodeDef> {
        self.defs.iter().find(|d| d.name == name)
    }
    pub fn entries(&self) -> (Option<ModuleEntry>, Vec<Entry>) {
        self.intent.as_ref().map(|f| f.model()).unwrap_or((None, Vec::new()))
    }
}

/// The names of every `test "..."` block in a source text.
pub fn test_names_of(src: &str) -> Vec<String> {
    test_names(src)
}

fn test_names(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for l in src.lines() {
        let Some(rest) = l.trim_start().strip_prefix("test") else { continue };
        let rest = rest.trim_start();
        if !l.trim_start()[4..].starts_with(char::is_whitespace) {
            continue;
        }
        let Some(rest) = rest.strip_prefix('"') else { continue };
        let mut name = String::new();
        let mut chars = rest.chars();
        let mut closed = false;
        while let Some(c) = chars.next() {
            match c {
                '"' => {
                    closed = true;
                    break;
                }
                '\\' => match chars.next() {
                    Some('n') => name.push('\n'),
                    Some('t') => name.push('\t'),
                    Some(o) => name.push(o),
                    None => {}
                },
                c => name.push(c),
            }
        }
        if closed {
            out.push(name);
        }
    }
    out
}

/// Parse a module's own text into its defs. `all_defs` is every def name in
/// the program, so a call to a def of another module is seen as a call;
/// `hashes` maps def names to body hashes.
fn load_mod(text: &ModuleText, all_defs: &HashSet<String>, hashes: &HashMap<String, String>) -> Result<Mod, String> {
    let prog = crate::parse_keeping_rungs(&text.src, SourceId(0)).map_err(|ds| format!("{}: {}", text.path, crate::join_errs(&ds)))?;
    let li = LineIndex::new(&text.src);
    let ol = Outliner { defs: all_defs, li: &li };
    let raw: Vec<&str> = text.src.lines().collect();
    let sc = crate::scan(&text.src);
    let fence = fenced(&raw);
    let mut defs = Vec::new();
    let mut main_body: Vec<Stmt> = Vec::new();
    for item in &prog.items {
        match item {
            // a lambda body's def is part of the lambda (crate::surface)
            Item::Def(d) if crate::update::is_expression_def(&d.name) => {}
            Item::Def(d) => {
                let line = li.line(d.span.start);
                let end = li.line(d.span.end.saturating_sub(1));
                let sig_line = raw.get(line - 1).copied().unwrap_or("").trim();
                let sig = sig_line.trim_start_matches("def ").split('{').next().unwrap_or("").trim_end().to_string();
                let outline = ol.stmts(&d.body);
                let mut cd = CodeDef {
                    name: d.name.clone(),
                    sig,
                    line,
                    end,
                    hash: hashes.get(&d.name).cloned(),
                    stmts: count_stmts(&d.body),
                    outline,
                    inline: Inline::default(),
                    placeholder: matches!(d.body.as_slice(), [Stmt::Lens { lens, .. }] if matches!(**lens, crate::ast::Lens::Placeholder { .. })),
                };
                cd.inline = inline_of(&cd, &raw, &sc, &fence);
                defs.push(cd);
            }
            Item::Stmt(s) => main_body.push(s.clone()),
            Item::Record(_) | Item::Enum(_) | Item::Rung(_) => {}
        }
    }
    if !main_body.is_empty() {
        let outline = ol.stmts(&main_body);
        let mut cd = CodeDef {
            name: MAIN_DEF.to_string(),
            sig: MAIN_DEF.to_string(),
            line: 0,
            end: 0,
            hash: hashes.get(MAIN_DEF).cloned(),
            stmts: count_stmts(&main_body),
            outline,
            inline: Inline::default(),
            placeholder: false,
        };
        cd.inline.controls = control_notes(&cd.outline, &raw);
        defs.push(cd);
    }
    let (intent, intent_errors) = match &text.intent {
        Some(t) => match IntentFile::parse(t) {
            Ok(f) => (Some(f), Vec::new()),
            Err(e) => (None, e),
        },
        None => (None, Vec::new()),
    };
    Ok(Mod { text: text.clone(), defs, tests: test_names(&text.src), intent, intent_errors })
}

/// A def's inline intent, read the way the doc tools read it: `#|` notes
/// attach to the next non-blank code line, so the def's doc is the run of
/// annotation and blank lines directly above it.
fn inline_of(d: &CodeDef, raw: &[&str], sc: &crate::Scan<'_>, fence: &[bool]) -> Inline {
    let mut inl = Inline::default();
    // the def's line in the scanner's annotation-free numbering
    let code_line = sc.raw_of.iter().position(|&r| r == d.line).map(|k| k + 1);
    if let Some(cl) = code_line {
        if let Some(sp) = sc.spans.iter().find(|sp| sp.start == cl) {
            for level in 1..=3u8 {
                // the doc tools keep the last note of each level
                if let Some(n) = sc.notes.iter().rev().find(|n| n.line == cl && n.level == level) {
                    inl.fields.push((FIELDS[level as usize - 1].to_string(), n.text.clone(), 0));
                }
            }
            for (l, ts) in &sc.tags_at {
                if *l == cl {
                    inl.tags.extend(ts.iter().cloned());
                }
            }
            for (l, n, desc) in &sc.vars_at {
                if *l >= sp.start && *l <= sp.end {
                    inl.gloss.push((n.clone(), desc.clone(), 0));
                }
            }
        }
    }
    // where those lines are: above the def, up to the previous code line
    let mut k = d.line - 1; // 0-based index of the def line
    let mut above: Vec<usize> = Vec::new();
    while k > 0 {
        let l = raw[k - 1];
        if fence.get(k - 1).copied().unwrap_or(false) {
            break;
        }
        if l.trim().is_empty() || annotation(l).is_some() {
            above.push(k);
            k -= 1;
        } else {
            break;
        }
    }
    above.reverse();
    for &r in &above {
        match annotation(raw[r - 1]) {
            Some(a @ 1..=3) => {
                inl.lines.push(r);
                if let Some(f) = inl.fields.iter_mut().find(|f| f.0 == FIELDS[a as usize - 1] && f.2 == 0) {
                    f.2 = r;
                }
            }
            Some(4) => inl.lines.push(r),
            Some(5) => {
                inl.lines.push(r);
                let h = raw[r - 1].trim_start()[2..].trim().to_string();
                inl.stamp = Some((h, r));
            }
            _ => {}
        }
    }
    // glossary lines inside the body
    for r in d.line..=d.end.min(raw.len()) {
        if annotation(raw[r - 1]) == Some(4) {
            inl.lines.push(r);
        }
    }
    // the first raw line of each glossary entry, in order
    let gl: Vec<usize> = inl.lines.iter().copied().filter(|&r| annotation(raw[r - 1]) == Some(4) && !raw[r - 1].trim_start()[2..].starts_with("  ")).collect();
    for (g, r) in inl.gloss.iter_mut().zip(gl) {
        g.2 = r;
    }
    inl.controls = control_notes(&d.outline, raw);
    inl
}

/// `#|` notes directly above a loop or an `if` that stays in the outline:
/// the inline home of a control line's text.
fn control_notes(outline: &[ONode], raw: &[&str]) -> Vec<(usize, String, Vec<usize>)> {
    let mut out = Vec::new();
    fn go(ns: &[ONode], raw: &[&str], out: &mut Vec<(usize, String, Vec<usize>)>) {
        let keep = kept_controls(ns, octl, |n| has_step(&n.children));
        for (n, keep) in ns.iter().zip(keep) {
            if matches!(n.kind, OKind::Control(Ctl::Loop) | OKind::Control(Ctl::If)) && keep && n.line >= 2 {
                let mut k = n.line - 1;
                let mut lines = Vec::new();
                while k >= 1 && annotation(raw[k - 1]) == Some(1) {
                    lines.push(k);
                    k -= 1;
                }
                if !lines.is_empty() {
                    lines.reverse();
                    let text = lines.iter().map(|&r| raw[r - 1].trim_start()[2..].trim()).collect::<Vec<_>>().join(" ");
                    out.push((n.line, text, lines));
                }
            }
            go(&n.children, raw, out);
        }
    }
    go(outline, raw, &mut out);
    out
}

/// The word the code gives the control statement at a raw line.
fn code_control_text(d: &CodeDef, line: usize) -> Option<String> {
    fn go(ns: &[ONode], line: usize) -> Option<String> {
        ns.iter().find_map(|n| if matches!(n.kind, OKind::Control(_)) && n.line == line { Some(n.text.clone()) } else { go(&n.children, line) })
    }
    go(&d.outline, line)
}

/// Which control nodes of a list stay in an outline: those that hold a
/// step. An `if` and the `else if` and `else` lines after it are one
/// structure, and all of them stay when any of them holds a step.
fn kept_controls<T>(ns: &[T], ctl: impl Fn(&T) -> Option<Ctl>, holds: impl Fn(&T) -> bool) -> Vec<bool> {
    let mut keep: Vec<bool> = ns.iter().map(|n| ctl(n).is_some() && holds(n)).collect();
    let mut i = 0;
    while i < ns.len() {
        if ctl(&ns[i]) == Some(Ctl::If) {
            let mut j = i + 1;
            while j < ns.len() && matches!(ctl(&ns[j]), Some(Ctl::ElseIf) | Some(Ctl::Else)) {
                j += 1;
            }
            let any = keep[i..j].iter().any(|k| *k);
            for k in &mut keep[i..j] {
                *k = any;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    keep
}

fn octl(n: &ONode) -> Option<Ctl> {
    if let OKind::Control(c) = n.kind { Some(c) } else { None }
}

fn nctl(n: &Node) -> Option<Ctl> {
    if let NodeKind::Control(c) = n.kind { Some(c) } else { None }
}

fn has_step(ns: &[ONode]) -> bool {
    ns.iter().any(|n| matches!(n.kind, OKind::Step(_)) || has_step(&n.children))
}

/// A program loaded for the lens tools: the root module and the modules it uses.
pub struct Loaded {
    pub root: Mod,
    pub used: Vec<Mod>,
    /// why the body hashes are missing, when the program does not build
    pub hash_error: Option<String>,
    /// what each def calls, from `deps_json` (the `viv deps` graph), with the
    /// top level's calls under `<main>`: the call graph `explain` walks
    pub calls: HashMap<String, Vec<String>>,
    /// the program with every `#! use` expanded, for facts that need all of
    /// it (which defs write into which arguments)
    pub expanded: String,
    /// the tests of the module's companion test file (see `Input`)
    pub companion_tests: Vec<String>,
}

impl Loaded {
    /// A def anywhere in the program, and the module it belongs to. The
    /// module's own defs win over a used module's.
    pub fn find(&self, name: &str) -> Option<(&Mod, &CodeDef)> {
        std::iter::once(&self.root).chain(self.used.iter()).find_map(|m| m.def(name).map(|d| (m, d)))
    }
}

impl Loaded {
    /// The parse errors of every `.intent` file the program reaches, as
    /// `path:line: message` lines.
    pub fn intent_errors(&self) -> Vec<String> {
        std::iter::once(&self.root)
            .chain(self.used.iter())
            .flat_map(|m| m.intent_errors.iter().map(move |e| format!("{}:{}: {}", m.text.intent_path, e.line, e.message)))
            .collect()
    }
}

/// Parse the module and every module it uses, and hash the program's defs.
pub fn load(input: &Input) -> Result<Loaded, String> {
    let (hashes, hash_error) = match crate::def_hashes(&input.expanded) {
        Ok(hs) => (
            hs.into_iter()
                .filter(|h| h.kind == "def" || h.kind == "main")
                .map(|h| (h.name, h.body_short))
                .collect::<HashMap<_, _>>(),
            None,
        ),
        Err(e) => (HashMap::new(), Some(e)),
    };
    // the def names of every module, so calls across modules are seen
    let mut all_defs: HashSet<String> = HashSet::new();
    for m in std::iter::once(&input.root).chain(input.used.iter()) {
        if let Ok(p) = crate::parse_keeping_rungs(&m.src, SourceId(0)) {
            for it in &p.items {
                if let Item::Def(d) = it {
                    all_defs.insert(d.name.clone());
                }
            }
        }
    }
    let root = load_mod(&input.root, &all_defs, &hashes)?;
    // a used module's hashes are not the module's own business here, and its
    // `<main>` is not part of this program
    let none = HashMap::new();
    let mut used = Vec::new();
    for m in &input.used {
        let mut md = load_mod(m, &all_defs, &none)?;
        md.defs.retain(|d| d.name != MAIN_DEF);
        used.push(md);
    }
    Ok(Loaded { root, used, hash_error, calls: deps_calls(&input.expanded), expanded: input.expanded.clone(), companion_tests: input.companion_tests.clone() })
}

/// The call lists of `deps_json`, read back from its JSON (names never hold
/// quotes). Empty when the program does not parse.
fn deps_calls(expanded: &str) -> HashMap<String, Vec<String>> {
    let Ok(dj) = crate::deps_json(expanded) else { return HashMap::new() };
    let list = |s: &str| -> Vec<String> {
        s.chars().take_while(|c| *c != ']').collect::<String>().split(',').filter(|x| !x.is_empty()).map(|x| x.trim_matches('"').to_string()).collect()
    };
    let mut out = HashMap::new();
    for cap in dj.split("{\"name\":\"").skip(1) {
        let name: String = cap.chars().take_while(|c| *c != '"').collect();
        out.insert(name, cap.split("\"calls\":[").nth(1).map(list).unwrap_or_default());
    }
    out.insert(MAIN_DEF.to_string(), dj.split("\"main_calls\":[").nth(1).map(list).unwrap_or_default());
    out
}

// ---------------------------------------------------------------------------
// The intent of a def, from both homes
// ---------------------------------------------------------------------------

/// Where an entry lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    Inline,
    File,
}

impl Home {
    fn name(self) -> &'static str {
        match self {
            Home::Inline => "inline",
            Home::File => "file",
        }
    }
}

/// A def's intent with both homes merged. Where both homes hold the same
/// anchor (which `lens check` reports), the file's wins.
#[derive(Clone, Debug, Default)]
pub struct Merged {
    pub fields: Vec<(String, String, Home)>,
    pub gloss: Vec<(String, String, Home)>,
    /// the outline: the file's if it has one, else built from the code
    pub outline: Vec<Node>,
    pub stamp_file: Option<String>,
    pub stamp_inline: Option<String>,
    /// the file's line anchors outside every step
    pub details: Vec<Detail>,
}

impl Merged {
    pub fn field(&self, label: &str) -> Option<&str> {
        self.fields.iter().find(|f| f.0 == label).map(|f| f.1.as_str())
    }
}

/// The code outline as an intent outline: steps and the control structures
/// that hold steps, with the texts the intent gives them.
fn code_as_intent(d: &CodeDef, step_text: &dyn Fn(&str) -> String, control_text: &dyn Fn(usize) -> Option<String>) -> Vec<Node> {
    fn go(ns: &[ONode], st: &dyn Fn(&str) -> String, ct: &dyn Fn(usize) -> Option<String>) -> Vec<Node> {
        let mut out = Vec::new();
        let keep = kept_controls(ns, octl, |n| has_step(&n.children));
        for (n, keep) in ns.iter().zip(keep) {
            match &n.kind {
                OKind::Step(id) => out.push(Node { kind: NodeKind::Step(id.clone()), text: st(id), line: n.line, children: go(&n.children, st, ct), details: vec![] }),
                OKind::Control(c) if keep => {
                    let text = ct(n.line).unwrap_or_else(|| n.text.clone());
                    out.push(Node { kind: NodeKind::Control(*c), text, line: n.line, children: go(&n.children, st, ct), details: vec![] })
                }
                // a test or a control holding no step: its steps (if any) are listed where they are
                _ => out.extend(go(&n.children, st, ct)),
            }
        }
        out
    }
    go(&d.outline, step_text, control_text)
}

/// A step's text from either home: the file's if it has one, else the code's.
pub fn file_step_texts(e: Option<&Entry>) -> HashMap<String, String> {
    let mut m = HashMap::new();
    fn go(ns: &[Node], m: &mut HashMap<String, String>) {
        for n in ns {
            if let NodeKind::Step(id) = &n.kind {
                if !n.text.is_empty() {
                    m.entry(id.clone()).or_insert_with(|| n.text.clone());
                }
            }
            go(&n.children, m);
        }
    }
    if let Some(e) = e {
        go(&e.outline, &mut m);
    }
    m
}

/// Pair each code control that holds steps with the file control at the same
/// place, when the two outlines have the same shape: code line -> (file line,
/// the file's text).
fn control_map(code: &[Node], file: &[Node]) -> HashMap<usize, (usize, String)> {
    let mut out = HashMap::new();
    fn prune(ns: &[Node]) -> Vec<&Node> {
        let keep = kept_controls(ns, nctl, |n| node_has_step(&n.children));
        ns.iter().zip(keep).filter(|(n, k)| matches!(n.kind, NodeKind::Step(_)) || *k).map(|(n, _)| n).collect()
    }
    fn go(code: &[Node], file: &[Node], out: &mut HashMap<usize, (usize, String)>) {
        let (c, f) = (prune(code), prune(file));
        if c.len() != f.len() {
            return;
        }
        for (a, b) in c.iter().zip(f.iter()) {
            match (&a.kind, &b.kind) {
                (NodeKind::Control(x), NodeKind::Control(y)) if x == y => {
                    out.insert(a.line, (b.line, b.text.clone()));
                    go(&a.children, &b.children, out);
                }
                (NodeKind::Step(x), NodeKind::Step(y)) if x == y => go(&a.children, &b.children, out),
                _ => return,
            }
        }
    }
    go(code, file, &mut out);
    out
}

fn node_has_step(ns: &[Node]) -> bool {
    ns.iter().any(|n| matches!(n.kind, NodeKind::Step(_)) || node_has_step(&n.children))
}

/// The intent of one def, both homes merged.
pub fn merged(d: &CodeDef, e: Option<&Entry>) -> Merged {
    let mut m = Merged::default();
    for label in FIELDS {
        let file = e.and_then(|e| e.field(label));
        match (file, d.inline.field(label)) {
            (Some(t), _) if !t.text.is_empty() => m.fields.push((label.to_string(), t.text.clone(), Home::File)),
            (_, Some(f)) => m.fields.push((label.to_string(), f.1.clone(), Home::Inline)),
            // an empty field in the file (a skeleton's) is shown empty, to be filled
            (Some(t), None) => m.fields.push((label.to_string(), t.text.clone(), Home::File)),
            (None, None) => {}
        }
    }
    if let Some(e) = e {
        for g in &e.gloss {
            m.gloss.push((g.key.clone(), g.text.clone(), Home::File));
        }
    }
    for (n, t, _) in &d.inline.gloss {
        if !m.gloss.iter().any(|g| &g.0 == n) {
            m.gloss.push((n.clone(), t.clone(), Home::Inline));
        }
    }
    let fst = file_step_texts(e);
    let step_text = |id: &str| -> String {
        fst.get(id).cloned().or_else(|| d.step(id).map(|n| n.text.clone())).unwrap_or_default()
    };
    let inline_ctl: HashMap<usize, String> = d.inline.controls.iter().map(|(l, t, _)| (*l, t.clone())).collect();
    m.outline = match e.filter(|e| !e.outline.is_empty()) {
        Some(e) => {
            // the file's outline, with step texts the file leaves empty taken from the code
            fn fill(ns: &mut [Node], st: &dyn Fn(&str) -> String) {
                for n in ns {
                    if let NodeKind::Step(id) = &n.kind {
                        if n.text.is_empty() {
                            n.text = st(id);
                        }
                    }
                    fill(&mut n.children, st);
                }
            }
            let mut o = e.outline.clone();
            fill(&mut o, &step_text);
            // a bare control line in the file takes the text of a note above its statement
            let cmap = control_map(&code_as_intent(d, &|_| String::new(), &|_| None), &e.outline);
            let by_file_line: HashMap<usize, String> = cmap.iter().filter_map(|(cl, (fl, _))| inline_ctl.get(cl).map(|t| (*fl, t.clone()))).collect();
            fn fill_ctl(ns: &mut [Node], by: &HashMap<usize, String>) {
                for n in ns {
                    if matches!(n.kind, NodeKind::Control(_)) && is_bare_control(&n.text) {
                        if let Some(t) = by.get(&n.line) {
                            n.text = t.clone();
                        }
                    }
                    fill_ctl(&mut n.children, by);
                }
            }
            fill_ctl(&mut o, &by_file_line);
            o
        }
        None => code_as_intent(d, &step_text, &|l| inline_ctl.get(&l).cloned()),
    };
    m.stamp_file = e.and_then(|e| e.stamp.clone());
    m.details = e.map(|e| e.details.clone()).unwrap_or_default();
    m.stamp_inline = d.inline.stamp.as_ref().map(|s| s.0.clone());
    m
}

/// The intent's words for the story lens: every def's `what` and every
/// step's text (the `.intent` file's where it has one, else the code's).
pub fn story_words(l: &Loaded) -> crate::story::Words {
    let mut w = crate::story::Words::default();
    for m in std::iter::once(&l.root).chain(l.used.iter()) {
        let (_, entries) = m.entries();
        for d in &m.defs {
            let e = entries.iter().find(|e| e.name == d.name);
            let mg = merged(d, e);
            if let Some(what) = mg.field("what") {
                w.what.entry(d.name.clone()).or_insert_with(|| what.to_string());
            }
            fn go(ns: &[Node], def: &str, out: &mut HashMap<String, String>) {
                for n in ns {
                    if let NodeKind::Step(id) = &n.kind {
                        out.entry(format!("{def}/{id}")).or_insert_with(|| n.text.clone());
                    }
                    go(&n.children, def, out);
                }
            }
            go(&mg.outline, &d.name, &mut w.steps);
            if d.placeholder {
                w.unwritten.push(d.name.clone());
            }
        }
    }
    w
}

// ---------------------------------------------------------------------------
// lens init
// ---------------------------------------------------------------------------

/// The skeleton `.intent` of a module: a block for every def with its current
/// body hash, every step id and the control structure around the steps, and
/// no text. Anchors that already have an inline home are left out, so the
/// skeleton never gives an anchor a second home; a step whose text is inline
/// is still listed, since the outline is the def's structure.
pub fn init(l: &Loaded) -> String {
    let m = &l.root;
    let module = ModuleEntry { name: m.text.name.clone(), text: String::new(), line: 0, every_var: false };
    let mut blocks = Vec::new();
    for d in &m.defs {
        if d.name == MAIN_DEF && d.steps().is_empty() {
            continue;
        }
        let fields = if d.name == MAIN_DEF {
            Vec::new()
        } else {
            // `what` is the field every public def must have; `how` and `why` are the author's call
            if d.inline.field("what").is_none() { vec![("what".to_string(), String::new())] } else { Vec::new() }
        };
        let outline = code_as_intent(d, &|_| String::new(), &|_| None);
        blocks.push(Block { name: d.name.clone(), stamp: d.hash.clone(), by: None, fields, gloss: Vec::new(), outline, details: Vec::new() });
    }
    render_file(Some(&module), &blocks, true)
}

// ---------------------------------------------------------------------------
// view
// ---------------------------------------------------------------------------

/// The code lens: the source without its intent lines (`#|`, `#||`, `#|||`,
/// `#:`, `#@`). The post header between `###` fences is left as it is.
pub fn view_code(src: &str) -> String {
    let raw: Vec<&str> = src.lines().collect();
    let fence = fenced(&raw);
    let mut out = String::new();
    // an inline `debug { }` block is the debug lens, not code
    let mut debug_depth: i64 = 0;
    for (i, l) in raw.iter().enumerate() {
        if !fence[i] && matches!(annotation(l), Some(1..=5)) {
            continue;
        }
        if !fence[i] && (debug_depth > 0 || crate::is_debug_open(l.trim_start())) {
            debug_depth += crate::brace_change(l);
            continue;
        }
        out.push_str(l);
        out.push('\n');
    }
    if !src.ends_with('\n') && out.ends_with('\n') {
        out.pop();
    }
    out
}

/// The intent lens: every def's intent, from whichever home holds it, as
/// pseudocode in the `.intent` layout without the stamps.
pub fn view_intent(l: &Loaded) -> String {
    let m = &l.root;
    let (module, entries) = m.entries();
    let blocks: Vec<Block> = m
        .defs
        .iter()
        .filter(|d| d.name != MAIN_DEF || !d.steps().is_empty() || entries.iter().any(|e| e.name == MAIN_DEF))
        .map(|d| {
            let mg = merged(d, entries.iter().find(|e| e.name == d.name));
            Block {
                name: d.name.clone(),
                stamp: None,
                by: None,
                fields: mg.fields.iter().map(|f| (f.0.clone(), f.1.clone())).collect(),
                gloss: mg.gloss.iter().map(|g| (g.0.clone(), g.1.clone())).collect(),
                outline: mg.outline,
                details: mg.details,
            }
        })
        .collect();
    render_file(module.as_ref(), &blocks, false)
}

/// The `all` lens: the code with every entry inline next to what it explains:
/// the def's intent above it, each step's text in its header, each control
/// line above its statement.
pub fn view_all(l: &Loaded) -> String {
    let (src, _, _) = to_inline(l, None, true);
    src
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", crate::esc(s))
}

fn json_nodes(ns: &[Node], code: Option<&CodeDef>, raw: &[&str]) -> String {
    let items: Vec<String> = ns
        .iter()
        .map(|n| {
            let (kind, id) = match &n.kind {
                NodeKind::Step(id) => ("step", id.clone()),
                NodeKind::Control(c) => ("control", c.sig().to_string()),
                NodeKind::Note => ("note", String::new()),
            };
            let mut s = format!("{{\"kind\":\"{kind}\"");
            if kind == "step" {
                s.push_str(&format!(",\"id\":{}", json_str(&id)));
                if let Some(st) = code.and_then(|d| d.step(&id)) {
                    let body: Vec<&str> = raw[st.line.saturating_sub(1)..st.end.min(raw.len())].iter().copied().filter(|l| !matches!(annotation(l), Some(1..=5))).collect();
                    s.push_str(&format!(",\"line\":{},\"end\":{},\"code\":{}", st.line, st.end, json_str(&body.join("\n"))));
                }
            } else if kind == "control" {
                s.push_str(&format!(",\"control\":{}", json_str(&id)));
            }
            s.push_str(&format!(",\"text\":{},\"children\":{}}}", json_str(&n.text), json_nodes(&n.children, code, raw)));
            s
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// `view --json`: for each def, its intent (merged, with the home of each
/// entry) and its code, and for each step its text and its code, plus the
/// rendered text of the lens asked for.
pub fn view_json(l: &Loaded, lens: &str, text: &str) -> String {
    let m = &l.root;
    let (module, entries) = m.entries();
    let raw: Vec<&str> = m.text.src.lines().collect();
    let defs: Vec<String> = m
        .defs
        .iter()
        .map(|d| {
            let e = entries.iter().find(|e| e.name == d.name);
            let mg = merged(d, e);
            let fields: Vec<String> = mg.fields.iter().map(|(k, t, h)| format!("{}:{{\"text\":{},\"home\":\"{}\"}}", json_str(k), json_str(t), h.name())).collect();
            let gloss: Vec<String> = mg.gloss.iter().map(|(k, t, h)| format!("{{\"name\":{},\"text\":{},\"home\":\"{}\"}}", json_str(k), json_str(t), h.name())).collect();
            let code = if d.line > 0 {
                raw[d.line - 1..d.end.min(raw.len())].iter().copied().filter(|l| !matches!(annotation(l), Some(1..=5))).collect::<Vec<_>>().join("\n")
            } else {
                String::new()
            };
            let opt = |o: &Option<String>| o.as_ref().map(|s| json_str(s)).unwrap_or_else(|| "null".into());
            format!(
                "{{\"name\":{},\"sig\":{},\"line\":{},\"end\":{},\"hash\":{},\"stamp\":{{\"file\":{},\"inline\":{}}},\"intent\":{{{}}},\"glossary\":[{}],\"outline\":{},\"code\":{}}}",
                json_str(&d.name),
                json_str(&d.sig),
                d.line,
                d.end,
                opt(&d.hash),
                opt(&mg.stamp_file),
                opt(&mg.stamp_inline),
                fields.join(","),
                gloss.join(","),
                json_nodes(&mg.outline, Some(d), &raw),
                json_str(&code)
            )
        })
        .collect();
    let module_json = match &module {
        Some(me) => format!("{{\"name\":{},\"text\":{}}}", json_str(&me.name), json_str(&me.text)),
        None => "null".into(),
    };
    format!(
        "{{\"module\":{},\"path\":{},\"intentFile\":{},\"lens\":{},\"entry\":{},\"defs\":[{}],\"text\":{}}}",
        json_str(&m.text.name),
        json_str(&m.text.path),
        if m.text.intent.is_some() { json_str(&m.text.intent_path) } else { "null".into() },
        json_str(lens),
        module_json,
        defs.join(","),
        json_str(text)
    )
}

// ---------------------------------------------------------------------------
// Moving entries between homes, and stamping
// ---------------------------------------------------------------------------

/// Edits to a source text, applied bottom-up so line numbers stay valid.
#[derive(Default)]
struct Edits {
    /// (raw line 1-based, lines to insert above it)
    insert: Vec<(usize, Vec<String>)>,
    replace: HashMap<usize, String>,
    delete: HashSet<usize>,
}

impl Edits {
    fn apply(&self, src: &str) -> String {
        let mut lines: Vec<String> = src.lines().map(|s| s.to_string()).collect();
        let n = lines.len();
        let mut out: Vec<String> = Vec::with_capacity(n);
        for (i, l) in lines.drain(..).enumerate() {
            let r = i + 1;
            for (at, ins) in &self.insert {
                if *at == r {
                    out.extend(ins.iter().cloned());
                }
            }
            if self.delete.contains(&r) {
                continue;
            }
            out.push(self.replace.get(&r).cloned().unwrap_or(l));
        }
        let mut s = out.join("\n");
        if src.ends_with('\n') || src.is_empty() {
            s.push('\n');
        }
        s
    }
}

fn quote(s: &str) -> String {
    let mut q = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => q.push_str("\\\""),
            '\\' => q.push_str("\\\\"),
            '\n' => q.push_str("\\n"),
            '\t' => q.push_str("\\t"),
            c => q.push(c),
        }
    }
    q.push('"');
    q
}

/// A step header line with its text replaced (`None` removes it).
fn set_step_text(line: &str, text: Option<&str>) -> Option<String> {
    let (_, _, brace) = crate::step_header(line)?;
    let lead = &line[..line.len() - line.trim_start().len()];
    let rest = line.trim_start()["step".len()..].trim_start();
    let id: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    Some(match text {
        Some(t) if !t.is_empty() => format!("{lead}step {id} {} {}", quote(t), &line[brace..]),
        _ => format!("{lead}step {id} {}", &line[brace..]),
    })
}

/// `#|`-style lines for a text, wrapped, at an indentation.
fn comment_lines(prefix: &str, text: &str, lead: &str) -> Vec<String> {
    wrap(text, 88usize.saturating_sub(lead.len() + prefix.len() + 1).max(30)).into_iter().map(|p| format!("{lead}{prefix} {p}")).collect()
}

/// The result of moving entries: the new source, the new `.intent` text
/// (`None` when nothing is left in it), and a line for each entry that
/// could not move and why.
pub type Moved = (String, Option<String>, Vec<String>);

/// Move file entries inline. With `for_view`, stamps are left out and an
/// anchor written in both homes is shown once: this is the `all` lens.
pub fn to_inline(l: &Loaded, only: Option<&[String]>, for_view: bool) -> Moved {
    let m = &l.root;
    let src = &m.text.src;
    let raw: Vec<&str> = src.lines().collect();
    let (module, entries) = m.entries();
    let mut ed = Edits::default();
    let mut kept: Vec<String> = Vec::new();
    // entry index -> what is left of it in the file (None: nothing)
    let mut remaining: HashMap<usize, Option<Block>> = HashMap::new();
    fn walk(ns: &mut [Node], f: &mut dyn FnMut(&mut Node)) {
        for n in ns {
            f(n);
            walk(&mut n.children, f);
        }
    }
    for (ei, e) in entries.iter().enumerate() {
        if only.is_some_and(|o| !o.iter().any(|n| n == &e.name)) {
            continue;
        }
        let Some(d) = m.def(&e.name) else {
            if !for_view {
                kept.push(format!("{}: the module has no def by that name; the entry stays in the file", e.name));
            }
            continue;
        };
        let mut left = Block::from_entry(e);
        let lead = if d.line > 0 { raw[d.line - 1][..raw[d.line - 1].len() - raw[d.line - 1].trim_start().len()].to_string() } else { String::new() };
        let mut above: Vec<String> = Vec::new();
        // fields and glossary go above the def, in `#|` `#||` `#|||` `#:` order
        if d.name == MAIN_DEF {
            if !for_view && (!left.fields.iter().all(|f| f.1.is_empty()) || !left.gloss.is_empty()) {
                kept.push(format!("{}: the top level has no inline home for fields or glossary; they stay in the file", e.name));
            }
        } else {
            let mut keep_fields = Vec::new();
            for (k, t) in &left.fields {
                if t.is_empty() {
                    continue;
                }
                if !FIELDS.contains(&k.as_str()) {
                    // `entry` and `serves` have no inline form
                    keep_fields.push((k.clone(), t.clone()));
                    continue;
                }
                if d.inline.field(k).is_some() {
                    if !for_view {
                        kept.push(format!("{}: `{k}` is also written inline; the file's stays in the file", e.name));
                        keep_fields.push((k.clone(), t.clone()));
                    }
                    continue;
                }
                let lvl = FIELDS.iter().position(|f| f == k).unwrap_or(0) + 1;
                above.extend(comment_lines(&format!("#{}", "|".repeat(lvl)), t, &lead));
            }
            let mut keep_gloss = Vec::new();
            for (k, t) in &left.gloss {
                if d.inline.gloss.iter().any(|g| &g.0 == k) {
                    if !for_view {
                        kept.push(format!("{}: glossary `{k}` is also written inline; the file's stays in the file", e.name));
                        keep_gloss.push((k.clone(), t.clone()));
                    }
                    continue;
                }
                above.push(format!("{lead}#: {k}  {t}").trim_end().to_string());
            }
            left.fields = keep_fields;
            left.gloss = keep_gloss;
        }
        // step texts go into the step headers
        let mut outline = std::mem::take(&mut left.outline);
        walk(&mut outline, &mut |n: &mut Node| {
            let NodeKind::Step(id) = &n.kind else { return };
            if n.text.is_empty() {
                return;
            }
            let Some(st) = d.step(id) else {
                if !for_view {
                    kept.push(format!("{}/{id}: no such step in the code; it stays in the file", e.name));
                }
                return;
            };
            if !st.text.is_empty() {
                if !for_view {
                    kept.push(format!("{}/{id}: its text is also written inline; the file's stays in the file", e.name));
                }
                return;
            }
            if let Some(new) = set_step_text(raw[st.line - 1], Some(&n.text)) {
                ed.replace.insert(st.line, new);
                n.text.clear();
            }
        });
        // control texts go above their statements, where the two shapes agree;
        // an `else` has no line of its own to go above
        let cmap = control_map(&code_as_intent(d, &|_| String::new(), &|_| None), &e.outline);
        let by_file_line: HashMap<usize, usize> = cmap.iter().map(|(cl, (fl, _))| (*fl, *cl)).collect();
        walk(&mut outline, &mut |n: &mut Node| match n.kind {
            NodeKind::Control(c) => {
                if is_bare_control(&n.text) {
                    return;
                }
                match by_file_line.get(&n.line) {
                    Some(&cl) if matches!(c, Ctl::Loop | Ctl::If) && !d.inline.controls.iter().any(|x| x.0 == cl) => {
                        let lead = &raw[cl - 1][..raw[cl - 1].len() - raw[cl - 1].trim_start().len()];
                        ed.insert.push((cl, comment_lines("#|", &n.text, lead)));
                        n.text = code_control_text(d, cl).unwrap_or_else(|| default_control_word(c).to_string());
                    }
                    _ => {
                        if !for_view {
                            kept.push(format!("{}: `{}` has no place in the code to go; it stays in the file", e.name, n.text));
                        }
                    }
                }
            }
            NodeKind::Note => {
                if !for_view {
                    kept.push(format!("{}: `{}` has no inline home; it stays in the file", e.name, n.text));
                }
            }
            NodeKind::Step(_) => {}
        });
        left.outline = outline;
        let emptied = left.is_empty_of_text();
        // the stamp moves with the last of the entry; while part of the block
        // stays, the block's stamp covers the def
        if !for_view && emptied {
            match (&e.stamp, &d.inline.stamp) {
                (Some(s), None) if d.name != MAIN_DEF => above.push(format!("{lead}#@ {s}")),
                (Some(s), Some((h, _))) if s != h => kept.push(format!("{}: the file's stamp @{s} is dropped; the inline stamp #@ {h} stays", e.name)),
                _ => {}
            }
        }
        if !above.is_empty() {
            ed.insert.push((d.line, above));
        }
        remaining.insert(ei, if emptied { None } else { Some(left) });
    }
    let new_src = ed.apply(src);
    if for_view {
        return (new_src, None, kept);
    }
    // rebuild the file: untouched blocks as written, touched ones rendered from what is left
    let Some(f) = &m.intent else { return (new_src, None, kept) };
    let mut out_lines: Vec<String> = Vec::new();
    let mut i = 0;
    while i < f.lines.len() {
        if let Some((ei, e)) = entries.iter().enumerate().find(|(_, e)| e.first == i) {
            if let Some(left) = remaining.get(&ei) {
                match left {
                    Some(b) => out_lines.extend(render_block(b, true)),
                    // drop the block, and the blank line after it
                    None if e.last + 1 < f.lines.len() && f.lines[e.last + 1].kind == Kind::Blank => {
                        i = e.last + 2;
                        continue;
                    }
                    None => {}
                }
                i = e.last + 1;
                continue;
            }
        }
        out_lines.push(f.lines[i].raw.clone());
        i += 1;
    }
    while out_lines.last().is_some_and(|l| l.trim().is_empty()) {
        out_lines.pop();
    }
    let only_module = out_lines.iter().all(|l| l.trim().is_empty() || l.starts_with("module ")) && module.as_ref().map_or(true, |m| m.text.is_empty());
    if only_module {
        return (new_src, None, kept);
    }
    let mut text = out_lines.join("\n");
    text.push('\n');
    (new_src, Some(text), kept)
}

/// Move inline entries into the `.intent` file: fields, glossary, step texts,
/// control notes, and the `#@` stamp. An anchor the file already holds stays
/// where it is and is reported.
pub fn to_file(l: &Loaded, only: Option<&[String]>) -> Moved {
    let m = &l.root;
    let src = &m.text.src;
    let raw: Vec<&str> = src.lines().collect();
    let (module, entries) = m.entries();
    let mut ed = Edits::default();
    let mut kept = Vec::new();
    let mut changed: HashMap<String, Block> = HashMap::new();
    let mut new_blocks: Vec<Block> = Vec::new();
    for d in &m.defs {
        if only.is_some_and(|o| !o.iter().any(|n| n == &d.name)) {
            continue;
        }
        let inl = &d.inline;
        let has_steps_text = d.steps().iter().any(|(n, _)| !n.text.is_empty());
        if inl.fields.is_empty() && inl.gloss.is_empty() && inl.stamp.is_none() && !has_steps_text && inl.controls.is_empty() {
            continue;
        }
        let e = entries.iter().find(|e| e.name == d.name);
        let mut b = e.map(Block::from_entry).unwrap_or_else(|| Block { name: d.name.clone(), ..Block::default() });
        let mut moved_lines: HashSet<usize> = HashSet::new();
        // fields
        for (k, t, _) in &inl.fields {
            match b.fields.iter_mut().find(|f| &f.0 == k) {
                Some(f) if !f.1.is_empty() => {
                    kept.push(format!("{}: `{k}` is also in the file; the inline one stays inline", d.name));
                    continue;
                }
                Some(f) => f.1 = t.clone(),
                None => b.fields.push((k.clone(), t.clone())),
            }
            let lvl = FIELDS.iter().position(|f| f == k).unwrap_or(0) as u8 + 1;
            moved_lines.extend(inl.lines.iter().copied().filter(|&r| annotation(raw[r - 1]) == Some(lvl)));
        }
        b.fields.sort_by_key(|f| FIELDS.iter().position(|x| *x == f.0).unwrap_or(9));
        // glossary: every `#:` line of the def moves together unless one name clashes
        let clash: Vec<&String> = inl.gloss.iter().map(|g| &g.0).filter(|n| b.gloss.iter().any(|x| &&x.0 == n)).collect();
        if clash.is_empty() {
            for (k, t, _) in &inl.gloss {
                b.gloss.push((k.clone(), t.clone()));
            }
            moved_lines.extend(inl.lines.iter().copied().filter(|&r| annotation(raw[r - 1]) == Some(4)));
        } else {
            kept.push(format!("{}: glossary {} also in the file; the inline glossary stays inline", d.name, clash.iter().map(|c| format!("`{c}`")).collect::<Vec<_>>().join(", ")));
        }
        // the stamp
        if let Some((h, r)) = &inl.stamp {
            match &b.stamp {
                Some(s) if s != h => kept.push(format!("{}: stamped @{h} inline and @{s} in the file; the inline stamp stays inline", d.name)),
                _ => {
                    b.stamp = Some(h.clone());
                    moved_lines.insert(*r);
                }
            }
        }
        // the outline: the file's, or the code's
        let file_texts = file_step_texts(e);
        if b.outline.is_empty() {
            let ctl: HashMap<usize, String> = inl.controls.iter().map(|(l, t, _)| (*l, t.clone())).collect();
            b.outline = code_as_intent(d, &|id| d.step(id).map(|n| n.text.clone()).unwrap_or_default(), &|l| ctl.get(&l).cloned());
            for (st, _) in d.steps() {
                if !st.text.is_empty() {
                    if let Some(new) = set_step_text(raw[st.line - 1], None) {
                        ed.replace.insert(st.line, new);
                    }
                }
            }
            for (_, _, ls) in &inl.controls {
                moved_lines.extend(ls.iter().copied());
            }
        } else {
            let mut missing = Vec::new();
            for (st, _) in d.steps() {
                let OKind::Step(id) = &st.kind else { continue };
                if st.text.is_empty() {
                    continue;
                }
                if file_texts.contains_key(id) {
                    kept.push(format!("{}/{id}: its text is also in the file; the inline text stays inline", d.name));
                    continue;
                }
                fn set(ns: &mut [Node], id: &str, t: &str) -> bool {
                    for n in ns {
                        if n.kind == NodeKind::Step(id.to_string()) {
                            n.text = t.to_string();
                            return true;
                        }
                        if set(&mut n.children, id, t) {
                            return true;
                        }
                    }
                    false
                }
                if set(&mut b.outline, id, &st.text) {
                    if let Some(new) = set_step_text(raw[st.line - 1], None) {
                        ed.replace.insert(st.line, new);
                    }
                } else {
                    missing.push(id.clone());
                }
            }
            if !missing.is_empty() {
                kept.push(format!("{}: the file's outline has no line for step {}; its text stays inline", d.name, missing.join(", ")));
            }
            // control notes, where the shapes agree and the file has only the bare word
            let code_nodes = code_as_intent(d, &|_| String::new(), &|_| None);
            let cmap = control_map(&code_nodes, &b.outline);
            for (cl, text, ls) in &inl.controls {
                let Some((_, ft)) = cmap.get(cl) else {
                    kept.push(format!("{}: the note above line {cl} has no control line in the file's outline to go to; it stays inline", d.name));
                    continue;
                };
                fn set_ctl(ns: &mut [Node], old: &str, new: &str) -> bool {
                    for n in ns {
                        if matches!(n.kind, NodeKind::Control(_)) && n.text == old {
                            n.text = new.to_string();
                            return true;
                        }
                        if set_ctl(&mut n.children, old, new) {
                            return true;
                        }
                    }
                    false
                }
                if is_bare_control(ft) && set_ctl(&mut b.outline, ft, text) {
                    moved_lines.extend(ls.iter().copied());
                } else {
                    kept.push(format!("{}: the control line `{ft}` already has text in the file; the note above line {cl} stays inline", d.name));
                }
            }
        }
        for r in moved_lines {
            ed.delete.insert(r);
        }
        if e.is_some() {
            changed.insert(d.name.clone(), b);
        } else {
            new_blocks.push(b);
        }
    }
    let new_src = ed.apply(src);
    // rebuild the file
    let mut out_lines: Vec<String> = Vec::new();
    match &m.intent {
        Some(f) => {
            let mut i = 0;
            while i < f.lines.len() {
                if let Some(e) = entries.iter().find(|e| e.first == i) {
                    if let Some(b) = changed.get(&e.name) {
                        out_lines.extend(render_block(b, true));
                        i = e.last + 1;
                        continue;
                    }
                }
                out_lines.push(f.lines[i].raw.clone());
                i += 1;
            }
        }
        None => out_lines.extend(render_module(&module.clone().unwrap_or(ModuleEntry { name: m.text.name.clone(), text: String::new(), line: 0, every_var: false }))),
    }
    while out_lines.last().is_some_and(|l| l.trim().is_empty()) {
        out_lines.pop();
    }
    for b in &new_blocks {
        out_lines.push(String::new());
        out_lines.extend(render_block(b, true));
    }
    let mut text = out_lines.join("\n");
    text.push('\n');
    (new_src, Some(text), kept)
}

/// Stamp a def with its current body hash in every home that holds its
/// intent: the file block's header, and the inline `#@` line (added when the
/// def's intent is inline and it has none). Returns the new source and file.
pub fn stamp(l: &Loaded, def: &str) -> Result<(String, Option<String>, String), String> {
    let m = &l.root;
    let d = m.def(def).ok_or_else(|| format!("{} has no def named `{def}`", m.text.path))?;
    let hash = d.hash.clone().ok_or_else(|| format!("cannot hash `{def}`: {}", l.hash_error.clone().unwrap_or_else(|| "the program does not build".into())))?;
    let (_, entries) = m.entries();
    let e = entries.iter().find(|e| e.name == def);
    let src = &m.text.src;
    let raw: Vec<&str> = src.lines().collect();
    let mut ed = Edits::default();
    let mut homes = Vec::new();
    if d.line > 0 && (d.inline.stamp.is_some() || (e.is_none() && d.inline.has_intent(d))) {
        let lead = &raw[d.line - 1][..raw[d.line - 1].len() - raw[d.line - 1].trim_start().len()];
        match &d.inline.stamp {
            Some((_, r)) => {
                let lead = &raw[r - 1][..raw[r - 1].len() - raw[r - 1].trim_start().len()];
                ed.replace.insert(*r, format!("{lead}#@ {hash}"));
            }
            None => ed.insert.push((d.line, vec![format!("{lead}#@ {hash}")])),
        }
        homes.push("inline");
    }
    let mut file = m.intent.clone();
    if let (Some(e), Some(f)) = (e, file.as_mut()) {
        f.lines[e.first] = Line { raw: header_line(&e.name, Some(&hash), e.by.as_deref()), indent: 0, kind: Kind::Header { name: e.name.clone(), stamp: Some(hash.clone()), by: e.by.clone() } };
        homes.push("file");
    }
    if homes.is_empty() {
        return Err(format!("`{def}` has no intent to stamp: write its `#|` lines or its block in {}", m.text.intent_path));
    }
    Ok((ed.apply(src), file.map(|f| f.print()), format!("stamped {def} @{hash} ({})", homes.join(" and "))))
}

// ---------------------------------------------------------------------------
// lens check
// ---------------------------------------------------------------------------

/// One problem `lens check` found.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub path: String,
    pub line: usize,
    /// the anchor: `def`, `def/step`, `def:name`, or `*` for the module
    pub anchor: String,
    pub kind: &'static str,
    pub message: String,
    pub detail: Vec<String>,
}

impl Finding {
    pub fn render(&self) -> String {
        let mut s = format!("{}:{}: {}: {}\n", self.path, self.line, self.kind, self.message);
        for d in &self.detail {
            s.push_str(&format!("    {d}\n"));
        }
        s
    }
    pub fn json(&self) -> String {
        format!(
            "{{\"path\":{},\"line\":{},\"anchor\":{},\"kind\":\"{}\",\"message\":{},\"detail\":[{}]}}",
            json_str(&self.path),
            self.line,
            json_str(&self.anchor),
            self.kind,
            json_str(&self.message),
            self.detail.iter().map(|d| json_str(d)).collect::<Vec<_>>().join(",")
        )
    }
}

/// Defs marked as test helpers with a `#~ test` tag above them: they exist
/// for the module's tests and need no `what`. The mark is explicit because a
/// library's entry point is often called only from its own tests, and must
/// not be exempted for that.
fn test_helpers(m: &Mod) -> HashSet<String> {
    m.defs.iter().filter(|d| d.inline.tags.iter().any(|t| t == "test")).map(|d| d.name.clone()).collect()
}

/// A word of prose that is a number: `20`, `1e-10`, `0.5`, `-75.98`, `3.`,
/// `50%`, `1,000`. A word with letters in it (`6-31G`, `2D`, `H2O`) is a name.
fn number_in(w: &str) -> Option<String> {
    let w = w.trim_matches(|c: char| "()[]{},;:!?\"'".contains(c));
    let w = w.trim_end_matches('.');
    is_number_word(w).then(|| w.to_string())
}

fn is_number_word(w: &str) -> bool {
    let w = w.trim_matches(|c: char| "()[]{},;:!?\"'".contains(c));
    let w = w.trim_end_matches('.');
    let w = w.trim_start_matches(|c: char| "~<>=≈±+-".contains(c));
    let w = w.trim_end_matches('%');
    if w.is_empty() || !w.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    let plain = w.replace(',', "");
    if w.contains(',') && !w.split(',').skip(1).all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit())) {
        return false;
    }
    // digits, one optional fraction, one optional exponent
    let (mant, exp) = match plain.find(|c| c == 'e' || c == 'E') {
        Some(k) => (&plain[..k], Some(&plain[k + 1..])),
        None => (plain.as_str(), None),
    };
    let mant_ok = {
        let mut parts = mant.splitn(2, '.');
        let a = parts.next().unwrap_or("");
        let b = parts.next();
        !a.is_empty() && a.chars().all(|c| c.is_ascii_digit()) && b.map_or(true, |b| b.chars().all(|c| c.is_ascii_digit()))
    };
    let exp_ok = exp.map_or(true, |e| {
        let e = e.strip_prefix(|c| c == '-' || c == '+').unwrap_or(e);
        !e.is_empty() && e.chars().all(|c| c.is_ascii_digit())
    });
    mant_ok && exp_ok
}

/// The citations `[<kind> "name"]` in a text, in order.
pub fn citations(text: &str, kind: &str) -> Vec<String> {
    let open = format!("[{kind} \"");
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(k) = rest.find(&open) {
        let after = &rest[k + open.len()..];
        match after.find("\"]") {
            Some(e) => {
                out.push(after[..e].to_string());
                rest = &after[e + 2..];
            }
            None => break,
        }
    }
    out
}

/// The citations `[test "name"]` in a text, and the numbers outside code
/// spans (between backticks), math (between dollars) and citations.
fn numbers_and_citations(text: &str) -> (Vec<String>, Vec<String>) {
    let mut cites = Vec::new();
    let mut prose = String::new();
    // a hold citation is not prose either
    let unheld = {
        let mut t = text.to_string();
        for h in citations(text, "hold") {
            t = t.replace(&format!("[hold \"{h}\"]"), " ");
        }
        t
    };
    let mut rest = unheld.as_str();
    while let Some(k) = rest.find("[test \"") {
        prose.push_str(&rest[..k]);
        let after = &rest[k + 7..];
        match after.find("\"]") {
            Some(e) => {
                cites.push(after[..e].to_string());
                prose.push(' ');
                rest = &after[e + 2..];
            }
            None => {
                prose.push_str(&rest[k..]);
                rest = "";
            }
        }
    }
    prose.push_str(rest);
    // drop math, then code spans
    let prose = crate::tex::without_math(&prose);
    let mut plain = String::new();
    for (k, part) in prose.split('`').enumerate() {
        if k % 2 == 0 {
            plain.push_str(part);
            plain.push(' ');
        }
    }
    let nums = plain.split_whitespace().filter_map(number_in).collect();
    (nums, cites)
}

/// The shape of an outline, for comparing the code's with the intent's:
/// steps by id, control structures that hold steps by kind, notes dropped.
fn shape_of_intent(ns: &[Node]) -> String {
    let keep = kept_controls(ns, nctl, |n| node_has_step(&n.children));
    let parts: Vec<String> = ns
        .iter()
        .zip(keep)
        .filter_map(|(n, keep)| match &n.kind {
            NodeKind::Step(id) => {
                let inner = shape_of_intent(&n.children);
                Some(if inner.is_empty() { id.clone() } else { format!("{id}[{inner}]") })
            }
            NodeKind::Control(c) if keep => Some(format!("{}[{}]", c.sig(), shape_of_intent(&n.children))),
            _ => None,
        })
        .collect();
    parts.join(" ")
}

/// Line diff of two texts (longest common subsequence), `-`/`+` lines only.
fn line_diff(old: &str, new: &str) -> Vec<String> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let (n, m) = (a.len(), b.len());
    let mut t = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[i][j] = if a[i] == b[j] { t[i + 1][j + 1] + 1 } else { t[i + 1][j].max(t[i][j + 1]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            i += 1;
            j += 1;
        } else if i < n && (j == m || t[i + 1][j] >= t[i][j + 1]) {
            out.push(format!("- {}", a[i]));
            i += 1;
        } else {
            out.push(format!("+ {}", b[j]));
            j += 1;
        }
    }
    out
}

/// A def's source text (its `def` line to its closing brace), for diffs.
pub fn def_text(src: &str, name: &str) -> Option<String> {
    let prog = crate::parse_keeping_rungs(src, SourceId(0)).ok()?;
    prog.items.iter().find_map(|it| match it {
        Item::Def(d) if d.name == name => Some(src[d.span.start as usize..d.span.end as usize].to_string()),
        _ => None,
    })
}

/// Every text of an entry's outline, with its line: step texts, control and
/// note lines, and detail lines (equations with their citations).
pub fn entry_texts(e: &Entry) -> Vec<(usize, String)> {
    fn walk(ns: &[Node], out: &mut Vec<(usize, String)>) {
        for n in ns {
            out.push((n.line, n.text.clone()));
            for dt in &n.details {
                out.push((dt.line, dt.text.clone()));
            }
            walk(&n.children, out);
        }
    }
    let mut out: Vec<(usize, String)> = e.details.iter().map(|dt| (dt.line, dt.text.clone())).collect();
    walk(&e.outline, &mut out);
    out
}

/// The `{name}`, `{name#k}` and `{name:p}` placeholders of a `run` or
/// `predict` line, outside its math: (name, which run of the step, 1 by
/// default; significant digits, 4 by default).
pub fn placeholders(text: &str) -> Vec<(String, usize, usize)> {
    let prose = crate::tex::without_math(text);
    let mut out = Vec::new();
    let mut rest = prose.as_str();
    while let Some(k) = rest.find('{') {
        let after = &rest[k + 1..];
        let Some(e) = after.find('}') else { break };
        let inner = &after[..e];
        let (head, digits) = match inner.split_once(':') {
            Some((h, d)) => (h, d.parse::<usize>().unwrap_or(4)),
            None => (inner, 4),
        };
        let (name, nth) = match head.split_once('#') {
            Some((n, k)) => (n, k.parse::<usize>().unwrap_or(1).max(1)),
            None => (head, 1),
        };
        if is_ident(name) {
            out.push((name.to_string(), nth, digits));
        }
        rest = &after[e + 1..];
    }
    out
}

/// Every `[hold "sentence"]` an entry of the module cites must be a hold of
/// the debug lens (`holds`: the sentences of the module's `.debug` file and
/// those of the modules it uses). An equation that cites a hold is then a
/// check that runs whenever the program runs in debug mode.
pub fn check_hold_citations(l: &Loaded, holds: &[String]) -> Vec<Finding> {
    let m = &l.root;
    let (_, entries) = m.entries();
    let mut out = Vec::new();
    for e in &entries {
        let mut texts: Vec<(usize, String)> = e.fields.iter().map(|t| (t.line, t.text.clone())).collect();
        texts.extend(entry_texts(e));
        for (line, t) in texts {
            for c in citations(&t, "hold") {
                if !holds.iter().any(|h| h == &c) {
                    out.push(Finding {
                        path: m.text.intent_path.clone(),
                        line,
                        anchor: e.name.clone(),
                        kind: "citation",
                        message: format!("`{}` cites [hold \"{c}\"], and no hold with that sentence is in the debug lens of {} or the modules it uses", e.name, m.text.name),
                        detail: vec![],
                    });
                }
            }
        }
    }
    out
}

/// Check a module's lenses against its code. `stamped_source(def, hash)`
/// returns the def's source as of a version whose body hash was `hash`, when
/// the host can find one (main.rs looks through git), so a stale entry can be
/// shown with what changed.
pub fn check(l: &Loaded, stamped_source: &mut dyn FnMut(&str, &str) -> Option<String>) -> Vec<Finding> {
    let m = &l.root;
    let path = m.text.path.clone();
    let ipath = m.text.intent_path.clone();
    let mut out: Vec<Finding> = Vec::new();
    let mut find = |p: &str, line: usize, anchor: &str, kind: &'static str, message: String, detail: Vec<String>| {
        out.push(Finding { path: p.to_string(), line, anchor: anchor.to_string(), kind, message, detail });
    };
    for e in &m.intent_errors {
        find(&ipath, e.line, "*", "parse", e.message.clone(), vec![]);
    }
    if !m.intent_errors.is_empty() {
        return out;
    }
    if let Some(e) = &l.hash_error {
        find(&path, 1, "*", "build", "the program does not build, so no body can be hashed".into(), e.lines().take(4).map(String::from).collect());
    }
    let (_, entries) = m.entries();
    // the tests a citation may name: this module's, its companion test file's
    // and every used module's
    let tests: HashSet<&str> = std::iter::once(m).chain(l.used.iter()).flat_map(|md| md.tests.iter().map(String::as_str)).chain(l.companion_tests.iter().map(String::as_str)).collect();
    let helpers = test_helpers(m);

    // file entries: one block per def, for a def the module has
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for e in &entries {
        if let Some(first) = seen.get(e.name.as_str()) {
            find(&ipath, e.line, &e.name, "duplicate", format!("`{}` has a second block; its first is at line {first}", e.name), vec![]);
            continue;
        }
        seen.insert(&e.name, e.line);
        if m.def(&e.name).is_none() {
            let renamed = e.stamp.as_ref().and_then(|s| m.defs.iter().find(|d| d.hash.as_ref() == Some(s)));
            let detail = match renamed {
                Some(d) => vec![format!("`{}` has the body this entry was stamped for; was the def renamed?", d.name)],
                None => vec![],
            };
            find(&ipath, e.line, &e.name, "orphan", format!("an entry for `{}`, which {} does not define", e.name, path), detail);
        }
    }

    for d in &m.defs {
        let e = entries.iter().find(|e| e.name == d.name);
        let dup_block = entries.iter().filter(|x| x.name == d.name).count() > 1;
        let at_def = if d.line > 0 { d.line } else { 1 };
        let is_main = d.name == MAIN_DEF;
        // one home per anchor, and one line per anchor within a home
        let mut dup = false;
        if let Some(e) = e {
            for label in CARD_FIELDS {
                let in_file: Vec<&Text> = e.fields.iter().filter(|t| t.key == label).collect();
                if in_file.len() > 1 {
                    find(&ipath, in_file[1].line, &d.name, "duplicate", format!("`{}` has `{label}` twice in the file", d.name), vec![]);
                    dup = true;
                }
                if let (Some(t), Some(f)) = (in_file.first(), d.inline.field(label)) {
                    if !t.text.is_empty() {
                        find(&ipath, t.line, &d.name, "duplicate", format!("`{}` has `{label}` both here and inline at {}:{}; an anchor has one home", d.name, path, f.2), vec![]);
                        dup = true;
                    }
                }
            }
            let mut gseen: HashSet<&str> = HashSet::new();
            for g in &e.gloss {
                let anchor = format!("{}:{}", d.name, g.key);
                if !gseen.insert(&g.key) {
                    find(&ipath, g.line, &anchor, "duplicate", format!("`{anchor}` is in the file's glossary twice"), vec![]);
                    dup = true;
                } else if let Some(ig) = d.inline.gloss.iter().find(|x| x.0 == g.key) {
                    find(&ipath, g.line, &anchor, "duplicate", format!("`{anchor}` is both here and inline at {}:{}; an anchor has one home", path, ig.2), vec![]);
                    dup = true;
                }
            }
            let mut sseen: HashSet<String> = HashSet::new();
            fn walk<'a>(ns: &'a [Node], f: &mut dyn FnMut(&'a Node)) {
                for n in ns {
                    f(n);
                    walk(&n.children, f);
                }
            }
            let mut step_dups = Vec::new();
            walk(&e.outline, &mut |n| {
                if let NodeKind::Step(id) = &n.kind {
                    if !sseen.insert(id.clone()) {
                        step_dups.push((n.line, id.clone(), None));
                    } else if !n.text.is_empty() {
                        if let Some(st) = d.step(id).filter(|st| !st.text.is_empty()) {
                            step_dups.push((n.line, id.clone(), Some(st.line)));
                        }
                    }
                }
            });
            for (line, id, inline_at) in step_dups {
                let anchor = format!("{}/{id}", d.name);
                match inline_at {
                    None => find(&ipath, line, &anchor, "duplicate", format!("step `{anchor}` is listed twice"), vec![]),
                    Some(il) => find(&ipath, line, &anchor, "duplicate", format!("step `{anchor}` has text both here and inline at {path}:{il}; an anchor has one home"), vec![]),
                }
                dup = true;
            }
        }
        let mut igseen: HashSet<&str> = HashSet::new();
        for g in &d.inline.gloss {
            if !igseen.insert(&g.0) {
                find(&path, g.2, &format!("{}:{}", d.name, g.0), "duplicate", format!("`{}:{}` is in the inline glossary twice", d.name, g.0), vec![]);
                dup = true;
            }
        }
        if dup_block {
            continue;
        }

        // staleness: every home's stamp must be the current body hash
        let mut stale = false;
        let stale_detail = |old: &str, stamped_source: &mut dyn FnMut(&str, &str) -> Option<String>| -> Vec<String> {
            let mut det = Vec::new();
            let new = d.hash.clone().unwrap_or_default();
            let old_h = crate::DefHash { name: d.name.clone(), kind: if is_main { "main" } else { "def" }, node_short: String::new(), body_short: old.to_string(), display: String::new() };
            let display = if is_main { "top-level program".to_string() } else { format!("def {}", d.sig) };
            let new_h = crate::DefHash { name: d.name.clone(), kind: old_h.kind, node_short: String::new(), body_short: new.clone(), display };
            det.extend(crate::structural_diff(&[old_h], &[new_h]).lines().map(|s| s.trim_end().to_string()));
            if !is_main {
                match stamped_source(&d.name, old) {
                    Some(old_src) => {
                        let now = def_text(&m.text.src, &d.name).unwrap_or_default();
                        let diff = line_diff(&old_src, &now);
                        let shown = diff.len().min(40);
                        det.extend(diff[..shown].iter().cloned());
                        if diff.len() > shown {
                            det.push(format!("... {} more changed lines", diff.len() - shown));
                        }
                    }
                    None => det.push(format!("(no version with body {old} found; `viv diff <that version> {path}` shows the change)")),
                }
            }
            det
        };
        // steps in the code and in the file's outline
        let code_ids: Vec<String> = d.steps().iter().filter_map(|(n, _)| if let OKind::Step(id) = &n.kind { Some(id.clone()) } else { None }).collect();
        let mut file_ids: Vec<(String, usize)> = Vec::new();
        fn ids(ns: &[Node], out: &mut Vec<(String, usize)>) {
            for n in ns {
                if let NodeKind::Step(id) = &n.kind {
                    out.push((id.clone(), n.line));
                }
                ids(&n.children, out);
            }
        }
        if let Some(e) = e {
            ids(&e.outline, &mut file_ids);
        }
        let step_changes = |det: &mut Vec<String>| {
            for (id, _) in &file_ids {
                if !code_ids.contains(id) {
                    det.push(format!("the intent names step `{id}`, which the code no longer has"));
                }
            }
            if e.is_some_and(|e| !e.outline.is_empty()) {
                for id in &code_ids {
                    if !file_ids.iter().any(|(f, _)| f == id) {
                        det.push(format!("the code has step `{id}`, which the intent's outline does not list"));
                    }
                }
            }
        };
        if let (Some(e), Some(h)) = (e, &d.hash) {
            match &e.stamp {
                Some(s) if s != h => {
                    let mut det = stale_detail(s, stamped_source);
                    step_changes(&mut det);
                    find(&ipath, e.line, &d.name, "stale", format!("`{}` was stamped @{s} and its body is now {h}; re-read the entry, then `viv lens stamp {path} {}`", d.name, d.name), det);
                    stale = true;
                }
                None => {
                    find(&ipath, e.line, &d.name, "unstamped", format!("`{}` has no stamp; review its entry, then `viv lens stamp {path} {}`", d.name, d.name), vec![]);
                    stale = true;
                }
                _ => {}
            }
        }
        if let Some(h) = &d.hash {
            if !is_main {
                match &d.inline.stamp {
                    Some((s, r)) if s != h => {
                        let mut det = stale_detail(s, stamped_source);
                        if e.is_none() {
                            step_changes(&mut det);
                        }
                        find(&path, *r, &d.name, "stale", format!("`{}` was stamped #@ {s} and its body is now {h}; re-read its intent, then `viv lens stamp {path} {}`", d.name, d.name), det);
                        stale = true;
                    }
                    None if e.is_none() && d.inline.has_intent(d) => {
                        find(&path, at_def, &d.name, "unstamped", format!("`{}` has inline intent and no `#@` stamp; review it, then `viv lens stamp {path} {}`", d.name, d.name), vec![]);
                        stale = true;
                    }
                    _ => {}
                }
            }
        }
        let mg = merged(d, e);
        if !stale && !dup {
            // every step in the code has text somewhere, and every step the file names exists
            let mut step_errs = 0;
            let fst = file_step_texts(e);
            for (st, _) in d.steps() {
                let OKind::Step(id) = &st.kind else { continue };
                if st.text.is_empty() && !fst.contains_key(id) {
                    find(&path, st.line, &format!("{}/{id}", d.name), "step-text", format!("step `{}/{id}` has no text in any lens", d.name), vec![]);
                    step_errs += 1;
                }
            }
            for (id, line) in &file_ids {
                if !code_ids.contains(id) {
                    find(&ipath, *line, &format!("{}/{id}", d.name), "step-missing", format!("the intent names step `{}/{id}`, which the code does not have", d.name), vec![]);
                    step_errs += 1;
                }
            }
            // the nesting of steps and control lines agrees with the code
            if step_errs == 0 {
                if let Some(e) = e.filter(|e| !e.outline.is_empty()) {
                    let code = shape_of_intent(&code_as_intent(d, &|_| String::new(), &|_| None));
                    let file = shape_of_intent(&e.outline);
                    if code != file {
                        find(&ipath, e.outline[0].line, &d.name, "nesting", format!("the outline of `{}` does not match the code", d.name), vec![format!("code:   {code}"), format!("intent: {file}")]);
                    }
                }
            }
        }
        // coverage
        if !is_main {
            let public = !d.name.starts_with('_') && !helpers.contains(&d.name);
            if public && mg.field("what").map_or(true, |w| w.trim().is_empty()) {
                // point at the file's empty `what` line when the def has a block
                let (p, line) = match e {
                    Some(e) => (&ipath, e.field("what").map(|t| t.line).unwrap_or(e.line)),
                    None => (&path, at_def),
                };
                find(p, line, &d.name, "what", format!("public def `{}` has no `what`", d.name), vec![]);
            }
            // a hot def may be explained statement by statement instead: `=`
            // lines at the top of its outline
            let anchored = e.is_some_and(|e| e.details.iter().any(|dt| dt.label == "="));
            if d.stmts > 25 && d.steps().is_empty() && !anchored {
                find(&path, at_def, &d.name, "steps", format!("`{}` has {} statements and no steps; name its phases with `step`, or explain its statements with `=` lines in its entry", d.name, d.stmts), vec![]);
            }
        }
        // `entry` says who calls the def; `serves` is citations and nothing else
        if let Some(e) = e {
            for t in e.fields.iter().filter(|t| t.key == "entry" || t.key == "serves") {
                if t.key == "entry" && t.text.trim().is_empty() {
                    find(&ipath, t.line, &d.name, "entry", format!("`{}` entry: say who calls it from outside the repository's programs", d.name), vec![]);
                }
                if t.key == "serves" {
                    let (targets, rest) = crate::lens_cov::serves_targets(&t.text);
                    if targets.is_empty() || !rest.is_empty() {
                        find(&ipath, t.line, &d.name, "serves", format!("`{}` serves: write only citations, `[test \"...\"]`, `[post \"slug\"]`, `[program \"path\"]`, `[hold \"...\"]` or `[claim \"id\"]`", d.name), vec![]);
                    }
                }
            }
        }
        // numbers must cite a test
        for (label, text, home) in &mg.fields {
            let (nums, cites) = numbers_and_citations(text);
            let (p, line) = match home {
                Home::File => (&ipath, e.and_then(|e| e.field(label)).map(|t| t.line).unwrap_or(1)),
                Home::Inline => (&path, d.inline.field(label).map(|f| f.2).filter(|&l| l > 0).unwrap_or(at_def)),
            };
            for c in &cites {
                if !tests.contains(c.as_str()) {
                    find(p, line, &d.name, "citation", format!("`{}` {label} cites [test \"{c}\"], and no test by that name is in {} or the modules it uses", d.name, m.text.name), vec![]);
                }
            }
            if !nums.is_empty() && cites.is_empty() {
                find(p, line, &d.name, "number", format!("`{}` {label} states {} with no [test \"...\"] citation", d.name, nums.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")), vec![]);
            }
        }
        // math: every `$...$` and `$$...$$` is LaTeX that KaTeX can typeset,
        // and an equation's [test "..."] names a test
        let mut texts: Vec<(&str, usize, String)> = Vec::new();
        if let Some(e) = e {
            for t in e.fields.iter().chain(e.gloss.iter()) {
                texts.push((&ipath, t.line, t.text.clone()));
            }
            for (line, t) in entry_texts(e) {
                texts.push((&ipath, line, t));
            }
        }
        for f in d.inline.fields.iter().chain(d.inline.gloss.iter()) {
            texts.push((&path, if f.2 > 0 { f.2 } else { at_def }, f.1.clone()));
        }
        for (p, line, t) in &texts {
            let (spans, closed) = crate::tex::math_spans(t);
            if !closed {
                find(p, *line, &d.name, "tex", "a `$` that opens math is never closed (write `\\$` for a dollar sign)".into(), vec![]);
            }
            for sp in spans {
                if let Err(why) = crate::tex::check(&sp) {
                    find(p, *line, &d.name, "tex", format!("`{}`: {why}", sp.trim()), vec![]);
                }
            }
        }
        if let Some(e) = e {
            for (line, t) in entry_texts(e) {
                for c in citations(&t, "test") {
                    if !tests.contains(c.as_str()) && t.starts_with("$$") {
                        find(&ipath, line, &d.name, "citation", format!("an equation of `{}` cites [test \"{c}\"], and no test by that name is in {} or the modules it uses", d.name, m.text.name), vec![]);
                    }
                }
            }
        }
    }
    // line anchors: each must name a statement of its step (or of the def)
    let parsed = crate::parse_keeping_rungs(&m.text.src, SourceId(0)).ok();
    if let Some(prog) = parsed {
        // the whole program's defs, so `= f()` can name a def of a used module
        let whole = crate::parse_keeping_rungs(&l.expanded, SourceId(0)).ok();
        let cx = whole.as_ref().map(crate::graph::Cx::of).unwrap_or_else(|| crate::graph::Cx::of(&prog));
        let lines = crate::graph::Lines::new(&m.text.src);
        for e in &entries {
            let Some(def) = prog.items.iter().find_map(|i| if let Item::Def(d) = i { (d.name == e.name).then_some(d) } else { None }) else { continue };
            let cands = crate::graph::anchor_candidates(def, &cx, &lines);
            // a `run` or `predict` line's `{name}` is a value of this def
            let known: HashSet<&str> = def.params.iter().map(|p| p.name.as_str()).chain(cands.iter().flat_map(|c| c.2.iter().map(String::as_str))).collect();
            for (step, dt) in crate::graph::entry_anchors(&e.details, &e.outline) {
                if dt.label == "run" || dt.label == "predict" {
                    for (name, _, _) in placeholders(&dt.text) {
                        if !known.contains(name.as_str()) {
                            let anchor = step.as_ref().map(|s| format!("{}/{s}", e.name)).unwrap_or_else(|| e.name.clone());
                            find(&ipath, dt.line, &anchor, "placeholder", format!("`{{{name}}}` names no variable of `{}`: a `{}` line fills in values the def assigns", e.name, dt.label), vec![]);
                        }
                    }
                }
                if dt.label != "=" {
                    continue;
                }
                if crate::graph::resolve_anchor(&cands, step.as_deref(), &dt).is_none() {
                    let anchor = match &step {
                        Some(sid) => format!("{}/{sid}", e.name),
                        None => e.name.clone(),
                    };
                    let (name, call, k) = dt.target();
                    let what = if call { format!("calls `{name}`") } else { format!("assigns `{name}`") };
                    let nth = if k > 1 { format!(" {k} times") } else { String::new() };
                    find(&ipath, dt.line, &anchor, "anchor", format!("`= {}` names no statement: nothing in `{anchor}` {what}{nth}", dt.anchor), vec![]);
                }
            }
        }
        // variable docs: each glossary line starts to hold at a statement,
        // and with `documents every variable` each variable has one. What a
        // call writes into its arguments is known from the whole program.
        let every = m.entries().0.is_some_and(|me| me.every_var);
        for d in &m.defs {
            let e = entries.iter().find(|e| e.name == d.name);
            let mg = merged(d, e);
            let keys: Vec<String> = mg.gloss.iter().map(|g| g.0.clone()).collect();
            let has_entry = e.is_some();
            if !has_entry && !keys.iter().any(|k| k.contains('@')) {
                continue;
            }
            let Some((body, params, start)) = crate::graph::body_of(&prog, &d.name) else { continue };
            let stmts = crate::graph::doc_statements(&body, &cx, &lines);
            let def_line = lines.line(start as u32);
            let gloss_line = |k: &str| -> (&str, usize) {
                match e.and_then(|e| e.gloss.iter().find(|g| g.key == k)) {
                    Some(g) => (ipath.as_str(), g.line),
                    None => (path.as_str(), d.inline.gloss.iter().find(|g| g.0 == k).map(|g| g.2).unwrap_or(if d.line > 0 { d.line } else { 1 })),
                }
            };
            let known = crate::graph::def_variables(&params, &stmts);
            let touched: HashSet<&str> = stmts.iter().flat_map(|s| s.touches.iter().map(String::as_str)).chain(params.iter().map(String::as_str)).collect();
            for (key, name, at) in crate::graph::doc_sites(&body, &params, def_line, &cx, &lines, &keys) {
                let (p, line) = gloss_line(&key);
                if every && mg.gloss.iter().find(|g| g.0 == key).is_some_and(|g| crate::tex::split_symbol(&g.1).1.trim().is_empty()) {
                    find(p, line, &format!("{}:{key}", d.name), "doc", format!("`{}:{key}` gives a symbol and no words: say what `{name}` holds", d.name), vec![]);
                }
                if key.contains('@') || every {
                    if !touched.contains(name.as_str()) {
                        find(p, line, &format!("{}:{key}", d.name), "doc", format!("`{}:{key}` documents `{name}`, which `{}` does not use", d.name, d.name), vec![]);
                        continue;
                    }
                }
                if let Err(why) = at {
                    if key.contains('@') {
                        find(p, line, &format!("{}:{key}", d.name), "doc-site", format!("`{}:{key}` names no statement: {why}", d.name), vec![]);
                    }
                }
            }
            if every && has_entry {
                let documented: HashSet<String> = keys.iter().filter_map(|k| split_doc_key(k).map(|(n, _, _)| n)).collect();
                let at = e.map(|e| e.line).unwrap_or(1);
                for v in &known {
                    if !documented.contains(v) {
                        find(&ipath, at, &format!("{}:{v}", d.name), "undocumented", format!("`{}` has no glossary line for `{v}` (the module {EVERY_VARIABLE}): say what it holds", d.name), vec![]);
                    }
                }
            }
        }
    }
    // reading paths: every stop names a def, step or statement of the
    // program, every figure is one the program draws, and their prose
    // follows the rules of any other intent text
    if let Some(f) = &m.intent {
        let srcs: Vec<&str> = std::iter::once(m).chain(l.used.iter()).map(|md| md.text.src.as_str()).collect();
        for pb in f.paths() {
            if pb.stops.is_empty() {
                find(&ipath, pb.line, &format!("path {}", pb.id), "path", format!("path `{}` has no stops", pb.id), vec![]);
            }
            for st in &pb.stops {
                let anchor = format!("path {}:{}", pb.id, st.target);
                let Some((md, cd)) = l.find(&st.def) else {
                    find(&ipath, st.line, &anchor, "path", format!("`{}`: the program has no def `{}`", st.target, st.def), vec![]);
                    continue;
                };
                if let Some(sid) = &st.step {
                    if cd.step(sid).is_none() {
                        find(&ipath, st.line, &anchor, "path", format!("`{}`: `{}` has no step `{sid}`", st.target, st.def), vec![]);
                        continue;
                    }
                }
                let parsed = crate::parse_keeping_rungs(&md.text.src, SourceId(0)).ok();
                let known: HashSet<String> = match parsed.as_ref().and_then(|p| crate::graph::body_of(p, &st.def).map(|b| (p, b))) {
                    Some((p, (body, params, _))) => {
                        let cx = crate::graph::Cx::of(p);
                        let lines = crate::graph::Lines::new(&md.text.src);
                        let stmts = crate::graph::doc_statements(&body, &cx, &lines);
                        if let (Some(a), Some(def)) = (&st.anchor, p.items.iter().find_map(|i| if let Item::Def(dd) = i { (dd.name == st.def).then_some(dd) } else { None })) {
                            let cands = crate::graph::anchor_candidates(def, &cx, &lines);
                            let dt = Detail { label: "=".into(), anchor: a.clone(), text: String::new(), line: st.line, under: String::new() };
                            if crate::graph::resolve_anchor(&cands, st.step.as_deref(), &dt).is_none() {
                                find(&ipath, st.line, &anchor, "path", format!("`{}`: `={a}` names no statement of `{}`", st.target, st.def), vec![]);
                            }
                        }
                        crate::graph::def_variables(&params, &stmts).into_iter().collect()
                    }
                    None => HashSet::new(),
                };
                for dt in &st.details {
                    match dt.label.as_str() {
                        "figure" => {
                            let lit = format!("figure({}", quote(&dt.text));
                            if dt.text.is_empty() || !srcs.iter().any(|s| s.contains(&lit)) {
                                find(&ipath, dt.line, &anchor, "figure", format!("`figure {}`: no `figure(\"{}\", ...)` call in the program draws it", dt.text, dt.text), vec![]);
                            }
                        }
                        "run" | "predict" => {
                            for (name, _, _) in placeholders(&dt.text) {
                                if !known.contains(&name) {
                                    find(&ipath, dt.line, &anchor, "placeholder", format!("`{{{name}}}` names no variable of `{}`", st.def), vec![]);
                                }
                            }
                        }
                        _ => {}
                    }
                    let (nums, cites) = numbers_and_citations(&dt.text);
                    let prose = dt.label == ">" || dt.label == "predict";
                    for c in &cites {
                        if !tests.contains(c.as_str()) {
                            find(&ipath, dt.line, &anchor, "citation", format!("a path line cites [test \"{c}\"], and no test by that name is in {} or the modules it uses", m.text.name), vec![]);
                        }
                    }
                    if prose && !nums.is_empty() && cites.is_empty() {
                        find(&ipath, dt.line, &anchor, "number", format!("a path line states {} with no [test \"...\"] citation (or write it as a `run` line with {{name}})", nums.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")), vec![]);
                    }
                    let (spans, closed) = crate::tex::math_spans(&dt.text);
                    if !closed {
                        find(&ipath, dt.line, &anchor, "tex", "a `$` that opens math is never closed".into(), vec![]);
                    }
                    for sp in spans {
                        if let Err(why) = crate::tex::check(&sp) {
                            find(&ipath, dt.line, &anchor, "tex", format!("`{}`: {why}", sp.trim()), vec![]);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Check a module's `.debug` file (the debug lens) against its code: the file parses; each entry names a def of
/// the module (or `*`); each `at <step>` names a step of that def; and each
/// entry carries its def's current body hash, as an `.intent` entry does.
pub fn check_debug(l: &Loaded, dpath: &str, text: &str) -> Vec<Finding> {
    let m = &l.root;
    let path = &m.text.path;
    let mut out: Vec<Finding> = Vec::new();
    let mut find = |line: usize, anchor: &str, kind: &'static str, message: String| {
        out.push(Finding { path: dpath.to_string(), line, anchor: anchor.to_string(), kind, message, detail: vec![] });
    };
    if let Err(e) = crate::debug_lens::check_file(text) {
        for l in e.lines() {
            let (line, msg) = l.split_once(": ").unwrap_or(("0", l));
            find(line.parse().unwrap_or(0), "*", "parse", msg.to_string());
        }
        return out;
    }
    let mut seen: HashMap<String, usize> = HashMap::new();
    for e in crate::debug_lens::file_entries(text) {
        if e.name == "*" {
            continue;
        }
        if let Some(first) = seen.insert(e.name.clone(), e.line) {
            find(e.line, &e.name, "duplicate", format!("`{}` has a second entry; its first is at line {first}", e.name));
            continue;
        }
        let Some(d) = m.def(&e.name) else {
            find(e.line, &e.name, "orphan", format!("an entry for `{}`, which {path} does not define", e.name));
            continue;
        };
        for (line, id) in &e.at {
            if d.step(id).is_none() {
                find(*line, &format!("{}/{id}", d.name), "step", format!("`{}` has no step `{id}`", d.name));
            }
        }
        match (&e.stamp, &d.hash) {
            (None, _) => find(e.line, &d.name, "unstamped", format!("`{}` has no stamp; review its entry, then write `@{}` after the name", d.name, d.hash.clone().unwrap_or_default())),
            (Some(s), Some(h)) if s != h => {
                find(e.line, &d.name, "stale", format!("`{}` was stamped @{s} and its body is now {h}; check that the watches and holds still fit, then re-stamp it @{h}", d.name))
            }
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------------

/// One line of an explanation.
#[derive(Clone, Debug, PartialEq)]
pub struct ENode {
    /// "def", "step", "control", "test", "call"
    pub kind: &'static str,
    /// the def's or step's name; the control's word
    pub name: String,
    pub text: String,
    /// the module a def or call belongs to
    pub module: String,
    /// a call that names the def as a function value (a registry member)
    pub value: bool,
    /// "", "expanded above", "recursive", "depth" (it calls or steps
    /// further, and --depth stops here), or "unwritten" (a placeholder)
    pub note: &'static str,
    pub children: Vec<ENode>,
}

/// The intent walked along the static call graph from `from` (the top level
/// when `None`): each def's steps and control lines in order, and at each call
/// the callee's intent, expanded in place. `depth` k shows calls down to k
/// levels below the start (0: the def's own steps only). A def is expanded
/// once; later calls to it say so.
pub fn explain(l: &Loaded, from: Option<&str>, depth: Option<usize>) -> Result<ENode, String> {
    let start = from.unwrap_or(MAIN_DEF);
    let (m, d) = l.find(start).ok_or_else(|| {
        let names: Vec<&str> = l.root.defs.iter().map(|d| d.name.as_str()).collect();
        format!("no def named `{start}` in {} or the modules it uses (its own: {})", l.root.text.path, names.join(", "))
    })?;
    let mut expanded: HashMap<String, bool> = HashMap::new();
    let mut path: Vec<String> = Vec::new();
    Ok(expand_def(l, m, d, 0, depth, &mut expanded, &mut path))
}

pub(crate) fn what_of(m: &Mod, d: &CodeDef) -> String {
    let (_, entries) = m.entries();
    merged(d, entries.iter().find(|e| e.name == d.name)).field("what").unwrap_or("").to_string()
}

/// `expanded` records each def expanded so far and whether it had anything
/// under it; a def with nothing under it is listed again without a note.
fn expand_def(l: &Loaded, m: &Mod, d: &CodeDef, level: usize, depth: Option<usize>, expanded: &mut HashMap<String, bool>, path: &mut Vec<String>) -> ENode {
    path.push(d.name.clone());
    let (_, entries) = m.entries();
    let e = entries.iter().find(|e| e.name == d.name);
    let fst = file_step_texts(e);
    let inline_ctl: HashMap<usize, String> = d.inline.controls.iter().map(|(l, t, _)| (*l, t.clone())).collect();
    let code_nodes = code_as_intent(d, &|_| String::new(), &|_| None);
    let file_ctl = e.map(|e| control_map(&code_nodes, &e.outline)).unwrap_or_default();
    let ctx = Ctx { l, d, fst: &fst, inline_ctl: &inline_ctl, file_ctl: &file_ctl };
    let children = ctx.nodes(&d.outline, level, depth, expanded, path);
    path.pop();
    expanded.insert(d.name.clone(), !children.is_empty());
    ENode { kind: "def", name: d.name.clone(), text: what_of(m, d), module: m.text.name.clone(), value: false, note: if d.placeholder { "unwritten" } else { "" }, children }
}

struct Ctx<'a> {
    l: &'a Loaded,
    d: &'a CodeDef,
    fst: &'a HashMap<String, String>,
    inline_ctl: &'a HashMap<usize, String>,
    file_ctl: &'a HashMap<usize, (usize, String)>,
}

impl Ctx<'_> {
    fn nodes(&self, ns: &[ONode], level: usize, depth: Option<usize>, expanded: &mut HashMap<String, bool>, path: &mut Vec<String>) -> Vec<ENode> {
        let mut out: Vec<ENode> = Vec::new();
        let keep = kept_controls(ns, octl, |n| has_step(&n.children));
        for (n, keep) in ns.iter().zip(keep) {
            match &n.kind {
                OKind::Step(id) => {
                    let text = self.fst.get(id).cloned().unwrap_or_else(|| n.text.clone());
                    let children = self.nodes(&n.children, level, depth, expanded, path);
                    out.push(ENode { kind: "step", name: id.clone(), text, module: String::new(), value: false, note: "", children });
                }
                OKind::Control(c) if keep => {
                    let text = self.file_ctl.get(&n.line).map(|x| x.1.clone()).filter(|t| !is_bare_control(t)).or_else(|| self.inline_ctl.get(&n.line).cloned()).or_else(|| self.file_ctl.get(&n.line).map(|x| x.1.clone())).unwrap_or_else(|| n.text.clone());
                    let children = self.nodes(&n.children, level, depth, expanded, path);
                    out.push(ENode { kind: "control", name: c.sig().to_string(), text, module: String::new(), value: false, note: "", children });
                }
                OKind::Test(name) => {
                    let children = self.nodes(&n.children, level, depth, expanded, path);
                    if !children.is_empty() {
                        out.push(ENode { kind: "test", name: name.clone(), text: n.text.clone(), module: String::new(), value: false, note: "", children });
                    }
                }
                OKind::Call { name, value } => {
                    // the call graph is `viv deps`'s; a def named as a value is
                    // what a registry can pick, which `viv deps` does not list
                    if !*value && self.l.calls.get(&self.d.name).is_some_and(|cs| !cs.contains(name)) {
                        continue;
                    }
                    // one line per callee in a list; a def both passed and called is called
                    if let Some(x) = out.iter_mut().find(|x| x.kind == "call" && &x.name == name) {
                        x.value &= *value;
                        continue;
                    }
                    // `--depth k` shows calls down to k levels below the start
                    if depth.is_some_and(|k| level + 1 > k) {
                        continue;
                    }
                    let Some((cm, cd)) = self.l.find(name) else { continue };
                    let mut node = ENode { kind: "call", name: name.clone(), text: what_of(cm, cd), module: cm.text.name.clone(), value: *value, note: "", children: vec![] };
                    if cd.placeholder {
                        node.note = "unwritten";
                    } else if path.contains(name) {
                        node.note = "recursive";
                    } else if let Some(&had) = expanded.get(name) {
                        if had {
                            node.note = "expanded above";
                        }
                    } else if depth.is_some_and(|k| level + 1 >= k) {
                        if !cd.outline.is_empty() {
                            node.note = "depth";
                        }
                    } else {
                        node.children = expand_def(self.l, cm, cd, level + 1, depth, expanded, path).children;
                    }
                    out.push(node);
                }
                // a control holding no step: its calls are listed in place
                _ => {
                    let inner = self.nodes(&n.children, level, depth, expanded, path);
                    for x in inner {
                        if x.kind == "call" {
                            if let Some(y) = out.iter_mut().find(|y| y.kind == "call" && y.name == x.name) {
                                y.value &= x.value;
                                continue;
                            }
                        }
                        out.push(x);
                    }
                }
            }
        }
        out
    }
}

/// An explanation as indented text: defs and calls by name with their `what`,
/// steps as `id: text`, control lines as written.
pub fn explain_text(n: &ENode) -> String {
    fn go(n: &ENode, indent: usize, out: &mut String) {
        let pad = "  ".repeat(indent);
        let line = match n.kind {
            "def" => {
                let mut s = if n.text.is_empty() { n.name.clone() } else { format!("{}  {}", n.name, n.text) };
                if n.note == "unwritten" {
                    s.push_str("  (unwritten: a placeholder)");
                }
                s
            }
            "call" => {
                let mut s = format!("-> {}", n.name);
                if !n.text.is_empty() {
                    s.push_str(&format!("  {}", n.text));
                }
                if n.value {
                    s.push_str("  (as a value)");
                }
                match n.note {
                    "expanded above" => s.push_str("  (expanded above)"),
                    "recursive" => s.push_str("  (recursive)"),
                    "unwritten" => s.push_str("  (unwritten: a placeholder)"),
                    _ => {}
                }
                s
            }
            "step" => {
                if n.text.is_empty() {
                    format!("{}:", n.name)
                } else {
                    format!("{}: {}", n.name, n.text)
                }
            }
            _ => n.text.clone(),
        };
        out.push_str(&format!("{pad}{line}\n"));
        for c in &n.children {
            go(c, indent + 1, out);
        }
    }
    let mut out = String::new();
    go(n, 0, &mut out);
    out
}

pub fn explain_json(n: &ENode) -> String {
    format!(
        "{{\"kind\":\"{}\",\"name\":{},\"text\":{},\"module\":{},\"value\":{},\"note\":\"{}\",\"children\":[{}]}}",
        n.kind,
        json_str(&n.name),
        json_str(&n.text),
        json_str(&n.module),
        n.value,
        n.note,
        n.children.iter().map(explain_json).collect::<Vec<_>>().join(",")
    )
}
