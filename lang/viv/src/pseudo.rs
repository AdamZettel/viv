// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The pseudocode view: a def read as an algorithm box.
//!
//! `viv view <file> --lens pseudocode [--def d] [--words-only]` prints each def
//! of the module (or the one def named) as:
//!
//! - a heading, the def's signature, and its `what` from the intent;
//! - each step as its intent text, numbered, with the statements inside it
//!   indented under it;
//! - each control structure as a skeleton line (`for each head:`, `if ...:`,
//!   `otherwise:`), with the outline's authored control text where the intent
//!   has one;
//! - each statement as written, with every phrase shown as its words (the
//!   opening `take` left out) and `=` written `←`. A statement that is one
//!   phrase is marked `[phrase]`, one with a phrase inside it `[part]`, and one
//!   with none `[code]`; a `=` line entry of the intent is shown under its
//!   statement.
//!
//! With `--words-only` a `[code]` statement is a gap, `[code, line N]`, so the
//! view shows only the words someone wrote: intent and phrases. Nothing in the
//! view is generated from the code but the skeleton words; the counts at the
//! top of each def say how much of it reads as words.

use std::collections::HashMap;

use knotkit::span::SourceId;

use crate::ast::{Def, Item, Program, Stmt};
use crate::lens::{merged, Loaded, Mod, Node, NodeKind, OKind, ONode};
use crate::parser::MAIN_DEF;

const WIDTH: usize = 78;

/// How a statement reads.
#[derive(Clone, Copy, PartialEq)]
enum Reads {
    /// the statement's value is one phrase
    Phrase,
    /// a phrase is part of it
    Part,
    /// no phrase
    Code,
}

#[derive(Default)]
struct Counts {
    phrase: usize,
    part: usize,
    code: usize,
}

struct Ctx<'a> {
    src: &'a str,
    prog: &'a Program,
    /// the step texts of the def's intent, by step id
    steps: HashMap<String, String>,
    /// the outline's control lines, by the code line they stand for
    controls: HashMap<usize, String>,
    /// the `=` line entries: (anchor name, whether it names a call, k) -> text
    lines: Vec<((String, bool, usize), String)>,
    /// how many statements so far assigned each name, for `x#2` anchors
    assigned: HashMap<String, usize>,
    words_only: bool,
    counts: Counts,
    step_no: Vec<usize>,
    /// where the header of the step entered last ends (`step id:` or
    /// `step id {`): a statement on the header's line starts after it
    step_head_end: u32,
    out: Vec<String>,
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in text.split_whitespace() {
        if !cur.is_empty() && cur.chars().count() + 1 + w.chars().count() > width {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn line_of(src: &str, at: u32) -> usize {
    src[..(at as usize).min(src.len())].matches('\n').count() + 1
}

impl Ctx<'_> {
    fn push(&mut self, indent: usize, text: &str, mark: &str) {
        let pad = "  ".repeat(indent);
        let lines = wrap(text, WIDTH.saturating_sub(pad.len()).max(30));
        let n = lines.len();
        for (k, l) in lines.into_iter().enumerate() {
            // a wrapped line hangs two spaces in
            let l = if k > 0 { format!("  {l}") } else { l };
            if k + 1 == n && !mark.is_empty() {
                let used = pad.chars().count() + l.chars().count();
                let gap = (WIDTH + 2).saturating_sub(used).max(2);
                self.out.push(format!("{pad}{l}{}{mark}", " ".repeat(gap)));
            } else {
                self.out.push(format!("{pad}{l}"));
            }
        }
    }

    /// The source of a statement from `start` to `end`, with each phrase in it
    /// as its words, and how it reads.
    fn words(&self, start: u32, end: u32) -> (String, Reads) {
        let mut inside: Vec<(u32, u32)> = self.prog.phrases.iter().filter(|u| u.span.source == SourceId(0) && u.span.start >= start && u.span.end <= end).map(|u| (u.span.start, u.span.end)).collect();
        inside.sort();
        // a phrase in a phrase's hole is part of the outer one
        let mut outer: Vec<(u32, u32)> = Vec::new();
        for p in inside {
            if !outer.last().is_some_and(|o| p.0 < o.1) {
                outer.push(p);
            }
        }
        let mut text = String::new();
        let mut at = start as usize;
        for (a, b) in &outer {
            text.push_str(&self.src[at..*a as usize]);
            text.push_str(&strip_take(&self.src[*a as usize..*b as usize]));
            at = *b as usize;
        }
        text.push_str(&self.src[at..end as usize]);
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let reads = if outer.is_empty() {
            Reads::Code
        } else {
            // one phrase that is the whole value: after `=`, after `return`,
            // or the whole statement
            let value_start = value_start(&self.src[start as usize..end as usize]).map(|k| start as usize + k);
            let lone = outer.len() == 1 && {
                let (a, b) = outer[0];
                let before = &self.src[value_start.unwrap_or(start as usize)..a as usize];
                let after = &self.src[b as usize..end as usize];
                before.trim().is_empty() && after.trim().is_empty()
            };
            if lone { Reads::Phrase } else { Reads::Part }
        };
        (arrow(&text), reads)
    }

    fn line_entry(&self, name: &str, call: bool, k: usize) -> Option<&str> {
        self.lines.iter().find(|(t, _)| t.0 == name && t.1 == call && t.2 == k).map(|(_, s)| s.as_str())
    }

    fn stmt(&mut self, s: &Stmt, indent: usize, i: &mut usize, body: &[Stmt]) {
        match s {
            Stmt::StepEnter { id, span, .. } => {
                self.step_head_end = span.end;
                let n = self.step_no.last_mut().map(|n| {
                    *n += 1;
                    *n
                });
                let text = self.steps.get(id).cloned().filter(|t| !t.is_empty()).unwrap_or_else(|| format!("[step {id}: no intent]"));
                self.push(indent, &format!("{}. {text}", n.unwrap_or(1)), "");
                self.step_no.push(0);
                // the statements up to the matching exit, one level in
                *i += 1;
                while *i < body.len() {
                    if matches!(&body[*i], Stmt::StepExit { id: x, .. } if x == id) {
                        break;
                    }
                    let st = body[*i].clone();
                    self.stmt(&st, indent + 1, i, body);
                    *i += 1;
                }
                self.step_no.pop();
            }
            Stmt::StepExit { .. } | Stmt::Lens { .. } | Stmt::Test { .. } | Stmt::Narrate { .. } => {}
            Stmt::For { var, body: inner, span, .. } => {
                let line = line_of(self.src, span.start);
                let text = self.controls.get(&line).cloned().unwrap_or_else(|| format!("for each {var}"));
                self.push(indent, &format!("{text}:"), "");
                self.block(inner, indent + 1);
            }
            Stmt::While { cond, body: inner, span } => {
                let line = line_of(self.src, span.start);
                let text = self.controls.get(&line).cloned().unwrap_or_else(|| {
                    let c = cond.span();
                    format!("repeat while {}", self.words(c.start, c.end).0)
                });
                self.push(indent, &format!("{text}:"), "");
                self.block(inner, indent + 1);
            }
            Stmt::If { cond, then_body, else_body, span } => {
                let line = line_of(self.src, span.start);
                let text = self.controls.get(&line).cloned().unwrap_or_else(|| {
                    let c = cond.span();
                    format!("if {}", self.words(c.start, c.end).0)
                });
                self.push(indent, &format!("{text}:"), "");
                self.block(then_body, indent + 1);
                if !else_body.is_empty() {
                    self.push(indent, "otherwise:", "");
                    self.block(else_body, indent + 1);
                }
            }
            other => {
                let sp = other.span();
                let line = line_of(self.src, sp.start);
                // the statement's own text: from its first line's start, so an
                // `a, b = ...` keeps its names
                let mut line_start = self.src[..sp.start as usize].rfind('\n').map(|k| k + 1).unwrap_or(0) as u32;
                // a one-line step's statement: after its `step id:` on that line
                if (line_start..=sp.start).contains(&self.step_head_end) {
                    line_start = self.step_head_end;
                }
                let (text, reads) = self.words(line_start, sp.end);
                let mark = match reads {
                    Reads::Phrase => {
                        self.counts.phrase += 1;
                        "[phrase]"
                    }
                    Reads::Part => {
                        self.counts.part += 1;
                        "[part]"
                    }
                    Reads::Code => {
                        self.counts.code += 1;
                        "[code]"
                    }
                };
                let anchor = match other {
                    Stmt::Assign { name, .. } => {
                        let k = self.assigned.entry(name.clone()).or_insert(0);
                        *k += 1;
                        Some((name.clone(), false, *k))
                    }
                    Stmt::ExprStmt { expr: crate::ast::Expr::Call { callee, .. }, .. } => Some((callee.clone(), true, 1)),
                    _ => None,
                };
                let entry = anchor.and_then(|(n, c, k)| self.line_entry(&n, c, k).map(str::to_string));
                if reads == Reads::Code && self.words_only {
                    match &entry {
                        Some(e) => self.push(indent, e, ""),
                        None => self.push(indent, &format!("[code, line {line}]"), ""),
                    }
                    return;
                }
                self.push(indent, &text, mark);
                if let Some(e) = entry {
                    self.push(indent + 2, &format!("— {e}"), "");
                }
            }
        }
    }

    fn block(&mut self, body: &[Stmt], indent: usize) {
        let mut i = 0;
        while i < body.len() {
            let s = body[i].clone();
            self.stmt(&s, indent, &mut i, body);
            i += 1;
        }
    }
}

/// `take the norm of v` as `the norm of v`; a bare verb stays.
fn strip_take(phrase: &str) -> String {
    phrase.strip_prefix("take ").map(str::to_string).unwrap_or_else(|| phrase.to_string())
}

/// Where the value of an assignment or a `return` starts in a statement's text.
fn value_start(text: &str) -> Option<usize> {
    if let Some(r) = text.trim_start().strip_prefix("return ") {
        return Some(text.len() - r.len());
    }
    let mut depth = 0i32;
    let b = text.as_bytes();
    for (k, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '=' if depth == 0 => {
                let prev = if k > 0 { b[k - 1] } else { b' ' };
                let next = b.get(k + 1).copied().unwrap_or(b' ');
                if !b"=!<>+-*/".contains(&prev) && next != b'=' {
                    return Some(k + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The statement's first top-level ` = ` written ` ← `.
fn arrow(text: &str) -> String {
    match value_start(text) {
        Some(k) if text[..k].ends_with('=') && !text.trim_start().starts_with("return ") => format!("{}←{}", &text[..k - 1], &text[k..]),
        _ => text.to_string(),
    }
}

fn collect_lines(nodes: &[Node], out: &mut Vec<((String, bool, usize), String)>) {
    for n in nodes {
        for d in &n.details {
            if d.label == "=" {
                out.push((d.target(), d.text.split_whitespace().collect::<Vec<_>>().join(" ")));
            }
        }
        collect_lines(&n.children, out);
    }
}

/// The intent outline's step texts by id, and its control lines in order.
fn collect_outline(nodes: &[Node], controls: &mut Vec<String>, steps: &mut HashMap<String, String>) {
    for n in nodes {
        match &n.kind {
            NodeKind::Control(_) => controls.push(n.text.clone()),
            NodeKind::Step(id) => {
                steps.insert(id.clone(), n.text.clone());
            }
            NodeKind::Note => {}
        }
        collect_outline(&n.children, controls, steps);
    }
}

/// The code lines of the control structures that hold steps, in order: the
/// ones the intent outline has a control line for.
fn code_controls(nodes: &[ONode], out: &mut Vec<usize>) {
    fn holds_step(ns: &[ONode]) -> bool {
        ns.iter().any(|n| matches!(n.kind, OKind::Step(_)) || holds_step(&n.children))
    }
    for n in nodes {
        if matches!(n.kind, OKind::Control(_)) && holds_step(&n.children) {
            out.push(n.line);
        }
        code_controls(&n.children, out);
    }
}

fn def_view(m: &Mod, prog: &Program, d: &Def, words_only: bool) -> Vec<String> {
    let (_, entries) = m.entries();
    let code = m.def(&d.name);
    let mg = code.map(|c| merged(c, entries.iter().find(|e| e.name == d.name)));
    let mut steps = HashMap::new();
    let mut controls = HashMap::new();
    let mut lines = Vec::new();
    if let (Some(mg), Some(code)) = (&mg, code) {
        // the outline's control lines, matched in order to the code's
        // control structures that hold steps
        let mut texts = Vec::new();
        collect_outline(&mg.outline, &mut texts, &mut steps);
        let mut at = Vec::new();
        code_controls(&code.outline, &mut at);
        if at.len() == texts.len() {
            controls = at.into_iter().zip(texts).collect();
        }
        for det in &mg.details {
            if det.label == "=" {
                lines.push((det.target(), det.text.split_whitespace().collect::<Vec<_>>().join(" ")));
            }
        }
        collect_lines(&mg.outline, &mut lines);
    }
    let mut cx = Ctx {
        src: &m.text.src,
        prog,
        steps,
        controls,
        lines,
        assigned: HashMap::new(),
        words_only,
        counts: Counts::default(),
        step_no: vec![0],
        step_head_end: 0,
        out: Vec::new(),
    };
    cx.block(&d.body, 1);
    let sig = format!("{}({})", d.name, d.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>().join(", "));
    let c = &cx.counts;
    let total = c.phrase + c.part + c.code;
    let mut out = vec![format!("{sig}  —  {total} statements: {} a phrase, {} with a phrase in them, {} code", c.phrase, c.part, c.code)];
    if let Some(w) = mg.as_ref().and_then(|mg| mg.field("what")) {
        for l in wrap(w, WIDTH - 2) {
            out.push(format!("  {l}"));
        }
    }
    out.push(String::new());
    out.extend(cx.out);
    out
}

/// The pseudocode view of the loaded program's module, or of one def in it or
/// in a module it uses.
pub fn view(l: &Loaded, def: Option<&str>, words_only: bool) -> Result<String, String> {
    let mods: Vec<&Mod> = std::iter::once(&l.root).chain(l.used.iter()).collect();
    let mut blocks: Vec<Vec<String>> = Vec::new();
    for m in mods {
        let prog = crate::parse_keeping_rungs(&m.text.src, SourceId(0)).map_err(|ds| format!("{}: {}", m.text.path, crate::join_errs(&ds)))?;
        for it in &prog.items {
            let Item::Def(d) = it else { continue };
            // a lambda body's def is shown in the lambda (crate::surface)
            if crate::update::is_expression_def(&d.name) {
                continue;
            }
            let wanted = match def {
                Some(name) => d.name == name,
                None => std::ptr::eq(m, &l.root) && d.name != MAIN_DEF,
            };
            if wanted {
                blocks.push(def_view(m, &prog, d, words_only));
            }
        }
        if def.is_some() && !blocks.is_empty() {
            break;
        }
    }
    if blocks.is_empty() {
        return Err(match def {
            Some(name) => format!("no def `{name}` in the program or the modules it uses"),
            None => "the module has no def".to_string(),
        });
    }
    Ok(blocks.into_iter().map(|b| b.join("\n") + "\n").collect::<Vec<_>>().join("\n"))
}
