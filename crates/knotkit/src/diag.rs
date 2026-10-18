// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Diagnostics as data (1.7). Errors, warnings, and notes are structured
//! values, never printed strings. They are produced by the compiler and
//! *consumed* by renderers; adding a renderer touches no producing code.

use crate::span::{Frame, SourceMap, Span};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

/// A span with an explanatory message attached.
#[derive(Clone, Debug)]
pub struct Label {
    pub span: Span,
    pub message: String,
}

/// A structured diagnostic: severity, a headline message, an optional primary
/// label (the location the diagnostic is "about"), secondary labels, and notes.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub primary: Option<Label>,
    pub secondary: Vec<Label>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            primary: None,
            secondary: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn with_primary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.primary = Some(Label { span, message: message.into() });
        self
    }

    pub fn with_secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.secondary.push(Label { span, message: message.into() });
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

/// Consumes diagnostics and produces a presentation. The kit can grow ANSI,
/// JSON, and HTML renderers (1.7) without any producer being aware of them.
pub trait Renderer {
    fn render(&self, diag: &Diagnostic, sources: &SourceMap) -> String;
}

/// A plain-text renderer, suitable for tests and simple CLI output.
pub struct PlainRenderer;

impl Renderer for PlainRenderer {
    fn render(&self, diag: &Diagnostic, sources: &SourceMap) -> String {
        let mut out = String::new();
        out.push_str(diag.severity.as_str());
        out.push_str(": ");
        out.push_str(&diag.message);
        out.push('\n');

        if let Some(label) = &diag.primary {
            render_label(&mut out, label, sources, '^');
        }
        for label in &diag.secondary {
            render_label(&mut out, label, sources, '-');
        }
        for note in &diag.notes {
            out.push_str("  note: ");
            out.push_str(note);
            out.push('\n');
        }
        out
    }
}

fn render_label(out: &mut String, label: &Label, sources: &SourceMap, caret: char) {
    use std::fmt::Write;
    // The label is shown where its text came from: a line of a module spliced
    // into a program is shown in the module, a line of generated code at the
    // source it was generated from. Each generator on the way is a note.
    let chain = sources.resolve(label.span);
    let span = chain.last().map_or(label.span, |f| f.span);
    let Some((line, col, line_text)) = sources.locate(span.source, span.start) else {
        let _ = writeln!(out, "  --> <unknown location>: {}", label.message);
        return;
    };
    let name = sources
        .file(span.source)
        .map(|f| f.name.as_str())
        .unwrap_or("<source>");
    let _ = writeln!(out, "  --> {}:{}:{}", name, line, col);
    let gutter = format!("{:>4} | ", line);
    let _ = writeln!(out, "{}{}", gutter, line_text);

    // Underline the span. A span that came through a generator is clamped to
    // its first line (a generated line maps to a whole statement, which may
    // run over several); a span in the text as parsed is not, as before.
    let mut width = span.end.saturating_sub(span.start).max(1) as usize;
    if chain.len() > 1 {
        width = width.min(line_text.len().saturating_sub(col as usize - 1)).max(1);
    }
    let pad = " ".repeat(gutter.len() + (col as usize - 1));
    let underline: String = std::iter::repeat(caret).take(width).collect();
    let _ = writeln!(out, "{}{} {}", pad, underline, label.message);
    for note in via_notes(&chain, sources) {
        let _ = writeln!(out, "  note: {}", note);
    }
}

/// One note per generator in an origin chain, naming where in the generated
/// text the span was: "via py2viv: prog.py.viv:40:5". A composite (spliced
/// modules) adds no note: the spliced text is not something a person reads.
pub fn via_notes(chain: &[Frame], sources: &SourceMap) -> Vec<String> {
    let mut notes = Vec::new();
    for w in chain.windows(2) {
        let Some(via) = &w[1].via else { continue };
        let at = w[0].span;
        let name = sources.file(at.source).map_or("<source>", |f| f.name.as_str());
        match sources.locate(at.source, at.start) {
            Some((line, col, _)) => notes.push(format!("via {via}: {name}:{line}:{col}")),
            None => notes.push(format!("via {via}")),
        }
    }
    notes
}
