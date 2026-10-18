// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! A text writer that remembers where each line of its output came from.
//!
//! A printer (a translator's emitter, a pretty-printer, a code generator)
//! writes through a [`Writer`], giving each piece of text the span of the node
//! it prints. [`Writer::finish`] returns the text and a [`LineMap`]: for each
//! output line, the span of the first piece written on it that had one. The
//! text can then be registered with [`crate::span::SourceMap::add_derived`],
//! so a diagnostic located in the printed text is shown at the source the
//! printer read.

use crate::span::{LineMap, Span};

#[derive(Default)]
pub struct Writer {
    text: String,
    lines: Vec<Option<Span>>,
}

impl Writer {
    pub fn new() -> Self {
        Self { text: String::new(), lines: vec![None] }
    }

    /// Append `s`, which came from `origin` (`None` for text with no source:
    /// a helper, punctuation the printer chose). Each line `s` starts or
    /// continues takes `origin` unless an earlier write already gave it one.
    pub fn write(&mut self, s: &str, origin: Option<Span>) {
        let mut rest = s;
        loop {
            let (part, more) = match rest.find('\n') {
                Some(n) => (&rest[..n], Some(&rest[n + 1..])),
                None => (rest, None),
            };
            if !part.is_empty() || more.is_some() {
                let last = self.lines.last_mut().expect("a writer always has a current line");
                if last.is_none() {
                    *last = origin;
                }
            }
            self.text.push_str(part);
            match more {
                Some(m) => {
                    self.text.push('\n');
                    self.lines.push(None);
                    rest = m;
                }
                None => break,
            }
        }
    }

    /// Append `s` and end the line.
    pub fn line(&mut self, s: &str, origin: Option<Span>) {
        self.write(s, origin);
        self.write("\n", None);
    }

    /// The text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text and each line's origin. A text that ends with a newline has
    /// no entry for the empty line after it.
    pub fn finish(mut self) -> (String, LineMap) {
        if self.text.ends_with('\n') || self.text.is_empty() {
            self.lines.pop();
        }
        (self.text, LineMap::new(self.lines))
    }
}
