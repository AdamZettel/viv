// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Source locations (1.2) and the source map renderers read from.
//!
//! A `Span` is per-instance metadata: it records where a node came from. It is
//! deliberately *not* part of a node's structural hash (see [`crate::ir::NodeId`]),
//! so two structurally identical nodes share an id but may carry different spans.

/// Identifies a source file within a compilation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SourceId(pub u32);

/// A half-open byte range `[start, end)` within a source file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub source: SourceId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(source: SourceId, start: u32, end: u32) -> Self {
        Self { source, start, end }
    }

    /// A zero-width span for nodes with no source origin (e.g. kit-generated).
    pub fn synthetic(source: SourceId) -> Self {
        Self { source, start: 0, end: 0 }
    }

    /// The smallest span covering both `self` and `other` (must share a source).
    pub fn merge(self, other: Span) -> Span {
        debug_assert_eq!(self.source, other.source, "cannot merge spans across sources");
        Span {
            source: self.source,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// One source file: a name, its full text, the language it is written in and
/// where it came from.
#[derive(Clone, Debug)]
pub struct SourceFile {
    pub name: String,
    pub text: String,
    /// The language tag of the text (`viv`, `python`, ...), as the language
    /// that registered it names it; empty when nobody said. The kit does not
    /// interpret it: renderers and tools may show it.
    pub lang: String,
    /// Where the text came from: written by a person, spliced together from
    /// other sources, or generated from another source.
    pub origin: Origin,
    /// The byte offset of each line's start, made on the first lookup: a
    /// lookup is then a binary search, not a scan of the text (a program
    /// with 20,000 error sites in 12 MB of source spent a minute scanning).
    line_starts: std::sync::OnceLock<Vec<u32>>,
}

impl SourceFile {
    /// (0-based line, byte offset of its start) for a byte offset.
    fn line_of(&self, offset: u32) -> (usize, usize) {
        let starts = self.line_starts.get_or_init(|| {
            let mut v = vec![0u32];
            v.extend(self.text.bytes().enumerate().filter(|&(_, b)| b == b'\n').map(|(i, _)| i as u32 + 1));
            v
        });
        let offset = offset.min(self.text.len() as u32);
        // the last line starting at or before offset; a line starting exactly
        // at offset (just after a newline at offset - 1) is that line
        let line = starts.partition_point(|&st| st <= offset) - 1;
        (line, starts[line] as usize)
    }
}

/// Where a source's text came from. Spans stay in the text a parser saw;
/// following a span back to the text a person wrote happens only when a
/// renderer asks ([`SourceMap::resolve`]), so origins never touch the IR or
/// its hashes.
#[derive(Clone, Debug, Default)]
pub enum Origin {
    /// The text as a person wrote it, or a text whose origin is not recorded.
    #[default]
    Written,
    /// A text assembled from pieces of other sources: a program with its
    /// modules spliced in. Bytes outside every piece (a marker line the
    /// splicer wrote) belong to this source itself.
    Composite(Vec<Piece>),
    /// A text generated from another source: a translation, a rule's
    /// instantiation, a printer's output. `via` names what generated it
    /// (`py2viv`, `rule np.dot`); `map` gives, for each of its lines, the span
    /// in the parent source that line came from.
    Derived { via: String, map: LineMap },
}

/// A byte range of a composite source that is a copy of bytes of another
/// source: `at` in the composite, starting at `offset` in `source`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub at: (u32, u32),
    pub source: SourceId,
    pub offset: u32,
}

/// For each line of a generated text (0-based), the span it came from, or
/// `None` for a line with no origin (a helper the generator added, a blank
/// line). This is the shape of a translator's line map and of a C backend's
/// position table; [`crate::print::Writer`] produces one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineMap {
    lines: Vec<Option<Span>>,
}

impl LineMap {
    pub fn new(lines: Vec<Option<Span>>) -> Self {
        Self { lines }
    }

    /// The origin of line `line` (0-based); `None` past the end.
    pub fn get(&self, line: usize) -> Option<Span> {
        self.lines.get(line).copied().flatten()
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn lines(&self) -> &[Option<Span>] {
        &self.lines
    }
}

/// One step of a span's origin chain: the span, in its own source, and what
/// produced that source from the next one (the `via` of a derived source;
/// `None` for a composite or the last frame).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub span: Span,
    pub via: Option<String>,
}

/// How far `resolve` follows a chain before it stops: a guard against a map
/// that points in a circle, far above any real depth (a module spliced into a
/// program, generated from a Python file, through a rule, is three).
const MAX_CHAIN: usize = 32;

/// Holds every source file in a compilation and resolves byte offsets to
/// human-facing (line, column) pairs. Renderers (1.7) consult this; nothing in
/// the compiler core needs it, which keeps positions out of the IR itself.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// Add a written source with no language tag.
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> SourceId {
        self.add_lang(name, text, "")
    }

    /// Add a written source in language `lang`.
    pub fn add_lang(&mut self, name: impl Into<String>, text: impl Into<String>, lang: impl Into<String>) -> SourceId {
        let id = SourceId(self.files.len() as u32);
        self.files.push(SourceFile {
            name: name.into(),
            text: text.into(),
            lang: lang.into(),
            origin: Origin::Written,
            line_starts: std::sync::OnceLock::new(),
        });
        id
    }

    /// Add a source generated from another: `map` gives each of its lines' origin.
    pub fn add_derived(
        &mut self,
        name: impl Into<String>,
        text: impl Into<String>,
        lang: impl Into<String>,
        via: impl Into<String>,
        map: LineMap,
    ) -> SourceId {
        let id = self.add_lang(name, text, lang);
        self.set_origin(id, Origin::Derived { via: via.into(), map });
        id
    }

    /// Add a source generated at one place in another: a rule's instantiation
    /// at a call site. Every line of it comes from `site`.
    pub fn add_generated(
        &mut self,
        name: impl Into<String>,
        text: impl Into<String>,
        lang: impl Into<String>,
        via: impl Into<String>,
        site: Span,
    ) -> SourceId {
        let text = text.into();
        let n = text.lines().count().max(1);
        self.add_derived(name, text, lang, via, LineMap::new(vec![Some(site); n]))
    }

    /// Record where a source came from. A composite is usually added first
    /// (it is the text the parser reads, so its spans are in it) and its
    /// pieces' sources after, which is why this is separate from `add`.
    /// A composite's pieces are sorted here.
    pub fn set_origin(&mut self, id: SourceId, origin: Origin) {
        let origin = match origin {
            Origin::Composite(mut pieces) => {
                pieces.sort_by_key(|p| p.at.0);
                Origin::Composite(pieces)
            }
            o => o,
        };
        if let Some(f) = self.files.get_mut(id.0 as usize) {
            f.origin = origin;
        }
    }

    pub fn file(&self, id: SourceId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The origin chain of `span`: the span itself first, then each span it
    /// was copied or generated from, ending at the first source with no
    /// further origin (usually what a person wrote). A span in a written
    /// source is a chain of one.
    pub fn resolve(&self, span: Span) -> Vec<Frame> {
        let mut chain = vec![Frame { span, via: None }];
        let mut cur = span;
        while chain.len() < MAX_CHAIN {
            let Some(file) = self.file(cur.source) else { break };
            let next = match &file.origin {
                Origin::Written => None,
                Origin::Composite(pieces) => map_composite(pieces, cur).map(|s| (s, None)),
                Origin::Derived { via, map } => {
                    let (line, _) = file.line_of(cur.start);
                    map.get(line).map(|s| (s, Some(via.clone())))
                }
            };
            let Some((span, via)) = next else { break };
            chain.push(Frame { span, via });
            cur = span;
        }
        chain
    }

    /// The last span of `span`'s origin chain.
    pub fn origin_of(&self, span: Span) -> Span {
        self.resolve(span).last().map_or(span, |f| f.span)
    }

    /// 1-based (line, column) for a byte offset, plus the text of that line.
    /// This is the position in the source `id` itself; [`SourceMap::resolve`]
    /// follows it back to where the text came from.
    pub fn locate(&self, id: SourceId, offset: u32) -> Option<(u32, u32, &str)> {
        let file = self.file(id)?;
        let offset = offset.min(file.text.len() as u32) as usize;
        let (line, line_start) = file.line_of(offset as u32);
        let line_no = line as u32 + 1;
        let line_end = file.text[line_start..]
            .find('\n')
            .map(|n| line_start + n)
            .unwrap_or(file.text.len());
        let col = (offset - line_start) as u32 + 1;
        Some((line_no, col, &file.text[line_start..line_end]))
    }

    /// The byte offset of 1-based `line` and `col` (columns count bytes) in
    /// source `id`, clamped to the text; `None` for an unknown source.
    pub fn offset_of(&self, id: SourceId, line: u32, col: u32) -> Option<u32> {
        let text = &self.file(id)?.text;
        let mut start = 0usize;
        for _ in 1..line.max(1) {
            match text[start..].find('\n') {
                Some(n) => start += n + 1,
                None => return Some(text.len() as u32),
            }
        }
        let end = text[start..].find('\n').map_or(text.len(), |n| start + n);
        Some((start + (col.max(1) as usize - 1)).min(end) as u32)
    }
}

/// A span of a composite, carried into the piece its start is in; its end is
/// clamped to that piece.
fn map_composite(pieces: &[Piece], span: Span) -> Option<Span> {
    let i = pieces.partition_point(|p| p.at.0 <= span.start);
    let p = pieces.get(i.checked_sub(1)?)?;
    if span.start >= p.at.1 {
        return None;
    }
    let end = span.end.clamp(span.start, p.at.1);
    Some(Span { source: p.source, start: p.offset + (span.start - p.at.0), end: p.offset + (end - p.at.0) })
}
