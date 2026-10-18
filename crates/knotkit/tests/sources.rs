// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Source origins: composite and derived sources, the origin chain, the
//! renderer that follows it, and the writer that records line origins.

use knotkit::diag::{Diagnostic, PlainRenderer, Renderer};
use knotkit::print::Writer;
use knotkit::span::{LineMap, Origin, Piece, SourceMap, Span};

/// A program with one module spliced in: the expanded text is the composite,
/// registered first (its spans are what a parser produces), the files after.
fn spliced() -> (SourceMap, String) {
    let main = "x = 1\nimport m\nshow x\n";
    let module = "def f(a) {\n    return a[5]\n}\n";
    let expanded = "x = 1\n# ---- m ----\ndef f(a) {\n    return a[5]\n}\n# ---- end ----\nshow x\n";
    let mut sm = SourceMap::new();
    let c = sm.add_lang("prog.viv", expanded, "viv");
    let m = sm.add_lang("prog.viv", main, "viv");
    let md = sm.add_lang("lib/m.viv", module, "viv");
    // line by line: x = 1 (main line 1), the module's three lines, show x (main line 3)
    let line = |t: &str, n: usize| t.lines().take(n).map(|l| l.len() as u32 + 1).sum::<u32>();
    let pieces = vec![
        Piece { at: (0, 6), source: m, offset: 0 },
        Piece { at: (line(expanded, 2), line(expanded, 5)), source: md, offset: 0 },
        Piece { at: (line(expanded, 6), line(expanded, 7)), source: m, offset: line(main, 2) },
    ];
    sm.set_origin(c, Origin::Composite(pieces));
    (sm, expanded.to_string())
}

#[test]
fn a_written_source_is_its_own_origin() {
    let mut sm = SourceMap::new();
    let id = sm.add("a.viv", "x = 1\n");
    let s = Span::new(id, 4, 5);
    assert_eq!(sm.resolve(s).len(), 1);
    assert_eq!(sm.origin_of(s), s);
    assert_eq!(sm.file(id).unwrap().lang, "");
}

#[test]
fn locate_through_a_composite() {
    let (sm, expanded) = spliced();
    let at = expanded.find("a[5]").unwrap() as u32;
    let o = sm.origin_of(Span::new(knotkit::span::SourceId(0), at, at + 4));
    assert_eq!(sm.file(o.source).unwrap().name, "lib/m.viv");
    let (line, col, text) = sm.locate(o.source, o.start).unwrap();
    assert_eq!((line, col, text), (2, 12, "    return a[5]"));
    assert_eq!(o.end - o.start, 4);
    // the main program's last line comes back as its line 3
    let at = expanded.find("show").unwrap() as u32;
    let o = sm.origin_of(Span::new(knotkit::span::SourceId(0), at, at + 4));
    assert_eq!(sm.locate(o.source, o.start).unwrap().0, 3);
    // a marker line belongs to the composite itself
    let at = expanded.find("# ---- m").unwrap() as u32;
    let s = Span::new(knotkit::span::SourceId(0), at, at + 1);
    assert_eq!(sm.origin_of(s), s);
}

#[test]
fn a_span_is_clamped_to_its_piece() {
    let (sm, expanded) = spliced();
    let at = expanded.find("return").unwrap() as u32;
    let o = sm.origin_of(Span::new(knotkit::span::SourceId(0), at, expanded.len() as u32));
    let module = &sm.file(o.source).unwrap().text;
    assert_eq!(o.end as usize, module.len());
}

#[test]
fn locate_through_a_derived_source_and_a_composite() {
    // prog.py -> (py2viv) generated Viv -> spliced after a prelude line
    let py = "import numpy as np\na = np.zeros(3)\nx = a[5]\n";
    let viv = "# header\na = zeros(3)\nx = a[5.0]\n";
    let expanded = format!("pre = 1\n{viv}");
    let mut sm = SourceMap::new();
    let c = sm.add_lang("prog.py", expanded.as_str(), "viv");
    let p = sm.add_lang("prog.py", py, "python");
    let stmt = |line: u32, text: &str| {
        let start = py.lines().take(line as usize - 1).map(|l| l.len() as u32 + 1).sum::<u32>();
        Span::new(p, start, start + text.len() as u32)
    };
    let map = LineMap::new(vec![None, Some(stmt(2, "a = np.zeros(3)")), Some(stmt(3, "x = a[5]"))]);
    let g = sm.add_derived("prog.py.viv", viv, "viv", "py2viv", map);
    sm.set_origin(c, Origin::Composite(vec![Piece { at: (8, expanded.len() as u32), source: g, offset: 0 }]));

    let at = expanded.find("a[5.0]").unwrap() as u32;
    let chain = sm.resolve(Span::new(c, at, at + 6));
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[1].span.source, g);
    assert_eq!(chain[2].via.as_deref(), Some("py2viv"));
    let o = chain[2].span;
    assert_eq!(sm.file(o.source).unwrap().lang, "python");
    assert_eq!(sm.locate(o.source, o.start).unwrap(), (3, 1, "x = a[5]"));

    // the renderer shows the Python line and names the generator
    let d = Diagnostic::error("index 5 out of bounds").with_primary(Span::new(c, at, at + 6), "here");
    let text = PlainRenderer.render(&d, &sm);
    assert_eq!(
        text,
        "error: index 5 out of bounds\n  --> prog.py:3:1\n   3 | x = a[5]\n       ^^^^^^^^ here\n  note: via py2viv: prog.py.viv:3:5\n"
    );
    // a generated line with no origin stays in the generated text
    let at = expanded.find("# header").unwrap() as u32;
    let chain = sm.resolve(Span::new(c, at, at + 1));
    assert_eq!(chain.last().unwrap().span.source, g);
}

#[test]
fn a_rule_instantiation_points_at_its_call_site() {
    let mut sm = SourceMap::new();
    let p = sm.add_lang("lj.py", "e = np.dot(r, w)\n", "python");
    let site = Span::new(p, 4, 16);
    let g = sm.add_generated("<np.dot>", "py_vdot(r, w)\n", "viv", "rule np.dot", site);
    let o = sm.resolve(Span::new(g, 0, 7));
    assert_eq!(o.last().unwrap().span, site);
    assert_eq!(o.last().unwrap().via.as_deref(), Some("rule np.dot"));
}

#[test]
fn rendering_a_written_source_is_unchanged() {
    let mut sm = SourceMap::new();
    let id = sm.add("f.viv", "x = 1\ny = x +\n");
    let d = Diagnostic::error("expected an expression").with_primary(Span::new(id, 10, 13), "here").with_note("a note");
    assert_eq!(
        PlainRenderer.render(&d, &sm),
        "error: expected an expression\n  --> f.viv:2:5\n   2 | y = x +\n           ^^^ here\n  note: a note\n"
    );
}

#[test]
fn a_cycle_ends() {
    let mut sm = SourceMap::new();
    let a = sm.add("a", "x\n");
    let b = sm.add("b", "y\n");
    sm.set_origin(a, Origin::Derived { via: "f".into(), map: LineMap::new(vec![Some(Span::new(b, 0, 1))]) });
    sm.set_origin(b, Origin::Derived { via: "g".into(), map: LineMap::new(vec![Some(Span::new(a, 0, 1))]) });
    assert!(sm.resolve(Span::new(a, 0, 1)).len() <= 32);
}

#[test]
fn offset_of_is_the_inverse_of_locate() {
    let mut sm = SourceMap::new();
    let id = sm.add("f", "ab\ncdef\n\ng");
    for off in 0..11u32 {
        let (line, col, _) = sm.locate(id, off).unwrap();
        assert_eq!(sm.offset_of(id, line, col), Some(off.min(10)));
    }
    assert_eq!(sm.offset_of(id, 2, 99), Some(7));
}

#[test]
fn the_writer_records_each_line_origin() {
    let mut sm = SourceMap::new();
    let p = sm.add("src", "aaaa\nbbbb\n");
    let a = Span::new(p, 0, 4);
    let b = Span::new(p, 5, 9);
    let mut w = Writer::new();
    w.line("# generated", None);
    w.write("x = ", Some(a));
    w.write("f(", Some(b)); // the line already has an origin: a
    w.line(")", None);
    w.write("y = 1\nz = 2\n", Some(b));
    w.line("", None);
    let (text, map) = w.finish();
    assert_eq!(text, "# generated\nx = f()\ny = 1\nz = 2\n\n");
    assert_eq!(map.lines(), &[None, Some(a), Some(b), Some(b), None]);
    assert_eq!(map.len(), text.lines().count());
    // registered, its lines lead back to the source
    let g = sm.add_derived("out", text.as_str(), "viv", "printer", map);
    let at = text.find("z = 2").unwrap() as u32;
    assert_eq!(sm.origin_of(Span::new(g, at, at + 1)), b);
}

#[test]
fn a_writer_without_a_final_newline_keeps_its_last_line() {
    let mut w = Writer::new();
    w.write("a\nb", None);
    let (text, map) = w.finish();
    assert_eq!(text, "a\nb");
    assert_eq!(map.len(), 2);
    let (text, map) = Writer::new().finish();
    assert_eq!((text.as_str(), map.len()), ("", 0));
}

/// locate gives the line and column of every byte, newlines included, the
/// same as counting the newlines before it.
#[test]
fn locate_agrees_with_counting_newlines() {
    let text = "ab\ncd\n\nef\n";
    let mut sm = SourceMap::new();
    let id = sm.add("t.viv", text);
    for off in 0..=text.len() {
        let before = &text[..off];
        let line = before.matches('\n').count() as u32 + 1;
        let start = before.rfind('\n').map_or(0, |n| n + 1);
        let end = text[start..].find('\n').map_or(text.len(), |n| start + n);
        let got = sm.locate(id, off as u32).unwrap();
        assert_eq!(got, (line, (off - start) as u32 + 1, &text[start..end]), "offset {off}");
    }
}
