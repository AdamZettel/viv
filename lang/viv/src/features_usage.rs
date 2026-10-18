// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! How much of the catalog (`crate::features`) a program uses, and where it
//! writes by hand what a feature does (`viv features --usage`).
//!
//! Two scans of one source text:
//!
//! - **Uses**, from the tokens (`crate::lexer`), so a module that only parses
//!   with the modules it uses is still counted: calls, identifiers, the syntax
//!   forms in [`FORMS`], `#! use` lines, and (from the parse, when it parses)
//!   the phrases by group. A detector of the catalog (`crate::features::Detector`)
//!   reads its count from these.
//! - **Hand-written equivalents**, from the parsed program: the patterns of
//!   [`PATTERNS`], the ones a count by reading found (W1, W2, W5, W7 and
//!   W13), found by the shape of the code. Each is a place to look
//!   at, not a verdict: a loop with a recurrence or an early exit is left
//!   alone, but a loop the patterns match may still be written that way for a
//!   reason (an order of summation a test pins, for one).
//!
//! The comprehensions, `for x in` loops and other forms the parser writes as
//! loops over hidden variables (`crate::parser::HIDDEN_PREFIX`) are not
//! reported as hand-written loops.

use std::collections::HashMap;

use knotkit::span::SourceId;

use crate::ast::{BinOp, Def, Expr, Item, Stmt};
use crate::lexer::{Tok, Token};
use crate::parser::HIDDEN_PREFIX;

/// The syntax forms the token scan counts (`form:NAME` detectors).
pub const FORMS: &[&str] = &[
    "comprehension", "map-comprehension", "generator-arg", "if-expr", "negative-index", "slice", "slice-write", "in-op", "key-arg",
    "enumerate-start", "zip-for", "for-in", "for-from", "for-by", "for-reversed", "rows-for", "compound-assign", "index-compound-assign",
    "power-op", "floor-div", "mod-op", "tuple-return", "unpack", "step-block", "one-line-step", "phrase", "user-phrase", "lambda",
    "method-call", "fstring", "print-sep", "record-decl", "field-default", "constructor-def", "record-literal", "field-assign", "record-update", "record-list-type", "enum-decl", "match",
    "optional-type", "int-type", "int-list-type", "str-type", "complex-literal", "matmul", "parallel-for", "test-block", "debug-block",
    "placeholder-def", "rung", "fence", "annotation", "docstring", "typed-param", "return-type", "while-loop", "show", "narrate", "use-line", "import-line",
];

/// The hand-written patterns: id, what the code does, what to write instead.
pub const PATTERNS: &[(&str, &str, &str)] = &[
    ("reduction-loop", "a loop that adds (or multiplies, or takes the max of) one element per pass into a scalar", "sum, prod, amax/amin, dot, or sum of a comprehension; sum(x, ordered=true) adds left to right, in the loop's order (W5)"),
    ("dot-loop", "a loop that adds products of two arrays' elements into a scalar or an element", "dot (left to right, in the loop's order), sum(a * b, ordered=true), @, einsum (W5)"),
    ("fill-loop", "a loop whose one statement writes element i (or [i, j]) of an array from other arrays", "a whole-array expression, broadcasting, outer, or a comprehension (W5)"),
    ("copy-loop", "a loop that copies element i of one array into another", "copy(a), a slice a[lo:hi], or a slice write (W5)"),
    ("append-loop", "a loop whose one statement appends to an array, possibly under an if", "a comprehension [e for x in xs if c] (W5)"),
    ("count-loop", "a loop that adds 1 to a counter under an if", "sum of a mask, count_nonzero, or len of a comprehension with a filter (W5)"),
    ("argmax-loop", "a loop that keeps the best element (or its index) found so far", "amax/amin, argmax/argmin, max(xs, key=f) (W5)"),
    ("swap-sort", "nested loops that swap elements: a sort written by hand", "sort, sorted(xs, key=f), argsort(xs, key=f)"),
    ("round-index", "round, floor or ceil of a number used as an index, a loop bound or a size", "an Int, `//` (W13)"),
    ("while-counter", "a while loop that steps a counter by one: a for loop written by hand", "for i to n, for i from a to b by s"),
    ("field-by-field", "three or more fields of a record computed one by one from the same fields of other records", "record arithmetic (a + b, lr * g), zeros_like(r), for_leaves (W1)"),
    ("slot-index", "an array read through three or more ALL-CAPS slot constants: a record kept as numbers", "a record with named fields, or an array of records (W1, W7)"),
    ("flat-params", "three or more slices of one vector at computed offsets: parameters kept in one flat array", "a record of arrays, value_and_grad over records (W2)"),
    ("getter", "a def whose body is one read of a parameter's field or of a constant index", "the field itself (p.x), a record, or a user phrase"),
];

/// What one source text uses, and the hand-written patterns in it.
#[derive(Default, Debug)]
pub struct Scan {
    pub forms: HashMap<&'static str, usize>,
    pub calls: HashMap<String, usize>,
    pub words: HashMap<String, usize>,
    /// the modules its `#! use` lines name
    pub uses: Vec<String>,
    /// phrases by `crate::phrases` group, and the program's own phrases under ""
    pub phrase_groups: HashMap<String, usize>,
    pub findings: Vec<Finding>,
    /// why the text did not lex or parse, when it did not (the token counts
    /// then cover what lexed; the patterns need the parse)
    pub problem: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Finding {
    pub pattern: &'static str,
    /// 1-based
    pub line: usize,
    pub what: String,
}

impl Scan {
    fn form(&mut self, f: &'static str) {
        debug_assert!(FORMS.contains(&f), "{f}");
        *self.forms.entry(f).or_insert(0) += 1;
    }

    /// How many times a detector of the catalog fires here (a `file:`
    /// detector is the caller's: it is about the files beside this one).
    pub fn count(&self, d: &crate::features::Detector, src: &str) -> usize {
        use crate::features::Detector as D;
        match d {
            D::Call(n) => self.calls.get(n).copied().unwrap_or(0),
            D::Word(n) => self.words.get(n).copied().unwrap_or(0),
            D::Use(m) => self.uses.iter().filter(|u| *u == m || u.starts_with(&format!("{m}/"))).count(),
            D::File(_) => 0,
            D::Line(p) => src.lines().filter(|l| l.trim_start().starts_with(p.as_str())).count(),
            D::Form(f) => match f.strip_prefix("phrase-group:") {
                Some(g) => self.phrase_groups.get(g).copied().unwrap_or(0),
                None => self.forms.get(f.as_str()).copied().unwrap_or(0),
            },
        }
    }
}

/// Scan one source text.
pub fn scan(src: &str) -> Scan {
    let mut s = Scan::default();
    lines(src, &mut s);
    match crate::lexer::lex(src, SourceId(0)) {
        Ok(toks) => tokens(&toks, src, &mut s),
        Err(d) => s.problem = Some(format!("does not lex: {}", d.first().map(|d| d.message.clone()).unwrap_or_default())),
    }
    match crate::parse(src, SourceId(0)) {
        Ok(p) => {
            for u in &p.phrases {
                let g = u.pattern.and_then(|k| crate::phrases::TABLE.get(k)).map(|p| p.group).unwrap_or("");
                *s.phrase_groups.entry(g.to_string()).or_insert(0) += 1;
            }
            // a generated file (reference data, a corpus copy) is not code anyone writes
            let generated = src.lines().take(8).any(|l| l.contains("do not edit") || l.contains("Do not edit"));
            if !generated {
                patterns(&p.items, src, &mut s.findings);
            }
        }
        Err(d) => {
            if s.problem.is_none() {
                s.problem = Some(format!("does not parse alone: {}", d.first().map(|d| d.message.clone()).unwrap_or_default()));
            }
        }
    }
    s
}

// ---- lines -----------------------------------------------------------------------

fn lines(src: &str, s: &mut Scan) {
    let mut fences = 0;
    for l in src.lines() {
        let t = l.trim_start();
        if l.trim() == "###" {
            fences += 1;
            continue;
        }
        if let Some(m) = t.strip_prefix("#! use ") {
            s.uses.push(m.trim().to_string());
            s.form("use-line");
        }
        if fences % 2 == 0 && crate::imports::starts_import(l) {
            s.form("import-line");
        }
        if ["#|", "#:", "#~", "#[", "#]"].iter().any(|p| t.starts_with(p)) {
            s.form("annotation");
        }
    }
    for _ in 0..fences / 2 {
        s.form("fence");
    }
    // the modules import lines name, as every name they could be (only a
    // module's real name matches a `use:` detector)
    for st in crate::imports::scan(src).unwrap_or_default() {
        for item in &st.items {
            let mut paths = vec![item.path().to_vec()];
            if let crate::imports::ImportItem::From { path, names } = item {
                paths.extend(names.iter().map(|(n, _)| {
                    let mut p = path.clone();
                    p.push(n.clone());
                    p
                }));
            }
            for p in paths {
                for c in crate::imports::candidates("", &p) {
                    let c = crate::imports::canonical(&c);
                    if !c.starts_with("./") && !s.uses.contains(&c) {
                        s.uses.push(c);
                    }
                }
            }
        }
    }
}

// ---- tokens ----------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Br {
    Paren,
    Bracket,
    Brace,
}

struct Open {
    br: Br,
    /// a `[` that indexes (after a name or a closing bracket)
    index: bool,
    /// a `for` inside it (a comprehension or a generator)
    has_for: bool,
    has_colon: bool,
    newline: bool,
    /// the name before a `(`
    callee: Option<String>,
    commas: usize,
}

fn ident(t: Option<&Token>) -> Option<&str> {
    match t.map(|t| &t.tok) {
        Some(Tok::Ident(n)) => Some(n.as_str()),
        _ => None,
    }
}

fn is(t: Option<&Token>, k: &Tok) -> bool {
    t.is_some_and(|t| std::mem::discriminant(&t.tok) == std::mem::discriminant(k))
}

/// A token that ends an expression: what a `[` after it indexes, and what an
/// `if` after it makes an if-expression.
fn ends_expr(t: Option<&Token>) -> bool {
    match t.map(|t| &t.tok) {
        Some(Tok::Ident(n)) => !matches!(n.as_str(), "in" | "from" | "by" | "match" | "using" | "with" | "at"),
        Some(Tok::RParen | Tok::RBracket | Tok::Str(_) | Tok::FStr(_) | Tok::Num(..) | Tok::Imag(_) | Tok::True | Tok::False) => true,
        _ => false,
    }
}

fn starts_stmt(t: Option<&Token>) -> bool {
    match t.map(|t| &t.tok) {
        None => true,
        Some(Tok::Newline | Tok::LBrace | Tok::RBrace) => true,
        _ => false,
    }
}

fn type_name(n: &str) -> bool {
    n.chars().next().is_some_and(|c| c.is_uppercase()) && !matches!(n, "Num" | "Arr" | "CArr" | "Bool" | "Str" | "Int" | "Map")
}

fn tokens(toks: &[Token], src: &str, s: &mut Scan) {
    let at = |i: usize| toks.get(i);
    let prev = |i: usize| if i == 0 { None } else { toks.get(i - 1) };
    let mut stack: Vec<Open> = Vec::new();
    // the depth of a `for` whose `in` has not been seen yet
    let mut for_heads: Vec<usize> = Vec::new();
    for i in 0..toks.len() {
        let t = &toks[i].tok;
        let depth = stack.len();
        // a keyword the lexer reserves counts as the word it is spelled with
        if !matches!(t, Tok::Ident(_)) {
            let sp = toks[i].span;
            if let Some(w) = src.get(sp.start as usize..sp.end as usize) {
                if crate::features::KEYWORDS.contains(&w) {
                    *s.words.entry(w.to_string()).or_insert(0) += 1;
                }
            }
        }
        match t {
            Tok::Ident(n) => {
                *s.words.entry(n.clone()).or_insert(0) += 1;
                if is(at(i + 1), &Tok::LParen) && !is(prev(i), &Tok::Def) {
                    *s.calls.entry(n.clone()).or_insert(0) += 1;
                }
                match n.as_str() {
                    "in" => {
                        if for_heads.last() == Some(&depth) {
                            for_heads.pop();
                            match ident(at(i + 1)) {
                                Some("zip") => s.form("zip-for"),
                                Some("rows") => s.form("rows-for"),
                                _ => {}
                            }
                        } else {
                            s.form("in-op");
                        }
                    }
                    "from" if for_heads.last() == Some(&depth) || prev_for(toks, i) => s.form("for-from"),
                    "by" if prev_for(toks, i) => s.form("for-by"),
                    "reversed" if prev_for(toks, i) && !is(at(i + 1), &Tok::LParen) => s.form("for-reversed"),
                    "key" if is(at(i + 1), &Tok::Eq) && stack.last().is_some_and(|o| o.br == Br::Paren) => s.form("key-arg"),
                    "sep" if is(at(i + 1), &Tok::Eq) && stack.last().is_some_and(|o| o.br == Br::Paren) => s.form("print-sep"),
                    "start" if is(at(i + 1), &Tok::Eq) && stack.last().is_some_and(|o| o.callee.as_deref() == Some("enumerate")) => s.form("enumerate-start"),
                    "step" if starts_stmt(prev(i)) && ident(at(i + 1)).is_some() => {
                        let k = if matches!(at(i + 2).map(|t| &t.tok), Some(Tok::Str(_))) { i + 3 } else { i + 2 };
                        // the lexer writes a one-line step `step a: x = 1` as the
                        // braced step, its `{` spanning the `:` (lexer.rs
                        // one_line_steps), so the source text under the brace
                        // tells the two apart
                        match at(k) {
                            Some(t) if t.tok == Tok::LBrace => {
                                if src.get(t.span.start as usize..t.span.end as usize) == Some(":") {
                                    s.form("one-line-step")
                                } else {
                                    s.form("step-block")
                                }
                            }
                            Some(t) if t.tok == Tok::Colon => s.form("one-line-step"),
                            _ => {}
                        }
                    }
                    "phrase" if starts_stmt(prev(i)) && matches!(at(i + 1).map(|t| &t.tok), Some(Tok::Str(_))) => s.form("user-phrase"),
                    "take" | "integrate" | "solve" | "differentiate" | "sum"
                        if (n == "take" || crate::phrases::is_bare_verb(n)) && matches!(at(i + 1).map(|t| &t.tok), Some(Tok::Ident(_) | Tok::Num(..))) && !is(prev(i), &Tok::Dot) =>
                    {
                        s.form("phrase")
                    }
                    "enum" if starts_stmt(prev(i)) && ident(at(i + 1)).is_some() && is(at(i + 2), &Tok::LBrace) => s.form("enum-decl"),
                    "match"
                        if (starts_stmt(prev(i)) || is(prev(i), &Tok::Eq) || is(prev(i), &Tok::Return))
                            && !matches!(at(i + 1).map(|t| &t.tok), Some(Tok::Eq | Tok::LParen | Tok::Dot | Tok::LBracket | Tok::Comma | Tok::Newline | Tok::PlusEq | Tok::MinusEq)) =>
                    {
                        s.form("match")
                    }
                    "debug" if starts_stmt(prev(i)) && (is(at(i + 1), &Tok::LBrace) || (matches!(at(i + 1).map(|t| &t.tok), Some(Tok::Ident(_) | Tok::Star)) && is(at(i + 2), &Tok::LBrace))) => {
                        s.form("debug-block")
                    }
                    "placeholder" if is(at(i + 1), &Tok::Def) => s.form("placeholder-def"),
                    "Int" if matches!(prev(i).map(|t| &t.tok), Some(Tok::Colon | Tok::Arrow)) => s.form("int-type"),
                    "Str" if matches!(prev(i).map(|t| &t.tok), Some(Tok::Colon | Tok::Arrow)) => s.form("str-type"),
                    _ => {}
                }
                // a record literal: `Name { field: ...`
                if type_name(n) && is(at(i + 1), &Tok::LBrace) && ident(at(i + 2)).is_some() && is(at(i + 3), &Tok::Colon) && !is(prev(i), &Tok::Record) && ident(prev(i)) != Some("enum") {
                    s.form("record-literal");
                }
                // `p.x = ...` at the start of a statement
                if starts_stmt(prev(i)) && is(at(i + 1), &Tok::Dot) && ident(at(i + 2)).is_some() {
                    let mut k = i + 3;
                    let mut d = 0i32;
                    while let Some(tk) = at(k) {
                        match tk.tok {
                            Tok::LParen | Tok::LBracket => d += 1,
                            Tok::RParen | Tok::RBracket => d -= 1,
                            Tok::Dot | Tok::Ident(_) if d == 0 => {}
                            _ if d > 0 => {}
                            _ => break,
                        }
                        k += 1;
                    }
                    if matches!(at(k).map(|t| &t.tok), Some(Tok::Eq | Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq | Tok::SlashSlashEq)) {
                        s.form("field-assign");
                    }
                }
                // `a, b = ...` at the start of a statement
                if starts_stmt(prev(i)) {
                    let mut k = i + 1;
                    let mut d = 0i32;
                    let mut comma = false;
                    while let Some(tk) = at(k) {
                        match tk.tok {
                            Tok::LParen | Tok::LBracket => d += 1,
                            Tok::RParen | Tok::RBracket => d -= 1,
                            Tok::Comma if d == 0 => comma = true,
                            Tok::Eq if d == 0 => {
                                if comma {
                                    s.form("unpack");
                                }
                                break;
                            }
                            Tok::Newline | Tok::LBrace | Tok::Eof => break,
                            _ => {}
                        }
                        k += 1;
                    }
                }
            }
            Tok::LParen | Tok::LBracket | Tok::LBrace => {
                let br = match t {
                    Tok::LParen => Br::Paren,
                    Tok::LBracket => Br::Bracket,
                    _ => Br::Brace,
                };
                // a type `[Name]` after `:` or `->`
                if br == Br::Bracket && matches!(prev(i).map(|t| &t.tok), Some(Tok::Colon | Tok::Arrow)) && is(at(i + 2), &Tok::RBracket) {
                    match ident(at(i + 1)) {
                        Some("Int") => s.form("int-list-type"),
                        Some(n) if type_name(n) => s.form("record-list-type"),
                        _ => {}
                    }
                }
                let callee = if br == Br::Paren { ident(prev(i)).map(str::to_string) } else { None };
                stack.push(Open { br, index: br == Br::Bracket && ends_expr(prev(i)), has_for: false, has_colon: false, newline: false, callee, commas: 0 });
            }
            Tok::RParen | Tok::RBracket | Tok::RBrace => {
                if let Some(o) = stack.pop() {
                    if o.br == Br::Bracket && o.index && o.has_colon {
                        s.form("slice");
                        if matches!(at(i + 1).map(|t| &t.tok), Some(Tok::Eq | Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq | Tok::SlashSlashEq)) {
                            s.form("slice-write");
                        }
                    }
                    if o.br == Br::Bracket && o.index && matches!(at(i + 1).map(|t| &t.tok), Some(Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq | Tok::SlashSlashEq)) {
                        s.form("index-compound-assign");
                    }
                    for_heads.retain(|d| *d <= stack.len());
                }
            }
            Tok::For => {
                match stack.last_mut() {
                    Some(o) if !o.has_for && o.br == Br::Bracket && !o.index => {
                        o.has_for = true;
                        s.form("comprehension");
                    }
                    Some(o) if !o.has_for && o.br == Br::Brace && !o.newline => {
                        o.has_for = true;
                        s.form("map-comprehension");
                    }
                    Some(o) if !o.has_for && o.br == Br::Paren => {
                        o.has_for = true;
                        s.form("generator-arg");
                    }
                    _ => {}
                }
                if ident(at(i + 2)) == Some("in") || (is(at(i + 2), &Tok::Comma) && ident(at(i + 4)) == Some("in")) {
                    s.form("for-in");
                }
                for_heads.push(depth);
            }
            Tok::To => {
                if for_heads.last() == Some(&depth) {
                    for_heads.pop();
                }
            }
            Tok::Newline => {
                if let Some(o) = stack.last_mut() {
                    o.newline = true;
                }
            }
            Tok::Colon => {
                if let Some(o) = stack.last_mut() {
                    if o.br == Br::Bracket && o.index {
                        o.has_colon = true;
                    }
                }
                if stack.last().is_some_and(|o| o.br == Br::Paren) && matches!(prev(i).map(|t| &t.tok), Some(Tok::Ident(_))) && in_def_params(toks, i) {
                    s.form("typed-param");
                }
            }
            Tok::Comma => {
                if let Some(o) = stack.last_mut() {
                    o.commas += 1;
                    if o.br == Br::Paren && o.callee.as_deref() == Some("enumerate") && o.commas == 1 && !is(at(i + 1), &Tok::RParen) && ident(at(i + 1)) != Some("start") {
                        s.form("enumerate-start");
                    }
                }
            }
            Tok::Minus => {
                if let Some(o) = stack.last() {
                    if o.br == Br::Bracket && o.index && matches!(prev(i).map(|t| &t.tok), Some(Tok::LBracket | Tok::Comma | Tok::Colon)) && matches!(at(i + 1).map(|t| &t.tok), Some(Tok::Num(..))) {
                        s.form("negative-index");
                    }
                }
            }
            Tok::If => {
                let filter = stack.last().is_some_and(|o| o.has_for);
                if !filter && !starts_stmt(prev(i)) && !is(prev(i), &Tok::Else) && !is(prev(i), &Tok::Colon) && ends_expr(prev(i)) {
                    s.form("if-expr");
                }
            }
            Tok::Return => {
                // `return a, b`: a comma at this depth before the line ends
                let mut k = i + 1;
                let mut d = 0i32;
                while let Some(tk) = at(k) {
                    match tk.tok {
                        Tok::LParen | Tok::LBracket | Tok::LBrace => d += 1,
                        Tok::RParen | Tok::RBracket | Tok::RBrace if d > 0 => d -= 1,
                        Tok::Comma if d == 0 => {
                            s.form("tuple-return");
                            break;
                        }
                        Tok::Newline | Tok::RBrace | Tok::Eof => break,
                        _ => {}
                    }
                    k += 1;
                }
            }
            Tok::Def => {
                // `def Head(c) -> Head`: a constructor def
                if ident(at(i + 1)).is_some_and(|n| n.starts_with(|c: char| c.is_ascii_uppercase())) {
                    s.form("constructor-def");
                }
                // `def f(...) -> T`
                let mut k = i + 1;
                let mut d = 0i32;
                while let Some(tk) = at(k) {
                    match tk.tok {
                        Tok::LParen => d += 1,
                        Tok::RParen => {
                            d -= 1;
                            if d == 0 {
                                if is(at(k + 1), &Tok::Arrow) {
                                    s.form("return-type");
                                }
                                // a docstring: the first statement of the body is a string
                                let mut b = k + 1;
                                while at(b).is_some_and(|t| !matches!(t.tok, Tok::LBrace | Tok::Newline)) {
                                    b += 1;
                                }
                                if is(at(b), &Tok::LBrace) {
                                    let mut c = b + 1;
                                    while is(at(c), &Tok::Newline) {
                                        c += 1;
                                    }
                                    if matches!(at(c).map(|t| &t.tok), Some(Tok::Str(_))) && matches!(at(c + 1).map(|t| &t.tok), Some(Tok::Newline)) {
                                        s.form("docstring");
                                    }
                                }
                                break;
                            }
                        }
                        Tok::LBrace | Tok::Eof => break,
                        _ => {}
                    }
                    k += 1;
                }
            }
            Tok::Dot if is(at(i + 1), &Tok::Dot) && stack.last().is_some_and(|o| o.br == Br::Brace) => s.form("record-update"),
            Tok::Dot => {
                if ends_expr(prev(i)) && ident(at(i + 1)).is_some() && is(at(i + 2), &Tok::LParen) {
                    s.form("method-call");
                }
            }
            Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq | Tok::SlashSlashEq => s.form("compound-assign"),
            Tok::Caret => s.form("power-op"),
            Tok::SlashSlash => s.form("floor-div"),
            Tok::Percent => s.form("mod-op"),
            Tok::At => s.form("matmul"),
            Tok::Fn => s.form("lambda"),
            Tok::FStr(_) => s.form("fstring"),
            Tok::Record => {
                s.form("record-decl");
                // `width: Int = 16`: a field default, an `=` inside the braces
                let mut k = i + 1;
                let mut d = 0i32;
                while let Some(tk) = at(k) {
                    match tk.tok {
                        Tok::LBrace | Tok::LParen | Tok::LBracket => d += 1,
                        Tok::RBrace | Tok::RParen | Tok::RBracket => {
                            d -= 1;
                            if d <= 0 {
                                break;
                            }
                        }
                        Tok::Eq if d == 1 => s.form("field-default"),
                        Tok::Eof => break,
                        _ => {}
                    }
                    k += 1;
                }
            }
            Tok::Question => s.form("optional-type"),
            Tok::Imag(_) => s.form("complex-literal"),
            Tok::Parallel => s.form("parallel-for"),
            Tok::Test => s.form("test-block"),
            Tok::Rung => s.form("rung"),
            Tok::While => s.form("while-loop"),
            Tok::Show => s.form("show"),
            Tok::Narrate => s.form("narrate"),
            _ => {}
        }
    }
}

/// Whether the token before `i` on its line, back to a `for`, is that `for`'s
/// head (the `from`, `by` and `reversed` of a for statement).
fn prev_for(toks: &[Token], i: usize) -> bool {
    let mut k = i;
    while k > 0 {
        k -= 1;
        match toks[k].tok {
            Tok::For => return true,
            Tok::Newline | Tok::LBrace | Tok::RBrace | Tok::Eq => return false,
            _ => {}
        }
    }
    false
}

/// Whether the `:` at `i` is inside the parameter list of a `def`.
fn in_def_params(toks: &[Token], i: usize) -> bool {
    let mut d = 0i32;
    let mut k = i;
    while k > 0 {
        k -= 1;
        match toks[k].tok {
            Tok::RParen | Tok::RBracket => d += 1,
            Tok::LParen | Tok::LBracket if d > 0 => d -= 1,
            Tok::LParen => return k >= 2 && matches!(toks[k - 2].tok, Tok::Def),
            Tok::Newline | Tok::LBrace | Tok::RBrace => return false,
            _ => {}
        }
    }
    false
}

// ---- patterns ----------------------------------------------------------------------

fn line_of(src: &str, at: u32) -> usize {
    src[..(at as usize).min(src.len())].matches('\n').count() + 1
}

fn hidden(name: &str) -> bool {
    name.starts_with(HIDDEN_PREFIX)
}

/// The statements of a body that are code, without step markers and lens statements.
fn code(body: &[Stmt]) -> Vec<&Stmt> {
    body.iter().filter(|s| !matches!(s, Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. })).collect()
}

/// Whether `e` reads a variable named `v`.
fn reads(e: &Expr, v: &str) -> bool {
    match e {
        Expr::Var { name, .. } => name == v,
        Expr::Index { base, indices, .. } => reads(base, v) || indices.iter().any(|i| reads(i, v)),
        Expr::Lambda { body, .. } => reads(body, v),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => reads(operand, v),
        Expr::Binary { lhs, rhs, .. } => reads(lhs, v) || reads(rhs, v),
        Expr::Call { args, .. } | Expr::ArrLit { elems: args, .. } => args.iter().any(|a| reads(a, v)),
        Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| reads(x, v)),
        Expr::Field { base, .. } => reads(base, v),
        _ => false,
    }
}

/// Whether `e` reads an element of some array indexed by the loop variable
/// `i` (or is the element variable `elem` of a `for x in` loop).
fn element_read(e: &Expr, i: &str, elem: Option<&str>) -> bool {
    match e {
        Expr::Var { name, .. } => elem == Some(name.as_str()),
        Expr::Index { base, indices, .. } => indices.iter().any(|x| reads(x, i)) || element_read(base, i, elem),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => element_read(operand, i, elem),
        Expr::Binary { lhs, rhs, .. } => element_read(lhs, i, elem) || element_read(rhs, i, elem),
        Expr::Call { args, .. } => args.iter().any(|a| element_read(a, i, elem)),
        Expr::Field { base, .. } => element_read(base, i, elem),
        _ => false,
    }
}

/// `a[..i..] * b[..i..]`: a product of two element reads.
fn product_of_reads(e: &Expr, i: &str) -> bool {
    match e {
        Expr::Binary { op: BinOp::Mul, lhs, rhs, .. } => {
            let idx = |x: &Expr| matches!(x, Expr::Index { indices, .. } if indices.iter().any(|k| reads(k, i)));
            (idx(lhs) && idx(rhs)) || product_of_reads(lhs, i) || product_of_reads(rhs, i)
        }
        _ => false,
    }
}

fn is_var(e: &Expr, v: &str) -> bool {
    matches!(e, Expr::Var { name, .. } if name == v)
}

fn finding(out: &mut Vec<Finding>, pattern: &'static str, src: &str, at: u32, what: String) {
    out.push(Finding { pattern, line: line_of(src, at), what });
}

struct Loop<'a> {
    var: &'a str,
    /// the element variable of a `for x in` loop the parser rewrote
    elem: Option<&'a str>,
    body: Vec<&'a Stmt>,
    span_start: u32,
}

/// A `for` loop as written: a `for x in xs` loop is a loop over a hidden
/// counter whose first statement is `x = xs[k]`.
fn as_loop(s: &Stmt) -> Option<Loop<'_>> {
    let Stmt::For { var, body, span, .. } = s else { return None };
    let mut b = code(body);
    let mut elem = None;
    if hidden(var) {
        match b.first() {
            Some(Stmt::Assign { name, value: Expr::Index { indices, .. }, .. }) if !hidden(name) && indices.iter().any(|x| is_var(x, var)) => {
                elem = Some(name.as_str());
                b.remove(0);
            }
            _ => return None,
        }
    }
    Some(Loop { var, elem, body: b, span_start: span.start })
}

/// The patterns of one loop whose body is one statement.
fn loop_patterns(l: &Loop, nested_in: usize, src: &str, out: &mut Vec<Finding>) {
    let [only] = l.body.as_slice() else { return };
    let i = l.var;
    match only {
        // s = s + e[i]
        Stmt::Assign { name, value, .. } if !hidden(name) => {
            if let Expr::Binary { op, lhs, rhs, .. } = value {
                if matches!(op, BinOp::Add | BinOp::Mul | BinOp::Sub) && is_var(lhs, name) && element_read(rhs, i, l.elem) && !reads(rhs, name) {
                    let (p, w) = if product_of_reads(rhs, i) {
                        ("dot-loop", format!("`{name}` accumulates products of elements"))
                    } else if matches!(**rhs, Expr::Num { .. }) {
                        return;
                    } else {
                        ("reduction-loop", format!("`{name}` accumulates one element per pass"))
                    };
                    finding(out, p, src, l.span_start, w);
                    return;
                }
            }
            if let Expr::Call { callee, args, .. } = value {
                if (callee == "max" || callee == "min") && args.len() == 2 && args.iter().any(|a| is_var(a, name)) && args.iter().any(|a| element_read(a, i, l.elem) && !reads(a, name)) {
                    finding(out, "argmax-loop", src, l.span_start, format!("`{name}` = {callee}({name}, element)"));
                    return;
                }
                if callee == "append" && args.len() == 2 && is_var(&args[0], name) {
                    finding(out, "append-loop", src, l.span_start, format!("`{name}` grows by one element per pass"));
                }
            }
        }
        // b[i] = f(a[i]); b[i] = a[i]
        Stmt::SetIndex { base, indices, value, .. } => {
            let Expr::Var { name: b, .. } = &**base else { return };
            if hidden(b) || indices.len() != 1 || !is_var(&indices[0], i) || reads(value, b) {
                return;
            }
            if matches!(value, Expr::Index { indices: ix, .. } if ix.len() == 1 && is_var(&ix[0], i)) {
                finding(out, "copy-loop", src, l.span_start, format!("`{b}[{i}]` copied element by element"));
            } else if element_read(value, i, l.elem) && nested_in == 0 {
                if product_of_reads(value, i) && matches!(value, Expr::Binary { op: BinOp::Add, .. }) {
                    finding(out, "dot-loop", src, l.span_start, format!("`{b}[{i}]` accumulates products"));
                } else {
                    finding(out, "fill-loop", src, l.span_start, format!("`{b}[{i}]` written from other arrays' element {i}"));
                }
            }
        }
        Stmt::If { cond, then_body, else_body, .. } if else_body.is_empty() => {
            let then = code(then_body);
            match then.as_slice() {
                [Stmt::Assign { name, value: Expr::Call { callee, args, .. }, .. }] if callee == "append" && args.len() == 2 && is_var(&args[0], name) && !hidden(name) => {
                    finding(out, "append-loop", src, l.span_start, format!("`{name}` grows under an if"));
                }
                [Stmt::Assign { name, value: Expr::Binary { op: BinOp::Add, lhs, rhs, .. }, .. }] if is_var(lhs, name) && matches!(**rhs, Expr::Num { value, .. } if value == 1.0) && !hidden(name) => {
                    finding(out, "count-loop", src, l.span_start, format!("`{name}` counts the passes where a test holds"));
                }
                assigns if !assigns.is_empty() && assigns.iter().all(|s| matches!(s, Stmt::Assign { .. })) => {
                    if let Expr::Binary { op: BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le, lhs, rhs, .. } = cond {
                        let names: Vec<&str> = assigns.iter().filter_map(|s| if let Stmt::Assign { name, .. } = s { Some(name.as_str()) } else { None }).collect();
                        let other_side_is_best = |a: &Expr, b: &Expr| element_read(a, i, l.elem) && matches!(b, Expr::Var { name, .. } if names.contains(&name.as_str()));
                        if other_side_is_best(lhs, rhs) || other_side_is_best(rhs, lhs) {
                            finding(out, "argmax-loop", src, l.span_start, format!("keeps the best so far in `{}`", names.join("`, `")));
                        }
                    }
                }
                _ => {}
            }
        }
        // for i { for j { B[i, j] = ... } }
        Stmt::For { .. } => {
            if let Some(inner) = as_loop(only) {
                if let [Stmt::SetIndex { base, indices, value, .. }] = inner.body.as_slice() {
                    if let Expr::Var { name: b, .. } = &**base {
                        let ij = indices.len() == 2 && is_var(&indices[0], i) && is_var(&indices[1], inner.var);
                        if ij && !reads(value, b) && !hidden(b) && nested_in == 0 {
                            finding(out, "fill-loop", src, l.span_start, format!("`{b}[{i}, {}]` written element by element", inner.var));
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// A swap of two elements of one array through a temporary.
fn swaps(body: &[&Stmt]) -> bool {
    body.windows(3).any(|w| match (w[0], w[1], w[2]) {
        (Stmt::Assign { name: t, value: Expr::Index { base: b0, .. }, .. }, Stmt::SetIndex { base: b1, value: v1, .. }, Stmt::SetIndex { base: b2, value: v2, .. }) => {
            let same = |x: &Expr, y: &Expr| matches!((x, y), (Expr::Var { name: a, .. }, Expr::Var { name: b, .. }) if a == b);
            same(b0, b1) && same(b1, b2) && matches!(v1, Expr::Index { base, .. } if same(base, b0)) && is_var(v2, t)
        }
        _ => false,
    })
}

fn stmts(body: &[Stmt], depth: usize, src: &str, out: &mut Vec<Finding>) {
    let b = code(body);
    // a swap, under an if that compares elements of the array it swaps, inside two loops: a sort written by hand
    if depth >= 2 {
        for s in &b {
            if let Stmt::If { cond, then_body, span, .. } = s {
                let then = code(then_body);
                let swapped = then.iter().find_map(|t| if let Stmt::SetIndex { base, .. } = t { if let Expr::Var { name, .. } = &**base { Some(name.clone()) } else { None } } else { None });
                if let Some(arr) = swapped {
                    if swaps(&then) && reads(cond, &arr) {
                        finding(out, "swap-sort", src, span.start, format!("`{arr}` swapped where a comparison of its elements says so, inside nested loops"));
                    }
                }
            }
        }
    }
    // a while loop with a counter: `while i < n { ...; i = i + 1 }`
    for s in &b {
        if let Stmt::While { cond, body: wb, span } = s {
            if let Expr::Binary { op: BinOp::Lt | BinOp::Le, lhs, .. } = cond {
                if let Expr::Var { name: i, .. } = &**lhs {
                    let wbc = code(wb);
                    let last = wbc.last();
                    let steps = matches!(last, Some(Stmt::Assign { name, value: Expr::Binary { op: BinOp::Add, lhs, rhs, .. }, .. })
                        if name == i && is_var(lhs, i) && matches!(**rhs, Expr::Num { value, .. } if value == 1.0));
                    let other_writes = wbc[..wbc.len().saturating_sub(1)].iter().any(|s| matches!(s, Stmt::Assign { name, .. } if name == i));
                    let exits = has_exit(wb);
                    if steps && !other_writes && !exits && !hidden(i) {
                        finding(out, "while-counter", src, span.start, format!("`{i}` steps by one to a bound"));
                    }
                }
            }
        }
    }
    // three or more fields of one record set from the same fields of others
    let mut run: Vec<(&str, u32)> = Vec::new();
    let flush = |run: &mut Vec<(&str, u32)>, out: &mut Vec<Finding>| {
        if run.len() >= 3 {
            finding(out, "field-by-field", src, run[0].1, format!("{} fields of `{}` set one by one", run.len(), run[0].0));
        }
        run.clear();
    };
    for s in &b {
        match s {
            Stmt::SetField { base, field, value, span, .. } => match &**base {
                Expr::Var { name, .. } if reads_field(value, field) => {
                    if run.first().is_some_and(|(n, _)| *n != name.as_str()) {
                        flush(&mut run, out);
                    }
                    run.push((name.as_str(), span.start));
                }
                _ => flush(&mut run, out),
            },
            _ => flush(&mut run, out),
        }
    }
    flush(&mut run, out);
    for s in &b {
        if let Some(l) = as_loop(s) {
            loop_patterns(&l, depth, src, out);
        }
        match s {
            Stmt::For { body, .. } | Stmt::While { body, .. } => stmts(body, depth + 1, src, out),
            Stmt::If { then_body, else_body, .. } => {
                stmts(then_body, depth, src, out);
                stmts(else_body, depth, src, out);
            }
            Stmt::Test { body, .. } => stmts(body, depth, src, out),
            _ => {}
        }
        each_expr(s, &mut |e| exprs(e, src, out));
    }
}

fn has_exit(body: &[Stmt]) -> bool {
    body.iter().any(|s| match s {
        Stmt::Break { .. } | Stmt::Return { .. } => true,
        Stmt::If { then_body, else_body, .. } => has_exit(then_body) || has_exit(else_body),
        _ => false,
    })
}

fn reads_field(e: &Expr, f: &str) -> bool {
    match e {
        Expr::Field { base, field, .. } => field == f || reads_field(base, f),
        Expr::Index { base, indices, .. } => reads_field(base, f) || indices.iter().any(|i| reads_field(i, f)),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => reads_field(operand, f),
        Expr::Binary { lhs, rhs, .. } => reads_field(lhs, f) || reads_field(rhs, f),
        Expr::Call { args, .. } => args.iter().any(|a| reads_field(a, f)),
        _ => false,
    }
}

/// Every expression a statement holds directly.
fn each_expr(s: &Stmt, f: &mut dyn FnMut(&Expr)) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => f(value),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => f(expr),
        Stmt::SetIndex { base, indices, value, .. } => {
            f(base);
            indices.iter().for_each(&mut *f);
            f(value);
        }
        Stmt::SetField { base, value, .. } => {
            f(base);
            f(value);
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } => f(cond),
        Stmt::For { bound, .. } => f(bound),
        _ => {}
    }
}

fn is_rounding(e: &Expr) -> bool {
    matches!(e, Expr::Call { callee, args, .. } if matches!(callee.as_str(), "round" | "floor" | "ceil") && args.len() == 1)
}

/// The patterns inside one expression: rounding used as an index or a size,
/// a record literal built field by field from other records.
fn exprs(e: &Expr, src: &str, out: &mut Vec<Finding>) {
    match e {
        Expr::Index { base, indices, span } => {
            if indices.iter().any(is_rounding) {
                finding(out, "round-index", src, span.start, "a rounded number as an index".to_string());
            }
            exprs(base, src, out);
            indices.iter().for_each(|i| exprs(i, src, out));
        }
        Expr::Call { callee, args, span } => {
            if matches!(callee.as_str(), "zeros" | "ones" | "reshape" | "arange" | "eye" | "int_zeros") && args.iter().any(is_rounding) {
                finding(out, "round-index", src, span.start, format!("a rounded number as a size of {callee}"));
            }
            args.iter().for_each(|a| exprs(a, src, out));
        }
        Expr::RecordLit { fields, span, type_name, .. } => {
            let same = fields.iter().filter(|(f, _, v)| reads_field(v, f) && !matches!(v, Expr::Field { .. })).count();
            if fields.len() >= 3 && same == fields.len() {
                finding(out, "field-by-field", src, span.start, format!("every field of a `{type_name}` computed from the same field of others"));
            }
            fields.iter().for_each(|(_, _, v)| exprs(v, src, out));
        }
        Expr::Binary { lhs, rhs, .. } => {
            exprs(lhs, src, out);
            exprs(rhs, src, out);
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => exprs(operand, src, out),
        Expr::ArrLit { elems, .. } => elems.iter().for_each(|x| exprs(x, src, out)),
        Expr::Field { base, .. } => exprs(base, src, out),
        Expr::Lambda { body, .. } => exprs(body, src, out),
        _ => {}
    }
}

fn caps(n: &str) -> bool {
    n.len() >= 2 && n.chars().any(|c| c.is_ascii_uppercase()) && n.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Per def: an array read through three or more slot constants, a vector
/// sliced at three or more computed offsets, a getter.
fn def_patterns(d: &Def, src: &str, out: &mut Vec<Finding>) {
    let mut slots: HashMap<String, (std::collections::HashSet<String>, u32)> = HashMap::new();
    let mut slices: HashMap<String, (usize, u32)> = HashMap::new();
    fn walk(e: &Expr, slots: &mut HashMap<String, (std::collections::HashSet<String>, u32)>, slices: &mut HashMap<String, (usize, u32)>) {
        match e {
            Expr::Index { base, indices, span } => {
                if let Expr::Var { name, .. } = &**base {
                    for i in indices {
                        if let Expr::Var { name: k, .. } = i {
                            if caps(k) {
                                let ent = slots.entry(name.clone()).or_insert_with(|| (Default::default(), span.start));
                                ent.0.insert(k.clone());
                            }
                        }
                    }
                }
                walk(base, slots, slices);
                indices.iter().for_each(|i| walk(i, slots, slices));
            }
            Expr::Call { callee, args, span } => {
                if callee == crate::interp::slices::GET {
                    if let (Some(Expr::Var { name, .. }), Some(Expr::ArrLit { elems, .. })) = (args.first(), args.get(1)) {
                        if elems.iter().any(|x| matches!(x, Expr::Binary { .. } | Expr::Var { .. })) {
                            let ent = slices.entry(name.clone()).or_insert((0, span.start));
                            ent.0 += 1;
                        }
                    }
                }
                args.iter().for_each(|a| walk(a, slots, slices));
            }
            Expr::Binary { lhs, rhs, .. } => {
                walk(lhs, slots, slices);
                walk(rhs, slots, slices);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk(operand, slots, slices),
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|x| walk(x, slots, slices)),
            Expr::Field { base, .. } => walk(base, slots, slices),
            Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, v)| walk(v, slots, slices)),
            Expr::Lambda { body, .. } => walk(body, slots, slices),
            _ => {}
        }
    }
    fn all(body: &[Stmt], f: &mut dyn FnMut(&Expr)) {
        for s in body {
            each_expr(s, f);
            match s {
                Stmt::For { body, .. } | Stmt::While { body, .. } | Stmt::Test { body, .. } => all(body, f),
                Stmt::If { then_body, else_body, .. } => {
                    all(then_body, f);
                    all(else_body, f);
                }
                _ => {}
            }
        }
    }
    all(&d.body, &mut |e| walk(e, &mut slots, &mut slices));
    let mut slot_list: Vec<_> = slots.into_iter().filter(|(_, (ks, _))| ks.len() >= 3).collect();
    slot_list.sort_by_key(|(_, (_, at))| *at);
    for (name, (ks, at)) in slot_list {
        finding(out, "slot-index", src, at, format!("`{name}` read through {} slot constants in `{}`", ks.len(), d.name));
    }
    let mut slice_list: Vec<_> = slices.into_iter().filter(|(_, (n, _))| *n >= 3).collect();
    slice_list.sort_by_key(|(_, (_, at))| *at);
    for (name, (n, at)) in slice_list {
        finding(out, "flat-params", src, at, format!("`{name}` sliced at {n} computed offsets in `{}`", d.name));
    }
    // a getter: `def f(p) { return p.x }` or `return a[K]`
    let body: Vec<&Stmt> = code(&d.body).into_iter().filter(|s| !matches!(s, Stmt::ExprStmt { expr: Expr::Str { .. }, .. })).collect();
    if let [Stmt::Return { value, .. }] = body.as_slice() {
        let param = |e: &Expr| matches!(e, Expr::Var { name, .. } if d.params.iter().any(|p| &p.name == name));
        let getter = match value {
            Expr::Field { base, .. } => param(base),
            Expr::Index { base, indices, .. } => param(base) && indices.iter().all(|i| matches!(i, Expr::Num { .. }) || matches!(i, Expr::Var { name, .. } if caps(name))),
            _ => false,
        };
        if getter {
            finding(out, "getter", src, d.name_span.start, format!("`{}` returns one field or slot of its argument", d.name));
        }
    }
}

fn patterns(items: &[Item], src: &str, out: &mut Vec<Finding>) {
    let mut top: Vec<Stmt> = Vec::new();
    for it in items {
        match it {
            Item::Def(d) => {
                // a def the phrases' lowering added from lang/viv/std is not this file's code
                if hidden(&d.name) || (crate::phrases::std_def_names().contains(&d.name) && !src.contains(&format!("def {}(", d.name))) {
                    continue;
                }
                stmts(&d.body, 0, src, out);
                def_patterns(d, src, out);
            }
            Item::Stmt(s) => top.push(s.clone()),
            Item::Rung(r) => patterns(&r.items, src, out),
            _ => {}
        }
    }
    stmts(&top, 0, src, out);
    out.sort_by_key(|f| f.line);
    out.dedup_by(|a, b| a.line == b.line && a.pattern == b.pattern);
}
