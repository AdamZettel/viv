// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Phrases: math written as English, `take the integral of x^2 from 1 to 2`,
//! read by the parser as an ordinary call.
//!
//! A phrase starts with the soft keyword `take` followed by a word or a
//! number, or with one of the verbs in [`BARE_VERBS`] followed by a word or a
//! number (`integrate x^2 from 1 to 2`, `solve x^2 = 2 between 0 and 2`). In
//! no other place can a name be followed directly by a name or a number, so
//! `take` and the verbs stay usable as variable and def names.
//!
//! A phrase runs to the end of its line, or to a `,`, a closing bracket, a
//! `{`, a `:`, an `if` or an `else` outside brackets. Its words are matched
//! against [`TABLE`]. A pattern is literal words and holes: `{v}` is an
//! expression, `{f:fn}` an expression in one free variable (it becomes a
//! `fn(x) -> ...`, and may be an equation `lhs = rhs`), `{x:var}` a single
//! name. A literal written `a|b` matches either word. The parser tries every
//! way to split the phrase among the holes of every pattern, keeps the splits
//! whose holes parse as complete expressions, and needs exactly one: none is
//! an error naming the closest patterns, and two or more is an error listing
//! the readings. It never picks one reading over another.
//!
//! The matched pattern's `lowers` is Viv text with the holes in it. The parser
//! parses that text and puts the holes' expressions in, so the interpreter,
//! the C backend and the wasm backend see only calls they already run. A hole
//! the lowering reads more than once is computed once, into a hidden variable
//! (`crate::surface`). A lowering that calls a def of `lang/viv/std` (the
//! quadratures and root finders of `contracts.viv`, `r_newton` of
//! `phrases.viv`) gets that def added to the program by [`provide`], in
//! `crate::parse`; a program with no phrase is not changed.

use std::collections::{HashMap, HashSet};

use knotkit::diag::Diagnostic;
use knotkit::span::{SourceId, Span};

use crate::ast::*;
use crate::lexer::{Tok, Token};

/// One phrase form.
pub struct Pattern {
    /// The heading the reference page lists the pattern under.
    pub group: &'static str,
    /// Literal words and holes (see the module comment).
    pub words: &'static str,
    /// The Viv expression the phrase is, with the holes in it. `{f'}` is the
    /// derivative of the `fn` hole `f`: `fn(x) -> deriv(fn(u) -> <f>, x)`.
    /// `{f#text}`, `{f#var}`, `{f#names}` and `{f#vals}` hand the hole to the
    /// computer algebra system (lib/cas): its expression as the CAS's text, its
    /// variable, the other names it reads (joined by commas) and their values
    /// (crate::cas_check::hole_text).
    pub lowers: &'static str,
    /// Ported from Knot's table (knot/src/phrases.hpp); the rest are Viv's.
    pub knot: bool,
}

const fn k(group: &'static str, words: &'static str, lowers: &'static str) -> Pattern {
    Pattern { group, words, lowers, knot: true }
}
const fn v(group: &'static str, words: &'static str, lowers: &'static str) -> Pattern {
    Pattern { group, words, lowers, knot: false }
}

const AGG: &str = "Aggregates";
const MAT: &str = "Matrices";
const VEC: &str = "Vectors";
const SIZE: &str = "Sizes and elements";
const MATH: &str = "Elementary functions";
const LIN: &str = "Geometry and linear algebra";
const MAKE: &str = "Construction";
const CALC: &str = "Integrals, roots and derivatives";
const LEARN: &str = "Probability and learning";

/// The verbs that open a phrase without `take`.
pub const BARE_VERBS: &[&str] = &["integrate", "solve", "differentiate", "sum"];

/// Every phrase form. The order is the order of the reference page; matching
/// does not depend on it.
pub const TABLE: &[Pattern] = &[
    // ---- aggregates (Knot) -------------------------------------------------
    k(AGG, "the sum of {v}", "sum({v})"),
    k(AGG, "the product of {v}", "prod({v})"),
    k(AGG, "the mean of {v}", "mean({v})"),
    k(AGG, "the average of {v}", "mean({v})"),
    k(AGG, "the median of {v}", "0.5 * (take({v}, argsort({v}), axis=0)[(len({v}) - 1) // 2] + take({v}, argsort({v}), axis=0)[len({v}) // 2])"),
    k(AGG, "the max of {v}", "amax({v})"),
    k(AGG, "the maximum of {v}", "amax({v})"),
    k(AGG, "the min of {v}", "amin({v})"),
    k(AGG, "the minimum of {v}", "amin({v})"),
    k(AGG, "the variance of {v}", "var({v})"),
    k(AGG, "the standard deviation of {v}", "std({v})"),
    v(AGG, "the cumulative sum of {v}", "cumsum({v})"),
    v(AGG, "the cumulative product of {v}", "cumprod({v})"),
    // ---- matrices ----------------------------------------------------------
    k(MAT, "the transpose of {M}", "transpose({M})"),
    k(MAT, "the trace of {M}", "trace({M})"),
    v(MAT, "the diagonal of {M}", "diag({M})"),
    v(MAT, "the diagonal part of {M}", "diag(diag({M}))"),
    v(MAT, "the off-diagonal part of {M}", "{M} - diag(diag({M}))"),
    v(MAT, "the sum of the off-diagonal entries of {M}", "sum({M} - diag(diag({M})))"),
    v(MAT, "the off-diagonal row sums of {M}", "sum({M} - diag(diag({M})), axis=1)"),
    v(MAT, "the off-diagonal column sums of {M}", "sum({M} - diag(diag({M})), axis=0)"),
    v(MAT, "the upper triangle of {M}", "triu({M})"),
    v(MAT, "the lower triangle of {M}", "tril({M})"),
    v(MAT, "the strict upper triangle of {M}", "triu({M}, k=1)"),
    v(MAT, "the strict lower triangle of {M}", "tril({M}, k=-1)"),
    v(MAT, "the row sums of {M}", "sum({M}, axis=1)"),
    v(MAT, "the column sums of {M}", "sum({M}, axis=0)"),
    v(MAT, "the sum|mean of each row of {M}", "{sum|mean}({M}, axis=1)"),
    v(MAT, "the sum|mean of each column of {M}", "{sum|mean}({M}, axis=0)"),
    v(MAT, "the max|maximum of each row of {M}", "amax({M}, axis=1)"),
    v(MAT, "the max|maximum of each column of {M}", "amax({M}, axis=0)"),
    v(MAT, "the min|minimum of each row of {M}", "amin({M}, axis=1)"),
    v(MAT, "the min|minimum of each column of {M}", "amin({M}, axis=0)"),
    v(MAT, "the inverse of {M}", "inv({M})"),
    v(MAT, "sum the off-diagonal entries of {M}", "sum({M} - diag(diag({M})))"),
    v(MAT, "sum the off-diagonal entries of each row of {M}", "sum({M} - diag(diag({M})), axis=1)"),
    v(MAT, "sum the off-diagonal entries of each column of {M}", "sum({M} - diag(diag({M})), axis=0)"),
    v(MAT, "sum each row of {M}", "sum({M}, axis=1)"),
    v(MAT, "sum each column of {M}", "sum({M}, axis=0)"),
    // ---- vectors (Knot) ----------------------------------------------------
    k(VEC, "the L2 norm of {v}", "norm({v})"),
    v(VEC, "the Frobenius norm of {M}", "norm({M})"),
    k(VEC, "the norm of {v}", "norm({v})"),
    k(VEC, "the magnitude of {v}", "norm({v})"),
    k(VEC, "the dot product of {u} and {v}", "dot({u}, {v})"),
    k(VEC, "the inner product of {u} and {v}", "dot({u}, {v})"),
    v(VEC, "the outer product of {u} and {v}", "outer({u}, {v})"),
    // ---- sizes and elements ------------------------------------------------
    k(SIZE, "the size of {v}", "len({v})"),
    k(SIZE, "the length of {v}", "len({v})"),
    k(SIZE, "the number of rows of {M}", "shape({M})[0]"),
    k(SIZE, "the number of columns of {M}", "shape({M})[1]"),
    v(SIZE, "the number of elements of {A}", "numel({A})"),
    v(SIZE, "the shape of {A}", "shape({A})"),
    v(SIZE, "the first element of {v}", "{v}[0]"),
    v(SIZE, "the last element of {v}", "{v}[len({v}) - 1]"),
    v(SIZE, "element {i} of {v}", "{v}[{i}]"),
    v(SIZE, "row {i} of {M}", "{M}[{i}, :]"),
    v(SIZE, "column {j} of {M}", "{M}[:, {j}]"),
    v(SIZE, "{v} sorted", "take({v}, argsort({v}), axis=0)"),
    v(SIZE, "the reverse of {v}", "take({v}, arange(len({v}) - 1, -1, -1), axis=0)"),
    // ---- elementary functions (Knot) ---------------------------------------
    k(MATH, "the sine of {x}", "sin({x})"),
    k(MATH, "the cosine of {x}", "cos({x})"),
    k(MATH, "the tangent of {x}", "tan({x})"),
    k(MATH, "the arcsine of {x}", "asin({x})"),
    k(MATH, "the arccosine of {x}", "acos({x})"),
    k(MATH, "the arctangent of {x}", "atan({x})"),
    k(MATH, "the exponential of {x}", "exp({x})"),
    k(MATH, "the natural log of {x}", "log({x})"),
    k(MATH, "the logarithm of {x}", "log({x})"),
    k(MATH, "the square root of {x}", "sqrt({x})"),
    k(MATH, "the absolute value of {x}", "abs({x})"),
    k(MATH, "the floor of {x}", "floor({x})"),
    k(MATH, "the ceiling of {x}", "ceil({x})"),
    k(MATH, "the nearest integer to {x}", "round({x})"),
    // ---- geometry and linear algebra (Knot) --------------------------------
    k(LIN, "the matrix product of {A} and {B}", "{A} @ {B}"),
    k(LIN, "the projection of {u} onto {v}", "dot({u}, {v}) / dot({v}, {v}) * {v}"),
    k(LIN, "the component of {u} along {v}", "dot({u}, {v}) / dot({v}, {v}) * {v}"),
    k(LIN, "the unit vector along {v}", "{v} / norm({v})"),
    k(LIN, "the unit vector in the direction of {v}", "{v} / norm({v})"),
    k(LIN, "the angle between {u} and {v}", "acos(dot({u}, {v}) / (norm({u}) * norm({v})))"),
    k(LIN, "the distance from {u} to {v}", "norm({u} - {v})"),
    k(LIN, "the distance between {u} and {v}", "norm({u} - {v})"),
    v(LIN, "the solution of {A} @ {x:var} = {b}", "lin_solve({A}, {b})"),
    v(LIN, "solve {A} @ {x:var} = {b}", "lin_solve({A}, {b})"),
    v(LIN, "the residual of {A} @ {x} = {b}", "{A} @ {x} - {b}"),
    v(LIN, "the residual norm of {A} @ {x} = {b}", "norm({A} @ {x} - {b})"),
    v(LIN, "the largest residual of {A} @ {x} = {b}", "amax(abs({A} @ {x} - {b}))"),
    v(LIN, "the largest difference between {u} and {v}", "amax(abs({u} - {v}))"),
    // ---- construction ------------------------------------------------------
    v(MAKE, "a vector of zeros of length {n}", "zeros({n})"),
    v(MAKE, "a vector of ones of length {n}", "ones({n})"),
    v(MAKE, "a matrix of zeros with {m} rows and {n} columns", "zeros({m}, {n})"),
    v(MAKE, "the identity matrix of size {n}", "eye({n})"),
    v(MAKE, "{n} evenly spaced points from {a} to {b}", "linspace({a}, {b}, {n})"),
    v(MAKE, "the range from {a} to {b}", "arange({a}, {b})"),
    // ---- probability and learning (the GPT rewrite) --
    v(SIZE, "the first row of {M}", "{M}[0, :]"),
    v(SIZE, "the last row of {M}", "{M}[len({M}) - 1, :]"),
    v(SIZE, "the last {n} elements of {v}", "{v}[max(0, len({v}) - {n}):len({v})]"),
    v(LEARN, "the softmax of {v}", "softmax({v})"),
    v(LEARN, "the softmax of each row of {M}", "softmax_rows({M})"),
    v(LEARN, "the cross-entropy of {L} against {y}", "cross_entropy({L}, {y})"),
    v(LEARN, "the causal mask of size {n}", "triu(ones({n}, {n}), k=1)"),
    v(LEARN, "the gradient of {f:call} with respect to {x:var}", "grad({f})"),
    v(LEARN, "the value and gradient of {f:call} with respect to {x:var}", "value_and_grad({f})"),
    // ---- integrals, roots and derivatives ----------------------------------
    k(CALC, "the integral of {f:fn} from {a} to {b} with {n} intervals", "simpson({f}, {a}, {b}, {n})"),
    k(CALC, "the integral of {f:fn} from {a} to {b}", "q_gauss({f}, {a}, {b})"),
    v(CALC, "the integral of {f:fn} from {a} to {b} using every method", "viv__every_integrate({f}, {a}, {b})"),
    v(CALC, "the integral of {f:fn} from {a} to {b} using simpson with {n} intervals", "simpson({f}, {a}, {b}, {n})"),
    v(CALC, "the integral of {f:fn} from {a} to {b} using the trapezoid rule with {n} intervals", "trapezoid({f}, {a}, {b}, {n})"),
    v(CALC, "the integral of {f:fn} from {a} to {b} using gauss-legendre", "q_gauss({f}, {a}, {b})"),
    v(CALC, "the integral of {f:fn} from {a} to {b} using adaptive simpson", "q_adaptive({f}, {a}, {b})"),
    v(CALC, "the integral of {f:fn} from {a} to {b} using romberg", "q_romberg({f}, {a}, {b})"),
    k(CALC, "the root|solution of {f:fn} starting at {x0}", "r_newton({f}, {f'}, {x0})"),
    v(CALC, "the root|solution of {f:fn} starting at {x0} using newton", "r_newton({f}, {f'}, {x0})"),
    v(CALC, "the root|solution of {f:fn} using newton starting at {x0}", "r_newton({f}, {f'}, {x0})"),
    k(CALC, "the root|solution of {f:fn} between {a} and {b}", "r_safe({f}, {a}, {b})"),
    v(CALC, "the root|solution of {f:fn} between {a} and {b} using every method", "viv__every_find_root({f}, {a}, {b})"),
    v(CALC, "the root|solution of {f:fn} between {a} and {b} using bisection", "r_bisect({f}, {a}, {b})"),
    v(CALC, "the root|solution of {f:fn} between {a} and {b} using illinois", "r_illinois({f}, {a}, {b})"),
    v(CALC, "the root|solution of {f:fn} between {a} and {b} using regula falsi", "r_illinois({f}, {a}, {b})"),
    v(CALC, "the root|solution of {f:fn} between {a} and {b} using false position", "r_illinois({f}, {a}, {b})"),
    v(CALC, "the root|solution of {f:fn} between {a} and {b} using safeguarded secant", "r_safe({f}, {a}, {b})"),
    k(CALC, "the derivative of {f:fn} at {x}", "deriv({f}, {x})"),
    v(CALC, "the derivative of {f:fn} at {x} using the symbolic derivative", "cas_deriv_at({f#text}, {f#var}, {x}, {f#names}, {f#vals})"),
    // ---- the same, without `take` ------------------------------------------
    v(CALC, "integrate {f:fn} from {a} to {b}", "q_gauss({f}, {a}, {b})"),
    v(CALC, "integrate {f:fn} from {a} to {b} using every method", "viv__every_integrate({f}, {a}, {b})"),
    v(CALC, "integrate {f:fn} from {a} to {b} with {n} intervals", "simpson({f}, {a}, {b}, {n})"),
    v(CALC, "integrate {f:fn} from {a} to {b} using simpson with {n} intervals", "simpson({f}, {a}, {b}, {n})"),
    v(CALC, "integrate {f:fn} from {a} to {b} using the trapezoid rule with {n} intervals", "trapezoid({f}, {a}, {b}, {n})"),
    v(CALC, "integrate {f:fn} from {a} to {b} using gauss-legendre", "q_gauss({f}, {a}, {b})"),
    v(CALC, "integrate {f:fn} from {a} to {b} using adaptive simpson", "q_adaptive({f}, {a}, {b})"),
    v(CALC, "integrate {f:fn} from {a} to {b} using romberg", "q_romberg({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} starting at {x0}", "r_newton({f}, {f'}, {x0})"),
    v(CALC, "solve {f:fn} starting at {x0} using newton", "r_newton({f}, {f'}, {x0})"),
    v(CALC, "solve {f:fn} using newton starting at {x0}", "r_newton({f}, {f'}, {x0})"),
    v(CALC, "solve {f:fn} between {a} and {b}", "r_safe({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} between {a} and {b} using every method", "viv__every_find_root({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} between {a} and {b} using bisection", "r_bisect({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} between {a} and {b} using illinois", "r_illinois({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} between {a} and {b} using regula falsi", "r_illinois({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} between {a} and {b} using false position", "r_illinois({f}, {a}, {b})"),
    v(CALC, "solve {f:fn} between {a} and {b} using safeguarded secant", "r_safe({f}, {a}, {b})"),
    v(CALC, "differentiate {f:fn} at {x}", "deriv({f}, {x})"),
    v(CALC, "differentiate {f:fn} at {x} using the symbolic derivative", "cas_deriv_at({f#text}, {f#var}, {x}, {f#names}, {f#vals})"),
];

/// The library files a lowering may call into, by the path the reference
/// page names them under, and their text.
pub const STD: &[(&str, &str)] = &[
    ("lang/viv/std/contracts.viv", include_str!("../std/contracts.viv")),
    ("lang/viv/std/phrases.viv", include_str!("../std/phrases.viv")),
];

// ---- patterns -------------------------------------------------------------------

/// What a hole holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// any expression
    Expr,
    /// an expression in one free variable, or an equation, made a function
    Fn,
    /// one name
    Var,
    /// a call of a def, `loss(w, X, y)`: the lowering spreads it into a
    /// call's arguments, `grad({f})` -> `grad(loss, w, X, y)`
    Call,
}

/// One element of a pattern.
#[derive(Clone, Debug)]
pub enum El {
    /// a word, or one of several (`root|solution`)
    Lit(Vec<String>),
    Hole { name: String, kind: Kind },
}

/// A pattern's words as elements. `off-diagonal` is the three words
/// `off`, `-`, `diagonal`, as the lexer reads it.
pub fn elements(words: &str) -> Vec<El> {
    let mut out = Vec::new();
    for w in comma_words(words) {
        let w = w.as_str();
        if let Some(inner) = w.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
            let (name, kind) = match inner.split_once(':') {
                Some((n, "fn")) => (n, Kind::Fn),
                Some((n, "var")) => (n, Kind::Var),
                Some((n, "call")) => (n, Kind::Call),
                Some((n, other)) => panic!("phrase hole `{n}` has unknown kind `{other}`"),
                None => (inner, Kind::Expr),
            };
            out.push(El::Hole { name: name.to_string(), kind });
        } else if w.contains('|') {
            out.push(El::Lit(w.split('|').map(str::to_string).collect()));
        } else {
            let mut first = true;
            for part in w.split('-') {
                if !first {
                    out.push(El::Lit(vec!["-".to_string()]));
                }
                first = false;
                if !part.is_empty() {
                    out.push(El::Lit(vec![part.to_string()]));
                }
            }
        }
    }
    out
}

/// The words of a pattern, with a `,` its own word: `q, k and v` is `q`,
/// `,`, `k`, `and`, `v`, as the lexer reads it.
pub fn comma_words(words: &str) -> Vec<String> {
    words.replace(',', " , ").split_whitespace().map(str::to_string).collect()
}

/// Whether a pattern has a `,` among its words: only such a pattern reads a
/// phrase past a comma.
pub fn has_comma(words: &str) -> bool {
    words.contains(',')
}

/// The pattern as a reader writes it: holes in capitals, `a|b` kept.
pub fn display(words: &str) -> String {
    words
        .split_whitespace()
        .map(|w| match w.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
            Some(inner) => inner.split(':').next().unwrap_or(inner).to_uppercase(),
            None => w.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" ,", ",")
}

/// A pattern with its holes unnamed and its words in lower case: two
/// patterns with the same key read the same phrases.
pub fn shape_key(words: &str) -> String {
    words
        .split_whitespace()
        .map(|w| if w.starts_with('{') { "{}".to_string() } else { w.to_lowercase() })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether `p` is opened by a verb rather than by `take`.
pub fn bare_verb(p: &Pattern) -> Option<&'static str> {
    let first = p.words.split_whitespace().next()?;
    BARE_VERBS.iter().copied().find(|v| *v == first)
}

pub fn is_bare_verb(name: &str) -> bool {
    BARE_VERBS.contains(&name)
}

/// A token as a pattern word: names, and the keywords and the `-` a pattern
/// can contain, by their source text.
pub fn word(t: &Tok) -> Option<String> {
    Some(match t {
        Tok::Ident(s) => s.clone(),
        Tok::Minus => "-".into(),
        Tok::Comma => ",".into(),
        Tok::At => "@".into(),
        Tok::Eq => "=".into(),
        Tok::To => "to".into(),
        Tok::And => "and".into(),
        Tok::Or => "or".into(),
        Tok::Not => "not".into(),
        Tok::For => "for".into(),
        Tok::While => "while".into(),
        Tok::Return => "return".into(),
        Tok::Show => "show".into(),
        Tok::Test => "test".into(),
        Tok::True => "true".into(),
        Tok::False => "false".into(),
        _ => return None,
    })
}

/// One token of a phrase, as the matcher sees it.
pub struct PTok {
    pub word: Option<String>,
    /// brackets open before this token, counted from the phrase's start
    pub depth: i32,
    pub ident: bool,
}

/// The phrase's tokens for matching, and the bracket depth after the last.
pub fn ptoks(toks: &[Token]) -> (Vec<PTok>, i32) {
    let mut depth = 0;
    let mut out = Vec::with_capacity(toks.len());
    for t in toks {
        out.push(PTok { word: word(&t.tok), depth, ident: matches!(t.tok, Tok::Ident(_)) });
        match t.tok {
            Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
            Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
            _ => {}
        }
    }
    (out, depth)
}

/// Every way to split the phrase `toks` among the holes of `els`: for each,
/// the token range of every hole, in order. A literal matches only outside
/// brackets, and a hole is not empty and holds whole bracket pairs.
pub fn splits(els: &[El], toks: &[PTok], end_depth: i32) -> Vec<Vec<(usize, usize)>> {
    fn depth_at(toks: &[PTok], end_depth: i32, i: usize) -> i32 {
        if i < toks.len() { toks[i].depth } else { end_depth }
    }
    fn lit_at(alts: &[String], toks: &[PTok], i: usize) -> bool {
        i < toks.len() && toks[i].depth == 0 && toks[i].word.as_ref().is_some_and(|w| alts.iter().any(|a| a == w))
    }
    fn go(els: &[El], k: usize, pos: usize, toks: &[PTok], end_depth: i32, cur: &mut Vec<(usize, usize)>, out: &mut Vec<Vec<(usize, usize)>>) {
        let n = toks.len();
        if k == els.len() {
            if pos == n {
                out.push(cur.clone());
            }
            return;
        }
        match &els[k] {
            El::Lit(alts) => {
                if lit_at(alts, toks, pos) {
                    go(els, k + 1, pos + 1, toks, end_depth, cur, out);
                }
            }
            El::Hole { kind: Kind::Var, .. } => {
                if pos < n && toks[pos].ident && toks[pos].depth == 0 {
                    cur.push((pos, pos + 1));
                    go(els, k + 1, pos + 1, toks, end_depth, cur, out);
                    cur.pop();
                }
            }
            El::Hole { .. } => {
                let last = k + 1 == els.len();
                for end in pos + 1..=n {
                    if last && end != n {
                        continue;
                    }
                    if depth_at(toks, end_depth, end) != 0 {
                        continue;
                    }
                    if let Some(El::Lit(alts)) = els.get(k + 1) {
                        if !lit_at(alts, toks, end) {
                            continue;
                        }
                    }
                    cur.push((pos, end));
                    go(els, k + 1, end, toks, end_depth, cur, out);
                    cur.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    if toks.first().is_some_and(|t| t.depth != 0) {
        return out;
    }
    go(els, 0, 0, toks, end_depth, &mut Vec::new(), &mut out);
    out
}

/// The patterns closest to a phrase that matched none, best first: by the
/// longest run of the pattern's words found in the phrase in order.
pub fn closest(words: &[String], bare: Option<&str>, n: usize) -> Vec<&'static Pattern> {
    let mut scored: Vec<(usize, usize, &'static Pattern)> = TABLE
        .iter()
        .enumerate()
        .filter(|(_, p)| bare_verb(p) == bare)
        .map(|(i, p)| {
            let lits: Vec<Vec<String>> = elements(p.words)
                .into_iter()
                .filter_map(|e| if let El::Lit(a) = e { Some(a) } else { None })
                .collect();
            // longest common subsequence of the pattern's words and the phrase's
            let mut dp = vec![vec![0usize; words.len() + 1]; lits.len() + 1];
            for a in 0..lits.len() {
                for b in 0..words.len() {
                    dp[a + 1][b + 1] = if lits[a].contains(&words[b]) { dp[a][b] + 1 } else { dp[a][b + 1].max(dp[a + 1][b]) };
                }
            }
            let common = dp[lits.len()][words.len()];
            // more words in common first; then fewer words the phrase lacks
            (common, usize::MAX - (lits.len() - common), i)
        })
        .map(|(c, m, i)| (c, m, &TABLE[i]))
        .collect();
    scored.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
    scored.into_iter().filter(|s| s.0 > 0).take(n).map(|s| s.2).collect()
}

/// The methods a phrase like `words` can name: the words after `using` in
/// the patterns whose other literal words all appear in the phrase.
pub fn methods_like(words: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in TABLE {
        let ws: Vec<&str> = p.words.split_whitespace().collect();
        let Some(u) = ws.iter().position(|w| *w == "using") else { continue };
        let tail: Vec<&str> = ws[u + 1..].iter().take_while(|w| !w.starts_with('{') && **w != "with" && **w != "starting").copied().collect();
        let rest: Vec<El> = ws.iter().enumerate().filter(|(i, _)| *i < u || *i > u + tail.len()).flat_map(|(_, w)| elements(w)).collect();
        let present = rest.iter().all(|e| match e {
            El::Lit(alts) => alts.iter().any(|a| words.contains(a)),
            El::Hole { .. } => true,
        });
        let m = tail.join(" ");
        if present && !m.is_empty() && !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

/// The def names the std files define (`STD`), from their text: each
/// `def` and each `contract`.
pub fn std_def_names() -> &'static HashSet<String> {
    static NAMES: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        let mut s = HashSet::new();
        for (_, text) in STD {
            for line in text.lines() {
                // a contract is a def by its name, with its dispatcher and
                // its every-method def (crate::contracts)
                if let Some(rest) = line.strip_prefix("def ") {
                    if let Some(name) = rest.split('(').next() {
                        s.insert(name.trim().to_string());
                    }
                }
                if let Some(rest) = line.strip_prefix("contract ") {
                    if let Some(name) = rest.split('(').next() {
                        let name = name.trim();
                        s.insert(name.to_string());
                        s.insert(format!("{}{name}", crate::contracts::BY));
                        s.insert(format!("{}{name}", crate::contracts::EVERY_DEF));
                    }
                }
            }
        }
        s
    })
}

/// One phrase as the parser read it: where it is, which pattern, its holes
/// and the std defs its lowering calls.
#[derive(Clone, Debug)]
pub struct PhraseUse {
    pub span: Span,
    /// the pattern of `TABLE`, or `None` for the program's own phrase
    pub pattern: Option<usize>,
    /// the pattern's words, in the table's notation
    pub words: String,
    /// for the program's own phrase, where the body of its declaration is
    pub body: Option<Span>,
    /// for a phrase a used module declares, read from outside the program's
    /// text (`crate::used_phrases`), the text of its body
    pub body_text: Option<String>,
    pub holes: Vec<Hole>,
    pub needs: Vec<String>,
}

/// One hole of a phrase as read: its name, all of it, where its
/// expression's text is, and for a `fn` hole the variable it is a function of.
#[derive(Clone, Debug)]
pub struct Hole {
    pub name: String,
    pub span: Span,
    pub text: HoleText,
    pub var: Option<String>,
}

/// Where a hole's expression is written: one piece of text, or the two
/// sides of an equation, which the lowering takes as `lhs - (rhs)`.
#[derive(Clone, Copy, Debug)]
pub enum HoleText {
    Plain(Span),
    Equation(Span, Span),
}

// ---- names a phrase hole can read ------------------------------------------------

/// The names `toks` assigns or binds: `x = ...` (and `+=` and the rest),
/// `a, b = ...`, the names of a `for` head, a def's parameters are added by
/// the caller. Read from tokens, before the statements are parsed, so a name
/// assigned later in the same body counts: a phrase in a loop can read a
/// variable the loop sets after it. `skip_defs` leaves out what is inside
/// def bodies (for the top level).
pub fn assigned(toks: &[Token], skip_defs: bool) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut i = 0;
    let n = toks.len();
    let assign = |t: &Tok| matches!(t, Tok::Eq | Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq);
    while i < n {
        match &toks[i].tok {
            Tok::Def if skip_defs => {
                // to the def's `{`, then past its matching `}`
                while i < n && toks[i].tok != Tok::LBrace {
                    i += 1;
                }
                let mut depth = 0;
                while i < n {
                    match toks[i].tok {
                        Tok::LBrace => depth += 1,
                        Tok::RBrace => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            Tok::For => {
                let mut j = i + 1;
                while j < n {
                    if let Tok::Ident(s) = &toks[j].tok {
                        out.insert(s.clone());
                    }
                    if toks.get(j + 1).map(|t| &t.tok) == Some(&Tok::Comma) {
                        j += 2;
                    } else {
                        break;
                    }
                }
            }
            Tok::Ident(s) => {
                // only a name that starts a statement is assigned: the `x = b`
                // inside `solve A @ x = b` is an equation, not an assignment
                let starts_stmt = i == 0 || matches!(toks[i - 1].tok, Tok::Newline | Tok::LBrace | Tok::RBrace);
                if starts_stmt {
                    if toks.get(i + 1).is_some_and(|t| assign(&t.tok)) {
                        out.insert(s.clone());
                    } else if toks.get(i + 1).map(|t| &t.tok) == Some(&Tok::Comma) {
                        // `a, b = ...`: names and commas up to the `=`
                        let mut j = i;
                        let mut names = Vec::new();
                        while j + 1 < n {
                            if let Tok::Ident(x) = &toks[j].tok {
                                names.push(x.clone());
                            } else {
                                break;
                            }
                            if toks[j + 1].tok == Tok::Comma {
                                j += 2;
                            } else {
                                break;
                            }
                        }
                        if toks.get(j + 1).is_some_and(|t| t.tok == Tok::Eq) {
                            out.extend(names);
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// The names `e` reads that no lambda inside it binds, in order of first
/// appearance.
pub fn free_names(e: &Expr) -> Vec<String> {
    fn go(e: &Expr, bound: &mut Vec<String>, out: &mut Vec<String>) {
        match e {
            Expr::Var { name, .. } => {
                if !bound.contains(name) && !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Expr::Lambda { params, body, .. } => {
                let before = bound.len();
                bound.extend(params.iter().cloned());
                go(body, bound, out);
                bound.truncate(before);
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| go(a, bound, out)),
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|a| go(a, bound, out)),
            Expr::Index { base, indices, .. } => {
                go(base, bound, out);
                indices.iter().for_each(|a| go(a, bound, out));
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => go(operand, bound, out),
            Expr::Binary { lhs, rhs, .. } => {
                go(lhs, bound, out);
                go(rhs, bound, out);
            }
            Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| go(x, bound, out)),
            Expr::Field { base, .. } => go(base, bound, out),
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }
    let mut out = Vec::new();
    go(e, &mut Vec::new(), &mut out);
    out
}

/// Every sub-expression of `e`, children after their parent, mutably.
pub fn walk_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    f(e);
    match e {
        Expr::Lambda { body, .. } => walk_mut(body, f),
        Expr::Call { args, .. } => args.iter_mut().for_each(|a| walk_mut(a, f)),
        Expr::ArrLit { elems, .. } => elems.iter_mut().for_each(|a| walk_mut(a, f)),
        Expr::Index { base, indices, .. } => {
            walk_mut(base, f);
            indices.iter_mut().for_each(|a| walk_mut(a, f));
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk_mut(operand, f),
        Expr::Binary { lhs, rhs, .. } => {
            walk_mut(lhs, f);
            walk_mut(rhs, f);
        }
        Expr::RecordLit { fields, .. } => fields.iter_mut().for_each(|(_, _, x)| walk_mut(x, f)),
        Expr::Field { base, .. } => walk_mut(base, f),
        Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } | Expr::Var { .. } => {}
    }
}

/// `e` with the variable `from` renamed `to`, except where a lambda inside
/// binds `from` again.
pub fn rename(e: &mut Expr, from: &str, to: &str) {
    match e {
        Expr::Var { name, .. } if name == from => *name = to.to_string(),
        Expr::Lambda { params, body, .. } => {
            if !params.iter().any(|p| p == from) {
                rename(body, from, to);
            }
        }
        Expr::Call { args, .. } => args.iter_mut().for_each(|a| rename(a, from, to)),
        Expr::ArrLit { elems, .. } => elems.iter_mut().for_each(|a| rename(a, from, to)),
        Expr::Index { base, indices, .. } => {
            rename(base, from, to);
            indices.iter_mut().for_each(|a| rename(a, from, to));
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => rename(operand, from, to),
        Expr::Binary { lhs, rhs, .. } => {
            rename(lhs, from, to);
            rename(rhs, from, to);
        }
        Expr::RecordLit { fields, .. } => fields.iter_mut().for_each(|(_, _, x)| rename(x, from, to)),
        Expr::Field { base, .. } => rename(base, from, to),
        _ => {}
    }
}

/// Every span in `e` set to `sp`: a program's own phrase's body, put where
/// the phrase is.
pub fn set_spans(e: &mut Expr, sp: Span) {
    walk_mut(e, &mut |x: &mut Expr| match x {
        Expr::Imag { span, .. }
        | Expr::Bool { span, .. }
        | Expr::Num { span, .. }
        | Expr::Str { span, .. }
        | Expr::Var { span, .. }
        | Expr::ArrLit { span, .. }
        | Expr::Index { span, .. }
        | Expr::Lambda { span, .. }
        | Expr::Neg { span, .. }
        | Expr::Not { span, .. }
        | Expr::Binary { span, .. }
        | Expr::Call { span, .. } => *span = sp,
        Expr::RecordLit { span, type_span, fields, .. } => {
            *span = sp;
            *type_span = sp;
            for (_, fs, _) in fields.iter_mut() {
                *fs = sp;
            }
        }
        Expr::Field { span, field_span, .. } => {
            *span = sp;
            *field_span = sp;
        }
    });
}

/// `text` with every whole-word `name` replaced by `with`.
pub fn replace_word(text: &str, name: &str, with: &str) -> String {
    let mut out = String::new();
    let b = text.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(name) && (i == 0 || !word(b[i - 1])) && (i + name.len() == text.len() || !word(b[i + name.len()])) {
            out.push_str(with);
            i += name.len();
        } else {
            let c = text[i..].chars().next().expect("a character");
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// Whether reading `e` twice reads the same value and does nothing else.
pub fn trivial(e: &Expr) -> bool {
    match e {
        Expr::Num { .. } | Expr::Var { .. } | Expr::Bool { .. } | Expr::Str { .. } => true,
        Expr::Neg { operand, .. } => matches!(**operand, Expr::Num { .. }),
        Expr::Field { base, .. } => trivial(base),
        _ => false,
    }
}

/// The text of a lowering with the holes' own text in it: what a reader
/// would have written instead of the phrase (`viv phrases <file>`). A `fn`
/// hole is written `fn(x) -> <text>`.
pub fn lowered_text(p: &Pattern, holes: &[(String, String, Option<String>)], chosen: &[String]) -> String {
    let mut out = p.lowers.to_string();
    for (name, text, var) in holes {
        // a call hole spread into the lowering's call: `loss(w, X)` in
        // `grad({f})` is `grad(loss, w, X)`
        if let (Some(open), true) = (text.find('('), p.words.contains(&format!("{{{name}:call}}"))) {
            let spread = format!("{}, {}", &text[..open], text[open + 1..].trim_end().strip_suffix(')').unwrap_or(&text[open + 1..]));
            out = out.replace(&format!("{{{name}}}"), &spread);
            continue;
        }
        let shown = match var {
            Some(x) => format!("fn({x}) -> {text}"),
            None => {
                let simple = text.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.');
                if simple { text.clone() } else { format!("({text})") }
            }
        };
        if let Some(x) = var {
            out = out.replace(&format!("{{{name}'}}"), &format!("fn({x}) -> deriv(fn({x}) -> {text}, {x})"));
            if out.contains(&format!("{{{name}#")) {
                // the hole handed to the CAS: the other names are the words of
                // the text that are not the variable and not a call
                let mut names: Vec<String> = Vec::new();
                let b: Vec<char> = text.chars().collect();
                let mut i = 0;
                while i < b.len() {
                    if b[i].is_alphabetic() || b[i] == '_' {
                        let st = i;
                        while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                            i += 1;
                        }
                        let w: String = b[st..i].iter().collect();
                        let call = b[i..].iter().find(|c| !c.is_whitespace()) == Some(&'(');
                        if !call && w != *x && w != "pi" && !names.contains(&w) {
                            names.push(w);
                        }
                    } else {
                        i += 1;
                    }
                }
                let vals = if names.is_empty() { "zeros(0)".to_string() } else { format!("[{}]", names.join(", ")) };
                out = out
                    .replace(&format!("{{{name}#text}}"), &format!("\"{text}\""))
                    .replace(&format!("{{{name}#var}}"), &format!("\"{x}\""))
                    .replace(&format!("{{{name}#names}}"), &format!("\"{}\"", names.join(",")))
                    .replace(&format!("{{{name}#vals}}"), &vals);
            }
        }
        out = out.replace(&format!("{{{name}}}"), &shown);
    }
    replace_alt(&out, chosen)
}

/// A lowering's `{a|b}` with the word the phrase used (one of `chosen`, the
/// words the phrase matched a literal `a|b` with).
pub fn replace_alt(lowers: &str, chosen: &[String]) -> String {
    let mut out = String::new();
    let mut rest = lowers;
    while let Some(i) = rest.find('{') {
        let Some(j) = rest[i..].find('}') else { break };
        let inner = &rest[i + 1..i + j];
        out.push_str(&rest[..i]);
        if inner.contains('|') {
            let pick = inner.split('|').find(|a| chosen.iter().any(|c| c == a)).unwrap_or_else(|| inner.split('|').next().unwrap_or(""));
            out.push_str(pick);
        } else {
            out.push_str(&rest[i..i + j + 1]);
        }
        rest = &rest[i + j + 1..];
    }
    out.push_str(rest);
    out
}

// ---- the std defs a lowering calls ------------------------------------------------

/// Add to `program` the std defs its phrases call (`PhraseUse::needs`) and
/// the defs those call, unless the program has them. A def the program has
/// with the same text as the std one (the program `#! use`s the std file) is
/// the program's; one with other text is an error, since the phrase would
/// call a def it was not written for. The added defs carry the span of the
/// first phrase that needs them, so an error inside one points at the phrase.
pub fn provide(mut program: Program, src: &str) -> Result<Program, Vec<Diagnostic>> {
    let mut needs: Vec<(String, Span)> = Vec::new();
    for u in &program.phrases {
        for n in &u.needs {
            if !needs.iter().any(|(m, _)| m == n) {
                needs.push((n.clone(), u.span));
            }
        }
    }
    if needs.is_empty() {
        return Ok(program);
    }
    let at = needs[0].1;
    // the std defs, with their own text, and parsed with every span `at`
    let mut by_name: HashMap<String, (String, Def, &'static str)> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    // a contract's def comes with its enum and its selection (crate::contracts)
    let mut contract_items: HashMap<String, (Item, Item)> = HashMap::new();
    for (path, text) in STD {
        let toks = crate::lexer::lex(text, SourceId(0)).map_err(|ds| ds)?;
        let own = crate::parser::parse(toks.clone(), SourceId(0))?;
        let moved: Vec<Token> = toks.into_iter().map(|t| Token { tok: t.tok, span: at }).collect();
        let placed = crate::parser::parse(moved, SourceId(0))?;
        let contracts = crate::contracts::scan(text);
        for (a, b) in own.items.iter().zip(placed.items.iter()) {
            if let (Item::Def(d), Item::Def(p)) = (a, b) {
                let body = text[d.span.start as usize..d.span.end as usize].to_string();
                order.push(d.name.clone());
                by_name.insert(d.name.clone(), (body, p.clone(), path));
            }
        }
        for c in &contracts {
            let e = placed.items.iter().find(|i| matches!(i, Item::Enum(x) if x.name == c.enum_name()));
            let sel = format!("{}{}", crate::contracts::USING, c.name);
            let u = placed.items.iter().find(|i| matches!(i, Item::Stmt(Stmt::Assign { name, .. }) if *name == sel));
            if let (Some(e), Some(u)) = (e, u) {
                contract_items.insert(c.name.clone(), (e.clone(), u.clone()));
            }
        }
    }
    let mut have: HashMap<String, Span> = HashMap::new();
    fn collect(items: &[Item], have: &mut HashMap<String, Span>) {
        for it in items {
            match it {
                Item::Def(d) => {
                    have.insert(d.name.clone(), d.span);
                }
                Item::Rung(r) => collect(&r.items, have),
                _ => {}
            }
        }
    }
    collect(&program.items, &mut have);
    // the closure of the needs over the std defs' calls
    let mut want: Vec<String> = Vec::new();
    let mut stack: Vec<(String, Span)> = needs.clone();
    while let Some((name, why)) = stack.pop() {
        if want.contains(&name) {
            continue;
        }
        let Some((text, def, path)) = by_name.get(&name) else { continue };
        if let Some(sp) = have.get(&name) {
            let theirs = src.get(sp.start as usize..sp.end as usize).unwrap_or("");
            if theirs.trim() != text.trim() {
                return Err(vec![Diagnostic::error(format!("this program defines its own `{name}`, and a phrase calls the `{name}` of {path}"))
                    .with_primary(Span::new(sp.source, sp.start, sp.start + 4 + name.len() as u32), "the program's def")
                    .with_secondary(why, "the phrase that needs it")
                    .with_note(format!("rename the program's `{name}`, or call it directly instead of through the phrase"))]);
            }
        } else {
            want.push(name.clone());
        }
        let mut calls: Vec<String> = Vec::new();
        crate::defun::walk_block(&def.body, &mut |e: &Expr| match e {
            Expr::Call { callee, .. } => calls.push(callee.clone()),
            Expr::Var { name, .. } => calls.push(name.clone()),
            _ => {}
        });
        for c in calls {
            if by_name.contains_key(&c) && !want.contains(&c) {
                stack.push((c, why));
            }
        }
    }
    for name in order {
        if want.contains(&name) {
            let (_, def, _) = by_name.remove(&name).expect("a std def");
            let owner = name.strip_prefix(crate::contracts::BY).or_else(|| name.strip_prefix(crate::contracts::EVERY_DEF)).unwrap_or(&name).to_string();
            if let Some((e, u)) = contract_items.remove(&owner) {
                // the selection is set before the program's first statement
                program.items.insert(0, u);
                program.items.insert(0, e);
            }
            program.items.push(Item::Def(def));
        }
    }
    Ok(program)
}

// ---- the table and a program's phrases, as text --------------------------------

/// A pattern as written in a program: `take the sum of V`, `integrate F ...`.
pub fn written(p: &Pattern) -> String {
    let d = display(p.words);
    if bare_verb(p).is_some() { d } else { format!("take {d}") }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `viv phrases`: every pattern and what it lowers to, by group.
pub fn table_text() -> String {
    let mut out = String::new();
    let mut group = "";
    for p in TABLE {
        if p.group != group {
            group = p.group;
            out.push_str(&format!("\n{group}\n"));
        }
        out.push_str(&format!("  {:<78} {}\n", written(p), p.lowers));
    }
    out.push_str(&format!("\n{} patterns ({} from Knot's table).\n", TABLE.len(), TABLE.iter().filter(|p| p.knot).count()));
    out
}

/// `viv phrases --json`: the table as the reference page reads it.
pub fn table_json() -> String {
    let rows: Vec<String> = TABLE
        .iter()
        .map(|p| {
            format!(
                "{{\"group\":{},\"phrase\":{},\"words\":{},\"lowers\":{},\"knot\":{},\"bare\":{}}}",
                json_str(p.group),
                json_str(&written(p)),
                json_str(p.words),
                json_str(p.lowers),
                p.knot,
                bare_verb(p).is_some()
            )
        })
        .collect();
    let std: Vec<String> = STD.iter().map(|(path, _)| json_str(path)).collect();
    format!(
        "{{\"patterns\":[{}],\"count\":{},\"knot\":{},\"verbs\":[{}],\"std\":[{}]}}",
        rows.join(","),
        TABLE.len(),
        TABLE.iter().filter(|p| p.knot).count(),
        BARE_VERBS.iter().map(|v| json_str(v)).collect::<Vec<_>>().join(","),
        std.join(",")
    )
}

/// `viv phrases <file>`: each phrase of a program, its line, the phrase as
/// written and the call it is, with the holes' own text (Knot printed this
/// line to stderr on every parse; here it is a command, so a run's output is
/// only the program's).
pub fn uses_report(src: &str) -> Result<String, String> {
    uses_report_mapped(src, &|l| Some(l))
}

/// `uses_report` of a program whose `#! use` lines are expanded in `src`:
/// `line_of` maps a line of `src` (1-based) to the program's own line, or
/// None for a line a module brought, whose phrases are left out.
pub fn uses_report_mapped(src: &str, line_of: &dyn Fn(usize) -> Option<usize>) -> Result<String, String> {
    let prog = crate::parse_keeping_rungs(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    let text = |sp: Span| src.get(sp.start as usize..sp.end as usize).unwrap_or("").to_string();
    let line = |sp: Span| line_of(src[..(sp.start as usize).min(src.len())].matches('\n').count() + 1).unwrap_or(0);
    let mut out = String::new();
    for u in prog.phrases.iter().filter(|u| line_of(src[..(u.span.start as usize).min(src.len())].matches('\n').count() + 1).is_some()) {
        let holes: Vec<(String, String, Option<String>)> = u
            .holes
            .iter()
            .map(|h| {
                let t = match h.text {
                    HoleText::Plain(sp) => text(sp),
                    HoleText::Equation(l, r) if text(r).trim() == "0" => text(l),
                    HoleText::Equation(l, r) => format!("{} - ({})", text(l), text(r)),
                };
                (h.name.clone(), t, h.var.clone())
            })
            .collect();
        let phrase = text(u.span);
        let chosen: Vec<String> = phrase.split_whitespace().map(str::to_string).collect();
        let call = match (u.pattern, u.body) {
            (Some(i), _) => lowered_text(&TABLE[i], &holes, &chosen),
            (None, Some(b)) => {
                let mut t = u.body_text.clone().unwrap_or_else(|| text(b));
                for (name, h, _) in &holes {
                    let simple = h.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.');
                    t = replace_word(&t, name, &if simple { h.clone() } else { format!("({h})") });
                }
                let whose = if u.body_text.is_some() { "a used module's phrase" } else { "the program's phrase" };
                format!("{t}  ({whose} `{}`)", display(&u.words))
            }
            (None, None) => String::new(),
        };
        out.push_str(&format!("{:>4}  {}\n      = {}\n", line(u.span), phrase, call));
        if !u.needs.is_empty() {
            out.push_str(&format!("      calls {} (lang/viv/std)\n", u.needs.join(", ")));
        }
    }
    if prog.phrases.is_empty() {
        out.push_str("no phrases\n");
    }
    Ok(out)
}
