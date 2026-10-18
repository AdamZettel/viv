// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv as LaTeX: a printer from
//! expression and statement ASTs to LaTeX, in the spirit of Julia's Latexify
//! or sympy's `latex()`, and a check that a piece of LaTeX is something KaTeX
//! can typeset.
//!
//! The explorer shows, beside a line of code, the line as math. Because the
//! LaTeX is printed from the code's own syntax tree, the equation shown for a
//! line cannot drift from what runs; an edit to the line changes its math.
//!
//! What it prints:
//! - `a / b` as `\frac{a}{b}`, `sqrt(x)` as `\sqrt{x}`, `pow(a, b)` as `a^{b}`;
//! - `abs(x)` as `|x|`, `enorm(n, v)` and `norm(v)` as `\lVert v \rVert`;
//! - `a[i]` as `a_{i}`, `A[i, j]` as `A_{ij}`, `A @ B` as `A\,B`,
//!   `transpose(A)` as `A^{\top}`;
//! - `min`, `max`, the elementary functions, comparisons and `and`, `or`,
//!   `not` in conditions;
//! - names the intent's glossary gives a symbol (`delta  $\Delta$  ...`) as
//!   that symbol; one-letter names as themselves; other names as
//!   `\mathrm{name}`.
//!
//! Parentheses are added exactly where the precedence of the operators needs
//! them (a fraction groups its parts, so it needs none inside).
//!
//! An assignment whose two sides would print the same (`fnorm = enorm(m,
//! fvec)` where the glossary makes `fnorm` the norm of `fvec`) would read as
//! `x = x`. Its right side is then printed as a call by name,
//! `\lVert f \rVert = \operatorname{enorm}(m, f)`; a right side that is not a
//! call leaves the line out.

use std::collections::HashMap;

use crate::ast::{BinOp, Expr, Stmt};

/// Symbols by name: what the glossary gives each name in a def's scope.
pub type Symbols = HashMap<String, String>;

// Binding strength of what an expression renders to. A child whose strength
// is below what its parent needs gets parentheses.
const P_OR: u8 = 1;
const P_AND: u8 = 2;
const P_NOT: u8 = 3;
const P_CMP: u8 = 4;
const P_ADD: u8 = 5;
const P_MUL: u8 = 6;
const P_NEG: u8 = 7;
/// a fraction, a power, a transpose: fine as a factor, not as a base
const P_POST: u8 = 8;
const P_ATOM: u8 = 9;

/// More arguments than this and a call to a def shows the first few and `\ldots`.
const MAX_ARGS: usize = 5;
const SHOWN_ARGS: usize = 3;

/// The LaTeX of an expression.
pub fn expr(e: &Expr, syms: &Symbols) -> String {
    render(e, syms).0
}

fn paren(s: String) -> String {
    format!("\\left({s}\\right)")
}

/// `s` if it binds at least `need`, else in parentheses.
fn at(r: (String, u8), need: u8) -> String {
    if r.1 >= need {
        r.0
    } else {
        paren(r.0)
    }
}

/// A name with its underscores escaped, for `\mathrm` and `\operatorname`.
fn esc_name(name: &str) -> String {
    name.replace('_', "\\_")
}

/// A name as math: its symbol, itself when it is one letter, else upright.
fn name_tex(name: &str, syms: &Symbols) -> (String, u8) {
    if let Some(s) = syms.get(name) {
        let s = s.trim().to_string();
        let p = if is_atomic(&s) { P_ATOM } else { P_ADD };
        return (s, p);
    }
    if name.chars().count() == 1 && name.chars().all(|c| c.is_ascii_alphabetic()) {
        return (name.to_string(), P_ATOM);
    }
    (format!("\\mathrm{{{}}}", esc_name(name)), P_ATOM)
}

/// Whether a piece of LaTeX can take a subscript or be a factor without
/// parentheses: one symbol with its scripts (`x`, `\Delta`, `x_{1}`,
/// `\mathrm{ared}`), or something its own delimiters close (`\lVert D p
/// \rVert`, `|x|`, `\left( ... \right)`). Anything with an operator, a
/// comma or a space between parts at the top level is not.
pub fn is_atomic(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return false;
    }
    for (open, close) in [("\\lVert", "\\rVert"), ("\\left(", "\\right)"), ("\\left|", "\\right|"), ("\\lvert", "\\rvert"), ("\\langle", "\\rangle")] {
        if t.starts_with(open) && t.ends_with(close) && t.matches(open).count() == 1 {
            return true;
        }
    }
    if t.starts_with('|') && t.ends_with('|') && t.matches('|').count() == 2 {
        return true;
    }
    // walk the top level: a single command or letter, then only scripts
    let b = t.as_bytes();
    let mut i = 0;
    let mut depth = 0i32;
    let mut heads = 0;
    while i < b.len() {
        let c = b[i] as char;
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ if depth > 0 => {}
            '\\' => {
                // a command word, or a one-character command
                let mut j = i + 1;
                while j < b.len() && (b[j] as char).is_ascii_alphabetic() {
                    j += 1;
                }
                if j == i + 1 {
                    // `\,` `\;` `\ ` are spaces: two parts
                    return false;
                }
                let word = &t[i + 1..j];
                if matches!(word, "frac" | "dfrac" | "tfrac" | "cdot" | "times" | "le" | "ge" | "ne" | "leq" | "geq" | "pm") {
                    return false;
                }
                // a font or accent command's group is part of its head
                heads += 1;
                i = j;
                continue;
            }
            '^' | '_' => {
                // a script: its argument is a group or one token
                i += 1;
                if i < b.len() && b[i] == b'\\' {
                    let mut j = i + 1;
                    while j < b.len() && (b[j] as char).is_ascii_alphabetic() {
                        j += 1;
                    }
                    i = j;
                } else if i < b.len() && b[i] == b'{' {
                    continue;
                } else {
                    i += 1;
                }
                continue;
            }
            ' ' | '+' | '-' | '/' | '=' | '<' | '>' | ',' | '*' => return false,
            // a run of digits is one number; each letter is its own symbol
            _ if c.is_ascii_digit() || c == '.' => {
                if i == 0 || !((b[i - 1] as char).is_ascii_digit() || b[i - 1] == b'.') {
                    heads += 1;
                }
            }
            _ if c.is_ascii_alphabetic() => heads += 1,
            '\'' => {}
            _ => return false,
        }
        i += 1;
    }
    depth == 0 && heads == 1
}

/// A number as a person writes it: `1`, `0.5`, `1.5 \times 10^{-8}`, `10^{-4}`.
pub fn num(v: f64) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "\\mathrm{NaN}".into() } else if v > 0.0 { "\\infty".into() } else { "-\\infty".into() };
    }
    let a = v.abs();
    if v == v.trunc() && a < 1e6 {
        return format!("{}", v as i64);
    }
    if a >= 1e-3 && a < 1e6 {
        return format!("{v}");
    }
    let e = a.log10().floor() as i32;
    let m = v / 10f64.powi(e);
    // the mantissa as the shortest decimal that is still this number's
    let ms = {
        let mut s = format!("{:.12}", m);
        if s.contains('.') {
            s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        }
        s
    };
    if ms == "1" {
        format!("10^{{{e}}}")
    } else if ms == "-1" {
        format!("-10^{{{e}}}")
    } else {
        format!("{ms} \\times 10^{{{e}}}")
    }
}

fn args_tex(args: &[Expr], syms: &Symbols) -> String {
    args.iter().map(|a| expr(a, syms)).collect::<Vec<_>>().join(", ")
}

fn contains_var(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Var { name: n, .. } => n == name,
        Expr::Index { base, indices, .. } => contains_var(base, name) || indices.iter().any(|i| contains_var(i, name)),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => contains_var(operand, name),
        Expr::Binary { lhs, rhs, .. } => contains_var(lhs, name) || contains_var(rhs, name),
        Expr::Call { args, .. } => args.iter().any(|a| contains_var(a, name)),
        Expr::ArrLit { elems, .. } => elems.iter().any(|x| contains_var(x, name)),
        Expr::Field { base, .. } => contains_var(base, name),
        Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, v)| contains_var(v, name)),
        Expr::Lambda { .. } | Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => false,
    }
}

/// `base_{indices}`: one-letter indices run together (`A_{ij}`), others are
/// separated by commas (`J_{j + \mathrm{ii},\,j}`).
fn subscript(base: (String, u8), idx: Vec<String>) -> String {
    // a base with parts is parenthesized; one that already has a script is
    // braced, so the new subscript is its own
    let b = if base.1 < P_ATOM {
        paren(base.0)
    } else if has_top_script(&base.0) {
        format!("{{{}}}", base.0)
    } else {
        base.0
    };
    let short = idx.iter().all(|s| s.chars().count() == 1);
    let inner = if short { idx.join("") } else { idx.join(",\\,") };
    format!("{b}_{{{inner}}}")
}

fn has_top_script(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            '_' | '^' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn render(e: &Expr, syms: &Symbols) -> (String, u8) {
    match e {
        Expr::Num { value, .. } => {
            let s = num(*value);
            let p = if s.starts_with('-') { P_NEG } else if s.contains("\\times") || s.contains('^') { P_POST } else { P_ATOM };
            (s, p)
        }
        Expr::Imag { value, .. } => (format!("{}\\,i", num(*value)), P_MUL),
        Expr::Bool { value, .. } => (format!("\\text{{{value}}}"), P_ATOM),
        Expr::Str { value, .. } => {
            let clean: String = value.chars().filter(|c| !"{}\\$%#&^_~".contains(*c)).collect();
            (format!("\\text{{``{clean}''}}"), P_ATOM)
        }
        Expr::Var { name, .. } => name_tex(name, syms),
        Expr::ArrLit { elems, .. } => {
            let rows: Vec<&Expr> = elems.iter().collect();
            let matrix = rows.iter().all(|r| matches!(r, Expr::ArrLit { .. }));
            if matrix && !rows.is_empty() {
                let body: Vec<String> = rows
                    .iter()
                    .map(|r| if let Expr::ArrLit { elems, .. } = r { elems.iter().map(|x| expr(x, syms)).collect::<Vec<_>>().join(" & ") } else { String::new() })
                    .collect();
                (format!("\\begin{{bmatrix}} {} \\end{{bmatrix}}", body.join(" \\\\ ")), P_ATOM)
            } else {
                (format!("\\left({}\\right)", args_tex(elems, syms)), P_ATOM)
            }
        }
        Expr::Index { base, indices, .. } => {
            let b = render(base, syms);
            let idx: Vec<String> = indices.iter().map(|i| expr(i, syms)).collect();
            (subscript(b, idx), P_ATOM)
        }
        Expr::Lambda { params, .. } => (format!("\\operatorname{{fn}}({})", params.iter().map(|p| name_tex(p, syms).0).collect::<Vec<_>>().join(", ")), P_ATOM),
        Expr::Neg { operand, .. } => {
            let r = render(operand, syms);
            // `-(a b)` keeps its parentheses: the code negates the product
            let s = if r.1 > P_NEG && !r.0.starts_with('-') { r.0 } else { paren(r.0) };
            (format!("-{s}"), P_NEG)
        }
        Expr::Not { operand, .. } => (format!("\\lnot {}", at(render(operand, syms), P_ADD)), P_NOT),
        Expr::Binary { op, lhs, rhs, .. } => binary(*op, lhs, rhs, syms),
        Expr::Call { callee, args, .. } => call(callee, args, syms),
        // a field the literal left out, filled from its default (crate::ctor,
        // which gives it the type's span), is not printed: the literal as written
        Expr::RecordLit { type_name, type_span, fields, .. } => (
            format!("\\mathrm{{{}}}\\{{{}\\}}", esc_name(type_name), fields.iter().filter(|(_, s, _)| s != type_span).map(|(n, _, v)| format!("\\mathrm{{{}}}: {}", esc_name(n), expr(v, syms))).collect::<Vec<_>>().join(", ")),
            P_ATOM,
        ),
        Expr::Field { base, field, .. } => (format!("{}.\\mathrm{{{}}}", at(render(base, syms), P_ATOM), esc_name(field)), P_ATOM),
    }
}

fn binary(op: BinOp, lhs: &Expr, rhs: &Expr, syms: &Symbols) -> (String, u8) {
    let l = render(lhs, syms);
    let r = render(rhs, syms);
    match op {
        BinOp::Div => {
            // a fraction groups its parts: no parentheses inside
            (format!("\\frac{{{}}}{{{}}}", l.0, r.0), P_POST)
        }
        BinOp::Add | BinOp::Sub => {
            let ls = at(l, P_ADD);
            // `a + (-b)` and `a - (b - c)` keep their parentheses
            let rs = if r.1 < P_ADD || (op == BinOp::Sub && r.1 == P_ADD) || r.0.starts_with('-') { paren(r.0) } else { r.0 };
            (format!("{ls} {} {rs}", if op == BinOp::Add { "+" } else { "-" }), P_ADD)
        }
        BinOp::Mul | BinOp::Matmul => {
            let ls = at(l, P_MUL);
            let rs = if r.1 < P_MUL || r.1 == P_NEG || r.0.starts_with('-') { paren(r.0) } else { r.0 };
            // a number after a factor would run into it: 2 \cdot 3, x \cdot 2
            let sep = if op == BinOp::Mul && rs.starts_with(|c: char| c.is_ascii_digit()) { " \\cdot " } else { "\\," };
            (format!("{ls}{sep}{rs}"), P_MUL)
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::EqEq | BinOp::Ne => {
            let sym = match op {
                BinOp::Lt => "<",
                BinOp::Le => "\\le",
                BinOp::Gt => ">",
                BinOp::Ge => "\\ge",
                BinOp::EqEq => "=",
                _ => "\\ne",
            };
            (format!("{} {sym} {}", at(l, P_ADD), at(r, P_ADD)), P_CMP)
        }
        // floor division as the floor of the fraction, `%` as mod (Viv's
        // floor rule: the result has the sign of the divisor)
        BinOp::FloorDiv => (format!("\\left\\lfloor \\frac{{{}}}{{{}}} \\right\\rfloor", l.0, r.0), P_ATOM),
        BinOp::Mod => {
            let rs = if r.1 <= P_MUL || r.0.starts_with('-') { paren(r.0) } else { r.0 };
            (format!("{} \\bmod {rs}", at(l, P_MUL)), P_MUL)
        }
        BinOp::And => (format!("{} \\;\\text{{and}}\\; {}", at(l, P_AND), at(r, P_AND + 1)), P_AND),
        BinOp::Or => (format!("{} \\;\\text{{or}}\\; {}", at(l, P_OR), at(r, P_OR + 1)), P_OR),
    }
}

fn call(callee: &str, args: &[Expr], syms: &Symbols) -> (String, u8) {
    let callee = callee.strip_prefix(crate::contracts::EVERY_DEF).unwrap_or(callee);
    let a = |k: usize| render(&args[k], syms);
    let n = args.len();
    match (callee, n) {
        ("sqrt", 1) => (format!("\\sqrt{{{}}}", a(0).0), P_ATOM),
        ("abs" | "fabs", 1) => (format!("\\left|{}\\right|", a(0).0), P_ATOM),
        ("enorm", 2) => (format!("\\left\\lVert {} \\right\\rVert", a(1).0), P_ATOM),
        ("norm" | "enorm", 1) => (format!("\\left\\lVert {} \\right\\rVert", a(0).0), P_ATOM),
        ("min" | "max", _) if n >= 1 => (format!("\\{callee}\\left({}\\right)", args_tex(args, syms)), P_ATOM),
        ("transpose", 1) => (format!("{}^{{\\top}}", at(a(0), P_ATOM)), P_POST),
        ("pow", 2) => (format!("{}^{{{}}}", at(a(0), P_ATOM), a(1).0), P_POST),
        ("exp", 1) => (format!("e^{{{}}}", a(0).0), P_POST),
        ("log" | "ln", 1) => (format!("\\ln {}", at(a(0), P_ATOM)), P_MUL),
        ("log10", 1) => (format!("\\log_{{10}} {}", at(a(0), P_ATOM)), P_MUL),
        ("sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh", 1) => (format!("\\{callee} {}", at(a(0), P_ATOM)), P_MUL),
        ("asin" | "acos" | "atan", 1) => (format!("\\arc{} {}", &callee[1..], at(a(0), P_ATOM)), P_MUL),
        ("floor", 1) => (format!("\\left\\lfloor {} \\right\\rfloor", a(0).0), P_ATOM),
        ("ceil", 1) => (format!("\\left\\lceil {} \\right\\rceil", a(0).0), P_ATOM),
        ("mod", 2) => (format!("{} \\bmod {}", at(a(0), P_MUL), at(a(1), P_MUL + 1)), P_MUL),
        ("sum", 1) => (format!("\\textstyle\\sum {}", at(a(0), P_MUL)), P_MUL),
        ("zeros", _) if n >= 1 => (format!("\\mathbf{{0}}_{{{}}}", args.iter().map(|x| expr(x, syms)).collect::<Vec<_>>().join(" \\times ")), P_ATOM),
        // a quadrature of a `fn(x) -> e` (what `take the integral of e from a
        // to b` lowers to, crate::phrases): the integral it computes
        ("integrate" | "q_gauss" | "q_adaptive" | "q_romberg", 3) | ("simpson" | "trapezoid", 4) if lambda1(&args[0]).is_some() => {
            let (x, body) = lambda1(&args[0]).expect("a lambda");
            let d = name_tex(x, syms).0;
            (format!("\\int_{{{}}}^{{{}}} {}\\,d{d}", a(1).0, a(2).0, at(render(body, syms), P_ADD)), P_ADD)
        }
        // a root finder of a `fn(x) -> e`: the x in [a, b] (or near x0) with e = 0
        ("find_root" | "r_bisect" | "r_illinois" | "r_safe", 3) if lambda1(&args[0]).is_some() => {
            let (x, body) = lambda1(&args[0]).expect("a lambda");
            let v = name_tex(x, syms).0;
            (format!("{v} \\in \\left[{}, {}\\right] : {} = 0", a(1).0, a(2).0, render(body, syms).0), P_CMP)
        }
        ("r_newton", 3) if lambda1(&args[0]).is_some() => {
            let (x, body) = lambda1(&args[0]).expect("a lambda");
            let v = name_tex(x, syms).0;
            (format!("{v} : {} = 0,\\ {v}_{{0}} = {}", render(body, syms).0, a(2).0), P_CMP)
        }
        // the symbolic derivative (`... using the symbolic derivative`,
        // lib/cas): the expression is the CAS's text in the first argument
        ("cas_deriv_at", 5) if cas_text_arg(&args[0], &args[1]).is_some() => {
            let (body, x) = cas_text_arg(&args[0], &args[1]).expect("a CAS text");
            let v = name_tex(&x, syms).0;
            (format!("\\left.\\frac{{d}}{{d{v}}} {}\\right|_{{{v} = {}}}", at(render(&body, syms), P_ADD), a(2).0), P_ATOM)
        }
        ("deriv", 2) if lambda1(&args[0]).is_some() => {
            let (x, body) = lambda1(&args[0]).expect("a lambda");
            let v = name_tex(x, syms).0;
            (format!("\\left.\\frac{{d}}{{d{v}}} {}\\right|_{{{v} = {}}}", at(render(body, syms), P_ADD), a(1).0), P_ATOM)
        }
        // enums and optionals: a value is its variant's
        // name, upright, and its payload; a match's test reads `m is V`
        (c, _) if c == crate::parser::ENUM_IS && n == 2 => match &args[1] {
            Expr::Str { value, .. } => (format!("{} \\text{{ is }} \\mathrm{{{}}}", a(0).0, esc_name(value)), P_CMP),
            _ => (String::new(), P_ATOM),
        },
        (c, 3) if c == crate::parser::ENUM_FIELD => match (&args[1], &args[2]) {
            (Expr::Str { value, .. }, Expr::Num { value: k, .. }) => (format!("{}.\\mathrm{{{}}}_{{{}}}", at(a(0), P_POST), esc_name(value), num(*k)), P_POST),
            _ => (String::new(), P_ATOM),
        },
        ("None", 0) => ("\\mathrm{None}".to_string(), P_ATOM),
        (c, _) if c == "Some" || c.split_once('.').is_some_and(|(e, v)| !e.is_empty() && !v.is_empty() && e.starts_with(|ch: char| ch.is_uppercase())) => {
            let v = c.rsplit('.').next().unwrap_or(c);
            if n == 0 {
                (format!("\\mathrm{{{}}}", esc_name(v)), P_ATOM)
            } else {
                named_call(&format!("\\mathrm{{{}}}", esc_name(v)), args, syms)
            }
        }
        _ => {
            let head = match syms.get(callee) {
                Some(s) => at((s.clone(), if is_atomic(s) { P_ATOM } else { P_ADD }), P_ATOM),
                None => format!("\\operatorname{{{}}}", esc_name(callee)),
            };
            named_call(&head, args, syms)
        }
    }
}

/// The expression and the variable a `cas_deriv_at` call hands to lib/cas, as
/// Viv text in its first two arguments.
fn cas_text_arg(text: &Expr, var: &Expr) -> Option<(Expr, String)> {
    match (text, var) {
        (Expr::Str { value: t, .. }, Expr::Str { value: v, .. }) => crate::derivs::parse_expr(t).ok().map(|e| (e, v.clone())),
        _ => None,
    }
}

/// The parameter and body of a one-parameter lambda.
fn lambda1(e: &Expr) -> Option<(&str, &Expr)> {
    match e {
        Expr::Lambda { params, body, .. } if params.len() == 1 => Some((params[0].as_str(), body)),
        _ => None,
    }
}

/// `head(args)`, the first few arguments and `\ldots` when there are many.
fn named_call(head: &str, args: &[Expr], syms: &Symbols) -> (String, u8) {
    let shown = if args.len() > MAX_ARGS {
        format!("{}, \\ldots", args[..SHOWN_ARGS].iter().map(|x| expr(x, syms)).collect::<Vec<_>>().join(", "))
    } else {
        args_tex(args, syms)
    };
    (format!("{head}\\left({shown}\\right)"), P_ATOM)
}

/// LaTeX with the sizing of its delimiters and its spaces taken out, so two
/// pieces that typeset as the same symbols compare equal: `\lVert f \rVert`
/// and `\left\lVert f \right\rVert`.
fn typeset_key(tex: &str) -> String {
    tex.replace("\\left", "").replace("\\right", "").chars().filter(|c| !c.is_whitespace()).collect()
}

/// The right side of an assignment to `lhs`. When it would print as the left
/// side does (`fnorm = enorm(m, fvec)` with the glossary's `fnorm` being
/// `\lVert f \rVert` and `fvec` being `f`), the line would read as `x = x`;
/// then a call on the right is printed by its name, `\operatorname{enorm}(m, f)`,
/// which says how the value is computed, and any other right side leaves the
/// line out (None).
fn rhs_of(lhs: &str, value: &Expr, syms: &Symbols) -> Option<String> {
    let rhs = expr(value, syms);
    if typeset_key(&rhs) != typeset_key(lhs) {
        return Some(rhs);
    }
    match value {
        Expr::Call { callee, args, .. } => Some(named_call(&format!("\\operatorname{{{}}}", esc_name(callee)), args, syms).0),
        _ => None,
    }
}

/// What a statement is, as the explorer's math shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct StmtTex {
    /// `assign`, `if`, `while`, `for`, `return` or `call`
    pub kind: &'static str,
    pub tex: String,
}

/// The LaTeX of one statement, or None for what has no math (a step marker,
/// a test, `break`). An assignment that reads its own target is written
/// `x \gets ...`, so `iter = iter + 1` does not read as an equation.
pub fn stmt(s: &Stmt, syms: &Symbols) -> Option<StmtTex> {
    match s {
        Stmt::Assign { name, value, .. } => {
            let lhs = name_tex(name, syms).0;
            let rel = if contains_var(value, name) { "\\gets" } else { "=" };
            let rhs = rhs_of(&lhs, value, syms)?;
            Some(StmtTex { kind: "assign", tex: format!("{lhs} {rel} {rhs}") })
        }
        Stmt::SetIndex { base, indices, value, .. } => {
            let b = render(base, syms);
            let idx: Vec<String> = indices.iter().map(|i| expr(i, syms)).collect();
            let lhs = subscript(b, idx);
            let target = if let Expr::Var { name, .. } = base.as_ref() { Some(name.as_str()) } else { None };
            let rel = if target.is_some_and(|t| contains_var(value, t)) { "\\gets" } else { "=" };
            let rhs = rhs_of(&lhs, value, syms)?;
            Some(StmtTex { kind: "assign", tex: format!("{lhs} {rel} {rhs}") })
        }
        Stmt::SetField { base, field, value, .. } => {
            let lhs = format!("{}.\\mathrm{{{}}}", render(base, syms).0, field.replace('_', "\\_"));
            let rhs = rhs_of(&lhs, value, syms)?;
            Some(StmtTex { kind: "assign", tex: format!("{lhs} = {rhs}") })
        }
        Stmt::If { cond, .. } => Some(StmtTex { kind: "if", tex: expr(cond, syms) }),
        Stmt::While { cond, .. } => {
            if matches!(cond, Expr::Bool { value: true, .. }) {
                None
            } else {
                Some(StmtTex { kind: "while", tex: expr(cond, syms) })
            }
        }
        Stmt::For { var, bound, .. } => {
            let v = name_tex(var, syms).0;
            let last = match bound {
                Expr::Num { value, .. } => num(value - 1.0),
                // `for i to j + 1` runs to j
                Expr::Binary { op: BinOp::Add, lhs, rhs, .. } if matches!(rhs.as_ref(), Expr::Num { value, .. } if *value == 1.0) => expr(lhs, syms),
                _ => format!("{} - 1", at(render(bound, syms), P_ADD)),
            };
            Some(StmtTex { kind: "for", tex: format!("{v} = 0, \\ldots, {last}") })
        }
        Stmt::Return { value, .. } => Some(StmtTex { kind: "return", tex: format!("\\text{{return }} {}", expr(value, syms)) }),
        // a type declaration and a match's coverage check have no math
        Stmt::ExprStmt { expr: Expr::Call { callee, .. }, .. } if callee == crate::parser::DECLARE || callee == crate::parser::MATCH_REST => None,
        Stmt::ExprStmt { expr: e @ Expr::Call { .. }, .. } => Some(StmtTex { kind: "call", tex: expr(e, syms) }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Symbols and math spans in the intent's text
// ---------------------------------------------------------------------------

/// A glossary text that starts with a symbol, `$\Delta$  the trust radius`:
/// the symbol and the rest. Without one, `(None, text)`.
pub fn split_symbol(text: &str) -> (Option<String>, String) {
    let t = text.trim_start();
    if let Some(rest) = t.strip_prefix('$') {
        if !rest.starts_with('$') {
            if let Some(k) = rest.find('$') {
                let sym = rest[..k].trim().to_string();
                if !sym.is_empty() {
                    return (Some(sym), rest[k + 1..].trim_start().to_string());
                }
            }
        }
    }
    (None, text.to_string())
}

/// The math in a text: every `$$ ... $$` and `$ ... $` span, without its
/// dollars, and whether every dollar was closed. `\$` is a dollar sign.
pub fn math_spans(text: &str) -> (Vec<String>, bool) {
    let t = text.replace("\\$", "");
    let mut out = Vec::new();
    let mut rest = t.as_str();
    loop {
        let Some(k) = rest.find('$') else { return (out, true) };
        let display = rest[k..].starts_with("$$");
        let open = if display { 2 } else { 1 };
        let after = &rest[k + open..];
        let close = if display { after.find("$$") } else { after.find('$') };
        match close {
            Some(e) => {
                out.push(after[..e].to_string());
                rest = &after[e + open..];
            }
            None => return (out, false),
        }
    }
}

/// A text with its math spans replaced by a space: what is left is prose.
pub fn without_math(text: &str) -> String {
    let t = text.replace("\\$", "");
    let mut out = String::new();
    let mut rest = t.as_str();
    while let Some(k) = rest.find('$') {
        out.push_str(&rest[..k]);
        let display = rest[k..].starts_with("$$");
        let open = if display { 2 } else { 1 };
        let after = &rest[k + open..];
        let close = if display { after.find("$$") } else { after.find('$') };
        match close {
            Some(e) => {
                out.push(' ');
                rest = &after[e + open..];
            }
            None => {
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// Can KaTeX typeset it?
// ---------------------------------------------------------------------------

/// The commands this check accepts: the part of KaTeX an explanation of
/// numerical code uses. KaTeX supports more; a command outside this list is
/// reported so that it is looked at, and the site's build renders every
/// piece of math with KaTeX itself (www/scripts/explore-build.mjs), which is
/// the final word.
const COMMANDS: &[&str] = &[
    // Greek
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi",
    "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "ell", "hbar", "imath", "jmath",
    // fonts and text
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "boldsymbol", "bm", "text", "textrm", "textbf", "textit", "operatorname",
    "displaystyle", "textstyle", "scriptstyle",
    // structures
    "frac", "dfrac", "tfrac", "sqrt", "binom", "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "middle", "begin", "end", "overline", "underline", "overbrace", "underbrace", "hat", "widehat", "bar", "tilde", "widetilde", "dot", "ddot", "vec",
    "boxed", "phantom", "substack", "stackrel", "overset", "underset", "not",
    // delimiters
    "lVert", "rVert", "lvert", "rvert", "Vert", "vert", "langle", "rangle", "lfloor", "rfloor", "lceil", "rceil", "lbrace", "rbrace",
    // operators and relations
    "sum", "prod", "int", "lim", "min", "max", "sup", "inf", "arg", "argmin", "argmax", "det", "log", "ln", "exp", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin",
    "arccos", "arctan", "sinh", "cosh", "tanh", "bmod", "pmod", "mod", "le", "ge", "leq", "geq", "ne", "neq", "approx", "sim", "simeq", "equiv",
    "propto", "ll", "gg", "prec", "preceq", "succ", "succeq", "in", "notin", "subset", "subseteq", "cup", "cap", "setminus", "to", "gets", "mapsto", "leftarrow", "rightarrow",
    "Leftarrow", "Rightarrow", "leftrightarrow", "Leftrightarrow", "implies", "iff", "uparrow", "downarrow", "cdot", "times", "div", "pm", "mp",
    "ast", "star", "circ", "bullet", "oplus", "otimes", "perp", "parallel", "mid", "nabla", "partial", "infty", "forall", "exists", "neg", "lnot",
    "land", "lor", "wedge", "vee", "top", "bot", "intercal", "dagger", "prime", "emptyset", "varnothing",
    // dots and spacing
    "ldots", "cdots", "vdots", "ddots", "dots", "quad", "qquad", "colon",
];

/// Whether `tex` is LaTeX KaTeX can typeset: braces balance, `\left` and
/// `\right` pair up, every `\begin{env}` has its `\end{env}`, each command is
/// one this check knows, and nothing KaTeX would read as a math delimiter or
/// a comment (`$`, `%`) is inside. Err names the first problem.
pub fn check(tex: &str) -> Result<(), String> {
    let b = tex.as_bytes();
    let mut depth = 0i32;
    let mut lr = 0i32;
    let mut envs: Vec<String> = Vec::new();
    let mut i = 0;
    if tex.trim().is_empty() {
        return Err("empty math".into());
    }
    while i < b.len() {
        let c = b[i] as char;
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return Err("a `}` with no `{` before it".into());
                }
            }
            '$' => return Err("a `$` inside math".into()),
            '%' => return Err("a `%` (a comment in LaTeX: it would hide the rest)".into()),
            '#' => return Err("a `#` (only macro definitions use it)".into()),
            '^' | '_' => {
                let next = tex[i + 1..].trim_start();
                if next.is_empty() || next.starts_with('}') || next.starts_with('^') || next.starts_with('_') {
                    return Err(format!("`{c}` with nothing after it"));
                }
            }
            '\\' => {
                let mut j = i + 1;
                while j < b.len() && (b[j] as char).is_ascii_alphabetic() {
                    j += 1;
                }
                if j == i + 1 {
                    // a one-character command: `\,` `\;` `\!` `\ ` `\{` `\}` `\|` `\\` `\_`
                    if i + 1 >= b.len() {
                        return Err("a `\\` at the end".into());
                    }
                    let d = b[i + 1] as char;
                    if !",;:! {}|\\_&#$%".contains(d) {
                        return Err(format!("`\\{d}` is not a command KaTeX knows"));
                    }
                    i += 2;
                    continue;
                }
                let word = &tex[i + 1..j];
                if !COMMANDS.contains(&word) {
                    return Err(format!("`\\{word}` is not a command this check knows (add it to COMMANDS in lang/viv/src/tex.rs if KaTeX has it)"));
                }
                match word {
                    "left" => lr += 1,
                    "right" => {
                        lr -= 1;
                        if lr < 0 {
                            return Err("a `\\right` with no `\\left` before it".into());
                        }
                    }
                    "begin" | "end" => {
                        let rest = &tex[j..];
                        let Some(inner) = rest.strip_prefix('{').and_then(|r| r.split_once('}')).map(|(e, _)| e.to_string()) else {
                            return Err(format!("`\\{word}` without an environment name"));
                        };
                        if !matches!(inner.as_str(), "cases" | "matrix" | "pmatrix" | "bmatrix" | "vmatrix" | "aligned" | "array" | "gathered" | "smallmatrix") {
                            return Err(format!("the environment `{inner}` is not one this check knows"));
                        }
                        if word == "begin" {
                            envs.push(inner);
                        } else if envs.pop().as_deref() != Some(inner.as_str()) {
                            return Err(format!("`\\end{{{inner}}}` does not close the environment that is open"));
                        }
                    }
                    _ => {}
                }
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return Err("the braces do not balance".into());
    }
    if lr != 0 {
        return Err("a `\\left` with no `\\right`".into());
    }
    if let Some(e) = envs.pop() {
        return Err(format!("`\\begin{{{e}}}` is never closed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Item;
    use knotkit::span::SourceId;

    /// The LaTeX of each statement in the body of `def t()`.
    fn lines(body: &str, syms: &[(&str, &str)]) -> Vec<String> {
        let src = format!("def t(a, b, c, d, x, y, n, A: Arr, v: Arr) {{\n{body}\n}}\n");
        let prog = crate::parse(&src, SourceId(0)).expect("the test program parses");
        let syms: Symbols = syms.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let Some(Item::Def(d)) = prog.items.iter().find(|i| matches!(i, Item::Def(_))) else { panic!("no def") };
        fn go(ss: &[Stmt], syms: &Symbols, out: &mut Vec<String>) {
            for s in ss {
                if let Some(t) = stmt(s, syms) {
                    out.push(t.tex);
                }
                match s {
                    Stmt::If { then_body, else_body, .. } => {
                        go(then_body, syms, out);
                        go(else_body, syms, out);
                    }
                    Stmt::While { body, .. } | Stmt::For { body, .. } => go(body, syms, out),
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        go(&d.body, &syms, &mut out);
        out
    }
    fn one(body: &str) -> String {
        lines(body, &[]).remove(0)
    }

    #[test]
    fn enums_and_matches_print_as_their_variants() {
        // the program's enum is declared outside the def `lines` writes
        let src = "enum E {\n    Leaf(Int)\n    Pair(E, E)\n    Empty\n}\ndef t(e: E) {\n    x = E.Pair(E.Leaf(1), E.Empty)\n    o = Some(2)\n    match e {\n        Leaf(k) => print(k)\n        _ => print(0)\n    }\n}\n";
        let prog = crate::parse(src, SourceId(0)).expect("parses");
        let Some(Item::Def(d)) = prog.items.iter().find(|i| matches!(i, Item::Def(_))) else { panic!("no def") };
        let syms = Symbols::new();
        let out: Vec<String> = d.body.iter().filter_map(|s| stmt(s, &syms)).map(|t| t.tex).collect();
        assert_eq!(out[0], "x = \\mathrm{Pair}\\left(\\mathrm{Leaf}\\left(1\\right), \\mathrm{Empty}\\right)");
        assert_eq!(out[1], "o = \\mathrm{Some}\\left(2\\right)");
        assert!(out[3].contains("\\text{ is } \\mathrm{Leaf}"), "{out:?}");
    }

    #[test]
    fn fractions_roots_and_scripts() {
        assert_eq!(one("r = a / b"), "r = \\frac{a}{b}");
        assert_eq!(one("r = sqrt(a + b)"), "r = \\sqrt{a + b}");
        assert_eq!(one("r = pow(a, 2.0)"), "r = a^{2}");
        assert_eq!(one("r = pow(a + b, c)"), "r = \\left(a + b\\right)^{c}");
        assert_eq!(one("r = pow(a / b, c)"), "r = \\left(\\frac{a}{b}\\right)^{c}");
        assert_eq!(one("r = abs(a - b)"), "r = \\left|a - b\\right|");
        assert_eq!(one("r = v[n]"), "r = v_{n}");
        assert_eq!(one("r = A[a, b]"), "r = A_{ab}");
        assert_eq!(one("r = A[a + 1.0, b]"), "r = A_{a + 1,\\,b}");
        assert_eq!(one("r = transpose(A)"), "r = A^{\\top}");
        assert_eq!(one("r = A @ transpose(A)"), "r = A\\,A^{\\top}");
        assert_eq!(one("r = enorm(n, v)"), "r = \\left\\lVert v \\right\\rVert");
        assert_eq!(one("r = min(a, b * c)"), "r = \\min\\left(a, b\\,c\\right)");
    }

    #[test]
    fn phrases_show_the_math_they_compute() {
        // a phrase lowers to a call of a quadrature, a root finder or `deriv`
        // with a lambda (crate::phrases); the math view shows the integral,
        // the equation and the derivative, not the call
        assert_eq!(one("r = take the integral of u^2 from a to b"), "r = \\int_{a}^{b} u^{2}\\,du");
        assert_eq!(one("r = integrate exp(-a * u) from 0 to 1 with 60 intervals"), "r = \\int_{0}^{1} e^{-a\\,u}\\,du");
        assert_eq!(one("r = solve cos(u) = u between 0 and 1 using bisection"), "r = u \\in \\left[0, 1\\right] : \\cos u - u = 0");
        assert_eq!(one("r = take the root of u^2 - a starting at 1 using newton"), "r = u : u^{2} - a = 0,\\ u_{0} = 1");
        assert_eq!(one("r = take the derivative of sin(u) * u at x"), "r = \\left.\\frac{d}{du} \\sin u\\,u\\right|_{u = x}");
    }

    #[test]
    fn precedence_and_parentheses() {
        // only where the operators need them
        assert_eq!(one("r = (a + b) * c"), "r = \\left(a + b\\right)\\,c");
        assert_eq!(one("r = a + b * c"), "r = a + b\\,c");
        assert_eq!(one("r = a - (b - c)"), "r = a - \\left(b - c\\right)");
        assert_eq!(one("r = a - (b + c)"), "r = a - \\left(b + c\\right)");
        assert_eq!(one("r = (a - b) - c"), "r = a - b - c");
        assert_eq!(one("r = a + (b - c)"), "r = a + b - c");
        assert_eq!(one("r = a * (b * c)"), "r = a\\,b\\,c");
        assert_eq!(one("r = a / (b + c)"), "r = \\frac{a}{b + c}");
        assert_eq!(one("r = (a + b) / c * d"), "r = \\frac{a + b}{c}\\,d");
        assert_eq!(one("r = -(a * b)"), "r = -\\left(a\\,b\\right)");
        assert_eq!(one("r = -a"), "r = -a");
        assert_eq!(one("r = a * -b"), "r = a\\,\\left(-b\\right)");
        assert_eq!(one("r = a - -b"), "r = a - \\left(-b\\right)");
        assert_eq!(one("r = 2.0 * a"), "r = 2\\,a");
        assert_eq!(one("r = a * 2.0"), "r = a \\cdot 2");
        assert_eq!(one("r = 0.5 * a / (b + 0.5 * c)"), "r = \\frac{0.5\\,a}{b + 0.5\\,c}");
    }

    #[test]
    fn floor_division_and_mod() {
        assert_eq!(one("r = a // b"), "r = \\left\\lfloor \\frac{a}{b} \\right\\rfloor");
        assert_eq!(one("r = (a + b) // 2.0"), "r = \\left\\lfloor \\frac{a + b}{2} \\right\\rfloor");
        assert_eq!(one("r = a % b"), "r = a \\bmod b");
        assert_eq!(one("r = (a + b) % (c * d)"), "r = \\left(a + b\\right) \\bmod \\left(c\\,d\\right)");
        assert_eq!(one("r = a % b + 1.0"), "r = a \\bmod b + 1");
        for t in [one("r = a // b"), one("r = (a + b) % (c * d)")] {
            check(&t).unwrap();
        }
    }

    #[test]
    fn conditions() {
        assert_eq!(one("if a <= b {\n}"), "a \\le b");
        assert_eq!(one("if a != 0.0 and b >= c {\n}"), "a \\ne 0 \\;\\text{and}\\; b \\ge c");
        assert_eq!(one("if a < b or c > d and x == y {\n}"), "a < b \\;\\text{or}\\; c > d \\;\\text{and}\\; x = y");
        assert_eq!(one("if (a < b or c > d) and x == y {\n}"), "\\left(a < b \\;\\text{or}\\; c > d\\right) \\;\\text{and}\\; x = y");
        assert_eq!(one("while not (a < b) {\n}"), "\\lnot \\left(a < b\\right)");
        assert_eq!(one("for j to n {\n}"), "j = 0, \\ldots, n - 1");
        assert_eq!(one("for j to n - a {\n}"), "j = 0, \\ldots, n - a - 1");
        assert_eq!(one("for j to a + 1.0 {\n}"), "j = 0, \\ldots, a");
        assert!(lines("while true {\n}", &[]).is_empty());
    }

    #[test]
    fn symbols_from_the_glossary() {
        let syms = [("delta", "\\Delta"), ("pnorm", "\\lVert D p \\rVert"), ("qtf", "Q^\\top f"), ("par", "\\lambda"), ("x1", "x_1")];
        let l = lines("delta = 0.5 * min(delta, pnorm / 0.1)\npar = par / a\nb = qtf[n]\nc = x1[n]\nd = pnorm * pnorm\niters = 1.0", &syms);
        assert_eq!(l[0], "\\Delta \\gets 0.5\\,\\min\\left(\\Delta, \\frac{\\lVert D p \\rVert}{0.1}\\right)");
        assert_eq!(l[1], "\\lambda \\gets \\frac{\\lambda}{a}");
        // a symbol with parts is parenthesized before a subscript
        assert_eq!(l[2], "b = \\left(Q^\\top f\\right)_{n}");
        // a symbol with its own subscript is braced
        assert_eq!(l[3], "c = {x_1}_{n}");
        assert_eq!(l[4], "d = \\lVert D p \\rVert\\,\\lVert D p \\rVert");
        assert_eq!(l[5], "\\mathrm{iters} = 1");
    }

    #[test]
    fn a_line_whose_sides_print_the_same_names_its_call() {
        // lmder: fnorm = enorm(m, fvec), with fnorm's symbol the norm of fvec's
        let syms = [("fnorm", "\\lVert f \\rVert"), ("fvec", "f"), ("xnorm", "\\lVert D x \\rVert"), ("g", "f")];
        let l = lines("fnorm = enorm(n, fvec)\nxnorm = enorm(n, v)\ng = fvec\nfnorm = fnorm", &syms);
        assert_eq!(l[0], "\\lVert f \\rVert = \\operatorname{enorm}\\left(n, f\\right)");
        // sides that differ print as before
        assert_eq!(l[1], "\\lVert D x \\rVert = \\left\\lVert v \\right\\rVert");
        // a right side that is no call, printing as the left side does, is left out
        assert_eq!(l.len(), 2, "{l:?}");
        for t in &l {
            check(t).unwrap();
        }
        assert_eq!(lines("v[n] = enorm(n, v)", &[("v", "\\lVert v \\rVert")]).len(), 1);
    }

    #[test]
    fn numbers_and_calls() {
        assert_eq!(num(0.0001), "10^{-4}");
        assert_eq!(num(1.5e-8), "1.5 \\times 10^{-8}");
        assert_eq!(num(0.25), "0.25");
        assert_eq!(num(100.0), "100");
        assert_eq!(one("r = lmpar(a, b, c, d, x, y, n)"), "r = \\operatorname{lmpar}\\left(a, b, c, \\ldots\\right)");
        assert_eq!(one("r = f(a, b)"), "r = \\operatorname{f}\\left(a, b\\right)");
        assert_eq!(one("v[n] = -v[n]"), "v_{n} \\gets -v_{n}");
        assert_eq!(one("a = a + 1.0"), "a \\gets a + 1");
        assert_eq!(one("return a"), "\\text{return } a");
    }

    #[test]
    fn every_printed_line_passes_the_katex_check() {
        let l = lines(
            "r = (a + b) / c * sqrt(d)\nr = abs(A[a, b] - v[n]) / max(a, b)\nr = enorm(n, v) / pow(x, 2.0)\nif a <= 0.25 or not (b > c) {\n}\nfor j to n - 1.0 {\n}\nr = mod(a, b)\nr = [[a, b], [c, d]]\nr = exp(-a) * log(b)",
            &[],
        );
        for t in &l {
            assert!(check(t).is_ok(), "{t}: {:?}", check(t));
        }
    }

    #[test]
    fn the_katex_check_finds_what_would_not_typeset() {
        assert!(check("\\frac{a}{b}").is_ok());
        assert!(check("\\left\\lVert D p \\right\\rVert \\le \\Delta").is_ok());
        assert!(check("\\begin{cases} a & b \\\\ c & d \\end{cases}").is_ok());
        assert!(check("\\frac{a}{b").is_err());
        assert!(check("a}").is_err());
        assert!(check("\\left( a").is_err());
        assert!(check("\\fracc{a}{b}").is_err());
        assert!(check("x^").is_err());
        assert!(check("50% of it").is_err());
        assert!(check("\\begin{cases} a \\end{pmatrix}").is_err());
        assert!(check("").is_err());
    }

    #[test]
    fn symbols_and_math_spans_in_text() {
        assert_eq!(split_symbol("$\\Delta$  the trust radius"), (Some("\\Delta".to_string()), "the trust radius".to_string()));
        assert_eq!(split_symbol("the trust radius"), (None, "the trust radius".to_string()));
        assert_eq!(split_symbol("$$x$$ display"), (None, "$$x$$ display".to_string()));
        let (spans, closed) = math_spans("with $\\rho = 3/4$ and $$a + b$$ and a \\$5 bill");
        assert_eq!(spans, vec!["\\rho = 3/4".to_string(), "a + b".to_string()]);
        assert!(closed);
        assert!(!math_spans("an open $x").1);
        assert_eq!(without_math("a $x = 1$ b").split_whitespace().collect::<Vec<_>>(), vec!["a", "b"]);
        assert!(is_atomic("\\Delta"));
        assert!(is_atomic("x_{1}"));
        assert!(is_atomic("\\lVert D p \\rVert"));
        assert!(is_atomic("\\mathrm{ared}"));
        assert!(!is_atomic("Q^\\top f"));
        assert!(!is_atomic("a + b"));
        assert!(!is_atomic("\\frac{a}{b}"));
    }
}
