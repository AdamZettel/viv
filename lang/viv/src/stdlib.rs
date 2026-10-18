// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Small builtins Python programs reach for, done in the front end so every
//! engine runs the same thing (the Rosetta study, branch fix-w1):
//!
//! - `pi`, `e`, `tau` and `inf`, Python's math constants, as numbers wherever
//!   the program does not bind the name itself;
//! - `min(a, b, c, ...)` and `max(...)` of three or more values, as nested
//!   two-argument calls;
//! - `reversed(x)` as the slice `x[::-1]` (an array, a string, a list);
//! - `y = pop(xs)`, `y = pop(xs, i)` and `pop(xs, i)` as statements, by way
//!   of `viv__del`, the call `del xs[i]` is written as (crate::parser);
//! - the defs of lang/viv/std/builtins.viv a program calls and does not
//!   define (`fsum`, `hypot`, `copysign`, and the number forms of `sorted`,
//!   `unique` and `del`, which the lowering and the interpreter pick by the
//!   argument's type), added to the program.
//!
//! A program that uses none of these comes back unchanged.

use crate::ast::{Def, Expr, Item, Program, Stmt};
use knotkit::diag::Diagnostic;
use knotkit::span::{SourceId, Span};
use std::collections::HashSet;

thread_local! {
    /// Names bound outside the text being parsed: a notebook cell's session
    /// (its earlier cells' globals and defs), which the constants must not
    /// shadow.
    static KNOWN: std::cell::RefCell<HashSet<String>> = std::cell::RefCell::new(HashSet::new());
}

/// Run `f` (a parse) with `names` bound outside its text.
pub fn with_known<T>(names: HashSet<String>, f: impl FnOnce() -> T) -> T {
    KNOWN.with(|k| *k.borrow_mut() = names);
    let out = f();
    KNOWN.with(|k| k.borrow_mut().clear());
    out
}

/// The callee `del xs[i]` is written as: `xs = viv__del(xs, i)`.
pub const DEL: &str = "viv__del";

/// The constants, and their values.
pub const CONSTANTS: [(&str, f64); 4] = [("pi", std::f64::consts::PI), ("e", std::f64::consts::E), ("tau", std::f64::consts::TAU), ("inf", f64::INFINITY)];

/// The defs of builtins.viv, by the names that bring each in.
const PROVIDES: &[(&str, &[&str])] = &[
    ("fsum", &["fsum"]),
    ("hypot", &["hypot"]),
    ("copysign", &["copysign"]),
    ("sorted", &["viv__sorted_num"]),
    ("unique", &["viv__unique_num"]),
    (DEL, &["viv__del_num", "viv__del_strs"]),
];

/// The text of lang/viv/std/builtins.viv.
pub const TEXT: &str = include_str!("../std/builtins.viv");

/// The number form a builtin's call takes for a real array, if it has one.
pub fn number_form(callee: &str) -> Option<&'static str> {
    match callee {
        "sorted" => Some("viv__sorted_num"),
        "unique" => Some("viv__unique_num"),
        DEL => Some("viv__del_num"),
        _ => None,
    }
}

fn each_stmt(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in body {
        f(s);
        match s {
            Stmt::If { then_body, else_body, .. } => {
                each_stmt(then_body, f);
                each_stmt(else_body, f);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => each_stmt(body, f),
            _ => {}
        }
    }
}

fn exprs_mut(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => crate::phrases::walk_mut(value, f),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => crate::phrases::walk_mut(expr, f),
        Stmt::SetIndex { base, indices, value, .. } => {
            crate::phrases::walk_mut(base, f);
            indices.iter_mut().for_each(|e| crate::phrases::walk_mut(e, f));
            crate::phrases::walk_mut(value, f);
        }
        Stmt::SetField { base, value, .. } => {
            crate::phrases::walk_mut(base, f);
            crate::phrases::walk_mut(value, f);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            crate::phrases::walk_mut(cond, f);
            then_body.iter_mut().for_each(|s| exprs_mut(s, f));
            else_body.iter_mut().for_each(|s| exprs_mut(s, f));
        }
        Stmt::While { cond, body, .. } => {
            crate::phrases::walk_mut(cond, f);
            body.iter_mut().for_each(|s| exprs_mut(s, f));
        }
        Stmt::For { bound, body, .. } => {
            crate::phrases::walk_mut(bound, f);
            body.iter_mut().for_each(|s| exprs_mut(s, f));
        }
        Stmt::Test { body, .. } => body.iter_mut().for_each(|s| exprs_mut(s, f)),
        _ => {}
    }
}

/// `y = pop(xs[, i])` and `pop(xs[, i])` in a block, rewritten (in place,
/// recursively): the element read, then `xs = viv__del(xs, i)`.
fn pops(body: Vec<Stmt>, hidden: &mut usize) -> Result<Vec<Stmt>, Diagnostic> {
    let mut out = Vec::with_capacity(body.len());
    for s in body {
        let pop_of = |e: &Expr| -> Option<(String, Span, Option<Expr>, Span)> {
            let Expr::Call { callee, args, span } = e else { return None };
            if callee != "pop" {
                return None;
            }
            match args.as_slice() {
                [Expr::Var { name, span: ns }] => Some((name.clone(), *ns, None, *span)),
                [Expr::Var { name, span: ns }, i] => Some((name.clone(), *ns, Some(i.clone()), *span)),
                _ => None,
            }
        };
        let (target, call) = match &s {
            Stmt::Assign { name, value, .. } => (Some(name.clone()), pop_of(value)),
            Stmt::ExprStmt { expr, .. } => (None, pop_of(expr)),
            _ => (None, None),
        };
        if let Some((xs, xs_span, i, span)) = call {
            let i = i.unwrap_or(Expr::Num { value: -1.0, span });
            // the index once, before the list changes
            let iv = format!("viv__pop{hidden}");
            *hidden += 1;
            out.push(Stmt::Assign { name: iv.clone(), name_span: span, value: i, span });
            let at = Expr::Var { name: iv, span };
            let list = Expr::Var { name: xs.clone(), span: xs_span };
            if let Some(y) = target {
                let item = Expr::Index { base: Box::new(list.clone()), indices: vec![at.clone()], span };
                out.push(Stmt::Assign { name: y, name_span: span, value: item, span });
            }
            let rest = Expr::Call { callee: DEL.to_string(), args: vec![list, at], span };
            out.push(Stmt::Assign { name: xs, name_span: xs_span, value: rest, span });
            continue;
        }
        out.push(match s {
            Stmt::If { cond, then_body, else_body, span } => Stmt::If { cond, then_body: pops(then_body, hidden)?, else_body: pops(else_body, hidden)?, span },
            Stmt::While { cond, body, span } => Stmt::While { cond, body: pops(body, hidden)?, span },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For { var, var_span, bound, body: pops(body, hidden)?, parallel, span },
            Stmt::Test { name, body, span } => Stmt::Test { name, body: pops(body, hidden)?, span },
            other => other,
        });
    }
    Ok(out)
}

/// The program with the constants, the calls above and the std defs it
/// needs (see the module's doc).
pub fn transform(mut program: Program, source: SourceId) -> Result<Program, Vec<Diagnostic>> {
    let defs: HashSet<String> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    // every name the program binds: an assignment, a parameter, a loop
    // variable, a declaration, a def
    let mut bound: HashSet<String> = defs.clone();
    KNOWN.with(|k| bound.extend(k.borrow().iter().cloned()));
    let mut calls: HashSet<String> = HashSet::new();
    let mut params: Vec<String> = Vec::new();
    {
        let mut note = |s: &Stmt| match s {
            Stmt::Assign { name, .. } => {
                bound.insert(name.clone());
            }
            Stmt::For { var, .. } => {
                bound.insert(var.clone());
            }
            _ => {}
        };
        for item in &program.items {
            match item {
                Item::Def(d) => {
                    params.extend(d.params.iter().map(|p| p.name.clone()));
                    each_stmt(&d.body, &mut note);
                }
                Item::Stmt(s) => each_stmt(std::slice::from_ref(s), &mut note),
                _ => {}
            }
        }
    }
    bound.extend(params);
    // a lambda's parameters bind their names too
    let mut lambda_params = Vec::new();
    {
        let mut see = |e: &mut Expr| {
            if let Expr::Lambda { params, .. } = e {
                lambda_params.extend(params.iter().cloned());
            }
        };
        let mut items = std::mem::take(&mut program.items);
        for item in items.iter_mut() {
            match item {
                Item::Def(d) => d.body.iter_mut().for_each(|s| exprs_mut(s, &mut see)),
                Item::Stmt(s) => exprs_mut(s, &mut see),
                _ => {}
            }
        }
        program.items = items;
    }
    bound.extend(lambda_params);
    let builtin = |n: &str| !defs.contains(n);
    // pop statements first: they write the `viv__del` calls the defs follow
    if builtin("pop") {
        let mut hidden = 0;
        let mut items = Vec::with_capacity(program.items.len());
        for item in std::mem::take(&mut program.items) {
            let next = match item {
                Item::Def(mut d) => {
                    d.body = pops(d.body, &mut hidden).map_err(|e| vec![e])?;
                    Item::Def(d)
                }
                Item::Stmt(s) => {
                    let mut v = pops(vec![s], &mut hidden).map_err(|e| vec![e])?;
                    let last = v.pop().expect("one statement at least");
                    items.extend(v.into_iter().map(Item::Stmt));
                    Item::Stmt(last)
                }
                other => other,
            };
            items.push(next);
        }
        program.items = items;
    }
    let consts: Vec<(&str, f64)> = CONSTANTS.iter().copied().filter(|(n, _)| !bound.contains(*n)).collect();
    let mut rewrite = |e: &mut Expr| match e {
        Expr::Var { name, span } => {
            if let Some((_, v)) = consts.iter().find(|(n, _)| n == name) {
                *e = Expr::Num { value: *v, span: *span };
            }
        }
        Expr::Call { callee, args, span } => {
            // min(a, b, c) is min(min(a, b), c)
            if (callee == "min" || callee == "max") && args.len() >= 3 && builtin(callee) {
                let name = callee.clone();
                let mut it = std::mem::take(args).into_iter();
                let mut acc = it.next().expect("three or more");
                for a in it {
                    let sp = acc.span().merge(a.span());
                    acc = Expr::Call { callee: name.clone(), args: vec![acc, a], span: sp };
                }
                *e = acc;
            } else if callee == "reversed" && args.len() == 1 && builtin("reversed") {
                // reversed(x) is x[::-1]
                let sp = *span;
                let n = |v: f64| Expr::Num { value: v, span: sp };
                let spec = Expr::ArrLit { elems: vec![n(4.0), n(0.0), n(0.0), n(-1.0)], span: sp };
                *e = Expr::Call { callee: crate::interp::slices::GET.to_string(), args: vec![args.remove(0), spec], span: sp };
            } else {
                calls.insert(callee.clone());
            }
        }
        _ => {}
    };
    for item in program.items.iter_mut() {
        match item {
            Item::Def(d) => d.body.iter_mut().for_each(|s| exprs_mut(s, &mut rewrite)),
            Item::Stmt(s) => exprs_mut(s, &mut rewrite),
            _ => {}
        }
    }
    // the std defs
    let wanted: Vec<&str> = PROVIDES.iter().filter(|(n, _)| calls.contains(*n) && (*n == DEL || builtin(n))).flat_map(|(_, ds)| ds.iter().copied()).filter(|d| !defs.contains(*d)).collect();
    if wanted.is_empty() {
        return Ok(program);
    }
    let at = Span::synthetic(source);
    let toks: Vec<crate::lexer::Token> = crate::lexer::lex(TEXT, source)?.into_iter().map(|t| crate::lexer::Token { tok: t.tok, span: at }).collect();
    let std = crate::parser::parse(toks, source)?;
    for item in std.items {
        if let Item::Def(d) = item {
            if wanted.contains(&d.name.as_str()) {
                program.items.push(Item::Def(Def { ..d }));
            }
        }
    }
    Ok(program)
}

/// Where the spans of std/groupby.viv's tokens start: past the text of any
/// program (1 GiB).
const STD_SPANS: u32 = 1 << 30;

/// The text of lang/viv/std/groupby.viv: `factorize` and `factorize2`.
pub const GROUPBY: &str = include_str!("../std/groupby.viv");

/// The group-by defs of std/groupby.viv (`factorize`, `factorize2`,
/// `match_rows`) a program
/// calls and does not define, added to it. This runs before
/// `crate::tuples::transform`, since both return several values and `u, ids =
/// factorize(keys)` is unpacked by that pass. A program that calls neither
/// comes back unchanged.
pub fn provide_groupby(mut program: Program, source: SourceId) -> Result<Program, Vec<Diagnostic>> {
    const NAMES: [&str; 10] = ["factorize", "factorize2", "match_rows", "key_range", "dense_span", "factorize_dense", "dense_order", "group_sum", "group_sum_num", "group_sum2"];
    let defs: HashSet<String> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    let mut calls: HashSet<String> = HashSet::new();
    {
        let mut see = |e: &mut Expr| {
            if let Expr::Call { callee, .. } = e {
                if NAMES.contains(&callee.as_str()) {
                    calls.insert(callee.clone());
                }
            }
        };
        let mut items = std::mem::take(&mut program.items);
        for item in items.iter_mut() {
            match item {
                Item::Def(d) => d.body.iter_mut().for_each(|s| exprs_mut(s, &mut see)),
                Item::Stmt(s) => exprs_mut(s, &mut see),
                _ => {}
            }
        }
        program.items = items;
    }
    // the std defs a called one calls in turn (factorize2 calls factorize,
    // which calls key_range, ...)
    const DEPS: [(&str, &[&str]); 6] = [
        ("factorize2", &["factorize"]),
        ("factorize", &["key_range", "dense_span", "factorize_dense"]),
        ("group_sum", &["key_range", "dense_span", "dense_order"]),
        ("group_sum_num", &["key_range", "dense_span", "dense_order"]),
        ("group_sum2", &["key_range", "dense_span", "factorize2"]),
        ("match_rows", &[]),
    ];
    loop {
        let before = calls.len();
        for (name, deps) in DEPS {
            if calls.contains(name) {
                calls.extend(deps.iter().map(|d| d.to_string()));
            }
        }
        if calls.len() == before {
            break;
        }
    }
    calls.retain(|c| !defs.contains(c));
    if calls.is_empty() {
        return Ok(program);
    }
    // Each token keeps its own span, moved past any program's text: the
    // lowering decides which literals and calls are Ints by span, and the
    // interpreter computes them by the same spans, so the defs' nodes must
    // not share one, and the parser's table of integer literals
    // (`Program::whole`) is by span too. A location there renders at the program's end (SourceMap
    // clamps it).
    let toks: Vec<crate::lexer::Token> = crate::lexer::lex(GROUPBY, source)?
        .into_iter()
        .map(|t| crate::lexer::Token { tok: t.tok, span: Span { source, start: t.span.start + STD_SPANS, end: t.span.end + STD_SPANS } })
        .collect();
    let std = crate::parser::parse(toks, source)?;
    // the integer literals, which an Int position reads exactly by span
    program.whole.extend(std.whole);
    for item in std.items {
        if let Item::Def(d) = item {
            if calls.contains(&d.name) {
                program.items.push(Item::Def(d));
            }
        }
    }
    Ok(program)
}
