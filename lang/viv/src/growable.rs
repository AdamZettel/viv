// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Growable arrays, the static half (the idioms study, R17):
//! `a = append(a, x)` and `a = extend(a, b)`.
//!
//! `check` runs in `crate::parse`, so every engine refuses the same programs:
//! `append` and `extend` are written only as the whole right side of an
//! assignment to the variable they grow, and never in a `parallel for`.
//!
//! `prepare` runs at the start of `crate::lower::lower`, so the C and wasm
//! backends (which share its IR) see the same rewritten program; the
//! interpreter decides at run time from the reference count and needs none of
//! it. For each def, and for the top level, the variables grown by an append
//! are TRACKED, and the pass
//!
//! - marks each `a = append(a, x)` of a tracked `a` as one that may grow in
//!   place (a third argument, 1; 0 for a top-level variable a def can hand
//!   out, whose appends always copy);
//! - SEALS a tracked variable (`a = viv__seal(a)`) just before an assignment
//!   or a `return` whose value can carry its handle: `b = a`, `return a`, a
//!   record or a closure holding it, a call that may give it back;
//! - seals the value of every other assignment to a tracked variable, and a
//!   tracked parameter on entry, since that storage came from elsewhere.
//!
//! A sealed array is grown by copying, so after the pass the storage an
//! in-place append writes to or moves is held by the variable being rebound
//! and nothing else: no other handle exists to see the growth, or to be left
//! pointing at freed memory.
//!
//! A program with no append comes back as `None` and is lowered untouched.

use crate::ast::{Expr, Item, Program, Stmt};
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::HashSet;

/// The builtins that grow an array.
pub const GROW: [&str; 2] = ["append", "extend"];
/// The call the pass writes to seal a variable's handle.
pub const SEAL: &str = "viv__seal";

/// Builtins whose result never holds an array argument's storage: numbers,
/// fresh arrays, or nothing. A call of any other builtin is assumed to
/// give its argument back.
const FRESH: &[&str] = &[
    "len", "numel", "shape", "sum", "min", "max", "print", "fmt", "show", "plot", "scatter", "bars", "plot2d", "plot3d", "quiver", "wire",
    "verts", "histogram", "all", "any", "copy", "concat", "vcat", "hcat", crate::interp::slices::GET, crate::interp::slices::SET, "assert_near",
    "assert_true", "assert_gt", "assert_lt", "assert_allclose", "take", "slab", "setslab", "put", "text", "sort", "fill", "transpose", "where",
    "roll", "exp", "log", "ln", "sqrt", "sin", "cos", "tan", "abs", "fabs", "floor", "ceil", "round", "tanh", "sinh", "cosh", "__ad_push",
    "__ad_restore", "__ad_head", "__ad_tail", "append", "extend",
];

/// The string builtins: each gives a new string, a new
/// list or a number, never a list it was handed.
fn fresh_text(callee: &str) -> bool {
    use crate::interp::strings as s;
    s::is_builtin(callee) || [s::FSTR, s::IN, s::STR_OF, s::REPR_OF, s::FMT_OF, "str", "repr"].contains(&callee)
}

fn is_grow(callee: &str, defs: &HashSet<String>) -> bool {
    GROW.contains(&callee) && !defs.contains(callee)
}

fn def_names(program: &Program) -> HashSet<String> {
    program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect()
}

// ---- walking -----------------------------------------------------------------

/// Every expression a statement evaluates itself (not those of the blocks it holds).
fn own_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => vec![value],
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => vec![expr],
        Stmt::SetIndex { base, indices, value, .. } => {
            let mut v = vec![&**base];
            v.extend(indices.iter());
            v.push(value);
            v
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } => vec![cond],
        Stmt::For { bound, .. } => vec![bound],
        _ => vec![],
    }
}

/// The blocks a statement holds.
fn blocks(s: &Stmt) -> Vec<&Vec<Stmt>> {
    match s {
        Stmt::If { then_body, else_body, .. } => vec![then_body, else_body],
        Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => vec![body],
        _ => vec![],
    }
}

fn subexprs(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::ArrLit { elems, .. } => elems.iter().collect(),
        Expr::Index { base, indices, .. } => {
            let mut v = vec![&**base];
            v.extend(indices.iter());
            v
        }
        Expr::Lambda { body, .. } => vec![body],
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => vec![operand],
        Expr::Binary { lhs, rhs, .. } => vec![lhs, rhs],
        Expr::Call { args, .. } => args.iter().collect(),
        Expr::RecordLit { fields, .. } => fields.iter().map(|f| &f.2).collect(),
        Expr::Field { base, .. } => vec![base],
        _ => vec![],
    }
}

fn mentions(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::Var { name: n, .. } if n == name) || subexprs(e).into_iter().any(|x| mentions(x, name))
}

/// `a = append(a, x)`: the grown variable, when `s` is one.
fn grows<'s>(s: &'s Stmt, defs: &HashSet<String>) -> Option<&'s str> {
    let Stmt::Assign { name, value: Expr::Call { callee, args, .. }, .. } = s else { return None };
    if !is_grow(callee, defs) || args.len() < 2 {
        return None;
    }
    matches!(&args[0], Expr::Var { name: n, .. } if n == name).then_some(name.as_str())
}

// ---- the check -----------------------------------------------------------------

/// Refuse an `append` or `extend` written anywhere but `a = append(a, x)`,
/// and one in a `parallel for`.
pub fn check(program: &Program) -> Result<(), Vec<Diagnostic>> {
    let defs = def_names(program);
    let mut errs = Vec::new();
    for item in &program.items {
        match item {
            Item::Def(d) => check_block(&d.body, &defs, false, &mut errs),
            Item::Stmt(s) => check_block(std::slice::from_ref(s), &defs, false, &mut errs),
            _ => {}
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

fn check_block(stmts: &[Stmt], defs: &HashSet<String>, parallel: bool, errs: &mut Vec<Diagnostic>) {
    for s in stmts {
        // the one place a grow may be: the whole right side, rebinding its first argument
        let mut rest: Vec<&Expr> = own_exprs(s);
        if let Stmt::Assign { name, value: Expr::Call { callee, args, span }, .. } = s {
            if is_grow(callee, defs) {
                rest.clear();
                if args.len() != 2 {
                    let what = if callee == "append" { "the array, then the number or row that goes on its end" } else { "the two vectors" };
                    errs.push(Diagnostic::error(format!("`{callee}` takes 2 arguments: {what}")).with_primary(*span, format!("{} given", args.len())));
                } else if !matches!(&args[0], Expr::Var { name: n, .. } if n == name) {
                    errs.push(misplaced(callee, *span, Some(name)));
                } else if parallel {
                    errs.push(
                        Diagnostic::error(format!("`{callee}` cannot grow an array inside a `parallel for`"))
                            .with_primary(*span, "the iterations run at once, and in no fixed order")
                            .with_note("collect into a sized array, `out[i] = ...`, or append in a plain `for`"),
                    );
                }
                rest.extend(args.iter().skip(1));
            }
        }
        for e in rest {
            check_expr(e, defs, errs);
        }
        let par = parallel || matches!(s, Stmt::For { parallel: true, .. });
        for b in blocks(s) {
            check_block(b, defs, par, errs);
        }
    }
}

fn check_expr(e: &Expr, defs: &HashSet<String>, errs: &mut Vec<Diagnostic>) {
    if let Expr::Call { callee, span, .. } = e {
        if is_grow(callee, defs) {
            errs.push(misplaced(callee, *span, None));
        }
    }
    for x in subexprs(e) {
        check_expr(x, defs, errs);
    }
}

fn misplaced(callee: &str, span: Span, target: Option<&str>) -> Diagnostic {
    let label = match target {
        Some(t) => format!("this assigns `{t}`, which is not the array it grows"),
        None => "not the right side of an assignment to the array it grows".to_string(),
    };
    Diagnostic::error(format!("`{callee}` gives back the longer array, so it is written `a = {callee}(a, x)`, with the same name on both sides"))
        .with_primary(span, label)
        .with_note("an alias taken before the append (`b = a`) keeps the array as it was; `copy(a)` gives a separate array to grow")
}

// ---- the rewrite ---------------------------------------------------------------

/// What the pass needs from type inference: whether a def returns a number,
/// a boolean or a complex number, so that its result cannot hold an array.
pub type Scalar<'a> = &'a dyn Fn(&str) -> bool;

struct Cx<'a> {
    defs: &'a HashSet<String>,
    scalar: Scalar<'a>,
    /// the tracked variables of the body being rewritten
    tracked: HashSet<String>,
    /// the tracked variables whose appends always copy
    copy_only: HashSet<String>,
}

/// Whether the value of `e` can hold `name`'s storage after the statement.
fn carries(e: &Expr, name: &str, cx: &Cx) -> bool {
    match e {
        Expr::Var { name: n, .. } => n == name,
        // numbers, and arithmetic, which makes a new array
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => false,
        Expr::Index { .. } | Expr::Neg { .. } | Expr::Not { .. } | Expr::Binary { .. } => false,
        // a closure keeps what it mentions
        Expr::Lambda { body, .. } => mentions(body, name),
        Expr::Call { callee, args, .. } => {
            if cx.defs.contains(callee) {
                !(cx.scalar)(callee) && args.iter().any(|a| mentions(a, name))
            } else if FRESH.contains(&callee.as_str()) || fresh_text(callee) || crate::interp::arrays::is_name(callee) {
                false
            } else {
                args.iter().any(|a| mentions(a, name))
            }
        }
        Expr::ArrLit { elems, .. } => elems.iter().any(|x| carries(x, name, cx)),
        Expr::RecordLit { fields, .. } => fields.iter().any(|f| carries(&f.2, name, cx)),
        Expr::Field { base, .. } => carries(base, name, cx),
    }
}

fn seal(name: &str, value: Expr, span: Span) -> Stmt {
    let value = Expr::Call { callee: SEAL.to_string(), args: vec![value], span };
    Stmt::Assign { name: name.to_string(), name_span: span, value, span }
}

fn tracked_in(stmts: &[Stmt], defs: &HashSet<String>, out: &mut HashSet<String>) {
    for s in stmts {
        if let Some(n) = grows(s, defs) {
            out.insert(n.to_string());
        }
        for b in blocks(s) {
            tracked_in(b, defs, out);
        }
    }
}

fn rewrite_block(stmts: &[Stmt], cx: &Cx) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        let sp = s.span();
        // seal every tracked variable this statement's stored value can carry
        let stored = match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => Some(value),
            _ => None,
        };
        let grown = grows(s, cx.defs);
        if let Some(v) = stored {
            let mut names: Vec<&String> = cx.tracked.iter().collect();
            names.sort();
            for t in names {
                if grown.is_none() && carries(v, t, cx) {
                    out.push(seal(t, Expr::Var { name: t.clone(), span: sp }, sp));
                }
            }
        }
        out.push(match s {
            Stmt::Assign { name, name_span, value, span } if cx.tracked.contains(name) => {
                let value = match (grown, value) {
                    (Some(_), Expr::Call { callee, args, span: cs }) => {
                        let own = if cx.copy_only.contains(name) { 0.0 } else { 1.0 };
                        let mut args = args.clone();
                        args.push(Expr::Num { value: own, span: *cs });
                        Expr::Call { callee: callee.clone(), args, span: *cs }
                    }
                    // storage from elsewhere: sealed, so its first append copies
                    _ => Expr::Call { callee: SEAL.to_string(), args: vec![value.clone()], span: value.span() },
                };
                Stmt::Assign { name: name.clone(), name_span: *name_span, value, span: *span }
            }
            Stmt::If { cond, then_body, else_body, span } => Stmt::If {
                cond: cond.clone(),
                then_body: rewrite_block(then_body, cx),
                else_body: rewrite_block(else_body, cx),
                span: *span,
            },
            Stmt::While { cond, body, span } => Stmt::While { cond: cond.clone(), body: rewrite_block(body, cx), span: *span },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For {
                var: var.clone(),
                var_span: *var_span,
                bound: bound.clone(),
                body: rewrite_block(body, cx),
                parallel: *parallel,
                span: *span,
            },
            Stmt::Test { name, body, span } => Stmt::Test { name: name.clone(), body: rewrite_block(body, cx), span: *span },
            other => other.clone(),
        });
    }
    out
}

/// The names a def binds: its parameters and every variable it assigns.
fn locals_of(params: &[String], body: &[Stmt], out: &mut HashSet<String>) {
    out.extend(params.iter().cloned());
    fn walk(stmts: &[Stmt], out: &mut HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::Assign { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::For { var, .. } => {
                    out.insert(var.clone());
                }
                _ => {}
            }
            for b in blocks(s) {
                walk(b, out);
            }
        }
    }
    walk(body, out);
}

/// Whether a stored value in `stmts` can carry the global `name`.
fn hands_out(stmts: &[Stmt], name: &str, cx: &Cx) -> bool {
    stmts.iter().any(|s| {
        let stored = match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => carries(value, name, cx),
            _ => false,
        };
        stored || blocks(s).into_iter().any(|b| hands_out(b, name, cx))
    })
}

/// The program with its appends marked and its tracked variables sealed, or
/// `None` when it grows no array.
pub fn prepare(program: &Program, scalar: Scalar) -> Option<Program> {
    let defs = def_names(program);
    let mut any = false;
    let mut main_tracked = HashSet::new();
    for item in &program.items {
        match item {
            Item::Def(d) => {
                let mut t = HashSet::new();
                tracked_in(&d.body, &defs, &mut t);
                any |= !t.is_empty();
            }
            Item::Stmt(s) => tracked_in(std::slice::from_ref(s), &defs, &mut main_tracked),
            _ => {}
        }
    }
    if !any && main_tracked.is_empty() {
        return None;
    }
    // a top-level variable a def can hand out (return it, or store it) is
    // reachable through that def's result, so its appends copy
    let mut main_cx = Cx { defs: &defs, scalar, tracked: main_tracked.clone(), copy_only: HashSet::new() };
    for item in &program.items {
        if let Item::Def(d) = item {
            let mut locals = HashSet::new();
            locals_of(&d.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(), &d.body, &mut locals);
            for t in &main_tracked {
                if !locals.contains(t) && hands_out(&d.body, t, &main_cx) {
                    main_cx.copy_only.insert(t.clone());
                }
            }
        }
    }
    let mut items = Vec::with_capacity(program.items.len());
    for item in &program.items {
        match item {
            Item::Def(d) => {
                let mut tracked = HashSet::new();
                tracked_in(&d.body, &defs, &mut tracked);
                if tracked.is_empty() {
                    items.push(item.clone());
                    continue;
                }
                let cx = Cx { defs: &defs, scalar, tracked, copy_only: HashSet::new() };
                let mut body = Vec::new();
                // a tracked parameter's storage is the caller's
                for p in &d.params {
                    if cx.tracked.contains(&p.name) {
                        body.push(seal(&p.name, Expr::Var { name: p.name.clone(), span: p.span }, p.span));
                    }
                }
                body.extend(rewrite_block(&d.body, &cx));
                let mut d = d.clone();
                d.body = body;
                items.push(Item::Def(d));
            }
            Item::Stmt(s) => {
                for s in rewrite_block(std::slice::from_ref(s), &main_cx) {
                    items.push(Item::Stmt(s));
                }
            }
            other => items.push(other.clone()),
        }
    }
    Some(Program { items, debug: program.debug.clone(), memory: program.memory.clone(), phrases: program.phrases.clone(), whole: program.whole.clone() })
}
