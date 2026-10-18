// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Reverse-mode automatic differentiation as a source-to-source pass over the
//! AST: `grad(f, x, ...)` and `loss, g = value_and_grad(f, x, ...)`
//! (v4).
//!
//! The pass runs at the end of `crate::parse`, so the interpreter and both
//! compiled backends see the same rewritten program. A program that calls
//! neither is returned untouched.
//!
//! For each differentiated def `f(x, a, b)` it adds one def,
//! `viv__ad_grad_f(x, a, b, value_out)`, built in four steps:
//!
//! 1. `f`'s locals are renamed apart (`viv__ad0_name`), its `return`s become
//!    assignments of one result variable, and every call whose arguments
//!    depend on `x` is inlined (the callee's locals renamed apart too), until
//!    none is left. A call with constant arguments stays a call.
//! 2. Activity: the variables whose values depend on `x` differentiably.
//! 3. The forward sweep: `f`'s statements, with a push onto the runtime tape
//!    (`__ad_push`) before each statement that overwrites a value, a push of
//!    the branch each `if` took and of the pass count of each loop, and at
//!    the end of each pass of a loop, a push of the variables the pass
//!    assigned for the first time.
//! 4. The reverse sweep: the statements last to first, loops run backward
//!    over the recorded count, each `if` down its recorded branch. Each
//!    assignment restores the old value from the tape and adds each operand's
//!    share of the adjoint into that operand's adjoint (`viv__ad_d_name`),
//!    recomputing the operand values it needs from the restored state.
//!
//! What it cannot differentiate it refuses, with the location: there is no
//! silent zero.

use crate::ast::{BinOp, Def, Expr, Item, Lens, Param, Program, Stmt, TypeExpr};
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::{BTreeSet, HashMap, HashSet};

/// The prefix of every name this pass makes. The parser's own hidden names are
/// `viv__` followed by a short word that never starts with `ad`, and a user's
/// names never start with `viv__`.
const P: &str = "viv__ad";

/// The callee the parser writes for `a, b = value_and_grad(...)`: the call,
/// with the second name as its first argument.
pub const VALUE_AND_GRAD_PAIR: &str = "viv__ad_value_and_grad";

type R<T> = Result<T, Diagnostic>;

/// Builtins whose value does not change as their arguments do, or has
/// derivative 0 wherever it has one (piecewise constant).
const CONSTANT: &[&str] = &[
    "floor", "ceil", "round", "len", "numel", "shape", "zeros", "linspace", "all", "any", "rand", "srand", "histogram", "czeros",
    // the array vocabulary (interp/arrays.rs): constructors, indices, counts, draws
    "eye", "ones", "zeros_like", "ones_like", "arange", "argmax", "argmin", "argsort", "flatnonzero", "count_nonzero", "randn",
];

/// Builtins with an adjoint rule (`Gen::back`).
const DIFFERENTIABLE: &[&str] = &[
    "exp", "log", "ln", "sqrt", "sin", "cos", "tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "erf", "abs", "fabs", "pow",
    "atan2", "min", "max", "sum", "transpose", "take", "slab", "where", SLICE_GET, "copy", "concat", "vcat", "hcat", "append", "extend",
    // the array vocabulary (interp/arrays.rs): reductions along an axis or of
    // the whole array, shapes, and the two products that give a number
    "mean", "var", "std", "prod", "amax", "amin", "reshape", "ravel", "flatten", "swapaxes", "dot", "norm",
    // `sum(a, ordered=true)`: its adjoint is sum's
    "__sum_ordered",
];

/// The calls the parser writes for a slice read and a slice write.
const SLICE_GET: &str = crate::interp::slices::GET;
const SLICE_SET: &str = crate::interp::slices::SET;

// ---- small constructors ------------------------------------------------------

fn var(n: &str, span: Span) -> Expr {
    Expr::Var { name: n.to_string(), span }
}
fn num(value: f64, span: Span) -> Expr {
    Expr::Num { value, span }
}
fn call(callee: &str, args: Vec<Expr>, span: Span) -> Expr {
    Expr::Call { callee: callee.to_string(), args, span }
}
fn bin(op: BinOp, l: Expr, r: Expr, span: Span) -> Expr {
    Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span }
}
fn neg(e: Expr, span: Span) -> Expr {
    Expr::Neg { operand: Box::new(e), span }
}
fn index(base: Expr, indices: Vec<Expr>, span: Span) -> Expr {
    Expr::Index { base: Box::new(base), indices, span }
}
fn assign(n: &str, value: Expr, span: Span) -> Stmt {
    Stmt::Assign { name: n.to_string(), name_span: span, value, span }
}
fn set_index(base: &str, indices: Vec<Expr>, value: Expr, span: Span) -> Stmt {
    Stmt::SetIndex { base: Box::new(var(base, span)), indices, value, span }
}
fn expr_stmt(expr: Expr, span: Span) -> Stmt {
    Stmt::ExprStmt { expr, span }
}
fn push(e: Expr, span: Span) -> Stmt {
    expr_stmt(call("__ad_push", vec![e], span), span)
}
/// `v = __ad_pop(v)`: the value on top of the tape, which has `v`'s type.
fn pop_into(v: &str, span: Span) -> Stmt {
    assign(v, call("__ad_pop", vec![var(v, span)], span), span)
}
/// The adjoint variable of `v`.
fn dname(v: &str) -> String {
    format!("{P}_d_{v}")
}

/// The name as the program wrote it, for messages.
pub fn shown(n: &str) -> String {
    if let Some(rest) = n.strip_prefix(P) {
        if let Some(pos) = rest.find('_') {
            if rest[..pos].chars().all(|c| c.is_ascii_digit()) && pos > 0 {
                return rest[pos + 1..].to_string();
            }
        }
    }
    n.to_string()
}

// ---- walking the tree --------------------------------------------------------

/// Every name a block binds: assignment targets and loop variables.
fn assigned_names(stmts: &[Stmt], out: &mut BTreeSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::For { var, body, .. } => {
                out.insert(var.clone());
                assigned_names(body, out);
            }
            Stmt::If { then_body, else_body, .. } => {
                assigned_names(then_body, out);
                assigned_names(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => assigned_names(body, out),
            _ => {}
        }
    }
}

fn visit_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match e {
        Expr::ArrLit { elems, .. } => elems.iter().for_each(|x| visit_expr(x, f)),
        Expr::Index { base, indices, .. } => {
            visit_expr(base, f);
            indices.iter().for_each(|x| visit_expr(x, f));
        }
        Expr::Lambda { body, .. } => visit_expr(body, f),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => visit_expr(operand, f),
        Expr::Binary { lhs, rhs, .. } => {
            visit_expr(lhs, f);
            visit_expr(rhs, f);
        }
        Expr::Call { args, .. } => args.iter().for_each(|x| visit_expr(x, f)),
        Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| visit_expr(x, f)),
        Expr::Field { base, .. } => visit_expr(base, f),
        _ => {}
    }
}

/// Every expression in a block, statement by statement, bodies included.
fn visit_block_exprs(stmts: &[Stmt], f: &mut dyn FnMut(&Expr)) {
    for s in stmts {
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => visit_expr(value, f),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => visit_expr(expr, f),
            Stmt::SetIndex { base, indices, value, .. } => {
                visit_expr(base, f);
                indices.iter().for_each(|x| visit_expr(x, f));
                visit_expr(value, f);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                visit_expr(cond, f);
                visit_block_exprs(then_body, f);
                visit_block_exprs(else_body, f);
            }
            Stmt::While { cond, body, .. } => {
                visit_expr(cond, f);
                visit_block_exprs(body, f);
            }
            Stmt::For { bound, body, .. } => {
                visit_expr(bound, f);
                visit_block_exprs(body, f);
            }
            Stmt::Test { body, .. } => visit_block_exprs(body, f),
            _ => {}
        }
    }
}

fn visit_stmts(stmts: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in stmts {
        f(s);
        match s {
            Stmt::If { then_body, else_body, .. } => {
                visit_stmts(then_body, f);
                visit_stmts(else_body, f);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => visit_stmts(body, f),
            _ => {}
        }
    }
}

pub(crate) fn rename_expr(e: &Expr, map: &HashMap<String, String>) -> Expr {
    let r = |x: &Expr| rename_expr(x, map);
    match e {
        Expr::Var { name, span } => Expr::Var { name: map.get(name).cloned().unwrap_or_else(|| name.clone()), span: *span },
        Expr::ArrLit { elems, span } => Expr::ArrLit { elems: elems.iter().map(r).collect(), span: *span },
        Expr::Index { base, indices, span } => {
            Expr::Index { base: Box::new(r(base)), indices: indices.iter().map(r).collect(), span: *span }
        }
        Expr::Lambda { params, body, span } => {
            // the lambda's own parameters shadow the renaming
            let mut inner = map.clone();
            for p in params {
                inner.remove(p);
            }
            Expr::Lambda { params: params.clone(), body: Box::new(rename_expr(body, &inner)), span: *span }
        }
        Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(r(operand)), span: *span },
        Expr::Not { operand, span } => Expr::Not { operand: Box::new(r(operand)), span: *span },
        Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op: *op, lhs: Box::new(r(lhs)), rhs: Box::new(r(rhs)), span: *span },
        Expr::Call { callee, args, span } => Expr::Call { callee: callee.clone(), args: args.iter().map(r).collect(), span: *span },
        Expr::RecordLit { type_name, type_span, fields, span } => Expr::RecordLit {
            type_name: type_name.clone(),
            type_span: *type_span,
            fields: fields.iter().map(|(n, s, x)| (n.clone(), *s, r(x))).collect(),
            span: *span,
        },
        Expr::Field { base, field, field_span, span } => {
            Expr::Field { base: Box::new(r(base)), field: field.clone(), field_span: *field_span, span: *span }
        }
        other => other.clone(),
    }
}

fn rename_block(stmts: &[Stmt], map: &HashMap<String, String>) -> Vec<Stmt> {
    let n = |x: &String| map.get(x).cloned().unwrap_or_else(|| x.clone());
    let e = |x: &Expr| rename_expr(x, map);
    stmts
        .iter()
        .map(|s| match s {
            Stmt::Assign { name, name_span, value, span } => Stmt::Assign { name: n(name), name_span: *name_span, value: e(value), span: *span },
            Stmt::Return { value, span } => Stmt::Return { value: e(value), span: *span },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt { expr: e(expr), span: *span },
            Stmt::Show { expr, span } => Stmt::Show { expr: e(expr), span: *span },
            Stmt::SetIndex { base, indices, value, span } => {
                Stmt::SetIndex { base: Box::new(e(base)), indices: indices.iter().map(e).collect(), value: e(value), span: *span }
            }
            Stmt::If { cond, then_body, else_body, span } => Stmt::If {
                cond: e(cond),
                then_body: rename_block(then_body, map),
                else_body: rename_block(else_body, map),
                span: *span,
            },
            Stmt::While { cond, body, span } => Stmt::While { cond: e(cond), body: rename_block(body, map), span: *span },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For {
                var: n(var),
                var_span: *var_span,
                bound: e(bound),
                body: rename_block(body, map),
                parallel: *parallel,
                span: *span,
            },
            Stmt::Test { name, body, span } => Stmt::Test { name: name.clone(), body: rename_block(body, map), span: *span },
            other => other.clone(),
        })
        .collect()
}

/// `e` with every read of the variable `name` replaced by `with`.
fn subst_expr(e: &Expr, name: &str, with: &Expr) -> Expr {
    let r = |x: &Expr| subst_expr(x, name, with);
    match e {
        Expr::Var { name: n, .. } if n == name => with.clone(),
        Expr::ArrLit { elems, span } => Expr::ArrLit { elems: elems.iter().map(r).collect(), span: *span },
        Expr::Index { base, indices, span } => Expr::Index { base: Box::new(r(base)), indices: indices.iter().map(r).collect(), span: *span },
        Expr::Lambda { params, body, span } if !params.iter().any(|p| p == name) => Expr::Lambda { params: params.clone(), body: Box::new(r(body)), span: *span },
        Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(r(operand)), span: *span },
        Expr::Not { operand, span } => Expr::Not { operand: Box::new(r(operand)), span: *span },
        Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op: *op, lhs: Box::new(r(lhs)), rhs: Box::new(r(rhs)), span: *span },
        Expr::Call { callee, args, span } => Expr::Call { callee: callee.clone(), args: args.iter().map(r).collect(), span: *span },
        Expr::RecordLit { type_name, type_span, fields, span } => Expr::RecordLit {
            type_name: type_name.clone(),
            type_span: *type_span,
            fields: fields.iter().map(|(n, s, x)| (n.clone(), *s, r(x))).collect(),
            span: *span,
        },
        Expr::Field { base, field, field_span, span } => Expr::Field { base: Box::new(r(base)), field: field.clone(), field_span: *field_span, span: *span },
        other => other.clone(),
    }
}

/// `stmts` with every read of `name` replaced by `with` (a place of the
/// differentiated record that `name` was given).
fn subst_block(stmts: &[Stmt], name: &str, with: &Expr) -> Vec<Stmt> {
    let e = |x: &Expr| subst_expr(x, name, with);
    stmts
        .iter()
        .map(|s| match s {
            Stmt::Assign { name: n, name_span, value, span } => Stmt::Assign { name: n.clone(), name_span: *name_span, value: e(value), span: *span },
            Stmt::Return { value, span } => Stmt::Return { value: e(value), span: *span },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt { expr: e(expr), span: *span },
            Stmt::Show { expr, span } => Stmt::Show { expr: e(expr), span: *span },
            Stmt::SetIndex { base, indices, value, span } => Stmt::SetIndex { base: Box::new(e(base)), indices: indices.iter().map(e).collect(), value: e(value), span: *span },
            Stmt::SetField { base, field, field_span, value, span } => Stmt::SetField { base: Box::new(e(base)), field: field.clone(), field_span: *field_span, value: e(value), span: *span },
            Stmt::If { cond, then_body, else_body, span } => Stmt::If { cond: e(cond), then_body: subst_block(then_body, name, with), else_body: subst_block(else_body, name, with), span: *span },
            Stmt::While { cond, body, span } => Stmt::While { cond: e(cond), body: subst_block(body, name, with), span: *span },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For { var: var.clone(), var_span: *var_span, bound: e(bound), body: subst_block(body, name, with), parallel: *parallel, span: *span },
            Stmt::Test { name: t, body, span } => Stmt::Test { name: t.clone(), body: subst_block(body, name, with), span: *span },
            other => other.clone(),
        })
        .collect()
}

fn contains_return(stmts: &[Stmt]) -> bool {
    let mut found = false;
    visit_stmts(stmts, &mut |s| found |= matches!(s, Stmt::Return { .. }));
    found
}

/// Whether every path through the block ends in a `return`.
fn always_returns(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Return { .. } => true,
        Stmt::If { then_body, else_body, .. } => always_returns(then_body) && always_returns(else_body),
        _ => false,
    })
}

/// The block with one exit: each `return e` becomes `ret = e`, and the
/// statements after an `if` that returns on one branch move into the other.
fn one_exit(stmts: Vec<Stmt>, ret: &str) -> R<Vec<Stmt>> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut it = stmts.into_iter();
    while let Some(s) = it.next() {
        match s {
            Stmt::Return { value, span } => {
                out.push(assign(ret, value, span));
                return Ok(out);
            }
            Stmt::If { cond, then_body, else_body, span } if contains_return(&then_body) || contains_return(&else_body) => {
                let rest: Vec<Stmt> = it.collect();
                let mut then_body = then_body;
                let mut else_body = else_body;
                if !always_returns(&then_body) {
                    then_body.extend(rest.iter().cloned());
                }
                if !always_returns(&else_body) {
                    else_body.extend(rest);
                }
                out.push(Stmt::If { cond, then_body: one_exit(then_body, ret)?, else_body: one_exit(else_body, ret)?, span });
                return Ok(out);
            }
            Stmt::For { ref body, span, .. } | Stmt::While { ref body, span, .. } if contains_return(body) => {
                let mut at = span;
                visit_stmts(body, &mut |x| {
                    if let Stmt::Return { span, .. } = x {
                        at = *span;
                    }
                });
                return Err(Diagnostic::error("`grad` cannot differentiate a `return` inside a loop")
                    .with_primary(at, "returns from inside a loop")
                    .with_note("set a variable, leave the loop by its condition, and return after it"));
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Whether any statement in `stmts` names the variable `u` in an expression
/// (a read, or the array a write goes into).
fn mentions(u: &str, stmts: &[Stmt]) -> bool {
    let mut found = false;
    visit_block_exprs(stmts, &mut |e| {
        if let Expr::Var { name, .. } = e {
            found |= name == u;
        }
    });
    found
}

/// Whether the block, run from its start, uses `u` before it assigns `u`
/// anew: `Some(true)` for a use first, `Some(false)` for an assignment on
/// every path first, `None` when it does neither. A loop's body may run no
/// times, so an assignment inside one never counts as the assignment.
fn exposed(u: &str, stmts: &[Stmt]) -> Option<bool> {
    let reads = |e: &Expr| {
        let mut uses = false;
        visit_expr(e, &mut |x| {
            if let Expr::Var { name, .. } = x {
                uses |= name == u;
            }
        });
        uses
    };
    for s in stmts {
        match s {
            Stmt::Assign { name, value, .. } => {
                if reads(value) {
                    return Some(true);
                }
                if name == u {
                    return Some(false);
                }
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                if reads(cond) {
                    return Some(true);
                }
                match (exposed(u, then_body), exposed(u, else_body)) {
                    (Some(true), _) | (_, Some(true)) => return Some(true),
                    (Some(false), Some(false)) => return Some(false),
                    _ => {}
                }
            }
            Stmt::For { bound: head, body, .. } | Stmt::While { cond: head, body, .. } => {
                if reads(head) || exposed(u, body) == Some(true) {
                    return Some(true);
                }
            }
            other => {
                if mentions(u, std::slice::from_ref(other)) {
                    return Some(true);
                }
            }
        }
    }
    None
}

/// Refuse `v = u` for arrays when `u` is used again later and either name is
/// written. `after` holds the statements that run after each enclosing
/// block, innermost first; `loops` the bodies of the enclosing loops, whose
/// next pass runs again from the top.
fn alias_check<'b>(stmts: &'b [Stmt], after: &mut Vec<&'b [Stmt]>, loops: &mut Vec<&'b [Stmt]>, written: &HashSet<String>) -> R<()> {
    for (i, s) in stmts.iter().enumerate() {
        let rest = &stmts[i + 1..];
        match s {
            Stmt::Assign { name, value: Expr::Var { name: from, .. }, span, .. } if written.contains(name) || written.contains(from) => {
                // live: a later use of `from` before `from` is assigned anew
                let mut live = None;
                for block in std::iter::once(rest).chain(after.iter().rev().copied()) {
                    live = exposed(from, block);
                    if live.is_some() {
                        break;
                    }
                }
                let live = live == Some(true) || loops.iter().any(|b| exposed(from, b) == Some(true));
                if live {
                    return Err(Diagnostic::error(format!("`{} = {}` makes two names for one array, and one of them is written", shown(name), shown(from)))
                        .with_primary(*span, "the names share storage, and both are used after this line")
                        .with_note(format!("write a copy: `{} = {} + 0.0`", shown(name), shown(from))));
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                after.push(rest);
                alias_check(then_body, after, loops, written)?;
                alias_check(else_body, after, loops, written)?;
                after.pop();
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                after.push(rest);
                loops.push(body);
                alias_check(body, after, loops, written)?;
                loops.pop();
                after.pop();
            }
            _ => {}
        }
    }
    Ok(())
}

/// The expression that gives `e`'s value in the reverse sweep: `e` itself
/// for a leaf or a constant, a temporary for an active interior node.
fn value_of(e: &Expr, vals: &HashMap<*const Expr, Expr>) -> Expr {
    vals.get(&(e as *const Expr)).cloned().unwrap_or_else(|| e.clone())
}

// ---- records built inside differentiated code ----------------------------------

/// Scalar replacement of the records a body builds: a variable every
/// assignment of which is a record literal, or another such variable of the
/// same type, and every use of which is a field read or such an assignment,
/// becomes one variable per field, `<var>__<field>`. `t = T { a: e1, b: e2 }`
/// is `t__a = e1` then `t__b = e2` (declaration order, the order every engine
/// computes a literal's fields in), `u = t` is `u__a = t__a` and `u__b = t__b`,
/// and `t.a` is `t__a`. Repeated until nothing changes, so a record inside a
/// record comes apart too. A record the body only receives (a parameter, or
/// the result of a call with constant arguments) is left alone: it is not
/// differentiated, and a field read of it is a constant.
fn scalarize(mut body: Vec<Stmt>, layouts: &HashMap<String, Vec<String>>) -> Vec<Stmt> {
    loop {
        // the candidates and their types
        let mut ty: HashMap<String, String> = HashMap::new();
        let mut bad: HashSet<String> = HashSet::new();
        visit_stmts(&body, &mut |s| {
            if let Stmt::Assign { name, value: Expr::RecordLit { type_name, .. }, .. } = s {
                if !layouts.contains_key(type_name) {
                    bad.insert(name.clone());
                }
                if let Some(old) = ty.insert(name.clone(), type_name.clone()) {
                    if &old != type_name {
                        bad.insert(name.clone());
                    }
                }
            }
        });
        // `u = t`: u takes t's type
        loop {
            let mut grew = false;
            visit_stmts(&body, &mut |s| {
                if let Stmt::Assign { name, value: Expr::Var { name: from, .. }, .. } = s {
                    if let Some(t) = ty.get(from).cloned() {
                        match ty.get(name) {
                            None => {
                                ty.insert(name.clone(), t);
                                grew = true;
                            }
                            Some(old) if *old != t => {
                                bad.insert(name.clone());
                            }
                            _ => {}
                        }
                    }
                }
            });
            if !grew {
                break;
            }
        }
        if ty.is_empty() {
            return body;
        }
        // every assignment of a candidate is a literal or a candidate, and
        // every other use is a field read; otherwise it stays a record
        loop {
            let before = bad.len();
            let ok = |n: &str, bad: &HashSet<String>| ty.contains_key(n) && !bad.contains(n);
            let mut uses_whole: Vec<String> = Vec::new();
            visit_stmts(&body, &mut |s| match s {
                Stmt::Assign { name, value, .. } => {
                    let whole = matches!(value, Expr::RecordLit { .. }) || matches!(value, Expr::Var { name: v, .. } if ty.contains_key(v));
                    if ty.contains_key(name) && !whole {
                        uses_whole.push(name.clone());
                    }
                    if let Expr::Var { name: v, .. } = value {
                        if ty.contains_key(v) && !ok(name, &bad) {
                            uses_whole.push(v.clone());
                        }
                    }
                }
                Stmt::For { var, .. } if ty.contains_key(var) => uses_whole.push(var.clone()),
                _ => {}
            });
            // a candidate named anywhere but as a field's base or a whole
            // assignment's right side
            fn whole_uses(e: &Expr, under_field: bool, top_assign: bool, ty: &HashMap<String, String>, out: &mut Vec<String>) {
                match e {
                    Expr::Var { name, .. } => {
                        if ty.contains_key(name) && !under_field && !top_assign {
                            out.push(name.clone());
                        }
                    }
                    Expr::Field { base, .. } => whole_uses(base, true, false, ty, out),
                    Expr::ArrLit { elems, .. } => elems.iter().for_each(|x| whole_uses(x, false, false, ty, out)),
                    Expr::Index { base, indices, .. } => {
                        whole_uses(base, false, false, ty, out);
                        indices.iter().for_each(|x| whole_uses(x, false, false, ty, out));
                    }
                    Expr::Lambda { body, .. } => whole_uses(body, false, false, ty, out),
                    Expr::Neg { operand, .. } | Expr::Not { operand, .. } => whole_uses(operand, false, false, ty, out),
                    Expr::Binary { lhs, rhs, .. } => {
                        whole_uses(lhs, false, false, ty, out);
                        whole_uses(rhs, false, false, ty, out);
                    }
                    Expr::Call { args, .. } => args.iter().for_each(|x| whole_uses(x, false, false, ty, out)),
                    Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| whole_uses(x, false, false, ty, out)),
                    _ => {}
                }
            }
            visit_stmts(&body, &mut |s| match s {
                Stmt::Assign { value, .. } => whole_uses(value, false, true, &ty, &mut uses_whole),
                Stmt::Return { value, .. } => whole_uses(value, false, false, &ty, &mut uses_whole),
                Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => whole_uses(expr, false, false, &ty, &mut uses_whole),
                Stmt::SetIndex { base, indices, value, .. } => {
                    whole_uses(base, false, false, &ty, &mut uses_whole);
                    indices.iter().for_each(|x| whole_uses(x, false, false, &ty, &mut uses_whole));
                    whole_uses(value, false, false, &ty, &mut uses_whole);
                }
                Stmt::If { cond, .. } | Stmt::While { cond, .. } => whole_uses(cond, false, false, &ty, &mut uses_whole),
                Stmt::For { bound, .. } => whole_uses(bound, false, false, &ty, &mut uses_whole),
                _ => {}
            });
            bad.extend(uses_whole);
            if bad.len() == before {
                break;
            }
        }
        let split: HashMap<String, String> = ty.into_iter().filter(|(n, _)| !bad.contains(n)).collect();
        if split.is_empty() {
            return body;
        }
        body = split_block(body, &split, layouts);
    }
}

fn split_name(var: &str, field: &str) -> String {
    format!("{var}__{field}")
}

fn split_block(stmts: Vec<Stmt>, split: &HashMap<String, String>, layouts: &HashMap<String, Vec<String>>) -> Vec<Stmt> {
    let e = |x: Expr| split_expr(x, split);
    let b = |xs: Vec<Stmt>| split_block(xs, split, layouts);
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        match s {
            Stmt::Assign { name, value, span, .. } if split.contains_key(&name) => {
                let fields = &layouts[&split[&name]];
                match value {
                    Expr::RecordLit { fields: given, .. } => {
                        let mut given: Vec<(String, Span, Expr)> = given;
                        for f in fields {
                            if let Some(k) = given.iter().position(|(g, _, _)| g == f) {
                                let (_, _, v) = given.remove(k);
                                out.push(assign(&split_name(&name, f), e(v), span));
                            }
                        }
                    }
                    Expr::Var { name: from, span: vspan } => {
                        for f in fields {
                            out.push(assign(&split_name(&name, f), var(&split_name(&from, f), vspan), span));
                        }
                    }
                    other => out.push(Stmt::Assign { name: name.clone(), name_span: span, value: e(other), span }),
                }
            }
            Stmt::Assign { name, name_span, value, span } => out.push(Stmt::Assign { name, name_span, value: e(value), span }),
            Stmt::Return { value, span } => out.push(Stmt::Return { value: e(value), span }),
            Stmt::ExprStmt { expr, span } => out.push(Stmt::ExprStmt { expr: e(expr), span }),
            Stmt::Show { expr, span } => out.push(Stmt::Show { expr: e(expr), span }),
            Stmt::SetIndex { base, indices, value, span } => out.push(Stmt::SetIndex {
                base: Box::new(e(*base)),
                indices: indices.into_iter().map(e).collect(),
                value: e(value),
                span,
            }),
            Stmt::If { cond, then_body, else_body, span } => out.push(Stmt::If { cond: e(cond), then_body: b(then_body), else_body: b(else_body), span }),
            Stmt::While { cond, body, span } => out.push(Stmt::While { cond: e(cond), body: b(body), span }),
            Stmt::For { var, var_span, bound, body, parallel, span } => out.push(Stmt::For { var, var_span, bound: e(bound), body: b(body), parallel, span }),
            Stmt::Test { name, body, span } => out.push(Stmt::Test { name, body: b(body), span }),
            other => out.push(other),
        }
    }
    out
}

fn split_expr(e: Expr, split: &HashMap<String, String>) -> Expr {
    let r = |x: Expr| split_expr(x, split);
    match e {
        Expr::Field { base, field, field_span, span } => match *base {
            Expr::Var { name, span: vspan } if split.contains_key(&name) => {
                let _ = vspan;
                var(&split_name(&name, &field), span)
            }
            other => Expr::Field { base: Box::new(r(other)), field, field_span, span },
        },
        Expr::ArrLit { elems, span } => Expr::ArrLit { elems: elems.into_iter().map(r).collect(), span },
        Expr::Index { base, indices, span } => Expr::Index { base: Box::new(r(*base)), indices: indices.into_iter().map(r).collect(), span },
        Expr::Lambda { params, body, span } => Expr::Lambda { params, body: Box::new(r(*body)), span },
        Expr::Neg { operand, span } => neg(r(*operand), span),
        Expr::Not { operand, span } => Expr::Not { operand: Box::new(r(*operand)), span },
        Expr::Binary { op, lhs, rhs, span } => bin(op, r(*lhs), r(*rhs), span),
        Expr::Call { callee, args, span } => call(&callee, args.into_iter().map(r).collect(), span),
        Expr::RecordLit { type_name, type_span, fields, span } => {
            Expr::RecordLit { type_name, type_span, fields: fields.into_iter().map(|(f, s, x)| (f, s, r(x))).collect(), span }
        }
        other => other,
    }
}

// ---- the regions of inlined defs ----------------------------------------------

/// `body`, the renamed statements of the def `def`, between the markers of
/// its region (ast `Lens::Inline`): the enter marker carries `map`, each name
/// of the def's source with the name the region gives it, sorted by the
/// source name. The markers run forward only, like every lens statement, so
/// the sweeps pass them over; they are what lets the call graph, the steps,
/// the story and the explorer's graph show the gradient's forward pass as
/// the defs it was written in.
fn inline_region(def: &str, map: &HashMap<String, String>, body: Vec<Stmt>, span: Span) -> Vec<Stmt> {
    let mut names: Vec<(String, String)> = map.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
    names.sort();
    let mark = |enter: bool, names: Vec<(String, String)>| Stmt::Lens { lens: Box::new(Lens::Inline { def: def.to_string(), enter, names }), span };
    let mut out = Vec::with_capacity(body.len() + 2);
    out.push(mark(true, names));
    out.extend(body);
    out.push(mark(false, Vec::new()));
    out
}

/// A region's enter marker: the def and its renaming.
pub fn region_enter(s: &Stmt) -> Option<(&str, &[(String, String)])> {
    match s {
        Stmt::Lens { lens, .. } => match &**lens {
            Lens::Inline { def, enter: true, names } => Some((def, names)),
            _ => None,
        },
        _ => None,
    }
}

/// The index of the exit marker that closes the region opened at `stmts[i]`.
pub fn region_close(stmts: &[Stmt], i: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (j, s) in stmts.iter().enumerate().skip(i) {
        if let Stmt::Lens { lens, .. } = s {
            if let Lens::Inline { enter, .. } = &**lens {
                if *enter {
                    depth += 1;
                } else {
                    depth -= 1;
                    if depth == 0 {
                        return Some(j);
                    }
                }
            }
        }
    }
    None
}

/// `body` with each region (at any depth) replaced by one call of its def
/// with no arguments: what the region is to a reader of the gradient def,
/// which is how the explorer's graph (`crate::graph`) sees it.
pub fn collapse_regions(body: &[Stmt]) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        if let Some((def, _)) = region_enter(&body[i]) {
            if let Some(j) = region_close(body, i) {
                let span = body[i].span();
                out.push(expr_stmt(call(def, Vec::new(), span), span));
                i = j + 1;
                continue;
            }
        }
        out.push(match &body[i] {
            Stmt::If { cond, then_body, else_body, span } => Stmt::If { cond: cond.clone(), then_body: collapse_regions(then_body), else_body: collapse_regions(else_body), span: *span },
            Stmt::While { cond, body, span } => Stmt::While { cond: cond.clone(), body: collapse_regions(body), span: *span },
            Stmt::For { var, var_span, bound, body, parallel, span } => {
                Stmt::For { var: var.clone(), var_span: *var_span, bound: bound.clone(), body: collapse_regions(body), parallel: *parallel, span: *span }
            }
            other => other.clone(),
        });
        i += 1;
    }
    out
}

/// The name a gradient def generated for `f` has.
pub fn grad_def_name(f: &str) -> String {
    format!("{P}_grad_{f}")
}

/// Whether `name` is one this pass made: a renamed local, a temporary, the
/// tape's helpers' variables, or the cell a value comes back in.
pub fn is_generated(name: &str) -> bool {
    name.starts_with(P)
}

/// The def a generated gradient def differentiates, or None for any other def.
pub fn grad_def_of(name: &str) -> Option<&str> {
    name.strip_prefix(P).and_then(|r| r.strip_prefix("_grad_"))
}

// ---- the pass ------------------------------------------------------------------

/// What a def writes: which of its parameters, and which globals.
#[derive(Clone, Default)]
struct Writes {
    params: Vec<bool>,
    globals: BTreeSet<String>,
}

/// The argument positions a builtin writes into, if it writes any.
pub(crate) fn builtin_writes(callee: &str) -> Option<std::ops::RangeInclusive<usize>> {
    crate::builtins::writes(callee)
}

pub fn transform(program: Program) -> Result<Program, Vec<Diagnostic>> {
    let defs: HashMap<String, Def> =
        program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some((d.name.clone(), d.clone())) } else { None }).collect();
    let own_grad = defs.contains_key("grad");
    let own_vg = defs.contains_key("value_and_grad");
    // A program that does not differentiate anything is returned as it came.
    let mut uses = false;
    for item in &program.items {
        let mut look = |e: &Expr| {
            if let Expr::Call { callee, .. } = e {
                uses |= (callee == "grad" && !own_grad) || (callee == "value_and_grad" && !own_vg) || callee == VALUE_AND_GRAD_PAIR;
            }
        };
        match item {
            Item::Def(d) => visit_block_exprs(&d.body, &mut look),
            Item::Stmt(s) => visit_block_exprs(std::slice::from_ref(s), &mut look),
            _ => {}
        }
    }
    if !uses {
        return Ok(program);
    }
    let writes = def_writes(&defs);
    let recursive = recursive_defs(&defs);
    let layouts: HashMap<String, Vec<String>> = program
        .items
        .iter()
        .filter_map(|i| if let Item::Record(r) = i { Some((r.name.clone(), r.fields.iter().map(|f| f.name.clone()).collect())) } else { None })
        .collect();
    let rtypes: HashMap<String, Vec<(String, TypeExpr)>> = program
        .items
        .iter()
        .filter_map(|i| if let Item::Record(r) = i { Some((r.name.clone(), r.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect())) } else { None })
        .collect();
    let mut ad = Ad { defs, writes, recursive, layouts, rtypes, generated: Vec::new(), made: HashMap::new(), sites: 0, own_grad, own_vg };
    let mut items = Vec::with_capacity(program.items.len());
    let mut errs = Vec::new();
    for item in program.items {
        match item {
            Item::Def(mut d) => match ad.site_block(std::mem::take(&mut d.body)) {
                Ok(body) => {
                    d.body = body;
                    items.push(Item::Def(d));
                }
                Err(e) => errs.push(e),
            },
            Item::Stmt(s) => {
                let mut out = Vec::new();
                match ad.site_stmt(s, &mut out) {
                    Ok(()) => items.extend(out.into_iter().map(Item::Stmt)),
                    Err(e) => errs.push(e),
                }
            }
            other => items.push(other),
        }
    }
    if !errs.is_empty() {
        return Err(errs);
    }
    items.extend(ad.generated.into_iter().map(Item::Def));
    Ok(Program { items, debug: program.debug, memory: program.memory, phrases: program.phrases, whole: program.whole })
}

/// The writes of every def, to a fixpoint through calls.
fn def_writes(defs: &HashMap<String, Def>) -> HashMap<String, Writes> {
    let mut w: HashMap<String, Writes> =
        defs.iter().map(|(n, d)| (n.clone(), Writes { params: vec![false; d.params.len()], globals: BTreeSet::new() })).collect();
    let mut names: Vec<&String> = defs.keys().collect();
    names.sort();
    loop {
        let mut changed = false;
        for name in &names {
            let d = &defs[*name];
            let params: Vec<&str> = d.params.iter().map(|p| p.name.as_str()).collect();
            let mut locals = BTreeSet::new();
            assigned_names(&d.body, &mut locals);
            let mut hit: Vec<String> = Vec::new();
            visit_stmts(&d.body, &mut |s| {
                if let Stmt::SetIndex { base, .. } = s {
                    if let Expr::Var { name, .. } = &**base {
                        hit.push(name.clone());
                    }
                }
            });
            visit_block_exprs(&d.body, &mut |e| {
                if let Expr::Call { callee, args, .. } = e {
                    if let Some(cw) = w.get(callee) {
                        for (i, a) in args.iter().enumerate() {
                            if let (Expr::Var { name, .. }, Some(true)) = (a, cw.params.get(i)) {
                                hit.push(name.clone());
                            }
                        }
                        hit.extend(cw.globals.iter().cloned());
                    } else if let Some(k) = builtin_writes(callee) {
                        for i in k {
                            if let Some(Expr::Var { name, .. }) = args.get(i) {
                                hit.push(name.clone());
                            }
                        }
                    }
                }
            });
            let mut me = w[*name].clone();
            for h in hit {
                if let Some(i) = params.iter().position(|p| *p == h) {
                    me.params[i] = true;
                } else if !locals.contains(&h) {
                    me.globals.insert(h);
                }
            }
            let old = &w[*name];
            if me.params != old.params || me.globals != old.globals {
                w.insert((*name).clone(), me);
                changed = true;
            }
        }
        if !changed {
            return w;
        }
    }
}

/// The defs that can reach themselves through calls.
fn recursive_defs(defs: &HashMap<String, Def>) -> HashSet<String> {
    let calls: HashMap<&String, BTreeSet<String>> = defs
        .iter()
        .map(|(n, d)| {
            let mut c = BTreeSet::new();
            visit_block_exprs(&d.body, &mut |e| {
                if let Expr::Call { callee, .. } = e {
                    if defs.contains_key(callee) {
                        c.insert(callee.clone());
                    }
                }
            });
            (n, c)
        })
        .collect();
    let mut out = HashSet::new();
    for start in defs.keys() {
        let mut seen = HashSet::new();
        let mut stack: Vec<String> = calls[start].iter().cloned().collect();
        while let Some(n) = stack.pop() {
            if &n == start {
                out.insert(start.clone());
                break;
            }
            if seen.insert(n.clone()) {
                if let Some(next) = calls.get(&n) {
                    stack.extend(next.iter().cloned());
                }
            }
        }
    }
    out
}

struct Ad {
    defs: HashMap<String, Def>,
    writes: HashMap<String, Writes>,
    recursive: HashSet<String>,
    /// each record type's fields, in declaration order
    layouts: HashMap<String, Vec<String>>,
    /// and their types
    rtypes: HashMap<String, Vec<(String, TypeExpr)>>,
    generated: Vec<Def>,
    made: HashMap<String, String>,
    sites: usize,
    own_grad: bool,
    own_vg: bool,
}

impl Ad {
    fn site_block(&mut self, stmts: Vec<Stmt>) -> R<Vec<Stmt>> {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts {
            self.site_stmt(s, &mut out)?;
        }
        Ok(out)
    }

    /// Rewrite the calls of `grad` and `value_and_grad` in one statement.
    fn site_stmt(&mut self, s: Stmt, out: &mut Vec<Stmt>) -> R<()> {
        let s = match s {
            Stmt::Assign { name, name_span, value: Expr::Call { callee, args, span: cspan }, span } if callee == VALUE_AND_GRAD_PAIR => {
                if self.own_vg {
                    return Err(Diagnostic::error("`loss, g = value_and_grad(...)` needs the builtin `value_and_grad`")
                        .with_primary(cspan, "this program defines its own `value_and_grad`"));
                }
                let mut args = args.into_iter();
                let Some(Expr::Var { name: gname, span: gspan }) = args.next() else {
                    return Err(Diagnostic::error("internal: the two-result form lost its second name").with_primary(cspan, "here"));
                };
                let rest = args.map(|a| self.site_expr(a)).collect::<R<Vec<_>>>()?;
                let (gen, mut fargs) = self.site_call(rest, cspan, "value_and_grad")?;
                self.sites += 1;
                let v = format!("{P}_v{}", self.sites);
                out.push(assign(&v, call("zeros", vec![num(1.0, cspan)], cspan), cspan));
                fargs.push(var(&v, cspan));
                out.push(Stmt::Assign { name: gname, name_span: gspan, value: call(&gen, fargs, cspan), span });
                out.push(Stmt::Assign { name, name_span, value: index(var(&v, cspan), vec![num(0.0, cspan)], cspan), span });
                return Ok(());
            }
            Stmt::Assign { name, name_span, value, span } => Stmt::Assign { name, name_span, value: self.site_expr(value)?, span },
            Stmt::Return { value, span } => Stmt::Return { value: self.site_expr(value)?, span },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt { expr: self.site_expr(expr)?, span },
            Stmt::Show { expr, span } => Stmt::Show { expr: self.site_expr(expr)?, span },
            Stmt::SetIndex { base, indices, value, span } => Stmt::SetIndex {
                base: Box::new(self.site_expr(*base)?),
                indices: indices.into_iter().map(|x| self.site_expr(x)).collect::<R<Vec<_>>>()?,
                value: self.site_expr(value)?,
                span,
            },
            Stmt::If { cond, then_body, else_body, span } => Stmt::If {
                cond: self.site_expr(cond)?,
                then_body: self.site_block(then_body)?,
                else_body: self.site_block(else_body)?,
                span,
            },
            Stmt::While { cond, body, span } => Stmt::While { cond: self.site_expr(cond)?, body: self.site_block(body)?, span },
            Stmt::For { var, var_span, bound, body, parallel, span } => {
                Stmt::For { var, var_span, bound: self.site_expr(bound)?, body: self.site_block(body)?, parallel, span }
            }
            Stmt::Test { name, body, span } => Stmt::Test { name, body: self.site_block(body)?, span },
            other => other,
        };
        out.push(s);
        Ok(())
    }

    fn site_expr(&mut self, e: Expr) -> R<Expr> {
        Ok(match e {
            Expr::Call { callee, args, span } => {
                let args = args.into_iter().map(|a| self.site_expr(a)).collect::<R<Vec<_>>>()?;
                if callee == "grad" && !self.own_grad {
                    let (gen, mut fargs) = self.site_call(args, span, "grad")?;
                    fargs.push(call("zeros", vec![num(1.0, span)], span));
                    return Ok(call(&gen, fargs, span));
                }
                if (callee == "value_and_grad" && !self.own_vg) || callee == VALUE_AND_GRAD_PAIR {
                    return Err(Diagnostic::error("`value_and_grad` gives two results")
                        .with_primary(span, "used here as one value")
                        .with_note("write it on its own line: `loss, g = value_and_grad(f, x, ...)`, or use `grad(f, x, ...)` for the gradient alone"));
                }
                Expr::Call { callee, args, span }
            }
            Expr::ArrLit { elems, span } => Expr::ArrLit { elems: elems.into_iter().map(|x| self.site_expr(x)).collect::<R<Vec<_>>>()?, span },
            Expr::Index { base, indices, span } => Expr::Index {
                base: Box::new(self.site_expr(*base)?),
                indices: indices.into_iter().map(|x| self.site_expr(x)).collect::<R<Vec<_>>>()?,
                span,
            },
            Expr::Lambda { params, body, span } => Expr::Lambda { params, body: Box::new(self.site_expr(*body)?), span },
            Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(self.site_expr(*operand)?), span },
            Expr::Not { operand, span } => Expr::Not { operand: Box::new(self.site_expr(*operand)?), span },
            Expr::Binary { op, lhs, rhs, span } => {
                Expr::Binary { op, lhs: Box::new(self.site_expr(*lhs)?), rhs: Box::new(self.site_expr(*rhs)?), span }
            }
            Expr::RecordLit { type_name, type_span, fields, span } => Expr::RecordLit {
                type_name,
                type_span,
                fields: fields.into_iter().map(|(n, s, x)| Ok((n, s, self.site_expr(x)?))).collect::<R<Vec<_>>>()?,
                span,
            },
            Expr::Field { base, field, field_span, span } => Expr::Field { base: Box::new(self.site_expr(*base)?), field, field_span, span },
            other => other,
        })
    }

    /// Check a call's `(f, x, ...)` and make `f`'s gradient def; returns its
    /// name and the arguments to pass it (all but `f`).
    fn site_call(&mut self, args: Vec<Expr>, span: Span, what: &str) -> R<(String, Vec<Expr>)> {
        if args.len() < 2 {
            return Err(Diagnostic::error(format!("`{what}` takes (f, x, ...): a def and the value to differentiate at"))
                .with_primary(span, "missing arguments"));
        }
        let Expr::Var { name: f, span: fspan } = &args[0] else {
            return Err(Diagnostic::error(format!("`{what}`'s first argument must name a def"))
                .with_primary(args[0].span(), "not the name of a def"));
        };
        let Some(def) = self.defs.get(f) else {
            return Err(Diagnostic::error(format!("`{what}`: `{f}` is not a def of this program"))
                .with_primary(*fspan, "no def by this name")
                .with_note("the function to differentiate is named, not written inline: `def f(x) { ... }` then `grad(f, x)`"));
        };
        if def.params.len() != args.len() - 1 {
            return Err(Diagnostic::error(format!(
                "`{f}` takes {} argument(s); this `{what}` passes {}",
                def.params.len(),
                args.len() - 1
            ))
            .with_primary(span, "wrong number of arguments for the def"));
        }
        let gen = self.generate(f, span)?;
        Ok((gen, args[1..].to_vec()))
    }

    fn generate(&mut self, f: &str, site: Span) -> R<String> {
        if let Some(g) = self.made.get(f) {
            return Ok(g.clone());
        }
        let def = self.defs[f].clone();
        let name = grad_def_name(f);
        // a record of arrays is differentiated through its arrays
        // (records)
        let xrec = match def.params.first().and_then(|p| p.ty.as_ref()) {
            Some(TypeExpr::Named { name, span }) if self.rtypes.contains_key(name) => {
                record_of_arrays(&self.rtypes, name).map_err(|(rec, field, what)| {
                    Diagnostic::error(format!("`grad` differentiates a record of arrays, and the field `{field}` of `{rec}` is a {what}"))
                        .with_primary(*span, "the differentiated record")
                        .with_secondary(site, format!("`{f}` is differentiated for this call"))
                        .with_note("keep numbers such as sizes in a record of their own, passed as another argument")
                })?;
                Some(name.clone())
            }
            _ => None,
        };
        let mut g = Gen {
            defs: &self.defs,
            layouts: &self.layouts,
            rtypes: &self.rtypes,
            writes: &self.writes,
            recursive: &self.recursive,
            inst: 0,
            tmp: 0,
            x: String::new(),
            xrec,
            locals: HashSet::new(),
            active: HashSet::new(),
            write_only: HashSet::new(),
            conservative: true,
        };
        let (params, body) = g.build(&def).map_err(|d| d.with_secondary(site, format!("`{f}` is differentiated for this call")))?;
        // the def's own annotations, so a record argument has its type; the
        // last parameter is the value's one-element array
        let n = params.len();
        let params = params
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                let ty = if i + 1 == n { Some(TypeExpr::Arr) } else { def.params.get(i).and_then(|p| p.ty.clone()) };
                Param { name, span: def.name_span, ty }
            })
            .collect();
        self.generated.push(Def { name: name.clone(), name_span: def.name_span, params, ret: None, body, span: def.span });
        self.made.insert(f.to_string(), name.clone());
        Ok(name)
    }
}

/// Whether the record `rec` is a record of arrays: every field a real
/// array, a record of arrays or an array of them. The first field that is
/// not, as (record, field, what it is).
fn record_of_arrays(rtypes: &HashMap<String, Vec<(String, TypeExpr)>>, rec: &str) -> Result<(), (String, String, &'static str)> {
    for (f, t) in rtypes.get(rec).into_iter().flatten() {
        match t {
            TypeExpr::Arr => {}
            TypeExpr::Named { name, .. } | TypeExpr::RecList { name, .. } if rtypes.contains_key(name) => record_of_arrays(rtypes, name)?,
            other => {
                let what = match other {
                    TypeExpr::Num => "number",
                    TypeExpr::CArr => "complex array",
                    TypeExpr::Bool => "boolean",
                    TypeExpr::Str => "string",
                    TypeExpr::StrList => "list of strings",
                    _ => "value grad does not follow",
                };
                return Err((rec.to_string(), f.clone(), what));
            }
        }
    }
    Ok(())
}

/// What a place rooted at the differentiated record holds.
#[derive(Clone, Debug, PartialEq)]
enum Part {
    /// an array: its adjoint is the array at the same place of the gradient
    Arr,
    /// a record of the named type
    Rec(String),
    /// an array of records of the named type
    Recs(String),
}

/// The state of one gradient def being built.
struct Gen<'a> {
    defs: &'a HashMap<String, Def>,
    layouts: &'a HashMap<String, Vec<String>>,
    rtypes: &'a HashMap<String, Vec<(String, TypeExpr)>>,
    writes: &'a HashMap<String, Writes>,
    recursive: &'a HashSet<String>,
    /// the next inline instance number (0 is `f` itself)
    inst: usize,
    /// the next temporary's number
    tmp: usize,
    /// the differentiated parameter, renamed
    x: String,
    /// its record type, when it is a record of arrays
    xrec: Option<String>,
    locals: HashSet<String>,
    active: HashSet<String>,
    /// arrays written and never read: their writes are left as they are
    write_only: HashSet<String>,
    /// during inlining: a call with an active argument counts as active
    conservative: bool,
}

/// The rename map for a def body at instance `k`: every parameter and every
/// assigned name to `viv__ad<k>_name`.
fn rename_map(def: &Def, k: usize) -> HashMap<String, String> {
    let mut names = BTreeSet::new();
    assigned_names(&def.body, &mut names);
    for p in &def.params {
        names.insert(p.name.clone());
    }
    names.into_iter().map(|n| (n.clone(), format!("{P}{k}_{n}"))).collect()
}

impl<'a> Gen<'a> {
    /// What the place `e` holds, when it is rooted at the differentiated
    /// record: the record itself, then fields and elements of arrays of
    /// records. `None` for anything else.
    fn part(&self, e: &Expr) -> Option<Part> {
        let rec = self.xrec.as_ref()?;
        match e {
            Expr::Var { name, .. } if *name == self.x => Some(Part::Rec(rec.clone())),
            Expr::Field { base, field, .. } => {
                let Part::Rec(r) = self.part(base)? else { return None };
                let (_, t) = self.rtypes.get(&r)?.iter().find(|(f, _)| f == field)?;
                match t {
                    TypeExpr::Arr => Some(Part::Arr),
                    TypeExpr::Named { name, .. } => Some(Part::Rec(name.clone())),
                    TypeExpr::RecList { name, .. } => Some(Part::Recs(name.clone())),
                    _ => None,
                }
            }
            Expr::Index { base, indices, .. } if indices.len() == 1 => match self.part(base)? {
                Part::Recs(r) => Some(Part::Rec(r)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether `e` is a record, or an array of records, inside the
    /// differentiated record (not one of its arrays).
    fn whole(&self, e: &Expr) -> bool {
        matches!(self.part(e), Some(Part::Rec(_) | Part::Recs(_)))
    }

    /// The place of `e`'s adjoint: the same place in the gradient.
    fn dplace(&self, e: &Expr) -> Expr {
        match e {
            Expr::Var { name, span } if *name == self.x => var(&dname(name), *span),
            Expr::Field { base, field, field_span, span } => Expr::Field { base: Box::new(self.dplace(base)), field: field.clone(), field_span: *field_span, span: *span },
            Expr::Index { base, indices, span } => Expr::Index { base: Box::new(self.dplace(base)), indices: indices.clone(), span: *span },
            other => other.clone(),
        }
    }

    /// The place `e` with each index that is not a number computed into a
    /// temporary first (in `pre`), so the place a name stands for does not
    /// move while the name is in use; the reverse sweep reads the
    /// temporaries back from the tape like any other variable.
    fn freeze(&mut self, e: Expr, pre: &mut Vec<Stmt>) -> Expr {
        match e {
            Expr::Field { base, field, field_span, span } => Expr::Field { base: Box::new(self.freeze(*base, pre)), field, field_span, span },
            Expr::Index { base, indices, span } => {
                let base = Box::new(self.freeze(*base, pre));
                let mut idx = Vec::with_capacity(indices.len());
                for i in indices {
                    if matches!(i, Expr::Num { .. }) {
                        idx.push(i);
                    } else {
                        let t = self.fresh("i");
                        let sp = i.span();
                        pre.push(assign(&t, i, sp));
                        idx.push(var(&t, sp));
                    }
                }
                Expr::Index { base, indices: idx, span }
            }
            other => other,
        }
    }

    /// A local given a record inside the differentiated record
    /// (`b = model.blocks[l]`) names that place: every read of it reads the
    /// place, so its arrays' adjoints go to the gradient at that place.
    /// Such a local is given its place once.
    fn alias_locals(&mut self, body: Vec<Stmt>) -> R<Vec<Stmt>> {
        let mut count: HashMap<String, usize> = HashMap::new();
        visit_stmts(&body, &mut |s| {
            if let Stmt::Assign { name, .. } = s {
                *count.entry(name.clone()).or_insert(0) += 1;
            }
        });
        let mut aliases: Vec<(String, Expr)> = Vec::new();
        let body = self.alias_block(body, &count, &mut aliases)?;
        Ok(aliases.iter().fold(body, |b, (n, place)| subst_block(&b, n, place)))
    }

    fn alias_block(&mut self, stmts: Vec<Stmt>, count: &HashMap<String, usize>, aliases: &mut Vec<(String, Expr)>) -> R<Vec<Stmt>> {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts {
            match s {
                Stmt::Assign { name, value, span, .. } if self.whole(&value) => {
                    if count.get(&name).copied().unwrap_or(0) > 1 {
                        return Err(Diagnostic::error(format!("`{}` names a part of the differentiated record, and is given a value more than once", shown(&name)))
                            .with_primary(span, "given a part of the record here")
                            .with_note("give each part its own name, or pass it to a def"));
                    }
                    let place = self.freeze(value, &mut out);
                    aliases.push((name, place));
                }
                Stmt::If { cond, then_body, else_body, span } => {
                    let then_body = self.alias_block(then_body, count, aliases)?;
                    let else_body = self.alias_block(else_body, count, aliases)?;
                    out.push(Stmt::If { cond, then_body, else_body, span });
                }
                Stmt::While { cond, body, span } => {
                    let body = self.alias_block(body, count, aliases)?;
                    out.push(Stmt::While { cond, body, span });
                }
                Stmt::For { var, var_span, bound, body, parallel, span } => {
                    let body = self.alias_block(body, count, aliases)?;
                    out.push(Stmt::For { var, var_span, bound, body, parallel, span });
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }

    fn fresh(&mut self, what: &str) -> String {
        self.tmp += 1;
        format!("{P}_{what}{}", self.tmp)
    }

    /// The gradient def's parameters and body.
    fn build(&mut self, def: &Def) -> R<(Vec<String>, Vec<Stmt>)> {
        let map = rename_map(def, 0);
        self.inst = 1;
        let mut params: Vec<String> = def.params.iter().map(|p| map[&p.name].clone()).collect();
        self.x = params[0].clone();
        let span = def.name_span;
        self.check_body(&def.body, &map)?;
        let ret = format!("{P}_ret");
        // `f`'s own statements are a region of `f` (`inline_region`), like
        // those of every def inlined into it
        let mut body = inline_region(&def.name, &map, one_exit(rename_block(&def.body, &map), &ret)?, span);

        // Inline the calls that take an active argument until none is left.
        loop {
            body = self.alias_locals(body)?;
            self.refresh(&params, &body);
            let (next, changed) = self.inline_block(body)?;
            body = next;
            if !changed {
                break;
            }
        }
        // A record built inside the differentiated code (the values of a
        // def that returns several, `return a, b`) becomes one variable per
        // field, so the sweeps see numbers and arrays only.
        body = scalarize(body, self.layouts);
        self.conservative = false;
        self.refresh(&params, &body);
        body = self.hoist_block(body)?;
        self.refresh(&params, &body);
        self.check_writes(&params, &body)?;

        // The sweeps.
        let mut d: HashSet<String> = params.iter().cloned().collect();
        let (mut fwd, rev) = self.block(&body, &mut d, false)?;
        if !d.contains(&ret) {
            return Err(Diagnostic::error(format!("`{}` does not return a number on every path", def.name))
                .with_primary(def.name_span, "`grad` differentiates the value this def returns"));
        }
        let val = format!("{P}_val");
        fwd.push(set_index(&val, vec![num(0.0, span)], var(&ret, span), span));
        let mut own: Vec<&String> = d.iter().filter(|v| self.active.contains(*v)).collect();
        own.sort();
        for v in own {
            // a record of arrays: a gradient of the same structure, all zero
            let zero = if *v == self.x && self.xrec.is_some() { "zeros_like" } else { "__ad_zero" };
            fwd.push(assign(&dname(v), call(zero, vec![var(v, span)], span), span));
        }
        if self.active.contains(&ret) {
            fwd.push(assign(&dname(&ret), num(1.0, span), span));
        }
        fwd.extend(rev);
        fwd.push(Stmt::Return { value: var(&dname(&self.x), span), span });
        params.push(val);
        Ok((params, fwd))
    }

    /// Recompute the locals and the active set of `body`.
    fn refresh(&mut self, params: &[String], body: &[Stmt]) {
        let mut names = BTreeSet::new();
        assigned_names(body, &mut names);
        self.locals = names.into_iter().chain(params.iter().cloned()).collect();
        self.active = self.activity(body);
    }

    /// Refuse what the transform cannot follow, before renaming hides names.
    fn check_body(&self, body: &[Stmt], map: &HashMap<String, String>) -> R<()> {
        let mut err: Option<Diagnostic> = None;
        visit_block_exprs(body, &mut |e| {
            if err.is_some() {
                return;
            }
            if let Expr::Call { callee, span, .. } = e {
                if callee == "grad" || callee == "value_and_grad" || callee == VALUE_AND_GRAD_PAIR {
                    err = Some(Diagnostic::error("`grad` inside a function being differentiated is not supported")
                        .with_primary(*span, "nested differentiation")
                        .with_note("forward-over-reverse and grad of grad are deferred"));
                } else if map.contains_key(callee) {
                    err = Some(Diagnostic::error(format!("`grad` cannot follow a call through the function value `{callee}`"))
                        .with_primary(*span, "a call through a parameter or variable")
                        .with_note("call the def by its name"));
                }
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        let mut bad: Option<Diagnostic> = None;
        visit_stmts(body, &mut |s| {
            if bad.is_some() {
                return;
            }
            match s {
                Stmt::Break { span } | Stmt::Continue { span } => {
                    bad = Some(Diagnostic::error("`grad` cannot differentiate `break` or `continue` yet")
                        .with_primary(*span, "here")
                        .with_note("write the loop's exit into its condition (`while`), or guard the rest of the body with `if`"));
                }
                Stmt::For { parallel: true, span, .. } => {
                    bad = Some(Diagnostic::error("`grad` cannot differentiate a `parallel for`")
                        .with_primary(*span, "its passes would push onto the tape in no fixed order"));
                }
                Stmt::Test { span, .. } => {
                    bad = Some(Diagnostic::error("a `test` inside a function being differentiated is not supported")
                        .with_primary(*span, "move the test to the top level"));
                }
                _ => {}
            }
        });
        match bad {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    // ---- activity ----------------------------------------------------------

    fn is_def(&self, callee: &str) -> bool {
        self.defs.contains_key(callee) && !self.locals.contains(callee)
    }

    /// Whether `e`'s value depends differentiably on an active variable.
    fn active_in(&self, e: &Expr, act: &HashSet<String>) -> bool {
        match e {
            Expr::Var { name, .. } => act.contains(name),
            Expr::Index { base, .. } => self.active_in(base, act),
            Expr::ArrLit { elems, .. } => elems.iter().any(|x| self.active_in(x, act)),
            Expr::Neg { operand, .. } => self.active_in(operand, act),
            Expr::Binary { op, lhs, rhs, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Matmul | BinOp::Mod => {
                    self.active_in(lhs, act) || self.active_in(rhs, act)
                }
                _ => false,
            },
            Expr::Call { callee, args, .. } => {
                if self.is_def(callee) {
                    return self.conservative && args.iter().any(|a| self.active_in(a, act));
                }
                match callee.as_str() {
                    c if CONSTANT.contains(&c) => false,
                    "take" | "slab" | SLICE_GET => args.first().is_some_and(|a| self.active_in(a, act)),
                    "where" => args.iter().skip(1).any(|a| self.active_in(a, act)),
                    _ => args.iter().any(|a| self.active_in(a, act)),
                }
            }
            Expr::Lambda { body, .. } => self.active_in(body, act),
            Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| self.active_in(x, act)),
            Expr::Field { base, .. } => self.active_in(base, act),
            _ => false,
        }
    }

    fn active_expr(&self, e: &Expr) -> bool {
        self.active_in(e, &self.active)
    }

    fn activity(&self, body: &[Stmt]) -> HashSet<String> {
        let mut act: HashSet<String> = HashSet::new();
        act.insert(self.x.clone());
        loop {
            let before = act.len();
            self.act_block(body, &mut act);
            if act.len() == before {
                return act;
            }
        }
    }

    fn act_block(&self, stmts: &[Stmt], act: &mut HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::Assign { name, value, .. } => {
                    if self.active_in(value, act) {
                        act.insert(name.clone());
                    }
                    self.act_call_writes(value, act);
                }
                Stmt::SetIndex { base, value, .. } => {
                    if let Expr::Var { name, .. } = &**base {
                        if self.active_in(value, act) {
                            act.insert(name.clone());
                        }
                    }
                }
                Stmt::ExprStmt { expr, .. } => self.act_call_writes(expr, act),
                Stmt::If { then_body, else_body, .. } => {
                    self.act_block(then_body, act);
                    self.act_block(else_body, act);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => self.act_block(body, act),
                _ => {}
            }
        }
    }

    /// An array a call writes becomes active when the call takes an active
    /// argument: `setslab(A, k, S)` and `put(A, idx, V)` exactly, a call of a
    /// def (while inlining) conservatively.
    fn act_call_writes(&self, e: &Expr, act: &mut HashSet<String>) {
        let Expr::Call { callee, args, .. } = e else { return };
        let any_active = args.iter().any(|a| self.active_in(a, act));
        if !any_active {
            return;
        }
        if self.is_def(callee) {
            if let Some(w) = self.writes.get(callee) {
                for (i, a) in args.iter().enumerate() {
                    if let (Expr::Var { name, .. }, Some(true)) = (a, w.params.get(i)) {
                        act.insert(name.clone());
                    }
                }
            }
        } else if matches!(callee.as_str(), "setslab" | "put" | SLICE_SET) && args.len() == 3 && self.active_in(&args[2], act) {
            if let Expr::Var { name, .. } = &args[0] {
                act.insert(name.clone());
            }
        }
    }

    // ---- inlining ----------------------------------------------------------

    /// Whether `e` has a call of a def with an active argument (outside
    /// lambdas, whose bodies are not run where they are written).
    fn has_active_call(&self, e: &Expr) -> bool {
        match e {
            Expr::Call { callee, args, .. } => {
                (self.is_def(callee) && args.iter().any(|a| self.active_expr(a))) || args.iter().any(|a| self.has_active_call(a))
            }
            Expr::Lambda { .. } => false,
            Expr::ArrLit { elems, .. } => elems.iter().any(|x| self.has_active_call(x)),
            Expr::Index { base, indices, .. } => self.has_active_call(base) || indices.iter().any(|x| self.has_active_call(x)),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.has_active_call(operand),
            Expr::Binary { lhs, rhs, .. } => self.has_active_call(lhs) || self.has_active_call(rhs),
            Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| self.has_active_call(x)),
            Expr::Field { base, .. } => self.has_active_call(base),
            _ => false,
        }
    }

    /// Whether `callee` is a def that returns a record and can be inlined:
    /// its call is inlined even with constant arguments, so the record comes
    /// apart into variables (`scalarize`) the tape can hold.
    fn returns_record(&self, callee: &str) -> bool {
        self.is_def(callee)
            && !self.recursive.contains(callee)
            && matches!(&self.defs[callee].ret, Some(crate::ast::TypeExpr::Named { name, .. }) if self.layouts.contains_key(name))
    }

    /// Whether `e` has a call that `hoist_inline` inlines.
    fn needs_inline(&self, e: &Expr) -> bool {
        if self.has_active_call(e) {
            return true;
        }
        let mut found = false;
        visit_expr(e, &mut |x| {
            if let Expr::Call { callee, .. } = x {
                found |= self.returns_record(callee);
            }
        });
        found
    }

    fn inline_block(&mut self, stmts: Vec<Stmt>) -> R<(Vec<Stmt>, bool)> {
        let mut out = Vec::with_capacity(stmts.len());
        let mut changed = false;
        for s in stmts {
            match s {
                Stmt::Assign { name, name_span, value, span } if self.needs_inline(&value) => {
                    let value = self.hoist_inline(value, &mut out)?;
                    out.push(Stmt::Assign { name, name_span, value, span });
                    changed = true;
                }
                Stmt::SetIndex { base, indices, value, span } if indices.iter().chain([&value]).any(|x| self.needs_inline(x)) => {
                    let mut idx = Vec::with_capacity(indices.len());
                    for i in indices {
                        idx.push(self.hoist_inline(i, &mut out)?);
                    }
                    let value = self.hoist_inline(value, &mut out)?;
                    out.push(Stmt::SetIndex { base, indices: idx, value, span });
                    changed = true;
                }
                Stmt::ExprStmt { expr, span } if self.needs_inline(&expr) => {
                    let e = self.hoist_inline(expr, &mut out)?;
                    // the value of a call made for its effect is dropped
                    if !matches!(e, Expr::Var { .. }) {
                        out.push(Stmt::ExprStmt { expr: e, span });
                    }
                    changed = true;
                }
                Stmt::Show { expr, span } if self.needs_inline(&expr) => {
                    let expr = self.hoist_inline(expr, &mut out)?;
                    out.push(Stmt::Show { expr, span });
                    changed = true;
                }
                Stmt::If { cond, then_body, else_body, span } => {
                    self.no_active_call(&cond, "an `if` condition")?;
                    let (t, c1) = self.inline_block(then_body)?;
                    let (e, c2) = self.inline_block(else_body)?;
                    changed |= c1 || c2;
                    out.push(Stmt::If { cond, then_body: t, else_body: e, span });
                }
                Stmt::While { cond, body, span } => {
                    self.no_active_call(&cond, "a `while` condition")?;
                    let (b, c) = self.inline_block(body)?;
                    changed |= c;
                    out.push(Stmt::While { cond, body: b, span });
                }
                Stmt::For { var, var_span, bound, body, parallel, span } => {
                    self.no_active_call(&bound, "a `for` bound")?;
                    let (b, c) = self.inline_block(body)?;
                    changed |= c;
                    out.push(Stmt::For { var, var_span, bound, body: b, parallel, span });
                }
                other => out.push(other),
            }
        }
        Ok((out, changed))
    }

    fn no_active_call(&self, e: &Expr, what: &str) -> R<()> {
        if self.has_active_call(e) {
            return Err(Diagnostic::error(format!("`grad` cannot follow a call with a differentiated argument in {what}"))
                .with_primary(e.span(), "the call takes a value that depends on the differentiated argument")
                .with_note("compute it into a variable on the line before"));
        }
        Ok(())
    }

    /// `e` with its calls of defs moved out, left to right, into statements
    /// in `pre`: a call with an active argument is inlined, any other is
    /// assigned to a temporary.
    fn hoist_inline(&mut self, e: Expr, pre: &mut Vec<Stmt>) -> R<Expr> {
        Ok(match e {
            Expr::Call { callee, args, span } if self.is_def(&callee) => {
                let inline = args.iter().any(|a| self.active_expr(a)) || self.returns_record(&callee);
                let mut hoisted = Vec::with_capacity(args.len());
                for a in args {
                    hoisted.push(self.hoist_inline(a, pre)?);
                }
                if inline {
                    self.inline_call(&callee, hoisted, span, pre)?
                } else {
                    let t = self.fresh("c");
                    pre.push(assign(&t, call(&callee, hoisted, span), span));
                    var(&t, span)
                }
            }
            Expr::Call { callee, args, span } => {
                let mut hoisted = Vec::with_capacity(args.len());
                for a in args {
                    hoisted.push(self.hoist_inline(a, pre)?);
                }
                Expr::Call { callee, args: hoisted, span }
            }
            Expr::Binary { op: op @ (BinOp::And | BinOp::Or), lhs, rhs, span } => {
                let e = Expr::Binary { op, lhs, rhs, span };
                self.no_active_call(&e, "`and` or `or` (the right side may not run)")?;
                e
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let l = self.hoist_inline(*lhs, pre)?;
                let r = self.hoist_inline(*rhs, pre)?;
                bin(op, l, r, span)
            }
            Expr::Index { base, indices, span } => {
                let b = self.hoist_inline(*base, pre)?;
                let mut idx = Vec::with_capacity(indices.len());
                for i in indices {
                    idx.push(self.hoist_inline(i, pre)?);
                }
                index(b, idx, span)
            }
            Expr::Neg { operand, span } => neg(self.hoist_inline(*operand, pre)?, span),
            Expr::Not { operand, span } => Expr::Not { operand: Box::new(self.hoist_inline(*operand, pre)?), span },
            Expr::ArrLit { elems, span } => {
                let mut out = Vec::with_capacity(elems.len());
                for x in elems {
                    out.push(self.hoist_inline(x, pre)?);
                }
                Expr::ArrLit { elems: out, span }
            }
            other => {
                if self.has_active_call(&other) {
                    return Err(Diagnostic::error("`grad` cannot follow a call with a differentiated argument here")
                        .with_primary(other.span(), "inside a record or a field access")
                        .with_note("compute it into a variable on the line before"));
                }
                other
            }
        })
    }

    /// Inline one call: the callee's body, renamed apart, into `pre`; returns
    /// the variable that holds its result.
    fn inline_call(&mut self, callee: &str, args: Vec<Expr>, span: Span, pre: &mut Vec<Stmt>) -> R<Expr> {
        if self.recursive.contains(callee) {
            return Err(Diagnostic::error(format!("`grad` cannot differentiate through the recursive def `{callee}`"))
                .with_primary(span, "this call takes a differentiated value")
                .with_note("write the recursion as a loop"));
        }
        let def = &self.defs[callee];
        if def.params.len() != args.len() {
            return Err(Diagnostic::error(format!("`{callee}` expects {} argument(s), got {}", def.params.len(), args.len()))
                .with_primary(span, "wrong number of arguments"));
        }
        let k = self.inst;
        self.inst += 1;
        let mut assigned = BTreeSet::new();
        assigned_names(&def.body, &mut assigned);
        let mut map = rename_map(def, k);
        let mut aliases: Vec<(String, Expr)> = Vec::new();
        for (p, a) in def.params.iter().zip(args) {
            match a {
                // a record inside the differentiated record: the callee
                // reads that place wherever it reads the parameter
                a if !assigned.contains(&p.name) && self.whole(&a) => {
                    let place = self.freeze(a, pre);
                    aliases.push((map[&p.name].clone(), place));
                }
                Expr::Lambda { span, .. } => {
                    return Err(Diagnostic::error(format!("`grad` cannot follow a function passed to `{callee}`"))
                        .with_primary(span, "a function value"));
                }
                Expr::Var { name, span: vspan } if self.defs.contains_key(&name) && !self.locals.contains(&name) => {
                    return Err(Diagnostic::error(format!("`grad` cannot follow a function passed to `{callee}`"))
                        .with_primary(vspan, "a function value"));
                }
                // A variable the callee never rebinds is used under its own
                // name, which is what Viv's aliasing means: a write into
                // the parameter is a write into the caller's array.
                Expr::Var { name, .. } if !assigned.contains(&p.name) => {
                    map.insert(p.name.clone(), name);
                }
                other => {
                    let nm = map[&p.name].clone();
                    pre.push(assign(&nm, other, span));
                }
            }
        }
        let body = rename_block(&def.body, &map);
        let body = aliases.iter().fold(body, |b, (n, place)| subst_block(&b, n, place));
        let ret = format!("{P}{k}_ret");
        let body = one_exit(body, &ret)?;
        self.check_body(&def.body, &rename_map(def, k))?;
        pre.extend(inline_region(callee, &map, body, span));
        Ok(var(&ret, span))
    }

    // ---- the last hoisting -------------------------------------------------

    fn is_pure(&self, callee: &str) -> bool {
        // pure builtins, from the one table (crate::builtins): the reverse
        // sweep may evaluate them again
        !self.is_def(callee) && crate::builtins::is_pure(callee)
    }

    /// Move out of `e`, into temporaries, every call the reverse sweep must not
    /// make again: calls of defs, and builtins that are not pure.
    fn hoist_opaque(&mut self, e: Expr, pre: &mut Vec<Stmt>) -> Expr {
        match e {
            Expr::Call { callee, args, span } => {
                let args: Vec<Expr> = args.into_iter().map(|a| self.hoist_opaque(a, pre)).collect();
                if self.is_pure(&callee) {
                    Expr::Call { callee, args, span }
                } else {
                    let t = self.fresh("c");
                    pre.push(assign(&t, call(&callee, args, span), span));
                    var(&t, span)
                }
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let l = self.hoist_opaque(*lhs, pre);
                let r = self.hoist_opaque(*rhs, pre);
                bin(op, l, r, span)
            }
            Expr::Index { base, indices, span } => {
                let b = self.hoist_opaque(*base, pre);
                let idx = indices.into_iter().map(|i| self.hoist_opaque(i, pre)).collect();
                index(b, idx, span)
            }
            Expr::Neg { operand, span } => neg(self.hoist_opaque(*operand, pre), span),
            Expr::ArrLit { elems, span } => Expr::ArrLit { elems: elems.into_iter().map(|x| self.hoist_opaque(x, pre)).collect(), span },
            other => other,
        }
    }

    /// Whether `e` calls anything that writes an array (below its top).
    fn nested_writer(&self, e: &Expr) -> bool {
        let mut found = false;
        let top = e as *const Expr;
        visit_expr(e, &mut |x| {
            if std::ptr::eq(x, top) {
                return;
            }
            if let Expr::Call { callee, .. } = x {
                let def_writes = self.is_def(callee) && self.writes.get(callee).is_some_and(|w| w.params.iter().any(|b| *b) || !w.globals.is_empty());
                if def_writes || (!self.is_def(callee) && builtin_writes(callee).is_some()) {
                    found = true;
                }
            }
        });
        found
    }

    fn hoist_block(&mut self, stmts: Vec<Stmt>) -> R<Vec<Stmt>> {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts {
            match s {
                Stmt::Assign { name, name_span, value, span } => {
                    let value = if self.active_expr(&value) || self.nested_writer(&value) { self.hoist_opaque(value, &mut out) } else { value };
                    out.push(Stmt::Assign { name, name_span, value, span });
                }
                Stmt::SetIndex { base, indices, value, span } => {
                    let Expr::Var { name: bname, .. } = &*base else {
                        return Err(Diagnostic::error("`grad` can follow a write into an array variable only")
                            .with_primary(base.span(), "not a variable"));
                    };
                    // Indices are evaluated again in the reverse sweep: a call,
                    // or a read of the array being written, goes into a temporary.
                    let mut idx = Vec::with_capacity(indices.len());
                    for i in indices {
                        let mut reads_base = false;
                        let mut has_call = false;
                        visit_expr(&i, &mut |x| match x {
                            Expr::Var { name, .. } if name == bname => reads_base = true,
                            Expr::Call { callee, .. } if !self.is_pure(callee) => has_call = true,
                            _ => {}
                        });
                        if reads_base || has_call {
                            let i = self.hoist_opaque(i, &mut out);
                            let t = self.fresh("i");
                            let sp = i.span();
                            out.push(assign(&t, i, sp));
                            idx.push(var(&t, sp));
                        } else {
                            idx.push(i);
                        }
                    }
                    let value = if self.active_expr(&value) || self.nested_writer(&value) { self.hoist_opaque(value, &mut out) } else { value };
                    out.push(Stmt::SetIndex { base, indices: idx, value, span });
                }
                Stmt::ExprStmt { expr: Expr::Call { callee, args, span: cspan }, span } => {
                    // a top-level call keeps its place; its arguments are hoisted
                    // when they call anything that writes
                    let args = args
                        .into_iter()
                        .map(|a| if self.nested_writer(&a) || matches!(&a, Expr::Call { callee, .. } if !self.is_pure(callee)) { self.hoist_opaque(a, &mut out) } else { a })
                        .collect();
                    out.push(Stmt::ExprStmt { expr: Expr::Call { callee, args, span: cspan }, span });
                }
                Stmt::If { cond, then_body, else_body, span } => {
                    out.push(Stmt::If { cond, then_body: self.hoist_block(then_body)?, else_body: self.hoist_block(else_body)?, span })
                }
                Stmt::While { cond, body, span } => out.push(Stmt::While { cond, body: self.hoist_block(body)?, span }),
                Stmt::For { var, var_span, bound, body, parallel, span } => {
                    out.push(Stmt::For { var, var_span, bound, body: self.hoist_block(body)?, parallel, span })
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }

    // ---- writes and aliases ------------------------------------------------

    /// The array variables a top-level call writes into.
    fn call_writes(&self, e: &Expr) -> Vec<String> {
        let Expr::Call { callee, args, .. } = e else { return Vec::new() };
        let mut out = Vec::new();
        if self.is_def(callee) {
            if let Some(w) = self.writes.get(callee) {
                for (i, a) in args.iter().enumerate() {
                    if let (Expr::Var { name, .. }, Some(true)) = (a, w.params.get(i)) {
                        out.push(name.clone());
                    }
                }
            }
        } else if let Some(k) = builtin_writes(callee) {
            for i in k {
                if let Some(Expr::Var { name, .. }) = args.get(i) {
                    out.push(name.clone());
                }
            }
        }
        out.dedup();
        out
    }

    /// Find the arrays written and never read (their writes stay as they
    /// are), and refuse the writes and aliases the tape cannot follow.
    fn check_writes(&mut self, params: &[String], body: &[Stmt]) -> R<()> {
        // writes: SetIndex bases, and the arrays calls write into
        let mut written: Vec<(String, Span)> = Vec::new();
        visit_stmts(body, &mut |s| match s {
            Stmt::SetIndex { base, span, .. } => {
                if let Expr::Var { name, .. } = &**base {
                    written.push((name.clone(), *span));
                }
            }
            Stmt::Assign { value, span, .. } | Stmt::ExprStmt { expr: value, span } => {
                for w in self.call_writes(value) {
                    written.push((w, *span));
                }
            }
            _ => {}
        });
        // reads: every variable in an expression, except as the target of a
        // write (a SetIndex base, or an argument a call writes into)
        let mut reads: HashSet<String> = HashSet::new();
        let note_reads = |e: &Expr, skip: &[String], reads: &mut HashSet<String>| {
            visit_expr(e, &mut |x| {
                if let Expr::Var { name, .. } = x {
                    if !skip.contains(name) {
                        reads.insert(name.clone());
                    }
                }
            });
        };
        let mut stack: Vec<&Stmt> = body.iter().collect();
        while let Some(s) = stack.pop() {
            match s {
                Stmt::SetIndex { indices, value, .. } => {
                    for i in indices {
                        note_reads(i, &[], &mut reads);
                    }
                    note_reads(value, &[], &mut reads);
                }
                Stmt::Assign { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => {
                    let skip = self.call_writes(value);
                    // a written argument is read too when the call is a def
                    // (it may read before it writes); a builtin's outputs are not
                    let skip = if matches!(value, Expr::Call { callee, .. } if !self.is_def(callee)) { skip } else { Vec::new() };
                    if let Expr::Call { args, .. } = value {
                        for a in args {
                            if !matches!(a, Expr::Var { name, .. } if skip.contains(name)) {
                                note_reads(a, &[], &mut reads);
                            }
                        }
                    } else {
                        note_reads(value, &[], &mut reads);
                    }
                }
                Stmt::If { cond, then_body, else_body, .. } => {
                    note_reads(cond, &[], &mut reads);
                    stack.extend(then_body.iter());
                    stack.extend(else_body.iter());
                }
                Stmt::While { cond, body, .. } => {
                    note_reads(cond, &[], &mut reads);
                    stack.extend(body.iter());
                }
                Stmt::For { bound, body, .. } => {
                    note_reads(bound, &[], &mut reads);
                    stack.extend(body.iter());
                }
                _ => {}
            }
        }
        let mut locals_assigned = BTreeSet::new();
        assigned_names(body, &mut locals_assigned);
        self.write_only.clear();
        for (w, span) in &written {
            if !reads.contains(w) {
                self.write_only.insert(w.clone());
                continue;
            }
            if !locals_assigned.contains(w) {
                let what = if params.contains(w) { "parameter" } else { "global" };
                return Err(Diagnostic::error(format!("`grad` cannot follow a write into the {what} array `{}`, which the function also reads", shown(w)))
                    .with_primary(*span, "written here")
                    .with_note("the reverse sweep restores what the forward sweep overwrote; work on a copy made inside the function (`a = p + 0.0`)"));
            }
            if w == &self.x {
                return Err(Diagnostic::error("`grad` cannot follow a write into the argument being differentiated")
                    .with_primary(*span, "written here")
                    .with_note("work on a copy: `y = x + 0.0`"));
            }
        }
        // `b = a` for an array: the two names share storage. That is fine
        // when `a` is not used again (a def returning its local array), and
        // when neither name is ever written; otherwise the adjoints of the
        // two names would drift apart.
        let written_names: HashSet<String> = written.iter().map(|(w, _)| w.clone()).collect();
        alias_check(body, &mut Vec::new(), &mut Vec::new(), &written_names)
    }

    // ---- the sweeps --------------------------------------------------------

    /// Refuse a read of a local that may not be assigned yet.
    fn check_reads(&self, e: &Expr, d: &HashSet<String>) -> R<()> {
        let mut bad: Option<(String, Span)> = None;
        visit_expr(e, &mut |x| {
            if let Expr::Var { name, span } = x {
                if bad.is_none() && self.locals.contains(name) && !d.contains(name) {
                    bad = Some((name.clone(), *span));
                }
            }
        });
        match bad {
            None => Ok(()),
            Some((name, span)) => {
                if name.ends_with("_ret") && name.starts_with(P) {
                    return Err(Diagnostic::error("a def whose value is used here does not return a value on every path")
                        .with_primary(span, "this call's value"));
                }
                Err(Diagnostic::error(format!("`grad` needs `{}` assigned before this read on every path", shown(&name)))
                    .with_primary(span, "may not be assigned yet here, on the first pass")
                    .with_note("assign it before the loop or the `if`"))
            }
        }
    }

    /// The forward and reverse sweeps of a block. `d` holds the locals
    /// assigned on every path so far and is updated; the reverse sweep comes
    /// back in execution order (last statement first).
    fn block(&mut self, stmts: &[Stmt], d: &mut HashSet<String>, in_loop: bool) -> R<(Vec<Stmt>, Vec<Stmt>)> {
        let mut fwd = Vec::with_capacity(stmts.len());
        let mut revs: Vec<Vec<Stmt>> = Vec::with_capacity(stmts.len());
        for s in stmts {
            let mut rev = Vec::new();
            self.stmt(s, d, in_loop, &mut fwd, &mut rev)?;
            revs.push(rev);
        }
        Ok((fwd, revs.into_iter().rev().flatten().collect()))
    }

    /// A nested block: its sweeps with the variables it assigns first saved at
    /// the end of each pass (inside a loop), restored, and given a zero
    /// adjoint at the start of its reverse sweep. `keep` are the variables
    /// the enclosing block owns (assigned on both branches of an `if`).
    fn finish(&self, fwd: &mut Vec<Stmt>, rev: Vec<Stmt>, d_in: &HashSet<String>, d_out: &HashSet<String>, keep: &HashSet<String>, in_loop: bool, span: Span) -> Vec<Stmt> {
        let mut own: Vec<&String> = d_out.iter().filter(|v| !d_in.contains(*v) && !keep.contains(*v)).collect();
        own.sort();
        let mut out = Vec::new();
        if in_loop {
            for v in &own {
                fwd.push(push(var(v, span), span));
            }
            for v in own.iter().rev() {
                out.push(pop_into(v, span));
            }
        }
        for v in &own {
            if self.active.contains(*v) {
                out.push(assign(&dname(v), call("__ad_zero", vec![var(v, span)], span), span));
            }
        }
        out.extend(rev);
        out
    }

    fn stmt(&mut self, s: &Stmt, d: &mut HashSet<String>, in_loop: bool, fwd: &mut Vec<Stmt>, rev: &mut Vec<Stmt>) -> R<()> {
        match s {
            Stmt::Assign { name, value, span, .. } => {
                self.check_reads(value, d)?;
                let sp = *span;
                // the arrays a call writes into, saved and restored in place
                let writes: Vec<String> = self.call_writes(value).into_iter().filter(|w| !self.write_only.contains(w)).collect();
                if let Expr::Call { callee, .. } = value {
                    if !writes.is_empty() && self.active_expr(value) {
                        return Err(self.no_rule(callee, value.span()));
                    }
                }
                let defined = d.contains(name);
                for w in &writes {
                    fwd.push(push(var(w, sp), sp));
                }
                if defined {
                    fwd.push(push(var(name, sp), sp));
                }
                fwd.push(s.clone());
                let act_v = self.active.contains(name);
                let act_e = act_v && self.active_expr(value);
                let seed = self.fresh("g");
                if act_e {
                    rev.push(assign(&seed, var(&dname(name), sp), sp));
                }
                if defined {
                    rev.push(pop_into(name, sp));
                    if act_v {
                        rev.push(assign(&dname(name), call("__ad_zero", vec![var(name, sp)], sp), sp));
                    }
                }
                for w in writes.iter().rev() {
                    rev.push(expr_stmt(call("__ad_restore", vec![var(w, sp)], sp), sp));
                }
                if act_e {
                    self.adjoint(value, &seed, rev)?;
                }
                d.insert(name.clone());
            }
            Stmt::SetIndex { base, indices, value, span } => {
                let sp = *span;
                let Expr::Var { name: a, .. } = &**base else { unreachable!("hoist_block keeps SetIndex bases variables") };
                for i in indices {
                    self.check_reads(i, d)?;
                }
                self.check_reads(value, d)?;
                self.check_reads(base, d)?;
                if self.write_only.contains(a) {
                    fwd.push(s.clone());
                    return Ok(());
                }
                let elem = index(var(a, sp), indices.clone(), sp);
                fwd.push(push(elem.clone(), sp));
                fwd.push(s.clone());
                let act_a = self.active.contains(a);
                let seed = self.fresh("g");
                if act_a {
                    let de = index(var(&dname(a), sp), indices.clone(), sp);
                    rev.push(assign(&seed, de, sp));
                    rev.push(set_index(&dname(a), indices.clone(), num(0.0, sp), sp));
                }
                rev.push(set_index(a, indices.clone(), call("__ad_pop", vec![num(0.0, sp)], sp), sp));
                if act_a && self.active_expr(value) {
                    self.adjoint(value, &seed, rev)?;
                }
            }
            Stmt::ExprStmt { expr, span } => {
                self.check_reads(expr, d)?;
                let sp = *span;
                match expr {
                    Expr::Call { callee, args, .. } if !self.is_def(callee) && matches!(callee.as_str(), "setslab" | "put" | SLICE_SET) && args.len() == 3 => {
                        let Expr::Var { name: a, .. } = &args[0] else {
                            return Err(Diagnostic::error(format!("`grad` can follow `{callee}` into an array variable only"))
                                .with_primary(args[0].span(), "not a variable"));
                        };
                        if self.write_only.contains(a) {
                            fwd.push(s.clone());
                            return Ok(());
                        }
                        let read = match callee.as_str() {
                            "setslab" => "slab",
                            "put" => "take",
                            _ => SLICE_GET,
                        };
                        let what = args[1].clone();
                        let old = call(read, vec![var(a, sp), what.clone()], sp);
                        fwd.push(push(old.clone(), sp));
                        fwd.push(s.clone());
                        let act_a = self.active.contains(a);
                        let seed = self.fresh("g");
                        if act_a {
                            rev.push(assign(&seed, call(read, vec![var(&dname(a), sp), what.clone()], sp), sp));
                            let zero = call("__ad_zero", vec![var(&seed, sp)], sp);
                            rev.push(expr_stmt(call(callee, vec![var(&dname(a), sp), what.clone(), zero], sp), sp));
                        }
                        rev.push(expr_stmt(call(callee, vec![var(a, sp), what, call("__ad_pop", vec![old], sp)], sp), sp));
                        if act_a && self.active_expr(&args[2]) {
                            if callee == SLICE_SET {
                                // a number written into a slice took the sum
                                // of the slice's adjoint
                                let mut vals = HashMap::new();
                                self.values(&args[2], &mut vals, rev)?;
                                self.acc(&args[2], var(&seed, sp), true, &vals, rev)?;
                            } else {
                                self.adjoint(&args[2], &seed, rev)?;
                            }
                        }
                    }
                    _ => {
                        if self.active_expr(expr) {
                            if let Expr::Call { callee, .. } = expr {
                                if builtin_writes(callee).is_some() || self.is_def(callee) {
                                    return Err(self.no_rule(callee, expr.span()));
                                }
                            }
                        }
                        let writes: Vec<String> = self.call_writes(expr).into_iter().filter(|w| !self.write_only.contains(w)).collect();
                        for w in &writes {
                            fwd.push(push(var(w, sp), sp));
                        }
                        fwd.push(s.clone());
                        for w in writes.iter().rev() {
                            rev.push(expr_stmt(call("__ad_restore", vec![var(w, sp)], sp), sp));
                        }
                    }
                }
            }
            Stmt::If { cond, then_body, else_body, span } => {
                self.check_reads(cond, d)?;
                let sp = *span;
                let d_in = d.clone();
                let mut dt = d_in.clone();
                let (mut tf, tr) = self.block(then_body, &mut dt, in_loop)?;
                let mut de = d_in.clone();
                let (mut ef, er) = self.block(else_body, &mut de, in_loop)?;
                let join: HashSet<String> = dt.intersection(&de).filter(|v| !d_in.contains(*v)).cloned().collect();
                let tr = self.finish(&mut tf, tr, &d_in, &dt, &join, in_loop, sp);
                let er = self.finish(&mut ef, er, &d_in, &de, &join, in_loop, sp);
                if tr.is_empty() && er.is_empty() {
                    fwd.push(Stmt::If { cond: cond.clone(), then_body: tf, else_body: ef, span: sp });
                } else {
                    tf.push(push(num(1.0, sp), sp));
                    ef.push(push(num(0.0, sp), sp));
                    fwd.push(Stmt::If { cond: cond.clone(), then_body: tf, else_body: ef, span: sp });
                    let taken = bin(BinOp::Gt, call("__ad_pop", vec![num(0.0, sp)], sp), num(0.5, sp), sp);
                    rev.push(Stmt::If { cond: taken, then_body: tr, else_body: er, span: sp });
                }
                d.extend(join);
            }
            Stmt::For { var: v, var_span, bound, body, span, .. } => {
                self.check_reads(bound, d)?;
                let sp = *span;
                let mut db = d.clone();
                db.insert(v.clone());
                let d_in = db.clone();
                let (mut bf, br) = self.block(body, &mut db, true)?;
                let br = self.finish(&mut bf, br, &d_in, &db, &HashSet::new(), true, sp);
                if br.is_empty() {
                    fwd.push(Stmt::For { var: v.clone(), var_span: *var_span, bound: bound.clone(), body: bf, parallel: false, span: sp });
                    return Ok(());
                }
                let n = self.fresh("n");
                let defined = d.contains(v);
                if defined {
                    fwd.push(push(var(v, sp), sp));
                }
                fwd.push(assign(&n, bound.clone(), sp));
                fwd.push(Stmt::For { var: v.clone(), var_span: *var_span, bound: var(&n, sp), body: bf, parallel: false, span: sp });
                fwd.push(push(var(&n, sp), sp));
                // reverse: the same passes, last first
                let m = self.fresh("m");
                let k = self.fresh("k");
                rev.push(assign(&n, call("__ad_pop", vec![num(0.0, sp)], sp), sp));
                rev.push(assign(&m, call("ceil", vec![var(&n, sp)], sp), sp));
                let mut rbody = vec![assign(v, bin(BinOp::Sub, bin(BinOp::Sub, var(&m, sp), num(1.0, sp), sp), var(&k, sp), sp), sp)];
                rbody.extend(br);
                rev.push(Stmt::For { var: k, var_span: *var_span, bound: var(&m, sp), body: rbody, parallel: false, span: sp });
                if defined {
                    rev.push(pop_into(v, sp));
                }
            }
            Stmt::While { cond, body, span } => {
                self.check_reads(cond, d)?;
                let sp = *span;
                let mut db = d.clone();
                let d_in = db.clone();
                let (mut bf, br) = self.block(body, &mut db, true)?;
                let br = self.finish(&mut bf, br, &d_in, &db, &HashSet::new(), true, sp);
                if br.is_empty() {
                    fwd.push(Stmt::While { cond: cond.clone(), body: bf, span: sp });
                    return Ok(());
                }
                let c = self.fresh("w");
                let k = self.fresh("k");
                fwd.push(assign(&c, num(0.0, sp), sp));
                bf.push(assign(&c, bin(BinOp::Add, var(&c, sp), num(1.0, sp), sp), sp));
                fwd.push(Stmt::While { cond: cond.clone(), body: bf, span: sp });
                fwd.push(push(var(&c, sp), sp));
                rev.push(assign(&c, call("__ad_pop", vec![num(0.0, sp)], sp), sp));
                rev.push(Stmt::For { var: k, var_span: sp, bound: var(&c, sp), body: br, parallel: false, span: sp });
            }
            Stmt::Show { expr, .. } => {
                self.check_reads(expr, d)?;
                fwd.push(s.clone());
            }
            Stmt::Return { span, .. } | Stmt::Break { span } | Stmt::Continue { span } => {
                return Err(Diagnostic::error("internal: `grad` met a jump after it rewrote the jumps").with_primary(*span, "here"));
            }
            Stmt::SetField { span, .. } => {
                return Err(Diagnostic::error("`grad` does not differentiate through a write into a record")
                    .with_primary(*span, "a field written in differentiated code")
                    .with_note("build the record with its values instead (`R { f: v, ..r }`), or write it outside the differentiated def"));
            }
            // narration, steps, lens statements and docstrings run forward only
            _ => fwd.push(s.clone()),
        }
        Ok(())
    }

    fn no_rule(&self, callee: &str, span: Span) -> Diagnostic {
        let what = if self.is_def(callee) { format!("the def `{callee}`, which writes into an array it is given,") } else { format!("`{callee}`") };
        Diagnostic::error(format!("`grad` has no adjoint for {what} with an argument that depends on the differentiated value"))
            .with_primary(span, "no adjoint rule")
            .with_note("the differentiable builtins are + - * / % @, exp log sqrt sin cos tan tanh sinh cosh asin acos atan erf abs pow atan2 min max, sum mean var std prod amax amin (whole or along an axis), transpose swapaxes reshape ravel flatten take slab where copy concat vcat hcat dot norm, indexing, slices, setslab and put")
    }

    // ---- adjoints of expressions -------------------------------------------

    /// Emit into `rev` the statements that add `e`'s share of the adjoint in
    /// the variable `seed` into its operands' adjoints.
    fn adjoint(&mut self, e: &Expr, seed: &str, rev: &mut Vec<Stmt>) -> R<()> {
        let mut vals: HashMap<*const Expr, Expr> = HashMap::new();
        self.values(e, &mut vals, rev)?;
        let sp = e.span();
        self.back(e, var(seed, sp), &vals, rev)
    }

    /// Recompute, into temporaries, the value of every active interior node
    /// of `e`, children first.
    fn values(&mut self, e: &Expr, vals: &mut HashMap<*const Expr, Expr>, rev: &mut Vec<Stmt>) -> R<()> {
        if !self.active_expr(e) {
            return Ok(());
        }
        let sp = e.span();
        if self.whole(e) {
            return Err(Diagnostic::error("`grad` differentiates the arrays of a record, not arithmetic on whole records")
                .with_primary(sp, "a record inside the differentiated record")
                .with_note("work on its arrays (`b.attn.Wq`), or pass the record to a def"));
        }
        let rebuilt = match e {
            Expr::Var { .. } => return Ok(()),
            Expr::Index { base, indices, .. } => {
                self.values(base, vals, rev)?;
                index(value_of(base, vals), indices.clone(), sp)
            }
            Expr::Neg { operand, .. } => {
                self.values(operand, vals, rev)?;
                neg(value_of(operand, vals), sp)
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                self.values(lhs, vals, rev)?;
                self.values(rhs, vals, rev)?;
                bin(*op, value_of(lhs, vals), value_of(rhs, vals), sp)
            }
            Expr::Call { callee, args, .. } => {
                if !DIFFERENTIABLE.contains(&callee.as_str()) || self.is_def(callee) {
                    return Err(self.no_rule(callee, sp));
                }
                for a in args {
                    self.values(a, vals, rev)?;
                }
                call(callee, args.iter().map(|a| value_of(a, vals)).collect(), sp)
            }
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    self.values(x, vals, rev)?;
                }
                Expr::ArrLit { elems: elems.iter().map(|x| value_of(x, vals)).collect(), span: sp }
            }
            // an array of the differentiated record is read where it is
            Expr::Field { .. } if self.part(e) == Some(Part::Arr) => return Ok(()),
            other => {
                return Err(Diagnostic::error("`grad` cannot differentiate this kind of expression")
                    .with_primary(other.span(), "no adjoint rule (a function value, a record or a complex number)"));
            }
        };
        let t = self.fresh("y");
        rev.push(assign(&t, rebuilt, sp));
        vals.insert(e as *const Expr, var(&t, sp));
        Ok(())
    }

    /// Add `contrib` into the adjoint of `child`: a variable's adjoint, an
    /// element of an array's adjoint, or (for an interior node) a temporary
    /// that is then carried further down. `bcast`: the operation broadcast a
    /// number against an array, so the share is summed down when the child
    /// was a number.
    fn acc(&mut self, child: &Expr, contrib: Expr, bcast: bool, vals: &HashMap<*const Expr, Expr>, rev: &mut Vec<Stmt>) -> R<()> {
        if !self.active_expr(child) {
            return Ok(());
        }
        let sp = child.span();
        let contrib = if bcast { call("__ad_unb", vec![contrib, value_of(child, vals)], sp) } else { contrib };
        match child {
            Expr::Var { name, .. } => {
                let dn = dname(name);
                rev.push(assign(&dn, bin(BinOp::Add, var(&dn, sp), contrib, sp), sp));
                Ok(())
            }
            Expr::Index { base, indices, .. } if matches!(**base, Expr::Var { .. }) => {
                let Expr::Var { name, .. } = &**base else { unreachable!() };
                let dn = dname(name);
                let cur = index(var(&dn, sp), indices.clone(), sp);
                rev.push(set_index(&dn, indices.clone(), bin(BinOp::Add, cur, contrib, sp), sp));
                Ok(())
            }
            // an array of the differentiated record: into the gradient's
            // array at the same place
            Expr::Field { base, field, field_span, .. } if self.part(child) == Some(Part::Arr) => {
                let d = self.dplace(child);
                let base = Box::new(self.dplace(base));
                rev.push(Stmt::SetField { base, field: field.clone(), field_span: *field_span, value: bin(BinOp::Add, d, contrib, sp), span: sp });
                Ok(())
            }
            // an element of one
            Expr::Index { base, indices, .. } if self.part(base) == Some(Part::Arr) => {
                let d = self.dplace(base);
                let cur = index(d.clone(), indices.clone(), sp);
                rev.push(Stmt::SetIndex { base: Box::new(d), indices: indices.clone(), value: bin(BinOp::Add, cur, contrib, sp), span: sp });
                Ok(())
            }
            _ => {
                let t = self.fresh("g");
                rev.push(assign(&t, contrib, sp));
                self.back(child, var(&t, sp), vals, rev)
            }
        }
    }

    /// Carry the adjoint `g` (a variable) of `e` to `e`'s operands.
    fn back(&mut self, e: &Expr, g: Expr, vals: &HashMap<*const Expr, Expr>, rev: &mut Vec<Stmt>) -> R<()> {
        let sp = e.span();
        let v = |x: &Expr| value_of(x, vals);
        match e {
            Expr::Var { .. } => self.acc(e, g, false, vals, rev),
            Expr::Field { .. } if self.part(e) == Some(Part::Arr) => self.acc(e, g, false, vals, rev),
            Expr::Index { base, indices, .. } => {
                if matches!(**base, Expr::Var { .. }) || self.part(base) == Some(Part::Arr) {
                    return self.acc(e, g, false, vals, rev);
                }
                // an element of a computed array: the adjoint is that array's
                // zero with `g` in the one element
                let t = self.fresh("g");
                rev.push(assign(&t, call("__ad_zero", vec![v(base)], sp), sp));
                rev.push(set_index(&t, indices.clone(), g, sp));
                self.back(base, var(&t, sp), vals, rev)
            }
            Expr::Neg { operand, .. } => self.acc(operand, neg(g, sp), false, vals, rev),
            Expr::Binary { op, lhs, rhs, .. } => {
                let (a, b) = (v(lhs), v(rhs));
                match op {
                    BinOp::Add => {
                        self.acc(lhs, g.clone(), true, vals, rev)?;
                        self.acc(rhs, g, true, vals, rev)
                    }
                    BinOp::Sub => {
                        self.acc(lhs, g.clone(), true, vals, rev)?;
                        self.acc(rhs, neg(g, sp), true, vals, rev)
                    }
                    BinOp::Mul => {
                        self.acc(lhs, bin(BinOp::Mul, g.clone(), b, sp), true, vals, rev)?;
                        self.acc(rhs, bin(BinOp::Mul, g, a, sp), true, vals, rev)
                    }
                    BinOp::Div => {
                        self.acc(lhs, bin(BinOp::Div, g.clone(), b.clone(), sp), true, vals, rev)?;
                        let q = bin(BinOp::Div, bin(BinOp::Mul, g, a, sp), bin(BinOp::Mul, b.clone(), b, sp), sp);
                        self.acc(rhs, neg(q, sp), true, vals, rev)
                    }
                    BinOp::Mod => {
                        self.acc(lhs, g.clone(), true, vals, rev)?;
                        let q = bin(BinOp::Mul, g, bin(BinOp::FloorDiv, a, b, sp), sp);
                        self.acc(rhs, neg(q, sp), true, vals, rev)
                    }
                    BinOp::Matmul => {
                        self.acc(lhs, call("__ad_mm_a", vec![g.clone(), a.clone(), b.clone()], sp), false, vals, rev)?;
                        self.acc(rhs, call("__ad_mm_b", vec![g, a, b], sp), false, vals, rev)
                    }
                    _ => Ok(()),
                }
            }
            Expr::Call { callee, args, .. } => self.back_call(e, callee, args, g, vals, rev),
            Expr::ArrLit { elems, .. } => {
                for (i, x) in elems.iter().enumerate() {
                    let fi = num(i as f64, sp);
                    if let Expr::ArrLit { elems: row, .. } = x {
                        for (j, y) in row.iter().enumerate() {
                            self.acc(y, index(g.clone(), vec![fi.clone(), num(j as f64, sp)], sp), false, vals, rev)?;
                        }
                    } else {
                        self.acc(x, index(g.clone(), vec![fi], sp), false, vals, rev)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn back_call(&mut self, e: &Expr, callee: &str, args: &[Expr], g: Expr, vals: &HashMap<*const Expr, Expr>, rev: &mut Vec<Stmt>) -> R<()> {
        let sp = e.span();
        let v = |x: &Expr| value_of(x, vals);
        let y = v(e);
        let one = num(1.0, sp);
        let mul = |l: Expr, r: Expr| bin(BinOp::Mul, l, r, sp);
        let div = |l: Expr, r: Expr| bin(BinOp::Div, l, r, sp);
        let sub = |l: Expr, r: Expr| bin(BinOp::Sub, l, r, sp);
        let add = |l: Expr, r: Expr| bin(BinOp::Add, l, r, sp);
        let f = |n: &str, a: Vec<Expr>| call(n, a, sp);
        // the one-argument builtins of the shared table (lib/cas/rules/derivatives.toml,
        // crate::derivs), which the CAS's `diff` reads too
        if args.len() == 1 {
            if let Some(adj) = crate::derivs::adjoint(callee, g.clone(), v(&args[0]), y.clone(), sp) {
                return self.acc(&args[0], adj, false, vals, rev);
            }
        }
        match (callee, args.len()) {
            ("abs" | "fabs", 1) => {
                let u = v(&args[0]);
                let zero = num(0.0, sp);
                let sign = sub(f("__ad_cmp", vec![u.clone(), zero.clone()]), f("__ad_cmp", vec![zero, u]));
                self.acc(&args[0], mul(g, sign), false, vals, rev)
            }
            ("pow", 2) => {
                let (a, c) = (v(&args[0]), v(&args[1]));
                if self.active_expr(&args[0]) {
                    let d = mul(c.clone(), f("pow", vec![a.clone(), sub(c, one)]));
                    self.acc(&args[0], mul(g.clone(), d), true, vals, rev)?;
                }
                if self.active_expr(&args[1]) {
                    self.acc(&args[1], mul(g, mul(y, f("log", vec![a]))), true, vals, rev)?;
                }
                Ok(())
            }
            ("atan2", 2) => {
                let (p, q) = (v(&args[0]), v(&args[1]));
                let r = add(mul(p.clone(), p.clone()), mul(q.clone(), q.clone()));
                self.acc(&args[0], div(mul(g.clone(), q), r.clone()), true, vals, rev)?;
                self.acc(&args[1], neg(div(mul(g, p), r), sp), true, vals, rev)
            }
            ("max" | "min", 2) => {
                // torch.maximum / torch.minimum: the larger (smaller) operand
                // takes the gradient, and each takes half on a tie
                let (a, b) = (v(&args[0]), v(&args[1]));
                let (first, second) = if callee == "max" { ((a.clone(), b.clone()), (b, a)) } else { ((b.clone(), a.clone()), (a, b)) };
                self.acc(&args[0], mul(g.clone(), f("__ad_cmp", vec![first.0, first.1])), true, vals, rev)?;
                self.acc(&args[1], mul(g, f("__ad_cmp", vec![second.0, second.1])), true, vals, rev)
            }
            ("max" | "min", 1) => {
                // shared evenly among the entries equal to the result (torch.amax)
                let a = v(&args[0]);
                let hit = if callee == "max" { f("__ad_cmp", vec![a, y]) } else { f("__ad_cmp", vec![y, a]) };
                let t = self.fresh("mask");
                rev.push(assign(&t, mul(num(2.0, sp), hit), sp));
                let share = div(mul(g, var(&t, sp)), f("sum", vec![var(&t, sp)]));
                self.acc(&args[0], share, false, vals, rev)
            }
            ("sum" | "__sum_ordered", 1) => {
                let a = v(&args[0]);
                self.acc(&args[0], add(f("__ad_zero", vec![a]), g), false, vals, rev)
            }
            // A reduction along axis k: `g` has the reduced shape (or the
            // kept one, with keepdims), and `__ad_keep` gives it the kept
            // shape, so that adding it to a zero of `a`'s shape broadcasts it
            // back along the axis. `n` is the axis's length.
            ("sum" | "mean" | "var" | "std" | "prod" | "amax" | "amin", 2 | 3) => {
                let a = v(&args[0]);
                let k = args[1].clone();
                let keep = |x: Expr| f("__ad_keep", vec![x, a.clone(), k.clone()]);
                let spread = |x: Expr| add(f("__ad_zero", vec![a.clone()]), x);
                let n = div(f("numel", vec![a.clone()]), f("numel", vec![y.clone()]));
                let centered = || sub(a.clone(), f("mean", vec![a.clone(), k.clone(), Expr::Bool { value: true, span: sp }]));
                let share = match callee {
                    "sum" => spread(keep(g)),
                    "mean" => div(spread(keep(g)), n),
                    // d var / d a = 2 (a - mean) / n
                    "var" => mul(spread(keep(g)), div(mul(num(2.0, sp), centered()), n)),
                    // d std / d a = (a - mean) / (n std)
                    "std" => mul(spread(keep(div(g, y.clone()))), div(centered(), n)),
                    // the product of the other entries along the axis
                    "prod" => mul(spread(keep(g)), f("__ad_prodax", vec![a.clone(), k.clone()])),
                    // amax, amin: shared evenly among the entries equal to the
                    // result, as torch.amax and torch.amin
                    _ => {
                        let t = self.fresh("mask");
                        rev.push(assign(&t, bin(BinOp::EqEq, a.clone(), spread(keep(y.clone())), sp), sp));
                        let count = f("sum", vec![var(&t, sp), k.clone(), Expr::Bool { value: true, span: sp }]);
                        mul(var(&t, sp), spread(div(keep(g), count)))
                    }
                };
                self.acc(&args[0], share, false, vals, rev)
            }
            // the same reductions of the whole array, which give a number
            ("mean" | "var" | "std" | "prod" | "amax" | "amin", 1) => {
                let a = v(&args[0]);
                let n = f("numel", vec![a.clone()]);
                let centered = || sub(a.clone(), f("mean", vec![a.clone()]));
                let share = match callee {
                    "mean" => add(f("__ad_zero", vec![a.clone()]), div(g, n)),
                    "var" => mul(g, div(mul(num(2.0, sp), centered()), n)),
                    "std" => mul(div(g, y), div(centered(), n)),
                    "prod" => mul(g, f("__ad_prod", vec![a.clone()])),
                    _ => {
                        let hit = if callee == "amax" { f("__ad_cmp", vec![a.clone(), y]) } else { f("__ad_cmp", vec![y, a.clone()]) };
                        let t = self.fresh("mask");
                        rev.push(assign(&t, mul(num(2.0, sp), hit), sp));
                        div(mul(g, var(&t, sp)), f("sum", vec![var(&t, sp)]))
                    }
                };
                self.acc(&args[0], share, false, vals, rev)
            }
            // a reshape keeps the entries in order: the adjoint is `g`'s
            // entries in `a`'s shape
            ("reshape", _) | ("ravel" | "flatten", 1) => {
                let a = v(&args[0]);
                self.acc(&args[0], f("__ad_like", vec![g, a]), false, vals, rev)
            }
            ("transpose", 1) => self.acc(&args[0], f("transpose", vec![g]), false, vals, rev),
            // the inverse permutation is argsort of the permutation
            ("transpose", 2) => self.acc(&args[0], f("transpose", vec![g, f("argsort", vec![args[1].clone()])]), false, vals, rev),
            ("swapaxes", 3) => self.acc(&args[0], f("swapaxes", vec![g, args[1].clone(), args[2].clone()]), false, vals, rev),
            ("dot", 2) => {
                let (a, b) = (v(&args[0]), v(&args[1]));
                self.acc(&args[0], mul(g.clone(), b), false, vals, rev)?;
                self.acc(&args[1], mul(g, a), false, vals, rev)
            }
            // d |a| / d a = a / |a|
            ("norm", 1) => {
                let a = v(&args[0]);
                self.acc(&args[0], mul(div(g, y), a), false, vals, rev)
            }
            // take along an axis: each slice of g added back at the index it
            // was taken from, in order, so a repeated index collects every share
            ("take", 3) => {
                let a = v(&args[0]);
                self.acc(&args[0], f("__ad_takeax", vec![g, a, args[1].clone(), args[2].clone()]), false, vals, rev)
            }
            ("take", 2) => {
                let a = v(&args[0]);
                self.acc(&args[0], f("__ad_take", vec![g, a, args[1].clone()]), false, vals, rev)
            }
            ("slab", 2) => {
                let a = v(&args[0]);
                let t = self.fresh("g");
                rev.push(assign(&t, f("__ad_zero", vec![a]), sp));
                rev.push(expr_stmt(f("setslab", vec![var(&t, sp), args[1].clone(), g]), sp));
                self.acc(&args[0], var(&t, sp), false, vals, rev)
            }
            // a slice: its adjoint written into a zero array of the source's shape
            (SLICE_GET, 2) => {
                let a = v(&args[0]);
                let t = self.fresh("g");
                rev.push(assign(&t, f("__ad_zero", vec![a]), sp));
                rev.push(expr_stmt(f(SLICE_SET, vec![var(&t, sp), args[1].clone(), g]), sp));
                self.acc(&args[0], var(&t, sp), false, vals, rev)
            }
            ("copy", 1) => self.acc(&args[0], g, false, vals, rev),
            // a join: each argument takes its piece, at the sum of the
            // extents of the arguments before it
            ("concat" | "vcat" | "hcat", n) if n >= 1 => {
                let mode = num(
                    match callee {
                        "concat" => 0.0,
                        "vcat" => 1.0,
                        _ => 2.0,
                    },
                    sp,
                );
                let mut off = num(0.0, sp);
                for (k, x) in args.iter().enumerate() {
                    let piece = f("__ad_cat", vec![g.clone(), mode.clone(), off.clone(), v(x)]);
                    self.acc(x, piece, false, vals, rev)?;
                    if k + 1 < n {
                        off = add(off, f("__ad_extent", vec![mode.clone(), v(x)]));
                    }
                }
                Ok(())
            }
            // a grown array: its first numel(a) elements are a's share, the
            // rest the added value's (a number, a row or a vector)
            ("append" | "extend", 2) => {
                self.acc(&args[0], f("__ad_head", vec![g.clone(), v(&args[0])]), false, vals, rev)?;
                self.acc(&args[1], f("__ad_tail", vec![g, v(&args[0]), v(&args[1])]), false, vals, rev)
            }
            ("where", 3) => {
                let m = args[0].clone();
                let zero = num(0.0, sp);
                self.acc(&args[1], f("where", vec![m.clone(), g.clone(), zero.clone()]), true, vals, rev)?;
                self.acc(&args[2], f("where", vec![m, zero, g]), true, vals, rev)
            }
            _ => Err(self.no_rule(callee, sp)),
        }
    }
}
