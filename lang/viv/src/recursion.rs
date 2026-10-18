// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The recursion limit: the same depth, and the same located error, in every
//! engine.
//!
//! The interpreter walks the tree on the native stack, so a deep recursion
//! used to end it with a Rust stack overflow and no location, while the
//! compiled program went much deeper. Now every def that can call itself
//! (directly, or through other defs) counts its nesting, and a call that goes
//! past `MAX_DEPTH` stops the run with an error at the def.
//!
//! The count is written in Viv, as an AST rewrite both engine paths apply
//! (`interp::parse_items_diag` and `crate::build_ir`), so the three engines
//! run the same check with no support of their own. A def cannot assign a
//! global, so the counter is a one-element global array, `viv__depth`:
//!
//! ```text
//! viv__depth = zeros(1)                  # first statement of the program
//! def f(n) {
//!     viv__depth[0] += 1
//!     if viv__depth[0] > MAX_DEPTH { <stop: recursion deeper than ...> }
//!     ...
//!     viv__r1 = <value>                  # each `return <value>`
//!     viv__depth[0] -= 1
//!     return viv__r1
//!     ...
//!     viv__depth[0] -= 1                 # at the end, if the body can reach it
//! }
//! ```
//!
//! A program with no recursive def is returned unchanged. `build_ir_plain`,
//! which the def hashes are computed from, does not apply the rewrite, so a
//! def's hash is the hash of the code as written.

use std::collections::{HashMap, HashSet};

use knotkit::span::Span;

use crate::ast::*;
use crate::parser::HIDDEN_PREFIX;

/// The deepest nesting of calls to recursive defs a run may reach. Python's
/// default limit is 1000. The bound on it is the direct wasm backend: in node
/// (and a browser) the JavaScript engine's stack, about 1 MB, held 5000
/// nested calls of a one-line recursive def and between 3000 and 4000 of one
/// with ten locals and an array, so 2500 leaves room. The interpreter runs on
/// a thread with room for it (`viv run`), and native C has room to spare.
pub const MAX_DEPTH: u32 = 2500;

/// The words of the error, one place for every engine.
pub fn message() -> String {
    format!("recursion deeper than {MAX_DEPTH} nested calls")
}

/// Add the depth count to every recursive def of `program`.
pub fn guard(mut program: Program) -> Program {
    let defs: HashSet<String> = program
        .items
        .iter()
        .filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None })
        .collect();
    // Who names whom: a call, or a def passed or stored as a value (which
    // may be called later, through `apply` or a registry).
    let mut edges: HashMap<String, HashSet<String>> = HashMap::new();
    for item in &program.items {
        if let Item::Def(d) = item {
            let mut out = HashSet::new();
            stmts_names(&d.body, &defs, &mut out);
            edges.insert(d.name.clone(), out);
        }
    }
    let recursive: HashSet<String> = defs.iter().filter(|d| reaches(&edges, d, d)).cloned().collect();
    if recursive.is_empty() {
        return program;
    }
    let counter = format!("{HIDDEN_PREFIX}depth");
    let mut n = 0u32;
    for item in program.items.iter_mut() {
        if let Item::Def(d) = item {
            if recursive.contains(&d.name) {
                n += 1;
                guard_def(d, &counter, &format!("{HIDDEN_PREFIX}r{n}"));
            }
        }
    }
    // the counter, before anything runs
    let sp = Span::synthetic(program_source(&program));
    let init = Stmt::Assign {
        name: counter.clone(),
        name_span: sp,
        value: Expr::Call { callee: "zeros".to_string(), args: vec![Expr::Num { value: 1.0, span: sp }], span: sp },
        span: sp,
    };
    program.items.insert(0, Item::Stmt(init));
    program
}

/// The source id of the program's first item (all items share one).
fn program_source(program: &Program) -> knotkit::span::SourceId {
    program
        .items
        .iter()
        .find_map(|i| match i {
            Item::Def(d) => Some(d.span.source),
            Item::Stmt(s) => Some(s.span().source),
            _ => None,
        })
        .unwrap_or(knotkit::span::SourceId(0))
}

/// Whether `from` reaches `to` through one or more edges.
fn reaches(edges: &HashMap<String, HashSet<String>>, from: &str, to: &str) -> bool {
    let mut seen = HashSet::new();
    let mut stack: Vec<&str> = edges.get(from).map(|s| s.iter().map(|x| x.as_str()).collect()).unwrap_or_default();
    while let Some(d) = stack.pop() {
        if d == to {
            return true;
        }
        if seen.insert(d) {
            if let Some(next) = edges.get(d) {
                stack.extend(next.iter().map(|x| x.as_str()));
            }
        }
    }
    false
}

fn guard_def(d: &mut Def, counter: &str, ret: &str) {
    let sp = d.name_span;
    let cell = || Expr::Index {
        base: Box::new(Expr::Var { name: counter.to_string(), span: sp }),
        indices: vec![Expr::Num { value: 0.0, span: sp }],
        span: sp,
    };
    let bump = |op: BinOp| Stmt::SetIndex {
        base: Box::new(Expr::Var { name: counter.to_string(), span: sp }),
        indices: vec![Expr::Num { value: 0.0, span: sp }],
        value: Expr::Binary { op, lhs: Box::new(cell()), rhs: Box::new(Expr::Num { value: 1.0, span: sp }), span: sp },
        span: sp,
    };
    let check = Stmt::If {
        cond: Expr::Binary {
            op: BinOp::Gt,
            lhs: Box::new(cell()),
            rhs: Box::new(Expr::Num { value: MAX_DEPTH as f64, span: sp }),
            span: sp,
        },
        then_body: vec![Stmt::Lens { lens: Box::new(Lens::Stop { message: message(), kind: crate::errkind::ErrKind::Recursion }), span: sp }],
        else_body: Vec::new(),
        span: sp,
    };
    let leave = bump(BinOp::Sub);
    let mut body = std::mem::take(&mut d.body);
    rewrite_returns(&mut body, ret, &leave);
    // A docstring stays the first statement.
    let doc = matches!(body.first(), Some(Stmt::ExprStmt { expr: Expr::Str { .. }, .. }));
    let mut out = Vec::with_capacity(body.len() + 4);
    let mut rest = body.into_iter();
    if doc {
        out.extend(rest.next());
    }
    out.push(bump(BinOp::Add));
    out.push(check);
    // The temporary holding a return value takes the declared type where a
    // value adapts to it: `return 1` in a def that
    // returns an Int, `return None` in one that returns an optional. An
    // undeclared temporary would make the literal a number first.
    if let Some(t @ (TypeExpr::Int | TypeExpr::IntList | TypeExpr::Opt(_))) = &d.ret {
        let decl = Expr::Call {
            callee: crate::parser::DECLARE.to_string(),
            args: vec![Expr::Str { value: ret.to_string(), span: sp }, Expr::Str { value: crate::parser::type_text(t), span: sp }],
            span: sp,
        };
        out.push(Stmt::ExprStmt { expr: decl, span: sp });
    }
    out.extend(rest);
    if !ends_in_return(&out) {
        out.push(leave);
    }
    d.body = out;
}

/// `return v` becomes `ret = v; <leave>; return ret`, at any depth of the
/// body (a lambda has no statements, so none is inside one).
fn rewrite_returns(stmts: &mut Vec<Stmt>, ret: &str, leave: &Stmt) {
    let mut i = 0;
    while i < stmts.len() {
        match &mut stmts[i] {
            Stmt::If { then_body, else_body, .. } => {
                rewrite_returns(then_body, ret, leave);
                rewrite_returns(else_body, ret, leave);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => rewrite_returns(body, ret, leave),
            Stmt::Return { .. } => {
                let Stmt::Return { value, span } = stmts.remove(i) else { unreachable!() };
                let vspan = value.span();
                stmts.insert(i, Stmt::Assign { name: ret.to_string(), name_span: vspan, value, span: vspan });
                stmts.insert(i + 1, leave.clone());
                stmts.insert(i + 2, Stmt::Return { value: Expr::Var { name: ret.to_string(), span: vspan }, span });
                i += 2;
            }
            _ => {}
        }
        i += 1;
    }
}

/// Whether the end of `stmts` cannot be reached: it ends in a `return`, or
/// in an `if` whose two branches both end so.
fn ends_in_return(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Return { .. }) => true,
        Some(Stmt::If { then_body, else_body, .. }) => ends_in_return(then_body) && ends_in_return(else_body),
        _ => false,
    }
}

fn stmts_names(stmts: &[Stmt], defs: &HashSet<String>, out: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => expr_names(value, defs, out),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => expr_names(expr, defs, out),
            Stmt::Test { body, .. } => stmts_names(body, defs, out),
            Stmt::SetIndex { base, indices, value, .. } => {
                expr_names(base, defs, out);
                for e in indices {
                    expr_names(e, defs, out);
                }
                expr_names(value, defs, out);
            }
            Stmt::SetField { base, value, .. } => {
                expr_names(base, defs, out);
                expr_names(value, defs, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                expr_names(cond, defs, out);
                stmts_names(then_body, defs, out);
                stmts_names(else_body, defs, out);
            }
            Stmt::While { cond, body, .. } => {
                expr_names(cond, defs, out);
                stmts_names(body, defs, out);
            }
            Stmt::For { bound, body, .. } => {
                expr_names(bound, defs, out);
                stmts_names(body, defs, out);
            }
            Stmt::Narrate { .. }
            | Stmt::Break { .. }
            | Stmt::Continue { .. }
            | Stmt::StepEnter { .. }
            | Stmt::StepExit { .. }
            | Stmt::Lens { .. } => {}
        }
    }
}

fn expr_names(e: &Expr, defs: &HashSet<String>, out: &mut HashSet<String>) {
    match e {
        Expr::Var { name, .. } => {
            if defs.contains(name) {
                out.insert(name.clone());
            }
        }
        Expr::Call { callee, args, .. } => {
            if defs.contains(callee) {
                out.insert(callee.clone());
            }
            for a in args {
                expr_names(a, defs, out);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            expr_names(lhs, defs, out);
            expr_names(rhs, defs, out);
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_names(operand, defs, out),
        Expr::ArrLit { elems, .. } => {
            for a in elems {
                expr_names(a, defs, out);
            }
        }
        Expr::Index { base, indices, .. } => {
            expr_names(base, defs, out);
            for a in indices {
                expr_names(a, defs, out);
            }
        }
        Expr::Lambda { body, .. } => expr_names(body, defs, out),
        Expr::Field { base, .. } => expr_names(base, defs, out),
        Expr::RecordLit { fields, .. } => {
            for (_, _, a) in fields {
                expr_names(a, defs, out);
            }
        }
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => {}
    }
}

/// The first call of `name` in the program (its defs and top-level
/// statements), by its span.
pub(crate) fn first_call(program: &Program, name: &str) -> Option<Span> {
    let mut found = None;
    for item in &program.items {
        match item {
            Item::Def(d) => stmts_calls(&d.body, name, &mut found),
            Item::Stmt(s) => stmts_calls(std::slice::from_ref(s), name, &mut found),
            _ => {}
        }
        if found.is_some() {
            break;
        }
    }
    found
}

fn stmts_calls(stmts: &[Stmt], name: &str, out: &mut Option<Span>) {
    for s in stmts {
        if out.is_some() {
            return;
        }
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => expr_calls(value, name, out),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => expr_calls(expr, name, out),
            Stmt::Test { body, .. } => stmts_calls(body, name, out),
            Stmt::SetIndex { base, indices, value, .. } => {
                expr_calls(base, name, out);
                for e in indices {
                    expr_calls(e, name, out);
                }
                expr_calls(value, name, out);
            }
            Stmt::SetField { base, value, .. } => {
                expr_calls(base, name, out);
                expr_calls(value, name, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                expr_calls(cond, name, out);
                stmts_calls(then_body, name, out);
                stmts_calls(else_body, name, out);
            }
            Stmt::While { cond, body, .. } => {
                expr_calls(cond, name, out);
                stmts_calls(body, name, out);
            }
            Stmt::For { bound, body, .. } => {
                expr_calls(bound, name, out);
                stmts_calls(body, name, out);
            }
            Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
        }
    }
}

fn expr_calls(e: &Expr, name: &str, out: &mut Option<Span>) {
    if out.is_some() {
        return;
    }
    match e {
        Expr::Call { callee, args, span } => {
            if callee == name {
                *out = Some(*span);
                return;
            }
            for a in args {
                expr_calls(a, name, out);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            expr_calls(lhs, name, out);
            expr_calls(rhs, name, out);
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_calls(operand, name, out),
        Expr::ArrLit { elems, .. } => {
            for a in elems {
                expr_calls(a, name, out);
            }
        }
        Expr::Index { base, indices, .. } => {
            expr_calls(base, name, out);
            for a in indices {
                expr_calls(a, name, out);
            }
        }
        Expr::Lambda { body, .. } => expr_calls(body, name, out),
        Expr::Field { base, .. } => expr_calls(base, name, out),
        Expr::RecordLit { fields, .. } => {
            for (_, _, a) in fields {
                expr_calls(a, name, out);
            }
        }
        Expr::Var { .. } | Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => {}
    }
}
