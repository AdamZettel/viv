// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Several results: `return a, b` and `a, b = e` (the idioms study, R5).
//!
//! A pass over the AST at the start of `crate::parse`, so the interpreter and
//! both compiled backends see the same rewritten program, made of records,
//! assignments and index reads they already run. A program with neither form
//! comes back untouched.
//!
//! - A def with `return a, b` returns a record the pass declares for it,
//!   `viv__ret_<def> { _0, _1 }`, whose field types the lowering infers from
//!   the returns (`TypeExpr::Infer`). Every `return` of that def gives the same
//!   number of values; `return g(x)` of a def `g` that gives as many forwards
//!   them.
//! - `a, b = f(x)` of such a def is `t = f(x)`, then `a = t._0`, `b = t._1`.
//!   The record never has a name the program can use: a call of `f` anywhere
//!   else (inside an expression, as a value) is refused, located.
//! - `a, b = b, a` computes every value on the right into a hidden variable
//!   first, then assigns the targets left to right, as Python does.
//! - `B, C, H, W = shape(X)`, or any other right side, unpacks an array:
//!   `t = e`, a stop unless `len(t)` is the number of names, then `B = t[0]`...
//! - `loss, g = value_and_grad(f, x, ...)` is left to `crate::grad` (the
//!   parser keeps its own form for it).
//!
//! A target is a name, an element `a[i]`, or a slice `a[lo:hi]`. The targets'
//! indices are computed after the right side, as in Python.

use crate::ast::{BinOp, Def, Expr, FieldDecl, Item, Lens, Program, RecordDecl, Stmt, TypeExpr};
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::{HashMap, HashSet};

/// The call the parser writes for a comma list: `return a, b`, and the right
/// side of `a, b = b, a`.
pub const TUPLE: &str = "viv__tuple";
/// The call the parser writes for `a, b = e`: an array literal of the
/// targets, then `e`.
pub const UNPACK: &str = "viv__unpack";
/// The prefix of the hidden variable an array is unpacked from, which the
/// lowering names in its error when the value is not an array.
pub const UNPACKED: &str = "viv__unp";

/// The record a def that returns several values returns.
pub fn record_name(def: &str) -> String {
    format!("viv__ret_{def}")
}

/// Whether a record type is one this pass declared, and for which def.
pub fn tuple_def(record: &str) -> Option<&str> {
    record.strip_prefix("viv__ret_")
}

fn field(i: usize) -> String {
    format!("_{i}")
}

type R<T> = Result<T, Diagnostic>;

thread_local! {
    /// The defs that return several values and were defined before the
    /// program being parsed, by how many: a notebook's earlier cells, which
    /// `crate::interp::run_cell` parses apart from the cell it runs.
    static KNOWN: std::cell::RefCell<HashMap<String, usize>> = std::cell::RefCell::new(HashMap::new());
}

/// Run `f` (a parse) knowing that the defs in `known` return several values:
/// `a, b = f(x)` of one of them takes its values apart, and a def that
/// returns its result returns as many.
pub fn with_known<T>(known: HashMap<String, usize>, f: impl FnOnce() -> T) -> T {
    KNOWN.with(|k| *k.borrow_mut() = known);
    let out = f();
    KNOWN.with(|k| k.borrow_mut().clear());
    out
}

pub fn transform(mut program: Program) -> Result<Program, Vec<Diagnostic>> {
    if !uses_tuples(&program) && KNOWN.with(|k| k.borrow().is_empty()) {
        return Ok(program);
    }
    let arity = arities(&program).map_err(|d| vec![d])?;
    let mut records = Vec::new();
    let defs = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    let mut t = Tuples { arity: arity.clone(), defs, hidden: 0 };
    let mut items = Vec::with_capacity(program.items.len() + arity.len());
    for item in std::mem::take(&mut program.items) {
        items.push(match item {
            Item::Def(mut d) => {
                let n = arity.get(&d.name).copied();
                if let Some(n) = n {
                    if let Some(ret) = &d.ret {
                        return Err(vec![Diagnostic::error(format!("`{}` returns {n} values, so it takes no `-> T`", d.name))
                            .with_primary(ret.span().unwrap_or(d.name_span), "remove this annotation")]);
                    }
                    let name = record_name(&d.name);
                    records.push(Item::Record(RecordDecl {
                        name: name.clone(),
                        name_span: d.name_span,
                        fields: (0..n).map(|i| FieldDecl { name: field(i), name_span: d.name_span, ty: TypeExpr::Infer, default: None }).collect(),
                        span: d.span,
                    }));
                    d.ret = Some(TypeExpr::Named { name, span: d.name_span });
                }
                let locals = locals_of(&d);
                let cur = Cur { def: Some((d.name.clone(), n)), locals };
                d.body = t.block(std::mem::take(&mut d.body), &cur).map_err(|e| vec![e])?;
                Item::Def(d)
            }
            Item::Stmt(s) => {
                let cur = Cur { def: None, locals: HashSet::new() };
                let mut out = Vec::new();
                t.stmt(s, &cur, &mut out).map_err(|e| vec![e])?;
                if out.len() == 1 {
                    Item::Stmt(out.pop().expect("one"))
                } else {
                    for s in out {
                        items.push(Item::Stmt(s));
                    }
                    continue;
                }
            }
            other => other,
        });
    }
    // after the program's own records (a returned value may be one of them)
    // and before any def: a record is declared before it is used
    let at = items.iter().rposition(|i| matches!(i, Item::Record(_))).map_or(0, |k| k + 1);
    items.splice(at..at, records);
    program.items = items;
    Ok(program)
}

/// Whether the program has a `TUPLE` or an `UNPACK` anywhere.
fn uses_tuples(p: &Program) -> bool {
    let mut found = false;
    let mut see = |e: &Expr| {
        if let Expr::Call { callee, args, .. } = e {
            found |= callee == TUPLE || callee == UNPACK || crate::interp::arrays::multi_result(callee, args.len()).is_some();
        }
    };
    for item in &p.items {
        match item {
            Item::Def(d) => d.body.iter().for_each(|s| visit_stmt(s, &mut see)),
            Item::Stmt(s) => visit_stmt(s, &mut see),
            _ => {}
        }
    }
    found
}

fn visit_stmt(s: &Stmt, f: &mut dyn FnMut(&Expr)) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => visit_expr(value, f),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => visit_expr(expr, f),
        Stmt::SetIndex { base, indices, value, .. } => {
            visit_expr(base, f);
            indices.iter().for_each(|x| visit_expr(x, f));
            visit_expr(value, f);
        }
        Stmt::SetField { base, value, .. } => {
            visit_expr(base, f);
            visit_expr(value, f);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            visit_expr(cond, f);
            then_body.iter().chain(else_body).for_each(|s| visit_stmt(s, f));
        }
        Stmt::While { cond, body, .. } => {
            visit_expr(cond, f);
            body.iter().for_each(|s| visit_stmt(s, f));
        }
        Stmt::For { bound, body, .. } => {
            visit_expr(bound, f);
            body.iter().for_each(|s| visit_stmt(s, f));
        }
        Stmt::Test { body, .. } => body.iter().for_each(|s| visit_stmt(s, f)),
        _ => {}
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

/// The `return` values of a body, nested blocks included.
fn returns<'a>(body: &'a [Stmt], out: &mut Vec<&'a Expr>) {
    for s in body {
        match s {
            Stmt::Return { value, .. } => out.push(value),
            Stmt::If { then_body, else_body, .. } => {
                returns(then_body, out);
                returns(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => returns(body, out),
            _ => {}
        }
    }
}

/// How many values each def that returns several values gives. A def with a
/// `return a, b` gives that many; a def whose `return g(x)` calls one of them
/// forwards it, found by a fixpoint. Then every `return` of those defs is
/// checked to give the same number.
fn arities(p: &Program) -> R<HashMap<String, usize>> {
    let defs: Vec<&Def> = p.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d) } else { None }).collect();
    // the ones defined before this program, unless it defines them again
    let mut arity: HashMap<String, usize> = KNOWN.with(|k| k.borrow().clone());
    for d in &defs {
        arity.remove(&d.name);
    }
    for d in &defs {
        let mut rs = Vec::new();
        returns(&d.body, &mut rs);
        let mut first: Option<(usize, Span)> = None;
        for r in rs {
            if let Expr::Call { callee, args, span } = r {
                if callee == TUPLE {
                    match first {
                        None => first = Some((args.len(), *span)),
                        Some((n, at)) if n != args.len() => {
                            return Err(Diagnostic::error(format!("`{}` returns {} values here, and {n} before", d.name, args.len()))
                                .with_primary(*span, format!("{} values", args.len()))
                                .with_secondary(at, format!("{n} values"))
                                .with_note("every `return` of a def gives the same number of values"));
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some((n, _)) = first {
            arity.insert(d.name.clone(), n);
        }
    }
    loop {
        let mut changed = false;
        for d in &defs {
            if arity.contains_key(&d.name) {
                continue;
            }
            let mut rs = Vec::new();
            returns(&d.body, &mut rs);
            for r in rs {
                if let Expr::Call { callee, .. } = r {
                    if let Some(&n) = arity.get(callee) {
                        arity.insert(d.name.clone(), n);
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    for d in &defs {
        let Some(&n) = arity.get(&d.name) else { continue };
        let mut rs = Vec::new();
        returns(&d.body, &mut rs);
        for r in rs {
            let gives = match r {
                Expr::Call { callee, args, .. } if callee == TUPLE => args.len(),
                Expr::Call { callee, .. } if arity.contains_key(callee) => arity[callee],
                _ => 1,
            };
            if gives != n {
                let what = if gives == 1 { "one value".to_string() } else { format!("{gives} values") };
                return Err(Diagnostic::error(format!("`{}` returns {n} values, but this `return` gives {what}", d.name))
                    .with_primary(r.span(), what)
                    .with_note("every `return` of a def gives the same number of values"));
            }
        }
    }
    Ok(arity)
}

/// The names a def binds: its parameters and every name it assigns (a local
/// that happens to share a def's name is not the def).
fn locals_of(d: &Def) -> HashSet<String> {
    fn walk(body: &[Stmt], out: &mut HashSet<String>) {
        for s in body {
            match s {
                Stmt::Assign { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::For { var, body, .. } => {
                    out.insert(var.clone());
                    walk(body, out);
                }
                Stmt::If { then_body, else_body, .. } => {
                    walk(then_body, out);
                    walk(else_body, out);
                }
                Stmt::While { body, .. } | Stmt::Test { body, .. } => walk(body, out),
                Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } if callee == UNPACK => {
                    if let Some(Expr::ArrLit { elems, .. }) = args.first() {
                        for e in elems {
                            if let Expr::Var { name, .. } = e {
                                out.insert(name.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut out: HashSet<String> = d.params.iter().map(|p| p.name.clone()).collect();
    walk(&d.body, &mut out);
    out
}

struct Tuples {
    arity: HashMap<String, usize>,
    /// every def of the program: a def named `svd` is the program's own
    defs: HashSet<String>,
    hidden: usize,
}

/// Where the statements being rewritten are: the def (and how many values it
/// returns, if several), and its local names.
struct Cur {
    def: Option<(String, Option<usize>)>,
    locals: HashSet<String>,
}

fn var(name: &str, span: Span) -> Expr {
    Expr::Var { name: name.to_string(), span }
}

impl Tuples {
    fn fresh(&mut self) -> String {
        self.hidden += 1;
        format!("{}tup{}", crate::parser::HIDDEN_PREFIX, self.hidden)
    }

    fn block(&mut self, body: Vec<Stmt>, cur: &Cur) -> R<Vec<Stmt>> {
        let mut out = Vec::with_capacity(body.len());
        for s in body {
            self.stmt(s, cur, &mut out)?;
        }
        Ok(out)
    }

    fn stmt(&mut self, s: Stmt, cur: &Cur, out: &mut Vec<Stmt>) -> R<()> {
        let s = match s {
            Stmt::Return { value, span } => return self.ret(value, span, cur, out),
            Stmt::ExprStmt { expr: Expr::Call { callee, args, span: cspan }, span } if callee == UNPACK => {
                let mut args = args.into_iter();
                let (Some(Expr::ArrLit { elems: targets, .. }), Some(rhs)) = (args.next(), args.next()) else {
                    return Err(Diagnostic::error("internal: an unpacking lost its targets").with_primary(cspan, "here"));
                };
                return self.unpack(targets, rhs, span, cur, out);
            }
            // a call whose values are dropped
            Stmt::ExprStmt { expr: Expr::Call { callee, args, span: cspan }, span } if self.arity.contains_key(&callee) => {
                for a in &args {
                    self.check(a, cur)?;
                }
                Stmt::ExprStmt { expr: Expr::Call { callee, args, span: cspan }, span }
            }
            Stmt::Assign { name, name_span, value, span } => {
                self.check(&value, cur)?;
                Stmt::Assign { name, name_span, value, span }
            }
            Stmt::ExprStmt { expr, span } => {
                self.check(&expr, cur)?;
                Stmt::ExprStmt { expr, span }
            }
            Stmt::Show { expr, span } => {
                self.check(&expr, cur)?;
                Stmt::Show { expr, span }
            }
            Stmt::SetIndex { base, indices, value, span } => {
                self.check(&base, cur)?;
                for i in &indices {
                    self.check(i, cur)?;
                }
                self.check(&value, cur)?;
                Stmt::SetIndex { base, indices, value, span }
            }
            Stmt::SetField { base, field, field_span, value, span } => {
                self.check(&base, cur)?;
                self.check(&value, cur)?;
                Stmt::SetField { base, field, field_span, value, span }
            }
            Stmt::If { cond, then_body, else_body, span } => {
                self.check(&cond, cur)?;
                Stmt::If { cond, then_body: self.block(then_body, cur)?, else_body: self.block(else_body, cur)?, span }
            }
            Stmt::While { cond, body, span } => {
                self.check(&cond, cur)?;
                Stmt::While { cond, body: self.block(body, cur)?, span }
            }
            Stmt::For { var, var_span, bound, body, parallel, span } => {
                self.check(&bound, cur)?;
                Stmt::For { var, var_span, bound, body: self.block(body, cur)?, parallel, span }
            }
            Stmt::Test { name, body, span } => Stmt::Test { name, body: self.block(body, cur)?, span },
            other => other,
        };
        out.push(s);
        Ok(())
    }

    /// Refuse a use of a def that returns several values anywhere but where
    /// its values are unpacked, returned or dropped, and a comma list anywhere
    /// but a `return` or the right side of an unpacking.
    fn check(&self, e: &Expr, cur: &Cur) -> R<()> {
        let mut err: Option<Diagnostic> = None;
        visit_expr(e, &mut |x| {
            if err.is_some() {
                return;
            }
            match x {
                Expr::Call { callee, span, .. } if callee == TUPLE => {
                    err = Some(Diagnostic::error("a list of values separated by commas is only a `return` or the right side of `a, b = ...`")
                        .with_primary(*span, "several values"));
                }
                Expr::Call { callee, span, .. } if callee == UNPACK => {
                    err = Some(Diagnostic::error("`a, b = ...` is a statement of its own").with_primary(*span, "here"));
                }
                Expr::Call { callee, span, .. } if self.arity.contains_key(callee) && !cur.locals.contains(callee) => {
                    let n = self.arity[callee];
                    let names: Vec<String> = (0..n).map(|i| ["a", "b", "c", "d", "e", "f", "g", "h"].get(i).map(|s| s.to_string()).unwrap_or(format!("v{i}"))).collect();
                    err = Some(Diagnostic::error(format!("`{callee}` returns {n} values, so it cannot be part of an expression"))
                        .with_primary(*span, format!("{n} values"))
                        .with_note(format!("unpack them on a line of their own: `{} = {callee}(...)`", names.join(", "))));
                }
                Expr::Call { callee, args, span } if self.builtin_results(callee, args.len(), cur).is_some() => {
                    let n = self.builtin_results(callee, args.len(), cur).expect("checked");
                    let names = if n == 2 { "vals, vecs" } else { "U, s, Vt" };
                    err = Some(Diagnostic::error(format!("`{callee}(A)` gives {n} values, so it cannot be part of an expression"))
                        .with_primary(*span, format!("{n} values"))
                        .with_note(format!("unpack them on a line of their own: `{names} = {callee}(A)`")));
                }
                Expr::Var { name, span } if self.arity.contains_key(name) && !cur.locals.contains(name) => {
                    err = Some(Diagnostic::error(format!("`{name}` returns {} values, so it cannot be passed as a value", self.arity[name]))
                        .with_primary(*span, "a def used as a value")
                        .with_note("a function value (and `grad`) takes a def that returns one value; wrap the call in a def that returns the one you need"));
                }
                _ => {}
            }
        });
        err.map_or(Ok(()), Err)
    }

    /// How many values the builtin call `callee(...)` with `n` arguments gives
    /// when it gives several (interp/arrays.rs, `multi_result`) and the
    /// program has no def or local of that name.
    fn builtin_results(&self, callee: &str, n: usize, cur: &Cur) -> Option<usize> {
        if self.defs.contains(callee) || cur.locals.contains(callee) {
            return None;
        }
        crate::interp::arrays::multi_result(callee, n)
    }

    fn ret(&mut self, value: Expr, span: Span, cur: &Cur, out: &mut Vec<Stmt>) -> R<()> {
        let tuple_def = match &cur.def {
            Some((name, Some(n))) => Some((name.clone(), *n)),
            _ => None,
        };
        match value {
            Expr::Call { callee, args, span: vspan } if callee == TUPLE => {
                let Some((def, _)) = tuple_def else {
                    return Err(Diagnostic::error("`return a, b` returns from a def").with_primary(vspan, "outside a def"));
                };
                for a in &args {
                    self.check(a, cur)?;
                }
                let fields = args.into_iter().enumerate().map(|(i, a)| (field(i), a.span(), a)).collect();
                let lit = Expr::RecordLit { type_name: record_name(&def), type_span: vspan, fields, span: vspan };
                out.push(Stmt::Return { value: lit, span });
            }
            Expr::Call { callee, args, span: vspan } if tuple_def.is_some() && self.arity.contains_key(&callee) && !cur.locals.contains(&callee) => {
                let (def, n) = tuple_def.expect("checked");
                for a in &args {
                    self.check(a, cur)?;
                }
                let call = Expr::Call { callee: callee.clone(), args, span: vspan };
                if callee == def {
                    out.push(Stmt::Return { value: call, span });
                } else {
                    // another def's record: its fields into this def's
                    let t = self.fresh();
                    out.push(Stmt::Assign { name: t.clone(), name_span: vspan, value: call, span });
                    let fields = (0..n)
                        .map(|i| (field(i), vspan, Expr::Field { base: Box::new(var(&t, vspan)), field: field(i), field_span: vspan, span: vspan }))
                        .collect();
                    out.push(Stmt::Return { value: Expr::RecordLit { type_name: record_name(&def), type_span: vspan, fields, span: vspan }, span });
                }
            }
            value => {
                self.check(&value, cur)?;
                out.push(Stmt::Return { value, span });
            }
        }
        Ok(())
    }

    fn unpack(&mut self, targets: Vec<Expr>, rhs: Expr, span: Span, cur: &Cur, out: &mut Vec<Stmt>) -> R<()> {
        let n = targets.len();
        for t in &targets {
            match t {
                Expr::Var { .. } => {}
                Expr::Index { base, indices, .. } => {
                    self.check(base, cur)?;
                    for i in indices {
                        self.check(i, cur)?;
                    }
                }
                Expr::Call { callee, args, .. } if callee == crate::interp::slices::GET && args.len() == 2 => {
                    for a in args {
                        self.check(a, cur)?;
                    }
                }
                // a field of a place, `r.f` (crate::lower checks the place)
                Expr::Field { base, .. } => self.check(base, cur)?,
                other => {
                    return Err(Diagnostic::error("cannot unpack into this")
                        .with_primary(other.span(), "not a name, an element or a slice")
                        .with_note("each target of `a, b = ...` is a name, `a[i]`, `a[lo:hi]` or a field `r.f`"));
                }
            }
        }
        let values: Vec<Expr> = match rhs {
            Expr::Call { callee, args, span: vspan } if callee == TUPLE => {
                if args.len() != n {
                    return Err(Diagnostic::error(format!("{n} names, but {} values", args.len()))
                        .with_primary(vspan, format!("{} values", args.len()))
                        .with_note("`a, b = x, y` takes one value for each name"));
                }
                // every value first, then the targets: a swap reads the old values
                let mut vals = Vec::with_capacity(n);
                for a in args {
                    self.check(&a, cur)?;
                    let t = self.fresh();
                    let sp = a.span();
                    out.push(Stmt::Assign { name: t.clone(), name_span: sp, value: a, span: sp });
                    vals.push(var(&t, sp));
                }
                vals
            }
            Expr::Call { callee, args, span: vspan } if self.arity.contains_key(&callee) && !cur.locals.contains(&callee) => {
                let m = self.arity[&callee];
                if m != n {
                    return Err(Diagnostic::error(format!("`{callee}` returns {m} values, but {n} names unpack them"))
                        .with_primary(vspan, format!("{m} values"))
                        .with_note("give one name for each value"));
                }
                for a in &args {
                    self.check(a, cur)?;
                }
                let t = self.fresh();
                out.push(Stmt::Assign { name: t.clone(), name_span: vspan, value: Expr::Call { callee, args, span: vspan }, span });
                (0..n).map(|i| Expr::Field { base: Box::new(var(&t, vspan)), field: field(i), field_span: vspan, span: vspan }).collect()
            }
            // `vals, vecs = eig_sym(A)`, `w, V = eig_herm(A)`, `U, s, Vt =
            // svd(A)`: the builtin that fills arrays, with arrays allocated here
            Expr::Call { callee, mut args, span: vspan } if self.builtin_results(&callee, args.len(), cur).is_some() => {
                let m = self.builtin_results(&callee, args.len(), cur).expect("checked");
                if m != n {
                    return Err(Diagnostic::error(format!("`{callee}(A)` gives {m} values, but {n} names unpack them"))
                        .with_primary(vspan, format!("{m} values"))
                        .with_note("give one name for each value"));
                }
                let a = args.pop().expect("one argument");
                self.check(&a, cur)?;
                let sp = vspan;
                let call = |f: &str, args: Vec<Expr>| Expr::Call { callee: f.to_string(), args, span: sp };
                let hold = |me: &mut Self, value: Expr, out: &mut Vec<Stmt>| {
                    let t = me.fresh();
                    out.push(Stmt::Assign { name: t.clone(), name_span: sp, value, span });
                    var(&t, sp)
                };
                let t = hold(self, a, out);
                let rows = hold(self, call("len", vec![t.clone()]), out);
                let outs: Vec<Expr> = if callee == "svd" {
                    let cols = hold(self, Expr::Binary { op: BinOp::Div, lhs: Box::new(call("numel", vec![t.clone()])), rhs: Box::new(rows.clone()), span: sp }, out);
                    let k = hold(self, call("min", vec![rows.clone(), cols.clone()]), out);
                    vec![
                        hold(self, call("zeros", vec![rows.clone(), k.clone()]), out),
                        hold(self, call("zeros", vec![k.clone()]), out),
                        hold(self, call("zeros", vec![k, cols]), out),
                    ]
                } else {
                    let square = if callee == "eig_herm" { "czeros" } else { "zeros" };
                    vec![hold(self, call("zeros", vec![rows.clone()]), out), hold(self, call(square, vec![rows.clone(), rows]), out)]
                };
                let mut full = vec![t];
                full.extend(outs.iter().cloned());
                out.push(Stmt::ExprStmt { expr: call(&callee, full), span });
                outs
            }
            Expr::Call { callee, span: vspan, .. } if callee == "value_and_grad" => {
                return Err(Diagnostic::error("`value_and_grad` unpacks into two names")
                    .with_primary(vspan, "two results")
                    .with_note("write `loss, g = value_and_grad(f, x, ...)`"));
            }
            e => {
                // an array: exactly one element for each name
                self.check(&e, cur)?;
                let sp = e.span();
                self.hidden += 1;
                let t = format!("{UNPACKED}{}", self.hidden);
                out.push(Stmt::Assign { name: t.clone(), name_span: sp, value: e, span: sp });
                let len = Expr::Call { callee: "len".into(), args: vec![var(&t, sp)], span: sp };
                out.push(Stmt::If {
                    cond: Expr::Binary { op: BinOp::Ne, lhs: Box::new(len), rhs: Box::new(Expr::Num { value: n as f64, span: sp }), span: sp },
                    then_body: vec![Stmt::Lens {
                        lens: Box::new(Lens::Stop { message: format!("cannot unpack this array into {n} names: its length is not {n}"), kind: crate::errkind::ErrKind::Value }),
                        span: sp,
                    }],
                    else_body: Vec::new(),
                    span: sp,
                });
                (0..n).map(|i| Expr::Index { base: Box::new(var(&t, sp)), indices: vec![Expr::Num { value: i as f64, span: sp }], span: sp }).collect()
            }
        };
        for (target, value) in targets.into_iter().zip(values) {
            out.push(assign_target(target, value, span));
        }
        Ok(())
    }
}

/// `target = value`, for a target `unpack` has checked.
fn assign_target(target: Expr, value: Expr, span: Span) -> Stmt {
    match target {
        Expr::Var { name, span: name_span } => Stmt::Assign { name, name_span, value, span },
        Expr::Index { base, indices, .. } => Stmt::SetIndex { base, indices, value, span },
        Expr::Field { base, field, field_span, .. } => Stmt::SetField { base, field, field_span, value, span },
        Expr::Call { mut args, .. } => {
            args.push(value);
            Stmt::ExprStmt { expr: Expr::Call { callee: crate::interp::slices::SET.to_string(), args, span }, span }
        }
        _ => unreachable!("targets are checked first"),
    }
}
