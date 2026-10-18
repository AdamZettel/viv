// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Operations over a record's arrays:
//! `a + b`, `s * r`, `-r`, `sqrt(r)`, `zeros_like(r)`, `sum(r)` and
//! `for_leaves(f, r1, r2, ...)`, for a record or an array of records.
//!
//! They take a record of arrays: every field an array, a record of arrays,
//! or an array of them, at any depth. A number, a boolean or a string field
//! is refused, with the field named: `p + q` of two `Point { x: Num, ... }`
//! would otherwise have to choose between adding the numbers and keeping
//! them as the record's structure (a size), and either choice surprises
//! someone. Sizes go in a record of their own, passed beside the record of
//! arrays.
//!
//! The compiled engines run each as a def the lowering writes in Viv, one
//! per operation and type (`viv__rec_add_Gpt`, `viv__recs_add_Block`, ...),
//! made after the program's own defs are lowered and lowered like them; the
//! C and the wasm backends see ordinary defs. The interpreter walks the
//! record itself (`Interp::rec_zip`), the same way: arrays combined by the
//! array operation, a record or an array of records by the same operation
//! on its fields or elements, in declaration order.

use super::*;
use crate::ast::Param;
use std::collections::BTreeMap;

/// An operation the lowering writes a def for.
#[derive(Clone, Debug)]
pub(super) enum RecFn {
    /// `a op b` for `+ - * /`; a side that is a number is marked
    Bin { op: BinOp, ty: Ty, lnum: bool, rnum: bool },
    /// `-r`
    Neg { ty: Ty },
    /// an elementwise math function (`sqrt`, `exp`, ...) of each array
    Map { f: String, ty: Ty },
    /// `zeros_like(r)`: the same structure, every array zero
    Zeros { ty: Ty },
    /// `sum(r)`: the fields' sums, added in declaration order
    Sum { ty: Ty },
    /// `for_leaves(f, r1, ..., rk, extra...)`: `f` called on each array of
    /// the `k` records at once, with the extra arguments after
    Leaves { f: String, ty: Ty, k: usize, extra: Vec<Ty> },
}

fn rec_of(ty: Ty) -> &'static str {
    match ty {
        Ty::Record(r) | Ty::Recs(r) => r,
        _ => unreachable!("a record operation is on a record"),
    }
}

impl RecFn {
    fn ty(&self) -> Ty {
        match self {
            RecFn::Bin { ty, .. } | RecFn::Neg { ty } | RecFn::Map { ty, .. } | RecFn::Zeros { ty } | RecFn::Sum { ty } | RecFn::Leaves { ty, .. } => *ty,
        }
    }

    /// The def's name: `viv__rec_<what>_<T>` for a record, `viv__recs_...`
    /// for an array of them.
    pub(super) fn name(&self) -> String {
        let kind = if matches!(self.ty(), Ty::Recs(_)) { "recs" } else { "rec" };
        let what = match self {
            RecFn::Bin { op, lnum, rnum, .. } => {
                let o = match op {
                    BinOp::Add => "add",
                    BinOp::Sub => "sub",
                    BinOp::Mul => "mul",
                    _ => "div",
                };
                if *lnum {
                    format!("n{o}")
                } else if *rnum {
                    format!("{o}n")
                } else {
                    o.to_string()
                }
            }
            RecFn::Neg { .. } => "neg".to_string(),
            RecFn::Map { f, .. } => format!("map_{f}"),
            RecFn::Zeros { .. } => "zeros".to_string(),
            RecFn::Sum { .. } => "sum".to_string(),
            RecFn::Leaves { f, k, .. } => format!("leaves{k}_{f}"),
        };
        format!("viv__{kind}_{what}_{}", rec_of(self.ty()))
    }

    /// The same operation on a field's (or an element's) type.
    fn on(&self, ty: Ty) -> RecFn {
        let mut r = self.clone();
        match &mut r {
            RecFn::Bin { ty: t, .. } | RecFn::Neg { ty: t } | RecFn::Map { ty: t, .. } | RecFn::Zeros { ty: t } | RecFn::Sum { ty: t } | RecFn::Leaves { ty: t, .. } => *t = ty,
        }
        r
    }

    /// How the operation is named in an error: the interpreter's words.
    fn word(&self) -> String {
        match self {
            RecFn::Bin { op, .. } => format!("`{}`", op.c_op()),
            RecFn::Neg { .. } => "`-`".to_string(),
            RecFn::Map { f, .. } => format!("`{f}`"),
            RecFn::Zeros { .. } => "`zeros_like`".to_string(),
            RecFn::Sum { .. } => "`sum`".to_string(),
            RecFn::Leaves { .. } => "`for_leaves`".to_string(),
        }
    }

    fn params(&self) -> Vec<Ty> {
        let ty = self.ty();
        match self {
            RecFn::Bin { lnum, rnum, .. } => vec![if *lnum { Ty::Num } else { ty }, if *rnum { Ty::Num } else { ty }],
            RecFn::Leaves { k, extra, .. } => std::iter::repeat(ty).take(*k).chain(extra.iter().copied()).collect(),
            _ => vec![ty],
        }
    }

    fn ret(&self) -> Ty {
        match self {
            RecFn::Sum { .. } => Ty::Num,
            RecFn::Leaves { .. } => Ty::Void,
            _ => self.ty(),
        }
    }
}

/// The fields of the record type `r`.
fn fields_of(sigs: &Sigs, r: &str) -> Vec<(String, Ty)> {
    sigs.records.get(r).map(|i| i.fields.clone()).unwrap_or_default()
}

/// Refuse a record that is not a record of arrays: a field that is not a
/// real array (or, for `+ - * /` and `-r`, a complex one), a record of
/// arrays or an array of them.
fn check_walkable(sigs: &Sigs, rf: &RecFn, ty: Ty, span: Span) -> Result<(), Diagnostic> {
    let rec = rec_of(ty);
    for (f, ft) in fields_of(sigs, rec) {
        let ok = match ft {
            Ty::Arr(Elem::Num) => true,
            Ty::Arr(Elem::Cplx) => matches!(rf, RecFn::Bin { .. } | RecFn::Neg { .. }),
            Ty::Record(_) | Ty::Recs(_) => {
                check_walkable(sigs, rf, ft, span)?;
                true
            }
            _ => false,
        };
        if !ok {
            return Err(Diagnostic::error(not_arrays(&rf.word(), rec, &f, ft.name()))
                .with_primary(span, "this record")
                .with_note("an operation over a record takes a record of arrays; keep numbers such as sizes in a record of their own, passed beside it"));
        }
    }
    Ok(())
}

/// Ask for the def of `rf` and call it on `kids`: a node of the operation's
/// result type. The def is written after the program's defs are lowered.
pub(super) fn request(prog: &mut IrProgram<Viv>, sigs: &Sigs, rf: RecFn, kids: Vec<NodeId>, span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    check_walkable(sigs, &rf, rf.ty(), span)?;
    let name = rf.name();
    let ret = rf.ret();
    sigs.rec_fns.borrow_mut().entry(name.clone()).or_insert(rf);
    Ok((prog.add(Viv::Call { callee: name }, span, kids), ret))
}

/// `a op b` with a record or an array of records on either side: both
/// records of one type, or a record and a number.
pub(super) fn binary(prog: &mut IrProgram<Viv>, sigs: &Sigs, op: BinOp, l: (NodeId, Ty, Span), r: (NodeId, Ty, Span), span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
        return Err(Diagnostic::error(REC_OPERATORS).with_primary(span, "an operator over records"));
    }
    let is_rec = |t: Ty| matches!(t, Ty::Record(_) | Ty::Recs(_));
    let ty = if is_rec(l.1) { l.1 } else { r.1 };
    let word = format!("`{}`", op.c_op());
    for (_, t, sp) in [l, r] {
        if !(t == ty || t == Ty::Num) {
            return Err(Diagnostic::error(format!("{word} takes records of one type, or a record and a number; the first is a {} and this is a {}", ty.name(), t.name()))
                .with_primary(sp, "a different type"));
        }
    }
    let rf = RecFn::Bin { op, ty, lnum: l.1 == Ty::Num, rnum: r.1 == Ty::Num };
    request(prog, sigs, rf, vec![l.0, r.0], span)
}

/// Every def `wanted` needs: itself, and the same operation on each record
/// and array-of-records field, at any depth.
fn closure(sigs: &Sigs, wanted: Vec<RecFn>) -> BTreeMap<String, RecFn> {
    let mut out = BTreeMap::new();
    let mut stack = wanted;
    while let Some(rf) = stack.pop() {
        let name = rf.name();
        if out.contains_key(&name) {
            continue;
        }
        match rf.ty() {
            Ty::Recs(r) => stack.push(rf.on(Ty::Record(r))),
            Ty::Record(r) => {
                for (_, ft) in fields_of(sigs, r) {
                    if matches!(ft, Ty::Record(_) | Ty::Recs(_)) {
                        stack.push(rf.on(ft));
                    }
                }
            }
            _ => {}
        }
        out.insert(name, rf);
    }
    out
}

/// Write and lower the defs asked for while lowering the program (and the
/// ones those need), each once.
pub(super) fn emit(prog: &mut IrProgram<Viv>, sigs: &mut Sigs, made: &mut std::collections::HashSet<String>) -> Result<(), Diagnostic> {
    loop {
        let wanted: Vec<RecFn> = sigs.rec_fns.borrow().iter().filter(|(n, _)| !made.contains(*n)).map(|(_, r)| r.clone()).collect();
        if wanted.is_empty() {
            return Ok(());
        }
        let all = closure(sigs, wanted);
        // every signature first: the defs call each other by name
        for (name, rf) in &all {
            if made.contains(name) {
                continue;
            }
            let params = rf.params();
            sigs.arity.insert(name.clone(), params.len());
            sigs.params.insert(name.clone(), params.iter().map(|t| Some(*t)).collect());
            sigs.ret.insert(name.clone(), Some(rf.ret()));
            if let Ty::Recs(_) = rf.ty() {
                if !matches!(rf, RecFn::Sum { .. } | RecFn::Leaves { .. }) {
                    sigs.local_tys.insert(name.clone(), HashMap::from([("out".to_string(), rf.ty())]));
                }
            }
            sigs.rec_fns.borrow_mut().entry(name.clone()).or_insert_with(|| rf.clone());
        }
        for (name, rf) in all {
            if !made.insert(name.clone()) {
                continue;
            }
            let def = def_of(&name, &rf, sigs);
            let params: Vec<Option<Ty>> = rf.params().into_iter().map(Some).collect();
            let id = lower_def_as(prog, &def, sigs, &params, Some(rf.ret()))?;
            prog.add_root(id);
        }
    }
}

// ---- the defs, written as Viv -----------------------------------------------

fn sp() -> Span {
    Span::synthetic(SourceId(0))
}
fn v(n: &str) -> Expr {
    Expr::Var { name: n.to_string(), span: sp() }
}
fn fld(base: Expr, f: &str) -> Expr {
    Expr::Field { base: Box::new(base), field: f.to_string(), field_span: sp(), span: sp() }
}
fn at(base: Expr, i: Expr) -> Expr {
    Expr::Index { base: Box::new(base), indices: vec![i], span: sp() }
}
fn call(f: &str, args: Vec<Expr>) -> Expr {
    Expr::Call { callee: f.to_string(), args, span: sp() }
}
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary { op, lhs: Box::new(a), rhs: Box::new(b), span: sp() }
}
fn num(x: f64) -> Expr {
    Expr::Num { value: x, span: sp() }
}
fn assign(n: &str, value: Expr) -> Stmt {
    Stmt::Assign { name: n.to_string(), name_span: sp(), value, span: sp() }
}
fn stop_if(cond: Expr, message: String) -> Stmt {
    Stmt::If { cond, then_body: vec![Stmt::Lens { lens: Box::new(Lens::Stop { message, kind: crate::errkind::ErrKind::Value }), span: sp() }], else_body: Vec::new(), span: sp() }
}

/// The parameter names of the def: `a`, `b` for the two sides, `a0 a1 ...`
/// for the records of `for_leaves` and `e0 e1 ...` for its extra arguments.
fn param_names(rf: &RecFn) -> Vec<String> {
    match rf {
        RecFn::Bin { .. } => vec!["a".into(), "b".into()],
        RecFn::Leaves { k, extra, .. } => (0..*k).map(|i| format!("a{i}")).chain((0..extra.len()).map(|i| format!("e{i}"))).collect(),
        _ => vec!["a".into()],
    }
}

fn def_of(name: &str, rf: &RecFn, sigs: &Sigs) -> Def {
    let names = param_names(rf);
    let params = names.iter().map(|n| Param { name: n.clone(), span: sp(), ty: None }).collect();
    let body = match rf.ty() {
        Ty::Recs(r) => recs_body(rf, r, &names),
        Ty::Record(r) => record_body(rf, r, &names, sigs),
        _ => unreachable!(),
    };
    Def { name: name.to_string(), name_span: sp(), params, ret: None, body, span: sp() }
}

/// The record operands among the parameters (not a number side, not an
/// extra argument of `for_leaves`).
fn record_params(rf: &RecFn, names: &[String]) -> Vec<usize> {
    match rf {
        RecFn::Bin { lnum, rnum, .. } => [(0, !*lnum), (1, !*rnum)].into_iter().filter(|(_, r)| *r).map(|(i, _)| i).collect(),
        RecFn::Leaves { k, .. } => (0..*k).collect(),
        _ => (0..names.len()).collect(),
    }
}

/// An array of records: the operation on each element, the arrays first
/// required to have one length.
fn recs_body(rf: &RecFn, r: &str, names: &[String]) -> Vec<Stmt> {
    let recs = record_params(rf, names);
    let first = &names[recs[0]];
    let mut body = Vec::new();
    for &k in &recs[1..] {
        body.push(stop_if(bin(BinOp::Ne, call("len", vec![v(&names[k])]), call("len", vec![v(first)])), rec_lengths_differ(&rf.word(), r)));
    }
    let elem = rf.on(Ty::Record(Box::leak(r.to_string().into_boxed_str())));
    let args = |i: &str| -> Vec<Expr> {
        names.iter().enumerate().map(|(k, n)| if recs.contains(&k) { at(v(n), v(i)) } else { v(n) }).collect()
    };
    let one = call(&elem.name(), args("i"));
    let loop_body = match rf {
        RecFn::Sum { .. } => vec![assign("s", bin(BinOp::Add, v("s"), one))],
        RecFn::Leaves { .. } => vec![Stmt::ExprStmt { expr: one, span: sp() }],
        _ => vec![assign("out", call("append", vec![v("out"), one]))],
    };
    match rf {
        RecFn::Sum { .. } => body.push(assign("s", num(0.0))),
        RecFn::Leaves { .. } => {}
        _ => body.push(assign("out", Expr::ArrLit { elems: Vec::new(), span: sp() })),
    }
    body.push(Stmt::For { var: "i".into(), var_span: sp(), bound: call("len", vec![v(first)]), body: loop_body, parallel: false, span: sp() });
    match rf {
        RecFn::Sum { .. } => body.push(Stmt::Return { value: v("s"), span: sp() }),
        RecFn::Leaves { .. } => {}
        _ => body.push(Stmt::Return { value: v("out"), span: sp() }),
    }
    body
}

/// A record: the operation on each array field, the same operation's def on
/// each record field, and each other field kept (and, for two records,
/// required equal), field by field in declaration order, as the interpreter
/// walks them; then the record of the results.
fn record_body(rf: &RecFn, r: &str, names: &[String], sigs: &Sigs) -> Vec<Stmt> {
    let recs = record_params(rf, names);
    let first = &names[recs[0]];
    let mut body = Vec::new();
    let arg = |k: usize, f: &str| -> Expr { if recs.contains(&k) { fld(v(&names[k]), f) } else { v(&names[k]) } };
    let mut values: Vec<(String, Span, Expr)> = Vec::new();
    // a field's value, into a local of its own, in declaration order
    fn keep(body: &mut Vec<Stmt>, values: &mut Vec<(String, Span, Expr)>, f: &str, e: Expr) {
        let t = format!("f_{f}");
        body.push(assign(&t, e));
        values.push((f.to_string(), sp(), v(&t)));
    }
    if let RecFn::Sum { .. } = rf {
        body.push(assign("s", num(0.0)));
    }
    for (f, ft) in fields_of(sigs, r) {
        match ft {
            Ty::Arr(_) => match rf {
                RecFn::Bin { op, .. } => keep(&mut body, &mut values, &f, bin(*op, arg(0, &f), arg(1, &f))),
                RecFn::Neg { .. } => keep(&mut body, &mut values, &f, Expr::Neg { operand: Box::new(fld(v(first), &f)), span: sp() }),
                RecFn::Map { f: m, .. } => keep(&mut body, &mut values, &f, call(m, vec![fld(v(first), &f)])),
                RecFn::Zeros { .. } => keep(&mut body, &mut values, &f, call("zeros_like", vec![fld(v(first), &f)])),
                RecFn::Sum { .. } => body.push(assign("s", bin(BinOp::Add, v("s"), call("sum", vec![fld(v(first), &f)])))),
                RecFn::Leaves { f: g, .. } => {
                    let args = (0..names.len()).map(|k| arg(k, &f)).collect();
                    body.push(Stmt::ExprStmt { expr: call(g, args), span: sp() });
                }
            },
            Ty::Record(_) | Ty::Recs(_) => {
                let sub = rf.on(ft).name();
                let args: Vec<Expr> = (0..names.len()).map(|k| arg(k, &f)).collect();
                match rf {
                    RecFn::Sum { .. } => body.push(assign("s", bin(BinOp::Add, v("s"), call(&sub, args)))),
                    RecFn::Leaves { .. } => body.push(Stmt::ExprStmt { expr: call(&sub, args), span: sp() }),
                    _ => keep(&mut body, &mut values, &f, call(&sub, args)),
                }
            }
            _ => unreachable!("check_walkable refuses a field that is not an array or a record of arrays"),
        }
    }
    match rf {
        RecFn::Sum { .. } => body.push(Stmt::Return { value: v("s"), span: sp() }),
        RecFn::Leaves { .. } => {}
        _ => body.push(Stmt::Return { value: Expr::RecordLit { type_name: r.to_string(), type_span: sp(), fields: values, span: sp() }, span: sp() }),
    }
    body
}
