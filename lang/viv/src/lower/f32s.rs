// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `Float32` and the packed arrays in the checked lowering and the C backend.
//!
//! A Float32 operation is a call of the runtime (`viv_f_*`), a packed array
//! operation one of `viv_p_*`. In C each is a small static function on
//! `float` (inlined); in wasm the arithmetic is the f32 instruction and the
//! rest is the runtime (`vrt_f_*`, `vrt_p_*`, interp/wasm_rt.rs), which runs
//! interp/f32s.rs, the interpreter's own code. A literal beside a Float32
//! becomes a Float32 here, and its span is recorded (`LITS`) so the
//! interpreter, which reads the tree, makes the same literals Float32s.

use super::*;
use crate::ast::Store;

thread_local! {
    /// The spans of the number literals this lowering made Float32s.
    static LITS: std::cell::RefCell<std::collections::HashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
    /// A Float32 literal refused where it was bound (`adapt_or_note`),
    /// reported with the lowering's other errors.
    static ERRS: std::cell::RefCell<Vec<Diagnostic>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The spans of the array literals this lowering packed, and their kind.
    static ARRS: std::cell::RefCell<HashMap<(u32, u32), Store>> = std::cell::RefCell::new(Default::default());
}

/// Start lowering a program: nothing decided yet.
pub(super) fn begin() {
    LITS.with(|d| d.borrow_mut().clear());
    ARRS.with(|d| d.borrow_mut().clear());
    ERRS.with(|d| d.borrow_mut().clear());
}

/// The refusals `adapt_or_note` noted.
pub(super) fn take_errors() -> Vec<Diagnostic> {
    ERRS.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

/// `adapt`, with a refusal noted for the end of the lowering.
pub(super) fn adapt_or_note(prog: &mut IrProgram<Viv>, got: (NodeId, Ty), want: Ty, e: &Expr) -> (NodeId, Ty) {
    match adapt(prog, got, want, e) {
        Ok(r) => r,
        Err(d) => {
            ERRS.with(|x| x.borrow_mut().push(d));
            got
        }
    }
}

/// The forms of an expression with a Float32 or a packed array in them:
/// `-x`, `a[i]`, `a op b`, the builtins, and `print` of them. `None` hands
/// the expression back to `lower_expr_inner` unchanged (what was lowered
/// here is content-addressed, so lowering it again adds nothing). Out of
/// line, so the recursive lowering's frame does not grow.
#[inline(never)]
pub(super) fn expr_hook(prog: &mut IrProgram<Viv>, expr: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    match expr {
        Expr::Neg { operand, span } if peek_f32(operand, scope, sigs) => {
            let (oid, ty) = match lower_expr(prog, operand, scope, sigs) {
                Ok(r) => r,
                Err(d) => return Some(Err(d)),
            };
            (ty == Ty::F32).then(|| Ok((call(prog, "viv_f_neg", *span, vec![oid]), Ty::F32)))
        }
        Expr::Index { base, indices, span } if peek_arr(base, scope, sigs).is_some() => {
            let r = (|| {
                let (bid, bty) = lower_expr(prog, base, scope, sigs)?;
                let Ty::Arr(Elem::P(k)) = bty else { return Ok(None) };
                let mut kids = vec![bid];
                kids.extend(index_kids(prog, indices, scope, sigs)?);
                lower_get(prog, k, kids, *span).map(Some)
            })();
            r.transpose()
        }
        Expr::Binary { op, lhs, rhs, span } if !matches!(op, BinOp::And | BinOp::Or) && (peek_f32(lhs, scope, sigs) || peek_f32(rhs, scope, sigs)) => {
            let r = (|| {
                let (lid, lty) = lower_expr(prog, lhs, scope, sigs)?;
                let (rid, rty) = lower_expr(prog, rhs, scope, sigs)?;
                combine(prog, *op, (lid, lty, lhs), (rid, rty, rhs), *span)
            })();
            r.transpose()
        }
        Expr::Call { callee, args, span } if !sigs.arity.contains_key(callee.as_str()) => {
            if callee == "print" && args.iter().any(|a| may_be_mine(a, scope, sigs) || matches!(a, Expr::Call { callee: c, args: fa, .. } if c == "fmt" && fa.first().is_some_and(|x| may_be_mine(x, scope, sigs)))) {
                let wrote: Vec<Expr> = args.iter().map(|a| print_arg(a, scope, sigs)).collect();
                // nothing rewritten: the print as it is (no second visit)
                if format!("{wrote:?}") == format!("{args:?}") {
                    return None;
                }
                let args = wrote;
                return Some(lower_expr(prog, &Expr::Call { callee: callee.clone(), args, span: *span }, scope, sigs));
            }
            lower_call(prog, callee, args, *span, scope, sigs)
        }
        _ => None,
    }
}

/// `a[i] = v` of a packed array and `show x` of a Float32 or packed array;
/// `None` for every other statement. Out of line, as `expr_hook` is.
#[inline(never)]
pub(super) fn stmt_hook(prog: &mut IrProgram<Viv>, stmt: &Stmt, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<NodeId, Diagnostic>> {
    match stmt {
        Stmt::SetIndex { base, indices, value, span } if peek_arr(base, scope, sigs).is_some() => {
            let r = (|| {
                let (bid, bty) = lower_expr(prog, base, scope, sigs)?;
                let Ty::Arr(Elem::P(k)) = bty else { return Ok(None) };
                let mut kids = vec![bid];
                kids.extend(index_kids(prog, indices, scope, sigs)?);
                lower_store(prog, k, kids, value, *span, scope, sigs).map(Some)
            })();
            r.transpose()
        }
        Stmt::Show { expr, span } if may_be_mine(expr, scope, sigs) => {
            let (eid, ty) = match lower_expr(prog, expr, scope, sigs) {
                Ok(r) => r,
                Err(d) => return Some(Err(d)),
            };
            if !matches!(ty, Ty::F32 | Ty::Arr(Elem::P(_))) {
                return None;
            }
            let label = match expr {
                Expr::Var { name, .. } if !name.starts_with(crate::parser::HIDDEN_PREFIX) => name.clone(),
                _ => String::new(),
            };
            let mut parts = Vec::new();
            if !label.is_empty() {
                parts.push(PrintPart::Text(format!("{label}: ")));
            }
            parts.push(PrintPart::Str);
            let text = bare_text(prog, eid, ty, *span);
            let pid = prog.add(Viv::Print { parts }, *span, vec![text]);
            Some(Ok(prog.add(Viv::ExprStmt, *span, vec![pid])))
        }
        _ => None,
    }
}

/// The hidden builtins `print_arg` writes: a Float32 or packed array as its
/// bare text, a Float32 as its shortest text.
const BARE: &str = "viv__f32_bare";
const REPR: &str = "viv__f32_repr";

/// A `print` argument with a Float32 or packed array in it, rewritten to
/// what `print` already writes: its text (bare or `fmt(x)`), or, in a
/// format spec, the number it is.
fn print_arg(a: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Expr {
    let probe = |e: &Expr| lower_expr(&mut IrProgram::new(), e, scope, sigs).ok().map(|(_, t)| t);
    let mine = |t: Option<Ty>| matches!(t, Some(Ty::F32 | Ty::Arr(Elem::P(_))));
    let wrap = |name: &str, e: &Expr| Expr::Call { callee: name.into(), args: vec![e.clone()], span: e.span() };
    match a {
        Expr::Call { callee, args, span } if callee == "fmt" && args.first().is_some_and(|x| probe(x) == Some(Ty::F32)) => {
            if args.len() == 1 {
                return wrap(REPR, &args[0]);
            }
            let mut wide = args.clone();
            wide[0] = wrap("float", &args[0]);
            Expr::Call { callee: callee.clone(), args: wide, span: *span }
        }
        Expr::Str { .. } => a.clone(),
        e if mine(probe(e)) => wrap(BARE, e),
        e => e.clone(),
    }
}

/// The words a program needs to hold a Float32 or a packed array at all.
pub fn mentions(src: &str) -> bool {
    ["Float32", "BFloat16", "Float16", "Int8", "Int16", "Int32", "UInt8"]
        .iter()
        .chain(Store::ALL.iter().map(|k| k.conv()).collect::<Vec<_>>().iter())
        .any(|w| src.contains(w))
}

/// What the interpreter needs from the lowering of `src`: the literals made
/// Float32s and the array literals packed.
pub fn spans(src: &str) -> (std::collections::HashSet<(u32, u32)>, HashMap<(u32, u32), Store>) {
    if !mentions(src) {
        return Default::default();
    }
    let _ = crate::build_ir(src, knotkit::span::SourceId(0));
    (LITS.with(|d| d.borrow().clone()), ARRS.with(|d| d.borrow().clone()))
}

pub(super) fn arr_name(k: Store) -> &'static str {
    match k {
        Store::F32 => "Float32 array",
        Store::BF16 => "BFloat16 array",
        Store::F16 => "Float16 array",
        Store::I8 => "Int8 array",
        Store::I16 => "Int16 array",
        Store::I32 => "Int32 array",
        Store::U8 => "UInt8 array",
    }
}

fn call(prog: &mut IrProgram<Viv>, name: &str, span: Span, kids: Vec<NodeId>) -> NodeId {
    prog.add(Viv::Call { callee: name.to_string() }, span, kids)
}

fn kind_node(prog: &mut IrProgram<Viv>, k: Store, span: Span) -> NodeId {
    prog.add(Viv::Num(k.code() as f64), span, vec![])
}

/// The value of a number literal, `-` applied: `0.5`, `-2`.
fn literal(e: &Expr) -> Option<(f64, Span)> {
    match e {
        Expr::Num { value, span } => Some((*value, *span)),
        Expr::Neg { operand, .. } => match &**operand {
            Expr::Num { value, span } => Some((-*value, *span)),
            _ => None,
        },
        _ => None,
    }
}

/// `v` is exactly halfway between two float32s: rounding the decimal text
/// to a double first may have moved it there, so its nearest float32 is not
/// known from the double alone.
fn midpoint(v: f64) -> bool {
    let f = v as f32;
    if !f.is_finite() || f as f64 == v {
        return false;
    }
    let (lo, hi) = if (f as f64) < v { (f, f32::from_bits(f.to_bits() + if f >= 0.0 { 1 } else { u32::MAX })) } else { (f32::from_bits(f.to_bits() - if f > 0.0 { 1 } else { u32::MAX }), f) };
    (lo as f64 + hi as f64) / 2.0 == v || ((lo as f64) - v).abs() == ((hi as f64) - v).abs()
}

/// The literal `e` as a Float32 node.
fn lower_lit(prog: &mut IrProgram<Viv>, e: &Expr) -> Result<Option<NodeId>, Diagnostic> {
    let Some((v, at)) = literal(e) else { return Ok(None) };
    if midpoint(v) {
        return Err(Diagnostic::error("this literal is exactly halfway between two Float32 values once read as a double, so which one it means is not known")
            .with_primary(e.span(), "a Float32 literal")
            .with_note("write it with fewer digits, or as float32(...) of the number meant"));
    }
    LITS.with(|d| d.borrow_mut().insert((at.start, at.end)));
    let n = prog.add(Viv::Num(v as f32 as f64), e.span(), vec![]);
    Ok(Some(call(prog, "viv_f_of", e.span(), vec![n])))
}

/// A value bound where a `want` goes (a variable, a parameter, a return, a
/// field): a literal where a Float32 goes becomes one.
pub(super) fn adapt(prog: &mut IrProgram<Viv>, got: (NodeId, Ty), want: Ty, e: &Expr) -> Result<(NodeId, Ty), Diagnostic> {
    if want == Ty::F32 && got.1 == Ty::Num {
        if let Some(id) = lower_lit(prog, e)? {
            return Ok((id, Ty::F32));
        }
    }
    Ok(got)
}

pub(super) fn no_mix(op: BinOp, span: Span, num: Ty) -> Diagnostic {
    no_mix_in(op_word(op), span, num)
}

fn no_mix_in(op: &str, span: Span, num: Ty) -> Diagnostic {
    let fix = if num == Ty::Int { "float32(n)" } else { "float32(y)" };
    Diagnostic::error(format!("a Float32 and {} do not mix in {op}: write {fix} to round the {}, or float(x) to widen the Float32", if num == Ty::Int { "an Int" } else { "a number (float64)" }, if num == Ty::Int { "Int" } else { "number" }))
        .with_primary(span, "mixed precision")
}

/// `a op b` with a Float32 on at least one side; `None` when neither side is
/// one.
pub(super) fn combine(prog: &mut IrProgram<Viv>, op: BinOp, l: (NodeId, Ty, &Expr), r: (NodeId, Ty, &Expr), span: Span) -> Result<Option<(NodeId, Ty)>, Diagnostic> {
    if l.1 != Ty::F32 && r.1 != Ty::F32 {
        return Ok(None);
    }
    let side = |prog: &mut IrProgram<Viv>, (id, ty, e): (NodeId, Ty, &Expr)| -> Result<NodeId, Diagnostic> {
        if ty == Ty::F32 {
            return Ok(id);
        }
        if matches!(ty, Ty::Num | Ty::Int) {
            if let Some(n) = lower_lit(prog, e)? {
                return Ok(n);
            }
            return Err(no_mix(op, e.span(), ty));
        }
        Err(Diagnostic::error(format!("a Float32 and {} do not mix", art(ty))).with_primary(e.span(), format!("this is {}", art(ty))))
    };
    let a = side(prog, l)?;
    let b = side(prog, r)?;
    use BinOp::*;
    let f = match op {
        Add => "viv_f_add",
        Sub => "viv_f_sub",
        Mul => "viv_f_mul",
        Div => "viv_f_div",
        Lt | Le | Gt | Ge | EqEq | Ne => {
            // a comparison of the two widened: the same answer, exactly
            let wa = call(prog, "viv_f_wide", l.2.span(), vec![a]);
            let wb = call(prog, "viv_f_wide", r.2.span(), vec![b]);
            return Ok(Some((prog.add(Viv::Bin { op, cplx: false }, span, vec![wa, wb]), Ty::Bool)));
        }
        _ => {
            return Err(Diagnostic::error(format!("`{}` is not defined on Float32 in this version", op_word(op)))
                .with_primary(span, "a Float32 operation")
                .with_note("+ - * / and comparisons are; write float(x) for the float64 operation"))
        }
    };
    Ok(Some((call(prog, f, span, vec![a, b]), Ty::F32)))
}

fn op_word(op: BinOp) -> &'static str {
    use BinOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
        Mod => "%",
        FloorDiv => "//",
        Matmul => "@",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        EqEq => "==",
        Ne => "!=",
        And => "and",
        Or => "or",
    }
}

/// An index as a number node (an Int converted exactly).
fn index_num(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, span: Span) -> NodeId {
    ints::to_num(prog, id, ty, span).0
}

/// `a[i]` of a packed array: a Float32 or an Int.
pub(super) fn lower_get(prog: &mut IrProgram<Viv>, k: Store, kids: Vec<NodeId>, span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    if kids.len() != 2 {
        return Err(Diagnostic::error(format!("{} has rank 1, so it takes one index", art(Ty::Arr(Elem::P(k))))).with_primary(span, format!("{} indices given", kids.len() - 1)));
    }
    let kn = kind_node(prog, k, span);
    let mut kids = kids;
    kids.push(kn);
    Ok(if k.is_float() { (call(prog, "viv_p_getf", span, kids), Ty::F32) } else { (call(prog, "viv_p_geti", span, kids), Ty::Int) })
}

/// The index children of a store or read lowered as numbers.
pub(super) fn index_kids(prog: &mut IrProgram<Viv>, indices: &[Expr], scope: &Scope<'_>, sigs: &Sigs) -> Result<Vec<NodeId>, Diagnostic> {
    let mut out = Vec::new();
    for ix in indices {
        let (id, ty) = lower_expr(prog, ix, scope, sigs)?;
        expect_index(ty, ix.span())?;
        out.push(index_num(prog, id, ty, ix.span()));
    }
    Ok(out)
}

/// `a[i] = v` of a packed array: a float kind takes a Float32 (a literal
/// adapts) and rounds it; an integer kind takes an Int and checks it.
pub(super) fn lower_store(prog: &mut IrProgram<Viv>, k: Store, kids: Vec<NodeId>, value: &Expr, span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    if kids.len() != 2 {
        return Err(Diagnostic::error(format!("{} has rank 1, so it takes one index", art(Ty::Arr(Elem::P(k))))).with_primary(span, format!("{} indices given", kids.len() - 1)));
    }
    let mut kids = kids;
    let f = if k.is_float() {
        let (vid, vty) = lower_expr(prog, value, scope, sigs)?;
        let (vid, vty) = adapt(prog, (vid, vty), Ty::F32, value)?;
        if vty != Ty::F32 {
            return Err(Diagnostic::error(format!("{} holds Float32s, and this is {}", art(Ty::Arr(Elem::P(k))), art(vty)))
                .with_primary(value.span(), "not a Float32")
                .with_note("write float32(x) to round a number to a Float32"));
        }
        kids.push(vid);
        "viv_p_setf"
    }
    else {
        let vid = ints::lower_as_int(prog, value, scope, sigs)?;
        kids.push(vid);
        "viv_p_seti"
    };
    let kn = kind_node(prog, k, span);
    kids.push(kn);
    let set = call(prog, f, span, kids);
    Ok(prog.add(Viv::ExprStmt, span, vec![set]))
}

/// `show x` and `print(x)`: a Float32 as its bare number, a packed array as
/// `[0.1, 0.25, 3]`; a string node.
pub(super) fn bare_text(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, span: Span) -> NodeId {
    match ty {
        Ty::Arr(Elem::P(k)) => {
            let kn = kind_node(prog, k, span);
            call(prog, "viv_p_str", span, vec![id, kn])
        }
        _ => call(prog, "viv_f_bare", span, vec![id]),
    }
}

/// The builtins with a Float32 or packed form. `None` when the call is not
/// one of them (nothing has been added to `prog` then).
pub(super) fn lower_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    if callee == BARE || callee == REPR {
        let r = (|| {
            let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
            Ok((if callee == REPR { call(prog, "viv_f_repr", span, vec![id]) } else { bare_text(prog, id, ty, span) }, Ty::Str))
        })();
        return Some(r);
    }
    let ours = Store::of_zeros(callee).is_some() || Store::of_conv(callee).is_some() || callee == "int64";
    let shared = matches!(callee, "float" | "sqrt" | "abs" | "min" | "max" | "sum" | "dot" | "len" | "fmt" | "str") || crate::interp::f32s::FUNCS.contains(&callee);
    if !ours && !shared {
        return None;
    }
    if !ours {
        // only when an argument is a Float32 or a packed array; the syntax
        // says first whether one can be, so a program with none lowers
        // nothing twice
        if !args.iter().take(2).any(|a| may_be_mine(a, scope, sigs)) {
            return None;
        }
        let probe = |e: &Expr| lower_expr(&mut IrProgram::new(), e, scope, sigs).ok().map(|(_, t)| t);
        let mine = |t: Option<Ty>| matches!(t, Some(Ty::F32 | Ty::Arr(Elem::P(_))));
        if !args.iter().take(2).any(|a| mine(probe(a))) {
            return None;
        }
    }
    Some(lower_call_inner(prog, callee, args, span, scope, sigs))
}

/// Whether `e` is a Float32, read from its syntax and the types already
/// known: a Float32 variable, a float element of a packed array, a def
/// that returns one, the Float32 builtins, and arithmetic with such a part.
/// The hooks act only where this holds, so a program with no Float32 lowers
/// nothing twice.
fn peek_f32(e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> bool {
    match e {
        Expr::Var { name, .. } => scope.get(name) == Some(Ty::F32),
        Expr::Index { base, .. } => matches!(peek_arr(base, scope, sigs), Some(k) if k.is_float()),
        Expr::Neg { operand, .. } => peek_f32(operand, scope, sigs),
        Expr::Binary { op, lhs, rhs, .. } => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) && (peek_f32(lhs, scope, sigs) || peek_f32(rhs, scope, sigs)),
        Expr::Call { callee, args, .. } => {
            if let Some(r) = sigs.ret.get(callee.as_str()) {
                return *r == Some(Ty::F32);
            }
            let a0 = args.first();
            match callee.as_str() {
                "float32" | "bfloat16" | "float16" => a0.is_some_and(|a| peek_arr(a, scope, sigs).is_none() && !matches!(a, Expr::ArrLit { .. }) && !matches!(scope_ty(a, scope), Some(Ty::Arr(_)))),
                "sum" => matches!(a0.and_then(|a| peek_arr(a, scope, sigs)), Some(k) if k.is_float()),
                "dot" => args.iter().all(|a| peek_arr(a, scope, sigs) == Some(Store::F32)),
                "min" | "max" => args.iter().any(|a| peek_f32(a, scope, sigs)),
                f => (f == "sqrt" || f == "abs" || crate::interp::f32s::FUNCS.contains(&f)) && a0.is_some_and(|a| peek_f32(a, scope, sigs)),
            }
        }
        _ => false,
    }
}

fn scope_ty(e: &Expr, scope: &Scope<'_>) -> Option<Ty> {
    if let Expr::Var { name, .. } = e { scope.get(name) } else { None }
}

/// The kind of the packed array `e` is, when its syntax says.
fn peek_arr(e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Option<Store> {
    match e {
        Expr::Var { name, .. } => match scope.get(name) {
            Some(Ty::Arr(Elem::P(k))) => Some(k),
            _ => None,
        },
        Expr::Call { callee, args, .. } => {
            if let Some(r) = sigs.ret.get(callee.as_str()) {
                return match r {
                    Some(Ty::Arr(Elem::P(k))) => Some(*k),
                    _ => None,
                };
            }
            if let Some(k) = Store::of_zeros(callee) {
                return Some(k);
            }
            let k = Store::of_conv(callee)?;
            let a = args.first()?;
            (matches!(a, Expr::ArrLit { .. }) || matches!(scope_ty(a, scope), Some(Ty::Arr(_))) || peek_arr(a, scope, sigs).is_some()).then_some(k)
        }
        _ => None,
    }
}

/// Whether `e` is a Float32 or a packed array.
fn may_be_mine(e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> bool {
    peek_f32(e, scope, sigs) || peek_arr(e, scope, sigs).is_some()
}

fn one<'a>(callee: &str, args: &'a [Expr], span: Span) -> Result<&'a Expr, Diagnostic> {
    match args {
        [a] => Ok(a),
        _ => Err(Diagnostic::error(format!("`{callee}` takes one argument")).with_primary(span, format!("{} given", args.len()))),
    }
}

fn lower_call_inner(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let refuse = |what: &str, ty: Ty, at: Span| Diagnostic::error(format!("`{callee}` does not take {}", art(ty))).with_primary(at, format!("this is {}", art(ty))).with_note(what.to_string());
    if let Some(k) = Store::of_zeros(callee) {
        let a = one(callee, args, span)?;
        let (id, ty) = lower_expr(prog, a, scope, sigs)?;
        expect_index(ty, a.span())?;
        let n = index_num(prog, id, ty, a.span());
        let kn = kind_node(prog, k, span);
        return Ok((call(prog, "viv_p_zeros", span, vec![n, kn]), Ty::Arr(Elem::P(k))));
    }
    if let Some(k) = Store::of_conv(callee) {
        let a = one(callee, args, span)?;
        // a literal array where a packed one is made: `float32([0.1, 0.2])`
        let (id, ty) = lower_expr(prog, a, scope, sigs)?;
        let kn = kind_node(prog, k, span);
        return match (k, ty) {
            (Store::F32, Ty::Num) => Ok((call(prog, "viv_f_of", span, vec![id]), Ty::F32)),
            (Store::F32, Ty::Int) => Ok((call(prog, "viv_f_ofi", span, vec![id]), Ty::F32)),
            (Store::F32, Ty::F32) => Ok((id, Ty::F32)),
            (Store::BF16 | Store::F16, Ty::F32) => Ok((call(prog, if k == Store::BF16 { "viv_f_bf16" } else { "viv_f_f16" }, span, vec![id]), Ty::F32)),
            (Store::F32, Ty::Arr(Elem::Num)) => {
                if let Expr::ArrLit { span: ls, .. } = a {
                    ARRS.with(|d| d.borrow_mut().insert((ls.start, ls.end), k));
                }
                Ok((call(prog, "viv_p_of_arr", span, vec![id, kn]), Ty::Arr(Elem::P(k))))
            }
            (_, Ty::Arr(Elem::Int)) if !k.is_float() => Ok((call(prog, "viv_p_of_iarr", span, vec![id, kn]), Ty::Arr(Elem::P(k)))),
            (_, Ty::Arr(Elem::P(from))) if from.is_float() == k.is_float() => {
                let fk = kind_node(prog, from, span);
                Ok((call(prog, "viv_p_conv", span, vec![id, fk, kn]), Ty::Arr(Elem::P(k))))
            }
            _ => Err(refuse(
                match k {
                    Store::F32 => "float32 takes a number, an Int, a Float32, an array (Arr), or a BFloat16 or Float16 array",
                    Store::BF16 | Store::F16 => "bfloat16 and float16 take a Float32 or a Float32, BFloat16 or Float16 array; round a number with float32(...) first",
                    _ => "an integer storage type is made from an [Int] or another integer storage array",
                },
                ty,
                a.span(),
            )),
        };
    }
    let a0 = args.first().ok_or_else(|| Diagnostic::error(format!("`{callee}` needs an argument")).with_primary(span, "here"))?;
    let (id, ty) = lower_expr(prog, a0, scope, sigs)?;
    match callee {
        "int64" => match ty {
            Ty::Arr(Elem::P(k)) if !k.is_float() => {
                let kn = kind_node(prog, k, span);
                Ok((call(prog, "viv_p_to_iarr", span, vec![id, kn]), Ty::Arr(Elem::Int)))
            }
            _ => Err(refuse("int64 takes an Int8, Int16, Int32 or UInt8 array", ty, a0.span())),
        },
        "float" => match ty {
            Ty::F32 => Ok((call(prog, "viv_f_wide", span, vec![id]), Ty::Num)),
            Ty::Arr(Elem::P(k)) if k.is_float() => {
                let kn = kind_node(prog, k, span);
                Ok((call(prog, "viv_p_to_arr", span, vec![id, kn]), Ty::Arr(Elem::Num)))
            }
            _ => Err(refuse("float takes a Float32 or a Float32, BFloat16 or Float16 array here", ty, a0.span())),
        },
        "len" => match ty {
            Ty::Arr(Elem::P(_)) => Ok((call(prog, "viv_p_len", span, vec![id]), Ty::Num)),
            _ => Err(refuse("len of a packed array", ty, a0.span())),
        },
        "sum" => match ty {
            Ty::Arr(Elem::P(k)) if args.len() == 1 => {
                let kn = kind_node(prog, k, span);
                Ok(if k.is_float() { (call(prog, "viv_p_sum", span, vec![id, kn]), Ty::F32) } else { (call(prog, "viv_p_isum", span, vec![id, kn]), Ty::Int) })
            }
            _ => Err(refuse("sum takes one packed array", ty, a0.span())),
        },
        "dot" => {
            let b = args.get(1).ok_or_else(|| Diagnostic::error("`dot` takes two arrays").with_primary(span, "one given"))?;
            let (bid, bty) = lower_expr(prog, b, scope, sigs)?;
            if ty != Ty::Arr(Elem::P(Store::F32)) || bty != ty || args.len() != 2 {
                return Err(Diagnostic::error("`dot` of Float32 arrays takes two [Float32]").with_primary(span, format!("given {} and {}", art(ty), art(bty))));
            }
            Ok((call(prog, "viv_p_dot", span, vec![id, bid]), Ty::F32))
        }
        "fmt" | "str" if ty == Ty::F32 && args.len() == 1 => Ok((call(prog, "viv_f_repr", span, vec![id]), Ty::Str)),
        "fmt" if ty == Ty::F32 => {
            // a format spec: the number the Float32 is
            let mut wide = args.to_vec();
            wide[0] = Expr::Call { callee: "float".into(), args: vec![a0.clone()], span: a0.span() };
            lower_expr(prog, &Expr::Call { callee: "fmt".into(), args: wide, span }, scope, sigs)
        }
        "min" | "max" if args.len() == 2 => {
            let (bid, bty) = lower_expr(prog, &args[1], scope, sigs)?;
            let a = adapt(prog, (id, ty), Ty::F32, a0)?;
            let b = adapt(prog, (bid, bty), Ty::F32, &args[1])?;
            if a.1 != Ty::F32 || b.1 != Ty::F32 {
                let (t, e) = if a.1 != Ty::F32 { (a.1, a0) } else { (b.1, &args[1]) };
                return Err(no_mix_in(callee, e.span(), t));
            }
            Ok((call(prog, if callee == "min" { "viv_f_min" } else { "viv_f_max" }, span, vec![a.0, b.0]), Ty::F32))
        }
        f if ty == Ty::F32 && args.len() == 1 && (f == "sqrt" || f == "abs" || crate::interp::f32s::FUNCS.contains(&f)) => {
            Ok((call(prog, &format!("viv_f_{f}"), span, vec![id]), Ty::F32))
        }
        _ => Err(Diagnostic::error(format!("`{callee}` is not defined on {} in this version", art(ty)))
            .with_primary(a0.span(), format!("this is {}", art(ty)))
            .with_note("float(x) widens a Float32 to a number")),
    }
}

/// What inference knows of a call of ours; `None` when
/// the call is not one of ours.
pub(super) fn infer_call(callee: &str, arg_tys: &[Option<Ty>]) -> Option<Option<Ty>> {
    if let Some(k) = Store::of_zeros(callee) {
        return Some(Some(Ty::Arr(Elem::P(k))));
    }
    let a0 = arg_tys.first().copied().flatten();
    if let Some(k) = Store::of_conv(callee) {
        return Some(match a0 {
            Some(Ty::Num | Ty::Int | Ty::F32) if k.is_float() => Some(Ty::F32),
            Some(Ty::Arr(_)) => Some(Ty::Arr(Elem::P(k))),
            _ => None,
        });
    }
    let mine = matches!(a0, Some(Ty::F32 | Ty::Arr(Elem::P(_)))) || matches!(arg_tys.get(1).copied().flatten(), Some(Ty::F32));
    if !mine {
        return None;
    }
    Some(match (callee, a0) {
        ("int64", _) => Some(Ty::Arr(Elem::Int)),
        ("float", Some(Ty::F32)) => Some(Ty::Num),
        ("float", Some(Ty::Arr(_))) => Some(Ty::Arr(Elem::Num)),
        ("len", _) => Some(Ty::Num),
        ("sum", Some(Ty::Arr(Elem::P(k)))) => Some(if k.is_float() { Ty::F32 } else { Ty::Int }),
        ("dot", _) => Some(Ty::F32),
        ("fmt" | "str", _) => Some(Ty::Str),
        ("min" | "max" | "sqrt" | "abs", _) => Some(Ty::F32),
        (f, Some(Ty::F32)) if crate::interp::f32s::FUNCS.contains(&f) => Some(Ty::F32),
        _ => return None,
    })
}

/// What inference knows of `a op b` with a Float32 on a side.
pub(super) fn infer_binary(op: BinOp, lt: Option<Ty>, rt: Option<Ty>) -> Option<Option<Ty>> {
    if lt != Some(Ty::F32) && rt != Some(Ty::F32) {
        return None;
    }
    use BinOp::*;
    Some(match op {
        Add | Sub | Mul | Div => Some(Ty::F32),
        Lt | Le | Gt | Ge | EqEq | Ne | And | Or => Some(Ty::Bool),
        _ => None,
    })
}

// ---- the runtime -------------------------------------------------------------

/// What one argument or result of a runtime call is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FK {
    /// a Float32: `float` in C, an f32 in wasm
    F,
    /// a number: `double`, an f64
    N,
    /// an Int: `long long`, an i64
    I,
    /// a string: `viv_str`, an i32
    S,
    /// an array: `viv_arr`, an i32
    A,
}

pub(crate) struct FFn {
    pub c: &'static str,
    pub args: &'static [FK],
    pub ret: FK,
    pub stops: bool,
}

use FK::{A, F, I, N, S};

/// Every runtime call Float32 and packed code makes. The wasm backend
/// imports each as `vrt_f_*` / `vrt_p_*`, except the arithmetic it writes
/// as an instruction (`wasm_op`).
pub(crate) const RUNTIME: &[FFn] = &[
    FFn { c: "viv_f_of", args: &[N], ret: F, stops: false },
    FFn { c: "viv_f_ofi", args: &[I], ret: F, stops: false },
    FFn { c: "viv_f_wide", args: &[F], ret: N, stops: false },
    FFn { c: "viv_f_add", args: &[F, F], ret: F, stops: false },
    FFn { c: "viv_f_sub", args: &[F, F], ret: F, stops: false },
    FFn { c: "viv_f_mul", args: &[F, F], ret: F, stops: false },
    FFn { c: "viv_f_div", args: &[F, F], ret: F, stops: false },
    FFn { c: "viv_f_neg", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_abs", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_sqrt", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_min", args: &[F, F], ret: F, stops: false },
    FFn { c: "viv_f_max", args: &[F, F], ret: F, stops: false },
    FFn { c: "viv_f_exp", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_log", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_sin", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_cos", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_tanh", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_bf16", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_f16", args: &[F], ret: F, stops: false },
    FFn { c: "viv_f_repr", args: &[F], ret: S, stops: false },
    FFn { c: "viv_f_bare", args: &[F], ret: S, stops: false },
    FFn { c: "viv_p_zeros", args: &[N, N], ret: A, stops: true },
    FFn { c: "viv_p_getf", args: &[A, N, N], ret: F, stops: true },
    FFn { c: "viv_p_geti", args: &[A, N, N], ret: I, stops: true },
    FFn { c: "viv_p_setf", args: &[A, N, F, N], ret: N, stops: true },
    FFn { c: "viv_p_seti", args: &[A, N, I, N], ret: N, stops: true },
    FFn { c: "viv_p_len", args: &[A], ret: N, stops: false },
    FFn { c: "viv_p_conv", args: &[A, N, N], ret: A, stops: true },
    FFn { c: "viv_p_of_arr", args: &[A, N], ret: A, stops: true },
    FFn { c: "viv_p_of_iarr", args: &[A, N], ret: A, stops: true },
    FFn { c: "viv_p_to_arr", args: &[A, N], ret: A, stops: true },
    FFn { c: "viv_p_to_iarr", args: &[A, N], ret: A, stops: true },
    FFn { c: "viv_p_sum", args: &[A, N], ret: F, stops: false },
    FFn { c: "viv_p_isum", args: &[A, N], ret: I, stops: true },
    FFn { c: "viv_p_dot", args: &[A, A], ret: F, stops: true },
    FFn { c: "viv_p_str", args: &[A, N], ret: S, stops: false },
];

pub(crate) fn rt_fn(c: &str) -> Option<&'static FFn> {
    if !c.starts_with("viv_f_") && !c.starts_with("viv_p_") {
        return None;
    }
    RUNTIME.iter().find(|f| f.c == c)
}

/// The wasm instruction a call is, when it is one: (opcode).
pub(crate) fn wasm_op(c: &str) -> Option<u8> {
    Some(match c {
        "viv_f_of" => 0xb6,   // f32.demote_f64
        "viv_f_ofi" => 0xb4,  // f32.convert_i64_s
        "viv_f_wide" => 0xbb, // f64.promote_f32
        "viv_f_add" => 0x92,
        "viv_f_sub" => 0x93,
        "viv_f_mul" => 0x94,
        "viv_f_div" => 0x95,
        "viv_f_neg" => 0x8c,
        "viv_f_abs" => 0x8b,
        "viv_f_sqrt" => 0x91,
        _ => return None,
    })
}

pub(super) fn emit_call(node: &Node<Viv>, callee: &str, cx: &mut CEmitCx<Viv>) -> Option<CExpr> {
    let f = rt_fn(callee)?;
    let id = node.id;
    cx.need(if callee.starts_with("viv_p_") || matches!(callee, "viv_f_repr" | "viv_f_bare") { "f32_arr" } else { "f32s" });
    let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
    if f.stops {
        args.push(CExpr::raw(id, site_row(node.span).to_string()));
    }
    Some(CExpr::call(id, callee, args))
}

pub(super) fn fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "f32s",
        includes: &["#include <float.h>", "#include <math.h>", "#include <string.h>", "#include <stdint.h>"],
        deps: &[],
        code: C_F32S,
    }
}

pub(super) fn arr_fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "f32_arr",
        includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>"],
        deps: &["f32s", "ints", "str_core", "arr_type", "arr_new", "bounds", "fmt_g", "print"],
        code: C_F32_ARR,
    }
}

const C_F32S: &str = r#"/* Float32: interp/f32s.rs, line for line. Every
   operation is one IEEE binary32 operation: no excess precision, no
   contraction (the program is built with -ffp-contract=off) */
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "Viv's Float32 needs FLT_EVAL_METHOD == 0: float expressions evaluated in float"
#endif
#if defined(__GNUC__)
#define VIV_F_UNUSED static inline __attribute__((unused))
#else
#define VIV_F_UNUSED static inline
#endif
VIV_F_UNUSED float viv_f_of(double x) { return (float)x; }
VIV_F_UNUSED float viv_f_ofi(long long n) { return (float)n; }
VIV_F_UNUSED double viv_f_wide(float x) { return (double)x; }
VIV_F_UNUSED float viv_f_add(float a, float b) { return a + b; }
VIV_F_UNUSED float viv_f_sub(float a, float b) { return a - b; }
VIV_F_UNUSED float viv_f_mul(float a, float b) { return a * b; }
VIV_F_UNUSED float viv_f_div(float a, float b) { return a / b; }
VIV_F_UNUSED float viv_f_neg(float a) { return -a; }
VIV_F_UNUSED float viv_f_abs(float a) { return fabsf(a); }
VIV_F_UNUSED float viv_f_sqrt(float a) { return sqrtf(a); }
VIV_F_UNUSED float viv_f_min(float a, float b) { return b < a ? b : a; }
VIV_F_UNUSED float viv_f_max(float a, float b) { return b > a ? b : a; }
/* Viv's double function, rounded once, with the inputs where that is not the
   correctly rounded float32 (interp/f32s.rs, func and EXCEPTIONS) */
VIV_F_UNUSED uint32_t viv_f_bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
VIV_F_UNUSED float viv_f_of_bits(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
VIV_F_UNUSED float viv_f_exp(float x) { return (float)viv_exp_cr((double)x); }
VIV_F_UNUSED float viv_f_log(float x) {
    uint32_t u = viv_f_bits(x);
    if (u == 0x3c413d3au) return viv_f_of_bits(0xc08e158fu);
    if (u == 0x41178febu) return viv_f_of_bits(0x400fe5e7u);
    if (u == 0x4c5d65a5u) return viv_f_of_bits(0x418f034bu);
    if (u == 0x65d890d3u) return viv_f_of_bits(0x4254d1f9u);
    if (u == 0x6f31a8ecu) return viv_f_of_bits(0x42845a89u);
    return (float)viv_log((double)x);
}
VIV_F_UNUSED float viv_f_sin(float x) {
    uint32_t u = viv_f_bits(x);
    if (u == 0x46199998u) return viv_f_of_bits(0xbeb1fa5du);
    if (u == 0xc6199998u) return viv_f_of_bits(0x3eb1fa5du);
    return (float)viv_sin((double)x);
}
VIV_F_UNUSED float viv_f_cos(float x) {
    uint32_t u = viv_f_bits(x);
    if (u == 0x5f18b878u) return viv_f_of_bits(0x3f7f14bbu);
    if (u == 0x6115cb11u) return viv_f_of_bits(0x3f78142fu);
    if (u == 0xdf18b878u) return viv_f_of_bits(0x3f7f14bbu);
    if (u == 0xe115cb11u) return viv_f_of_bits(0x3f78142fu);
    return (float)viv_cos((double)x);
}
VIV_F_UNUSED float viv_f_tanh(float x) { return (float)viv_tanh((double)x); }
/* round to nearest even on 16 bits (interp/f32s.rs, bf16_bits, f16_bits) */
VIV_F_UNUSED uint16_t viv_bf16_bits(float x) {
    uint32_t u = viv_f_bits(x);
    if (x != x) return (uint16_t)((u >> 16) | 0x0040);
    uint32_t lsb = (u >> 16) & 1;
    return (uint16_t)((u + 0x7fff + lsb) >> 16);
}
VIV_F_UNUSED float viv_bf16_value(uint16_t h) { return viv_f_of_bits((uint32_t)h << 16); }
VIV_F_UNUSED uint16_t viv_f16_bits(float x) {
    uint32_t u = viv_f_bits(x);
    uint16_t sign = (uint16_t)((u >> 16) & 0x8000);
    uint32_t a = u & 0x7fffffff;
    if (a > 0x7f800000) return (uint16_t)(sign | 0x7e00 | ((a >> 13) & 0x3ff));
    if (a == 0x7f800000) return (uint16_t)(sign | 0x7c00);
    if (a >= 0x477ff000) return (uint16_t)(sign | 0x7c00);
    int e = (int)(a >> 23);
    if (e >= 113) {
        uint32_t m = a - (112u << 23);
        uint32_t lsb = (m >> 13) & 1;
        return (uint16_t)(sign | ((m + 0x0fff + lsb) >> 13));
    }
    if (e < 102) return sign;
    uint32_t mant = (a & 0x007fffff) | 0x00800000;
    uint32_t shift = (uint32_t)(126 - e);
    uint32_t q = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (q & 1))) q++;
    return (uint16_t)(sign | q);
}
VIV_F_UNUSED float viv_f16_value(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0x1f) return viv_f_of_bits(sign | 0x7f800000 | (m << 13));
    if (e == 0) {
        /* a subnormal or zero: m 2^-24, exact in a float */
        float v = (float)m * 5.9604644775390625e-8f;
        return sign ? -v : v;
    }
    return viv_f_of_bits(sign | ((e + 112) << 23) | (m << 13));
}
VIV_F_UNUSED float viv_f_bf16(float x) { return viv_bf16_value(viv_bf16_bits(x)); }
VIV_F_UNUSED float viv_f_f16(float x) { return viv_f16_value(viv_f16_bits(x)); }"#;

const C_F32_ARR: &str = r#"/* The packed arrays: a viv_arr whose dims[0] is the
   element count and whose size counts the doubles its bytes fill, so a copy
   that moves `size` doubles moves the bytes. Kinds, in order: Float32,
   BFloat16, Float16, Int8, Int16, Int32, UInt8 (ast.rs, Store::ALL) */
static const int viv_p_width[7] = { 4, 2, 2, 1, 2, 4, 1 };
static const char* viv_p_name[7] = { "a Float32", "a BFloat16", "a Float16", "an Int8", "an Int16", "an Int32", "a UInt8" };
static const long long viv_p_lo[7] = { 0, 0, 0, -128, -32768, -2147483648LL, 0 };
static const long long viv_p_hi[7] = { 0, 0, 0, 127, 32767, 2147483647LL, 255 };
static viv_arr viv_p_new(long long n, int k) {
    long long bytes = n * viv_p_width[k];
    int d[1]; d[0] = (int)((bytes + 7) / 8);
    viv_arr a = viv_arr_new(1, d);
    a.dims[0] = (int)n;
    return a;
}
VIV_F_UNUSED viv_arr viv_p_zeros(double n, double k, int site) {
    (void)site;
    double x[1] = { n }; int d[1];
    viv_zeros_dims(1, x, d, 1);
    return viv_p_new(d[0], (int)k);
}
VIV_F_UNUSED double viv_p_len(viv_arr a) { return (double)a.dims[0]; }
VIV_F_UNUSED float viv_p_rdf(const unsigned char* b, long long i, int k) {
    if (k == 0) { float x; memcpy(&x, b + 4 * i, 4); return x; }
    uint16_t h; memcpy(&h, b + 2 * i, 2);
    return k == 1 ? viv_bf16_value(h) : viv_f16_value(h);
}
VIV_F_UNUSED long long viv_p_rdi(const unsigned char* b, long long i, int k) {
    switch (k) {
    case 3: return (long long)(signed char)b[i];
    case 4: { int16_t v; memcpy(&v, b + 2 * i, 2); return v; }
    case 5: { int32_t v; memcpy(&v, b + 4 * i, 4); return v; }
    default: return (long long)b[i];
    }
}
VIV_F_UNUSED void viv_p_wrf(unsigned char* b, long long i, float x, int k) {
    if (k == 0) { memcpy(b + 4 * i, &x, 4); return; }
    uint16_t h = k == 1 ? viv_bf16_bits(x) : viv_f16_bits(x);
    memcpy(b + 2 * i, &h, 2);
}
/* an Int that does not fit where it is stored (interp/f32s.rs, no_fit) */
static void viv_p_nofit(long long v, int k, int site) {
    char m[160];
    snprintf(m, sizeof m, "%lld does not fit in %s (%lld to %lld)", v, viv_p_name[k], viv_p_lo[k], viv_p_hi[k]);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
VIV_F_UNUSED void viv_p_wri(unsigned char* b, long long i, long long v, int k, int site) {
    if (v < viv_p_lo[k] || v > viv_p_hi[k]) viv_p_nofit(v, k, site);
    switch (k) {
    case 3: b[i] = (unsigned char)(signed char)v; return;
    case 4: { int16_t x = (int16_t)v; memcpy(b + 2 * i, &x, 2); return; }
    case 5: { int32_t x = (int32_t)v; memcpy(b + 4 * i, &x, 4); return; }
    default: b[i] = (unsigned char)v; return;
    }
}
VIV_F_UNUSED float viv_p_getf(viv_arr a, double i, double k, int site) {
    return viv_p_rdf((const unsigned char*)a.data, viv_ix(i, a.dims[0], 0, site), (int)k);
}
VIV_F_UNUSED long long viv_p_geti(viv_arr a, double i, double k, int site) {
    return viv_p_rdi((const unsigned char*)a.data, viv_ix(i, a.dims[0], 0, site), (int)k);
}
VIV_F_UNUSED double viv_p_setf(viv_arr a, double i, float v, double k, int site) {
    viv_p_wrf((unsigned char*)a.data, viv_ix(i, a.dims[0], 0, site), v, (int)k);
    return 0.0;
}
VIV_F_UNUSED double viv_p_seti(viv_arr a, double i, long long v, double k, int site) {
    viv_p_wri((unsigned char*)a.data, viv_ix(i, a.dims[0], 0, site), v, (int)k, site);
    return 0.0;
}
VIV_F_UNUSED viv_arr viv_p_conv(viv_arr a, double from, double to, int site) {
    int f = (int)from, t = (int)to;
    long long n = a.dims[0];
    viv_arr b = viv_p_new(n, t);
    const unsigned char* s = (const unsigned char*)a.data;
    unsigned char* o = (unsigned char*)b.data;
    for (long long i = 0; i < n; i++) {
        if (f <= 2) viv_p_wrf(o, i, viv_p_rdf(s, i, f), t);
        else viv_p_wri(o, i, viv_p_rdi(s, i, f), t, site);
    }
    return b;
}
VIV_F_UNUSED viv_arr viv_p_of_arr(viv_arr a, double k, int site) {
    (void)site;
    viv_arr b = viv_p_new(a.size, (int)k);
    for (long long i = 0; i < a.size; i++) viv_p_wrf((unsigned char*)b.data, i, (float)a.data[i], (int)k);
    return b;
}
VIV_F_UNUSED viv_arr viv_p_of_iarr(viv_arr a, double k, int site) {
    viv_arr b = viv_p_new(a.size, (int)k);
    for (long long i = 0; i < a.size; i++) viv_p_wri((unsigned char*)b.data, i, ((long long*)a.data)[i], (int)k, site);
    return b;
}
VIV_F_UNUSED viv_arr viv_p_to_arr(viv_arr a, double k, int site) {
    (void)site;
    int d[1]; d[0] = a.dims[0];
    viv_arr b = viv_arr_new(1, d);
    for (long long i = 0; i < d[0]; i++) b.data[i] = (double)viv_p_rdf((const unsigned char*)a.data, i, (int)k);
    return b;
}
VIV_F_UNUSED viv_arr viv_p_to_iarr(viv_arr a, double k, int site) {
    (void)site;
    int d[1]; d[0] = a.dims[0];
    viv_arr b = viv_arr_new(1, d);
    for (long long i = 0; i < d[0]; i++) ((long long*)b.data)[i] = viv_p_rdi((const unsigned char*)a.data, i, (int)k);
    return b;
}
/* NumPy's pairwise sum in float32, from +0.0 in blocks of 8192
   (interp/f32s.rs, pairwise and sum) */
static float viv_p_pw(const unsigned char* b, long long off, long long n, int k) {
    if (n < 8) {
        float s = -0.0f;
        for (long long i = 0; i < n; i++) s += viv_p_rdf(b, off + i, k);
        return s;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; j++) r[j] = viv_p_rdf(b, off + j, k);
        long long i = 8;
        for (; i < n - n % 8; i += 8) for (int j = 0; j < 8; j++) r[j] += viv_p_rdf(b, off + i + j, k);
        float s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += viv_p_rdf(b, off + i, k);
        return s;
    }
    long long n2 = n / 2;
    n2 -= n2 % 8;
    return viv_p_pw(b, off, n2, k) + viv_p_pw(b, off + n2, n - n2, k);
}
VIV_F_UNUSED float viv_p_sum(viv_arr a, double k) {
    float s = 0.0f;
    long long n = a.dims[0];
    for (long long b = 0; b < n; b += 8192) s += viv_p_pw((const unsigned char*)a.data, b, n - b < 8192 ? n - b : 8192, (int)k);
    return s;
}
VIV_F_UNUSED long long viv_p_isum(viv_arr a, double k, int site) {
    long long s = 0;
    for (long long i = 0; i < a.dims[0]; i++) s = viv_i_add(s, viv_p_rdi((const unsigned char*)a.data, i, (int)k), site);
    return s;
}
VIV_F_UNUSED float viv_p_dot(viv_arr a, viv_arr b, int site) {
    if (a.dims[0] != b.dims[0]) {
        char m[160];
        snprintf(m, sizeof m, "dot: the arrays have %d and %d elements", a.dims[0], b.dims[0]);
        viv_i_fail(VIV_E_VALUE, m, site);
    }
    const float* x = (const float*)a.data;
    const float* y = (const float*)b.data;
    float s = 0.0f;
    for (long long i = 0; i < a.dims[0]; i++) {
        float p = x[i] * y[i];
        s += p;
    }
    return s;
}
/* the shortest digits that read back as x in float32, laid out as viv_repr
   lays out a double (interp/f32s.rs, repr) */
static void viv_repr32(char* out, float x) {
    if (x != x) { strcpy(out, "nan"); return; }
    if (x > FLT_MAX || x < -FLT_MAX) { strcpy(out, x < 0 ? "-inf" : "inf"); return; }
    if (x == 0.0f) { strcpy(out, signbit(x) ? "-0.0" : "0.0"); return; }
    float a = x < 0 ? -x : x;
    char b[48], t[48], d[24];
    unsigned long long n = 0;
    int ex = 0, p;
    for (p = 1; p <= 9; p++) {
        snprintf(b, sizeof b, "%.*e", p - 1, (double)a);
        const char* c = b;
        n = 0;
        for (; *c != 'e'; c++) if (*c >= '0' && *c <= '9') n = n * 10 + (unsigned long long)(*c - '0');
        ex = atoi(c + 1);
        float r = strtof(b, 0);
        if (r != a) {
            unsigned long long lo = 1, hi;
            for (int q = 1; q < p; q++) lo *= 10;
            hi = lo * 10;
            if ((double)r < (double)a) { n++; if (n == hi) { n = lo; ex++; } }
            else { n--; if (n < lo) { n = hi - 1; ex--; } }
            snprintf(t, sizeof t, "%llue%d", n, ex - (p - 1));
            if (strtof(t, 0) != a) continue;
        }
        break;
    }
    snprintf(d, sizeof d, "%llu", n);
    int nd = (int)strlen(d);
    while (nd > 1 && d[nd - 1] == '0') nd--;
    d[nd] = 0;
    char* s = out;
    if (x < 0) *s++ = '-';
    int decpt = ex + 1;
    if (-4 < decpt && decpt <= 16) {
        if (decpt <= 0) {
            *s++ = '0'; *s++ = '.';
            for (int q = 0; q < -decpt; q++) *s++ = '0';
            strcpy(s, d);
        } else if (decpt >= nd) {
            strcpy(s, d); s += nd;
            for (int q = 0; q < decpt - nd; q++) *s++ = '0';
            strcpy(s, ".0");
        } else {
            memcpy(s, d, (size_t)decpt); s += decpt;
            *s++ = '.';
            strcpy(s, d + decpt);
        }
    } else {
        *s++ = d[0];
        if (nd > 1) { *s++ = '.'; strcpy(s, d + 1); s += nd - 1; }
        sprintf(s, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    }
}
VIV_F_UNUSED viv_str viv_f_repr(float x) {
    char b[48];
    viv_repr32(b, x);
    return viv_s_make(b, (long)strlen(b), 0);
}
VIV_F_UNUSED viv_str viv_f_bare(float x) {
    const char* t = viv_g((double)x);
    return viv_s_make(t, (long)strlen(t), 0);
}
VIV_F_UNUSED viv_str viv_p_str(viv_arr a, double k) {
    int kk = (int)k;
    long cap = 2 + 34 * (long)a.dims[0], n = 0;
    char* b = (char*)malloc((size_t)cap);
    if (!b) viv_s_nomem();
    b[n++] = '[';
    for (int i = 0; i < a.dims[0]; i++) {
        if (i) { b[n++] = ','; b[n++] = ' '; }
        if (kk <= 2) {
            const char* t = viv_g((double)viv_p_rdf((const unsigned char*)a.data, i, kk));
            size_t l = strlen(t);
            memcpy(b + n, t, l); n += (long)l;
        }
        else n += snprintf(b + n, (size_t)(cap - n), "%lld", viv_p_rdi((const unsigned char*)a.data, i, kk));
    }
    b[n++] = ']';
    viv_str r = viv_s_make(b, n, 0);
    free(b);
    return r;
}"#;
