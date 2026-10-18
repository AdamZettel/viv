// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Integers in the checked lowering and the C backend.
//!
//! An integer literal (`Program::whole`) and a call of `len`, `int`, `floor`,
//! `ceil` or `round` ADAPT: in an Int position they are computed as Ints,
//! everywhere else as numbers, as they always were. Which ones were computed
//! as Ints is recorded (`DECIDED`), so the interpreter, which reads the tree
//! and not the IR, computes the same ones as Ints (`int_spans`).
//!
//! Int arithmetic is a call of the runtime (`viv_i_*`): checked, stopping with
//! the interpreter's words (interp/ints.rs) and, in C, the location. A program
//! with no Int reaches none of it: its IR and its C are what they were.

use super::*;

thread_local! {
    /// The spans of the program's integer literals (`Program::whole`), set by
    /// `lower` for the program being lowered.
    pub(super) static WHOLE: std::cell::RefCell<std::collections::HashMap<(u32, u32), i64>> = std::cell::RefCell::new(Default::default());
    /// The spans of the literals and calls this lowering computed as Ints.
    static DECIDED: std::cell::RefCell<std::collections::HashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
    /// The spans of the Int arguments this lowering converted to numbers: an
    /// Int passed where a def takes a number.
    static NUMED: std::cell::RefCell<std::collections::HashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
    /// The spans of the array literals this lowering made Int arrays (`[]`
    /// and `[1, 2]` where an [Int] goes). Kept apart from DECIDED: a literal
    /// can share its span with a call the parser wrote around it.
    static ARRS: std::cell::RefCell<std::collections::HashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
}

/// Start lowering `program`: its integer literals, and nothing decided yet.
pub(super) fn begin(program: &Program) {
    WHOLE.with(|w| *w.borrow_mut() = program.whole.clone());
    DECIDED.with(|d| d.borrow_mut().clear());
    NUMED.with(|d| d.borrow_mut().clear());
    ARRS.with(|d| d.borrow_mut().clear());
}

fn decide_arr(span: Span) {
    ARRS.with(|d| d.borrow_mut().insert((span.start, span.end)));
}

/// The spans `decide_arr` recorded, for the interpreter.
pub fn arr_spans() -> std::collections::HashSet<(u32, u32)> {
    ARRS.with(|d| d.borrow().clone())
}

/// The Int at `span` is passed where a def takes a number: the interpreter
/// converts it there too.
pub(super) fn numed(span: Span) {
    NUMED.with(|d| d.borrow_mut().insert((span.start, span.end)));
}

/// The spans `numed` recorded, for the interpreter (with `int_spans`).
pub fn numed_spans() -> std::collections::HashSet<(u32, u32)> {
    NUMED.with(|d| d.borrow().clone())
}

/// The exact value of the integer literal at `span`, if one is there.
fn whole(span: Span) -> Option<i64> {
    WHOLE.with(|w| w.borrow().get(&(span.start, span.end)).copied())
}

/// An integer literal's exact value: the number it holds when that is exact
/// (every integer below 2^53), else the value the lexer read from its digits.
/// (Two counts the parser writes for a loop can share a span, and each holds
/// its own value.)
pub(crate) fn literal_value(value: f64, span: Span) -> i64 {
    if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
        value as i64
    } else {
        whole(span).unwrap_or(value as i64)
    }
}

fn decide(span: Span) {
    DECIDED.with(|d| d.borrow_mut().insert((span.start, span.end)));
}

/// The literals and calls of `src` that the lowering computes as Ints, by
/// span: what the interpreter must compute as Ints too. Empty for a program
/// that names no Int (it is not lowered for this).
pub fn int_spans(src: &str) -> std::collections::HashSet<(u32, u32)> {
    if !mentions_ints(src) {
        return Default::default();
    }
    let _ = crate::build_ir(src, SourceId(0));
    DECIDED.with(|d| d.borrow().clone())
}

/// Whether `src` can hold an Int at all: an Int comes only from an `Int`
/// annotation (a type, a payload, `x: Int = e`), the Int builtins, or a
/// variable that holds what `int(...)` gives.
pub fn mentions_ints(src: &str) -> bool {
    let word = |w: &str| {
        src.match_indices(w).any(|(i, _)| {
            let before = src[..i].chars().next_back();
            let after = src[i + w.len()..].chars().next();
            !before.is_some_and(|c| c.is_alphanumeric() || c == '_') && !after.is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
    };
    // `n = int(x)` can make an Int too (infer_int_vars)
    // the integer storage arrays read and store Ints
    word("Int") || word("int_zeros") || super::f32s::mentions(src) || src.match_indices("int(").any(|(i, _)| !src[..i].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_'))
}

/// Whether `e` holds a call of `int`, through `- + * // %` and `abs`, `min`
/// and `max`: what makes a variable or a returned slot an Int.
pub(super) fn has_int_call(e: &Expr) -> bool {
    match e {
        Expr::Call { callee, args, .. } if callee == "int" => args.len() == 1,
        Expr::Call { callee, args, .. } if matches!(callee.as_str(), "abs" | "min" | "max") => args.iter().any(has_int_call),
        Expr::Neg { operand, .. } => has_int_call(operand),
        Expr::Binary { lhs, rhs, .. } => has_int_call(lhs) || has_int_call(rhs),
        _ => false,
    }
}

/// The builtins whose result is a whole number and adapts like a literal.
const ADAPTING: [&str; 5] = ["len", "int", "floor", "ceil", "round"];

/// Whether `e` adapts in an Int position.
pub(super) fn adapts(e: &Expr, sigs: &Sigs) -> bool {
    match e {
        Expr::Num { span, .. } => whole(*span).is_some(),
        Expr::Neg { operand, .. } => adapts(operand, sigs),
        Expr::Call { callee, .. } if sigs.arity.contains_key(callee.as_str()) => false,
        // `abs`, `min` and `max` of whole numbers are whole (B305)
        Expr::Call { callee, args, .. } if callee == "abs" => args.len() == 1 && adapts(&args[0], sigs),
        Expr::Call { callee, args, .. } if callee == "min" || callee == "max" => args.len() == 2 && args.iter().all(|a| adapts(a, sigs)),
        Expr::Call { callee, args, .. } => ADAPTING.contains(&callee.as_str()) && !args.is_empty(),
        Expr::Binary { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod, lhs, rhs, .. } => adapts(lhs, sigs) && adapts(rhs, sigs),
        _ => false,
    }
}

/// Whether `e` is certainly an Int, read without lowering it: a variable,
/// field, element or call result typed Int, or arithmetic on one. A `false`
/// may be an Int still (the lowering then finds out); a program with no Int
/// gets `false` everywhere.
pub(super) fn peek_int(e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> bool {
    match e {
        Expr::Var { name, .. } => scope.get(name) == Some(Ty::Int),
        Expr::Neg { operand, .. } => peek_int(operand, scope, sigs),
        Expr::Binary { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod, lhs, rhs, .. } => {
            (peek_int(lhs, scope, sigs) && (peek_int(rhs, scope, sigs) || adapts(rhs, sigs))) || (peek_int(rhs, scope, sigs) && adapts(lhs, sigs))
        }
        Expr::Index { base, .. } => matches!(&**base, Expr::Var { name, .. } if scope.get(name) == Some(Ty::Arr(Elem::Int))),
        Expr::Field { base, field, .. } => match &**base {
            Expr::Var { name, .. } => match scope.get(name) {
                Some(Ty::Record(r)) => sigs.records.get(r).is_some_and(|i| i.fields.iter().any(|(f, t)| f == field && *t == Ty::Int)),
                _ => false,
            },
            _ => false,
        },
        Expr::Call { callee, args, .. } => {
            if sigs.arity.contains_key(callee.as_str()) {
                return sigs.ret.get(callee.as_str()).copied().flatten() == Some(Ty::Int);
            }
            match callee.as_str() {
                "abs" | "floor" | "ceil" | "round" | "int" => args.len() == 1 && peek_int(&args[0], scope, sigs),
                "min" | "max" | "pow" => {
                    args.len() == 2 && args.iter().all(|a| peek_int(a, scope, sigs) || adapts(a, sigs)) && args.iter().any(|a| peek_int(a, scope, sigs))
                }
                c if BITS.contains(&c) => args.iter().any(|a| peek_int(a, scope, sigs)),
                "wrap_add" | "wrap_sub" | "wrap_mul" | "umulhi" => true,
                c => crate::interp::bytes::reader(c).is_some_and(|(_, num)| !num),
            }
        }
        _ => false,
    }
}

/// The bit builtins with an Int form (and their 32-bit-word form on numbers).
const BITS: [&str; 7] = ["band", "bor", "bxor", "bnot", "shl", "shr", "sar"];

/// `e` computed as an Int, for an Int position. An expression that adapts is
/// computed in Ints; anything else must be an Int already.
pub(super) fn lower_as_int(prog: &mut IrProgram<Viv>, e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    match e {
        Expr::Num { value, span } if adapts(e, sigs) => {
            decide(*span);
            Ok(prog.add(Viv::Int(literal_value(*value, *span)), *span, vec![]))
        }
        Expr::Neg { operand, span } if adapts(operand, sigs) => {
            // `-3` is the literal -3, not the negation of 3
            if let Expr::Num { value, span: lspan } = &**operand {
                decide(*lspan);
                let v = literal_value(*value, *lspan);
                return Ok(prog.add(Viv::Int(v.checked_neg().unwrap_or(i64::MIN)), *span, vec![]));
            }
            let x = lower_as_int(prog, operand, scope, sigs)?;
            Ok(call(prog, "viv_i_neg", *span, vec![x]))
        }
        Expr::Call { callee, args, span } if matches!(callee.as_str(), "abs" | "min" | "max") && adapts(e, sigs) => {
            // each operand an Int, then the Int form (B305)
            let ids = args.iter().map(|a| lower_as_int(prog, a, scope, sigs)).collect::<Result<Vec<_>, _>>()?;
            Ok(call(prog, &format!("viv_i_{callee}"), *span, ids))
        }
        Expr::Call { span, .. } if adapts(e, sigs) => {
            let (id, ty) = lower_expr(prog, e, scope, sigs)?;
            if ty == Ty::Int {
                return Ok(id);
            }
            expect_num(ty, *span, "an Int")?;
            decide(*span);
            Ok(call(prog, "viv_i_of", *span, vec![id]))
        }
        Expr::Binary { op, lhs, rhs, span } if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod) && adapts(e, sigs) => {
            let l = lower_as_int(prog, lhs, scope, sigs)?;
            let r = lower_as_int(prog, rhs, scope, sigs)?;
            Ok(int_arith(prog, *op, *span, l, r))
        }
        _ => {
            let (id, ty) = lower_expr(prog, e, scope, sigs)?;
            if ty != Ty::Int {
                return Err(not_int(ty, e.span()));
            }
            Ok(id)
        }
    }
}

/// A value of type `ty` where an Int is needed.
pub(super) fn not_int(ty: Ty, span: Span) -> Diagnostic {
    let d = Diagnostic::error(format!("an Int is needed here, and this is a {}", ty.name())).with_primary(span, format!("this is a {}", ty.name()));
    if ty == Ty::Num {
        d.with_note("a number becomes an Int only where you say how: `int(x)` (toward zero), `round(x)`, `floor(x)` or `ceil(x)`; an integer literal (`3`, not `3.0`) is an Int where an Int is expected")
    } else {
        d
    }
}

/// `e` lowered for a binding of type `want` (an assignment, an argument, a
/// field, a return): an expression that adapts is computed as an Int where
/// an Int is wanted.
pub(super) fn lower_for(prog: &mut IrProgram<Viv>, e: &Expr, want: Option<Ty>, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    if want == Some(Ty::Int) && adapts(e, sigs) {
        return Some(lower_as_int(prog, e, scope, sigs).map(|id| (id, Ty::Int)));
    }
    // the growable pass seals what a tracked Int array is bound to
    if let (Some(Ty::Arr(Elem::Int)), Expr::Call { callee, args, span }) = (want, e) {
        if callee == crate::growable::SEAL && args.len() == 1 {
            let r = lower_for(prog, &args[0], want, scope, sigs)?;
            return Some(r.map(|(id, t)| (call(prog, "viv_iseal", *span, vec![id]), t)));
        }
    }
    // `[1, 2, 3]` where an Int array is bound: each element an Int
    if let (Some(Ty::Arr(Elem::Int)), Expr::ArrLit { elems, span }) = (want, e) {
        // `[]` where an Int array goes: an empty one, which `append` grows
        if elems.is_empty() {
            decide_arr(*span);
            return Some(Ok((call(prog, "viv_ivec", *span, vec![]), Ty::Arr(Elem::Int))));
        }
        if elems.iter().any(|x| matches!(x, Expr::ArrLit { .. })) {
            return Some(Err(Diagnostic::error("an Int array's literal lists Ints").with_primary(*span, "an Int array has rank 1")));
        }
        let mut ids = Vec::with_capacity(elems.len());
        for x in elems {
            match lower_as_int(prog, x, scope, sigs) {
                Ok(id) => ids.push(id),
                Err(d) => return Some(Err(d)),
            }
        }
        decide_arr(*span);
        return Some(Ok((call(prog, "viv_ivec", *span, ids), Ty::Arr(Elem::Int))));
    }
    None
}

fn call(prog: &mut IrProgram<Viv>, name: &str, span: Span, kids: Vec<NodeId>) -> NodeId {
    prog.add(Viv::Call { callee: name.to_string() }, span, kids)
}

fn int_arith(prog: &mut IrProgram<Viv>, op: BinOp, span: Span, l: NodeId, r: NodeId) -> NodeId {
    let f = match op {
        BinOp::Add => "viv_i_add",
        BinOp::Sub => "viv_i_sub",
        BinOp::Mul => "viv_i_mul",
        BinOp::FloorDiv => "viv_i_fdiv",
        BinOp::Mod => "viv_i_fmod",
        _ => unreachable!("not Int arithmetic"),
    };
    call(prog, f, span, vec![l, r])
}

/// `a / b` where both are Ints (or adapt beside one): the parts, for an exact
/// quotient (`ceil(a / b)`, `floor(a / b)`, a `for` loop's pass count).
pub(super) fn int_quotient<'e>(e: &'e Expr, scope: &Scope<'_>, sigs: &Sigs) -> Option<(&'e Expr, &'e Expr)> {
    let Expr::Binary { op: BinOp::Div, lhs, rhs, .. } = e else { return None };
    let ok = |x: &Expr| peek_int(x, scope, sigs) || adapts(x, sigs);
    (ok(lhs) && ok(rhs) && (peek_int(lhs, scope, sigs) || peek_int(rhs, scope, sigs))).then_some((&**lhs, &**rhs))
}

/// `ceil(a / b)` (`up`) or `floor(a / b)` of two Ints, exactly, as an Int.
pub(super) fn lower_quotient(prog: &mut IrProgram<Viv>, lhs: &Expr, rhs: &Expr, up: bool, span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    let a = lower_as_int(prog, lhs, scope, sigs)?;
    let b = lower_as_int(prog, rhs, scope, sigs)?;
    Ok(call(prog, if up { "viv_i_cdiv" } else { "viv_i_fdiv" }, span, vec![a, b]))
}

/// An Int where a number is needed: exact, or the program stops.
pub(super) fn to_num(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, span: Span) -> (NodeId, Ty) {
    if ty == Ty::Int {
        (call(prog, "viv_i_num", span, vec![id]), Ty::Num)
    } else {
        (id, ty)
    }
}

/// `a op b` with an Int on at least one side, or `None` when neither side
/// is one (the caller's rules apply, unchanged). An operand that adapts is
/// computed as an Int beside an Int.
pub(super) fn lower_binary(
    prog: &mut IrProgram<Viv>,
    op: BinOp,
    lhs: &Expr,
    rhs: &Expr,
    span: Span,
    scope: &Scope<'_>,
    sigs: &Sigs,
) -> Result<Option<(NodeId, Ty)>, Diagnostic> {
    use BinOp::*;
    if matches!(op, And | Or | Matmul) {
        return Ok(None);
    }
    let (la, ra) = (adapts(lhs, sigs), adapts(rhs, sigs));
    let (lp, rp) = (peek_int(lhs, scope, sigs), peek_int(rhs, scope, sigs));
    if !(lp || rp) {
        return Ok(None);
    }
    // the side that adapts beside a certain Int is an Int; lowered in source order
    let (lid, lty) = if la && rp { (lower_as_int(prog, lhs, scope, sigs)?, Ty::Int) } else { lower_expr(prog, lhs, scope, sigs)? };
    let (rid, rty) = if ra && (lp || lty == Ty::Int) { (lower_as_int(prog, rhs, scope, sigs)?, Ty::Int) } else { lower_expr(prog, rhs, scope, sigs)? };
    combine(prog, op, (lid, lty, lhs.span()), (rid, rty, rhs.span()), span).map(Some)
}

/// `a op b` for lowered operands, when one is an Int; `Err` when neither is
/// a number-like operand an Int can meet.
pub(super) fn combine(prog: &mut IrProgram<Viv>, op: BinOp, l: (NodeId, Ty, Span), r: (NodeId, Ty, Span), span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    use BinOp::*;
    let (lid, lty, lspan) = l;
    let (rid, rty, rspan) = r;
    // an Int beside a Float32
    if lty == Ty::F32 || rty == Ty::F32 {
        return Err(super::f32s::no_mix(op, if lty == Ty::F32 { rspan } else { lspan }, Ty::Int));
    }
    match (lty, rty) {
        (Ty::Int, Ty::Int) => Ok(match op {
            Add | Sub | Mul | FloorDiv | Mod => (int_arith(prog, op, span, lid, rid), Ty::Int),
            Div => {
                let (a, _) = to_num(prog, lid, Ty::Int, span);
                let (b, _) = to_num(prog, rid, Ty::Int, span);
                (prog.add(Viv::Bin { op, cplx: false }, span, vec![a, b]), Ty::Num)
            }
            Lt | Le | Gt | Ge | EqEq | Ne => (prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Bool),
            And | Or | Matmul => unreachable!("refused before"),
        }),
        // an Int beside a number: compared exactly, or converted exactly
        (Ty::Int, Ty::Num) | (Ty::Num, Ty::Int) if matches!(op, Lt | Le | Gt | Ge | EqEq | Ne) => {
            // cmp(i, x) is the sign of i - x (NaN for a NaN x); with the Int
            // on the right the comparison turns around
            let (i, x, op) = if lty == Ty::Int { (lid, rid, op) } else { (rid, lid, flip(op)) };
            let c = call(prog, "viv_i_cmp", span, vec![i, x]);
            let zero = prog.add(Viv::Num(0.0), span, vec![]);
            Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![c, zero]), Ty::Bool))
        }
        _ => {
            // a conversion that stops is located at the operator, as the
            // interpreter locates it
            let (a, at) = to_num(prog, lid, lty, span);
            let (b, bt) = to_num(prog, rid, rty, span);
            if strings::is_text(at) || strings::is_text(bt) {
                return strings::lower_binop(prog, op, (a, at, lspan), (b, bt, rspan), span);
            }
            if matches!(op, Mod | FloorDiv) && at == Ty::Num && bt == Ty::Num {
                let site = prog.add(Viv::Num(site_at(span) as f64), span, vec![]);
                let callee = if op == Mod { "viv_floormod" } else { "viv_floordiv" };
                return Ok((prog.add(Viv::Call { callee: callee.to_string() }, span, vec![a, b, site]), Ty::Num));
            }
            lower_binop(prog, op, (a, at, lspan), (b, bt, rspan), span)
        }
    }
}

fn flip(op: BinOp) -> BinOp {
    use BinOp::*;
    match op {
        Lt => Gt,
        Le => Ge,
        Gt => Lt,
        Ge => Le,
        o => o,
    }
}

/// The calls with an Int form, when an argument is an Int; `None` for
/// everything else (the caller's rules apply, unchanged).
pub(super) fn lower_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    if sigs.arity.contains_key(callee) {
        return None;
    }
    let any_int = args.iter().any(|a| peek_int(a, scope, sigs));
    let r = (|| -> Result<Option<(NodeId, Ty)>, Diagnostic> {
        let one = |prog: &mut IrProgram<Viv>| -> Result<NodeId, Diagnostic> {
            if args.len() != 1 {
                return Err(Diagnostic::error(format!("`{callee}` takes one value")).with_primary(span, "wrong number of arguments"));
            }
            lower_as_int(prog, &args[0], scope, sigs)
        };
        let two = |prog: &mut IrProgram<Viv>| -> Result<(NodeId, NodeId), Diagnostic> {
            if args.len() != 2 {
                return Err(Diagnostic::error(format!("`{callee}` takes two values")).with_primary(span, "wrong number of arguments"));
            }
            Ok((lower_as_int(prog, &args[0], scope, sigs)?, lower_as_int(prog, &args[1], scope, sigs)?))
        };
        // `ceil(a / b)` and `floor(a / b)` of two Ints: the exact quotient, rounded
        if let ("ceil" | "floor", [q]) = (callee, args) {
            if let Some((l, r)) = int_quotient(q, scope, sigs) {
                return Ok(Some((lower_quotient(prog, l, r, callee == "ceil", span, scope, sigs)?, Ty::Int)));
            }
        }
        Ok(Some(match callee {
            // always Int: the unsigned helpers
            "wrap_add" | "wrap_sub" | "wrap_mul" | "umulhi" => {
                let (a, b) = two(prog)?;
                (call(prog, &format!("viv_i_{callee}"), span, vec![a, b]), Ty::Int)
            }
            "ult" => {
                let (a, b) = two(prog)?;
                (call(prog, "viv_i_ult", span, vec![a, b]), Ty::Bool)
            }
            "ustr" => {
                let a = one(prog)?;
                (call(prog, "viv_i_ustr", span, vec![a]), Ty::Str)
            }
            "fround" => {
                // the nearest binary32
                if args.len() != 1 {
                    return Err(Diagnostic::error("`fround` takes one number").with_primary(span, "wrong number of arguments"));
                }
                let (x, xt) = lower_expr(prog, &args[0], scope, sigs)?;
                let (x, xt) = to_num(prog, x, xt, args[0].span());
                expect_num(xt, args[0].span(), "a number")?;
                (call(prog, "viv_i_fround", span, vec![x]), Ty::Num)
            }
            c if crate::interp::bytes::reader(c).is_some() => {
                // a little-endian reader: bytes, offset
                if args.len() != 2 {
                    return Err(Diagnostic::error(format!("`{c}` takes an array of bytes and an offset")).with_primary(span, "wrong number of arguments"));
                }
                let (b, bt) = lower_expr(prog, &args[0], scope, sigs)?;
                if bt != Ty::Arr(Elem::Int) {
                    return Err(Diagnostic::error(format!("`{c}` reads an array of Ints, one byte each (`read_bytes` gives one)")).with_primary(args[0].span(), "not an [Int]"));
                }
                let off = lower_as_int(prog, &args[1], scope, sigs)?;
                let num = crate::interp::bytes::reader(c).is_some_and(|(_, num)| num);
                (call(prog, &format!("viv_i_{c}"), span, vec![b, off]), if num { Ty::Num } else { Ty::Int })
            }
            "int_zeros" => {
                if args.len() != 1 {
                    return Err(Diagnostic::error("`int_zeros` takes one length").with_primary(span, "an array of Ints is rank 1"));
                }
                let (n, nt) = lower_expr(prog, &args[0], scope, sigs)?;
                let (n, nt) = to_num(prog, n, nt, args[0].span());
                expect_num(nt, args[0].span(), "a length")?;
                // an Int array's block is a number array's: 8 zero bytes each
                (prog.add(Viv::Zeros, span, vec![n]), Ty::Arr(Elem::Int))
            }
            _ if !any_int => return Ok(None),
            "abs" => {
                let a = one(prog)?;
                (call(prog, "viv_i_abs", span, vec![a]), Ty::Int)
            }
            "floor" | "ceil" | "round" | "int" if args.len() == 1 => (one(prog)?, Ty::Int),
            "float" if args.len() == 1 => {
                let a = one(prog)?;
                (call(prog, "viv_i_float", span, vec![a]), Ty::Num)
            }
            "min" | "max" if args.len() == 2 && args.iter().all(|a| peek_int(a, scope, sigs) || adapts(a, sigs)) => {
                let (a, b) = two(prog)?;
                (call(prog, &format!("viv_i_{callee}"), span, vec![a, b]), Ty::Int)
            }
            "pow" if args.len() == 2 && args.iter().all(|a| peek_int(a, scope, sigs) || adapts(a, sigs)) => {
                let (a, b) = two(prog)?;
                (call(prog, "viv_i_pow", span, vec![a, b]), Ty::Int)
            }
            "bnot" => {
                let a = one(prog)?;
                (call(prog, "viv_i_bnot", span, vec![a]), Ty::Int)
            }
            c if BITS.contains(&c) => {
                let (a, b) = two(prog)?;
                (call(prog, &format!("viv_i_{c}"), span, vec![a, b]), Ty::Int)
            }
            _ => return Ok(None),
        }))
    })();
    match r {
        Ok(Some(x)) => Some(Ok(x)),
        Ok(None) => None,
        Err(d) => Some(Err(d)),
    }
}

/// A builtin that takes numbers, called with an Int: the call with each Int
/// argument converted exactly (`sqrt(n)`). `None` when no argument is
/// certainly an Int.
pub(super) fn promote_args(callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Option<Vec<Expr>> {
    if sigs.arity.contains_key(callee) || callee == NUM_EXACT || !args.iter().any(|a| peek_int(a, scope, sigs)) {
        return None;
    }
    Some(
        args.iter()
            .map(|a| if peek_int(a, scope, sigs) { Expr::Call { callee: NUM_EXACT.to_string(), args: vec![a.clone()], span } } else { a.clone() })
            .collect(),
    )
}

/// The call `promote_args` wraps an Int argument in: the Int as a number,
/// exactly. It exists only inside the lowering, never in a program's tree.
pub(super) const NUM_EXACT: &str = "viv__num_exact";

/// `viv__num_exact(n)`.
pub(super) fn lower_num_exact(prog: &mut IrProgram<Viv>, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
    Ok(to_num(prog, id, ty, span))
}

/// An Int as text in a format (`print`, `str`, `fmt`): its digits for the
/// bare, repr and `d` formats, in `width` columns; a float format converts it
/// exactly and writes the number.
pub(super) fn text_or_num(prog: &mut IrProgram<Viv>, id: NodeId, f: crate::printfmt::Fmt, span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    let w = prog.add(Viv::Num(f.width as f64), span, vec![]);
    text_or_num_at(prog, id, f, w, span)
}

/// `text_or_num` with the width as a node (a value, `{n:>{w}}`).
pub(super) fn text_or_num_at(prog: &mut IrProgram<Viv>, id: NodeId, f: crate::printfmt::Fmt, w: NodeId, span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    use crate::printfmt::Kind;
    Ok(match f.kind {
        Kind::Bare | Kind::Repr | Kind::D | Kind::Str if f.flags == 0 => (call(prog, "viv_i_str", span, vec![id, w]), Ty::Str),
        // the flags of a Python spec (`+05d`, `,`, `>8`), or a width that
        // is a value: printfmt::int_text
        Kind::Bare | Kind::Repr | Kind::D | Kind::Str => {
            let k = prog.add(Viv::Num(f.word() as f64), span, vec![]);
            (call(prog, "viv_i_fmt", span, vec![id, k, w]), Ty::Str)
        }
        _ => to_num(prog, id, Ty::Int, span),
    })
}

// ---- inference ----------------------------------------------------------------

/// The type of `a op b` when an Int is on either side (an operand that
/// adapts is an Int beside an Int), or `None` to leave it to the rules for
/// numbers.
pub(super) fn infer_binary(op: BinOp, lt: Option<Ty>, rt: Option<Ty>, lhs: &Expr, rhs: &Expr, sigs: &Sigs) -> Option<Option<Ty>> {
    use BinOp::*;
    let lt = if lt == Some(Ty::Num) && rt == Some(Ty::Int) && adapts(lhs, sigs) { Some(Ty::Int) } else { lt };
    let rt = if rt == Some(Ty::Num) && lt == Some(Ty::Int) && adapts(rhs, sigs) { Some(Ty::Int) } else { rt };
    if lt != Some(Ty::Int) && rt != Some(Ty::Int) {
        return None;
    }
    Some(match op {
        Lt | Le | Gt | Ge | EqEq | Ne | And | Or => Some(Ty::Bool),
        Add | Sub | Mul | FloorDiv | Mod if lt == rt => Some(Ty::Int),
        Div if lt == rt => Some(Ty::Num),
        Matmul => None,
        _ => match (lt, rt) {
            (Some(Ty::Int), Some(t)) | (Some(t), Some(Ty::Int)) => Some(t),
            _ => None,
        },
    })
}

/// The type of a builtin's call when it has an Int form and an argument is
/// an Int; `None` to leave it to the other rules.
pub(super) fn infer_call(callee: &str, args: &[Expr], arg_tys: &[Option<Ty>], sigs: &Sigs) -> Option<Option<Ty>> {
    if sigs.arity.contains_key(callee) {
        return None;
    }
    let int_or_adapts = |k: usize| arg_tys.get(k).copied().flatten() == Some(Ty::Int) || args.get(k).is_some_and(|a| adapts(a, sigs));
    // `ceil(a / b)` and `floor(a / b)` of two Ints are Ints
    if let ("ceil" | "floor", [Expr::Binary { op: BinOp::Div, lhs, rhs, .. }]) = (callee, args) {
        if quotient_of_ints(lhs, rhs, sigs) {
            return Some(Some(Ty::Int));
        }
    }
    let any_int = arg_tys.iter().any(|t| *t == Some(Ty::Int));
    Some(match callee {
        "wrap_add" | "wrap_sub" | "wrap_mul" | "umulhi" => Some(Ty::Int),
        "ult" => Some(Ty::Bool),
        "ustr" => Some(Ty::Str),
        "int_zeros" => Some(Ty::Arr(Elem::Int)),
        "fround" => Some(Ty::Num),
        c if crate::interp::bytes::reader(c).is_some() => Some(if crate::interp::bytes::reader(c).is_some_and(|(_, num)| num) { Ty::Num } else { Ty::Int }),
        _ if !any_int => return None,
        "abs" | "floor" | "ceil" | "round" | "int" | "bnot" if args.len() == 1 => Some(Ty::Int),
        "float" => Some(Ty::Num),
        "min" | "max" | "pow" if args.len() == 2 && int_or_adapts(0) && int_or_adapts(1) => Some(Ty::Int),
        c if BITS.contains(&c) => Some(Ty::Int),
        _ => return None,
    })
}

/// Whether `a / b` divides two Ints, by the types inference has so far (an
/// operand that adapts counts beside an Int). Inference has no scope, so it
/// reads the types `INFER_TYS` holds for the def being walked.
pub(super) fn quotient_of_ints(lhs: &Expr, rhs: &Expr, sigs: &Sigs) -> bool {
    INFER_TYS.with(|t| {
        let t = t.borrow();
        let ty = |e: &Expr| match e {
            Expr::Var { name, .. } => t.get(name).copied(),
            _ => None,
        };
        let int = |e: &Expr| ty(e) == Some(Ty::Int) || matches!(e, Expr::Binary { lhs, .. } if ty(lhs) == Some(Ty::Int));
        let ok = |e: &Expr| int(e) || adapts(e, sigs);
        ok(lhs) && ok(rhs) && (int(lhs) || int(rhs))
    })
}

thread_local! {
    /// The variable types inference has for the statement it is reading
    /// (`quotient_of_ints`).
    pub(super) static INFER_TYS: std::cell::RefCell<HashMap<String, Ty>> = std::cell::RefCell::new(HashMap::new());
}

// ---- the C runtime -------------------------------------------------------------

/// What one argument or result of an Int runtime call is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IK {
    /// an Int: `long long` in C, an i64 in wasm
    I,
    /// a number: `double`, an f64
    N,
    /// a boolean: `double` (1.0 or 0.0), an f64
    B,
    /// a string: `viv_str`, an i32
    S,
    /// an array: `viv_arr`, an i32
    A,
}

/// One runtime call: its C name, arguments and result; `stops`: it can stop
/// the program, so C passes its viv_sites row last.
pub(crate) struct IFn {
    pub c: &'static str,
    pub args: &'static [IK],
    pub ret: IK,
    pub stops: bool,
}

use IK::{A, B, I, N, S};

/// Every runtime call Int code makes. The wasm backend imports each as
/// `vrt_i_<name>`.
pub(crate) const RUNTIME: &[IFn] = &[
    IFn { c: "viv_i_add", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_sub", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_mul", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_fdiv", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_fmod", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_cdiv", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_pow", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_neg", args: &[I], ret: I, stops: true },
    IFn { c: "viv_i_abs", args: &[I], ret: I, stops: true },
    IFn { c: "viv_i_num", args: &[I], ret: N, stops: true },
    IFn { c: "viv_i_of", args: &[N], ret: I, stops: true },
    IFn { c: "viv_i_cmp", args: &[I, N], ret: N, stops: false },
    IFn { c: "viv_i_float", args: &[I], ret: N, stops: false },
    IFn { c: "viv_i_min", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_max", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_band", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_bor", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_bxor", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_bnot", args: &[I], ret: I, stops: false },
    IFn { c: "viv_i_shl", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_shr", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_sar", args: &[I, I], ret: I, stops: true },
    IFn { c: "viv_i_wrap_add", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_wrap_sub", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_wrap_mul", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_umulhi", args: &[I, I], ret: I, stops: false },
    IFn { c: "viv_i_ult", args: &[I, I], ret: B, stops: false },
    IFn { c: "viv_i_str", args: &[I, N], ret: S, stops: false },
    IFn { c: "viv_i_fmt", args: &[I, N, N], ret: S, stops: false },
    IFn { c: "viv_i_ustr", args: &[I], ret: S, stops: false },
    IFn { c: "viv_iarr_str", args: &[A], ret: S, stops: false },
    // `a = append(a, n)` of an Int array, and its seal (interp/growable.rs)
    IFn { c: "viv_iappend", args: &[A, I, N], ret: A, stops: true },
    IFn { c: "viv_iseal", args: &[A], ret: A, stops: false },
    // Int arrays as real arrays are used (B343): extend and slices move the
    // 8-byte slots as the real versions do; sum, min, max and sort read Ints
    IFn { c: "viv_iextend", args: &[A, A, N], ret: A, stops: true },
    IFn { c: "viv_igetslice", args: &[A, A], ret: A, stops: true },
    IFn { c: "viv_isetslice", args: &[A, A, A], ret: N, stops: true },
    IFn { c: "viv_isum", args: &[A], ret: I, stops: true },
    IFn { c: "viv_iminv", args: &[A], ret: I, stops: true },
    IFn { c: "viv_imaxv", args: &[A], ret: I, stops: true },
    IFn { c: "viv_isort", args: &[A], ret: N, stops: false },
    // the little-endian readers (interp/bytes.rs)
    IFn { c: "viv_i_le_u8", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_i8", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_u16", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_i16", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_u32", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_i32", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_u64", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_i64", args: &[A, I], ret: I, stops: true },
    IFn { c: "viv_i_le_f16", args: &[A, I], ret: N, stops: true },
    IFn { c: "viv_i_le_bf16", args: &[A, I], ret: N, stops: true },
    IFn { c: "viv_i_le_f32", args: &[A, I], ret: N, stops: true },
    IFn { c: "viv_i_le_f64", args: &[A, I], ret: N, stops: true },
    IFn { c: "viv_i_fround", args: &[N], ret: N, stops: false },
];

pub(crate) fn rt_fn(c: &str) -> Option<&'static IFn> {
    RUNTIME.iter().find(|f| f.c == c)
}

/// `viv_i_*(args)`: the call, with its viv_sites row last when it can stop.
/// From this many elements an Int array literal is a table, not a variadic call.
const IVEC_TAB_MIN: usize = 16;

/// `(const long long[]){...}` for an Int array literal of at least
/// IVEC_TAB_MIN integer constants; None otherwise.
fn int_table(node: &Node<Viv>, cx: &CEmitCx<Viv>) -> Option<String> {
    if node.children.len() < IVEC_TAB_MIN {
        return None;
    }
    let mut out = String::with_capacity(node.children.len() * 8 + 24);
    out.push_str("(const long long[]){");
    for (i, &c) in node.children.iter().enumerate() {
        let v = match cx.kind(c) {
            Viv::Int(v) if v != i64::MIN => v,
            _ => return None,
        };
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{v}LL"));
    }
    out.push('}');
    Some(out)
}

/// The table form's constructor (see int_table).
pub(super) fn ivec_tab_fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "ivec_tab",
        includes: &["#include <string.h>"],
        deps: &["int_arr"],
        code: "static viv_arr viv_ivec_tab(int n, const long long* t) {\n    int dims[1]; dims[0] = n;\n    viv_arr a = viv_arr_new(1, dims);\n    memcpy(a.data, t, (size_t)n * sizeof(long long));\n    return a;\n}",
    }
}

pub(super) fn emit_call(node: &Node<Viv>, callee: &str, cx: &mut CEmitCx<Viv>) -> Option<CExpr> {
    let id = node.id;
    // an Int array's literal: n, then its elements
    if callee == "viv_ivec" {
        cx.need("int_arr");
        let n = node.children.len();
        let elems: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
        if let Some(init) = int_table(node, cx) {
            // (const long long[]){e0, e1, ...} instead of a variadic call:
            // clang's code for a call is superlinear in its argument count,
            // and past 65,535 arguments its arm64 code reads garbage (a
            // 115,484-element literal of node indices came back corrupted)
            cx.need("ivec_tab");
            return Some(CExpr::call(id, "viv_ivec_tab", vec![CExpr::lit(id, n.to_string()), CExpr::raw(id, init)]));
        }
        let mut args = vec![CExpr::lit(id, n.to_string())];
        args.extend(elems);
        return Some(CExpr::call(id, "viv_ivec", args));
    }
    // an Int array's element, read or written, checked as any element is
    if callee == "viv_iget" || callee == "viv_iset" {
        cx.need("int_arr");
        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
        let at = site_at(node.span);
        // an Int index (an Int variable or counter, an Int element, Int
        // arithmetic) is used as the 64-bit integer it is: converting it to
        // a double and back cost NPB CG's inner loop a third of its time
        let int_index = node.children.get(1).is_some_and(|&i| super::intidx::is_int_node(cx.program(), i));
        // a number index with an exact integer form (a loop counter, say):
        // the 64-bit path real arrays take (int_access in lower.rs), behind
        // the same guards, with the double path when a guard fails
        let set = callee == "viv_iset";
        if !int_index && node.children.len() == if set { 3 } else { 2 } {
            if let Some(r) = super::intidx::ready(cx, node.children[1]) {
                let mut mask = 0u32;
                if super::intidx::assumed(cx.program(), node.children[0], node.children[1]) {
                    mask = 1;
                } else if r.lo >= 0.0 {
                    if let (Some((sym, off)), Some(dim)) = (&r.ub, super::intidx::base_dim(cx.program(), node.children[0], 0)) {
                        if *sym == dim && *off <= 0 {
                            mask = 1;
                        }
                    }
                }
                super::intidx::note(&loc_text(node.span), 1, if r.guards.is_empty() { "int" } else { "guarded" }, mask);
                let base = node.children[0];
                let mut fast = vec![cx.expr(base), r.c];
                if set {
                    fast.push(cx.expr(node.children[2]));
                }
                fast.push(CExpr::lit(id, mask.to_string()));
                let f = with_site(if set { "viv_iset_m" } else { "viv_iget_m" }, at, node.span, &mut fast, id);
                let fast = CExpr::call(id, &f, fast);
                let Some(test) = super::intidx::guard_c(id, &r.guards) else { return Some(fast) };
                cx.need("libm");
                cx.need("trip");
                let f = with_site(callee, at, node.span, &mut args, id);
                return Some(CExpr::ternary(id, test, fast, CExpr::call(id, &f, args)));
            }
        }
        // an Int counter index the loop proves inside the array (its fixed
        // length, or the loop's checked-once version): no check
        if int_index && node.children.len() == if set { 3 } else { 2 } && super::intidx::ctr_access_proved(cx.program(), node.children[0], node.children[1]) {
            super::intidx::note(&loc_text(node.span), 1, "int", 1);
            let mut fast = vec![cx.expr(node.children[0]), cx.expr(node.children[1])];
            if set {
                fast.push(cx.expr(node.children[2]));
            }
            fast.push(CExpr::lit(id, "1"));
            let f = with_site(if set { "viv_iset_m" } else { "viv_iget_m" }, at, node.span, &mut fast, id);
            return Some(CExpr::call(id, &f, fast));
        }
        let callee = match (callee, int_index) {
            ("viv_iget", true) => "viv_iget_l",
            ("viv_iset", true) => "viv_iset_l",
            (c, _) => c,
        };
        let f = with_site(callee, at, node.span, &mut args, id);
        return Some(CExpr::call(id, &f, args));
    }
    let f = rt_fn(callee)?;
    // an operation the facts prove cannot leave 64 bits (or a conversion
    // that cannot fail) needs no check (lower/intidx.rs)
    if let Some(e) = unchecked(node, callee, cx) {
        return Some(e);
    }
    cx.need(match callee {
        "viv_i_str" | "viv_i_fmt" | "viv_i_ustr" | "viv_iarr_str" => "str_int",
        "viv_iappend" | "viv_iseal" => "growable",
        "viv_iextend" | "viv_igetslice" | "viv_isetslice" | "viv_isum" | "viv_iminv" | "viv_imaxv" | "viv_isort" => "int_arr_ops",
        c if c.starts_with("viv_i_le_") || c == "viv_i_fround" => "int_bytes",
        _ => "ints",
    });
    let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
    if f.stops {
        args.push(CExpr::raw(id, site_row(node.span).to_string()));
    }
    Some(CExpr::call(id, callee, args))
}

/// `lo + k`, the value of `for i from lo to hi` on pass k (parser.rs
/// build_for): k counts the `hi - lo` passes, and that difference was a
/// checked Int subtraction, so `lo + k` lies in `[lo, hi)` and cannot leave 64
/// bits. Both names are hidden ones the parser made for this loop alone, or
/// `lo` is a literal. NPB CG's row loop `for k from rowstr[j] to rowstr[j + 1]`
/// spent a sixth of its time on this check.
fn for_from_value(cx: &mut CEmitCx<Viv>, k: &[NodeId]) -> bool {
    let hidden = |c: NodeId, what: &str| match &cx.program().get(c).kind {
        Viv::Var { name } => name
            .strip_prefix(crate::parser::HIDDEN_PREFIX)
            .and_then(|r| r.strip_prefix(what))
            .is_some_and(|r| !r.is_empty() && r.chars().all(|c| c.is_ascii_digit())),
        _ => false,
    };
    k.len() == 2 && (hidden(k[0], "lo") || matches!(cx.program().get(k[0]).kind, Viv::Int(_))) && hidden(k[1], "k")
}

/// `a + b`, `a - b`, `a * b`, `-a` of Ints whose result the facts bound
/// within 2^62, a number converted to an Int that the facts prove an integer
/// in range, and an Int converted to a number that the facts bound by 2^53:
/// the plain C operation, which cannot overflow or fail there.
fn unchecked(node: &Node<Viv>, callee: &str, cx: &mut CEmitCx<Viv>) -> Option<CExpr> {
    use super::intidx::{fact, Av, Fact, EXACT};
    let id = node.id;
    let k = &node.children;
    let bounded = |a: Av, lim: f64| matches!(a, Av::F(Fact::Range { lo, hi, int: true }) if lo >= -lim && hi <= lim);
    match callee {
        "viv_i_add" | "viv_i_sub" | "viv_i_mul" => {
            if !matches!(fact(cx.program(), id), Av::F(Fact::Range { .. })) && !(callee == "viv_i_add" && for_from_value(cx, k)) {
                return None;
            }
            let op = match callee {
                "viv_i_add" => "+",
                "viv_i_sub" => "-",
                _ => "*",
            };
            let a = cx.expr(k[0]);
            let b = cx.expr(k[1]);
            Some(CExpr::binary(id, a, op, b))
        }
        "viv_i_neg" => {
            if !matches!(fact(cx.program(), id), Av::F(Fact::Range { .. })) {
                return None;
            }
            Some(CExpr::unary(id, "-", cx.expr(k[0])))
        }
        "viv_i_of" => {
            if !bounded(fact(cx.program(), k[0]), 4611686018427387904.0) {
                return None;
            }
            Some(CExpr::unary(id, "(long long)", cx.expr(k[0])))
        }
        "viv_i_num" => {
            if !bounded(fact(cx.program(), k[0]), EXACT) {
                return None;
            }
            Some(CExpr::unary(id, "(double)", cx.expr(k[0])))
        }
        _ => None,
    }
}

/// The C runtime of Ints: interp/ints.rs, line for line.
pub(super) fn fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "ints",
        includes: &["#include <stdio.h>", "#include <math.h>", "#include <unistd.h>", "#include <limits.h>", "#include <float.h>"],
        deps: &["sites", "fmt_g"],
        code: C_INTS,
    }
}

const C_INTS: &str = r#"/* Int: 64-bit, checked; every message is
   interp/ints.rs's, and in a checked build the location follows it */
#if defined(__GNUC__)
#define VIV_I_UNUSED __attribute__((unused))
#define VIV_I_COLD __attribute__((noinline, cold))
#else
#define VIV_I_UNUSED
#define VIV_I_COLD
#endif
VIV_I_UNUSED static void viv_i_fail(int kind, const char* m, int site) {
    viv_fflush(stdout);
#ifdef VIV_BOUNDS
    viv_errf("error: %s\n%s", m, viv_sites[site]);
#else
    (void)site;
    viv_errf("error: %s\n", m);
#endif
    viv_stop(kind);
}
VIV_I_UNUSED static void viv_i_ovf(long long a, const char* op, long long b, int site) {
    char m[160];
    snprintf(m, sizeof m, "integer overflow: %lld %s %lld does not fit in an Int", a, op, b);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
VIV_I_UNUSED static void viv_i_ovf1(const char* what, long long a, int site) {
    char m[160];
    snprintf(m, sizeof m, "integer overflow: %s(%lld) does not fit in an Int", what, a);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
#if defined(__GNUC__) || defined(__clang__)
VIV_I_UNUSED static long long viv_i_add(long long a, long long b, int site) { long long r; if (__builtin_add_overflow(a, b, &r)) viv_i_ovf(a, "+", b, site); return r; }
VIV_I_UNUSED static long long viv_i_sub(long long a, long long b, int site) { long long r; if (__builtin_sub_overflow(a, b, &r)) viv_i_ovf(a, "-", b, site); return r; }
VIV_I_UNUSED static int viv_i_mul_ovf(long long a, long long b, long long* r) { return __builtin_mul_overflow(a, b, r); }
#else
VIV_I_UNUSED static long long viv_i_add(long long a, long long b, int site) { if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b)) viv_i_ovf(a, "+", b, site); return a + b; }
VIV_I_UNUSED static long long viv_i_sub(long long a, long long b, int site) { if ((b < 0 && a > LLONG_MAX + b) || (b > 0 && a < LLONG_MIN + b)) viv_i_ovf(a, "-", b, site); return a - b; }
VIV_I_UNUSED static int viv_i_mul_ovf(long long a, long long b, long long* r) {
    if (a == 0 || b == 0) { *r = 0; return 0; }
    if (a == -1) { if (b == LLONG_MIN) return 1; *r = -b; return 0; }
    if (b == -1) { if (a == LLONG_MIN) return 1; *r = -a; return 0; }
    if (a > 0 ? (b > 0 ? a > LLONG_MAX / b : b < LLONG_MIN / a) : (b > 0 ? a < LLONG_MIN / b : a < LLONG_MAX / b)) return 1;
    *r = a * b;
    return 0;
}
#endif
VIV_I_UNUSED static long long viv_i_mul(long long a, long long b, int site) { long long r; if (viv_i_mul_ovf(a, b, &r)) viv_i_ovf(a, "*", b, site); return r; }
VIV_I_UNUSED static long long viv_i_fdiv(long long a, long long b, int site) {
    if (b == 0) viv_i_fail(VIV_E_ZERODIV, "division by zero in `a // b`", site);
    if (a == LLONG_MIN && b == -1) viv_i_ovf(a, "//", b, site);
    long long q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
/* the ceiling of a / b, exactly: -((-a) // b) without forming -a */
VIV_I_UNUSED static long long viv_i_cdiv(long long a, long long b, int site) {
    if (b == 0) viv_i_fail(VIV_E_ZERODIV, "division by zero in `a / b`", site);
    if (a == LLONG_MIN && b == -1) viv_i_ovf(a, "/", b, site);
    long long q = a / b;
    return (a % b != 0 && ((a < 0) == (b < 0))) ? q + 1 : q;
}
VIV_I_UNUSED static long long viv_i_fmod(long long a, long long b, int site) {
    if (b == 0) viv_i_fail(VIV_E_ZERODIV, "division by zero in `a % b`", site);
    if (b == -1) return 0;
    long long r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
VIV_I_UNUSED static long long viv_i_pow(long long a, long long k, int site) {
    if (k < 0) viv_i_fail(VIV_E_VALUE, "an Int to a negative power is not an Int; write float(n) ^ k", site);
    long long r = 1, b = a, e = k;
    while (e > 0) {
        if (e & 1) { if (viv_i_mul_ovf(r, b, &r)) viv_i_ovf(a, "^", k, site); }
        e >>= 1;
        if (e > 0) { if (viv_i_mul_ovf(b, b, &b)) viv_i_ovf(a, "^", k, site); }
    }
    return r;
}
VIV_I_UNUSED static long long viv_i_neg(long long a, int site) { if (a == LLONG_MIN) viv_i_ovf1("-", a, site); return -a; }
VIV_I_UNUSED static long long viv_i_abs(long long a, int site) { if (a == LLONG_MIN) viv_i_ovf1("abs", a, site); return a < 0 ? -a : a; }
VIV_I_UNUSED static double viv_i_num(long long a, int site) {
    if (a > 9007199254740992LL || a < -9007199254740992LL) {
        char m[160];
        snprintf(m, sizeof m, "converting %lld to a number would change it; write float(n) to round it", a);
        viv_i_fail(VIV_E_OVERFLOW, m, site);
    }
    return (double)a;
}
/* the messages of viv_i_of, out of line so that the check itself inlines
   into a hot loop (`l: Int = int(max(g1, g2))` once per sample in NPB EP) */
VIV_I_UNUSED VIV_I_COLD static void viv_i_of_fail(double x, int site) {
    char b[40], m[96];
    if (x != x) viv_i_fail(VIV_E_VALUE, "cannot convert float NaN to integer", site);
    if (x > DBL_MAX || x < -DBL_MAX) viv_i_fail(VIV_E_OVERFLOW, "cannot convert float infinity to integer", site);
    viv_rt_text(b, sizeof b, x);
    if (x - floor(x) != 0.0) { snprintf(m, sizeof m, "%s is not an integer", b); viv_i_fail(VIV_E_VALUE, m, site); }
    snprintf(m, sizeof m, "%s does not fit in an Int", b);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
VIV_I_UNUSED static inline long long viv_i_of(double x, int site) {
    /* NaN fails every comparison and an infinity the range check */
    if (!(x - floor(x) == 0.0 && x >= -9223372036854775808.0 && x < 9223372036854775808.0)) viv_i_of_fail(x, site);
    return (long long)x;
}
VIV_I_UNUSED static double viv_i_cmp(long long a, double x) {
    if (x != x) return x;
    if (a <= 9007199254740992LL && a >= -9007199254740992LL) { double y = (double)a; return y < x ? -1.0 : y > x ? 1.0 : 0.0; }
    if (x >= 9223372036854775808.0) return -1.0;
    if (x < -9223372036854775808.0) return 1.0;
    long long b = (long long)x;
    return a < b ? -1.0 : a > b ? 1.0 : 0.0;
}
VIV_I_UNUSED static double viv_i_float(long long a) { return (double)a; }
VIV_I_UNUSED static long long viv_i_min(long long a, long long b) { return b < a ? b : a; }
VIV_I_UNUSED static long long viv_i_max(long long a, long long b) { return b > a ? b : a; }
VIV_I_UNUSED static long long viv_i_band(long long a, long long b) { return a & b; }
VIV_I_UNUSED static long long viv_i_bor(long long a, long long b) { return a | b; }
VIV_I_UNUSED static long long viv_i_bxor(long long a, long long b) { return a ^ b; }
VIV_I_UNUSED static long long viv_i_bnot(long long a) { return ~a; }
VIV_I_UNUSED static void viv_i_shift_fail(const char* name, long long k, int site) {
    char m[96];
    snprintf(m, sizeof m, "`%s` takes a shift count of 0 or more, not %lld", name, k);
    viv_i_fail(VIV_E_VALUE, m, site);
}
VIV_I_UNUSED static long long viv_i_shl(long long a, long long k, int site) { if (k < 0) viv_i_shift_fail("shl", k, site); return k >= 64 ? 0 : (long long)((unsigned long long)a << k); }
VIV_I_UNUSED static long long viv_i_shr(long long a, long long k, int site) { if (k < 0) viv_i_shift_fail("shr", k, site); return k >= 64 ? 0 : (long long)((unsigned long long)a >> k); }
VIV_I_UNUSED static long long viv_i_sar(long long a, long long k, int site) { if (k < 0) viv_i_shift_fail("sar", k, site); return k >= 64 ? (a < 0 ? -1 : 0) : (a < 0 ? ~(~a >> k) : a >> k); }
VIV_I_UNUSED static long long viv_i_wrap_add(long long a, long long b) { return (long long)((unsigned long long)a + (unsigned long long)b); }
VIV_I_UNUSED static long long viv_i_wrap_sub(long long a, long long b) { return (long long)((unsigned long long)a - (unsigned long long)b); }
VIV_I_UNUSED static long long viv_i_wrap_mul(long long a, long long b) { return (long long)((unsigned long long)a * (unsigned long long)b); }
/* the high half of the 128-bit product of two unsigned 64-bit patterns, from
   32-bit halves (no __int128 needed) */
VIV_I_UNUSED static long long viv_i_umulhi(long long a, long long b) {
    unsigned long long x = (unsigned long long)a, y = (unsigned long long)b;
    unsigned long long x0 = x & 0xffffffffULL, x1 = x >> 32, y0 = y & 0xffffffffULL, y1 = y >> 32;
    unsigned long long p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
    unsigned long long mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    return (long long)(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32));
}
VIV_I_UNUSED static double viv_i_ult(long long a, long long b) { return (unsigned long long)a < (unsigned long long)b ? 1.0 : 0.0; }"#;

/// `viv_i_str` and `viv_i_ustr`: an Int as a string. They make strings, so
/// they live beside the strings' runtime (`str_core`).
pub(super) fn str_fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "str_int",
        includes: &["#include <stdio.h>", "#include <string.h>", "#include <stdlib.h>"],
        deps: &["str_core", "arr_type", "print"],
        code: r#"/* an Int as print, str and fmt(n, "d") write it: digits, right aligned in
   `width` columns (interp/ints.rs, text) */
VIV_S_UNUSED static viv_str viv_i_str(long long a, double width) {
    char b[160];
    int w = (int)width;
    int n = snprintf(b, sizeof b, "%*lld", w < 0 ? 0 : w > 100 ? 100 : w, a);
    return viv_s_make(b, n, 0);
}
/* an Int in a spec with flags (`+05d`, `,`): its digits, then viv_numfin
   (interp's printfmt::int_text) */
VIV_S_UNUSED static viv_str viv_i_fmt(long long a, double word, double width) {
    char b[32], o[1024];
    viv_s_fmtwp(-1.0, width, 40.0, 0);
    snprintf(b, sizeof b, "%lld", a);
    viv_numfin(o, b, (int)word, (int)width);
    return viv_s_make(o, (long)strlen(o), 0);
}
VIV_S_UNUSED static viv_str viv_i_ustr(long long a) {
    char b[32];
    int n = snprintf(b, sizeof b, "%llu", (unsigned long long)a);
    return viv_s_make(b, n, 0);
}
/* an Int array as print and show write it: [1, 2, 3] */
VIV_S_UNUSED static viv_str viv_iarr_str(viv_arr a) {
    long cap = 2 + 22 * (long)a.size, n = 0;
    char* b = (char*)malloc((size_t)cap);
    if (!b) viv_s_nomem();
    b[n++] = '[';
    for (int i = 0; i < a.size; i++) n += snprintf(b + n, (size_t)(cap - n), i ? ", %lld" : "%lld", ((long long*)a.data)[i]);
    b[n++] = ']';
    viv_str r = viv_s_make(b, n, 0);
    free(b);
    return r;
}"#,
    }
}

/// `extend`, slices, `sum`, `min`, `max` and `sort` of Int arrays
/// (B343): interp/ints.rs's `arr_op` and the wasm runtime's `vrt_i*`
/// check for check. Extend and slices are the real versions on the same
/// 8-byte slots, moved bit for bit.
pub(super) fn arr_ops_fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "int_arr_ops",
        includes: &["#include <stdlib.h>"],
        deps: &["ints", "int_arr", "growable", "slices"],
        code: r#"VIV_I_UNUSED static viv_arr viv_iextend(viv_arr a, viv_arr b, double own, int site) { return viv_extend(a, b, own, site); }
VIV_I_UNUSED static viv_arr viv_igetslice(viv_arr a, viv_arr spec, int site) { return viv_getslice(a, spec, 1.0, site); }
VIV_I_UNUSED static double viv_isetslice(viv_arr a, viv_arr spec, viv_arr v, int site) { return viv_setslice(a, spec, v, 1.0, site); }
VIV_I_UNUSED static long long viv_isum(viv_arr a, int site) {
    const long long* x = (const long long*)a.data;
    long long s = 0;
    for (int i = 0; i < a.size; i++) s = viv_i_add(s, x[i], site);
    return s;
}
VIV_I_UNUSED static long long viv_iext(viv_arr a, int big, int site) {
    const long long* x = (const long long*)a.data;
    if (a.size == 0) viv_i_fail(VIV_E_VALUE, big ? "`max` of an empty Int array" : "`min` of an empty Int array", site);
    long long m = x[0];
    for (int i = 1; i < a.size; i++) if (big ? x[i] > m : x[i] < m) m = x[i];
    return m;
}
VIV_I_UNUSED static long long viv_iminv(viv_arr a, int site) { return viv_iext(a, 0, site); }
VIV_I_UNUSED static long long viv_imaxv(viv_arr a, int site) { return viv_iext(a, 1, site); }
VIV_I_UNUSED static int viv_icmp(const void* p, const void* q) {
    long long a = *(const long long*)p, b = *(const long long*)q;
    return (a > b) - (a < b);
}
VIV_I_UNUSED static double viv_isort(viv_arr a) {
    if (a.size > 1) qsort(a.data, (size_t)a.size, sizeof(long long), viv_icmp);
    return 0.0;
}"#,
    }
}

/// The little-endian readers: interp/bytes.rs,
/// line for line.
pub(super) fn bytes_fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "int_bytes",
        includes: &["#include <stdio.h>", "#include <string.h>"],
        deps: &["ints", "int_arr"],
        code: r#"/* the `w` bytes at `off`, one unsigned pattern, little-endian; each element
   checked to be a byte and the window to be inside */
VIV_I_UNUSED static unsigned long long viv_le_pat(const char* name, viv_arr a, long long off, int w, int site) {
    long long n = a.size;
    const long long* x = (const long long*)a.data;
    char m[200];
    if (off < 0 || off > n - w) {
        snprintf(m, sizeof m, "`%s`: %d bytes at offset %lld are not inside an array of %lld", name, w, off, n);
        viv_i_fail(VIV_E_INDEX, m, site);
    }
    unsigned long long u = 0;
    for (int k = 0; k < w; k++) {
        long long v = x[off + k];
        if (v < 0 || v > 255) {
            snprintf(m, sizeof m, "`%s`: element %lld is %lld, not a byte (0 to 255)", name, off + k, v);
            viv_i_fail(VIV_E_VALUE, m, site);
        }
        u |= (unsigned long long)v << (8 * k);
    }
    return u;
}
VIV_I_UNUSED static double viv_le_bits(unsigned long long b) { double d; memcpy(&d, &b, 8); return d; }
/* a binary16 as the double it is, exactly (subnormals, infinities, NaN payloads) */
VIV_I_UNUSED static double viv_le_half(unsigned long long h) {
    unsigned long long s = (h >> 15) & 1, e = (h >> 10) & 0x1f, mt = h & 0x3ff, bits;
    if (e == 0) {
        if (mt == 0) bits = s << 63;
        else {
            unsigned long long lead = 9;
            while (!((mt >> lead) & 1)) lead--;
            bits = (s << 63) | ((lead + 1023 - 24) << 52) | ((mt << (52 - lead)) & ((1ULL << 52) - 1));
        }
    }
    else if (e == 0x1f) bits = (s << 63) | (0x7ffULL << 52) | (mt << 42);
    else bits = (s << 63) | ((e + 1023 - 15) << 52) | (mt << 42);
    return viv_le_bits(bits);
}
VIV_I_UNUSED static double viv_le_single(unsigned long long u) { unsigned int w = (unsigned int)u; float f; memcpy(&f, &w, 4); return (double)f; }
VIV_I_UNUSED static long long viv_i_le_u8(viv_arr a, long long o, int site) { return (long long)viv_le_pat("le_u8", a, o, 1, site); }
VIV_I_UNUSED static long long viv_i_le_i8(viv_arr a, long long o, int site) { return (long long)(signed char)viv_le_pat("le_i8", a, o, 1, site); }
VIV_I_UNUSED static long long viv_i_le_u16(viv_arr a, long long o, int site) { return (long long)viv_le_pat("le_u16", a, o, 2, site); }
VIV_I_UNUSED static long long viv_i_le_i16(viv_arr a, long long o, int site) { return (long long)(short)viv_le_pat("le_i16", a, o, 2, site); }
VIV_I_UNUSED static long long viv_i_le_u32(viv_arr a, long long o, int site) { return (long long)viv_le_pat("le_u32", a, o, 4, site); }
VIV_I_UNUSED static long long viv_i_le_i32(viv_arr a, long long o, int site) { return (long long)(int)viv_le_pat("le_i32", a, o, 4, site); }
VIV_I_UNUSED static long long viv_i_le_i64(viv_arr a, long long o, int site) { return (long long)viv_le_pat("le_i64", a, o, 8, site); }
VIV_I_UNUSED static long long viv_i_le_u64(viv_arr a, long long o, int site) {
    unsigned long long u = viv_le_pat("le_u64", a, o, 8, site);
    if (u > 9223372036854775807ULL) {
        char m[160];
        snprintf(m, sizeof m, "`le_u64`: %llu does not fit in an Int; le_i64 reads the same bits", u);
        viv_i_fail(VIV_E_OVERFLOW, m, site);
    }
    return (long long)u;
}
VIV_I_UNUSED static double viv_i_le_f16(viv_arr a, long long o, int site) { return viv_le_half(viv_le_pat("le_f16", a, o, 2, site)); }
VIV_I_UNUSED static double viv_i_le_bf16(viv_arr a, long long o, int site) { return viv_le_single(viv_le_pat("le_bf16", a, o, 2, site) << 16); }
VIV_I_UNUSED static double viv_i_le_f32(viv_arr a, long long o, int site) { return viv_le_single(viv_le_pat("le_f32", a, o, 4, site)); }
VIV_I_UNUSED static double viv_i_le_f64(viv_arr a, long long o, int site) { return viv_le_bits(viv_le_pat("le_f64", a, o, 8, site)); }
/* fround(x): the nearest binary32, ties to even (JavaScript's Math.fround) */
VIV_I_UNUSED static double viv_i_fround(double x) { volatile float f = (float)x; return (double)f; }"#,
    }
}

/// The elements of an Int array: read and written as `int64_t`, through the
/// bounds checks every element goes through (`viv_ix`).
pub(super) fn arr_fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "int_arr",
        includes: &["#include <stdarg.h>"],
        deps: &["arr_type", "bounds", "arr_new"],
        code: "static viv_arr viv_ivec(int n, ...) {\n    int dims[1]; dims[0] = n;\n    viv_arr a = viv_arr_new(1, dims);\n    va_list ap; va_start(ap, n);\n    for (int i = 0; i < n; i++) ((long long*)a.data)[i] = va_arg(ap, long long);\n    va_end(ap);\n    return a;\n}\nVIV_ACCESS long long viv_iget_at(viv_arr a, double i, int site) {\n    viv_rank(a.rank, 1, site);\n    return ((long long*)a.data)[viv_ix(i, a.dims[0], 0, site)];\n}\nVIV_ACCESS long long viv_iget(viv_arr a, double i) { return viv_iget_at(a, i, 0); }\nVIV_ACCESS void viv_iset_at(viv_arr a, double i, long long v, int site) {\n    viv_rank(a.rank, 1, site);\n    ((long long*)a.data)[viv_ix(i, a.dims[0], 0, site)] = v;\n}\nVIV_ACCESS void viv_iset(viv_arr a, double i, long long v) { viv_iset_at(a, i, v, 0); }\n/* the same with an index that is an Int: no trip through a double */\nVIV_ACCESS long long viv_iget_l_at(viv_arr a, long long i, int site) {\n    viv_rank(a.rank, 1, site);\n    return ((long long*)a.data)[viv_ixl(i, a.dims[0], 0, site)];\n}\nVIV_ACCESS void viv_iset_l_at(viv_arr a, long long i, long long v, int site) {\n    viv_rank(a.rank, 1, site);\n    ((long long*)a.data)[viv_ixl(i, a.dims[0], 0, site)] = v;\n}\nVIV_ACCESS long long viv_iget_l(viv_arr a, long long i) { return viv_iget_l_at(a, i, 0); }\n/* a number index with an exact integer form (lower/intidx.rs): as an Int,\n   with bit 0 of m set when the compiler proved it inside the array */\nVIV_ACCESS long long viv_iget_m_at(viv_arr a, long long i, int m, int site) {\n    viv_rank(a.rank, 1, site);\n    return ((long long*)a.data)[viv_ixp(m, 0, i, a.dims[0], site)];\n}\nVIV_ACCESS void viv_iset_m_at(viv_arr a, long long i, long long v, int m, int site) {\n    viv_rank(a.rank, 1, site);\n    ((long long*)a.data)[viv_ixp(m, 0, i, a.dims[0], site)] = v;\n}\nVIV_ACCESS long long viv_iget_m(viv_arr a, long long i, int m) { return viv_iget_m_at(a, i, m, 0); }\nVIV_ACCESS void viv_iset_m(viv_arr a, long long i, long long v, int m) { viv_iset_m_at(a, i, v, m, 0); }\nVIV_ACCESS void viv_iset_l(viv_arr a, long long i, long long v) { viv_iset_l_at(a, i, v, 0); }",
    }
}

// ---- variables that hold what int() gives -----------------------------------

/// `n = int(x)`: a variable written without a type whose every assignment
/// is an Int -- `int(...)`, an integer literal, another such variable or an
/// Int, or `+ - * // %` of those -- and at least one of them a call of
/// `int`, is an Int, as Python's `int` gives one. Such a variable is never
/// given a fraction, so holding it as an Int changes no value; a variable
/// that is (`k = int(x)`, then `k = k / 2`) stays a number. `len`, `floor`,
/// `round` and `ceil` keep their meaning (D3): they adapt where an Int goes,
/// and are numbers elsewhere. The program comes back with
/// `viv__declare(name, "Int")` at the top of each def (or of the program)
/// for each such variable; `None` when there is none.
pub(super) fn infer_int_vars(program: &Program) -> Option<Program> {
    // the hidden variable crate::surface's `early` computes an operand into,
    // to keep it before a lifted if-expression to its right: an Int when
    // what it holds is one, as `n = int(...)` is (B308: `int(g(x)) - (k if
    // c else k)` was Int minus a number)
    fn early_temp(n: &str) -> bool {
        n.strip_prefix(crate::parser::HIDDEN_PREFIX).and_then(|r| r.strip_prefix('e')).is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
    }
    use std::collections::HashSet;
    let user_int = program.items.iter().any(|i| matches!(i, Item::Def(d) if d.name == "int"));
    if user_int {
        return None;
    }
    // every assignment in a scope, by name; names that cannot be candidates
    fn walk<'a>(body: &'a [Stmt], asg: &mut HashMap<String, Vec<&'a Expr>>, not: &mut HashSet<String>, ints: &mut HashSet<String>) {
        for s in body {
            match s {
                Stmt::Assign { name, value, .. } => asg.entry(name.clone()).or_default().push(value),
                Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } if callee == crate::parser::DECLARE => {
                    if let [Expr::Str { value: n, .. }, Expr::Str { value: t, .. }] = args.as_slice() {
                        not.insert(n.clone());
                        if t == "Int" {
                            ints.insert(n.clone());
                        }
                    }
                }
                Stmt::SetIndex { base, .. } => {
                    if let Expr::Var { name, .. } = &**base {
                        not.insert(name.clone());
                    }
                }
                Stmt::For { var, body, .. } => {
                    not.insert(var.clone());
                    walk(body, asg, not, ints);
                }
                Stmt::If { then_body, else_body, .. } => {
                    walk(then_body, asg, not, ints);
                    walk(else_body, asg, not, ints);
                }
                Stmt::While { body, .. } | Stmt::Test { body, .. } => walk(body, asg, not, ints),
                _ => {}
            }
        }
    }
    fn integral(e: &Expr, cands: &HashSet<String>, ints: &HashSet<String>) -> bool {
        match e {
            Expr::Call { callee, args, .. } if callee == "int" => args.len() == 1,
            Expr::Call { callee, args, .. } if callee == "abs" => args.len() == 1 && integral(&args[0], cands, ints),
            Expr::Call { callee, args, .. } if callee == "min" || callee == "max" => args.len() == 2 && args.iter().all(|a| integral(a, cands, ints)),
            Expr::Call { .. } => false,
            Expr::Num { span, .. } => whole(*span).is_some(),
            Expr::Var { name, .. } => cands.contains(name) || ints.contains(name),
            Expr::Neg { operand, .. } => integral(operand, cands, ints),
            Expr::Binary { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod, lhs, rhs, .. } => {
                integral(lhs, cands, ints) && integral(rhs, cands, ints)
            }
            _ => false,
        }
    }
    let scope_ints = |body: &[Stmt], params: &[String], int_params: &[String]| -> Vec<String> {
        let mut asg = HashMap::new();
        let mut not: HashSet<String> = params.iter().cloned().collect();
        let mut ints: HashSet<String> = int_params.iter().cloned().collect();
        walk(body, &mut asg, &mut not, &mut ints);
        let mut cands: HashSet<String> = asg
            .iter()
            .filter(|(n, vs)| !not.contains(*n) && (!n.starts_with("viv__") || early_temp(n)) && vs.iter().any(|v| has_int_call(v)))
            .map(|(n, _)| n.clone())
            .collect();
        loop {
            let keep: HashSet<String> = cands.iter().filter(|n| asg[*n].iter().all(|v| integral(v, &cands, &ints))).cloned().collect();
            if keep.len() == cands.len() {
                break;
            }
            cands = keep;
        }
        let mut out: Vec<String> = cands.into_iter().collect();
        out.sort();
        out
    };
    let declare = |name: &str, span: Span| Stmt::ExprStmt {
        expr: Expr::Call {
            callee: crate::parser::DECLARE.to_string(),
            args: vec![Expr::Str { value: name.to_string(), span }, Expr::Str { value: "Int".to_string(), span }],
            span,
        },
        span,
    };
    let mut changed = false;
    let mut out = Program { items: program.items.clone(), debug: program.debug.clone(), memory: program.memory.clone(), phrases: program.phrases.clone(), whole: program.whole.clone() };
    for item in out.items.iter_mut() {
        if let Item::Def(d) = item {
            let params: Vec<String> = d.params.iter().map(|p| p.name.clone()).collect();
            let int_params: Vec<String> = d.params.iter().filter(|p| matches!(p.ty, Some(crate::ast::TypeExpr::Int))).map(|p| p.name.clone()).collect();
            let names = scope_ints(&d.body, &params, &int_params);
            if !names.is_empty() {
                changed = true;
                let mut body: Vec<Stmt> = names.iter().map(|n| declare(n, d.name_span)).collect();
                body.append(&mut d.body);
                d.body = body;
            }
        }
    }
    let top: Vec<Stmt> = out.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    let names = scope_ints(&top, &[], &[]);
    if !names.is_empty() {
        changed = true;
        let Some(at) = top.first().map(|s| s.span()) else { return None };
        let mut items: Vec<Item> = names.iter().map(|n| Item::Stmt(declare(n, at))).collect();
        items.append(&mut out.items);
        out.items = items;
    }
    changed.then_some(out)
}
