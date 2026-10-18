// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Strings in the checked lowering and the C backend.
//!
//! The rules are `crate::interp::strings`'s; this file types string
//! expressions, turns them into calls of the runtime (`viv_s_*` for a string,
//! `viv_ss_*` for a list of strings), and supplies the C runtime, the C twin
//! of the rules (`interp/strings.c`). A program without strings reaches none
//! of it: its IR and its C are what they were.

use super::*;
use crate::interp::strings::{self as rules, StrSpec, D, K};
use knotkit::backend::RuntimeFragment;

pub(super) fn ty_of(k: K) -> Ty {
    match k {
        K::N => Ty::Num,
        K::B => Ty::Bool,
        K::S => Ty::Str,
        K::L => Ty::Arr(Elem::Str),
        K::A => Ty::Arr(Elem::Num),
        K::I => Ty::Arr(Elem::Int),
    }
}

/// The kinds only the `db_*` builtins take (crate::db).
fn db_kind(name: &str, t: Ty) -> Option<K> {
    match t {
        _ if !name.starts_with("db_") => None,
        Ty::Arr(Elem::Num) => Some(K::A),
        Ty::Arr(Elem::Int) => Some(K::I),
        _ => None,
    }
}

pub(super) fn kind_of(t: Ty) -> Option<K> {
    match t {
        Ty::Num => Some(K::N),
        Ty::Bool => Some(K::B),
        Ty::Str => Some(K::S),
        Ty::Arr(Elem::Str) => Some(K::L),
        _ => None,
    }
}

/// The calls that write an Int's digits (`str`, `repr`, `fmt` and the forms
/// f-strings use) rather than take it as a number.
pub(super) fn takes_ints(callee: &str) -> bool {
    matches!(callee, "str" | "repr" | "fmt") || callee == rules::STR_OF || callee == rules::REPR_OF || callee == rules::FMT_OF
}

/// A string, or a list of strings.
pub(super) fn is_text(t: Ty) -> bool {
    matches!(t, Ty::Str | Ty::Arr(Elem::Str))
}

// ---- inference ----------------------------------------------------------------------

/// The type of `a op b` when a string or a list of strings is on either side.
pub(super) fn infer_binary(op: BinOp, lt: Option<Ty>, rt: Option<Ty>) -> Option<Ty> {
    let text = |t: Option<Ty>| t.is_some_and(is_text);
    if !(text(lt) || text(rt)) {
        return None;
    }
    let list = |t: Option<Ty>| t == Some(Ty::Arr(Elem::Str));
    match op {
        BinOp::Add => Some(if list(lt) || list(rt) { Ty::Arr(Elem::Str) } else { Ty::Str }),
        BinOp::Mul => Some(Ty::Str),
        BinOp::EqEq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Some(Ty::Bool),
        _ => None,
    }
}

/// The registry's entries for `name` that a call with these argument types
/// could be.
fn candidates(name: &str, arg_tys: &[Option<Ty>]) -> Vec<&'static rules::Builtin> {
    rules::BUILTINS
        .iter()
        .filter(|b| {
            b.name == name
                && arg_tys.len() <= b.args.len()
                && arg_tys.len() >= b.args.len() - b.opt.len()
                && arg_tys.iter().zip(b.args).all(|(t, k)| match t {
                    None => true,
                    // an Int where a number goes is converted exactly (ints::promote_args)
                    Some(t) => kind_of(*t).or_else(|| db_kind(name, *t)) == Some(*k) || (*k == K::L && *t == Ty::Arr(Elem::Num)) || (*k == K::N && *t == Ty::Int),
                })
        })
        .collect()
}

/// The type of a call that strings decide: `Some(t)` (t `None` while an
/// argument's type is not known), or `None` for a call that is not theirs.
pub(super) fn infer_call(callee: &str, arg_tys: &[Option<Ty>], sigs: &Sigs) -> Option<Option<Ty>> {
    if sigs.arity.contains_key(callee) {
        return None;
    }
    let a0 = arg_tys.first().copied().flatten();
    let a1 = arg_tys.get(1).copied().flatten();
    match callee {
        rules::FSTR | rules::STR_OF | rules::REPR_OF | rules::FMT_OF | "str" | "fmt" => Some(Some(Ty::Str)),
        rules::IN => Some(Some(Ty::Bool)),
        "repr" => Some(Some(Ty::Str)),
        c if c == SLICE_GET && a0 == Some(Ty::Str) => Some(Some(Ty::Str)),
        "append" | "extend" => match (a0, a1) {
            (Some(Ty::Arr(Elem::Str)), _) | (_, Some(Ty::Str)) | (_, Some(Ty::Arr(Elem::Str))) => Some(Some(Ty::Arr(Elem::Str))),
            (None, None) if !sigs.index_default => Some(None),
            _ => None,
        },
        c if rules::is_builtin(c) => {
            let cs = candidates(c, arg_tys);
            match cs.first() {
                Some(b) if cs.iter().all(|x| x.ret == b.ret) => Some(Some(ty_of(b.ret))),
                Some(_) => Some(None),
                None => None,
            }
        }
        _ => None,
    }
}

/// The type the registry says argument `j` of `callee` has, when every entry
/// of that name agrees.
pub(super) fn param_ty(callee: &str, j: usize) -> Option<Ty> {
    let mut ks = rules::BUILTINS.iter().filter(|b| b.name == callee && j < b.args.len()).map(|b| b.args[j]);
    let first = ks.next()?;
    ks.all(|k| k == first).then(|| ty_of(first))
}

// ---- lowering -----------------------------------------------------------------------------

/// The type `e` lowers to, found by lowering it into a scratch program (so a
/// builtin that takes arrays and strings alike can choose before it lowers).
fn peek(e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Option<Ty> {
    lower_expr(&mut IrProgram::new(), e, scope, sigs).ok().map(|(_, t)| t)
}

fn call(prog: &mut IrProgram<Viv>, name: &str, span: Span, kids: Vec<NodeId>) -> NodeId {
    prog.add(Viv::Call { callee: name.to_string() }, span, kids)
}

fn num(prog: &mut IrProgram<Viv>, span: Span, v: f64) -> NodeId {
    prog.add(Viv::Num(v), span, vec![])
}

fn lit(prog: &mut IrProgram<Viv>, span: Span, s: &str) -> NodeId {
    prog.add(Viv::StrLit(s.to_string()), span, vec![])
}

fn a_or_an(t: Ty) -> String {
    let n = t.name();
    format!("{} {n}", if n.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" })
}

/// `e`, where a value of type `want` goes: `[]` is an empty list of strings
/// when `want` is one.
pub(super) fn lower_expected(prog: &mut IrProgram<Viv>, e: &Expr, want: Option<Ty>, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    // an integer literal where an Int is bound
    if let Some(r) = super::ints::lower_for(prog, e, want, scope, sigs) {
        return r;
    }
    // `None` and `Some(x)` where an optional is bound
    if let Some(r) = super::enums::lower_for(prog, e, want, scope, sigs) {
        return r;
    }
    // `map()` where a map of strings goes makes one
    if let (Some(t @ Ty::Map(..)), Expr::Call { callee, args, span }) = (want, e) {
        if callee == "map" && args.is_empty() && !sigs.arity.contains_key("map") {
            return Ok(lower_new_map(prog, t, *span));
        }
    }
    // `[]` where an array of records goes: an empty one
    if let (Some(t @ Ty::Recs(rec)), Expr::ArrLit { elems, span }) = (want, e) {
        if elems.is_empty() {
            return Ok((prog.add(Viv::RecArrLit { rec: rec.to_string() }, *span, vec![]), t));
        }
    }
    if let (Some(Ty::Recs(_)), Expr::Call { callee, args, .. }) = (want, e) {
        if callee == crate::growable::SEAL && args.len() == 1 {
            return lower_expected(prog, &args[0], want, scope, sigs);
        }
    }
    if want == Some(Ty::Arr(Elem::Str)) {
        if let Expr::ArrLit { elems, span } = e {
            if elems.is_empty() {
                return Ok((call(prog, "viv_ss_new", *span, vec![]), Ty::Arr(Elem::Str)));
            }
        }
        // the growable pass seals what a tracked list is bound to: `[]` inside it
        if let Expr::Call { callee, args, span } = e {
            if callee == crate::growable::SEAL && args.len() == 1 {
                let (id, t) = lower_expected(prog, &args[0], want, scope, sigs)?;
                return Ok(if t == Ty::Arr(Elem::Str) { (call(prog, "viv_ss_seal", *span, vec![id]), t) } else { (id, t) });
            }
        }
    }
    lower_expr(prog, e, scope, sigs)
}

/// `[a, b, c]` of strings.
pub(super) fn lower_list_literal(prog: &mut IrProgram<Viv>, elems: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let mut ids = Vec::with_capacity(elems.len());
    for e in elems {
        let (id, t) = lower_expr(prog, e, scope, sigs)?;
        if t != Ty::Str {
            return Err(Diagnostic::error(format!("a list of strings cannot hold {}", a_or_an(t))).with_primary(e.span(), format!("this is {}", a_or_an(t))));
        }
        ids.push(id);
    }
    Ok((call(prog, "viv_ss_of", span, ids), Ty::Arr(Elem::Str)))
}

/// `s[i]` of a string, `l[i]` of a list of strings.
pub(super) fn lower_index(prog: &mut IrProgram<Viv>, bid: NodeId, bty: Ty, indices: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    if indices.len() != 1 {
        return Err(Diagnostic::error(format!("{} takes one index; this has {}", a_or_an(bty), indices.len())).with_primary(span, "one index"));
    }
    let (iid, ity) = lower_expr(prog, &indices[0], scope, sigs)?;
    let (iid, ity) = super::ints::to_num(prog, iid, ity, indices[0].span());
    expect_num(ity, indices[0].span(), "an index")?;
    let f = if bty == Ty::Str { "viv_s_at" } else { "viv_ss_at" };
    Ok((call(prog, f, span, vec![bid, iid]), Ty::Str))
}

/// `a op b` with a string or a list of strings on either side.
pub(super) fn lower_binop(prog: &mut IrProgram<Viv>, op: BinOp, (lid, lty, lsp): (NodeId, Ty, Span), (rid, rty, rsp): (NodeId, Ty, Span), span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    use BinOp::*;
    let s = Ty::Str;
    let l = Ty::Arr(Elem::Str);
    let bad = || {
        let at = if is_text(lty) { rsp } else { lsp };
        let t = if is_text(lty) { rty } else { lty };
        Diagnostic::error(format!("`{}` does not apply to {} and {}", op.c_op(), a_or_an(lty), a_or_an(rty))).with_primary(at, format!("this is {}", a_or_an(t)))
    };
    Ok(match op {
        Add if lty == s && rty == s => (call(prog, "viv_s_cat", span, vec![lid, rid]), s),
        Add if lty == l && rty == l => (call(prog, "viv_ss_cat", span, vec![lid, rid]), l),
        Mul if lty == s && rty == Ty::Num => (call(prog, "viv_s_rep", span, vec![lid, rid]), s),
        Mul if lty == Ty::Num && rty == s => (call(prog, "viv_s_rep", span, vec![rid, lid]), s),
        EqEq | Ne if lty == s && rty == s => {
            let eq = call(prog, "viv_s_eq", span, vec![lid, rid]);
            let one = num(prog, span, 1.0);
            (prog.add(Viv::Bin { op, cplx: false }, span, vec![eq, one]), Ty::Bool)
        }
        EqEq | Ne if lty == l && rty == l => {
            let eq = call(prog, "viv_ss_eq", span, vec![lid, rid]);
            let one = num(prog, span, 1.0);
            (prog.add(Viv::Bin { op, cplx: false }, span, vec![eq, one]), Ty::Bool)
        }
        Lt | Le | Gt | Ge if lty == s && rty == s => {
            // viv_s_cmp is -1, 0 or 1: its sign against 0 is the order
            let c = call(prog, "viv_s_cmp", span, vec![lid, rid]);
            let zero = num(prog, span, 0.0);
            (prog.add(Viv::Bin { op, cplx: false }, span, vec![c, zero]), Ty::Bool)
        }
        _ => return Err(bad()),
    })
}

/// The value form of `str`, `repr` and `fmt` (and an f-string's fields).
fn lower_text_of(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let is_fmt = callee == "fmt" || callee == rules::FMT_OF;
    let is_repr = callee == "repr" || callee == rules::REPR_OF;
    let shown = if is_fmt { "fmt" } else if is_repr { "repr" } else { "str" };
    if args.is_empty() || (!is_fmt && args.len() != 1) {
        return Err(Diagnostic::error(format!("`{shown}` takes one value{}", if is_fmt { " and at most one format" } else { "" })).with_primary(span, "wrong number of arguments"));
    }
    let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
    let s = Ty::Str;
    if is_fmt && args.len() > 1 {
        // a width or precision given as a value (`fmt(x, ".*f", p)`, an
        // f-string's `{x:.{p}f}`): its node, a number; the engines check it
        let mut stars = Vec::new();
        for a in args.iter().skip(2) {
            let (sid, sty) = lower_expr(prog, a, scope, sigs)?;
            let (sid, sty) = super::ints::to_num(prog, sid, sty, a.span());
            expect_num(sty, a.span(), "a format's width or precision")?;
            stars.push(sid);
        }
        if is_text(ty) {
            let Expr::Str { value, span: ssp } = &args[1] else {
                return Err(Diagnostic::error("`fmt` of a string takes a format string, written in the source, such as \">10\" or \".3\"")
                    .with_primary(args[1].span(), "not a string literal"));
            };
            let spec = StrSpec::parse(value).map_err(|m| Diagnostic::error(format!("format \"{value}\": {m}")).with_primary(*ssp, "not a string format"))?;
            if spec.star_count() != stars.len() {
                return Err(Diagnostic::error("`fmt` takes a value and at most one format").with_primary(span, "wrong number of arguments"));
            }
            let text = if ty == s { id } else { call(prog, "viv_ss_repr", span, vec![id]) };
            let mut kids = vec![text];
            let mut it = stars.into_iter();
            for (k, v) in spec.encode().into_iter().enumerate() {
                let star = (k == 2 && spec.stars & 1 != 0) || (k == 3 && spec.stars & 2 != 0);
                kids.push(if star { it.next().expect("counted") } else { num(prog, span, v) });
            }
            return Ok((call(prog, "viv_s_fmt_str", span, kids), s));
        }
        let f = crate::printfmt::Fmt::of_args(args, span).map_err(|(at, m, label)| Diagnostic::error(m).with_primary(at, label))?;
        let (wid, pid) = star_nodes(prog, f, stars, span);
        // an Int: its digits, or a float format of it
        if ty == Ty::Int {
            let (t, tt) = super::ints::text_or_num_at(prog, id, f, wid, span)?;
            if tt == s {
                return Ok((t, s));
            }
            return lower_num_text_at(prog, t, tt, f, &args[0], (wid, pid), span);
        }
        return lower_num_text_at(prog, id, ty, f, &args[0], (wid, pid), span);
    }
    Ok(match ty {
        Ty::Str if is_repr || is_fmt => (call(prog, "viv_s_repr", span, vec![id]), s),
        Ty::Str => (id, s),
        Ty::Arr(Elem::Str) => (call(prog, "viv_ss_repr", span, vec![id]), s),
        Ty::Bool => (call(prog, "viv_s_str_bool", span, vec![id]), s),
        Ty::Cplx => (call(prog, "viv_s_str_cplx", span, vec![id]), s),
        Ty::Int => super::ints::text_or_num(prog, id, crate::printfmt::Fmt::BARE, span)?,
        Ty::Arr(Elem::Int) => (call(prog, "viv_iarr_str", span, vec![id]), s),
        Ty::Enum(_) => (super::enums::text(prog, id, ty, span), s),
        // `str(x)` writes a whole number as its digits, `repr(x)` and `fmt(x)`
        // as Python's repr
        Ty::Num | Ty::Arr(Elem::Num) => {
            let f = if is_repr || is_fmt { crate::printfmt::Fmt::REPR } else { crate::printfmt::Fmt::STR };
            return lower_num_text(prog, id, ty, f, &args[0], span);
        }
        t => {
            return Err(Diagnostic::error(format!("`{shown}` cannot write {}", a_or_an(t)))
                .with_primary(args[0].span(), format!("this is {}", a_or_an(t))))
        }
    })
}

/// The width and precision nodes of a format: the values of its `*`
/// fields, in order, or its numbers.
fn star_nodes(prog: &mut IrProgram<Viv>, f: crate::printfmt::Fmt, stars: Vec<NodeId>, span: Span) -> (NodeId, NodeId) {
    let mut it = stars.into_iter();
    let w = if f.flags & crate::printfmt::FL_STAR_WIDTH != 0 { it.next() } else { None };
    let p = if f.flags & crate::printfmt::FL_STAR_PREC != 0 { it.next() } else { None };
    (w.unwrap_or_else(|| num(prog, span, f.width as f64)), p.unwrap_or_else(|| num(prog, span, f.prec as f64)))
}

/// A number or a real array in a format, as a string.
fn lower_num_text(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, f: crate::printfmt::Fmt, x: &Expr, span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    let wp = (num(prog, span, f.width as f64), num(prog, span, f.prec as f64));
    lower_num_text_at(prog, id, ty, f, x, wp, span)
}

/// `lower_num_text` with the width and precision as nodes.
fn lower_num_text_at(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, f: crate::printfmt::Fmt, x: &Expr, (wid, pid): (NodeId, NodeId), span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    let name = match ty {
        Ty::Num => "viv_s_fmt_num",
        Ty::Arr(Elem::Num) => "viv_s_fmt_arr",
        t => {
            return Err(Diagnostic::error("`fmt` takes a real number or a real array").with_primary(x.span(), format!("this is {}", a_or_an(t))));
        }
    };
    let kids = vec![id, num(prog, span, f.word() as f64), pid, wid];
    Ok((call(prog, name, span, kids), Ty::Str))
}

/// The calls strings lower: `Some` for a call that is theirs.
pub(super) fn lower_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    if sigs.arity.contains_key(callee) {
        return None;
    }
    Some(match callee {
        rules::FSTR => (|| {
            let mut ids = Vec::with_capacity(args.len());
            for a in args {
                let (id, t) = lower_expr(prog, a, scope, sigs)?;
                if t != Ty::Str {
                    return Err(Diagnostic::error("an f-string's part is not text").with_primary(a.span(), format!("this is {}", a_or_an(t))));
                }
                ids.push(id);
            }
            Ok((call(prog, "viv_s_catn", span, ids), Ty::Str))
        })(),
        rules::IN => (|| {
            // a number in a real array is `any(a == x)`, as NumPy's `x in a`;
            // a key in a map is `has(m, k)`
            match peek(&args[1], scope, sigs) {
                Some(Ty::Arr(Elem::Num)) => {
                    let (cid, _) = lower_expr(prog, &args[1], scope, sigs)?;
                    let (xid, xty) = lower_expr(prog, &args[0], scope, sigs)?;
                    if xty != Ty::Num {
                        return Err(Diagnostic::error("`in` looks for a number in an array")
                            .with_primary(args[0].span(), format!("this is {}", a_or_an(xty))));
                    }
                    let (eq, _) = super::lower_binop(prog, BinOp::EqEq, (cid, Ty::Arr(Elem::Num), args[1].span()), (xid, Ty::Num, args[0].span()), span)?;
                    return Ok((call(prog, "viv_any", span, vec![eq]), Ty::Bool));
                }
                Some(Ty::Map(..)) => {
                    return super::lower_map_call(prog, "has", &[args[1].clone(), args[0].clone()], span, scope, sigs);
                }
                _ => {}
            }
            let (xid, xty) = lower_expr(prog, &args[0], scope, sigs)?;
            let (cid, cty) = lower_expr(prog, &args[1], scope, sigs)?;
            match (xty, cty) {
                (Ty::Str, Ty::Str) => Ok((call(prog, "viv_s_contains", span, vec![cid, xid]), Ty::Bool)),
                (Ty::Str, Ty::Arr(Elem::Str)) => Ok((call(prog, "viv_ss_contains", span, vec![cid, xid]), Ty::Bool)),
                (x, Ty::Str | Ty::Arr(Elem::Str)) => Err(Diagnostic::error(format!("`in` looks for a string in {}", a_or_an(cty)))
                    .with_primary(args[0].span(), format!("this is {}", a_or_an(x)))),
                (_, c) => Err(Diagnostic::error("`in` takes a string, a list of strings, a real array or a map on its right")
                    .with_primary(args[1].span(), format!("this is {}", a_or_an(c)))),
            }
        })(),
        "str" | "repr" | "fmt" | rules::STR_OF | rules::REPR_OF | rules::FMT_OF => lower_text_of(prog, callee, args, span, scope, sigs),
        "stop" if args.len() == 1 => (|| {
            let (id, t) = lower_expr(prog, &args[0], scope, sigs)?;
            if t != Ty::Str {
                return Err(Diagnostic::error("`stop` takes one string").with_primary(args[0].span(), format!("this is {}", a_or_an(t))));
            }
            Ok((call(prog, "viv_s_stop", span, vec![id]), Ty::Num))
        })(),
        c if c == SLICE_GET || c == SLICE_SET => {
            let t = peek(&args[0], scope, sigs)?;
            if !is_text(t) {
                return None;
            }
            if c == SLICE_SET {
                let m = if t == Ty::Str { rules::STR_IMMUTABLE } else { rules::STRS_IMMUTABLE };
                return Some(Err(Diagnostic::error(m.to_string()).with_primary(span, "a string or a list of strings is not changed in place")));
            }
            (|| {
                let (bid, _) = lower_expr(prog, &args[0], scope, sigs)?;
                let Expr::ArrLit { elems, .. } = &args[1] else { unreachable!("the parser writes a slice's spec as a literal") };
                if elems.len() != 4 {
                    return Err(Diagnostic::error(format!("{} takes one index or range; this has {}", a_or_an(t), elems.len() / 4)).with_primary(span, "one range"));
                }
                let mut kids = vec![bid];
                for e in elems {
                    let (id, ty) = lower_expr(prog, e, scope, sigs)?;
                    // an Int bound converts, as an Int index does (B342)
                    let (id, ty) = super::ints::to_num(prog, id, ty, e.span());
                    expect_num(ty, e.span(), "a slice bound")?;
                    kids.push(id);
                }
                let f = if t == Ty::Str { "viv_s_slice" } else { "viv_ss_slice" };
                Ok((call(prog, f, span, kids), t))
            })()
        }
        c if c == crate::growable::SEAL => {
            let t = peek(&args[0], scope, sigs)?;
            if t != Ty::Arr(Elem::Str) {
                return None;
            }
            lower_expr(prog, &args[0], scope, sigs).map(|(id, t)| (call(prog, "viv_ss_seal", span, vec![id]), t))
        }
        "append" | "extend" if args.len() >= 2 => {
            let t0 = peek(&args[0], scope, sigs);
            let t1 = peek(&args[1], scope, sigs);
            let list = Some(Ty::Arr(Elem::Str));
            let empty = matches!(&args[0], Expr::ArrLit { elems, .. } if elems.is_empty());
            let ours = t0 == list || ((empty || t0 == Some(Ty::Arr(Elem::Num))) && (t1 == Some(Ty::Str) || t1 == list) && scope_holds_nothing(&args[0], scope));
            if !ours {
                return None;
            }
            (|| {
                let (aid, _) = lower_expected(prog, &args[0], list, scope, sigs)?;
                let (xid, xty) = lower_expected(prog, &args[1], if callee == "extend" { list } else { None }, scope, sigs)?;
                let want = if callee == "append" { Ty::Str } else { Ty::Arr(Elem::Str) };
                if xty != want {
                    let what = if callee == "append" { "`append` adds a string to a list of strings" } else { "`extend` joins two lists of strings" };
                    return Err(Diagnostic::error(what).with_primary(args[1].span(), format!("this is {}", a_or_an(xty))));
                }
                let own = match args.get(2) {
                    Some(Expr::Num { value, .. }) => *value,
                    _ => 0.0,
                };
                let oid = num(prog, span, own);
                let f = if callee == "append" { "viv_ss_append" } else { "viv_ss_extend" };
                Ok((call(prog, f, span, vec![aid, xid, oid]), Ty::Arr(Elem::Str)))
            })()
        }
        c if rules::is_builtin(c) => lower_registry(prog, c, args, span, scope, sigs),
        _ => return None,
    })
}

/// `a = append(a, s)` where `a` is `[]`: a variable whose type the program has
/// not fixed as an array of numbers (it may be a new list of strings).
fn scope_holds_nothing(e: &Expr, scope: &Scope<'_>) -> bool {
    match e {
        Expr::ArrLit { elems, .. } => elems.is_empty(),
        Expr::Var { name, .. } => scope.get(name) != Some(Ty::Arr(Elem::Num)),
        _ => false,
    }
}

/// A call of a registry builtin, its defaults filled in.
fn lower_registry(prog: &mut IrProgram<Viv>, name: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    // an empty `[]` stands for an empty list where every entry wants one
    // (or, where the entries take a string or a list, an empty `[]`, which
    // cannot be a string: `sorted([])`, `unique([])`)
    let lists: Vec<bool> = (0..args.len())
        .map(|j| {
            let ks: Vec<K> = rules::BUILTINS.iter().filter(|b| b.name == name && j < b.args.len()).map(|b| b.args[j]).collect();
            let empty = matches!(&args[j], Expr::ArrLit { elems, .. } if elems.is_empty());
            !ks.is_empty() && (ks.iter().all(|k| *k == K::L) || (empty && ks.contains(&K::L) && ks.iter().all(|k| matches!(k, K::L | K::S))))
        })
        .collect();
    let mut ids = Vec::with_capacity(args.len());
    let mut tys = Vec::with_capacity(args.len());
    for (a, &l) in args.iter().zip(&lists) {
        let (id, t) = lower_expected(prog, a, if l { Some(Ty::Arr(Elem::Str)) } else { None }, scope, sigs)?;
        ids.push(id);
        tys.push(t);
    }
    let kinds: Vec<Option<K>> = tys.iter().map(|t| kind_of(*t).or_else(|| db_kind(name, *t))).collect();
    let found = if kinds.iter().any(|k| k.is_none()) { None } else { rules::resolve(name, &kinds) };
    let Some(b) = found else {
        let d = Diagnostic::error(rules::usage(name));
        let bad = tys.iter().zip(args).find(|(t, _)| kind_of(**t).or_else(|| db_kind(name, **t)).is_none());
        return Err(match bad {
            Some((t, a)) => d.with_primary(a.span(), format!("this is {}", a_or_an(*t))),
            None => d.with_primary(span, "these arguments"),
        });
    };
    // `float(x)` of a number is x
    if b.c == "viv_num_float" {
        return Ok((ids[0], Ty::Num));
    }
    let missing = b.args.len() - args.len();
    for d in &b.opt[b.opt.len() - missing..] {
        ids.push(match d {
            D::N(x) => num(prog, span, *x),
            D::S(s) => lit(prog, span, s),
        });
    }
    Ok((call(prog, b.c, span, ids), ty_of(b.ret)))
}

/// `show s`: `s: 'text'`, the repr, as a print line.
pub(super) fn show(prog: &mut IrProgram<Viv>, label: &str, id: NodeId, ty: Ty, span: Span) -> NodeId {
    let f = if ty == Ty::Str { "viv_s_repr" } else { "viv_ss_repr" };
    let r = call(prog, f, span, vec![id]);
    let mut parts = Vec::new();
    if !label.is_empty() {
        parts.push(PrintPart::Text(format!("{label}: ")));
    }
    parts.push(PrintPart::Str);
    let p = prog.add(Viv::Print { parts }, span, vec![r]);
    prog.add(Viv::ExprStmt, span, vec![p])
}

/// A `print` argument that is a string or a list of strings, or `fmt` of
/// one: the node that gives its text.
pub(super) fn print_part(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, span: Span) -> NodeId {
    if ty == Ty::Str {
        id
    } else {
        call(prog, "viv_ss_repr", span, vec![id])
    }
}

/// Whether evaluating under `id` makes a string or a list of strings.
pub(super) fn makes_strings(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = prog.get(id);
    if let Viv::Call { callee } = &n.kind {
        if allocates(callee) {
            return true;
        }
    }
    n.children.iter().any(|&c| makes_strings(prog, c))
}

/// Whether a runtime call makes a string or a list (from the one table of
/// what runtime calls give, `crate::builtins`).
pub(super) fn allocates(callee: &str) -> bool {
    matches!(crate::builtins::runtime_gives(callee), crate::builtins::Gives::Str | crate::builtins::Gives::Arr(Elem::Str))
}

// ---- the C runtime ----------------------------------------------------------------------------

/// Every runtime call strings make, and what its C function returns and
/// takes: `ret` and `args` as registry kinds, plus (for the wasm backend) the
/// import's name. The registry's builtins are listed from `rules::BUILTINS`.
pub(crate) struct RtFn {
    pub c: &'static str,
    pub args: &'static [K],
    pub ret: K,
    /// the C function stops with a located error: its viv_sites row goes last
    pub stops: bool,
}

macro_rules! rt {
    ($c:expr, [$($a:expr),*], $ret:expr, $stops:expr) => {
        RtFn { c: $c, args: &[$($a),*], ret: $ret, stops: $stops }
    };
}

use K::{B as KB, L as KL, N as KN, S as KS};

/// The runtime calls that are not registry builtins.
pub(crate) const RUNTIME: &[RtFn] = &[
    rt!("viv_s_cat", [KS, KS], KS, true),
    rt!("viv_s_rep", [KS, KN], KS, true),
    rt!("viv_s_eq", [KS, KS], KN, false),
    rt!("viv_s_cmp", [KS, KS], KN, false),
    rt!("viv_s_len", [KS], KN, false),
    rt!("viv_s_at", [KS, KN], KS, true),
    rt!("viv_s_slice", [KS, KN, KN, KN, KN], KS, true),
    rt!("viv_ss_len", [KL], KN, false),
    rt!("viv_ss_at", [KL, KN], KS, true),
    rt!("viv_ss_slice", [KL, KN, KN, KN, KN], KL, true),
    rt!("viv_ss_cat", [KL, KL], KL, false),
    rt!("viv_ss_eq", [KL, KL], KN, false),
    rt!("viv_ss_new", [], KL, false),
    rt!("viv_ss_append", [KL, KS, KN], KL, false),
    rt!("viv_ss_extend", [KL, KL, KN], KL, false),
    rt!("viv_ss_seal", [KL], KL, false),
    rt!("viv_s_str_bool", [KB], KS, false),
    rt!("viv_s_fmt_num", [KN, KN, KN, KN], KS, false),
    rt!("viv_s_fmt_str", [KS, KN, KN, KN, KN], KS, true),
    rt!("viv_s_stop", [KS], KN, true),
];

/// A call's runtime entry: the registry's, or `RUNTIME`'s. (`viv_s_catn`,
/// `viv_ss_of`, `viv_s_str_cplx` and `viv_s_fmt_arr` take other arguments and
/// are handled on their own.)
pub(crate) fn rt_fn(callee: &str) -> Option<RtFn> {
    if let Some(b) = rules::BUILTINS.iter().find(|b| b.c == callee) {
        return Some(RtFn { c: b.c, args: b.args, ret: b.ret, stops: true });
    }
    RUNTIME.iter().find(|r| r.c == callee).map(|r| RtFn { c: r.c, args: r.args, ret: r.ret, stops: r.stops })
}

/// The C type of a registry kind.
fn c_ty(k: K) -> &'static str {
    match k {
        K::N | K::B => "double",
        K::S => "viv_str",
        K::L => "viv_strs",
        K::A | K::I => "viv_arr",
    }
}

/// The C runtime of strings: `interp/strings.c`, cut into fragments at its
/// `//@ name: deps` lines, and the character tables printed from
/// `unicode_tables`.
pub(super) fn fragments() -> Vec<RuntimeFragment> {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<Vec<(&'static str, &'static [&'static str], &'static str)>>> = const { std::cell::RefCell::new(None) };
    }
    let parts = CACHE.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| {
                let mut out = Vec::new();
                let src = concat!(include_str!("../interp/strings.c"), include_str!("../interp/db.c"));
                let mut name: Option<&'static str> = None;
                let mut deps: &'static [&'static str] = &[];
                let mut code = String::new();
                let flush = |name: Option<&'static str>, deps: &'static [&'static str], code: &mut String, out: &mut Vec<_>| {
                    if let Some(n) = name {
                        out.push((n, deps, &*Box::leak(std::mem::take(code).into_boxed_str())));
                    }
                };
                for line in src.lines() {
                    if let Some(head) = line.strip_prefix("//@ ") {
                        flush(name, deps, &mut code, &mut out);
                        let (n, d) = head.split_once(':').unwrap_or((head, ""));
                        name = Some(Box::leak(n.trim().to_string().into_boxed_str()));
                        let ds: Vec<&'static str> = d.split_whitespace().map(|x| &*Box::leak(x.to_string().into_boxed_str())).collect();
                        deps = Box::leak(ds.into_boxed_slice());
                    } else if name.is_some() {
                        code.push_str(line);
                        code.push('\n');
                    }
                }
                flush(name, deps, &mut code, &mut out);
                for (t, n) in TABLES {
                    out.push((t, &[], Box::leak(n().into_boxed_str())));
                }
                out
            })
            .clone()
    });
    let mut frags: Vec<RuntimeFragment> = parts
        .into_iter()
        .map(|(name, deps, code)| RuntimeFragment {
            name,
            includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <stdarg.h>", "#include <math.h>", "#include <float.h>", "#include <limits.h>", "#include <errno.h>", "#include <unistd.h>"],
            deps,
            code,
        })
        .collect();
    frags.push(lits_fragment());
    frags
}

/// The character tables, by fragment name.
const TABLES: [(&str, fn() -> String); 12] = [
    ("str_t_space", || ranges("viv_t_space", rules_tables::SPACE)),
    ("str_t_alpha", || ranges("viv_t_alpha", rules_tables::ALPHA)),
    ("str_t_digit", || ranges("viv_t_digit", rules_tables::DIGIT)),
    ("str_t_numeric", || ranges("viv_t_numeric", rules_tables::NUMERIC)),
    ("str_t_upper", || ranges("viv_t_upper", rules_tables::UPPER)),
    ("str_t_lower", || ranges("viv_t_lower", rules_tables::LOWER)),
    ("str_t_title", || ranges("viv_t_title", rules_tables::TITLE)),
    ("str_t_printable", || ranges("viv_t_printable", rules_tables::PRINTABLE)),
    ("str_t_ignorable", || ranges("viv_t_ignorable", rules_tables::CASE_IGNORABLE)),
    ("str_t_cased", || ranges("viv_t_cased", rules_tables::CASED)),
    ("str_t_decimal", decimal_table),
    ("str_t_case", case_tables),
];

use crate::interp::unicode_tables as rules_tables;

fn ranges(name: &str, t: &[(u32, u32)]) -> String {
    let mut s = format!("static const unsigned {name}[{}][2] = {{", t.len());
    for (k, (a, b)) in t.iter().enumerate() {
        if k % 8 == 0 {
            s.push_str("\n   ");
        }
        s.push_str(&format!(" {{0x{a:X}, 0x{b:X}}},"));
    }
    s.push_str(&format!("\n}};\n#define {}_N {}\n", name.to_uppercase(), t.len()));
    s
}

fn decimal_table() -> String {
    let t = rules_tables::DECIMAL;
    let mut s = format!("static const unsigned viv_t_decimal[{}][3] = {{", t.len());
    for (k, (a, b, z)) in t.iter().enumerate() {
        if k % 6 == 0 {
            s.push_str("\n   ");
        }
        s.push_str(&format!(" {{0x{a:X}, 0x{b:X}, 0x{z:X}}},"));
    }
    s.push_str(&format!("\n}};\n#define VIV_T_DECIMAL_N {}\n", t.len()));
    s
}

fn case_tables() -> String {
    let mut s = String::new();
    for (name, t, pool) in [
        ("viv_t_toupper", rules_tables::TO_UPPER, rules_tables::TO_UPPER_POOL),
        ("viv_t_tolower", rules_tables::TO_LOWER, rules_tables::TO_LOWER_POOL),
    ] {
        s.push_str(&format!("static const unsigned {name}[{}][3] = {{", t.len()));
        for (k, (a, b, n)) in t.iter().enumerate() {
            if k % 6 == 0 {
                s.push_str("\n   ");
            }
            s.push_str(&format!(" {{0x{a:X}, 0x{b:X}, {n}}},"));
        }
        s.push_str(&format!("\n}};\n#define {}_N {}\n", name.to_uppercase(), t.len()));
        s.push_str(&format!("static const unsigned {name}_pool[{}] = {{", pool.len().max(1)));
        for (k, p) in pool.iter().enumerate() {
            if k % 10 == 0 {
                s.push_str("\n   ");
            }
            s.push_str(&format!(" 0x{p:X},"));
        }
        if pool.is_empty() {
            s.push_str(" 0");
        }
        s.push_str("\n};\n");
    }
    s
}

// ---- string literals in C ----------------------------------------------------------------

thread_local! {
    /// The string literals of the program being emitted, by first use: each
    /// is made once, when the run first reaches it (`VIV_LIT`), and lives for
    /// the run.
    static LITS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The other translation units' literals in a split emission, and which
    /// unit `LITS` is (`lits_enter`): each unit numbers its own.
    static LITS_PARKED: std::cell::RefCell<(Option<String>, std::collections::HashMap<Option<String>, Vec<String>>)> =
        std::cell::RefCell::new((None, std::collections::HashMap::new()));
}

/// Make `unit` the translation unit whose literals are numbered and rendered.
pub(crate) fn lits_enter(unit: Option<&str>) {
    let want = unit.map(str::to_string);
    LITS_PARKED.with(|p| {
        let mut p = p.borrow_mut();
        if p.0 == want {
            return;
        }
        let cur = LITS.with(|l| std::mem::take(&mut *l.borrow_mut()));
        let old = std::mem::replace(&mut p.0, want.clone());
        if !cur.is_empty() {
            p.1.insert(old, cur);
        }
        let next = p.1.remove(&want).unwrap_or_default();
        LITS.with(|l| *l.borrow_mut() = next);
    });
}

/// A literal's C expression.
pub(super) fn c_lit(id: NodeId, s: &str) -> CExpr {
    let k = LITS.with(|l| {
        let mut l = l.borrow_mut();
        match l.iter().position(|x| x == s) {
            Some(k) => k,
            None => {
                l.push(s.to_string());
                l.len() - 1
            }
        }
    });
    let nch = rules::count_chars(s.as_bytes());
    CExpr::raw(id, format!("VIV_LIT({k}, {}, {}, {nch})", c_str(s), s.len()))
}

/// The table the literals live in, sized when the runtime is rendered (after
/// every item is emitted); the list is cleared for the next program.
fn lits_fragment() -> RuntimeFragment {
    let n = LITS.with(|l| std::mem::take(&mut *l.borrow_mut()).len());
    let code = format!("static viv_str viv_lits[{}];", n.max(1));
    RuntimeFragment { name: "str_lits", includes: &[], deps: &["str_core"], code: Box::leak(code.into_boxed_str()) }
}

/// The fragment a runtime call's C function is in.
pub(super) fn fragment_of(callee: &str) -> &'static str {
    match callee {
        c if c.starts_with("viv_s_db_") => "str_db",
        "viv_s_upper" | "viv_s_lower" => "str_case",
        c if c.starts_with("viv_s_is") => "str_pred",
        c if c.contains("strip") || c.contains("split") => "str_ws",
        "viv_s_find" | "viv_s_rfind" | "viv_s_index" | "viv_s_rindex" | "viv_s_count" | "viv_s_replace" | "viv_s_startswith" | "viv_s_endswith"
        | "viv_s_contains" => "str_search",
        "viv_s_join" => "str_ws",
        "viv_ss_index" => "str_repr",
        "viv_s_ljust" | "viv_s_rjust" | "viv_s_center" | "viv_s_zfill" | "viv_s_fmt_str" => "str_pad",
        "viv_s_float" | "viv_s_int" | "viv_num_int" | "viv_s_chr" | "viv_s_ord" => "str_num",
        "viv_s_repr" | "viv_ss_repr" => "str_repr",
        "viv_s_str_bool" | "viv_s_str_cplx" | "viv_s_fmt_num" | "viv_s_fmt_arr" => "str_fmt",
        "viv_s_read_text" | "viv_s_read_lines" | "viv_s_read_bytes" | "viv_s_write_text" => "str_file",
        c if c.starts_with("viv_ss_") => "str_list",
        _ => "str_core",
    }
}

/// The C of a string runtime call, or None when `callee` is not one.
pub(super) fn emit_call(node: &Node<Viv>, callee: &str, cx: &mut CEmitCx<Viv>) -> Option<CExpr> {
    if let Some(e) = emit_text_call(node, callee, cx) {
        return Some(e);
    }
    let id = node.id;
    let is_ours = callee.starts_with("viv_s_") || callee.starts_with("viv_ss_") || callee == "viv_num_int";
    if !is_ours {
        return None;
    }
    cx.need(fragment_of(callee));
    cx.need("str_lits");
    let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
    match callee {
        // the variadic ones: their count first
        "viv_s_catn" | "viv_ss_of" => {
            args.insert(0, CExpr::lit(id, node.children.len().to_string()));
            if callee == "viv_s_catn" {
                args.push(CExpr::raw(id, site_row(node.span).to_string()));
            }
        }
        "viv_s_str_cplx" | "viv_s_fmt_arr" => {}
        c => {
            if rt_fn(c).is_some_and(|r| r.stops) {
                args.push(CExpr::raw(id, site_row(node.span).to_string()));
            }
        }
    }
    Some(CExpr::call(id, callee.to_string(), args))
}

/// The prototype of a runtime call, as its C twin declares it (checked by a
/// test against `strings.c`).
pub(crate) fn c_prototype(r: &RtFn) -> String {
    let mut ps: Vec<String> = r.args.iter().enumerate().map(|(i, k)| format!("{} a{i}", c_ty(*k))).collect();
    if r.stops {
        ps.push("int site".into());
    }
    if ps.is_empty() {
        ps.push("void".into());
    }
    format!("{} {}({})", c_ty(r.ret), r.c, ps.join(", "))
}

/// Whether `e` is a string or a list of strings (and which), found without
/// lowering it into the program.
pub(super) fn peek_text(e: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Option<Ty> {
    match e {
        Expr::Str { .. } => Some(Ty::Str),
        Expr::ArrLit { elems, .. } => match elems.first() {
            Some(first) if peek_text(first, scope, sigs) == Some(Ty::Str) => Some(Ty::Arr(Elem::Str)),
            _ => None,
        },
        Expr::Num { .. } | Expr::Bool { .. } | Expr::Imag { .. } | Expr::Lambda { .. } => None,
        _ => peek(e, scope, sigs).filter(|t| is_text(*t)),
    }
}

/// `figure(title, xlabel, ylabel[, xlo, xhi, ylo, yhi[, style]])` with a
/// title or a label that is a string value: `viv_figure_s` or
/// `viv_figure_win_s`, the style still a literal.
pub(super) fn lower_figure(prog: &mut IrProgram<Viv>, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let mut ids = Vec::with_capacity(args.len());
    for a in &args[..3] {
        let (id, t) = lower_expr(prog, a, scope, sigs)?;
        if t != Ty::Str {
            return Err(Diagnostic::error("`figure`'s title and labels are strings").with_primary(a.span(), format!("this is {}", a_or_an(t))));
        }
        ids.push(id);
    }
    for a in args.iter().skip(3).take(4) {
        let (id, t) = lower_expr(prog, a, scope, sigs)?;
        if t != Ty::Num {
            return Err(Diagnostic::error("`figure`'s window arguments must be numbers: xlo, xhi, ylo, yhi").with_primary(a.span(), "not a number"));
        }
        ids.push(id);
    }
    let style = match args.get(7) {
        Some(Expr::Str { value, .. }) => value.clone(),
        Some(other) => {
            return Err(Diagnostic::error("`figure`'s eighth argument must be a style string literal (e.g. \"contour\")").with_primary(other.span(), "not a string literal"))
        }
        None => String::new(),
    };
    if ids.len() == 3 {
        return Ok((call(prog, "viv_figure_s", span, ids), Ty::Num));
    }
    ids.push(lit(prog, span, &style));
    Ok((call(prog, "viv_figure_win_s", span, ids), Ty::Num))
}

/// An animation verb with text that is a string value: `viv_anim_s`, its verb
/// first. A verb whose text is one of its keywords (`xscale("log")`, a
/// `style`) keeps a literal, which the compiler checks; other text is checked
/// when it is known, as a literal's is when it is compiled.
pub(super) fn lower_anim(prog: &mut IrProgram<Viv>, v: &AnimVerb, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    if matches!(v.name, "xscale" | "yscale" | "style" | "share" | "grid" | "aspect" | "origin" | "colormap") {
        let a = args[..v.texts].iter().find(|a| !matches!(a, Expr::Str { .. })).expect("a text that is not a literal");
        return Err(Diagnostic::error(format!("`{}` is written {}", v.name, v.usage)).with_primary(a.span(), "this must be a string literal"));
    }
    let mut ids = vec![lit(prog, span, v.name)];
    for a in &args[..v.texts] {
        let (id, t) = lower_expr(prog, a, scope, sigs)?;
        if t != Ty::Str {
            return Err(Diagnostic::error(format!("`{}` is written {}", v.name, v.usage)).with_primary(a.span(), format!("this must be a string, and is {}", a_or_an(t))));
        }
        ids.push(id);
    }
    for (k, a) in args[v.texts..].iter().enumerate() {
        let (id, t) = lower_expr(prog, a, scope, sigs)?;
        if k < v.arrays {
            if t != Ty::Arr(Elem::Num) {
                return Err(Diagnostic::error(format!("`{}` is written {}", v.name, v.usage)).with_primary(a.span(), format!("this must be a real array, and is {}", a_or_an(t))));
            }
        } else {
            expect_num(t, a.span(), &format!("`{}`'s argument", v.name))?;
        }
        ids.push(id);
    }
    Ok((call(prog, "viv_anim_s", span, ids), Ty::Num))
}

/// The C of `viv_figure_s`, `viv_figure_win_s` and `viv_anim_s`.
pub(super) fn emit_text_call(node: &Node<Viv>, callee: &str, cx: &mut CEmitCx<Viv>) -> Option<CExpr> {
    let id = node.id;
    match callee {
        "viv_figure_s" | "viv_figure_win_s" => {
            cx.need("str_figure");
            cx.need("str_lits");
            let args = node.children.iter().map(|&c| cx.expr(c)).collect();
            Some(CExpr::call(id, callee.to_string(), args))
        }
        "viv_anim_s" => {
            cx.need("str_figure");
            cx.need("str_lits");
            let kids = &node.children;
            let Viv::StrLit(verb) = cx.kind(kids[0]) else { unreachable!("the verb first") };
            let v = anim_verb(&verb).expect("an animation verb");
            let mut args = vec![CExpr::raw(id, site_row(node.span).to_string()), CExpr::raw(id, c_str(&verb)), CExpr::lit(id, v.texts.to_string())];
            // the C takes three text slots: `note` has the most, an id, its text and a style
            assert!(v.texts <= 3, "`{verb}` has more texts than viv_anim_s takes");
            for k in 0..3 {
                args.push(if k < v.texts { cx.expr(kids[1 + k]) } else { CExpr::raw(id, "(viv_str)0") });
            }
            args.push(CExpr::lit(id, if anim_data_verb(&verb) { "1" } else { "0" }));
            args.push(CExpr::lit(id, v.arrays.to_string()));
            for k in 0..3 {
                args.push(if k < v.arrays { cx.expr(kids[1 + v.texts + k]) } else { CExpr::raw(id, "(viv_arr){0}") });
            }
            let nn = kids.len() - 1 - v.texts - v.arrays;
            args.push(CExpr::lit(id, nn.to_string()));
            for k in 0..6 {
                args.push(if k < nn { cx.expr(kids[1 + v.texts + v.arrays + k]) } else { CExpr::lit(id, "0.0") });
            }
            Some(CExpr::call(id, "viv_anim_s", args))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C twin declares every runtime call with the types the emitted
    /// calls pass: the kinds in order, then `int site` for one that stops.
    #[test]
    fn c_functions_take_what_the_calls_pass() {
        let src = concat!(include_str!("../interp/strings.c"), include_str!("../interp/db.c"));
        let mut bad = Vec::new();
        for f in crate::lower::string_runtime() {
            let head = format!(" {}(", f.c);
            let Some(at) = src.find(&head) else {
                bad.push(format!("{}: not in strings.c", f.c));
                continue;
            };
            let line_start = src[..at].rfind('\n').map_or(0, |k| k + 1);
            let decl = &src[line_start..];
            let open = decl.find('(').expect("a parameter list");
            let close = decl.find(')').expect("a parameter list");
            let ret = decl[..open].split_whitespace().rev().nth(1).unwrap_or("");
            let params: Vec<String> = decl[open + 1..close]
                .split(',')
                .map(|p| p.trim().rsplit_once(' ').map_or(p.trim().to_string(), |(t, _)| t.trim().to_string()))
                .filter(|t| t != "void" && !t.is_empty())
                .collect();
            let mut want: Vec<String> = f.args.iter().map(|k| c_ty(*k).to_string()).collect();
            if f.stops {
                want.push("int".into());
            }
            if ret != c_ty(f.ret) || params != want {
                bad.push(format!("{}: C has {ret} ({}) but the calls pass {} ({})", f.c, params.join(", "), c_ty(f.ret), want.join(", ")));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
