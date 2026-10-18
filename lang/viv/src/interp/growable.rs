// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Growable arrays: `[]`, `a = append(a, x)` and `a = extend(a, b)`
//!.
//!
//! `append(a, x)` is the array `a` with `x` on the end: a number onto a
//! vector, or a vector as a new last row of a matrix (an empty `[]` takes a
//! first row and becomes a 1 x n matrix). `extend(a, b)` is vector `a`
//! followed by vector `b`. Both are values: the result is a new array, and no
//! alias of `a` ever sees it change. They are written `a = append(a, x)`,
//! rebinding the name they read (`crate::growable` refuses any other form),
//! which is what lets an engine grow `a`'s storage in place when nothing else
//! holds it, with room doubling, so n appends cost O(n) and not O(n^2).
//!
//! This file is the one statement of the rules: the interpreter's arms and the
//! wasm runtime call `grown` here, and `C_RUNTIME` below is their C twin,
//! check for check and word for word. Each engine decides that nothing else
//! holds the storage before it grows in place.

use super::{ArrData, Val};
use std::cell::RefCell;
use std::rc::Rc;

/// What is added to the array.
#[derive(Clone, Copy)]
pub(crate) enum Tail<'a> {
    /// a number
    Num,
    /// an array, of this shape
    Arr(&'a [usize]),
}

/// The shape `a` has once `callee` ("append" or "extend") adds `tail` to it,
/// or why it cannot.
pub(crate) fn grown(callee: &str, a: &[usize], tail: Tail) -> Result<Vec<usize>, String> {
    if callee == "extend" {
        if a.len() != 1 {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`extend` joins vectors; this array has rank {}", a.len())));
        }
        return match tail {
            Tail::Num => Err("`extend` joins two vectors; add one number with `append`".into()),
            Tail::Arr(b) if b.len() != 1 => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`extend` joins vectors; this array has rank {}", b.len()))),
            Tail::Arr(b) => Ok(vec![a[0] + b[0]]),
        };
    }
    match tail {
        Tail::Num => {
            if a.len() != 1 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`append`: a number goes on the end of a vector, and this array has rank {}", a.len())));
            }
            Ok(vec![a[0] + 1])
        }
        Tail::Arr(r) => {
            if r.len() != 1 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`append`: a row must be a vector, and this array has rank {}", r.len())));
            }
            match a.len() {
                1 if a[0] == 0 => Ok(vec![1, r[0]]),
                1 => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!(
                    "`append`: an array goes on as a new row of a matrix, and this is a vector of {} numbers (join two vectors with `extend`)",
                    a[0]
                ))),
                2 if a[1] != r[0] => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`append`: the row has {} numbers, and the matrix has {} columns", r[0], a[1]))),
                2 => Ok(vec![a[0] + 1, a[1]]),
                k => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`append`: an array goes on as a new row of a matrix, and this array has rank {k}"))),
            }
        }
    }
}

/// The Int an Int array's append adds (the lowering refused anything else).
pub(super) fn int_tail(x: &Val) -> Result<i64, String> {
    match x {
        Val::Int(i) => Ok(*i),
        _ => Err("an Int array holds Ints".into()),
    }
}

/// The message for an argument that is not a real array or a number.
fn not_real(callee: &str, first: bool) -> String {
    if first {
        format!("`{callee}` takes a real array first")
    } else if callee == "extend" {
        "`extend` joins two real vectors".into()
    } else {
        "`append` adds a number, or a row (a real vector)".into()
    }
}

/// Grow `target` in place: `callee` of `x`. The interpreter calls this on an
/// array only it holds (`Rc::strong_count` is 1), and on a fresh copy
/// otherwise. `Vec` doubles its own capacity, so a run of these is O(n).
pub(super) fn grow(callee: &str, target: &mut ArrData, x: &Val) -> Result<(), String> {
    match x {
        Val::Num(v) => {
            target.shape = grown(callee, &target.shape, Tail::Num)?;
            target.data.push(*v);
        }
        // an Int on a real array: the number it is, exactly
        Val::Int(i) => {
            let v = super::ints::to_num(*i)?;
            target.shape = grown(callee, &target.shape, Tail::Num)?;
            target.data.push(v);
        }
        Val::Arr(b) => {
            // `a = extend(a, a)` reaches here with `b` a second handle on the
            // storage, so it is never the same RefCell as `target`
            let bb = b.borrow();
            target.shape = grown(callee, &target.shape, Tail::Arr(&bb.shape))?;
            target.data.extend_from_slice(&bb.data);
        }
        _ => return Err(not_real(callee, false)),
    }
    Ok(())
}

fn arr(shape: Vec<usize>, data: Vec<f64>) -> Val {
    Val::Arr(Rc::new(RefCell::new(ArrData { shape, data })))
}

/// The interpreter's arms: `append` and `extend` anywhere but the rebinding
/// statement (which `Interp` runs in place), and the adjoint helpers the
/// autodiff pass writes for them. `None`: not one of these builtins.
pub(super) fn builtin(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    Ok(Some(match (callee, a.len()) {
        ("append", 2) if matches!(a[0], Val::IArr(_)) => {
            let Val::IArr(t) = &a[0] else { unreachable!() };
            let mut copy = t.borrow().clone();
            copy.push(int_tail(&a[1])?);
            Val::IArr(Rc::new(RefCell::new(copy)))
        }
        // two Int arrays joined (B343)
        ("extend", 2) if matches!(a[0], Val::IArr(_)) => {
            let (Val::IArr(t), Val::IArr(u)) = (&a[0], &a[1]) else { return Err("`extend` of an Int array joins another Int array".into()) };
            let mut copy = t.borrow().clone();
            copy.extend_from_slice(&u.borrow());
            Val::IArr(Rc::new(RefCell::new(copy)))
        }
        ("append" | "extend", 2) => {
            let Val::Arr(t) = &a[0] else { return Err(not_real(callee, true)) };
            let mut copy = {
                let tb = t.borrow();
                ArrData { shape: tb.shape.clone(), data: tb.data.clone() }
            };
            grow(callee, &mut copy, &a[1])?;
            Val::Arr(Rc::new(RefCell::new(copy)))
        }
        // `__ad_head(g, a)`: the first numel(a) elements of the adjoint g of
        // `append(a, x)` or `extend(a, x)`, in a's shape -- a's share
        ("__ad_head", 2) => {
            let (Val::Arr(g), Val::Arr(like)) = (&a[0], &a[1]) else {
                return Err("autodiff: `grad` differentiates numbers and real arrays only".into());
            };
            let (gb, lb) = (g.borrow(), like.borrow());
            let n = lb.data.len();
            if n > gb.data.len() {
                return Err(ad_short(&gb.shape, &lb.shape));
            }
            arr(lb.shape.clone(), gb.data[..n].to_vec())
        }
        // `__ad_tail(g, a, x)`: the rest of g, in x's shape (a number for a number) -- x's share
        ("__ad_tail", 3) => {
            let (Val::Arr(g), Val::Arr(like)) = (&a[0], &a[1]) else {
                return Err("autodiff: `grad` differentiates numbers and real arrays only".into());
            };
            let (gb, lb) = (g.borrow(), like.borrow());
            let off = lb.data.len();
            match &a[2] {
                Val::Num(_) => {
                    if off >= gb.data.len() {
                        return Err(ad_short(&gb.shape, &[1]));
                    }
                    Val::Num(gb.data[off])
                }
                Val::Arr(x) => {
                    let xb = x.borrow();
                    let n = xb.data.len();
                    if off + n > gb.data.len() {
                        return Err(ad_short(&gb.shape, &xb.shape));
                    }
                    arr(xb.shape.clone(), gb.data[off..off + n].to_vec())
                }
                _ => return Err("autodiff: `grad` differentiates numbers and real arrays only".into()),
            }
        }
        _ => return Ok(None),
    }))
}

pub(crate) fn ad_short(g: &[usize], like: &[usize]) -> String {
    format!("autodiff: the adjoint of an append has shape {g:?}, too small for a piece of shape {like:?}")
}

/// The C runtime's twin (fragment "growable"; it stops through the slices
/// fragment's `viv_sl_fail`, so an error carries its location in a debug
/// build exactly as a slice's does).
///
/// The capacity of a growable array lives in its HANDLE, in `dims[7]`: the
/// runtime allocates rank 4 at most (`MAX_RANK`), so `dims[4..8]` are never a
/// dimension, and `viv_arr_new` sets `dims[7]` to 0. A handle whose `dims[7]`
/// is 0 is SEALED: its storage may be shared, so the next append copies. Only
/// `viv_append_*` and `viv_extend` return an unsealed handle, only into the
/// variable they rebind, and the compiler seals that variable before any use
/// that could copy the handle anywhere else (`crate::growable`).
pub const C_RUNTIME: &str = r#"/* Growable arrays: a = append(a, x), a = extend(a, b)
   (lang/viv/src/interp/growable.rs has the rules). dims[7] of a handle is the
   capacity of its storage in doubles, or 0 when the handle is sealed. */
#define VIV_GCAP(a) ((a).dims[7])
VIV_SL_UNUSED static viv_arr viv_seal(viv_arr a) { VIV_GCAP(a) = 0; return a; }
/* file `nw` where the arena had `old` (realloc moved the storage) */
VIV_SL_UNUSED static void viv_g_refile(double* old, double* nw) {
    for (size_t i = viv_arena_top; i-- > 0; ) {
        if (viv_arena[i] == old) { viv_arena[i] = nw; return; }
    }
    viv_arena_push(nw);
}
/* room for `more` doubles past a.size: in place when `own` (the compiler
   proved this variable the only holder) and the handle is unsealed, else a
   fresh block holding a copy */
VIV_SL_UNUSED static viv_arr viv_g_room(viv_arr a, long more, double own) {
    long need = (long)a.size + more;
    if (own != 0.0 && VIV_GCAP(a) > 0 && VIV_GCAP(a) >= need) return a;
    long cap = VIV_GCAP(a) > 0 ? VIV_GCAP(a) : a.size;
    cap = 2 * cap > need ? 2 * cap : need;
    if (cap < 8) cap = 8;
    double* p;
    if (own != 0.0 && VIV_GCAP(a) > 0) {
        p = (double*)realloc(a.data, (size_t)cap * sizeof(double));
        if (!p) { viv_fflush(stdout); viv_errf("error: out of memory growing an array\n"); viv_stop_exit(VIV_E_MEMORY); }
        viv_g_refile(a.data, p);
    } else {
        p = (double*)malloc((size_t)cap * sizeof(double));
        if (!p) { viv_fflush(stdout); viv_errf("error: out of memory growing an array\n"); viv_stop_exit(VIV_E_MEMORY); }
        if (a.size) memcpy(p, a.data, (size_t)a.size * sizeof(double));
        viv_arena_push(p);
    }
    a.data = p;
    VIV_GCAP(a) = (int)cap;
    return a;
}
VIV_SL_UNUSED static viv_arr viv_append_n(viv_arr a, double x, double own, int site) {
    char msg[160];
    if (a.rank != 1) {
        snprintf(msg, sizeof msg, "`append`: a number goes on the end of a vector, and this array has rank %d", a.rank);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    a = viv_g_room(a, 1, own);
    a.data[a.size] = x;
    a.size += 1; a.dims[0] += 1;
    return a;
}
/* an Int array's append: the same storage, its slots read as long long */
VIV_SL_UNUSED static viv_arr viv_iappend(viv_arr a, long long x, double own, int site) {
    (void)site;
    a = viv_g_room(a, 1, own);
    ((long long*)a.data)[a.size] = x;
    a.size += 1; a.dims[0] += 1;
    return a;
}
VIV_SL_UNUSED static viv_arr viv_iseal(viv_arr a) { VIV_GCAP(a) = 0; return a; }
VIV_SL_UNUSED static viv_arr viv_append_r(viv_arr a, viv_arr r, double own, int site) {
    char msg[200];
    if (r.rank != 1) {
        snprintf(msg, sizeof msg, "`append`: a row must be a vector, and this array has rank %d", r.rank);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    if (a.rank == 1 && a.dims[0] != 0) {
        snprintf(msg, sizeof msg, "`append`: an array goes on as a new row of a matrix, and this is a vector of %d numbers (join two vectors with `extend`)", a.dims[0]);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    if (a.rank == 2 && a.dims[1] != r.dims[0]) {
        snprintf(msg, sizeof msg, "`append`: the row has %d numbers, and the matrix has %d columns", r.dims[0], a.dims[1]);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    if (a.rank != 1 && a.rank != 2) {
        snprintf(msg, sizeof msg, "`append`: an array goes on as a new row of a matrix, and this array has rank %d", a.rank);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    double* old = a.data;
    a = viv_g_room(a, r.size, own);
    /* the row may be this array's own storage, moved by the realloc */
    if (r.size) memcpy(a.data + a.size, r.data == old ? a.data : r.data, (size_t)r.size * sizeof(double));
    if (a.rank == 1) { a.rank = 2; a.dims[0] = 0; a.dims[1] = r.dims[0]; }
    a.size += r.size; a.dims[0] += 1;
    return a;
}
VIV_SL_UNUSED static viv_arr viv_extend(viv_arr a, viv_arr b, double own, int site) {
    char msg[120];
    if (a.rank != 1 || b.rank != 1) {
        snprintf(msg, sizeof msg, "`extend` joins vectors; this array has rank %d", a.rank != 1 ? a.rank : b.rank);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    double* old = a.data;
    a = viv_g_room(a, b.size, own);
    if (b.size) memcpy(a.data + a.size, b.data == old ? a.data : b.data, (size_t)b.size * sizeof(double));
    a.size += b.size; a.dims[0] += b.size;
    return a;
}
/* the adjoint helpers of append and extend: a's share of g (its first
   numel(a) elements, in a's shape) and x's share (the rest) */
VIV_SL_UNUSED static void viv_g_adshort(viv_arr g, int rank, const int* dims) {
    char gs[120], ls[120], msg[300];
    viv_sl_shape(gs, sizeof gs, g.rank, g.dims); viv_sl_shape(ls, sizeof ls, rank, dims);
    snprintf(msg, sizeof msg, "autodiff: the adjoint of an append has shape %s, too small for a piece of shape %s", gs, ls);
    viv_sl_fail(VIV_E_ERROR, msg, 0, 0);
}
VIV_SL_UNUSED static viv_arr viv_grow_head(viv_arr g, viv_arr like) {
    if (like.size > g.size) viv_g_adshort(g, like.rank, like.dims);
    viv_arr r = viv_arr_new(like.rank, like.dims);
    if (r.size) memcpy(r.data, g.data, (size_t)r.size * sizeof(double));
    return r;
}
VIV_SL_UNUSED static double viv_grow_tail_n(viv_arr g, viv_arr like, double x) {
    (void)x;
    if (like.size >= g.size) { int one = 1; viv_g_adshort(g, 1, &one); }
    return g.data[like.size];
}
VIV_SL_UNUSED static viv_arr viv_grow_tail_a(viv_arr g, viv_arr like, viv_arr x) {
    if (like.size + x.size > g.size) viv_g_adshort(g, x.rank, x.dims);
    viv_arr r = viv_arr_new(x.rank, x.dims);
    if (r.size) memcpy(r.data, g.data + like.size, (size_t)r.size * sizeof(double));
    return r;
}"#;
