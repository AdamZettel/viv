// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Slices, `copy`, `concat`, `hcat` and `vcat`.
//!
//! A slice is a COPY, never a view: `b = a[1:3]` makes a new array, and
//! `a[1:3] = b` copies `b` into `a`. That keeps the aliasing rule (only `b = a`
//! and parameter passing share storage) and the arena as they are. Where no
//! program can tell the difference, an engine reads a slice in place instead
//! of copying it: a slice that a reduction or an elementwise expression
//! consumes at once (crate::fuse; `axes` below gives its layout).
//!
//! The parser writes `a[lo:hi:s, j]` as `viv__slice_get(a, spec)` and
//! `a[lo:hi, j] = v` as `viv__slice_set(a, spec, v)`, where `spec` is an array
//! literal with four numbers per index written: `[kind, lo, hi, step]`, kind 0
//! for a plain index (`lo` is the index; the axis is dropped), 1 for a range
//! with an end, 2 for a range to the end of the axis (`hi` unused); 3 and 4
//! are 1 and 2 with the start left out before a step the parser cannot see
//! is positive (`a[::s]`), which for an array starts at 0. So every
//! engine sees two calls it already knows how to route.
//!
//! This file is the one statement of the rules: the interpreter's arms and the
//! wasm runtime's in-place versions call `plan` and `cat` here, and `C_RUNTIME`
//! below is their C twin, check for check and word for word.
//!
//! The rules:
//! - Bounds are read as an index is (`as i64`: toward zero, NaN as 0).
//! - A negative index or bound counts from the end, as in Python: `a[-1]`
//!   is the last element, `a[:-1]` all but it. After that, a range needs
//!   `0 <= lo <= hi <= n` and a step of 1 or more; `lo == hi` is an empty
//!   slice. There is no clamping: a bound outside the axis stops the run,
//!   located, in every engine.
//! - Axes left out at the end are taken whole (`X[0:2]` on a matrix is its
//!   first two rows), as in NumPy; more indices than the rank is an error.
//! - A write takes a number (every selected element set to it) or an array of
//!   exactly the slice's shape. The value is copied before the write, so
//!   `a[1:4] = a[0:3]` shifts, as in NumPy.

use super::{ArrData, Val};
use std::cell::RefCell;
use std::rc::Rc;

/// The callee the parser writes for a slice read.
pub const GET: &str = "viv__slice_get";
/// The callee the parser writes for a slice write.
pub const SET: &str = "viv__slice_set";

/// A refused slice: the message, and the note printed under the location.
pub(crate) struct SliceErr {
    pub msg: String,
    pub note: Option<&'static str>,
}

fn err(msg: String) -> SliceErr {
    SliceErr { msg, note: None }
}

/// Where a slice's elements are: the result's shape, and the flat offset (in
/// elements) of each of its elements in the source, in row-major order.
pub(crate) struct Plan {
    pub shape: Vec<usize>,
    pub offs: Vec<usize>,
}

/// A slice axis by axis: where each axis starts, its step and count, the
/// source's strides, and the result's shape.
pub(crate) struct Axes {
    pub start: Vec<i64>,
    pub step: Vec<i64>,
    pub count: Vec<usize>,
    pub stride: Vec<usize>,
    pub shape: Vec<usize>,
}

impl Axes {
    /// The slice as (first offset, step) when its elements are evenly
    /// spaced in the source, row-major (crate::fuse reads it in place).
    pub fn affine(&self) -> Option<(i64, i64)> {
        let start: i64 = (0..self.start.len()).map(|d| self.start[d] * self.stride[d] as i64).sum();
        let (mut st, mut want) = (1i64, None::<i64>);
        for d in (0..self.start.len()).rev() {
            if self.count[d] <= 1 {
                continue;
            }
            let s = self.step[d] * self.stride[d] as i64;
            match want {
                None => st = s,
                Some(w) if w != s => return None,
                _ => {}
            }
            want = Some(s * self.count[d] as i64);
        }
        Some((start, st))
    }
}

/// The elements `spec` selects from an array of `shape`.
pub(crate) fn plan(shape: &[usize], spec: &[f64]) -> Result<Plan, SliceErr> {
    let a = axes(shape, spec)?;
    let rank = shape.len();
    let (start, step, count, stride, out) = (a.start, a.step, a.count, a.stride, a.shape);
    let total: usize = count.iter().product();
    let mut offs = Vec::with_capacity(total);
    if total > 0 {
        let mut idx = vec![0usize; rank];
        loop {
            offs.push((0..rank).map(|d| (start[d] + idx[d] as i64 * step[d]) as usize * stride[d]).sum());
            // the last axis moves fastest
            let mut d = rank;
            loop {
                if d == 0 {
                    return Ok(Plan { shape: out, offs });
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < count[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
    Ok(Plan { shape: out, offs })
}

/// The axes of the slice `spec` selects from an array of `shape`, with every
/// check a slice makes.
pub(crate) fn axes(shape: &[usize], spec: &[f64]) -> Result<Axes, SliceErr> {
    let rank = shape.len();
    let m = spec.len() / 4;
    if m > rank {
        return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("array has rank {rank} but was indexed with {m} index/indices"))));
    }
    let (mut start, mut step, mut count) = (vec![0i64; rank], vec![1i64; rank], vec![1usize; rank]);
    let mut out = Vec::new();
    for d in 0..rank {
        let n = shape[d];
        let (kind, lo, hi, st) = if d < m {
            let s = &spec[4 * d..4 * d + 4];
            (s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64)
        } else {
            (2, 0, 0, 1)
        };
        // a negative index or bound counts from the end, as in Python
        let wrap = |i: i64| if i < 0 { i.saturating_add(n as i64) } else { i };
        if kind == 0 {
            let at = wrap(lo);
            if at < 0 || at as usize >= n {
                return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("index {lo} out of bounds for dimension {d} (size {n})"))));
            }
            start[d] = at;
            continue;
        }
        // a negative step walks down: from `lo` (the last element when it is
        // left out) to just after `hi` (through the first element when it is
        // left out), as in Python, and with no clamping, as a positive step
        if st < 0 {
            let n = n as i64;
            let first = if kind == 3 || kind == 4 { n - 1 } else { wrap(lo) };
            let stop = if kind == 2 || kind == 4 { -1 } else { wrap(hi) };
            let c = if (kind == 3 || kind == 4) && n == 0 {
                0
            } else {
                if first < 0 || (kind != 2 && kind != 4 && stop < 0) {
                    let x = if first < 0 { lo } else { hi };
                    return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("slice bound {x} is before the start of dimension {d} (size {n})"))));
                }
                if first >= n {
                    return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("slice start {first} is past the end of dimension {d} (size {n})"))));
                }
                if stop > first {
                    return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("slice start {first} is before its end {stop} in dimension {d}, and the step is negative"))));
                }
                if first == stop { 0 } else { ((first - stop - 1) / -st + 1) as usize }
            };
            start[d] = first;
            step[d] = st;
            count[d] = c;
            out.push(c);
            continue;
        }
        // kinds 3 and 4 are a range whose start was left out before a step
        // that is not a positive literal (`a[::s]`): for an array the step
        // must be positive, so the start is 0, as the spec holds it
        let hi = if kind == 2 || kind == 4 { n as i64 } else { hi };
        if st == 0 {
            return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("slice step {st} in dimension {d} must not be zero"))));
        }
        let (lo0, hi0) = (lo, hi);
        let (lo, hi) = (wrap(lo), wrap(hi));
        if lo < 0 || hi < 0 {
            let x = if lo < 0 { lo0 } else { hi0 };
            return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("slice bound {x} is before the start of dimension {d} (size {n})"))));
        }
        if hi as usize > n {
            return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("slice end {hi} is past the end of dimension {d} (size {n})"))));
        }
        if lo > hi {
            return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("slice start {lo} is after its end {hi} in dimension {d}"))));
        }
        let c = if hi == lo { 0 } else { ((hi - lo - 1) / st + 1) as usize };
        start[d] = lo;
        step[d] = st;
        count[d] = c;
        out.push(c);
    }
    if out.is_empty() {
        return Err(err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "a slice needs at least one range (`lo:hi` or `:`)")));
    }
    let mut stride = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    Ok(Axes { start, step, count, stride, shape: out })
}

/// The message for a written value whose shape is not the slice's.
pub(crate) fn shape_mismatch(value: &[usize], slice: &[usize]) -> String {
    crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("slice assignment: the value has shape {value:?}, the slice has shape {slice:?}"))
}

/// `concat` (mode 0), `vcat` (1) and `hcat` (2) of two arrays, or of one when
/// `b` is `None`. `w` is the element width in doubles (1 real, 2 complex). A
/// vector is a row to `vcat` and a column to `hcat`, so `hcat(v, w)` is
/// `transpose(vcat(v, w))`. Several arguments are folded left, two at a time,
/// in every engine.
pub(crate) fn cat(mode: u8, a: (&[usize], &[f64]), b: Option<(&[usize], &[f64])>, w: usize) -> Result<(Vec<usize>, Vec<f64>), String> {
    let name = ["concat", "vcat", "hcat"][mode as usize];
    let check = |s: &[usize]| -> Result<(), String> {
        match (mode, s.len()) {
            (0, 1) | (1 | 2, 1 | 2) => Ok(()),
            (0, r) => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`concat` joins vectors; this array has rank {r}"))),
            (_, r) => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` joins vectors and matrices; this array has rank {r}"))),
        }
    };
    // (rows, cols) as vcat and hcat see them
    let rc = |s: &[usize]| -> (usize, usize) {
        match (mode, s.len()) {
            (1, 1) => (1, s[0]),
            (2, 1) => (s[0], 1),
            _ => (s[0], s[1]),
        }
    };
    check(a.0)?;
    let Some(b) = b else {
        let shape = if mode == 0 { a.0.to_vec() } else { let (r, c) = rc(a.0); vec![r, c] };
        return Ok((shape, a.1.to_vec()));
    };
    check(b.0)?;
    match mode {
        0 => {
            let mut data = a.1.to_vec();
            data.extend_from_slice(b.1);
            Ok((vec![a.0[0] + b.0[0]], data))
        }
        1 => {
            let ((ra, ca), (rb, cb)) = (rc(a.0), rc(b.0));
            if ca != cb {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`vcat`: the arrays have {ca} and {cb} columns")));
            }
            let mut data = a.1.to_vec();
            data.extend_from_slice(b.1);
            Ok((vec![ra + rb, ca], data))
        }
        _ => {
            let ((ra, ca), (rb, cb)) = (rc(a.0), rc(b.0));
            if ra != rb {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`hcat`: the arrays have {ra} and {rb} rows")));
            }
            let mut data = Vec::with_capacity(a.1.len() + b.1.len());
            for i in 0..ra {
                data.extend_from_slice(&a.1[w * i * ca..w * (i + 1) * ca]);
                data.extend_from_slice(&b.1[w * i * cb..w * (i + 1) * cb]);
            }
            Ok((vec![ra, ca + cb], data))
        }
    }
}

/// How far an argument of `concat` (mode 0), `vcat` (1) or `hcat` (2) reaches
/// along the joined axis: the autodiff pass sums these into offsets.
pub(crate) fn extent(mode: u8, s: &[usize]) -> usize {
    match (mode, s.len()) {
        (0, _) => s.first().copied().unwrap_or(0),
        // a vector is one row to vcat and one column to hcat
        (_, 1) => 1,
        (1, _) => s[0],
        _ => s.get(1).copied().unwrap_or(1),
    }
}

/// The piece of a joined array's adjoint `g` that belongs to an argument of
/// shape `like` placed at `off` along the joined axis, in `like`'s shape.
pub(crate) fn part(mode: u8, g: (&[usize], &[f64]), off: usize, like: &[usize]) -> Result<Vec<f64>, String> {
    let n: usize = like.iter().product();
    let e = extent(mode, like);
    let bad = || format!("autodiff: the adjoint of a join has shape {:?}, too small for a piece of shape {like:?} at {off}", g.0);
    match mode {
        0 | 1 => {
            // whole rows (or the vector's elements) in order
            let per = if mode == 0 { 1 } else { g.0.get(1).copied().unwrap_or(1) };
            let (lo, hi) = (off * per, (off + e) * per);
            if hi > g.1.len() || hi - lo != n {
                return Err(bad());
            }
            Ok(g.1[lo..hi].to_vec())
        }
        _ => {
            if g.0.len() != 2 || off + e > g.0[1] || g.0[0] * e != n {
                return Err(bad());
            }
            let (rows, cols) = (g.0[0], g.0[1]);
            let mut data = Vec::with_capacity(n);
            for i in 0..rows {
                data.extend_from_slice(&g.1[i * cols + off..i * cols + off + e]);
            }
            Ok(data)
        }
    }
}

fn arr(shape: Vec<usize>, data: Vec<f64>, cplx: bool) -> Val {
    let a = Rc::new(RefCell::new(ArrData { shape, data }));
    if cplx { Val::CArr(a) } else { Val::Arr(a) }
}

fn fail(e: SliceErr, notes: &RefCell<Vec<String>>) -> String {
    if let Some(n) = e.note {
        notes.borrow_mut().push(n.to_string());
    }
    e.msg
}

/// The interpreter's arms. `None`: not one of these builtins.
pub(super) fn builtin(callee: &str, a: &[Val], notes: &RefCell<Vec<String>>) -> Result<Option<Val>, String> {
    let spec_of = |v: &Val| -> Result<Vec<f64>, String> {
        match v {
            Val::Arr(s) => Ok(s.borrow().data.clone()),
            _ => Err("internal: a slice's index list must be a real array".into()),
        }
    };
    Ok(Some(match (callee, a.len()) {
        // an Int array: the same plan over its Ints (B343)
        (GET, 2) if matches!(a[0], Val::IArr(_)) => {
            let Val::IArr(x) = &a[0] else { unreachable!() };
            let xb = x.borrow();
            let p = plan(&[xb.len()], &spec_of(&a[1])?).map_err(|e| fail(e, notes))?;
            Val::IArr(Rc::new(RefCell::new(p.offs.iter().map(|&o| xb[o]).collect())))
        }
        (SET, 3) if matches!(a[0], Val::IArr(_)) => {
            let Val::IArr(x) = &a[0] else { unreachable!() };
            let n = x.borrow().len();
            let p = plan(&[n], &spec_of(&a[1])?).map_err(|e| fail(e, notes))?;
            let Val::IArr(v) = &a[2] else { return Err("a slice of an Int array takes an Int array".into()) };
            // the value is read (copied) before the target is written
            let vals = v.borrow().clone();
            if [vals.len()] != p.shape[..] {
                return Err(shape_mismatch(&[vals.len()], &p.shape));
            }
            let mut xb = x.borrow_mut();
            for (j, &o) in p.offs.iter().enumerate() {
                xb[o] = vals[j];
            }
            Val::Num(0.0)
        }
        (GET, 2) => {
            let (x, cplx) = match &a[0] {
                Val::Arr(x) => (x, false),
                Val::CArr(x) => (x, true),
                _ => return Err("only an array can be sliced".into()),
            };
            let spec = spec_of(&a[1])?;
            let xb = x.borrow();
            let p = plan(&xb.shape, &spec).map_err(|e| fail(e, notes))?;
            let w = if cplx { 2 } else { 1 };
            let mut data = Vec::with_capacity(w * p.offs.len());
            for &o in &p.offs {
                data.extend_from_slice(&xb.data[w * o..w * o + w]);
            }
            arr(p.shape, data, cplx)
        }
        (SET, 3) => {
            let (x, cplx) = match &a[0] {
                Val::Arr(x) => (x, false),
                Val::CArr(x) => (x, true),
                _ => return Err("only an array can be sliced".into()),
            };
            let spec = spec_of(&a[1])?;
            let shape = x.borrow().shape.clone();
            let p = plan(&shape, &spec).map_err(|e| fail(e, notes))?;
            // the value is read (copied) before the target is written
            let (vals, fill): (Vec<f64>, bool) = match (&a[2], cplx) {
                (Val::Num(v), false) => (vec![*v], true),
                (Val::Cplx(re, im), true) => (vec![*re, *im], true),
                (Val::Arr(v), false) | (Val::CArr(v), true) => {
                    let vb = v.borrow();
                    if vb.shape != p.shape {
                        return Err(shape_mismatch(&vb.shape, &p.shape));
                    }
                    (vb.data.clone(), false)
                }
                (_, false) => return Err("a slice of a real array takes a number or a real array".into()),
                (_, true) => return Err("a slice of a complex array takes a complex number or a complex array".into()),
            };
            let w = if cplx { 2 } else { 1 };
            let mut xb = x.borrow_mut();
            for (j, &o) in p.offs.iter().enumerate() {
                let src = if fill { &vals[..] } else { &vals[w * j..w * j + w] };
                xb.data[w * o..w * o + w].copy_from_slice(src);
            }
            Val::Num(0.0)
        }
        ("copy", 1) => match &a[0] {
            Val::Arr(x) => arr(x.borrow().shape.clone(), x.borrow().data.clone(), false),
            Val::CArr(x) => arr(x.borrow().shape.clone(), x.borrow().data.clone(), true),
            _ => return Err("`copy` takes an array".into()),
        },
        ("concat" | "vcat" | "hcat", n) if n >= 1 => {
            let mode = match callee {
                "concat" => 0,
                "vcat" => 1,
                _ => 2,
            };
            let cplx = matches!(a[0], Val::CArr(_));
            let parts: Vec<(Vec<usize>, Vec<f64>)> = a
                .iter()
                .map(|v| match (v, cplx) {
                    (Val::Arr(x), false) | (Val::CArr(x), true) => Ok((x.borrow().shape.clone(), x.borrow().data.clone())),
                    (Val::Arr(_) | Val::CArr(_), _) => Err(format!("`{callee}`: the arrays must all be real or all complex")),
                    _ => Err(format!("`{callee}` takes arrays")),
                })
                .collect::<Result<_, _>>()?;
            let w = if cplx { 2 } else { 1 };
            let (s0, d0) = &parts[0];
            let mut acc = cat(mode, (s0, d0), parts.get(1).map(|(s, d)| (&s[..], &d[..])), w)?;
            for (s, d) in parts.iter().skip(2) {
                acc = cat(mode, (&acc.0, &acc.1), Some((s, d)), w)?;
            }
            arr(acc.0, acc.1, cplx)
        }
        // the autodiff pass's helpers for `concat`, `vcat` and `hcat`
        ("__ad_extent", 2) => {
            let mode = a[0].as_num("a join's mode")? as u8;
            match &a[1] {
                Val::Arr(x) => Val::Num(extent(mode, &x.borrow().shape) as f64),
                _ => return Err("autodiff: `grad` differentiates numbers and real arrays only".into()),
            }
        }
        ("__ad_cat", 4) => {
            let (Val::Arr(g), Val::Arr(like)) = (&a[0], &a[3]) else {
                return Err("autodiff: `grad` differentiates numbers and real arrays only".into());
            };
            let mode = a[1].as_num("a join's mode")? as u8;
            let off = a[2].as_num("a join's offset")? as usize;
            let (gb, lb) = (g.borrow(), like.borrow());
            let data = part(mode, (&gb.shape, &gb.data), off, &lb.shape)?;
            arr(lb.shape.clone(), data, false)
        }
        _ => return Ok(None),
    }))
}

/// The C runtime's twin of `plan`, the writes and `cat`, and of the autodiff
/// helpers. The notes come from `crate::hints`, so their words are shared.
pub fn c_runtime() -> &'static str {
    static CODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CODE.get_or_init(|| {
        let c_str = |s: &str| format!("{s:?}");
        let _ = c_str;
        format!("{}\n{}", C_HEAD, C_BODY)
    })
}

/// A complex number written into a slice of a complex array (needs <complex.h>).
pub const C_CPLX: &str = "VIV_SL_UNUSED static double viv_setslice_c(viv_arr a, viv_arr spec, double complex z, int site) {
    viv_slplan p; viv_sl_plan(&p, a, spec, site);
    for (long long j = 0; j < p.total; j++) { long long o = viv_sl_off(&p, j); a.data[2 * o] = creal(z); a.data[2 * o + 1] = cimag(z); }
    return 0.0;
}";

const C_HEAD: &str = "/* Slices, copy, concat, vcat, hcat: copies, never views. The rules and the
   words are lang/viv/src/interp/slices.rs's, check for check. A spec holds
   four numbers per index written: kind (0 index, 1 lo:hi, 2 lo:, 3 :hi:s, 4 ::s), lo, hi,
   step. An error stops the program with the interpreter's words and, in a
   debug build, its location. */";

const C_BODY: &str = r#"#if defined(__GNUC__)
#define VIV_SL_UNUSED __attribute__((unused))
#else
#define VIV_SL_UNUSED
#endif
#if defined(__GNUC__)
__attribute__((noreturn))
#endif
static void viv_sl_fail(int kind, const char* msg, const char* note, int site) {
    viv_fflush(stdout);
#ifdef VIV_BOUNDS
    viv_errf("error: %s\n%s", msg, viv_sites[site]);
#else
    (void)site;
    viv_errf("error: %s\n", msg);
#endif
    if (note) viv_errf("  note: %s\n", note);
    viv_stop(kind);
}
/* a bound read as Rust's `as i64` reads it */
VIV_SL_UNUSED static long long viv_sl_i64(double x) {
    return x != x ? 0
         : x >= 9223372036854775807.0 ? LLONG_MAX
         : x < -9223372036854775807.0 ? LLONG_MIN
         : (long long)x;
}
/* "[2, 3]", as Rust prints a shape */
VIV_SL_UNUSED static void viv_sl_shape(char* buf, size_t cap, int rank, const int* dims) {
    size_t k = (size_t)snprintf(buf, cap, "[");
    for (int d = 0; d < rank && k < cap; d++) k += (size_t)snprintf(buf + k, cap - k, d ? ", %d" : "%d", dims[d]);
    if (k < cap) snprintf(buf + k, cap - k, "]");
}
typedef struct { int rank, orank; int odims[8]; long long start[8], step[8], cnt[8], stride[8]; long long total; } viv_slplan;
VIV_SL_UNUSED static void viv_sl_plan(viv_slplan* p, viv_arr a, viv_arr spec, int site) {
    char msg[200];
    int rank = a.rank, m = spec.size / 4;
    if (m > rank) {
        snprintf(msg, sizeof msg, "array has rank %d but was indexed with %d index/indices", rank, m);
        viv_sl_fail(VIV_E_INDEX, msg, 0, site);
    }
    p->rank = rank; p->orank = 0; p->total = 1;
    for (int d = 0; d < rank; d++) {
        long long n = a.dims[d], kind = 2, lo = 0, hi = 0, st = 1;
        if (d < m) {
            kind = viv_sl_i64(spec.data[4 * d]); lo = viv_sl_i64(spec.data[4 * d + 1]);
            hi = viv_sl_i64(spec.data[4 * d + 2]); st = viv_sl_i64(spec.data[4 * d + 3]);
        }
        p->step[d] = 1; p->cnt[d] = 1;
        /* a negative index or bound counts from the end, as in Python */
        if (kind == 0) {
            long long at = lo < 0 ? lo + n : lo;
            if (at < 0 || at >= n) {
                snprintf(msg, sizeof msg, "index %lld out of bounds for dimension %d (size %lld)", lo, d, n);
                viv_sl_fail(VIV_E_INDEX, msg, 0, site);
            }
            p->start[d] = at;
            continue;
        }
        /* a negative step walks down (slices.rs, plan) */
        if (st < 0) {
            long long first = (kind == 3 || kind == 4) ? n - 1 : (lo < 0 ? lo + n : lo);
            long long stop = (kind == 2 || kind == 4) ? -1 : (hi < 0 ? hi + n : hi);
            long long c = 0;
            if (!((kind == 3 || kind == 4) && n == 0)) {
                if (first < 0 || (kind != 2 && kind != 4 && stop < 0)) {
                    snprintf(msg, sizeof msg, "slice bound %lld is before the start of dimension %d (size %lld)", first < 0 ? lo : hi, d, n);
                    viv_sl_fail(VIV_E_INDEX, msg, 0, site);
                }
                if (first >= n) {
                    snprintf(msg, sizeof msg, "slice start %lld is past the end of dimension %d (size %lld)", first, d, n);
                    viv_sl_fail(VIV_E_INDEX, msg, 0, site);
                }
                if (stop > first) {
                    snprintf(msg, sizeof msg, "slice start %lld is before its end %lld in dimension %d, and the step is negative", first, stop, d);
                    viv_sl_fail(VIV_E_INDEX, msg, 0, site);
                }
                c = first == stop ? 0 : (first - stop - 1) / -st + 1;
            }
            p->start[d] = first; p->step[d] = st; p->cnt[d] = c;
            p->odims[p->orank++] = (int)c;
            continue;
        }
        if (kind == 2 || kind == 4) hi = n;
        if (st == 0) {
            snprintf(msg, sizeof msg, "slice step %lld in dimension %d must not be zero", st, d);
            viv_sl_fail(VIV_E_VALUE, msg, 0, site);
        }
        {
            long long lo0 = lo, hi0 = hi;
            if (lo < 0) lo += n;
            if (hi < 0) hi += n;
            if (lo < 0 || hi < 0) {
                snprintf(msg, sizeof msg, "slice bound %lld is before the start of dimension %d (size %lld)", lo < 0 ? lo0 : hi0, d, n);
                viv_sl_fail(VIV_E_INDEX, msg, 0, site);
            }
        }
        if (hi > n) {
            snprintf(msg, sizeof msg, "slice end %lld is past the end of dimension %d (size %lld)", hi, d, n);
            viv_sl_fail(VIV_E_INDEX, msg, 0, site);
        }
        if (lo > hi) {
            snprintf(msg, sizeof msg, "slice start %lld is after its end %lld in dimension %d", lo, hi, d);
            viv_sl_fail(VIV_E_INDEX, msg, 0, site);
        }
        p->start[d] = lo; p->step[d] = st;
        p->cnt[d] = hi == lo ? 0 : (hi - lo - 1) / st + 1;
        p->odims[p->orank++] = (int)p->cnt[d];
    }
    if (p->orank == 0) viv_sl_fail(VIV_E_VALUE, "a slice needs at least one range (`lo:hi` or `:`)", 0, site);
    for (int d = rank - 1; d >= 0; d--) p->stride[d] = d == rank - 1 ? 1 : p->stride[d + 1] * a.dims[d + 1];
    for (int d = 0; d < rank; d++) p->total *= p->cnt[d];
}
/* the flat offset of the j-th selected element (row-major over the axes) */
VIV_SL_UNUSED static long long viv_sl_off(const viv_slplan* p, long long j) {
    long long off = 0;
    for (int d = p->rank - 1; d >= 0; d--) {
        long long i = j % p->cnt[d]; j /= p->cnt[d];
        off += (p->start[d] + i * p->step[d]) * p->stride[d];
    }
    return off;
}
VIV_SL_UNUSED static viv_arr viv_getslice(viv_arr a, viv_arr spec, double wf, int site) {
    viv_slplan p; viv_sl_plan(&p, a, spec, site);
    int w = (int)wf;
    viv_arr r = w == 2 ? viv_carr_new(p.orank, p.odims) : viv_arr_new(p.orank, p.odims);
    for (long long j = 0; j < p.total; j++) memcpy(r.data + w * j, a.data + w * viv_sl_off(&p, j), (size_t)w * sizeof(double));
    return r;
}
VIV_SL_UNUSED static double viv_setslice(viv_arr a, viv_arr spec, viv_arr v, double wf, int site) {
    viv_slplan p; viv_sl_plan(&p, a, spec, site);
    int w = (int)wf, same = v.rank == p.orank;
    for (int d = 0; same && d < v.rank; d++) same = v.dims[d] == p.odims[d];
    if (!same) {
        char vs[120], ps[120], msg[300];
        viv_sl_shape(vs, sizeof vs, v.rank, v.dims); viv_sl_shape(ps, sizeof ps, p.orank, p.odims);
        snprintf(msg, sizeof msg, "slice assignment: the value has shape %s, the slice has shape %s", vs, ps);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    /* the value is copied before the target is written: it may be the target */
    size_t bytes = (size_t)w * (size_t)p.total * sizeof(double);
    double* tmp = (double*)malloc(bytes ? bytes : 1);
    memcpy(tmp, v.data, bytes);
    for (long long j = 0; j < p.total; j++) memcpy(a.data + w * viv_sl_off(&p, j), tmp + w * j, (size_t)w * sizeof(double));
    free(tmp);
    return 0.0;
}
VIV_SL_UNUSED static double viv_setslice_n(viv_arr a, viv_arr spec, double x, int site) {
    viv_slplan p; viv_sl_plan(&p, a, spec, site);
    for (long long j = 0; j < p.total; j++) a.data[viv_sl_off(&p, j)] = x;
    return 0.0;
}
VIV_SL_UNUSED static viv_arr viv_copy(viv_arr a, double wf) {
    int w = (int)wf;
    viv_arr r = w == 2 ? viv_carr_new(a.rank, a.dims) : viv_arr_new(a.rank, a.dims);
    memcpy(r.data, a.data, (size_t)w * (size_t)a.size * sizeof(double));
    return r;
}
/* concat (mode 0), vcat (1), hcat (2) of a and b (viv_cat), or of a alone
   (viv_cat1).
   A vector is a row to vcat and a column to hcat. */
VIV_SL_UNUSED static void viv_cat_rc(int mode, viv_arr s, int* r, int* c) {
    if (s.rank == 1) { if (mode == 1) { *r = 1; *c = s.dims[0]; } else { *r = s.dims[0]; *c = 1; } }
    else { *r = s.dims[0]; *c = s.dims[1]; }
}
VIV_SL_UNUSED static void viv_cat_check(int mode, viv_arr s, int site) {
    char msg[120];
    if (mode == 0 && s.rank != 1) {
        snprintf(msg, sizeof msg, "`concat` joins vectors; this array has rank %d", s.rank);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
    if (mode != 0 && s.rank != 1 && s.rank != 2) {
        snprintf(msg, sizeof msg, "`%s` joins vectors and matrices; this array has rank %d", mode == 1 ? "vcat" : "hcat", s.rank);
        viv_sl_fail(VIV_E_VALUE, msg, 0, site);
    }
}
VIV_SL_UNUSED static viv_arr viv_cat1(double modef, viv_arr a, double wf, int site) {
    int mode = (int)modef, w = (int)wf, ra, ca, dims[2];
    viv_cat_check(mode, a, site);
    if (mode == 0) return viv_copy(a, wf);
    viv_cat_rc(mode, a, &ra, &ca); dims[0] = ra; dims[1] = ca;
    viv_arr r = w == 2 ? viv_carr_new(2, dims) : viv_arr_new(2, dims);
    memcpy(r.data, a.data, (size_t)w * (size_t)a.size * sizeof(double));
    return r;
}
VIV_SL_UNUSED static viv_arr viv_cat(double modef, viv_arr a, viv_arr b, double wf, int site) {
    int mode = (int)modef, w = (int)wf, ra, ca, rb, cb, dims[2];
    char msg[120];
    viv_cat_check(mode, a, site);
    viv_cat_check(mode, b, site);
    if (mode == 0) {
        dims[0] = a.dims[0] + b.dims[0];
        viv_arr r = w == 2 ? viv_carr_new(1, dims) : viv_arr_new(1, dims);
        memcpy(r.data, a.data, (size_t)w * (size_t)a.size * sizeof(double));
        memcpy(r.data + (size_t)w * a.size, b.data, (size_t)w * (size_t)b.size * sizeof(double));
        return r;
    }
    viv_cat_rc(mode, a, &ra, &ca); viv_cat_rc(mode, b, &rb, &cb);
    if (mode == 1) {
        if (ca != cb) { snprintf(msg, sizeof msg, "`vcat`: the arrays have %d and %d columns", ca, cb); viv_sl_fail(VIV_E_VALUE, msg, 0, site); }
        dims[0] = ra + rb; dims[1] = ca;
        viv_arr r = w == 2 ? viv_carr_new(2, dims) : viv_arr_new(2, dims);
        memcpy(r.data, a.data, (size_t)w * (size_t)a.size * sizeof(double));
        memcpy(r.data + (size_t)w * a.size, b.data, (size_t)w * (size_t)b.size * sizeof(double));
        return r;
    }
    if (ra != rb) { snprintf(msg, sizeof msg, "`hcat`: the arrays have %d and %d rows", ra, rb); viv_sl_fail(VIV_E_VALUE, msg, 0, site); }
    dims[0] = ra; dims[1] = ca + cb;
    viv_arr r = w == 2 ? viv_carr_new(2, dims) : viv_arr_new(2, dims);
    for (int i = 0; i < ra; i++) {
        memcpy(r.data + (size_t)w * i * (ca + cb), a.data + (size_t)w * i * ca, (size_t)w * ca * sizeof(double));
        memcpy(r.data + (size_t)w * (i * (ca + cb) + ca), b.data + (size_t)w * i * cb, (size_t)w * cb * sizeof(double));
    }
    return r;
}
/* the autodiff pass's helpers for a join: an argument's reach along the
   joined axis, and its piece of the adjoint */
VIV_SL_UNUSED static double viv_ad_extent(double modef, viv_arr s) {
    int mode = (int)modef;
    if (mode == 0) return s.rank ? s.dims[0] : 0;
    if (s.rank == 1) return 1;
    return mode == 1 ? s.dims[0] : s.dims[1];
}
VIV_SL_UNUSED static viv_arr viv_ad_cat(viv_arr g, double modef, double offf, viv_arr like) {
    int mode = (int)modef, off = (int)offf, e = (int)viv_ad_extent(modef, like);
    viv_arr r = viv_arr_new(like.rank, like.dims);
    int ok;
    if (mode != 2) {
        int per = mode == 0 ? 1 : (g.rank > 1 ? g.dims[1] : 1);
        ok = (off + e) * per <= g.size && e * per == like.size;
        if (ok) memcpy(r.data, g.data + (size_t)off * per, (size_t)like.size * sizeof(double));
    } else {
        ok = g.rank == 2 && off + e <= g.dims[1] && g.dims[0] * e == like.size;
        if (ok) for (int i = 0; i < g.dims[0]; i++) memcpy(r.data + (size_t)i * e, g.data + (size_t)i * g.dims[1] + off, (size_t)e * sizeof(double));
    }
    if (!ok) {
        char gs[120], ls[120], msg[300];
        viv_sl_shape(gs, sizeof gs, g.rank, g.dims); viv_sl_shape(ls, sizeof ls, like.rank, like.dims);
        snprintf(msg, sizeof msg, "autodiff: the adjoint of a join has shape %s, too small for a piece of shape %s at %d", gs, ls, off);
        viv_sl_fail(VIV_E_ERROR, msg, 0, 0);
    }
    return r;
}"#;
