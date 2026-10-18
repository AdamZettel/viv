// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The array vocabulary: shapes,
//! constructors, products, axis reductions, broadcasting, sorting, gathers
//! and masks, scans, shaped random draws, `clip`, `einsum`, and the solvers
//! that return their results.
//!
//! This file is the one statement of the rules. The interpreter calls
//! `builtin`; the wasm runtime reaches the same code through its bridge
//! (`interp/wasm_rt.rs`), and broadcasting through `broadcast` and `expand`;
//! `C_RUNTIME` below is the C twin, loop for loop and word for word. The
//! compiler reads `SIGS` for each builtin's arguments and result
//! (`crate::lower`, `lower_array_call`), so the engines cannot disagree about
//! a signature.
//!
//! NumPy is the reference. Where NumPy's result does not depend on an order
//! of operations (a reshape, a gather, `outer`, `cumsum`, `argsort`) the
//! result is NumPy's bit for bit; where it does, the order is written down:
//!
//! - `sum`, `mean`, `var`, `std` along an axis and `trace` add in NumPy's
//!   order (tools/py2viv/probe_numpy_order.py): from 0.0, a contiguous run in
//!   blocks of 8192 entries, each block by NumPy's pairwise sum; along an axis
//!   that is not the last (with a later axis longer than 1), the slabs added
//!   in order from 0.0. `mean(a)` of the whole array is that sum over every
//!   entry divided by the count, and `sum(a)` of the whole array is that sum
//!   (from release 0.0.1; before, it added left to right). This order binds
//!   the exact engines; fast mode is free to add in another.
//! - `prod`, `cumsum`, `cumprod` go left to right, as NumPy does.
//! //! - `sum(a, ordered=true)` adds every entry left to right (row-major) from
//!   0.0, as a Fortran or C loop does; the parser writes it as `__sum_ordered`.
//! - `dot` and `norm` go left to right from 0.0 (NumPy calls BLAS, which
//!   reorders); `norm(a)` is `sqrt(dot)` over every entry, which is what
//!   `sqrt(sum(a * a))` gives.
//! - `einsum` loops over the output entries in row-major order, and for each
//!   over the summed indices in row-major order (indices ordered by their
//!   first appearance in the subscripts), adding each product (operands
//!   multiplied left to right) to 0.0 in turn. NumPy's einsum has its own
//!   unrolled loops, so it agrees to rounding, not to the bit.

use super::{ArrData, Val, LIMIT_OF, MAX_ELEMS};
use std::cell::RefCell;
use std::rc::Rc;

/// One builtin form: its name, how many arguments, what each one is and what
/// it gives. `params` has one letter per argument: `a` a real array, `n` a
/// number, `b` a boolean, `s` a string literal, `x` a number or a real array.
/// `ret`: `a` an array, `n` a number, `v` nothing (0), `x` what the first
/// argument is, `e` a number or an array by `einsum`'s subscripts.
pub struct Sig {
    pub name: &'static str,
    pub params: &'static str,
    pub ret: char,
    /// the C runtime fragment the compiled call needs
    pub frag: &'static str,
    /// the first argument written in place, if any
    pub writes: Option<usize>,
}

const fn s(name: &'static str, params: &'static str, ret: char) -> Sig {
    Sig { name, params, ret, frag: "arrays", writes: None }
}
const fn sf(name: &'static str, params: &'static str, ret: char, frag: &'static str) -> Sig {
    Sig { name, params, ret, frag, writes: None }
}

/// Every form this file adds. A name may appear with several argument counts;
/// a form missing here (`sum(a)`, `transpose(A)`, `rand()`, ...) is an older
/// builtin, handled where it always was.
pub const SIGS: &[Sig] = &[
    // shapes
    s("reshape", "an", 'a'),
    s("reshape", "ann", 'a'),
    s("reshape", "annn", 'a'),
    s("reshape", "annnn", 'a'),
    s("ravel", "a", 'a'),
    s("flatten", "a", 'a'),
    s("transpose", "aa", 'a'),
    s("swapaxes", "ann", 'a'),
    // constructors
    s("eye", "n", 'a'),
    s("eye", "nn", 'a'),
    s("eye", "nnn", 'a'),
    s("diag", "a", 'a'),
    s("diag", "an", 'a'),
    s("ones", "n", 'a'),
    s("ones", "nn", 'a'),
    s("ones", "nnn", 'a'),
    s("ones", "nnnn", 'a'),
    s("zeros_like", "a", 'a'),
    s("ones_like", "a", 'a'),
    s("arange", "n", 'a'),
    s("arange", "nn", 'a'),
    s("arange", "nnn", 'a'),
    s("linspace", "nnnb", 'a'),
    // products
    s("outer", "aa", 'a'),
    s("dot", "aa", 'n'),
    // `sum(a, ordered=true)` (the parser writes it so): left to right
    s("__sum_ordered", "a", 'n'),
    s("norm", "a", 'n'),
    s("trace", "a", 'n'),
    s("trace", "an", 'n'),
    s("triu", "a", 'a'),
    s("triu", "an", 'a'),
    s("tril", "a", 'a'),
    s("tril", "an", 'a'),
    // reductions
    s("sum", "an", 'a'),
    s("sum", "anb", 'a'),
    s("mean", "a", 'n'),
    s("mean", "an", 'a'),
    s("mean", "anb", 'a'),
    s("var", "a", 'n'),
    s("var", "an", 'a'),
    s("var", "anb", 'a'),
    s("std", "a", 'n'),
    s("std", "an", 'a'),
    s("std", "anb", 'a'),
    s("prod", "a", 'n'),
    s("prod", "an", 'a'),
    s("prod", "anb", 'a'),
    s("amax", "a", 'n'),
    s("amax", "an", 'a'),
    s("amax", "anb", 'a'),
    s("amin", "a", 'n'),
    s("amin", "an", 'a'),
    s("amin", "anb", 'a'),
    s("argmax", "a", 'n'),
    s("argmax", "an", 'a'),
    s("argmin", "a", 'n'),
    s("argmin", "an", 'a'),
    s("any", "an", 'a'),
    s("any", "anb", 'a'),
    s("all", "an", 'a'),
    s("all", "anb", 'a'),
    s("count_nonzero", "a", 'n'),
    // order, gathers, masks
    s("argsort", "a", 'a'),
    s("take", "aan", 'a'),
    s("extract", "aa", 'a'),
    Sig { name: "putmask", params: "aax", ret: 'v', frag: "arrays", writes: Some(0) },
    s("flatnonzero", "a", 'a'),
    // scans
    s("cumsum", "a", 'a'),
    s("cumsum", "an", 'a'),
    s("cumprod", "a", 'a'),
    s("cumprod", "an", 'a'),
    s("diff", "a", 'a'),
    s("diff", "an", 'a'),
    s("diff", "ann", 'a'),
    // elementwise
    s("clip", "xnn", 'x'),
    // random draws, from rand()'s generator
    sf("rand", "n", 'a', "ar_rand"),
    sf("rand", "nn", 'a', "ar_rand"),
    sf("rand", "nnn", 'a', "ar_rand"),
    sf("rand", "nnnn", 'a', "ar_rand"),
    sf("randn", "", 'n', "ar_rand"),
    sf("randn", "n", 'a', "ar_rand"),
    sf("randn", "nn", 'a', "ar_rand"),
    sf("randn", "nnn", 'a', "ar_rand"),
    sf("randn", "nnnn", 'a', "ar_rand"),
    // contractions
    sf("einsum", "sa", 'e', "ar_einsum"),
    sf("einsum", "saa", 'e', "ar_einsum"),
    sf("einsum", "saaa", 'e', "ar_einsum"),
    // solvers that return their results
    sf("lin_solve", "aa", 'a', "ar_solve"),
    sf("inv", "a", 'a', "ar_solve"),
    sf("cholesky", "a", 'a', "ar_chol"),
    Sig { name: "svd", params: "aaaa", ret: 'v', frag: "ar_svd", writes: Some(1) },
];

/// The form `name` with `n` arguments, if this file adds it. Looked up in a
/// table built once: the interpreter asks for every builtin call it makes.
pub fn sig(name: &str, n: usize) -> Option<&'static Sig> {
    static BY_NAME: std::sync::OnceLock<std::collections::HashMap<&'static str, Vec<&'static Sig>>> = std::sync::OnceLock::new();
    let t = BY_NAME.get_or_init(|| {
        let mut m: std::collections::HashMap<&'static str, Vec<&'static Sig>> = std::collections::HashMap::new();
        for s in SIGS {
            m.entry(s.name).or_default().push(s);
        }
        m
    });
    t.get(name)?.iter().copied().find(|s| s.params.len() == n)
}

/// Whether the C function `c` (a `c_name`) gives an array.
pub fn c_gives_array(c: &str) -> bool {
    match of_c_name(c) {
        Some((s, scalar)) => s.ret == 'a' || (matches!(s.ret, 'x' | 'e') && !scalar),
        None => false,
    }
}

/// Whether every form of `name` computes a value and changes nothing: no
/// argument written, no draw from the generator.
pub fn is_pure(name: &str) -> bool {
    is_name(name) && SIGS.iter().filter(|s| s.name == name).all(|s| s.writes.is_none() && s.frag != "ar_rand")
}

/// Whether `name` is a builtin this file adds in some form.
pub fn is_name(name: &str) -> bool {
    SIGS.iter().any(|s| s.name == name)
}

/// The C function a compiled call of `name` with `n` arguments calls; a
/// `scalar` form (a number where an array could be: `clip` of a number,
/// `putmask` of a number, an `einsum` that sums everything) ends in `n`.
pub fn c_name(name: &str, n: usize, scalar: bool) -> String {
    format!("viv_ar_{name}_{n}{}", if scalar { "n" } else { "" })
}

/// The builtin a C function of `c_name` stands for, and whether it is the
/// scalar form.
pub fn of_c_name(c: &str) -> Option<(&'static Sig, bool)> {
    let rest = c.strip_prefix("viv_ar_")?;
    let (name, n) = rest.rsplit_once('_')?;
    let (n, scalar) = match n.strip_suffix('n') {
        Some(k) => (k, true),
        None => (n, false),
    };
    Some((sig(name, n.parse().ok()?)?, scalar))
}

/// The builtins that give the value `a, b = f(A)` unpacks, by the number of
/// results: `eig_sym(A)` and `eig_herm(A)` give (values, vectors), `svd(A)`
/// gives (U, s, Vt). `crate::tuples` writes each as the call that fills
/// arrays it allocates.
pub fn multi_result(name: &str, nargs: usize) -> Option<usize> {
    match (name, nargs) {
        ("eig_sym" | "eig_herm", 1) => Some(2),
        ("svd", 1) => Some(3),
        _ => None,
    }
}

type Nd = (Vec<usize>, Vec<f64>);

// ---- helpers -------------------------------------------------------------

/// A number read as an index or a count is, toward zero (NaN as 0).
fn int(x: f64) -> i64 {
    x as i64
}

fn new(shape: Vec<usize>, data: Vec<f64>) -> Val {
    Val::Arr(Rc::new(RefCell::new(ArrData { shape, data })))
}

fn size_of(shape: &[usize]) -> usize {
    shape.iter().fold(1usize, |p, &d| p.saturating_mul(d))
}

fn limit(name: &str, shape: &[usize]) -> Result<(), String> {
    let total = size_of(shape);
    if total > MAX_ELEMS {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`{name}`: an array of {total} elements exceeds the {MAX_ELEMS}-element limit{LIMIT_OF}")));
    }
    Ok(())
}

/// A shape as Rust's `{:?}` writes it, `[2, 3]`, which is also the C runtime's.
pub(crate) fn shape_text(shape: &[usize]) -> String {
    format!("{shape:?}")
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        st[d] = st[d + 1] * shape[d + 1];
    }
    st
}

/// `k` as an axis of a rank-`rank` array: NumPy's rule, -rank to rank - 1.
fn axis(name: &str, k: f64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let i = int(k);
    if i < -r || i >= r {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`{name}`: axis {i} is out of bounds for an array of rank {rank}")));
    }
    Ok(if i < 0 { (i + r) as usize } else { i as usize })
}

/// The dimensions `before`, `along` and `after` an axis.
fn around(shape: &[usize], k: usize) -> (usize, usize, usize) {
    (shape[..k].iter().product(), shape[k], shape[k + 1..].iter().product())
}

/// The shape a reduction along axis `k` leaves.
fn reduced(name: &str, shape: &[usize], k: usize, keep: bool) -> Result<Vec<usize>, String> {
    if keep {
        let mut s = shape.to_vec();
        s[k] = 1;
        return Ok(s);
    }
    if shape.len() == 1 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` along the only axis of a vector gives a number: write `{name}(a)`")));
    }
    let mut s = shape.to_vec();
    s.remove(k);
    Ok(s)
}

/// NumPy's pairwise sum of `f` (numpy/core/src/umath/loops_utils.h.src):
/// below 8 entries left to right from 0; up to 128, eight running sums
/// combined as a tree; above, split in two at a multiple of 8.
pub fn pairwise(f: &[f64]) -> f64 {
    let n = f.len();
    if n < 8 {
        let mut s = 0.0;
        for &x in f {
            s += x;
        }
        return s;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&f[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += f[i + j];
            }
            i += 8;
        }
        let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            s += f[i];
            i += 1;
        }
        return s;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise(&f[..n2]) + pairwise(&f[n2..])
}

/// NumPy's sum of a contiguous run: from 0.0, block by block of 8192 entries
/// (its buffer), each block by the pairwise sum.
pub fn npsum(f: &[f64]) -> f64 {
    if crate::digits::on() {
        return crate::digits::sum(f, npsum_native);
    }
    npsum_native(f)
}

fn npsum_native(f: &[f64]) -> f64 {
    let mut s = 0.0;
    for b in f.chunks(8192) {
        s += pairwise(b);
    }
    s
}

/// Combine the slabs along axis `k` in order, entry by entry: each reduced
/// entry starts at `init` (or at slab 0 when there is none) and takes the
/// later slabs through `f` one at a time. Row-major over the slabs, so the
/// memory is read in order; each entry sees its own values in the axis's order.
fn slabs(shape: &[usize], data: &[f64], k: usize, init: Option<f64>, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let (before, along, after) = around(shape, k);
    let mut r = vec![0.0; before * after];
    for o in 0..before {
        let out = &mut r[o * after..(o + 1) * after];
        let first = match init {
            Some(v) => {
                out.fill(v);
                0
            }
            None => {
                out.copy_from_slice(&data[o * along * after..o * along * after + after]);
                1
            }
        };
        for j in first..along {
            let row = &data[(o * along + j) * after..(o * along + j + 1) * after];
            for (t, &x) in out.iter_mut().zip(row) {
                *t = f(*t, x);
            }
        }
    }
    r
}

/// `sum` along axis `k` in NumPy's order, as a flat vector of the reduced
/// entries (row-major over the axes that remain).
fn sum_axis(shape: &[usize], data: &[f64], k: usize) -> Vec<f64> {
    let (before, along, after) = around(shape, k);
    if after == 1 {
        return (0..before).map(|o| npsum(&data[o * along..o * along + along])).collect();
    }
    slabs(shape, data, k, Some(0.0), |t, x| t + x)
}

/// argmax (argmin) along axis `k`: for each entry the first index of the
/// largest (smallest) value, or of the first NaN.
fn arg_axis(shape: &[usize], data: &[f64], k: usize, is_max: bool) -> Vec<f64> {
    let (before, along, after) = around(shape, k);
    let mut r = vec![0.0; before * after];
    for o in 0..before {
        let mut best: Vec<f64> = data[o * along * after..o * along * after + after].to_vec();
        let mut nan: Vec<bool> = best.iter().map(|x| x.is_nan()).collect();
        let at = &mut r[o * after..(o + 1) * after];
        for j in 1..along {
            let row = &data[(o * along + j) * after..(o * along + j + 1) * after];
            for i in 0..after {
                let x = row[i];
                if nan[i] {
                    continue;
                }
                if x.is_nan() {
                    nan[i] = true;
                    at[i] = j as f64;
                } else if (is_max && x > best[i]) || (!is_max && x < best[i]) {
                    best[i] = x;
                    at[i] = j as f64;
                }
            }
        }
    }
    r
}

/// NumPy's `maximum` of two: a NaN wins, and 0.0 is above -0.0 (IEEE 754's
/// maximum, which NumPy's vector loops compute on arm64).
fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || a > b || (a == b && !a.is_sign_negative()) {
        a
    } else {
        b
    }
}
/// NumPy's `minimum`: a NaN wins, and -0.0 is below 0.0.
fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || a < b || (a == b && a.is_sign_negative()) {
        a
    } else {
        b
    }
}

/// The index of the largest (smallest) entry: the first one, and the first
/// NaN if there is one, as NumPy's argmax.
fn arg_ext(it: &mut dyn Iterator<Item = f64>, is_max: bool) -> f64 {
    let mut best = f64::NAN;
    let mut at = 0usize;
    for (j, x) in it.enumerate() {
        if j == 0 {
            best = x;
            if x.is_nan() {
                return 0.0;
            }
            continue;
        }
        if x.is_nan() {
            return j as f64;
        }
        if (is_max && x > best) || (!is_max && x < best) {
            best = x;
            at = j;
        }
    }
    at as f64
}

/// Permute the axes of an array: axis `d` of the result is axis `perm[d]` of
/// the argument.
fn permute(shape: &[usize], data: &[f64], perm: &[usize]) -> Nd {
    let rank = shape.len();
    let st = strides(shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let n = data.len();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let mut off = 0;
        for d in 0..rank {
            off += idx[d] * st[perm[d]];
        }
        out.push(data[off]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    (out_shape, out)
}

/// `transpose(a)` of a real array of any rank: its axes reversed (a vector
/// is copied). The rank-2 case is the old builtin's.
pub(crate) fn transpose_any(shape: &[usize], data: &[f64]) -> Nd {
    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    permute(shape, data, &perm)
}

// ---- broadcasting ------------------------------------------------------------

/// The shape two arrays broadcast to under NumPy's rule (align the last axes;
/// each pair of lengths equal, or one of them 1), or None.
pub fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for d in 0..r {
        let x = if d + a.len() >= r { a[d + a.len() - r] } else { 1 };
        let y = if d + b.len() >= r { b[d + b.len() - r] } else { 1 };
        out[d] = if x == y {
            x
        } else if x == 1 {
            y
        } else if y == 1 {
            x
        } else {
            return None;
        };
    }
    Some(out)
}

/// The entries of an array of `shape` repeated out to the broadcast shape
/// `to`, a row (the last axis) at a time: copied when its length is the
/// row's, repeated when it is 1.
pub fn expand(shape: &[usize], data: &[f64], to: &[usize]) -> Vec<f64> {
    let r = to.len();
    let n = size_of(to);
    if n == 0 {
        return Vec::new();
    }
    let lead = r - shape.len();
    let st = strides(shape);
    let last = to[r - 1];
    let whole = shape[shape.len() - 1] == last;
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; r];
    for _ in 0..n / last {
        let mut off = 0;
        for d in lead..r - 1 {
            if shape[d - lead] != 1 {
                off += idx[d] * st[d - lead];
            }
        }
        if whole {
            out.extend_from_slice(&data[off..off + last]);
        } else {
            out.extend(std::iter::repeat(data[off]).take(last));
        }
        for d in (0..r - 1).rev() {
            idx[d] += 1;
            if idx[d] < to[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Two arrays of different shapes brought to their broadcast shape, or the
/// shape-mismatch error the engines have always given.
pub fn broadcast_pair(sa: &[usize], da: &[f64], sb: &[usize], db: &[f64], prefix: &str) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>), String> {
    let Some(to) = broadcast(sa, sb) else {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("{prefix}array shape mismatch: {sa:?} vs {sb:?}")));
    };
    let total = size_of(&to);
    if total > MAX_ELEMS {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("broadcasting {sa:?} against {sb:?} gives {total} elements, over the {MAX_ELEMS}-element limit{LIMIT_OF}")));
    }
    Ok((to.clone(), expand(sa, da, &to), expand(sb, db, &to)))
}

/// The adjoint of an operand that was broadcast: `g` (of the result's shape)
/// summed down to `like`'s shape, entries added in `g`'s row-major order.
pub fn unbroadcast(g_shape: &[usize], g: &[f64], like: &[usize]) -> Vec<f64> {
    let r = g_shape.len();
    let lead = r - like.len().min(r);
    let st = strides(like);
    let mut out = vec![0.0; size_of(like)];
    let mut idx = vec![0usize; r];
    for &x in g {
        let mut off = 0;
        for d in lead..r {
            if like[d - lead] != 1 {
                off += idx[d] * st[d - lead];
            }
        }
        out[off] += x;
        for d in (0..r).rev() {
            idx[d] += 1;
            if idx[d] < g_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

// ---- adjoints of the vocabulary (crate::grad) -----------------------------------

/// The shape `shape` has after a reduction along axis `k` with keepdims: the
/// shape `__ad_keep` gives a reduction's adjoint, so that it broadcasts back
/// along the axis.
pub fn ad_keep_shape(shape: &[usize], k: f64) -> Result<Vec<usize>, String> {
    let k = ad_axis(k, shape.len())?;
    let mut s = shape.to_vec();
    s[k] = 1;
    Ok(s)
}

/// `k` as an axis of a rank-`rank` array, for a reduction's adjoint.
pub fn ad_axis(k: f64, rank: usize) -> Result<usize, String> {
    axis("a reduction's adjoint", k, rank)
}

/// For each entry, the product of the other entries along axis `k` (of the
/// whole array when `k` is None): `p[j] * s[j]`, where `p[j]` is the product
/// of the entries before `j`, multiplied left to right from 1, and `s[j]` that
/// of the entries after it, multiplied right to left from 1. This is the
/// adjoint of `prod`, and it is right where an entry is zero, where dividing
/// the product by the entry is not.
pub fn ad_prod_others(shape: &[usize], data: &[f64], k: Option<usize>) -> Vec<f64> {
    let (before, along, after) = match k {
        Some(k) => around(shape, k),
        None => (1, data.len(), 1),
    };
    let mut r = vec![0.0; data.len()];
    for o in 0..before {
        for i in 0..after {
            let at = |j: usize| (o * along + j) * after + i;
            let mut p = 1.0;
            for j in 0..along {
                r[at(j)] = p;
                p *= data[at(j)];
            }
            let mut q = 1.0;
            for j in (0..along).rev() {
                r[at(j)] *= q;
                q *= data[at(j)];
            }
        }
    }
    r
}

/// The adjoint of `take(a, idx, k)`: each slice of `g` added into a zero of
/// `a`'s shape at the index along axis `k` it was taken from, in the order
/// `take` read them, so a repeated index collects every share.
pub fn ad_take_axis(g: &[f64], shape: &[usize], idx: &[f64], k: f64) -> Result<Vec<f64>, String> {
    let k = axis("take", k, shape.len())?;
    let (before, along, after) = around(shape, k);
    if g.len() != before * idx.len() * after {
        return Err("autodiff: `take`'s adjoint has the wrong number of entries".into());
    }
    let mut r = vec![0.0; size_of(shape)];
    let mut w = 0;
    for o in 0..before {
        for &f in idx {
            let j = int(f);
            if f.is_nan() || j < 0 || j as usize >= along {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`take`: index {} out of bounds for axis {k} (size {along})", super::format_g(f))));
            }
            let base = (o * along + j as usize) * after;
            for t in 0..after {
                r[base + t] += g[w + t];
            }
            w += after;
        }
    }
    Ok(r)
}

// ---- the builtins -------------------------------------------------------------

/// The arguments of one call, read by what the signature says they are.
struct Args<'a> {
    name: &'static str,
    a: &'a [Val],
}

impl<'a> Args<'a> {
    fn arr(&self, i: usize) -> Result<Nd, String> {
        match &self.a[i] {
            Val::Arr(x) => {
                let b = x.borrow();
                Ok((b.shape.clone(), b.data.clone()))
            }
            Val::CArr(_) => Err(format!("`{}` takes real arrays; this one is complex", self.name)),
            _ => Err(format!("`{}` takes an array as argument {}", self.name, i + 1)),
        }
    }
    fn num(&self, i: usize) -> Result<f64, String> {
        self.a[i].as_num(&format!("`{}`'s argument {}", self.name, i + 1))
    }
    fn flag(&self, i: usize) -> Result<bool, String> {
        match &self.a[i] {
            Val::Bool(b) => Ok(*b),
            _ => Err(format!("`{}` takes true or false as argument {}", self.name, i + 1)),
        }
    }
}

/// Dimensions for a new array, as `zeros` reads them (a negative one is 0).
fn dims(args: &Args, from: usize, to: usize) -> Result<Vec<usize>, String> {
    let mut d = Vec::new();
    for i in from..to {
        let x = int(args.num(i)?);
        d.push(if x < 0 { 0 } else { x.min(i32::MAX as i64) as usize });
    }
    Ok(d)
}

fn reshape(name: &str, size: usize, want: &[f64]) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(want.len());
    let mut free = None;
    for (d, &w) in want.iter().enumerate() {
        let k = int(w);
        if k == -1 {
            if free.is_some() {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}`: only one dimension can be -1")));
            }
            free = Some(d);
            out.push(1);
        } else if k < 0 {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}`: dimension {k} is negative")));
        } else {
            out.push(k as usize);
        }
    }
    let known = size_of(&out);
    if let Some(d) = free {
        if known == 0 || size % known != 0 {
            let spec: Vec<i64> = want.iter().map(|&w| int(w)).collect();
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}`: cannot reshape an array of {size} elements into shape {spec:?}")));
        }
        out[d] = size / known;
    }
    if size_of(&out) != size {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}`: cannot reshape an array of {size} elements into shape {out:?}")));
    }
    Ok(out)
}

/// NumPy's arange: the length `ceil((stop - start) / step)`, the first entry
/// `start`, the second `start + step`, and entry i from 2 on `start + i * d`
/// with `d` the second entry minus the first.
fn arange(start: f64, stop: f64, step: f64) -> Result<Vec<f64>, String> {
    if step == 0.0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`arange`: the step is zero"));
    }
    let delta = stop - start;
    let q = delta / step;
    let len: i64 = if q == 0.0 && delta != 0.0 {
        if q.is_sign_negative() {
            0
        } else {
            1
        }
    } else {
        if q.is_nan() || q.ceil() > MAX_ELEMS as f64 {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`arange`: the length {} is not a number of elements below the {MAX_ELEMS}-element limit{LIMIT_OF}", super::format_g(q.ceil()))));
        }
        q.ceil() as i64
    };
    if len <= 0 {
        return Ok(Vec::new());
    }
    let n = len as usize;
    let mut out = vec![0.0; n];
    out[0] = start;
    if n > 1 {
        out[1] = start + step;
        let d = out[1] - out[0];
        for (i, v) in out.iter_mut().enumerate().skip(2) {
            *v = start + i as f64 * d;
        }
    }
    Ok(out)
}

/// NumPy's linspace: `i * step + start` with `step = (stop - start) / div`
/// (`div` is n - 1, or n without the endpoint), the last entry set to `stop`
/// when it is included. A zero step (or none) scales `i / div` by the span.
fn linspace(start: f64, stop: f64, n: i64, endpoint: bool) -> Result<Vec<f64>, String> {
    if n < 0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`linspace`: the number of points {n} is negative")));
    }
    let n = n as usize;
    if n > MAX_ELEMS {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`linspace`: an array of {n} elements exceeds the {MAX_ELEMS}-element limit{LIMIT_OF}")));
    }
    let div = if endpoint { n as f64 - 1.0 } else { n as f64 };
    let delta = stop - start;
    let step = delta / div;
    let mut y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    if div > 0.0 {
        if step == 0.0 {
            for v in y.iter_mut() {
                *v = *v / div * delta;
            }
        } else {
            for v in y.iter_mut() {
                *v *= step;
            }
        }
    } else {
        for v in y.iter_mut() {
            *v *= delta;
        }
    }
    for v in y.iter_mut() {
        *v += start;
    }
    if endpoint && n > 1 {
        y[n - 1] = stop;
    }
    Ok(y)
}

/// NumPy's `clip` of one entry (as NumPy 1.24 computes it): a NaN entry stays
/// NaN, a NaN bound is no bound, and otherwise `x > lo ? x : lo`, then
/// `m < hi ? m : hi`, so a tie takes the bound (`clip(-0.0, 0.0, 1.0)` is 0.0).
pub fn clip1(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let m = if lo.is_nan() || x > lo { x } else { lo };
    if hi.is_nan() || m < hi {
        m
    } else {
        hi
    }
}

/// Box-Muller from two uniforms, cosine branch: lib/random's `randn`.
fn normal(u1: f64, u2: f64) -> f64 {
    use crate::vmath::VMath;
    (-2.0 * u1.vln()).sqrt() * (2.0 * 3.141592653589793 * u2).vcos()
}

/// One `einsum`, parsed: for each operand its letters; the output letters;
/// every letter in order of first appearance.
struct Einsum {
    ops: Vec<Vec<u8>>,
    out: Vec<u8>,
}

fn einsum_parse(spec: &str, nops: usize) -> Result<Einsum, String> {
    let t: String = spec.chars().filter(|c| *c != ' ').collect();
    let (ins, out) = match t.find("->") {
        Some(p) => (&t[..p], Some(&t[p + 2..])),
        None => (&t[..], None),
    };
    let ops: Vec<Vec<u8>> = ins.split(',').map(|p| p.as_bytes().to_vec()).collect();
    if ops.len() != nops {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: the subscripts name {} operands and the call passes {nops}", ops.len())));
    }
    let letter = |c: u8| c.is_ascii_alphabetic();
    for &c in ops.iter().flatten() {
        if !letter(c) {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: a subscript is a letter, not '{}'", c as char)));
        }
    }
    let count = |c: u8| ops.iter().flatten().filter(|&&x| x == c).count();
    let out: Vec<u8> = match out {
        Some(o) => {
            let o = o.as_bytes().to_vec();
            for (k, &c) in o.iter().enumerate() {
                if !letter(c) {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: a subscript is a letter, not '{}'", c as char)));
                }
                if o[..k].contains(&c) {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: the output names '{}' twice", c as char)));
                }
                if count(c) == 0 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: the output's '{}' is in no operand", c as char)));
                }
            }
            o
        }
        // NumPy's implicit output: the letters written once, in character order
        None => (b'A'..=b'z').filter(|&c| letter(c) && count(c) == 1).collect(),
    };
    if out.len() > crate::lower::MAX_RANK {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: the output has rank {}; arrays have rank {} at most", out.len(), crate::lower::MAX_RANK)));
    }
    Ok(Einsum { ops, out })
}

/// The rank of `einsum`'s result for `spec` (0: a number), for the compiler.
pub fn einsum_rank(spec: &str, nops: usize) -> Result<usize, String> {
    einsum_parse(spec, nops).map(|e| e.out.len())
}

fn einsum(spec: &str, operands: &[Nd]) -> Result<Nd, String> {
    let e = einsum_parse(spec, operands.len())?;
    let mut letters: Vec<u8> = Vec::new();
    let mut size: Vec<usize> = Vec::new();
    for (k, (shape, _)) in operands.iter().enumerate() {
        let sub = &e.ops[k];
        if sub.len() != shape.len() {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!(
                "`einsum`: operand {} has rank {} but its subscripts '{}' name {} indices",
                k + 1,
                shape.len(),
                String::from_utf8_lossy(sub),
                sub.len()
            )));
        }
        for (d, &c) in sub.iter().enumerate() {
            match letters.iter().position(|&x| x == c) {
                Some(p) if size[p] != shape[d] => {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`einsum`: index '{}' has length {} in one place and {} in another", c as char, size[p], shape[d])));
                }
                Some(_) => {}
                None => {
                    letters.push(c);
                    size.push(shape[d]);
                }
            }
        }
    }
    let pos = |c: u8| letters.iter().position(|&x| x == c).expect("a letter of the operands");
    let outp: Vec<usize> = e.out.iter().map(|&c| pos(c)).collect();
    let sump: Vec<usize> = (0..letters.len()).filter(|p| !outp.contains(p)).collect();
    // each operand's stride per letter (a letter written twice in one operand
    // walks its diagonal)
    let lstr: Vec<Vec<usize>> = operands
        .iter()
        .enumerate()
        .map(|(k, (shape, _))| {
            let st = strides(shape);
            let mut per = vec![0usize; letters.len()];
            for (d, &c) in e.ops[k].iter().enumerate() {
                per[pos(c)] += st[d];
            }
            per
        })
        .collect();
    let out_shape: Vec<usize> = outp.iter().map(|&p| size[p]).collect();
    limit("einsum", &out_shape)?;
    let nout = size_of(&out_shape);
    let nsum = size_of(&sump.iter().map(|&p| size[p]).collect::<Vec<_>>());
    // each operand's offset is kept as the indices step, an odometer over the
    // output indices and one over the summed ones
    let nops = operands.len();
    let mut out = Vec::with_capacity(nout);
    let mut oi = vec![0usize; outp.len()];
    let mut base = vec![0usize; nops];
    for _ in 0..nout {
        let mut si = vec![0usize; sump.len()];
        let mut off = base.clone();
        let mut acc = 0.0;
        for _ in 0..nsum {
            let mut prod = operands[0].1[off[0]];
            for k in 1..nops {
                prod *= operands[k].1[off[k]];
            }
            acc += prod;
            for d in (0..sump.len()).rev() {
                let p = sump[d];
                si[d] += 1;
                if si[d] < size[p] {
                    for k in 0..nops {
                        off[k] += lstr[k][p];
                    }
                    break;
                }
                si[d] = 0;
                for k in 0..nops {
                    off[k] -= (size[p] - 1) * lstr[k][p];
                }
            }
        }
        out.push(acc);
        for d in (0..outp.len()).rev() {
            let p = outp[d];
            oi[d] += 1;
            if oi[d] < out_shape[d] {
                for k in 0..nops {
                    base[k] += lstr[k][p];
                }
                break;
            }
            oi[d] = 0;
            for k in 0..nops {
                base[k] -= (size[p] - 1) * lstr[k][p];
            }
        }
    }
    Ok((out_shape, out))
}

/// One-sided Jacobi SVD of an m x n matrix with m >= n, lib/svd.viv's `svd`
/// operation for operation: U (m x n), s (descending), V (n x n).
fn svd_tall(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const SWEEPS: usize = 60;
    const TOL: f64 = 1.0e-15;
    let mut w = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    alpha += w[i * n + p] * w[i * n + p];
                    beta += w[i * n + q] * w[i * n + q];
                    gamma += w[i * n + p] * w[i * n + q];
                }
                let scale = (alpha * beta).sqrt();
                if scale > 0.0 {
                    off = crate::vmath::fmax(off, gamma.abs() / scale);
                    if gamma.abs() > TOL * scale {
                        let zeta = (beta - alpha) / (2.0 * gamma);
                        let mut t = 1.0 / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                        if zeta < 0.0 {
                            t = -t;
                        }
                        let c = 1.0 / (1.0 + t * t).sqrt();
                        let sn = c * t;
                        for i in 0..m {
                            let (ap, aq) = (w[i * n + p], w[i * n + q]);
                            w[i * n + p] = c * ap - sn * aq;
                            w[i * n + q] = sn * ap + c * aq;
                        }
                        for i in 0..n {
                            let (vp, vq) = (v[i * n + p], v[i * n + q]);
                            v[i * n + p] = c * vp - sn * vq;
                            v[i * n + q] = sn * vp + c * vq;
                        }
                    }
                }
            }
        }
        if off < TOL {
            break;
        }
    }
    let mut s = vec![0.0; n];
    for j in 0..n {
        let mut total = 0.0;
        for i in 0..m {
            total += w[i * n + j] * w[i * n + j];
        }
        s[j] = total.sqrt();
    }
    for a in 0..n {
        let mut best = a;
        for j in a..n {
            if s[j] > s[best] {
                best = j;
            }
        }
        if best != a {
            s.swap(a, best);
            for i in 0..m {
                w.swap(i * n + a, i * n + best);
            }
            for i in 0..n {
                v.swap(i * n + a, i * n + best);
            }
        }
    }
    let mut u = vec![0.0; m * n];
    for j in 0..n {
        if s[j] > 0.0 {
            for i in 0..m {
                u[i * n + j] = w[i * n + j] / s[j];
            }
        } else {
            u[j * n + j] = 1.0;
        }
    }
    (u, s, v)
}

/// `svd(A)`: U (m x k), s (k), Vt (k x n), k = min(m, n), A = U diag(s) Vt.
pub(crate) fn svd(shape: &[usize], a: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if shape.len() != 2 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`svd` takes a matrix; this array has rank {}", shape.len())));
    }
    let (m, n) = (shape[0], shape[1]);
    if a.iter().any(|x| !x.is_finite()) {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`svd`: the matrix has a NaN or an infinity"));
    }
    if m >= n {
        let (u, s, v) = svd_tall(a, m, n);
        // Vt = V transposed
        let mut vt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vt[i * n + j] = v[j * n + i];
            }
        }
        return Ok((u, s, vt));
    }
    // a wide matrix: A^T = U' S V'^T, so A = V' S U'^T
    let mut at = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            at[j * m + i] = a[i * n + j];
        }
    }
    let (u2, s, v2) = svd_tall(&at, n, m);
    // U = V' (m x m); Vt = U'^T (m x n)
    let mut vt = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            vt[i * n + j] = u2[j * m + i];
        }
    }
    Ok((v2, s, vt))
}

/// A builtin of this file, or `Ok(None)` when `callee` with these arguments is
/// not one (an older builtin, or a program's own def, handles it). `draw`
/// gives the next `rand()`.
pub(super) fn builtin(callee: &str, a: &[Val], draw: &mut dyn FnMut() -> f64) -> Result<Option<Val>, String> {
    // the old one-argument transpose handles a real matrix and a complex one
    if callee == "transpose" && a.len() == 1 {
        return Ok(match &a[0] {
            Val::Arr(x) if x.borrow().shape.len() != 2 => {
                let b = x.borrow();
                let (s, d) = transpose_any(&b.shape, &b.data);
                Some(new(s, d))
            }
            _ => None,
        });
    }
    let Some(sig) = sig(callee, a.len()) else { return Ok(None) };
    // the solvers that return their results are the older builtins with an
    // array they allocate (`Interp::call`, `returning`)
    if matches!(sig.frag, "ar_solve" | "ar_chol") {
        return Ok(None);
    }
    let args = Args { name: sig.name, a };
    let name = sig.name;
    let num = |v: f64| Ok(Some(Val::Num(v)));
    let out = |s: Vec<usize>, d: Vec<f64>| -> Result<Option<Val>, String> { Ok(Some(new(s, d))) };
    match (name, a.len()) {
        ("reshape", _) => {
            let (s, d) = args.arr(0)?;
            let want: Vec<f64> = (1..a.len()).map(|i| args.num(i)).collect::<Result<_, _>>()?;
            let _ = s;
            let shape = reshape(name, d.len(), &want)?;
            out(shape, d)
        }
        ("ravel" | "flatten", 1) => {
            let (_, d) = args.arr(0)?;
            out(vec![d.len()], d)
        }
        ("transpose", 2) => {
            let (s, d) = args.arr(0)?;
            let (ps, pd) = args.arr(1)?;
            if ps.len() != 1 || pd.len() != s.len() {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`transpose`: the axes must be a vector of {} entries, one per axis", s.len())));
            }
            let mut perm = Vec::with_capacity(s.len());
            for &p in &pd {
                let k = axis(name, p, s.len())?;
                if perm.contains(&k) {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`transpose`: axis {k} is named twice")));
                }
                perm.push(k);
            }
            let (s2, d2) = permute(&s, &d, &perm);
            out(s2, d2)
        }
        ("swapaxes", 3) => {
            let (s, d) = args.arr(0)?;
            let (i, j) = (axis(name, args.num(1)?, s.len())?, axis(name, args.num(2)?, s.len())?);
            let mut perm: Vec<usize> = (0..s.len()).collect();
            perm.swap(i, j);
            let (s2, d2) = permute(&s, &d, &perm);
            out(s2, d2)
        }
        ("eye", n) => {
            let rows = dims(&args, 0, 1)?[0];
            let cols = if n >= 2 { dims(&args, 1, 2)?[0] } else { rows };
            let k = if n == 3 { int(args.num(2)?) } else { 0 };
            limit(name, &[rows, cols])?;
            let mut d = vec![0.0; rows * cols];
            for i in 0..rows {
                let j = i as i64 + k;
                if j >= 0 && (j as usize) < cols {
                    d[i * cols + j as usize] = 1.0;
                }
            }
            out(vec![rows, cols], d)
        }
        ("diag", n) => {
            let (s, d) = args.arr(0)?;
            let k = if n == 2 { int(args.num(1)?) } else { 0 };
            match s.len() {
                1 => {
                    let len = s[0];
                    let size = len + k.unsigned_abs() as usize;
                    limit(name, &[size, size])?;
                    let mut r = vec![0.0; size * size];
                    for i in 0..len {
                        let (row, col) = if k >= 0 { (i, i + k as usize) } else { (i + (-k) as usize, i) };
                        r[row * size + col] = d[i];
                    }
                    out(vec![size, size], r)
                }
                2 => {
                    let (rows, cols) = (s[0] as i64, s[1] as i64);
                    let len = if k >= 0 { rows.min(cols - k) } else { (rows + k).min(cols) }.max(0) as usize;
                    let mut r = Vec::with_capacity(len);
                    for i in 0..len {
                        let (row, col) = if k >= 0 { (i, i + k as usize) } else { (i + (-k) as usize, i) };
                        r.push(d[row * s[1] + col]);
                    }
                    out(vec![len], r)
                }
                r => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`diag` takes a vector or a matrix; this array has rank {r}"))),
            }
        }
        ("ones", n) => {
            let d = dims(&args, 0, n)?;
            limit(name, &d)?;
            let n = size_of(&d);
            out(d, vec![1.0; n])
        }
        ("zeros_like" | "ones_like", 1) => {
            let (s, d) = args.arr(0)?;
            let v = if name == "ones_like" { 1.0 } else { 0.0 };
            out(s, vec![v; d.len()])
        }
        ("arange", n) => {
            let (start, stop, step) = match n {
                1 => (0.0, args.num(0)?, 1.0),
                2 => (args.num(0)?, args.num(1)?, 1.0),
                _ => (args.num(0)?, args.num(1)?, args.num(2)?),
            };
            let d = arange(start, stop, step)?;
            out(vec![d.len()], d)
        }
        ("linspace", 4) => {
            let d = linspace(args.num(0)?, args.num(1)?, int(args.num(2)?), args.flag(3)?)?;
            out(vec![d.len()], d)
        }
        ("outer", 2) => {
            let ((_, u), (_, v)) = (args.arr(0)?, args.arr(1)?);
            limit(name, &[u.len(), v.len()])?;
            let mut r = Vec::with_capacity(u.len() * v.len());
            for &x in &u {
                for &y in &v {
                    r.push(x * y);
                }
            }
            out(vec![u.len(), v.len()], r)
        }
        ("dot", 2) => {
            let ((su, u), (sv, v)) = (args.arr(0)?, args.arr(1)?);
            if su.len() != 1 || sv.len() != 1 || u.len() != v.len() {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`dot` takes two vectors of one length (for matrices write A @ B); these have shapes {} and {}", shape_text(&su), shape_text(&sv))));
            }
            if crate::digits::on() {
                return num(crate::digits::dot(&u, &v));
            }
            let mut t = 0.0;
            for i in 0..u.len() {
                t += u[i] * v[i];
            }
            num(t)
        }
        ("__sum_ordered", 1) => {
            let (_, d) = args.arr(0)?;
            if crate::digits::on() {
                return num(crate::digits::sum_ordered(&d));
            }
            let mut t = 0.0;
            for &x in &d {
                t += x;
            }
            num(t)
        }
        ("norm", 1) => {
            let (_, d) = args.arr(0)?;
            let mut t = 0.0;
            for &x in &d {
                t += x * x;
            }
            num(t.sqrt())
        }
        ("trace", n) => {
            let (s, d) = args.arr(0)?;
            if s.len() != 2 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`trace` takes a matrix; this array has rank {}", s.len())));
            }
            let k = if n == 2 { int(args.num(1)?) } else { 0 };
            let (rows, cols) = (s[0] as i64, s[1] as i64);
            let len = if k >= 0 { rows.min(cols - k) } else { (rows + k).min(cols) }.max(0) as usize;
            let mut diag = Vec::with_capacity(len);
            for i in 0..len {
                let (row, col) = if k >= 0 { (i, i + k as usize) } else { (i + (-k) as usize, i) };
                diag.push(d[row * s[1] + col]);
            }
            num(npsum(&diag))
        }
        ("triu" | "tril", n) => {
            let (s, mut d) = args.arr(0)?;
            if s.len() != 2 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` takes a matrix; this array has rank {}", s.len())));
            }
            let k = if n == 2 { int(args.num(1)?) } else { 0 };
            for i in 0..s[0] {
                for j in 0..s[1] {
                    let off = j as i64 - i as i64;
                    let keep = if name == "triu" { off >= k } else { off <= k };
                    if !keep {
                        d[i * s[1] + j] = 0.0;
                    }
                }
            }
            out(s, d)
        }
        ("sum" | "mean" | "var" | "std" | "prod" | "amax" | "amin" | "any" | "all", 1) => {
            let (_, d) = args.arr(0)?;
            let n = d.len() as f64;
            match name {
                "mean" => num(npsum(&d) / n),
                "var" | "std" => {
                    let m = npsum(&d) / n;
                    let sq: Vec<f64> = d.iter().map(|&x| (x - m) * (x - m)).collect();
                    let v = npsum(&sq) / n;
                    num(if name == "std" { v.sqrt() } else { v })
                }
                "prod" => {
                    let mut t = 1.0;
                    for &x in &d {
                        t *= x;
                    }
                    num(t)
                }
                _ => {
                    if d.is_empty() {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` of an empty array")));
                    }
                    let f = if name == "amax" { np_max } else { np_min };
                    num(d[1..].iter().fold(d[0], |m, &x| f(m, x)))
                }
            }
        }
        ("sum" | "mean" | "var" | "std" | "prod" | "amax" | "amin" | "any" | "all", n) => {
            let (s, d) = args.arr(0)?;
            let k = axis(name, args.num(1)?, s.len())?;
            let keep = n == 3 && args.flag(2)?;
            let rs = reduced(name, &s, k, keep)?;
            let along = s[k] as f64;
            let r = match name {
                "sum" => sum_axis(&s, &d, k),
                "mean" => sum_axis(&s, &d, k).into_iter().map(|t| t / along).collect(),
                "var" | "std" => {
                    let m: Vec<f64> = sum_axis(&s, &d, k).into_iter().map(|t| t / along).collect();
                    let mut ms = s.clone();
                    ms[k] = 1;
                    let me = expand(&ms, &m, &s);
                    let sq: Vec<f64> = d.iter().zip(&me).map(|(&x, &y)| (x - y) * (x - y)).collect();
                    sum_axis(&s, &sq, k)
                        .into_iter()
                        .map(|t| if name == "std" { (t / along).sqrt() } else { t / along })
                        .collect()
                }
                "prod" => slabs(&s, &d, k, Some(1.0), |t, x| t * x),
                "any" => slabs(&s, &d, k, Some(0.0), |t, x| if t != 0.0 || x != 0.0 { 1.0 } else { 0.0 }),
                "all" => slabs(&s, &d, k, Some(1.0), |t, x| if t != 0.0 && x != 0.0 { 1.0 } else { 0.0 }),
                _ => {
                    if s[k] == 0 {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` along an axis of length 0")));
                    }
                    slabs(&s, &d, k, None, if name == "amax" { np_max } else { np_min })
                }
            };
            out(rs, r)
        }
        ("argmax" | "argmin", 1) => {
            let (_, d) = args.arr(0)?;
            if d.is_empty() {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` of an empty array")));
            }
            num(arg_ext(&mut d.iter().copied(), name == "argmax"))
        }
        ("argmax" | "argmin", 2) => {
            let (s, d) = args.arr(0)?;
            let k = axis(name, args.num(1)?, s.len())?;
            let rs = reduced(name, &s, k, false)?;
            if s[k] == 0 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` along an axis of length 0")));
            }
            let r = arg_axis(&s, &d, k, name == "argmax");
            out(rs, r)
        }
        ("count_nonzero", 1) => {
            let (_, d) = args.arr(0)?;
            num(d.iter().filter(|&&x| x != 0.0).count() as f64)
        }
        ("argsort", 1) => {
            let (s, d) = args.arr(0)?;
            let len = *s.last().unwrap_or(&0);
            let mut r = Vec::with_capacity(d.len());
            for row in d.chunks(len.max(1)) {
                let mut idx: Vec<usize> = (0..row.len()).collect();
                // NumPy's order: ascending, -0.0 equal to 0.0, NaN after every number; stable
                idx.sort_by(|&i, &j| {
                    let (x, y) = (row[i], row[j]);
                    match (x.is_nan(), y.is_nan()) {
                        (true, true) => std::cmp::Ordering::Equal,
                        (true, false) => std::cmp::Ordering::Greater,
                        (false, true) => std::cmp::Ordering::Less,
                        _ => x.partial_cmp(&y).expect("numbers"),
                    }
                });
                r.extend(idx.iter().map(|&i| i as f64));
            }
            out(s, r)
        }
        ("take", 3) => {
            let (s, d) = args.arr(0)?;
            let (is, id) = args.arr(1)?;
            let k = axis(name, args.num(2)?, s.len())?;
            let (before, along, after) = around(&s, k);
            let mut rs: Vec<usize> = s[..k].to_vec();
            rs.extend_from_slice(&is);
            rs.extend_from_slice(&s[k + 1..]);
            if rs.len() > crate::lower::MAX_RANK {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`take`: the result would have rank {}; arrays have rank {} at most", rs.len(), crate::lower::MAX_RANK)));
            }
            limit(name, &rs)?;
            let mut r = Vec::with_capacity(size_of(&rs));
            for o in 0..before {
                for &f in &id {
                    let j = int(f);
                    if f.is_nan() || j < 0 || j as usize >= along {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`take`: index {} out of bounds for axis {k} (size {along})", super::format_g(f))));
                    }
                    let base = (o * along + j as usize) * after;
                    r.extend_from_slice(&d[base..base + after]);
                }
            }
            out(rs, r)
        }
        ("extract", 2) => {
            let ((ms, m), (s, d)) = (args.arr(0)?, args.arr(1)?);
            if ms != s {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`extract`: array shape mismatch: {} vs {}", shape_text(&ms), shape_text(&s))));
            }
            let r: Vec<f64> = m.iter().zip(&d).filter(|(&c, _)| c != 0.0).map(|(_, &x)| x).collect();
            out(vec![r.len()], r)
        }
        ("putmask", 3) => {
            let Val::Arr(target) = &a[0] else { return Err("`putmask` takes a real array first".into()) };
            let (ms, m) = args.arr(1)?;
            let shape = target.borrow().shape.clone();
            if ms != shape {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`putmask`: array shape mismatch: {} vs {}", shape_text(&shape), shape_text(&ms))));
            }
            // the values are read before the target is written: they may be it
            let vals: Vec<f64> = match &a[2] {
                Val::Num(x) => vec![*x; m.len()],
                Val::Arr(v) => {
                    let vb = v.borrow();
                    if vb.shape != shape {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`putmask`: array shape mismatch: {} vs {}", shape_text(&shape), shape_text(&vb.shape))));
                    }
                    vb.data.clone()
                }
                _ => return Err("`putmask` takes a number or a real array of the target's shape as its values".into()),
            };
            let mut t = target.borrow_mut();
            for (i, &c) in m.iter().enumerate() {
                if c != 0.0 {
                    t.data[i] = vals[i];
                }
            }
            num(0.0)
        }
        ("flatnonzero", 1) => {
            let (_, d) = args.arr(0)?;
            let r: Vec<f64> = d.iter().enumerate().filter(|(_, &x)| x != 0.0).map(|(i, _)| i as f64).collect();
            out(vec![r.len()], r)
        }
        ("cumsum" | "cumprod", n) => {
            let (s, mut d) = args.arr(0)?;
            let is_sum = name == "cumsum";
            if n == 1 {
                let mut t = if is_sum { 0.0 } else { 1.0 };
                for (i, v) in d.iter_mut().enumerate() {
                    t = if i == 0 { *v } else if is_sum { t + *v } else { t * *v };
                    *v = t;
                }
                return out(vec![d.len()], d);
            }
            let k = axis(name, args.num(1)?, s.len())?;
            let (before, along, after) = around(&s, k);
            for o in 0..before {
                for j in 1..along {
                    for i in 0..after {
                        let (p, q) = ((o * along + j - 1) * after + i, (o * along + j) * after + i);
                        d[q] = if is_sum { d[p] + d[q] } else { d[p] * d[q] };
                    }
                }
            }
            out(s, d)
        }
        ("diff", n) => {
            let (mut s, mut d) = args.arr(0)?;
            let times = if n >= 2 { int(args.num(1)?) } else { 1 };
            if times < 0 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`diff`: the order {times} is negative")));
            }
            let k = if n == 3 { axis(name, args.num(2)?, s.len())? } else { s.len() - 1 };
            for _ in 0..times {
                let (before, along, after) = around(&s, k);
                if along == 0 {
                    break;
                }
                let mut r = Vec::with_capacity(before * (along - 1) * after);
                for o in 0..before {
                    for j in 1..along {
                        for i in 0..after {
                            r.push(d[(o * along + j) * after + i] - d[(o * along + j - 1) * after + i]);
                        }
                    }
                }
                s[k] = along - 1;
                d = r;
            }
            out(s, d)
        }
        ("clip", 3) => {
            let (lo, hi) = (args.num(1)?, args.num(2)?);
            match &a[0] {
                Val::Num(x) => num(clip1(*x, lo, hi)),
                Val::Arr(_) => {
                    let (s, d) = args.arr(0)?;
                    out(s, d.into_iter().map(|x| clip1(x, lo, hi)).collect())
                }
                _ => Err("`clip` takes a number or a real array first".into()),
            }
        }
        ("rand", n) => {
            let s = dims(&args, 0, n)?;
            limit(name, &s)?;
            let n = size_of(&s);
            let d: Vec<f64> = (0..n).map(|_| draw()).collect();
            out(s, d)
        }
        ("randn", 0) => {
            let u1 = draw();
            let u2 = draw();
            num(normal(u1, u2))
        }
        ("randn", n) => {
            let s = dims(&args, 0, n)?;
            limit(name, &s)?;
            let n = size_of(&s);
            let mut d = Vec::with_capacity(n);
            for _ in 0..n {
                let u1 = draw();
                let u2 = draw();
                d.push(normal(u1, u2));
            }
            out(s, d)
        }
        ("einsum", n) => {
            let Val::Str(spec) = &a[0] else { return Err("`einsum` takes its subscripts first, as a string".into()) };
            let ops: Vec<Nd> = (1..n).map(|i| args.arr(i)).collect::<Result<_, _>>()?;
            let (s, d) = einsum(&spec.text, &ops)?;
            if s.is_empty() {
                return num(d[0]);
            }
            out(s, d)
        }
        ("svd", 4) => {
            let (s, d) = args.arr(0)?;
            let (u, sv, vt) = svd(&s, &d)?;
            let (m, n) = (s[0], s[1]);
            let k = m.min(n);
            for (i, (want, vals)) in [(vec![m, k], u), (vec![k], sv), (vec![k, n], vt)].into_iter().enumerate() {
                let Val::Arr(t) = &a[i + 1] else { return Err("`svd` fills three real arrays".into()) };
                let mut tb = t.borrow_mut();
                if tb.shape != want {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`svd`: output {} has shape {}; it needs {}", i + 1, shape_text(&tb.shape), shape_text(&want))));
                }
                tb.data.copy_from_slice(&vals);
            }
            num(0.0)
        }
        _ => Ok(None),
    }
}

/// The C runtime's twin of this file (fragment "arrays"), and the fragments
/// for the random draws, einsum, the returning solvers and svd.
pub const C_RUNTIME: &str = r#"/* The array vocabulary (lang/viv/src/interp/arrays.rs): the same loops in
   the same order and the same words as the interpreter's. An error stops
   the program with the interpreter's message and, in a debug build, its
   location. */
#if defined(__GNUC__)
#define VIV_AR_UNUSED __attribute__((unused))
#else
#define VIV_AR_UNUSED
#endif
#if defined(__GNUC__)
__attribute__((noreturn))
#endif
static void viv_ar_fail(int kind, const char* msg, int site) {
    viv_fflush(stdout);
#ifdef VIV_BOUNDS
    viv_errf("error: %s\n%s", msg, viv_sites[site]);
#else
    (void)site;
    viv_errf("error: %s\n", msg);
#endif
    viv_stop(kind);
}
/* a number read as an index or a count, as Rust's `as i64` reads it */
VIV_AR_UNUSED static long long viv_ar_int(double x) {
    return x != x ? 0 : x >= 9223372036854775807.0 ? LLONG_MAX : x < -9223372036854775807.0 ? LLONG_MIN : (long long)x;
}
/* "[2, 3]", as Rust prints a shape */
VIV_AR_UNUSED static void viv_ar_shape(char* buf, size_t cap, int rank, const int* dims) {
    size_t k = (size_t)snprintf(buf, cap, "[");
    for (int d = 0; d < rank && k < cap; d++) k += (size_t)snprintf(buf + k, cap - k, d ? ", %d" : "%d", dims[d]);
    if (k < cap) snprintf(buf + k, cap - k, "]");
}
VIV_AR_UNUSED static void viv_ar_limit(const char* name, int rank, const int* dims, int site) {
    double total = 1.0;
    for (int d = 0; d < rank; d++) total *= (double)dims[d];
    if (total > 2147483647.0) {
        char m[200];
        snprintf(m, sizeof m, "`%s`: an array of %.0f elements exceeds the 2147483647-element limit", name, total);
        viv_ar_fail(VIV_E_OVERFLOW, m, site);
    }
}
VIV_AR_UNUSED static viv_arr viv_ar_new(const char* name, int rank, const int* dims, int site) {
    viv_ar_limit(name, rank, dims, site);
    return viv_arr_new(rank, dims);
}
VIV_AR_UNUSED static int viv_ar_axis(const char* name, double kf, int rank, int site) {
    long long k = viv_ar_int(kf);
    if (k < -rank || k >= rank) {
        char m[160];
        snprintf(m, sizeof m, "`%s`: axis %lld is out of bounds for an array of rank %d", name, k, rank);
        viv_ar_fail(VIV_E_INDEX, m, site);
    }
    return (int)(k < 0 ? k + rank : k);
}
VIV_AR_UNUSED static void viv_ar_around(viv_arr a, int k, long long* before, long long* along, long long* after) {
    *before = 1; *after = 1; *along = a.dims[k];
    for (int d = 0; d < k; d++) *before *= a.dims[d];
    for (int d = k + 1; d < a.rank; d++) *after *= a.dims[d];
}
/* the shape a reduction along axis k leaves */
VIV_AR_UNUSED static int viv_ar_reduced(const char* name, viv_arr a, int k, int keep, int* dims, int site) {
    if (keep) { for (int d = 0; d < a.rank; d++) dims[d] = d == k ? 1 : a.dims[d]; return a.rank; }
    if (a.rank == 1) {
        char m[160];
        snprintf(m, sizeof m, "`%s` along the only axis of a vector gives a number: write `%s(a)`", name, name);
        viv_ar_fail(VIV_E_VALUE, m, site);
    }
    int r = 0;
    for (int d = 0; d < a.rank; d++) if (d != k) dims[r++] = a.dims[d];
    return r;
}
/* NumPy's pairwise sum, and its sum of a contiguous run in blocks of 8192 */
static double viv_ar_pairwise(const double* f, long long n) {
    if (n < 8) {
        double s = 0.0;
        for (long long i = 0; i < n; i++) s += f[i];
        return s;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = f[j];
        long long i = 8;
        for (; i < n - n % 8; i += 8) for (int j = 0; j < 8; j++) r[j] += f[i + j];
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += f[i];
        return s;
    }
    long long n2 = n / 2;
    n2 -= n2 % 8;
    return viv_ar_pairwise(f, n2) + viv_ar_pairwise(f + n2, n - n2);
}
static double viv_ar_npsum(const double* f, long long n) {
    double s = 0.0;
    for (long long b = 0; b < n; b += 8192) s += viv_ar_pairwise(f + b, n - b < 8192 ? n - b : 8192);
    return s;
}
/* sum along axis k in NumPy's order, into r (row-major over the other axes):
   a contiguous run pairwise, otherwise the slabs added in order from 0.0 */
VIV_AR_UNUSED static void viv_ar_sum_axis(viv_arr a, int k, double* r) {
    long long before, along, after;
    viv_ar_around(a, k, &before, &along, &after);
    for (long long o = 0; o < before; o++) {
        if (after == 1) { r[o] = viv_ar_npsum(a.data + o * along, along); continue; }
        double* t = r + o * after;
        for (long long i = 0; i < after; i++) t[i] = 0.0;
        for (long long j = 0; j < along; j++) {
            const double* row = a.data + (o * along + j) * after;
            for (long long i = 0; i < after; i++) t[i] += row[i];
        }
    }
}
/* NumPy's maximum and minimum: a NaN wins, and -0.0 is below 0.0 */
static double viv_ar_npmax(double a, double b) { return (a != a || a > b || (a == b && !signbit(a))) ? a : b; }
static double viv_ar_npmin(double a, double b) { return (a != a || a < b || (a == b && signbit(a))) ? a : b; }
/* the index of the first largest (smallest) entry, or of the first NaN */
VIV_AR_UNUSED static double viv_ar_argext(const double* f, long long n, long long stride, int is_max) {
    double best = f[0];
    long long at = 0;
    if (best != best) return 0.0;
    for (long long j = 1; j < n; j++) {
        double x = f[j * stride];
        if (x != x) return (double)j;
        if ((is_max && x > best) || (!is_max && x < best)) { best = x; at = j; }
    }
    return (double)at;
}
/* axis d of the result is axis perm[d] of a */
VIV_AR_UNUSED static viv_arr viv_ar_permute(viv_arr a, const int* perm) {
    int od[8]; long long st[8], idx[8];
    for (int d = a.rank - 1; d >= 0; d--) st[d] = d == a.rank - 1 ? 1 : st[d + 1] * a.dims[d + 1];
    for (int d = 0; d < a.rank; d++) { od[d] = a.dims[perm[d]]; idx[d] = 0; }
    viv_arr r = viv_arr_new(a.rank, od);
    for (long long q = 0; q < r.size; q++) {
        long long off = 0;
        for (int d = 0; d < a.rank; d++) off += idx[d] * st[perm[d]];
        r.data[q] = a.data[off];
        for (int d = a.rank - 1; d >= 0; d--) { if (++idx[d] < od[d]) break; idx[d] = 0; }
    }
    return r;
}
/* a's entries repeated out to the broadcast shape (rank, to), a row at a time */
VIV_AR_UNUSED static viv_arr viv_ar_expand(viv_arr a, int rank, const int* to) {
    long long st[8], idx[8];
    int lead = rank - a.rank;
    for (int d = a.rank - 1; d >= 0; d--) st[d] = d == a.rank - 1 ? 1 : st[d + 1] * a.dims[d + 1];
    for (int d = 0; d < rank; d++) idx[d] = 0;
    viv_arr r = viv_arr_new(rank, to);
    if (r.size == 0) return r;
    long long last = to[rank - 1], w = 0;
    int whole = a.dims[a.rank - 1] == last;
    for (long long q = 0; q < r.size / last; q++) {
        long long off = 0;
        for (int d = lead; d < rank - 1; d++) if (a.dims[d - lead] != 1) off += idx[d] * st[d - lead];
        if (whole) { memcpy(r.data + w, a.data + off, (size_t)last * sizeof(double)); w += last; }
        else for (long long i = 0; i < last; i++) r.data[w++] = a.data[off];
        for (int d = rank - 2; d >= 0; d--) { if (++idx[d] < to[d]) break; idx[d] = 0; }
    }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_reshape(viv_arr a, int n, const double* want, int site) {
    int out[8], freed = -1;
    char m[240];
    for (int d = 0; d < n; d++) {
        long long k = viv_ar_int(want[d]);
        if (k == -1) {
            if (freed >= 0) viv_ar_fail(VIV_E_VALUE, "`reshape`: only one dimension can be -1", site);
            freed = d; out[d] = 1;
        } else if (k < 0) {
            snprintf(m, sizeof m, "`reshape`: dimension %lld is negative", k);
            viv_ar_fail(VIV_E_VALUE, m, site);
        } else out[d] = (int)k;
    }
    long long known = 1;
    for (int d = 0; d < n; d++) known *= out[d];
    if (freed >= 0) {
        if (known == 0 || a.size % known != 0) {
            int k = snprintf(m, sizeof m, "`reshape`: cannot reshape an array of %d elements into shape [", a.size);
            for (int d = 0; d < n; d++) k += snprintf(m + k, sizeof m - (size_t)k, d ? ", %lld" : "%lld", viv_ar_int(want[d]));
            snprintf(m + k, sizeof m - (size_t)k, "]");
            viv_ar_fail(VIV_E_VALUE, m, site);
        }
        out[freed] = (int)(a.size / known);
        known *= out[freed];
    }
    if (known != a.size) {
        char s[120];
        viv_ar_shape(s, sizeof s, n, out);
        snprintf(m, sizeof m, "`reshape`: cannot reshape an array of %d elements into shape %s", a.size, s);
        viv_ar_fail(VIV_E_VALUE, m, site);
    }
    viv_arr r = viv_arr_new(n, out);
    memcpy(r.data, a.data, (size_t)a.size * sizeof(double));
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_reshape_2(viv_arr a, double d0, int site) { double w[1] = { d0 }; return viv_ar_reshape(a, 1, w, site); }
VIV_AR_UNUSED static viv_arr viv_ar_reshape_3(viv_arr a, double d0, double d1, int site) { double w[2] = { d0, d1 }; return viv_ar_reshape(a, 2, w, site); }
VIV_AR_UNUSED static viv_arr viv_ar_reshape_4(viv_arr a, double d0, double d1, double d2, int site) { double w[3] = { d0, d1, d2 }; return viv_ar_reshape(a, 3, w, site); }
VIV_AR_UNUSED static viv_arr viv_ar_reshape_5(viv_arr a, double d0, double d1, double d2, double d3, int site) { double w[4] = { d0, d1, d2, d3 }; return viv_ar_reshape(a, 4, w, site); }
VIV_AR_UNUSED static viv_arr viv_ar_ravel_1(viv_arr a, int site) {
    (void)site;
    int d[1]; d[0] = a.size;
    viv_arr r = viv_arr_new(1, d);
    memcpy(r.data, a.data, (size_t)a.size * sizeof(double));
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_flatten_1(viv_arr a, int site) { return viv_ar_ravel_1(a, site); }
VIV_AR_UNUSED static viv_arr viv_ar_transpose_2(viv_arr a, viv_arr axes, int site) {
    int perm[8];
    char m[160];
    if (axes.rank != 1 || axes.size != a.rank) {
        snprintf(m, sizeof m, "`transpose`: the axes must be a vector of %d entries, one per axis", a.rank);
        viv_ar_fail(VIV_E_VALUE, m, site);
    }
    for (int d = 0; d < a.rank; d++) {
        perm[d] = viv_ar_axis("transpose", axes.data[d], a.rank, site);
        for (int e = 0; e < d; e++) if (perm[e] == perm[d]) {
            snprintf(m, sizeof m, "`transpose`: axis %d is named twice", perm[d]);
            viv_ar_fail(VIV_E_VALUE, m, site);
        }
    }
    return viv_ar_permute(a, perm);
}
VIV_AR_UNUSED static viv_arr viv_ar_swapaxes_3(viv_arr a, double i, double j, int site) {
    int perm[8];
    int ii = viv_ar_axis("swapaxes", i, a.rank, site), jj = viv_ar_axis("swapaxes", j, a.rank, site);
    for (int d = 0; d < a.rank; d++) perm[d] = d;
    perm[ii] = jj; perm[jj] = ii;
    return viv_ar_permute(a, perm);
}
/* a dimension for a new array, as zeros reads it (a negative one is 0) */
VIV_AR_UNUSED static int viv_ar_dim(double x) { long long k = viv_ar_int(x); return k < 0 ? 0 : k > 2147483647LL ? 2147483647 : (int)k; }
VIV_AR_UNUSED static viv_arr viv_ar_eye(double nf, double mf, double kf, int site) {
    int d[2]; d[0] = viv_ar_dim(nf); d[1] = viv_ar_dim(mf);
    long long k = viv_ar_int(kf);
    viv_arr r = viv_ar_new("eye", 2, d, site);
    for (long long i = 0; i < d[0]; i++) { long long j = i + k; if (j >= 0 && j < d[1]) r.data[i * d[1] + j] = 1.0; }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_eye_1(double n, int site) { return viv_ar_eye(n, n, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_eye_2(double n, double m, int site) { return viv_ar_eye(n, m, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_eye_3(double n, double m, double k, int site) { return viv_ar_eye(n, m, k, site); }
/* the length of diagonal k of a rows x cols matrix, and where entry i is */
VIV_AR_UNUSED static long long viv_ar_diaglen(long long rows, long long cols, long long k) {
    long long len = k >= 0 ? (rows < cols - k ? rows : cols - k) : (rows + k < cols ? rows + k : cols);
    return len < 0 ? 0 : len;
}
VIV_AR_UNUSED static viv_arr viv_ar_diag_2(viv_arr a, double kf, int site) {
    long long k = viv_ar_int(kf);
    char m[120];
    if (a.rank == 1) {
        long long len = a.dims[0], size = len + (k < 0 ? -k : k);
        int d[2]; d[0] = (int)size; d[1] = (int)size;
        viv_arr r = viv_ar_new("diag", 2, d, site);
        for (long long i = 0; i < len; i++) {
            long long row = k >= 0 ? i : i - k, col = k >= 0 ? i + k : i;
            r.data[row * size + col] = a.data[i];
        }
        return r;
    }
    if (a.rank == 2) {
        long long len = viv_ar_diaglen(a.dims[0], a.dims[1], k);
        int d[1]; d[0] = (int)len;
        viv_arr r = viv_arr_new(1, d);
        for (long long i = 0; i < len; i++) {
            long long row = k >= 0 ? i : i - k, col = k >= 0 ? i + k : i;
            r.data[i] = a.data[row * a.dims[1] + col];
        }
        return r;
    }
    snprintf(m, sizeof m, "`diag` takes a vector or a matrix; this array has rank %d", a.rank);
    viv_ar_fail(VIV_E_VALUE, m, site);
    return a;
}
VIV_AR_UNUSED static viv_arr viv_ar_diag_1(viv_arr a, int site) { return viv_ar_diag_2(a, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_filled(const char* name, int rank, const double* dd, double v, int site) {
    int d[4];
    for (int i = 0; i < rank; i++) d[i] = viv_ar_dim(dd[i]);
    viv_arr r = viv_ar_new(name, rank, d, site);
    for (int i = 0; i < r.size; i++) r.data[i] = v;
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_ones_1(double a, int site) { double d[1] = { a }; return viv_ar_filled("ones", 1, d, 1.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_ones_2(double a, double b, int site) { double d[2] = { a, b }; return viv_ar_filled("ones", 2, d, 1.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_ones_3(double a, double b, double c, int site) { double d[3] = { a, b, c }; return viv_ar_filled("ones", 3, d, 1.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_ones_4(double a, double b, double c, double e, int site) { double d[4] = { a, b, c, e }; return viv_ar_filled("ones", 4, d, 1.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_zeros_like_1(viv_arr a, int site) { (void)site; return viv_arr_new(a.rank, a.dims); }
VIV_AR_UNUSED static viv_arr viv_ar_ones_like_1(viv_arr a, int site) {
    (void)site;
    viv_arr r = viv_arr_new_nz(a.rank, a.dims);
    for (int i = 0; i < r.size; i++) r.data[i] = 1.0;
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_arange_3(double start, double stop, double step, int site) {
    char m[200];
    if (step == 0.0) viv_ar_fail(VIV_E_VALUE, "`arange`: the step is zero", site);
    double delta = stop - start, q = delta / step;
    long long len;
    if (q == 0.0 && delta != 0.0) len = signbit(q) ? 0 : 1;
    else {
        if (q != q || ceil(q) > 2147483647.0) {
            snprintf(m, sizeof m, "`arange`: the length %s is not a number of elements below the 2147483647-element limit", viv_g(ceil(q)));
            viv_ar_fail(VIV_E_OVERFLOW, m, site);
        }
        len = (long long)ceil(q);
    }
    int d[1]; d[0] = len <= 0 ? 0 : (int)len;
    viv_arr r = viv_arr_new(1, d);
    if (d[0] > 0) r.data[0] = start;
    if (d[0] > 1) {
        r.data[1] = start + step;
        double dl = r.data[1] - r.data[0];
        for (int i = 2; i < d[0]; i++) r.data[i] = start + (double)i * dl;
    }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_arange_1(double stop, int site) { return viv_ar_arange_3(0.0, stop, 1.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_arange_2(double start, double stop, int site) { return viv_ar_arange_3(start, stop, 1.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_linspace_4(double start, double stop, double nf, double endpoint, int site) {
    char m[160];
    long long n = viv_ar_int(nf);
    if (n < 0) { snprintf(m, sizeof m, "`linspace`: the number of points %lld is negative", n); viv_ar_fail(VIV_E_VALUE, m, site); }
    if (n > 2147483647LL) { snprintf(m, sizeof m, "`linspace`: an array of %lld elements exceeds the 2147483647-element limit", n); viv_ar_fail(VIV_E_OVERFLOW, m, site); }
    int d[1]; d[0] = (int)n;
    viv_arr r = viv_arr_new(1, d);
    double div = endpoint != 0.0 ? (double)n - 1.0 : (double)n, delta = stop - start, step = delta / div;
    for (int i = 0; i < d[0]; i++) {
        double y = (double)i;
        if (div > 0.0) y = step == 0.0 ? y / div * delta : y * step;
        else y = y * delta;
        r.data[i] = y + start;
    }
    if (endpoint != 0.0 && n > 1) r.data[n - 1] = stop;
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_outer_2(viv_arr u, viv_arr v, int site) {
    int d[2]; d[0] = u.size; d[1] = v.size;
    viv_arr r = viv_ar_new("outer", 2, d, site);
    for (int i = 0; i < u.size; i++) for (int j = 0; j < v.size; j++) r.data[(long long)i * v.size + j] = u.data[i] * v.data[j];
    return r;
}
VIV_AR_UNUSED static double viv_ar_dot_2(viv_arr u, viv_arr v, int site) {
    if (u.rank != 1 || v.rank != 1 || u.size != v.size) {
        char su[120], sv[120], m[320];
        viv_ar_shape(su, sizeof su, u.rank, u.dims); viv_ar_shape(sv, sizeof sv, v.rank, v.dims);
        snprintf(m, sizeof m, "`dot` takes two vectors of one length (for matrices write A @ B); these have shapes %s and %s", su, sv);
        viv_ar_fail(VIV_E_VALUE, m, site);
    }
    double t = 0.0;
    for (int i = 0; i < u.size; i++) t += u.data[i] * v.data[i];
    return t;
}
VIV_AR_UNUSED static double viv_ar___sum_ordered_1(viv_arr a, int site) {
    (void)site;
    double t = 0.0;
    for (long long i = 0; i < a.size; i++) t += a.data[i];
    return t;
}
VIV_AR_UNUSED static double viv_ar_norm_1(viv_arr a, int site) {
    (void)site;
    double t = 0.0;
    for (int i = 0; i < a.size; i++) t += a.data[i] * a.data[i];
    return sqrt(t);
}
VIV_AR_UNUSED static double viv_ar_trace_2(viv_arr a, double kf, int site) {
    char m[120];
    if (a.rank != 2) { snprintf(m, sizeof m, "`trace` takes a matrix; this array has rank %d", a.rank); viv_ar_fail(VIV_E_VALUE, m, site); }
    long long k = viv_ar_int(kf), len = viv_ar_diaglen(a.dims[0], a.dims[1], k);
    double* g = (double*)malloc((size_t)(len ? len : 1) * sizeof(double));
    for (long long i = 0; i < len; i++) {
        long long row = k >= 0 ? i : i - k, col = k >= 0 ? i + k : i;
        g[i] = a.data[row * a.dims[1] + col];
    }
    double s = viv_ar_npsum(g, len);
    free(g);
    return s;
}
VIV_AR_UNUSED static double viv_ar_trace_1(viv_arr a, int site) { return viv_ar_trace_2(a, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_tri(const char* name, viv_arr a, double kf, int upper, int site) {
    char m[120];
    if (a.rank != 2) { snprintf(m, sizeof m, "`%s` takes a matrix; this array has rank %d", name, a.rank); viv_ar_fail(VIV_E_VALUE, m, site); }
    long long k = viv_ar_int(kf);
    viv_arr r = viv_arr_new(2, a.dims);
    for (long long i = 0; i < a.dims[0]; i++) for (long long j = 0; j < a.dims[1]; j++) {
        long long off = j - i;
        int keep = upper ? off >= k : off <= k;
        r.data[i * a.dims[1] + j] = keep ? a.data[i * a.dims[1] + j] : 0.0;
    }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_triu_1(viv_arr a, int site) { return viv_ar_tri("triu", a, 0.0, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_triu_2(viv_arr a, double k, int site) { return viv_ar_tri("triu", a, k, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_tril_1(viv_arr a, int site) { return viv_ar_tri("tril", a, 0.0, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_tril_2(viv_arr a, double k, int site) { return viv_ar_tri("tril", a, k, 0, site); }
/* the reductions of a whole array: 0 mean, 1 var, 2 std, 3 prod, 4 amax, 5 amin */
VIV_AR_UNUSED static double viv_ar_whole(const char* name, int op, viv_arr a, int site) {
    long long n = a.size;
    if (op <= 2) {
        double mu = viv_ar_npsum(a.data, n) / (double)n;
        if (op == 0) return mu;
        double* sq = (double*)malloc((size_t)(n ? n : 1) * sizeof(double));
        for (long long i = 0; i < n; i++) sq[i] = (a.data[i] - mu) * (a.data[i] - mu);
        double v = viv_ar_npsum(sq, n) / (double)n;
        free(sq);
        return op == 2 ? sqrt(v) : v;
    }
    if (op == 3) { double t = 1.0; for (long long i = 0; i < n; i++) t *= a.data[i]; return t; }
    if (n == 0) { char m[120]; snprintf(m, sizeof m, "`%s` of an empty array", name); viv_ar_fail(VIV_E_VALUE, m, site); }
    double t = a.data[0];
    for (long long i = 1; i < n; i++) t = op == 4 ? viv_ar_npmax(t, a.data[i]) : viv_ar_npmin(t, a.data[i]);
    return t;
}
VIV_AR_UNUSED static double viv_ar_mean_1(viv_arr a, int site) { return viv_ar_whole("mean", 0, a, site); }
VIV_AR_UNUSED static double viv_ar_var_1(viv_arr a, int site) { return viv_ar_whole("var", 1, a, site); }
VIV_AR_UNUSED static double viv_ar_std_1(viv_arr a, int site) { return viv_ar_whole("std", 2, a, site); }
VIV_AR_UNUSED static double viv_ar_prod_1(viv_arr a, int site) { return viv_ar_whole("prod", 3, a, site); }
VIV_AR_UNUSED static double viv_ar_amax_1(viv_arr a, int site) { return viv_ar_whole("amax", 4, a, site); }
VIV_AR_UNUSED static double viv_ar_amin_1(viv_arr a, int site) { return viv_ar_whole("amin", 5, a, site); }
/* the reductions along an axis: 0 sum, 1 mean, 2 var, 3 std, 4 prod, 5 amax, 6 amin, 7 any, 8 all */
VIV_AR_UNUSED static viv_arr viv_ar_axred(const char* name, int op, viv_arr a, double kf, double keepf, int site) {
    int k = viv_ar_axis(name, kf, a.rank, site), rd[8];
    int rr = viv_ar_reduced(name, a, k, keepf != 0.0, rd, site);
    long long before, along, after;
    viv_ar_around(a, k, &before, &along, &after);
    if ((op == 5 || op == 6) && along == 0) { char m[120]; snprintf(m, sizeof m, "`%s` along an axis of length 0", name); viv_ar_fail(VIV_E_VALUE, m, site); }
    viv_arr r = viv_arr_new(rr, rd);
    if (op <= 3) {
        viv_ar_sum_axis(a, k, r.data);
        for (int i = 0; i < r.size; i++) r.data[i] = r.data[i] / (double)along;
        if (op == 0) viv_ar_sum_axis(a, k, r.data);
        if (op >= 2) {
            double* sq = (double*)malloc((size_t)(a.size ? a.size : 1) * sizeof(double));
            for (long long o = 0; o < before; o++) for (long long j = 0; j < along; j++) for (long long i = 0; i < after; i++) {
                long long q = (o * along + j) * after + i;
                double y = r.data[o * after + i];
                sq[q] = (a.data[q] - y) * (a.data[q] - y);
            }
            viv_arr t = a; t.data = sq;
            viv_ar_sum_axis(t, k, r.data);
            for (int i = 0; i < r.size; i++) r.data[i] = op == 3 ? sqrt(r.data[i] / (double)along) : r.data[i] / (double)along;
            free(sq);
        }
        return r;
    }
    /* the slabs combined in order, entry by entry (the interpreter's `slabs`) */
    for (long long o = 0; o < before; o++) {
        double* t = r.data + o * after;
        long long first = 0;
        if (op == 5 || op == 6) { memcpy(t, a.data + o * along * after, (size_t)after * sizeof(double)); first = 1; }
        else for (long long i = 0; i < after; i++) t[i] = (op == 4 || op == 8) ? 1.0 : 0.0;
        for (long long j = first; j < along; j++) {
            const double* row = a.data + (o * along + j) * after;
            for (long long i = 0; i < after; i++) {
                double x = row[i];
                if (op == 4) t[i] = t[i] * x;
                else if (op == 7) t[i] = (t[i] != 0.0 || x != 0.0) ? 1.0 : 0.0;
                else if (op == 8) t[i] = (t[i] != 0.0 && x != 0.0) ? 1.0 : 0.0;
                else t[i] = op == 5 ? viv_ar_npmax(t[i], x) : viv_ar_npmin(t[i], x);
            }
        }
    }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_sum_2(viv_arr a, double k, int site) { return viv_ar_axred("sum", 0, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_sum_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("sum", 0, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_mean_2(viv_arr a, double k, int site) { return viv_ar_axred("mean", 1, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_mean_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("mean", 1, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_var_2(viv_arr a, double k, int site) { return viv_ar_axred("var", 2, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_var_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("var", 2, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_std_2(viv_arr a, double k, int site) { return viv_ar_axred("std", 3, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_std_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("std", 3, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_prod_2(viv_arr a, double k, int site) { return viv_ar_axred("prod", 4, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_prod_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("prod", 4, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_amax_2(viv_arr a, double k, int site) { return viv_ar_axred("amax", 5, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_amax_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("amax", 5, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_amin_2(viv_arr a, double k, int site) { return viv_ar_axred("amin", 6, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_amin_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("amin", 6, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_any_2(viv_arr a, double k, int site) { return viv_ar_axred("any", 7, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_any_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("any", 7, a, k, keep, site); }
VIV_AR_UNUSED static viv_arr viv_ar_all_2(viv_arr a, double k, int site) { return viv_ar_axred("all", 8, a, k, 0.0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_all_3(viv_arr a, double k, double keep, int site) { return viv_ar_axred("all", 8, a, k, keep, site); }
VIV_AR_UNUSED static double viv_ar_argext_1(const char* name, viv_arr a, int is_max, int site) {
    if (a.size == 0) { char m[120]; snprintf(m, sizeof m, "`%s` of an empty array", name); viv_ar_fail(VIV_E_VALUE, m, site); }
    return viv_ar_argext(a.data, a.size, 1, is_max);
}
VIV_AR_UNUSED static double viv_ar_argmax_1(viv_arr a, int site) { return viv_ar_argext_1("argmax", a, 1, site); }
VIV_AR_UNUSED static double viv_ar_argmin_1(viv_arr a, int site) { return viv_ar_argext_1("argmin", a, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_argext_2(const char* name, viv_arr a, double kf, int is_max, int site) {
    int k = viv_ar_axis(name, kf, a.rank, site), rd[8];
    int rr = viv_ar_reduced(name, a, k, 0, rd, site);
    long long before, along, after, w = 0;
    viv_ar_around(a, k, &before, &along, &after);
    if (along == 0) { char m[120]; snprintf(m, sizeof m, "`%s` along an axis of length 0", name); viv_ar_fail(VIV_E_VALUE, m, site); }
    viv_arr r = viv_arr_new(rr, rd);
    double* best = (double*)malloc((size_t)(after ? after : 1) * sizeof(double));
    char* nan = (char*)malloc((size_t)(after ? after : 1));
    (void)w;
    for (long long o = 0; o < before; o++) {
        double* at = r.data + o * after;
        for (long long i = 0; i < after; i++) { best[i] = a.data[o * along * after + i]; nan[i] = best[i] != best[i]; }
        for (long long j = 1; j < along; j++) {
            const double* row = a.data + (o * along + j) * after;
            for (long long i = 0; i < after; i++) {
                double x = row[i];
                if (nan[i]) continue;
                if (x != x) { nan[i] = 1; at[i] = (double)j; }
                else if ((is_max && x > best[i]) || (!is_max && x < best[i])) { best[i] = x; at[i] = (double)j; }
            }
        }
    }
    free(best); free(nan);
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_argmax_2(viv_arr a, double k, int site) { return viv_ar_argext_2("argmax", a, k, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_argmin_2(viv_arr a, double k, int site) { return viv_ar_argext_2("argmin", a, k, 0, site); }
VIV_AR_UNUSED static double viv_ar_count_nonzero_1(viv_arr a, int site) {
    (void)site;
    double c = 0.0;
    for (int i = 0; i < a.size; i++) if (a.data[i] != 0.0) c += 1.0;
    return c;
}
/* i strictly before j in NumPy's order: ascending, NaN after every number */
static int viv_ar_before(double x, double y) {
    if (x != x) return 0;
    if (y != y) return 1;
    return x < y;
}
VIV_AR_UNUSED static viv_arr viv_ar_argsort_1(viv_arr a, int site) {
    (void)site;
    viv_arr r = viv_arr_new(a.rank, a.dims);
    long long len = a.rank ? a.dims[a.rank - 1] : 0;
    if (len == 0 || a.size == 0) return r;
    long long* src = (long long*)malloc((size_t)len * sizeof(long long));
    long long* dst = (long long*)malloc((size_t)len * sizeof(long long));
    for (long long row = 0; row < a.size / len; row++) {
        const double* f = a.data + row * len;
        for (long long i = 0; i < len; i++) src[i] = i;
        /* a bottom-up merge sort: stable, as Rust's sort_by */
        for (long long w = 1; w < len; w *= 2) {
            for (long long lo = 0; lo < len; lo += 2 * w) {
                long long mid = lo + w < len ? lo + w : len, hi = lo + 2 * w < len ? lo + 2 * w : len;
                long long i = lo, j = mid, k = lo;
                while (i < mid && j < hi) dst[k++] = viv_ar_before(f[src[j]], f[src[i]]) ? src[j++] : src[i++];
                while (i < mid) dst[k++] = src[i++];
                while (j < hi) dst[k++] = src[j++];
            }
            long long* t = src; src = dst; dst = t;
        }
        for (long long i = 0; i < len; i++) r.data[row * len + i] = (double)src[i];
    }
    free(src); free(dst);
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_take_3(viv_arr a, viv_arr ix, double kf, int site) {
    int k = viv_ar_axis("take", kf, a.rank, site), rd[8], rr = 0;
    char m[200];
    long long before, along, after;
    viv_ar_around(a, k, &before, &along, &after);
    for (int d = 0; d < k; d++) rd[rr++] = a.dims[d];
    int rank = k + ix.rank + a.rank - k - 1;
    if (rank > 4) { snprintf(m, sizeof m, "`take`: the result would have rank %d; arrays have rank 4 at most", rank); viv_ar_fail(VIV_E_VALUE, m, site); }
    for (int d = 0; d < ix.rank; d++) rd[rr++] = ix.dims[d];
    for (int d = k + 1; d < a.rank; d++) rd[rr++] = a.dims[d];
    viv_arr r = viv_ar_new("take", rr, rd, site);
    long long w = 0;
    for (long long o = 0; o < before; o++) for (int q = 0; q < ix.size; q++) {
        double f = ix.data[q];
        long long j = viv_ar_int(f);
        if (f != f || j < 0 || j >= along) {
            snprintf(m, sizeof m, "`take`: index %s out of bounds for axis %d (size %lld)", viv_g(f), k, along);
            viv_ar_fail(VIV_E_INDEX, m, site);
        }
        memcpy(r.data + w, a.data + (o * along + j) * after, (size_t)after * sizeof(double));
        w += after;
    }
    return r;
}
VIV_AR_UNUSED static void viv_ar_same(const char* name, viv_arr a, viv_arr b, int site) {
    int same = a.rank == b.rank;
    for (int d = 0; same && d < a.rank; d++) same = a.dims[d] == b.dims[d];
    if (!same) {
        char sa[120], sb[120], m[320];
        viv_ar_shape(sa, sizeof sa, a.rank, a.dims); viv_ar_shape(sb, sizeof sb, b.rank, b.dims);
        snprintf(m, sizeof m, "`%s`: array shape mismatch: %s vs %s", name, sa, sb);
        viv_ar_fail(VIV_E_VALUE, m, site);
    }
}
VIV_AR_UNUSED static viv_arr viv_ar_extract_2(viv_arr mask, viv_arr a, int site) {
    viv_ar_same("extract", mask, a, site);
    int d[1]; d[0] = 0;
    for (int i = 0; i < a.size; i++) if (mask.data[i] != 0.0) d[0]++;
    viv_arr r = viv_arr_new(1, d);
    int w = 0;
    for (int i = 0; i < a.size; i++) if (mask.data[i] != 0.0) r.data[w++] = a.data[i];
    return r;
}
VIV_AR_UNUSED static double viv_ar_putmask_3n(viv_arr a, viv_arr mask, double v, int site) {
    viv_ar_same("putmask", a, mask, site);
    for (int i = 0; i < a.size; i++) if (mask.data[i] != 0.0) a.data[i] = v;
    return 0.0;
}
VIV_AR_UNUSED static double viv_ar_putmask_3(viv_arr a, viv_arr mask, viv_arr v, int site) {
    viv_ar_same("putmask", a, mask, site);
    viv_ar_same("putmask", a, v, site);
    /* the values are read before the target is written: they may be it */
    double* tmp = (double*)malloc((size_t)(a.size ? a.size : 1) * sizeof(double));
    memcpy(tmp, v.data, (size_t)a.size * sizeof(double));
    for (int i = 0; i < a.size; i++) if (mask.data[i] != 0.0) a.data[i] = tmp[i];
    free(tmp);
    return 0.0;
}
VIV_AR_UNUSED static viv_arr viv_ar_flatnonzero_1(viv_arr a, int site) {
    (void)site;
    int d[1]; d[0] = 0;
    for (int i = 0; i < a.size; i++) if (a.data[i] != 0.0) d[0]++;
    viv_arr r = viv_arr_new(1, d);
    int w = 0;
    for (int i = 0; i < a.size; i++) if (a.data[i] != 0.0) r.data[w++] = (double)i;
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_scan(const char* name, viv_arr a, int flat, double kf, int is_sum, int site) {
    viv_arr r;
    if (flat) {
        int d[1]; d[0] = a.size;
        r = viv_arr_new(1, d);
        double t = 0.0;
        for (int i = 0; i < a.size; i++) { t = i == 0 ? a.data[i] : is_sum ? t + a.data[i] : t * a.data[i]; r.data[i] = t; }
        return r;
    }
    int k = viv_ar_axis(name, kf, a.rank, site);
    long long before, along, after;
    viv_ar_around(a, k, &before, &along, &after);
    r = viv_arr_new(a.rank, a.dims);
    memcpy(r.data, a.data, (size_t)a.size * sizeof(double));
    for (long long o = 0; o < before; o++) for (long long j = 1; j < along; j++) for (long long i = 0; i < after; i++) {
        long long p = (o * along + j - 1) * after + i, q = (o * along + j) * after + i;
        r.data[q] = is_sum ? r.data[p] + r.data[q] : r.data[p] * r.data[q];
    }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_cumsum_1(viv_arr a, int site) { return viv_ar_scan("cumsum", a, 1, 0.0, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_cumsum_2(viv_arr a, double k, int site) { return viv_ar_scan("cumsum", a, 0, k, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_cumprod_1(viv_arr a, int site) { return viv_ar_scan("cumprod", a, 1, 0.0, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_cumprod_2(viv_arr a, double k, int site) { return viv_ar_scan("cumprod", a, 0, k, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_diff(viv_arr a, double nf, int k, int site) {
    long long times = viv_ar_int(nf);
    if (times < 0) { char m[120]; snprintf(m, sizeof m, "`diff`: the order %lld is negative", times); viv_ar_fail(VIV_E_VALUE, m, site); }
    viv_arr cur = viv_arr_new(a.rank, a.dims);
    memcpy(cur.data, a.data, (size_t)a.size * sizeof(double));
    for (long long t = 0; t < times; t++) {
        long long before, along, after, w = 0;
        viv_ar_around(cur, k, &before, &along, &after);
        if (along == 0) break;
        int d[8];
        for (int e = 0; e < cur.rank; e++) d[e] = e == k ? (int)along - 1 : cur.dims[e];
        viv_arr r = viv_arr_new(cur.rank, d);
        for (long long o = 0; o < before; o++) for (long long j = 1; j < along; j++) for (long long i = 0; i < after; i++)
            r.data[w++] = cur.data[(o * along + j) * after + i] - cur.data[(o * along + j - 1) * after + i];
        cur = r;
    }
    return cur;
}
VIV_AR_UNUSED static viv_arr viv_ar_diff_1(viv_arr a, int site) { return viv_ar_diff(a, 1.0, a.rank - 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_diff_2(viv_arr a, double n, int site) { return viv_ar_diff(a, n, a.rank - 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_diff_3(viv_arr a, double n, double k, int site) {
    long long t = viv_ar_int(n);
    if (t < 0) { char m[120]; snprintf(m, sizeof m, "`diff`: the order %lld is negative", t); viv_ar_fail(VIV_E_VALUE, m, site); }
    return viv_ar_diff(a, n, viv_ar_axis("diff", k, a.rank, site), site);
}
/* NumPy's clip of one entry: a NaN entry stays NaN, a NaN bound is no bound */
static double viv_ar_clip1(double x, double lo, double hi) {
    if (x != x) return x;
    double m = (lo != lo || x > lo) ? x : lo;
    /* a bound equal to the entry wins, so clip(0.0, -0.0, 1.0) is -0.0.
       Clang at -O2 on arm64 turns the select above into fmaxnm when the
       bound is a known constant, and fmaxnm returns +0.0 for that tie;
       these lines restore the bound's zero */
    if (m == 0.0 && lo == 0.0) m = lo;
    double r = (hi != hi || m < hi) ? m : hi;
    if (r == 0.0 && hi == 0.0) r = hi;
    return r;
}
VIV_AR_UNUSED static double viv_ar_clip_3n(double x, double lo, double hi, int site) { (void)site; return viv_ar_clip1(x, lo, hi); }
VIV_AR_UNUSED static viv_arr viv_ar_clip_3(viv_arr a, double lo, double hi, int site) {
    (void)site;
    viv_arr r = viv_arr_new(a.rank, a.dims);
    for (int i = 0; i < a.size; i++) r.data[i] = viv_ar_clip1(a.data[i], lo, hi);
    return r;
}"#;


/// The adjoint of an operand that was broadcast (fragment "ad_unb"): `g`
/// summed down to `like`'s shape, `unbroadcast`'s twin.
pub const C_AD_UNB: &str = r#"/* the adjoint of an operand that was broadcast (autodiff): g summed down to like's shape */
static viv_arr viv_ad_unb_aa(viv_arr g, viv_arr like) {
    int same = g.rank == like.rank;
    for (int d = 0; same && d < g.rank; d++) same = g.dims[d] == like.dims[d];
    if (same) return g;
    long long st[8], idx[8];
    int lead = g.rank - like.rank;
    for (int d = like.rank - 1; d >= 0; d--) st[d] = d == like.rank - 1 ? 1 : st[d + 1] * like.dims[d + 1];
    for (int d = 0; d < g.rank; d++) idx[d] = 0;
    viv_arr r = viv_arr_new(like.rank, like.dims);
    if (g.size == 0 || g.rank == 0) return r;
    /* a row of g's last axis at a time, in g's order, so each entry of r
       takes its terms in the order the element-by-element walk gave them:
       along the row when like has that axis, else all into one entry */
    long long last = g.dims[g.rank - 1];
    int along = like.rank > 0 && like.dims[like.rank - 1] != 1;
    for (long long q = 0; q < g.size; q += last) {
        long long off = 0;
        for (int d = lead; d < g.rank - 1; d++) if (like.dims[d - lead] != 1) off += idx[d] * st[d - lead];
        const double* gp = g.data + q;
        double* rp = r.data + off;
        if (along) for (long long j = 0; j < last; j++) rp[j] += gp[j];
        else for (long long j = 0; j < last; j++) rp[0] += gp[j];
        for (int d = g.rank - 2; d >= 0; d--) { if (++idx[d] < g.dims[d]) break; idx[d] = 0; }
    }
    return r;
}"#;

/// The adjoints of the vocabulary's reductions, reshapes and `take` along an
/// axis (fragment "ad_arr"): `ad_keep_shape`, `ad_prod_others` and
/// `ad_take_axis`'s twins, loop for loop.
pub const C_AD_ARR: &str = r#"/* autodiff: the adjoints of reductions along an axis, reshapes and take along an axis */
static void viv_ad_arr_fail(int kind, const char* msg) { viv_fflush(stdout); viv_errf("error: autodiff: %s\n", msg); viv_stop_exit(kind); }
static int viv_ad_axis(double kf, int rank) {
    int k = (int)kf;
    if (k < -rank || k >= rank) viv_ad_arr_fail(VIV_E_INDEX, "an axis out of bounds in a reduction's adjoint");
    return k < 0 ? k + rank : k;
}
/* g's entries in the shape a has after a reduction along k with keepdims */
static viv_arr viv_ad_keep(viv_arr g, viv_arr a, double kf) {
    int k = viv_ad_axis(kf, a.rank), d[8];
    for (int i = 0; i < a.rank; i++) d[i] = a.dims[i];
    d[k] = 1;
    viv_arr r = viv_arr_new(a.rank, d);
    if (r.size != g.size) viv_ad_arr_fail(VIV_E_ERROR, "a reduction's adjoint has the wrong number of entries");
    for (int i = 0; i < r.size; i++) r.data[i] = g.data[i];
    return r;
}
/* g's entries in a's shape: the adjoint of a reshape */
static viv_arr viv_ad_like(viv_arr g, viv_arr a) {
    viv_arr r = viv_arr_new(a.rank, a.dims);
    if (r.size != g.size) viv_ad_arr_fail(VIV_E_ERROR, "a reshape's adjoint has the wrong number of entries");
    for (int i = 0; i < r.size; i++) r.data[i] = g.data[i];
    return r;
}
/* each entry: the product of the others along the axis, p[j] * s[j] */
static viv_arr viv_ad_prod_lanes(viv_arr a, long long before, long long along, long long after) {
    viv_arr r = viv_arr_new(a.rank, a.dims);
    for (long long o = 0; o < before; o++) for (long long i = 0; i < after; i++) {
        double p = 1.0, q = 1.0;
        for (long long j = 0; j < along; j++) { long long at = (o * along + j) * after + i; r.data[at] = p; p *= a.data[at]; }
        for (long long j = along - 1; j >= 0; j--) { long long at = (o * along + j) * after + i; r.data[at] *= q; q *= a.data[at]; }
    }
    return r;
}
static viv_arr viv_ad_prod(viv_arr a) { return viv_ad_prod_lanes(a, 1, a.size, 1); }
static viv_arr viv_ad_prodax(viv_arr a, double kf) {
    int k = viv_ad_axis(kf, a.rank);
    long long before = 1, after = 1;
    for (int d = 0; d < k; d++) before *= a.dims[d];
    for (int d = k + 1; d < a.rank; d++) after *= a.dims[d];
    return viv_ad_prod_lanes(a, before, a.dims[k], after);
}
/* the adjoint of take(a, idx, k): each slice of g added back where it was read */
static viv_arr viv_ad_takeax(viv_arr g, viv_arr a, viv_arr ix, double kf) {
    int k = viv_ad_axis(kf, a.rank);
    long long before = 1, after = 1, along = a.dims[k];
    for (int d = 0; d < k; d++) before *= a.dims[d];
    for (int d = k + 1; d < a.rank; d++) after *= a.dims[d];
    if ((long long)g.size != before * ix.size * after) viv_ad_arr_fail(VIV_E_ERROR, "take's adjoint has the wrong number of entries");
    viv_arr r = viv_arr_new(a.rank, a.dims);
    long long w = 0;
    for (long long o = 0; o < before; o++) for (int q = 0; q < ix.size; q++) {
        double f = ix.data[q];
        long long j = (long long)f;
        if (f != f || j < 0 || j >= along) {
            viv_fflush(stdout);
            viv_errf("error: `take`: index %s out of bounds for axis %d (size %lld)\n", viv_g(f), k, along);
            viv_stop_exit(VIV_E_INDEX);
        }
        long long base = (o * along + j) * after;
        for (long long t = 0; t < after; t++) r.data[base + t] += g.data[w + t];
        w += after;
    }
    return r;
}"#;

/// Shaped draws from `rand()`'s generator (fragment "ar_rand"): entries in
/// row-major order; `randn` takes two uniforms per entry, lib/random's formula.
pub const C_RAND: &str = r#"static double viv_ar_normal(void) {
    double u1 = viv_rand();
    double u2 = viv_rand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2);
}
VIV_AR_UNUSED static viv_arr viv_ar_draws(int rank, const double* dd, int normal, int site) {
    int d[4];
    for (int i = 0; i < rank; i++) d[i] = viv_ar_dim(dd[i]);
    viv_arr r = viv_ar_new(normal ? "randn" : "rand", rank, d, site);
    for (int i = 0; i < r.size; i++) r.data[i] = normal ? viv_ar_normal() : viv_rand();
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_rand_1(double a, int site) { double d[1] = { a }; return viv_ar_draws(1, d, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_rand_2(double a, double b, int site) { double d[2] = { a, b }; return viv_ar_draws(2, d, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_rand_3(double a, double b, double c, int site) { double d[3] = { a, b, c }; return viv_ar_draws(3, d, 0, site); }
VIV_AR_UNUSED static viv_arr viv_ar_rand_4(double a, double b, double c, double e, int site) { double d[4] = { a, b, c, e }; return viv_ar_draws(4, d, 0, site); }
VIV_AR_UNUSED static double viv_ar_randn_0(int site) { (void)site; return viv_ar_normal(); }
VIV_AR_UNUSED static viv_arr viv_ar_randn_1(double a, int site) { double d[1] = { a }; return viv_ar_draws(1, d, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_randn_2(double a, double b, int site) { double d[2] = { a, b }; return viv_ar_draws(2, d, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_randn_3(double a, double b, double c, int site) { double d[3] = { a, b, c }; return viv_ar_draws(3, d, 1, site); }
VIV_AR_UNUSED static viv_arr viv_ar_randn_4(double a, double b, double c, double e, int site) { double d[4] = { a, b, c, e }; return viv_ar_draws(4, d, 1, site); }"#;

/// `einsum` (fragment "ar_einsum"): the interpreter's parse, checks and loops.
pub const C_EINSUM: &str = r#"static int viv_ar_letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static viv_arr viv_ar_einsum(const char* spec, int nops, viv_arr* op, int site) {
    char t[256], m[300];
    int tl = 0;
    for (const char* p = spec; *p && tl < 255; p++) if (*p != ' ') t[tl++] = *p;
    t[tl] = 0;
    char* arrow = strstr(t, "->");
    int inl = arrow ? (int)(arrow - t) : tl;
    char sub[4][16]; int subn[4], nsub = 0, cur = 0;
    subn[0] = 0;
    for (int i = 0; i <= inl; i++) {
        if (i == inl || t[i] == ',') { if (nsub < 4) subn[nsub] = cur; nsub++; cur = 0; continue; }
        if (nsub < 4 && cur < 15) sub[nsub][cur] = t[i];
        cur++;
    }
    if (nsub != nops) { snprintf(m, sizeof m, "`einsum`: the subscripts name %d operands and the call passes %d", nsub, nops); viv_ar_fail(VIV_E_VALUE, m, site); }
    for (int k = 0; k < nops; k++) for (int d = 0; d < subn[k] && d < 15; d++)
        if (!viv_ar_letter(sub[k][d])) { snprintf(m, sizeof m, "`einsum`: a subscript is a letter, not '%c'", sub[k][d]); viv_ar_fail(VIV_E_VALUE, m, site); }
    char out[16]; int nout = 0;
    #define VIV_AR_COUNT(c, n) do { n = 0; for (int k_ = 0; k_ < nops; k_++) for (int d_ = 0; d_ < subn[k_]; d_++) if (sub[k_][d_] == (c)) n++; } while (0)
    if (arrow) {
        for (const char* p = arrow + 2; *p; p++) {
            int cnt;
            if (!viv_ar_letter(*p)) { snprintf(m, sizeof m, "`einsum`: a subscript is a letter, not '%c'", *p); viv_ar_fail(VIV_E_VALUE, m, site); }
            for (int e = 0; e < nout; e++) if (out[e] == *p) { snprintf(m, sizeof m, "`einsum`: the output names '%c' twice", *p); viv_ar_fail(VIV_E_VALUE, m, site); }
            VIV_AR_COUNT(*p, cnt);
            if (cnt == 0) { snprintf(m, sizeof m, "`einsum`: the output's '%c' is in no operand", *p); viv_ar_fail(VIV_E_VALUE, m, site); }
            if (nout < 15) out[nout] = *p;
            nout++;
        }
    } else {
        for (int c = 'A'; c <= 'z'; c++) { int cnt; if (!viv_ar_letter((char)c)) continue; VIV_AR_COUNT(c, cnt); if (cnt == 1 && nout < 15) out[nout++] = (char)c; }
    }
    #undef VIV_AR_COUNT
    if (nout > 4) { snprintf(m, sizeof m, "`einsum`: the output has rank %d; arrays have rank 4 at most", nout); viv_ar_fail(VIV_E_VALUE, m, site); }
    char let[64]; long long size[64]; int nl = 0;
    for (int k = 0; k < nops; k++) {
        if (subn[k] != op[k].rank) {
            char s[20]; int n = subn[k] < 15 ? subn[k] : 15;
            memcpy(s, sub[k], (size_t)n); s[n] = 0;
            snprintf(m, sizeof m, "`einsum`: operand %d has rank %d but its subscripts '%s' name %d indices", k + 1, op[k].rank, s, subn[k]);
            viv_ar_fail(VIV_E_VALUE, m, site);
        }
        for (int d = 0; d < subn[k]; d++) {
            int p = -1;
            for (int e = 0; e < nl; e++) if (let[e] == sub[k][d]) p = e;
            if (p >= 0 && size[p] != op[k].dims[d]) {
                snprintf(m, sizeof m, "`einsum`: index '%c' has length %lld in one place and %d in another", sub[k][d], size[p], op[k].dims[d]);
                viv_ar_fail(VIV_E_VALUE, m, site);
            }
            if (p < 0) { let[nl] = sub[k][d]; size[nl] = op[k].dims[d]; nl++; }
        }
    }
    int outp[4], sump[64], ns = 0;
    for (int e = 0; e < nout; e++) for (int p = 0; p < nl; p++) if (let[p] == out[e]) outp[e] = p;
    for (int p = 0; p < nl; p++) { int in_out = 0; for (int e = 0; e < nout; e++) if (outp[e] == p) in_out = 1; if (!in_out) sump[ns++] = p; }
    long long lstr[4][64];
    for (int k = 0; k < nops; k++) {
        long long st[8];
        for (int d = op[k].rank - 1; d >= 0; d--) st[d] = d == op[k].rank - 1 ? 1 : st[d + 1] * op[k].dims[d + 1];
        for (int p = 0; p < nl; p++) lstr[k][p] = 0;
        for (int d = 0; d < subn[k]; d++) for (int p = 0; p < nl; p++) if (let[p] == sub[k][d]) lstr[k][p] += st[d];
    }
    int od[4]; long long nsum = 1;
    for (int e = 0; e < nout; e++) od[e] = (int)size[outp[e]];
    for (int e = 0; e < ns; e++) nsum *= size[sump[e]];
    viv_arr r;
    if (nout == 0) { int one[1] = { 1 }; r = viv_arr_new(1, one); }
    else r = viv_ar_new("einsum", nout, od, site);
    /* each operand's offset kept as the indices step (the interpreter's odometers) */
    long long oi[4] = { 0, 0, 0, 0 }, si[64], base[4] = { 0, 0, 0, 0 }, off[4];
    for (long long q = 0; q < r.size; q++) {
        for (int e = 0; e < ns; e++) si[e] = 0;
        for (int k = 0; k < nops; k++) off[k] = base[k];
        double acc = 0.0;
        /* the last summed index runs innermost, in a loop of its own */
        long long inner = ns ? size[sump[ns - 1]] : 1, is0 = ns ? lstr[0][sump[ns - 1]] : 0, is1 = ns && nops > 1 ? lstr[1][sump[ns - 1]] : 0, is2 = ns && nops > 2 ? lstr[2][sump[ns - 1]] : 0;
        for (long long s = 0; s < nsum; s += inner) {
            const double* p0 = op[0].data + off[0];
            if (nops == 1) for (long long t = 0; t < inner; t++) acc += p0[t * is0];
            else if (nops == 2) { const double* p1 = op[1].data + off[1]; for (long long t = 0; t < inner; t++) acc += p0[t * is0] * p1[t * is1]; }
            else { const double* p1 = op[1].data + off[1]; const double* p2 = op[2].data + off[2]; for (long long t = 0; t < inner; t++) acc += p0[t * is0] * p1[t * is1] * p2[t * is2]; }
            if (ns) si[ns - 1] = inner - 1;
            if (ns) for (int k = 0; k < nops; k++) off[k] += (inner - 1) * lstr[k][sump[ns - 1]];
            for (int e = ns - 1; e >= 0; e--) {
                int p = sump[e];
                if (++si[e] < size[p]) { for (int k = 0; k < nops; k++) off[k] += lstr[k][p]; break; }
                si[e] = 0;
                for (int k = 0; k < nops; k++) off[k] -= (size[p] - 1) * lstr[k][p];
            }
        }
        r.data[q] = acc;
        for (int e = nout - 1; e >= 0; e--) {
            int p = outp[e];
            if (++oi[e] < od[e]) { for (int k = 0; k < nops; k++) base[k] += lstr[k][p]; break; }
            oi[e] = 0;
            for (int k = 0; k < nops; k++) base[k] -= (size[p] - 1) * lstr[k][p];
        }
    }
    return r;
}
VIV_AR_UNUSED static viv_arr viv_ar_einsum_2(const char* s, viv_arr a, int site) { viv_arr o[1]; o[0] = a; return viv_ar_einsum(s, 1, o, site); }
VIV_AR_UNUSED static viv_arr viv_ar_einsum_3(const char* s, viv_arr a, viv_arr b, int site) { viv_arr o[2]; o[0] = a; o[1] = b; return viv_ar_einsum(s, 2, o, site); }
VIV_AR_UNUSED static viv_arr viv_ar_einsum_4(const char* s, viv_arr a, viv_arr b, viv_arr c, int site) { viv_arr o[3]; o[0] = a; o[1] = b; o[2] = c; return viv_ar_einsum(s, 3, o, site); }
VIV_AR_UNUSED static double viv_ar_einsum_2n(const char* s, viv_arr a, int site) { return viv_ar_einsum_2(s, a, site).data[0]; }
VIV_AR_UNUSED static double viv_ar_einsum_3n(const char* s, viv_arr a, viv_arr b, int site) { return viv_ar_einsum_3(s, a, b, site).data[0]; }
VIV_AR_UNUSED static double viv_ar_einsum_4n(const char* s, viv_arr a, viv_arr b, viv_arr c, int site) { return viv_ar_einsum_4(s, a, b, c, site).data[0]; }"#;

/// `x = lin_solve(A, b)` and `inv(A)` (fragment "ar_solve"): the older solvers
/// with the array they fill allocated here, after the interpreter's checks.
pub const C_SOLVE: &str = r#"VIV_AR_UNUSED static viv_arr viv_ar_lin_solve_2(viv_arr A, viv_arr b, int site) {
    if (A.rank != 2 || A.dims[0] != A.dims[1]) viv_ar_fail(VIV_E_VALUE, b.rank == 2 ? "`lin_solve_multi` needs a square 2-D matrix" : "`lin_solve` needs a square 2-D matrix", site);
    int n = A.dims[0];
    if (b.rank == 2) {
        if (b.dims[0] != n) viv_ar_fail(VIV_E_VALUE, "`lin_solve_multi`: B must be n x m", site);
        viv_arr X = viv_arr_new(2, b.dims);
        viv_lin_solve_multi_at(A, b, X, site);
        return X;
    }
    if (b.size < n) viv_ar_fail(VIV_E_VALUE, "`lin_solve`: b is too small", site);
    int d[1]; d[0] = n;
    viv_arr x = viv_arr_new(1, d);
    viv_lin_solve_at(A, b, x, site);
    return x;
}
VIV_AR_UNUSED static viv_arr viv_ar_inv_1(viv_arr A, int site) {
    if (A.rank != 2 || A.dims[0] != A.dims[1]) viv_ar_fail(VIV_E_VALUE, "`lin_solve_multi` needs a square 2-D matrix", site);
    /* not `I`: complex.h defines it as a macro */
    viv_arr eye = viv_arr_new(2, A.dims);
    for (int i = 0; i < A.dims[0]; i++) eye.data[i * A.dims[0] + i] = 1.0;
    viv_arr X = viv_arr_new(2, A.dims);
    viv_lin_solve_multi_at(A, eye, X, site);
    return X;
}"#;

/// `L = cholesky(A)` (fragment "ar_chol").
pub const C_CHOL: &str = r#"VIV_AR_UNUSED static viv_arr viv_ar_cholesky_1(viv_arr A, int site) {
    if (A.rank != 2 || A.dims[0] != A.dims[1]) viv_ar_fail(VIV_E_VALUE, "`cholesky` needs a square 2-D matrix", site);
    viv_arr L = viv_arr_new(2, A.dims);
    viv_cholesky(A, L);
    return L;
}"#;

/// `svd(A, U, s, Vt)` (fragment "ar_svd"): one-sided Jacobi, the
/// interpreter's `svd_tall` operation for operation.
pub const C_SVD: &str = r#"static void viv_ar_svd_tall(const double* a, int m, int n, double* u, double* s, double* v) {
    double* w = (double*)malloc((size_t)(m * n ? m * n : 1) * sizeof(double));
    memcpy(w, a, (size_t)m * (size_t)n * sizeof(double));
    for (int i = 0; i < n * n; i++) v[i] = 0.0;
    for (int i = 0; i < n; i++) v[i * n + i] = 1.0;
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (int p = 0; p + 1 < n; p++) for (int q = p + 1; q < n; q++) {
            double alpha = 0.0, beta = 0.0, gamma = 0.0;
            for (int i = 0; i < m; i++) {
                alpha += w[i * n + p] * w[i * n + p];
                beta += w[i * n + q] * w[i * n + q];
                gamma += w[i * n + p] * w[i * n + q];
            }
            double scale = sqrt(alpha * beta);
            if (scale > 0.0) {
                off = fmax(off, fabs(gamma) / scale);
                if (fabs(gamma) > 1.0e-15 * scale) {
                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t = 1.0 / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                    if (zeta < 0.0) t = -t;
                    double c = 1.0 / sqrt(1.0 + t * t), sn = c * t;
                    for (int i = 0; i < m; i++) {
                        double ap = w[i * n + p], aq = w[i * n + q];
                        w[i * n + p] = c * ap - sn * aq;
                        w[i * n + q] = sn * ap + c * aq;
                    }
                    for (int i = 0; i < n; i++) {
                        double vp = v[i * n + p], vq = v[i * n + q];
                        v[i * n + p] = c * vp - sn * vq;
                        v[i * n + q] = sn * vp + c * vq;
                    }
                }
            }
        }
        if (off < 1.0e-15) break;
    }
    for (int j = 0; j < n; j++) {
        double total = 0.0;
        for (int i = 0; i < m; i++) total += w[i * n + j] * w[i * n + j];
        s[j] = sqrt(total);
    }
    for (int a0 = 0; a0 < n; a0++) {
        int best = a0;
        for (int j = a0; j < n; j++) if (s[j] > s[best]) best = j;
        if (best != a0) {
            double t = s[a0]; s[a0] = s[best]; s[best] = t;
            for (int i = 0; i < m; i++) { t = w[i * n + a0]; w[i * n + a0] = w[i * n + best]; w[i * n + best] = t; }
            for (int i = 0; i < n; i++) { t = v[i * n + a0]; v[i * n + a0] = v[i * n + best]; v[i * n + best] = t; }
        }
    }
    for (int i = 0; i < m * n; i++) u[i] = 0.0;
    for (int j = 0; j < n; j++) {
        if (s[j] > 0.0) { for (int i = 0; i < m; i++) u[i * n + j] = w[i * n + j] / s[j]; }
        else u[j * n + j] = 1.0;
    }
    free(w);
}
static void viv_ar_svd_out(int k, viv_arr t, int rank, int d0, int d1, int site) {
    int ok = t.rank == rank && t.dims[0] == d0 && (rank == 1 || t.dims[1] == d1);
    if (!ok) {
        char s[120], want[120], m[320];
        int w[2]; w[0] = d0; w[1] = d1;
        viv_ar_shape(s, sizeof s, t.rank, t.dims); viv_ar_shape(want, sizeof want, rank, w);
        snprintf(m, sizeof m, "`svd`: output %d has shape %s; it needs %s", k, s, want);
        viv_ar_fail(VIV_E_VALUE, m, site);
    }
}
VIV_AR_UNUSED static double viv_ar_svd_4(viv_arr A, viv_arr U, viv_arr S, viv_arr Vt, int site) {
    char msg[120];
    if (A.rank != 2) { snprintf(msg, sizeof msg, "`svd` takes a matrix; this array has rank %d", A.rank); viv_ar_fail(VIV_E_VALUE, msg, site); }
    for (int i = 0; i < A.size; i++) if (!(A.data[i] - A.data[i] == 0.0)) viv_ar_fail(VIV_E_VALUE, "`svd`: the matrix has a NaN or an infinity", site);
    int m = A.dims[0], n = A.dims[1], k = m < n ? m : n;
    viv_ar_svd_out(1, U, 2, m, k, site);
    viv_ar_svd_out(2, S, 1, k, 0, site);
    viv_ar_svd_out(3, Vt, 2, k, n, site);
    if (m >= n) {
        double* v = (double*)malloc((size_t)(n * n ? n * n : 1) * sizeof(double));
        viv_ar_svd_tall(A.data, m, n, U.data, S.data, v);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Vt.data[i * n + j] = v[j * n + i];
        free(v);
        return 0.0;
    }
    double* at = (double*)malloc((size_t)(m * n) * sizeof(double));
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) at[j * m + i] = A.data[i * n + j];
    double* u2 = (double*)malloc((size_t)(n * m) * sizeof(double));
    viv_ar_svd_tall(at, n, m, u2, S.data, U.data);
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) Vt.data[i * n + j] = u2[j * m + i];
    free(at); free(u2);
    return 0.0;
}"#;
