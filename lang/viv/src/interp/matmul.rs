// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `@` on real arrays: the product, its batched form, and its adjoints.
//!
//! This file is the one statement of the rules. The interpreter and the wasm
//! runtime (`interp/wasm_rt.rs`, `vrt_matmul`) call `matmul` here, the
//! autodiff helpers `__ad_mm_a` and `__ad_mm_b` call `adjoint_a` and
//! `adjoint_b`, and `C_MATMUL` below is the C twin, loop for loop.
//!
//! **Shapes** are NumPy's `matmul`. A 1-D left operand is a row (`1 x k`)
//! and a 1-D right operand a column (`k x 1`), and the axis each adds is
//! dropped from the result. Two 2-D operands give the matrix product. An
//! operand of rank 3 or more is a stack of matrices in its last two axes; the
//! axes before them are batch axes and broadcast as NumPy's do, so a matrix
//! multiplies every matrix of a stack. Two vectors are refused (`dot(u, v)`
//! is their product).
//!
//! **The order of the sums.** Each entry of the result is
//!
//! ```text
//! r[i, j] = (((0 + a[i, 0] b[0, j]) + a[i, 1] b[1, j]) + ...) + a[i, k-1] b[k-1, j]
//! ```
//!
//! with every term whose `a[i, p]` is zero left out (a zero times an
//! infinity or a NaN in `b` does not reach the result). This is the order of
//! the plain i-p-j loop Viv has always used, so the result is the same bit
//! for bit as it was before the loop was blocked, and the same in every
//! engine. Each matrix of a batch is one such product.
//!
//! **The blocking.** The loop runs over blocks of `NB` columns of `b` and the
//! result, and within each over blocks of `KB` rows of `b`, so the block of
//! `b` being read (`KB x NB` doubles, 256 KB) stays in the cache while every
//! row of `a` passes over it. Within a block the result is taken in tiles of
//! four rows by eight columns, held in registers while the depth of the
//! block runs: each of the tile's 32 entries takes one product per `p`, in
//! the order of `p`, and is written back once. Columns and rows that do not
//! fill a tile are done a row at a time. For a fixed entry `(i, j)` the depth
//! blocks come in order, and within a block the depth `p` does too, so the
//! blocking does not change the order above. A tile's four rows are updated
//! together only when none of their four `a[i, p]` is zero; otherwise each
//! row with a nonzero one is updated alone, which keeps the zero rule entry
//! by entry. Products and sums are separate operations (no fused
//! multiply-add, as everywhere in Viv), which is what keeps the bits the
//! same as the plain loop's and the same in every engine.

use super::arrays;

/// Columns of a block of `b` and of the result.
pub const NB: usize = 256;
/// Rows of a block of `b` (the depth of the product).
pub const KB: usize = 128;
/// Rows of the result updated in one pass over a row of `b`.
const ROWS: usize = 4;

/// Columns of a tile of the result kept in registers while the depth runs.
const COLS: usize = 8;

/// `r += a @ b` for one `m x k` by `k x n` product, row-major, in the order
/// the module comment states. `r` starts at zero for a plain product.
pub fn kernel(a: &[f64], b: &[f64], r: &mut [f64], m: usize, k: usize, n: usize) {
    // the digits lens (crate::digits) reorders, fuses or perturbs the sums
    if crate::digits::on() && crate::digits::matmul(a, b, r, m, k, n) {
        return;
    }
    if m == 0 || n == 0 {
        return;
    }
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + NB).min(n);
        let mut p0 = 0;
        while p0 < k {
            let p1 = (p0 + KB).min(k);
            let mut i = 0;
            while i + ROWS <= m {
                let (a0, a1, a2, a3) = (&a[i * k..], &a[(i + 1) * k..], &a[(i + 2) * k..], &a[(i + 3) * k..]);
                // tiles of four rows by eight columns, held in `c` over the depth
                let mut j = j0;
                while j + COLS <= j1 {
                    let mut c = [[0.0f64; COLS]; ROWS];
                    for (u, cu) in c.iter_mut().enumerate() {
                        cu.copy_from_slice(&r[(i + u) * n + j..(i + u) * n + j + COLS]);
                    }
                    for p in p0..p1 {
                        let x = [a0[p], a1[p], a2[p], a3[p]];
                        let bp: &[f64; COLS] = b[p * n + j..p * n + j + COLS].try_into().expect("a tile's columns");
                        if x[0] != 0.0 && x[1] != 0.0 && x[2] != 0.0 && x[3] != 0.0 {
                            for v in 0..COLS {
                                let bv = bp[v];
                                c[0][v] += x[0] * bv;
                                c[1][v] += x[1] * bv;
                                c[2][v] += x[2] * bv;
                                c[3][v] += x[3] * bv;
                            }
                        } else {
                            for u in 0..ROWS {
                                if x[u] != 0.0 {
                                    for v in 0..COLS {
                                        c[u][v] += x[u] * bp[v];
                                    }
                                }
                            }
                        }
                    }
                    for (u, cu) in c.iter().enumerate() {
                        r[(i + u) * n + j..(i + u) * n + j + COLS].copy_from_slice(cu);
                    }
                    j += COLS;
                }
                // the columns left over, a row at a time
                for u in 0..ROWS {
                    let q = &mut r[(i + u) * n + j..(i + u) * n + j1];
                    for p in p0..p1 {
                        let x = a[(i + u) * k + p];
                        if x == 0.0 {
                            continue;
                        }
                        for (y, &bv) in q.iter_mut().zip(&b[p * n + j..p * n + j1]) {
                            *y += x * bv;
                        }
                    }
                }
                i += ROWS;
            }
            // the rows left over
            while i < m {
                let q = &mut r[i * n + j0..i * n + j1];
                for p in p0..p1 {
                    let x = a[i * k + p];
                    if x == 0.0 {
                        continue;
                    }
                    for (y, &bv) in q.iter_mut().zip(&b[p * n + j0..p * n + j1]) {
                        *y += x * bv;
                    }
                }
                i += 1;
            }
            p0 = p1;
        }
        j0 = j1;
    }
}

/// The parts of `a @ b`: each operand's matrix (`m x k`, `k x n`), the batch
/// shape the two broadcast to, and the result's shape.
struct Plan {
    m: usize,
    k: usize,
    n: usize,
    batch: Vec<usize>,
    out: Vec<usize>,
}

/// `@` of two 1-D arrays: NumPy gives their dot product; Viv refuses it and
/// names the builtins that say which product is meant (every engine's words).
pub const VECTORS: &str = "`@` of two vectors is refused: dot(u, v) is their dot product, outer(u, v) their outer product";

fn plan(sa: &[usize], sb: &[usize]) -> Result<Plan, String> {
    let (ra, rb) = (sa.len(), sb.len());
    if ra == 1 && rb == 1 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, VECTORS));
    }
    if ra == 0 || rb == 0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`@` needs arrays (2-D, or a 1-D vector operand)"));
    }
    let (m, k) = if ra == 1 { (1, sa[0]) } else { (sa[ra - 2], sa[ra - 1]) };
    let (k2, n) = if rb == 1 { (sb[0], 1) } else { (sb[rb - 2], sb[rb - 1]) };
    if k != k2 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`@` shape mismatch: {m}x{k} @ {k2}x{n}")));
    }
    let ba = &sa[..ra.saturating_sub(2)];
    let bb = &sb[..rb.saturating_sub(2)];
    let Some(batch) = arrays::broadcast(ba, bb) else {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`@`: the stacks of matrices have shapes {ba:?} and {bb:?}, which do not broadcast")));
    };
    let mut out = batch.clone();
    if ra > 1 {
        out.push(m);
    }
    if rb > 1 {
        out.push(n);
    }
    if out.len() > crate::lower::MAX_RANK {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`@`: the result would have rank {}; arrays have rank {} at most", out.len(), crate::lower::MAX_RANK)));
    }
    Ok(Plan { m, k, n, batch, out })
}

/// The flat offset, in matrices, of batch entry `q` (row-major over `batch`)
/// in an operand whose batch axes are `own` (aligned at the right, an axis of
/// length 1 repeated).
fn batch_offset(q: usize, batch: &[usize], own: &[usize]) -> usize {
    let lead = batch.len() - own.len();
    let mut rest = q;
    let mut off = 0;
    let mut stride = 1;
    for d in (0..batch.len()).rev() {
        let i = rest % batch[d];
        rest /= batch[d];
        if d >= lead {
            let len = own[d - lead];
            if len != 1 {
                off += i * stride;
            }
            stride *= len;
        }
    }
    off
}

/// The shape of `a @ b`, or why there is none.
pub fn shape(sa: &[usize], sb: &[usize]) -> Result<Vec<usize>, String> {
    let p = plan(sa, sb)?;
    let count: usize = p.batch.iter().product();
    let total = count.saturating_mul(p.m.saturating_mul(p.n));
    if total > super::MAX_ELEMS {
        return Err(if count == 1 {
            crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`@`: result of {}x{} exceeds the {}-element limit{}", p.m, p.n, super::MAX_ELEMS, super::LIMIT_OF))
        } else {
            crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`@`: result of {} matrices of {}x{} exceeds the {}-element limit{}", count, p.m, p.n, super::MAX_ELEMS, super::LIMIT_OF))
        });
    }
    Ok(p.out)
}

/// `a @ b` written into `r`, which has the result's size (`shape`) and is
/// zero.
pub fn matmul_into(sa: &[usize], da: &[f64], sb: &[usize], db: &[f64], r: &mut [f64]) -> Result<(), String> {
    let p = plan(sa, sb)?;
    let count: usize = p.batch.iter().product();
    let (ba, bb) = (&sa[..sa.len().saturating_sub(2)], &sb[..sb.len().saturating_sub(2)]);
    let (sza, szb, szr) = (p.m * p.k, p.k * p.n, p.m * p.n);
    for q in 0..count {
        let oa = batch_offset(q, &p.batch, ba) * sza;
        let ob = batch_offset(q, &p.batch, bb) * szb;
        kernel(&da[oa..oa + sza], &db[ob..ob + szb], &mut r[q * szr..(q + 1) * szr], p.m, p.k, p.n);
    }
    Ok(())
}

/// `a @ b`: the result's shape and entries.
pub fn matmul(sa: &[usize], da: &[f64], sb: &[usize], db: &[f64]) -> Result<(Vec<usize>, Vec<f64>), String> {
    let out = shape(sa, sb)?;
    let mut r = vec![0.0; out.iter().product()];
    matmul_into(sa, da, sb, db, &mut r)?;
    Ok((out, r))
}

/// An array with its last two axes swapped (a vector is left as it is).
fn swap_last(shape: &[usize], data: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let r = shape.len();
    if r < 2 {
        return (shape.to_vec(), data.to_vec());
    }
    let (rows, cols) = (shape[r - 2], shape[r - 1]);
    let mut s = shape.to_vec();
    s.swap(r - 2, r - 1);
    let mut out = vec![0.0; data.len()];
    let each = rows * cols;
    for q in 0..data.len() / each.max(1) {
        let (src, dst) = (&data[q * each..(q + 1) * each], &mut out[q * each..(q + 1) * each]);
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
    (s, out)
}

/// An operand as a stack of matrices: a vector on the left is a row, on the
/// right a column.
fn as_matrix(shape: &[usize], left: bool) -> Vec<usize> {
    match (shape.len(), left) {
        (1, true) => vec![1, shape[0]],
        (1, false) => vec![shape[0], 1],
        _ => shape.to_vec(),
    }
}

/// The result's shape with the axes a vector operand dropped put back.
fn full_out(sa: &[usize], sb: &[usize]) -> Result<Vec<usize>, String> {
    let p = plan(sa, sb)?;
    let mut s = p.batch;
    s.push(p.m);
    s.push(p.n);
    Ok(s)
}

/// The adjoint of `a` in `c = a @ b`, given `g`, the adjoint of `c`:
/// `g @ swap(b)`, summed over the batch axes `a` was broadcast along (in
/// `g`'s row-major order, as every unbroadcast is), in `a`'s shape.
pub fn adjoint_a(g: &[f64], sa: &[usize], sb: &[usize], db: &[f64]) -> Result<Vec<f64>, String> {
    let g_full = full_out(sa, sb)?;
    let (sbt, dbt) = swap_last(&as_matrix(sb, false), db);
    let (s, d) = matmul(&g_full, g, &sbt, &dbt)?;
    Ok(arrays::unbroadcast(&s, &d, &as_matrix(sa, true)))
}

/// The adjoint of `b`: `swap(a) @ g`, summed over the batch axes `b` was
/// broadcast along, in `b`'s shape.
pub fn adjoint_b(g: &[f64], sa: &[usize], da: &[f64], sb: &[usize]) -> Result<Vec<f64>, String> {
    let g_full = full_out(sa, sb)?;
    let (sat, dat) = swap_last(&as_matrix(sa, true), da);
    let (s, d) = matmul(&sat, &dat, &g_full, g)?;
    Ok(arrays::unbroadcast(&s, &d, &as_matrix(sb, false)))
}

/// The C twin (fragment "matmul"): `viv_mm_kernel` is `kernel` and
/// `viv_matmul` is `matmul`, its shapes checked first by `viv_arr_chk`, which
/// gives the same messages.
pub const C_MATMUL: &str = r#"/* `@` on real arrays (lang/viv/src/interp/matmul.rs): the same blocked loop,
   which adds each entry's terms in the order of the plain i-p-j loop, a term
   left out where a[i, p] is zero, so every engine gives the same bits. A
   tile of 4 rows by 8 columns of r stays in registers over a block of the
   depth. */
#ifdef VIV_FAST_LA
/* the exact build's product, for the shadow measurement: the plain i-p-j
   loop, whose bits the blocked loop below keeps, with contraction off (the
   fast build's flags honor this pragma) */
static void viv_mm_exact_ref(const double* a, const double* b, double* r, int m, int k, int n) {
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            double x = a[(size_t)i * k + p];
            if (x == 0.0) continue;
            for (int j = 0; j < n; j++) r[(size_t)i * n + j] += x * b[(size_t)p * n + j];
        }
}
#endif
static void viv_mm_kernel(const double* restrict a, const double* restrict b, double* restrict r, int m, int k, int n) {
    if (m == 0 || n == 0) return;
#ifdef VIV_FAST_LA
    /* fast mode: dgemm, in its own order of the sums,
       when the product is large enough to pay for the call and b is finite
       (the loop below leaves out a zero a[i, p] times an infinity or NaN in
       b; dgemm does not, so such a b keeps the loop) */
    if ((long long)m * k * n >= VIV_FAST_MM_MIN && viv_fast_finite(b, (long long)k * n)) {
        FILE* sh = viv_fast_shadow();
        double* r0 = 0;
        if (sh) {
            r0 = (double*)malloc((size_t)m * n * sizeof(double));
            if (r0) for (long long q = 0; q < (long long)m * n; q++) r0[q] = r[q];
        }
        cblas_dgemm(VIV_CBLAS_ROW_MAJOR, VIV_CBLAS_NO_TRANS, VIV_CBLAS_NO_TRANS, m, n, k, 1.0, a, k, b, n, 1.0, r, n);
        if (r0) {
            /* the exact build's answer, and each entry's bound: both sums meet
               |error| <= (k + 1) u (|r0| + sum_p |a||b|), so they differ by
               at most twice that */
            viv_mm_exact_ref(a, b, r0, m, k, n);
            double worst = 0.0, maxdiff = 0.0;
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
                double mag = 0.0;
                for (int p = 0; p < k; p++) mag += fabs(a[(size_t)i * k + p] * b[(size_t)p * n + j]);
                double bound = 2.0 * (k + 1) * 0x1p-53 * (mag + fabs(r0[(size_t)i * n + j]));
                double d = fabs(r[(size_t)i * n + j] - r0[(size_t)i * n + j]);
                if (d > maxdiff) maxdiff = d;
                double ratio = d == 0.0 ? 0.0 : d / bound;
                if (ratio > worst) worst = ratio;
            }
            fprintf(sh, "gemm m=%d k=%d n=%d maxdiff=%.3e of_bound=%.3e\n", m, k, n, maxdiff, worst);
            fflush(sh);
            free(r0);
        }
        return;
    }
#endif
    for (int j0 = 0; j0 < n; j0 += 256) {
        int j1 = j0 + 256 < n ? j0 + 256 : n;
        for (int p0 = 0; p0 < k; p0 += 128) {
            int p1 = p0 + 128 < k ? p0 + 128 : k;
            int i = 0;
            for (; i + 4 <= m; i += 4) {
                const double* a0 = a + (size_t)i * k; const double* a1 = a0 + k; const double* a2 = a1 + k; const double* a3 = a2 + k;
                int j = j0;
                for (; j + 8 <= j1; j += 8) {
                    double c[4][8];
                    for (int u = 0; u < 4; u++) for (int v = 0; v < 8; v++) c[u][v] = r[(size_t)(i + u) * n + j + v];
                    for (int p = p0; p < p1; p++) {
                        double x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
                        const double* bp = b + (size_t)p * n + j;
                        if (x0 != 0.0 && x1 != 0.0 && x2 != 0.0 && x3 != 0.0) {
                            for (int v = 0; v < 8; v++) {
                                double bv = bp[v];
                                c[0][v] += x0 * bv; c[1][v] += x1 * bv; c[2][v] += x2 * bv; c[3][v] += x3 * bv;
                            }
                        } else {
                            if (x0 != 0.0) for (int v = 0; v < 8; v++) c[0][v] += x0 * bp[v];
                            if (x1 != 0.0) for (int v = 0; v < 8; v++) c[1][v] += x1 * bp[v];
                            if (x2 != 0.0) for (int v = 0; v < 8; v++) c[2][v] += x2 * bp[v];
                            if (x3 != 0.0) for (int v = 0; v < 8; v++) c[3][v] += x3 * bp[v];
                        }
                    }
                    for (int u = 0; u < 4; u++) for (int v = 0; v < 8; v++) r[(size_t)(i + u) * n + j + v] = c[u][v];
                }
                for (int u = 0; u < 4; u++) {
                    double* q = r + (size_t)(i + u) * n;
                    const double* au = a + (size_t)(i + u) * k;
                    for (int p = p0; p < p1; p++) {
                        double x = au[p]; if (x == 0.0) continue;
                        const double* bp = b + (size_t)p * n;
                        for (int jj = j; jj < j1; jj++) q[jj] += x * bp[jj];
                    }
                }
            }
            for (; i < m; i++) {
                double* q = r + (size_t)i * n;
                for (int p = p0; p < p1; p++) {
                    double x = a[(size_t)i * k + p]; if (x == 0.0) continue;
                    const double* bp = b + (size_t)p * n;
                    for (int j = j0; j < j1; j++) q[j] += x * bp[j];
                }
            }
        }
    }
}
/* the batch offset, in matrices, of entry q of `batch` in an operand whose
   batch axes are `own` (aligned at the right, an axis of length 1 repeated) */
static long long viv_mm_off(long long q, int nb, const int* batch, int no, const int* own) {
    int lead = nb - no;
    long long off = 0, stride = 1;
    for (int d = nb - 1; d >= 0; d--) {
        long long i = q % batch[d];
        q /= batch[d];
        if (d >= lead) {
            int len = own[d - lead];
            if (len != 1) off += i * stride;
            stride *= len;
        }
    }
    return off;
}
/* a @ b, the shapes already checked (viv_arr_chk): a vector on the left is a
   row and on the right a column, the axis it adds dropped; the axes before
   the last two are batch axes, broadcast */
static viv_arr viv_matmul(viv_arr a, viv_arr b) {
    int m = a.rank == 1 ? 1 : a.dims[a.rank - 2], k = a.dims[a.rank - 1];
    int n = b.rank == 1 ? 1 : b.dims[b.rank - 1];
    int na = a.rank > 2 ? a.rank - 2 : 0, nb = b.rank > 2 ? b.rank - 2 : 0;
    int nr = na > nb ? na : nb, batch[8], out[8], ro = 0;
    for (int d = 0; d < nr; d++) {
        int x = d + na >= nr ? a.dims[d + na - nr] : 1, y = d + nb >= nr ? b.dims[d + nb - nr] : 1;
        batch[d] = x == 1 ? y : x;
        out[ro++] = batch[d];
    }
    if (a.rank > 1) out[ro++] = m;
    if (b.rank > 1) out[ro++] = n;
    viv_arr r = viv_arr_new(ro, out);
    long long count = 1;
    for (int d = 0; d < nr; d++) count *= batch[d];
    long long sza = (long long)m * k, szb = (long long)k * n, szr = (long long)m * n;
    for (long long q = 0; q < count; q++) {
        long long oa = viv_mm_off(q, nr, batch, na, a.dims) * sza;
        long long ob = viv_mm_off(q, nr, batch, nb, b.dims) * szb;
        viv_mm_kernel(a.data + oa, b.data + ob, r.data + q * szr, m, k, n);
    }
    return r;
}"#;

/// The adjoints of `@` in C (fragment "mm_adj"): `adjoint_a` and
/// `adjoint_b`'s twin, which the autodiff helpers `viv_ad_mm_a` and
/// `viv_ad_mm_b` call for a batched product.
pub const C_MM_ADJ: &str = r#"/* an array with its last two axes swapped (a new array) */
static viv_arr viv_mm_swap(viv_arr a) {
    if (a.rank < 2) return a;
    int d[8];
    for (int i = 0; i < a.rank; i++) d[i] = a.dims[i];
    int rows = d[a.rank - 2], cols = d[a.rank - 1];
    d[a.rank - 2] = cols; d[a.rank - 1] = rows;
    viv_arr r = viv_arr_new(a.rank, d);
    long long each = (long long)rows * cols;
    for (long long q = 0; each > 0 && q < a.size / each; q++)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) r.data[q * each + (long long)j * rows + i] = a.data[q * each + (long long)i * cols + j];
    return r;
}
/* an operand as a stack of matrices, sharing its data: a vector on the left
   is a row, on the right a column */
static viv_arr viv_mm_as_matrix(viv_arr a, int left) {
    if (a.rank != 1) return a;
    viv_arr v = a;
    v.rank = 2;
    if (left) { v.dims[0] = 1; v.dims[1] = a.dims[0]; } else { v.dims[0] = a.dims[0]; v.dims[1] = 1; }
    return v;
}
/* the adjoint of a (which == 0) or of b (which == 1) in c = a @ b, given g:
   g @ swap(b) or swap(a) @ g, summed over the batch axes the operand was
   broadcast along, in the operand's shape */
static viv_arr viv_mm_adj(viv_arr g, viv_arr a, viv_arr b, int which) {
    viv_arr a2 = viv_mm_as_matrix(a, 1), b2 = viv_mm_as_matrix(b, 0);
    int na = a2.rank - 2, nb = b2.rank - 2, nr = na > nb ? na : nb;
    viv_arr gf = g;
    gf.rank = nr + 2;
    for (int d = 0; d < nr; d++) {
        int x = d + na >= nr ? a2.dims[d + na - nr] : 1, y = d + nb >= nr ? b2.dims[d + nb - nr] : 1;
        gf.dims[d] = x == 1 ? y : x;
    }
    gf.dims[nr] = a2.dims[a2.rank - 2];
    gf.dims[nr + 1] = b2.dims[b2.rank - 1];
    viv_arr full = which == 0 ? viv_matmul(gf, viv_mm_swap(b2)) : viv_matmul(viv_mm_swap(a2), gf);
    viv_arr like = which == 0 ? a2 : b2;
    viv_arr r = viv_ad_unb_aa(full, like);
    viv_arr out = which == 0 ? a : b;
    out.data = r.data;
    return out;
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The plain i-p-j loop with the zero rule: what `@` computed before it
    /// was blocked.
    fn plain(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut r = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let x = a[i * k + p];
                if x == 0.0 {
                    continue;
                }
                for j in 0..n {
                    r[i * n + j] += x * b[p * n + j];
                }
            }
        }
        r
    }

    #[test]
    fn the_blocked_loop_gives_the_plain_loops_bits() {
        let mut seed = 12345u64;
        let mut draw = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        // sizes on both sides of the blocks and of four rows
        for &(m, k, n) in &[(1, 1, 1), (3, 5, 2), (4, 4, 4), (7, 130, 9), (9, 300, 261), (13, 129, 513), (5, 257, 1)] {
            let mut a: Vec<f64> = (0..m * k).map(|_| draw()).collect();
            let mut b: Vec<f64> = (0..k * n).map(|_| draw() * 1.0e3).collect();
            // zeros in a, and an infinity and a NaN in b behind some of them
            for i in (0..a.len()).step_by(7) {
                a[i] = 0.0;
            }
            b[0] = f64::INFINITY;
            if b.len() > 3 {
                b[3] = f64::NAN;
            }
            let mut r = vec![0.0; m * n];
            kernel(&a, &b, &mut r, m, k, n);
            let want = plain(&a, &b, m, k, n);
            for (x, y) in r.iter().zip(&want) {
                assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()), "{m}x{k}x{n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn a_stack_of_matrices_times_one_matrix_is_each_product() {
        let a: Vec<f64> = (0..24).map(|x| x as f64 * 0.5 - 3.0).collect();
        let b: Vec<f64> = (0..20).map(|x| (x as f64).sin()).collect();
        let (s, r) = matmul(&[2, 3, 4], &a, &[4, 5], &b).unwrap();
        assert_eq!(s, vec![2, 3, 5]);
        for q in 0..2 {
            assert_eq!(&r[q * 15..(q + 1) * 15], &plain(&a[q * 12..(q + 1) * 12], &b, 3, 4, 5)[..]);
        }
        // a vector on the right drops its axis
        let (s, _) = matmul(&[2, 3, 4], &a, &[4], &b[..4]).unwrap();
        assert_eq!(s, vec![2, 3]);
        // batch axes that do not broadcast
        assert!(matmul(&[2, 3, 4], &a, &[3, 4, 5], &[0.0; 60]).is_err());
    }
}
