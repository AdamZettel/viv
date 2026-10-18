// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Fused elementwise expressions and reductions over views, for the compiled
//! engines (the C backend and the direct wasm backend).
//!
//! `y = a*x + b*z - c` lowers to one array operation per operator, each of
//! which allocates its result; `sum(d[o:o + n], ordered=true)` copies the
//! slice before adding it. Here such an expression tree becomes a small
//! postfix program that one runtime call runs: the C runtime's `viv_fz` and
//! `viv_fz_red` (fragment "fuse", `C_RUNTIME` below) and the wasm runtime's
//! `vrt_fz` and `vrt_fz_red` (interp/wasm_rt.rs) are twins.
//!
//! In C, a tree assigned to a variable (`y = a*x + b*z - c`) has its element
//! loop written out instead (lower.rs, `fuse_assign_c`, with `element_expr`
//! below): `viv_fz_prep` does the walk and makes the result, then one loop
//! over plain pointers computes each element as one expression, which the C
//! compiler vectorizes as it does a hand-written loop. The block interpreter
//! (`viv_fz_block`) does the same operations a block at a time, each through
//! memory.
//!
//! The rules, which keep every result and every error what it was:
//!
//! - Each element is computed with the operations of the unfused tree, in the
//!   same order, on the same operands: the bits do not change (the exact
//!   build compiles with `-ffp-contract=off`).
//! - A slice inside the tree is read where it lies, as a view; it is never
//!   copied. Only the tree's own operations read it, and they finish before
//!   the call returns, so no write can come between the read and the view.
//! - The runtime first walks the program in the order the unfused code runs
//!   (the operands of an operation before it, left to right), planning each
//!   slice with the slice runtime's checks. A slice out of bounds stops the
//!   run there with its own words and location, as before. Where two arrays
//!   meet with different shapes (a broadcast or a mismatch), or a slice is
//!   not evenly spaced in memory, the call runs the unfused operations
//!   instead, one at a time, which broadcast or stop exactly as before.
//! - An operand that is not a plain value (a call, an indexed element) is
//!   evaluated before the runtime call, so it is allowed only where the
//!   unfused code also evaluated it before any check that could stop the
//!   run: no check precedes it in that order.
//!
//! The program is a list of numbers:
//!
//! ```text
//!   ARR k            the array arrs[k]
//!   NUM k            the number nums[k]
//!   SLICE k m j s    arrs[k] sliced by the spec nums[j .. j + 4m] (site s)
//!   ADD s .. DIV s   two arrays of one shape (site s of the shape check)
//!   ADDS .. POWS     an array and a number, children [array, number]
//!   MAP f            fns[f] of each element
//!   END              the end of an operand (a reduction's; `dot` has two)
//! ```

use crate::lower::{ArrKind, Viv};
use knotkit::ir::{IrProgram, NodeId};

pub const ARR: f64 = 0.0;
pub const NUM: f64 = 1.0;
pub const SLICE: f64 = 2.0;
pub const ADD: f64 = 3.0;
pub const SUB: f64 = 4.0;
pub const MUL: f64 = 5.0;
pub const DIV: f64 = 6.0;
pub const ADDS: f64 = 7.0;
pub const SUBS: f64 = 8.0;
pub const SUBSR: f64 = 9.0;
pub const SCALE: f64 = 10.0;
pub const DIVS: f64 = 11.0;
pub const DIVSR: f64 = 12.0;
pub const POWS: f64 = 13.0;
pub const MAP: f64 = 14.0;
pub const END: f64 = 15.0;

/// The reductions a view or a fused tree can feed, by the callee the
/// lowering writes, with the number the runtime knows each by and how many
/// array operands it takes.
pub const REDUCTIONS: &[(&str, u8, usize)] = &[
    ("viv_sum", 0, 1),
    ("viv_ar___sum_ordered_1", 1, 1),
    ("viv_ar_prod_1", 2, 1),
    ("viv_ar_mean_1", 3, 1),
    ("viv_ar_norm_1", 4, 1),
    ("viv_ar_dot_2", 5, 2),
    ("viv_maxval", 6, 1),
    ("viv_minval", 7, 1),
];

/// The one-argument math a tree may map over its elements: the C names the
/// lowering writes (`Viv::MapMath`), which the wasm runtime's `MAP_FNS` lists
/// too.
pub const MAP_NAMES: &[&str] = &["fabs", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "floor", "ceil"];

/// Fusion is on unless `VIV_NOFUSE` is set (to compare the two builds).
pub fn enabled() -> bool {
    std::env::var_os("VIV_NOFUSE").is_none()
}

/// A tree as a program: the program, the array and number operands in the
/// order the program names them, the mapped functions, and the sites.
#[derive(Debug, Default)]
pub struct Plan {
    pub code: Vec<f64>,
    pub arrs: Vec<NodeId>,
    pub nums: Vec<NodeId>,
    pub fns: Vec<String>,
    /// operations (not operands) in the program
    pub ops: usize,
    pub slices: usize,
    /// the spans that need a site, in the order of the `site` slots in
    /// `code`: (index into code, span)
    pub sites: Vec<(usize, knotkit::span::Span)>,
    /// a check (a slice or a two-array operation) has been placed
    checked: bool,
}

fn kind_op(k: ArrKind) -> Option<(f64, bool)> {
    use ArrKind::*;
    Some(match k {
        Add => (ADD, true),
        Sub => (SUB, true),
        Had => (MUL, true),
        Div => (DIV, true),
        AddS => (ADDS, false),
        SubS => (SUBS, false),
        SubSR => (SUBSR, false),
        Scale => (SCALE, false),
        DivS => (DIVS, false),
        DivSR => (DIVSR, false),
        PowS => (POWS, false),
        _ => return None,
    })
}

/// Is this node an elementwise operation a program can hold?
pub fn fusable(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    match &prog.get(id).kind {
        Viv::Arr { kind } => kind_op(*kind).is_some(),
        Viv::MapMath { cname } => MAP_NAMES.contains(&cname.as_str()),
        _ => false,
    }
}

/// A value whose evaluation can neither stop the run nor be seen: a
/// variable, a literal, a record's field, or arithmetic on those.
fn quiet(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = prog.get(id);
    match &n.kind {
        Viv::Var { .. } | Viv::Num(_) | Viv::Int(_) | Viv::BoolLit(_) => true,
        Viv::Field { .. } | Viv::Neg => n.children.iter().all(|&c| quiet(prog, c)),
        Viv::Bin { op, cplx: false } => {
            matches!(op, crate::ast::BinOp::Add | crate::ast::BinOp::Sub | crate::ast::BinOp::Mul | crate::ast::BinOp::Div) && n.children.iter().all(|&c| quiet(prog, c))
        }
        _ => false,
    }
}

/// The runtimes' limits: arrays (and the depth of the stack), numbers.
pub const MAX_ARRS: usize = 20;
pub const MAX_NUMS: usize = 60;

impl Plan {
    fn fits(&self) -> bool {
        self.arrs.len() <= MAX_ARRS && self.nums.len() <= MAX_NUMS && self.depth() <= MAX_ARRS
    }

    /// The deepest the runtime's stack goes.
    fn depth(&self) -> usize {
        let (mut top, mut most, mut i) = (0usize, 0usize, 0usize);
        while i < self.code.len() {
            let op = self.code[i];
            let (push, pop, w) = match op as u8 {
                0 | 1 => (1, 0, 2),
                2 => (1, 0, 5),
                3..=6 => (0, 1, 2),
                7..=13 => (0, 1, 1),
                14 => (0, 0, 2),
                _ => (0, 0, 1),
            };
            top = top + push - pop;
            most = most.max(top);
            i += w;
        }
        most
    }

    fn operand(&mut self, prog: &IrProgram<Viv>, id: NodeId, arr: bool) -> bool {
        if self.checked && !quiet(prog, id) {
            return false;
        }
        if arr {
            self.code.extend([ARR, self.arrs.len() as f64]);
            self.arrs.push(id);
        } else {
            self.code.extend([NUM, self.nums.len() as f64]);
            self.nums.push(id);
        }
        true
    }

    fn site(&mut self, span: knotkit::span::Span) {
        self.sites.push((self.code.len(), span));
        self.code.push(0.0);
    }

    /// Add an array-valued node, in the order the unfused code evaluates it.
    fn array(&mut self, prog: &IrProgram<Viv>, id: NodeId) -> bool {
        let n = prog.get(id);
        match &n.kind {
            Viv::Arr { kind } => {
                let Some((op, two)) = kind_op(*kind) else { return self.operand(prog, id, true) };
                if !self.array(prog, n.children[0]) {
                    return false;
                }
                let ok = if two { self.array(prog, n.children[1]) } else { self.operand(prog, n.children[1], false) };
                if !ok {
                    return false;
                }
                self.code.push(op);
                if two {
                    self.site(n.span);
                    self.checked = true;
                }
                self.ops += 1;
                true
            }
            Viv::MapMath { cname } if MAP_NAMES.contains(&cname.as_str()) => {
                if !self.array(prog, n.children[0]) {
                    return false;
                }
                let f = match self.fns.iter().position(|g| g == cname) {
                    Some(f) => f,
                    None => {
                        self.fns.push(cname.clone());
                        self.fns.len() - 1
                    }
                };
                self.code.extend([MAP, f as f64]);
                self.ops += 1;
                true
            }
            // a slice of a real array: its base and its bounds are operands,
            // and the slice is read in place
            // (`VIEW`: a slice whose bounds were checked where it was assigned,
            // with the same values: it cannot stop the run, nor can its bounds)
            Viv::Call { callee } if (callee == "viv_getslice" || callee == VIEW_C) && matches!(prog.get(n.children[2]).kind, Viv::Num(w) if w == 1.0) => {
                let pre = callee == VIEW_C;
                let (base, spec) = (n.children[0], n.children[1]);
                let sp = prog.get(spec);
                let Viv::ArrLit { dims } = &sp.kind else { return self.operand(prog, id, true) };
                if dims.len() != 1 || dims[0] % 4 != 0 || dims[0] == 0 {
                    return self.operand(prog, id, true);
                }
                if self.checked && !pre && !quiet(prog, base) {
                    return false;
                }
                let k = self.arrs.len();
                self.arrs.push(base);
                let j = self.nums.len();
                for &c in &sp.children.clone() {
                    if self.checked && !pre && !quiet(prog, c) {
                        return false;
                    }
                    self.nums.push(c);
                }
                self.code.extend([SLICE, k as f64, (dims[0] / 4) as f64, j as f64]);
                self.site(n.span);
                self.checked |= !pre;
                self.slices += 1;
                true
            }
            _ => self.operand(prog, id, true),
        }
    }
}

/// The program for an elementwise tree rooted at `root`, when fusing it
/// saves an array: two operations or more, or one reading a slice.
pub fn plan_tree(prog: &IrProgram<Viv>, root: NodeId) -> Option<Plan> {
    if !fusable(prog, root) {
        return None;
    }
    let mut p = Plan::default();
    if !p.array(prog, root) {
        return None;
    }
    (p.fits() && (p.ops >= 2 || p.slices >= 1)).then_some(p)
}

/// The program for a reduction's operands (`callee` one of `REDUCTIONS`):
/// worth it when an operand is a slice or an elementwise tree.
pub fn plan_reduction(prog: &IrProgram<Viv>, args: &[NodeId], arity: usize) -> Option<Plan> {
    if args.len() < arity {
        return None;
    }
    let mut p = Plan::default();
    for &a in &args[..arity] {
        if !p.array(prog, a) {
            return None;
        }
        p.code.push(END);
    }
    (p.fits() && (p.ops >= 1 || p.slices >= 1)).then_some(p)
}

/// The C runtime: `viv_fz` runs an elementwise program, `viv_fz_red` a
/// reduction over one (or, for `dot`, two). Twin of interp/wasm_rt.rs
/// (`vrt_fz`, `vrt_fz_red`).
pub const C_RUNTIME: &str = r#"/* fused elementwise expressions and reductions over views (lang/viv/src/fuse.rs) */
/* the bounds of a slice read in place where it is used (viv__slice_check) */
static double viv_slicecheck(viv_arr a, viv_arr spec, int site) { viv_slplan p; viv_sl_plan(&p, a, spec, site); return 0.0; }
typedef double (*viv_fz_fn)(double);
#define VIV_FZ_B 256
#define VIV_FZ_DEPTH 24
#define VIV_FZ_NUMS 64
typedef struct { int rank, dims[8]; const double* p; long long st; } viv_fz_v;
/* a slice planned with the slice runtime's checks (a bound out of range stops
   the run here, with its site); 0 when its elements are not evenly spaced */
static int viv_fz_slice(viv_fz_v* e, viv_arr a, const double* spec, int m, int site) {
    viv_arr sp; sp.rank = 1; sp.dims[0] = 4 * m; sp.size = 4 * m; sp.data = (double*)spec;
    viv_slplan p; viv_sl_plan(&p, a, sp, site);
    long long start = 0, st = 1, want = -1;
    for (int d = 0; d < p.rank; d++) start += p.start[d] * p.stride[d];
    for (int d = p.rank - 1; d >= 0; d--) {
        if (p.cnt[d] <= 1) continue;
        long long s = p.step[d] * p.stride[d];
        if (want < 0) st = s;
        else if (s != want) return 0;
        want = s * p.cnt[d];
    }
    e->rank = p.orank;
    for (int d = 0; d < p.orank; d++) e->dims[d] = p.odims[d];
    e->p = p.total ? a.data + start : a.data; e->st = st;
    return 1;
}
static int viv_fz_same(const viv_fz_v* a, const viv_fz_v* b) {
    if (a->rank != b->rank) return 0;
    for (int d = 0; d < a->rank; d++) if (a->dims[d] != b->dims[d]) return 0;
    return 1;
}
/* the walk in the unfused order: shapes (sh), the operands' views (lv), the
   checks. Returns how many values it left, or -1: run the unfused operations */
static int viv_fz_walk(const double* c, int nc, const viv_arr* arrs, const double* nums, viv_fz_v* sh, viv_fz_v* lv) {
    int top = 0, nl = 0;
    for (int i = 0; i < nc;) {
        int op = (int)c[i];
        if (top >= VIV_FZ_DEPTH - 1 || nl >= VIV_FZ_DEPTH) return -1;
        if (op == 0) {
            viv_arr a = arrs[(int)c[i + 1]];
            viv_fz_v* e = &lv[nl++];
            e->rank = a.rank; for (int d = 0; d < a.rank; d++) e->dims[d] = a.dims[d];
            e->p = a.data; e->st = 1; sh[top++] = *e; i += 2;
        } else if (op == 1) {
            sh[top].rank = -1; top++; i += 2;
        } else if (op == 2) {
            if (!viv_fz_slice(&lv[nl], arrs[(int)c[i + 1]], nums + (int)c[i + 3], (int)c[i + 2], (int)c[i + 4])) return -1;
            sh[top++] = lv[nl++]; i += 5;
        } else if (op <= 6) {
            if (!viv_fz_same(&sh[top - 2], &sh[top - 1])) return -1;
            top--; i += 2;
        } else if (op <= 13) {
            top--; i += 1;
        } else if (op == 14) {
            i += 2;
        } else {
            i += 1;
        }
    }
    return top;
}
/* elements [b0, b0 + n) of the values the program leaves, into q[]. Stack
   entry k has two buffers of B numbers; an operation writes the one its
   operand is not in, so no loop reads what it writes (restrict) */
static void viv_fz_block(const double* c, int nc, const double* nums, const viv_fz_fn* fns, const viv_fz_v* lv, long long b0, int n, double* bufs, int B, const double** q) {
    double v[VIV_FZ_DEPTH];
    int top = 0, nl = 0;
#define VIV_FZ_OUT(k) (bufs + (2 * (k) + (q[k] == bufs + 2 * (k) * B)) * B)
    for (int i = 0; i < nc;) {
        int op = (int)c[i];
        if (op == 0 || op == 2) {
            const viv_fz_v* e = &lv[nl++];
            if (e->st == 1) q[top] = e->p + b0;
            else { double* restrict o = bufs + 2 * top * B; const double* p = e->p + b0 * e->st; long long st = e->st; for (int k = 0; k < n; k++) o[k] = p[k * st]; q[top] = o; }
            top++; i += op == 0 ? 2 : 5;
        } else if (op == 1) {
            v[top++] = nums[(int)c[i + 1]]; i += 2;
        } else if (op <= 6) {
            const double* restrict a = q[top - 2]; const double* restrict b = q[top - 1]; double* restrict o = VIV_FZ_OUT(top - 2);
            if (op == 3) for (int k = 0; k < n; k++) o[k] = a[k] + b[k];
            else if (op == 4) for (int k = 0; k < n; k++) o[k] = a[k] - b[k];
            else if (op == 5) for (int k = 0; k < n; k++) o[k] = a[k] * b[k];
            else for (int k = 0; k < n; k++) o[k] = a[k] / b[k];
            q[top - 2] = o; top--; i += 2;
        } else if (op <= 13) {
            const double* restrict a = q[top - 2]; double s = v[top - 1]; double* restrict o = VIV_FZ_OUT(top - 2);
            switch (op) {
            case 7: for (int k = 0; k < n; k++) o[k] = a[k] + s; break;
            case 8: for (int k = 0; k < n; k++) o[k] = a[k] - s; break;
            case 9: for (int k = 0; k < n; k++) o[k] = s - a[k]; break;
            case 10: for (int k = 0; k < n; k++) o[k] = a[k] * s; break;
            case 11: for (int k = 0; k < n; k++) o[k] = a[k] / s; break;
            case 12: for (int k = 0; k < n; k++) o[k] = s / a[k]; break;
            default: for (int k = 0; k < n; k++) o[k] = pow(a[k], s); break;
            }
            q[top - 2] = o; top--; i += 1;
        } else if (op == 14) {
            viv_fz_fn f = fns[(int)c[i + 1]]; const double* restrict a = q[top - 1]; double* restrict o = VIV_FZ_OUT(top - 1);
            for (int k = 0; k < n; k++) o[k] = f(a[k]);
            q[top - 1] = o; i += 2;
        } else {
            i += 1;
        }
    }
#undef VIV_FZ_OUT
}
/* the unfused operations, one at a time, in the unfused order: what they
   broadcast or stop with is what the program did before it was fused */
static int viv_fz_mat(const double* c, int nc, const viv_arr* arrs, const double* nums, const viv_fz_fn* fns, viv_arr* out) {
    viv_arr a[VIV_FZ_DEPTH]; double v[VIV_FZ_DEPTH];
    int top = 0;
    for (int i = 0; i < nc;) {
        int op = (int)c[i];
        if (op == 0) { a[top++] = arrs[(int)c[i + 1]]; i += 2; }
        else if (op == 1) { v[top++] = nums[(int)c[i + 1]]; i += 2; }
        else if (op == 2) {
            int m = (int)c[i + 2];
            viv_arr sp; sp.rank = 1; sp.dims[0] = 4 * m; sp.size = 4 * m; sp.data = (double*)(nums + (int)c[i + 3]);
            a[top++] = viv_getslice(arrs[(int)c[i + 1]], sp, 1.0, (int)c[i + 4]); i += 5;
        } else if (op <= 6) {
            viv_arr (*f)(viv_arr, viv_arr) = op == 3 ? viv_arr_add : op == 4 ? viv_arr_sub : op == 5 ? viv_arr_had : viv_arr_div;
            a[top - 2] = viv_arr_chk(f, a[top - 2], a[top - 1], 0, (int)c[i + 1]); top--; i += 2;
        } else if (op <= 13) {
            viv_arr x = a[top - 2]; double s = v[top - 1];
            a[top - 2] = op == 7 ? viv_arr_adds(x, s) : op == 8 ? viv_arr_subs(x, s) : op == 9 ? viv_arr_subsr(x, s)
                : op == 10 ? viv_arr_scale(x, s) : op == 11 ? viv_arr_divs(x, s) : op == 12 ? viv_arr_divsr(x, s) : viv_arr_pows(x, s);
            top--; i += 1;
        } else if (op == 14) { a[top - 1] = viv_arr_map(fns[(int)c[i + 1]], a[top - 1]); i += 2; }
        else i += 1;
    }
    for (int k = 0; k < top; k++) out[k] = a[k];
    return top;
}
static viv_arr viv_fz_run(const double* c, int nc, const viv_arr* arrs, const double* nums, const viv_fz_fn* fns) {
    viv_fz_v sh[VIV_FZ_DEPTH], lv[VIV_FZ_DEPTH];
    if (viv_fz_walk(c, nc, arrs, nums, sh, lv) != 1) { viv_arr r; viv_fz_mat(c, nc, arrs, nums, fns, &r); return r; }
    viv_arr r = viv_arr_new(sh[0].rank, sh[0].dims);
    /* the blocks are as long as the array, to VIV_FZ_B: a short array keeps a
       short frame */
    int B = r.size < VIV_FZ_B ? (r.size ? r.size : 1) : VIV_FZ_B;
    double bufs[2 * VIV_FZ_DEPTH * B]; const double* q[VIV_FZ_DEPTH];
    for (long long b0 = 0; b0 < r.size; b0 += B) {
        int n = r.size - b0 < B ? (int)(r.size - b0) : B;
        viv_fz_block(c, nc, nums, fns, lv, b0, n, bufs, B, q);
        memcpy(r.data + b0, q[0], (size_t)n * sizeof(double));
    }
    return r;
}
/* kind: 0 sum (NumPy's order), 1 ordered sum, 2 prod, 3 mean, 4 norm, 5 dot,
   6 max, 7 min -- each in the order of the function it stands for */
static double viv_fz_red_run(int kind, const double* c, int nc, const viv_arr* arrs, const double* nums, const viv_fz_fn* fns, int site) {
    viv_fz_v sh[VIV_FZ_DEPTH], lv[VIV_FZ_DEPTH];
    int left = viv_fz_walk(c, nc, arrs, nums, sh, lv);
    long long size = 1;
    if (left >= 1) for (int d = 0; d < sh[0].rank; d++) size *= sh[0].dims[d];
    if (left == 2 && !(sh[0].rank == 1 && sh[1].rank == 1 && sh[0].dims[0] == sh[1].dims[0])) left = -1;
    if (left != (kind == 5 ? 2 : 1)) {
        viv_arr r[2]; viv_fz_mat(c, nc, arrs, nums, fns, r);
        switch (kind) {
        case 0: return viv_sum(r[0]);
        case 1: return viv_ar___sum_ordered_1(r[0], site);
        case 2: return viv_ar_prod_1(r[0], site);
        case 3: return viv_ar_mean_1(r[0], site);
        case 4: return viv_ar_norm_1(r[0], site);
        case 5: return viv_ar_dot_2(r[0], r[1], site);
        case 6: return viv_maxval(r[0]);
        default: return viv_minval(r[0]);
        }
    }
    int B = size < VIV_FZ_B ? (size ? (int)size : 1) : VIV_FZ_B;
    double bufs[2 * VIV_FZ_DEPTH * B]; const double* q[VIV_FZ_DEPTH];
    double t = kind == 2 ? 1.0 : kind == 6 ? -INFINITY : kind == 7 ? INFINITY : 0.0;
    if (kind == 0 || kind == 3) {
        /* NumPy's order: blocks of 8192 entries, each by the pairwise sum */
        double chunk[size < 8192 ? (size ? size : 1) : 8192];
        for (long long c0 = 0; c0 < size; c0 += 8192) {
            int len = size - c0 < 8192 ? (int)(size - c0) : 8192;
            for (int b = 0; b < len; b += B) {
                int n = len - b < B ? len - b : B;
                viv_fz_block(c, nc, nums, fns, lv, c0 + b, n, bufs, B, q);
                memcpy(chunk + b, q[0], (size_t)n * sizeof(double));
            }
            t += viv_sum_pw(chunk, len);
        }
        return kind == 3 ? t / (double)size : t;
    }
    for (long long b0 = 0; b0 < size; b0 += B) {
        int n = size - b0 < B ? (int)(size - b0) : B;
        viv_fz_block(c, nc, nums, fns, lv, b0, n, bufs, B, q);
        const double* x = q[0];
        switch (kind) {
        case 1: for (int k = 0; k < n; k++) t += x[k]; break;
        case 2: for (int k = 0; k < n; k++) t *= x[k]; break;
        case 4: for (int k = 0; k < n; k++) t += x[k] * x[k]; break;
        case 5: { const double* y = q[1]; for (int k = 0; k < n; k++) t += x[k] * y[k]; break; }
        case 6: for (int k = 0; k < n; k++) t = fmax(t, x[k]); break;
        default: for (int k = 0; k < n; k++) t = fmin(t, x[k]); break;
        }
    }
    return kind == 4 ? sqrt(t) : t;
}
/* the calls the compiled program makes: the program, its functions, then na
   arrays and nn numbers */
#define VIV_FZ_ARGS(na, nn) viv_arr arrs[VIV_FZ_DEPTH]; double nums[VIV_FZ_NUMS]; \
    va_list ap; va_start(ap, nn); \
    for (int k = 0; k < na; k++) arrs[k] = va_arg(ap, viv_arr); \
    for (int k = 0; k < nn; k++) nums[k] = va_arg(ap, double); \
    va_end(ap)
/* the walk and the result's block for a tree whose loop the compiler wrote
   out (lower.rs, fuse_assign_c): the operands' views in lv, the result,
   not yet written, in r. 0 when the walk asks for the unfused operations, which
   viv_fz_run then does (its walk is the same, and checks nothing this one
   did not, so it stops where this one would have) */
static int viv_fz_prep(const double* c, int nc, const viv_arr* arrs, const double* nums, viv_fz_v* lv, viv_arr* r) {
    viv_fz_v sh[VIV_FZ_DEPTH];
    if (viv_fz_walk(c, nc, arrs, nums, sh, lv) != 1 || sh[0].rank < 0) return 0;
    /* the generated loop writes every element */
    *r = viv_arr_new_nz(sh[0].rank, sh[0].dims);
    return 1;
}
static viv_arr viv_fz(const double* c, int nc, const viv_fz_fn* fns, int na, int nn, ...) {
    VIV_FZ_ARGS(na, nn);
    return viv_fz_run(c, nc, arrs, nums, fns);
}
static double viv_fz_red(int kind, int site, const double* c, int nc, const viv_fz_fn* fns, int na, int nn, ...) {
    VIV_FZ_ARGS(na, nn);
    return viv_fz_red_run(kind, c, nc, arrs, nums, fns, site);
}"#;

/// A tree's element as one C expression, for the loop the compiler writes
/// out in place of `viv_fz` (lower.rs, `fuse_assign_c`): leaf `j` (the
/// `j`-th ARR or SLICE in the program, the order `viv_fz_walk` fills its
/// views) is `leaf(j)`, number `k` is `viv_fzn[k]`. Each operation is the one
/// `viv_fz_block` does, on the same operands, so with contraction off every
/// element has the same bits. Returns the expression and the leaf count, and
/// whether a leaf is a slice (whose elements may be spaced).
pub fn element_expr(plan: &Plan, leaf: &dyn Fn(usize) -> String) -> Option<(String, usize, bool)> {
    let c = &plan.code;
    let mut st: Vec<String> = Vec::new();
    let (mut nl, mut sliced, mut i) = (0usize, false, 0usize);
    while i < c.len() {
        let op = c[i];
        if op == ARR || op == SLICE {
            st.push(leaf(nl));
            nl += 1;
            sliced |= op == SLICE;
            i += if op == ARR { 2 } else { 5 };
        } else if op == NUM {
            st.push(format!("viv_fzn[{}]", c[i + 1] as usize));
            i += 2;
        } else if op <= DIV {
            let b = st.pop()?;
            let a = st.pop()?;
            let o = if op == ADD { "+" } else if op == SUB { "-" } else if op == MUL { "*" } else { "/" };
            st.push(format!("({a} {o} {b})"));
            i += 2;
        } else if op <= POWS {
            let s = st.pop()?;
            let a = st.pop()?;
            st.push(match op {
                x if x == ADDS => format!("({a} + {s})"),
                x if x == SUBS => format!("({a} - {s})"),
                x if x == SUBSR => format!("({s} - {a})"),
                x if x == SCALE => format!("({a} * {s})"),
                x if x == DIVS => format!("({a} / {s})"),
                x if x == DIVSR => format!("({s} / {a})"),
                _ => format!("pow({a}, {s})"),
            });
            i += 1;
        } else if op == MAP {
            let a = st.pop()?;
            st.push(format!("{}({a})", plan.fns.get(c[i + 1] as usize)?));
            i += 2;
        } else {
            return None;
        }
    }
    if st.len() != 1 {
        return None;
    }
    Some((st.pop()?, nl, sliced))
}

// ---- views of slice locals (the AST, before lowering; `crate::build_ir`) ----
//
// `x = d[o:o + n]` followed by `sum(x, ordered=true)` and `dot(x, y)` copies
// the slice once. When every use of `x` is one a program above reads in place
// (an operand of a reduction, of elementwise arithmetic, of mapped math),
// the slice is written at each use instead, and the assignment becomes a
// check of the bounds (`viv__slice_check`), so a bound out of range stops the
// run where it did. Nothing between the assignment and the last use may write
// an array, call a def of the program, or assign the variables the slice
// reads, so each use sees the elements the copy held. Only the compiled
// engines see this; the body hashes (`build_ir_plain`) and the interpreter
// read the program as written.

use crate::ast::{BinOp, Expr, Item, Program, Stmt};
use std::collections::HashSet;

/// The callee of the bounds check left where a substituted slice was.
pub const CHECK: &str = "viv__slice_check";
/// The callee of a substituted slice at a use: `viv__slice_get`, whose bounds
/// were checked (`CHECK`); the lowering writes it as `VIEW_C`.
pub const VIEW: &str = "viv__slice_view";
pub const VIEW_C: &str = "viv_getslice_v";

const REDUCE: &[&str] = &["sum", "__sum_ordered", "prod", "mean", "norm", "max", "min"];
const MAPPED: &[&str] = &["abs", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "floor", "ceil"];

pub(crate) fn quiet_ast(e: &Expr) -> bool {
    match e {
        Expr::Var { .. } | Expr::Num { .. } => true,
        Expr::Field { base, .. } | Expr::Neg { operand: base, .. } => quiet_ast(base),
        Expr::Binary { op, lhs, rhs, .. } => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) && quiet_ast(lhs) && quiet_ast(rhs),
        _ => false,
    }
}

fn root_var(e: &Expr) -> Option<&str> {
    match e {
        Expr::Var { name, .. } => Some(name),
        Expr::Field { base, .. } => root_var(base),
        _ => None,
    }
}

fn vars_of(e: &Expr, out: &mut HashSet<String>) {
    each_child(e, &mut |c| vars_of(c, out));
    if let Expr::Var { name, .. } = e {
        out.insert(name.clone());
    }
}

fn each_child(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    match e {
        Expr::ArrLit { elems, .. } => elems.iter().for_each(|x| f(x)),
        Expr::Index { base, indices, .. } => {
            f(base);
            indices.iter().for_each(|x| f(x));
        }
        Expr::Lambda { body, .. } => f(body),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => f(operand),
        Expr::Binary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Expr::Call { args, .. } => args.iter().for_each(|x| f(x)),
        Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| f(x)),
        Expr::Field { base, .. } => f(base),
        _ => {}
    }
}

fn each_child_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::ArrLit { elems, .. } => elems.iter_mut().for_each(|x| f(x)),
        Expr::Index { base, indices, .. } => {
            f(base);
            indices.iter_mut().for_each(|x| f(x));
        }
        Expr::Lambda { body, .. } => f(body),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => f(operand),
        Expr::Binary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        Expr::Call { args, .. } => args.iter_mut().for_each(|x| f(x)),
        Expr::RecordLit { fields, .. } => fields.iter_mut().for_each(|(_, _, x)| f(x)),
        Expr::Field { base, .. } => f(base),
        _ => {}
    }
}

/// Every expression of a statement (not of nested blocks), and its blocks.
fn stmt_parts(s: &Stmt) -> (Vec<&Expr>, Vec<&Vec<Stmt>>) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => (vec![value], vec![]),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => (vec![expr], vec![]),
        Stmt::Test { body, .. } => (vec![], vec![body]),
        Stmt::SetIndex { base, indices, value, .. } => {
            let mut v: Vec<&Expr> = vec![base];
            v.extend(indices.iter());
            v.push(value);
            (v, vec![])
        }
        Stmt::SetField { base, value, .. } => (vec![base, value], vec![]),
        Stmt::If { cond, then_body, else_body, .. } => (vec![cond], vec![then_body, else_body]),
        Stmt::While { cond, body, .. } => (vec![cond], vec![body]),
        Stmt::For { bound, body, .. } => (vec![bound], vec![body]),
        _ => (vec![], vec![]),
    }
}

fn stmt_parts_mut(s: &mut Stmt) -> (Vec<&mut Expr>, Vec<&mut Vec<Stmt>>) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => (vec![value], vec![]),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => (vec![expr], vec![]),
        Stmt::Test { body, .. } => (vec![], vec![body]),
        Stmt::SetIndex { base, indices, value, .. } => {
            let mut v: Vec<&mut Expr> = vec![&mut **base];
            v.extend(indices.iter_mut());
            v.push(value);
            (v, vec![])
        }
        Stmt::SetField { base, value, .. } => (vec![&mut **base, value], vec![]),
        Stmt::If { cond, then_body, else_body, .. } => (vec![cond], vec![then_body, else_body]),
        Stmt::While { cond, body, .. } => (vec![cond], vec![body]),
        Stmt::For { bound, body, .. } => (vec![bound], vec![body]),
        _ => (vec![], vec![]),
    }
}

/// Uses of `name` in an expression: (all, those in a position read in place).
fn uses_expr(e: &Expr, name: &str, ok_here: bool, n: &mut (usize, usize)) {
    if let Expr::Var { name: v, .. } = e {
        if v == name {
            n.0 += 1;
            n.1 += ok_here as usize;
        }
        return;
    }
    let child_ok = match e {
        Expr::Binary { op, .. } => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div),
        Expr::Neg { .. } => true,
        Expr::Call { callee, args, .. } => {
            (REDUCE.contains(&callee.as_str()) || MAPPED.contains(&callee.as_str())) && args.len() == 1 || callee == "dot" && args.len() == 2
        }
        _ => false,
    };
    if let Expr::Call { callee, args, .. } = e {
        if callee == "pow" && args.len() == 2 {
            uses_expr(&args[0], name, true, n);
            uses_expr(&args[1], name, false, n);
            return;
        }
    }
    each_child(e, &mut |c| uses_expr(c, name, child_ok, n));
}

fn uses_stmts(stmts: &[Stmt], name: &str, n: &mut (usize, usize)) {
    for s in stmts {
        let (es, bs) = stmt_parts(s);
        for e in es {
            uses_expr(e, name, false, n);
        }
        for b in bs {
            uses_stmts(b, name, n);
        }
        if let Stmt::Lens { .. } = s {
            // a lens statement may read anything: count it as a use that is not
            // read in place
            n.0 += 1;
        }
    }
}

fn assigns(stmts: &[Stmt], name: &str) -> usize {
    let mut k = 0;
    for s in stmts {
        match s {
            Stmt::Assign { name: n, .. } if n == name => k += 1,
            Stmt::For { var, .. } if var == name => k += 1,
            _ => {}
        }
        for b in stmt_parts(s).1 {
            k += assigns(b, name);
        }
    }
    k
}

/// Nothing in `stmts` writes an array, calls one of the program's defs, or
/// assigns one of `roots`.
fn region_safe(stmts: &[Stmt], roots: &HashSet<String>, defs: &HashSet<String>) -> bool {
    fn expr_safe(e: &Expr, defs: &HashSet<String>) -> bool {
        let here = match e {
            Expr::Call { callee, .. } => {
                !defs.contains(callee)
                    && callee != crate::interp::slices::SET
                    && callee != "fill"
                    && !crate::interp::arrays::SIGS.iter().any(|s| s.name == callee && s.writes.is_some())
            }
            Expr::Lambda { .. } => false,
            _ => true,
        };
        let mut ok = here;
        each_child(e, &mut |c| ok = ok && expr_safe(c, defs));
        ok
    }
    stmts.iter().all(|s| {
        let fine = match s {
            Stmt::Assign { name, .. } => !roots.contains(name),
            Stmt::For { var, .. } => !roots.contains(var),
            Stmt::SetIndex { .. } | Stmt::SetField { .. } | Stmt::Lens { .. } => false,
            _ => true,
        };
        let (es, bs) = stmt_parts(s);
        fine && es.iter().all(|e| expr_safe(e, defs)) && bs.iter().all(|b| region_safe(b, roots, defs))
    })
}

fn replace_var(e: &mut Expr, name: &str, with: &Expr) {
    if matches!(e, Expr::Var { name: v, .. } if v == name) {
        *e = with.clone();
        return;
    }
    each_child_mut(e, &mut |c| replace_var(c, name, with));
}

fn replace_stmts(stmts: &mut [Stmt], name: &str, with: &Expr) {
    for s in stmts {
        let (es, bs) = stmt_parts_mut(s);
        for e in es {
            replace_var(e, name, with);
        }
        for b in bs {
            replace_stmts(b, name, with);
        }
    }
}

fn block(stmts: &mut Vec<Stmt>, body: &[Stmt], defs: &HashSet<String>, globals: &HashSet<String>, done: &mut Vec<String>) {
    for k in 0..stmts.len() {
        let cand = match &stmts[k] {
            Stmt::Assign { name, value: value @ Expr::Call { callee, args, .. }, .. }
                if callee == crate::interp::slices::GET && args.len() == 2 && !globals.contains(name) =>
            {
                let Expr::ArrLit { elems, .. } = &args[1] else { continue };
                if root_var(&args[0]).is_none() || !elems.iter().all(quiet_ast) {
                    continue;
                }
                (name.clone(), value.clone())
            }
            _ => continue,
        };
        let (name, slice) = cand;
        if done.contains(&name) || assigns(body, &name) != 1 {
            continue;
        }
        let mut total = (0, 0);
        uses_stmts(body, &name, &mut total);
        if total.0 == 0 || total.0 != total.1 {
            continue;
        }
        // the uses all lie in this block, after the assignment
        let mut last = None;
        let mut here = (0, 0);
        for (j, s) in stmts.iter().enumerate().skip(k + 1) {
            let mut n = (0, 0);
            uses_stmts(std::slice::from_ref(s), &name, &mut n);
            if n.0 > 0 {
                last = Some(j);
                here.0 += n.0;
            }
        }
        let Some(last) = last else { continue };
        if here.0 != total.0 {
            continue;
        }
        let mut roots = HashSet::new();
        vars_of(&slice, &mut roots);
        if !region_safe(&stmts[k + 1..=last], &roots, defs) {
            continue;
        }
        let Expr::Call { args, span, .. } = &slice else { continue };
        let span = *span;
        stmts[k] = Stmt::ExprStmt { expr: Expr::Call { callee: CHECK.to_string(), args: args.clone(), span }, span };
        let mut view = slice.clone();
        if let Expr::Call { callee, .. } = &mut view {
            *callee = VIEW.to_string();
        }
        replace_stmts(&mut stmts[k + 1..=last], &name, &view);
        done.push(name);
    }
    for s in stmts.iter_mut() {
        for b in stmt_parts_mut(s).1 {
            block(b, body, defs, globals, done);
        }
    }
}

/// The pass (see above). `VIV_NOFUSE` turns it off with the fusion.
pub fn substitute(mut prog: Program) -> Program {
    if !enabled() {
        return prog;
    }
    let defs: HashSet<String> = prog.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    let globals: HashSet<String> = prog.items.iter().filter_map(|i| if let Item::Stmt(Stmt::Assign { name, .. }) = i { Some(name.clone()) } else { None }).collect();
    for item in &mut prog.items {
        if let Item::Def(d) = item {
            let body = d.body.clone();
            let mut done = Vec::new();
            block(&mut d.body, &body, &defs, &globals, &mut done);
        }
    }
    prog
}
