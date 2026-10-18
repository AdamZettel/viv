// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The form compiler: a weak-form `Problem` -> Viv source.
//!
//! Two paths:
//!  * Linear (`bilinear`/`linear`): affine-P1 closed-form stiffness + one-point load.
//!  * Nonlinear (`residual`): a semilinear reaction-diffusion residual
//!    `F = ∫ κ ∇u·∇v + S(u,x)·v`. The Jacobian `J = ∫ κ ∇du·∇v + S'(u,x)·du·v` is
//!    formed by **symbolically differentiating** S w.r.t. u (`diff_u`), and a Newton
//!    loop is emitted. The reaction/source is integrated with a 3-point rule that
//!    evaluates the current field u_h at quadrature points (the step beyond the
//!    linear closed form). This is UFL's `derivative(F, u)`, scoped to the
//!    semilinear case (nonlinear diffusion and grad-nonlinearities are deferred).

use crate::expr::E;
use crate::parse::{Forms, Problem};

pub fn compile(problem: &Problem) -> Result<String, String> {
    match &problem.forms {
        Forms::Linear { bilinear, linear } => compile_linear(problem, bilinear, linear),
        Forms::Nonlinear { residual, newton_max } => compile_nonlinear(problem, residual, *newton_max),
    }
}

// ───────────────────────────── linear path ─────────────────────────────────

fn compile_linear(problem: &Problem, bilinear: &E, linear: &E) -> Result<String, String> {
    let stiff_coeff = classify_bilinear(bilinear)?;
    let load_coeff = classify_linear(linear)?;

    let mut setup = String::new();
    let ke = if is_one(&stiff_coeff) {
        "(bv[a] * bv[c] + cv[a] * cv[c]) / (4.0 * area)".to_string()
    } else {
        setup.push_str(&format!("    kappa = {}\n", emit_scalar(&stiff_coeff, None, "xc", "yc", &problem.coeffs)?));
        "kappa * (bv[a] * bv[c] + cv[a] * cv[c]) / (4.0 * area)".to_string()
    };
    setup.push_str(&format!("    fval = {}", emit_scalar(&load_coeff, None, "xc", "yc", &problem.coeffs)?));
    let exact = emit_scalar(&problem.exact, None, "node_x[gI]", "node_y[gI]", &problem.coeffs)?;

    Ok(LINEAR_TEMPLATE
        .replace("@@MESH@@", SOLVE_AND_MESH)
        .replace("@@N@@", &problem.mesh_n.to_string())
        .replace("@@SETUP@@", &setup)
        .replace("@@KE@@", &ke)
        .replace("@@EXACT@@", &exact)
        .replace("@@BC@@", &format!("{:?}", problem.dirichlet)))
}

/// `[scalar *] dot(grad(u), grad(v))` -> the scalar coefficient (or `1.0`).
fn classify_bilinear(form: &E) -> Result<E, String> {
    let mut facs = Vec::new();
    flatten_product(form, &mut facs);
    let mut coeff = Vec::new();
    let mut saw = false;
    for f in facs {
        if is_stiffness(&f) {
            if saw {
                return Err("v1 supports a single `dot(grad(u), grad(v))` term".into());
            }
            saw = true;
        } else if mentions_arg(&f) {
            return Err(format!("unsupported bilinear factor: {f:?}"));
        } else {
            coeff.push(f);
        }
    }
    if !saw {
        return Err("bilinear form must contain `dot(grad(u), grad(v))`".into());
    }
    Ok(product(coeff))
}

/// `[scalar *] v` -> the scalar coefficient.
fn classify_linear(form: &E) -> Result<E, String> {
    let mut facs = Vec::new();
    flatten_product(form, &mut facs);
    let mut coeff = Vec::new();
    let mut saw = false;
    for f in facs {
        if f == E::Var("v".into()) {
            if saw {
                return Err("`v` appears more than once in the linear form".into());
            }
            saw = true;
        } else if mentions_arg(&f) {
            return Err(format!("unsupported linear factor (v1 = `f * v`): {f:?}"));
        } else {
            coeff.push(f);
        }
    }
    if !saw {
        return Err("linear form must be linear in the test function `v`".into());
    }
    Ok(product(coeff))
}

// ───────────────────────────── nonlinear path ──────────────────────────────

fn compile_nonlinear(problem: &Problem, residual: &E, newton_max: u32) -> Result<String, String> {
    if problem.dirichlet != 0.0 {
        return Err("nonlinear forms support only homogeneous Dirichlet (`dirichlet: 0.0`) for now".into());
    }
    // Split the residual into signed terms, each linear in `v`.
    let mut terms = Vec::new();
    flatten_sum(residual, 1, &mut terms);

    let mut kappa = 0.0f64; // total constant diffusion coefficient on ∇u·∇v
    let mut s_terms: Vec<E> = Vec::new(); // signed coefficients of `v` -> the reaction/source S(u,x)
    for (term, sign) in terms {
        let mut facs = Vec::new();
        flatten_product(&term, &mut facs);
        let has_stiff = facs.iter().any(is_stiffness);
        let has_test = facs.iter().any(|f| *f == E::Var("v".into()));
        if has_stiff {
            let mut coeff = 1.0;
            for f in &facs {
                if is_stiffness(f) {
                    continue;
                }
                match f {
                    E::Num(v) => coeff *= v,
                    _ => return Err("the diffusion coefficient on `dot(grad(u),grad(v))` must be constant (v2)".into()),
                }
            }
            kappa += sign as f64 * coeff;
        } else if has_test {
            let mut coeff_facs = Vec::new();
            for f in facs {
                if f == E::Var("v".into()) {
                    continue;
                }
                if has_form_construct(&f) {
                    return Err(format!("a reaction/source coefficient may depend on u, x, y only: {f:?}"));
                }
                coeff_facs.push(f);
            }
            let mut c = product(coeff_facs);
            if sign < 0 {
                c = E::Neg(Box::new(c));
            }
            s_terms.push(c);
        } else {
            return Err("each residual term must contain `dot(grad(u),grad(v))` or the test `v`".into());
        }
    }

    // S(u,x) and its symbolic u-derivative S'(u,x) -- the Jacobian's reaction kernel.
    let s = sum(s_terms);
    let sprime = diff_u(&s);

    let sval = emit_scalar(&s, Some("uh"), "xq", "yq", &problem.coeffs)?;
    let spexpr = emit_scalar(&sprime, Some("uh"), "xq", "yq", &problem.coeffs)?;
    let exact = emit_scalar(&problem.exact, None, "node_x[gI]", "node_y[gI]", &problem.coeffs)?;

    Ok(NONLINEAR_TEMPLATE
        .replace("@@MESH@@", SOLVE_AND_MESH)
        .replace("@@N@@", &problem.mesh_n.to_string())
        .replace("@@MAXIT@@", &newton_max.to_string())
        .replace("@@KAPPA@@", &format!("{kappa:?}"))
        .replace("@@SVAL@@", &sval)
        .replace("@@SPRIME@@", &spexpr)
        .replace("@@EXACT@@", &exact))
}

// ───────────────────────────── symbolic differentiation ────────────────────

/// d/du of a scalar expression. `u` is the differentiation variable; `x`, `y`,
/// numbers, and (u-independent) coefficients are constants. Uses simplifying
/// constructors so the result stays small and correct.
fn diff_u(e: &E) -> E {
    match e {
        E::Num(_) => E::Num(0.0),
        E::Var(n) if n == "u" => E::Num(1.0),
        E::Var(_) => E::Num(0.0),
        E::Neg(a) => e_neg(diff_u(a)),
        E::Bin('+', a, b) => e_add(diff_u(a), diff_u(b)),
        E::Bin('-', a, b) => e_sub(diff_u(a), diff_u(b)),
        E::Bin('*', a, b) => e_add(e_mul(diff_u(a), (**b).clone()), e_mul((**a).clone(), diff_u(b))),
        E::Bin('/', a, b) => {
            // (a'b - a b') / b^2
            let num = e_sub(e_mul(diff_u(a), (**b).clone()), e_mul((**a).clone(), diff_u(b)));
            E::Bin('/', Box::new(num), Box::new(E::Bin('*', b.clone(), b.clone())))
        }
        E::Call(name, args) if name == "exp" && args.len() == 1 => e_mul(E::Call("exp".into(), args.clone()), diff_u(&args[0])),
        E::Call(name, args) if name == "sqrt" && args.len() == 1 => {
            // a' / (2 sqrt(a))
            let denom = e_mul(E::Num(2.0), E::Call("sqrt".into(), args.clone()));
            E::Bin('/', Box::new(diff_u(&args[0])), Box::new(denom))
        }
        E::Call(name, args) if name == "pow" && args.len() == 2 => {
            // d(a^n) = n a^(n-1) a'  (n constant)
            if let E::Num(n) = args[1] {
                let inner = E::Call("pow".into(), vec![args[0].clone(), E::Num(n - 1.0)]);
                e_mul(e_mul(E::Num(n), inner), diff_u(&args[0]))
            } else {
                // Non-constant exponent: not differentiated in v2.
                E::Num(0.0)
            }
        }
        E::Bin(op, ..) => panic!("unexpected operator `{op}` in scalar derivative"),
        E::Grad(_) | E::Dot(..) | E::Call(..) => E::Num(0.0),
    }
}

fn e_neg(a: E) -> E {
    if is_zero(&a) {
        a
    } else {
        E::Neg(Box::new(a))
    }
}
fn e_add(a: E, b: E) -> E {
    if is_zero(&a) {
        b
    } else if is_zero(&b) {
        a
    } else {
        E::Bin('+', Box::new(a), Box::new(b))
    }
}
fn e_sub(a: E, b: E) -> E {
    if is_zero(&b) {
        a
    } else if is_zero(&a) {
        e_neg(b)
    } else {
        E::Bin('-', Box::new(a), Box::new(b))
    }
}
fn e_mul(a: E, b: E) -> E {
    if is_zero(&a) || is_zero(&b) {
        E::Num(0.0)
    } else if is_one(&a) {
        b
    } else if is_one(&b) {
        a
    } else {
        E::Bin('*', Box::new(a), Box::new(b))
    }
}

// ───────────────────────────── tree helpers ────────────────────────────────

fn flatten_product(e: &E, out: &mut Vec<E>) {
    if let E::Bin('*', a, b) = e {
        flatten_product(a, out);
        flatten_product(b, out);
    } else {
        out.push(e.clone());
    }
}

/// Split a sum/difference into signed leaf terms.
fn flatten_sum(e: &E, sign: i32, out: &mut Vec<(E, i32)>) {
    match e {
        E::Bin('+', a, b) => {
            flatten_sum(a, sign, out);
            flatten_sum(b, sign, out);
        }
        E::Bin('-', a, b) => {
            flatten_sum(a, sign, out);
            flatten_sum(b, -sign, out);
        }
        E::Neg(a) => flatten_sum(a, -sign, out),
        _ => out.push((e.clone(), sign)),
    }
}

fn product(mut facs: Vec<E>) -> E {
    match facs.len() {
        0 => E::Num(1.0),
        1 => facs.pop().unwrap(),
        _ => {
            let mut it = facs.into_iter();
            let mut acc = it.next().unwrap();
            for f in it {
                acc = E::Bin('*', Box::new(acc), Box::new(f));
            }
            acc
        }
    }
}

fn sum(terms: Vec<E>) -> E {
    let mut acc = E::Num(0.0);
    for t in terms {
        acc = e_add(acc, t);
    }
    acc
}

fn is_stiffness(e: &E) -> bool {
    if let E::Dot(a, b) = e {
        if let (E::Grad(ga), E::Grad(gb)) = (&**a, &**b) {
            return matches!((as_arg(ga), as_arg(gb)), (Some('u'), Some('v')) | (Some('v'), Some('u')));
        }
    }
    false
}

fn as_arg(e: &E) -> Option<char> {
    match e {
        E::Var(n) if n == "u" => Some('u'),
        E::Var(n) if n == "v" => Some('v'),
        _ => None,
    }
}

fn is_one(e: &E) -> bool {
    matches!(e, E::Num(v) if *v == 1.0)
}
fn is_zero(e: &E) -> bool {
    matches!(e, E::Num(v) if *v == 0.0)
}

/// References a trial/test argument (`u`/`v`, bare or under grad)?
fn mentions_arg(e: &E) -> bool {
    match e {
        E::Var(n) => n == "u" || n == "v",
        E::Num(_) => false,
        E::Grad(a) | E::Neg(a) => mentions_arg(a),
        E::Dot(a, b) | E::Bin(_, a, b) => mentions_arg(a) || mentions_arg(b),
        E::Call(_, args) => args.iter().any(mentions_arg),
    }
}

/// Contains a form construct that may not appear in a scalar coefficient
/// (`grad`, `dot`, or the test function `v`)? (`u` is allowed — the field.)
fn has_form_construct(e: &E) -> bool {
    match e {
        E::Var(n) => n == "v",
        E::Num(_) => false,
        E::Grad(_) | E::Dot(..) => true,
        E::Neg(a) => has_form_construct(a),
        E::Bin(_, a, b) => has_form_construct(a) || has_form_construct(b),
        E::Call(_, args) => args.iter().any(has_form_construct),
    }
}

// ───────────────────────────── scalar emission ─────────────────────────────

/// Emit a scalar expression as Viv, binding `u`/`x`/`y` to the given Viv
/// expressions and inlining coefficient definitions.
fn emit_scalar(e: &E, u: Option<&str>, x: &str, y: &str, coeffs: &[(String, E)]) -> Result<String, String> {
    match e {
        E::Num(v) => Ok(format!("{v:?}")),
        E::Var(n) if n == "u" => u.map(str::to_string).ok_or_else(|| "`u` is not valid here".into()),
        E::Var(n) if n == "x" => Ok(x.to_string()),
        E::Var(n) if n == "y" => Ok(y.to_string()),
        E::Var(n) if n == "v" => Err("`v` is not valid in a scalar expression".into()),
        E::Var(n) => {
            let def = coeffs
                .iter()
                .find(|(name, _)| name == n)
                .map(|(_, def)| def)
                .ok_or_else(|| format!("unknown coefficient or variable `{n}`"))?;
            Ok(format!("({})", emit_scalar(def, u, x, y, coeffs)?))
        }
        E::Neg(a) => Ok(format!("(-({}))", emit_scalar(a, u, x, y, coeffs)?)),
        E::Bin(op, a, b) => Ok(format!("({} {} {})", emit_scalar(a, u, x, y, coeffs)?, op, emit_scalar(b, u, x, y, coeffs)?)),
        E::Call(name, args) => {
            let parts: Result<Vec<String>, String> = args.iter().map(|a| emit_scalar(a, u, x, y, coeffs)).collect();
            Ok(format!("{name}({})", parts?.join(", ")))
        }
        E::Grad(_) | E::Dot(..) => Err("grad/dot are not valid in a scalar expression".into()),
    }
}

// ───────────────────────────── templates ───────────────────────────────────

const SOLVE_AND_MESH: &str = r#"def solve_linear(A, b, x, n) {
    for k to n {
        piv = k
        big = A[k, k]
        if big < 0.0 { big = -big }
        for i to n {
            if i > k {
                ai = A[i, k]
                if ai < 0.0 { ai = -ai }
                if ai > big {
                    big = ai
                    piv = i
                }
            }
        }
        if piv > k {
            for j to n {
                tmp = A[k, j]
                A[k, j] = A[piv, j]
                A[piv, j] = tmp
            }
            tb = b[k]
            b[k] = b[piv]
            b[piv] = tb
        }
        for i to n {
            if i > k {
                f = A[i, k] / A[k, k]
                for j to n {
                    A[i, j] = A[i, j] - f * A[k, j]
                }
                b[i] = b[i] - f * b[k]
            }
        }
    }
    for kk to n {
        i = n - 1.0 - kk
        s = b[i]
        for j to n {
            if j > i {
                s = s - A[i, j] * x[j]
            }
        }
        x[i] = s / A[i, i]
    }
}

N = @@N@@
nn = (N + 1.0) * (N + 1.0)
ne = 2.0 * N * N
node_x = zeros(nn)
node_y = zeros(nn)
for i to N + 1.0 {
    for j to N + 1.0 {
        idx = i * (N + 1.0) + j
        node_x[idx] = i / N
        node_y[idx] = j / N
    }
}
elem = zeros(ne, 3)
ecount = 0.0
for i to N {
    for j to N {
        m00 = i * (N + 1.0) + j
        m10 = (i + 1.0) * (N + 1.0) + j
        m01 = i * (N + 1.0) + (j + 1.0)
        m11 = (i + 1.0) * (N + 1.0) + (j + 1.0)
        elem[ecount, 0] = m00
        elem[ecount, 1] = m10
        elem[ecount, 2] = m11
        ecount = ecount + 1.0
        elem[ecount, 0] = m00
        elem[ecount, 1] = m11
        elem[ecount, 2] = m01
        ecount = ecount + 1.0
    }
}
"#;

// Linear driver: the v1 closed-form assembly (kept for the linear path).
const LINEAR_TEMPLATE: &str = r#"# GENERATED by weft (linear weak form). Do not edit by hand.
@@MESH@@
K = zeros(nn, nn)
rhs = zeros(nn)
bv = zeros(3)
cv = zeros(3)
nl = zeros(3)
for el to ne {
    na = elem[el, 0]
    nb = elem[el, 1]
    nc = elem[el, 2]
    xa = node_x[na]
    ya = node_y[na]
    xb = node_x[nb]
    yb = node_y[nb]
    xcc = node_x[nc]
    ycc = node_y[nc]
    detJ = (xb - xa) * (ycc - ya) - (xcc - xa) * (yb - ya)
    area = 0.5 * detJ
    if area < 0.0 { area = -area }
    bv[0] = yb - ycc
    bv[1] = ycc - ya
    bv[2] = ya - yb
    cv[0] = xcc - xb
    cv[1] = xa - xcc
    cv[2] = xb - xa
    nl[0] = na
    nl[1] = nb
    nl[2] = nc
    xc = (xa + xb + xcc) / 3.0
    yc = (ya + yb + ycc) / 3.0
@@SETUP@@
    for a to 3 {
        gI = nl[a]
        for c to 3 {
            gJ = nl[c]
            ke = @@KE@@
            K[gI, gJ] = K[gI, gJ] + ke
        }
        rhs[gI] = rhs[gI] + fval * area / 3.0
    }
}
for gI to nn {
    xx = node_x[gI]
    yy = node_y[gI]
    onb = xx < 1.0e-9 or xx > 1.0 - 1.0e-9 or yy < 1.0e-9 or yy > 1.0 - 1.0e-9
    if onb {
        for gJ to nn {
            K[gI, gJ] = 0.0
        }
        K[gI, gI] = 1.0
        rhs[gI] = @@BC@@
    }
}
u = zeros(nn)
solve_linear(K, rhs, u, nn)
err2 = 0.0
for gI to nn {
    ue = @@EXACT@@
    d = u[gI] - ue
    err2 = err2 + d * d
}
rms = sqrt(err2 / nn)
center = N / 2.0 * (N + 1.0) + N / 2.0
u_center = u[center]
show N
show rms
show u_center
"#;

// Nonlinear driver: Newton, with the symbolic Jacobian's reaction kernel S'(u).
// The reaction/source S and S' are integrated with a 3-point edge-midpoint rule
// (exact to degree 2), evaluating the current field u_h at each quadrature point.
const NONLINEAR_TEMPLATE: &str = r#"# GENERATED by weft (nonlinear residual; Jacobian via symbolic d/du). Do not edit.
@@MESH@@
# P1 shape-function values at the 3 edge-midpoint quadrature points (weight area/3).
ph = zeros(3, 3)
ph[0, 0] = 0.0
ph[0, 1] = 0.5
ph[0, 2] = 0.5
ph[1, 0] = 0.5
ph[1, 1] = 0.0
ph[1, 2] = 0.5
ph[2, 0] = 0.5
ph[2, 1] = 0.5
ph[2, 2] = 0.0

u = zeros(nn)
resnorm = 1.0
for nit to @@MAXIT@@ {
    J = zeros(nn, nn)
    r = zeros(nn)
    bv = zeros(3)
    cv = zeros(3)
    nl = zeros(3)
    un = zeros(3)
    for el to ne {
        na = elem[el, 0]
        nb = elem[el, 1]
        nc = elem[el, 2]
        xa = node_x[na]
        ya = node_y[na]
        xb = node_x[nb]
        yb = node_y[nb]
        xcc = node_x[nc]
        ycc = node_y[nc]
        detJ = (xb - xa) * (ycc - ya) - (xcc - xa) * (yb - ya)
        area = 0.5 * detJ
        if area < 0.0 { area = -area }
        bv[0] = yb - ycc
        bv[1] = ycc - ya
        bv[2] = ya - yb
        cv[0] = xcc - xb
        cv[1] = xa - xcc
        cv[2] = xb - xa
        nl[0] = na
        nl[1] = nb
        nl[2] = nc
        un[0] = u[na]
        un[1] = u[nb]
        un[2] = u[nc]
        kappa = @@KAPPA@@
        sub = un[0] * bv[0] + un[1] * bv[1] + un[2] * bv[2]
        suc = un[0] * cv[0] + un[1] * cv[1] + un[2] * cv[2]
        for a to 3 {
            gI = nl[a]
            r[gI] = r[gI] + kappa * (sub * bv[a] + suc * cv[a]) / (4.0 * area)
            for c to 3 {
                gJ = nl[c]
                J[gI, gJ] = J[gI, gJ] + kappa * (bv[a] * bv[c] + cv[a] * cv[c]) / (4.0 * area)
            }
        }
        for qq to 3 {
            uh = un[0] * ph[qq, 0] + un[1] * ph[qq, 1] + un[2] * ph[qq, 2]
            xq = xa * ph[qq, 0] + xb * ph[qq, 1] + xcc * ph[qq, 2]
            yq = ya * ph[qq, 0] + yb * ph[qq, 1] + ycc * ph[qq, 2]
            wq = area / 3.0
            sval = @@SVAL@@
            sp = @@SPRIME@@
            for a to 3 {
                gI = nl[a]
                r[gI] = r[gI] + sval * ph[qq, a] * wq
                for c to 3 {
                    gJ = nl[c]
                    J[gI, gJ] = J[gI, gJ] + sp * ph[qq, a] * ph[qq, c] * wq
                }
            }
        }
    }
    for gI to nn {
        xx = node_x[gI]
        yy = node_y[gI]
        onb = xx < 1.0e-9 or xx > 1.0 - 1.0e-9 or yy < 1.0e-9 or yy > 1.0 - 1.0e-9
        if onb {
            for gJ to nn {
                J[gI, gJ] = 0.0
            }
            J[gI, gI] = 1.0
            r[gI] = 0.0
        }
    }
    resnorm = 0.0
    for gI to nn {
        resnorm = resnorm + r[gI] * r[gI]
    }
    resnorm = sqrt(resnorm)
    show resnorm
    negr = zeros(nn)
    for gI to nn {
        negr[gI] = -r[gI]
    }
    delta = zeros(nn)
    solve_linear(J, negr, delta, nn)
    for gI to nn {
        u[gI] = u[gI] + delta[gI]
    }
    if resnorm < 1.0e-9 { break }
}

err2 = 0.0
for gI to nn {
    ue = @@EXACT@@
    d = u[gI] - ue
    err2 = err2 + d * d
}
rms = sqrt(err2 / nn)
center = N / 2.0 * (N + 1.0) + N / 2.0
u_center = u[center]
show rms
show u_center
"#;
