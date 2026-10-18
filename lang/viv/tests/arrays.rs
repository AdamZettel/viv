// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The array vocabulary (lang/viv/src/interp/arrays.rs), checked in all three
//! engines: the interpreter (`viv run`), Viv -> C -> native, and the direct
//! wasm backend run in node (skipped, with a note, when node or a viv.wasm
//! with the runtime exports is missing; VIV_WASM=<path> names one).
//!
//! - tests/arrays/numpy_exact.viv: 4,812 cases NumPy computes in an order Viv
//!   follows; every engine prints tests/arrays/numpy_exact.txt byte for byte
//!   (tools/arrays/numpy_cases.py writes both from NumPy 1.24)
//! - tests/arrays/numpy_tol.viv: the cases held to NumPy within a tolerance
//!   (BLAS, LAPACK, einsum's own loops); every test passes in every engine
//! - the errors, word for word and located, in every engine
//! - the solvers that return their results give the bits the forms that fill
//!   an array give; `vals, vecs = eig_sym(A)` and `U, s, Vt = svd(A)`
//! - broadcasting under `grad`: an operand's gradient summed back to its shape

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_arrays_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, PartialEq)]
struct Ran {
    code: Option<i32>,
    out: String,
    err: String,
}

fn ran(o: std::process::Output) -> Ran {
    Ran { code: o.status.code(), out: String::from_utf8_lossy(&o.stdout).into_owned(), err: String::from_utf8_lossy(&o.stderr).into_owned() }
}

fn wasm_runtime() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    Some(wasm)
}

fn engines(tag: &str, src: &str) -> (Ran, Ran, Option<Ran>) {
    let dir = scratch(tag);
    let p = dir.join("p.viv");
    std::fs::write(&p, src).unwrap();
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q"]).arg(&p).output().unwrap());
    let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{tag}: viv refused the program for C:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "{tag}: the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    let n = ran(Command::new(dir.join("p")).output().unwrap());
    let w = wasm_runtime().map(|wasm| {
        let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.wasm"]).output().unwrap();
        assert!(c.status.success(), "{tag}: viv refused the program for wasm:\n{}", String::from_utf8_lossy(&c.stderr));
        ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap())
    });
    if w.is_none() {
        eprintln!("note: {tag}: no viv.wasm with the runtime exports, or no node; the direct wasm engine was skipped");
    }
    (i, n, w)
}

/// Every engine exits 0 and prints exactly `want`.
fn prints(tag: &str, src: &str, want: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!((i.code, i.err.as_str()), (Some(0), ""), "{tag}: interpreter");
    if i.out != want {
        let first = i.out.lines().zip(want.lines()).position(|(a, b)| a != b).unwrap_or(0);
        panic!(
            "{tag}: the interpreter differs from the expected output at line {}:\n  got  {}\n  want {}",
            first + 1,
            i.out.lines().nth(first).unwrap_or(""),
            want.lines().nth(first).unwrap_or("")
        );
    }
    assert!(i == n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert!(w.code == Some(0) && w.out == want, "{tag}: direct wasm differs\n{}", w.err);
    }
}

/// Every engine stops with exit 4 (a run-time error) after printing `out`, with `error: <msg>`;
/// the interpreter and native agree byte for byte, location included.
fn stops(tag: &str, src: &str, out: &str, msg: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!(i.code, Some(4), "{tag}: interpreter {i:?}");
    assert_eq!(i.out, out, "{tag}");
    assert!(i.err.starts_with(&format!("error: {msg}\n  --> p.viv:")), "{tag}: {}", i.err);
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str(), w.err.as_str()), (Some(4), out, format!("error: {msg}\n").as_str()), "{tag}: direct wasm");
    }
}

/// The compiler refuses `src` with a message containing `msg`, and so does
/// `viv run`.
fn refused(tag: &str, src: &str, msg: &str) {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let c = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap());
    assert_ne!(c.code, Some(0), "{tag}: the compiler accepted it");
    assert!(c.err.contains(msg), "{tag}: compiler said\n{}", c.err);
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap());
    assert_ne!(i.code, Some(0), "{tag}: the interpreter ran it");
    assert!(i.err.contains(msg), "{tag}: interpreter said\n{}", i.err);
}

fn case_file(name: &str) -> String {
    std::fs::read_to_string(root().join("lang/viv/tests/arrays").join(name)).unwrap()
}

// ------------------------------------------------------------ against NumPy

#[test]
fn every_exact_case_is_numpys_result_in_every_engine() {
    prints("numpy_exact", &case_file("numpy_exact.viv"), &case_file("numpy_exact.txt"));
}

#[test]
fn every_tolerance_case_passes_in_every_engine() {
    let src = case_file("numpy_tol.viv");
    let tests = src.matches("\ntest \"").count();
    let want: String = src.lines().filter_map(|l| l.strip_prefix("test \"")).map(|l| format!("PASS {}\n", &l[..l.find('"').unwrap()])).collect();
    assert_eq!(want.lines().count(), tests);
    prints("numpy_tol", &src, &want);
}

// ------------------------------------------------------------ batched @

/// `@` on stacks of matrices, NumPy's matmul: the batch axes broadcast, and a
/// vector operand's axis is dropped. Small whole numbers, so NumPy's BLAS
/// gives the exact result too (the expected text is NumPy 1.24's).
#[test]
fn a_stack_of_matrices_multiplies_as_numpys_matmul_in_every_engine() {
    let src = r#"C = reshape([-5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0], 2, 3, 4) @ reshape([-2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0], 2, 4, 5)
print(shape(C))
print(ravel(C))
C = reshape([-4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0], 2, 3, 4) @ reshape([-1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0], 4, 5)
print(shape(C))
print(ravel(C))
C = reshape([-3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0], 3, 4) @ reshape([0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0], 2, 4, 5)
print(shape(C))
print(ravel(C))
C = reshape([-2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0], 2, 1, 3, 4) @ reshape([1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0], 3, 4, 2)
print(shape(C))
print(ravel(C))
C = reshape([-1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0], 2, 3, 4) @ [2.0, -2.0, 5.0, 1.0]
print(shape(C))
print(ravel(C))
C = [0.0, -4.0, 3.0, -1.0] @ reshape([3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0], 2, 4, 5)
print(shape(C))
print(ravel(C))
C = reshape([1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0], 1, 3, 4) @ reshape([4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0], 2, 4, 2)
print(shape(C))
print(ravel(C))
C = reshape([2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0, -5.0, 2.0], 3, 4) @ reshape([5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0, -1.0], 4, 2)
print(shape(C))
print(ravel(C))
C = reshape([3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0, 3.0], 3, 4) @ [-5.0, 2.0, -2.0, 5.0]
print(shape(C))
print(ravel(C))
C = [4.0, 0.0, -4.0, 3.0] @ reshape([-4.0, 3.0, -1.0, -5.0, 2.0, -2.0, 5.0, 1.0, -3.0, 4.0, 0.0, -4.0], 4, 3)
print(shape(C))
print(ravel(C))
"#;
    let want = r#"[2, 3, 5]
[26, -29, -29, 26, -29, 6, 9, 12, 4, 7, -14, -30, 20, -7, -23, -8, -32, 10, 8, 17, -1, 22, -10, -31, 25, -5, 21, -8, 29, -33]
[2, 3, 5]
[-21, 7, 24, -14, -30, 16, -8, -32, 10, 8, -13, -1, 22, -10, -31, -9, -5, 21, -8, 29, -5, -9, 20, -6, -32, -1, -13, 19, -4, 28]
[2, 3, 5]
[28, -4, 19, -13, -1, -32, -6, 20, -9, -5, 29, -8, 21, -5, -9, -7, -28, 28, -4, 19, -7, 30, -32, -6, 20, -7, -33, 29, -8, 21]
[2, 3, 3, 2]
[17, 24, 17, -39, -27, 19, 18, 25, 20, -36, -33, 13, -36, 26, 23, -33, 16, 7, 17, -44, -27, 14, 6, -16, 24, -37, -29, 12, 6, -16, 20, -30, 13, 10, -16, -16]
[2, 3]
[16, -3, 22, -19, 28, -24]
[2, 5]
[-30, 0, 8, -28, 2, 9, 6, -30, 0, 8]
[2, 3, 2]
[12, -29, -37, 24, 13, -33, 14, -27, -44, 17, 19, -27]
[3, 2]
[19, -27, -39, 17, 24, 17]
[3]
[3, 3, 3]
[3]
[-24, 8, -4]
"#;
    prints("batched_matmul", src, want);
}

#[test]
fn a_batched_product_that_does_not_fit_stops_with_the_same_words_everywhere() {
    stops("bmm_inner", "c = ones(2, 3, 4) @ ones(2, 5, 5)\n", "", "`@` shape mismatch: 3x4 @ 5x5");
    stops("bmm_batch", "c = ones(2, 3, 4) @ ones(3, 4, 5)\n", "", "`@`: the stacks of matrices have shapes [2] and [3], which do not broadcast");
    stops("bmm_vectors", "c = ones(3) @ ones(3)\n", "", "`@` of two vectors is refused: dot(u, v) is their dot product, outer(u, v) their outer product");
}

/// `where(m, a, b)` broadcasts its three arguments as NumPy's `where` does: a
/// causal mask of one matrix applies to every matrix of a stack.
#[test]
fn where_broadcasts_its_arguments_in_every_engine() {
    let src = "m = triu(ones(3, 3), k=1)\nS = reshape(arange(18.0), 2, 3, 3)\nprint(where(m, -1.0, S))\nprint(where(m, S, 0.5))\nprint(where(ones(2, 1), [1.0, 2.0, 3.0], -1.0))\n";
    let want = "[[[0, -1, -1], [3, 4, -1], [6, 7, 8]], [[9, -1, -1], [12, 13, -1], [15, 16, 17]]]\n[[[0.5, 1, 2], [0.5, 0.5, 5], [0.5, 0.5, 0.5]], [[0.5, 10, 11], [0.5, 0.5, 14], [0.5, 0.5, 0.5]]]\n[[1, 2, 3], [1, 2, 3]]\n";
    prints("where_bcast", src, want);
}

// ------------------------------------------------------------ the errors

#[test]
fn a_shape_that_does_not_fit_stops_with_the_same_words_everywhere() {
    stops("reshape", "a = arange(6)\nshow len(a)\nb = reshape(a, 4, 2)\n", "6\n", "`reshape`: cannot reshape an array of 6 elements into shape [4, 2]");
    stops("reshape1", "b = reshape(arange(6), -1, 4)\n", "", "`reshape`: cannot reshape an array of 6 elements into shape [-1, 4]");
    stops("axis", "s = sum(ones(2, 3), axis=2)\n", "", "`sum`: axis 2 is out of bounds for an array of rank 2");
    stops("vector", "s = mean(ones(3), axis=0)\n", "", "`mean` along the only axis of a vector gives a number: write `mean(a)`");
    stops("bcast", "c = ones(2, 3) + ones(3, 2)\n", "", "array shape mismatch: [2, 3] vs [3, 2]");
    stops("bcast_max", "c = max(ones(2, 3), ones(2))\n", "", "`max`: array shape mismatch: [2, 3] vs [2]");
    stops("take", "t = take(ones(2, 3), [0.0, 3.0], axis=1)\n", "", "`take`: index 3 out of bounds for axis 1 (size 3)");
    stops("einsum", "e = einsum(\"ij,jk->ik\", ones(2, 3), ones(4, 2))\n", "", "`einsum`: index 'j' has length 3 in one place and 4 in another");
    stops("dot", "d = dot(ones(3), ones(2))\n", "", "`dot` takes two vectors of one length (for matrices write A @ B); these have shapes [3] and [2]");
    stops("empty", "m = amax(zeros(0))\n", "", "`amax` of an empty array");
    stops("arange", "a = arange(0.0, 1.0, 0.0)\n", "", "`arange`: the step is zero");
    stops("putmask", "a = ones(3)\nputmask(a, ones(2) > 0.0, 1.0)\n", "", "`putmask`: array shape mismatch: [3] vs [2]");
    stops("diag", "d = diag(ones(2, 2, 2))\n", "", "`diag` takes a vector or a matrix; this array has rank 3");
    stops("svd", "U = zeros(2, 2)\ns = zeros(2)\nV = zeros(2, 2)\nsvd(ones(2, 3), U, s, V)\n", "", "`svd`: output 3 has shape [2, 2]; it needs [2, 3]");
    stops("transpose_axes", "t = transpose(ones(2, 3), axes=[0.0, 0.0])\n", "", "`transpose`: axis 0 is named twice");
}

#[test]
fn a_form_the_compiler_can_see_is_wrong_is_refused_by_both() {
    refused("einsum_spec", "s = \"ij\"\ne = einsum(s, ones(2, 2))\n", "`einsum` takes its subscripts as a string literal");
    refused("einsum_ops", "e = einsum(\"ij,jk->ik\", ones(2, 2))\n", "`einsum`: the subscripts name 2 operands and the call passes 1");
    refused("eig_expr", "A = ones(2, 2)\nx = eig_sym(A)[0]\n", "`eig_sym(A)` gives 2 values, so it cannot be part of an expression");
    refused("svd_names", "U, s = svd(ones(3, 2))\n", "`svd(A)` gives 3 values, but 2 names unpack them");
    refused("kind", "r = reshape(2.0, 1)\n", "`reshape` takes a real array as argument 1");
    refused("keepdims", "s = sum(ones(2, 2), axis=0, keepdims=1.0)\n", "`sum` takes true or false as argument 3");
}

// ------------------------------------------------------------ returning solvers

#[test]
fn a_solver_that_returns_its_result_gives_the_bits_of_the_one_that_fills_an_array() {
    let src = r#"A = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]
b = [1.0, -2.0, 0.5]
B = [[1.0, 0.0], [-2.0, 1.0], [0.5, 3.0]]
x1 = zeros(3)
lin_solve(A, b, x1)
print(all(lin_solve(A, b) == x1))
X1 = zeros(3, 2)
lin_solve_multi(A, B, X1)
print(all(lin_solve(A, B) == X1))
I1 = zeros(3, 3)
lin_solve_multi(A, eye(3), I1)
print(all(inv(A) == I1))
L1 = zeros(3, 3)
cholesky(A, L1)
print(all(cholesky(A) == L1))
v1 = zeros(3)
Q1 = zeros(3, 3)
eig_sym(A, v1, Q1)
vals, vecs = eig_sym(A)
print(all(vals == v1) and all(vecs == Q1))
def smallest(M) {
    w, V = eig_sym(M)
    return w[0]
}
print(smallest(A) == v1[0])
U, s, Vt = svd(hcat(A, b))
print(fmt(s, ".12g"))
print(max(abs(U @ (diag(s) @ Vt) - hcat(A, b))) < 1e-14)
"#;
    prints("returning", src, "true\ntrue\ntrue\ntrue\ntrue\ntrue\n[4.73273350432, 3.30264951872, 1.88911083147]\ntrue\n");
}

// ------------------------------------------------------------ broadcasting and grad

#[test]
fn a_broadcast_operand_gets_its_gradient_summed_back_to_its_shape() {
    // f(b) = sum((X + b)^2) for a row b: df/db[j] = sum_i 2 (X[i, j] + b[j])
    let src = r#"X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
def f(b: Arr, X: Arr) {
    Y = X + b
    return sum(Y * Y)
}
g = grad(f, [0.5, -1.0, 2.0], X)
print(fmt(g))
def h(c: Arr, X: Arr) {
    Y = X * c
    return sum(Y)
}
print(fmt(grad(h, [[2.0], [3.0]], X)))
"#;
    prints("bcast_grad", src, "[12.0, 10.0, 26.0]\n[[6.0], [15.0]]\n");
}

#[test]
fn transpose_reverses_the_axes_of_any_rank() {
    let src = "a = reshape(arange(24), 2, 3, 4)\nt = transpose(a)\nprint(shape(t))\nprint(t[3, 2, 1] == a[1, 2, 3])\nprint(fmt(transpose([1.0, 2.0])))\n";
    prints("transpose", src, "[4, 3, 2]\ntrue\n[1.0, 2.0]\n");
}

// ------------------------------------------------------------ ordered sums

#[test]
fn an_ordered_sum_adds_left_to_right_as_a_loop_does_in_every_engine() {
    // ten entries: NumPy's pairwise sum keeps eight running sums, so the 1.0
    // after 1.0e16 is lost in one order and kept in the other
    let src = r#"x = [1.0e16, 1.0, -1.0e16, 1.0, 3.0, 1.0e-3, 7.0, 2.0, 1.0, 1.0]
s = 0.0
for i to len(x) {
    s += x[i]
}
A = reshape(x, 2, 5)
print(fmt(sum(x), ".17g"))
print(fmt(sum(x, ordered=true), ".17g"))
print(sum(x, ordered=true) == s, sum(x, ordered=true) != sum(x))
print(sum(A, ordered=true) == s, sum(x, ordered=false) == sum(x))
t = 0.0
for i to len(x) {
    t += x[i] * x[i]
}
print(dot(x, x, ordered=true) == t, prod(x, ordered=true) == prod(x))
def f(v: Arr) {
    return sum(v * v, ordered=true)
}
print(fmt(grad(f, [1.0, -2.0, 3.0])))
"#;
    prints("ordered_sum", src, "14.000999999999999\n15.001000000000001\ntrue true\ntrue true\ntrue true\n[2.0, -4.0, 6.0]\n");
}

#[test]
fn ordered_takes_a_literal_and_no_axis() {
    refused("ordered_var", "b = true\nprint(sum([1.0, 2.0], ordered=b))\n", "`ordered=` is `true` or `false` as written");
    refused("ordered_axis", "print(sum([[1.0, 2.0]], axis=0, ordered=true))\n", "takes no `axis=`");
    refused("ordered_dot_false", "print(dot([1.0], [2.0], ordered=false))\n", "always goes left to right");
}
