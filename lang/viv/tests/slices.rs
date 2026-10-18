// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Slices, `copy`, `concat`, `vcat`, `hcat`.
//!
//! Each program runs in the interpreter (`viv run`), as C compiled natively,
//! and (when node and a viv.wasm with the runtime are present) as a module
//! from the direct wasm backend. A program that runs must print the same bytes
//! in all three; one that stops must stop with the same words, and the
//! interpreter and C with the same location and note. (A wasm module reports
//! the message only, as it does for every run-time error.)

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
    let dir = std::env::temp_dir().join(format!("viv_slices_{}_{tag}", std::process::id()));
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
    Ran {
        code: o.status.code(),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn interp(dir: &Path) -> Ran {
    ran(Command::new(viv_bin()).current_dir(dir).args(["run", "-q", "p.viv"]).output().unwrap())
}

fn native(dir: &Path) -> Option<Ran> {
    if Command::new(cc()).arg("--version").output().is_err() {
        return None;
    }
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2", "-Wall", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    Some(ran(Command::new(dir.join("p")).output().unwrap()))
}

/// The direct wasm backend in node, or None when there is no runtime to run it.
fn direct(dir: &Path) -> Option<Ran> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(12).any(|w| w == b"vrt_getslice") {
        eprintln!("skip wasm: {} has no slice runtime (www/build-wasm.sh builds one)", wasm.display());
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap();
    assert!(c.status.success(), "the wasm backend refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    Some(ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap()))
}

/// Run `src` in every engine; they must agree. Returns the interpreter's run.
fn all(tag: &str, src: &str) -> Ran {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let i = interp(&dir);
    if let Some(n) = native(&dir) {
        assert_eq!(i, n, "interpreter and C disagree on\n{src}");
    }
    if let Some(w) = direct(&dir) {
        if i.code == Some(0) {
            assert_eq!(i, w, "interpreter and wasm disagree on\n{src}");
        } else {
            assert_eq!(w.code, i.code, "{src}");
            assert_eq!(w.err.lines().next(), i.err.lines().next(), "interpreter and wasm stop differently on\n{src}");
        }
    }
    i
}

fn prints(tag: &str, src: &str, out: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(0), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
}

/// A program that stops: stderr is exactly `err`.
fn stops(tag: &str, src: &str, out: &str, err: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(4), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
    assert_eq!(r.err, err, "{tag}");
}

#[test]
fn reads_of_a_vector() {
    prints(
        "vec",
        "a = [10.0, 11.0, 12.0, 13.0, 14.0]\nshow a[1:3]\nshow a[:2]\nshow a[3:]\nshow a[:]\nshow a[0:5:2]\nshow a[1::2]\nshow len(a[2:2])\nshow a[1.9:3.2]\n",
        "[11, 12]\n[10, 11]\n[13, 14]\n[10, 11, 12, 13, 14]\n[10, 12, 14]\n[11, 13]\n0\n[11, 12]\n",
    );
}

#[test]
fn a_step_held_in_a_variable() {
    // the start is left out and the parser cannot see the step is positive
    // (a string would count from its end for a negative one): an array starts at 0
    prints(
        "vstep",
        "a = [10.0, 11.0, 12.0, 13.0, 14.0]\ns = 2\nshow a[::s]\nshow a[:4:s]\nshow a[1::s]\nA = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nshow A[::s, ::s]\n",
        "[10, 12, 14]\n[10, 12]\n[11, 13]\n[[1, 3],\n [7, 9]]\n",
    );
}

#[test]
fn rows_columns_and_blocks() {
    prints(
        "mat",
        "M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\nshow M[1, :]\nshow M[:, 2]\nshow M[0:2, 1:3]\nshow shape(M[1:])\nshow shape(M[:, 0:1])\n\
         T = zeros(2, 3, 4)\nT[1, 2, 3] = 7\nshow shape(T[:, 1:3, 0])\nshow T[1, 2, :]\n",
        "[4, 5, 6]\n[3, 6]\n[[2, 3],\n [5, 6]]\n[1, 3]\n[2, 1]\n[2, 2]\n[0, 0, 0, 7]\n",
    );
}

#[test]
fn a_slice_is_a_copy() {
    prints("copy", "a = [1.0, 2.0, 3.0]\nb = a[0:2]\nb[0] = 9\nshow a[0]\nc = copy(a)\nc[1] = 9\nshow a[1]\n", "1\n2\n");
}

#[test]
fn writes() {
    prints(
        "write",
        "b = zeros(5)\nb[1:3] = [1.0, 2.0]\nb[3:] = 4\nb[::2] += 0.5\nshow b\nG = zeros(2, 3)\nG[:, 1] = [1.0, 2.0]\nG[1, :] += 10\nshow G\n\
         s = [1.0, 2.0, 3.0, 4.0]\ns[1:4] = s[0:3]\nshow s\n",
        "b: [0.5, 1, 2.5, 4, 4.5]\nG: [[0, 1, 0],\n    [10, 12, 10]]\ns: [1, 1, 2, 3]\n",
    );
}

#[test]
fn joins() {
    prints(
        "joins",
        "show concat([1.0], [2.0, 3.0], [4.0])\nshow vcat([1.0, 2.0], [3.0, 4.0])\nshow hcat([1.0, 2.0], [3.0, 4.0])\nshow hcat([5.0, 6.0])\nshow vcat([5.0, 6.0])\n\
         M = [[1.0, 2.0], [3.0, 4.0]]\nshow hcat(M, [9.0, 9.0])\nshow vcat(M, [7.0, 8.0], M[0:1, :])\n",
        "[1, 2, 3, 4]\n[[1, 2],\n [3, 4]]\n[[1, 3],\n [2, 4]]\n[[5],\n [6]]\n[[5, 6]]\n[[1, 2, 9],\n [3, 4, 9]]\n[[1, 2],\n [3, 4],\n [7, 8],\n [1, 2]]\n",
    );
}

#[test]
fn complex_arrays() {
    prints(
        "cplx",
        "z = czeros(3)\nz[0:2] = complex([1.0, 2.0], [3.0, 4.0])\nz[2:] = 5.0 + 6.0i\nw = z[1:]\nshow re(w)\nshow im(w)\nshow re(concat(z[0:1], z[2:3]))\n",
        "[2, 5]\n[4, 6]\n[1, 5]\n",
    );
}

#[test]
fn compound_assignment_evaluates_the_indices_once() {
    prints("once", "v = zeros(4)\nsrand(5)\nv[0:1 + floor(3 * rand())] += 1\nsrand(5)\nshow sum(v) == 1 + floor(3 * rand())\n", "true\n");
}

#[test]
fn a_user_def_named_like_a_join_wins() {
    prints("own", "def concat(a: Arr, b: Arr) {\n    return 7\n}\nshow concat([1.0], [2.0])\n", "7\n");
}

#[test]
fn grad_through_slices_and_joins() {
    // d/dx of sum((x[1:] * x[:2]) ^ 2) + sum(hcat(x, x)[:, 1]) + x[0] * sum(vcat(x, x)[1, :])
    let src = "def f(x: Arr) {\n    y = zeros(3)\n    y[0:2] = x[1:] * x[:2]\n    y[2] = x[2]\n    return sum(y * y) + sum(hcat(x, x)[:, 1]) + x[0] * sum(vcat(x, x)[1, :])\n}\n\
               g = grad(f, [1.0, 2.0, 3.0])\nshow g\n";
    // by hand: y0 = x1 x0, y1 = x2 x1, y2 = x2; sum(y^2) + (x0 + x1 + x2) + x0 (x0 + x1 + x2)
    // d/dx0 = 2 y0 x1 + 1 + (x0 + x1 + x2) + x0 = 8 + 1 + 6 + 1 = 16
    // d/dx1 = 2 y0 x0 + 2 y1 x2 + 1 + x0 = 4 + 36 + 1 + 1 = 42
    // d/dx2 = 2 y1 x1 + 2 y2 + 1 + x0 = 24 + 6 + 1 + 1 = 32
    prints("grad", src, "g: [16, 42, 32]\n");
}

#[test]
fn grad_through_matrix_slices_matches_central_differences() {
    // blocks, rows, columns, a column write, copy and the joins on a matrix,
    // each gradient entry checked against a central difference in every engine
    let src = "def f(x: Arr) {\n    A = vcat(x[0:3], x[3:6])\n    B = copy(A)\n    B[:, 1] = A[:, 2] * A[:, 0]\n    C = hcat(B[0:2, 1:3], A[:, 0:1])\n    r = C[1, :]\n    return sum(C * C) + sum(r * r * r) + sum(B[0, :] * A[1, :])\n}\n\
               x0 = [0.3, -1.2, 0.7, 2.0, 0.5, -0.4]\n\
               g = grad(f, x0)\n\
               h = 1.0e-6\n\
               worst = 0.0\n\
               for i to len(x0) {\n    xp = copy(x0)\n    xm = copy(x0)\n    xp[i] += h\n    xm[i] -= h\n    worst = max(worst, abs(g[i] - (f(xp) - f(xm)) / (2.0 * h)))\n}\n\
               show worst < 1.0e-7\n";
    prints("gradfd", src, "true\n");
}

#[test]
fn grad_of_a_filled_slice_sums_its_adjoint() {
    prints("gradfill", "def f(x: Arr) {\n    y = zeros(4)\n    y[1:4] = x[0]\n    return sum(y * y)\n}\nshow grad(f, [2.0])\n", "[12]\n");
}

#[test]
fn a_negative_bound_counts_from_the_end() {
    prints(
        "neg",
        "a = [1.0, 2.0, 3.0]\nshow a[-2:]\nshow a[:-1]\nshow a[-3:-1]\nshow a[-1:]\nM = [[1.0, 2.0], [3.0, 4.0]]\nshow M[-1, :]\nshow M[:, -1]\nM[-1, -1] = 9.0\nshow M\n",
        "[2, 3]\n[1, 2]\n[1, 2]\n[3]\n[3, 4]\n[2, 4]\nM: [[1, 2],\n    [3, 9]]\n",
    );
}

#[test]
fn a_bound_before_the_start_stops() {
    stops(
        "negpast",
        "a = [1.0, 2.0, 3.0]\nshow 1\nshow a[-4:]\n",
        "1\n",
        "error: slice bound -4 is before the start of dimension 0 (size 3)\n  --> p.viv:3:6\n   3 | show a[-4:]\n            ^^^^^^ here\n",
    );
}

#[test]
fn bad_bounds_stop_every_engine_with_the_same_words() {
    let cases = [
        ("past", "a = zeros(3)\nb = a[1:4]\n", "error: slice end 4 is past the end of dimension 0 (size 3)\n  --> p.viv:2:5\n   2 | b = a[1:4]\n           ^^^^^^ here\n".to_string()),
        ("after", "a = zeros(3)\nb = a[2:1]\n", "error: slice start 2 is after its end 1 in dimension 0\n  --> p.viv:2:5\n   2 | b = a[2:1]\n           ^^^^^^ here\n".to_string()),
        ("step", "a = zeros(3)\nb = a[::0]\n", "error: slice step 0 in dimension 0 must not be zero\n  --> p.viv:2:5\n   2 | b = a[::0]\n           ^^^^^^ here\n".to_string()),
        (
            "index",
            "M = zeros(2, 3)\nb = M[-3, :]\n",
            "error: index -3 out of bounds for dimension 0 (size 2)\n  --> p.viv:2:5\n   2 | b = M[-3, :]\n           ^^^^^^^^ here\n".to_string(),
        ),
        ("rank", "a = zeros(3)\nb = a[0:1, 0]\n", "error: array has rank 1 but was indexed with 2 index/indices\n  --> p.viv:2:5\n   2 | b = a[0:1, 0]\n           ^^^^^^^^^ here\n".to_string()),
        (
            "shape",
            "M = zeros(2, 3)\nM[:, 0] = [1.0, 2.0, 3.0]\n",
            "error: slice assignment: the value has shape [3], the slice has shape [2]\n  --> p.viv:2:1\n   2 | M[:, 0] = [1.0, 2.0, 3.0]\n       ^^^^^^^^^^^^^^^^^^^^^^^^^ here\n".to_string(),
        ),
        ("hcat", "b = hcat(zeros(2), zeros(3))\n", "error: `hcat`: the arrays have 2 and 3 rows\n  --> p.viv:1:5\n   1 | b = hcat(zeros(2), zeros(3))\n           ^^^^^^^^^^^^^^^^^^^^^^^^ here\n".to_string()),
        ("vcat", "b = vcat(zeros(2), zeros(2, 3))\n", "error: `vcat`: the arrays have 2 and 3 columns\n  --> p.viv:1:5\n   1 | b = vcat(zeros(2), zeros(2, 3))\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n".to_string()),
        ("concat", "b = concat(zeros(2), zeros(2, 2))\n", "error: `concat` joins vectors; this array has rank 2\n  --> p.viv:1:5\n   1 | b = concat(zeros(2), zeros(2, 2))\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n".to_string()),
    ];
    for (tag, src, err) in cases {
        stops(tag, src, "", &err);
    }
}

#[test]
fn a_slice_of_a_number_is_refused_before_it_runs() {
    let dir = scratch("num");
    std::fs::write(dir.join("p.viv"), "x = 3.0\nshow x[0:1]\n").unwrap();
    let r = interp(&dir);
    assert_eq!(r.code, Some(3), "{r:?}");
    assert!(r.err.contains("only an array can be sliced"), "{}", r.err);
}
