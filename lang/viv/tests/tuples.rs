// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Several results:
//! `return a, b`, `a, b = f(x)`, swaps, and unpacking an array such as
//! `shape(X)`, in the interpreter, as C, and (when node and a viv.wasm with
//! the runtime are present) as a module from the direct wasm backend.

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
    let dir = std::env::temp_dir().join(format!("viv_tuples_{}_{tag}", std::process::id()));
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
    assert_eq!(r.code, Some(1), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
    assert_eq!(r.err, err, "{tag}");
}


/// The messages the compile gives for `src` (empty when it lowers).
fn errors(src: &str) -> Vec<String> {
    match viv::build_ir(src, knotkit::span::SourceId(0)) {
        Ok(_) => Vec::new(),
        Err(ds) => ds.into_iter().map(|d| d.message).collect(),
    }
}

#[test]
fn several_results() {
    prints(
        "basic",
        "def divmod2(a, b) {\n    q = floor(a / b)\n    return q, a - q * b\n}\nq, r = divmod2(17, 5)\nshow q\nshow r\n",
        "q: 3\nr: 2\n",
    );
    prints(
        "arrays",
        "def center(x: Arr) {\n    m = sum(x) / len(x)\n    return m, x - m, len(x)\n}\nm, d, n = center([1, 2, 6])\nshow m\nshow d\nshow n\n",
        "m: 3\nd: [-2, -1, 3]\nn: 3\n",
    );
    prints(
        "forward",
        "def two(a) {\n    return a, 2 * a\n}\ndef fwd(a) {\n    if a > 10 {\n        return a, a\n    }\n    return two(a)\n}\nx, y = fwd(4)\nshow y\nx, y = fwd(40)\nshow y\n",
        "y: 8\ny: 40\n",
    );
    prints(
        "recursive",
        "def fib2(n) {\n    if n == 0 {\n        return 0, 1\n    }\n    p, c = fib2(n - 1)\n    return c, p + c\n}\na, b = fib2(10)\nshow a\n",
        "a: 55\n",
    );
    prints(
        "kinds",
        "record P { x: Num }\ndef k(a) {\n    return a > 0, a * 1i, P { x: -a }\n}\nok, z, p = k(2)\nshow ok\nshow z\nshow p.x\n",
        "ok: true\nz: 0+2i\n-2\n",
    );
    prints(
        "dropped",
        "def two(a) {\n    print(a)\n    return a, a\n}\ntwo(3)\n",
        "3\n",
    );
    prints(
        "in a def",
        "def two(a) {\n    return a, a + 1\n}\ndef use(n) {\n    s = 0\n    for k to n {\n        a, b = two(k)\n        s = s + a * b\n    }\n    return s\n}\nshow use(4)\n",
        "20\n",
    );
}

#[test]
fn swaps_and_arrays() {
    prints("swap", "a = 1\nb = 2\na, b = b, a\nshow a\nshow b\n", "a: 2\nb: 1\n");
    prints("rotate", "a = 1\nb = 2\nc = 3\na, b, c = c, a, b\nshow a\nshow b\nshow c\n", "a: 3\nb: 1\nc: 2\n");
    prints("elements", "v = [1, 2, 3]\nv[0], v[2] = v[2], v[0]\nshow v\n", "v: [3, 2, 1]\n");
    // the targets' indices are computed after the right side, as in Python
    prints("index order", "v = [0, 0, 0]\ni = 0\ni, v[i] = 2, 7\nshow v\n", "v: [0, 0, 7]\n");
    prints("slice", "v = zeros(4)\nv[0:2], w = [1, 2], 5\nshow v\nshow w\n", "v: [1, 2, 0, 0]\nw: 5\n");
    prints("shape", "X = zeros(2, 3, 4, 5)\nB, C, H, W = shape(X)\nshow B * 1000 + C * 100 + H * 10 + W\n", "2345\n");
    prints("vector", "lo, hi = [-1, 4]\nshow hi - lo\n", "5\n");
}

#[test]
fn an_array_of_the_wrong_length_stops() {
    let r = all("short", "X = zeros(2, 3)\nshow 1\nB, C, H = shape(X)\nshow B\n");
    assert_eq!(r.code, Some(4), "{r:?}");
    assert_eq!(r.out, "1\n");
    assert!(r.err.starts_with("error: cannot unpack this array into 3 names: its length is not 3"), "{}", r.err);
    assert!(r.err.contains("p.viv:3:"), "located at the statement: {}", r.err);
}

#[test]
fn value_and_grad_keeps_its_form() {
    prints(
        "vg",
        "def sq(x: Arr) {\n    return sum(x * x)\n}\nloss, g = value_and_grad(sq, [1, 2])\nshow loss\nshow g\n",
        "loss: 5\ng: [2, 4]\n",
    );
}

#[test]
fn refused_with_a_location() {
    let two = "def f(a) {\n    return a, a\n}\n";
    let cases: &[(String, &str)] = &[
        (format!("{two}x = f(1) + 2"), "`f` returns 2 values, so it cannot be part of an expression"),
        (format!("{two}show f(1)"), "`f` returns 2 values, so it cannot be part of an expression"),
        (format!("{two}g = f"), "`f` returns 2 values, so it cannot be passed as a value"),
        (format!("{two}x, y, z = f(1)"), "`f` returns 2 values, but 3 names unpack them"),
        ("def f(a) {\n    if a > 0 {\n        return a, a\n    }\n    return a\n}".into(), "`f` returns 2 values, but this `return` gives one value"),
        ("def f(a) {\n    if a > 0 {\n        return a, a\n    }\n    return a, a, a\n}".into(), "`f` returns 3 values here, and 2 before"),
        ("a, b = 1, 2, 3".into(), "2 names, but 3 values"),
        ("return 1, 2".into(), "`return a, b` returns from a def"),
        (format!("{two}def g(a) -> Num {{\n    return a, a\n}}"), "`g` returns 2 values, so it takes no `-> T`"),
        ("record P { x: Num }\ndef f(p: P) {\n    p.x, y = 1, 2\n    return y\n}\nprint(f(P { x: 1 }))".into(), "`p` is a parameter, and a def does not change the values it is given"),
        ("a, 3 = 1, 2".into(), "cannot unpack into this"),
        ("a, b += 1, 2".into(), "expected `=` after the names to unpack into"),
        ("def sq(x: Arr) {\n    return sum(x * x)\n}\nx = [1.0]\nloss, x[0] = value_and_grad(sq, x)".into(), "`value_and_grad` unpacks into two names"),
    ];
    for (src, want) in cases {
        let errs = errors(src);
        assert!(errs.iter().any(|e| e.contains(want)), "program:\n{src}\nexpected an error containing {want:?}, got {errs:?}");
        let err = viv::interp::run(src).expect_err(src);
        assert!(err.contains(want), "the interpreter, on\n{src}\nexpected {want:?}, got {err:?}");
    }
    // a type that differs between two returns: the compile refuses it
    let src = "def g(a) {\n    if a > 0 {\n        return a, a\n    }\n    return [a], a\n}\np, q = g(1)";
    let errs = errors(src);
    assert!(errs.iter().any(|e| e.contains("value 1 that `g` returns is a number in one `return` and an array in another")), "{errs:?}");
}

#[test]
fn grad_follows_several_results() {
    // A def that returns several values, called inside differentiated code,
    // is inlined and its record split into one variable per value (grad.rs,
    // `scalarize`); so is a record built there. Checked against central
    // differences in every engine.
    let src = "record P { a: Num, v: Arr }
def lin(w: Arr, x) {
    if x > 0 {
        return w[0] * x + w[1], w * x
    }
    return w[1], w
}
def mk(w: Arr) -> P {
    return P { v: w * 2, a: w[0] }
}
def consts(c) {
    return c, 2 * c
}
def loss(w: Arr) {
    y, g = lin(w, 3.0)
    s = 0
    for k to 3 {
        y2, g2 = lin(g, k - 1)
        s = s + y2 * y2 + sum(g2 * g2)
        p, q = consts(k)
        s = s + p * q * y
    }
    r = mk(w)
    return y * y + s + r.a * sum(r.v)
}
def fd(w: Arr, i) {
    h = 1.0e-5
    wp = copy(w)
    wm = copy(w)
    wp[i] = wp[i] + h
    wm[i] = wm[i] - h
    return (loss(wp) - loss(wm)) / (2 * h)
}
w = [2.0, 1.0]
g = grad(loss, w)
show g
l, g2 = value_and_grad(loss, w)
show l
test \"against central differences\" {
    assert_near(g[0], fd(w, 0), 1.0e-6)
    assert_near(g[1], fd(w, 1), 1.0e-6)
    assert_near(g2[0], g[0], 0)
}
";
    prints("grad", src, "g: [244, 172]\nl: 365\nPASS against central differences\n");
}

#[test]
fn grad_refuses_a_recursive_tuple_def_in_a_loop() {
    let src = "def fib2(n) {\n    if n == 0 {\n        return 0, 1\n    }\n    p, c = fib2(n - 1)\n    return c, p + c\n}\ndef loss(w: Arr) {\n    s = 0\n    for k to 3 {\n        a, b = fib2(k)\n        s = s + a * w[0] + b * w[1]\n    }\n    return s * s\n}\nshow grad(loss, [1.0, 2.0])\n";
    let errs = errors(src);
    assert!(errs.iter().any(|e| e.contains("`grad` cannot keep the values `fib2` returns on its tape")), "{errs:?}");
}

#[test]
fn stop_ends_the_run_where_it_is_written() {
    // `stop("...")`: the unpacking check's stop, as a statement a program
    // (and py2viv's `raise`) can write
    let r = all("stop", "def f(x) {\n    if x < 0 {\n        stop(\"math domain error\")\n    }\n    return sqrt(x)\n}\nshow f(4)\nshow f(-1)\nshow 3\n");
    assert_eq!(r.code, Some(4), "{r:?}");
    assert_eq!(r.out, "2\n");
    assert!(r.err.starts_with("error: math domain error\n"), "{}", r.err);
    assert!(r.err.contains("p.viv:3:9"), "{}", r.err);
    let errs = errors("stop(3)");
    assert!(errs.iter().any(|e| e.contains("`stop` takes one string")), "{errs:?}");
}

#[test]
fn a_notebook_cell_takes_apart_the_values_of_a_def_an_earlier_cell_wrote() {
    // a notebook runs each cell after the ones before it (interp::run_cell),
    // and parses the cell on its own: the defs the earlier cells wrote still
    // count as returning several values, directly and through a def that
    // returns their result
    let first = "def two(x) {\n    return x, 2.0 * x\n}\n";
    let second = "def fwd(x) {\n    return two(x)\n}\np, q = fwd(3.0)\nprint(p + q)\na, b = two(1.0)\nprint(b)\n";
    let out = viv::interp::run_cell("", first).expect("the first cell runs");
    assert_eq!(out, "");
    let out = viv::interp::run_cell(first, second).expect("the second cell runs, continuing the session");
    assert_eq!(out.lines().filter(|l| !l.starts_with("__")).collect::<Vec<_>>(), ["9", "2"]);
    // and from a fresh session, which parses the earlier cells again
    let out = viv::interp::run_cell(&format!("{first}x = 0.0"), second).expect("the second cell runs in a new session");
    assert_eq!(out.lines().filter(|l| !l.starts_with("__")).collect::<Vec<_>>(), ["9", "2"]);
}

/// B290: `zv, zpv = plasma_z(zeta)` over lib/zfunc, whose complex results
/// inference finds only in its pass that defaults indexing, then `abs(zv)`:
/// abs said "a number" first and the unpack's complex was refused ("type
/// cannot change"). The math functions that take a complex number say so
/// only in the last pass now.
#[test]
fn complex_results_found_late_are_unpacked_and_taken_abs_of() {
    let dir = scratch("late_complex");
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(root().join("lib"), dir.join("lib")).unwrap();
    std::fs::write(
        dir.join("p.viv"),
        "import zfunc\n\ndef plasma_z(zeta: Complex) {\n    return zfunc.zf_z(zeta), zfunc.zf_zprime(zeta)\n}\n\n\
         test \"abs of an unpacked complex\" {\n    zv, zpv = plasma_z(0.7 + 0.4i)\n    assert_lt(abs(zv), 10.0)\n    assert_gt(abs(zpv), 0.0)\n    print(fmt(abs(zv)), fmt(abs(sqrt(zpv))))\n}\n",
    )
    .unwrap();
    let i = interp(&dir);
    assert_eq!(i.code, Some(0), "{i:?}");
    assert!(i.out.contains("PASS abs of an unpacked complex"), "{i:?}");
    if let Some(n) = native(&dir) {
        assert_eq!(i, n, "native differs from the interpreter");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
