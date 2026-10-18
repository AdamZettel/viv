// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Language semantics that every engine must share, checked in all three:
//! the interpreter (`viv run`), Viv -> C -> native, and the direct wasm
//! backend run in node (skipped, with a note, when node or a viv.wasm with the
//! runtime exports is missing; VIV_WASM=<path> names one).
//!
//! - `and` / `or` short-circuit
//! - `%` floored modulo, `//` floor division, `^` power (R16)
//! - comparisons on arrays, `where`, `all`, `any` (R15)
//! - lib/random.viv and lib/stats.viv (R12)

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
    // under the target directory, which is inside the repository, so `#! use
    // lib/random` finds the repository root as it does for a program there
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_semantics_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// What one engine did: exit code, stdout, stderr.
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

/// A viv.wasm with the runtime exports, and node, or None.
fn wasm_runtime() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    Some(wasm)
}

/// The three engines on `src`: (interpreter, native, direct wasm if available).
fn engines(tag: &str, src: &str) -> (Ran, Ran, Option<Ran>) {
    let dir = scratch(tag);
    let p = dir.join("p.viv");
    std::fs::write(&p, src).unwrap();
    // by absolute path, so `#! use` finds the repository root from the file (a
    // bare `p.viv` does not yet: gap R14); the diagnostics
    // still name it `p.viv`
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
    (i, n, w)
}

/// Run `src` in every engine; each must exit 0 and print exactly `want`.
fn prints(tag: &str, src: &str, want: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), want, ""), "{tag}: interpreter");
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "{tag}: direct wasm\n{}", w.err);
    }
}

/// Run `src` in every engine; each must stop with exit 1, print `out` first
/// and report an error whose first line is `error: <msg>`. The interpreter
/// and native C must agree byte for byte, the location included. The wasm
/// module, run by tools/run-wasm.mjs, prints the same output and then the message
/// without a location, as it does for an index out of bounds.
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

/// The compiler refuses `src` with a message containing `msg`, and so does the
/// interpreter (at run time).
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

// ------------------------------------------------------------ and / or (R18)

#[test]
fn and_or_short_circuit_in_every_engine() {
    // the probe from study/idioms/thealgorithms-python: the guard keeps a[-1] from being read
    prints(
        "guard",
        "a = [1.0, 2.0]\nj = -1.0\nif j >= 0.0 and a[j] > 0.0 {\n    show j\n}\nshow 7.0\n",
        "7\n",
    );
    // the right side runs only when the left does not decide; a def with a side
    // effect counts how often it ran
    prints(
        "count",
        r#"calls = zeros(1)
def hit(v) {
    calls[0] = calls[0] + 1.0
    return v > 0.0
}
a = [1.0, 2.0]
j = -1.0
t = j >= 0.0 and a[j] > 0.0
show t
u = j < 0.0 or a[j] > 0.0
show u
show hit(-1.0) and hit(1.0)
show calls[0]
show hit(1.0) or hit(1.0)
show calls[0]
show hit(1.0) and hit(-1.0)
show calls[0]
show not (hit(-1.0) or hit(-1.0) and hit(1.0))
show calls[0]
k = 0.0
while k < 2.0 and a[k] > 0.0 {
    k += 1.0
}
show k
"#,
        "t: false\nu: true\nfalse\n1\ntrue\n2\nfalse\n4\ntrue\n6\nk: 2\n",
    );
}

// ------------------------------------------------------- % // ^ (R16)

#[test]
fn percent_is_floored_modulo_with_python_signs() {
    // every sign combination, fractions, and a remainder that rounds up to the
    // divisor; the values are Python's (`-7 % 3` is 2, `-1e-20 % 1` is 1.0)
    prints(
        "mod",
        "show -7 % 3\nshow 7 % -3\nshow -7 % -3\nshow 7 % 3\nshow -6 % 3\nshow 5.5 % 2\nshow -5.5 % 2\n\
         show -1e-20 % 1\nshow 1 % 0.1 == 0.09999999999999995\nk = -1.0\nshow (k - 1) % 5\n",
        "2\n-2\n-1\n1\n0\n1.5\n0.5\n1\ntrue\n3\n",
    );
    // `mod` stays C's fmod: the result takes the sign of the dividend
    prints("fmod", "show mod(-7, 3)\nshow mod(7, -3)\nshow mod(5.5, 2)\n", "-1\n1\n1.5\n");
}

#[test]
fn double_slash_is_floor_division_with_python_rounding() {
    // `1 // 0.1` is 9 in Python (floor(1 / 0.1) would be 10), and for integers
    // a == (a // b) * b + a % b
    prints(
        "floordiv",
        "show -7 // 2\nshow 7 // -2\nshow 7 // 2\nshow -7.5 // 2\nshow 1 // 0.1\ni = 0.0\nshow (i - 1) // 2\n\
         ok = true\nfor a to 21 {\n    for b to 7 {\n        x = a - 10\n        y = b - 3\n        if y != 0 {\n\
                    ok = ok and (x // y) * y + x % y == x\n        }\n    }\n}\nshow ok\n",
        "-4\n-4\n3\n-4\n9\n-1\nok: true\n",
    );
}

#[test]
fn caret_is_pow() {
    // tighter than unary minus on its left, right-associative, a signed exponent
    // on its right, elementwise on an array base, and inside `deriv`
    prints(
        "pow",
        "show 2 ^ 10\nshow -2 ^ 2\nshow 2 ^ -1\nshow 2 ^ 3 ^ 2\nx = [1.0, 2.0, 3.0]\nshow x ^ 2\n\
         show 3 * 2 ^ 2 + 1\nshow deriv(fn(t) -> t ^ 3, 2.0)\n",
        "1024\n-4\n0.5\n512\n[1, 4, 9]\n13\n12\n",
    );
}

#[test]
fn a_zero_divisor_stops_every_engine() {
    stops("mod0", "n = 0.0\nshow 5 % 3\nshow 7 % n\n", "2\n", "division by zero in `a % b`");
    stops("div0", "n = 0.0\nshow 5 // 3\nx = 1.0\ny = 7 // n + x\n", "1\n", "division by zero in `a // b`");
}

#[test]
fn percent_and_double_slash_take_numbers() {
    refused("modarr", "a = [1.0, 2.0]\nb = a % 2.0\nshow b\n", "`%`");
}

// ------------------------------------------ array comparisons, where, all, any (R15)

#[test]
fn comparisons_on_arrays_give_masks_of_ones_and_zeros() {
    // elementwise between arrays, and against a scalar on either side; the
    // result is an array of 1.0 and 0.0, so it multiplies: ReLU's adjoint on
    // any rank is dY * (Y > 0)
    prints(
        "cmp",
        "a = [1.0, -2.0, 3.0, 0.0]\nb = [1.0, 2.0, -3.0, 0.0]\nshow a > 0.0\nshow a == b\nshow a != b\n\
         show 0.0 < a\nshow 1.0 >= a\nshow a <= b\nshow a >= b\nshow a < b\ndY = [10.0, 20.0, 30.0, 40.0]\n\
         show dY * (a > 0.0)\nM = [[1.0, -1.0], [-2.0, 2.0]]\nshow sum(M > 0.0)\nm = M < 0.0\nshow m[1, 0]\n",
        "[1, 0, 1, 0]\n[1, 0, 0, 1]\n[0, 1, 1, 0]\n[1, 0, 1, 0]\n[1, 1, 0, 1]\n[1, 1, 0, 1]\n[1, 0, 1, 1]\n[0, 1, 0, 0]\n\
         [10, 0, 30, 0]\n2\n1\n",
    );
}

#[test]
fn all_any_and_where() {
    prints(
        "where",
        "a = [1.0, -2.0, 3.0, 0.0]\nb = [1.0, 2.0, -3.0, 0.0]\nshow all(a == a)\nshow all(a == b)\nshow any(a == b)\n\
         show any(a > 5.0)\nshow all(zeros(0))\nshow any(zeros(0))\nif all(a == a) and not any(a > 5.0) {\n    show 1\n}\n\
         show where(a > 0.0, a, 0.0)\nshow where(a > 0.0, 1.0, b)\nshow where(a > 0.0, a, b)\nshow where(a > 0.0, 7.0, -7.0)\n",
        "true\nfalse\ntrue\nfalse\ntrue\nfalse\n1\n[1, 0, 3, 0]\n[1, 2, 1, 0]\n[1, 2, 3, 0]\n[7, -7, 7, -7]\n",
    );
    // a program's own def of the name wins, in every engine
    prints("owndef", "def all(a: Arr) {\n    return 42.0\n}\nshow all([1.0])\n", "42\n");
}

#[test]
fn an_array_comparison_is_not_a_condition() {
    // `if a == b` on arrays stays an error, and the note names all(a == b)
    refused(
        "ifarr",
        "a = [1.0, 2.0]\nb = [1.0, 2.0]\nif a == b {\n    show 1\n}\n",
        "test it with `all(a == b)` or `any(a > 0)`",
    );
}

// ------------------------------------------------- lib/random, lib/stats (R12)

#[test]
fn lib_random_and_lib_stats_draw_the_same_in_every_engine() {
    // their own tests hold the draws to the Python replay (tools/random_replay.py)
    for lib in ["random", "stats"] {
        let src = std::fs::read_to_string(root().join(format!("lib/{lib}.viv"))).unwrap();
        let (i, n, w) = engines(lib, &src);
        assert_eq!(i.code, Some(0), "lib/{lib}: {i:?}");
        assert!(i.out.contains("PASS ") && !i.out.contains("FAIL "), "lib/{lib}: {}", i.out);
        assert_eq!(i, n, "lib/{lib}: native differs");
        if let Some(w) = w {
            assert_eq!(w.out, i.out, "lib/{lib}: direct wasm differs");
        }
    }
    // used as modules: the documented formulas, draw by draw (these values are
    // tools/random_replay.py's for seed 7)
    prints(
        "uses",
        "from random import arange, choice, pi, randint, randn_matrix, randperm, shuffle\nfrom stats import mean, std\nsrand(7.0)\nshow randint(-3.0, 4.0)\nshow randperm(5)\nx = arange(4)\nshuffle(x)\n\
         show x\nshow choice([0.5, 0.5])\nshow rand()\nZ = randn_matrix(2, 2)\nshow shape(Z)\nshow mean([1.0, 2.0, 6.0])\n\
         show std([1.0, 3.0])\nshow pi\n",
        "-3\n[1, 3, 2, 4, 0]\nx: [2, 0, 1, 3]\n0\n0.477543\n[2, 2]\n3\n1\npi: 3.14159\n",
    );
}

#[test]
fn the_python_replay_matches_the_libraries() {
    let py = if Command::new("python3").arg("--version").output().is_ok() { "python3" } else { return };
    let o = Command::new(py)
        .arg(root().join("tools/random_replay.py"))
        .arg("--check")
        .env("VIV", viv_bin())
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{out}{}", String::from_utf8_lossy(&o.stderr));
}

// ------------------------------------------------------------ type inference

#[test]
fn an_array_from_a_def_stays_an_array_through_pow_and_print() {
    // `print(q)` and `x ^ 2` used to fix q and u0 as numbers before the defs'
    // return types arrived
    prints(
        "late_ret_print",
        "def ramp() {\n    return linspace(0.0, 1.0, 3)\n}\nq = ramp()\nprint(q)\nx = ramp() * 2.0\nu0 = exp(x ^ 2)\nprint(u0)\n",
        "[0, 0.5, 1]\n[1, 2.71828, 54.5982]\n",
    );
}

#[test]
fn a_def_called_in_a_def_is_typed_by_the_later_call_site() {
    // gf's parameters are typed only by step's call, which is typed only by the
    // top level's; max(a, b) used to answer "a number" before either arrived
    prints(
        "callee_in_def",
        "def gf(a, b) {\n    return max(a, b)\n}\ndef step(u, r) {\n    f = gf(u, roll(u, -1))\n    return u - r * (f - roll(f, 1))\n}\n\
         def lo(a, b) {\n    return -min(a, b)\n}\nu = linspace(0.0, 1.0, 5)\nprint(step(u, 0.1))\nprint(lo(u, 0.5))\n",
        "[0.075, 0.225, 0.475, 0.725, 1]\n[0, -0.25, -0.5, -0.5, -0.5]\n",
    );
}

#[test]
fn complex_values_known_late_are_not_typed_real() {
    // the lib/lobpcg bugs: `r * factor` in a def reached through a function
    // parameter, `conj(v) @ transpose(v)` with v from a def, and
    // `linspace(0.0, STEPS, n)` with STEPS from a def
    prints(
        "late_complex",
        "def damp(r, x) {\n    factor = 1.0 / (1.0 + abs(x))\n    return r * factor\n}\ndef drive(pre, r, x) {\n    return pre(r, x)\n}\n\
         def wave(n) {\n    return complex(linspace(0.0, 1.0, n), linspace(1.0, 2.0, n))\n}\n\
         def block() {\n    return transpose(complex([[1.0, 0.0]], [[0.0, 1.0]]))\n}\ndef nsteps() {\n    return 4\n}\n\
         z = drive(damp, wave(3), linspace(0.0, 2.0, 3))\nprint(im(z))\n\
         test \"g\" {\n    v = block()\n    g = conj(v) @ transpose(v)\n    print(im(g))\n}\n\
         STEPS = nsteps()\nprint(linspace(0.0, STEPS, 5))\n",
        "[1, 0.75, 0.666667]\n[[0, 1], [-1, 0]]\nPASS g (no assertions)\n[0, 1, 2, 3, 4]\n",
    );
}

#[test]
fn an_uncalled_def_that_types_some_way_does_not_stop_the_program() {
    // a test-only helper whose parameter nothing types: guessed a real array,
    // conj() of it is refused, but as a complex array it types, and nothing
    // runs it
    prints(
        "uncalled",
        "def overlap_error(x) {\n    g = conj(x) @ transpose(x)\n    return max(abs(g))\n}\nprint(1.0)\n",
        "1\n",
    );
    // one that no typing fixes is still reported
    refused("uncalled_bad", "def f(n) {\n    return m\n}\nprint(1.0)\n", "unknown variable `m`");
}

#[test]
fn a_name_the_top_level_only_reads_is_unknown_in_every_engine() {
    // B299: `print(nosuch)` and `show x, nan` compiled to C reading a zero,
    // since the argument's type was guessed and the name made a global; the
    // interpreter reports the unknown name
    refused("unbound_print", "print(nosuch)\n", "variable `nosuch`");
    refused("unbound_show", "x = 1.0\nshow x, nan\n", "variable `nan`");
    // B306 (vivsmith's reducer): the same name, line and column from `viv
    // check`, the interpreter, and the C and wasm compiles
    let dir = scratch("unbound_located");
    std::fs::write(dir.join("p.viv"), "x = 1.0\nprint(zz)\n").unwrap();
    for args in [&["check", "p.viv"][..], &["run", "-q", "p.viv"], &["p.viv", "-o", "p.c"], &["p.viv", "-o", "p.wasm"]] {
        let r = ran(Command::new(viv_bin()).current_dir(&dir).args(args).output().unwrap());
        assert_ne!(r.code, Some(0), "{args:?} accepted an unknown name");
        assert!(r.err.contains("unknown variable `zz`") && r.err.contains("p.viv:2:7"), "{args:?} said\n{}", r.err);
    }
    // a name the top level binds later is still a global a def reads
    prints("bound_later", "def f() {\n    return k + 1.0\n}\nk = 2.0\nprint(f())\n", "3\n");
}

#[test]
fn cplx_names_the_complex_type_as_complex_does() {
    // B291: `viv features` and the reference call the type `Cplx`; only
    // `Complex` was taken in an annotation
    prints("cplx_name", "def half(z: Cplx) -> Cplx {\n    return z / 2.0\n}\ndef twice(z: Complex) -> Complex {\n    return 2.0 * z\n}\nshow half(twice(2.0 + 4.0i))\n", "2+4i\n");
}

#[test]
fn an_uncalled_def_with_an_array_and_a_number_parameter_does_not_stop_the_program() {
    // B281: `f` an array (`len(f)`), `s` a number passed to a def of
    // numbers; no single guess for both types the def, a mix does. A module
    // whose only caller is its test block, imported, is this program.
    prints(
        "uncalled_mixed",
        "def mx_w(th) {\n    return [th, 1.0 - th]\n}\ndef mx(f, s) {\n    n = len(f)\n    w = mx_w(s)\n    return n + len(w)\n}\nprint(1.0)\n",
        "1\n",
    );
}

// ------------------------------------------------------------ sort

#[test]
fn sort_is_stable_with_negative_zero_first_and_nan_last() {
    prints(
        "sort_order",
        "a = [3.0, 0.0, -1.0, -0.0, 2.0, 0.0, -0.0, 1.0]\nsort(a)\nprint(a)\n\
         print(1.0 / a[1], 1.0 / a[2], 1.0 / a[3], 1.0 / a[4])\n\
         nan = 0.0 / 0.0\nb = [2.0, nan, -3.0, nan, 1.0]\nsort(b)\nprint(b)\n",
        "[-1, 0, 0, 0, 0, 1, 2, 3]\n-inf -inf inf inf\n[-3, 1, 2, nan, nan]\n",
    );
}

#[test]
fn the_value_of_sort_is_refused_where_it_is_used() {
    refused("sort_value", "def f(w) {\n    return sort(w)\n}\nx = f([2.0, 1.0])\n", "`sort` sorts its array in place and has no value");
    refused("sort_assign", "a = [2.0, 1.0]\ns = sort(a)\n", "`s = copy(a)`, then `sort(s)`");
}

// ------------------------------------------------------------ show in C

#[test]
fn show_writes_a_tie_as_the_interpreter_does() {
    // macOS's printf("%g", 7890305.0) writes 7.89030e+06; the C runtime's show
    // now writes numbers with Viv's own formatter (printfmt.rs, viv_g). A
    // whole number is its digits, so the tie shows
    // where six significant digits are asked for, and in a number past 10^16
    prints(
        "show_tie",
        "x = 7890305.0\nshow x\nprint(x, fmt(x, 6))\nshow [x, -0.0, 1234565.0, 7890305e10]\nz = complex(x, -2.5)\nshow z\nfigure(\"t\", \"x\", \"y\")\n",
        "x: 7890305\n7890305 7.8903e+06\n[7890305, 0, 1234565, 7.8903e+16]\nz: 7890305-2.5i\nfigure: t ; x ; y\n",
    );
}

// ------------------------------------ found by py2viv's generated rule tests

#[test]
fn negation_and_pow_of_an_array_a_def_returned() {
    prints(
        "neg_def_array",
        "def f(a) {\n    return a * 2.0\n}\na = [1.0, 2.0]\nc = f(a)\nb = -c\nshow b\nd = f(a) ^ 2.0\nshow d\n",
        "b: [-2, -4]\nd: [4, 16]\n",
    );
}

#[test]
fn array_shapes_are_checked_in_every_engine() {
    // the C read past the shorter array and printed a result
    let pre = "u = [1.0, 2.0, 3.0]\nv = [1.0, 2.0]\nm = [[1.0, 2.0], [3.0, 4.0]]\nc = complex(v, zeros(2))\n";
    stops("shape_had", &format!("{pre}show u * v\n"), "", "array shape mismatch: [3] vs [2]");
    stops("shape_add", &format!("{pre}show v + u * 1.0\n"), "", "array shape mismatch: [2] vs [3]");
    stops("shape_div", &format!("{pre}show u / m\n"), "", "array shape mismatch: [3] vs [2, 2]");
    stops("shape_cmp", &format!("{pre}show u < v\n"), "", "array shape mismatch: [3] vs [2]");
    stops("shape_matmul", &format!("{pre}show m @ u\n"), "", "`@` shape mismatch: 2x2 @ 3x1");
    stops("shape_max", &format!("{pre}show max(u, v)\n"), "", "`max`: array shape mismatch: [3] vs [2]");
    stops("shape_min", &format!("{pre}show min(v, u)\n"), "", "`min`: array shape mismatch: [2] vs [3]");
    // a real array times a complex one names them in the order written
    stops("shape_real_cplx", &format!("{pre}show u * c\n"), "", "array shape mismatch: [3] vs [2]");
    stops("shape_cplx_real", &format!("{pre}show c * u\n"), "", "array shape mismatch: [2] vs [3]");
    prints("shape_ok", &format!("{pre}show v * v\nshow m @ v\nshow v * c\n"), "[1, 4]\n[5, 11]\n[1+0i, 4+0i]\n");
}

#[test]
fn max_and_min_of_two_zeros() {
    // wasm's max(-0.0, 0.0) was -0.0 (Rust's f64::max compiled to wasm)
    prints(
        "zero_max_min",
        "show 1.0 / max(-0.0, 0.0)\nshow 1.0 / max(0.0, -0.0)\nshow 1.0 / min(0.0, -0.0)\nshow 1.0 / min(-0.0, 0.0)\na = [-0.0, 0.0]\nshow 1.0 / max(a, 0.0)\nshow 1.0 / min(a, -0.0)\nshow 1.0 / max(a, [0.0, -0.0])\nshow 1.0 / max([-0.0, -0.0])\n",
        "inf\ninf\n-inf\n-inf\n[inf, inf]\n[-inf, -inf]\n[inf, inf]\n-inf\n",
    );
}

#[test]
fn negating_a_complex_array_keeps_the_sign_of_a_zero() {
    // C and wasm scaled by -1 + 0i, which gives 0 + 0i an imaginary part of +0
    prints(
        "cneg",
        "z = czeros(2)\ny = -z\nshow 1.0 / re(y)\nshow 1.0 / im(y)\nshow 1.0 / im(-y)\n",
        "[-inf, -inf]\n[-inf, -inf]\n[inf, inf]\n",
    );
}

#[test]
fn erf_and_gamma_at_infinity() {
    prints(
        "erf_gamma_inf",
        "inf = 1.0 / 0.0\nshow erf(inf)\nshow erf(-inf)\nshow erf(1e200)\nshow gamma(inf)\nshow gamma(200.0)\nshow erf(0.5)\nshow gamma(4.5)\n",
        "1\n-1\n1\ninf\ninf\n0.5205\n11.6317\n",
    );
}

#[test]
fn the_symmetric_eigensolvers_stop_on_a_nan() {
    // spectrum_sym's sort panicked the interpreter; the C exited 0 with nan
    stops(
        "spectrum_nan",
        "nan = 0.0 / 0.0\nm = [[1.0, nan], [nan, 2.0]]\ne = zeros(2)\nspectrum_sym(m, e)\nshow e\n",
        "",
        "`spectrum_sym`: the matrix has a NaN or an infinity",
    );
    stops(
        "eig_inf",
        "inf = 1.0 / 0.0\nm = [[1.0, 0.0], [0.0, inf]]\ne = zeros(2)\nv = zeros(2, 2)\neig_sym(m, e, v)\nshow e\n",
        "",
        "`eig_sym`: the matrix has a NaN or an infinity",
    );
}

#[test]
fn a_singular_lin_solve_stops_with_the_same_words_everywhere() {
    // the C wrote "lin_solve: ..." with no location, before the program's output
    stops(
        "lin_solve_singular",
        "print(\"before\")\nm = [[1.0, 2.0], [2.0, 4.0]]\nb = [1.0, 1.0]\nx = zeros(2)\nlin_solve(m, b, x)\nshow x\n",
        "before\n",
        "`lin_solve`: matrix is singular to working precision (pivot 1)",
    );
    stops(
        "lin_solve_multi_singular",
        "print(\"before\")\nm = [[1.0, 2.0], [2.0, 4.0]]\nb = [[1.0, 1.0], [1.0, 1.0]]\nx = zeros(2, 2)\nlin_solve_multi(m, b, x)\nshow x\n",
        "before\n",
        "`lin_solve_multi`: matrix is singular to working precision (pivot 1)",
    );
}

/// `[1.0] + row` is Python's list join, which Viv's `+` is not: refused
/// where the code shows it is a list (a one-element literal beside an array,
/// or literals of different lengths); vector arithmetic passes.
#[test]
fn a_list_join_written_with_plus_is_refused() {
    refused("join1", "row = [k * 1.0 for k to 3]\nr = [1.0] + row\n", "does not join them");
    refused("join2", "row = []\nrow = append(row, 2.0)\nr = row + [0.0]\n", "does not join them");
    refused("join3", "r = [1.0, 2.0] + [3.0, 4.0, 5.0]\n", "does not join them");
    refused("join4", "a = [k * 1.0 for k to 3]\nb = [3.0]\nc = a + b\n", "does not join them");
    // NumPy's broadcast of a one-element array stays
    prints("broadcast", "v = [1.0, 2.0]\nprint([10.0] + v, [1.0, 2.0] + [3.0])\n", "[11, 12] [4, 5]\n");
    prints("vector_add", "v = [1.0, 2.0] + [3.0, 4.0]\nw = v + [10.0, 20.0]\nprint(v, w, v + 1.0)\n", "[4, 6] [14, 26] [5, 7]\n");
}

#[test]
fn a_test_block_an_error_stops_fails() {
    // B344: the interpreter printed `PASS t (no assertions)` and then the
    // error; every engine now prints the test's FAIL line first
    stops("test_stop", "test \"t\" {\n    stop(\"boom\")\n}\n", "FAIL t -- an error stopped the program\n", "boom");
    stops(
        "test_stop_index",
        "a = [1.0, 2.0]\ntest \"ok\" {\n    assert_true(a[0] == 1.0)\n}\ntest \"t\" {\n    print(a[5])\n}\n",
        "PASS ok\nFAIL t -- an error stopped the program\n",
        "index 5 out of bounds for dimension 0 (size 2)",
    );
}
