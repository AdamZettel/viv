// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Performance tier 1: integer loop counters,
//! integer indices behind their guards, bounds checks proved away, the Int
//! checks the facts drop, the recursion guard and `clock()`, each checked to
//! print what it printed in all three engines: the interpreter (`viv run`),
//! Viv -> C -> native, and the direct wasm backend run in node (skipped when
//! node or a viv.wasm with the runtime exports is missing).

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
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_perf_int_{}_{tag}", std::process::id()));
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



// ------------------------------------------------------------ integer loop counters

#[test]
fn an_integer_counter_makes_the_interpreters_passes() {
    // a fractional bound runs up to the next whole number; 0, a negative
    // bound and NaN run none (viv_trip)
    prints("frac", "for i to 2.5 {\n    print(i)\n}\n", "0\n1\n2\n");
    prints("none", "for i to -3 {\n    print(i)\n}\nfor i to 0 {\n    print(i)\n}\nn = 0.0 / 0.0\nfor i to n {\n    print(i)\n}\nprint(9)\n", "9\n");
    // the body assigns the variable and the bound: the count and the
    // variable's value after the loop are the interpreter's
    prints("assign", "m = 3\nfor i to m {\n    m = 10\n    i = 7\n    print(i)\n}\nprint(i)\nprint(m)\n", "7\n7\n7\n7\n10\n");
    prints("int", "n: Int = 3\nfor j to n {\n    print(j)\n}\nfor k to n reversed {\n    print(k)\n}\n", "0\n1\n2\n2\n1\n0\n");
    // counters compared as integers
    prints("cmp", "for i to 3 {\n    for j to 3 {\n        if i == j and j < 2 {\n            print(i)\n        }\n    }\n}\n", "0\n1\n");
}

// ------------------------------------------------------------ integer indices

#[test]
fn an_integer_index_reads_what_the_double_index_read() {
    let v = "v = [10.0, 20.0, 30.0, 40.0]\n";
    // a parameter the call sites make an integer; from the end too
    prints("param", &format!("{v}def at(a: Arr, x) {{\n    return a[x]\n}}\nprint(at(v, 2))\nprint(at(v, -1))\n"), "30\n40\n");
    // a step variable, and a counter plus a variable
    prints("step", &format!("{v}p = 0\ns = 0.0\nfor i to 2 {{\n    s = s + v[p] + v[p + i]\n    p = p + 1\n}}\nprint(s)\n"), "70\n");
    // an index read from a table goes through its guard
    prints("table", &format!("{v}t = [3.0, 0.0]\nprint(v[t[0]] + v[t[1] + 1])\n"), "60\n");
}

#[test]
fn a_guard_that_fails_takes_the_double_path_and_its_message() {
    let f = "def f(a: Arr, x) {\n    s = 0.0\n    for i to 2 {\n        s = s + a[x + i]\n    }\n    return s\n}\nv = [10.0, 20.0, 30.0, 40.0]\n";
    stops("half", &format!("{f}print(f(v, 0))\nprint(f(v, 0.5))\n"), "30\n", "index 0.5 is not an integer");
    stops("big", &format!("{f}print(f(v, 3000000000))\n"), "", "index 3000000000 out of bounds for dimension 0 (size 4)");
    stops("nan", &format!("{f}print(f(v, 0.0 / 0.0))\n"), "", "index nan is not an integer");
    stops("inf", &format!("{f}print(f(v, 1.0 / 0.0))\n"), "", "index inf is not an integer");
    // a step variable runs out of the array
    stops("stepout", "v = zeros(3)\np = 0\nfor i to 5 {\n    v[p] = i\n    p = p + 1\n}\n", "", "index 3 out of bounds for dimension 0 (size 3)");
    // from the end, past the start
    stops("fromend", "v = zeros(3)\nfor i to 5 {\n    v[-1 - i] = i\n}\n", "", "index -4 out of bounds for dimension 0 (size 3)");
}

#[test]
fn an_int_index_past_2_53_stops_where_the_interpreter_converts_it() {
    // the interpreter converts an Int index to a number, which stops; C and
    // wasm rounded it and reported it out of bounds before this branch
    stops(
        "huge",
        "v = [1.0, 2.0]\nk: Int = 9007199254740993\nprint(v[k])\n",
        "",
        "converting 9007199254740993 to a number would change it; write float(n) to round it",
    );
    stops("intout", "v = [1.0, 2.0]\nk: Int = 5\nprint(v[k])\n", "", "index 5 out of bounds for dimension 0 (size 2)");
}

// ------------------------------------------------------------ bounds checks proved away

#[test]
fn a_proved_access_reads_what_the_checked_one_read() {
    // `j < i + 1 <= n` on zeros(n, n), and `len(a)` of a parameter
    let src = "def g(n) {\n    A = zeros(n, n)\n    for i to n {\n        for j to i + 1 {\n            A[i, j] = i + j\n        }\n    }\n    return A\n}\n\
               def tot(a: Arr) {\n    s = 0.0\n    for i to len(a) {\n        s = s + a[i]\n    }\n    return s\n}\n\
               A = g(4)\nprint(A[3, 2] + A[2, 0])\nprint(tot([1.0, 2.0, 3.0]))\n";
    prints("proved", src, "7\n6\n");
}

#[test]
fn what_the_proof_cannot_see_is_still_checked() {
    // a fractional size: zeros(2.5) has 2 elements, the loop makes 3 passes
    stops("fraczeros", "def h(n) {\n    A = zeros(n)\n    for i to n {\n        A[i] = 1\n    }\n    return A\n}\nprint(h(2.5))\n", "", "index 2 out of bounds for dimension 0 (size 2)");
    // the array is made again, smaller, inside the loop
    stops("remade", "def r(n) {\n    A = zeros(n)\n    for i to n {\n        if i == 1 {\n            A = zeros(1)\n        }\n        A[i] = 1\n    }\n    return 0\n}\nprint(r(3))\n", "", "index 1 out of bounds for dimension 0 (size 1)");
    // a parameter the body replaces
    stops("param", "def s(a: Arr) {\n    for i to len(a) {\n        a = zeros(1)\n        print(a[i])\n    }\n    return 0\n}\nprint(s(zeros(3)))\n", "0\n", "index 1 out of bounds for dimension 0 (size 1)");
}

// ------------------------------------------------------------ loop versioning
// An innermost loop that indexes arrays at its counter runs a version
// without those checks only after one test shows every such array is rank 1
// and covers every pass; `and` of comparisons there evaluates both sides
// (`&`). Otherwise the loop runs as it was, and stops where it stopped.

const VERSIONED: &str = "def f(a: [Int], b: Arr) -> Num {\n    s = 0.0\n    for i to len(a) {\n        if a[i] > 0 and b[i] > 0.0 {\n            s += b[i]\n            print(s)\n        }\n    }\n    return s\n}\n";

#[test]
fn a_versioned_loop_reads_what_the_checked_one_read() {
    prints("vok", &format!("{VERSIONED}a: [Int] = [1, 0, 3]\nprint(f(a, [1.0, 2.0, 3.0]))\n"), "1\n4\n4\n");
    // the short column is never read past its end: there `and` still stops at
    // the first false comparison
    prints("vguard", &format!("{VERSIONED}a: [Int] = [1, 2, 0, 0]\nprint(f(a, [1.0, 2.0]))\n"), "1\n3\n3\n");
}

#[test]
fn a_versioned_loop_stops_where_the_checked_one_stops() {
    // a column shorter than the loop: the passes before run, then the read past its end stops
    stops("vshort", &format!("{VERSIONED}a: [Int] = [1, 2, 3]\nprint(f(a, [1.0, 2.0]))\n"), "1\n3\n", "index 2 out of bounds for dimension 0 (size 2)");
    // a matrix where a column is read: the rank check stops it
    stops("vrank", &format!("{VERSIONED}a: [Int] = [1, 2]\nprint(f(a, zeros(2, 2)))\n"), "", "array has rank 2 but was indexed with 1 index/indices");
    // a write past the end of a shorter array
    stops("vwrite", "def w(a: Arr, b: Arr) {\n    for i to len(a) {\n        b[i] = a[i] * 2.0\n        print(b[i])\n    }\n    return 0\n}\nprint(w([1.0, 2.0, 3.0], zeros(2)))\n", "2\n4\n", "index 2 out of bounds for dimension 0 (size 2)");
    // an Int counter (the bound an Int) over a shorter Int column
    stops("vintctr", "def f(a: [Int], b: [Int]) -> Int {\n    n = int(len(a))\n    s: Int = 0\n    for i to n {\n        s += b[i]\n        print(s)\n    }\n    return s\n}\na: [Int] = [1, 2, 3]\nb: [Int] = [5, 6]\nprint(f(a, b))\n", "5\n11\n", "index 2 out of bounds for dimension 0 (size 2)");
    // an Int array's own length proves its reads; one pass more is still caught
    stops("intpast", "xs = int_zeros(3)\nfor i to 4 {\n    xs[i] = 1\n}\n", "", "index 3 out of bounds for dimension 0 (size 3)");
}

// ------------------------------------------------------------ Int checks the facts drop

#[test]
fn an_int_overflow_the_facts_cannot_rule_out_still_stops() {
    prints("counter", "n: Int = 4\ns: Int = 0\nfor i to n {\n    s = s + i + 1\n}\nprint(s)\n", "10\n");
    stops("ovf", "def big(n: Int) -> Int {\n    return n + 1\n}\nprint(big(9223372036854775807))\n", "", "integer overflow: 9223372036854775807 + 1 does not fit in an Int");
}

// ------------------------------------------------------------ the recursion guard

#[test]
fn the_recursion_limit_is_unchanged() {
    let src = "def down(n) {\n    if n == 0 {\n        return 0\n    }\n    return 1 + down(n - 1)\n}\nprint(down(2400))\nprint(down(2600))\n";
    stops("depth", src, "2400\n", "recursion deeper than 2500 nested calls");
}

// ------------------------------------------------------------ clock()

#[test]
fn clock_runs_only_in_a_timing_program() {
    refused("untimed", "t = clock()\nprint(t)\n", "`clock()` reads the wall time, which differs from run to run");
    prints("timed", "#! timing\nt0 = clock()\ns = 0.0\nfor i to 1000 {\n    s = s + i\n}\nt1 = clock()\nprint(t1 >= t0)\nprint(s)\n", "true\n499500\n");
}
