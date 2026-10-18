// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Language and tooling fixes the Hartree-Fock library leans on.
//!
//! Each test is one failure measured while planning lib/hf, pinned so it stays
//! fixed: a long literal that took clang 11 s, an `apply` with 18 arguments that
//! printed nan, a duplicate def the interpreter ran anyway, a module test that
//! leaked into its importer, a rank-5 array that panicked the compiler, a def
//! that read a global the compiled code could not see, registry results that
//! needed `-> Arr`, and a tool that stopped at the first clang error.
//!
//! Every program here that runs is run in both engines, and the outputs must
//! match byte for byte.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// ---------------------------------------------------------------- helpers

/// A fresh scratch directory for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_hf_lang_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p
}

/// The `viv` binary this test run built.
fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// Build C text with `cc -O2` and run it. Returns (stdout, seconds cc took), or
/// `None` when there is no C compiler, so the suite passes on a machine without one.
fn build_and_run(c: &str, dir: &Path) -> Option<(String, f64)> {
    let src = write(dir, "p.c", c);
    let t0 = Instant::now();
    let built = Command::new(cc()).args(["-std=c99", "-O2"]).arg(&src).arg("-lm").arg("-o").arg(dir.join("p")).output().ok()?;
    let secs = t0.elapsed().as_secs_f64();
    assert!(built.status.success(), "generated C did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    let out = Command::new(dir.join("p")).output().unwrap();
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), secs))
}

fn interp(src: &str) -> String {
    viv::interp::run(src)
        .expect("should interpret")
        .lines()
        .filter(|l| !l.starts_with("__"))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// Run `src` in both engines; they must print the same bytes. Returns the output.
fn agree(src: &str, tag: &str) -> String {
    let i = interp(src);
    let c = viv::compile_c(src).unwrap_or_else(|e| panic!("should compile:\n{e}"));
    if let Some((n, _)) = build_and_run(&c, &scratch(tag)) {
        assert_eq!(i, n, "the engines disagree");
    }
    i
}

// ---------------------------------------------------------------- 1. literals

/// 18,145 values: the packed-8 ERI vector of water 6-31G* in Cartesian d, the
/// largest table lib/hf plans to carry. Emitted as one variadic `viv_vec` call,
/// `cc -O2` took 10.99 s on it.
const BIG_N: usize = 18_145;

fn big_literal_program() -> String {
    // A fixed linear congruential sequence, so the table is the same every run;
    // `{:?}` writes the shortest string that reads back to the same double.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut vals = Vec::with_capacity(BIG_N);
    for _ in 0..BIG_N {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        vals.push(format!("{:?}", 2.0 * u - 1.0));
    }
    format!(
        "def eri_table() {{\n    return [{}]\n}}\nT = eri_table()\nshow len(T)\nshow T[0]\nshow T[{}]\ns = 0.0\nfor i to len(T) {{\n    s = s + T[i]\n}}\nshow s\n",
        vals.join(", "),
        BIG_N - 1
    )
}

#[test]
fn a_long_literal_compiles_as_a_table_in_under_a_second() {
    let src = big_literal_program();
    let c = viv::compile_c(&src).expect("should compile");
    assert!(c.contains("viv_vec_tab(18145, (const double[]){"), "the literal should be emitted as a table");
    let i = interp(&src);
    let Some((n, secs)) = build_and_run(&c, &scratch("biglit")) else { return };
    assert_eq!(i, n, "the engines disagree");
    assert!(i.starts_with("18145\n"), "{i}");
    assert!(secs < 1.0, "cc -O2 took {secs:.2} s on the 18,145-element literal");
}

/// 100,000 Int constants, more than a C call can carry: as a variadic
/// `viv_ivec` call clang's arm64 code read garbage past 65,535 arguments (a
/// tree model's node indices came back as 6157775296) and took 10 s.
#[test]
fn a_long_int_literal_is_a_table_and_reads_back() {
    const N: i64 = 100_000;
    let vals: Vec<String> = (0..N).map(|i| ((i * 7919) % 1_000_003 - 500_000).to_string()).collect();
    let src = format!(
        "A: [Int] = [{}]\ns: Int = 0\nfor i to len(A) {{\n    s = s + A[i]\n}}\nprint(s)\nprint(A[{}])\nprint(A[70000])\n",
        vals.join(", "),
        N - 1
    );
    let c = viv::compile_c(&src).expect("should compile");
    assert!(c.contains("viv_ivec_tab(100000, (const long long[]){"), "the literal should be emitted as a table");
    let want: i64 = (0..N).map(|i| (i * 7919) % 1_000_003 - 500_000).sum();
    let i = interp(&src);
    assert!(i.starts_with(&format!("{want}\n")), "{i}");
    let Some((n, secs)) = build_and_run(&c, &scratch("bigint")) else { return };
    assert_eq!(i, n, "the engines disagree");
    assert!(secs < 2.0, "cc -O2 took {secs:.2} s on the 100,000-element Int literal");
}

#[test]
fn a_short_literal_keeps_the_variadic_form() {
    let c = viv::compile_c("v = [1.0, -2.0, 3.0]\nshow v[1]\n").unwrap();
    assert!(c.contains("viv_vec(3,"), "{c}");
    assert!(!c.contains("viv_vec_tab"), "{c}");
}

#[test]
fn a_long_matrix_literal_and_a_non_constant_one_agree() {
    // 4 x 4 constants (a table), then 16 elements with a variable in them
    // (variadic, because a table needs constants).
    let mut src = String::from("M = [");
    for r in 0..4 {
        let row: Vec<String> = (0..4).map(|c| format!("{:?}", (r * 4 + c) as f64 - 7.5)).collect();
        src.push_str(&format!("[{}]{}", row.join(", "), if r < 3 { ", " } else { "]\n" }));
    }
    src.push_str("x = 2.0\nv = [x, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]\n");
    src.push_str("show M[3, 2]\nshow M[0, 1]\nshow v[0] + v[15]\n");
    let c = viv::compile_c(&src).unwrap();
    assert!(c.contains("viv_mat_tab(4, 4,"), "{c}");
    assert!(c.contains("viv_vec(16,"), "{c}");
    // M[3, 2] = 3*4 + 2 - 7.5, M[0, 1] = 1 - 7.5
    assert_eq!(agree(&src, "matlit"), "6.5\n-6.5\n17\n");
}

// ---------------------------------------------------------------- 2. apply arity

/// Probe p3_18num: 18 numeric arguments through `apply`. `viv_apply` copies
/// into a 16-slot buffer, so the compiled program printed `nan` and `0.5`
/// where the interpreter printed 18017 and 18017.5.
const P3_18NUM: &str = "def s1(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17) {
    return a17 * 1000.0 + a16
}
def s2(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17) {
    return a17 * 1000.0 + a16 + 0.5
}
ms = [s1, s2]
show apply(ms[0], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
show apply(ms[1], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
";

#[test]
fn apply_with_more_than_16_arguments_is_refused_by_both_engines() {
    let dir = scratch("p3");
    let p = write(&dir, "p3_18num.viv", P3_18NUM);
    let out = Command::new(viv_bin()).arg(&p).arg("-o").arg(dir.join("x.c")).output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "should be refused");
    assert!(err.contains("16"), "{err}");
    assert!(!dir.join("x.c").exists(), "no C should be written");
    let e = viv::interp::run(P3_18NUM).unwrap_err();
    assert!(e.contains("at most 16"), "{e}");
}

#[test]
fn apply_with_16_arguments_still_works() {
    let src = "def s(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15) {
    return a15 * 100.0 + a14
}
ms = [s, s]
show apply(ms[1], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
";
    assert_eq!(agree(src, "p3_16"), "1615\n");
}

// ---------------------------------------------------------------- 3. duplicate defs

/// Two modules that define one def are two defs:
/// each module's top-level names are its own, so `ma.f` and `mb.f` are
/// different functions in both engines, and importing both by the one name
/// is an error at the import line.
#[test]
fn two_modules_defining_one_def_are_two_defs() {
    let dir = scratch("dup");
    write(&dir, "ma.viv", "def f(x) {\n    return x + 1.0\n}\n");
    write(&dir, "mb.viv", "def f(x) {\n    return x + 2.0\n}\n");
    let p = write(&dir, "prog.viv", "import ma\nimport mb\nshow ma.f(1.0)\nshow mb.f(1.0)\n");
    let run = Command::new(viv_bin()).arg("run").arg("-q").arg(&p).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout), "2\n3\n", "{}", String::from_utf8_lossy(&run.stderr));
    let comp = Command::new(viv_bin()).arg(&p).arg("-o").arg(dir.join("x.c")).output().unwrap();
    assert!(comp.status.success(), "{}", String::from_utf8_lossy(&comp.stderr));
    let both = write(&dir, "both.viv", "from ma import f\nfrom mb import f\nshow f(1.0)\n");
    let run = Command::new(viv_bin()).arg("run").arg(&both).output().unwrap();
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success());
    assert!(err.contains("`f` is imported twice"), "{err}");
}

#[test]
fn a_notebook_cell_may_still_redefine_a_def() {
    viv::interp::run_cell("", "def f(x) {\n    return x\n}\n").unwrap();
    let out = viv::interp::run_cell("def f(x) {\n    return x\n}\n", "def f(x) {\n    return 2.0 * x\n}\nshow f(3.0)\n").unwrap();
    assert!(out.starts_with("6\n"), "{out}");
}

// ---------------------------------------------------------------- 4. module tests

#[test]
fn a_module_test_with_a_comment_after_its_brace_is_stripped() {
    let dir = scratch("modtest");
    write(
        &dir,
        "mt.viv",
        "def g(x) {\n    return x * 2.0\n}\ntest \"module test #1\" {   # a note after the brace\n    show 2222\n    assert_near(g(1.0), 2.0, 1e-12)\n}\n",
    );
    let p = write(&dir, "prog.viv", "from mt import g\nshow g(3.0)\n");
    let out = Command::new(viv_bin()).arg("run").arg("-q").arg(&p).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "6\n");
}

// ---------------------------------------------------------------- 5. both-engines.sh

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

#[test]
fn both_engines_reports_a_clang_failure_and_goes_on() {
    if Command::new(cc()).arg("--version").output().is_err() {
        return;
    }
    let dir = scratch("be");
    // Viv accepts this, and clang rejects it: the def's C name collides with a
    // runtime function. If Viv starts refusing such names, pick another program
    // Viv accepts and clang does not.
    write(
        &dir,
        "a_clang_fail.viv",
        "def viv_arr_new(x) {\n    return x + 1.0\n}\nv = zeros(3)\nshow viv_arr_new(v[0])\ntest \"a\" {\n    assert_near(1.0, 1.0, 1e-12)\n}\n",
    );
    write(&dir, "c_ok.viv", "test \"ok\" {\n    assert_near(2.0, 2.0, 1e-12)\n}\nshow 5\n");
    let out = Command::new("sh")
        .arg(repo_root().join("tools/both-engines.sh"))
        .arg(&dir)
        .env("VIV", viv_bin())
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.lines().any(|l| l.starts_with("a_clang_fail") && l.contains("DOES NOT BUILD (cc)")), "{text}");
    assert!(
        text.lines().any(|l| l.starts_with("c_ok") && l.contains("1 tests pass, 0 failed; identical output")),
        "{text}"
    );
}

// ---------------------------------------------------------------- 6. numel, shape

#[test]
fn numel_and_shape_agree_in_both_engines() {
    let src = "show numel(zeros(2,3,4))
s = shape(zeros(2,3,4))
show s
def count(a: Arr) {
    return numel(a) + len(shape(a))
}
show count(zeros(5,2))
c = czeros(3,2)
show numel(c)
show shape(c)
ms = [count, count]
show numel(ms)
";
    assert_eq!(agree(src, "numel"), "24\ns: [2, 3, 4]\n12\n6\n[3, 2]\n2\n");
}

// ---------------------------------------------------------------- 7. rank 5

#[test]
fn rank_five_is_a_diagnostic_in_both_engines() {
    for (src, want) in [
        ("A = zeros(2,2,2,2,2)\n", "takes 1 to 4 dimensions"),
        ("A = czeros(2,2,2,2,2)\n", "takes 1 to 4 dimensions"),
        ("A = zeros(2,2,2,2)\nshow A[0,0,0,0,0]\n", "at most 4 indices"),
        ("A = zeros(2,2,2,2)\nA[0,0,0,0,1] = 3.0\n", "at most 4 indices"),
    ] {
        let e = viv::compile_c(src).expect_err("should be refused");
        assert!(e.contains(want), "{src}: {e}");
        assert!(viv::interp::run(src).is_err(), "{src}: the interpreter should refuse it too");
    }
}

// ---------------------------------------------------------------- 8. apply results

/// Probe p4: a def returns an apply result, and the registry travels through
/// two levels of defs. The compiler used to type `inner` as returning a number
/// unless it was written `-> Arr`.
const P4: &str = "def dbl(v, s) {
    out = zeros(len(v))
    for i to len(v) {
        out[i] = s * v[i]
    }
    return out
}
def add(v, s) {
    out = zeros(len(v))
    for i to len(v) {
        out[i] = s * 0.5 * (v[i] + v[i])
    }
    return out
}
def methods() {
    return [dbl, add]
}
def inner(ms, v, s, k) {
    return apply(ms[k], v, s)
}
def outer(ms, v, s) {
    n = len(ms)
    w = 0.0
    a = inner(ms, v, s, 0)
    for k to n {
        b = inner(ms, v, s, k)
        for i to len(v) {
            w = max(w, abs(a[i] - b[i]))
        }
    }
    return w
}
x = [1.0, 2.0, 3.0]
show outer(methods(), x, 3.0)
show outer([dbl, add], x, 3.0)
r = inner(methods(), x, 3.0, 1)
show r[2]
";

/// Probe p12: a top-level variable takes an array apply result and is never
/// indexed. It used to go undeclared in the C code.
const P12: &str = "def d2(v) {
    out = zeros(len(v))
    for i to len(v) {
        out[i] = v[i]
    }
    return out
}
ms = [d2, d2]
x = [1.0, 2.0, 3.0]
g = apply(ms[1], x)
show 7
";

#[test]
fn a_def_returning_apply_needs_no_annotation() {
    assert_eq!(agree(P4, "p4"), "0\n0\n9\n");
}

#[test]
fn a_top_level_apply_result_is_declared() {
    assert_eq!(agree(P12, "p12"), "7\n");
    // and read without indexing it first (p8a, p9, p10)
    let read = "def d2(v) {\n    out = zeros(len(v))\n    for i to len(v) {\n        out[i] = v[i]\n    }\n    return out\n}\nms = [d2, d2]\nx = [1.0, 2.0, 3.0]\ng = apply(ms[1], x)\nq = g\nt = take(q, [0.0, 2.0])\nshow t[1]\nshow len(g)\n";
    assert_eq!(agree(read, "p9"), "3\n3\n");
}

// ---------------------------------------------------------------- 9. c9

/// Probe c9: `f` reads the global `k`, then assigns a local `k`. The C code
/// hoists the local to the top of `f`, so the read saw an array there, and clang
/// rejected `k + n`. The interpreter read the global and printed 5.
const C9: &str = "k = 4.0
def f(n) {
    a = k + n
    k = zeros(2)
    return a + k[0]
}
print(f(1.0))
";

#[test]
fn a_global_read_before_its_local_is_refused_before_c_is_written() {
    let e = viv::compile_c(C9).expect_err("Viv should refuse it, so clang never sees it");
    assert!(e.contains("`k`"), "{e}");
    let e = viv::interp::run(C9).expect_err("the interpreter should refuse it too");
    assert!(e.contains("`k`"), "{e}");
    // With matching types the old C was accepted and silently read 0 for `k`.
    let same = "k = 4.0\ndef f(n) {\n    a = k + n\n    k = 2.0\n    return a + k\n}\nshow f(1.0)\n";
    assert!(viv::compile_c(same).is_err());
}

#[test]
fn a_local_assigned_on_every_path_before_its_read_is_fine() {
    let src = "k = 4.0
def f(n) {
    if n > 0 {
        k = 1.0
    } else {
        k = 3.0
    }
    return k
}
def g(n) {
    t = 0.0
    for k to n {
        t = t + k
    }
    return t
}
show f(0.0)
show g(3.0)
show k
";
    assert_eq!(agree(src, "c9ok"), "3\n3\nk: 4\n");
}

// ---------------------------------------------------------------- 10. registry kinds

#[test]
fn a_registry_signature_mismatch_names_the_kinds() {
    let src = "def uses(v, bp, n) {
    out = zeros(n)
    for i to n {
        out[i] = v[i] + bp[i]
    }
    return out
}
def ignores(v, bp, n) {
    out = zeros(n)
    for i to n {
        out[i] = v[i]
    }
    return out
}
ms = [uses, ignores]
r = apply(ms[1], [1.0, 2.0], [0.0, 0.0], 2)
show r[1]
";
    let e = viv::compile_c(src).expect_err("mismatched registry");
    assert!(e.contains("`ignores` is (Arr, Num, Num) -> Arr but `uses` is (Arr, Arr, Num) -> Arr"), "{e}");
    // The fix the note suggests works.
    let fixed = src.replace("def ignores(v, bp, n)", "def ignores(v, bp: Arr, n)");
    assert_eq!(agree(&fixed, "p7a"), "2\n");
}
