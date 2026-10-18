// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Fast mode: the same C built with fused multiply-add,
//! the platform libm and BLAS, and `viv check --fast`, which holds it to the
//! exact engines. These tests need a C compiler; without one they say so and
//! pass.

use std::path::PathBuf;
use std::process::Command;

fn viv() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn have_cc() -> bool {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    Command::new(cc).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn program(tag: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_fast_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{tag}.viv"));
    std::fs::write(&p, src).unwrap();
    p
}

/// A product large enough for dgemm, a solve large enough for LAPACK, and
/// libm calls: every freedom fast mode takes.
const ALL_FREEDOMS: &str = "n = 48
A = zeros(n, n)
B = zeros(n, n)
b = zeros(n)
for i to n {
    b[i] = sin(i * 0.7)
    for j to n {
        A[i, j] = cos(i * 1.3 + j * 0.37)
        B[i, j] = exp(-0.01 * (i - j) * (i - j)) * pow(1.01, i)
    }
}
for i to n {
    A[i, i] = A[i, i] + n
}
C = A @ B
x = zeros(n)
lin_solve(A, b, x)
print(fmt(sum(C), 17), fmt(x[3], 17), fmt(C[5, 7], 17))
print(\"count\", n)
test \"the solve's residual\" {
    assert_allclose(A @ x, b, 1.0e-10)
}
";

#[test]
fn the_fast_build_is_within_tolerance_of_the_exact_engines() {
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let p = program("all", ALL_FREEDOMS);
    let out = Command::new(viv()).args(["check", "--fast", "--json"]).arg(&p).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.contains("\"ok\":true") && text.contains("\"references_agree\":true"), "{text}");
}

#[test]
fn without_its_three_freedoms_the_fast_build_prints_the_exact_bytes() {
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let p = program("none", ALL_FREEDOMS);
    let out = Command::new(viv()).args(["check", "--fast", "--without", "contract,libm,blas", "--json"]).arg(&p).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    // every number the same text in both runs
    let num = |key: &str| -> u64 {
        let k = text.find(&format!("\"{key}\":")).expect(key) + key.len() + 3;
        text[k..].chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap()
    };
    assert_eq!(num("numbers"), num("identical"), "{text}");
}

#[test]
fn a_changed_integer_or_test_result_fails_the_check() {
    // fast mode may move a number by its tolerance, never a count or a verdict
    let r = viv::fastcmp::compare("iterations 12\nPASS t\n", "iterations 13\nPASS t\n", 1e-10);
    assert!(!r.ok());
    let r = viv::fastcmp::compare("x 1.5\nPASS t\n", "x 1.5\nFAIL t\n", 1e-10);
    assert!(!r.ok());
}

#[test]
fn build_states_its_flags_and_makes_a_program_that_runs() {
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let flags = Command::new(viv()).args(["build", "--cc-flags", "--fast"]).output().unwrap();
    let f = String::from_utf8_lossy(&flags.stdout);
    assert!(f.contains("-DVIV_FAST") && f.contains("-ffp-contract=fast"), "{f}");
    assert!(!f.contains("-DVIV_RELEASE"), "fast mode keeps the bounds checks: {f}");
    let exact = Command::new(viv()).args(["build", "--cc-flags"]).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&exact.stdout).trim(), "-O3");
    let p = program("build", "x = exp(1.0)\nprint(fmt(x, 6))\n");
    let exe = p.with_extension("");
    let b = Command::new(viv()).args(["build", "--fast"]).arg(&p).arg("-o").arg(&exe).output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let run = Command::new(&exe).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout), "2.71828\n");
}

#[test]
fn the_c_is_the_same_text_in_both_modes() {
    // the mode is the compile line: the C carries both, behind VIV_FAST
    let c = viv::compile_c("a = zeros(2, 2)\nb = a @ a\nprint(exp(1.0))\n").unwrap();
    assert!(c.starts_with("#pragma STDC FP_CONTRACT OFF\n"));
    assert!(c.contains("#pragma clang fp contract(fast)"));
    assert!(c.contains("#if !defined(VIV_FAST) || defined(VIV_FAST_KEEP_LIBM)\n#define sin viv_sin"));
    assert!(c.contains("#ifdef VIV_FAST_LA"));
    // contraction turns on after Viv's libm, which turns it off for itself
    assert!(c.rfind("#pragma STDC FP_CONTRACT OFF").unwrap() < c.find("#pragma clang fp contract(fast)").unwrap());
}

#[test]
fn the_fast_build_fuses_multiply_adds_in_the_programs_own_arithmetic() {
    // (a a - c) 2^54 is 0 rounded twice and 1 fused; the input goes through
    // rand() so the C compiler cannot fold it
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let src = "def f(a, c) {\n    return (a * a - c) * pow(2.0, 54.0)\n}\nx = 27.0 + floor(rand() * 0.5)\nprint(f(1.0 + pow(2.0, -x), 1.0 + pow(2.0, 1.0 - x)))\n";
    let p = program("fma", src);
    let run = |extra: &[&str]| {
        let o = Command::new(viv()).args(["run", "--fast"]).args(extra).arg(&p).output().unwrap();
        String::from_utf8_lossy(&o.stdout).to_string()
    };
    assert_eq!(run(&[]), "1\n");
    assert_eq!(run(&["--without", "contract"]), "0\n");
}

#[test]
fn a_program_may_name_its_variables_what_system_headers_name_types() {
    // `wide`, `Fixed`, `Boolean` are types in MacTypes.h, which
    // <Accelerate/Accelerate.h> brings in; the fast build declares the BLAS
    // itself, so they stay free for a program's own names
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let p = program("names", "wide = zeros(40, 40)\nFixed = 2.0\nBoolean = 3.0\nfor i to 40 {\n    wide[i, i] = Fixed + Boolean\n}\nb = wide @ wide\nprint(sum(b))\n");
    let out = Command::new(viv()).args(["check", "--fast"]).arg(&p).output().unwrap();
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}
