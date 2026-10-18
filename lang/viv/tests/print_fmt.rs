// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `print` with text and numbers, `fmt`, and `show a, b` (crate::printfmt):
//! what each form writes, that the interpreter and
//! Viv -> C -> native write the same bytes, that the Rust formatter and its C
//! twin agree with each other and with C's own `%.*g`, and that the forms that
//! are not supported are refused with a location.
//!
//! The million-double sweep across all three engines and against CPython is
//! tools/print-fmt-sweep/run.sh; this is the part that runs with `cargo test`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn have_cc() -> bool {
    Command::new(cc()).arg("--version").output().is_ok()
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_print_fmt_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn interp(dir: &Path) -> (bool, String, String) {
    let o = Command::new(viv_bin()).current_dir(dir).args(["run", "-q", "p.viv"]).output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn native(dir: &Path) -> String {
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    let r = Command::new(dir.join("p")).output().unwrap();
    String::from_utf8_lossy(&r.stdout).into_owned()
}

/// The interpreter's output of `src`, after requiring C to write the same.
fn run(tag: &str, src: &str) -> String {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let (ok, out, err) = interp(&dir);
    assert!(ok, "the interpreter failed on\n{src}\n{err}");
    if have_cc() {
        assert_eq!(out, native(&dir), "the engines disagree on\n{src}");
    }
    out
}

/// The message both engines refuse `src` with.
fn refused(tag: &str, src: &str) -> String {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let (ok, _, err) = interp(&dir);
    assert!(!ok, "the interpreter accepted\n{src}");
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(!c.status.success(), "the compiler accepted\n{src}");
    let cerr = String::from_utf8_lossy(&c.stderr).into_owned();
    let first = |s: &str| s.lines().take(4).collect::<Vec<_>>().join("\n");
    assert_eq!(first(&err), first(&cerr), "the engines refuse differently");
    err
}

#[test]
fn text_and_numbers_on_one_line() {
    let out = run(
        "mixed",
        "loss = 0.6931471805599453\n\
         print(\"loss\", loss)\n\
         print(\"loss\", fmt(loss))\n\
         print(\"loss\", fmt(loss, 17))\n\
         print(\"t =\", fmt(1.5, \".3f\"), \"s\")\n\
         print(\"[\", fmt(0.1 + 0.2, \"10.4e\"), \"] [\", fmt(12, \"5d\"), \"] [\", fmt(0.25, \"8\"), \"]\", sep=\"\")\n\
         print()\n\
         print(\"text\")\n",
    );
    assert_eq!(
        out,
        "loss 0.693147\n\
         loss 0.6931471805599453\n\
         loss 0.69314718055994529\n\
         t = 1.500 s\n\
         [3.0000e-01] [   12] [    0.25]\n\
         \n\
         text\n"
    );
}

#[test]
fn arrays_booleans_complex() {
    let out = run(
        "values",
        "v = [1, 2.5, -0.0]\nM = [[1, 2], [3, 4]]\nok = 2 > 1\nz = 1 + 2i\n\
         print(\"v =\", v, \"M =\", M)\n\
         print(fmt(v, \".2f\"), fmt(M))\n\
         print(ok, z)\n",
    );
    assert_eq!(out, "v = [1, 2.5, 0] M = [[1, 2], [3, 4]]\n[1.00, 2.50, -0.00] [[1.0, 2.0], [3.0, 4.0]]\ntrue 1+2i\n");
}

#[test]
fn show_several_values() {
    let out = run("show", "a = 1.5\nb = 0.1 + 0.2\nv = [1, 2]\nshow a, b, v\nshow fmt(b), fmt(b, 17)\nshow a\n");
    assert_eq!(out, "a: 1.5  b: 0.3  v: [1, 2]\nb: 0.30000000000000004  b: 0.30000000000000004\na: 1.5\n");
}

#[test]
fn repr_is_pythons() {
    let out = run(
        "repr",
        "print(fmt(-0.0), fmt(1.0e16), fmt(1.0e15), fmt(1.0e-4), fmt(1.0e-5), fmt(4.9e-324), fmt(1.0 / 0.0), fmt(-1.0 / 0.0), fmt(0.0 / 0.0))\n",
    );
    assert_eq!(out, "-0.0 1e+16 1000000000000000.0 0.0001 1e-05 5e-324 inf -inf nan\n");
}

#[test]
fn one_number_is_the_print_it_was() {
    // the generated C of `print(x)` is unchanged: no `viv_print`, only viv_print_num
    let dir = scratch("old");
    std::fs::write(dir.join("p.viv"), "x = 0.1 + 0.2\nprint(x)\nprint(-0.0)\n").unwrap();
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success());
    let text = std::fs::read_to_string(dir.join("p.c")).unwrap();
    assert!(text.contains("viv_print_num(") && !text.contains("viv_print("), "{text}");
    assert_eq!(run("old2", "x = 0.1 + 0.2\nprint(x)\nprint(-0.0)\n"), "0.3\n0\n");
}

#[test]
fn a_programs_own_fmt_wins() {
    let out = run("own", "def fmt(x) {\n    return 2 * x\n}\nprint(\"twice:\", fmt(4))\ny = fmt(1)\nshow y\n");
    assert_eq!(out, "twice: 8\ny: 2\n");
}

#[test]
fn refusals_are_located() {
    // `fmt` is a string anywhere now, not only a print argument
    assert_eq!(run("outside", "x = 1.5\ny = fmt(x)\nshow y\nshow 2 * fmt(x)\n"), "y: '1.5'\n'1.51.5'\n");
    let e = refused("digits", "x = 1.5\nprint(fmt(x, 18))\n");
    assert!(e.contains("1 to 17 significant digits") && e.contains("p.viv:2:14"), "{e}");
    let e = refused("noprec", "x = 1.5\nprint(fmt(x, \".3\"))\n");
    assert!(e.contains("write \".3g\""), "{e}");
    // Python's flags are formats (sign, zero, fill and alignment, grouping)
    assert_eq!(run("sign", "x = -1.5\nprint(fmt(-x, \"+.2f\"), fmt(x, \"06.2f\"), fmt(1234.5, \",.1f\"))\n"), "+1.50 -01.50 1,234.5\n");
    let e = refused("alt", "x = 1.5\nprint(fmt(x, \"#.2f\"))\n");
    assert!(e.contains("is not supported"), "{e}");
    let e = refused("runtime", "x = 1.5\nn = 3\nprint(fmt(x, n))\n");
    assert!(e.contains("written in the source") && e.contains("p.viv:3:14"), "{e}");
    let e = refused("bool", "x = 1.5\nprint(fmt(x > 1, \".2f\"))\n");
    assert!(e.contains("`fmt` takes a real number or a real array"), "{e}");
    let e = refused("end", "x = 1.5\nprint(x, end=\"\")\n");
    assert!(e.contains("`print` has no `end=`") && e.contains("p.viv:2:10"), "{e}");
}

/// The C twin (`C_RUNTIME`) and the Rust formatter, on the same doubles, and
/// both against the C library's own `%.*g` for every precision 1 to 17.
#[test]
fn rust_and_c_twins_agree() {
    if !have_cc() {
        return;
    }
    let mut xs: Vec<f64> = vec![0.0, -0.0, 1.0, 0.1, 0.5, 2.5, 1e16, 1e15, 1e-4, 1e-5, 5e-324, 2.2250738585072014e-308, f64::MAX, 9007199254740993.0];
    let mut s = 0x2545F4914F6CDD1Du64;
    while xs.len() < 60_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = match xs.len() % 4 {
            0 => f64::from_bits(s),
            1 => (s >> 11) as f64 / 2f64.powi((s % 60) as i32),
            2 => ((s >> 40) as f64) / 10f64.powi((s % 20) as i32),
            _ => 2f64.powi((s % 2098) as i32 - 1074),
        };
        if x.is_finite() {
            xs.push(x);
        }
    }
    let dir = scratch("twins");
    let input: String = xs.iter().map(|x| format!("{:016x}\n", x.to_bits())).collect();
    std::fs::write(dir.join("in.txt"), input).unwrap();
    let prog = format!(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdarg.h>\n#include <float.h>\n#include <math.h>\n\
         {}\ntypedef struct {{ int rank; int dims[8]; int size; double* data; }} viv_arr;\n{}\n{}\n\
         int main(void) {{\n  unsigned long long b; char r[64];\n  while (scanf(\"%llx\", &b) == 1) {{\n    double x; memcpy(&x, &b, 8);\n    viv_repr(r, x); printf(\"%s\", r);\n    for (int p = 1; p <= 17; p++) {{ viv_fmt_g(r, sizeof r, x, p); printf(\" %s\", r); }}\n    for (int p = 1; p <= 17; p++) printf(\" %.*g\", p, x);\n    printf(\"\\n\");\n  }}\n  return 0;\n}}\n",
        viv::lower::EMBED_C,
        viv::printfmt::C_FMT_G,
        viv::printfmt::C_RUNTIME
    );
    std::fs::write(dir.join("t.c"), prog).unwrap();
    let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", "-O2", "-w", "t.c", "-lm", "-o", "t"]).output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let o = Command::new(dir.join("t")).stdin(std::fs::File::open(dir.join("in.txt")).unwrap()).output().unwrap();
    let c_out = String::from_utf8(o.stdout).unwrap();
    let (mut n, mut libc_ties) = (0, 0);
    for (line, &x) in c_out.lines().zip(&xs) {
        let mut want = viv::printfmt::repr(x);
        let mut gs = Vec::new();
        for p in 1..=17 {
            gs.push(viv::printfmt::g(x, p));
            want.push(' ');
            want.push_str(&gs[p as usize - 1]);
        }
        let fields: Vec<&str> = line.split(' ').collect();
        assert_eq!(fields[..18].join(" "), want, "x = {x:e} ({:016x})", x.to_bits());
        // the C library's own %.*g: the same digits, except that macOS keeps
        // a trailing zero after a tie (%.6g of 7890305 is 7.89030e+06 there)
        for (p, libc) in fields[18..].iter().enumerate() {
            if *libc != gs[p] {
                let (m, e) = libc.split_once('e').map(|(m, e)| (m, format!("e{e}"))).unwrap_or((libc, String::new()));
                let trimmed = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
                assert_eq!(format!("{trimmed}{e}"), gs[p], "libc %.{}g of {x:e} differs in its digits", p + 1);
                libc_ties += 1;
            }
        }
        n += 1;
    }
    assert_eq!(n, xs.len());
    eprintln!("{n} doubles; the C library's %.*g kept a trailing zero {libc_ties} times");
}

/// A whole number below 10^16 is its digits in `print`, `show`, `str`,
/// f-strings and arrays; `fmt(x)` and `repr(x)` stay Python's repr.
#[test]
fn whole_numbers_print_as_digits_everywhere() {
    let src = "x = 3.0\nxs = [1.0, 2.5]\nbig = 1059490.0\nprint(x, str(x), f\"{x}\", xs, f\"{xs}\", big, str(big))\nprint(fmt(x), repr(x), fmt(xs), -0.0, str(-0.0), 1e16, str(0.1 + 0.2))\n";
    assert_eq!(run("whole", src), "3 3 3 [1, 2.5] [1, 2.5] 1059490 1059490\n3.0 3.0 [1.0, 2.5] 0 0 1e+16 0.30000000000000004\n");
}
