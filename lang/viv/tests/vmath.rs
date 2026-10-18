// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv's libm has two copies, and they must be one algorithm: the Rust in
//! `src/vmath/` (the interpreter and the direct wasm backend call it) and the
//! C in `src/vmath.c` (a compiled program carries it). This compiles the C
//! with the flags `viv` uses, runs both on the same inputs, and requires every
//! result to agree in every bit (NaNs agree as NaNs).
//!
//! The inputs: special values, five random families (the sweep in
//! tools/libm-sweep: [-1, 1], [-10, 10], [-1000, 1000], magnitudes 1e-20 to
//! 1e20, and [-700, 700]), exp's ends ([-746, -708] and [709, 709.79]),
//! arguments near multiples of pi/2 and huge
//! arguments for the trig reduction, and pairs for pow, atan2 and hypot.

use viv::vmath;

const N: usize = 40_000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn inputs() -> Vec<f64> {
    let mut x = vec![
        0.0, -0.0, 1.0, -1.0, 0.5, 2.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::MIN_POSITIVE,
        5e-324, -5e-324, 1e-310, f64::MAX, -f64::MAX, 709.782712893384, 709.79, -745.13, -745.2, 710.5,
        1e22, 1e300, -1e300, 3.0e9, 1.0e15, 0.9999999999999999, 1.0000000000000002, 1e-8, 1e-300,
    ];
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    for k in 1..2000 {
        // near k * pi/2, where the reduction's cancellation is worst
        let c = k as f64 * std::f64::consts::FRAC_PI_2;
        x.push(c);
        x.push(f64::from_bits(c.to_bits() + 1));
        x.push(f64::from_bits(c.to_bits() - 1));
    }
    for _ in 0..N {
        x.push(r.next() * 2.0 - 1.0);
        x.push((r.next() * 2.0 - 1.0) * 10.0);
        x.push((r.next() * 2.0 - 1.0) * 1000.0);
        let s = if r.next() < 0.5 { -1.0 } else { 1.0 };
        x.push(s * 10f64.powf(r.next() * 40.0 - 20.0));
        x.push((r.next() * 2.0 - 1.0) * 700.0);
        // exp's two ends: results below 2^-1021 (subnormal ones among them),
        // and the last binade before overflow
        x.push(-708.0 - r.next() * 38.0);
        x.push(709.0 + r.next() * 0.79);
        // any bit pattern: huge trig arguments, subnormals, NaN payloads
        x.push(f64::from_bits(((r.next() * (1u64 << 53) as f64) as u64) << 11 | (r.next() * 2048.0) as u64));
    }
    x
}

fn pairs() -> Vec<(f64, f64)> {
    let sp = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-310, 1e308, 3.0, -3.0];
    let mut p: Vec<(f64, f64)> = sp.iter().flat_map(|&a| sp.iter().map(move |&b| (a, b))).collect();
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..N {
        p.push((r.next() * 10.0, (r.next() * 2.0 - 1.0) * 20.0));
        p.push(((r.next() * 2.0 - 1.0) * 1e3, ((r.next() * 40.0) - 20.0).round()));
        p.push(((r.next() * 2.0 - 1.0) * 5.0, (r.next() * 2.0 - 1.0) * 5.0));
        p.push((1.0 + (r.next() - 0.5) * 1e-6, (r.next() * 2.0 - 1.0) * 1e8));
    }
    p
}

const UNARY: [(&str, fn(f64) -> f64); 16] = [
    ("sin", vmath::sin),
    ("cos", vmath::cos),
    ("tan", vmath::tan),
    // a program's exp (C: the #define to viv_exp_cr), and musl's, which
    // cosh and sinh still call
    ("exp", vmath::exp),
    ("viv_exp", vmath::exp_musl),
    ("expm1", vmath::expm1),
    ("log", vmath::log),
    ("log10", vmath::log10),
    ("atan", vmath::atan),
    ("asin", vmath::asin),
    ("acos", vmath::acos),
    ("sinh", vmath::sinh),
    ("cosh", vmath::cosh),
    ("tanh", vmath::tanh),
    ("erf", vmath::erf),
    ("tgamma", vmath::gamma),
];
const BINARY: [(&str, fn(f64, f64) -> f64); 3] = [("pow", vmath::pow), ("atan2", vmath::atan2), ("hypot", vmath::hypot)];
// complex z -> (re, im); cabs is (|z|, 0)
const COMPLEX: [&str; 7] = ["cexp", "clog", "csqrt", "csin", "ccos", "ctan", "cabs"];

fn rust_complex(name: &str, re: f64, im: f64) -> (f64, f64) {
    match name {
        "cabs" => (vmath::cabs(re, im), 0.0),
        "cexp" => vmath::cexp(re, im),
        "clog" => vmath::clog(re, im),
        "csqrt" => vmath::csqrt(re, im),
        "csin" => vmath::csin(re, im),
        "ccos" => vmath::ccos(re, im),
        "ctan" => vmath::ctan(re, im),
        _ => unreachable!(),
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn write_f64s(path: &std::path::Path, v: impl Iterator<Item = f64>) {
    let bytes: Vec<u8> = v.flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn read_f64s(path: &std::path::Path) -> Vec<f64> {
    std::fs::read(path).unwrap().chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn the_c_twin_agrees_with_the_rust_bit_for_bit() {
    let dir = std::env::temp_dir().join(format!("viv_vmath_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let x = inputs();
    let p = pairs();
    write_f64s(&dir.join("x.bin"), x.iter().copied());
    write_f64s(&dir.join("p.bin"), p.iter().flat_map(|&(a, b)| [a, b]));

    // The C side: vmath.c exactly as a compiled program gets it, then a main
    // that writes each function's results.
    let mut c = String::from(include_str!("../src/vmath.c"));
    c.push_str("\n#include <stdio.h>\n#include <stdlib.h>\n");
    c.push_str(
        "static double *rd(const char *f, long *n) {\n    FILE *fp = fopen(f, \"rb\"); fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);\n    double *d = malloc(sz); if (fread(d, 1, sz, fp) != (size_t)sz) exit(2); fclose(fp); *n = sz / 8; return d;\n}\nstatic void wr(const char *f, const double *o, long n) { FILE *fp = fopen(f, \"wb\"); fwrite(o, 8, n, fp); fclose(fp); }\n",
    );
    c.push_str("int main(void) {\n    long n, m; double *x = rd(\"x.bin\", &n); double *p = rd(\"p.bin\", &m); m /= 2;\n    double *o = malloc(8 * (n > 2 * m ? n : 2 * m));\n");
    for (name, _) in UNARY {
        c.push_str(&format!("    for (long i = 0; i < n; i++) o[i] = {name}(x[i]);\n    wr(\"c_{name}.bin\", o, n);\n"));
    }
    for (name, _) in BINARY {
        c.push_str(&format!("    for (long i = 0; i < m; i++) o[i] = {name}(p[2 * i], p[2 * i + 1]);\n    wr(\"c_{name}.bin\", o, m);\n"));
    }
    for name in COMPLEX {
        c.push_str(&format!(
            "    for (long i = 0; i < m; i++) {{ viv_cparts u; u.z = {name}(viv_cmake(p[2 * i], p[2 * i + 1])); o[2 * i] = u.p[0]; o[2 * i + 1] = {im}; }}\n    wr(\"c_{name}.bin\", o, 2 * m);\n",
            im = if name == "cabs" { "0.0" } else { "u.p[1]" },
        ));
    }
    c.push_str("    return 0;\n}\n");
    // cabs returns a double; give it the complex shape the loop expects
    let c = c.replace("u.z = cabs(", "u.z = viv_cmake(0, 0); u.p[0] = cabs(");
    std::fs::write(dir.join("t.c"), c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let built = match std::process::Command::new(&cc).current_dir(&dir).args(["-std=c99", "-O2", "t.c", "-lm", "-o", "t"]).output() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("no C compiler ({cc}); the twin check did not run");
            return;
        }
    };
    assert!(built.status.success(), "vmath.c did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    let ran = std::process::Command::new(dir.join("t")).current_dir(&dir).status().unwrap();
    assert!(ran.success());

    let mut report = Vec::new();
    for (name, f) in UNARY {
        let cv = read_f64s(&dir.join(format!("c_{name}.bin")));
        let bad: Vec<usize> = (0..x.len()).filter(|&i| !same(f(x[i]), cv[i])).collect();
        if let Some(&i) = bad.first() {
            report.push(format!("{name}: {} of {} differ, first at x = {:e} ({:?}): Rust {:e}, C {:e}", bad.len(), x.len(), x[i], x[i].to_bits(), f(x[i]), cv[i]));
        }
    }
    for (name, f) in BINARY {
        let cv = read_f64s(&dir.join(format!("c_{name}.bin")));
        let bad: Vec<usize> = (0..p.len()).filter(|&i| !same(f(p[i].0, p[i].1), cv[i])).collect();
        if let Some(&i) = bad.first() {
            report.push(format!("{name}: {} of {} differ, first at {:?}: Rust {:e}, C {:e}", bad.len(), p.len(), p[i], f(p[i].0, p[i].1), cv[i]));
        }
    }
    for name in COMPLEX {
        let cv = read_f64s(&dir.join(format!("c_{name}.bin")));
        let bad: Vec<usize> = (0..p.len())
            .filter(|&i| {
                let (re, im) = rust_complex(name, p[i].0, p[i].1);
                !same(re, cv[2 * i]) || !same(im, cv[2 * i + 1])
            })
            .collect();
        if let Some(&i) = bad.first() {
            report.push(format!("{name}: {} of {} differ, first at {:?}", bad.len(), p.len(), p[i]));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(report.is_empty(), "the twins disagree:\n{}", report.join("\n"));
}

/// Quadruples (ar, ai, br, bi) for the complex operators: random parts over
/// 600 decades, parts far apart in size, and every pairing of special values.
fn quads() -> Vec<[f64; 4]> {
    let sp = [0.0, -0.0, 1.0, -3.0, 0.5, 1e300, -1e-300, 1e160, 1e-160, f64::MAX, 5e-324, 1e-310, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    let mut q = Vec::new();
    for &a in &sp {
        for &b in &sp {
            q.push([1.5, -2.0, a, b]);
            q.push([a, b, 3.0, 4.0]);
            q.push([a, 1.0, b, a]);
        }
    }
    let mut r = Rng(0x1234_5678_9abc_def1);
    let mut part = |r: &mut Rng, span: f64| {
        let s = if r.next() < 0.5 { -1.0 } else { 1.0 };
        s * 10f64.powf(r.next() * 2.0 * span - span)
    };
    for _ in 0..N {
        q.push([part(&mut r, 300.0), part(&mut r, 300.0), part(&mut r, 300.0), part(&mut r, 300.0)]);
        q.push([part(&mut r, 3.0), part(&mut r, 3.0), part(&mut r, 3.0), part(&mut r, 3.0)]);
        q.push([part(&mut r, 1.0), part(&mut r, 1.0), part(&mut r, 30.0), part(&mut r, 1.0)]);
    }
    q
}

/// `+ - * /` on complex scalars: vmath.c's viv_cadd, viv_csub, viv_cmul and
/// viv_cdiv (what a compiled program calls) against the Rust the interpreter
/// and the wasm runtime call, bit for bit. The division is Smith's algorithm;
/// C's own `/` on `double complex` is a different, platform-chosen one.
#[test]
fn complex_arithmetic_agrees_bit_for_bit() {
    let dir = std::env::temp_dir().join(format!("viv_vmath_cops_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let q = quads();
    write_f64s(&dir.join("q.bin"), q.iter().flatten().copied());
    let mut c = String::from(include_str!("../src/vmath.c"));
    c.push_str("\n#include <stdio.h>\n#include <stdlib.h>\n");
    c.push_str(
        "int main(void) {\n    FILE *fp = fopen(\"q.bin\", \"rb\"); fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);\n    double *q = malloc(sz); if (fread(q, 1, sz, fp) != (size_t)sz) return 2; fclose(fp);\n    long n = sz / 32; double *o = malloc(16 * n);\n",
    );
    for op in ["cadd", "csub", "cmul", "cdiv"] {
        c.push_str(&format!(
            "    for (long i = 0; i < n; i++) {{ viv_cparts u; u.z = viv_{op}(viv_cmake(q[4*i], q[4*i+1]), viv_cmake(q[4*i+2], q[4*i+3])); o[2*i] = u.p[0]; o[2*i+1] = u.p[1]; }}\n    fp = fopen(\"c_{op}.bin\", \"wb\"); fwrite(o, 16, n, fp); fclose(fp);\n"
        ));
    }
    c.push_str("    return 0;\n}\n");
    std::fs::write(dir.join("t.c"), c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let Ok(built) = std::process::Command::new(&cc).current_dir(&dir).args(["-std=c99", "-O2", "t.c", "-lm", "-o", "t"]).output() else {
        eprintln!("no C compiler ({cc}); the check did not run");
        return;
    };
    assert!(built.status.success(), "did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    assert!(std::process::Command::new(dir.join("t")).current_dir(&dir).status().unwrap().success());
    let rust: [(&str, fn(f64, f64, f64, f64) -> (f64, f64)); 4] = [
        ("cadd", |a, b, c, d| (a + c, b + d)),
        ("csub", |a, b, c, d| (a - c, b - d)),
        ("cmul", vmath::cmul),
        ("cdiv", vmath::cdiv),
    ];
    let mut report = Vec::new();
    for (op, f) in rust {
        let cv = read_f64s(&dir.join(format!("c_{op}.bin")));
        let bad: Vec<usize> = (0..q.len())
            .filter(|&i| {
                let (re, im) = f(q[i][0], q[i][1], q[i][2], q[i][3]);
                !same(re, cv[2 * i]) || !same(im, cv[2 * i + 1])
            })
            .collect();
        if let Some(&i) = bad.first() {
            report.push(format!("{op}: {} of {} differ, first at {:?}", bad.len(), q.len(), q[i]));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(report.is_empty(), "the twins disagree:\n{}", report.join("\n"));
    // Smith's algorithm keeps what the textbook formula loses
    assert_eq!(vmath::cdiv(1e300, 1e300, 1e200, 1e200), (1e100, 0.0));
    assert_eq!(vmath::cdiv(1.0, 2.0, 3.0, 0.0), (1.0 / 3.0, 2.0 / 3.0));
}

/// The functions keep their textbook values where those are exact, and stay
/// within an ulp or two of the platform's elsewhere (a sanity check that the
/// port is the libm it claims to be, not a bit-level contract).
#[test]
fn the_values_are_the_functions() {
    assert_eq!(vmath::exp(0.0), 1.0);
    assert_eq!(vmath::log(1.0), 0.0);
    assert_eq!(vmath::log10(1000.0), 3.0);
    assert_eq!(vmath::pow(2.0, 10.0), 1024.0);
    assert_eq!(vmath::hypot(3.0, 4.0), 5.0);
    assert_eq!(vmath::sin(0.0), 0.0);
    assert_eq!(vmath::atan2(1.0, 1.0), std::f64::consts::FRAC_PI_4);
    let mut r = Rng(12345);
    for _ in 0..10_000 {
        let x = (r.next() * 2.0 - 1.0) * 50.0;
        let ulps = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
        assert!(ulps(vmath::sin(x), x.sin()) <= 2, "sin({x})");
        assert!(ulps(vmath::exp(x), x.exp()) <= 2, "exp({x})");
        assert!(ulps(vmath::cosh(x), x.cosh()) <= 3, "cosh({x})");
        assert!(ulps(vmath::log(x.abs()), x.abs().ln()) <= 2, "log({x})");
    }
}

/// `exp` is correctly rounded (vmath/exp_cr.rs): on 4,000 inputs its
/// result is the double nearest e^x, which Python's decimal module computed
/// at 80 digits (tools/vmath/exp_table.py --cases). exp(1) is the double
/// nearest e (bugs/ledger.jsonl B292). tools/vmath/check_exp.py holds it to
/// mpmath on a denser set, with the hard cases.
#[test]
fn exp_cr_is_correctly_rounded() {
    assert_eq!(vmath::exp(1.0), std::f64::consts::E);
    assert_eq!(vmath::exp(1.0).to_bits(), vmath::exp_cr(1.0).to_bits());
    let cases = include_str!("vmath_exp_cases.txt");
    let mut wrong = Vec::new();
    for l in cases.lines() {
        let (x, y) = l.split_once(' ').unwrap();
        let x = f64::from_bits(u64::from_str_radix(x, 16).unwrap());
        let y = f64::from_bits(u64::from_str_radix(y, 16).unwrap());
        if vmath::exp_cr(x) != y {
            wrong.push(format!("exp_cr({x:e}) = {:e}, the nearest double is {y:e}", vmath::exp_cr(x)));
        }
    }
    assert!(wrong.is_empty(), "{} of {} wrong:\n{}", wrong.len(), cases.lines().count(), wrong.join("\n"));
}

/// fma: vmath.c's viv_fma (musl's integer fma, which compiled programs use)
/// against Rust's f64::mul_add (the interpreter's and the wasm runtime's),
/// on the quads' first three entries and on products made to cancel, with
/// subnormal, huge and zero operands among them.
#[test]
fn fma_agrees_bit_for_bit() {
    let dir = std::env::temp_dir().join(format!("viv_vmath_fma_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut t: Vec<[f64; 4]> = quads();
    for q in quads() {
        t.push([q[0], q[1], -(q[0] * q[1]), 0.0]);
        t.push([q[0] * 1e-300, q[1] * 1e-10, 0.0, 0.0]);
        t.push([q[0] * 1e300, q[1] * 1e10, -q[2], 0.0]);
    }
    write_f64s(&dir.join("q.bin"), t.iter().flatten().copied());
    let mut c = String::from("#include <stdint.h>\n#include <float.h>\n");
    c.push_str(include_str!("../src/vmath_fma.c"));
    c.push_str("\n#include <stdio.h>\n#include <stdlib.h>\n");
    c.push_str(
        "int main(void) {\n    FILE *fp = fopen(\"q.bin\", \"rb\"); fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);\n    double *q = malloc(sz); if (fread(q, 1, sz, fp) != (size_t)sz) return 2; fclose(fp);\n    long n = sz / 32; double *o = malloc(8 * n);\n    for (long i = 0; i < n; i++) o[i] = fma(q[4*i], q[4*i+1], q[4*i+2]);\n    fp = fopen(\"c_fma.bin\", \"wb\"); fwrite(o, 8, n, fp); fclose(fp);\n    return 0;\n}\n",
    );
    std::fs::write(dir.join("t.c"), c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let Ok(built) = std::process::Command::new(&cc).current_dir(&dir).args(["-std=c99", "-O2", "-ffp-contract=off", "t.c", "-o", "t"]).output() else {
        eprintln!("no C compiler ({cc}); the check did not run");
        return;
    };
    assert!(built.status.success(), "did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    assert!(std::process::Command::new(dir.join("t")).current_dir(&dir).status().unwrap().success());
    let cv = read_f64s(&dir.join("c_fma.bin"));
    let _ = std::fs::remove_dir_all(&dir);
    let bad: Vec<usize> = (0..t.len()).filter(|&i| !same(t[i][0].mul_add(t[i][1], t[i][2]), cv[i])).collect();
    assert!(bad.is_empty(), "{} of {} differ, first at {:?}", bad.len(), t.len(), bad.first().map(|&i| t[i]));
    // one rounding: 0.1 * 10 - 1 is 2^-54 fused, 0 unfused
    assert_eq!(0.1f64.mul_add(10.0, -1.0), 5.551115123125783e-17);
}
