// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Differential cross-check (the ATLAS/CMS detector): run the *same* whole Viv
//! program through two independent implementations —
//!   1. the reference C++ `viv` interpreter (viv-lang-ref), and
//!   2. our kit-viv pipeline (build IR -> emit C -> compile -> run)
//! — and require their printed output agrees. A disagreement is a real bug in
//! one of them (or a hallucinated semantics on our side), surfaced precisely.
//!
//! Now that both languages share the `def`/`return`/`print` scalar surface, the
//! shared unit is a whole program: one `.viv` source feeds both sides.
//!
//! Skips cleanly if the reference binary or a C compiler is unavailable.

use std::path::PathBuf;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

/// Whole programs, double-only, that both implementations can run. They print
/// numbers; the harness compares the full sequence of printed values.
const CORPUS: &[&str] = &[
    // Arithmetic: precedence, associativity, unary minus, nesting.
    "print(2.0 + 3.0 * 4.0)\n\
     print((2.0 + 3.0) * 4.0)\n\
     print(2.0 - 3.0 - 4.0)\n\
     print(12.0 / 2.0 / 3.0)\n\
     print(-(2.0) + 5.0)\n\
     print(-(-(5.0)))\n\
     print(2.0 + -3.0 + 4.0)\n",
    // Functions and cross-calls.
    "def area(r) { return 3.14159 * r * r }\n\
     def twice(n) { return n + n }\n\
     print(area(2.0))\n\
     print(twice(21.0))\n\
     print(area(3.0) + twice(5.0))\n",
    // Locals: a declared-then-used temporary.
    "def hyp2(a, b) {\n  s = a * a + b * b\n  return s\n}\nprint(hyp2(3.0, 4.0))\n",
    // Locals reassigned in sequence.
    "def f(x) {\n  y = x + 1.0\n  y = y * 2.0\n  return y\n}\nprint(f(10.0))\n",
    // Recursion + if/else (early return).
    "def fact(n) {\n  if n <= 1.0 { return 1.0 }\n  return n * fact(n - 1.0)\n}\nprint(fact(6.0))\n",
    // while loop with accumulator (function-scoped locals).
    "def sumto(n) {\n  s = 0.0\n  i = 1.0\n  while i <= n {\n    s = s + i\n    i = i + 1.0\n  }\n  return s\n}\nprint(sumto(10.0))\n",
    // for loop with break and continue, and boolean operators in conditions.
    "def f(n) {\n  s = 0.0\n  for i to n {\n    if i == 2.0 { continue }\n    if i >= 7.0 and i > 0.0 { break }\n    s = s + i\n  }\n  return s\n}\nprint(f(100.0))\n",
    // Branch-introduced binding, used after the branch (function scoping).
    "def g(x) {\n  if x > 0.0 { y = 1.0 } else { y = 2.0 }\n  return y\n}\nprint(g(5.0))\nprint(g(-5.0))\n",
    // Arrays: vector + matrix literals, indexing, mutation, zeros, loop-fill.
    "v = [10.0, 20.0, 30.0]\nv[1] = 99.0\nprint(v[0])\nprint(v[1])\n\
     A = [[1.0, 2.0], [3.0, 4.0]]\nA[0, 1] = 7.0\nprint(A[0, 1])\nprint(A[1, 0])\n\
     z = zeros(2, 3)\nz[1, 2] = 5.0\nprint(z[1, 2])\n\
     acc = zeros(4)\nfor i to 4 { acc[i] = i * i }\ns = 0.0\nfor i to 4 { s = s + acc[i] }\nprint(s)\n",
    // Matrix arithmetic, matmul (@), transpose, scalar multiple, vector divide.
    "A = [[1.0, 2.0], [3.0, 4.0]]\nB = [[5.0, 6.0], [7.0, 8.0]]\n\
     C = A + B\nprint(C[0, 0])\nprint(C[1, 1])\n\
     D = A @ B\nprint(D[0, 1])\nprint(D[1, 0])\n\
     F = transpose(A) @ B @ A\nprint(F[0, 0])\nprint(F[1, 1])\n\
     E = 2.0 * A\nprint(E[1, 1])\n\
     u = [2.0, 4.0, 6.0]\nw = u / 2.0\nprint(w[2])\n",
    // Array-typed function params AND returns, inferred with no annotations.
    "def dot(a, b) {\n  s = 0.0\n  for i to 3 { s = s + a[i] * b[i] }\n  return s\n}\n\
     def scale_vec(v, k) {\n  r = zeros(3)\n  for i to 3 { r[i] = v[i] * k }\n  return r\n}\n\
     def norm_sq(x) { return dot(x, x) }\n\
     u = [1.0, 2.0, 3.0]\nw = [4.0, 5.0, 6.0]\n\
     print(dot(u, w))\nprint(norm_sq(u))\n\
     y = scale_vec(u, 10.0)\nprint(y[0])\nprint(y[2])\n",
    // Docstring, compound assignment, and math builtins.
    "def poly(x) {\n  \"2*(x + x*x)\"\n  s = 0.0\n  s += x\n  s += x * x\n  s *= 2.0\n  return s\n}\n\
     print(poly(3.0))\nprint(sqrt(16.0))\nprint(pow(2.0, 10.0))\nprint(exp(0.0))\n\
     y = 5.0\ny -= 2.0\ny /= 3.0\nprint(y)\n",
    // Closures + simpson: a capturing lambda (the Boys-function pattern), a
    // capture-free lambda, and a lambda capturing a global.
    "def boys(t_param) { return simpson(fn(t) -> exp(-t_param * t * t), 0.0, 1.0, 60) }\n\
     print(boys(0.5))\n\
     print(simpson(fn(x) -> x * x, 0.0, 1.0, 100))\n\
     k = 3.0\nprint(simpson(fn(x) -> k * x, 0.0, 2.0, 100))\n",
];

#[test]
fn kit_and_reference_agree_on_programs() {
    let Some(reference) = reference_binary() else {
        eprintln!("skipping differential test: no viv-lang-ref `viv` binary");
        return;
    };
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let dir = std::env::temp_dir().join(format!("viv_diff_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut mismatches = Vec::new();
    let mut compared = 0;

    for (i, src) in CORPUS.iter().enumerate() {
        let reference_out = match run_reference(&reference, &dir, i, src) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skipping: reference run failed on program {i}: {e}");
                return;
            }
        };
        let kit_out = match run_kit(&cc, &dir, i, src) {
            Ok(Some(v)) => v,
            Ok(None) => {
                eprintln!("skipping differential test: no C compiler `{cc}`");
                return;
            }
            Err(e) => panic!("kit-viv failed on program {i}:\n{src}\n{e}"),
        };

        compared += 1;
        if reference_out.len() != kit_out.len() || !lists_eq(&reference_out, &kit_out) {
            mismatches.push(format!(
                "program {i}:\n{src}  reference={reference_out:?}\n  kit={kit_out:?}"
            ));
        }
    }

    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        mismatches.is_empty(),
        "differential mismatches ({} of {} programs):\n{}",
        mismatches.len(),
        compared,
        mismatches.join("\n")
    );
    assert!(compared > 0, "no programs were compared");
    eprintln!("differential: {compared} programs agree across both implementations");
}

fn lists_eq(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
}

/// Relative tolerance is generous because the reference prints ~6 significant
/// digits; a genuine disagreement is orders larger.
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
}

/// Numeric output lines. Non-numeric lines (e.g. `narrate`'s `# ...` commentary)
/// are ignored, so the comparison is over computed values only.
fn parse_numbers(stdout: &str) -> Result<Vec<f64>, String> {
    Ok(stdout
        .lines()
        .map(|l| l.trim())
        .filter_map(|l| l.parse::<f64>().ok())
        .collect())
}

fn reference_binary() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KNOT_REF") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../viv-lang-ref/viv");
    p.exists().then_some(p)
}

fn run_reference(bin: &PathBuf, dir: &PathBuf, idx: usize, src: &str) -> Result<Vec<f64>, String> {
    let path = dir.join(format!("ref_{idx}.viv"));
    std::fs::write(&path, src).map_err(|e| e.to_string())?;
    let out = Command::new(bin).arg(&path).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    parse_numbers(&String::from_utf8_lossy(&out.stdout))
}

/// Build the program with kit-viv, emit C (which contains its own `main`),
/// compile, run, parse printed numbers. Returns Ok(None) if no C compiler.
fn run_kit(cc: &str, dir: &PathBuf, idx: usize, src: &str) -> Result<Option<Vec<f64>>, String> {
    let program = viv::build_ir(src, SourceId(0))
        .map_err(|d| format!("{:?}", d.iter().map(|x| &x.message).collect::<Vec<_>>()))?;
    let emission = CBackend.emit(&program, &MemoryIo::new()).map_err(|_| "C emission failed")?;
    let c = emission.artifact.as_c().ok_or("not C")?;

    let c_path = dir.join(format!("prog_{idx}.c"));
    let bin_path = dir.join(format!("run_{idx}"));
    std::fs::write(&c_path, c).map_err(|e| e.to_string())?;

    let compile = Command::new(cc)
        .args(["-std=c99", "-O2"])
        .arg(&c_path)
        .arg("-o")
        .arg(&bin_path)
        .output();
    let compile = match compile {
        Ok(o) => o,
        Err(_) => return Ok(None),
    };
    if !compile.status.success() {
        return Err(format!("C did not compile:\n{}\n--- C ---\n{c}", String::from_utf8_lossy(&compile.stderr)));
    }
    let run = Command::new(&bin_path).output().map_err(|e| e.to_string())?;
    parse_numbers(&String::from_utf8_lossy(&run.stdout)).map(Some)
}
