// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Validates the weft pipeline (weak form -> Viv -> C): the generated Poisson
//! solver must (a) reproduce the hand-written fem_poisson.viv bit-for-bit, and
//! (b) converge at the theoretical O(h²) rate. Skips if no C compiler.

use std::collections::HashMap;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

/// A directory for one compile-and-run. Two tests build the same size (nl16)
/// in parallel, so each run gets its own, or one test removes it while the
/// other is still compiling in it.
fn scratch(tag: &str) -> std::path::PathBuf {
    static RUN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let k = RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("weft_{}_{tag}_{k}", std::process::id()))
}

/// Compile Viv source to C, run it, return the `name: value` outputs.
fn run_viv(viv_src: &str, tag: &str) -> Option<HashMap<String, f64>> {
    let program = viv::build_ir(viv_src, SourceId(0)).expect("generated Viv should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let dir = scratch(tag);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let out = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("p.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(
        String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter_map(|l| {
                let (k, v) = l.split_once(':')?;
                Some((k.trim().to_string(), v.trim().parse::<f64>().ok()?))
            })
            .collect(),
    )
}

/// Like `run_viv` but preserves the output order (for the Newton residual trace).
fn run_viv_lines(viv_src: &str, tag: &str) -> Option<Vec<(String, f64)>> {
    let program = viv::build_ir(viv_src, SourceId(0)).expect("generated Viv should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let dir = scratch(tag);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let out = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("p.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(
        String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter_map(|l| {
                let (k, v) = l.split_once(':')?;
                Some((k.trim().to_string(), v.trim().parse::<f64>().ok()?))
            })
            .collect(),
    )
}

fn weft_nonlinear(n: u32) -> Option<Vec<(String, f64)>> {
    // Semilinear  -lap(u) + 1000 u^3 = f,  manufactured u = x(1-x)y(1-y).
    let src = format!(
        "mesh: unit_square({n})\n\
         coefficient f = 2.0 * (x * (1.0 - x) + y * (1.0 - y)) + 1000.0 * (x * (1.0 - x) * y * (1.0 - y)) * (x * (1.0 - x) * y * (1.0 - y)) * (x * (1.0 - x) * y * (1.0 - y))\n\
         residual: dot(grad(u), grad(v)) + 1000.0 * u * u * u * v - f * v\n\
         dirichlet: 0.0\n\
         exact: x * (1.0 - x) * y * (1.0 - y)\n\
         newton: 25\n"
    );
    let viv_src = weft::compile_to_viv(&src).expect("weft should compile");
    run_viv_lines(&viv_src, &format!("nl{n}"))
}

#[test]
fn newton_converges_quadratically() {
    // A correct symbolic Jacobian => quadratic convergence (residual roughly
    // squares each step). A wrong one => linear convergence or divergence. So this
    // is the test that `derivative(F, u)` is correct.
    let Some(out) = weft_nonlinear(16) else { return };
    let res: Vec<f64> = out.iter().filter(|(k, _)| k == "resnorm").map(|(_, v)| *v).collect();
    assert!(res.len() >= 3, "expected several Newton iterations, got {res:?}");
    assert!(*res.last().unwrap() < 1e-9, "Newton did not converge: {res:?}");
    assert!(res.len() <= 8, "too many iterations for Newton ({}), suggests a wrong Jacobian: {res:?}", res.len());
    // Some consecutive step must be (near-)quadratic: r_{k+1} <= 10 * r_k^2.
    let quadratic = res.windows(2).any(|w| w[0] < 1e-2 && w[0] > 1e-12 && w[1] <= 10.0 * w[0] * w[0]);
    assert!(quadratic, "no quadratic step found (Jacobian likely inconsistent): {res:?}");
}

#[test]
fn nonlinear_converges_at_second_order() {
    let Some(coarse) = weft_nonlinear(8) else { return };
    let fine = weft_nonlinear(16).expect("N=16");
    let last = |o: &[(String, f64)], key: &str| o.iter().rev().find(|(k, _)| k == key).unwrap().1;
    let ratio = last(&coarse, "rms") / last(&fine, "rms");
    assert!((3.0..=4.5).contains(&ratio), "convergence ratio {ratio} not ~4 (O(h^2))");
    assert!((last(&fine, "u_center") - 0.0625).abs() < 5e-3, "u_center = {}", last(&fine, "u_center"));
}

fn weft_poisson(n: u32) -> String {
    format!(
        "mesh: unit_square({n})\n\
         coefficient f = 2.0 * (x * (1.0 - x) + y * (1.0 - y))\n\
         bilinear: dot(grad(u), grad(v))\n\
         linear: f * v\n\
         dirichlet: 0.0\n\
         exact: x * (1.0 - x) * y * (1.0 - y)\n"
    )
}

fn run_weft(n: u32) -> Option<HashMap<String, f64>> {
    let viv_src = weft::compile_to_viv(&weft_poisson(n)).expect("weft should compile");
    run_viv(&viv_src, &format!("weft{n}"))
}

#[test]
fn weft_poisson_converges_at_second_order() {
    let Some(coarse) = run_weft(8) else { return };
    let fine = run_weft(16).expect("N=16");
    let ratio = coarse["rms"] / fine["rms"];
    assert!((3.0..=4.5).contains(&ratio), "convergence ratio {ratio} not ~4 (O(h^2))");
    assert!((fine["u_center"] - 0.0625).abs() < 5e-3, "u_center = {}", fine["u_center"]);
}

#[test]
fn weft_matches_hand_written_fem() {
    // The compiled weak form must reproduce the hand-coded FEM exactly: the form
    // compiler's element kernel is the same arithmetic, not an approximation.
    let Some(generated) = run_weft(8) else { return };
    let hand_src = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../viv/examples/fem_poisson.viv"),
    )
    .expect("fem_poisson.viv");
    let hand = run_viv(&hand_src, "hand8").expect("hand FEM");
    assert!((generated["rms"] - hand["rms"]).abs() < 1e-12, "rms: weft {} vs hand {}", generated["rms"], hand["rms"]);
    assert!((generated["u_center"] - hand["u_center"]).abs() < 1e-12, "u_center differs");
}

#[test]
fn rejects_unsupported_forms() {
    // v1 supports stiffness + load; a mass term `u*v` is out of scope and must error
    // with a clear message rather than silently miscompile.
    let src = "mesh: unit_square(4)\nbilinear: dot(grad(u), grad(v)) + u * v\nlinear: v\ndirichlet: 0.0\nexact: 0.0\n";
    let err = weft::compile_to_viv(src).unwrap_err();
    assert!(err.contains("single") || err.contains("unsupported"), "expected an unsupported-form error, got: {err}");
}
