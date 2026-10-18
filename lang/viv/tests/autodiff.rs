// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Validates the autodiff examples: forward mode (dual
//! numbers) and reverse mode (a tape) must produce exact derivatives, agree with
//! each other, and drive gradient descent to the known minimum.

use std::collections::HashMap;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn run(file: &str, tag: &str) -> Option<HashMap<String, f64>> {
    let src = std::fs::read_to_string(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples").join(file)).expect(file);
    let program = viv::build_ir(&src, SourceId(0)).expect("should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let dir = std::env::temp_dir().join(format!("viv_ad_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("a");
    let compile = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("a.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(compile.status.success(), "C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| {
                let (k, v) = l.split_once(':')?;
                Some((k.trim().to_string(), v.trim().parse::<f64>().ok()?))
            })
            .collect(),
    )
}

#[test]
fn forward_mode_is_exact() {
    let Some(v) = run("autodiff_forward.viv", "fwd") else { return };
    // f(x)=x^3-2x+exp(-x^2): f'(2) = 12 - 2 - 4 e^-4 = 9.926737.
    assert!((v["f_deriv"] - 9.926737).abs() < 1e-4, "f'(2) = {}", v["f_deriv"]);
    // g(x,y)=x^2 y + exp(xy): grad at (1.5,0.5) = (2.558467, 5.425401).
    assert!((v["dg_dx"] - 2.558467).abs() < 1e-4, "dg/dx = {}", v["dg_dx"]);
    assert!((v["dg_dy"] - 5.425401).abs() < 1e-4, "dg/dy = {}", v["dg_dy"]);
}

#[test]
fn automatic_deriv_builtin_is_exact() {
    // `deriv(fn(x) -> expr, x0)`: the compiler dualizes the expression. Same
    // results as the hand-written dual library, but automatic.
    let Some(v) = run("autodiff_deriv.viv", "deriv") else { return };
    assert!((v["d1"] - 9.926737).abs() < 1e-4, "f'(2) = {}", v["d1"]);
    assert!((v["d2"] - 0.948683).abs() < 1e-5, "d/dx sqrt(1+x^2)|_3 = {}", v["d2"]);
    assert!((v["x"] - 2.0_f64.sqrt()).abs() < 1e-4, "Newton sqrt(2) = {}", v["x"]);
}

#[test]
fn reverse_mode_matches_forward_and_optimizes() {
    let Some(v) = run("autodiff_reverse.viv", "rev") else { return };
    // Same gradient as forward mode, but from one backward pass.
    assert!((v["dg_dx"] - 2.558467).abs() < 1e-4, "dg/dx = {}", v["dg_dx"]);
    assert!((v["dg_dy"] - 5.425401).abs() < 1e-4, "dg/dy = {}", v["dg_dy"]);
    // Gradient descent on (x-3)^2 + 2(y+1)^2 reaches the minimum (3, -1).
    assert!((v["x_min"] - 3.0).abs() < 1e-3, "x_min = {}", v["x_min"]);
    assert!((v["y_min"] + 1.0).abs() < 1e-3, "y_min = {}", v["y_min"]);
}
