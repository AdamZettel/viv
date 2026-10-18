// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Validates the 2D Poisson FEM (examples/fem_poisson.viv): linear (P1)
//! triangular elements on the unit square with a manufactured solution. The
//! check is the *convergence rate* -- the RMS nodal error must fall by ~4x when
//! the mesh is halved (O(h^2), the theoretical P1 rate) -- plus the center value
//! approaching the exact 0.0625. Compiled serially; skips if no C compiler.

use std::collections::HashMap;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

/// Run a FEM example specialized to a given mesh resolution N.
fn run_fem_file(file: &str, n: u32) -> Option<HashMap<String, f64>> {
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples").join(file);
    let template = std::fs::read_to_string(&src_path).unwrap_or_else(|_| panic!("{file} should exist"));
    let src = template.replacen("N = 8", &format!("N = {n}"), 1);
    let program = viv::build_ir(&src, SourceId(0)).expect("should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();

    // One directory per call: both tests run fem_poisson.viv at N = 8 at the
    // same time, and with a shared directory one removed it while the other
    // was still compiling into it (ENOENT, now and then).
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("viv_fem_{}_{}_{}_{call}", std::process::id(), file.replace('.', "_"), n));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("f.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("f");
    let compile = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("f.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
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

fn run_fem(n: u32) -> Option<HashMap<String, f64>> {
    run_fem_file("fem_poisson.viv", n)
}

#[test]
fn poisson_p1_converges_at_second_order() {
    let Some(coarse) = run_fem(8) else { return };
    let fine = run_fem(16).expect("N=16 run");

    let rms_c = coarse["rms"];
    let rms_f = fine["rms"];
    // P1 elements: L2/nodal error is O(h^2), so halving h cuts the error ~4x.
    let ratio = rms_c / rms_f;
    assert!((3.0..=4.5).contains(&ratio), "convergence ratio {ratio} (rms {rms_c} -> {rms_f}) not ~4x (O(h^2))");
    // Solution at the domain center approaches the exact value u(0.5,0.5)=0.0625.
    assert!((fine["u_center"] - 0.0625).abs() < 5e-3, "u_center = {} (exact 0.0625)", fine["u_center"]);
    assert!(rms_f < 5e-4, "fine-mesh RMS error {rms_f} too large");
}

#[test]
fn records_refactor_matches_struct_of_arrays() {
    // The records-based FEM (Mesh/Tri records) must produce the identical result
    // to the loose-arrays version -- records are zero-cost bundling, not a change
    // in semantics.
    let Some(plain) = run_fem_file("fem_poisson.viv", 8) else { return };
    let recs = run_fem_file("fem_poisson_records.viv", 8).expect("records FEM run");
    assert!((plain["rms"] - recs["rms"]).abs() < 1e-12, "rms differs: {} vs {}", plain["rms"], recs["rms"]);
    assert!((plain["u_center"] - recs["u_center"]).abs() < 1e-12, "u_center differs");
}
