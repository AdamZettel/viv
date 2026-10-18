// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Validates the p-orbital McMurchie-Davidson engine: water (H2O) HF/STO-3G
//! must reproduce the published Crawford reference E(SCF) = -74.9420787 Eh.
//! Compiled serially (the `parallel for` pragma is a no-op without -fopenmp).
//! Skips if no C compiler.

use std::collections::HashMap;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn run_example(name: &str) -> Option<HashMap<String, f64>> {
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples").join(name);
    let src = std::fs::read_to_string(&src_path).unwrap_or_else(|_| panic!("{name} should exist"));
    let program = viv::build_ir(&src, SourceId(0)).expect("should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("should emit C").artifact.as_c().expect("C").to_string();

    // Per-example dir so tests running in parallel don't clobber each other.
    let dir = std::env::temp_dir().join(format!("viv_scf_{}_{}", std::process::id(), name.replace('.', "_")));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let compile = match Command::new(&cc)
        .args(["-std=c99", "-O2", "-Wno-unknown-pragmas"])
        .arg(dir.join("p.c"))
        .arg("-lm")
        .arg("-o")
        .arg(&bin)
        .output()
    {
        Ok(o) => o,
        Err(_) => return None,
    };
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
fn mcmurchie_engine_reproduces_h2() {
    // The general s/p engine must give the same s-only H2 energy.
    let Some(vals) = run_example("scf_mcmurchie.viv") else { return };
    assert!((vals["E_total"] - (-1.11671)).abs() < 1e-4, "H2 via MD engine: {}", vals["E_total"]);
}

#[test]
fn water_hf_sto3g_energy() {
    // p-orbitals: water HF/STO-3G vs the published Crawford reference.
    let Some(vals) = run_example("scf_water.viv") else { return };
    let e = vals["E_total"];
    assert!((e - (-74.9420787)).abs() < 1e-4, "water HF/STO-3G energy = {e}, expected -74.94208");
}

#[test]
#[ignore = "slow (~10s serial); validates d-orbitals: run with `--ignored`"]
fn water_hf_631gd_energy() {
    // d-orbitals (6-31G*): variationally below STO-3G/6-31G and above the HF
    // basis-set limit (~-76.067) at the Crawford geometry.
    let Some(vals) = run_example("scf_water_631gd.viv") else { return };
    let e = vals["E_total"];
    assert!(e < -75.99 && e > -76.06, "water HF/6-31G* energy = {e}, expected ~-76.019");
}
