// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Validates numerical-gradient geometry optimization (examples/geomopt.viv):
//! Newton on the finite-difference derivatives of the RHF/STO-3G energy must relax
//! a diatomic to its equilibrium bond length. Compiled serially; skips if no cc.

use std::collections::HashMap;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn run(src: &str, tag: &str) -> Option<HashMap<String, f64>> {
    let program = viv::build_ir(src, SourceId(0)).expect("should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let dir = std::env::temp_dir().join(format!("viv_geom_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("g.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("g");
    let compile = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("g.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
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

fn template() -> String {
    std::fs::read_to_string(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/geomopt.viv")).expect("geomopt.viv")
}

#[test]
fn h2_relaxes_to_equilibrium_bond_length() {
    // Default molecule is H2. HF/STO-3G equilibrium bond length is ~1.346 bohr.
    let Some(v) = run(&template(), "h2") else { return };
    assert!((v["Re"] - 1.346).abs() < 0.01, "H2 R_e = {} bohr, expected ~1.346", v["Re"]);
    // The optimized energy is the minimum: below the -1.11671 at the usual R=1.4.
    assert!(v["Emin"] < -1.1167 && v["Emin"] > -1.120, "H2 E_min = {}", v["Emin"]);
}

#[test]
#[ignore = "slow (~20s): several full water SCFs per Newton step; run with `--ignored`"]
fn water_relaxes_to_equilibrium_geometry() {
    // Polyatomic: optimize water in 2 symmetry-reduced internal DOF. HF/STO-3G
    // water equilibrium is r_OH ~ 0.989 A (~1.87 bohr), angle ~ 100 deg.
    let src = std::fs::read_to_string(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/geomopt_water.viv")).expect("geomopt_water.viv");
    let Some(v) = run(&src, "water") else { return };
    assert!((v["r_oh"] - 1.870).abs() < 0.02, "r_OH = {} bohr, expected ~1.87", v["r_oh"]);
    let angle_deg = v["cos_hoh"].acos().to_degrees();
    assert!((angle_deg - 100.0).abs() < 3.0, "H-O-H angle = {angle_deg} deg, expected ~100");
    // The optimized energy is below the (stretched) Crawford geometry's -74.9421.
    assert!(v["e_opt"] < -74.96, "E_opt = {}", v["e_opt"]);
}

#[test]
fn hf_molecule_relaxes_to_equilibrium() {
    // Switch the diatomic to HF (F + H): R_e ~ 1.81 bohr (~0.956 A) in STO-3G.
    let src = template()
        .replacen("atom_Z[0] = 1.0", "atom_Z[0] = 9.0", 1)
        .replacen("n_basis = 2", "n_basis = 6", 1)
        .replacen("n_occ = 1", "n_occ = 5", 1)
        .replacen("R = 1.20", "R = 1.70", 1);
    let Some(v) = run(&src, "hf") else { return };
    assert!((v["Re"] - 1.806).abs() < 0.05, "HF R_e = {} bohr, expected ~1.81", v["Re"]);
    assert!(v["Emin"] < -98.57, "HF E_min = {}", v["Emin"]);
}
