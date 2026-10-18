// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! A small molecule database: for each molecule, the molecule-driven STO-3G HF
//! driver (examples/scf_sto3g.viv) is specialized to that geometry, compiled,
//! run, and the total energy compared against a reference value.
//!
//! Reference confidence:
//!   * FIRM  — H2 (Szabo & Ostlund) and H2O (Crawford projects), matched tightly.
//!   * LIT   — literature HF/STO-3G total energies (geometry/convention dependent);
//!             checked loosely, mainly to confirm the element basis data.
//!
//! Run the table with:  cargo test -p viv --test molecule_db -- --nocapture

use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

struct Atom {
    z: f64,
    x: f64,
    y: f64,
    z_coord: f64,
}

struct Mol {
    name: &'static str,
    charge: f64,
    atoms: Vec<Atom>,
    reference: f64,
    /// Allowed |E - reference| in mEh. Tight (<=1) for firm references (H2, H2O,
    /// the well-known atom energies); looser where the reference is approximate.
    tol: f64,
    source: &'static str,
    /// Reference MP2 correlation energy (Eh), where a firm one is known.
    mp2_ref: Option<f64>,
}

fn a(z: f64, x: f64, y: f64, zc: f64) -> Atom {
    Atom { z, x, y, z_coord: zc }
}

/// Specialize the driver template's MOL_BEGIN..MOL_END block for one molecule.
fn specialize(template: &str, m: &Mol) -> String {
    let mut block = String::new();
    block.push_str(&format!("n_atoms = {}\n", m.atoms.len()));
    block.push_str(&format!("mol_charge = {:?}\n", m.charge));
    block.push_str("atom_Z = zeros(n_atoms)\natom_x = zeros(n_atoms)\natom_y = zeros(n_atoms)\natom_z = zeros(n_atoms)\n");
    for (i, at) in m.atoms.iter().enumerate() {
        block.push_str(&format!("atom_Z[{i}] = {:?}\n", at.z));
        block.push_str(&format!("atom_x[{i}] = {:?}\n", at.x));
        block.push_str(&format!("atom_y[{i}] = {:?}\n", at.y));
        block.push_str(&format!("atom_z[{i}] = {:?}\n", at.z_coord));
    }
    let begin = "# === MOL_BEGIN ===\n";
    let end = "# === MOL_END ===";
    let b = template.find(begin).expect("MOL_BEGIN marker") + begin.len();
    let e = template.find(end).expect("MOL_END marker");
    format!("{}{}{}", &template[..b], block, &template[e..])
}

/// Returns (E_SCF, MP2 correlation energy).
fn run_energy(source: &str, dir: &std::path::Path, tag: &str) -> Option<(f64, f64)> {
    let program = viv::build_ir(source, SourceId(0)).expect("should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let c_path = dir.join(format!("{tag}.c"));
    let bin = dir.join(tag);
    std::fs::write(&c_path, &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let compile = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(&c_path).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(compile.status.success(), "{tag} C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new(&bin).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let get = |key: &str| stdout.lines().find_map(|l| l.split_once(':').filter(|(k, _)| k.trim() == key).and_then(|(_, v)| v.trim().parse::<f64>().ok()));
    Some((get("E_total")?, get("emp2").unwrap_or(f64::NAN)))
}

#[test]
fn molecule_database_vs_references() {
    let template = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/scf_sto3g.viv"),
    )
    .expect("scf_sto3g.viv");

    // Geometries in bohr.
    let s = 2.05 / 3.0_f64.sqrt(); // CH4 tetrahedral arm
    let db = vec![
        // Closed-shell atoms: unambiguous geometry, well-known STO-3G energies
        // -- a clean check of the per-element basis data (He, Be, Ne span H-Ne).
        Mol { name: "He",   charge: 0.0, atoms: vec![a(2.0,0.,0.,0.)], reference: -2.807784, tol: 1.0, source: "STO-3G atom", mp2_ref: None },
        Mol { name: "Be",   charge: 0.0, atoms: vec![a(4.0,0.,0.,0.)], reference: -14.351880, tol: 1.0, source: "STO-3G atom", mp2_ref: None },
        Mol { name: "Ne",   charge: 0.0, atoms: vec![a(10.0,0.,0.,0.)], reference: -126.604522, tol: 1.0, source: "STO-3G atom", mp2_ref: None },
        Mol { name: "H2",   charge: 0.0, atoms: vec![a(1.0,0.,0.,0.), a(1.0,0.,0.,1.4)], reference: -1.1167, tol: 1.0, source: "Szabo & Ostlund", mp2_ref: None },
        Mol { name: "LiH",  charge: 0.0, atoms: vec![a(3.0,0.,0.,0.), a(1.0,0.,0.,3.015)], reference: -7.8633, tol: 5.0, source: "lit. HF/STO-3G", mp2_ref: None },
        Mol { name: "HeH+", charge: 1.0, atoms: vec![a(2.0,0.,0.,0.), a(1.0,0.,0.,1.4632)], reference: -2.8418, tol: 5.0, source: "lit. HF/STO-3G", mp2_ref: None },
        Mol { name: "HF",   charge: 0.0, atoms: vec![a(9.0,0.,0.,0.), a(1.0,0.,0.,1.7328)], reference: -98.5708, tol: 5.0, source: "lit. HF/STO-3G", mp2_ref: None },
        // Water carries a firm MP2 reference (Crawford projects, project #4).
        Mol { name: "H2O",  charge: 0.0, atoms: vec![a(8.0,0.,-0.143225816552,0.), a(1.0,1.638036840407,1.136548822547,0.), a(1.0,-1.638036840407,1.136548822547,0.)], reference: -74.9420787, tol: 1.0, source: "Crawford projects", mp2_ref: Some(-0.049149636120) },
        Mol { name: "N2",   charge: 0.0, atoms: vec![a(7.0,0.,0.,0.), a(7.0,0.,0.,2.074)], reference: -107.4960, tol: 5.0, source: "lit. HF/STO-3G", mp2_ref: None },
        Mol { name: "CO",   charge: 0.0, atoms: vec![a(6.0,0.,0.,0.), a(8.0,0.,0.,2.132)], reference: -111.2238, tol: 5.0, source: "lit. HF/STO-3G", mp2_ref: None },
        Mol { name: "CH4",  charge: 0.0, atoms: vec![a(6.0,0.,0.,0.), a(1.0,s,s,s), a(1.0,s,-s,-s), a(1.0,-s,s,-s), a(1.0,-s,-s,s)], reference: -39.7269, tol: 5.0, source: "lit. HF/STO-3G", mp2_ref: None },
        // Period 3 (3s/3p shell). Mg validates the period-3 machinery; the
        // reference is only approximate, hence the wide tolerance.
        Mol { name: "Mg",   charge: 0.0, atoms: vec![a(12.0,0.,0.,0.)], reference: -197.0, tol: 60.0, source: "STO-3G atom (approx)", mp2_ref: None },
    ];

    let dir = std::env::temp_dir().join(format!("viv_moldb_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Probe the compiler once.
    if run_energy(&specialize(&template, &db[0]), &dir, "probe").is_none() {
        eprintln!("skipping molecule_database: no C compiler");
        return;
    }

    eprintln!("\n {:<6} {:>14} {:>14} {:>10}   {:>13}   {}", "mol", "E_SCF (kit)", "ref (SCF)", "err mEh", "E_corr(MP2)", "source");
    eprintln!(" {}", "-".repeat(82));
    let mut failures = Vec::new();
    for m in &db {
        let (e, emp2) = run_energy(&specialize(&template, m), &dir, m.name).expect("energy");
        let err_meh = (e - m.reference) * 1000.0;
        let mark = if m.tol <= 1.0 { "*" } else { " " };
        eprintln!("{}{:<6} {:>14.5} {:>14.5} {:>10.3}   {:>13.6}   {}", mark, m.name, e, m.reference, err_meh, emp2, m.source);
        if err_meh.abs() > m.tol {
            failures.push(format!("{}: SCF {e} vs {} (err {err_meh:.3} mEh, tol {})", m.name, m.reference, m.tol));
        }
        // MP2 correlation energy, where a firm reference exists (asserted to 1 uEh).
        if let Some(mref) = m.mp2_ref {
            let merr_ueh = (emp2 - mref) * 1.0e6;
            if merr_ueh.abs() > 1.0 {
                failures.push(format!("{}: MP2 corr {emp2} vs {mref} (err {merr_ueh:.3} uEh)", m.name));
            }
        }
    }
    eprintln!(" {}\n (* = firm reference, |err| asserted < 1 mEh; MP2 corr reported for all, asserted for H2O vs Crawford)\n", "-".repeat(82));
    let _ = std::fs::remove_dir_all(&dir);

    assert!(failures.is_empty(), "reference mismatches:\n{}", failures.join("\n"));
}
