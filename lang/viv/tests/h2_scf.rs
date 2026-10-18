// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The north-star test: compile and run the real Hartree-Fock SCF for H2 in
//! STO-3G (from viv-lang-ref) through kit-viv, and check the total energy
//! against the Szabo & Ostlund reference value, -1.1167 hartree.
//!
//! Skips cleanly if the reference source or a C compiler is unavailable.

use std::path::PathBuf;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

#[test]
fn h2_sto3g_scf_energy() {
    let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/h2_sto3g.viv");
    let src = std::fs::read_to_string(&src_path).expect("examples/h2_sto3g.viv should exist");

    // Front end + backend: real quantum-chemistry source -> C.
    let program = viv::build_ir(&src, SourceId(0)).expect("H2 SCF should lower");
    let c = CBackend
        .emit(&program, &MemoryIo::new())
        .expect("H2 SCF should emit C")
        .artifact
        .as_c()
        .expect("C")
        .to_string();

    let dir = std::env::temp_dir().join(format!("viv_h2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c_path = dir.join("h2.c");
    let bin = dir.join("h2");
    std::fs::write(&c_path, &c).unwrap();

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let compile = match Command::new(&cc).args(["-std=c99", "-O2"]).arg(&c_path).arg("-lm").arg("-o").arg(&bin).output() {
        Ok(o) => o,
        Err(_) => {
            eprintln!("skipping h2_sto3g_scf_energy: no C compiler `{cc}`");
            return;
        }
    };
    assert!(compile.status.success(), "H2 C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));

    let run = Command::new(&bin).output().unwrap();
    let out = String::from_utf8_lossy(&run.stdout);
    let _ = std::fs::remove_dir_all(&dir);

    // The program's own embedded `test` must pass, and the converged energy
    // must round to -1.11671 (matches the reference exactly). Asserting on the
    // NAMED line is stronger than the old `assert_near PASS`: it proves which
    // test passed, not merely that some assertion somewhere did.
    assert!(
        out.contains("PASS H2/STO-3G total energy near textbook -1.117"),
        "embedded SCF test did not pass. output:\n{out}"
    );
    assert!(out.contains("E_total: -1.11671"), "wrong total energy. output:\n{out}");
    assert!(run.status.success(), "H2 binary exited non-zero");
}
