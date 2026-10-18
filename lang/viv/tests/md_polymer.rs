// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Validates the molecular-dynamics example by physics: a velocity-Verlet
//! trajectory must conserve total energy (bounded, no drift), and the extended
//! chain must coil (radius of gyration decreases). Compiled serially (the
//! `parallel for` pragma is a no-op without -fopenmp). Skips if no C compiler.

use std::collections::HashMap;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn show_values(stdout: &str) -> HashMap<String, f64> {
    stdout
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            Some((k.trim().to_string(), v.trim().parse::<f64>().ok()?))
        })
        .collect()
}

#[test]
fn md_polymer_conserves_energy_and_coils() {
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/md_polymer.viv");
    let src = std::fs::read_to_string(&src_path).expect("md_polymer.viv should exist");

    let program = viv::build_ir(&src, SourceId(0)).expect("MD should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("MD should emit C").artifact.as_c().expect("C").to_string();

    let dir = std::env::temp_dir().join(format!("viv_md_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("md.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("md");
    let compile = match Command::new(&cc)
        .args(["-std=c99", "-O2", "-Wno-unknown-pragmas"])
        .arg(dir.join("md.c"))
        .arg("-lm")
        .arg("-o")
        .arg(&bin)
        .output()
    {
        Ok(o) => o,
        Err(_) => return,
    };
    assert!(compile.status.success(), "MD C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));

    let out = Command::new(&bin).output().unwrap();
    let vals = show_values(&String::from_utf8_lossy(&out.stdout));
    let _ = std::fs::remove_dir_all(&dir);

    let e_start = vals["e_start"];
    let e_min = vals["e_min"];
    let e_max = vals["e_max"];
    let rg_start = vals["rg_start"];
    let rg_final = vals["rg_final"];

    // Energy conserved: bounded oscillation, no secular drift.
    let drift = (e_max - e_min) / e_start.abs();
    assert!(drift.abs() < 0.05, "energy not conserved: drift = {drift}");
    // Physical result: the extended chain coils up.
    assert!(rg_final < rg_start * 0.8, "chain did not coil: rg {rg_start} -> {rg_final}");
}
