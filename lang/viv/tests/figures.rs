// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The figure-record corpus.
//!
//! `tests/figures/` holds record streams (`NAME.out`) and the chart each one
//! must parse to (`NAME.json`). The parsers are tested against them elsewhere
//! (www/scripts/figuretest.mjs, tools/jupyter/tests/test_figures_corpus.py).
//! Here, the engine half: every probe program `NAME.viv` must print exactly
//! `NAME.out`, in the interpreter and, when a C compiler is present, as
//! Viv -> C -> native. (tools/three-engines.sh lang/viv/tests/figures adds the
//! direct wasm backend.)

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/figures")
}

fn probes() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(corpus())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "viv"))
        .collect();
    v.sort();
    v
}

/// The program's own output: the interpreter also prints its call graph and trace.
fn own(out: &[u8]) -> String {
    String::from_utf8_lossy(out)
        .lines()
        .filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__"))
        .map(|l| format!("{l}\n"))
        .collect()
}

fn expected(probe: &Path) -> String {
    let want = probe.with_extension("out");
    std::fs::read_to_string(&want).unwrap_or_else(|_| panic!("{} has no {}", probe.display(), want.display()))
}

#[test]
fn every_probe_has_a_parse() {
    for p in probes() {
        assert!(p.with_extension("json").exists(), "{} has no expected parse (.json)", p.display());
    }
    for e in std::fs::read_dir(corpus()).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|e| e == "out") {
            assert!(p.with_extension("json").exists(), "{} has no expected parse (.json)", p.display());
        }
    }
}

#[test]
fn the_interpreter_prints_the_corpus() {
    let viv = env!("CARGO_BIN_EXE_viv");
    for p in probes() {
        let r = Command::new(viv).arg("run").arg(&p).output().unwrap();
        assert!(r.status.success(), "{}: {}", p.display(), String::from_utf8_lossy(&r.stderr));
        assert_eq!(own(&r.stdout), expected(&p), "{} (interpreter)", p.display());
    }
}

#[test]
fn compiled_c_prints_the_same_bytes() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let viv = env!("CARGO_BIN_EXE_viv");
    let dir = std::env::temp_dir().join(format!("viv_figures_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for p in probes() {
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let c = dir.join(format!("{stem}.c"));
        let exe = dir.join(&stem);
        let r = Command::new(viv).arg(&p).arg("-o").arg(&c).output().unwrap();
        assert!(r.status.success(), "{}: {}", p.display(), String::from_utf8_lossy(&r.stderr));
        let r = Command::new(&cc).arg("-O2").arg("-o").arg(&exe).arg(&c).arg("-lm").output().unwrap();
        assert!(r.status.success(), "{}: {}", p.display(), String::from_utf8_lossy(&r.stderr));
        let r = Command::new(&exe).output().unwrap();
        assert_eq!(own(&r.stdout), expected(&p), "{} (compiled C)", p.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
