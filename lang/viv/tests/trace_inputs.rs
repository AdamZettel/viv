// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! VIV_TRACE_INPUTS (lang/viv/src/trace.rs): the files a run reads, which the
//! gate's cache keys a verdict on (tools/gate/cache.py). A file left out of
//! the trace would let a stale verdict through, so the trace must name the
//! program, each module an import resolves to, each place an import looked
//! and found nothing, the lens files, and each file read_text read.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("viv_trace_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.canonicalize().unwrap()
}

/// The trace lines of `viv run <prog>` (cwd `dir`), sorted and deduplicated.
fn trace(dir: &Path, prog: &str) -> Vec<String> {
    let t = dir.join("trace.txt");
    let _ = std::fs::remove_file(&t);
    let out = Command::new(viv()).arg("run").arg(prog).current_dir(dir).env("VIV_TRACE_INPUTS", &t).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut lines: Vec<String> = std::fs::read_to_string(&t).unwrap_or_default().lines().map(str::to_string).collect();
    lines.sort();
    lines.dedup();
    lines
}

#[test]
fn the_trace_names_the_program_its_modules_and_what_it_reads() {
    let d = scratch("modules");
    std::fs::write(d.join("helper.viv"), "def twice(x) {\n    return 2 * x\n}\n").unwrap();
    std::fs::write(d.join("data.txt"), "21\n").unwrap();
    std::fs::write(d.join("main.viv"), "import helper\ns = read_text(\"data.txt\")\nprint(helper.twice(len(s)))\n").unwrap();
    let lines = trace(&d, "main.viv");
    let has = |kind: &str, file: &str, flag: Option<&str>| {
        let p = d.join(file).display().to_string();
        lines.iter().any(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            f[0] == kind && f[1] == p && flag.map_or(true, |x| f.get(2) == Some(&x))
        })
    };
    assert!(has("f", "main.viv", Some("1")), "the program: {lines:#?}");
    assert!(has("f", "helper.viv", Some("1")), "the module: {lines:#?}");
    assert!(has("f", "data.txt", Some("1")), "the file read_text read: {lines:#?}");
    assert!(has("f", "helper.memory", Some("0")), "an absent lens file is an input too: {lines:#?}");
    assert!(has("io", "main.viv", None), "a program naming read_text is marked: {lines:#?}");
}

#[test]
fn without_the_variable_nothing_is_written() {
    let d = scratch("quiet");
    std::fs::write(d.join("p.viv"), "print(1)\n").unwrap();
    let out = Command::new(viv()).arg("run").arg("p.viv").current_dir(&d).env_remove("VIV_TRACE_INPUTS").output().unwrap();
    assert!(out.status.success());
    assert!(!d.join("trace.txt").exists());
}
