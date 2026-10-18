// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Lists the feature catalog's examples (lang/viv/features/examples) for
//! `crate::features`, so the compiler carries every example and its recorded
//! output, and a file added or removed there changes the catalog on the next
//! build without a list kept by hand.

use std::path::Path;

fn main() {
    sqlite();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("features");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut rows = Vec::new();
    for (sub, interp) in [("examples", false), ("examples/interp", true)] {
        let d = dir.join(sub);
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut names: Vec<String> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".viv"))
            .map(|n| n.trim_end_matches(".viv").to_string())
            .collect();
        names.sort();
        for n in names {
            let file = |ext: &str| {
                let p = d.join(format!("{n}.{ext}"));
                if p.exists() { format!("Some(include_str!({:?}))", p.display().to_string()) } else { "None".to_string() }
            };
            rows.push(format!(
                "    ExampleFile {{ name: {n:?}, interp_only: {interp}, viv: include_str!({:?}), out: {}, cmd: {}, intent: {}, debug: {}, memory: {}, arch: {} }},\n",
                d.join(format!("{n}.viv")).display().to_string(),
                file("out"),
                file("cmd"),
                file("intent"),
                file("debug"),
                file("memory"),
                file("arch"),
            ));
        }
    }
    // the hand-written areas, and the generated tables beside them
    let mut texts = Vec::new();
    for name in ["syntax", "types", "arrays", "autodiff", "correctness", "lenses", "tools", "performance", "libraries", "builtins", "modules"] {
        let p = dir.join(format!("{name}.txt"));
        let text = if p.exists() { format!("include_str!({:?})", p.display().to_string()) } else { "\"\"".to_string() };
        texts.push(format!("    ({name:?}, {text}),\n"));
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("feature_examples.rs");
    std::fs::write(
        out,
        format!("pub const EXAMPLE_FILES: &[ExampleFile] = &[\n{}];\n\npub const CATALOG_FILES: &[(&str, &str)] = &[\n{}];\n", rows.concat(), texts.concat()),
    )
    .unwrap();
}

/// SQLite (third_party/sqlite), compiled once
/// into a static library the `viv` binary links, for `viv db` and the
/// interpreter's `db_*` builtins. Not for the browser module (wasm32): there
/// the builtins say they are not available. The system C compiler is
/// called directly, so no build dependency is added.
fn sqlite() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.starts_with("wasm32") {
        return;
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/sqlite/sqlite3.c");
    println!("cargo:rerun-if-changed={}", src.display());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("sqlite3.o");
    let lib = out.join("libvivsqlite3.a");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let st = std::process::Command::new(&cc)
        .args(["-c", "-O2", "-fPIC", "-w", "-DSQLITE_OMIT_LOAD_EXTENSION", "-DSQLITE_DEFAULT_FOREIGN_KEYS=1"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("build.rs: cannot run the C compiler for SQLite");
    assert!(st.success(), "build.rs: compiling third_party/sqlite/sqlite3.c failed");
    let _ = std::fs::remove_file(&lib);
    let st = std::process::Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().expect("build.rs: cannot run ar");
    assert!(st.success(), "build.rs: ar failed for SQLite");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vivsqlite3");
}
