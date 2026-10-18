// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// REUSE-IgnoreStart (the fixtures below carry SPDX tags of their own)
//! `viv export` tracks the license of each module a program imports: the
//! exported C starts with the list, NOTICES holds the texts, a GPL module
//! refuses the export and an MPL one warns (lang/viv/src/licenses.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_licenses_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in files {
        std::fs::write(dir.join(name), text).unwrap();
    }
    dir
}

fn infer(dir: &Path, file: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_viv")).current_dir(dir).env("VIV_ROOT", repo()).args(["export", "--infer", file, "-o", "out"]).output().unwrap()
}

const PROG: &str = "import helper\nimport minpack.enorm\n\ndef twice(x: Num) -> Num {\n    return helper.h(x) + minpack.enorm.enorm(1, [x])\n}\n";

fn helper(license: &str) -> String {
    format!("def h(x: Num) -> Num {{\n    return 2.0 * x\n}}\n\n# SPDX-License-Identifier: {license}\n# SPDX-FileCopyrightText: 2026 A. Fixture\n")
}

#[test]
fn the_exported_c_lists_every_module_and_its_license() {
    let dir = scratch("mit", &[("prog.viv", PROG), ("helper.viv", &helper("MIT"))]);
    let out = infer(&dir, "prog.viv");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(!err.contains("warning:"), "{err}");
    let c = std::fs::read_to_string(dir.join("out/prog.c")).unwrap();
    let top: String = c.lines().take(30).collect::<Vec<_>>().join("\n");
    assert!(top.starts_with("/* SPDX-License-Identifier: "), "{top}");
    assert!(top.contains("helper.viv\n         MIT; 2026 A. Fixture"), "{top}");
    assert!(top.contains("enorm.viv\n         MIT OR Apache-2.0"), "{top}");
    assert!(top.contains("MIT AND SunPro"), "{top}");
    assert!(top.contains("SPDX-FileCopyrightText: 2026 A. Fixture"), "{top}");
    let notices = std::fs::read_to_string(dir.join("out/NOTICES")).unwrap();
    for want in ["Permission is hereby granted", "Apache License", "Developed at SunSoft"] {
        assert!(notices.contains(want), "NOTICES lacks {want}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_gpl_module_refuses_the_export() {
    let dir = scratch("gpl", &[("prog.viv", PROG), ("helper.viv", &helper("GPL-3.0-or-later"))]);
    let out = infer(&dir, "prog.viv");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("GPL-licensed code") && err.contains("helper.viv (GPL-3.0-or-later)"), "{err}");
    assert!(!dir.join("out/prog.c").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_mpl_module_warns_and_names_the_file() {
    let dir = scratch("mpl", &[("prog.viv", PROG), ("helper.viv", &helper("MPL-2.0"))]);
    let out = infer(&dir, "prog.viv");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(err.lines().any(|l| l.starts_with("warning: helper.viv is MPL-2.0") && l.contains("source available")), "{err}");
    let notices = std::fs::read_to_string(dir.join("out/NOTICES")).unwrap();
    assert!(notices.contains("Mozilla Public License Version 2.0"));
    let _ = std::fs::remove_dir_all(&dir);
}
// REUSE-IgnoreEnd
