// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Where a diagnostic is shown.
//!
//! A program is parsed as one text: its `#! use` modules spliced in, or the
//! Viv that py2viv translated a Python file into. Every span points into that
//! text. The driver records where each line of it came from (knotkit's
//! composite and derived sources), and every renderer follows that back, so:
//!
//! - an error inside a `#! use`d module is shown at the module's own file and
//!   line, in `viv run`, `viv check`, the C build, the wasm build, the C
//!   program's run-time errors, and `--json`;
//! - a `.py` file (or one declaring `#! lang python`) goes through py2viv, and
//!   every error is shown at the Python file, line and column: a construct
//!   py2viv refuses, a type error the compiler finds in the translation, and
//!   an index out of bounds at run time. The interpreter and the C program
//!   print the same location; the direct wasm backend reports the message
//!   without a location, as it does for every run-time error.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn have_cc() -> bool {
    Command::new(cc()).arg("--version").output().is_ok()
}

fn have_python() -> bool {
    Command::new("python3").arg("--version").output().map_or(false, |o| o.status.success())
}

/// A scratch directory that is a repository root (it holds a Cargo.toml), so
/// its modules are named by their paths under it, as in this repository.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_origins_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    dir
}

#[derive(Debug, PartialEq)]
struct Ran {
    code: Option<i32>,
    out: String,
    err: String,
}

fn ran(o: std::process::Output) -> Ran {
    Ran {
        code: o.status.code(),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// `viv <args>` in `dir`, with py2viv from this repository.
fn viv(dir: &Path, args: &[&str]) -> Ran {
    ran(Command::new(viv_bin())
        .current_dir(dir)
        .env("VIV_PY2VIV", root().join("tools/py2viv/py2viv.py"))
        .args(args)
        .output()
        .unwrap())
}

/// The C program built from `file` and run, or None without a C compiler.
fn native(dir: &Path, file: &str) -> Option<Ran> {
    if !have_cc() {
        return None;
    }
    let c = viv(dir, &[file, "-o", "p.c"]);
    assert_eq!(c.code, Some(0), "viv refused {file}:\n{}", c.err);
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    Some(ran(Command::new(dir.join("p")).output().unwrap()))
}

/// The direct wasm module built from `file` and run in node, or None when
/// there is no node or no viv.wasm with the runtime.
fn direct(dir: &Path, file: &str) -> Option<Ran> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    let c = viv(dir, &[file, "-o", "p.wasm"]);
    assert_eq!(c.code, Some(0), "the wasm backend refused {file}:\n{}", c.err);
    Some(ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap()))
}

/// Every way a program is compiled refuses it, and each says `want`: `viv
/// run`, `viv check`, the C build and the wasm build.
fn refused_everywhere(dir: &Path, file: &str, want: &[&str]) {
    for args in [vec!["run", file], vec!["check", file], vec![file, "-o", "p.c"], vec![file, "-o", "p.wasm"]] {
        let r = viv(dir, &args);
        // viv run and viv check exit 3 for a program that does not compile;
        // the builds exit 1
        let code = if matches!(args[0], "run" | "check") { 3 } else { 1 };
        assert_eq!(r.code, Some(code), "`viv {}` should fail:\n{}", args.join(" "), r.err);
        for w in want {
            assert!(r.err.contains(w), "`viv {}` does not say {w:?}:\n{}", args.join(" "), r.err);
        }
    }
}

// ---------------------------------------------------------------- modules

#[test]
fn a_compile_error_in_a_module_is_shown_in_the_module() {
    let dir = scratch("modcompile");
    std::fs::write(dir.join("lib/m.viv"), "# a module\ndef twice(x) {\n    return 2 * y\n}\n").unwrap();
    std::fs::write(dir.join("p.viv"), "from m import twice\nshow twice(21)\n").unwrap();
    refused_everywhere(&dir, "p.viv", &["  --> lib/m.viv:3:16\n", "   3 |     return 2 * y\n"]);
    // as data too
    // the envelope is on stdout
    let j = viv(&dir, &["check", "--json", "p.viv"]);
    assert!(j.out.contains(r#""file":"lib/m.viv","line":3,"col":16"#), "{}", j.out);
    assert!(!j.out.contains(r#""via""#), "a module is not generated code:\n{}", j.out);
}

#[test]
fn a_run_time_error_in_a_module_is_shown_in_the_module_by_both_engines() {
    let dir = scratch("modrun");
    std::fs::write(dir.join("lib/m.viv"), "def at(a, i) {\n    return a[i]\n}\n").unwrap();
    std::fs::write(dir.join("p.viv"), "x = 1\nfrom m import at\na = zeros(3)\nshow at(a, 1)\nshow at(a, 5)\n").unwrap();
    let i = viv(&dir, &["run", "-q", "p.viv"]);
    assert_eq!(i.code, Some(4));
    assert!(i.err.contains("error: index 5 out of bounds for dimension 0 (size 3)\n  --> lib/m.viv:2:12\n   2 |     return a[i]\n"), "{}", i.err);
    if let Some(n) = native(&dir, "p.viv") {
        assert_eq!(i, n, "the engines disagree");
    }
    // a line of the program itself, after the module, keeps its own number
    std::fs::write(dir.join("p.viv"), "x = 1\nfrom m import at\na = zeros(3)\nshow a[7]\n").unwrap();
    let i = viv(&dir, &["run", "-q", "p.viv"]);
    assert!(i.err.contains("  --> p.viv:4:6\n   4 | show a[7]\n"), "{}", i.err);
}

#[test]
fn a_program_with_no_module_reads_as_before() {
    let dir = scratch("plain");
    std::fs::write(dir.join("p.viv"), "a = zeros(2)\nshow a[2]\n").unwrap();
    let i = viv(&dir, &["run", "-q", "p.viv"]);
    assert!(i.err.ends_with("  --> p.viv:2:6\n   2 | show a[2]\n            ^^^^ here\n"), "{}", i.err);
}

/// B250 (bugs/ledger.jsonl): a `--use` file is read into the text before the
/// program, and an error in it was shown at a line of that whole text under
/// the program's name (`report.viv:1231` for dft/pwdft_it.viv:253). It is
/// shown in the `--use` file, and in the module that file imports, by both
/// engines, and the program's own lines keep their numbers.
#[test]
fn an_error_in_a_use_file_is_shown_in_that_file_by_both_engines() {
    let dir = scratch("usefile");
    std::fs::write(dir.join("lib/m.viv"), "def at(a, i) {\n    return a[i]\n}\n").unwrap();
    std::fs::write(dir.join("deck.viv"), "# the deck\nN = 3.0\n").unwrap();
    std::fs::write(dir.join("engine.viv"), "# the engine\nfrom m import at\na = zeros(N)\nshow at(a, 1)\nshow a[N + 2.0]\n").unwrap();
    std::fs::write(dir.join("p.viv"), "show 1.0\n").unwrap();
    let args = ["--use", "deck.viv", "--use", "engine.viv", "p.viv"];
    let run = |extra: &[&str]| viv(&dir, &[&["run", "-q"][..], &args[..], extra].concat());
    let i = run(&[]);
    assert_eq!(i.code, Some(4));
    assert!(i.err.contains("error: index 5 out of bounds for dimension 0 (size 3)\n  --> engine.viv:5:6\n   5 | show a[N + 2.0]\n"), "{}", i.err);
    let build = |dir: &Path| -> Option<Ran> {
        if !have_cc() {
            return None;
        }
        let c = viv(dir, &[&args[..], &["-o", "p.c"]].concat());
        assert_eq!(c.code, Some(0), "{}", c.err);
        let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
        assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
        Some(ran(Command::new(dir.join("p")).output().unwrap()))
    };
    if let Some(n) = build(&dir) {
        assert_eq!(i, n, "the engines disagree");
    }
    // in the module the `--use` file imports
    std::fs::write(dir.join("engine.viv"), "# the engine\nfrom m import at\na = zeros(N)\nshow at(a, 1)\nshow at(a, 7)\n").unwrap();
    let i = run(&[]);
    assert!(i.err.contains("  --> lib/m.viv:2:12\n   2 |     return a[i]\n"), "{}", i.err);
    if let Some(n) = build(&dir) {
        assert_eq!(i, n, "the engines disagree");
    }
    // a compile error in the deck, and a line of the program after the preludes
    std::fs::write(dir.join("deck.viv"), "# the deck\nN = 3.0 +\n").unwrap();
    let c = viv(&dir, &["check", "--use", "deck.viv", "p.viv"]);
    assert!(c.err.contains("--> deck.viv:2:"), "{}", c.err);
    std::fs::write(dir.join("deck.viv"), "# the deck\nN = 3.0\n").unwrap();
    std::fs::write(dir.join("engine.viv"), "# the engine\nfrom m import at\na = zeros(N)\n").unwrap();
    std::fs::write(dir.join("p.viv"), "show 1.0\nshow a[9]\n").unwrap();
    let i = run(&[]);
    assert!(i.err.contains("  --> p.viv:2:6\n   2 | show a[9]\n"), "{}", i.err);
}

// ---------------------------------------------------------------- Python

const REFUSED: &str = "import numpy as np\n\ntry:\n    x = 1.0\nexcept ValueError:\n    x = 2.0\nprint(x)\n";
const TYPE_ERROR: &str = "import numpy as np\n\nz = 1.0 + 2.0j\na = np.zeros(3)\na[0] = z\nprint(a[0])\n";
const OUT_OF_BOUNDS: &str = "import numpy as np\n\na = np.zeros(3)\nfor i in range(3):\n    a[i] = i * 2.0\nprint(a[1])\nx = a[5]\nprint(x)\n";

#[test]
fn a_construct_py2viv_refuses_is_shown_in_the_python_file() {
    if !have_python() {
        return;
    }
    let dir = scratch("pyrefuse");
    std::fs::write(dir.join("p.py"), REFUSED).unwrap();
    refused_everywhere(
        &dir,
        "p.py",
        &["error: not numeric Python: `try`/`except`\n  --> p.py:3:1\n   3 | try:\n", "  note: Check the condition"],
    );
    let j = viv(&dir, &["run", "--json", "p.py"]);
    assert!(j.out.contains(r#""message":"not numeric Python: `try`/`except`","file":"p.py","line":3,"col":1"#), "{}", j.out);
}

#[test]
fn a_type_error_in_the_translation_is_shown_in_the_python_file() {
    if !have_python() {
        return;
    }
    let dir = scratch("pytype");
    std::fs::write(dir.join("p.py"), TYPE_ERROR).unwrap();
    refused_everywhere(
        &dir,
        "p.py",
        &["error: cannot store a complex number in a real array\n  --> p.py:5:1\n   5 | a[0] = z\n", "  note: via py2viv: p.py.viv:"],
    );
    let j = viv(&dir, &["check", "--json", "p.py"]);
    assert!(j.out.contains(r#""file":"p.py","line":5,"col":1"#), "{}", j.out);
    assert!(j.out.contains(r#""via":["via py2viv: p.py.viv:"#), "{}", j.out);
}

#[test]
fn an_index_out_of_bounds_at_run_time_is_shown_in_the_python_file() {
    if !have_python() {
        return;
    }
    let dir = scratch("pyoob");
    std::fs::write(dir.join("p.py"), OUT_OF_BOUNDS).unwrap();
    let located = "error: index 5 out of bounds for dimension 0 (size 3)\n  --> p.py:7:1\n   7 | x = a[5]\n       ^^^^^^^^ here\n  note: via py2viv: p.py.viv:";
    let i = viv(&dir, &["run", "-q", "p.py"]);
    assert_eq!((i.code, i.out.as_str()), (Some(4), "2.0\n"), "{}", i.err);
    assert!(i.err.starts_with(located), "{}", i.err);
    assert_eq!(viv(&dir, &["check", "p.py"]).code, Some(0), "the program checks; it fails only when it runs");
    if let Some(n) = native(&dir, "p.py") {
        assert_eq!(i, n, "the interpreter and the C program disagree");
    }
    // the wasm module stops with the same words, and no location (as for every program)
    if let Some(d) = direct(&dir, "p.py") {
        assert_eq!((d.code, d.out.as_str()), (Some(4), "2.0\n"), "{}", d.err);
        assert_eq!(d.err, "error: index 5 out of bounds for dimension 0 (size 3)\n");
    }
}

#[test]
fn a_file_declaring_lang_python_is_python_whatever_its_name() {
    if !have_python() {
        return;
    }
    let dir = scratch("pylang");
    std::fs::write(dir.join("p.viv"), format!("#! lang python\n{OUT_OF_BOUNDS}")).unwrap();
    let i = viv(&dir, &["run", "-q", "p.viv"]);
    assert!(i.err.contains("  --> p.viv:8:1\n   8 | x = a[5]\n"), "{}", i.err);
}

#[test]
fn a_python_program_that_runs_prints_what_cpython_prints() {
    if !have_python() {
        return;
    }
    let dir = scratch("pyok");
    std::fs::write(dir.join("p.py"), "import numpy as np\n\na = np.arange(4.0)\nprint(a.sum(), a[3] / 2)\n").unwrap();
    let i = viv(&dir, &["run", "-q", "p.py"]);
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), "6.0 1.5\n", ""));
    if let Some(n) = native(&dir, "p.py") {
        assert_eq!(n.out, i.out);
    }
    // the translation is a program like any other: it has content addresses
    let h = viv(&dir, &["hash", "p.py"]);
    assert_eq!(h.code, Some(0), "{}", h.err);
}

#[test]
fn a_python_module_used_by_a_viv_program_is_shown_in_python() {
    if !have_python() {
        return;
    }
    let dir = scratch("pymod");
    std::fs::write(dir.join("lib/k.py"), "#! lang python\nimport numpy as np\n\ndef at(a, i):\n    return a[int(i)]\n").unwrap();
    std::fs::write(dir.join("p.viv"), "from k import at\na = zeros(3)\nshow at(a, 1)\nshow at(a, 4)\n").unwrap();
    // a module named lib/k is lib/k.viv; a Python module is found the same way,
    // so it is written as lib/k.viv here, declaring its language
    std::fs::rename(dir.join("lib/k.py"), dir.join("lib/k.viv")).unwrap();
    let i = viv(&dir, &["run", "-q", "p.viv"]);
    assert_eq!(i.code, Some(4), "{}", i.err);
    assert!(i.err.contains("  --> lib/k.viv:5:5\n   5 |     return a[int(i)]\n"), "{}", i.err);
    if let Some(n) = native(&dir, "p.viv") {
        assert_eq!(i, n, "the engines disagree");
    }
}
