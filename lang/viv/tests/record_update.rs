// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Record update, `T { f: e, ..base }`: the three
//! engines print the same bytes, the update copies every field it does not
//! write, the base is computed once, and the mistakes are refused, located.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_record_update_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn interp(src: &str) -> String {
    viv::interp::run(src).expect("interprets").lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

fn graph_line(src: &str) -> String {
    viv::interp::run(src).expect("interprets").lines().find(|l| l.starts_with("__graph__")).unwrap_or("").to_string()
}

/// The C build's output, or None without a C compiler.
fn native(dir: &Path, src: &str) -> Option<String> {
    let c = viv::compile_c(src).expect("compiles to C");
    let file = dir.join("p.c");
    std::fs::write(&file, c).unwrap();
    let bin = dir.join("p");
    let built = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into())).arg("-O2").arg(&file).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    Some(String::from_utf8_lossy(&Command::new(&bin).output().unwrap().stdout).into_owned())
}

/// The direct wasm module's output, or None without node and a runtime.
fn direct(dir: &Path, src: &str) -> Option<String> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    let m = dir.join("p.wasm");
    std::fs::write(&m, viv::compile_wasm(src).expect("compiles to wasm")).unwrap();
    let out = Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(&m).env("VIV_WASM", &wasm).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn same_everywhere(tag: &str, src: &str) -> String {
    let want = interp(src);
    let dir = scratch(tag);
    if let Some(out) = native(&dir, src) {
        assert_eq!(out, want, "C");
    }
    if let Some(out) = direct(&dir, src) {
        assert_eq!(out, want, "wasm");
    }
    want
}

const OPTS: &str = r#"
record Inner { k: Num, v: Arr }
record Opts { guess: Num, tol: Num, maxit: Num, w: Arr, inner: Inner }
def defaults() -> Opts {
    return Opts { guess: 0, tol: 1.0e-9, maxit: 100, w: [1.0, 2.0], inner: Inner { k: 4, v: [7.0] } }
}
def run(o: Opts) {
    return o.guess * 1000 + o.maxit + o.w[1] + o.inner.k
}
"#;

#[test]
fn an_update_copies_the_fields_it_does_not_write_in_every_engine() {
    let src = format!(
        "{OPTS}o = defaults()\na = Opts {{ guess: 2, ..o }}\nshow a.guess\nshow a.tol\nshow a.maxit\nshow a.inner.v\n\
         b = Opts {{ maxit: 7, guess: 3, ..defaults() }}\nshow run(b)\nshow run(Opts {{ w: [5.0, 6.0], inner: Inner {{ k: 1, ..o.inner }}, ..defaults() }})\n\
         c = Opts {{ ..o }}\nshow run(c)\nfor k to 3 {{\n    show run(Opts {{ guess: k, ..defaults() }})\n}}\n"
    );
    let out = same_everywhere("copy", &src);
    assert_eq!(out, "2\n1e-09\n100\n[7]\n3013\n107\n106\n106\n1106\n2106\n", "{out}");
}

#[test]
fn the_base_is_computed_once_after_the_new_values() {
    // `tick` prints when it runs: once per update, after the value it is given
    let src = format!(
        "{OPTS}def tick(x) {{\n    print(\"tick\", x, sep=\"\")\n    return x\n}}\ndef base() -> Opts {{\n    print(\"base\")\n    return defaults()\n}}\n\
         a = Opts {{ maxit: tick(2), guess: tick(1), ..base() }}\nshow run(a)\n"
    );
    let out = same_everywhere("once", &src);
    // the new values in the record's declaration order (guess, then maxit), then the base
    assert_eq!(out, "tick1\ntick2\nbase\n1008\n", "{out}");
}

#[test]
fn one_def_per_record_and_set_of_fields_and_none_for_a_name() {
    let src = format!("{OPTS}o = defaults()\na = Opts {{ guess: 1, ..defaults() }}\nb = Opts {{ guess: 2, ..defaults() }}\nc = Opts {{ tol: 2.0, ..defaults() }}\nd = Opts {{ guess: 3, ..o }}\nshow a.guess + b.guess + c.tol + d.guess\n");
    let c = viv::compile_c(&src).unwrap();
    assert!(c.contains("viv__with_Opts__guess("));
    assert!(c.contains("viv__with_Opts__tol("));
    assert_eq!(c.matches("viv_rec_Opts viv__with_Opts__guess(").count(), 2, "a prototype and a definition, once");
    // the call graph shows the caller's calls, not the update's def
    let g = graph_line(&src);
    assert!(g.contains("defaults") && !g.contains("viv__with"), "{g}");
}

#[test]
fn mistakes_are_refused_where_they_are_written() {
    let head = "record Opts { guess: Num, tol: Num }\ndef defaults() -> Opts {\n    return Opts { guess: 0, tol: 1.0 }\n}\n";
    for (body, message) in [
        ("a = Opts { gues: 2, ..defaults() }", "record `Opts` has no field `gues`"),
        ("a = Opts { guess: 2, guess: 3, ..defaults() }", "the field `guess` is given twice"),
        ("a = Opts { ..defaults(), guess: 2 }", "the record to update comes last"),
        ("a = Nope { guess: 2, ..defaults() }", "there is no record `Nope` to update"),
        ("a = Opts { guess: 2, ..sqrt(2.0) }", "the record after `..` should be a Opts but is a number"),
    ] {
        let src = format!("{head}{body}\nshow a.guess\n");
        let err = viv::compile_c(&src).expect_err(body);
        assert!(err.contains(message), "{body}: {err}");
    }
}
