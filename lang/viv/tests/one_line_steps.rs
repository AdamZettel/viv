// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The one-line step, `step attn_residual: X1 = X + Y`: the braced step of one statement, to every tool.
//!
//! What is checked here:
//! - a program in one-line steps and the same program with each written
//!   `step name { ... }` on the same lines give the same syntax tree, body
//!   hashes, step events, story (with a debug lens), explorer outline,
//!   intent and debug views, and output in all three engines;
//! - against the braced form on several lines, the hashes, the stamps
//!   `viv lens init` writes, the pseudocode and the story are the same;
//! - the grammar: a statement that opens a block (`for`, `if` with `else` on
//!   its own line) runs to where the block closes, a one-line step may hold a
//!   braced or a one-line step, `;` ends it, a `}` closing the block around it
//!   ends it, and `step` stays an ordinary name elsewhere (a variable, a record
//!   field, a phrase's word);
//! - the refusals: nothing after the `:`, or the statement on the next line.
//!
//! The C and wasm runs skip when there is no C compiler, node, or a viv.wasm
//! at www/public/viv/viv.wasm (VIV_WASM names another).

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_one_line_steps_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `viv <args> p.viv` in `dir`: (exit 0, stdout, stderr), with the file's
/// name taken out so two programs compare.
fn viv(dir: &Path, args: &[&str]) -> (bool, String, String) {
    let o = Command::new(viv_bin()).current_dir(dir).args(args).output().unwrap();
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let clean = |b: &[u8]| String::from_utf8_lossy(b).replace(&name, "DIR").replace("p.viv", "P");
    (o.status.success(), clean(&o.stdout), clean(&o.stderr))
}

/// The program's directory with `p.viv` and, when given, `p.debug` and `p.intent`.
fn program(tag: &str, src: &str, debug: Option<&str>, intent: Option<&str>) -> PathBuf {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    if let Some(d) = debug {
        std::fs::write(dir.join("p.debug"), d).unwrap();
    }
    if let Some(i) = intent {
        std::fs::write(dir.join("p.intent"), i).unwrap();
    }
    dir
}

/// The syntax tree with the digits taken out (the columns of the spans
/// differ by the width of `:` and `{ }`).
fn ast_shape(dir: &Path) -> String {
    let (ok, out, err) = viv(dir, &["p.viv", "--emit", "ast"]);
    assert!(ok, "{err}");
    out.chars().filter(|c| !c.is_ascii_digit()).collect()
}

/// stdout of the program in C and in direct wasm, each when its tools exist.
fn compiled(dir: &Path) -> (Option<String>, Option<String>) {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let (ok, _, err) = viv(dir, &["p.viv", "-o", "p.c"]);
    assert!(ok, "C: {err}");
    let native = Command::new(&cc).current_dir(dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().ok().filter(|o| o.status.success()).map(|_| {
        String::from_utf8_lossy(&Command::new(dir.join("p")).output().unwrap().stdout).into_owned()
    });
    let wasm_rt = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let wasm = (wasm_rt.exists() && Command::new("node").arg("--version").output().is_ok()).then(|| {
        let (ok, _, err) = viv(dir, &["p.viv", "-o", "p.wasm"]);
        assert!(ok, "wasm: {err}");
        let o = Command::new("node").current_dir(dir).arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm_rt).output().unwrap();
        String::from_utf8_lossy(&o.stdout).into_owned()
    });
    (native, wasm)
}

// A def with one-line steps of every kind, a step inside a braced step, a
// one-line step inside an `if` on one line, and a one-line step at the top
// level; `ONE_LINE` and `SAME_LINES` differ only in `:` against `{ ... }`.
const ONE_LINE: &str = r#"def f(X: Arr) {
    step norm: N = X * 2.0
    step shift "add one": Y = N + 1.0
    step bump: for i to 3 {
        Y[i] = Y[i] + i
    }
    step clip: if Y[0] > 0.0 {
        Y[0] = 0.0
    }
    else {
        Y[0] = 1.0
    }
    step outer {
        step inner: Y = Y * 1.0
    }
    if Y[1] > 0.0 { step tail: Y = Y + 0.5 }
    step total: s = sum(Y)
    return s
}
step run: v = f([1.0, 2.0, 3.0])
print(v)
"#;

const SAME_LINES: &str = r#"def f(X: Arr) {
    step norm { N = X * 2.0 }
    step shift "add one" { Y = N + 1.0 }
    step bump { for i to 3 {
        Y[i] = Y[i] + i
    } }
    step clip { if Y[0] > 0.0 {
        Y[0] = 0.0
    }
    else {
        Y[0] = 1.0
    } }
    step outer {
        step inner { Y = Y * 1.0 }
    }
    if Y[1] > 0.0 { step tail { Y = Y + 0.5 } }
    step total { s = sum(Y) }
    return s
}
step run { v = f([1.0, 2.0, 3.0]) }
print(v)
"#;

const MULTI_LINE: &str = r#"def f(X: Arr) {
    step norm {
        N = X * 2.0
    }
    step shift "add one" {
        Y = N + 1.0
    }
    step bump {
        for i to 3 {
            Y[i] = Y[i] + i
        }
    }
    step clip {
        if Y[0] > 0.0 {
            Y[0] = 0.0
        }
        else {
            Y[0] = 1.0
        }
    }
    step outer {
        step inner {
            Y = Y * 1.0
        }
    }
    if Y[1] > 0.0 {
        step tail {
            Y = Y + 0.5
        }
    }
    step total {
        s = sum(Y)
    }
    return s
}
step run {
    v = f([1.0, 2.0, 3.0])
}
print(v)
"#;

const DEBUG: &str = "f\n  watch at norm      N\n  watch at total     s\n  hold at clip       Y[0] >= 0.0   \"the first entry is not negative\"\n";

#[test]
fn every_tool_reads_a_one_line_step_as_the_braced_step() {
    let a = program("a", ONE_LINE, Some(DEBUG), None);
    let b = program("b", SAME_LINES, Some(DEBUG), None);
    assert_eq!(ast_shape(&a), ast_shape(&b), "syntax tree");
    for args in [
        &["run", "p.viv"][..],
        &["hash", "p.viv"],
        &["story", "p.viv"],
        &["story", "p.viv", "--json"],
        &["explain", "p.viv", "--json"],
        &["view", "p.viv", "--lens", "intent"],
        &["view", "p.viv", "--lens", "debug"],
        &["view", "p.viv", "--lens", "story"],
        &["lens", "init", "p.viv", "--stdout"],
        &["lsfunc", "p.viv"],
    ] {
        let (oa, ta, ea) = viv(&a, args);
        let (ob, tb, eb) = viv(&b, args);
        assert!(oa && ob, "viv {args:?}:\n{ea}\n{eb}");
        assert_eq!(ta, tb, "viv {args:?}");
    }
    // the doc tool reads steps from the text: the same steps, lines and depths
    let doc = |dir: &Path| {
        let (ok, out, err) = viv(dir, &["doc", "p.viv"]);
        assert!(ok, "{err}");
        out[out.find("\"functions\":").expect("functions")..].to_string()
    };
    assert_eq!(doc(&a), doc(&b), "doc");
    let (_, out, _) = viv(&a, &["run", "-q", "p.viv"]);
    assert_eq!(out, "16.5\n");
    let (na, wa) = compiled(&a);
    let (nb, wb) = compiled(&b);
    assert_eq!(na, nb, "C");
    assert_eq!(wa, wb, "wasm");
    if let Some(n) = na {
        assert!(n.contains("16.5\n"), "C printed {n}");
    }
    if let Some(w) = wa {
        assert!(w.starts_with("16.5\n"), "wasm printed {w}");
    }
}

#[test]
fn against_the_braced_form_on_several_lines_only_the_line_numbers_move() {
    let a = program("one", ONE_LINE, Some(DEBUG), None);
    let b = program("multi", MULTI_LINE, Some(DEBUG), None);
    assert_eq!(ast_shape(&a), ast_shape(&b), "syntax tree");
    for args in [
        &["hash", "p.viv"][..],
        &["lens", "init", "p.viv", "--stdout"],
        &["view", "p.viv", "--lens", "pseudocode"],
        &["story", "p.viv"],
        &["run", "-q", "p.viv"],
    ] {
        let (oa, ta, ea) = viv(&a, args);
        let (ob, tb, eb) = viv(&b, args);
        assert!(oa && ob, "viv {args:?}:\n{ea}\n{eb}");
        assert_eq!(ta, tb, "viv {args:?}");
    }
    // the pseudocode shows a one-line step's statement without its header
    let (_, pseudo, _) = viv(&a, &["view", "p.viv", "--lens", "pseudocode"]);
    assert!(pseudo.contains("\n    N ← X * 2.0 ") && !pseudo.contains("step norm: N"), "{pseudo}");
}

#[test]
fn intent_entries_for_steps_stay_fresh_across_the_two_forms() {
    let a = program("intent_one", ONE_LINE, None, None);
    let (ok, init, err) = viv(&a, &["lens", "init", "p.viv", "--stdout"]);
    assert!(ok, "{err}");
    // the stamps written for the one-line program hold for the braced one
    let b = program("intent_multi", MULTI_LINE, None, Some(&init));
    // (the entries have no text yet, which the check reports; no stamp is stale)
    let (_, check, err) = viv(&b, &["lens", "check", "p.viv"]);
    assert!(check.contains("step-text: step `f/norm` has no text"), "{check}\n{err}");
    assert!(!check.contains("stale") && !err.contains("stale"), "{check}\n{err}");
}

/// Run `src` in the interpreter; (stdout, stderr, exit 0).
fn run(tag: &str, src: &str) -> (String, String, bool) {
    let dir = program(tag, src, None, None);
    let (ok, out, err) = viv(&dir, &["run", "-q", "p.viv"]);
    (out, err, ok)
}

#[test]
fn the_grammar_cases() {
    // `;` is a line break: the second statement is after the step
    let (out, _, ok) = run("semi", "x = 0.0\nstep a: x = 1.0; y = x + 1.0\nprint(x, y)\n");
    assert!(ok);
    assert_eq!(out, "1 2\n");
    let dir = program("semi_ev", "x = 0.0\nstep a: x = 1.0; y = x + 1.0\nprint(y)\n", None, None);
    let (_, ev, _) = viv(&dir, &["run", "p.viv"]);
    assert!(ev.contains(r#""events":["+<main>/a@1","-<main>/a@1"]"#), "{ev}");
    // a one-line step holding a one-line step
    let (out, _, ok) = run("nested", "step a: step b: x = 2.0\nprint(x)\n");
    assert!(ok);
    assert_eq!(out, "2\n");
    // `step` as a variable, a record field and a parameter
    let (out, err, ok) = run("names", "record R { step: Num }\nstep = 0.5\nr = R { step: step }\ndef g(step) {\n    return step * 2.0\n}\nprint(g(r.step))\n");
    assert!(ok, "{err}");
    assert_eq!(out, "1\n");
    // an assignment whose value is a phrase
    let (out, err, ok) = run("phrase", "v = [1.0, 2.0, 3.0]\nstep total: s = take the sum of v\nprint(s)\n");
    assert!(ok, "{err}");
    assert_eq!(out, "6\n");
    // a statement that is a bare call, and one with a comment after it
    let (out, err, ok) = run("call", "step say: print(\"hi\")   # a comment\nstep again: print(\"there\")\n");
    assert!(ok, "{err}");
    assert_eq!(out, "hi\nthere\n");
}

#[test]
fn a_one_line_step_needs_its_statement_on_its_line() {
    for (tag, src) in [("empty", "step a:\n    x = 1.0\nprint(x)\n"), ("eof", "x = 1.0\nstep a:"), ("brace", "def f() {\n    step a: }\n    return 1.0\n}\nprint(f())\n")] {
        let (_, err, ok) = run(tag, src);
        assert!(!ok, "{tag}: ran");
        assert!(err.contains("a one-line step holds one statement, written after its `:` on the same line"), "{tag}: {err}");
    }
    // a second step of the same name is refused in either form
    let (_, err, ok) = run("dup", "def f() {\n    step a: x = 1.0\n    step a {\n        x = 2.0\n    }\n    return x\n}\nprint(f())\n");
    assert!(!ok);
    assert!(err.contains("`f` already has a step named `a`"), "{err}");
}
