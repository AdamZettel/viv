// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The phrase extensions the GPT rewrite needed (section
//! 15): phrases a module declares reach the programs that `#! use` it, a
//! phrase may be `given` names it reads from where it is used, a phrase may
//! hold commas, and a `call` hole takes a call of a def (`the gradient of
//! loss(w, X, y) with respect to w`). Each runs in the interpreter, as C and
//! as wasm with the same output, or is refused by both the interpreter and
//! the compiler.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_phrase_modules_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, PartialEq)]
struct Ran {
    code: Option<i32>,
    out: String,
    err: String,
}

fn ran(o: std::process::Output) -> Ran {
    Ran { code: o.status.code(), out: String::from_utf8_lossy(&o.stdout).into_owned(), err: String::from_utf8_lossy(&o.stderr).into_owned() }
}

/// `p.viv` and the modules beside it, run in every engine there is; the
/// engines must agree byte for byte. Returns the interpreter's run.
fn all(tag: &str, src: &str, modules: &[(&str, &str)]) -> Ran {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    for (name, text) in modules {
        std::fs::write(dir.join(format!("{name}.viv")), text).unwrap();
    }
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap());
    assert_eq!(i.code, Some(0), "{tag}: the interpreter stopped:\n{}", i.err);
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_ok() {
        let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
        assert!(c.status.success(), "{tag}: viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
        let b = Command::new(&cc).current_dir(&dir).args(["-std=c99", "-O0", "p.c", "-lm", "-o", "p"]).output().unwrap();
        assert!(b.status.success(), "{tag}: the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
        let n = ran(Command::new(dir.join("p")).output().unwrap());
        assert_eq!(i, n, "{tag}: interpreter and C disagree");
    }
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    if std::fs::read(&wasm).is_ok_and(|b| b.windows(12).any(|w| w == b"vrt_getslice")) && Command::new("node").arg("--version").output().is_ok() {
        let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap();
        assert!(c.status.success(), "{tag}: the wasm backend refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
        let w = ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap());
        assert_eq!(i, w, "{tag}: interpreter and wasm disagree");
    }
    i
}

/// The interpreter's refusal of `src`; the compiler refuses it too.
fn refused(tag: &str, src: &str) -> String {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let c = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap());
    assert_ne!(c.code, Some(0), "{tag}: the compiler accepted it");
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap());
    assert_ne!(i.code, Some(0), "{tag}: the interpreter ran it");
    i.err
}

const MODULE: &str = "\
phrase \"the spread of v\" -> amax(v) - amin(v)
record Layers { scale: Num, rows: Arr }
phrase \"the scaled row of this layer\" given c, layer -> c.scale * c.rows[layer]
def spread_of(v) {
    return take the spread of v
}
";

#[test]
fn a_modules_phrases_reach_the_programs_that_use_it() {
    let src = "\
from m import Layers, spread_of
w = [1.0, 5.0, 2.0]
print(take the spread of w)
print(spread_of(w))
def row(c: Layers, layer) {
    return take the scaled row of this layer
}
print(row(Layers { scale: 2.0, rows: [1.0, 2.0, 3.0] }, 2))
";
    let r = all("module", src, &[("m", MODULE)]);
    assert_eq!(r.out, "4\n4\n6\n");
    // `viv phrases` resolves the module first, and says whose phrase it is
    let dir = scratch("module-report");
    std::fs::write(dir.join("p.viv"), src).unwrap();
    std::fs::write(dir.join("m.viv"), MODULE).unwrap();
    let o = ran(Command::new(viv_bin()).current_dir(&dir).args(["phrases", "p.viv"]).output().unwrap());
    assert!(o.out.contains("   3  take the spread of w\n      = amax(w) - amin(w)  (a used module's phrase `the spread of V`)"), "{}", o.out);
    assert!(o.out.contains("c.scale * c.rows[layer]"), "{}", o.out);
}

#[test]
fn a_module_read_on_its_own_knows_the_phrases_of_the_modules_it_uses() {
    // the lens tools parse each module's own text; the phrases of the modules
    // its imports name come from the modules `expand_uses` resolved
    viv::remember_module("phrase_modules_test/m", MODULE);
    let own = "import phrase_modules_test.m\ndef f(v) {\n    return take the spread of v\n}\n";
    let prog = viv::parse_keeping_rungs(own, knotkit::span::SourceId(0)).expect("the module parses on its own");
    assert_eq!(prog.phrases.len(), 1);
    assert_eq!(prog.phrases[0].body_text.as_deref(), Some("amax(v) - amin(v)"));
    // without the import the phrase is unknown
    assert!(viv::parse_keeping_rungs("def f(v) {\n    return take the spread of v\n}\n", knotkit::span::SourceId(0)).is_err());
}

#[test]
fn a_given_name_is_read_where_the_phrase_is_used() {
    // a use where `c` is not a variable is refused, naming the given name
    let e = refused(
        "given-missing",
        "record L { scale: Num, rows: Arr }\nphrase \"the scaled row of this layer\" given c, layer -> c.scale * c.rows[layer]\ndef f(layer) {\n    return take the scaled row of this layer\n}\nprint(f(1))\n",
    );
    assert!(e.contains("reads `c` from where it is used, and there is no `c` here"), "{e}");
    // a given name the body does not read is refused at the declaration
    let e = refused("given-unread", "phrase \"the double of v\" given w -> 2.0 * v\nprint(take the double of 3.0)\n");
    assert!(e.contains("the phrase is given `w` and its body does not read it"), "{e}");
    // the hole `v` is the use's expression, even where the def has its own
    // `v`; the given `k` is the def's variable
    let r = all(
        "given-hygiene",
        "phrase \"the scaled v\" given k -> k * v\ndef f(k, v) {\n    return take the scaled v + 1.0\n}\nprint(f(2.0, 5.0))\n",
        &[],
    );
    assert_eq!(r.out, "12\n");
}

#[test]
fn a_phrase_may_hold_commas() {
    let src = "\
phrase \"the mix of q, k and v\" -> q + 10.0 * k + 100.0 * v
v = [3.0, 4.0]
print(take the mix of 1.0, 2.0 and 3.0)
print([take the mix of 1.0, 2.0 and 3.0, 5.0])
print(take the norm of v, 3)
";
    let r = all("commas", src, &[]);
    assert_eq!(r.out, "321\n[321, 5]\n5 3\n");
    // a phrase that reads both at its first comma and past it is refused
    let e = refused("commas-two", "phrase \"the first of a\" -> a\nphrase \"the first of a, b\" -> a + b\nprint(take the first of 1.0, 2.0)\n");
    assert!(e.contains("can be read 2 ways") && e.contains("ending before `, 2`"), "{e}");
}

#[test]
fn a_call_hole_takes_a_call_of_a_def() {
    let src = "\
def loss(w, s) {
    return s * sum(w * w)
}
w0 = [1.0, 2.0]
print(take the gradient of loss(w0, 3.0) with respect to w0)
value, slope = take the value and gradient of loss(w0, 3.0) with respect to w0
print(value)
print(slope)
";
    let r = all("call", src, &[]);
    assert_eq!(r.out, "[6, 12]\n15\n[6, 12]\n");
    // the gradient is taken with respect to the first argument, as `grad` does
    let e = refused("call-second", "def loss(s, w) {\n    return s * sum(w * w)\n}\nw0 = [1.0, 2.0]\nprint(take the gradient of loss(3.0, w0) with respect to w0)\n");
    assert!(e.contains("the gradient is taken with respect to the first argument of `loss`, and that is not `w0`"), "{e}");
    // the hole is a call of a def: a builtin's call is not one
    let e = refused("call-builtin", "w0 = [1.0, 2.0]\nprint(take the gradient of sum(w0) with respect to w0)\n");
    assert!(e.contains("expected a call of a def"), "{e}");
}

#[test]
fn the_gpt_writes_its_operations_as_phrases() {
    // lib/gpt and lib/nn: each phrase lowers to the call it stands for, so
    // `viv phrases` lists them with those calls (since the model became a
    // record, the phrases that looked a matrix
    // up in the flat parameter vector are gone: the code reads it where it is)
    let o = ran(Command::new(viv_bin()).current_dir(root()).args(["phrases", "lib/gpt/gpt.viv"]).output().unwrap());
    assert_eq!(o.code, Some(0), "{}", o.err);
    for (phrase, call) in [
        ("take the last n elements of out", "out[max(0, len(out) - n):len(out)]"),
        ("take the value and gradient of sequence_loss(model, xs, ys) with respect to model", "value_and_grad(sequence_loss, model, xs, ys)"),
        ("take the cross-entropy of logits against y", "cross_entropy(logits, y)"),
    ] {
        assert!(o.out.contains(&format!("{phrase}\n      = {call}")), "`{phrase}` as `{call}` in\n{}", o.out);
    }
}

#[test]
fn the_pseudocode_view_reads_phrases_as_words() {
    // `viv view --lens pseudocode`: each step as its intent, each statement as
    // written with its phrases as words, and how many statements are phrases
    let o = ran(Command::new(viv_bin()).current_dir(root()).args(["view", "lib/gpt/gpt.viv", "--lens", "pseudocode", "--def", "attention"]).output().unwrap());
    assert_eq!(o.code, Some(0), "{}", o.err);
    assert!(o.out.starts_with("attention(X, Wq, Wk, Wv)  —  10 statements: 4 a phrase, 0 with a phrase in them, 6 code\n"), "{}", o.out);
    for line in ["3. hide the future: a position may not attend to a later one", "future ← the causal mask of size T", "A ← the softmax of each row of S", "Q ← X @ Wq"] {
        assert!(o.out.contains(line), "`{line}` in\n{}", o.out);
    }
    assert!(o.out.lines().any(|l| l.contains("S ← S masked where future") && l.ends_with("[phrase]")), "{}", o.out);
    // with --words-only a statement with no phrase and no line entry is a gap
    let w = ran(Command::new(viv_bin()).current_dir(root()).args(["view", "lib/gpt/gpt.viv", "--lens", "pseudocode", "--def", "attention", "--words-only"]).output().unwrap());
    assert!(w.out.contains("[code, line ") && !w.out.contains("Q ← X @ Wq"), "{}", w.out);
}
