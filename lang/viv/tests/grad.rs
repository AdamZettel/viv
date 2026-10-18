// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Reverse-mode autodiff, `grad` and `value_and_grad`
//! (`viv::grad`): a program that does not use them is untouched, the
//! probes in study/idioms/autodiff-probes pass and print the same bytes in
//! the interpreter and in C, and what the transform cannot follow is refused
//! with a located error.

use std::path::PathBuf;
use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn probe(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../study/idioms/autodiff-probes").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The program's own output from the interpreter (no call-graph lines).
fn interp(src: &str) -> String {
    let out = viv::interp::run(src).unwrap_or_else(|e| panic!("the interpreter stopped: {e}"));
    out.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

/// The program's output compiled to C; None when there is no C compiler.
fn native(src: &str, tag: &str) -> Option<String> {
    let program = viv::build_ir(src, SourceId(0)).unwrap_or_else(|d| panic!("does not lower: {:?}", d.first().map(|x| &x.message)));
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let dir = std::env::temp_dir().join(format!("viv_grad_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("a");
    let compile = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("a.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(compile.status.success(), "C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn check_probe(name: &str) {
    let src = probe(name);
    let a = interp(&src);
    assert!(a.contains("PASS "), "{name}: no test ran:\n{a}");
    assert!(!a.contains("FAIL ") && !a.contains("error"), "{name}:\n{a}");
    if let Some(b) = native(&src, name.trim_end_matches(".viv")) {
        assert_eq!(a, b, "{name}: the interpreter and C print different bytes");
    }
}

#[test]
fn scalar_probe_passes_in_both_engines() {
    check_probe("scalar.viv");
}

#[test]
fn array_probe_passes_in_both_engines() {
    check_probe("arrays.viv");
}

#[test]
fn axes_probe_passes_in_both_engines() {
    // reductions along an axis, broadcasting between ranks, reshapes and
    // take along an axis, against central differences
    check_probe("axes.viv");
}

#[test]
fn calls_probe_passes_in_both_engines() {
    check_probe("calls.viv");
}

#[test]
fn a_program_without_grad_is_untouched() {
    let src = "def grad_like(x) {\n    grad = x * 2.0\n    return grad\n}\ngrad = zeros(3)\nshow grad_like(1.5)\n";
    let toks = viv::lexer::lex(src, SourceId(0)).unwrap();
    let parsed = viv::parser::parse(toks, SourceId(0)).unwrap();
    let before = format!("{parsed:?}");
    let after = format!("{:?}", viv::grad::transform(parsed).unwrap());
    assert_eq!(before, after);
}

#[test]
fn a_programs_own_grad_def_wins() {
    let src = "def grad(f, x) {\n    return 7.0\n}\ndef sq(x) {\n    return x * x\n}\nshow grad(sq, 3.0)\n";
    assert_eq!(interp(src), "7\n");
}

#[test]
fn value_and_grad_gives_the_value_and_the_gradient() {
    let src = "def f(x) {\n    return x * x * x\n}\nv, g = value_and_grad(f, 2.0)\nshow v\nshow g\nshow grad(f, 3.0)\n";
    assert_eq!(interp(src), "v: 8\ng: 12\n27\n");
    if let Some(c) = native(src, "vg") {
        assert_eq!(c, interp(src));
    }
}

// ---- the defs a gradient's forward pass was written in ----

/// Two defs with steps, one inlined into the other, differentiated inside a
/// loop so each step runs more than once.
const STEPPED: &str = "def inner(x: Arr) {\n    step square {\n        y = x * x\n    }\n    return y\n}\ndef outer(x: Arr) {\n    t = 0.0\n    for k to 2 {\n        step first {\n            a = inner(x)\n        }\n        step total {\n            t = t + sum(a)\n        }\n    }\n    return t\n}\nloss, g = value_and_grad(outer, [1.0, 2.0, 3.0])\nprint(loss)\nshow g\n";

/// A line of the output that starts with `tag`.
fn tagged<'a>(out: &'a str, tag: &str) -> Option<&'a str> {
    out.lines().find(|l| l.starts_with(tag))
}

/// The program's whole output (step events included) from the direct wasm
/// backend, run in node against viv.wasm's runtime; None when node or a
/// viv.wasm with the runtime is missing.
fn direct_wasm(src: &str, tag: &str) -> Option<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root.join("www/public/viv/viv.wasm"));
    let runtime = std::fs::read(&wasm).ok()?;
    if !runtime.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    let bytes = viv::compile_wasm(src).expect("compiles to wasm");
    let dir = std::env::temp_dir().join(format!("viv_grad_wasm_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let m = dir.join("a.wasm");
    std::fs::write(&m, bytes).unwrap();
    let out = Command::new("node").arg(root.join("tools/run-wasm.mjs")).arg(&m).env("VIV_WASM", &wasm).output().ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn steps_inside_a_gradient_are_named_by_the_defs_they_were_written_in_in_every_engine() {
    let full = viv::interp::run(STEPPED).expect("runs");
    let steps = tagged(&full, "__steps__").expect("a __steps__ line").to_string();
    // each step by its own def: inner's square, not outer's
    assert!(steps.contains("\"+outer/first@1\",\"+inner/square@1\",\"-inner/square@1\",\"-outer/first@1\""), "{steps}");
    assert!(!steps.contains("viv__ad"), "{steps}");
    if let Some(c) = native(STEPPED, "steps") {
        assert_eq!(tagged(&c, "__steps__"), Some(steps.as_str()), "C names the steps differently");
    }
    match direct_wasm(STEPPED, "steps") {
        Some(w) => assert_eq!(tagged(&w, "__steps__"), Some(steps.as_str()), "wasm names the steps differently"),
        None => eprintln!("skip wasm: no node or no viv.wasm with the runtime (www/build-wasm.sh)"),
    }
}

#[test]
fn the_call_graph_counts_each_inlined_def_as_a_call() {
    let full = viv::interp::run(STEPPED).expect("runs");
    let graph = tagged(&full, "__graph__").expect("a __graph__ line");
    for edge in ["{\"from\":\"<main>\",\"to\":\"viv__ad_grad_outer\",\"count\":1}", "{\"from\":\"viv__ad_grad_outer\",\"to\":\"outer\",\"count\":1}", "{\"from\":\"outer\",\"to\":\"inner\",\"count\":2}"] {
        assert!(graph.contains(edge), "no {edge} in {graph}");
    }
}

// ---- refusals: each is an error at the line that causes it ----

/// The first error's message and the line of its primary label.
fn refusal(src: &str) -> (String, usize) {
    let Err(ds) = viv::parse(src, SourceId(0)) else { panic!("accepted:\n{src}") };
    let d = &ds[0];
    let at = d.primary.as_ref().expect("a location").span.start as usize;
    (d.message.clone(), src[..at].matches('\n').count() + 1)
}

fn refused(src: &str, words: &str, line: usize) {
    let (msg, at) = refusal(src);
    assert!(msg.contains(words), "message `{msg}` lacks `{words}`");
    assert_eq!(at, line, "`{msg}` located at line {at}, not {line}");
}

#[test]
fn break_is_refused() {
    refused("def f(x) {\n    s = 0.0\n    for i to 3 {\n        if i > 1 {\n            break\n        }\n        s = s + x\n    }\n    return s\n}\nshow grad(f, 1.0)\n", "`break`", 5);
}

#[test]
fn return_inside_a_loop_is_refused() {
    refused("def f(x) {\n    for i to 3 {\n        return x\n    }\n    return 0.0\n}\nshow grad(f, 1.0)\n", "inside a loop", 3);
}

#[test]
fn a_read_that_may_come_before_the_assignment_is_refused() {
    refused(
        "def f(x) {\n    for i to 3 {\n        if i > 0 {\n            y = y * x\n        }\n        else {\n            y = x\n        }\n    }\n    return x\n}\nshow grad(f, 1.0)\n",
        "`y` assigned before",
        4,
    );
}

#[test]
fn an_alias_that_is_written_is_refused() {
    refused(
        "def f(x) {\n    a = zeros(2)\n    b = a\n    b[0] = x\n    return a[0] * x + b[1]\n}\nshow grad(f, 1.0)\n",
        "two names for one array",
        3,
    );
}

#[test]
fn a_write_into_a_parameter_it_reads_is_refused() {
    refused(
        "def f(x, p: Arr) {\n    p[0] = x\n    return p[0] * x\n}\nshow grad(f, 1.0, zeros(2))\n",
        "parameter array `p`",
        2,
    );
}

#[test]
fn a_builtin_without_an_adjoint_is_refused() {
    refused("def f(x) {\n    y = x + 1.0\n    return gamma(y)\n}\nshow grad(f, 1.0)\n", "no adjoint for `gamma`", 3);
}

#[test]
fn a_solver_on_a_differentiated_matrix_is_refused() {
    refused(
        "def f(A: Arr) {\n    vals = zeros(2)\n    vecs = zeros(2, 2)\n    eig_sym(A, vals, vecs)\n    return vals[0]\n}\nshow grad(f, [[2.0, 1.0], [1.0, 3.0]])\n",
        "no adjoint for `eig_sym`",
        4,
    );
}

#[test]
fn recursion_is_refused() {
    refused(
        "def r(x, n) {\n    if n <= 0 {\n        return x\n    }\n    return r(x * x, n - 1)\n}\ndef f(x) {\n    return r(x, 3)\n}\nshow grad(f, 1.0)\n",
        "recursive def `r`",
        8,
    );
}

#[test]
fn nested_grad_is_refused() {
    refused("def sq(x) {\n    return x * x\n}\ndef f(x) {\n    return grad(sq, x)\n}\nshow grad(f, 1.0)\n", "inside a function being differentiated", 5);
}

#[test]
fn value_and_grad_as_one_value_is_refused() {
    refused("def f(x) {\n    return x * x\n}\nshow value_and_grad(f, 1.0)\n", "two results", 4);
}

#[test]
fn one_value_does_not_unpack() {
    // `a, b = f(x)` of a def that returns one number: the lowering names it
    // (two results are `return a, b`, crate::tuples)
    let errs = viv::build_ir("def f(x) {\n    return x\n}\na, b = f(1.0)\n", SourceId(0)).err().expect("refused");
    assert!(errs[0].message.contains("cannot unpack a number into several names"), "{}", errs[0].message);
    assert_eq!(errs.len(), 1);
}

#[test]
fn a_function_value_as_f_is_refused() {
    refused("show grad(fn(x) -> x * x, 1.0)\n", "must name a def", 1);
}

#[test]
fn a_differentiated_call_in_a_while_condition_is_refused() {
    refused(
        "def g(x) {\n    return x * 2.0\n}\ndef f(x) {\n    s = x\n    while g(s) < 10.0 {\n        s = s * 2.0\n    }\n    return s\n}\nshow grad(f, 1.0)\n",
        "`while` condition",
        6,
    );
}

#[test]
fn a_wrong_argument_count_is_refused() {
    refused("def f(x, y) {\n    return x * y\n}\nshow grad(f, 1.0)\n", "takes 2 argument", 4);
}
