// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Skein end-to-end: compile Scheme -> C -> run, check the printed output. This
//! exercises the dynamic value model, heap-allocated lists, recursion, and
//! lambda-lifted closures (with capture). Skips if no C compiler.

use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

/// Compile Scheme source to C, run it, return stdout.
fn run(src: &str, tag: &str) -> Option<String> {
    let program = skein::build_ir(src, SourceId(0)).expect("should compile");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    let dir = std::env::temp_dir().join(format!("skein_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let out = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("p.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}\n--- C ---\n{c}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

#[test]
fn recursion_and_arithmetic() {
    let src = "(define (fact n) (if (= n 0) 1 (* n (fact (- n 1)))))\n(display (fact 10))";
    let Some(out) = run(src, "fact") else { return };
    assert_eq!(out.trim(), "3628800");
}

#[test]
fn closures_capture_and_higher_order() {
    // `adder` returns a closure capturing `n`; `map` is higher-order; lists are
    // heap-allocated cons cells.
    let src = "\
(define (range a b) (if (= a b) (quote ()) (cons a (range (+ a 1) b))))
(define (map f xs) (if (null? xs) (quote ()) (cons (f (car xs)) (map f (cdr xs)))))
(define (adder n) (lambda (x) (+ x n)))
(display (map (adder 100) (range 0 5)))";
    let Some(out) = run(src, "closures") else { return };
    assert_eq!(out.trim(), "(100 101 102 103 104)");
}

#[test]
fn quote_cond_let() {
    let src = "\
(define (classify n)
  (cond ((< n 0) (quote neg))
        ((= n 0) (quote zero))
        (else (quote pos))))
(display (classify -5)) (newline)
(display (classify 7)) (newline)
(display (let ((a 3) (b 4)) (+ a b))) (newline)
(display (let* ((a 3) (b (+ a 1))) (* a b))) (newline)
(display (quote (1 2 (3 4))))";
    let Some(out) = run(src, "qcl") else { return };
    assert_eq!(out, "neg\npos\n7\n12\n(1 2 (3 4))");
}

#[test]
fn proper_tail_calls_run_in_constant_stack() {
    // A million tail-recursive iterations. Without the trampoline this overflows
    // the C stack; with it, tail calls bounce and run in O(1) stack.
    let src = "\
(define (sum-to n acc) (if (= n 0) acc (sum-to (- n 1) (+ acc n))))
(display (sum-to 1000000 0))";
    let Some(out) = run(src, "tco") else { return };
    assert_eq!(out.trim(), "500000500000");
}

#[test]
fn rejects_unbound_variable() {
    let err = skein::build_ir("(display foo)", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("unbound variable")), "{err:?}");
}

#[test]
fn interpreter_agrees_with_compiler() {
    // The interpreter (browser path) and the compiler (native path) must produce
    // the same output -- redundant cross-check (ATLAS/CMS).
    let progs = [
        "(define (fact n) (if (= n 0) 1 (* n (fact (- n 1))))) (display (fact 10))",
        "(define (range a b) (if (= a b) (quote ()) (cons a (range (+ a 1) b))))\n(define (map f xs) (if (null? xs) (quote ()) (cons (f (car xs)) (map f (cdr xs)))))\n(define (adder n) (lambda (x) (+ x n)))\n(display (map (adder 100) (range 0 5)))",
        "(display (let* ((a 3) (b (+ a 1))) (* a b)))",
    ];
    for src in progs {
        let interp = skein::interp::run(src).expect("interp");
        let Some(compiled) = run(src, "xcheck") else { return };
        assert_eq!(interp, compiled, "interp vs compiler disagree on:\n{src}");
    }
}
