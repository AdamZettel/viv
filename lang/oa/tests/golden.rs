// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Golden tests for the operator-algebra probe: emit C, compile, run, and check
//! the printed matrices against hand-computed operator algebra. (There is no
//! reference interpreter for this language; its cross-check is that the kit
//! supported it with no Knot-specific changes — only the principled
//! generalization of `CType` — and that the algebra comes out right.)

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn emit_c(src: &str) -> String {
    let program = oa::build_ir(src, SourceId(0)).expect("build_ir should succeed");
    CBackend
        .emit(&program, &MemoryIo::new())
        .expect("emission")
        .artifact
        .as_c()
        .expect("C")
        .to_string()
}

fn run(src: &str) -> Option<Vec<String>> {
    use std::process::Command;
    let c = emit_c(src);
    let dir = std::env::temp_dir().join(format!("oa_golden_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c_path = dir.join("prog.c");
    let bin = dir.join("run");
    std::fs::write(&c_path, &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let compile = match Command::new(&cc).args(["-std=c99", "-Wall", "-Werror"]).arg(&c_path).arg("-o").arg(&bin).output() {
        Ok(o) => o,
        Err(_) => return None,
    };
    assert!(compile.status.success(), "oa-generated C did not compile:\n{}\n--- C ---\n{c}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&out.stdout).lines().map(|l| l.to_string()).collect())
}

#[test]
fn emits_with_custom_runtime_and_named_type() {
    let c = emit_c("let a = X * Z\nshow a");
    // The DSL's own runtime, via the prelude hook.
    assert!(c.contains("typedef struct { double m[4]; } oa_op;"), "missing runtime:\n{c}");
    // The custom C type, via the generalized CType::Named.
    assert!(c.contains("oa_op a = oa_mul(oa_X(), oa_Z());"), "let/decl wrong:\n{c}");
    assert!(c.contains("oa_show(a);"), "show wrong:\n{c}");
}

#[test]
fn operator_algebra_is_correct() {
    let Some(out) = run(
        "show X * Z\n\
         show comm(X, Z)\n\
         show scale(2.0, X) + I\n\
         let p = X * Z\n\
         let q = Z * X\n\
         show p - q\n",
    ) else {
        eprintln!("skipping: no C compiler");
        return;
    };
    assert_eq!(
        out,
        vec![
            "[[0, -1], [1, 0]]",   // XZ
            "[[0, -2], [2, 0]]",   // [X, Z] = XZ - ZX
            "[[1, 2], [2, 1]]",    // 2X + I
            "[[0, -2], [2, 0]]",   // p - q = XZ - ZX = [X, Z]
        ]
    );
}

#[test]
fn rejects_unknown_operator() {
    let err = oa::build_ir("show Y", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("unknown operator `Y`")));
}

#[test]
fn rejects_bare_number() {
    let err = oa::build_ir("show 2.0", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("not an operator")));
}
