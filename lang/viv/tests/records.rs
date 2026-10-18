// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Records: value structs with named fields. Checks the
//! runtime semantics (by-value scalars, shared array handles) by compiling and
//! running, plus the static errors from §7. Skips runtime checks if no C compiler.

use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

/// Compile and run a Viv program; return its stdout lines. None if no compiler.
fn run(src: &str, tag: &str) -> Option<Vec<String>> {
    let program = viv::build_ir(src, SourceId(0)).expect("should lower");
    let c = CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string();
    // a directory per call: two tests that pass the same tag run at once and
    // one removed the other's directory while it was compiling (B093)
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let k = CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("viv_rec_{}_{tag}_{k}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("r.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("r");
    let out = Command::new(&cc).args(["-std=c99", "-O2", "-Wno-unknown-pragmas"]).arg(dir.join("r.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}\n--- C ---\n{c}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&run.stdout).lines().map(|l| l.trim().to_string()).collect())
}

/// Lower a program and return the error messages (empty if it type-checks).
fn errors(src: &str) -> Vec<String> {
    match viv::build_ir(src, SourceId(0)) {
        Ok(_) => Vec::new(),
        Err(ds) => ds.into_iter().map(|d| d.message).collect(),
    }
}

#[test]
fn record_by_value_roundtrip() {
    let src = r#"
record Vec3 { x: Num, y: Num, z: Num }
def add3(p: Vec3, q: Vec3) -> Vec3 {
    return Vec3 { x: p.x + q.x, y: p.y + q.y, z: p.z + q.z }
}
a = Vec3 { x: 1.0, y: 2.0, z: 3.0 }
b = Vec3 { z: 30.0, x: 10.0, y: 20.0 }
c = add3(a, b)
show c.x
show c.y
show c.z
"#;
    let Some(out) = run(src, "roundtrip") else { return };
    // Field order in the literal is irrelevant (designated initializers). `show`
    // labels only bare variables, so a field expression prints just the value.
    assert_eq!(out, vec!["11", "22", "33"]);
}

#[test]
fn array_field_is_a_shared_handle() {
    // §3.3: copying a record copies the array *handle*; storage is shared.
    let src = r#"
record Box { v: Arr }
xs = zeros(3)
xs[0] = 5.0
b1 = Box { v: xs }
b2 = b1
b2.v[1] = 9.0
show b1.v[0]
show b1.v[1]
"#;
    let Some(out) = run(src, "shared") else { return };
    assert_eq!(out, vec!["5", "9"]);
}

#[test]
fn static_errors() {
    // §7 errors. Each program should fail to lower with a relevant message.
    let cases: &[(&str, &str)] = &[
        ("record V { x: Num }\np = W { x: 1.0 }\nshow p.x", "unknown record type"),
        ("record V { x: Num, y: Num }\np = V { x: 1.0 }\nshow p.x", "missing field"),
        ("record V { x: Num }\np = V { x: 1.0, w: 2.0 }\nshow p.x", "no field `w`"),
        ("record V { x: Num }\np = V { x: 1.0, x: 2.0 }\nshow p.x", "set more than once"),
        ("record V { x: Num }\np = V { x: 1.0 }\nshow p.w", "no field `w`"),
        ("n = 1.0\nshow n.x", "cannot access field"),
        ("record V { x: Num }\nrecord V { x: Num }", "already defined"),
        ("record V { x: Num, x: Num }", "duplicate field"),
        ("record V { x: Num }\na = V { x: 1.0 }\nb = V { x: 2.0 }\nc = a == b", "records take + - * / only"),
    ];
    for (src, want) in cases {
        let errs = errors(src);
        assert!(errs.iter().any(|e| e.contains(want)), "program:\n{src}\nexpected an error containing {want:?}, got {errs:?}");
    }
}

/// The interpreter's output with the call-graph lines taken off.
fn interp(src: &str) -> Result<Vec<String>, String> {
    viv::interp::run(src).map(|out| out.lines().filter(|l| !l.starts_with("__")).map(|l| l.trim().to_string()).collect())
}

#[test]
fn interpreter_runs_records_as_c_does() {
    // R1: the same programs, the same lines.
    let cases: &[(&str, &str)] = &[
        (
            "roundtrip",
            "record Vec3 { x: Num, y: Num, z: Num }\ndef add3(p: Vec3, q: Vec3) -> Vec3 {\n    return Vec3 { x: p.x + q.x, y: p.y + q.y, z: p.z + q.z }\n}\na = Vec3 { x: 1.0, y: 2.0, z: 3.0 }\nb = Vec3 { z: 30.0, x: 10.0, y: 20.0 }\nc = add3(a, b)\nshow c.x\nshow c.y\nshow c.z\n",
        ),
        ("shared", "record Box { v: Arr }\nxs = zeros(3)\nxs[0] = 5.0\nb1 = Box { v: xs }\nb2 = b1\nb2.v[1] = 9.0\nshow b1.v[0]\nshow b1.v[1]\nshow xs\n"),
        (
            "nested",
            "record P { x: Num, y: Num }\nrecord S { a: P, b: P }\ndef mid(s: S) -> P {\n    return P { x: (s.a.x + s.b.x) / 2, y: (s.a.y + s.b.y) / 2 }\n}\ns = S { a: P { x: 0, y: 0 }, b: P { x: 4, y: 2 } }\nm = mid(s)\nshow m.x\nshow m.y\nt = s\nshow t.b.x\n",
        ),
        ("loop", "record Acc { n: Num, v: Arr }\na = Acc { n: 0, v: zeros(2) }\nfor i to 3 {\n    a = Acc { n: a.n + i, v: a.v }\n    a.v[0] += 1\n}\nshow a.n\nshow a.v\n"),
    ];
    for (tag, src) in cases {
        let got = interp(src).unwrap_or_else(|e| panic!("{tag}: the interpreter stopped: {e}"));
        if let Some(c) = run(src, tag) {
            assert_eq!(got, c, "{tag}: interpreter and C differ");
        }
    }
}

#[test]
fn interpreter_refuses_what_the_compile_refuses() {
    // Checked at run time here (the browser's run path does not lower
    // first), with the lowering's own words.
    let cases: &[(&str, &str)] = &[
        ("record V { x: Num }\nrecord W { x: Num }\np = V { x: 1.0 }\nq = W { x: 2.0 }\nps = [p, q]", "an array of records holds records of one type"),
        ("record V { x: Num, y: Num }\np = V { x: 1.0 }", "missing field"),
        ("record V { x: Num }\np = V { x: 1.0, w: 2.0 }", "no field `w`"),
        ("record V { x: Num }\np = V { x: 1.0 }\nshow p.w", "no field `w`"),
        ("record V { x: Num }\np = V { x: zeros(2) }", "field `x` of `V` is a number but this value is an array"),
        ("record V { x: Num }\na = V { x: 1.0 }\nb = a + a", "`+` over a record acts on its arrays, and the field `x` of `V` is a number"),
        ("record V { x: Num }\na = V { x: 1.0 }\nshow a", "cannot `show` a V"),
        ("record V { x: Num }\nrecord W { x: Num }\ndef f(p: V) {\n    return p.x\n}\nshow f(W { x: 1 })", "argument 1 of `f` should be a V but is a W"),
        ("record V { x: Num }\nrecord V { x: Num }", "already defined"),
    ];
    for (src, want) in cases {
        let err = viv::interp::run(src).expect_err(src);
        assert!(err.contains(want), "program:\n{src}\nexpected an error containing {want:?}, got {err:?}");
        // and the compile refuses it too
        let errs = errors(src);
        assert!(!errs.is_empty(), "the compile accepted:\n{src}");
    }
}

#[test]
fn field_assignment_changes_the_variable_s_record() {
    // fields are assignable (tests/objects.rs)
    assert!(errors("record V { x: Num }\np = V { x: 1.0 }\np.x = 2.0").is_empty());
    let Some(out) = run("record V { x: Num }\np = V { x: 1.0 }\nq = p\np.x = 2.0\nshow p.x\nshow q.x\n", "assign") else { return };
    assert_eq!(out, vec!["2", "1"]);
}

/// B093: two runs with one tag at the same time each get their own
/// directory (they shared one, and one removed it under the other).
#[test]
fn two_runs_with_one_tag_at_once_both_print() {
    let src = "show 7.0\n";
    let a = std::thread::spawn(move || run(src, "same"));
    let b = std::thread::spawn(move || run(src, "same"));
    for r in [a.join().unwrap(), b.join().unwrap()] {
        if let Some(out) = r {
            assert_eq!(out, vec!["7"]);
        }
    }
}
