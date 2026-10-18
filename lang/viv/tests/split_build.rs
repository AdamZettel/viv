// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Separate compilation: `viv build` splits
//! a program's C into one translation unit per imported module and caches
//! each unit's object. The split build must print what the one-file build
//! prints, byte for byte: the shared runtime state (the arena, the random
//! numbers, the test counters, the steps) is one copy across the units.
//! These tests need a C compiler; without one they say so and pass.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn have_cc() -> bool {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    Command::new(cc).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

const MODULE: &str = "###
post: a module the split build compiles apart
###
record Pt { x: Num, y: Num }
M_SCALE = 1.5
def m_norms(n) -> Arr {
    step fill {
        v = zeros(n)
        for i to n {
            v[i] = sqrt(i * M_SCALE + rand())
        }
    }
    return v
}
def m_label(k) -> Str {
    return f\"item {k:.0f}\" + \" done\"
}
def m_pt(a) -> Pt {
    return Pt { x: a, y: 2.0 * a }
}
def m_check(v: Arr) {
    assert_lt(v[0], 10.0)
    return sum(v)
}
def m_oob(v: Arr) {
    return v[len(v)]
}
";

const PROGRAM: &str = "from m import m_norms, m_label, m_pt, m_check, m_oob
print(rand())
w = m_norms(5)
print(w)
print(m_label(3))
q = m_pt(1.5)
print(q.x + q.y)
print(rand())
test \"checks in the unit and in the program\" {
    assert_near(m_check(w), sum(w), 0.0)
    assert_lt(w[1], 0.0)
}
print(m_oob(w))
";

fn setup(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_split_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("m.viv"), MODULE).unwrap();
    std::fs::write(dir.join("p.viv"), PROGRAM).unwrap();
    dir
}

fn build_and_run(dir: &Path, args: &[&str], exe: &str) -> (String, String) {
    let b = Command::new(viv()).current_dir(dir).env("VIV_CACHE_DIR", dir.join("cache")).arg("build").args(args).args(["p.viv", "-o", exe]).output().unwrap();
    assert!(b.status.success(), "viv build {args:?}: {}{}", String::from_utf8_lossy(&b.stdout), String::from_utf8_lossy(&b.stderr));
    let r = Command::new(dir.join(exe)).current_dir(dir).output().unwrap();
    (String::from_utf8_lossy(&b.stdout).into_owned(), format!("{}\n--stderr--\n{}\n--status {:?}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr), r.status.code()))
}

#[test]
fn the_split_build_prints_what_the_one_file_build_prints() {
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let dir = setup("same");
    let (_, single) = build_and_run(&dir, &["--single"], "s1");
    let (line, split) = build_and_run(&dir, &[], "s2");
    assert!(line.contains("2 units, 0 from the cache"), "{line}");
    assert_eq!(single, split);
    // the module's out-of-bounds read names the module's own line
    assert!(split.contains("m.viv:26:12"), "{split}");
    // the release and fast modes too
    for mode in [["--release"], ["--fast"]] {
        let (_, a) = build_and_run(&dir, &[&["--single"][..], &mode[..]].concat(), "r1");
        let (_, b) = build_and_run(&dir, &mode, "r2");
        assert_eq!(a, b, "{mode:?}");
    }
}

#[test]
fn a_second_build_takes_every_unit_from_the_cache() {
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let dir = setup("cache");
    let (first, a) = build_and_run(&dir, &[], "c1");
    assert!(first.contains("2 compiled"), "{first}");
    let (second, b) = build_and_run(&dir, &[], "c2");
    assert!(second.contains("2 units, 2 from the cache, 0 compiled"), "{second}");
    assert_eq!(a, b);
    // a change to the program recompiles its own unit only
    std::fs::write(dir.join("p.viv"), PROGRAM.replace("print(rand())\nw =", "print(rand() + 1.0)\nw =")).unwrap();
    let (third, _) = build_and_run(&dir, &[], "c3");
    assert!(third.contains("2 units, 1 from the cache, 1 compiled"), "{third}");
    // another mode is another entry
    let (fourth, _) = build_and_run(&dir, &["--release"], "c4");
    assert!(fourth.contains("0 from the cache"), "{fourth}");
}

// B260: a program global named like a record field the module reads (`o.x`
// in the module, `x` an enum global in the program) kept the module in the
// program's unit, and `viv build` emitted the whole program once more for
// each such module. A field is not a reference to the global.
const FIELD_MODULE: &str = "###
post: a module that reads a record's field named like the program's global
###
record Opts { x: Num, method: Num }
def f_twice(o: Opts) {
    return o.x * 2.0 + o.method
}
";

const FIELD_PROGRAM: &str = "from f import Opts, f_twice
enum Method {
    Rhf
    Uhf
}
method = Method.Uhf
x = 3.0
def code(m: Method) {
    match m {
        Rhf => { return 1.0 }
        Uhf => { return 2.0 }
    }
}
print(f_twice(Opts { x: x, method: code(method) }))
";

#[test]
fn a_field_named_like_an_enum_global_does_not_keep_the_module_in_the_program() {
    if !have_cc() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("viv_split_{}_field", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("f.viv"), FIELD_MODULE).unwrap();
    std::fs::write(dir.join("p.viv"), FIELD_PROGRAM).unwrap();
    let s = Command::new(viv()).current_dir(&dir).args(["p.viv", "--split", "out"]).output().unwrap();
    let line = String::from_utf8_lossy(&s.stdout).into_owned();
    assert!(s.status.success(), "{line}{}", String::from_utf8_lossy(&s.stderr));
    assert!(line.contains("(1 units)"), "{line}");
    assert!(!line.contains("kept in main.c"), "{line}");
    let (_, single) = build_and_run(&dir, &["--single"], "s1");
    let (_, split) = build_and_run(&dir, &[], "s2");
    assert_eq!(single, split);
    assert!(split.starts_with("8\n"), "{split}");
}
