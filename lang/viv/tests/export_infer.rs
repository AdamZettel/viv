// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv export --infer file.viv`: a kernel collection with no manifest, of
//! the file's top-level defs whose parameters are annotated (the Python
//! package vipyr's entry point).
//!
//! - the annotated defs are exported, with results read from the compiled
//!   program: one value, a tuple, an `[Int]` array, a complex number;
//! - an unannotated def, a private one, one taking a string and one
//!   returning nothing are left out, each with a note on stderr and in
//!   `<name>.notes`;
//! - the stub `<name>.pyi` has each kernel's signature and its `#|` text;
//! - the C builds and, through the wrapper, gives the defs' values;
//! - a file with nothing to export, and a file that does not parse, are
//!   refused, the second located at its line and column.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A directory outside any checkout, with `files` in it.
fn scratch(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_infer_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in files {
        std::fs::write(dir.join(name), text).unwrap();
    }
    dir
}

fn infer(dir: &Path, file: &str) -> std::process::Output {
    Command::new(viv_bin()).current_dir(dir).env("VIV_ROOT", repo()).args(["export", "--infer", file, "-o", "out"]).output().unwrap()
}

const KERN: &str = r#"import minpack.enorm

#| The sum of squares of x.
#: x  the values
def sumsq(x: Arr) -> Num {
    s = 0.0
    for v in x {
        s += v * v
    }
    return s
}

#| MINPACK's overflow-safe norm.
def safe_norm(x: Arr) -> Num {
    return minpack.enorm.enorm(len(x), x)
}

def minmax(x: Arr) {
    return min(x), max(x)
}

def count_up(n: Int) -> [Int] {
    out: [Int] = []
    for i to n {
        out = append(out, i)
    }
    return out
}

def total(xs: [Int]) -> Int {
    t: Int = 0
    for v in xs {
        t += v
    }
    return t
}

def turn(z: Complex) {
    "z turned a quarter."
    return z * 1.0i
}

def untyped(x) {
    return x
}

def _helper(x: Num) -> Num {
    return x
}

def shout(s: Str) -> Num {
    return 1.0
}

def nothing(x: Num) {
    print(x)
}
"#;

#[test]
fn annotated_defs_are_exported_and_the_others_left_out_with_a_note() {
    let dir = scratch("kern", &[("kern.viv", KERN)]);
    let out = infer(&dir, "kern.viv");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    for (def, why) in [("untyped", "has no type"), ("_helper", "private"), ("shout", "strings cannot"), ("nothing", "returns nothing")] {
        assert!(err.lines().any(|l| l.starts_with("note:") && l.contains(&format!("`{def}`")) && l.contains(why)), "no note on {def}: {err}");
    }
    // the imported module's own defs are not the file's
    assert!(!err.contains("`enorm`"), "{err}");
    let o = dir.join("out");
    let notes = std::fs::read_to_string(o.join("kern.notes")).unwrap();
    assert_eq!(notes.lines().count(), 4, "{notes}");
    let pyi = std::fs::read_to_string(o.join("kern.pyi")).unwrap();
    assert!(pyi.contains("__all__ = [\"sumsq\", \"safe_norm\", \"minmax\", \"count_up\", \"total\", \"turn\"]"), "{pyi}");
    assert!(pyi.contains("def sumsq(x: npt.ArrayLike) -> float:"), "{pyi}");
    assert!(pyi.contains("The sum of squares of x."), "{pyi}");
    assert!(pyi.contains("def minmax(x: npt.ArrayLike) -> Tuple[float, float]:"), "{pyi}");
    assert!(pyi.contains("def count_up(n: int) -> npt.NDArray[np.int64]:"), "{pyi}");
    assert!(pyi.contains("def total(xs: npt.ArrayLike) -> int:"), "{pyi}");
    assert!(pyi.contains("def turn(z: complex) -> complex:") && pyi.contains("z turned a quarter."), "{pyi}");

    // built and called through the wrapper
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let lib = o.join(if cfg!(target_os = "macos") { "libkern.dylib" } else { "libkern.so" });
    let b = Command::new(&cc).current_dir(&o).args(["-O2", "-ffp-contract=off", "-fvisibility=hidden", "-shared", "-fPIC", "kern.c", "-o"]).arg(&lib).arg("-lm").output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let has_numpy = Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success());
    if !has_numpy {
        eprintln!("skipped the Python calls: no python3 with numpy");
        return;
    }
    let py = "import kern, numpy as np\n\
              assert kern.sumsq(np.array([1.0, 2.0, 3.0])) == 14.0\n\
              assert kern.safe_norm(np.array([3.0, 4.0])) == 5.0\n\
              assert kern.minmax(np.array([3.0, -1.0, 5.0])) == (-1.0, 5.0)\n\
              c = kern.count_up(4)\n\
              assert c.dtype == np.int64 and list(c) == [0, 1, 2, 3], c\n\
              assert kern.total(np.array([1, 2, 3])) == 6\n\
              assert kern.turn(1 + 2j) == (1 + 2j) * 1j\n\
              assert kern.__all__ == ['sumsq', 'safe_norm', 'minmax', 'count_up', 'total', 'turn']\n\
              print('ok')\n";
    let r = Command::new("python3").current_dir(&o).args(["-c", py]).output().unwrap();
    assert!(r.status.success() && String::from_utf8_lossy(&r.stdout).contains("ok"), "{}", String::from_utf8_lossy(&r.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_with_nothing_to_export_is_refused() {
    let dir = scratch("none", &[("bare.viv", "def f(x) {\n    return x\n}\n")]);
    let out = infer(&dir, "bare.viv");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("no def can be exported") && err.contains("`f`"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_compile_error_is_located_in_the_file() {
    let dir = scratch("bad", &[("bad.viv", "def f(x: Num) -> Num {\n    return x +\n}\n")]);
    let out = infer(&dir, "bad.viv");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.lines().any(|l| l.trim_start().starts_with("--> ") && l.contains("bad.viv:")), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_module_name_that_is_not_a_name_is_refused() {
    let dir = scratch("dash", &[("my-kern.viv", "def f(x: Num) -> Num {\n    return x\n}\n")]);
    let out = infer(&dir, "my-kern.viv");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("must be a name"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// B293: a file of defs alone has no top level, the C backend writes no
/// `main` for it, and the kernel entry's init called the missing
/// `viv_program_main`; the export now gives it an empty one.
#[test]
fn a_module_of_defs_alone_builds() {
    let dir = scratch("defs", &[("twice.viv", "def twice(x: Num) -> Num {\n    return 2.0 * x\n}\n")]);
    let out = infer(&dir, "twice.viv");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let b = Command::new(&cc).current_dir(dir.join("out")).args(["-O2", "-ffp-contract=off", "-fvisibility=hidden", "-shared", "-fPIC", "twice.c", "-o", "libtwice.so", "-lm"]).output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

/// An array argument the def only reads is the caller's buffer
/// (`vivk_lend_arr`); one any name holding its storage writes is copied
/// (`vivk_in_arr`): an element write, an alias written, a builtin that
/// writes, a def that writes, a def of another module (not followed), a
/// top-level name that outlives the call.
#[test]
fn read_only_array_arguments_are_lent_and_written_ones_copied() {
    let src = "\
G = [0.0]

def poke(x: Arr) {
    x[0] = 1
}

def peek(x: Arr) -> Num {
    return x[0]
}

def reads(a: Arr) -> Num {
    s = 0.0
    for i to len(a) {
        s += a[i]
    }
    return s + peek(a)
}

def rebinds(a: Arr) -> Num {
    a = a + 1
    a = append(a, 2.0)
    return a[0]
}

def ints(k: [Int]) -> Int {
    return k[0]
}

def writes(a: Arr) -> Num {
    a[0] = 2
    return a[0]
}

def alias_writes(a: Arr) -> Num {
    b = a
    c = b
    c[0] = 5
    return a[0]
}

def sorts(a: Arr) -> Num {
    b = a
    sort(b)
    return b[0]
}

def calls_writer(a: Arr) -> Num {
    poke(a)
    return a[0]
}

def keeps(a: Arr) -> Num {
    G = a
    return G[0]
}

def mixed(a: Arr, b: Arr) -> Num {
    b[0] = a[0]
    return b[0]
}
";
    let dir = scratch("lend", &[("lend.viv", src)]);
    let out = infer(&dir, "lend.viv");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let c = std::fs::read_to_string(dir.join("out/lend.c")).unwrap();
    // each kernel's body, from its comment to the next
    let body = |k: &str| -> String {
        let at = c.find(&format!("static int vivk_body_{k}(void)")).unwrap_or_else(|| panic!("no body for {k}"));
        c[at..].split("return 0;").next().unwrap().to_string()
    };
    for k in ["reads", "rebinds", "ints", "peek"] {
        assert!(body(k).contains("vivk_lend_arr(&a->"), "{k} should lend:\n{}", body(k));
        assert!(!body(k).contains("vivk_in_arr("), "{k} should not copy:\n{}", body(k));
    }
    for k in ["writes", "alias_writes", "sorts", "calls_writer", "keeps"] {
        assert!(body(k).contains("vivk_in_arr(&a->"), "{k} should copy:\n{}", body(k));
    }
    let m = body("mixed");
    assert!(m.contains("vivk_lend_arr(&a->a)") && m.contains("vivk_in_arr(&a->b)"), "{m}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A module whose def takes `grad` of a def that calls into another module
/// exports (B295): the file read alone, for its defs' signatures, is not
/// put through the `grad` rewrite, which needs the imported defs.
#[test]
fn grad_through_an_imported_def_exports() {
    let dir = scratch(
        "gradimp",
        &[
            ("aa.viv", "def sq(x: Num) -> Num {\n    return x * x\n}\n"),
            ("bb.viv", "from aa import sq\n\ndef w(x: Num) -> Num {\n    return sq(x)\n}\n\ndef g(x: Num) -> Num {\n    return grad(w, x)\n}\n"),
        ],
    );
    let out = infer(&dir, "bb.viv");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn keyword_arguments_to_an_imported_def_export() {
    // B297: the file read alone did not know the imported def's signature
    let dir = scratch(
        "kwimp",
        &[
            ("aa.viv", "def f(x: Num, a: Num = 2.0) -> Num {\n    return a * x\n}\n"),
            ("bb.viv", "from aa import f\n\ndef g(x: Num, a: Num) -> Num {\n    return f(x, a=a)\n}\n"),
        ],
    );
    let out = infer(&dir, "bb.viv");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

/// B347: a def named like a C library function (`index`, strings.h; `time`,
/// time.h) is emitted with the C backend's `_`, and the exported entry
/// calls it by that name, not libc's function.
#[test]
fn a_def_named_like_a_libc_function_exports_and_is_the_one_called() {
    let src = "def index(b: [Int]) -> Int {\n    return len(b) + 1\n}\n\ndef time(x: Num) -> Num {\n    return 2.0 * x\n}\n";
    let dir = scratch("libcname", &[("libcname.viv", src)]);
    let out = infer(&dir, "libcname.viv");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let o = dir.join("out");
    let c = std::fs::read_to_string(o.join("libcname.c")).unwrap();
    assert!(c.contains("vivk_res = index_(") && c.contains("vivk_res = time_("), "the entries call index and time unescaped");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let lib = o.join(if cfg!(target_os = "macos") { "liblibcname.dylib" } else { "liblibcname.so" });
    let b = Command::new(&cc).current_dir(&o).args(["-O2", "-ffp-contract=off", "-fvisibility=hidden", "-shared", "-fPIC", "libcname.c", "-o"]).arg(&lib).arg("-lm").output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let has_numpy = Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success());
    if has_numpy {
        let py = "import libcname, numpy as np\nassert libcname.index(np.array([4, 5, 6])) == 4\nassert libcname.time(1.5) == 3.0\nprint('ok')\n";
        let r = Command::new("python3").current_dir(&o).args(["-c", py]).output().unwrap();
        assert!(r.status.success() && String::from_utf8_lossy(&r.stdout).contains("ok"), "{}", String::from_utf8_lossy(&r.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
