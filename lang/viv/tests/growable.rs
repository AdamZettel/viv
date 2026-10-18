// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Growable arrays:
//! `[]`, `a = append(a, x)` and `a = extend(a, b)`.
//!
//! Each program runs in the interpreter (`viv run`), as C compiled natively,
//! and (when node and a viv.wasm with the runtime are present) as a module
//! from the direct wasm backend. A program that runs must print the same bytes
//! in all three; one that stops must stop with the same words, and the
//! interpreter and C with the same location. (A wasm module reports the
//! message only, as it does for every run-time error.)

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

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_growable_{}_{tag}", std::process::id()));
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
    Ran {
        code: o.status.code(),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn interp(dir: &Path) -> Ran {
    ran(Command::new(viv_bin()).current_dir(dir).args(["run", "-q", "p.viv"]).output().unwrap())
}

fn native(dir: &Path) -> Option<Ran> {
    if Command::new(cc()).arg("--version").output().is_err() {
        return None;
    }
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2", "-Wall", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    Some(ran(Command::new(dir.join("p")).output().unwrap()))
}

/// The direct wasm backend in node, or None when there is no runtime to run it.
fn direct(dir: &Path) -> Option<Ran> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(12).any(|w| w == b"vrt_append_n") {
        eprintln!("skip wasm: {} has no growable-array runtime (www/build-wasm.sh builds one)", wasm.display());
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap();
    assert!(c.status.success(), "the wasm backend refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    Some(ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap()))
}

/// Run `src` in every engine; they must agree. Returns the interpreter's run.
fn all(tag: &str, src: &str) -> Ran {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let i = interp(&dir);
    if let Some(n) = native(&dir) {
        assert_eq!(i, n, "interpreter and C disagree on\n{src}");
    }
    if let Some(w) = direct(&dir) {
        if i.code == Some(0) {
            assert_eq!(i, w, "interpreter and wasm disagree on\n{src}");
        } else {
            assert_eq!(w.code, i.code, "{src}");
            assert_eq!(w.err.lines().next(), i.err.lines().next(), "interpreter and wasm stop differently on\n{src}");
        }
    }
    i
}

fn prints(tag: &str, src: &str, out: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(0), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
}

/// A program that stops: stderr is exactly `err`.
fn stops(tag: &str, src: &str, out: &str, err: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(4), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
    assert_eq!(r.err, err, "{tag}");
}

#[test]
fn collect_while_scanning() {
    prints(
        "scan",
        "kept = []\nfor i to 20 {\n    if mod(i * i, 7) == 2 {\n        kept = append(kept, i)\n    }\n}\nshow kept\nshow len(kept)\nshow []\nshow len([])\n",
        "kept: [3, 4, 10, 11, 17, 18]\n6\n[]\n0\n",
    );
}

#[test]
fn rows_and_vectors() {
    prints(
        "rows",
        "m = []\nfor i to 3 {\n    m = append(m, [i, i * i])\n}\nshow m\nv = [1, 2]\nv = extend(v, [3])\nv = extend(v, v)\nshow v\ne = []\ne = extend(e, [])\nshow shape(e)\nz = zeros(0, 2)\nz = append(z, [7, 8])\nshow z\n",
        "m: [[0, 0],\n    [1, 1],\n    [2, 4]]\nv: [1, 2, 3, 1, 2, 3]\n[0]\nz: [[7, 8]]\n",
    );
}

#[test]
fn an_alias_keeps_the_array_as_it_was() {
    // before the append `b` and `a` share storage (a write reaches both);
    // after it they do not, however much room the storage had
    prints(
        "alias",
        "a = []\nfor i to 3 {\n    a = append(a, i)\n}\nb = a\na[0] = 5\na = append(a, 9)\na[1] = 7\nshow b\nshow a\n\
         record R {\n    xs: Arr\n}\nr = R { xs: a }\na = append(a, 1)\nshow len(r.xs)\n",
        "b: [5, 1, 2]\na: [5, 7, 2, 9]\n4\n",
    );
}

#[test]
fn defs_grow_copies_of_what_they_are_given() {
    prints(
        "defs",
        "def grown(p) {\n    p = append(p, 0)\n    return p\n}\ndef count(p) {\n    return len(p)\n}\n\
         def upto(n) {\n    r = []\n    for i to n {\n        r = append(r, i)\n    }\n    return r\n}\n\
         a = [1, 2]\nfor i to 3 {\n    a = append(a, count(a))\n}\nc = grown(a)\nshow a\nshow c\nu = upto(4)\nu = append(u, 4)\nshow u\n",
        "a: [1, 2, 2, 3, 4]\nc: [1, 2, 2, 3, 4, 0]\nu: [0, 1, 2, 3, 4]\n",
    );
}

#[test]
fn a_global_a_def_hands_out_is_copied() {
    prints(
        "global",
        "g = []\ndef snapshot() {\n    return g\n}\ndef total() {\n    s = 0\n    for i to len(g) {\n        s += g[i]\n    }\n    return s\n}\n\
         for i to 3 {\n    g = append(g, i + 1)\n}\ns = snapshot()\ng = append(g, 10)\ng[0] = -1\nshow s\nshow g\nshow total()\n",
        "s: [1, 2, 3]\ng: [-1, 2, 3, 10]\n14\n",
    );
}

#[test]
fn reads_and_writes_between_appends() {
    // the compiled engines cache an array's header; each append must refresh it
    prints(
        "rw",
        "def run(n) {\n    a = []\n    s = 0\n    for i to n {\n        a = append(a, i)\n        a[len(a) - 1] += 1\n        s += a[len(a) - 1] + a[0]\n    }\n    return s\n}\nshow run(100)\n\
         b = []\nfor i to 50 {\n    b = append(b, 2 * i)\n    b[0] = b[0] + b[len(b) - 1]\n}\nshow b[0]\nshow len(b)\n",
        "5150\n2450\n50\n",
    );
}

#[test]
fn many_doublings() {
    prints(
        "big",
        "big = []\nfor i to 100000 {\n    big = append(big, 3 * i + 1)\n}\nbad = 0\nfor i to len(big) {\n    if big[i] != 3 * i + 1 {\n        bad += 1\n    }\n}\nshow bad\nshow len(big)\n\
         m = []\nfor i to 2000 {\n    m = append(m, [i, -i, 2 * i])\n}\nshow shape(m)\nshow m[1999, 2]\n",
        "bad: 0\n100000\n[2000, 3]\n3998\n",
    );
}

#[test]
fn several_values_back() {
    prints(
        "tuple",
        "def split(n) {\n    ev = []\n    od = []\n    for i to n {\n        if mod(i, 2) == 0 {\n            ev = append(ev, i)\n        } else {\n            od = append(od, i)\n        }\n    }\n    return ev, od\n}\n\
         e, o = split(5)\ne = append(e, 100)\nshow e\nshow o\n",
        "e: [0, 2, 4, 100]\no: [1, 3]\n",
    );
}

#[test]
fn a_user_def_named_append_wins() {
    prints("own", "def append(a: Arr, x) {\n    return 7\n}\nb = append([1.0], 2)\nshow b\n", "b: 7\n");
}

#[test]
fn run_time_errors_are_located_in_every_engine() {
    let cases = [
        (
            "num",
            "m = zeros(2, 2)\nm = append(m, 3)\n",
            "error: `append`: a number goes on the end of a vector, and this array has rank 2\n  --> p.viv:2:5\n   2 | m = append(m, 3)\n           ^^^^^^^^^^^^ here\n",
        ),
        (
            "vec",
            "v = [1, 2]\nv = append(v, [3, 4])\n",
            "error: `append`: an array goes on as a new row of a matrix, and this is a vector of 2 numbers (join two vectors with `extend`)\n  --> p.viv:2:5\n   2 | v = append(v, [3, 4])\n           ^^^^^^^^^^^^^^^^^ here\n",
        ),
        (
            "cols",
            "m = zeros(2, 3)\nm = append(m, [1, 2])\n",
            "error: `append`: the row has 2 numbers, and the matrix has 3 columns\n  --> p.viv:2:5\n   2 | m = append(m, [1, 2])\n           ^^^^^^^^^^^^^^^^^ here\n",
        ),
        (
            "row",
            "m = []\nm = append(m, zeros(2, 2))\n",
            "error: `append`: a row must be a vector, and this array has rank 2\n  --> p.viv:2:5\n   2 | m = append(m, zeros(2, 2))\n           ^^^^^^^^^^^^^^^^^^^^^^ here\n",
        ),
        (
            "ext",
            "m = zeros(2, 2)\nm = extend(m, [1])\n",
            "error: `extend` joins vectors; this array has rank 2\n  --> p.viv:2:5\n   2 | m = extend(m, [1])\n           ^^^^^^^^^^^^^^ here\n",
        ),
        (
            "oob",
            "a = []\na = append(a, 1)\nshow a[1]\n",
            "error: index 1 out of bounds for dimension 0 (size 1)\n  --> p.viv:3:6\n   3 | show a[1]\n            ^^^^ here\n",
        ),
    ];
    for (tag, src, err) in cases {
        stops(tag, src, "", err);
    }
}

/// A program every engine refuses before it runs, with these words.
fn refused(tag: &str, src: &str, words: &str) {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let r = interp(&dir);
    assert_eq!(r.code, Some(3), "{tag}: {r:?}");
    assert!(r.err.contains(words), "{tag}: {}", r.err);
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(!c.status.success(), "{tag}: C accepted it");
    assert!(String::from_utf8_lossy(&c.stderr).contains(words), "{tag}: {}", String::from_utf8_lossy(&c.stderr));
}

#[test]
fn an_append_is_written_rebinding_its_array() {
    let form = "is written `a = append(a, x)`, with the same name on both sides";
    refused("other", "a = [1]\nb = append(a, 2)\n", form);
    refused("inside", "a = [1]\nshow len(append(a, 2))\n", form);
    refused("field", "record R {\n    xs: Arr\n}\nr = R { xs: [1] }\nr = append(r.xs, 2)\n", form);
    refused("arity", "a = [1]\na = append(a)\n", "`append` takes 2 arguments");
    refused("ext", "a = [1]\nc = extend(a, a)\n", "is written `a = extend(a, x)`");
    refused("par", "a = []\nparallel for i to 4 {\n    a = append(a, i)\n}\n", "cannot grow an array inside a `parallel for`");
    refused("cplx", "z = czeros(2)\nz = append(z, 1)\n", "`append` takes a real array first");
    refused("bool", "a = [1]\na = append(a, true)\n", "`append` adds a number, or a row");
}

#[test]
fn grad_through_append_and_extend_matches_central_differences() {
    let src = "def f(x: Arr) {\n    ys = []\n    for i to len(x) {\n        ys = append(ys, x[i] * x[i])\n        if x[i] > 0 {\n            ys = append(ys, 3 * x[i])\n        }\n    }\n    zs = [1, 2]\n    zs = extend(zs, ys)\n\
               m = []\n    m = append(m, [x[0], x[1]])\n    m = append(m, [x[1] * x[2], 1])\n    return sum(zs * zs) + sum(m * m)\n}\n\
               x0 = [0.3, -1.2, 0.7]\ng = grad(f, x0)\nh = 1.0e-6\nworst = 0.0\n\
               for i to len(x0) {\n    xp = copy(x0)\n    xm = copy(x0)\n    xp[i] += h\n    xm[i] -= h\n    worst = max(worst, abs(g[i] - (f(xp) - f(xm)) / (2.0 * h)))\n}\n\
               show worst < 1.0e-6\n";
    prints("grad", src, "true\n");
}

#[test]
fn the_compiler_grows_in_place_only_what_nothing_else_holds() {
    // `out` grows in place (1.0) and is sealed before `b = out` copies its
    // handle; `g` is handed out by a def, so its appends copy (0.0)
    let dir = scratch("marks");
    std::fs::write(
        dir.join("p.viv"),
        "g = []\ndef snap() {\n    return g\n}\nout = []\nfor i to 5 {\n    out = append(out, i)\n    g = append(g, i)\n}\nb = out\ns = snap()\n",
    )
    .unwrap();
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
    let code = std::fs::read_to_string(dir.join("p.c")).unwrap();
    for want in ["out = viv_append_n(out, i, 1.0,", "g = viv_append_n(g, i, 0.0,", "out = viv_seal(out);"] {
        assert!(code.contains(want), "no `{want}` in the C");
    }
}
