// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv export` end to end on a small program: the
//! manifest, the generated C built as a shared library and called from C
//! and from the generated Python wrapper.
//!
//! - every parameter kind (Num, Int, Bool, Str, Enum, Arr, IntArr) replaces
//!   its deck's value only when given, and each deck keeps its own values;
//! - results of every kind come back, arrays with the shape the run gave
//!   them, and a result a deck does not compute is absent;
//! - the program's test blocks come back as a table;
//! - a bad parameter is refused before any work, in C and in Python;
//! - an error inside the program returns its kind, Viv's message and the
//!   line in the program's own file, and the next run works;
//! - runs repeat exactly, and the memory in use after a run does not grow
//!   from one run to the next;
//! - Ctrl-C stops a silent loop through the wrapper, and the next run works;
//! - a molecule-shaped program (the PySCF port's interface, section 16):
//!   enums the program declares, chosen by published labels and spelled as
//!   PySCF spells a basis name; annotated parameter lines; a result whose
//!   rank follows a parameter, and is held to it; and the refusals of bad
//!   input (an unknown label, coordinates of the wrong shape, an atomic
//!   number out of range, atoms without coordinates, an impossible spin).

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn have(cmd: &str, arg: &str) -> bool {
    Command::new(cmd).arg(arg).output().map_or(false, |o| o.status.success())
}

/// A scratch repository (it holds a Cargo.toml) with the program in it.
fn scratch(tag: &str) -> PathBuf {
    scratch_with(tag, FILES)
}

/// A scratch repository with `files` in its `prog` directory.
fn scratch_with(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_export_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::create_dir_all(dir.join("prog")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    for (name, text) in files {
        std::fs::write(dir.join("prog").join(name), text).unwrap();
    }
    dir
}

const FILES: &[(&str, &str)] = &[
    (
        "toy.export.toml",
        r#"name = "toy"
about = "prog/toy.intent"
files = ["prog/engine.viv"]

[decks]
a = { deck = "prog/a.viv", checks = "prog/a_check.viv" }
b = { deck = "prog/b.viv" }

[params]
N = { type = "Int", min = 1, max = 1000 }
SCALE = { type = "Num", min = 0.0 }
LOUD = "Bool"
LABEL = "Str"
MODE = { type = "Enum", values = ["fast", "slow"] }
W = { type = "Arr", shape = [-1] }
IDS = { type = "IntArr", min = 0, max = 20 }
IX = { type = "Int", min = 0 }
LOOPS = { type = "Int", min = 0 }

[groups]
voice = ["LOUD", "LABEL"]

[results]
OUT = { type = "Num", doc = "the weighted total" }
COUNT = "Int"
NAME = "Str"
V = "Arr"
M = "Arr"
JS = "IntArr"
CHECKED = "Num"
"#,
    ),
    ("toy.intent", "module prog/engine\n  A toy program for the export tests: a weighted sum and some arrays.\n"),
    (
        "a.viv",
        "N = 3.0          #: N  how many entries\nSCALE = 2.0      #: SCALE  a factor\nLOUD = false\nLABEL = \"alpha\"\nMODE = \"fast\"\nW = [1.0, 2.0, 3.0]\nIDS = int_zeros(2)\nIX = 0.0\nLOOPS = 0.0\n",
    ),
    (
        "b.viv",
        "SCALE = 10.0\nN = 5.0\nLOUD = true\nLABEL = \"beta\"\nMODE = \"slow\"\nW = [1.0, 1.0]\nIDS = int_zeros(3)\nIX = 1.0\nLOOPS = 0.0\n",
    ),
    (
        "engine.viv",
        "def total(w, s) {\n    t = 0.0\n    for i to len(w) {\n        t = t + w[i] * s\n    }\n    return t\n}\nOUT = total(W, SCALE) + N\nV = zeros(N)\nfor i to N {\n    V[i] = i * SCALE\n}\nM = zeros(N, 2)\nM[0, 1] = OUT\nNAME = LABEL + \"!\"\nCOUNT = int(N)\nJS = IDS\nprint(\"out\", OUT)\nif LOUD {\n    print(\"loud\")\n}\nif MODE == \"slow\" {\n    print(\"slow\")\n}\nfor k to LOOPS {\n    OUT = OUT + 0.0 * total(V, 1.0)\n}\nshow V[IX]\n",
    ),
    ("a_check.viv", "CHECKED = 1.0\ntest \"the total is positive\" {\n    assert_gt(OUT, 0.0)\n}\ntest \"the scale is two\" {\n    assert_near(SCALE, 2.0, 0.0)\n}\n"),
];

/// `viv export` in `dir`: the generated directory.
fn export(dir: &Path) -> PathBuf {
    let out = dir.join("out");
    let r = Command::new(viv_bin()).current_dir(dir).args(["export", "prog/toy.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "viv export failed:\n{}", String::from_utf8_lossy(&r.stderr));
    out
}

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toy.h"
#if defined(__APPLE__)
#include <malloc/malloc.h>
static double in_use(void) { malloc_statistics_t s; malloc_zone_statistics(NULL, &s); return (double)s.size_in_use; }
#elif defined(__GLIBC__)
#include <malloc.h>
static double in_use(void) { struct mallinfo2 m = mallinfo2(); return (double)m.uordblks; }
#else
static double in_use(void) { return 0.0; }
#endif
static void line(const char* s, long n, int stream, void* ctx) { (void)n; printf("  %s%s\n", stream == 2 ? "[2] " : "", s); }
static int busy_seen = 0;
static void reenter(const char* s, long n, int stream, void* ctx) {
    (void)s; (void)n; (void)stream; (void)ctx;
    toy_params p; memset(&p, 0, sizeof p);
    toy_results r; char msg[64];
    if (toy_run(&p, &r, msg, sizeof msg) == TOY_BUSY) busy_seen = 1;
}
static void show(const char* what, toy_params* p) {
    toy_results r; char msg[1024];
    int code = toy_run(p, &r, msg, sizeof msg);
    printf("%s: code %d", what, code);
    if (code) printf(" kind %d msg <%s>\n", r.error_kind, msg);
    else {
        printf(" have %llu out %.17g count %lld name %s v[%lld] m[%lld,%lld] js[%lld]", (unsigned long long)r.have, r.out, (long long)r.count, r.name,
               (long long)r.v.dims[0], (long long)r.m.dims[0], (long long)r.m.dims[1], (long long)r.js.dims[0]);
        if (r.have & (1u << 6)) printf(" checked %g", r.checked);
        printf("\n");
        for (int i = 0; i < r.ntests; i++) printf("  test <%s> %d <%s>\n", r.tests[i].name, r.tests[i].passed, r.tests[i].detail);
    }
    toy_free(&r);
}
int main(void) {
    toy_params p; memset(&p, 0, sizeof p);
    p.on_line = line;
    show("deck a", &p);
    p.material = TOY_MATERIAL_B;
    show("deck b", &p);
    p.material = TOY_MATERIAL_A;
    double w[4] = {1, 1, 1, 1}; int64_t wd[1] = {4};
    int64_t ids[5] = {7, 8, 9, 10, 11}; int64_t idd[1] = {5};
    p.given = TOY_GIVEN_N | TOY_GIVEN_SCALE | TOY_GIVEN_LOUD | TOY_GIVEN_LABEL | TOY_GIVEN_MODE | TOY_GIVEN_W | TOY_GIVEN_IDS;
    p.n = 4; p.scale = 0.5; p.loud = 1; p.label = "given"; p.mode = TOY_MODE_SLOW;
    p.w.data = w; p.w.rank = 1; p.w.dims = wd;
    p.ids.data = ids; p.ids.rank = 1; p.ids.dims = idd;
    show("given", &p);
    p.given = TOY_GIVEN_N; p.n = 0;
    show("n below its min", &p);
    p.given = TOY_GIVEN_SCALE; p.scale = 1.0 / 0.0;
    show("an infinite scale", &p);
    int64_t wd2[2] = {2, 2};
    p.given = TOY_GIVEN_W; p.w.rank = 2; p.w.dims = wd2;
    show("w of rank 2", &p);
    p.given = TOY_GIVEN_LOUD; p.loud = 1;
    show("loud without its label", &p);
    int64_t big[2] = {3, 30}; int64_t bd[1] = {2};
    p.given = TOY_GIVEN_IDS; p.ids.data = big; p.ids.dims = bd;
    show("an id past its max", &p);
    p.given = TOY_GIVEN_IX; p.ix = 10;
    show("ix past the end", &p);
    p.given = 0; p.on_line = 0;
    show("after the error", &p);
    p.on_line = reenter;
    show("a run inside a run", &p);
    printf("busy %d\n", busy_seen);
    p.on_line = 0;
    double base = 0;
    for (int k = 0; k < 40; k++) {
        toy_results r; char msg[256];
        p.given = TOY_GIVEN_N; p.n = 200;
        toy_run(&p, &r, msg, sizeof msg);
        toy_free(&r);
        if (k == 9) base = in_use();
    }
    printf("growth %.0f bytes over 30 runs\n", in_use() - base);
    printf("abi %d version %s\n", toy_abi(), strncmp(toy_version(), "viv ", 4) == 0 ? "ok" : toy_version());
    return 0;
}
"#;

#[test]
fn an_exported_program_runs_from_c_with_its_parameters_results_and_errors() {
    if !have(&cc(), "--version") {
        return;
    }
    let dir = scratch("c");
    let out = export(&dir);
    for f in ["toy.c", "toy.h", "toy.py", "README.md", "test_toy.py"] {
        assert!(out.join(f).exists(), "{f} is written");
    }
    let c = std::fs::read_to_string(out.join("toy.c")).unwrap();
    assert!(!c.contains(&*dir.to_string_lossy()), "the C names no local path");
    std::fs::write(out.join("driver.c"), DRIVER).unwrap();
    let b = Command::new(cc()).current_dir(&out).args(["-O2", "-ffp-contract=off", "-w", "driver.c", "toy.c", "-lm", "-o", "driver"]).output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let r = Command::new(out.join("driver")).output().unwrap();
    let got = String::from_utf8_lossy(&r.stdout).into_owned();
    assert!(r.status.success() && r.stderr.is_empty(), "{got}\n{}", String::from_utf8_lossy(&r.stderr));
    let want_start = "  out 15\n  0\n  PASS the total is positive\n  PASS the scale is two\n\
                      deck a: code 0 have 127 out 15 count 3 name alpha! v[3] m[3,2] js[2] checked 1\n  \
                      test <the total is positive> 1 <>\n  test <the scale is two> 1 <>\n  out 25\n  loud\n  slow\n  10\n\
                      deck b: code 0 have 63 out 25 count 5 name beta! v[5] m[5,2] js[3]\n  out 6\n  loud\n  slow\n  0\n  \
                      PASS the total is positive\n  FAIL the scale is two -- assert_near: |0.5 - 2| = 1.5 > 0\n\
                      given: code 0 have 127 out 6 count 4 name given! v[4] m[4,2] js[5] checked 1\n  test <the total is positive> 1 <>\n  \
                      test <the scale is two> 0 <assert_near: |0.5 - 2| = 1.5 > 0>\n";
    assert!(got.starts_with(want_start), "{got}");
    for (what, want) in [
        ("n below its min", "code 3 kind 0 msg <n is 0; it must be at least 1>"),
        ("an infinite scale", "code 3 kind 0 msg <scale is inf; it must be a finite number>"),
        ("w of rank 2", "code 3 kind 0 msg <w has rank 2>"),
        ("loud without its label", "code 3 kind 0 msg <loud, label are given together or not at all (group voice)>"),
        ("an id past its max", "code 3 kind 0 msg <ids[1] (flat) is 30; each element must be in [0, 20]>"),
        ("after the error", "code 0 have 127 out 15 count 3 name alpha!"),
        ("a run inside a run", "code 0"),
    ] {
        assert!(got.contains(&format!("{what}: {want}")), "{what}: {got}");
    }
    assert!(got.contains("ix past the end: code 1 kind 2 msg <error: index 10 out of bounds for dimension 0 (size 3)\n  --> prog/engine.viv:28:6\n  28 | show V[IX]\n"), "{got}");
    assert!(got.contains("busy 1\n"), "{got}");
    let growth: f64 = got.lines().find_map(|l| l.strip_prefix("growth ")).and_then(|l| l.split(' ').next()).and_then(|v| v.parse().ok()).expect("a growth line");
    assert!(growth.abs() < 65536.0, "memory in use grew by {growth} bytes over 30 runs");
    assert!(got.contains("abi 1 version ok"), "{got}");
    let _ = std::fs::remove_dir_all(&dir);
}

const PY: &str = r#"
import os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
import inspect
import toy
assert list(inspect.signature(toy.run).parameters)[:4] == ["material", "n", "scale", "loud"], inspect.signature(toy.run)
assert "how many entries" in toy.run.__doc__ and "default: a 3, b 5" in toy.run.__doc__, toy.run.__doc__
r = toy.run(material="a")
assert r["out"] == 15.0 and r["count"] == 3 and r["name"] == "alpha!", r
assert r["v"].shape == (3,) and r["m"].shape == (3, 2) and r["js"].dtype.kind == "i", r
assert r["checked"] == 1.0 and [t[1] for t in r["tests"]] == [True, True], r
assert r["log"] == ["out 15", "0", "PASS the total is positive", "PASS the scale is two"], r["log"]
assert list(r["v"]) == [0.0, 2.0, 4.0] and r["m"][0, 1] == 15.0, r
b = toy.run(material="b")
assert b["checked"] is None and b["out"] == 25.0, b
g = toy.run(material="a", n=4, w=[1.0, 1.0, 1.0, 1.0], ids=[7, 8, 9], mode="slow", loud=True, label="given", scale=0.5)
assert g["out"] == 6.0 and g["name"] == "given!" and list(g["js"]) == [7, 8, 9] and "slow" in g["log"], g
for bad, exc in [(dict(n=1.5), ValueError), (dict(n="3"), TypeError), (dict(n=True), TypeError), (dict(n=0), ValueError),
                 (dict(mode="medium"), ValueError), (dict(scale=float("nan")), ValueError), (dict(w=[[1.0]]), ValueError),
                 (dict(nope=1), TypeError), (dict(material="z"), ValueError)]:
    try:
        toy.run(**bad)
    except exc:
        pass
    else:
        raise AssertionError("not refused: %r" % bad)
try:
    toy.run(material="a", ix=10)
except IndexError as e:
    assert isinstance(e, toy.VivError) and "prog/engine.viv:28:6" in str(e), e
else:
    raise AssertionError("no IndexError")
assert toy.run(material="a")["out"] == 15.0
# a process started in the background inherits SIGINT ignored, and Python
# then raises no KeyboardInterrupt: put the default handler back first
signal.signal(signal.SIGINT, signal.default_int_handler)
threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
t = time.time()
try:
    toy.run(material="a", n=1000, loops=10 ** 9)
except KeyboardInterrupt:
    took = time.time() - t
    assert took < 5.0, took
else:
    raise AssertionError("not interrupted")
assert toy.run(material="a")["out"] == 15.0
print("python ok")
"#;

#[test]
fn the_python_wrapper_runs_refuses_raises_and_stops_on_ctrl_c() {
    if !have(&cc(), "--version") || !Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success()) {
        return;
    }
    let dir = scratch("py");
    let out = export(&dir);
    std::fs::write(dir.join("t.py"), PY).unwrap();
    // the first import builds libtoy with the exact flags
    let r = Command::new("python3").arg(dir.join("t.py")).arg(&out).output().unwrap();
    let so = String::from_utf8_lossy(&r.stdout);
    assert!(r.status.success() && so.contains("python ok"), "{so}\n{}", String::from_utf8_lossy(&r.stderr));
    assert!(std::fs::read_dir(&out).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().starts_with("libtoy.")), "the library is named libtoy");
    // and the generated test file passes
    let r = Command::new("python3").arg(out.join("test_toy.py")).output().unwrap();
    assert!(r.status.success(), "{}\n{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_manifest_the_program_cannot_take_is_refused_with_the_line() {
    let dir = scratch("refuse");
    std::fs::write(dir.join("prog/b.viv"), "SCALE = 10.0\nN = 5.0\nN = 6.0\nLOUD = true\nLABEL = \"beta\"\nMODE = \"slow\"\nW = [1.0, 1.0]\nIDS = int_zeros(3)\nIX = 1.0\nLOOPS = 0.0\n").unwrap();
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/toy.export.toml", "-o", "out"]).output().unwrap();
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(!r.status.success() && err.contains("the parameter `N` is assigned more than once") && err.contains("--> prog/b.viv:3:1"), "{err}");
    // a parameter whose global has another type
    std::fs::write(dir.join("prog/b.viv"), "SCALE = 10.0\nN = 5.0\nLOUD = true\nLABEL = 3.0\nMODE = \"slow\"\nW = [1.0, 1.0]\nIDS = int_zeros(3)\nIX = 1.0\nLOOPS = 0.0\n").unwrap();
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/toy.export.toml", "-o", "out"]).output().unwrap();
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(!r.status.success(), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A GCC, if one is installed: `gcc-15` down to `gcc-11`, or a `gcc` that is
/// not clang under that name.
fn real_gcc() -> Option<String> {
    for c in ["gcc-15", "gcc-14", "gcc-13", "gcc-12", "gcc-11", "gcc"] {
        if let Ok(o) = Command::new(c).arg("--version").output() {
            let v = String::from_utf8_lossy(&o.stdout).to_lowercase();
            if o.status.success() && !v.contains("clang") {
                return Some(c.to_string());
            }
        }
    }
    None
}

/// B251 (bugs/ledger.jsonl): GCC's loop vectorizer turned the runtime's
/// complex products (viv_carr_had) into FCMLA, a fused multiply-add, on
/// aarch64 even under -ffp-contract=off: 121 of these 256 products differed
/// from the interpreter's. The exact build of a GCC prints the
/// interpreter's bytes. Skipped without a GCC.
#[test]
fn gcc_gives_the_interpreters_bits_for_complex_products() {
    let Some(gcc) = real_gcc() else { return };
    let src = "n = 256.0\na = czeros(n)\nb = czeros(n)\nfor k to n {\n    a[k] = sin(0.37 * k + 0.1) + cos(1.3 * k) * 1i\n    b[k] = cos(0.11 * k + 0.2) + sin(2.1 * k + 0.3) * 1i\n}\nc = a * b\nd = c * (0.3 - 1.1i)\nfor k to n {\n    print(fmt(re(c[k])), fmt(im(c[k])), fmt(re(d[k])), fmt(im(d[k])))\n}\n";
    let want = viv::interp::run(src).expect("interprets").lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect::<String>();
    let dir = std::env::temp_dir().join(format!("viv_export_gcc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), viv::compile_c(src).unwrap()).unwrap();
    // compiled with GCC; linked by GCC, or by cc where GCC's link fails
    // (Homebrew GCC with some macOS linkers)
    let built = Command::new(&gcc).current_dir(&dir).args(["-O2", "-ffp-contract=off", "-w", "-c", "p.c", "-o", "p.o"]).output().unwrap();
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let linked = Command::new(&gcc).current_dir(&dir).args(["p.o", "-o", "p", "-lm"]).output().unwrap();
    if !linked.status.success() {
        let l = Command::new(cc()).current_dir(&dir).args(["p.o", "-o", "p", "-lm"]).output().unwrap();
        assert!(l.status.success(), "{}", String::from_utf8_lossy(&l.stderr));
    }
    let out = Command::new(dir.join("p")).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let got = String::from_utf8_lossy(&out.stdout);
    let differ = got.lines().zip(want.lines()).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "{differ} of 256 products differ from the interpreter's under {gcc}");
    assert_eq!(got, want);
}

// ---- a molecule-shaped program: the PySCF port's interface (section 16) ----

const MOL: &[(&str, &str)] = &[
    (
        "mol.export.toml",
        r#"name = "mol"
about = "prog/mol.viv"
files = ["prog/mol.viv"]
cases = "prog/mol.cases.json"

[params]
atoms = { type = "[Int]", min = 1, max = 18 }
coords = { type = "Arr", shape = ["len(atoms)", 3] }
basis = { type = "BasisSet", labels = { "sto-3g" = "Sto3g", "6-31g*" = "Pople631gs" }, normalize = "pyscf_basis" }
method = { type = "Method", labels = { "rhf" = "Rhf", "uhf" = "Uhf" } }
charge = { type = "Int", min = -18, max = 18 }
spin = { type = "Int", min = 0 }

[groups]
geometry = ["atoms", "coords"]

[results]
e = "Num"
eps = { type = "Arr", rank = { by = "method", rhf = 1, uhf = 2 } }

[reference]
program = "nothing: a toy"
"#,
    ),
    ("mol.cases.json", "{\"water\": {}, \"h2_uhf\": {\"atoms\": [1, 1], \"coords\": [[0, 0, 0], [0, 0, 0.74]], \"method\": \"uhf\", \"basis\": \"6-31G*\"}}\n"),
    (
        "mol.viv",
        r#"###
post: A molecule-shaped toy for the export tests
agent: claude

A fake energy from the atoms, the basis and the method, with the PySCF port's parameters.
###

enum BasisSet {
    Sto3g
    Pople631gs
}

enum Method {
    Rhf
    Uhf
}

#: atoms   atomic numbers, one per atom
#: coords  positions, angstrom
#: basis   the basis set
#: method  restricted or unrestricted
atoms: [Int] = [8, 1, 1]
coords = [[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]]
basis = BasisSet.Sto3g
method = Method.Rhf
charge: Int = 0
spin: Int = 0

def nelectron(z: [Int], q: Int) -> Int {
    n: Int = -q
    for a in z {
        n += a
    }
    return n
}

ne = nelectron(atoms, charge)
if spin > ne or (ne + spin) % 2 != 0 {
    stop(f"mol: {ne:d} electrons cannot have spin {spin:d}")
}
scale = 1.0
if basis == BasisSet.Pople631gs {
    scale = 2.0
}
#: e    the fake energy
e = -1.0 * float(ne) * scale - sum(coords * coords)
natm = len(atoms)
if method == Method.Rhf {
    eps = zeros(natm)
}
else {
    eps = zeros(2, natm)
}
print("energy", e)
"#,
    ),
];

const MOL_DRIVER: &str = r#"
#include <stdio.h>
#include <string.h>
#include "mol.h"
static void show(const char* what, mol_params* p) {
    mol_results r; char msg[1024];
    int code = mol_run(p, &r, msg, sizeof msg);
    printf("%s: code %d", what, code);
    if (code) printf(" kind %d msg <%s>\n", r.error_kind, msg);
    else printf(" e %g eps rank %lld\n", r.e, (long long)r.eps.rank);
    mol_free(&r);
}
int main(void) {
    mol_params p; memset(&p, 0, sizeof p);
    show("water", &p);
    p.given = MOL_GIVEN_BASIS; p.basis = MOL_BASIS_6_31G_STAR;
    show("6-31g*", &p);
    p.given = MOL_GIVEN_METHOD | MOL_GIVEN_CHARGE | MOL_GIVEN_SPIN; p.method = MOL_METHOD_UHF; p.charge = 1; p.spin = 1;
    show("uhf cation", &p);
    int64_t z[2] = {1, 1}; int64_t zd[1] = {2};
    double xyz[6] = {0, 0, 0, 0, 0, 0.5}; int64_t xd[2] = {2, 3};
    p.given = MOL_GIVEN_ATOMS | MOL_GIVEN_COORDS;
    p.atoms.data = z; p.atoms.rank = 1; p.atoms.dims = zd;
    p.coords.data = xyz; p.coords.rank = 2; p.coords.dims = xd;
    show("h2", &p);
    p.given = MOL_GIVEN_BASIS; p.basis = 7;
    show("a basis past the labels", &p);
    int64_t xd3[2] = {3, 3}; double xyz3[9] = {0};
    p.given = MOL_GIVEN_ATOMS | MOL_GIVEN_COORDS; p.coords.dims = xd3; p.coords.data = xyz3;
    show("coords of three atoms for two", &p);
    int64_t xd2[2] = {2, 2};
    p.coords.dims = xd2; p.coords.data = xyz;
    show("coords of two columns", &p);
    int64_t zbad[2] = {1, 19};
    p.coords.dims = xd; p.atoms.data = zbad;
    show("an atomic number past argon", &p);
    p.atoms.data = z;
    p.given = MOL_GIVEN_ATOMS;
    show("atoms without coords", &p);
    p.given = MOL_GIVEN_SPIN; p.spin = 1;
    show("water with spin 1", &p);
    p.given = MOL_GIVEN_SPIN; p.spin = 12;
    show("water with spin 12", &p);
    p.given = 0;
    show("after the refusals", &p);
    return 0;
}
"#;

#[test]
fn a_molecule_program_takes_enums_by_label_annotated_lines_and_a_rank_by_method() {
    if !have(&cc(), "--version") {
        return;
    }
    let dir = scratch_with("mol_c", MOL);
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/mol.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "viv export failed:\n{}", String::from_utf8_lossy(&r.stderr));
    let out = dir.join("out");
    let h = std::fs::read_to_string(out.join("mol.h")).unwrap();
    for want in ["#define MOL_BASIS_STO_3G 0\n", "#define MOL_BASIS_6_31G_STAR 1\n", "#define MOL_METHOD_UHF 1\n", "sto-3g = BasisSet.Sto3g", "rank 1 when method is rhf, rank 2 when method is uhf", "atomic numbers, one per atom", "the fake energy"] {
        assert!(h.contains(want), "{want:?} in the header:\n{h}");
    }
    assert!(out.join("cases.json").exists());
    let readme = std::fs::read_to_string(out.join("README.md")).unwrap();
    assert!(readme.contains("- program: nothing: a toy") && readme.contains("A fake energy from the atoms"), "{readme}");
    std::fs::write(out.join("driver.c"), MOL_DRIVER).unwrap();
    let b = Command::new(cc()).current_dir(&out).args(["-O2", "-ffp-contract=off", "-w", "driver.c", "mol.c", "-lm", "-o", "driver"]).output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let r = Command::new(out.join("driver")).output().unwrap();
    let got = String::from_utf8_lossy(&r.stdout).into_owned();
    assert!(r.status.success(), "{got}\n{}", String::from_utf8_lossy(&r.stderr));
    for want in [
        "water: code 0 e -11.8352 eps rank 1\n",
        "6-31g*: code 0 e -21.8352 eps rank 1\n",
        "uhf cation: code 0 e -10.8352 eps rank 2\n",
        "h2: code 0 e -2.25 eps rank 1\n",
        "a basis past the labels: code 3 kind 0 msg <basis is 7; there are 2 choices>\n",
        "coords of three atoms for two: code 3 kind 0 msg <coords has 3 in dimension 0; it must have len(atoms) = 2>\n",
        "coords of two columns: code 3 kind 0 msg <coords has 2 in dimension 1; it must have 3>\n",
        "an atomic number past argon: code 3 kind 0 msg <atoms[1] (flat) is 19; each element must be in [1, 18]>\n",
        "atoms without coords: code 3 kind 0 msg <atoms, coords are given together or not at all (group geometry)>\n",
        "after the refusals: code 0 e -11.8352 eps rank 1\n",
    ] {
        assert!(got.contains(want), "{want:?}:\n{got}");
    }
    // the program's own stop: an odd spin for ten electrons, and more
    // unpaired electrons than electrons
    for what in ["water with spin 1", "water with spin 12"] {
        let line = got.lines().find(|l| l.starts_with(&format!("{what}: "))).unwrap_or_default();
        assert!(line.starts_with(&format!("{what}: code 1 kind ")) && line.contains("electrons cannot have spin"), "{line}\n{got}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const MOL_PY: &str = r#"
import inspect, sys
sys.path.insert(0, sys.argv[1])
import mol
assert list(inspect.signature(mol.run).parameters)[:6] == ["atoms", "coords", "basis", "method", "charge", "spin"], inspect.signature(mol.run)
doc = mol.run.__doc__
for want in ["one of sto-3g, 6-31g*", "default: \"sto-3g\"", "default: \"rhf\"", "atomic numbers, one per atom", "rank 1 when method is rhf", "spelled as PySCF reads a basis name"]:
    assert want in doc, (want, doc)
a = mol.run()
b = mol.run(basis="6-31G*")
c = mol.run(basis="631gs")
assert b["e"] == c["e"] == a["e"] - 10.0, (a, b, c)
u = mol.run(method="uhf", charge=1, spin=1)
assert u["eps"].shape == (2, 3) and a["eps"].shape == (3,), (u, a)
for bad, exc in [(dict(basis="6-31g**"), ValueError), (dict(basis=3), TypeError), (dict(method="RHF"), ValueError),
                 (dict(atoms=[8, 19, 1], coords=[[0, 0, 0]] * 3), ValueError), (dict(atoms=[8, 1, 1]), ValueError),
                 (dict(atoms=[1, 1], coords=[[0, 0, 0]] * 3), ValueError), (dict(spin=1), mol.VivError)]:
    try:
        mol.run(**bad)
    except exc as e:
        if exc is mol.VivError:
            assert "cannot have spin" in str(e), e
    else:
        raise AssertionError("not refused: %r" % bad)
print("python ok")
"#;

#[test]
fn the_molecule_wrapper_reads_labels_as_pyscf_spells_them_and_refuses_bad_input() {
    if !have(&cc(), "--version") || !Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success()) {
        return;
    }
    let dir = scratch_with("mol_py", MOL);
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/mol.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "viv export failed:\n{}", String::from_utf8_lossy(&r.stderr));
    let out = dir.join("out");
    std::fs::write(dir.join("t.py"), MOL_PY).unwrap();
    let r = Command::new("python3").arg(dir.join("t.py")).arg(&out).output().unwrap();
    let so = String::from_utf8_lossy(&r.stdout);
    assert!(r.status.success() && so.contains("python ok"), "{so}\n{}", String::from_utf8_lossy(&r.stderr));
    // the generated test runs every case of cases.json
    let r = Command::new("python3").arg(out.join("test_mol.py")).output().unwrap();
    let so = String::from_utf8_lossy(&r.stdout);
    assert!(r.status.success() && so.contains("ok water") && so.contains("ok h2_uhf"), "{so}\n{}", String::from_utf8_lossy(&r.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A result of another rank than the manifest states for the run's method
/// is an error, not a silent change of shape.
#[test]
fn a_result_of_another_rank_than_the_manifest_says_is_an_error() {
    if !have(&cc(), "--version") {
        return;
    }
    let toml = MOL[0].1.replace("rank = { by = \"method\", rhf = 1, uhf = 2 }", "rank = { by = \"method\", rhf = 2, uhf = 2 }");
    let files: Vec<(&str, &str)> = vec![("mol.export.toml", &toml), MOL[1], MOL[2]];
    let dir = scratch_with("mol_rank", &files);
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/mol.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "viv export failed:\n{}", String::from_utf8_lossy(&r.stderr));
    let out = dir.join("out");
    std::fs::write(out.join("driver.c"), MOL_DRIVER).unwrap();
    let b = Command::new(cc()).current_dir(&out).args(["-O2", "-ffp-contract=off", "-w", "driver.c", "mol.c", "-lm", "-o", "driver"]).output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let got = String::from_utf8_lossy(&Command::new(out.join("driver")).output().unwrap().stdout).into_owned();
    assert!(got.contains("water: code 1 kind 1 msg <error: the result eps has rank 1; the manifest says 2 when method is rhf>"), "{got}");
    assert!(got.contains("uhf cation: code 0 e -10.8352 eps rank 2"), "{got}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_enum_parameter_the_program_cannot_take_is_refused() {
    for (from, to, why) in [
        ("\"6-31g*\" = \"Pople631gs\"", "\"6-31g*\" = \"Pople631gss\"", "the label `6-31g*` names `BasisSet.Pople631gss`, which is not a variant"),
        (", \"6-31g*\" = \"Pople631gs\"", "", "the variant `BasisSet.Pople631gs` has no label in the manifest"),
        ("type = \"BasisSet\"", "type = \"BasisSets\"", "the program declares no enum `BasisSets`"),
        ("basis = { type = \"BasisSet\", labels = { \"sto-3g\" = \"Sto3g\", \"6-31g*\" = \"Pople631gs\" }, normalize = \"pyscf_basis\" }", "basis = \"BasisSet\"", "takes `labels`"),
        ("rhf = 1, uhf = 2", "rhf = 1", "names no rank for method = uhf"),
        ("by = \"method\"", "by = \"charge\"", "not an enum parameter"),
        ("normalize = \"pyscf_basis\"", "normalize = \"pyscf\"", "normalize (lower, pyscf_basis)"),
    ] {
        let toml = MOL[0].1.replace(from, to);
        assert_ne!(toml, MOL[0].1, "{from}");
        let files: Vec<(&str, &str)> = vec![("mol.export.toml", &toml), MOL[1], MOL[2]];
        let dir = scratch_with("mol_refuse", &files);
        let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/mol.export.toml", "-o", "out"]).output().unwrap();
        let err = String::from_utf8_lossy(&r.stderr);
        assert!(!r.status.success() && err.contains(why), "{to}: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A program that stops with each kind of error (crate::errkind), chosen by
/// a parameter: the wrapper raises that kind's exception, and it is the
/// kind the interpreter reports for the same program and deck.
const KINDS_ENGINE: &str = "def deep(n) {\n    return deep(n + 1.0)\n}\nOUT = 0.0\na = zeros(3)\n\
if WHICH == 1.0 {\n    OUT = a[WHICH + 4.0]\n}\n\
if WHICH == 2.0 {\n    b = zeros(WHICH)\n    c = a + b\n    OUT = c[0]\n}\n\
if WHICH == 3.0 {\n    big = zeros(100000.0 * WHICH, 100000.0)\n    OUT = big[0, 0]\n}\n\
if WHICH == 4.0 {\n    OUT = 5.0 % (WHICH - 4.0)\n}\n\
if WHICH == 5.0 {\n    OUT = deep(0.0)\n}\n\
if WHICH == 6.0 {\n    stop(\"the deck is empty\")\n}\n\
if WHICH == 7.0 {\n    m = map()\n    m[1] = 2\n    OUT = m[WHICH]\n}\n\
show OUT\n";

const KINDS_PY: &str = r#"
import sys
sys.path.insert(0, sys.argv[1])
import kinds
assert kinds.run(material="a")["out"] == 0.0
want = {1: ("VivIndexError", IndexError), 2: ("VivValueError", ValueError), 3: ("VivOverflowError", OverflowError),
        4: ("VivZeroDivisionError", ZeroDivisionError), 5: ("VivRecursionError", RecursionError),
        6: ("VivStop", kinds.VivError), 7: ("VivError", kinds.VivError)}
for which, (name, base) in sorted(want.items()):
    try:
        kinds.run(material="a", which=which)
    except kinds.VivError as e:
        assert type(e).__name__ == name and isinstance(e, base), (which, type(e), e)
        assert str(e).startswith("error: "), (which, e)
        print(which, name)
    else:
        raise AssertionError("no error for %d" % which)
assert kinds.run(material="a")["out"] == 0.0
print("kinds ok")
"#;

#[test]
fn each_error_kind_raises_its_exception_and_is_the_interpreters_kind() {
    let have_py = Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success());
    let dir = std::env::temp_dir().join(format!("viv_export_{}_kinds", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("prog")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    let deck = "WHICH = 0.0      #: WHICH  which error to stop with (0: none)\n";
    for (name, text) in [
        ("kinds.export.toml", "name = \"kinds\"\nabout = \"prog/kinds.intent\"\nfiles = [\"prog/fail.viv\"]\n\n[decks]\na = { deck = \"prog/a.viv\" }\n\n[params]\nWHICH = { type = \"Int\", min = 0, max = 9 }\n\n[results]\nOUT = \"Num\"\n"),
        ("kinds.intent", "module prog/fail\n  Stops with the error kind its parameter names.\n"),
        ("a.viv", deck),
        ("fail.viv", KINDS_ENGINE),
    ] {
        std::fs::write(dir.join("prog").join(name), text).unwrap();
    }
    // the interpreter, on the deck with each WHICH
    use viv::errkind::ErrKind;
    let order = [ErrKind::Index, ErrKind::Value, ErrKind::Overflow, ErrKind::ZeroDiv, ErrKind::Recursion, ErrKind::Stop, ErrKind::Error];
    let mut names = Vec::new();
    for (i, want) in order.iter().enumerate() {
        let src = format!("WHICH = {}.0\n{KINDS_ENGINE}", i + 1);
        let run = std::thread::Builder::new().stack_size(512 << 20).spawn(move || viv::interp::run_diagnosed(&src, false)).unwrap().join().unwrap();
        assert_eq!(run.kind, Some(*want), "WHICH = {}: {:?}", i + 1, run.errors.first().map(|d| d.message.clone()));
        names.push(format!("{} {}", i + 1, want.python_class().unwrap()));
    }
    if !have(&cc(), "--version") || !have_py {
        return;
    }
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/kinds.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "viv export failed:\n{}", String::from_utf8_lossy(&r.stderr));
    std::fs::write(dir.join("t.py"), KINDS_PY).unwrap();
    let r = Command::new("python3").arg(dir.join("t.py")).arg(dir.join("out")).output().unwrap();
    let so = String::from_utf8_lossy(&r.stdout);
    assert!(r.status.success() && so.contains("kinds ok"), "{so}\n{}", String::from_utf8_lossy(&r.stderr));
    // the wrapper's class for each is the interpreter's kind's
    for n in names {
        assert!(so.contains(&format!("{n}\n")), "{n}: {so}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A program that imports a module: the module's def has a local named like
/// a parameter (B270, the same cause as B262), its `###` text names another
/// parameter (B272, fixed with B263), and a refusal in its text is shown at
/// the module's own file and line (B273). An exclusive minimum is refused
/// at its bound in Python and in C.
#[test]
fn an_imported_modules_locals_and_text_do_not_refuse_parameters_and_its_errors_name_it() {
    let module = "###\nThis module's MAX_ITER and H are the caller's.\n###\n\ndef g(x: Num, b: Num = 2.0) -> Num {\n    H = x * 3.0\n    return H + b\n}\n";
    let prog = "from mm import g\n\n#: H  the spacing\nH = 0.5\nMAX_ITER = 10\nOUT = g(H, b=MAX_ITER)\nOK = OUT > 1.0\n";
    let toml = "name = \"imported\"\nfiles = [\"prog/prog.viv\"]\n[params]\nH = { type = \"Num\", min_exclusive = 0.0, max = 1.0 }\nMAX_ITER = \"Int\"\n[results]\nOUT = \"Num\"\nOK = \"Bool\"\n";
    let dir = scratch_with("imported", &[("mm.viv", module), ("prog.viv", prog), ("imp.export.toml", toml)]);
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/imp.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let c = std::fs::read_to_string(dir.join("out/imported.c")).unwrap();
    assert!(c.contains("if (!(p->h > 0.0)) return imported_bad(msg, cap, \"h is %.17g; it must be above 0\""), "the C refuses h = 0");
    let py = std::fs::read_to_string(dir.join("out/imported.py")).unwrap();
    assert!(py.contains("h (float, above 0 and at most 1): the spacing"), "the docstring: {py}");
    if Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success()) {
        let script = "import sys\nsys.path.insert(0, sys.argv[1])\nimport imported as imp\nassert imp.run()['out'] == 11.5\nassert imp.run(h=0.25)['ok']\nfor bad in (0.0, -1.0, 1.5):\n    try:\n        imp.run(h=bad)\n    except ValueError as e:\n        print('refused', bad, e)\n    else:\n        raise AssertionError(bad)\nprint('python ok')\n";
        let r = Command::new("python3").args(["-c", script, dir.join("out").to_str().unwrap()]).output().unwrap();
        let so = String::from_utf8_lossy(&r.stdout);
        assert!(r.status.success() && so.contains("refused 0.0 h is 0.0; it must be above 0.0") && so.contains("python ok"), "{so}{}", String::from_utf8_lossy(&r.stderr));
    }
    // B273: H assigned only inside the module's def: the error is at mm.viv's line 6
    std::fs::write(dir.join("prog/prog.viv"), "from mm import g\n\nMAX_ITER = 10\nOUT = g(1.0, b=MAX_ITER)\nOK = OUT > 1.0\n").unwrap();
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/imp.export.toml", "-o", "out"]).output().unwrap();
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(!r.status.success() && err.contains("the parameter `H` is assigned only inside `g`") && err.contains("--> prog/mm.viv:6:5"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A kernel collection: a module whose
/// constant is computed at its top level (`TAB = make_table(4.0)`, run once
/// at init), complex arguments and results, a def with two results (a tuple
/// in Python), an Int argument with a range, an array with an exclusive
/// minimum, a square matrix; NaN refused in Python and in C.
const KERN: &[(&str, &str)] = &[
    ("tab.viv", "def make_table(n: Num) -> Arr {\n    t = zeros(n)\n    for i to n {\n        t[i] = (i + 1.0) * 0.5\n    }\n    return t\n}\nTAB = make_table(4.0)\n"),
    (
        "kern.viv",
        "import tab\n\ndef scale(z: Cplx, k: Int) {\n    return z * tab.TAB[k], abs(z) * tab.TAB[k]\n}\n\ndef total(x: Arr) -> Num {\n    return sum(x) + tab.TAB[0]\n}\n\ndef sq(A: Arr) -> Arr {\n    return A @ A\n}\n\ndef fails(x: Num) -> Num {\n    if x > 1.0 {\n        stop(\"x is too big\")\n    }\n    return x\n}\n",
    ),
    (
        "kx.export.toml",
        r#"name = "kx"
kind = "kernels"
prefix = "kx_"
files = ["prog/kern.viv"]

[kernels.scale]
viv = "prog/kern.viv:scale"
args = [{ name = "z", type = "Cplx" }, { name = "k", type = "Int", min = 0, max = 3 }]
results = [{ name = "w", type = "Complex" }, { name = "m", type = "Num" }]
reference = "z times the table's entry k"

[kernels.total]
viv = "prog/kern.viv:total"
args = [{ name = "x", type = "Arr", rank = 1, min_exclusive = 0.0 }]
results = [{ name = "t", type = "Num" }]

[kernels.sq]
viv = "prog/kern.viv:sq"
args = [{ name = "A", type = "Arr", square = true }]
results = [{ name = "B", type = "Arr" }]

[kernels.fails]
viv = "prog/kern.viv:fails"
args = [{ name = "x", type = "Num" }]
results = [{ name = "y", type = "Num" }]
"#,
    ),
];

const KERN_PY: &str = r#"
import ctypes, math, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import kx
w, m = kx.scale(1 + 2j, 2)
assert w == (1 + 2j) * 1.5 and m == abs(1 + 2j) * 1.5, (w, m)
assert kx.total(np.array([1.0, 2.0])) == 3.5
B = kx.sq(np.array([[1.0, 2.0], [3.0, 4.0]]))
assert B.shape == (2, 2) and B.tolist() == [[7.0, 10.0], [15.0, 22.0]], B
for call, why in [(lambda: kx.scale(complex(math.nan, 0.0), 0), "finite"), (lambda: kx.scale(1j, 4), "at most 3"),
                  (lambda: kx.total(np.array([1.0, 0.0])), "above 0"), (lambda: kx.sq(np.ones((2, 3))), "square"),
                  (lambda: kx.total(np.ones((2, 2))), "rank")]:
    try:
        call()
    except ValueError as e:
        assert why in str(e), (why, e)
    else:
        raise AssertionError(why)
# the C entry refuses a NaN by itself, before any work
A, R, f, free = kx._entries["scale"]
a, r = A(), R()
a.z = (ctypes.c_double * 2)(math.nan, 0.0)
a.k = 0
msg = ctypes.create_string_buffer(256)
assert f(ctypes.byref(a), ctypes.byref(r), msg, 256) == 3 and b"finite" in msg.value, msg.value
# an error inside a kernel raises its kind, and the next call works
try:
    kx.fails(2.0)
except kx.VivStop as e:
    assert "x is too big" in str(e) and "prog/kern.viv:" in str(e), e
else:
    raise AssertionError("no VivStop")
assert kx.fails(0.5) == 0.5
assert kx.scale(1 + 2j, 2)[0] == (1 + 2j) * 1.5
assert "scale(z, k) -> w, m" in kx.scale.__doc__ and "z times the table" in kx.scale.__doc__
print("kernels ok")
"#;

#[test]
fn a_kernel_collection_exports_each_def_with_complex_tuples_and_an_init() {
    let dir = scratch_with("kern", KERN);
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/kx.export.toml", "-o", "out"]).output().unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let h = std::fs::read_to_string(dir.join("out/kx.h")).unwrap();
    assert!(h.contains("typedef struct { double _Complex z; int64_t k; } kx_scale_args;") && h.contains("typedef struct { double _Complex w; double m; int error_kind; } kx_scale_results;"), "{h}");
    assert!(h.contains("int kx_init(kx_line_fn on_line, void* ctx, char* msg, size_t cap);"), "{h}");
    if !have(&cc(), "--version") {
        return;
    }
    let lib = dir.join("out/libkx.so");
    let b = Command::new(cc()).args(["-O2", "-ffp-contract=off", "-fvisibility=hidden", "-shared", "-fPIC", "-Wall"]).arg(dir.join("out/kx.c")).arg("-o").arg(&lib).arg("-lm").output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    if Command::new("python3").args(["-c", "import numpy"]).output().map_or(false, |o| o.status.success()) {
        let r = Command::new("python3").args(["-c", KERN_PY, dir.join("out").to_str().unwrap()]).env("KX_LIB", &lib).output().unwrap();
        let so = String::from_utf8_lossy(&r.stdout);
        assert!(r.status.success() && so.contains("kernels ok"), "{so}{}", String::from_utf8_lossy(&r.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_kernel_that_does_not_agree_with_its_def_is_refused() {
    for (from, to, why) in [
        ("{ name = \"k\", type = \"Int\", min = 0, max = 3 }", "{ name = \"j\", type = \"Int\" }", "the manifest's arguments are (z, j), and the def's parameters are (z, k)"),
        ("{ name = \"k\", type = \"Int\", min = 0, max = 3 }", "{ name = \"k\", type = \"Num\" }", "the def's `k` is Int, and the manifest says Num"),
        ("results = [{ name = \"w\", type = \"Complex\" }, { name = \"m\", type = \"Num\" }]", "results = [{ name = \"w\", type = \"Complex\" }]", "the def gives 2 results, and the manifest names 1"),
        ("results = [{ name = \"t\", type = \"Num\" }]", "results = [{ name = \"t\", type = \"Arr\" }]", "the result `t` is Num in the def, and the manifest says Arr"),
        ("viv = \"prog/kern.viv:sq\"", "viv = \"prog/kern.viv:square\"", "prog/kern.viv defines no `square`"),
        ("[kernels.sq]", "[kernels.sq]\nwrites = [\"A\"]", "`writes` (arguments a kernel writes in place) is not built yet"),
    ] {
        let toml = KERN[2].1.replace(from, to);
        assert_ne!(toml, KERN[2].1, "{from}");
        let files: Vec<(&str, &str)> = vec![KERN[0], KERN[1], ("kx.export.toml", &toml)];
        let dir = scratch_with("kern_refuse", &files);
        let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/kx.export.toml", "-o", "out"]).output().unwrap();
        let err = String::from_utf8_lossy(&r.stderr);
        assert!(!r.status.success() && err.contains(why), "{to}: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
    // a parameter with no annotation: the def must say what the entry takes
    let kern = KERN[1].1.replace("def total(x: Arr) -> Num", "def total(x) -> Num");
    let files: Vec<(&str, &str)> = vec![KERN[0], ("kern.viv", &kern), KERN[2]];
    let dir = scratch_with("kern_annot", &files);
    let r = Command::new(viv_bin()).current_dir(&dir).args(["export", "prog/kx.export.toml", "-o", "out"]).output().unwrap();
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(!r.status.success() && err.contains("the parameter `x` has no type; annotate it (`x: Arr`)"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
