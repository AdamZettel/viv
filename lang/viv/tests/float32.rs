// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Float32 and the packed arrays, checked in all three
//! engines: the interpreter (`viv run`), Viv -> C -> native, and the direct
//! wasm backend run in node (skipped when node or a viv.wasm with the
//! runtime exports is missing; VIV_WASM=<path> names one).

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
    // under the target directory, which is inside the repository, so `#! use
    // lib/random` finds the repository root as it does for a program there
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_float32_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// What one engine did: exit code, stdout, stderr.
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

/// A viv.wasm with the runtime exports, and node, or None.
fn wasm_runtime() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    Some(wasm)
}

/// The three engines on `src`: (interpreter, native, direct wasm if available).
fn engines(tag: &str, src: &str) -> (Ran, Ran, Option<Ran>) {
    let dir = scratch(tag);
    let p = dir.join("p.viv");
    std::fs::write(&p, src).unwrap();
    // by absolute path, so `#! use` finds the repository root from the file (a
    // bare `p.viv` does not yet: gap R14); the diagnostics
    // still name it `p.viv`
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q"]).arg(&p).output().unwrap());
    let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{tag}: viv refused the program for C:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "{tag}: the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    let n = ran(Command::new(dir.join("p")).output().unwrap());
    let w = wasm_runtime().map(|wasm| {
        let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.wasm"]).output().unwrap();
        assert!(c.status.success(), "{tag}: viv refused the program for wasm:\n{}", String::from_utf8_lossy(&c.stderr));
        ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap())
    });
    (i, n, w)
}

/// Run `src` in every engine; each must exit 0 and print exactly `want`.
fn prints(tag: &str, src: &str, want: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), want, ""), "{tag}: interpreter");
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "{tag}: direct wasm\n{}", w.err);
    }
}

/// Run `src` in every engine; each must stop with exit 4 (a run-time error), print `out` first
/// and report an error whose first line is `error: <msg>`. The interpreter
/// and native C must agree byte for byte, the location included. The wasm
/// module, run by tools/run-wasm.mjs, prints the same output and then the message
/// without a location, as it does for an index out of bounds.
fn stops(tag: &str, src: &str, out: &str, msg: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!(i.code, Some(4), "{tag}: interpreter {i:?}");
    assert_eq!(i.out, out, "{tag}");
    assert!(i.err.starts_with(&format!("error: {msg}\n  --> p.viv:")), "{tag}: {}", i.err);
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str(), w.err.as_str()), (Some(4), out, format!("error: {msg}\n").as_str()), "{tag}: direct wasm");
    }
}

/// The compiler refuses `src` with a message containing `msg`, and so does the
/// interpreter (at run time).
fn refused(tag: &str, src: &str, msg: &str) {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let c = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap());
    assert_ne!(c.code, Some(0), "{tag}: the compiler accepted it");
    assert!(c.err.contains(msg), "{tag}: compiler said\n{}", c.err);
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap());
    assert_ne!(i.code, Some(0), "{tag}: the interpreter ran it");
    assert!(i.err.contains(msg), "{tag}: interpreter said\n{}", i.err);
}



/// The generated cases (tools/float32/cases.py): every line against NumPy
/// float32, mpmath's correctly rounded float32 functions, NumPy's float16
/// rounding and a bfloat16 rounding written from the IEEE definition.
#[test]
fn cases_match_their_references_in_every_engine() {
    let dir = root().join("lang/viv/tests/float32");
    let src = std::fs::read_to_string(dir.join("cases.viv")).unwrap();
    let want = std::fs::read_to_string(dir.join("cases.txt")).unwrap();
    prints("cases", &src, &want);
}

#[test]
fn literals_take_the_float32_side() {
    prints("lits", "x: Float32 = 0.1\ny = x * 3 + 1\nprint(fmt(x), fmt(y), y)\nprint(fmt(float(x)))\n", "0.1 1.3 1.3\n0.10000000149011612\n");
}

#[test]
fn packed_arrays_print_and_convert() {
    prints(
        "packed",
        "w = float32([0.1, 0.25, 3.0])\nprint(w)\nshow w\nh = bfloat16(w)\nprint(h, fmt(h[0]))\nprint(float(h))\nq: [Int] = [1, -2, 3]\nb = int8(q)\nprint(b, sum(b), int64(b))\n",
        "[0.1, 0.25, 3]\nw: [0.1, 0.25, 3]\n[0.100098, 0.25, 3] 0.100097656\n[0.100098, 0.25, 3]\n[1, -2, 3] 2 [1, -2, 3]\n",
    );
}

#[test]
fn a_store_out_of_range_stops() {
    stops("i8", "q = int8_zeros(2)\nq[0] = 127\nprint(q)\nq[1] = 128\n", "[127, 0]\n", "128 does not fit in an Int8 (-128 to 127)");
    stops("u8", "q: [Int] = [1, 256]\nb = uint8(q)\n", "", "256 does not fit in a UInt8 (0 to 255)");
}

#[test]
fn mixing_precisions_is_refused() {
    refused("num", "x: Float32 = 0.1\nn = 2.5\ny = x * n\n", "a Float32 and a number (float64) do not mix in *: write float32(y) to round the number, or float(x) to widen the Float32");
    refused("int", "x: Float32 = 0.1\nk: Int = 3\ny = x + k\n", "a Float32 and an Int do not mix in +: write float32(n) to round the Int, or float(x) to widen the Float32");
    refused("store", "w = float32_zeros(2)\nn = 0.5\nw[0] = n\n", "a Float32 array holds Float32s, and this is a number");
    refused("mod", "x: Float32 = 0.5\ny = x % 2\n", "`%` is not defined on Float32 in this version");
}
