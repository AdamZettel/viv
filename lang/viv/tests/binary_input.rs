// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Binary input: `read_bytes` and the little-endian
//! readers, in all three engines -- the interpreter, Viv -> C -> native, and
//! the direct wasm backend in node (skipped when node or a viv.wasm with the
//! runtime exports is missing; VIV_WASM=<path> names one). The reference is
//! Python's struct module (tools/bytes/gen_cases.py; skipped without python3).

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_bytes_{}_{tag}", std::process::id()));
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

fn wasm_runtime() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(17).any(|w| w == b"vrt_s_read_bytes\0") && !bytes.windows(16).any(|w| w == b"vrt_s_read_bytes") {
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    Some(wasm)
}

/// The three engines on `p.viv` in `dir`, with `files` beside it (the wasm
/// run attaches each, as a page does).
fn engines(tag: &str, dir: &Path, files: &[&str]) -> (Ran, Ran, Option<Ran>) {
    let p = dir.join("p.viv");
    let i = ran(Command::new(viv_bin()).current_dir(dir).args(["run", "-q"]).arg(&p).output().unwrap());
    let c = Command::new(viv_bin()).current_dir(dir).arg(&p).args(["-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{tag}: viv refused the program for C:\n{}", String::from_utf8_lossy(&c.stderr));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let b = Command::new(cc).current_dir(dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "{tag}: the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    let n = ran(Command::new(dir.join("p")).current_dir(dir).output().unwrap());
    let w = wasm_runtime().map(|wasm| {
        let c = Command::new(viv_bin()).current_dir(dir).arg(&p).args(["-o", "p.wasm"]).output().unwrap();
        assert!(c.status.success(), "{tag}: viv refused the program for wasm:\n{}", String::from_utf8_lossy(&c.stderr));
        let mut cmd = Command::new("node");
        cmd.current_dir(dir).arg(root().join("tools/run-wasm.mjs"));
        for f in files {
            cmd.arg(format!("--file={f}={}", dir.join(f).display()));
        }
        ran(cmd.arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap())
    });
    (i, n, w)
}

fn setup(tag: &str, src: &str, data: &[u8]) -> PathBuf {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    std::fs::write(dir.join("d.bin"), data).unwrap();
    dir
}

fn prints(tag: &str, src: &str, data: &[u8], want: &str) {
    let dir = setup(tag, src, data);
    let (i, n, w) = engines(tag, &dir, &["d.bin"]);
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), want, ""), "{tag}: interpreter");
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "{tag}: direct wasm\n{}", w.err);
    }
}

/// Every engine stops with `error: <msg>`; the interpreter and C agree byte
/// for byte (location included), wasm prints the message without one.
fn stops(tag: &str, src: &str, data: &[u8], msg: &str) {
    let dir = setup(tag, src, data);
    let (i, n, w) = engines(tag, &dir, &["d.bin"]);
    assert_eq!(i.code, Some(4), "{tag}: interpreter {i:?}");
    assert!(i.err.starts_with(&format!("error: {msg}\n  --> p.viv:")), "{tag}: {}", i.err);
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.err.as_str()), (Some(4), format!("error: {msg}\n").as_str()), "{tag}: direct wasm");
    }
}

#[test]
fn every_reader_at_every_offset_matches_python_struct() {
    let dir = scratch("cases");
    let gen = Command::new("python3").arg(root().join("tools/bytes/gen_cases.py")).arg(&dir).output();
    let Ok(gen) = gen else {
        eprintln!("skipped: no python3");
        return;
    };
    assert!(gen.status.success(), "gen_cases.py: {}", String::from_utf8_lossy(&gen.stderr));
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/bytes/cases.viv"), dir.join("p.viv")).unwrap();
    let want = std::fs::read_to_string(dir.join("cases.txt")).unwrap();
    let (i, n, w) = engines("cases", &dir, &["cases.bin"]);
    assert_eq!(i.code, Some(0), "interpreter: {}", i.err);
    let first_diff = |got: &str| got.lines().zip(want.lines()).find(|(a, b)| a != b).map(|(a, b)| format!("{a}  (struct: {b})"));
    assert!(i.out == want, "interpreter differs from struct: {:?}", first_diff(&i.out));
    assert!(n.out == want, "C differs from struct: {:?}", first_diff(&n.out));
    if let Some(w) = w {
        assert!(w.out == want, "wasm differs from struct: {:?} {}", first_diff(&w.out), w.err);
    }
}

/// Python's `f.seek(offset); f.read(count)`: fewer bytes where the file
/// ends, none past it.
#[test]
fn a_window_of_a_file() {
    let data: Vec<u8> = (0u8..=255).collect();
    prints(
        "window",
        "a = read_bytes(\"d.bin\")\nw = read_bytes(\"d.bin\", 250, 3)\nt = read_bytes(\"d.bin\", 254)\ne = read_bytes(\"d.bin\", 256)\nshort = read_bytes(\"d.bin\", 253, 10)\npast = read_bytes(\"d.bin\", 300, 4)\nprint(len(a), w, t, len(e), short, len(past))\n",
        &data,
        "256 [250, 251, 252] [254, 255] 0 [253, 254, 255] 0\n",
    );
}

#[test]
fn readers_with_an_int_offset_and_a_literal() {
    prints(
        "offsets",
        "b = read_bytes(\"d.bin\")\nk: Int = 1\nprint(le_u16(b, k), le_u16(b, 1), le_f32(b, k + 2), le_i64(b, 0))\n",
        &[0xff, 0x34, 0x12, 0, 0, 0x80, 0xbf, 0xff],
        "4660 4660 -1 -18155135996644097\n",
    );
}

#[test]
fn the_errors_say_what_is_wrong() {
    let eight = [0xffu8; 8];
    stops("past_end", "b = read_bytes(\"d.bin\")\nprint(le_u32(b, 6))\n", &eight, "`le_u32`: 4 bytes at offset 6 are not inside an array of 8");
    stops("negative", "b = read_bytes(\"d.bin\")\nk: Int = -1\nprint(le_u8(b, k))\n", &eight, "`le_u8`: 1 bytes at offset -1 are not inside an array of 8");
    stops("not_a_byte", "b: [Int] = [1, 300]\nprint(le_u16(b, 0))\n", &eight, "`le_u16`: element 1 is 300, not a byte (0 to 255)");
    stops("u64_over", "b = read_bytes(\"d.bin\")\nprint(le_u64(b, 0))\n", &eight, "`le_u64`: 18446744073709551615 does not fit in an Int; le_i64 reads the same bits");
    stops("bad_offset", "b = read_bytes(\"d.bin\", 1.5)\nprint(b)\n", &eight, "`read_bytes` takes a whole offset and count of 0 or more");
}

#[test]
fn a_missing_file() {
    let dir = setup("missing", "b = read_bytes(\"nope.bin\")\nprint(b)\n", &[]);
    let (i, n, _) = engines("missing", &dir, &[]);
    assert_eq!(i.code, Some(4));
    assert!(i.err.starts_with("error: No such file or directory: 'nope.bin'"), "{}", i.err);
    assert_eq!(i, n);
}

#[test]
fn a_reader_takes_an_int_array() {
    let dir = scratch("refused");
    std::fs::write(dir.join("p.viv"), "b = [1.0, 2.0]\nprint(le_u16(b, 0))\n").unwrap();
    let c = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap());
    assert_ne!(c.code, Some(0));
    assert!(c.err.contains("reads an array of Ints"), "{}", c.err);
}
