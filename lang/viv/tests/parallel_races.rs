// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Stage 1 of the memory model, the races: what a pass of a `parallel for`
//! may write (crate::parallel).
//!
//! The three programs of study/memcheck/parallel that gave wrong answers
//! under OpenMP (41,889 for 200,000) are refused by the interpreter and both
//! compilers, with an error that names the write and suggests the fix; so are
//! the other ways a pass can share an element. What a pass owns compiles and
//! prints the same bytes in every engine. A loop the memory lens pins as
//! disjoint compiles, and every checked engine stops the run when two passes
//! do write one element; a release build trusts the pin.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_races_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// Build the C with `flags` and run it: (stdout, stderr, peak resident
/// memory, success). None without a C compiler or when `flags` cannot build.
fn native(dir: &Path, src: &str, flags: &[&str]) -> Option<(String, String, u64, bool)> {
    let c = viv::compile_c(src).expect("compiles to C");
    let probe = "#include <sys/resource.h>\n#include <stdio.h>\n#include <stdlib.h>\nstatic void viv_t_report(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); fprintf(stderr, \"maxrss %ld\\n\", (long)u.ru_maxrss); }\n__attribute__((constructor)) static void viv_t_init(void) { atexit(viv_t_report); }\n";
    let file = dir.join("p.c");
    std::fs::write(&file, format!("{probe}{c}")).unwrap();
    let bin = dir.join("p");
    let built = Command::new(cc()).arg("-O2").arg("-w").args(flags).arg(&file).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    if !built.status.success() {
        return None;
    }
    let out = Command::new(&bin).env("ASAN_OPTIONS", "detect_leaks=0").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    let rss = err.lines().find_map(|l| l.strip_prefix("maxrss ")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
    let err: String = err.lines().filter(|l| !l.starts_with("maxrss ")).map(|l| format!("{l}\n")).collect();
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), err, rss, out.status.success()))
}

fn interp(src: &str) -> String {
    viv::interp::run(src).expect("interprets").lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

/// The module on node against viv.wasm's runtime: (stdout, stderr, memory).
fn direct(dir: &Path, src: &str) -> Option<(String, String, u64)> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(11).any(|w| w == b"vrt_dj_note") {
        eprintln!("skip wasm: {} predates this runtime (www/build-wasm.sh)", wasm.display());
        return None;
    }
    let m = dir.join("p.wasm");
    std::fs::write(&m, viv::compile_wasm(src).expect("compiles to wasm")).unwrap();
    let out = Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg("--memory").arg(&m).env("VIV_WASM", &wasm).output().ok()?;
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    let mem = err.lines().find_map(|l| l.strip_prefix("memory ")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
    let err: String = err.lines().filter(|l| !l.starts_with("memory ")).map(|l| format!("{l}\n")).collect();
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), err, mem))
}

// ---------------------------------------------------------------- races

/// What every engine says about a program it refuses: the headline, the
/// labels and the notes of the compiler's diagnostic, after checking that
/// the interpreter and both backends refuse it with that headline.
fn refused(src: &str) -> String {
    let i = viv::interp::run(src).expect_err("the interpreter refuses it");
    let c = viv::compile_c(src).expect_err("the C backend refuses it");
    let w = viv::compile_wasm(src).expect_err("the wasm backend refuses it");
    let first = |s: &str| s.lines().next().unwrap_or_default().to_string();
    assert_eq!(first(&i), first(&c));
    assert_eq!(first(&c), first(&w));
    let ds = viv::build_ir(src, knotkit::span::SourceId(0)).err().expect("refused");
    let d = &ds[0];
    let mut text = d.message.clone();
    for l in d.primary.iter().chain(d.secondary.iter()) {
        text.push('\n');
        text.push_str(&l.message);
    }
    for n in &d.notes {
        text.push('\n');
        text.push_str(n);
    }
    text
}

fn probe(name: &str) -> String {
    std::fs::read_to_string(root().join(format!("study/memcheck/parallel/{name}.viv"))).unwrap().replace("NITER", "1000")
}

#[test]
fn a_parallel_for_writing_one_element_from_every_pass_is_refused() {
    let e = refused(&probe("race_same_cell"));
    assert!(e.contains("two passes of this `parallel for` can write the same element of `acc`"), "{e}");
    assert!(e.contains("does not depend on `i`"), "{e}");
    assert!(e.contains("reduction"), "{e}");
}

#[test]
fn a_parallel_for_reading_its_neighbors_writes_is_refused() {
    let e = refused(&probe("race_neighbor"));
    assert!(e.contains("reads elements of `a` that other passes write"), "{e}");
    assert!(e.contains("copy"), "{e}");
}

#[test]
fn a_parallel_for_writing_through_a_def_is_refused() {
    let e = refused(&probe("race_through_def"));
    assert!(e.contains("`bump` writes `acc` here at an index that does not come from `i`"), "{e}");
}

#[test]
fn more_races_are_refused() {
    let cases = [
        // a global written by a def the body calls
        ("G = zeros(4)\ndef add1(k) {\n    G[0] = G[0] + k\n    return 0\n}\nout = zeros(4)\nparallel for i to 4 {\n    out[i] = add1(i)\n}\nprint(G)\n", "writes the global `G`"),
        // two positions: pass i writes column i of one array and row i of the same array
        ("m = zeros(4, 4)\nparallel for i to 4 {\n    m[i, 0] = 1.0\n    m[0, i] = 2.0\n}\nprint(m)\n", "can write the same element of `m`"),
        // an alias of a shared array
        ("a = zeros(4)\nparallel for i to 4 {\n    t = a\n    t[0] = i\n}\nprint(a)\n", "can write the same element of `a`"),
        // a record holding a shared array, written through its field
        ("record R {\n    xs: Arr\n}\na = zeros(4)\nparallel for i to 4 {\n    r = R { xs: a }\n    r.xs[0] = i\n}\nprint(a)\n", "can write the same element of `a`"),
        // a write through a def passed as a function value (seen after defun)
        ("def bump(v: Arr, j) {\n    v[j] = v[j] + 1.0\n    return 0.0\n}\ndef run(f, v: Arr) {\n    return f(v, 0)\n}\nacc = zeros(4)\nparallel for i to 4 {\n    t = run(bump, acc)\n}\nprint(acc)\n", "can write the same element"),
        // a slice that is not the pass's own
        ("m = zeros(4, 4)\nparallel for i to 4 {\n    m[:, i] = zeros(4) + i\n    m[i, :] = zeros(4)\n}\nprint(m)\n", "same element of `m`"),
        // a builtin writing the whole array
        ("a = zeros(4)\nparallel for i to 4 {\n    fill(a, i)\n}\nprint(a)\n", "same element of `a`"),
        // a def reading the global the passes write
        ("G = zeros(4)\ndef peek(k) {\n    return G[3 - k]\n}\nparallel for i to 4 {\n    G[i] = peek(i) + 1.0\n}\nprint(G)\n", "reads the global `G`"),
        // leaving the loop from a pass
        ("a = zeros(4)\nparallel for i to 4 {\n    if i > 1 {\n        break\n    }\n    a[i] = 1.0\n}\nprint(a)\n", "cannot `break`"),
        // two offsets of the pass variable: pass i and pass i + 1 both write a[i + 1]
        ("a = zeros(6)\nparallel for i to 4 {\n    a[i] = 1.0\n    a[i + 1] = 2.0\n}\nprint(a)\n", "same element of `a`"),
        // a read at another offset than the write (the stencil's neighbor)
        ("a = zeros(6)\nparallel for i to 4 {\n    a[i + 1] = a[i] + 1.0\n}\nprint(a)\n", "reads elements of `a` that other passes write"),
        // a derived index assigned again in the pass is not the pass's own
        ("a = zeros(8)\nparallel for i to 4 {\n    j = i + 1\n    j = 2\n    a[j] = 1.0\n}\nprint(a)\n", "same element of `a`"),
    ];
    for (src, want) in cases {
        let e = refused(src);
        assert!(e.contains(want), "expected `{want}` in:\n{e}");
    }
}

#[test]
fn a_pass_may_write_what_it_owns() {
    // each of these writes only elements indexed by the loop variable, or an
    // array the pass made, and every engine runs it to the same bytes
    let cases = [
        "a = zeros(8)\nparallel for i to 8 {\n    a[i] = a[i] + i * 2.0\n}\nprint(a)\n",
        "def put(v: Arr, j, x) {\n    v[j] = x\n    return 0\n}\na = zeros(8)\nparallel for i to 8 {\n    put(a, i, i * i)\n}\nprint(a)\n",
        "m = zeros(4, 3)\nparallel for i to 4 {\n    for j to 3 {\n        m[i, j] = i + j\n    }\n}\nprint(m)\n",
        "m = zeros(3, 4)\nparallel for i to 4 {\n    m[:, i] = zeros(3) + i\n}\nprint(m)\n",
        "W = zeros(4, 2, 2)\nparallel for c to 4 {\n    mine = zeros(2, 2)\n    mine[0, 0] = c\n    mine[1, 1] = -c\n    setslab(W, c, mine)\n}\nprint(W)\n",
        "def norm2(r: Arr) {\n    return sum(r * r)\n}\nA = zeros(4, 3) + 1.0\nout = zeros(4)\nparallel for i to 4 {\n    out[i] = norm2(slab(A, i)) + A[i, 0]\n}\nprint(out)\n",
        // the pass variable plus a constant, directly and through a local
        // assigned once (NPB MG's planes i3 = p + 1 of a grid with ghosts)
        "a = zeros(6)\nparallel for i to 4 {\n    a[i + 1] = a[i + 1] + i\n}\nprint(a)\n",
        "u = zeros(6, 3)\nparallel for p to 4 {\n    i3 = p + 1\n    for j to 3 {\n        u[i3, j] = u[i3, j] + p * j\n    }\n}\nprint(u)\n",
    ];
    let dir = scratch("own");
    for src in cases {
        let want = interp(src);
        assert!(!want.is_empty());
        if let Some((out, _, _, ok)) = native(&dir, src, &[]) {
            assert!(ok, "{src}");
            assert_eq!(out, want, "{src}");
        }
        if let Some((out, _, _)) = direct(&dir, src) {
            assert_eq!(out, want, "{src}");
        }
    }
}

// ---------------------------------------------------------------- pins

/// A scatter through a permutation: disjoint when `P` is one, and the memory
/// lens says so. The second call's `P` repeats an index.
const SCATTER: &str = "def put_at(G: Arr, v, at) {\n    G[at] = v\n    return 0\n}\ndef own1_scatter(P: Arr) {\n    n = len(P)\n    G = zeros(n)\n    parallel for q to n {\n        put_at(G, q * 10.0, P[q])\n    }\n    return G\n}\nprint(own1_scatter([3, 1, 0, 2]))\nprint(own1_scatter([3, 1, 3, 2]))\n";
const SCATTER_LENS: &str = "own1_scatter\n  disjoint  parallel for q: G   \"P is a permutation\"\n";

/// The lens files a driver hands the compiler are process-wide
/// (`memory_lens::set_loaded`), so the tests that set them take turns.
static LENS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn a_disjoint_pin_is_tested_while_the_loop_runs() {
    let _turn = LENS.lock().unwrap_or_else(|e| e.into_inner());
    viv::memory_lens::set_loaded(vec![("own1.memory".to_string(), SCATTER_LENS.to_string())]);
    let want_err = "error: passes 0 and 2 both write G[3] in the parallel for over `q` in `own1_scatter`, which own1.memory pins as disjoint";
    // the interpreter
    let run = viv::interp::run(SCATTER);
    let e = run.expect_err("the overlap stops the run");
    assert!(format!("error: {e}").starts_with(want_err), "{e}");
    let dir = scratch("pin");
    // the checked C: the first scatter prints, the second stops
    if let Some((out, err, _, ok)) = native(&dir, SCATTER, &[]) {
        assert!(!ok);
        assert_eq!(out, "[20, 10, 30, 0]\n");
        assert!(err.starts_with(want_err), "{err}");
    }
    // the release C trusts the pin
    if let Some((out, _, _, ok)) = native(&dir, SCATTER, &["-DVIV_RELEASE"]) {
        assert!(ok);
        assert_eq!(out, "[20, 10, 30, 0]\n[0, 10, 30, 20]\n");
    }
    // the wasm module
    if let Some((_, err, _)) = direct(&dir, SCATTER) {
        assert!(err.starts_with(want_err), "{err}");
    }
    // without the lens the loop is refused
    viv::memory_lens::set_loaded(Vec::new());
    let e = viv::compile_c(SCATTER).expect_err("unpinned, the loop is refused");
    assert!(e.contains("two passes of this `parallel for` can write the same element of `G`"), "{e}");
}

/// The same scatter two calls down, through a local alias, with a scratch
/// array the inner def makes for itself (its writes need no record).
const DEEP: &str = "def put2(H: Arr, v, at) {\n    scratch = zeros(3)\n    scratch[1] = v\n    H[at] = scratch[1]\n    return 0\n}\ndef put1(G: Arr, v, at) {\n    t = G\n    return put2(t, v, at)\n}\ndef own1_deep(P: Arr) {\n    n = len(P)\n    G = zeros(n)\n    parallel for q to n {\n        put1(G, q * 10.0, P[q])\n    }\n    return G\n}\nprint(own1_deep([3, 1, 0, 2]))\nprint(own1_deep([3, 1, 3, 2]))\n";
const DEEP_LENS: &str = "own1_deep\n  disjoint  parallel for q: G   \"P is a permutation\"\n";

#[test]
fn a_disjoint_pin_is_tested_through_calls_and_aliases() {
    let _turn = LENS.lock().unwrap_or_else(|e| e.into_inner());
    viv::memory_lens::set_loaded(vec![("deep.memory".to_string(), DEEP_LENS.to_string())]);
    let want_err = "error: passes 0 and 2 both write G[3] in the parallel for over `q` in `own1_deep`, which deep.memory pins as disjoint";
    let e = viv::interp::run(DEEP).expect_err("the overlap stops the run");
    assert!(format!("error: {e}").starts_with(want_err), "{e}");
    let c = viv::compile_c(DEEP).unwrap();
    // put2's write of H is recorded, its scratch array's is not
    assert!(c.contains("viv_dj_set1_at(H,") || c.contains("viv_dj_set1(H,"), "the write through the alias is recorded");
    assert!(!c.contains("viv_dj_set1_at(scratch,") && !c.contains("viv_dj_set1(scratch,"), "the def's own array is not");
    let dir = scratch("deep");
    if let Some((out, err, _, ok)) = native(&dir, DEEP, &[]) {
        assert!(!ok);
        assert_eq!(out, "[20, 10, 30, 0]\n");
        assert!(err.starts_with(want_err), "{err}");
    }
    if let Some((_, err, _)) = direct(&dir, DEEP) {
        assert!(err.starts_with(want_err), "{err}");
    }
    viv::memory_lens::set_loaded(Vec::new());
}

#[test]
fn the_repository_carries_every_memory_lens_file() {
    // the site and the build scripts compile expanded text, so the compiler
    // carries the repository's lens files (viv::memory_lens::REPO)
    let mut found = Vec::new();
    let mut stack = vec![root()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                // the feature catalog's examples run only through the CLI, which
                // reads a lens beside its program; carrying one would apply its
                // pin (a def named `scatter`) to every program's `scatter`
                if p.ends_with("lang/viv/features/examples") {
                    continue;
                }
                if !name.starts_with('.') && !matches!(name.as_str(), "target" | "node_modules" | "www") {
                    stack.push(p);
                }
            } else if name.ends_with(".memory") {
                found.push(p.strip_prefix(root()).unwrap().to_string_lossy().to_string());
            }
        }
    }
    found.sort();
    let carried: Vec<String> = viv::memory_lens::REPO.iter().map(|(_, f, _)| f.to_string()).collect();
    assert_eq!(found, carried);
    for (_, file, text) in viv::memory_lens::REPO {
        assert_eq!(std::fs::read_to_string(root().join(file)).unwrap(), *text, "{file} is carried as it is on disk");
        viv::memory_lens::parse_file(text, file).unwrap();
    }
}

#[test]
fn a_lens_file_reads_only_what_this_stage_checks() {
    use viv::memory_lens::parse_file;
    assert!(parse_file("f @1234\n  disjoint parallel for q: A, B \"why\"\n", "x.memory").is_ok());
    assert!(parse_file("f\n  disjoint parallel for q: A\n", "x.memory").unwrap_err().contains("says why"));
    assert!(parse_file("f\n  bounded per iteration \"x\"\n", "x.memory").unwrap_err().contains("stage 2"));
    assert!(parse_file("  disjoint parallel for q: A \"x\"\n", "x.memory").unwrap_err().contains("after the line naming its def"));
}
