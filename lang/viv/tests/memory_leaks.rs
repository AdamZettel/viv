// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Stage 1 of the memory model, the leaks: what study/memcheck measured
//! growing with the work, closed with today's memory model.
//!
//! Each test runs a program at a short and at a ten times longer loop, in the
//! compiled C (peak resident memory) and in the direct wasm module (its
//! memory), and requires the long run to stay within a margin far below what
//! the leak added before the fix (study/memcheck/scale.tsv gives
//! the numbers). Every engine prints the
//! same bytes, and an AddressSanitizer and UndefinedBehaviorSanitizer build of
//! the C, where the compiler has one, reports nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_leaks_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

const MB: u64 = 1 << 20;

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
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        eprintln!("skip wasm: {} has no runtime exports (www/build-wasm.sh)", wasm.display());
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

/// A leak program at `short` and ten times `short` passes: every engine
/// agrees, the C and the wasm memory stay flat, ASan reads nothing freed.
fn flat(tag: &str, program: impl Fn(usize) -> String, short: usize, margin: u64) {
    let dir = scratch(tag);
    let (a, b) = (program(short), program(10 * short));
    let want = interp(&a);
    if let (Some((out, _, s, ok)), Some((_, _, l, ok2))) = (native(&dir, &a, &[]), native(&dir, &b, &[])) {
        assert!(ok && ok2, "{tag}: the C runs");
        assert_eq!(out, want, "{tag}: C");
        if s != 0 {
            assert!(l < s + margin, "{tag}: C peak RSS {s} for {short} passes, {l} for {} passes", 10 * short);
        }
    }
    if let (Some((out, _, s)), Some((_, _, l))) = (direct(&dir, &a), direct(&dir, &b)) {
        assert_eq!(out, want, "{tag}: wasm");
        assert!(l < s + margin, "{tag}: wasm memory {s} for {short} passes, {l} for {} passes", 10 * short);
    }
    if let Some((out, err, _, ok)) = native(&dir, &a, &["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]) {
        assert!(ok && !err.contains("runtime error"), "{tag}: the sanitizer build: {err}");
        assert_eq!(out, want, "{tag}: the sanitizer build");
    }
}

// ---------------------------------------------------------------- leaks

/// `keys(m)` and `values(m)` of a number map, every pass of a loop at the
/// top level and in a def: 16 KB a call before the fix.
fn keys_program(passes: usize) -> String {
    format!(
        r#"
mp = map()
for i to 2000 {{
    mp[i] = i * 0.5
}}
def work(n) {{
    s = 0.0
    for k to n {{
        s = s + sum(keys(mp)) - sum(values(mp))
    }}
    return s
}}
s = 0.0
for k to {passes} {{
    s = s + sum(keys(mp)) + sum(values(mp))
}}
print(s, work({passes}))
"#
    )
}

#[test]
fn keys_and_values_in_a_loop_run_in_the_memory_of_one_pass() {
    // 10,000 more passes of two 16 KB arrays, twice: over 600 MB
    flat("keys", keys_program, 1000, 64 * MB);
    let c = viv::compile_c(&keys_program(10)).unwrap();
    assert!(c.matches("viv_arena_release_kept(viv_lmark)").count() >= 4, "both loops release each pass");
}

/// A `while` whose condition allocates and whose body does not.
fn while_program(passes: usize) -> String {
    format!(
        r#"
def g(v) {{
    return v * 2.0
}}
a = linspace(0.0, 1.0, 2000)
s = 0.0
while sum(g(a)) + s < 2000.0 + {passes} {{
    s = s + 1.0
}}
def inner(n) {{
    t = 0.0
    while len(g(a)) + t < 2000 + n {{
        t = t + 1.0
    }}
    return t
}}
print(s, inner({passes}))
"#
    )
}

#[test]
fn a_while_whose_condition_allocates_runs_in_the_memory_of_one_pass() {
    // 9,000 more passes of a 16 KB array, twice: 280 MB
    flat("while", while_program, 1000, 64 * MB);
}

/// Maps made in a def and dropped there: numbers, strings, a copy; and maps
/// a def returns, keeps in a record, or a loop's variable holds.
fn maps_program(passes: usize) -> String {
    format!(
        r#"
record Box {{
    m: Map
    n: Num
}}
def count(k) {{
    mm = map()
    for i to 200 {{
        mm[i] = k + i
    }}
    c = copy(mm)
    return len(c) + mm[3]
}}
def words(k) {{
    t = map()
    for i to 50 {{
        t["w" + str(i)] = "v" + str(k)
    }}
    return len(t) + len(t["w7"])
}}
def make(k) {{
    m = map()
    m[k] = k * 3.0
    return m
}}
def boxed(k) -> Box {{
    m = map()
    m[1] = k
    return Box {{ m: m, n: k }}
}}
s = 0.0
last = map()
for k to {passes} {{
    s = s + count(k) + words(k)
    last = make(k)
    b = boxed(k)
    s = s + last[k] + b.m[1]
}}
print(s)
print(last)
print(b.m, b.n)
"#
    )
}

#[test]
fn maps_made_in_defs_are_freed_when_nothing_holds_them() {
    // 18,000 more calls, each making two maps of 200 entries (8 KB) and one
    // of 50 strings: over 300 MB
    flat("maps", maps_program, 2000, 64 * MB);
}

/// study/compare/strings/parse_numbers.viv: a map of strings made in a def.
#[test]
fn the_compare_program_that_leaked_a_map_of_strings_now_frees_it() {
    let path = root().join("study/compare/strings/parse_numbers.viv");
    let src = std::fs::read_to_string(&path).unwrap();
    let dir = scratch("parse_numbers");
    let want = interp(&src);
    let Some((out, _, _, ok)) = native(&dir, &src, &[]) else { return };
    assert!(ok);
    assert_eq!(out, want);
    // `leaks` reports blocks no pointer reaches (macOS); it found 33 before
    if Command::new("leaks").arg("--help").output().is_ok() {
        let run = Command::new("leaks").arg("--atExit").arg("--").arg(dir.join("p")).output().unwrap();
        let text = String::from_utf8_lossy(&run.stdout);
        if let Some(line) = text.lines().find(|l| l.contains("leaks for")) {
            assert!(line.contains(": 0 leaks for"), "{line}");
        }
    }
}

/// A `parallel for` whose body allocates, in a def called many times.
fn par_program(calls: usize) -> String {
    format!(
        r#"
def row(i) {{
    return linspace(0.0, i, 64)
}}
def sweep(n) {{
    out = zeros(n)
    parallel for i to n {{
        r = row(i)
        if i % 7 == 0 {{
            continue
        }}
        out[i] = sum(r)
    }}
    return sum(out)
}}
s = 0.0
for k to {calls} {{
    s = s + sweep(1000)
}}
print(s)
"#
    )
}

#[test]
fn a_parallel_for_frees_each_pass_on_the_thread_that_ran_it() {
    // without OpenMP: the loop arena's rule (and the last pass's `r` kept, as
    // the interpreter keeps it)
    flat("par", par_program, 20, 64 * MB);
    // with OpenMP, where the compiler has it: 82 MB after 200 calls before
    let dir = scratch("par_omp");
    let omp = ["-Xpreprocessor", "-fopenmp", "-I/opt/homebrew/opt/libomp/include", "-L/opt/homebrew/opt/libomp/lib", "-lomp"];
    let gnu = ["-fopenmp"];
    for flags in [&omp[..], &gnu[..]] {
        if let (Some((o1, _, s, ok1)), Some((o2, _, l, ok2))) = (native(&dir, &par_program(20), flags), native(&dir, &par_program(200), flags)) {
            assert!(ok1 && ok2);
            assert_eq!(o1, interp(&par_program(20)));
            assert_eq!(o2, interp(&par_program(200)));
            if s != 0 {
                assert!(l < s + 16 * MB, "OpenMP peak RSS {s} for 20 calls, {l} for 200 calls");
            }
            return;
        }
    }
    eprintln!("skip: no OpenMP here");
}
