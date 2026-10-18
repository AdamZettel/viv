// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Loops free what their iterations allocate, and a def returning a record
//! frees everything but the record.
//!
//! Each program allocates arrays every time round a loop: at the top level,
//! inside a def, in a `while` with a `continue`, and through a def that
//! returns a record of arrays. Run for STEPS and for ten times STEPS, the
//! compiled program's peak resident memory and the direct wasm module's
//! memory must stay flat (before the fix the C build grew by every
//! iteration's arrays: 838 MB against 26 MB at 400 steps of 100,000 doubles).
//! Every engine prints the same bytes, and an AddressSanitizer build of the
//! C, where the compiler has one, reports no use of freed memory.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_loop_arena_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// The program: arrays of `n` doubles, allocated `steps` times in each loop.
fn program(n: usize, steps: usize) -> String {
    format!(
        r#"
record Pair {{ a: Arr, b: Arr, s: Num }}
record Nest {{ p: Pair, c: Arr }}
N = {n}
STEPS = {steps}
def pair(x: Arr, k) -> Pair {{
    t = x * 2.0
    u = t + 1.0
    return Pair {{ a: x + k, b: zeros(len(x)), s: sum(u) - sum(t) }}
}}
def nest(x: Arr) -> Nest {{
    return Nest {{ p: pair(x, 1.0), c: x * 3.0 }}
}}
def in_def(x: Arr) {{
    s = 0.0
    y = x
    for k to STEPS {{
        y = y * 1.0 + 0.0
        p = pair(y, k)
        s = s + p.s + p.a[0]
    }}
    return s
}}
x = zeros(N) + 1.0
total = 0.0
for k to STEPS {{
    x = x * 1.0
    q = pair(x, 1.0)
    total = total + q.a[0] + q.s
}}
i = 0
kept = zeros(N)
while i < STEPS {{
    i = i + 1
    w = nest(x)
    if i > 3 {{
        continue
    }}
    kept = w.p.a + w.c
}}
show total
show in_def(x)
show kept[0]
show w.p.a[N - 1]
"#
    )
}

/// Build the C with `flags` and run it: (stdout, peak resident memory in the
/// platform's unit, or 0 when it cannot be read). None without a C compiler.
fn native(dir: &Path, src: &str, flags: &[&str]) -> Option<(String, u64, bool)> {
    let c = viv::compile_c(src).expect("compiles to C");
    // report the peak resident set size on stderr at exit
    let probe = "#include <sys/resource.h>\n#include <stdio.h>\n#include <stdlib.h>\nstatic void viv_t_report(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); fprintf(stderr, \"maxrss %ld\\n\", (long)u.ru_maxrss); }\n__attribute__((constructor)) static void viv_t_init(void) { atexit(viv_t_report); }\n";
    let file = dir.join("p.c");
    std::fs::write(&file, format!("{probe}{c}")).unwrap();
    let bin = dir.join("p");
    let built = Command::new(cc()).arg("-O2").args(flags).arg(&file).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    if !built.status.success() {
        return None;
    }
    let out = Command::new(&bin).output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    let rss = err.lines().find_map(|l| l.strip_prefix("maxrss ")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), rss, out.status.success()))
}

fn interp(src: &str) -> String {
    viv::interp::run(src).expect("interprets").lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

/// The module on node against viv.wasm's runtime: (stdout, memory in bytes).
fn direct(dir: &Path, src: &str) -> Option<(String, u64)> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        eprintln!("skip wasm: {} has no runtime exports", wasm.display());
        return None;
    }
    let m = dir.join("p.wasm");
    std::fs::write(&m, viv::compile_wasm(src).expect("compiles to wasm")).unwrap();
    let out = Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg("--memory").arg(&m).env("VIV_WASM", &wasm).output().ok()?;
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    let mem = err.lines().find_map(|l| l.strip_prefix("memory ")).and_then(|v| v.trim().parse().ok())?;
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), mem))
}

const N: usize = 100_000;
/// ru_maxrss is in bytes on macOS and in KB on Linux; the margin is taken in
/// bytes, which is the stricter of the two readings
const MB: u64 = 1 << 20;
const STEPS: usize = 20;

#[test]
fn every_engine_prints_the_same_bytes() {
    let src = program(N, STEPS);
    let want = interp(&src);
    assert!(want.contains("total: "), "{want}");
    let dir = scratch("same");
    if let Some((out, _, ok)) = native(&dir, &src, &[]) {
        assert!(ok);
        assert_eq!(out, want, "C");
    }
    if let Some((out, _)) = direct(&dir, &src) {
        assert_eq!(out, want, "wasm");
    }
}

#[test]
fn the_compiled_program_runs_in_the_memory_of_one_iteration() {
    let dir = scratch("rss");
    let Some((_, short, _)) = native(&dir, &program(N, STEPS), &[]) else { return };
    let Some((_, long, _)) = native(&dir, &program(N, 10 * STEPS), &[]) else { return };
    if short == 0 {
        return;
    }
    // ten times the iterations: 180 more of them, eight arrays of 800 KB
    // each, would add 1.1 GB; the allocator's own slack is a few MB
    assert!(long < short + 64 * MB, "peak RSS {short} for {STEPS} steps, {long} for {} steps", 10 * STEPS);
}

#[test]
fn the_wasm_module_runs_in_the_memory_of_one_iteration() {
    let dir = scratch("wasm");
    let Some((_, short)) = direct(&dir, &program(N, STEPS)) else { return };
    let Some((_, long)) = direct(&dir, &program(N, 10 * STEPS)) else { return };
    assert!(long < short + 64 * MB, "wasm memory {short} for {STEPS} steps, {long} for {} steps", 10 * STEPS);
}

#[test]
fn a_def_returning_a_record_frees_its_temporaries() {
    // straight-line calls, no loop to release them: each call allocates four
    // arrays and returns two; the other two are freed when it returns
    let calls: String = (0..40).map(|k| format!("q{k} = pair(x, {k})\nshow q{k}.a[0]\n")).collect();
    let src = format!(
        "record Pair {{ a: Arr, b: Arr, s: Num }}\ndef pair(x: Arr, k) -> Pair {{\n    t = x * 2.0\n    u = t + 1.0\n    return Pair {{ a: x + k, b: zeros(len(x)), s: sum(u) - sum(t) }}\n}}\nx = zeros({N}) + 1.0\n{calls}"
    );
    let dir = scratch("rec");
    let want = interp(&src);
    if let Some((out, _, ok)) = native(&dir, &src, &[]) {
        assert!(ok);
        assert_eq!(out, want);
    }
    if let Some((out, _)) = direct(&dir, &src) {
        assert_eq!(out, want);
    }
    let c = viv::compile_c(&src).unwrap();
    assert!(c.contains("viv_arena_release_kept(viv_mark)"), "pair keeps its record's arrays and frees the rest");
}

#[test]
fn no_freed_array_is_read_under_address_sanitizer() {
    let dir = scratch("asan");
    let src = program(N / 10, STEPS);
    let Some((out, _, ok)) = native(&dir, &src, &["-fsanitize=address", "-g"]) else {
        eprintln!("skip: no AddressSanitizer build here");
        return;
    };
    assert!(ok, "the AddressSanitizer build failed at run time");
    assert_eq!(out, interp(&src));
}

// ------------------------------------------------ strings
//
// Strings live on the same arena, and the loop keeps what its variables hold
// through the same keep list: a string its block, a list of strings its block
// and chunks, a record every array and string in it.

/// Text made every time round a loop: a string of `n` bytes per pass at the
/// top level and in a def, a record with a string, a list and an array
/// returned by a def called in the loop, and a record update of it.
fn text_program(n: usize, steps: usize) -> String {
    format!(
        r#"
record Line {{ name: Str, words: [Str], xs: Arr }}
N = {n}
STEPS = {steps}
def line(k) -> Line {{
    pad = "x" * N
    s = "line " + str(k) + " " + pad
    return Line {{ name: upper(s[:8]), words: split(s), xs: zeros(3) + k }}
}}
def lengths(steps) {{
    total = 0.0
    last = ""
    for k to steps {{
        t = "y" * N + str(k)
        total = total + len(t)
        last = t[N:]
    }}
    return total + len(last)
}}
kept = line(0)
first = ""
words = []
i = 0
while i < STEPS {{
    i = i + 1
    l = line(i)
    r = Line {{ name: l.name + "!", ..l }}
    if i % 3 == 0 {{
        continue
    }}
    if i == 2 {{
        kept = r
        first = l.name
    }}
    words = append(words, r.name)
}}
acc = ""
for k to STEPS {{
    big = "z" * N
    acc = acc + big[:1]
}}
show len(words)
print(join(",", words[:3]))
print(kept.name, first, len(kept.words), kept.xs[0], len(kept.words[2]))
show lengths(STEPS)
show acc
"#
    )
}

#[test]
fn strings_every_engine_prints_the_same_bytes() {
    let src = text_program(N, STEPS);
    let want = interp(&src);
    assert!(want.contains("LINE 2 X! LINE 2 X 3 2 100000"), "{want}");
    let dir = scratch("text_same");
    if let Some((out, _, ok)) = native(&dir, &src, &[]) {
        assert!(ok);
        assert_eq!(out, want, "C");
    }
    if let Some((out, _)) = direct(&dir, &src) {
        assert_eq!(out, want, "wasm");
    }
}

#[test]
fn strings_made_in_loops_run_in_the_memory_of_one_iteration() {
    // ten times the passes: 180 more, each making several strings of 100 KB
    // (a line, its split, a padded copy), would add well over 64 MB
    let dir = scratch("text_rss");
    if let (Some((_, short, _)), Some((_, long, _))) = (native(&dir, &text_program(N, STEPS), &[]), native(&dir, &text_program(N, 10 * STEPS), &[])) {
        if short != 0 {
            assert!(long < short + 64 * MB, "C peak RSS {short} for {STEPS} steps, {long} for {} steps", 10 * STEPS);
        }
    }
    if let (Some((_, short)), Some((_, long))) = (direct(&dir, &text_program(N, STEPS)), direct(&dir, &text_program(N, 10 * STEPS))) {
        assert!(long < short + 64 * MB, "wasm memory {short} for {STEPS} steps, {long} for {} steps", 10 * STEPS);
    }
}

#[test]
fn no_freed_string_is_read_under_address_sanitizer() {
    let dir = scratch("text_asan");
    let src = text_program(N / 10, STEPS);
    let Some((out, _, ok)) = native(&dir, &src, &["-fsanitize=address", "-g"]) else {
        eprintln!("skip: no AddressSanitizer build here");
        return;
    };
    assert!(ok, "the AddressSanitizer build failed at run time");
    assert_eq!(out, interp(&src));
}

// ------------------------------------------------ maps of strings
//
// A map is not on the arena and owns a copy of every string it holds, so a
// string made in an iteration (or in a def) and stored in a map made
// outside survives the iteration's release, and a read hands out a new
// string the loop then frees. Four keys stay live, so the map's own memory
// is bounded, and ten times the passes must not grow the process.

/// Strings of `n` bytes stored into maps made outside the loop: under string
/// keys, as string values, through a def that stores into a map it is
/// given, and read back into loop variables.
fn map_program(n: usize, steps: usize) -> String {
    format!(
        r#"
N = {n}
STEPS = {steps}
KEEP = 4
def fill(m: Map, k) {{
    s = "d" * N + str(k)
    m[s[N:]] = s
    return len(m)
}}
def fresh(k) {{
    m = map()
    m[k] = "fresh " + str(k)
    return m
}}
byname = map()
bynum = map()
given = map()
last = ""
total = 0.0
for k to STEPS {{
    line = "x" * N + f"{{k:d}}"
    key = f"key{{mod(k, KEEP):d}}"
    byname[key] = len(line)
    bynum[mod(k, KEEP)] = line
    last = bynum[mod(k, KEEP)]
    total = total + byname[key] + len(last)
    fill(given, mod(k, KEEP))
    f = fresh(k)
    total = total + len(f[k])
    if has(given, "1.0") {{
        x = remove(given, "1.0")
        total = total + len(x)
    }}
}}
show len(byname)
show keys(byname)
show len(bynum[0])
show last[N:]
show keys(given)
show total
"#
    )
}

#[test]
fn maps_of_strings_every_engine_prints_the_same_bytes() {
    let src = map_program(N, STEPS);
    let want = interp(&src);
    assert!(want.contains("['key0', 'key1', 'key2', 'key3']"), "{want}");
    let dir = scratch("maps_same");
    if let Some((out, _, ok)) = native(&dir, &src, &[]) {
        assert!(ok);
        assert_eq!(out, want, "C");
    }
    if let Some((out, _)) = direct(&dir, &src) {
        assert_eq!(out, want, "wasm");
    }
}

#[test]
fn strings_stored_in_maps_in_loops_run_in_bounded_memory() {
    // ten times the passes: each makes and stores strings of 100 KB, which
    // would add well over 64 MB if the maps or the loop kept them
    let dir = scratch("maps_rss");
    if let (Some((_, short, _)), Some((_, long, _))) = (native(&dir, &map_program(N, STEPS), &[]), native(&dir, &map_program(N, 10 * STEPS), &[])) {
        if short != 0 {
            assert!(long < short + 64 * MB, "C peak RSS {short} for {STEPS} steps, {long} for {} steps", 10 * STEPS);
        }
    }
    if let (Some((_, short)), Some((_, long))) = (direct(&dir, &map_program(N, STEPS)), direct(&dir, &map_program(N, 10 * STEPS))) {
        assert!(long < short + 64 * MB, "wasm memory {short} for {STEPS} steps, {long} for {} steps", 10 * STEPS);
    }
}

#[test]
fn no_string_a_map_holds_is_freed_under_address_sanitizer() {
    let dir = scratch("maps_asan");
    let src = map_program(N / 10, STEPS);
    let Some((out, _, ok)) = native(&dir, &src, &["-fsanitize=address", "-g"]) else {
        eprintln!("skip: no AddressSanitizer build here");
        return;
    };
    assert!(ok, "the AddressSanitizer build failed at run time");
    assert_eq!(out, interp(&src));
}
