// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The error boundary and output routing of an exported program
//! (lang/viv/src/embed.c).
//!
//! A program build behaves as it always has: errors to stderr and exit
//! status 1, with the failed-assert summary only where the site used
//! `exit`. Built with `-DVIV_EMBED`, the same C returns from a run with the
//! error's kind, its message and its source line, sends every printed line
//! to a callback, records each test block in a table, stops at the next loop
//! pass after an interrupt, and refuses to compile a call of `exit`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("viv_embed_{}_{tag}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// The program's C, with its source locations, named `prog.viv`.
fn c_of(src: &str) -> String {
    viv::lower::with_sites("prog.viv", src, || viv::compile_c(src)).expect("compiles to C")
}

/// A driver that runs the program `runs` times through `viv_embed_run` and
/// prints what came back: each line as `L<stream>:<text>`, the status, the
/// message and the test table. `extra` is C run from the line callback.
fn driver(runs: u32, on_line: &str) -> String {
    format!(
        r#"
#undef main
static int viv_t_lines = 0;
static void viv_t_line(const char* line, long n, int stream, void* ctx) {{
    (void)ctx; (void)n;
    viv_t_lines++;
    fprintf(stdout, "L%d:%s\n", stream, line);
    {on_line}
}}
int main(void) {{
    for (int r = 0; r < {runs}; r++) {{
        char msg[2048];
        int code = viv_embed_run(viv_program_main, viv_t_line, 0, msg, sizeof msg);
        fprintf(stdout, "code %d\n", code);
        if (code) fprintf(stdout, "msg <%s>\n", msg);
        int n = 0;
        viv_embed_test_row* t = viv_embed_take_tests(&n);
        for (int i = 0; i < n; i++) fprintf(stdout, "test <%s> %d <%s>\n", t[i].name, t[i].passed, t[i].detail);
        viv_embed_free_tests(t, n);
        viv_embed_stop_flag = 0;
    }}
    return 0;
}}
"#
    )
}

/// Build `c` (with `-DVIV_EMBED` when `embed`) and run it: (stdout, stderr,
/// exit status). None without a C compiler.
fn build_run(dir: &Path, c: &str, embed: bool) -> Option<(String, String, i32)> {
    let file = dir.join("p.c");
    std::fs::write(&file, c).unwrap();
    let bin = dir.join("p");
    let mut cmd = Command::new(cc());
    cmd.arg("-O1").arg("-w").arg("-ffp-contract=off");
    if embed {
        cmd.arg("-DVIV_EMBED");
    }
    let built = cmd.arg(&file).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(built.status.success(), "the C builds:\n{}", String::from_utf8_lossy(&built.stderr));
    let out = Command::new(&bin).output().unwrap();
    Some((String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned(), out.status.code().unwrap_or(-1)))
}

fn embed_run(tag: &str, src: &str, runs: u32, on_line: &str) -> Option<String> {
    let dir = scratch(tag);
    let c = format!("{}\n{}", c_of(src), driver(runs, on_line));
    let (out, err, status) = build_run(&dir, &c, true)?;
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(status, 0, "{tag}: the caller's process ends normally\nstdout:\n{out}\nstderr:\n{err}");
    assert_eq!(err, "", "{tag}: nothing reaches the process's stderr");
    Some(out)
}

#[test]
fn an_index_error_returns_its_kind_message_and_line_to_the_caller() {
    let src = "print(\"before\")\na = zeros(3)\nshow a[5]\nprint(\"after\")\n";
    let Some(out) = embed_run("index", src, 2, "") else { return };
    let want_run = "L1:before\ncode 2\nmsg <error: index 5 out of bounds for dimension 0 (size 3)\n  --> prog.viv:3:6\n   3 | show a[5]\n            ^^^^ here>\n";
    // twice: the first error left the library ready for the next run
    assert_eq!(out, format!("{want_run}{want_run}"));
}

#[test]
fn a_division_by_zero_and_a_stop_have_their_own_kinds() {
    let Some(out) = embed_run("zerodiv", "x = 0.0\ny = 5.0 % x\nshow y\n", 1, "") else { return };
    assert!(out.starts_with("code 5\nmsg <error: division by zero"), "{out}");
    let Some(out) = embed_run("stop", "stop(\"the deck is empty\")\n", 1, "") else { return };
    assert!(out.starts_with("code 9\nmsg <error: the deck is empty"), "{out}");
}

#[test]
fn printed_lines_go_to_the_callback_and_tests_to_the_table() {
    let src = "print(\"one\")\nshow 2.0\ntest \"passes\" {\n    assert_near(1.0, 1.0, 0.0)\n}\ntest \"fails\" {\n    assert_near(1.0, 2.0, 0.0)\n    assert_near(1.0, 3.0, 0.0)\n}\nprint(\"end\")\n";
    let Some(out) = embed_run("lines", src, 1, "") else { return };
    assert_eq!(
        out,
        "L1:one\nL1:2\nL1:PASS passes\nL1:FAIL fails -- assert_near: |1 - 2| = 1 > 0 (and 1 more)\nL1:end\ncode 0\n\
         test <passes> 1 <>\ntest <fails> 0 <assert_near: |1 - 2| = 1 > 0 (and 1 more)>\n"
    );
}

#[test]
fn an_interrupt_stops_a_silent_loop_at_its_next_pass() {
    // the callback interrupts at the first line; the def's loops print nothing
    let src = "def work(n) {\n    s = 0.0\n    for i to n {\n        for j to n {\n            s = s + 1.0\n        }\n    }\n    return s\n}\nprint(\"start\")\nt = 0.0\nfor k to 1000000 {\n    t = t + work(1000.0)\n}\nshow t\n";
    let Some(out) = embed_run("interrupt", src, 1, "viv_embed_stop_flag = 1;") else { return };
    assert_eq!(out, "L1:start\ncode 8\nmsg <error: the run was interrupted>\n");
}

#[test]
fn a_program_build_keeps_its_exits() {
    let dir = scratch("plain");
    // an exit site: the failed-assert summary follows the message, and the
    // status stays 4, a run-time error, as `viv run` exits
    let a = "assert_near(1.0, 2.0, 0.0)\nx = tridiag_solve([1.0], [1.0, 2.0], [1.0], [1.0])\n";
    let Some((out, err, status)) = build_run(&dir, &c_of(a), false) else { return };
    assert_eq!((out.as_str(), err.as_str(), status), ("assert_near FAIL: |1 - 2| = 1 > 0\n", "tridiag_solve takes four 1-D arrays of one length: (sub, diag, super, rhs)\n1 assertion FAILED\n", 4));
    // an _exit(4) site: no summary
    let b = "assert_near(1.0, 2.0, 0.0)\na = zeros(3)\nshow a[5]\n";
    let (out, err, status) = build_run(&dir, &c_of(b), false).unwrap();
    assert_eq!(out, "assert_near FAIL: |1 - 2| = 1 > 0\n");
    assert_eq!(err, "error: index 5 out of bounds for dimension 0 (size 3)\n  --> prog.viv:3:6\n   3 | show a[5]\n            ^^^^ here\n");
    assert_eq!(status, 4);
    // a failed assert alone: the summary at exit, status 1
    let (_, err, status) = build_run(&dir, &c_of("assert_near(1.0, 2.0, 0.0)\n"), false).unwrap();
    assert_eq!((err.as_str(), status), ("1 assertion FAILED\n", 1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_embed_build_refuses_an_exit_the_runtime_does_not_route() {
    let dir = scratch("poison");
    let c = format!("{}\nint viv_t_bad(void) {{ exit(1); }}\n", c_of("show 1.0\n"));
    let file = dir.join("p.c");
    std::fs::write(&file, c).unwrap();
    let Ok(built) = Command::new(cc()).arg("-c").arg("-DVIV_EMBED").arg(&file).arg("-o").arg(dir.join("p.o")).output() else { return };
    let err = String::from_utf8_lossy(&built.stderr);
    assert!(!built.status.success() && err.contains("poison"), "exit is poisoned under VIV_EMBED:\n{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The runtime text calls no exit and writes to no stream except through
/// embed.c: a new site in any fragment fails here before it can end a
/// caller's process (the `-DVIV_EMBED` build also refuses exit by poison).
#[test]
fn the_runtime_writes_and_stops_only_through_embed_c() {
    // the runtime without its comments, each comment's newlines kept
    let text = viv::lower::runtime_source();
    let mut rt = String::new();
    let mut rest = text.as_str();
    while let Some(k) = rest.find("/*") {
        rt.push_str(&rest[..k]);
        let end = rest[k..].find("*/").map_or(rest.len(), |e| k + e + 2);
        rt.extend(rest[k..end].chars().filter(|&c| c == '\n'));
        rest = &rest[end..];
    }
    rt.push_str(rest);
    let mut bad = Vec::new();
    for (i, line) in rt.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for word in ["exit(", "printf(", "putchar(", "puts(", "perror(", "atexit("] {
            let mut rest = code;
            while let Some(k) = rest.find(word) {
                let before = rest[..k].chars().last();
                let ok = matches!(before, Some(c) if c.is_alphanumeric() || c == '_');
                if !ok && !(word == "atexit(" && line.contains("viv_fail_exit")) {
                    bad.push(format!("{}: {}", i + 1, line.trim()));
                }
                rest = &rest[k + word.len()..];
            }
        }
        if (code.contains("stderr") || code.contains("stdout")) && !code.contains("viv_fflush(stdout)") && !code.contains(", stdout)") {
            bad.push(format!("{}: {}", i + 1, line.trim()));
        }
    }
    assert!(bad.is_empty(), "runtime lines that bypass viv_stop / viv_printf:\n{}", bad.join("\n"));
}

/// One program for each kind of error (crate::errkind): the interpreter reports the kind, and the exported C returns the
/// same one. Several stops of a kind come from different places (the
/// runtime's checks, a fail helper, a check the parser inserted, a builtin
/// of the interpreter's), so a kind given at one site and not its twin shows.
const KIND_CASES: &[(&str, viv::errkind::ErrKind, &str)] = {
    use viv::errkind::ErrKind::*;
    &[
        ("an index past the end", Index, "a = zeros(3)\nk = 5.0\nshow a[k]\n"),
        ("a string index", Index, "s = \"abc\"\nk = 7.0\nshow s[k]\n"),
        ("a slice past the end", Index, "a = zeros(3)\nk = 9.0\nb = a[1:k]\nshow b\n"),
        ("take past the end", Index, "a = zeros(3)\nb = take(a, [5.0])\nshow b\n"),
        ("a shape mismatch", Value, "a = zeros(3)\nb = zeros(2)\nshow a + b\n"),
        ("a for step of 0", Value, "s = 0.0\nfor i from 0 to 3 by s {\n    show i\n}\n"),
        ("a reshape that cannot be", Value, "show reshape(zeros(6), 4.0)\n"),
        ("not positive definite", Value, "A = [[1.0, 2.0], [2.0, 1.0]]\nL = cholesky(A)\nshow L\n"),
        ("an unpack of the wrong length", Value, "x, y = [1.0, 2.0, 3.0]\nshow x\n"),
        ("not an int", Value, "show int(\"12x\")\n"),
        ("not a float", Value, "x = float(\"1e\")\nshow x\n"),
        ("an array past the limit", Overflow, "n = 100000.0\na = zeros(n, n)\nshow a[0, 0]\n"),
        ("an Int past 64 bits", Overflow, "def big(n: Int) -> Int {\n    return n + 1\n}\nprint(big(9223372036854775807))\n"),
        ("a remainder by zero", ZeroDiv, "x = 0.0\nshow 5.0 % x\n"),
        ("an Int division by zero", ZeroDiv, "a = 5\nb = 0\nshow a // b\n"),
        ("the recursion limit", Recursion, "def f(n) {\n    return f(n + 1.0)\n}\nshow f(0.0)\n"),
        ("stop", Stop, "stop(\"the deck is empty\")\n"),
        ("stop with a computed message", Stop, "m = \"the deck \" + \"is empty\"\nstop(m)\n"),
        ("a missing key", Error, "m = map()\nm[1] = 2\nshow m[3]\n"),
        ("a missing file", Error, "s = read_text(\"/nonexistent/viv_kinds.txt\")\nshow s\n"),
    ]
};

#[test]
fn every_kind_is_the_same_in_the_interpreter_and_the_exported_c() {
    for &(what, kind, src) in KIND_CASES {
        // the interpreter recurses on the native stack: room for the limit
        let src_owned = src.to_string();
        let run = std::thread::Builder::new()
            .stack_size(512 << 20)
            .spawn(move || viv::interp::run_diagnosed(&src_owned, false))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(run.kind, Some(kind), "{what}: the interpreter's kind ({:?})", run.errors.first().map(|d| d.message.clone()));
        let Some(out) = embed_run("kind", src, 1, "") else { return };
        assert!(out.contains(&format!("code {}\nmsg <", kind.code())), "{what}: the C returns {kind:?} ({})\n{out}", kind.code());
    }
}
