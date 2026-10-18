// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Engine parity on programs that go wrong.
//!
//! Every program here runs in BOTH engines -- `viv run` (the interpreter) and
//! Viv -> C -> native -- and the test requires the same stdout, the same
//! stderr and the same exit code from each. The programs are the ones the
//! engines used to disagree on: an index out of bounds (the C read or wrote
//! memory it did not own and carried on), and a failed assert (the summary
//! went to stdout natively and to stderr in the interpreter).
//!
//! Also here: run-time errors carry a source location, `--json` diagnostics,
//! and the notes for Python habits (`**`, `range`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn have_cc() -> bool {
    Command::new(cc()).arg("--version").output().is_ok()
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_parity_{}_{tag}", std::process::id()));
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

/// `viv run -q p.viv`, from inside the scratch directory so the location
/// reads `p.viv:line:col`.
fn interp(dir: &Path, args: &[&str]) -> Ran {
    ran(Command::new(viv_bin()).current_dir(dir).arg("run").arg("-q").args(args).arg("p.viv").output().unwrap())
}

/// `viv p.viv -o p.c`, `cc -O2 [flags] p.c -lm`, run it.
fn native(dir: &Path, flags: &[&str]) -> Ran {
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2"]).args(flags).args(["p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    ran(Command::new(dir.join("p")).output().unwrap())
}

/// Run `src` in both engines, require them to agree on everything, and
/// return what they did.
fn both(tag: &str, src: &str) -> Option<Ran> {
    if !have_cc() {
        return None;
    }
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let i = interp(&dir, &[]);
    let n = native(&dir, &[]);
    assert_eq!(i, n, "the engines disagree on\n{src}");
    Some(i)
}

/// A program that stops at run time: exit 1, and stderr is exactly `err`.
fn stops(tag: &str, src: &str, out: &str, err: &str) {
    let Some(r) = both(tag, src) else { return };
    assert_eq!(r.code, Some(4), "{r:?}");
    assert_eq!(r.out, out);
    assert_eq!(r.err, err);
}

// ---------------------------------------------------------------- 1. bounds

#[test]
fn a_negative_index_counts_from_the_end_and_past_the_start_stops() {
    stops(
        "neg",
        "a = [1.0, 2.0]\nshow a[-1]\nshow a[-2]\nshow a[-3]\nshow 99\n",
        "2\n1\n",
        "error: index -3 out of bounds for dimension 0 (size 2)\n  --> p.viv:4:6\n   4 | show a[-3]\n            ^^^^^ here\n",
    );
}

#[test]
fn an_index_past_the_end_stops_both_engines() {
    stops(
        "past",
        "a = [1.0, 2.0]\nshow a[5]\n",
        "",
        "error: index 5 out of bounds for dimension 0 (size 2)\n  --> p.viv:2:6\n   2 | show a[5]\n            ^^^^ here\n",
    );
}

#[test]
fn a_write_out_of_bounds_stops_both_engines() {
    // the location of a store is the whole statement, in both engines
    stops(
        "write",
        "b = zeros(2, 3)\nb[1, 2] = 7\nshow b[1, 2]\ni = 3\nb[1, i] = 8\n",
        "7\n",
        "error: index 3 out of bounds for dimension 1 (size 3)\n  --> p.viv:5:1\n   5 | b[1, i] = 8\n       ^^^^^^^^^^^ here\n",
    );
}

#[test]
fn every_rank_is_checked_in_every_dimension() {
    // rank 1..4, reads and writes, each dimension in turn: each program stops
    // at its bad index, with the dimension and size the interpreter names
    let cases = [
        ("r2get", "a = zeros(2, 3)\nshow a[2, 0]\n", "index 2 out of bounds for dimension 0 (size 2)"),
        ("r2getj", "a = zeros(2, 3)\nshow a[0, 3]\n", "index 3 out of bounds for dimension 1 (size 3)"),
        ("r3get", "a = zeros(2, 3, 4)\nshow a[1, 2, 4]\n", "index 4 out of bounds for dimension 2 (size 4)"),
        ("r3set", "a = zeros(2, 3, 4)\na[1, -4, 0] = 1\n", "index -4 out of bounds for dimension 1 (size 3)"),
        ("r4get", "a = zeros(2, 2, 2, 2)\nshow a[1, 1, 1, 2]\n", "index 2 out of bounds for dimension 3 (size 2)"),
        ("r4set", "a = zeros(2, 2, 2, 2)\na[0, 0, 5, 0] = 1\n", "index 5 out of bounds for dimension 2 (size 2)"),
        ("r1set", "a = zeros(4)\na[4] = 1\n", "index 4 out of bounds for dimension 0 (size 4)"),
    ];
    for (tag, src, msg) in cases {
        let Some(r) = both(tag, src) else { return };
        assert_eq!(r.code, Some(4), "{tag}: {r:?}");
        assert!(r.err.starts_with(&format!("error: {msg}\n  --> p.viv:2:")), "{tag}: {}", r.err);
    }
}

#[test]
fn with_two_bad_indices_both_engines_name_the_first_dimension() {
    let Some(r) = both("twobad", "a = zeros(2, 3)\nshow a[9, 9]\n") else { return };
    assert!(r.err.starts_with("error: index 9 out of bounds for dimension 0 (size 2)\n"), "{}", r.err);
}

#[test]
fn complex_arrays_are_checked() {
    let Some(r) = both("cget", "z = czeros(3)\nz[1] = 2.0 + 1.0i\nshow z[1]\nshow z[3]\n") else { return };
    assert_eq!(r.code, Some(4));
    assert!(r.err.starts_with("error: index 3 out of bounds for dimension 0 (size 3)\n  --> p.viv:4:6\n"), "{}", r.err);
    let Some(r) = both("cset2", "z = czeros(2, 2)\nz[0, -3] = 1.0i\n") else { return };
    assert!(r.err.starts_with("error: index -3 out of bounds for dimension 1 (size 2)\n"), "{}", r.err);
}

#[test]
fn a_function_registry_is_checked_like_an_array() {
    // a negative index counts from the end, as for any array; past the
    // start it stops
    let src = "def one(x) { return x }\ndef two(x) { return 2 * x }\nms = [one, two]\nshow apply(ms[1], 3)\nshow apply(ms[-2], 3)\nshow apply(ms[-3], 3)\n";
    stops(
        "fnarr",
        src,
        "6\n3\n",
        "error: index -3 out of bounds for dimension 0 (size 2)\n  --> p.viv:6:12\n   6 | show apply(ms[-3], 3)\n                  ^^^^^^ here\n",
    );
}

#[test]
fn a_repeated_index_expression_reports_its_own_line() {
    // `a[i]` twice is one IR node unless its site is part of the node; the
    // C must name line 5, where it failed, not line 2, where it first appears
    let src = "a = [1.0, 2.0]\ni = 0\nshow a[i]\ni = 2\nshow a[i]\n";
    let Some(r) = both("repeat", src) else { return };
    assert!(r.err.contains("  --> p.viv:5:6\n"), "{}", r.err);
}

#[test]
fn an_error_inside_a_def_is_located_in_the_def() {
    let src = "def last(v: Arr) {\n    return v[len(v)]\n}\nshow last([1.0, 2.0, 3.0])\n";
    let Some(r) = both("indef", src) else { return };
    assert!(r.err.starts_with("error: index 3 out of bounds for dimension 0 (size 3)\n  --> p.viv:2:12\n   2 |     return v[len(v)]\n"), "{}", r.err);
}

#[test]
fn an_error_in_a_loop_keeps_the_output_before_it() {
    let src = "a = zeros(3)\nfor i to 5 {\n    show i\n    a[i] = i\n}\n";
    let Some(r) = both("loop", src) else { return };
    assert_eq!(r.out, "i: 0\ni: 1\ni: 2\ni: 3\n");
    assert!(r.err.starts_with("error: index 3 out of bounds"), "{}", r.err);
}

#[test]
fn the_wrong_number_of_indices_stops_both_engines_with_the_rank() {
    // The interpreter counts the indices before it reads any of them, so a
    // rank mismatch is reported as one even when an index is also out of
    // bounds (r5). The C used to read a[5] on a 2 x 3 array as a flat index.
    let cases = [
        ("rk1", "a = zeros(2, 3)\nshow 1\nr = a[5]\nshow r\n", "1\n", "rank 2 but was indexed with 1", "  --> p.viv:3:5\n"),
        ("rk2", "a = zeros(2, 3)\nshow a[1]\nshow 7\n", "", "rank 2 but was indexed with 1", "  --> p.viv:2:6\n"),
        ("rk3", "a = zeros(4)\na[1, 1] = 3\n", "", "rank 1 but was indexed with 2", "  --> p.viv:2:1\n"),
        ("rk4", "a = czeros(4)\na[1, 1] = 3\n", "", "rank 1 but was indexed with 2", "  --> p.viv:2:1\n"),
        ("rk5", "a = zeros(4)\na[9, 9] = 3\n", "", "rank 1 but was indexed with 2", "  --> p.viv:2:1\n"),
        ("rk6", "a = zeros(2, 2, 2)\nshow a[0, 0]\n", "", "rank 3 but was indexed with 2", "  --> p.viv:2:6\n"),
        ("rk7", "z = czeros(2, 2)\nshow z[0, 0, 0]\n", "", "rank 2 but was indexed with 3", "  --> p.viv:2:6\n"),
        ("rk8", "a = zeros(2, 2, 2, 2)\na[0, 0, 0] = 1\n", "", "rank 4 but was indexed with 3", "  --> p.viv:2:1\n"),
    ];
    for (tag, src, out, msg, at) in cases {
        let Some(r) = both(tag, src) else { return };
        assert_eq!(r.code, Some(4), "{tag}: {r:?}");
        assert_eq!(r.out, out, "{tag}");
        let head = format!("error: array has {msg} index/indices\n{at}");
        assert!(r.err.starts_with(&head), "{tag}: {}", r.err);
    }
}

#[test]
fn the_accessors_add_no_c_warnings() {
    // A program that uses one accessor pulls in its plain and `_at` forms;
    // neither may be reported unused, and a release build (no checks) must
    // not leave the site parameter unused.
    if !have_cc() {
        return;
    }
    let dir = scratch("warn");
    std::fs::write(dir.join("p.viv"), "a = zeros(3)\na[1] = 2\nshow a[1]\nz = czeros(2, 2)\nz[0, 1] = 1.0i\nshow z[0, 1]\n").unwrap();
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success());
    for flags in [&[][..], &["-DVIV_RELEASE"][..]] {
        let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", "-Wall", "-Wextra", "-O2"]).args(flags).args(["p.c", "-lm", "-o", "p"]).output().unwrap();
        assert!(b.status.success());
        let err = String::from_utf8_lossy(&b.stderr);
        for name in ["viv_get", "viv_set", "viv_cget", "viv_cset", "viv_ix", "viv_rank", "'site'"] {
            assert!(!err.contains(name), "{flags:?}: {err}");
        }
    }
}

#[test]
fn eig_sym_gives_the_same_bits_at_every_optimization_level() {
    // clang -O2 merges cos(x) and sin(x) into one sincos call whose last bit
    // can differ; so does an optimized Rust build. Both engines now make the
    // two plain calls, so the interpreter (built either way) and the C at -O0
    // and -O2 agree. This SCF loop is the one that showed it: iteration 18's
    // energy change differed in the 10th digit.
    if !have_cc() {
        return;
    }
    let ex = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/lenses/scf_demo.viv");
    let dir = scratch("sincos");
    std::fs::copy(&ex, dir.join("p.viv")).unwrap();
    let strip = |s: String| s.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect::<String>();
    let i = interp(&dir, &[]);
    let o0 = native(&dir, &["-O0"]);
    let o2 = native(&dir, &[]);
    assert_eq!(strip(i.out), strip(o0.out.clone()));
    assert_eq!(o0.out, o2.out);
    // and the fft's twiddles and Bluestein chirp (22 = 2 x 11), the other
    // places a cos sits next to a sin
    std::fs::write(dir.join("p.viv"), "x = czeros(22)\nfor i to 22 {\n    x[i] = sin(0.37 * i * i)\n}\nfft(x)\nshow x\n").unwrap();
    let i = interp(&dir, &[]);
    let o0 = native(&dir, &["-O0"]);
    let o2 = native(&dir, &[]);
    assert_eq!(i.out, o0.out);
    assert_eq!(o0.out, o2.out);
}

// ---------------------------------------------------------------- 1b. non-integer indices

#[test]
fn a_fractional_index_is_an_error_in_both_engines() {
    // an index must be exactly an integer:
    // a fraction stops the run where it used to be cut toward zero
    for (tag, ix, shown) in [("frac", "1.7", "1.7"), ("negfrac", "-0.5", "-0.5"), ("near", "2.999", "2.999")] {
        let src = format!("a = [10.0, 20.0, 30.0]\nshow a[0]\nshow a[{ix}]\n");
        let Some(r) = both(tag, &src) else { return };
        assert_eq!((r.code, r.out.as_str()), (Some(4), "10\n"), "{r:?}");
        assert!(r.err.starts_with(&format!("error: index {shown} is not an integer\n")), "{}", r.err);
    }
}

#[test]
fn nan_and_huge_indices_read_as_the_interpreter_reads_them() {
    // NaN is not an integer; a huge double is one, and saturates as `as i64`
    // does: +huge is i64::MAX, -huge i64::MIN. (int) of those is undefined
    // behavior in C; viv_ix does what Rust does.
    let Some(r) = both("nan", "a = [10.0, 20.0]\nk = 0 / 0\nshow a[k]\n") else { return };
    assert_eq!((r.code, r.out.as_str()), (Some(4), ""));
    assert!(r.err.starts_with("error: index nan is not an integer\n"), "{}", r.err);
    let Some(r) = both("huge", "a = [10.0, 20.0]\nshow a[1e300]\n") else { return };
    assert!(r.err.starts_with("error: index 9223372036854775807 out of bounds for dimension 0 (size 2)\n"), "{}", r.err);
    let Some(r) = both("nhuge", "a = [10.0, 20.0]\nshow a[-1e300]\n") else { return };
    assert!(r.err.starts_with("error: index -9223372036854775808 out of bounds for dimension 0 (size 2)\n"), "{}", r.err);
}

// ---------------------------------------------------------------- 1c. -DVIV_RELEASE

#[test]
fn viv_release_compiles_the_checks_out_and_changes_nothing_else() {
    if !have_cc() {
        return;
    }
    let dir = scratch("release");
    let src = "a = zeros(3, 3)\nfor i to 3 {\n    for j to 3 {\n        a[i, j] = i * 3 + j\n    }\n}\nshow a\nshow a[2, 1]\nshow a[-1, -2]\n";
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let debug = native(&dir, &[]);
    let release = native(&dir, &["-DVIV_RELEASE"]);
    assert_eq!(debug, release);
    let c = std::fs::read_to_string(dir.join("p.c")).unwrap();
    assert!(c.contains("#ifndef VIV_BOUNDS\n"), "{c}");
    assert!(c.contains("#define viv_ix(x, n, k, site) ((void)(site), viv_ix_wrap((int)fmin(fmax((x), -2147483648.0), 2147483647.0), (n)))\n#define viv_rank(have, want, site) ((void)(site))\n"), "{c}");
}

#[test]
fn the_c_without_a_source_still_checks_and_calls_the_plain_accessors() {
    // compile_c (the browser's "show the C") has no file to point at: the
    // access is still checked, and the call reads as it always did
    // (a whole-number index is an integer)
    let c = viv::compile_c("v = [1.0, 2.0]\nshow v[1]\nv[0] = 3\n").unwrap();
    assert!(c.contains("viv_lget1(v, 1LL, 0)"), "{c}");
    assert!(!c.contains("viv_lget1_at(v,") && !c.contains("viv_get1_at(v,"), "{c}");
    assert!(c.contains("VIV_ACCESS int viv_ixl("), "{c}");
}

#[test]
fn node_ids_do_not_change_without_sites() {
    // `viv hash` and the lens stamps hash the IR; the site field must not
    // move them. Two identical `a[i]` still share one node outside a C compile.
    let src = "a = [1.0, 2.0]\ni = 0\nshow a[i]\nshow a[i]\n";
    let plain = viv::build_ir(src, knotkit::span::SourceId(0)).unwrap();
    let sited = viv::lower::with_sites("p.viv", src, || viv::build_ir(src, knotkit::span::SourceId(0)).unwrap());
    // the second a[i], and the `show` holding it, are nodes of their own only with sites
    assert_eq!(sited.len(), plain.len() + 2);
}

// ---------------------------------------------------------------- 2. failed asserts

#[test]
fn the_failed_assert_summary_is_on_stderr_in_both_engines() {
    let Some(r) = both("assert", "assert_near(1.0, 2.0, 0.1)\nshow 3\ntest \"t\" {\n    assert_near(1.0, 1.5, 0.1)\n}\n") else { return };
    assert_eq!(r.code, Some(1));
    assert_eq!(r.out, "assert_near FAIL: |1 - 2| = 1 > 0.1\n3\nFAIL t -- assert_near: |1 - 1.5| = 0.5 > 0.1\n");
    assert_eq!(r.err, "2 assertions FAILED\n");
}

#[test]
fn a_run_that_stops_prints_the_error_and_no_summary() {
    let Some(r) = both("assertstop", "assert_near(1.0, 2.0, 0.1)\na = [1.0]\nshow a[1]\n") else { return };
    assert_eq!(r.code, Some(4));
    assert!(r.err.starts_with("error: index 1 out of bounds"), "{}", r.err);
    assert!(!r.err.contains("FAILED"), "{}", r.err);
}

#[test]
fn both_engines_sh_compares_the_error_and_the_summary() {
    if !have_cc() {
        return;
    }
    let dir = scratch("be");
    std::fs::write(dir.join("oob.viv"), "a = [1.0, 2.0]\nshow a[0]\nshow a[-3]\n").unwrap();
    std::fs::write(dir.join("fails.viv"), "test \"no\" {\n    assert_near(1.0, 2.0, 0.1)\n}\n").unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("sh").arg(root.join("tools/both-engines.sh")).arg(&dir).env("VIV", viv_bin()).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    // both fail (an error, a failed test), and in both the engines agree
    assert_eq!(out.status.code(), Some(1), "{text}");
    for name in ["oob", "fails"] {
        assert!(text.lines().any(|l| l.starts_with(name) && l.ends_with("identical output")), "{text}");
    }
}

// ---------------------------------------------------------------- 3. locations, --json, hints

#[test]
fn a_front_end_error_under_viv_run_has_its_location() {
    let dir = scratch("parse");
    std::fs::write(dir.join("p.viv"), "x = 1\ny = x + * 2\n").unwrap();
    let r = interp(&dir, &[]);
    assert_eq!(r.code, Some(3));
    assert!(r.err.contains("  --> p.viv:2:"), "{}", r.err);
}

#[test]
fn viv_run_json_reports_the_error_as_data() {
    let dir = scratch("json");
    std::fs::write(dir.join("p.viv"), "a = [1.0, 2.0]\nshow a[-3]\n").unwrap();
    // the envelope on stdout, the program's printing in result.stdout
    let r = interp(&dir, &["--json"]);
    assert_eq!(r.code, Some(4));
    assert!(r.err.is_empty(), "{}", r.err);
    assert!(
        r.out.contains(concat!(
            r#""diagnostics":[{"severity":"error","rule":"R-index","message":"index -3 out of bounds for dimension 0 (size 2)","#,
            r#""file":"p.viv","line":2,"col":6,"end_line":2,"end_col":11,"label":"here","source":"show a[-3]","#,
            r#""notes":[]}]"#
        )),
        "{}",
        r.out
    );
    // a clean run says so: an empty list, and exit 0
    std::fs::write(dir.join("p.viv"), "show 1\n").unwrap();
    let r = interp(&dir, &["--json"]);
    assert_eq!(r.code, Some(0));
    assert!(r.out.starts_with(r#"{"viv":1,"cmd":"run","ok":true,"exit":0,"diagnostics":[],"result":{"file":"p.viv","stdout":"1\n""#), "{}", r.out);
    // the failed-assert summary is a diagnostic with no location
    std::fs::write(dir.join("p.viv"), "assert_near(1.0, 2.0, 0.1)\n").unwrap();
    let r = interp(&dir, &["--json"]);
    assert_eq!(r.code, Some(1));
    assert!(r.out.contains(r#""rule":"T-fail","message":"1 assertion FAILED","file":null,"line":null,"col":null"#), "{}", r.out);
}

#[test]
fn a_compile_with_json_reports_front_end_errors_as_data() {
    let dir = scratch("cjson");
    std::fs::write(dir.join("p.viv"), "x = 2 ** 3\n").unwrap();
    let o = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "--json", "-o", "p.c"]).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success());
    assert!(err.starts_with(r#"{"diagnostics":[{"severity":"error","rule":"H-pow","message":"expected an expression","file":"p.viv","line":1,"col":8,"#), "{err}");
    assert!(err.contains("write `x ^ y` (or `pow(x, y)`)"), "{err}");
}

/// The text of every diagnostic `viv run` and the compiler print for `src`.
fn both_diags(tag: &str, src: &str) -> (String, String) {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let r = interp(&dir, &[]);
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    (r.err, String::from_utf8_lossy(&c.stderr).into_owned())
}

#[test]
fn python_habits_get_a_note_saying_what_to_write() {
    let cases = [
        ("pow", "x = 2 ** 3\nshow x\n", "  note: Viv has no `**` operator: write `x ^ y` (or `pow(x, y)`) for x to the power y\n"),
        ("forin", "for i in range(3) {\n    show i\n}\n", "  note: a Viv loop is `for i to n { ... }`"),
        ("range", "x = range(3)\nshow x\n", "  note: Viv has no `range`: loop with `for i to n { ... }`"),
    ];
    for (tag, src, note) in cases {
        let (run, compile) = both_diags(tag, src);
        assert!(run.contains(note), "viv run, {tag}:\n{run}");
        assert!(compile.contains(note), "compile, {tag}:\n{compile}");
        assert!(run.contains("  --> p.viv:1:"), "viv run, {tag}:\n{run}");
    }
}

/// The generated C names a program by its file name, not by the absolute
/// path it was given: that C is hashed into the wasm manifests and compiled
/// into served modules, so it must not depend on the machine or checkout
/// (dft/build-wasm.sh and engines/build-wasm.mjs pass absolute paths). The
/// interpreter, given the same absolute path, prints the same location.
#[test]
fn an_absolute_path_is_named_by_its_file_name_in_both_engines() {
    if !have_cc() {
        return;
    }
    let src = "a = zeros(2)\nshow a[2]\n";
    let (d1, d2) = (scratch("abs1"), scratch("abs2"));
    let mut cs = Vec::new();
    for d in [&d1, &d2] {
        let p = d.join("p.viv");
        std::fs::write(&p, src).unwrap();
        // run from the other directory, so no path is relative to the cwd
        let o = Command::new(viv_bin()).current_dir(if d == &d1 { &d2 } else { &d1 }).arg(&p).arg("-o").arg(d.join("p.c")).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        cs.push(std::fs::read_to_string(d.join("p.c")).unwrap());
    }
    assert_eq!(cs[0], cs[1], "the C depends on the directory the program is in");
    let dir = d1.display().to_string();
    assert!(!cs[0].contains(&dir), "the C names the local path {dir}");
    // both engines, by absolute path: the same location, `p.viv:2:6`
    let run = ran(Command::new(viv_bin()).arg("run").arg("-q").arg(d1.join("p.viv")).output().unwrap());
    let b = Command::new(cc()).current_dir(&d1).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success());
    let nat = ran(Command::new(d1.join("p")).output().unwrap());
    assert_eq!(run, nat);
    assert!(run.err.contains("  --> p.viv:2:6\n"), "{}", run.err);
}

/// `viv run prog.viv`, a bare file name with no `./`, finds the modules under
/// the repository root (the nearest ancestor holding `Cargo.toml`), as
/// `lib/params` is found for a program in `study/idioms/ddpm`. `Path::parent`
/// of a bare name is `Some("")`, which the lookup used to take as the
/// program's directory: canonicalizing it failed and the root search found
/// nothing. An empty parent is now the current directory.
#[test]
fn a_bare_file_name_finds_the_modules_under_the_repository_root() {
    let dir = scratch("bare");
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    std::fs::write(dir.join("lib/m.viv"), "def twice(x) { return 2 * x }\n").unwrap();
    std::fs::write(dir.join("sub/p.viv"), "from m import twice\nshow twice(21)\n").unwrap();
    let sub = dir.join("sub");
    let r = interp(&sub, &[]);
    assert_eq!((r.code, r.out.as_str()), (Some(0), "42\n"), "{}", r.err);
    if have_cc() {
        assert_eq!(native(&sub, &[]).out, "42\n");
    }
}

// ---------------------------------------------------------------- 4. syntax

#[test]
fn a_number_followed_by_a_name_is_an_error_that_says_to_multiply() {
    let src = "x = 3.0\ny = 2x\nshow y\n";
    let (run, compile) = both_diags("juxt", src);
    for (who, err) in [("viv run", &run), ("compile", &compile)] {
        assert!(err.contains("error: a number cannot be followed directly by a name: `2x`"), "{who}:\n{err}");
        assert!(err.contains("  --> p.viv:2:5"), "{who}:\n{err}");
        assert!(err.contains("  note: Viv has no implicit multiplication: write `2 * x`"), "{who}:\n{err}");
    }
    // the number's own suffixes still lex: an exponent and an imaginary unit
    let r = both("juxtok", "a = 1e3\nb = 2.5i\nshow a\nshow b\n").unwrap_or(Ran { code: Some(0), out: "a: 1000\nb: 0+2.5i\n".into(), err: String::new() });
    assert_eq!((r.code, r.out.as_str()), (Some(0), "a: 1000\nb: 0+2.5i\n"), "{}", r.err);
}

#[test]
fn a_name_may_use_unicode_letters() {
    let Some(r) = both("uni", "Δx = 0.5\nσ = 2\nshow Δx * σ\ndef gaußian(x) { return x + σ }\nshow gaußian(Δx)\n") else { return };
    assert_eq!((r.code, r.out.as_str()), (Some(0), "1\n2.5\n"), "{}", r.err);
    // a stray non-ASCII character is one error, with the whole character
    let (run, _) = both_diags("unistray", "x = 1 · 2\n");
    assert!(run.contains("unexpected character '·'"), "{run}");
}

#[test]
fn compound_assignment_on_an_element_evaluates_each_index_once() {
    let src = "srand(7)\nfirst = rand()\nsecond = rand()\nsrand(7)\nc = zeros(10)\nc[floor(10 * rand())] += 1\nshow rand() == second\nshow c[floor(10 * first)]\nshow sum(c)\n\
               a = zeros(2, 3)\na[1, 2] += 5\na[1, 2] *= 3\na[1, 2] -= 1\na[1, 2] /= 2\nshow a[1, 2]\n";
    let Some(r) = both("idxop", src) else { return };
    assert_eq!((r.code, r.out.as_str()), (Some(0), "true\n1\n1\n7\n"), "{}", r.err);
}

#[test]
fn compound_assignment_out_of_bounds_stops_at_the_statement() {
    stops(
        "idxoob",
        "a = zeros(2)\na[2] += 1\n",
        "",
        "error: index 2 out of bounds for dimension 0 (size 2)\n  --> p.viv:2:1\n   2 | a[2] += 1\n       ^^^^ here\n",
    );
}

#[test]
fn for_from_to_by_reversed_walk_the_values_the_reference_names() {
    let src = "for i from 2 to 5 {\n    print(i)\n}\nfor i from 10 to 0 by -3 {\n    print(i)\n}\nfor t to 3 reversed {\n    print(t)\n}\n\
               for i from 1 to 10 by 4 reversed {\n    print(i)\n}\nfor i from 0 to 1 by 0.25 {\n    print(i)\n}\nfor i from 4 to 4 {\n    print(99)\n}\n\
               n = 2\nfor i from 0 to n {\n    n = 10\n    print(i)\n}\n";
    let Some(r) = both("forr", src) else { return };
    assert_eq!(
        (r.code, r.out.as_str()),
        (Some(0), "2\n3\n4\n10\n7\n4\n1\n2\n1\n0\n9\n5\n1\n0\n0.25\n0.5\n0.75\n0\n1\n"),
        "{}",
        r.err
    );
}

#[test]
fn for_to_reads_its_bound_once_and_counts_its_own_passes() {
    // The C backend compiled `for i to n` to `for (i = 0.0; i < n; i = i + 1.0)`:
    // n re-read each pass (a body growing n never stopped; the bound's call
    // ran each pass), i the counter (assigning it skipped passes), and i left
    // at n after the loop. The interpreter and the wasm backend read the
    // bound once and count with a counter of their own. `parallel for` also
    // re-read its bound without OpenMP, and truncated it: 2 passes for 2.5.
    let src = "def count_calls(calls) -> Num {\n    calls[0] += 1\n    return 3\n}\n\
               n = 3\np = 0\nfor i to n {\n    n += 1\n    p += 1\n    if p > 10 {\n        break\n    }\n}\nshow p\n\
               calls = zeros(1)\nfor i to count_calls(calls) {\n}\nshow calls[0]\n\
               for j to 4 {\n    j = j + 1\n    print(j)\n}\n\
               for i to 5 {\n}\nshow i\nk = -1\nfor k to 0 {\n}\nshow k\n\
               m = 3\nt = 0\nfor a to m {\n    for b to m {\n        m += 1\n        t += 1\n    }\n    if t > 100 {\n        break\n    }\n}\nshow t\n\
               x = 2.5\nh = zeros(4)\nparallel for i to x {\n    h[i] = 1\n}\nshow sum(h)\n";
    let Some(r) = both("forbound", src) else { return };
    assert_eq!(
        (r.code, r.out.as_str()),
        (Some(0), "p: 3\n1\n1\n2\n3\n4\ni: 4\nk: -1\nt: 21\n3\n"),
        "{}",
        r.err
    );
}

#[test]
fn a_for_step_of_zero_is_an_error() {
    // written as a number: at compile time, in both
    let (run, compile) = both_diags("by0", "for i from 0 to 3 by 0 {\n    print(i)\n}\n");
    for err in [&run, &compile] {
        assert!(err.contains("error: a `for` step of 0 never reaches the end\n  --> p.viv:1:22"), "{err}");
    }
    // known only at run time: the run stops there, in both, with the location
    stops(
        "by0run",
        "print(1)\ns = 0\nfor i from 0 to 3 by s {\n    print(i)\n}\n",
        "1\n",
        "error: a `for` step of 0 never reaches the end\n  --> p.viv:3:22\n   3 | for i from 0 to 3 by s {\n                            ^ here\n",
    );
    // parallel for keeps its one form
    let (run, _) = both_diags("parby", "parallel for i from 1 to 3 {\n    print(i)\n}\n");
    assert!(run.contains("a `parallel for` counts from 0 by 1"), "{run}");
}

// ---------------------------------------------------------------- 5. one verdict

/// `viv run` and the compile print the same front-end diagnostics, byte for
/// byte, and both fail.
fn same_refusal(tag: &str, src: &str) -> String {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let r = interp(&dir, &[]);
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    let cerr = String::from_utf8_lossy(&c.stderr).into_owned();
    assert_eq!(r.code, Some(3), "viv run accepted {tag}:\n{}", r.out);
    assert!(!c.status.success(), "the compile accepted {tag}");
    assert_eq!(r.err, cerr, "{tag}: the two refusals differ");
    assert_eq!(r.out, "", "{tag}: viv run printed before refusing");
    r.err
}

#[test]
fn a_variable_named_like_a_def_from_a_module_is_another_name() {
    // makemore's `block_size` against lib/schur's `def block_size`: the C
    // compile used to fail in cc, and the interpreter ran it; with module
    // namespaces they are two names, in both engines
    let dir = scratch("collide");
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    std::fs::write(dir.join("lib/schur.viv"), "def block_size(T, k) {\n    return k + 1\n}\ndef grow(k) {\n    return block_size(0, k)\n}\n").unwrap();
    std::fs::write(dir.join("sub/p.viv"), "from schur import grow\nblock_size = 11\nshow block_size\nshow grow(1)\n").unwrap();
    let sub = dir.join("sub");
    let r = interp(&sub, &[]);
    assert_eq!((r.code, r.out.as_str()), (Some(0), "block_size: 11\n2\n"), "{}", r.err);
    if have_cc() {
        assert_eq!(native(&sub, &[]).out, "block_size: 11\n2\n");
    }
    // within one file a def and a top-level variable cannot share a name
    let e = same_refusal("collidedef", "def block_size(T, k) {\n    return k + 1\n}\nblock_size = 11\nshow block_size\n");
    assert!(e.starts_with("error: `block_size` is the name of a def and of a top-level variable\n"), "{e}");
    assert!(e.contains("^^^^^^^^^^ the variable"), "{e}");
    assert!(e.contains("---------- the def"), "{e}");
    // a loop variable collides too
    let e = same_refusal("collidefor", "def f(x) {\n    return x\n}\nfor f to 3 {\n    show f\n}\n");
    assert!(e.contains("`f` is the name of a def and of a top-level variable"), "{e}");
}

#[test]
fn a_loop_variable_later_given_an_array_is_refused_by_both() {
    // ddpm: `for t to T`, then `t = draw_steps(B)`
    let e = same_refusal("retype", "for t to 3 {\n    show t\n}\nt = zeros(2)\nshow t\n");
    assert!(e.contains("`t` is a number but is assigned a"), "{e}");
}

#[test]
fn an_unknown_function_in_a_branch_never_taken_is_refused_by_both() {
    let e = same_refusal("unknownfn", "def f(x) {\n    if x < 0.0 { error(\"negative\") }\n    return sqrt(x)\n}\nshow f(4.0)\n");
    assert!(e.starts_with("error: call to unknown function `error`\n  --> p.viv:2:18"), "{e}");
}

#[test]
fn recursion_stops_at_the_same_depth_with_the_same_error_in_both() {
    let def = "def depth(n) {\n    if n == 0.0 {\n        return 0.0\n    }\n    return 1.0 + depth(n - 1.0)\n}\n";
    // 2000 deep: the interpreter used to overflow its stack here
    let Some(r) = both("rec2000", &format!("{def}show depth(2000.0)\n")) else { return };
    assert_eq!((r.code, r.out.as_str()), (Some(0), "2000\n"), "{}", r.err);
    // the limit: depth(n) makes n + 1 nested calls
    let max = viv::recursion::MAX_DEPTH as f64;
    let Some(r) = both("recmax", &format!("{def}show depth({})\n", max - 1.0)) else { return };
    assert_eq!(r.out, format!("{}\n", max - 1.0));
    stops(
        "recpast",
        &format!("{def}print(1)\nshow depth({max})\n"),
        "1\n",
        &format!("error: recursion deeper than {} nested calls\n  --> p.viv:1:5\n   1 | def depth(n) {{\n           ^^^^^ here\n", viv::recursion::MAX_DEPTH),
    );
}

#[test]
fn the_recursion_count_keeps_every_way_out_of_a_def() {
    // mutual recursion, a docstring, an array result, a def that falls off
    // its end, and returns inside loops: the count must come back to 0, so
    // the same recursion run many times over never reaches the limit
    let src = "def even(n) {\n    \"n is even\"\n    if n == 0.0 {\n        return true\n    }\n    return odd(n - 1.0)\n}\n\
               def odd(n) {\n    if n == 0.0 {\n        return false\n    }\n    return even(n - 1.0)\n}\n\
               def ramp(n) -> Arr {\n    if n == 0.0 {\n        return zeros(1)\n    }\n    a = ramp(n - 1.0)\n    a[0] += n\n    return a\n}\n\
               def fill_up(a: Arr, n) {\n    if n > 0.0 {\n        a[n - 1.0] = n\n        fill_up(a, n - 1.0)\n    }\n}\n\
               def first_over(n, lim) {\n    for i to 10 {\n        if i * n > lim {\n            return i\n        }\n    }\n    return first_over(n + 1.0, lim)\n}\n\
               c = 0\nfor k to 5000 {\n    if even(10.0) {\n        c += 1\n    }\n}\nshow c\nshow odd(7.0)\nshow ramp(100.0)[0]\n\
               b = zeros(4)\nfor k to 2000 {\n    fill_up(b, 4.0)\n}\nshow b\nshow first_over(1.0, 50.0)\n";
    let Some(r) = both("reckeep", src) else { return };
    assert_eq!((r.code, r.out.as_str()), (Some(0), "c: 5000\ntrue\n5050\nb: [1, 2, 3, 4]\n9\n"), "{}", r.err);
}
