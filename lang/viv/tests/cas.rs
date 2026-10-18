// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The computer algebra system and its first client:
//! lib/cas, the derivative table it shares with grad.rs, and
//! `viv check --phrases` (crate::cas_check).
//!
//! The library's own tests run with the library (`tools/three-engines.sh
//! lib/cas`, and `lib/cas/check` for the acceptance programs). These check
//! what sits between the library and the compiler: that the generated files
//! are what their sources give, that grad.rs and the CAS read one table and
//! agree on it, that the CAS's LaTeX is tex.rs's, and what the phrase check
//! refuses.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// A program that uses lib/cas, run in the interpreter.
fn run_cas(src: &str) -> String {
    let text = viv::expand_uses(src, &mut |name: &str| viv::cas_check::lib_module(name).map(str::to_string).ok_or_else(|| format!("no module {name}"))).unwrap();
    // on a thread with the stack `viv run` gives a program (a test thread's is small)
    std::thread::Builder::new().stack_size(1 << 30).spawn(move || viv::interp::run(&text)).unwrap().join().unwrap().unwrap_or_else(|e| panic!("{e}"))
}

fn no_modules(name: &str) -> Result<String, String> {
    Err(format!("no module {name}"))
}

/// `viv check --phrases` of `src`, as the CLI runs it: the checks and the records.
fn check(src: &str) -> (Vec<viv::cas_check::Check>, Vec<viv::cas_check::Record>) {
    let mut user = no_modules;
    let mut resolve = viv::cas_check::resolver(&mut user);
    let (program, checks) = viv::cas_check::generate(src, &mut resolve).unwrap();
    let (_, recs) = viv::cas_check::run(&program, &mut resolve).unwrap();
    (checks, recs)
}

#[test]
fn the_generated_derivative_module_is_the_table() {
    assert_eq!(read("lib/cas/derivs.viv"), viv::derivs::viv_module(), "lib/cas/derivs.viv is out of date: run `viv cas derivs > lib/cas/derivs.viv`");
}

#[test]
fn the_committed_check_programs_are_what_the_check_writes() {
    for name in ["acceptance", "pion_loop"] {
        let src = read(&format!("lib/cas/check/{name}.viv"));
        let mut user = no_modules;
        let mut resolve = viv::cas_check::resolver(&mut user);
        let (program, _) = viv::cas_check::generate(&src, &mut resolve).unwrap();
        assert_eq!(read(&format!("lib/cas/check/{name}_check.viv")), program, "lib/cas/check/{name}_check.viv is out of date: run `viv check --phrases --emit lib/cas/check/{name}_check.viv lib/cas/check/{name}.viv`");
    }
}

#[test]
fn every_closed_form_phrase_of_the_acceptance_program_agrees() {
    let (checks, recs) = check(&read("lib/cas/check/acceptance.viv"));
    let agree = recs.iter().filter(|r| r.get("outcome") == "agrees").count();
    let refused: Vec<&str> = recs.iter().filter(|r| r.get("outcome") == "refused").map(|r| r.get("phrase")).collect();
    let disagree: Vec<&str> = recs.iter().filter(|r| r.get("outcome") == "disagrees").map(|r| r.get("phrase")).collect();
    assert!(disagree.is_empty(), "{disagree:?}");
    assert!(agree >= 20, "{agree} agree");
    assert_eq!(refused, vec!["integrate exp(x^3) from 0 to 1", "solve x^5 - x - 1 between 1 and 2"]);
    assert_eq!(checks.len(), recs.len());
    // the owner's example: 7/3 exactly, against Gauss-Legendre's number
    let r = recs.iter().find(|r| r.get("phrase") == "integrate x^2 from 1 to 2").unwrap();
    assert_eq!(r.get("exact"), "7 / 3");
    assert_eq!(r.get("tex"), "\\frac{7}{3}");
    assert_eq!(r.get("numeric"), "2.3333333333333326");
}

#[test]
fn grad_and_the_cas_agree_on_every_row_of_the_table() {
    // one def per head, its gradient by grad.rs, and the CAS's derivative
    let mut src = String::from("import cas.cas as cas_module\nfrom cas.calculus import cas_diff\nfrom cas.store import cas_fn, cas_sym\nfrom cas.eval import cas_value_at\nx = cas_sym(\"x\")\n");
    for (i, r) in viv::derivs::rules().iter().enumerate() {
        src.push_str(&format!("def head_{i}(t) {{\n    return {}(t)\n}}\n", r.head));
        for a in &r.at {
            src.push_str(&format!("print(\"{}\", {a:?}, fmt(grad(head_{i}, {a:?}), \".17g\"), fmt(cas_value_at(cas_diff(cas_fn(\"{}\", x), x), x, {a:?}), \".17g\"))\n", r.head, r.head));
        }
    }
    let out = run_cas(&src);
    let mut n = 0;
    for l in out.lines().filter(|l| !l.starts_with("__")) {
        let w: Vec<&str> = l.split_whitespace().collect();
        let (g, c): (f64, f64) = (w[2].parse().unwrap(), w[3].parse().unwrap());
        assert!((g - c).abs() <= 1e-14 * (1.0 + c.abs()), "{l}");
        n += 1;
    }
    assert_eq!(n, viv::derivs::rules().iter().map(|r| r.at.len()).sum::<usize>());
}

#[test]
fn cas_tex_is_tex_rs_on_cas_text() {
    let corpus = [
        "x^2 - 2*x + 1",
        "7/3",
        "-1/2",
        "2*x/y",
        "-(x/y)",
        "sqrt(x + 1)",
        "(-x)^3",
        "1/(x + 1)^2",
        "exp(-x^2)",
        "3*sin(x)",
        "exp(-2*x)/2",
        "sqrt(pi)/2*erf(1)",
        "log(1 + x^2)",
        "atan(x)*x - x^3/3 + cos(2*x)",
        "x*exp(x) - exp(x)",
        "(sqrt(1929)/18 + 5/2)^(1/3)",
        "a_b + 2*y_1",
        "asin(x/2)",
        "abs(x - 1)",
    ];
    let mut src = String::from("import cas.cas as cas_module\nfrom cas.parse import cas\nfrom cas.show import cas_tex, cas_text\n");
    for c in corpus {
        src.push_str(&format!("e = cas(\"{c}\")\nprint(cas_text(e))\nprint(cas_tex(e))\n"));
    }
    let out = run_cas(&src.replace("e = cas", "print(\"--\")\ne = cas"));
    let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with("__")).collect();
    let mut syms = viv::tex::Symbols::new();
    syms.insert("pi".into(), "\\pi".into());
    let mut n = 0;
    for chunk in lines.split(|l| *l == "--").filter(|c| !c.is_empty()) {
        let (text, tex) = (chunk[0], chunk[1]);
        let e = viv::derivs::parse_expr(text).unwrap();
        assert_eq!(viv::tex::expr(&e, &syms), tex, "the CAS printed `{text}`");
        n += 1;
    }
    assert_eq!(n, corpus.len());
}

#[test]
fn the_check_refuses_what_it_cannot_hand_to_the_cas() {
    let src = "\
def boys0(T) {
    return integrate exp(-T * t * t) from 0 to 1
}
s = 1.5
s = 2.5
def g(z) {
    return z * z
}
a = integrate s * x from 0 to 1
b = integrate g(x) from 0 to 1
c = take the transpose of [[1.0, 2.0], [3.0, 4.0]]
";
    let (checks, recs) = check(src);
    let why: Vec<String> = checks.iter().map(|c| c.refused.clone().unwrap_or_default()).collect();
    assert!(why[0].contains("`T`, a variable of `boys0`"), "{why:?}");
    assert!(why[1].contains("assigns more than once"), "{why:?}");
    assert!(why[2].contains("calls `g`"), "{why:?}");
    assert!(why[3].contains("no closed form"), "{why:?}");
    assert!(recs.iter().all(|r| r.get("outcome") == "refused"));
}

#[test]
fn a_cert_entry_goes_stale_when_its_phrase_changes() {
    let (a, ra) = check("p = integrate x^2 from 1 to 2\n");
    let cert = viv::cas_check::cert_text("t.viv", &a, &ra);
    assert!(cert.contains("exact     7 / 3"), "{cert}");
    assert!(viv::cas_check::stale(&cert, &a).is_empty());
    let (b, _) = check("p = integrate x^2 from 1 to 3\n");
    assert_eq!(viv::cas_check::stale(&cert, &b).len(), 1);
}

#[test]
fn the_symbolic_derivative_needs_lib_cas() {
    let dir = std::env::temp_dir().join(format!("viv_cas_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.viv"), "print(differentiate atan(x) at 1 using the symbolic derivative)\n").unwrap();
    let o = Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("unknown function `cas_deriv_at`") && err.contains("add `import cas.cas`"), "{err}");
    // with it, from a directory outside the repository: the compiler carries lib/cas
    std::fs::write(dir.join("p.viv"), "import cas.cas\nprint(differentiate atan(x) at 1 using the symbolic derivative)\n").unwrap();
    let o = Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&o.stdout).lines().find(|l| !l.starts_with("__")), Some("0.5"), "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn a_module_map_a_def_reads_with_has_takes_string_keys() {
    // `has(m, k)` in a def taught a local copy of the module's map, which
    // stayed a map of numbers (lower.rs, the backward pass of a call)
    let dir = std::env::temp_dir().join(format!("viv_cas_maps_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = "M = map()\ndef look(k) {\n    key = \"a\" + fmt(k, \"d\")\n    if has(M, key) {\n        return M[key]\n    }\n    M[key] = k\n    return k\n}\nprint(look(3), look(3), len(M))\n";
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let o = Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&o.stdout).lines().next(), Some("3 3 1"), "{}", String::from_utf8_lossy(&o.stderr));
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
}
