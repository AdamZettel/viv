// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Lenses, milestone L1: `step <id> "<text>" { ... }`.
//!
//! What is checked here:
//! - `step` is a keyword only as `step <id> "<text>" {` or `step <id> {`; a
//!   variable or a def named `step` means what it always meant;
//! - a step is a block, not a scope;
//! - the interpreter and the compiled C record the same step events, on every
//!   way out of a step (falling off the end, `return`, `break`, `continue`),
//!   and a release build records none and prints exactly the program's output;
//! - the doc tools (lsfunc, doc, whatis) list each def's steps, and a file
//!   without steps documents exactly as before;
//! - a def's body hash ignores the def's name and the steps' text.
//!
//! The runtime checks skip if there is no C compiler.

use std::process::Command;

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

/// The C a program compiles to.
fn c_of(src: &str) -> String {
    let program = viv::build_ir(src, SourceId(0)).expect("should lower");
    CBackend.emit(&program, &MemoryIo::new()).expect("emit").artifact.as_c().expect("C").to_string()
}

/// Compile and run; stdout. `release` builds with -DVIV_RELEASE. None if no compiler.
fn native(src: &str, tag: &str, release: bool) -> Option<String> {
    let c = c_of(src);
    let dir = std::env::temp_dir().join(format!("viv_lens_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let mut cmd = Command::new(&cc);
    cmd.args(["-std=c99", "-O2", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-Wno-unused-parameter", "-Wno-unused-function"]);
    if release {
        cmd.arg("-DVIV_RELEASE");
    }
    let out = cmd.arg(dir.join("p.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}\n--- C ---\n{c}", String::from_utf8_lossy(&out.stderr));
    let warnings = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(!warnings.contains("warning"), "the C for steps should compile cleanly:\n{warnings}");
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&run.stdout).to_string())
}

/// The interpreter's output without the call-graph lines, which only it prints.
fn interp(src: &str, release: bool) -> String {
    let out = if release { viv::interp::run_release(src) } else { viv::interp::run(src) }.expect("interpreter run");
    out.lines()
        .filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__"))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// The events of the `__steps__` line, or None when there is none.
fn steps_line(out: &str) -> Option<String> {
    out.lines().find(|l| l.starts_with("__steps__ ")).map(|l| l["__steps__ ".len()..].to_string())
}

/// Run in both engines, debug and release, and require: identical debug
/// output (step events included), identical release output, and a release
/// output that is the debug output minus the `__steps__` line. Returns the
/// debug output.
fn both(src: &str, tag: &str) -> String {
    let i = interp(src, false);
    let ir = interp(src, true);
    let without_steps: String = i.lines().filter(|l| !l.starts_with("__steps__")).map(|l| format!("{l}\n")).collect();
    assert_eq!(ir, without_steps, "a release run prints the program's output and nothing else");
    if let (Some(n), Some(nr)) = (native(src, tag, false), native(src, &format!("{tag}_rel"), true)) {
        assert_eq!(n, i, "the interpreter and the compiled program disagree");
        assert_eq!(nr, ir, "the release builds disagree");
    }
    i
}

const NESTED: &str = r#"
def f(n) {
    step init "start from zero" {
        acc = 0.0
    }
    for i to n {
        step add "add one term" {
            step sq {
                t = i * i
            }
            acc = acc + t
            if i == 1 {
                continue
            }
            if i == 2 {
                break
            }
        }
    }
    step finish "hand the sum back" {
        if acc > 100 {
            return acc
        }
        return acc + 0.5
    }
}
step top "the program" {
    show f(5)
}
"#;

#[test]
fn nested_steps_agree_in_both_engines_on_every_way_out() {
    let out = both(NESTED, "nested");
    assert!(out.starts_with("5.5\n"), "{out}");
    // entry; the loop's first iteration falls off the end of `add`, the second
    // leaves it by `continue`, the third by `break`; `finish` is left by `return`
    let want = concat!(
        r#"{"total":18,"events":["+<main>/top@1","+f/init@1","-f/init@1","#,
        r#""+f/add@1","+f/sq@2","-f/sq@2","-f/add@1","#,
        r#""+f/add@1","+f/sq@2","-f/sq@2","-f/add@1","#,
        r#""+f/add@1","+f/sq@2","-f/sq@2","-f/add@1","#,
        r#""+f/finish@1","-f/finish@1","-<main>/top@1"]}"#
    );
    assert_eq!(steps_line(&out).expect("a __steps__ line"), want);
}

#[test]
fn a_step_is_a_block_not_a_scope() {
    let src = r#"
def g(x) {
    step double "double it" {
        y = 2 * x
    }
    step add {
        z = y + 1
    }
    return z
}
step here {
    w = g(3)
}
show w
"#;
    let out = both(src, "scope");
    assert!(out.starts_with("w: 7\n"), "{out}");
}

#[test]
fn step_as_a_name_still_parses() {
    // the shapes lib/radial.viv and pde/laplace_2d.viv use
    let src = r#"
step = 0.25
h = step * 2
x = 1.0
x = x + step
show h
show x
"#;
    let out = both(src, "name");
    assert_eq!(out, "h: 0.5\nx: 1.25\n");
    // no step anywhere: the C has no step runtime at all
    assert!(!c_of(src).contains("viv_step"));
    // a def called step, and a parameter called step
    let src = "def step(x) {\n    return x + 1\n}\ndef adv(x, step) {\n    return x + step\n}\nshow step(3)\nshow adv(1, 2)\n";
    assert_eq!(both(src, "defname"), "4\n3\n");
}

#[test]
fn a_step_in_a_def_never_called_prints_no_events() {
    let src = "def f() {\n    step a {\n        x = 1\n    }\n    return x\n}\nshow 2\n";
    let out = both(src, "uncalled");
    assert_eq!(out, "2\n");
}

#[test]
fn steps_inside_a_while_loop_and_a_test_block() {
    let src = r#"
def count_down(n) {
    k = n
    while k > 0 {
        step dec "take one away" {
            k = k - 1
            if k == 1 {
                break
            }
        }
    }
    return k
}
test "steps inside a test" {
    step check "call it" {
        assert_near(count_down(3), 1, 0)
    }
}
"#;
    let out = both(src, "while");
    assert!(out.starts_with("PASS steps inside a test\n"), "{out}");
    assert_eq!(
        steps_line(&out).unwrap(),
        r#"{"total":6,"events":["+<main>/check@1","+count_down/dec@1","-count_down/dec@1","+count_down/dec@1","-count_down/dec@1","-<main>/check@1"]}"#
    );
}

#[test]
fn events_past_the_cap_are_counted_but_not_listed() {
    let src = "def f(n) {\n    s = 0\n    for i to n {\n        step add {\n            s = s + i\n        }\n    }\n    return s\n}\nshow f(700)\n";
    let out = both(src, "cap");
    let line = steps_line(&out).unwrap();
    assert!(line.starts_with(r#"{"total":1400,"events":["#), "{}", &line[..60]);
    assert_eq!(line.matches("f/add@1").count(), 1000, "the first 1000 events are kept");
}

#[test]
fn a_specialized_def_records_its_source_name() {
    // `apply` takes a function, so the compiled program calls a copy named
    // after both; its steps are still `apply`'s, as the interpreter says
    let src = r#"
def sq(x) {
    return x * x
}
def apply(f, x) {
    step call "call the function" {
        y = f(x)
    }
    return y
}
show apply(sq, 3)
"#;
    assert!(c_of(src).contains("\"apply\", \"call\""), "{}", c_of(src));
    let out = both(src, "defun");
    assert_eq!(steps_line(&out).unwrap(), r#"{"total":2,"events":["+apply/call@1","-apply/call@1"]}"#);
}

#[test]
fn step_errors() {
    let err = |src: &str| match viv::build_ir(src, SourceId(0)) {
        Ok(_) => String::new(),
        Err(ds) => ds.into_iter().map(|d| d.message).collect::<Vec<_>>().join("\n"),
    };
    let e = err("def f(n) {\n    parallel for i to n {\n        step a {\n            x = i\n        }\n    }\n    return 0\n}\n");
    assert!(e.contains("cannot sit inside a `parallel for`"), "{e}");
    let e = err("def f() {\n    step a {\n        x = 1\n    }\n    step a {\n        x = 2\n    }\n    return x\n}\n");
    assert!(e.contains("`f` already has a step named `a`"), "{e}");
    let e = err("step a \"text\"\nx = 1\n");
    assert!(e.contains("expected `{`"), "{e}");
    // the same id in two defs is two anchors, f/a and g/a
    assert_eq!(err("def f() {\n    step a {\n        x = 1\n    }\n    return x\n}\ndef g() {\n    step a {\n        x = 1\n    }\n    return x\n}\nshow f() + g()\n"), "");
    // and the interpreter refuses what the compiler refuses
    assert!(viv::interp::run("step a {\nparallel for i to 2 {\nstep b {\nx = 1\n}\n}\n}\n").is_err());
}

#[test]
fn the_same_step_in_two_defs_keeps_each_defs_name() {
    // identical step bodies are one IR node (content addressing); the events
    // must still name the def they ran in
    let src = r#"
def f() {
    step a {
        x = 1
    }
    return x
}
def g() {
    step a {
        x = 1
    }
    return x
}
show f() + g()
"#;
    let out = both(src, "shared");
    assert_eq!(steps_line(&out).unwrap(), r#"{"total":4,"events":["+f/a@1","-f/a@1","+g/a@1","-g/a@1"]}"#);
}

#[test]
fn doc_tools_list_steps() {
    let ls = viv::lsfunc(NESTED);
    assert!(ls.contains("\n       3  step init  start from zero\n"), "{ls}");
    assert!(ls.contains("\n       8    step sq\n"), "{ls}");
    assert!(ls.contains("\n      20  step finish  hand the sum back\n"), "{ls}");
    let dj = viv::doc_json(NESTED);
    assert!(dj.contains(r#""steps":[{"id":"init","text":"start from zero","depth":1,"line":3,"rawLine":3},"#), "{dj}");
    assert!(dj.contains(r#"{"id":"sq","text":"","depth":2,"line":8,"rawLine":8}"#), "{dj}");
    // the top-level step is listed at file level, under <main>
    assert!(dj.ends_with(r#","steps":[{"id":"top","text":"the program","depth":1,"line":27,"rawLine":27}]}"#), "{dj}");
    let w = viv::whatis(NESTED, "f/sq");
    assert_eq!(w, "f/sq — step in f, line 8\n");
    let w = viv::whatis(NESTED, "f");
    assert!(w.contains("    step add  add one term  (line 7)\n      step sq  (line 8)\n"), "{w}");
    let st = viv::steps(NESTED);
    assert_eq!(st.iter().map(|s| (s.def.as_str(), s.id.as_str(), s.depth)).collect::<Vec<_>>(),
        [("f", "init", 1), ("f", "add", 1), ("f", "sq", 2), ("f", "finish", 1), ("<main>", "top", 1)]);
    // src shows the steps as they are written
    assert!(viv::fn_src(NESTED, "f").contains("step add \"add one term\" {"));
}

#[test]
fn doc_tools_unchanged_without_steps() {
    // `step` used as a name is not a step
    let src = "#| advance by one step\ndef adv(x, step) {\n    return x + step\n}\nstep = 0.5\nshow adv(1, step)\n";
    assert!(viv::steps(src).is_empty());
    assert_eq!(viv::lsfunc(src), "   2-4  adv(x, step)\n          advance by one step\n");
    assert!(!viv::doc_json(src).contains("\"steps\""));
}

#[test]
fn body_hash_ignores_the_def_name_and_the_step_text() {
    let hash = |src: &str, name: &str| {
        viv::def_hashes(src).unwrap().into_iter().find(|d| d.name == name).map(|d| d.body_short).unwrap()
    };
    let a = "def f(x) {\n    step double \"double it\" {\n        y = 2 * x\n    }\n    return y\n}\nshow f(1)\n";
    let renamed = a.replace("def f(", "def g(").replace("show f(", "show g(");
    let retexted = a.replace("double it", "twice x");
    let edited = a.replace("2 * x", "3 * x");
    let reid = a.replace("step double", "step triple");
    assert_eq!(hash(a, "f"), hash(&renamed, "g"), "a rename keeps the body hash, so `viv diff` still sees a rename");
    assert_eq!(hash(a, "f"), hash(&retexted, "f"), "the text is intent, not code");
    assert_ne!(hash(a, "f"), hash(&edited, "f"));
    assert_ne!(hash(a, "f"), hash(&reid, "f"), "the id is what anchors point at, so it is part of the code");
    assert!(viv::diff_report(a, &renamed).unwrap().contains("renamed"), "{}", viv::diff_report(a, &renamed).unwrap());
}

const SQ: &str = "def sq(i) {\n    step s \"square\" {\n        v = i * i\n    }\n    return v\n}\n";

#[test]
fn a_parallel_loop_cannot_reach_a_step_through_a_call() {
    // The compiled loop runs on several threads, so a step it reaches would
    // record its events in no fixed order. Refused however the step is reached:
    // a direct call, a chain of calls, a def passed in as a function value, and
    // a lambda that calls a stepping def.
    let refused = [
        ("direct", format!("{SQ}x = zeros(8)\nparallel for i to 8 {{\n    x[i] = sq(i)\n}}\nshow sum(x)\n"), "`sq`"),
        ("chain", format!("{SQ}def outer(i) {{\n    return sq(i) + 1\n}}\nx = zeros(8)\nparallel for i to 8 {{\n    x[i] = outer(i)\n}}\nshow sum(x)\n"), "`outer`"),
        ("forwarded", format!("{SQ}def run(f, n) {{\n    x = zeros(n)\n    parallel for i to n {{\n        x[i] = f(i)\n    }}\n    return sum(x)\n}}\nshow run(sq, 8)\n"), "`sq`"),
        ("lambda", format!("{SQ}def run(f, n) {{\n    x = zeros(n)\n    parallel for i to n {{\n        x[i] = f(i)\n    }}\n    return sum(x)\n}}\nshow run(fn(t) -> sq(t) + 1, 8)\n"), "`__lam0`"),
    ];
    for (tag, src, through) in refused {
        let e = match viv::build_ir(&src, SourceId(0)) {
            Ok(_) => panic!("{tag}: the compiler should refuse\n{src}"),
            Err(ds) => ds.into_iter().map(|d| d.message).collect::<Vec<_>>().join("\n"),
        };
        assert!(e.contains(&format!("a `parallel for` reaches a `step` through {through}")), "{tag}: {e}");
        // the interpreter refuses the same program, so neither engine runs it
        assert!(viv::interp::run(&src).is_err(), "{tag}: the interpreter should refuse");
    }
}

#[test]
fn a_parallel_loop_beside_steps_still_runs() {
    // Not refused: a parallel loop that calls only step-free defs, a value a
    // stepping def returned (a number, not a function), a stepping def called
    // from a plain `for`, and a step around the parallel loop.
    let src = format!(
        "{SQ}def cube(i) {{\n    return i * i * i\n}}\ne = sq(3)\nx = zeros(8)\nparallel for i to 8 {{\n    x[i] = cube(i) + e\n}}\nfor i to 2 {{\n    x[i] = sq(i)\n}}\nstep total {{\n    parallel for i to 8 {{\n        x[i] = x[i] + 1\n    }}\n    show sum(x)\n}}\n"
    );
    let out = both(&src, "par_ok");
    assert_eq!(
        steps_line(&out).unwrap(),
        r#"{"total":8,"events":["+sq/s@1","-sq/s@1","+sq/s@1","-sq/s@1","+sq/s@1","-sq/s@1","+<main>/total@1","-<main>/total@1"]}"#
    );
}

#[test]
fn the_c_step_recorder_stops_on_a_worker_thread() {
    // A second line of defense behind the static check: built with OpenMP, a
    // step that runs inside a parallel region exits with an error instead of
    // racing on the event array. The C is edited by hand to force the call,
    // since the compiler refuses to write it. Skips if no OpenMP is available.
    let src = format!("{SQ}def cube(i) {{\n    return i * i * i\n}}\nshow sq(2)\nx = zeros(64)\nparallel for i to 64 {{\n    x[i] = cube(i)\n}}\nshow sum(x)\n");
    let c = c_of(&src);
    assert!(c.contains("omp_in_parallel()"), "the recorder should check for a worker thread");
    // (the index is an integer)
    let forced = c.replacen("viv_lset1(x, (long long)(i), cube(i)", "viv_lset1(x, (long long)(i), sq(i)", 1);
    assert_ne!(forced, c, "the parallel loop body should call cube:\n{c}");
    let dir = std::env::temp_dir().join(format!("viv_lens_{}_omp", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &forced).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let brew = "/opt/homebrew/opt/libomp";
    let flag_sets: [Vec<String>; 2] = [
        vec!["-fopenmp".into()],
        vec!["-Xclang".into(), "-fopenmp".into(), format!("-I{brew}/include"), format!("-L{brew}/lib"), "-lomp".into()],
    ];
    let built = flag_sets.iter().any(|flags| {
        Command::new(&cc)
            .args(["-std=c99", "-O2"])
            .args(flags)
            .arg(dir.join("p.c"))
            .arg("-lm")
            .arg("-o")
            .arg(&bin)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    });
    if !built {
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("skipping: no OpenMP C compiler");
        return;
    }
    let run = Command::new(&bin).env("OMP_NUM_THREADS", "4").output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!run.status.success(), "a step on a worker thread should stop the program");
    assert!(String::from_utf8_lossy(&run.stderr).contains("error: step `sq/s` ran inside a parallel for"));
}

// ---------------------------------------------------------------------------
// Milestones L2-L4: the `.intent` format, `lens init`, `view`, `lens check`,
// `lens stamp`, `lens move` and `explain` (lang/viv/src/lens.rs).
// ---------------------------------------------------------------------------

use viv::lens::{self, Input, IntentFile, ModuleText};

/// The spec's own example, verbatim.
const SPEC_INTENT: &str = "module lib/hf/rhf
  The Hartree-Fock SCF loop for closed shells, and the parts UHF and ROHF share.

rhf_scf  @3fa9c21e
  what  Solve the restricted Hartree-Fock equations by iterating to self-consistency.
  how   Roothaan-Hall in an orthogonalized basis, with DIIS on the commutator FDS - SDF.
  why   A registry of J/K builders is passed in so one loop serves in-core, packed
        and direct integrals; opts[OPT_JK] picks which.
  D     density matrix, AO basis, n x n
  F     Fock matrix, AO basis

  orth         orthogonalize the basis
  guess        start from a density
  repeat until the density and energy stop changing
    jk           build J and K from the current density
    fock         form F = H + J - K/2
    solve        solve F C = S C e for the orbitals
    density      rebuild the density from the occupied orbitals
    extrapolate  extrapolate the Fock matrix with DIIS
  return the energy, the density, and the convergence record
";

#[test]
fn the_intent_format_round_trips_byte_for_byte() {
    let f = IntentFile::parse(SPEC_INTENT).expect("the spec's example parses");
    assert_eq!(f.print(), SPEC_INTENT);
    // odd layouts survive too: no final newline, trailing spaces, several
    // blank lines, a provenance signature, an empty field, a block with no card
    for text in [
        "a  @0123abcd  by adam\n  what  x\n\n  s1  one\n    s2  two   \n\n\n\nb\n  what\n",
        "a\n\n  s1\n  if the thing holds\n    s2  inside\n  else\n    s3",
        "",
        "\n",
        "module m\n  first paragraph\n\n  second paragraph\n",
    ] {
        let f = IntentFile::parse(text).unwrap_or_else(|e| panic!("{text:?}: {e:?}"));
        assert_eq!(f.print(), text);
    }
    let (module, entries) = f.model();
    let module = module.unwrap();
    assert_eq!(module.name, "lib/hf/rhf");
    assert_eq!(module.text, "The Hartree-Fock SCF loop for closed shells, and the parts UHF and ROHF share.");
    let e = &entries[0];
    assert_eq!((e.name.as_str(), e.stamp.as_deref(), e.line), ("rhf_scf", Some("3fa9c21e"), 4));
    assert_eq!(e.field("why").unwrap().text, "A registry of J/K builders is passed in so one loop serves in-core, packed and direct integrals; opts[OPT_JK] picks which.");
    assert_eq!(e.gloss.iter().map(|g| g.key.as_str()).collect::<Vec<_>>(), ["D", "F"]);
    let shape: Vec<String> = e
        .outline
        .iter()
        .map(|n| match &n.kind {
            lens::NodeKind::Step(id) => id.clone(),
            lens::NodeKind::Control(_) => format!("{} > {}", n.text, n.children.len()),
            lens::NodeKind::Note => format!("note: {}", n.text),
        })
        .collect();
    assert_eq!(shape, ["orth", "guess", "repeat until the density and energy stop changing > 5", "note: return the energy, the density, and the convergence record"]);
    assert_eq!(e.outline[2].children[1].text, "form F = H + J - K/2");
}

#[test]
fn intent_parse_errors_name_the_line() {
    let err = |t: &str| IntentFile::parse(t).err().map(|e| (e[0].line, e[0].message.clone())).expect("should not parse");
    assert_eq!(err("  what  floating\n").0, 1);
    assert!(err("f\n\n    a  one\n  b  two\n   c  three\n").1.contains("does not line up"));
    assert!(err("f  @xyz\n").1.contains("not a body hash"));
    assert!(err("f  extra words\n").1.contains("expected `@<hash>`"));
    assert!(err("f\n\n  return x\n    a  under a note\n").1.contains("only a step or a control line"));
    assert!(err("f\n\twhat  tabs\n").1.contains("tabs"));
    assert!(err("f\n  what  x\n      continued\n  9bad  x\n").1.contains("glossary name"));
}

/// A module the fixture uses.
const UTIL: &str = "#| Half of a number.
def half(x) {
    return 0.5 * x
}

test \"half of 4 is 2\" {
    assert_near(half(4), 2, 0)
}
";

/// The fixture: steps inside a loop, a test helper, a private def.
const ITER: &str = "from fx.util import half

def iterate(x0, tol) {
    step guess {
        x = x0
        dx = 1.0
    }
    n = 0
    while dx > tol {
        step update {
            xn = half(x) + 1.0
        }
        step measure {
            dx = abs(xn - x)
            x = xn
            n = n + 1
        }
    }
    return x
}

def _scratch(x) {
    return x
}

#~ test
def check_close(a, b) {
    return abs(a - b)
}

test \"the fixed point of x/2 + 1 is 2\" {
    assert_near(iterate(0.0, 1e-12), 2, 1e-9)
    assert_near(check_close(1, 1), 0, 0)
}
";

/// The fixture's intent, with `@H` standing for iterate's body hash.
const ITER_INTENT: &str = "module fx/iter
  A fixed-point iteration.

iterate  @H
  what  Find the fixed point of x/2 + 1 [test \"the fixed point of x/2 + 1 is 2\"].
  how   Each step halves the error, a factor 0.5 [test \"half of 4 is 2\"].
  x     the current iterate

  guess    start from x0
  repeat until the step is shorter than tol
    update   apply the map
    measure  measure the step and move
  return the iterate
";

/// The lens tools' input for a source, an intent file, and the modules it uses.
fn input(src: &str, intent: Option<&str>, used: &[(&str, &str)]) -> Input {
    let expanded = viv::expand_uses(src, &mut |name: &str| {
        used.iter().find(|(n, _)| *n == name).map(|(_, t)| t.to_string()).ok_or_else(|| format!("no module {name}"))
    })
    .unwrap();
    // `@H` is iterate's hash in the fixture as written, so an edited copy is stale
    let intent = intent.map(|t| t.replace("@H", &format!("@{}", hash_of(ITER, "iterate"))));
    Input {
        root: ModuleText { name: "fx/iter".into(), path: "fx/iter.viv".into(), src: src.to_string(), intent_path: "fx/iter.intent".into(), intent },
        used: used
            .iter()
            .map(|(n, t)| ModuleText { name: n.to_string(), path: format!("{n}.viv"), src: t.to_string(), intent_path: format!("{n}.intent"), intent: None })
            .collect(),
        expanded,
        companion_tests: vec![],
    }
}

fn hash_of(src: &str, name: &str) -> String {
    let expanded = viv::expand_uses(src, &mut |_| Ok(UTIL.to_string())).unwrap();
    viv::def_hashes(&expanded).unwrap().into_iter().find(|h| h.name == name).unwrap().body_short
}

fn check(src: &str, intent: Option<&str>) -> Vec<lens::Finding> {
    let l = lens::load(&input(src, intent, &[("fx/util", UTIL)])).unwrap();
    lens::check(&l, &mut |_, _| None)
}

/// The findings as `kind anchor` lines, for comparing.
fn kinds(fs: &[lens::Finding]) -> Vec<String> {
    fs.iter().map(|f| format!("{} {}", f.kind, f.anchor)).collect()
}

#[test]
fn init_writes_every_def_step_and_hash_and_no_text() {
    let l = lens::load(&input(ITER, None, &[("fx/util", UTIL)])).unwrap();
    let skel = lens::init(&l);
    let want = format!(
        "module fx/iter\n\niterate  @{}\n  what\n\n  guess\n  repeat\n    update\n    measure\n\n_scratch  @{}\n  what\n\ncheck_close  @{}\n  what\n",
        hash_of(ITER, "iterate"),
        hash_of(ITER, "_scratch"),
        hash_of(ITER, "check_close")
    );
    assert_eq!(skel, want);
    assert_eq!(IntentFile::parse(&skel).unwrap().print(), skel, "the skeleton round-trips");
    // the hashes are the ones `viv hash` prints
    assert!(viv::hash_report(&l_expanded(ITER)).unwrap().contains(&format!("body {}  def iterate(2 params)", hash_of(ITER, "iterate"))));
    // an anchor written inline is left out, so it never gets a second home;
    // a step with inline text is still listed, since the outline is structure
    let inline = ITER.replace("def iterate", "#| Iterate.\ndef iterate").replace("step update {", "step update \"apply the map\" {");
    let l = lens::load(&input(&inline, None, &[("fx/util", UTIL)])).unwrap();
    let skel = lens::init(&l);
    assert!(skel.contains(&format!("iterate  @{}\n\n  guess\n  repeat\n    update\n", hash_of(&inline, "iterate"))), "{skel}");
    // and the skeleton checks clean except for the text it asks for
    let l = lens::load(&input(ITER, Some(&lens::init(&lens::load(&input(ITER, None, &[("fx/util", UTIL)])).unwrap())), &[("fx/util", UTIL)])).unwrap();
    let fs = lens::check(&l, &mut |_, _| None);
    assert_eq!(kinds(&fs), ["step-text iterate/guess", "step-text iterate/update", "step-text iterate/measure", "what iterate"]);
}

fn l_expanded(src: &str) -> String {
    viv::expand_uses(src, &mut |_| Ok(UTIL.to_string())).unwrap()
}

#[test]
fn a_covered_module_checks_clean() {
    let fs = check(ITER, Some(ITER_INTENT));
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn each_seeded_failure_reports_exactly_one_error() {
    let cases: Vec<(&str, String, String, &str)> = vec![
        // edit a body: the entry is stale
        ("edit a body", ITER.replace("n = n + 1", "n = n + 2"), ITER_INTENT.to_string(), "stale iterate"),
        // delete a step from the intent: the code's step has no text
        ("delete a step from the intent", ITER.to_string(), ITER_INTENT.replace("    update   apply the map\n", ""), "step-text iterate/update"),
        // delete a step from the code: the body changed, so the entry is stale
        ("delete a step from the code", ITER.replace("        step update {\n            xn = half(x) + 1.0\n        }\n", "        xn = half(x) + 1.0\n"), ITER_INTENT.to_string(), "stale iterate"),
        // add an uncited number
        ("add an uncited number", ITER.to_string(), ITER_INTENT.replace("  x     the", "  why   It converges in about 40 steps.\n  x     the"), "number iterate"),
        // duplicate an anchor: `what` inline as well as in the file
        ("duplicate an anchor", ITER.replace("def iterate", "#| Iterate.\ndef iterate"), ITER_INTENT.to_string(), "duplicate iterate"),
        ("duplicate a field in the file", ITER.to_string(), ITER_INTENT.replace("  x     the", "  what  Again.\n  x     the"), "duplicate iterate"),
        ("duplicate a step's text", ITER.replace("step update {", "step update \"apply it\" {"), ITER_INTENT.to_string(), "duplicate iterate/update"),
        ("duplicate a glossary name", ITER.replace("def iterate", "#: x  the iterate\ndef iterate"), ITER_INTENT.to_string(), "duplicate iterate:x"),
        ("cite a test that does not exist", ITER.to_string(), ITER_INTENT.replace("half of 4 is 2", "half of 4 is two"), "citation iterate"),
        ("nest a step differently", ITER.to_string(), ITER_INTENT.replace("    measure  measure", "  measure  measure"), "nesting iterate"),
        ("name a step the code lacks", ITER.to_string(), ITER_INTENT.replace("  return the iterate", "  finish  hand it back\n  return the iterate"), "step-missing iterate/finish"),
        ("an entry for a def that is gone", ITER.to_string(), format!("{ITER_INTENT}\ngone  @0123abcd\n  what  Nothing.\n"), "orphan gone"),
        ("an unstamped entry", ITER.to_string(), ITER_INTENT.replace("iterate  @H", "iterate"), "unstamped iterate"),
        ("drop the what", ITER.to_string(), ITER_INTENT.replace("  what  Find the fixed point of x/2 + 1 [test \"the fixed point of x/2 + 1 is 2\"].\n", ""), "what iterate"),
        ("a public def without a what", format!("{ITER}\ndef extra(x) {{\n    return x\n}}\n"), ITER_INTENT.to_string(), "what extra"),
    ];
    for (what, src, intent, want) in cases {
        let fs = check(&src, Some(&intent));
        assert_eq!(kinds(&fs), [want], "{what}:\n{}", fs.iter().map(|f| f.render()).collect::<String>());
    }
}

#[test]
fn a_citation_may_name_a_test_of_the_companion_test_file() {
    // a test that moved to fx/tests/iter.viv, beside fx/iter.viv, so that
    // programs using the module do not load what it needs
    let intent = ITER_INTENT.replace("half of 4 is 2", "half of 8 is 4");
    let mut i = input(ITER, Some(&intent), &[("fx/util", UTIL)]);
    let fs = lens::check(&lens::load(&i).unwrap(), &mut |_, _| None);
    assert_eq!(kinds(&fs), ["citation iterate"]);
    i.companion_tests = vec!["half of 8 is 4".to_string()];
    let fs = lens::check(&lens::load(&i).unwrap(), &mut |_, _| None);
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn a_stale_entry_shows_what_changed() {
    let edited = ITER.replace("n = n + 1", "n = n + 2").replace("        step update {\n            xn = half(x) + 1.0\n        }\n", "        xn = half(x) + 1.0\n");
    let l = lens::load(&input(&edited, Some(ITER_INTENT), &[("fx/util", UTIL)])).unwrap();
    let old = hash_of(ITER, "iterate");
    let mut asked = Vec::new();
    let fs = lens::check(&l, &mut |def, h| {
        asked.push((def.to_string(), h.to_string()));
        lens::def_text(ITER, def)
    });
    assert_eq!(asked, [("iterate".to_string(), old.clone())], "the host is asked for the stamped version");
    assert_eq!(fs.len(), 1);
    let f = &fs[0];
    assert_eq!((f.path.as_str(), f.line, f.kind), ("fx/iter.intent", 4, "stale"));
    let new = hash_of(&edited, "iterate");
    assert_eq!(
        f.detail,
        [
            format!("edited     def iterate(x0, tol)  (body {old} -> {new})"),
            "-         step update {".to_string(),
            "-             xn = half(x) + 1.0".to_string(),
            "-         }".to_string(),
            "+         xn = half(x) + 1.0".to_string(),
            "-             n = n + 1".to_string(),
            "+             n = n + 2".to_string(),
            "the intent names step `update`, which the code no longer has".to_string(),
        ]
    );
}

#[test]
fn inline_intent_is_stamped_with_a_hash_line() {
    let src = "#| Double it.\ndef dbl(x) {\n    step twice \"multiply by two\" {\n        y = 2 * x\n    }\n    return y\n}\n";
    let load = |s: &str| lens::load(&input(s, None, &[])).unwrap();
    assert_eq!(kinds(&lens::check(&load(src), &mut |_, _| None)), ["unstamped dbl"]);
    let (stamped, file, msg) = lens::stamp(&load(src), "dbl").unwrap();
    let h = viv::def_hashes(src).unwrap().into_iter().find(|d| d.name == "dbl").unwrap().body_short;
    assert_eq!(stamped, src.replace("def dbl", &format!("#@ {h}\ndef dbl")));
    assert_eq!(file, None);
    assert_eq!(msg, format!("stamped dbl @{h} (inline)"));
    assert!(lens::check(&load(&stamped), &mut |_, _| None).is_empty());
    // an edit makes it stale; stamping again replaces the line
    let edited = stamped.replace("2 * x", "x + x");
    assert_eq!(kinds(&lens::check(&load(&edited), &mut |_, _| None)), ["stale dbl"]);
    let (restamped, _, _) = lens::stamp(&load(&edited), "dbl").unwrap();
    assert_eq!(restamped.matches("#@ ").count(), 1);
    assert!(lens::check(&load(&restamped), &mut |_, _| None).is_empty());
    // the stamp line is invisible to the doc tools and to both engines
    assert!(viv::doc_json(&stamped).contains("\"what\":\"Double it.\""));
    assert_eq!(viv::lsfunc(&stamped), viv::lsfunc(src).replace("   2-7  dbl(x)", "   3-8  dbl(x)").replace("       3  step", "       4  step"));
    let prog = format!("{stamped}show dbl(21)\n");
    assert_eq!(both(&prog, "stamp"), both(&format!("{src}show dbl(21)\n"), "nostamp"));
    // a def with no intent anywhere has nothing to stamp
    assert!(lens::stamp(&load("def f() {\n    return 1\n}\n"), "f").unwrap_err().contains("no intent to stamp"));
}

#[test]
fn stamping_a_file_entry_rewrites_its_header() {
    let intent = ITER_INTENT.replace("iterate  @H", "iterate  @00000000  by adam");
    let l = lens::load(&input(ITER, Some(&intent), &[("fx/util", UTIL)])).unwrap();
    let (src, file, msg) = lens::stamp(&l, "iterate").unwrap();
    assert_eq!(src, ITER, "the source is untouched when the def's intent is in the file");
    let h = hash_of(ITER, "iterate");
    assert_eq!(file.unwrap(), intent.replace("iterate  @00000000  by adam", &format!("iterate  @{h}  by adam")));
    assert_eq!(msg, format!("stamped iterate @{h} (file)"));
}

#[test]
fn the_code_lens_hides_intent_lines() {
    let src = "###\npost: a title\n#| this is prose in the header\n###\n\n#| What f does.\n#|| How.\n#: x  the input\n#@ 0123abcd\n#~ tag\ndef f(x) {\n    #: y  twice x\n    y = 2 * x   # a plain comment stays\n    return y\n}\n";
    assert_eq!(
        lens::view_code(src),
        "###\npost: a title\n#| this is prose in the header\n###\n\n#~ tag\ndef f(x) {\n    y = 2 * x   # a plain comment stays\n    return y\n}\n"
    );
}

#[test]
fn the_intent_and_all_lenses_read_either_home() {
    let l = lens::load(&input(ITER, Some(ITER_INTENT), &[("fx/util", UTIL)])).unwrap();
    let intent = lens::view_intent(&l);
    assert_eq!(
        intent,
        "module fx/iter\n  A fixed-point iteration.\n\niterate\n  what  Find the fixed point of x/2 + 1 [test \"the fixed point of x/2 + 1 is 2\"].\n  how   Each step halves the error, a factor 0.5 [test \"half of 4 is 2\"].\n  x     the current iterate\n\n  guess    start from x0\n  repeat until the step is shorter than tol\n    update   apply the map\n    measure  measure the step and move\n  return the iterate\n\n_scratch\n\ncheck_close\n"
    );
    // the same intent written inline reads the same, apart from the note, which has no inline home
    let (inline_src, rest, kept) = lens::to_inline(&l, None, false);
    assert_eq!(kept, ["iterate: `return the iterate` has no inline home; it stays in the file"]);
    let rest = rest.unwrap();
    let l2 = lens::load(&input(&inline_src, Some(&rest), &[("fx/util", UTIL)])).unwrap();
    assert_eq!(lens::view_intent(&l2), intent);
    // all: the code with the intent beside what it explains
    let all = lens::view_all(&l);
    assert!(all.starts_with("from fx.util import half\n\n#| Find the fixed point of x/2 + 1 [test \"the fixed point of x/2 + 1 is 2\"].\n#|| Each step halves the error, a factor 0.5 [test \"half of 4 is 2\"].\n#: x  the current iterate\ndef iterate(x0, tol) {\n    step guess \"start from x0\" {\n"), "{all}");
    assert!(all.contains("    #| repeat until the step is shorter than tol\n    while dx > tol {\n        step update \"apply the map\" {\n"), "{all}");
    assert!(!all.contains("#@"), "the all lens shows no stamps");
    // the code lens of the all lens is the code
    assert_eq!(lens::view_code(&all).replace(" \"start from x0\"", "").replace(" \"apply the map\"", "").replace(" \"measure the step and move\"", ""), ITER);
}

#[test]
fn moving_entries_to_the_file_and_back_changes_nothing_else() {
    // start inline and stamped
    let l = lens::load(&input(ITER, Some(ITER_INTENT), &[("fx/util", UTIL)])).unwrap();
    let (inline_src, rest, _) = lens::to_inline(&l, None, false);
    // the note stays in the file, so the stamp stays there too; drop the note to move everything
    let no_note = ITER_INTENT.replace("  return the iterate\n", "");
    let l = lens::load(&input(ITER, Some(&no_note), &[("fx/util", UTIL)])).unwrap();
    let (all_inline, left, kept) = lens::to_inline(&l, None, false);
    assert!(kept.is_empty(), "{kept:?}");
    assert_eq!(left.as_deref(), Some("module fx/iter\n  A fixed-point iteration.\n"), "the module's own text has no inline home");
    assert!(all_inline.contains(&format!("#: x  the current iterate\n#@ {}\ndef iterate", hash_of(ITER, "iterate"))), "{all_inline}");
    assert!(lens::check(&lens::load(&input(&all_inline, left.as_deref(), &[("fx/util", UTIL)])).unwrap(), &mut |_, _| None).is_empty());
    // and back: the file is what it was, the source is the plain code again
    let l = lens::load(&input(&all_inline, left.as_deref(), &[("fx/util", UTIL)])).unwrap();
    let (src, file, kept) = lens::to_file(&l, None);
    assert!(kept.is_empty(), "{kept:?}");
    assert_eq!(src, ITER);
    assert_eq!(file.unwrap(), no_note.replace("@H", &format!("@{}", hash_of(ITER, "iterate"))));
    // a partial move: the note keeps the block, and its stamp, in the file
    let rest = rest.unwrap();
    assert_eq!(rest, format!("module fx/iter\n  A fixed-point iteration.\n\niterate  @{}\n\n  guess\n  repeat\n    update\n    measure\n  return the iterate\n", hash_of(ITER, "iterate")));
    assert!(!inline_src.contains("#@"), "while the block stays, its stamp covers the def");
    assert!(lens::check(&lens::load(&input(&inline_src, Some(&rest), &[("fx/util", UTIL)])).unwrap(), &mut |_, _| None).is_empty());
    // moving one def only touches that def
    let two = format!("{ITER}\n#| Add one.\ndef inc(x) {{\n    return x + 1\n}}\n");
    let l = lens::load(&input(&two, None, &[("fx/util", UTIL)])).unwrap();
    let (src, file, _) = lens::to_file(&l, Some(&["inc".to_string()]));
    assert_eq!(src, two.replace("#| Add one.\n", ""));
    assert_eq!(file.unwrap(), "module fx/iter\n\ninc\n  what  Add one.\n");
}

#[test]
fn test_helpers_and_private_defs_need_no_what_and_long_defs_need_steps() {
    // `check_close` is tagged as a test helper and `_scratch` is private: no findings
    assert!(check(ITER, Some(ITER_INTENT)).is_empty());
    // without the tag check_close is public; being called only from a test
    // does not exempt a def, since that is how a library's entry point looks
    let fs = check(&ITER.replace("#~ test\n", ""), Some(ITER_INTENT));
    assert_eq!(kinds(&fs), ["what check_close"]);
    // 26 statements and no step
    let body: String = (0..26).map(|k| format!("    a{k} = {k}\n")).collect();
    let long = format!("#| Long.\ndef long() {{\n{body}}}\n");
    let fs = lens::check(&lens::load(&input(&long, None, &[])).unwrap(), &mut |_, _| None);
    assert_eq!(kinds(&fs), ["unstamped long", "steps long"]);
    assert!(fs[1].message.contains("26 statements"), "{}", fs[1].message);
    let ok = format!("#| Short enough.\ndef long() {{\n{}}}\n", (0..25).map(|k| format!("    a{k} = {k}\n")).collect::<String>());
    assert_eq!(kinds(&lens::check(&lens::load(&input(&ok, None, &[])).unwrap(), &mut |_, _| None)), ["unstamped long"]);
}

#[test]
fn which_words_are_numbers() {
    let src = |what: &str| format!("#| {what}\ndef f() {{\n    return 1\n}}\n");
    let numbers = |what: &str| -> Vec<String> {
        let s = src(what);
        let h = viv::def_hashes(&s).unwrap().into_iter().find(|d| d.name == "f").unwrap().body_short;
        let s = s.replace("def f", &format!("#@ {h}\ndef f"));
        lens::check(&lens::load(&input(&s, None, &[])).unwrap(), &mut |_, _| None).into_iter().map(|f| f.message).collect()
    };
    assert!(numbers("Water in 6-31G, a 2D grid, H2O and O(N^4) work.").is_empty(), "names are not numbers");
    assert!(numbers("Returns `0` and reads `opts[9]`.").is_empty(), "code spans are not prose");
    let m = numbers("Converges in 20 steps to 1e-10, about 50%, of 1,000 (3.) and -75.98.");
    assert_eq!(m, ["`f` what states `20`, `1e-10`, `50%`, `1,000`, `3`, `-75.98` with no [test \"...\"] citation"]);
}

#[test]
fn a_top_level_step_needs_text_too() {
    let src = "step main_work {\n    x = 1\n}\nshow x\n";
    let fs = lens::check(&lens::load(&input(src, None, &[])).unwrap(), &mut |_, _| None);
    assert_eq!(kinds(&fs), ["step-text <main>/main_work"]);
    let with_text = src.replace("step main_work {", "step main_work \"set x\" {");
    assert!(lens::check(&lens::load(&input(&with_text, None, &[])).unwrap(), &mut |_, _| None).is_empty());
    // or text in a `<main>` block
    let l = lens::load(&input(src, None, &[])).unwrap();
    let skel = lens::init(&l);
    let h = viv::def_hashes(src).unwrap().into_iter().find(|d| d.name == "<main>").unwrap().body_short;
    assert_eq!(skel, format!("module fx/iter\n\n<main>  @{h}\n\n  main_work\n"));
    let filled = skel.replace("  main_work\n", "  main_work  set x\n");
    assert!(lens::check(&lens::load(&input(src, Some(&filled), &[])).unwrap(), &mut |_, _| None).is_empty());
}

const EXPLAIN_SRC: &str = "from fx.util import half

#| Count down to zero.
def down(n) {
    if n > 0 {
        return down(n - 1)
    }
    return 0
}

#| Pick a halver from a registry.
def pick(k) {
    R = [half, half]
    return R
}

#| Run everything.
def top(x) {
    step prepare \"get ready\" {
        a = half(x)
        b = down(3)
    }
    step finish \"finish up\" {
        c = half(a) + down(1)
        r = pick(0)
    }
    return c
}
";

#[test]
fn explain_walks_the_call_graph() {
    let l = lens::load(&input(EXPLAIN_SRC, None, &[("fx/util", UTIL)])).unwrap();
    let e = lens::explain(&l, Some("top"), None).unwrap();
    assert_eq!(
        lens::explain_text(&e),
        "top  Run everything.
  prepare: get ready
    -> half  Half of a number.
    -> down  Count down to zero.
      -> down  Count down to zero.  (recursive)
  finish: finish up
    -> half  Half of a number.
    -> down  Count down to zero.  (expanded above)
    -> pick  Pick a halver from a registry.
      -> half  Half of a number.  (as a value)
"
    );
    // --depth 1: the direct calls, not expanded; --depth 0: the steps alone
    let e1 = lens::explain(&l, Some("top"), Some(1)).unwrap();
    assert_eq!(lens::explain_text(&e1), "top  Run everything.\n  prepare: get ready\n    -> half  Half of a number.\n    -> down  Count down to zero.\n  finish: finish up\n    -> half  Half of a number.\n    -> down  Count down to zero.\n    -> pick  Pick a halver from a registry.\n");
    let e0 = lens::explain(&l, Some("top"), Some(0)).unwrap();
    assert_eq!(lens::explain_text(&e0), "top  Run everything.\n  prepare: get ready\n  finish: finish up\n");
    // the module a callee comes from, in the JSON
    let j = lens::explain_json(&e1);
    assert!(j.contains(r#"{"kind":"call","name":"half","text":"Half of a number.","module":"fx/util","value":false,"note":"","children":[]}"#), "{j}");
    assert!(j.contains(r#""name":"down","text":"Count down to zero.","module":"fx/iter","value":false,"note":"depth""#), "{j}");
    json_ok(&j);
    assert!(lens::explain(&l, Some("nothing"), None).unwrap_err().contains("no def named `nothing`"));
    // from the top level: the tests and what they call
    let l = lens::load(&input(ITER, Some(ITER_INTENT), &[("fx/util", UTIL)])).unwrap();
    let t = lens::explain_text(&lens::explain(&l, None, None).unwrap());
    assert!(t.starts_with("<main>\n  test \"the fixed point of x/2 + 1 is 2\"\n    -> iterate  Find the fixed point"), "{t}");
    assert!(t.contains("      guess: start from x0\n      repeat until the step is shorter than tol\n        update: apply the map\n          -> half  Half of a number.\n        measure: measure the step and move\n    -> check_close\n"), "{t}");
}

#[test]
fn an_if_and_its_else_are_one_structure() {
    // the step sits in the else branch; the `if` stays in the outline with it
    let src = "#| Relax toward zero.\ndef relax(x, tol) {\n    while abs(x) > 0 {\n        if abs(x) < tol {\n            x = 0\n        }\n        else {\n            step shrink \"halve it\" {\n                x = x * 0.5\n            }\n        }\n    }\n    return x\n}\n";
    let l = lens::load(&input(src, None, &[])).unwrap();
    let skel = lens::init(&l);
    let h = viv::def_hashes(src).unwrap().into_iter().find(|d| d.name == "relax").unwrap().body_short;
    assert_eq!(skel, format!("module fx/iter\n\nrelax  @{h}\n\n  repeat\n    if\n    else\n      shrink\n"));
    let intent = format!("relax  @{h}\n\n  repeat until x is zero\n    if x is already below tol, snap it to zero\n    else\n      shrink\n");
    assert!(lens::check(&lens::load(&input(src, Some(&intent), &[])).unwrap(), &mut |_, _| None).is_empty());
    // leaving the `if` out of the intent is a different shape
    let fs = lens::check(&lens::load(&input(src, Some(&intent.replace("    if x is already below tol, snap it to zero\n", "")), &[])).unwrap(), &mut |_, _| None);
    assert_eq!(kinds(&fs), ["nesting relax"]);
    assert_eq!(fs[0].detail, ["code:   loop[if[] else[shrink]]", "intent: loop[else[shrink]]"]);
    let e = lens::explain(&lens::load(&input(src, Some(&intent), &[])).unwrap(), Some("relax"), None).unwrap();
    assert_eq!(lens::explain_text(&e), "relax  Relax toward zero.\n  repeat until x is zero\n    if x is already below tol, snap it to zero\n    else\n      shrink: halve it\n");
}

/// Every def a def's explanation calls directly (not as a value), at depth 1.
fn direct_calls(n: &lens::ENode, out: &mut std::collections::BTreeSet<String>) {
    for c in &n.children {
        if c.kind == "call" {
            if !c.value {
                out.insert(c.name.clone());
            }
        } else {
            direct_calls(c, out);
        }
    }
}

#[test]
fn explain_walks_the_graph_viv_deps_prints() {
    // every def of lib/hf/hf.viv and the modules it uses: the calls explain
    // shows are exactly the calls `viv deps` lists
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |name: &str| std::fs::read_to_string(root.join(format!("{name}.viv"))).map_err(|e| e.to_string());
    let src = read("lib/hf/hf").unwrap();
    let expanded = viv::expand_uses(&src, &mut |n: &str| read(n)).unwrap();
    let mut used = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let uses = |unit: &str, src: &str| -> Vec<String> { viv::imports::known_refs(unit, src, &|c: &str| root.join(format!("{c}.viv")).is_file()).into_iter().map(|(m, _)| m).collect() };
    let mut stack = uses("lib/hf/hf", &src);
    stack.reverse();
    while let Some(n) = stack.pop() {
        if !seen.insert(n.clone()) {
            continue;
        }
        let t = read(&n).unwrap();
        let mut inner = uses(&n, &t);
        inner.reverse();
        stack.extend(inner);
        used.push(ModuleText { name: n.clone(), path: n.clone(), src: t, intent_path: String::new(), intent: None });
    }
    let input = Input { root: ModuleText { name: "lib/hf/hf".into(), path: "lib/hf/hf.viv".into(), src, intent_path: String::new(), intent: None }, used, expanded: expanded.clone(), companion_tests: vec![] };
    let l = lens::load(&input).unwrap();
    let deps = viv::deps_json(&expanded).unwrap();
    let mut checked = 0;
    for cap in deps.split("{\"name\":\"").skip(1) {
        let name: String = cap.chars().take_while(|c| *c != '"').collect();
        let want: std::collections::BTreeSet<String> = cap.split("\"calls\":[").nth(1).unwrap().split(']').next().unwrap().split(',').filter(|x| !x.is_empty()).map(|x| x.trim_matches('"').to_string()).collect();
        let e = lens::explain(&l, Some(&name), Some(1)).unwrap();
        let mut got = std::collections::BTreeSet::new();
        direct_calls(&e, &mut got);
        assert_eq!(got, want, "{name}");
        checked += 1;
    }
    assert!(checked > 300, "{checked} defs");
}

/// A small JSON checker: the text is one JSON value and nothing else.
fn json_ok(s: &str) {
    fn ws(b: &[u8], mut i: usize) -> usize {
        while i < b.len() && (b[i] as char).is_ascii_whitespace() {
            i += 1;
        }
        i
    }
    fn value(b: &[u8], i: usize) -> Result<usize, usize> {
        let i = ws(b, i);
        match b.get(i) {
            Some(b'{') => {
                let mut i = ws(b, i + 1);
                if b.get(i) == Some(&b'}') {
                    return Ok(i + 1);
                }
                loop {
                    i = string(b, ws(b, i))?;
                    i = ws(b, i);
                    if b.get(i) != Some(&b':') {
                        return Err(i);
                    }
                    i = ws(b, value(b, i + 1)?);
                    match b.get(i) {
                        Some(b',') => i += 1,
                        Some(b'}') => return Ok(i + 1),
                        _ => return Err(i),
                    }
                }
            }
            Some(b'[') => {
                let mut i = ws(b, i + 1);
                if b.get(i) == Some(&b']') {
                    return Ok(i + 1);
                }
                loop {
                    i = ws(b, value(b, i)?);
                    match b.get(i) {
                        Some(b',') => i += 1,
                        Some(b']') => return Ok(i + 1),
                        _ => return Err(i),
                    }
                }
            }
            Some(b'"') => string(b, i),
            Some(c) if c.is_ascii_digit() || *c == b'-' => {
                let mut j = i + 1;
                while j < b.len() && (b[j].is_ascii_digit() || b".eE+-".contains(&b[j])) {
                    j += 1;
                }
                Ok(j)
            }
            _ => {
                for w in ["true", "false", "null"] {
                    if b[i..].starts_with(w.as_bytes()) {
                        return Ok(i + w.len());
                    }
                }
                Err(i)
            }
        }
    }
    fn string(b: &[u8], i: usize) -> Result<usize, usize> {
        if b.get(i) != Some(&b'"') {
            return Err(i);
        }
        let mut j = i + 1;
        while j < b.len() {
            match b[j] {
                b'\\' => j += 2,
                b'"' => return Ok(j + 1),
                c if c < 0x20 => return Err(j),
                _ => j += 1,
            }
        }
        Err(j)
    }
    let b = s.trim_end().as_bytes();
    match value(b, 0) {
        Ok(end) if ws(b, end) == b.len() => {}
        Ok(end) | Err(end) => panic!("not JSON at byte {end}: ...{}...", &s[end.saturating_sub(40)..(end + 40).min(s.len())]),
    }
}

#[test]
fn view_json_carries_each_anchor_with_its_intent_and_code() {
    let l = lens::load(&input(ITER, Some(ITER_INTENT), &[("fx/util", UTIL)])).unwrap();
    let text = lens::view_intent(&l);
    let j = lens::view_json(&l, "intent", &text);
    json_ok(&j);
    let h = hash_of(ITER, "iterate");
    assert!(j.starts_with(r#"{"module":"fx/iter","path":"fx/iter.viv","intentFile":"fx/iter.intent","lens":"intent","entry":{"name":"fx/iter","text":"A fixed-point iteration."},"defs":[{"name":"iterate","sig":"iterate(x0, tol)","line":3,"end":20,"#), "{j}");
    assert!(j.contains(&format!(r#""hash":"{h}","stamp":{{"file":"{h}","inline":null}},"intent":{{"what":{{"text":"Find the fixed point of x/2 + 1 [test \"the fixed point of x/2 + 1 is 2\"].","home":"file"}}"#)), "{j}");
    assert!(j.contains(r#"{"kind":"step","id":"update","line":10,"end":12,"code":"        step update {\n            xn = half(x) + 1.0\n        }","text":"apply the map","children":[]}"#), "{j}");
    assert!(j.contains(r#"{"kind":"control","control":"loop","text":"repeat until the step is shorter than tol","children":["#), "{j}");
    assert!(j.ends_with(&format!(",\"text\":{}}}", format!("{text:?}").replace("\\'", "'"))), "the rendered lens rides along");
}

/// A scratch directory that looks like a repository: a `Cargo.toml` marks
/// its root, and `lib` points at this repository's own `lib`, read only.
fn scratch_repo(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_lens_cli_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("fx")).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "").unwrap();
    let lib = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lib").canonicalize().unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(lib, dir.join("lib")).unwrap();
    dir
}

fn viv(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_viv")).args(args).output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stdout).to_string(), String::from_utf8_lossy(&out.stderr).to_string())
}

#[test]
fn the_cli_inits_views_checks_stamps_and_moves() {
    let dir = scratch_repo("flow");
    std::fs::write(dir.join("fx/util.viv"), UTIL).unwrap();
    let prog = dir.join("fx/iter.viv");
    std::fs::write(&prog, ITER).unwrap();
    let p = prog.to_str().unwrap();
    let ip = dir.join("fx/iter.intent");
    // init writes the skeleton and will not overwrite it
    let (ok, out, _) = viv(&["lens", "init", p]);
    assert!(ok, "{out}");
    assert_eq!(out, format!("wrote {} (3 defs)\n", ip.display()));
    let skel = std::fs::read_to_string(&ip).unwrap();
    assert!(skel.starts_with(&format!("module fx/iter\n\niterate  @{}\n", hash_of(ITER, "iterate"))), "{skel}");
    let (ok, _, err) = viv(&["lens", "init", p]);
    assert!(!ok && err.contains("exists; `--force` replaces it"), "{err}");
    // the intent lens prints the skeleton's structure
    let (ok, out, _) = viv(&["view", p, "--lens", "intent"]);
    assert!(ok);
    assert_eq!(out, "module fx/iter\n\niterate\n  what\n\n  guess\n  repeat\n    update\n    measure\n\n_scratch\n  what\n\ncheck_close\n  what\n");
    // check fails on the empty skeleton, then passes once it is written
    let (ok, out, _) = viv(&["lens", "check", p]);
    assert!(!ok);
    assert!(out.ends_with(&format!("lens check {p}: 4 problems\n")), "{out}");
    std::fs::write(&ip, ITER_INTENT.replace("@H", "@00000000")).unwrap();
    let (ok, out, _) = viv(&["lens", "check", p, "--json"]);
    assert!(!ok);
    json_ok(&out);
    assert!(out.contains(r#""kind":"stale""#) && out.ends_with(",\"problems\":1}\n"), "{out}");
    let (ok, out, _) = viv(&["lens", "stamp", p, "iterate"]);
    assert!(ok, "{out}");
    let (ok, out, _) = viv(&["lens", "check", p]);
    assert!(ok, "{out}");
    assert_eq!(out, format!("lens check {p}: ok\n"));
    // move everything inline, then back
    let (ok, out, _) = viv(&["lens", "move", p, "--to-inline"]);
    assert!(ok, "{out}");
    assert!(out.contains("kept: iterate: `return the iterate` has no inline home; it stays in the file"), "{out}");
    let (ok, _, _) = viv(&["lens", "check", p]);
    assert!(ok);
    let (ok, _, _) = viv(&["lens", "move", p, "--to-file"]);
    assert!(ok);
    assert_eq!(std::fs::read_to_string(&prog).unwrap(), ITER);
    let (ok, out, _) = viv(&["lens", "check", p]);
    assert!(ok, "{out}");
    // view --json and explain --json are JSON
    let (ok, out, _) = viv(&["view", p, "--lens", "all", "--json"]);
    assert!(ok);
    json_ok(&out);
    let (ok, out, _) = viv(&["explain", p, "--from", "iterate", "--json"]);
    assert!(ok);
    json_ok(&out);
    // the story lens (L5) runs the program and tells the run by its steps
    let (ok, out, err) = viv(&["view", p, "--lens", "story"]);
    assert!(ok, "{err}");
    assert!(out.starts_with("<main>\n  iterate  Find the fixed point"), "{out}");
    assert!(out.contains("\n    iteration 1\n"), "{out}");
    // an intent file that does not parse stops every verb but check, which reports it
    let good = std::fs::read_to_string(&ip).unwrap();
    std::fs::write(&ip, format!("{good}  stray\n\tbad\n")).unwrap();
    let (ok, _, err) = viv(&["view", p]);
    assert!(!ok && err.contains(&format!("{}:", ip.display())), "{err}");
    let (ok, out, _) = viv(&["lens", "check", p]);
    assert!(!ok && out.contains(": parse: "), "{out}");
    std::fs::write(&ip, good).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_cli_on_the_hartree_fock_library() {
    // read-only: a copy of lib/hf/scf_loop.viv in a scratch repository whose
    // `lib` is this repository's, so its imports of lib/hf resolve
    let dir = scratch_repo("hf");
    let lib = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lib");
    let rhf_src = std::fs::read_to_string(lib.join("hf/scf_loop.viv")).unwrap();
    let rhf = dir.join("fx/scf_loop.viv");
    std::fs::write(&rhf, &rhf_src).unwrap();
    let p = rhf.to_str().unwrap();
    let (ok, out, err) = viv(&["lens", "init", p]);
    assert!(ok, "{out}{err}");
    let skel = std::fs::read_to_string(dir.join("fx/scf_loop.intent")).unwrap();
    assert_eq!(IntentFile::parse(&skel).unwrap().print(), skel, "round trip");
    // every def of the module is in it with the hash `viv hash` prints
    let (_, hashes, _) = viv(&["hash", p]);
    let own: Vec<String> = rhf_src.lines().filter_map(|l| l.strip_prefix("def ")).map(|r| r.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect()).collect();
    assert!(own.len() > 10);
    for name in &own {
        let h = hashes.lines().find(|l| l.ends_with(&format!(" def {name}(")) || l.contains(&format!(" def {name}("))).unwrap().split_whitespace().nth(2).unwrap().to_string();
        assert!(skel.contains(&format!("\n{name}  @{h}\n")), "{name} @{h}");
    }
    let (ok, view, _) = viv(&["view", p, "--lens", "intent"]);
    assert!(ok);
    for name in &own {
        assert!(view.contains(&format!("\n{name}\n")), "{name}");
    }
    // the code lens of a file is its source minus the intent lines and stamps
    let (_, code, _) = viv(&["view", p, "--lens", "code"]);
    assert_eq!(code.lines().count() + rhf_src.lines().filter(|l| { let t = l.trim_start(); t.starts_with("#|") || t.starts_with("#:") || t.starts_with("#@") }).count(), rhf_src.lines().count());
    // explain follows hf_energy through the pipeline down to the SCF loop.
    // lib/hf has steps, and a call is printed inside the step (and the loop)
    // it sits in, one level below it, as explain_walks_the_call_graph checks:
    // hf_energy's `scf` step holds hf_scf; scf's `solve` step holds
    // scf_solve, which calls scf_iterate; the loop's `coulomb_exchange` and
    // `fock` steps hold jk_of and den_fock.
    let hf = lib.join("hf/hf.viv");
    let (ok, out, err) = viv(&["explain", hf.to_str().unwrap(), "--from", "hf_energy"]);
    assert!(ok, "{err}");
    assert!(out.starts_with("hf_energy  The whole pipeline"), "{out}");
    let mut from = 0;
    for line in [
        "\n  scf: ",
        "\n    -> hf_scf  ",
        "\n      -> scf  ",
        "\n        solve: ",
        "\n          -> scf_solve  ",
        "\n            -> scf_iterate  ",
        "\n              repeat ",
        "\n                coulomb_exchange: ",
        "\n                  -> jk_of  ",
        "\n                fock: ",
        "\n                  -> den_fock  ",
    ] {
        let at = out[from..].find(line).unwrap_or_else(|| panic!("no {line:?} after byte {from} in:\n{out}"));
        from += at + line.len();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// Milestones L5 and L6: the debug
// lens (watch, hold, trace), the story lens, and `placeholder` defs.
//
// What is checked here:
// - a debug lens, inline or in a `.debug` file, records into one `__story__`
//   line that both engines write byte for byte, and never changes the
//   program's own output; a release build of either engine records nothing
//   and prints exactly what the program without the lens prints;
// - iterations end on every way out of the loop body, after the steps the
//   body left open, in both engines; watched arrays are whole up to 4096
//   elements and summarized above that;
// - a failed hold is in the story and makes the run fail, in both engines;
// - a def's body hash does not depend on its debug lens;
// - the lens is refused where it could not be honest: a def or step that does
//   not exist, a hold that calls a def, a traced def a `parallel for` reaches;
// - `viv story` reads the line back and prints one line per iteration;
// - a placeholder type-checks as a signature and a call to it is an error
//   naming it, in both engines; lsfunc and explain show it as unwritten.
// ===========================================================================

use viv::story;

/// The program's own output: every line but the engines' data lines.
fn own_output(out: &str) -> String {
    out.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

/// The `__story__` line's JSON.
fn story_of(out: &str) -> String {
    story::story_line(out).unwrap_or_else(|| panic!("no __story__ line in:\n{out}")).to_string()
}

/// Compile and run with the given flags; (exit ok, stdout, stderr). None if no compiler.
fn native_full(src: &str, tag: &str, release: bool) -> Option<(bool, String, String)> {
    let c = c_of(src);
    let dir = std::env::temp_dir().join(format!("viv_lens5_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let mut cmd = Command::new(&cc);
    cmd.args(["-std=c99", "-O2", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-Wno-unused-parameter", "-Wno-unused-function"]);
    if release {
        cmd.arg("-DVIV_RELEASE");
    }
    let out = cmd.arg(dir.join("p.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}\n--- C ---\n{c}", String::from_utf8_lossy(&out.stderr));
    let warnings = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(!warnings.contains("warning"), "the C for the debug lens should compile cleanly:\n{warnings}");
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some((run.status.success(), String::from_utf8_lossy(&run.stdout).to_string(), String::from_utf8_lossy(&run.stderr).to_string()))
}

/// A program with its debug lens, and the same program without it. Both
/// engines must agree on the debug run (story included); the lens must not
/// change the program's output, a release run of either engine, or the C of
/// the program without it. Returns the debug run's output.
fn lens_both(with: &str, without: &str, tag: &str) -> String {
    let i = interp(with, false);
    let plain = interp(without, false);
    assert_eq!(own_output(&i), own_output(&plain), "the debug lens changed the program's output");
    assert!(i.contains("__story__ "), "{i}");
    assert!(!plain.contains("__story__"), "no lens, no story");
    let (ir, plain_r) = (interp(with, true), interp(without, true));
    assert_eq!(ir, plain_r, "a release run ignores the debug lens");
    assert!(!ir.contains("__story__"));
    if let (Some(n), Some(nr), Some(pr)) = (native(with, tag, false), native(with, &format!("{tag}_r"), true), native(without, &format!("{tag}_p"), true)) {
        assert_eq!(n, i, "the interpreter and the compiled program disagree");
        assert_eq!(nr, pr, "a release build ignores the debug lens");
        assert_eq!(nr, ir, "the release builds disagree");
    }
    i
}

/// The loop of a Hartree-Fock-like SCF in a few lines: steps in a loop, a
/// value that converges, and a helper the story can trace.
const SCF_CODE: &str = r#"
def half(x) {
    return x / 2.0
}

def run(n) {
    x = 1.0
    v = [1.0, 2.0]
    it = 0
    step start "begin at one" {
        y = x
    }
    while it < n {
        step shrink "halve x" {
            y = x
            x = half(y)
        }
        step count {
            it = it + 1
            v[0] = x
        }
        if it == 3 {
            break
        }
    }
    return x
}
print(run(5))
"#;

const SCF_LENS: &str = r#"
debug run {
    watch per iteration x, it, v
    hold x > 0.0 "x stays positive"
    hold at shrink x == y / 2.0 within 1e-15 "x halves"
    watch at start y
    trace half
}
"#;

#[test]
fn a_debug_lens_records_the_story_in_both_engines_and_changes_no_output() {
    let with = format!("{SCF_CODE}{SCF_LENS}");
    let out = lens_both(&with, SCF_CODE, "story");
    let st = story_of(&out);
    // calls, steps, iterations and watches, in the order they happened
    assert!(st.starts_with(r#"{"total":38,"events":[["call","run"],["enter","run","start",1],["watch","run","start",{"y":1.00000000000000000e+00}],["exit","run","start",1],["iter","run",1],["enter","run","shrink",1],["call","half"],["ret","half"],["hold","run","shrink",true,"x halves",0.00000000000000000e+00],["exit","run","shrink",1],"#), "{st}");
    assert!(st.contains(r#"["watch","run","iteration",{"x":5.00000000000000000e-01,"it":1.00000000000000000e+00,"v":{"shape":[2],"values":[5.00000000000000000e-01,2.00000000000000000e+00]}}],["hold","run","iteration",true,"x stays positive",null],["iter_end","run"]"#), "{st}");
    // the third iteration ends at the `break`, and still records its end
    assert!(st.ends_with(r#"["iter","run",3],["enter","run","shrink",1],["call","half"],["ret","half"],["hold","run","shrink",true,"x halves",0.00000000000000000e+00],["exit","run","shrink",1],["enter","run","count",1],["exit","run","count",1],["watch","run","iteration",{"x":1.25000000000000000e-01,"it":3.00000000000000000e+00,"v":{"shape":[2],"values":[1.25000000000000000e-01,2.00000000000000000e+00]}}],["hold","run","iteration",true,"x stays positive",null],["iter_end","run"],["ret","run"]]}"#), "{st}");
}

#[test]
fn the_debug_file_and_the_inline_block_are_the_same_lens() {
    // the `.debug` layout, turned into blocks line for line
    let file = "# checks\nrun  @0000abcd  by agent\n  watch per iteration  x, it, v\n  hold                 x > 0.0   \"x stays positive\"\n  hold at shrink       x == y / 2.0 within 1e-15  \"x halves\"\n  watch at start       y\n  trace                half\n";
    let blocks = viv::debug_lens::file_to_source(file);
    assert_eq!(blocks.lines().count(), file.lines().count(), "one line for one line, so errors name the file's lines");
    let from_file = interp(&format!("{SCF_CODE}\n{blocks}"), false);
    let inline = interp(&format!("{SCF_CODE}{SCF_LENS}"), false);
    assert_eq!(story_of(&from_file), story_of(&inline));
    // and a `debug { }` block directly before the def is the same again
    let before = SCF_CODE.replace("def run(n) {", &format!("debug {{\n{}\n}}\ndef run(n) {{", SCF_LENS.lines().filter(|l| l.starts_with("    ")).collect::<Vec<_>>().join("\n")));
    assert_eq!(story_of(&interp(&before, false)), story_of(&inline));
    // the entries, with their stamps, for `lens check`
    let entries = viv::debug_lens::file_entries(file);
    assert_eq!(entries.len(), 1);
    assert_eq!((entries[0].name.as_str(), entries[0].line, entries[0].stamp.as_deref()), ("run", 2, Some("0000abcd")));
    assert_eq!(entries[0].at, vec![(5, "shrink".to_string()), (6, "start".to_string())]);
    // a file that does not parse names its own line
    let err = viv::debug_lens::check_file("run\n  watch x\n  hold x > 0 no sentence\n").unwrap_err();
    assert!(err.starts_with("3: a hold ends with the sentence it checks"), "{err}");
}

#[test]
fn a_failed_hold_fails_the_run_and_is_in_the_story() {
    let with = format!("{SCF_CODE}{}", SCF_LENS.replace("x > 0.0", "x > 0.2"));
    let (out, fails, holds) = viv::interp::run_checked_holds(&with, false).unwrap();
    assert_eq!((fails, holds), (0, 1), "one hold failed, in the third iteration");
    assert_eq!(own_output(&out), own_output(&interp(SCF_CODE, false)), "a failed hold prints nothing");
    assert!(story_of(&out).contains(r#"["hold","run","iteration",false,"x stays positive",null]"#));
    if let Some((ok, n, err)) = native_full(&with, "hold", false) {
        assert!(!ok, "the compiled program fails too");
        assert_eq!(err, "1 hold FAILED\n");
        assert_eq!(n, interp(&with, false));
    }
    if let Some((ok, _, err)) = native_full(&with, "hold_r", true) {
        assert!(ok && err.is_empty(), "a release build checks no holds");
    }
    // the CLI's exit code
    let dir = std::env::temp_dir().join(format!("viv_lens5_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("h.viv");
    std::fs::write(&p, &with).unwrap();
    let (ok, out, err) = viv(&["run", p.to_str().unwrap()]);
    assert!(!ok && err == "1 hold FAILED\n", "{err}");
    assert!(out.starts_with("0.125\n"));
    let (ok, _, _) = viv(&["run", "--release", p.to_str().unwrap()]);
    assert!(ok, "a release run checks no holds");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn iterations_end_on_every_way_out_in_both_engines() {
    let code = r#"
def f(n) {
    acc = 0.0
    for i to n {
        step add {
            acc = acc + i
            if i == 1 {
                continue
            }
            if i == 3 {
                return acc
            }
        }
        step more {
            acc = acc + 0.5
        }
    }
    return acc
}
step top "the program" {
    for k to 2 {
        step call {
            show f(5)
        }
    }
}
"#;
    let lens = "debug f {\n    watch per iteration acc, i\n}\n";
    let out = lens_both(&format!("{code}{lens}"), code, "ways");
    let st = story_of(&out);
    // `continue` in iteration 2 and `return` in iteration 4: each closes the
    // open step, then ends the iteration
    assert!(st.contains(r#"["iter","f",2],["enter","f","add",1],["exit","f","add",1],["watch","f","iteration",{"acc":1.50000000000000000e+00,"i":1.00000000000000000e+00}],["iter_end","f"]"#), "{st}");
    assert!(st.contains(r#"["iter","f",4],["enter","f","add",1],["exit","f","add",1],["watch","f","iteration",{"acc":7.00000000000000000e+00,"i":3.00000000000000000e+00}],["iter_end","f"],["ret","f"]"#), "{st}");
    // the top level's loop holds a step, so its iterations are recorded too
    assert!(st.starts_with(r#"{"total":"#) && st.contains(r#"["enter","<main>","top",1],["iter","<main>",1],["enter","<main>","call",2],["call","f"]"#), "{st}");
}

#[test]
fn watched_arrays_are_whole_up_to_4096_elements_then_summarized() {
    let code = r#"
def grid(n) {
    A = zeros(n, n)
    for i to n {
        for j to n {
            A[i, j] = 1.0 / (1.0 + i + j)
        }
    }
    return A
}
def g() {
    step small {
        S = grid(64)
    }
    step big {
        B = grid(65)
    }
    return B[0, 0] + S[0, 0]
}
print(g())
"#;
    let lens = "debug g {\n    watch at small S\n    watch at big B\n}\n";
    let st = story_of(&lens_both(&format!("{code}{lens}"), code, "arrays"));
    assert!(st.contains(r#"["watch","g","small",{"S":{"shape":[64,64],"values":[1.00000000000000000e+00,5.00000000000000000e-01,"#), "64 x 64 = 4096 is whole");
    assert_eq!(st.matches("e-").count() + st.matches("e+").count() >= 4096, true);
    let big = st.split(r#"["watch","g","big",{"B":"#).nth(1).unwrap();
    assert!(big.starts_with(r#"{"shape":[65,65],"min":7.75193798449612392e-03,"max":1.00000000000000000e+00,"norm":"#), "{big}");
    assert!(big.contains(r#","trace":"#));
}

#[test]
fn the_body_hash_does_not_depend_on_the_debug_lens() {
    let with = format!("{SCF_CODE}{SCF_LENS}");
    let h = |src: &str| viv::def_hashes(src).unwrap().into_iter().map(|h| (h.name, h.body_short)).collect::<Vec<_>>();
    assert_eq!(h(&with), h(SCF_CODE));
    // and the doc tools read the def the same way
    assert_eq!(viv::lsfunc(&with), viv::lsfunc(SCF_CODE));
    let before = SCF_CODE.replace("def run(n) {", "#| Halve x until it is small.\ndebug {\n    watch x\n}\ndef run(n) {");
    assert!(viv::lsfunc(&before).contains("run(n)\n          Halve x until it is small.\n"), "a def's doc above its debug block still reaches it");
    // the code lens leaves an inline debug block out, as it leaves out `#|` lines
    assert_eq!(lens::view_code(&before), lens::view_code(SCF_CODE));
    // a program without a lens has no story machinery in its C
    assert!(!c_of(SCF_CODE).contains("viv_story"));
    assert!(!format!("{:?}", viv::parse(SCF_CODE, SourceId(0)).unwrap()).contains("debug"), "--emit ast prints as before");
}

#[test]
fn the_lens_is_refused_where_it_cannot_be_honest() {
    let err = |lens: &str| -> String {
        match viv::interp::run(&format!("{SCF_CODE}{lens}")) {
            Ok(o) => panic!("should be refused:\n{lens}\n{o}"),
            Err(e) => {
                let c = viv::compile_c(&format!("{SCF_CODE}{lens}")).unwrap_err();
                assert_eq!(c, e, "both engines refuse it the same way");
                e
            }
        }
    };
    assert!(err("debug nosuch {\n    watch x\n}\n").contains("names `nosuch`, which is not a def"));
    assert!(err("debug run {\n    watch at nosuch x\n}\n").contains("`run` has no step `nosuch`"));
    assert!(err("debug half {\n    watch x\n}\n").contains("`half` has no loop for a per-iteration watch or hold"));
    assert!(err("debug run {\n    hold half(x) > 0.0 \"no calls\"\n}\n").contains("a hold cannot call `half`"));
    assert!(err("debug run {\n    hold print(x) > 0.0 \"no printing\"\n}\n").contains("a hold cannot call `print`"));
    assert!(err("debug run {\n    watch half\n}\n").contains("`half` is a def, not a value to watch"));
    assert!(err("debug * {\n    watch x\n}\n").contains("`debug * { }` can only trace"));
    assert!(err("debug run {\n    hold x > 0.0 within 1.0 \"x\"\n}\n").contains("`within` needs `a == b` before it"));
    assert!(err("debug {\n    watch x\n}\nprint(1)\n").contains("must come right before the def"));
    // a traced def reached from a parallel loop would record from several threads
    let par = "def sq(x) {\n    return x * x\n}\ns = zeros(4)\nparallel for i to 4 {\n    s[i] = sq(i)\n}\ndebug * {\n    trace sq\n}\n";
    let e = viv::interp::run(par).unwrap_err();
    assert!(e.contains("a `parallel for` reaches a `step` through `sq`"), "{e}");
    // without the lens the same program runs
    assert!(viv::interp::run(&par[..par.find("debug").unwrap()]).is_ok());
}

/// The whole output of the direct wasm backend, run in node against
/// viv.wasm's runtime; None when node or a viv.wasm with the runtime is
/// missing (www/build-wasm.sh).
fn direct_wasm(src: &str, tag: &str) -> Option<String> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let wasm = std::env::var("VIV_WASM").map(std::path::PathBuf::from).unwrap_or_else(|_| root.join("www/public/viv/viv.wasm"));
    let runtime = std::fs::read(&wasm).ok()?;
    if !runtime.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    let bytes = viv::compile_wasm(src).expect("compiles to wasm");
    let dir = std::env::temp_dir().join(format!("viv_lens_wasm_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let m = dir.join("a.wasm");
    std::fs::write(&m, bytes).unwrap();
    let out = Command::new("node").arg(root.join("tools/run-wasm.mjs")).arg(&m).env("VIV_WASM", &wasm).output().ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A loop whose debug lens watches values the code does not hold: the change
/// of `e` taken where `e` is computed (before `eprev` catches up), a norm, and
/// a hold over the computed change.
const COMPUTED_CODE: &str = r#"
def f(n) {
    e = 0.0
    eprev = 0.0
    v = [3.0, 4.0]
    for i to n {
        step energy {
            e = e + 1.0 / (i + 1.0)
        }
        if e - eprev < 1.0e-9 {
            break
        }
        step next {
            eprev = e
            v[0] = v[0] * 0.5
        }
    }
    return e
}
print(f(3))
"#;

const COMPUTED_LENS: &str = "debug f {\n    watch at energy de = e - eprev\n    watch per iteration e, de, nv = sqrt(sum(v * v))\n    hold at energy de > 0.0 \"e rises\"\n}\n";

#[test]
fn a_watch_computes_a_value_the_code_does_not_hold_in_every_engine() {
    let with = format!("{COMPUTED_CODE}{COMPUTED_LENS}");
    // the interpreter and C, debug and release: the lens changes no output,
    // and a release build computes and records nothing
    let out = lens_both(&with, COMPUTED_CODE, "computed");
    let st = story_of(&out);
    // de where it is computed, and again at the iteration's end, where the
    // code's own e - eprev is 0
    assert!(st.contains(r#"["watch","f","energy",{"de":1.00000000000000000e+00}],["hold","f","energy",true,"e rises",null]"#), "{st}");
    assert!(st.contains(r#"["watch","f","iteration",{"e":1.00000000000000000e+00,"de":1.00000000000000000e+00,"nv":4.27200187265876519e+00}]"#), "{st}");
    assert!(st.contains(r#"["watch","f","iteration",{"e":1.50000000000000000e+00,"de":5.00000000000000000e-01,"#), "{st}");
    // the direct wasm backend records the same story
    match direct_wasm(&with, "computed") {
        Some(w) => assert_eq!(story_of(&w), st, "wasm tells a different story"),
        None => eprintln!("skip wasm: no node or no viv.wasm with the runtime (www/build-wasm.sh)"),
    }
    // a computed watch takes a name of its own and reads, as a hold does
    let err = |lens: &str| viv::interp::run(&format!("{COMPUTED_CODE}{lens}")).unwrap_err();
    assert!(err("debug f {\n    watch at energy e = eprev + 1.0\n}\n").contains("`e` is a variable of the program"));
    assert!(err("debug f {\n    watch at energy w = f(1)\n}\n").contains("a watch cannot call `f`"));
}

#[test]
fn the_story_prints_one_line_per_iteration() {
    let out = interp(&format!("{SCF_CODE}{SCF_LENS}"), false);
    let st = story::parse(&story_of(&out)).unwrap();
    assert_eq!((st.total, st.kept), (38, 38));
    assert_eq!(st.hold_counts(), (6, 0));
    let mut words = story::Words::default();
    words.steps.insert("run/shrink".into(), "halve x".into());
    words.steps.insert("run/start".into(), "begin at one".into());
    words.what.insert("run".into(), "Halve x a few times.".into());
    let text = story::render_text(&st, &words, story::Opts::default());
    assert_eq!(
        text,
        "<main>\n  run             Halve x a few times\n    begin at one  y 1\n    iteration 1   x 0.5  it 1  v [0.5, 2]\n    iteration 2   x 0.25  it 2  v [0.25, 2]\n    iteration 3   x 0.125  it 3  v [0.125, 2]\nholds: 6 checked, 0 failed\n"
    );
    // expanded, each iteration shows its steps and the calls in them
    let text = story::render_text(&st, &words, story::Opts { depth: None, expand: true });
    assert!(text.contains("    iteration 1   x 0.5  it 1  v [0.5, 2]\n      halve x\n        half\n      count\n"), "{text}");
    // --depth 1: the calls the top level makes, not the calls they make
    let text = story::render_text(&st, &words, story::Opts { depth: Some(1), expand: true });
    assert!(text.contains("      halve x\n      count\n"), "{text}");
    // a failed hold shows on its iteration's line even when it is collapsed
    let bad = interp(&format!("{SCF_CODE}{}", SCF_LENS.replace("x > 0.0", "x > 0.2")), false);
    let text = story::render_text(&story::parse(&story_of(&bad)).unwrap(), &words, story::Opts::default());
    assert!(text.contains("    iteration 3   x 0.125  it 3  v [0.125, 2]\n      hold FAILED at iteration: x stays positive\n"), "{text}");
    assert!(text.ends_with("holds: 6 checked, 1 failed\n"));
    // the JSON form is JSON, with the words filled in
    let j = story::render_json(&st, &words);
    json_ok(&j);
    assert!(j.contains(r#"{"kind":"step","def":"run","id":"shrink","text":"halve x","watch":{},"holds":[{"at":"shrink","ok":true,"sentence":"x halves","worst":0.00000000000000000e+00}],"#), "{j}");
}

#[test]
fn a_placeholder_is_a_signature_and_calling_it_is_an_error_naming_it() {
    let src = "#| Density-fitted J and K.\nplaceholder def jk_df(X: Arr, D: Arr) -> Arr\nplaceholder def guess(n)\n\ndef run(D: Arr) {\n    J = jk_df(D, D)\n    return J[0] + guess(1.0)\n}\nprint(2.0)\n";
    // it type-checks as a signature: a call with the wrong arity or the wrong
    // use of its result is refused at compile time
    assert!(viv::compile_c(src).is_ok());
    assert!(viv::compile_c(&src.replace("jk_df(D, D)", "jk_df(D)")).is_err());
    assert!(viv::compile_c(&src.replace("J[0] + guess(1.0)", "guess(1.0)[0]")).is_err(), "guess returns a number");
    // not called, it costs nothing; called, it fails naming itself, in both engines
    assert_eq!(interp(src, false), "2\n");
    let called = format!("{src}print(run(zeros(2)))\n");
    let e = viv::interp::run(&called).unwrap_err();
    assert_eq!(e, "`jk_df` is a placeholder: it has a signature and no body yet");
    if let Some((ok, out, err)) = native_full(&called, "ph", false) {
        assert!(!ok);
        assert_eq!(out, "2\n");
        assert_eq!(err, format!("error: {e}\n"));
    }
    // a placeholder with a body is refused
    assert!(viv::interp::run("placeholder def f(x) {\n    return x\n}\n").unwrap_err().contains("placeholder `f` has a body"));
    // lsfunc lists it, and `--placeholders` lists only them
    let ls = viv::lsfunc(src);
    assert!(ls.contains("   2-2  jk_df(X: Arr, D: Arr) -> Arr  [placeholder: no body yet]\n          Density-fitted J and K.\n"), "{ls}");
    assert!(ls.contains("   5-8  run(D: Arr)\n"), "{ls}");
    let only = viv::lsfunc_placeholders(src);
    assert_eq!(only, "   2-2  jk_df(X: Arr, D: Arr) -> Arr  [placeholder: no body yet]\n          Density-fitted J and K.\n   3-3  guess(n)  [placeholder: no body yet]\n");
    assert_eq!(viv::lsfunc_placeholders("def f(x) {\n    return x\n}\n"), "no placeholders\n");
    assert!(viv::doc_json(src).contains(r#""name":"jk_df","line":1,"end":1,"rawLine":2,"rawEnd":2,"#) && viv::doc_json(src).contains(r#","placeholder":true}"#));
    // explain shows it in the call graph, unwritten
    let input = Input { root: ModuleText { name: "fx/ph".into(), path: "fx/ph.viv".into(), src: src.to_string(), intent_path: "fx/ph.intent".into(), intent: None }, used: vec![], expanded: src.to_string(), companion_tests: vec![] };
    let l = lens::load(&input).unwrap();
    let text = lens::explain_text(&lens::explain(&l, Some("run"), None).unwrap());
    assert_eq!(text, "run\n  -> jk_df  Density-fitted J and K.  (unwritten: a placeholder)\n  -> guess  (unwritten: a placeholder)\n");
    let text = lens::explain_text(&lens::explain(&l, Some("guess"), None).unwrap());
    assert_eq!(text, "guess  (unwritten: a placeholder)\n");
    // its body hash is its signature's, stable
    assert_eq!(hash_of(src, "jk_df"), hash_of(&src.replace("#| Density-fitted J and K.\n", ""), "jk_df"));
}

#[test]
fn lens_check_holds_the_debug_file_to_the_code() {
    let src = "def f(n) {\n    x = 0.0\n    for i to n {\n        step add {\n            x = x + i\n        }\n    }\n    return x\n}\nprint(f(3))\n";
    let input = Input { root: ModuleText { name: "fx/d".into(), path: "fx/d.viv".into(), src: src.to_string(), intent_path: "fx/d.intent".into(), intent: None }, used: vec![], expanded: src.to_string(), companion_tests: vec![] };
    let l = lens::load(&input).unwrap();
    let h = hash_of(src, "f");
    let kinds = |text: &str| -> Vec<String> { lens::check_debug(&l, "fx/d.debug", text).iter().map(|f| format!("{}:{}", f.line, f.kind)).collect() };
    assert_eq!(kinds(&format!("f  @{h}\n  watch x\n")), Vec::<String>::new());
    assert_eq!(kinds("f\n  watch x\n"), vec!["1:unstamped"]);
    assert_eq!(kinds("f  @00000000\n  watch x\n"), vec!["1:stale"]);
    assert_eq!(kinds(&format!("g  @{h}\n  watch x\n")), vec!["1:orphan"]);
    assert_eq!(kinds(&format!("f  @{h}\n  watch at nope x\n")), vec!["2:step"]);
    assert_eq!(kinds(&format!("f  @{h}\n  hold x \n")), vec!["2:parse"]);
    assert_eq!(kinds(&format!("f  @{h}\n  watch x\nf  @{h}\n  watch i\n")), vec!["3:duplicate"]);
}

#[test]
fn the_example_tells_the_same_story_in_both_engines() {
    let ex = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/lenses/scf_demo.viv");
    let p = ex.to_str().unwrap();
    // it runs clean, its lenses check clean, and its placeholder is listed
    let (ok, out, err) = viv(&["run", p]);
    assert!(ok, "{err}");
    assert!(out.starts_with("-1.78011\n33\nPASS "), "{out}");
    let (ok, out, _) = viv(&["lens", "check", p]);
    assert!(ok, "{out}");
    let (_, out, _) = viv(&["lsfunc", p, "--placeholders"]);
    assert!(out.starts_with("  43-43  diis(F: Arr, D: Arr) -> Arr  [placeholder: no body yet]\n"), "{out}");
    // the story: the intent's words, one line per iteration
    let (ok, text, err) = viv(&["story", p]);
    assert!(ok, "{err}");
    assert!(text.starts_with("<main>\n  scf  "), "{text}");
    assert!(text.contains("    start from the lowest orbital of the core Hamiltonian  D 3x3 |2|\n"), "{text}");
    assert!(text.contains("    iteration 1 "), "{text}");
    assert!(text.contains("    iteration 33 "), "{text}");
    assert!(!text.contains("iteration 34"));
    assert!(text.ends_with("holds: 99 checked, 0 failed\n"), "{text}");
    // the compiled program tells it byte for byte (skipped without a compiler)
    if Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into())).arg("--version").output().is_ok() {
        for extra in [&[][..], &["--json"][..], &["--expand"][..]] {
            let mut a = vec!["story", p];
            a.extend_from_slice(extra);
            let (_, i, _) = viv(&a);
            a.extend_from_slice(&["--engine", "c"]);
            let (ok, c, err) = viv(&a);
            assert!(ok, "{err}");
            assert_eq!(c, i, "{extra:?}");
        }
    }
}

// --- L7: the viewer's lenses (tools/viewer.mjs) -------------------------------
//
// The viewer keeps "the same def and step in view" across lenses by anchors:
// `def` and `def/step`, read from `viv view --json`, `viv story --json`,
// `viv explain --json` and `viv view --lens debug`. These tests hold the
// binary to what the viewer relies on, then run the viewer's own test.

use viv::story::Json;

fn jget<'a>(j: &'a Json, k: &str) -> Option<&'a Json> {
    if let Json::Obj(fs) = j { fs.iter().find(|(x, _)| x == k).map(|(_, v)| v) } else { None }
}
fn jstr(j: &Json, k: &str) -> String {
    match jget(j, k) {
        Some(Json::Str(s)) => s.clone(),
        _ => String::new(),
    }
}
fn jnum(j: &Json, k: &str) -> i64 {
    match jget(j, k) {
        Some(Json::Num(n)) => n.parse().unwrap(),
        _ => -1,
    }
}
fn jkids<'a>(j: &'a Json, k: &str) -> &'a [Json] {
    match jget(j, k) {
        Some(Json::Arr(xs)) => xs,
        _ => &[],
    }
}
fn json_of(args: &[&str]) -> Json {
    let (_, out, err) = viv(args);
    viv::story::parse_json(out.trim()).unwrap_or_else(|e| panic!("{args:?}: {e}\n{out}\n{err}"))
}

#[test]
fn every_lens_names_its_places_by_the_anchors_the_code_lens_has() {
    let ex = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/lenses/scf_demo.viv");
    let p = ex.to_str().unwrap();

    // the code lens: every def and step, with line ranges that nest
    let view = json_of(&["view", p, "--lens", "code", "--json"]);
    let mut anchors: Vec<String> = Vec::new();
    fn steps(def: &str, items: &[Json], lo: i64, hi: i64, out: &mut Vec<String>) {
        for o in items {
            let (line, end) = (jnum(o, "line"), jnum(o, "end"));
            if jstr(o, "kind") == "step" {
                assert!(lo <= line && line <= end && end <= hi, "{def}/{} at {line}-{end} is outside {lo}-{hi}", jstr(o, "id"));
                out.push(format!("{def}/{}", jstr(o, "id")));
                steps(def, jkids(o, "children"), line, end, out);
            } else {
                steps(def, jkids(o, "children"), lo, hi, out);
            }
        }
    }
    for d in jkids(&view, "defs") {
        let name = jstr(d, "name");
        if name == "<main>" {
            continue;
        }
        anchors.push(name.clone());
        steps(&name, jkids(d, "outline"), jnum(d, "line"), jnum(d, "end"), &mut anchors);
    }
    assert_eq!(
        anchors.join(" "),
        "fock density energy diis scf scf/guess scf/fock scf/solve scf/density scf/mix scf/energy scf/check"
    );
    let known = |a: &str| a == "<main>" || anchors.iter().any(|x| x == a);

    // the story: every call and step it tells is one of those anchors
    let story = json_of(&["story", p, "--json"]);
    let mut told = Vec::new();
    fn walk_story(n: &Json, out: &mut Vec<String>) {
        match jstr(n, "kind").as_str() {
            "call" | "main" => out.push(jstr(n, "def")),
            "step" => out.push(format!("{}/{}", jstr(n, "def"), jstr(n, "id"))),
            _ => {}
        }
        for c in jkids(n, "children") {
            walk_story(c, out);
        }
    }
    walk_story(jget(&story, "root").unwrap(), &mut told);
    assert!(told.iter().any(|a| a == "scf/fock"), "{told:?}");
    for a in &told {
        assert!(known(a), "the story tells {a}, which the code lens does not have");
    }

    // explain: a step belongs to the call it sits in
    // the envelope's result, every child kept
    let ex = json_of(&["explain", p, "--json", "--max-lines", "0"]).get("result").cloned().expect("explain --json has a result");
    let mut walked = Vec::new();
    fn walk_explain(n: &Json, def: &str, out: &mut Vec<String>) {
        let mut inner = def.to_string();
        match jstr(n, "kind").as_str() {
            "call" | "def" => {
                inner = jstr(n, "name");
                out.push(inner.clone());
            }
            "step" => out.push(format!("{def}/{}", jstr(n, "name"))),
            _ => {}
        }
        for c in jkids(n, "children") {
            walk_explain(c, &inner, out);
        }
    }
    walk_explain(&ex, "", &mut walked);
    assert!(walked.iter().any(|a| a == "scf/fock"), "{walked:?}");
    for a in &walked {
        assert!(known(a), "explain walks {a}, which the code lens does not have");
    }

    // the debug lens: a def per unindented line, `watch at` / `hold at` a step of it
    let (ok, dbg, err) = viv(&["view", p, "--lens", "debug"]);
    assert!(ok, "{err}");
    let mut def = String::new();
    let mut at = Vec::new();
    for l in dbg.lines() {
        if !l.starts_with(' ') && !l.is_empty() {
            def = l.split_whitespace().next().unwrap().to_string();
            at.push(def.clone());
        } else if let Some(rest) = l.trim_start().strip_prefix("hold at ").or_else(|| l.trim_start().strip_prefix("watch at ")) {
            at.push(format!("{def}/{}", rest.split_whitespace().next().unwrap()));
        }
    }
    assert_eq!(at.join(" "), "scf scf/fock scf/density scf/guess scf/check");
    for a in &at {
        assert!(known(a), "the debug lens names {a}, which the code lens does not have");
    }
}

#[test]
fn the_viewer_shows_four_lenses_on_one_anchor() {
    // tools/viewer.test.mjs starts the viewer on a free port against this
    // checkout with the binary built here; it needs node (and, for the key and
    // scroll-lock checks, CHROME=<headless Chromium>)
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("skip: no node");
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("node").arg(root.join("tools/viewer.test.mjs")).env("VIV", env!("CARGO_BIN_EXE_viv")).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.ends_with("all passed\n") || text.starts_with("skip:"), "{text}");
}

/// The fixture's intent with math: a glossary symbol, a def-level equation,
/// an equation under a line anchor, a multi-line equation, and `run` and
/// `predict` lines with values from the run.
const MATH_INTENT: &str = "module fx/iter
  A fixed-point iteration.

iterate  @H
  what  Find the fixed point of $x/2 + 1$ [test \"the fixed point of x/2 + 1 is 2\"].
  x     $x_k$  the current iterate

  $$ x_{k+1} = \\tfrac12 x_k + 1 $$  [test \"the fixed point of x/2 + 1 is 2\"]
  guess    start from x0
  repeat until the step is shorter than tol
    update   apply the map
      = xn  the next iterate
        $$ x_{n} = \\frac{x}{2} + 1 $$  [hold \"the map halves the error\"]
      predict  from x0 = 0, what is the first xn?
      run  the first update gives {xn}, the third {xn#3:6}
    measure  measure the step and move
      $$ \\delta = \\left|x_{n} -
           x\\right| $$
  return the iterate
";

#[test]
fn equations_run_lines_and_symbols_parse_and_round_trip() {
    let text = MATH_INTENT.replace("@H", "@0123abcd");
    let f = IntentFile::parse(&text).expect("parses");
    assert_eq!(f.print(), text);
    let (_, entries) = f.model();
    let e = &entries[0];
    assert_eq!(viv::tex::split_symbol(&e.gloss[0].text), (Some("x_k".to_string()), "the current iterate".to_string()));
    assert_eq!(e.details.len(), 1);
    assert_eq!((e.details[0].label.as_str(), e.details[0].under.as_str()), ("$$", ""));
    let update = &e.outline[1].children[0];
    let got: Vec<(&str, &str, &str)> = update.details.iter().map(|d| (d.label.as_str(), d.anchor.as_str(), d.under.as_str())).collect();
    assert_eq!(got, [("=", "xn", ""), ("$$", "", "xn"), ("predict", "", ""), ("run", "", "")]);
    assert_eq!(lens::placeholders(&update.details[3].text), [("xn".to_string(), 1, 4), ("xn".to_string(), 3, 6)]);
    let measure = &e.outline[1].children[1];
    assert_eq!(measure.details[0].text, "$$ \\delta = \\left|x_{n} -\nx\\right| $$");
    // the checker is content with all of it
    let fs = check(ITER, Some(MATH_INTENT));
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
    // numbers inside math need no citation
    let fs = check(ITER, Some(&MATH_INTENT.replace("  x     $x_k$", "  why   It contracts by $\\tfrac{1}{2}$ per step.\n  x     $x_k$")));
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn a_paragraph_after_an_equation_belongs_to_the_prose_not_the_display() {
    // `>` is another paragraph of the prose above it; with an equation in
    // between, it must not be appended to the equation and typeset as math
    let text = MATH_INTENT
        .replace("@H", "@0123abcd")
        .replace("[hold \"the map halves the error\"]\n", "[hold \"the map halves the error\"]\n        > Halving is why it converges.\n");
    let f = IntentFile::parse(&text).expect("parses");
    assert_eq!(f.print(), text);
    let (_, entries) = f.model();
    let update = &entries[0].outline[1].children[0];
    assert_eq!(update.details[0].text, "the next iterate\nHalving is why it converges.");
    assert!(!update.details[1].text.contains("Halving"), "{}", update.details[1].text);
}

#[test]
fn math_failures_are_reported() {
    let cases: Vec<(&str, String, &str)> = vec![
        ("a command KaTeX lacks", MATH_INTENT.replace("$x/2 + 1$", "$\\fracc{x}{2}$"), "tex iterate"),
        ("an unclosed dollar", MATH_INTENT.replace("$x/2 + 1$", "$x/2 + 1"), "tex iterate"),
        ("a bad glossary symbol", MATH_INTENT.replace("$x_k$", "$x_{k$"), "tex iterate"),
        ("unbalanced \\left", MATH_INTENT.replace("\\left|x_{n}", "|x_{n}"), "tex iterate"),
        ("an equation citing a test that does not exist", MATH_INTENT.replace("+ 1 $$  [test \"the fixed point of x/2 + 1 is 2\"]", "+ 1 $$  [test \"no such test\"]"), "citation iterate"),
        ("a value the def does not have", MATH_INTENT.replace("{xn#3:6}", "{xm}"), "placeholder iterate/update"),
    ];
    for (what, intent, want) in cases {
        let fs = check(ITER, Some(&intent));
        assert_eq!(kinds(&fs), [want], "{what}:\n{}", fs.iter().map(|f| f.render()).collect::<String>());
    }
    // a hold citation names a hold of the debug lens
    let l = lens::load(&input(ITER, Some(MATH_INTENT), &[("fx/util", UTIL)])).unwrap();
    assert!(lens::check_hold_citations(&l, &["the map halves the error".to_string()]).is_empty());
    assert_eq!(kinds(&lens::check_hold_citations(&l, &["something else".to_string()])), ["citation iterate"]);
}

#[test]
fn equation_parse_errors_name_the_line() {
    let err = |t: &str| IntentFile::parse(t).err().map(|e| (e[0].line, e[0].message.clone())).expect("should not parse");
    let open = err("f\n\n  s  one\n    $$ x = \n      y\n");
    assert_eq!(open.0, 4);
    assert!(open.1.contains("never closed"));
    assert!(err("f\n\n  s  one\n    = x  the x\n      t  a step under an anchor\n").1.contains("under a line anchor"));
    // `run` at the top of an outline is a step id, as `why` is
    let f = IntentFile::parse("f\n\n  run  run the thing\n").unwrap();
    assert!(matches!(&f.model().1[0].outline[0].kind, lens::NodeKind::Step(id) if id == "run"));
}

// ---------------------------------------------------------------------------
// Variable docs, `documents every variable`, and reading paths
// ---------------------------------------------------------------------------

/// ITER_INTENT with every variable of iterate documented, one of them at a
/// later use, and a path through it.
const DOCUMENTED: &str = "module fx/iter  documents every variable
  A fixed-point iteration.

iterate  @H
  what       Find the fixed point of x/2 + 1 [test \"the fixed point of x/2 + 1 is 2\"].
  x0         $x_0$  the starting point
  tol        how short a step ends the iteration
  x          $x$  the current iterate
  dx         the length of the last step
  dx@measure the length of the step just taken
  xn         the next iterate
  n          how many steps have been taken

  guess    start from x0
  repeat until the step is shorter than tol
    update   apply the map
    measure  measure the step and move

path core  The iteration in two stops
  iterate/update   The map
    > Halving the error each time.
    run The iterate is {x}.
  iterate=n        The count
";

#[test]
fn documented_variables_and_paths_parse_and_round_trip() {
    let text = DOCUMENTED.replace("@H", "@0123abcd");
    let f = IntentFile::parse(&text).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(f.print(), text);
    let (module, entries) = f.model();
    assert!(module.unwrap().every_var);
    let keys: Vec<&str> = entries[0].gloss.iter().map(|g| g.key.as_str()).collect();
    assert!(keys.contains(&"dx@measure"));
    // the path is a block of its own: the entry above it does not grow
    assert_eq!(entries[0].outline.len(), 2);
    let paths = f.paths();
    assert_eq!(paths.len(), 1);
    assert_eq!((paths[0].id.as_str(), paths[0].title.as_str()), ("core", "The iteration in two stops"));
    let st = &paths[0].stops;
    assert_eq!((st[0].def.as_str(), st[0].step.as_deref(), st[0].anchor.as_deref()), ("iterate", Some("update"), None));
    assert_eq!(st[0].details.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(), [">", "run"]);
    assert_eq!((st[1].def.as_str(), st[1].step.as_deref(), st[1].anchor.as_deref()), ("iterate", None, Some("n")));
    assert_eq!(lens::split_doc_key("temp@evaluate#2"), Some(("temp".into(), Some("evaluate".into()), Some(2))));
    assert_eq!(lens::split_doc_key("s2@#2"), Some(("s2".into(), None, Some(2))));
    assert_eq!(lens::split_doc_key("x@"), None);
    let err = |t: &str| IntentFile::parse(t).err().map(|e| e[0].message.clone()).expect("should not parse");
    assert!(err("module m  every variable\n").contains("documents every variable"));
    assert!(err("f\n  x@#0  no zeroth\n").contains("name@step"));
    assert!(err("path core  t\n  a//b  bad\n").contains("path stop"));
    assert!(err("path core  t\n  a/b  ok\n    note  bad\n").contains("figure"));
}

#[test]
fn a_documented_module_with_a_path_checks_clean() {
    let fs = check(ITER, Some(DOCUMENTED));
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn each_variable_doc_and_path_failure_reports_exactly_one_error() {
    let cases: Vec<(&str, String, &str)> = vec![
        ("a variable with no doc", DOCUMENTED.replace("  xn         the next iterate\n", ""), "undocumented iterate:xn"),
        ("a parameter with no doc", DOCUMENTED.replace("  tol        how short a step ends the iteration\n", ""), "undocumented iterate:tol"),
        ("a later use in a step that does not use it", DOCUMENTED.replace("dx@measure", "dx@guess#2"), "doc-site iterate:dx@guess#2"),
        ("a later use in a step that is not there", DOCUMENTED.replace("dx@measure", "dx@nowhere"), "doc-site iterate:dx@nowhere"),
        ("a doc that gives a symbol and no words", DOCUMENTED.replace("  xn         the next iterate", "  xn         $x_n$"), "doc iterate:xn"),
        ("a doc for a name the def does not use", DOCUMENTED.replace("  n          how many", "  zz         nothing\n  n          how many"), "doc iterate:zz"),
        ("a path stop at a step that is not there", DOCUMENTED.replace("iterate/update   The map", "iterate/apply   The map"), "path path core:iterate/apply"),
        ("a path stop at a def that is not there", DOCUMENTED.replace("iterate=n        The count", "nothing          The count"), "path path core:nothing"),
        ("a path anchor that names no statement", DOCUMENTED.replace("iterate=n", "iterate=m"), "path path core:iterate=m"),
        ("a figure the program does not draw", DOCUMENTED.replace("    run The iterate", "    figure A picture\n    run The iterate"), "figure path core:iterate/update"),
        ("a path placeholder that is not a variable", DOCUMENTED.replace("{x}", "{y}"), "placeholder path core:iterate/update"),
        ("an uncited number in path prose", DOCUMENTED.replace("Halving the error each time.", "Halving the error 40 times."), "number path core:iterate/update"),
    ];
    for (what, intent, want) in cases {
        let fs = check(ITER, Some(&intent));
        assert_eq!(kinds(&fs), [want], "{what}:\n{}", fs.iter().map(|f| f.render()).collect::<String>());
    }
    // without `documents every variable`, a missing doc is not an error
    let fs = check(ITER, Some(&DOCUMENTED.replace("  documents every variable", "").replace("  xn         the next iterate\n", "")));
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn a_doc_starts_to_hold_at_its_first_real_use() {
    use viv::graph::{doc_sites, Cx, Lines};
    let src = "def f(a, n) {
    w = zeros(n)
    t = 0.0
    step one {
        t = a + 1.0
        w[0] = t
    }
    step two {
        t = 2.0 * t
        t = t + a
    }
    return w
}
";
    let prog = viv::parse(src, knotkit::span::SourceId(0)).unwrap();
    let cx = Cx::of(&prog);
    let lines = Lines::new(src);
    let (body, params, start) = viv::graph::body_of(&prog, "f").unwrap();
    let keys: Vec<String> = ["a", "w", "w@one", "t", "t@two", "t@two#2", "t@#1"].iter().map(|s| s.to_string()).collect();
    let got: Vec<(String, usize)> = doc_sites(&body, &params, lines.line(start as u32), &cx, &lines, &keys).into_iter().map(|(k, _, l)| (k, l.unwrap())).collect();
    let want: Vec<(String, usize)> = vec![
        // a parameter: the def's line
        ("a".into(), 1),
        // with later docs, the first use of any kind: the allocation
        ("w".into(), 2),
        ("w@one".into(), 6),
        ("t".into(), 3),
        ("t@two".into(), 9),
        ("t@two#2".into(), 10),
        ("t@#1".into(), 3),
    ];
    assert_eq!(got, want);
    // a variable with one doc skips its bare initialization
    let got = doc_sites(&body, &params, 1, &cx, &lines, &["t".to_string()]);
    assert_eq!(got[0].2, Ok(5));
}

/// B274: a module read on its own knows the signatures of the defs it
/// imports by name, so `f(1.0, b=3.0)` with `from m import f` reads in the
/// lens tools as it runs in `viv run`.
#[test]
fn the_lens_tools_read_keyword_arguments_to_an_imported_def() {
    let dir = scratch_repo("kwimport");
    std::fs::write(dir.join("fx/mm.viv"), "def scaled(x: Num, b: Num = 2.0) -> Num {\n    return x * b\n}\n").unwrap();
    let prog = dir.join("fx/kw.viv");
    std::fs::write(&prog, "from mm import scaled\nfrom mm import scaled as sc\n\nY = scaled(1.0, b=3.0) + sc(2.0, b=0.5)\ntest \"keywords reach the import\" {\n    assert_near(Y, 4.0, 0.0)\n}\n").unwrap();
    let p = prog.to_str().unwrap();
    let (ok, out, err) = viv(&["run", "-q", p]);
    assert!(ok && out.contains("PASS keywords reach the import"), "{out}{err}");
    let (ok, out, err) = viv(&["lens", "check", p]);
    assert!(ok && out.contains("lens check") && !err.contains("unknown keyword"), "{out}{err}");
    let (ok, out, err) = viv(&["explain", p]);
    assert!(ok && out.contains("scaled"), "{out}{err}");
    let (ok, out, err) = viv(&["lens", "coverage", p]);
    assert!(ok && !err.contains("unknown keyword"), "{out}{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
