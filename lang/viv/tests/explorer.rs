// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The explorer's compiler side: the graph
//! of defs, steps and loops, each step's interface, the line-level recording,
//! and the map from the expanded program back to its modules.
//!
//! What is checked here:
//! - a step's interface is what it reads that was set before it, what it
//!   writes that is read after it (including by the next pass of a loop and
//!   by the caller), the globals it reads and the defs it calls;
//! - values handed from step to step become data edges, loop-carried ones
//!   marked;
//! - a line-level recording records the same story in the interpreter and
//!   the compiled C, changes nothing the program prints, records a hot loop
//!   inside a step as one event, records lines only in the first iteration
//!   and the iterations asked for, and stays small however long the loops run;
//! - the plane-wave DFT engine (dft/pwdft.viv) and the Hartree-Fock SCF
//!   (lib/hf/scf_loop.viv) have the graphs their code says they have.
//!
//! The C checks skip if there is no C compiler.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use viv::graph;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A program with its imports expanded from the repository.
fn expand(src: &str) -> String {
    viv::expand_uses(src, &mut |name: &str| std::fs::read_to_string(repo().join(format!("{name}.viv"))).map_err(|e| format!("{name}: {e}"))).expect("expands")
}

fn node<'a>(g: &'a graph::Graph, id: &str) -> &'a graph::GNode {
    g.node(id).unwrap_or_else(|| panic!("no node {id}; nodes: {:?}", g.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>()))
}

fn names(xs: &[String]) -> Vec<&str> {
    xs.iter().map(String::as_str).collect()
}

/// The explorer's run in the interpreter: (output, the story JSON).
fn run_lines(src: &str, iters: &graph::Iters) -> (String, String) {
    let (out, _, _) = graph::explore_run(src, iters).expect("runs");
    let story = viv::story::story_line(&out).expect("a story").to_string();
    (out, story)
}

/// The same run compiled to C; its stdout, or None without a C compiler.
fn native_lines(src: &str, iters: &graph::Iters, tag: &str) -> Option<String> {
    let c = graph::explore_c(src, iters).expect("compiles to C");
    let dir = std::env::temp_dir().join(format!("viv_explorer_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let out = Command::new(&cc).args(["-std=c99", "-O2", "-Wall", "-Wno-unknown-pragmas", "-Wno-unused-parameter", "-Wno-unused-function"]).arg(dir.join("p.c")).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(out.status.success(), "C did not compile:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&run.stdout).to_string())
}

fn own(out: &str) -> String {
    out.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

/// The story's events, parsed.
fn events(story: &str) -> Vec<viv::story::Json> {
    match viv::story::parse_json(story).expect("story JSON") {
        viv::story::Json::Obj(fs) => match fs.into_iter().find(|(k, _)| k == "events").map(|(_, v)| v) {
            Some(viv::story::Json::Arr(xs)) => xs,
            _ => panic!("no events"),
        },
        _ => panic!("not an object"),
    }
}

fn event_where(e: &viv::story::Json) -> (String, String, String) {
    let viv::story::Json::Arr(f) = e else { panic!() };
    let s = |k: usize| match f.get(k) {
        Some(viv::story::Json::Str(s)) => s.clone(),
        _ => String::new(),
    };
    (s(0), s(1), s(2))
}

const SCF: &str = r#"
TOL = 1.0e-8

def half(x) {
    return x / 2.0
}

def fill_with(a: Arr, v) {
    for i to len(a) {
        a[i] = v
    }
}

def run(n) {
    x = 1.0
    work = zeros(3)
    it = 0
    step start "begin at one" {
        y = x
        scratch = y * 2.0
        z = scratch + 1.0
    }
    while it < n {
        step shrink "halve x" {
            y = x
            x = half(y)
        }
        step keep "remember x in the work array" {
            fill_with(work, x)
            it = it + 1
        }
        if abs(x) < TOL {
            break
        }
    }
    return x + z + work[0]
}
print(run(5))
"#;

#[test]
fn a_steps_interface_is_what_it_reads_from_before_and_writes_for_after() {
    let g = graph::graph_of_source(SCF).expect("parses");
    let start = node(&g, "run/start");
    assert_eq!(names(&start.reads), ["x"], "start reads x, set before it");
    // scratch is start's own business: set and used inside it
    // y is read later, but only after shrink sets it again; z is read by the return
    assert_eq!(names(&start.writes), ["z"]);
    let shrink = node(&g, "run/shrink");
    assert_eq!(names(&shrink.reads), ["x"]);
    assert_eq!(names(&shrink.writes), ["x"], "x is read by keep and by the next pass");
    assert_eq!(names(&shrink.calls), ["half"]);
    let keep = node(&g, "run/keep");
    assert_eq!(names(&keep.reads), ["work", "x", "it"]);
    assert_eq!(names(&keep.writes), ["work", "it"], "fill_with writes into its first parameter");
    assert_eq!(names(&keep.calls), ["fill_with"]);
    // the loop is a box of its own, holding both steps
    let lp = node(&g, "run/loop@23");
    assert_eq!(lp.kind, "loop");
    assert_eq!(lp.name, "while it < n");
    assert_eq!(shrink.parent.as_deref(), Some("run/loop@23"));
    assert_eq!(start.parent.as_deref(), Some("run"));
    // a def's writes are the parameters it writes into
    assert_eq!(names(&node(&g, "fill_with").writes), ["a"]);
    assert_eq!(names(&node(&g, "run").globals), ["TOL"]);
}

#[test]
fn values_handed_between_steps_are_data_edges() {
    let g = graph::graph_of_source(SCF).expect("parses");
    let data: Vec<(String, String, Vec<String>, bool)> = g.edges.iter().filter(|e| e.kind == "data").map(|e| (e.from.clone(), e.to.clone(), e.vars.clone(), e.carried)).collect();
    assert!(data.contains(&("run/shrink".into(), "run/keep".into(), vec!["x".into()], false)), "{data:?}");
    // keep's `it` feeds the next pass of keep itself? no: it is read by the
    // loop's condition, which is not a step; and keep reads `it` it wrote
    // itself on the last pass, which is not an edge between two steps
    assert!(!data.iter().any(|(f, t, _, _)| f == t), "{data:?}");
    let calls: Vec<(String, String)> = g.edges.iter().filter(|e| e.kind == "call").map(|e| (e.from.clone(), e.to.clone())).collect();
    assert!(calls.contains(&("run/shrink".into(), "half".into())));
    assert!(calls.contains(&("run/keep".into(), "fill_with".into())));
}

#[test]
fn a_loop_carried_value_is_marked() {
    let src = r#"
def f(n) {
    d = 1.0
    e = 0.0
    for i to n {
        step use "use the density" {
            e = e + d
        }
        step make "make the next density" {
            d = d / 2.0
        }
    }
    return e
}
print(f(3))
"#;
    let g = graph::graph_of_source(src).expect("parses");
    let e = g.edges.iter().find(|e| e.kind == "data" && e.from == "f/make" && e.to == "f/use").expect("make feeds use on the next pass");
    assert!(e.carried);
    assert_eq!(names(&e.vars), ["d"]);
}

#[test]
fn line_recording_matches_in_both_engines_and_changes_no_output() {
    let prog = graph::explore_program(SCF, "");
    let (out, story) = run_lines(&prog, &graph::Iters::default());
    let plain = viv::interp::run(SCF).expect("runs");
    assert_eq!(own(&out), own(&plain), "recording lines changes nothing the program prints");
    // the first iteration line by line, the others only by steps
    let ev = events(&story);
    let lines: Vec<(String, String, String)> = ev.iter().map(event_where).filter(|(k, _, at)| k == "watch" && at.starts_with("line:")).collect();
    assert!(lines.iter().any(|(_, d, at)| d == "run" && at == "line:26"), "the first pass records shrink's lines: {lines:?}");
    let run_lines_n = lines.iter().filter(|(_, d, _)| d == "run").count();
    // 3 lines before the loop and 3 in start, 4 in the first pass, 1 return
    assert_eq!(run_lines_n, 3 + 3 + 4 + 1, "{lines:?}");
    // which branch the `if` took is recorded on every pass
    let branches = ev.iter().map(event_where).filter(|(_, _, at)| at.starts_with("br:")).count();
    assert_eq!(branches, 5);
    // the C records the same story
    let prog_c = prog.clone();
    if let Some(n) = native_lines(&prog_c, &graph::Iters::default(), "scf") {
        assert_eq!(n.lines().filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__")).collect::<Vec<_>>(), out.lines().filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__") && !l.starts_with("__explore__")).collect::<Vec<_>>());
    }
}

#[test]
fn chosen_iterations_are_recorded_line_by_line_too() {
    let prog = graph::explore_program(SCF, "");
    let mut iters = graph::Iters::default();
    iters.iters.insert("run".to_string(), vec![4i64]);
    let (_, story) = run_lines(&prog, &iters);
    let ev = events(&story);
    // find the iteration each line event is in
    let mut it = 0;
    let mut seen: Vec<i64> = Vec::new();
    for e in &ev {
        let (k, d, at) = event_where(e);
        if k == "iter" && d == "run" {
            if let viv::story::Json::Arr(f) = e {
                if let Some(viv::story::Json::Num(n)) = f.get(2) {
                    it = n.parse::<i64>().unwrap();
                }
            }
        }
        if k == "watch" && d == "run" && at == "line:26" {
            seen.push(it);
        }
    }
    assert_eq!(seen, [1, 4]);
    if let Some(n) = native_lines(&prog, &iters, "iters") {
        assert_eq!(viv::story::story_line(&n), Some(story.as_str()));
    }
}

#[test]
fn a_hot_loop_is_one_event_and_a_recording_stays_small() {
    // 200 passes of an outer loop, each with a step holding a 20,000-pass
    // inner loop: four million inner passes, a few thousand events
    let src = r#"
def work(n, m) {
    acc = zeros(8)
    s = 0.0
    for pass to n {
        step fill "fill the accumulator" {
            for k to m {
                acc[mod(k, 8.0)] = acc[mod(k, 8.0)] + 1.0 / (1.0 + k)
            }
            s = s + acc[0]
        }
        step check "look at the sum" {
            if s > 1.0e9 {
                break
            }
        }
    }
    return s
}
print(work(200, 20000))
"#;
    let prog = graph::explore_program(src, "");
    let (out, story) = run_lines(&prog, &graph::Iters::default());
    let ev = events(&story);
    let lines = ev.iter().map(event_where).filter(|(_, _, at)| at.starts_with("line:")).count();
    assert!(lines < 20, "only the first pass is recorded line by line, and the inner loop is one line: {lines}");
    assert!(ev.len() < 200 * 8, "one pass costs seven events (the pass, two steps, one branch): {}", ev.len());
    assert!(story.len() < 200_000, "the story stays small: {} bytes", story.len());
    // the inner loop's event carries what it assigned
    let table = graph::lines_table(&prog);
    let hot = table.iter().find(|r| r.kind == "loop").expect("the inner loop is in the table");
    assert_eq!(hot.watch, ["acc"]);
    assert_eq!(hot.step.as_deref(), Some("fill"));
    assert!(hot.gated);
    if let Some(n) = native_lines(&prog, &graph::Iters::default(), "hot") {
        assert_eq!(viv::story::story_line(&n), viv::story::story_line(&out));
    }
}

#[test]
fn the_story_lens_ignores_line_events() {
    let prog = graph::explore_program(SCF, "");
    let (out, _) = run_lines(&prog, &graph::Iters::default());
    let (plain, _, _) = viv::interp::run_checked_holds(&prog, false).expect("runs");
    let a = viv::story::parse(viv::story::story_line(&out).unwrap()).unwrap();
    let b = viv::story::parse(viv::story::story_line(&plain).unwrap()).unwrap();
    assert_eq!(a.root, b.root, "the story's tree is the same with and without the lines");
}

#[test]
fn expanded_lines_map_back_to_their_modules() {
    let root = std::fs::read_to_string(repo().join("scf/story_water.viv")).unwrap();
    let mut texts: HashMap<String, String> = HashMap::new();
    let expanded = viv::expand_uses(&root, &mut |name: &str| {
        let t = std::fs::read_to_string(repo().join(format!("{name}.viv"))).map_err(|e| e.to_string())?;
        texts.insert(name.to_string(), t.clone());
        Ok(t)
    })
    .unwrap();
    let (names, map) = graph::line_map(&root, "scf/story_water", &expanded, &texts);
    assert!(names.len() > 10, "{names:?}");
    let mut checked = 0;
    for (x, line) in expanded.lines().enumerate() {
        match map[x + 1] {
            Some((m, n)) => {
                let text = if m == 0 { &root } else { &texts[&names[m]] };
                assert_eq!(text.lines().nth(n - 1), Some(line), "expanded line {} should be {}:{n}", x + 1, names[m]);
                checked += 1;
            }
            None => assert!(line.starts_with("# ---- "), "only marker lines are unmapped: {line}"),
        }
    }
    assert!(checked > 5000);
}

#[test]
fn the_hartree_fock_loop_has_the_graph_its_code_says() {
    let src = expand(&std::fs::read_to_string(repo().join("scf/story_water.viv")).unwrap());
    let g = graph::graph_of_source(&src).expect("parses");
    let jk = node(&g, "scf_iterate/coulomb_exchange");
    // the incremental build from the change in density, or a full build (lib/hf/scf_loop.viv)
    assert_eq!(names(&jk.calls), ["jk_update", "jk_of"]);
    assert!(jk.reads.contains(&"D".to_string()));
    // JK, and `since`: iterations since the last full build
    assert_eq!(names(&jk.writes), ["JK", "since"]);
    let density = node(&g, "scf_iterate/new_density");
    assert_eq!(names(&density.writes), ["D"], "the new density is the only value the density step hands on");
    let e = g.edges.iter().find(|e| e.kind == "data" && e.from == "scf_iterate/new_density" && e.to == "scf_iterate/coulomb_exchange").expect("the density feeds the next J and K");
    assert!(e.carried);
    assert!(g.nodes.iter().any(|n| n.kind == "loop" && n.def == "scf_iterate" && n.name == "for it from 1 to setup.options.max_iter + 1"));
}

#[test]
fn the_dense_dft_engine_has_the_graph_its_code_says() {
    // the engine after silicon's deck and the shared modules, as dft/run.sh
    // and dft/lens.sh put it together
    let engine = std::fs::read_to_string(repo().join("dft/pwdft.viv")).unwrap();
    let src = format!(
        "from dft.si import ATSPEC, ECUT, GTH_NL, HPRJ, MAXIT, NAT, NBANDS, NPRJ, NSPEC, RPRJ, SCF_TOL, SMEAR\n\
         from dft.kmesh import KFRAC, KWT, NKPT\n\
         from dft.pw_common import (\n    anderson_step, B, CH0, CHI, CHL, CHS, DV, fermi_level, fill_occupations, G2, gth_proj,\n    \
         HA_EV, lda_xc, legendre, MM, NCH, NDIM, NELEC, NG, NGRID, NM, NPTS, NPWMAX, OMEGA, PATHK,\n    PI, TAU, VPS, wrap, ZERO3,\n)\n{engine}"
    );
    let program = graph::explore_program(&expand(&src), "");
    let g = graph::graph_of_source(&program).expect("parses");
    // the SCF loop is top-level code: steps of <main>, inside its while loop
    let diag = node(&g, "<main>/diagonalize");
    assert!(diag.parent.as_deref().is_some_and(|p| p.starts_with("<main>/loop@")), "{:?}", diag.parent);
    assert!(diag.calls.contains(&"build_h".to_string()), "{:?}", diag.calls);
    let xc = node(&g, "<main>/xc");
    let pot = node(&g, "<main>/potential");
    let handed: Vec<&String> = xc.writes.iter().filter(|v| pot.reads.contains(v)).collect();
    assert!(!handed.is_empty(), "xc hands the potential step its output: xc writes {:?}, potential reads {:?}", xc.writes, pot.reads);
    assert!(g.edges.iter().any(|e| e.kind == "data" && e.from == "<main>/xc" && e.to == "<main>/potential"));
    // the mixed density goes round to the next pass's xc
    assert!(g.edges.iter().any(|e| e.kind == "data" && e.to == "<main>/xc" && e.carried), "{:?}", g.edges.iter().filter(|e| e.to == "<main>/xc").collect::<Vec<_>>());
    let keep = node(&g, "setup_slot/keep");
    assert!(keep.parent.as_deref().is_some_and(|p| p.starts_with("setup_slot/loop@")));
    // the line table: the SCF loop's steps are recorded line by line in the
    // first pass only, and the plane-wave loops inside build_h are one line each
    let table = graph::lines_table(&program);
    assert!(table.iter().any(|r| r.def == "<main>" && r.kind == "iterloop"));
    assert!(table.iter().filter(|r| r.def == "<main>" && r.step.as_deref() == Some("diagonalize")).all(|r| r.gated));
}

#[test]
fn a_nested_loop_with_steps_is_recorded_line_by_line_in_its_first_pass_only() {
    let src = r#"
def grid(n) {
    total = 0.0
    for i to n {
        for j to n {
            step cell "one cell of the grid" {
                v = i * n + j
                total = total + v
            }
        }
    }
    return total
}
print(grid(6))
"#;
    let prog = graph::explore_program(src, "");
    let (out, story) = run_lines(&prog, &graph::Iters::default());
    let ev = events(&story);
    let cell_lines = ev.iter().map(event_where).filter(|(k, d, at)| k == "watch" && d == "grid" && (at == "line:7" || at == "line:8")).count();
    assert_eq!(cell_lines, 2, "the cell's two lines, once: the first pass of i, the first pass of j");
    if let Some(n) = native_lines(&prog, &graph::Iters::default(), "nested") {
        assert_eq!(viv::story::story_line(&n), viv::story::story_line(&out));
    }
}

#[test]
fn every_pass_of_a_nested_loop_is_recorded_when_asked() {
    // the explorer asks for this when a run is small (every trial step of
    // lmder, so the Read view's pass selector has values for each)
    let src = r#"
def grid(n) {
    total = 0.0
    for i to n {
        for j to n {
            step cell "one cell of the grid" {
                v = i * n + j
                total = total + v
            }
        }
    }
    return total
}
print(grid(3))
"#;
    let prog = graph::explore_program(src, "");
    let mut req = graph::Iters::default();
    // the outer loop is the def's iteration loop: its passes are asked for by
    // number; the inner one is recorded in every pass of each
    req.iters.insert("grid".to_string(), vec![2, 3]);
    req.every.insert("grid".to_string());
    let (out, story) = run_lines(&prog, &req);
    let ev = events(&story);
    let cell_lines = ev.iter().map(event_where).filter(|(k, d, at)| k == "watch" && d == "grid" && (at == "line:7" || at == "line:8")).count();
    assert_eq!(cell_lines, 2 * 3 * 3, "the cell's two lines in every pass of i and of j");
    let (plain, _) = run_lines(&prog, &graph::Iters::default());
    assert_eq!(own(&out), own(&plain), "recording more changes nothing the program prints");
    if let Some(n) = native_lines(&prog, &req, "every") {
        assert_eq!(viv::story::story_line(&n), viv::story::story_line(&out));
    }
}

#[test]
fn the_intent_file_explains_single_statements_by_anchor() {
    use viv::lens::IntentFile;
    let text = "\
norm  @00000000
  what  The length of a vector.

  = s      the sum of squares so far
  >        a second paragraph about it
  = sqrt() the square root, once
  scale    scale the vector
    why    so the caller gets a unit vector
    = v#2  the second write into v
";
    let f = IntentFile::parse(text).expect("parses");
    assert_eq!(f.print(), text, "round trip");
    let (_, es) = f.model();
    let e = &es[0];
    assert_eq!(e.details.len(), 2, "two anchors outside every step");
    assert_eq!(e.details[0].anchor, "s");
    assert_eq!(e.details[0].text, "the sum of squares so far\na second paragraph about it");
    assert_eq!(e.details[1].target(), ("sqrt".to_string(), true, 1));
    let step = &e.outline[0];
    assert!(step.children.is_empty(), "why and = lines are not child steps");
    assert_eq!(step.details.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(), ["why", "="]);
    assert_eq!(step.details[1].target(), ("v".to_string(), false, 2));
    // a bad anchor is a parse error naming the line
    let bad = IntentFile::parse("f\n  what  x\n\n  = 3x  no\n").unwrap_err();
    assert_eq!(bad[0].line, 4);
}

#[test]
fn an_anchor_resolves_to_its_statement_and_lens_check_reports_one_that_does_not() {
    let src = "def norm(v: Arr) {\n    s = 0.0\n    for i to len(v) {\n        s = s + v[i] * v[i]\n    }\n    step scale \"scale the vector\" {\n        r = sqrt(s)\n        for i to len(v) {\n            v[i] = v[i] / r\n        }\n    }\n    return r\n}\nprint(norm([3.0, 4.0]))\n";
    let intent = "norm\n  what  The length of a vector, and the vector scaled to length one.\n\n  = s#2  add one square\n  scale  scale the vector\n    = r  the length\n    = q  nothing assigns q\n";
    let prog = viv::parse(src, knotkit::span::SourceId(0)).unwrap();
    let def = prog.items.iter().find_map(|i| if let viv::ast::Item::Def(d) = i { Some(d) } else { None }).unwrap();
    let cands = graph::anchor_candidates(def, &graph::Cx::of(&prog), &graph::Lines::new(src));
    let (_, es) = viv::lens::IntentFile::parse(intent).unwrap().model();
    let a = graph::entry_anchors(&es[0].details, &es[0].outline);
    assert_eq!(graph::resolve_anchor(&cands, None, &a[0].1), Some(4), "the second statement assigning s, inside the loop");
    assert_eq!(graph::resolve_anchor(&cands, Some("scale"), &a[1].1), Some(7));
    assert_eq!(graph::resolve_anchor(&cands, Some("scale"), &a[2].1), None);
    let input = viv::lens::Input {
        root: viv::lens::ModuleText { name: "fx/norm".into(), path: "fx/norm.viv".into(), src: src.into(), intent_path: "fx/norm.intent".into(), intent: Some(intent.into()) },
        used: vec![],
        expanded: src.into(),
        companion_tests: vec![],
    };
    let l = viv::lens::load(&input).unwrap();
    let fs = viv::lens::check(&l, &mut |_, _| None);
    let anchors: Vec<&viv::lens::Finding> = fs.iter().filter(|f| f.kind == "anchor").collect();
    assert_eq!(anchors.len(), 1, "{fs:?}");
    assert!(anchors[0].message.contains("`= q`"), "{}", anchors[0].message);
    assert_eq!(anchors[0].line, 7);
}

const HOT: &str = r#"
def norm2(v: Arr) {
    s = 0.0
    for i to len(v) {
        s = s + v[i] * v[i]
    }
    return s
}

def run(n) {
    acc = 0.0
    for k to n {
        step add "add one norm" {
            acc = acc + norm2([k, 1.0])
        }
    }
    return acc
}
print(run(50))
"#;

#[test]
fn a_def_without_steps_is_recorded_line_by_line_in_its_first_call_without_changing_the_story() {
    let prog = graph::explore_program(HOT, "");
    let mut req = graph::Iters::default();
    req.defs.insert("norm2".to_string());
    let (out, story) = run_lines(&prog, &req);
    let (plain_out, plain_story) = run_lines(&prog, &graph::Iters::default());
    assert_eq!(own(&out), own(&plain_out));
    let ev = events(&story);
    let norm_lines = ev.iter().map(event_where).filter(|(k, d, at)| k == "watch" && d == "norm2" && at.starts_with("line:")).count();
    assert_eq!(norm_lines, 3, "s = 0, the loop as one line, the return: once, in the first of 50 calls");
    assert!(!ev.iter().map(event_where).any(|(k, d, _)| k == "call" && d == "norm2"), "no call events: the story records what it did before");
    // without the line events the two stories are the same
    let strip = |s: &str| events(s).into_iter().filter(|e| !event_where(e).2.starts_with("line:") || event_where(e).1 != "norm2").map(|e| e.text()).collect::<Vec<_>>();
    assert_eq!(strip(&story), strip(&plain_story));
    if let Some(n) = native_lines(&prog, &req, "anchored") {
        assert_eq!(viv::story::story_line(&n), viv::story::story_line(&out));
    }
}

#[test]
fn a_def_with_steps_is_recorded_line_by_line_in_its_first_call_and_the_calls_asked_for() {
    let src = "def f(x) {\n    step twice \"double it\" {\n        y = 2.0 * x\n    }\n    return y\n}\nt = 0.0\nfor k to 5 {\n    t = t + f(k)\n}\nprint(t)\n";
    let prog = graph::explore_program(src, "");
    let count = |req: &graph::Iters| -> usize {
        let (_, story) = run_lines(&prog, req);
        events(&story).iter().map(event_where).filter(|(k, d, at)| k == "watch" && d == "f" && at == "line:3").count()
    };
    assert_eq!(count(&graph::Iters::default()), 1);
    let mut req = graph::Iters::default();
    req.calls.insert("f".to_string(), vec![4]);
    assert_eq!(count(&req), 2);
    if let Some(n) = native_lines(&prog, &req, "calls") {
        let (out, _) = run_lines(&prog, &req);
        assert_eq!(viv::story::story_line(&n), viv::story::story_line(&out));
    }
}

#[test]
fn a_def_that_takes_a_function_has_a_body_hash_of_its_own() {
    let src = "def twice(f, x) {\n    return f(f(x))\n}\ndef inc(x) {\n    return x + 1.0\n}\ndef dbl(x) {\n    return 2.0 * x\n}\nprint(twice(inc, 1.0))\nprint(twice(dbl, 1.0))\n";
    let hs = viv::def_hashes(src).unwrap();
    let h = |s: &str| hs.iter().find(|h| h.name == "twice").map(|h| h.body_short.clone()).unwrap_or_else(|| panic!("no hash for twice in {s}"));
    let a = h(src);
    // the same def, called with one function fewer: the hash does not depend on its callers
    let fewer = src.replace("print(twice(dbl, 1.0))\n", "");
    assert_eq!(viv::def_hashes(&fewer).unwrap().iter().find(|x| x.name == "twice").map(|x| x.body_short.clone()), Some(a.clone()));
    // an edit to its body changes it; a comment does not
    let edited = src.replace("return f(f(x))", "return f(f(f(x)))");
    assert_ne!(viv::def_hashes(&edited).unwrap().iter().find(|x| x.name == "twice").map(|x| x.body_short.clone()), Some(a.clone()));
    let commented = src.replace("    return f(f(x))\n", "    # apply it twice\n    return f(f(x))\n");
    assert_eq!(viv::def_hashes(&commented).unwrap().iter().find(|x| x.name == "twice").map(|x| x.body_short.clone()), Some(a));
}

#[test]
fn the_bundle_carries_each_lines_math_with_the_glossarys_symbols() {
    use viv::lens::{Input, ModuleText};
    let src = "def shrink(delta, pnorm) {\n    step cut {\n        if pnorm > delta {\n            delta = 0.5 * min(delta, pnorm / 0.1)\n        }\n    }\n    return delta\n}\nd = shrink(2.0, 3.0)\nprint(d)\n";
    let intent = "shrink\n  what   Shrink the radius.\n  delta  $\\Delta$  the trust radius\n  pnorm  $\\lVert D p \\rVert$  the step's length\n\n  cut  cut the radius\n    $$ \\Delta \\le \\tfrac12 \\min(\\Delta, 10 \\lVert D p \\rVert) $$\n";
    let input = Input {
        root: ModuleText { name: "fx/shrink".into(), path: "fx/shrink.viv".into(), src: src.into(), intent_path: "fx/shrink.intent".into(), intent: Some(intent.into()) },
        used: vec![],
        expanded: src.into(),
        companion_tests: vec![],
    };
    let b = graph::bundle_json(&input, "").expect("a bundle");
    let j = viv::story::parse_json(&b).expect("json");
    let viv::story::Json::Obj(top) = &j else { panic!("not an object") };
    let get = |k: &str| top.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).unwrap();
    let viv::story::Json::Arr(rows) = get("tex") else { panic!("no tex table") };
    let texts: Vec<String> = rows
        .iter()
        .filter_map(|r| if let viv::story::Json::Obj(f) = r { f.iter().find(|(n, _)| n == "tex").and_then(|(_, v)| if let viv::story::Json::Str(s) = v { Some(s.clone()) } else { None }) } else { None })
        .collect();
    assert!(texts.contains(&"\\lVert D p \\rVert > \\Delta".to_string()), "{texts:?}");
    assert!(texts.contains(&"\\Delta \\gets 0.5\\,\\min\\left(\\Delta, \\frac{\\lVert D p \\rVert}{0.1}\\right)".to_string()), "{texts:?}");
    assert!(texts.iter().any(|t| t.starts_with("d = \\operatorname{shrink}")), "the top level has its math too: {texts:?}");
    // the symbols travel with the intent, and the glossary text loses its symbol
    assert!(b.contains("\"symbols\":{\"delta\":\"\\\\Delta\",\"pnorm\":\"\\\\lVert D p \\\\rVert\"}"), "{b}");
    assert!(b.contains("[\"delta\",\"the trust radius\"]"));
    // the equation is a detail of its step
    assert!(b.contains("\"label\":\"$$\""));
}

/// A gradient (`value_and_grad`) of a def with steps that calls another def
/// with steps, inside a loop, with a debug lens on the inner def.
const GRADIENT: &str = "def inner(x: Arr) {\n    step square {\n        y = x * x\n    }\n    return y\n}\ndef outer(x: Arr) {\n    t = 0.0\n    for k to 2 {\n        step first {\n            a = inner(x)\n        }\n        step total {\n            t = t + sum(a)\n        }\n    }\n    return t\n}\nloss, g = value_and_grad(outer, [1.0, 2.0, 3.0])\nprint(loss)\nshow g\n";
const GRADIENT_LENS: &str = "debug inner {\n    watch at square y\n    hold at square min(y) >= 0.0 \"a square is not negative\"\n}\n";

#[test]
fn a_gradients_graph_is_its_forward_defs() {
    let prog = graph::explore_program(GRADIENT, "");
    let g = graph::graph_of_source(&prog).expect("graph");
    let gd = node(&g, "viv__ad_grad_outer");
    // one box that calls the def it differentiates, and has no steps of its own
    assert_eq!(gd.name, "gradient of outer");
    assert!(!gd.stepped);
    // in the names of the def it differentiates, with nothing the pass made
    assert_eq!(names(&gd.params), ["x"]);
    assert!(!gd.globals.iter().chain(&gd.writes).any(|v| v.starts_with("viv__")), "{:?} {:?}", gd.globals, gd.writes);
    assert!(!g.nodes.iter().any(|n| n.kind == "step" && n.def == "viv__ad_grad_outer"), "the forward pass's steps belong to outer and inner");
    assert!(g.edges.iter().any(|e| e.kind == "call" && e.from == "viv__ad_grad_outer" && e.to == "outer"));
    // outer and inner keep their own steps and calls
    assert_eq!(names(&node(&g, "outer/first").calls), ["inner"]);
    assert!(node(&g, "inner/square").writes.contains(&"y".to_string()));
    let json = graph::nodes_edges_json(&g);
    assert!(json.contains("\"label\":\"gradient of outer\""), "{json}");
}

#[test]
fn a_gradients_recording_names_its_forward_defs_in_both_engines() {
    let prog = graph::explore_program(GRADIENT, GRADIENT_LENS);
    let (out, story) = run_lines(&prog, &graph::Iters::default());
    assert_eq!(own(&out), "28\ng: [4, 8, 12]\n");
    let ev: Vec<(String, String, String)> = events(&story).iter().map(event_where).collect();
    let has = |k: &str, d: &str, at: &str| ev.iter().any(|(a, b, c)| a == k && b == d && c == at);
    // the gradient is a call, and so is each def inlined into it
    assert!(has("call", "viv__ad_grad_outer", "") && has("ret", "viv__ad_grad_outer", ""));
    assert_eq!(ev.iter().filter(|(k, d, _)| k == "call" && d == "inner").count(), 2);
    assert_eq!(ev.iter().filter(|(k, d, _)| k == "ret" && d == "inner").count(), 2);
    assert!(has("iter", "outer", ""), "outer's loop is its iteration loop");
    // each step under the def it was written in
    assert!(has("enter", "inner", "square") && has("enter", "outer", "first"));
    // the lens on inner, and inner's lines, under the names its source gives them
    assert!(has("watch", "inner", "square") && has("hold", "inner", "square"));
    assert!(has("watch", "inner", "line:3") && has("watch", "outer", "line:11"));
    assert!(story.contains("[\"watch\",\"inner\",\"line:3\",{\"y\":"), "{story}");
    assert!(story.contains("[\"watch\",\"outer\",\"line:11\",{\"a\":"), "{story}");
    // nothing the gradient added shows through
    let hidden: Vec<&(String, String, String)> = ev.iter().filter(|(k, d, _)| d.starts_with("viv__ad") && k != "call" && k != "ret").collect();
    assert!(hidden.is_empty(), "{hidden:?}");
    assert!(story.match_indices("\"viv__ad").all(|(i, _)| story[i..].starts_with("\"viv__ad_grad_outer\"]")), "a renamed variable in the story:\n{story}");
    // a hold that failed would be counted
    assert!(story.contains("true,\"a square is not negative\""));
    if let Some(n) = native_lines(&prog, &graph::Iters::default(), "gradient") {
        assert_eq!(viv::story::story_line(&n), viv::story::story_line(&out), "C records a different story");
    }
}
