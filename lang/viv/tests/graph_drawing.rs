// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The whole program as a graph, drawn by Viv:
//! the graph as data (`graph_of_self`, `program_graph`, src/graphlit.rs), the
//! layout and drawing libraries (lib/graphlayout.viv, lib/graphdraw.viv) and
//! the diagram marks they print (`box`, `arrow`, `note`), drawn by `viv plot`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(name: &str) -> Result<String, String> {
    std::fs::read_to_string(repo().join(format!("{name}.viv"))).map_err(|e| format!("{name}: {e}"))
}

/// The resolver is process-wide; the tests that install one take turns.
static TURN: Mutex<()> = Mutex::new(());

fn resolver() -> viv::graphlit::Resolver {
    Arc::new(read)
}

fn run(src: &str) -> String {
    let text = viv::expand_uses(src, &mut |n: &str| read(n)).expect("expands");
    let out = viv::interp::run(&text).unwrap_or_else(|e| panic!("{e}"));
    out.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

#[test]
fn the_libraries_pass_their_tests() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    viv::set_module_resolver(Some(resolver()));
    for m in ["lib/proggraph", "lib/graphlayout", "lib/graphdraw"] {
        let out = run(&read(m).unwrap());
        let pass = out.lines().filter(|l| l.starts_with("PASS ")).count();
        assert!(pass > 0 && !out.contains("FAIL"), "{m}:\n{out}");
    }
}

#[test]
fn graph_of_self_is_the_graph_as_data() {
    let src = "from proggraph import node_index\n\
        def f(x) {\n    step twice \"double it\" {\n        y = 2 * x\n    }\n    step more {\n        z = y + 1\n    }\n    return z\n}\n\
        g = graph_of_self()\n\
        print(join(\" \", g.ids))\n\
        print(join(\"|\", g.kinds))\n\
        print(g.texts[node_index(g, \"f/twice\")])\n\
        k = node_index(g, \"f/more\")\n\
        print(g.reads[k], \"<-\", g.labels[g.parents[k]])\n\
        for e to len(g.edge_kinds) {\n    print(g.edge_kinds[e], g.ids[g.edge_from[e]], g.ids[g.edge_to[e]], g.edge_vars[e])\n}\n";
    let out = run(src);
    let want = "ProgramGraph node_index node_count edge_count reachable f f/twice f/more <main>\n\
        record|def|def|def|def|def|step|step|def\n\
        double it\n\
        y <- f\n\
        call reachable node_index \n\
        data f/twice f/more y\n\
        call <main> node_index \n";
    assert_eq!(out, want);
}

#[test]
fn program_graph_needs_a_resolver_and_the_record() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    viv::set_module_resolver(None);
    let src = format!("{}\ng = program_graph(\"scf/story_water\")\n", read("lib/proggraph").unwrap());
    let err = viv::interp::run(&src).unwrap_err();
    assert!(err.contains("nothing here can read modules"), "{err}");
    let err = viv::interp::run("g = graph_of_self()\n").unwrap_err();
    assert!(err.contains("does not declare one"), "{err}");
    let err = viv::interp::run("record ProgramGraph { name: Str }\ng = graph_of_self()\n").unwrap_err();
    assert!(err.contains("not declared as"), "{err}");
    // a program's own def of that name is its own
    let out = viv::interp::run("def graph_of_self() {\n    return 4\n}\nprint(graph_of_self())\n").unwrap();
    assert!(out.starts_with("4\n"), "{out}");
}

#[test]
fn diagram_styles_are_checked() {
    for (src, word) in [
        ("figure(\"t\", \"\", \"\")\nbox(\"a\", \"s9\", 0, 0, 1, 1)\n", "s9"),
        ("figure(\"t\", \"\", \"\")\narrow(\"a\", \"dash dot\", [0.0, 1.0], [0.0, 1.0])\n", "twice"),
        ("figure(\"t\", \"\", \"\")\nnote(\"a\", \"x\", \"huge\", 0, 0)\n", "huge"),
    ] {
        let e = viv::interp::run(src).unwrap_err();
        assert!(e.contains(word), "{e}");
        let c = viv::compile_c(src).unwrap_err();
        assert!(c.contains(word), "{c}");
    }
}

/// The drawing of a program: its records, and the boxes and arrows in them.
struct Drawing {
    boxes: Vec<(String, [f64; 4])>,
    arrows: Vec<(String, Vec<f64>, Vec<f64>)>,
    records: String,
}

fn draw(module: &str) -> Drawing {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let records = viv::graphlit::draw_records(module, resolver()).unwrap_or_else(|e| panic!("{module}: {e}"));
    let nums = |s: &str| s.split_whitespace().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>();
    let arr = |s: &str| s.trim().trim_start_matches('[').trim_end_matches(']').split(", ").map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>();
    let mut d = Drawing { boxes: Vec::new(), arrows: Vec::new(), records: records.clone() };
    for l in records.lines() {
        let f: Vec<&str> = l.splitn(2, ": ").nth(1).unwrap_or("").split(" ; ").collect();
        if l.starts_with("box: ") {
            let v = nums(f[2]);
            d.boxes.push((f[0].to_string(), [v[0], v[1], v[2], v[3]]));
        } else if l.starts_with("arrow: ") {
            d.arrows.push((f[0].to_string(), arr(f[2]), arr(f[3])));
        }
    }
    d
}

/// The nodes `lib/graphdraw` draws: every def with steps the run reaches,
/// the defs on a call path to one, `<main>`, and their steps and loops.
fn expected_nodes(module: &str) -> usize {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let src = format!("from graphdraw import chosen_defs\ng = program_graph(\"{module}\")\nc = chosen_defs(g)\nn = 0\nfor k to len(g.ids) {{\n    if g.kinds[k] != \"record\" and c[g.owners[k]] == 1 {{\n        n += 1\n    }}\n}}\nprint(n)\n");
    viv::set_module_resolver(Some(resolver()));
    let out = run(&src);
    out.lines().next().unwrap().trim().parse::<f64>().unwrap() as usize
}

fn overlaps(a: &[f64; 4], b: &[f64; 4]) -> bool {
    let tol = 0.01;
    a[0] < b[2] - tol && b[0] < a[2] - tol && a[1] < b[3] - tol && b[1] < a[3] - tol
}

fn inside(a: &[f64; 4], b: &[f64; 4]) -> bool {
    let tol = 0.01;
    a[0] >= b[0] - tol && a[2] <= b[2] + tol && a[1] >= b[1] - tol && a[3] <= b[3] + tol
}

fn check(module: &str) {
    let d = draw(module);
    let n = expected_nodes(module);
    assert_eq!(d.boxes.len(), n, "{module}: one box per drawn node");
    // two boxes either do not meet or one holds the other, and a node's box
    // is inside its parent's
    for (i, (ia, a)) in d.boxes.iter().enumerate() {
        for (ib, b) in d.boxes.iter().skip(i + 1) {
            if overlaps(a, b) {
                assert!(inside(a, b) || inside(b, a), "{module}: {ia} and {ib} overlap");
            }
        }
        if let Some((parent, _)) = ia.rsplit_once('/') {
            if let Some((_, p)) = d.boxes.iter().find(|(id, _)| id == parent) {
                assert!(inside(a, p), "{module}: {ia} is outside {parent}");
            }
        }
    }
    // every arrow runs from its source's box to its target's
    let near = |x: f64, y: f64, id: &str| {
        let (_, b) = d.boxes.iter().find(|(i, _)| i == id).unwrap_or_else(|| panic!("{module}: no box {id}"));
        let tol = 0.5;
        x >= b[0] - tol && x <= b[2] + tol && y >= b[1] - tol && y <= b[3] + tol
    };
    let mut calls = 0;
    for (id, xs, ys) in &d.arrows {
        let Some((kind, ends)) = id.split_once(':') else { continue };
        if kind == "legend" {
            continue;
        }
        let (from, to) = ends.split_once("->").unwrap();
        assert!(xs.len() >= 2 && xs.len() == ys.len(), "{module}: {id}");
        assert!(near(xs[0], ys[0], from), "{module}: {id} does not start on {from}");
        assert!(near(xs[xs.len() - 1], ys[ys.len() - 1], to), "{module}: {id} does not end on {to}");
        calls += (kind == "call") as usize;
    }
    assert!(calls > 0, "{module}: no calls drawn");
    // the same program draws the same bytes
    assert_eq!(draw(module).records, d.records, "{module}: the drawing changed between two runs");
    // and viv plot makes it an SVG with a rect per box, each with its node's id
    let svg = viv::figsvg::render_page(&d.records, 640.0);
    assert_eq!(svg.matches("<rect data-id=").count(), n, "{module}");
    assert!(svg.contains(r#"data-id="<main>""#) || svg.contains(r#"data-id="&lt;main&gt;""#));
}

#[test]
fn the_water_scf_is_drawn() {
    check("scf/story_water");
    let d = draw("scf/story_water");
    for id in ["scf_with", "integrals", "scf_iterate", "scf_iterate/fock"] {
        assert!(d.boxes.iter().any(|(b, _)| b == id), "no box {id}");
    }
    // the density is carried to the next pass of the SCF loop
    assert!(d.arrows.iter().any(|(id, _, _)| id.starts_with("next:scf_iterate/")), "no carried value");
}

#[test]
fn the_minpack_programs_are_drawn() {
    check("lib/minpack/story_rosenbrock");
    check("posts/2026-09-25-fitting-electron-mobility-with-minpack");
    check("dft/explore_si");
    check("pde/heat_1d");
}

/// A multi-path: closed, one box; open, a fork,
/// a path per member and a join, with the default's path solid, the member
/// that is a drawn def on its own box, and nothing overlapping.
#[test]
fn a_multipath_is_a_fork_its_paths_and_a_join() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    viv::set_module_resolver(Some(resolver()));
    let prog = |open: u32| {
        format!(
            "from graphdraw import default_view, draw_view_mp, MultiPaths\n\
             from proggraph import node_index\n\
             def leaf_a(x) {{\n    return x\n}}\n\
             def leaf_b(x) {{\n    return x + 0\n}}\n\
             def leaf_c(x) {{\n    return x * 1\n}}\n\
             def pick_ms() {{\n    return [leaf_a, leaf_b, leaf_c]\n}}\n\
             def host(x) {{\n    step compute \"compute it\" {{\n        y = apply(pick_ms()[0], x)\n    }}\n    return y\n}}\n\
             print(host(1.0))\n\
             g = graph_of_self()\n\
             v = default_view(g)\n\
             mp = MultiPaths {{ ids: [\"pick\"], titles: [\"compute the thing\"], contracts: [\"within 1e-12\"], notes: [\"3 independent, 2 seen agreeing\"], hosts: [node_index(g, \"host\")], host_steps: [node_index(g, \"host/compute\")], members: [\"leaf_a leaf_b leaf_c\"], defaults: [\"leaf_a\"], ran: [\"leaf_a\"], open: [{open}] }}\n\
             draw_view_mp(g, v, mp, \"t\")\n"
        )
    };
    let closed = run(&prog(0));
    assert!(closed.contains("box: mp:pick@host ; s3 tint ;"), "{closed}");
    assert!(closed.contains("multi-path ×3"), "{closed}");
    assert!(closed.contains("agree within 1e-12"), "{closed}");
    assert!(closed.contains("arrow: mp:pick@host/in ; ink ;"), "{closed}");
    assert!(!closed.contains("/fork"), "{closed}");
    let open = run(&prog(1));
    for want in [
        "box: mp:pick@host/fork ; s3 tint diamond ;",
        "box: mp:pick@host/join ; s3 tint diamond ;",
        "box: mp:pick@host/m:leaf_a ; s1 tint ;",
        "box: mp:pick@host/m:leaf_b ; muted plain dash ;",
        "arrow: mp:pick@host/to:leaf_a ; ink thick ;",
        "arrow: mp:pick@host/from:leaf_c ; muted dash ;",
        "note: mp:pick@host/in#count ; compute the thing ;",
        "arrow: legend:multipath ; s3 nohead ;",
    ] {
        assert!(open.contains(want), "missing `{want}` in\n{open}");
    }
    // the boxes do not overlap, and the fork comes before its members, the
    // members before the join (left to right)
    let boxes: Vec<(String, [f64; 4])> = open
        .lines()
        .filter_map(|l| l.strip_prefix("box: "))
        .map(|l| {
            let f: Vec<&str> = l.split(" ; ").collect();
            let v: Vec<f64> = f[2].split_whitespace().map(|x| x.parse().unwrap()).collect();
            (f[0].to_string(), [v[0].min(v[2]), v[1].min(v[3]), v[0].max(v[2]), v[1].max(v[3])])
        })
        .filter(|(id, _)| !id.contains('/') || id.starts_with("mp:"))
        .collect();
    let top: Vec<&(String, [f64; 4])> = boxes.iter().filter(|(id, _)| !id.starts_with("legend") && (id.starts_with("mp:") || !id.contains('/'))).collect();
    for (i, a) in top.iter().enumerate() {
        for b in &top[i + 1..] {
            assert!(!overlaps(&a.1, &b.1), "{} overlaps {}", a.0, b.0);
        }
    }
    // along the way the defs run (`across`: left to right; `down`: top to bottom)
    let down = open.lines().any(|l| l.starts_with("figure: ") && l.contains(" ; down ;"));
    let at = |id: &str| boxes.iter().find(|b| b.0 == id).map(|b| if down { -b.1[3] } else { b.1[0] }).unwrap_or_else(|| panic!("{id}"));
    assert!(at("mp:pick@host/fork") < at("mp:pick@host/m:leaf_b") && at("mp:pick@host/m:leaf_b") < at("mp:pick@host/join"));
    // both renderers draw the diamond
    let svg = viv::figsvg::render_page(&open, 900.0);
    assert!(svg.contains("data-id=\"mp:pick@host/fork\" d=\"M"), "no diamond path in the SVG");
}
