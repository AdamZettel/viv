// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The architecture lens (src/arch.rs): the
//! `.arch` file, the layer rules, the budgets, the multi-paths and their
//! independence, the ratchet, the JSON, and the multi-paths the explorer's
//! bundle carries. The last tests read the repository: lib/hf/layers.arch
//! must report the baseline it was written against.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use viv::arch::{self, Agreement, ModSrc, ProgramFacts, RunGraph, World};

fn m(name: &str, src: &str) -> ModSrc {
    ModSrc { name: name.into(), path: format!("{name}.viv"), src: src.into(), intent: None }
}

const LENS: &str = "\
# a small system
system demo  a demonstration
  covers   lib/demo ref

layer base  library
  what     the bottom
  modules  lib/demo/base lib/demo/table
layer top  library
  modules  lib/demo/top lib/demo/stage
layer refs  verification
  modules  ref/*

order  base < top

shared  lib/demo/table
  why   the table accessors

private  COL_*
  owner  lib/demo/table
  why    the column layout

budget def pick
  fanin  1
budget def top_entry
  reach  leaf 1

multipath solve  solve the thing
  intent    solve_of
  registry  solve_methods
  members   solve_a solve_b solve_c
  default   solve_a
  through   solve_of
  contract  within 1e-12 [test \"the members agree\"]
  checked   [test \"the members agree\"]
";

fn demo_world() -> Vec<ModSrc> {
    vec![
        m("lib/demo/table", "COL_X = 0\nCOL_Y = 1\ndef tab_x(t: Arr, i) {\n    return t[i, COL_X]\n}\n"),
        m("lib/demo/base", "import demo.table\nimport ref.pack\ndef leaf(x) {\n    return x + 1\n}\ndef pick(x) {\n    return x\n}\ndef reads_cols(t: Arr) {\n    return t[0, COL_Y]\n}\ntest \"base\" {\n    assert_near(ref_value(), 1.0, 0.0)\n}\n"),
        m(
            "lib/demo/stage",
            "import demo.base\nimport demo.table\n\
             def helper(x) {\n    return x * 2\n}\n\
             def solve_a(x) {\n    return helper(x) + pick(x)\n}\n\
             def solve_b(x) {\n    return helper(x) - pick(x) + tab_x(zeros(1, 2), 0)\n}\n\
             def solve_c(x) {\n    return x + tab_x(zeros(1, 2), 0) + pick(x)\n}\n\
             def solve_methods() {\n    return [solve_a, solve_b, solve_c]\n}\n\
             #| Solve the thing with the member the options pick.\n\
             def solve_of(x, k) {\n    ms = solve_methods()\n    return apply(ms[k], x)\n}\n\
             test \"the members agree\" {\n    assert_near(solve_of(1.0, 0), solve_of(1.0, 1), 1e-12)\n}\n",
        ),
        m("lib/demo/top", "import demo.stage\nimport demo.extra\ndef top_entry(x) {\n    return mid(x) + pick(x)\n}\ndef mid(x) {\n    return leaf(x)\n}\ndef top_solve(x) {\n    return solve_of(x, 0)\n}\n"),
        m("lib/demo/extra", "def extra_fn(x) {\n    return x\n}\n"),
        m("ref/pack", "import demo.top\ndef ref_value() {\n    return top_entry(0.0)\n}\n"),
    ]
}

fn covered() -> Vec<String> {
    ["lib/demo/table", "lib/demo/base", "lib/demo/stage", "lib/demo/top", "ref/pack", "lib/demo/extra"].iter().map(|s| s.to_string()).collect()
}

fn check_demo() -> (arch::ArchFile, arch::Checked) {
    let a = arch::parse("demo.arch", LENS);
    let mut w = World::new(&demo_world());
    let c = arch::check(&a, &mut w, &covered(), &[], &Agreement::new());
    (a, c)
}

fn keys(c: &arch::Checked) -> Vec<String> {
    c.findings.iter().map(|f| f.key.clone()).collect()
}

#[test]
fn the_file_reads_back_as_written() {
    let a = arch::parse("demo.arch", LENS);
    assert!(a.errors.is_empty(), "{:?}", a.errors);
    assert_eq!(a.system, "demo");
    assert_eq!(a.covers, vec!["lib/demo", "ref"]);
    assert_eq!(a.layers.len(), 3);
    assert_eq!(a.layers[2].kind, arch::Kind::Verification);
    assert_eq!(a.order, vec!["base", "top"]);
    assert_eq!(a.private[0].owners, vec!["lib/demo/table"]);
    assert_eq!(a.budgets.len(), 2);
    assert_eq!(a.budgets[1].limits[0].arg.as_deref(), Some("leaf"));
    let s = &a.stages[0];
    assert!(s.multi);
    assert_eq!(s.members, vec!["solve_a", "solve_b", "solve_c"]);
    assert_eq!(s.through, vec!["solve_of"]);
    assert_eq!(s.intent_def(), Some("solve_of"));
}

#[test]
fn a_bad_file_says_what_is_wrong_and_where() {
    let a = arch::parse("bad.arch", "layer a  somewhere\n  modules x\n  colour red\nwidget w\norder a < b\nmultipath one  just one\n  members only\n");
    let msgs: Vec<String> = a.errors.iter().map(|(l, m)| format!("{l}: {m}")).collect();
    let all = msgs.join("\n");
    assert!(all.contains("1: layer `a`: the kind is `somewhere`"), "{all}");
    assert!(all.contains("3: `colour` is not a field of `layer`"), "{all}");
    assert!(all.contains("4: `widget` is not a block"), "{all}");
    assert!(all.contains("`order` names `b`, which is not a layer"), "{all}");
    assert!(all.contains("`multipath one` has 1 member"), "{all}");
}

#[test]
fn patterns_match_one_segment_or_any_depth() {
    assert!(arch::matches("lib/basis/*", "lib/basis/pople631g"));
    assert!(!arch::matches("lib/basis/*", "lib/basis/x/y"));
    assert!(arch::matches("scf/suite/**", "scf/suite/ref/sts"));
    assert!(arch::matches("scf/lock_*", "scf/lock_water"));
    assert!(!arch::matches("scf/lock_*", "scf/story_water"));
    assert!(arch::matches("lib/hf/hf", "lib/hf/hf"));
    assert!(!arch::matches("lib/hf/hf", "lib/hf/hfx"));
}

#[test]
fn a_library_that_uses_verification_code_is_reported_with_who_needs_it() {
    let (_, c) = check_demo();
    let f = c.findings.iter().find(|f| f.key == "verification-dep lib/demo/base -> ref/pack").expect("found");
    assert_eq!(f.kind, "verification-dep");
    assert_eq!((f.path.as_str(), f.line), ("lib/demo/base.viv", 2));
    assert!(f.message.contains("only its tests need it"), "{}", f.message);
    assert!(f.detail.iter().any(|d| d.contains("from its tests, first at lib/demo/base.viv:")), "{:?}", f.detail);
    // the verification module may use the library freely
    assert!(!keys(&c).iter().any(|k| k.starts_with("verification-dep ref/pack")), "{:?}", keys(&c));
}

#[test]
fn a_use_of_a_later_layer_is_upward_and_a_module_in_no_layer_is_unassigned() {
    let src = demo_world();
    let mut mods = src.clone();
    // base (the bottom layer) reaches up into stage
    mods[1].src = format!("import demo.stage\n{}", mods[1].src);
    let a = arch::parse("demo.arch", LENS);
    let mut w = World::new(&mods);
    let c = arch::check(&a, &mut w, &covered(), &[], &Agreement::new());
    let f = c.findings.iter().find(|f| f.kind == "upward").expect("an upward use");
    assert_eq!(f.key, "upward lib/demo/base -> lib/demo/stage");
    assert_eq!(f.line, 1);
    assert!(keys(&c).contains(&"unassigned lib/demo/extra".to_string()), "{:?}", keys(&c));
}

#[test]
fn a_private_name_read_outside_its_owner_is_reported_with_its_lines() {
    let (_, c) = check_demo();
    let f = c.findings.iter().find(|f| f.kind == "private").expect("private");
    assert_eq!(f.key, "private lib/demo/base reads COL_*");
    assert!(f.message.contains("COL_Y"), "{}", f.message);
    assert_eq!(f.line, 10);
    // the owner reads its own columns
    assert!(!c.findings.iter().any(|f| f.kind == "private" && f.path.contains("table")));
}

#[test]
fn budgets_count_callers_and_the_shortest_chain() {
    let (_, c) = check_demo();
    let fan = c.findings.iter().find(|f| f.key == "budget def pick fanin").expect("pick has 4 callers");
    assert!(fan.message.contains("4 callers; its budget is 1"), "{}", fan.message);
    let reach = c.findings.iter().find(|f| f.key == "budget def top_entry reach leaf").expect("reach");
    assert!(reach.message.contains("in 2 calls"), "{}", reach.message);
    assert!(reach.detail[0].contains("top_entry -> mid -> leaf"), "{:?}", reach.detail);
    assert_eq!(c.values.get("budget def pick fanin"), Some(&4));
}

#[test]
fn members_that_share_code_are_one_implementation_and_shared_modules_do_not_count() {
    let (a, c) = check_demo();
    let f = &c.facts[0];
    // solve_a and solve_b share `helper`; solve_c shares only `pick` (a
    // budgeted def of the base layer, not stage machinery) and `tab_x`, of a
    // shared module
    assert_eq!(a.stages[0].id, "solve");
    let fams: Vec<Vec<String>> = f.families.clone();
    assert!(fams.iter().any(|x| x.contains(&"solve_a".to_string()) && x.contains(&"solve_b".to_string())), "{fams:?}");
    assert!(f.ties.iter().any(|(x, y, s)| x == "solve_a" && y == "solve_b" && s.contains(&"helper".to_string())), "{:?}", f.ties);
    // `pick` ties all three: shared code outside the shared modules
    assert_eq!(fams.len(), 1, "{fams:?}");
    // with `pick`'s module declared shared, solve_c stands alone
    let lens = LENS.replace("shared  lib/demo/table", "shared  lib/demo/table lib/demo/base");
    let a2 = arch::parse("demo.arch", &lens);
    let mut w = World::new(&demo_world());
    let c2 = arch::check(&a2, &mut w, &covered(), &[], &Agreement::new());
    assert_eq!(c2.facts[0].families.len(), 2, "{:?}", c2.facts[0].families);
}

#[test]
fn a_default_reached_through_drivers_is_indirect_and_a_wrong_registry_is_caught() {
    let (_, c) = check_demo();
    let f = c.findings.iter().find(|f| f.kind == "indirect-default").expect("indirect");
    assert_eq!(f.key, "indirect-default solve");
    assert!(f.message.contains("through solve_of and the registry `solve_methods`"), "{}", f.message);
    // members the registry does not hold, a default that is not a member
    let lens = LENS.replace("members   solve_a solve_b solve_c", "members   solve_a solve_c").replace("default   solve_a", "default   solve_b");
    let a = arch::parse("demo.arch", &lens);
    let mut w = World::new(&demo_world());
    let c = arch::check(&a, &mut w, &covered(), &[], &Agreement::new());
    let k = keys(&c);
    assert!(k.contains(&"stage solve members".to_string()), "{k:?}");
    assert!(k.contains(&"stage solve default".to_string()), "{k:?}");
    // a contract in words must cite its test, and the test must exist
    let lens = LENS.replace("contract  within 1e-12 [test \"the members agree\"]", "contract  within 1e-12").replace("checked   [test \"the members agree\"]", "checked   [test \"no such test\"]");
    let a = arch::parse("demo.arch", &lens);
    let mut w = World::new(&demo_world());
    let c = arch::check(&a, &mut w, &covered(), &[], &Agreement::new());
    let k = keys(&c);
    assert!(k.contains(&"stage solve contract citation".to_string()), "{k:?}");
    assert!(k.contains(&"stage solve cites test no such test".to_string()), "{k:?}");
    // with no `through`, the default must be called directly
    let lens = LENS.replace("  through   solve_of\n", "");
    let a = arch::parse("demo.arch", &lens);
    let mut w = World::new(&demo_world());
    let c = arch::check(&a, &mut w, &covered(), &[], &Agreement::new());
    assert!(keys(&c).contains(&"stage solve direct".to_string()), "{:?}", keys(&c));
}

#[test]
fn the_intent_of_a_stage_is_its_intent_defs_what() {
    let w = World::new(&demo_world());
    assert_eq!(arch::what_of(&w, "solve_of").as_deref(), Some("Solve the thing with the member the options pick."));
    assert_eq!(arch::what_of(&w, "helper"), None);
}

const RUN: &str = "1\nPASS the members agree\nFAIL other -- x\n__graph__ {\"nodes\":[\"<main>\",\"a\",\"b\",\"c\",\"d\"],\"edges\":[{\"from\":\"<main>\",\"to\":\"a\",\"count\":1},{\"from\":\"a\",\"to\":\"b\",\"count\":3},{\"from\":\"b\",\"to\":\"c\",\"count\":2},{\"from\":\"c\",\"to\":\"a\",\"count\":1},{\"from\":\"a\",\"to\":\"d\",\"count\":1},{\"from\":\"b\",\"to\":\"b\",\"count\":9}]}\n";

#[test]
fn a_runs_deepest_chain_visits_each_def_once() {
    let g = RunGraph::of_output(RUN).expect("reads");
    assert_eq!(g.nodes.len(), 5);
    assert_eq!(g.passed, vec!["the members agree"]);
    assert_eq!(g.failed, vec!["other"]);
    assert_eq!(g.deepest(), vec!["<main>", "a", "b", "c"]);
}

#[test]
fn a_program_report_counts_what_it_loads_and_runs_and_its_stages() {
    let a = arch::parse("demo.arch", LENS);
    let mut w = World::new(&demo_world());
    let _ = arch::assign_layers(&a, &mut w, &covered());
    let run = "__graph__ {\"nodes\":[\"<main>\",\"top_entry\",\"mid\",\"leaf\",\"pick\",\"solve_of\",\"solve_methods\",\"solve_a\",\"helper\"],\"edges\":[{\"from\":\"<main>\",\"to\":\"top_entry\",\"count\":1},{\"from\":\"top_entry\",\"to\":\"mid\",\"count\":1},{\"from\":\"mid\",\"to\":\"leaf\",\"count\":1},{\"from\":\"top_entry\",\"to\":\"pick\",\"count\":1},{\"from\":\"<main>\",\"to\":\"solve_of\",\"count\":1},{\"from\":\"solve_of\",\"to\":\"solve_methods\",\"count\":1},{\"from\":\"solve_of\",\"to\":\"solve_a\",\"count\":1},{\"from\":\"solve_a\",\"to\":\"helper\",\"count\":1}]}\n";
    let p = ProgramFacts { name: "lib/demo/top".into(), modules: vec!["lib/demo/top".into(), "lib/demo/stage".into(), "lib/demo/base".into(), "lib/demo/table".into(), "ref/pack".into(), "lib/demo/extra".into()], expanded_defs: 0, run: RunGraph::of_output(run) };
    let facts: Vec<arch::StageFacts> = a.stages.iter().map(|s| arch::stage_facts(&a, &w, s)).collect();
    let mut agreement = Agreement::new();
    agreement.insert("solve".into(), vec!["solve_a".into(), "solve_c".into()]);
    let mut values = BTreeMap::new();
    let r = arch::report(&a, &w, &p, &facts, &agreement, &mut values);
    assert_eq!(r.modules, 6);
    assert_eq!(r.ran, Some(9));
    assert!(r.idle.iter().any(|(m, _, _)| m == "ref/pack"), "{:?}", r.idle);
    // two chains of three calls: the first in name order is kept
    assert_eq!(r.deepest, vec!["<main>", "solve_of", "solve_a", "helper"]);
    // the stage's driver sits on it
    assert_eq!(r.indirection, vec!["solve_of"]);
    assert!(r.pairs.iter().any(|(f, t, _, _)| f == "lib/demo/top" && t == "lib/demo/base"), "{:?}", r.pairs);
    let st = &r.stages[0];
    assert_eq!(st.ran, vec!["solve_a"]);
    assert!(st.on_path && !st.direct);
    // solve_a and solve_c were seen agreeing, but `pick` ties all three into one family
    assert_eq!(st.agreeing_families, Some(1));
    assert!(!st.robust());
    assert_eq!(arch::robustness(&r), (0, 1));
    let text = arch::report_text(&r);
    assert!(text.contains("robustness: 0 of 1 stages"), "{text}");
    // the JSON parses and says the same
    let j = viv::story::parse_json(&arch::report_json(&r, &a, &facts)).expect("json");
    assert_eq!(j.get("ran").and_then(|x| x.num()), Some(9.0));
    let stages = match j.get("stages") {
        Some(viv::story::Json::Arr(v)) => v.clone(),
        _ => panic!("stages"),
    };
    assert_eq!(stages[0].get("independent").and_then(|x| x.num()), Some(1.0));
}

#[test]
fn check_json_is_json_with_every_finding() {
    let (_, c) = check_demo();
    let j = viv::story::parse_json(&arch::check_json("demo.arch", &c)).expect("json");
    let n = c.findings.iter().filter(|f| f.is_violation()).count();
    assert_eq!(j.get("violations").and_then(|x| x.num()), Some(n as f64));
    match j.get("findings") {
        Some(viv::story::Json::Arr(v)) => {
            assert_eq!(v.len(), c.findings.len());
            assert!(v.iter().all(|f| f.get("key").is_some() && f.get("line").is_some() && f.get("kind").is_some()));
        }
        _ => panic!("findings"),
    }
}

#[test]
fn the_ratchet_lets_violations_only_go_down() {
    let (a, c) = check_demo();
    let mut agreement = Agreement::new();
    agreement.insert("solve".into(), vec!["solve_a".into(), "solve_b".into(), "solve_c".into()]);
    let rec = arch::record_of(&c, &agreement);
    let mut all = BTreeMap::new();
    all.insert("demo.arch".to_string(), rec.clone());
    let text = arch::ratchet_text(&all);
    let back = arch::parse_ratchet(&text).expect("reads back");
    assert_eq!(back["demo.arch"], rec);
    // the same state is neither worse nor better
    let (worse, better) = arch::ratchet_compare(&rec, &rec, &a, &c.facts);
    assert!(worse.is_empty() && better.is_empty(), "{worse:?} {better:?}");
    // a new violation is worse; a fixed one is better
    let mut now = rec.clone();
    now.violations.insert("upward x -> y".into());
    let first = now.violations.iter().find(|v| v.starts_with("private")).cloned().unwrap();
    now.violations.remove(&first);
    let (worse, better) = arch::ratchet_compare(&now, &rec, &a, &c.facts);
    assert_eq!(worse, vec!["new violation: upward x -> y".to_string()]);
    assert!(better.iter().any(|b| b.contains(&first)), "{better:?}");
    // a budget over its limit may not grow
    let mut now = rec.clone();
    *now.values.get_mut("budget def pick fanin").unwrap() += 1;
    let (worse, _) = arch::ratchet_compare(&now, &rec, &a, &c.facts);
    assert_eq!(worse, vec!["budget def pick fanin: 4 -> 5".to_string()]);
}

// ---------------------------------------------------------------------------
// The repository
// ---------------------------------------------------------------------------

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut es: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    es.sort();
    for p in es {
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|e| e == "viv") {
            out.push(p);
        }
    }
}

fn repo_mod(name: &str) -> Option<ModSrc> {
    let src = std::fs::read_to_string(repo().join(format!("{name}.viv"))).ok()?;
    Some(ModSrc { name: name.into(), path: format!("{name}.viv"), src, intent: std::fs::read_to_string(repo().join(format!("{name}.intent"))).ok() })
}

/// The modules lib/hf/layers.arch covers, and what they use.
fn hf_world(a: &arch::ArchFile) -> (Vec<ModSrc>, Vec<String>) {
    let root = repo().canonicalize().unwrap();
    let mut names = Vec::new();
    for c in &a.covers {
        let d = root.join(c);
        if d.is_dir() {
            let mut ps = Vec::new();
            walk(&d, &mut ps);
            for p in ps {
                names.push(p.strip_prefix(&root).unwrap().with_extension("").display().to_string());
            }
        } else {
            names.push(c.clone());
        }
    }
    let covered = names.clone();
    let mut out: Vec<ModSrc> = Vec::new();
    let mut k = 0;
    while k < names.len() {
        let n = names[k].clone();
        k += 1;
        if out.iter().any(|m| m.name == n) {
            continue;
        }
        let Some(m) = repo_mod(&n) else { continue };
        for (u, _) in arch::uses_of(&m.name, &m.src, &|c: &str| repo().join(format!("{c}.viv")).is_file()) {
            names.push(u);
        }
        out.push(m);
    }
    (out, covered)
}

fn hf_lens() -> arch::ArchFile {
    arch::parse("lib/hf/layers.arch", &std::fs::read_to_string(repo().join("lib/hf/layers.arch")).unwrap())
}

#[test]
fn the_hf_lens_reports_its_baseline() {
    let a = hf_lens();
    assert!(a.errors.is_empty(), "{:?}", a.errors);
    let (mods, covered) = hf_world(&a);
    let mut w = World::new(&mods);
    let c = arch::check(&a, &mut w, &covered, &[], &Agreement::new());
    let k = keys(&c);
    let has = |x: &str| assert!(k.iter().any(|y| y == x), "missing `{x}` in\n{}", k.join("\n"));
    // no library module loads a reference pack: the tests that need them
    // live in lib/hf/tests
    assert!(!k.iter().any(|x| x.starts_with("verification-dep") && x.contains("lib/hfref/")), "{}", k.join("\n"));
    // no library module uses verification code: the cross-checked pipeline
    // (lib/hf/hf) and the contract page (lib/hf/integrals) are verification
    assert!(!k.iter().any(|x| x.starts_with("verification-dep")), "{}", k.join("\n"));
    // the J/K registry left the library with the contract page
    assert!(!k.iter().any(|x| x.starts_with("upward")), "{}", k.join("\n"));
    // the basis table's columns read outside the modules that define and
    // build the tables: the integral kernels still do, in their loops
    for mm in ["lib/hf/md_eri", "lib/hf/md_onebody"] {
        has(&format!("private {mm} reads AT_* SH_* PR_*"));
    }
    for mm in ["lib/hf/molecule", "lib/hf/cart2sph", "lib/hf/jk_direct", "lib/hf/scf"] {
        assert!(!k.iter().any(|x| x == &format!("private {mm} reads AT_* SH_* PR_*")), "{mm} reads the columns");
    }
    // the SCF kind decided once, when the setup is made (and checked once more
    // by scf_solve)
    assert!(!k.iter().any(|x| x == "budget def scf_method fanin"), "{}", k.join("\n"));
    // one entry path: scf_with -> scf -> scf_solve -> scf_iterate
    assert!(!k.iter().any(|x| x == "budget def scf_with reach scf_iterate"), "{}", k.join("\n"));
    // every default is called by name from library code: no registry or
    // sweep driver stands on the production path
    assert!(!k.iter().any(|x| x.starts_with("indirect-default")), "{}", k.join("\n"));
    for (s, f) in a.stages.iter().zip(&c.facts) {
        assert!(s.default.is_none() || f.direct, "the default of `{}` is not called directly", s.id);
    }
    // no stage declaration is out of date, and no covered module is outside a layer
    assert!(!c.findings.iter().any(|f| f.kind == "stage" || f.kind == "unassigned" || f.kind == "lens"), "{}", c.findings.iter().filter(|f| f.kind == "stage" || f.kind == "unassigned" || f.kind == "lens").map(|f| f.render()).collect::<String>());
    // two ERI families besides MD, which its two members share
    let eri = &c.facts[a.stages.iter().position(|s| s.id == "eri").unwrap()];
    assert_eq!(eri.families.len(), 3, "{:?}", eri.families);
    assert_eq!(eri.tolerance.as_ref().map(|t| t.1), Some(1.0e-12));
}

#[test]
fn the_water_program_loads_no_reference_pack() {
    let a = hf_lens();
    let root = "scf/story_water";
    // the modules, as the expansion includes them (lib/basis/named's
    // all-tables default stays out: the program uses lib/basis/pople631g)
    let src = std::fs::read_to_string(repo().join(format!("{root}.viv"))).unwrap();
    let exp = viv::expand_uses_mapped(&src, &mut |n: &str| std::fs::read_to_string(repo().join(format!("{n}.viv"))).map_err(|e| e.to_string()), &mut |_, n, _| Err(format!("{n}: not Viv"))).unwrap();
    let mut modules = vec![root.to_string()];
    modules.extend(exp.units.iter().skip(1).map(|u| u.name.clone()));
    // 30 since lib/hf/mdg_kernels and lib/hf/mdg_eri (the generated ERI
    // kernels and their driver) joined the production path
    assert_eq!(modules.len(), 30);
    assert!(modules.iter().filter(|m| m.starts_with("lib/basis/")).count() == 3, "{modules:?}");
    let (mut mods, covered) = hf_world(&a);
    for n in &modules {
        if !mods.iter().any(|m| &m.name == n) {
            mods.push(repo_mod(n).unwrap());
        }
    }
    let mut w = World::new(&mods);
    let _ = arch::assign_layers(&a, &mut w, &covered);
    let p = ProgramFacts { name: root.into(), modules, expanded_defs: 0, run: None };
    let facts: Vec<arch::StageFacts> = a.stages.iter().map(|s| arch::stage_facts(&a, &w, s)).collect();
    let mut values = BTreeMap::new();
    let r = arch::report(&a, &w, &p, &facts, &Agreement::new(), &mut values);
    assert_eq!(r.modules, 30);
    assert!(r.layers.iter().all(|l| l.0 != "reference" || l.1 == 0), "{:?}", r.layers);
    // one basis table, the lookup and the table record; lib/xyz, lib/hf/basis,
    // molecule, mole and opts
    assert_eq!(r.layers.iter().find(|l| l.0 == "input").map(|l| l.1), Some(8));
}

#[test]
fn the_bundle_carries_the_programs_multipaths_and_where_they_hang() {
    let root = "scf/story_water";
    let mut names = vec![root.to_string()];
    let mut k = 0;
    while k < names.len() {
        let src = std::fs::read_to_string(repo().join(format!("{}.viv", names[k]))).unwrap();
        for (u, _) in arch::uses_of(&names[k], &src, &|c: &str| repo().join(format!("{c}.viv")).is_file()) {
            if !names.contains(&u) {
                names.push(u);
            }
        }
        k += 1;
    }
    let js = |s: &str| viv::debug_lens::json_str(s);
    let module = |n: &str| {
        let src = std::fs::read_to_string(repo().join(format!("{n}.viv"))).unwrap();
        let mut o = format!("{{\"name\":{},\"src\":{}", js(n), js(&src));
        for kind in ["intent", "debug"] {
            if let Ok(t) = std::fs::read_to_string(repo().join(format!("{n}.{kind}"))) {
                o.push_str(&format!(",\"{kind}\":{}", js(&t)));
            }
        }
        o.push('}');
        o
    };
    let used: Vec<String> = names[1..].iter().map(|n| module(n)).collect();
    let lens = std::fs::read_to_string(repo().join("lib/hf/layers.arch")).unwrap();
    // the modules the stages are defined in that the program does not load,
    // with what they use, as www/scripts/explore-build.mjs sends them
    let a = hf_lens();
    let (world, _) = hf_world(&a);
    let mut extra: Vec<String> = Vec::new();
    for s in &a.stages {
        let contract = s.contract.as_ref().and_then(|c| c.split_whitespace().next().map(String::from));
        for n in s.members.iter().chain(s.intent.iter()).chain(s.registry.iter()).chain(contract.iter()) {
            if let Some(m) = world.iter().find(|m| m.src.lines().any(|l| l.starts_with(&format!("def {n}(")))) {
                if !names.contains(&m.name) && !extra.contains(&m.name) {
                    extra.push(m.name.clone());
                }
            }
        }
    }
    let mut k = 0;
    while k < extra.len() {
        let src = std::fs::read_to_string(repo().join(format!("{}.viv", extra[k]))).unwrap();
        for (u, _) in arch::uses_of(&extra[k], &src, &|c: &str| repo().join(format!("{c}.viv")).is_file()) {
            if !names.contains(&u) && !extra.contains(&u) {
                extra.push(u);
            }
        }
        k += 1;
    }
    assert!(extra.iter().any(|m| m == "lib/hf/hgp_eri") && !extra.iter().any(|m| m.starts_with("lib/hfref/")), "{extra:?}");
    let extra_j: Vec<String> = extra.iter().map(|n| module(n)).collect();
    let req = format!(
        "{{\"root\":{},\"used\":[{}],\"arch\":{{\"path\":\"lib/hf/layers.arch\",\"text\":{},\"modules\":[{}]}},\"agreeing\":{{\"eri\":[\"mderi_screened\",\"hgp_eri\"]}}}}",
        module(root),
        used.join(","),
        js(&lens),
        extra_j.join(",")
    );
    let out = viv::graph::bundle_request_json(&req);
    assert!(!out.starts_with("error"), "{}", &out[..out.len().min(400)]);
    let b = viv::story::parse_json(&out).expect("the bundle is JSON");
    let mp = b.get("multipaths").expect("multipaths");
    let stages = match mp.get("stages") {
        Some(viv::story::Json::Arr(v)) => v.clone(),
        _ => panic!("stages"),
    };
    let eri = stages.iter().find(|s| s.get("id").and_then(|x| x.str()) == Some("eri")).expect("eri");
    let hosts = match eri.get("hosts") {
        Some(viv::story::Json::Arr(v)) => v.iter().filter_map(|h| h.get("def").and_then(|d| d.str()).map(String::from)).collect::<Vec<_>>(),
        _ => panic!("hosts"),
    };
    // the stage hangs where the library calls its default by name, not at the
    // contract def of the verification layer
    assert!(hosts.contains(&"ao_eri".to_string()) && !hosts.contains(&"int_eri".to_string()), "{hosts:?}");
    assert_eq!(eri.get("agreeing_families").and_then(|x| x.num()), Some(2.0));
    assert!(eri.get("what").and_then(|x| x.str()).is_some_and(|w| !w.is_empty()));
    let host_of = |id: &str| -> Vec<String> {
        let s = stages.iter().find(|s| s.get("id").and_then(|x| x.str()) == Some(id)).unwrap();
        match s.get("hosts") {
            Some(viv::story::Json::Arr(v)) => v.iter().filter_map(|h| h.get("def").and_then(|d| d.str()).map(String::from)).collect(),
            _ => Vec::new(),
        }
    };
    assert!(host_of("overlap").contains(&"ao_overlap".to_string()));
    let hc = host_of("kinetic");
    assert!(hc.contains(&"ao_hcore".to_string()), "{hc:?}");
    // jk_of picks the member: the picture folds it into the fork, which hangs where it is called
    assert!(host_of("jk").contains(&"scf_iterate".to_string()), "{:?}", host_of("jk"));
}
