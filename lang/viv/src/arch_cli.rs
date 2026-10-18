// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv arch check` and `viv arch report`: the
//! file system side. viv::arch does the work on texts; this part finds the
//! `.arch` files and the modules they cover, runs the programs the budgets
//! name, and runs the tests a multi-path cites when asked to measure
//! agreement (`--verify`).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;


use viv::arch::{self, Agreement, ArchFile, ModSrc, ProgramFacts, RunGraph, World};

pub const USAGE: &str = "usage: viv arch check [<file.arch>...] [--json] [--check|--update] [--static] [--verify] [--ratchet <file>]
       viv arch report <program.viv> [--arch <file.arch>] [--json] [--static] [--verify] [--from <run output>]
  check: the layer rules, the budgets and the multi-path declarations of each lens (with no file, every
         .arch file in the repository); --check fails when anything is worse than arch_ratchet.json
         records, --update records the current state.
  report: what one program loads, runs and reaches, measured against the lens that covers it.
  --static skips running programs (no run-time budgets, no ran counts); --verify runs the tests each
  multi-path cites and records which members were seen agreeing.";

const SKIP_DIRS: [&str; 7] = ["target", "node_modules", ".git", "dist", "public", ".astro", "www"];

fn repo_root() -> PathBuf {
    let mut p = std::fs::canonicalize(".").unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if p.join("Cargo.toml").exists() && p.join("lib").is_dir() {
            return p;
        }
        if !p.pop() {
            return PathBuf::from(".");
        }
    }
}

fn walk(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut es: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    es.sort();
    for p in es {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if p.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) && !name.starts_with("target") && !name.starts_with('.') {
                walk(&p, ext, out);
            }
        } else if name.ends_with(ext) {
            out.push(p);
        }
    }
}

/// A path relative to the root, as findings print it.
fn rel(root: &Path, p: &Path) -> String {
    let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    abs.strip_prefix(root).map(|r| r.display().to_string()).unwrap_or_else(|_| p.display().to_string())
}

fn read(p: &Path) -> Option<String> {
    crate::TracedIo.read_to_string(p).ok()
}

fn mod_src(root: &Path, name: &str) -> Option<ModSrc> {
    let p = root.join(format!("{name}.viv"));
    let src = read(&p)?;
    Some(ModSrc { name: name.to_string(), path: format!("{name}.viv"), src, intent: read(&root.join(format!("{name}.intent"))) })
}

/// The modules a lens covers (every `.viv` under its `covers` directories),
/// and every module those use, transitively.
fn world_of(root: &Path, a: &ArchFile, extra: &[String]) -> (Vec<ModSrc>, Vec<String>) {
    let mut names: Vec<String> = Vec::new();
    for c in &a.covers {
        let d = root.join(c);
        if d.is_dir() {
            let mut ps = Vec::new();
            walk(&d, ".viv", &mut ps);
            for p in ps {
                names.push(rel(root, &p).trim_end_matches(".viv").to_string());
            }
        } else if root.join(format!("{c}.viv")).exists() {
            names.push(c.clone());
        }
    }
    let covered = names.clone();
    names.extend(extra.iter().cloned());
    let mut out: Vec<ModSrc> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut k = 0;
    while k < names.len() {
        let n = names[k].clone();
        k += 1;
        if !seen.insert(n.clone()) {
            continue;
        }
        let Some(m) = mod_src(root, &n) else { continue };
        for (u, _) in arch::uses_of(&m.name, &m.src, &|c: &str| root.join(format!("{c}.viv")).is_file()) {
            if !seen.contains(&u) {
                names.push(u);
            }
        }
        out.push(m);
    }
    (out, covered)
}

/// Run a program in the interpreter, in debug mode as `viv run` does, and
/// read its call graph and test results.
fn run_program(path: &str) -> Result<RunGraph, String> {
    let src = crate::TracedIo.read_to_string(&PathBuf::from(path)).map_err(|e| format!("cannot read {path}: {e}"))?;
    let expanded = super::expand_uses_from(&src, path)?;
    let h = std::thread::Builder::new().stack_size(super::INTERP_STACK).spawn(move || viv::interp::run(&expanded)).map_err(|e| e.to_string())?;
    let out = h.join().map_err(|_| "the interpreter panicked".to_string())??;
    RunGraph::of_output(&out).ok_or_else(|| format!("{path}: the run printed no call graph"))
}

/// What a program loads, and (unless static) its run.
fn program_facts(root: &Path, name: &str, run: bool, from: Option<&str>) -> Result<ProgramFacts, String> {
    let path = root.join(format!("{name}.viv"));
    let ps = path.display().to_string();
    let input = super::lens_input(&ps)?;
    let mut modules = vec![name.to_string()];
    modules.extend(input.used.iter().map(|m| m.name.clone()));
    let expanded_defs = viv::parse(&input.expanded, knotkit::span::SourceId(0)).map(|p| p.items.iter().filter(|i| matches!(i, viv::ast::Item::Def(_))).count()).unwrap_or(0);
    let run = match from {
        Some(f) => Some(RunGraph::of_output(&read(Path::new(f)).ok_or_else(|| format!("cannot read {f}"))?).ok_or_else(|| format!("{f}: no `__graph__` line"))?),
        None if run => Some(run_program(&ps)?),
        None => None,
    };
    Ok(ProgramFacts { name: name.to_string(), modules, expanded_defs, run })
}

/// Run the tests each multi-path cites and record which of its members ran
/// in a run where those tests passed.
fn verify(root: &Path, a: &ArchFile, w: &World) -> Result<Agreement, String> {
    let tests = arch::tests_of(w);
    let mut runs: BTreeMap<String, RunGraph> = BTreeMap::new();
    let mut out = Agreement::new();
    for s in a.stages.iter().filter(|s| s.multi) {
        let mut files: Vec<(String, Vec<String>)> = Vec::new();
        for text in s.checked.iter() {
            for t in viv::lens::citations(text, "test") {
                if let Some((m, _)) = tests.get(&t) {
                    let name = w.mods[*m].name.clone();
                    match files.iter_mut().find(|f| f.0 == name) {
                        Some(f) => f.1.push(t),
                        None => files.push((name, vec![t])),
                    }
                }
            }
            for p in viv::lens::citations(text, "program") {
                files.push((p.trim_end_matches(".viv").to_string(), Vec::new()));
            }
        }
        let mut seen: Vec<String> = Vec::new();
        for (file, want) in files {
            if !runs.contains_key(&file) {
                eprintln!("viv arch: running {file}.viv for the stages that cite it");
                let g = run_program(&root.join(format!("{file}.viv")).display().to_string()).map_err(|e| format!("--verify: {e}"))?;
                runs.insert(file.clone(), g);
            }
            let g = &runs[&file];
            let passed = want.iter().all(|t| g.passed.contains(t)) && !want.iter().any(|t| g.failed.contains(t)) && (!want.is_empty() || g.failed.is_empty());
            if !passed {
                continue;
            }
            for m in &s.members {
                if g.nodes.contains(m) && !seen.contains(m) {
                    seen.push(m.clone());
                }
            }
        }
        seen.sort_by_key(|m| s.members.iter().position(|x| x == m));
        out.insert(s.id.clone(), seen);
    }
    Ok(out)
}

fn fail(m: String) -> ExitCode {
    eprintln!("{m}");
    ExitCode::FAILURE
}

fn ratchet_path(root: &Path, args: &[String]) -> PathBuf {
    match args.iter().position(|a| a == "--ratchet").and_then(|k| args.get(k + 1)) {
        Some(p) => PathBuf::from(p),
        None => root.join("arch_ratchet.json"),
    }
}

fn positional(args: &[String], skip: usize) -> Vec<String> {
    let mut pos = Vec::new();
    let mut k = skip;
    while k < args.len() {
        let a = &args[k];
        if matches!(a.as_str(), "--ratchet" | "--arch" | "--from") {
            k += 2;
            continue;
        }
        if !a.starts_with("--") {
            pos.push(a.clone());
        }
        k += 1;
    }
    pos
}

/// The lens that covers a program: the `.arch` file whose `covers` holds it,
/// or the one given.
fn arch_for(root: &Path, program: &str, given: Option<&String>) -> Result<(String, ArchFile), String> {
    let cands: Vec<PathBuf> = match given {
        Some(g) => vec![PathBuf::from(g)],
        None => {
            let mut v = Vec::new();
            walk(root, ".arch", &mut v);
            v
        }
    };
    for p in &cands {
        let text = read(p).ok_or_else(|| format!("cannot read {}", p.display()))?;
        let r = rel(root, p);
        let a = arch::parse(&r, &text);
        if given.is_some() || a.covers.iter().any(|c| program == c || program.starts_with(&format!("{c}/"))) {
            return Ok((r, a));
        }
    }
    Err(format!("no .arch file covers {program} (give one with --arch)"))
}

fn program_name(root: &Path, p: &str) -> String {
    rel(root, Path::new(p)).trim_end_matches(".viv").to_string()
}

/// `viv arch report <program.viv>`.
pub fn report(args: &[String]) -> ExitCode {
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|k| args.get(k + 1)).cloned();
    let root = repo_root();
    let pos = positional(args, 3);
    let Some(prog) = pos.first() else { return fail(USAGE.to_string()) };
    let name = program_name(&root, prog);
    let (apath, a) = match arch_for(&root, &name, value("--arch").as_ref()) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    let facts = match program_facts(&root, &name, !flag("--static"), value("--from").as_deref()) {
        Ok(f) => f,
        Err(e) => return fail(format!("viv arch report: {e}")),
    };
    let (mods, covered) = world_of(&root, &a, &facts.modules);
    let mut w = World::new(&mods);
    let _ = arch::assign_layers(&a, &mut w, &covered);
    let agreement = if flag("--verify") {
        match verify(&root, &a, &w) {
            Ok(g) => g,
            Err(e) => return fail(e),
        }
    } else {
        recorded_agreement(&root, args, &apath)
    };
    let sfacts: Vec<arch::StageFacts> = a.stages.iter().map(|s| arch::stage_facts(&a, &w, s)).collect();
    let mut values = BTreeMap::new();
    let r = arch::report(&a, &w, &facts, &sfacts, &agreement, &mut values);
    if flag("--json") {
        println!("{}", arch::report_json(&r, &a, &sfacts));
    } else {
        print!("{}", arch::report_text(&r));
    }
    ExitCode::SUCCESS
}

fn recorded_agreement(root: &Path, args: &[String], apath: &str) -> Agreement {
    read(&ratchet_path(root, args))
        .and_then(|t| arch::parse_ratchet(&t).ok())
        .and_then(|m| m.get(apath).cloned())
        .map(|r| r.agreeing.into_iter().collect())
        .unwrap_or_default()
}

/// `viv arch check [<file.arch>...]`.
pub fn check(args: &[String]) -> ExitCode {
    let flag = |f: &str| args.iter().any(|a| a == f);
    let root = repo_root();
    let pos = positional(args, 3);
    let files: Vec<PathBuf> = if pos.is_empty() {
        let mut v = Vec::new();
        walk(&root, ".arch", &mut v);
        v
    } else {
        pos.iter().map(PathBuf::from).collect()
    };
    if files.is_empty() {
        return fail(format!("no .arch files\n{USAGE}"));
    }
    let rp = ratchet_path(&root, args);
    let mut recorded = match read(&rp).map(|t| arch::parse_ratchet(&t)) {
        None => BTreeMap::new(),
        Some(Ok(r)) => r,
        Some(Err(e)) => return fail(e),
    };
    let mut worse_all: Vec<String> = Vec::new();
    let mut jsons: Vec<String> = Vec::new();
    let mut total = 0;
    for f in &files {
        let Some(text) = read(f) else { return fail(format!("cannot read {}", f.display())) };
        let apath = rel(&root, f);
        let a = arch::parse(&apath, &text);
        // the programs the budgets name
        let progs: Vec<String> = a.budgets.iter().filter_map(|b| b.program.clone()).collect();
        let mut facts = Vec::new();
        for p in &progs {
            match program_facts(&root, p, !flag("--static"), None) {
                Ok(x) => facts.push(x),
                Err(e) => return fail(format!("{apath}: budget program {p}: {e}")),
            }
        }
        let extra: Vec<String> = facts.iter().flat_map(|x| x.modules.clone()).collect();
        let (mods, covered) = world_of(&root, &a, &extra);
        let mut w = World::new(&mods);
        let _ = arch::assign_layers(&a, &mut w, &covered);
        let was = recorded.get(&apath).cloned().unwrap_or_default();
        let agreement: Agreement = if flag("--verify") {
            match verify(&root, &a, &w) {
                Ok(g) => g,
                Err(e) => return fail(e),
            }
        } else {
            was.agreeing.clone().into_iter().collect()
        };
        let c = arch::check(&a, &mut w, &covered, &facts, &agreement);
        total += c.findings.iter().filter(|x| x.is_violation()).count();
        if flag("--json") {
            jsons.push(arch::check_json(&apath, &c));
        } else {
            print!("{}", arch::check_text(&apath, &c));
        }
        let mut now = arch::record_of(&c, &agreement);
        if flag("--static") {
            // values the run gives are not measured: keep the recorded ones
            for (k, v) in &was.values {
                now.values.entry(k.clone()).or_insert(*v);
            }
            for v in &was.violations {
                if v.starts_with("budget program") && v.ends_with(" depth") {
                    now.violations.insert(v.clone());
                }
            }
        }
        if flag("--check") {
            let (worse, better) = arch::ratchet_compare(&now, &was, &a, &c.facts);
            if !recorded.contains_key(&apath) {
                worse_all.push(format!("{apath}: not in {} (`viv arch check --update` records it)", rp.display()));
            }
            worse_all.extend(worse.into_iter().map(|x| format!("{apath}: {x}")));
            for b in better {
                eprintln!("{apath}: better than recorded: {b} (`viv arch check --update` records it)");
            }
        }
        if flag("--update") {
            recorded.insert(apath.clone(), now);
        }
    }
    if flag("--json") {
        println!("{{\"lenses\":[{}],\"violations\":{total}}}", jsons.join(","));
    }
    if flag("--update") {
        if let Err(e) = std::fs::write(&rp, arch::ratchet_text(&recorded)) {
            return fail(format!("cannot write {}: {e}", rp.display()));
        }
        eprintln!("wrote {}", rp.display());
    }
    if flag("--check") {
        if !worse_all.is_empty() {
            for x in &worse_all {
                eprintln!("{x}");
            }
            return fail(format!("viv arch check: {} worse than {} records", worse_all.len(), rp.display()));
        }
        return ExitCode::SUCCESS;
    }
    if total > 0 && !flag("--update") {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
