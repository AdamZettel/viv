// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv lens coverage`, `viv lens needed` and `viv debt`: the file system side. viv::lens_cov does the work on texts.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;


use viv::lens_cov::{self, Known, ModCov, Needed, RepoFile};

pub const USAGE: &str = "usage: viv lens coverage [<file.viv|dir>...] [--json] [--check|--update] [--ratchet <file>]
       viv lens needed [<file.viv|dir>...] [--json] [--program <file.viv>]
       viv debt [<file.viv|dir>...] [--json] [--limit <n>] [--run <program.viv>]
  With no files, the modules the ratchet file (lens_coverage.json at the repository root) lists.";

/// Directories the repository walk skips: build output, dependencies, and the
/// site's copies of programs.
const SKIP_DIRS: [&str; 7] = ["target", "node_modules", ".git", "dist", "public", ".astro", "www"];

/// The repository root: the nearest ancestor of the current directory that
/// holds `Cargo.toml` and `lib/`.
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

/// Every `.viv` file under a directory, sorted.
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut es: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    es.sort();
    for p in es {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if p.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) && !name.starts_with("target") && !name.starts_with('.') {
                walk(&p, out);
            }
        } else if name.ends_with(".viv") {
            out.push(p);
        }
    }
}

/// The repository name of a file: its path under the root without `.viv`.
fn repo_name(root: &Path, p: &Path) -> String {
    let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    abs.strip_prefix(root).map(|r| r.with_extension("").display().to_string()).unwrap_or_else(|_| p.with_extension("").display().to_string())
}

/// Every `.viv` file of the repository, with its uses resolved to names.
fn repo_files(root: &Path) -> Vec<RepoFile> {
    let mut paths = Vec::new();
    walk(root, &mut paths);
    let mut out = Vec::new();
    for p in paths {
        let Ok(src) = crate::TracedIo.read_to_string(&p) else { continue };
        let here = p.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let unit = repo_name(root, &p);
        let uses = super::use_paths(&unit, &src, &here, root).iter().map(|mp| repo_name(root, mp)).collect();
        let ip = super::intent_path_of(&p);
        out.push(RepoFile { name: repo_name(root, &p), path: p.strip_prefix(root).unwrap_or(&p).display().to_string(), src, intent_path: ip.display().to_string(), intent: super::read_opt(&ip), uses });
    }
    out
}

/// What a `serves` citation may name.
fn known_of(root: &Path, files: &[RepoFile]) -> Known {
    let mut k = Known::default();
    for f in files {
        for t in viv::lens::test_names_of(&f.src) {
            k.tests.insert(t);
        }
        if f.name.starts_with("posts/") {
            k.posts.insert(f.name.trim_start_matches("posts/").to_string());
        }
        let dp = root.join(format!("{}.debug", f.name));
        if let Some(Ok(blocks)) = super::read_opt(&dp).map(|t| viv::debug_lens::check_file(&t)) {
            for b in &blocks {
                for item in &b.items {
                    if let viv::ast::DebugItem::Hold { sentence, .. } = item {
                        k.holds.insert(sentence.clone());
                    }
                }
            }
        }
    }
    // claim ids: twelve hex digits in quotes, as the claims database keys them
    for p in ["posts/claim_verdicts.json", "www/src/data/claims.json"] {
        if let Some(t) = super::read_opt(&root.join(p)) {
            for part in t.split('"') {
                if part.len() == 12 && part.chars().all(|c| c.is_ascii_hexdigit()) {
                    k.claims.insert(part.to_string());
                }
            }
        }
    }
    k
}

/// The programs the explorer runs (www/src/data/explorables.mjs), else every
/// `story_*.viv`.
fn stories(root: &Path, files: &[RepoFile]) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(t) = super::read_opt(&root.join("www/src/data/explorables.mjs")) {
        for part in t.split("file: '").skip(1) {
            let f: String = part.chars().take_while(|c| *c != '\'').collect();
            out.push(f.trim_end_matches(".viv").to_string());
        }
    }
    if out.is_empty() {
        out = files.iter().filter(|f| f.name.rsplit('/').next().is_some_and(|b| b.starts_with("story_"))).map(|f| f.name.clone()).collect();
    }
    out
}

fn ratchet_path(root: &Path, args: &[String]) -> PathBuf {
    let v = args.iter().position(|a| a == "--ratchet").and_then(|k| args.get(k + 1));
    match v {
        Some(p) => PathBuf::from(p),
        None => root.join("lens_coverage.json"),
    }
}

/// The files the positional arguments name (directories walked), or the
/// ratchet's modules when there are none.
fn targets(root: &Path, pos: &[String], ratchet: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    if pos.is_empty() {
        let text = super::read_opt(ratchet).ok_or_else(|| format!("no files named and no ratchet file at {}", ratchet.display()))?;
        // relative to the current directory when it is the root, so messages read short
        let here = std::fs::canonicalize(".").ok();
        for (name, _) in lens_cov::parse_ratchet(&text)? {
            out.push(if here.as_deref() == Some(root) { PathBuf::from(format!("{name}.viv")) } else { root.join(format!("{name}.viv")) });
        }
        return Ok(out);
    }
    for p in pos {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            walk(&pb, &mut out);
        } else {
            out.push(pb);
        }
    }
    Ok(out)
}

/// Coverage of each target, with dead and serves counts from `needed`.
fn covs(paths: &[PathBuf], n: &Needed) -> Result<Vec<ModCov>, String> {
    let mut out = Vec::new();
    for p in paths {
        let ps = p.display().to_string();
        if !p.exists() {
            return Err(format!("{ps}: no such file (a module the ratchet lists was removed or renamed? `viv lens coverage --update` records the modules as they are)"));
        }
        let l = viv::lens::load(&super::lens_input(&ps)?)?;
        let mut c = lens_cov::coverage(&l);
        lens_cov::add_needed(&mut c, n);
        out.push(c);
    }
    Ok(out)
}

/// Positional arguments after the verb, flags and their values left out.
fn positional(args: &[String], skip: usize) -> Vec<String> {
    let mut pos = Vec::new();
    let mut k = skip;
    while k < args.len() {
        let a = &args[k];
        if a == "--ratchet" || a == "--program" || a == "--limit" || a == "--run" {
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

fn fail(m: String) -> ExitCode {
    eprintln!("{m}");
    ExitCode::FAILURE
}

/// `viv lens coverage`.
pub fn coverage(args: &[String]) -> ExitCode {
    let flag = |f: &str| args.iter().any(|a| a == f);
    let root = repo_root();
    let rp = ratchet_path(&root, args);
    let pos = positional(args, 3);
    let paths = match targets(&root, &pos, &rp) {
        Ok(p) => p,
        Err(e) => return fail(format!("{e}\n{USAGE}")),
    };
    let files = repo_files(&root);
    let n = lens_cov::needed(&files, &known_of(&root, &files), &stories(&root, &files));
    let cs = match covs(&paths, &n) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    if flag("--json") {
        println!("{}", lens_cov::coverage_json(&cs));
    } else {
        print!("{}", lens_cov::coverage_text(&cs));
    }
    let recorded = super::read_opt(&rp).map(|t| lens_cov::parse_ratchet(&t)).transpose();
    let recorded = match recorded {
        Ok(r) => r.unwrap_or_default(),
        Err(e) => return fail(format!("{}: {e}", rp.display())),
    };
    if flag("--update") {
        let mut all = recorded;
        for c in &cs {
            let r = lens_cov::Recorded::of(c);
            match all.iter_mut().find(|(n, _)| n == &c.name) {
                Some(x) => x.1 = r,
                None => all.push((c.name.clone(), r)),
            }
        }
        if let Err(e) = super::write_file(&rp.display().to_string(), &lens_cov::ratchet_text(&all)) {
            return fail(e);
        }
        eprintln!("recorded {} modules in {}", cs.len(), rp.display());
        return ExitCode::SUCCESS;
    }
    if flag("--check") {
        let mut drops = Vec::new();
        let mut gains = Vec::new();
        for c in &cs {
            match recorded.iter().find(|(n, _)| n == &c.name) {
                Some((_, was)) => {
                    let (d, g) = lens_cov::ratchet_compare(&c.name, &lens_cov::Recorded::of(c), was);
                    drops.extend(d);
                    gains.extend(g);
                }
                None => gains.push(format!("{}: not in {} yet", c.name, rp.display())),
            }
        }
        for g in &gains {
            eprintln!("ratchet: {g}");
        }
        if !gains.is_empty() {
            eprintln!("ratchet: `viv lens coverage --update` records the new numbers");
        }
        if !drops.is_empty() {
            for d in &drops {
                eprintln!("ratchet: FAIL {d}");
            }
            let bad: HashSet<&str> = drops.iter().map(|d| d.split(':').next().unwrap_or("")).collect();
            for c in cs.iter().filter(|c| bad.contains(c.name.as_str())) {
                for i in c.items.iter().filter(|i| matches!(i.kind, "stale" | "unstamped" | "dead" | "serves")) {
                    eprintln!("  {}:{}: {}: {}", i.path, i.line, i.kind, i.message);
                }
            }
            return ExitCode::FAILURE;
        }
        eprintln!("ratchet: {} modules hold their coverage", cs.len());
    }
    ExitCode::SUCCESS
}

/// `viv lens needed`.
pub fn needed(args: &[String]) -> ExitCode {
    let flag = |f: &str| args.iter().any(|a| a == f);
    let root = repo_root();
    let files = repo_files(&root);
    if let Some(prog) = args.iter().position(|a| a == "--program").and_then(|k| args.get(k + 1)) {
        let name = repo_name(&root, &PathBuf::from(prog));
        let (total, un) = lens_cov::unreached_by(&files, &name);
        if flag("--json") {
            let list: Vec<String> = un.iter().map(|(m, d)| format!("[\"{m}\",\"{d}\"]")).collect();
            println!("{{\"program\":\"{name}\",\"defined\":{total},\"reached\":{},\"unreached\":[{}]}}", total - un.len(), list.join(","));
        } else {
            println!("{name}: its modules define {total} defs; its top level and tests reach {} of them statically", total - un.len());
            let mut last = String::new();
            for (m, d) in &un {
                if m != &last {
                    print!("\n{m}:");
                    last = m.clone();
                }
                print!(" {d}");
            }
            println!();
        }
        return ExitCode::SUCCESS;
    }
    let rp = ratchet_path(&root, args);
    let pos = positional(args, 3);
    let paths = match targets(&root, &pos, &rp) {
        Ok(p) => p,
        Err(e) => return fail(format!("{e}\n{USAGE}")),
    };
    let n = lens_cov::needed(&files, &known_of(&root, &files), &stories(&root, &files));
    let names: Vec<String> = paths.iter().map(|p| repo_name(&root, p)).collect();
    let mut dead = 0;
    let mut bad = 0;
    let mut json = Vec::new();
    for m in &names {
        let ds = n.dead_in(m);
        let bs = n.bad_serves_in(m);
        dead += ds.len();
        bad += bs.len();
        if flag("--json") {
            json.push(format!(
                "{{\"module\":\"{m}\",\"dead\":[{}],\"serves\":[{}]}}",
                ds.iter().map(|d| format!("{{\"def\":\"{}\",\"path\":\"{}\",\"line\":{}}}", d.name, d.path, d.line)).collect::<Vec<_>>().join(","),
                bs.iter().map(|(d, k, t)| format!("{{\"def\":\"{}\",\"kind\":\"{k}\",\"target\":{},\"line\":{}}}", d.name, viv::debug_lens::json_str(t), d.line)).collect::<Vec<_>>().join(",")
            ));
        } else {
            for d in ds {
                println!("{}:{}: dead: `{}` is reached from no program, test or entry point in the repository", d.path, d.line, d.name);
            }
            for (d, k, t) in bs {
                println!("{}:{}: serves: `{}` serves [{k} \"{t}\"], which does not exist", d.path, d.line, d.name);
            }
        }
    }
    if flag("--json") {
        println!("{{\"modules\":[{}],\"dead\":{dead},\"serves\":{bad}}}", json.join(","));
    } else {
        if flag("--verbose") {
            for (p, e) in &n.unparsed {
                eprintln!("note: {p} does not parse, so its calls are not counted: {e}");
            }
        } else if !n.unparsed.is_empty() {
            eprintln!("note: {} files do not parse, so their calls are not counted (--verbose lists them)", n.unparsed.len());
        }
        println!("lens needed: {} modules, {dead} dead defs, {bad} `serves` of nothing", names.len());
    }
    if dead + bad == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// `viv debt`.
pub fn debt(args: &[String]) -> ExitCode {
    let flag = |f: &str| args.iter().any(|a| a == f);
    let root = repo_root();
    let rp = ratchet_path(&root, args);
    let pos = positional(args, 2);
    let paths = match targets(&root, &pos, &rp) {
        Ok(p) => p,
        Err(e) => return fail(format!("{e}\n{USAGE}")),
    };
    let files = repo_files(&root);
    let n = lens_cov::needed(&files, &known_of(&root, &files), &stories(&root, &files));
    let cs = match covs(&paths, &n) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    if flag("--json") {
        println!("{}", lens_cov::coverage_json(&cs));
    } else {
        let limit = args.iter().position(|a| a == "--limit").and_then(|k| args.get(k + 1)).and_then(|v| v.parse::<usize>().ok());
        // `--run <program>`: run it once in the interpreter and put the defs
        // it calls first, the most called first
        let run = match args.iter().position(|a| a == "--run").and_then(|k| args.get(k + 1)) {
            Some(p) => match run_graph(p) {
                Ok(r) => Some(r),
                Err(e) => return fail(format!("viv debt --run {p}: {e}")),
            },
            None => None,
        };
        print!("{}", lens_cov::debt_text(&cs, &n, limit, run.as_ref()));
    }
    ExitCode::SUCCESS
}

/// Run a program in the interpreter (debug mode, as `viv run` does) and
/// return each def's call count.
fn run_graph(path: &str) -> Result<std::collections::HashMap<String, u64>, String> {
    let src = crate::TracedIo.read_to_string(&PathBuf::from(path)).map_err(|e| format!("cannot read {path}: {e}"))?;
    let expanded = super::expand_uses_from(&src, path)?;
    let h = std::thread::Builder::new().stack_size(super::INTERP_STACK).spawn(move || viv::interp::run(&expanded)).map_err(|e| e.to_string())?;
    let out = h.join().map_err(|_| "the interpreter panicked".to_string())??;
    Ok(lens_cov::run_calls(&out))
}
