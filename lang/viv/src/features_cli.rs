// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv features`: the catalog (viv::features) from the command line, and
//! `--usage`, how much of it a corpus uses (viv::features_usage).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use viv::features::{self, Detector, Entry};
use viv::features_usage::{self, Finding, Scan, PATTERNS};

/// The corpus `--usage` reads when no path is given, under the repository root.
pub const CORPUS: &[&str] = &["lib", "posts", "study", "scf", "dft", "pde", "engines"];

/// Directories a walk skips: build output, dependencies, the site's copies.
const SKIP: &[&str] = &["target", "node_modules", ".git", "dist", "public", ".astro", "www"];

const USAGE: &str = "usage: viv features                       every area and its entries
       viv features <name or area>         one entry: what it does, when to use it, an example and its output
       viv features --search <words>       the entries that mention all the words, best first
       viv features --json [--run]         the whole catalog (--run: builtin and phrase examples with their output)
       viv features --usage [<path>...] [--json] [--examples <n>]
                                           how often each feature is used, and code written by hand that a feature does;
                                           with no path: lib posts study scf dft pde engines
       viv features --write-modules        rewrite lang/viv/features/modules.txt from the module headers";

/// The repository root: the nearest ancestor of the current directory that
/// holds Cargo.toml and lib/.
fn repo_root() -> Option<PathBuf> {
    let cwd = std::fs::canonicalize(".").ok()?;
    cwd.ancestors().find(|a| a.join("Cargo.toml").exists() && a.join("lib").is_dir()).map(Path::to_path_buf)
}

pub fn cli(args: &[String]) -> ExitCode {
    let rest: Vec<&str> = args.iter().skip(2).map(String::as_str).collect();
    let c = features::catalog();
    if !c.errors.is_empty() {
        eprintln!("viv features: the catalog has {} problem(s):", c.errors.len());
        for e in &c.errors {
            eprintln!("  {e}");
        }
    }
    match rest.first().copied() {
        None => {
            print!("{}", features::overview());
            ExitCode::SUCCESS
        }
        Some("-h" | "--help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("--json") => {
            println!("{}", features::json(rest.contains(&"--run")));
            ExitCode::SUCCESS
        }
        Some("--search") if rest.len() > 1 => {
            print!("{}", features::search(&rest[1..].join(" ")));
            ExitCode::SUCCESS
        }
        Some("--usage") => usage(&rest[1..]),
        Some("--write-modules") => write_modules(),
        Some(a) if a.starts_with("--") => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
        Some(_) => {
            print!("{}", features::lookup(&rest.join(" ")));
            ExitCode::SUCCESS
        }
    }
}

// ---- --write-modules ---------------------------------------------------------------

fn write_modules() -> ExitCode {
    let Some(root) = repo_root() else {
        eprintln!("viv features --write-modules: run it inside the repository (no ancestor holds Cargo.toml and lib/)");
        return ExitCode::FAILURE;
    };
    let mods = features::read_modules(&root);
    let text = features::modules_text(&mods);
    let path = root.join("lang/viv/features/modules.txt");
    match std::fs::write(&path, text) {
        Ok(()) => {
            println!("wrote lang/viv/features/modules.txt: {} modules", mods.len());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("viv features --write-modules: {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

// ---- --usage -------------------------------------------------------------------------

fn walk(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_dir() {
        if p.file_name().is_some_and(|n| SKIP.contains(&n.to_string_lossy().as_ref())) {
            return;
        }
        let mut es: Vec<PathBuf> = std::fs::read_dir(p).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        es.sort();
        for e in es {
            walk(&e, out);
        }
    } else if p.extension().is_some_and(|x| x == "viv") {
        out.push(p.to_path_buf());
    }
}

/// One file of the corpus, scanned.
struct FileScan {
    shown: String,
    top: String,
    src: String,
    scan: Scan,
    /// the companion files beside it, by extension (`intent`, `debug`, ...)
    companions: Vec<String>,
}

fn scan_file(f: &Path, shown: String, top: String) -> Option<FileScan> {
    let src = std::fs::read_to_string(f).ok()?;
    // a Python program under `#! lang python` is py2viv's, not Viv text
    if viv::lang_of(&src).as_deref() == Some("python") {
        return None;
    }
    let mut scan = features_usage::scan(&src);
    let mut companions = Vec::new();
    for ext in ["intent", "debug", "memory", "arch"] {
        let p = f.with_extension(ext);
        if p.exists() {
            companions.push(ext.to_string());
            // the debug lens is Viv text too: its watches and holds count
            if ext == "debug" {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    let d = features_usage::scan(&t);
                    for (k, v) in d.words {
                        *scan.words.entry(k).or_insert(0) += v;
                    }
                    for (k, v) in d.calls {
                        *scan.calls.entry(k).or_insert(0) += v;
                    }
                    for (k, v) in d.forms {
                        *scan.forms.entry(k).or_insert(0) += v;
                    }
                }
            }
        }
    }
    // an `.arch` file is the directory's, not the module's
    if f.parent().is_some_and(|d| std::fs::read_dir(d).map(|r| r.flatten().any(|e| e.path().extension().is_some_and(|x| x == "arch"))).unwrap_or(false)) && !companions.contains(&"arch".to_string()) {
        companions.push("arch-dir".to_string());
    }
    Some(FileScan { shown, top, src, scan, companions })
}

fn count(e: &Entry, f: &FileScan) -> usize {
    e.usage
        .iter()
        .map(|d| match d {
            Detector::File(ext) => usize::from(f.companions.iter().any(|c| c == ext || (ext == "arch" && c == "arch-dir"))),
            d => f.scan.count(d, &f.src),
        })
        .sum()
}

struct Tally {
    uses: usize,
    files: usize,
    tops: BTreeMap<String, usize>,
}

fn usage(args: &[&str]) -> ExitCode {
    let json = args.contains(&"--json");
    let mut examples = 3usize;
    let mut paths: Vec<String> = Vec::new();
    let mut k = 0;
    while k < args.len() {
        match args[k] {
            "--json" => {}
            "--examples" => {
                examples = args.get(k + 1).and_then(|n| n.parse().ok()).unwrap_or(3);
                k += 1;
            }
            p => paths.push(p.to_string()),
        }
        k += 1;
    }
    let root = repo_root();
    let (bases, default): (Vec<(PathBuf, String)>, bool) = if paths.is_empty() {
        let Some(root) = root.clone() else {
            eprintln!("viv features --usage: with no path, run it inside the repository (no ancestor holds Cargo.toml and lib/)");
            return ExitCode::FAILURE;
        };
        (CORPUS.iter().map(|d| (root.join(d), d.to_string())).filter(|(p, _)| p.exists()).collect(), true)
    } else {
        (paths.iter().map(|p| (PathBuf::from(p), p.clone())).collect(), false)
    };
    let mut files: Vec<FileScan> = Vec::new();
    for (base, label) in &bases {
        let mut fs = Vec::new();
        walk(base, &mut fs);
        if !base.exists() {
            eprintln!("viv features --usage: {label}: no such file or directory");
            return ExitCode::FAILURE;
        }
        for f in fs {
            let shown = match (&root, std::fs::canonicalize(&f)) {
                (Some(r), Ok(abs)) => abs.strip_prefix(r).map(|p| p.display().to_string()).unwrap_or_else(|_| f.display().to_string()),
                _ => f.display().to_string(),
            };
            if let Some(s) = scan_file(&f, shown, label.clone()) {
                files.push(s);
            }
        }
    }
    let c = features::catalog();
    // per entry: uses, files, uses per corpus directory
    let mut tallies: Vec<(&Entry, Tally)> = Vec::new();
    for e in c.entries.iter().filter(|e| !e.usage.is_empty()) {
        let mut t = Tally { uses: 0, files: 0, tops: BTreeMap::new() };
        for f in &files {
            let n = count(e, f);
            if n > 0 {
                t.uses += n;
                t.files += 1;
                *t.tops.entry(f.top.clone()).or_insert(0) += n;
            }
        }
        tallies.push((e, t));
    }
    // the patterns
    let mut found: HashMap<&str, Vec<(&FileScan, &Finding)>> = HashMap::new();
    for f in &files {
        for x in &f.scan.findings {
            found.entry(x.pattern).or_default().push((f, x));
        }
    }
    let unparsed: Vec<&FileScan> = files.iter().filter(|f| f.scan.problem.is_some()).collect();
    if json {
        println!("{}", usage_json(&files, &tallies, &found, &unparsed));
        return ExitCode::SUCCESS;
    }
    let mut out = String::new();
    let mut per_top: Vec<(String, usize)> = Vec::new();
    for f in &files {
        match per_top.iter_mut().find(|(t, _)| *t == f.top) {
            Some(x) => x.1 += 1,
            None => per_top.push((f.top.clone(), 1)),
        }
    }
    let _ = default;
    out.push_str(&format!("viv features --usage: {} files ({})\n", files.len(), per_top.iter().map(|(t, n)| format!("{t} {n}")).collect::<Vec<_>>().join(", ")));
    out.push_str(&format!("{} do not parse on their own (their uses are counted from the tokens; the hand-written patterns need the parse)\n", unparsed.len()));

    out.push_str("\nHand-written features, fewest uses first (uses / files)\n");
    for area in features::HAND_AREAS {
        let mut rows: Vec<&(&Entry, Tally)> = tallies.iter().filter(|(e, _)| e.area == *area).collect();
        if rows.is_empty() {
            continue;
        }
        rows.sort_by(|a, b| a.1.uses.cmp(&b.1.uses).then(a.0.id.cmp(&b.0.id)));
        out.push_str(&format!("  {area}\n"));
        for (e, t) in rows {
            let tops = t.tops.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ");
            out.push_str(&format!("    {:<34} {:>6} / {:<4} {}\n", e.id, t.uses, t.files, if t.uses == 0 { "NEVER USED".to_string() } else { tops }));
        }
    }
    let unused = |area: &str| -> Vec<String> { tallies.iter().filter(|(e, t)| e.area == area && t.uses == 0).map(|(e, _)| e.id.clone()).collect() };
    let rare = |area: &str| -> Vec<String> {
        tallies.iter().filter(|(e, t)| e.area == area && t.uses > 0 && t.files <= 2).map(|(e, t)| format!("{} ({} in {})", e.id, t.uses, t.files)).collect()
    };
    for (area, what) in [("builtins", "Builtins"), ("phrases", "Phrase groups"), ("keywords", "Keywords"), ("lens-kinds", "Lens files")] {
        let total = tallies.iter().filter(|(e, _)| e.area == area).count();
        let u = unused(area);
        out.push_str(&format!("\n{what} never used: {} of {total}\n", u.len()));
        if !u.is_empty() {
            out.push_str(&wrap(&u.join(", "), "  "));
        }
        let r = rare(area);
        if !r.is_empty() {
            out.push_str(&format!("{what} used in one or two files: {}\n", r.len()));
            out.push_str(&wrap(&r.join(", "), "  "));
        }
    }

    out.push_str("\nWritten by hand where a feature does it (findings / files): what to use instead\n");
    let mut pats: Vec<(&str, &str, &str, usize)> = PATTERNS.iter().map(|(id, what, instead)| (*id, *what, *instead, found.get(id).map_or(0, |v| v.len()))).collect();
    pats.sort_by(|a, b| b.3.cmp(&a.3));
    for (id, what, instead, n) in pats {
        let hits = found.get(id).map(Vec::as_slice).unwrap_or(&[]);
        let nfiles = {
            let mut s: Vec<&str> = hits.iter().map(|(f, _)| f.shown.as_str()).collect();
            s.sort();
            s.dedup();
            s.len()
        };
        out.push_str(&format!("  {id:<16} {n:>5} / {nfiles:<4} {what}\n"));
        out.push_str(&format!("  {:<16} {:>12} use: {instead}\n", "", ""));
        if n == 0 {
            continue;
        }
        // the files with the most, then a few places
        let mut by_file: Vec<(&str, usize)> = Vec::new();
        for (f, _) in hits {
            match by_file.iter_mut().find(|(s, _)| *s == f.shown) {
                Some(x) => x.1 += 1,
                None => by_file.push((f.shown.as_str(), 1)),
            }
        }
        by_file.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        out.push_str(&format!(
            "  {:<16} {:>12} most in: {}\n",
            "",
            "",
            by_file.iter().take(4).map(|(s, n)| format!("{s} {n}")).collect::<Vec<_>>().join(", ")
        ));
        for (f, x) in hits.iter().take(examples) {
            let line = f.src.lines().nth(x.line - 1).unwrap_or("").trim();
            let line: String = if line.chars().count() > 80 { line.chars().take(77).collect::<String>() + "..." } else { line.to_string() };
            out.push_str(&format!("      {}:{}  {}   ({})\n", f.shown, x.line, line, x.what));
        }
    }
    print!("{out}");
    ExitCode::SUCCESS
}

fn wrap(text: &str, indent: &str) -> String {
    let mut out = String::new();
    let mut line = String::new();
    for w in text.split(", ") {
        if !line.is_empty() && indent.len() + line.len() + w.len() + 2 > 100 {
            out.push_str(&format!("{indent}{line},\n"));
            line.clear();
        }
        if !line.is_empty() {
            line.push_str(", ");
        }
        line.push_str(w);
    }
    if !line.is_empty() {
        out.push_str(&format!("{indent}{line}\n"));
    }
    out
}

fn usage_json(files: &[FileScan], tallies: &[(&Entry, Tally)], found: &HashMap<&str, Vec<(&FileScan, &Finding)>>, unparsed: &[&FileScan]) -> String {
    use viv::debug_lens::json_str;
    let entries = tallies
        .iter()
        .map(|(e, t)| {
            let tops = t.tops.iter().map(|(k, v)| format!("{}:{v}", json_str(k))).collect::<Vec<_>>().join(",");
            format!("{{\"id\":{},\"area\":{},\"uses\":{},\"files\":{},\"by_dir\":{{{tops}}}}}", json_str(&e.id), json_str(&e.area), t.uses, t.files)
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let pats = PATTERNS
        .iter()
        .map(|(id, what, instead)| {
            let hits = found.get(id).map(Vec::as_slice).unwrap_or(&[]);
            let places = hits.iter().map(|(f, x)| format!("{{\"file\":{},\"line\":{},\"what\":{}}}", json_str(&f.shown), x.line, json_str(&x.what))).collect::<Vec<_>>().join(",");
            format!("{{\"id\":{},\"what\":{},\"instead\":{},\"count\":{},\"places\":[{places}]}}", json_str(id), json_str(what), json_str(instead), hits.len())
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let unp = unparsed.iter().map(|f| format!("{{\"file\":{},\"problem\":{}}}", json_str(&f.shown), json_str(f.scan.problem.as_deref().unwrap_or("")))).collect::<Vec<_>>().join(",");
    format!("{{\"files\":{},\"unparsed\":[{unp}],\"entries\":[\n{entries}\n],\"patterns\":[\n{pats}\n]}}", files.len())
}
