// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The feature catalog (crate::features, `viv features`) stays true:
//!
//! - it builds with no problem: every entry has its fields, every `see:`
//!   names an entry, every example file is an entry's and every entry's
//!   example exists;
//! - every hand-written example prints its recorded output (`.out`) in the
//!   interpreter, or for a verb's example (`.cmd`) the verb prints it, and
//!   every example program prints the same bytes as C and as wasm;
//! - every builtin's example (features/builtins.txt) runs clean in the
//!   interpreter and prints the same as C and as wasm;
//! - each generated list matches the source that defines what it lists: the
//!   builtins the compiler's tables know, the verbs main.rs dispatches, the
//!   views `viv view --lens` takes, the keywords of the lexer and the parser,
//!   and features/modules.txt against the module headers;
//! - the CLI prints the overview, an entry, a search and JSON, and
//!   `--usage` counts uses and finds the hand-written patterns.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use viv::features::{self, Detector};

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn examples_dir(interp_only: bool) -> PathBuf {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("features/examples");
    if interp_only { d.join("interp") } else { d }
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_features_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_catalog_builds_with_no_problem() {
    let c = features::catalog();
    assert!(c.errors.is_empty(), "the catalog's problems:\n{}", c.errors.join("\n"));
    let hand = c.entries.iter().filter(|e| e.hand()).count();
    assert!(hand >= 100, "{hand} hand-written entries");
    // every area has entries
    for (key, _, _) in features::AREAS {
        assert!(c.entries.iter().any(|e| e.area == *key), "no entry in area {key}");
    }
    // every example file is an entry's example
    let used: HashSet<&str> = c.entries.iter().filter(|e| e.hand()).flat_map(|e| e.examples.iter().map(|x| x.name.as_str())).collect();
    for f in features::EXAMPLE_FILES {
        assert!(used.contains(f.name), "features/examples/{}.viv is no entry's example", f.name);
        assert!(f.out.is_some(), "features/examples/{}.viv has no .out", f.name);
    }
    // a hand-written entry's source files exist
    for e in c.entries.iter().filter(|e| e.hand()) {
        for d in &e.doc {
            let path = d.split(" section").next().unwrap().split(" (").next().unwrap().trim();
            assert!(root().join(path).exists(), "{} `{}`: doc {path} does not exist", e.source, e.id);
        }
    }
}

/// `sh -c` a command in `dir`, both streams as the shell interleaves them.
fn sh(dir: &Path, cmd: &str) -> (i32, String) {
    let o = Command::new("sh").current_dir(dir).arg("-c").arg(format!("{cmd} 2>&1")).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned())
}

/// What tools/three-engines.sh compares: the output without the diagnostic
/// lines and the failure summaries.
fn comparable(out: &str) -> String {
    out.lines()
        .filter(|l| !(l.starts_with("__graph__") || l.starts_with("__trace__") || l.ends_with("assertions FAILED") || l.ends_with("assertion FAILED") || l.ends_with("holds FAILED") || l.ends_with("hold FAILED")))
        .map(|l| format!("{l}\n"))
        .collect()
}

#[test]
fn every_hand_example_prints_its_recorded_output() {
    let mut bad = Vec::new();
    for f in features::EXAMPLE_FILES {
        let dir = examples_dir(f.interp_only);
        let (cmd, want_exit) = features::cmd_of(f);
        let args = cmd.unwrap_or_else(|| vec!["run".to_string(), "-q".to_string(), format!("{}.viv", f.name)]);
        let o = Command::new(viv_bin()).current_dir(&dir).args(&args).output().unwrap();
        let out = String::from_utf8_lossy(&o.stdout).into_owned();
        let code = o.status.code().unwrap_or(-1);
        if code != want_exit {
            bad.push(format!("{}: `viv {}` exited {code}, not {want_exit}:\n{}", f.name, args.join(" "), String::from_utf8_lossy(&o.stderr)));
        }
        if Some(out.as_str()) != f.out {
            bad.push(format!("{}: `viv {}` printed\n{out}\nand {}.out holds\n{}", f.name, args.join(" "), f.name, f.out.unwrap_or("")));
        }
        if out.contains("/Users/") || out.contains("/home/") || out.contains("/tmp/") {
            bad.push(format!("{}: the output holds an absolute path", f.name));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n----\n"));
}

/// The interpreter's, C's and the direct wasm backend's output of a program
/// in `dir`, compared as tools/three-engines.sh compares them. None for an
/// engine this machine cannot run.
fn three(dir: &Path, name: &str) -> Result<(), String> {
    let viv = viv_bin();
    let (_, i) = sh(dir, &format!("'{viv}' run '{name}.viv'"));
    let interp = comparable(&i);
    let tmp = scratch(&format!("3e_{}", name.replace(|c: char| !c.is_ascii_alphanumeric(), "_")));
    if Command::new(cc()).arg("--version").output().is_ok() {
        let c = tmp.join(format!("{name}.c"));
        let (code, err) = sh(dir, &format!("'{viv}' '{name}.viv' -o '{}'", c.display()));
        if code != 0 {
            return Err(format!("{name}: does not compile to C:\n{err}"));
        }
        let exe = tmp.join(name);
        let (code, err) = sh(dir, &format!("{} -std=c99 -O0 -o '{}' '{}' -lm", cc(), exe.display(), c.display()));
        if code != 0 {
            return Err(format!("{name}: the C does not build:\n{err}"));
        }
        let (_, n) = sh(dir, &format!("'{}'", exe.display()));
        if comparable(&n) != interp {
            return Err(format!("{name}: C prints\n{}\nthe interpreter\n{interp}", comparable(&n)));
        }
    }
    let wasm_rt = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let has_rt = std::fs::read(&wasm_rt).map(|b| b.windows(12).any(|w| w == b"vrt_getslice")).unwrap_or(false);
    if has_rt && Command::new("node").arg("--version").output().is_ok() {
        let w = tmp.join(format!("{name}.wasm"));
        let (code, err) = sh(dir, &format!("'{viv}' '{name}.viv' -o '{}'", w.display()));
        if code != 0 {
            return Err(format!("{name}: does not compile to wasm:\n{err}"));
        }
        let (_, d) = sh(dir, &format!("VIV_WASM='{}' node '{}' '{}'", wasm_rt.display(), root().join("tools/run-wasm.mjs").display(), w.display()));
        if comparable(&d) != interp {
            return Err(format!("{name}: wasm prints\n{}\nthe interpreter\n{interp}", comparable(&d)));
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

/// Run `jobs` on up to four threads; the errors they return.
fn parallel<T: Sync>(items: &[T], job: impl Fn(&T) -> Result<(), String> + Sync) -> Vec<String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let errs = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(it) = items.get(k) else { break };
                if let Err(e) = job(it) {
                    errs.lock().unwrap().push(e);
                }
            });
        }
    });
    errs.into_inner().unwrap()
}

/// A Viv name is emitted as a C identifier, so every name C or its headers
/// already take must be escaped: `I` is complex.h's imaginary-unit macro (the
/// builtin probes' `viv_arr I = ...` did not compile), `complex` its type
/// macro, `errno` a macro, `main` the entry point, `int` and `double` keywords.
/// The program uses a complex scalar so that complex.h is included.
#[test]
fn names_c_takes_are_escaped_and_print_the_same_in_three_engines() {
    let src = r#"def main(double) {
    return double * 2.0
}
def scale(I, complex) {
    errno = I * complex
    return errno
}
z = complex(1.0, 2.0)
I = zeros(3)
I[1] = 2.0
int = 4.0
errno = 5.0
double = main(int)
complex = scale(errno, 3.0)
real = re(z)
imag = im(z)
show I
show int
show errno
show double
show complex
show real + imag
"#;
    let c = viv::compile_c(src).expect("compiles to C");
    assert!(c.contains("#include <complex.h>"), "the program should pull in complex.h");
    for decl in ["viv_arr I_", "double int_", "double errno_", "double double_", "double complex_", "main_("] {
        assert!(c.contains(decl), "`{decl}` not in the C");
    }
    let dir = scratch("c_names");
    std::fs::write(dir.join("c_names.viv"), src).unwrap();
    three(&dir, "c_names").unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_hand_example_prints_the_same_in_three_engines() {
    let files: Vec<&features::ExampleFile> = features::EXAMPLE_FILES.iter().filter(|f| !f.interp_only).collect();
    let errs = parallel(&files, |f| three(&examples_dir(false), f.name));
    assert!(errs.is_empty(), "{}", errs.join("\n----\n"));
}

/// A probe program as a def of its own, so many run in one program: its
/// lines indented into `def probe_<k>() { ... }`.
fn as_def(k: usize, probe: &str) -> String {
    let body: String = probe.lines().map(|l| format!("    {l}\n")).collect();
    format!("def feature_probe_{k}() {{\n{body}}}\nfeature_probe_{k}()\n")
}

#[test]
fn every_builtin_example_runs_and_prints_the_same_in_three_engines() {
    let rows = features::builtin_rows();
    assert!(rows.len() >= 230, "{} rows", rows.len());
    let mut bad = Vec::new();
    for r in &rows {
        match viv::interp::run(&r.probe) {
            Ok(out) if !out.lines().any(|l| l.starts_with("error")) => {}
            Ok(out) => bad.push(format!("{}: {}\n{out}", r.name, r.probe)),
            Err(e) => bad.push(format!("{}: {}\nerror: {e}", r.name, r.probe)),
        }
    }
    assert!(bad.is_empty(), "builtin examples that do not run:\n{}", bad.join("\n----\n"));
    // the probes that declare nothing run as defs of one program; the rest each on its own
    let dir = scratch("probes");
    let (mut one, mut alone) = (String::new(), Vec::new());
    for (k, r) in rows.iter().enumerate() {
        if r.probe.lines().any(|l| l.starts_with("def ") || l.starts_with("record ") || l.starts_with("enum ") || l.starts_with("#!")) {
            let name = format!("probe_{k}");
            std::fs::write(dir.join(format!("{name}.viv")), format!("{}\n", r.probe)).unwrap();
            alone.push(name);
        } else {
            one.push_str(&as_def(k, &r.probe));
        }
    }
    std::fs::write(dir.join("probes.viv"), one).unwrap();
    alone.push("probes".to_string());
    let errs = parallel(&alone, |n| three(&dir, n));
    assert!(errs.is_empty(), "{}", errs.join("\n----\n"));
}

/// The builtins the interpreter dispatches by name in interp.rs (as
/// crate::builtins's own test reads them).
fn interp_names() -> Vec<String> {
    let src = include_str!("../src/interp.rs");
    let mut out = Vec::new();
    for (k, _) in src.match_indices("(\"") {
        let rest = &src[k + 2..];
        let Some(end) = rest.find('"') else { continue };
        let name = &rest[..end];
        let after = &rest[end + 1..];
        if name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') && !name.is_empty() && after.starts_with(", ") && after[2..].starts_with(|c: char| c.is_ascii_digit() || c == '_') {
            out.push(name.to_string());
        }
    }
    for (k, _) in src.match_indices("callee == \"") {
        let rest = &src[k + 11..];
        if let Some(end) = rest.find('"') {
            out.push(rest[..end].to_string());
        }
    }
    // names the compiler matches only to refuse them with a hint, not builtins:
    // `range` (Viv has none; crate::hints::RANGE points at `for i to n`)
    out.retain(|n| !viv::hints::REFUSED_NAMES.contains(&n.as_str()));
    out
}

#[test]
fn every_builtin_the_compiler_knows_has_a_row() {
    use viv::builtins::{EFFECTS, PURE, READERS, WRITERS};
    let mut known: BTreeSet<String> = interp_names().into_iter().collect();
    known.extend(PURE.iter().chain(EFFECTS).chain(READERS).map(|s| s.to_string()));
    known.extend(WRITERS.iter().map(|(n, _, _)| n.to_string()));
    known.extend(viv::features::interp_module_builtins());
    known.retain(|n| !n.starts_with("__") && !n.starts_with("viv") && !features::KEYWORDS.contains(&n.as_str()));
    let mut documented: HashSet<String> = HashSet::new();
    for r in features::builtin_rows() {
        documented.extend(features::sig_names(&r.sig));
        documented.insert(r.name.split_whitespace().next().unwrap_or("").to_string());
    }
    let missing: Vec<&String> = known.iter().filter(|n| !documented.contains(*n)).collect();
    assert!(missing.is_empty(), "builtins with no row in lang/viv/features/builtins.txt: {missing:?}");
    assert!(known.len() > 180, "{} builtins", known.len());
}

/// Every verb main.rs dispatches on `args[1]`.
fn dispatched_verbs() -> BTreeSet<String> {
    let src = include_str!("../src/main.rs");
    let mut out = BTreeSet::new();
    let quoted = |s: &str| -> Vec<String> { s.split('"').skip(1).step_by(2).map(str::to_string).collect() };
    for line in src.lines() {
        let l = line.trim();
        if l.starts_with("if args.len() >= 2 && ") {
            let cond = &l["if args.len() >= 2 && ".len()..];
            if cond.starts_with("args[1] ==") || cond.starts_with("(args[1] ==") || cond.starts_with("matches!(args[1].as_str()") {
                // the verb names of the condition, before any other test
                let head = cond.split("&& args.iter()").next().unwrap();
                out.extend(quoted(head));
            }
        }
    }
    out
}

#[test]
fn the_verbs_are_the_verbs_main_dispatches() {
    let listed: BTreeSet<String> = features::VERBS.iter().map(|(n, _, _)| n.to_string()).filter(|n| n != "compile").collect();
    let found = dispatched_verbs();
    assert!(found.len() > 20, "{found:?}");
    assert_eq!(listed, found, "crate::features::VERBS against main.rs");
}

#[test]
fn the_view_lenses_are_the_ones_viv_view_takes() {
    let src = include_str!("../src/main.rs");
    let start = src.find("let text = match lens.as_str() {").expect("the view's match");
    let end = start + src[start..].find("other => return fail").expect("its last arm");
    let mut found = BTreeSet::new();
    for l in src[start..end].lines() {
        let t = l.trim();
        if t.starts_with('"') {
            if let Some(name) = t.split('"').nth(1) {
                found.insert(name.to_string());
            }
        }
    }
    let listed: BTreeSet<String> = features::LENSES.iter().filter(|(_, k, _)| *k == "view").map(|(n, _, _)| n.to_string()).collect();
    assert_eq!(listed, found, "crate::features::LENSES against `viv view --lens`");
    // the lens files are the extensions main.rs reads beside a module
    // (arch_cli finds `.arch` files by walking the tree for the suffix ".arch")
    for (name, kind, _) in features::LENSES.iter().filter(|(_, k, _)| *k == "file") {
        let _ = kind;
        let ext = name.trim_start_matches('.');
        let read = src.contains(&format!("with_extension(\"{ext}\")")) || include_str!("../src/arch_cli.rs").contains(&format!("\"{ext}\"")) || include_str!("../src/arch_cli.rs").contains(&format!("\".{ext}\"")) || include_str!("../src/lens_cli.rs").contains(&format!("\"{ext}\""));
        assert!(read, "no tool reads the `{name}` file");
    }
}

#[test]
fn the_keywords_are_the_lexer_s_and_the_parser_s() {
    let lexer = include_str!("../src/lexer.rs");
    let parser = include_str!("../src/parser.rs");
    let mut found = BTreeSet::new();
    for l in lexer.lines() {
        let t = l.trim();
        if let Some((word, rest)) = t.strip_prefix('"').and_then(|r| r.split_once('"')) {
            if rest.trim_start().starts_with("=> Tok::") && word.chars().all(|c| c.is_ascii_lowercase()) {
                found.insert(word.to_string());
            }
        }
    }
    // the soft keywords: `at_word(k, "w")` and `Tok::Ident(s)) if s == "w"`
    for pat in ["at_word(0, \"", "at_word(1, \"", "at_word(2, \"", "if s == \"", "if e == \""] {
        for (k, _) in parser.match_indices(pat) {
            let rest = &parser[k + pat.len()..];
            if let Some(end) = rest.find('"') {
                found.insert(rest[..end].to_string());
            }
        }
    }
    let listed: BTreeSet<String> = features::KEYWORDS.iter().map(|s| s.to_string()).collect();
    let missing: Vec<&String> = found.difference(&listed).collect();
    assert!(missing.is_empty(), "keywords the lexer or parser reads that crate::features::KEYWORDS does not list: {missing:?}");
    // and each is documented by a hand-written entry
    let c = features::catalog();
    for k in features::KEYWORDS {
        assert!(c.entries.iter().any(|e| e.hand() && e.keywords.iter().any(|w| w == k)), "no hand-written entry lists the keyword `{k}` in its keywords");
    }
}

#[test]
fn modules_txt_is_written_from_the_module_headers_now() {
    let mods = features::read_modules(&root());
    assert!(mods.len() > 100, "{} modules", mods.len());
    let now = features::modules_text(&mods);
    assert!(now == features::catalog_text("modules"), "lang/viv/features/modules.txt is stale: run `viv features --write-modules` in the repository");
}

fn viv(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(viv_bin()).current_dir(root()).args(args).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn the_cli_prints_the_overview_an_entry_a_search_and_json() {
    let (code, out, err) = viv(&["features"]);
    assert_eq!(code, 0, "{err}");
    assert!(err.is_empty(), "the catalog reports problems:\n{err}");
    for h in ["Syntax (", "Autodiff (", "Builtins (", "Phrases (", "CLI verbs (", "Modules ("] {
        assert!(out.contains(h), "the overview has no `{h}`:\n{out}");
    }
    // a hand-written entry: what, when, the example and its output, the source
    let (_, out, _) = viv(&["features", "list-comprehension"]);
    for h in ["What it does", "When to use it", "Example", "Output", "Source"] {
        assert!(out.contains(h), "{h}:\n{out}");
    }
    // a builtin by its name, with its example run now
    let (_, out, _) = viv(&["features", "argsort"]);
    assert!(out.contains("argsort(") && out.contains("Output (the interpreter, now)"), "{out}");
    // an area
    let (_, out, _) = viv(&["features", "autodiff"]);
    assert!(out.contains("value-and-grad"), "{out}");
    // a verb by its name
    let (_, out, _) = viv(&["features", "lint"]);
    assert!(out.contains("viv lint"), "{out}");
    // a search
    let (_, out, _) = viv(&["features", "--search", "integral"]);
    assert!(out.contains("phrases-"), "{out}");
    // json parses, and holds every entry
    let (_, out, _) = viv(&["features", "--json", "--max-lines", "0"]);
    let env = viv::story::parse_json(&out).expect("viv features --json is JSON");
    // the envelope: the catalog is its result
    let j = env.get("result").cloned().expect("an envelope with a result");
    let n = match j.get("entries") {
        Some(viv::story::Json::Arr(v)) => v.len(),
        _ => 0,
    };
    assert_eq!(n, features::catalog().entries.len());
}

/// B350: the reference page credited `viv features` with a count it had
/// filtered (a local-only entry left out), so the page said 181 where the CLI
/// says 182. The page now prints the catalog's own count from features.json;
/// this holds the CLI, the data file and the built page to one number.
#[test]
fn the_reference_page_states_the_count_viv_features_prints() {
    let (code, out, err) = viv(&["features"]);
    assert_eq!(code, 0, "{err}");
    let hand: usize = out
        .split(" written by hand")
        .next()
        .and_then(|s| s.rsplit(' ').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no `N written by hand` in:\n{out}"));
    let data = std::fs::read_to_string(root().join("www/src/data/features.json")).expect("www/src/data/features.json");
    let j = viv::story::parse_json(&data).expect("features.json is JSON");
    let in_data = j.get("hand").and_then(|v| v.num()).expect("features.json has a number `hand`") as usize;
    assert_eq!(in_data, hand, "www/src/data/features.json is stale: run `node www/scripts/features-build.mjs`");
    let page = std::fs::read_to_string(root().join("www/src/pages/reference.astro")).expect("reference.astro");
    assert!(page.contains("{features.hand} features written up in"), "reference.astro must print the catalog's count, features.hand");
    // the built page, when there is one
    if let Ok(html) = std::fs::read_to_string(root().join("www/dist/reference/index.html")) {
        let txt = html.replace("<!-- -->", "");
        let shown: Option<usize> = txt
            .split(" features written up in")
            .next()
            .and_then(|s| s.rsplit(|c: char| !c.is_ascii_digit()).next())
            .and_then(|s| s.parse().ok());
        assert_eq!(shown, Some(hand), "www/dist/reference/index.html states a different count; rebuild the site");
    }
}

#[test]
fn usage_counts_the_forms_and_finds_the_patterns() {
    let src = "xs = [1.0, 2.0, 3.0]\nys = [x * x for x in xs if x > 1.0]\nlast = xs[-1]\ns = 0.0\nfor i to len(xs) {\n    s = s + xs[i]\n}\nb = zeros(3)\nfor i to len(xs) {\n    b[i] = 2.0 * xs[i]\n}\nk = 0\nwhile k < 3 {\n    print(k)\n    k = k + 1\n}\nbig = 0.0 if s < 1.0 else s\nn = xs[round(1.2)]\nprint(n)\n";
    let s = viv::features_usage::scan(src);
    assert_eq!(s.problem, None);
    let form = |f: &str| s.forms.get(f).copied().unwrap_or(0);
    assert_eq!(form("comprehension"), 1);
    assert_eq!(form("negative-index"), 1);
    assert_eq!(form("if-expr"), 1, "the comprehension's filter is not an if-expression");
    assert_eq!(form("while-loop"), 1);
    assert_eq!(s.calls.get("zeros"), Some(&1));
    assert_eq!(s.count(&Detector::Call("print".to_string()), src), 2);
    let pats: Vec<&str> = s.findings.iter().map(|f| f.pattern).collect();
    for p in ["reduction-loop", "fill-loop", "while-counter", "round-index"] {
        assert!(pats.contains(&p), "{p} not found: {pats:?}");
    }
    // the comprehension's own loop is not reported
    assert!(!pats.contains(&"append-loop"), "{pats:?}");
    // a hand sort and a hand argmax
    let src2 = "def bsort(a) {\n    for i to len(a) {\n        for j to len(a) - 1 {\n            if a[j] > a[j + 1] {\n                t = a[j]\n                a[j] = a[j + 1]\n                a[j + 1] = t\n            }\n        }\n    }\n}\ndef best(a) {\n    m = a[0]\n    for i to len(a) {\n        if a[i] > m {\n            m = a[i]\n        }\n    }\n    return m\n}\n";
    let s2 = viv::features_usage::scan(src2);
    let pats: Vec<&str> = s2.findings.iter().map(|f| f.pattern).collect();
    assert!(pats.contains(&"argmax-loop"), "{pats:?}");
    // the swap sits under an `if` inside the inner loop
    assert!(pats.contains(&"swap-sort"), "{pats:?}");
    // `viv features --usage` over the examples finds the comprehension example's comprehension
    let (code, out, err) = viv(&["features", "--usage", "lang/viv/features/examples"]);
    assert_eq!(code, 0, "{err}");
    let line = out.lines().find(|l| l.trim_start().starts_with("list-comprehension")).unwrap_or("");
    assert!(!line.contains("NEVER USED"), "{line}\n{out}");
}

/// The lexer writes `step a: stmt` as the braced step before anything reads
/// the tokens, so `--usage` must tell the two forms apart by the source text
/// under the `{` (it once counted no one-line step anywhere, though lib/gpt
/// is written in them).
#[test]
fn usage_counts_one_line_steps_and_braced_steps_apart() {
    let src = "def f(x) {\n    step double: y = 2.0 * x\n    step half \"halve it\": z = y / 2.0\n    step done {\n        w = z\n    }\n    return w\n}\nshow f(1.0)\n";
    let s = viv::features_usage::scan(src);
    assert_eq!(s.problem, None);
    assert_eq!(s.forms.get("one-line-step").copied(), Some(2), "{:?}", s.forms);
    assert_eq!(s.forms.get("step-block").copied(), Some(1), "{:?}", s.forms);
}
