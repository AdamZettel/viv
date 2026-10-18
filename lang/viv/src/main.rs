// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The Viv CLI: `viv <input.viv> [-o <output.c>]`.
//!
//! All I/O goes through the kit's injected `IoEnvironment` (1.8); diagnostics
//! are rendered from structured values (1.7).

use std::path::PathBuf;
use std::process::ExitCode;

use knotkit::backend::{Backend, CBackend};
use knotkit::diag::{PlainRenderer, Renderer};
use knotkit::io::{IoEnvironment, NativeIo};
use knotkit::span::SourceMap;

mod agent;
mod arch_cli;
mod db_cli;
mod fast_cli;
mod features_cli;
mod claims_cli;
mod lens_cli;
mod lint;

fn read_src(path: &str) -> Result<String, String> {
    load(path, false, "", &[]).map(|l| l.text).map_err(|e| e.text())
}

/// A program's `#! use <module>` lines, resolved against the file system: a
/// module named `dft/pwdft_it` is `dft/pwdft_it.viv` beside the program, or
/// under the repository root (the nearest ancestor holding `Cargo.toml`), or
/// under `lang/viv/examples` for `examples/<name>`.
fn expand_uses_from(src: &str, path: &str) -> Result<String, String> {
    let (here, root) = module_dirs(path);
    let mut files = vec![PathBuf::from(path)];
    let out = viv::expand_uses(src, &mut |name: &str| {
        let c = match module_path(&here, &root, name) {
            Ok(c) => c,
            Err(e) => return embedded_module(name).ok_or(e),
        };
        files.push(c.clone());
        TracedIo.read_to_string(&c).map_err(|e| format!("error: cannot read {}: {}", c.display(), e))
    });
    // the memory lens of the program and its modules (viv::memory_lens), as `load` reads it
    let mut lenses: Vec<(String, String)> = Vec::new();
    for f in &files {
        let lp = f.with_extension("memory");
        if let Some(t) = read_opt(&lp) {
            let shown = module_display(&root, &lp);
            if !lenses.iter().any(|(n, _)| *n == shown) {
                lenses.push((shown, t));
            }
        }
    }
    viv::memory_lens::set_loaded(lenses);
    out
}

/// A module the compiler carries: lib/cas (viv::cas_check::LIB), so a program
/// anywhere can `#! use lib/cas/cas`. A file of that name beside the program
/// or under its repository is read first.
fn embedded_module(name: &str) -> Option<String> {
    viv::cas_check::lib_module(name.strip_prefix('/').unwrap_or(name)).map(str::to_string)
}

/// The directory a program's modules are looked up from, and the repository
/// root: the nearest ancestor holding `Cargo.toml` and `lib/`, so a program
/// under lang/viv (whose own Cargo.toml is nearer) reads `#! use lib/...` from
/// the repository's lib/ as every other program does; else the nearest
/// ancestor holding `Cargo.toml`.
fn module_dirs(path: &str) -> (PathBuf, PathBuf) {
    // A bare file name (`viv run prog.viv`) has the parent `Some("")`, not
    // `None`: treat an empty parent as the current directory.
    let here = match PathBuf::from(path).parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let start = std::fs::canonicalize(&here).unwrap_or(here.clone());
    if let Some(repo) = start.ancestors().find(|a| a.join("Cargo.toml").exists() && a.join("lib").is_dir()) {
        return (here, repo.to_path_buf());
    }
    // outside a checkout: the root an installed compiler names (the Python
    // package vipyr sets it to the copy of lib/ it ships)
    if let Some(r) = std::env::var_os("VIV_ROOT").map(PathBuf::from).filter(|r| r.join("lib").is_dir()) {
        return (here, r);
    }
    let mut root = start;
    while !root.join("Cargo.toml").exists() {
        if !root.pop() { break; }
    }
    (here, root)
}

/// The file a `#! use <name>` line names: `<name>.viv` beside the program, or
/// under the repository root, or under `lang/viv/examples` for `examples/<name>`.
fn module_path(here: &std::path::Path, root: &std::path::Path, name: &str) -> Result<PathBuf, String> {
    // an import's candidates (viv::imports::candidates): `./a/b` is beside the
    // program only, `/a/b` under the repository root only
    if let Some(rest) = name.strip_prefix("./") {
        let c = here.join(format!("{rest}.viv"));
        return if !rest.contains("..") && viv::trace::exists(&c) { Ok(c) } else { Err(format!("no module {rest}.viv beside the program")) };
    }
    if let Some(rest) = name.strip_prefix('/') {
        let rel = if let Some(r) = rest.strip_prefix("examples/") { format!("lang/viv/examples/{r}") } else { rest.to_string() };
        let c = root.join(format!("{rel}.viv"));
        return if !rest.contains("..") && viv::trace::exists(&c) { Ok(c) } else { Err(format!("no module {rest}.viv under {}", root.display())) };
    }
    if name.contains("..") {
        return Err(format!("`#! use {name}`: a module name is a path under the repository, like dft/pwdft_it"));
    }
    let rel = if let Some(rest) = name.strip_prefix("examples/") { format!("lang/viv/examples/{rest}") } else { name.to_string() };
    let candidates = [here.join(format!("{name}.viv")), root.join(format!("{rel}.viv")), root.join(format!("{name}.viv"))];
    for c in candidates {
        if viv::trace::exists(&c) {
            return Ok(c);
        }
    }
    Err(format!("`#! use {name}`: no module by that name (looked for {name}.viv beside the program and under {})", root.display()))
}

/// `program_graph("m")` (viv::graphlit) reads a module as `#! use m` would
/// from a program in `here`.
fn install_resolver(here: &std::path::Path, root: &std::path::Path) {
    let (here, root) = (here.to_path_buf(), root.to_path_buf());
    viv::set_module_resolver(Some(std::sync::Arc::new(move |name: &str| {
        let c = module_path(&here, &root, name)?;
        TracedIo.read_to_string(&c).map_err(|e| format!("cannot read {}: {}", c.display(), e))
    })));
}

/// The name a program goes by in its diagnostics, in both engines. The
/// compiled program carries its run-time error locations in the C
/// (`viv::lower::with_sites`), and that C is committed by hash and compiled
/// into served wasm, so the name must not depend on the machine or on the
/// directory the compiler ran in: a relative path stays as given, and an
/// absolute path becomes its file name. (Relative to the current directory
/// would still depend on where the build ran: from `/`, every path is under
/// it.) `viv run` names the file the same way, so both engines print the same
/// location.
fn display_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if !p.is_absolute() {
        return path.to_string();
    }
    p.file_name().map_or_else(|| path.to_string(), |f| f.to_string_lossy().into_owned())
}

/// A program as the engines run it: `read_src`, plus the debug lens of the
/// program and of every module it uses, from their `.debug` files, appended as
/// `debug` blocks. A `.debug` file that does
/// not parse is an error naming its own lines. The doc and lens tools read
/// `read_src`, which leaves the debug lens out.
fn read_program(path: &str) -> Result<String, String> {
    load(path, true, "", &[]).map(|l| l.text).map_err(|e| e.text())
}

/// A program loaded for the engines: the text they parse, and the source map
/// that says where each part of it came from.
struct Loaded {
    /// The program as the engines see it: preludes, the program with its
    /// `#! use` modules spliced in (a Python program translated first), and
    /// the debug lens. It is source 0 of `sources`.
    text: String,
    /// Source 0 is `text`. When the program uses a module or is written in
    /// Python, the files it came from follow, and source 0 is a composite of
    /// them (knotkit's `Origin`), so a diagnostic is shown in the module or
    /// the Python file it is about.
    sources: SourceMap,
}

/// Why a program could not be loaded: a message with no location (a file
/// that cannot be read), or diagnostics located in the sources (py2viv's
/// refusals, in the Python file).
enum LoadError {
    Plain(String),
    Located(Vec<knotkit::diag::Diagnostic>, SourceMap),
    /// A `--set` that does not apply, in `apply_sets`'s words.
    Set(String),
}

impl LoadError {
    fn text(&self) -> String {
        match self {
            LoadError::Plain(m) | LoadError::Set(m) => m.clone(),
            LoadError::Located(ds, sources) => ds.iter().map(|d| PlainRenderer.render(d, sources)).collect::<String>().trim_end().to_string(),
        }
    }

    /// Report it as `viv run` and the compile report errors. A bad `--set`
    /// is reported as each did before there was a loader: `viv run` with
    /// `apply_sets`'s words, the compile with "error: " before them.
    fn report(&self, json: bool, set_prefix: &str) -> ExitCode {
        match self {
            LoadError::Plain(m) => fail_plain(m, json),
            LoadError::Set(m) => fail_plain(&format!("{set_prefix}{m}"), json),
            LoadError::Located(ds, sources) => {
                if json {
                    eprintln!("{}", diags_json(ds, sources));
                } else {
                    for d in ds {
                        eprint!("{}", PlainRenderer.render(d, sources));
                    }
                }
                ExitCode::FAILURE
            }
        }
    }
}

/// A Python program as py2viv translated it: the Viv text, and for each line
/// of it the Python it came from, `[line, col, end_line, end_col]` (1-based,
/// columns in bytes), `None` for a line with no Python origin.
struct PyTranslation {
    viv: String,
    posmap: Vec<Option<[u32; 4]>>,
}

/// A file is Python when its name ends in `.py` or it declares `#! lang python`.
fn is_python(path: &str, src: &str) -> bool {
    path.ends_with(".py") || viv::lang_of(src).as_deref() == Some("python")
}

/// py2viv, the numeric-Python translator (tools/py2viv/py2viv.py): the file
/// named by VIV_PY2VIV, or the one in the repository that holds the program,
/// or the one in the repository `viv` was built in.
fn py2viv_script(program: &std::path::Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("VIV_PY2VIV") {
        return Some(PathBuf::from(p));
    }
    let rel = std::path::Path::new("tools/py2viv/py2viv.py");
    let from_program = std::fs::canonicalize(program).ok().and_then(|p| p.ancestors().map(|a| a.join(rel)).find(|c| c.exists()));
    from_program.or_else(|| std::env::current_exe().ok().and_then(|e| e.ancestors().map(|a| a.join(rel)).find(|c| c.exists())))
}

/// Translate the Python file `file` (its text is `src`, and `name` is what
/// diagnostics call it) by running py2viv under CPython (VIV_PYTHON, default
/// python3). The translation is strict: a construct outside the numeric
/// subset is refused, and every refusal comes back as a diagnostic located in
/// the Python file.
fn py2viv(file: &std::path::Path, name: &str, src: &str) -> Result<PyTranslation, LoadError> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let Some(script) = py2viv_script(file) else {
        return Err(LoadError::Plain(format!("error: {name} is Python, and py2viv (tools/py2viv/py2viv.py) was not found: set VIV_PY2VIV to its path")));
    };
    // the translation is what py2viv's sources make of the file: all of them are inputs
    if let Some(d) = script.parent() {
        viv::trace::note_dir(d);
    }
    let python = std::env::var("VIV_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let stem = std::env::temp_dir().join(format!("viv-py2viv-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let (viv_out, manifest_out) = (stem.with_extension("viv"), stem.with_extension("json"));
    let ran = std::process::Command::new(&python).arg(&script).arg(file).arg("-o").arg(&viv_out).arg("--manifest").arg(&manifest_out).output();
    let viv_text = std::fs::read_to_string(&viv_out);
    let manifest = std::fs::read_to_string(&manifest_out);
    let _ = std::fs::remove_file(&viv_out);
    let _ = std::fs::remove_file(&manifest_out);
    let ran = ran.map_err(|e| LoadError::Plain(format!("error: {name} is Python, and running py2viv with {python} failed: {e}")))?;
    let Ok(manifest) = manifest else {
        return Err(LoadError::Plain(format!("error: py2viv did not translate {name}:\n{}", String::from_utf8_lossy(&ran.stderr).trim_end())));
    };
    let m = viv::story::parse_json(&manifest).map_err(|e| LoadError::Plain(format!("error: py2viv wrote a manifest that is not JSON: {e}")))?;
    let num = |j: Option<&viv::story::Json>| j.and_then(|v| v.num()).map_or(0, |x| x as u32);
    let mut sources = SourceMap::new();
    let py = sources.add_lang(name, src, "python");
    // a span of the Python file from 1-based line/col to end_line/end_col; a
    // span that runs past its first line ends at the end of that line
    let span_of = |sources: &SourceMap, line: u32, col: u32, end_line: u32, end_col: u32| {
        let start = sources.offset_of(py, line, col).unwrap_or(0);
        let end = if end_line == line { sources.offset_of(py, line, end_col).unwrap_or(start) } else { sources.offset_of(py, line, u32::MAX).unwrap_or(start) };
        knotkit::span::Span::new(py, start, end.max(start))
    };
    if !ran.status.success() {
        let Some(viv::story::Json::Arr(refusals)) = m.get("refusals") else {
            return Err(LoadError::Plain(format!("error: py2viv did not translate {name}:\n{}", String::from_utf8_lossy(&ran.stderr).trim_end())));
        };
        let mut diags = Vec::new();
        for r in refusals {
            let (line, col) = (num(r.get("line")), num(r.get("col")));
            let (end_line, end_col) = (r.get("end_line").map_or(line, |_| num(r.get("end_line"))), r.get("end_col").map_or(col + 1, |_| num(r.get("end_col"))));
            let what = r.get("what").and_then(|v| v.str()).unwrap_or("");
            let gap = r.get("gap").and_then(|v| v.str()).map(|g| format!(" (lifted by {g})")).unwrap_or_default();
            let mut d = knotkit::diag::Diagnostic::error(format!("not numeric Python: {what}{gap}"))
                .with_primary(span_of(&sources, line, col, end_line, end_col), "outside the numeric subset");
            if let Some(hint) = r.get("hint").and_then(|v| v.str()) {
                d = d.with_note(hint);
            }
            diags.push(d);
        }
        if let Some(last) = diags.pop() {
            diags.push(last.with_note("the numeric subset, and what to write instead: tools/py2viv/SUBSET.md"));
        }
        return Err(LoadError::Located(diags, sources));
    }
    let viv_text = viv_text.map_err(|e| LoadError::Plain(format!("error: py2viv wrote no Viv for {name}: {e}")))?;
    let mut posmap = Vec::new();
    if let Some(viv::story::Json::Arr(entries)) = m.get("posmap") {
        for e in entries {
            posmap.push(match e {
                viv::story::Json::Arr(v) if v.len() == 4 => Some([num(v.first()), num(v.get(1)), num(v.get(2)), num(v.get(3))]),
                _ => None,
            });
        }
    }
    Ok(PyTranslation { viv: viv_text, posmap })
}

/// The name a module goes by in diagnostics: its path under the repository
/// root, which is the same in every checkout (a location in a module ends up
/// in the C's location table, which is hashed and compiled into served
/// wasm), or `display_path` outside a repository.
fn module_display(root: &std::path::Path, file: &std::path::Path) -> String {
    if root.join("Cargo.toml").exists() {
        if let Ok(abs) = std::fs::canonicalize(file) {
            if let Ok(rel) = abs.strip_prefix(root) {
                return rel.display().to_string();
            }
        }
    }
    display_path(&file.display().to_string())
}

/// What one spliced text is, for the source map: its name, and for a Python
/// file its text and translation.
struct UnitFile {
    name: String,
    python: Option<(String, PyTranslation)>,
}

/// Load a program for the engines: read it, translate it if it is Python,
/// splice in its `#! use` modules (translating any written in Python), append
/// the debug lens when `debug`, apply the `--set`s and put the preludes first.
/// The source map records where every line came from.
fn load(path: &str, debug: bool, prelude: &str, sets: &[(String, String)]) -> Result<Loaded, LoadError> {
    let raw = TracedIo.read_to_string(&PathBuf::from(path)).map_err(|e| LoadError::Plain(format!("error: cannot read {}: {}", path, e)))?;
    load_text(path, None, raw, debug, prelude, sets)
}

/// `load` of a program whose text is `raw`: `path` is where it is, for the
/// modules beside it and its lens files (it need not exist), and `name` what
/// diagnostics call it (by default `display_path(path)`).
fn load_text(path: &str, name: Option<&str>, raw: String, debug: bool, prelude: &str, sets: &[(String, String)]) -> Result<Loaded, LoadError> {
    use std::cell::RefCell;
    let (here, root) = module_dirs(path);
    install_resolver(&here, &root);
    let shown = name.map_or_else(|| display_path(path), str::to_string);
    let python = is_python(path, &raw);
    let mut units: Vec<UnitFile> = Vec::new();
    let program = if python {
        let t = py2viv(std::path::Path::new(path), &shown, &raw)?;
        let viv = t.viv.clone();
        units.push(UnitFile { name: shown.clone(), python: Some((raw.clone(), t)) });
        viv
    } else {
        units.push(UnitFile { name: shown.clone(), python: None });
        raw
    };
    let debug_text = RefCell::new(if debug { debug_source(&PathBuf::from(path)).map_err(LoadError::Plain)? } else { String::new() });
    // each module as it is resolved: its file, and its translation when it is Python
    let files: RefCell<Vec<(String, PathBuf)>> = RefCell::new(Vec::new());
    let translations: RefCell<Vec<(String, String, PyTranslation)>> = RefCell::new(Vec::new());
    let located: RefCell<Option<LoadError>> = RefCell::new(None);
    let expansion = viv::expand_uses_mapped(
        &program,
        &mut |name: &str| {
            let c = match module_path(&here, &root, name) {
                Ok(c) => c,
                Err(e) => return embedded_module(name).ok_or(e),
            };
            let text = TracedIo.read_to_string(&c).map_err(|e| format!("error: cannot read {}: {}", c.display(), e))?;
            // one file may be found by two names (`./nn` beside a program in
            // lib/, and `/lib/nn`): its lens files are read once
            let canon = std::fs::canonicalize(&c).unwrap_or(c.clone());
            let again = files.borrow().iter().any(|(_, f)| std::fs::canonicalize(f).unwrap_or(f.clone()) == canon);
            if debug && !again {
                debug_text.borrow_mut().push_str(&debug_source(&c)?);
            }
            files.borrow_mut().push((viv::imports::canonical(name), c));
            Ok(text)
        },
        &mut |lang: &str, name: &str, text: &str| {
            if lang != "python" {
                return Err(format!("module `{name}`: it is written in `{lang}`, and nothing here can translate that"));
            }
            let file = files.borrow().iter().rev().find(|(n, _)| n == name).map(|(_, f)| f.clone()).unwrap_or_else(|| PathBuf::from(name));
            match py2viv(&file, &module_display(&root, &file), text) {
                Ok(t) => {
                    let viv = t.viv.clone();
                    translations.borrow_mut().push((name.to_string(), text.to_string(), t));
                    Ok(viv)
                }
                Err(e) => {
                    let msg = e.text();
                    *located.borrow_mut() = Some(e);
                    Err(msg)
                }
            }
        },
    );
    let expansion = match expansion {
        Ok(e) => e,
        Err(msg) => return Err(located.into_inner().unwrap_or(LoadError::Plain(msg))),
    };
    let debug_text = debug_text.into_inner();
    let body = if debug_text.is_empty() { expansion.text.clone() } else { format!("{}\n{debug_text}", expansion.text) };
    let body = viv::apply_sets(&body, sets).map_err(LoadError::Set)?;
    let text = format!("{prelude}{body}");
    viv::trace::note_runtime_reads(std::path::Path::new(path), &text);

    let files = files.into_inner();
    // the memory lens (`<module>.memory`) of the program, its modules and its
    // preludes: the `disjoint` pins the compiler accepts (viv::memory_lens)
    let mut lenses: Vec<(String, String)> = Vec::new();
    for f in std::iter::once(PathBuf::from(path)).chain(files.iter().map(|(_, f)| f.clone())) {
        let lp = f.with_extension("memory");
        if let Some(text) = read_opt(&lp) {
            let shown = module_display(&root, &lp);
            if !lenses.iter().any(|(n, _)| *n == shown) {
                lenses.push((shown, text));
            }
        }
    }
    lenses.extend(PRELUDE_LENSES.with(|p| p.borrow().clone()));
    viv::memory_lens::set_loaded(lenses);
    let mut translations = translations.into_inner();
    for u in &expansion.units[1..] {
        let file = files.iter().find(|(n, _)| *n == u.name).map(|(_, f)| f.clone()).unwrap_or_else(|| PathBuf::from(&u.name));
        let python = match translations.iter().position(|(n, _, _)| *n == u.name) {
            Some(k) if u.lang.as_deref() == Some("python") => {
                let (_, src, t) = translations.remove(k);
                Some((src, t))
            }
            _ => None,
        };
        units.push(UnitFile { name: module_display(&root, &file), python });
    }
    let name0 = if python { format!("{shown}.viv") } else { shown };
    let sources = source_map(&name0, &text, prelude, &prelude_parts(prelude), &expansion, units);
    Ok(Loaded { text, sources })
}

/// The source map of a loaded program. Source 0 is the text the engines
/// parse. When nothing was spliced or translated it is the only source, as it
/// always was. Otherwise every spliced file follows (a Python file, then the
/// Viv py2viv generated from it, derived from it by its position map), and
/// each line of source 0 that is a copy of a line of one of them is recorded
/// as such (a composite): the preludes, the debug lens, the splice markers and
/// a line a `--set` changed stay in source 0.
fn source_map(name0: &str, text: &str, prelude: &str, preludes: &PreludeParts, expansion: &viv::Expansion, units: Vec<UnitFile>) -> SourceMap {
    use knotkit::span::{LineMap, Origin, Piece, Span};
    let mut sources = SourceMap::new();
    let id0 = sources.add_lang(name0, text, "viv");
    if units.len() == 1 && units[0].python.is_none() && preludes.0.is_empty() {
        return sources;
    }
    // the prelude comes first in the text, and each part of it cut from a
    // file is a copy of that file's bytes (B250: a location in a `--use` file
    // was given as a line of the whole text, under the main program's name)
    let mut pieces = Vec::new();
    let mut at = 0usize;
    let mut ids: Vec<(String, knotkit::span::SourceId)> = Vec::new();
    for part in &preludes.0 {
        if let Some((name, offset)) = &part.file {
            let id = match ids.iter().find(|(n, _)| n == name) {
                Some(&(_, id)) => Some(id),
                None => preludes.1.iter().find(|(n, _)| n == name).map(|(_, sub)| {
                    let id = import_sources(&mut sources, sub);
                    ids.push((name.clone(), id));
                    id
                }),
            };
            if let (Some(id), false) = (id, part.text.is_empty()) {
                pieces.push(Piece { at: (at as u32, (at + part.text.len()) as u32), source: id, offset: *offset as u32 });
            }
        }
        at += part.text.len();
    }
    let mut ids = Vec::new();
    for (u, file) in expansion.units.iter().zip(units) {
        let id = match file.python {
            None => sources.add_lang(file.name, u.text.clone(), "viv"),
            Some((py_src, t)) => {
                let py = sources.add_lang(file.name.clone(), py_src, "python");
                let lines = t
                    .posmap
                    .iter()
                    .map(|e| {
                        e.map(|[line, col, end_line, end_col]| {
                            let start = sources.offset_of(py, line, col).unwrap_or(0);
                            let end = if end_line == line { sources.offset_of(py, line, end_col) } else { sources.offset_of(py, line, u32::MAX) };
                            Span::new(py, start, end.unwrap_or(start).max(start))
                        })
                    })
                    .collect();
                sources.add_derived(format!("{}.viv", file.name), t.viv, "viv", "py2viv", LineMap::new(lines))
            }
        };
        ids.push(id);
    }
    // where each line of `text` starts
    let starts = |t: &str| -> Vec<usize> { std::iter::once(0).chain(t.match_indices('\n').map(|(i, _)| i + 1)).collect() };
    let text_starts = starts(text);
    let unit_starts: Vec<Vec<usize>> = expansion.units.iter().map(|u| starts(&u.text)).collect();
    let line_at = |t: &str, st: &[usize], k: usize| -> Option<(usize, usize)> {
        let s = *st.get(k)?;
        if s > t.len() || (s == t.len() && k > 0) {
            return None;
        }
        Some((s, t[s..].find('\n').map_or(t.len(), |n| s + n)))
    };
    let first = prelude.matches('\n').count();
    // a program with no `#! use` line was not expanded: its line k is line k of unit 0
    let count = if expansion.lines.is_empty() { expansion.text.lines().count() } else { expansion.lines.len() };
    for k in 0..count {
        let origin = if expansion.lines.is_empty() { Some((0, k)) } else { expansion.lines[k] };
        let Some((u, j)) = origin else { continue };
        let (Some((a, b)), Some((ua, ub))) = (line_at(text, &text_starts, first + k), line_at(&expansion.units[u].text, &unit_starts[u], j)) else { continue };
        if text[a..b] != expansion.units[u].text[ua..ub] {
            continue;
        }
        // the newline goes with its line
        let end = if b < text.len() { b + 1 } else { b };
        pieces.push(Piece { at: (a as u32, end as u32), source: ids[u], offset: ua as u32 });
    }
    sources.set_origin(id0, Origin::Composite(pieces));
    sources
}

/// The `.debug` file beside a module, as `debug` blocks; empty when there is none.
fn debug_source(module: &std::path::Path) -> Result<String, String> {
    let p = module.with_extension("debug");
    let Some(text) = read_opt(&p) else { return Ok(String::new()) };
    viv::debug_lens::check_file(&text).map_err(|e| e.lines().map(|l| format!("{}:{l}", p.display())).collect::<Vec<_>>().join("\n"))?;
    Ok(viv::debug_lens::file_to_source(&text))
}

fn read_and(path: &str, f: fn(&str) -> Result<String, String>) -> Result<String, String> {
    f(&read_src(path)?)
}

/// Collect every `--use <file.viv>` prelude named in the args and read them,
/// concatenated in order. Preludes are prepended to the program source before
/// parsing, so BOTH the interpreter and the C backend see one combined source
/// (the only mechanism that is compileable by construction: it is text). This
/// is how a Viv-level stdlib — the contract strategies — reaches a program
/// without an import system.
fn read_preludes(args: &[String]) -> Result<String, String> {
    let mut out = String::new();
    let mut lenses = Vec::new();
    let mut parts = Vec::new();
    let mut files = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--use" {
            if let Some(p) = args.get(i + 1) {
                // the file with its own `#! use` and import lines spliced in,
                // and the map of where each line of that came from
                let Loaded { text, sources } = load(p, false, "", &[]).map_err(|e| e.text())?;
                out.push_str(&text);
                out.push('\n');
                parts.push(PreludePart { file: Some((display_path(p), 0)), text });
                parts.push(PreludePart { file: None, text: "\n".to_string() });
                files.push((display_path(p), sources));
                let lp = PathBuf::from(p).with_extension("memory");
                if let Some(text) = read_opt(&lp) {
                    lenses.push((lp.display().to_string(), text));
                }
            }
        }
        i += 1;
    }
    PRELUDE_LENSES.with(|p| *p.borrow_mut() = lenses);
    set_prelude_parts(parts, files);
    Ok(out)
}

/// A piece of a prelude: a slice of a loaded file (its name, and the byte
/// offset of the slice in the file's loaded text, which has its `#! use` and
/// import lines spliced in), or text the tool wrote (`file: None`): the
/// newline `read_preludes` puts after each `--use` file, or the lines
/// `viv export` writes around and between its files.
#[derive(Clone)]
struct PreludePart {
    file: Option<(String, usize)>,
    text: String,
}

/// What a prelude is made of: its parts in order, and the source map of each
/// file a part is cut from (`load`'s: its source 0 is the loaded text).
type PreludeParts = (Vec<PreludePart>, Vec<(String, SourceMap)>);

/// Record what the next prelude is made of.
fn set_prelude_parts(parts: Vec<PreludePart>, files: Vec<(String, SourceMap)>) {
    PRELUDE_PARTS.with(|p| *p.borrow_mut() = (parts, files));
}

/// The parts `prelude` is made of; none when `prelude` is not the text the
/// parts were last recorded for.
fn prelude_parts(prelude: &str) -> PreludeParts {
    let (parts, files) = PRELUDE_PARTS.with(|p| p.borrow().clone());
    let joined: String = parts.iter().map(|p| p.text.as_str()).collect();
    if prelude.is_empty() || joined != prelude {
        return (Vec::new(), Vec::new());
    }
    (parts, files)
}

/// Copy every source of `sub` into `into`, with the origins between them
/// carried over; returns the id `sub`'s source 0 has in `into`.
fn import_sources(into: &mut SourceMap, sub: &SourceMap) -> knotkit::span::SourceId {
    use knotkit::span::{LineMap, Origin, Piece, SourceId, Span};
    let base = into.len() as u32;
    let moved = |id: SourceId| SourceId(id.0 + base);
    for i in 0..sub.len() {
        let f = sub.file(SourceId(i as u32)).expect("a source of the map");
        into.add_lang(f.name.clone(), f.text.clone(), f.lang.clone());
    }
    for i in 0..sub.len() {
        let f = sub.file(SourceId(i as u32)).expect("a source of the map");
        let origin = match &f.origin {
            Origin::Written => continue,
            Origin::Composite(ps) => Origin::Composite(ps.iter().map(|p| Piece { at: p.at, source: moved(p.source), offset: p.offset }).collect()),
            Origin::Derived { via, map } => Origin::Derived {
                via: via.clone(),
                map: LineMap::new(map.lines().iter().map(|s| s.map(|s| Span::new(moved(s.source), s.start, s.end))).collect()),
            },
        };
        into.set_origin(SourceId(i as u32 + base), origin);
    }
    SourceId(base)
}

thread_local! {
    /// What the prelude `read_preludes` (or `viv export`) made last is made
    /// of: where each line of it came from (`source_map`).
    static PRELUDE_PARTS: std::cell::RefCell<PreludeParts> = std::cell::RefCell::new((Vec::new(), Vec::new()));
    /// The memory lens files of the `--use` preludes, which `load` hands to
    /// the compiler with the program's own.
    static PRELUDE_LENSES: std::cell::RefCell<Vec<(String, String)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Every `--set NAME=value` in the args.
fn read_sets(args: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut sets = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--set" {
            let Some(kv) = args.get(i + 1) else { return Err("--set needs NAME=value".to_string()) };
            let Some((k, v)) = kv.split_once('=') else { return Err(format!("--set {kv}: expected NAME=value")) };
            sets.push((k.trim().to_string(), v.trim().to_string()));
            i += 1;
        }
        // `--using CONTRACT=method`, applied with the sets
        if args[i] == "--using" {
            let Some(kv) = args.get(i + 1) else { return Err("--using needs CONTRACT=method".to_string()) };
            let Some((k, v)) = kv.split_once('=') else { return Err(format!("--using {kv}: expected CONTRACT=method")) };
            sets.push((format!("{}{}", viv::USING_SET, k.trim()), v.trim().to_string()));
            i += 1;
        }
        i += 1;
    }
    Ok(sets)
}

fn plot_cli(args: &[String]) -> ExitCode {
    let usage = "usage: viv plot [records.out | -] [-o out.svg] [--width N] [--chart K]\n  \
                 draws the figure records a run printed as SVG: every chart, one under another,\n  \
                 or the K-th (from 1) with --chart";
    let (mut input, mut out, mut width, mut chart) = (None::<String>, None::<String>, 640.0f64, None::<usize>);
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--width" | "--chart" if i + 1 >= args.len() => {
                eprintln!("{usage}");
                return ExitCode::FAILURE;
            }
            "-o" => { out = Some(args[i + 1].clone()); i += 1; }
            "--width" => match args[i + 1].parse::<f64>() {
                Ok(w) if w >= 200.0 => { width = w; i += 1; }
                _ => { eprintln!("--width takes a number of pixels, 200 or more"); return ExitCode::FAILURE; }
            },
            "--chart" => match args[i + 1].parse::<usize>() {
                Ok(k) if k >= 1 => { chart = Some(k); i += 1; }
                _ => { eprintln!("--chart takes a chart's number, from 1"); return ExitCode::FAILURE; }
            },
            "-h" | "--help" => { println!("{usage}"); return ExitCode::SUCCESS; }
            a if input.is_none() => input = Some(a.to_string()),
            _ => { eprintln!("{usage}"); return ExitCode::FAILURE; }
        }
        i += 1;
    }
    let text = match input.as_deref() {
        None | Some("-") => {
            let mut s = String::new();
            if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut s) {
                eprintln!("viv plot: reading stdin: {e}");
                return ExitCode::FAILURE;
            }
            s
        }
        Some(p) => match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => { eprintln!("viv plot: {p}: {e}"); return ExitCode::FAILURE; }
        },
    };
    let svg = match chart {
        None => viv::figsvg::render_page(&text, width),
        Some(k) => {
            let all = viv::figsvg::render_all(&text, width);
            match all.get(k - 1) {
                Some(s) => format!("{s}\n"),
                None => { eprintln!("viv plot: the output has {} chart(s), not {k}", all.len()); return ExitCode::FAILURE; }
            }
        }
    };
    let bad = viv::figsvg::take_label_errors(&text);
    if !bad.is_empty() {
        for e in &bad {
            eprintln!("viv plot: {e}");
        }
        return ExitCode::FAILURE;
    }
    match out {
        Some(p) => {
            if let Err(e) = std::fs::write(&p, svg) {
                eprintln!("viv plot: {p}: {e}");
                return ExitCode::FAILURE;
            }
        }
        None => print!("{svg}"),
    }
    ExitCode::SUCCESS
}

/// The interpreter thread's stack for `viv run`: room for
/// `viv::recursion::MAX_DEPTH` nested calls with frames to spare.
const INTERP_STACK: usize = 1 << 30;

/// The compiler's version, as `viv --version` prints it.
const VIV_VERSION: &str = concat!("viv ", env!("CARGO_PKG_VERSION"));

fn main() -> ExitCode {
    // a panic is a bug in viv: exit 70, after the
    // usual message
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // a reader that closed the pipe (`viv ... | head`) is not a bug
        let msg = info.payload().downcast_ref::<String>().map(String::as_str).or_else(|| info.payload().downcast_ref::<&str>().copied()).unwrap_or("");
        if msg.contains("Broken pipe") {
            std::process::exit(141);
        }
        default_hook(info);
        eprintln!("error: an internal error in viv (a bug); please report it with the command that caused it");
        std::process::exit(agent::INTERNAL as i32);
    }));
    // `--summary`, `--max-lines N` and `--page K` bound what an agent-facing
    // command prints (agent.rs); no command parses them itself
    let (args, budget) = match agent::take_budget(std::env::args().collect()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(agent::USAGE);
        }
    };
    // the commands an agent starts with (agent.rs): help, map, skill, hook,
    // and the envelope around the commands that keep their own JSON
    if args.len() >= 2 && args[1] == "help" {
        return agent::help(&args);
    }
    if args.get(1).is_some_and(|a| a == "--help" || a == "-h") {
        let mut a = args.clone();
        a[1] = "help".into();
        return agent::help(&a);
    }
    if args.len() >= 3 && args[2..].iter().any(|a| a == "--help" || a == "-h") && agent::COMMANDS.iter().any(|c| c.name == args[1]) {
        return agent::help(&["viv".into(), "help".into(), args[1].clone()]);
    }
    if args.len() >= 2 && args[1] == "map" {
        return agent::map(&args, &budget);
    }
    if args.len() >= 2 && args[1] == "skill" {
        return agent::skill(&args);
    }
    if args.len() >= 2 && args[1] == "hook" {
        return agent::hook(&args);
    }
    if args.len() >= 2 && args[1] == "fix" && args.iter().any(|a| a == "--apply-json") {
        return agent::apply_json(&args);
    }
    if args.len() >= 2 && agent::WRAPPED.contains(&args[1].as_str()) && args.iter().any(|a| a == "--json") && !agent::raw_json() {
        return agent::wrap(&args[1], &args, &budget);
    }
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        println!("{VIV_VERSION}");
        return ExitCode::SUCCESS;
    }

    // VIV_STEP_BUDGET=N: the interpreter's infinite-loop guard stops a run
    // past N steps instead of 2e9 (interp.rs STEP_BUDGET), for a long run
    // that is not a hang
    if let Ok(v) = std::env::var("VIV_STEP_BUDGET") {
        match v.trim().replace('_', "").parse::<f64>() {
            Ok(n) if n >= 1.0 && n.fract() == 0.0 && n < 1.8e19 => viv::interp::set_step_budget(n as u64),
            _ => {
                eprintln!("error: VIV_STEP_BUDGET={v}: a whole number of steps, such as 2e10");
                return ExitCode::FAILURE;
            }
        }
    }

    // The lens tools: `viv lens init|check|stamp|move`,
    // `viv view` and `viv explain`. They read source and intent files and
    // never run or compile anything.
    if args.len() >= 2 && matches!(args[1].as_str(), "lens" | "view" | "explain") {
        return lens_cli(&args);
    }
    // `viv facts --json <file>`: what the compiler knows about one file, for
    // the program database (facts.rs, tools/vivdb)
    // `viv db query|show|queries|build`: the program database through the
    // SQLite this binary carries (db_cli.rs, crate::db)
    if args.len() >= 2 && args[1] == "db" {
        return db_cli::main(&args[2..]);
    }
    if args.len() >= 2 && args[1] == "facts" {
        let Some(path) = args[2..].iter().find(|a| !a.starts_with("--")) else {
            eprintln!("usage: viv facts --json <file.viv>");
            return ExitCode::FAILURE;
        };
        let src = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read {path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        // the lens model (imports resolved) when the file loads; the facts
        // the parser alone gives otherwise
        let loaded = lens_input(path).and_then(|i| viv::lens::load(&i));
        if let Err(e) = &loaded {
            eprintln!("note: {path}: {}", e.lines().next().unwrap_or(""));
        }
        println!("{}", viv::facts::facts_json(path, &src, loaded.as_ref().ok()));
        return ExitCode::SUCCESS;
    }
    // `viv debt`: what the lenses still owe, the most central first: defs
    // and steps with no intent, stale entries, dead defs (lens_cli.rs)
    if args.len() >= 2 && args[1] == "debt" {
        return lens_cli::debt(&args);
    }
    // `viv arch check|report`: the architecture lens:
    // layers, budgets and multi-paths, checked against the code (arch_cli.rs)
    if args.len() >= 2 && args[1] == "arch" {
        return match args.get(2).map(String::as_str) {
            Some("check") => arch_cli::check(&args),
            Some("report") => arch_cli::report(&args),
            _ => {
                eprintln!("{}", arch_cli::USAGE);
                ExitCode::FAILURE
            }
        };
    }
    // `viv phrases [--json] [file]`: the phrase table, or what each phrase of
    // a program is (crate::phrases)
    if args.len() >= 2 && args[1] == "phrases" {
        let rest: Vec<&String> = args[2..].iter().collect();
        if rest.is_empty() {
            print!("{}", viv::phrases::table_text());
            return ExitCode::SUCCESS;
        }
        if rest.len() == 1 && rest[0] == "--json" {
            println!("{}", viv::phrases::table_json());
            return ExitCode::SUCCESS;
        }
        let path = rest[0];
        // resolve the `#! use` lines first, so the phrases the used modules
        // declare are known (viv::used_phrases)
        let report = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}")).and_then(|src| {
            expand_uses_from(&src, path)?;
            viv::phrases::uses_report(&src)
        });
        return match report {
            Ok(r) => {
                print!("{r}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    // `viv features [...]`: what Viv can do, and how much of it a corpus
    // uses (viv::features, features_cli.rs)
    // Fast mode (fast_cli.rs): `viv build`, and
    // `viv run --fast` / `viv check --fast`, which build natively
    if args.len() >= 2 && args[1] == "build" {
        return fast_cli::build(&args);
    }
    if args.len() >= 2 && args[1] == "run" && args.iter().any(|a| a == "--fast") {
        return fast_cli::run(&args);
    }
    if args.len() >= 2 && args[1] == "check" && args.iter().any(|a| a == "--fast") {
        return fast_cli::check(&args);
    }
    if args.len() >= 2 && args[1] == "features" {
        return features_cli::cli(&args);
    }
    // `viv claims`: the claims and their status (claims_cli.rs, tools/claims/run.py)
    if args.len() >= 2 && args[1] == "claims" {
        return claims_cli::cli(&args);
    }
    // `viv cas derivs`: lib/cas/derivs.viv, the derivative table as the CAS
    // reads it, written from lib/cas/rules/derivatives.toml (crate::derivs)
    if args.len() >= 2 && args[1] == "cas" {
        return match args.get(2).map(String::as_str) {
            Some("derivs") => {
                print!("{}", viv::derivs::viv_module());
                ExitCode::SUCCESS
            }
            _ => {
                eprintln!("usage: viv cas derivs    (the derivative table as lib/cas/derivs.viv)");
                ExitCode::FAILURE
            }
        };
    }
    // `viv lint <file or dir>...`: integer literals used as sizes and bounds
    // (lint.rs)
    if args.len() >= 2 && args[1] == "lint" {
        // each file's imports resolved first, so a file read alone knows the
        // signatures of the defs it imports and may call them with keyword
        // arguments (viv::used_sigs; B297)
        for a in args.iter().skip(2).filter(|a| !a.starts_with("--")) {
            let mut files = Vec::new();
            lint::viv_files(std::path::Path::new(a), &mut files);
            for f in &files {
                if let Ok(src) = std::fs::read_to_string(f) {
                    let _ = expand_uses_from(&src, &f.display().to_string());
                }
            }
        }
        return lint::cli(&args);
    }
    // `viv plot [records] [-o out.svg] [--width N] [--chart K]`: a run's figure
    // records drawn as SVG, the same bytes every time
    // (figsvg.rs). Reads the file, or stdin when none is named or it is `-`.
    // `viv export <manifest>`: a whole program as one C file and a Python
    // wrapper (viv::export)
    if args.len() >= 2 && args[1] == "export" {
        return export_cli(&args[2..]);
    }
    if args.len() >= 2 && args[1] == "cuda" {
        return cuda_cli(&args[2..]);
    }

    if args.len() >= 2 && args[1] == "plot" {
        return plot_cli(&args);
    }
    // `viv story <file>`: run the program in debug mode and tell the run in
    // the intent's words.
    if args.len() >= 2 && args[1] == "story" {
        return story_cli(&args);
    }
    // `viv graph <file>`: the program as a graph of defs, steps and the loops
    // that hold them, with each step's interface. `--json` prints it for the explorer; `--bundle` adds the
    // program as run, the line table, the intent and the debug lens; `--map`
    // is the code map of the defs an entry reaches (lang/viv/src/codemap.rs).
    if args.len() >= 2 && args[1] == "graph" {
        return graph_cli(&args);
    }
    // `viv atlas <file>`: the atlas of a program, its defs by module with the
    // calls between them and the counts of one run, as one page that opens
    // offline, as text (`--text`) or as JSON (`--json`) (lang/viv/src/atlas.rs)
    if args.len() >= 2 && args[1] == "atlas" {
        return atlas_cli(&args);
    }

    // `viv doc <file>`: extract the file's annotations (the #| / #[ / #]
    // convention) into JSON for code-browsing tools. Annotations are ordinary
    // comments to the compiler; this verb is the machine-readable side of the
    // contract. The syntax is in the `doc` entry of the features on /reference.
    // `viv imports --migrate [--write] <files...>`: each file's `#! use`
    // lines as import statements (viv::imports::migrate). Every file is
    // read and rewritten in memory first, so the order does not matter;
    // `--write` then writes them. A report line per file goes to stderr.
    if args.len() >= 2 && args[1] == "imports" {
        let write = args.iter().any(|a| a == "--write");
        viv::imports::LEGACY_USE.with(|c| c.set(true));
        let files: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
        let mut results: Vec<(String, String)> = Vec::new();
        let mut bad = 0;
        let (mut uses, mut imports) = (0, 0);
        for f in &files {
            let src = match TracedIo.read_to_string(&PathBuf::from(f.as_str())) {
                Ok(s) => s,
                Err(e) => { eprintln!("{f}: {e}"); bad += 1; continue; }
            };
            if !src.lines().any(|l| l.starts_with("#! use ")) {
                continue;
            }
            let (here, root) = module_dirs(f);
            let exp = viv::expand_uses_mapped(&src, &mut |name: &str| {
                let c = match module_path(&here, &root, name) {
                    Ok(c) => c,
                    Err(e) => return embedded_module(name).ok_or(e),
                };
                TracedIo.read_to_string(&c).map_err(|e| format!("cannot read {}: {}", c.display(), e))
            }, &mut |lang, name, _| Err(format!("{name}: written in {lang}")));
            let m = exp.and_then(|e| viv::imports::migrate(&src, &e));
            match m {
                Ok(m) => {
                    for (name, ms) in &m.ambiguous {
                        eprintln!("{f}: `{name}` is defined in {}: imported from the last", ms.join(", "));
                    }
                    uses += m.uses;
                    imports += m.imports;
                    results.push((f.to_string(), m.text));
                }
                Err(e) => { eprintln!("{f}: {e}"); bad += 1; }
            }
        }
        eprintln!("{} files, {uses} `#! use` lines, {imports} import lines, {bad} not migrated", results.len());
        for (f, t) in &results {
            if write {
                if let Err(e) = std::fs::write(f, t) {
                    eprintln!("{f}: {e}");
                    bad += 1;
                }
            } else {
                println!("==> {f}\n{t}");
            }
        }
        return if bad == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    // `viv fix --keywords [--write] <files...>`: name every builtin option a
    // file writes by position (viv::kwargs::name_options), keeping a file
    // only when its definitions hash as before (the filled call is the call
    // the engines read either way).
    if args.len() >= 2 && args[1] == "fix" {
        if !args.iter().any(|a| a == "--keywords") {
            eprintln!("usage: viv fix --keywords [--write] <file.viv>...");
            return ExitCode::FAILURE;
        }
        let write = args.iter().any(|a| a == "--write");
        // --json: the envelope, with each file's edits as `viv fix
        // --apply-json` reads them (agent.rs)
        let json = args.iter().any(|a| a == "--json");
        let mut json_files: Vec<String> = Vec::new();
        let (mut files_changed, mut named, mut bad, mut unchecked) = (0, 0, 0, 0);
        // the files are read as they were written, options by position included
        viv::parser::POSITIONAL_OPTIONS.with(|c| c.set(true));
        for f in args[2..].iter().filter(|a| !a.starts_with("--")) {
            let src = match TracedIo.read_to_string(&PathBuf::from(f.as_str())) {
                Ok(s) => s,
                Err(e) => { eprintln!("{f}: {e}"); bad += 1; continue; }
            };
            // a program in another language (`#! lang python`) is not Viv's to rewrite
            if viv::lang_of(&src).is_some_and(|l| l != "viv") {
                continue;
            }
            let before = expand_uses_from(&src, f).and_then(|e| {
                let defs = viv::lexer::lex(&e, knotkit::span::SourceId(0)).map(|ts| ts.windows(2).filter_map(|w| match (&w[0].tok, &w[1].tok) { (viv::lexer::Tok::Def, viv::lexer::Tok::Ident(n)) => Some(n.clone()), _ => None }).collect::<std::collections::HashSet<String>>()).unwrap_or_default();
                viv::hash_report(&e).map(|h| (h, defs))
            });
            // a file that is not a program on its own (a deck module, a probe
            // that is meant to fail) is rewritten without the hash check
            let (h0, defs) = match before {
                Ok(x) => (Some(x.0), x.1),
                // its own defs, at least
                Err(_) => (None, viv::lexer::lex(&src, knotkit::span::SourceId(0)).map(|ts| ts.windows(2).filter_map(|w| match (&w[0].tok, &w[1].tok) { (viv::lexer::Tok::Def, viv::lexer::Tok::Ident(n)) => Some(n.clone()), _ => None }).collect()).unwrap_or_default()),
            };
            let (new, n) = match viv::kwargs::name_options(&src, &defs) {
                Ok(x) => x,
                Err(e) => { eprintln!("{f}: {e}"); bad += 1; continue; }
            };
            if n == 0 {
                continue;
            }
            let after = if h0.is_some() { expand_uses_from(&new, f).and_then(|e| viv::hash_report(&e)).map(Some) } else { Ok(None) };
            if h0.is_none() {
                unchecked += 1;
            }
            match after {
                Ok(h1) if h1 == h0 => {
                    files_changed += 1;
                    named += n;
                    if write {
                        if let Err(e) = std::fs::write(f, &new) { eprintln!("{f}: {e}"); bad += 1; }
                    } else if !json {
                        println!("{f}: {n} option{} named", if n == 1 { "" } else { "s" });
                    }
                    if json {
                        let edits = agent::line_edits(f, &src, &new);
                        json_files.push(format!(r#"{{"file":{},"named":{n},"written":{write},"fix":{{"message":"name the options","edits":[{}]}}}}"#, viv::debug_lens::json_str(f), edits.join(",")));
                    }
                }
                Ok(_) => { eprintln!("{f}: the definitions hash differently after the rewrite; left as it is"); bad += 1; }
                Err(e) => { eprintln!("{f}: after the rewrite: {}", e.lines().next().unwrap_or("")); bad += 1; }
            }
        }
        if json {
            // each file's fix as a diagnostic, so `viv fix --apply-json` reads this envelope as it is
            let diags: Vec<String> = json_files
                .iter()
                .filter_map(|f| viv::story::parse_json(f).ok())
                .map(|f| {
                    let file = f.get("file").map(|x| x.text()).unwrap_or_default();
                    format!(r#"{{"severity":"note","rule":"S11-options","message":"positional options can be named","file":{file},"line":null,"col":null,"notes":[],"fix":{}}}"#, f.get("fix").map(|x| x.text()).unwrap_or_default())
                })
                .collect();
            let result = format!(r#"{{"files":[{}],"named":{named},"not_rewritten":{bad}}}"#, json_files.join(","));
            return agent::print_envelope("fix", if bad == 0 { agent::OK } else { agent::COMPILE }, &format!("[{}]", diags.join(",")), &result, false, None);
        }
        eprintln!("{files_changed} files, {named} options named ({unchecked} files not programs on their own, rewritten without the hash check), {bad} not rewritten");
        return if bad == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    // `viv expand <file>`: the program as the engines read it, its imports
    // and `#! use` lines spliced and its names resolved (crate::imports).
    if args.len() >= 2 && args[1] == "expand" {
        let Some(path) = args.get(2) else {
            eprintln!("usage: viv expand <input.viv>");
            return ExitCode::FAILURE;
        };
        return match read_src(path) {
            Ok(src) => {
                print!("{src}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() >= 2 && args[1] == "src" {
        let (Some(path), Some(name)) = (args.get(2), args.get(3)) else {
            eprintln!("usage: viv src <input.viv> <fn>");
            return ExitCode::FAILURE;
        };
        return match read_src(path) {
            Ok(src) => {
                print!("{}", viv::fn_src(&src, name));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{}", e);
                ExitCode::FAILURE
            }
        };
    }
    // `viv run <file>`: execute through the tree-walking interpreter (the same
    // path the browser workbench's `viv_run` uses). The interpreter, the wasm
    // backend, and the C backend are meant to agree to the digit — this verb
    // makes the interpreter runnable from the CLI so it can be diffed natively.
    // `viv digits <file>`: which digits of the printed numbers are real
    // (viv::digits)
    if args.len() >= 2 && args[1] == "digits" {
        return digits_cli(&args);
    }
    if args.len() >= 2 && args[1] == "run" {
        // `viv run [--release] <file>`: release mode makes contracted solves
        // (spectrum_sym) auto-select their method; the default (debug) runs
        // the full method sweep and fails loudly on dissent.
        let release = args.iter().any(|a| a == "--release");
        // --quiet: suppress the __graph__/__trace__/__steps__ diagnostic lines
        let quiet = args.iter().any(|a| a == "--quiet" || a == "-q");
        // --json: diagnostics (errors, and the failed-assert summary) go to
        // stderr as one JSON object, for tools and models, instead of text
        let json = args.iter().any(|a| a == "--json");
        // `--use <file>` prepends a prelude; skip both the flag and its value
        let mut skip_next = false;
        let Some(path) = args.iter().skip(2).find(|a| {
            if skip_next { skip_next = false; return false; }
            if a.as_str() == "--use" || a.as_str() == "--set" || a.as_str() == "--using" { skip_next = true; return false; }
            !matches!(a.as_str(), "--release" | "--quiet" | "-q" | "--json")
        }) else {
            return usage_failed("run", "usage: viv run [--release] [--quiet] [--json] [--summary] [--max-lines N] [--use <prelude.viv>] [--set NAME=value] [--using CONTRACT=method] <input.viv>", "U-usage", json);
        };
        set_given_path(path);
        let prelude = match read_preludes(&args[2..]) {
            Ok(p) => p,
            Err(e) => return usage_failed("run", &e, "U-file", json),
        };
        let sets = match read_sets(&args[2..]) {
            Ok(s) => s,
            Err(e) => return usage_failed("run", &format!("error: {e}"), "U-set", json),
        };
        // The source map holds the program as the engines see it, under the
        // name the compiled program uses too (display_path), so a run-time
        // error reads like a compile error: file:line:col, the line, and a
        // caret under the part that failed. A location in a `#! use`d module,
        // or in the Viv translated from a Python file, is shown in the module
        // or the Python file (`load`).
        let Loaded { text: src, sources } = match load(path, true, &prelude, &sets) {
            Ok(l) => l,
            Err(e) => return load_failed(&e, "run", json, ""),
        };
        // The interpreter walks the tree on the native stack, and a recursion
        // as deep as the limit every engine shares (viv::recursion::MAX_DEPTH)
        // needs more than the main thread's 8 MB. The stack is reserved, not
        // committed: a run that does not recurse touches little of it.
        let run = {
            let src = src.clone();
            match std::thread::Builder::new().stack_size(INTERP_STACK).spawn(move || viv::interp::run_diagnosed(&src, release)) {
                Ok(h) => match h.join() {
                    Ok(run) => run,
                    Err(_) => return usage_failed("run", "error: the interpreter stopped unexpectedly", "I-internal", json),
                },
                Err(e) => return usage_failed("run", &format!("error: cannot start the interpreter: {e}"), "I-internal", json),
            }
        };
        let envelope = json && !agent::legacy_stderr();
        if envelope {
            return run_envelope(path, run, &sources, &budget);
        }
        if quiet {
            for line in run.out.lines() {
                if line.starts_with("__graph__") || line.starts_with("__trace__") || line.starts_with("__steps__") || line.starts_with("__story__") {
                    continue;
                }
                println!("{}", line);
            }
        } else {
            print!("{}", run.out);
        }
        // What the program printed comes first, then why it stopped -- the
        // order the compiled program's two streams have too.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let mut diags = run.errors;
        let stopped = !diags.is_empty();
        if !stopped {
            // a failed hold is in the story, not in the output; like a
            // failed assert, it makes the run a failure. Both summaries go to
            // stderr, as the compiled program writes them.
            if run.holds > 0 {
                diags.push(knotkit::diag::Diagnostic::error(format!("{} hold{} FAILED", run.holds, if run.holds == 1 { "" } else { "s" })));
            }
            // failed asserts make the run itself a failure, so CI and
            // harnesses can gate on the exit code
            if run.fails > 0 {
                diags.push(knotkit::diag::Diagnostic::error(format!("{} assertion{} FAILED", run.fails, if run.fails == 1 { "" } else { "s" })));
            }
        }
        if json {
            eprintln!("{}", diags_json_kind(&diags, &sources, run.kind));
        } else if stopped {
            for d in &diags {
                eprint!("{}", PlainRenderer.render(d, &sources));
            }
        } else {
            // the summaries are bare lines, not "error: ..." diagnostics
            for d in &diags {
                eprintln!("{}", d.message);
            }
        }
        return ExitCode::from(exit_of(&diags, run.kind));
    }
    // `viv check [--json] [--use <prelude>] [--set NAME=value] <file>`: the
    // checks `viv run` makes before it runs (the compiler's front end, which
    // the C and the wasm builds share, and the interpreter's own), and
    // nothing else. Silent and exit 0 when the program would run; its
    // diagnostics and exit 1 when not. A Python file is translated first.
    // `viv check --phrases [--json] [--cert] [--emit out.viv] <file>`: every
    // phrase with a closed form held to lib/cas's exact value (cas_check.rs)
    if args.len() >= 2 && args[1] == "check" && args.iter().any(|a| a == "--phrases") {
        return check_phrases(&args[2..]);
    }
    if args.len() >= 2 && args[1] == "check" {
        let json = args.iter().any(|a| a == "--json");
        let mut skip_next = false;
        let Some(path) = args.iter().skip(2).find(|a| {
            if skip_next { skip_next = false; return false; }
            if a.as_str() == "--use" || a.as_str() == "--set" || a.as_str() == "--using" { skip_next = true; return false; }
            a.as_str() != "--json"
        }) else {
            return usage_failed("check", "usage: viv check [--json] [--summary] [--max-lines N] [--use <prelude.viv>] [--set NAME=value] <input.viv | input.py>", "U-usage", json);
        };
        set_given_path(path);
        let prelude = match read_preludes(&args[2..]) {
            Ok(p) => p,
            Err(e) => return usage_failed("check", &e, "U-file", json),
        };
        let sets = match read_sets(&args[2..]) {
            Ok(s) => s,
            Err(e) => return usage_failed("check", &format!("error: {e}"), "U-set", json),
        };
        let Loaded { text: src, sources } = match load(path, true, &prelude, &sets) {
            Ok(l) => l,
            Err(e) => return load_failed(&e, "check", json, ""),
        };
        let diags = viv::interp::check_diagnosed(&src);
        let exit = exit_of(&diags, None);
        if json && !agent::legacy_stderr() {
            let (arr, rules) = diags_array(&diags, &sources, None);
            let mut counts: Vec<(String, usize)> = Vec::new();
            for r in &rules {
                match counts.iter_mut().find(|(k, _)| k == r) {
                    Some(c) => c.1 += 1,
                    None => counts.push((r.to_string(), 1)),
                }
            }
            let counts = counts.iter().map(|(k, n)| format!("{}:{n}", viv::debug_lens::json_str(k))).collect::<Vec<_>>().join(",");
            let (arr, cut) = bound_diags(arr, &budget);
            let result = format!(r#"{{"file":{},"counts":{{{counts}}}}}"#, viv::debug_lens::json_str(path));
            let more = cut.then(|| agent::more_cursor(&budget));
            return agent::print_envelope("check", exit, &arr, &result, cut, more.as_deref());
        }
        if json {
            eprintln!("{}", diags_json(&diags, &sources));
        } else {
            for d in &diags {
                eprint!("{}", PlainRenderer.render(d, &sources));
            }
        }
        return ExitCode::from(exit);
    }
    if args.len() >= 2 && args[1] == "ladder" {
        // `viv ladder <file.viv>`: run every rung of a ladder and report, as JSON.
        // Exit code 1 if any rung breaks a rule.
        let Some(path) = args.get(2) else {
            eprintln!("usage: viv ladder <input.viv>");
            return ExitCode::FAILURE;
        };
        return match read_src(path).and_then(|src| viv::ladder::report(&src, &mut |text| viv::interp::run(text))) {
            Ok(json) => {
                println!("{}", json);
                if json.contains("\"broken\":[\"") { ExitCode::FAILURE } else { ExitCode::SUCCESS }
            }
            Err(e) => { eprintln!("error: {}", e); ExitCode::FAILURE }
        };
    }
    if args.len() >= 2 && args[1] == "lsfunc" {
        let Some(path) = args.get(2) else {
            eprintln!("usage: viv lsfunc <input.viv> [--placeholders]");
            return ExitCode::FAILURE;
        };
        let only = args.iter().any(|a| a == "--placeholders");
        return match read_src(path) {
            Ok(src) => {
                print!("{}", if only { viv::lsfunc_placeholders(&src) } else { viv::lsfunc(&src) });
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{}", e);
                ExitCode::FAILURE
            }
        };
    }
    if args.len() >= 2 && args[1] == "lsvar" {
        let Some(path) = args.get(2) else {
            eprintln!("usage: viv lsvar <input.viv> [fn]");
            return ExitCode::FAILURE;
        };
        let filter = args.get(3).map(String::as_str).unwrap_or("");
        return match read_src(path) {
            Ok(src) => {
                print!("{}", viv::lsvar(&src, filter));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{}", e);
                ExitCode::FAILURE
            }
        };
    }
    if args.len() >= 2 && args[1] == "whatis" {
        let (Some(path), Some(name)) = (args.get(2), args.get(3)) else {
            eprintln!("usage: viv whatis <input.viv> <name>");
            return ExitCode::FAILURE;
        };
        let io = NativeIo;
        return match io.read_to_string(&PathBuf::from(path)) {
            Ok(src) => {
                print!("{}", viv::whatis(&src, name));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: cannot read {}: {}", path, e);
                ExitCode::FAILURE
            }
        };
    }
    if args.len() >= 2 && args[1] == "doc" {
        let Some(path) = args.get(2) else {
            eprintln!("usage: viv doc <input.viv> [fn]");
            return ExitCode::FAILURE;
        };
        let io = NativeIo;
        return match io.read_to_string(&PathBuf::from(path)) {
            Ok(src) => {
                match args.get(3) {
                    Some(fname) => print!("{}", doc_card(&src, fname)),
                    None => println!("{}", viv::doc_json(&src)),
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: cannot read {}: {}", path, e);
                ExitCode::FAILURE
            }
        };
    }

    // `viv deps <file>`: the static function-dependency graph as JSON.
    // `viv canon <file>`: rewrite with defs in canonical dependency order
    // (callees before callers, original order breaking ties), top-level code
    // unchanged. Function order is presentation; this is the canonical
    // presentation.
    if args.len() >= 2 && (args[1] == "deps" || args[1] == "canon") {
        let Some(path) = args.get(2) else {
            eprintln!("usage: viv {} <input.viv>", args[1]);
            return ExitCode::FAILURE;
        };
        let io = NativeIo;
        let src = match io.read_to_string(&PathBuf::from(path)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read {}: {}", path, e);
                return ExitCode::FAILURE;
            }
        };
        return match args[1].as_str() {
            "deps" => match if args.iter().any(|a| a == "--json") {
                viv::deps_json(&src)
            } else {
                viv::deps_report(&src)
            } {
                Ok(j) => {
                    println!("{}", j);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{}", e);
                    ExitCode::FAILURE
                }
            },
            _ => match viv::canon(&src) {
                Ok(t) => {
                    print!("{}", t);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{}", e);
                    ExitCode::FAILURE
                }
            },
        };
    }

    // `viv hash <file>`: per-definition content addresses. `viv diff <a> <b>`:
    // structural diff by those addresses — detects edits, renames (same body,
    // new name), additions, removals. The embryo of language-level VCS; see
    // the `hash-diff-canon` entry of the features on /reference.
    if args.len() >= 2 && (args[1] == "hash" || args[1] == "diff") {
        return match args[1].as_str() {
            "hash" => {
                let Some(path) = args.get(2) else {
                    eprintln!("usage: viv hash <input.viv>");
                    return ExitCode::FAILURE;
                };
                match read_and(path, viv::hash_report) {
                    Ok(t) => {
                        print!("{}", t);
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        ExitCode::FAILURE
                    }
                }
            }
            _ => {
                let (Some(a), Some(b)) = (args.get(2), args.get(3)) else {
                    eprintln!("usage: viv diff <old.viv> <new.viv>");
                    return ExitCode::FAILURE;
                };
                match (read_src(a), read_src(b)) {
                    (Ok(o), Ok(n)) => match viv::diff_report(&o, &n) {
                        Ok(t) => {
                            print!("{}", t);
                            ExitCode::SUCCESS
                        }
                        Err(e) => {
                            eprintln!("{}", e);
                            ExitCode::FAILURE
                        }
                    },
                    (Err(e), _) | (_, Err(e)) => {
                        eprintln!("{}", e);
                        ExitCode::FAILURE
                    }
                }
            }
        };
    }

    let Some((input, output, emit)) = parse_args(&args[1..]) else {
        eprintln!("usage: viv <input.viv | input.py> [-o <output.c|output.wasm>] [--emit tokens|ast|ir]");
        eprintln!("       viv run|check <input.viv | input.py>   (a .py file, or one with `#! lang python`, is translated by tools/py2viv)");
        eprintln!("       viv doc|hash|deps|canon <input.viv>   |   viv diff <old.viv> <new.viv>
       viv whatis <input.viv> <name>   (glossary lookup: what is this variable?)
       viv lsfunc <input.viv> [--placeholders]   (table of contents: every function + doc)
       viv lsvar <input.viv> [fn]      (the whole variable glossary, by function)
       viv src <input.viv> <fn>        (one function's source, annotations and all)
       viv deps <input.viv> [--json]   (call graph: pretty by default, --json for tools)
       viv view <input.viv> [--lens code|intent|pseudocode|debug|story|memory|all] [--json]
       viv explain <input.viv> [--from <def>] [--depth <k>] [--json]
       viv story <input.viv> [--depth <k>] [--expand] [--json] [--engine interp|c|fast] [--from <run output>]
       viv lens init|check|stamp|move ...   (the intent lens: `viv lens` for usage)
       viv features [<name> | --search <words> | --usage [paths]]   (what Viv can do: consult it before writing code)
       viv claims [run ...]            (bounded statements checked by programs: claims/, tools/claims/run.py)
       viv help [--agent | <command>]  (every command; --agent: the guide for an agent)");
        return ExitCode::from(agent::USAGE);
    };

    let io = NativeIo;
    let prelude = match read_preludes(&args[1..]) {
        Ok(p) => p,
        Err(e) => { eprintln!("{}", e); return ExitCode::FAILURE; }
    };
    let sets = match read_sets(&args[1..]) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: {}", e); return ExitCode::FAILURE; }
    };
    let json = JSON.load(std::sync::atomic::Ordering::Relaxed);
    // The source map owns file text so diagnostics can show line/column, in
    // the module or Python file a line came from (`load`).
    let Loaded { text: src, sources } = match load(&input.display().to_string(), true, &prelude, &sets) {
        Ok(l) => l,
        Err(e) => return e.report(json, "error: "),
    };
    let source_id = knotkit::span::SourceId(0);

    // Debug emissions: print the requested intermediate representation and stop.
    // These are the compiler's own artifacts, verbatim — the docs pages that
    // trace a program through the pipeline are generated from these.
    if let Some(what) = emit {
        return match what.as_str() {
            "tokens" => match viv::lexer::lex(&src, source_id) {
                Ok(toks) => {
                    for t in &toks {
                        println!("{:?}", t.tok);
                    }
                    ExitCode::SUCCESS
                }
                Err(diags) => render_diags(&diags, &sources),
            },
            "ast" => match viv::parse(&src, source_id) {
                Ok(ast) => {
                    println!("{:#?}", ast);
                    ExitCode::SUCCESS
                }
                Err(diags) => render_diags(&diags, &sources),
            },
            "ir" => match viv::build_ir(&src, source_id) {
                Ok(prog) => {
                    for root in prog.roots() {
                        print_ir(&prog, *root, 0);
                    }
                    ExitCode::SUCCESS
                }
                Err(diags) => render_diags(&diags, &sources),
            },
            other => {
                eprintln!("error: unknown --emit '{}': expected tokens|ast|ir", other);
                ExitCode::FAILURE
            }
        };
    }

    // The direct wasm backend: `-o out.wasm` emits a WebAssembly module from
    // the same IR the C backend compiles (run it with tools/run-wasm.mjs).
    let wasm_out = output.clone().unwrap_or_else(|| input.with_extension("c"));
    if wasm_out.extension().map_or(false, |e| e == "wasm") {
        let program = match viv::build_ir(&src, source_id) {
            Ok(p) => p,
            Err(diags) => return render_diags(&diags, &sources),
        };
        return match viv::wasm_backend::emit_wasm(&program) {
            Ok(bytes) => {
                if let Err(e) = io.write(&wasm_out, &bytes) {
                    eprintln!("error: cannot write {}: {}", wasm_out.display(), e);
                    return ExitCode::FAILURE;
                }
                println!("wrote {} ({} bytes wasm)", wasm_out.display(), bytes.len());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {}", e);
                ExitCode::FAILURE
            }
        };
    }

    if let Some(dir) = SPLIT.get() {
        return write_split(dir, &sources, &src, source_id);
    }

    // Array indexing in the C carries its source location, so an index out
    // of bounds stops the compiled program with the interpreter's words and
    // the same file:line:col (viv::lower::with_sites).
    let (program, emission) = match viv::lower::with_sites_map(sources.clone(), &src, || {
        let program = viv::build_ir(&src, source_id)?;
        let emission = CBackend.emit(&program, &io)?;
        Ok::<_, Vec<knotkit::diag::Diagnostic>>((program, emission))
    }) {
        Ok(x) => x,
        Err(diags) => return render_diags(&diags, &sources),
    };

    let Some(c) = emission.artifact.as_c() else {
        eprintln!("error: C backend did not produce C");
        return ExitCode::FAILURE;
    };

    let out_path = output.unwrap_or_else(|| input.with_extension("c"));
    if let Err(e) = io.write(&out_path, c.as_bytes()) {
        eprintln!("error: cannot write {}: {}", out_path.display(), e);
        return ExitCode::FAILURE;
    }

    println!(
        "wrote {} ({} IR nodes, {} debug entries)",
        out_path.display(),
        program.len(),
        emission.debug_map.entries.len()
    );
    if JSON.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("{}", diags_json(&[], &sources));
    }
    ExitCode::SUCCESS
}

/// Report an error that has no source location (a file that cannot be read,
/// a bad `--set`): as text, or under `--json` as a one-diagnostic object.
fn fail_plain(msg: &str, json: bool) -> ExitCode {
    if json {
        let d = knotkit::diag::Diagnostic::error(msg.strip_prefix("error: ").unwrap_or(msg));
        eprintln!("{}", diags_json(&[d], &SourceMap::new()));
    } else {
        eprintln!("{}", msg);
    }
    ExitCode::FAILURE
}

fn parse_args(args: &[String]) -> Option<(PathBuf, Option<PathBuf>, Option<String>)> {
    let mut input = None;
    let mut output = None;
    let mut emit = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i)?));
            }
            "--emit" => {
                i += 1;
                emit = Some(args.get(i)?.clone());
            }
            "--use" | "--set" | "--using" => {
                i += 1;               // skip the value; handled by read_preludes / read_sets
            }
            "--json" => JSON.store(true, std::sync::atomic::Ordering::Relaxed),
            "--split" => {
                i += 1;
                let _ = SPLIT.set(PathBuf::from(args.get(i)?));
            }
            other => {
                if input.is_some() {
                    return None;
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    Some((input?, output, emit))
}

// ---------------------------------------------------------------------------
// `viv canon`: the canonical presentation of a program. Definitions are
// emitted in dependency order — callees before callers, original order
// breaking ties, mutual-recursion groups kept together — each def carrying
// the comment block directly above it. Top-level statements follow in their
// original order (their sequence is semantics; def order is not).
// ---------------------------------------------------------------------------

/// `viv doc <file> <fn>`: one function's card — signature line, the
/// what/how/why doc, tags, and its static calls. Introlang's `doc` query.
fn doc_card(src: &str, fname: &str) -> String {
    let dj = viv::doc_json(src);
    let deps = viv::deps_json(src).unwrap_or_default();
    let take = |hay: &str, key: &str| -> String {
        hay.split(&format!("\"{}\":\"", key))
            .nth(1)
            .map(|r| r.chars().take_while(|c| *c != '"').collect())
            .unwrap_or_default()
    };
    // functions carry "line" right after "name"; vars entries carry "desc" —
    // anchor on the function form, then brace-match to the object's real end
    let anchor = format!("{{\"name\":\"{}\",\"line\":", fname);
    let Some(start) = dj.find(&anchor) else {
        return format!("no such function: {}\n", fname);
    };
    let mut depth = 0usize;
    let mut end = dj.len();
    for (i, c) in dj[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 { end = start + i + 1; break; }
            }
            _ => {}
        }
    }
    let obj = dj[start..end].to_string();
    let mut out = format!("ƒ {}\n", fname);
    for (label, key) in [("what", "what"), ("how", "how"), ("why", "why")] {
        let v = take(&obj, key);
        if !v.is_empty() {
            out.push_str(&format!("  {:>4}: {}\n", label, v));
        }
    }
    if let Some(tags) = obj.split("\"tags\":[").nth(1) {
        let t: String = tags.chars().take_while(|c| *c != ']').collect();
        if !t.is_empty() {
            out.push_str(&format!("  tags: {}\n", t.replace('"', "")));
        }
    }
    if let Some(vs) = obj.split("\"vars\":[").nth(1) {
        // descriptions may contain ']' — the array really ends at the next key
        let v: &str = vs.split("],\"lineNotes\":[").next().unwrap_or("");
        for entry in v.split("},{") {
            let name: String = entry.split("\"name\":\"").nth(1).map(|r| r.chars().take_while(|c| *c != '"').collect()).unwrap_or_default();
            let desc: String = entry.split("\"desc\":\"").nth(1).map(|r| r.chars().take_while(|c| *c != '"').collect()).unwrap_or_default();
            if !name.is_empty() {
                out.push_str(&format!("     : {:<8} {}\n", name, desc.replace("\\\"", "\"")));
            }
        }
    }
    for st in viv::steps(src).iter().filter(|st| st.def == fname) {
        let text = if st.text.is_empty() { String::new() } else { format!("  {}", st.text) };
        out.push_str(&format!("  {}step {}{}  (line {})\n", "  ".repeat(st.depth as usize - 1), st.id, text, st.line));
    }
    let line = obj
        .split("\"rawLine\":")
        .nth(1)
        .or_else(|| obj.split("\"line\":").nth(1))
        .map(|r| r.chars().take_while(|c| c.is_ascii_digit()).collect::<String>())
        .unwrap_or_default();
    out.push_str(&format!("  line: {}\n", line));
    if let Some(f) = deps
        .split("{\"name\":\"")
        .skip(1)
        .find(|o| o.starts_with(&format!("{}\"", fname)))
    {
        if let Some(cl) = f.split("\"calls\":[").nth(1) {
            let c: String = cl.chars().take_while(|c| *c != ']').collect();
            if !c.is_empty() {
                out.push_str(&format!(" calls: {}\n", c.replace('"', "")));
            }
        }
    }
    out
}

fn render_diags(diags: &[knotkit::diag::Diagnostic], sources: &SourceMap) -> ExitCode {
    if JSON.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("{}", diags_json(diags, sources));
        return ExitCode::FAILURE;
    }
    let renderer = PlainRenderer;
    for d in diags {
        eprint!("{}", renderer.render(d, sources));
    }
    ExitCode::FAILURE
}

/// `--json` on a compile (`viv <file> --json`): diagnostics go to stderr as
/// JSON. Set once, from the arguments, before anything is reported.
static JSON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `--split <dir>` on a compile: the C as separate translation units
/// (`write_split`), for `viv build`'s object cache.
static SPLIT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// A unit's file name: its module path with `/` as `__` and no extension,
/// and a part's number after `__part` (`lib/hf/mdg_kernels.viv#2`).
fn unit_slug(name: &str) -> String {
    let (module, part) = name.split_once('#').unwrap_or((name, ""));
    let base = module.trim_end_matches(".viv").replace(['/', '\\'], "__").replace("..", "_");
    if part.is_empty() { base } else { format!("{base}__part{part}") }
}

/// The source bytes of a module that go into one unit before the next part
/// begins (`VIV_UNIT_BYTES` overrides it). C compile time grows with a
/// unit's size and a unit compiles on one core.
const UNIT_BYTES: usize = 160_000;

/// `viv f.viv --split <dir>`: the program's C as translation units.
/// Each imported module's defs become a
/// unit of their own, `<dir>/units/<slug>.c`, whose text depends only on
/// those defs and what they use, so `viv build` compiles each distinct text
/// once and keeps the object; the program's own defs, `main`, the globals and
/// the runtime's state are `<dir>/main.c`. `<dir>/units.txt` lists the units,
/// `slug<TAB>module`. `VIV_SPLIT_UNITS=<module,...>` limits the units to
/// those modules (the others stay in main.c). A module whose C would use the
/// program's enum declarations stays in main.c too.
fn write_split(dir: &std::path::Path, sources: &SourceMap, src: &str, source_id: knotkit::span::SourceId) -> ExitCode {
    let only: Option<Vec<String>> = std::env::var("VIV_SPLIT_UNITS").ok().filter(|v| !v.trim().is_empty()).map(|v| v.split(',').map(|x| x.trim().to_string()).collect());
    let own = sources.file(source_id).map(|f| f.name.clone()).unwrap_or_default();
    let unit_bytes: usize = std::env::var("VIV_UNIT_BYTES").ok().and_then(|v| v.parse().ok()).filter(|n: &usize| *n > 0).unwrap_or(UNIT_BYTES);
    let mut refused: Vec<String> = Vec::new();
    loop {
        let res = viv::lower::with_sites_map(sources.clone(), src, || {
            let program = viv::build_ir(src, source_id)?;
            let module_of = |root: knotkit::ir::NodeId| -> Option<String> {
                let node = program.get(root);
                if !matches!(node.kind, viv::lower::Viv::Func { .. }) {
                    return None;
                }
                let frame = sources.resolve(node.span).last()?.clone();
                let name = sources.file(frame.span.source)?.name.clone();
                if name == own || refused.contains(&name) {
                    return None;
                }
                if let Some(only) = &only {
                    if !only.iter().any(|m| m == &name || m.trim_end_matches(".viv") == name.trim_end_matches(".viv")) {
                        return None;
                    }
                }
                Some(name)
            };
            // a large module is several units of about UNIT_BYTES of source
            // each, in its own order, so that a cold build compiles its parts
            // at once (lib/hf/mdg_kernels: four)
            let mut parts: std::collections::HashMap<knotkit::ir::NodeId, String> = std::collections::HashMap::new();
            let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for &root in program.roots() {
                let Some(name) = module_of(root) else { continue };
                let span = program.get(root).span;
                let at = seen.entry(name.clone()).or_insert(0);
                let part = *at / unit_bytes;
                *at += (span.end - span.start) as usize;
                parts.insert(root, if part == 0 { name } else { format!("{name}#{part}") });
            }
            let unit_of = |root: knotkit::ir::NodeId| -> Option<String> { parts.get(&root).cloned() };
            CBackend.emit_split(&program, &unit_of)
        });
        match res {
            Ok((main_c, units)) => {
                let udir = dir.join("units");
                if let Err(e) = std::fs::create_dir_all(&udir) {
                    eprintln!("error: cannot make {}: {e}", udir.display());
                    return ExitCode::FAILURE;
                }
                let mut list = String::new();
                let write = |p: PathBuf, t: &str| std::fs::write(&p, t).map_err(|e| format!("error: cannot write {}: {e}", p.display()));
                if let Err(e) = write(dir.join("main.c"), &main_c) {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
                for u in &units {
                    let slug = unit_slug(&u.name);
                    if let Err(e) = write(udir.join(format!("{slug}.c")), &u.text) {
                        eprintln!("{e}");
                        return ExitCode::FAILURE;
                    }
                    list.push_str(&format!("{slug}\t{}\n", u.name));
                }
                if let Err(e) = write(dir.join("units.txt"), &list) {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
                println!("wrote {} ({} units{})", dir.display(), units.len(), if refused.is_empty() { String::new() } else { format!("; kept in main.c: {}", refused.join(", ")) });
                return ExitCode::SUCCESS;
            }
            Err(diags) => {
                // a unit that names the program's enum declarations: keep its
                // module in main.c and emit again
                let named = diags.iter().find_map(|d| d.message.strip_prefix("the unit `").and_then(|r| r.split('`').next()).map(|u| u.split('#').next().unwrap_or(u).to_string()));
                match named {
                    Some(n) if !refused.contains(&n) => refused.push(n),
                    _ => return render_diags(&diags, sources),
                }
            }
        }
    }
}

/// Diagnostics as one JSON object on one line, for tools and models:
///
///     {"diagnostics":[{"severity":"error","message":"...","file":"f.viv",
///       "line":3,"col":7,"end_line":3,"end_col":12,"label":"here",
///       "source":"show a[-1]","notes":["..."]}]}
///
/// Lines and columns are 1-based, columns count bytes, and `end_col` is one
/// past the last character the label covers. A diagnostic with no location
/// (the failed-assert summary, say) has null for file, line and col.
fn diags_json(diags: &[knotkit::diag::Diagnostic], sources: &SourceMap) -> String {
    diags_json_kind(diags, sources, None)
}

/// `{"diagnostics":[...]}`, the form `check --json` and `run --json` wrote to
/// stderr before the envelope (VIV_JSON_STDERR=1), with rule ids and fixes.
fn diags_json_kind(diags: &[knotkit::diag::Diagnostic], sources: &SourceMap, kind: Option<viv::errkind::ErrKind>) -> String {
    format!(r#"{{"diagnostics":{}}}"#, diags_array(diags, sources, kind).0)
}

/// The input path as the command line gave it, so a diagnostic names the
/// file by that path rather than the base name the engines use
/// (`display_path`).
static GIVEN_PATH: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

fn set_given_path(path: &str) {
    if let Ok(mut g) = GIVEN_PATH.lock() {
        *g = Some(path.to_string());
    }
}

/// A source map's file name as the command line gave it.
fn given_name(file: &str) -> String {
    match GIVEN_PATH.lock().ok().and_then(|g| g.clone()) {
        Some(p) if file == display_path(&p) => p,
        _ => file.to_string(),
    }
}

/// The diagnostics as a JSON array, each with its rule id (agent::rule_of)
/// and, for a hint whose edit is certain, a fix; and the rules in order.
/// `kind` is the run-time error kind of a stopped run.
fn diags_array(diags: &[knotkit::diag::Diagnostic], sources: &SourceMap, kind: Option<viv::errkind::ErrKind>) -> (String, Vec<&'static str>) {
    use knotkit::diag::Severity;
    use viv::debug_lens::json_str;
    let mut rules = Vec::new();
    let mut one = |d: &knotkit::diag::Diagnostic| -> String {
        let severity = match d.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        let rule = agent::rule_of(&d.message, &d.notes, kind);
        rules.push(rule);
        let mut at = String::from(r#""file":null,"line":null,"col":null"#);
        let mut fix = String::new();
        if let Some(label) = &d.primary {
            // where the text came from: a module's own line, a Python line
            // (knotkit's origin chain); a program with neither is its own origin
            let chain = sources.resolve(label.span);
            let span = chain.last().map_or(label.span, |f| f.span);
            if let Some((line, col, text)) = sources.locate(span.source, span.start) {
                let text = text.to_string();
                let (end_line, end_col, _) = sources.locate(span.source, span.end).unwrap_or((line, col, ""));
                let file = given_name(sources.file(span.source).map_or("<source>", |f| f.name.as_str()));
                at = format!(
                    r#""file":{},"line":{line},"col":{col},"end_line":{end_line},"end_col":{end_col},"label":{},"source":{}"#,
                    json_str(&file),
                    json_str(&label.message),
                    json_str(&text)
                );
                // the generators on the way ("via py2viv: p.py.viv:40:5"),
                // only when there were any, so other diagnostics read as before
                let via = knotkit::diag::via_notes(&chain, sources);
                if !via.is_empty() {
                    at.push_str(&format!(r#","via":[{}]"#, via.iter().map(|v| json_str(v)).collect::<Vec<_>>().join(",")));
                }
                if end_line == line {
                    if let Some(f) = agent::fix_of(rule, &file, line as usize, col as usize, end_col as usize, &text, &d.notes) {
                        fix = format!(r#","fix":{f}"#);
                    }
                }
            }
        }
        let notes = d.notes.iter().map(|n| json_str(n)).collect::<Vec<_>>().join(",");
        format!(r#"{{"severity":"{severity}","rule":"{rule}","message":{},{at},"notes":[{notes}]{fix}}}"#, json_str(&d.message))
    };
    let items: Vec<String> = diags.iter().map(&mut one).collect();
    (format!("[{}]", items.join(",")), rules)
}

/// The exit code of a command that reported these diagnostics:
/// 0 with none.
fn exit_of(diags: &[knotkit::diag::Diagnostic], kind: Option<viv::errkind::ErrKind>) -> u8 {
    if diags.is_empty() {
        return agent::OK;
    }
    let rules: Vec<&str> = diags.iter().map(|d| agent::rule_of(&d.message, &d.notes, kind)).collect();
    agent::exit_for_rules(rules.into_iter())
}

/// Keep the first diagnostics of a JSON array under the budget.
fn bound_diags(arr: String, budget: &agent::Budget) -> (String, bool) {
    match viv::story::parse_json(&arr) {
        Ok(j) => {
            let (j, cut) = agent::bound_json(viv::story::Json::Obj(vec![("d".into(), j)]), &agent::Budget { summary: false, max_lines: if budget.summary { agent::SUMMARY_ITEMS } else { budget.max_lines }, ..*budget });
            (j.get("d").map(|d| d.text()).unwrap_or(arr), cut)
        }
        Err(_) => (arr, false),
    }
}

/// A usage error, or a failure before the program loads: the envelope under
/// --json, the message otherwise; the exit code from the rule.
fn usage_failed(cmd: &str, msg: &str, rule: &str, json: bool) -> ExitCode {
    let code = agent::exit_for_rule(rule);
    if json && !agent::legacy_stderr() {
        return agent::print_envelope(cmd, code, &format!("[{}]", agent::plain_diag(msg, rule)), "null", false, None);
    }
    fail_plain(msg, json);
    ExitCode::from(code)
}

/// A program that did not load (a file that cannot be read, a bad --set, a
/// module or Python translation that fails): reported as before, or as the
/// envelope, with the exit code of its rules.
fn load_failed(e: &LoadError, cmd: &str, json: bool, set_prefix: &str) -> ExitCode {
    let plain_rule = |m: &str| if m.contains("cannot read") { "U-file" } else { "E-check" };
    if json && !agent::legacy_stderr() {
        let (arr, rules): (String, Vec<&str>) = match e {
            LoadError::Plain(m) => (format!("[{}]", agent::plain_diag(m, plain_rule(m))), vec![plain_rule(m)]),
            LoadError::Set(m) => (format!("[{}]", agent::plain_diag(&format!("{set_prefix}{m}"), "U-set")), vec!["U-set"]),
            LoadError::Located(ds, sources) => diags_array(ds, sources, None),
        };
        return agent::print_envelope(cmd, agent::exit_for_rules(rules.into_iter()), &arr, "null", false, None);
    }
    let _ = e.report(json, set_prefix);
    ExitCode::from(match e {
        LoadError::Plain(m) => agent::exit_for_rule(plain_rule(m)),
        LoadError::Set(_) => agent::USAGE,
        LoadError::Located(ds, _) => exit_of(ds, None),
    })
}

/// `viv run --json`: the program's printing (bounded) and its test counts in
/// `result`, the diagnostics beside them, on stdout. The explorer's lines
/// (`__graph__`, `__trace__`, ...) are left out, as `--quiet` leaves them out.
fn run_envelope(path: &str, run: viv::interp::Run, sources: &SourceMap, budget: &agent::Budget) -> ExitCode {
    use viv::debug_lens::json_str;
    let out: String = run
        .out
        .lines()
        .filter(|l| !(l.starts_with("__graph__") || l.starts_with("__trace__") || l.starts_with("__steps__") || l.starts_with("__story__")))
        .map(|l| format!("{l}\n"))
        .collect();
    let mut diags = run.errors;
    if diags.is_empty() {
        if run.holds > 0 {
            diags.push(knotkit::diag::Diagnostic::error(format!("{} hold{} FAILED", run.holds, if run.holds == 1 { "" } else { "s" })));
        }
        if run.fails > 0 {
            diags.push(knotkit::diag::Diagnostic::error(format!("{} assertion{} FAILED", run.fails, if run.fails == 1 { "" } else { "s" })));
        }
    }
    let (arr, rules) = diags_array(&diags, sources, run.kind);
    let exit = if diags.is_empty() { agent::OK } else { agent::exit_for_rules(rules.iter().copied()) };
    // a program that does not compile never ran: no result
    let result = if exit == agent::COMPILE {
        "null".to_string()
    } else {
        let pass = out.lines().filter(|l| l.starts_with("PASS ")).count();
        let failed: Vec<String> = out.lines().filter_map(|l| l.strip_prefix("FAIL ")).map(|n| json_str(n.split(" -- ").next().unwrap_or(n))).collect();
        let (shown, cut, n) = agent::bound_text(&out, budget, true);
        let r = format!(
            r#"{{"file":{},"stdout":{},"stdout_lines":{n},"tests":{{"pass":{pass},"fail":{},"failed":[{}]}},"holds_failed":{}}}"#,
            json_str(path),
            json_str(&shown),
            failed.len(),
            failed.join(","),
            run.holds
        );
        if cut {
            let more = if budget.summary { "the last 20 lines are shown: drop --summary, or --max-lines 0 for all".to_string() } else { format!("the last {} of {n} lines are shown: --max-lines 0 for all", budget.max_lines) };
            return agent::print_envelope("run", exit, &arr, &r, true, Some(&more));
        }
        r
    };
    agent::print_envelope("run", exit, &arr, &result, false, None)
}

/// Print an IR subtree: one node per line, indented by depth, showing the
/// short structural hash (the content address) and the node kind.
fn print_ir(prog: &knotkit::ir::IrProgram<viv::Viv>, id: knotkit::ir::NodeId, depth: usize) {
    let node = prog.get(id);
    let mut kind = format!("{:?}", node.kind);
    // A def's `src` (the name its step events use) is shown only where it
    // differs from the def's own name, in a specialization, so the dump of a
    // program reads as it did before the field existed.
    if let viv::lower::Viv::Func { name, src, .. } = &node.kind {
        if name == src {
            kind = kind.replacen(&format!(", src: {:?}", src), "", 1);
        }
    }
    // An index node's `at` is 0 outside a C compile (it is set only by
    // `lower::with_sites`); the dump leaves it out, as it did before the field.
    let kind = kind.replace("Index { at: 0 }", "Index").replace(", at: 0 }", " }");
    println!("{}{}  {}", "  ".repeat(depth), id.short(), kind);
    for child in node.children.clone() {
        print_ir(prog, child, depth + 1);
    }
}

// ---------------------------------------------------------------------------
// `viv hash` / `viv diff`: definition-level content addresses.
//
// The IR store already gives every node a BLAKE3 structural identity. A def's
// *node* hash includes its own name (the name is payload); its *body* hash —
// BLAKE3 over the ordered child NodeIds — deliberately does not. So:
//   node hash changes  ⇐  anything about the def changed (incl. its name)
//   body hash changes  ⇐  the def's CODE changed (name is irrelevant)
// Formatting and comments were never hashed at all (spans are metadata), so
// reformatting is invisible to both. `viv diff` exploits the split: same
// name + same body = unchanged; same name + new body = edited; new name +
// same body = RENAMED — detected with zero heuristics.
// ---------------------------------------------------------------------------





// ---------------------------------------------------------------------------
// The lens tools. lens.rs does the work on texts; this part reads the files,
// finds the modules a program uses and their `.intent` files, and looks in
// git for the version of a def an intent entry was stamped against.
// ---------------------------------------------------------------------------

const LENS_USAGE: &str = "usage: viv lens init <file.viv> [--force] [--stdout]   (write <file>.intent: every def, step and hash, no text)
       viv lens check <file.viv>... [--json]            (staleness, coverage, steps, one home per anchor, cited numbers)
       viv lens stamp <file.viv> <def>                  (the entry still holds: record the def's current body hash)
       viv lens coverage [<file.viv|dir>...] [--json] [--check|--update]  (explained defs, steps, glossary; the ratchet)
       viv lens needed [<file.viv|dir>...] [--program <file.viv>]         (dead defs; `serves` of nothing)
       viv debt [<file.viv|dir>...] [--limit <n>] [--run <program.viv>]  (what is missing, stale or dead, the most central first)
       viv lens move <file.viv> --to-file|--to-inline [<def>...]
       viv view <file.viv> [--lens code|intent|pseudocode|debug|story|memory|all] [--def d] [--words-only] [--json]
       viv explain <file.viv> [--from <def>] [--depth <k>] [--json]
       viv story <file.viv> [--depth <k>] [--expand] [--json] [--engine interp|c|fast] [--from <run output>]
       viv arch check|report ...                         (the architecture lens: layers, budgets, multi-paths; `viv arch` for its usage)";

/// The files a source text's import lines (and `#! use` lines) name, in
/// order: `unit` is the text's module name, empty for a program.
fn use_paths(unit: &str, src: &str, here: &std::path::Path, root: &std::path::Path) -> Vec<PathBuf> {
    viv::imports::module_refs(unit, src).into_iter().filter_map(|(cands, _)| cands.iter().find_map(|c| module_path(here, root, c).ok())).fold(Vec::new(), |mut v, p| {
        if !v.contains(&p) {
            v.push(p);
        }
        v
    })
}

fn intent_path_of(p: &std::path::Path) -> PathBuf {
    p.with_extension("intent")
}

fn read_opt(p: &std::path::Path) -> Option<String> {
    if viv::trace::exists(p) { TracedIo.read_to_string(p).ok() } else { None }
}

/// NativeIo, with every read noted for the gate's cache (viv::trace).
struct TracedIo;

impl TracedIo {
    fn read_to_string(&self, p: &std::path::Path) -> std::io::Result<String> {
        let r = NativeIo.read_to_string(p);
        viv::trace::note(p, r.is_ok());
        r
    }
}

/// A program and everything the lens tools need about it: its own text and
/// intent file, every module its `#! use` lines reach with theirs, and its
/// expanded source.
fn lens_input(path: &str) -> Result<viv::lens::Input, String> {
    let src = TracedIo.read_to_string(&PathBuf::from(path)).map_err(|e| format!("error: cannot read {}: {}", path, e))?;
    let expanded = expand_uses_from(&src, path)?;
    let (here, root) = module_dirs(path);
    // the module's name: its path under the repository, without `.viv`
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    let stem = || PathBuf::from(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let name = if root.join("Cargo.toml").exists() {
        abs.strip_prefix(&root).map(|r| r.with_extension("").display().to_string()).unwrap_or_else(|_| stem())
    } else {
        stem()
    };
    let ip = intent_path_of(&PathBuf::from(path));
    let root_mod = viv::lens::ModuleText { name, path: path.to_string(), src: src.clone(), intent_path: ip.display().to_string(), intent: read_opt(&ip) };
    // the modules used, in the order the expansion first splices them; the
    // expansion decides, so a module an `#! if-none-used` block leaves out
    // is not counted
    let mut used: Vec<viv::lens::ModuleText> = Vec::new();
    let exp = viv::expand_uses_mapped(&src, &mut |name: &str| {
        let c = module_path(&here, &root, name)?;
        TracedIo.read_to_string(&c).map_err(|e| format!("error: cannot read {}: {}", c.display(), e))
    }, &mut |lang, name, _| Err(format!("module `{name}`: it is written in `{lang}`, and nothing here can translate that")))?;
    for u in exp.units.iter().skip(1) {
        let p = module_path(&here, &root, &u.name)?;
        let ip = intent_path_of(&p);
        let text = TracedIo.read_to_string(&p).map_err(|e| format!("error: cannot read {}: {}", p.display(), e))?;
        used.push(viv::lens::ModuleText { name: u.name.clone(), path: p.display().to_string(), src: text, intent_path: ip.display().to_string(), intent: read_opt(&ip) });
    }
    // the companion test file, `<dir>/tests/<name>.viv`: its tests may be cited
    let companion = PathBuf::from(path).parent().map(|d| d.join("tests").join(PathBuf::from(path).file_name().unwrap_or_default()));
    let companion_tests = companion.and_then(|p| read_opt(&p)).map(|t| viv::lens::test_names_of(&t)).unwrap_or_default();
    Ok(viv::lens::Input { root: root_mod, used, expanded, companion_tests })
}

/// Finds, in the file's git history, the source of a def as of a commit whose
/// body hash for it was the stamped one. Each commit is hashed once.
struct GitHistory {
    path: String,
    commits: Option<Vec<String>>,
    hashed: std::collections::HashMap<String, Option<(String, Vec<viv::DefHash>)>>,
}

impl GitHistory {
    fn git(dir: &std::path::Path, args: &[&str]) -> Option<String> {
        let out = std::process::Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
    }
    fn find(&mut self, def: &str, hash: &str) -> Option<String> {
        let p = std::fs::canonicalize(&self.path).ok()?;
        let dir = p.parent()?.to_path_buf();
        let top = PathBuf::from(Self::git(&dir, &["rev-parse", "--show-toplevel"])?.trim());
        let rel = p.strip_prefix(std::fs::canonicalize(&top).ok()?).ok()?.display().to_string();
        if self.commits.is_none() {
            let log = Self::git(&top, &["log", "--format=%H", "-n", "50", "--", &rel]).unwrap_or_default();
            self.commits = Some(log.lines().map(String::from).collect());
        }
        for c in self.commits.clone().unwrap_or_default() {
            let entry = self.hashed.entry(c.clone()).or_insert_with(|| {
                let text = Self::git(&top, &["show", &format!("{c}:{rel}")])?;
                let expanded = expand_uses_from(&text, &self.path).ok()?;
                let hs = viv::def_hashes(&expanded).ok()?;
                Some((text, hs))
            });
            if let Some((text, hs)) = entry {
                if hs.iter().any(|h| h.name == def && h.body_short == hash) {
                    return viv::lens::def_text(text, def);
                }
            }
        }
        None
    }
}

fn write_file(path: &str, text: &str) -> Result<(), String> {
    NativeIo.write(&PathBuf::from(path), text.as_bytes()).map_err(|e| format!("error: cannot write {}: {}", path, e))
}

/// `viv view --lens debug`: the debug lens of the program and the modules it
/// uses, inline blocks and `.debug` files alike, in the `.debug` layout.
/// `viv view --lens memory`: the memory facts of the program and the modules
/// it uses, each def located in the file it was written in.
fn memory_view(path: &str, def: Option<&str>, json: bool) -> Result<String, String> {
    let l = load(path, true, "", &[]).map_err(|e| e.text())?;
    let sources = l.sources.clone();
    let locate = move |span: knotkit::span::Span| -> String {
        let at = sources.origin_of(span);
        match (sources.file(at.source), sources.locate(at.source, at.start)) {
            (Some(f), Some((line, _, _))) => format!("{}:{line}", f.name),
            _ => String::new(),
        }
    };
    let mut facts = viv::memory_view::facts(&l.text, &locate)?;
    if let Some(d) = def {
        if !facts.iter().any(|m| m.name == d) {
            return Err(format!("{path} has no def `{d}`"));
        }
    } else {
        // the file's own defs (and its top level), not its modules'
        let own = format!("{path}:");
        facts.retain(|m| m.at.is_empty() || m.at.starts_with(&own));
        if !l.text.lines().any(|x| !x.starts_with(' ') && !x.is_empty()) {
            facts.retain(|m| !m.at.is_empty());
        }
    }
    Ok(if json { format!("{}\n", viv::memory_view::json(&facts, def)) } else { viv::memory_view::text(&facts, def) })
}

fn debug_view(path: &str) -> Result<String, String> {
    let src = read_program(path)?;
    let prog = viv::parse(&src, knotkit::span::SourceId(0)).map_err(|ds| ds.iter().map(|d| format!("error: {}", d.message)).collect::<Vec<_>>().join("\n"))?;
    Ok(viv::debug_lens::view(&prog, &src))
}

#[derive(Default)]
struct StoryArgs {
    depth: Option<usize>,
    expand: bool,
    json: bool,
    engine: Option<String>,
    from: Option<String>,
}

/// Run a program in debug mode with the story on, in the interpreter or
/// compiled to C, and return its output.
fn story_run(src: &str, engine: &str) -> Result<String, String> {
    match engine {
        "interp" => viv::interp::run_checked_holds(src, false).map(|(out, _, _)| out).map_err(|e| format!("error: {e}")),
        // `fast`: the fast build; its story says so
        "c" | "fast" => {
            let c = viv::compile_c(src)?;
            let dir = std::env::temp_dir().join(format!("viv-story-{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(|e| format!("error: cannot make {}: {e}", dir.display()))?;
            let (cfile, bin) = (dir.join("story.c"), dir.join("story"));
            std::fs::write(&cfile, c).map_err(|e| format!("error: cannot write {}: {e}", cfile.display()))?;
            let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
            let flags = fast_cli::cc_flags(if engine == "fast" { fast_cli::Mode::Fast } else { fast_cli::Mode::Exact });
            let built = std::process::Command::new(&cc).args(&flags).arg("-o").arg(&bin).arg(&cfile).arg("-lm").output().map_err(|e| format!("error: cannot run {cc}: {e}"))?;
            if !built.status.success() {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(format!("error: {cc} rejected the generated C:\n{}", String::from_utf8_lossy(&built.stderr)));
            }
            let ran = std::process::Command::new(&bin).output().map_err(|e| format!("error: cannot run the compiled program: {e}"));
            let _ = std::fs::remove_dir_all(&dir);
            Ok(String::from_utf8_lossy(&ran?.stdout).to_string())
        }
        other => Err(format!("unknown engine `{other}`: expected interp, c or fast")),
    }
}

/// The story of one run, rendered, and whether every hold passed.
fn story_text(path: &str, a: &StoryArgs) -> Result<(String, bool), String> {
    let out = match &a.from {
        Some(f) => TracedIo.read_to_string(&PathBuf::from(f)).map_err(|e| format!("error: cannot read {f}: {e}"))?,
        None => {
            // `debug * { }` turns the story on for a program without a debug lens
            let src = format!("{}\ndebug * {{\n}}\n", read_program(path)?);
            story_run(&src, a.engine.as_deref().unwrap_or("interp"))?
        }
    };
    let line = viv::story::story_line(&out).ok_or_else(|| "error: the run printed no story (was it a release run?)".to_string())?;
    let story = viv::story::parse(line)?;
    // the intent's words; a program whose lens does not load is told by its ids
    let words = lens_input(path).and_then(|i| viv::lens::load(&i)).map(|l| viv::lens::story_words(&l)).unwrap_or_default();
    let ok = story.hold_counts().1 == 0;
    let text = if a.json {
        format!("{}\n", viv::story::render_json(&story, &words))
    } else {
        viv::story::render_text(&story, &words, viv::story::Opts { depth: a.depth, expand: a.expand })
    };
    Ok((text, ok))
}

fn graph_cli(args: &[String]) -> ExitCode {
    const USAGE: &str = "usage: viv graph <file.viv> [--json | --bundle | --run | --map [--entry <def>] | --svg [-o out.svg] | --records]";
    let out_file = args.iter().position(|a| a == "-o").and_then(|k| args.get(k + 1)).cloned();
    let entry = args.iter().position(|a| a == "--entry").and_then(|k| args.get(k + 1)).cloned();
    let Some(path) = args.iter().skip(2).find(|a| !a.starts_with('-') && Some(*a) != out_file.as_ref() && Some(*a) != entry.as_ref()) else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    // `--svg`: the whole program as a picture, drawn by lib/graphdraw.viv in
    // the interpreter (`viv::graphlit::draw_records`) and made SVG by `viv plot`
    if args.iter().any(|a| a == "--svg" || a == "--records") {
        return match graph_drawing(path) {
            Ok(records) => {
                let text = if args.iter().any(|a| a == "--records") { records } else { viv::figsvg::render_page(&records, 640.0) };
                match &out_file {
                    Some(p) => match std::fs::write(p, text) {
                        Ok(()) => ExitCode::SUCCESS,
                        Err(e) => {
                            eprintln!("viv graph: {p}: {e}");
                            ExitCode::FAILURE
                        }
                    },
                    None => {
                        print!("{text}");
                        ExitCode::SUCCESS
                    }
                }
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
    let run = || -> Result<String, String> {
        let input = lens_input(path)?;
        // `--map`: every def the entry reaches, by module, with its call
        // sites (lang/viv/src/codemap.rs); the vipyr code maps
        if args.iter().any(|a| a == "--map") {
            return viv::codemap::map_json(&input, entry.as_deref()).map(|j| format!("{j}\n"));
        }
        // the debug lens of the program and of each module it uses, in the
        // order `read_program` appends them
        let mut debug = debug_source(&PathBuf::from(path))?;
        for m in &input.used {
            debug.push_str(&debug_source(&PathBuf::from(&m.path))?);
        }
        if args.iter().any(|a| a == "--bundle") {
            return viv::graph::bundle_json(&input, &debug).map(|b| format!("{b}\n"));
        }
        let program = viv::graph::explore_program(&input.expanded, &debug);
        if args.iter().any(|a| a == "--run") {
            // the run the explorer records: interpreter, debug mode, lines on
            return viv::graph::explore_run(&program, &Default::default()).map_err(|e| format!("error: {e}")).map(|(out, _, _)| out);
        }
        let g = viv::graph::graph_of_source(&program)?;
        if args.iter().any(|a| a == "--json") {
            let table = viv::graph::lines_table(&program);
            let rows = table.len();
            Ok(format!("{{\"name\":\"{}\",{},\"lineCount\":{rows}}}\n", input.root.name.replace('\\', "/").replace('"', ""), viv::graph::nodes_edges_json(&g)))
        } else {
            Ok(viv::graph::graph_text(&g))
        }
    };
    match run() {
        Ok(t) => {
            print!("{t}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// The step budget of `viv atlas`'s one run, a twentieth of `viv run`'s: a
/// hung run stops in seconds and the atlas shows the calls the text makes.
/// The explorer's programs (www/src/data/explorables.mjs) fit well inside it.
const ATLAS_STEP_BUDGET: u64 = 100_000_000;

fn atlas_cli(args: &[String]) -> ExitCode {
    const USAGE: &str = "usage: viv atlas <file.viv> [-o <file.atlas.html>] [--text | --json] [--no-run] [--no-walk]";
    let out_file = args.iter().position(|a| a == "-o").and_then(|k| args.get(k + 1)).cloned();
    let Some(path) = args.iter().skip(2).find(|a| !a.starts_with('-') && Some(*a) != out_file.as_ref()).cloned() else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    if let Some(bad) = args.iter().skip(2).find(|a| a.starts_with('-') && !["-o", "--text", "--json", "--no-run", "--no-walk"].contains(&a.as_str())) {
        eprintln!("viv atlas: no option {bad}\n{USAGE}");
        return ExitCode::FAILURE;
    }
    let as_text = args.iter().any(|a| a == "--text");
    let as_json = args.iter().any(|a| a == "--json");
    let no_run = args.iter().any(|a| a == "--no-run");
    let no_walk = args.iter().any(|a| a == "--no-walk");
    let made = || -> Result<String, String> {
        let input = lens_input(&path)?;
        let mut debug = debug_source(&PathBuf::from(&path))?;
        for m in &input.used {
            debug.push_str(&debug_source(&PathBuf::from(&m.path))?);
        }
        let map = viv::atlas::portable(&viv::codemap::map_json(&input, None)?)?;
        // one run, as www/scripts/explore-build.mjs makes it: the explorer's
        // program, released, for its call graph
        let calls = if no_run {
            None
        } else {
            // a program that loops forever or runs long must not hang the
            // command: the run says it has started, and it stops past
            // ATLAS_STEP_BUDGET steps (VIV_STEP_BUDGET raises it)
            if std::env::var_os("VIV_STEP_BUDGET").is_none() {
                viv::interp::set_step_budget(ATLAS_STEP_BUDGET);
            }
            eprintln!("viv atlas: running {path} once for its call counts (--no-run skips the run)");
            let program = viv::graph::explore_program(&input.expanded, &debug);
            match viv::interp::run_release(&program) {
                Ok(out) => match viv::atlas::calls_of(&out) {
                    Some(c) => Some(c),
                    None => {
                        eprintln!("viv atlas: the run printed no call graph; the atlas shows the calls the text makes");
                        None
                    }
                },
                Err(e) => {
                    eprintln!("viv atlas: the run stopped ({}); the atlas shows the calls the text makes", e.lines().next().unwrap_or(""));
                    None
                }
            }
        };
        if as_text {
            return viv::atlas::text(&map, calls.as_ref());
        }
        if as_json {
            let bundle = viv::graph::bundle_json(&input, &debug)?;
            return Ok(viv::atlas::json(&bundle, &map, calls.as_ref()));
        }
        let svg = graph_drawing(&path).map(|records| viv::figsvg::render_page(&records, 640.0));
        let name = input.root.name.replace('\\', "/");
        // the walk: the run recorded as the site's walk records it
        // (viv::atlas_walk), when the run finished. The bundle carries the
        // run's calls as explore-build.mjs's does: the defs a `parallel for`
        // reaches through them stay untraced (lib/zoomwalk.js parallelReached)
        let bundle = viv::graph::bundle_json(&input, &debug)?;
        let bundle = match &calls {
            Some(c) => {
                let b = bundle.trim_end();
                format!("{},\"calls\":{}}}", b.strip_suffix('}').unwrap_or(b), viv::atlas::calls_json(c))
            }
            None => bundle,
        };
        let recorded = if no_walk {
            Err("No walk: viv atlas was asked for none (--no-walk).".to_string())
        } else if calls.is_none() {
            Err(if no_run { "No walk: viv atlas did not run the program (--no-run).".to_string() } else { "No walk: the run did not finish, so there is nothing to walk through.".to_string() })
        } else {
            eprintln!("viv atlas: recording the walk, the run line by line (--no-walk skips it)");
            viv::atlas_walk::record_walk(&bundle, viv::atlas::walk_cap()).map_err(|e| format!("No walk: the run could not be recorded ({e})."))
        };
        let recorded = recorded.and_then(|w| viv::atlas::walk_fits(&w, viv::atlas::walk_cap()).map(|()| w));
        if let Err(why) = &recorded {
            if !no_walk && calls.is_some() {
                eprintln!("viv atlas: {why}");
            }
        }
        let walk: viv::atlas::WalkData = match &recorded {
            Ok(w) => Ok((bundle.as_str(), w)),
            Err(e) => Err(e.clone()),
        };
        Ok(viv::atlas::html(&name, &map, calls.as_ref(), svg.as_deref().map_err(|e| e.as_str()), &walk))
    };
    let text = match made() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    // the page goes to a file: -o, or <stem>.atlas.html here
    let target = out_file.or_else(|| (!as_text && !as_json).then(|| format!("{}.atlas.html", PathBuf::from(&path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "program".into()))));
    match target {
        Some(p) => match std::fs::write(&p, &text) {
            Ok(()) => {
                if !as_text && !as_json {
                    println!("wrote {p} ({} KB): open it in a browser; it needs no network", text.len() / 1024);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("viv atlas: {p}: {e}");
                ExitCode::FAILURE
            }
        },
        None => {
            print!("{text}");
            ExitCode::SUCCESS
        }
    }
}

/// The figure records of `path`'s drawing. The program is the module named by
/// its path without `.viv`; the drawing's own modules (lib/graphdraw and what
/// it uses) are read beside the program's repository, or, for a program
/// outside one, from the repository this compiler was built in.
fn graph_drawing(path: &str) -> Result<String, String> {
    let text = TracedIo.read_to_string(&PathBuf::from(path)).map_err(|e| format!("error: cannot read {path}: {e}"))?;
    let (here, root) = module_dirs(path);
    let name = display_path(path).trim_end_matches(".viv").replace('\\', "/");
    // a release build (VIV_RELEASE set when it is compiled, as for vipyr's
    // wheel) carries no path of the machine it was built on: its library is
    // the root's (VIV_ROOT)
    let home = if option_env!("VIV_RELEASE").is_some() { root.clone() } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..") };
    let target = name.clone();
    let resolve: viv::graphlit::Resolver = std::sync::Arc::new(move |m: &str| {
        if m == target {
            return Ok(text.clone());
        }
        let c = module_path(&here, &root, m).or_else(|_| module_path(&home, &home, m))?;
        TracedIo.read_to_string(&c).map_err(|e| format!("cannot read {}: {}", c.display(), e))
    });
    viv::graphlit::draw_records(&name, resolve)
}

fn story_cli(args: &[String]) -> ExitCode {
    const USAGE: &str = "usage: viv story <file.viv> [--depth <k>] [--expand] [--json] [--engine interp|c|fast] [--from <run output>]";
    let mut a = StoryArgs::default();
    let mut path: Option<String> = None;
    let mut k = 2;
    while k < args.len() {
        match args[k].as_str() {
            "--depth" => {
                match args.get(k + 1).map(|d| d.parse::<usize>()) {
                    Some(Ok(d)) => a.depth = Some(d),
                    _ => {
                        eprintln!("--depth takes a whole number");
                        return ExitCode::FAILURE;
                    }
                }
                k += 1;
            }
            "--engine" => {
                a.engine = args.get(k + 1).cloned();
                k += 1;
            }
            "--from" => {
                a.from = args.get(k + 1).cloned();
                k += 1;
            }
            "--expand" => a.expand = true,
            "--json" => a.json = true,
            other if !other.starts_with("--") && path.is_none() => path = Some(other.to_string()),
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::FAILURE;
            }
        }
        k += 1;
    }
    let Some(path) = path else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    match story_text(&path, &a) {
        Ok((text, ok)) => {
            print!("{text}");
            if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn lens_cli(args: &[String]) -> ExitCode {
    let fail = |m: String| {
        eprintln!("{m}");
        ExitCode::FAILURE
    };
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|k| args.get(k + 1)).cloned();
    // positional arguments after the verb (and after the sub-verb for `lens`)
    let skip = if args[1] == "lens" { 3 } else { 2 };
    let mut pos: Vec<String> = Vec::new();
    let mut k = skip;
    while k < args.len() {
        let a = &args[k];
        if a == "--lens" || a == "--from" || a == "--depth" {
            k += 2;
            continue;
        }
        if !a.starts_with("--") {
            pos.push(a.clone());
        }
        k += 1;
    }
    let load = |path: &str| -> Result<viv::lens::Loaded, String> { viv::lens::load(&lens_input(path)?) };
    // every verb but `check` (which reports them as findings) stops at an
    // `.intent` file it cannot parse, rather than show a lens without it
    let load_clean = |path: &str| -> Result<viv::lens::Loaded, String> {
        let l = load(path)?;
        let errs = l.intent_errors();
        if errs.is_empty() { Ok(l) } else { Err(errs.join("\n")) }
    };

    match args[1].as_str() {
        "view" => {
            let Some(path) = pos.first() else { return fail(LENS_USAGE.to_string()) };
            let lens = value("--lens").unwrap_or_else(|| "intent".to_string());
            let l = match load_clean(path) {
                Ok(l) => l,
                Err(e) => return fail(e),
            };
            let text = match lens.as_str() {
                "code" => viv::lens::view_code(&l.root.text.src),
                "intent" => viv::lens::view_intent(&l),
                "all" => viv::lens::view_all(&l),
                // the def as an algorithm box: intent, phrases as words (viv::pseudo)
                "pseudocode" => match viv::pseudo::view(&l, value("--def").as_deref(), flag("--words-only")) {
                    Ok(t) => t,
                    Err(e) => return fail(e),
                },
                "debug" => match debug_view(path) {
                    Ok(t) => t,
                    Err(e) => return fail(e),
                },
                "story" => match story_text(path, &StoryArgs::default()) {
                    Ok((t, _)) => t,
                    Err(e) => return fail(e),
                },
                // what each def does with the values it is given and makes:
                // inferred, never written
                "memory" => {
                    return match memory_view(path, value("--def").as_deref(), flag("--json")) {
                        Ok(t) => {
                            print!("{t}");
                            ExitCode::SUCCESS
                        }
                        Err(e) => fail(e),
                    };
                }
                other => return fail(format!("unknown lens `{other}`: expected code, intent, pseudocode, debug, story, memory or all")),
            };
            if flag("--json") {
                println!("{}", viv::lens::view_json(&l, &lens, &text));
            } else {
                print!("{text}");
            }
            ExitCode::SUCCESS
        }
        "explain" => {
            let Some(path) = pos.first() else { return fail(LENS_USAGE.to_string()) };
            let depth = match value("--depth").map(|d| d.parse::<usize>()) {
                None => None,
                Some(Ok(d)) => Some(d),
                Some(Err(_)) => return fail("--depth takes a whole number".to_string()),
            };
            let l = match load_clean(path) {
                Ok(l) => l,
                Err(e) => return fail(e),
            };
            match viv::lens::explain(&l, value("--from").as_deref(), depth) {
                Ok(n) => {
                    if flag("--json") {
                        println!("{}", viv::lens::explain_json(&n));
                    } else {
                        print!("{}", viv::lens::explain_text(&n));
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => fail(format!("error: {e}")),
            }
        }
        _ => {
            let Some(sub) = args.get(2) else { return fail(LENS_USAGE.to_string()) };
            match sub.as_str() {
                "init" => {
                    let Some(path) = pos.first() else { return fail(LENS_USAGE.to_string()) };
                    let l = match load(path) {
                        Ok(l) => l,
                        Err(e) => return fail(e),
                    };
                    if let Some(e) = &l.hash_error {
                        eprintln!("warning: {path} does not build, so the skeleton has no body hashes:\n{e}");
                    }
                    let text = viv::lens::init(&l);
                    if flag("--stdout") {
                        print!("{text}");
                        return ExitCode::SUCCESS;
                    }
                    let ip = l.root.text.intent_path.clone();
                    if l.root.text.intent.is_some() && !flag("--force") {
                        return fail(format!("{ip} exists; `--force` replaces it, `--stdout` prints the skeleton instead"));
                    }
                    match write_file(&ip, &text) {
                        Ok(()) => {
                            println!("wrote {ip} ({} defs)", text.lines().filter(|l| !l.is_empty() && !l.starts_with(' ') && !l.starts_with("module ")).count());
                            ExitCode::SUCCESS
                        }
                        Err(e) => fail(e),
                    }
                }
                "check" => {
                    if pos.is_empty() {
                        return fail(LENS_USAGE.to_string());
                    }
                    let json = flag("--json");
                    let mut total = 0;
                    let mut files_json = Vec::new();
                    for path in &pos {
                        let l = match load(path) {
                            Ok(l) => l,
                            Err(e) => return fail(e),
                        };
                        let mut hist = GitHistory { path: path.clone(), commits: None, hashed: Default::default() };
                        let mut findings = viv::lens::check(&l, &mut |def, hash| hist.find(def, hash));
                        // the debug lens, when the module has a `.debug` file
                        let dp = PathBuf::from(path).with_extension("debug");
                        if let Some(text) = read_opt(&dp) {
                            findings.extend(viv::lens::check_debug(&l, &dp.display().to_string(), &text));
                        }
                        // the holds an equation may cite: this module's debug lens and its used modules'
                        let mut holds: Vec<String> = Vec::new();
                        for m in std::iter::once(&l.root).chain(l.used.iter()) {
                            let p = PathBuf::from(&m.text.path).with_extension("debug");
                            if let Some(Ok(blocks)) = read_opt(&p).map(|t| viv::debug_lens::check_file(&t)) {
                                for b in &blocks {
                                    for item in &b.items {
                                        if let viv::ast::DebugItem::Hold { sentence, .. } = item {
                                            holds.push(sentence.clone());
                                        }
                                    }
                                }
                            }
                        }
                        findings.extend(viv::lens::check_hold_citations(&l, &holds));
                        total += findings.len();
                        if json {
                            files_json.push(format!(
                                "{{\"path\":\"{}\",\"findings\":[{}]}}",
                                path.replace('\\', "\\\\").replace('"', "\\\""),
                                findings.iter().map(|f| f.json()).collect::<Vec<_>>().join(",")
                            ));
                        } else {
                            for f in &findings {
                                print!("{}", f.render());
                            }
                            let n = findings.len();
                            println!("lens check {path}: {}", if n == 0 { "ok".to_string() } else { format!("{n} problem{}", if n == 1 { "" } else { "s" }) });
                        }
                    }
                    if json {
                        println!("{{\"files\":[{}],\"problems\":{total}}}", files_json.join(","));
                    }
                    if total == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
                }
                "coverage" => lens_cli::coverage(args),
                "needed" => lens_cli::needed(args),
                "stamp" => {
                    let (Some(path), Some(def)) = (pos.first(), pos.get(1)) else { return fail(LENS_USAGE.to_string()) };
                    let l = match load_clean(path) {
                        Ok(l) => l,
                        Err(e) => return fail(e),
                    };
                    match viv::lens::stamp(&l, def) {
                        Ok((src, file, msg)) => {
                            if src != l.root.text.src {
                                if let Err(e) = write_file(path, &src) {
                                    return fail(e);
                                }
                            }
                            if let Some(f) = file {
                                if Some(&f) != l.root.text.intent.as_ref() {
                                    if let Err(e) = write_file(&l.root.text.intent_path, &f) {
                                        return fail(e);
                                    }
                                }
                            }
                            println!("{msg}");
                            ExitCode::SUCCESS
                        }
                        Err(e) => fail(format!("error: {e}")),
                    }
                }
                "move" => {
                    let Some(path) = pos.first() else { return fail(LENS_USAGE.to_string()) };
                    let (to_file, to_inline) = (flag("--to-file"), flag("--to-inline"));
                    if to_file == to_inline {
                        return fail("viv lens move: say --to-file or --to-inline".to_string());
                    }
                    let l = match load_clean(path) {
                        Ok(l) => l,
                        Err(e) => return fail(e),
                    };
                    let only: Vec<String> = pos[1..].to_vec();
                    let only = if only.is_empty() { None } else { Some(only.as_slice()) };
                    let (src, file, kept) = if to_file { viv::lens::to_file(&l, only) } else { viv::lens::to_inline(&l, only, false) };
                    if src != l.root.text.src {
                        if let Err(e) = write_file(path, &src) {
                            return fail(e);
                        }
                    }
                    let ip = l.root.text.intent_path.clone();
                    match file {
                        Some(f) => {
                            if Some(&f) != l.root.text.intent.as_ref() {
                                if let Err(e) = write_file(&ip, &f) {
                                    return fail(e);
                                }
                            }
                        }
                        None => {
                            if l.root.text.intent.is_some() {
                                if let Err(e) = std::fs::remove_file(&ip) {
                                    return fail(format!("error: cannot remove {ip}: {e}"));
                                }
                                println!("removed {ip}: nothing is left in it");
                            }
                        }
                    }
                    for k in &kept {
                        println!("kept: {k}");
                    }
                    println!("moved {} {}", if to_file { "to" } else { "out of" }, ip);
                    ExitCode::SUCCESS
                }
                _ => fail(LENS_USAGE.to_string()),
            }
        }
    }
}

/// `viv check --phrases [--json] [--cert] [--emit out.viv] <file>`: the
/// program's phrases held to the closed forms of lib/cas (viv::cas_check).
/// Exit 1 when a phrase's number disagrees with its exact value.
fn check_phrases(args: &[String]) -> ExitCode {
    let json = args.iter().any(|a| a == "--json");
    let cert = args.iter().any(|a| a == "--cert");
    let emit = args.iter().position(|a| a == "--emit").and_then(|i| args.get(i + 1)).cloned();
    let Some(path) = args.iter().enumerate().find(|(i, a)| !a.starts_with("--") && (*i == 0 || args[*i - 1] != "--emit")).map(|(_, a)| a.clone()) else {
        eprintln!("usage: viv check --phrases [--json] [--cert] [--emit out.viv] <input.viv>");
        return ExitCode::FAILURE;
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (here, root) = module_dirs(&path);
    let mut user = |name: &str| -> Result<String, String> {
        let c = module_path(&here, &root, name)?;
        std::fs::read_to_string(&c).map_err(|e| format!("error: cannot read {}: {}", c.display(), e))
    };
    let mut resolve = viv::cas_check::resolver(&mut user);
    let (program, checks) = match viv::cas_check::generate(&src, &mut resolve) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(out) = &emit {
        if let Err(e) = std::fs::write(out, &program) {
            eprintln!("error: {out}: {e}");
            return ExitCode::FAILURE;
        }
    }
    let (shown, recs) = match viv::cas_check::run(&program, &mut resolve) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: the check program did not run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (text, ok) = viv::cas_check::report(&checks, &shown, &recs);
    if json {
        println!("{}", viv::cas_check::json(&checks, &recs));
    } else {
        print!("{text}");
    }
    if cert {
        let cp = PathBuf::from(&path).with_extension("cert");
        if let Ok(old) = std::fs::read_to_string(&cp) {
            for s in viv::cas_check::stale(&old, &checks) {
                eprintln!("stale: the entry @{s} of {} names a phrase that changed or is gone", cp.display());
            }
        }
        let module = module_display(&root, &PathBuf::from(&path));
        if let Err(e) = std::fs::write(&cp, viv::cas_check::cert_text(&module, &checks, &recs)) {
            eprintln!("error: {}: {e}", cp.display());
            return ExitCode::FAILURE;
        }
    }
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// `viv export <name>.export.toml [-o DIR] [--reports DIR]`: write `<name>.c`,
/// `<name>.h`, `<name>.py`, `README.md` and `test_<name>.py` into DIR
/// (default `target/export/<name>` under the repository). `--reports DIR`
/// also writes, for each deck, the Viv file that prints the results in full,
/// for the exact-bits check (tools/export/check.py). `--with-sqlite` copies
/// SQLite's amalgamation beside a program that uses the `db_*` builtins and
/// compiles it with the program (crate::db); a program that uses them must
/// say so, and one that does not never includes it.
/// `viv cuda <file>.viv [-o <out>.cu]`: the CUDA C of the program's kernels
/// and eligible `parallel for` loops, generated, not compiled.
fn cuda_cli(args: &[String]) -> ExitCode {
    let usage = "usage: viv cuda <file>.viv [-o <file>.cu] [--set NAME=value ...]";
    let (mut file, mut out) = (None::<String>, None::<PathBuf>);
    let sets = match read_sets(args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" if i + 1 < args.len() => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            // a global constant set from the command line, as in a run: the
            // generated CUDA C writes constants in as literals
            "--set" if i + 1 < args.len() => i += 1,
            a if file.is_none() && !a.starts_with('-') => file = Some(a.to_string()),
            _ => {
                eprintln!("{usage}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }
    let Some(file) = file else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    let Loaded { text: src, sources } = match load(&file, false, "", &sets) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{}", e.text());
            return ExitCode::FAILURE;
        }
    };
    let toks = match viv::lexer::lex(&src, knotkit::span::SourceId(0)) {
        Ok(t) => t,
        Err(d) => return render_diags(&d, &sources),
    };
    let program = match viv::cuda::raw(|| viv::parser::parse(toks, knotkit::span::SourceId(0))) {
        Ok(p) => p,
        Err(d) => return render_diags(&d, &sources),
    };
    let name = std::path::Path::new(&file).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or(file.clone());
    match viv::cuda::emit(&program, &src, &name) {
        Ok(cu) => match out {
            Some(o) => match std::fs::write(&o, cu) {
                Ok(()) => {
                    println!("wrote {}", o.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("cannot write {}: {e}", o.display());
                    ExitCode::FAILURE
                }
            },
            None => {
                print!("{cu}");
                ExitCode::SUCCESS
            }
        },
        Err(d) => render_diags(&d, &sources),
    }
}

fn export_cli(args: &[String]) -> ExitCode {
    let usage = "usage: viv export <name>.export.toml [-o <dir>] [--reports <dir>] [--with-sqlite]\n       viv export --infer <file>.viv [-o <dir>]   (a kernel collection of the file's annotated defs)";
    let (mut manifest, mut out, mut reports) = (None::<String>, None::<PathBuf>, None::<PathBuf>);
    let mut infer = false;
    let mut with_sqlite = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--with-sqlite" => with_sqlite = true,
            "-o" | "--reports" if i + 1 >= args.len() => {
                eprintln!("{usage}");
                return ExitCode::FAILURE;
            }
            "-o" => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--reports" => {
                reports = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "-h" | "--help" => {
                println!("{usage}");
                return ExitCode::SUCCESS;
            }
            "--infer" => infer = true,
            a if manifest.is_none() && !a.starts_with('-') => manifest = Some(a.to_string()),
            _ => {
                eprintln!("{usage}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }
    let Some(manifest) = manifest else {
        eprintln!("{usage}");
        return ExitCode::FAILURE;
    };
    let done = if infer {
        if reports.is_some() {
            eprintln!("{usage}");
            return ExitCode::FAILURE;
        }
        export_infer(&manifest, out)
    } else {
        export_run(&manifest, out, reports, with_sqlite)
    };
    match done {
        Ok(dir) => {
            println!("wrote {}", dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprint!("{}", if e.ends_with('\n') { e } else { format!("{e}\n") });
            ExitCode::FAILURE
        }
    }
}

/// Compile the assembled program: the text the engines see, the IR and the
/// C. `maps` holds each file's source map as `export_read` loaded it.
fn export_compile(manifest: &str, a: &viv::export::Assembly, maps: &[(String, SourceMap)]) -> Result<(String, knotkit::ir::IrProgram<viv::lower::Viv>, String), String> {
    let prelude: String = a.parts.iter().map(|p| p.text.as_str()).collect();
    let parts = a.parts.iter().map(|p| PreludePart { file: p.file.clone(), text: p.text.clone() }).collect();
    set_prelude_parts(parts, maps.to_vec());
    let Loaded { text: src, sources } = load_text(manifest, None, String::new(), true, &prelude, &[]).map_err(|e| e.text())?;
    let rendered = |ds: &[knotkit::diag::Diagnostic]| ds.iter().map(|d| PlainRenderer.render(d, &sources)).collect::<String>();
    let (program, emission) = viv::lower::with_sites_map(sources.clone(), &src, || {
        let program = viv::build_ir(&src, knotkit::span::SourceId(0))?;
        let emission = CBackend.emit(&program, &NativeIo)?;
        Ok::<_, Vec<knotkit::diag::Diagnostic>>((program, emission))
    })
    .map_err(|ds| rendered(&ds))?;
    let c = emission.artifact.as_c().ok_or("error: the C backend wrote no C")?.to_string();
    Ok((src, program, c))
}

/// The source files a program was compiled from, (path under `root`,
/// text), for its license notices: each file a person wrote or one derived
/// from another language, not the spliced texts.
fn license_sources(maps: &[&SourceMap], root: &std::path::Path) -> Vec<(String, String)> {
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut out: Vec<(String, String)> = Vec::new();
    for map in maps {
        for i in 0..map.len() {
            let Some(f) = map.file(knotkit::span::SourceId(i as _)) else { continue };
            // the program's modules; not a manifest, not a spliced text
            let module = [".viv", ".vivh", ".py"].iter().any(|e| f.name.ends_with(e));
            if matches!(f.origin, knotkit::span::Origin::Composite(_)) || !module {
                continue;
            }
            let p = std::path::Path::new(&f.name);
            let abs = std::fs::canonicalize(p).or_else(|_| std::fs::canonicalize(root.join(p))).unwrap_or_else(|_| p.to_path_buf());
            // under the program's root, or under the checkout it comes from
            let checkout = abs.ancestors().find(|d| d.join("REUSE.toml").is_file());
            let name = abs
                .strip_prefix(&root)
                .ok()
                .or_else(|| checkout.and_then(|c| abs.strip_prefix(c).ok()))
                .map(|r| r.display().to_string())
                .unwrap_or_else(|| f.name.clone());
            if !out.iter().any(|(n, _)| *n == name) {
                out.push((name, f.text.clone()));
            }
        }
    }
    out
}

/// The exported C with its license comment on top, and its NOTICES; a GPL
/// module refuses the export, an MPL or LGPL one prints a warning.
fn export_licenses(c_file: String, srcs: Vec<(String, String)>, sqlite: bool, root: &std::path::Path) -> Result<(String, String), String> {
    let fallback = |name: &str| viv::licenses::reuse_toml(&root.join(name)).or_else(|| viv::licenses::reuse_toml(std::path::Path::new(name)));
    let mut cs = viv::licenses::gather(&srcs, &fallback);
    cs.extend(viv::licenses::runtime(sqlite));
    for w in viv::licenses::check(&cs)? {
        eprintln!("{w}");
    }
    let extra = |id: &str| std::fs::read_to_string(root.join("LICENSES").join(format!("{id}.txt"))).ok();
    Ok((format!("{}{}", viv::licenses::c_comment(&cs), c_file), viv::licenses::notices(&cs, &extra)))
}

/// Every global of a compiled program, with its type.
fn export_globals(program: &knotkit::ir::IrProgram<viv::lower::Viv>) -> std::collections::HashMap<String, viv::lower::Ty> {
    let mut out = std::collections::HashMap::new();
    for &root in program.roots() {
        if let viv::lower::Viv::Global { name, ty } = &program.get(root).kind {
            out.insert(name.clone(), *ty);
        }
    }
    out
}

fn export_run(manifest: &str, out: Option<PathBuf>, reports: Option<PathBuf>, with_sqlite: bool) -> Result<PathBuf, String> {
    use viv::export;
    let text = std::fs::read_to_string(manifest).map_err(|e| format!("error: cannot read {manifest}: {e}"))?;
    let m = export::parse_manifest(&text).map_err(|e| format!("error: {manifest}: {e}"))?;
    if m.is_kernels {
        // a manifest's paths are under the repository root; a manifest
        // outside any checkout (the root then being VIV_ROOT's) names its
        // files from its own directory
        let (here, root) = module_dirs(manifest);
        let here = std::fs::canonicalize(&here).unwrap_or(here);
        let base = if here.starts_with(std::fs::canonicalize(&root).unwrap_or(root.clone())) { root } else { here };
        return export_kernels_run(manifest, &m, out, &base, None);
    }
    let (_, root) = module_dirs(manifest);
    // each file as `--use` reads it: its imports spliced in, with the map of
    // where its lines came from
    let maps: std::cell::RefCell<Vec<(String, SourceMap)>> = std::cell::RefCell::new(Vec::new());
    let read = |p: &str| -> Result<String, String> {
        let abs = root.join(p);
        let raw = std::fs::read_to_string(&abs).map_err(|e| format!("error: cannot read {p} (paths in the manifest are under {}): {e}", root.display()))?;
        if p.ends_with(".viv") {
            let l = load_text(&abs.display().to_string(), Some(p), raw, false, "", &[]).map_err(|e| e.text())?;
            if !maps.borrow().iter().any(|(n, _)| n == p) {
                maps.borrow_mut().push((p.to_string(), l.sources));
            }
            Ok(l.text)
        } else {
            Ok(raw)
        }
    };
    viv::lower::set_exporting(true);
    // a first compile, for the type of each global; then the program with
    // each parameter read where it is assigned
    let map_of = |p: &str| maps.borrow().iter().find(|(n, _)| n == p).map(|(_, s)| s.clone());
    let first = export::assemble_mapped(&m, &read, None, &map_of)?;
    let (_, program, _) = export_compile(manifest, &first, &maps.borrow())?;
    let globals = export_globals(&program);
    let a = export::assemble_mapped(&m, &read, Some(&globals), &map_of)?;
    let (src, program, c) = export_compile(manifest, &a, &maps.borrow())?;
    let globals = export_globals(&program);
    // the program's hash: its defs' hashes (`viv hash`), hashed
    let hashes = viv::hash_report(&src).map_err(|e| format!("error: hashing the program: {e}"))?;
    viv::lower::set_exporting(false);

    let commit = std::process::Command::new("git").arg("-C").arg(&root).args(["rev-parse", "--short=12", "HEAD"]).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(|| "unknown".into());
    let program_hash = viv::objcache::sha256_hex(hashes.as_bytes());
    let version = format!("viv {commit}; program {}", &program_hash[..16]);
    let about = match &m.about {
        Some(p) if p.ends_with(".viv") => export::fence_text(&std::fs::read_to_string(root.join(p)).map_err(|e| format!("error: cannot read {p}: {e}"))?),
        Some(p) => export::intent_text(&read(p)?),
        None => String::new(),
    };
    let comp = export::Compiled { c: &c, globals: &globals, version: version.clone() };
    let c_file = export::c_file(&m, &a, &comp)?;
    let dir = out.unwrap_or_else(|| root.join("target").join("export").join(&m.name));
    std::fs::create_dir_all(&dir).map_err(|e| format!("error: cannot make {}: {e}", dir.display()))?;
    let hashes_src = std::path::Path::new(manifest).with_file_name(format!("{}.exact_hashes.json", m.name));
    let recorded = std::fs::read_to_string(&hashes_src).ok();
    let sqlite = viv::db::uses_sqlite(&c_file);
    if sqlite && !with_sqlite {
        return Err(format!("error: {manifest}: the program uses the db_* builtins (SQLite); export it with --with-sqlite"));
    }
    let srcs = license_sources(&maps.borrow().iter().map(|(_, s)| s).collect::<Vec<_>>(), &root);
    let (c_file, notices) = export_licenses(c_file, srcs, sqlite, &root)?;
    let mut py = export::python(&m, &a, &about, &version);
    if sqlite {
        py = viv::db::py_with_sqlite(&py);
        let src = viv::db::amalgamation_dir().join("sqlite3.c");
        std::fs::copy(&src, dir.join("sqlite3.c")).map_err(|e| format!("error: cannot copy {}: {e}", src.display()))?;
    }
    let mut files = vec![
        (format!("{}.c", m.name), c_file),
        (format!("{}.h", m.name), export::header(&m, &a, &version)),
        (format!("{}.py", m.name), py),
        ("README.md".to_string(), export::readme(&m, &about, &version)),
        (format!("test_{}.py", m.name), export::test_py(&m, recorded.as_deref())),
        ("NOTICES".to_string(), notices),
    ];
    if let Some(h) = &recorded {
        files.push(("exact_hashes.json".to_string(), h.clone()));
    }
    if let Some(cases) = &m.cases {
        files.push(("cases.json".to_string(), read(cases)?));
    }
    for (name, text) in &files {
        std::fs::write(dir.join(name), text).map_err(|e| format!("error: cannot write {}: {e}", dir.join(name).display()))?;
    }
    if let Some(rdir) = reports {
        std::fs::create_dir_all(&rdir).map_err(|e| format!("error: cannot make {}: {e}", rdir.display()))?;
        let names: Vec<String> = if m.decks.is_empty() { vec![String::new()] } else { m.decks.iter().map(|d| d.name.clone()).collect() };
        for (di, name) in names.iter().enumerate() {
            let file = rdir.join(if name.is_empty() { "report.viv".to_string() } else { format!("{name}_report.viv") });
            std::fs::write(&file, export::report_viv(&m, &globals, &a.produced[di])).map_err(|e| format!("error: cannot write {}: {e}", file.display()))?;
        }
    }
    Ok(dir)
}

/// `viv export` of a kernel collection:
/// the manifest's files imported as modules, compiled once, an entry point
/// per kernel after the program's C.
/// `viv export --infer file.viv [-o DIR]`: the kernel collection of the
/// file's top-level defs with annotated parameters (export_kernels::infer),
/// exported as a manifest's collection would be. The defs left out are
/// listed on stderr and in `<name>.notes`.
fn export_infer(file: &str, out: Option<PathBuf>) -> Result<PathBuf, String> {
    use viv::export_kernels as ek;
    let path = std::path::Path::new(file);
    let Some(stem) = path.file_name().and_then(|n| n.to_str()).and_then(|n| n.strip_suffix(".viv")) else {
        return Err(format!("error: {file}: `viv export --infer` takes a .viv file"));
    };
    if !viv::export::c_ident(stem) {
        return Err(format!("error: {file}: the module's name `{stem}` must be a name (letters, digits, `_`), to be imported"));
    }
    let raw = std::fs::read_to_string(path).map_err(|e| format!("error: cannot read {file}: {e}"))?;
    let base = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let rel = format!("{stem}.viv");
    // resolve the imports first, so that the file read alone knows the
    // signatures of the defs it imports and may call them with keyword
    // arguments (viv::used_sigs reads the remembered module texts; B297); an
    // import that does not resolve is reported by the build below
    let _ = expand_uses_from(&raw, file);
    // the file alone, for its own defs (not the ones its imports bring)
    let mut own = SourceMap::new();
    own.add(file, raw.clone());
    let prog = viv::parse_before_grad(&raw, knotkit::span::SourceId(0)).map_err(|ds| ds.iter().map(|d| PlainRenderer.render(d, &own)).collect::<String>())?;
    let (m, notes) = ek::infer(stem, &rel, &prog, &raw);
    // the collection's program is `import <stem>`, run from the file's directory
    let anchor = base.join(format!("{stem}.export.toml"));
    export_kernels_run(&anchor.display().to_string(), &m, out, &base, Some(notes))
}

/// `base` is the directory the manifest's file paths are under (the
/// repository root for a manifest; the file's directory for `--infer`).
/// `inferred`, for `--infer`: the notes on the defs left out so far; the
/// results are then read from the compiled program, not checked against it.
fn export_kernels_run(manifest: &str, m_in: &viv::export::Manifest, out: Option<PathBuf>, base: &std::path::Path, inferred: Option<Vec<String>>) -> Result<PathBuf, String> {
    use viv::export_kernels as ek;
    use viv::lower::Viv;
    let root = base.to_path_buf();
    // each file parsed on its own, for its defs' parameters and annotations
    let mut files = Vec::new();
    let mut lic_maps: Vec<SourceMap> = Vec::new();
    let mut m = m_in.clone();
    for f in m.files.clone() {
        let f = &f;
        let abs = root.join(f);
        let raw = std::fs::read_to_string(&abs).map_err(|e| format!("error: cannot read {f} (paths in the manifest are under {}): {e}", root.display()))?;
        // a kernel's help opens with its def's `#|` docstring, unless the manifest gives a `doc`
        for k in m.kernels.iter_mut().filter(|k| k.file == *f && !k.info.iter().any(|(key, _)| key == "doc")) {
            if let Some(d) = ek::def_doc(&raw, &k.def) {
                k.info.insert(0, ("doc".to_string(), d));
            }
        }
        let l = load_text(&abs.display().to_string(), Some(f), raw, false, "", &[]).map_err(|e| e.text())?;
        let prog = viv::parse_before_grad(&l.text, knotkit::span::SourceId(0)).map_err(|ds| ds.iter().map(|d| PlainRenderer.render(d, &l.sources)).collect::<String>())?;
        files.push((f.clone(), prog));
        lic_maps.push(l.sources);
    }
    // the collection: one import per file, so each keeps its own top level
    viv::lower::set_exporting(true);
    let text = ek::program_text(&m)?;
    let Loaded { text: src, sources } = load_text(manifest, Some(manifest), text, false, "", &[]).map_err(|e| e.text())?;
    let rendered = |ds: &[knotkit::diag::Diagnostic]| ds.iter().map(|d| PlainRenderer.render(d, &sources)).collect::<String>();
    let (program, emission) = viv::lower::with_sites_map(sources.clone(), &src, || {
        let program = viv::build_ir(&src, knotkit::span::SourceId(0))?;
        let emission = CBackend.emit(&program, &NativeIo)?;
        Ok::<_, Vec<knotkit::diag::Diagnostic>>((program, emission))
    })
    .map_err(|ds| rendered(&ds))?;
    viv::lower::set_exporting(false);
    let c = emission.artifact.as_c().ok_or("error: the C backend wrote no C")?.to_string();
    let mut funcs = std::collections::HashMap::new();
    let mut records = std::collections::HashMap::new();
    for &r in program.roots() {
        match &program.get(r).kind {
            Viv::Func { name, params, ret, .. } => {
                funcs.insert(name.clone(), (params.iter().map(|(_, t)| *t).collect::<Vec<_>>(), *ret));
            }
            Viv::RecordDef { name, fields } => {
                records.insert(name.clone(), fields.iter().map(|(f, t)| (f.to_string(), *t)).collect::<Vec<_>>());
            }
            _ => {}
        }
    }
    let mut notes = inferred.clone().unwrap_or_default();
    if inferred.is_some() {
        ek::infer_results(&mut m, &funcs, &records, &mut notes);
        for n in &notes {
            eprintln!("note: {n}");
        }
        if m.kernels.is_empty() {
            return Err(format!("error: {}: no def can be exported: a kernel's parameters are annotated (Num, Int, Bool, Complex, Arr, [Int]) and it returns a value\n", m.files[0]));
        }
    }
    let mut defs = ek::resolve(&m, &files, &funcs, &records)?;
    if let Ok(whole) = viv::parse(&src, knotkit::span::SourceId(0)) {
        ek::lend_over(&m, &mut defs, &whole);
    }
    let globals = export_globals(&program);
    let hashes = viv::hash_report(&src).map_err(|e| format!("error: hashing the program: {e}"))?;
    let commit = if inferred.is_some() {
        VIV_VERSION.trim_start_matches("viv ").to_string()
    } else {
        std::process::Command::new("git").arg("-C").arg(&root).args(["rev-parse", "--short=12", "HEAD"]).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(|| "unknown".into())
    };
    let version = format!("viv {commit}; program {}", &viv::objcache::sha256_hex(hashes.as_bytes())[..16]);
    let about = match &m.about {
        Some(p) if p.ends_with(".viv") => viv::export::fence_text(&std::fs::read_to_string(root.join(p)).map_err(|e| format!("error: cannot read {p}: {e}"))?),
        Some(p) if p.ends_with(".intent") => viv::export::intent_text(&std::fs::read_to_string(root.join(p)).map_err(|e| format!("error: cannot read {p}: {e}"))?),
        // a kernel collection's `about` may be the text itself
        Some(t) => t.clone(),
        None => String::new(),
    };
    let c_file = ek::c_file(&m, &defs, &c, &globals, &version)?;
    let srcs = license_sources(&lic_maps.iter().chain([&sources]).collect::<Vec<_>>(), &root);
    let (c_file, notices) = export_licenses(c_file.clone(), srcs, viv::db::uses_sqlite(&c_file), &root)?;
    let dir = out.unwrap_or_else(|| root.join("target").join("export").join(&m.name));
    std::fs::create_dir_all(&dir).map_err(|e| format!("error: cannot make {}: {e}", dir.display()))?;
    let hashes_src = std::path::Path::new(manifest).with_file_name(format!("{}.exact_hashes.json", m.name));
    let mut out_files = vec![
        (format!("{}.c", m.name), c_file),
        (format!("{}.h", m.name), ek::header(&m, &version)),
        (format!("{}.py", m.name), ek::python(&m, &about, &version)),
        ("README.md".to_string(), ek::readme(&m, &about, &version)),
        (format!("test_{}.py", m.name), ek::test_py(&m)),
        (format!("{}.pyi", m.name), ek::stub(&m, &about, &version)),
        ("NOTICES".to_string(), notices),
    ];
    if inferred.is_some() {
        out_files.push((format!("{}.notes", m.name), notes.iter().map(|n| format!("{n}\n")).collect()));
    }
    if let Ok(h) = std::fs::read_to_string(&hashes_src) {
        out_files.push(("exact_hashes.json".to_string(), h));
    }
    if let Some(cases) = &m.cases {
        out_files.push(("cases.json".to_string(), std::fs::read_to_string(root.join(cases)).map_err(|e| format!("error: cannot read {cases}: {e}"))?));
    }
    for (name, text) in &out_files {
        std::fs::write(dir.join(name), text).map_err(|e| format!("error: cannot write {}: {e}", dir.join(name).display()))?;
    }
    Ok(dir)
}

const DIGITS_USAGE: &str = "usage: viv digits [options] <file.viv>
  reruns the program in the interpreter with its floating-point evaluation
  perturbed and reports, per printed number, the leading digits that stay
  the same, and which lines move the rest
  --no-order        leave reductions in the program's order
  --randoms N       random permutations of each reduction (default 3)
  --no-fma          leave a*b + c unfused
  --mca MODE        Monte Carlo arithmetic: mca (default), rr, pb, or off
  --samples N       MCA samples (default 16)
  --seed N          the first seed (default 1)
  --attrib N        lines to perturb one at a time, most-executed first (default 8; 0: none)
  --trace N         numbers whose lost digits no such line explains, traced to a line by bisection (default 2)
  --json            the report as JSON
  -o FILE           also write the JSON report to FILE (the explorer's lens)
  --set NAME=value  as for viv run";

fn digits_cli(args: &[String]) -> ExitCode {
    let mut o = viv::digits::Options::default();
    let mut json = false;
    let mut json_out: Option<String> = None;
    let mut path = None;
    let mut i = 2;
    let num = |a: Option<&String>, flag: &str| -> Result<u64, String> {
        a.and_then(|v| v.parse::<u64>().ok()).ok_or_else(|| format!("{flag} takes a whole number"))
    };
    while i < args.len() {
        let a = args[i].as_str();
        let r: Result<(), String> = match a {
            "--no-order" => Ok(o.orders = false),
            "--no-fma" => Ok(o.fma = false),
            "--json" => Ok(json = true),
            "--randoms" => num(args.get(i + 1), a).map(|n| { o.randoms = n as usize; i += 1; }),
            "--samples" => num(args.get(i + 1), a).map(|n| { o.samples = n as usize; i += 1; }),
            "--seed" => num(args.get(i + 1), a).map(|n| { o.seed = n; i += 1; }),
            "--attrib" => num(args.get(i + 1), a).map(|n| { o.attrib = n as usize; i += 1; }),
            "--trace" => num(args.get(i + 1), a).map(|n| { o.trace = n as usize; i += 1; }),
            "--mca" => match args.get(i + 1).and_then(|m| viv::digits::Mca::parse(m)) {
                Some(m) => { o.mca = m; i += 1; Ok(()) }
                None => Err("--mca takes mca, rr, pb or off".into()),
            },
            "--set" | "--using" => { i += 1; Ok(()) }
            "-o" => match args.get(i + 1) {
                Some(f) => { json_out = Some(f.clone()); i += 1; Ok(()) }
                None => Err("-o takes a file".into()),
            },
            "-h" | "--help" => { println!("{DIGITS_USAGE}"); return ExitCode::SUCCESS; }
            _ if a.starts_with("--") => Err(format!("unknown option {a}")),
            _ => Ok(path = Some(a.to_string())),
        };
        if let Err(e) = r {
            eprintln!("error: {e}\n{DIGITS_USAGE}");
            return ExitCode::FAILURE;
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("{DIGITS_USAGE}");
        return ExitCode::FAILURE;
    };
    if o.samples == 1 {
        o.samples = 2;
    }
    let sets = match read_sets(&args[2..]) {
        Ok(s) => s,
        Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
    };
    let Loaded { text: src, sources } = match load(&path, false, "", &sets) {
        Ok(l) => l,
        Err(e) => return e.report(json, ""),
    };
    let text = src.clone();
    let report = match std::thread::Builder::new().stack_size(INTERP_STACK).spawn(move || {
        viv::digits::analyze(&text, &o, &|s| viv::interp::run(s))
    }) {
        Ok(h) => match h.join() {
            Ok(r) => r,
            Err(_) => Err("the interpreter stopped unexpectedly".into()),
        },
        Err(e) => Err(format!("cannot start the interpreter: {e}")),
    };
    let report = match report {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: the unperturbed run stopped: {e}");
            return ExitCode::FAILURE;
        }
    };
    // a line of the program text, as file:line in the file it came from
    let starts: Vec<usize> = std::iter::once(0).chain(src.match_indices('\n').map(|(k, _)| k + 1)).collect();
    let loc = |line: u32| -> (String, String) {
        let off = starts.get(line.saturating_sub(1) as usize).copied().unwrap_or(0) as u32;
        let sp = sources.origin_of(knotkit::span::Span::new(knotkit::span::SourceId(0), off, off));
        let name = sources.file(sp.source).map(|f| f.name.clone()).unwrap_or_default();
        match sources.locate(sp.source, sp.start) {
            Some((l, _, t)) => (format!("{name}:{l}"), t.to_string()),
            None => (format!("{name}:{line}"), String::new()),
        }
    };
    if let Some(f) = &json_out {
        if let Err(e) = std::fs::write(f, report.json(&loc) + "\n") {
            eprintln!("error: cannot write {f}: {e}");
            return ExitCode::FAILURE;
        }
    }
    if json {
        println!("{}", report.json(&loc));
    } else {
        print!("{}", report.text(&loc));
    }
    ExitCode::SUCCESS
}
