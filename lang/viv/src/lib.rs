// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv: a general-purpose numerical language built on knotkit.
//!
//! The pipeline is: source -> tokens -> AST -> (checked lowering) -> IR. The IR
//! is the source of truth (1.1); the backend consumes it via the kit.

pub mod ast;
pub mod builtins;
pub mod memory_lens;
pub mod memory_view;
pub mod parallel;
pub mod defun;
pub mod interp;
pub mod lexer;
pub mod lower;
pub mod fuse;
pub mod cuda;
pub mod parser;
pub mod wasm_backend;
pub mod vmath;
pub mod ladder;
pub mod lens;
pub mod lens_cov;
pub mod debug_lens;
pub mod story;
pub mod graph;
pub mod codemap;
pub mod atlas;
pub mod atlas_walk;
pub mod tex;
pub mod hints;
pub mod recursion;
pub mod figsvg;
pub mod mathtext;
mod mathtext_metrics;
pub mod grad;
pub mod printfmt;
pub mod tuples;
pub mod update;
pub mod surface;
pub mod ctor;
pub mod phrases;
pub mod contracts;
pub mod stdlib;
pub mod pseudo;
pub mod derivs;
pub mod cas_check;
pub mod growable;
pub mod graphlit;
pub mod arch;
pub mod features;
pub mod features_usage;
pub mod imports;
pub mod kwargs;
pub mod fastcmp;
pub mod trace;
pub mod objcache;
pub mod export;
pub mod licenses;
pub mod export_kernels;
pub mod errkind;
pub mod db;
pub mod facts;
pub mod digits;
pub use graphlit::set_module_resolver;

/// How deep `#! use` lines may nest: the modules a program uses, the ones
/// they use, and so on. The deepest chain in the repository is 9 modules
/// (scf/stability: lib/hf/stab, the SCF loop, the guess, J and K, the direct
/// J/K member, direct J/K by McMurchie-Davidson, the ERIs, the Hermite
/// tables, the basis); a chain twice as long is more likely a mistake than a
/// design.
/// www/src/lib/uses.js holds the same number.
pub const MAX_USE_DEPTH: usize = 16;

/// `#! use <module>`: a program made of other programs.
///
/// A line `#! use dft/pwdft_it` is replaced, in place, by the text of that
/// module -- so a module that reads globals (the DFT engine reads its deck's
/// `ECUT`) is used AFTER the lines that set them, and a library of functions
/// is used at the top. The module's own `#! use` lines are expanded in turn.
/// Where a module's text comes from is the host's business (a file beside the
/// program, the repository, a post on the site): `resolve` answers by name.
/// Every `#!` line of the included text other than its own `#! use` lines is
/// dropped, so a post used as a module does not bring its header along.
///
/// A module is included once. Two programs that both use `lib/schur`, and a
/// program that uses `lib/lqr` and `lib/are` when the first already uses the
/// second, all name it more than once; the second and later mentions expand to
/// nothing. Without that, the definitions arrive twice and the C backend
/// rejects the program for redefining every function in the module.
///
/// A module's own `test` blocks are left behind. They belong to the module and
/// run when the module is the program; a reader who presses Run on a post that
/// uses `lib/chain` is asking to run the post, and should not also pay seven
/// seconds to re-verify a triple pendulum's energy drift. The library is
/// checked by running the library.
///
/// `#! if-used <module>` ... `#! end-if` keeps the lines between only when
/// the program uses that module, anywhere in its `#! use` closure, before or
/// after this line. A lookup can then serve whichever of several modules a
/// program loads without loading them all: lib/basis/named finds a basis set
/// by name among the tables the program uses. The block holds no `#! use`
/// line of its own, and blocks nest.
///
/// `#! if-none-used <module> <module> ...` ... `#! end-if` keeps its lines,
/// `#! use` lines included, when the program's other `#! use` lines (those
/// outside such blocks) reach none of the modules it names: a default.
/// lib/basis/named loads all eight tables this way for a program that names
/// none, and one table for a program that uses that table's module.
pub fn expand_uses(src: &str, resolve: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<String, String> {
    expand_uses_with(src, resolve, &mut |lang, name, _| {
        Err(format!("module `{name}`: it is written in `{lang}`, and nothing here can translate that"))
    })
}

/// `expand_uses`, with dialects.
///
/// A used module may open with `#! lang <name>`, meaning it is not Viv. It is
/// handed to `translate(lang, module, source)`, which returns Viv source, and
/// that is what gets spliced -- its own `#! use` lines expanded in turn, like
/// any other module.
///
/// The translation is INJECTED rather than looked up here, for the same reason
/// the C toolchain is (the knotkit commitments 1.8): this crate is the vocabulary of
/// one language and has no business knowing the others exist. A driver that can
/// see `weft` supplies a weft translator; one that cannot, does not, and a
/// program that asks for a dialect it has no translator for is told so.
pub fn expand_uses_with(
    src: &str,
    resolve: &mut dyn FnMut(&str) -> Result<String, String>,
    translate: &mut dyn FnMut(&str, &str, &str) -> Result<String, String>,
) -> Result<String, String> {
    expand_uses_mapped(src, resolve, translate).map(|e| e.text)
}

/// A program with its `#! use` lines expanded, and where each of its lines
/// came from, so a driver can show a location in a module at the module's own
/// line (knotkit's composite sources).
#[derive(Debug)]
pub struct Expansion {
    /// The expanded program: exactly what `expand_uses_with` returns.
    pub text: String,
    /// What was spliced: `units[0]` is the program itself, then each module
    /// in the order it was first included.
    pub units: Vec<Unit>,
    /// For each line of `text` (0-based), the unit and the 0-based line of
    /// that unit's `text` it is a copy of; `None` for a line the expansion
    /// wrote (the `# ---- #! use` markers). Empty when the program uses no
    /// module and `text` is the program unchanged.
    pub lines: Vec<Option<(usize, usize)>>,
}

/// One text spliced into an expansion.
#[derive(Debug)]
pub struct Unit {
    /// The module's name as its `#! use` line gives it; empty for the program.
    pub name: String,
    /// The Viv text the lines were copied from: the module as resolved, or,
    /// for a module in another language, its translation.
    pub text: String,
    /// The language the module declares with `#! lang`, when it is not Viv.
    pub lang: Option<String>,
}

/// `expand_uses_with`, keeping the origin of every line (`Expansion`).
///
/// Import statements (crate::imports) are lowered first, in the program and
/// in each module as it is read, to `#! import` lines the splice reads as it
/// reads `#! use`; once the program is spliced, the names are resolved and
/// renamed (`imports::resolve`). A program with no import and no `private`
/// is spliced exactly as before.
pub fn expand_uses_mapped(
    src: &str,
    resolve: &mut dyn FnMut(&str) -> Result<String, String>,
    translate: &mut dyn FnMut(&str, &str, &str) -> Result<String, String>,
) -> Result<Expansion, String> {
    expand_uses_after(src, resolve, translate, &[])
}

/// `expand_uses_mapped` for a text that runs after others in one session (a
/// notebook's later cell): the modules in `seen` are already in the session,
/// so they are not spliced again (an import of one still binds its names).
pub fn expand_uses_after(
    src: &str,
    resolve: &mut dyn FnMut(&str) -> Result<String, String>,
    translate: &mut dyn FnMut(&str, &str, &str) -> Result<String, String>,
    seen: &[String],
) -> Result<Expansion, String> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    let maybe = |t: &str| t.contains("import") || t.lines().any(|l| l.starts_with("private "));
    // each module's text as read, by name: Some(text) or None (no such module)
    let fetched: RefCell<HashMap<String, Result<String, String>>> = RefCell::new(HashMap::new());
    let resolve = RefCell::new(resolve);
    let fetch_raw = |name: &str| -> Result<String, String> {
        if let Some(r) = fetched.borrow().get(name) {
            return r.clone();
        }
        // `/lib/x`, an import's candidate under the repository root: a
        // resolver that knows modules only by name (a map, lib/cas) is
        // asked for `lib/x`
        let mut r = (resolve.borrow_mut())(name);
        if r.is_err() {
            if let Some(plain) = name.strip_prefix('/') {
                r = (resolve.borrow_mut())(plain);
            }
        }
        // the module is asked for again by its canonical name (`lib/x`), and
        // a resolver with side effects (the CLI's reads the module's lens
        // files) must see it once
        if r.is_ok() {
            if let Some(plain) = name.strip_prefix('/') {
                fetched.borrow_mut().entry(plain.to_string()).or_insert_with(|| r.clone());
            }
        }
        fetched.borrow_mut().insert(name.to_string(), r.clone());
        r
    };
    let lowered: RefCell<HashMap<String, imports::Lowered>> = RefCell::new(HashMap::new());
    let originals: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
    let lower_text = |unit: &str, text: &str| -> Result<String, String> {
        if let Some(e) = imports::refuse_use(unit, text) {
            return Err(e);
        }
        if !maybe(text) {
            return Ok(text.to_string());
        }
        let low = imports::lower(unit, text, &mut |c: &str| fetch_raw(c).ok())?;
        let t = low.text.clone();
        if low.has_imports || !low.privates.is_empty() {
            originals.borrow_mut().insert(unit.to_string(), text.to_string());
            lowered.borrow_mut().insert(unit.to_string(), low);
        }
        Ok(t)
    };
    let program = lower_text("", src)?;
    let any_import = |t: &str| t.lines().any(|l| l.starts_with(imports::IMPORT_DIRECTIVE));
    let mut res = |name: &str| -> Result<String, String> {
        let text = fetch_raw(name)?;
        if lang_of(&text).is_some_and(|l| l != "viv") {
            return Ok(text);
        }
        lower_text(name, &text)
    };
    let mut tr = |lang: &str, name: &str, text: &str| -> Result<String, String> {
        let viv = translate(lang, name, text)?;
        lower_text(name, &viv)
    };
    let mut exp = splice_mapped(&program, &mut res, &mut tr, seen)?;
    let lowered = lowered.into_inner();
    if lowered.is_empty() && !any_import(&program) {
        return Ok(exp);
    }
    let units: Vec<imports::UnitIn> = exp
        .units
        .iter()
        .map(|u| imports::UnitIn {
            name: &u.name,
            lowered: lowered.get(&u.name),
            uses: u.text.lines().filter_map(|l| l.trim_end().strip_prefix("#! use ").map(|n| n.trim().to_string())).filter(|n| !n.is_empty()).collect(),
        })
        .collect();
    let lines: Vec<Option<(usize, usize)>> = if exp.lines.is_empty() { (0..exp.text.lines().count()).map(|k| Some((0, k))).collect() } else { exp.lines.clone() };
    let text = imports::resolve(&exp.text, &lines, &units)?;
    drop(units);
    exp.text = text;
    let originals = originals.into_inner();
    for u in exp.units.iter_mut() {
        if let Some(t) = originals.get(&u.name) {
            u.text = t.clone();
        }
    }
    Ok(exp)
}

fn splice_mapped(
    src: &str,
    resolve: &mut dyn FnMut(&str) -> Result<String, String>,
    translate: &mut dyn FnMut(&str, &str, &str) -> Result<String, String>,
    seen: &[String],
) -> Result<Expansion, String> {
    type Origin = Option<(usize, usize)>;
    struct Out {
        text: String,
        lines: Vec<Origin>,
        units: Vec<Unit>,
    }
    impl Out {
        fn push(&mut self, line: &str, origin: Origin) {
            self.text.push_str(line);
            self.text.push('\n');
            self.lines.push(origin);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn go(
        src: &[(String, Origin)],
        resolve: &mut dyn FnMut(&str) -> Result<String, String>,
        translate: &mut dyn FnMut(&str, &str, &str) -> Result<String, String>,
        stack: &mut Vec<String>,
        seen: &mut Vec<String>,
        used: &Closure,
        out: &mut Out,
    ) -> Result<(), String> {
        // the blocks open: whether each keeps its lines, and whether it may
        // hold `#! use` lines (an `#! if-none-used` block)
        let mut keep: Vec<(bool, bool)> = Vec::new();
        for (line, origin) in src {
            let t = line.trim_end();
            if let Some(b) = block_open(t) {
                keep.push(match b {
                    Block::Used(m) => (in_closure(&used.all, &m), false),
                    Block::NoneUsed(ms) => (ms.iter().all(|m| !in_closure(&used.plain, m)), true),
                });
                continue;
            }
            if t == END_IF {
                if keep.pop().is_none() {
                    return Err(format!("`{END_IF}` with no block open"));
                }
                continue;
            }
            if t == imports::CONTINUED {
                continue;
            }
            if let Some(name) = use_line(t) {
                if keep.iter().any(|(_, uses)| !uses) {
                    return Err(format!("an import of `{}` inside an `{}` block: that block chooses lines, not modules", name.trim(), IF_USED.trim_end()));
                }
            }
            if keep.iter().any(|(k, _)| !k) {
                continue;
            }
            // `#! import a b`: the modules of one import line, each as `#! use`
            if let Some(names) = t.strip_prefix(imports::IMPORT_DIRECTIVE) {
                let each: Vec<(String, Origin)> = names.split_whitespace().map(|n| (format!("#! use {n}"), *origin)).collect();
                go(&each, resolve, translate, stack, seen, used, out)?;
                continue;
            }
            if let Some(name) = t.strip_prefix("#! use ") {
                let name = name.trim();
                if name.is_empty() {
                    return Err("`#! use` names no module".to_string());
                }
                if stack.iter().any(|s| s == name) {
                    return Err(format!("module `{name}` imports itself, through {}", stack.join(" -> ")));
                }
                if stack.len() >= MAX_USE_DEPTH {
                    return Err(format!("module `{name}`: modules nested more than {MAX_USE_DEPTH} deep"));
                }
                if seen.iter().any(|s| s == name) {
                    out.push(&format!("# ---- #! use {name} (already included) ----"), None);
                    continue;
                }
                let mut text = resolve(name)?;
                remember_module(name, &text);
                // `#! lang <name>` on a module means it is not Viv. Translate it
                // first; everything after this point sees Viv source.
                let mut dialect = None;
                if let Some(lang) = lang_of(&text) {
                    if lang != "viv" {
                        text = translate(&lang, name, &text)?;
                        dialect = Some(lang);
                    }
                }
                seen.push(name.to_string());
                stack.push(name.to_string());
                out.push(&format!("# ---- #! use {name} ----"), None);
                let unit = out.units.len();
                let mut inner: Vec<(String, Origin)> = Vec::new();
                let mut depth: i32 = 0;
                let mut in_test = false;
                for (k, l) in text.lines().enumerate() {
                    // `test "..." {   # note` is a test header too: the brace is
                    // judged on the code before the comment.
                    if !in_test && l.trim_start().starts_with("test ") && strip_comment(l).trim_end().ends_with('{') {
                        in_test = true;
                        depth = 0;
                    }
                    if in_test {
                        let mut quoted = false;
                        let mut prev_backslash = false;
                        for ch in l.chars() {
                            if quoted {
                                if ch == '"' && !prev_backslash { quoted = false; }
                                prev_backslash = ch == '\\' && !prev_backslash;
                                continue;
                            }
                            match ch {
                                '"' => quoted = true,
                                '{' => depth += 1,
                                '}' => depth -= 1,
                                '#' => break, // a comment runs to end of line
                                _ => {}
                            }
                            prev_backslash = false;
                        }
                        if depth <= 0 { in_test = false; }
                        continue;
                    }
                    if use_line(l).is_some() || block_open(l.trim_end()).is_some() || l.trim_end() == END_IF || !l.starts_with("#!") {
                        inner.push((l.to_string(), Some((unit, k))));
                    }
                }
                out.units.push(Unit { name: name.to_string(), text, lang: dialect });
                go(&inner, resolve, translate, stack, seen, used, out)?;
                out.push(&format!("# ---- end of {name} ----"), None);
                stack.pop();
            } else {
                out.push(line, *origin);
            }
        }
        if !keep.is_empty() {
            return Err(format!("a block is never closed with `{END_IF}`"));
        }
        Ok(())
    }
    let program = Unit { name: String::new(), text: src.to_string(), lang: None };
    if !src.contains("#! use ") && !src.contains(imports::IMPORT_DIRECTIVE) && !src.contains(IF_USED) && !src.contains(IF_NONE_USED) {
        return Ok(Expansion { text: src.to_string(), units: vec![program], lines: Vec::new() });
    }
    // the program's `#! use` closure first, for the blocks: `plain`, what the
    // `#! use` lines outside `#! if-none-used` blocks reach, decides those
    // blocks; `all`, with the lines of the blocks that hold, decides
    // `#! if-used`. Each module is resolved (and translated) once, and the
    // expansion below reads the same texts.
    let mut texts: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut translated: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut viv_of: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut used = Closure::default();
    let mut grow = |start: Vec<String>, set: &mut std::collections::HashSet<String>, fallback: bool, plain: &std::collections::HashSet<String>| -> Result<(), String> {
        let mut todo = start;
        while let Some(name) = todo.pop() {
            if !set.insert(name.clone()) {
                continue;
            }
            if !viv_of.contains_key(&name) {
                let text = resolve(&name)?;
                let mut viv = text.clone();
                if let Some(lang) = lang_of(&text) {
                    if lang != "viv" {
                        viv = translate(&lang, &name, &text)?;
                        translated.insert(name.clone(), viv.clone());
                    }
                }
                texts.insert(name.clone(), text);
                viv_of.insert(name.clone(), viv);
            }
            let (plain_uses, defaults) = block_uses(&viv_of[&name]);
            todo.extend(plain_uses);
            if fallback {
                for (ms, uses) in defaults {
                    if ms.iter().all(|m| !in_closure(plain, m)) {
                        todo.extend(uses);
                    }
                }
            }
        }
        Ok(())
    };
    let (root_plain, root_defaults) = block_uses(src);
    let mut plain = std::collections::HashSet::new();
    grow(root_plain.clone(), &mut plain, false, &std::collections::HashSet::new())?;
    let mut all = std::collections::HashSet::new();
    let mut start = root_plain;
    for (ms, uses) in root_defaults {
        if ms.iter().all(|m| !in_closure(&plain, m)) {
            start.extend(uses);
        }
    }
    grow(start, &mut all, true, &plain)?;
    used.plain = plain;
    used.all = all;
    let mut cached = |name: &str| -> Result<String, String> { texts.get(name).cloned().ok_or_else(|| format!("internal: `{name}` was not resolved")) };
    let mut cached_translation = |_: &str, name: &str, _: &str| -> Result<String, String> { translated.get(name).cloned().ok_or_else(|| format!("internal: `{name}` was not translated")) };
    let lines: Vec<(String, Origin)> = src.lines().enumerate().map(|(k, l)| (l.to_string(), Some((0, k)))).collect();
    let mut out = Out { text: String::new(), lines: Vec::new(), units: vec![program] };
    go(&lines, &mut cached, &mut cached_translation, &mut Vec::new(), &mut seen.to_vec(), &used, &mut out)?;
    Ok(Expansion { text: out.text, units: out.units, lines: out.lines })
}

/// Whether a module named in an `#! if-used` or `#! if-none-used` line is in
/// a set of module names: by its path (`lib/basis/sto3g`) or by the dotted
/// name an import writes (`basis.sto3g`).
fn in_closure(set: &std::collections::HashSet<String>, m: &str) -> bool {
    if set.contains(m) {
        return true;
    }
    if m.contains('/') {
        return false;
    }
    let p = m.replace('.', "/");
    set.contains(&format!("lib/{p}")) || set.contains(&p)
}

/// A line that splices modules: `#! use <module>`, or the `#! import`
/// line an import statement is lowered to (crate::imports).
fn use_line(line: &str) -> Option<&str> {
    line.strip_prefix("#! use ").or_else(|| line.strip_prefix(imports::IMPORT_DIRECTIVE))
}

/// `#! if-used <module>`: the lines up to the matching `#! end-if` are kept
/// only when the program uses `<module>`; `#! if-none-used <modules>`: when
/// the program's other `#! use` lines reach none of them (see `expand_uses`).
const IF_USED: &str = "#! if-used ";
const IF_NONE_USED: &str = "#! if-none-used ";
const END_IF: &str = "#! end-if";

/// What a program uses, for its blocks (see `expand_uses_mapped`).
#[derive(Default)]
struct Closure {
    /// what the `#! use` lines outside `#! if-none-used` blocks reach
    plain: std::collections::HashSet<String>,
    /// everything the expansion includes
    all: std::collections::HashSet<String>,
}

enum Block {
    Used(String),
    NoneUsed(Vec<String>),
}

/// The block a line opens, if it opens one.
fn block_open(line: &str) -> Option<Block> {
    if let Some(m) = line.strip_prefix(IF_USED) {
        return Some(Block::Used(m.trim().to_string()));
    }
    line.strip_prefix(IF_NONE_USED).map(|ms| Block::NoneUsed(ms.split_whitespace().map(String::from).collect()))
}

/// A text's `#! use` lines outside `#! if-none-used` blocks, and each such
/// block's modules with the `#! use` lines inside it.
fn block_uses(text: &str) -> (Vec<String>, Vec<(Vec<String>, Vec<String>)>) {
    let mut plain = Vec::new();
    let mut defaults: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    // the open blocks; for an `#! if-none-used` one, its index in `defaults`
    let mut open: Vec<Option<usize>> = Vec::new();
    for l in text.lines() {
        let t = l.trim_end();
        match block_open(t) {
            Some(Block::Used(_)) => open.push(None),
            Some(Block::NoneUsed(ms)) => {
                defaults.push((ms, Vec::new()));
                open.push(Some(defaults.len() - 1));
            }
            None if t == END_IF => {
                open.pop();
            }
            None => {
                let names: Vec<String> = match (t.strip_prefix("#! use "), t.strip_prefix(imports::IMPORT_DIRECTIVE)) {
                    (Some(n), _) => vec![n.trim().to_string()],
                    (_, Some(ns)) => ns.split_whitespace().map(String::from).collect(),
                    _ => Vec::new(),
                };
                for n in names.into_iter().filter(|n| !n.is_empty()) {
                    match open.iter().rev().find_map(|b| *b) {
                        Some(k) => defaults[k].1.push(n),
                        None => plain.push(n),
                    }
                }
            }
        }
    }
    (plain, defaults)
}


/// A line with its `#` comment removed. A `#` inside a string literal is not a
/// comment, so `test "case #3" {` keeps its brace.
fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    let mut prev_backslash = false;
    for (i, ch) in line.char_indices() {
        if quoted {
            if ch == '"' && !prev_backslash {
                quoted = false;
            }
            prev_backslash = ch == '\\' && !prev_backslash;
            continue;
        }
        match ch {
            '"' => quoted = true,
            '#' => return &line[..i],
            _ => {}
        }
    }
    line
}

/// The dialect a module declares, from a `#! lang <name>` line before any code.
/// A module with no such line is Viv, which is why every existing module keeps
/// working without saying so.
pub fn lang_of(src: &str) -> Option<String> {
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#! lang ") {
            let name = rest.trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        // a declaration comes before anything that is not blank or a comment
        if !t.is_empty() && !t.starts_with('#') {
            return None;
        }
    }
    None
}

/// How `--using CONTRACT=method` is passed among the `--set`s.
pub const USING_SET: &str = "using:";

/// `--set NAME=value`: replace the value of a top-level constant before the
/// program is read. This is how one program runs at two sizes -- `N = 200` for
/// a browser tab, `--set N=20000` for a night on a cluster -- with nothing but
/// that number changed, so the two runs are the same program by content
/// address of every function. Only a top-level `NAME = <number>` line (or
/// `NAME: Type = <number>`, which keeps its type) is touched; a name the program has no such line for is an error, since a
/// silently ignored --set is how a "big run" turns out to have been the small one.
pub fn apply_sets(src: &str, sets: &[(String, String)]) -> Result<String, String> {
    // `--using CONTRACT=method` arrives as a set named USING_SET + CONTRACT
    let mut src = std::borrow::Cow::Borrowed(src);
    for (name, value) in sets {
        if let Some(c) = name.strip_prefix(USING_SET) {
            src = std::borrow::Cow::Owned(contracts::apply_using(&src, c, value)?);
        }
    }
    let src: &str = &src;
    let mut lines: Vec<String> = src.lines().map(|l| l.to_string()).collect();
    for (name, value) in sets.iter().filter(|(n, _)| !n.starts_with(USING_SET)) {
        // a number replaces a number; a double-quoted string with no quote or
        // backslash inside replaces a string (`--set 'CLASS="W"'`, how the NAS
        // benchmarks pick a problem class)
        let is_str = |v: &str| {
            v.len() >= 2 && v.starts_with('"') && v.ends_with('"') && !v[1..v.len() - 1].contains(['"', '\\', '#'])
        };
        let want_str = is_str(value);
        // an enum variant replaces an enum variant (`--set eigensolver=LowestEigenpairs.Dense`)
        let is_variant = |v: &str| {
            let mut parts = v.split('.');
            let (Some(a), Some(b), None) = (parts.next(), parts.next(), parts.next()) else { return false };
            let ident = |w: &str| w.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && w.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            ident(a) && ident(b)
        };
        let want_variant = is_variant(value);
        if !want_str && !want_variant && value.parse::<f64>().is_err() {
            return Err(format!("--set {name}={value}: the value must be a number, a double-quoted string or an enum variant (`Type.Variant`)"));
        }
        let hit = lines.iter().position(|l| {
            let t = l.trim_end();
            let rest = match t.strip_prefix(name.as_str()) { Some(r) => r, None => return false };
            let rest = rest.trim_start();
            // a typed constant, `N: Int = 200`, keeps its type
            let rest = match rest.strip_prefix(':') {
                Some(r) => match r.split_once('=') {
                    Some((ty, v)) if ty.trim().chars().all(|c| c.is_ascii_alphanumeric() || c == '_') => v,
                    _ => return false,
                },
                None => match rest.strip_prefix('=') { Some(r) => r, None => return false },
            };
            let rest = rest.trim();
            let val = rest.split('#').next().unwrap_or("").trim();
            !t.starts_with(' ') && !t.starts_with('\t') && if want_str { is_str(val) } else if want_variant { is_variant(val) } else { val.parse::<f64>().is_ok() }
        });
        match hit {
            Some(i) => {
                let comment = lines[i].find('#').map(|k| lines[i][k..].to_string());
                let lhs = lines[i].split_once('=').map_or(name.as_str(), |(l, _)| l).trim_end().to_string();
                lines[i] = match comment {
                    Some(c) => format!("{lhs} = {value}    {c}"),
                    None => format!("{lhs} = {value}"),
                };
            }
            None => {
                let kind = if want_str { "\"<string>\"" } else if want_variant { "Type.Variant" } else { "<number>" };
                return Err(format!("--set {name}: the program has no top-level `{name} = {kind}` line to set"));
            }
        }
    }
    let mut out = lines.join("\n");
    if src.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}


use knotkit::diag::Diagnostic;
use knotkit::ir::IrProgram;
use knotkit::span::SourceId;

pub use lower::Viv;

fn join_errs(ds: &[Diagnostic]) -> String {
    ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n")
}

/// Compile Viv source to C text — the "show the generated C" view. Runs the
/// real lower + C backend, so it covers the entire language (unlike the v1
/// interpreter).
pub fn compile_c(src: &str) -> Result<String, String> {
    use knotkit::backend::{Backend, CBackend};
    use knotkit::io::MemoryIo;
    let prog = build_ir(src, SourceId(0)).map_err(|ds| join_errs(&ds))?;
    let emission = CBackend.emit(&prog, &MemoryIo::new()).map_err(|ds| join_errs(&ds))?;
    Ok(emission.artifact.as_c().expect("C backend produces C").to_string())
}

/// `compile_c` with each diagnostic rendered as the command line renders it:
/// `error: ...`, then `--> <file>:<line>:<col>` and the source line with its
/// caret, so an editor can place the error where it is.
pub fn compile_c_located(src: &str, file: &str) -> Result<String, String> {
    use knotkit::backend::{Backend, CBackend};
    use knotkit::diag::{PlainRenderer, Renderer};
    use knotkit::io::MemoryIo;
    let mut map = knotkit::span::SourceMap::new();
    map.add(file.to_string(), src.to_string());
    let rendered = |ds: &[Diagnostic]| ds.iter().map(|d| PlainRenderer.render(d, &map)).collect::<String>().trim_end().to_string();
    let prog = build_ir(src, SourceId(0)).map_err(|ds| rendered(&ds))?;
    let emission = CBackend.emit(&prog, &MemoryIo::new()).map_err(|ds| rendered(&ds))?;
    Ok(emission.artifact.as_c().expect("C backend produces C").to_string())
}

/// Compile source straight to a WebAssembly module with the direct backend
/// (no emcc, no LLVM). It lowers through `build_ir`, the same checked IR the C
/// backend compiles, debug lens included, so the module runs the program the
/// C build runs. The module imports its runtime from viv.wasm
/// (`wasm_backend`'s header says what it imports); `viv_emit_wasm` is this
/// function's browser export -- compile in the tab, then run those bytes.
pub fn compile_wasm(src: &str) -> Result<Vec<u8>, String> {
    let prog = build_ir(src, SourceId(0)).map_err(|ds| join_errs(&ds))?;
    wasm_backend::emit_wasm(&prog)
}

/// Annotation extraction v2 (the `viv doc` engine). The lexical convention:
///   #| what   #|| how   #||| why   — attach to the NEXT code line
///   #~ tag, tag                    — tags (function-level, above a def)
///   #: name description            — variable glossary entry (every name
///                                     a reader might have to mentally resolve)
///   #[ label: note ... #]          — labeled regions
/// All ordinary comments to the compiler. v2 groups the result per function:
/// markers directly above a `def` become that function's structured doc
/// (what / how / why / tags — introlang's summary / purpose / why reborn),
/// markers inside a def body become its line notes.
///
/// `scan` is the one shared pass; doc_json / whatis / lsfunc / lsvar are views
/// over its result.
struct Note {
    line: usize,
    level: u8,
    text: String,
}
struct FnSpan {
    name: String,
    start: usize,
    end: usize,
    /// a `placeholder def`: one line, no body
    placeholder: bool,
}
struct Scan<'a> {
    code_lines: Vec<&'a str>,
    raw_of: Vec<usize>, // raw source line (1-based) of each stripped code line
    notes: Vec<Note>,
    tags_at: Vec<(usize, Vec<String>)>,
    vars_at: Vec<(usize, String, String)>,
    blocks: Vec<(usize, usize, String, String)>,
    spans: Vec<FnSpan>,
}

fn scan(src: &str) -> Scan<'_> {
    let mut code_lines: Vec<&str> = Vec::new();
    let mut raw_of: Vec<usize> = Vec::new();
    let mut notes: Vec<Note> = Vec::new();
    let mut tags_pending: Vec<String> = Vec::new();
    let mut tags_at: Vec<(usize, Vec<String>)> = Vec::new(); // attached to next code line
    let mut vars_pending: Vec<(String, String)> = Vec::new();
    let mut vars_at: Vec<(usize, String, String)> = Vec::new();
    let mut blocks: Vec<(usize, usize, String, String)> = Vec::new();
    let mut open_blocks: Vec<(usize, String, String)> = Vec::new();
    let mut pending: Vec<(u8, String)> = Vec::new();
    // brace depth inside a `debug { }` block (the debug lens), whose lines are
    // neither code nor notes: a def's doc lines above the block still reach
    // the def below it
    let mut debug_depth: i64 = 0;

    for (raw_idx, raw) in src.lines().enumerate() {
        let t = raw.trim_start();
        if debug_depth > 0 || is_debug_open(t) {
            debug_depth += brace_change(t);
            continue;
        }
        // An annotation may run over several lines: a line at the SAME level as
        // the one before it continues that note instead of starting another.
        // (Before this, a three-line `#|` doc comment became three notes, and a
        // function's `what` was whichever came last.)
        let note = |level: u8, rest: &str, pending: &mut Vec<(u8, String)>| match pending.last_mut() {
            Some((l, text)) if *l == level => {
                text.push(' ');
                text.push_str(rest.trim());
            }
            _ => pending.push((level, rest.trim().to_string())),
        };
        if let Some(rest) = t.strip_prefix("#|||") {
            note(3, rest, &mut pending);
        } else if let Some(rest) = t.strip_prefix("#||") {
            note(2, rest, &mut pending);
        } else if let Some(rest) = t.strip_prefix("#|") {
            note(1, rest, &mut pending);
        } else if let Some(rest) = t.strip_prefix("#:") {
            // `#: NAME description`; a line indented past the name column continues it
            if rest.starts_with("  ") && !vars_pending.is_empty() {
                let last = vars_pending.last_mut().unwrap();
                last.1.push(' ');
                last.1.push_str(rest.trim());
            } else if let Some((name, desc)) = rest.trim().split_once(char::is_whitespace) {
                vars_pending.push((name.trim().to_string(), desc.trim().to_string()));
            }
        } else if let Some(rest) = t.strip_prefix("#~") {
            tags_pending.extend(
                rest.split(|c| c == ',' || c == ' ')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty()),
            );
        } else if let Some(rest) = t.strip_prefix("#[") {
            let text = rest.trim();
            let (label, note) = match text.split_once(':') {
                Some((l, n)) => (l.trim().to_string(), n.trim().to_string()),
                None => (String::new(), text.to_string()),
            };
            open_blocks.push((code_lines.len() + 1, label, note));
        } else if t.strip_prefix("#]").is_some() {
            if let Some((from, label, note)) = open_blocks.pop() {
                blocks.push((from, code_lines.len(), label, note));
            }
        } else if t.starts_with("#@") {
            // `#@ <hash>`: a def's intent stamp (lens.rs). It is neither code
            // nor a note, so the notes above it still reach the def below it.
        } else {
            code_lines.push(raw);
            raw_of.push(raw_idx + 1);
            if !raw.trim().is_empty() {
                for (level, text) in pending.drain(..) {
                    notes.push(Note { line: code_lines.len(), level, text });
                }
                if !tags_pending.is_empty() {
                    tags_at.push((code_lines.len(), std::mem::take(&mut tags_pending)));
                }
                for (n, d) in vars_pending.drain(..) {
                    vars_at.push((code_lines.len(), n, d));
                }
            }
        }
    }
    for (from, label, note) in open_blocks.into_iter().rev() {
        blocks.push((from, code_lines.len(), label, note));
    }

    // find def spans in the stripped code (start line -> closing brace)
    let mut spans: Vec<FnSpan> = Vec::new();
    let mut i = 0usize;
    while i < code_lines.len() {
        let t = code_lines[i].trim_start();
        if let Some(rest) = t.strip_prefix("placeholder def ") {
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            spans.push(FnSpan { name, start: i + 1, end: i + 1, placeholder: true });
            i += 1;
        } else if let Some(rest) = t.strip_prefix("def ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let start = i + 1;
            let mut depth = 0i64;
            let mut started = false;
            while i < code_lines.len() {
                let mut in_str = false;
                for ch in code_lines[i].chars() {
                    match ch {
                        '"' => in_str = !in_str,
                        '{' if !in_str => {
                            depth += 1;
                            started = true;
                        }
                        '}' if !in_str => depth -= 1,
                        _ => {}
                    }
                }
                i += 1;
                if started && depth == 0 {
                    break;
                }
            }
            spans.push(FnSpan { name, start, end: i, placeholder: false });
        } else {
            i += 1;
        }
    }

    Scan { code_lines, raw_of, notes, tags_at, vars_at, blocks, spans }
}

/// Opening minus closing braces on a line, outside strings and `#` comments.
pub(crate) fn brace_change(line: &str) -> i64 {
    let mut d = 0;
    let mut in_str = false;
    for ch in line.chars() {
        match ch {
            '"' => in_str = !in_str,
            '#' if !in_str => break,
            '{' if !in_str => d += 1,
            '}' if !in_str => d -= 1,
            _ => {}
        }
    }
    d
}

/// Whether a line opens a debug block: `debug {`, `debug name {`, `debug * {`.
pub(crate) fn is_debug_open(t: &str) -> bool {
    let Some(rest) = t.strip_prefix("debug") else { return false };
    let rest = rest.trim_start();
    if rest.starts_with('{') {
        return t.len() > 5 && (t.as_bytes()[5] == b'{' || t.as_bytes()[5].is_ascii_whitespace());
    }
    if !t[5..].starts_with(char::is_whitespace) {
        return false;
    }
    let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '*').collect();
    !name.is_empty() && rest[name.len()..].trim_start().starts_with('{')
}

fn esc(sv: &str) -> String {
    let mut out = String::with_capacity(sv.len() + 8);
    for c in sv.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// One `step` as the doc tools see it: the def it sits in (`<main>` at file
/// scope), its id and text, its nesting inside that def (1 = outermost), and
/// its raw source line (`line`) and its line in `doc_json`'s annotation-free
/// `code` (`code_line`, which is what a def's `line` counts in).
#[derive(Clone, Debug, PartialEq)]
pub struct StepInfo {
    pub def: String,
    pub id: String,
    pub text: String,
    pub depth: u32,
    pub line: usize,
    pub code_line: usize,
}

/// A step header at the start of a code line -- `step <id> "<text>" {`,
/// `step <id> {`, or the one-line `step <id> ["<text>"]: <statement>` -- as
/// (id, text, byte offset of its `{` or `:`). The parser's rule, read off one
/// line: `step` followed by a name and then a string, `{` or `:`.
fn step_header(line: &str) -> Option<(String, String, usize)> {
    let lead = line.len() - line.trim_start().len();
    let rest = line.trim_start().strip_prefix("step")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let after_ws = rest.trim_start();
    let id: String = after_ws.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if id.is_empty() || !id.starts_with(|c: char| c.is_alphabetic() || c == '_') {
        return None;
    }
    let mut pos = lead + 4 + (rest.len() - after_ws.len()) + id.len();
    let tail = &line[pos..];
    pos += tail.len() - tail.trim_start().len();
    let mut text = String::new();
    if line[pos..].starts_with('"') {
        // the same minimal escapes the lexer reads: \" \\ \n \t
        let mut chars = line[pos + 1..].char_indices();
        let mut closed = None;
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => {
                    closed = Some(i);
                    break;
                }
                '\\' => match chars.next() {
                    Some((_, 'n')) => text.push('\n'),
                    Some((_, 't')) => text.push('\t'),
                    Some((_, other)) => text.push(other),
                    None => {}
                },
                c => text.push(c),
            }
        }
        pos += 1 + closed? + 1;
        let tail = &line[pos..];
        pos += tail.len() - tail.trim_start().len();
    }
    (line[pos..].starts_with('{') || line[pos..].starts_with(':')).then_some((id, text, pos))
}

/// Every step in the source, in order. Text-based like the rest of `scan`, so
/// it answers for a file that does not parse, and it counts braces the way the
/// def-span finder does, outside strings and `#` comments.
pub fn steps(src: &str) -> Vec<StepInfo> {
    let sc = scan(src);
    let raw = |l: usize| sc.raw_of.get(l.wrapping_sub(1)).copied().unwrap_or(l);
    let mut out = Vec::new();
    let mut depth = 0i64;
    // brace depth just outside each open step, innermost last
    let mut open: Vec<i64> = Vec::new();
    // the depths of the one-line steps still open, innermost last
    let mut pending_one_line: Vec<i64> = Vec::new();
    for (i, line) in sc.code_lines.iter().enumerate() {
        let line_no = i + 1;
        let def = sc
            .spans
            .iter()
            .find(|sp| line_no >= sp.start && line_no <= sp.end)
            .map(|sp| sp.name.clone());
        if line.trim_start().starts_with("def ") {
            open.clear();
        }
        // a one-line step is open until its statement ends: the end of a
        // line at the depth it started at, unless `else` starts the next line
        let mut one_line = None;
        if let Some((id, text, at)) = step_header(line) {
            let def = def.unwrap_or_else(|| parser::MAIN_DEF.to_string());
            out.push(StepInfo { def, id, text, depth: open.len() as u32 + 1, line: raw(line_no), code_line: line_no });
            open.push(depth);
            if line[at..].starts_with(':') {
                one_line = Some(depth);
            }
        }
        let mut one_line_open: Vec<i64> = one_line.into_iter().collect();
        let mut in_str = false;
        let mut escaped = false;
        for ch in line.chars() {
            if in_str {
                match ch {
                    _ if escaped => escaped = false,
                    '\\' => escaped = true,
                    '"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match ch {
                '"' => in_str = true,
                '#' => break,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    while open.last().is_some_and(|&d| depth <= d) {
                        open.pop();
                    }
                }
                _ => {}
            }
        }
        // one-line steps whose statement ends with this line
        pending_one_line.append(&mut one_line_open);
        let else_next = sc.code_lines.get(i + 1).is_some_and(|l| l.trim_start().starts_with("else"));
        if !else_next {
            // (one whose block closed on this line was taken off `open` by its `}`)
            while let Some(&d) = pending_one_line.last().filter(|d| **d >= depth) {
                pending_one_line.pop();
                if d == depth && open.last() == Some(&d) {
                    open.pop();
                }
            }
        }
    }
    out
}

/// Answer "what is this variable?" from the glossary: every `#: name desc`
/// entry matching `query`, with its enclosing function for context.
pub fn whatis(src: &str, query: &str) -> String {
    let sc = scan(src);
    let all_steps = steps(src);
    let raw = |l: usize| sc.raw_of.get(l.wrapping_sub(1)).copied().unwrap_or(l);
    let scope_of = |line: usize| -> String {
        sc.spans
            .iter()
            .find(|sp| line >= sp.start && line <= sp.end)
            .map(|sp| format!("in {} (lines {}-{})", sp.name, raw(sp.start), raw(sp.end)))
            .unwrap_or_else(|| "at file scope".to_string())
    };
    let mut out = String::new();
    // a function by that name answers first: signature, doc, span
    if let Some(sp) = sc.spans.iter().find(|sp| sp.name == query) {
        let def_line = sc.code_lines[sp.start - 1].trim().trim_start_matches("def ");
        let sig = def_line.split('{').next().unwrap_or(def_line).trim_end();
        out.push_str(&format!("{} — function, lines {}-{}\n    {}\n", query, raw(sp.start), raw(sp.end), sig));
        let what = sc.notes.iter().find(|n| n.line == sp.start && n.level == 1);
        let tags: Vec<String> = sc
            .tags_at
            .iter()
            .filter(|(l, _)| *l == sp.start)
            .flat_map(|(_, ts)| ts.iter().cloned())
            .collect();
        if let Some(w) = what {
            out.push_str(&format!("    {}", w.text));
            if !tags.is_empty() {
                out.push_str(&format!("  [{}]", tags.join(", ")));
            }
            out.push('\n');
        } else if !tags.is_empty() {
            out.push_str(&format!("    [{}]\n", tags.join(", ")));
        }
        for st in all_steps.iter().filter(|st| st.def == query) {
            let text = if st.text.is_empty() { String::new() } else { format!("  {}", st.text) };
            out.push_str(&format!("    {}step {}{}  (line {})\n", "  ".repeat(st.depth as usize - 1), st.id, text, st.line));
        }
    }
    // a step, by its anchor `def/id` or by its bare id
    for st in &all_steps {
        if query == format!("{}/{}", st.def, st.id) || query == st.id {
            out.push_str(&format!("{}/{} — step in {}, line {}\n", st.def, st.id, st.def, st.line));
            if !st.text.is_empty() {
                out.push_str(&format!("    {}\n", st.text));
            }
        }
    }
    let hits: Vec<&(usize, String, String)> =
        sc.vars_at.iter().filter(|(_, n, _)| n == query).collect();
    for (l, n, d) in &hits {
        out.push_str(&format!("{} — {}\n    {}\n", n, d, scope_of(*l)));
    }
    if !out.is_empty() {
        return out;
    }
    // no exact hit: undocumented but present in the code, or a near-miss name
    let lq = query.to_lowercase();
    let near: Vec<String> = sc
        .vars_at
        .iter()
        .filter(|(_, n, _)| n.to_lowercase().contains(&lq) || lq.contains(&n.to_lowercase()))
        .map(|(l, n, d)| format!("  {} — {}  [{}]", n, d, scope_of(*l)))
        .collect();
    out = format!("no glossary entry for \"{}\"", query);
    let used_somewhere = sc.code_lines.iter().any(|cl| {
        cl.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .any(|w| w == query)
    });
    if used_somewhere {
        out.push_str(" (the name appears in the code — add a `#: ");
        out.push_str(query);
        out.push_str(" <description>` line)");
    }
    out.push('\n');
    if !near.is_empty() {
        out.push_str("close names:\n");
        out.push_str(&near.join("\n"));
        out.push('\n');
    }
    out
}

/// `viv src`: one function's raw source — its annotation block and body,
/// with raw line numbers in the gutter (they match any editor).
pub fn fn_src(src: &str, name: &str) -> String {
    let sc = scan(src);
    let Some(sp) = sc.spans.iter().find(|sp| sp.name == name) else {
        let known: Vec<&str> = sc.spans.iter().map(|sp| sp.name.as_str()).collect();
        return format!("no such function: {} (have: {})\n", name, known.join(", "));
    };
    let raw_start = sc.raw_of.get(sp.start - 1).copied().unwrap_or(sp.start);
    let raw_end = sc.raw_of.get(sp.end - 1).copied().unwrap_or(sp.end);
    let lines: Vec<&str> = src.lines().collect();
    // pull in the comment/annotation block sitting directly above the def
    let mut s0 = raw_start - 1;
    while s0 > 0 && lines[s0 - 1].trim_start().starts_with('#') {
        s0 -= 1;
    }
    let mut out = String::new();
    for (i, l) in lines[s0..raw_end.min(lines.len())].iter().enumerate() {
        out.push_str(&format!("{:>4}  {}\n", s0 + 1 + i, l));
    }
    out
}

/// `viv lsfunc`: the file's table of contents — every function with its line
/// span, signature, WHAT doc, and tags.
pub fn lsfunc(src: &str) -> String {
    lsfunc_where(src, false)
}

/// `viv lsfunc --placeholders`: only the defs that are still placeholders,
/// the signatures an agent has yet to write.
pub fn lsfunc_placeholders(src: &str) -> String {
    lsfunc_where(src, true)
}

fn lsfunc_where(src: &str, only_placeholders: bool) -> String {
    let sc = scan(src);
    if sc.spans.is_empty() {
        return "no functions\n".to_string();
    }
    if only_placeholders && !sc.spans.iter().any(|sp| sp.placeholder) {
        return "no placeholders\n".to_string();
    }
    let raw = |l: usize| sc.raw_of.get(l.wrapping_sub(1)).copied().unwrap_or(l);
    let all_steps = steps(src);
    let mut out = String::new();
    for sp in sc.spans.iter().filter(|sp| sp.placeholder || !only_placeholders) {
        // signature straight from the def line, minus the opening brace
        let def_line = sc.code_lines[sp.start - 1].trim().trim_start_matches("placeholder ").trim_start_matches("def ");
        let sig = def_line
            .split('{')
            .next()
            .unwrap_or(def_line)
            .trim_end()
            .to_string();
        let mark = if sp.placeholder { "  [placeholder: no body yet]" } else { "" };
        out.push_str(&format!("{:>4}-{}  {}{}\n", raw(sp.start), raw(sp.end), sig, mark));
        let what = sc
            .notes
            .iter()
            .find(|n| n.line == sp.start && n.level == 1)
            .map(|n| n.text.clone());
        let tags: Vec<String> = sc
            .tags_at
            .iter()
            .filter(|(l, _)| *l == sp.start)
            .flat_map(|(_, ts)| ts.iter().cloned())
            .collect();
        if what.is_some() || !tags.is_empty() {
            let mut doc = String::new();
            if let Some(w) = what {
                doc.push_str(&w);
            }
            if !tags.is_empty() {
                if !doc.is_empty() {
                    doc.push_str("  ");
                }
                doc.push_str(&format!("[{}]", tags.join(", ")));
            }
            out.push_str(&format!("          {}\n", doc));
        }
        // its steps, one line each, indented by nesting
        for st in all_steps.iter().filter(|st| st.def == sp.name) {
            let indent = "  ".repeat(st.depth as usize - 1);
            let text = if st.text.is_empty() { String::new() } else { format!("  {}", st.text) };
            out.push_str(&format!("    {:>4}  {}step {}{}\n", st.line, indent, st.id, text));
        }
    }
    out
}

/// `viv lsvar`: the whole variable glossary, grouped by function. An optional
/// function name narrows it to one group.
pub fn lsvar(src: &str, filter: &str) -> String {
    let sc = scan(src);
    if sc.vars_at.is_empty() {
        return "no glossary entries — document variables with `#: name description`\n".to_string();
    }
    // group name -> entries, functions in span order, file scope last
    let mut groups: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for sp in &sc.spans {
        let vars: Vec<(String, String)> = sc
            .vars_at
            .iter()
            .filter(|(l, _, _)| *l >= sp.start && *l <= sp.end)
            .map(|(_, n, d)| (n.clone(), d.clone()))
            .collect();
        if !vars.is_empty() {
            groups.push((sp.name.clone(), vars));
        }
    }
    let loose: Vec<(String, String)> = sc
        .vars_at
        .iter()
        .filter(|(l, _, _)| !sc.spans.iter().any(|sp| *l >= sp.start && *l <= sp.end))
        .map(|(_, n, d)| (n.clone(), d.clone()))
        .collect();
    if !loose.is_empty() {
        groups.push(("(file)".to_string(), loose));
    }
    if !filter.is_empty() {
        groups.retain(|(g, _)| g == filter);
        if groups.is_empty() {
            return format!("no glossary entries in \"{}\" — try `viv lsvar <file>` for all of them\n", filter);
        }
    }
    let mut out = String::new();
    for (g, vars) in &groups {
        out.push_str(&format!("{}:\n", g));
        let w = vars.iter().map(|(n, _)| n.chars().count()).max().unwrap_or(0);
        for (n, d) in vars {
            out.push_str(&format!("  {:<w$}  {}\n", n, d, w = w));
        }
    }
    out
}

pub fn doc_json(src: &str) -> String {
    let sc = scan(src);
    let (code_lines, notes, tags_at, vars_at, blocks, spans) =
        (&sc.code_lines, &sc.notes, &sc.tags_at, &sc.vars_at, &sc.blocks, &sc.spans);

    // steps, as JSON objects; a key appears only where there are steps, so a
    // file without them documents exactly as it did before steps existed
    let all_steps = steps(src);
    let steps_json = |def: &str| -> String {
        let list: Vec<String> = all_steps
            .iter()
            .filter(|st| st.def == def)
            .map(|st| format!(
                "{{\"id\":\"{}\",\"text\":\"{}\",\"depth\":{},\"line\":{},\"rawLine\":{}}}",
                esc(&st.id), esc(&st.text), st.depth, st.code_line, st.line
            ))
            .collect();
        if list.is_empty() { String::new() } else { format!(",\"steps\":[{}]", list.join(",")) }
    };
    // group: notes on a def line = its doc; notes inside its span = line notes
    let mut fn_json: Vec<String> = Vec::new();
    let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for sp in spans {
        let (mut what, mut how, mut why) = (String::new(), String::new(), String::new());
        let mut line_notes: Vec<String> = Vec::new();
        for (idx, n) in notes.iter().enumerate() {
            if n.line == sp.start {
                used.insert(idx);
                match n.level {
                    1 => what = n.text.clone(),
                    2 => how = n.text.clone(),
                    _ => why = n.text.clone(),
                }
            } else if n.line > sp.start && n.line <= sp.end {
                used.insert(idx);
                line_notes.push(format!(
                    "{{\"line\":{},\"level\":{},\"note\":\"{}\"}}",
                    n.line, n.level, esc(&n.text)
                ));
            }
        }
        let tags: Vec<String> = tags_at
            .iter()
            .filter(|(l, _)| *l == sp.start)
            .flat_map(|(_, ts)| ts.iter().cloned())
            .collect();
        let vars: Vec<String> = vars_at
            .iter()
            .filter(|(l, _, _)| *l >= sp.start && *l <= sp.end)
            .map(|(_, n, d)| format!("{{\"name\":\"{}\",\"desc\":\"{}\"}}", esc(n), esc(d)))
            .collect();
        let raw = |l: usize| sc.raw_of.get(l.wrapping_sub(1)).copied().unwrap_or(l);
        fn_json.push(format!(
            "{{\"name\":\"{}\",\"line\":{},\"end\":{},\"rawLine\":{},\"rawEnd\":{},\"what\":\"{}\",\"how\":\"{}\",\"why\":\"{}\",\"tags\":[{}],\"vars\":[{}],\"lineNotes\":[{}]{}{}}}",
            esc(&sp.name), sp.start, sp.end, raw(sp.start), raw(sp.end),
            esc(&what), esc(&how), esc(&why),
            tags.iter().map(|t| format!("\"{}\"", esc(t))).collect::<Vec<_>>().join(","),
            vars.join(","),
            line_notes.join(","),
            steps_json(&sp.name),
            if sp.placeholder { ",\"placeholder\":true" } else { "" }
        ));
    }
    let loose: Vec<String> = notes
        .iter()
        .enumerate()
        .filter(|(i, _)| !used.contains(i))
        .map(|(_, n)| format!(
            "{{\"line\":{},\"level\":{},\"note\":\"{}\"}}",
            n.line, n.level, esc(&n.text)
        ))
        .collect();
    let blocks_json: Vec<String> = blocks
        .iter()
        .map(|(f, t, l, n)| format!(
            "{{\"from\":{},\"to\":{},\"label\":\"{}\",\"note\":\"{}\"}}",
            f, t, esc(l), esc(n)
        ))
        .collect();
    let file_vars: Vec<String> = vars_at
        .iter()
        .filter(|(l, _, _)| !spans.iter().any(|sp| *l >= sp.start && *l <= sp.end))
        .map(|(_, n, d)| format!("{{\"name\":\"{}\",\"desc\":\"{}\"}}", esc(n), esc(d)))
        .collect();
    format!(
        "{{\"code\":\"{}\",\"functions\":[{}],\"vars\":[{}],\"lineNotes\":[{}],\"blockNotes\":[{}]{}}}",
        esc(&code_lines.join("\n")),
        fn_json.join(","),
        file_vars.join(","),
        loose.join(","),
        blocks_json.join(","),
        steps_json(parser::MAIN_DEF)
    )
}

/// The static dependency graph: which functions each def calls (user functions
/// only), plus the calls made from top-level code. This is the language-level
/// answer to "what depends on what" — no execution required, unlike the
/// interpreter's runtime call graph.
pub fn deps_json(src: &str) -> Result<String, String> {
    use ast::{Expr, Item, Stmt};
    let prog = parse(src, SourceId(0)).map_err(|ds| join_errs(&ds))?;
    let def_names: std::collections::HashSet<String> = prog
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Def(d) if !update::is_expression_def(&d.name) => Some(d.name.clone()),
            _ => None,
        })
        .collect();

    fn walk_expr(e: &Expr, names: &std::collections::HashSet<String>, out: &mut Vec<String>) {
        match e {
            Expr::Call { callee, args, .. } => {
                if names.contains(callee) && !out.contains(callee) {
                    out.push(callee.clone());
                }
                for a in args {
                    walk_expr(a, names, out);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                walk_expr(lhs, names, out);
                walk_expr(rhs, names, out);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk_expr(operand, names, out),
            Expr::Index { base, indices, .. } => {
                walk_expr(base, names, out);
                for i in indices {
                    walk_expr(i, names, out);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for el in elems {
                    walk_expr(el, names, out);
                }
            }
            Expr::Lambda { body, .. } => walk_expr(body, names, out),
            Expr::Field { base, .. } => walk_expr(base, names, out),
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    walk_expr(v, names, out);
                }
            }
            _ => {}
        }
    }
    fn walk_stmt(st: &Stmt, names: &std::collections::HashSet<String>, out: &mut Vec<String>) {
        match st {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } | Stmt::Show { expr: value, .. } | Stmt::ExprStmt { expr: value, .. } => walk_expr(value, names, out),
            Stmt::SetIndex { base, indices, value, .. } => {
                walk_expr(base, names, out);
                for i in indices {
                    walk_expr(i, names, out);
                }
                walk_expr(value, names, out);
            }
            Stmt::SetField { base, value, .. } => {
                walk_expr(base, names, out);
                walk_expr(value, names, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                walk_expr(cond, names, out);
                for s in then_body {
                    walk_stmt(s, names, out);
                }
                for s in else_body {
                    walk_stmt(s, names, out);
                }
            }
            Stmt::While { cond, body, .. } => {
                walk_expr(cond, names, out);
                for s in body {
                    walk_stmt(s, names, out);
                }
            }
            Stmt::For { bound, body, .. } => {
                walk_expr(bound, names, out);
                for s in body {
                    walk_stmt(s, names, out);
                }
            }
            Stmt::Test { body, .. } => {
                for s in body {
                    walk_stmt(s, names, out);
                }
            }
            _ => {}
        }
    }

    fn esc(sv: &str) -> String {
        sv.replace('\\', "\\\\").replace('"', "\\\"")
    }
    let mut fns = Vec::new();
    let mut main_calls: Vec<String> = Vec::new();
    for item in &prog.items {
        match item {
            // a def the record-update pass declared (`T { f: e, ..base }`
            // made into `viv__with_T__f(e, base)`), or one a lambda's body
            // got (crate::surface), is not the program's: its arguments are
            // walked where the expression stands
            Item::Def(d) if update::is_expression_def(&d.name) => {}
            Item::Def(d) => {
                let mut calls = Vec::new();
                for st in &d.body {
                    walk_stmt(st, &def_names, &mut calls);
                }
                fns.push(format!(
                    "{{\"name\":\"{}\",\"params\":{},\"calls\":[{}]}}",
                    esc(&d.name),
                    d.params.len(),
                    calls.iter().map(|c| format!("\"{}\"", esc(c))).collect::<Vec<_>>().join(",")
                ));
            }
            Item::Stmt(s) => walk_stmt(s, &def_names, &mut main_calls),
            Item::Record(_) | Item::Enum(_) | Item::Rung(_) => {}
        }
    }
    Ok(format!(
        "{{\"functions\":[{}],\"main_calls\":[{}]}}",
        fns.join(","),
        main_calls.iter().map(|c| format!("\"{}\"", esc(c))).collect::<Vec<_>>().join(",")
    ))
}

/// `viv canon`: rewrite with defs in canonical dependency order (callees
/// before callers, original order breaking ties), top-level code unchanged.
/// Function order is presentation; this is the canonical presentation.
pub fn canon(src: &str) -> Result<String, String> {
    let deps = deps_json(src)?; // also validates the parse
    // segment the SOURCE text: def blocks (with attached leading comments) vs rest
    let lines: Vec<&str> = src.lines().collect();
    let mut segments: Vec<(Option<String>, Vec<String>)> = Vec::new(); // (def name, lines)
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if t.starts_with("def ") {
            // pull the comment block attached directly above out of `rest`
            let mut attached: Vec<String> = Vec::new();
            while let Some(last) = rest.last() {
                let lt = last.trim_start();
                if lt.starts_with('#') && !lt.is_empty() {
                    attached.insert(0, rest.pop().unwrap());
                } else {
                    break;
                }
            }
            let name: String = t[4..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let mut block = attached;
            // brace-count to the end of the def (quotes stripped per line first)
            let mut depth = 0i64;
            let mut started = false;
            while i < lines.len() {
                let line = lines[i];
                let mut in_str = false;
                for ch in line.chars() {
                    match ch {
                        '"' => in_str = !in_str,
                        '{' if !in_str => {
                            depth += 1;
                            started = true;
                        }
                        '}' if !in_str => depth -= 1,
                        _ => {}
                    }
                }
                block.push(line.to_string());
                i += 1;
                if started && depth == 0 {
                    break;
                }
            }
            segments.push((Some(name), block));
        } else {
            rest.push(lines[i].to_string());
            i += 1;
        }
    }
    // parse the deps JSON minimally: name -> calls
    let mut order: Vec<String> = Vec::new();
    let mut calls: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for cap in deps.split("{\"name\":\"").skip(1) {
        let name: String = cap.chars().take_while(|c| *c != '"').collect();
        let cl = cap.split("\"calls\":[").nth(1).unwrap_or("");
        let cl: String = cl.chars().take_while(|c| *c != ']').collect();
        let list: Vec<String> = cl
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_matches('"').to_string())
            .collect();
        order.push(name.clone());
        calls.insert(name, list);
    }
    // topological sort, callees first; DFS in original order; cycles keep
    // original relative order (visited-set breaks the loop)
    let mut sorted: Vec<String> = Vec::new();
    let mut visiting: std::collections::HashSet<String> = std::collections::HashSet::new();
    fn visit(
        n: &str,
        calls: &std::collections::HashMap<String, Vec<String>>,
        visiting: &mut std::collections::HashSet<String>,
        sorted: &mut Vec<String>,
    ) {
        if sorted.iter().any(|x| x == n) || !visiting.insert(n.to_string()) {
            return;
        }
        if let Some(cs) = calls.get(n) {
            for c in cs {
                if c != n {
                    visit(c, calls, visiting, sorted);
                }
            }
        }
        if !sorted.iter().any(|x| x == n) {
            sorted.push(n.to_string());
        }
    }
    for n in &order {
        visit(n, &calls, &mut visiting, &mut sorted);
    }
    // emit: leading file comments stay first, then defs, then remaining code
    let mut head: Vec<String> = Vec::new();
    let mut tail: Vec<String> = rest;
    if segments.first().is_some() {
        // file header = comment/blank lines at the very top of `rest`
        let mut cut = 0;
        for (k, l) in tail.iter().enumerate() {
            let lt = l.trim_start();
            if lt.is_empty() || lt.starts_with('#') {
                cut = k + 1;
            } else {
                break;
            }
        }
        head = tail.drain(..cut).collect();
    }
    let mut out = String::new();
    for l in &head {
        out.push_str(l);
        out.push('\n');
    }
    for name in &sorted {
        if let Some((_, block)) = segments.iter().find(|(n, _)| n.as_deref() == Some(name)) {
            for l in block {
                out.push_str(l);
                out.push('\n');
            }
            out.push('\n');
        }
    }
    // drop leading blank lines of the tail (defs already added spacing)
    let skip = tail.iter().take_while(|l| l.trim().is_empty()).count();
    for l in &tail[skip..] {
        out.push_str(l);
        out.push('\n');
    }
    // canonical also means idempotent: collapse runs of blank lines
    while out.contains("\n\n\n") {
        out = out.replace("\n\n\n", "\n\n");
    }
    Ok(out)
}

/// One definition's content addresses: the IR node id (identity) and the
/// body hash (code, independent of the def's own name).
pub struct DefHash {
    pub name: String,
    pub kind: &'static str, // "def" | "main" | "global"
    pub node_short: String,
    pub body_short: String,
    pub display: String,
}

/// Per-definition content addresses for a source text.
pub fn def_hashes(src: &str) -> Result<Vec<DefHash>, String> {
    // The program as code: a def's hash is the same with or without its
    // debug lens, which is what lets a `.debug` entry be stamped with it.
    let prog = build_ir_plain(src, SourceId(0))
        .map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;

    let mut out = Vec::new();
    for root in prog.roots() {
        let node = prog.get(*root);
        // Body identity: BLAKE3 over the ordered child ids — the def's code,
        // independent of the def's own name.
        let mut h = knotkit::ir::Hasher::new();
        for c in &node.children {
            h.update(c.as_bytes());
        }
        let body = h.finalize();
        let body_short = body.to_hex().chars().take(8).collect::<String>();
        let (name, kind, display) = match &node.kind {
            Viv::Func { name, params, .. } => (
                name.clone(),
                "def",
                format!("def {}({} param{})", name, params.len(), if params.len() == 1 { "" } else { "s" }),
            ),
            Viv::Main { .. } => ("<main>".to_string(), "main", "top-level program".to_string()),
            Viv::Global { name, .. } => (name.clone(), "global", format!("global {}", name)),
            other => (format!("{:?}", other), "other", format!("{:?}", other)),
        };
        out.push(DefHash { name, kind, node_short: node.id.short(), body_short, display });
    }
    // A def that takes a function argument is compiled only as its
    // specializations (`lmder__mgh_fcn`), whose bodies depend on the caller.
    // It is hashed from its own tokens instead, so a lens entry for it can
    // still be stamped and go stale.
    let names: std::collections::HashSet<String> = out.iter().map(|h| h.name.clone()).collect();
    let ast_defs: Vec<String> = parse(src, SourceId(0)).map(|p| p.items.iter().filter_map(|i| if let ast::Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect()).unwrap_or_default();
    // every def the compiled program has only as specializations
    let generic: Vec<String> = ast_defs.into_iter().filter(|d| !names.contains(d) && prog.roots().iter().any(|r| matches!(&prog.get(*r).kind, Viv::Func { name, .. } if name.starts_with(&format!("{d}__"))))).collect();
    if !generic.is_empty() {
        let ast = parse(src, SourceId(0)).map_err(|ds| join_errs(&ds))?;
        let toks = lexer::lex(src, SourceId(0)).map_err(|ds| join_errs(&ds))?;
        for g in generic {
            let Some(d) = ast.items.iter().find_map(|i| if let ast::Item::Def(d) = i { (d.name == g).then_some(d) } else { None }) else { continue };
            let mut h = knotkit::ir::Hasher::new();
            // the parameters and the body: every token after the name, newlines left out
            for t in toks.iter().filter(|t| t.span.start > d.name_span.start && t.span.end <= d.span.end && !matches!(t.tok, lexer::Tok::Newline | lexer::Tok::Eof)) {
                // a number as it was always written here (`Num(1.0)`), whatever
                // the token now records of an integer's exact value
                let text = match &t.tok {
                    lexer::Tok::Num(v, _) => format!("Num({v:?})"),
                    other => format!("{other:?}"),
                };
                h.update(text.as_bytes());
            }
            let body = h.finalize();
            let body_short = body.to_hex().chars().take(8).collect::<String>();
            out.push(DefHash { name: g.clone(), kind: "def", node_short: body_short.clone(), body_short, display: format!("def {}({} params, specialized per function argument)", g, d.params.len()) });
        }
    }
    Ok(out)
}

/// Structural diff by content address: unchanged / edited / renamed
/// (same body, new name) / removed / added, as display text.
pub fn structural_diff(old: &[DefHash], new: &[DefHash]) -> String {
    let mut out = String::new();
    let mut unmatched_new: Vec<&DefHash> = new.iter().collect();
    let mut removed: Vec<&DefHash> = Vec::new();
    for o in old {
        if let Some(pos) = unmatched_new
            .iter()
            .position(|n| n.name == o.name && n.kind == o.kind && n.body_short == o.body_short)
        {
            let n = unmatched_new.remove(pos);
            out.push_str(&format!("unchanged  {}  (body {})", n.display, n.body_short));
        out.push('\n');
        } else if let Some(pos) = unmatched_new
            .iter()
            .position(|n| n.name == o.name && n.kind == o.kind)
        {
            let n = unmatched_new.remove(pos);
            out.push_str(&format!("edited     {}  (body {} -> {})", n.display, o.body_short, n.body_short));
        out.push('\n');
        } else {
            removed.push(o);
        }
    }
    // Second pass over the leftovers: same body under a new name = a rename.
    for o in removed {
        if let Some(pos) = unmatched_new
            .iter()
            .position(|n| n.kind == o.kind && n.body_short == o.body_short)
        {
            let n = unmatched_new.remove(pos);
            out.push_str(&format!("renamed    {} -> {}  (body {} unchanged)", o.name, n.name, n.body_short));
        out.push('\n');
        } else {
            out.push_str(&format!("removed    {}", o.display));
        out.push('\n');
        }
    }
    for n in unmatched_new {
        out.push_str(&format!("added      {}  (body {})", n.display, n.body_short));
        out.push('\n');
    }
    out
}

/// A host's expansion (the site's www/src/lib/uses.js): `request` is JSON,
/// `{"src": <program>, "modules": {<name>: <text or null>}}`, the texts the
/// host has fetched so far by the names the compiler asked for (null: no such
/// module). The answer is `{"text": <expanded program>}`, or `{"need":
/// [<names>]}` when the expansion asked for a module the request does not
/// carry (the host fetches them and asks again), or `{"error": <message>}`.
/// The modules the compiler carries (lib/cas) need not be sent. One
/// implementation of imports, so the site and the CLI cannot disagree.
pub fn expand_request(request: &str) -> String {
    use crate::story::Json;
    let q = |s: &str| crate::debug_lens::json_str(s);
    let req = match crate::story::parse_json(request) {
        Ok(j) => j,
        Err(e) => return format!("{{\"error\":{}}}", q(&format!("the request is not JSON: {e}"))),
    };
    let field = |k: &str| -> Option<Json> { if let Json::Obj(kv) = &req { kv.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()) } else { None } };
    let src = field("src").and_then(|j| j.str().map(str::to_string)).unwrap_or_default();
    let mut have: std::collections::HashMap<String, Option<String>> = std::collections::HashMap::new();
    if let Some(Json::Obj(kv)) = field("modules") {
        for (k, v) in kv {
            have.insert(k, v.str().map(str::to_string));
        }
    }
    let seen: Vec<String> = match field("seen") {
        Some(Json::Arr(xs)) => xs.iter().filter_map(|x| x.str().map(str::to_string)).collect(),
        _ => Vec::new(),
    };
    let mut need: Vec<String> = Vec::new();
    let out = expand_uses_after(&src, &mut |name: &str| match have.get(name) {
        Some(Some(t)) => Ok(t.clone()),
        Some(None) => Err(format!("no module `{name}`")),
        None => match crate::cas_check::lib_module(name) {
            Some(t) => Ok(t.to_string()),
            None => {
                if !need.iter().any(|n| n == name) {
                    need.push(name.to_string());
                }
                Err(format!("no module `{name}` yet"))
            }
        },
    }, &mut |lang, name, _| Err(format!("module `{name}`: it is written in `{lang}`, and the page cannot translate that")), &seen);
    if !need.is_empty() {
        return format!("{{\"need\":[{}]}}", need.iter().map(|n| q(n)).collect::<Vec<_>>().join(","));
    }
    match out {
        Ok(e) => {
            let units: Vec<String> = e.units.iter().skip(1).map(|u| q(&u.name)).collect();
            format!("{{\"text\":{},\"units\":[{}]}}", q(&e.text), units.join(","))
        }
        Err(e) => format!("{{\"error\":{}}}", q(&e)),
    }
}

/// `viv hash` as text: one line per definition.
pub fn hash_report(src: &str) -> Result<String, String> {
    let defs = def_hashes(src)?;
    Ok(defs
        .iter()
        .map(|d| format!("{}  body {}  {}\n", d.node_short, d.body_short, d.display))
        .collect())
}

/// `viv diff` as text: structural diff of two source texts.
pub fn diff_report(old_src: &str, new_src: &str) -> Result<String, String> {
    Ok(structural_diff(&def_hashes(old_src)?, &def_hashes(new_src)?))
}

/// `viv deps` for humans: one line per function — what it calls, who calls
/// it. (`deps_json` stays the machine format.)
pub fn deps_report(src: &str) -> Result<String, String> {
    let dj = deps_json(src)?;
    // parse our own emission (names never contain quotes)
    let mut fns: Vec<(String, String, Vec<String>)> = Vec::new(); // (name, params, calls)
    for cap in dj.split("{\"name\":\"").skip(1) {
        let name: String = cap.chars().take_while(|c| *c != '"').collect();
        let params: String = cap
            .split("\"params\":")
            .nth(1)
            .map(|r| r.chars().take_while(|c| c.is_ascii_digit()).collect())
            .unwrap_or_default();
        let calls: Vec<String> = cap
            .split("\"calls\":[")
            .nth(1)
            .map(|r| r.chars().take_while(|c| *c != ']').collect::<String>())
            .unwrap_or_default()
            .split(',')
            .filter(|x| !x.is_empty())
            .map(|x| x.trim_matches('"').to_string())
            .collect();
        fns.push((name, params, calls));
    }
    let main_calls: Vec<String> = dj
        .split("\"main_calls\":[")
        .nth(1)
        .map(|r| r.chars().take_while(|c| *c != ']').collect::<String>())
        .unwrap_or_default()
        .split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.trim_matches('"').to_string())
        .collect();
    let callers_of = |name: &str| -> Vec<String> {
        let mut cs: Vec<String> = fns
            .iter()
            .filter(|(n, _, calls)| n != name && calls.iter().any(|c| c == name))
            .map(|(n, _, _)| n.clone())
            .collect();
        if main_calls.iter().any(|c| c == name) {
            cs.push("top level".to_string());
        }
        cs
    };
    let w = fns.iter().map(|(n, _, _)| n.chars().count()).max().unwrap_or(0);
    let mut out = String::new();
    for (name, params, calls) in &fns {
        let mut line = format!("{:<w$}  ({} param{})", name, params, if params == "1" { "" } else { "s" }, w = w);
        if !calls.is_empty() {
            let shown: Vec<&str> = calls.iter().map(String::as_str).filter(|c| c != &name.as_str()).collect();
            if calls.iter().any(|c| c == name) {
                line.push_str("  recursive");
            }
            if !shown.is_empty() {
                line.push_str(&format!("  calls {}", shown.join(", ")));
            }
        }
        let cs = callers_of(name);
        if !cs.is_empty() {
            line.push_str(&format!("  ◂ {}", cs.join(", ")));
        }
        out.push_str(&line);
        out.push('\n');
    }
    if !main_calls.is_empty() {
        out.push_str(&format!("top level calls: {}\n", main_calls.join(", ")));
    }
    Ok(out)
}

/// Minimal wasm ABI for the browser workbench (no wasm-bindgen, mirrors Skein):
/// the page writes source bytes into a buffer from `viv_alloc`, calls
/// `viv_run`/`viv_compile`, and reads back a (ptr, len) packed into the u64.
#[cfg(target_arch = "wasm32")]
mod wasm_api {
    #[no_mangle]
    pub extern "C" fn viv_alloc(len: usize) -> *mut u8 {
        let mut v = vec![0u8; len];
        let p = v.as_mut_ptr();
        std::mem::forget(v);
        p
    }
    fn input(ptr: *mut u8, len: usize) -> String {
        let v = unsafe { Vec::from_raw_parts(ptr, len, len) };
        String::from_utf8_lossy(&v).into_owned()
    }
    fn pack(s: String) -> u64 {
        let bytes = s.into_bytes();
        let len = bytes.len() as u64;
        let ptr = bytes.as_ptr() as u64;
        std::mem::forget(bytes);
        (ptr << 32) | len
    }
    #[no_mangle]
    pub extern "C" fn viv_run(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::interp::run(&with_lib_cas(input(ptr, len))).unwrap_or_else(|e| format!("error: {e}")))
    }
    /// `viv_run` for the interactive views: a run that stops with an error
    /// gives what the program printed before it, then a NUL byte, then
    /// `error: ...`. A clean run gives the output, as `viv_run` does.
    #[no_mangle]
    pub extern "C" fn viv_run_keep(ptr: *mut u8, len: usize) -> u64 {
        pack(kept(crate::interp::run_keeping(&with_lib_cas(input(ptr, len)), false)))
    }
    /// `viv_run_cell` for the notebook view, keeping the cell's output before an
    /// error the way `viv_run_keep` does.
    #[no_mangle]
    pub extern "C" fn viv_run_cell_keep(ptr: *mut u8, total: usize, prefix_len: usize) -> u64 {
        let all = input(ptr, total);
        let cut = prefix_len.min(all.len());
        let (prefix, cell) = all.split_at(cut);
        pack(kept(crate::interp::run_cell_keeping(prefix, cell)))
    }
    fn kept(r: Result<String, (String, String)>) -> String {
        match r {
            Ok(out) => out,
            Err((printed, e)) => format!("{printed}\0error: {e}"),
        }
    }
    /// Release-mode run: contracted solves auto-select instead of sweeping.
    #[no_mangle]
    pub extern "C" fn viv_run_release(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::interp::run_release(&with_lib_cas(input(ptr, len))).unwrap_or_else(|e| format!("error: {e}")))
    }
    /// A program whose remaining imports name only lib/cas, which the
    /// compiler carries (crate::cas_check::LIB), with them expanded; any other
    /// program as it came (the host expands the modules it serves, through
    /// `viv_expand`).
    fn with_lib_cas(src: String) -> String {
        if !crate::imports::scan(&src).is_ok_and(|s| !s.is_empty()) {
            return src;
        }
        // an import the host did not expand, of a module the compiler does not
        // carry, stops the run with the compiler's words (it is never dropped)
        match crate::expand_uses(&src, &mut |name: &str| crate::cas_check::lib_module(name).map(str::to_string).ok_or_else(|| format!("no module `{name}`"))) {
            Ok(t) => t,
            Err(e) => format!("stop({})\n", crate::debug_lens::json_str(&e)),
        }
    }
    /// Incremental cell run: the buffer holds prefix+cell concatenated;
    /// `prefix_len` is the byte length of the prefix. When the prefix matches
    /// the live session, only the cell executes (see interp::run_cell).
    #[no_mangle]
    pub extern "C" fn viv_run_cell(ptr: *mut u8, total: usize, prefix_len: usize) -> u64 {
        let all = input(ptr, total);
        let cut = prefix_len.min(all.len());
        let (prefix, cell) = all.split_at(cut);
        pack(crate::interp::run_cell(prefix, cell).unwrap_or_else(|e| format!("error: {e}")))
    }
    #[no_mangle]
    pub extern "C" fn viv_compile(ptr: *mut u8, len: usize) -> u64 {
        // located, as the command line prints it: the editors put their
        // squiggle at the `--> program.viv:line:col` it gives
        pack(crate::compile_c_located(&input(ptr, len), "program.viv").unwrap_or_else(|e| if e.starts_with("error") { e } else { format!("error: {e}") }))
    }
    fn pack_bytes(bytes: Vec<u8>) -> u64 {
        let len = bytes.len() as u64;
        let ptr = bytes.as_ptr() as u64;
        std::mem::forget(bytes);
        (ptr << 32) | len
    }
    /// Compile source to a native wasm module and return its raw bytes. Success
    /// is self-describing: a valid module always begins with the wasm magic
    /// `\0asm` (00 61 73 6d), which no error string starts with, so the caller
    /// checks the first four bytes to tell a module from a UTF-8 error message.
    #[no_mangle]
    pub extern "C" fn viv_emit_wasm(ptr: *mut u8, len: usize) -> u64 {
        match crate::compile_wasm(&input(ptr, len)) {
            Ok(bytes) => pack_bytes(bytes),
            Err(e) => pack(format!("error: {e}")),
        }
    }
    #[no_mangle]
    pub extern "C" fn viv_deps(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::deps_json(&input(ptr, len)).unwrap_or_else(|e| format!("error: {e}")))
    }
    #[no_mangle]
    pub extern "C" fn viv_doc(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::doc_json(&input(ptr, len)))
    }
    #[no_mangle]
    pub extern "C" fn viv_whatis(sp: *mut u8, sl: usize, np: *mut u8, nl: usize) -> u64 {
        let (src, name) = (input(sp, sl), input(np, nl));
        pack(crate::whatis(&src, &name))
    }
    #[no_mangle]
    pub extern "C" fn viv_src(sp: *mut u8, sl: usize, np: *mut u8, nl: usize) -> u64 {
        let (src, name) = (input(sp, sl), input(np, nl));
        pack(crate::fn_src(&src, &name))
    }
    #[no_mangle]
    pub extern "C" fn viv_lsfunc(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::lsfunc(&input(ptr, len)))
    }
    #[no_mangle]
    pub extern "C" fn viv_lsvar(sp: *mut u8, sl: usize, fp: *mut u8, fl: usize) -> u64 {
        let (src, filter) = (input(sp, sl), input(fp, fl));
        pack(crate::lsvar(&src, &filter))
    }
    #[no_mangle]
    pub extern "C" fn viv_canon(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::canon(&input(ptr, len)).unwrap_or_else(|e| format!("error: {e}")))
    }
    #[no_mangle]
    /// `viv ladder`: the per-rung report of a ladder, as JSON, each rung run
    /// by the interpreter.
    #[no_mangle]
    pub extern "C" fn viv_ladder(ptr: *mut u8, len: usize) -> u64 {
        let src = input(ptr, len);
        pack(crate::ladder::report(&src, &mut |text| crate::interp::run(text)).unwrap_or_else(|e| format!("error: {e}")))
    }

    #[no_mangle]
    pub extern "C" fn viv_hash(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::hash_report(&input(ptr, len)).unwrap_or_else(|e| format!("error: {e}")))
    }
    /// The explorer's bundle for a program: `crate::graph::bundle_request_json`.
    #[no_mangle]
    pub extern "C" fn viv_explore_bundle(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::graph::bundle_request_json(&input(ptr, len)))
    }

    /// The whole program as a picture: `crate::graphlit::draw_request_json`.
    #[no_mangle]
    pub extern "C" fn viv_graph_draw(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::graphlit::draw_request_json(&input(ptr, len)))
    }

    /// Run a program as the explorer does: `crate::graph::explore_run_json`.
    #[no_mangle]
    pub extern "C" fn viv_explore_run(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::graph::explore_run_json(&input(ptr, len)))
    }

    /// The phrase table as JSON for an empty input (`crate::phrases::table_json`);
    /// for a program, what each of its phrases is (`crate::phrases::uses_report`).
    #[no_mangle]
    pub extern "C" fn viv_phrases(ptr: *mut u8, len: usize) -> u64 {
        let src = input(ptr, len);
        if src.trim().is_empty() {
            return pack(crate::phrases::table_json());
        }
        pack(crate::phrases::uses_report(&src).unwrap_or_else(|e| format!("error: {e}")))
    }

    /// `viv_expand`: a program's imports, spliced and resolved by the same
    /// code the CLI runs (crate::expand_request), the module texts given by
    /// the host. The answer asks for the modules it still needs.
    #[no_mangle]
    pub extern "C" fn viv_expand(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::expand_request(&input(ptr, len)))
    }

    /// `viv features <args>` (crate::features::cli_text): the arguments as one
    /// string, the text the CLI prints; `--json` gives the whole catalog.
    #[no_mangle]
    pub extern "C" fn viv_features(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::features::cli_text(&input(ptr, len)))
    }

    #[no_mangle]
    pub extern "C" fn viv_diff(op: *mut u8, ol: usize, np: *mut u8, nl: usize) -> u64 {
        let (o, n) = (input(op, ol), input(np, nl));
        pack(crate::diff_report(&o, &n).unwrap_or_else(|e| format!("error: {e}")))
    }
}

/// Parse source into an AST (the AST does not escape past `build_ir`).
pub fn parse(src: &str, source: SourceId) -> Result<ast::Program, Vec<Diagnostic>> {
    let program = parse_before_grad(src, source)?;
    // `grad` and `value_and_grad` (v4): a program that
    // calls neither comes back unchanged.
    let program = grad::transform(program)?;
    // `graph_of_self()` and `program_graph("m")` (crate::graphlit): a program
    // that calls neither comes back unchanged.
    graphlit::transform(program, src)
}

/// `parse` up to the rewrite of `grad`, which needs every def a gradient
/// reaches: what a reader of one module's own defs, with its imports not
/// spliced in, can ask for (`viv export` reads each file this way; B295).
pub fn parse_before_grad(src: &str, source: SourceId) -> Result<ast::Program, Vec<Diagnostic>> {
    let program = parse_keeping_rungs(src, source)?;
    // A ladder's rungs, flattened: the program is its top rung. `viv ladder`
    // is the one caller that wants them kept apart.
    let program = if program.items.iter().any(|i| matches!(i, ast::Item::Rung(_))) {
        let flat = {
            let (base, rungs, tail) = ladder::split(src, &program);
            ladder::prefix(&base, &rungs, &tail, rungs.len())
        };
        let toks = lexer::lex(&flat, source)?;
        parser::parse(toks, source)?
    } else {
        program
    };
    // the std defs the program's phrases call (crate::phrases): a program
    // with no phrase comes back unchanged
    let program = phrases::provide(program, src)?;
    // `return a, b` and `a, b = e` (crate::tuples): a program with neither
    // comes back unchanged.
    // factorize and factorize2 (lang/viv/std/groupby.viv), before the
    // tuples pass that unpacks their results
    let program = stdlib::provide_groupby(program, source)?;
    // `kernel def` (crate::cuda): the emulator's loops over blocks and
    // threads, for the engines; a reader of the text as written
    // (`parse_keeping_rungs`) sees the kernel as written
    let program = cuda::lower(program, src)?;
    let program = tuples::transform(program)?;
    // `T { f: e, ..base }` (crate::update): a program with none comes back
    // unchanged.
    let program = update::transform(program)?;
    // pi, e, inf; min(a, b, c); reversed; pop; the std defs a program calls
    // (crate::stdlib; after the unpacking above, so `e, f = g()` binds `e`):
    // a program with none of them comes back unchanged
    let program = stdlib::transform(program, source)?;
    // `a = append(a, x)` is the one form a grow is written in (crate::growable)
    growable::check(&program)?;
    check_clock(&program, src)?;
    check_figures(&program)?;
    printfmt::check(&program)?;
    check_parallel_records(&program)?;
    // what each pass of a `parallel for` writes (crate::parallel), with the
    // loops the memory lens pins as disjoint (crate::memory_lens)
    let pins = memory_lens::pins_for(src)?;
    let mut program = program;
    program.memory = parallel::check(&program, &pins)?;
    Ok(program)
}

/// A step must not run inside a `parallel for`. The parser refuses a step
/// written in the loop body; this refuses one the body reaches through a call.
/// The compiled loop runs its iterations on several threads, so the step's
/// events would be recorded in no fixed order (and the C recorder is not
/// thread-safe), while the interpreter runs the iterations in order: the two
/// engines could not print the same `__steps__` line.
///
/// Statically: a def "runs a step" if its body has one or names (calls, or
/// passes as a function value) a def that does; a top-level binding whose value
/// names such a def as a value counts too, since it then holds a function. A parallel body
/// that names any of them is refused. A function passed in through a parameter
/// is resolved by checking the defunctionalized program as well, where
/// `run(sq)` has become `run__sq` with `sq` named in its body.
///
/// A program with no step is not checked at all, so this adds nothing to the
/// programs that existed before steps.
///
/// The debug lens records the same way (`debug_lens::instrument` checks the
/// instrumented program again), so a def with a lens statement -- a traced
/// def, a watch, a hold -- counts as one that runs a step. A placeholder body
/// records nothing.
pub(crate) fn check_parallel_records(program: &ast::Program) -> Result<(), Vec<Diagnostic>> {
    use ast::{Expr, Item, Stmt};
    use knotkit::span::Span;

    fn walk_stmts(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
        for s in body {
            f(s);
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    walk_stmts(then_body, f);
                    walk_stmts(else_body, f);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => walk_stmts(body, f),
                _ => {}
            }
        }
    }
    /// A statement that records into the step events or the story.
    fn records(s: &Stmt) -> bool {
        match s {
            Stmt::StepEnter { .. } => true,
            Stmt::Lens { lens, .. } => !matches!(**lens, ast::Lens::Placeholder { .. }),
            _ => false,
        }
    }
    /// Every name a block uses as a callee (true) or a value (false), with where.
    fn names(body: &[Stmt]) -> Vec<(String, Span, bool)> {
        let mut out = Vec::new();
        defun::walk_block(body, &mut |e: &Expr| match e {
            Expr::Call { callee, span, .. } => out.push((callee.clone(), *span, true)),
            Expr::Var { name, span } => out.push((name.clone(), *span, false)),
            _ => {}
        });
        out
    }
    fn bodies(program: &ast::Program) -> Vec<(Option<&str>, &[Stmt])> {
        program
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Def(d) => Some((Some(d.name.as_str()), d.body.as_slice())),
                Item::Stmt(s) => Some((None, std::slice::from_ref(s))),
                _ => None,
            })
            .collect()
    }
    fn check(program: &ast::Program) -> Result<(), Vec<Diagnostic>> {
        let items = bodies(program);
        // Which defs (and top-level bindings) run a step, to a fixpoint.
        let mut stepping: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (name, body) in &items {
            let mut has = false;
            walk_stmts(body, &mut |s| has |= records(s));
            if let (Some(n), true) = (name, has) {
                stepping.insert(n.to_string());
            }
        }
        let used: Vec<(Option<&str>, &[Stmt], Vec<(String, Span, bool)>)> =
            items.iter().map(|(n, b)| (*n, *b, names(b))).collect();
        loop {
            let before = stepping.len();
            for (name, body, refs) in &used {
                match name {
                    Some(n) => {
                        if refs.iter().any(|(r, _, _)| stepping.contains(r)) {
                            stepping.insert(n.to_string());
                        }
                    }
                    // A top-level binding holds a stepping function only if its
                    // value names one as a value (`r = [sq, cube]`); a binding
                    // that stores what such a function returned (`e = sq(2)`)
                    // is a plain value.
                    None => {
                        if let [Stmt::Assign { name, .. }] = body {
                            if refs.iter().any(|(r, _, call)| !call && stepping.contains(r)) {
                                stepping.insert(name.clone());
                            }
                        }
                    }
                }
            }
            if stepping.len() == before {
                break;
            }
        }
        for (_, body, _) in &used {
            let mut found: Option<(String, Span)> = None;
            walk_stmts(body, &mut |s| {
                if let (None, Stmt::For { parallel: true, body, .. }) = (&found, s) {
                    found = names(body).into_iter().find(|(r, _, _)| stepping.contains(r)).map(|(r, sp, _)| (r, sp));
                }
            });
            if let Some((name, span)) = found {
                return Err(vec![Diagnostic::error(format!("a `parallel for` reaches a `step` through `{name}`"))
                    .with_primary(span, format!("`{name}` runs a step or records into the story"))
                    .with_note("the iterations run on several threads, so the order of the step's events would not be fixed; move the step out of the def or make the loop a plain `for`")]);
            }
        }
        Ok(())
    }

    let mut any = false;
    for (_, body) in bodies(program) {
        walk_stmts(body, &mut |s| any |= records(s));
    }
    if !any {
        return Ok(());
    }
    check(program)?;
    // A program the compiler cannot specialize is interpreter-only; the check
    // above is all that applies to it.
    match defun::specialize(ast::Program { items: program.items.clone(), debug: Vec::new(), memory: Vec::new(), phrases: Vec::new(), whole: program.whole.clone() }) {
        Ok(spec) => check(&spec),
        Err(_) => Ok(()),
    }
}

/// Parse without flattening a ladder's rungs.
pub fn parse_keeping_rungs(src: &str, source: SourceId) -> Result<ast::Program, Vec<Diagnostic>> {
    let toks = lexer::lex(src, source)?;
    parser::parse_with_sigs(toks, source, &used_phrases(src), &used_sigs(src))
}

/// The signatures of the defs `src` imports by name (`from a.b import f, g as
/// h`, under the name it binds), from the modules `expand_uses` has resolved
/// in this process: what a module read on its own needs to read keyword
/// arguments to them (B274). A spliced program has the defs in its text and
/// needs none; empty when `src` imports nothing remembered.
pub fn used_sigs(src: &str) -> std::collections::HashMap<String, Vec<kwargs::SigParam>> {
    let mut out = std::collections::HashMap::new();
    let Ok(stmts) = imports::scan(src) else { return out };
    if stmts.is_empty() {
        return out;
    }
    let texts = match module_texts().lock() {
        Ok(m) => m.clone(),
        Err(_) => return out,
    };
    for s in &stmts {
        for item in &s.items {
            let imports::ImportItem::From { path, names } = item else { continue };
            // the module, as the program's own candidates name it, or else a
            // remembered module whose name ends in the path (a module read
            // on its own does not know its unit name)
            let tail = format!("/{}", path.join("/"));
            let found = imports::candidates("", path).iter().map(|c| imports::canonical(c)).find(|c| texts.contains_key(c)).or_else(|| {
                let mut keys: Vec<&String> = texts.keys().filter(|k| k.ends_with(&tail) || **k == tail[1..]).collect();
                keys.sort();
                keys.first().map(|k| k.to_string())
            });
            let Some(text) = found.and_then(|k| texts.get(&k)) else { continue };
            let Ok(toks) = lexer::lex(text, SourceId(u32::MAX)) else { continue };
            // a contract's def, with its selection and settings (crate::contracts)
            let toks = contracts::desugar(toks.clone(), &contracts::imported_names(&toks).into_iter().collect()).unwrap_or(toks);
            let sigs = kwargs::def_sigs(&toks);
            for (n, alias) in names {
                if let Some(sig) = sigs.get(n) {
                    out.insert(alias.clone().unwrap_or_else(|| n.clone()), sig.clone());
                }
            }
        }
    }
    out
}

// ---- the phrases a module's `#! use` lines bring in ---------------------------------

/// The text of every module `expand_uses` has resolved in this process, by
/// the name its `#! use` line gives it. A tool that parses a module on its own
/// (the lens tools read each module's defs from the module's own text) reads
/// the phrases the module's `#! use` lines bring in from here.
fn module_texts() -> &'static std::sync::Mutex<std::collections::HashMap<String, String>> {
    static TEXTS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, String>>> = std::sync::OnceLock::new();
    TEXTS.get_or_init(Default::default)
}

/// Record the text of module `name`, as `#! use name` resolved it.
pub fn remember_module(name: &str, text: &str) {
    if let Ok(mut m) = module_texts().lock() {
        m.insert(name.to_string(), text.to_string());
    }
}

/// The `phrase` declarations of the modules `src`'s `#! use` lines name, and
/// of the modules those use, each module once: what a program that has not
/// been expanded needs to read the phrases those modules declare. Empty when
/// `src` uses no module (an expanded program has the declarations in it).
pub fn used_phrases(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let texts = match module_texts().lock() {
        Ok(m) => m.clone(),
        Err(_) => return out,
    };
    // `#! use` names, and the modules import lines name, as remembered
    let uses = |unit: &str, text: &str| -> Vec<String> { imports::known_refs(unit, text, &|c: &str| texts.contains_key(c)).into_iter().map(|(m, _)| m).collect() };
    let mut seen: Vec<String> = Vec::new();
    let mut stack = uses("", src);
    stack.reverse();
    while let Some(name) = stack.pop() {
        if seen.contains(&name) {
            continue;
        }
        seen.push(name.clone());
        let Some(text) = texts.get(&name) else { continue };
        for l in text.lines() {
            if l.starts_with("phrase \"") {
                out.push(l.to_string());
            }
        }
        let mut more = uses(&name, text);
        more.reverse();
        stack.extend(more);
    }
    out
}

/// A chart's title and axis labels are part of the PROGRAM. Whatever draws a
/// `plot` -- a web page, a notebook, a terminal -- shows what `figure(title,
/// xlabel, ylabel)` declared and adds nothing of its own, so a program that
/// plots must say what it is plotting first.
///
/// Checked here, on the AST, because every engine and every tool parses through
/// this one function: the interpreter and the compiler refuse the same programs.
///
/// The rule, statically: a top-level plot must come after a top-level `figure`
/// in source order. A `def` that plots must declare its own figure first, or the
/// program must declare one at the top level (the def is then a drawing helper,
/// called after it).
/// The line that marks a program as a timing program: one that may read the
/// wall time with `clock()`. Its output differs from run to run, so the
/// three-engine comparisons (tools/three-engines.sh, tools/hub-golden.sh)
/// leave its runs out.
pub const TIMING_MARK: &str = "#! timing";

/// Whether the program text carries `#! timing` on a line of its own.
pub fn is_timing(src: &str) -> bool {
    src.lines().any(|l| l.trim_end() == TIMING_MARK)
}

/// `clock()` is refused outside a `#! timing` program, in every engine.
fn check_clock(program: &ast::Program, src: &str) -> Result<(), Vec<Diagnostic>> {
    let own = program.items.iter().any(|i| matches!(i, ast::Item::Def(d) if d.name == "clock"));
    if own || is_timing(src) {
        return Ok(());
    }
    match recursion::first_call(program, "clock") {
        Some(span) => Err(vec![Diagnostic::error("`clock()` reads the wall time, which differs from run to run")
            .with_primary(span, "in a program not marked as a timing program")
            .with_note("put `#! timing` on a line of its own at the top of the program; the engine comparisons then leave its output out")]),
        None => Ok(()),
    }
}

fn check_figures(program: &ast::Program) -> Result<(), Vec<Diagnostic>> {
    use ast::{Expr, Item, Stmt};
    // the plots, and the settings of a figure (xscale, yscale, aspect, style,
    // yright, origin, colormap, clim, levels), which mean nothing without one
    const PLOTS: [&str; 17] =
        ["plot", "plot2d", "plot3d", "scatter", "quiver", "bars", "fill_between", "rangebars", "xscale", "yscale", "aspect", "style", "yright", "origin", "colormap", "clim", "levels"];
    // the settings of a grid of figures, which mean nothing without one
    const GRID_SETTINGS: [&str; 1] = ["share"];

    /// What has been declared so far, in source order: a figure, a grid.
    #[derive(Clone, Copy, Default)]
    struct Declared {
        figure: bool,
        grid: bool,
    }

    /// Walk statements in source order; returns the first call found that
    /// needs a declaration not yet made, and what it needs.
    fn first_bare_plot<'a>(body: &'a [Stmt], declared: &mut Declared, own: &[&str]) -> Option<(&'a str, knotkit::span::Span, &'static str)> {
        for s in body {
            let call = match s {
                Stmt::ExprStmt { expr: Expr::Call { callee, span, .. }, .. } => Some((callee.as_str(), *span)),
                Stmt::Assign { value: Expr::Call { callee, span, .. }, .. } => Some((callee.as_str(), *span)),
                _ => None,
            };
            // a program's own `def share(...)` is its function, not the grid setting
            if let Some((callee, span)) = call.filter(|(c, _)| !own.contains(c)) {
                if callee == "figure" {
                    declared.figure = true;
                } else if callee == "grid" {
                    declared.grid = true;
                } else if PLOTS.contains(&callee) && !declared.figure {
                    return Some((callee, span, "figure"));
                } else if GRID_SETTINGS.contains(&callee) && !declared.grid {
                    return Some((callee, span, "grid"));
                }
            }
            let found = match s {
                Stmt::If { then_body, else_body, .. } => {
                    first_bare_plot(then_body, declared, own).or_else(|| first_bare_plot(else_body, declared, own))
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => first_bare_plot(body, declared, own),
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }

    let top: Vec<Stmt> = program.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    let defs: Vec<&str> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.as_str()) } else { None }).collect();
    let mut declared = Declared::default();
    let mut bare = first_bare_plot(&top, &mut declared, &defs).map(|(c, sp, need)| (c.to_string(), sp, need));
    let top_level = declared;
    if bare.is_none() {
        for item in &program.items {
            if let Item::Def(def) = item {
                let mut own = top_level;
                if let Some((c, sp, need)) = first_bare_plot(&def.body, &mut own, &defs) {
                    bare = Some((c.to_string(), sp, need));
                    break;
                }
            }
        }
    }
    match bare {
        None => Ok(()),
        Some((callee, span, "grid")) => Err(vec![Diagnostic::error(format!("`{callee}` needs a `grid(title, rows, cols)` before it"))
            .with_primary(span, "no grid of figures has been declared for this setting")]),
        Some((callee, span, _)) => Err(vec![Diagnostic::error(format!("`{callee}` needs a `figure(title, xlabel, ylabel)` before it"))
            .with_primary(span, "no figure has been declared for this chart")
            .with_note("a chart's title and axis labels are part of the program: nothing outside it adds them")]),
    }
}

/// Parse, lower, and type-check source into Viv IR, with the debug lens
/// compiled in (it runs only in a debug build; see `debug_lens`).
pub fn build_ir(src: &str, source: SourceId) -> Result<IrProgram<Viv>, Vec<Diagnostic>> {
    // The recursion limit is Viv code both engines run (`recursion::guard`);
    // the interpreter adds the same in `interp::parse_items_diag`.
    let ast = debug_lens::instrument(recursion::guard(parse(src, source)?))?;
    // Function-valued arguments are rewritten away first (specialization), so
    // lowering only ever sees first-order calls — the compiled backends match
    // the interpreter's surface semantics.
    let ast = defun::specialize(ast)?;
    // slices read in place where they are used (crate::fuse): the compiled
    // engines only, after everything that reads the program as written
    let ast = fuse::substitute(ast);
    lower::lower(&ast, source)
}

/// `build_ir` without the debug lens: the program as code, which is what the
/// body hashes are computed from.
pub fn build_ir_plain(src: &str, source: SourceId) -> Result<IrProgram<Viv>, Vec<Diagnostic>> {
    let mut ast = parse(src, source)?;
    ast.debug.clear();
    ast.memory.clear();
    // Function-valued arguments are rewritten away first (specialization), so
    // lowering only ever sees first-order calls — the compiled backends match
    // the interpreter's surface semantics.
    let ast = defun::specialize(ast)?;
    lower::lower(&ast, source)
}
