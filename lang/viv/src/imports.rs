// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Imports and module namespaces.
//!
//! Python's three forms, with Python's binding rules:
//!
//! ```viv
//! import nn                          # nn.rmsnorm(X)
//! import hf.scf                      # binds `hf`: hf.scf.run(...)
//! import hf.scf as scf               # scf.run(...)
//! from nn import rmsnorm, softmax    # rmsnorm(X)
//! from hf import scf, mole           # submodules: scf.run(...), mole.water(...)
//! ```
//!
//! All of it happens in the front end, on text, before anything parses the
//! program, so the interpreter, C and wasm see a whole program with unique
//! names as they always have, and a run pays nothing for it:
//!
//! 1. **Lowering** (`lower`): each import line of a text becomes a
//!    `#! import <module>` line naming the file it found, which
//!    `crate::expand_uses_mapped` splices as it splices `#! use` (once per
//!    program, in place). A line of a parenthesized list after the first
//!    becomes `#! import-continued`, which the splice drops, so every other
//!    line keeps its number. `private ` before a top-level `def`, record,
//!    enum or constant is blanked to spaces and remembered.
//! 2. **Resolution** (`resolve`): once the program is spliced, each unit's
//!    lines (the program's, each module's) are parsed on their own to find
//!    the names the unit defines and the names it reads. A name a unit reads
//!    from a module it reaches only through another (a transitive module) is
//!    an error; so is a read of a name that is `private` to another module,
//!    an import of a name a module does not define, and an import that
//!    collides with a name the file defines.
//! 3. **Renaming**: a top-level name two units define becomes two names, the
//!    module's written `<module path>__<name>` (`lib_cas_cas__K_NUM`), the
//!    program's left as it is; a qualified name (`nn.rmsnorm`) becomes the
//!    name it means; an alias (`from nn import softmax as sm`) becomes the
//!    name it stands for. The rewrite goes through the lexer's tokens, so a
//!    field (`r.n`), a keyword argument (`f(n=1)`) and a record field
//!    (`R { n: 1 }`) keep their spelling.
//!
//! A module still reads a name it neither defines nor imports from the
//! program that imports it, as `#! use` let it: the decks of dft/ set
//! `ECUT` and `NKMESH` before they import the engine that reads them. What a
//! module cannot read is a name of a module it imports only through another.
//!
//! Search path, for `import a.b` in the program: `a/b.viv` beside the program
//! or at the repository root (the CLI's resolver looks there in that order),
//! then `lib/a/b.viv`; in a module `d/m`: `d/a/b.viv`, `lib/a/b.viv`,
//! `a/b.viv`. Two different files that match are an error naming both.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::{Expr, Item as AstItem, Program, Stmt, TypeExpr};
use crate::lexer::{FPart, Tok, Token};
use knotkit::span::SourceId;

/// The separator between a module's path and a name of it, in a renamed name.
pub const SEP: &str = "__";

/// The line a lowered import becomes, before the modules it names.
pub const IMPORT_DIRECTIVE: &str = "#! import ";
/// A line of a parenthesized import list after the first.
pub const CONTINUED: &str = "#! import-continued";

/// One import statement as written.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportStmt {
    /// its first line, 0-based
    pub line: usize,
    /// how many lines it takes
    pub nlines: usize,
    pub items: Vec<ImportItem>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImportItem {
    /// `import a.b` or `import a.b as c`
    Module { path: Vec<String>, alias: Option<String> },
    /// `from a.b import n, m as k`
    From { path: Vec<String>, names: Vec<(String, Option<String>)> },
}

impl ImportItem {
    pub fn path(&self) -> &[String] {
        match self {
            ImportItem::Module { path, .. } | ImportItem::From { path, .. } => path,
        }
    }
}

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_') && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

/// A line with its `#` comment removed (a `#` in a string is not one).
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

/// Whether a line (its leading space removed) starts an import statement:
/// `import <name>` or `from <dotted> import ...`.
pub fn starts_import(t: &str) -> bool {
    if let Some(rest) = t.strip_prefix("import ") {
        return rest.trim_start().chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    }
    let Some(rest) = t.strip_prefix("from ") else { return false };
    let rest = rest.trim_start();
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    let dotted = &rest[..end];
    let after = rest[end..].trim_start();
    !dotted.is_empty()
        && dotted.split('.').all(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        && (after == "import" || after.starts_with("import ") || after.starts_with("import("))
}

fn parse_dotted(s: &str, line: usize) -> Result<Vec<String>, String> {
    let segs: Vec<String> = s.trim().split('.').map(|x| x.trim().to_string()).collect();
    for seg in &segs {
        if !is_ident(seg) {
            if seg.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return Err(format!("line {}: `{}`: a module name is a name, and a name cannot start with a digit (the Pople basis sets are basis.pople631g, basis.pople631gs and basis.pople631gss)", line + 1, s.trim()));
            }
            return Err(format!("line {}: `{}` is not a module name (names joined by `.`, like hf.scf)", line + 1, s.trim()));
        }
    }
    Ok(segs)
}

/// `name` or `name as alias`.
fn parse_named(s: &str, line: usize) -> Result<(String, Option<String>), String> {
    let words: Vec<&str> = s.split_whitespace().collect();
    match words.as_slice() {
        [n] if is_ident(n) => Ok((n.to_string(), None)),
        [n, "as", a] if is_ident(n) && is_ident(a) => Ok((n.to_string(), Some(a.to_string()))),
        _ => Err(format!("line {}: `{}`: an imported name is `name` or `name as other`", line + 1, s.trim())),
    }
}

/// Every import statement of a text, in order. Lines inside a `###` header
/// are prose. An import that is not at the start of its line (inside a def or
/// a block) is an error, as are `import *` and relative imports.
pub fn scan(text: &str) -> Result<Vec<ImportStmt>, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut fence = false;
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if l.trim() == "###" {
            fence = !fence;
            i += 1;
            continue;
        }
        let t = l.trim_start();
        if fence || !starts_import(t) {
            i += 1;
            continue;
        }
        if t.len() != l.len() {
            return Err(format!("line {}: an import goes at the top level of a file, not inside a def or a block", i + 1));
        }
        // the statement: up to the `)` that closes a parenthesized list
        let mut stmt = strip_comment(l).trim_end().to_string();
        let first = i;
        if stmt.contains('(') && !stmt.contains(')') {
            loop {
                i += 1;
                let Some(next) = lines.get(i) else {
                    return Err(format!("line {}: the import's `(` is never closed", first + 1));
                };
                stmt.push(' ');
                stmt.push_str(strip_comment(next).trim());
                if next.contains(')') {
                    break;
                }
            }
        }
        let items = parse_stmt(&stmt, first)?;
        out.push(ImportStmt { line: first, nlines: i - first + 1, items });
        i += 1;
    }
    Ok(out)
}

fn parse_stmt(stmt: &str, line: usize) -> Result<Vec<ImportItem>, String> {
    let t = stmt.trim();
    // `import post "<slug>" as name`: a post used as a module, the one form
    // with a string, and it needs `as`
    if let Some(rest) = t.strip_prefix("import post \"") {
        let (slug, after) = rest.split_once('"').ok_or_else(|| format!("line {}: the post's slug is never closed", line + 1))?;
        let words: Vec<&str> = after.split_whitespace().collect();
        let ok = !slug.is_empty() && slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '-');
        return match words.as_slice() {
            ["as", a] if ok && is_ident(a) => Ok(vec![ImportItem::Module { path: vec![format!("post:{slug}")], alias: Some(a.to_string()) }]),
            _ => Err(format!("line {}: a post is imported as `import post \"<slug>\" as <name>`", line + 1)),
        };
    }
    if let Some(rest) = t.strip_prefix("import ") {
        let mut items = Vec::new();
        for part in rest.split(',') {
            let words: Vec<&str> = part.split_whitespace().collect();
            let (path, alias) = match words.as_slice() {
                [p] => (parse_dotted(p, line)?, None),
                [p, "as", a] if is_ident(a) => (parse_dotted(p, line)?, Some(a.to_string())),
                _ => return Err(format!("line {}: `import {}`: write `import a.b` or `import a.b as c`", line + 1, part.trim())),
            };
            items.push(ImportItem::Module { path, alias });
        }
        return Ok(items);
    }
    let rest = t.strip_prefix("from ").unwrap_or(t);
    let at = rest.find(" import").ok_or_else(|| format!("line {}: `from` needs `import`", line + 1))?;
    let module = &rest[..at];
    if module.trim().starts_with('.') {
        return Err(format!("line {}: a relative import (`from .`): write the module's name", line + 1));
    }
    let path = parse_dotted(module, line)?;
    let mut list = rest[at + " import".len()..].trim();
    if let Some(inner) = list.strip_prefix('(') {
        list = inner.strip_suffix(')').ok_or_else(|| format!("line {}: the import's `(` is never closed", line + 1))?;
    }
    if list.trim() == "*" {
        return Err(format!("line {}: `import *` is refused: name what the file reads, so a reader can see where each name comes from", line + 1));
    }
    let mut names = Vec::new();
    for part in list.split(',') {
        if part.trim().is_empty() {
            continue;
        }
        names.push(parse_named(part, line)?);
    }
    if names.is_empty() {
        return Err(format!("line {}: `from {}` imports no name", line + 1, module.trim()));
    }
    Ok(vec![ImportItem::From { path, names }])
}

/// The names a dotted module path may mean, from a unit named `unit` (empty
/// for the program), in the order they are tried. `./a/b` is `a/b.viv`
/// beside the program, `/a/b` is `a/b.viv` under the repository root (the
/// resolver reads both forms); a module `d/m` looks beside itself first, as
/// `/d/a/b`. `canonical` turns the one found into the module's name.
pub fn candidates(unit: &str, path: &[String]) -> Vec<String> {
    let p = path.join("/");
    let mut out: Vec<String> = Vec::new();
    let mut add = |s: String| {
        if !out.contains(&s) {
            out.push(s);
        }
    };
    if path.first().is_some_and(|s| s == "lib" || s == "examples") {
        add(format!("/{p}"));
        return out;
    }
    if path.first().is_some_and(|s| s.starts_with("post:")) {
        add(path[0].clone());
        return out;
    }
    if unit.is_empty() {
        add(format!("./{p}"));
    } else {
        match unit.rsplit_once('/') {
            Some((dir, _)) if dir == "." || dir.starts_with("./") => add(format!("{dir}/{p}")),
            Some((dir, _)) => add(format!("/{dir}/{p}")),
            // a module at the repository root: beside it is the root
            None => add(format!("/{p}")),
        }
    }
    add(format!("/lib/{p}"));
    add(format!("/{p}"));
    out
}

/// A module's name, from the candidate that found it: `/lib/hf/scf` is
/// `lib/hf/scf`; `./seq` (beside the program) stays as it is.
pub fn canonical(c: &str) -> String {
    if c.starts_with("./") {
        return c.to_string();
    }
    c.strip_prefix('/').unwrap_or(c).to_string()
}

/// A module's dotted name as an import line writes it: `lib/hf/scf` is
/// `hf.scf`, `dft/kmesh` is `dft.kmesh`, `./seq` is `seq`.
pub fn dotted_of(canon: &str) -> String {
    let c = canon.strip_prefix("./").unwrap_or(canon);
    c.strip_prefix("lib/").unwrap_or(c).replace('/', ".")
}

/// The name a top-level name of module `canon` takes when it is renamed.
pub fn mangle(canon: &str, name: &str) -> String {
    let m: String = canon.strip_prefix("./").unwrap_or(canon).chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("{m}{SEP}{name}")
}

/// What an import binds, once its modules are found.
#[derive(Clone, Debug, Default)]
pub struct Pkg {
    /// the module this name is, when it is one
    pub module: Option<String>,
    /// `import a.b` binds `a`, whose `b` is the module
    pub kids: BTreeMap<String, Pkg>,
}

#[derive(Clone, Debug)]
pub enum Binding {
    /// a module name: `import a.b` (local `a`), `import a.b as c`, `from a import b` of a module
    Alias { local: String, pkg: Pkg, line: usize },
    /// a name of a module: `from m import n as local`
    Name { local: String, module: String, name: String, line: usize },
}

/// A text with its import lines lowered, and what they bind.
#[derive(Clone, Debug, Default)]
pub struct Lowered {
    pub text: String,
    pub bindings: Vec<Binding>,
    /// the modules the import lines name, in order
    pub modules: Vec<String>,
    /// top-level names marked `private`
    pub privates: Vec<String>,
    /// whether the text has an import line
    pub has_imports: bool,
}

/// Top-level names a text defines, read from its lines (for `from a import
/// b`, to tell a name from a submodule before anything is parsed).
pub fn line_tops(text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut fence = false;
    for l in text.lines() {
        if l.trim() == "###" {
            fence = !fence;
            continue;
        }
        if fence || l.starts_with(|c: char| c.is_whitespace()) {
            continue;
        }
        let t = l.strip_prefix("private ").unwrap_or(l).trim_start();
        let word = |s: &str| s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).next().unwrap_or("").to_string();
        if let Some(r) = t.strip_prefix("def ").or_else(|| t.strip_prefix("record ")).or_else(|| t.strip_prefix("enum ")).or_else(|| t.strip_prefix("contract ")) {
            let w = word(r.trim_start());
            if is_ident(&w) {
                out.insert(w);
            }
            continue;
        }
        let w = word(t);
        if is_ident(&w) {
            let after = t[w.len()..].trim_start();
            let typed = after.strip_prefix(':').map(|a| a.contains('=') && !a.contains("==")).unwrap_or(false);
            if (after.starts_with('=') && !after.starts_with("==")) || typed {
                out.insert(w);
            }
        }
    }
    out
}

thread_local! {
    /// Whether a `#! use` line is still read: only `viv imports`, which
    /// rewrites them, sets this.
    pub static LEGACY_USE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The error for a `#! use` line, which imports replaced: `None` when the
/// text has none (or `viv imports` is reading it).
pub fn refuse_use(unit: &str, text: &str) -> Option<String> {
    if LEGACY_USE.with(|c| c.get()) {
        return None;
    }
    let mut fence = false;
    for (i, l) in text.lines().enumerate() {
        if l.trim() == "###" {
            fence = !fence;
            continue;
        }
        if fence {
            continue;
        }
        if let Some(m) = l.trim_end().strip_prefix("#! use ") {
            let at = if unit.is_empty() { String::new() } else { format!("{unit}: ") };
            let m = m.trim();
            return Some(format!(
                "{at}line {}: `#! use {m}` is an import now: write `import {}` (or `from {} import <names>`); `viv imports --write <file>` rewrites a file's `#! use` lines",
                i + 1,
                dotted_of(m),
                dotted_of(m)
            ));
        }
    }
    None
}

/// The names of Python's math module that Viv has, as builtins
/// (lang/viv/std/builtins.viv's among them) or constants (crate::stdlib).
pub const MATH_NAMES: &[&str] = &[
    "pi", "e", "tau", "inf", "sqrt", "exp", "expm1", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
    "tanh", "floor", "ceil", "fabs", "pow", "hypot", "copysign", "fsum", "erf", "gamma",
];

/// `text` with its imports of Python's math module removed (each line left
/// empty, so every other line keeps its number) and `math.sqrt` written
/// `sqrt`; `None` when it imports no such module (or a math.viv exists).
fn python_math(unit: &str, text: &str, stmts: &[ImportStmt], fetch: &mut dyn FnMut(&str) -> Option<String>) -> Result<Option<String>, String> {
    let is_math = |p: &[String]| p.len() == 1 && p[0] == "math";
    let mut hits = Vec::new();
    let mut aliases: Vec<String> = Vec::new();
    for st in stmts {
        if !st.items.iter().all(|i| is_math(i.path())) {
            continue;
        }
        if candidates(unit, &["math".to_string()]).iter().any(|c| fetch(c).is_some()) {
            return Ok(None);
        }
        for item in &st.items {
            match item {
                ImportItem::Module { alias, .. } => aliases.push(alias.clone().unwrap_or_else(|| "math".to_string())),
                ImportItem::From { names, .. } => {
                    for (n, alias) in names {
                        if !MATH_NAMES.contains(&n.as_str()) {
                            return Err(format!("line {}: `math.{n}` is not one of Viv's builtins; Python's math names Viv has: {}", st.line + 1, MATH_NAMES.join(", ")));
                        }
                        if alias.is_some() {
                            return Err(format!("line {}: `from math import {n} as ...`: the math names are builtins, used by their own names", st.line + 1));
                        }
                    }
                }
            }
        }
        hits.push((st.line, st.nlines));
    }
    if hits.is_empty() {
        return Ok(None);
    }
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    for (l, n) in hits {
        for k in l..(l + n).min(lines.len()) {
            lines[k] = String::new();
        }
    }
    let mut t = lines.join("\n");
    if text.ends_with('\n') {
        t.push('\n');
    }
    if aliases.is_empty() {
        return Ok(Some(t));
    }
    // `math.sqrt(x)` is `sqrt(x)`: the alias and its dot go, through the
    // lexer's tokens (a string or a comment keeps its text)
    let toks = crate::lexer::lex(&t, SourceId(0)).map_err(|ds| ds.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let mut cuts: Vec<(usize, usize)> = Vec::new();
    for k in 0..toks.len().saturating_sub(2) {
        if let (Tok::Ident(a), Tok::Dot, Tok::Ident(n)) = (&toks[k].tok, &toks[k + 1].tok, &toks[k + 2].tok) {
            let after_dot = k > 0 && matches!(toks[k - 1].tok, Tok::Dot);
            if aliases.contains(a) && !after_dot {
                if !MATH_NAMES.contains(&n.as_str()) {
                    return Err(format!("`{a}.{n}`: `{n}` is not one of Viv's builtins; Python's math names Viv has: {}", MATH_NAMES.join(", ")));
                }
                cuts.push((toks[k].span.start as usize, toks[k + 1].span.end as usize));
            }
        }
    }
    for (a, b) in cuts.into_iter().rev() {
        t.replace_range(a..b, "");
    }
    Ok(Some(t))
}

/// Lower a text's import lines (see the module comment). `unit` is the
/// text's canonical module name, empty for the program; `fetch` returns a
/// module's text by canonical name, or `None` when there is no such module.
pub fn lower(unit: &str, text: &str, fetch: &mut dyn FnMut(&str) -> Option<String>) -> Result<Lowered, String> {
    let at = |m: String| if unit.is_empty() { m } else { format!("{unit}: {m}") };
    let has_private = text.lines().any(|l| l.starts_with("private "));
    if !text.contains("import") && !has_private {
        return Ok(Lowered { text: text.to_string(), ..Lowered::default() });
    }
    let stmts = scan(text).map_err(at)?;
    // `import math` and `from math import pi, sqrt`, with no math.viv to find:
    // Python's math module, whose names are Viv's builtins and constants
    if let Some(t) = python_math(unit, text, &stmts, fetch).map_err(at)? {
        return lower(unit, &t, fetch);
    }
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let mut low = Lowered { has_imports: !stmts.is_empty(), ..Lowered::default() };
    // `private` before a top-level def, record, enum or constant
    let mut fence = false;
    for l in lines.iter_mut() {
        if l.trim() == "###" {
            fence = !fence;
            continue;
        }
        if fence {
            continue;
        }
        if let Some(rest) = l.strip_prefix("private ") {
            let tops = line_tops(rest);
            let Some(name) = tops.into_iter().next() else {
                return Err(at("`private` goes before a top-level def, record, enum or constant".to_string()));
            };
            low.privates.push(name);
            *l = format!("{}{rest}", " ".repeat("private ".len()));
        }
    }
    let mut find = |path: &[String], line: usize| -> Result<Option<(String, String)>, String> {
        let mut hits: Vec<(String, String)> = Vec::new();
        for c in candidates(unit, path) {
            if let Some(t) = fetch(&c) {
                hits.push((canonical(&c), t));
            }
        }
        if hits.len() > 1 {
            let first = &hits[0].1;
            if hits.iter().any(|(_, t)| t != first) {
                let names: Vec<String> = hits.iter().map(|(c, _)| format!("{c}.viv")).collect();
                return Err(at(format!("line {}: `{}` finds {}; write the full name (`lib.{}`), or rename one", line + 1, path.join("."), names.join(" and "), path.join("."))));
            }
            if let Some(k) = hits.iter().position(|(c, _)| c.starts_with("lib/")) {
                return Ok(Some(hits.swap_remove(k)));
            }
        }
        Ok(hits.into_iter().next())
    };
    let missing = |path: &[String], line: usize| at(format!("line {}: `{}`: no module by that name (looked for {})", line + 1, path.join("."), candidates(unit, path).iter().map(|c| format!("{}.viv", c.strip_prefix('/').unwrap_or(c))).collect::<Vec<_>>().join(", ")));
    for s in &stmts {
        let mut mods: Vec<String> = Vec::new();
        for item in &s.items {
            match item {
                ImportItem::Module { path, alias } => {
                    let Some((canon, _)) = find(path, s.line)? else { return Err(missing(path, s.line)) };
                    mods.push(canon.clone());
                    match alias {
                        Some(a) => low.bindings.push(Binding::Alias { local: a.clone(), pkg: Pkg { module: Some(canon), kids: BTreeMap::new() }, line: s.line }),
                        None => {
                            // `import a.b.c` binds `a`, with `b.c` under it
                            let mut pkg = Pkg { module: Some(canon), kids: BTreeMap::new() };
                            for seg in path[1..].iter().rev() {
                                let mut up = Pkg::default();
                                up.kids.insert(seg.clone(), pkg);
                                pkg = up;
                            }
                            if path.len() > 1 {
                                // `a` itself, when it is a module too
                                if let Ok(Some((c, _))) = find(&path[..1], s.line) {
                                    pkg.module = Some(c);
                                }
                            }
                            merge_alias(&mut low.bindings, &path[0], pkg, s.line);
                        }
                    }
                }
                ImportItem::From { path, names } => {
                    let here = find(path, s.line)?;
                    let tops = here.as_ref().map(|(_, t)| line_tops(t)).unwrap_or_default();
                    for (n, alias) in names {
                        let local = alias.clone().unwrap_or_else(|| n.clone());
                        let mut sub_path = path.clone();
                        sub_path.push(n.clone());
                        let sub = if here.is_some() && tops.contains(n) { None } else { find(&sub_path, s.line)? };
                        match (sub, &here) {
                            (Some((canon, _)), _) => {
                                mods.push(canon.clone());
                                low.bindings.push(Binding::Alias { local, pkg: Pkg { module: Some(canon), kids: BTreeMap::new() }, line: s.line });
                            }
                            (None, Some((canon, _))) => {
                                if !mods.contains(canon) {
                                    mods.push(canon.clone());
                                }
                                // a contract brings its enum and its selection
                                // (crate::contracts), under the names it gives them
                                let contract = here.as_ref().is_some_and(|(_, t)| t.lines().any(|l| l.starts_with(&format!("contract {n}("))));
                                low.bindings.push(Binding::Name { local, module: canon.clone(), name: n.clone(), line: s.line });
                                if contract {
                                    for c in [crate::contracts::camel(n), format!("{}{n}", crate::contracts::USING)] {
                                        if !names.iter().any(|(m, a)| *m == c || a.as_deref() == Some(c.as_str())) {
                                            low.bindings.push(Binding::Name { local: c.clone(), module: canon.clone(), name: c, line: s.line });
                                        }
                                    }
                                }
                            }
                            (None, None) => return Err(missing(path, s.line)),
                        }
                    }
                }
            }
        }
        // a module named twice on one line is spliced once
        let mut uniq: Vec<String> = Vec::new();
        for m in mods {
            if !uniq.contains(&m) {
                uniq.push(m);
            }
        }
        lines[s.line] = format!("{IMPORT_DIRECTIVE}{}", uniq.join(" "));
        for k in 1..s.nlines {
            lines[s.line + k] = CONTINUED.to_string();
        }
        for m in uniq {
            if !low.modules.contains(&m) {
                low.modules.push(m);
            }
        }
    }
    let mut out = lines.join("\n");
    if text.ends_with('\n') {
        out.push('\n');
    }
    low.text = out;
    Ok(low)
}

fn merge_alias(bindings: &mut Vec<Binding>, local: &str, pkg: Pkg, line: usize) {
    fn merge(into: &mut Pkg, from: Pkg) {
        if from.module.is_some() {
            into.module = from.module;
        }
        for (k, v) in from.kids {
            merge(into.kids.entry(k).or_default(), v);
        }
    }
    for b in bindings.iter_mut() {
        if let Binding::Alias { local: l, pkg: p, .. } = b {
            if l == local {
                merge(p, pkg);
                return;
            }
        }
    }
    bindings.push(Binding::Alias { local: local.to_string(), pkg, line });
}

// ---- reading a unit: what it defines, binds and reads ------------------------

/// What one unit of a spliced program defines and reads.
#[derive(Default, Debug)]
pub struct Facts {
    /// top-level defs, records, enums and constants, and names assigned in
    /// top-level blocks
    pub tops: HashSet<String>,
    /// the defs, records, enums and constants of the top level itself: a
    /// name two units define this way is two names
    pub defined: HashSet<String>,
    /// every name bound anywhere: tops, parameters, locals, loop variables
    pub bound: HashSet<String>,
    /// names read where they are free (not a local or parameter of the
    /// def, test or lambda reading them), with the byte offset of the read
    /// and the name before a `.` in front of it (`nn` in `nn.rmsnorm`)
    pub reads: Vec<(String, u32, Option<String>)>,
    pub parsed: bool,
}

/// Read a unit's code: parsed when it parses, else from its lines.
pub fn facts(code: &str, phrases: &[String]) -> Facts {
    let toks = match crate::lexer::lex(code, SourceId(0)) {
        Ok(t) => t,
        Err(_) => return Facts { tops: line_tops(code), defined: line_tops(code), bound: line_tops(code), ..Facts::default() },
    };
    // the phrases the other units declare (a unit's own are in its code)
    let own: HashSet<&str> = code.lines().filter(|l| l.starts_with("phrase \"")).collect();
    let phrases: Vec<String> = phrases.iter().filter(|p| !own.contains(p.as_str())).cloned().collect();
    let program = match crate::parser::parse_with(toks, SourceId(0), &phrases) {
        Ok(p) => p,
        Err(d) => {
            if std::env::var_os("VIV_IMPORTS_DEBUG").is_some() {
                let at = d.first().and_then(|d| d.primary.as_ref().map(|p| p.span.start)).unwrap_or(0) as usize;
                eprintln!("imports: a unit does not parse: {} at {:?}", d.first().map(|d| d.message.as_str()).unwrap_or(""), code.get(at.saturating_sub(60)..(at + 40).min(code.len())));
            }
            return Facts { tops: line_tops(code), defined: line_tops(code), bound: line_tops(code), ..Facts::default() };
        }
    };
    let mut f = Facts { parsed: true, ..Facts::default() };
    program_facts(&program, code, &mut f);
    f
}

fn hidden(n: &str) -> bool {
    n.starts_with(crate::parser::HIDDEN_PREFIX)
}

fn program_facts(p: &Program, code: &str, f: &mut Facts) {
    let mut items: Vec<&AstItem> = p.items.iter().collect();
    let mut k = 0;
    while k < items.len() {
        if let AstItem::Rung(r) = items[k] {
            items.extend(r.items.iter());
        }
        k += 1;
    }
    // the tops first: a def may read a constant defined after it
    for it in &items {
        match it {
            AstItem::Def(d) => {
                f.tops.insert(d.name.clone());
            }
            AstItem::Record(r) => {
                f.tops.insert(r.name.clone());
            }
            AstItem::Enum(e) => {
                f.tops.insert(e.name.clone());
            }
            AstItem::Stmt(s) => {
                if let Stmt::Assign { name, .. } = s {
                    f.defined.insert(name.clone());
                }
                top_assigns(s, &mut f.tops)
            }
            AstItem::Rung(_) => {}
        }
        if let AstItem::Def(crate::ast::Def { name, .. }) | AstItem::Record(crate::ast::RecordDecl { name, .. }) | AstItem::Enum(crate::ast::EnumDecl { name, .. }) = it {
            f.defined.insert(name.clone());
        }
    }
    f.tops.retain(|n| !hidden(n));
    f.defined.retain(|n| !hidden(n));
    f.bound.extend(f.tops.iter().cloned());
    let mut reads: Vec<(String, u32)> = Vec::new();
    for it in &items {
        match it {
            AstItem::Def(d) => {
                let mut locals: HashSet<String> = d.params.iter().map(|p| p.name.clone()).collect();
                for s in &d.body {
                    assigns(s, &mut locals);
                }
                f.bound.extend(locals.iter().cloned());
                for p in &d.params {
                    if let Some(t) = &p.ty {
                        ty_reads(t, &mut reads);
                    }
                }
                if let Some(t) = &d.ret {
                    ty_reads(t, &mut reads);
                }
                let mut r = Vec::new();
                for s in &d.body {
                    stmt_reads(s, &mut r, &mut f.bound);
                }
                reads.extend(r.into_iter().filter(|(n, _)| !locals.contains(n)));
            }
            AstItem::Record(r) => {
                for fd in &r.fields {
                    ty_reads(&fd.ty, &mut reads);
                    if let Some(e) = &fd.default {
                        let s = Stmt::ExprStmt { expr: e.clone(), span: e.span() };
                        stmt_reads(&s, &mut reads, &mut f.bound);
                    }
                }
            }
            AstItem::Enum(e) => {
                for v in &e.variants {
                    for t in &v.payload {
                        ty_reads(t, &mut reads);
                    }
                }
            }
            AstItem::Stmt(Stmt::Test { body, .. }) => {
                let mut locals = HashSet::new();
                for s in body {
                    assigns(s, &mut locals);
                }
                f.bound.extend(locals.iter().cloned());
                let mut r = Vec::new();
                for s in body {
                    stmt_reads(s, &mut r, &mut f.bound);
                }
                reads.extend(r.into_iter().filter(|(n, _)| !locals.contains(n) || f.tops.contains(n)));
            }
            AstItem::Stmt(s) => {
                let mut locals = HashSet::new();
                for_vars(s, &mut locals);
                f.bound.extend(locals.iter().cloned());
                let mut r = Vec::new();
                stmt_reads(s, &mut r, &mut f.bound);
                reads.extend(r.into_iter().filter(|(n, _)| !locals.contains(n) || f.tops.contains(n)));
            }
            AstItem::Rung(_) => {}
        }
    }
    // a read counts when the name is written where the node says it is: a
    // phrase's body, read at the place the phrase is used, is not
    // and the name before a `.` in front of it, when there is one
    let word_at = |n: &str, at: u32| -> Option<Option<String>> {
        let s = code.get(at as usize..).unwrap_or("");
        let end = s.find('\n').unwrap_or(s.len());
        let line = &s[..end];
        let mut from = 0;
        while let Some(k) = line[from..].find(n) {
            let a = from + k;
            let b = a + n.len();
            let before = line[..a].chars().last();
            let after = line[b..].chars().next();
            let edge = |c: Option<char>| !c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            if edge(before) && edge(after) {
                let pre = &line[..a];
                let qual = pre.strip_suffix('.').map(|p| {
                    let start = p.rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map(|i| i + 1).unwrap_or(0);
                    p[start..].to_string()
                });
                return Some(qual.filter(|q| is_ident(q)));
            }
            from = b;
        }
        None
    };
    for (n, at) in reads {
        if hidden(&n) {
            continue;
        }
        if let Some(q) = word_at(&n, at) {
            f.reads.push((n, at, q));
        }
    }
}

fn top_assigns(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Assign { name, .. } => {
            out.insert(name.clone());
        }
        Stmt::If { then_body, else_body, .. } => then_body.iter().chain(else_body).for_each(|s| top_assigns(s, out)),
        Stmt::While { body, .. } | Stmt::For { body, .. } => body.iter().for_each(|s| top_assigns(s, out)),
        _ => {}
    }
}

fn for_vars(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::For { var, body, .. } => {
            out.insert(var.clone());
            body.iter().for_each(|s| for_vars(s, out));
        }
        Stmt::If { then_body, else_body, .. } => then_body.iter().chain(else_body).for_each(|s| for_vars(s, out)),
        Stmt::While { body, .. } => body.iter().for_each(|s| for_vars(s, out)),
        _ => {}
    }
}

fn assigns(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Assign { name, .. } => {
            out.insert(name.clone());
        }
        Stmt::For { var, body, .. } => {
            out.insert(var.clone());
            body.iter().for_each(|s| assigns(s, out));
        }
        Stmt::If { then_body, else_body, .. } => then_body.iter().chain(else_body).for_each(|s| assigns(s, out)),
        Stmt::While { body, .. } | Stmt::Test { body, .. } => body.iter().for_each(|s| assigns(s, out)),
        Stmt::Lens { lens, .. } => {
            if let crate::ast::Lens::Bind { name, .. } = lens.as_ref() {
                out.insert(name.clone());
            }
        }
        _ => {}
    }
}

fn ty_reads(t: &TypeExpr, out: &mut Vec<(String, u32)>) {
    match t {
        TypeExpr::Named { name, span } | TypeExpr::RecList { name, span } => out.push((name.clone(), span.start)),
        TypeExpr::Opt(inner) => ty_reads(inner, out),
        _ => {}
    }
}

fn stmt_reads(s: &Stmt, out: &mut Vec<(String, u32)>, bound: &mut HashSet<String>) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => expr_reads(value, out, bound),
        Stmt::SetIndex { base, indices, value, .. } => {
            expr_reads(base, out, bound);
            indices.iter().for_each(|i| expr_reads(i, out, bound));
            expr_reads(value, out, bound);
        }
        Stmt::SetField { base, value, .. } => {
            expr_reads(base, out, bound);
            expr_reads(value, out, bound);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            expr_reads(cond, out, bound);
            then_body.iter().chain(else_body).for_each(|s| stmt_reads(s, out, bound));
        }
        Stmt::While { cond, body, .. } => {
            expr_reads(cond, out, bound);
            body.iter().for_each(|s| stmt_reads(s, out, bound));
        }
        Stmt::For { bound: b, body, .. } => {
            expr_reads(b, out, bound);
            body.iter().for_each(|s| stmt_reads(s, out, bound));
        }
        Stmt::Test { body, .. } => body.iter().for_each(|s| stmt_reads(s, out, bound)),
        Stmt::Lens { lens, .. } => {
            if let crate::ast::Lens::Bind { value, .. } = lens.as_ref() {
                expr_reads(value, out, bound);
            }
        }
        _ => {}
    }
}

fn expr_reads(e: &Expr, out: &mut Vec<(String, u32)>, bound: &mut HashSet<String>) {
    fn go(e: &Expr, lam: &mut Vec<String>, out: &mut Vec<(String, u32)>, bound: &mut HashSet<String>) {
        match e {
            Expr::Var { name, span } => {
                if !lam.contains(name) {
                    out.push((name.clone(), span.start));
                }
            }
            Expr::Lambda { params, body, .. } => {
                let before = lam.len();
                lam.extend(params.iter().cloned());
                bound.extend(params.iter().cloned());
                go(body, lam, out, bound);
                lam.truncate(before);
            }
            Expr::Call { callee, args, span } => {
                let head = callee.split('.').next().unwrap_or(callee);
                if !lam.contains(&head.to_string()) {
                    out.push((head.to_string(), span.start));
                }
                args.iter().for_each(|a| go(a, lam, out, bound));
            }
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|a| go(a, lam, out, bound)),
            Expr::Index { base, indices, .. } => {
                go(base, lam, out, bound);
                indices.iter().for_each(|a| go(a, lam, out, bound));
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => go(operand, lam, out, bound),
            Expr::Binary { lhs, rhs, .. } => {
                go(lhs, lam, out, bound);
                go(rhs, lam, out, bound);
            }
            Expr::RecordLit { type_name, type_span, fields, .. } => {
                if !type_name.is_empty() {
                    out.push((type_name.clone(), type_span.start));
                }
                fields.iter().for_each(|(_, _, x)| go(x, lam, out, bound));
            }
            Expr::Field { base, .. } => go(base, lam, out, bound),
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }
    go(e, &mut Vec::new(), out, bound);
}

// ---- resolution and renaming over a spliced program ---------------------------

/// One unit of a spliced program, as `resolve` reads it.
pub struct UnitIn<'a> {
    /// canonical module name; empty for the program
    pub name: &'a str,
    pub lowered: Option<&'a Lowered>,
    /// the `#! use` names its text has (the old splice)
    pub uses: Vec<String>,
}

/// Resolve and rename a spliced program. `text` is the splice's output,
/// `origin[i]` the unit and unit line of its line `i` (`None` for a marker),
/// `units` each unit in the splice's order. Returns the program's text with
/// every line renamed, or the first error.
pub fn resolve(text: &str, origin: &[Option<(usize, usize)>], units: &[UnitIn]) -> Result<String, String> {
    let lines: Vec<&str> = text.lines().collect();
    let n = units.len();
    // each unit's code: its lines in order, and where each sits in `text`
    let code = unit_codes(text, origin, n);
    let legacy = units.iter().any(|u| !u.uses.is_empty());
    let phrases: Vec<String> = lines.iter().filter(|l| l.starts_with("phrase \"")).map(|l| l.to_string()).collect();
    let facts: Vec<Facts> = code.iter().map(|(c, _)| facts(c, &phrases)).collect();
    let index: HashMap<&str, usize> = units.iter().enumerate().map(|(k, u)| (u.name, k)).collect();
    let empty = Lowered::default();
    let low = |k: usize| units[k].lowered.unwrap_or(&empty);
    let shown = |k: usize| if units[k].name.is_empty() { "the program".to_string() } else { dotted_of(units[k].name) };
    let line_of = |k: usize, at: u32| -> usize {
        let c = &code[k].0;
        let row = c[..(at as usize).min(c.len())].matches('\n').count();
        code[k].1.get(row).and_then(|&i| origin[i]).map(|(_, l)| l + 1).unwrap_or(0)
    };
    // which units define each top-level name
    let mut definers: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut defined_by: HashMap<&str, Vec<usize>> = HashMap::new();
    for (k, f) in facts.iter().enumerate() {
        for t in &f.tops {
            definers.entry(t.as_str()).or_default().push(k);
        }
        for t in &f.defined {
            defined_by.entry(t.as_str()).or_default().push(k);
        }
    }
    let unit_of = |canon: &str| index.get(canon).copied();
    // checks on what each unit's imports bind
    for k in 0..n {
        let l = low(k);
        let mut locals: HashMap<&str, usize> = HashMap::new();
        for b in &l.bindings {
            let (local, line) = match b {
                Binding::Alias { local, line, .. } | Binding::Name { local, line, .. } => (local.as_str(), *line),
            };
            if let Some(prev) = locals.insert(local, line) {
                if prev != line || matches!(b, Binding::Name { .. }) {
                    return Err(format!("{}, line {}: `{local}` is imported twice (lines {} and {})", shown(k), line + 1, prev + 1, line + 1));
                }
            }
            if facts[k].tops.contains(local) {
                return Err(format!("{}, line {}: `{local}` is imported here and defined in this file; import it under another name (`as`) or rename the definition", shown(k), line + 1));
            }
            match b {
                Binding::Name { module, name, line, .. } => {
                    let Some(m) = unit_of(module) else { continue };
                    if !facts[m].tops.contains(name) {
                        return Err(format!("{}, line {}: {} has no `{name}`", shown(k), line + 1, dotted_of(module)));
                    }
                    if low(m).privates.contains(name) {
                        return Err(format!("{}, line {}: `{name}` is private to {}", shown(k), line + 1, dotted_of(module)));
                    }
                }
                Binding::Alias { local, line, .. } => {
                    if facts[k].bound.contains(local) {
                        return Err(format!("{}, line {}: `{local}` names a module here and a variable in this file; import it under another name (`import ... as`)", shown(k), line + 1));
                    }
                }
            }
        }
    }
    // the names to rename across the whole program: a top-level name two
    // units define (not in a program that still has `#! use`, whose modules
    // share one namespace), and a name a qualified read or an alias would
    // otherwise capture
    let mut mangled: HashSet<(usize, String)> = HashSet::new();
    if !legacy {
        for (name, ks) in &defined_by {
            if ks.len() > 1 {
                for &k in ks {
                    if k != 0 {
                        mangled.insert((k, name.to_string()));
                    }
                }
            }
        }
    }
    // the per-unit renaming, and the tokens to rewrite
    let mut out_lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    let internal = |m: usize, name: &str, mangled: &HashSet<(usize, String)>| -> String {
        if mangled.contains(&(m, name.to_string())) {
            mangle(units[m].name, name)
        } else {
            name.to_string()
        }
    };
    // capture: a target written into a unit that binds that name itself
    for k in 0..n {
        for b in &low(k).bindings {
            if let Binding::Name { local, module, name, .. } = b {
                if local != name && facts[k].bound.contains(name) {
                    if let Some(m) = unit_of(module) {
                        mangled.insert((m, name.clone()));
                    }
                }
            }
        }
        // qualified reads: every name of an aliased module the unit also binds
        let mut pkgs: Vec<&Pkg> = Vec::new();
        for b in &low(k).bindings {
            if let Binding::Alias { pkg, .. } = b {
                pkgs.push(pkg);
            }
        }
        while let Some(p) = pkgs.pop() {
            if let Some(m) = p.module.as_deref().and_then(unit_of) {
                for t in &facts[m].tops {
                    if facts[k].bound.contains(t) && m != k {
                        mangled.insert((m, t.clone()));
                    }
                }
            }
            pkgs.extend(p.kids.values());
        }
    }
    // visibility: a free read of a name defined by a module this unit reaches
    // only through another, or private to another module
    if !legacy {
        for k in 0..n {
            if !facts[k].parsed || !low(k).has_imports {
                continue;
            }
            let mut direct: HashSet<usize> = HashSet::new();
            let mut names: HashSet<&str> = HashSet::new();
            let mut stack: Vec<&Pkg> = Vec::new();
            for b in &low(k).bindings {
                match b {
                    Binding::Name { local, .. } => {
                        names.insert(local);
                    }
                    Binding::Alias { local, pkg, .. } => {
                        names.insert(local);
                        stack.push(pkg);
                    }
                }
            }
            while let Some(p) = stack.pop() {
                if let Some(m) = p.module.as_deref().and_then(unit_of) {
                    direct.insert(m);
                }
                stack.extend(p.kids.values());
            }
            // the import closure
            let mut closure: Vec<usize> = low(k).modules.iter().filter_map(|m| unit_of(m)).collect();
            let mut j = 0;
            while j < closure.len() {
                let m = closure[j];
                for nm in low(m).modules.iter().chain(units[m].uses.iter()) {
                    if let Some(x) = unit_of(nm) {
                        if !closure.contains(&x) && x != k {
                            closure.push(x);
                        }
                    }
                }
                j += 1;
            }
            let aliases: HashSet<&str> = low(k).bindings.iter().filter_map(|b| if let Binding::Alias { local, .. } = b { Some(local.as_str()) } else { None }).collect();
            for (r, at, qual) in &facts[k].reads {
                if facts[k].tops.contains(r) || names.contains(r.as_str()) {
                    continue;
                }
                // `nn.rmsnorm`: the rewrite below checks a qualified name
                if qual.as_deref().is_some_and(|q| aliases.contains(q)) {
                    continue;
                }
                let Some(ds) = definers.get(r.as_str()) else { continue };
                for &d in ds {
                    if d == k {
                        continue;
                    }
                    if low(d).privates.contains(r) && closure.contains(&d) {
                        return Err(format!("{}, line {}: `{r}` is private to {}", shown(k), line_of(k, *at), dotted_of(units[d].name)));
                    }
                    if closure.contains(&d) {
                        let how = if direct.contains(&d) { "which this file imports by its module name" } else { "which this file imports only through another module" };
                        return Err(format!(
                            "{}, line {}: `{r}` is defined in {}, {how}; import it: `from {} import {r}`",
                            shown(k),
                            line_of(k, *at),
                            dotted_of(units[d].name),
                            dotted_of(units[d].name)
                        ));
                    }
                }
            }
        }
    }
    // the rewrite, unit by unit
    for k in 0..n {
        let l = low(k);
        let mut map: HashMap<String, String> = HashMap::new();
        for t in &facts[k].tops {
            if mangled.contains(&(k, t.clone())) {
                map.insert(t.clone(), mangle(units[k].name, t));
            }
        }
        let mut aliases: HashMap<String, Pkg> = HashMap::new();
        for b in &l.bindings {
            match b {
                Binding::Name { local, module, name, .. } => {
                    let target = unit_of(module).map(|m| internal(m, name, &mangled)).unwrap_or_else(|| name.clone());
                    if &target != local {
                        map.insert(local.clone(), target);
                    }
                }
                Binding::Alias { local, pkg, .. } => {
                    aliases.insert(local.clone(), pkg.clone());
                }
            }
        }
        if map.is_empty() && aliases.is_empty() {
            continue;
        }
        let code_k = &code[k].0;
        let edits = rename_edits(code_k, &map, &aliases, &|canon: &str, name: &str| -> Result<String, String> {
            let Some(m) = unit_of(canon) else { return Ok(name.to_string()) };
            if !facts[m].tops.contains(name) {
                return Err(format!("{} has no `{name}`", dotted_of(canon)));
            }
            if low(m).privates.contains(&name.to_string()) {
                return Err(format!("`{name}` is private to {}", dotted_of(canon)));
            }
            Ok(internal(m, name, &mangled))
        })
        .map_err(|(e, at)| format!("{}, line {}: {e}", shown(k), line_of(k, at)))?;
        if edits.is_empty() {
            continue;
        }
        let mut c = code_k.clone();
        for (a, b, s) in edits.into_iter().rev() {
            c.replace_range(a..b, &s);
        }
        for (row, l) in c.lines().enumerate() {
            if let Some(&i) = code[k].1.get(row) {
                out_lines[i] = l.to_string();
            }
        }
    }
    let mut out = out_lines.join("\n");
    if text.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

/// Each unit's code in a spliced program: its lines in order, and the index
/// in the program's text of each.
pub fn unit_codes(text: &str, origin: &[Option<(usize, usize)>], n: usize) -> Vec<(String, Vec<usize>)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut code: Vec<(String, Vec<usize>)> = vec![(String::new(), Vec::new()); n];
    for (i, o) in origin.iter().enumerate() {
        if let Some((u, _)) = o {
            if let Some(l) = lines.get(i) {
                code[*u].0.push_str(l);
                code[*u].0.push('\n');
                code[*u].1.push(i);
            }
        }
    }
    code
}

/// What `migrate` did to one file.
#[derive(Debug, Default)]
pub struct Migration {
    pub text: String,
    /// `#! use` lines replaced
    pub uses: usize,
    /// import lines written
    pub imports: usize,
    /// names read that more than one module defines: (name, the modules), the
    /// last of which the import names
    pub ambiguous: Vec<(String, Vec<String>)>,
}

/// `from m import a, b`, the names sorted (case aside); past 100 columns, a
/// parenthesized list four spaces in, a line of names at most 96 wide.
pub fn from_line(dotted: &str, names: &[String]) -> Vec<String> {
    let mut names = names.to_vec();
    names.sort_by_key(|a| (a.to_lowercase(), a.clone()));
    let one = format!("from {dotted} import {}", names.join(", "));
    if one.len() <= 100 {
        return vec![one];
    }
    let mut out = vec![format!("from {dotted} import (")];
    let mut cur = String::from("   ");
    for nm in &names {
        if cur.len() + nm.len() + 2 > 96 {
            out.push(cur.trim_end().to_string());
            cur = String::from("   ");
        }
        cur.push(' ');
        cur.push_str(nm);
        cur.push(',');
    }
    out.push(cur.trim_end().to_string());
    out.push(")".to_string());
    out
}

/// Rewrite a file's `#! use` lines as imports: each
/// `#! use m` becomes `from m import <the names the file reads from m>`
/// (`import m` when it reads none), followed by a `from t import ...` for
/// each module `t` that `m` brought in and the file reads from. Nothing else
/// changes. `exp` is the file's expansion by the old splice, with the file
/// as the program.
pub fn migrate(src: &str, exp: &crate::Expansion) -> Result<Migration, String> {
    let n = exp.units.len();
    let lines: Vec<Option<(usize, usize)>> = if exp.lines.is_empty() { (0..exp.text.lines().count()).map(|k| Some((0, k))).collect() } else { exp.lines.clone() };
    let code = unit_codes(&exp.text, &lines, n);
    let phrases: Vec<String> = exp.text.lines().filter(|l| l.starts_with("phrase \"")).map(|l| l.to_string()).collect();
    let facts: Vec<Facts> = code.iter().map(|(c, _)| facts(c, &phrases)).collect();
    if !facts[0].parsed {
        return Err("the file does not parse on its own".to_string());
    }
    let index: HashMap<&str, usize> = exp.units.iter().enumerate().map(|(k, u)| (u.name.as_str(), k)).collect();
    // the names the file reads, by the unit that defines them
    let mut wanted: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut ambiguous = Vec::new();
    // what the file's own import statements bind already (a file half migrated)
    let mut bound_here: HashSet<String> = HashSet::new();
    for st in scan(src).unwrap_or_default() {
        for item in &st.items {
            match item {
                ImportItem::Module { path, alias } => {
                    bound_here.insert(alias.clone().unwrap_or_else(|| path[0].clone()));
                }
                ImportItem::From { names, .. } => bound_here.extend(names.iter().map(|(n, a)| a.clone().unwrap_or_else(|| n.clone()))),
            }
        }
    }
    for (r, _, _) in &facts[0].reads {
        if facts[0].tops.contains(r) || bound_here.contains(r) {
            continue;
        }
        let ds: Vec<usize> = (1..n).filter(|&k| facts[k].tops.contains(r)).collect();
        let Some(&d) = ds.last() else { continue };
        if ds.len() > 1 && !ambiguous.iter().any(|(x, _): &(String, Vec<String>)| x == r) {
            ambiguous.push((r.clone(), ds.iter().map(|&k| exp.units[k].name.clone()).collect()));
        }
        let v = wanted.entry(d).or_default();
        if !v.contains(r) {
            v.push(r.clone());
        }
    }
    // the file's `#! use` lines, and the units each brought in
    let src_lines: Vec<&str> = src.lines().collect();
    let mut direct: Vec<(usize, usize)> = Vec::new(); // (line, unit)
    for (i, l) in src_lines.iter().enumerate() {
        if let Some(m) = l.trim_end().strip_prefix("#! use ") {
            let m = m.trim();
            match index.get(m) {
                Some(&u) => direct.push((i, u)),
                None => direct.push((i, usize::MAX)),
            }
        }
    }
    let owner = |t: usize| -> Option<usize> { direct.iter().filter(|(_, u)| *u != usize::MAX && *u <= t).max_by_key(|(_, u)| *u).map(|(i, _)| *i) };
    let mut extra: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &t in wanted.keys() {
        if direct.iter().any(|(_, u)| *u == t) {
            continue;
        }
        match owner(t) {
            Some(i) => extra.entry(i).or_default().push(t),
            None => return Err(format!("`{}` comes from {}, which no `#! use` line of the file brings in", wanted[&t][0], exp.units[t].name)),
        }
    }
    // an import that binds no name the file reads: `import a.b` binds `a`,
    // so when the file has an `a`, `from a import b`, else `import a.b as b_module`
    let taken = |n: &str| facts[0].bound.contains(n) || wanted.values().flatten().any(|w| w == n);
    let render = |u: usize| -> Vec<String> {
        let dotted = dotted_of(&exp.units[u].name);
        match wanted.get(&u) {
            Some(names) => from_line(&dotted, names),
            None => {
                let segs: Vec<&str> = dotted.split('.').collect();
                if !taken(segs[0]) {
                    return vec![format!("import {dotted}")];
                }
                let last = segs[segs.len() - 1];
                if segs.len() > 1 && !taken(last) {
                    return vec![format!("from {} import {last}", segs[..segs.len() - 1].join("."))];
                }
                vec![format!("import {dotted} as {last}_module")]
            }
        }
    };
    let mut m = Migration { ambiguous, ..Migration::default() };
    let mut out: Vec<String> = Vec::new();
    let mut done: HashSet<usize> = HashSet::new();
    for (i, l) in src_lines.iter().enumerate() {
        let Some(&(_, u)) = direct.iter().find(|(j, _)| *j == i) else {
            out.push(l.to_string());
            continue;
        };
        m.uses += 1;
        if u == usize::MAX {
            // a module the splice left out (an `#! if-none-used` block that does not hold)
            let name = l.trim_end().trim_start_matches("#! use ").trim();
            out.push(format!("import {}", dotted_of(name)));
            m.imports += 1;
            continue;
        }
        if done.insert(u) {
            for r in render(u) {
                out.push(r);
            }
            m.imports += 1;
        }
        for &t in extra.get(&i).map(|v| v.as_slice()).unwrap_or(&[]) {
            if done.insert(t) {
                out.extend(render(t));
                m.imports += 1;
            }
        }
    }
    let mut text = out.join("\n");
    if src.ends_with('\n') {
        text.push('\n');
    }
    m.text = text;
    Ok(m)
}

/// The edits renaming a unit's code: `(start, end, replacement)`, in order.
#[allow(clippy::type_complexity)]
fn rename_edits(
    code: &str,
    map: &HashMap<String, String>,
    aliases: &HashMap<String, Pkg>,
    qualified: &dyn Fn(&str, &str) -> Result<String, String>,
) -> Result<Vec<(usize, usize, String)>, (String, u32)> {
    let toks = crate::lexer::lex(code, SourceId(0)).map_err(|d| (d.first().map(|d| d.message.clone()).unwrap_or_default(), 0))?;
    let mut edits = Vec::new();
    edit_stream(&toks, map, aliases, qualified, &mut edits)?;
    edits.sort_by_key(|e| e.0);
    Ok(edits)
}

#[derive(Clone, Copy, PartialEq)]
enum Br {
    Call,
    Header,
    Other,
    Brace,
}

#[allow(clippy::type_complexity)]
fn edit_stream(
    toks: &[Token],
    map: &HashMap<String, String>,
    aliases: &HashMap<String, Pkg>,
    qualified: &dyn Fn(&str, &str) -> Result<String, String>,
    edits: &mut Vec<(usize, usize, String)>,
) -> Result<(), (String, u32)> {
    let mut stack: Vec<Br> = Vec::new();
    let tok = |j: usize| toks.get(j).map(|t| &t.tok);
    let mut k = 0;
    while k < toks.len() {
        let t = &toks[k];
        match &t.tok {
            Tok::LParen => {
                let kind = match (k.checked_sub(2).and_then(tok), k.checked_sub(1).and_then(tok)) {
                    (Some(Tok::Def), Some(Tok::Ident(_))) => Br::Header,
                    (_, Some(Tok::Ident(_) | Tok::RParen | Tok::RBracket)) => Br::Call,
                    _ => Br::Other,
                };
                stack.push(kind);
            }
            Tok::LBracket => stack.push(Br::Other),
            Tok::LBrace => stack.push(Br::Brace),
            Tok::RParen | Tok::RBracket | Tok::RBrace => {
                stack.pop();
            }
            Tok::FStr(parts) => {
                for p in parts {
                    if let FPart::Field { toks: ft, eq, spec_fields, .. } = p {
                        // `{x=}` prints its own text, which a rename would change
                        if eq.is_none() {
                            edit_stream(ft, map, aliases, qualified, edits)?;
                        }
                        for sf in spec_fields {
                            edit_stream(sf, map, aliases, qualified, edits)?;
                        }
                    }
                }
            }
            Tok::Ident(name) => {
                let after_dot = k > 0 && matches!(tok(k - 1), Some(Tok::Dot)) && !(k > 1 && matches!(tok(k - 2), Some(Tok::Dot)));
                if after_dot {
                    k += 1;
                    continue;
                }
                // a qualified name: alias.name, alias.sub.name
                if let Some(pkg) = aliases.get(name) {
                    let mut p = pkg;
                    let mut j = k;
                    loop {
                        let (Some(Tok::Dot), Some(Tok::Ident(next))) = (tok(j + 1), tok(j + 2)) else {
                            return Err((format!("`{name}` is a module: write `{name}.<name>`"), t.span.start));
                        };
                        j += 2;
                        if let Some(kid) = p.kids.get(next) {
                            p = kid;
                            continue;
                        }
                        let Some(canon) = &p.module else {
                            return Err((format!("no module `{next}` under `{name}`"), t.span.start));
                        };
                        let target = qualified(canon, next).map_err(|e| (e, t.span.start))?;
                        edits.push((t.span.start as usize, toks[j].span.end as usize, target));
                        break;
                    }
                    k = j + 1;
                    continue;
                }
                if let Some(target) = map.get(name) {
                    let inner = stack.last().copied();
                    let next = tok(k + 1);
                    let kwarg = matches!(next, Some(Tok::Eq)) && inner == Some(Br::Call);
                    let field = matches!(next, Some(Tok::Colon)) && inner == Some(Br::Brace) && !typed_local(toks, k + 1);
                    if !kwarg && !field {
                        edits.push((t.span.start as usize, t.span.end as usize, target.clone()));
                    }
                }
            }
            _ => {}
        }
        k += 1;
    }
    Ok(())
}

/// Whether the `:` at `toks[c]` is a typed assignment's (`x: Int = 3`) and
/// not a record field's (`R { x: 3 }`): an `=` before the statement ends.
fn typed_local(toks: &[Token], c: usize) -> bool {
    let mut depth = 0i32;
    for t in &toks[c + 1..] {
        match t.tok {
            Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
            Tok::RParen | Tok::RBracket => depth -= 1,
            Tok::RBrace if depth == 0 => return false,
            Tok::RBrace => depth -= 1,
            Tok::Comma if depth == 0 => return false,
            Tok::Newline | Tok::Eof => return false,
            Tok::Eq if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// The `from m import ...` lines a generated program needs: for each
/// module of `modules` (canonical name and text), in order, the names of it
/// `body` reads that no earlier module gave; a module that gives none is
/// imported with `import m` when `always` names it.
pub fn import_lines(body: &str, modules: &[(&str, &str)], always: &[&str]) -> String {
    let idents: HashSet<String> = crate::lexer::lex(body, SourceId(0))
        .map(|ts| ts.into_iter().filter_map(|t| if let Tok::Ident(n) = t.tok { Some(n) } else { None }).collect())
        .unwrap_or_default();
    let mut given: HashSet<String> = HashSet::new();
    let mut out = String::new();
    for (canon, text) in modules {
        let names: Vec<String> = line_tops(text).into_iter().filter(|n| idents.contains(n) && !given.contains(n)).collect();
        given.extend(names.iter().cloned());
        if !names.is_empty() {
            for l in from_line(&dotted_of(canon), &names) {
                out.push_str(&l);
                out.push('\n');
            }
        } else if always.contains(canon) {
            out.push_str(&format!("import {}\n", dotted_of(canon)));
        }
    }
    out
}

/// For tools that read a text's modules without resolving them: each import
/// line's modules as the candidate names `candidates` gives, with the line
/// (1-based). `#! use` lines are included as written.
pub fn module_refs(unit: &str, text: &str) -> Vec<(Vec<String>, usize)> {
    let mut out: Vec<(Vec<String>, usize)> = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if let Some(n) = l.trim_end().strip_prefix("#! use ") {
            let n = n.trim();
            if !n.is_empty() {
                out.push((vec![n.to_string()], i + 1));
            }
        }
        // a lowered import (`lower`) names its modules canonically
        if let Some(ns) = l.trim_end().strip_prefix(IMPORT_DIRECTIVE) {
            for n in ns.split_whitespace() {
                out.push((vec![n.to_string()], i + 1));
            }
        }
    }
    if let Ok(stmts) = scan(text) {
        for s in stmts {
            for item in &s.items {
                match item {
                    ImportItem::Module { path, .. } => out.push((candidates(unit, path), s.line + 1)),
                    ImportItem::From { path, names } => {
                        out.push((candidates(unit, path), s.line + 1));
                        for (n, _) in names {
                            let mut p = path.clone();
                            p.push(n.clone());
                            out.push((candidates(unit, &p), s.line + 1));
                        }
                    }
                }
            }
        }
    }
    out.sort_by_key(|(_, l)| *l);
    out
}

/// `module_refs`, each resolved against a set of known module names: the
/// first candidate that is one. A `from a import b` whose `b` is a name, not
/// a submodule, finds nothing for `a.b` and is left out.
pub fn known_refs(unit: &str, text: &str, known: &dyn Fn(&str) -> bool) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    for (cands, line) in module_refs(unit, text) {
        if let Some(c) = cands.iter().map(|c| canonical(c)).find(|c| known(c)) {
            if !out.iter().any(|(m, _)| m == &c) {
                out.push((c, line));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_the_three_forms() {
        let s = scan("import nn\nimport hf.scf as scf, cas\nfrom nn import (\n    a,\n    b as c,\n)\nx = 1\n").unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[1].items.len(), 2);
        assert_eq!(s[2].nlines, 4);
        assert_eq!(s[2].items, vec![ImportItem::From { path: vec!["nn".into()], names: vec![("a".into(), None), ("b".into(), Some("c".into()))] }]);
    }

    #[test]
    fn refuses_star_relative_indented_and_digits() {
        assert!(scan("from nn import *\n").unwrap_err().contains("import *"));
        assert!(scan("def f() {\n    import nn\n}\n").unwrap_err().contains("top level"));
        assert!(scan("import basis.631g\n").unwrap_err().contains("digit"));
        assert!(scan("###\nfrom one table we read\nfrom x import y\n###\n").unwrap().is_empty());
    }

    #[test]
    fn candidates_in_order() {
        assert_eq!(candidates("", &["hf".into(), "scf".into()]), vec!["./hf/scf", "/lib/hf/scf", "/hf/scf"]);
        assert_eq!(candidates("lib/hf/scf", &["mole".into()]), vec!["/lib/hf/mole", "/lib/mole", "/mole"]);
        assert_eq!(candidates("./seq", &["x".into()]), vec!["./x", "/lib/x", "/x"]);
        assert_eq!(candidates("", &["lib".into(), "nn".into()]), vec!["/lib/nn"]);
        assert_eq!(canonical("/lib/nn"), "lib/nn");
        assert_eq!(dotted_of("./seq"), "seq");
        assert_eq!(dotted_of("lib/hf/scf"), "hf.scf");
    }
}
