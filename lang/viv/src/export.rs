// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv export`: a whole Viv program as one C file and a Python wrapper.
//!
//! A manifest (`<name>.export.toml`) names the program's files, its decks (a
//! choice of compiled-in inputs, such as the material of dft/), its
//! parameters (top-level globals a deck assigns) and its results (globals
//! the program computes). From it `viv export` writes:
//!
//! - `<name>.c`: the program, its runtime and an exported entry, built with
//!   `-DVIV_EMBED` (lang/viv/src/embed.c): an error returns to the caller,
//!   output goes to a callback, the program's state is restored before
//!   each run;
//! - `<name>.h`: the entry's declarations for C callers;
//! - `<name>.py`: a ctypes wrapper with a typed `run(...)`;
//! - `README.md` and `test_<name>.py`.
//!
//! This module holds what does not touch the file system or the compiler's
//! loader: the manifest, the program text it assembles, the checks that
//! refuse a parameter or result, and the generated text. main.rs drives it.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::ast::{Expr, Item, Program, Stmt};
use crate::lower::{Elem, Ty};

// ------------------------------------------------------------------ TOML

/// A value of the manifest: the part of TOML a manifest needs.
#[derive(Clone, Debug, PartialEq)]
pub enum Toml {
    Str(String),
    Num(f64),
    Bool(bool),
    Arr(Vec<Toml>),
    Table(Vec<(String, Toml)>),
}

impl Toml {
    fn str(&self) -> Option<&str> {
        match self {
            Toml::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// The sections of a TOML text in order, each with its keys in order and the
/// line each key is on. The top level is the section "". Keys are bare or
/// quoted; values are strings, numbers, booleans, arrays (on one or more
/// lines) and inline tables.
pub fn parse_toml(text: &str) -> Result<Vec<(String, Vec<(String, Toml, usize)>)>, String> {
    let mut out: Vec<(String, Vec<(String, Toml, usize)>)> = vec![(String::new(), Vec::new())];
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let lineno = i + 1;
        let line = strip_comment(lines[i]).trim().to_string();
        i += 1;
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            let Some(name) = rest.strip_suffix(']') else { return Err(format!("line {lineno}: a section header ends with `]`")) };
            out.push((name.trim().to_string(), Vec::new()));
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { return Err(format!("line {lineno}: expected `key = value`")) };
        let key = key.trim().trim_matches('"').to_string();
        let mut value = value.trim().to_string();
        // a multi-line literal string, '''...''': the lines as they are (no
        // escapes, no comments), less a line break right after the opening
        if let Some(first) = value.strip_prefix("'''") {
            let mut body = String::new();
            if let Some(end) = first.find("'''") {
                if !first[end + 3..].trim().is_empty() {
                    return Err(format!("line {lineno}: unexpected text after the value"));
                }
                body.push_str(&first[..end]);
            } else {
                // the raw line, not the comment-stripped one: a `#` is text here
                let raw = lines[i - 1].split_once("'''").map(|(_, r)| r).unwrap_or("");
                if !raw.trim().is_empty() {
                    body.push_str(raw);
                    body.push('\n');
                }
                loop {
                    let Some(l) = lines.get(i) else { return Err(format!("line {lineno}: a ''' string is not closed")) };
                    i += 1;
                    if let Some(k) = l.find("'''") {
                        if !strip_comment(&l[k + 3..]).trim().is_empty() {
                            return Err(format!("line {}: unexpected text after the value", i));
                        }
                        body.push_str(&l[..k]);
                        break;
                    }
                    body.push_str(l);
                    body.push('\n');
                }
            }
            let section = out.last_mut().expect("the top level");
            if section.1.iter().any(|(k, _, _)| *k == key) {
                return Err(format!("line {lineno}: `{key}` is given twice"));
            }
            section.1.push((key, Toml::Str(body), lineno));
            continue;
        }
        // an array may continue on the lines below, until its brackets close
        while depth(&value) > 0 && i < lines.len() {
            value.push(' ');
            value.push_str(strip_comment(lines[i]).trim());
            i += 1;
        }
        let mut p = Parser { s: value.as_bytes(), at: 0 };
        let v = p.value().map_err(|e| format!("line {lineno}: {e}"))?;
        p.ws();
        if p.at != p.s.len() {
            return Err(format!("line {lineno}: unexpected text after the value"));
        }
        let section = out.last_mut().expect("the top level");
        if section.1.iter().any(|(k, _, _)| *k == key) {
            return Err(format!("line {lineno}: `{key}` is given twice"));
        }
        section.1.push((key, v, lineno));
    }
    Ok(out)
}

/// A line without its `#` comment (a `#` inside a string stays).
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    let mut esc = false;
    for (k, c) in line.char_indices() {
        match c {
            '\\' if in_str => esc = !esc,
            '"' if !esc => in_str = !in_str,
            '#' if !in_str => return &line[..k],
            _ => esc = false,
        }
        if c != '\\' {
            esc = false;
        }
    }
    line
}

/// How many brackets and braces are open at the end of `s`, outside strings.
fn depth(s: &str) -> i32 {
    let (mut d, mut in_str, mut esc) = (0, false, false);
    for c in s.chars() {
        match c {
            '\\' if in_str => {
                esc = !esc;
                continue;
            }
            '"' if !esc => in_str = !in_str,
            '[' | '{' if !in_str => d += 1,
            ']' | '}' if !in_str => d -= 1,
            _ => {}
        }
        esc = false;
    }
    d
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.at < self.s.len() && (self.s[self.at] as char).is_whitespace() {
            self.at += 1;
        }
    }
    fn value(&mut self) -> Result<Toml, String> {
        self.ws();
        let Some(&c) = self.s.get(self.at) else { return Err("a value is missing".into()) };
        match c {
            b'"' => {
                self.at += 1;
                let mut out = String::new();
                loop {
                    let Some(&c) = self.s.get(self.at) else { return Err("a string is not closed".into()) };
                    self.at += 1;
                    match c {
                        b'"' => break,
                        b'\\' => {
                            let e = self.s.get(self.at).copied().ok_or("a string is not closed")?;
                            self.at += 1;
                            out.push(match e {
                                b'n' => '\n',
                                b't' => '\t',
                                b'"' => '"',
                                b'\\' => '\\',
                                _ => return Err(format!("unknown escape `\\{}`", e as char)),
                            });
                        }
                        _ => {
                            // the bytes of one UTF-8 character
                            let start = self.at - 1;
                            let mut end = self.at;
                            while end < self.s.len() && (self.s[end] & 0xC0) == 0x80 {
                                end += 1;
                            }
                            out.push_str(std::str::from_utf8(&self.s[start..end]).map_err(|_| "a string is not UTF-8")?);
                            self.at = end;
                        }
                    }
                }
                Ok(Toml::Str(out))
            }
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.s.get(self.at) == Some(&b']') {
                        self.at += 1;
                        return Ok(Toml::Arr(items));
                    }
                    items.push(self.value()?);
                    self.ws();
                    match self.s.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {}
                        _ => return Err("expected `,` or `]` in an array".into()),
                    }
                }
            }
            b'{' => {
                self.at += 1;
                let mut items: Vec<(String, Toml)> = Vec::new();
                loop {
                    self.ws();
                    if self.s.get(self.at) == Some(&b'}') {
                        self.at += 1;
                        return Ok(Toml::Table(items));
                    }
                    let start = self.at;
                    let key = if self.s.get(self.at) == Some(&b'"') {
                        // a quoted key may hold spaces (`"every dense" = ...`)
                        self.at += 1;
                        while self.at < self.s.len() && self.s[self.at] != b'"' {
                            self.at += 1;
                        }
                        let k = String::from_utf8_lossy(&self.s[start + 1..self.at.min(self.s.len())]).to_string();
                        self.at += 1;
                        k
                    } else {
                        while self.at < self.s.len() && !matches!(self.s[self.at], b'=' | b' ') {
                            self.at += 1;
                        }
                        String::from_utf8_lossy(&self.s[start..self.at]).to_string()
                    };
                    self.ws();
                    if self.s.get(self.at) != Some(&b'=') {
                        return Err(format!("expected `=` after `{key}` in a table"));
                    }
                    self.at += 1;
                    let v = self.value()?;
                    if items.iter().any(|(k, _)| *k == key) {
                        return Err(format!("`{key}` is given twice in a table"));
                    }
                    items.push((key, v));
                    self.ws();
                    match self.s.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {}
                        _ => return Err("expected `,` or `}` in a table".into()),
                    }
                }
            }
            _ => {
                let start = self.at;
                while self.at < self.s.len() && !matches!(self.s[self.at], b',' | b']' | b'}' | b' ') {
                    self.at += 1;
                }
                let word = std::str::from_utf8(&self.s[start..self.at]).unwrap_or("");
                match word {
                    "true" => Ok(Toml::Bool(true)),
                    "false" => Ok(Toml::Bool(false)),
                    _ => word.replace('_', "").parse::<f64>().map(Toml::Num).map_err(|_| format!("`{word}` is not a value")),
                }
            }
        }
    }
}

// -------------------------------------------------------------- manifest

/// What a parameter or a result is, across the boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// `double`, a Python float
    Num,
    /// `int64_t`, a Python int; a deck's `Num` global takes it converted
    Int,
    /// `int` 0 or 1, a Python bool
    Bool,
    /// `const char*`, UTF-8, a Python str
    Str,
    /// one of the names, an `int` in C, a `str` in Python; the global is a
    /// Str (the name), a Num or an Int (its index)
    Enum(Vec<String>),
    /// numbers of any rank (or `rank`), a float64 ndarray
    Arr,
    /// a complex number: `double _Complex` in C, a Python complex. A kernel
    /// collection's arguments and results only (section 11)
    Cplx,
    /// Ints, rank 1, an int64 ndarray
    IntArr,
    /// an enum the program declares (`enum BasisSet { Sto3g ... }`), its
    /// global holding a variant: an `int` in C (the label's position), a
    /// `str` in Python. `labels` maps each published name to a variant, in
    /// the manifest's order; every variant has one label.
    Variant { ty: String, labels: Vec<(String, String)> },
}

impl Kind {
    pub(crate) fn word(&self) -> &'static str {
        match self {
            Kind::Num => "Num",
            Kind::Int => "Int",
            Kind::Bool => "Bool",
            Kind::Str => "Str",
            Kind::Enum(_) => "Enum",
            Kind::Arr => "Arr",
            Kind::IntArr => "IntArr",
            Kind::Cplx => "Cplx",
            Kind::Variant { .. } => "Enum",
        }
    }

    /// The names a caller chooses among: an Enum's values, a Variant's labels.
    pub fn choices(&self) -> Option<Vec<String>> {
        match self {
            Kind::Enum(vs) => Some(vs.clone()),
            Kind::Variant { labels, .. } => Some(labels.iter().map(|(l, _)| l.clone()).collect()),
            _ => None,
        }
    }
}

/// How a caller's spelling of an enum's name is read before it is compared
/// with the labels (`normalize` on a parameter): `lower` ignores case;
/// `pyscf_basis` is PySCF's rule for basis names (pyscf.gto.basis,
/// `_format_basis_name`): lower case, without `-`, `_` and spaces, and with
/// `*` read as `s`, so "6-31G*", "631g*" and "631gs" are one name.
pub const NORMALIZE_RULES: &[&str] = &["lower", "pyscf_basis"];

/// `s` under the normalization `rule` (one of NORMALIZE_RULES).
pub fn normalize(rule: &str, s: &str) -> String {
    let low = s.to_lowercase();
    match rule {
        "pyscf_basis" => low.chars().filter(|c| !matches!(c, '-' | '_' | ' ')).map(|c| if c == '*' { 's' } else { c }).collect(),
        _ => low,
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub kind: Kind,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// `min` is exclusive (`min_exclusive = 0.0`: above 0, not at it)
    pub min_excl: bool,
    /// `max` is exclusive (`max_exclusive`)
    pub max_excl: bool,
    /// an array parameter's rank, when the program needs one
    pub rank: Option<usize>,
    /// a NaN or an infinity is refused at the boundary unless this is set
    pub nonfinite: bool,
    /// a matrix argument must be square (`square = true`, a rank-2 Arr)
    pub square: bool,
    /// an array's shape, each dimension a fixed size, any size (-1), or the
    /// length of another array parameter (its name), checked when both are given
    pub shape: Option<Vec<Dim>>,
    pub doc: Option<String>,
    /// an Enum's or a Variant's: how Python reads the caller's spelling
    /// (NORMALIZE_RULES)
    pub normalize: Option<String>,
    /// the reference page's: the value's units ("fm", "MeV fm^3") and the
    /// values it is meant for, in words ("0.05 to 0.2")
    pub units: Option<String>,
    pub range: Option<String>,
}

/// One dimension of an array parameter's `shape`.
#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    Any,
    Fixed(i64),
    LenOf(String),
}

#[derive(Clone, Debug)]
pub struct ResultSpec {
    pub name: String,
    pub kind: Kind,
    pub doc: Option<String>,
    /// an array result's rank, when the manifest states it
    pub rank: Option<Rank>,
}

/// An array result's rank: one number, or one per choice of an enum
/// parameter (`rank = { by = "method", rhf = 1, uhf = 2 }`). The run is held
/// to it: a result of another rank is an error, not a silent change of shape.
#[derive(Clone, Debug, PartialEq)]
pub enum Rank {
    Fixed(usize),
    By { param: String, ranks: Vec<(String, usize)> },
}

impl Rank {
    /// The rank in words, for the header and the docstring.
    pub fn text(&self) -> String {
        match self {
            Rank::Fixed(r) => format!("rank {r}"),
            Rank::By { param, ranks } => ranks.iter().map(|(l, r)| format!("rank {r} when {} is {l}", param.to_lowercase())).collect::<Vec<_>>().join(", "),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Deck {
    pub name: String,
    pub deck: String,
    pub checks: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    /// the program's name: a C identifier, and the prefix of every exported symbol
    pub name: String,
    /// an intent file whose module text opens the docstring
    pub about: Option<String>,
    /// the program's files, in order, after the deck
    pub files: Vec<String>,
    /// the name of the enum parameter that picks a deck ("material")
    pub deck_param: String,
    pub decks: Vec<Deck>,
    pub params: Vec<Param>,
    /// parameters a caller gives all together or not at all (`[groups]`)
    pub groups: Vec<(String, Vec<String>)>,
    pub results: Vec<ResultSpec>,
    /// a JSON file of named runs, `{"case": {"param": value, ...}}`, which
    /// the generated test runs and tools/export/check.py checks bit for bit
    pub cases: Option<String>,
    /// `[reference]`: what the program is checked against outside Viv
    pub reference: Vec<(String, String)>,
    /// `kind = "kernels"`: a collection of defs, each its own entry point
    /// (section 11), rather than one program run
    pub is_kernels: bool,
    /// a kernel collection's C prefix (`vkern_`); `<name>_` by default
    pub prefix: String,
    /// a kernel collection's ABI number, `abi = 1`
    pub abi: u32,
    /// `[kernels.<name>]`, in the manifest's order
    pub kernels: Vec<Kernel>,
    /// the reference page:
    /// `summary`, `description` and `notes` (markdown with KaTeX math),
    /// `see_also`, `references` (each a table: text, doi, url) and
    /// `[[examples]]` (each a caption and the code), in the manifest's order
    pub page: Page,
}

/// The reference-page keys of a manifest, for the README and the docs.
#[derive(Clone, Debug, Default)]
pub struct Page {
    pub summary: Option<String>,
    pub description: Option<String>,
    pub notes: Option<String>,
    pub see_also: Vec<String>,
    pub references: Vec<Vec<(String, String)>>,
    pub examples: Vec<(String, String)>,
}

/// One entry point of a kernel collection: `[kernels.<name>]`.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    /// the file that holds the def, as `files` names it, and the def
    pub file: String,
    pub def: String,
    /// in order, as the def takes them: each a parameter's keys and `name`
    pub args: Vec<Param>,
    /// in order: one result, or the def's several (a Python tuple)
    pub results: Vec<KResult>,
    /// every other key of the table (`source`, `reference`, `compare`,
    /// `tolerance`, `why`, ...), as text for the README, in order
    pub info: Vec<(String, String)>,
    /// the def's record parameters (`viv export --infer` only): each one's
    /// fields are consecutive `args`, named `<param>__<field>`
    pub rec_args: Vec<RecArg>,
    /// the def returns this record: `results` are its fields, by name
    pub rec_result: Option<String>,
}

/// A record parameter of a kernel's def, passed as one argument per field
/// (Arr or `[Int]` fields only): `args[first..first + fields.len()]`.
#[derive(Clone, Debug)]
pub struct RecArg {
    pub param: String,
    pub record: String,
    pub first: usize,
    pub fields: Vec<String>,
}

/// A kernel's result: its name (the C field and the tuple's position), its
/// type and its description (`shape` and `doc`).
#[derive(Clone, Debug)]
pub struct KResult {
    pub name: String,
    pub kind: Kind,
    pub doc: Option<String>,
}

pub fn c_ident(s: &str) -> bool {
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Read a manifest. Every error names the line.
pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let sections = parse_toml(text)?;
    let mut m = Manifest { name: String::new(), about: None, files: Vec::new(), deck_param: "material".into(), decks: Vec::new(), params: Vec::new(), groups: Vec::new(), results: Vec::new(), cases: None, reference: Vec::new(), is_kernels: false, prefix: String::new(), abi: 1, kernels: Vec::new(), page: Page::default() };
    for (section, keys) in &sections {
        for (key, v, line) in keys {
            let at = |msg: String| format!("line {line}: {msg}");
            match (section.as_str(), key.as_str()) {
                ("", "name") => m.name = v.str().ok_or_else(|| at("`name` is a string".into()))?.to_string(),
                ("", "about") => m.about = Some(v.str().ok_or_else(|| at("`about` is a path".into()))?.to_string()),
                ("", "deck_param") => m.deck_param = v.str().ok_or_else(|| at("`deck_param` is a name".into()))?.to_string(),
                // a program (this); a kernel collection is the same manifest with `[kernels.<name>]`
                // tables, not built yet
                ("", "kind") => match v.str() {
                    Some("program") => {}
                    Some("kernels") => m.is_kernels = true,
                    _ => return Err(at("`kind` is \"program\" or \"kernels\"".into())),
                },
                ("", "prefix") => m.prefix = v.str().filter(|p| c_ident(p)).ok_or_else(|| at("`prefix` is the start of a C identifier, such as \"vkern_\"".into()))?.to_string(),
                ("", "abi") => match v {
                    Toml::Num(x) if *x >= 1.0 && x.fract() == 0.0 && *x < 1e6 => m.abi = *x as u32,
                    _ => return Err(at("`abi` is a whole number from 1".into())),
                },
                ("", "files") => {
                    let Toml::Arr(items) = v else { return Err(at("`files` is a list of paths".into())) };
                    for it in items {
                        m.files.push(it.str().ok_or_else(|| at("`files` is a list of paths".into()))?.to_string());
                    }
                }
                ("", "cases") => m.cases = Some(v.str().ok_or_else(|| at("`cases` is a path".into()))?.to_string()),
                ("", "summary" | "description" | "notes") => {
                    let text = v.str().ok_or_else(|| at(format!("`{key}` is a string (markdown; ''' for several lines)")))?.to_string();
                    *match key.as_str() {
                        "summary" => &mut m.page.summary,
                        "description" => &mut m.page.description,
                        _ => &mut m.page.notes,
                    } = Some(text);
                }
                ("", "see_also") => {
                    let Toml::Arr(items) = v else { return Err(at("`see_also` is a list of names".into())) };
                    for it in items {
                        m.page.see_also.push(it.str().ok_or_else(|| at("`see_also` is a list of names".into()))?.to_string());
                    }
                }
                ("", "references") => {
                    let bad = || at("`references` is a list of tables, { text = \"...\", doi = \"...\" }, with text and doi or url".into());
                    let Toml::Arr(items) = v else { return Err(bad()) };
                    for it in items {
                        let Toml::Table(t) = it else { return Err(bad()) };
                        let mut fields = Vec::new();
                        for (k, x) in t {
                            if !matches!(k.as_str(), "key" | "text" | "doi" | "url") {
                                return Err(at(format!("a reference's `{k}` is not one of key, text, doi, url")));
                            }
                            fields.push((k.clone(), x.str().ok_or_else(bad)?.to_string()));
                        }
                        if !fields.iter().any(|(k, _)| k == "text") {
                            return Err(bad());
                        }
                        m.page.references.push(fields);
                    }
                }
                ("", other) => return Err(at(format!("unknown key `{other}` (name, kind, about, files, deck_param, cases, prefix, abi, summary, description, notes, see_also, references)"))),
                (sec, kname) if sec == "kernels" => return Err(at(format!("`{kname}` under [kernels]: each kernel is a table of its own, [kernels.<name>]"))),
                (sec, key) if sec.starts_with("kernels.") => {
                    let kname = &sec["kernels.".len()..];
                    if !m.kernels.iter().any(|k| k.name == kname) {
                        if !c_ident(kname) {
                            return Err(at(format!("kernel `{kname}`: a kernel's name is a C identifier")));
                        }
                        m.kernels.push(Kernel { name: kname.to_string(), file: String::new(), def: String::new(), args: Vec::new(), results: Vec::new(), info: Vec::new(), rec_args: Vec::new(), rec_result: None });
                    }
                    let k = m.kernels.iter_mut().find(|k| k.name == kname).expect("pushed above");
                    parse_kernel_key(k, key, v).map_err(at)?;
                }
                // `[[examples]]`: each a table of its own, the caption and the code
                ("[examples]", key) => {
                    let text = v.str().ok_or_else(|| at(format!("an example's `{key}` is a string")))?.to_string();
                    let newest = keys.first().map(|(_, _, l)| l) == Some(line) || m.page.examples.is_empty();
                    if newest {
                        m.page.examples.push((String::new(), String::new()));
                    }
                    let ex = m.page.examples.last_mut().expect("pushed above");
                    match key {
                        "caption" => ex.0 = text,
                        "code" => ex.1 = text,
                        _ => return Err(at(format!("an example's `{key}` is not caption or code"))),
                    }
                }
                ("decks", name) => {
                    let Toml::Table(t) = v else { return Err(at(format!("deck `{name}` is a table: {{ deck = \"...\", checks = \"...\" }}"))) };
                    let get = |k: &str| t.iter().find(|(n, _)| n == k).and_then(|(_, v)| v.str()).map(str::to_string);
                    if let Some((bad, _)) = t.iter().find(|(n, _)| n != "deck" && n != "checks") {
                        return Err(at(format!("deck `{name}`: unknown key `{bad}` (deck, checks)")));
                    }
                    let deck = get("deck").ok_or_else(|| at(format!("deck `{name}` has no `deck` file")))?;
                    m.decks.push(Deck { name: name.to_string(), deck, checks: get("checks") });
                }
                ("params", name) => m.params.push(parse_param(name, v).map_err(at)?),
                ("groups", name) => {
                    let Toml::Arr(items) = v else { return Err(at(format!("group `{name}` is a list of parameter names"))) };
                    let names: Vec<String> = items.iter().filter_map(|x| x.str().map(str::to_string)).collect();
                    if names.len() != items.len() || names.len() < 2 {
                        return Err(at(format!("group `{name}` lists two or more parameter names")));
                    }
                    m.groups.push((name.to_string(), names));
                }
                ("results", name) => {
                    let (kind, doc, rank) = match v {
                        Toml::Str(k) => (parse_kind(k, None).map_err(at)?, None, None),
                        Toml::Table(t) => {
                            let k = t.iter().find(|(n, _)| n == "type").and_then(|(_, v)| v.str()).ok_or_else(|| at(format!("result `{name}` has no `type`")))?;
                            if let Some((bad, _)) = t.iter().find(|(n, _)| !matches!(n.as_str(), "type" | "doc" | "from" | "rank")) {
                                return Err(at(format!("result `{name}`: unknown key `{bad}` (type, doc, from, rank)")));
                            }
                            let rank = match t.iter().find(|(n, _)| n == "rank").map(|(_, v)| v) {
                                None => None,
                                Some(r) => Some(parse_rank(name, r).map_err(at)?),
                            };
                            (parse_kind(k, None).map_err(at)?, t.iter().find(|(n, _)| n == "doc").and_then(|(_, v)| v.str()).map(str::to_string), rank)
                        }
                        _ => return Err(at(format!("result `{name}` is a type or a table"))),
                    };
                    if matches!(kind, Kind::Enum(_)) {
                        return Err(at(format!("result `{name}`: an Enum is a parameter type")));
                    }
                    if rank.is_some() && !matches!(kind, Kind::Arr | Kind::IntArr) {
                        return Err(at(format!("result `{name}`: rank is for an array")));
                    }
                    m.results.push(ResultSpec { name: name.to_string(), kind, doc, rank });
                }
                // what the exported program is held to outside Viv (design
                // section 9, item 4): free text for the README
                ("reference", key) => {
                    let text = match v {
                        Toml::Str(s) => s.clone(),
                        Toml::Num(x) => format!("{x:e}"),
                        _ => return Err(at(format!("reference `{key}` is a string or a number"))),
                    };
                    m.reference.push((key.to_string(), text));
                }
                (s, _) => return Err(at(format!("unknown section `[{s}]` (decks, params, groups, results, reference, [examples])"))),
            }
        }
    }
    if !c_ident(&m.name) {
        return Err(format!("`name` must be a C identifier, not {:?}", m.name));
    }
    if m.prefix.is_empty() {
        m.prefix = format!("{}_", m.name);
    }
    if m.is_kernels {
        check_kernels_manifest(&m)?;
    } else {
        if !m.kernels.is_empty() {
            return Err("[kernels.<name>] tables are for `kind = \"kernels\"`".into());
        }
        for (what, name, kind) in m.params.iter().map(|p| ("parameter", &p.name, &p.kind)).chain(m.results.iter().map(|r| ("result", &r.name, &r.kind))) {
            if *kind == Kind::Cplx {
                return Err(format!("the {what} `{name}`: a complex number crosses the boundary as a kernel's argument or result only (kind = \"kernels\")"));
            }
        }
    }
    if m.files.is_empty() {
        return Err("`files` names no file".into());
    }
    for p in &m.params {
        if p.name == m.deck_param && !m.decks.is_empty() {
            return Err(format!("the parameter `{}` has the deck parameter's name", p.name));
        }
    }
    let mut lowered: Vec<String> = Vec::new();
    for n in m.params.iter().map(|p| p.name.to_lowercase()).chain(std::iter::once(m.deck_param.clone())) {
        if lowered.contains(&n) {
            return Err(format!("two parameters are `{n}` in Python"));
        }
        if !c_ident(&n) || PY_KEYWORDS.contains(&n.as_str()) {
            return Err(format!("`{n}` cannot be a parameter's name in C and Python"));
        }
        lowered.push(n);
    }
    // B280: a name the generated C or Python already uses
    for n in m.params.iter().map(|p| p.name.to_lowercase()).chain((!m.decks.is_empty()).then(|| m.deck_param.clone())) {
        if let Some(why) = reserved_name(&n, PARAM_FIELDS) {
            return Err(format!("the parameter `{n}` cannot be exported under that name: {why}"));
        }
    }
    // an enum's constants are `<NAME>_<PARAM>_<CHOICE>`, beside the error
    // kinds `<NAME>_E_<KIND>`
    for p in m.params.iter().filter(|p| p.kind.choices().is_some()).map(|p| p.name.to_lowercase()).chain((!m.decks.is_empty()).then(|| m.deck_param.clone())) {
        if p == "e" {
            return Err("an enum parameter named `e` would share its C constants' prefix with the error kinds (`_E_`); rename it".into());
        }
    }
    let mut seen: Vec<String> = Vec::new();
    for r in &m.results {
        let n = r.name.to_lowercase();
        if !c_ident(&n) {
            return Err(format!("the result `{}` is not a C identifier", r.name));
        }
        if let Some(why) = reserved_name(&n, RESULT_FIELDS) {
            return Err(format!("the result `{n}` cannot be exported under that name: {why}"));
        }
        if seen.contains(&n) {
            return Err(format!("two results are `{n}` in Python and C"));
        }
        seen.push(n);
    }
    for (g, names) in &m.groups {
        for n in names {
            if !m.params.iter().any(|p| p.name == *n) {
                return Err(format!("group `{g}` names `{n}`, which is not a parameter"));
            }
        }
    }
    for prm in &m.params {
        for d in prm.shape.iter().flatten() {
            if let Dim::LenOf(other) = d {
                if !m.params.iter().any(|q| q.name == *other && matches!(q.kind, Kind::Arr | Kind::IntArr)) {
                    return Err(format!("parameter `{}`: its shape names `{other}`, which is not an array parameter", prm.name));
                }
            }
        }
    }
    for r in &m.results {
        if let Some(Rank::By { param, ranks }) = &r.rank {
            let Some(prm) = m.params.iter().find(|q| q.name == *param) else {
                return Err(format!("result `{}`: its rank is by `{param}`, which is not a parameter", r.name));
            };
            let Some(choices) = prm.kind.choices() else {
                return Err(format!("result `{}`: its rank is by `{param}`, which is not an enum parameter", r.name));
            };
            for (l, _) in ranks {
                if !choices.contains(l) {
                    return Err(format!("result `{}`: `{l}` is not one of {param}'s choices ({})", r.name, choices.join(", ")));
                }
            }
            for c in &choices {
                if !ranks.iter().any(|(l, _)| l == c) {
                    return Err(format!("result `{}`: its rank names no rank for {param} = {c}", r.name));
                }
            }
        }
    }
    for d in &m.decks {
        if !c_ident(&d.name) {
            return Err(format!("deck `{}`: a deck's name is a C identifier", d.name));
        }
    }
    Ok(m)
}

/// C's keywords (C99 and C11), and the lowercase names the C library or
/// the compiler defines as object-like macros: a field of that name in the
/// generated structs does not compile (B280).
const C_RESERVED: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    "_bool", "_complex", "_imaginary", "_alignas", "_alignof", "_atomic", "_generic", "_noreturn", "_static_assert", "_thread_local",
    "bool", "true", "false", "complex", "imaginary", "errno", "stdin", "stdout", "stderr", "unix", "linux", "i386", "main",
];

/// The fields of the params struct beside the parameters.
const PARAM_FIELDS: &[&str] = &["given", "on_line", "ctx"];
/// The fields of the results struct beside the results, and the keys of the
/// Python result beside them.
const RESULT_FIELDS: &[&str] = &["have", "ntests", "tests", "error_kind", "log"];

/// Why `n` (lowercase, as it is in C and Python) cannot name a parameter or
/// result: a C keyword or macro, or a field of the struct it goes in.
pub(crate) fn reserved_name(n: &str, fields: &[&str]) -> Option<String> {
    if C_RESERVED.contains(&n) {
        return Some(format!("`{n}` is a keyword or a macro in C; rename the global"));
    }
    if fields.contains(&n) {
        return Some(format!("the generated struct has a field `{n}` of its own; rename the global"));
    }
    None
}

/// A value of a kernel's table as README text.
fn toml_text(v: &Toml) -> String {
    match v {
        Toml::Str(s) => s.clone(),
        Toml::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => format!("{}", *x as i64),
        Toml::Num(x) if (1e-3..1e6).contains(&x.abs()) => format!("{x}"),
        Toml::Num(x) => format!("{x:e}"),
        Toml::Bool(b) => b.to_string(),
        Toml::Arr(xs) => xs.iter().map(toml_text).collect::<Vec<_>>().join(", "),
        Toml::Table(t) => t.iter().map(|(k, x)| format!("{k} = {}", toml_text(x))).collect::<Vec<_>>().join(", "),
    }
}

/// One key of `[kernels.<name>]`.
fn parse_kernel_key(k: &mut Kernel, key: &str, v: &Toml) -> Result<(), String> {
    let name = k.name.clone();
    // a list of inline tables, each with its `name`
    let named = |v: &Toml, what: &str| -> Result<Vec<(String, Vec<(String, Toml)>)>, String> {
        let Toml::Arr(items) = v else { return Err(format!("kernel `{name}`: `{what}` is a list of tables, [{{ name = \"x\", type = \"Num\" }}, ...]")) };
        let mut out = Vec::new();
        for it in items {
            let Toml::Table(t) = it else { return Err(format!("kernel `{name}`: each of `{what}` is a table with `name` and `type`")) };
            let n = t.iter().find(|(f, _)| f == "name").and_then(|(_, x)| x.str()).ok_or_else(|| format!("kernel `{name}`: each of `{what}` has a `name`"))?;
            out.push((n.to_string(), t.iter().filter(|(f, _)| f != "name").cloned().collect()));
        }
        Ok(out)
    };
    match key {
        "viv" => {
            let spec = v.str().ok_or_else(|| format!("kernel `{name}`: `viv` is \"file.viv:def\""))?;
            let Some((file, def)) = spec.rsplit_once(':') else { return Err(format!("kernel `{name}`: `viv` is \"file.viv:def\", not {spec:?}")) };
            if !c_ident(def) {
                return Err(format!("kernel `{name}`: `{def}` is not a def's name"));
            }
            k.file = file.to_string();
            k.def = def.to_string();
        }
        "args" => {
            for (an, t) in named(v, "args")? {
                if !c_ident(&an) || PY_KEYWORDS.contains(&an.as_str()) {
                    return Err(format!("kernel `{name}`: `{an}` cannot be an argument's name in C and Python"));
                }
                let p = parse_param(&an, &Toml::Table(t)).map_err(|e| format!("kernel `{name}`: {e}"))?;
                if !matches!(p.kind, Kind::Num | Kind::Int | Kind::Bool | Kind::Cplx | Kind::Arr) {
                    return Err(format!("kernel `{name}`: the argument `{an}` is {}; a kernel takes Num, Int, Bool, Complex and Arr", p.kind.word()));
                }
                k.args.push(p);
            }
        }
        "results" => {
            for (rn, t) in named(v, "results")? {
                if !c_ident(&rn) {
                    return Err(format!("kernel `{name}`: `{rn}` cannot be a result's name in C"));
                }
                let ty = t.iter().find(|(f, _)| f == "type").and_then(|(_, x)| x.str()).ok_or_else(|| format!("kernel `{name}`: the result `{rn}` has no `type`"))?;
                let kind = parse_kind(ty, None).map_err(|e| format!("kernel `{name}`: the result `{rn}`: {e}"))?;
                if !matches!(kind, Kind::Num | Kind::Int | Kind::Bool | Kind::Cplx | Kind::Arr) {
                    return Err(format!("kernel `{name}`: the result `{rn}` is {}; a kernel gives Num, Int, Bool, Complex and Arr", kind.word()));
                }
                let mut doc: Vec<String> = Vec::new();
                for (f, x) in &t {
                    match (f.as_str(), x) {
                        ("type", _) => {}
                        ("doc", Toml::Str(d)) => doc.insert(0, d.clone()),
                        // its shape in words, for the docstring: "n_max + 1"
                        ("shape", x) => doc.push(format!("shape {}", toml_text(x))),
                        (other, _) => return Err(format!("kernel `{name}`: the result `{rn}`: `{other}` is not one of type, doc, shape")),
                    }
                }
                k.results.push(KResult { name: rn, kind, doc: (!doc.is_empty()).then(|| doc.join("; ")) });
            }
        }
        "writes" => match v {
            Toml::Arr(xs) if xs.is_empty() => {}
            _ => return Err(format!("kernel `{name}`: `writes` (arguments a kernel writes in place) is not built yet; return the array instead")),
        },
        other => k.info.push((other.to_string(), toml_text(v))),
    }
    Ok(())
}

/// The entry points every kernel collection has beside its kernels.
const KERNEL_FIXED: &[&str] = &["init", "abi", "version", "interrupt", "set_output", "ntests", "test_name", "test_passed", "test_detail"];

/// What a kernel manifest must hold, past each key's own check.
pub(crate) fn check_kernels_manifest(m: &Manifest) -> Result<(), String> {
    if !m.decks.is_empty() || !m.params.is_empty() || !m.results.is_empty() || !m.groups.is_empty() {
        return Err("a kernel collection has [kernels.<name>] tables, not [decks], [params], [groups] or [results]".into());
    }
    if m.kernels.is_empty() {
        return Err("`kind = \"kernels\"` and no [kernels.<name>] table".into());
    }
    for k in &m.kernels {
        let at = |e: String| format!("kernel `{}`: {e}", k.name);
        if k.def.is_empty() {
            return Err(at("no `viv = \"file.viv:def\"`".into()));
        }
        if !m.files.contains(&k.file) {
            return Err(at(format!("its def is in {}, which `files` does not name", k.file)));
        }
        if KERNEL_FIXED.contains(&k.name.as_str()) || ["_free", "_args", "_results"].iter().any(|s| k.name.ends_with(s)) {
            return Err(at(format!("`{}{}` is a name the collection's own entry points use", m.prefix, k.name)));
        }
        if k.results.is_empty() {
            return Err(at("no `results`; a kernel gives one or more".into()));
        }
        let mut seen: Vec<String> = Vec::new();
        for n in k.args.iter().map(|a| &a.name).chain(k.results.iter().map(|r| &r.name)) {
            if seen.contains(n) {
                return Err(at(format!("two arguments or results are `{n}`")));
            }
            if let Some(why) = reserved_name(&n.to_lowercase(), &["error_kind"]) {
                return Err(at(format!("`{n}`: {why}")));
            }
            seen.push(n.clone());
        }
    }
    Ok(())
}

const PY_KEYWORDS: &[&str] = &["and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "none", "true", "false", "log", "on_line", "lib"];

/// The type words of the manifest; any other `type` is an enum the program declares.
const KIND_WORDS: &[&str] = &["Num", "Int", "Bool", "Str", "Arr", "IntArr", "[Int]", "Enum", "CArr", "Complex", "Cplx"];

fn parse_kind(word: &str, values: Option<&Toml>) -> Result<Kind, String> {
    Ok(match word {
        "Num" => Kind::Num,
        "Int" => Kind::Int,
        "Bool" => Kind::Bool,
        "Str" => Kind::Str,
        "Arr" => Kind::Arr,
        "IntArr" | "[Int]" => Kind::IntArr,
        "Enum" => {
            let Some(Toml::Arr(vs)) = values else { return Err("an Enum lists its `values`".into()) };
            let names: Vec<String> = vs.iter().filter_map(|v| v.str().map(str::to_string)).collect();
            if names.len() != vs.len() || names.is_empty() {
                return Err("an Enum's `values` are strings".into());
            }
            Kind::Enum(names)
        }
        // `Cplx` is the name `viv features` gives; `Complex` the one k1's
        // adapters annotate with. A program's parameters refuse it (below)
        "Complex" | "Cplx" => Kind::Cplx,
        "CArr" => return Err(format!("`{word}`: complex arrays cannot cross the boundary yet")),
        other => return Err(format!("`{other}` is not a type the export takes: Num, Int, Bool, Str, Enum, Arr, IntArr (records, maps and function values are refused)")),
    })
}

/// A result's `rank`: a number from 1 to 8, or `{ by = "param", label = n, ... }`.
fn parse_rank(name: &str, v: &Toml) -> Result<Rank, String> {
    let one = |x: &Toml| match x {
        Toml::Num(r) if *r >= 1.0 && *r <= 8.0 && r.fract() == 0.0 => Some(*r as usize),
        _ => None,
    };
    match v {
        Toml::Table(t) => {
            let by = t.iter().find(|(n, _)| n == "by").and_then(|(_, v)| v.str()).ok_or_else(|| format!("result `{name}`: a rank table names its parameter with `by`"))?;
            let mut ranks = Vec::new();
            for (k, x) in t.iter().filter(|(n, _)| n != "by") {
                ranks.push((k.clone(), one(x).ok_or_else(|| format!("result `{name}`: the rank for {k} is a number from 1 to 8"))?));
            }
            Ok(Rank::By { param: by.to_string(), ranks })
        }
        x => one(x).map(Rank::Fixed).ok_or_else(|| format!("result `{name}`: rank is a number from 1 to 8, or a table by an enum parameter")),
    }
}

fn parse_param(name: &str, v: &Toml) -> Result<Param, String> {
    let mut p = Param { name: name.to_string(), kind: Kind::Num, min: None, max: None, min_excl: false, max_excl: false, square: false, rank: None, nonfinite: false, shape: None, doc: None, normalize: None, units: None, range: None };
    // a word that is not a type of the manifest may be an enum of the program
    let hint = |k: &str, e: String| if KIND_WORDS.contains(&k) || !c_ident(k) { e } else { format!("{e}; an enum the program declares takes `labels`, its published names") };
    match v {
        Toml::Str(k) => p.kind = parse_kind(k, None).map_err(|e| hint(k, e))?,
        Toml::Table(t) => {
            let get = |k: &str| t.iter().find(|(n, _)| n == k).map(|(_, v)| v);
            let k = get("type").and_then(Toml::str).ok_or_else(|| format!("parameter `{name}` has no `type`"))?;
            p.kind = match get("labels") {
                // an enum the program declares: its published names
                Some(Toml::Table(ls)) if !KIND_WORDS.contains(&k) && c_ident(k) => {
                    let mut labels = Vec::new();
                    for (label, variant) in ls {
                        let Some(v) = variant.str() else { return Err(format!("parameter `{name}`: each label names a variant of `{k}`, as a string")) };
                        if label.is_empty() || label.contains('\0') {
                            return Err(format!("parameter `{name}`: a label is a nonempty name"));
                        }
                        if labels.iter().any(|(_, w): &(String, String)| w == v) {
                            return Err(format!("parameter `{name}`: the variant `{v}` has two labels; give it one"));
                        }
                        labels.push((label.clone(), v.to_string()));
                    }
                    if labels.is_empty() {
                        return Err(format!("parameter `{name}`: `labels` is empty"));
                    }
                    Kind::Variant { ty: k.to_string(), labels }
                }
                Some(_) => return Err(format!("parameter `{name}`: `labels` is a table of published name = \"Variant\", for an enum the program declares (`type` is its name)")),
                None => parse_kind(k, get("values")).map_err(|e| hint(k, e))?,
            };
            for (key, val) in t {
                match (key.as_str(), val) {
                    ("type", _) | ("values", _) | ("labels", _) => {}
                    ("normalize", Toml::Str(r)) if NORMALIZE_RULES.contains(&r.as_str()) => p.normalize = Some(r.clone()),
                    ("min" | "min_exclusive", Toml::Num(x)) => {
                        if p.min.is_some() {
                            return Err(format!("parameter `{name}`: give `min` or `min_exclusive`, not both"));
                        }
                        p.min = Some(*x);
                        p.min_excl = key == "min_exclusive";
                    }
                    ("max" | "max_exclusive", Toml::Num(x)) => {
                        if p.max.is_some() {
                            return Err(format!("parameter `{name}`: give `max` or `max_exclusive`, not both"));
                        }
                        p.max = Some(*x);
                        p.max_excl = key == "max_exclusive";
                    }
                    ("rank", Toml::Num(x)) if *x >= 1.0 && *x <= 8.0 && x.fract() == 0.0 => p.rank = Some(*x as usize),
                    ("doc", Toml::Str(s)) => p.doc = Some(s.clone()),
                    ("units", Toml::Str(s)) => p.units = Some(s.clone()),
                    ("range", Toml::Str(s)) => p.range = Some(s.clone()),
                    ("nonfinite", Toml::Bool(b)) => p.nonfinite = *b,
                    ("square", Toml::Bool(b)) => p.square = *b,
                    // the enum parameter that picks a contract's method
                    // read by vipyr's `using=`
                    ("contract", Toml::Str(_)) if get("labels").is_some() => {}
                    ("shape", Toml::Arr(ds)) if !ds.is_empty() && ds.len() <= 8 => {
                        let mut dims = Vec::new();
                        for d in ds {
                            dims.push(match d {
                                Toml::Num(x) if *x == -1.0 => Dim::Any,
                                Toml::Num(x) if *x >= 0.0 && x.fract() == 0.0 => Dim::Fixed(*x as i64),
                                // `"ATOMS"`, or `"len(ATOMS)"` as the vivscf manifest wrote it
                                Toml::Str(n) => Dim::LenOf(n.strip_prefix("len(").and_then(|x| x.strip_suffix(')')).unwrap_or(n).trim().to_string()),
                                _ => return Err(format!("parameter `{name}`: a shape's dimension is a size, -1 or another array parameter's name")),
                            });
                        }
                        p.shape = Some(dims);
                    }
                    (k, _) => return Err(format!("parameter `{name}`: `{k}` is not one of type, min, max, min_exclusive, max_exclusive, rank, square, shape, values, labels, normalize ({}), nonfinite, doc, units, range, contract (or has the wrong kind of value)", NORMALIZE_RULES.join(", "))),
                }
            }
        }
        _ => return Err(format!("parameter `{name}` is a type or a table")),
    }
    if (p.min.is_some() || p.max.is_some()) && !matches!(p.kind, Kind::Num | Kind::Int | Kind::Arr | Kind::IntArr) {
        return Err(format!("parameter `{name}`: min and max are for numbers and arrays (each element)"));
    }
    if p.rank.is_some() && p.kind != Kind::Arr {
        return Err(format!("parameter `{name}`: rank is for an Arr"));
    }
    if let Some(sh) = &p.shape {
        if !matches!(p.kind, Kind::Arr | Kind::IntArr) {
            return Err(format!("parameter `{name}`: shape is for an array"));
        }
        if p.rank.is_some_and(|r| r != sh.len()) || (p.kind == Kind::IntArr && sh.len() != 1) {
            return Err(format!("parameter `{name}`: its shape has {} dimensions", sh.len()));
        }
        if p.kind == Kind::Arr {
            p.rank = Some(sh.len());
        }
    }
    if p.nonfinite && !matches!(p.kind, Kind::Num | Kind::Arr | Kind::Cplx) {
        return Err(format!("parameter `{name}`: nonfinite is for a Num, a Cplx or an Arr"));
    }
    if p.square {
        if p.kind != Kind::Arr || p.rank.is_some_and(|r| r != 2) {
            return Err(format!("parameter `{name}`: square is for a matrix, an Arr of rank 2"));
        }
        p.rank = Some(2);
    }
    if let Some(rule) = &p.normalize {
        let Some(choices) = p.kind.choices() else { return Err(format!("parameter `{name}`: normalize is for an enum")) };
        let mut seen: Vec<(String, &String)> = Vec::new();
        for c in &choices {
            let n = normalize(rule, c);
            if let Some((_, other)) = seen.iter().find(|(k, _)| *k == n) {
                return Err(format!("parameter `{name}`: `{other}` and `{c}` are one name under normalize = \"{rule}\""));
            }
            seen.push((n, c));
        }
    }
    Ok(p)
}

// -------------------------------------------------------- the program text

/// A piece of the assembled program: a slice of a file (its name, and the
/// byte offset of the slice in it), or text `viv export` wrote (`None`).
#[derive(Clone, Debug)]
pub struct Part {
    pub file: Option<(String, usize)>,
    pub text: String,
}

/// Where a parameter is read in C: its slot in one of the entry's tables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Slot {
    Num(usize),
    Int(usize),
    Str(usize),
    Arr(usize),
}

/// The program as `viv export` assembles it, and what the generated
/// interface needs to know about it.
#[derive(Clone, Debug, Default)]
pub struct Assembly {
    pub parts: Vec<Part>,
    /// each parameter's slot (by manifest order); `None` before typing
    pub slots: Vec<Option<Slot>>,
    /// the deck's text of each parameter's value, per deck (or the one
    /// program value when there are no decks): for the docstring
    pub defaults: Vec<Vec<Option<String>>>,
    /// per deck (or once), per result: whether that deck's run assigns it
    pub produced: Vec<Vec<bool>>,
    /// each `#:` glossary line of the files: name -> text. A parameter's or
    /// result's entry is the nearest `#: NAME` line at or above its
    /// assignment in the same file; other names keep their first entry.
    pub glossary: HashMap<String, String>,
    /// per parameter, per deck: the nearest `#: NAME` line at or above the
    /// deck's assignment, when the deck has one (decks may describe a
    /// parameter differently: a cutoff chosen for that material)
    pub deck_help: Vec<Vec<Option<String>>>,
    /// for each parameter of kind Variant (by manifest order; empty for the
    /// others): the tag of the variant each label names, in label order
    pub variant_tags: Vec<Vec<usize>>,
}

/// The number of slots of each table the entry fills.
pub fn table_sizes(slots: &[Option<Slot>], decks: bool) -> (usize, usize, usize, usize) {
    let mut n = (usize::from(decks), 0, 0, 0);
    for s in slots.iter().flatten() {
        match *s {
            Slot::Num(k) => n.0 = n.0.max(k + 1),
            Slot::Int(k) => n.1 = n.1.max(k + 1),
            Slot::Str(k) => n.2 = n.2.max(k + 1),
            Slot::Arr(k) => n.3 = n.3.max(k + 1),
        }
    }
    n
}

/// A refusal, located in a file: Viv's diagnostic text.
fn located(file: &str, text: &str, span: knotkit::span::Span, msg: &str, label: &str) -> String {
    use knotkit::diag::{Diagnostic, PlainRenderer, Renderer};
    let mut map = knotkit::span::SourceMap::new();
    let id = map.add(file.to_string(), text.to_string());
    let span = knotkit::span::Span::new(id, span.start, span.end);
    PlainRenderer.render(&Diagnostic::error(msg.to_string()).with_primary(span, label), &map)
}

fn line_of(text: &str, byte: usize) -> usize {
    text[..byte.min(text.len())].matches('\n').count()
}

/// The byte where the line after the one holding `byte` starts.
fn next_line_start(text: &str, byte: usize) -> usize {
    text[byte.min(text.len())..].find('\n').map_or(text.len(), |k| byte + k + 1)
}

/// Every assignment of `name` in `stmts` and the blocks inside them, with
/// whether it is at the top (not nested in a block).
fn assigns<'a>(stmts: &'a [Stmt], name: &str, top: bool, out: &mut Vec<(&'a Stmt, bool)>) {
    for s in stmts {
        match s {
            Stmt::Assign { name: n, .. } if n == name => out.push((s, top)),
            Stmt::If { then_body, else_body, .. } => {
                assigns(then_body, name, false, out);
                assigns(else_body, name, false, out);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => assigns(body, name, false, out),
            _ => {}
        }
    }
}

fn stmt_span(s: &Stmt) -> knotkit::span::Span {
    match s {
        Stmt::Assign { span, .. } => *span,
        _ => knotkit::span::Span::synthetic(knotkit::span::SourceId(0)),
    }
}

/// Whether `name` is assigned anywhere in a file's top level (blocks
/// included, defs not: a def's assignment makes a local).
fn assigned_anywhere(p: &Program, name: &str) -> bool {
    let mut out = Vec::new();
    for item in &p.items {
        if let Item::Stmt(s) = item {
            assigns(std::slice::from_ref(s), name, true, &mut out);
        }
    }
    !out.is_empty()
}

/// A file's `#:` glossary: `#: NAME  what it is` -> NAME -> "what it is".
fn glossary(text: &str, out: &mut HashMap<String, String>) {
    for l in text.lines() {
        if let Some(k) = l.find("#:") {
            let rest = l[k + 2..].trim();
            if let Some((name, what)) = rest.split_once(char::is_whitespace) {
                if c_ident(name) {
                    out.entry(name.to_string()).or_insert_with(|| what.trim().to_string());
                }
            }
        }
    }
}

/// The nearest `#: NAME text` of `text` on line `line` (0-based) or above it.
fn glossary_above(text: &str, name: &str, line: usize) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    for l in lines[..(line + 1).min(lines.len())].iter().rev() {
        if let Some(k) = l.find("#:") {
            if let Some((n, what)) = l[k + 2..].trim().split_once(char::is_whitespace) {
                if n == name {
                    return Some(what.trim().to_string());
                }
            }
        }
    }
    None
}

/// Whether a top-level statement of `p` that ends before byte `at` reads the
/// variable `name` (a def's body is not looked into: it reads the global
/// when it is called).
fn read_before(p: &Program, name: &str, at: u32) -> bool {
    p.items.iter().any(|item| match item {
        Item::Stmt(s) => s.span().end <= at && stmt_reads(s, name),
        _ => false,
    })
}

fn stmt_reads(s: &Stmt, v: &str) -> bool {
    let body = |b: &[Stmt]| b.iter().any(|x| stmt_reads(x, v));
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => expr_reads(value, v),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => expr_reads(expr, v),
        Stmt::Test { body: b, .. } => body(b),
        Stmt::SetIndex { base, indices, value, .. } => expr_reads(base, v) || indices.iter().any(|i| expr_reads(i, v)) || expr_reads(value, v),
        Stmt::SetField { base, value, .. } => expr_reads(base, v) || expr_reads(value, v),
        Stmt::If { cond, then_body, else_body, .. } => expr_reads(cond, v) || body(then_body) || body(else_body),
        Stmt::While { cond, body: b, .. } => expr_reads(cond, v) || body(b),
        Stmt::For { bound, body: b, .. } => expr_reads(bound, v) || body(b),
        _ => false,
    }
}

fn expr_reads(e: &Expr, v: &str) -> bool {
    match e {
        Expr::Var { name, .. } => name == v,
        Expr::Index { base, indices, .. } => expr_reads(base, v) || indices.iter().any(|i| expr_reads(i, v)),
        Expr::Lambda { params, body, .. } => !params.iter().any(|p| p == v) && expr_reads(body, v),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_reads(operand, v),
        Expr::Binary { lhs, rhs, .. } => expr_reads(lhs, v) || expr_reads(rhs, v),
        Expr::Call { args, .. } | Expr::ArrLit { elems: args, .. } => args.iter().any(|a| expr_reads(a, v)),
        Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| expr_reads(x, v)),
        Expr::Field { base, .. } => expr_reads(base, v),
        Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => false,
    }
}

/// The global's Viv type, as the typing pass found it.
fn type_word(t: Ty) -> String {
    match t {
        Ty::Num => "Num".into(),
        Ty::Int => "Int".into(),
        Ty::Bool => "Bool".into(),
        Ty::Str => "Str".into(),
        Ty::Arr(Elem::Num) => "an array of numbers".into(),
        Ty::Arr(Elem::Int) => "[Int]".into(),
        Ty::Enum(n) => format!("the enum {n}"),
        other => format!("{other:?}"),
    }
}

/// The `__export_*` call a parameter's global reads, and the slot it reads
/// from, for a parameter of `kind` and a global of type `ty`.
fn reader(kind: &Kind, ty: Ty, counts: &mut [usize; 4]) -> Option<(&'static str, Slot)> {
    let mut next = |t: usize| {
        let k = counts[t];
        counts[t] += 1;
        k
    };
    Some(match (kind, ty) {
        (Kind::Num, Ty::Num) | (Kind::Int, Ty::Num) | (Kind::Bool, Ty::Num) | (Kind::Enum(_), Ty::Num) => ("__export_num", Slot::Num(next(0))),
        (Kind::Int, Ty::Int) | (Kind::Enum(_), Ty::Int) => ("__export_int", Slot::Int(next(1))),
        (Kind::Bool, Ty::Bool) => ("__export_bool", Slot::Int(next(1))),
        (Kind::Str, Ty::Str) | (Kind::Enum(_), Ty::Str) => ("__export_str", Slot::Str(next(2))),
        (Kind::Arr, Ty::Arr(Elem::Num)) => ("__export_arr", Slot::Arr(next(3))),
        (Kind::Variant { ty: want, .. }, Ty::Enum(n)) if n == want => ("__export_num", Slot::Num(next(0))),
        (Kind::IntArr, Ty::Arr(Elem::Int)) => ("__export_iarr", Slot::Arr(next(3))),
        _ => return None,
    })
}

/// The name of the global that holds the deck's index in the program.
pub const DECK_GLOBAL: &str = "__export_deck";

/// Assemble the program: each deck's statements in a block run when its
/// index is the deck parameter's value, then the files, then each deck's
/// checks the same way. With `types` (the globals' types from a first
/// compile of the text this returns without them), each parameter's
/// assignment is followed by a line that replaces its value when the caller
/// gave one. `read` reads a file by its manifest path.
pub fn assemble(m: &Manifest, read: &dyn Fn(&str) -> Result<String, String>, types: Option<&HashMap<String, Ty>>) -> Result<Assembly, String> {
    assemble_mapped(m, read, types, &|_| None)
}

/// One file of the manifest as `assemble` reads it: its text with its
/// imports spliced in, parsed, and the map of where each line came from.
struct Unit {
    name: String,
    text: String,
    prog: Program,
    map: Option<knotkit::span::SourceMap>,
}

impl Unit {
    /// A refusal at `span` of this unit's text, shown in the file the line
    /// came from: an imported module's line under the module's name and its
    /// own line number (B273), when the unit has a map.
    fn located(&self, span: knotkit::span::Span, msg: &str, label: &str) -> String {
        match &self.map {
            Some(map) if map.file(knotkit::span::SourceId(0)).is_some_and(|f| f.text == self.text) => {
                use knotkit::diag::{Diagnostic, PlainRenderer, Renderer};
                let span = knotkit::span::Span::new(knotkit::span::SourceId(0), span.start, span.end);
                PlainRenderer.render(&Diagnostic::error(msg.to_string()).with_primary(span, label), map)
            }
            _ => located(&self.name, &self.text, span, msg, label),
        }
    }
}

/// `assemble`, with `map_of(path)` giving the source map of a file `read`
/// read (main.rs keeps them): refusals in an imported module's text are then
/// shown in that module.
pub fn assemble_mapped(m: &Manifest, read: &dyn Fn(&str) -> Result<String, String>, types: Option<&HashMap<String, Ty>>, map_of: &dyn Fn(&str) -> Option<knotkit::span::SourceMap>) -> Result<Assembly, String> {
    let load = |path: &str| -> Result<Unit, String> {
        let text = read(path)?;
        let map = map_of(path);
        let prog = crate::parse(&text, knotkit::span::SourceId(0)).map_err(|ds| {
            use knotkit::diag::{PlainRenderer, Renderer};
            match &map {
                Some(map) => ds.iter().map(|d| PlainRenderer.render(d, map)).collect::<String>(),
                None => {
                    let mut map = knotkit::span::SourceMap::new();
                    map.add(path.to_string(), text.clone());
                    ds.iter().map(|d| PlainRenderer.render(d, &map)).collect::<String>()
                }
            }
        })?;
        Ok(Unit { name: path.to_string(), text, prog, map })
    };
    let decks: Vec<Unit> = m.decks.iter().map(|d| load(&d.deck)).collect::<Result<_, _>>()?;
    let checks: Vec<Option<Unit>> = m.decks.iter().map(|d| d.checks.as_deref().map(load).transpose()).collect::<Result<_, _>>()?;
    let files: Vec<Unit> = m.files.iter().map(|f| load(f)).collect::<Result<_, _>>()?;
    let mut errors = Vec::new();

    // a deck and a check file go inside a block: statements only
    for u in decks.iter().chain(checks.iter().flatten()) {
        for item in &u.prog.items {
            let what = match item {
                Item::Def(d) => Some(("a def", d.span)),
                Item::Record(r) => Some(("a record", r.span)),
                Item::Enum(e) => Some(("an enum", e.span)),
                Item::Rung(r) => Some(("a rung", r.span)),
                Item::Stmt(_) => None,
            };
            if let Some((what, span)) = what {
                errors.push(u.located(span, &format!("a deck or check file holds statements only, and this is {what}"), "move it to one of the program's files"));
            }
        }
        if u.text.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with("import ") || t.starts_with("from ") || t.starts_with("#! use")
        }) {
            errors.push(format!("error: {} imports a module; a deck or check file holds statements only\n", u.name));
        }
    }

    // an enum parameter's type is declared in the program, with no payloads,
    // and every variant has one label
    let mut variant_tags: Vec<Vec<usize>> = vec![Vec::new(); m.params.len()];
    for (pi, p) in m.params.iter().enumerate() {
        let Kind::Variant { ty, labels } = &p.kind else { continue };
        let decl = files.iter().chain(decks.iter()).flat_map(|u| u.prog.items.iter()).find_map(|it| match it {
            Item::Enum(e) if e.name == *ty => Some(e),
            _ => None,
        });
        let Some(decl) = decl else {
            errors.push(format!("error: the parameter `{}` is of type `{ty}`, and the program declares no enum `{ty}`\n", p.name));
            continue;
        };
        if let Some(v) = decl.variants.iter().find(|v| !v.payload.is_empty()) {
            errors.push(format!("error: the parameter `{}`: the variant `{ty}.{}` holds values; an enum parameter's variants are names only\n", p.name, v.name));
            continue;
        }
        for (label, variant) in labels {
            match decl.variants.iter().position(|v| v.name == *variant) {
                Some(tag) => variant_tags[pi].push(tag),
                None => errors.push(format!("error: the parameter `{}`: the label `{label}` names `{ty}.{variant}`, which is not a variant (they are {})\n", p.name, decl.variants.iter().map(|v| v.name.as_str()).collect::<Vec<_>>().join(", "))),
            }
        }
        for v in &decl.variants {
            if !labels.iter().any(|(_, w)| *w == v.name) {
                errors.push(format!("error: the parameter `{}`: the variant `{ty}.{}` has no label in the manifest\n", p.name, v.name));
            }
        }
    }

    // where each parameter is assigned: once in each deck, or else once in the files
    let ndecks = decks.len().max(1);
    let mut sites: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; ndecks]; m.params.len()];
    // (unit index: 0.. decks, then 1000 + file index; byte where its line after starts)
    let mut defaults = vec![vec![None; ndecks]; m.params.len()];
    for (pi, p) in m.params.iter().enumerate() {
        let mut in_files: Vec<(usize, &Stmt, bool)> = Vec::new();
        // a def that assigns the name makes a local of its own (the files'
        // imports are spliced in, so lib/hf's `method = setup.method` is
        // here); refused only when no top level assigns the parameter
        let mut in_def: Option<String> = None;
        for (fi, f) in files.iter().enumerate() {
            let mut out = Vec::new();
            for item in &f.prog.items {
                match item {
                    Item::Stmt(s) => assigns(std::slice::from_ref(s), &p.name, true, &mut out),
                    Item::Def(d) => {
                        let mut inner = Vec::new();
                        assigns(&d.body, &p.name, true, &mut inner);
                        if let (Some((s, _)), None) = (inner.first(), &in_def) {
                            in_def = Some(f.located(stmt_span(s), &format!("the parameter `{}` is assigned only inside `{}`, where it is the def's own local", p.name, d.name), "a parameter is a global the top level sets once"));
                        }
                    }
                    _ => {}
                }
            }
            in_files.extend(out.into_iter().map(|(s, top)| (fi, s, top)));
        }
        for (fi, s, top) in &in_files {
            if !top {
                errors.push(files[*fi].located(stmt_span(s), &format!("the parameter `{}` is assigned inside a block", p.name), "a parameter is assigned once, at the top level"));
            }
        }
        let file_site = |errors: &mut Vec<String>| -> Option<(usize, usize)> {
            match in_files.as_slice() {
                [(fi, s, true)] => {
                    let f = &files[*fi];
                    let sp = stmt_span(s);
                    if read_before(&f.prog, &p.name, sp.start) {
                        errors.push(f.located(sp, &format!("`{}` is read before this assignment", p.name), "a parameter's value must be set before the program reads it"));
                    }
                    Some((1000 + fi, next_line_start(&f.text, sp.end.saturating_sub(1) as usize)))
                }
                [] => None,
                [_, (fi, s, _), ..] => {
                    errors.push(files[*fi].located(stmt_span(s), &format!("the parameter `{}` is assigned more than once", p.name), "a parameter is assigned once"));
                    None
                }
                [(_, _, false)] => None,
            }
        };
        if decks.is_empty() {
            match file_site(&mut errors) {
                Some(site) => {
                    sites[pi][0] = Some(site);
                    let (fi, s, _) = in_files[0];
                    if let Stmt::Assign { value, .. } = s {
                        defaults[pi][0] = Some(expr_text(&files[fi].text, value));
                    }
                }
                None if in_files.is_empty() => errors.push(in_def.clone().unwrap_or_else(|| format!("error: the parameter `{}` is assigned nowhere in the program's files\n", p.name))),
                None => {}
            }
            continue;
        }
        for (di, d) in decks.iter().enumerate() {
            let mut out = Vec::new();
            for item in &d.prog.items {
                if let Item::Stmt(s) = item {
                    assigns(std::slice::from_ref(s), &p.name, true, &mut out);
                }
            }
            match out.as_slice() {
                [] => {
                    if in_files.is_empty() {
                        errors.push(in_def.clone().unwrap_or_else(|| format!("error: the parameter `{}` is assigned neither in deck {} ({}) nor in the program's files\n", p.name, m.decks[di].name, d.name)));
                    } else if let Some(site) = file_site(&mut Vec::new()) {
                        // the program's own value for every deck that sets none
                        sites[pi][di] = Some(site);
                        let (fi, s, _) = in_files[0];
                        if let Stmt::Assign { value, .. } = s {
                            defaults[pi][di] = Some(expr_text(&files[fi].text, value));
                        }
                    }
                }
                [(s, true)] => {
                    if let Some((fi, fs, _)) = in_files.first() {
                        errors.push(files[*fi].located(stmt_span(fs), &format!("the parameter `{}` is assigned here, after deck {} sets it", p.name, m.decks[di].name), "a parameter is assigned once"));
                    }
                    let sp = stmt_span(s);
                    if read_before(&d.prog, &p.name, sp.start) {
                        errors.push(d.located(sp, &format!("`{}` is read before this assignment", p.name), "a parameter's value must be set before the deck reads it"));
                    }
                    sites[pi][di] = Some((di, next_line_start(&d.text, sp.end.saturating_sub(1) as usize)));
                    if let Stmt::Assign { value, .. } = s {
                        defaults[pi][di] = Some(expr_text(&d.text, value));
                    }
                }
                [(s, false)] => errors.push(d.located(stmt_span(s), &format!("the parameter `{}` is assigned inside a block", p.name), "a parameter is assigned once, at the top level")),
                [_, (s, _), ..] => errors.push(d.located(stmt_span(s), &format!("the parameter `{}` is assigned more than once", p.name), "a parameter is assigned once")),
            }
        }
    }

    // the line after each assignment, by unit: (byte, parameter)
    let mut inserts: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut slots = vec![None; m.params.len()];
    let mut counts = [usize::from(!m.decks.is_empty()), 0, 0, 0];
    let mut lines_for: Vec<String> = vec![String::new(); m.params.len()];
    if let Some(types) = types {
        for (pi, p) in m.params.iter().enumerate() {
            let Some(&ty) = types.get(&p.name) else {
                errors.push(format!("error: the parameter `{}` is not a global of the program\n", p.name));
                continue;
            };
            let Some((call, slot)) = reader(&p.kind, ty, &mut counts) else {
                errors.push(format!("error: the parameter `{}` is {} in the program, and the manifest says {}\n", p.name, type_word(ty), p.kind.word()));
                continue;
            };
            let k = match slot {
                Slot::Num(k) | Slot::Int(k) | Slot::Str(k) | Slot::Arr(k) => k,
            };
            slots[pi] = Some(slot);
            lines_for[pi] = match &p.kind {
                // the label's position, read as the variant it names: one
                // line per label, so the program stays plain Viv
                Kind::Variant { ty, labels } => {
                    let mut t = format!("if __export_given({pi}) {{\n");
                    for (li, (_, variant)) in labels.iter().enumerate() {
                        let _ = writeln!(t, "    if {call}({k}) == {li}.0 {{ {} = {ty}.{variant} }}", p.name);
                    }
                    t.push_str("}\n");
                    t
                }
                _ => format!("if __export_given({pi}) {{ {} = {call}({k}) }}\n", p.name),
            };
        }
    }
    for (pi, per) in sites.iter().enumerate() {
        let mut seen = Vec::new();
        for &(unit, at) in per.iter().flatten() {
            if !seen.contains(&(unit, at)) {
                seen.push((unit, at));
                inserts.entry(unit).or_default().push((at, pi));
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.concat());
    }

    // the text, in parts
    let mut parts: Vec<Part> = Vec::new();
    let gen = |parts: &mut Vec<Part>, t: String| parts.push(Part { file: None, text: t });
    let cut = |parts: &mut Vec<Part>, unit: usize, u: &Unit| {
        let mut at = 0;
        let mut ins = inserts.get(&unit).cloned().unwrap_or_default();
        ins.sort();
        for (byte, pi) in ins {
            if byte > at {
                parts.push(Part { file: Some((u.name.clone(), at)), text: u.text[at..byte].to_string() });
            }
            if !u.text[..byte].ends_with('\n') {
                gen(parts, "\n".into());
            }
            if !lines_for[pi].is_empty() {
                gen(parts, lines_for[pi].clone());
            }
            at = byte;
        }
        if at < u.text.len() {
            parts.push(Part { file: Some((u.name.clone(), at)), text: u.text[at..].to_string() });
        }
        gen(parts, "\n".into());
    };
    gen(&mut parts, format!("# `viv export` of {}: the decks, the program's files, then the checks\n", m.name));
    if !m.decks.is_empty() {
        gen(&mut parts, format!("{DECK_GLOBAL} = __export_num(0)\n"));
    }
    for (di, d) in decks.iter().enumerate() {
        gen(&mut parts, format!("if {DECK_GLOBAL} == {di}.0 {{\n"));
        cut(&mut parts, di, d);
        gen(&mut parts, "}\n".into());
    }
    for (fi, f) in files.iter().enumerate() {
        cut(&mut parts, 1000 + fi, f);
    }
    for (di, c) in checks.iter().enumerate() {
        if let Some(c) = c {
            gen(&mut parts, format!("if {DECK_GLOBAL} == {di}.0 {{\n"));
            cut(&mut parts, 5000 + di, c);
            gen(&mut parts, "}\n".into());
        }
    }

    // what each deck's run produces
    let produced = (0..ndecks)
        .map(|di| {
            m.results
                .iter()
                .map(|r| {
                    files.iter().any(|f| assigned_anywhere(&f.prog, &r.name))
                        || decks.get(di).is_some_and(|d| assigned_anywhere(&d.prog, &r.name))
                        || checks.get(di).and_then(|c| c.as_ref()).is_some_and(|c| assigned_anywhere(&c.prog, &r.name))
                })
                .collect()
        })
        .collect();
    // an enum parameter's default, by its label
    for (pi, p) in m.params.iter().enumerate() {
        if let Kind::Variant { ty, labels } = &p.kind {
            for d in defaults[pi].iter_mut().flatten() {
                if let Some((label, _)) = d.strip_prefix(&format!("{ty}.")).and_then(|v| labels.iter().find(|(_, w)| w == v)) {
                    *d = format!("{label:?}");
                }
            }
        }
    }
    // the glossary: a parameter's or result's own line first (the nearest
    // `#: NAME` at or above where its file assigns it), then every other line
    let mut gl = HashMap::new();
    let units: Vec<&Unit> = decks.iter().chain(files.iter()).chain(checks.iter().flatten()).collect();
    for name in m.params.iter().map(|p| &p.name).chain(m.results.iter().map(|r| &r.name)) {
        for u in &units {
            let mut out = Vec::new();
            for item in &u.prog.items {
                if let Item::Stmt(st) = item {
                    assigns(std::slice::from_ref(st), name, true, &mut out);
                }
            }
            let Some((st, _)) = out.first() else { continue };
            if let Some(text) = glossary_above(&u.text, name, line_of(&u.text, stmt_span(st).start as usize)) {
                gl.insert(name.clone(), text);
                break;
            }
        }
    }
    for u in &units {
        glossary(&u.text, &mut gl);
    }
    let deck_help = m
        .params
        .iter()
        .map(|p| {
            decks
                .iter()
                .map(|d| {
                    let mut out = Vec::new();
                    for item in &d.prog.items {
                        if let Item::Stmt(st) = item {
                            assigns(std::slice::from_ref(st), &p.name, true, &mut out);
                        }
                    }
                    out.first().and_then(|(st, _)| glossary_above(&d.text, &p.name, line_of(&d.text, stmt_span(st).start as usize)))
                })
                .collect()
        })
        .collect();
    Ok(Assembly { parts, slots, defaults, produced, glossary: gl, deck_help, variant_tags })
}

/// The source text of an expression, on one line.
fn expr_text(text: &str, e: &Expr) -> String {
    let sp = e.span();
    let s = text.get(sp.start as usize..sp.end as usize).unwrap_or("");
    // a contract's selection (`using_x = X.Method`) is written by the compiler
    // and carries the declaration's span (crate::contracts)
    if s.starts_with("contract ") {
        if let Some(c) = crate::contracts::scan(text).into_iter().find(|c| c.span.start == sp.start) {
            let v = match c.start {
                Some(k) => c.variant(k),
                None => crate::contracts::EVERY.to_string(),
            };
            return format!("{}.{v}", c.enum_name());
        }
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ------------------------------------------------------- the C interface

/// What the generated entry needs to know about the compiled program.
pub struct Compiled<'a> {
    /// the program's C, as the compiler wrote it
    pub c: &'a str,
    /// every global: name -> type
    pub globals: &'a HashMap<String, Ty>,
    /// "viv <commit>; program <hash>"
    pub version: String,
}

/// A result's field type, and its C name in the header.
fn result_field(kind: &Kind, prefix: &str) -> String {
    match kind {
        Kind::Num | Kind::Bool => "double".into(),
        Kind::Int => "int64_t".into(),
        Kind::Cplx => "double _Complex".into(),
        Kind::Str => "char*".into(),
        Kind::Arr => format!("{prefix}_array"),
        Kind::IntArr => format!("{prefix}_iarray"),
        Kind::Enum(_) | Kind::Variant { .. } => unreachable!("refused by parse_manifest"),
    }
}

fn param_field(kind: &Kind, prefix: &str) -> String {
    match kind {
        Kind::Num => "double".into(),
        Kind::Int => "int64_t".into(),
        Kind::Cplx => "double _Complex".into(),
        Kind::Bool | Kind::Enum(_) | Kind::Variant { .. } => "int".into(),
        Kind::Str => "const char*".into(),
        Kind::Arr => format!("{prefix}_arg"),
        Kind::IntArr => format!("{prefix}_iarg"),
    }
}

/// The C name the compiler gave a global: as written, or with `_` after it
/// when it is a C reserved word (knotkit's escape_ident). Found in the C.
fn c_global(c: &str, name: &str, ty: Ty) -> Option<String> {
    let ct = ty.c_type_name();
    for cand in [name.to_string(), format!("{name}_")] {
        if c.lines().any(|l| l == format!("{ct} {cand};")) {
            return Some(cand);
        }
    }
    None
}

/// The runtime state an exported run keeps from the run before: buffers it
/// reuses and caches of values that never change. Everything else of the
/// program's file-scope state is restored to what it was at load before each
/// run (the program's globals included), so a run depends only on its
/// parameters. Each entry says why.
pub const KEEP_ACROSS_RUNS: &[(&str, &str)] = &[
    ("viv_arena", "the arena's block list: the entry releases every block the run made"),
    ("viv_arena_top", "the arena's top: the entry releases to the top it found"),
    ("viv_arena_cap", "the arena list's capacity"),
    ("viv_kept", "a buffer reused by each loop release"),
    ("viv_kept_cap", "that buffer's capacity"),
    ("viv_fft_plans", "the FFT plan cache: plans by size, the same for every run"),
    ("viv_story_buf", "the story's buffer, reused"),
    ("viv_story_cap", "its capacity"),
    ("viv_ad_t", "the autodiff tape's buffer, reused"),
    ("viv_ad_cap", "its capacity"),
    ("viv_lits", "string literals, made once and never changed"),
    ("viv_fast_shadow_on", "a fast build's shadow log switch, read from the environment once"),
    ("viv_fast_shadow_f", "its file"),
];

/// The program's file-scope state to restore before each run: every mutable
/// file-scope variable of the C, less `KEEP_ACROSS_RUNS` and the export's
/// own (embed.c's `viv_embed_*`, the parameter tables).
pub fn per_run_state(c: &str, globals: &HashMap<String, Ty>) -> Vec<String> {
    let mut out = Vec::new();
    for line in c.lines() {
        if let Some((names, mutable)) = crate::lower::static_data_line(line) {
            for n in names {
                if mutable && !n.starts_with("viv_embed_") && !n.starts_with("viv_export_") && !KEEP_ACROSS_RUNS.iter().any(|(k, _)| *k == n) && !out.contains(&n) {
                    out.push(n);
                }
            }
        }
    }
    let mut names: Vec<&String> = globals.keys().collect();
    names.sort();
    for n in names {
        if let Some(cn) = c_global(c, n, globals[n]) {
            if !out.contains(&cn) {
                out.push(cn);
            }
        }
    }
    out
}

/// `pwdft.h`: the entry's declarations.
pub fn header(m: &Manifest, a: &Assembly, version: &str) -> String {
    let p = &m.name;
    let up = p.to_uppercase();
    let mut h = String::new();
    let _ = writeln!(h, "/* {p}.h -- written by `viv export`; do not edit.\n   {version} */");
    let _ = writeln!(h, "#ifndef {up}_H\n#define {up}_H\n#include <stddef.h>\n#include <stdint.h>\n");
    let _ = writeln!(h, "#define {up}_ABI 1\n");
    if !m.decks.is_empty() {
        let _ = writeln!(h, "/* {}: which deck the run uses */", m.deck_param);
        for (i, d) in m.decks.iter().enumerate() {
            let _ = writeln!(h, "#define {up}_{}_{} {i}", m.deck_param.to_uppercase(), d.name.to_uppercase());
        }
        h.push('\n');
    }
    for prm in &m.params {
        if let Some(vs) = prm.kind.choices() {
            match &prm.kind {
                Kind::Variant { ty, labels } => {
                    let pairs: Vec<String> = labels.iter().map(|(l, v)| format!("{l} = {ty}.{v}")).collect();
                    let _ = writeln!(h, "/* {}: {}\n   (each label, and the variant of the Viv source it chooses) */", prm.name.to_lowercase(), pairs.join(", ").replace("*/", "* /"));
                }
                _ => {
                    let _ = writeln!(h, "/* {}: */", prm.name.to_lowercase());
                }
            }
            for (i, v) in vs.iter().enumerate() {
                let _ = writeln!(h, "#define {up}_{}_{} {i}", prm.name.to_uppercase(), enum_ident(v));
            }
            h.push('\n');
        }
    }
    for (i, prm) in m.params.iter().enumerate() {
        let _ = writeln!(h, "#define {up}_GIVEN_{} (1ull << {i})", prm.name.to_uppercase());
    }
    let _ = writeln!(h, "\n/* an array argument, borrowed for the call: row-major, `dims[0..rank)` */");
    let _ = writeln!(h, "typedef struct {{ const double* data; int64_t rank; const int64_t* dims; }} {p}_arg;");
    let _ = writeln!(h, "typedef struct {{ const int64_t* data; int64_t rank; const int64_t* dims; }} {p}_iarg;");
    let _ = writeln!(h, "/* an array result, allocated by the library and freed by {p}_free */");
    let _ = writeln!(h, "typedef struct {{ double* data; int64_t rank; int64_t dims[8]; }} {p}_array;");
    let _ = writeln!(h, "typedef struct {{ int64_t* data; int64_t rank; int64_t dims[8]; }} {p}_iarray;");
    let _ = writeln!(h, "/* one `test` block of the program: its name, whether it passed, its first failure */");
    let _ = writeln!(h, "typedef struct {{ char* name; int passed; char* detail; }} {p}_test;");
    let _ = writeln!(h, "/* each printed line, without its newline; stream 1 is output, 2 a diagnostic */");
    let _ = writeln!(h, "typedef void (*{p}_line_fn)(const char* line, long n, int stream, void* ctx);\n");
    let _ = writeln!(h, "typedef struct {{");
    if !m.decks.is_empty() {
        let _ = writeln!(h, "    int {};", m.deck_param);
    }
    let _ = writeln!(h, "    uint64_t given;            /* {up}_GIVEN_* bits: the fields below the caller set */");
    for prm in &m.params {
        let doc = prm.doc.clone().or_else(|| a.glossary.get(&prm.name).cloned()).unwrap_or_default();
        let _ = writeln!(h, "    {} {};{}", param_field(&prm.kind, p), prm.name.to_lowercase(), if doc.is_empty() { String::new() } else { format!("  /* {} */", doc.replace("*/", "* /")) });
    }
    let _ = writeln!(h, "    {p}_line_fn on_line;         /* NULL drops the lines */");
    let _ = writeln!(h, "    void* ctx;");
    let _ = writeln!(h, "}} {p}_params;\n");
    let _ = writeln!(h, "typedef struct {{");
    let _ = writeln!(h, "    uint64_t have;             /* bit i: result i (in this order) was computed by this run */");
    for r in &m.results {
        let mut doc = r.doc.clone().or_else(|| a.glossary.get(&r.name).cloned()).unwrap_or_default();
        if let Some(rank) = &r.rank {
            doc = if doc.is_empty() { rank.text() } else { format!("{doc}; {}", rank.text()) };
        }
        let _ = writeln!(h, "    {} {};{}", result_field(&r.kind, p), r.name.to_lowercase(), if doc.is_empty() { String::new() } else { format!("  /* {} */", doc.replace("*/", "* /")) });
    }
    let _ = writeln!(h, "    int ntests;\n    {p}_test* tests;");
    let _ = writeln!(h, "    int error_kind;            /* with {up}_ERROR: one of {up}_E_* */");
    let _ = writeln!(h, "}} {p}_results;\n");
    let _ = writeln!(h, "enum {{ {up}_OK = 0, {up}_ERROR = 1, {up}_INTERRUPTED = 2, {up}_BAD_PARAM = 3, {up}_BUSY = 4 }};");
    let _ = writeln!(h, "enum {{ {up}_E_ERROR = 1, {up}_E_INDEX = 2, {up}_E_VALUE = 3, {up}_E_OVERFLOW = 4, {up}_E_ZERODIV = 5,\n       {up}_E_RECURSION = 6, {up}_E_MEMORY = 7, {up}_E_INTERRUPT = 8, {up}_E_STOP = 9 }};\n");
    let _ = writeln!(h, "/* Run the program once. 0 on success; otherwise msg (cap bytes) holds the\n   error with its source line. One run at a time per process: {up}_BUSY. */");
    let _ = writeln!(h, "int {p}_run(const {p}_params* p, {p}_results* r, char* msg, size_t cap);");
    let _ = writeln!(h, "/* free every block of r; safe on a zeroed r and twice */");
    let _ = writeln!(h, "void {p}_free({p}_results* r);");
    let _ = writeln!(h, "/* stop the run in progress at its next loop pass; from any thread */");
    let _ = writeln!(h, "void {p}_interrupt(void);");
    let _ = writeln!(h, "int {p}_abi(void);");
    let _ = writeln!(h, "const char* {p}_version(void);");
    let _ = writeln!(h, "#endif");
    h
}

/// An enum value as a C identifier: `6-31g*` -> `6_31G_STAR`.
pub(crate) fn enum_ident(v: &str) -> String {
    let mut s = String::new();
    for c in v.chars() {
        match c {
            '*' => s.push_str("_STAR"),
            '+' => s.push_str("_PLUS"),
            c if c.is_ascii_alphanumeric() => s.push(c.to_ascii_uppercase()),
            _ => s.push('_'),
        }
    }
    s
}

/// The C after the program: the parameter tables, the state restore, the
/// exported functions.
pub fn entry(m: &Manifest, a: &Assembly, comp: &Compiled) -> Result<String, String> {
    let p = &m.name;
    let up = p.to_uppercase();
    let c = comp.c;
    let has = |w: &str| c.contains(w);
    let mut e = String::new();
    let (nn, ni, ns, na) = table_sizes(&a.slots, !m.decks.is_empty());
    let _ = writeln!(e, "/* ---- the exported entry (`viv export`) ---- */");
    let _ = writeln!(e, "#undef main");
    let _ = writeln!(e, "#if defined(__GNUC__)\n#define {up}_EXPORT __attribute__((visibility(\"default\")))\n#else\n#define {up}_EXPORT\n#endif");
    let _ = writeln!(e, "static int {p}_busy = 0;");
    let export_rt = has("viv_export_given_v");
    if export_rt {
        let _ = writeln!(e, "static unsigned char {p}_given_t[{}];", m.params.len().max(1));
        let _ = writeln!(e, "static double {p}_num_t[{}];", nn.max(1));
        let _ = writeln!(e, "static long long {p}_int_t[{}];", ni.max(1));
        let _ = writeln!(e, "static const char* {p}_str_t[{}];", ns.max(1));
        let _ = writeln!(e, "static viv_export_arg {p}_arr_t[{}];", na.max(1));
    } else if !m.params.is_empty() || !m.decks.is_empty() {
        return Err("the program reads no parameter (internal: the export runtime is missing)".into());
    }

    // the state at load, restored before every run after the first
    let state = per_run_state(c, comp.globals);
    let _ = writeln!(e, "\n/* The program's state as it was at load: restored before each run, so a\n   run depends only on its parameters. Kept across runs instead:\n   viv export's KEEP_ACROSS_RUNS (lang/viv/src/export.rs). */");
    let _ = writeln!(e, "static int {p}_loaded = 0;");
    for (i, n) in state.iter().enumerate() {
        let _ = writeln!(e, "static unsigned char {p}_s{i}[sizeof({n})];");
    }
    let _ = writeln!(e, "static void {p}_restore(void) {{");
    let _ = writeln!(e, "    if (!{p}_loaded) {{");
    for (i, n) in state.iter().enumerate() {
        let _ = writeln!(e, "        memcpy({p}_s{i}, (const void*)&{n}, sizeof({n}));");
    }
    let _ = writeln!(e, "        {p}_loaded = 1;\n        return;\n    }}");
    for (i, n) in state.iter().enumerate() {
        let _ = writeln!(e, "    memcpy((void*)&{n}, {p}_s{i}, sizeof({n}));");
    }
    let _ = writeln!(e, "}}\n");

    // helpers
    let _ = writeln!(e, "static int {p}_bad(char* msg, size_t cap, const char* fmt, ...) {{\n    if (msg && cap) {{\n        va_list ap;\n        va_start(ap, fmt);\n        vsnprintf(msg, cap, fmt, ap);\n        va_end(ap);\n    }}\n    {p}_busy = 0;\n    return {up}_BAD_PARAM;\n}}");
    let _ = writeln!(e, "static int {p}_arg_ok(int64_t rank, const int64_t* dims, const void* data, long long dst[8], int want, char* why, size_t cap) {{\n    long long n = 1;\n    if (rank < 1 || rank > 8 || (want && rank != want)) {{ snprintf(why, cap, \"has rank %lld\", (long long)rank); return 0; }}\n    if (!dims) {{ snprintf(why, cap, \"has no dims\"); return 0; }}\n    for (int i = 0; i < rank; i++) {{\n        if (dims[i] < 0 || dims[i] > 2147483647LL) {{ snprintf(why, cap, \"has dimension %lld\", (long long)dims[i]); return 0; }}\n        n *= dims[i];\n        if (n > 2147483647LL) {{ snprintf(why, cap, \"has more than 2147483647 elements\"); return 0; }}\n        dst[i] = dims[i];\n    }}\n    if (n > 0 && !data) {{ snprintf(why, cap, \"has no data\"); return 0; }}\n    return 1;\n}}");
    let _ = writeln!(e, "static char* {p}_dup(const char* s, long n) {{\n    char* d = (char*)malloc((size_t)n + 1);\n    if (d) {{ memcpy(d, s, (size_t)n); d[n] = 0; }}\n    return d;\n}}");

    // the run
    let _ = writeln!(e, "\n{up}_EXPORT int {p}_abi(void) {{ return {up}_ABI; }}");
    let _ = writeln!(e, "{up}_EXPORT const char* {p}_version(void) {{ return {:?}; }}", comp.version);
    let _ = writeln!(e, "{up}_EXPORT void {p}_interrupt(void) {{ viv_embed_stop_flag = 1; }}");
    let _ = writeln!(e, "\n{up}_EXPORT int {p}_run(const {p}_params* p, {p}_results* r, char* msg, size_t cap) {{");
    let _ = writeln!(e, "    if (msg && cap) msg[0] = 0;\n    if (!p || !r) return {up}_BAD_PARAM;\n    memset(r, 0, sizeof *r);\n    if ({p}_busy) return {up}_BUSY;\n    {p}_busy = 1;");
    // checks of the parameters, before any work
    if !m.decks.is_empty() {
        let _ = writeln!(e, "    if (p->{dp} < 0 || p->{dp} >= {n}) return {p}_bad(msg, cap, \"{dp} is %d; there are {n} decks\", p->{dp});", dp = m.deck_param, n = m.decks.len());
        let _ = writeln!(e, "    {p}_num_t[0] = (double)p->{};", m.deck_param);
    }
    for (pi, prm) in m.params.iter().enumerate() {
        let f = prm.name.to_lowercase();
        let bit = format!("(p->given & ((uint64_t)1 << {pi}))");
        let _ = writeln!(e, "    {p}_given_t[{pi}] = {bit} != 0;");
        let _ = writeln!(e, "    if ({bit}) {{");
        let range = |e: &mut String, v: &str, fmt: &str| {
            if let Some(lo) = prm.min {
                let (op, words) = if prm.min_excl { (">", "above") } else { (">=", "at least") };
                let _ = writeln!(e, "        if (!({v} {op} {lo:?})) return {p}_bad(msg, cap, \"{f} is {fmt}; it must be {words} {}\", {v});", fmt_num(lo));
            }
            if let Some(hi) = prm.max {
                let (op, words) = if prm.max_excl { ("<", "below") } else { ("<=", "at most") };
                let _ = writeln!(e, "        if (!({v} {op} {hi:?})) return {p}_bad(msg, cap, \"{f} is {fmt}; it must be {words} {}\", {v});", fmt_num(hi));
            }
        };
        match (&prm.kind, a.slots[pi]) {
            (Kind::Num, Some(Slot::Num(k))) => {
                if !prm.nonfinite {
                    let _ = writeln!(e, "        if (!isfinite(p->{f})) return {p}_bad(msg, cap, \"{f} is %g; it must be a finite number\", p->{f});");
                }
                range(&mut e, &format!("p->{f}"), "%.17g");
                let _ = writeln!(e, "        {p}_num_t[{k}] = p->{f};");
            }
            (Kind::Int, Some(slot)) => {
                range(&mut e, &format!("(double)p->{f}"), "%.17g");
                match slot {
                    Slot::Num(k) => {
                        let _ = writeln!(e, "        if (p->{f} > 9007199254740992LL || p->{f} < -9007199254740992LL) return {p}_bad(msg, cap, \"{f} is %lld, past 2^53, and the program holds it as a number\", (long long)p->{f});");
                        let _ = writeln!(e, "        {p}_num_t[{k}] = (double)p->{f};");
                    }
                    Slot::Int(k) => {
                        let _ = writeln!(e, "        {p}_int_t[{k}] = (long long)p->{f};");
                    }
                    _ => unreachable!(),
                }
            }
            (Kind::Bool, Some(slot)) => {
                let _ = writeln!(e, "        if (p->{f} != 0 && p->{f} != 1) return {p}_bad(msg, cap, \"{f} is %d; a Bool is 0 or 1\", p->{f});");
                match slot {
                    Slot::Num(k) => {
                        let _ = writeln!(e, "        {p}_num_t[{k}] = (double)p->{f};");
                    }
                    Slot::Int(k) => {
                        let _ = writeln!(e, "        {p}_int_t[{k}] = p->{f};");
                    }
                    _ => unreachable!(),
                }
            }
            (Kind::Str, Some(Slot::Str(k))) => {
                let _ = writeln!(e, "        if (!p->{f}) return {p}_bad(msg, cap, \"{f} is NULL\");");
                let _ = writeln!(e, "        {p}_str_t[{k}] = p->{f};");
            }
            (Kind::Enum(_) | Kind::Variant { .. }, Some(slot)) => {
                let vs = prm.kind.choices().unwrap_or_default();
                let names: Vec<String> = vs.iter().map(|v| format!("{v:?}")).collect();
                let _ = writeln!(e, "        static const char* const names[] = {{ {} }};", names.join(", "));
                let _ = writeln!(e, "        if (p->{f} < 0 || p->{f} >= {n}) return {p}_bad(msg, cap, \"{f} is %d; there are {n} choices\", p->{f});", n = vs.len());
                match slot {
                    Slot::Num(k) => {
                        let _ = writeln!(e, "        {p}_num_t[{k}] = (double)p->{f};");
                    }
                    Slot::Int(k) => {
                        let _ = writeln!(e, "        {p}_int_t[{k}] = p->{f};");
                    }
                    Slot::Str(k) => {
                        let _ = writeln!(e, "        {p}_str_t[{k}] = names[p->{f}];");
                    }
                    _ => unreachable!(),
                }
                let _ = writeln!(e, "        (void)names;");
            }
            (Kind::Arr | Kind::IntArr, Some(Slot::Arr(k))) => {
                let want = prm.rank.unwrap_or(0);
                let _ = writeln!(e, "        char why[96];");
                let _ = writeln!(e, "        if (!{p}_arg_ok(p->{f}.rank, p->{f}.dims, (const void*)p->{f}.data, {p}_arr_t[{k}].dims, {want}, why, sizeof why)) return {p}_bad(msg, cap, \"{f} %s\", why);");
                if prm.kind == Kind::Arr && !prm.nonfinite {
                    let _ = writeln!(e, "        {{\n            long long n = 1;\n            for (int i = 0; i < p->{f}.rank; i++) n *= p->{f}.dims[i];\n            for (long long i = 0; i < n; i++) if (!isfinite(p->{f}.data[i])) return {p}_bad(msg, cap, \"{f}[%lld] (flat) is %g; it must hold finite numbers\", i, p->{f}.data[i]);\n        }}");
                }
                // each element within min and max
                if prm.min.is_some() || prm.max.is_some() {
                    let (lo, hi) = (prm.min.map_or("-1.0/0.0".to_string(), |x| format!("{x:?}")), prm.max.map_or("1.0/0.0".to_string(), |x| format!("{x:?}")));
                    let (lop, hip) = (if prm.min_excl { ">" } else { ">=" }, if prm.max_excl { "<" } else { "<=" });
                    let (lb, hb) = (if prm.min_excl { "(" } else { "[" }, if prm.max_excl { ")" } else { "]" });
                    let _ = writeln!(e, "        {{\n            long long n = 1;\n            for (int i = 0; i < p->{f}.rank; i++) n *= p->{f}.dims[i];\n            for (long long i = 0; i < n; i++) {{\n                double x = (double)p->{f}.data[i];\n                if (!(x {lop} {lo} && x {hip} {hi})) return {p}_bad(msg, cap, \"{f}[%lld] (flat) is %g; each element must be in {lb}%g, %g{hb}\", i, x, {lo}, {hi});\n            }}\n        }}");
                }
                for (i, d) in prm.shape.iter().flatten().enumerate() {
                    if let Dim::Fixed(n) = d {
                        let _ = writeln!(e, "        if (p->{f}.dims[{i}] != {n}) return {p}_bad(msg, cap, \"{f} has %lld in dimension {i}; it must have {n}\", (long long)p->{f}.dims[{i}]);");
                    }
                }
                let _ = writeln!(e, "        {p}_arr_t[{k}].data = (const void*)p->{f}.data;\n        {p}_arr_t[{k}].rank = (int)p->{f}.rank;");
            }
            _ => return Err(format!("internal: no slot for the parameter `{}`", prm.name)),
        }
        let _ = writeln!(e, "    }}");
    }
    for prm in &m.params {
        let f = prm.name.to_lowercase();
        let bit = |name: &str| m.params.iter().position(|q| q.name == name).unwrap_or(0);
        for (i, d) in prm.shape.iter().flatten().enumerate() {
            if let Dim::LenOf(other) = d {
                let (o, ob) = (other.to_lowercase(), bit(other));
                let _ = writeln!(e, "    if ((p->given & ((uint64_t)1 << {})) && (p->given & ((uint64_t)1 << {ob})) && p->{f}.dims[{i}] != p->{o}.dims[0])\n        return {p}_bad(msg, cap, \"{f} has %lld in dimension {i}; it must have len({o}) = %lld\", (long long)p->{f}.dims[{i}], (long long)p->{o}.dims[0]);", bit(&prm.name));
            }
        }
    }
    for (g, names) in &m.groups {
        let mask: u64 = names.iter().map(|n| 1u64 << m.params.iter().position(|q| q.name == *n).unwrap_or(0)).fold(0, |a, b| a | b);
        let low: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        let _ = writeln!(e, "    if ((p->given & {mask}ull) != 0 && (p->given & {mask}ull) != {mask}ull)\n        return {p}_bad(msg, cap, \"{} are given together or not at all (group {g})\");", low.join(", "));
    }
    if export_rt {
        let _ = writeln!(e, "    viv_export_given_v = {p}_given_t;\n    viv_export_num_v = {p}_num_t;\n    viv_export_int_v = {p}_int_t;\n    viv_export_str_v = {p}_str_t;\n    viv_export_arr_v = {p}_arr_t;");
    }
    let _ = writeln!(e, "    {p}_restore();");
    let arena = has("viv_arena_top") && has("static void viv_arena_release(");
    if arena {
        let _ = writeln!(e, "    size_t mark = viv_arena_top;");
    }
    let _ = writeln!(e, "    viv_embed_stop_flag = 0;");
    let _ = writeln!(e, "    int code = viv_embed_run(viv_program_main, p->on_line, p->ctx, msg, cap);");
    // results
    let _ = writeln!(e, "    if (code == 0) {{");
    if !m.decks.is_empty() {
        let _ = writeln!(e, "        switch (p->{}) {{", m.deck_param);
        for (di, row) in a.produced.iter().enumerate() {
            let mut bits: u64 = 0;
            for (ri, &yes) in row.iter().enumerate() {
                if yes {
                    bits |= 1 << ri;
                }
            }
            let _ = writeln!(e, "        case {di}: r->have = {bits}ull; break;");
        }
        let _ = writeln!(e, "        }}");
    } else {
        let mut bits: u64 = 0;
        for (ri, &yes) in a.produced.first().map(|v| v.as_slice()).unwrap_or(&[]).iter().enumerate() {
            if yes {
                bits |= 1 << ri;
            }
        }
        let _ = writeln!(e, "        r->have = {bits}ull;");
    }
    for (ri, res) in m.results.iter().enumerate() {
        let Some(&ty) = comp.globals.get(&res.name) else {
            return Err(format!("error: the result `{}` is not a global of the program\n", res.name));
        };
        let ok = matches!((&res.kind, ty), (Kind::Num, Ty::Num) | (Kind::Bool, Ty::Bool) | (Kind::Bool, Ty::Num) | (Kind::Int, Ty::Int) | (Kind::Int, Ty::Num) | (Kind::Str, Ty::Str) | (Kind::Arr, Ty::Arr(Elem::Num)) | (Kind::IntArr, Ty::Arr(Elem::Int)));
        if !ok {
            return Err(format!("error: the result `{}` is {} in the program, and the manifest says {}\n", res.name, type_word(ty), res.kind.word()));
        }
        let Some(g) = c_global(c, &res.name, ty) else { return Err(format!("internal: the global `{}` was not found in the C", res.name)) };
        let f = res.name.to_lowercase();
        let _ = writeln!(e, "        if (r->have & ((uint64_t)1 << {ri})) {{");
        match (&res.kind, ty) {
            (Kind::Int, Ty::Num) => {
                let _ = writeln!(e, "            r->{f} = (int64_t){g};");
            }
            (Kind::Num | Kind::Bool | Kind::Int, _) => {
                let _ = writeln!(e, "            r->{f} = {g};");
            }
            (Kind::Str, _) => {
                let _ = writeln!(e, "            r->{f} = {g} ? {p}_dup({g}->s, {g}->len) : {p}_dup(\"\", 0);");
            }
            (Kind::Arr | Kind::IntArr, _) => {
                let elem = if res.kind == Kind::Arr { "double" } else { "int64_t" };
                let _ = writeln!(e, "            r->{f}.rank = {g}.rank;");
                let _ = writeln!(e, "            for (int i = 0; i < {g}.rank && i < 8; i++) r->{f}.dims[i] = {g}.dims[i];");
                let _ = writeln!(e, "            r->{f}.data = ({elem}*)malloc((size_t)({g}.size > 0 ? {g}.size : 1) * sizeof({elem}));");
                let _ = writeln!(e, "            if (r->{f}.data && {g}.size > 0) memcpy(r->{f}.data, {g}.data, (size_t){g}.size * sizeof({elem}));");
            }
            _ => unreachable!(),
        }
        let _ = writeln!(e, "        }}");
    }
    // each array result of a stated rank has it
    for (ri, res) in m.results.iter().enumerate() {
        let Some(rank) = &res.rank else { continue };
        let f = res.name.to_lowercase();
        let fail = |e: &mut String, want: &str, when: &str| {
            let _ = writeln!(e, "            if (r->{f}.rank != {want}) {{\n                if (msg && cap) snprintf(msg, cap, \"error: the result {f} has rank %lld; the manifest says %d{when}\", (long long)r->{f}.rank, (int)({want}));\n                code = VIV_E_ERROR;\n            }}");
        };
        let _ = writeln!(e, "        if (code == 0 && (r->have & ((uint64_t)1 << {ri}))) {{");
        match rank {
            Rank::Fixed(n) => fail(&mut e, &n.to_string(), ""),
            Rank::By { param, ranks } => {
                let pi = m.params.iter().position(|q| q.name == *param).expect("checked by parse_manifest");
                let prm = &m.params[pi];
                let choices = prm.kind.choices().unwrap_or_default();
                let Some(&gty) = comp.globals.get(param) else { return Err(format!("error: the parameter `{param}` is not a global of the program\n")) };
                let Some(g) = c_global(c, param, gty) else { return Err(format!("internal: the global `{param}` was not found in the C")) };
                // the case each choice is: a variant's tag, or the global's index
                let key = match (&prm.kind, gty) {
                    (Kind::Variant { .. }, Ty::Enum(_)) => Some(format!("{g}->tag")),
                    (Kind::Enum(_), Ty::Num | Ty::Int) => Some(format!("(int){g}")),
                    _ => None,
                };
                if let Some(key) = key {
                    let _ = writeln!(e, "            int want = 0;\n            const char* when = \"\";\n            switch ({key}) {{");
                    for (li, label) in choices.iter().enumerate() {
                        let case = match &prm.kind {
                            Kind::Variant { .. } => a.variant_tags[pi][li],
                            _ => li,
                        };
                        let r = ranks.iter().find(|(l, _)| l == label).map(|(_, r)| *r).unwrap_or(0);
                        let _ = writeln!(e, "            case {case}: want = {r}; when = {:?}; break;", format!(" when {} is {label}", param.to_lowercase()));
                    }
                    let _ = writeln!(e, "            default: break;\n            }}");
                    let _ = writeln!(e, "            if (want && r->{f}.rank != want) {{\n                if (msg && cap) snprintf(msg, cap, \"error: the result {f} has rank %lld; the manifest says %d%s\", (long long)r->{f}.rank, want, when);\n                code = VIV_E_ERROR;\n            }}");
                }
            }
        }
        let _ = writeln!(e, "        }}");
    }
    let _ = writeln!(e, "    }}");
    let _ = writeln!(e, "    {{\n        int n = 0;\n        viv_embed_test_row* t = viv_embed_take_tests(&n);\n        if (n > 0) {{\n            r->tests = ({p}_test*)malloc((size_t)n * sizeof({p}_test));\n            if (r->tests) {{\n                for (int i = 0; i < n; i++) {{ r->tests[i].name = t[i].name; r->tests[i].passed = t[i].passed; r->tests[i].detail = t[i].detail; }}\n                r->ntests = n;\n                free(t);\n            }} else viv_embed_free_tests(t, n);\n        }} else free(t);\n    }}");
    if arena {
        let _ = writeln!(e, "    viv_arena_release(mark, 0);");
    }
    let _ = writeln!(e, "    {p}_busy = 0;");
    let _ = writeln!(e, "    if (code == 0) return {up}_OK;\n    r->error_kind = code;\n    return code == VIV_E_INTERRUPT ? {up}_INTERRUPTED : {up}_ERROR;\n}}");
    // free
    let _ = writeln!(e, "\n{up}_EXPORT void {p}_free({p}_results* r) {{\n    if (!r) return;");
    for res in &m.results {
        let f = res.name.to_lowercase();
        match res.kind {
            Kind::Str => {
                let _ = writeln!(e, "    free(r->{f});\n    r->{f} = 0;");
            }
            Kind::Arr | Kind::IntArr => {
                let _ = writeln!(e, "    free(r->{f}.data);\n    r->{f}.data = 0;");
            }
            _ => {}
        }
    }
    let _ = writeln!(e, "    for (int i = 0; i < r->ntests; i++) {{ free(r->tests[i].name); free(r->tests[i].detail); }}\n    free(r->tests);\n    r->tests = 0;\n    r->ntests = 0;\n}}");
    Ok(e)
}

/// A parameter's range in words, for the docstring: ", from 0 to 1",
/// ", above 0", ", above 0 and at most 1" (`each` says "each element ").
pub(crate) fn range_words(prm: &Param, each: &str) -> String {
    let lo = prm.min.map(|x| if prm.min_excl { format!("above {}", fmt_num(x)) } else { format!("at least {}", fmt_num(x)) });
    let hi = prm.max.map(|x| if prm.max_excl { format!("below {}", fmt_num(x)) } else { format!("at most {}", fmt_num(x)) });
    match (prm.min, prm.max, lo, hi) {
        (Some(a), Some(b), _, _) if !prm.min_excl && !prm.max_excl => format!(", {each}from {} to {}", fmt_num(a), fmt_num(b)),
        (_, _, Some(l), Some(h)) => format!(", {each}{l} and {h}"),
        (_, _, Some(l), None) => format!(", {each}{l}"),
        (_, _, None, Some(h)) => format!(", {each}{h}"),
        _ => String::new(),
    }
}

pub(crate) fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// The whole `pwdft.c`: the header, the program, the entry.
pub fn c_file(m: &Manifest, a: &Assembly, comp: &Compiled) -> Result<String, String> {
    let p = &m.name;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "/* {p}.c -- written by `viv export`; do not edit.\n   {}\n\n   One file, C99 with GNU attributes, the C library only. Build it:\n\n       cc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}.so -lm\n\n   (exact: the interpreter's bits; README.md has the fast build). The\n   interface is {p}.h, pasted below. */",
        comp.version
    );
    let _ = writeln!(out, "#ifndef VIV_EMBED\n#define VIV_EMBED 1\n#endif");
    out.push_str(&header(m, a, &comp.version));
    out.push_str("#include <stdarg.h>\n#include <string.h>\n#include <stdlib.h>\n#include <math.h>\n");
    out.push('\n');
    out.push_str(comp.c);
    out.push('\n');
    out.push_str(&entry(m, a, comp)?);
    Ok(out)
}

// ------------------------------------------------------------- Python

pub(crate) fn py_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// The first paragraph of a program's `###` text block, after its header
/// lines (`post: ...`, `agent: ...`), for the docstring.
pub fn fence_text(src: &str) -> String {
    let mut lines = src.lines().skip_while(|l| l.trim() != "###");
    lines.next();
    let mut out = Vec::new();
    let mut header = true;
    for l in lines {
        let t = l.trim();
        if t == "###" {
            break;
        }
        if header {
            // a key is one word or two (`scale large:`), as www/src/lib/postCheck.js HEADER_FIELD
            let word = |w: &str| !w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
            let key = t.split_once(':').map(|(k, _)| k);
            if key.is_some_and(|k| {
                let mut ws = k.split(' ');
                let first = ws.next().unwrap_or("");
                let second = ws.next();
                ws.next().is_none() && word(first) && first.starts_with(|c: char| c.is_ascii_lowercase()) && second.map_or(true, word)
            }) || t.is_empty()
            {
                continue;
            }
            header = false;
        }
        if t.is_empty() {
            break;
        }
        out.push(t.to_string());
    }
    out.join(" ")
}

/// The text of the `module` paragraph of an intent file, for the docstring.
pub fn intent_text(intent: &str) -> String {
    let mut out = Vec::new();
    let mut in_module = false;
    for l in intent.lines() {
        if l.starts_with("module ") {
            in_module = true;
            continue;
        }
        if in_module {
            if !l.starts_with(' ') && !l.trim().is_empty() {
                break;
            }
            if l.trim().is_empty() && !out.is_empty() {
                break;
            }
            if !l.trim().is_empty() {
                out.push(l.trim().to_string());
            }
        }
    }
    out.join(" ")
}

pub(crate) fn wrap(text: &str, width: usize, indent: &str) -> String {
    let mut out = String::new();
    let mut line = String::new();
    for w in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + w.len() > width {
            let _ = writeln!(out, "{indent}{line}");
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(w);
    }
    if !line.is_empty() {
        let _ = writeln!(out, "{indent}{line}");
    }
    out
}

/// `pwdft.py`: the ctypes wrapper.
pub fn python(m: &Manifest, a: &Assembly, about: &str, version: &str) -> String {
    let p = &m.name;
    let up = p.to_uppercase();
    let mut s = String::new();
    // the docstring of run()
    let mut doc = String::new();
    if !about.is_empty() {
        doc.push_str(&wrap(about, 72, "    "));
        doc.push('\n');
    }
    if m.decks.is_empty() {
        doc.push_str("    Parameters (keyword only; None keeps the program's value):\n\n");
    } else {
        doc.push_str("    Parameters (after the first, keyword only; None keeps the deck's value):\n\n");
    }
    if !m.decks.is_empty() {
        let names: Vec<&str> = m.decks.iter().map(|d| d.name.as_str()).collect();
        let _ = writeln!(doc, "    {}: one of {}", m.deck_param, names.join(", "));
    }
    for (pi, prm) in m.params.iter().enumerate() {
        let what = prm.doc.clone().or_else(|| a.glossary.get(&prm.name).cloned()).unwrap_or_default();
        let ty = match &prm.kind {
            Kind::Num => "float".to_string(),
            Kind::Int => "int".to_string(),
            Kind::Cplx => "complex".to_string(),
            Kind::Bool => "bool".to_string(),
            Kind::Str => "str".to_string(),
            Kind::Enum(vs) => format!("one of {}", vs.join(", ")),
            Kind::Variant { labels, .. } => format!("one of {}", labels.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join(", ")),
            Kind::Arr => match prm.rank {
                Some(r) => format!("float64 array of rank {r}"),
                None => "float64 array".to_string(),
            },
            Kind::IntArr => "int64 array".to_string(),
        };
        let mut range = String::new();
        let each = if matches!(prm.kind, Kind::Arr | Kind::IntArr) { "each element " } else { "" };
        range.push_str(&range_words(prm, each));
        let _ = writeln!(doc, "    {} ({ty}{range}): {what}", prm.name.to_lowercase());
        match prm.normalize.as_deref() {
            Some("pyscf_basis") => {
                let _ = writeln!(doc, "        spelled as PySCF reads a basis name: any case, `-`, `_` and spaces ignored, `*` read as `s`");
            }
            Some(_) => {
                let _ = writeln!(doc, "        in any case");
            }
            None => {}
        }
        let shown = |v: &String| -> String {
            match (&prm.kind, v.parse::<f64>()) {
                (Kind::Int, Ok(x)) if x.fract() == 0.0 => format!("{}", x as i64),
                _ => v.clone(),
            }
        };
        let defaults: Vec<String> = if m.decks.is_empty() {
            a.defaults[pi].iter().flatten().map(shown).collect()
        } else {
            m.decks.iter().zip(&a.defaults[pi]).filter_map(|(d, v)| v.as_ref().map(|v| format!("{} {}", d.name, shown(v)))).collect()
        };
        if !defaults.is_empty() {
            let _ = writeln!(doc, "        default: {}", defaults.join(", "));
        }
        // a deck that describes the parameter in its own words
        if prm.doc.is_none() {
            for (d, h) in m.decks.iter().zip(a.deck_help.get(pi).into_iter().flatten()) {
                if let Some(h) = h.as_ref().filter(|h| **h != what) {
                    let _ = writeln!(doc, "        in {}: {h}", d.name);
                }
            }
        }
    }
    for (_, names) in &m.groups {
        let low: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        let _ = writeln!(doc, "    {} are given together or not at all.", low.join(", "));
    }
    for prm in &m.params {
        if let Some(sh) = &prm.shape {
            let dims: Vec<String> = sh.iter().map(|d| match d { Dim::Any => "any".to_string(), Dim::Fixed(n) => n.to_string(), Dim::LenOf(o) => format!("len({})", o.to_lowercase()) }).collect();
            let _ = writeln!(doc, "    {} has shape ({}).", prm.name.to_lowercase(), dims.join(", "));
        }
    }
    let rs: Vec<String> = m.results.iter().map(|r| r.name.to_lowercase()).collect();
    doc.push('\n');
    doc.push_str(&wrap(&format!("log: a list each printed line is appended to; on_line: called with each line. Returns a dict: {} (None when this run does not compute it), tests (the program's test blocks: (name, passed, detail)) and log (the printed lines).", rs.join(", ")), 72, "    "));
    doc.push('\n');
    for r in &m.results {
        let what = r.doc.clone().or_else(|| a.glossary.get(&r.name).cloned());
        let rank = r.rank.as_ref().map(Rank::text);
        let line = match (what, rank) {
            (Some(w), Some(k)) => format!("{w} ({k})"),
            (Some(w), None) => w,
            (None, Some(k)) => k,
            (None, None) => continue,
        };
        let _ = writeln!(doc, "    {}: {line}", r.name.to_lowercase());
    }
    let _ = writeln!(doc, "\n    {version}");

    let _ = writeln!(s, "\"\"\"{p}: a Viv program, exported by `viv export`.\n");
    let _ = writeln!(s, "    import {p}\n    r = {p}.run({})\n", if m.decks.is_empty() { String::new() } else { format!("{}={:?}", m.deck_param, m.decks[0].name) });
    let _ = writeln!(s, "help({p}.run) lists the parameters. The library lib{p}.so (.dylib, .dll) is built\nfrom {p}.c beside this file on first import when it is missing or older than\n{p}.c, with $CC (default cc):\n\n    cc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}.so -lm\n\nSet {up}_FAST=1 for the fast build (fused multiply-adds and the platform's libm:\nlast-bit differences from the exact build), {up}_LIB to load a library\nbuilt elsewhere. The printed lines go to the logger \"{p}\" at INFO.\n\nWritten by `viv export`; do not edit. {version}\n\"\"\"");
    s.push_str(PY_BODY_1);
    let _ = writeln!(s, "\nNAME = {:?}\nVERSION = {}\nABI = 1", p, py_str(version));
    let _ = writeln!(s, "DECK_PARAM = {:?}", m.deck_param);
    let decks: Vec<String> = m.decks.iter().map(|d| format!("{:?}", d.name)).collect();
    let _ = writeln!(s, "DECKS = [{}]", decks.join(", "));
    // params: (python name, kind, min, max, rank, values)
    let _ = writeln!(s, "PARAMS = [");
    for prm in &m.params {
        let values = match prm.kind.choices() {
            Some(vs) => format!("[{}]", vs.iter().map(|v| py_str(v)).collect::<Vec<_>>().join(", ")),
            None => "None".into(),
        };
        let num = |x: Option<f64>| x.map_or("None".to_string(), |v| format!("{v:?}"));
        let norm = prm.normalize.as_ref().map_or("None".to_string(), |r| format!("{r:?}"));
        let tf = |b: bool| if b { "True" } else { "False" };
        let _ = writeln!(s, "    ({:?}, {:?}, {}, {}, {}, {}, {}, {norm}, {}, {}),", prm.name.to_lowercase(), prm.kind.word(), num(prm.min), num(prm.max), prm.rank.map_or("None".to_string(), |r| r.to_string()), values, tf(prm.nonfinite), tf(prm.min_excl), tf(prm.max_excl));
    }
    let _ = writeln!(s, "]");
    let _ = writeln!(s, "RESULTS = [");
    for r in &m.results {
        let _ = writeln!(s, "    ({:?}, {:?}),", r.name.to_lowercase(), r.kind.word());
    }
    let _ = writeln!(s, "]");
    s.push_str(PY_LOAD);
    s.push_str(PY_PROGRAM_BIND);
    s.push_str(PY_COMMON);
    s.push_str(PY_RUN);
    let names: Vec<String> = m.params.iter().map(|p| p.name.to_lowercase()).collect();
    let kws: Vec<String> = names.iter().map(|n| format!("{n}=None")).collect();
    let _ = writeln!(
        s,
        "\n\ndef run({}*, {}{}log=None, on_line=None):",
        if m.decks.is_empty() { String::new() } else { format!("{}={:?}, ", m.deck_param, m.decks[0].name) },
        kws.join(", "),
        if kws.is_empty() { "" } else { ", " }
    );
    let _ = write!(s, "    \"\"\"");
    let _ = write!(s, "{}", doc.trim_start());
    let _ = writeln!(s, "    \"\"\"");
    let dict: Vec<String> = names.iter().map(|n| format!("{n:?}: {n}")).collect();
    let _ = writeln!(s, "    return _run({}, {{{}}}, log, on_line)", if m.decks.is_empty() { "None".to_string() } else { m.deck_param.clone() }, dict.join(", "));
    s
}

pub(crate) const PY_BODY_1: &str = r#"
import ctypes as _C
import logging as _logging
import os as _os
import subprocess as _subprocess
import sys as _sys
import threading as _threading

if _sys.version_info < (3, 9):
    raise ImportError("a module written by `viv export` needs Python 3.9 or newer")

import numpy as _np
"#;

pub(crate) const PY_LOAD: &str = r#"
_log = _logging.getLogger(NAME)
_here = _os.path.dirname(_os.path.abspath(__file__))
_src = _os.path.join(_here, NAME + ".c")
_ext = ".dll" if _sys.platform == "win32" else ".dylib" if _sys.platform == "darwin" else ".so"
_FAST = _os.environ.get(NAME.upper() + "_FAST") == "1"
_lib_path = _os.environ.get(NAME.upper() + "_LIB") or _os.path.join(_here, "lib" + NAME + ("_fast" if _FAST else "") + _ext)

EXACT_FLAGS = ["-O2", "-ffp-contract=off"]
FAST_FLAGS = ["-O3", "-DVIV_FAST", "-ffp-contract=fast"]
COMMON_FLAGS = ["-fvisibility=hidden", "-shared", "-fPIC"]


def build(path=None, fast=False, cc=None):
    """Compile NAME.c into a shared library at `path` (default: beside this
    file) with the exact flags, or the fast ones; returns the path."""
    path = path or _lib_path
    cc = cc or _os.environ.get("CC", "cc")
    cmd = [cc] + (FAST_FLAGS if fast else EXACT_FLAGS) + COMMON_FLAGS + [_src, "-o", path, "-lm"]
    try:
        out = _subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise ImportError("%s: no C compiler (%s); build the library with:\n    %s" % (NAME, e, " ".join(cmd)))
    if out.returncode != 0:
        raise ImportError("%s: building the library failed:\n    %s\n%s" % (NAME, " ".join(cmd), out.stderr))
    return path


def _stale():
    if not _os.path.exists(_lib_path):
        return True
    return _os.path.exists(_src) and _os.path.getmtime(_src) > _os.path.getmtime(_lib_path)


if not _os.environ.get(NAME.upper() + "_LIB") and _stale():
    build(_lib_path, fast=_FAST)

_LINE = _C.CFUNCTYPE(None, _C.c_char_p, _C.c_long, _C.c_int, _C.c_void_p)


class _Arg(_C.Structure):
    _fields_ = [("data", _C.c_void_p), ("rank", _C.c_int64), ("dims", _C.POINTER(_C.c_int64))]


class _Array(_C.Structure):
    _fields_ = [("data", _C.c_void_p), ("rank", _C.c_int64), ("dims", _C.c_int64 * 8)]


class _Test(_C.Structure):
    _fields_ = [("name", _C.c_char_p), ("passed", _C.c_int), ("detail", _C.c_char_p)]


_CT = {"Num": _C.c_double, "Int": _C.c_int64, "Bool": _C.c_int, "Str": _C.c_char_p, "Enum": _C.c_int, "Arr": _Arg, "IntArr": _Arg}
_RT = {"Num": _C.c_double, "Int": _C.c_int64, "Bool": _C.c_double, "Str": _C.c_void_p, "Arr": _Array, "IntArr": _Array}
"#;

/// A program's structs and entry points.
const PY_PROGRAM_BIND: &str = r#"

class _Params(_C.Structure):
    _fields_ = ([(DECK_PARAM, _C.c_int)] if DECKS else []) + [("given", _C.c_uint64)] + \
        [(p[0], _CT[p[1]]) for p in PARAMS] + [("on_line", _LINE), ("ctx", _C.c_void_p)]


class _Results(_C.Structure):
    _fields_ = [("have", _C.c_uint64)] + [(n, _RT[k]) for (n, k) in RESULTS] + \
        [("ntests", _C.c_int), ("tests", _C.POINTER(_Test)), ("error_kind", _C.c_int)]


_lib = _C.CDLL(_lib_path)
_run_c = getattr(_lib, NAME + "_run")
_run_c.argtypes = [_C.POINTER(_Params), _C.POINTER(_Results), _C.c_char_p, _C.c_size_t]
_run_c.restype = _C.c_int
_free_c = getattr(_lib, NAME + "_free")
_free_c.argtypes = [_C.POINTER(_Results)]
_interrupt_c = getattr(_lib, NAME + "_interrupt")
_interrupt_c.argtypes = []
getattr(_lib, NAME + "_version").restype = _C.c_char_p
if getattr(_lib, NAME + "_abi")() != ABI:
    raise ImportError("%s: %s was built from another version of %s.c (ABI %d, this wrapper %d); delete it to rebuild"
                      % (NAME, _lib_path, NAME, getattr(_lib, NAME + "_abi")(), ABI))


def version():
    """The compiler commit and program hash the library was generated from."""
    return getattr(_lib, NAME + "_version")().decode()
"#;

/// The errors, the parameter checks and the array copy, for programs and kernels alike.
pub(crate) const PY_COMMON: &str = r#"

class VivError(RuntimeError):
    """A run-time error inside the program. The message is Viv's, with the
    source line. `kind` names the error; the subclasses below are also the
    matching Python exception."""
    kind = "error"


class VivIndexError(VivError, IndexError):
    kind = "index"


class VivValueError(VivError, ValueError):
    kind = "value"


class VivOverflowError(VivError, OverflowError):
    kind = "overflow"


class VivZeroDivisionError(VivError, ZeroDivisionError):
    kind = "zerodiv"


class VivRecursionError(VivError, RecursionError):
    kind = "recursion"


class VivMemoryError(VivError, MemoryError):
    kind = "memory"


class VivStop(VivError):
    """The program stopped itself (`stop(...)` or a check it carries)."""
    kind = "stop"


_KINDS = {1: VivError, 2: VivIndexError, 3: VivValueError, 4: VivOverflowError, 5: VivZeroDivisionError,
          6: VivRecursionError, 7: VivMemoryError, 9: VivStop}
_INTEGRAL = (int, _np.integer)


def normalize(rule, s):
    """A caller's spelling of an enum's name as the comparison reads it:
    `lower` ignores case; `pyscf_basis` is PySCF's rule for basis names (lower
    case, without -, _ and spaces) with * read as s, so "6-31G*", "631g*" and
    "631gs" are one name."""
    low = s.lower()
    if rule == "pyscf_basis":
        return "".join("s" if c == "*" else c for c in low if c not in "-_ ")
    return low


def _check(name, kind, lo, hi, rank, values, nonfinite, v, norm=None, lo_excl=False, hi_excl=False):
    """The parameter as C takes it; TypeError or ValueError before any work."""
    if kind == "Num":
        if isinstance(v, bool) or not isinstance(v, (int, float, _np.integer, _np.floating)):
            raise TypeError("%s must be a number, not %r" % (name, v))
        v = float(v)
        if not nonfinite and not _np.isfinite(v):
            raise ValueError("%s is %r; it must be a finite number" % (name, v))
    elif kind == "Int":
        if isinstance(v, bool):
            raise TypeError("%s must be an int, not %r" % (name, v))
        if isinstance(v, (float, _np.floating)):
            if not float(v).is_integer():
                raise ValueError("%s must be a whole number, not %r" % (name, v))
            v = int(v)
        if not isinstance(v, _INTEGRAL):
            raise TypeError("%s must be an int, not %r" % (name, v))
        v = int(v)
        if not -2 ** 63 <= v < 2 ** 63:
            raise OverflowError("%s does not fit in 64 bits" % name)
    elif kind == "Bool":
        if not isinstance(v, (bool, _np.bool_)):
            raise TypeError("%s must be True or False, not %r" % (name, v))
        return int(bool(v))
    elif kind == "Cplx":
        if isinstance(v, (bool, _np.bool_)) or not isinstance(v, (int, float, complex, _np.number)):
            raise TypeError("%s must be a number, not %r" % (name, v))
        v = complex(v)
        if not nonfinite and not (_np.isfinite(v.real) and _np.isfinite(v.imag)):
            raise ValueError("%s is %r; it must be finite" % (name, v))
        return v
    elif kind == "Str":
        if not isinstance(v, str):
            raise TypeError("%s must be a str, not %r" % (name, v))
        if "\0" in v:
            raise ValueError("%s holds a NUL character" % name)
        return v.encode()
    elif kind == "Enum":
        if not isinstance(v, str):
            raise TypeError("%s must be a str, one of %s, not %r" % (name, ", ".join(values), v))
        keys = [normalize(norm, x) for x in values] if norm else values
        k = normalize(norm, v) if norm else v
        if k not in keys:
            raise ValueError("%s must be one of %s, not %r" % (name, ", ".join(values), v))
        return keys.index(k)
    elif kind in ("Arr", "IntArr"):
        a = _np.ascontiguousarray(v, dtype=_np.float64 if kind == "Arr" else _np.int64)
        if kind == "IntArr" and _np.asarray(v).dtype.kind == "f" and not _np.all(_np.asarray(v) == a):
            raise ValueError("%s must hold whole numbers" % name)
        if a.ndim == 0 or a.ndim > 8 or (rank is not None and a.ndim != rank):
            raise ValueError("%s must be an array of rank %s, not %d" % (name, rank or "1 to 8", a.ndim))
        if a.size >= 2 ** 31:
            raise ValueError("%s has more than 2147483647 elements" % name)
        if kind == "Arr" and not nonfinite and not _np.all(_np.isfinite(a)):
            raise ValueError("%s holds a NaN or an infinity" % name)
        if a.size and lo is not None and not _np.all(a > lo if lo_excl else a >= lo):
            raise ValueError("%s: each element must be %s %r" % (name, "above" if lo_excl else "at least", lo))
        if a.size and hi is not None and not _np.all(a < hi if hi_excl else a <= hi):
            raise ValueError("%s: each element must be %s %r" % (name, "below" if hi_excl else "at most", hi))
        return a
    if lo is not None and not (v > lo if lo_excl else v >= lo):
        raise ValueError("%s is %r; it must be %s %r" % (name, v, "above" if lo_excl else "at least", lo))
    if hi is not None and not (v < hi if hi_excl else v <= hi):
        raise ValueError("%s is %r; it must be %s %r" % (name, v, "below" if hi_excl else "at most", hi))
    return v


def _array(a, ctype, dtype):
    shape = tuple(a.dims[i] for i in range(a.rank))
    n = 1
    for d in shape:
        n *= d
    if n == 0 or not a.data:
        return _np.zeros(shape, dtype=dtype)
    buf = _C.cast(a.data, _C.POINTER(ctype))
    return _np.ctypeslib.as_array(buf, shape=(n,)).copy().reshape(shape)
"#;

/// A program's run.
const PY_RUN: &str = r#"

def _run(deck, params, log, on_line):
    p = _Params()
    keep = []                        # what the C borrows during the call
    if DECKS:
        if deck not in DECKS:
            raise ValueError("%s must be one of %s, not %r" % (DECK_PARAM, ", ".join(DECKS), deck))
        setattr(p, DECK_PARAM, DECKS.index(deck))
    given = 0
    for i, (name, kind, lo, hi, rank, values, nonfinite, norm, lo_excl, hi_excl) in enumerate(PARAMS):
        v = params.get(name)
        if v is None:
            continue
        v = _check(name, kind, lo, hi, rank, values, nonfinite, v, norm, lo_excl, hi_excl)
        given |= 1 << i
        if kind in ("Arr", "IntArr"):
            dims = (_C.c_int64 * v.ndim)(*v.shape)
            keep += [v, dims]
            setattr(p, name, _Arg(v.ctypes.data_as(_C.c_void_p), v.ndim, dims))
        else:
            if kind == "Str":
                keep.append(v)
            setattr(p, name, v)
    p.given = given
    lines = []

    def line_cb(raw, n, stream, _ctx):
        text = _C.string_at(raw, n).decode("utf-8", "replace")
        lines.append(text)
        if stream == 2:
            _log.warning("%s", text)
        elif text.startswith("__"):
            _log.debug("%s", text)  # a record for Viv's tools (`__steps__ {...}`), not output
        else:
            _log.info("%s", text)
        if log is not None:
            log.append(text)
        if on_line is not None:
            try:
                on_line(text)
            except Exception:       # a callback error must not unwind through C
                _log.exception("on_line raised")

    cb = _LINE(line_cb)
    p.on_line = cb
    r = _Results()
    msg = _C.create_string_buffer(4096)
    box = {}

    done = _threading.Event()

    def work():
        try:
            box["code"] = _run_c(_C.byref(p), _C.byref(r), msg, len(msg))
        finally:
            done.set()

    # The run in a worker thread: Ctrl-C reaches the main thread here, between
    # waits, and stops the run at its next loop pass. (An Event, not
    # Thread.join: a join that Ctrl-C interrupts can report the thread ended
    # while it still runs, in CPython 3.11.)
    t = _threading.Thread(target=work, name=NAME + ".run", daemon=True)
    t.start()
    try:
        while not done.wait(0.1):
            pass
    except KeyboardInterrupt:
        while True:
            _interrupt_c()
            try:
                if done.wait(0.1):
                    break
            except KeyboardInterrupt:
                pass
        t.join()
        _free_c(_C.byref(r))
        raise
    code = box.get("code", 1)
    try:
        if code == 4:
            raise RuntimeError("%s: a run is already in progress in this process; run several in separate processes" % NAME)
        if code == 3:
            raise ValueError(msg.value.decode("utf-8", "replace"))
        if code == 2:
            raise KeyboardInterrupt("%s.run was interrupted" % NAME)
        if code != 0:
            raise _KINDS.get(r.error_kind, VivError)(msg.value.decode("utf-8", "replace"))
        out = {}
        for i, (name, kind) in enumerate(RESULTS):
            if not (r.have >> i) & 1:
                out[name] = None
            elif kind == "Num":
                out[name] = float(getattr(r, name))
            elif kind == "Int":
                out[name] = int(getattr(r, name))
            elif kind == "Bool":
                out[name] = bool(getattr(r, name))
            elif kind == "Str":
                out[name] = _C.string_at(getattr(r, name)).decode("utf-8", "replace")
            elif kind == "Arr":
                out[name] = _array(getattr(r, name), _C.c_double, _np.float64)
            else:
                out[name] = _array(getattr(r, name), _C.c_int64, _np.int64)
        out["tests"] = [(r.tests[i].name.decode(), bool(r.tests[i].passed), r.tests[i].detail.decode()) for i in range(r.ntests)]
        out["log"] = lines
        return out
    finally:
        _free_c(_C.byref(r))
"#;

/// `README.md` of an exported program.
pub fn readme(m: &Manifest, about: &str, version: &str) -> String {
    let p = &m.name;
    let up = p.to_uppercase();
    let mut s = String::new();
    let _ = writeln!(s, "# {p}\n");
    if let Some(summary) = &m.page.summary {
        let _ = writeln!(s, "{}\n", summary.trim_end());
    }
    if !about.is_empty() {
        let _ = writeln!(s, "{}\n", wrap(about, 78, "").trim_end());
    }
    if let Some(d) = &m.page.description {
        let _ = writeln!(s, "{}\n", d.trim_end());
    }
    let _ = writeln!(s, "A program written in Viv and exported as one C file with a Python wrapper.\nThe files here are written by `viv export`; edit the Viv source, not these.\n");
    let _ = writeln!(s, "- `{p}.c`: the program and the part of Viv's runtime it uses. C99 with GNU\n  attributes; it needs a C compiler and the C library, nothing else.\n- `{p}.h`: the C interface, also pasted at the top of `{p}.c`.\n- `{p}.py`: the Python interface (ctypes and NumPy; Python 3.9 or newer).\n- `test_{p}.py`: its tests.\n");
    let _ = writeln!(s, "## Build\n\n```sh\ncc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}.so -lm\n```\n");
    let _ = writeln!(s, "`import {p}` runs that line itself (with `$CC`, default `cc`) when the library\nis missing or older than `{p}.c`. On macOS the file is `lib{p}.dylib`. The name\nstarts with `lib` because Python would try to import `{p}.so` as an extension\nmodule.\n");
    let _ = writeln!(s, "`-ffp-contract=off` must stay: without it GCC fuses multiplies and adds and\nthe results change in the last bits. This exact build gives the bits Viv's\ninterpreter gives, on every platform and compiler: the same algorithm, the\nsame roundings, Viv's own `exp`, `log`, `sin` and the rest. That is what\n\"exact\" means here: reproducible, not more correct than the method.\n");
    let _ = writeln!(s, "The fast build, `{up}_FAST=1` for the wrapper:\n\n```sh\ncc -O3 -DVIV_FAST -ffp-contract=fast -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}_fast.so -lm\n```\n\nIt uses fused multiply-adds and the platform's `exp` and `log`, so results\ndiffer from the exact build in the last bits. Adding `-march=native` (or\n`-mcpu=native`) is safe when the library runs on the machine that built it.\n");
    let _ = writeln!(s, "## Use\n\n```python\nimport {p}\n\nr = {p}.run({})\nr[\"tests\"]          # the program's own checks: (name, passed, detail)\nhelp({p}.run)       # the parameters, their meaning and their defaults\n```\n", if m.decks.is_empty() { String::new() } else { format!("{}=\"{}\"", m.deck_param, m.decks[0].name) });
    let _ = writeln!(s, "A parameter left out keeps {}. The printed lines go to the\nlogger `{p}` at INFO (`logging.basicConfig(level=logging.INFO)` shows them)\nand to `r[\"log\"]`. An error in the program raises `{p}.VivError` (also an\n`IndexError`, `ValueError`, ... by kind) with Viv's message and source line;\nthe next run works. Ctrl-C stops a run at its next loop pass. One run at a\ntime per process: run several at once in separate processes\n(`concurrent.futures.ProcessPoolExecutor`).\n", if m.decks.is_empty() { "the program's own value" } else { "the deck's value" });
    if !m.reference.is_empty() {
        let _ = writeln!(s, "## Reference\n\nWhat the program is checked against outside Viv:\n");
        for (k, v) in &m.reference {
            let _ = writeln!(s, "- {k}: {v}");
        }
        s.push('\n');
    }
    if let Some(notes) = &m.page.notes {
        let _ = writeln!(s, "## Notes\n\n{}\n", notes.trim_end());
    }
    let documented: Vec<&Param> = m.params.iter().filter(|q| q.units.is_some() || q.range.is_some()).collect();
    if !documented.is_empty() {
        let _ = writeln!(s, "## Parameters\n\n| parameter | units | range | meaning |\n|---|---|---|---|");
        for q in documented {
            let cell = |x: &Option<String>| x.as_deref().unwrap_or("").replace('|', "\\|");
            let _ = writeln!(s, "| `{}` | {} | {} | {} |", q.name.to_lowercase(), cell(&q.units), cell(&q.range), cell(&q.doc));
        }
        s.push('\n');
    }
    if !m.page.examples.is_empty() {
        let _ = writeln!(s, "## Examples\n");
        for (caption, code) in &m.page.examples {
            let _ = writeln!(s, "{}\n\n```python\n{}\n```\n", caption.trim_end(), code.trim_end());
        }
    }
    if !m.page.references.is_empty() {
        let _ = writeln!(s, "## References\n");
        for r in &m.page.references {
            let get = |k: &str| r.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
            let link = match (get("doi"), get("url")) {
                (Some(d), _) => format!(" doi:[{d}](https://doi.org/{d})"),
                (None, Some(u)) => format!(" <{u}>"),
                _ => String::new(),
            };
            let _ = writeln!(s, "- {}{link}", get("text").unwrap_or(""));
        }
        s.push('\n');
    }
    if !m.page.see_also.is_empty() {
        let _ = writeln!(s, "## See also\n\n{}\n", m.page.see_also.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", "));
    }
    if m.cases.is_some() {
        let _ = writeln!(s, "## Cases\n\n`cases.json` holds named runs (`{{\"case\": {{parameter: value}}}}`). `test_{p}.py`\nruns each (`{up}_CASES=a,b` picks some) and, in the exact build, compares\nits result hash with `exact_hashes.json`, recorded from the library, a native\nbuild and Viv's interpreter. A case's `_expect` holds results it is compared\nwith outside Viv, each a value, a tolerance and where the value comes from\n(`{{\"e_tot\": {{\"value\": -74.963, \"tol\": 1e-10, \"from\": \"PySCF 2.5.0\"}}}}`); an array\nresult is held to the largest difference of its elements.\n");
    }
    let _ = writeln!(s, "## Version\n\n{version}");
    s
}

/// `test_pwdft.py`: the deck's tests pass through the wrapper; the exact
/// build's result hashes match the ones `viv export` recorded (when the
/// export was given them); errors and bad parameters raise.
pub fn test_py(m: &Manifest, hashes: Option<&str>) -> String {
    let p = &m.name;
    let mut s = String::new();
    let _ = writeln!(s, "\"\"\"Tests of the exported {p}: python -m pytest test_{p}.py, or python test_{p}.py.\nWritten by `viv export`; do not edit.\"\"\"");
    let _ = writeln!(s, "import hashlib\nimport json\nimport os\nimport struct\nimport sys\n\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport {p}  # noqa: E402\n");
    let _ = writeln!(s, "HASHES = os.path.join(os.path.dirname(os.path.abspath(__file__)), \"exact_hashes.json\")\n");
    s.push_str(TEST_HASH);
    if !m.decks.is_empty() {
        let names: Vec<String> = m.decks.iter().map(|d| format!("{:?}", d.name)).collect();
        let checked: Vec<String> = m.decks.iter().filter(|d| d.checks.is_some()).map(|d| format!("{:?}", d.name)).collect();
        let _ = writeln!(s, "\nDECKS = [{}]\nWITH_CHECKS = [{}]\n", names.join(", "), checked.join(", "));
        let _ = writeln!(s, "\ndef test_every_deck_passes_its_tests_and_matches_the_recorded_bits():\n    want = json.load(open(HASHES)) if os.path.exists(HASHES) else {{}}\n    for deck in DECKS:\n        r = {p}.run({}=deck)\n        failed = [t for t in r[\"tests\"] if not t[1]]\n        assert not failed, (deck, failed)\n        assert r[\"tests\"] or deck not in WITH_CHECKS, deck\n        if deck in want and not {p}._FAST:\n            assert result_hash(r) == want[deck], deck\n", m.deck_param);
    } else if m.cases.is_some() {
        let _ = writeln!(s, "CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), \"cases.json\")\n");
        let _ = writeln!(s, "{}", TEST_CASES.replace("{p}", p).replace("{UP}", &p.to_uppercase()));
    } else {
        let _ = writeln!(s, "\ndef test_the_program_passes_its_tests_and_matches_the_recorded_bits():\n    want = json.load(open(HASHES)) if os.path.exists(HASHES) else {{}}\n    r = {p}.run()\n    failed = [t for t in r[\"tests\"] if not t[1]]\n    assert not failed, failed\n    if \"\" in want and not {p}._FAST:\n        assert result_hash(r) == want[\"\"]\n");
    }
    let _ = writeln!(s, "\ndef test_an_unknown_parameter_is_refused():\n    try:\n        {p}.run(no_such_parameter=1)\n    except TypeError:\n        return\n    raise AssertionError(\"no TypeError\")\n");
    if hashes.is_some() {
        let _ = writeln!(s, "\n# the hashes were recorded by `viv export` from the interpreter and a native build");
    }
    let _ = writeln!(s, "\nif __name__ == \"__main__\":\n    for name, f in list(globals().items()):\n        if name.startswith(\"test_\"):\n            f()\n            print(\"ok\", name)");
    s
}

/// The generated test of a program with `cases`: each case's parameters are
/// its keys, less `_expect`, which holds results to compare with a value of
/// a reference outside Viv (`{"e_tot": {"value": -74.963, "tol": 1e-10}}`;
/// an array result elementwise, by the largest difference).
const TEST_CASES: &str = r#"
def _expected(name, r, expect):
    import numpy as np
    for res, e in expect.items():
        got, want = np.asarray(r[res], dtype=float), np.asarray(e["value"], dtype=float)
        assert got.shape == want.shape, (name, res, got.shape, want.shape)
        diff = float(np.max(np.abs(got - want))) if got.size else 0.0
        assert diff <= e["tol"], "%s: %s is %.3g from the reference%s, past its tolerance %.3g" % (
            name, res, diff, " (" + e["from"] + ")" if e.get("from") else "", e["tol"])


def test_every_case_passes_its_tests_and_matches_the_recorded_bits():
    want = json.load(open(HASHES)) if os.path.exists(HASHES) else {}
    cases = json.load(open(CASES))
    pick = os.environ.get("{UP}_CASES")
    for name, case in cases.items():
        if pick and name not in pick.split(","):
            continue
        params = {k: v for k, v in case.items() if not k.startswith("_")}
        r = {p}.run(**params)
        failed = [t for t in r["tests"] if not t[1]]
        assert not failed, (name, failed)
        _expected(name, r, case.get("_expect", {}))
        if name in want and not {p}._FAST:
            assert result_hash(r) == want[name], name
        print("ok", name, flush=True)
"#;

pub(crate) const TEST_HASH: &str = r#"
def result_hash(r):
    """SHA-256 of a run's results: each result's name, then its bytes (a
    number as an IEEE double, an int as 8 bytes, an array's shape and its
    elements as doubles or 8-byte ints), in name order."""
    h = hashlib.sha256()
    for name in sorted(k for k in r if k not in ("tests", "log")):
        v = r[name]
        h.update(name.encode() + b"\0")
        if v is None:
            h.update(b"none")
        elif isinstance(v, bool):
            h.update(struct.pack("<d", float(v)))
        elif isinstance(v, int):
            h.update(struct.pack("<q", v))
        elif isinstance(v, float):
            h.update(struct.pack("<d", v))
        elif isinstance(v, str):
            h.update(v.encode())
        else:
            h.update(struct.pack("<q", v.ndim) + b"".join(struct.pack("<q", d) for d in v.shape))
            h.update(v.astype("<f8" if v.dtype.kind == "f" else "<i8").tobytes())
    return h.hexdigest()
"#;

/// A deck's report file: prints every result at full precision, one line
/// each value (`RESULT name index value`), for a native or interpreted run
/// of the program to be compared with the library's bits.
pub fn report_viv(m: &Manifest, types: &HashMap<String, Ty>, produced: &[bool]) -> String {
    let mut s = String::from("# written by `viv export` for its exact-bits check: every result in full\n");
    for (r, &yes) in m.results.iter().zip(produced) {
        if !yes {
            continue;
        }
        match types.get(&r.name) {
            Some(Ty::Num) => {
                let _ = writeln!(s, "print(\"RESULT {n} 0\", fmt({n}))", n = r.name);
            }
            // `fmt` takes numbers only (B261)
            Some(Ty::Bool) => {
                let _ = writeln!(s, "if {n} {{\n    print(\"RESULT {n} 0 1\")\n}}\nelse {{\n    print(\"RESULT {n} 0 0\")\n}}", n = r.name);
            }
            Some(Ty::Int) => {
                let _ = writeln!(s, "print(\"RESULT {n} 0\", {n})", n = r.name);
            }
            // a string as it is: check.py reads the rest of the line
            Some(Ty::Str) => {
                let _ = writeln!(s, "print(\"RESULT {n} 0\", {n})", n = r.name);
            }
            Some(Ty::Arr(Elem::Num)) | Some(Ty::Arr(Elem::Int)) => {
                let fmtv = if matches!(types.get(&r.name), Some(Ty::Arr(Elem::Int))) { format!("__x_{}", r.name) } else { format!("fmt(__x_{})", r.name) };
                let _ = writeln!(s, "print(\"SHAPE {n}\", shape({n}))", n = r.name);
                let _ = writeln!(s, "__flat_{n} = ravel({n})\nfor __k_{n} to numel({n}) {{\n    __x_{n} = __flat_{n}[__k_{n}]\n    print(\"RESULT {n}\", __k_{n}, {fmtv})\n}}", n = r.name);
            }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_reads_the_reference_page() {
        let text = "name = \"p\"\nfiles = [\"a.viv\"]\nsummary = \"One line.\"\nnotes = '''\n### Math\n\n$\\frac{a}{b}$ # not a comment\n'''\nsee_also = [\"vp.x\"]\nreferences = [\n  { key = \"A1\", text = \"A. Author, J. 1, 2 (2000).\", doi = \"10.1/x\" },\n]\n[params]\nH = { type = \"Num\", units = \"fm\", range = \"0.05 to 0.2\", doc = \"the spacing\" }\n[[examples]]\ncaption = \"First.\"\ncode = '''\nprint(1)\n'''\n[[examples]]\ncaption = \"Second.\"\ncode = '''print(2)'''\n";
        let m = parse_manifest(text).unwrap();
        assert_eq!(m.page.summary.as_deref(), Some("One line."));
        assert_eq!(m.page.notes.as_deref(), Some("### Math\n\n$\\frac{a}{b}$ # not a comment\n"));
        assert_eq!(m.page.see_also, vec!["vp.x"]);
        assert_eq!(m.page.references[0][2], ("doi".to_string(), "10.1/x".to_string()));
        assert_eq!(m.params[0].units.as_deref(), Some("fm"));
        assert_eq!(m.params[0].range.as_deref(), Some("0.05 to 0.2"));
        assert_eq!(m.page.examples, vec![("First.".to_string(), "print(1)\n".to_string()), ("Second.".to_string(), "print(2)".to_string())]);
        let readme = readme(&m, "", "v");
        assert!(readme.contains("## Notes") && readme.contains("doi:[10.1/x](https://doi.org/10.1/x)") && readme.contains("| `h` | fm | 0.05 to 0.2 | the spacing |"));
        assert!(parse_manifest("name = \"p\"\nfiles = [\"a.viv\"]\nnotes = '''\nopen\n").unwrap_err().contains("not closed"));
        assert!(parse_manifest("name = \"p\"\nfiles = [\"a.viv\"]\nreferences = [{ title = \"x\" }]\n").unwrap_err().contains("title"));
        assert!(parse_manifest("name = \"p\"\nfiles = [\"a.viv\"]\n[[examples]]\nwhy = \"x\"\n").unwrap_err().contains("caption or code"));
    }

    #[test]
    fn the_manifest_reads_its_sections_and_refuses_what_it_cannot_take() {
        let m = parse_manifest(
            "name = \"p\"\nfiles = [\"a.viv\",\n  \"b.viv\"]  # two\n[decks]\nx = { deck = \"x.viv\", checks = \"xc.viv\" }\n[params]\nN = { type = \"Int\", min = 1, max = 8 }\nB = { type = \"Enum\", values = [\"sto-3g\", \"6-31g*\"] }\nZ = \"IntArr\"\n[results]\nE = \"Num\"\nC = { type = \"Arr\", doc = \"coefficients\" }\n",
        )
        .unwrap();
        assert_eq!(m.files, vec!["a.viv", "b.viv"]);
        assert_eq!(m.decks[0].checks.as_deref(), Some("xc.viv"));
        assert_eq!(m.params[0].kind, Kind::Int);
        assert_eq!(m.params[0].max, Some(8.0));
        assert_eq!(m.params[1].kind, Kind::Enum(vec!["sto-3g".into(), "6-31g*".into()]));
        assert_eq!(m.params[2].kind, Kind::IntArr);
        assert_eq!(m.results[1].doc.as_deref(), Some("coefficients"));
        assert_eq!(enum_ident("6-31g*"), "6_31G_STAR");
        let k = parse_manifest("name = \"k1\"\nkind = \"kernels\"\nprefix = \"vkern_\"\nabi = 2\nfiles = [\"a.viv\"]\n[kernels.pz]\nviv = \"a.viv:plasma_z\"\nargs = [{ name = \"zeta\", type = \"Complex\" }, { name = \"A\", type = \"Arr\", square = true, nonfinite = true }]\nresults = [{ name = \"Z\", type = \"Complex\", shape = \"n + 1\" }]\ntolerance = { rel = 2.0e-14 }\n").unwrap();
        assert!(k.is_kernels && k.prefix == "vkern_" && k.abi == 2);
        assert_eq!((k.kernels[0].file.as_str(), k.kernels[0].def.as_str()), ("a.viv", "plasma_z"));
        assert_eq!(k.kernels[0].args[0].kind, Kind::Cplx);
        assert!(k.kernels[0].args[1].square && k.kernels[0].args[1].rank == Some(2) && k.kernels[0].args[1].nonfinite);
        assert_eq!(k.kernels[0].results[0].doc.as_deref(), Some("shape n + 1"));
        assert_eq!(k.kernels[0].info, vec![("tolerance".to_string(), "rel = 2e-14".to_string())]);
        let g = parse_manifest("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nZ = { type = \"IntArr\", min = 1, max = 18 }\nXYZ = { type = \"Arr\", shape = [\"Z\", 3] }\n[groups]\ngeometry = [\"Z\", \"XYZ\"]\n").unwrap();
        assert_eq!(g.params[1].shape, Some(vec![Dim::LenOf("Z".into()), Dim::Fixed(3)]));
        assert_eq!(g.params[1].rank, Some(2));
        assert_eq!(g.groups, vec![("geometry".to_string(), vec!["Z".to_string(), "XYZ".to_string()])]);
        assert!(parse_manifest("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nXYZ = { type = \"Arr\", shape = [\"Q\", 3] }\n").unwrap_err().contains("not an array parameter"));
        for (bad, why) in [
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nR = \"Rec\"\n", "not a type"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nM = \"Map\"\n", "not a type"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[results]\nE = { type = \"Enum\", doc = \"a\" }\n", "Enum"),
            ("name = \"p-q\"\nfiles = [\"a.viv\"]\n", "C identifier"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nS = { type = \"Str\", min = 1 }\n", "min and max"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nA = { type = \"Num\" }\na = \"Num\"\n", "two parameters"),
            // B280: names the generated C or Python already uses
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nDOUBLE = \"Num\"\n", "keyword or a macro in C"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nINT = \"Int\"\n", "keyword or a macro in C"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nGIVEN = \"Num\"\n", "field `given`"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[results]\nTESTS = \"Num\"\n", "field `tests`"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[results]\nerrno = \"Num\"\n", "keyword or a macro in C"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\ndeck_param = \"case\"\n[decks]\nx = { deck = \"x.viv\" }\n", "keyword or a macro in C"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nE = { type = \"Enum\", values = [\"index\"] }\n", "error kinds"),
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nX = { type = \"Num\", min = 0, min_exclusive = 0 }\n", "not both"),
            // kernels: Complex is a kernel's type only; a kernel names its def
            ("name = \"p\"\nfiles = [\"a.viv\"]\n[params]\nZ = \"Cplx\"\n", "kernel's argument or result only"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[kernels.f]\nresults = [{ name = \"y\", type = \"Num\" }]\n", "no `viv"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[kernels.f]\nviv = \"b.viv:f\"\nresults = [{ name = \"y\", type = \"Num\" }]\n", "which `files` does not name"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[kernels.init]\nviv = \"a.viv:f\"\nresults = [{ name = \"y\", type = \"Num\" }]\n", "own entry points"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[kernels.f]\nviv = \"a.viv:f\"\nargs = [{ name = \"x\", type = \"Str\" }]\nresults = [{ name = \"y\", type = \"Num\" }]\n", "a kernel takes Num, Int, Bool, Complex and Arr"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[kernels.f]\nviv = \"a.viv:f\"\nargs = [{ name = \"double\", type = \"Num\" }]\nresults = [{ name = \"y\", type = \"Num\" }]\n", "keyword or a macro in C"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[kernels.f]\nviv = \"a.viv:f\"\nargs = [{ name = \"A\", type = \"Arr\", rank = 1, square = true }]\nresults = [{ name = \"y\", type = \"Num\" }]\n", "square is for a matrix"),
            ("name = \"p\"\nkind = \"kernels\"\nfiles = [\"a.viv\"]\n[params]\nN = \"Num\"\n[kernels.f]\nviv = \"a.viv:f\"\nresults = [{ name = \"y\", type = \"Num\" }]\n", "not [decks], [params]"),
        ] {
            let e = parse_manifest(bad).unwrap_err();
            assert!(e.contains(why), "{bad:?}: {e}");
        }
    }

    fn reader_of(files: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Result<String, String> {
        move |p: &str| files.iter().find(|(n, _)| *n == p).map(|(_, t)| t.to_string()).ok_or(format!("no {p}"))
    }

    #[test]
    fn a_parameter_is_read_where_its_deck_assigns_it() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[decks]\nx = { deck = \"x.viv\" }\ny = { deck = \"y.viv\" }\n[params]\nN = \"Int\"\nA = \"Num\"\n[results]\nOUT = \"Num\"\n").unwrap();
        let files: &'static [(&'static str, &'static str)] = &[("x.viv", "N = 2.0  #: N  the size\nA = 1.0\n"), ("y.viv", "A = 3.0\nN = 4.0\n"), ("e.viv", "OUT = N * A\n")];
        let mut types = HashMap::new();
        types.insert("N".to_string(), Ty::Num);
        types.insert("A".to_string(), Ty::Num);
        let a = assemble(&m, &reader_of(files), Some(&types)).unwrap();
        let text: String = a.parts.iter().map(|p| p.text.as_str()).collect();
        assert!(text.contains("if __export_deck == 0.0 {\nN = 2.0  #: N  the size\nif __export_given(0) { N = __export_num(1) }\nA = 1.0\nif __export_given(1) { A = __export_num(2) }\n"), "{text}");
        assert!(text.contains("A = 3.0\nif __export_given(1) { A = __export_num(2) }\nN = 4.0\nif __export_given(0) { N = __export_num(1) }\n"), "{text}");
        assert_eq!(a.defaults[0], vec![Some("2.0".to_string()), Some("4.0".to_string())]);
        assert_eq!(a.glossary.get("N").map(String::as_str), Some("the size"));
        // each deck's own `#:` line (the first deck's is the main text)
        let files2: &'static [(&'static str, &'static str)] = &[("x.viv", "N = 2.0  #: N  the size\nA = 1.0\n"), ("y.viv", "A = 3.0\n#: N  the size, doubled for y\nN = 4.0\n"), ("e.viv", "OUT = N * A\n")];
        let a2 = assemble(&m, &reader_of(files2), Some(&types)).unwrap();
        assert_eq!(a2.deck_help[0], vec![Some("the size".to_string()), Some("the size, doubled for y".to_string())]);
        let py = python(&m, &a2, "", "v");
        assert!(py.contains("        in y: the size, doubled for y\n") && !py.contains("in x:"), "{py}");
        // every part cut from a file is that file's bytes at its offset
        for part in &a.parts {
            if let Some((name, at)) = &part.file {
                let full = reader_of(files)(name).unwrap();
                assert_eq!(&full[*at..*at + part.text.len()], part.text);
            }
        }
    }

    #[test]
    fn an_enum_parameter_is_read_by_its_labels_and_its_default_shown_as_one() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[params]\nb = { type = \"Basis\", labels = { \"6-31g*\" = \"P\", \"sto-3g\" = \"S\" }, normalize = \"pyscf_basis\" }\nq = \"Int\"\n[results]\nE = { type = \"Arr\", rank = { by = \"b\", \"6-31g*\" = 2, \"sto-3g\" = 1 } }\n").unwrap();
        assert_eq!(m.params[0].kind, Kind::Variant { ty: "Basis".into(), labels: vec![("6-31g*".into(), "P".into()), ("sto-3g".into(), "S".into())] });
        assert_eq!(m.results[0].rank.as_ref().map(Rank::text).as_deref(), Some("rank 2 when b is 6-31g*, rank 1 when b is sto-3g"));
        assert_eq!(normalize("pyscf_basis", "6-31G*"), normalize("pyscf_basis", "631gs"));
        let files: &'static [(&'static str, &'static str)] = &[("e.viv", "enum Basis {\n    S\n    P\n}\n#: b  the basis\nb = Basis.S\nq: Int = 2\nE = zeros(2)\n")];
        let mut types = HashMap::new();
        types.insert("b".to_string(), Ty::Enum("Basis"));
        types.insert("q".to_string(), Ty::Int);
        let a = assemble(&m, &reader_of(files), Some(&types)).unwrap();
        let text: String = a.parts.iter().map(|p| p.text.as_str()).collect();
        assert!(text.contains("b = Basis.S\nif __export_given(0) {\n    if __export_num(0) == 0.0 { b = Basis.P }\n    if __export_num(0) == 1.0 { b = Basis.S }\n}\n"), "{text}");
        assert!(text.contains("q: Int = 2\nif __export_given(1) { q = __export_int(0) }\n"), "{text}");
        assert_eq!(a.defaults[0], vec![Some("\"sto-3g\"".to_string())]);
        assert_eq!(a.variant_tags[0], vec![1, 0]);
        assert_eq!(a.glossary.get("b").map(String::as_str), Some("the basis"));
        // a module path or a field with the parameter's name is not a read of it
        let files: &'static [(&'static str, &'static str)] = &[("e.viv", "enum Basis {\n    S\n    P\n}\nr = 0.0\nbb = r.b\nb = Basis.S\nq = 1\n")];
        assert!(assemble(&m, &reader_of(files), None).is_ok());
        let files: &'static [(&'static str, &'static str)] = &[("e.viv", "enum Basis {\n    S\n    P\n}\nc = b\nb = Basis.S\nq = 1\n")];
        assert!(assemble(&m, &reader_of(files), None).unwrap_err().contains("read before"));
    }

    /// B262: the imports of a program's files are spliced in, so lib/hf's
    /// `method = setup.method` (a local of `scf_result`) was refused as an
    /// assignment of the parameter `method`. A def's assignment makes a local.
    #[test]
    fn a_def_local_named_like_a_parameter_is_not_the_parameter() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[params]\nmethod = \"Num\"\n").unwrap();
        let files: &'static [(&'static str, &'static str)] = &[("e.viv", "def f(s) {\n    method = s * 2.0\n    return method\n}\nmethod = 1.0\nshow f(method)\n")];
        assert!(assemble(&m, &reader_of(files), None).is_ok());
    }

    /// B272: an imported module's `###` text is spliced in before the
    /// program's own, and its words are not reads of a parameter (the check
    /// reads parsed statements since B263).
    #[test]
    fn a_parameters_name_in_a_modules_text_block_is_not_a_read() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[params]\nMAX_ITER = \"Num\"\n").unwrap();
        let files: &'static [(&'static str, &'static str)] = &[("e.viv", "###\nThis module's MIX, E_TOL and MAX_ITER.\n###\ndef f(a) {\n    return a\n}\nMAX_ITER = 10.0\nshow f(MAX_ITER)\n")];
        assert!(assemble(&m, &reader_of(files), None).is_ok());
    }

    /// B261: the report of a Bool result went through `fmt`, which takes
    /// numbers only, so the native and interpreted runs of check.py failed to
    /// compile for any program with a Bool result.
    #[test]
    fn a_bool_result_is_reported_without_fmt() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[results]\nok = \"Bool\"\ne = \"Num\"\n").unwrap();
        let mut types = HashMap::new();
        types.insert("ok".to_string(), Ty::Bool);
        types.insert("e".to_string(), Ty::Num);
        let r = report_viv(&m, &types, &[true, true]);
        assert!(r.contains("if ok {\n    print(\"RESULT ok 0 1\")\n}\nelse {\n    print(\"RESULT ok 0 0\")\n}") && !r.contains("fmt(ok)"), "{r}");
        let src = format!("ok = 1.0 > 0.5\ne = 2.0\n{r}");
        let out = crate::interp::run(&src).expect("the report runs");
        assert!(out.contains("RESULT ok 0 1\nRESULT e 0 2.0\n"), "{out}");
    }

    /// A Str result was left out of the report, so check.py compared the
    /// library's string with nothing and called every run different.
    #[test]
    fn a_str_result_is_reported() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[results]\nid = \"Str\"\n").unwrap();
        let mut types = HashMap::new();
        types.insert("id".to_string(), Ty::Str);
        let r = report_viv(&m, &types, &[true]);
        let src = format!("id = \"run 7-a\"\n{r}");
        let out = crate::interp::run(&src).expect("the report runs");
        assert!(out.contains("RESULT id 0 run 7-a\n"), "{out}");
    }

    #[test]
    fn a_parameter_set_twice_or_read_first_or_in_a_def_is_refused() {
        let m = parse_manifest("name = \"p\"\nfiles = [\"e.viv\"]\n[decks]\nx = { deck = \"x.viv\" }\n[params]\nN = \"Num\"\n").unwrap();
        for (deck, engine, why) in [
            ("N = 2.0\nN = 3.0\n", "show N\n", "more than once"),
            ("M = N\nN = 2.0\n", "show N\n", "read before"),
            ("N = 2.0\n", "N = 5.0\n", "after deck x sets it"),
            ("M = 1.0\n", "def f(a) {\n    N = a\n    return N\n}\nshow f(1.0)\n", "inside `f`"),
            ("if 1.0 > 0.0 {\n    N = 2.0\n}\n", "show N\n", "inside a block"),
            ("M = 1.0\n", "show M\n", "assigned neither"),
            ("def g(a) {\n    return a\n}\nN = 1.0\n", "show N\n", "statements only"),
        ] {
            let files: Vec<(&'static str, &'static str)> = vec![("x.viv", Box::leak(deck.to_string().into_boxed_str())), ("e.viv", Box::leak(engine.to_string().into_boxed_str()))];
            let files: &'static [(&'static str, &'static str)] = Box::leak(files.into_boxed_slice());
            let e = assemble(&m, &reader_of(files), None).unwrap_err();
            assert!(e.contains(why), "{deck:?} / {engine:?}: {e}");
        }
        // the type of the global must be one the parameter's kind can set
        let files: &'static [(&'static str, &'static str)] = &[("x.viv", "N = 2.0\n"), ("e.viv", "show N\n")];
        let mut types = HashMap::new();
        types.insert("N".to_string(), Ty::Str);
        let e = assemble(&m, &reader_of(files), Some(&types)).unwrap_err();
        assert!(e.contains("is Str in the program, and the manifest says Num"), "{e}");
    }
}
