// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The architecture lens: which layer each
//! module belongs to, which way the layers may depend on each other, what a
//! program may cost (budgets), and which stages have several implementations
//! of one intent (multi-paths).
//!
//! An `.arch` file declares all of it; this module reads the file, reads the
//! modules it covers, and says where the code and the file disagree:
//!
//! - `verification-dep`: a library module uses or calls a verification
//!   module (a sweep driver, a reference pack, a lock program);
//! - `upward`: a module uses or calls a module of a later layer;
//! - `private`: a module reads names another module owns (a table's columns);
//! - `budget`: a program loads more, or calls deeper, than its budget; a
//!   dispatch def has more callers than its budget; an entry point takes more
//!   calls than its budget to reach a def;
//! - `indirect-default`: the production path reaches a stage's default member
//!   only through a registry and its drivers, not by a direct call;
//! - `stage`: a multi-path declaration that does not match the code (a member
//!   the registry does not hold, a contract that names no tolerance);
//! - `unassigned`: a covered module in no layer, or in two.
//!
//! It also measures: a program's modules and defs loaded against the defs its
//! run called, its deepest call chain, fan-in and fan-out, the calls between
//! modules, and per stage the number of independent implementations and how
//! many of them were seen agreeing. Two members are independent when the code
//! each reaches shares no def outside the stage machinery and the declared
//! `shared` modules.
//!
//! Like lens.rs, nothing here reads the file system or runs anything:
//! arch_cli (main.rs's side) hands over the texts and the runs.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use knotkit::span::SourceId;

use crate::ast::*;
use crate::debug_lens::json_str;
use crate::graph::Lines;

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

/// What a layer is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// production code: may depend on earlier library layers only
    Library,
    /// programs: may depend on any library layer
    Program,
    /// tests, sweeps, reference data, lock programs: may depend on anything,
    /// and nothing in the library may depend on them
    Verification,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Library => "library",
            Kind::Program => "program",
            Kind::Verification => "verification",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub id: String,
    pub kind: Kind,
    pub what: String,
    /// module patterns: `lib/hf/basis`, `lib/basis/*` (one level), `scf/suite/**` (any depth)
    pub modules: Vec<String>,
    pub line: usize,
}

#[derive(Clone, Debug)]
pub struct Shared {
    pub modules: Vec<String>,
    pub why: String,
    pub line: usize,
}

/// Names (patterns like `SH_*`) only the owner modules may read.
#[derive(Clone, Debug)]
pub struct Private {
    pub names: Vec<String>,
    pub owners: Vec<String>,
    pub why: String,
    pub line: usize,
}

/// One limit of a budget: `modules 20`, `fanin 2`, `reach scf_iterate 3`.
#[derive(Clone, Debug)]
pub struct Limit {
    pub metric: String,
    pub arg: Option<String>,
    pub value: usize,
    pub line: usize,
}

#[derive(Clone, Debug)]
pub struct Budget {
    /// `budget program <name>` or `budget def <name>`
    pub program: Option<String>,
    pub def: Option<String>,
    pub limits: Vec<Limit>,
    pub why: String,
    pub line: usize,
}

/// A stage: `multipath` (several implementations of one intent) or `stage`
/// (one implementation).
#[derive(Clone, Debug, Default)]
pub struct Stage {
    pub id: String,
    pub title: String,
    pub multi: bool,
    /// the def whose intent entry is the stage's intent (the default member when not given)
    pub intent: Option<String>,
    /// the def that returns the members as function values
    pub registry: Option<String>,
    pub members: Vec<String>,
    /// the member the production path runs
    pub default: Option<String>,
    /// the defs the production path passes through to reach the default
    pub through: Vec<String>,
    /// `<def> within <constant>`, or text with a citation
    pub contract: Option<String>,
    /// citations of the tests and programs where the members are run side by side
    pub checked: Option<String>,
    pub why: String,
    pub line: usize,
    /// the line of each field, for findings
    pub lines: HashMap<String, usize>,
}

impl Stage {
    fn at(&self, field: &str) -> usize {
        self.lines.get(field).copied().unwrap_or(self.line)
    }
    /// The intent def: the declared one, else the default member, else the only member.
    pub fn intent_def(&self) -> Option<&str> {
        self.intent.as_deref().or(self.default.as_deref()).or(self.members.first().map(String::as_str))
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArchFile {
    pub path: String,
    pub system: String,
    pub title: String,
    pub covers: Vec<String>,
    pub layers: Vec<Layer>,
    /// the library and program layers, bottom first
    pub order: Vec<String>,
    pub order_line: usize,
    pub shared: Vec<Shared>,
    pub private: Vec<Private>,
    pub budgets: Vec<Budget>,
    pub stages: Vec<Stage>,
    /// (line, message) for what could not be read
    pub errors: Vec<(usize, String)>,
}

impl ArchFile {
    pub fn layer(&self, id: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }
    /// A layer's place in `order` (library and program layers).
    fn rank(&self, id: &str) -> Option<usize> {
        self.order.iter().position(|o| o == id)
    }
}

/// Whether a module name matches a pattern: `*` matches within one path
/// segment, `**` any depth.
pub fn matches(pattern: &str, name: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix("**") {
        return name.starts_with(prefix);
    }
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some(b'*') => (0..=s.len()).take_while(|&k| k == 0 || s[k - 1] != b'/').any(|k| go(&p[1..], &s[k..])),
            Some(c) => s.first() == Some(c) && go(&p[1..], &s[1..]),
        }
    }
    go(pattern.as_bytes(), name.as_bytes())
}

fn name_matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix) && name.len() > prefix.len(),
        None => pattern == name,
    }
}

const BLOCKS: [&str; 8] = ["system", "layer", "order", "shared", "private", "budget", "multipath", "stage"];

fn keys_of(block: &str) -> &'static [&'static str] {
    match block {
        "system" => &["covers"],
        "layer" => &["what", "modules"],
        "shared" => &["why", "modules"],
        "private" => &["owner", "why"],
        "budget" => &["modules", "defs", "depth", "fanin", "reach", "why"],
        "multipath" | "stage" => &["intent", "registry", "members", "default", "through", "contract", "checked", "why"],
        _ => &[],
    }
}

/// Split a header's rest into its first word and the text after two or more
/// spaces (or after the first word, when there are no two spaces).
fn head_title(rest: &str) -> (String, String) {
    let rest = rest.trim();
    match rest.find("  ") {
        Some(k) => (rest[..k].trim().to_string(), rest[k..].trim().to_string()),
        None => match rest.split_once(' ') {
            Some((a, b)) => (a.to_string(), b.trim().to_string()),
            None => (rest.to_string(), String::new()),
        },
    }
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

/// Read an `.arch` file. What cannot be read is in `errors`; the rest is kept.
pub fn parse(path: &str, text: &str) -> ArchFile {
    let mut f = ArchFile { path: path.to_string(), ..Default::default() };
    // the blocks: (keyword, header rest, header line, fields (key, value, line))
    let mut blocks: Vec<(String, String, usize, Vec<(String, String, usize)>)> = Vec::new();
    // the indent of the last field's key: a line indented deeper continues it
    let mut key_indent = usize::MAX;
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if !raw.starts_with(' ') && !raw.starts_with('\t') {
            let (kw, rest) = match t.split_once(' ') {
                Some((a, b)) => (a.to_string(), b.to_string()),
                None => (t.to_string(), String::new()),
            };
            if !BLOCKS.contains(&kw.as_str()) {
                f.errors.push((line, format!("`{kw}` is not a block of an architecture lens (expected one of {})", BLOCKS.join(", "))));
                blocks.push(("?".into(), String::new(), line, Vec::new()));
                continue;
            }
            blocks.push((kw, rest, line, Vec::new()));
            continue;
        }
        let Some(b) = blocks.last_mut() else {
            f.errors.push((line, "an indented line before any block".into()));
            continue;
        };
        if b.0 == "?" {
            continue;
        }
        let first = t.split_whitespace().next().unwrap_or("");
        let indent = raw.len() - raw.trim_start().len();
        if b.3.is_empty() {
            key_indent = usize::MAX;
        }
        if indent > key_indent && !b.3.is_empty() {
            let last = b.3.last_mut().expect("checked");
            last.1.push(' ');
            last.1.push_str(t);
        } else if keys_of(&b.0).contains(&first) {
            b.3.push((first.to_string(), t[first.len()..].trim().to_string(), line));
            key_indent = indent;
        } else if b.0 == "shared" && b.3.is_empty() {
            // the header's module list continues
            b.1.push(' ');
            b.1.push_str(t);
        } else {
            f.errors.push((line, format!("`{first}` is not a field of `{}` (expected one of {})", b.0, keys_of(&b.0).join(", "))));
        }
    }
    let field = |fs: &[(String, String, usize)], k: &str| fs.iter().find(|x| x.0 == k).map(|x| (x.1.clone(), x.2));
    for (kw, rest, line, fs) in blocks {
        match kw.as_str() {
            "system" => {
                let (id, title) = head_title(&rest);
                f.system = id;
                f.title = title;
                f.covers = field(&fs, "covers").map(|v| words(&v.0)).unwrap_or_default();
            }
            "layer" => {
                let (id, kind) = head_title(&rest);
                let kind = match kind.as_str() {
                    "library" => Kind::Library,
                    "program" => Kind::Program,
                    "verification" => Kind::Verification,
                    other => {
                        f.errors.push((line, format!("layer `{id}`: the kind is `{other}`; it must be library, program or verification")));
                        Kind::Library
                    }
                };
                if f.layers.iter().any(|l| l.id == id) {
                    f.errors.push((line, format!("layer `{id}` is declared twice")));
                }
                f.layers.push(Layer { id, kind, what: field(&fs, "what").map(|v| v.0).unwrap_or_default(), modules: field(&fs, "modules").map(|v| words(&v.0)).unwrap_or_default(), line });
            }
            "order" => {
                f.order = rest.split('<').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                f.order_line = line;
            }
            "shared" => {
                let mut modules = words(&rest);
                if let Some((m, _)) = field(&fs, "modules") {
                    modules.extend(words(&m));
                }
                f.shared.push(Shared { modules, why: field(&fs, "why").map(|v| v.0).unwrap_or_default(), line });
            }
            "private" => {
                let owners = field(&fs, "owner").map(|v| words(&v.0)).unwrap_or_default();
                if owners.is_empty() {
                    f.errors.push((line, "a `private` block needs an `owner` line: the modules that may read the names".into()));
                }
                f.private.push(Private { names: words(&rest), owners, why: field(&fs, "why").map(|v| v.0).unwrap_or_default(), line });
            }
            "budget" => {
                let ws = words(&rest);
                let (program, def) = match (ws.first().map(String::as_str), ws.get(1)) {
                    (Some("program"), Some(p)) => (Some(p.trim_end_matches(".viv").to_string()), None),
                    (Some("def"), Some(d)) => (None, Some(d.clone())),
                    _ => {
                        f.errors.push((line, "a budget is `budget program <name>` or `budget def <name>`".into()));
                        continue;
                    }
                };
                let mut limits = Vec::new();
                for (k, v, l) in &fs {
                    if k == "why" {
                        continue;
                    }
                    let vs = words(v);
                    let (arg, num) = match (k.as_str(), vs.as_slice()) {
                        ("reach", [a, n]) => (Some(a.clone()), n.parse::<usize>().ok()),
                        ("reach", _) => (None, None),
                        (_, [n]) => (None, n.parse::<usize>().ok()),
                        _ => (None, None),
                    };
                    let Some(value) = num else {
                        f.errors.push((*l, format!("`{k} {v}`: expected {}", if k == "reach" { "`reach <def> <calls>`" } else { "a whole number" })));
                        continue;
                    };
                    let fits = match k.as_str() {
                        "modules" | "defs" | "depth" => program.is_some(),
                        _ => def.is_some(),
                    };
                    if !fits {
                        f.errors.push((*l, format!("`{k}` is a limit of a {} budget", if program.is_some() { "def" } else { "program" })));
                        continue;
                    }
                    limits.push(Limit { metric: k.clone(), arg, value, line: *l });
                }
                f.budgets.push(Budget { program, def, limits, why: field(&fs, "why").map(|v| v.0).unwrap_or_default(), line });
            }
            "multipath" | "stage" => {
                let (id, title) = head_title(&rest);
                let mut s = Stage { id: id.clone(), title, multi: kw == "multipath", line, ..Default::default() };
                for (k, v, l) in &fs {
                    s.lines.insert(k.clone(), *l);
                    match k.as_str() {
                        "intent" => s.intent = Some(v.trim().to_string()),
                        "registry" => s.registry = Some(v.trim().to_string()),
                        "members" => s.members = words(v),
                        "default" => s.default = Some(v.trim().to_string()),
                        "through" => s.through = words(v),
                        "contract" => s.contract = Some(v.trim().to_string()),
                        "checked" => s.checked = Some(v.trim().to_string()),
                        _ => s.why = v.clone(),
                    }
                }
                if s.members.is_empty() {
                    f.errors.push((line, format!("`{kw} {id}` has no `members` line")));
                }
                if s.multi && s.members.len() < 2 {
                    f.errors.push((line, format!("`multipath {id}` has {} member; a stage with one implementation is written `stage {id}`", s.members.len())));
                }
                if !s.multi && s.members.len() > 1 {
                    f.errors.push((line, format!("`stage {id}` has {} members; several implementations of one intent are a `multipath`", s.members.len())));
                }
                if f.stages.iter().any(|o| o.id == id) {
                    f.errors.push((line, format!("stage `{id}` is declared twice")));
                }
                f.stages.push(s);
            }
            _ => {}
        }
    }
    for o in &f.order {
        match f.layer(o) {
            None => f.errors.push((f.order_line, format!("`order` names `{o}`, which is not a layer"))),
            Some(l) if l.kind == Kind::Verification => f.errors.push((f.order_line, format!("`order` names the verification layer `{o}`; verification layers may use anything and are left out of the order"))),
            _ => {}
        }
    }
    for l in &f.layers {
        if l.kind != Kind::Verification && !f.order.contains(&l.id) {
            f.errors.push((l.line, format!("layer `{}` is not in the `order` line", l.id)));
        }
    }
    f
}

// ---------------------------------------------------------------------------
// The code: modules, defs, and what each def calls
// ---------------------------------------------------------------------------

/// A module as the CLI hands it over.
#[derive(Clone, Debug, Default)]
pub struct ModSrc {
    /// the repository name `#! use` knows it by: `lib/hf/integrals`
    pub name: String,
    /// the path findings are reported under
    pub path: String,
    pub src: String,
    /// the module's `.intent` file, if it has one
    pub intent: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DKind {
    Def,
    /// a module's top level
    Top,
    /// a `test` block
    Test,
}

/// A reference from a def to another: a call, or the def named as a value.
#[derive(Clone, Debug)]
pub struct Ref {
    pub to: String,
    pub line: usize,
    pub value: bool,
}

#[derive(Clone, Debug)]
pub struct DefInfo {
    pub name: String,
    pub kind: DKind,
    pub module: usize,
    pub line: usize,
    pub refs: Vec<Ref>,
    /// every name read, with its line (globals and locals alike)
    pub reads: Vec<(String, usize)>,
    /// `return [a, b, c]` with every element a name: the registry's list
    pub list: Option<Vec<String>>,
    /// `return f(...)` as the whole body: the def it forwards to
    pub forwards: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ModInfo {
    pub name: String,
    pub path: String,
    pub uses: Vec<(String, usize)>,
    pub layer: Option<usize>,
    pub defs: Vec<usize>,
    pub parse_error: Option<String>,
    pub intent: Option<String>,
    pub src: String,
}

/// Every covered module, parsed.
#[derive(Clone, Debug, Default)]
pub struct World {
    pub mods: Vec<ModInfo>,
    pub defs: Vec<DefInfo>,
    /// def name -> indices of the defs (kind Def) that carry it
    pub by_name: HashMap<String, Vec<usize>>,
    pub mod_index: HashMap<String, usize>,
}

fn walk_expr(e: &Expr, lines: &Lines, refs: &mut Vec<Ref>, reads: &mut Vec<(String, usize)>) {
    match e {
        Expr::Var { name, span } => {
            let l = lines.line(span.start);
            refs.push(Ref { to: name.clone(), line: l, value: true });
            reads.push((name.clone(), l));
        }
        Expr::Call { callee, args, span } => {
            refs.push(Ref { to: callee.clone(), line: lines.line(span.start), value: false });
            for a in args {
                walk_expr(a, lines, refs, reads);
            }
        }
        Expr::ArrLit { elems, .. } => {
            for x in elems {
                walk_expr(x, lines, refs, reads);
            }
        }
        Expr::Index { base, indices, .. } => {
            walk_expr(base, lines, refs, reads);
            for x in indices {
                walk_expr(x, lines, refs, reads);
            }
        }
        Expr::Lambda { body, .. } => walk_expr(body, lines, refs, reads),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk_expr(operand, lines, refs, reads),
        Expr::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, lines, refs, reads);
            walk_expr(rhs, lines, refs, reads);
        }
        Expr::RecordLit { fields, .. } => {
            for (_, _, x) in fields {
                walk_expr(x, lines, refs, reads);
            }
        }
        Expr::Field { base, .. } => walk_expr(base, lines, refs, reads),
        Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
    }
}

fn walk_stmt(s: &Stmt, lines: &Lines, refs: &mut Vec<Ref>, reads: &mut Vec<(String, usize)>) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => walk_expr(value, lines, refs, reads),
        Stmt::SetIndex { base, indices, value, .. } => {
            walk_expr(base, lines, refs, reads);
            for i in indices {
                walk_expr(i, lines, refs, reads);
            }
            walk_expr(value, lines, refs, reads);
        }
        Stmt::SetField { base, value, .. } => {
            walk_expr(base, lines, refs, reads);
            walk_expr(value, lines, refs, reads);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            walk_expr(cond, lines, refs, reads);
            for s in then_body.iter().chain(else_body) {
                walk_stmt(s, lines, refs, reads);
            }
        }
        Stmt::While { cond, body, .. } => {
            walk_expr(cond, lines, refs, reads);
            for s in body {
                walk_stmt(s, lines, refs, reads);
            }
        }
        Stmt::For { bound, body, .. } => {
            walk_expr(bound, lines, refs, reads);
            for s in body {
                walk_stmt(s, lines, refs, reads);
            }
        }
        Stmt::Test { body, .. } => {
            for s in body {
                walk_stmt(s, lines, refs, reads);
            }
        }
        _ => {}
    }
}

/// `return [a, b, c]` as a def's whole body, every element a name.
fn body_list(body: &[Stmt]) -> Option<Vec<String>> {
    let [Stmt::Return { value: Expr::ArrLit { elems, .. }, .. }] = body else { return None };
    let names: Vec<String> = elems.iter().filter_map(|e| if let Expr::Var { name, .. } = e { Some(name.clone()) } else { None }).collect();
    (names.len() == elems.len() && !names.is_empty()).then_some(names)
}

/// `return f(...)` as a def's whole body.
fn body_forward(body: &[Stmt]) -> Option<String> {
    let [Stmt::Return { value: Expr::Call { callee, .. }, .. }] = body else { return None };
    Some(callee.clone())
}

/// The modules a module's import lines (and any `#! use` lines) name, with
/// their lines: `name` is the module's own name, `known` says whether a
/// module of a name exists (crate::imports::known_refs).
pub fn uses_of(name: &str, src: &str, known: &dyn Fn(&str) -> bool) -> Vec<(String, usize)> {
    crate::imports::known_refs(name, src, known)
}

impl World {
    /// Parse every module. A module that does not parse is kept, with its
    /// `#! use` lines and no defs, and reported.
    pub fn new(mods: &[ModSrc]) -> World {
        let mut w = World::default();
        let names: HashSet<&str> = mods.iter().map(|m| m.name.as_str()).collect();
        for m in mods {
            let k = w.mods.len();
            w.mod_index.insert(m.name.clone(), k);
            let mut info = ModInfo { name: m.name.clone(), path: m.path.clone(), uses: uses_of(&m.name, &m.src, &|c: &str| names.contains(c)), layer: None, defs: Vec::new(), parse_error: None, intent: m.intent.clone(), src: m.src.clone() };
            match crate::parse_keeping_rungs(&m.src, SourceId(0)) {
                Err(ds) => info.parse_error = Some(ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; ")),
                Ok(p) => {
                    let lines = Lines::new(&m.src);
                    let mut top_refs = Vec::new();
                    let mut top_reads = Vec::new();
                    let mut items: Vec<&Item> = p.items.iter().collect();
                    // a ladder's rungs hold items too
                    let mut k2 = 0;
                    while k2 < items.len() {
                        if let Item::Rung(r) = items[k2] {
                            items.extend(r.items.iter());
                        }
                        k2 += 1;
                    }
                    for it in items {
                        match it {
                            Item::Def(d) if crate::update::is_expression_def(&d.name) => {}
                            Item::Def(d) => {
                                let mut refs = Vec::new();
                                let mut reads = Vec::new();
                                for s in &d.body {
                                    walk_stmt(s, &lines, &mut refs, &mut reads);
                                }
                                let di = DefInfo { name: d.name.clone(), kind: DKind::Def, module: k, line: lines.line(d.span.start), refs, reads, list: body_list(&d.body), forwards: body_forward(&d.body) };
                                w.by_name.entry(d.name.clone()).or_default().push(w.defs.len());
                                info.defs.push(w.defs.len());
                                w.defs.push(di);
                            }
                            Item::Stmt(Stmt::Test { name, body, span }) => {
                                let mut refs = Vec::new();
                                let mut reads = Vec::new();
                                for s in body {
                                    walk_stmt(s, &lines, &mut refs, &mut reads);
                                }
                                info.defs.push(w.defs.len());
                                w.defs.push(DefInfo { name: format!("test \"{name}\""), kind: DKind::Test, module: k, line: lines.line(span.start), refs, reads, list: None, forwards: None });
                            }
                            Item::Stmt(s) => walk_stmt(s, &lines, &mut top_refs, &mut top_reads),
                            _ => {}
                        }
                    }
                    info.defs.push(w.defs.len());
                    w.defs.push(DefInfo { name: "<top>".into(), kind: DKind::Top, module: k, line: 1, refs: top_refs, reads: top_reads, list: None, forwards: None });
                }
            }
            w.mods.push(info);
        }
        // keep only references to defs
        for d in 0..w.defs.len() {
            let refs: Vec<Ref> = w.defs[d].refs.iter().filter(|r| w.by_name.contains_key(&r.to)).cloned().collect();
            w.defs[d].refs = refs;
        }
        w
    }

    /// The modules a module's `#! use` lines reach, itself included.
    pub fn use_closure(&self, m: usize) -> Vec<usize> {
        let mut out = vec![m];
        let mut k = 0;
        while k < out.len() {
            for (u, _) in &self.mods[out[k]].uses {
                if let Some(&j) = self.mod_index.get(u) {
                    if !out.contains(&j) {
                        out.push(j);
                    }
                }
            }
            k += 1;
        }
        out
    }

    /// The def a name means in a module: its own, else the first module of
    /// its use closure that defines it, else any.
    pub fn resolve(&self, from_mod: usize, name: &str) -> Option<usize> {
        let cands = self.by_name.get(name)?;
        if cands.len() == 1 {
            return Some(cands[0]);
        }
        if let Some(&d) = cands.iter().find(|&&d| self.defs[d].module == from_mod) {
            return Some(d);
        }
        for m in self.use_closure(from_mod) {
            if let Some(&d) = cands.iter().find(|&&d| self.defs[d].module == m) {
                return Some(d);
            }
        }
        cands.first().copied()
    }

    /// A def by name (the first that carries it).
    pub fn def(&self, name: &str) -> Option<&DefInfo> {
        self.by_name.get(name).and_then(|v| v.first()).map(|&d| &self.defs[d])
    }

    fn def_index(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).and_then(|v| v.first()).copied()
    }

    /// What a def forwards to, followed while each def is only `return f(...)`.
    pub fn unwrap(&self, name: &str) -> String {
        let mut cur = name.to_string();
        for _ in 0..8 {
            match self.def(&cur).and_then(|d| d.forwards.clone()) {
                Some(n) if self.by_name.contains_key(&n) => cur = n,
                _ => break,
            }
        }
        cur
    }

    /// The defs a def reaches by calls and function values (itself not
    /// included), not entering `stop`.
    pub fn reach(&self, from: usize, stop: &HashSet<usize>) -> HashSet<usize> {
        let mut seen = HashSet::new();
        let mut q = VecDeque::from([from]);
        while let Some(d) = q.pop_front() {
            let m = self.defs[d].module;
            for r in &self.defs[d].refs {
                if let Some(t) = self.resolve(m, &r.to) {
                    if t != from && !stop.contains(&t) && seen.insert(t) {
                        q.push_back(t);
                    }
                }
            }
        }
        seen
    }

    /// Whether the line is inside the module's text (for messages).
    fn path_of(&self, m: usize) -> &str {
        &self.mods[m].path
    }
}

/// The `what` of a def: its inline `#|` lines, or its `.intent` entry.
pub fn what_of(w: &World, name: &str) -> Option<String> {
    let d = w.def(name)?;
    let m = &w.mods[d.module];
    if let Some(it) = &m.intent {
        if let Ok(f) = crate::lens::IntentFile::parse(it) {
            let (_, entries) = f.model();
            if let Some(e) = entries.iter().find(|e| e.name == name) {
                if let Some(t) = e.field("what") {
                    let s = t.text.trim().to_string();
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
            }
        }
    }
    // the comment lines right above the def: `#|` lines, skipping `#@`, `#:` and `#~`
    let lines: Vec<&str> = m.src.lines().collect();
    let mut k = d.line.saturating_sub(1);
    let mut what: Vec<String> = Vec::new();
    while k > 0 {
        let l = lines[k - 1].trim_start();
        if let Some(rest) = l.strip_prefix("#|") {
            if !rest.starts_with('|') {
                what.push(rest.trim().to_string());
            }
        } else if !(l.starts_with("#@") || l.starts_with("#:") || l.starts_with("#~")) {
            break;
        }
        k -= 1;
    }
    what.reverse();
    let s = what.join(" ").trim().to_string();
    (!s.is_empty()).then_some(s)
}

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

/// One thing the check found.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub path: String,
    pub line: usize,
    /// `verification-dep`, `upward`, `private`, `budget`, `indirect-default`,
    /// `stage`, `unassigned`, `lens` (the file itself), `note`
    pub kind: &'static str,
    /// what the ratchet records: stable across edits that move lines
    pub key: String,
    pub message: String,
    pub detail: Vec<String>,
}

impl Finding {
    pub fn render(&self) -> String {
        let mut s = format!("{}:{}: {}: {}\n", self.path, self.line, self.kind, self.message);
        for d in &self.detail {
            s.push_str(&format!("    {d}\n"));
        }
        s
    }
    pub fn json(&self) -> String {
        format!(
            "{{\"path\":{},\"line\":{},\"kind\":\"{}\",\"key\":{},\"message\":{},\"detail\":[{}]}}",
            json_str(&self.path),
            self.line,
            self.kind,
            json_str(&self.key),
            json_str(&self.message),
            self.detail.iter().map(|d| json_str(d)).collect::<Vec<_>>().join(",")
        )
    }
    /// A note says something worth knowing that is not a violation.
    pub fn is_violation(&self) -> bool {
        !matches!(self.kind, "note" | "lens")
    }
}

/// Assign each module its layer. Returns the findings for modules in no
/// layer or in two.
pub fn assign_layers(a: &ArchFile, w: &mut World, covered: &[String]) -> Vec<Finding> {
    let mut out = Vec::new();
    for m in 0..w.mods.len() {
        let name = w.mods[m].name.clone();
        let hits: Vec<usize> = a.layers.iter().enumerate().filter(|(_, l)| l.modules.iter().any(|p| matches(p, &name))).map(|(k, _)| k).collect();
        w.mods[m].layer = hits.first().copied();
        if hits.len() > 1 {
            out.push(Finding {
                path: a.path.clone(),
                line: a.layers[hits[1]].line,
                kind: "unassigned",
                key: format!("twice {name}"),
                message: format!("`{name}` is in two layers, {}", hits.iter().map(|&k| format!("`{}`", a.layers[k].id)).collect::<Vec<_>>().join(" and ")),
                detail: Vec::new(),
            });
        }
        if hits.is_empty() && covered.contains(&name) {
            out.push(Finding {
                path: w.mods[m].path.clone(),
                line: 1,
                kind: "unassigned",
                key: format!("unassigned {name}"),
                message: format!("`{name}` is under `covers` ({}) but in no layer", a.covers.join(" ")),
                detail: vec![format!("add it to a layer's `modules` in {}", a.path)],
            });
        }
    }
    out
}

fn layer_name(a: &ArchFile, w: &World, m: usize) -> String {
    match w.mods[m].layer {
        Some(l) => format!("{} layer `{}`", a.layers[l].kind.name(), a.layers[l].id),
        None => "no layer".into(),
    }
}

fn kind_of(a: &ArchFile, w: &World, m: usize) -> Option<Kind> {
    w.mods[m].layer.map(|l| a.layers[l].kind)
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The calls from module `m` into module `t`: (caller def index, ref).
fn calls_into(w: &World, m: usize, t: usize) -> Vec<(usize, Ref)> {
    let mut out = Vec::new();
    for &d in &w.mods[m].defs {
        for r in &w.defs[d].refs {
            if let Some(x) = w.resolve(m, &r.to) {
                if w.defs[x].module == t {
                    out.push((d, r.clone()));
                }
            }
        }
    }
    out
}

/// The defs of library and program modules that only verification reaches:
/// every def or test that references them is a test block, a def of a
/// verification module, or another such def (test helpers, lock helpers).
pub fn verification_only(a: &ArchFile, w: &World) -> HashSet<usize> {
    let mut referrers: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (d, di) in w.defs.iter().enumerate() {
        for r in &di.refs {
            if let Some(t) = w.resolve(di.module, &r.to) {
                if t != d {
                    referrers.entry(t).or_default().insert(d);
                }
            }
        }
    }
    let is_ver = |d: usize| w.defs[d].kind == DKind::Test || kind_of(a, w, w.defs[d].module) == Some(Kind::Verification);
    let mut only: HashSet<usize> = HashSet::new();
    loop {
        let mut grew = false;
        for (d, di) in w.defs.iter().enumerate() {
            if di.kind != DKind::Def || only.contains(&d) || is_ver(d) {
                continue;
            }
            let Some(rs) = referrers.get(&d) else { continue };
            if rs.iter().all(|&r| is_ver(r) || only.contains(&r)) {
                only.insert(d);
                grew = true;
            }
        }
        if !grew {
            return only;
        }
    }
}

/// Layer rules: no library module depends on a verification module; no
/// module depends on a later layer.
pub fn check_layers(a: &ArchFile, w: &World) -> Vec<Finding> {
    let mut out = Vec::new();
    let vonly = verification_only(a, w);
    for m in 0..w.mods.len() {
        let Some(mk) = kind_of(a, w, m) else { continue };
        if mk == Kind::Verification {
            continue;
        }
        let my_rank = w.mods[m].layer.and_then(|l| a.rank(&a.layers[l].id));
        // every module this one depends on: its `#! use` lines, and the
        // modules of the defs it calls
        let mut targets: BTreeMap<usize, Option<usize>> = BTreeMap::new();
        for (u, line) in &w.mods[m].uses {
            if let Some(&t) = w.mod_index.get(u) {
                targets.entry(t).or_insert(Some(*line));
            }
        }
        for &d in &w.mods[m].defs {
            for r in &w.defs[d].refs {
                if let Some(x) = w.resolve(m, &r.to) {
                    let t = w.defs[x].module;
                    if t != m {
                        targets.entry(t).or_insert(None);
                    }
                }
            }
        }
        for (t, use_line) in targets {
            let Some(tk) = kind_of(a, w, t) else { continue };
            let calls = calls_into(w, m, t);
            // a reference from a test, or from a def only tests reach, is verification's
            let is_test = |d: usize| w.defs[d].kind == DKind::Test || vonly.contains(&d);
            let from_code: Vec<&(usize, Ref)> = calls.iter().filter(|(d, _)| !is_test(*d)).collect();
            let from_tests = calls.len() - from_code.len();
            let mut helpers: Vec<String> = calls.iter().filter(|(d, _)| vonly.contains(d)).map(|(d, _)| w.defs[*d].name.clone()).collect();
            helpers.sort();
            helpers.dedup();
            let (me, them) = (&w.mods[m].name, &w.mods[t].name);
            let first = |v: &[&(usize, Ref)]| v.first().map(|(d, r)| format!("{}:{} in `{}` ({} `{}`)", w.path_of(m), r.line, w.defs[*d].name, if r.value { "names" } else { "calls" }, r.to));
            let mut detail = Vec::new();
            if !from_code.is_empty() {
                detail.push(format!("{} from its defs, first at {}", plural(from_code.len(), "reference", "references"), first(&from_code).unwrap_or_default()));
                let mut callers: Vec<String> = from_code.iter().map(|(d, _)| w.defs[*d].name.clone()).collect();
                callers.sort();
                callers.dedup();
                detail.push(format!("defs that reach into it: {}", cut_list(&callers, 8)));
            }
            if from_tests > 0 {
                let tv: Vec<&(usize, Ref)> = calls.iter().filter(|(d, _)| is_test(*d)).collect();
                detail.push(format!("{} from its tests, first at {}", plural(from_tests, "reference", "references"), first(&tv).unwrap_or_default()));
                if !helpers.is_empty() {
                    detail.push(format!("test helpers among them (defs only tests reach): {}", cut_list(&helpers, 6)));
                }
            }
            if calls.is_empty() {
                detail.push("nothing in it is called: the import line only loads it".into());
            }
            let line = use_line.or_else(|| calls.first().map(|(_, r)| r.line)).unwrap_or(1);
            let how = if use_line.is_some() { "uses" } else { "calls into, without an import line of its own," };
            if mk == Kind::Library && tk == Kind::Verification {
                let only_tests = from_code.is_empty() && from_tests > 0;
                let mut message = format!("`{me}` ({}) {how} `{them}` ({})", layer_name(a, w, m), layer_name(a, w, t));
                if only_tests {
                    message.push_str(": only its tests need it, and every program that uses this module loads it");
                } else if !from_code.is_empty() {
                    message.push_str(": the library's own code runs verification code");
                } else {
                    message.push_str(": nothing in it is called, the line only loads it");
                }
                out.push(Finding { path: w.path_of(m).to_string(), line, kind: "verification-dep", key: format!("verification-dep {me} -> {them}"), message, detail });
                continue;
            }
            if tk == Kind::Verification {
                continue;
            }
            let t_rank = w.mods[t].layer.and_then(|l| a.rank(&a.layers[l].id));
            if let (Some(mr), Some(tr)) = (my_rank, t_rank) {
                if tr > mr {
                    out.push(Finding {
                        path: w.path_of(m).to_string(),
                        line,
                        kind: "upward",
                        key: format!("upward {me} -> {them}"),
                        message: format!("`{me}` ({}) {how} `{them}` ({}), a later layer", layer_name(a, w, m), layer_name(a, w, t)),
                        detail,
                    });
                }
            }
        }
    }
    out
}

fn cut_list(xs: &[String], n: usize) -> String {
    if xs.len() <= n {
        xs.join(", ")
    } else {
        format!("{}, and {} more", xs[..n].join(", "), xs.len() - n)
    }
}

/// Private names: a library or program module other than the owners that
/// reads one.
pub fn check_private(a: &ArchFile, w: &World) -> Vec<Finding> {
    let mut out = Vec::new();
    for p in &a.private {
        for m in 0..w.mods.len() {
            if p.owners.iter().any(|o| matches(o, &w.mods[m].name)) {
                continue;
            }
            if !matches!(kind_of(a, w, m), Some(Kind::Library) | Some(Kind::Program)) {
                continue;
            }
            let mut hits: Vec<(String, usize)> = Vec::new();
            for &d in &w.mods[m].defs {
                for (n, l) in &w.defs[d].reads {
                    if p.names.iter().any(|pat| name_matches(pat, n)) {
                        hits.push((n.clone(), *l));
                    }
                }
            }
            if hits.is_empty() {
                continue;
            }
            hits.sort_by_key(|h| h.1);
            let mut names: Vec<String> = hits.iter().map(|h| h.0.clone()).collect();
            names.sort();
            names.dedup();
            out.push(Finding {
                path: w.path_of(m).to_string(),
                line: hits[0].1,
                kind: "private",
                key: format!("private {} reads {}", w.mods[m].name, p.names.join(" ")),
                message: format!("`{}` reads {} that {} owns{}", w.mods[m].name, cut_list(&names, 6), p.owners.join(" and "), if p.why.is_empty() { String::new() } else { format!(": {}", first_clause(&p.why)) }),
                detail: vec![format!("{} of them, at lines {}", hits.len(), cut_list(&hits.iter().map(|h| h.1.to_string()).collect::<Vec<_>>(), 10))],
            });
        }
    }
    out
}

fn first_clause(s: &str) -> String {
    s.split(';').next().unwrap_or(s).trim().to_string()
}

/// The distinct defs of library and program modules that call or name `name`.
pub fn fan_in(a: &ArchFile, w: &World, name: &str) -> Vec<String> {
    let Some(target) = w.def_index(name) else { return Vec::new() };
    let mut out = BTreeSet::new();
    for (d, di) in w.defs.iter().enumerate() {
        if di.kind != DKind::Def || d == target {
            continue;
        }
        if !matches!(kind_of(a, w, di.module), Some(Kind::Library) | Some(Kind::Program)) {
            continue;
        }
        if di.refs.iter().any(|r| r.to == name && w.resolve(di.module, &r.to) == Some(target)) {
            out.insert(di.name.clone());
        }
    }
    out.into_iter().collect()
}

/// The shortest chain of calls from one def to another, both included.
pub fn shortest_path(w: &World, from: &str, to: &str) -> Option<Vec<String>> {
    let s = w.def_index(from)?;
    let t = w.def_index(to)?;
    let mut prev: HashMap<usize, usize> = HashMap::new();
    let mut q = VecDeque::from([s]);
    let mut seen = HashSet::from([s]);
    while let Some(d) = q.pop_front() {
        if d == t {
            let mut path = vec![t];
            let mut c = t;
            while let Some(&p) = prev.get(&c) {
                path.push(p);
                c = p;
            }
            path.reverse();
            return Some(path.into_iter().map(|k| w.defs[k].name.clone()).collect());
        }
        let m = w.defs[d].module;
        for r in &w.defs[d].refs {
            if r.value {
                continue;
            }
            if let Some(x) = w.resolve(m, &r.to) {
                if seen.insert(x) {
                    prev.insert(x, d);
                    q.push_back(x);
                }
            }
        }
    }
    None
}

/// The def budgets: fan-in and reach.
pub fn check_def_budgets(a: &ArchFile, w: &World, values: &mut BTreeMap<String, usize>) -> Vec<Finding> {
    let mut out = Vec::new();
    for b in &a.budgets {
        let Some(def) = &b.def else { continue };
        let Some(di) = w.def(def) else {
            out.push(Finding { path: a.path.clone(), line: b.line, kind: "lens", key: format!("lens budget {def}"), message: format!("`budget def {def}`: no def by that name in the covered modules"), detail: Vec::new() });
            continue;
        };
        for l in &b.limits {
            match l.metric.as_str() {
                "fanin" => {
                    let callers = fan_in(a, w, def);
                    values.insert(format!("budget def {def} fanin"), callers.len());
                    if callers.len() > l.value {
                        out.push(Finding {
                            path: w.path_of(di.module).to_string(),
                            line: di.line,
                            kind: "budget",
                            key: format!("budget def {def} fanin"),
                            message: format!("`{def}` has {} callers; its budget is {}{}", callers.len(), l.value, if b.why.is_empty() { String::new() } else { format!(" ({})", b.why) }),
                            detail: vec![format!("callers: {}", callers.join(", ")), format!("budget at {}:{}", a.path, l.line)],
                        });
                    }
                }
                "reach" => {
                    let to = l.arg.clone().unwrap_or_default();
                    let key = format!("budget def {def} reach {to}");
                    match shortest_path(w, def, &to) {
                        None => out.push(Finding { path: a.path.clone(), line: l.line, kind: "lens", key, message: format!("`{def}` does not reach `{to}` by calls"), detail: Vec::new() }),
                        Some(p) => {
                            let n = p.len() - 1;
                            values.insert(key.clone(), n);
                            if n > l.value {
                                out.push(Finding {
                                    path: w.path_of(di.module).to_string(),
                                    line: di.line,
                                    kind: "budget",
                                    key,
                                    message: format!("`{def}` reaches `{to}` in {n} calls; its budget is {}{}", l.value, if b.why.is_empty() { String::new() } else { format!(" ({})", b.why) }),
                                    detail: vec![format!("the shortest chain: {}", p.join(" -> ")), format!("budget at {}:{}", a.path, l.line)],
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Stages and multi-paths
// ---------------------------------------------------------------------------

/// A stage as measured: its independent families and what the code says.
#[derive(Clone, Debug, Default)]
pub struct StageFacts {
    pub id: String,
    /// the members grouped into independent families (each family: member names)
    pub families: Vec<Vec<String>>,
    /// for members in one family: (a, b, a few defs both reach)
    pub ties: Vec<(String, String, Vec<String>)>,
    /// the default is called by name from library code (the production path is direct)
    pub direct: bool,
    /// the library defs that call the default by name
    pub direct_callers: Vec<String>,
    /// the tolerance the contract names, and its value
    pub tolerance: Option<(String, f64)>,
    pub intent_what: Option<String>,
    pub citations: Vec<(String, String)>,
}

/// The defs of the stage machinery of every stage: registries, drivers
/// (`through`) and the intent defs of the other stages, plus the members of
/// the other stages.
fn machinery(a: &ArchFile, w: &World, except: &str) -> HashSet<usize> {
    let mut out = HashSet::new();
    for s in &a.stages {
        let mut names: Vec<&String> = s.through.iter().chain(s.registry.iter()).collect();
        if s.id != except {
            names.extend(s.members.iter());
            names.extend(s.intent.iter());
        }
        for n in names {
            if let Some(v) = w.by_name.get(n) {
                out.extend(v.iter().copied());
            }
        }
    }
    out
}

/// Measure one stage: families, ties, whether its default is called directly.
pub fn stage_facts(a: &ArchFile, w: &World, s: &Stage) -> StageFacts {
    let mut f = StageFacts { id: s.id.clone(), ..Default::default() };
    // defs not to enter: the stage machinery, and the shared modules other
    // than the ones this stage's members live in
    let mut stop = machinery(a, w, &s.id);
    let member_mods: HashSet<usize> = s.members.iter().filter_map(|m| w.def(&w.unwrap(m)).map(|d| d.module)).chain(s.members.iter().filter_map(|m| w.def(m).map(|d| d.module))).collect();
    for sh in &a.shared {
        for (k, m) in w.mods.iter().enumerate() {
            if sh.modules.iter().any(|p| matches(p, &m.name)) && !member_mods.contains(&k) {
                stop.extend(m.defs.iter().copied());
            }
        }
    }
    let mut own: Vec<usize> = Vec::new();
    for m in &s.members {
        if let Some(d) = w.def_index(m) {
            own.push(d);
        }
    }
    for &d in &own {
        stop.remove(&d);
    }
    // each member's code: itself, what it forwards to, and what they reach
    let mut code: Vec<(String, HashSet<usize>)> = Vec::new();
    for m in &s.members {
        let Some(d) = w.def_index(m) else { continue };
        let mut set = w.reach(d, &stop);
        set.insert(d);
        code.push((m.clone(), set));
    }
    // union-find over members that share a def
    let n = code.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        p[x] = r;
        r
    }
    for i in 0..n {
        for j in i + 1..n {
            let mut common: Vec<String> = code[i].1.intersection(&code[j].1).map(|&d| w.defs[d].name.clone()).collect();
            if common.is_empty() {
                continue;
            }
            common.sort();
            common.truncate(4);
            f.ties.push((code[i].0.clone(), code[j].0.clone(), common));
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj {
                parent[rj] = ri;
            }
        }
    }
    let mut fams: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        fams.entry(r).or_default().push(code[i].0.clone());
    }
    f.families = fams.into_values().collect();
    // direct: the default (or what it forwards to) is called by name from
    // library code other than its registry and the stage's drivers
    if let Some(def) = &s.default {
        let targets: HashSet<String> = [def.clone(), w.unwrap(def)].into_iter().collect();
        let skip: HashSet<String> = s.through.iter().chain(s.registry.iter()).cloned().collect();
        let mut callers = BTreeSet::new();
        for di in &w.defs {
            if di.kind != DKind::Def || skip.contains(&di.name) || targets.contains(&di.name) {
                continue;
            }
            if kind_of(a, w, di.module) != Some(Kind::Library) {
                continue;
            }
            if di.refs.iter().any(|r| !r.value && targets.contains(&r.to)) {
                callers.insert(di.name.clone());
            }
        }
        f.direct = !callers.is_empty();
        f.direct_callers = callers.into_iter().collect();
    }
    if let Some(c) = &s.contract {
        let ws = words(c);
        if ws.len() == 3 && ws[1] == "within" {
            if let Some(v) = constant_value(w, &ws[2]).or_else(|| ws[2].parse::<f64>().ok()) {
                f.tolerance = Some((ws[2].clone(), v));
            }
        } else if let Some(k) = ws.iter().position(|x| x == "within") {
            if let Some(v) = ws.get(k + 1).and_then(|x| x.parse::<f64>().ok()) {
                f.tolerance = Some((ws[k + 1].clone(), v));
            }
        }
    }
    f.intent_what = s.intent_def().and_then(|d| what_of(w, d));
    for text in s.checked.iter().chain(s.contract.iter()) {
        for kind in ["test", "program"] {
            for c in crate::lens::citations(text, kind) {
                if !f.citations.contains(&(kind.to_string(), c.clone())) {
                    f.citations.push((kind.to_string(), c));
                }
            }
        }
    }
    f
}

/// A top-level constant's value: `NAME = <number>` in any covered module.
pub fn constant_value(w: &World, name: &str) -> Option<f64> {
    for m in &w.mods {
        for l in m.src.lines() {
            if let Some(rest) = l.strip_prefix(name) {
                if let Some(v) = rest.trim_start().strip_prefix('=') {
                    let v = v.split('#').next().unwrap_or("").trim();
                    if let Ok(x) = v.parse::<f64>() {
                        return Some(x);
                    }
                }
            }
        }
    }
    None
}

/// The tests of every covered module, by name: (module index, line).
pub fn tests_of(w: &World) -> HashMap<String, (usize, usize)> {
    let mut out = HashMap::new();
    for d in &w.defs {
        if d.kind == DKind::Test {
            let name = d.name.trim_start_matches("test \"").trim_end_matches('"').to_string();
            out.entry(name).or_insert((d.module, d.line));
        }
    }
    out
}

/// Check the stage declarations against the code.
pub fn check_stages(a: &ArchFile, w: &World, facts: &[StageFacts]) -> Vec<Finding> {
    let mut out = Vec::new();
    let tests = tests_of(w);
    let lens = |line: usize, key: String, message: String| Finding { path: a.path.clone(), line, kind: "stage", key, message, detail: Vec::new() };
    for (s, f) in a.stages.iter().zip(facts) {
        let what = if s.multi { "multipath" } else { "stage" };
        // every def it names exists
        let mut named: Vec<(&str, &String)> = s.members.iter().map(|m| ("members", m)).collect();
        named.extend(s.through.iter().map(|t| ("through", t)));
        for (k, v) in [("intent", &s.intent), ("registry", &s.registry), ("default", &s.default)] {
            if let Some(v) = v {
                named.push((k, v));
            }
        }
        for (field, name) in named {
            if w.def(name).is_none() {
                out.push(lens(s.at(field), format!("stage {} {field} {name}", s.id), format!("`{what} {}` names `{name}` ({field}), which no covered module defines", s.id)));
            }
        }
        if let Some(d) = &s.default {
            if !s.members.contains(d) {
                out.push(lens(s.at("default"), format!("stage {} default", s.id), format!("`{what} {}`: the default `{d}` is not one of its members", s.id)));
            }
        }
        if let Some(r) = &s.registry {
            if let Some(di) = w.def(r) {
                match &di.list {
                    None => out.push(lens(s.at("registry"), format!("stage {} registry", s.id), format!("`{what} {}`: `{r}` is not a registry (a def whose body is `return [f, g, ...]`)", s.id))),
                    Some(list) if list != &s.members => out.push(Finding {
                        path: a.path.clone(),
                        line: s.at("members"),
                        kind: "stage",
                        key: format!("stage {} members", s.id),
                        message: format!("`{what} {}` lists members {}; the registry `{r}` holds {}", s.id, s.members.join(" "), list.join(" ")),
                        detail: vec![format!("{}:{}", w.path_of(di.module), di.line)],
                    }),
                    _ => {}
                }
            }
        }
        if s.multi {
            match &s.contract {
                None => out.push(lens(s.line, format!("stage {} contract", s.id), format!("`multipath {}` has no `contract`: what must its members agree on, and how closely?", s.id))),
                Some(c) => {
                    let ws = words(c);
                    if ws.len() == 3 && ws[1] == "within" {
                        match w.def(&ws[0]) {
                            None => out.push(lens(s.at("contract"), format!("stage {} contract def", s.id), format!("`multipath {}`: the contract def `{}` does not exist", s.id, ws[0]))),
                            Some(cd) => {
                                if f.tolerance.is_none() {
                                    out.push(lens(s.at("contract"), format!("stage {} contract tolerance", s.id), format!("`multipath {}`: `{}` is neither a top-level constant nor a number", s.id, ws[2])));
                                }
                                let reads_tol = cd.reads.iter().any(|(n, _)| n == &ws[2]);
                                let names_reg = s.registry.as_ref().map_or(true, |r| cd.refs.iter().any(|x| &x.to == r));
                                if !reads_tol || !names_reg {
                                    out.push(lens(
                                        s.at("contract"),
                                        format!("stage {} contract body", s.id),
                                        format!("`multipath {}`: the contract def `{}` does not {}", s.id, ws[0], if !reads_tol { format!("read `{}`", ws[2]) } else { "run the registry".to_string() }),
                                    ));
                                }
                            }
                        }
                    } else if crate::lens::citations(c, "test").is_empty() {
                        out.push(lens(s.at("contract"), format!("stage {} contract citation", s.id), format!("`multipath {}`: a contract written in words cites the test that checks it: `[test \"...\"]`", s.id)));
                    }
                }
            }
            if s.checked.is_none() {
                out.push(lens(s.line, format!("stage {} checked", s.id), format!("`multipath {}` has no `checked` line: which tests run its members side by side?", s.id)));
            }
        }
        for (kind, c) in &f.citations {
            let ok = match kind.as_str() {
                "test" => tests.contains_key(c),
                _ => w.mod_index.contains_key(c.trim_end_matches(".viv")),
            };
            if !ok {
                out.push(lens(s.at("checked"), format!("stage {} cites {kind} {c}", s.id), format!("`{what} {}` cites [{kind} \"{c}\"], which does not exist in the covered modules", s.id)));
            }
        }
        if s.intent_def().is_some() && f.intent_what.is_none() && w.def(s.intent_def().unwrap_or("")).is_some() {
            out.push(Finding {
                path: a.path.clone(),
                line: s.at("intent"),
                kind: "note",
                key: format!("stage {} intent", s.id),
                message: format!("`{what} {}`: its intent def `{}` has no `what`, so the stage has no intent text of its own", s.id, s.intent_def().unwrap_or("")),
                detail: Vec::new(),
            });
        }
        // the production path: the default is called directly by the library.
        // A `through` line says it is not: the drivers named there stand
        // between the library and the member.
        if s.multi {
            if let Some(d) = &s.default {
                if w.def(d).is_some() && !s.through.is_empty() {
                    let drivers = format!("{} and the registry `{}`", s.through.join(" -> "), s.registry.clone().unwrap_or_default());
                    // where: the first library call of the first driver
                    let t0 = &s.through[0];
                    let at = w
                        .defs
                        .iter()
                        .filter(|x| x.kind == DKind::Def && kind_of(a, w, x.module) == Some(Kind::Library) && !s.through.contains(&x.name))
                        .find_map(|x| x.refs.iter().find(|r| &r.to == t0 && !r.value).map(|r| (x.module, r.line)));
                    match at {
                        None => out.push(lens(s.at("through"), format!("stage {} through", s.id), format!("`multipath {}`: no library def calls `{t0}`, the first def its `through` line names", s.id))),
                        Some((m, line)) => out.push(Finding {
                            path: w.path_of(m).to_string(),
                            line,
                            kind: "indirect-default",
                            key: format!("indirect-default {}", s.id),
                            message: format!("the production path reaches `{d}`, the default of `{}`, only as a registry member, through {drivers}", s.id),
                            detail: vec![format!("the library should call `{d}` directly; the cross-check of the other members belongs in the verification layer")],
                        }),
                    }
                } else if w.def(d).is_some() && !f.direct {
                    out.push(lens(
                        s.at("default"),
                        format!("stage {} direct", s.id),
                        format!("`multipath {}` has no `through` line, so its default `{d}` should be called directly, but no library def calls it by name", s.id),
                    ));
                }
            }
        }
    }
    // registries no stage declares
    let declared: HashSet<&String> = a.stages.iter().filter_map(|s| s.registry.as_ref()).collect();
    for d in &w.defs {
        let Some(list) = &d.list else { continue };
        if d.kind != DKind::Def || list.len() < 2 || declared.contains(&d.name) || kind_of(a, w, d.module) != Some(Kind::Library) {
            continue;
        }
        out.push(Finding {
            path: w.path_of(d.module).to_string(),
            line: d.line,
            kind: "note",
            key: format!("undeclared registry {}", d.name),
            message: format!("`{}` is a registry of {} ({}) that no `multipath` declares", d.name, list.len(), cut_list(list, 5)),
            detail: Vec::new(),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Programs: what one program loads, runs and reaches
// ---------------------------------------------------------------------------

/// A run's call graph, from the interpreter's `__graph__` line.
#[derive(Clone, Debug, Default)]
pub struct RunGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String, u64)>,
    pub passed: Vec<String>,
    pub failed: Vec<String>,
}

impl RunGraph {
    /// Read a run's output: its `__graph__` line and its PASS and FAIL lines.
    pub fn of_output(out: &str) -> Option<RunGraph> {
        let line = out.lines().find(|l| l.starts_with("__graph__ "))?;
        let v = crate::story::parse_json(&line["__graph__ ".len()..]).ok()?;
        let mut g = RunGraph::default();
        if let Some(crate::story::Json::Arr(ns)) = v.get("nodes") {
            g.nodes = ns.iter().filter_map(|n| n.str().map(String::from)).collect();
        }
        if let Some(crate::story::Json::Arr(es)) = v.get("edges") {
            for e in es {
                if let (Some(f), Some(t)) = (e.get("from").and_then(|x| x.str()), e.get("to").and_then(|x| x.str())) {
                    g.edges.push((f.to_string(), t.to_string(), e.get("count").and_then(|x| x.num()).unwrap_or(0.0) as u64));
                }
            }
        }
        for l in out.lines() {
            if let Some(t) = l.strip_prefix("PASS ") {
                g.passed.push(t.trim_end_matches(" (no assertions)").to_string());
            } else if let Some(t) = l.strip_prefix("FAIL ") {
                g.failed.push(t.split(" -- ").next().unwrap_or(t).to_string());
            }
        }
        Some(g)
    }

    fn callees(&self) -> BTreeMap<&str, Vec<&str>> {
        let mut m: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (f, t, _) in &self.edges {
            if f != t {
                m.entry(f.as_str()).or_default().push(t.as_str());
            }
        }
        for v in m.values_mut() {
            v.sort();
            v.dedup();
        }
        m
    }

    /// The longest chain of distinct defs from `<main>` through the run's
    /// call edges (the depth of the call graph; each def once on a chain).
    pub fn deepest(&self) -> Vec<String> {
        let callees = self.callees();
        let mut memo: HashMap<&str, Vec<&str>> = HashMap::new();
        fn go<'a>(n: &'a str, callees: &BTreeMap<&'a str, Vec<&'a str>>, on: &mut HashSet<&'a str>, memo: &mut HashMap<&'a str, Vec<&'a str>>) -> Vec<&'a str> {
            if let Some(p) = memo.get(n) {
                if p.iter().all(|x| !on.contains(x)) {
                    return p.clone();
                }
            }
            on.insert(n);
            let mut best: Vec<&str> = Vec::new();
            for &c in callees.get(n).map(Vec::as_slice).unwrap_or(&[]) {
                if on.contains(c) {
                    continue;
                }
                let p = go(c, callees, on, memo);
                if p.len() > best.len() {
                    best = p;
                }
            }
            on.remove(n);
            let mut out = vec![n];
            out.extend(best);
            memo.insert(n, out.clone());
            out
        }
        let mut on = HashSet::new();
        go(crate::parser::MAIN_DEF, &callees, &mut on, &mut memo).into_iter().map(String::from).collect()
    }
}

/// What one program loads.
#[derive(Clone, Debug, Default)]
pub struct ProgramFacts {
    /// the program's repository name
    pub name: String,
    /// the root and every module its `#! use` lines reach, in load order
    pub modules: Vec<String>,
    /// the defs of the expanded program, as the compiler counts them (record
    /// updates and other generated defs included)
    pub expanded_defs: usize,
    pub run: Option<RunGraph>,
}

/// One stage as one program meets it.
#[derive(Clone, Debug, Default)]
pub struct StageInProgram {
    pub id: String,
    pub title: String,
    pub multi: bool,
    pub members: Vec<String>,
    pub default: String,
    /// members that ran in the program's run
    pub ran: Vec<String>,
    pub families: usize,
    /// members seen agreeing (measured by `--verify`, or recorded), and the
    /// families among them; None when never measured
    pub agreeing: Option<Vec<String>>,
    pub agreeing_families: Option<usize>,
    pub on_path: bool,
    /// the production path calls the default directly (no `through` drivers)
    pub direct: bool,
    pub through: Vec<String>,
    pub contract: String,
    pub intent: String,
    pub what: String,
}

impl StageInProgram {
    /// A stage is robust when at least two independent implementations were seen agreeing.
    pub fn robust(&self) -> bool {
        self.agreeing_families.unwrap_or(0) >= 2
    }
}

/// The per-program measurements of `viv arch report`.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub program: String,
    pub arch: String,
    pub modules: usize,
    pub defs: usize,
    pub expanded_defs: usize,
    pub ran: Option<usize>,
    /// per layer: (layer, modules, defs, defs that ran)
    pub layers: Vec<(String, usize, usize, Option<usize>)>,
    /// modules loaded of which nothing ran: (module, layer, defs)
    pub idle: Vec<(String, String, usize)>,
    pub deepest: Vec<String>,
    /// defs on the deepest chain that are stage machinery (drivers, registries)
    pub indirection: Vec<String>,
    /// (def, distinct callers) in the run, the most first
    pub fan_in: Vec<(String, usize)>,
    pub fan_out: Vec<(String, usize)>,
    /// (from module, to module, distinct def pairs, calls)
    pub pairs: Vec<(String, String, usize, u64)>,
    /// (prefix, modules whose defs carry it)
    pub prefixes: Vec<(String, Vec<String>)>,
    pub named_by_prefix: (usize, usize),
    pub stages: Vec<StageInProgram>,
    pub findings: Vec<Finding>,
}

fn prefix_of(name: &str) -> Option<String> {
    let k = name.find('_')?;
    (k > 0 && k < name.len() - 1).then(|| name[..=k].to_string())
}

/// Recorded agreement, per stage: the members seen agreeing (from the ratchet
/// or a `--verify` run).
pub type Agreement = HashMap<String, Vec<String>>;

/// Measure a program against the lens.
pub fn report(a: &ArchFile, w: &World, p: &ProgramFacts, facts: &[StageFacts], agreement: &Agreement, values: &mut BTreeMap<String, usize>) -> Report {
    let mut r = Report { program: p.name.clone(), arch: a.path.clone(), modules: p.modules.len(), expanded_defs: p.expanded_defs, ..Default::default() };
    let mods: Vec<usize> = p.modules.iter().filter_map(|m| w.mod_index.get(m).copied()).collect();
    let ran: HashSet<&str> = p.run.as_ref().map(|g| g.nodes.iter().map(String::as_str).collect()).unwrap_or_default();
    let has_run = p.run.is_some();
    let def_names = |m: usize| -> Vec<&str> { w.mods[m].defs.iter().filter(|&&d| w.defs[d].kind == DKind::Def).map(|&d| w.defs[d].name.as_str()).collect() };
    r.defs = mods.iter().map(|&m| def_names(m).len()).sum();
    r.ran = p.run.as_ref().map(|g| g.nodes.len());
    // per layer
    let mut per: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new();
    for &m in &mods {
        let (lid, order) = match w.mods[m].layer {
            Some(l) => (a.layers[l].id.clone(), l),
            None => ("(no layer)".into(), usize::MAX),
        };
        let names = def_names(m);
        let nran = names.iter().filter(|n| ran.contains(*n)).count();
        let e = per.entry(lid.clone()).or_insert((order, 0, 0, 0));
        e.1 += 1;
        e.2 += names.len();
        e.3 += nran;
        if has_run && nran == 0 && w.mods[m].name != p.name {
            r.idle.push((w.mods[m].name.clone(), lid, names.len()));
        }
    }
    let mut layers: Vec<(usize, String, usize, usize, usize)> = per.into_iter().map(|(k, v)| (v.0, k, v.1, v.2, v.3)).collect();
    layers.sort();
    r.layers = layers.into_iter().map(|(_, k, m, d, n)| (k, m, d, has_run.then_some(n))).collect();
    // the machinery: drivers and registries of every stage
    let mut mach: HashSet<&str> = HashSet::new();
    for s in &a.stages {
        mach.extend(s.through.iter().map(String::as_str));
        mach.extend(s.registry.iter().map(String::as_str));
    }
    if let Some(g) = &p.run {
        r.deepest = g.deepest();
        r.indirection = r.deepest.iter().filter(|d| mach.contains(d.as_str())).cloned().collect();
        let mut fin: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut fout: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut pairs: BTreeMap<(String, String), (usize, u64)> = BTreeMap::new();
        let module_of = |d: &str| -> Option<String> { w.by_name.get(d).and_then(|v| v.iter().find(|&&k| mods.contains(&w.defs[k].module)).or(v.first())).map(|&k| w.mods[w.defs[k].module].name.clone()) };
        for (f, t, c) in &g.edges {
            if f == t {
                continue;
            }
            fin.entry(t).or_default().insert(f);
            fout.entry(f).or_default().insert(t);
            let fm = if f == crate::parser::MAIN_DEF { Some(p.name.clone()) } else { module_of(f) };
            if let (Some(fm), Some(tm)) = (fm, module_of(t)) {
                if fm != tm {
                    let e = pairs.entry((fm, tm)).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += c;
                }
            }
        }
        let top = |m: BTreeMap<&str, BTreeSet<&str>>| {
            let mut v: Vec<(String, usize)> = m.into_iter().map(|(k, s)| (k.to_string(), s.len())).collect();
            v.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
            v.truncate(8);
            v
        };
        r.fan_in = top(fin);
        r.fan_out = top(fout);
        let mut pv: Vec<(String, String, usize, u64)> = pairs.into_iter().map(|((a, b), (n, c))| (a, b, n, c)).collect();
        pv.sort_by(|x, y| y.2.cmp(&x.2).then(y.3.cmp(&x.3)).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
        r.pairs = pv;
    }
    // prefixes standing in for namespaces
    let mut by_prefix: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let (mut with, mut all) = (0, 0);
    for &m in &mods {
        for n in def_names(m) {
            all += 1;
            if let Some(px) = prefix_of(n) {
                with += 1;
                by_prefix.entry(px).or_default().insert(w.mods[m].name.clone());
            }
        }
    }
    r.named_by_prefix = (with, all);
    r.prefixes = by_prefix.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();
    // stages
    for (s, f) in a.stages.iter().zip(facts) {
        let in_prog = s.members.iter().chain(s.intent.iter()).any(|m| w.def(m).map_or(false, |d| mods.contains(&d.module)));
        if !in_prog {
            continue;
        }
        // a member ran when it did; the default also when the def it only
        // forwards to did (`int_contract_closed` is `return bas_contract(...)`,
        // and the library calls `bas_contract`)
        let impl_ran = |m: &String| ran.contains(m.as_str()) || (s.default.as_ref() == Some(m) && ran.contains(w.unwrap(m).as_str()));
        let ran_members: Vec<String> = s.members.iter().filter(|m| impl_ran(m)).cloned().collect();
        let on_path = !ran_members.is_empty() || s.intent.as_ref().map_or(false, |i| ran.contains(i.as_str())) || s.registry.as_ref().map_or(false, |i| ran.contains(i.as_str()));
        let agreeing = agreement.get(&s.id).cloned();
        let agreeing_families = agreeing.as_ref().map(|ag| f.families.iter().filter(|fam| fam.iter().any(|m| ag.contains(m))).count());
        r.stages.push(StageInProgram {
            id: s.id.clone(),
            title: s.title.clone(),
            multi: s.multi,
            members: s.members.clone(),
            default: s.default.clone().or_else(|| s.members.first().cloned()).unwrap_or_default(),
            ran: ran_members,
            families: f.families.len(),
            agreeing,
            agreeing_families,
            on_path: on_path || !has_run,
            direct: !s.multi || s.through.is_empty(),
            through: s.through.clone(),
            contract: s.contract.clone().unwrap_or_default(),
            intent: s.intent_def().unwrap_or("").to_string(),
            what: f.intent_what.clone().unwrap_or_default(),
        });
    }
    // the program's budgets
    for b in &a.budgets {
        if b.program.as_deref() != Some(p.name.as_str()) {
            continue;
        }
        for l in &b.limits {
            let key = format!("budget program {} {}", p.name, l.metric);
            let (have, what) = match l.metric.as_str() {
                "modules" => (Some(r.modules), "modules loaded"),
                "defs" => (Some(if r.expanded_defs > 0 { r.expanded_defs } else { r.defs }), "defs loaded"),
                _ => (p.run.as_ref().map(|_| r.deepest.len().saturating_sub(1)), "calls on its deepest chain"),
            };
            let Some(have) = have else { continue };
            values.insert(key.clone(), have);
            if have > l.value {
                let mut detail = vec![format!("budget at {}:{}", a.path, l.line)];
                match l.metric.as_str() {
                    "modules" | "defs" => {
                        let idle_defs: usize = r.idle.iter().map(|x| x.2).sum();
                        if has_run {
                            detail.insert(0, format!("{} of the {} modules ran nothing ({} defs); the run called {} defs", r.idle.len(), r.modules, idle_defs, r.ran.unwrap_or(0)));
                        }
                    }
                    _ => detail.insert(0, format!("the chain: {}", r.deepest.join(" -> "))),
                }
                r.findings.push(Finding {
                    path: format!("{}.viv", p.name),
                    line: 1,
                    kind: "budget",
                    key,
                    message: format!("`{}` has {have} {what}; its budget is {}{}", p.name, l.value, if b.why.is_empty() { String::new() } else { format!(" ({})", first_clause(&b.why)) }),
                    detail,
                });
            }
        }
    }
    r
}

/// The program-level robustness: (stages on the path with >= 2 agreeing
/// independent paths, stages on the path).
pub fn robustness(r: &Report) -> (usize, usize) {
    let on: Vec<&StageInProgram> = r.stages.iter().filter(|s| s.on_path).collect();
    (on.iter().filter(|s| s.robust()).count(), on.len())
}

fn fmt_int(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `viv arch report` as text.
pub fn report_text(r: &Report) -> String {
    let mut s = String::new();
    s.push_str(&format!("{} against {}\n\n", r.program, r.arch));
    s.push_str(&format!("loaded   {} modules, {} defs as the compiler counts them ({} written in the modules, the rest generated)\n", r.modules, fmt_int(r.expanded_defs), fmt_int(r.defs)));
    match r.ran {
        Some(n) => s.push_str(&format!("ran      {} defs ({:.0}% of those loaded)\n", n, 100.0 * n as f64 / r.expanded_defs.max(r.defs).max(1) as f64)),
        None => s.push_str("ran      not measured (--static)\n"),
    }
    s.push_str("\nby layer                  modules     defs      ran\n");
    for (l, m, d, n) in &r.layers {
        s.push_str(&format!("  {:<22} {:>8} {:>8} {:>8}\n", l, m, fmt_int(*d), n.map(fmt_int).unwrap_or_else(|| "-".into())));
    }
    if !r.idle.is_empty() {
        s.push_str(&format!("\nloaded, and nothing in them ran ({} modules, {} defs):\n", r.idle.len(), fmt_int(r.idle.iter().map(|x| x.2).sum())));
        let mut by: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (m, l, _) in &r.idle {
            by.entry(l.as_str()).or_default().push(m.as_str());
        }
        for (l, ms) in by {
            let names: Vec<String> = ms.iter().map(|m| m.to_string()).collect();
            s.push_str(&format!("  {l}: {}\n", cut_list(&names, 10)));
        }
    }
    if !r.deepest.is_empty() {
        s.push_str(&format!("\ndeepest call chain, {} calls:\n  {}\n", r.deepest.len() - 1, r.deepest.join(" -> ")));
        if !r.indirection.is_empty() {
            s.push_str(&format!("  of which stage machinery (drivers and registries): {}\n", r.indirection.join(", ")));
        }
    }
    if !r.fan_in.is_empty() {
        s.push_str(&format!("\nmost callers (fan-in):   {}\n", r.fan_in.iter().map(|(d, n)| format!("{d} {n}")).collect::<Vec<_>>().join(", ")));
        s.push_str(&format!("most callees (fan-out):  {}\n", r.fan_out.iter().map(|(d, n)| format!("{d} {n}")).collect::<Vec<_>>().join(", ")));
    }
    if !r.pairs.is_empty() {
        s.push_str(&format!("\ncalls between modules ({} pairs; distinct caller-callee edges, calls):\n", r.pairs.len()));
        for (a, b, n, c) in r.pairs.iter().take(15) {
            s.push_str(&format!("  {a} -> {b}  {n} edges, {} calls\n", fmt_int(*c as usize)));
        }
        if r.pairs.len() > 15 {
            s.push_str(&format!("  and {} more pairs\n", r.pairs.len() - 15));
        }
    }
    let shared: Vec<&(String, Vec<String>)> = r.prefixes.iter().filter(|p| p.1.len() > 1).collect();
    s.push_str(&format!(
        "\nnames: {} of {} defs carry a prefix (`den_`, `accel_`, ...), {} prefixes in all; the prefixes stand in for namespaces\n",
        fmt_int(r.named_by_prefix.0),
        fmt_int(r.named_by_prefix.1),
        r.prefixes.len()
    ));
    if !shared.is_empty() {
        s.push_str(&format!("  prefixes used by more than one module: {}\n", shared.iter().map(|(p, ms)| format!("{p} ({})", ms.len())).collect::<Vec<_>>().join(", ")));
    }
    if !r.stages.is_empty() {
        s.push_str("\nstages                paths  independent  agreeing  ran                         production path\n");
        for st in &r.stages {
            let ag = match (&st.agreeing, st.agreeing_families) {
                _ if !st.multi => "-".into(),
                (Some(_), Some(f)) => f.to_string(),
                _ => "?".into(),
            };
            let ran = if st.ran.is_empty() { "-".to_string() } else { st.ran.join(" ") };
            let path = if !st.on_path {
                "not on it".to_string()
            } else if st.multi && !st.direct {
                format!("{} via {}", st.default, st.through.join(" -> "))
            } else {
                format!("{} direct", st.default)
            };
            s.push_str(&format!("  {:<18} {:>6} {:>12} {:>9}  {:<27} {}\n", st.id, st.members.len(), st.families, ag, fit(&ran, 27), path));
        }
        let (good, all) = robustness(r);
        let weak: Vec<&str> = r.stages.iter().filter(|x| x.on_path && !x.robust()).map(|x| x.id.as_str()).collect();
        let strong: Vec<&str> = r.stages.iter().filter(|x| x.on_path && x.robust()).map(|x| x.id.as_str()).collect();
        s.push_str(&format!("\nrobustness: {good} of {all} stages on the production path have at least 2 independent implementations seen agreeing\n"));
        if !strong.is_empty() {
            s.push_str(&format!("  two or more: {}\n", strong.join(", ")));
        }
        if !weak.is_empty() {
            s.push_str(&format!("  one only:    {}\n", weak.join(", ")));
        }
        if r.stages.iter().any(|x| x.agreeing.is_none() && x.multi) {
            s.push_str("  (? = never measured: `viv arch report --verify` runs the tests the lens cites)\n");
        }
    }
    if !r.findings.is_empty() {
        s.push('\n');
        for f in &r.findings {
            s.push_str(&f.render());
        }
    }
    s
}

fn fit(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

fn strs(xs: &[String]) -> String {
    format!("[{}]", xs.iter().map(|x| json_str(x)).collect::<Vec<_>>().join(","))
}

/// A stage as JSON (the report's, and the explorer bundle's).
pub fn stage_json(st: &StageInProgram, f: &StageFacts, s: &Stage) -> String {
    let fams: Vec<String> = f.families.iter().map(|x| strs(x)).collect();
    let ties: Vec<String> = f.ties.iter().map(|(a, b, c)| format!("{{\"a\":{},\"b\":{},\"shared\":{}}}", json_str(a), json_str(b), strs(c))).collect();
    let cites: Vec<String> = f.citations.iter().map(|(k, c)| format!("{{\"kind\":{},\"name\":{}}}", json_str(k), json_str(c))).collect();
    format!(
        "{{\"id\":{},\"title\":{},\"multi\":{},\"intent\":{},\"what\":{},\"registry\":{},\"members\":{},\"default\":{},\"through\":{},\"contract\":{},\"tolerance\":{},\"checked\":[{}],\"families\":[{}],\"ties\":[{}],\"independent\":{},\"ran\":{},\"agreeing\":{},\"agreeing_families\":{},\"on_path\":{},\"direct\":{},\"direct_callers\":{},\"robust\":{}}}",
        json_str(&st.id),
        json_str(&st.title),
        st.multi,
        json_str(&st.intent),
        json_str(&st.what),
        s.registry.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
        strs(&st.members),
        json_str(&st.default),
        strs(&s.through),
        json_str(&st.contract),
        f.tolerance.as_ref().map(|(n, v)| format!("{{\"name\":{},\"value\":{}}}", json_str(n), v)).unwrap_or_else(|| "null".into()),
        cites.join(","),
        fams.join(","),
        ties.join(","),
        f.families.len(),
        strs(&st.ran),
        st.agreeing.as_ref().map(|v| strs(v)).unwrap_or_else(|| "null".into()),
        st.agreeing_families.map(|n| n.to_string()).unwrap_or_else(|| "null".into()),
        st.on_path,
        st.direct,
        strs(&f.direct_callers),
        st.robust()
    )
}

/// `viv arch report --json`.
pub fn report_json(r: &Report, a: &ArchFile, facts: &[StageFacts]) -> String {
    let layers: Vec<String> = r.layers.iter().map(|(l, m, d, n)| format!("{{\"layer\":{},\"modules\":{m},\"defs\":{d},\"ran\":{}}}", json_str(l), n.map(|x| x.to_string()).unwrap_or_else(|| "null".into()))).collect();
    let idle: Vec<String> = r.idle.iter().map(|(m, l, d)| format!("{{\"module\":{},\"layer\":{},\"defs\":{d}}}", json_str(m), json_str(l))).collect();
    let pairs: Vec<String> = r.pairs.iter().map(|(a, b, n, c)| format!("{{\"from\":{},\"to\":{},\"edges\":{n},\"calls\":{c}}}", json_str(a), json_str(b))).collect();
    let fan = |v: &[(String, usize)]| v.iter().map(|(d, n)| format!("{{\"def\":{},\"n\":{n}}}", json_str(d))).collect::<Vec<_>>().join(",");
    let prefixes: Vec<String> = r.prefixes.iter().map(|(p, ms)| format!("{{\"prefix\":{},\"modules\":{}}}", json_str(p), strs(ms))).collect();
    let stages: Vec<String> = r
        .stages
        .iter()
        .filter_map(|st| {
            let k = a.stages.iter().position(|s| s.id == st.id)?;
            Some(stage_json(st, &facts[k], &a.stages[k]))
        })
        .collect();
    let (good, all) = robustness(r);
    format!(
        "{{\"program\":{},\"arch\":{},\"modules\":{},\"defs\":{},\"expanded_defs\":{},\"ran\":{},\"layers\":[{}],\"idle\":[{}],\"deepest\":{},\"indirection\":{},\"fan_in\":[{}],\"fan_out\":[{}],\"pairs\":[{}],\"prefixes\":[{}],\"named_by_prefix\":[{},{}],\"stages\":[{}],\"robustness\":{{\"robust\":{good},\"stages\":{all}}},\"findings\":[{}]}}",
        json_str(&r.program),
        json_str(&r.arch),
        r.modules,
        r.defs,
        r.expanded_defs,
        r.ran.map(|n| n.to_string()).unwrap_or_else(|| "null".into()),
        layers.join(","),
        idle.join(","),
        strs(&r.deepest),
        strs(&r.indirection),
        fan(&r.fan_in),
        fan(&r.fan_out),
        pairs.join(","),
        prefixes.join(","),
        r.named_by_prefix.0,
        r.named_by_prefix.1,
        stages.join(","),
        r.findings.iter().map(Finding::json).collect::<Vec<_>>().join(",")
    )
}

// ---------------------------------------------------------------------------
// The whole check, and the ratchet
// ---------------------------------------------------------------------------

/// Everything `viv arch check` finds for one lens file, with the measured
/// budget values (for the ratchet).
pub struct Checked {
    pub findings: Vec<Finding>,
    pub values: BTreeMap<String, usize>,
    pub facts: Vec<StageFacts>,
}

/// Check a lens against its modules. `covered` are the modules under
/// `covers`; `programs` the facts of the programs the budgets name (those
/// not measured are left out).
pub fn check(a: &ArchFile, w: &mut World, covered: &[String], programs: &[ProgramFacts], agreement: &Agreement) -> Checked {
    let mut findings: Vec<Finding> = a.errors.iter().map(|(l, m)| Finding { path: a.path.clone(), line: *l, kind: "lens", key: format!("lens {m}"), message: m.clone(), detail: Vec::new() }).collect();
    for m in &w.mods {
        if let Some(e) = &m.parse_error {
            if covered.contains(&m.name) {
                findings.push(Finding { path: m.path.clone(), line: 1, kind: "note", key: format!("parse {}", m.name), message: format!("`{}` does not parse, so its calls are not seen: {}", m.name, e), detail: Vec::new() });
            }
        }
    }
    findings.extend(assign_layers(a, w, covered));
    for l in &a.layers {
        for p in &l.modules {
            if !w.mods.iter().any(|m| matches(p, &m.name)) {
                findings.push(Finding { path: a.path.clone(), line: l.line, kind: "lens", key: format!("lens layer {} {p}", l.id), message: format!("layer `{}` lists `{p}`, which matches no module", l.id), detail: Vec::new() });
            }
        }
    }
    findings.extend(check_layers(a, w));
    findings.extend(check_private(a, w));
    let mut values = BTreeMap::new();
    findings.extend(check_def_budgets(a, w, &mut values));
    let facts: Vec<StageFacts> = a.stages.iter().map(|s| stage_facts(a, w, s)).collect();
    findings.extend(check_stages(a, w, &facts));
    for p in programs {
        let r = report(a, w, p, &facts, agreement, &mut values);
        findings.extend(r.findings);
    }
    findings.sort_by(|x, y| (x.kind == "note", x.kind, &x.path, x.line, &x.key).cmp(&(y.kind == "note", y.kind, &y.path, y.line, &y.key)));
    Checked { findings, values, facts }
}

/// The ratchet's record for one lens file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recorded {
    pub violations: BTreeSet<String>,
    pub values: BTreeMap<String, usize>,
    /// per stage: the members seen agreeing when last measured
    pub agreeing: BTreeMap<String, Vec<String>>,
}

/// Read `arch_ratchet.json`: lens path -> its record.
pub fn parse_ratchet(text: &str) -> Result<BTreeMap<String, Recorded>, String> {
    use crate::story::Json;
    let v = crate::story::parse_json(text).map_err(|e| format!("arch_ratchet.json: {e}"))?;
    let Json::Obj(files) = v else { return Err("arch_ratchet.json: expected an object of lens files".into()) };
    let mut out = BTreeMap::new();
    for (path, rec) in files {
        let mut r = Recorded::default();
        if let Some(Json::Arr(vs)) = rec.get("violations") {
            r.violations = vs.iter().filter_map(|x| x.str().map(String::from)).collect();
        }
        if let Some(Json::Obj(vs)) = rec.get("values") {
            for (k, x) in vs {
                if let Some(n) = x.num() {
                    r.values.insert(k.clone(), n as usize);
                }
            }
        }
        if let Some(Json::Obj(st)) = rec.get("agreeing") {
            for (k, x) in st {
                if let Json::Arr(ms) = x {
                    r.agreeing.insert(k.clone(), ms.iter().filter_map(|m| m.str().map(String::from)).collect());
                }
            }
        }
        out.insert(path.clone(), r);
    }
    Ok(out)
}

/// Write the ratchet, one entry per line so a change reads as a diff.
pub fn ratchet_text(all: &BTreeMap<String, Recorded>) -> String {
    let mut s = String::from("{\n");
    let n = all.len();
    for (i, (path, r)) in all.iter().enumerate() {
        s.push_str(&format!("  {}: {{\n    \"violations\": [\n", json_str(path)));
        let vs: Vec<String> = r.violations.iter().map(|v| format!("      {}", json_str(v))).collect();
        s.push_str(&vs.join(",\n"));
        s.push_str("\n    ],\n    \"values\": {\n");
        let vals: Vec<String> = r.values.iter().map(|(k, v)| format!("      {}: {v}", json_str(k))).collect();
        s.push_str(&vals.join(",\n"));
        s.push_str("\n    },\n    \"agreeing\": {\n");
        let ag: Vec<String> = r.agreeing.iter().map(|(k, v)| format!("      {}: {}", json_str(k), strs(v))).collect();
        s.push_str(&ag.join(",\n"));
        s.push_str(&format!("\n    }}\n  }}{}\n", if i + 1 < n { "," } else { "" }));
    }
    s.push_str("}\n");
    s
}

/// What the check records for the ratchet.
pub fn record_of(c: &Checked, agreement: &Agreement) -> Recorded {
    Recorded {
        violations: c.findings.iter().filter(|f| f.is_violation()).map(|f| f.key.clone()).collect(),
        values: c.values.clone(),
        agreeing: agreement.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    }
}

/// Compare a check against its record: (what got worse, what got better).
pub fn ratchet_compare(now: &Recorded, was: &Recorded, a: &ArchFile, facts: &[StageFacts]) -> (Vec<String>, Vec<String>) {
    let mut worse = Vec::new();
    let mut better = Vec::new();
    for v in now.violations.difference(&was.violations) {
        worse.push(format!("new violation: {v}"));
    }
    for v in was.violations.difference(&now.violations) {
        better.push(format!("fixed: {v}"));
    }
    for (k, n) in &now.values {
        // a value over its budget may not grow; under it, it may move freely
        if let Some(&o) = was.values.get(k) {
            if *n > o && now.violations.contains(k) {
                worse.push(format!("{k}: {o} -> {n}"));
            } else if *n < o && was.violations.contains(k) {
                better.push(format!("{k}: {o} -> {n}"));
            }
        }
    }
    // agreement may only grow: families seen agreeing, per stage
    for (id, ms) in &was.agreeing {
        let Some(nowm) = now.agreeing.get(id) else { continue };
        let fam = |list: &[String]| -> usize {
            a.stages.iter().position(|s| &s.id == id).map(|k| facts[k].families.iter().filter(|f| f.iter().any(|m| list.contains(m))).count()).unwrap_or(0)
        };
        let (o, n) = (fam(ms), fam(nowm));
        if n < o {
            worse.push(format!("stage {id}: {o} -> {n} independent implementations seen agreeing"));
        } else if n > o {
            better.push(format!("stage {id}: {o} -> {n} independent implementations seen agreeing"));
        }
    }
    (worse, better)
}

/// `viv arch check --json`.
pub fn check_json(path: &str, c: &Checked) -> String {
    let violations = c.findings.iter().filter(|f| f.is_violation()).count();
    let vals: Vec<String> = c.values.iter().map(|(k, v)| format!("{}:{v}", json_str(k))).collect();
    format!("{{\"arch\":{},\"violations\":{violations},\"findings\":[{}],\"values\":{{{}}}}}", json_str(path), c.findings.iter().map(Finding::json).collect::<Vec<_>>().join(","), vals.join(","))
}

/// `viv arch check` as text: the findings, then a count by kind.
pub fn check_text(path: &str, c: &Checked) -> String {
    let mut s = String::new();
    for f in &c.findings {
        s.push_str(&f.render());
    }
    let mut by: BTreeMap<&str, usize> = BTreeMap::new();
    for f in &c.findings {
        *by.entry(f.kind).or_insert(0) += 1;
    }
    let violations = c.findings.iter().filter(|f| f.is_violation()).count();
    s.push_str(&format!(
        "{path}: {} ({})\n",
        plural(violations, "violation", "violations"),
        by.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ")
    ));
    s
}

// ---------------------------------------------------------------------------
// The explorer: the multi-paths of one program, for the bundle
// ---------------------------------------------------------------------------

/// A place a multi-path hangs from in a program's graph: the def, and the
/// step of it that makes the call (when the call is in a step).
#[derive(Clone, Debug, PartialEq)]
pub struct Host {
    pub def: String,
    pub step: Option<String>,
}

/// The hosts of each stage in a program's static graph: the defs (and steps)
/// that call the stage's first driver, when the driver serves this stage
/// alone or the caller names the stage's registry; with no driver, the defs
/// that call the default directly.
pub fn hosts(a: &ArchFile, g: &crate::graph::Graph) -> Vec<(String, Vec<Host>)> {
    let names = |id: &str| -> Vec<String> { g.node(id).map(|n| n.calls.clone()).unwrap_or_default() };
    let mut out = Vec::new();
    for s in &a.stages {
        let mut hs: Vec<Host> = Vec::new();
        let entry = s.through.first().or(s.default.as_ref());
        let Some(entry) = entry else {
            out.push((s.id.clone(), hs));
            continue;
        };
        // the first driver of other stages present in this program
        let shared_entry = a.stages.iter().any(|o| o.id != s.id && o.through.first() == Some(entry) && o.registry.as_ref().map_or(false, |r| g.node(r).is_some()));
        let drivers_name_registry = s.registry.as_ref().map_or(false, |r| s.through.iter().any(|t| names(t).contains(r)));
        for n in &g.nodes {
            if n.kind != "def" || s.through.contains(&n.id) || s.registry.as_ref() == Some(&n.id) || s.members.contains(&n.id) {
                continue;
            }
            if !n.calls.contains(entry) {
                continue;
            }
            let ok = s.through.is_empty() || !shared_entry || drivers_name_registry || s.registry.as_ref().map_or(false, |r| n.calls.contains(r));
            if !ok {
                continue;
            }
            // the step that makes the call, when there is one
            let step = g.edges.iter().find(|e| e.kind == "call" && &e.to == entry && e.from.starts_with(&format!("{}/", n.id))).map(|e| e.from.clone());
            hs.push(Host { def: n.id.clone(), step });
        }
        out.push((s.id.clone(), hs));
    }
    out
}

/// The multi-paths of a program for the explorer's bundle: one object per
/// stage that is in the program, with its hosts. `mods` are the program's
/// modules; `extra` are modules the program does not load that define what
/// the stages name (the members that are not on the production path, the
/// registries, the intent defs), so the stages' implementations and intent
/// are measured as `viv arch report` measures them.
pub fn bundle_stages(arch_path: &str, arch_text: &str, mods: &[ModSrc], extra: &[ModSrc], program: &crate::graph::Graph, agreement: &Agreement) -> String {
    let a = parse(arch_path, arch_text);
    // the picture folds a stage's selector into its fork: a stage with no
    // drivers whose intent def calls the default by name (`orth_of`, `jk_of`:
    // the def that picks the member the options name) is drawn as the fork
    // where that def is called, as the drivers of a `through` line are
    let mut drawn = a.clone();
    for s in drawn.stages.iter_mut() {
        if !s.multi || !s.through.is_empty() {
            continue;
        }
        let (Some(i), Some(d)) = (s.intent.clone(), s.default.clone()) else { continue };
        if i != d && program.node(&i).is_some_and(|n| n.calls.contains(&d)) {
            s.through = vec![i];
        }
    }
    let all: Vec<ModSrc> = mods.iter().chain(extra.iter().filter(|e| !mods.iter().any(|m| m.name == e.name))).cloned().collect();
    let mut w = World::new(&all);
    let names: Vec<String> = mods.iter().map(|m| m.name.clone()).collect();
    let _ = assign_layers(&a, &mut w, &[]);
    let facts: Vec<StageFacts> = a.stages.iter().map(|s| stage_facts(&a, &w, s)).collect();
    let p = ProgramFacts { name: names.first().cloned().unwrap_or_default(), modules: names, expanded_defs: 0, run: None };
    let mut values = BTreeMap::new();
    let r = report(&a, &w, &p, &facts, agreement, &mut values);
    let hs = hosts(&drawn, program);
    let items: Vec<String> = r
        .stages
        .iter()
        .filter_map(|st| {
            let k = a.stages.iter().position(|s| s.id == st.id)?;
            let h = hs.iter().find(|(id, _)| id == &st.id).map(|(_, v)| v.clone()).unwrap_or_default();
            let hj: Vec<String> = h.iter().map(|x| format!("{{\"def\":{},\"step\":{}}}", json_str(&x.def), x.step.as_deref().map(json_str).unwrap_or_else(|| "null".into()))).collect();
            let body = stage_json(st, &facts[k], &drawn.stages[k]);
            Some(format!("{},\"hosts\":[{}]}}", body.trim_end_matches('}'), hj.join(",")))
        })
        .collect();
    let (good, all) = {
        // robustness over the stages present, as recorded
        let present: Vec<&StageInProgram> = r.stages.iter().collect();
        (present.iter().filter(|s| s.robust()).count(), present.len())
    };
    format!("{{\"arch\":{},\"system\":{},\"stages\":[{}],\"robustness\":{{\"robust\":{good},\"stages\":{all}}}}}", json_str(arch_path), json_str(&a.system), items.join(","))
}
