// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv facts --json <file>`: what the compiler knows about one file, as one
//! JSON object, for the program database (tools/vivdb).
//!
//! Nothing here is a new analysis. Defs, spans and parameters come from the
//! parser; signatures, steps, `#|` docs and intent state from the lens code
//! (lens.rs); feature uses from features_usage.rs; `ir_hash` is the body
//! hash `viv hash` prints. Two things are new:
//!
//! - `ast_hash`, the `vivhash1` encoding of a def's parsed AST: parameters,
//!   types as written and the body, with callees referenced by name. It
//!   leaves out the def's own name, comments, docs, step headings and the
//!   debug lens, so whitespace, comments and renames do not move it.
//! - `closure_hash`, a Merkle hash over the def's `ast_hash`, the
//!   `closure_hash` of each def it calls (in this module or a module it
//!   imports) and the names of the builtins it calls. A recursive group of
//!   defs is hashed as one unit.

use crate::ast::{Def, Expr, Item, Program, Stmt, TypeExpr};
use crate::debug_lens::json_str as q;
use crate::graph::Lines;
use knotkit::span::SourceId;
use std::collections::{BTreeSet, HashMap, HashSet};

/// The version tag that starts every `vivhash1` input.
pub const VIVHASH_TAG: &[u8] = b"vivhash1\0";

// ---------------------------------------------------------------------------
// vivhash1: a canonical encoding of the parsed AST
// ---------------------------------------------------------------------------

struct Enc {
    h: knotkit::ir::Hasher,
}

impl Enc {
    fn new() -> Enc {
        let mut h = knotkit::ir::Hasher::new();
        h.update(VIVHASH_TAG);
        Enc { h }
    }
    fn tag(&mut self, t: u8) {
        self.h.update(&[t]);
    }
    fn n(&mut self, n: usize) {
        self.h.update(&(n as u64).to_le_bytes());
    }
    fn s(&mut self, s: &str) {
        self.n(s.len());
        self.h.update(s.as_bytes());
    }
    /// A number as its shortest round-trip text.
    fn num(&mut self, v: f64) {
        self.s(&format!("{v:?}"));
    }
    fn hex(self) -> String {
        self.h.finalize().to_hex().to_string()
    }

    fn ty(&mut self, t: &TypeExpr) {
        match t {
            TypeExpr::Num => self.tag(1),
            TypeExpr::Arr => self.tag(2),
            TypeExpr::CArr => self.tag(3),
            TypeExpr::Bool => self.tag(4),
            TypeExpr::Str => self.tag(5),
            TypeExpr::StrList => self.tag(6),
            TypeExpr::Int => self.tag(7),
            TypeExpr::IntList => self.tag(8),
            TypeExpr::F32 => self.tag(12),
            TypeExpr::Packed(k) => {
                self.tag(13);
                self.tag(k.code());
            }
            TypeExpr::Opt(inner) => {
                self.tag(9);
                self.ty(inner);
            }
            TypeExpr::Named { name, .. } => {
                self.tag(10);
                self.s(name);
            }
            TypeExpr::RecList { name, .. } => {
                self.tag(11);
                self.s(name);
            }
            TypeExpr::Fn { arity } => {
                self.tag(12);
                self.n(*arity);
            }
            TypeExpr::Infer => self.tag(13),
        }
    }
    fn opt_ty(&mut self, t: &Option<TypeExpr>) {
        match t {
            Some(t) => self.ty(t),
            None => self.tag(0),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Imag { value, .. } => {
                self.tag(0x41);
                self.num(*value);
            }
            Expr::Bool { value, .. } => {
                self.tag(0x42);
                self.tag(*value as u8);
            }
            Expr::Num { value, .. } => {
                self.tag(0x43);
                self.num(*value);
            }
            Expr::Str { value, .. } => {
                self.tag(0x44);
                self.s(value);
            }
            Expr::Var { name, .. } => {
                self.tag(0x45);
                self.s(name);
            }
            Expr::ArrLit { elems, .. } => {
                self.tag(0x46);
                self.exprs(elems);
            }
            Expr::Index { base, indices, .. } => {
                self.tag(0x47);
                self.expr(base);
                self.exprs(indices);
            }
            Expr::Lambda { params, body, .. } => {
                self.tag(0x48);
                self.n(params.len());
                for p in params {
                    self.s(p);
                }
                self.expr(body);
            }
            Expr::Neg { operand, .. } => {
                self.tag(0x49);
                self.expr(operand);
            }
            Expr::Not { operand, .. } => {
                self.tag(0x4a);
                self.expr(operand);
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                self.tag(0x4b);
                self.tag(op.tag());
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Call { callee, args, .. } => {
                self.tag(0x4c);
                self.s(callee);
                self.exprs(args);
            }
            Expr::RecordLit { type_name, fields, .. } => {
                self.tag(0x4d);
                self.s(type_name);
                self.n(fields.len());
                for (f, _, v) in fields {
                    self.s(f);
                    self.expr(v);
                }
            }
            Expr::Field { base, field, .. } => {
                self.tag(0x4e);
                self.expr(base);
                self.s(field);
            }
        }
    }
    fn exprs(&mut self, es: &[Expr]) {
        self.n(es.len());
        for e in es {
            self.expr(e);
        }
    }

    /// Statements, with the step markers and the debug lens left out: they
    /// are text about the code, not the code.
    fn stmts(&mut self, ss: &[Stmt]) {
        let kept: Vec<&Stmt> = ss.iter().filter(|s| !matches!(s, Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. })).collect();
        self.n(kept.len());
        for s in kept {
            self.stmt(s);
        }
    }
    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Assign { name, value, .. } => {
                self.tag(0x61);
                self.s(name);
                self.expr(value);
            }
            Stmt::Return { value, .. } => {
                self.tag(0x62);
                self.expr(value);
            }
            Stmt::ExprStmt { expr, .. } => {
                self.tag(0x63);
                self.expr(expr);
            }
            Stmt::Narrate { text, .. } => {
                self.tag(0x64);
                self.s(text);
            }
            Stmt::Show { expr, .. } => {
                self.tag(0x65);
                self.expr(expr);
            }
            Stmt::Test { name, body, .. } => {
                self.tag(0x66);
                self.s(name);
                self.stmts(body);
            }
            Stmt::SetIndex { base, indices, value, .. } => {
                self.tag(0x67);
                self.expr(base);
                self.exprs(indices);
                self.expr(value);
            }
            Stmt::SetField { base, field, value, .. } => {
                self.tag(0x68);
                self.expr(base);
                self.s(field);
                self.expr(value);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.tag(0x69);
                self.expr(cond);
                self.stmts(then_body);
                self.stmts(else_body);
            }
            Stmt::While { cond, body, .. } => {
                self.tag(0x6a);
                self.expr(cond);
                self.stmts(body);
            }
            Stmt::For { var, bound, body, parallel, .. } => {
                self.tag(0x6b);
                self.s(var);
                self.tag(*parallel as u8);
                self.expr(bound);
                self.stmts(body);
            }
            Stmt::Break { .. } => self.tag(0x6c),
            Stmt::Continue { .. } => self.tag(0x6d),
            Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
        }
    }
}

/// The `vivhash1` identity of a def: full 64 hex digits.
pub fn ast_hash(d: &Def) -> String {
    let mut e = Enc::new();
    e.tag(b'D');
    e.n(d.params.len());
    for p in &d.params {
        e.s(&p.name);
        e.opt_ty(&p.ty);
    }
    e.opt_ty(&d.ret);
    e.stmts(&d.body);
    e.hex()
}

/// The `vivhash1` identity of a test block or of the top level.
fn block_hash(kind: u8, label: &str, body: &[Stmt]) -> String {
    let mut e = Enc::new();
    e.tag(kind);
    e.s(label);
    e.stmts(body);
    e.hex()
}

// ---------------------------------------------------------------------------
// calls
// ---------------------------------------------------------------------------

/// Every name a body calls, or names as a value, with the byte offset.
fn names_used(ss: &[Stmt], out: &mut Vec<(String, u32, bool)>) {
    fn ex(e: &Expr, out: &mut Vec<(String, u32, bool)>) {
        match e {
            Expr::Call { callee, args, span } => {
                out.push((callee.clone(), span.start, false));
                for a in args {
                    ex(a, out);
                }
            }
            Expr::Var { name, span } => out.push((name.clone(), span.start, true)),
            Expr::Binary { lhs, rhs, .. } => {
                ex(lhs, out);
                ex(rhs, out);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => ex(operand, out),
            Expr::Index { base, indices, .. } => {
                ex(base, out);
                for i in indices {
                    ex(i, out);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    ex(x, out);
                }
            }
            Expr::Lambda { body, .. } => ex(body, out),
            Expr::Field { base, .. } => ex(base, out),
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    ex(v, out);
                }
            }
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }
    for s in ss {
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => ex(value, out),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => ex(expr, out),
            Stmt::SetIndex { base, indices, value, .. } => {
                ex(base, out);
                for i in indices {
                    ex(i, out);
                }
                ex(value, out);
            }
            Stmt::SetField { base, value, .. } => {
                ex(base, out);
                ex(value, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                ex(cond, out);
                names_used(then_body, out);
                names_used(else_body, out);
            }
            Stmt::While { cond, body, .. } => {
                ex(cond, out);
                names_used(body, out);
            }
            Stmt::For { bound, body, .. } => {
                ex(bound, out);
                names_used(body, out);
            }
            // a test's calls belong to the test, not to the code around it
            Stmt::Test { .. } => {}
            _ => {}
        }
    }
}

/// The last segment of a dotted callee (`scf.rhf` is `rhf`).
fn base_name(n: &str) -> &str {
    n.rsplit('.').next().unwrap_or(n)
}

/// One def the program can reach: its module, AST hash and resolved calls.
struct Known {
    module: String,
    ast: String,
    /// (callee as written, line, target, is a value reference)
    calls: Vec<(String, usize, String, bool)>,
}

struct Builtins {
    names: HashSet<String>,
}

impl Builtins {
    fn new() -> Builtins {
        let mut names: HashSet<String> = crate::features::builtin_rows().into_iter().map(|r| r.name).collect();
        for n in crate::builtins::PURE.iter().chain(crate::builtins::EFFECTS.iter()) {
            names.insert(n.to_string());
        }
        Builtins { names }
    }
    fn has(&self, n: &str) -> bool {
        self.names.contains(n) || crate::interp::strings::is_builtin(n)
    }
}

/// Resolve each name a body uses: `local`, `module:<m>`, `record`,
/// `builtin`, or `unresolved` (a parameter called as a function, an enum
/// variant). A value reference is kept only when it names a def.
fn resolve(
    used: &[(String, u32, bool)],
    lines: &Lines,
    module: &str,
    defs_of: &HashMap<String, Vec<String>>,
    records: &HashSet<String>,
    builtins: &Builtins,
) -> Vec<(String, usize, String, bool)> {
    let mut out = Vec::new();
    for (name, at, value) in used {
        let b = base_name(name);
        let target = match defs_of.get(b) {
            Some(ms) if ms.iter().any(|m| m == module) => "local".to_string(),
            Some(ms) => format!("module:{}", ms[0]),
            None if *value => continue,
            None if records.contains(b) => "record".to_string(),
            None if builtins.has(b) => "builtin".to_string(),
            None => "unresolved".to_string(),
        };
        out.push((name.clone(), lines.line(*at), target, *value));
    }
    out
}

/// Every def item of a program, ladders' rungs included (the last
/// definition of a name wins, as in the flattened program).
fn defs_of_program(p: &Program) -> Vec<&Def> {
    fn go<'a>(items: &'a [Item], out: &mut Vec<&'a Def>) {
        for it in items {
            match it {
                Item::Def(d) => {
                    out.retain(|x: &&Def| x.name != d.name);
                    out.push(d);
                }
                Item::Rung(r) => go(&r.items, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    go(&p.items, &mut out);
    out
}

fn top_stmts(p: &Program) -> Vec<&Stmt> {
    fn go<'a>(items: &'a [Item], out: &mut Vec<&'a Stmt>) {
        for it in items {
            match it {
                Item::Stmt(s) => out.push(s),
                Item::Rung(r) => go(&r.items, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    go(&p.items, &mut out);
    out
}

fn records_of(p: &Program) -> HashSet<String> {
    p.items
        .iter()
        .filter_map(|i| match i {
            Item::Record(r) => Some(r.name.clone()),
            Item::Enum(e) => Some(e.name.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// closure_hash: Tarjan's strongly connected components over the known defs
// ---------------------------------------------------------------------------

fn closure_hashes(known: &HashMap<String, Known>) -> HashMap<String, String> {
    // node = "module\0name"
    let keys: Vec<&String> = {
        let mut k: Vec<&String> = known.keys().collect();
        k.sort();
        k
    };
    let idx: HashMap<&str, usize> = keys.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let edges: Vec<Vec<usize>> = keys
        .iter()
        .map(|k| {
            let kn = &known[*k];
            let mut v: Vec<usize> = kn
                .calls
                .iter()
                .filter_map(|(n, _, t, _)| {
                    let m = if t == "local" { kn.module.clone() } else { t.strip_prefix("module:")?.to_string() };
                    idx.get(format!("{m}\0{}", base_name(n)).as_str()).copied()
                })
                .collect();
            v.sort();
            v.dedup();
            v
        })
        .collect();
    let n = keys.len();
    let (mut index, mut low, mut on) = (vec![usize::MAX; n], vec![0; n], vec![false; n]);
    let (mut stack, mut next, mut comp, mut comps) = (Vec::new(), 0usize, vec![usize::MAX; n], Vec::<Vec<usize>>::new());
    // iterative Tarjan: (node, next edge)
    for s in 0..n {
        if index[s] != usize::MAX {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(s, 0)];
        index[s] = next;
        low[s] = next;
        next += 1;
        stack.push(s);
        on[s] = true;
        while let Some(&mut (v, ref mut ei)) = work.last_mut() {
            if *ei < edges[v].len() {
                let w = edges[v][*ei];
                *ei += 1;
                if index[w] == usize::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on[w] = true;
                    work.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                work.pop();
                if let Some(&(u, _)) = work.last() {
                    low[u] = low[u].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut c = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on[w] = false;
                        comp[w] = comps.len();
                        c.push(w);
                        if w == v {
                            break;
                        }
                    }
                    comps.push(c);
                }
            }
        }
    }
    // Tarjan emits components callees first, so one pass in order suffices.
    let mut comp_hash: Vec<String> = Vec::with_capacity(comps.len());
    for (ci, members) in comps.iter().enumerate() {
        let mut asts: Vec<&str> = members.iter().map(|m| known[keys[*m]].ast.as_str()).collect();
        asts.sort();
        let mut outside: BTreeSet<String> = BTreeSet::new();
        for m in members {
            for w in &edges[*m] {
                if comp[*w] != ci {
                    outside.insert(comp_hash[comp[*w]].clone());
                }
            }
            for (name, _, t, _) in &known[keys[*m]].calls {
                if t == "builtin" {
                    outside.insert(format!("builtin:{}", base_name(name)));
                }
            }
        }
        let mut e = Enc::new();
        e.tag(b'C');
        e.n(asts.len());
        for a in asts {
            e.s(a);
        }
        e.n(outside.len());
        for o in &outside {
            e.s(o);
        }
        comp_hash.push(e.hex());
    }
    let mut out = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        let c = &comps[comp[i]];
        let h = if c.len() == 1 {
            comp_hash[comp[i]].clone()
        } else {
            let mut e = Enc::new();
            e.tag(b'M');
            e.s(&comp_hash[comp[i]]);
            e.s(&known[*k].ast);
            e.hex()
        };
        out.insert((*k).clone(), h);
    }
    out
}

// ---------------------------------------------------------------------------
// the report
// ---------------------------------------------------------------------------

fn ty_text(t: &TypeExpr) -> String {
    match t {
        TypeExpr::Num => "Num".into(),
        TypeExpr::Arr => "Arr".into(),
        TypeExpr::CArr => "CArr".into(),
        TypeExpr::Bool => "Bool".into(),
        TypeExpr::Str => "Str".into(),
        TypeExpr::StrList => "[Str]".into(),
        TypeExpr::Int => "Int".into(),
        TypeExpr::IntList => "[Int]".into(),
        TypeExpr::F32 => "Float32".into(),
        TypeExpr::Packed(k) => format!("[{}]", k.name()),
        TypeExpr::Opt(i) => format!("{}?", ty_text(i)),
        TypeExpr::Named { name, .. } => name.clone(),
        TypeExpr::RecList { name, .. } => format!("[{name}]"),
        TypeExpr::Fn { arity } => format!("Fn{arity}"),
        TypeExpr::Infer => "_".into(),
    }
}

fn calls_json(calls: &[(String, usize, String, bool)]) -> String {
    let xs: Vec<String> = calls
        .iter()
        .map(|(n, l, t, v)| format!("{{\"name\":{},\"line\":{l},\"target\":{},\"value\":{v}}}", q(n), q(t)))
        .collect();
    format!("[{}]", xs.join(","))
}

/// The facts of one file. `loaded` is the lens model of it (with the
/// modules it imports); `src` its bytes; `path` how to name it.
pub fn facts_json(path: &str, src: &str, loaded: Option<&crate::lens::Loaded>) -> String {
    let builtins = Builtins::new();
    let lines = Lines::new(src);
    let module = loaded.map(|l| l.root.text.name.clone()).unwrap_or_else(|| path.trim_end_matches(".viv").to_string());
    let prog = match crate::parse_keeping_rungs(src, SourceId(0)) {
        Ok(p) => p,
        Err(ds) => {
            let msg = ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n");
            return format!("{{\"vivfacts\":1,\"path\":{},\"module\":{},\"bytes\":{},\"parse_error\":{}}}", q(path), q(&module), src.len(), q(&msg));
        }
    };

    // every def the program can reach: this module's and its imports'
    let mut parsed: Vec<(String, Program, String)> = Vec::new();
    if let Some(l) = loaded {
        for m in &l.used {
            if let Ok(p) = crate::parse_keeping_rungs(&m.text.src, SourceId(0)) {
                parsed.push((m.text.name.clone(), p, m.text.src.clone()));
            }
        }
    }
    let mut defs_of: HashMap<String, Vec<String>> = HashMap::new();
    let mut records = records_of(&prog);
    for d in defs_of_program(&prog) {
        defs_of.entry(d.name.clone()).or_default().push(module.clone());
    }
    for (m, p, _) in &parsed {
        records.extend(records_of(p));
        for d in defs_of_program(p) {
            defs_of.entry(d.name.clone()).or_default().push(m.clone());
        }
    }
    let mut known: HashMap<String, Known> = HashMap::new();
    let add = |m: &str, p: &Program, text: &str, known: &mut HashMap<String, Known>| {
        let ls = Lines::new(text);
        for d in defs_of_program(p) {
            let mut used = Vec::new();
            names_used(&d.body, &mut used);
            let calls = resolve(&used, &ls, m, &defs_of, &records, &builtins);
            known.insert(format!("{m}\0{}", d.name), Known { module: m.to_string(), ast: ast_hash(d), calls });
        }
    };
    for (m, p, text) in &parsed {
        add(m, p, text, &mut known);
    }
    add(&module, &prog, src, &mut known);
    let closure = closure_hashes(&known);

    // the lens model: signatures, steps, docs, ir hashes, intent state
    let code_defs: HashMap<&str, &crate::lens::CodeDef> = loaded.map(|l| l.root.defs.iter().map(|d| (d.name.as_str(), d)).collect()).unwrap_or_default();
    let mut state: HashMap<String, (&'static str, String)> = HashMap::new();
    let mut file_entries: HashMap<String, Option<String>> = HashMap::new();
    if let Some(l) = loaded {
        for e in l.root.entries().1 {
            file_entries.insert(e.name.clone(), e.stamp.clone());
        }
        for f in crate::lens::check(l, &mut |_, _| None) {
            if (f.kind == "stale" || f.kind == "unstamped") && !f.anchor.contains(['/', ':']) {
                state.insert(f.anchor.clone(), (f.kind, f.path.clone()));
            }
        }
    }

    let mut defs_out: Vec<String> = Vec::new();
    for d in defs_of_program(&prog) {
        let key = format!("{module}\0{}", d.name);
        let kn = &known[&key];
        let cd = code_defs.get(d.name.as_str());
        let params: Vec<String> = d
            .params
            .iter()
            .map(|p| format!("{{\"name\":{},\"type\":{}}}", q(&p.name), p.ty.as_ref().map(|t| q(&ty_text(t))).unwrap_or("null".into())))
            .collect();
        let inline = cd.map(|c| &c.inline);
        let (istate, stamp, source) = if let Some(stamp) = file_entries.get(&d.name) {
            (state.get(&d.name).map(|s| s.0).unwrap_or("current"), stamp.clone(), "file")
        } else if let Some(i) = inline.filter(|i| !i.fields.is_empty() || !i.gloss.is_empty() || i.stamp.is_some()) {
            (state.get(&d.name).map(|s| s.0).unwrap_or("current"), i.stamp.as_ref().map(|s| s.0.clone()), "inline")
        } else {
            ("missing", None, "none")
        };
        let steps: Vec<String> = cd
            .map(|c| {
                c.steps()
                    .iter()
                    .map(|(n, depth)| {
                        let id = match &n.kind {
                            crate::lens::OKind::Step(s) => s.clone(),
                            _ => String::new(),
                        };
                        format!("{{\"id\":{},\"text\":{},\"line_start\":{},\"line_end\":{},\"depth\":{depth}}}", q(&id), q(&n.text), n.line, n.end)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let docs: Vec<String> = inline
            .map(|i| {
                i.fields
                    .iter()
                    .map(|(k, t, l)| (k.as_str(), t, l))
                    .chain(i.gloss.iter().map(|(k, t, l)| (k.as_str(), t, l)))
                    .map(|(k, t, l)| format!("{{\"label\":{},\"text\":{},\"line\":{l}}}", q(k), q(t)))
                    .collect()
            })
            .unwrap_or_default();
        defs_out.push(format!(
            "{{\"name\":{},\"kind\":\"def\",\"byte_start\":{},\"byte_end\":{},\"line_start\":{},\"line_end\":{},\"signature\":{},\"params\":[{}],\"ret\":{},\"ast_hash\":\"{}\",\"closure_hash\":\"{}\",\"ir_hash\":{},\"intent\":{{\"state\":\"{istate}\",\"stamp\":{},\"source\":\"{source}\"}},\"steps\":[{}],\"docs\":[{}],\"calls\":{}}}",
            q(&d.name),
            d.span.start,
            d.span.end,
            lines.line(d.span.start),
            lines.line(d.span.end.saturating_sub(1)),
            cd.map(|c| q(&c.sig)).unwrap_or("null".into()),
            params.join(","),
            d.ret.as_ref().map(|t| q(&ty_text(t))).unwrap_or("null".into()),
            kn.ast,
            closure[&key],
            cd.and_then(|c| c.hash.as_ref()).map(|h| q(h)).unwrap_or("null".into()),
            stamp.map(|s| q(&s)).unwrap_or("null".into()),
            steps.join(","),
            docs.join(","),
            calls_json(&kn.calls),
        ));
    }

    // the top level and its tests
    let tops = top_stmts(&prog);
    let mut tests_out = Vec::new();
    let mut main_body: Vec<Stmt> = Vec::new();
    for s in &tops {
        if let Stmt::Test { name, body, span } = s {
            let mut used = Vec::new();
            names_used(body, &mut used);
            let calls = resolve(&used, &lines, &module, &defs_of, &records, &builtins);
            tests_out.push(format!(
                "{{\"name\":{},\"byte_start\":{},\"byte_end\":{},\"line_start\":{},\"ast_hash\":\"{}\",\"calls\":{}}}",
                q(name),
                span.start,
                span.end,
                lines.line(span.start),
                block_hash(b'T', name, body),
                calls_json(&calls)
            ));
        } else {
            main_body.push((*s).clone());
        }
    }
    let mut used = Vec::new();
    names_used(&main_body, &mut used);
    let main_calls = resolve(&used, &lines, &module, &defs_of, &records, &builtins);
    let main_json = format!(
        "{{\"ast_hash\":\"{}\",\"ir_hash\":{},\"statements\":{},\"calls\":{}}}",
        block_hash(b'P', "", &main_body),
        code_defs.get(crate::parser::MAIN_DEF).and_then(|c| c.hash.as_ref()).map(|h| q(h)).unwrap_or("null".into()),
        main_body.len(),
        calls_json(&main_calls)
    );

    // imports: `import`/`from` statements and `#! use` lines
    let mut imports: BTreeSet<String> = BTreeSet::new();
    if let Ok(st) = crate::imports::scan(src) {
        for s in st {
            for it in s.items {
                imports.insert(it.path().join("/"));
            }
        }
    }
    let scan = crate::features_usage::scan(src);
    let used_modules: Vec<String> = loaded.map(|l| l.used.iter().map(|m| q(&m.text.name)).collect()).unwrap_or_default();

    // features: the catalog's detectors over this text (a `file:` detector
    // is about the files beside this one, and is left to the caller)
    let mut feats = Vec::new();
    for e in crate::features::catalog().entries.iter().filter(|e| !e.usage.is_empty()) {
        let n: usize = e.usage.iter().map(|d| if matches!(d, crate::features::Detector::File(_)) { 0 } else { scan.count(d, src) }).sum();
        if n > 0 {
            feats.push(format!("{{\"feature\":{},\"count\":{n}}}", q(&e.id)));
        }
    }
    let hand: Vec<String> = scan
        .findings
        .iter()
        .map(|f| {
            let feature = crate::features_usage::PATTERNS.iter().find(|p| p.0 == f.pattern).map(|p| p.2).unwrap_or("");
            format!("{{\"pattern\":{},\"feature\":{},\"line\":{}}}", q(f.pattern), q(feature), f.line)
        })
        .collect();

    format!(
        "{{\"vivfacts\":1,\"hash\":\"vivhash1\",\"path\":{},\"module\":{},\"bytes\":{},\"parse_error\":null,\"hash_error\":{},\"imports\":[{}],\"used_modules\":[{}],\"defs\":[{}],\"tests\":[{}],\"main\":{},\"features\":[{}],\"handwritten\":[{}]}}",
        q(path),
        q(&module),
        src.len(),
        loaded.and_then(|l| l.hash_error.as_ref()).map(|e| q(e)).unwrap_or("null".into()),
        imports.iter().map(|s| q(s)).collect::<Vec<_>>().join(","),
        used_modules.join(","),
        defs_out.join(","),
        tests_out.join(","),
        main_json,
        feats.join(","),
        hand.join(",")
    )
}

/// `(name, ast_hash)` for every def of a source text, parsed on its own.
pub fn ast_hashes(src: &str) -> Result<Vec<(String, String)>, String> {
    let p = crate::parse_keeping_rungs(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    Ok(defs_of_program(&p).into_iter().map(|d| (d.name.clone(), ast_hash(d))).collect())
}
