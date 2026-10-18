// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! What a `parallel for` may write.
//!
//! The passes of a `parallel for` run at once on several threads in an
//! OpenMP build, and in order everywhere else. So that every build prints the
//! same bytes, each pass must own a disjoint part of every array it writes:
//!
//! - a pass writes an array the passes share only at elements indexed by the
//!   loop variable in one fixed position (`a[i]`, `m[i, j]`, `t[k, i]`,
//!   `setslab(a, i, s)`, `a[i, :] = row`), directly or through a def that
//!   writes its parameter at an index it is given (`put_row(a, i)` where
//!   `put_row` writes `v[j, ...]` and `j` is its parameter);
//! - a pass reads a shared array it writes only at its own elements;
//! - no pass writes a global through a def, and no def a pass calls reads a
//!   global the passes write;
//! - a pass does not leave the loop (`break`, `return`), which OpenMP forbids.
//!
//! Arrays a pass makes for itself (the variables it assigns) are its own.
//! Scalars keep the rule they had (`lower::analyze_parallel`: private,
//! reduced or refused). A write the compiler cannot prove disjoint is refused
//! with the write named and a fix suggested, unless the loop is pinned in the
//! module's memory lens (`crate::memory_lens`): `disjoint parallel for q: Gp`
//! says the passes write disjoint elements of `Gp` for a reason the compiler
//! cannot see, and every checked build tests it while it runs.
//!
//! The analysis summarizes each def a `parallel for` reaches (which
//! parameters it writes and reads, and at which index; which globals it
//! writes and reads), to a fixed point over the call graph, and then checks
//! each `parallel for` body against the summaries. It runs on the AST, in
//! `crate::parse`, so the interpreter and both compilers refuse the same
//! programs; and, when the loops reach a call through a function value, once
//! more on the program after `defun`, where that call has become a call of a
//! def.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use knotkit::diag::Diagnostic;
use knotkit::span::Span;

use crate::ast::{Expr, Item, Program, Stmt};

/// The index id of the loop variable (a def's index ids are its parameters'
/// positions).
const LOOP: usize = usize::MAX;

/// How some code reaches the elements of one array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Acc {
    /// at elements whose index in position `pos` is the index `id` (a
    /// parameter of the def, or the loop variable) plus the constant `off`:
    /// disjoint for different values of it (`u[p + 1, j]` on pass p writes
    /// plane p + 1, which no other pass writes)
    At { pos: usize, id: usize, off: i64 },
    /// at any element
    Any,
}

/// `i`, `i + c`, `c + i` or `i - c` with c a whole-number literal: the
/// variable and the offset.
fn offset_index(e: &Expr) -> Option<(&str, i64)> {
    let lit = |e: &Expr| match e {
        Expr::Num { value, .. } if value.fract() == 0.0 && value.abs() < 1e15 => Some(*value as i64),
        _ => None,
    };
    match e {
        Expr::Var { name, .. } => Some((name.as_str(), 0)),
        Expr::Binary { op: crate::ast::BinOp::Add, lhs, rhs, .. } => match (&**lhs, &**rhs) {
            (Expr::Var { name, .. }, r) => lit(r).map(|c| (name.as_str(), c)),
            (l, Expr::Var { name, .. }) => lit(l).map(|c| (name.as_str(), c)),
            _ => None,
        },
        Expr::Binary { op: crate::ast::BinOp::Sub, lhs, rhs, .. } => match &**lhs {
            Expr::Var { name, .. } => lit(rhs).map(|c| (name.as_str(), -c)),
            _ => None,
        },
        _ => None,
    }
}

fn join(a: Option<Acc>, b: Acc) -> Acc {
    match a {
        None => b,
        Some(a) if a == b => a,
        _ => Acc::Any,
    }
}

/// One access, where it is, and what made it.
#[derive(Clone, Debug)]
struct Seen {
    acc: Acc,
    span: Span,
    /// the def through which it happens, when not directly
    via: Option<String>,
    /// made by a builtin that writes its argument (not an element write)
    builtin: bool,
}

/// What a def does to its parameters and to globals.
#[derive(Clone, Debug, Default, PartialEq)]
struct Sum {
    /// per parameter: how the def writes it, and how it reads it
    w: Vec<Option<Acc>>,
    r: Vec<Option<Acc>>,
    /// globals whose elements the def (or a def it calls) writes, and reads
    wg: BTreeSet<String>,
    rg: BTreeSet<String>,
    /// per parameter: the def may return it
    ret: Vec<bool>,
    /// per parameter: some write of it is a builtin's (`setslab`, `fill`,
    /// ...), not an element write a checked build can record
    wb: Vec<bool>,
}

/// Builtins that read an array's shape, not its elements.
const SHAPE_ONLY: &[&str] = &["len", "shape", "numel", "rows", "cols", "ndim", "size"];

/// The effects of a block on the arrays it can reach by name.
#[derive(Default)]
struct Eff {
    w: BTreeMap<String, Seen>,
    r: BTreeMap<String, Seen>,
    /// globals written or read through called defs: (global, def, call span)
    wg: Vec<(String, String, Span)>,
    rg: Vec<(String, String, Span)>,
    /// may return the named root
    ret: BTreeSet<String>,
    /// `break` or `return` outside any nested loop
    leaves: Option<(Span, &'static str)>,
}

impl Eff {
    fn write(&mut self, root: &str, acc: Acc, span: Span, via: Option<String>, builtin: bool) {
        let e = self.w.entry(root.to_string()).or_insert(Seen { acc, span, via: via.clone(), builtin });
        let joined = join(Some(e.acc), acc);
        if joined != e.acc {
            // the first write that breaks the pattern is the one to name
            *e = Seen { acc: joined, span, via, builtin: builtin || e.builtin };
        } else {
            e.builtin |= builtin;
        }
    }
    fn read(&mut self, root: &str, acc: Acc, span: Span, via: Option<String>) {
        let e = self.r.entry(root.to_string()).or_insert(Seen { acc, span, via: via.clone(), builtin: false });
        let joined = join(Some(e.acc), acc);
        if joined != e.acc {
            *e = Seen { acc: joined, span, via, builtin: false };
        }
    }
}

/// The names a block assigns (its locals, or a loop body's privates),
/// loop variables included.
fn assigned(body: &[Stmt], out: &mut HashSet<String>) {
    for s in body {
        match s {
            Stmt::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::If { then_body, else_body, .. } => {
                assigned(then_body, out);
                assigned(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => assigned(body, out),
            Stmt::For { var, body, .. } => {
                out.insert(var.clone());
                assigned(body, out);
            }
            _ => {}
        }
    }
}

/// How many times a block assigns `name` (a loop over it counts too).
fn assign_count(body: &[Stmt], name: &str) -> usize {
    body.iter()
        .map(|s| match s {
            Stmt::Assign { name: n, .. } => usize::from(n == name),
            Stmt::If { then_body, else_body, .. } => assign_count(then_body, name) + assign_count(else_body, name),
            Stmt::While { body, .. } | Stmt::Test { body, .. } => assign_count(body, name),
            Stmt::For { var, body, .. } => usize::from(var == name) + assign_count(body, name),
            _ => 0,
        })
        .sum()
}

/// Everything the analysis of one block needs to know about names.
struct Scope<'a> {
    sums: &'a HashMap<String, Sum>,
    params: &'a HashMap<String, Vec<String>>,
    /// is `name` an array this analysis follows (a parameter, a global, a
    /// shared array of the loop)?
    tracked: &'a dyn Fn(&str) -> bool,
    /// `Var(name)` as an index: its id (a parameter's position, or `LOOP`)
    index_id: &'a dyn Fn(&str) -> Option<usize>,
    /// a local that may hold a tracked array: the arrays it may hold
    alias: HashMap<String, BTreeSet<String>>,
    /// depth of loops nested inside the analysed block
    depth: usize,
    /// a pass's locals that are the loop variable plus a constant (`i3 = p +
    /// 1` once at the top of the pass, never assigned again): the constant
    derived: HashMap<String, i64>,
}

impl Scope<'_> {
    /// The tracked arrays an expression may be: a tracked name, a field of
    /// one (a record's array), a local holding one, a call of a def that may
    /// return its argument.
    fn roots(&self, e: &Expr) -> BTreeSet<String> {
        match e {
            Expr::Var { name, .. } => {
                if (self.tracked)(name) {
                    BTreeSet::from([name.clone()])
                } else {
                    self.alias.get(name).cloned().unwrap_or_default()
                }
            }
            Expr::Field { base, .. } => self.roots(base),
            // a record holds its array fields by handle: `r = R { xs: a }`
            // makes `r.xs` the array `a`
            Expr::RecordLit { fields, .. } => fields.iter().flat_map(|(_, _, v)| self.roots(v)).collect(),
            Expr::Call { callee, args, .. } => match self.sums.get(callee) {
                Some(s) => {
                    let mut out = BTreeSet::new();
                    for (k, a) in args.iter().enumerate() {
                        if s.ret.get(k).copied().unwrap_or(false) {
                            out.extend(self.roots(a));
                        }
                    }
                    out
                }
                None => BTreeSet::new(),
            },
            _ => BTreeSet::new(),
        }
    }

    /// The access an index list makes: `At` when some position holds an
    /// index variable, alone or plus or minus a whole-number literal.
    fn index_acc(&self, indices: &[Expr]) -> Acc {
        for (pos, ix) in indices.iter().enumerate() {
            if let Some((name, off)) = offset_index(ix) {
                if let Some(id) = (self.index_id)(name) {
                    return Acc::At { pos, id, off };
                }
                if let Some(d) = self.derived.get(name) {
                    return Acc::At { pos, id: LOOP, off: off + d };
                }
            }
        }
        Acc::Any
    }

    /// The access a slice spec makes (`viv__slice_get/set`'s array literal:
    /// four numbers per item, kind 0 an index in `lo`).
    fn spec_acc(&self, spec: &Expr) -> Acc {
        if let Expr::ArrLit { elems, .. } = spec {
            for (pos, item) in elems.chunks(4).enumerate() {
                if let [Expr::Num { value, .. }, Expr::Var { name, .. }, ..] = item {
                    if *value == 0.0 {
                        if let Some(id) = (self.index_id)(name) {
                            return Acc::At { pos, id, off: 0 };
                        }
                    }
                }
            }
        }
        Acc::Any
    }

    /// An access `inner` a callee makes through its parameter, seen from the
    /// call: `At { pos, j }` stays disjoint when argument `j` is an index here.
    fn through(&self, inner: Acc, args: &[Expr]) -> Acc {
        match inner {
            Acc::At { pos, id, off } => match args.get(id) {
                Some(Expr::Var { name, .. }) => match (self.index_id)(name) {
                    Some(here) => Acc::At { pos, id: here, off },
                    None => Acc::Any,
                },
                _ => Acc::Any,
            },
            Acc::Any => Acc::Any,
        }
    }

    fn expr(&self, e: &Expr, eff: &mut Eff) {
        match e {
            Expr::Var { name, span } => {
                for r in self.roots(&Expr::Var { name: name.clone(), span: *span }) {
                    eff.read(&r, Acc::Any, *span, None);
                }
            }
            Expr::Index { base, indices, span } => {
                let roots = self.roots(base);
                if roots.is_empty() {
                    self.expr(base, eff);
                } else {
                    let acc = self.index_acc(indices);
                    for r in roots {
                        eff.read(&r, acc, *span, None);
                    }
                }
                for i in indices {
                    self.expr(i, eff);
                }
            }
            Expr::Field { base, span, .. } => {
                for r in self.roots(base) {
                    eff.read(&r, Acc::Any, *span, None);
                }
                if self.roots(base).is_empty() {
                    self.expr(base, eff);
                }
            }
            Expr::Call { callee, args, span } => self.call(callee, args, *span, eff),
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    self.expr(x, eff);
                }
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.expr(operand, eff),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs, eff);
                self.expr(rhs, eff);
            }
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    self.expr(v, eff);
                }
            }
            // a lambda captures numbers only
            Expr::Lambda { .. } | Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => {}
        }
    }

    fn call(&self, callee: &str, args: &[Expr], span: Span, eff: &mut Eff) {
        if let Some(s) = self.sums.get(callee) {
            for (k, a) in args.iter().enumerate() {
                let roots = self.roots(a);
                if roots.is_empty() {
                    self.expr(a, eff);
                    continue;
                }
                for r in &roots {
                    if let Some(w) = s.w.get(k).copied().flatten() {
                        let builtin = s.wb.get(k).copied().unwrap_or(false);
                        eff.write(r, self.through(w, args), span, Some(callee.to_string()), builtin);
                    }
                    if let Some(rd) = s.r.get(k).copied().flatten() {
                        eff.read(r, self.through(rd, args), span, Some(callee.to_string()));
                    }
                }
            }
            for g in &s.wg {
                eff.wg.push((g.clone(), callee.to_string(), span));
            }
            for g in &s.rg {
                eff.rg.push((g.clone(), callee.to_string(), span));
            }
            return;
        }
        // a builtin
        let written = crate::builtins::written_args(callee, args.len());
        let shape_only = SHAPE_ONLY.contains(&callee);
        for (k, a) in args.iter().enumerate() {
            let roots = self.roots(a);
            if roots.is_empty() {
                self.expr(a, eff);
                continue;
            }
            if written.contains(&k) {
                let acc = match (callee, k) {
                    ("setslab", 0) => match args.get(1) {
                        Some(Expr::Var { name, .. }) => (self.index_id)(name).map_or(Acc::Any, |id| Acc::At { pos: 0, id, off: 0 }),
                        _ => Acc::Any,
                    },
                    (c, 0) if c == crate::interp::slices::SET => args.get(1).map_or(Acc::Any, |s| self.spec_acc(s)),
                    _ => Acc::Any,
                };
                for r in &roots {
                    eff.write(r, acc, span, None, true);
                }
            } else if !shape_only {
                let acc = match (callee, k) {
                    (c, 0) if c == crate::interp::slices::GET => args.get(1).map_or(Acc::Any, |s| self.spec_acc(s)),
                    ("slab", 0) => match args.get(1) {
                        Some(Expr::Var { name, .. }) => (self.index_id)(name).map_or(Acc::Any, |id| Acc::At { pos: 0, id, off: 0 }),
                        _ => Acc::Any,
                    },
                    _ => Acc::Any,
                };
                for r in &roots {
                    eff.read(r, acc, span, None);
                }
            }
        }
    }

    fn block(&self, body: &[Stmt], eff: &mut Eff) {
        for s in body {
            self.stmt(s, eff);
        }
    }

    fn stmt(&self, s: &Stmt, eff: &mut Eff) {
        match s {
            // `t = a` makes `t` another name for `a` (`aliases`), which
            // reads no element: what `t` is used for counts instead
            Stmt::Assign { value: Expr::Var { .. } | Expr::Field { .. }, .. } if !self.roots(match s {
                Stmt::Assign { value, .. } => value,
                _ => unreachable!(),
            })
            .is_empty() => {}
            Stmt::Assign { value, .. } => self.expr(value, eff),
            Stmt::Return { value, span } => {
                eff.ret.extend(self.roots(value));
                self.expr(value, eff);
                if self.depth == 0 && eff.leaves.is_none() {
                    eff.leaves = Some((*span, "return"));
                }
            }
            Stmt::Break { span } => {
                if self.depth == 0 && eff.leaves.is_none() {
                    eff.leaves = Some((*span, "break"));
                }
            }
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => self.expr(expr, eff),
            Stmt::SetIndex { base, indices, value, span } => {
                let roots = self.roots(base);
                let acc = self.index_acc(indices);
                for r in &roots {
                    eff.write(r, acc, *span, None, false);
                }
                if roots.is_empty() {
                    self.expr(base, eff);
                }
                for i in indices {
                    self.expr(i, eff);
                }
                self.expr(value, eff);
            }
            // a field write changes the whole value its root holds
            Stmt::SetField { base, value, span, .. } => {
                let roots = self.roots(base);
                for r in &roots {
                    eff.write(r, Acc::Any, *span, None, false);
                }
                if roots.is_empty() {
                    self.expr(base, eff);
                }
                self.expr(value, eff);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.expr(cond, eff);
                self.block(then_body, eff);
                self.block(else_body, eff);
            }
            Stmt::While { cond, body, .. } => {
                self.expr(cond, eff);
                self.nested(body, eff);
            }
            Stmt::For { bound, body, .. } => {
                self.expr(bound, eff);
                self.nested(body, eff);
            }
            Stmt::Test { body, .. } => self.block(body, eff),
            Stmt::Narrate { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
        }
    }

    fn nested(&self, body: &[Stmt], eff: &mut Eff) {
        let inner = Scope { sums: self.sums, params: self.params, tracked: self.tracked, index_id: self.index_id, alias: self.alias.clone(), depth: self.depth + 1, derived: self.derived.clone() };
        inner.block(body, eff);
    }
}

/// The locals of a block that may hold a tracked array (`t = a`, `t = r.xs`,
/// `t = f(a)` with `f` returning its argument), to a fixed point.
fn aliases(body: &[Stmt], sums: &HashMap<String, Sum>, params: &HashMap<String, Vec<String>>, tracked: &dyn Fn(&str) -> bool, index_id: &dyn Fn(&str) -> Option<usize>) -> HashMap<String, BTreeSet<String>> {
    fn assigns<'b>(body: &'b [Stmt], out: &mut Vec<(&'b str, &'b Expr)>) {
        for s in body {
            match s {
                Stmt::Assign { name, value, .. } => out.push((name, value)),
                Stmt::If { then_body, else_body, .. } => {
                    assigns(then_body, out);
                    assigns(else_body, out);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => assigns(body, out),
                _ => {}
            }
        }
    }
    let mut list = Vec::new();
    assigns(body, &mut list);
    let mut sc = Scope { sums, params, tracked, index_id, alias: HashMap::new(), depth: 0, derived: HashMap::new() };
    loop {
        let mut changed = false;
        for (name, value) in &list {
            if tracked(name) {
                continue;
            }
            let roots = sc.roots(value);
            if roots.is_empty() {
                continue;
            }
            let e = sc.alias.entry(name.to_string()).or_default();
            let before = e.len();
            e.extend(roots);
            changed |= e.len() != before;
        }
        if !changed {
            return sc.alias;
        }
    }
}

fn defs(program: &Program) -> Vec<&crate::ast::Def> {
    program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d) } else { None }).collect()
}

/// The top-level names: what a def reads as a global.
fn globals(program: &Program) -> HashSet<String> {
    let mut out = HashSet::new();
    let top: Vec<Stmt> = program.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    assigned(&top, &mut out);
    out
}

/// The summary of every def in `only`, to a fixed point over the call graph
/// (`only` holds every def they call).
fn summaries(program: &Program, only: &HashSet<String>) -> (HashMap<String, Sum>, HashMap<String, Vec<String>>) {
    let ds: Vec<&crate::ast::Def> = defs(program).into_iter().filter(|d| only.contains(&d.name)).collect();
    let globals = globals(program);
    let params: HashMap<String, Vec<String>> = ds.iter().map(|d| (d.name.clone(), d.params.iter().map(|p| p.name.clone()).collect())).collect();
    let mut sums: HashMap<String, Sum> = ds
        .iter()
        .map(|d| (d.name.clone(), Sum { w: vec![None; d.params.len()], r: vec![None; d.params.len()], ret: vec![false; d.params.len()], wb: vec![false; d.params.len()], ..Default::default() }))
        .collect();
    loop {
        let mut changed = false;
        for d in &ds {
            let ps: Vec<String> = d.params.iter().map(|p| p.name.clone()).collect();
            let mut local = HashSet::new();
            assigned(&d.body, &mut local);
            let is_param = |n: &str| ps.iter().any(|p| p == n);
            let tracked = |n: &str| is_param(n) || (globals.contains(n) && !local.contains(n));
            // a parameter the def never reassigns is a fixed index
            let index_id = |n: &str| if local.contains(n) { None } else { ps.iter().position(|p| p == n) };
            let alias = aliases(&d.body, &sums, &params, &tracked, &index_id);
            let sc = Scope { sums: &sums, params: &params, tracked: &tracked, index_id: &index_id, alias, depth: 0, derived: HashMap::new() };
            let mut eff = Eff::default();
            sc.block(&d.body, &mut eff);
            let mut s = Sum { w: vec![None; ps.len()], r: vec![None; ps.len()], ret: vec![false; ps.len()], wb: vec![false; ps.len()], ..Default::default() };
            for (k, p) in ps.iter().enumerate() {
                s.w[k] = eff.w.get(p).map(|x| x.acc);
                s.wb[k] = eff.w.get(p).is_some_and(|x| x.builtin);
                s.r[k] = eff.r.get(p).map(|x| x.acc);
                s.ret[k] = eff.ret.contains(p);
            }
            for (n, _) in eff.w.iter().filter(|(n, _)| !is_param(n)) {
                s.wg.insert(n.clone());
            }
            for (n, _) in eff.r.iter().filter(|(n, _)| !is_param(n)) {
                s.rg.insert(n.clone());
            }
            s.wg.extend(eff.wg.iter().map(|(g, _, _)| g.clone()));
            s.rg.extend(eff.rg.iter().map(|(g, _, _)| g.clone()));
            if sums.get(&d.name) != Some(&s) {
                sums.insert(d.name.clone(), s);
                changed = true;
            }
        }
        if !changed {
            return (sums, params);
        }
    }
}

/// A `parallel for` pinned in a memory lens: the passes write disjoint
/// elements of `arrays`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pin {
    /// the def the loop is in (`<main>` at the top level)
    pub def: String,
    /// the loop variable
    pub var: String,
    pub arrays: Vec<String>,
    /// the lens file that pins it, for messages
    pub file: String,
}

/// A loop a pin applies to: the loop's span, the arrays whose writes the
/// checked builds test, and the words naming it in the run-time error.
#[derive(Clone, Debug, PartialEq)]
pub struct Pinned {
    pub span: Span,
    pub arrays: Vec<String>,
    pub what: String,
}

/// Check every `parallel for` of the program. `pins` are the memory lens's
/// `disjoint` entries; the ones a loop needs are returned.
pub fn check(program: &Program, pins: &[Pin]) -> Result<Vec<Pinned>, Vec<Diagnostic>> {
    let mut any = false;
    for item in &program.items {
        let body: &[Stmt] = match item {
            Item::Def(d) => &d.body,
            Item::Stmt(s) => std::slice::from_ref(s),
            _ => continue,
        };
        walk_loops(body, &mut |_| any = true);
    }
    if !any {
        return Ok(Vec::new());
    }
    let mut pinned = check_one(program, pins)?;
    // after defun, a call through a function value is a call of a def; a
    // program whose loops reach none is not specialized again here
    let (_, indirect) = reached(program);
    if !indirect {
        return Ok(pinned);
    }
    if let Ok(spec) = crate::defun::specialize(Program { items: program.items.clone(), debug: Vec::new(), memory: Vec::new(), phrases: Vec::new(), whole: program.whole.clone() }) {
        for p in check_one(&spec, pins)? {
            if !pinned.contains(&p) {
                pinned.push(p);
            }
        }
    }
    Ok(pinned)
}

fn walk_loops<'a>(body: &'a [Stmt], f: &mut dyn FnMut(&'a Stmt)) {
    for s in body {
        match s {
            Stmt::For { parallel, body, .. } => {
                if *parallel {
                    f(s);
                }
                walk_loops(body, f);
            }
            Stmt::If { then_body, else_body, .. } => {
                walk_loops(then_body, f);
                walk_loops(else_body, f);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => walk_loops(body, f),
            _ => {}
        }
    }
}

/// The names a block calls (true), or names as a value (false: a def passed
/// to a def is a function value).
fn named(body: &[Stmt]) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    crate::defun::walk_block(body, &mut |e: &Expr| match e {
        Expr::Call { callee, .. } => out.push((callee.clone(), true)),
        Expr::Var { name, .. } => out.push((name.clone(), false)),
        _ => {}
    });
    out
}

/// The defs the program's `parallel for` bodies reach, through calls and
/// function values: the only ones the check needs to summarize. And whether
/// any of that code calls through a function value, which only the program
/// after `defun` resolves.
fn reached(program: &Program) -> (HashSet<String>, bool) {
    let ds: HashMap<&str, &crate::ast::Def> = defs(program).into_iter().map(|d| (d.name.as_str(), d)).collect();
    let mut todo: Vec<(String, bool)> = Vec::new();
    for item in &program.items {
        let body: &[Stmt] = match item {
            Item::Def(d) => &d.body,
            Item::Stmt(s) => std::slice::from_ref(s),
            _ => continue,
        };
        walk_loops(body, &mut |lp| {
            if let Stmt::For { body, .. } = lp {
                todo.extend(named(body));
            }
        });
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut indirect = false;
    while let Some((n, call)) = todo.pop() {
        if n == "apply" && call {
            indirect = true;
        }
        if let Some(d) = ds.get(n.as_str()) {
            // a def named as a value is a function value
            indirect |= !call;
            if seen.insert(n.clone()) {
                // a def's parameter holding a function, or a def passed on
                indirect |= d.params.iter().any(|p| matches!(p.ty, Some(crate::ast::TypeExpr::Fn { .. })));
                todo.extend(named(&d.body));
            }
        }
    }
    (seen, indirect)
}

fn check_one(program: &Program, pins: &[Pin]) -> Result<Vec<Pinned>, Vec<Diagnostic>> {
    let (only, _) = reached(program);
    let (sums, params) = summaries(program, &only);
    let globals = globals(program);
    // the defs that call `for_leaves`, themselves or through other defs
    let mut uses_leaves: HashSet<String> = HashSet::new();
    loop {
        let before = uses_leaves.len();
        for d in defs(program) {
            let mut hit = false;
            crate::defun::walk_block(&d.body, &mut |e: &Expr| {
                if let Expr::Call { callee, .. } = e {
                    hit |= callee == "for_leaves" || uses_leaves.contains(callee);
                }
            });
            if hit {
                uses_leaves.insert(d.name.clone());
            }
        }
        if uses_leaves.len() == before {
            break;
        }
    }
    let mut out = Vec::new();
    for item in &program.items {
        let (def, body): (&str, &[Stmt]) = match item {
            Item::Def(d) => (d.name.as_str(), &d.body),
            Item::Stmt(s) => (crate::parser::MAIN_DEF, std::slice::from_ref(s)),
            _ => continue,
        };
        let mut loops = Vec::new();
        walk_loops(body, &mut |s| loops.push(s));
        for lp in loops {
            let Stmt::For { var, body, span, .. } = lp else { continue };
            let pin = pins.iter().find(|p| p.def == def && p.var == *var);
            if let Some(p) = check_loop(def, var, body, *span, &sums, &params, &globals, pin, &uses_leaves).map_err(|d| vec![d])? {
                out.push(p);
            }
        }
    }
    Ok(out)
}

/// The fix to suggest for a write the passes share.
fn fix_for(root: &str, var: &str, via: &Option<String>) -> String {
    match via {
        Some(f) => format!(
            "fix: give `{f}` the loop variable `{var}` as the index it writes at, so pass `{var}` writes only elements indexed by `{var}`; or let each pass fill an array of its own and file it with `setslab({root}, {var}, mine)`; or, if the passes write different elements for a reason the compiler cannot see, pin the loop in the module's .memory lens: `disjoint parallel for {var}: {root} \"why\"`"
        ),
        None => format!(
            "fix: let pass `{var}` write only elements indexed by `{var}` (`{root}[{var}]`, `{root}[{var}, j]`); a total over the passes is a reduction (`total += x`, with `total` a number); or, if the passes write different elements for a reason the compiler cannot see, pin the loop in the module's .memory lens: `disjoint parallel for {var}: {root} \"why\"`"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn check_loop(
    def: &str,
    var: &str,
    body: &[Stmt],
    span: Span,
    sums: &HashMap<String, Sum>,
    params: &HashMap<String, Vec<String>>,
    globals: &HashSet<String>,
    pin: Option<&Pin>,
    uses_leaves: &HashSet<String>,
) -> Result<Option<Pinned>, Diagnostic> {
    // `for_leaves` writes through the def it calls, at every array of the
    // records it is given: no index of the loop says which, so a pass is
    // refused one, itself or through a def
    let mut leaves: Option<Span> = None;
    crate::defun::walk_block(body, &mut |e: &Expr| {
        if let Expr::Call { callee, span, .. } = e {
            if (callee == "for_leaves" || uses_leaves.contains(callee)) && leaves.is_none() {
                leaves = Some(*span);
            }
        }
    });
    if let Some(at) = leaves {
        return Err(Diagnostic::error("a pass of a `parallel for` cannot call `for_leaves`")
            .with_primary(at, "writes through a def into whole records")
            .with_secondary(span, "the loop")
            .with_note("call it in a plain `for`, or give each pass its own records"));
    }
    let mut private = HashSet::new();
    assigned(body, &mut private);
    let loop_var_fixed = !private.contains(var);
    let tracked = |n: &str| n != var && !private.contains(n) && !sums.contains_key(n);
    let index_id = |n: &str| (n == var && loop_var_fixed).then_some(LOOP);
    let alias = aliases(body, sums, params, &tracked, &index_id);
    // `i3 = p + 1` at the top of a pass, never assigned again: i3 indexes as
    // p + 1 does
    let mut derived = HashMap::new();
    if loop_var_fixed {
        for st in body {
            if let Stmt::Assign { name, value, .. } = st {
                if let Some((v, off)) = offset_index(value) {
                    if v == var && name != var && assign_count(body, name) == 1 {
                        derived.insert(name.clone(), off);
                    }
                }
            }
        }
    }
    let sc = Scope { sums, params, tracked: &tracked, index_id: &index_id, alias, depth: 0, derived };
    let mut eff = Eff::default();
    sc.block(body, &mut eff);
    let header = format!("parallel for {var}");
    if let Some((at, what)) = eff.leaves {
        return Err(Diagnostic::error(format!("a pass of a `parallel for` cannot `{what}`"))
            .with_primary(at, format!("this `{what}` leaves the loop from inside a pass"))
            .with_secondary(span, "the loop")
            .with_note("the passes run at once on several threads, and OpenMP does not let one of them leave the loop; use a plain `for`, or compute a flag in each pass and act on it after the loop"));
    }
    let pinned_arr = |r: &str| pin.is_some_and(|p| p.arrays.iter().any(|a| a == r));
    let mut used_pins: Vec<String> = Vec::new();
    // writes
    for (root, seen) in &eff.w {
        if pinned_arr(root) && seen.builtin {
            return Err(Diagnostic::error(format!("a `disjoint` pin covers element writes of `{root}`, and this one is a builtin's"))
                .with_primary(seen.span, format!("this call writes `{root}` as a whole"))
                .with_note(format!("a checked build tests the pin by recording each `{root}[...] = value` a pass makes; write the elements one by one, or remove the pin and write `{root}[{var}, ...]`")));
        }
        let ok = matches!(seen.acc, Acc::At { id: LOOP, .. });
        if ok {
            continue;
        }
        if pinned_arr(root) {
            used_pins.push(root.clone());
            continue;
        }
        let at = match &seen.via {
            Some(f) => format!("`{f}` writes `{root}` here at an index that does not come from `{var}`, so two passes can write the same element"),
            None if seen.builtin => format!("this writes `{root}` beyond the elements indexed by `{var}`, so two passes can write the same element"),
            None => format!("the index of `{root}` written here does not depend on `{var}`, so two passes can write the same element"),
        };
        let mut d = Diagnostic::error(format!("two passes of this `parallel for` can write the same element of `{root}`"))
            .with_primary(seen.span, at)
            .with_secondary(span, format!("memory lens: {header} writes `{root}` at elements that are not its own"));
        if let Some(f) = &seen.via {
            d = d.with_note(format!("`{f}` writes its argument in place; the passes share `{root}`, and only an index the loop variable gives is known to differ from pass to pass"));
        }
        return Err(d.with_note(fix_for(root, var, &seen.via)));
    }
    // globals written through a called def
    for (g, f, at) in &eff.wg {
        if pinned_arr(g) {
            used_pins.push(g.clone());
            continue;
        }
        return Err(Diagnostic::error(format!("`{f}` writes the global `{g}`, which every pass of this `parallel for` shares"))
            .with_primary(*at, format!("each pass calls `{f}`, which writes elements of `{g}`"))
            .with_secondary(span, "the loop")
            .with_note(format!("fix: pass `{g}` to `{f}` with the loop variable `{var}` as the index it writes at, or call `{f}` in a plain `for`")));
    }
    // reads of what the passes write
    let written: BTreeMap<&String, Acc> = eff.w.iter().map(|(r, s)| (r, s.acc)).collect();
    for (root, seen) in &eff.r {
        let Some(&wacc) = written.get(root) else { continue };
        let own = match (wacc, seen.acc) {
            (Acc::At { pos: wp, id: LOOP, off: wo }, Acc::At { pos: rp, id: LOOP, off: ro }) => wp == rp && wo == ro,
            _ => false,
        };
        if own {
            continue;
        }
        let what = match &seen.via {
            Some(f) => format!("`{f}` reads `{root}` here at elements other passes write"),
            None => format!("this reads `{root}` at elements other passes write"),
        };
        return Err(Diagnostic::error(format!("a pass of this `parallel for` reads elements of `{root}` that other passes write"))
            .with_primary(seen.span, what)
            .with_secondary(eff.w[root].span, format!("the passes write `{root}` here"))
            .with_note(format!("which value a pass reads would depend on which pass ran first; read only `{root}[{var}, ...]`, or read from a copy made before the loop: `{root}0 = copy({root})`")));
    }
    for (g, f, at) in &eff.rg {
        if written.contains_key(g) && globals.contains(g.as_str()) {
            return Err(Diagnostic::error(format!("`{f}` reads the global `{g}`, which the passes of this `parallel for` write"))
                .with_primary(*at, format!("`{f}` reads elements of `{g}` that other passes write"))
                .with_secondary(eff.w[g].span, format!("the passes write `{g}` here"))
                .with_note(format!("pass `{f}` the element it needs instead, or read from a copy made before the loop")));
        }
    }
    used_pins.sort();
    used_pins.dedup();
    if used_pins.is_empty() {
        return Ok(None);
    }
    let p = pin.expect("a used pin");
    Ok(Some(Pinned { span, arrays: used_pins, what: format!("the parallel for over `{var}` in `{def}`, which {} pins as disjoint", p.file) }))
}
