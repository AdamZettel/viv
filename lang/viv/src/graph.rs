// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The program as a graph: defs, the steps
//! inside them and the loops that hold steps are the nodes; calls and the
//! values steps hand each other are the edges. What the explorer draws and
//! moves through.
//!
//! Each step's **interface** is computed here, from the code, never written
//! by hand:
//! - `reads`: the variables the step reads that were set before it (a
//!   parameter, or a variable some other statement of the def assigns);
//! - `writes`: the variables the step assigns that are read after it (later
//!   in the def, anywhere in a loop around the step, since the next pass
//!   reads them, or by the caller: an array parameter the step writes into);
//! - `globals`: top-level names the step reads (the deck's constants);
//! - `calls`: the defs the step calls, or names as function values.
//!
//! A variable the step both sets and uses only inside itself is neither: it
//! is the step's own business. That is the point of a step: a named, bounded
//! region that stays where it runs, whose inputs and outputs the compiler can
//! state.
//!
//! The analysis is on the syntax, flow-insensitive inside a statement and in
//! source order across statements; it is conservative where it cannot know.
//! An array passed to a def that writes into that parameter, or to a builtin
//! that fills an output array (`eig_sym(A, vals, vecs)`), counts as written.
//!
//! `graph_json` also carries the explorer's line table (which statements a
//! line-level recording records, `crate::debug_lens::lines_table`) and the
//! map from the expanded program's lines back to each module's own lines.

use std::collections::{HashMap, HashSet};

use knotkit::span::{SourceId, Span};

use crate::ast::*;
use crate::debug_lens::json_str;
use crate::parser::MAIN_DEF;


/// What a statement or expression reads, writes and calls, each name once,
/// in the order first met.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Eff {
    pub reads: Vec<String>,
    pub writes: Vec<String>,
    pub calls: Vec<String>,
}

fn add(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

/// The program-wide facts an effect needs: which names are defs, and which
/// parameters (by position) each def writes into.
pub struct Cx {
    pub defs: HashSet<String>,
    pub mutates: HashMap<String, Vec<usize>>,
}

impl Cx {
    /// The defs of a program and, by a fixed point over the call graph, the
    /// array parameters each writes into.
    pub fn of(program: &Program) -> Cx {
        let defs: Vec<&Def> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d) } else { None }).collect();
        let mut cx = Cx { defs: defs.iter().map(|d| d.name.clone()).collect(), mutates: HashMap::new() };
        loop {
            let mut changed = false;
            for d in &defs {
                let mut e = Eff::default();
                for s in &d.body {
                    stmt_eff(s, &cx, &mut e);
                }
                let m: Vec<usize> = d.params.iter().enumerate().filter(|(_, p)| e.writes.contains(&p.name)).map(|(i, _)| i).collect();
                if cx.mutates.get(&d.name) != Some(&m) {
                    cx.mutates.insert(d.name.clone(), m);
                    changed = true;
                }
            }
            if !changed {
                return cx;
            }
        }
    }
}

/// The effect of an expression.
pub fn expr_eff(e: &Expr, cx: &Cx, out: &mut Eff) {
    match e {
        Expr::Var { name, .. } => {
            if cx.defs.contains(name) {
                add(&mut out.calls, name);
            } else {
                add(&mut out.reads, name);
            }
        }
        Expr::Call { callee, args, .. } => {
            // a record update's def is part of the expression (crate::update)
            if cx.defs.contains(callee) && !crate::update::is_expression_def(callee) {
                add(&mut out.calls, callee);
            }
            for a in args {
                expr_eff(a, cx, out);
            }
            let written: Vec<usize> = match cx.mutates.get(callee) {
                Some(m) => m.clone(),
                // `for_leaves(f, r1, ..., extra...)` writes the arguments
                // at the places `f` writes its parameters, one further on
                None if callee == "for_leaves" => match args.first() {
                    Some(Expr::Var { name: f, .. }) => cx.mutates.get(f).map(|m| m.iter().map(|j| j + 1).collect()).unwrap_or_default(),
                    _ => Vec::new(),
                },
                // a builtin: the arguments the one table says it writes (crate::builtins)
                None => crate::builtins::written_args(callee, args.len()),
            };
            // an argument written is its variable written, or the variable
            // at the root of the place it is (`r.w`, `model.blocks[l]`)
            for k in written {
                if let Some(name) = args.get(k).and_then(target_var) {
                    if !cx.defs.contains(name) {
                        add(&mut out.writes, name);
                    }
                }
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            expr_eff(lhs, cx, out);
            expr_eff(rhs, cx, out);
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_eff(operand, cx, out),
        Expr::Index { base, indices, .. } => {
            expr_eff(base, cx, out);
            for i in indices {
                expr_eff(i, cx, out);
            }
        }
        Expr::ArrLit { elems, .. } => {
            for x in elems {
                expr_eff(x, cx, out);
            }
        }
        Expr::Lambda { params, body, .. } => {
            let mut inner = Eff::default();
            expr_eff(body, cx, &mut inner);
            for r in inner.reads.iter().filter(|r| !params.contains(r)) {
                add(&mut out.reads, r);
            }
            for c in &inner.calls {
                add(&mut out.calls, c);
            }
        }
        Expr::Field { base, .. } => expr_eff(base, cx, out),
        Expr::RecordLit { fields, .. } => {
            for (_, _, v) in fields {
                expr_eff(v, cx, out);
            }
        }
        Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
    }
}

/// The root variable an assignment target writes: `A` for `A[i] = ...` and
/// for `m.node_x[i] = ...`.
fn target_var(e: &Expr) -> Option<&str> {
    match e {
        Expr::Var { name, .. } => Some(name),
        Expr::Field { base, .. } | Expr::Index { base, .. } => target_var(base),
        _ => None,
    }
}

/// The effect of a statement, nested statements included. Lens statements
/// and step markers have none.
pub fn stmt_eff(s: &Stmt, cx: &Cx, out: &mut Eff) {
    match s {
        Stmt::Assign { name, value, .. } => {
            expr_eff(value, cx, out);
            add(&mut out.writes, name);
        }
        Stmt::SetIndex { base, indices, value, .. } => {
            expr_eff(base, cx, out);
            for i in indices {
                expr_eff(i, cx, out);
            }
            expr_eff(value, cx, out);
            if let Some(v) = target_var(base) {
                add(&mut out.writes, v);
            }
        }
        Stmt::SetField { base, value, .. } => {
            expr_eff(base, cx, out);
            expr_eff(value, cx, out);
            if let Some(v) = target_var(base) {
                add(&mut out.writes, v);
            }
        }
        Stmt::Return { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => expr_eff(value, cx, out),
        Stmt::If { cond, then_body, else_body, .. } => {
            expr_eff(cond, cx, out);
            for s in then_body.iter().chain(else_body) {
                stmt_eff(s, cx, out);
            }
        }
        Stmt::While { cond, body, .. } => {
            expr_eff(cond, cx, out);
            for s in body {
                stmt_eff(s, cx, out);
            }
        }
        Stmt::For { var, bound, body, .. } => {
            expr_eff(bound, cx, out);
            add(&mut out.writes, var);
            for s in body {
                stmt_eff(s, cx, out);
            }
        }
        Stmt::Test { .. } | Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// A def, flattened
// ---------------------------------------------------------------------------

/// One statement of a def as the analysis sees it: a simple statement, or
/// the header of an `if` or a loop (its condition or bound). `guarded` counts
/// the `if`s and loops around it, so a write here is certain to happen only
/// when it is no deeper than the region being asked about.
#[derive(Clone, Debug)]
struct Leaf {
    eff: Eff,
    guarded: usize,
}

#[derive(Clone, Debug)]
struct StepSpan {
    id: String,
    text: String,
    depth: u32,
    span: Span,
    /// leaves [start, end)
    start: usize,
    end: usize,
    guarded: usize,
    /// the step and loop this one sits in (indices into the steps and loops)
    parent_step: Option<usize>,
    parent_loop: Option<usize>,
    /// every loop around it, outermost first
    loops: Vec<usize>,
}

#[derive(Clone, Debug)]
struct LoopSpan {
    span: Span,
    header: String,
    start: usize,
    end: usize,
    parent_step: Option<usize>,
    parent_loop: Option<usize>,
    has_step: bool,
    /// a `for` loop's variable
    var: Option<String>,
}

#[derive(Default)]
struct Flat {
    leaves: Vec<Leaf>,
    steps: Vec<StepSpan>,
    loops: Vec<LoopSpan>,
}

fn has_step(body: &[Stmt]) -> bool {
    body.iter().any(|s| match s {
        Stmt::StepEnter { .. } => true,
        Stmt::If { then_body, else_body, .. } => has_step(then_body) || has_step(else_body),
        Stmt::While { body, .. } | Stmt::For { body, .. } => has_step(body),
        _ => false,
    })
}

fn flatten(body: &[Stmt], cx: &Cx, src: &str, guarded: usize, open: &mut Vec<usize>, loops: &mut Vec<usize>, f: &mut Flat) {
    for s in body {
        match s {
            Stmt::StepEnter { id, text, depth, span, .. } => {
                let k = f.steps.len();
                f.steps.push(StepSpan {
                    id: id.clone(),
                    text: text.clone(),
                    depth: *depth,
                    span: *span,
                    start: f.leaves.len(),
                    end: f.leaves.len(),
                    guarded,
                    parent_step: open.last().copied(),
                    parent_loop: loops.last().copied(),
                    loops: loops.clone(),
                });
                open.push(k);
            }
            Stmt::StepExit { span, .. } => {
                if let Some(k) = open.pop() {
                    f.steps[k].end = f.leaves.len();
                    let st = f.steps[k].span;
                    f.steps[k].span = Span { end: span.end, ..st };
                }
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                let mut e = Eff::default();
                expr_eff(cond, cx, &mut e);
                f.leaves.push(Leaf { eff: e, guarded });
                flatten(then_body, cx, src, guarded + 1, open, loops, f);
                // an empty leaf between the branches, so the two are never
                // read as one block (a write in one does not reach the other)
                f.leaves.push(Leaf { eff: Eff::default(), guarded });
                flatten(else_body, cx, src, guarded + 1, open, loops, f);
            }
            Stmt::While { cond, body, span } | Stmt::For { bound: cond, body, span, .. } => {
                let mut e = Eff::default();
                expr_eff(cond, cx, &mut e);
                if let Stmt::For { var, .. } = s {
                    add(&mut e.writes, var);
                }
                let k = f.loops.len();
                let header = src.get(span.start as usize..span.end as usize).and_then(|t| t.lines().next()).unwrap_or("").trim().trim_end_matches('{').trim().to_string();
                f.loops.push(LoopSpan { span: *span, header, start: f.leaves.len(), end: 0, parent_step: open.last().copied(), parent_loop: loops.last().copied(), has_step: has_step(body), var: if let Stmt::For { var, .. } = s { Some(var.clone()) } else { None } });
                f.leaves.push(Leaf { eff: e, guarded });
                loops.push(k);
                flatten(body, cx, src, guarded + 1, open, loops, f);
                loops.pop();
                f.loops[k].end = f.leaves.len();
            }
            Stmt::Test { .. } | Stmt::Lens { .. } => {}
            other => {
                let mut e = Eff::default();
                stmt_eff(other, cx, &mut e);
                f.leaves.push(Leaf { eff: e, guarded });
            }
        }
    }
}

/// The interface of one region of a def: leaves [start, end), `guarded` the
/// depth the region starts at, `loops` the loops around it.
fn interface(f: &Flat, start: usize, end: usize, guarded: usize, loops: &[usize], params: &[String]) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    // upward-exposed reads: read before a certain write inside the region
    let mut certain: HashSet<&str> = HashSet::new();
    let mut exposed: Vec<String> = Vec::new();
    let mut written: Vec<String> = Vec::new();
    let mut calls: Vec<String> = Vec::new();
    for l in &f.leaves[start..end] {
        for r in &l.eff.reads {
            if !certain.contains(r.as_str()) {
                add(&mut exposed, r);
            }
        }
        for w in &l.eff.writes {
            add(&mut written, w);
        }
        for c in &l.eff.calls {
            add(&mut calls, c);
        }
        if l.guarded == guarded {
            for w in &l.eff.writes {
                certain.insert(w);
            }
        }
    }
    let outside = |range: &mut dyn Iterator<Item = usize>| -> (HashSet<String>, HashSet<String>) {
        let (mut r, mut w) = (HashSet::new(), HashSet::new());
        for k in range {
            r.extend(f.leaves[k].eff.reads.iter().cloned());
            w.extend(f.leaves[k].eff.writes.iter().cloned());
        }
        (r, w)
    };
    let (_, w_out) = outside(&mut (0..start).chain(end..f.leaves.len()));
    let mut all_w: HashSet<String> = w_out.clone();
    all_w.extend(written.iter().cloned());
    let param = |v: &str| params.iter().any(|p| p == v);
    let reads: Vec<String> = exposed.iter().filter(|v| param(v) || w_out.contains(*v)).cloned().collect();
    let globals: Vec<String> = exposed.iter().filter(|v| !param(v) && !all_w.contains(*v)).cloned().collect();
    // read after: later in the def, or earlier in a loop around the region
    // (the next pass reads it), unless something certain to run writes it
    // first: a statement at the top of the def (or of that loop's body), or
    // the header of a `for` over it whose body holds the read
    // A write at leaf k is certain to have run before a read at leaf j > k
    // when the two are in one block: no leaf between them, j included, is
    // shallower than k.
    let killed = |v: &str, from: usize, j: usize| -> bool {
        (from..j).any(|k| f.leaves[k].eff.writes.iter().any(|w| w == v) && (k + 1..=j).all(|m| f.leaves[m].guarded >= f.leaves[k].guarded))
            || f.loops.iter().any(|lp| lp.var.as_deref() == Some(v) && lp.start >= from && lp.start < j && j < lp.end)
    };
    let live = |v: &str| -> bool {
        (end..f.leaves.len()).any(|j| f.leaves[j].eff.reads.iter().any(|r| r == v) && !killed(v, end, j))
            || loops.iter().any(|&lp| {
                let (ls, le) = (f.loops[lp].start, f.loops[lp].end);
                (ls..le).filter(|j| *j < start || *j >= end).any(|j| {
                    // a read before the region, on the next pass; the loop's
                    // header (leaf ls) runs before it again
                    f.leaves[j].eff.reads.iter().any(|r| r == v) && !(j < start && killed(v, ls + 1, j)) && !(j >= end && killed(v, end, j))
                })
            })
    };
    let _ = outside;
    let writes: Vec<String> = written.iter().filter(|v| live(v) || param(v)).cloned().collect();
    (reads, writes, globals, calls)
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

/// A node of the graph.
#[derive(Clone, Debug, Default)]
pub struct GNode {
    /// `def`, `step`, `loop` or `record`
    pub kind: &'static str,
    /// `f` for a def, `f/id` for a step, `f/loop@<line>` for a loop
    pub id: String,
    pub def: String,
    /// the step's id; a loop's header (`while it < maxit`)
    pub name: String,
    /// a step's inline text
    pub text: String,
    pub parent: Option<String>,
    /// lines in the program as given (1-based, first and last)
    pub line: usize,
    pub end: usize,
    pub params: Vec<String>,
    pub reads: Vec<String>,
    pub writes: Vec<String>,
    pub globals: Vec<String>,
    pub calls: Vec<String>,
    /// a def: it has steps
    pub stepped: bool,
    /// a loop: it holds a step
    pub has_step: bool,
}

/// An edge: a call (`from` a def or step, `to` a def) or a value one step
/// hands another (`vars`); `carried` when it reaches the next pass of a loop.
#[derive(Clone, Debug, PartialEq)]
pub struct GEdge {
    pub kind: &'static str,
    pub from: String,
    pub to: String,
    pub vars: Vec<String>,
    pub carried: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<GNode>,
    pub edges: Vec<GEdge>,
}

impl Graph {
    pub fn node(&self, id: &str) -> Option<&GNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

/// 1-based line of a byte offset.
pub struct Lines {
    starts: Vec<u32>,
}

impl Lines {
    pub fn new(src: &str) -> Lines {
        let mut starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i as u32 + 1);
            }
        }
        Lines { starts }
    }
    pub fn line(&self, offset: u32) -> usize {
        match self.starts.binary_search(&offset) {
            Ok(k) => k + 1,
            Err(k) => k,
        }
    }
    pub fn starts(&self) -> &[u32] {
        &self.starts
    }
}

/// The graph of a parsed program (its debug blocks are ignored). `src` is
/// the text it was parsed from, for line numbers and loop headers.
pub fn graph(program: &Program, src: &str) -> Graph {
    let cx = Cx::of(program);
    let lines = Lines::new(src);
    let mut g = Graph::default();
    for item in &program.items {
        if let Item::Record(r) = item {
            g.nodes.push(GNode {
                kind: "record",
                id: r.name.clone(),
                def: r.name.clone(),
                name: r.name.clone(),
                line: lines.line(r.span.start),
                end: lines.line(r.span.end.saturating_sub(1)),
                params: r.fields.iter().map(|f| f.name.clone()).collect(),
                ..Default::default()
            });
        }
    }
    // the top level is one more def, `<main>`
    let top: Vec<Stmt> = program.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    // a gradient def (crate::grad) is drawn as what it runs: each region of
    // an inlined def is a call of that def, whose own box shows its steps
    let collapsed: HashMap<String, Vec<Stmt>> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Def(d) if crate::grad::grad_def_of(&d.name).is_some() => Some((d.name.clone(), crate::grad::collapse_regions(&d.body))),
            _ => None,
        })
        .collect();
    let mut bodies: Vec<(String, Vec<String>, &[Stmt], Option<Span>)> = Vec::new();
    for item in &program.items {
        // a lambda body's def is part of the lambda's line (crate::surface)
        if let Item::Def(d) = item {
            if crate::update::is_expression_def(&d.name) {
                continue;
            }
            let body: &[Stmt] = collapsed.get(&d.name).map(Vec::as_slice).unwrap_or(&d.body);
            bodies.push((d.name.clone(), d.params.iter().map(|p| p.name.clone()).collect(), body, Some(d.span)));
        }
    }
    bodies.push((MAIN_DEF.to_string(), Vec::new(), &top, None));
    for (name, params, body, span) in bodies {
        let mut f = Flat::default();
        flatten(body, &cx, src, 0, &mut Vec::new(), &mut Vec::new(), &mut f);
        let (reads, writes, globals, calls) = interface(&f, 0, f.leaves.len(), 0, &[], &params);
        let _ = (reads, writes);
        let (line, end) = match span {
            Some(sp) => (lines.line(sp.start), lines.line(sp.end.saturating_sub(1))),
            None => (0, 0),
        };
        let mutated: Vec<String> = cx.mutates.get(&name).map(|m| m.iter().filter_map(|k| params.get(*k).cloned()).collect()).unwrap_or_default();
        // a gradient def is shown in the names of the def it differentiates:
        // its parameters as that def's, without the cell its value comes back
        // in, and none of the names the pass made among what it reads
        let (params, mutated, globals) = if crate::grad::grad_def_of(&name).is_some() {
            let keep = |xs: &[String]| xs.iter().filter(|p| !p.starts_with("viv__ad_")).map(|p| crate::grad::shown(p)).filter(|p| !crate::grad::is_generated(p)).collect::<Vec<_>>();
            (keep(&params), keep(&mutated), globals.into_iter().filter(|v| !crate::grad::is_generated(v)).collect())
        } else {
            (params, mutated, globals)
        };
        g.nodes.push(GNode {
            kind: "def",
            id: name.clone(),
            def: name.clone(),
            name: crate::grad::grad_def_of(&name).map(|f| format!("gradient of {f}")).unwrap_or_else(|| name.clone()),
            line,
            end,
            params: params.clone(),
            reads: params.clone(),
            writes: mutated,
            globals,
            calls: calls.clone(),
            stepped: !f.steps.is_empty(),
            ..Default::default()
        });
        let step_id = |k: usize| format!("{name}/{}", f.steps[k].id);
        let loop_id = |k: usize| format!("{name}/loop@{}", lines.line(f.loops[k].span.start));
        // loops that hold a step are boxes; the rest are statements
        for (k, lp) in f.loops.iter().enumerate() {
            if !lp.has_step {
                continue;
            }
            // the loop's parent: whichever of its step and loop is innermost
            let parent = match (lp.parent_step, lp.parent_loop) {
                (Some(s), Some(l)) => Some(if f.loops[l].start > f.steps[s].start { loop_id(l) } else { step_id(s) }),
                (Some(s), None) => Some(step_id(s)),
                (None, Some(l)) => Some(loop_id(l)),
                (None, None) => Some(name.clone()),
            };
            g.nodes.push(GNode {
                kind: "loop",
                id: loop_id(k),
                def: name.clone(),
                name: lp.header.clone(),
                parent,
                line: lines.line(lp.span.start),
                end: lines.line(lp.span.end.saturating_sub(1)),
                has_step: true,
                ..Default::default()
            });
        }
        for (k, st) in f.steps.iter().enumerate() {
            let (reads, writes, globals, calls) = interface(&f, st.start, st.end, st.guarded, &st.loops, &params);
            let parent = match (st.parent_step, st.parent_loop) {
                (Some(s), Some(l)) => Some(if f.loops[l].start >= f.steps[s].start { loop_id(l) } else { step_id(s) }),
                (Some(s), None) => Some(step_id(s)),
                (None, Some(l)) => Some(loop_id(l)),
                (None, None) => Some(name.clone()),
            };
            for c in &calls {
                g.edges.push(GEdge { kind: "call", from: step_id(k), to: c.clone(), vars: Vec::new(), carried: false });
            }
            g.nodes.push(GNode {
                kind: "step",
                id: step_id(k),
                def: name.clone(),
                name: st.id.clone(),
                text: st.text.clone(),
                parent,
                line: lines.line(st.span.start),
                end: lines.line(st.span.end.saturating_sub(1)),
                reads,
                writes,
                globals,
                calls,
                ..Default::default()
            });
            let _ = st.depth;
        }
        // calls made outside every step belong to the def itself
        let mut in_steps: HashSet<usize> = HashSet::new();
        for st in &f.steps {
            in_steps.extend(st.start..st.end);
        }
        let mut loose: Vec<String> = Vec::new();
        for (k, l) in f.leaves.iter().enumerate() {
            if !in_steps.contains(&k) {
                for c in &l.eff.calls {
                    add(&mut loose, c);
                }
            }
        }
        for c in loose {
            g.edges.push(GEdge { kind: "call", from: name.clone(), to: c, vars: Vec::new(), carried: false });
        }
        // data: for each step, where each variable it reads was last written
        // by another step; a loop-carried value comes from a step later in
        // the same loop, on the previous pass
        let ids: Vec<String> = (0..f.steps.len()).map(step_id).collect();
        let contains = |a: usize, b: usize| f.steps[a].start <= f.steps[b].start && f.steps[b].end <= f.steps[a].end && a != b;
        let mut data: Vec<GEdge> = Vec::new();
        for b in 0..f.steps.len() {
            let node = g.nodes.iter().find(|n| n.id == ids[b]).expect("pushed above");
            for v in node.reads.clone() {
                let writes_v = |a: usize| g.nodes.iter().any(|n| n.id == ids[a] && n.writes.contains(&v));
                let before = (0..f.steps.len()).filter(|&a| a != b && !contains(a, b) && !contains(b, a) && f.steps[a].end <= f.steps[b].start && writes_v(a)).max_by_key(|&a| (f.steps[a].end, std::cmp::Reverse(f.steps[a].depth)));
                let mut src_step = before.map(|a| (a, false));
                // loop-carried: a writer after b inside a loop around b
                if let Some(&lp) = f.steps[b].loops.last() {
                    let (ls, le) = (f.loops[lp].start, f.loops[lp].end);
                    let after = (0..f.steps.len())
                        .filter(|&a| a != b && !contains(a, b) && !contains(b, a) && f.steps[a].start >= f.steps[b].end && f.steps[a].end <= le && f.steps[a].start >= ls && writes_v(a))
                        .max_by_key(|&a| (f.steps[a].end, std::cmp::Reverse(f.steps[a].depth)));
                    if let Some(a) = after {
                        // the value may come from before the loop on the first pass
                        // and from the later step on every other; show the later step
                        if before.map_or(true, |p| f.steps[p].end <= ls) {
                            src_step = Some((a, true));
                        }
                    }
                }
                if let Some((a, carried)) = src_step {
                    match data.iter_mut().find(|e| e.from == ids[a] && e.to == ids[b] && e.carried == carried) {
                        Some(e) => add(&mut e.vars, &v),
                        None => data.push(GEdge { kind: "data", from: ids[a].clone(), to: ids[b].clone(), vars: vec![v.clone()], carried }),
                    }
                }
            }
        }
        g.edges.extend(data);
    }
    // each def's own nodes in source order, the def first
    let order: HashMap<String, usize> = g.nodes.iter().enumerate().filter(|(_, n)| n.kind == "def" || n.kind == "record").map(|(i, n)| (n.id.clone(), i)).collect();
    g.nodes.sort_by_key(|n| (order.get(&n.def).copied().unwrap_or(usize::MAX), n.kind != "def", n.line));
    g
}

/// Parse `src` and build its graph. Parse errors come back as text.
pub fn graph_of_source(src: &str) -> Result<Graph, String> {
    let mut program = crate::parse(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    program.debug.clear();
    Ok(graph(&program, src))
}

fn list(xs: &[String]) -> String {
    format!("[{}]", xs.iter().map(|x| json_str(x)).collect::<Vec<_>>().join(","))
}

/// The graph as JSON (see the design, section 11).
pub fn nodes_edges_json(g: &Graph) -> String {
    let nodes: Vec<String> = g
        .nodes
        .iter()
        .map(|n| {
            let mut f = vec![format!("\"id\":{}", json_str(&n.id)), format!("\"kind\":\"{}\"", n.kind), format!("\"def\":{}", json_str(&n.def)), format!("\"line\":{}", n.line), format!("\"end\":{}", n.end)];
            f.push(format!("\"parent\":{}", n.parent.as_deref().map(json_str).unwrap_or_else(|| "null".into())));
            match n.kind {
                "def" => {
                    f.push(format!("\"params\":{}", list(&n.params)));
                    f.push(format!("\"writes\":{}", list(&n.writes)));
                    f.push(format!("\"globals\":{}", list(&n.globals)));
                    f.push(format!("\"calls\":{}", list(&n.calls)));
                    f.push(format!("\"stepped\":{}", n.stepped));
                    // a gradient def (crate::grad) reads as "gradient of f"
                    if n.name != n.id {
                        f.push(format!("\"label\":{}", json_str(&n.name)));
                    }
                }
                "step" => {
                    f.push(format!("\"step\":{}", json_str(&n.name)));
                    f.push(format!("\"text\":{}", json_str(&n.text)));
                    f.push(format!("\"reads\":{}", list(&n.reads)));
                    f.push(format!("\"writes\":{}", list(&n.writes)));
                    f.push(format!("\"globals\":{}", list(&n.globals)));
                    f.push(format!("\"calls\":{}", list(&n.calls)));
                }
                "loop" => f.push(format!("\"header\":{}", json_str(&n.name))),
                "record" => f.push(format!("\"fields\":{}", list(&n.params))),
                _ => {}
            }
            format!("{{{}}}", f.join(","))
        })
        .collect();
    let edges: Vec<String> = g
        .edges
        .iter()
        .map(|e| {
            let mut s = format!("{{\"kind\":\"{}\",\"from\":{},\"to\":{}", e.kind, json_str(&e.from), json_str(&e.to));
            if e.kind == "data" {
                s.push_str(&format!(",\"vars\":{},\"carried\":{}", list(&e.vars), e.carried));
            }
            s.push('}');
            s
        })
        .collect();
    format!("\"nodes\":[{}],\"edges\":[{}]", nodes.join(","), edges.join(","))
}

/// The graph as text, for `viv graph`: each def with steps, its steps
/// indented under it with their interfaces.
pub fn graph_text(g: &Graph) -> String {
    let mut out = String::new();
    fn kids<'a>(g: &'a Graph, parent: &str) -> Vec<&'a GNode> {
        g.nodes.iter().filter(|n| n.parent.as_deref() == Some(parent)).collect()
    }
    fn go(g: &Graph, n: &GNode, indent: usize, out: &mut String) {
        let pad = "  ".repeat(indent);
        match n.kind {
            "step" => {
                out.push_str(&format!("{pad}step {}", n.name));
                if !n.text.is_empty() {
                    out.push_str(&format!("  \"{}\"", n.text));
                }
                out.push('\n');
                let io = |label: &str, xs: &[String], out: &mut String| {
                    if !xs.is_empty() {
                        out.push_str(&format!("{pad}  {label} {}\n", xs.join(", ")));
                    }
                };
                io("in: ", &n.reads, out);
                io("out:", &n.writes, out);
                io("uses:", &n.globals, out);
                io("calls:", &n.calls, out);
            }
            "loop" => out.push_str(&format!("{pad}{}\n", n.name)),
            _ => {}
        }
        for k in kids(g, &n.id) {
            go(g, k, indent + 1, out);
        }
    }
    for d in g.nodes.iter().filter(|n| n.kind == "def" && n.stepped) {
        out.push_str(&format!("{}({})\n", d.id, d.params.join(", ")));
        for k in kids(g, &d.id) {
            go(g, k, 1, &mut out);
        }
        let flows: Vec<&GEdge> = g.edges.iter().filter(|e| e.kind == "data" && g.node(&e.to).is_some_and(|n| n.def == d.id)).collect();
        for e in flows {
            let short = |s: &str| s.rsplit('/').next().unwrap_or(s).to_string();
            out.push_str(&format!("  {} -> {}: {}{}\n", short(&e.from), short(&e.to), e.vars.join(", "), if e.carried { "  (next pass)" } else { "" }));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Lines of the expanded program, back to their modules
// ---------------------------------------------------------------------------

/// Which lines of a module `expand_uses` keeps: everything but its `test`
/// blocks and its `#!` lines other than `#! use` (the same rule, written
/// again here; `a_modules_lines_map_back` holds the two together).
pub fn kept_lines(text: &str) -> Vec<bool> {
    // an import statement's lines after its first are dropped (crate::imports)
    let continued: std::collections::HashSet<usize> = crate::imports::scan(text).unwrap_or_default().iter().flat_map(|s| s.line + 1..s.line + s.nlines).collect();
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut in_test = false;
    for l in text.lines() {
        if !in_test && l.trim_start().starts_with("test ") && strip_comment(l).trim_end().ends_with('{') {
            in_test = true;
            depth = 0;
        }
        if in_test {
            let mut quoted = false;
            let mut prev_backslash = false;
            for ch in l.chars() {
                if quoted {
                    if ch == '"' && !prev_backslash {
                        quoted = false;
                    }
                    prev_backslash = ch == '\\' && !prev_backslash;
                    continue;
                }
                match ch {
                    '"' => quoted = true,
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    '#' => break,
                    _ => {}
                }
                prev_backslash = false;
            }
            if depth <= 0 {
                in_test = false;
            }
            out.push(false);
            continue;
        }
        out.push((l.starts_with("#! use ") || !l.starts_with("#!")) && !continued.contains(&out.len()));
    }
    out
}

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

/// For each line of the expanded program (1-based, index 0 unused): the
/// module it came from (an index into `names`, the root first) and its line
/// there; `None` for the marker lines `expand_uses` writes. `texts` gives
/// each used module's source by name.
pub fn line_map(root: &str, root_name: &str, expanded: &str, texts: &HashMap<String, String>) -> (Vec<String>, Vec<Option<(usize, usize)>>) {
    // The expander's own record of where each line came from, when the
    // modules in `texts` expand to `expanded` again: exact, `#! if-used`
    // blocks included. A module in another language has no text here that
    // expands, and is mapped by counting kept lines below.
    let mut resolve = |name: &str| texts.get(name).cloned().ok_or_else(|| format!("no text for {name}"));
    if let Ok(e) = crate::expand_uses_mapped(root, &mut resolve, &mut |lang, name, _| Err(format!("{name}: {lang}"))) {
        if e.text == expanded {
            let names: Vec<String> = e.units.iter().enumerate().map(|(k, u)| if k == 0 { root_name.to_string() } else { u.name.clone() }).collect();
            let mut out: Vec<Option<(usize, usize)>> = vec![None];
            if e.lines.is_empty() {
                out.extend((1..=expanded.lines().count()).map(|n| Some((0, n))));
            } else {
                out.extend(e.lines.iter().map(|o| o.map(|(u, n)| (u, n + 1))));
            }
            return (names, out);
        }
    }
    let mut names = vec![root_name.to_string()];
    let mut out: Vec<Option<(usize, usize)>> = vec![None];
    // a stack of (module index, the module's kept line numbers, next one)
    let kept_of = |t: &str| -> Vec<usize> { kept_lines(t).iter().enumerate().filter(|(_, k)| **k).map(|(i, _)| i + 1).collect() };
    let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(0, (1..=root.lines().count()).collect(), 0)];
    for l in expanded.lines() {
        if let Some(rest) = l.strip_prefix("# ---- #! use ") {
            let top = stack.last_mut().expect("the root");
            top.2 += 1; // the `#! use` line the marker stands for
            if rest.ends_with("(already included) ----") {
                out.push(None);
                continue;
            }
            let name = rest.trim_end_matches(" ----").to_string();
            let k = names.iter().position(|n| *n == name).unwrap_or_else(|| {
                names.push(name.clone());
                names.len() - 1
            });
            let kept = texts.get(&name).map(|t| kept_of(t)).unwrap_or_default();
            stack.push((k, kept, 0));
            out.push(None);
            continue;
        }
        if l.starts_with("# ---- end of ") && stack.len() > 1 {
            stack.pop();
            out.push(None);
            continue;
        }
        let top = stack.last_mut().expect("the root");
        let line = top.1.get(top.2).copied();
        top.2 += 1;
        out.push(line.map(|n| (top.0, n)));
    }
    (names, out)
}

/// The line map as runs: `[expanded line, module, module line, count]`.
pub fn line_map_json(names: &[String], map: &[Option<(usize, usize)>]) -> String {
    let mut runs: Vec<[usize; 4]> = Vec::new();
    for (x, m) in map.iter().enumerate() {
        let Some((k, n)) = m else { continue };
        if let Some(r) = runs.last_mut() {
            if r[1] == *k && r[0] + r[3] == x && r[2] + r[3] == *n {
                r[3] += 1;
                continue;
            }
        }
        runs.push([x, *k, *n, 1]);
    }
    format!(
        "\"modules\":{},\"lineMap\":[{}]",
        list(names),
        runs.iter().map(|r| format!("[{},{},{},{}]", r[0], r[1], r[2], r[3])).collect::<Vec<_>>().join(",")
    )
}

// ---------------------------------------------------------------------------
// The bundle the explorer loads
// ---------------------------------------------------------------------------

/// Every value the line-level recording may watch, by def (`<main>` for the
/// top level): the variables whose type a watch records (numbers, booleans,
/// real arrays) in every compiled copy of the def.
pub fn watchable(src: &str) -> HashMap<String, HashSet<String>> {
    use crate::lower::{Elem, Ty, Viv};
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    let Ok(ir) = crate::build_ir_plain(src, SourceId(0)) else { return out };
    let ok = |t: &Ty| matches!(t, Ty::Num | Ty::Bool | Ty::Arr(Elem::Num));
    let mut seen: HashMap<String, HashMap<String, bool>> = HashMap::new();
    // a specialization (`lmder__mgh_fcn`) belongs to its source def
    let ast_defs: HashSet<String> = crate::parse(src, SourceId(0)).map(|p| p.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect()).unwrap_or_default();
    let source_of = |name: &str| -> String {
        if ast_defs.contains(name) {
            return name.to_string();
        }
        let mut cut = name.len();
        while let Some(k) = name[..cut].rfind("__") {
            if ast_defs.contains(&name[..k]) {
                return name[..k].to_string();
            }
            cut = k;
        }
        name.to_string()
    };
    for root in ir.roots() {
        let node = ir.get(*root);
        let (def, vars): (String, Vec<&(String, Ty)>) = match &node.kind {
            Viv::Func { name, params, locals, .. } => (source_of(name), params.iter().chain(locals.iter()).collect()),
            Viv::Main { locals, globals } => (MAIN_DEF.to_string(), locals.iter().chain(globals.iter()).collect()),
            _ => continue,
        };
        let m = seen.entry(def).or_default();
        let here: HashMap<&str, bool> = vars.iter().map(|(n, t)| (n.as_str(), ok(t))).collect();
        // a name must be watchable in every copy that has it
        for (n, w) in &here {
            let e = m.entry(n.to_string()).or_insert(true);
            *e = *e && *w;
        }
    }
    for (def, m) in seen {
        out.insert(def, m.into_iter().filter(|(_, w)| *w).map(|(n, _)| n).collect());
    }
    out
}

/// The table of statements a line-level recording records, for a program
/// as run (its debug blocks included).
pub fn lines_table(src: &str) -> Vec<crate::debug_lens::LineRec> {
    lines_table_with(src, &HashSet::new())
}

/// `lines_table`, with defs without steps recorded line by line too.
pub fn lines_table_with(src: &str, defs: &HashSet<String>) -> Vec<crate::debug_lens::LineRec> {
    let opts = crate::debug_lens::LineOpts { starts: Lines::new(src).starts().to_vec(), watchable: watchable(src), first: 1, iters: HashMap::new(), defs: defs.clone(), calls: HashMap::new(), every: HashSet::new() };
    let (_, table) = crate::debug_lens::with_lines(opts, || crate::parse(src, SourceId(0)).ok().map(crate::debug_lens::instrument));
    table
}

fn lines_json(table: &[crate::debug_lens::LineRec]) -> String {
    let rows: Vec<String> = table
        .iter()
        .map(|r| {
            format!(
                "{{\"def\":{},\"line\":{},\"end\":{},\"kind\":\"{}\",\"step\":{},\"watch\":{},\"gated\":{}}}",
                json_str(&r.def),
                r.line,
                r.end,
                r.kind,
                r.step.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                list(&r.watch),
                r.gated
            )
        })
        .collect();
    format!("[{}]", rows.join(","))
}

/// The program the explorer runs: the expanded source, the debug blocks of
/// every module (`debug` source, from their `.debug` files), and an empty
/// `debug * { }`, which turns the story on for a program that has no lens.
pub fn explore_program(expanded: &str, debug: &str) -> String {
    let mut p = expanded.to_string();
    if !p.ends_with('\n') {
        p.push('\n');
    }
    p.push_str(debug);
    if !p.ends_with('\n') {
        p.push('\n');
    }
    p.push_str("debug * {\n}\n");
    p
}

/// What the explorer loads for one program (section
/// 11): the program it runs, its graph, the line table, the map back to each
/// module's lines, and the intent and debug lenses of every def.
pub fn bundle_json(input: &crate::lens::Input, debug: &str) -> Result<String, String> {
    let program = explore_program(&input.expanded, debug);
    let parsed = crate::parse(&program, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    let mut plain = crate::parse(&program, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    plain.debug.clear();
    let g = graph(&plain, &program);
    // the intent, loaded now: its line anchors decide which defs without
    // steps are recorded line by line
    let loaded = crate::lens::load(input)?;
    let cx = Cx::of(&plain);
    let plines = Lines::new(&program);
    let mut details: Vec<String> = Vec::new();
    let mut line_defs: Vec<String> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    for m in std::iter::once(&loaded.root).chain(loaded.used.iter()) {
        let (_, entries) = m.entries();
        for e in &entries {
            let Some(def) = plain.items.iter().find_map(|i| if let Item::Def(d) = i { (d.name == e.name).then_some(d) } else { None }) else { continue };
            let cands = anchor_candidates(def, &cx, &plines);
            let anchors = entry_anchors(&e.details, &e.outline);
            if anchors.iter().any(|(s, d)| s.is_none() && d.label == "=") && !g.node(&e.name).is_some_and(|n| n.stepped) {
                line_defs.push(e.name.clone());
            }
            for (step, d) in anchors {
                let line = if d.label == "=" { resolve_anchor(&cands, step.as_deref(), &d) } else { None };
                if d.label == "=" && line.is_none() {
                    problems.push(format!("{}: `= {}` in {} names no statement", m.text.intent_path, d.anchor, e.name));
                }
                details.push(format!(
                    "{{\"def\":{},\"step\":{},\"label\":{},\"anchor\":{},\"text\":{},\"line\":{},\"under\":{}}}",
                    json_str(&e.name),
                    step.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                    json_str(&d.label),
                    json_str(&d.anchor),
                    json_str(&d.text),
                    line.map(|l| l.to_string()).unwrap_or_else(|| "null".into()),
                    json_str(&d.under)
                ));
            }
        }
    }
    let table = lines_table_with(&program, &line_defs.iter().cloned().collect());
    // the lines, back to their modules
    let mut texts: HashMap<String, String> = HashMap::new();
    for m in &input.used {
        texts.insert(m.name.clone(), m.src.clone());
    }
    let (names, map) = line_map(&input.root.src, &input.root.name, &input.expanded, &texts);
    // the intent lens of every def, both homes merged
    let words = crate::lens::story_words(&loaded);
    let mut intents: Vec<String> = Vec::new();
    let mut module_texts: Vec<String> = Vec::new();
    // each def's symbols (the glossary's `$...$`), and the defs the reading
    // view explains: those with intent text or steps
    let mut symbols: HashMap<String, crate::tex::Symbols> = HashMap::new();
    let mut explained: HashSet<String> = HashSet::new();
    let mut every_var: Vec<String> = Vec::new();
    for m in std::iter::once(&loaded.root).chain(loaded.used.iter()) {
        let (module, entries) = m.entries();
        if let Some(me) = &module {
            module_texts.push(format!("{}:{}", json_str(&m.text.name), json_str(&me.text)));
            if me.every_var {
                every_var.push(m.text.name.clone());
            }
        }
        for d in &m.defs {
            let e = entries.iter().find(|e| e.name == d.name);
            let mg = crate::lens::merged(d, e);
            let mut fields: Vec<String> = mg.fields.iter().filter(|(_, t, _)| !t.is_empty()).map(|(k, t, _)| format!("{}:{}", json_str(k), json_str(t))).collect();
            // why the def is needed beyond its callers: who calls it from outside the programs, and what it serves
            // (`entry` in the bundle already says whether the def has a block
            // in an `.intent` file, so the field goes out as `entryPoint`)
            for (k, out) in [("entry", "entryPoint"), ("serves", "serves")] {
                if let Some(t) = e.and_then(|e| e.field(k)).filter(|t| !t.text.trim().is_empty()) {
                    fields.push(format!("{}:{}", json_str(out), json_str(&t.text)));
                }
            }
            let mut syms = crate::tex::Symbols::new();
            // the variable docs, each with the program line it starts to hold at
            let keys: Vec<String> = mg.gloss.iter().map(|g| g.0.clone()).collect();
            let (docs, vars) = match body_of(&plain, &d.name) {
                Some((body, params, start)) if e.is_some() || keys.iter().any(|k| k.contains('@')) => {
                    let sites = doc_sites(&body, &params, plines.line(start as u32), &cx, &plines, &keys);
                    let stmts = doc_statements(&body, &cx, &plines);
                    let docs: Vec<String> = sites
                        .iter()
                        .zip(mg.gloss.iter())
                        .map(|((key, name, at), (_, t, _))| {
                            let (sym, text) = crate::tex::split_symbol(t);
                            let (step, nth) = match crate::lens::split_doc_key(key) {
                                Some((_, st, k)) => (st, k),
                                None => (None, None),
                            };
                            format!(
                                "{{\"key\":{},\"name\":{},\"step\":{},\"nth\":{},\"symbol\":{},\"text\":{},\"line\":{}}}",
                                json_str(key),
                                json_str(name),
                                step.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                                nth.map(|k| k.to_string()).unwrap_or_else(|| "null".into()),
                                sym.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                                json_str(&text),
                                at.as_ref().map(|l| l.to_string()).unwrap_or_else(|_| "null".into())
                            )
                        })
                        .collect();
                    (docs, def_variables(&params, &stmts))
                }
                _ => (Vec::new(), Vec::new()),
            };
            let gloss: Vec<String> = mg
                .gloss
                .iter()
                .filter(|(k, _, _)| !k.contains('@'))
                .map(|(k, t, _)| {
                    let (sym, text) = crate::tex::split_symbol(t);
                    if let Some(sym) = sym {
                        syms.insert(k.clone(), sym);
                    }
                    format!("[{},{}]", json_str(k), json_str(&text))
                })
                .collect();
            // the reading view gives a section to a def with an `.intent`
            // entry or steps; a def documented only inline is a one-liner
            if e.is_some() || g.node(&d.name).is_some_and(|n| n.stepped) || line_defs.contains(&d.name) {
                explained.insert(d.name.clone());
            }
            let mut sk: Vec<(&String, &String)> = syms.iter().collect();
            sk.sort();
            intents.push(format!(
                "{}:{{\"module\":{},\"sig\":{},\"entry\":{},{}\"glossary\":[{}],\"symbols\":{{{}}},\"docs\":[{}],\"vars\":{}}}",
                json_str(&d.name),
                json_str(&m.text.name),
                json_str(&d.sig),
                e.is_some(),
                fields.iter().map(|f| format!("{f},")).collect::<String>(),
                gloss.join(","),
                sk.iter().map(|(k, v)| format!("{}:{}", json_str(k), json_str(v))).collect::<Vec<_>>().join(","),
                docs.join(","),
                list(&vars)
            ));
            symbols.insert(d.name.clone(), syms);
        }
    }
    explained.insert(MAIN_DEF.to_string());
    let tex_rows = tex_table(&plain, &cx, &plines, &explained, &symbols);
    let steps: Vec<String> = {
        let mut ks: Vec<(&String, &String)> = words.steps.iter().collect();
        ks.sort();
        ks.iter().map(|(k, v)| format!("{}:{}", json_str(k), json_str(v))).collect()
    };
    // the debug lens, as written
    let mut dbg: Vec<String> = Vec::new();
    for b in &parsed.debug {
        let Some(def) = &b.def else { continue };
        for item in &b.items {
            let (kind, pos, span) = match item {
                DebugItem::Watch { pos, span, .. } => ("watch", Some(pos), *span),
                DebugItem::Hold { pos, span, .. } => ("hold", Some(pos), *span),
                DebugItem::Trace { span, .. } => ("trace", None, *span),
            };
            let at = match pos {
                Some(DebugPos::At(id, _)) => id.clone(),
                Some(DebugPos::PerIteration) => crate::debug_lens::PER_ITERATION.to_string(),
                None => String::new(),
            };
            let text = program.get(span.start as usize..span.end as usize).unwrap_or("").split_whitespace().collect::<Vec<_>>().join(" ");
            let sentence = match item {
                DebugItem::Hold { sentence, .. } => json_str(sentence),
                _ => "null".into(),
            };
            // a watch's names, computed ones (`name = expr`) included
            let names = match item {
                DebugItem::Watch { names, .. } => format!("[{}]", names.iter().map(|(n, _)| json_str(n)).collect::<Vec<_>>().join(",")),
                _ => "null".into(),
            };
            dbg.push(format!("{{\"def\":{},\"kind\":\"{kind}\",\"at\":{},\"text\":{},\"sentence\":{sentence},\"names\":{names}}}", json_str(def), json_str(&at), json_str(&text)));
        }
    }
    // the root module's reading paths, each stop with the program line of
    // the statement it names, when it names one
    let mut paths: Vec<String> = Vec::new();
    if let Some(f) = &loaded.root.intent {
        for pb in f.paths() {
            let stops: Vec<String> = pb
                .stops
                .iter()
                .map(|st| {
                    let line = st.anchor.as_ref().and_then(|a| {
                        let def = plain.items.iter().find_map(|i| if let Item::Def(dd) = i { (dd.name == st.def).then_some(dd) } else { None })?;
                        let cands = anchor_candidates(def, &cx, &plines);
                        let dt = crate::lens::Detail { label: "=".into(), anchor: a.clone(), text: String::new(), line: st.line, under: String::new() };
                        resolve_anchor(&cands, st.step.as_deref(), &dt)
                    });
                    let dets: Vec<String> = st.details.iter().map(|d| format!("{{\"label\":{},\"text\":{}}}", json_str(&d.label), json_str(&d.text))).collect();
                    format!(
                        "{{\"target\":{},\"def\":{},\"step\":{},\"anchor\":{},\"text\":{},\"line\":{},\"details\":[{}]}}",
                        json_str(&st.target),
                        json_str(&st.def),
                        st.step.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                        st.anchor.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                        json_str(&st.text),
                        line.map(|l| l.to_string()).unwrap_or_else(|| "null".into()),
                        dets.join(",")
                    )
                })
                .collect();
            paths.push(format!("{{\"id\":{},\"title\":{},\"stops\":[{}]}}", json_str(&pb.id), json_str(&pb.title), stops.join(",")));
        }
    }
    Ok(format!(
        "{{\"format\":1,\"name\":{},\"paths\":[{}],\"everyVar\":{},\"program\":{},{},\"lines\":{},\"tex\":[{}],\"lineDefs\":{},\"details\":[{}],\"problems\":{},{},\"intent\":{{{}}},\"steps\":{{{}}},\"moduleText\":{{{}}},\"debug\":[{}]}}",
        json_str(&input.root.name),
        paths.join(","),
        list(&every_var),
        json_str(&program),
        nodes_edges_json(&g),
        lines_json(&table),
        tex_rows.join(","),
        list(&line_defs),
        details.join(","),
        list(&problems),
        line_map_json(&names, &map),
        intents.join(","),
        steps.join(","),
        module_texts.join(","),
        dbg.join(",")
    ))
}

/// The statement table of the reading view: every assignment, condition,
/// loop header, return and call statement of the explained defs (and the top
/// level), each with its LaTeX (`crate::tex`), the defs it calls and the names
/// it reads or writes. Printed from the syntax tree of the program as run, so
/// the math shown for a line is the line's.
fn tex_table(program: &Program, cx: &Cx, lines: &Lines, explained: &HashSet<String>, symbols: &HashMap<String, crate::tex::Symbols>) -> Vec<String> {
    fn go(def: &str, body: &[Stmt], cx: &Cx, lines: &Lines, syms: &crate::tex::Symbols, out: &mut Vec<String>) {
        for s in body {
            if let Some(t) = crate::tex::stmt(s, syms) {
                let mut e = Eff::default();
                let compound = match s {
                    Stmt::If { cond, .. } | Stmt::While { cond, .. } => {
                        expr_eff(cond, cx, &mut e);
                        true
                    }
                    Stmt::For { var, bound, .. } => {
                        expr_eff(bound, cx, &mut e);
                        add(&mut e.writes, var);
                        true
                    }
                    other => {
                        stmt_eff(other, cx, &mut e);
                        false
                    }
                };
                let sp = s.span();
                let line = lines.line(sp.start);
                let end = if compound { line } else { lines.line(sp.end.saturating_sub(1)).max(line) };
                let mut names = e.reads.clone();
                for w in &e.writes {
                    add(&mut names, w);
                }
                out.push(format!(
                    "{{\"def\":{},\"line\":{line},\"end\":{end},\"kind\":\"{}\",\"tex\":{},\"calls\":{},\"writes\":{},\"names\":{}}}",
                    json_str(def),
                    t.kind,
                    json_str(&t.tex),
                    list(&e.calls),
                    list(&e.writes),
                    list(&names)
                ));
            }
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    go(def, then_body, cx, lines, syms, out);
                    go(def, else_body, cx, lines, syms, out);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } => go(def, body, cx, lines, syms, out),
                _ => {}
            }
        }
    }
    let none = crate::tex::Symbols::new();
    let mut out = Vec::new();
    let mut top: Vec<Stmt> = Vec::new();
    for item in &program.items {
        match item {
            Item::Def(d) if explained.contains(&d.name) => go(&d.name, &d.body, cx, lines, symbols.get(&d.name).unwrap_or(&none), &mut out),
            Item::Stmt(s) => top.push(s.clone()),
            _ => {}
        }
    }
    go(MAIN_DEF, &top, cx, lines, symbols.get(MAIN_DEF).unwrap_or(&none), &mut out);
    out
}

/// What a line-level recording records beyond its defaults (the first
/// pass of every iteration loop, and the first call of every def with steps).
#[derive(Clone, Debug, Default)]
pub struct Iters {
    /// more passes to record line by line, by def
    pub iters: HashMap<String, Vec<i64>>,
    /// defs without steps to record line by line (in their first call)
    pub defs: HashSet<String>,
    /// more calls to record line by line, by def
    pub calls: HashMap<String, Vec<i64>>,
    /// defs whose nested loops holding a step are recorded in every pass
    pub every: HashSet<String>,
}

impl Iters {
    fn opts(&self, src: &str) -> crate::debug_lens::LineOpts {
        crate::debug_lens::LineOpts { starts: Lines::new(src).starts().to_vec(), watchable: watchable(src), first: 1, iters: self.iters.clone(), defs: self.defs.clone(), calls: self.calls.clone(), every: self.every.clone() }
    }
}

/// Run a program as the explorer does: in the interpreter, in debug mode,
/// with line-level recording on. Returns the run's output (the `__story__`
/// line included) and the number of failed asserts and holds.
pub fn explore_run(src: &str, req: &Iters) -> Result<(String, u32, u32), String> {
    crate::debug_lens::with_lines(req.opts(src), || crate::interp::run_checked_holds(src, false)).0
}

/// The C of a program with line-level recording on, for checking that the
/// compiled program records the same story (`tests/explorer.rs`).
pub fn explore_c(src: &str, req: &Iters) -> Result<String, String> {
    crate::debug_lens::with_lines(req.opts(src), || crate::compile_c(src)).0
}

/// `{"src": "...", "iters": {"def": [n, ...]}}`, as the wasm export takes it.
pub fn explore_run_json(request: &str) -> String {
    let req = match crate::story::parse_json(request) {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let crate::story::Json::Obj(fields) = &req else { return "error: the request is not an object".into() };
    let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v);
    let Some(crate::story::Json::Str(src)) = get("src") else { return "error: the request has no src".into() };
    let by_def = |k: &str| -> HashMap<String, Vec<i64>> {
        let mut out = HashMap::new();
        if let Some(crate::story::Json::Obj(by)) = get(k) {
            for (def, v) in by {
                if let crate::story::Json::Arr(xs) = v {
                    out.insert(def.clone(), xs.iter().filter_map(|x| if let crate::story::Json::Num(n) = x { n.parse::<f64>().ok().map(|f| f as i64) } else { None }).collect());
                }
            }
        }
        out
    };
    let mut iters = Iters { iters: by_def("iters"), calls: by_def("calls"), defs: HashSet::new(), every: HashSet::new() };
    let names = |k: &str| -> HashSet<String> {
        match get(k) {
            Some(crate::story::Json::Arr(ds)) => ds.iter().filter_map(|d| if let crate::story::Json::Str(s) = d { Some(s.clone()) } else { None }).collect(),
            _ => HashSet::new(),
        }
    };
    iters.defs = names("defs");
    iters.every = names("every");
    match explore_run(src, &iters) {
        Ok((out, fails, holds)) => format!("{out}__explore__ {{\"fails\":{fails},\"holds\":{holds}}}\n"),
        Err(e) => format!("error: {e}"),
    }
}

/// `{"root": {name, src, intent?, debug?}, "used": [{name, src, intent?, debug?}]}`:
/// the texts a program is made of, as the wasm export takes them. The
/// program is expanded here (`#! use` resolved against `used`), so the
/// browser and the command line build the same bundle.
pub fn bundle_request_json(request: &str) -> String {
    use crate::story::Json;
    let req = match crate::story::parse_json(request) {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let obj = |j: &Json, k: &str| -> Option<Json> { if let Json::Obj(f) = j { f.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()) } else { None } };
    let s = |j: &Json, k: &str| -> Option<String> { if let Some(Json::Str(t)) = obj(j, k) { Some(t) } else { None } };
    let module = |j: &Json| -> crate::lens::ModuleText {
        let name = s(j, "name").unwrap_or_default();
        crate::lens::ModuleText { path: format!("{name}.viv"), intent_path: format!("{name}.intent"), src: s(j, "src").unwrap_or_default(), intent: s(j, "intent"), name }
    };
    let Some(root_j) = obj(&req, "root") else { return "error: the request has no root".into() };
    let root = module(&root_j);
    let used_j: Vec<Json> = match obj(&req, "used") {
        Some(Json::Arr(xs)) => xs,
        _ => Vec::new(),
    };
    let by_name: HashMap<String, Json> = used_j.iter().map(|j| (s(j, "name").unwrap_or_default(), j.clone())).collect();
    let mut order: Vec<String> = Vec::new();
    let expanded = match crate::expand_uses(&root.src, &mut |name: &str| {
        let j = by_name.get(name).ok_or_else(|| format!("module `{name}`: the request does not carry it"))?;
        if !order.iter().any(|o| o == name) {
            order.push(name.to_string());
        }
        Ok(s(j, "src").unwrap_or_default())
    }) {
        Ok(e) => e,
        Err(e) => return format!("error: {e}"),
    };
    let mut debug = String::new();
    for j in std::iter::once(&root_j).chain(order.iter().filter_map(|n| by_name.get(n))) {
        if let Some(d) = s(j, "debug") {
            if let Err(e) = crate::debug_lens::check_file(&d) {
                return format!("error: {}.debug: {e}", s(j, "name").unwrap_or_default());
            }
            debug.push_str(&crate::debug_lens::file_to_source(&d));
        }
    }
    let input = crate::lens::Input { root, used: order.iter().filter_map(|n| by_name.get(n)).map(module).collect(), expanded, companion_tests: Vec::new() };
    let bundle = match bundle_json(&input, &debug) {
        Ok(b) => b,
        Err(e) => return format!("error: {e}"),
    };
    // the architecture lens, when the request carries one (`arch`: its path
    // and text; `agreeing`: the members each stage was seen agreeing with,
    // from arch_ratchet.json): the program's multi-paths, for the picture
    let Some(arch_j) = obj(&req, "arch") else { return bundle };
    let (Some(apath), Some(atext)) = (s(&arch_j, "path"), s(&arch_j, "text")) else { return "error: `arch` needs a path and a text".into() };
    let mut agreement = crate::arch::Agreement::new();
    if let Some(Json::Obj(fs)) = obj(&req, "agreeing") {
        for (k, v) in fs {
            if let Json::Arr(ms) = v {
                agreement.insert(k, ms.iter().filter_map(|m| m.str().map(String::from)).collect());
            }
        }
    }
    let mods: Vec<crate::arch::ModSrc> = std::iter::once(&input.root)
        .chain(input.used.iter())
        .map(|m| crate::arch::ModSrc { name: m.name.clone(), path: m.path.clone(), src: m.src.clone(), intent: m.intent.clone() })
        .collect();
    // `arch.modules`: modules the program does not load that define what the
    // lens's stages name, for their implementations and intent
    let extra: Vec<crate::arch::ModSrc> = match obj(&arch_j, "modules") {
        Some(Json::Arr(xs)) => xs
            .iter()
            .map(|j| {
                let m = module(j);
                crate::arch::ModSrc { name: m.name, path: m.path, src: m.src, intent: m.intent }
            })
            .collect(),
        _ => Vec::new(),
    };
    let g = match graph_of_source(&input.expanded) {
        Ok(g) => g,
        Err(e) => return format!("error: {e}"),
    };
    let mp = crate::arch::bundle_stages(&apath, &atext, &mods, &extra, &g, &agreement);
    let body = bundle.trim_end();
    format!("{},\"multipaths\":{mp}}}", body.strip_suffix('}').unwrap_or(body))
}

// ---------------------------------------------------------------------------
// Line anchors
// ---------------------------------------------------------------------------

/// The statements of a def that an anchor can name, in source order, nested
/// ones included (a statement inside a loop without steps can be explained,
/// and the explorer shows it with that loop's line): (step around it, first
/// line, what it writes, what it calls).
pub fn anchor_candidates(def: &Def, cx: &Cx, lines: &Lines) -> Vec<(Option<String>, usize, Vec<String>, Vec<String>)> {
    fn go(body: &[Stmt], cx: &Cx, lines: &Lines, steps: &mut Vec<String>, out: &mut Vec<(Option<String>, usize, Vec<String>, Vec<String>)>) {
        for s in body {
            match s {
                Stmt::StepEnter { id, .. } => steps.push(id.clone()),
                Stmt::StepExit { .. } => {
                    steps.pop();
                }
                Stmt::If { then_body, else_body, cond, span } => {
                    let mut e = Eff::default();
                    expr_eff(cond, cx, &mut e);
                    out.push((steps.last().cloned(), lines.line(span.start), Vec::new(), e.calls));
                    go(then_body, cx, lines, steps, out);
                    go(else_body, cx, lines, steps, out);
                }
                Stmt::While { body, cond, span } | Stmt::For { body, bound: cond, span, .. } => {
                    let mut e = Eff::default();
                    expr_eff(cond, cx, &mut e);
                    if let Stmt::For { var, .. } = s {
                        add(&mut e.writes, var);
                    }
                    out.push((steps.last().cloned(), lines.line(span.start), e.writes, e.calls));
                    go(body, cx, lines, steps, out);
                }
                Stmt::Test { .. } | Stmt::Lens { .. } => {}
                other => {
                    let mut e = Eff::default();
                    stmt_eff(other, cx, &mut e);
                    out.push((steps.last().cloned(), lines.line(other.span().start), e.writes, e.calls));
                }
            }
        }
    }
    let mut out = Vec::new();
    go(&def.body, cx, lines, &mut Vec::new(), &mut out);
    out
}

/// The first line of the statement a detail's anchor names, inside `step`
/// (or anywhere in the def when `step` is None), or None.
pub fn resolve_anchor(cands: &[(Option<String>, usize, Vec<String>, Vec<String>)], step: Option<&str>, d: &crate::lens::Detail) -> Option<usize> {
    let (name, call, k) = d.target();
    cands
        .iter()
        .filter(|(s, _, _, _)| step.is_none() || s.as_deref() == step)
        .filter(|(_, _, w, c)| if call { c.contains(&name) } else { w.contains(&name) })
        .nth(k - 1)
        .map(|(_, l, _, _)| *l)
}

/// The line anchors of an intent entry with their steps: (step id or None, detail).
pub fn entry_anchors(details: &[crate::lens::Detail], outline: &[crate::lens::Node]) -> Vec<(Option<String>, crate::lens::Detail)> {
    let mut out: Vec<(Option<String>, crate::lens::Detail)> = details.iter().map(|d| (None, d.clone())).collect();
    fn go(ns: &[crate::lens::Node], out: &mut Vec<(Option<String>, crate::lens::Detail)>) {
        for n in ns {
            if let crate::lens::NodeKind::Step(id) = &n.kind {
                for d in &n.details {
                    out.push((Some(id.clone()), d.clone()));
                }
            }
            go(&n.children, out);
        }
    }
    go(outline, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Variable docs
// ---------------------------------------------------------------------------

/// One statement of a def as the variable docs see it: the steps around it
/// (outermost first), its first line, every name it reads or writes, what it
/// writes, and whether it only initializes (`x = 0`, `ok = false`,
/// `w = zeros(n)`), which is not where a reader meets a variable. For an
/// `if` or a loop, only its header: the condition or the bound, and a `for`
/// loop's variable.
#[derive(Clone, Debug)]
pub struct DocStmt {
    pub steps: Vec<String>,
    pub line: usize,
    pub touches: Vec<String>,
    pub writes: Vec<String>,
    pub init: bool,
}

fn is_init(value: &Expr) -> bool {
    match value {
        Expr::Num { .. } | Expr::Bool { .. } => true,
        Expr::Neg { operand, .. } => matches!(**operand, Expr::Num { .. }),
        Expr::Call { callee, .. } => matches!(callee.as_str(), "zeros" | "ones"),
        _ => false,
    }
}

/// Every statement of a def in source order, nested ones included.
pub fn doc_statements(body: &[Stmt], cx: &Cx, lines: &Lines) -> Vec<DocStmt> {
    fn go(body: &[Stmt], cx: &Cx, lines: &Lines, steps: &mut Vec<String>, out: &mut Vec<DocStmt>) {
        for s in body {
            let mut e = Eff::default();
            match s {
                Stmt::StepEnter { id, .. } => {
                    steps.push(id.clone());
                    continue;
                }
                Stmt::StepExit { .. } => {
                    steps.pop();
                    continue;
                }
                Stmt::Test { .. } | Stmt::Lens { .. } => continue,
                Stmt::If { cond, then_body, else_body, span } => {
                    expr_eff(cond, cx, &mut e);
                    let mut touches = e.reads.clone();
                    touches.retain(|n| !n.starts_with("viv_"));
                    out.push(DocStmt { steps: steps.clone(), line: lines.line(span.start), touches, writes: Vec::new(), init: false });
                    go(then_body, cx, lines, steps, out);
                    go(else_body, cx, lines, steps, out);
                    continue;
                }
                Stmt::While { body, cond, span } | Stmt::For { body, bound: cond, span, .. } => {
                    expr_eff(cond, cx, &mut e);
                    if let Stmt::For { var, .. } = s {
                        add(&mut e.writes, var);
                    }
                    let mut touches = e.reads.clone();
                    for w in &e.writes {
                        add(&mut touches, w);
                    }
                    out.push(DocStmt { steps: steps.clone(), line: lines.line(span.start), touches, writes: e.writes, init: false });
                    go(body, cx, lines, steps, out);
                    continue;
                }
                other => stmt_eff(other, cx, &mut e),
            }
            let init = matches!(s, Stmt::Assign { value, .. } if is_init(value));
            let mut touches = e.reads.clone();
            for w in &e.writes {
                add(&mut touches, w);
            }
            out.push(DocStmt { steps: steps.clone(), line: lines.line(s.span().start), touches, writes: e.writes, init });
        }
    }
    let mut out = Vec::new();
    go(body, cx, lines, &mut Vec::new(), &mut out);
    out
}

/// The names a def has: its parameters, then every variable a statement
/// assigns, in the order first met. What `documents every variable` asks a
/// glossary line for.
pub fn def_variables(params: &[String], stmts: &[DocStmt]) -> Vec<String> {
    let mut out: Vec<String> = params.to_vec();
    for s in stmts {
        for w in &s.writes {
            if !w.starts_with("viv_") {
                add(&mut out, w);
            }
        }
    }
    out
}

/// The line where a glossary line of a def starts to hold (lens.rs
/// `split_doc_key`). A variable's own doc: the def's line for a parameter,
/// else the first statement that uses it other than a bare initialization
/// (the first at all when it has later docs, or when nothing else uses it).
/// `name@step`: the first statement in that step that uses it.
/// `name@step#k`, `name@#k`: the k-th statement in the step, or the def,
/// that assigns it. Err says why it names no statement.
pub fn doc_line(params: &[String], def_line: usize, stmts: &[DocStmt], key: &str, has_later: bool) -> Result<usize, String> {
    let Some((name, step, k)) = crate::lens::split_doc_key(key) else { return Err(format!("`{key}` is not a glossary name")) };
    let in_step = |s: &DocStmt| step.as_ref().map_or(true, |st| s.steps.iter().any(|x| x == st));
    if let Some(k) = k {
        return stmts
            .iter()
            .filter(|s| in_step(s) && s.writes.iter().any(|w| w == &name))
            .nth(k - 1)
            .map(|s| s.line)
            .ok_or_else(|| format!("nothing in {} assigns `{name}` {k} {}", step.as_ref().map(|s| format!("step `{s}`")).unwrap_or_else(|| "the def".into()), if k == 1 { "time" } else { "times" }));
    }
    if let Some(st) = &step {
        return stmts.iter().find(|s| in_step(s) && s.touches.iter().any(|t| t == &name)).map(|s| s.line).ok_or_else(|| format!("step `{st}` does not use `{name}`"));
    }
    if params.iter().any(|p| p == &name) {
        return Ok(def_line);
    }
    let uses = |s: &&DocStmt| s.touches.iter().any(|t| t == &name);
    let first = stmts.iter().find(uses).map(|s| s.line);
    if has_later {
        return first.ok_or_else(|| format!("nothing uses `{name}`"));
    }
    stmts.iter().find(|s| uses(s) && !s.init).map(|s| s.line).or(first).ok_or_else(|| format!("nothing uses `{name}`"))
}

/// The body, parameter names and first byte of a def of a parsed program,
/// or of its top level for `<main>` (byte 0).
pub fn body_of(prog: &Program, name: &str) -> Option<(Vec<Stmt>, Vec<String>, usize)> {
    if name == MAIN_DEF {
        let top: Vec<Stmt> = prog.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
        return Some((top, Vec::new(), 0));
    }
    prog.items.iter().find_map(|i| match i {
        Item::Def(d) if d.name == name => Some((d.body.clone(), d.params.iter().map(|p| p.name.clone()).collect(), d.span.start as usize)),
        _ => None,
    })
}

/// Each glossary line of a def with the line it starts to hold (see
/// `doc_line`): (key, the variable, Ok(line) or Err(why not)).
pub fn doc_sites(body: &[Stmt], params: &[String], def_line: usize, cx: &Cx, lines: &Lines, keys: &[String]) -> Vec<(String, String, Result<usize, String>)> {
    let stmts = doc_statements(body, cx, lines);
    keys.iter()
        .map(|k| {
            let name = crate::lens::split_doc_key(k).map(|(n, _, _)| n).unwrap_or_default();
            let has_later = !k.contains('@') && keys.iter().any(|o| o.contains('@') && o.split('@').next() == Some(name.as_str()));
            (k.clone(), name, doc_line(params, def_line, &stmts, k, has_later))
        })
        .collect()
}
