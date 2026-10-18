// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The debug lens, milestone L5:
//! what to watch, what must hold and what to trace, compiled into a debug
//! build of both engines.
//!
//! The lens is written as `debug { ... }` blocks in the source or in a
//! `<module>.debug` file beside it (`file_to_source` turns that file into the
//! same blocks, line for line). The parser keeps the blocks apart from the
//! code (`Program::debug`). `instrument` then adds `Stmt::Lens` statements to
//! the def bodies they are about, and both engines run those statements the
//! same way: they record events into the story and never print to stdout.
//!
//! Instrumenting happens after parsing and before anything else, and only for
//! a debug build: the interpreter skips it for `--release`, and the C backend
//! guards every lens statement with `VIV_STORY`, which `-DVIV_RELEASE` turns
//! off. The body hashes are computed from the program without it
//! (`crate::build_ir_plain`), so a def's hash does not depend on its lens.
//!
//! What the story records, and where:
//! - a call and its return, for every def that has steps, a debug block, or
//!   is named by a `trace` line;
//! - each iteration of a def's iteration loops (the outermost loops that hold
//!   a step; in a def without steps, its outermost loops), with the
//!   per-iteration watches and holds at the end of the iteration;
//! - the watches and holds `at <step>` at the end of that step.
//!
//! A program with no debug block is returned unchanged, so it compiles and
//! runs exactly as it did before the debug lens existed.

use std::collections::{HashMap, HashSet};

use knotkit::diag::Diagnostic;
use knotkit::span::{SourceId, Span};

use crate::ast::*;
use crate::parser::MAIN_DEF;

/// The `at` a per-iteration watch or hold is recorded under.
pub const PER_ITERATION: &str = "iteration";


/// A string as a JSON string literal, quotes included. The engines copy it
/// into the story as it is, so both write the same bytes.
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Every statement in `body`, nested blocks included, in source order.
fn walk(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in body {
        f(s);
        match s {
            Stmt::If { then_body, else_body, .. } => {
                walk(then_body, f);
                walk(else_body, f);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => walk(body, f),
            _ => {}
        }
    }
}

fn has_step(body: &[Stmt]) -> bool {
    let mut any = false;
    walk(body, &mut |s| any |= matches!(s, Stmt::StepEnter { .. }));
    any
}

fn step_ids(body: &[Stmt]) -> HashSet<String> {
    let mut ids = HashSet::new();
    walk(body, &mut |s| {
        if let Stmt::StepEnter { id, .. } = s {
            ids.insert(id.clone());
        }
    });
    ids
}

/// What the lens asks of one def, gathered from all of its blocks.
#[derive(Clone, Default)]
struct Plan {
    /// names watched per iteration, in order, each once
    iter_watch: Vec<(String, Span)>,
    iter_holds: Vec<(HoldCheck, String)>,
    /// by step id: names watched and holds checked at the end of the step
    at_watch: HashMap<String, Vec<(String, Span)>>,
    at_holds: HashMap<String, Vec<(HoldCheck, String)>>,
    /// the values a watch computes (`name = expr`), per iteration and by
    /// step: each is bound to its name just before the watches and holds of
    /// its place, which may read it, as may any later one
    iter_binds: Vec<(String, Expr)>,
    at_binds: HashMap<String, Vec<(String, Expr)>>,
    /// in a region `crate::grad` inlined: each source name's name in the
    /// region (the holds are renamed already; a watch reads through this)
    rename: HashMap<String, String>,
}

impl Plan {
    /// The plan for a region of the def whose locals the region renamed by
    /// `rename` (source name to region name).
    fn renamed(&self, rename: &HashMap<String, String>) -> Plan {
        let hold = |(c, t): &(HoldCheck, String)| {
            let c = match c {
                HoldCheck::Cond(e) => HoldCheck::Cond(crate::grad::rename_expr(e, rename)),
                HoldCheck::Near { diff, tol } => HoldCheck::Near { diff: crate::grad::rename_expr(diff, rename), tol: crate::grad::rename_expr(tol, rename) },
            };
            (c, t.clone())
        };
        let bind = |(n, e): &(String, Expr)| (n.clone(), crate::grad::rename_expr(e, rename));
        Plan {
            iter_watch: self.iter_watch.clone(),
            iter_holds: self.iter_holds.iter().map(hold).collect(),
            at_watch: self.at_watch.clone(),
            at_holds: self.at_holds.iter().map(|(k, v)| (k.clone(), v.iter().map(hold).collect())).collect(),
            iter_binds: self.iter_binds.iter().map(bind).collect(),
            at_binds: self.at_binds.iter().map(|(k, v)| (k.clone(), v.iter().map(bind).collect())).collect(),
            rename: rename.clone(),
        }
    }

    /// A watch of `names`, read through the region's renaming.
    fn watch(&self, def: &str, at: String, names: &[(String, Span)]) -> Lens {
        let vars = if self.rename.is_empty() { Vec::new() } else { names.iter().map(|(n, _)| self.rename.get(n).unwrap_or(n).clone()).collect() };
        Lens::Watch { def: def.to_string(), at, names: names.to_vec(), vars }
    }
}

fn add_names(to: &mut Vec<(String, Span)>, names: &[(String, Span)]) {
    for (n, sp) in names {
        if !to.iter().any(|(m, _)| m == n) {
            to.push((n.clone(), *sp));
        }
    }
}

/// A hold, or a value a watch computes, must not change what the program
/// does: it may read variables and call pure builtins, and nothing else.
/// `what` is `hold` or `watch`, for the message.
fn check_hold_expr(e: &Expr, defs: &HashSet<String>) -> Result<(), Diagnostic> {
    check_lens_expr(e, defs, "hold")
}

fn check_lens_expr(e: &Expr, defs: &HashSet<String>, what: &str) -> Result<(), Diagnostic> {
    let mut bad: Option<(String, Span)> = None;
    fn go(e: &Expr, defs: &HashSet<String>, bad: &mut Option<(String, Span)>) {
        if bad.is_some() {
            return;
        }
        match e {
            Expr::Call { callee, args, span } => {
                // a hold may call only builtins that compute a value and change
                // nothing: one that printed, drew, drew a random number or wrote
                // into an output array would make the debug build do something
                // the release build does not (crate::builtins::is_pure)
                if defs.contains(callee) || !crate::builtins::is_pure(callee) {
                    *bad = Some((callee.clone(), *span));
                    return;
                }
                for a in args {
                    go(a, defs, bad);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                go(lhs, defs, bad);
                go(rhs, defs, bad);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => go(operand, defs, bad),
            Expr::Index { base, indices, .. } => {
                go(base, defs, bad);
                for i in indices {
                    go(i, defs, bad);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    go(x, defs, bad);
                }
            }
            Expr::Field { base, .. } => go(base, defs, bad),
            Expr::Lambda { span, .. } => *bad = Some(("fn".to_string(), *span)),
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    go(v, defs, bad);
                }
            }
            _ => {}
        }
    }
    go(e, defs, &mut bad);
    match bad {
        None => Ok(()),
        Some((name, span)) => Err(Diagnostic::error(format!("a {what} cannot call `{name}`"))
            .with_primary(span, "a debug lens reads the program, it does not run parts of it")
            .with_note(format!("a {what} may read variables and call these builtins: {}", crate::builtins::PURE.join(", ")))),
    }
}

/// Add the debug lens to the program's def bodies. See the module comment.
pub fn instrument(mut program: Program) -> Result<Program, Vec<Diagnostic>> {
    if program.debug.is_empty() {
        return Ok(program);
    }
    // The blocks stay on the program: the lowering reads them to know the
    // story is on (it prints `__story__` at the end of main).
    let blocks = program.debug.clone();
    let defs: HashMap<String, usize> = program
        .items
        .iter()
        .enumerate()
        .filter_map(|(i, it)| if let Item::Def(d) = it { Some((d.name.clone(), i)) } else { None })
        .collect();
    let def_names: HashSet<String> = defs.keys().cloned().collect();
    // the program's globals: what its top level assigns
    let globals: HashSet<String> = program
        .items
        .iter()
        .filter_map(|it| if let Item::Stmt(Stmt::Assign { name, .. }) = it { Some(name.clone()) } else { None })
        .collect();
    let mut errs: Vec<Diagnostic> = Vec::new();
    let mut plans: HashMap<String, Plan> = HashMap::new();
    let mut traced: HashSet<String> = HashSet::new();
    let unknown = |name: &str, span: Span, what: &str| {
        Diagnostic::error(format!("the debug lens names `{name}`, which is not a def of this program")).with_primary(span, format!("no def by this name to {what}"))
    };
    for b in &blocks {
        if let Some(d) = &b.def {
            if !defs.contains_key(d) {
                errs.push(unknown(d, b.def_span, "attach the block to"));
                continue;
            }
            plans.entry(d.clone()).or_default();
        }
        for item in &b.items {
            match item {
                DebugItem::Trace { names, .. } => {
                    for (n, sp) in names {
                        if defs.contains_key(n) {
                            traced.insert(n.clone());
                        } else {
                            errs.push(unknown(n, *sp, "trace"));
                        }
                    }
                }
                DebugItem::Watch { pos, names, exprs, span } => {
                    let Some(d) = &b.def else {
                        errs.push(Diagnostic::error("`debug * { }` can only trace").with_primary(*span, "a watch belongs to a def: write it in `debug <def> { }`"));
                        continue;
                    };
                    for (n, sp) in names {
                        if defs.contains_key(n) {
                            errs.push(Diagnostic::error(format!("`{n}` is a def, not a value to watch")).with_primary(*sp, "a def"));
                        }
                    }
                    // a computed value gets a name of its own: one the def
                    // and the program do not use, so binding it changes
                    // nothing the code reads
                    let mut binds: Vec<(String, Expr)> = Vec::new();
                    if let Some(&i) = defs.get(d) {
                        let Item::Def(def) = &program.items[i] else { unreachable!() };
                        let taken = names_of(def);
                        for ((n, sp), e) in names.iter().zip(exprs) {
                            let Some(e) = e else { continue };
                            if taken.contains(n) || globals.contains(n) {
                                errs.push(
                                    Diagnostic::error(format!("`{n}` is a variable of the program; a computed watch needs a name of its own"))
                                        .with_primary(*sp, "already a variable")
                                        .with_note(format!("watch `{n}` itself, or give the expression another name: `watch ... {n}_w = ...`")),
                                );
                                continue;
                            }
                            if let Err(x) = check_lens_expr(e, &def_names, "watch") {
                                errs.push(x);
                                continue;
                            }
                            binds.push((n.clone(), e.clone()));
                        }
                    }
                    let plan = plans.get_mut(d).expect("added above");
                    match pos {
                        DebugPos::PerIteration => {
                            add_names(&mut plan.iter_watch, names);
                            plan.iter_binds.extend(binds);
                        }
                        DebugPos::At(id, _) => {
                            add_names(plan.at_watch.entry(id.clone()).or_default(), names);
                            plan.at_binds.entry(id.clone()).or_default().extend(binds);
                        }
                    }
                }
                DebugItem::Hold { pos, check, sentence, span } => {
                    let Some(d) = &b.def else {
                        errs.push(Diagnostic::error("`debug * { }` can only trace").with_primary(*span, "a hold belongs to a def: write it in `debug <def> { }`"));
                        continue;
                    };
                    let exprs: Vec<&Expr> = match check {
                        HoldCheck::Cond(e) => vec![e],
                        HoldCheck::Near { diff, tol } => vec![diff, tol],
                    };
                    for e in exprs {
                        if let Err(x) = check_hold_expr(e, &def_names) {
                            errs.push(x);
                        }
                    }
                    let plan = plans.get_mut(d).expect("added above");
                    let h = (check.clone(), json_str(sentence));
                    match pos {
                        DebugPos::PerIteration => plan.iter_holds.push(h),
                        DebugPos::At(id, _) => plan.at_holds.entry(id.clone()).or_default().push(h),
                    }
                }
            }
        }
        // every `at <step>` must name a step of the def
        if let Some(d) = &b.def {
            if let Some(&i) = defs.get(d) {
                let Item::Def(def) = &program.items[i] else { unreachable!() };
                let ids = step_ids(&def.body);
                for item in &b.items {
                    let pos = match item {
                        DebugItem::Watch { pos, .. } | DebugItem::Hold { pos, .. } => pos,
                        DebugItem::Trace { .. } => continue,
                    };
                    if let DebugPos::At(id, sp) = pos {
                        if !ids.contains(id) {
                            let mut have: Vec<&String> = ids.iter().collect();
                            have.sort();
                            errs.push(
                                Diagnostic::error(format!("`{d}` has no step `{id}`"))
                                    .with_primary(*sp, "no such step")
                                    .with_note(if have.is_empty() { format!("`{d}` has no steps") } else { format!("its steps: {}", have.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")) }),
                            );
                        }
                    }
                }
            }
        }
    }
    if !errs.is_empty() {
        return Err(errs);
    }

    // Instrument every def the story records, and the top level.
    // the defs' effects, for line-level recording only (see `with_lines`)
    let lines_on = LINES.with(|l| l.borrow().is_some());
    let defs_cx = if lines_on { crate::graph::Cx::of(&program) } else { crate::graph::Cx { defs: HashSet::new(), mutates: HashMap::new() } };
    let mut items = std::mem::take(&mut program.items);
    // the defs whose lines are recorded: those with steps, and those the
    // explorer names (a def whose intent anchors lines but has no steps);
    // each gets a slot in a call counter, so only its first call (and the
    // calls asked for) is recorded line by line
    let extra: HashSet<String> = LINES.with(|l| l.borrow().as_ref().map(|o| o.defs.clone()).unwrap_or_default());
    let mut line_slot: HashMap<String, usize> = HashMap::new();
    if lines_on {
        let parallel = parallel_reached(&items);
        for it in &items {
            if let Item::Def(d) = it {
                if has_step(&d.body) || (extra.contains(&d.name) && !parallel.contains(&d.name)) {
                    let k = line_slot.len();
                    line_slot.insert(d.name.clone(), k);
                }
            }
        }
    }
    // the regions of the defs `crate::grad` inlined into gradient defs are
    // instrumented as calls of those defs, with the same plans
    let region_defs: HashMap<String, (Vec<String>, bool)> = items
        .iter()
        .filter_map(|it| if let Item::Def(d) = it { Some((d.name.clone(), (d.params.iter().map(|p| p.name.clone()).collect(), has_step(&d.body)))) } else { None })
        .collect();
    let region_plans: HashMap<String, Plan> = plans.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    for item in items.iter_mut() {
        let Item::Def(d) = item else { continue };
        if crate::grad::grad_def_of(&d.name).is_some() {
            let rc = RegionCx { plans: &region_plans, defs: &region_defs, traced: &traced, line_slot: &line_slot, cx: &defs_cx };
            let (mut body, recorded) = rc.regions(std::mem::take(&mut d.body)).map_err(|e| vec![e])?;
            if recorded {
                body.insert(0, Stmt::Lens { lens: Box::new(Lens::Call { def: d.name.clone() }), span: d.span });
            }
            d.body = body;
            continue;
        }
        let plan = plans.remove(&d.name);
        let stepped = has_step(&d.body);
        let slot = line_slot.get(&d.name).copied();
        if plan.is_none() && !stepped && !traced.contains(&d.name) {
            if let Some(k) = slot {
                // lines only: no call event, so the story is what it was
                let body = std::mem::take(&mut d.body);
                d.body = lines_def(&d.name, body, d.params.iter().map(|p| p.name.clone()).collect(), &defs_cx, k);
            }
            continue;
        }
        let plan = plan.unwrap_or_default();
        let span = d.span;
        let mut body = instrument_body(&d.name, std::mem::take(&mut d.body), &plan, span).map_err(|e| vec![e])?;
        if let Some(k) = slot {
            body = lines_def(&d.name, body, d.params.iter().map(|p| p.name.clone()).collect(), &defs_cx, k);
        }
        body.insert(0, Stmt::Lens { lens: Box::new(Lens::Call { def: d.name.clone() }), span });
        d.body = body;
    }
    if !line_slot.is_empty() {
        let sp = program_span(&items);
        let zeros = Expr::Call { callee: "zeros".into(), args: vec![Expr::Num { value: line_slot.len() as f64, span: sp }], span: sp };
        items.insert(0, Item::Stmt(Stmt::Assign { name: LINE_CALLS.into(), name_span: sp, value: zeros, span: sp }));
    }
    // The top level: its iteration loops, when it has steps. Top-level
    // statements are separate items, so a loop's counter is an item of its
    // own, placed just before the loop; every item keeps its place.
    let top: Vec<Stmt> = items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    if has_step(&top) {
        let mut k = 0usize;
        let mut out: Vec<Item> = Vec::with_capacity(items.len() + 2);
        for it in items {
            match it {
                Item::Stmt(s) => out.extend(rewrite(MAIN_DEF, vec![s], &Plan::default(), false, false, &mut k).into_iter().map(Item::Stmt)),
                other => out.push(other),
            }
        }
        items = out;
    }
    program.items = lines_main(items, &defs_cx);
    // last, the values computed watches bind: assignments a release build
    // skips (`release_mode()` is true there in every engine)
    for it in program.items.iter_mut() {
        match it {
            Item::Def(d) => unbind(&mut d.body),
            Item::Stmt(s) => unbind(std::slice::from_mut(s)),
            _ => {}
        }
    }
    crate::check_parallel_records(&program)?;
    Ok(program)
}

/// Every name a def binds: its parameters, what it assigns and its loop
/// variables.
fn names_of(d: &Def) -> HashSet<String> {
    let mut out: HashSet<String> = d.params.iter().map(|p| p.name.clone()).collect();
    walk(&d.body, &mut |s| match s {
        Stmt::Assign { name, .. } => {
            out.insert(name.clone());
        }
        Stmt::For { var, .. } => {
            out.insert(var.clone());
        }
        _ => {}
    });
    out
}

/// Each `Lens::Bind` in `body` (nested blocks and iteration ends included)
/// becomes `if not release_mode() { name = value }`.
fn unbind(body: &mut [Stmt]) {
    for s in body.iter_mut() {
        match s {
            Stmt::Lens { lens, span } => {
                if let Lens::Bind { name, value } = &**lens {
                    let sp = *span;
                    let rel = Expr::Call { callee: "release_mode".into(), args: vec![], span: sp };
                    *s = Stmt::If {
                        cond: Expr::Not { operand: Box::new(rel), span: sp },
                        then_body: vec![Stmt::Assign { name: name.clone(), name_span: sp, value: value.clone(), span: sp }],
                        else_body: vec![],
                        span: sp,
                    };
                } else if let Lens::Iter { end, .. } = &mut **lens {
                    unbind(end);
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                unbind(then_body);
                unbind(else_body);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => unbind(body),
            _ => {}
        }
    }
}

/// What instrumenting the regions of a gradient def needs: each def's plan,
/// its parameters and whether it has steps, the traced defs, and the line
/// slots.
struct RegionCx<'a> {
    plans: &'a HashMap<String, Plan>,
    defs: &'a HashMap<String, (Vec<String>, bool)>,
    traced: &'a HashSet<String>,
    line_slot: &'a HashMap<String, usize>,
    cx: &'a crate::graph::Cx,
}

/// A stand-in for a region already instrumented, while the region around it
/// is: a lens statement, which every walk here passes over.
fn stand_in(k: usize, span: Span) -> Stmt {
    Stmt::Lens { lens: Box::new(Lens::Inline { def: String::new(), enter: false, names: vec![(String::new(), k.to_string())] }), span }
}

fn stand_in_of(s: &Stmt) -> Option<usize> {
    match s {
        Stmt::Lens { lens, .. } => match &**lens {
            Lens::Inline { def, enter: false, names } if def.is_empty() => names.first().and_then(|(_, k)| k.parse().ok()),
            _ => None,
        },
        _ => None,
    }
}

/// `f` over each statement list of a block, nested ones included.
fn map_lists(stmts: Vec<Stmt>, f: &mut dyn FnMut(Vec<Stmt>) -> Result<Vec<Stmt>, Diagnostic>) -> Result<Vec<Stmt>, Diagnostic> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in f(stmts)? {
        out.push(match s {
            Stmt::If { cond, then_body, else_body, span } => Stmt::If { cond, then_body: map_lists(then_body, f)?, else_body: map_lists(else_body, f)?, span },
            Stmt::While { cond, body, span } => Stmt::While { cond, body: map_lists(body, f)?, span },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For { var, var_span, bound, body: map_lists(body, f)?, parallel, span },
            other => other,
        });
    }
    Ok(out)
}

impl RegionCx<'_> {
    /// A gradient def's body with every region instrumented as a call of its
    /// def; true when the story records any of them.
    fn regions(&self, body: Vec<Stmt>) -> Result<(Vec<Stmt>, bool), Diagnostic> {
        let mut stash: Vec<Vec<Stmt>> = Vec::new();
        let mut any = false;
        let body = self.stash_regions(body, &mut stash, &mut any)?;
        Ok((unstash(body, &mut stash), any))
    }

    /// `body` with each outermost region instrumented and replaced by a
    /// stand-in that indexes `stash`.
    fn stash_regions(&self, body: Vec<Stmt>, stash: &mut Vec<Vec<Stmt>>, any: &mut bool) -> Result<Vec<Stmt>, Diagnostic> {
        map_lists(body, &mut |list: Vec<Stmt>| {
            let mut out = Vec::with_capacity(list.len());
            let mut i = 0;
            while i < list.len() {
                let Some((def, names)) = crate::grad::region_enter(&list[i]) else {
                    out.push(list[i].clone());
                    i += 1;
                    continue;
                };
                let j = crate::grad::region_close(&list, i).ok_or_else(|| Diagnostic::error(format!("internal: the region of `{def}` has no end")).with_primary(list[i].span(), "here"))?;
                let (region, recorded) = self.region(def, names, list[i + 1..j].to_vec(), list[i].span())?;
                *any |= recorded;
                let mut whole = vec![list[i].clone()];
                whole.extend(region);
                whole.push(list[j].clone());
                out.push(stand_in(stash.len(), list[i].span()));
                stash.push(whole);
                i = j + 1;
            }
            Ok(out)
        })
    }

    /// One region's statements (between its markers), instrumented as a call
    /// of `def`: its own regions first, then its steps, loops and lines.
    fn region(&self, def: &str, names: &[(String, String)], inner: Vec<Stmt>, span: Span) -> Result<(Vec<Stmt>, bool), Diagnostic> {
        let mut stash: Vec<Vec<Stmt>> = Vec::new();
        let mut any = false;
        let mut body = self.stash_regions(inner, &mut stash, &mut any)?;
        let rename: HashMap<String, String> = names.iter().cloned().collect();
        let (params, stepped) = self.defs.get(def).cloned().unwrap_or_default();
        let plan = self.plans.get(def);
        let recorded = plan.is_some() || stepped || self.traced.contains(def);
        if recorded {
            let plan = plan.map(|p| p.renamed(&rename)).unwrap_or_else(|| Plan { rename: rename.clone(), ..Plan::default() });
            body = instrument_body(def, body, &plan, span)?;
        }
        if let Some(&slot) = self.line_slot.get(def) {
            let label: HashMap<String, String> = names.iter().map(|(a, b)| (b.clone(), a.clone())).collect();
            let params: Vec<String> = params.iter().map(|p| rename.get(p).unwrap_or(p).clone()).collect();
            body = lines_body(def, body, params, self.cx, slot, label, &format!("{LINE_CALL}_{def}"));
        }
        if recorded {
            body.insert(0, Stmt::Lens { lens: Box::new(Lens::Call { def: def.to_string() }), span });
            body.push(Stmt::Lens { lens: Box::new(Lens::Ret { def: def.to_string() }), span });
        }
        Ok((unstash(body, &mut stash), recorded || any))
    }
}

/// `body` with each stand-in replaced by the statements it stands for.
fn unstash(body: Vec<Stmt>, stash: &mut Vec<Vec<Stmt>>) -> Vec<Stmt> {
    map_lists(body, &mut |list: Vec<Stmt>| {
        let mut out = Vec::with_capacity(list.len());
        for s in list {
            match stand_in_of(&s) {
                Some(k) => out.extend(std::mem::take(&mut stash[k])),
                None => out.push(s),
            }
        }
        Ok(out)
    })
    .expect("unstashing cannot fail")
}

/// The global array that counts the calls of each def recorded line by line.
const LINE_CALLS: &str = "viv_lncall";
/// A def's own copy of its call number.
const LINE_CALL: &str = "viv_lnc";

/// A span for statements the lens adds at the top of the program.
fn program_span(items: &[Item]) -> Span {
    items
        .iter()
        .find_map(|i| match i {
            Item::Def(d) => Some(d.span),
            Item::Stmt(s) => Some(s.span()),
            _ => None,
        })
        .map(|s| Span { end: s.start, ..s })
        .unwrap_or_else(|| Span::synthetic(SourceId(0)))
}

/// The counter variable of the `k`-th iteration loop of a def.
fn counter_name(def: &str, k: usize) -> String {
    let d = if def == MAIN_DEF { "main" } else { def };
    format!("viv_it_{d}_{k}")
}

/// A def's body with its iteration loops and step-end watches added.
fn instrument_body(def: &str, body: Vec<Stmt>, plan: &Plan, span: Span) -> Result<Vec<Stmt>, Diagnostic> {
    // Which outermost loops are iteration loops: those holding a step, or,
    // in a def without steps, all of them. A `parallel for` never is.
    let wants_iter = !plan.iter_watch.is_empty() || !plan.iter_holds.is_empty();
    let stepped_loops = {
        let mut n = 0;
        count_outer_loops(&body, &mut |s| {
            if loop_body(s).is_some_and(has_step) && !is_parallel(s) {
                n += 1;
            }
        });
        n
    };
    let all_loops = stepped_loops == 0;
    if wants_iter {
        let mut n = 0;
        count_outer_loops(&body, &mut |s| {
            if !is_parallel(s) && (all_loops || loop_body(s).is_some_and(has_step)) {
                n += 1;
            }
        });
        if n == 0 {
            return Err(Diagnostic::error(format!("`{def}` has no loop for a per-iteration watch or hold"))
                .with_primary(span, "no `while` or `for` in this def")
                .with_note("watch or hold at the end of a step instead: `watch at <step> ...`"));
        }
    }
    let mut k = 0usize;
    Ok(rewrite(def, body, plan, all_loops && wants_iter, false, &mut k))
}

/// The defs a `parallel for` reaches, directly or through the defs they call.
/// Their iterations run on several threads, so recording their lines would
/// give events in no fixed order: they are never recorded line by line.
fn parallel_reached(items: &[Item]) -> HashSet<String> {
    let mut callees: HashMap<String, Vec<String>> = HashMap::new();
    let mut todo: Vec<String> = Vec::new();
    let seed = |body: &[Stmt], todo: &mut Vec<String>| {
        walk(body, &mut |s| {
            if let Stmt::For { parallel: true, body, .. } = s {
                crate::defun::walk_block(body, &mut |e: &Expr| {
                    if let Expr::Call { callee, .. } = e {
                        todo.push(callee.clone());
                    }
                });
            }
        });
    };
    for it in items {
        match it {
            Item::Def(d) => {
                let mut cs = Vec::new();
                crate::defun::walk_block(&d.body, &mut |e: &Expr| {
                    if let Expr::Call { callee, .. } = e {
                        cs.push(callee.clone());
                    }
                });
                callees.insert(d.name.clone(), cs);
                seed(&d.body, &mut todo);
            }
            Item::Stmt(s) => seed(std::slice::from_ref(s), &mut todo),
            _ => {}
        }
    }
    let mut reached: HashSet<String> = HashSet::new();
    while let Some(n) = todo.pop() {
        if reached.insert(n.clone()) {
            if let Some(cs) = callees.get(&n) {
                todo.extend(cs.iter().cloned());
            }
        }
    }
    reached
}

fn loop_body(s: &Stmt) -> Option<&[Stmt]> {
    match s {
        Stmt::While { body, .. } | Stmt::For { body, .. } => Some(body),
        _ => None,
    }
}

fn is_parallel(s: &Stmt) -> bool {
    matches!(s, Stmt::For { parallel: true, .. })
}

/// Call `f` on each loop that is not inside another loop.
fn count_outer_loops(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in body {
        match s {
            Stmt::While { .. } | Stmt::For { .. } => f(s),
            Stmt::If { then_body, else_body, .. } => {
                count_outer_loops(then_body, f);
                count_outer_loops(else_body, f);
            }
            Stmt::Test { body, .. } => count_outer_loops(body, f),
            _ => {}
        }
    }
}

/// Rewrite a statement list: the watches and holds of a step go just before
/// its exit marker, and an iteration loop gets its counter and its `Iter`.
/// `all_loops`: every outermost loop is an iteration loop; `in_loop`: this
/// list is already inside a loop (so its loops are not outermost).
fn rewrite(def: &str, body: Vec<Stmt>, plan: &Plan, all_loops: bool, in_loop: bool, k: &mut usize) -> Vec<Stmt> {
    let lens = |l: Lens, span: Span| Stmt::Lens { lens: Box::new(l), span };
    let mut out = Vec::with_capacity(body.len());
    for s in body {
        match s {
            Stmt::StepExit { ref id, span, .. } => {
                for (name, value) in plan.at_binds.get(id).into_iter().flatten() {
                    out.push(lens(Lens::Bind { name: name.clone(), value: value.clone() }, span));
                }
                if let Some(names) = plan.at_watch.get(id) {
                    out.push(lens(plan.watch(def, id.clone(), names), span));
                }
                for (check, sentence) in plan.at_holds.get(id).into_iter().flatten() {
                    out.push(lens(Lens::Hold { def: def.to_string(), at: id.clone(), check: check.clone(), sentence: sentence.clone() }, span));
                }
                out.push(s);
            }
            Stmt::If { cond, then_body, else_body, span } => {
                let then_body = rewrite(def, then_body, plan, all_loops, in_loop, k);
                let else_body = rewrite(def, else_body, plan, all_loops, in_loop, k);
                out.push(Stmt::If { cond, then_body, else_body, span });
            }
            Stmt::Test { name, body, span } => {
                let body = rewrite(def, body, plan, all_loops, in_loop, k);
                out.push(Stmt::Test { name, body, span });
            }
            Stmt::While { .. } | Stmt::For { .. } => {
                let iterating = !in_loop && !is_parallel(&s) && (all_loops || loop_body(&s).is_some_and(has_step));
                let counter = if iterating {
                    *k += 1;
                    Some(counter_name(def, *k))
                } else {
                    None
                };
                let (inner, span) = match &s {
                    Stmt::While { body, span, .. } | Stmt::For { body, span, .. } => (body.clone(), *span),
                    _ => unreachable!(),
                };
                let mut inner = rewrite(def, inner, plan, all_loops, true, k);
                if let Some(counter) = &counter {
                    let mut end = Vec::new();
                    for (name, value) in &plan.iter_binds {
                        end.push(lens(Lens::Bind { name: name.clone(), value: value.clone() }, span));
                    }
                    if !plan.iter_watch.is_empty() {
                        end.push(lens(plan.watch(def, PER_ITERATION.to_string(), &plan.iter_watch), span));
                    }
                    for (check, sentence) in &plan.iter_holds {
                        end.push(lens(Lens::Hold { def: def.to_string(), at: PER_ITERATION.to_string(), check: check.clone(), sentence: sentence.clone() }, span));
                    }
                    end.push(lens(Lens::IterEnd { def: def.to_string() }, span));
                    inner.insert(0, lens(Lens::Iter { def: def.to_string(), counter: counter.clone(), end }, span));
                    out.push(Stmt::Assign { name: counter.clone(), name_span: span, value: Expr::Num { value: 0.0, span }, span });
                }
                out.push(match s {
                    Stmt::While { cond, span, .. } => Stmt::While { cond, body: inner, span },
                    Stmt::For { var, var_span, bound, parallel, span, .. } => Stmt::For { var, var_span, bound, body: inner, parallel, span },
                    _ => unreachable!(),
                });
            }
            other => out.push(other),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Line-level recording, for the explorer
// ---------------------------------------------------------------------------
//
// With `with_lines`, `instrument` also records the lines of every def that has
// steps, and of the top level: after each statement, a watch at `line:<n>`
// (n the statement's first line in the program) with the values that
// statement assigned; before a `return`, `break` or `continue`, the same
// watch with no values. Three rules keep a recording small:
//
// - A loop that holds no step is one statement. Its lines are never recorded,
//   only what it assigned, once, after it ends. Matrix fills and dot products
//   inside a step cost one event, however many times they go round.
// - Inside an iteration loop (the lens's own, with its counter), lines are
//   recorded in iterations 1..=`first` and in the iterations `iters` names
//   for that def, and nowhere else. The other iterations keep what the
//   story always had: step entries and exits, and the per-iteration watches.
//   Any other loop that holds a step gets a counter of its own
//   (`viv_ln_<def>_<k>`), and only its first pass is recorded line by line.
// - Which branch of an `if` ran is recorded in every iteration (a watch at
//   `br:<n>:1` or `br:<n>:0` with no values), since a branch that goes the
//   other way is one of the things the explorer looks for.
//
// A value is watched only where it is certain to exist (assigned on every path
// to that point, or a parameter) and only when a watch can record it (a number,
// a boolean or a real array, in every compiled copy of the def:
// `crate::graph::watchable`). The events are ordinary watches, so both engines
// record them the same way.

/// What a line-level recording records.
#[derive(Clone, Debug, Default)]
pub struct LineOpts {
    /// the byte offset where each line of the program starts
    pub starts: Vec<u32>,
    /// by def (`<main>` for the top level): the variables a watch can record
    pub watchable: HashMap<String, HashSet<String>>,
    /// iterations 1..=first of every iteration loop are recorded line by line
    pub first: i64,
    /// and these, by def
    pub iters: HashMap<String, Vec<i64>>,
    /// defs without steps to record line by line too (the explorer names the
    /// ones whose intent anchors lines)
    pub defs: HashSet<String>,
    /// a def's lines are recorded in its first call and in these, by def
    pub calls: HashMap<String, Vec<i64>>,
    /// defs whose other loops holding a step (lmder's trial steps inside
    /// its iteration loop) are recorded line by line in every pass, not only
    /// the first: the explorer asks for this when a run is small enough to
    /// record whole, so every pass and trial has its values
    pub every: HashSet<String>,
}

/// One statement a line-level recording knows about.
#[derive(Clone, Debug, PartialEq)]
pub struct LineRec {
    pub def: String,
    /// first and last line in the program
    pub line: usize,
    pub end: usize,
    /// `stmt`; `exit` (a return, break or continue, recorded before it runs);
    /// `loop` (a loop without steps: one event after it); `if`; `iterloop`
    /// (an iteration loop); `steploop` (another loop that holds a step)
    pub kind: &'static str,
    /// the innermost step around it
    pub step: Option<String>,
    /// the values its event records
    pub watch: Vec<String>,
    /// inside an iteration loop: recorded only in the iterations chosen
    pub gated: bool,
}

thread_local! {
    static LINES: std::cell::RefCell<Option<LineOpts>> = const { std::cell::RefCell::new(None) };
    static TABLE: std::cell::RefCell<Vec<LineRec>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with line-level recording on for every `instrument` it makes, and
/// return what it returned with the table of recorded statements.
pub fn with_lines<T>(opts: LineOpts, f: impl FnOnce() -> T) -> (T, Vec<LineRec>) {
    LINES.with(|l| *l.borrow_mut() = Some(opts));
    TABLE.with(|t| t.borrow_mut().clear());
    let out = f();
    LINES.with(|l| *l.borrow_mut() = None);
    let table = TABLE.with(|t| std::mem::take(&mut *t.borrow_mut()));
    (out, table)
}

/// Where a line's event may fire: in the chosen passes of the def's
/// iteration loop (`iter`, its counter), and only in the first pass of each
/// other loop holding a step that it sits in (`nested`, their counters).
#[derive(Clone, Debug, Default)]
struct Gate {
    iter: Option<String>,
    nested: Vec<String>,
    /// the def's call number (`viv_lnc`), in a def rather than the top level
    call: bool,
}

impl Gate {
    fn on(&self) -> bool {
        self.iter.is_some() || !self.nested.is_empty() || self.call
    }
}

struct LineWalk<'a> {
    def: String,
    opts: &'a LineOpts,
    cx: &'a crate::graph::Cx,
    steps: Vec<String>,
    table: Vec<LineRec>,
    /// loops holding a step, other than the iteration loops, numbered
    nested: usize,
    /// in a region `crate::grad` inlined: each renamed local's source name
    /// (empty in a def), which is what its events record it under
    label: HashMap<String, String>,
    /// the variable holding the def's call number
    call_var: String,
}

impl LineWalk<'_> {
    fn line(&self, offset: u32) -> usize {
        match self.opts.starts.binary_search(&offset) {
            Ok(k) => k + 1,
            Err(k) => k,
        }
    }
    fn lines(&self, span: Span) -> (usize, usize) {
        (self.line(span.start), self.line(span.end.saturating_sub(1).max(span.start)))
    }
    /// The name a variable is recorded under: its source name.
    fn label<'n>(&'n self, n: &'n str) -> &'n str {
        self.label.get(n).map(String::as_str).unwrap_or(n)
    }
    fn watchable(&self, names: &[String], defined: &HashSet<String>) -> Vec<String> {
        let w = self.opts.watchable.get(&self.def);
        names.iter().filter(|n| defined.contains(*n) && w.is_some_and(|w| w.contains(self.label(n))) && !self.label(n).starts_with("viv_")).cloned().collect()
    }
    /// A statement `crate::grad` added (a push onto its tape, a temporary, a
    /// pass count, the cell a `value_and_grad` value comes back in), which
    /// the source does not have: it runs, and records nothing.
    fn machinery(&self, s: &Stmt) -> bool {
        match s {
            Stmt::ExprStmt { expr: Expr::Call { callee, .. }, .. } => callee.starts_with("__ad_"),
            Stmt::Assign { name, .. } => name.starts_with("viv__ad") && !self.label.contains_key(name) && !name.ends_with("_ret"),
            _ => false,
        }
    }
    fn rec(&mut self, kind: &'static str, span: Span, watch: Vec<String>, gated: bool) {
        let (line, end) = self.lines(span);
        self.table.push(LineRec { def: self.def.clone(), line, end, kind, step: self.steps.last().cloned(), watch, gated });
    }
    /// The event: a watch at `at`, behind the gate when there is one.
    fn event(&self, at: String, names: &[String], span: Span, gate: &Gate) -> Stmt {
        let labels: Vec<(String, Span)> = names.iter().map(|n| (self.label(n).to_string(), span)).collect();
        let vars = if labels.iter().zip(names).any(|((l, _), n)| l != n) { names.to_vec() } else { Vec::new() };
        let w = Stmt::Lens { lens: Box::new(Lens::Watch { def: self.def.clone(), at, names: labels, vars }), span };
        if !gate.on() {
            return w;
        }
        let var = |name: &str| Expr::Var { name: name.to_string(), span };
        let num = |x: i64| Expr::Num { value: x as f64, span };
        let bin = |op: BinOp, l: Expr, r: Expr| Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span };
        let mut cond: Option<Expr> = None;
        if let Some(counter) = &gate.iter {
            let mut c = bin(BinOp::Le, var(counter), num(self.opts.first));
            for &i in self.opts.iters.get(&self.def).map(Vec::as_slice).unwrap_or(&[]) {
                c = bin(BinOp::Or, c, bin(BinOp::EqEq, var(counter), num(i)));
            }
            cond = Some(c);
        }
        let and = |c: Expr, cond: &mut Option<Expr>| {
            *cond = Some(match cond.take() {
                Some(prev) => bin(BinOp::And, prev, c),
                None => c,
            });
        };
        for n in &gate.nested {
            and(bin(BinOp::Le, var(n), num(1)), &mut cond);
        }
        if gate.call {
            let mut c = bin(BinOp::Le, var(&self.call_var), num(1));
            for &k in self.opts.calls.get(&self.def).map(Vec::as_slice).unwrap_or(&[]) {
                c = bin(BinOp::Or, c, bin(BinOp::EqEq, var(&self.call_var), num(k)));
            }
            and(c, &mut cond);
        }
        Stmt::If { cond: cond.expect("a gate that is on has a condition"), then_body: vec![w], else_body: Vec::new(), span }
    }
    fn walk(&mut self, body: Vec<Stmt>, gate: &Gate, defined: &mut HashSet<String>) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(body.len() * 2);
        for s in body {
            match s {
                Stmt::StepEnter { ref id, .. } => {
                    self.steps.push(id.clone());
                    out.push(s);
                }
                Stmt::StepExit { .. } => {
                    self.steps.pop();
                    out.push(s);
                }
                Stmt::Lens { .. } | Stmt::Test { .. } => out.push(s),
                Stmt::Assign { ref name, .. } if name.starts_with("viv_it_") => {
                    defined.insert(name.clone());
                    out.push(s);
                }
                ref m if self.machinery(m) => {
                    if let Stmt::Assign { name, .. } = m {
                        defined.insert(name.clone());
                    }
                    out.push(s);
                }
                // the value of a `return` in a region: the return's line, recorded before it
                Stmt::Assign { ref name, span, .. } if !self.label.is_empty() && name.starts_with("viv__ad") && name.ends_with("_ret") => {
                    self.rec("exit", span, Vec::new(), gate.on());
                    out.push(self.event(format!("line:{}", self.line(span.start)), &[], span, gate));
                    defined.insert(name.clone());
                    out.push(s);
                }
                Stmt::Return { span, .. } | Stmt::Break { span } | Stmt::Continue { span } => {
                    self.rec("exit", span, Vec::new(), gate.on());
                    out.push(self.event(format!("line:{}", self.line(span.start)), &[], span, gate));
                    out.push(s);
                }
                Stmt::If { cond, then_body, else_body, span } => {
                    self.rec("if", span, Vec::new(), gate.on());
                    let l = self.line(span.start);
                    // which branch ran: on every pass of the iteration loop,
                    // but only in the first pass of a nested loop
                    let bgate = Gate { iter: None, nested: gate.nested.clone(), call: gate.call };
                    let (mut dt, mut de) = (defined.clone(), defined.clone());
                    let mut t = vec![self.event(format!("br:{l}:1"), &[], span, &bgate)];
                    t.extend(self.walk(then_body, gate, &mut dt));
                    let mut e = vec![self.event(format!("br:{l}:0"), &[], span, &bgate)];
                    e.extend(self.walk(else_body, gate, &mut de));
                    defined.extend(dt.intersection(&de).cloned().collect::<Vec<_>>());
                    out.push(Stmt::If { cond, then_body: t, else_body: e, span });
                }
                Stmt::While { .. } | Stmt::For { .. } => {
                    let span = s.span();
                    let counter = match loop_body(&s).and_then(|b| b.first()) {
                        Some(Stmt::Lens { lens, .. }) => match &**lens {
                            Lens::Iter { counter, .. } => Some(counter.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    let stepped = loop_body(&s).is_some_and(has_step);
                    if counter.is_none() && !stepped {
                        // a loop without steps: one statement, one event after it
                        let mut e = crate::graph::Eff::default();
                        crate::graph::stmt_eff(&s, self.cx, &mut e);
                        let names = self.watchable(&e.writes, defined);
                        self.rec("loop", span, names.clone(), gate.on());
                        out.push(s);
                        out.push(self.event(format!("line:{}", self.line(span.start)), &names, span, gate));
                        continue;
                    }
                    self.rec(if counter.is_some() { "iterloop" } else { "steploop" }, span, Vec::new(), gate.on());
                    let mut inner = gate.clone();
                    let mut bump: Option<Stmt> = None;
                    match &counter {
                        Some(c) => inner.iter = Some(c.clone()),
                        None if self.opts.every.contains(&self.def) => {}
                        None => {
                            // count this loop's passes, so only its first is recorded line by line
                            self.nested += 1;
                            let d = if self.def == MAIN_DEF { "main" } else { self.def.as_str() };
                            let name = format!("viv_ln_{d}_{}", self.nested);
                            out.push(Stmt::Assign { name: name.clone(), name_span: span, value: Expr::Num { value: 0.0, span }, span });
                            let one = Expr::Binary { op: BinOp::Add, lhs: Box::new(Expr::Var { name: name.clone(), span }), rhs: Box::new(Expr::Num { value: 1.0, span }), span };
                            bump = Some(Stmt::Assign { name: name.clone(), name_span: span, value: one, span });
                            inner.nested.push(name);
                        }
                    }
                    let mut d = defined.clone();
                    let with_bump = |body: Vec<Stmt>| -> Vec<Stmt> {
                        match &bump {
                            Some(b) => std::iter::once(b.clone()).chain(body).collect(),
                            None => body,
                        }
                    };
                    match s {
                        Stmt::While { cond, body, span } => {
                            let body = self.walk(body, &inner, &mut d);
                            out.push(Stmt::While { cond, body: with_bump(body), span });
                        }
                        Stmt::For { var, var_span, bound, body, parallel, span } => {
                            d.insert(var.clone());
                            let body = self.walk(body, &inner, &mut d);
                            out.push(Stmt::For { var, var_span, bound, body: with_bump(body), parallel, span });
                        }
                        _ => unreachable!(),
                    }
                }
                other => {
                    let span = other.span();
                    let mut e = crate::graph::Eff::default();
                    crate::graph::stmt_eff(&other, self.cx, &mut e);
                    if let Stmt::Assign { name, .. } = &other {
                        defined.insert(name.clone());
                    }
                    let names = self.watchable(&e.writes, defined);
                    self.rec("stmt", span, names.clone(), gate.on());
                    out.push(other);
                    out.push(self.event(format!("line:{}", self.line(span.start)), &names, span, gate));
                }
            }
        }
        out
    }
}

/// A def's instrumented body with its lines recorded, when `with_lines` is on.
fn lines_def(def: &str, body: Vec<Stmt>, params: Vec<String>, cx: &crate::graph::Cx, slot: usize) -> Vec<Stmt> {
    lines_body(def, body, params, cx, slot, HashMap::new(), LINE_CALL)
}

/// `lines_def` for a body whose locals are renamed (`label`: each renamed
/// local's source name), keeping its call number in `call_var`.
fn lines_body(def: &str, body: Vec<Stmt>, params: Vec<String>, cx: &crate::graph::Cx, slot: usize, label: HashMap<String, String>, call_var: &str) -> Vec<Stmt> {
    LINES.with(|l| {
        let l = l.borrow();
        let Some(opts) = l.as_ref() else { return body };
        let mut w = LineWalk { def: def.to_string(), opts, cx, steps: Vec::new(), table: Vec::new(), nested: 0, label, call_var: call_var.to_string() };
        let mut defined: HashSet<String> = params.into_iter().collect();
        // count the call: viv_lncall[slot] += 1, and keep the count
        let sp = body.first().map(|s| Span { end: s.span().start, ..s.span() }).unwrap_or_else(|| Span::synthetic(SourceId(0)));
        let at = || Expr::Index { base: Box::new(Expr::Var { name: LINE_CALLS.into(), span: sp }), indices: vec![Expr::Num { value: slot as f64, span: sp }], span: sp };
        let bump = Stmt::SetIndex {
            base: Box::new(Expr::Var { name: LINE_CALLS.into(), span: sp }),
            indices: vec![Expr::Num { value: slot as f64, span: sp }],
            value: Expr::Binary { op: BinOp::Add, lhs: Box::new(at()), rhs: Box::new(Expr::Num { value: 1.0, span: sp }), span: sp },
            span: sp,
        };
        let keep = Stmt::Assign { name: call_var.into(), name_span: sp, value: at(), span: sp };
        let mut out = vec![bump, keep];
        out.extend(w.walk(body, &Gate { call: true, ..Gate::default() }, &mut defined));
        TABLE.with(|t| t.borrow_mut().extend(w.table));
        out
    })
}

/// The top level with its lines recorded, when `with_lines` is on. Each
/// top-level statement stays an item of its own.
fn lines_main(items: Vec<Item>, cx: &crate::graph::Cx) -> Vec<Item> {
    LINES.with(|l| {
        let l = l.borrow();
        let Some(opts) = l.as_ref() else { return items };
        let mut w = LineWalk { def: MAIN_DEF.to_string(), opts, cx, steps: Vec::new(), table: Vec::new(), nested: 0, label: HashMap::new(), call_var: LINE_CALL.to_string() };
        let mut defined: HashSet<String> = HashSet::new();
        let mut out = Vec::with_capacity(items.len() * 2);
        for it in items {
            match it {
                Item::Stmt(s) => out.extend(w.walk(vec![s], &Gate::default(), &mut defined).into_iter().map(Item::Stmt)),
                other => out.push(other),
            }
        }
        TABLE.with(|t| t.borrow_mut().extend(w.table));
        out
    })
}

// ---------------------------------------------------------------------------
// The `.debug` file
// ---------------------------------------------------------------------------

/// A `<module>.debug` file as `debug` blocks in Viv source, one line for one
/// line, so a parse error in it names the file's own line. The format:
///
/// ```text
/// # a comment
/// rhf_scf  @3fa9c21e
///   watch per iteration  E, dE, dnorm
///   hold at fock         F == transpose(F) within 1e-12  "the Fock matrix is symmetric"
///   trace                jk_select
/// *
///   trace                build_eri
/// ```
///
/// An entry starts at a line in column 0 naming the def (or `*` for the whole
/// program), optionally with its stamp `@<body hash>` and `by <who>`; its
/// items are the indented lines below it.
pub fn file_to_source(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut open = false;
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            out.push(String::new());
            continue;
        }
        if !line.starts_with(char::is_whitespace) {
            let name = t.split_whitespace().next().unwrap_or("");
            let close = if open { "} " } else { "" };
            out.push(format!("{close}debug {name} {{"));
            open = true;
        } else if open {
            out.push(line.to_string());
        } else {
            // an item before any entry: let the parser say where
            out.push(format!("debug ? {{ {} }}", t));
        }
    }
    // the last entry closes on its own last line, so the lines still match
    if open {
        if let Some(last) = out.last_mut() {
            last.push_str(" }");
        }
    }
    let mut s = out.join("\n");
    s.push('\n');
    s
}

/// One entry of a `.debug` file: the def it is about, its line, and its
/// stamp, for `lens check`.
#[derive(Clone, Debug, PartialEq)]
pub struct FileEntry {
    pub name: String,
    pub line: usize,
    pub stamp: Option<String>,
    /// `(line, the step an `at` names)`
    pub at: Vec<(usize, String)>,
}

/// The entries of a `.debug` file (the text lines only; `check_file` parses it).
pub fn file_entries(text: &str) -> Vec<FileEntry> {
    let mut out: Vec<FileEntry> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if !line.starts_with(char::is_whitespace) {
            let mut words = t.split_whitespace();
            let name = words.next().unwrap_or("").to_string();
            let stamp = words.find_map(|w| w.strip_prefix('@')).map(str::to_string);
            out.push(FileEntry { name, line: i + 1, stamp, at: Vec::new() });
        } else if let Some(e) = out.last_mut() {
            let words: Vec<&str> = t.split_whitespace().collect();
            if words.len() >= 3 && (words[0] == "watch" || words[0] == "hold") && words[1] == "at" {
                e.at.push((i + 1, words[2].to_string()));
            }
        }
    }
    out
}

/// Parse a `.debug` file on its own: the errors, each as `line: message`.
pub fn check_file(text: &str) -> Result<Vec<DebugBlock>, String> {
    let src = file_to_source(text);
    let line_of = |sp: Span| src[..(sp.start as usize).min(src.len())].matches('\n').count() + 1;
    let toks = crate::lexer::lex(&src, SourceId(0)).map_err(|ds| ds.iter().map(|d| format!("{}: {}", d.primary.as_ref().map(|p| line_of(p.span)).unwrap_or(0), d.message)).collect::<Vec<_>>().join("\n"))?;
    let prog = crate::parser::parse(toks, SourceId(0)).map_err(|ds| {
        ds.iter()
            .map(|d| {
                let line = d.primary.as_ref().map(|p| line_of(p.span)).unwrap_or(0);
                let label = d.primary.as_ref().map(|p| p.message.clone()).unwrap_or_default();
                if label.is_empty() { format!("{line}: {}", d.message) } else { format!("{line}: {} ({label})", d.message) }
            })
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    if let Some(it) = prog.items.first() {
        let span = match it {
            Item::Def(d) => d.span,
            Item::Stmt(s) => s.span(),
            Item::Record(r) => r.span,
            Item::Enum(e) => e.span,
            Item::Rung(r) => r.span,
        };
        return Err(format!("{}: a .debug file holds entries and their watch, hold and trace lines, not code", line_of(span)));
    }
    Ok(prog.debug)
}

/// The debug lens of a program as text, for `viv view --lens debug`: each
/// block as its def's entry, in the `.debug` file's layout.
pub fn view(program: &Program, src: &str) -> String {
    let mut out = String::new();
    for b in &program.debug {
        out.push_str(b.def.as_deref().unwrap_or("*"));
        out.push('\n');
        for item in &b.items {
            let sp = match item {
                DebugItem::Watch { span, .. } | DebugItem::Hold { span, .. } | DebugItem::Trace { span, .. } => *span,
            };
            let text = src.get(sp.start as usize..sp.end as usize).unwrap_or("");
            out.push_str("  ");
            out.push_str(&text.split_whitespace().collect::<Vec<_>>().join(" "));
            out.push('\n');
        }
    }
    out
}
