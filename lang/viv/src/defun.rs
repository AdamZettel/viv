// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Defunctionalization: rewrite function-valued arguments away before lowering,
//! so the C and wasm backends only ever see first-order calls.
//!
//! Viv's interpreter handles function values dynamically (a named `def` or a
//! `fn(x) -> ...` lambda can be passed to a user function and called through
//! the parameter). The compiled backends get the same surface semantics by
//! whole-program specialization instead:
//!
//!   def apply_twice(f, x) { return f(f(x)) }
//!   show apply_twice(triple, 2.0)
//!
//! becomes, before lowering,
//!
//!   def apply_twice__triple(x) { return triple(triple(x)) }
//!   show apply_twice__triple(2.0)
//!
//! Lambdas passed to user functions are first lifted to synthetic top-level
//! defs (`__lam0`, ...), which reduces every function value to a def name and
//! guarantees arguments are evaluated exactly once (no inline duplication).
//! Because function values cannot be stored in arrays or records in v1, every
//! function argument is statically known at its call site, so specialization
//! is complete: after this pass no function value remains.
//!
//! v1 restrictions (each reported as a clear diagnostic):
//!   - a lambda passed to a user function may reference only its own
//!     parameters, top-level bindings, and other functions — not enclosing
//!     locals (the compiled backends have no closure environments);
//!   - a function-typed parameter may be called or forwarded, not otherwise
//!     used as a value.
//!
//! Builtin higher-order callees (`simpson`, `trapezoid`, `deriv`) keep their
//! inline-lambda handling in lower.rs; this pass leaves their arguments alone.

use crate::ast::{Def, Expr, Item, Program, Stmt, TypeExpr};
use knotkit::diag::Diagnostic;
use std::collections::{HashMap, HashSet};

/// A function value known at compile time: the def it is, and the values it
/// captured, passed as trailing arguments at
/// each call. A def name or a lambda that reads no local captures nothing.
#[derive(Clone, Debug)]
struct FnRef {
    def: String,
    caps: Vec<Expr>,
}

impl FnRef {
    fn plain(def: String) -> Self {
        FnRef { def, caps: Vec::new() }
    }
}

/// Maps a function-typed name (a parameter, or a variable bound to a lambda)
/// to the function substituted for it.
type Env = HashMap<String, FnRef>;

/// The prefix of the variables that hold a closure's captured values when the
/// captured name is assigned again later (`viv__cap_g_a`).
const CAP_PREFIX: &str = "viv__cap_";

/// Whether `d` calls a function value it receives through a parameter by
/// `apply` (`apply(p, ...)`, `apply(p[i], ...)`: a registry, as `sweep` in
/// std/contracts.viv takes). Such a def has no type until a call gives its
/// parameter one, so when nothing names it, it is dropped like a higher-order
/// def every call specialized (a program that imports std/contracts for
/// `integrate` alone does not call `sweep`).
fn applies_param(d: &Def) -> bool {
    let params: HashSet<&str> = d.params.iter().map(|p| p.name.as_str()).collect();
    let mut hit = false;
    walk_block(&d.body, &mut |e: &Expr| {
        if let Expr::Call { callee, args, .. } = e {
            if callee == "apply" {
                let base = match args.first() {
                    Some(Expr::Index { base, .. }) => base.as_ref(),
                    Some(other) => other,
                    None => return,
                };
                if matches!(base, Expr::Var { name, .. } if params.contains(name.as_str())) {
                    hit = true;
                }
            }
        }
    });
    hit
}

pub fn specialize(program: Program) -> Result<Program, Vec<Diagnostic>> {
    let mut spec = Spec::default();
    let program_debug = program.debug.clone();
    let program_memory = program.memory.clone();
    let program_whole = program.whole.clone();

    // Pass 0: collect defs and top-level binding names.
    for item in &program.items {
        match item {
            Item::Def(d) => {
                spec.def_names.insert(d.name.clone());
                spec.defs.insert(d.name.clone(), d.clone());
            }
            Item::Stmt(s) => collect_assigned(std::slice::from_ref(s), &mut spec.globals),
            Item::Record(_) | Item::Enum(_) | Item::Rung(_) => {}
        }
    }

    // Pass 0b: which params are function-typed? A param is function-typed if
    // the body calls it, or forwards it into a function-typed position of
    // another def. Iterate to a fixpoint so forwarding chains resolve.
    for (name, d) in &spec.defs {
        let called: HashSet<String> = d
            .params
            .iter()
            .map(|p| p.name.clone())
            .filter(|p| block_calls_name(&d.body, p))
            .collect();
        if !called.is_empty() {
            spec.fn_params.insert(name.clone(), called);
        }
    }
    loop {
        let mut grew = false;
        let snapshot = spec.fn_params.clone();
        for (name, d) in &spec.defs {
            let params: HashSet<String> = d.params.iter().map(|p| p.name.clone()).collect();
            let mut forwards: Vec<(String, usize, String)> = Vec::new();
            collect_forwards(&d.body, &params, &mut forwards);
            for (callee, idx, pname) in forwards {
                let Some(callee_def) = spec.defs.get(&callee) else { continue };
                let Some(callee_param) = callee_def.params.get(idx) else { continue };
                if snapshot.get(&callee).is_some_and(|s| s.contains(&callee_param.name))
                    && spec.fn_params.entry(name.clone()).or_default().insert(pname)
                {
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }

    // Pass 1: rewrite everything. Defs with function-typed params are dropped
    // (only their specializations are emitted); everything else is rewritten in
    // place so nested calls inside first-order defs also specialize. Top-level
    // single-assignment lambda bindings (`g = fn(x) -> ...`) are lifted to defs
    // so `g` can be passed and called like any other function.
    let assign_counts = program_assign_counts(&program);
    let mut env = Env::new();
    let mut out_items: Vec<Item> = Vec::new();
    for item in program.items {
        match item {
            Item::Def(d) => {
                if spec.fn_params.contains_key(&d.name) {
                    // Function-carrying: normally only its specializations
                    // survive. Keep the original in case it is ALSO used as a
                    // value, which specialization cannot serve.
                    spec.dropped.insert(d.name.clone(), d);
                    continue;
                }
                let param_names: Vec<String> = d.params.iter().map(|p| p.name.clone()).collect();
                let locals = body_locals(&param_names, &d.body);
                spec.owner = Some(d.name.clone());
                let body = spec.rewrite_fn_body(&d.body, &env, &locals, &param_names);
                spec.owner = None;
                out_items.push(Item::Def(Def { body, ..d }));
            }
            Item::Rung(r) => out_items.push(Item::Rung(r)),
            Item::Stmt(s) => {
                // Lift `g = fn(x) -> ...` when `g` is assigned exactly once.
                if let Stmt::Assign { name, value: value @ Expr::Lambda { .. }, span, .. } = &s {
                    if assign_counts.get(name).copied().unwrap_or(0) == 1 {
                        if let Some(lifted) = spec.classify_fn_arg(value, &env, &HashSet::new()) {
                            // a top-level lambda reads the program's names, so it captures nothing
                            env.insert(name.clone(), lifted);
                            continue; // the binding itself compiles to nothing
                        }
                        let _ = span; // capture errors already recorded
                        continue;
                    }
                }
                let locals = spec.globals.clone();
                let s = spec.rewrite_stmt(&s, &env, &locals);
                out_items.push(Item::Stmt(s));
            }
            rec @ (Item::Record(_) | Item::Enum(_)) => out_items.push(rec),
        }
    }
    // The defs that call a registry they receive (`applies_param`), and the
    // defs that pass one of their parameters on to such a def (`sweep` hands
    // its registry to `sweep_spread`): typed only by a caller.
    let mut untyped: HashSet<String> = out_items.iter().filter_map(|i| match i {
        Item::Def(d) if applies_param(d) => Some(d.name.clone()),
        _ => None,
    }).collect();
    loop {
        let mut grew = false;
        for item in &out_items {
            let Item::Def(d) = item else { continue };
            if untyped.contains(&d.name) {
                continue;
            }
            let params: HashSet<&str> = d.params.iter().map(|p| p.name.as_str()).collect();
            let mut hit = false;
            walk_block(&d.body, &mut |e: &Expr| {
                if let Expr::Call { callee, args, .. } = e {
                    if untyped.contains(callee) && args.iter().any(|a| matches!(a, Expr::Var { name, .. } if params.contains(name.as_str()))) {
                        hit = true;
                    }
                }
            });
            if hit {
                untyped.insert(d.name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    // A def that every call site specialized is dead in its original form: its
    // function-taking parameter has no representation left, so reaching the
    // lowering it would be typed as a number and fail there instead of here. It
    // survives only while something still names it. Dropping one can orphan
    // another -- `sweep` was the only caller of `sweep_spread` -- so this
    // repeats until nothing more falls away.
    loop {
        let mut named: HashSet<String> = HashSet::new();
        for item in &out_items {
            match item {
                Item::Stmt(st) => collect_names(std::slice::from_ref(st), &mut named),
                Item::Def(d) if !spec.specialized_from.contains(&d.name) && !untyped.contains(&d.name) => collect_names(&d.body, &mut named),
                // a candidate for dropping does not get to keep another alive
                Item::Def(_) | Item::Record(_) | Item::Enum(_) | Item::Rung(_) => {}
            }
        }
        for d in &spec.new_defs {
            collect_names(&d.body, &mut named);
        }
        // a candidate something names keeps what it names
        loop {
            let size = named.len();
            for item in &out_items {
                if let Item::Def(d) = item {
                    if (spec.specialized_from.contains(&d.name) || untyped.contains(&d.name)) && named.contains(&d.name) {
                        collect_names(&d.body, &mut named);
                    }
                }
            }
            if named.len() == size {
                break;
            }
        }
        let before = out_items.len();
        out_items.retain(|item| match item {
            Item::Def(d) => !(spec.specialized_from.contains(&d.name) || untyped.contains(&d.name)) || named.contains(&d.name),
            _ => true,
        });
        if out_items.len() == before {
            break;
        }
    }

    // A higher-order def named in a VALUE position -- `ms = [r_bisect, ...]`,
    // `g = s_left` -- cannot be specialized, because which function it will be
    // called with is decided at run time by an index. Emit a generic copy of
    // each such def: its function parameters become runtime function values and
    // the calls through them become `apply`. Repeat, because a generic copy can
    // name another one.
    loop {
        let mut wanted: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for item in &out_items {
            if let Item::Stmt(st) = item {
                collect_var_names(std::slice::from_ref(st), &mut seen);
            }
            if let Item::Def(d) = item {
                collect_var_names(&d.body, &mut seen);
            }
        }
        for d in &spec.new_defs {
            collect_var_names(&d.body, &mut seen);
        }
        for name in seen {
            if spec.dropped.contains_key(&name) && !spec.generic_done.contains(&name) {
                wanted.push(name);
            }
        }
        if wanted.is_empty() {
            break;
        }
        wanted.sort();
        for name in wanted {
            spec.emit_generic(&name);
        }
    }
    spec.owner = None;

    // A copy is live when a statement or a kept def names it, directly or
    // through other live copies. Rewriting a def in place can make copies for
    // a call whose function argument is a parameter it does not know is a
    // function (`report(func, left_rect)` inside `every_rule(func)`); when the
    // def itself is dropped above, because every call specialized it, those
    // copies and the errors found in them go with it.
    let mut live: HashSet<String> = HashSet::new();
    let mut named: HashSet<String> = HashSet::new();
    for item in &out_items {
        match item {
            Item::Stmt(st) => collect_names(std::slice::from_ref(st), &mut named),
            Item::Def(d) => {
                live.insert(d.name.clone());
                collect_names(&d.body, &mut named);
            }
            Item::Record(_) | Item::Enum(_) | Item::Rung(_) => {}
        }
    }
    loop {
        let mut grew = false;
        for d in &spec.new_defs {
            if named.contains(&d.name) && live.insert(d.name.clone()) {
                collect_names(&d.body, &mut named);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    let dead = |o: &Option<String>| o.as_ref().is_some_and(|o| !live.contains(o));
    let owners = std::mem::take(&mut spec.error_owner);
    spec.errors = std::mem::take(&mut spec.errors).into_iter().zip(owners).filter(|(_, o)| !dead(o)).map(|(e, _)| e).collect();
    spec.new_defs.retain(|d| live.contains(&d.name));

    // Synthesized defs (lifted lambdas and specializations) go at the end; the
    // lowering hoists function signatures, so order is not significant.
    for d in spec.new_defs {
        out_items.push(Item::Def(d));
    }

    if spec.errors.is_empty() {
        Ok(Program { items: out_items, debug: program_debug, memory: program_memory, phrases: Vec::new(), whole: program_whole })
    } else {
        Err(spec.errors)
    }
}

#[derive(Default)]
struct Spec {
    defs: HashMap<String, Def>,
    def_names: HashSet<String>,
    globals: HashSet<String>,
    /// def name -> the subset of its params that are function-typed (called by
    /// the body, or forwarded into a function-typed position).
    fn_params: HashMap<String, HashSet<String>>,
    /// specialization key -> specialized def name.
    memo: HashMap<String, String>,
    /// Higher-order defs that specialization dropped, kept here in case one of
    /// them is also used as a VALUE somewhere -- in a registry, say -- in which
    /// case a generic copy has to survive after all.
    dropped: HashMap<String, Def>,
    /// Generic copies already emitted, so the fixpoint terminates.
    generic_done: HashSet<String>,
    /// Defs that at least one call site specialized.
    specialized_from: HashSet<String>,
    new_defs: Vec<Def>,
    lam_count: usize,
    errors: Vec<Diagnostic>,
    /// The def each error was found in (`None` for a top-level statement), so
    /// an error inside a copy that nothing reaches is dropped with the copy.
    error_owner: Vec<Option<String>>,
    /// The def being written now.
    owner: Option<String>,
}

impl Spec {
    fn err(&mut self, d: Diagnostic) {
        self.errors.push(d);
        self.error_owner.push(self.owner.clone());
    }

    fn rewrite_block(&mut self, body: &[Stmt], env: &Env, locals: &HashSet<String>) -> Vec<Stmt> {
        body.iter().map(|s| self.rewrite_stmt(s, env, locals)).collect()
    }

    /// Rewrite a function body: lift its direct single-assignment lambda
    /// bindings into synthetic defs first (so `g = fn(x) -> ...; g(3)` works in
    /// the compiled backends the way it does in the interpreter), then rewrite
    /// the remaining statements.
    fn rewrite_fn_body(
        &mut self,
        body: &[Stmt],
        outer_env: &Env,
        locals: &HashSet<String>,
        params: &[String],
    ) -> Vec<Stmt> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        count_assigns(body, &mut counts);
        let mut env = outer_env.clone();
        // a parameter or a local shadows a top-level lambda of the same name
        // (`op = fn(x) -> ...` at the top level, `def lanczos(op, ...)`)
        env.retain(|name, _| !params.contains(name) && !locals.contains(name));
        let mut out = Vec::with_capacity(body.len());
        for s in body {
            if let Stmt::Assign { name, value: value @ Expr::Lambda { .. }, span, .. } = s {
                if counts.get(name).copied().unwrap_or(0) == 1 && !params.contains(name) {
                    if let Some(mut lifted) = self.classify_fn_arg(value, &env, locals) {
                        // Capture by value: a captured name that is assigned
                        // again (or is a parameter that is assigned) is read
                        // here, into a variable of the closure's own, so a
                        // later write does not reach the closure. A name
                        // assigned once cannot change after the binding.
                        for cap in lifted.caps.iter_mut() {
                            let Expr::Var { name: c, .. } = cap else { continue };
                            let writes = counts.get(c.as_str()).copied().unwrap_or(0) + usize::from(params.contains(c));
                            if writes > 1 {
                                let snap = format!("{CAP_PREFIX}{name}_{c}");
                                out.push(Stmt::Assign { name: snap.clone(), name_span: *span, value: cap.clone(), span: *span });
                                *cap = Expr::Var { name: snap, span: *span };
                            }
                        }
                        env.insert(name.clone(), lifted);
                    }
                    continue; // binding compiles to nothing (errors recorded)
                }
            }
            out.push(self.rewrite_stmt(s, &env, locals));
        }
        out
    }

    fn rewrite_stmt(&mut self, stmt: &Stmt, env: &Env, locals: &HashSet<String>) -> Stmt {
        match stmt {
            Stmt::Assign { name, name_span, value, span } => Stmt::Assign {
                name: name.clone(),
                name_span: *name_span,
                value: self.rewrite_expr(value, env, locals),
                span: *span,
            },
            Stmt::Return { value, span } => Stmt::Return {
                value: self.rewrite_expr(value, env, locals),
                span: *span,
            },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt {
                expr: self.rewrite_expr(expr, env, locals),
                span: *span,
            },
            Stmt::Show { expr, span } => Stmt::Show {
                expr: self.rewrite_expr(expr, env, locals),
                span: *span,
            },
            Stmt::Test { name, body, span } => Stmt::Test {
                name: name.clone(),
                body: self.rewrite_block(body, env, locals),
                span: *span,
            },
            Stmt::SetIndex { base, indices, value, span } => Stmt::SetIndex {
                base: Box::new(self.rewrite_expr(base, env, locals)),
                indices: indices.iter().map(|e| self.rewrite_expr(e, env, locals)).collect(),
                value: self.rewrite_expr(value, env, locals),
                span: *span,
            },
            Stmt::SetField { base, field, field_span, value, span } => Stmt::SetField {
                base: Box::new(self.rewrite_expr(base, env, locals)),
                field: field.clone(),
                field_span: *field_span,
                value: self.rewrite_expr(value, env, locals),
                span: *span,
            },
            Stmt::If { cond, then_body, else_body, span } => Stmt::If {
                cond: self.rewrite_expr(cond, env, locals),
                then_body: self.rewrite_block(then_body, env, locals),
                else_body: self.rewrite_block(else_body, env, locals),
                span: *span,
            },
            Stmt::While { cond, body, span } => Stmt::While {
                cond: self.rewrite_expr(cond, env, locals),
                body: self.rewrite_block(body, env, locals),
                span: *span,
            },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For {
                var: var.clone(),
                var_span: *var_span,
                bound: self.rewrite_expr(bound, env, locals),
                body: self.rewrite_block(body, env, locals),
                parallel: *parallel,
                span: *span,
            },
            s @ (Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. }) => s.clone(),
        }
    }

    fn rewrite_expr(&mut self, expr: &Expr, env: &Env, locals: &HashSet<String>) -> Expr {
        match expr {
            Expr::Call { callee, args, span } => {
                // A call through a function-typed parameter: substitute the def.
                let target = env.get(callee).cloned();
                let effective = target.as_ref().map_or(callee.as_str(), |t| t.def.as_str());
                // a closure's captured values follow its own arguments
                let caps: Vec<Expr> = target.as_ref().map(|t| t.caps.clone()).unwrap_or_default();

                if self.def_names.contains(effective) || self.memo_names_contains(effective) {
                    // User function: classify each argument, specialize if any
                    // argument is a function value.
                    let mut fn_args: Vec<(usize, FnRef)> = Vec::new();
                    let mut plain: Vec<Expr> = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        if let Some(name) = self.classify_fn_arg(a, env, locals) {
                            fn_args.push((i, name));
                        } else {
                            plain.push(self.rewrite_expr(a, env, locals));
                        }
                    }
                    if fn_args.is_empty() {
                        // Still a plain call, but check no function-typed param went unfed.
                        if let Some(needed) = self.fn_params.get(effective) {
                            if !needed.is_empty() {
                                let mut names: Vec<String> = needed.iter().map(|p| format!("`{p}`")).collect();
                                names.sort();
                                self.err(
                                    Diagnostic::error(format!(
                                        "`{}` expects a function for parameter(s) {} — pass a def name or a `fn(x) -> ...`",
                                        effective,
                                        names.join(", ")
                                    ))
                                    .with_primary(*span, "missing function argument"),
                                );
                            }
                        }
                        plain.extend(caps);
                        return Expr::Call { callee: effective.to_string(), args: plain, span: *span };
                    }
                    self.specialized_from.insert(effective.to_string());
                    let spec_name = self.specialized(effective, &fn_args, *span);
                    plain.extend(caps);
                    for (_, f) in &fn_args {
                        plain.extend(f.caps.iter().cloned());
                    }
                    return Expr::Call { callee: spec_name, args: plain, span: *span };
                }

                // Builtin (or unknown — lower diagnoses those): rewrite args,
                // leaving lambdas inline for lower's own higher-order handling,
                // but still rewriting inside their bodies.
                let args = args
                    .iter()
                    .map(|a| match a {
                        Expr::Lambda { params, body, span } => {
                            let mut inner = locals.clone();
                            inner.extend(params.iter().cloned());
                            Expr::Lambda {
                                params: params.clone(),
                                body: Box::new(self.rewrite_expr(body, env, &inner)),
                                span: *span,
                            }
                        }
                        other => self.rewrite_expr(other, env, locals),
                    })
                    .collect();
                Expr::Call { callee: effective.to_string(), args, span: *span }
            }
            Expr::Binary { op, lhs, rhs, span } => Expr::Binary {
                op: *op,
                lhs: Box::new(self.rewrite_expr(lhs, env, locals)),
                rhs: Box::new(self.rewrite_expr(rhs, env, locals)),
                span: *span,
            },
            Expr::Neg { operand, span } => Expr::Neg {
                operand: Box::new(self.rewrite_expr(operand, env, locals)),
                span: *span,
            },
            Expr::Not { operand, span } => Expr::Not {
                operand: Box::new(self.rewrite_expr(operand, env, locals)),
                span: *span,
            },
            Expr::ArrLit { elems, span } => Expr::ArrLit {
                elems: elems.iter().map(|e| self.rewrite_expr(e, env, locals)).collect(),
                span: *span,
            },
            Expr::Index { base, indices, span } => Expr::Index {
                base: Box::new(self.rewrite_expr(base, env, locals)),
                indices: indices.iter().map(|e| self.rewrite_expr(e, env, locals)).collect(),
                span: *span,
            },
            Expr::Lambda { params, body, span } => {
                let mut inner = locals.clone();
                inner.extend(params.iter().cloned());
                Expr::Lambda {
                    params: params.clone(),
                    body: Box::new(self.rewrite_expr(body, env, &inner)),
                    span: *span,
                }
            }
            Expr::Field { base, field, field_span, span } => Expr::Field {
                base: Box::new(self.rewrite_expr(base, env, locals)),
                field: field.clone(),
                field_span: *field_span,
                span: *span,
            },
            Expr::RecordLit { type_name, type_span, fields, span } => Expr::RecordLit {
                type_name: type_name.clone(),
                type_span: *type_span,
                fields: fields
                    .iter()
                    .map(|(n, s, e)| (n.clone(), *s, self.rewrite_expr(e, env, locals)))
                    .collect(),
                span: *span,
            },
            // A function-typed parameter standing on its own -- `apply(g, f, x)`
            // passes `f` along as a value -- becomes the def it was specialized
            // to, which the lowering then emits as a function value.
            Expr::Var { name, span } if env.contains_key(name) => {
                let f = &env[name];
                if !f.caps.is_empty() {
                    self.err(closure_as_value(name, *span));
                }
                Expr::Var { name: f.def.clone(), span: *span }
            }
            e @ (Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } | Expr::Var { .. }) => e.clone(),
        }
    }

    /// Is this argument a function value? Returns the function it denotes.
    fn classify_fn_arg(&mut self, arg: &Expr, env: &Env, locals: &HashSet<String>) -> Option<FnRef> {
        match arg {
            Expr::Var { name, .. } => {
                if let Some(target) = env.get(name) {
                    return Some(target.clone()); // forwarding a function param
                }
                if !locals.contains(name) && self.def_names.contains(name) {
                    return Some(FnRef::plain(name.clone())); // a def used as a value
                }
                None
            }
            Expr::Lambda { params, body, span } => {
                // Lift to a synthetic def. The body may use its parameters,
                // top-level bindings and other functions; inside a def it may
                // also read the def's locals, which it captures by value: they
                // become trailing parameters of the lifted def, and each call
                // of the closure passes them.
                let mut free = HashSet::new();
                let bound: HashSet<String> = params.iter().cloned().collect();
                free_vars(body, &bound, &mut free);
                let in_def = self.owner.is_some();
                let mut captured: Vec<String> = Vec::new();
                let mut unknown: Vec<String> = Vec::new();
                for v in free.iter().filter(|v| !env.contains_key(*v)) {
                    if locals.contains(v) && (in_def || !self.globals.contains(v)) {
                        captured.push(v.clone());
                    } else if !self.globals.contains(v) && !self.def_names.contains(v) {
                        unknown.push(v.clone());
                    }
                }
                if !unknown.is_empty() {
                    unknown.sort();
                    let names: Vec<String> = unknown.iter().map(|s| format!("`{s}`")).collect();
                    self.err(
                        Diagnostic::error(format!(
                            "this lambda reads {}, which is not its parameter, a variable where it is made, or a top-level name",
                            names.join(", ")
                        ))
                        .with_primary(*span, "an unknown name"),
                    );
                    return None;
                }
                // A function the body calls that itself captured values: its
                // captured values are captured too, so the lifted def can pass them.
                let mut called: Vec<&String> = env.keys().filter(|k| !bound.contains(*k) && expr_calls_name(body, k)).collect();
                called.sort();
                for k in called {
                    for c in &env[k].caps {
                        if let Expr::Var { name, .. } = c {
                            captured.push(name.clone());
                        }
                    }
                }
                captured.sort();
                captured.dedup();
                let name = format!("__lam{}", self.lam_count);
                self.lam_count += 1;
                let mut inner: HashSet<String> = if in_def { HashSet::new() } else { self.globals.clone() };
                inner.extend(params.iter().cloned());
                inner.extend(captured.iter().cloned());
                let ret = self.rewrite_expr(body, env, &inner);
                let all: Vec<&String> = params.iter().chain(captured.iter()).collect();
                self.new_defs.push(Def {
                    name: name.clone(),
                    name_span: *span,
                    params: all.into_iter().map(|p| crate::ast::Param { name: p.clone(), span: *span, ty: None }).collect(),
                    ret: None,
                    body: vec![Stmt::Return { value: ret, span: *span }],
                    span: *span,
                });
                let caps = captured.into_iter().map(|c| Expr::Var { name: c, span: *span }).collect();
                Some(FnRef { def: name, caps })
            }
            _ => None,
        }
    }

    /// Emit the GENERIC copy of a higher-order def: the one that keeps its
    /// function parameters as runtime values instead of specializing them away.
    /// It keeps the original name, which is free because specialization dropped
    /// the original and names its own results `orig__argname`.
    fn emit_generic(&mut self, name: &str) {
        if !self.generic_done.insert(name.to_string()) {
            return;
        }
        let Some(def) = self.dropped.get(name).cloned() else { return };
        self.owner = Some(name.to_string());
        let fn_params = self.fn_params.get(name).cloned().unwrap_or_default();
        let mut params = Vec::with_capacity(def.params.len());
        for p in &def.params {
            if fn_params.contains(&p.name) {
                let Some(arity) = call_arity(&def.body, &p.name) else {
                    // Forwarded but never called here: its arity is whatever the
                    // def it is forwarded to calls it with, which this pass does
                    // not track. Say so rather than guess.
                    self.err(
                        Diagnostic::error(format!(
                            "`{name}` is used as a value, but its function parameter `{}` is only forwarded, never called -- so how many arguments it takes cannot be read off here",
                            p.name
                        ))
                        .with_primary(p.span, "call it here, or pass this def only to functions that call it"),
                    );
                    return;
                };
                params.push(crate::ast::Param { name: p.name.clone(), span: p.span, ty: Some(TypeExpr::Fn { arity }) });
            } else {
                params.push(p.clone());
            }
        }
        let body = self.generic_block(&def.body, &fn_params);
        self.new_defs.push(Def { name: name.to_string(), name_span: def.name_span, params, ret: def.ret.clone(), body, span: def.span });
    }

    fn generic_block(&mut self, body: &[Stmt], fns: &HashSet<String>) -> Vec<Stmt> {
        body.iter().map(|s| self.generic_stmt(s, fns)).collect()
    }

    fn generic_stmt(&mut self, stmt: &Stmt, fns: &HashSet<String>) -> Stmt {
        match stmt {
            Stmt::Assign { name, name_span, value, span } => Stmt::Assign {
                name: name.clone(),
                name_span: *name_span,
                value: self.generic_expr(value, fns),
                span: *span,
            },
            Stmt::SetIndex { base, indices, value, span } => Stmt::SetIndex {
                base: Box::new(self.generic_expr(base, fns)),
                indices: indices.iter().map(|e| self.generic_expr(e, fns)).collect(),
                value: self.generic_expr(value, fns),
                span: *span,
            },
            Stmt::SetField { base, field, field_span, value, span } => Stmt::SetField {
                base: Box::new(self.generic_expr(base, fns)),
                field: field.clone(),
                field_span: *field_span,
                value: self.generic_expr(value, fns),
                span: *span,
            },
            Stmt::Return { value, span } => Stmt::Return { value: self.generic_expr(value, fns), span: *span },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt { expr: self.generic_expr(expr, fns), span: *span },
            Stmt::Show { expr, span } => Stmt::Show { expr: self.generic_expr(expr, fns), span: *span },
            Stmt::Test { name, body, span } => Stmt::Test {
                name: name.clone(),
                body: self.generic_block(body, fns),
                span: *span,
            },
            Stmt::If { cond, then_body, else_body, span } => Stmt::If {
                cond: self.generic_expr(cond, fns),
                then_body: self.generic_block(then_body, fns),
                else_body: self.generic_block(else_body, fns),
                span: *span,
            },
            Stmt::While { cond, body, span } => Stmt::While {
                cond: self.generic_expr(cond, fns),
                body: self.generic_block(body, fns),
                span: *span,
            },
            Stmt::For { var, var_span, bound, body, parallel, span } => Stmt::For {
                var: var.clone(),
                var_span: *var_span,
                bound: self.generic_expr(bound, fns),
                body: self.generic_block(body, fns),
                parallel: *parallel,
                span: *span,
            },
            other => other.clone(),
        }
    }

    fn generic_expr(&mut self, e: &Expr, fns: &HashSet<String>) -> Expr {
        match e {
            Expr::Call { callee, args, span } => {
                let args: Vec<Expr> = args.iter().map(|a| self.generic_expr(a, fns)).collect();
                // Calling a function this def RECEIVED: an indirect call, which
                // is the whole difference between the generic copy and a
                // specialization.
                if fns.contains(callee) {
                    let mut all = vec![Expr::Var { name: callee.clone(), span: *span }];
                    all.extend(args);
                    return Expr::Call { callee: "apply".to_string(), args: all, span: *span };
                }
                // Handing one of them on: the callee cannot specialize on it
                // either, so it needs a generic copy too.
                if self.dropped.contains_key(callee)
                    && args.iter().any(|a| matches!(a, Expr::Var { name, .. } if fns.contains(name)))
                {
                    let c = callee.clone();
                    self.emit_generic(&c);
                }
                Expr::Call { callee: callee.clone(), args, span: *span }
            }
            Expr::Binary { op, lhs, rhs, span } => Expr::Binary {
                op: *op,
                lhs: Box::new(self.generic_expr(lhs, fns)),
                rhs: Box::new(self.generic_expr(rhs, fns)),
                span: *span,
            },
            Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(self.generic_expr(operand, fns)), span: *span },
            Expr::Not { operand, span } => Expr::Not { operand: Box::new(self.generic_expr(operand, fns)), span: *span },
            Expr::ArrLit { elems, span } => Expr::ArrLit {
                elems: elems.iter().map(|x| self.generic_expr(x, fns)).collect(),
                span: *span,
            },
            Expr::Index { base, indices, span } => Expr::Index {
                base: Box::new(self.generic_expr(base, fns)),
                indices: indices.iter().map(|x| self.generic_expr(x, fns)).collect(),
                span: *span,
            },
            Expr::Field { base, field, field_span, span } => Expr::Field {
                base: Box::new(self.generic_expr(base, fns)),
                field: field.clone(),
                field_span: *field_span,
                span: *span,
            },
            other => other.clone(),
        }
    }

    /// Get or create the specialization of `orig` with the given function
    /// arguments substituted (positions paired with def names).
    fn specialized(&mut self, orig: &str, fn_args: &[(usize, FnRef)], span: knotkit::span::Span) -> String {
        let key = format!(
            "{orig}|{}",
            fn_args.iter().map(|(i, f)| format!("{i}:{}/{}", f.def, f.caps.len())).collect::<Vec<_>>().join(",")
        );
        if let Some(name) = self.memo.get(&key) {
            return name.clone();
        }
        let Some(def) = self.defs.get(orig).cloned() else {
            // A specialization of a synthesized def cannot occur (lambdas are
            // first-order); anything else is a bug surfaced as a diagnostic.
            self.err(
                Diagnostic::error(format!("cannot specialize unknown function `{orig}`"))
                    .with_primary(span, "internal: no such def"),
            );
            return orig.to_string();
        };
        let spec_name = format!(
            "{orig}__{}",
            fn_args.iter().map(|(_, f)| f.def.trim_start_matches("__").to_string()).collect::<Vec<_>>().join("_")
        );
        // Memoize BEFORE rewriting the body so recursive calls terminate.
        self.memo.insert(key, spec_name.clone());

        let fn_positions: HashSet<usize> = fn_args.iter().map(|(i, _)| *i).collect();
        let mut env = Env::new();
        // a closure's captured values arrive as trailing parameters, after
        // those of the function the copy is of
        let mut cap_params: Vec<crate::ast::Param> = Vec::new();
        for (i, target) in fn_args {
            if let Some(p) = def.params.get(*i) {
                let mut caps = Vec::with_capacity(target.caps.len());
                for k in 0..target.caps.len() {
                    let name = format!("{CAP_PREFIX}{}_{k}", p.name);
                    cap_params.push(crate::ast::Param { name: name.clone(), span: p.span, ty: None });
                    caps.push(Expr::Var { name, span: p.span });
                }
                env.insert(p.name.clone(), FnRef { def: target.def.clone(), caps });
            }
        }
        let mut params: Vec<crate::ast::Param> = def
            .params
            .iter()
            .enumerate()
            .filter(|(i, _)| !fn_positions.contains(i))
            .map(|(_, p)| p.clone())
            .collect();
        params.extend(cap_params);
        let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
        let locals = body_locals(&param_names, &def.body);
        let outer = self.owner.replace(spec_name.clone());
        let body = self.rewrite_fn_body(&def.body, &env, &locals, &param_names);
        self.owner = outer;
        self.new_defs.push(Def {
            name: spec_name.clone(),
            name_span: def.name_span,
            params,
            ret: def.ret.clone(),
            body,
            span: def.span,
        });
        spec_name
    }

    fn memo_names_contains(&self, name: &str) -> bool {
        self.memo.values().any(|v| v == name) || self.new_defs.iter().any(|d| d.name == name)
    }
}

/// A closure that captured values used other than by a call or as an argument
/// of a def: returned, stored, or handed to a builtin.
fn closure_as_value(name: &str, span: knotkit::span::Span) -> Diagnostic {
    Diagnostic::error(format!("`{name}` is a closure: it can be called, or passed to a def, but not returned, stored or passed to a builtin"))
        .with_primary(span, "a closure used as a value")
        .with_note("a closure holds copies of the variables it reads, and only a call can pass them on; return or store the values it needs instead, or write a def")
}

/// How many arguments `name` is called with inside this body, if it is called
/// at all. A function parameter's arity is read off its use: `f(a)` makes `f`
/// a one-argument function, and nothing else has to declare it.
fn call_arity(body: &[Stmt], name: &str) -> Option<usize> {
    let mut found: Option<usize> = None;
    walk_block(body, &mut |e| {
        let Expr::Call { callee, args, .. } = e else { return };
        if found.is_some() {
            return;
        }
        if callee == name {
            found = Some(args.len());
        } else if callee == "apply" && matches!(args.first(), Some(Expr::Var { name: n, .. }) if n == name) {
            found = Some(args.len() - 1);
        }
    });
    found
}

/// Every `Var` name mentioned anywhere in a block. This is how a def used as a
/// VALUE is found: its name appears where a variable would.
fn collect_var_names(body: &[Stmt], out: &mut HashSet<String>) {
    walk_block(body, &mut |e| {
        if let Expr::Var { name, .. } = e {
            out.insert(name.clone());
        }
    });
}

/// Every name a block mentions at all: variables and call targets both. What
/// "is this def still reachable" asks.
fn collect_names(body: &[Stmt], out: &mut HashSet<String>) {
    walk_block(body, &mut |e| match e {
        Expr::Var { name, .. } => {
            out.insert(name.clone());
        }
        Expr::Call { callee, .. } => {
            out.insert(callee.clone());
        }
        _ => {}
    });
}

/// Visit every expression in a block, outermost first.
pub(crate) fn walk_block(body: &[Stmt], f: &mut impl FnMut(&Expr)) {
    for s in body {
        walk_stmt(s, f);
    }
}

fn walk_stmt(s: &Stmt, f: &mut impl FnMut(&Expr)) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => walk_expr(value, f),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => walk_expr(expr, f),
        Stmt::SetIndex { base, indices, value, span: _ } => {
            walk_expr(base, f);
            for i in indices {
                walk_expr(i, f);
            }
            walk_expr(value, f);
        }
        Stmt::SetField { base, value, .. } => {
            walk_expr(base, f);
            walk_expr(value, f);
        }
        Stmt::Test { body, .. } => walk_block(body, f),
        Stmt::If { cond, then_body, else_body, .. } => {
            walk_expr(cond, f);
            walk_block(then_body, f);
            walk_block(else_body, f);
        }
        Stmt::While { cond, body, .. } => {
            walk_expr(cond, f);
            walk_block(body, f);
        }
        Stmt::For { bound, body, .. } => {
            walk_expr(bound, f);
            walk_block(body, f);
        }
        Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
    }
}

fn walk_expr(e: &Expr, f: &mut impl FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Call { args, .. } => {
            for a in args {
                walk_expr(a, f);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk_expr(operand, f),
        Expr::ArrLit { elems, .. } => {
            for x in elems {
                walk_expr(x, f);
            }
        }
        Expr::Index { base, indices, .. } => {
            walk_expr(base, f);
            for i in indices {
                walk_expr(i, f);
            }
        }
        Expr::Lambda { body, .. } => walk_expr(body, f),
        Expr::Field { base, .. } => walk_expr(base, f),
        Expr::RecordLit { fields, .. } => {
            for (_, _, v) in fields {
                walk_expr(v, f);
            }
        }
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } | Expr::Var { .. } => {}
    }
}

/// Every name a body binds: parameters, assignment targets, loop variables.
/// Viv scoping is function-level, so one flat set is exact.
fn body_locals(params: &[String], body: &[Stmt]) -> HashSet<String> {
    let mut set: HashSet<String> = params.iter().cloned().collect();
    collect_assigned(body, &mut set);
    set
}

/// How many times each name is assigned across the whole body (nested included).
fn count_assigns(body: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in body {
        match s {
            Stmt::Assign { name, .. } => {
                *out.entry(name.clone()).or_insert(0) += 1;
            }
            Stmt::For { var, body, .. } => {
                *out.entry(var.clone()).or_insert(0) += 1;
                count_assigns(body, out);
            }
            Stmt::If { then_body, else_body, .. } => {
                count_assigns(then_body, out);
                count_assigns(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => count_assigns(body, out),
            _ => {}
        }
    }
}

/// Assignment counts across all top-level statements of the program.
fn program_assign_counts(program: &Program) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for item in &program.items {
        if let Item::Stmt(s) = item {
            count_assigns(std::slice::from_ref(s), &mut counts);
        }
    }
    counts
}

/// Collect (callee, argument index, param name) for every call argument that is
/// a bare Var naming one of `params` — the forwarding sites.
fn collect_forwards(body: &[Stmt], params: &HashSet<String>, out: &mut Vec<(String, usize, String)>) {
    fn walk_expr(e: &Expr, params: &HashSet<String>, out: &mut Vec<(String, usize, String)>) {
        match e {
            Expr::Call { callee, args, .. } => {
                for (i, a) in args.iter().enumerate() {
                    if let Expr::Var { name, .. } = a {
                        if params.contains(name) {
                            out.push((callee.clone(), i, name.clone()));
                        }
                    }
                    walk_expr(a, params, out);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                walk_expr(lhs, params, out);
                walk_expr(rhs, params, out);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk_expr(operand, params, out),
            Expr::ArrLit { elems, .. } => {
                for a in elems {
                    walk_expr(a, params, out);
                }
            }
            Expr::Index { base, indices, .. } => {
                walk_expr(base, params, out);
                for a in indices {
                    walk_expr(a, params, out);
                }
            }
            Expr::Lambda { body, .. } => walk_expr(body, params, out),
            Expr::Field { base, .. } => walk_expr(base, params, out),
            Expr::RecordLit { fields, .. } => {
                for (_, _, a) in fields {
                    walk_expr(a, params, out);
                }
            }
            Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } | Expr::Var { .. } => {}
        }
    }
    fn walk_stmt(s: &Stmt, params: &HashSet<String>, out: &mut Vec<(String, usize, String)>) {
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => walk_expr(value, params, out),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => walk_expr(expr, params, out),
            Stmt::Test { body, .. } => {
                for s in body {
                    walk_stmt(s, params, out);
                }
            }
            Stmt::SetIndex { base, indices, value, .. } => {
                walk_expr(base, params, out);
                for e in indices {
                    walk_expr(e, params, out);
                }
                walk_expr(value, params, out);
            }
            Stmt::SetField { base, value, .. } => {
                walk_expr(base, params, out);
                walk_expr(value, params, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                walk_expr(cond, params, out);
                for s in then_body {
                    walk_stmt(s, params, out);
                }
                for s in else_body {
                    walk_stmt(s, params, out);
                }
            }
            Stmt::While { cond, body, .. } => {
                walk_expr(cond, params, out);
                for s in body {
                    walk_stmt(s, params, out);
                }
            }
            Stmt::For { bound, body, .. } => {
                walk_expr(bound, params, out);
                for s in body {
                    walk_stmt(s, params, out);
                }
            }
            Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
        }
    }
    for s in body {
        walk_stmt(s, params, out);
    }
}

fn collect_assigned(body: &[Stmt], out: &mut HashSet<String>) {
    for s in body {
        match s {
            Stmt::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::For { var, body, .. } => {
                out.insert(var.clone());
                collect_assigned(body, out);
            }
            Stmt::If { then_body, else_body, .. } => {
                collect_assigned(then_body, out);
                collect_assigned(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => collect_assigned(body, out),
            _ => {}
        }
    }
}

/// Does the block contain a call whose callee is `name`?
fn block_calls_name(body: &[Stmt], name: &str) -> bool {
    body.iter().any(|s| stmt_calls_name(s, name))
}

fn stmt_calls_name(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => expr_calls_name(value, name),
        Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => expr_calls_name(expr, name),
        Stmt::Test { body, .. } => block_calls_name(body, name),
        Stmt::SetIndex { base, indices, value, .. } => {
            expr_calls_name(base, name)
                || indices.iter().any(|e| expr_calls_name(e, name))
                || expr_calls_name(value, name)
        }
        Stmt::SetField { base, value, .. } => expr_calls_name(base, name) || expr_calls_name(value, name),
        Stmt::If { cond, then_body, else_body, .. } => {
            expr_calls_name(cond, name) || block_calls_name(then_body, name) || block_calls_name(else_body, name)
        }
        Stmt::While { cond, body, .. } => expr_calls_name(cond, name) || block_calls_name(body, name),
        Stmt::For { bound, body, .. } => expr_calls_name(bound, name) || block_calls_name(body, name),
        Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => false,
    }
}

fn expr_calls_name(e: &Expr, name: &str) -> bool {
    match e {
        // `apply(p, ...)` is a call to `p` as surely as `p(...)` is -- it is the
        // form a body uses when the function came out of a registry, and the
        // form a generic copy is written in.
        Expr::Call { callee, args, .. } if callee == "apply" => {
            matches!(args.first(), Some(Expr::Var { name: n, .. }) if n == name)
                || args.iter().any(|a| expr_calls_name(a, name))
        }
        Expr::Call { callee, args, .. } => callee == name || args.iter().any(|a| expr_calls_name(a, name)),
        Expr::Binary { lhs, rhs, .. } => expr_calls_name(lhs, name) || expr_calls_name(rhs, name),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_calls_name(operand, name),
        Expr::ArrLit { elems, .. } => elems.iter().any(|a| expr_calls_name(a, name)),
        Expr::Index { base, indices, .. } => {
            expr_calls_name(base, name) || indices.iter().any(|a| expr_calls_name(a, name))
        }
        Expr::Lambda { body, .. } => expr_calls_name(body, name),
        Expr::Field { base, .. } => expr_calls_name(base, name),
        Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, a)| expr_calls_name(a, name)),
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } | Expr::Var { .. } => false,
    }
}

/// Free variables of an expression (Var nodes not bound by `bound`).
fn free_vars(e: &Expr, bound: &HashSet<String>, out: &mut HashSet<String>) {
    match e {
        Expr::Var { name, .. } => {
            if !bound.contains(name) {
                out.insert(name.clone());
            }
        }
        Expr::Call { args, .. } => {
            for a in args {
                free_vars(a, bound, out);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            free_vars(lhs, bound, out);
            free_vars(rhs, bound, out);
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => free_vars(operand, bound, out),
        Expr::ArrLit { elems, .. } => {
            for a in elems {
                free_vars(a, bound, out);
            }
        }
        Expr::Index { base, indices, .. } => {
            free_vars(base, bound, out);
            for a in indices {
                free_vars(a, bound, out);
            }
        }
        Expr::Lambda { params, body, .. } => {
            let mut inner = bound.clone();
            inner.extend(params.iter().cloned());
            free_vars(body, &inner, out);
        }
        Expr::Field { base, .. } => free_vars(base, bound, out),
        Expr::RecordLit { fields, .. } => {
            for (_, _, a) in fields {
                free_vars(a, bound, out);
            }
        }
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => {}
    }
}
