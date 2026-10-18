// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The program graph as Viv data: `graph_of_self()` and
//! `program_graph("module")`.
//!
//! Both are resolved when the program is read, like a `#! use` line: the call
//! is replaced by a `ProgramGraph` record literal holding the graph of
//! `crate::graph` as lists of strings and
//! arrays. So the interpreter, the C backend and the wasm backend see only a
//! record literal, and all three print the same bytes; nothing is computed at
//! run time and no engine needs a way to read files.
//!
//! `graph_of_self()` is the graph of the program that makes the call, as it is
//! read (its `#! use` modules spliced in). `program_graph("scf/story_water")`
//! is the graph of another program, named as a `#! use` line names a module:
//! the driver that reads the program installs the resolver
//! (`set_module_resolver`), and the module's own `#! use` lines are expanded
//! with it. The program must declare the record, which `lib/proggraph.viv`
//! does; the fields are checked against `FIELDS`.
//!
//! A program that calls neither comes back unchanged.

use std::sync::{Arc, Mutex};

use knotkit::diag::Diagnostic;
use knotkit::span::Span;

use crate::ast::{Expr, Item, Program, Stmt, TypeExpr};
use crate::graph::Graph;

/// The graph of the program that calls it.
pub const SELF: &str = "graph_of_self";
/// The graph of a module, by its `#! use` name.
pub const OTHER: &str = "program_graph";
/// The record both return.
pub const RECORD: &str = "ProgramGraph";

/// A field's type in `FIELDS`.
#[derive(Clone, Copy, PartialEq)]
enum F {
    Str,
    Strs,
    Arr,
}

/// The record's fields, in declaration order. Node `k` is entry `k` of every
/// node list; lists of names (`params`, `reads`, ...) are one string each,
/// the names separated by single spaces. Indices are node indices; `-1` is
/// none.
const FIELDS: [(&str, F); 19] = [
    ("name", F::Str),
    ("ids", F::Strs),
    ("kinds", F::Strs),
    ("labels", F::Strs),
    ("texts", F::Strs),
    ("parents", F::Arr),
    ("owners", F::Arr),
    ("lines", F::Arr),
    ("ends", F::Arr),
    ("params", F::Strs),
    ("reads", F::Strs),
    ("writes", F::Strs),
    ("globals", F::Strs),
    ("calls", F::Strs),
    ("edge_kinds", F::Strs),
    ("edge_from", F::Arr),
    ("edge_to", F::Arr),
    ("edge_vars", F::Strs),
    ("edge_carried", F::Arr),
];

/// The record as `lib/proggraph.viv` declares it, for the error that asks for it.
fn declaration() -> String {
    let fields: Vec<String> = FIELDS
        .iter()
        .map(|(n, t)| {
            let ty = match t {
                F::Str => "Str",
                F::Strs => "[Str]",
                F::Arr => "Arr",
            };
            format!("{n}: {ty}")
        })
        .collect();
    format!("record {RECORD} {{ {} }}", fields.join(", "))
}

/// How a driver finds a module's text by its `#! use` name.
pub type Resolver = Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

static RESOLVER: Mutex<Option<Resolver>> = Mutex::new(None);

/// Install (or, with `None`, remove) the resolver `program_graph` reads
/// modules with. The CLI installs the one it expands `#! use` lines with.
pub fn set_module_resolver(r: Option<Resolver>) {
    *RESOLVER.lock().unwrap_or_else(|e| e.into_inner()) = r;
}

fn resolver() -> Option<Resolver> {
    RESOLVER.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// How deep `program_graph` may nest: a module whose graph is taken may take
/// another's, but not without end.
const MAX_NESTING: usize = 4;

thread_local! {
    static NESTING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Replace every `graph_of_self()` and `program_graph("m")` with the graph.
/// `src` is the text `program` was parsed from.
pub fn transform(mut program: Program, src: &str) -> Result<Program, Vec<Diagnostic>> {
    let own = |n: &str| program.items.iter().any(|i| matches!(i, Item::Def(d) if d.name == n));
    let names: Vec<&str> = [SELF, OTHER].into_iter().filter(|n| !own(n)).collect();
    if names.is_empty() || !uses(&program, &names) {
        return Ok(program);
    }
    check_record(&program).map_err(|e| vec![e])?;
    // the graph of the program itself is taken before any call is replaced
    let own_graph: Option<Graph> = if names.contains(&SELF) && uses(&program, &[SELF]) { Some(crate::graph::graph(&program, src)) } else { None };
    let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    let mut err: Option<Diagnostic> = None;
    let mut rewrite = |e: &mut Expr| {
        if err.is_some() {
            return;
        }
        let Expr::Call { callee, args, span } = e else { return };
        if !names.contains(callee) {
            return;
        }
        let span = *span;
        let made = if callee == SELF {
            if !args.is_empty() {
                Err(Diagnostic::error(format!("`{SELF}()` takes no arguments")).with_primary(span, "the graph of this program"))
            } else {
                Ok(literal(own_graph.as_ref().expect("taken above"), "self", span))
            }
        } else {
            match args.as_slice() {
                [Expr::Str { value, .. }] => other(value, span).map(|g| literal(&g, value, span)),
                _ => Err(Diagnostic::error(format!("`{OTHER}` takes one string literal: a module name, as `#! use` names it"))
                    .with_primary(span, "like program_graph(\"scf/story_water\")")),
            }
        };
        match made {
            Ok(lit) => *e = lit,
            Err(d) => err = Some(d),
        }
    };
    let mut items = std::mem::take(&mut program.items);
    for item in &mut items {
        match item {
            Item::Def(d) => block(&mut d.body, &mut rewrite),
            Item::Stmt(s) => stmt(s, &mut rewrite),
            _ => {}
        }
    }
    if let Some(d) = err {
        return Err(vec![d]);
    }
    program.items = items;
    Ok(program)
}

/// The graph of the module named `name`, read with the installed resolver.
fn other(name: &str, span: Span) -> Result<Graph, Diagnostic> {
    let Some(resolve) = resolver() else {
        return Err(Diagnostic::error(format!("`{OTHER}(\"{name}\")`: nothing here can read modules"))
            .with_primary(span, "the module's text is not available")
            .with_note(format!("the command line reads modules as `#! use` does; a page that runs Viv gives the text of the program itself to `{SELF}()`")));
    };
    let depth = NESTING.with(|n| n.get());
    if depth >= MAX_NESTING {
        return Err(Diagnostic::error(format!("`{OTHER}` nested more than {MAX_NESTING} deep")).with_primary(span, "a graph of a graph of ..."));
    }
    let text = resolve(name).map_err(|m| Diagnostic::error(format!("`{OTHER}(\"{name}\")`: {m}")).with_primary(span, "this module"))?;
    let expanded = crate::expand_uses(&text, &mut |m: &str| resolve(m))
        .map_err(|m| Diagnostic::error(format!("`{OTHER}(\"{name}\")`: {m}")).with_primary(span, "this module"))?;
    NESTING.with(|n| n.set(depth + 1));
    let parsed = crate::parse(&expanded, knotkit::span::SourceId(0));
    NESTING.with(|n| n.set(depth));
    let mut program = parsed.map_err(|ds| {
        let first = ds.first().map(|d| d.message.clone()).unwrap_or_default();
        Diagnostic::error(format!("`{OTHER}(\"{name}\")`: the module does not parse: {first}")).with_primary(span, "this module")
    })?;
    program.debug.clear();
    Ok(crate::graph::graph(&program, &expanded))
}

/// The program must declare the record, with exactly these fields.
fn check_record(program: &Program) -> Result<(), Diagnostic> {
    let decl = program.items.iter().find_map(|i| match i {
        Item::Record(r) if r.name == RECORD => Some(r),
        _ => None,
    });
    let Some(r) = decl else {
        return Err(Diagnostic::error(format!("`{SELF}` and `{OTHER}` return a `{RECORD}`, and this program does not declare one"))
            .with_primary(Span::new(knotkit::span::SourceId(0), 0, 0), "add `from proggraph import ProgramGraph`")
            .with_note(declaration()));
    };
    let same = r.fields.len() == FIELDS.len()
        && r.fields.iter().zip(FIELDS.iter()).all(|(f, (n, t))| {
            f.name == *n
                && matches!((&f.ty, t), (TypeExpr::Str, F::Str) | (TypeExpr::StrList, F::Strs) | (TypeExpr::Arr, F::Arr))
        });
    if !same {
        return Err(Diagnostic::error(format!("`{RECORD}` is not declared as `{SELF}` fills it")).with_primary(r.name_span, "declared here").with_note(declaration()));
    }
    Ok(())
}

fn uses(p: &Program, names: &[&str]) -> bool {
    let mut found = false;
    let mut look = |e: &mut Expr| {
        if let Expr::Call { callee, .. } = e {
            found |= names.contains(&callee.as_str());
        }
    };
    // a read-only walk, written with the rewriting one on a copy
    let mut items: Vec<Item> = p.items.iter().filter(|i| matches!(i, Item::Def(_) | Item::Stmt(_))).cloned().collect();
    for item in &mut items {
        match item {
            Item::Def(d) => block(&mut d.body, &mut look),
            Item::Stmt(s) => stmt(s, &mut look),
            _ => {}
        }
    }
    found
}

// ---- the literal ---------------------------------------------------------------

fn strs(xs: Vec<String>, span: Span) -> Expr {
    Expr::ArrLit { elems: xs.into_iter().map(|value| Expr::Str { value, span }).collect(), span }
}

fn nums(xs: Vec<f64>, span: Span) -> Expr {
    Expr::ArrLit { elems: xs.into_iter().map(|value| num(value, span)).collect(), span }
}

/// A number literal; a negative one is written as the parser writes `-1`.
fn num(value: f64, span: Span) -> Expr {
    if value < 0.0 {
        Expr::Neg { operand: Box::new(Expr::Num { value: -value, span }), span }
    } else {
        Expr::Num { value, span }
    }
}

/// The graph as a `ProgramGraph` literal, every node of it at `span`.
fn literal(g: &Graph, name: &str, span: Span) -> Expr {
    let index = |id: &str| g.nodes.iter().position(|n| n.id == id).map(|k| k as f64).unwrap_or(-1.0);
    let ns = &g.nodes;
    let join = |xs: &[String]| xs.join(" ");
    // an edge to a name that is not a node (none today) is left out
    let edges: Vec<_> = g.edges.iter().filter(|e| index(&e.from) >= 0.0 && index(&e.to) >= 0.0).collect();
    let mut fields: Vec<(String, Expr)> = Vec::new();
    for (field, _) in FIELDS {
        let value = match field {
            "name" => Expr::Str { value: name.to_string(), span },
            "ids" => strs(ns.iter().map(|n| n.id.clone()).collect(), span),
            "kinds" => strs(ns.iter().map(|n| n.kind.to_string()).collect(), span),
            "labels" => strs(ns.iter().map(|n| n.name.clone()).collect(), span),
            "texts" => strs(ns.iter().map(|n| n.text.clone()).collect(), span),
            "parents" => nums(ns.iter().map(|n| n.parent.as_deref().map_or(-1.0, index)).collect(), span),
            "owners" => nums(ns.iter().map(|n| index(&n.def)).collect(), span),
            "lines" => nums(ns.iter().map(|n| n.line as f64).collect(), span),
            "ends" => nums(ns.iter().map(|n| n.end as f64).collect(), span),
            "params" => strs(ns.iter().map(|n| join(&n.params)).collect(), span),
            // a def's reads are its parameters, already in `params`
            "reads" => strs(ns.iter().map(|n| if n.kind == "step" { join(&n.reads) } else { String::new() }).collect(), span),
            "writes" => strs(ns.iter().map(|n| join(&n.writes)).collect(), span),
            "globals" => strs(ns.iter().map(|n| join(&n.globals)).collect(), span),
            "calls" => strs(ns.iter().map(|n| join(&n.calls)).collect(), span),
            "edge_kinds" => strs(edges.iter().map(|e| e.kind.to_string()).collect(), span),
            "edge_from" => nums(edges.iter().map(|e| index(&e.from)).collect(), span),
            "edge_to" => nums(edges.iter().map(|e| index(&e.to)).collect(), span),
            "edge_vars" => strs(edges.iter().map(|e| join(&e.vars)).collect(), span),
            _ => nums(edges.iter().map(|e| if e.carried { 1.0 } else { 0.0 }).collect(), span),
        };
        fields.push((field.to_string(), value));
    }
    Expr::RecordLit { type_name: RECORD.to_string(), type_span: span, fields: fields.into_iter().map(|(n, v)| (n, span, v)).collect(), span }
}

// ---- a walk that may rewrite expressions -------------------------------------------

fn block(body: &mut [Stmt], f: &mut dyn FnMut(&mut Expr)) {
    for s in body {
        stmt(s, f);
    }
}

fn stmt(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Assign { value, .. } | Stmt::Return { value, .. } => expr(value, f),
        Stmt::ExprStmt { expr: e, .. } | Stmt::Show { expr: e, .. } => expr(e, f),
        Stmt::SetIndex { base, indices, value, .. } => {
            expr(base, f);
            for i in indices {
                expr(i, f);
            }
            expr(value, f);
        }
        Stmt::SetField { base, value, .. } => {
            expr(base, f);
            expr(value, f);
        }
        Stmt::Test { body, .. } => block(body, f),
        Stmt::If { cond, then_body, else_body, .. } => {
            expr(cond, f);
            block(then_body, f);
            block(else_body, f);
        }
        Stmt::While { cond, body, .. } => {
            expr(cond, f);
            block(body, f);
        }
        Stmt::For { bound, body, .. } => {
            expr(bound, f);
            block(body, f);
        }
        Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
    }
}

fn expr(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    match e {
        Expr::Call { args, .. } => {
            for a in args.iter_mut() {
                expr(a, f);
            }
        }
        Expr::ArrLit { elems, .. } => {
            for a in elems.iter_mut() {
                expr(a, f);
            }
        }
        Expr::Index { base, indices, .. } => {
            expr(base, f);
            for i in indices.iter_mut() {
                expr(i, f);
            }
        }
        Expr::Lambda { body, .. } => expr(body, f),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr(operand, f),
        Expr::Binary { lhs, rhs, .. } => {
            expr(lhs, f);
            expr(rhs, f);
        }
        Expr::RecordLit { fields, .. } => {
            for (_, _, x) in fields.iter_mut() {
                expr(x, f);
            }
        }
        Expr::Field { base, .. } => expr(base, f),
        _ => {}
    }
    f(e);
}

// ---- the drawing: `viv graph --svg`, and the site's stage ----------------------------

/// The Viv program that draws `module`'s graph (lib/graphdraw.viv): what
/// `viv graph <program> --svg` and the site's explorer run.
pub fn draw_driver(module: &str) -> String {
    let lit = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    format!("from graphdraw import draw_program\n\ndraw_program({OTHER}({}), {})\n", lit(module), lit(module))
}

/// Draw `module`'s graph: run `draw_driver` in the interpreter, with
/// `resolve` for its modules (lib/graphdraw and the ones it uses, and the
/// program drawn). Returns the figure records.
pub fn draw_records(module: &str, resolve: Resolver) -> Result<String, String> {
    set_module_resolver(Some(resolve.clone()));
    let text = crate::expand_uses(&draw_driver(module), &mut |m: &str| resolve(m));
    let out = text.and_then(|t| crate::interp::run_release(&t));
    set_module_resolver(None);
    // the interpreter's own lines (`__graph__`, `__trace__`) are not the drawing's
    Ok(out?.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect())
}

/// `draw_records` for a page or a build that has the modules' texts:
/// `{"root": "scf/story_water", "modules": [{"name": ..., "src": ...}, ...]}`,
/// the program drawn and every module the drawing reads (its own `#! use`
/// modules, and lib/graphdraw with the ones it uses). Returns the records, or
/// `error: ...`. viv.wasm's `viv_graph_draw`.
pub fn draw_request_json(request: &str) -> String {
    use crate::story::Json;
    let req = match crate::story::parse_json(request) {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let field = |j: &Json, k: &str| -> Option<Json> {
        if let Json::Obj(f) = j {
            f.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
        } else {
            None
        }
    };
    let text = |j: &Json, k: &str| -> Option<String> {
        if let Some(Json::Str(t)) = field(j, k) {
            Some(t)
        } else {
            None
        }
    };
    let Some(root) = text(&req, "root") else { return "error: the request names no root".into() };
    let mut modules: Vec<(String, String)> = Vec::new();
    if let Some(Json::Arr(xs)) = field(&req, "modules") {
        for m in xs {
            modules.push((text(&m, "name").unwrap_or_default(), text(&m, "src").unwrap_or_default()));
        }
    }
    let resolve: Resolver = Arc::new(move |name: &str| {
        modules.iter().find(|(n, _)| n == name).map(|(_, s)| s.clone()).ok_or_else(|| format!("the request does not carry the module {name}"))
    });
    draw_records(&root, resolve).unwrap_or_else(|e| format!("error: {e}"))
}
