// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv lint`: integer literals where a size belongs.
//!
//! A size written as a number (`zeros(37)`, `for k to 32`, `if i < 37`)
//! says nothing about where it came from, and goes wrong silently when the
//! data it counts changes. The size is almost always `len(a)` or `shape(A)`
//! of an array the program has, or a constant the program names once. This
//! check reads the AST (not the text) and reports an integer literal other
//! than 0, 1 and 2 (and their negatives) in three places:
//!
//! - **size**: an argument of `zeros` or `czeros`, the point count of
//!   `linspace` (its third argument), the bin count of `histogram` (its
//!   fourth);
//! - **loop bound**: the bound of `for i to N`, and the `from`, `to` and `by`
//!   of `for i from a to b by s`;
//! - **index bound**: a comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`) of a
//!   variable used as an index (a loop variable, or any variable that appears
//!   inside `a[...]` in the same def) with a literal.
//!
//! Everything else is allowed: 0, 1 and 2 as offsets, strides and small
//! arities; literals in a `NAME = value` definition (that is where a
//! constant is named); literals inside an array literal (data); the expected
//! values of a test's assertions; literal indices (`b[6]`, a formula's
//! coefficients); non-integer numbers. Inside a size or a bound, `a[3]` and
//! `shape(A)[1]` are not looked into: an index there is not a size.
//!
//! The file itself is read, not the modules its `#! use` lines name: each
//! module is linted on its own. A ladder's rungs are read as written.

use std::collections::HashSet;

use viv::ast::{BinOp, Expr, Item, Stmt};
use knotkit::span::SourceId;

pub struct Finding {
    pub line: usize,
    pub col: usize,
    pub value: f64,
    pub what: String,
}

/// The literal's magnitude that is always allowed.
const SMALL: f64 = 2.0;

fn flagged(v: f64) -> bool {
    v.fract() == 0.0 && v.abs() > SMALL
}

struct Lint<'a> {
    src: &'a str,
    found: Vec<(u32, f64, String)>,
    index_vars: HashSet<String>,
}

impl<'a> Lint<'a> {
    fn flag(&mut self, at: u32, v: f64, what: String) {
        if !self.found.iter().any(|(s, _, _)| *s == at) {
            self.found.push((at, v, what));
        }
    }

    /// Every integer literal in `e` that is part of a size or bound: `e` and
    /// its arithmetic, calls included, but not the inside of an index.
    fn in_size(&mut self, e: &Expr, what: &str) {
        match e {
            Expr::Num { value, span } if flagged(*value) => self.flag(span.start, *value, what.to_string()),
            Expr::Neg { operand, .. } => self.in_size(operand, what),
            Expr::Binary { lhs, rhs, .. } => {
                self.in_size(lhs, what);
                self.in_size(rhs, what);
            }
            Expr::Call { args, .. } => {
                for a in args {
                    self.in_size(a, what);
                }
            }
            Expr::Field { base, .. } => self.in_size(base, what),
            _ => {}
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Call { callee, args, .. } => {
                let sized: Vec<usize> = match callee.as_str() {
                    "zeros" | "czeros" => (0..args.len()).collect(),
                    "linspace" => vec![2],
                    "histogram" => vec![3],
                    _ => Vec::new(),
                };
                for (k, a) in args.iter().enumerate() {
                    if sized.contains(&k) {
                        self.in_size(a, &format!("as an array size ({callee})"));
                    } else {
                        self.expr(a);
                    }
                }
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::EqEq | BinOp::Ne) {
                    for (var, lit) in [(&**lhs, &**rhs), (&**rhs, &**lhs)] {
                        if let (Expr::Var { name, .. }, Some((v, at))) = (var, literal(lit)) {
                            if self.index_vars.contains(name) && flagged(v) {
                                self.flag(at, v, format!("as a bound on the index `{name}`"));
                            }
                        }
                    }
                }
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Index { base, indices, .. } => {
                self.expr(base);
                for i in indices {
                    self.expr(i);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    self.expr(x);
                }
            }
            Expr::Lambda { body, .. } => self.expr(body),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.expr(operand),
            Expr::RecordLit { fields, .. } => {
                for (_, _, x) in fields {
                    self.expr(x);
                }
            }
            Expr::Field { base, .. } => self.expr(base),
            _ => {}
        }
    }

    fn stmts(&mut self, body: &[Stmt]) {
        for s in body {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Assign { name, value, .. } => {
                // the bounds of `for i from a to b by s`, which the parser
                // evaluates into hidden variables before the loop
                let hidden = name.strip_prefix(viv::parser::HIDDEN_PREFIX).map(|r| r.trim_end_matches(|c: char| c.is_ascii_digit()));
                if matches!(hidden, Some("lo" | "hi" | "by" | "n")) {
                    self.in_size(value, "as a loop bound");
                }
                self.expr(value);
            }
            Stmt::Return { value, .. } => self.expr(value),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => self.expr(expr),
            Stmt::Test { body, .. } => self.stmts(body),
            Stmt::SetIndex { base, indices, value, .. } => {
                self.expr(base);
                for i in indices {
                    self.expr(i);
                }
                self.expr(value);
            }
            Stmt::SetField { base, value, .. } => {
                self.expr(base);
                self.expr(value);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.expr(cond);
                self.stmts(then_body);
                self.stmts(else_body);
            }
            Stmt::While { cond, body, .. } => {
                self.expr(cond);
                self.stmts(body);
            }
            Stmt::For { bound, body, .. } => {
                self.in_size(bound, "as a loop bound");
                self.expr(bound);
                self.stmts(body);
            }
            _ => {}
        }
    }
}

/// `e` as a number written in the source, and where.
fn literal(e: &Expr) -> Option<(f64, u32)> {
    match e {
        Expr::Num { value, span } => Some((*value, span.start)),
        Expr::Neg { operand, .. } => literal(operand).map(|(v, at)| (-v, at)),
        _ => None,
    }
}

/// The loop variables and the variables used inside an index, in `body`.
fn index_vars(body: &[Stmt], out: &mut HashSet<String>) {
    fn in_index(e: &Expr, out: &mut HashSet<String>) {
        match e {
            Expr::Var { name, .. } => {
                out.insert(name.clone());
            }
            Expr::Binary { lhs, rhs, .. } => {
                in_index(lhs, out);
                in_index(rhs, out);
            }
            Expr::Neg { operand, .. } => in_index(operand, out),
            _ => {}
        }
    }
    fn expr(e: &Expr, out: &mut HashSet<String>) {
        match e {
            Expr::Index { base, indices, .. } => {
                expr(base, out);
                for i in indices {
                    in_index(i, out);
                    expr(i, out);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| expr(a, out)),
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|a| expr(a, out)),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr(operand, out),
            Expr::Lambda { body, .. } => expr(body, out),
            Expr::Field { base, .. } => expr(base, out),
            Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| expr(x, out)),
            _ => {}
        }
    }
    for s in body {
        match s {
            Stmt::Assign { name, value, .. } => {
                // `i = viv__k1 ...`: the variable of `for i from a to b`
                if let Expr::Binary { .. } | Expr::Var { .. } = value {
                    let mut vs = HashSet::new();
                    in_index(value, &mut vs);
                    if vs.iter().any(|v| v.starts_with(viv::parser::HIDDEN_PREFIX) && v[viv::parser::HIDDEN_PREFIX.len()..].starts_with('k')) {
                        out.insert(name.clone());
                    }
                }
                expr(value, out);
            }
            Stmt::Return { value, .. } => expr(value, out),
            Stmt::ExprStmt { expr: e, .. } | Stmt::Show { expr: e, .. } => expr(e, out),
            Stmt::Test { body, .. } => index_vars(body, out),
            Stmt::SetIndex { base, indices, value, .. } => {
                expr(base, out);
                for i in indices {
                    in_index(i, out);
                    expr(i, out);
                }
                expr(value, out);
            }
            Stmt::SetField { base, value, .. } => {
                expr(base, out);
                expr(value, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                expr(cond, out);
                index_vars(then_body, out);
                index_vars(else_body, out);
            }
            Stmt::While { cond, body, .. } => {
                expr(cond, out);
                index_vars(body, out);
            }
            Stmt::For { var, bound, body, .. } => {
                out.insert(var.clone());
                expr(bound, out);
                index_vars(body, out);
            }
            _ => {}
        }
    }
}

fn items<'a>(l: &mut Lint<'a>, its: &[Item], top: &mut Vec<Stmt>) {
    for it in its {
        match it {
            Item::Def(d) => {
                let mut vs = HashSet::new();
                index_vars(&d.body, &mut vs);
                let saved = std::mem::replace(&mut l.index_vars, vs);
                l.stmts(&d.body);
                l.index_vars = saved;
            }
            Item::Stmt(s) => top.push(s.clone()),
            Item::Rung(r) => items(l, &r.items, top),
            Item::Record(_) | Item::Enum(_) => {}
        }
    }
}

/// The findings in one file's source, in source order.
pub fn lint(src: &str) -> Result<Vec<Finding>, String> {
    let prog = viv::parse_keeping_rungs(src, SourceId(0)).map_err(|ds| {
        ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; ")
    })?;
    let mut l = Lint { src, found: Vec::new(), index_vars: HashSet::new() };
    let mut top = Vec::new();
    items(&mut l, &prog.items, &mut top);
    // the top level is one scope: its index variables are those of all of it
    let mut vs = HashSet::new();
    index_vars(&top, &mut vs);
    l.index_vars = vs;
    l.stmts(&top);
    let mut found = std::mem::take(&mut l.found);
    found.sort_by_key(|f| f.0);
    Ok(found
        .into_iter()
        .map(|(at, value, what)| {
            let before = &l.src[..at as usize];
            let line = before.matches('\n').count() + 1;
            let col = before.len() - before.rfind('\n').map_or(0, |k| k + 1) + 1;
            Finding { line, col, value, what }
        })
        .collect())
}

/// Every `.viv` file under `path` (itself, when it is a file), sorted.
pub fn viv_files(path: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        entries.sort();
        for p in entries {
            if p.is_dir() || p.extension().is_some_and(|x| x == "viv") {
                viv_files(&p, out);
            }
        }
    } else {
        out.push(path.to_path_buf());
    }
}

// ---- `viv lint --round`: rounding that an Int would make unnecessary -------------

/// A `round(...)` whose value is used as an index, a slice bound, a loop
/// bound or an array size: directly, or through a variable it is assigned to.
/// With an Int index, or a table of Ints,
/// the rounding goes; an index that is a number must already be exactly an
/// integer, so a `round` of a number read out of a table of integers does
/// nothing. Returns (finding, the total count of `round` calls in the file).
pub fn rounds(src: &str) -> Result<(Vec<Finding>, usize), String> {
    let prog = viv::parse_keeping_rungs(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; "))?;
    fn is_round(e: &Expr) -> bool {
        matches!(e, Expr::Call { callee, args, .. } if callee == "round" && args.len() == 1)
    }
    /// The positions of the `round` calls anywhere in `e`.
    fn rounds_in(e: &Expr, out: &mut Vec<u32>) {
        if is_round(e) {
            out.push(e.span().start);
        }
        match e {
            Expr::Call { args, .. } => args.iter().for_each(|a| rounds_in(a, out)),
            Expr::Binary { lhs, rhs, .. } => {
                rounds_in(lhs, out);
                rounds_in(rhs, out);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => rounds_in(operand, out),
            Expr::Index { base, indices, .. } => {
                rounds_in(base, out);
                indices.iter().for_each(|a| rounds_in(a, out));
            }
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|a| rounds_in(a, out)),
            Expr::Field { base, .. } => rounds_in(base, out),
            Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| rounds_in(x, out)),
            Expr::Lambda { body, .. } => rounds_in(body, out),
            _ => {}
        }
    }
    /// The variables named anywhere in `e`.
    fn vars_in(e: &Expr, out: &mut HashSet<String>) {
        match e {
            Expr::Var { name, .. } => {
                out.insert(name.clone());
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| vars_in(a, out)),
            Expr::Binary { lhs, rhs, .. } => {
                vars_in(lhs, out);
                vars_in(rhs, out);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => vars_in(operand, out),
            Expr::Index { base, indices, .. } => {
                vars_in(base, out);
                indices.iter().for_each(|a| vars_in(a, out));
            }
            Expr::Field { base, .. } => vars_in(base, out),
            _ => {}
        }
    }
    #[derive(Default)]
    struct Uses {
        /// the expressions used as an index, a bound or a size, with what
        places: Vec<(Expr, &'static str)>,
        /// every `name = value`
        assigns: Vec<(String, Expr)>,
        all: Vec<u32>,
    }
    fn walk_expr(e: &Expr, u: &mut Uses) {
        if is_round(e) {
            u.all.push(e.span().start);
        }
        match e {
            Expr::Index { base, indices, .. } => {
                walk_expr(base, u);
                for i in indices {
                    u.places.push((i.clone(), "as an index"));
                    walk_expr(i, u);
                }
            }
            Expr::Call { callee, args, .. } => {
                if callee == "viv__slice_get" {
                    if let Some(spec) = args.get(1) {
                        u.places.push((spec.clone(), "as a slice bound"));
                    }
                }
                if matches!(callee.as_str(), "zeros" | "czeros" | "int_zeros" | "linspace") {
                    for a in args {
                        u.places.push((a.clone(), "as an array size"));
                    }
                }
                args.iter().for_each(|a| walk_expr(a, u));
            }
            Expr::Binary { lhs, rhs, .. } => {
                walk_expr(lhs, u);
                walk_expr(rhs, u);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => walk_expr(operand, u),
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|a| walk_expr(a, u)),
            Expr::Field { base, .. } => walk_expr(base, u),
            Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, x)| walk_expr(x, u)),
            Expr::Lambda { body, .. } => walk_expr(body, u),
            _ => {}
        }
    }
    fn walk(body: &[Stmt], u: &mut Uses) {
        for s in body {
            match s {
                Stmt::Assign { name, value, .. } => {
                    let hidden = name.strip_prefix(viv::parser::HIDDEN_PREFIX).map(|r| r.trim_end_matches(|c: char| c.is_ascii_digit()));
                    if matches!(hidden, Some("lo" | "hi" | "by" | "n")) {
                        u.places.push((value.clone(), "as a loop bound"));
                    }
                    u.assigns.push((name.clone(), value.clone()));
                    walk_expr(value, u);
                }
                Stmt::Return { value: e, .. } | Stmt::ExprStmt { expr: e, .. } | Stmt::Show { expr: e, .. } => walk_expr(e, u),
                Stmt::SetIndex { base, indices, value, .. } => {
                    walk_expr(base, u);
                    for i in indices {
                        u.places.push((i.clone(), "as an index"));
                        walk_expr(i, u);
                    }
                    walk_expr(value, u);
                }
                Stmt::If { cond, then_body, else_body, .. } => {
                    walk_expr(cond, u);
                    walk(then_body, u);
                    walk(else_body, u);
                }
                Stmt::While { cond, body, .. } => {
                    walk_expr(cond, u);
                    walk(body, u);
                }
                Stmt::For { bound, body, .. } => {
                    u.places.push((bound.clone(), "as a loop bound"));
                    walk_expr(bound, u);
                    walk(body, u);
                }
                Stmt::Test { body, .. } => walk(body, u),
                _ => {}
            }
        }
    }
    fn scope(body: &[Stmt], found: &mut Vec<(u32, String)>, total: &mut usize) {
        let mut u = Uses::default();
        walk(body, &mut u);
        let mut all = u.all.clone();
        all.sort();
        all.dedup();
        *total += all.len();
        // the variables a place reads
        let mut placed: Vec<(String, &'static str)> = Vec::new();
        for (e, what) in &u.places {
            let mut at = Vec::new();
            rounds_in(e, &mut at);
            for a in at {
                found.push((a, (*what).to_string()));
            }
            let mut vs = HashSet::new();
            vars_in(e, &mut vs);
            placed.extend(vs.into_iter().map(|v| (v, *what)));
        }
        for (name, v) in &u.assigns {
            if let Some((_, what)) = placed.iter().find(|(p, _)| p == name) {
                let mut at = Vec::new();
                rounds_in(v, &mut at);
                for a in at {
                    found.push((a, format!("{what}, through `{name}`")));
                }
            }
        }
    }
    let mut found: Vec<(u32, String)> = Vec::new();
    let mut total = 0;
    let mut top = Vec::new();
    for item in &prog.items {
        match item {
            Item::Def(d) => scope(&d.body, &mut found, &mut total),
            Item::Stmt(s) => top.push(s.clone()),
            _ => {}
        }
    }
    scope(&top, &mut found, &mut total);
    found.sort_by_key(|f| f.0);
    found.dedup_by_key(|f| f.0);
    Ok((
        found
            .into_iter()
            .map(|(at, what)| {
                let before = &src[..at as usize];
                let line = before.matches('\n').count() + 1;
                let col = before.len() - before.rfind('\n').map_or(0, |k| k + 1) + 1;
                Finding { line, col, value: 0.0, what }
            })
            .collect(),
        total,
    ))
}

/// `viv lint [--count] <file or directory>...`: print each finding as
/// `path:line:col: <literal> <where>` and the line it is on, or with
/// `--count` one line per argument with the number of findings and of files
/// that have any. Exits 1 when there is a finding, 2 when a file cannot be
/// read or parsed.
pub fn cli(args: &[String]) -> std::process::ExitCode {
    let count = args.iter().any(|a| a == "--count");
    // `--round`: the `round(...)` an Int would make unnecessary
    let round = args.iter().any(|a| a == "--round");
    let paths: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
    if paths.is_empty() {
        eprintln!("usage: viv lint [--count] [--round] <file.viv or directory>...");
        return std::process::ExitCode::from(2);
    }
    let (mut total, mut broken) = (0usize, false);
    for arg in paths {
        let mut files = Vec::new();
        viv_files(std::path::Path::new(arg), &mut files);
        let (mut n, mut with) = (0usize, 0usize);
        let mut all_rounds = 0usize;
        for f in &files {
            let shown = f.display().to_string();
            let src = match std::fs::read_to_string(f) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{shown}: cannot read: {e}");
                    broken = true;
                    continue;
                }
            };
            let found = match if round { rounds(&src).map(|(f, t)| { all_rounds += t; f }) } else { lint(&src) } {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{shown}: does not parse: {e}");
                    broken = true;
                    continue;
                }
            };
            if !found.is_empty() {
                with += 1;
            }
            n += found.len();
            if !count {
                for x in &found {
                    let text = src.lines().nth(x.line - 1).unwrap_or("").trim();
                    if round {
                        println!("{shown}:{}:{}: round(...) {}\n    {text}", x.line, x.col, x.what);
                    } else {
                        println!("{shown}:{}:{}: {} {}\n    {text}", x.line, x.col, x.value, x.what);
                    }
                }
            }
        }
        if count && round {
            println!("{arg}: {n} of {all_rounds} round(...) calls used as an index, a bound or a size, in {with} of {} files", files.len());
        } else if count {
            println!("{arg}: {n} literal{} in {with} of {} files", if n == 1 { "" } else { "s" }, files.len());
        }
        total += n;
    }
    if broken {
        std::process::ExitCode::from(2)
    } else if total > 0 {
        std::process::ExitCode::from(1)
    } else {
        std::process::ExitCode::SUCCESS
    }
}
