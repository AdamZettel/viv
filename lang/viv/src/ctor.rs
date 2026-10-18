// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Record field defaults and constructor defs.
//!
//! ```viv
//! record GptConfig { vocab: Int, context: Int = 8, width: Int = 16 }
//! cfg = GptConfig { vocab: len(vocab) }          # context 8, width 16
//!
//! def Head(c: GptConfig) -> Head {
//!     hd = c.width / c.heads
//!     return Head { Wq: init(c.width, hd), Wk: init(c.width, hd), Wv: init(c.width, hd) }
//! }
//! heads = [Head(c) for h to c.heads]
//! ```
//!
//! A pass at the end of `crate::parser::parse`, after `crate::surface`, so
//! the interpreter, C and wasm see complete literals and ordinary defs, and a
//! run pays nothing:
//!
//! - **Defaults.** A field may declare a default, a constant expression: a
//!   literal, a top-level constant, or arithmetic of them. A literal that
//!   leaves the field out gets it; a field without one must still be given. A
//!   record update (`R { f: e, ..base }`) takes the fields it leaves out from
//!   `base`, not from the defaults.
//! - **Constructor defs.** `def Head(...) -> Head` is an ordinary def named
//!   after its record, and it must return that record. `Head(c)` calls it and
//!   `Head { ... }` is the literal (the parser tells them apart by `(` and
//!   `{`); `Head` in a type is the record. A record has at most one
//!   constructor, since a def name is declared once. `Head(c)` with no
//!   constructor def is refused with a note on the two ways to build one.
//! - **Evaluation order.** A literal's values run in the order they are
//!   written. The engines compute the values left in a literal in its
//!   record's declaration order, so a literal whose values draw random
//!   numbers or call defs must write those values in declaration order; one
//!   that does not is refused, which keeps the two orders the same whenever
//!   the difference could be seen. (A comprehension or an `if` expression in a
//!   value is computed before the statement, in the order written:
//!   crate::surface.)

use crate::ast::*;
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::{HashMap, HashSet};

type R<T> = Result<T, Vec<Diagnostic>>;

/// Builtins whose value depends on when they run (the random draws), besides
/// `crate::surface::EFFECTS`.
const DRAWS: &[&str] = &["rand", "randn", "srand"];

pub fn complete(mut program: Program) -> R<Program> {
    let mut records: HashMap<String, Vec<FieldDecl>> = HashMap::new();
    let mut defs: HashMap<String, (Span, Option<TypeExpr>)> = HashMap::new();
    let mut tops: HashSet<String> = HashSet::new();
    collect(&program.items, &mut records, &mut defs, &mut tops);
    let any_default = records.values().flatten().any(|f| f.default.is_some());
    let ctor_named = defs.keys().any(|d| records.contains_key(d));
    let mut errors = Vec::new();
    // a default is a constant
    for fields in records.values() {
        for f in fields {
            if let Some(e) = &f.default {
                if let Some((span, what)) = not_constant(e, &tops, &defs) {
                    errors.push(
                        Diagnostic::error(format!("the default of `{}` is not a constant: it {what}", f.name))
                            .with_primary(span, "here")
                            .with_note("a default is a literal, a top-level constant, or arithmetic of them; a value computed from parameters belongs in a constructor def, `def R(...) -> R`"),
                    );
                }
            }
        }
    }
    // a constructor def returns its record
    for (name, (span, ret)) in &defs {
        if records.contains_key(name) && !matches!(ret, Some(TypeExpr::Named { name: r, .. }) if r == name) {
            errors.push(
                Diagnostic::error(format!("`{name}` is the constructor of record `{name}`, so it returns a `{name}`"))
                    .with_primary(*span, format!("declare `-> {name}`"))
                    .with_note("a def named after a record builds that record; give a def that returns something else a snake_case name"),
            );
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let mut cx = Cx { records, defs: defs.keys().cloned().collect(), errors: Vec::new(), fill: any_default };
    if !cx.records.is_empty() || ctor_named {
        program.items = std::mem::take(&mut program.items).into_iter().map(|i| cx.item(i)).collect();
    }
    if cx.errors.is_empty() {
        Ok(program)
    } else {
        Err(cx.errors)
    }
}

fn collect(items: &[Item], records: &mut HashMap<String, Vec<FieldDecl>>, defs: &mut HashMap<String, (Span, Option<TypeExpr>)>, tops: &mut HashSet<String>) {
    for item in items {
        match item {
            Item::Record(r) => {
                records.insert(r.name.clone(), r.fields.clone());
            }
            Item::Def(d) => {
                defs.insert(d.name.clone(), (d.name_span, d.ret.clone()));
            }
            Item::Stmt(Stmt::Assign { name, .. }) => {
                tops.insert(name.clone());
            }
            Item::Rung(r) => collect(&r.items, records, defs, tops),
            Item::Stmt(_) | Item::Enum(_) => {}
        }
    }
}

/// Where and why `e` is not a constant expression, or `None` when it is.
fn not_constant(e: &Expr, tops: &HashSet<String>, defs: &HashMap<String, (Span, Option<TypeExpr>)>) -> Option<(Span, String)> {
    match e {
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => None,
        Expr::Var { name, span } => {
            if tops.contains(name) && !defs.contains_key(name) {
                None
            } else {
                Some((*span, format!("reads `{name}`, which is not a top-level constant")))
            }
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => not_constant(operand, tops, defs),
        Expr::Binary { lhs, rhs, .. } => not_constant(lhs, tops, defs).or_else(|| not_constant(rhs, tops, defs)),
        Expr::ArrLit { elems, .. } => elems.iter().find_map(|x| not_constant(x, tops, defs)),
        Expr::Call { callee, span, .. } => Some((*span, format!("calls `{callee}`"))),
        Expr::Index { span, .. } => Some((*span, "indexes an array".to_string())),
        Expr::Field { span, .. } => Some((*span, "reads a field".to_string())),
        Expr::RecordLit { span, .. } => Some((*span, "builds a record".to_string())),
        Expr::Lambda { span, .. } => Some((*span, "is a lambda".to_string())),
    }
}

struct Cx {
    records: HashMap<String, Vec<FieldDecl>>,
    defs: HashSet<String>,
    errors: Vec<Diagnostic>,
    /// whether any record declares a default (else literals are left as written)
    fill: bool,
}

impl Cx {
    fn item(&mut self, item: Item) -> Item {
        match item {
            Item::Def(mut d) => {
                d.body = self.block(d.body);
                Item::Def(d)
            }
            Item::Stmt(s) => Item::Stmt(self.stmt(s)),
            Item::Rung(mut r) => {
                r.items = std::mem::take(&mut r.items).into_iter().map(|i| self.item(i)).collect();
                Item::Rung(r)
            }
            other => other,
        }
    }

    fn block(&mut self, body: Vec<Stmt>) -> Vec<Stmt> {
        body.into_iter().map(|s| self.stmt(s)).collect()
    }

    fn stmt(&mut self, s: Stmt) -> Stmt {
        match s {
            Stmt::Assign { name, name_span, value, span } => Stmt::Assign { name, name_span, value: self.expr(value), span },
            Stmt::Return { value, span } => Stmt::Return { value: self.expr(value), span },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt { expr: self.expr(expr), span },
            Stmt::Show { expr, span } => Stmt::Show { expr: self.expr(expr), span },
            Stmt::SetIndex { base, indices, value, span } => Stmt::SetIndex {
                base: Box::new(self.expr(*base)),
                indices: indices.into_iter().map(|e| self.expr(e)).collect(),
                value: self.expr(value),
                span,
            },
            Stmt::SetField { base, field, field_span, value, span } => {
                Stmt::SetField { base: Box::new(self.expr(*base)), field, field_span, value: self.expr(value), span }
            }
            Stmt::If { cond, then_body, else_body, span } => {
                Stmt::If { cond: self.expr(cond), then_body: self.block(then_body), else_body: self.block(else_body), span }
            }
            Stmt::While { cond, body, span } => Stmt::While { cond: self.expr(cond), body: self.block(body), span },
            Stmt::For { var, var_span, bound, body, parallel, span } => {
                Stmt::For { var, var_span, bound: self.expr(bound), body: self.block(body), parallel, span }
            }
            Stmt::Test { name, body, span } => Stmt::Test { name, body: self.block(body), span },
            other => other,
        }
    }

    fn expr(&mut self, e: Expr) -> Expr {
        match e {
            // `R { f: e, ..base }`: the fields left out come from `base`
            Expr::Call { callee, mut args, span } if callee == crate::update::WITH && args.len() == 2 => {
                let base = self.expr(args.pop().expect("two arguments"));
                let lit = match args.pop().expect("two arguments") {
                    Expr::RecordLit { type_name, type_span, fields, span } => {
                        let fields = fields.into_iter().map(|(n, s, v)| (n, s, self.expr(v))).collect();
                        Expr::RecordLit { type_name, type_span, fields, span }
                    }
                    other => self.expr(other),
                };
                Expr::Call { callee, args: vec![lit, base], span }
            }
            Expr::Call { callee, args, span } => {
                if self.records.contains_key(&callee) && !self.defs.contains(&callee) {
                    self.errors.push(
                        Diagnostic::error(format!("`{callee}` is a record with no constructor def"))
                            .with_primary(span, "called like a def")
                            .with_note(format!("build it with a literal, `{callee} {{ field: value, ... }}`, or declare `def {callee}(...) -> {callee}` to call it this way")),
                    );
                }
                Expr::Call { callee, args: args.into_iter().map(|a| self.expr(a)).collect(), span }
            }
            Expr::RecordLit { type_name, type_span, fields, span } => {
                let mut fields: Vec<(String, Span, Expr)> = fields.into_iter().map(|(n, s, v)| (n, s, self.expr(v))).collect();
                if let Some(decl) = self.records.get(&type_name).cloned() {
                    self.check_order(&type_name, &decl, &fields);
                    if self.fill {
                        for f in &decl {
                            if let Some(d) = &f.default {
                                if !fields.iter().any(|(n, _, _)| n == &f.name) {
                                    fields.push((f.name.clone(), type_span, d.clone()));
                                }
                            }
                        }
                    }
                }
                Expr::RecordLit { type_name, type_span, fields, span }
            }
            Expr::Lambda { params, body, span } => Expr::Lambda { params, body: Box::new(self.expr(*body)), span },
            Expr::ArrLit { elems, span } => Expr::ArrLit { elems: elems.into_iter().map(|x| self.expr(x)).collect(), span },
            Expr::Index { base, indices, span } => {
                Expr::Index { base: Box::new(self.expr(*base)), indices: indices.into_iter().map(|x| self.expr(x)).collect(), span }
            }
            Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(self.expr(*operand)), span },
            Expr::Not { operand, span } => Expr::Not { operand: Box::new(self.expr(*operand)), span },
            Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op, lhs: Box::new(self.expr(*lhs)), rhs: Box::new(self.expr(*rhs)), span },
            Expr::Field { base, field, field_span, span } => Expr::Field { base: Box::new(self.expr(*base)), field, field_span, span },
            e @ (Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } | Expr::Var { .. }) => e,
        }
    }

    /// The values that draw or call a def must be written in declaration order
    /// (see the module comment).
    fn check_order(&mut self, type_name: &str, decl: &[FieldDecl], fields: &[(String, Span, Expr)]) {
        let pos = |n: &str| decl.iter().position(|f| f.name == n);
        let effectful: Vec<(usize, &String, Span)> =
            fields.iter().filter(|(_, _, v)| self.effects(v)).filter_map(|(n, s, _)| pos(n).map(|p| (p, n, *s))).collect();
        for w in effectful.windows(2) {
            if w[1].0 < w[0].0 {
                self.errors.push(
                    Diagnostic::error(format!("the values of `{}` and `{}` run in the order `{type_name}` declares its fields, not the order written", w[0].1, w[1].1))
                        .with_primary(w[1].2, format!("declared before `{}`", w[0].1))
                        .with_note("both draw random numbers or call a def, so the order shows; write the fields in their declaration order, or compute the values first"),
                );
                return;
            }
        }
    }

    fn effects(&self, e: &Expr) -> bool {
        match e {
            Expr::Call { callee, args, .. } => {
                self.defs.contains(callee)
                    || DRAWS.contains(&callee.as_str())
                    || crate::surface::EFFECTS.contains(&callee.as_str())
                    || args.iter().any(|a| self.effects(a))
            }
            Expr::ArrLit { elems, .. } => elems.iter().any(|a| self.effects(a)),
            Expr::Index { base, indices, .. } => self.effects(base) || indices.iter().any(|a| self.effects(a)),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.effects(operand),
            Expr::Binary { lhs, rhs, .. } => self.effects(lhs) || self.effects(rhs),
            Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| self.effects(x)),
            Expr::Field { base, .. } => self.effects(base),
            Expr::Lambda { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } | Expr::Var { .. } => false,
        }
    }
}
