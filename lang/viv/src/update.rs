// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Record update: `T { f: e, ..base }` is the record `base` with the fields
//! written given new values.
//!
//! A pass over the AST at the start of `crate::parse`, after `crate::tuples`,
//! so the interpreter and both compiled backends see only record literals,
//! field reads and calls they already run. A program with no update comes
//! back untouched.
//!
//! The parser writes `T { f: e, ..base }` as a call of `viv__with` on the
//! partial literal `T { f: e }` and `base`. This pass completes it:
//!
//! - when `base` is a name, into the literal itself, each field not written
//!   read from the name: `T { f: e, g: base.g, ... }`;
//! - otherwise into a call of a def the pass declares once per record and set
//!   of written fields, `viv__with_T__f(e, base)`, which returns that literal
//!   built from its parameters. `base` is then computed once, after the new
//!   values, wherever the expression stands.
//!
//! Either way the new values run in the record's declaration order, as the
//! values of any record literal do, and the
//! base after them.

use crate::ast::{Def, Expr, FieldDecl, Item, Param, Program, Stmt, TypeExpr};
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::HashMap;

/// The call the parser writes for `T { ..., ..base }`: the partial literal,
/// then the base.
pub const WITH: &str = "viv__with";

/// The prefix of the defs this pass declares.
pub const WITH_DEF: &str = "viv__with_";
/// The name of such a def's base parameter.
const BASE: &str = "viv__base";

type R<T> = Result<T, Diagnostic>;

pub fn transform(mut program: Program) -> Result<Program, Vec<Diagnostic>> {
    if !uses_update(&program) {
        return Ok(program);
    }
    let mut records: HashMap<String, Vec<FieldDecl>> = HashMap::new();
    for item in &program.items {
        if let Item::Record(r) = item {
            records.insert(r.name.clone(), r.fields.clone());
        }
    }
    let mut u = Update { records, defs: Vec::new(), declared: HashMap::new() };
    let mut items = Vec::with_capacity(program.items.len());
    for item in std::mem::take(&mut program.items) {
        items.push(match item {
            Item::Def(mut d) => {
                d.body = u.block(d.body).map_err(|e| vec![e])?;
                Item::Def(d)
            }
            Item::Stmt(s) => Item::Stmt(u.stmt(s).map_err(|e| vec![e])?),
            other => other,
        });
    }
    items.extend(u.defs.into_iter().map(Item::Def));
    program.items = items;
    Ok(program)
}

fn uses_update(p: &Program) -> bool {
    fn in_expr(e: &Expr) -> bool {
        match e {
            Expr::Call { callee, args, .. } => callee == WITH || args.iter().any(in_expr),
            Expr::ArrLit { elems, .. } => elems.iter().any(in_expr),
            Expr::Index { base, indices, .. } => in_expr(base) || indices.iter().any(in_expr),
            Expr::Lambda { body, .. } => in_expr(body),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => in_expr(operand),
            Expr::Binary { lhs, rhs, .. } => in_expr(lhs) || in_expr(rhs),
            Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| in_expr(x)),
            Expr::Field { base, .. } => in_expr(base),
            _ => false,
        }
    }
    fn in_stmt(s: &Stmt) -> bool {
        match s {
            Stmt::Assign { value, .. } | Stmt::Return { value, .. } => in_expr(value),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => in_expr(expr),
            Stmt::SetIndex { base, indices, value, .. } => in_expr(base) || indices.iter().any(in_expr) || in_expr(value),
            Stmt::SetField { base, value, .. } => in_expr(base) || in_expr(value),
            Stmt::If { cond, then_body, else_body, .. } => in_expr(cond) || then_body.iter().chain(else_body).any(in_stmt),
            Stmt::While { cond, body, .. } => in_expr(cond) || body.iter().any(in_stmt),
            Stmt::For { bound, body, .. } => in_expr(bound) || body.iter().any(in_stmt),
            Stmt::Test { body, .. } => body.iter().any(in_stmt),
            _ => false,
        }
    }
    p.items.iter().any(|item| match item {
        Item::Def(d) => d.body.iter().any(in_stmt),
        Item::Stmt(s) => in_stmt(s),
        _ => false,
    })
}

struct Update {
    records: HashMap<String, Vec<FieldDecl>>,
    /// the defs declared so far, in order
    defs: Vec<Def>,
    /// their names, for declaring each once
    declared: HashMap<String, ()>,
}

impl Update {
    fn block(&mut self, body: Vec<Stmt>) -> R<Vec<Stmt>> {
        body.into_iter().map(|s| self.stmt(s)).collect()
    }

    fn stmt(&mut self, s: Stmt) -> R<Stmt> {
        Ok(match s {
            Stmt::Assign { name, name_span, value, span } => Stmt::Assign { name, name_span, value: self.expr(value)?, span },
            Stmt::Return { value, span } => Stmt::Return { value: self.expr(value)?, span },
            Stmt::ExprStmt { expr, span } => Stmt::ExprStmt { expr: self.expr(expr)?, span },
            Stmt::Show { expr, span } => Stmt::Show { expr: self.expr(expr)?, span },
            Stmt::SetIndex { base, indices, value, span } => Stmt::SetIndex {
                base: Box::new(self.expr(*base)?),
                indices: self.exprs(indices)?,
                value: self.expr(value)?,
                span,
            },
            Stmt::SetField { base, field, field_span, value, span } => Stmt::SetField { base: Box::new(self.expr(*base)?), field, field_span, value: self.expr(value)?, span },
            Stmt::If { cond, then_body, else_body, span } => {
                Stmt::If { cond: self.expr(cond)?, then_body: self.block(then_body)?, else_body: self.block(else_body)?, span }
            }
            Stmt::While { cond, body, span } => Stmt::While { cond: self.expr(cond)?, body: self.block(body)?, span },
            Stmt::For { var, var_span, bound, body, parallel, span } => {
                Stmt::For { var, var_span, bound: self.expr(bound)?, body: self.block(body)?, parallel, span }
            }
            Stmt::Test { name, body, span } => Stmt::Test { name, body: self.block(body)?, span },
            other => other,
        })
    }

    fn exprs(&mut self, es: Vec<Expr>) -> R<Vec<Expr>> {
        es.into_iter().map(|e| self.expr(e)).collect()
    }

    fn expr(&mut self, e: Expr) -> R<Expr> {
        Ok(match e {
            Expr::Call { callee, args, span } if callee == WITH => {
                let mut args = self.exprs(args)?.into_iter();
                let (Some(lit), Some(base), None) = (args.next(), args.next(), args.next()) else {
                    return Err(Diagnostic::error("internal: a record update without its literal and base").with_primary(span, "here"));
                };
                self.complete(lit, base, span)?
            }
            Expr::Call { callee, args, span } => Expr::Call { callee, args: self.exprs(args)?, span },
            Expr::ArrLit { elems, span } => Expr::ArrLit { elems: self.exprs(elems)?, span },
            Expr::Index { base, indices, span } => Expr::Index { base: Box::new(self.expr(*base)?), indices: self.exprs(indices)?, span },
            Expr::Lambda { params, body, span } => Expr::Lambda { params, body: Box::new(self.expr(*body)?), span },
            Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(self.expr(*operand)?), span },
            Expr::Not { operand, span } => Expr::Not { operand: Box::new(self.expr(*operand)?), span },
            Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op, lhs: Box::new(self.expr(*lhs)?), rhs: Box::new(self.expr(*rhs)?), span },
            Expr::RecordLit { type_name, type_span, fields, span } => {
                let fields = fields.into_iter().map(|(n, s, x)| Ok((n, s, self.expr(x)?))).collect::<R<Vec<_>>>()?;
                Expr::RecordLit { type_name, type_span, fields, span }
            }
            Expr::Field { base, field, field_span, span } => Expr::Field { base: Box::new(self.expr(*base)?), field, field_span, span },
            other => other,
        })
    }

    /// `T { given, ..base }` as a whole literal, or a call of a declared def.
    fn complete(&mut self, lit: Expr, base: Expr, span: Span) -> R<Expr> {
        let Expr::RecordLit { type_name, type_span, fields: given, .. } = lit else {
            return Err(Diagnostic::error("internal: a record update without its literal").with_primary(span, "here"));
        };
        let Some(decl) = self.records.get(&type_name).cloned() else {
            return Err(Diagnostic::error(format!("there is no record `{type_name}` to update"))
                .with_primary(type_span, "not a record declared in this program"));
        };
        let mut values: HashMap<String, Expr> = HashMap::new();
        for (name, fspan, value) in given {
            if !decl.iter().any(|f| f.name == name) {
                return Err(Diagnostic::error(format!("record `{type_name}` has no field `{name}`"))
                    .with_primary(fspan, format!("its fields are {}", decl.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", "))));
            }
            if values.insert(name.clone(), value).is_some() {
                return Err(Diagnostic::error(format!("the field `{name}` is given twice")).with_primary(fspan, "given again here"));
            }
        }
        if let Expr::Var { name: bname, span: bspan } = &base {
            // every field in declaration order: a new value, or the base's
            let fields = decl
                .iter()
                .map(|f| {
                    let value = values.remove(&f.name).unwrap_or_else(|| Expr::Field {
                        base: Box::new(Expr::Var { name: bname.clone(), span: *bspan }),
                        field: f.name.clone(),
                        field_span: *bspan,
                        span: *bspan,
                    });
                    (f.name.clone(), f.name_span, value)
                })
                .collect();
            return Ok(Expr::RecordLit { type_name, type_span, fields, span });
        }
        // a def per record and set of new fields, its parameters in
        // declaration order, then the base
        let written: Vec<&FieldDecl> = decl.iter().filter(|f| values.contains_key(&f.name)).collect();
        let def = format!("{WITH_DEF}{type_name}{}", written.iter().map(|f| format!("__{}", f.name)).collect::<String>());
        if !self.declared.contains_key(&def) {
            self.declared.insert(def.clone(), ());
            let param = |f: &FieldDecl| format!("{WITH_DEF}{}", f.name);
            let mut params: Vec<Param> = written
                .iter()
                .map(|f| Param { name: param(f), span, ty: match &f.ty { TypeExpr::Infer => None, t => Some(t.clone()) } })
                .collect();
            params.push(Param { name: BASE.to_string(), span, ty: Some(TypeExpr::Named { name: type_name.clone(), span: type_span }) });
            let fields = decl
                .iter()
                .map(|f| {
                    let value = if values.contains_key(&f.name) {
                        Expr::Var { name: param(f), span }
                    } else {
                        Expr::Field { base: Box::new(Expr::Var { name: BASE.to_string(), span }), field: f.name.clone(), field_span: span, span }
                    };
                    (f.name.clone(), f.name_span, value)
                })
                .collect();
            let body = vec![Stmt::Return { value: Expr::RecordLit { type_name: type_name.clone(), type_span, fields, span }, span }];
            self.defs.push(Def {
                name: def.clone(),
                name_span: span,
                params,
                ret: Some(TypeExpr::Named { name: type_name.clone(), span: type_span }),
                body,
                span,
            });
        }
        let mut args: Vec<Expr> = written.iter().map(|f| values.remove(&f.name).expect("given")).collect();
        args.push(base);
        Ok(Expr::Call { callee: def, args, span })
    }
}

/// Whether `name` is a def this pass declared: the call graph and the
/// explorer show its caller, not it.
pub fn is_update_def(name: &str) -> bool {
    name.starts_with(WITH_DEF)
}

/// Whether `name` is a def the front end declared for part of an
/// expression: a record update's (this pass) or a lambda body's
/// (`crate::surface::LAMBDA_DEF`). The call graph and the explorer show the
/// expression's def, not it.
pub fn is_expression_def(name: &str) -> bool {
    is_update_def(name) || name.starts_with(crate::surface::LAMBDA_DEF)
}
