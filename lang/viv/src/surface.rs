// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Expressions that need statements to compute: comprehensions
//! (`[f(x) for x in xs if c]`), the if-expression (`a if c else b`), and the
//! calls with a `key=` (`max(xs, key=f)`, `sorted(xs, key=f)`,
//! `argsort(xs, key=f)`) and the generator forms of `all` and `any`.
//!
//! The parser writes each one as a call of `viv__lift` on a number, and keeps,
//! under that number, the statements that compute the value into a hidden
//! variable (`crate::parser::HIDDEN_PREFIX`), and the variable. This pass, run
//! at the end of `crate::parser::parse`, puts those statements in front of the
//! statement that uses the value and writes the variable in place of the call,
//! so the interpreter, the C backend and the wasm backend see only statements
//! they already run. A program with none of these forms comes back unchanged.
//!
//! Order and laziness are kept:
//!
//! - The branches of an if-expression are statements inside an `if`, so only
//!   the chosen one runs; the same holds for the right side of `and` and `or`
//!   when it holds a lifted value (`ok = n > 0 and a[0] if f else 0.0` reads
//!   `a[0]` only when `n > 0`).
//! - What runs before a lifted value in the expression still runs before it:
//!   a call of a def of the program (or of `print`, `rand`, the file builtins)
//!   written to its left is computed into a hidden variable first.
//! - A `while` whose condition holds a lifted value is written
//!   `while true { <statements>; if not c { break }; body }`, so the value is
//!   computed again before every test.
//!
//! A lambda body is a single expression with nowhere to put statements. A
//! lambda whose body holds a lifted form (`fn(text) -> [stoi[c] for c in
//! text]`) gets a def of its own, declared here
//! (`viv__fnbody_N`), whose body is the statements and a `return`; the lambda
//! becomes a call of it. The locals of the def that makes the lambda that the
//! body reads are the def's trailing parameters, and the call passes them, so
//! the lambda still captures them by value (crate::defun lifts it as any
//! other). A `key=` lambda is not a lambda at run time: the parser writes its
//! body in place.

use crate::ast::*;
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::HashSet;

/// The call the parser writes for a value computed by statements.
pub const LIFT: &str = "viv__lift";

/// The prefix of the def that computes a lambda's body when the body needs
/// statements. It is part of the expression that wrote the lambda: the call
/// graph and the explorer show the lambda, not it (`crate::update::is_expression_def`).
pub const LAMBDA_DEF: &str = "viv__fnbody_";

/// A value computed by statements: run `stmts`, then `result` (a hidden
/// variable) holds it.
#[derive(Clone, Debug)]
pub struct Lifted {
    pub stmts: Vec<Stmt>,
    pub result: Expr,
}

/// The placeholder expression for lifted value `id`.
pub fn placeholder(id: usize, span: Span) -> Expr {
    Expr::Call { callee: LIFT.to_string(), args: vec![Expr::Num { value: id as f64, span }], span }
}

/// The number of a placeholder, or `None` for any other expression.
pub fn placeholder_id(e: &Expr) -> Option<usize> {
    match e {
        Expr::Call { callee, args, .. } if callee == LIFT => match args.as_slice() {
            [Expr::Num { value, .. }] => Some(*value as usize),
            _ => None,
        },
        _ => None,
    }
}

/// Builtins whose call does something besides giving a value: they must run
/// in the order they are written, relative to a lifted value.
pub const EFFECTS: &[&str] = &[
    "print", "rand", "srand", "read_text", "read_lines", "read_bytes", "write_text", "stop", "checkpoint", "sort", "fft", "ifft",
];

type R<T> = Result<T, Diagnostic>;

/// Put every lifted value's statements in front of the statement that uses
/// it (see the module comment). `hidden` is the parser's count of hidden
/// variables, so the names made here are new; `defs` are the program's defs.
pub fn lift(mut program: Program, lifted: Vec<Lifted>, hidden: u32, defs: HashSet<String>) -> R<Program> {
    if lifted.is_empty() {
        return Ok(program);
    }
    let mut cx = Cx { lifted: lifted.into_iter().map(Some).collect(), hidden, defs, scope: HashSet::new(), new_defs: Vec::new(), lambdas: 0 };
    program.items = cx.items(std::mem::take(&mut program.items))?;
    program.items.extend(std::mem::take(&mut cx.new_defs).into_iter().map(Item::Def));
    for block in &program.debug {
        for item in &block.items {
            if let DebugItem::Hold { check, span, .. } = item {
                let bad = match check {
                    HoldCheck::Cond(e) => has_lift(e),
                    HoldCheck::Near { diff, tol } => has_lift(diff) || has_lift(tol),
                };
                if bad {
                    return Err(Diagnostic::error("a hold is one expression: a comprehension, an `if` expression or a `key=` cannot be in it")
                        .with_primary(*span, "in a debug block")
                        .with_note("compute the value in the def and watch or hold the variable"));
                }
            }
        }
    }
    if let Some(l) = cx.lifted.iter().flatten().next() {
        let span = l.result.span();
        return Err(Diagnostic::error("this expression cannot be computed here").with_primary(span, "no statement to compute it before"));
    }
    Ok(program)
}

/// Whether `e` holds a lifted value anywhere, lambda bodies included.
pub fn has_lift(e: &Expr) -> bool {
    match e {
        Expr::Call { callee, args, .. } => callee == LIFT || args.iter().any(has_lift),
        Expr::ArrLit { elems, .. } => elems.iter().any(has_lift),
        Expr::Index { base, indices, .. } => has_lift(base) || indices.iter().any(has_lift),
        Expr::Lambda { body, .. } => has_lift(body),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => has_lift(operand),
        Expr::Binary { lhs, rhs, .. } => has_lift(lhs) || has_lift(rhs),
        Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| has_lift(x)),
        Expr::Field { base, .. } => has_lift(base),
        Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } | Expr::Var { .. } => false,
    }
}

struct Cx {
    lifted: Vec<Option<Lifted>>,
    hidden: u32,
    defs: HashSet<String>,
    /// The names a lambda written here captures when it reads them: the
    /// parameters and locals of the def being written (and of the lambdas
    /// around it). Empty at the top level, where a lambda reads the program's
    /// names as a def does.
    scope: HashSet<String>,
    /// The defs declared for lambda bodies.
    new_defs: Vec<Def>,
    lambdas: usize,
}

/// The names a def's body binds: its assignments, loop variables and the
/// targets of `a, b = e`.
fn bound_names(body: &[Stmt], out: &mut HashSet<String>) {
    for s in body {
        match s {
            Stmt::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::For { var, body, .. } => {
                out.insert(var.clone());
                bound_names(body, out);
            }
            Stmt::If { then_body, else_body, .. } => {
                bound_names(then_body, out);
                bound_names(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => bound_names(body, out),
            Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } if callee == crate::tuples::UNPACK => {
                if let Some(Expr::ArrLit { elems, .. }) = args.first() {
                    for e in elems {
                        if let Expr::Var { name, .. } = e {
                            out.insert(name.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl Cx {
    /// `fn(params) -> body` whose body holds a lifted value: a def computes
    /// the body, and the lambda calls it with its parameters and the locals
    /// the body reads (see the module comment).
    fn lambda(&mut self, params: Vec<String>, body: Expr, span: Span) -> R<Expr> {
        let outer = self.scope.clone();
        self.scope.extend(params.iter().cloned());
        let mut stmts = Vec::new();
        let result = self.flat(body, &mut stmts);
        self.scope = outer;
        let result = result?;
        let mut read: HashSet<String> = HashSet::new();
        let mut note = |e: &Expr| match e {
            Expr::Var { name, .. } | Expr::Call { callee: name, .. } => {
                read.insert(name.clone());
            }
            _ => {}
        };
        crate::defun::walk_block(&stmts, &mut note);
        crate::defun::walk_block(&[Stmt::Return { value: result.clone(), span }], &mut note);
        let mut caps: Vec<String> = read.into_iter().filter(|n| self.scope.contains(n) && !params.contains(n)).collect();
        caps.sort();
        let name = format!("{LAMBDA_DEF}{}", self.lambdas);
        self.lambdas += 1;
        self.defs.insert(name.clone());
        let all: Vec<String> = params.iter().chain(caps.iter()).cloned().collect();
        stmts.push(Stmt::Return { value: result, span });
        self.new_defs.push(Def {
            name: name.clone(),
            name_span: span,
            params: all.iter().map(|p| Param { name: p.clone(), span, ty: None }).collect(),
            ret: None,
            body: stmts,
            span,
        });
        let args = all.into_iter().map(|p| Expr::Var { name: p, span }).collect();
        Ok(Expr::Lambda { params, body: Box::new(Expr::Call { callee: name, args, span }), span })
    }

    fn hidden_var(&mut self, what: &str, span: Span) -> (String, Expr) {
        self.hidden += 1;
        let name = format!("{}{what}{}", crate::parser::HIDDEN_PREFIX, self.hidden);
        (name.clone(), Expr::Var { name, span })
    }

    /// Whether computing `e` does something a reader could see besides its
    /// value, or holds a lifted value: then it runs before what follows it.
    /// A lambda is a value; its body runs later.
    fn effects(&self, e: &Expr) -> bool {
        match e {
            Expr::Call { callee, args, .. } => {
                callee == LIFT || self.defs.contains(callee) || EFFECTS.contains(&callee.as_str()) || args.iter().any(|a| self.effects(a))
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

    fn items(&mut self, items: Vec<Item>) -> R<Vec<Item>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Item::Def(mut d) => {
                    let mut scope: HashSet<String> = d.params.iter().map(|p| p.name.clone()).collect();
                    bound_names(&d.body, &mut scope);
                    let outer = std::mem::replace(&mut self.scope, scope);
                    let body = self.block(d.body);
                    self.scope = outer;
                    d.body = body?;
                    out.push(Item::Def(d));
                }
                Item::Stmt(s) => {
                    let mut stmts = Vec::new();
                    self.stmt(s, &mut stmts)?;
                    out.extend(stmts.into_iter().map(Item::Stmt));
                }
                Item::Rung(mut r) => {
                    r.items = self.items(std::mem::take(&mut r.items))?;
                    out.push(Item::Rung(r));
                }
                Item::Record(r) => out.push(Item::Record(r)),
                Item::Enum(e) => out.push(Item::Enum(e)),
            }
        }
        Ok(out)
    }

    fn block(&mut self, body: Vec<Stmt>) -> R<Vec<Stmt>> {
        let mut out = Vec::with_capacity(body.len());
        for s in body {
            self.stmt(s, &mut out)?;
        }
        Ok(out)
    }

    fn stmt(&mut self, s: Stmt, out: &mut Vec<Stmt>) -> R<()> {
        match s {
            Stmt::Assign { name, name_span, value, span } => {
                let value = self.flat(value, out)?;
                out.push(Stmt::Assign { name, name_span, value, span });
            }
            Stmt::Return { value, span } => {
                let value = self.flat(value, out)?;
                out.push(Stmt::Return { value, span });
            }
            Stmt::ExprStmt { expr, span } => {
                let expr = self.flat(expr, out)?;
                out.push(Stmt::ExprStmt { expr, span });
            }
            Stmt::Show { expr, span } => {
                let expr = self.flat(expr, out)?;
                out.push(Stmt::Show { expr, span });
            }
            Stmt::SetIndex { base, indices, value, span } => {
                // Python computes the value first, then the target
                let value = self.flat(value, out)?;
                let mut kids = vec![*base];
                kids.extend(indices);
                let mut kids = self.seq(kids, out)?;
                let base = Box::new(kids.remove(0));
                out.push(Stmt::SetIndex { base, indices: kids, value, span });
            }
            Stmt::SetField { base, field, field_span, value, span } => {
                let value = self.flat(value, out)?;
                let base = Box::new(self.flat(*base, out)?);
                out.push(Stmt::SetField { base, field, field_span, value, span });
            }
            Stmt::If { cond, then_body, else_body, span } => {
                let cond = self.flat(cond, out)?;
                let then_body = self.block(then_body)?;
                let else_body = self.block(else_body)?;
                out.push(Stmt::If { cond, then_body, else_body, span });
            }
            Stmt::While { cond, body, span } => {
                if has_lift(&cond) {
                    let cspan = cond.span();
                    let mut full = Vec::new();
                    let cond = self.flat(cond, &mut full)?;
                    full.push(Stmt::If {
                        cond: Expr::Not { operand: Box::new(cond), span: cspan },
                        then_body: vec![Stmt::Break { span: cspan }],
                        else_body: Vec::new(),
                        span: cspan,
                    });
                    full.extend(self.block(body)?);
                    out.push(Stmt::While { cond: Expr::Bool { value: true, span: cspan }, body: full, span });
                } else {
                    let body = self.block(body)?;
                    out.push(Stmt::While { cond, body, span });
                }
            }
            Stmt::For { var, var_span, bound, body, parallel, span } => {
                let bound = self.flat(bound, out)?;
                let body = self.block(body)?;
                out.push(Stmt::For { var, var_span, bound, body, parallel, span });
            }
            Stmt::Test { name, body, span } => {
                let body = self.block(body)?;
                out.push(Stmt::Test { name, body, span });
            }
            Stmt::Lens { lens, span } => {
                let lens = match *lens {
                    Lens::Iter { def, counter, end } => Lens::Iter { def, counter, end: self.block(end)? },
                    Lens::Hold { def, at, check, sentence } => {
                        let bad = match &check {
                            HoldCheck::Cond(e) => has_lift(e),
                            HoldCheck::Near { diff, tol } => has_lift(diff) || has_lift(tol),
                        };
                        if bad {
                            return Err(Diagnostic::error("a hold cannot hold a comprehension, an `if` expression or a `key=`").with_primary(span, "here"));
                        }
                        Lens::Hold { def, at, check, sentence }
                    }
                    other @ (Lens::Call { .. } | Lens::IterEnd { .. } | Lens::Watch { .. } | Lens::Placeholder { .. } | Lens::Stop { .. } | Lens::Inline { .. } | Lens::Ret { .. } | Lens::Bind { .. }) => other,
                };
                out.push(Stmt::Lens { lens: Box::new(lens), span });
            }
            s @ (Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. }) => out.push(s),
        }
        Ok(())
    }

    /// `e` with each lifted value replaced by its variable, the statements that
    /// compute them appended to `out` in the order the expression computes them.
    fn flat(&mut self, e: Expr, out: &mut Vec<Stmt>) -> R<Expr> {
        if !has_lift(&e) {
            return Ok(e);
        }
        if let Some(id) = placeholder_id(&e) {
            let l = self.lifted.get_mut(id).and_then(Option::take).expect("a lifted value is used once");
            for s in l.stmts {
                self.stmt(s, out)?;
            }
            return Ok(l.result);
        }
        match e {
            // `a and b`, `a or b`: b is computed only when a does not decide
            Expr::Binary { op: op @ (BinOp::And | BinOp::Or), lhs, rhs, span } if has_lift(&rhs) => {
                let l = self.flat(*lhs, out)?;
                let (name, t) = self.hidden_var("sc", span);
                out.push(Stmt::Assign { name: name.clone(), name_span: span, value: l, span });
                let mut inner = Vec::new();
                let r = self.flat(*rhs, &mut inner)?;
                inner.push(Stmt::Assign { name, name_span: span, value: r, span });
                let cond = if op == BinOp::And { t.clone() } else { Expr::Not { operand: Box::new(t.clone()), span } };
                out.push(Stmt::If { cond, then_body: inner, else_body: Vec::new(), span });
                Ok(t)
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let mut k = self.seq(vec![*lhs, *rhs], out)?;
                let rhs = k.pop().expect("two operands");
                let lhs = k.pop().expect("two operands");
                Ok(Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span })
            }
            Expr::Lambda { params, body, span } => self.lambda(params, *body, span),
            Expr::Neg { operand, span } => Ok(Expr::Neg { operand: Box::new(self.flat(*operand, out)?), span }),
            Expr::Not { operand, span } => Ok(Expr::Not { operand: Box::new(self.flat(*operand, out)?), span }),
            Expr::Field { base, field, field_span, span } => Ok(Expr::Field { base: Box::new(self.flat(*base, out)?), field, field_span, span }),
            // the targets of `a, b = e` are names and elements, not values
            Expr::Call { callee, mut args, span } if callee == crate::tuples::UNPACK && args.len() == 2 => {
                let rhs = args.pop().expect("two arguments");
                let rhs = self.flat(rhs, out)?;
                args.push(rhs);
                Ok(Expr::Call { callee, args, span })
            }
            Expr::Call { callee, args, span } => Ok(Expr::Call { callee, args: self.seq(args, out)?, span }),
            Expr::ArrLit { elems, span } => Ok(Expr::ArrLit { elems: self.seq(elems, out)?, span }),
            Expr::Index { base, indices, span } => {
                let mut kids = vec![*base];
                kids.extend(indices);
                let mut kids = self.seq(kids, out)?;
                let base = Box::new(kids.remove(0));
                Ok(Expr::Index { base, indices: kids, span })
            }
            Expr::RecordLit { type_name, type_span, fields, span } => {
                let (names, values): (Vec<(String, Span)>, Vec<Expr>) = fields.into_iter().map(|(n, s, v)| ((n, s), v)).unzip();
                let values = self.seq(values, out)?;
                let fields = names.into_iter().zip(values).map(|((n, s), v)| (n, s, v)).collect();
                Ok(Expr::RecordLit { type_name, type_span, fields, span })
            }
            e @ (Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } | Expr::Var { .. }) => Ok(e),
        }
    }

    /// Operands computed left to right. Those before the last one that holds
    /// a lifted value, and that hold one or have an effect, are computed into
    /// hidden variables first, so they still run before it.
    fn seq(&mut self, kids: Vec<Expr>, out: &mut Vec<Stmt>) -> R<Vec<Expr>> {
        let Some(last) = kids.iter().rposition(has_lift) else {
            return Ok(kids);
        };
        let mut done = Vec::with_capacity(kids.len());
        for (k, e) in kids.into_iter().enumerate() {
            done.push(if k < last { self.early(e, out)? } else if k == last { self.flat(e, out)? } else { e });
        }
        Ok(done)
    }

    /// An operand that must be computed before a lifted value to its right.
    fn early(&mut self, e: Expr, out: &mut Vec<Stmt>) -> R<Expr> {
        if !self.effects(&e) {
            return Ok(e);
        }
        match e {
            // a literal keeps its shape (a slice's spec is read as one)
            Expr::ArrLit { elems, span } => {
                let mut done = Vec::with_capacity(elems.len());
                for x in elems {
                    done.push(self.early(x, out)?);
                }
                Ok(Expr::ArrLit { elems: done, span })
            }
            e => {
                let span = e.span();
                let value = self.flat(e, out)?;
                let (name, v) = self.hidden_var("e", span);
                out.push(Stmt::Assign { name, name_span: span, value, span });
                Ok(v)
            }
        }
    }
}
