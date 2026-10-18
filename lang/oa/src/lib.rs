// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `oa`: a tiny operator-algebra DSL built on knotkit — a *second language*
//! whose only purpose is to probe the kit's generality (the ATLAS/CMS idea:
//! a redundant, independent exercise of the same machinery).
//!
//! It is deliberately unlike Knot:
//!   * its values are 2x2 operators, not numbers;
//!   * `*` is the (non-commutative) operator product, not multiplication;
//!   * it brings an entirely different C runtime via the same `CEmit::prelude`
//!     hook (1.6, A.3) — proving the kit imposes no runtime of its own.
//!
//! Surface:
//! ```text
//! let a = X * Z          # built-in operators X, Z, I; `*` is operator product
//! let b = Z * X
//! show a - b             # the commutator [X, Z]
//! show comm(X, Z)        # ...the same, via a builtin
//! show scale(2.0, X) + I # scalar multiple plus identity
//! ```
//!
//! The whole language — lexer, parser, IR, and C emission — lives in this one
//! file, so it reads as "here is an entire DSL on the kit."

use std::collections::HashSet;

use knotkit::backend::{CEmit, CEmitCx, CExpr, CItem, CStmt, CType, Runtime, RuntimeFragment};
use knotkit::diag::Diagnostic;
use knotkit::ir::{Hasher, IrProgram, Node, NodeId, NodeKind};
use knotkit::span::{SourceId, Span};

/// Parse, check, and lower `oa` source into kit IR.
pub fn build_ir(src: &str, source: SourceId) -> Result<IrProgram<Oa>, Vec<Diagnostic>> {
    let toks = lex(src, source)?;
    let items = Parser { toks, pos: 0 }.program().map_err(|d| vec![d])?;
    lower(&items, source)
}

// ───────────────────────────────── lexer ───────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Let,
    Show,
    LParen,
    RParen,
    Comma,
    Eq,
    Plus,
    Minus,
    Star,
    Ident(String),
    Num(f64),
    Newline,
    Eof,
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    span: Span,
}

fn lex(src: &str, source: SourceId) -> Result<Vec<Token>, Vec<Diagnostic>> {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut toks = Vec::new();
    let mut errs = Vec::new();
    let mut paren_depth: i32 = 0;
    let at = |a: usize, b: usize| Span::new(source, a as u32, b as u32);

    while i < n {
        let c = bytes[i] as char;
        match c {
            '\n' => {
                if paren_depth == 0 {
                    toks.push(Token { tok: Tok::Newline, span: at(i, i + 1) });
                }
                i += 1;
            }
            ';' => { toks.push(Token { tok: Tok::Newline, span: at(i, i + 1) }); i += 1; }
            c if c.is_whitespace() => i += 1,
            '#' => { while i < n && bytes[i] != b'\n' { i += 1; } }
            '(' => { toks.push(Token { tok: Tok::LParen, span: at(i, i + 1) }); paren_depth += 1; i += 1; }
            ')' => { toks.push(Token { tok: Tok::RParen, span: at(i, i + 1) }); paren_depth -= 1; i += 1; }
            ',' => { toks.push(Token { tok: Tok::Comma, span: at(i, i + 1) }); i += 1; }
            '=' => { toks.push(Token { tok: Tok::Eq, span: at(i, i + 1) }); i += 1; }
            '+' => { toks.push(Token { tok: Tok::Plus, span: at(i, i + 1) }); i += 1; }
            '-' => { toks.push(Token { tok: Tok::Minus, span: at(i, i + 1) }); i += 1; }
            '*' => { toks.push(Token { tok: Tok::Star, span: at(i, i + 1) }); i += 1; }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < n && (bytes[i] as char).is_ascii_digit() { i += 1; }
                if i < n && bytes[i] == b'.' {
                    i += 1;
                    while i < n && (bytes[i] as char).is_ascii_digit() { i += 1; }
                }
                match src[start..i].parse::<f64>() {
                    Ok(v) => toks.push(Token { tok: Tok::Num(v), span: at(start, i) }),
                    Err(_) => errs.push(Diagnostic::error("invalid number").with_primary(at(start, i), "here")),
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < n && { let ch = bytes[i] as char; ch.is_alphanumeric() || ch == '_' } { i += 1; }
                let text = &src[start..i];
                let tok = match text {
                    "let" => Tok::Let,
                    "show" => Tok::Show,
                    _ => Tok::Ident(text.to_string()),
                };
                toks.push(Token { tok, span: at(start, i) });
            }
            other => {
                errs.push(Diagnostic::error(format!("unexpected character {:?}", other)).with_primary(at(i, i + 1), "here"));
                i += 1;
            }
        }
    }
    toks.push(Token { tok: Tok::Eof, span: at(n, n) });
    if errs.is_empty() { Ok(toks) } else { Err(errs) }
}

// ────────────────────────────────── AST ────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum AstOp {
    Add,
    Sub,
    Mul,
}

enum Item {
    Let { name: String, name_span: Span, value: Expr },
    Show { value: Expr },
}

enum Expr {
    Num { value: f64, span: Span },
    Ident { name: String, span: Span },
    Call { callee: String, args: Vec<Expr>, span: Span },
    Bin { op: AstOp, lhs: Box<Expr>, rhs: Box<Expr>, span: Span },
}

impl Expr {
    fn span(&self) -> Span {
        match self {
            Expr::Num { span, .. } | Expr::Ident { span, .. } | Expr::Call { span, .. } | Expr::Bin { span, .. } => *span,
        }
    }
}

// ───────────────────────────────── parser ──────────────────────────────────

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

type PResult<T> = Result<T, Diagnostic>;

impl Parser {
    fn peek(&self) -> &Token { &self.toks[self.pos] }
    fn span(&self) -> Span { self.peek().span }
    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() { self.pos += 1; }
        t
    }
    fn at(&self, t: &Tok) -> bool { &self.peek().tok == t }
    fn skip_newlines(&mut self) { while self.at(&Tok::Newline) { self.bump(); } }
    fn expect(&mut self, t: Tok, what: &str) -> PResult<Token> {
        if self.peek().tok == t { Ok(self.bump()) } else { Err(self.err(format!("expected {what}"))) }
    }
    fn err(&self, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::error(msg).with_primary(self.span(), "here")
    }

    fn program(&mut self) -> PResult<Vec<Item>> {
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::Eof) { break; }
            items.push(self.item()?);
        }
        Ok(items)
    }

    fn item(&mut self) -> PResult<Item> {
        match self.peek().tok {
            Tok::Let => {
                self.bump();
                let (name, name_span) = self.ident("a name")?;
                self.expect(Tok::Eq, "`=`")?;
                let value = self.expr()?;
                Ok(Item::Let { name, name_span, value })
            }
            Tok::Show => {
                self.bump();
                let value = self.expr()?;
                Ok(Item::Show { value })
            }
            _ => Err(self.err("expected `let` or `show`")),
        }
    }

    fn ident(&mut self, what: &str) -> PResult<(String, Span)> {
        match &self.peek().tok {
            Tok::Ident(s) => {
                let s = s.clone();
                let span = self.span();
                self.bump();
                Ok((s, span))
            }
            _ => Err(self.err(format!("expected {what}"))),
        }
    }

    fn expr(&mut self) -> PResult<Expr> { self.additive() }

    fn additive(&mut self) -> PResult<Expr> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek().tok {
                Tok::Plus => AstOp::Add,
                Tok::Minus => AstOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.term()?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn term(&mut self) -> PResult<Expr> {
        let mut lhs = self.factor()?;
        while self.at(&Tok::Star) {
            self.bump();
            let rhs = self.factor()?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Bin { op: AstOp::Mul, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn factor(&mut self) -> PResult<Expr> {
        let tok = self.peek().tok.clone();
        let span = self.span();
        match tok {
            Tok::Num(v) => { self.bump(); Ok(Expr::Num { value: v, span }) }
            Tok::LParen => {
                self.bump();
                let e = self.expr()?;
                self.expect(Tok::RParen, "`)`")?;
                Ok(e)
            }
            Tok::Ident(name) => {
                self.bump();
                if self.at(&Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if !self.at(&Tok::RParen) {
                        loop {
                            args.push(self.expr()?);
                            if self.at(&Tok::Comma) { self.bump(); } else { break; }
                        }
                    }
                    let close = self.expect(Tok::RParen, "`)`")?;
                    Ok(Expr::Call { callee: name, args, span: span.merge(close.span) })
                } else {
                    Ok(Expr::Ident { name, span })
                }
            }
            _ => Err(self.err("expected an operator expression")),
        }
    }
}

// ─────────────────────────────── IR vocabulary ─────────────────────────────

/// The built-in operators.
#[derive(Clone, Copy, Debug)]
pub enum Builtin {
    X,
    Z,
    I,
}

impl Builtin {
    fn runtime(self) -> &'static str {
        match self {
            Builtin::X => "oa_X",
            Builtin::Z => "oa_Z",
            Builtin::I => "oa_I",
        }
    }
    /// The runtime fragment name this builtin needs.
    fn frag(self) -> &'static str {
        match self {
            Builtin::X => "X",
            Builtin::Z => "Z",
            Builtin::I => "I",
        }
    }
    fn tag(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OaOp {
    Add,
    Sub,
    Mul,
}

impl OaOp {
    fn runtime(self) -> &'static str {
        match self {
            OaOp::Add => "oa_add",
            OaOp::Sub => "oa_sub",
            OaOp::Mul => "oa_mul",
        }
    }
    /// The runtime fragment name this operator needs.
    fn frag(self) -> &'static str {
        match self {
            OaOp::Add => "add",
            OaOp::Sub => "sub",
            OaOp::Mul => "mul",
        }
    }
    fn tag(self) -> u8 {
        self as u8
    }
}

/// The `oa` IR vocabulary. All values are operators (`oa_op` in C).
#[derive(Clone, Debug)]
pub enum Oa {
    /// The whole program -> `main`. children: [items...].
    Program,
    /// `let name = value`. children: [value].
    Let { name: String },
    /// `show value`. children: [value].
    Show,
    /// A built-in operator (X/Z/I). leaf.
    Const(Builtin),
    /// A reference to a `let`-bound operator. leaf.
    Var { name: String },
    /// `+`/`-`/`*`. children: [lhs, rhs].
    Bin(OaOp),
    /// `comm(a, b)` = a*b - b*a. children: [a, b].
    Comm,
    /// `scale(s, m)` = s·m. children: [m]; the scalar is payload.
    Scale { s: f64 },
}

impl NodeKind for Oa {
    fn tag(&self) -> u32 {
        match self {
            Oa::Program => 1,
            Oa::Let { .. } => 2,
            Oa::Show => 3,
            Oa::Const(_) => 4,
            Oa::Var { .. } => 5,
            Oa::Bin(_) => 6,
            Oa::Comm => 7,
            Oa::Scale { .. } => 8,
        }
    }
    fn hash_payload(&self, h: &mut Hasher) {
        match self {
            Oa::Let { name } | Oa::Var { name } => {
                h.update(&(name.len() as u32).to_le_bytes());
                h.update(name.as_bytes());
            }
            Oa::Const(b) => { h.update(&[b.tag()]); }
            Oa::Bin(op) => { h.update(&[op.tag()]); }
            Oa::Scale { s } => { h.update(&s.to_bits().to_le_bytes()); }
            Oa::Program | Oa::Show | Oa::Comm => {}
        }
    }
}

// ──────────────────────────────── lowering ─────────────────────────────────

fn lower(items: &[Item], source: SourceId) -> Result<IrProgram<Oa>, Vec<Diagnostic>> {
    let mut errs = Vec::new();
    let mut prog = IrProgram::new();
    let mut scope: HashSet<String> = HashSet::new();
    let mut item_ids = Vec::new();

    for item in items {
        match item {
            Item::Let { name, name_span, value } => match lower_expr(&mut prog, value, &scope) {
                Ok(vid) => {
                    if scope.contains(name) {
                        errs.push(Diagnostic::error(format!("`{name}` is already defined")).with_primary(*name_span, "redefined here"));
                        continue;
                    }
                    scope.insert(name.clone());
                    item_ids.push(prog.add(Oa::Let { name: name.clone() }, *name_span, vec![vid]));
                }
                Err(e) => errs.push(e),
            },
            Item::Show { value } => match lower_expr(&mut prog, value, &scope) {
                Ok(vid) => item_ids.push(prog.add(Oa::Show, value.span(), vec![vid])),
                Err(e) => errs.push(e),
            },
        }
    }

    if !errs.is_empty() {
        return Err(errs);
    }
    let root = prog.add(Oa::Program, Span::synthetic(source), item_ids);
    prog.add_root(root);
    Ok(prog)
}

/// Lower an operator-typed expression. Numbers are not operators and are only
/// valid as the first argument of `scale` (handled there), so a bare number
/// here is an error.
fn lower_expr(prog: &mut IrProgram<Oa>, expr: &Expr, scope: &HashSet<String>) -> Result<NodeId, Diagnostic> {
    match expr {
        Expr::Num { span, .. } => Err(Diagnostic::error("a number is not an operator")
            .with_primary(*span, "numbers are only allowed as the scale factor")),
        Expr::Ident { name, span } => match name.as_str() {
            "X" => Ok(prog.add(Oa::Const(Builtin::X), *span, vec![])),
            "Z" => Ok(prog.add(Oa::Const(Builtin::Z), *span, vec![])),
            "I" => Ok(prog.add(Oa::Const(Builtin::I), *span, vec![])),
            _ if scope.contains(name) => Ok(prog.add(Oa::Var { name: name.clone() }, *span, vec![])),
            _ => Err(Diagnostic::error(format!("unknown operator `{name}`"))
                .with_primary(*span, "not a built-in (X/Z/I) or a `let`-bound name")),
        },
        Expr::Bin { op, lhs, rhs, span } => {
            let lid = lower_expr(prog, lhs, scope)?;
            let rid = lower_expr(prog, rhs, scope)?;
            let op = match op {
                AstOp::Add => OaOp::Add,
                AstOp::Sub => OaOp::Sub,
                AstOp::Mul => OaOp::Mul,
            };
            Ok(prog.add(Oa::Bin(op), *span, vec![lid, rid]))
        }
        Expr::Call { callee, args, span } => match callee.as_str() {
            "comm" => {
                if args.len() != 2 {
                    return Err(Diagnostic::error("`comm` takes 2 operators").with_primary(*span, "wrong arity"));
                }
                let a = lower_expr(prog, &args[0], scope)?;
                let b = lower_expr(prog, &args[1], scope)?;
                Ok(prog.add(Oa::Comm, *span, vec![a, b]))
            }
            "scale" => {
                if args.len() != 2 {
                    return Err(Diagnostic::error("`scale` takes a number and an operator").with_primary(*span, "wrong arity"));
                }
                let s = match &args[0] {
                    Expr::Num { value, .. } => *value,
                    other => {
                        return Err(Diagnostic::error("`scale`'s first argument must be a number literal")
                            .with_primary(other.span(), "expected a number"))
                    }
                };
                let m = lower_expr(prog, &args[1], scope)?;
                Ok(prog.add(Oa::Scale { s }, *span, vec![m]))
            }
            _ => Err(Diagnostic::error(format!("unknown function `{callee}`")).with_primary(*span, "no such builtin")),
        },
    }
}

// ─────────────────────────────── C emission ────────────────────────────────

impl CEmit for Oa {
    fn runtime() -> Runtime {
        // The operator-algebra runtime: 2x2 row-major real matrices, as on-demand
        // fragments. Entirely different from Knot's runtime, injected through the
        // same mechanism (1.6). A program that only uses `X * Z` and `show` pulls
        // in just {type, make, X, Z, mul, show} — not add/sub/scale/comm/I.
        Runtime::new()
            .with(RuntimeFragment {
                name: "type",
                includes: &[],
                deps: &[],
                code: "typedef struct { double m[4]; } oa_op;   /* 2x2 row-major: [m0 m1; m2 m3] */",
            })
            .with(RuntimeFragment {
                name: "make",
                includes: &[],
                deps: &["type"],
                code: "static oa_op oa_make(double a, double b, double c, double d) {\n    oa_op r; r.m[0] = a; r.m[1] = b; r.m[2] = c; r.m[3] = d; return r;\n}",
            })
            .with(RuntimeFragment { name: "X", includes: &[], deps: &["make"], code: "static oa_op oa_X(void) { return oa_make(0, 1, 1, 0); }" })
            .with(RuntimeFragment { name: "Z", includes: &[], deps: &["make"], code: "static oa_op oa_Z(void) { return oa_make(1, 0, 0, -1); }" })
            .with(RuntimeFragment { name: "I", includes: &[], deps: &["make"], code: "static oa_op oa_I(void) { return oa_make(1, 0, 0, 1); }" })
            .with(RuntimeFragment { name: "add", includes: &[], deps: &["make"], code: "static oa_op oa_add(oa_op a, oa_op b) { return oa_make(a.m[0]+b.m[0], a.m[1]+b.m[1], a.m[2]+b.m[2], a.m[3]+b.m[3]); }" })
            .with(RuntimeFragment { name: "sub", includes: &[], deps: &["make"], code: "static oa_op oa_sub(oa_op a, oa_op b) { return oa_make(a.m[0]-b.m[0], a.m[1]-b.m[1], a.m[2]-b.m[2], a.m[3]-b.m[3]); }" })
            .with(RuntimeFragment { name: "scale", includes: &[], deps: &["make"], code: "static oa_op oa_scale(double s, oa_op a) { return oa_make(s*a.m[0], s*a.m[1], s*a.m[2], s*a.m[3]); }" })
            .with(RuntimeFragment {
                name: "mul",
                includes: &[],
                deps: &["make"],
                code: "static oa_op oa_mul(oa_op a, oa_op b) {\n    return oa_make(\n        a.m[0]*b.m[0] + a.m[1]*b.m[2], a.m[0]*b.m[1] + a.m[1]*b.m[3],\n        a.m[2]*b.m[0] + a.m[3]*b.m[2], a.m[2]*b.m[1] + a.m[3]*b.m[3]);\n}",
            })
            .with(RuntimeFragment { name: "comm", includes: &[], deps: &["mul", "sub"], code: "static oa_op oa_comm(oa_op a, oa_op b) { return oa_sub(oa_mul(a, b), oa_mul(b, a)); }" })
            .with(RuntimeFragment {
                name: "show",
                includes: &["#include <stdio.h>"],
                deps: &["type"],
                code: "static void oa_show(oa_op a) { printf(\"[[%g, %g], [%g, %g]]\\n\", a.m[0], a.m[1], a.m[2], a.m[3]); }",
            })
    }

    fn emit_item(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CItem {
        match &node.kind {
            Oa::Program => CItem::Main {
                locals: Vec::new(),
                body: node.children.iter().map(|&c| cx.stmt(c)).collect(),
            },
            _ => unreachable!("non-program node as root: {:?}", node.kind),
        }
    }

    fn emit_stmt(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CStmt {
        let id = node.id;
        match &node.kind {
            Oa::Let { name } => {
                CStmt::decl(id, CType::Named("oa_op".to_string()), name.clone(), cx.expr(node.children[0]))
            }
            Oa::Show => {
                cx.need("show");
                let arg = cx.expr(node.children[0]);
                CStmt::expr(id, CExpr::call(id, "oa_show", vec![arg]))
            }
            _ => unreachable!("non-statement node in statement position: {:?}", node.kind),
        }
    }

    fn emit_expr(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CExpr {
        let id = node.id;
        match &node.kind {
            Oa::Const(b) => {
                cx.need(b.frag());
                CExpr::call(id, b.runtime(), vec![])
            }
            Oa::Var { name } => CExpr::var(id, name.clone()),
            Oa::Bin(op) => {
                cx.need(op.frag());
                let lhs = cx.expr(node.children[0]);
                let rhs = cx.expr(node.children[1]);
                CExpr::call(id, op.runtime(), vec![lhs, rhs])
            }
            Oa::Comm => {
                cx.need("comm");
                let a = cx.expr(node.children[0]);
                let b = cx.expr(node.children[1]);
                CExpr::call(id, "oa_comm", vec![a, b])
            }
            Oa::Scale { s } => {
                cx.need("scale");
                let m = cx.expr(node.children[0]);
                CExpr::call(id, "oa_scale", vec![CExpr::lit(id, format!("{:?}", s)), m])
            }
            _ => unreachable!("non-expression node in expression position: {:?}", node.kind),
        }
    }
}
