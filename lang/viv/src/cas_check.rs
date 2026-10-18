// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv check --phrases`: every phrase with a closed form becomes a check
//! against the computer algebra system (lib/cas).
//!
//! A phrase asks for a quantity: `integrate x^2 from 1 to 2` is the integral,
//! and Gauss-Legendre is how the program gets it. This module hands each
//! phrase's expression to the CAS, which is a Viv library: it writes a Viv
//! program that `#! use`s `lib/cas/phrases`, evaluates each phrase again, and
//! calls the CAS's check for its kind (`cas_check_integral`, `_root`,
//! `_deriv`, `_value`), then runs that program in the interpreter and reads
//! the `__cas__` lines it prints. The program is an ordinary Viv program
//! (`--emit` writes it), so the same checks run as C and as wasm.
//!
//! The compiler's part is the translation. A phrase's holes are Viv
//! expressions; the CAS reads Viv text of numbers, names, `+ - * / ^` and the
//! elementary functions. A name a hole reads, other than the phrase's own
//! variable, must be a constant the program writes at its top level once
//! (`c = 2.0`, `M = [[4.0, 1.0], [1.0, 3.0]]`), and its value goes in. A
//! hole that reads a run-time value, a parameter, an index or a def the CAS
//! does not know is refused with that reason; the refusal is reported, never
//! counted as a pass.

use std::collections::{HashMap, HashSet};

use knotkit::span::SourceId;

use crate::ast::{BinOp, Expr, Item, Program, Stmt};
use crate::phrases::{HoleText, PhraseUse, TABLE};

/// The library the generated program uses, compiled in, so the check runs
/// wherever the compiler does (a file outside the repository, the browser).
pub const LIB: &[(&str, &str)] = &[
    ("lib/cas/rational", include_str!("../../../lib/cas/rational.viv")),
    ("lib/cas/store", include_str!("../../../lib/cas/store.viv")),
    ("lib/cas/parse", include_str!("../../../lib/cas/parse.viv")),
    ("lib/cas/show", include_str!("../../../lib/cas/show.viv")),
    ("lib/cas/derivs", include_str!("../../../lib/cas/derivs.viv")),
    ("lib/cas/eval", include_str!("../../../lib/cas/eval.viv")),
    ("lib/cas/calculus", include_str!("../../../lib/cas/calculus.viv")),
    ("lib/cas/phrases", include_str!("../../../lib/cas/phrases.viv")),
    ("lib/cas/cas", include_str!("../../../lib/cas/cas.viv")),
];

/// The stack the generated program runs on: what `viv run` gives a program.
const STACK: usize = 1 << 30;

/// A module of [`LIB`] by its `#! use` name.
pub fn lib_module(name: &str) -> Option<&'static str> {
    // `/lib/cas/cas`: an import's candidate under the repository root (crate::imports)
    let name = name.strip_prefix('/').unwrap_or(name);
    LIB.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// The function heads the CAS knows (lib/cas/store `CAS_HEADS`, `sqrt`, `ln`).
const HEADS: &[&str] = &["exp", "log", "ln", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "erf", "abs"];

/// What the CAS holds a phrase to.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Integral,
    Root,
    Derivative,
    Value,
}

/// One phrase of the program, as the check sees it.
#[derive(Clone, Debug)]
pub struct Check {
    pub id: usize,
    /// the phrase's line in the program's own file (1-based)
    pub line: usize,
    /// the phrase as written
    pub phrase: String,
    pub kind: Option<Kind>,
    /// the Viv statement of the generated program that checks it
    pub call: String,
    /// why the phrase is not handed to the CAS
    pub refused: Option<String>,
    /// BLAKE3 over what the check reads (the CAS's input and the method), 8 hex digits
    pub stamp: String,
    /// the math the phrase computes, by tex.rs
    pub math: String,
    /// the phrase's variable, for an integral, a root or a derivative
    pub var: String,
}

/// One `__cas__` line of the generated program's output.
#[derive(Clone, Debug, Default)]
pub struct Record {
    pub fields: HashMap<String, String>,
}

impl Record {
    pub fn get(&self, k: &str) -> &str {
        self.fields.get(k).map(String::as_str).unwrap_or("")
    }
}

/// A program's top-level constants: a name assigned once, at the top level,
/// outside any block, never written into, with its value's expression.
fn constants(prog: &Program) -> HashMap<String, Expr> {
    let mut once: HashMap<String, (usize, Option<Expr>)> = HashMap::new();
    let mut written: HashSet<String> = HashSet::new();
    fn nested(ss: &[Stmt], out: &mut HashSet<String>) {
        for s in ss {
            match s {
                Stmt::Assign { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::SetIndex { base, .. } => {
                    if let Expr::Var { name, .. } = &**base {
                        out.insert(name.clone());
                    }
                }
                Stmt::SetField { base, .. } => {
                    if let Some((name, _)) = crate::lower::place_root(base) {
                        out.insert(name.to_string());
                    }
                }
                Stmt::For { var, body, .. } => {
                    out.insert(var.clone());
                    nested(body, out);
                }
                Stmt::If { then_body, else_body, .. } => {
                    nested(then_body, out);
                    nested(else_body, out);
                }
                Stmt::While { body, .. } | Stmt::Test { body, .. } => nested(body, out),
                _ => {}
            }
        }
    }
    for it in &prog.items {
        match it {
            Item::Stmt(Stmt::Assign { name, value, .. }) => {
                let e = once.entry(name.clone()).or_insert((0, None));
                e.0 += 1;
                e.1 = Some(value.clone());
            }
            Item::Stmt(s) => nested(std::slice::from_ref(s), &mut written),
            Item::Def(d) => {
                // a def writes into a module's array through `v[i] = ...`; a
                // plain assignment in a def makes a local
                fn set_into(ss: &[Stmt], out: &mut HashSet<String>) {
                    for s in ss {
                        match s {
                            Stmt::SetIndex { base, .. } => {
                                if let Expr::Var { name, .. } = &**base {
                                    out.insert(name.clone());
                                }
                            }
                            Stmt::For { body, .. } | Stmt::While { body, .. } | Stmt::Test { body, .. } => set_into(body, out),
                            Stmt::If { then_body, else_body, .. } => {
                                set_into(then_body, out);
                                set_into(else_body, out);
                            }
                            _ => {}
                        }
                    }
                }
                let mut inner = HashSet::new();
                set_into(&d.body, &mut inner);
                // the def's own variables are not the module's
                fn assigned_in(ss: &[Stmt], out: &mut HashSet<String>) {
                    for s in ss {
                        match s {
                            Stmt::Assign { name, .. } => {
                                out.insert(name.clone());
                            }
                            Stmt::For { var, body, .. } => {
                                out.insert(var.clone());
                                assigned_in(body, out);
                            }
                            Stmt::While { body, .. } | Stmt::Test { body, .. } => assigned_in(body, out),
                            Stmt::If { then_body, else_body, .. } => {
                                assigned_in(then_body, out);
                                assigned_in(else_body, out);
                            }
                            _ => {}
                        }
                    }
                }
                let mut own = HashSet::new();
                assigned_in(&d.body, &mut own);
                for n in inner {
                    if !d.params.iter().any(|p| p.name == n) && !own.contains(&n) {
                        written.insert(n);
                    }
                }
            }
            _ => {}
        }
    }
    let all: HashMap<String, Expr> = once.into_iter().filter(|(n, (k, _))| *k == 1 && !written.contains(n)).filter_map(|(n, (_, e))| e.map(|e| (n, e))).collect();
    // a constant is a number, or an array of numbers, written with numbers and other constants
    fn numeric(e: &Expr, all: &HashMap<String, Expr>) -> bool {
        match e {
            Expr::ArrLit { elems, .. } => elems.iter().all(|x| numeric(x, all)),
            _ => value_of(e, all, 0).is_some(),
        }
    }
    all.iter().filter(|(_, e)| numeric(e, &all)).map(|(n, e)| (n.clone(), e.clone())).collect()
}

/// The value of a constant expression in doubles, when it is one.
fn value_of(e: &Expr, consts: &HashMap<String, Expr>, depth: usize) -> Option<f64> {
    if depth > 32 {
        return None;
    }
    match e {
        Expr::Num { value, .. } => Some(*value),
        Expr::Var { name, .. } => consts.get(name).and_then(|c| value_of(c, consts, depth + 1)),
        Expr::Neg { operand, .. } => value_of(operand, consts, depth).map(|v| -v),
        Expr::Binary { op, lhs, rhs, .. } => {
            let (a, b) = (value_of(lhs, consts, depth)?, value_of(rhs, consts, depth)?);
            Some(match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                _ => return None,
            })
        }
        Expr::Call { callee, args, .. } if args.len() == 1 => {
            let a = value_of(&args[0], consts, depth)?;
            Some(match callee.as_str() {
                "acos" => a.acos(),
                "atan" => a.atan(),
                "sqrt" => a.sqrt(),
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Turns Viv expressions into the CAS's text.
struct Translator<'a> {
    consts: &'a HashMap<String, Expr>,
    /// the phrase's own variable
    var: Option<String>,
    /// the parameters and locals of the def the phrase is in
    locals: HashSet<String>,
    /// the def the phrase is in
    def: Option<String>,
}

impl Translator<'_> {
    /// The CAS text of a scalar expression, or why it has none.
    fn scalar(&self, e: &Expr, depth: usize) -> Result<String, String> {
        if depth > 32 {
            return Err("its constants refer to each other too deeply".into());
        }
        match e {
            // the shortest decimal that is this double: the CAS reads it exactly
            Expr::Num { value, .. } => Ok(format!("{value:?}").trim_end_matches(".0").to_string()),
            Expr::Var { name, .. } => {
                if self.var.as_deref() == Some(name.as_str()) {
                    return Ok(name.clone());
                }
                if self.locals.contains(name) {
                    let d = self.def.clone().unwrap_or_default();
                    return Err(format!("it reads `{name}`, a variable of `{d}`, whose value the CAS cannot know"));
                }
                let Some(c) = self.consts.get(name) else {
                    return Err(format!("it reads `{name}`, which the program computes at run time or assigns more than once"));
                };
                if value_of(c, self.consts, depth) == Some(std::f64::consts::PI) {
                    return Ok("pi".into());
                }
                self.scalar(c, depth + 1).map(|t| format!("({t})")).map_err(|why| format!("it reads `{name}`, and {why}"))
            }
            Expr::Neg { operand, .. } => Ok(format!("(-{})", self.scalar(operand, depth)?)),
            Expr::Binary { op, lhs, rhs, .. } => {
                let o = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    other => return Err(format!("it uses `{}`, which the CAS does not have", binop_text(*other))),
                };
                Ok(format!("({} {o} {})", self.scalar(lhs, depth)?, self.scalar(rhs, depth)?))
            }
            Expr::Call { callee, args, .. } if callee == "pow" && args.len() == 2 => Ok(format!("({})^({})", self.scalar(&args[0], depth)?, self.scalar(&args[1], depth)?)),
            Expr::Call { callee, args, .. } if args.len() == 1 && (HEADS.contains(&callee.as_str()) || callee == "fabs") => {
                let h = if callee == "fabs" { "abs" } else { callee.as_str() };
                Ok(format!("{h}({})", self.scalar(&args[0], depth)?))
            }
            Expr::Call { callee, .. } => Err(format!("it calls `{callee}`, which the CAS does not know")),
            Expr::Index { .. } => Err("it reads an element of an array, whose value the CAS cannot know".into()),
            Expr::ArrLit { .. } => Err("it is an array, and this check holds numbers".into()),
            _ => Err("it is not a number, a name or arithmetic".into()),
        }
    }

    /// The entries of a vector or a matrix written in the program, as CAS
    /// text, and the number of columns (0 for a vector).
    fn array(&self, e: &Expr, depth: usize) -> Result<(Vec<Vec<String>>, usize), String> {
        match e {
            Expr::Var { name, .. } => {
                if self.locals.contains(name) {
                    return Err(format!("it reads `{name}`, a variable of `{}`", self.def.clone().unwrap_or_default()));
                }
                let c = self.consts.get(name).ok_or_else(|| format!("it reads `{name}`, which is not an array the program writes out at its top level"))?;
                self.array(c, depth + 1)
            }
            Expr::ArrLit { elems, .. } => {
                if elems.iter().all(|r| matches!(r, Expr::ArrLit { .. })) && !elems.is_empty() {
                    let mut rows = Vec::new();
                    let mut cols = 0;
                    for r in elems {
                        let Expr::ArrLit { elems: es, .. } = r else { unreachable!() };
                        cols = es.len();
                        rows.push(es.iter().map(|x| self.scalar(x, depth)).collect::<Result<Vec<_>, _>>()?);
                    }
                    if rows.iter().any(|r| r.len() != cols) {
                        return Err("its rows have different lengths".into());
                    }
                    Ok((rows, cols))
                } else {
                    Ok((vec![elems.iter().map(|x| self.scalar(x, depth)).collect::<Result<Vec<_>, _>>()?], 0))
                }
            }
            _ => Err("it is not an array written out in the program".into()),
        }
    }
}

/// A Viv expression as the CAS's text, every name kept as a symbol (the
/// symbolic-derivative phrase passes the names' values at run time); or why
/// the CAS cannot read it.
pub fn hole_text(e: &Expr) -> Result<String, String> {
    let names: HashMap<String, Expr> = HashMap::new();
    let mut free = HashSet::new();
    free.extend(crate::phrases::free_names(e));
    let tr = Translator { consts: &names, var: None, locals: HashSet::new(), def: None };
    tr.symbolic(e, &free)
}

impl Translator<'_> {
    /// `scalar`, with the names in `free` read as symbols.
    fn symbolic(&self, e: &Expr, free: &HashSet<String>) -> Result<String, String> {
        match e {
            Expr::Var { name, .. } if free.contains(name) => Ok(name.clone()),
            Expr::Neg { operand, .. } => Ok(format!("(-{})", self.symbolic(operand, free)?)),
            Expr::Binary { op, lhs, rhs, .. } if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) => {
                let o = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    _ => "/",
                };
                Ok(format!("({} {o} {})", self.symbolic(lhs, free)?, self.symbolic(rhs, free)?))
            }
            Expr::Call { callee, args, .. } if callee == "pow" && args.len() == 2 => Ok(format!("({})^({})", self.symbolic(&args[0], free)?, self.symbolic(&args[1], free)?)),
            Expr::Call { callee, args, .. } if args.len() == 1 && (HEADS.contains(&callee.as_str()) || callee == "fabs") => {
                let h = if callee == "fabs" { "abs" } else { callee.as_str() };
                Ok(format!("{h}({})", self.symbolic(&args[0], free)?))
            }
            other => self.scalar(other, 0),
        }
    }
}

fn binop_text(op: BinOp) -> &'static str {
    match op {
        BinOp::Matmul => "@",
        BinOp::Mod => "%",
        BinOp::FloorDiv => "//",
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::EqEq | BinOp::Ne => "a comparison",
        BinOp::And | BinOp::Or => "a logical operator",
        _ => "this operator",
    }
}

/// A string as a Viv string literal.
fn lit(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The callee a pattern's lowering is (`q_gauss` for `q_gauss({f}, {a}, {b})`).
fn callee_of(lowers: &str) -> &str {
    let c = lowers.split('(').next().unwrap_or("").trim();
    // `viv__every_integrate`: the contract run by every method
    c.strip_prefix(crate::contracts::EVERY_DEF).unwrap_or(c)
}

/// The checks of a program: each phrase in the program's own file, what the
/// CAS is handed for it or why it is not, and the Viv program that runs them.
pub fn plan(src: &str, resolve: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<(String, Vec<Check>), String> {
    let exp = crate::expand_uses_mapped(src, resolve, &mut |lang: &str, name: &str, _t: &str| Err(format!("`#! use {name}`: it is written in `{lang}`; the phrase check reads Viv")))?;
    let text = exp.text;
    let prog = crate::parse_keeping_rungs(&text, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    let consts = constants(&prog);
    let span_text = |sp: knotkit::span::Span| text.get(sp.start as usize..sp.end as usize).unwrap_or("").to_string();
    // the line of the program's own file a byte of the expansion is on
    let line_of = |at: u32| -> Option<usize> {
        let l = text[..(at as usize).min(text.len())].matches('\n').count();
        if exp.lines.is_empty() {
            return Some(l + 1);
        }
        match exp.lines.get(l).copied().flatten() {
            Some((0, k)) => Some(k + 1),
            _ => None,
        }
    };
    // the defs, for the phrase's enclosing one
    let defs: Vec<&crate::ast::Def> = prog.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d) } else { None }).collect();
    // the constants the generated program needs, in the program's order
    // (the program's own: a module's constants come with the module's `#! use`)
    let mut const_order: Vec<String> = Vec::new();
    for it in &prog.items {
        if let Item::Stmt(Stmt::Assign { name, span, .. }) = it {
            if consts.contains_key(name) && !const_order.contains(name) && line_of(span.start).is_some() {
                const_order.push(name.clone());
            }
        }
    }
    // the modules the program uses, which the generated program uses too
    // (its import statements, lines and all, but lib/cas's, which come below)
    let mut uses: Vec<String> = Vec::new();
    if let Ok(stmts) = crate::imports::scan(src) {
        let lines: Vec<&str> = src.lines().collect();
        for s in stmts {
            if s.items.iter().any(|i| i.path().first().is_some_and(|p| p == "cas") || i.path().starts_with(&["lib".to_string(), "cas".to_string()])) {
                continue;
            }
            uses.extend(lines[s.line..s.line + s.nlines].iter().map(|l| l.to_string()));
        }
    }
    let mut checks = Vec::new();
    for (id, u) in prog.phrases.iter().enumerate() {
        let Some(line) = line_of(u.span.start) else { continue };
        let phrase = span_text(u.span);
        let enclosing = defs.iter().find(|d| d.span.start <= u.span.start && u.span.end <= d.span.end);
        let mut locals: HashSet<String> = HashSet::new();
        if let Some(d) = enclosing {
            locals.extend(d.params.iter().map(|p| p.name.clone()));
            let mut assigned = Vec::new();
            fn walk(ss: &[Stmt], out: &mut Vec<String>) {
                for s in ss {
                    match s {
                        Stmt::Assign { name, .. } => out.push(name.clone()),
                        Stmt::For { var, body, .. } => {
                            out.push(var.clone());
                            walk(body, out);
                        }
                        Stmt::If { then_body, else_body, .. } => {
                            walk(then_body, out);
                            walk(else_body, out);
                        }
                        Stmt::While { body, .. } | Stmt::Test { body, .. } => walk(body, out),
                        _ => {}
                    }
                }
            }
            walk(&d.body, &mut assigned);
            locals.extend(assigned);
        }
        let mut c = Check { id, line, phrase: phrase.clone(), kind: None, call: String::new(), refused: None, stamp: String::new(), math: String::new(), var: String::new() };
        match build(id, line, u, &consts, locals, enclosing.map(|d| d.name.clone()), &span_text) {
            Ok((kind, call, key)) => {
                c.kind = Some(kind);
                c.call = call;
                c.var = key.split('|').nth(2).unwrap_or("").to_string();
                let mut h = knotkit::ir::Hasher::new();
                h.update(key.as_bytes());
                c.stamp = h.finalize().to_hex().chars().take(8).collect();
            }
            Err(why) => c.refused = Some(why),
        }
        checks.push(c);
    }
    // the generated program
    let mut out = String::new();
    for u in &uses {
        out.push_str(u);
        out.push('\n');
    }
    out.push('\n');
    let mut needed: Vec<&String> = Vec::new();
    for n in &const_order {
        needed.push(n);
    }
    for n in needed {
        if let Some(e) = consts.get(n) {
            out.push_str(&format!("{n} = {}\n", span_text(e.span())));
        }
    }
    out.push('\n');
    for c in &mut checks {
        match &c.refused {
            Some(why) => out.push_str(&format!("cas_check_refused({}, {}, {}, {})\n", c.id, c.line, lit(&c.phrase), lit(why))),
            None => {
                out.push_str(&format!("cas_got_{} = {}\n", c.id, c.phrase));
                out.push_str(&c.call);
                out.push('\n');
            }
        }
    }
    out.push_str("cas_check_summary()\n\n");
    for c in &checks {
        match &c.refused {
            None => {
                let name = format!("line {}: {} agrees with the CAS, or the CAS has no closed form", c.line, c.phrase);
                out.push_str(&format!("test {} {{\n    assert_true(cas_check_holds({}))\n}}\n", lit(&name), c.id));
            }
            Some(_) => {
                let name = format!("line {}: {} is not handed to the CAS", c.line, c.phrase);
                out.push_str(&format!("test {} {{\n    assert_near(CAS_CHECK[{}], CHECK_REFUSED, 0)\n}}\n", lit(&name), c.id));
            }
        }
    }
    // lib/cas's names the checks read, each from its module; lib/cas/phrases
    // always, for its phrases
    let cas: Vec<(&str, &str)> = LIB.to_vec();
    let head = format!(
        "# Written by `viv check --phrases`: each phrase of the program held to the\n# closed form lib/cas finds for it (lang/viv/src/cas_check.rs).\n{}",
        crate::imports::import_lines(&out, &cas, &["lib/cas/phrases"])
    );
    out.insert_str(0, &head);
    // the math each phrase computes, by tex.rs, from the phrase read on its own;
    // the symbolic-derivative phrase needs lib/cas's def to be declared, and
    // this program is only read, never run, so a declaration of it does
    let mut prelude = String::from("def cas_deriv_at(t: Str, v: Str, a, n: Str, w: Arr) {\n    return a\n}\n");
    for n in &const_order {
        if let Some(e) = consts.get(n) {
            prelude.push_str(&format!("{n} = {}\n", span_text(e.span())));
        }
    }
    for c in &mut checks {
        c.math = math_of(&prelude, &c.phrase).unwrap_or_default();
    }
    Ok((out, checks))
}

/// tex.rs's LaTeX of a phrase, read as the value of an assignment after the constants.
fn math_of(prelude: &str, phrase: &str) -> Option<String> {
    let src = format!("{prelude}viv__math = {phrase}\n");
    let prog = crate::parse_keeping_rungs(&src, SourceId(0)).ok()?;
    let last = prog.items.iter().rev().find_map(|i| if let Item::Stmt(Stmt::Assign { name, value, .. }) = i { (name == "viv__math").then_some(value) } else { None })?;
    // `pi` is the constant here, as it is to the CAS
    let mut syms = crate::tex::Symbols::new();
    syms.insert("pi".into(), "\\pi".into());
    Some(crate::tex::expr(last, &syms))
}

/// The check call for one phrase: its kind, the Viv statement, and the text
/// its stamp hashes.
fn build(id: usize, line: usize, u: &PhraseUse, consts: &HashMap<String, Expr>, locals: HashSet<String>, def: Option<String>, span_text: &dyn Fn(knotkit::span::Span) -> String) -> Result<(Kind, String, String), String> {
    let Some(pi) = u.pattern else {
        return Err("it is the program's own phrase; the check reads the phrases of Viv's table".into());
    };
    let p = &TABLE[pi];
    let callee = callee_of(p.lowers);
    // a hole's expression, parsed from its text
    let hole = |name: &str| -> Result<(Expr, Option<String>), String> {
        let h = u.holes.iter().find(|h| h.name == name).ok_or_else(|| format!("the phrase has no hole `{name}`"))?;
        let t = match h.text {
            HoleText::Plain(sp) => span_text(sp),
            HoleText::Equation(l, r) => format!("({}) - ({})", span_text(l), span_text(r)),
        };
        let e = crate::derivs::parse_expr(&t).map_err(|e| format!("its part `{t}` does not read: {e}"))?;
        Ok((e, h.var.clone()))
    };
    let mut tr = Translator { consts, var: None, locals, def };
    let got = format!("cas_got_{id}");
    let phrase = lit(&span_text(u.span));
    match callee {
        "q_gauss" | "q_adaptive" | "q_romberg" | "integrate" | "simpson" | "trapezoid" => {
            let (f, var) = hole("f")?;
            tr.var = var.clone();
            let ft = tr.scalar(&f, 0).map_err(|w| format!("the integrand: {w}"))?;
            tr.var = None;
            let a = tr.scalar(&hole("a")?.0, 0).map_err(|w| format!("the lower limit: {w}"))?;
            let b = tr.scalar(&hole("b")?.0, 0).map_err(|w| format!("the upper limit: {w}"))?;
            let n = if callee == "simpson" || callee == "trapezoid" {
                let (ne, _) = hole("n")?;
                let v = value_of(&ne, consts, 0).ok_or("the number of intervals is not a constant")?;
                format!("{v}")
            } else {
                "0".into()
            };
            let var = var.unwrap_or_default();
            let key = format!("integral|{ft}|{var}|{a}|{b}|{callee}|{n}");
            Ok((Kind::Integral, format!("cas_check_integral({id}, {line}, {phrase}, {}, {}, {}, {}, {got}, {}, {n})", lit(&ft), lit(&var), lit(&a), lit(&b), lit(callee)), key))
        }
        "r_bisect" | "r_illinois" | "r_safe" | "find_root" | "r_newton" => {
            let (f, var) = hole("f")?;
            tr.var = var.clone();
            let ft = tr.scalar(&f, 0).map_err(|w| format!("the function: {w}"))?;
            tr.var = None;
            let (a, b) = if callee == "r_newton" {
                (tr.scalar(&hole("x0")?.0, 0).map_err(|w| format!("the starting point: {w}"))?, String::new())
            } else {
                (tr.scalar(&hole("a")?.0, 0).map_err(|w| format!("the bracket: {w}"))?, tr.scalar(&hole("b")?.0, 0).map_err(|w| format!("the bracket: {w}"))?)
            };
            let var = var.unwrap_or_default();
            let key = format!("root|{ft}|{var}|{a}|{b}|{callee}");
            Ok((Kind::Root, format!("cas_check_root({id}, {line}, {phrase}, {}, {}, {}, {}, {got}, {})", lit(&ft), lit(&var), lit(&a), lit(&b), lit(callee)), key))
        }
        "deriv" | "cas_deriv_at" => {
            let (f, var) = hole("f")?;
            tr.var = var.clone();
            let ft = tr.scalar(&f, 0).map_err(|w| format!("the function: {w}"))?;
            tr.var = None;
            let at = tr.scalar(&hole("x")?.0, 0).map_err(|w| format!("the point: {w}"))?;
            let var = var.unwrap_or_default();
            let key = format!("deriv|{ft}|{var}|{at}|{callee}");
            Ok((Kind::Derivative, format!("cas_check_deriv({id}, {line}, {phrase}, {}, {}, {}, {got}, {})", lit(&ft), lit(&var), lit(&at), lit(callee)), key))
        }
        _ => {
            let (text, how) = closed_form(p.words, p.lowers, &hole, &tr)?;
            let key = format!("value|{text}|{}", p.words);
            Ok((Kind::Value, format!("cas_check_value({id}, {line}, {phrase}, {}, {got}, {}, {})", lit(&text), lit(callee), lit(&how)), key))
        }
    }
}

/// The closed form of a phrase of an aggregate, a norm or a trace of arrays
/// the program writes out, as CAS text, and the words on what it is.
fn closed_form(words: &str, lowers: &str, hole: &dyn Fn(&str) -> Result<(Expr, Option<String>), String>, tr: &Translator) -> Result<(String, String), String> {
    let sum = |v: &[String]| v.iter().map(|t| format!("({t})")).collect::<Vec<_>>().join(" + ");
    let all = |rows: &[Vec<String>]| rows.iter().flatten().cloned().collect::<Vec<String>>();
    let arr = |name: &str| -> Result<(Vec<Vec<String>>, usize), String> { tr.array(&hole(name)?.0, 0) };
    match words {
        "the sum of {v}" => {
            let (rows, _) = arr("v")?;
            Ok((sum(&all(&rows)), "the sum of the entries".into()))
        }
        "the product of {v}" => {
            let (rows, _) = arr("v")?;
            Ok((all(&rows).iter().map(|t| format!("({t})")).collect::<Vec<_>>().join(" * "), "the product of the entries".into()))
        }
        "the mean of {v}" | "the average of {v}" => {
            let (rows, _) = arr("v")?;
            let es = all(&rows);
            Ok((format!("({}) / {}", sum(&es), es.len()), "the sum of the entries over their number".into()))
        }
        "the trace of {M}" => {
            let (rows, cols) = arr("M")?;
            if cols != rows.len() {
                return Err("the trace needs a square matrix written out in the program".into());
            }
            Ok((sum(&(0..cols).map(|i| rows[i][i].clone()).collect::<Vec<_>>()), "the sum of the diagonal".into()))
        }
        "the sum of the off-diagonal entries of {M}" | "sum the off-diagonal entries of {M}" => {
            let (rows, cols) = arr("M")?;
            if cols != rows.len() {
                return Err("the off-diagonal entries need a square matrix written out in the program".into());
            }
            let es: Vec<String> = (0..cols).flat_map(|i| (0..cols).filter(move |j| *j != i).map(move |j| (i, j))).map(|(i, j)| rows[i][j].clone()).collect();
            Ok((sum(&es), "the sum of the entries off the diagonal".into()))
        }
        "the L2 norm of {v}" | "the norm of {v}" | "the magnitude of {v}" | "the Frobenius norm of {M}" => {
            let h = if words.contains("{M}") { "M" } else { "v" };
            let (rows, _) = arr(h)?;
            let sq: Vec<String> = all(&rows).iter().map(|t| format!("({t})^2")).collect();
            Ok((format!("sqrt({})", sq.join(" + ")), "the square root of the sum of the squares".into()))
        }
        "the dot product of {u} and {v}" | "the inner product of {u} and {v}" => {
            let (a, ca) = arr("u")?;
            let (b, cb) = arr("v")?;
            let (a, b) = (all(&a), all(&b));
            if ca != 0 || cb != 0 || a.len() != b.len() {
                return Err("the dot product needs two vectors of one length written out in the program".into());
            }
            let ps: Vec<String> = a.iter().zip(&b).map(|(x, y)| format!("({x}) * ({y})")).collect();
            Ok((ps.join(" + "), "the sum of the products of the entries".into()))
        }
        "the distance from {u} to {v}" | "the distance between {u} and {v}" => {
            let (a, _) = arr("u")?;
            let (b, _) = arr("v")?;
            let (a, b) = (all(&a), all(&b));
            if a.len() != b.len() {
                return Err("the distance needs two vectors of one length".into());
            }
            let sq: Vec<String> = a.iter().zip(&b).map(|(x, y)| format!("(({x}) - ({y}))^2")).collect();
            Ok((format!("sqrt({})", sq.join(" + ")), "the norm of the difference".into()))
        }
        _ => {
            let c = callee_of(lowers);
            if HEADS.contains(&c) && words.matches('{').count() == 1 {
                let (x, _) = hole("x")?;
                let t = tr.scalar(&x, 0)?;
                return Ok((format!("{c}({t})"), format!("`{c}` of a constant")));
            }
            Err(format!("the CAS has no closed form for `{}` beyond the call itself", crate::phrases::display(words)))
        }
    }
}

/// The generated program and the checks (`plan`).
pub fn generate(src: &str, resolve: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<(String, Vec<Check>), String> {
    plan(src, resolve)
}

/// A module `#! use` names: lib/cas from the compiler, the rest from `resolve`.
pub fn resolver<'a>(resolve: &'a mut dyn FnMut(&str) -> Result<String, String>) -> impl FnMut(&str) -> Result<String, String> + 'a {
    move |name: &str| match lib_module(name) {
        Some(t) => Ok(t.to_string()),
        None => resolve(name),
    }
}

/// Run the generated program in the interpreter: its output, and its `__cas__` records.
pub fn run(program: &str, resolve: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<(String, Vec<Record>), String> {
    let expanded = crate::expand_uses(program, &mut |name: &str| match lib_module(name) {
        Some(t) => Ok(t.to_string()),
        None => resolve(name),
    })?;
    // the interpreter recurses with the program: on a thread with room for it
    // (the stack `viv run` gives a program)
    let out = std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || crate::interp::run(&expanded))
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "the check program's thread panicked".to_string())??;
    let mut recs = Vec::new();
    let mut shown = String::new();
    for l in out.lines() {
        if let Some(j) = l.strip_prefix("__cas__ ") {
            recs.push(Record { fields: flat_json(j) });
        } else if !l.starts_with("__graph__") && !l.starts_with("__trace__") {
            shown.push_str(l);
            shown.push('\n');
        }
    }
    Ok((shown, recs))
}

/// The fields of a flat JSON object of strings, numbers and nulls (what
/// lib/cas/phrases prints), each as its text.
pub fn flat_json(s: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let b: Vec<char> = s.trim().trim_start_matches('{').trim_end_matches('}').chars().collect();
    let mut i = 0;
    let read_str = |i: &mut usize| -> String {
        let mut out = String::new();
        *i += 1;
        while *i < b.len() && b[*i] != '"' {
            if b[*i] == '\\' && *i + 1 < b.len() {
                *i += 1;
                out.push(match b[*i] {
                    'n' => '\n',
                    c => c,
                });
            } else {
                out.push(b[*i]);
            }
            *i += 1;
        }
        *i += 1;
        out
    };
    while i < b.len() {
        while i < b.len() && b[i] != '"' {
            i += 1;
        }
        if i >= b.len() {
            break;
        }
        let k = read_str(&mut i);
        while i < b.len() && (b[i] == ':' || b[i] == ' ') {
            i += 1;
        }
        let v = if i < b.len() && b[i] == '"' {
            read_str(&mut i)
        } else {
            let st = i;
            while i < b.len() && b[i] != ',' {
                i += 1;
            }
            b[st..i].iter().collect::<String>()
        };
        m.insert(k, v);
        while i < b.len() && b[i] == ',' {
            i += 1;
        }
    }
    m
}

/// The `.cert` lens file of a program's phrase checks: one entry per phrase, stamped with the BLAKE3
/// of what its check reads, so an edited phrase makes its entry stale.
pub fn cert_text(module: &str, checks: &[Check], recs: &[Record]) -> String {
    let mut out = format!("cert {module}\n  The phrases of {module} held to the closed forms of lib/cas. Written by\n  `viv check --phrases --cert`; an entry whose stamp no longer matches its\n  phrase is stale.\n");
    for c in checks {
        let r = recs.iter().find(|r| r.get("id") == c.id.to_string());
        out.push('\n');
        let stamp = if c.stamp.is_empty() { "--------".to_string() } else { c.stamp.clone() };
        out.push_str(&format!("phrase @{stamp}  line {}\n", c.line));
        out.push_str(&format!("  says      {}\n", c.phrase));
        if !c.math.is_empty() {
            out.push_str(&format!("  math      {}\n", c.math));
        }
        match (r, &c.refused) {
            (_, Some(why)) => out.push_str(&format!("  outcome   refused: {why}\n")),
            (None, None) => out.push_str("  outcome   not run\n"),
            (Some(r), None) => {
                if r.get("outcome") == "refused" {
                    out.push_str(&format!("  outcome   refused: {}\n", r.get("how")));
                } else {
                    out.push_str(&format!("  exact     {}\n  tex       {}\n  how       {}\n  numeric   {} by {}\n  outcome   {}, |difference| {} within {}\n", r.get("exact"), r.get("tex"), r.get("how"), r.get("numeric"), r.get("method"), r.get("outcome"), r.get("difference"), r.get("tolerance")));
                }
                if !r.get("warning").is_empty() {
                    out.push_str(&format!("  warning   {}\n", r.get("warning")));
                }
            }
        }
    }
    out
}

/// The entries of an existing `.cert` text whose stamps the current checks no
/// longer have: the phrase changed, or is gone.
pub fn stale(old: &str, checks: &[Check]) -> Vec<String> {
    let now: HashSet<&str> = checks.iter().map(|c| c.stamp.as_str()).collect();
    old.lines()
        .filter_map(|l| l.strip_prefix("phrase @"))
        .filter_map(|l| l.split_whitespace().next())
        .filter(|s| *s != "--------" && !now.contains(s))
        .map(str::to_string)
        .collect()
}

/// The report `viv check --phrases` prints, and whether every check held.
pub fn report(checks: &[Check], shown: &str, recs: &[Record]) -> (String, bool) {
    let mut out = String::new();
    let mut ok = true;
    for l in shown.lines() {
        if l.starts_with("PASS ") || l.starts_with("FAIL ") {
            continue;
        }
        out.push_str(l);
        out.push('\n');
    }
    for r in recs {
        if r.get("outcome") == "disagrees" {
            ok = false;
        }
    }
    let n_math = checks.iter().filter(|c| !c.math.is_empty() && c.refused.is_none()).count();
    if n_math > 0 {
        out.push_str("\nthe math, with the exact values:\n");
        for c in checks.iter().filter(|c| c.refused.is_none()) {
            if let Some(r) = recs.iter().find(|r| r.get("id") == c.id.to_string()) {
                if r.get("outcome") != "refused" && !c.math.is_empty() {
                    let lhs = c.math.split(" : ").next().unwrap_or(&c.math);
                    // a value the CAS could only evaluate in doubles (past 2^53) is shown as one
                    let rhs = if r.get("tex").is_empty() { format!("\\approx {}", r.get("value")) } else { format!("= {}", r.get("tex")) };
                    let eq = if c.kind == Some(Kind::Root) { format!("{} \\Rightarrow {} {rhs}", c.math, c.var) } else { format!("{lhs} {rhs}") };
                    out.push_str(&format!("  line {:>3}  {eq}\n", c.line));
                }
            }
        }
    }
    (out, ok)
}

/// The checks as JSON, for the explorer: each with its math and the CAS's record.
pub fn json(checks: &[Check], recs: &[Record]) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"));
    let rows: Vec<String> = checks
        .iter()
        .map(|c| {
            let r = recs.iter().find(|r| r.get("id") == c.id.to_string());
            let mut fields = vec![
                format!("\"id\":{}", c.id),
                format!("\"line\":{}", c.line),
                format!("\"phrase\":{}", q(&c.phrase)),
                format!("\"stamp\":{}", q(&c.stamp)),
                format!("\"math\":{}", q(&c.math)),
            ];
            match (r, &c.refused) {
                (_, Some(why)) => fields.push(format!("\"outcome\":\"refused\",\"how\":{}", q(why))),
                (Some(r), None) => {
                    for k in ["outcome", "exact", "tex", "how", "method", "warning"] {
                        fields.push(format!("\"{k}\":{}", q(r.get(k))));
                    }
                    for k in ["value", "numeric", "difference", "tolerance"] {
                        let v = r.get(k);
                        fields.push(format!("\"{k}\":{}", if v.is_empty() { "null" } else { v }));
                    }
                }
                (None, None) => fields.push("\"outcome\":\"not run\"".into()),
            }
            format!("{{{}}}", fields.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}
