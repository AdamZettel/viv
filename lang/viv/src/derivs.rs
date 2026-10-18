// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The one table of local derivatives that the reverse-mode pass
//! (`crate::grad`) and the computer algebra system (`lib/cas`) both read:
//! `lib/cas/rules/derivatives.toml` (decision 10).
//!
//! Each row names a one-argument builtin (`head`) and its `adjoint`, a Viv
//! expression in `g` (the adjoint flowing into the call's value), `u` (the
//! argument) and `y` (the call's value). grad.rs adds the adjoint, with `g`,
//! `u` and `y` put in, into the argument's adjoint ([`adjoint`]). The CAS
//! cannot read a file at compile time and must run in a browser with no file
//! system, so it reads the same rows through `lib/cas/derivs.viv`, which
//! [`viv_module`] writes from this file (`viv cas derivs`); a test requires
//! the committed module to be what this file gives.
//!
//! The file is TOML, read here by a reader for the part of TOML it uses:
//! `[[rule]]` tables of `key = "string"` and `key = [numbers]` lines, and
//! `#` comments. Anything else in it is an error that names the line.

use std::collections::HashMap;
use std::sync::OnceLock;

use knotkit::span::{SourceId, Span};

use crate::ast::{BinOp, Expr, Item, Stmt};

/// The table's text, compiled in.
pub const TOML: &str = include_str!("../../../lib/cas/rules/derivatives.toml");

/// Where the table lives, for messages and the generated module's header.
pub const PATH: &str = "lib/cas/rules/derivatives.toml";

/// One row of the table.
#[derive(Clone, Debug)]
pub struct Rule {
    pub id: String,
    pub head: String,
    pub adjoint: String,
    pub at: Vec<f64>,
}

/// The rows of a table in the TOML subset described above.
pub fn read(text: &str) -> Result<Vec<Rule>, String> {
    let mut rules: Vec<HashMap<String, String>> = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line == "[[rule]]" {
            rules.push(HashMap::new());
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            return Err(format!("{PATH}:{}: expected `[[rule]]` or `key = value`, found `{line}`", n + 1));
        };
        let Some(cur) = rules.last_mut() else {
            return Err(format!("{PATH}:{}: `{}` before the first `[[rule]]`", n + 1, k.trim()));
        };
        cur.insert(k.trim().to_string(), v.trim().to_string());
    }
    let mut out = Vec::new();
    for (i, r) in rules.iter().enumerate() {
        let s = |k: &str| -> Result<String, String> {
            let v = r.get(k).ok_or_else(|| format!("{PATH}: rule {} has no `{k}`", i + 1))?;
            v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).map(str::to_string).ok_or_else(|| format!("{PATH}: rule {}: `{k}` is not a string", i + 1))
        };
        let at = match r.get("at") {
            None => Vec::new(),
            Some(v) => {
                let inner = v.strip_prefix('[').and_then(|x| x.strip_suffix(']')).ok_or_else(|| format!("{PATH}: rule {}: `at` is not a list", i + 1))?;
                inner.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| x.parse::<f64>().map_err(|_| format!("{PATH}: rule {}: `{x}` in `at` is not a number", i + 1))).collect::<Result<Vec<_>, _>>()?
            }
        };
        out.push(Rule { id: s("id")?, head: s("head")?, adjoint: s("adjoint")?, at });
    }
    Ok(out)
}

/// A line without its `#` comment (a `#` inside a string is kept).
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

/// The table's rows, read once.
pub fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| read(TOML).unwrap_or_else(|e| panic!("{e}")))
}

/// The heads the table has a rule for.
pub fn heads() -> Vec<&'static str> {
    rules().iter().map(|r| r.head.as_str()).collect()
}

/// The parsed adjoint of each head, with the constant subexpressions folded
/// (`2 / sqrt(pi)` is the number 1.1283791670955126, so grad's code computes
/// what it computed before the table existed, and a program's own `pi` is
/// never read).
fn parsed() -> &'static HashMap<String, Expr> {
    static P: OnceLock<HashMap<String, Expr>> = OnceLock::new();
    P.get_or_init(|| {
        let mut m = HashMap::new();
        for r in rules() {
            let e = parse_expr(&r.adjoint).unwrap_or_else(|e| panic!("{PATH}: rule `{}`: {e}", r.id));
            m.insert(r.head.clone(), fold(e));
        }
        m
    })
}

/// One Viv expression, read as the value of an assignment.
pub fn parse_expr(text: &str) -> Result<Expr, String> {
    let src = format!("viv__rule = {text}\n");
    let toks = crate::lexer::lex(&src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; "))?;
    let prog = crate::parser::parse(toks, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; "))?;
    match prog.items.as_slice() {
        [Item::Stmt(Stmt::Assign { value, .. })] => Ok(value.clone()),
        _ => Err(format!("`{text}` is not one expression")),
    }
}

/// The value of a subexpression with no name in it but `pi`, if it is one.
fn constant(e: &Expr) -> Option<f64> {
    match e {
        Expr::Num { value, .. } => Some(*value),
        Expr::Var { name, .. } if name == "pi" => Some(std::f64::consts::PI),
        Expr::Neg { operand, .. } => constant(operand).map(|v| -v),
        Expr::Binary { op, lhs, rhs, .. } => {
            let (a, b) = (constant(lhs)?, constant(rhs)?);
            match op {
                BinOp::Add => Some(a + b),
                BinOp::Sub => Some(a - b),
                BinOp::Mul => Some(a * b),
                BinOp::Div => Some(a / b),
                _ => None,
            }
        }
        Expr::Call { callee, args, .. } if callee == "sqrt" && args.len() == 1 => constant(&args[0]).map(f64::sqrt),
        _ => None,
    }
}

/// `e` with each constant subexpression that is not a lone number replaced
/// by its value.
fn fold(e: Expr) -> Expr {
    if !matches!(e, Expr::Num { .. }) {
        if let Some(v) = constant(&e) {
            return Expr::Num { value: v, span: e.span() };
        }
    }
    match e {
        Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(fold(*operand)), span },
        Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op, lhs: Box::new(fold(*lhs)), rhs: Box::new(fold(*rhs)), span },
        Expr::Call { callee, args, span } => Expr::Call { callee, args: args.into_iter().map(fold).collect(), span },
        other => other,
    }
}

/// The expression grad.rs adds into `u`'s adjoint for a call of `head`, with
/// `g`, `u` and `y` put in and every span `sp`; None when the table has no
/// row for `head`.
pub fn adjoint(head: &str, g: Expr, u: Expr, y: Expr, sp: Span) -> Option<Expr> {
    let mut e = parsed().get(head)?.clone();
    crate::phrases::set_spans(&mut e, sp);
    Some(put(e, &g, &u, &y))
}

fn put(e: Expr, g: &Expr, u: &Expr, y: &Expr) -> Expr {
    match e {
        Expr::Var { ref name, .. } if name == "g" => g.clone(),
        Expr::Var { ref name, .. } if name == "u" => u.clone(),
        Expr::Var { ref name, .. } if name == "y" => y.clone(),
        Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(put(*operand, g, u, y)), span },
        Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op, lhs: Box::new(put(*lhs, g, u, y)), rhs: Box::new(put(*rhs, g, u, y)), span },
        Expr::Call { callee, args, span } => Expr::Call { callee, args: args.into_iter().map(|a| put(a, g, u, y)).collect(), span },
        other => other,
    }
}

/// `lib/cas/derivs.viv`: the table as a Viv module the CAS reads, with a test
/// that every adjoint reads.
pub fn viv_module() -> String {
    let rs = rules();
    let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let heads: Vec<String> = rs.iter().map(|r| q(&r.head)).collect();
    let adjs: Vec<String> = rs.iter().map(|r| q(&r.adjoint)).collect();
    let mut at_rule = Vec::new();
    let mut at_x = Vec::new();
    for (i, r) in rs.iter().enumerate() {
        for x in &r.at {
            at_rule.push(format!("{i}"));
            at_x.push(format!("{x:?}"));
        }
    }
    let mut out = String::new();
    out.push_str("###\npost: The derivative table, as the CAS reads it\nagent: claude\n\n");
    out.push_str(&format!(
        "Generated from `{PATH}` by `viv cas derivs`; edit the TOML file and run\n`viv cas derivs > lib/cas/derivs.viv`. grad.rs reads the same file\n(lang/viv/src/derivs.rs), so the reverse-mode gradient and the CAS's `diff`\nuse one derivative for each head.\n\n"
    ));
    out.push_str("`CAS_DERIV_ADJ[i]` is the adjoint of `CAS_DERIV_HEAD[i]` in `g`, `u` and `y`;\nthe CAS reads it with `g = 1`, which is the derivative of `head(u)` with\nrespect to `u`. The test here reads every adjoint; lib/cas/calculus's test\nholds each derivative to a central difference at the table's points.\n\nUsed with `import cas.derivs`.\n###\nfrom cas.parse import cas\nfrom cas.store import cas_isfail\n\n");
    out.push_str(&format!("CAS_DERIV_ID = [{}]\n", rs.iter().map(|r| q(&r.id)).collect::<Vec<_>>().join(", ")));
    out.push_str(&format!("CAS_DERIV_HEAD = [{}]\n", heads.join(", ")));
    out.push_str(&format!("CAS_DERIV_ADJ = [{}]\n", adjs.join(", ")));
    out.push_str("# the points each row is tested at: the row, and x\n");
    out.push_str(&format!("CAS_DERIV_AT_RULE = [{}]\n", at_rule.join(", ")));
    out.push_str(&format!("CAS_DERIV_AT_X = [{}]\n", at_x.join(", ")));
    out.push_str("\ntest \"every adjoint in the table reads as an expression in g, u and y\" {\n    for i to len(CAS_DERIV_ADJ) {\n        assert_true(not cas_isfail(cas(CAS_DERIV_ADJ[i])))\n    }\n}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_reads_and_every_adjoint_parses() {
        assert!(rules().len() >= 14);
        for r in rules() {
            assert!(parsed().contains_key(&r.head), "{}", r.id);
        }
    }

    #[test]
    fn erf_folds_to_the_constant_grad_used_before() {
        let Some(Expr::Binary { rhs, .. }) = parsed().get("erf") else { panic!("erf") };
        let Expr::Binary { lhs, .. } = &**rhs else { panic!("erf's factor") };
        assert!(matches!(**lhs, Expr::Num { value, .. } if value == 1.128_379_167_095_512_6));
    }

    #[test]
    fn a_line_that_is_not_the_subset_is_named() {
        let e = read("[[rule]]\nid = \"x\"\nwhat\n").unwrap_err();
        assert!(e.contains(":3:"), "{e}");
    }
}
