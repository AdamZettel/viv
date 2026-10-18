// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Parser for `weft`: a line-oriented problem file plus a small expression grammar
//! shared by forms and scalar functions. Errors are plain strings (the surface is
//! tiny; rich spans can come later).

use crate::expr::E;

/// How the problem's forms are given: a linear (bilinear + linear) pair, or a
/// nonlinear residual whose Jacobian the compiler derives symbolically.
pub enum Forms {
    Linear { bilinear: E, linear: E },
    Nonlinear { residual: E, newton_max: u32 },
}

/// A parsed weak-form problem.
pub struct Problem {
    pub mesh_n: u32,
    pub coeffs: Vec<(String, E)>,
    pub forms: Forms,
    pub dirichlet: f64,
    pub exact: E,
}

pub fn parse_problem(src: &str) -> Result<Problem, String> {
    let mut mesh_n = None;
    let mut coeffs: Vec<(String, E)> = Vec::new();
    let mut bilinear = None;
    let mut linear = None;
    let mut residual = None;
    let mut newton_max = 25u32;
    let mut exact = None;
    let mut dirichlet = 0.0;

    for raw in src.lines() {
        // Strip `#` comments and surrounding whitespace.
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("coefficient ") {
            let (name, def) = rest.split_once('=').ok_or_else(|| format!("coefficient needs `= expr`: {line}"))?;
            coeffs.push((name.trim().to_string(), parse_expr(def.trim())?));
            continue;
        }
        let (key, value) = line.split_once(':').ok_or_else(|| format!("expected `key: value`: {line}"))?;
        let value = value.trim();
        match key.trim() {
            "mesh" => mesh_n = Some(parse_unit_square(value)?),
            "bilinear" => bilinear = Some(parse_expr(value)?),
            "linear" => linear = Some(parse_expr(value)?),
            "residual" => residual = Some(parse_expr(value)?),
            "newton" => newton_max = value.parse::<u32>().map_err(|_| format!("newton must be an integer: {value}"))?,
            "exact" => exact = Some(parse_expr(value)?),
            "dirichlet" => dirichlet = value.parse::<f64>().map_err(|_| format!("dirichlet must be a number: {value}"))?,
            other => return Err(format!("unknown key `{other}`")),
        }
    }

    let forms = match (residual, bilinear, linear) {
        (Some(residual), None, None) => Forms::Nonlinear { residual, newton_max },
        (None, Some(bilinear), Some(linear)) => Forms::Linear { bilinear, linear },
        (Some(_), _, _) => return Err("use either `residual:` or `bilinear:`/`linear:`, not both".into()),
        _ => return Err("need a `residual:` line, or both `bilinear:` and `linear:`".into()),
    };

    Ok(Problem {
        mesh_n: mesh_n.ok_or("missing `mesh:` line")?,
        coeffs,
        forms,
        dirichlet,
        exact: exact.ok_or("missing `exact:` line")?,
    })
}

fn parse_unit_square(s: &str) -> Result<u32, String> {
    let inner = s
        .strip_prefix("unit_square(")
        .and_then(|r| r.strip_suffix(')'))
        .ok_or_else(|| format!("mesh must be `unit_square(N)`: {s}"))?;
    inner.trim().parse::<u32>().map_err(|_| format!("mesh resolution must be an integer: {inner}"))
}

// ───────────────────────────── expression parser ───────────────────────────

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    LParen,
    RParen,
    Comma,
    Op(char),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit()) {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            let text = &s[start..i];
            out.push(Tok::Num(text.parse::<f64>().map_err(|_| format!("bad number `{text}`"))?));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && ((b[i] as char).is_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(s[start..i].to_string()));
        } else {
            match c {
                '(' => out.push(Tok::LParen),
                ')' => out.push(Tok::RParen),
                ',' => out.push(Tok::Comma),
                '+' | '-' | '*' | '/' => out.push(Tok::Op(c)),
                other => return Err(format!("unexpected character `{other}`")),
            }
            i += 1;
        }
    }
    Ok(out)
}

struct P {
    toks: Vec<Tok>,
    pos: usize,
}

pub fn parse_expr(s: &str) -> Result<E, String> {
    let mut p = P { toks: tokenize(s)?, pos: 0 };
    let e = p.add()?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens in expression: {s}"));
    }
    Ok(e)
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn add(&mut self) -> Result<E, String> {
        let mut lhs = self.mul()?;
        while let Some(Tok::Op(c @ ('+' | '-'))) = self.peek() {
            let c = *c;
            self.pos += 1;
            let rhs = self.mul()?;
            lhs = E::Bin(c, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn mul(&mut self) -> Result<E, String> {
        let mut lhs = self.unary()?;
        while let Some(Tok::Op(c @ ('*' | '/'))) = self.peek() {
            let c = *c;
            self.pos += 1;
            let rhs = self.unary()?;
            lhs = E::Bin(c, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn unary(&mut self) -> Result<E, String> {
        if let Some(Tok::Op('-')) = self.peek() {
            self.pos += 1;
            return Ok(E::Neg(Box::new(self.unary()?)));
        }
        self.atom()
    }
    fn atom(&mut self) -> Result<E, String> {
        match self.peek().cloned() {
            Some(Tok::Num(v)) => {
                self.pos += 1;
                Ok(E::Num(v))
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let e = self.add()?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if self.peek() == Some(&Tok::LParen) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.add()?);
                            if self.peek() == Some(&Tok::Comma) {
                                self.pos += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Tok::RParen)?;
                    match (name.as_str(), args.len()) {
                        ("grad", 1) => Ok(E::Grad(Box::new(args.pop().unwrap()))),
                        ("grad", n) => Err(format!("grad takes 1 argument, got {n}")),
                        ("dot", 2) => {
                            let b = args.pop().unwrap();
                            let a = args.pop().unwrap();
                            Ok(E::Dot(Box::new(a), Box::new(b)))
                        }
                        ("dot", n) => Err(format!("dot takes 2 arguments, got {n}")),
                        _ => Ok(E::Call(name, args)),
                    }
                } else {
                    Ok(E::Var(name))
                }
            }
            other => Err(format!("expected an expression, found {other:?}")),
        }
    }
    fn expect(&mut self, t: Tok) -> Result<(), String> {
        if self.peek() == Some(&t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {t:?}"))
        }
    }
}
