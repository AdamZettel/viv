// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Keyword arguments and defaults.
//!
//! ```viv
//! def init(rows, cols, scale = 1.0 / sqrt(rows)) {
//!     return scale * randn_matrix(rows, cols)
//! }
//! W = init(16, 8)
//! U = init(16, 8, scale=0.5)
//! m = amax(X, axis=-1, keepdims=true)
//! ```
//!
//! A parameter with a default is passed by name only; one without may be
//! passed by position or by name. A default may read the parameters before it
//! and the program's top-level names. A builtin's array operands are
//! positional and its options (`axis=`, `keepdims=`, `k=`, ...) are passed by
//! name, with NumPy's names.
//!
//! The parser fills every call out (crate::parser, `make_call`): the given
//! arguments go to their slots and the defaults are written in, so the
//! interpreter, C and wasm see the positional calls they always saw. A
//! default that reads an earlier parameter whose argument is not a name or a
//! number reads it through a hidden variable, computed once before the
//! statement (crate::surface), so each argument is computed once.
//!
//! The signatures are read from the tokens before anything is parsed
//! (`def_sigs`), so a call may come before the def it calls.

use std::collections::HashMap;

use crate::lexer::{Tok, Token};

/// One parameter of a def, as its head writes it.
#[derive(Clone, Debug)]
pub struct SigParam {
    pub name: String,
    /// the tokens of its default, `Eof` appended, when it has one
    pub default: Option<Vec<Token>>,
}

/// Every def's parameters, from the tokens: `def name(p, q: T, r = e) ...`.
pub fn def_sigs(toks: &[Token]) -> HashMap<String, Vec<SigParam>> {
    let mut out: HashMap<String, Vec<SigParam>> = HashMap::new();
    let mut k = 0;
    while k + 2 < toks.len() {
        let (Tok::Def, Tok::Ident(name), Tok::LParen) = (&toks[k].tok, &toks[k + 1].tok, &toks[k + 2].tok) else {
            k += 1;
            continue;
        };
        let mut params = Vec::new();
        let mut j = k + 3;
        let mut depth = 0i32;
        // the current parameter: its name, and where its default starts
        let mut cur: Option<(String, Option<usize>)> = None;
        let close = |cur: &mut Option<(String, Option<usize>)>, end: usize, params: &mut Vec<SigParam>| {
            if let Some((n, d)) = cur.take() {
                let default = d.map(|s| {
                    let mut v: Vec<Token> = toks[s..end].iter().filter(|t| t.tok != Tok::Newline).cloned().collect();
                    let at = toks.get(end).map(|t| t.span).unwrap_or(toks[s].span);
                    v.push(Token { tok: Tok::Eof, span: at });
                    v
                });
                params.push(SigParam { name: n, default });
            }
        };
        while j < toks.len() {
            match &toks[j].tok {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen if depth == 0 => {
                    close(&mut cur, j, &mut params);
                    break;
                }
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Comma if depth == 0 => close(&mut cur, j, &mut params),
                Tok::Ident(n) if depth == 0 && cur.is_none() => cur = Some((n.clone(), None)),
                Tok::Eq if depth == 0 => {
                    if let Some((_, d)) = cur.as_mut() {
                        if d.is_none() {
                            *d = Some(j + 1);
                        }
                    }
                }
                Tok::Eof => break,
                _ => {}
            }
            j += 1;
        }
        out.entry(name.clone()).or_insert(params);
        k = j;
    }
    out
}

/// A builtin's options: how many operands come first (positional), and each
/// option's name and slot in the positional call the engines read.
pub struct Options {
    pub operands: usize,
    pub opts: &'static [(&'static str, usize)],
}

const AXIS_KEEPDIMS: &[(&str, usize)] = &[("axis", 1), ("keepdims", 2)];
const AXIS: &[(&str, usize)] = &[("axis", 1)];

/// The builtins with options, NumPy's names for them.
pub fn options(name: &str) -> Option<Options> {
    let (operands, opts): (usize, &'static [(&'static str, usize)]) = match name {
        "sum" | "mean" | "var" | "std" | "prod" | "amax" | "amin" | "any" | "all" => (1, AXIS_KEEPDIMS),
        "argmax" | "argmin" | "cumsum" | "cumprod" => (1, AXIS),
        "diff" => (1, &[("n", 1), ("axis", 2)]),
        "take" => (2, &[("axis", 2)]),
        "trace" => (1, &[("offset", 1)]),
        "triu" | "tril" | "diag" => (1, &[("k", 1)]),
        "linspace" => (3, &[("endpoint", 3)]),
        "transpose" => (1, &[("axes", 1)]),
        _ => return None,
    };
    Some(Options { operands, opts })
}

/// The builtins that take `ordered=` (crate::parser, `ordered_option`):
/// `sum(a, ordered=true)` adds left to right instead of in NumPy's pairwise
/// order; `dot` and `prod` always go left to right and accept it to say so.
pub const ORDERED: &[&str] = &["sum", "dot", "prod"];

/// What an option left out between two given ones is: `diff`'s `n` is 1;
/// a reduction's `keepdims` needs its `axis` (there is no `axis=None` form).
pub fn gap(name: &str, opt: &str) -> Option<f64> {
    match (name, opt) {
        ("diff", "n") => Some(1.0),
        _ => None,
    }
}

/// The note an option written by position gets: the call with its options named.
pub fn positional_note(name: &str, o: &Options, given: usize) -> String {
    let named: Vec<String> = o.opts.iter().filter(|(_, s)| *s < given).map(|(n, _)| format!("{n}=")).collect();
    let operands: Vec<String> = (0..o.operands).map(|i| ["a", "b", "c"].get(i).unwrap_or(&"x").to_string()).collect();
    format!("{name}'s options are given by name: `{name}({}, {})`", operands.join(", "), named.iter().map(|n| format!("{n}...")).collect::<Vec<_>>().join(", "))
}

/// `viv fix --keywords`: a text with every builtin option written by
/// position given its name (`sum(X, -1, true)` is `sum(X, axis=-1,
/// keepdims=true)`). `defs` are the program's own defs, whose calls are not
/// builtins'. Returns the new text and how many options were named.
pub fn name_options(src: &str, defs: &std::collections::HashSet<String>) -> Result<(String, usize), String> {
    let toks = crate::lexer::lex(src, knotkit::span::SourceId(0)).map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
    let mut inserts: Vec<(usize, String)> = Vec::new();
    collect_options(&toks, defs, &mut inserts);
    inserts.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = src.to_string();
    let n = inserts.len();
    for (at, text) in inserts {
        out.insert_str(at, &text);
    }
    Ok((out, n))
}

fn collect_options(toks: &[Token], defs: &std::collections::HashSet<String>, inserts: &mut Vec<(usize, String)>) {
    for (k, t) in toks.iter().enumerate() {
        if let Tok::FStr(parts) = &t.tok {
            for p in parts {
                if let crate::lexer::FPart::Field { toks: ft, spec_fields, .. } = p {
                    collect_options(ft, defs, inserts);
                    for sf in spec_fields {
                        collect_options(sf, defs, inserts);
                    }
                }
            }
        }
        let Tok::Ident(name) = &t.tok else { continue };
        if defs.contains(name) || !matches!(toks.get(k + 1).map(|t| &t.tok), Some(Tok::LParen)) {
            continue;
        }
        let prev = k.checked_sub(1).map(|j| &toks[j].tok);
        if matches!(prev, Some(Tok::Def)) {
            continue;
        }
        let Some(o) = options(name) else { continue };
        // `X.sum(-1)`: the receiver is the first operand
        let shift = usize::from(matches!(prev, Some(Tok::Dot)));
        // the arguments: where each starts, at depth 0
        let mut starts: Vec<usize> = Vec::new();
        let mut depth = 0i32;
        let mut j = k + 2;
        let mut fresh = true;
        while j < toks.len() {
            match &toks[j].tok {
                Tok::LParen | Tok::LBracket | Tok::LBrace => {
                    if depth == 0 && fresh {
                        starts.push(j);
                        fresh = false;
                    }
                    depth += 1;
                }
                Tok::RParen if depth == 0 => break,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Comma if depth == 0 => fresh = true,
                Tok::Newline => {}
                _ => {
                    if depth == 0 && fresh {
                        starts.push(j);
                        fresh = false;
                    }
                }
            }
            j += 1;
        }
        for (i, &s) in starts.iter().enumerate() {
            let slot = i + shift;
            if slot < o.operands {
                continue;
            }
            // already a keyword argument
            if matches!((&toks[s].tok, toks.get(s + 1).map(|t| &t.tok)), (Tok::Ident(_), Some(Tok::Eq))) {
                continue;
            }
            if let Some((opt, _)) = o.opts.iter().find(|(_, x)| *x == slot) {
                inserts.push((toks[s].span.start as usize, format!("{opt}=")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knotkit::span::SourceId;

    #[test]
    fn reads_signatures_with_defaults() {
        let toks = crate::lexer::lex("def init(rows, cols: Num, scale = 1.0 / sqrt(rows), k = [1, 2]) {\n    return 0\n}\n", SourceId(0)).unwrap();
        let s = def_sigs(&toks);
        let p = &s["init"];
        assert_eq!(p.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), vec!["rows", "cols", "scale", "k"]);
        assert!(p[0].default.is_none() && p[1].default.is_none());
        assert_eq!(p[2].default.as_ref().unwrap().len(), 7); // 1.0 / sqrt ( rows ) Eof
        assert_eq!(p[3].default.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn names_positional_options() {
        let defs = std::collections::HashSet::new();
        let (t, n) = name_options("m = amax(X, -1, true)\ns = sum(f(a, b), 0)\nd = X.sum(1)\nt = take(W, ids, 0)\nk = sum(X, axis=1)\n", &defs).unwrap();
        assert_eq!(n, 5);
        assert_eq!(t, "m = amax(X, axis=-1, keepdims=true)\ns = sum(f(a, b), axis=0)\nd = X.sum(axis=1)\nt = take(W, ids, axis=0)\nk = sum(X, axis=1)\n");
    }
}
