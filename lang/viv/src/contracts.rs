// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Contracts and their methods.
//!
//! A contract names a problem: what it computes (`intent`), its signature,
//! what two methods must agree on and how closely (`agree`), and the methods
//! that solve it, the first (or the `default`) being the one a call runs:
//!
//! ```text
//! contract lowest_eigenpairs(apply_h, n: Int, k: Int) -> values: Arr, vectors: Arr {
//!     intent "the k lowest eigenvalues of a symmetric operator on n-vectors"
//!     methods lanczos, dense
//!     agree values within relative 1.0e-8
//! }
//! ```
//!
//! This pass runs on the tokens, before the parser reads them, and writes the
//! declaration as items every engine already runs: an enum of the methods
//! (`LowestEigenpairs { Lanczos, Dense, Every }`), the selection
//! (`using_lowest_eigenpairs = LowestEigenpairs.Lanczos`), the contract's def,
//! which takes the selection and each method's settings as keyword
//! parameters with defaults and calls the selected method (or, for `Every`,
//! every method, stopping at the first disagreement), and
//! `viv__every_<name>`, which phrases call. Selection is written
//!
//! ```text
//! using f = jacobi                       (the following calls)
//! using f = sel                          (a variable holding the method)
//! using f = every method
//! f(x) using jacobi                      (one call, rare)
//! ```
//!
//! and a method's settings in a top-level statement, used wherever it runs:
//!
//! ```text
//! settings lanczos(tol = 1.0e-12)
//! ```
//!
//! and `viv run --using f=dense` sets the selection a contract starts with
//! (`apply_using`, on the program's text, as `--set` is).
//!
//! Before writing anything it checks every method against the contract: the
//! def exists, takes the contract's parameters in order (an annotated
//! parameter must have the contract's type), has a default or a binding for
//! every further parameter, and returns as many values as the contract has
//! outputs. A method the program imports but whose text is not here (a module
//! read on its own) is not checked; the spliced program is.

use crate::lexer::{FPart, Tok, Token};
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::collections::{HashMap, HashSet};

type R<T> = Result<T, Diagnostic>;

/// The prefix the dispatching def had before it became the contract's own
/// def; kept so the tools that strip it still read older maps.
pub const BY: &str = "viv__by_";
/// The contract def's parameter that carries the method a call runs; it
/// defaults to the contract's selection, `using_<name>`.
pub const SEL: &str = "viv__sel";
/// The prefix of the def that runs every method of a contract (what a
/// phrase ending `using every method` calls).
pub const EVERY_DEF: &str = "viv__every_";
/// The prefix of the variable that holds a program's `using name = ...`
/// for a contract it imports (hidden, so the imports leave it alone).
pub const UNIT_SEL: &str = "viv__using_";
/// The prefix of a contract's selection variable.
pub const USING: &str = "using_";
/// The variant of a contract's enum that runs every method.
pub const EVERY: &str = "Every";

/// One method in a `methods` line: the def, and the expressions bound to
/// the parameters it takes beyond the contract's.
#[derive(Clone, Debug)]
pub struct Method {
    pub name: String,
    pub span: Span,
    pub binds: Vec<(String, Vec<Token>)>,
    /// the label of the group it is listed under (`dense: householder_ql,
    /// jacobi`), if any; labels never affect calling or selection by name
    pub group: Option<String>,
}

/// An `agree <output> within relative|absolute <tol>` line (`tol` None:
/// `exactly`). The tolerance is one token: a number or a name.
#[derive(Clone, Debug)]
pub struct Agree {
    pub output: String,
    pub span: Span,
    pub tol: Option<Tol>,
}

/// A tolerance: relative (`|a - b| <= tol * max(1, |reference|)`) or
/// absolute (`|a - b| <= tol`).
#[derive(Clone, Debug)]
pub struct Tol {
    pub relative: bool,
    pub value: Token,
}

impl Tol {
    pub fn text(&self) -> String {
        format!("{} {}", if self.relative { "relative" } else { "absolute" }, expr_text(std::slice::from_ref(&self.value)))
    }
}

/// A contract declaration, as read from the tokens.
#[derive(Clone, Debug)]
pub struct Contract {
    pub name: String,
    pub name_span: Span,
    /// the parameters: name, and the tokens of the type (empty: none written)
    pub params: Vec<(String, Vec<Token>)>,
    /// the outputs: name and the tokens of the type
    pub outputs: Vec<(String, Vec<Token>)>,
    pub intent: String,
    pub methods: Vec<Method>,
    /// index into `methods`
    pub default: usize,
    pub agree: Vec<Agree>,
    /// the selection it starts with: a method's index, or None for every method
    pub start: Option<usize>,
    /// with `start` None: the group whose methods all run (`using every
    /// dense method`); None for every method
    pub start_group: Option<String>,
    pub requires: Vec<Vec<Token>>,
    /// `consensus median`: with every method, the call returns the median of
    /// the methods' values and compares each with it (one Num output)
    pub median: bool,
    /// the settings of the methods: the parameters a method takes beyond the
    /// contract's that the `methods` line leaves unbound and that have a
    /// default (filled in by `check`)
    pub settings: Vec<Setting>,
    /// from `contract` to the closing `}`
    pub span: Span,
}

impl Contract {
    pub fn enum_name(&self) -> String {
        camel(&self.name)
    }
    pub fn variant(&self, k: usize) -> String {
        camel(&self.methods[k].name)
    }
    /// The group labels, in the order first listed.
    pub fn groups(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for m in &self.methods {
            if let Some(g) = &m.group {
                if !out.contains(g) {
                    out.push(g.clone());
                }
            }
        }
        out
    }
    /// The enum variant that runs every method of group `g`.
    pub fn every_variant(g: &str) -> String {
        format!("{EVERY}{}", camel(g))
    }
    pub fn start_label(&self) -> String {
        match (self.start, &self.start_group) {
            (Some(k), _) => self.methods[k].name.clone(),
            (None, Some(g)) => format!("every {g}"),
            (None, None) => "every".to_string(),
        }
    }
}

/// A method's setting: a parameter beyond the contract's with a default,
/// which a program gives with `settings m(name = value)`.
#[derive(Clone, Debug)]
pub struct Setting {
    pub method: usize,
    pub name: String,
    pub ty: Vec<Token>,
    /// the default, with the method's own names for the contract's
    /// parameters and its earlier settings written as the dispatcher's
    pub default: Vec<Token>,
}

/// The dispatcher's parameter that carries setting `name` of method `m`.
pub fn setting_param(m: &str, name: &str) -> String {
    format!("viv__set_{m}__{name}")
}

/// `q_gauss` -> `QGauss`, the name of a method's variant.
pub fn camel(name: &str) -> String {
    name.split('_').filter(|p| !p.is_empty()).map(|p| {
        let mut c = p.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    }).collect()
}

fn word(t: Option<&Token>, w: &str) -> bool {
    matches!(t.map(|t| &t.tok), Some(Tok::Ident(s)) if s == w)
}

fn ident(t: Option<&Token>) -> Option<String> {
    match t.map(|t| &t.tok) {
        Some(Tok::Ident(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Whether a contract starts at token `k`: `contract name (` at the start of
/// a line.
fn at_contract(toks: &[Token], k: usize) -> bool {
    (k == 0 || matches!(toks[k - 1].tok, Tok::Newline | Tok::LBrace | Tok::RBrace))
        && word(toks.get(k), "contract")
        && ident(toks.get(k + 1)).is_some()
        && matches!(toks.get(k + 2).map(|t| &t.tok), Some(Tok::LParen))
}

/// Whether the program has a contract (cheap: no contract, no pass).
fn any_contract(toks: &[Token]) -> bool {
    (0..toks.len()).any(|k| at_contract(toks, k))
}

fn err(span: Span, msg: impl Into<String>, label: impl Into<String>) -> Diagnostic {
    Diagnostic::error(msg).with_primary(span, label)
}

/// The tokens from `k` up to (not including) the first of `stops` at
/// bracket depth 0, and where that stop is.
fn until(toks: &[Token], mut k: usize, stops: &[Tok]) -> (Vec<Token>, usize) {
    let mut out = Vec::new();
    let mut depth = 0i32;
    while k < toks.len() {
        let t = &toks[k];
        if depth == 0 && (stops.contains(&t.tok) || t.tok == Tok::Eof) {
            break;
        }
        match t.tok {
            Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
            Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
            _ => {}
        }
        out.push(t.clone());
        k += 1;
    }
    (out, k)
}

/// The text of a type's tokens, for comparing and for messages.
pub fn type_text(toks: &[Token]) -> String {
    let mut s = String::new();
    for t in toks {
        match &t.tok {
            Tok::Ident(n) => s.push_str(n),
            Tok::LBracket => s.push('['),
            Tok::RBracket => s.push(']'),
            Tok::Question => s.push('?'),
            Tok::Comma => s.push_str(", "),
            other => s.push_str(&format!("{other:?}")),
        }
    }
    s
}

/// `every method` or `every <label> method` at `j`: the variant it selects
/// and where it ends. A label the declaration `c` does not have is an error.
fn every_at(toks: &[Token], j: usize, c: Option<&Contract>) -> R<Option<(String, usize)>> {
    if !word(toks.get(j), "every") {
        return Ok(None);
    }
    if word(toks.get(j + 1), "method") {
        return Ok(Some((EVERY.to_string(), j + 2)));
    }
    if let (Some(label), true) = (ident(toks.get(j + 1)), word(toks.get(j + 2), "method")) {
        if let Some(c) = c {
            let groups = c.groups();
            if !groups.contains(&label) {
                let msg = if groups.is_empty() {
                    format!("contract `{}` has no groups of methods; `every method` runs them all", c.name)
                } else {
                    format!("contract `{}` has no group `{label}`; its groups are {}", c.name, groups.join(", "))
                };
                return Err(err(toks[j + 1].span, msg, "not a group"));
            }
        }
        return Ok(Some((Contract::every_variant(&label), j + 3)));
    }
    Ok(None)
}

/// A comma-separated list of methods from `j`, each `m` or `m(x = e, ...)`;
/// returns them and where the list ends.
fn method_list(name: &str, toks: &[Token], mut j: usize, group: Option<&str>, have: &[Method]) -> R<(Vec<Method>, usize)> {
    let mut out: Vec<Method> = Vec::new();
    let at = toks.get(j).or(toks.last()).map(|t| t.span).expect("the tokens end in Eof");
    loop {
        let Some(m) = ident(toks.get(j)) else {
            return Err(err(toks.get(j).map_or(at, |t| t.span), format!("contract `{name}`: `methods` lists defs, separated by commas"), "expected a def's name"));
        };
        let mspan = toks[j].span;
        if m == "every" {
            return Err(err(mspan, format!("contract `{name}`: a method may not be called `every`"), "`using every method` means all of them"));
        }
        if have.iter().chain(out.iter()).any(|x| x.name == m) {
            return Err(err(mspan, format!("contract `{name}` lists `{m}` twice"), "listed again here"));
        }
        j += 1;
        let mut binds = Vec::new();
        if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::LParen)) {
            j += 1;
            loop {
                if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::RParen)) {
                    j += 1;
                    break;
                }
                let (Some(p), Some(Tok::Eq)) = (ident(toks.get(j)), toks.get(j + 1).map(|t| &t.tok)) else {
                    return Err(err(toks.get(j).map_or(mspan, |t| t.span), format!("contract `{name}`: `{m}(...)` binds the parameters `{m}` takes beyond the contract's, each `name = value`"), "expected `name = value`"));
                };
                let (e, end) = until(toks, j + 2, &[Tok::Comma, Tok::RParen]);
                if e.is_empty() {
                    return Err(err(toks[j].span, format!("contract `{name}`: `{p} =` needs a value"), "here"));
                }
                binds.push((p, e));
                j = end;
                if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Comma)) {
                    j += 1;
                }
            }
        }
        out.push(Method { name: m, span: mspan, binds, group: group.map(str::to_string) });
        if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Comma)) {
            j += 1;
            continue;
        }
        return Ok((out, j));
    }
}

/// Read the contract at `k` (`at_contract`); returns it and the index after
/// its closing `}`.
fn read(toks: &[Token], k: usize) -> R<(Contract, usize)> {
    let start = toks[k].span;
    let name = ident(toks.get(k + 1)).unwrap_or_default();
    let name_span = toks[k + 1].span;
    let mut j = k + 3;
    // parameters
    let mut params: Vec<(String, Vec<Token>)> = Vec::new();
    loop {
        match toks.get(j).map(|t| &t.tok) {
            Some(Tok::RParen) => {
                j += 1;
                break;
            }
            Some(Tok::Ident(p)) => {
                let p = p.clone();
                if params.iter().any(|(q, _)| *q == p) {
                    return Err(err(toks[j].span, format!("contract `{name}` has two parameters named `{p}`"), "named again here"));
                }
                j += 1;
                let mut ty = Vec::new();
                if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Colon)) {
                    let (t, e) = until(toks, j + 1, &[Tok::Comma, Tok::RParen]);
                    ty = t;
                    j = e;
                }
                params.push((p, ty));
                if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Comma)) {
                    j += 1;
                }
            }
            _ => return Err(err(toks.get(j).map_or(name_span, |t| t.span), format!("contract `{name}`: expected a parameter name or `)`"), "here")),
        }
    }
    // outputs
    if !matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Arrow)) {
        return Err(err(toks.get(j).map_or(name_span, |t| t.span), format!("contract `{name}`: expected `->` and its outputs, each `name: Type`"), "here")
            .with_note("`contract eigensolver(H: Arr, k: Int) -> energies: Arr, vectors: Arr { ... }`"));
    }
    j += 1;
    let mut outputs: Vec<(String, Vec<Token>)> = Vec::new();
    loop {
        let Some(o) = ident(toks.get(j)) else {
            return Err(err(toks.get(j).map_or(name_span, |t| t.span), format!("contract `{name}`: each output is written `name: Type`"), "expected an output's name"));
        };
        if !matches!(toks.get(j + 1).map(|t| &t.tok), Some(Tok::Colon)) {
            return Err(err(toks[j].span, format!("contract `{name}`: each output is written `name: Type`"), "this output has no type"));
        }
        if params.iter().any(|(p, _)| *p == o) || outputs.iter().any(|(q, _)| *q == o) {
            return Err(err(toks[j].span, format!("contract `{name}`: the output `{o}` has the name of a parameter or of another output"), "rename it"));
        }
        let (ty, e) = until(toks, j + 2, &[Tok::Comma, Tok::LBrace]);
        outputs.push((o, ty));
        j = e;
        if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Comma)) {
            j += 1;
            continue;
        }
        break;
    }
    if !matches!(toks.get(j).map(|t| &t.tok), Some(Tok::LBrace)) {
        return Err(err(toks.get(j).map_or(name_span, |t| t.span), format!("contract `{name}`: expected `{{`"), "here"));
    }
    j += 1;
    // clauses, one per line
    let mut intent: Option<String> = None;
    let mut methods: Vec<Method> = Vec::new();
    let mut default: Option<(String, Span)> = None;
    let mut agree: Vec<Agree> = Vec::new();
    let mut start_sel: Option<(Option<String>, Span)> = None;
    let mut requires = Vec::new();
    let mut median = false;
    let close;
    loop {
        while matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Newline)) {
            j += 1;
        }
        let Some(t) = toks.get(j) else { return Err(err(start, format!("contract `{name}` has no closing `}}`"), "opened here")) };
        if t.tok == Tok::RBrace {
            close = t.span;
            j += 1;
            break;
        }
        if t.tok == Tok::Eof {
            return Err(err(start, format!("contract `{name}` has no closing `}}`"), "opened here"));
        }
        let kw = ident(Some(t)).unwrap_or_default();
        let at = t.span;
        j += 1;
        match kw.as_str() {
            "intent" => match toks.get(j).map(|t| &t.tok) {
                Some(Tok::Str(s)) => {
                    intent = Some(s.clone());
                    j += 1;
                }
                _ => return Err(err(at, format!("contract `{name}`: `intent` takes a string, the plain statement of what the contract computes"), "here")),
            },
            "methods" if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Newline)) => {
                // a block of `label: m1, m2` lines
                let mut any = false;
                loop {
                    let mut q = j;
                    while matches!(toks.get(q).map(|t| &t.tok), Some(Tok::Newline)) {
                        q += 1;
                    }
                    let (Some(label), Some(Tok::Colon)) = (ident(toks.get(q)), toks.get(q + 1).map(|t| &t.tok)) else { break };
                    if label == "every" {
                        return Err(err(toks[q].span, format!("contract `{name}`: a group may not be called `every`"), "rename it"));
                    }
                    if methods.iter().any(|x: &Method| x.group.as_deref() == Some(label.as_str())) {
                        return Err(err(toks[q].span, format!("contract `{name}` has two groups labeled `{label}`"), "labeled again here"));
                    }
                    let (list, end) = method_list(&name, toks, q + 2, Some(&label), &methods)?;
                    methods.extend(list);
                    j = end;
                    any = true;
                }
                if !any {
                    return Err(err(at, format!("contract `{name}`: `methods` lists defs on its line (`methods a, b`), or on the lines after it, each `label: a, b`"), "here"));
                }
            }
            "methods" => {
                let (list, end) = method_list(&name, toks, j, None, &methods)?;
                methods.extend(list);
                j = end;
            }
            "default" => {
                let Some(m) = ident(toks.get(j)) else { return Err(err(at, format!("contract `{name}`: `default` names one of the methods"), "here")) };
                default = Some((m, toks[j].span));
                j += 1;
            }
            "agree" => {
                let Some(o) = ident(toks.get(j)) else { return Err(err(at, format!("contract `{name}`: `agree` names an output, then `within relative <tolerance>`, `within absolute <tolerance>` or `exactly`"), "here")) };
                let ospan = toks[j].span;
                if !outputs.iter().any(|(q, _)| *q == o) {
                    let names: Vec<&str> = outputs.iter().map(|(q, _)| q.as_str()).collect();
                    return Err(err(ospan, format!("contract `{name}` has no output `{o}`; its outputs are {}", names.join(", ")), "not an output"));
                }
                j += 1;
                if word(toks.get(j), "exactly") {
                    agree.push(Agree { output: o, span: ospan, tol: None });
                    j += 1;
                } else if word(toks.get(j), "within") {
                    let how = toks.get(j + 1);
                    let relative = word(how, "relative");
                    if !relative && !word(how, "absolute") {
                        return Err(err(toks.get(j + 1).map_or(toks[j].span, |t| t.span), format!("contract `{name}`: write `within relative <tolerance>` or `within absolute <tolerance>`"), "relative or absolute?")
                            .with_note("relative: |a - b| <= tol * max(1, |reference|); absolute: |a - b| <= tol"));
                    }
                    let (e, end) = until(toks, j + 2, &[Tok::Newline, Tok::RBrace]);
                    let one = e.len() == 1 && matches!(e[0].tok, Tok::Num(..) | Tok::Ident(_));
                    if !one {
                        let sp = e.first().map_or(toks[j + 1].span, |t| t.span.merge(e.last().expect("nonempty").span));
                        return Err(err(sp, format!("contract `{name}`: a tolerance is one number or one name, not a formula"), if e.is_empty() { "no tolerance" } else { "a formula" })
                            .with_note(format!("`agree {o} within relative 1.0e-8`, or give the formula a name at the top level and write `within absolute TOL`")));
                    }
                    agree.push(Agree { output: o, span: ospan, tol: Some(Tol { relative, value: e[0].clone() }) });
                    j = end;
                } else {
                    return Err(err(toks.get(j).map_or(ospan, |t| t.span), format!("contract `{name}`: after `agree {o}`, write `within relative <tolerance>`, `within absolute <tolerance>` or `exactly`"), "here"));
                }
            }
            "using" => {
                if word(toks.get(j), "every") && word(toks.get(j + 1), "method") {
                    start_sel = Some((None, toks[j].span));
                    j += 2;
                } else if word(toks.get(j), "every") && ident(toks.get(j + 1)).is_some() && word(toks.get(j + 2), "method") {
                    start_sel = Some((Some(format!("every:{}", ident(toks.get(j + 1)).expect("checked"))), toks[j].span));
                    j += 3;
                } else if let Some(m) = ident(toks.get(j)) {
                    start_sel = Some((Some(m), toks[j].span));
                    j += 1;
                } else {
                    return Err(err(at, format!("contract `{name}`: `using` names a method, or `every method`"), "here"));
                }
            }
            "consensus" => {
                if !word(toks.get(j), "median") {
                    return Err(err(at, format!("contract `{name}`: `consensus` takes `median`"), "here"));
                }
                if outputs.len() != 1 || type_text(&outputs[0].1) != "Num" {
                    return Err(err(at, format!("contract `{name}`: `consensus median` needs one output, a Num"), "here"));
                }
                median = true;
                j += 1;
            }
            "require" => {
                let (e, end) = until(toks, j, &[Tok::Newline, Tok::RBrace]);
                if e.is_empty() {
                    return Err(err(at, format!("contract `{name}`: `require` needs a condition"), "here"));
                }
                requires.push(e);
                j = end;
            }
            _ => {
                return Err(err(at, format!("contract `{name}`: a contract holds `intent`, `methods`, `default`, `agree`, `using`, `consensus` and `require` lines"), "not one of them"));
            }
        }
        if !matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Newline | Tok::RBrace)) {
            return Err(err(toks.get(j).map_or(at, |t| t.span), format!("contract `{name}`: one clause per line"), "unexpected here"));
        }
    }
    let span = start.merge(close);
    let Some(intent) = intent else {
        return Err(err(name_span, format!("contract `{name}` has no `intent` line"), "say what it computes")
            .with_note("`intent \"the lowest k eigenvalues of H\"`: a contract names a problem, in a plain statement"));
    };
    if methods.is_empty() {
        return Err(err(name_span, format!("contract `{name}` has no `methods` line"), "list the defs that solve it"));
    }
    let find = |m: &str, sp: Span, what: &str| -> R<usize> {
        methods.iter().position(|x| x.name == m).ok_or_else(|| {
            let names: Vec<&str> = methods.iter().map(|x| x.name.as_str()).collect();
            err(sp, format!("contract `{name}`: `{what} {m}` names no method; the methods are {}", names.join(", ")), "not a method")
        })
    };
    let default = match &default {
        Some((m, sp)) => find(m, *sp, "default")?,
        None => 0,
    };
    let mut start_group = None;
    let start_sel = match &start_sel {
        Some((Some(m), sp)) if m.starts_with("every:") => {
            let g = m["every:".len()..].to_string();
            if !methods.iter().any(|x| x.group.as_deref() == Some(g.as_str())) {
                return Err(err(*sp, format!("contract `{name}` has no group `{g}`"), "not a group"));
            }
            start_group = Some(g);
            None
        }
        Some((Some(m), sp)) => Some(find(m, *sp, "using")?),
        Some((None, _)) => None,
        None => Some(default),
    };
    if methods.len() > 1 && agree.is_empty() {
        return Err(err(name_span, format!("contract `{name}` has {} methods and no `agree` line", methods.len()), "say what two methods must agree on")
            .with_note("`agree values within relative 1.0e-8`, or `agree count exactly`"));
    }
    for a in &agree {
        let ty = type_text(&outputs.iter().find(|(o, _)| *o == a.output).expect("checked").1);
        match (ty.as_str(), &a.tol) {
            ("Num" | "Arr", _) | ("Int" | "Bool" | "Str" | "[Int]" | "[Str]", None) => {}
            ("Int" | "Bool" | "Str" | "[Int]" | "[Str]", Some(_)) => {
                return Err(err(a.span, format!("contract `{name}`: `{}` is {ty}; write `agree {} exactly`", a.output, a.output), "compared exactly"));
            }
            _ => return Err(err(a.span, format!("contract `{name}`: an `agree` line compares a Num, an Arr, an Int, a Bool or a Str; `{}` is {ty}", a.output), "here")),
        }
    }
    Ok((Contract { name, name_span, params, outputs, intent, methods, default, agree, start: start_sel, start_group, requires, median, settings: Vec::new(), span }, j))
}

/// Every contract in the tokens, in order, with the token range it covers.
pub fn read_all(toks: &[Token]) -> R<Vec<(usize, usize, Contract)>> {
    read_all_dups(toks).map(|(v, _)| v)
}

/// `read_all`, with the token ranges of repeated identical declarations.
fn read_all_dups(toks: &[Token]) -> R<(Vec<(usize, usize, Contract)>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut dropped = Vec::new();
    let mut k = 0;
    let mut depth = 0i32;
    while k < toks.len() {
        if depth == 0 && at_contract(toks, k) {
            let (c, end) = read(toks, k)?;
            let text = |a: usize, b: usize| toks[a..b].iter().map(|t| format!("{:?}", t.tok)).collect::<Vec<_>>().join(" ");
            if let Some((a, b, _)) = out.iter().find(|(_, _, d): &&(usize, usize, Contract)| d.name == c.name) {
                // one module spliced twice (a program file that is also
                // imported): the second copy is the same declaration, dropped
                if text(*a, *b) == text(k, end) {
                    dropped.push((k, end));
                    k = end;
                    continue;
                }
                return Err(err(c.name_span, format!("two contracts are named `{}`", c.name), "declared again here"));
            }
            out.push((k, end, c));
            k = end;
            continue;
        }
        match toks[k].tok {
            Tok::LBrace => depth += 1,
            Tok::RBrace => depth -= 1,
            _ => {}
        }
        k += 1;
    }
    Ok((out, dropped))
}

/// The contracts of a program's text, for the tools that show them (the
/// code map, `viv features`); empty when the text does not lex or holds none.
pub fn scan(src: &str) -> Vec<Contract> {
    if !src.contains("contract") {
        return Vec::new();
    }
    let Ok(toks) = crate::lexer::lex(src, knotkit::span::SourceId(0)) else { return Vec::new() };
    read_all(&toks).map(|v| v.into_iter().map(|(_, _, c)| c).collect()).unwrap_or_default()
}

/// A def's parameters (name and type tokens) and the counts of values its
/// `return`s give (one entry per `return`; 0 for a call, which may forward
/// several), read from the tokens.
struct DefSig {
    params: Vec<(String, Vec<Token>)>,
    /// each parameter's default, as tokens (None: it has none)
    defaults: Vec<Option<Vec<Token>>>,
    returns: Vec<usize>,
    annotated: bool,
}

fn def_sigs(toks: &[Token]) -> HashMap<String, DefSig> {
    let mut out = HashMap::new();
    let mut k = 0;
    while k + 2 < toks.len() {
        let (Tok::Def, Some(name), Tok::LParen) = (&toks[k].tok, ident(toks.get(k + 1)), &toks[k + 2].tok) else {
            k += 1;
            continue;
        };
        let mut j = k + 3;
        let mut params = Vec::new();
        let mut defaults = Vec::new();
        while let Some(p) = ident(toks.get(j)) {
            j += 1;
            let mut ty = Vec::new();
            if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Colon)) {
                let (t, e) = until(toks, j + 1, &[Tok::Comma, Tok::RParen, Tok::Eq]);
                ty = t;
                j = e;
            }
            let mut default = None;
            if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Eq)) {
                let (d, e) = until(toks, j + 1, &[Tok::Comma, Tok::RParen]);
                default = Some(d.into_iter().filter(|t| t.tok != Tok::Newline).collect());
                j = e;
            }
            params.push((p, ty));
            defaults.push(default);
            if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Comma)) {
                j += 1;
            }
        }
        let (_, close) = until(toks, j, &[Tok::RParen]);
        j = close + 1;
        let annotated = matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Arrow));
        // the body: from its `{` to the matching `}`
        while j < toks.len() && toks[j].tok != Tok::LBrace {
            j += 1;
        }
        let body_start = j;
        let mut depth = 0i32;
        let mut returns = Vec::new();
        while j < toks.len() {
            match toks[j].tok {
                Tok::LBrace => depth += 1,
                Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                // a nested def's returns are its own
                Tok::Def if j > body_start => break,
                Tok::Return => {
                    let (e, _) = until(toks, j + 1, &[Tok::Newline, Tok::RBrace]);
                    let commas = {
                        let mut d = 0i32;
                        e.iter().filter(|t| {
                            match t.tok {
                                Tok::LParen | Tok::LBracket | Tok::LBrace => d += 1,
                                Tok::RParen | Tok::RBracket | Tok::RBrace => d -= 1,
                                _ => {}
                            }
                            d == 0 && t.tok == Tok::Comma
                        }).count()
                    };
                    let call = e.len() >= 3 && matches!(e[0].tok, Tok::Ident(_)) && e[1].tok == Tok::LParen && matches!(e.last().map(|t| &t.tok), Some(Tok::RParen)) && commas == 0;
                    returns.push(if call { 0 } else { commas + 1 });
                }
                _ => {}
            }
            j += 1;
        }
        out.entry(name).or_insert(DefSig { params, defaults, returns, annotated });
        k += 3;
    }
    out
}

/// Check every method of `c` against the contract (see the module doc).
fn check(c: &Contract, sigs: &HashMap<String, DefSig>, outside: &HashSet<String>) -> R<Vec<Setting>> {
    let mut settings = Vec::new();
    let n = c.params.len();
    let sig_text = |ps: &[(String, Vec<Token>)]| ps.iter().map(|(p, t)| if t.is_empty() { p.clone() } else { format!("{p}: {}", type_text(t)) }).collect::<Vec<_>>().join(", ");
    for (mi, m) in c.methods.iter().enumerate() {
        let Some(s) = sigs.get(&m.name) else {
            if outside.contains(&m.name) {
                continue;
            }
            return Err(err(m.span, format!("contract `{}` lists `{}`, which is not a def of the program", c.name, m.name), "no such def"));
        };
        if s.params.len() < n {
            return Err(err(m.span, format!("`{}` takes {} parameters, and contract `{}` passes {}: ({})", m.name, s.params.len(), c.name, n, sig_text(&c.params)), "too few parameters")
                .with_note(format!("`{}` is ({})", m.name, sig_text(&s.params))));
        }
        for (i, ((cp, ct), (mp, mt))) in c.params.iter().zip(&s.params).enumerate() {
            if !ct.is_empty() && !mt.is_empty() && type_text(ct) != type_text(mt) {
                return Err(err(m.span, format!("`{}` takes `{mp}: {}` as parameter {}, where contract `{}` passes `{cp}: {}`", m.name, type_text(mt), i + 1, c.name, type_text(ct)), "signature differs")
                    .with_note(format!("contract `{}` is ({}); `{}` is ({})", c.name, sig_text(&c.params), m.name, sig_text(&s.params))));
            }
            if ct.is_empty() && !mt.is_empty() {
                return Err(err(m.span, format!("`{}` takes `{mp}: {}` as parameter {}, where contract `{}` gives `{cp}` no type", m.name, type_text(mt), i + 1, c.name), "annotate the contract's parameter"));
            }
        }
        let extra: Vec<&String> = s.params[n..].iter().map(|(p, _)| p).collect();
        for (b, toks) in &m.binds {
            if !extra.contains(&b) {
                let sp = toks.first().map_or(m.span, |t| t.span);
                return Err(err(sp, format!("`{}` has no parameter `{b}` beyond the contract's ({})", m.name, if extra.is_empty() { "it takes none".to_string() } else { extra.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") }), "not one of them"));
            }
        }
        let unbound: Vec<&str> = extra
            .iter()
            .enumerate()
            .filter(|(i, p)| !m.binds.iter().any(|(b, _)| b == **p) && s.defaults[n + i].is_none())
            .map(|(_, s)| s.as_str())
            .collect();
        // the method's names for the contract's parameters, and its settings,
        // as the dispatcher names them
        let mut rename: HashMap<String, String> = c.params.iter().zip(&s.params).map(|((cp, _), (mp, _))| (mp.clone(), cp.clone())).collect();
        for (i, (p, ty)) in s.params.iter().enumerate().skip(n) {
            let Some(d) = &s.defaults[i] else { continue };
            if m.binds.iter().any(|(b, _)| b == p) {
                continue;
            }
            let default = d.iter().map(|t| match &t.tok {
                Tok::Ident(x) if rename.contains_key(x) => Token { tok: Tok::Ident(rename[x].clone()), span: t.span },
                _ => t.clone(),
            }).collect();
            rename.insert(p.clone(), setting_param(&m.name, p));
            settings.push(Setting { method: mi, name: p.clone(), ty: ty.clone(), default });
        }
        if !unbound.is_empty() {
            let eg: Vec<String> = unbound.iter().map(|p| format!("{p} = ...")).collect();
            return Err(err(m.span, format!("`{}` takes {} beyond the contract's parameters with no default; give {} a default in `{}`, or bind {} in the `methods` line", m.name, unbound.join(", "), if unbound.len() == 1 { "it" } else { "them" }, m.name, if unbound.len() == 1 { "it" } else { "them" }), "unbound")
                .with_note(format!("`methods {}({})`", m.name, eg.join(", "))));
        }
        let want = c.outputs.len();
        let bad = s.returns.iter().find(|&&r| r > 1 && r != want).copied().or_else(|| {
            (want > 1 && !s.annotated && !s.returns.is_empty() && s.returns.iter().all(|&r| r == 1)).then_some(1)
        }).or_else(|| (want == 1 && s.returns.iter().any(|&r| r > 1)).then(|| *s.returns.iter().find(|&&r| r > 1).expect("found")));
        if let Some(got) = bad {
            let outs: Vec<&str> = c.outputs.iter().map(|(o, _)| o.as_str()).collect();
            return Err(err(m.span, format!("`{}` returns {got} value{}; contract `{}` has {want} output{} ({})", m.name, if got == 1 { "" } else { "s" }, c.name, if want == 1 { "" } else { "s" }, outs.join(", ")), "returns differ"));
        }
    }
    Ok(settings)
}

/// Builds generated tokens: text lexed once, with `viv__run_N` placeholders
/// replaced by token runs copied from the source (which keep their spans).
struct Gen {
    text: String,
    runs: Vec<Vec<Token>>,
}

impl Gen {
    fn new() -> Self {
        Gen { text: String::new(), runs: Vec::new() }
    }
    fn s(&mut self, t: &str) {
        self.text.push_str(t);
    }
    fn run(&mut self, toks: &[Token]) {
        self.text.push_str(&format!(" viv__run_{} ", self.runs.len()));
        self.runs.push(toks.to_vec());
    }
    fn tokens(self, span: Span) -> R<Vec<Token>> {
        let toks = crate::lexer::lex(&self.text, span.source).map_err(|ds| {
            ds.into_iter().next().unwrap_or_else(|| Diagnostic::error("contract: generated code does not lex")).with_note(self.text.clone())
        })?;
        let mut out = Vec::with_capacity(toks.len());
        for t in toks {
            if t.tok == Tok::Eof {
                break;
            }
            if let Tok::Ident(n) = &t.tok {
                if let Some(i) = n.strip_prefix("viv__run_").and_then(|r| r.parse::<usize>().ok()) {
                    out.extend(self.runs[i].iter().cloned());
                    continue;
                }
            }
            out.push(respan(t, span));
        }
        Ok(out)
    }
}

fn respan(mut t: Token, span: Span) -> Token {
    t.span = span;
    if let Tok::FStr(parts) = &mut t.tok {
        for p in parts.iter_mut() {
            if let FPart::Field { toks, spec_fields, span: s, .. } = p {
                *s = span;
                for x in toks.iter_mut() {
                    *x = respan(x.clone(), span);
                }
                for f in spec_fields.iter_mut() {
                    for x in f.iter_mut() {
                        *x = respan(x.clone(), span);
                    }
                }
            }
        }
    }
    t
}

/// The items a contract stands for (see the module doc).
fn lower(c: &Contract, sets: &[(String, String, Vec<Token>)]) -> R<Vec<Token>> {
    let e = c.enum_name();
    let name = &c.name;
    let mut g = Gen::new();
    // the enum of methods
    g.s(&format!("enum {e} {{\n"));
    for k in 0..c.methods.len() {
        g.s(&format!("    {}\n", c.variant(k)));
    }
    g.s(&format!("    {EVERY}\n"));
    for gr in c.groups() {
        g.s(&format!("    {}\n", Contract::every_variant(&gr)));
    }
    g.s("}\n");
    // the selection it starts with
    let start = match (c.start, &c.start_group) {
        (Some(k), _) => c.variant(k),
        (None, Some(gr)) => Contract::every_variant(gr),
        (None, None) => EVERY.to_string(),
    };
    g.s(&format!("{USING}{name} = {e}.{start}\n"));
    let params = |g: &mut Gen| {
        for (i, (p, ty)) in c.params.iter().enumerate() {
            if i > 0 {
                g.s(", ");
            }
            g.s(p);
            if !ty.is_empty() {
                g.s(": ");
                g.run(ty);
            }
        }
    };
    let ret = |g: &mut Gen| {
        if c.outputs.len() == 1 {
            g.s(" -> ");
            g.run(&c.outputs[0].1);
        }
    };
    let call = |g: &mut Gen, k: usize| {
        let m = &c.methods[k];
        g.s(&format!("{}(", m.name));
        let mut first = true;
        for (p, _) in &c.params {
            if !first {
                g.s(", ");
            }
            first = false;
            g.s(p);
        }
        for (b, v) in &m.binds {
            if !first {
                g.s(", ");
            }
            first = false;
            g.s(&format!("{b} = "));
            g.run(v);
        }
        for st in c.settings.iter().filter(|st| st.method == k) {
            if !first {
                g.s(", ");
            }
            first = false;
            g.s(&format!("{} = {}", st.name, setting_param(&m.name, &st.name)));
        }
        g.s(")");
    };
    let names = |k: usize| -> String { (0..c.outputs.len()).map(|o| format!("viv__c{k}_{o}")).collect::<Vec<_>>().join(", ") };
    // the dispatcher
    g.s(&format!("def {name}("));
    params(&mut g);
    g.s(&format!("{}{SEL}: {e} = {USING}{name}", if c.params.is_empty() { "" } else { ", " }));
    for st in &c.settings {
        g.s(&format!(", {}", setting_param(&c.methods[st.method].name, &st.name)));
        if !st.ty.is_empty() {
            g.s(": ");
            g.run(&st.ty);
        }
        g.s(" = ");
        g.run(&st.default);
    }
    g.s(")");
    ret(&mut g);
    g.s(" {\n");
    for r in &c.requires {
        g.s("    assert_true(");
        g.run(r);
        g.s(")\n");
    }
    // every method, and every method of each group: (variant, members)
    let mut runs: Vec<(String, Vec<usize>)> = Vec::new();
    if c.methods.len() > 1 {
        runs.push((EVERY.to_string(), (0..c.methods.len()).collect()));
    }
    for gr in c.groups() {
        runs.push((Contract::every_variant(&gr), (0..c.methods.len()).filter(|&k| c.methods[k].group.as_deref() == Some(gr.as_str())).collect()));
    }
    for (variant, members) in &runs {
        // the reference: the default when it is a member, else the first
        let lead = if members.contains(&c.default) { c.default } else { members[0] };
        let order: Vec<usize> = std::iter::once(lead).chain(members.iter().copied().filter(|&k| k != lead)).collect();
        g.s(&format!("    if viv__sel == {e}.{variant} {{\n"));
        for &k in &order {
            g.s(&format!("        {} = ", names(k)));
            call(&mut g, k);
            g.s("\n");
        }
        let median = c.median && order.len() > 1;
        if median {
            let all: Vec<String> = order.iter().map(|k| format!("viv__c{k}_0")).collect();
            g.s(&format!("        viv__all = [{}]\n        sort(viv__all)\n        viv__ref_0 = viv__all[{}]\n", all.join(", "), order.len() / 2));
        } else {
            for o in 0..c.outputs.len() {
                g.s(&format!("        viv__ref_{o} = viv__c{lead}_{o}\n"));
            }
        }
        let m0 = if median { "the median".to_string() } else { c.methods[lead].name.clone() };
        for a in &c.agree {
            let o = c.outputs.iter().position(|(x, _)| *x == a.output).expect("checked");
            let ty = type_text(&c.outputs[o].1);
            let d = format!("viv__ref_{o}");
            let others: Vec<usize> = if median { order.clone() } else { order.iter().skip(1).copied().collect() };
            for &k in &others {
                let mk = &c.methods[k].name;
                let x = format!("viv__c{k}_{o}");
                match (&a.tol, ty.as_str()) {
                    (Some(tol), "Num") => {
                        g.s("        viv__tol = ");
                        g.run(std::slice::from_ref(&tol.value));
                        if tol.relative {
                            g.s(&format!(" * max(1.0, abs({d}))"));
                        }
                        g.s(&format!("\n        viv__gap = abs({x} - {d})\n        if not (viv__gap <= viv__tol) {{\n"));
                        g.s(&format!("            stop(f\"contract {name}: {mk} disagrees with {m0} on {}: {{{x}}} and {{{d}}} are {{viv__gap}} apart, more than {{viv__tol}}\")\n        }}\n", a.output));
                    }
                    (Some(tol), _) => {
                        g.s("        viv__tol = ");
                        g.run(std::slice::from_ref(&tol.value));
                        if tol.relative {
                            g.s(&format!("\n        if numel({d}) > 0 {{\n            viv__tol = viv__tol * max(1.0, max(abs({d})))\n        }}"));
                        }
                        g.s(&format!("\n        if len(shape({x})) != len(shape({d})) or not all(shape({x}) == shape({d})) {{\n"));
                        g.s(&format!("            stop(f\"contract {name}: {mk} and {m0} give {} of different shapes, {{shape({x})}} and {{shape({d})}}\")\n        }}\n", a.output));
                        g.s(&format!("        viv__gap = 0.0\n        if numel({d}) > 0 {{\n            viv__gap = max(abs({x} - {d}))\n        }}\n        if not (viv__gap <= viv__tol) {{\n"));
                        g.s(&format!("            stop(f\"contract {name}: {mk} disagrees with {m0} on {}: {{viv__gap}} apart, more than {{viv__tol}}\")\n        }}\n", a.output));
                    }
                    (None, "Arr" | "[Int]" | "[Str]") => {
                        g.s(&format!("        if len({x}) != len({d}) or not all({x} == {d}) {{\n"));
                        g.s(&format!("            stop(f\"contract {name}: {mk} disagrees with {m0} on {}\")\n        }}\n", a.output));
                    }
                    (None, _) => {
                        g.s(&format!("        if {x} != {d} {{\n"));
                        g.s(&format!("            stop(f\"contract {name}: {mk} disagrees with {m0} on {}: {{{x}}} and {{{d}}}\")\n        }}\n", a.output));
                    }
                }
            }
        }
        let refs: Vec<String> = (0..c.outputs.len()).map(|o| if median { format!("viv__ref_{o}") } else { format!("viv__c{lead}_{o}") }).collect();
        g.s(&format!("        return {}\n    }}\n", refs.join(", ")));
    }
    if c.methods.len() > 1 {
        for k in (0..c.methods.len()).filter(|&k| k != c.default) {
            g.s(&format!("    if viv__sel == {e}.{} {{\n        return ", c.variant(k)));
            call(&mut g, k);
            g.s("\n    }\n");
        }
    }
    g.s("    return ");
    call(&mut g, c.default);
    g.s("\n}\n");
    // every method, as a def a phrase can call
    g.s(&format!("def {EVERY_DEF}{name}("));
    params(&mut g);
    g.s(")");
    ret(&mut g);
    g.s(&format!(" {{\n    return {name}("));
    for (p, _) in &c.params {
        g.s(&format!("{p}, "));
    }
    g.s(&format!("{SEL} = {e}.{EVERY}"));
    for (m, p, v) in sets {
        g.s(&format!(", {} = ", setting_param(m, p)));
        g.run(v);
    }
    g.s(")\n}\n");
    g.tokens(c.span)
}

/// Rewrite every contract declaration and selection in the tokens (see the
/// module doc). `outside`: defs the program imports whose text it does not
/// hold. Tokens with no contract come back unchanged.
pub fn desugar(toks: Vec<Token>, outside: &HashSet<String>) -> R<Vec<Token>> {
    if !any_contract(&toks) && !toks.iter().any(|t| matches!(&t.tok, Tok::Ident(w) if w == "using" || w == "settings")) {
        return Ok(toks);
    }
    let (decls, dropped) = read_all_dups(&toks)?;
    let sigs = def_sigs(&toks);
    let mut decls = decls;
    for (_, _, c) in decls.iter_mut() {
        c.settings = check(c, &sigs, outside)?;
    }
    let decls = decls;
    let by_name: HashMap<&str, &Contract> = decls.iter().map(|(_, _, c)| (c.name.as_str(), c)).collect();
    let (given, given_at) = read_settings(&toks)?;
    let globals = crate::phrases::assigned(&toks, true);
    // each setting must be one a method of the program's contracts takes
    for (m, list) in &given {
        let mine: Vec<&Contract> = decls.iter().map(|(_, _, c)| c).filter(|c| c.methods.iter().any(|x| x.name == *m)).collect();
        if mine.is_empty() {
            if decls.is_empty() || !imported_names(&toks).is_empty() {
                continue;
            }
            let (_, _, sp) = &list[0];
            return Err(err(*sp, format!("`{m}` is not a method of any contract here"), "not a method"));
        }
        for (p, _, sp) in list {
            let ok = mine.iter().any(|c| {
                let k = c.methods.iter().position(|x| x.name == *m).expect("listed");
                c.settings.iter().any(|st| st.method == k && st.name == *p)
            });
            if !ok {
                let k = mine[0].methods.iter().position(|x| x.name == *m).expect("listed");
                let names: Vec<&str> = mine[0].settings.iter().filter(|st| st.method == k).map(|st| st.name.as_str()).collect();
                let bound = mine[0].methods[k].binds.iter().any(|(b, _)| b == p);
                return Err(err(*sp, if bound {
                    format!("contract `{}` binds `{p}` of `{m}` in its `methods` line; `settings` cannot change it", mine[0].name)
                } else if names.is_empty() {
                    format!("`{m}` has no settings (parameters beyond the contract's, with a default)")
                } else {
                    format!("`{m}` has no setting `{p}`; its settings are {}", names.join(", "))
                }, "not a setting"));
            }
        }
    }
    // the settings a call of contract `n` passes (`c` its declaration)
    let sets_for = |c: Option<&Contract>| -> Vec<(String, String, Vec<Token>)> {
        let mut v = Vec::new();
        for (m, list) in &given {
            let Some(k) = c.map_or(Some(usize::MAX), |c| c.methods.iter().position(|x| x.name == *m)) else { continue };
            for (p, e, _) in list {
                if let Some(c) = c {
                    if !c.settings.iter().any(|st| st.method == k && st.name == *p) {
                        continue;
                    }
                }
                v.push((m.clone(), p.clone(), e.clone()));
            }
        }
        v.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        v
    };
    // a contract a module declares, imported by name: its enum and its
    // selection have the names the declaration gives them
    let imported: HashSet<String> = imported_names(&toks).into_iter().chain(outside.iter().cloned()).filter(|n| !by_name.contains_key(n.as_str())).collect();
    let mut out = Vec::with_capacity(toks.len() * 2);
    let mut k = 0;
    let mut depth = 0i32;
    let mut di = 0;
    // imported contracts this program has chosen a method for
    let mut unit_sel: HashSet<String> = HashSet::new();
    while k < toks.len() {
        if let Some(&(_, end)) = dropped.iter().chain(given_at.iter()).find(|(a, _)| *a == k) {
            k = end;
            continue;
        }
        if di < decls.len() && decls[di].0 == k {
            out.extend(lower(&decls[di].2, &sets_for(Some(&decls[di].2)))?);
            k = decls[di].1;
            di += 1;
            continue;
        }
        let line_start = k == 0 || matches!(toks[k - 1].tok, Tok::Newline | Tok::LBrace | Tok::RBrace);
        // `using name = method` or `using name = variable`: the selection
        // for every following call
        if line_start && word(toks.get(k), "using") && matches!(toks.get(k + 2).map(|t| &t.tok), Some(Tok::Eq)) {
            let n = ident(toks.get(k + 1)).unwrap_or_default();
            let local = by_name.get(n.as_str()).copied();
            // `using` starts no other statement: a name that is not one of
            // this text's contracts is one it imports (a module read on its
            // own does not see its import lines)
            if !n.is_empty() {
                if depth != 0 {
                    return Err(err(toks[k].span, format!("`using {n} = ...` is a top-level statement"), "inside a block")
                        .with_note(format!("to choose for one call, write `{n}(...) using <method>`")));
                }
                let (v, end) = call_using(&n, local, &globals, &toks, k + 3)?;
                let sp = toks[k].span.merge(toks[end - 1].span);
                let value = match v.strip_prefix('=') {
                    Some(var) => var.to_string(),
                    None => format!("{}.{v}", camel(&n)),
                };
                let mut g = Gen::new();
                match local {
                    // the contract's own selection, which every call reads
                    Some(c) => g.s(&format!("{USING}{} = {value}", c.name)),
                    // an imported contract: this program's selection, which
                    // its following calls pass
                    None => {
                        g.s(&format!("{UNIT_SEL}{n} = {value}"));
                        unit_sel.insert(n.clone());
                    }
                }
                out.extend(g.tokens(sp)?);
                k = end;
                continue;
            }
        }
        // `name(args)`, `name(args) using m`: the selection and the
        // program's `settings` for the contract's methods
        if let (Some(n), Some(Tok::LParen)) = (ident(toks.get(k)), toks.get(k + 1).map(|t| &t.tok)) {
            let prev_dot = k > 0 && toks[k - 1].tok == Tok::Dot;
            let local = by_name.get(n.as_str()).copied();
            let known = local.is_some() || imported.contains(&n) || unit_sel.contains(&n);
            if known && !prev_dot && !matches!(toks.get(k.wrapping_sub(1)).map(|t| &t.tok), Some(Tok::Def)) {
                let (_, close) = until(&toks, k + 2, &[Tok::RParen]);
                let using = word(toks.get(close + 1), "using");
                // a module read on its own sees every imported name as a
                // possible contract; the program's `settings` go only to one
                // it knows is: declared here, chosen for (`using`), or
                // imported with its enum
                let contract_here = local.is_some() || using || unit_sel.contains(&n) || imported.contains(&camel(&n));
                let sets = if contract_here { sets_for(local) } else { Vec::new() };
                let chosen = unit_sel.contains(&n) && local.is_none();
                if using || chosen || !sets.is_empty() {
                    let (sel, end) = if using {
                        let (v, e) = call_using(&n, local, &globals, &toks, close + 2)?;
                        (Some(v), e)
                    } else if chosen {
                        (Some(format!("={UNIT_SEL}{n}")), close + 1)
                    } else {
                        (None, close + 1)
                    };
                    let sp = toks[k].span;
                    let mut g = Gen::new();
                    let e = camel(&n);
                    g.s(&format!("{n}("));
                    let args = &toks[k + 2..close];
                    let mut first = args.is_empty();
                    if !args.is_empty() {
                        g.run(args);
                    }
                    let mut more = |g: &mut Gen| {
                        if !first {
                            g.s(", ");
                        }
                        first = false;
                    };
                    match sel {
                        Some(v) if v.starts_with('=') => {
                            more(&mut g);
                            g.s(&format!("{SEL} = {}", &v[1..]));
                        }
                        Some(v) => {
                            more(&mut g);
                            g.s(&format!("{SEL} = {e}.{v}"));
                        }
                        None => {}
                    }
                    for (m, name, v) in &sets {
                        more(&mut g);
                        g.s(&format!("{} = ", setting_param(m, name)));
                        g.run(v);
                    }
                    g.s(")");
                    out.extend(g.tokens(sp)?);
                    k = end;
                    continue;
                }
            }
        }
        match toks[k].tok {
            Tok::LBrace => depth += 1,
            Tok::RBrace => depth -= 1,
            _ => {}
        }
        out.push(toks[k].clone());
        k += 1;
    }
    Ok(out)
}

/// What follows `using` after a call of contract `n`: a method, `every
/// method`, or a top-level variable holding the contract's enum value.
/// Returns the selection (a variant, or `=name` for a variable) and where it
/// ends. `c` is the declaration, when the program holds it.
fn call_using(n: &str, c: Option<&Contract>, globals: &HashSet<String>, toks: &[Token], j: usize) -> R<(String, usize)> {
    let at = toks.get(j).or(toks.last()).map(|t| t.span).expect("the tokens end in Eof");
    if let Some(sel) = every_at(toks, j, c)? {
        return Ok(sel);
    }
    let Some(m) = ident(toks.get(j)) else {
        return Err(err(at, format!("`using` after a call of `{n}` names a method, `every method`, or a variable holding the method"), "here"));
    };
    if matches!(toks.get(j + 1).map(|t| &t.tok), Some(Tok::LParen)) {
        return Err(err(at, format!("`using` only chooses a method; give `{m}` its settings in a top-level `settings {m}(name = value)` statement"), "settings here"));
    }
    let method = match c {
        Some(c) => c.methods.iter().any(|x| x.name == m),
        // an imported contract: a name the program assigns at the top level
        // is a variable holding the selection, any other a method
        None => !globals.contains(&m),
    };
    if method {
        return Ok((camel(&m), j + 1));
    }
    if let Some(c) = c.filter(|_| !globals.contains(&m)) {
        let names: Vec<&str> = c.methods.iter().map(|x| x.name.as_str()).collect();
        return Err(err(at, format!("contract `{n}` has no method `{m}`; its methods are {} (or `every method`, or a variable holding the method)", names.join(", ")), "not a method"));
    }
    Ok((format!("={m}"), j + 1))
}

/// The `settings m(name = e, ...)` statements: method -> (setting, value,
/// where), and the token ranges they cover.
fn read_settings(toks: &[Token]) -> R<(HashMap<String, Vec<(String, Vec<Token>, Span)>>, Vec<(usize, usize)>)> {
    let mut out: HashMap<String, Vec<(String, Vec<Token>, Span)>> = HashMap::new();
    let mut ranges = Vec::new();
    let mut depth = 0i32;
    let mut k = 0;
    while k < toks.len() {
        let line_start = k == 0 || matches!(toks[k - 1].tok, Tok::Newline | Tok::LBrace | Tok::RBrace);
        if line_start && word(toks.get(k), "settings") && ident(toks.get(k + 1)).is_some() && matches!(toks.get(k + 2).map(|t| &t.tok), Some(Tok::LParen)) {
            let m = ident(toks.get(k + 1)).expect("checked");
            if depth != 0 {
                return Err(err(toks[k].span, format!("`settings {m}(...)` is a top-level statement"), "inside a block"));
            }
            let mut j = k + 3;
            loop {
                if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::RParen)) {
                    j += 1;
                    break;
                }
                let (Some(p), Some(Tok::Eq)) = (ident(toks.get(j)), toks.get(j + 1).map(|t| &t.tok)) else {
                    return Err(err(toks.get(j).map_or(toks[k].span, |t| t.span), format!("`settings {m}(...)` gives settings of `{m}`, each `name = value`"), "expected `name = value`"));
                };
                let (e, end) = until(toks, j + 2, &[Tok::Comma, Tok::RParen]);
                if e.is_empty() {
                    return Err(err(toks[j].span, format!("`{p} =` needs a value"), "here"));
                }
                let list = out.entry(m.clone()).or_default();
                if list.iter().any(|(q, _, _)| *q == p) {
                    return Err(err(toks[j].span, format!("`{m}`'s setting `{p}` is given twice"), "again here"));
                }
                list.push((p, e, toks[j].span));
                j = end;
                if matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Comma)) {
                    j += 1;
                }
            }
            ranges.push((k, j));
            k = j;
            continue;
        }
        match toks[k].tok {
            Tok::LBrace => depth += 1,
            Tok::RBrace => depth -= 1,
            _ => {}
        }
        k += 1;
    }
    Ok((out, ranges))
}

/// `--using name=method` (or `name=every`): make the contract `name` in the
/// program's text start with that selection, as `--set` sets a parameter.
/// An unknown contract or method is an error, naming what there is.
pub fn apply_using(src: &str, name: &str, method: &str) -> Result<String, String> {
    let all = scan(src);
    let Some(c) = all.iter().find(|c| c.name == name) else {
        let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        return Err(if names.is_empty() {
            format!("--using {name}={method}: the program declares no contract")
        } else {
            format!("--using {name}={method}: the program has no contract `{name}`; its contracts are {}", names.join(", "))
        });
    };
    let group = method.strip_prefix("every-").or_else(|| method.strip_prefix("every ")).map(str::to_string);
    let clause = if method == "every" {
        "using every method".to_string()
    } else if let Some(g) = group.filter(|g| c.groups().contains(g)) {
        format!("using every {g} method")
    } else if c.methods.iter().any(|m| m.name == method) {
        format!("using {method}")
    } else {
        let names: Vec<&str> = c.methods.iter().map(|m| m.name.as_str()).collect();
        let groups: Vec<String> = c.groups().iter().map(|g| format!("every-{g}")).collect();
        let more = if groups.is_empty() { String::new() } else { format!(", {}", groups.join(", ")) };
        return Err(format!("--using {name}={method}: contract `{name}` has the methods {} (or `every`{more})", names.join(", ")));
    };
    // the declaration's lines: from `contract name(` to its closing `}`
    let mut lines: Vec<String> = src.lines().map(str::to_string).collect();
    let head = format!("contract {name}(");
    let Some(first) = lines.iter().position(|l| l.trim_start().starts_with(&head)) else {
        return Err(format!("--using {name}={method}: cannot find the line `{head}...`"));
    };
    let mut depth = 0i32;
    let mut last = None;
    for (i, l) in lines.iter().enumerate().skip(first) {
        for ch in strip_strings(l).chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        if depth == 0 && strip_strings(l).contains('}') {
            last = Some(i);
            break;
        }
    }
    let Some(last) = last else { return Err(format!("--using {name}={method}: contract `{name}` has no closing `}}`")) };
    let indent = "    ";
    match (first + 1..last).find(|&i| lines[i].trim_start().starts_with("using ")) {
        Some(i) => lines[i] = format!("{indent}{clause}"),
        None => lines.insert(last, format!("{indent}{clause}")),
    }
    let mut out = lines.join("\n");
    if src.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

/// A line with its string literals and comment blanked, for counting braces.
fn strip_strings(l: &str) -> String {
    let mut out = String::new();
    let mut quoted = false;
    let mut prev = ' ';
    for ch in l.chars() {
        if quoted {
            if ch == '"' && prev != '\\' {
                quoted = false;
            }
        } else if ch == '"' {
            quoted = true;
        } else if ch == '#' {
            break;
        } else {
            out.push(ch);
        }
        prev = ch;
    }
    out
}

/// The contracts as JSON, for the code map and vipyr: name, intent, the
/// signature, the methods (canonical first), the default and the selection
/// it starts with, and what the methods must agree on.
pub fn json(c: &Contract, line: usize) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " "));
    let params: Vec<String> = c.params.iter().map(|(p, t)| format!("{{\"name\":{},\"type\":{}}}", q(p), q(&type_text(t)))).collect();
    let outputs: Vec<String> = c.outputs.iter().map(|(p, t)| format!("{{\"name\":{},\"type\":{}}}", q(p), q(&type_text(t)))).collect();
    let mut order: Vec<usize> = vec![c.default];
    order.extend((0..c.methods.len()).filter(|&k| k != c.default));
    let methods: Vec<String> = order.iter().map(|&k| q(&c.methods[k].name)).collect();
    let agree: Vec<String> = c.agree.iter().map(|a| {
        let tol = match &a.tol {
            Some(t) => q(&t.text()),
            None => "null".to_string(),
        };
        format!("{{\"output\":{},\"within\":{tol}}}", q(&a.output))
    }).collect();
    let groups: Vec<String> = c.groups().iter().map(|gr| {
        let ms: Vec<String> = c.methods.iter().filter(|m| m.group.as_deref() == Some(gr.as_str())).map(|m| q(&m.name)).collect();
        format!("{{\"label\":{},\"methods\":[{}]}}", q(gr), ms.join(","))
    }).collect();
    format!(
        "{{\"name\":{},\"line\":{line},\"intent\":{},\"params\":[{}],\"outputs\":[{}],\"methods\":[{}],\"groups\":[{}],\"default\":{},\"using\":{},\"agree\":[{}],\"enum\":{},\"selector\":{}}}",
        q(&c.name), q(&c.intent), params.join(","), outputs.join(","), methods.join(","), groups.join(","), q(&c.methods[c.default].name), q(&c.start_label()), agree.join(","), q(&c.enum_name()), q(&format!("{USING}{}", c.name))
    )
}

/// An expression's tokens as text (for showing a tolerance).
pub fn expr_text(toks: &[Token]) -> String {
    let mut s = String::new();
    for t in toks {
        let piece = match &t.tok {
            Tok::Ident(n) => n.clone(),
            Tok::Num(v, Some(i)) if (*i as f64) == *v => i.to_string(),
            Tok::Num(v, _) => format!("{v:e}").replace("e0", "").to_string(),
            Tok::Plus => "+".into(),
            Tok::Minus => "-".into(),
            Tok::Star => "*".into(),
            Tok::Slash => "/".into(),
            Tok::Caret => "^".into(),
            Tok::LParen => "(".into(),
            Tok::RParen => ")".into(),
            Tok::LBracket => "[".into(),
            Tok::RBracket => "]".into(),
            Tok::Comma => ",".into(),
            Tok::Dot => ".".into(),
            Tok::Lt => "<".into(),
            Tok::Gt => ">".into(),
            Tok::Le => "<=".into(),
            Tok::Ge => ">=".into(),
            Tok::Str(v) => format!("\"{v}\""),
            other => format!("{other:?}"),
        };
        let tight = matches!(t.tok, Tok::LParen | Tok::RParen | Tok::Comma | Tok::Dot | Tok::LBracket | Tok::RBracket)
            || s.ends_with('(') || s.ends_with('[') || s.ends_with('.') || s.is_empty();
        if !tight {
            s.push(' ');
        }
        s.push_str(&piece);
    }
    s
}

/// The names bound by the program's import lines (`from m import a, b as
/// c`, `import m`): defs whose text a module read on its own does not hold.
pub fn imported_names(toks: &[Token]) -> Vec<String> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < toks.len() {
        let line_start = k == 0 || toks[k - 1].tok == Tok::Newline;
        if line_start && (word(toks.get(k), "from") || word(toks.get(k), "import")) {
            let mut j = k;
            while j < toks.len() && !matches!(toks[j].tok, Tok::Newline | Tok::Eof) {
                if let Tok::Ident(n) = &toks[j].tok {
                    out.push(n.clone());
                }
                j += 1;
            }
            k = j;
        }
        k += 1;
    }
    out
}
