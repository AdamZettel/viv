// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Recursive-descent parser: tokens -> AST, aligned with the real Viv surface
//! (`def` blocks, statements terminated by newlines). Errors are diagnostics
//! (1.7). For v1 there is no error recovery; the first error stops the parse.

use knotkit::diag::Diagnostic;
use knotkit::span::{SourceId, Span};

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::lexer::{Tok, Token};
use crate::surface::{placeholder, Lifted};

pub fn parse(toks: Vec<Token>, source: SourceId) -> Result<Program, Vec<Diagnostic>> {
    parse_with(toks, source, &[])
}

/// `parse`, knowing the `phrase` declarations `outside` (one per string) of
/// the modules the program uses but does not have in its text: a module read
/// on its own reads the phrases of the modules it uses (`crate::used_phrases`).
pub fn parse_with(toks: Vec<Token>, source: SourceId, outside: &[String]) -> Result<Program, Vec<Diagnostic>> {
    parse_with_sigs(toks, source, outside, &HashMap::new())
}

/// `parse_with`, knowing also the signatures of the defs the text imports by
/// name (`from m import f`) but does not hold (`crate::used_sigs`): a module
/// read on its own may call them with keyword arguments, as the spliced
/// program does (B274).
pub fn parse_with_sigs(toks: Vec<Token>, _source: SourceId, outside: &[String], imported: &HashMap<String, Vec<crate::kwargs::SigParam>>) -> Result<Program, Vec<Diagnostic>> {
    // contracts (crate::contracts): written as the enum, defs and selections
    // every engine runs; tokens with none come back unchanged
    let foreign: HashSet<String> = imported.keys().cloned().chain(crate::contracts::imported_names(&toks)).collect();
    let toks = crate::contracts::desugar(without_imports(toks), &foreign).map_err(|d| vec![d])?;
    let mut defs = def_names(&toks);
    let mut sigs = crate::kwargs::def_sigs(&toks);
    for (name, sig) in imported {
        if !sigs.contains_key(name) {
            defs.insert(name.clone());
            sigs.insert(name.clone(), sig.clone());
        }
    }
    let globals = crate::phrases::assigned(&toks, true);
    let enums = enum_names(&toks);
    let mut p = Parser::new(toks, defs);
    p.globals = globals;
    p.enums = enums;
    p.sigs = sigs;
    for decl in outside {
        let toks = crate::lexer::lex(decl, SourceId(u32::MAX))?;
        let mut q = Parser::new(toks, HashSet::new());
        let mut u = q.phrase_decl().map_err(|d| vec![d])?;
        u.body_text = decl.get(u.body_span.start as usize..u.body_span.end as usize).map(str::to_string);
        if !p.user_phrases.iter().any(|v| crate::phrases::shape_key(&v.words) == crate::phrases::shape_key(&u.words)) {
            p.user_phrases.push(u);
        }
    }
    p.declare_phrases().map_err(|d| vec![d])?;
    let mut program = p.program().map_err(|d| vec![d])?;
    program.phrases = std::mem::take(&mut p.phrases);
    // comprehensions, `a if c else b` and `key=` (crate::surface): a program
    // with none comes back unchanged
    let program = crate::surface::lift(program, std::mem::take(&mut p.lifted), p.hidden, p.defs).map_err(|d| vec![d])?;
    // record field defaults and constructor defs (crate::ctor)
    crate::ctor::complete(program)
}

/// A module's text read on its own (by a lens, the architecture check, the
/// graph): its import statements are dropped, since what they mean is the
/// splice's business (crate::imports), and `private` before a top-level
/// definition is dropped, since privacy is checked there too. A spliced
/// program has neither left.
fn without_imports(toks: Vec<Token>) -> Vec<Token> {
    let word = |t: Option<&Token>, w: &str| matches!(t.map(|t| &t.tok), Some(Tok::Ident(s)) if s == w);
    let starts = |toks: &[Token], k: usize| k == 0 || matches!(toks[k - 1].tok, Tok::Newline);
    if !toks.iter().any(|t| matches!(&t.tok, Tok::Ident(s) if s == "import" || s == "private")) {
        return toks;
    }
    let mut out = Vec::with_capacity(toks.len());
    let mut k = 0;
    while k < toks.len() {
        if starts(&toks, k) {
            let import = word(toks.get(k), "import") && matches!(toks.get(k + 1).map(|t| &t.tok), Some(Tok::Ident(_)));
            let from = word(toks.get(k), "from") && matches!(toks.get(k + 1).map(|t| &t.tok), Some(Tok::Ident(_))) && {
                let mut j = k + 1;
                while matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Ident(_) | Tok::Dot)) && !word(toks.get(j), "import") {
                    j += 1;
                }
                word(toks.get(j), "import")
            };
            if import || from {
                while k < toks.len() && !matches!(toks[k].tok, Tok::Newline | Tok::Eof) {
                    k += 1;
                }
                if matches!(toks.get(k).map(|t| &t.tok), Some(Tok::Newline)) {
                    k += 1;
                }
                continue;
            }
            if word(toks.get(k), "private") && matches!(toks.get(k + 1).map(|t| &t.tok), Some(Tok::Def | Tok::Record | Tok::Ident(_))) {
                k += 1;
                continue;
            }
        }
        out.push(toks[k].clone());
        k += 1;
    }
    out
}

/// The program's enum types (`enum Name { ... }`), read from the tokens
/// before parsing, so `Name.Variant(...)` is a value wherever it is written.
fn enum_names(toks: &[Token]) -> HashSet<String> {
    toks.windows(3)
        .filter_map(|w| match (&w[0].tok, &w[1].tok, &w[2].tok) {
            (Tok::Ident(e), Tok::Ident(n), Tok::LBrace) if e == "enum" => Some(n.clone()),
            _ => None,
        })
        .collect()
}

/// The names the program defines with `def`, read from the tokens before
/// parsing, so a builtin the program replaces (`print`, `rows`, ...) is known
/// wherever it is called.
fn def_names(toks: &[Token]) -> HashSet<String> {
    toks.windows(2)
        .filter_map(|w| match (&w[0].tok, &w[1].tok) {
            (Tok::Def, Tok::Ident(n)) => Some(n.clone()),
            _ => None,
        })
        .collect()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// When true, a bare `Name { ... }` is *not* parsed as a record literal —
    /// set while parsing an `if`/`while`/`for` head, where the `{` opens the body
    /// block (Rust's "no struct literal in condition" rule). Reset inside any
    /// parenthesized/bracketed sub-expression.
    no_record_lit: bool,
    /// The def whose body is being parsed (`<main>` outside any def), the
    /// number of steps open around the current statement, and the step ids
    /// already used in this def: what a `step` records about where it sits.
    cur_def: String,
    step_depth: u32,
    step_ids: Vec<String>,
    /// Inside a `parallel for` body. A step there is refused: its events
    /// would arrive from several threads in no fixed order, so the compiled
    /// program could not record the sequence the interpreter records.
    in_parallel: u32,
    /// How many hidden variables the desugarings below have made. Each gets
    /// its own name (`viv__t1`, `viv__t2`, ...), so two of them never share a
    /// type or a value.
    hidden: u32,
    /// The program's defs (`def_names`).
    defs: HashSet<String>,
    /// The values computed by statements (comprehensions, if-expressions,
    /// `key=` calls), by the number their placeholder carries; `crate::surface`
    /// puts them in place.
    lifted: Vec<Lifted>,
    /// The names the def being parsed assigns, with its parameters, and the
    /// names the top level assigns (`crate::phrases::assigned`): the names a
    /// phrase's `fn` hole reads from the program rather than takes as its
    /// variable.
    scope: HashSet<String>,
    globals: HashSet<String>,
    /// The phrases read so far (`crate::phrases::PhraseUse`).
    phrases: Vec<crate::phrases::PhraseUse>,
    /// The program's own phrases, `phrase "the norm of M" -> sqrt(sum(M * M))`,
    /// read before anything else so a use may come before the declaration.
    user_phrases: Vec<UserPhrase>,
    /// Where the integer literals are, and their values (`Program::whole`).
    whole: HashMap<(u32, u32), i64>,
    /// The program's enum types (`enum_names`).
    enums: HashSet<String>,
    /// How many `match` statements have been read: each names its hidden
    /// variables by its number.
    matches: u32,
    /// Every def's parameters and their defaults (crate::kwargs::def_sigs):
    /// a call is filled out from them.
    sigs: HashMap<String, Vec<crate::kwargs::SigParam>>,
}

thread_local! {
    /// Whether a builtin's options may still be written by position: only
    /// `viv fix --keywords`, reading a file to migrate it, sets this.
    pub static POSITIONAL_OPTIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A phrase a program declares: its words in the table's notation (a hole
/// for each word that is a name the body reads), the body, where the
/// declaration and its body are.
#[derive(Clone)]
struct UserPhrase {
    words: String,
    body: Expr,
    span: Span,
    body_span: Span,
    /// `given theta, m`: names the body reads from the place the phrase is
    /// used, not from the program's top level
    given: Vec<(String, Span)>,
    /// the body's text, for a declaration read from a used module
    body_text: Option<String>,
}

/// The prefix of the variables a desugaring introduces (the index
/// temporaries of `a[i] += x`, the counter and bounds of `for i from a to b
/// by s`). Ordinary identifiers, so every engine runs them as it runs any
/// variable; the prefix keeps them out of the way of a program's own names.
pub const HIDDEN_PREFIX: &str = "viv__";

/// The def name a top-level step is recorded under.
pub const MAIN_DEF: &str = "<main>";

type PResult<T> = Result<T, Diagnostic>;

/// The calls strings add (`crate::interp::strings`): an f-string, `a in b`,
/// and the three ways an f-string field is written.
const FSTR: &str = crate::interp::strings::FSTR;
const IN: &str = crate::interp::strings::IN;
const STR_OF: &str = crate::interp::strings::STR_OF;
const REPR_OF: &str = crate::interp::strings::REPR_OF;
const FMT_OF: &str = crate::interp::strings::FMT_OF;

/// The calls a slice becomes (`crate::interp::slices` has the rules).
const SLICE_GET: &str = crate::interp::slices::GET;
const SLICE_SET: &str = crate::interp::slices::SET;

/// A `for` head (`Parser::for_head`), in a statement or a comprehension.
#[derive(Clone)]
enum Head {
    /// `i to n`, `i from a to b [by s] [reversed]`
    Count { var: String, var_span: Span, from: Option<Expr>, from_span: Span, bound: Expr, by: Option<Expr>, by_span: Span, reversed: bool, rev_span: Span },
    /// `x in e`, `i, x in enumerate(e[, start])`, `a, b in zip(e, f)`
    Each { names: Vec<(String, Span)>, over: Over, in_span: Span },
}

/// What the names of a `for x in ...` head walk over.
#[derive(Clone)]
enum Over {
    One(Src),
    /// the sequence and the index of its first element (0 when left out)
    Enumerate(Src, Option<Expr>),
    Zip(Vec<Src>),
}

/// A sequence a `for` walks: its elements, or with `rows(M)` its rows.
#[derive(Clone)]
struct Src {
    e: Expr,
    rows: bool,
}

/// One clause of a comprehension after its element.
enum Clause {
    For(Head),
    If(Expr),
}

/// What a comprehension makes.
#[derive(Clone, Copy, PartialEq)]
enum Collect {
    /// `[e for ...]`: the array (or list of strings) of the elements
    List,
    /// `all(c for ...)`, `any(c for ...)`: a boolean, computed as far as it
    /// takes to decide it, as in Python
    All,
    Any,
}

/// One item of an index list.
enum IndexItem {
    /// `i`
    At(Expr),
    /// `lo:hi:step`, any part left out
    Range { lo: Option<Expr>, hi: Option<Expr>, step: Option<Expr>, span: Span },
}

/// The index list of a slice as the array literal its builtin takes: four
/// numbers per item, `[kind, lo, hi, step]`, kind 0 for an index (in `lo`),
/// 1 for a range with an end, 2 for a range to the end of its axis.
fn slice_spec(items: Vec<IndexItem>, span: Span) -> Expr {
    let num = |value: f64, span: Span| Expr::Num { value, span };
    let mut elems = Vec::with_capacity(4 * items.len());
    for item in items {
        match item {
            IndexItem::At(x) => {
                let sp = x.span();
                elems.extend([num(0.0, sp), x, num(0.0, sp), num(1.0, sp)]);
            }
            IndexItem::Range { lo, hi, step, span: sp } => {
                // A start left out is 0 for a step of 1 or more. With a step
                // that is not a positive number, a string or a list of strings
                // starts from its end (Python's `s[::-1]`), so the spec says
                // the start was left out: kind 3 (an end written) or 4.
                let positive = |e: &Expr| matches!(e, Expr::Num { value, .. } if *value > 0.0);
                let open_start = lo.is_none() && step.as_ref().is_some_and(|s| !positive(s));
                let kind = match (open_start, hi.is_some()) {
                    (false, true) => 1.0,
                    (false, false) => 2.0,
                    (true, true) => 3.0,
                    (true, false) => 4.0,
                };
                elems.extend([
                    num(kind, sp),
                    lo.unwrap_or_else(|| num(0.0, sp)),
                    hi.unwrap_or_else(|| num(0.0, sp)),
                    step.unwrap_or_else(|| num(1.0, sp)),
                ]);
            }
        }
    }
    Expr::ArrLit { elems, span }
}

impl Parser {
    fn new(toks: Vec<Token>, defs: HashSet<String>) -> Parser {
        Parser {
            toks,
            pos: 0,
            no_record_lit: false,
            cur_def: MAIN_DEF.to_string(),
            step_depth: 0,
            step_ids: Vec::new(),
            in_parallel: 0,
            hidden: 0,
            defs,
            lifted: Vec::new(),
            scope: HashSet::new(),
            globals: HashSet::new(),
            phrases: Vec::new(),
            user_phrases: Vec::new(),
            whole: HashMap::new(),
            enums: HashSet::new(),
            matches: 0,
            sigs: HashMap::new(),
        }
    }

    fn peek(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn span(&self) -> Span {
        self.peek().span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn at(&self, t: &Tok) -> bool {
        &self.peek().tok == t
    }

    fn skip_newlines(&mut self) {
        while self.at(&Tok::Newline) {
            self.bump();
        }
    }

    fn expect(&mut self, t: Tok, what: &str) -> PResult<Token> {
        if self.peek().tok == t {
            Ok(self.bump())
        } else {
            Err(self.err_here(format!("expected {}", what)))
        }
    }

    fn err_here(&self, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::error(msg).with_primary(self.span(), format!("found {}", describe(&self.peek().tok)))
    }

    fn program(&mut self) -> PResult<Program> {
        let mut items = Vec::new();
        let mut debug = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::Eof) {
                break;
            }
            if self.at_phrase_decl() {
                // read already, by `declare_phrases`
                let snap = (self.lifted.len(), self.hidden, self.phrases.len());
                self.phrase_decl()?;
                self.lifted.truncate(snap.0);
                self.hidden = snap.1;
                self.phrases.truncate(snap.2);
                continue;
            }
            if self.at_debug() {
                debug.push(self.debug_block()?);
            } else if self.at_placeholder() {
                items.push(Item::Def(self.placeholder_def()?));
            } else if self.at(&Tok::Record) {
                items.push(Item::Record(self.record_decl()?));
            } else if self.at_enum() {
                items.push(Item::Enum(self.enum_decl()?));
            } else if self.at_kernel() {
                items.push(Item::Def(self.kernel_def()?));
            } else if self.at_schedule() {
                items.push(Item::Stmt(self.schedule()?));
            } else if self.at(&Tok::Def) {
                items.push(Item::Def(self.def()?));
            } else if self.at(&Tok::Rung) {
                items.push(Item::Rung(self.rung()?));
            } else {
                let mut stmts = Vec::new();
                self.stmt_into(&mut stmts)?;
                items.extend(stmts.into_iter().map(Item::Stmt));
            }
        }
        Ok(Program { items, debug, memory: Vec::new(), phrases: Vec::new(), whole: std::mem::take(&mut self.whole) })
    }

    /// `rung "title" { items }` -- one step of a ladder (§ ladders). The items
    /// are what a top level holds, less another rung.
    fn rung(&mut self) -> PResult<Rung> {
        let start = self.span();
        self.expect(Tok::Rung, "`rung`")?;
        let title = match &self.peek().tok {
            Tok::Str(s) => { let s = s.clone(); self.bump(); s }
            _ => return Err(self.err_here("a rung is written `rung \"what this rung adds\" { ... }`")),
        };
        self.expect(Tok::LBrace, "`{`")?;
        let body_start = self.span();
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) {
                break;
            }
            if self.at(&Tok::Eof) {
                return Err(self.err_here("this rung is never closed: expected `}`"));
            }
            if self.at(&Tok::Rung) {
                return Err(self.err_here("a rung cannot hold another rung"));
            }
            if self.at_placeholder() {
                items.push(Item::Def(self.placeholder_def()?));
            } else if self.at(&Tok::Record) {
                items.push(Item::Record(self.record_decl()?));
            } else if self.at_enum() {
                items.push(Item::Enum(self.enum_decl()?));
            } else if self.at_kernel() {
                items.push(Item::Def(self.kernel_def()?));
            } else if self.at_schedule() {
                items.push(Item::Stmt(self.schedule()?));
            } else if self.at(&Tok::Def) {
                items.push(Item::Def(self.def()?));
            } else {
                let mut stmts = Vec::new();
                self.stmt_into(&mut stmts)?;
                items.extend(stmts.into_iter().map(Item::Stmt));
            }
        }
        let close = self.expect(Tok::RBrace, "`}`")?;
        Ok(Rung { title, items, body_span: body_start.merge(close.span), span: start.merge(close.span) })
    }

    /// Whether the next tokens open a debug block: `debug {`, `debug name {`
    /// or `debug * {`. `debug` is a word only in this position.
    fn at_debug(&self) -> bool {
        let tok = |k: usize| self.toks.get(self.pos + k).map(|t| &t.tok);
        matches!(tok(0), Some(Tok::Ident(s)) if s == "debug")
            && (matches!(tok(1), Some(Tok::LBrace))
                || (matches!(tok(1), Some(Tok::Ident(_)) | Some(Tok::Star)) && matches!(tok(2), Some(Tok::LBrace))))
    }

    /// Whether the next tokens are `kernel def`.
    fn at_kernel(&self) -> bool {
        let tok = |k: usize| self.toks.get(self.pos + k).map(|t| &t.tok);
        matches!(tok(0), Some(Tok::Ident(s)) if s == "kernel") && matches!(tok(1), Some(Tok::Def))
    }

    /// `kernel def name(params) { body }`: a def marked as a GPU kernel
    /// (crate::cuda rewrites it).
    fn kernel_def(&mut self) -> PResult<Def> {
        let start = self.span();
        self.bump();
        let mut d = self.def()?;
        d.span = start.merge(d.span);
        crate::cuda::mark(&mut d);
        Ok(d)
    }

    /// Whether the next tokens are `on device name`.
    /// `on` and `device` are words only in this position.
    fn at_on_device(&self) -> bool {
        let tok = |k: usize| self.toks.get(self.pos + k).map(|t| &t.tok);
        matches!(tok(0), Some(Tok::Ident(s)) if s == "on")
            && matches!(tok(1), Some(Tok::Ident(s)) if s == "device")
            && matches!(tok(2), Some(Tok::Ident(_)))
    }

    /// `on device a, b { body }`: the arrays live on the device while the
    /// body runs. Kept as `if viv__device(a, b) { body }`, which crate::cuda
    /// checks and rewrites (`cuda::DEVICE`).
    fn on_device(&mut self) -> PResult<Stmt> {
        let start = self.span();
        self.bump();
        self.bump();
        let mut args = Vec::new();
        loop {
            let (name, span) = self.ident("the name of an array to keep on the device")?;
            args.push(Expr::Var { name, span });
            if !self.at(&Tok::Comma) {
                break;
            }
            self.bump();
        }
        let head = start.merge(args.last().map(|a| a.span()).unwrap_or(start));
        self.expect(Tok::LBrace, "`{` after the arrays of `on device`")?;
        let body = self.block()?;
        let close = self.expect(Tok::RBrace, "`}`")?;
        let cond = Expr::Call { callee: crate::cuda::DEVICE.into(), args, span: head };
        Ok(Stmt::If { cond, then_body: body, else_body: Vec::new(), span: start.merge(close.span) })
    }

    /// Whether the next tokens are `schedule name(` (a kernel's grid).
    fn at_schedule(&self) -> bool {
        let tok = |k: usize| self.toks.get(self.pos + k).map(|t| &t.tok);
        matches!(tok(0), Some(Tok::Ident(s)) if s == "schedule") && matches!(tok(1), Some(Tok::Ident(_))) && matches!(tok(2), Some(Tok::LParen))
    }

    /// `schedule name(key = value, ...)`, a top-level line: kept as the
    /// statement crate::cuda reads and removes (`cuda::SCHEDULE`, the name
    /// and each key as strings, each value after its key).
    fn schedule(&mut self) -> PResult<Stmt> {
        let start = self.span();
        self.bump();
        let (name, name_span) = self.ident("the name of the def the schedule is for")?;
        self.expect(Tok::LParen, "`(`")?;
        let mut args = vec![Expr::Str { value: name, span: name_span }];
        while !self.at(&Tok::RParen) {
            let (key, key_span) = self.ident("`name = value`")?;
            self.expect(Tok::Eq, "`=` after the schedule's key")?;
            args.push(Expr::Str { value: key, span: key_span });
            args.push(self.expr()?);
            if !self.at(&Tok::RParen) {
                self.expect(Tok::Comma, "`,` or `)`")?;
            }
        }
        let close = self.expect(Tok::RParen, "`)`")?;
        let span = start.merge(close.span);
        Ok(Stmt::ExprStmt { expr: Expr::Call { callee: crate::cuda::SCHEDULE.into(), args, span }, span })
    }

    /// Whether the next tokens are `placeholder def`.
    fn at_placeholder(&self) -> bool {
        let tok = |k: usize| self.toks.get(self.pos + k).map(|t| &t.tok);
        matches!(tok(0), Some(Tok::Ident(s)) if s == "placeholder") && matches!(tok(1), Some(Tok::Def))
    }

    fn at_word(&self, k: usize, word: &str) -> bool {
        matches!(self.toks.get(self.pos + k).map(|t| &t.tok), Some(Tok::Ident(s)) if s == word)
    }

    /// `debug [name | *] { items }`, one item per line:
    ///
    /// ```text
    /// watch [per iteration | at <step>] a, b, name = expr
    /// hold  [per iteration | at <step>] <expr> [within <tol>] "<sentence>"
    /// trace f, g
    /// ```
    ///
    /// A block without a name belongs to the def that follows it.
    fn debug_block(&mut self) -> PResult<DebugBlock> {
        let start = self.span();
        self.bump(); // `debug`
        let (def, def_span, unnamed) = match &self.peek().tok {
            Tok::LBrace => (None, start, true),
            Tok::Star => {
                let sp = self.span();
                self.bump();
                (None, sp, false)
            }
            _ => {
                let (n, sp) = self.ident("the def the debug block is about")?;
                (Some(n), sp, false)
            }
        };
        self.expect(Tok::LBrace, "`{`")?;
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) {
                break;
            }
            if self.at(&Tok::Eof) {
                return Err(self.err_here("this debug block is never closed: expected `}`"));
            }
            let (word, wspan) = self.ident("`watch`, `hold` or `trace`")?;
            let item = match word.as_str() {
                "watch" => {
                    let pos = self.debug_pos();
                    // `a, b, name = expr`: a variable of the code, or a value
                    // the lens computes under a name of its own
                    let mut names = Vec::new();
                    let mut exprs = Vec::new();
                    loop {
                        names.push(self.ident("a name to watch")?);
                        if self.at(&Tok::Eq) {
                            self.bump();
                            exprs.push(Some(self.expr()?));
                        } else {
                            exprs.push(None);
                        }
                        if !self.at(&Tok::Comma) {
                            break;
                        }
                        self.bump();
                    }
                    DebugItem::Watch { pos, names, exprs, span: wspan.merge(self.toks[self.pos.saturating_sub(1)].span) }
                }
                "hold" => {
                    let pos = self.debug_pos();
                    let e = self.expr()?;
                    let check = if self.at_word(0, "within") {
                        self.bump();
                        let tol = self.expr()?;
                        match e {
                            Expr::Binary { op: BinOp::EqEq, lhs, rhs, span } => {
                                HoldCheck::Near { diff: Expr::Binary { op: BinOp::Sub, lhs, rhs, span }, tol }
                            }
                            other => {
                                return Err(Diagnostic::error("`within` needs `a == b` before it")
                                    .with_primary(other.span(), "not an `==` comparison")
                                    .with_note("`hold x == y within tol \"...\"` holds when every |x - y| is at most tol"))
                            }
                        }
                    } else {
                        HoldCheck::Cond(e)
                    };
                    let sentence = match &self.peek().tok {
                        Tok::Str(s) => {
                            let s = s.clone();
                            self.bump();
                            s
                        }
                        _ => return Err(self.err_here("a hold ends with the sentence it checks, in quotes")),
                    };
                    DebugItem::Hold { pos, check, sentence, span: wspan.merge(self.toks[self.pos.saturating_sub(1)].span) }
                }
                "trace" => {
                    let names = self.name_list("a def to trace")?;
                    DebugItem::Trace { names, span: wspan.merge(self.toks[self.pos.saturating_sub(1)].span) }
                }
                other => {
                    return Err(Diagnostic::error(format!("a debug block holds `watch`, `hold` and `trace` lines, not `{other}`"))
                        .with_primary(wspan, "not a debug item"))
                }
            };
            items.push(item);
            if !self.at(&Tok::Newline) && !self.at(&Tok::RBrace) {
                return Err(self.err_here("one debug item per line"));
            }
        }
        let close = self.expect(Tok::RBrace, "`}`")?;
        let mut block = DebugBlock { def, def_span, items, span: start.merge(close.span) };
        if unnamed {
            // the def that follows, after any blank lines and comments
            let save = self.pos;
            self.skip_newlines();
            let k = if self.at_placeholder() { 2 } else { 1 };
            let named = (self.at(&Tok::Def) || self.at_placeholder())
                .then(|| self.toks.get(self.pos + k).cloned())
                .flatten()
                .and_then(|t| if let Tok::Ident(n) = t.tok { Some((n, t.span)) } else { None });
            self.pos = save;
            match named {
                Some((n, sp)) => {
                    block.def = Some(n);
                    block.def_span = sp;
                }
                None => {
                    return Err(Diagnostic::error("a `debug { }` block without a name must come right before the def it is about")
                        .with_primary(block.span, "no def follows")
                        .with_note("name the def instead: `debug scf { ... }`, or `debug * { ... }` for the whole program"))
                }
            }
        }
        Ok(block)
    }

    /// `per iteration`, `at <step>`, or nothing (which means per iteration).
    fn debug_pos(&mut self) -> DebugPos {
        if self.at_word(0, "per") && self.at_word(1, "iteration") {
            self.bump();
            self.bump();
            return DebugPos::PerIteration;
        }
        if self.at_word(0, "at") && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Ident(_))) {
            self.bump();
            let t = self.bump();
            if let Tok::Ident(id) = t.tok {
                return DebugPos::At(id, t.span);
            }
        }
        DebugPos::PerIteration
    }

    fn name_list(&mut self, what: &str) -> PResult<Vec<(String, Span)>> {
        let mut names = vec![self.ident(what)?];
        while self.at(&Tok::Comma) {
            self.bump();
            names.push(self.ident(what)?);
        }
        Ok(names)
    }

    /// `placeholder def name(params) [-> T]`: a def with a signature and no
    /// body yet. It type-checks as a
    /// signature; calling it is an error that names it.
    fn placeholder_def(&mut self) -> PResult<Def> {
        let start = self.span();
        self.bump(); // `placeholder`
        let (name, name_span, params, ret) = self.def_head()?;
        if self.at(&Tok::LBrace) {
            return Err(Diagnostic::error(format!("placeholder `{name}` has a body"))
                .with_primary(self.span(), "a placeholder is a signature only")
                .with_note("drop `placeholder` once the def is written"));
        }
        if !self.at(&Tok::Newline) && !self.at(&Tok::Eof) {
            return Err(self.err_here("a placeholder def ends after its signature"));
        }
        let span = start.merge(self.toks[self.pos.saturating_sub(1)].span);
        let body = vec![Stmt::Lens { lens: Box::new(Lens::Placeholder { def: name.clone() }), span }];
        Ok(Def { name, name_span, params, ret, body, span })
    }

    /// `def name(params) [-> T]`, up to the body.
    fn def_head(&mut self) -> PResult<(String, Span, Vec<Param>, Option<TypeExpr>)> {
        self.expect(Tok::Def, "`def`")?;
        let (name, name_span) = self.ident("function name")?;
        self.expect(Tok::LParen, "`(`")?;
        let mut params = Vec::new();
        // the first parameter with a default, after which every one has one
        let mut defaulted: Option<String> = None;
        if !self.at(&Tok::RParen) {
            loop {
                let (pname, pspan) = self.ident("parameter name")?;
                // Optional `: Type` annotation (required for record-typed params).
                let ty = if self.at(&Tok::Colon) {
                    self.bump();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                // `= default` (crate::kwargs): checked here, written into
                // each call that leaves the parameter out
                if self.at(&Tok::Eq) {
                    let eq = self.span();
                    self.bump();
                    let before = self.lifted.len();
                    let d = self.expr()?;
                    if self.lifted.len() != before {
                        return Err(Diagnostic::error("a default is one expression: a comprehension, an `if` expression or a `key=` cannot be in it").with_primary(d.span(), "here"));
                    }
                    let later: Vec<String> = self.sigs.get(&name).map(|ps| ps.iter().skip(params.len()).map(|p| p.name.clone()).collect()).unwrap_or_default();
                    check_default(&pname, &d, &later).map_err(|m| Diagnostic::error(m).with_primary(eq.merge(d.span()), "this default"))?;
                    defaulted.get_or_insert(pname.clone());
                } else if let Some(first) = &defaulted {
                    return Err(Diagnostic::error(format!("`{pname}` has no default and comes after `{first}`, which has one"))
                        .with_primary(pspan, "no default")
                        .with_note("the parameters with defaults come last"));
                }
                params.push(Param { name: pname, span: pspan, ty });
                if self.at(&Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(Tok::RParen, "`)`")?;
        // Optional `-> Type` return annotation (required when returning a record).
        let ret = if self.at(&Tok::Arrow) {
            self.bump();
            Some(self.parse_type()?)
        } else {
            None
        };
        Ok((name, name_span, params, ret))
    }

    fn def(&mut self) -> PResult<Def> {
        let start = self.span();
        let (name, name_span, params, ret) = self.def_head()?;
        self.expect(Tok::LBrace, "`{`")?;
        // what the body assigns, for the phrases in it
        let mut names = crate::phrases::assigned(&self.toks[self.pos..self.block_end()], false);
        names.extend(params.iter().map(|p| p.name.clone()));
        let outer_scope = std::mem::replace(&mut self.scope, names);
        let outer_def = std::mem::replace(&mut self.cur_def, name.clone());
        let outer_depth = std::mem::replace(&mut self.step_depth, 0);
        let outer_ids = std::mem::take(&mut self.step_ids);
        let body = self.block();
        self.cur_def = outer_def;
        self.scope = outer_scope;
        self.step_depth = outer_depth;
        self.step_ids = outer_ids;
        let body = body?;
        let close = self.expect(Tok::RBrace, "`}`")?;
        Ok(Def { name, name_span, params, ret, body, span: start.merge(close.span) })
    }

    /// A type annotation: `Num`, `Arr`, or a record type name; `T?` is an
    /// optional T.
    fn parse_type(&mut self) -> PResult<TypeExpr> {
        let mut t = self.parse_base_type()?;
        while self.at(&Tok::Question) {
            self.bump();
            t = TypeExpr::Opt(Box::new(t));
        }
        Ok(t)
    }

    fn parse_base_type(&mut self) -> PResult<TypeExpr> {
        let span = self.span();
        // `[Str]`: a list of strings; `[Int]`: an array of Ints; `[Block]`:
        // an array of records
        if self.at(&Tok::LBracket) {
            self.bump();
            if self.at_word(0, "Str") || self.at_word(0, "Int") {
                let int = self.at_word(0, "Int");
                self.bump();
                self.expect(Tok::RBracket, "`]`")?;
                return Ok(if int { TypeExpr::IntList } else { TypeExpr::StrList });
            }
            // `[Float32]`, `[BFloat16]`, `[Int8]`, ...: a packed array
            if let Tok::Ident(s) = &self.peek().tok {
                if let Some(k) = crate::ast::Store::of_name(s) {
                    self.bump();
                    self.expect(Tok::RBracket, "`]`")?;
                    return Ok(TypeExpr::Packed(k));
                }
            }
            let name = match &self.peek().tok {
                Tok::Ident(s) if !matches!(s.as_str(), "Num" | "Arr" | "CArr" | "Bool") => s.clone(),
                _ => return Err(self.err_here("a list type is `[Str]`, a list of strings, `[Int]`, an array of Ints, or `[R]`, an array of records of type R; an array of numbers is `Arr`")),
            };
            let name_span = self.span();
            self.bump();
            self.expect(Tok::RBracket, "`]`")?;
            return Ok(TypeExpr::RecList { name, span: name_span.merge(span) });
        }
        match &self.peek().tok {
            Tok::Ident(s) => {
                let s = s.clone();
                self.bump();
                Ok(match s.as_str() {
                    "Num" => TypeExpr::Num,
                    "Arr" => TypeExpr::Arr,
                    "CArr" => TypeExpr::CArr,
                    "Bool" => TypeExpr::Bool,
                    "Str" => TypeExpr::Str,
                    "Int" => TypeExpr::Int,
                    "Float32" => TypeExpr::F32,
                    _ => TypeExpr::Named { name: s, span },
                })
            }
            _ => Err(self.err_here("expected a type (`Num`, `Int`, `Bool`, `Str`, `Arr`, `CArr`, `[Str]`, `[Int]`, or a record name)")),
        }
    }

    /// `enum Name {`: an enum declaration, not a variable named `enum`.
    fn at_enum(&self) -> bool {
        self.at_word(0, "enum")
            && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Ident(_)))
            && matches!(self.toks.get(self.pos + 2).map(|t| &t.tok), Some(Tok::LBrace))
    }

    /// `enum Name { A, B(T, U), ... }` (commas and/or newlines separate the
    /// variants).
    fn enum_decl(&mut self) -> PResult<EnumDecl> {
        let start = self.span();
        self.bump();
        let (name, name_span) = self.ident("enum name")?;
        self.expect(Tok::LBrace, "`{`")?;
        let mut variants: Vec<VariantDecl> = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                break;
            }
            let (vname, vspan) = self.ident("variant name")?;
            if variants.iter().any(|v| v.name == vname) {
                return Err(Diagnostic::error(format!("`{name}` has two variants named `{vname}`")).with_primary(vspan, "named again here"));
            }
            let mut payload = Vec::new();
            if self.at(&Tok::LParen) {
                self.bump();
                loop {
                    payload.push(self.parse_type()?);
                    if self.at(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(Tok::RParen, "`)`")?;
            }
            variants.push(VariantDecl { name: vname, name_span: vspan, payload });
            self.skip_newlines();
            if self.at(&Tok::Comma) {
                self.bump();
            }
        }
        let close = self.expect(Tok::RBrace, "`}`")?;
        if variants.is_empty() {
            return Err(Diagnostic::error(format!("enum `{name}` has no variants")).with_primary(name_span, "an enum needs at least one"));
        }
        Ok(EnumDecl { name, name_span, variants, span: start.merge(close.span) })
    }

    /// `record Name { field: Type, field: Type = default, ... }` (commas
    /// and/or newlines separate fields; crate::ctor fills the defaults in).
    fn record_decl(&mut self) -> PResult<RecordDecl> {
        let start = self.span();
        self.expect(Tok::Record, "`record`")?;
        let (name, name_span) = self.ident("record name")?;
        self.expect(Tok::LBrace, "`{`")?;
        let mut fields = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                break;
            }
            let (fname, fspan) = self.ident("field name")?;
            self.expect(Tok::Colon, "`:`")?;
            let ty = self.parse_type()?;
            let default = if self.at(&Tok::Eq) {
                self.bump();
                Some(self.expr()?)
            } else {
                None
            };
            fields.push(FieldDecl { name: fname, name_span: fspan, ty, default });
            self.skip_newlines();
            if self.at(&Tok::Comma) {
                self.bump();
            }
        }
        let close = self.expect(Tok::RBrace, "`}`")?;
        Ok(RecordDecl { name, name_span, fields, span: start.merge(close.span) })
    }

    /// Statements until (but not consuming) the closing `}`.
    fn block(&mut self) -> PResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                break;
            }
            self.stmt_into(&mut stmts)?;
        }
        Ok(stmts)
    }

    /// Whether the next tokens open a step: `step <id> "<text>" {` or
    /// `step <id> {`. `step` is a keyword only in this position, so a program
    /// that names a variable `step` (`step = 1.0`, `x = x + step`) parses as
    /// it always did.
    fn at_step(&self) -> bool {
        let tok = |k: usize| self.toks.get(self.pos + k).map(|t| &t.tok);
        matches!(tok(0), Some(Tok::Ident(s)) if s == "step")
            && matches!(tok(1), Some(Tok::Ident(_)))
            && matches!(tok(2), Some(Tok::Str(_)) | Some(Tok::LBrace))
    }

    /// One statement, appended to `out`. A step appends several: its enter
    /// marker, its body's statements, and its exit marker (see `Stmt::StepEnter`).
    fn stmt_into(&mut self, out: &mut Vec<Stmt>) -> PResult<()> {
        if !self.at_step() {
            return self.stmt(out);
        }
        let start = self.span();
        self.bump(); // `step`
        let (id, id_span) = self.ident("step name")?;
        let mut text = String::new();
        if let Tok::Str(s) = &self.peek().tok {
            text = s.clone();
            self.bump();
        }
        if self.in_parallel > 0 {
            return Err(Diagnostic::error("a `step` cannot sit inside a `parallel for`")
                .with_primary(start.merge(id_span), "inside a parallel loop")
                .with_note("the iterations run on several threads, so the order of the step's events would not be fixed; put the step around the loop instead"));
        }
        if self.cur_def != MAIN_DEF {
            if self.step_ids.contains(&id) {
                return Err(Diagnostic::error(format!("`{}` already has a step named `{}`", self.cur_def, id))
                    .with_primary(id_span, "a second step with this name")
                    .with_note(format!("a step is addressed as `{}/{}`, so its name must be unique in the def", self.cur_def, id)));
            }
            self.step_ids.push(id.clone());
        }
        let open = self.expect(Tok::LBrace, "`{`")?;
        self.step_depth += 1;
        let depth = self.step_depth;
        out.push(Stmt::StepEnter { def: self.cur_def.clone(), id: id.clone(), text, depth, span: start.merge(open.span) });
        let body = self.block();
        self.step_depth -= 1;
        out.extend(body?);
        let close = self.expect(Tok::RBrace, "`}`")?;
        out.push(Stmt::StepExit { def: self.cur_def.clone(), id, depth, span: close.span });
        Ok(())
    }

    /// One statement, appended to `out`. The sugar that needs hidden
    /// variables (`a[i] += x`, `for i from a to b by s`) appends the
    /// assignments to them first, then the statement.
    fn stmt(&mut self, out: &mut Vec<Stmt>) -> PResult<()> {
        if self.at_phrase_decl() {
            return Err(Diagnostic::error("a phrase is declared at the top level of a program").with_primary(self.span(), "inside a block"));
        }
        match self.peek().tok {
            Tok::For => return self.for_stmt(false, out),
            Tok::Parallel => {
                self.bump();
                return self.for_stmt(true, out);
            }
            _ => {}
        }
        if self.at_on_device() {
            let s = self.on_device()?;
            out.push(s);
            return Ok(());
        }
        let s = self.simple_stmt(out)?;
        out.push(s);
        Ok(())
    }

    /// A fresh hidden variable: `viv__<what><n>`.
    fn hidden_name(&mut self, what: &str) -> String {
        self.hidden += 1;
        format!("{HIDDEN_PREFIX}{what}{}", self.hidden)
    }

    /// `e` itself when reading it again reads the same value (a number or a
    /// variable); otherwise a hidden variable assigned `e` once, here.
    fn once(&mut self, e: Expr, what: &str, out: &mut Vec<Stmt>) -> Expr {
        if matches!(e, Expr::Num { .. } | Expr::Var { .. }) {
            return e;
        }
        let span = e.span();
        let name = self.hidden_name(what);
        out.push(Stmt::Assign { name: name.clone(), name_span: span, value: e, span });
        Expr::Var { name, span }
    }

    /// A place (`r`, `r.f`, `xs[i].g`) whose indices are each read once:
    /// an index that is not a number or a variable is computed into a hidden
    /// variable first, so `r.xs[k(i)].f += 1` calls `k` once.
    fn once_place(&mut self, e: Expr, out: &mut Vec<Stmt>) -> Expr {
        match e {
            Expr::Field { base, field, field_span, span } => {
                let base = Box::new(self.once_place(*base, out));
                Expr::Field { base, field, field_span, span }
            }
            Expr::Index { base, indices, span } => {
                let base = Box::new(self.once_place(*base, out));
                let indices = indices.into_iter().map(|i| self.once(i, "i", out)).collect();
                Expr::Index { base, indices, span }
            }
            e => e,
        }
    }

    fn simple_stmt(&mut self, out: &mut Vec<Stmt>) -> PResult<Stmt> {
        if self.at_match() {
            return self.match_stmt(out);
        }
        match self.peek().tok {
            Tok::If => return self.if_stmt(),
            Tok::While => return self.while_stmt(),
            Tok::Break => {
                let span = self.span();
                self.bump();
                return Ok(Stmt::Break { span });
            }
            Tok::Continue => {
                let span = self.span();
                self.bump();
                return Ok(Stmt::Continue { span });
            }
            Tok::Narrate => {
                let span = self.span();
                self.bump();
                // A string literal becomes live narration text; any other
                // expression is commentary, parsed and dropped.
                let arg = self.expr()?;
                let text = match &arg {
                    Expr::Str { value, .. } => value.clone(),
                    _ => String::new(),
                };
                return Ok(Stmt::Narrate { text, span });
            }
            Tok::Show => {
                let start = self.span();
                self.bump();
                let expr = self.expr()?;
                let is_fmt = |e: &Expr| matches!(e, Expr::Call { callee, .. } if callee == "fmt");
                if !self.at(&Tok::Comma) && !is_fmt(&expr) {
                    return Ok(Stmt::Show { span: start.merge(expr.span()), expr });
                }
                // `show a, b` and `show fmt(x, 17)`: labeled values on one
                // line, which is `print("a: ", a, "  b: ", b)` (crate::printfmt).
                // A variable, or `fmt` of one, is labeled with its name.
                let mut items = vec![expr];
                while self.at(&Tok::Comma) {
                    self.bump();
                    items.push(self.expr()?);
                }
                let span = start.merge(items.last().map(|e| e.span()).unwrap_or(start));
                let mut args = Vec::with_capacity(2 * items.len());
                for (k, e) in items.into_iter().enumerate() {
                    let name = match &e {
                        Expr::Var { name, .. } => Some(name.clone()),
                        Expr::Call { callee, args, .. } if callee == "fmt" => match args.first() {
                            Some(Expr::Var { name, .. }) => Some(name.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    let mut text = String::from(if k > 0 { "  " } else { "" });
                    if let Some(name) = name {
                        text.push_str(&name);
                        text.push_str(": ");
                    }
                    if !text.is_empty() {
                        args.push(Expr::Str { value: text, span: e.span() });
                    }
                    args.push(e);
                }
                return Ok(Stmt::ExprStmt { expr: Expr::Call { callee: "print".into(), args, span }, span });
            }
            Tok::Test => {
                let start = self.span();
                self.bump();
                // The name is a string literal, and it is what the block
                // reports under. An unnamed `test { .. }` is allowed and
                // reports under the empty name.
                let mut name = String::new();
                if let Tok::Str(s) = &self.peek().tok {
                    name = s.clone();
                    self.bump();
                }
                let (body, end) = self.brace_block()?;
                return Ok(Stmt::Test { name, body, span: start.merge(end) });
            }
            Tok::Return => {
                let start = self.span();
                self.bump();
                // a bare `return` leaves the def: it gives 0, the value a def
                // that reaches its end without a `return` gives
                if matches!(self.peek().tok, Tok::Newline | Tok::RBrace | Tok::Eof) {
                    return Ok(Stmt::Return { span: start, value: Expr::Num { value: 0.0, span: start } });
                }
                let value = self.expr()?;
                // `return a, b`: several results, which `crate::tuples` turns
                // into a record the caller unpacks
                if self.at(&Tok::Comma) {
                    let value = self.comma_list(value)?;
                    return Ok(Stmt::Return { span: start.merge(value.span()), value });
                }
                return Ok(Stmt::Return { span: start.merge(value.span()), value });
            }
            _ => {}
        }
        // `del xs[i]`: `xs = viv__del(xs, i)`, a copy without element i
        // (lang/viv/std/builtins.viv; a program's own `del` is a name)
        if matches!(&self.peek().tok, Tok::Ident(n) if n == "del") && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Ident(_))) && !self.defs.contains("del") {
            let start = self.span();
            self.bump();
            let target = self.expr()?;
            let Expr::Index { base, indices, span } = target else {
                return Err(Diagnostic::error("`del` removes one element: `del xs[i]`").with_primary(target.span(), "not an element of a list"));
            };
            let (Expr::Var { name, span: name_span }, [i]) = (*base, indices.as_slice()) else {
                return Err(Diagnostic::error("`del` removes one element of a variable's list: `del xs[i]`").with_primary(span, "here"));
            };
            let value = Expr::Call { callee: crate::stdlib::DEL.to_string(), args: vec![Expr::Var { name: name.clone(), span: name_span }, i.clone()], span };
            return Ok(Stmt::Assign { name, name_span, value, span: start.merge(span) });
        }
        // Parse an expression; if `=` follows and it is a valid lvalue (a name
        // or an indexed name), it is an assignment. Otherwise it is a bare
        // expression statement (e.g. `print(x)`).
        let lhs = self.expr()?;
        let lhs_span = lhs.span();
        // `a, b = e`: several targets. `loss, g = value_and_grad(f, x, ...)`
        // (v4) is kept as an assignment of the first
        // name, with the second name as the call's first argument;
        // `crate::grad` expands it. Every other form -- `a, b = f(x)` of a
        // def that returns several values, `B, C, H, W = shape(X)`, a swap
        // `a, b = b, a` -- is an `UNPACK` call that `crate::tuples` expands.
        if self.at(&Tok::Comma) {
            let mut targets = vec![lhs];
            while self.at(&Tok::Comma) {
                self.bump();
                targets.push(self.expr()?);
            }
            if !self.at(&Tok::Eq) {
                return Err(Diagnostic::error("expected `=` after the names to unpack into")
                    .with_primary(self.span(), "expected `=`")
                    .with_note("`a, b = f(x)` unpacks; `+=` and the other compound forms take one target"));
            }
            self.bump();
            let first = self.expr()?;
            let rhs = if self.at(&Tok::Comma) { self.comma_list(first)? } else { first };
            let span = lhs_span.merge(rhs.span());
            if let [Expr::Var { name, span: name_span }, Expr::Var { name: second, span: second_span }] = targets.as_slice() {
                if let Expr::Call { callee, args, span: call_span } = &rhs {
                    if callee == "value_and_grad" {
                        let mut args = args.clone();
                        args.insert(0, Expr::Var { name: second.clone(), span: *second_span });
                        let value = Expr::Call { callee: crate::grad::VALUE_AND_GRAD_PAIR.to_string(), args, span: *call_span };
                        return Ok(Stmt::Assign { name: name.clone(), name_span: *name_span, value, span });
                    }
                }
            }
            let targets = Expr::ArrLit { elems: targets, span: lhs_span };
            let expr = Expr::Call { callee: crate::tuples::UNPACK.to_string(), args: vec![targets, rhs], span };
            return Ok(Stmt::ExprStmt { expr, span });
        }
        // `x: T = e`: an assignment that says the variable's type (design/
        // int_enum.md). It is `viv__declare("x", "T")`, which the lowering
        // and inference read as the type of `x`, then `x = e`; the engines
        // run nothing for the first.
        if let (Expr::Var { name, span: name_span }, true) = (&lhs, self.at(&Tok::Colon)) {
            let (name, name_span) = (name.clone(), *name_span);
            self.bump();
            let ty = self.parse_type()?;
            self.expect(Tok::Eq, "`=` after the type")?;
            let rhs = self.expr()?;
            let span = lhs_span.merge(rhs.span());
            // `xs: [Int] = [f(k) for k to n]`, `s: Int = -1 if c else 1`: the
            // comprehension's (the if-expression's) own variable has the
            // declared type, so it builds an Int array (holds an Int)
            if let Some(id) = crate::surface::placeholder_id(&rhs) {
                if let Some(l) = self.lifted.get_mut(id) {
                    let comp = matches!((&l.result, l.stmts.first()), (Expr::Var { name: hidden, .. }, Some(Stmt::Assign { name: first, value: Expr::ArrLit { elems, .. }, .. }))
                        if hidden == first && elems.is_empty() && type_text(&ty).starts_with('['));
                    let cond = matches!(&l.result, Expr::Var { name: hidden, .. } if hidden.starts_with("viv__if"));
                    if let (Expr::Var { name: hidden, .. }, true) = (&l.result, comp || cond) {
                        {
                            let d = Expr::Call {
                                callee: DECLARE.to_string(),
                                args: vec![Expr::Str { value: hidden.clone(), span: name_span }, Expr::Str { value: type_text(&ty), span: name_span }],
                                span: name_span,
                            };
                            l.stmts.insert(0, Stmt::ExprStmt { expr: d, span: name_span });
                        }
                    }
                }
            }
            let decl = Expr::Call {
                callee: DECLARE.to_string(),
                args: vec![Expr::Str { value: name.clone(), span: name_span }, Expr::Str { value: type_text(&ty), span: name_span }],
                span: name_span,
            };
            out.push(Stmt::ExprStmt { expr: decl, span: name_span });
            return Ok(Stmt::Assign { name, name_span, value: rhs, span });
        }
        // Plain (`=`) or compound (`+=`/`-=`/`*=`/`/=`) assignment.
        let compound = match self.peek().tok {
            Tok::PlusEq => Some(BinOp::Add),
            Tok::MinusEq => Some(BinOp::Sub),
            Tok::StarEq => Some(BinOp::Mul),
            Tok::SlashEq => Some(BinOp::Div),
            Tok::SlashSlashEq => Some(BinOp::FloorDiv),
            _ => None,
        };
        if self.at(&Tok::Eq) || compound.is_some() {
            self.bump();
            let rhs = self.expr()?;
            let span = lhs_span.merge(rhs.span());
            return match lhs {
                Expr::Var { name, span: name_span } => {
                    let value = match compound {
                        Some(op) => Expr::Binary {
                            op,
                            lhs: Box::new(Expr::Var { name: name.clone(), span: name_span }),
                            rhs: Box::new(rhs),
                            span,
                        },
                        None => rhs,
                    };
                    Ok(Stmt::Assign { name, name_span, value, span })
                }
                // `a[lo:hi, j] = v` writes through the builtin; `a[lo:hi] op= v`
                // reads the slice once and writes it back, each index
                // evaluated once
                Expr::Call { callee, mut args, span: idx_span } if callee == SLICE_GET && args.len() == 2 => {
                    let spec = args.pop().expect("two arguments");
                    let base = args.pop().expect("two arguments");
                    let spec = match (spec, compound) {
                        (Expr::ArrLit { elems, span }, Some(_)) => {
                            let elems = elems.into_iter().map(|e| self.once(e, "i", out)).collect();
                            Expr::ArrLit { elems, span }
                        }
                        (spec, _) => spec,
                    };
                    let value = match compound {
                        Some(op) => {
                            let read = Expr::Call { callee: SLICE_GET.to_string(), args: vec![base.clone(), spec.clone()], span: idx_span };
                            Expr::Binary { op, lhs: Box::new(read), rhs: Box::new(rhs), span }
                        }
                        None => rhs,
                    };
                    let expr = Expr::Call { callee: SLICE_SET.to_string(), args: vec![base, spec, value], span };
                    Ok(Stmt::ExprStmt { expr, span })
                }
                Expr::Index { base, indices, span: idx_span } => {
                    // an element of a record in an array (`xs[k(i)].v[j] = x`):
                    // the place's own indices are read once, before the value
                    let base = Box::new(self.once_place(*base, out));
                    let Some(op) = compound else {
                        return Ok(Stmt::SetIndex { base, indices, value: rhs, span });
                    };
                    // `a[i, j] op= x` is `a[i, j] = a[i, j] op x` with each
                    // index evaluated once: an index that is not a number or
                    // a variable is computed into a hidden variable first, so
                    // `a[randint(n)] += 1` draws one number, not two. (The
                    // right side cannot change a variable, so a variable
                    // index reads the same value both times.)
                    let indices: Vec<Expr> = indices.into_iter().map(|e| self.once(e, "i", out)).collect();
                    let read = Expr::Index { base: base.clone(), indices: indices.clone(), span: idx_span };
                    let value = Expr::Binary { op, lhs: Box::new(read), rhs: Box::new(rhs), span };
                    Ok(Stmt::SetIndex { base, indices, value, span })
                }
                // `r.f = v`, `model.blocks[l].attn.Wq = W`: a field of a place
                // (`crate::places` checks the place); `r.f op= v` reads the
                // field once and writes it back, each index of the place
                // evaluated once
                Expr::Field { base, field, field_span, span: fspan } => {
                    let base = Box::new(self.once_place(*base, out));
                    let Some(op) = compound else {
                        return Ok(Stmt::SetField { base, field, field_span, value: rhs, span });
                    };
                    let read = Expr::Field { base: base.clone(), field: field.clone(), field_span, span: fspan };
                    let value = Expr::Binary { op, lhs: Box::new(read), rhs: Box::new(rhs), span };
                    Ok(Stmt::SetField { base, field, field_span, value, span })
                }
                other => Err(Diagnostic::error("invalid assignment target")
                    .with_primary(other.span(), "cannot assign to this")),
            };
        }
        // `stop("message")`: end the run with an error, located at this line,
        // in every engine (the stop the recursion limit uses, `Lens::Stop`)
        if let Expr::Call { callee, args, span } = &lhs {
            if callee == "stop" {
                return match args.as_slice() {
                    [Expr::Str { value, .. }] => Ok(Stmt::Lens { lens: Box::new(Lens::Stop { message: value.clone(), kind: crate::errkind::ErrKind::Stop }), span: *span }),
                    // a string value: the message is known when the run gets here
                    [_] => Ok(Stmt::ExprStmt { span: lhs_span, expr: lhs }),
                    _ => Err(Diagnostic::error("`stop` takes one string: `stop(\"what went wrong\")`").with_primary(*span, "not one string")),
                };
            }
        }
        Ok(Stmt::ExprStmt { span: lhs_span, expr: lhs })
    }

    /// `match e {`: a match statement, not a variable named `match` (`match =
    /// -1.0`, `match(x)`, `match[i]`).
    fn at_match(&self) -> bool {
        self.at_word(0, "match")
            && !matches!(
                self.toks.get(self.pos + 1).map(|t| &t.tok),
                None | Some(Tok::Eq | Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq | Tok::SlashSlashEq | Tok::LParen | Tok::LBracket | Tok::Dot | Tok::Comma | Tok::Newline | Tok::Eof | Tok::Colon)
            )
    }

    /// `match e { Pattern => body ... }`,
    /// written as the `if` chain it means:
    ///
    /// ```text
    /// viv__mN_1 = e
    /// if viv__enum_is(viv__mN_1, "A") { <binders>; body } else if ... else { viv__match_rest(viv__mN_1, "A|B") }
    /// ```
    ///
    /// A binder is renamed apart (`viv__mN_A_v`, arm A) so it belongs to its arm. The
    /// lowering checks the rest against the scrutinee's enum: a variant no arm
    /// covers is refused there, where the type is known.
    fn match_stmt(&mut self, out: &mut Vec<Stmt>) -> PResult<Stmt> {
        let start = self.span();
        self.bump();
        let scrutinee = self.head_expr()?;
        self.expect(Tok::LBrace, "`{` after the value a `match` takes apart")?;
        self.matches += 1;
        let k = self.matches;
        struct Arm {
            variants: Vec<(String, Span)>,
            binders: Vec<(String, Span)>,
            wild: bool,
            body: Vec<Stmt>,
            span: Span,
        }
        let mut arms: Vec<Arm> = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                break;
            }
            let arm_start = self.span();
            if let Some(w) = arms.iter().find(|a| a.wild) {
                return Err(Diagnostic::error("this arm can never run: the arm `_` before it takes every value")
                    .with_primary(arm_start, "never reached")
                    .with_secondary(w.span, "`_` matches everything"));
            }
            let mut variants: Vec<(String, Span)> = Vec::new();
            let mut binders = Vec::new();
            let mut wild = false;
            if self.at_word(0, "_") {
                self.bump();
                wild = true;
            } else {
                loop {
                    let (v, vspan) = self.ident("a variant name or `_`")?;
                    if let Some((_, ps)) = arms.iter().flat_map(|a| a.variants.iter()).chain(variants.iter()).find(|(p, _)| *p == v) {
                        return Err(Diagnostic::error(format!("this arm can never run: `{v}` is matched by an arm before it"))
                            .with_primary(vspan, "matched again")
                            .with_secondary(*ps, "matched here first"));
                    }
                    variants.push((v, vspan));
                    if self.at(&Tok::LParen) {
                        self.bump();
                        loop {
                            let (b, bspan) = self.ident("a name for the payload, or `_`")?;
                            binders.push((b, bspan));
                            if self.at(&Tok::Comma) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                        self.expect(Tok::RParen, "`)`")?;
                    }
                    if self.at(&Tok::Pipe) {
                        self.bump();
                        continue;
                    }
                    break;
                }
                if variants.len() > 1 && !binders.is_empty() {
                    return Err(Diagnostic::error("an arm for several variants cannot name their payloads")
                        .with_primary(arm_start, "write one arm per variant to read a payload"));
                }
            }
            self.expect(Tok::FatArrow, "`=>` after the pattern")?;
            let body = if self.at(&Tok::LBrace) {
                self.brace_block()?.0
            } else {
                let mut b = Vec::new();
                self.stmt_into(&mut b)?;
                b
            };
            arms.push(Arm { variants, binders, wild, body, span: arm_start });
        }
        let close = self.expect(Tok::RBrace, "`}` to end the `match`")?;
        let span = start.merge(close.span);
        if arms.is_empty() {
            return Err(Diagnostic::error("a `match` with no arms").with_primary(span, "it needs one arm per variant, or `_`"));
        }
        let m = self.hidden_var(scrutinee, &format!("m{k}_"), out);
        let call = |callee: &str, args: Vec<Expr>, span: Span| Expr::Call { callee: callee.to_string(), args, span };
        let text = |s: &str, span: Span| Expr::Str { value: s.to_string(), span };
        // the rest: what the arms cover, for the lowering to check
        let mut chain: Vec<Stmt> = match arms.iter().position(|a| a.wild) {
            Some(i) => std::mem::take(&mut arms[i].body),
            None => {
                let covered: Vec<&str> = arms.iter().flat_map(|a| a.variants.iter().map(|(v, _)| v.as_str())).collect();
                vec![Stmt::ExprStmt { expr: call(MATCH_REST, vec![m.clone(), text(&covered.join("|"), span)], span), span }]
            }
        };
        for (ai, arm) in arms.into_iter().enumerate().collect::<Vec<_>>().into_iter().rev().filter(|(_, a)| !a.wild) {
            let mut cond: Option<Expr> = None;
            for (v, vspan) in &arm.variants {
                let is = call(ENUM_IS, vec![m.clone(), text(v, *vspan)], *vspan);
                cond = Some(match cond {
                    None => is,
                    Some(c) => Expr::Binary { op: BinOp::Or, lhs: Box::new(c), rhs: Box::new(is), span: *vspan },
                });
            }
            let mut body = Vec::new();
            let mut map = HashMap::new();
            if let [(v, _)] = arm.variants.as_slice() {
                for (j, (b, bspan)) in arm.binders.iter().enumerate() {
                    if b == "_" {
                        continue;
                    }
                    let hidden = format!("{HIDDEN_PREFIX}m{k}_{ai}_{b}");
                    map.insert(b.clone(), hidden.clone());
                    let value = call(ENUM_FIELD, vec![m.clone(), text(v, *bspan), Expr::Num { value: j as f64, span: *bspan }], *bspan);
                    body.push(Stmt::Assign { name: hidden, name_span: *bspan, value, span: *bspan });
                }
            }
            let mut arm_body = arm.body;
            self.subst_block(&mut arm_body, &map);
            body.extend(arm_body);
            let cond = cond.expect("an arm names a variant");
            chain = vec![Stmt::If { cond, then_body: body, else_body: chain, span: arm.span }];
        }
        Ok(chain.pop().expect("one statement"))
    }

    /// `first, e2, e3`: the rest of a comma list after its first expression,
    /// as one `TUPLE` call (`return a, b` and the right side of `a, b = b, a`).
    fn comma_list(&mut self, first: Expr) -> PResult<Expr> {
        let start = first.span();
        let mut elems = vec![first];
        while self.at(&Tok::Comma) {
            self.bump();
            elems.push(self.expr()?);
        }
        let span = start.merge(elems.last().expect("at least one").span());
        Ok(Expr::Call { callee: crate::tuples::TUPLE.to_string(), args: elems, span })
    }

    /// `{ <stmts> }`, returning the statements and the closing brace's span.
    fn brace_block(&mut self) -> PResult<(Vec<Stmt>, Span)> {
        self.expect(Tok::LBrace, "`{`")?;
        let stmts = self.block()?;
        let close = self.expect(Tok::RBrace, "`}`")?;
        Ok((stmts, close.span))
    }

    fn if_stmt(&mut self) -> PResult<Stmt> {
        let start = self.span();
        self.expect(Tok::If, "`if`")?;
        let cond = self.head_expr()?;
        let (then_body, mut end) = self.brace_block()?;
        // Allow `else` on its own line after the closing brace (Stroustrup style),
        // as well as the cuddled `} else {`. Newlines between `}` and `else` are
        // continuation whitespace; if no `else` follows, restore and end the `if`.
        let save = self.pos;
        self.skip_newlines();
        let else_body = if self.at(&Tok::Else) {
            self.bump();
            if self.at(&Tok::If) {
                // `else if ...` chains as a single nested If in the else branch.
                let nested = self.if_stmt()?;
                end = nested.span();
                vec![nested]
            } else {
                let (eb, ec) = self.brace_block()?;
                end = ec;
                eb
            }
        } else {
            self.pos = save;
            Vec::new()
        };
        Ok(Stmt::If { cond, then_body, else_body, span: start.merge(end) })
    }

    fn while_stmt(&mut self) -> PResult<Stmt> {
        let start = self.span();
        self.expect(Tok::While, "`while`")?;
        let cond = self.head_expr()?;
        let (body, end) = self.brace_block()?;
        Ok(Stmt::While { cond, body, span: start.merge(end) })
    }

    /// `for i to n { ... }`, and its longer forms
    ///
    /// ```text
    /// for i from a to b { ... }         i = a, a + 1, ... while i < b
    /// for i from a to b by s { ... }    i = a, a + s, ... while i < b (s > 0)
    ///                                   or while i > b (s < 0)
    /// for i to n reversed { ... }       the same values, last first
    /// ```
    ///
    /// `to` is always the exclusive end, as in `for i to n`: the loop runs
    /// `ceil((b - a) / s)` times (none when that is 0 or less), and the k-th
    /// pass has `i = a + k * s`, or `i = a + (count - 1 - k) * s` reversed.
    /// `a`, `b` and `s` are evaluated once, in that order, before the loop.
    /// A step of 0 is an error: at compile time when it is written as a
    /// number, at run time otherwise.
    ///
    /// The longer forms are sugar: the parser writes them as the plain loop
    /// over a hidden counter, with `i` assigned at the top of each pass, so
    /// every engine runs code it already runs. So are the `for x in e` forms
    /// (`build_for`).
    fn for_stmt(&mut self, parallel: bool, out: &mut Vec<Stmt>) -> PResult<()> {
        let start = self.span();
        self.expect(Tok::For, "`for`")?;
        let head = self.for_head(false)?;
        match &head {
            Head::Count { from, from_span, by, by_span, reversed, rev_span, .. } => {
                let plain = from.is_none() && by.is_none() && !reversed;
                if parallel && !plain {
                    let at = if from.is_some() { *from_span } else if by.is_some() { *by_span } else { *rev_span };
                    return Err(Diagnostic::error("a `parallel for` counts from 0 by 1")
                        .with_primary(at, "not allowed after `parallel`")
                        .with_note("write `parallel for i to n { ... }` and compute the index from i inside"));
                }
            }
            Head::Each { in_span, .. } => {
                if parallel {
                    return Err(Diagnostic::error("a `parallel for` counts from 0 by 1").with_primary(*in_span, "not allowed after `parallel`")
                        .with_note("write `parallel for i to len(e) { ... }` and index e inside"));
                }
            }
        }
        self.in_parallel += parallel as u32;
        let body = self.brace_block();
        self.in_parallel -= parallel as u32;
        let (body, end) = body?;
        self.build_for(head, body, parallel, start.merge(end), out)
    }

    /// A `for` head after the `for`: `i to n` and its longer forms, or
    /// `x in e`, `i, x in enumerate(e[, start])`, `a, b in zip(e, f, ...)`,
    /// where a sequence written `rows(M)` gives the rows of the matrix M.
    /// In a comprehension (`comp`) its expressions end before a filter's
    /// `if`, so they are read without the if-expression.
    fn for_head(&mut self, comp: bool) -> PResult<Head> {
        let (var, var_span) = self.ident("loop variable")?;
        // `for x in e` and `for i, x in enumerate(e)`
        if self.at_word(0, "in") || self.at(&Tok::Comma) {
            let mut names = vec![(var, var_span)];
            while self.at(&Tok::Comma) {
                self.bump();
                names.push(self.ident("the name of the element, after the index")?);
            }
            if !self.at_word(0, "in") {
                return Err(self.err_here("expected `in`"));
            }
            let in_span = self.span();
            self.bump();
            let seq = self.head_part(comp)?;
            if let Expr::Call { callee, .. } = &seq {
                if callee == "range" && names.len() == 1 {
                    return Err(Diagnostic::error("expected `to`").with_primary(in_span, "found identifier `in`").with_note(crate::hints::FOR_IN));
                }
            }
            let over = self.over(seq, names.len())?;
            return Ok(Head::Each { names, over, in_span });
        }
        let from_span = self.span();
        let from = if self.at_word(0, "from") {
            self.bump();
            Some(self.head_part(comp)?)
        } else {
            None
        };
        self.expect(Tok::To, "`to`")?;
        let bound = self.head_part(comp)?;
        let by_span = self.span();
        let by = if self.at_word(0, "by") {
            self.bump();
            Some(self.head_part(comp)?)
        } else {
            None
        };
        let reversed = self.at_word(0, "reversed");
        let rev_span = self.span();
        if reversed {
            self.bump();
        }
        Ok(Head::Count { var, var_span, from, from_span, bound, by, by_span, reversed, rev_span })
    }

    /// One expression of a `for` head.
    fn head_part(&mut self, comp: bool) -> PResult<Expr> {
        if comp {
            self.or_expr()
        } else {
            self.head_expr()
        }
    }

    /// What `for` names in `seq` walk over, for `n` names.
    fn over(&mut self, seq: Expr, n: usize) -> PResult<Over> {
        match seq {
            Expr::Call { callee, mut args, .. } if callee == "enumerate" && n == 2 && (args.len() == 1 || args.len() == 2) => {
                let start = if args.len() == 2 { args.pop() } else { None };
                let e = args.pop().expect("one argument");
                Ok(Over::Enumerate(self.src(e), start))
            }
            Expr::Call { callee, args, .. } if callee == "zip" && !self.defs.contains("zip") && n >= 2 && args.len() == n => {
                Ok(Over::Zip(args.into_iter().map(|a| self.src(a)).collect()))
            }
            e if n == 1 => Ok(Over::One(self.src(e))),
            other => {
                if let Expr::Call { callee, args, span } = &other {
                    if callee == "zip" && !self.defs.contains("zip") {
                        return Err(Diagnostic::error(format!("`zip` of {} sequences gives {} names, and this loop names {n}", args.len(), args.len()))
                            .with_primary(*span, "one name per sequence"));
                    }
                }
                Err(Diagnostic::error("`for i, x in ...` takes `enumerate(e)` or `zip(a, b, ...)`")
                    .with_primary(other.span(), "not a call of `enumerate` or `zip`")
                    .with_note("`for i, x in enumerate(s)` gives each index and element; `for a, b in zip(u, v)` walks two sequences together"))
            }
        }
    }

    /// A sequence of a `for` head: `rows(M)` is the rows of M, unless the
    /// program defines `rows`.
    fn src(&self, e: Expr) -> Src {
        match e {
            Expr::Call { callee, mut args, .. } if callee == "rows" && args.len() == 1 && !self.defs.contains("rows") => {
                Src { e: args.pop().expect("one argument"), rows: true }
            }
            e => Src { e, rows: false },
        }
    }

    /// The loop a head and a body make, appended to `out` with the hidden
    /// variables it assigns first.
    ///
    /// `for x in e { body }` is the plain loop over `e`'s indices, `e`
    /// evaluated once:
    ///
    /// ```text
    /// viv__itN = e
    /// for viv__kN to len(viv__itN) { x = viv__itN[viv__kN]; body }
    /// ```
    ///
    /// (with the enumerate form's `i` as the counter, or `i = k + start` when a
    /// start is given; `zip` runs to the shortest; the rows of `rows(M)` are
    /// `M[k, :]`, as many as `shape(M)[0]`).
    fn build_for(&mut self, head: Head, body: Vec<Stmt>, parallel: bool, span: Span, out: &mut Vec<Stmt>) -> PResult<()> {
        let (var, var_span, from, bound, by, reversed) = match head {
            Head::Count { var, var_span, from, bound, by, reversed, .. } => (var, var_span, from, bound, by, reversed),
            Head::Each { names, over, .. } => return self.build_each(names, over, body, span, out),
        };
        let plain = from.is_none() && by.is_none() && !reversed;
        if plain {
            out.push(Stmt::For { var, var_span, bound, body, parallel, span });
            return Ok(());
        }
        // A step written as a number is checked here.
        let literal = |e: &Expr| match e {
            Expr::Num { value, .. } => Some(*value),
            Expr::Neg { operand, .. } => match **operand {
                Expr::Num { value, .. } => Some(-value),
                _ => None,
            },
            _ => None,
        };
        if let Some(s) = by.as_ref().and_then(literal) {
            if s == 0.0 {
                return Err(Diagnostic::error("a `for` step of 0 never reaches the end")
                    .with_primary(by.as_ref().unwrap().span(), "step 0")
                    .with_note("`by s` adds s each pass: a positive s counts up to the end, a negative s counts down to it"));
            }
        }
        // a, b and s, each evaluated once, in order, into hidden variables
        // (a number stays a number)
        let num = |value: f64, span: Span| Expr::Num { value, span };
        let lo = match from {
            Some(e) if literal(&e).is_some() => num(literal(&e).unwrap(), e.span()),
            Some(e) => self.hidden_var(e, "lo", out),
            None => num(0.0, var_span),
        };
        let hi = if literal(&bound).is_some() { num(literal(&bound).unwrap(), bound.span()) } else { self.hidden_var(bound, "hi", out) };
        let step = match by {
            Some(e) if literal(&e).is_some() => num(literal(&e).unwrap(), e.span()),
            Some(e) => {
                let e_span = e.span();
                let v = self.hidden_var(e, "by", out);
                // a step of 0 known only at run time stops the run here
                out.push(Stmt::If {
                    cond: Expr::Binary { op: BinOp::EqEq, lhs: Box::new(v.clone()), rhs: Box::new(num(0.0, e_span)), span: e_span },
                    then_body: vec![Stmt::Lens {
                        lens: Box::new(Lens::Stop { message: "a `for` step of 0 never reaches the end".to_string(), kind: crate::errkind::ErrKind::Value }),
                        span: e_span,
                    }],
                    else_body: Vec::new(),
                    span: e_span,
                });
                v
            }
            None => num(1.0, var_span),
        };
        let is = |e: &Expr, x: f64| matches!(e, Expr::Num { value, .. } if *value == x);
        let bin = |op: BinOp, l: Expr, r: Expr| Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span };
        // the pass count: (b - a) / s; the plain loop already runs a
        // fractional count up to the next whole number, and none at all
        // for a count of 0 or less
        let mut count = if is(&lo, 0.0) { hi } else { bin(BinOp::Sub, hi, lo.clone()) };
        if !is(&step, 1.0) {
            count = bin(BinOp::Div, count, step.clone());
        }
        let k_name = self.hidden_name("k");
        let k = Expr::Var { name: k_name.clone(), span: var_span };
        // the pass number the value is computed from: k, or count - 1 - k
        let pass = if reversed {
            // reversed needs the whole number of passes
            let n = self.hidden_var(Expr::Call { callee: "ceil".to_string(), args: vec![count], span }, "n", out);
            count = n.clone();
            bin(BinOp::Sub, bin(BinOp::Sub, n, num(1.0, span)), k)
        } else {
            k
        };
        let offset = if is(&step, 1.0) { pass } else { bin(BinOp::Mul, pass, step) };
        let value = if is(&lo, 0.0) { offset } else { bin(BinOp::Add, lo, offset) };
        // the counts written here are integers: beside an Int bound they are
        // Ints too
        self.whole.entry((var_span.start, var_span.end)).or_insert(0);
        self.whole.entry((span.start, span.end)).or_insert(1);
        let mut full = Vec::with_capacity(body.len() + 1);
        full.push(Stmt::Assign { name: var, name_span: var_span, value, span: var_span });
        full.extend(body);
        out.push(Stmt::For { var: k_name, var_span, bound: count, body: full, parallel: false, span });
        Ok(())
    }

    /// `build_for` of the `for x in e` forms.
    fn build_each(&mut self, names: Vec<(String, Span)>, over: Over, body: Vec<Stmt>, span: Span, out: &mut Vec<Stmt>) -> PResult<()> {
        let mut names = names.into_iter();
        let (first, first_span) = names.next().expect("a loop names one variable at least");
        match over {
            Over::One(src) => {
                let seq_span = src.e.span();
                let it = self.hidden_var(src.e, "it", out);
                let k = self.hidden_name("k");
                let item = self.element(&it, Expr::Var { name: k.clone(), span: first_span }, src.rows, first_span);
                let mut full = Vec::with_capacity(body.len() + 1);
                full.push(Stmt::Assign { name: first, name_span: first_span, value: item, span: first_span });
                full.extend(body);
                let bound = self.length(it, src.rows, seq_span);
                out.push(Stmt::For { var: k, var_span: first_span, bound, body: full, parallel: false, span });
            }
            Over::Enumerate(src, start) => {
                let (x, x_span) = names.next().expect("two names");
                let seq_span = src.e.span();
                let it = self.hidden_var(src.e, "it", out);
                let mut full = Vec::with_capacity(body.len() + 2);
                // with a start, the counter is hidden and the index is
                // `k + start`; without one, the index is the counter
                // with an integer literal for a start (`start=1`) the index is
                // an Int, as Python's is: the counter counts in Ints to an Int
                // length
                let int_start = matches!(&start, Some(Expr::Num { span: sp, .. }) if self.whole.contains_key(&(sp.start, sp.end)));
                let mut int_bound = None;
                let declare = |name: &str, at: Span| Stmt::ExprStmt {
                    expr: Expr::Call {
                        callee: DECLARE.to_string(),
                        args: vec![Expr::Str { value: name.to_string(), span: at }, Expr::Str { value: "Int".to_string(), span: at }],
                        span: at,
                    },
                    span: at,
                };
                let (counter, counter_span) = match start {
                    None => (first, first_span),
                    Some(s) if int_start => {
                        let n = self.hidden_name("n");
                        let len = self.length(it.clone(), src.rows, seq_span);
                        let len = if src.rows { Expr::Call { callee: "int".to_string(), args: vec![len], span: seq_span } } else { len };
                        out.push(declare(&n, seq_span));
                        out.push(Stmt::Assign { name: n.clone(), name_span: seq_span, value: len, span: seq_span });
                        int_bound = Some(Expr::Var { name: n, span: seq_span });
                        out.push(declare(&first, first_span));
                        let k = self.hidden_name("k");
                        let kv = Expr::Var { name: k.clone(), span: first_span };
                        let value = Expr::Binary { op: BinOp::Add, lhs: Box::new(kv), rhs: Box::new(s), span: first_span };
                        full.push(Stmt::Assign { name: first, name_span: first_span, value, span: first_span });
                        (k, first_span)
                    }
                    Some(s) => {
                        let s = if matches!(s, Expr::Num { .. }) { s } else { self.hidden_var(s, "st", out) };
                        let k = self.hidden_name("k");
                        let kv = Expr::Var { name: k.clone(), span: first_span };
                        let value = Expr::Binary { op: BinOp::Add, lhs: Box::new(kv), rhs: Box::new(s), span: first_span };
                        full.push(Stmt::Assign { name: first, name_span: first_span, value, span: first_span });
                        (k, first_span)
                    }
                };
                let item = self.element(&it, Expr::Var { name: counter.clone(), span: counter_span }, src.rows, x_span);
                full.push(Stmt::Assign { name: x, name_span: x_span, value: item, span: x_span });
                full.extend(body);
                let bound = int_bound.unwrap_or_else(|| self.length(it, src.rows, seq_span));
                out.push(Stmt::For { var: counter, var_span: counter_span, bound, body: full, parallel: false, span });
            }
            Over::Zip(srcs) => {
                let mut its = Vec::with_capacity(srcs.len());
                for src in srcs {
                    let seq_span = src.e.span();
                    let it = self.hidden_var(src.e, "it", out);
                    its.push((it, src.rows, seq_span));
                }
                let k = self.hidden_name("k");
                let mut bound: Option<Expr> = None;
                let mut full = Vec::with_capacity(body.len() + its.len());
                let all = std::iter::once((first, first_span)).chain(names);
                for ((name, name_span), (it, rows, seq_span)) in all.zip(its) {
                    let item = self.element(&it, Expr::Var { name: k.clone(), span: first_span }, rows, name_span);
                    full.push(Stmt::Assign { name, name_span, value: item, span: name_span });
                    let n = self.length(it, rows, seq_span);
                    // `zip` stops at the end of the shortest
                    bound = Some(match bound {
                        None => n,
                        Some(b) => Expr::Call { callee: "min".to_string(), args: vec![b, n], span: seq_span },
                    });
                }
                full.extend(body);
                let bound = bound.expect("zip of two sequences at least");
                out.push(Stmt::For { var: k, var_span: first_span, bound, body: full, parallel: false, span });
            }
        }
        Ok(())
    }

    /// Element `k` of a `for` sequence held in `it`: `it[k]`, or the row
    /// `it[k, :]`.
    fn element(&self, it: &Expr, k: Expr, rows: bool, span: Span) -> Expr {
        if rows {
            let spec = slice_spec(vec![IndexItem::At(k), IndexItem::Range { lo: None, hi: None, step: None, span }], span);
            return Expr::Call { callee: SLICE_GET.to_string(), args: vec![it.clone(), spec], span };
        }
        Expr::Index { base: Box::new(it.clone()), indices: vec![k], span }
    }

    /// How many elements a `for` sequence held in `it` has: `len(it)`, or its
    /// number of rows, `shape(it)[0]`.
    fn length(&self, it: Expr, rows: bool, span: Span) -> Expr {
        if rows {
            let shape = Expr::Call { callee: "shape".to_string(), args: vec![it], span };
            return Expr::Index { base: Box::new(shape), indices: vec![Expr::Num { value: 0.0, span }], span };
        }
        Expr::Call { callee: "len".to_string(), args: vec![it], span }
    }

    /// A hidden variable assigned `e` here, even when `e` is a variable (a
    /// loop's bounds are fixed when it starts, and its body may change the
    /// variable).
    fn hidden_var(&mut self, e: Expr, what: &str, out: &mut Vec<Stmt>) -> Expr {
        let span = e.span();
        let name = self.hidden_name(what);
        out.push(Stmt::Assign { name: name.clone(), name_span: span, value: e, span });
        Expr::Var { name, span }
    }

    /// Parse an expression in an `if`/`while`/`for` head, where a trailing `{`
    /// opens the body block rather than a record literal.
    fn head_expr(&mut self) -> PResult<Expr> {
        let saved = self.no_record_lit;
        self.no_record_lit = true;
        let e = self.expr();
        self.no_record_lit = saved;
        e
    }

    fn ident(&mut self, what: &str) -> PResult<(String, Span)> {
        match &self.peek().tok {
            Tok::Ident(s) => {
                let s = s.clone();
                let span = self.span();
                self.bump();
                Ok((s, span))
            }
            _ => Err(self.err_here(format!("expected {}", what))),
        }
    }

    // Precedence ladder (low -> high): or, and, not, equality, relational,
    // additive, multiplicative (`*` `/` `@` `%` `//`), unary minus, power
    // (`^`), primary.
    fn expr(&mut self) -> PResult<Expr> {
        let e = self.or_expr()?;
        if self.at(&Tok::If) {
            return self.if_expr(e);
        }
        Ok(e)
    }

    /// `a if c else b`, after `a`: the value of `a` when `c` is true, else of
    /// `b`, and only the one chosen is computed. It binds loosest of all, as
    /// in Python; `a if c else b if d else e` is `a if c else (b if d else e)`.
    /// Written as `if c { t = a } else { t = b }` before the statement that
    /// uses it (crate::surface).
    fn if_expr(&mut self, then: Expr) -> PResult<Expr> {
        self.bump(); // `if`
        let cond = self.or_expr()?;
        if !self.at(&Tok::Else) {
            return Err(self.err_here("an `if` expression needs `else`: `a if c else b`"));
        }
        self.bump();
        let other = self.expr()?;
        let span = then.span().merge(other.span());
        let name = self.hidden_name("if");
        let assign = |value: Expr| Stmt::Assign { name: name.clone(), name_span: span, value, span };
        let stmts = vec![Stmt::If { cond, then_body: vec![assign(then)], else_body: vec![assign(other)], span }];
        Ok(self.lift(stmts, Expr::Var { name: name.clone(), span }, span))
    }

    /// A value computed by `stmts` into `result` (crate::surface).
    fn lift(&mut self, stmts: Vec<Stmt>, result: Expr, span: Span) -> Expr {
        self.lifted.push(Lifted { stmts, result });
        placeholder(self.lifted.len() - 1, span)
    }

    /// Parse a left-associative binary level given the next-tighter level and a
    /// token->op mapping.
    fn binary_level(
        &mut self,
        next: fn(&mut Self) -> PResult<Expr>,
        op_of: fn(&Tok) -> Option<BinOp>,
    ) -> PResult<Expr> {
        let mut lhs = next(self)?;
        while let Some(op) = op_of(&self.peek().tok) {
            self.bump();
            let rhs = next(self)?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn or_expr(&mut self) -> PResult<Expr> {
        self.binary_level(Self::and_expr, |t| matches!(t, Tok::Or).then_some(BinOp::Or))
    }

    fn and_expr(&mut self) -> PResult<Expr> {
        self.binary_level(Self::not_expr, |t| matches!(t, Tok::And).then_some(BinOp::And))
    }

    fn not_expr(&mut self) -> PResult<Expr> {
        if self.at(&Tok::Not) {
            let start = self.span();
            self.bump();
            let operand = self.not_expr()?;
            let span = start.merge(operand.span());
            return Ok(Expr::Not { operand: Box::new(operand), span });
        }
        self.equality()
    }

    fn equality(&mut self) -> PResult<Expr> {
        self.binary_level(Self::relational, |t| match t {
            Tok::EqEq => Some(BinOp::EqEq),
            Tok::Ne => Some(BinOp::Ne),
            _ => None,
        })
    }

    fn relational(&mut self) -> PResult<Expr> {
        let mut lhs = self.additive()?;
        // `lo <= x < hi`: Python's chain, `lo <= x and x < hi`, the middle
        // computed once (into a hidden variable unless it is a name or a number)
        let mut last: Option<Expr> = None;
        loop {
            // `a in b` and `a not in b`: membership, at the
            // level of the comparisons, as in Python
            let negated = self.at(&Tok::Not) && self.at_word(1, "in");
            if negated || self.at_word(0, "in") {
                self.bump();
                if negated {
                    self.bump();
                }
                let rhs = self.additive()?;
                let span = lhs.span().merge(rhs.span());
                let call = Expr::Call { callee: IN.to_string(), args: vec![lhs, rhs], span };
                lhs = if negated { Expr::Not { operand: Box::new(call), span } } else { call };
                continue;
            }
            let op = match self.peek().tok {
                Tok::Lt => BinOp::Lt,
                Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.bump();
            let rhs = self.additive()?;
            let span = lhs.span().merge(rhs.span());
            let simple = |e: &Expr| matches!(e, Expr::Var { .. } | Expr::Num { .. } | Expr::Str { .. });
            // the right side as this comparison reads it, and as the next one does
            let (rhs, next) = if simple(&rhs) || !matches!(self.peek().tok, Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge) {
                (rhs.clone(), rhs)
            } else {
                let at = rhs.span();
                let name = self.hidden_name("ch");
                let assign = Stmt::Assign { name: name.clone(), name_span: at, value: rhs, span: at };
                (self.lift(vec![assign], Expr::Var { name: name.clone(), span: at }, at), Expr::Var { name, span: at })
            };
            lhs = match last.take() {
                Some(mid) => {
                    let cmp = Expr::Binary { op, lhs: Box::new(mid), rhs: Box::new(rhs), span };
                    Expr::Binary { op: BinOp::And, lhs: Box::new(lhs), rhs: Box::new(cmp), span }
                }
                None => Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span },
            };
            last = Some(next);
        }
        Ok(lhs)
    }

    fn additive(&mut self) -> PResult<Expr> {
        self.binary_level(Self::multiplicative, |t| match t {
            Tok::Plus => Some(BinOp::Add),
            Tok::Minus => Some(BinOp::Sub),
            _ => None,
        })
    }

    fn multiplicative(&mut self) -> PResult<Expr> {
        self.binary_level(Self::unary, |t| match t {
            Tok::Star => Some(BinOp::Mul),
            Tok::Slash => Some(BinOp::Div),
            Tok::At => Some(BinOp::Matmul),
            Tok::Percent => Some(BinOp::Mod),
            Tok::SlashSlash => Some(BinOp::FloorDiv),
            _ => None,
        })
    }

    fn unary(&mut self) -> PResult<Expr> {
        if self.at(&Tok::Minus) {
            let start = self.span();
            self.bump();
            let operand = self.unary()?;
            let span = start.merge(operand.span());
            return Ok(Expr::Neg { operand: Box::new(operand), span });
        }
        self.power()
    }

    /// `x ^ y`, which is `pow(x, y)`: binds tighter than unary minus on its
    /// left (`-x ^ 2` is `-(x ^ 2)`), and is right-associative with a signed
    /// exponent on its right (`a ^ b ^ c` is `a ^ (b ^ c)`, `2 ^ -1` is 0.5),
    /// as Python's `**` and Julia's `^` are. It becomes the call, so it is
    /// `pow` in every engine, arrays and `deriv` included.
    fn power(&mut self) -> PResult<Expr> {
        let base = self.primary()?;
        if !self.at(&Tok::Caret) {
            return Ok(base);
        }
        self.bump();
        let exponent = self.unary()?;
        let span = base.span().merge(exponent.span());
        Ok(Expr::Call { callee: "pow".to_string(), args: vec![base, exponent], span })
    }

    /// One item inside `[...]`: an index `i`, or a range `lo:hi`, `lo:hi:s`,
    /// with any of the three left out (`:`, `lo:`, `:hi`, `::s`).
    fn index_item(&mut self) -> PResult<IndexItem> {
        let start = self.span();
        let lo = if self.at(&Tok::Colon) { None } else { Some(self.expr()?) };
        if !self.at(&Tok::Colon) {
            return Ok(IndexItem::At(lo.expect("an index item is an expression or has a `:`")));
        }
        let colon = self.bump().span;
        let ends = |p: &Self| p.at(&Tok::Comma) || p.at(&Tok::RBracket) || p.at(&Tok::Colon);
        let hi = if ends(self) { None } else { Some(self.expr()?) };
        let mut end = hi.as_ref().map_or(colon, |h| h.span());
        let step = if self.at(&Tok::Colon) {
            end = self.bump().span;
            if self.at(&Tok::Comma) || self.at(&Tok::RBracket) {
                None
            } else {
                let s = self.expr()?;
                end = s.span();
                Some(s)
            }
        } else {
            None
        };
        Ok(IndexItem::Range { lo, hi, step, span: start.merge(end) })
    }

    /// An atom followed by zero or more postfix operators: index brackets
    /// (`a[i][j]`) and field access (`p.x`, `m.node_x[i]`).
    fn primary(&mut self) -> PResult<Expr> {
        let mut e = self.atom()?;
        loop {
            if self.at(&Tok::LBracket) {
                self.bump();
                let mut indices = Vec::new();
                // Brackets are self-delimiting: record literals are allowed inside.
                let saved = self.no_record_lit;
                self.no_record_lit = false;
                let mut ranged = false;
                if !self.at(&Tok::RBracket) {
                    loop {
                        let item = self.index_item()?;
                        ranged |= !matches!(item, IndexItem::At(_));
                        indices.push(item);
                        if self.at(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.no_record_lit = saved;
                let close = self.expect(Tok::RBracket, "`]`")?;
                let span = e.span().merge(close.span);
                e = if ranged {
                    // a slice: a copy, read by a builtin call (interp/slices.rs)
                    let spec = slice_spec(indices, span);
                    Expr::Call { callee: SLICE_GET.to_string(), args: vec![e, spec], span }
                } else {
                    let indices = indices.into_iter().map(|i| if let IndexItem::At(x) = i { x } else { unreachable!() }).collect();
                    Expr::Index { base: Box::new(e), indices, span }
                };
            } else if self.at(&Tok::Dot) && matches!(&e, Expr::Var { name, .. } if self.enums.contains(name)) {
                // `Name.Variant` and `Name.Variant(a, b)`: an enum value
                // the call `Name.Variant(...)`
                self.bump();
                let Expr::Var { name, span: nspan } = e else { unreachable!() };
                let (variant, vspan) = self.ident("variant name")?;
                let mut args = Vec::new();
                let mut span = nspan.merge(vspan);
                if self.at(&Tok::LParen) {
                    let (a, kwargs, close, _) = self.call_args("")?;
                    if let Some((_, ks, _)) = kwargs.first() {
                        return Err(Diagnostic::error("a variant's payload is given in order, not by name").with_primary(*ks, "a keyword argument"));
                    }
                    args = a;
                    span = span.merge(close);
                }
                e = Expr::Call { callee: format!("{name}.{variant}"), args, span };
            } else if self.at(&Tok::Dot) {
                self.bump();
                let (field, field_span) = self.ident("field name")?;
                // `a.f(b, c)` is `f(a, b, c)`: a record has no function fields,
                // so a call here is never a field's
                if self.at(&Tok::LParen) {
                    let (mut args, kwargs, close, _) = self.call_args("")?;
                    args.insert(0, e);
                    let span = args[0].span().merge(close);
                    e = self.make_call(field, args, kwargs, span)?;
                    continue;
                }
                let span = e.span().merge(field_span);
                e = Expr::Field { base: Box::new(e), field, field_span, span };
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn atom(&mut self) -> PResult<Expr> {
        let tok = self.peek().tok.clone();
        let span = self.span();
        match tok {
            Tok::Num(v, whole) => {
                self.bump();
                if let Some(n) = whole {
                    self.whole.insert((span.start, span.end), n);
                }
                Ok(Expr::Num { value: v, span })
            }
            Tok::Imag(v) => {
                self.bump();
                Ok(Expr::Imag { value: v, span })
            }
            Tok::True | Tok::False => {
                let value = matches!(self.peek().tok, Tok::True);
                self.bump();
                Ok(Expr::Bool { value, span })
            }
            Tok::Str(_) | Tok::FStr(_) => {
                // literals side by side are one string, as in Python:
                // `"ab" "cd"` is "abcd"; an f-string among them is joined by `+`
                let mut e: Option<Expr> = None;
                loop {
                    let at = self.peek().span;
                    let part = match self.peek().tok.clone() {
                        Tok::Str(s) => {
                            self.bump();
                            Expr::Str { value: s, span: at }
                        }
                        Tok::FStr(parts) => {
                            self.bump();
                            self.fstring(parts, at)?
                        }
                        _ => break,
                    };
                    e = Some(match (e, part) {
                        (None, p) => p,
                        (Some(Expr::Str { value: a, span: sa }), Expr::Str { value: b, span: sb }) => Expr::Str { value: a + &b, span: sa.merge(sb) },
                        (Some(l), r) => {
                            let sp = l.span().merge(r.span());
                            Expr::Binary { op: BinOp::Add, lhs: Box::new(l), rhs: Box::new(r), span: sp }
                        }
                    });
                }
                Ok(e.expect("one literal at least"))
            }
            Tok::LBracket => self.arr_lit(),
            Tok::LBrace => self.map_comprehension(),
            Tok::Fn => self.lambda(),
            Tok::LParen => {
                self.bump();
                // Parentheses are self-delimiting: record literals allowed inside.
                let saved = self.no_record_lit;
                self.no_record_lit = false;
                let e = self.expr()?;
                self.no_record_lit = saved;
                self.expect(Tok::RParen, "`)`")?;
                Ok(e)
            }
            Tok::Ident(name) => {
                // `take the sum of v`, `integrate x^2 from 1 to 2` (crate::phrases)
                if (name == "take" || crate::phrases::is_bare_verb(&name)) && self.phrase_follows() {
                    return self.phrase(name == "take");
                }
                self.bump();
                // `None`: the empty optional, the call `None()`
                if name == "None" && !self.at(&Tok::LParen) {
                    return Ok(Expr::Call { callee: name, args: Vec::new(), span });
                }
                if self.at(&Tok::LParen) {
                    let (args, kwargs, close, done) = self.call_args(&name)?;
                    let whole = span.merge(close);
                    // `all(c for ...)` and `any(c for ...)`: the call is the loop
                    if done {
                        return Ok(args.into_iter().next().expect("the loop"));
                    }
                    // `chars("text")` is the one place a string literal becomes a
                    // value. It desugars here, to an array of Unicode scalar
                    // values, so that everything downstream -- both engines, the
                    // type checker, the content address -- sees an ordinary array
                    // literal and nothing else has to know about text at all.
                    if name == "chars" {
                        if let [Expr::Str { value, .. }] = args.as_slice() {
                            let elems = value
                                .chars()
                                .map(|c| Expr::Num { value: c as u32 as f64, span: whole })
                                .collect::<Vec<_>>();
                            if elems.is_empty() {
                                return Err(Diagnostic::error("`chars(\"\")` has no characters")
                                    .with_primary(whole, "an array cannot be empty"));
                            }
                            return Ok(Expr::ArrLit { elems, span: whole });
                        }
                        return Err(Diagnostic::error("`chars` takes one string literal, as in `chars(\"hello\")`")
                            .with_primary(whole, "not a string literal"));
                    }
                    self.make_call(name, args, kwargs, whole)
                } else if self.at(&Tok::LBrace) && !self.no_record_lit {
                    self.record_literal(name, span)
                } else {
                    Ok(Expr::Var { name, span })
                }
            }
            _ => {
                let mut d = self.err_here("expected an expression");
                // `x ** y`: the first `*` parsed as multiplication, and this
                // is the second, where the right operand should be
                if self.at(&Tok::Star) && self.pos > 0 && self.toks[self.pos - 1].tok == Tok::Star {
                    d = d.with_note(crate::hints::POW);
                }
                Err(d)
            }
        }
    }

    /// `( args )` after a callee: the arguments, the keyword arguments
    /// (`sep=" "`, `key=f`, `start=1`), and the span of the `)`. A generator
    /// as the only argument, `f(e for x in xs)`, is the comprehension
    /// `f([e for x in xs])`; for `all` and `any` it is their loop, which
    /// stops at the first element that decides, and the last value returned
    /// is true: the first argument is then the whole call.
    fn call_args(&mut self, callee: &str) -> PResult<(Vec<Expr>, Vec<(String, Span, Expr)>, Span, bool)> {
        let open = self.expect(Tok::LParen, "`(`")?;
        let mut args = Vec::new();
        let mut kwargs: Vec<(String, Span, Expr)> = Vec::new();
        let mut done = false;
        let saved = self.no_record_lit;
        self.no_record_lit = false;
        if !self.at(&Tok::RParen) {
            loop {
                if let (Tok::Ident(k), Some(Tok::Eq)) = (&self.peek().tok, self.toks.get(self.pos + 1).map(|t| &t.tok)) {
                    let (k, kspan) = (k.clone(), self.span());
                    self.bump();
                    self.bump();
                    if kwargs.iter().any(|(n, _, _)| *n == k) {
                        return Err(Diagnostic::error(format!("`{k}=` is given twice")).with_primary(kspan, "a second time"));
                    }
                    let v = self.expr()?;
                    kwargs.push((k, kspan, v));
                } else {
                    if let Some((k, kspan, _)) = kwargs.first() {
                        return Err(Diagnostic::error("an argument without a name cannot follow a keyword argument")
                            .with_primary(self.span(), format!("after `{k}=`"))
                            .with_secondary(*kspan, "the keyword argument")
                            .with_note("put the keyword arguments last"));
                    }
                    let e = self.expr()?;
                    if args.is_empty() && self.at(&Tok::For) {
                        let clauses = self.comp_clauses()?;
                        let span = open.span.merge(self.span());
                        let kind = match callee {
                            "all" if !self.defs.contains("all") => Collect::All,
                            "any" if !self.defs.contains("any") => Collect::Any,
                            _ => Collect::List,
                        };
                        args.push(self.comprehension(e, clauses, kind, span)?);
                        done = kind != Collect::List;
                        if !self.at(&Tok::RParen) {
                            return Err(self.err_here("a generator in a call is the call's only argument: write `f([e for x in xs], ...)`"));
                        }
                        break;
                    }
                    args.push(e);
                }
                if self.at(&Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.no_record_lit = saved;
        let close = self.expect(Tok::RParen, "`)`")?;
        Ok((args, kwargs, close.span, done))
    }

    /// The call `name(args, kwargs)`. The keyword arguments are those of
    /// `print` (`sep=`), `max` and `min` (`key=`), `sorted` and `argsort`
    /// (`key=`, `reverse=`) and `enumerate` (`start=`), for the builtins; a
    /// program's own def takes none.
    fn make_call(&mut self, name: String, args: Vec<Expr>, kwargs: Vec<(String, Span, Expr)>, span: Span) -> PResult<Expr> {
        let builtin = !self.defs.contains(&name);
        if name == "print" && builtin {
            return self.print_call(args, kwargs, span);
        }
        // a def of the program, filled out from its signature (crate::kwargs);
        // a call through a function value (a parameter or local of that
        // name) passes its arguments in order
        if !builtin {
            if self.scope.contains(&name) {
                if let Some((k, kspan, _)) = kwargs.first() {
                    return Err(Diagnostic::error(format!("`{k}=`: a call through `{name}` passes its arguments in order"))
                        .with_primary(*kspan, "a keyword argument")
                        .with_note(format!("`{name}` is a value here (a parameter or a variable), and the def it holds is not known until the call runs")));
                }
            } else if let Some(sig) = self.sigs.get(&name).cloned() {
                return self.fill_call(name, sig, args, kwargs, span);
            }
        }
        let (name, args, kwargs) = if builtin { self.ordered_option(name, args, kwargs, span)? } else { (name, args, kwargs) };
        if builtin {
            if let Some(o) = crate::kwargs::options(&name) {
                return self.builtin_options(name, o, args, kwargs, span);
            }
        }
        // `fmt(x, f".{p}f")`: a format whose width or precision is a value is
        // `fmt(x, ".*f", p)`, as an f-string field `{x:.{p}f}` is
        let args = match (builtin && name == "fmt", args.as_slice()) {
            (true, [x, spec]) => match crate::printfmt::star_spec(spec) {
                Some((text, fields)) => {
                    let mut a = vec![x.clone(), Expr::Str { value: text, span: spec.span() }];
                    a.extend(fields);
                    a
                }
                None => args,
            },
            _ => args,
        };
        if kwargs.is_empty() {
            return Ok(Expr::Call { callee: name, args, span });
        }
        let allowed: &[&str] = match name.as_str() {
            _ if !builtin => &[],
            "max" | "min" => &["key"],
            "sorted" | "argsort" => &["key", "reverse"],
            "enumerate" => &["start"],
            _ => &[],
        };
        for (k, kspan, _) in &kwargs {
            if !allowed.contains(&k.as_str()) {
                let what = if !builtin {
                    format!("`{name}` is a def of this program, and a def takes its arguments in order")
                } else if allowed.is_empty() {
                    format!("`{name}` takes no keyword arguments")
                } else {
                    format!("`{name}` takes {}", allowed.iter().map(|a| format!("`{a}=`")).collect::<Vec<_>>().join(" and "))
                };
                return Err(Diagnostic::error(format!("unknown keyword argument `{k}=`"))
                    .with_primary(*kspan, "not an argument of this call")
                    .with_note(format!("{what}; the builtins with keyword arguments are print(sep=), max and min (key=), sorted and argsort (key=, reverse=) and enumerate (start=)")));
            }
        }
        let mut kw: HashMap<String, (Span, Expr)> = kwargs.into_iter().map(|(k, s, v)| (k, (s, v))).collect();
        if name == "enumerate" {
            let mut args = args;
            if let Some((_, s)) = kw.remove("start") {
                if args.len() != 1 {
                    return Err(Diagnostic::error("`enumerate` takes one sequence and a start").with_primary(span, "`start=` and a second argument"));
                }
                args.push(s);
            }
            return Ok(Expr::Call { callee: name, args, span });
        }
        self.key_call(name, args, kw, span)
    }

    /// A call of a def, filled out (crate::kwargs): the positional arguments
    /// to the parameters without defaults, the keyword arguments by name,
    /// and each default left out written in, reading the arguments before it.
    fn fill_call(&mut self, name: String, sig: Vec<crate::kwargs::SigParam>, args: Vec<Expr>, kwargs: Vec<(String, Span, Expr)>, span: Span) -> PResult<Expr> {
        if kwargs.is_empty() && sig.iter().all(|p| p.default.is_none()) {
            return Ok(Expr::Call { callee: name, args, span });
        }
        let required = sig.iter().take_while(|p| p.default.is_none()).count();
        if args.len() > required {
            let extra = &args[required];
            return Err(match sig.get(required) {
                Some(p) => Diagnostic::error(format!("`{}` has a default, so it is given by name: `{}=...`", p.name, p.name))
                    .with_primary(extra.span(), format!("passed by position to `{name}`"))
                    .with_note("a parameter with a default is an option, and every option a call sets is named at the call"),
                None => Diagnostic::error(format!("`{name}` takes {} argument{}, and this call passes {}", sig.len(), if sig.len() == 1 { "" } else { "s" }, args.len())).with_primary(extra.span(), "one too many"),
            });
        }
        let mut slots: Vec<Option<Expr>> = vec![None; sig.len()];
        for (i, a) in args.into_iter().enumerate() {
            slots[i] = Some(a);
        }
        for (k, kspan, v) in kwargs {
            let Some(i) = sig.iter().position(|p| p.name == k) else {
                let names: Vec<String> = sig.iter().map(|p| if p.default.is_some() { format!("{}=", p.name) } else { p.name.clone() }).collect();
                return Err(Diagnostic::error(format!("`{name}` has no parameter `{k}`"))
                    .with_primary(kspan, "not a parameter of this def")
                    .with_note(format!("`{name}({})`", names.join(", "))));
            };
            if slots[i].is_some() {
                return Err(Diagnostic::error(format!("`{k}` is given twice: by position and by name")).with_primary(kspan, "the second time"));
            }
            slots[i] = Some(v);
        }
        // the arguments a default reads, each computed once
        let reads: std::collections::HashSet<String> = sig
            .iter()
            .filter_map(|p| p.default.as_ref())
            .flat_map(|d| d.iter().filter_map(|t| if let Tok::Ident(n) = &t.tok { Some(n.clone()) } else { None }))
            .collect();
        let mut out: Vec<Expr> = Vec::with_capacity(sig.len());
        let mut bound: Vec<(String, Expr)> = Vec::new();
        for (i, p) in sig.iter().enumerate() {
            let e = match slots[i].take() {
                Some(e) => e,
                None => match &p.default {
                    None => return Err(Diagnostic::error(format!("`{name}` needs `{}`", p.name)).with_primary(span, "missing here")),
                    Some(toks) => {
                        let d = self.default_expr(toks)?;
                        substitute(d, &bound)
                    }
                },
            };
            let e = if reads.contains(&p.name) && !matches!(e, Expr::Num { .. } | Expr::Var { .. } | Expr::Bool { .. } | Expr::Str { .. }) {
                let at = e.span();
                let v = self.hidden_name("arg");
                let ph = self.lift(vec![Stmt::Assign { name: v.clone(), name_span: at, value: e, span: at }], Expr::Var { name: v.clone(), span: at }, at);
                bound.push((p.name.clone(), Expr::Var { name: v, span: at }));
                ph
            } else {
                bound.push((p.name.clone(), e.clone()));
                e
            };
            out.push(e);
        }
        Ok(Expr::Call { callee: name, args: out, span })
    }

    /// A default's expression, from the tokens of the def's head.
    fn default_expr(&mut self, toks: &[Token]) -> PResult<Expr> {
        let mut q = Parser::new(toks.to_vec(), self.defs.clone());
        q.globals = self.globals.clone();
        q.enums = self.enums.clone();
        q.sigs = self.sigs.clone();
        q.hidden = self.hidden;
        q.cur_def = self.cur_def.clone();
        let e = q.expr()?;
        self.hidden = q.hidden;
        self.whole.extend(q.whole.drain());
        // a default's own lifted values (a comprehension) are refused where the def is read
        Ok(e)
    }

    /// `ordered=` (crate::kwargs::ORDERED): `sum(a, ordered=true)` adds every
    /// entry left to right from 0.0, the order of a Fortran or C loop, and is
    /// the call `__sum_ordered(a)`; `sum(a, ordered=false)` is `sum(a)`, in
    /// NumPy's order. `dot` and `prod` always go left to right, so on them
    /// `ordered=true` states the order the code depends on and changes nothing.
    /// The value is `true` or `false` as written: the order is chosen when the
    /// program is compiled.
    #[allow(clippy::type_complexity)]
    fn ordered_option(&mut self, name: String, args: Vec<Expr>, mut kwargs: Vec<(String, Span, Expr)>, span: Span) -> PResult<(String, Vec<Expr>, Vec<(String, Span, Expr)>)> {
        let Some(k) = kwargs.iter().position(|(n, _, _)| n == "ordered") else {
            return Ok((name, args, kwargs));
        };
        if !crate::kwargs::ORDERED.contains(&name.as_str()) {
            return Ok((name, args, kwargs));
        }
        let (_, kspan, v) = kwargs.remove(k);
        let Expr::Bool { value, .. } = v else {
            return Err(Diagnostic::error("`ordered=` is `true` or `false` as written")
                .with_primary(v.span(), "not a literal true or false")
                .with_note("the order of the additions is chosen when the program is compiled"));
        };
        if !value {
            if name != "sum" {
                return Err(Diagnostic::error(format!("`{name}` always goes left to right; there is no `ordered=false`"))
                    .with_primary(kspan, "`ordered=false`"));
            }
            return Ok((name, args, kwargs));
        }
        if name != "sum" {
            return Ok((name, args, kwargs));
        }
        if let Some((other, ospan, _)) = kwargs.first() {
            return Err(Diagnostic::error(format!("`sum(a, ordered=true)` adds every entry; it takes no `{other}=`"))
                .with_primary(*ospan, "with `ordered=true`")
                .with_note("an ordered sum along an axis is a loop over the other axes, each summing a slice"));
        }
        if args.len() != 1 {
            return Err(Diagnostic::error("`sum(a, ordered=true)` takes one array").with_primary(span, "the call"));
        }
        Ok(("__sum_ordered".to_string(), args, kwargs))
    }

    /// A builtin's options by name (crate::kwargs::options): written into
    /// their slots of the positional call the engines read.
    fn builtin_options(&mut self, name: String, o: crate::kwargs::Options, mut args: Vec<Expr>, kwargs: Vec<(String, Span, Expr)>, span: Span) -> PResult<Expr> {
        if args.len() > o.operands && !POSITIONAL_OPTIONS.with(|c| c.get()) {
            let (opt, _) = o.opts.iter().find(|(_, s)| *s == o.operands).copied().unwrap_or(("axis", o.operands));
            return Err(Diagnostic::error(format!("`{opt}` is an option of `{name}`, so it is given by name: `{opt}=...`"))
                .with_primary(args[o.operands].span(), "passed by position")
                .with_note(format!("{}; `viv fix --keywords` rewrites a file", crate::kwargs::positional_note(&name, &o, args.len()))));
        }
        if kwargs.is_empty() {
            return Ok(Expr::Call { callee: name, args, span });
        }
        let top = o.opts.iter().map(|(_, s)| *s).max().unwrap_or(0);
        let mut slots: Vec<Option<Expr>> = vec![None; top + 1];
        let given = args.len();
        for (i, a) in args.drain(..).enumerate() {
            slots[i] = Some(a);
        }
        for (k, kspan, v) in kwargs {
            let Some(&(_, s)) = o.opts.iter().find(|(n, _)| *n == k) else {
                return Err(Diagnostic::error(format!("`{name}` has no option `{k}=`"))
                    .with_primary(kspan, "not an option of this builtin")
                    .with_note(format!("`{name}` takes {}", o.opts.iter().map(|(n, _)| format!("`{n}=`")).collect::<Vec<_>>().join(", "))));
            };
            if s < given || slots[s].is_some() {
                return Err(Diagnostic::error(format!("`{k}=` is given twice")).with_primary(kspan, "the second time"));
            }
            slots[s] = Some(v);
        }
        let last = slots.iter().rposition(|s| s.is_some()).unwrap_or(0);
        let mut out = Vec::with_capacity(last + 1);
        for (i, s) in slots.into_iter().take(last + 1).enumerate() {
            match s {
                Some(e) => out.push(e),
                None => {
                    let opt = o.opts.iter().find(|(_, x)| *x == i).map(|(n, _)| *n).unwrap_or("?");
                    match crate::kwargs::gap(&name, opt) {
                        Some(v) => out.push(Expr::Num { value: v, span }),
                        None => {
                            let later = o.opts.iter().find(|(_, x)| *x > i).map(|(n, _)| *n).unwrap_or("?");
                            return Err(Diagnostic::error(format!("`{name}`'s `{later}=` needs `{opt}=`")).with_primary(span, format!("`{opt}=` is left out")));
                        }
                    }
                }
            }
        }
        Ok(Expr::Call { callee: name, args: out, span })
    }

    /// `print(a, b, ..., sep=" ")`: the arguments with `sep` between them, a
    /// space when it is left out, as Python writes them. The separator is
    /// joined to the text beside it (`print("x =", x)` becomes the call
    /// `print("x = ", x)`), and the engines write the arguments of the call
    /// they get one after another, so they run the calls they always ran.
    /// `sep` is a string literal or a variable. There is no `end=`: every
    /// `print` ends its line.
    fn print_call(&mut self, args: Vec<Expr>, kwargs: Vec<(String, Span, Expr)>, span: Span) -> PResult<Expr> {
        let mut sep = Expr::Str { value: " ".to_string(), span };
        for (k, kspan, v) in kwargs {
            match k.as_str() {
                "sep" => match v {
                    Expr::Str { .. } | Expr::Var { .. } => sep = v,
                    other => {
                        return Err(Diagnostic::error("`sep=` is a string literal or a variable")
                            .with_primary(other.span(), "computed here")
                            .with_note("assign the separator to a variable first"))
                    }
                },
                "end" => {
                    return Err(Diagnostic::error("`print` has no `end=`: every print ends its line")
                        .with_primary(kspan, "not an argument of `print`")
                        .with_note("build the whole line first, as an f-string or with `join`, and print it once"))
                }
                other => {
                    return Err(Diagnostic::error(format!("unknown keyword argument `{other}=`"))
                        .with_primary(kspan, "not an argument of `print`")
                        .with_note("`print` takes `sep=`, the text between its arguments (a space when left out)"))
                }
            }
        }
        let mut out: Vec<Expr> = Vec::with_capacity(2 * args.len());
        for (i, mut a) in args.into_iter().enumerate() {
            if i > 0 {
                match &sep {
                    Expr::Str { value, .. } if value.is_empty() => {}
                    Expr::Str { value, span: at } => {
                        if let Some(Expr::Str { value: before, .. }) = out.last_mut() {
                            before.push_str(value);
                        } else if let Expr::Str { value: next, .. } = &mut a {
                            next.insert_str(0, value);
                        } else {
                            out.push(Expr::Str { value: value.clone(), span: *at });
                        }
                    }
                    other => out.push(other.clone()),
                }
            }
            out.push(a);
        }
        Ok(Expr::Call { callee: "print".to_string(), args: out, span })
    }

    /// The clauses of a comprehension after its element: `for` heads (any
    /// head a `for` statement takes) and `if` filters.
    fn comp_clauses(&mut self) -> PResult<Vec<Clause>> {
        let mut clauses = Vec::new();
        loop {
            if self.at(&Tok::For) {
                self.bump();
                clauses.push(Clause::For(self.for_head(true)?));
            } else if self.at(&Tok::If) && !clauses.is_empty() {
                self.bump();
                clauses.push(Clause::If(self.or_expr()?));
            } else {
                break;
            }
        }
        Ok(clauses)
    }

    /// `[e for x in xs if c ...]`, and the generator forms of `all` and
    /// `any`: written as a hidden array grown by `append` in the loops the
    /// clauses name, computed before the statement that uses it
    /// (crate::surface):
    ///
    /// ```text
    /// viv__cmpN = []
    /// for x in xs { if c { viv__cmpN = append(viv__cmpN, e) } }
    /// ```
    ///
    /// Numbers make a vector, vectors of one length a matrix (a row each),
    /// strings a list of strings. The loop variables are hidden variables, so
    /// a comprehension neither reads nor changes a variable of the same name
    /// outside it, as in Python 3.
    fn comprehension(&mut self, elem: Expr, clauses: Vec<Clause>, kind: Collect, span: Span) -> PResult<Expr> {
        let mut map: HashMap<String, String> = HashMap::new();
        let mut done = Vec::with_capacity(clauses.len());
        for c in clauses {
            match c {
                Clause::For(mut head) => {
                    self.subst_head(&mut head, &map);
                    self.rename_head(&mut head, &mut map);
                    done.push(Clause::For(head));
                }
                Clause::If(mut cond) => {
                    self.subst(&mut cond, &map);
                    done.push(Clause::If(cond));
                }
            }
        }
        let mut elem = elem;
        self.subst(&mut elem, &map);
        let name = self.hidden_name("cmp");
        let res = Expr::Var { name: name.clone(), span };
        let assign = |value: Expr| Stmt::Assign { name: name.clone(), name_span: span, value, span };
        let bool_ = |value: bool| Expr::Bool { value, span };
        let (init, inner) = match kind {
            Collect::List => {
                let grow = Expr::Call { callee: "append".to_string(), args: vec![res.clone(), elem], span };
                (Expr::ArrLit { elems: Vec::new(), span }, assign(grow))
            }
            Collect::All => {
                let cond = Expr::Not { operand: Box::new(elem), span };
                (bool_(true), Stmt::If { cond, then_body: vec![assign(bool_(false)), Stmt::Break { span }], else_body: Vec::new(), span })
            }
            Collect::Any => (bool_(false), Stmt::If { cond: elem, then_body: vec![assign(bool_(true)), Stmt::Break { span }], else_body: Vec::new(), span }),
        };
        let mut stmts = vec![assign(init)];
        self.nest(done, inner, kind, &res, span, &mut stmts)?;
        Ok(self.lift(stmts, res, span))
    }

    /// The loops and filters of a comprehension around `inner`, outermost
    /// first. For `all` and `any`, a loop around another stops as soon as the
    /// inner one has decided.
    fn nest(&mut self, mut clauses: Vec<Clause>, inner: Stmt, kind: Collect, res: &Expr, span: Span, out: &mut Vec<Stmt>) -> PResult<()> {
        if clauses.is_empty() {
            out.push(inner);
            return Ok(());
        }
        let first = clauses.remove(0);
        let more_loops = clauses.iter().any(|c| matches!(c, Clause::For(_)));
        let mut body = Vec::new();
        self.nest(clauses, inner, kind, res, span, &mut body)?;
        match first {
            Clause::If(cond) => out.push(Stmt::If { cond, then_body: body, else_body: Vec::new(), span }),
            Clause::For(head) => {
                if kind != Collect::List && more_loops {
                    let decided = if kind == Collect::All { Expr::Not { operand: Box::new(res.clone()), span } } else { res.clone() };
                    body.push(Stmt::If { cond: decided, then_body: vec![Stmt::Break { span }], else_body: Vec::new(), span });
                }
                self.build_for(head, body, false, span, out)?;
            }
        }
        Ok(())
    }

    /// Give the names a comprehension's `for` binds hidden names, recorded in
    /// `map` for what follows.
    fn rename_head(&mut self, head: &mut Head, map: &mut HashMap<String, String>) {
        let mut rename = |p: &mut Self, n: &mut String| {
            let h = p.hidden_name("cv");
            map.insert(n.clone(), h.clone());
            *n = h;
        };
        match head {
            Head::Count { var, .. } => rename(self, var),
            Head::Each { names, .. } => {
                for (n, _) in names.iter_mut() {
                    rename(self, n);
                }
            }
        }
    }

    /// `subst` over a head's expressions (not the names it binds).
    fn subst_head(&mut self, head: &mut Head, map: &HashMap<String, String>) {
        match head {
            Head::Count { from, bound, by, .. } => {
                if let Some(e) = from {
                    self.subst(e, map);
                }
                self.subst(bound, map);
                if let Some(e) = by {
                    self.subst(e, map);
                }
            }
            Head::Each { over, .. } => match over {
                Over::One(s) => self.subst(&mut s.e, map),
                Over::Enumerate(s, start) => {
                    self.subst(&mut s.e, map);
                    if let Some(e) = start {
                        self.subst(e, map);
                    }
                }
                Over::Zip(srcs) => {
                    for s in srcs.iter_mut() {
                        self.subst(&mut s.e, map);
                    }
                }
            },
        }
    }

    /// Rename the variables `map` names, in `e` and in the statements of any
    /// value lifted inside it. A lambda's parameter hides a name of the same
    /// spelling in its body.
    fn subst(&mut self, e: &mut Expr, map: &HashMap<String, String>) {
        if map.is_empty() {
            return;
        }
        if let Some(id) = crate::surface::placeholder_id(e) {
            let span = e.span();
            let mut l = std::mem::replace(&mut self.lifted[id], Lifted { stmts: Vec::new(), result: Expr::Bool { value: false, span } });
            for s in l.stmts.iter_mut() {
                self.subst_stmt(s, map);
            }
            self.subst(&mut l.result, map);
            self.lifted[id] = l;
            return;
        }
        match e {
            Expr::Var { name, .. } => {
                if let Some(h) = map.get(name) {
                    *name = h.clone();
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Some(h) = map.get(callee) {
                    *callee = h.clone();
                }
                for a in args.iter_mut() {
                    self.subst(a, map);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for a in elems.iter_mut() {
                    self.subst(a, map);
                }
            }
            Expr::Index { base, indices, .. } => {
                self.subst(base, map);
                for a in indices.iter_mut() {
                    self.subst(a, map);
                }
            }
            Expr::Lambda { params, body, .. } => {
                let inner: HashMap<String, String> = map.iter().filter(|(k, _)| !params.contains(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
                self.subst(body, &inner);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.subst(operand, map),
            Expr::Binary { lhs, rhs, .. } => {
                self.subst(lhs, map);
                self.subst(rhs, map);
            }
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields.iter_mut() {
                    self.subst(v, map);
                }
            }
            Expr::Field { base, .. } => self.subst(base, map),
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }

    fn subst_block(&mut self, body: &mut [Stmt], map: &HashMap<String, String>) {
        for s in body.iter_mut() {
            self.subst_stmt(s, map);
        }
    }

    fn subst_stmt(&mut self, s: &mut Stmt, map: &HashMap<String, String>) {
        let rename = |n: &mut String| {
            if let Some(h) = map.get(n.as_str()) {
                *n = h.clone();
            }
        };
        match s {
            Stmt::Assign { name, value, .. } => {
                rename(name);
                self.subst(value, map);
            }
            Stmt::Return { value: e, .. } | Stmt::ExprStmt { expr: e, .. } | Stmt::Show { expr: e, .. } => self.subst(e, map),
            Stmt::SetIndex { base, indices, value, .. } => {
                self.subst(base, map);
                for i in indices.iter_mut() {
                    self.subst(i, map);
                }
                self.subst(value, map);
            }
            Stmt::SetField { base, value, .. } => {
                self.subst(base, map);
                self.subst(value, map);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.subst(cond, map);
                self.subst_block(then_body, map);
                self.subst_block(else_body, map);
            }
            Stmt::While { cond, body, .. } => {
                self.subst(cond, map);
                self.subst_block(body, map);
            }
            Stmt::For { var, bound, body, .. } => {
                rename(var);
                self.subst(bound, map);
                self.subst_block(body, map);
            }
            Stmt::Test { body, .. } => self.subst_block(body, map),
            Stmt::Lens { lens, .. } => match &mut **lens {
                Lens::Iter { end, .. } => self.subst_block(end, map),
                Lens::Hold { check, .. } => match check {
                    HoldCheck::Cond(e) => self.subst(e, map),
                    HoldCheck::Near { diff, tol } => {
                        self.subst(diff, map);
                        self.subst(tol, map);
                    }
                },
                Lens::Bind { value, .. } => self.subst(value, map),
                Lens::Call { .. } | Lens::IterEnd { .. } | Lens::Watch { .. } | Lens::Placeholder { .. } | Lens::Stop { .. } | Lens::Inline { .. } | Lens::Ret { .. } => {}
            },
            Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } => {}
        }
    }

    /// `f(v)` for the key of a `key=` call: `v` itself when there is no key,
    /// the lambda's body with its parameter replaced by `v` (`v` is a
    /// variable), a call of a def, builtin or function-valued variable named
    /// by the key, or `apply` of any other function value.
    fn apply_key(&mut self, key: Option<&Expr>, v: Expr) -> PResult<Expr> {
        let span = v.span();
        match key {
            None => Ok(v),
            Some(Expr::Lambda { params, body, span: lspan }) => {
                let [p] = params.as_slice() else {
                    return Err(Diagnostic::error("a key takes one argument: `key=fn(x) -> ...`").with_primary(*lspan, format!("{} parameters", params.len())));
                };
                let Expr::Var { name, .. } = &v else { unreachable!("a key is applied to a variable") };
                let map = HashMap::from([(p.clone(), name.clone())]);
                let mut b = (**body).clone();
                self.subst(&mut b, &map);
                Ok(b)
            }
            Some(Expr::Var { name, .. }) => Ok(Expr::Call { callee: name.clone(), args: vec![v], span }),
            Some(other) => Ok(Expr::Call { callee: "apply".to_string(), args: vec![other.clone(), v], span }),
        }
    }

    /// `max(xs, key=f)` and `min(xs, key=f)`: the first element whose key is
    /// largest (smallest), as Python's are: a later element replaces it only
    /// when its key is strictly larger, so ties keep the first and a NaN key
    /// is never chosen after the first element. `sorted(xs, key=f,
    /// reverse=r)`: the elements in the order of their keys, stable (equal
    /// keys keep their order, with `reverse` too, as in Python), a NaN key
    /// after every number. `argsort` gives the positions in that order
    /// instead. `xs` may be `rows(M)`.
    ///
    /// Written as loops over hidden variables before the statement that uses
    /// the value (crate::surface); the sort is a merge sort on the keys, so
    /// keys may be numbers or strings.
    fn key_call(&mut self, name: String, mut args: Vec<Expr>, mut kw: HashMap<String, (Span, Expr)>, span: Span) -> PResult<Expr> {
        if args.len() != 1 {
            return Err(Diagnostic::error(format!("`{name}` with a keyword argument takes one sequence"))
                .with_primary(span, format!("{} arguments", args.len()))
                .with_note("`max(xs, key=f)` is the element of xs whose f is largest"));
        }
        let key = kw.remove("key").map(|(_, e)| e);
        let reverse = match kw.remove("reverse") {
            None => false,
            Some((_, Expr::Bool { value, .. })) => value,
            Some((at, _)) => return Err(Diagnostic::error("`reverse=` is `true` or `false`, written in the source").with_primary(at, "not true or false")),
        };
        let src = self.src(args.pop().expect("one argument"));
        let seq_span = src.e.span();
        let num = |value: f64| Expr::Num { value, span };
        let var = |name: &str| Expr::Var { name: name.to_string(), span };
        let bin = |op: BinOp, l: Expr, r: Expr| Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span };
        let assign = |name: &str, value: Expr| Stmt::Assign { name: name.to_string(), name_span: span, value, span };
        let mut out = Vec::new();
        let it = self.hidden_var(src.e, "it", &mut out);
        let len = self.length(it.clone(), src.rows, seq_span);
        let n = self.hidden_var(len, "n", &mut out);
        let k = self.hidden_name("k");
        let x = self.hidden_name("x");
        let item = self.element(&it, var(&k), src.rows, span);
        // the keys, each computed once, in order
        let ks = self.hidden_name("ks");
        out.push(assign(&ks, Expr::ArrLit { elems: Vec::new(), span }));
        let kval = self.apply_key(key.as_ref(), var(&x))?;
        let grow = Expr::Call { callee: "append".to_string(), args: vec![var(&ks), kval], span };
        out.push(Stmt::For { var: k, var_span: span, bound: n.clone(), body: vec![assign(&x, item), assign(&ks, grow)], parallel: false, span });
        if name == "max" || name == "min" {
            // the first position whose key is larger (smaller) than every
            // key before it, as Python's max and min compare
            out.insert(
                2,
                Stmt::If {
                    cond: bin(BinOp::EqEq, n.clone(), num(0.0)),
                    then_body: vec![Stmt::Lens { lens: Box::new(Lens::Stop { message: format!("{name}() arg is an empty sequence"), kind: crate::errkind::ErrKind::Value }), span }],
                    else_body: Vec::new(),
                    span,
                },
            );
            let (best, j) = (self.hidden_name("best"), self.hidden_name("k"));
            out.push(assign(&best, num(0.0)));
            let key_at = |at: Expr| Expr::Index { base: Box::new(var(&ks)), indices: vec![at], span };
            let better = bin(if name == "max" { BinOp::Gt } else { BinOp::Lt }, key_at(var(&j)), key_at(var(&best)));
            let body = vec![Stmt::If { cond: better, then_body: vec![assign(&best, var(&j))], else_body: Vec::new(), span }];
            out.push(Stmt::For { var: j, var_span: span, bound: n.clone(), body, parallel: false, span });
            let picked = self.element(&it, var(&best), src.rows, span);
            let res = self.hidden_name("max");
            out.push(assign(&res, picked));
            return Ok(self.lift(out, var(&res), span));
        }
        // the order the keys sort in
        let ord = self.argsort_stmts(&ks, &n, reverse, span, &mut out);
        if name == "argsort" {
            return Ok(self.lift(out, ord, span));
        }
        let res = self.hidden_name("cmp");
        let j = self.hidden_name("k");
        out.push(assign(&res, Expr::ArrLit { elems: Vec::new(), span }));
        let at = Expr::Index { base: Box::new(ord), indices: vec![var(&j)], span };
        let picked = self.element(&it, at, src.rows, span);
        let grow = Expr::Call { callee: "append".to_string(), args: vec![var(&res), picked], span };
        out.push(Stmt::For { var: j, var_span: span, bound: n, body: vec![assign(&res, grow)], parallel: false, span });
        Ok(self.lift(out, var(&res), span))
    }

    /// Statements that leave in a new hidden variable (returned) the
    /// positions 0..n of the keys `ks` in sorted order: a bottom-up merge
    /// sort, stable, that takes the right run's element first only when it
    /// comes strictly before the left run's.
    fn argsort_stmts(&mut self, ks: &str, n: &Expr, reverse: bool, span: Span, out: &mut Vec<Stmt>) -> Expr {
        let num = |value: f64| Expr::Num { value, span };
        let var = |name: &str| Expr::Var { name: name.to_string(), span };
        let bin = |op: BinOp, l: Expr, r: Expr| Expr::Binary { op, lhs: Box::new(l), rhs: Box::new(r), span };
        let assign = |name: &str, value: Expr| Stmt::Assign { name: name.to_string(), name_span: span, value, span };
        let index = |a: &str, i: Expr| Expr::Index { base: Box::new(var(a)), indices: vec![i], span };
        let set = |a: &str, i: Expr, value: Expr| Stmt::SetIndex { base: Box::new(var(a)), indices: vec![i], value, span };
        let call = |f: &str, args: Vec<Expr>| Expr::Call { callee: f.to_string(), args, span };
        let [ord, tmp, sw, w, lo, mid, hi, i, j, q, r, p] =
            ["ord", "tmp", "sw", "w", "lo", "mid", "hi", "i", "j", "q", "r", "p"].map(|what| self.hidden_name(what));
        out.push(assign(&ord, call("zeros", vec![n.clone()])));
        out.push(Stmt::For { var: q.clone(), var_span: span, bound: n.clone(), body: vec![set(&ord, var(&q), var(&q))], parallel: false, span });
        out.push(assign(&tmp, call("zeros", vec![n.clone()])));
        out.push(assign(&w, num(1.0)));
        // key a before key b: a NaN comes after every number, either way
        let key_of = |at: &str| index(ks, index(&ord, var(at)));
        let (a, b) = (key_of(&j), key_of(&i));
        let order = bin(if reverse { BinOp::Gt } else { BinOp::Lt }, a.clone(), b.clone());
        let nan_last = bin(BinOp::And, bin(BinOp::EqEq, a.clone(), a), bin(BinOp::Ne, b.clone(), b));
        let before = bin(BinOp::Or, order, nan_last);
        let take_right = bin(BinOp::And, bin(BinOp::Lt, var(&j), var(&hi)), bin(BinOp::Or, bin(BinOp::Ge, var(&i), var(&mid)), before));
        let merge = Stmt::For {
            var: r.clone(),
            var_span: span,
            bound: bin(BinOp::Sub, var(&hi), var(&lo)),
            body: vec![
                assign(&p, bin(BinOp::Add, var(&lo), var(&r))),
                Stmt::If {
                    cond: take_right,
                    then_body: vec![set(&tmp, var(&p), index(&ord, var(&j))), assign(&j, bin(BinOp::Add, var(&j), num(1.0)))],
                    else_body: vec![set(&tmp, var(&p), index(&ord, var(&i))), assign(&i, bin(BinOp::Add, var(&i), num(1.0)))],
                    span,
                },
            ],
            parallel: false,
            span,
        };
        let pass = Stmt::While {
            cond: bin(BinOp::Lt, var(&lo), n.clone()),
            body: vec![
                assign(&mid, call("min", vec![bin(BinOp::Add, var(&lo), var(&w)), n.clone()])),
                assign(&hi, call("min", vec![bin(BinOp::Add, var(&lo), bin(BinOp::Mul, num(2.0), var(&w))), n.clone()])),
                assign(&i, var(&lo)),
                assign(&j, var(&mid)),
                merge,
                assign(&lo, var(&hi)),
            ],
            span,
        };
        out.push(Stmt::While {
            cond: bin(BinOp::Lt, var(&w), n.clone()),
            body: vec![
                assign(&lo, num(0.0)),
                pass,
                assign(&sw, var(&ord)),
                assign(&ord, var(&tmp)),
                assign(&tmp, var(&sw)),
                assign(&w, bin(BinOp::Mul, num(2.0), var(&w))),
            ],
            span,
        });
        var(&ord)
    }

    /// One f-string field's expression, parsed from its tokens.
    fn fstring_field(&mut self, toks: Vec<crate::lexer::Token>) -> PResult<Expr> {
        let mut sub = Parser {
            toks,
            pos: 0,
            no_record_lit: false,
            cur_def: self.cur_def.clone(),
            step_depth: 0,
            step_ids: Vec::new(),
            in_parallel: 0,
            hidden: self.hidden,
            defs: self.defs.clone(),
            lifted: std::mem::take(&mut self.lifted),
            scope: self.scope.clone(),
            globals: self.globals.clone(),
            phrases: std::mem::take(&mut self.phrases),
            user_phrases: self.user_phrases.clone(),
            whole: std::mem::take(&mut self.whole),
            enums: self.enums.clone(),
            matches: self.matches,
            sigs: self.sigs.clone(),
        };
        let e = sub.expr()?;
        self.whole = std::mem::take(&mut sub.whole);
        if !sub.at(&Tok::Eof) {
            return Err(sub.err_here("expected the end of the f-string field"));
        }
        self.hidden = sub.hidden;
        self.lifted = std::mem::take(&mut sub.lifted);
        self.phrases = std::mem::take(&mut sub.phrases);
        Ok(e)
    }

    /// An f-string: `f"a {x} b {y:.3f}"` is `viv__fstr("a ", viv__str(x), " b ",
    /// viv__fmt(y, ".3f"))`, the concatenation of its text and of each field as
    /// Python writes it: `{e}` is str(e), `{e!r}` repr(e), `{e!s}` str(e),
    /// `{e:spec}` fmt(e, spec) (after the conversion, if any), `{e=}` the
    /// field's text and repr(e). An f-string with no field is its text.
    fn fstring(&mut self, parts: Vec<crate::lexer::FPart>, span: Span) -> PResult<Expr> {
        use crate::lexer::FPart;
        let mut args = Vec::new();
        for part in parts {
            match part {
                FPart::Text(t) => args.push(Expr::Str { value: t, span }),
                FPart::Field { toks, eq, conv, spec, spec_fields, span: fspan } => {
                    let mut nested = Vec::with_capacity(spec_fields.len());
                    for sf in spec_fields {
                        nested.push(self.fstring_field(sf)?);
                    }
                    let e = self.fstring_field(toks)?;
                    let call = |callee: &str, args: Vec<Expr>| Expr::Call { callee: callee.to_string(), args, span: fspan };
                    let has_eq = eq.is_some();
                    if let Some(text) = eq {
                        args.push(Expr::Str { value: text, span: fspan });
                    }
                    let mut e = match conv {
                        Some('r') => call(REPR_OF, vec![e]),
                        Some(_) => call(STR_OF, vec![e]),
                        None => e,
                    };
                    e = match spec {
                        Some(sp) => {
                            let mut a = vec![e, Expr::Str { value: sp, span: fspan }];
                            a.extend(nested);
                            call(FMT_OF, a)
                        }
                        None if conv.is_some() => e,
                        None if has_eq => call(REPR_OF, vec![e]),
                        None => call(STR_OF, vec![e]),
                    };
                    args.push(e);
                }
            }
        }
        if args.iter().all(|a| matches!(a, Expr::Str { .. })) {
            let value: String = args.iter().map(|a| if let Expr::Str { value, .. } = a { value.as_str() } else { "" }).collect();
            return Ok(Expr::Str { value, span });
        }
        Ok(Expr::Call { callee: FSTR.to_string(), args, span })
    }

    /// `TypeName { field: expr, ... }` — a record literal. The type name has
    /// already been consumed; `type_span` is its span. `TypeName { field:
    /// expr, ..base }` is a record update (crate::update): written as a call
    /// of `viv__with` on the partial literal and the base, which the pass
    /// completes.
    fn record_literal(&mut self, type_name: String, type_span: Span) -> PResult<Expr> {
        self.expect(Tok::LBrace, "`{`")?;
        // Field values are in value position: record literals allowed within.
        let saved = self.no_record_lit;
        self.no_record_lit = false;
        let mut fields = Vec::new();
        let mut base = None;
        loop {
            self.skip_newlines();
            if self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                break;
            }
            if self.at(&Tok::Dot) && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Dot)) {
                self.bump();
                self.bump();
                base = Some(self.expr()?);
                self.skip_newlines();
                if self.at(&Tok::Comma) {
                    self.bump();
                    self.skip_newlines();
                }
                if !self.at(&Tok::RBrace) {
                    return Err(Diagnostic::error("the record to update comes last")
                        .with_primary(self.span(), "expected `}` after `..base`"));
                }
                break;
            }
            let (fname, fspan) = self.ident("field name")?;
            self.expect(Tok::Colon, "`:`")?;
            let value = self.expr()?;
            fields.push((fname, fspan, value));
            self.skip_newlines();
            if self.at(&Tok::Comma) {
                self.bump();
            }
        }
        self.no_record_lit = saved;
        let close = self.expect(Tok::RBrace, "`}`")?;
        let span = type_span.merge(close.span);
        let lit = Expr::RecordLit { type_name, type_span, fields, span };
        Ok(match base {
            Some(b) => Expr::Call { callee: crate::update::WITH.to_string(), args: vec![lit, b], span },
            None => lit,
        })
    }

    /// `fn(params) -> expr` — an anonymous function with a single-expression body.
    fn lambda(&mut self) -> PResult<Expr> {
        let start = self.span();
        self.expect(Tok::Fn, "`fn`")?;
        self.expect(Tok::LParen, "`(`")?;
        let mut params = Vec::new();
        if !self.at(&Tok::RParen) {
            loop {
                let (name, _) = self.ident("a parameter")?;
                params.push(name);
                if self.at(&Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(Tok::RParen, "`)`")?;
        self.expect(Tok::Arrow, "`->`")?;
        let body = self.expr()?;
        let span = start.merge(body.span());
        Ok(Expr::Lambda { params, body: Box::new(body), span })
    }

    /// `[ e, e, ... ]` — a vector literal, or `[ [..], [..] ]` for a matrix.
    fn arr_lit(&mut self) -> PResult<Expr> {
        let start = self.span();
        self.expect(Tok::LBracket, "`[`")?;
        let saved = self.no_record_lit;
        self.no_record_lit = false;
        let mut elems = Vec::new();
        if !self.at(&Tok::RBracket) {
            loop {
                elems.push(self.expr()?);
                // `[e for x in xs ...]`: a comprehension
                if elems.len() == 1 && self.at(&Tok::For) {
                    let e = elems.pop().expect("the element");
                    let clauses = self.comp_clauses()?;
                    self.no_record_lit = saved;
                    let close = self.expect(Tok::RBracket, "`]` after the comprehension")?;
                    return self.comprehension(e, clauses, Collect::List, start.merge(close.span));
                }
                if self.at(&Tok::Comma) {
                    self.bump();
                    // a trailing comma, `[a,\n b,\n]`
                    if self.at(&Tok::RBracket) {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        self.no_record_lit = saved;
        let close = self.expect(Tok::RBracket, "`]`")?;
        Ok(Expr::ArrLit { elems, span: start.merge(close.span) })
    }

    /// `{k: v for x in xs if c ...}`: a map comprehension, Python's dict
    /// comprehension. Written as a hidden map stored into in the loops the
    /// clauses name, before the statement that uses it (crate::surface):
    ///
    /// ```text
    /// viv__cmpN = map()
    /// for x in xs { if c { viv__cmpN[k] = v } }
    /// ```
    ///
    /// A key seen again keeps its first place and takes the later value, as
    /// in Python. The keys and values are numbers or strings, one kind each.
    /// A brace in an expression is only this:
    /// a map literal (`{1: 2}`) is refused, since `map()` and stores make one.
    fn map_comprehension(&mut self) -> PResult<Expr> {
        let open = self.expect(Tok::LBrace, "`{`")?;
        let saved = self.no_record_lit;
        self.no_record_lit = false;
        self.skip_newlines();
        let literal = || {
            Diagnostic::error("a brace in an expression is a map comprehension, `{k: v for x in xs}`")
                .with_note("a map literal is not Viv: make the map with `m = map()`, then store each key, `m[k] = v`")
        };
        if self.at(&Tok::RBrace) {
            return Err(literal().with_primary(open.span.merge(self.span()), "an empty map is `map()`"));
        }
        let key = self.or_expr()?;
        // `{x, y}` and `{c for c in s}`: Python's sets
        if matches!(self.peek().tok, Tok::Comma | Tok::RBrace | Tok::For) {
            return Err(Diagnostic::error("Viv has no set: a brace in an expression is a map comprehension, `{k: v for x in xs}`")
                .with_primary(open.span.merge(key.span()), "a set")
                .with_note("`unique(s)` gives the distinct characters of a string, or the distinct strings of a list, sorted: Python's `sorted(set(s))`; `x in xs` tests membership"));
        }
        if !self.at(&Tok::Colon) {
            return Err(literal().with_primary(self.span(), format!("expected `:` after the key, found {}", describe(&self.peek().tok))));
        }
        self.bump();
        let value = self.or_expr()?;
        self.skip_newlines();
        if !self.at(&Tok::For) {
            return Err(literal().with_primary(open.span.merge(value.span()), "no `for` after the first entry"));
        }
        let clauses = self.comp_clauses()?;
        self.skip_newlines();
        self.no_record_lit = saved;
        let close = self.expect(Tok::RBrace, "`}` after the map comprehension")?;
        let span = open.span.merge(close.span);
        let mut map: HashMap<String, String> = HashMap::new();
        let mut done = Vec::with_capacity(clauses.len());
        for c in clauses {
            match c {
                Clause::For(mut head) => {
                    self.subst_head(&mut head, &map);
                    self.rename_head(&mut head, &mut map);
                    done.push(Clause::For(head));
                }
                Clause::If(mut cond) => {
                    self.subst(&mut cond, &map);
                    done.push(Clause::If(cond));
                }
            }
        }
        let (mut key, mut value) = (key, value);
        self.subst(&mut key, &map);
        self.subst(&mut value, &map);
        let name = self.hidden_name("cmp");
        let res = Expr::Var { name: name.clone(), span };
        let make = Stmt::Assign { name: name.clone(), name_span: span, value: Expr::Call { callee: "map".to_string(), args: Vec::new(), span }, span };
        let store = Stmt::SetIndex { base: Box::new(res.clone()), indices: vec![key], value, span };
        let mut stmts = vec![make];
        self.nest(done, store, Collect::List, &res, span, &mut stmts)?;
        Ok(self.lift(stmts, res, span))
    }
}

// ---- phrases (crate::phrases) ----------------------------------

/// A phrase hole after parsing: its name, its expression, the variable it is a
/// function of (a `fn` hole), its kind and where it is.
struct HoleVal {
    name: String,
    expr: Expr,
    var: Option<String>,
    kind: crate::phrases::Kind,
    span: Span,
    /// where the expression's text is: the hole less a `with respect to x`,
    /// or the two sides of an equation
    text: crate::phrases::HoleText,
}

/// Why a reading of a phrase failed: the phrase's own failure (a hole with no
/// variable, two variables, two `=`), reported as it is; a hole that does not
/// parse; or a hole that parses with words left over, which means the phrase
/// has words no pattern has.
#[derive(Clone, Copy, PartialEq)]
enum Fail {
    Own,
    Parse,
    Leftover,
}
type HoleErr = (Fail, Diagnostic);

/// The callee `x: T = e` declares its type with: `viv__declare("x", "T")`.
pub const DECLARE: &str = "viv__declare";

/// The calls a `match` is written with:
/// whether a value is a variant, a variant's payload field, and the check
/// that the arms cover every variant.
pub const ENUM_IS: &str = "viv__enum_is";
pub const ENUM_FIELD: &str = "viv__enum_field";
pub const MATCH_REST: &str = "viv__match_rest";

/// A type annotation as the text `parse_type` reads back (`Int`, `[Int]`,
/// `Node`): what `viv__declare` carries.
pub fn type_text(t: &TypeExpr) -> String {
    match t {
        TypeExpr::Num => "Num".into(),
        TypeExpr::Arr => "Arr".into(),
        TypeExpr::CArr => "CArr".into(),
        TypeExpr::Bool => "Bool".into(),
        TypeExpr::Str => "Str".into(),
        TypeExpr::StrList => "[Str]".into(),
        TypeExpr::Int => "Int".into(),
        TypeExpr::IntList => "[Int]".into(),
        TypeExpr::F32 => "Float32".into(),
        TypeExpr::Packed(k) => format!("[{}]", k.name()),
        TypeExpr::Named { name, .. } => name.clone(),
        TypeExpr::RecList { name, .. } => format!("[{name}]"),
        TypeExpr::Fn { arity } => format!("Fn{arity}"),
        TypeExpr::Infer => "_".into(),
        TypeExpr::Opt(t) => format!("{}?", type_text(t)),
    }
}

/// The annotation `type_text` wrote, read back.
pub fn type_of_text(s: &str, span: Span) -> TypeExpr {
    if let Some(inner) = s.strip_suffix('?') {
        return TypeExpr::Opt(Box::new(type_of_text(inner, span)));
    }
    match s {
        "Num" => TypeExpr::Num,
        "Arr" => TypeExpr::Arr,
        "CArr" => TypeExpr::CArr,
        "Bool" => TypeExpr::Bool,
        "Str" => TypeExpr::Str,
        "[Str]" => TypeExpr::StrList,
        "Int" => TypeExpr::Int,
        "[Int]" => TypeExpr::IntList,
        "Float32" => TypeExpr::F32,
        "_" => TypeExpr::Infer,
        n if n.starts_with('[') && n.ends_with(']') && crate::ast::Store::of_name(&n[1..n.len() - 1]).is_some() => {
            TypeExpr::Packed(crate::ast::Store::of_name(&n[1..n.len() - 1]).expect("checked"))
        }
        n if n.starts_with('[') && n.ends_with(']') => TypeExpr::RecList { name: n[1..n.len() - 1].to_string(), span },
        n => TypeExpr::Named { name: n.to_string(), span },
    }
}

/// A token as text, for messages.
fn token_text(t: &Tok) -> String {
    match t {
        Tok::Ident(s) => s.clone(),
        Tok::Num(v, _) => format!("{v}"),
        Tok::Imag(v) => format!("{v}i"),
        Tok::Str(s) => format!("{s:?}"),
        other => crate::phrases::word(other).unwrap_or_else(|| describe(other).trim_matches('`').to_string()),
    }
}

fn tokens_text(toks: &[Token]) -> String {
    toks.iter().map(|t| token_text(&t.tok)).collect::<Vec<_>>().join(" ")
}

impl Parser {
    /// The words of pattern `i`: the table's, then the program's own.
    fn pat_words(&self, i: usize) -> String {
        let n = crate::phrases::TABLE.len();
        if i < n { crate::phrases::TABLE[i].words.to_string() } else { self.user_phrases[i - n].words.clone() }
    }

    /// `phrase "<words>" -> <expr>` at the current token.
    fn at_phrase_decl(&self) -> bool {
        matches!(&self.peek().tok, Tok::Ident(n) if n == "phrase") && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Str(_)))
    }

    /// Read every `phrase` declaration of the program (they sit at the top
    /// level, outside braces), so a phrase can be used before it is declared.
    fn declare_phrases(&mut self) -> PResult<()> {
        let mut depth = 0i32;
        let mut at = Vec::new();
        for (k, t) in self.toks.iter().enumerate() {
            match t.tok {
                Tok::LBrace => depth += 1,
                Tok::RBrace => depth -= 1,
                Tok::Ident(ref n) if depth == 0 && n == "phrase" && matches!(self.toks.get(k + 1).map(|t| &t.tok), Some(Tok::Str(_))) => at.push(k),
                _ => {}
            }
        }
        for k in at {
            self.pos = k;
            let snap = (self.lifted.len(), self.hidden, self.phrases.len());
            let decl = self.phrase_decl()?;
            if self.lifted.len() != snap.0 {
                return Err(Diagnostic::error("a phrase's body is one expression: a comprehension, an `if` expression or a `key=` cannot be in it")
                    .with_primary(decl.body_span, "here")
                    .with_note("write a def and call it from the phrase"));
            }
            self.lifted.truncate(snap.0);
            self.hidden = snap.1;
            self.phrases.truncate(snap.2);
            self.user_phrases.push(decl);
        }
        self.pos = 0;
        Ok(())
    }

    /// `phrase "the norm of M" -> sqrt(sum(M * M))`: a word of the phrase that
    /// is a name the body reads is a hole; the other words are literal.
    fn phrase_decl(&mut self) -> PResult<UserPhrase> {
        use crate::phrases as ph;
        let start = self.span();
        self.bump(); // `phrase`
        let Tok::Str(text) = self.peek().tok.clone() else { return Err(self.err_here("expected the phrase's words, in quotes")) };
        let text_span = self.span();
        self.bump();
        // `given a, b`: names read from where the phrase is used
        let mut given: Vec<(String, Span)> = Vec::new();
        if matches!(&self.peek().tok, Tok::Ident(g) if g == "given") {
            self.bump();
            loop {
                let Tok::Ident(n) = self.peek().tok.clone() else { return Err(self.err_here("expected a name after `given`")) };
                given.push((n, self.span()));
                self.bump();
                if self.at(&Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(Tok::Arrow, "`->` and the expression the phrase is")?;
        let body = self.expr()?;
        let body_span = body.span();
        let span = start.merge(body_span);
        let reads = ph::free_names(&body);
        for (g, gs) in &given {
            if !reads.contains(g) {
                return Err(Diagnostic::error(format!("the phrase is given `{g}` and its body does not read it")).with_primary(*gs, "not read"));
            }
        }
        let mut words = Vec::new();
        let mut holes = 0;
        let mut prev_hole = false;
        for w in ph::comma_words(&text) {
            let w = w.as_str();
            let name = w.chars().all(|c| c.is_alphanumeric() || c == '_') && !w.starts_with(|c: char| c.is_ascii_digit());
            // a given name written as a word is a literal: `this head`,
            // given head, matches the word and reads the use's `head`
            if name && reads.iter().any(|r| r == w) && !given.iter().any(|(g, _)| g == w) {
                if prev_hole {
                    return Err(Diagnostic::error(format!("`{w}` follows another hole in `{text}`"))
                        .with_primary(text_span, "two expressions side by side cannot be told apart")
                        .with_note("put a word between them"));
                }
                words.push(format!("{{{w}}}"));
                holes += 1;
                prev_hole = true;
            } else {
                if w.contains(['{', '}', '|']) {
                    return Err(Diagnostic::error(format!("a phrase's word cannot hold `{{`, `}}` or `|`: `{w}`")).with_primary(text_span, "here"));
                }
                words.push(w.to_string());
                prev_hole = false;
            }
        }
        if words.len() == holes {
            return Err(Diagnostic::error(format!("`{text}` has no word that is not a hole")).with_primary(text_span, "a phrase needs words to be recognized by"));
        }
        let words = words.join(" ");
        let shown = ph::display(&words);
        let key = ph::shape_key(&words);
        if let Some(p) = ph::TABLE.iter().find(|p| ph::bare_verb(p).is_none() && ph::shape_key(p.words) == key) {
            return Err(Diagnostic::error(format!("`take {shown}` is already a phrase of Viv's table"))
                .with_primary(text_span, format!("it lowers to `{}`", p.lowers))
                .with_note("`viv phrases` lists the table"));
        }
        if let Some(other) = self.user_phrases.iter().find(|u| ph::shape_key(&u.words) == key && u.span != span) {
            return Err(Diagnostic::error(format!("`take {shown}` is declared twice")).with_primary(text_span, "again").with_secondary(other.span, "first"));
        }
        Ok(UserPhrase { words, body, span, body_span, given, body_text: None })
    }

    /// Whether the name at the current token opens a phrase: it is followed by
    /// a name (other than `in`, which is `take in xs`) or a number. Nowhere
    /// else in Viv can a name be followed directly by either.
    fn phrase_follows(&self) -> bool {
        match self.toks.get(self.pos + 1).map(|t| &t.tok) {
            Some(Tok::Ident(s)) => s != "in",
            Some(Tok::Num(..)) => true,
            _ => false,
        }
    }

    /// The index of the `}` that closes the block whose `{` was just read.
    fn block_end(&self) -> usize {
        let mut depth = 1;
        let mut i = self.pos;
        while i < self.toks.len() {
            match self.toks[i].tok {
                Tok::LBrace => depth += 1,
                Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return i;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        self.toks.len()
    }

    /// A phrase, at `take` (`take`) or at its verb: the call it lowers to.
    fn phrase(&mut self, take: bool) -> PResult<Expr> {
        use crate::phrases::{self as ph, TABLE};
        let start = self.span();
        let verb = if take {
            self.bump();
            None
        } else {
            match &self.peek().tok {
                Tok::Ident(s) => Some(s.clone()),
                _ => None,
            }
        };
        // the phrase's tokens: to the end of the line, or to a closing
        // bracket, `{`, `:`, `if` or `else` outside brackets; and the commas
        // outside brackets before that, each a place the phrase may end
        let first = self.pos;
        let mut depth = 0i32;
        let mut k = self.pos;
        let mut ends: Vec<usize> = Vec::new();
        while k < self.toks.len() {
            let t = &self.toks[k].tok;
            if matches!(t, Tok::Eof | Tok::Newline) {
                break;
            }
            if depth == 0 && matches!(t, Tok::RParen | Tok::RBracket | Tok::RBrace | Tok::LBrace | Tok::Colon | Tok::If | Tok::Else) {
                break;
            }
            if depth == 0 && matches!(t, Tok::Comma) {
                ends.push(k - first);
            }
            match t {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                _ => {}
            }
            k += 1;
        }
        ends.push(k - first);
        let all_toks: Vec<Token> = self.toks[first..k].to_vec();
        // a phrase ends at its first comma, unless a pattern with a `,` among
        // its words reads it further
        let comma_patterns = TABLE.iter().any(|p| ph::has_comma(p.words)) || self.user_phrases.iter().any(|u| ph::has_comma(&u.words));
        if !comma_patterns {
            ends.truncate(1);
        }
        // every split of the phrase among the holes of every pattern, at
        // every place it may end; the splits whose holes parse are readings
        let mut ok: Vec<(usize, Vec<(usize, usize)>, usize)> = Vec::new();
        let mut first_splits: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        let mut errs: Vec<HoleErr> = Vec::new();
        for (e_at, &e) in ends.iter().enumerate() {
            let (pt, end_depth) = ph::ptoks(&all_toks[..e]);
            let mut splits: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
            for (i, p) in TABLE.iter().enumerate() {
                if ph::bare_verb(p) != verb.as_deref() || (e_at > 0 && !ph::has_comma(p.words)) {
                    continue;
                }
                for s in ph::splits(&ph::elements(p.words), &pt, end_depth) {
                    splits.push((i, s));
                }
            }
            if verb.is_none() {
                for (k, u) in self.user_phrases.iter().enumerate() {
                    if e_at > 0 && !ph::has_comma(&u.words) {
                        continue;
                    }
                    for s in ph::splits(&ph::elements(&u.words), &pt, end_depth) {
                        splits.push((TABLE.len() + k, s));
                    }
                }
            }
            // nothing a trial parse made is kept
            for (i, s) in &splits {
                let snap = (self.lifted.len(), self.hidden, self.phrases.len());
                let r = self.phrase_holes(*i, s, &all_toks[..e]);
                self.lifted.truncate(snap.0);
                self.hidden = snap.1;
                self.phrases.truncate(snap.2);
                match r {
                    Ok(_) => ok.push((*i, s.clone(), e)),
                    Err(err) if e_at == 0 => errs.push(err),
                    Err(_) => {}
                }
            }
            if e_at == 0 {
                first_splits = splits;
            }
        }
        let opener = if take { "take ".to_string() } else { String::new() };
        if ok.len() > 1 {
            let longest = ok.iter().map(|r| r.2).max().unwrap_or(ends[0]);
            let written = format!("{opener}{}", tokens_text(&all_toks[..longest]));
            let span = start.merge(all_toks[longest - 1].span);
            let mut d = Diagnostic::error(format!("`{written}` can be read {} ways", ok.len())).with_primary(span, "more than one phrase matches");
            for (i, s, e) in &ok {
                let holes: Vec<String> = ph::elements(&self.pat_words(*i))
                    .iter()
                    .filter_map(|e| if let ph::El::Hole { name, .. } = e { Some(name.to_uppercase()) } else { None })
                    .zip(s)
                    .map(|(n, (lo, hi))| format!("{n} = `{}`", tokens_text(&all_toks[*lo..*hi])))
                    .collect();
                let upto = if *e < longest { format!(", ending before `{}`", tokens_text(&all_toks[*e..longest])) } else { String::new() };
                d = d.with_note(format!("`{}` with {}{upto}", ph::display(&self.pat_words(*i)), holes.join(", ")));
            }
            return Err(d.with_note("put parentheses around the part that is one expression"));
        }
        let Some((i, s, e)) = ok.pop() else {
            let toks = &all_toks[..ends[0]];
            let span = start.merge(toks.last().map(|t| t.span).unwrap_or(start));
            let (pt, _) = ph::ptoks(toks);
            let written = format!("{opener}{}", tokens_text(toks));
            if let Some((_, d)) = errs.iter().find(|(f, _)| *f == Fail::Own) {
                return Err(d.clone());
            }
            if first_splits.len() == 1 {
                if let Some((_, d)) = errs.into_iter().find(|(f, _)| *f == Fail::Parse) {
                    return Err(d.with_note(format!("in the phrase `{}`", ph::display(&self.pat_words(first_splits[0].0)))));
                }
            }
            return Err(self.no_phrase(&written, &pt, verb.as_deref(), span));
        };
        self.pos = first + e;
        let toks: Vec<Token> = all_toks[..e].to_vec();
        let span = start.merge(toks.last().map(|t| t.span).unwrap_or(start));
        let (pt, _) = ph::ptoks(&toks);
        let holes = self.phrase_holes(i, &s, &toks).map_err(|(_, d)| d)?;
        // the words the phrase matched literals with, for a lowering's `{a|b}`
        let mut covered = vec![false; toks.len()];
        for (lo, hi) in &s {
            for c in covered.iter_mut().take(*hi).skip(*lo) {
                *c = true;
            }
        }
        let chosen: Vec<String> = pt.iter().zip(&covered).filter(|(_, c)| !**c).filter_map(|(t, _)| t.word.clone()).collect();
        self.lower_phrase(i, holes, &chosen, span)
    }

    /// The error for a phrase no pattern reads: the closest patterns, and the
    /// methods when it names one with `using`.
    fn no_phrase(&self, written: &str, pt: &[crate::phrases::PTok], verb: Option<&str>, span: Span) -> Diagnostic {
        use crate::phrases as ph;
        let words: Vec<String> = pt.iter().filter(|t| t.depth == 0).filter_map(|t| t.word.clone()).collect();
        let mut d = Diagnostic::error(format!("no phrase reads `{written}`")).with_primary(span, "not one of Viv's phrases");
        let close = ph::closest(&words, verb, 3);
        if !close.is_empty() {
            let opener = if verb.is_some() { "" } else { "take " };
            let list: Vec<String> = close.iter().map(|p| format!("`{opener}{}`", ph::display(p.words))).collect();
            d = d.with_note(format!("the closest phrases are {}", list.join(", ")));
        }
        if let Some(u) = words.iter().position(|w| w == "using") {
            let named = words[u + 1..].iter().take_while(|w| *w != "starting" && *w != "with").cloned().collect::<Vec<_>>().join(" ");
            let methods = ph::methods_like(&words);
            if !methods.is_empty() {
                d = d.with_note(format!("`{named}` is not a method this phrase knows; it knows {}", methods.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", ")));
            }
        }
        if words.iter().any(|w| w == "newton") && !words.iter().any(|w| w == "starting") {
            d = d.with_note("Newton's method needs a point to start from: add `starting at x0`");
        }
        d.with_note("`viv phrases` lists every phrase")
    }

    /// The holes of pattern `i` split as `s`, parsed.
    fn phrase_holes(&mut self, i: usize, s: &[(usize, usize)], toks: &[Token]) -> Result<Vec<HoleVal>, HoleErr> {
        use crate::phrases::{self as ph, Kind};
        let mut out = Vec::new();
        let holes: Vec<(String, Kind)> = ph::elements(&self.pat_words(i))
            .into_iter()
            .filter_map(|e| if let ph::El::Hole { name, kind } = e { Some((name, kind)) } else { None })
            .collect();
        for ((name, kind), (lo, hi)) in holes.into_iter().zip(s) {
            let ht = &toks[*lo..*hi];
            let span = ht[0].span.merge(ht[ht.len() - 1].span);
            let (expr, var, text) = match kind {
                Kind::Var => match &ht[0].tok {
                    Tok::Ident(x) => (Expr::Var { name: x.clone(), span }, None, ph::HoleText::Plain(span)),
                    _ => return Err((Fail::Parse, Diagnostic::error("expected a name").with_primary(span, "here"))),
                },
                Kind::Expr => (self.sub_expr(ht)?, None, ph::HoleText::Plain(span)),
                Kind::Call => {
                    let e = self.sub_expr(ht)?;
                    match &e {
                        Expr::Call { callee, .. } if self.defs.contains(callee) => {}
                        _ => {
                            return Err((Fail::Parse, Diagnostic::error("expected a call of a def, like `loss(w, X, y)`").with_primary(span, "the function and its arguments")));
                        }
                    }
                    (e, None, ph::HoleText::Plain(span))
                }
                Kind::Fn => {
                    let (e, x, t) = self.fn_hole(ht, span)?;
                    (e, Some(x), t)
                }
            };
            out.push(HoleVal { name, expr, var, kind, span, text });
        }
        // the variable a gradient is taken with respect to is the first
        // argument of the call: that is the one `grad` and `value_and_grad`
        // differentiate
        if let Some(c) = out.iter().find(|h| h.kind == Kind::Call) {
            if let (Expr::Call { callee, args, .. }, Some(x)) = (&c.expr, out.iter().find(|h| h.kind == Kind::Var)) {
                let Expr::Var { name: xn, .. } = &x.expr else { return Ok(out) };
                if !matches!(args.first(), Some(Expr::Var { name, .. }) if name == xn) {
                    return Err((Fail::Own, Diagnostic::error(format!("the gradient is taken with respect to the first argument of `{callee}`, and that is not `{xn}`"))
                        .with_primary(c.span, "the call")
                        .with_secondary(x.span, "the variable named here")
                        .with_note(format!("write the call with `{xn}` first, as `{callee}` is defined"))));
                }
            }
            return Ok(out);
        }
        // a name the lowering does not read (`solve A @ x = b`: the x) names
        // what the phrase is about, and must not be read by the other holes
        let lowers = ph::TABLE.get(i).map(|p| p.lowers).unwrap_or("");
        for h in out.iter().filter(|h| h.kind == Kind::Var && !lowers.contains(&format!("{{{}}}", h.name))) {
            let Expr::Var { name: x, .. } = &h.expr else { continue };
            if let Some(o) = out.iter().find(|o| o.kind != Kind::Var && ph::free_names(&o.expr).contains(x)) {
                return Err((Fail::Own, Diagnostic::error(format!("`{x}` is what the phrase solves for, and it is read in the {}", o.name.to_uppercase()))
                    .with_primary(o.span, format!("reads `{x}`"))
                    .with_secondary(h.span, "the unknown")));
            }
        }
        Ok(out)
    }

    /// The tokens `ht` as one complete expression.
    fn sub_expr(&mut self, ht: &[Token]) -> Result<Expr, HoleErr> {
        let mut sub = ht.to_vec();
        let end = ht.last().map(|t| t.span).unwrap_or_else(|| self.span());
        sub.push(Token { tok: Tok::Eof, span: Span::new(end.source, end.end, end.end) });
        let saved_toks = std::mem::replace(&mut self.toks, sub);
        let saved_pos = std::mem::replace(&mut self.pos, 0);
        let saved_nr = std::mem::replace(&mut self.no_record_lit, false);
        let r = match self.expr() {
            Ok(e) if self.at(&Tok::Eof) => Ok(e),
            Ok(_) => Err((Fail::Leftover, self.err_here("a phrase's part is one expression, and this is more"))),
            Err(d) => Err((Fail::Parse, d)),
        };
        self.toks = saved_toks;
        self.pos = saved_pos;
        self.no_record_lit = saved_nr;
        r
    }

    /// A `fn` hole: the expression (an equation `l = r` is `l - r`) and the
    /// variable it is a function of. The variable is the name written after
    /// `with respect to` or `for` at the hole's end, or else the one name the
    /// expression reads that the program does not assign here.
    fn fn_hole(&mut self, ht: &[Token], span: Span) -> Result<(Expr, String, crate::phrases::HoleText), HoleErr> {
        let (pt, _) = crate::phrases::ptoks(ht);
        let n = ht.len();
        let w = |k: usize| pt[k].word.as_deref();
        let mut body = ht;
        let mut named: Option<(String, Span)> = None;
        if n >= 5 && pt[n - 4].depth == 0 && w(n - 4) == Some("with") && w(n - 3) == Some("respect") && w(n - 2) == Some("to") && pt[n - 1].ident {
            named = Some((token_text(&ht[n - 1].tok), ht[n - 1].span));
            body = &ht[..n - 4];
        } else if n >= 3 && pt[n - 2].depth == 0 && ht[n - 2].tok == Tok::For && pt[n - 1].ident {
            named = Some((token_text(&ht[n - 1].tok), ht[n - 1].span));
            body = &ht[..n - 2];
        }
        let (bpt, _) = crate::phrases::ptoks(body);
        let eqs: Vec<usize> = (0..body.len()).filter(|&k| bpt[k].depth == 0 && body[k].tok == Tok::Eq).collect();
        let tspan = |t: &[Token]| t[0].span.merge(t[t.len() - 1].span);
        let shown = match eqs.as_slice() {
            [k] if *k > 0 && *k + 1 < body.len() => crate::phrases::HoleText::Equation(tspan(&body[..*k]), tspan(&body[*k + 1..])),
            _ if !body.is_empty() => crate::phrases::HoleText::Plain(tspan(body)),
            _ => crate::phrases::HoleText::Plain(span),
        };
        let expr = match eqs.as_slice() {
            [] => self.sub_expr(body)?,
            [k] if *k > 0 && *k + 1 < body.len() => {
                let l = self.sub_expr(&body[..*k])?;
                let r = self.sub_expr(&body[*k + 1..])?;
                if matches!(r, Expr::Num { value, .. } if value == 0.0) {
                    l
                } else {
                    let sp = l.span().merge(r.span());
                    Expr::Binary { op: BinOp::Sub, lhs: Box::new(l), rhs: Box::new(r), span: sp }
                }
            }
            _ => {
                return Err((Fail::Own, Diagnostic::error("an equation in a phrase has one `=`").with_primary(span, "this is not one equation")));
            }
        };
        let names: Vec<String> = crate::phrases::free_names(&expr).into_iter().filter(|x| !x.starts_with(HIDDEN_PREFIX)).collect();
        let text = tokens_text(body);
        if let Some((x, xspan)) = named {
            if !names.contains(&x) {
                return Err((Fail::Own, Diagnostic::error(format!("`{x}` does not appear in `{text}`")).with_primary(xspan, "the variable named here").with_secondary(span, "the expression")));
            }
            return Ok((expr, x, shown));
        }
        let free: Vec<String> = names.iter().filter(|x| !self.scope.contains(*x) && !self.globals.contains(*x) && !self.defs.contains(*x)).cloned().collect();
        match free.as_slice() {
            [x] => Ok((expr, x.clone(), shown)),
            [] => {
                let mut d = Diagnostic::error(format!("`{text}` has no free variable")).with_primary(span, "every name in it is a variable of the program here");
                if let Some(x) = names.first() {
                    d = d.with_note(format!("say which one the phrase runs over: `{text} with respect to {x}`, or `{text} for {x}`"));
                }
                Err((Fail::Own, d))
            }
            many => {
                let list = many.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", ");
                Err((Fail::Own, Diagnostic::error(format!("`{text}` has {} free variables: {list}", many.len()))
                    .with_primary(span, "a phrase needs a function of one variable")
                    .with_note(format!("name the one the phrase runs over: `{text} with respect to NAME`, NAME one of {list}; the others must be variables of the program"))))
            }
        }
    }

    /// The expression pattern `i`'s lowering is, with `holes` in it.
    fn lower_phrase(&mut self, i: usize, holes: Vec<HoleVal>, chosen: &[String], span: Span) -> PResult<Expr> {
        use crate::phrases::{self as ph, Kind};
        let (mut e, body) = if let Some(p) = ph::TABLE.get(i) {
            let mut text = ph::replace_alt(p.lowers, chosen);
            for h in &holes {
                text = text.replace(&format!("{{{}'}}", h.name), &format!("viv__phd_{}", h.name)).replace(&format!("{{{}}}", h.name), &format!("viv__ph_{}", h.name));
                for part in ["text", "var", "names", "vals"] {
                    text = text.replace(&format!("{{{}#{part}}}", h.name), &format!("viv__phc{part}_{}", h.name));
                }
            }
            let toks: Vec<Token> = crate::lexer::lex(&text, SourceId(0))
                .map_err(|ds| ds.into_iter().next().unwrap_or_else(|| Diagnostic::error("a phrase's lowering does not lex")))?
                .into_iter()
                .filter(|t| t.tok != Tok::Newline)
                .map(|t| Token { tok: t.tok, span })
                .collect();
            let mut tp = Parser::new(toks, HashSet::new());
            let e = tp.expr()?;
            if !tp.at(&Tok::Eof) || !tp.lifted.is_empty() {
                return Err(Diagnostic::error(format!("the lowering `{}` of a phrase is not one plain expression", p.lowers)).with_primary(span, "here"));
            }
            (e, None)
        } else {
            // the program's own phrase: its body, at the phrase, holes renamed
            let u = &self.user_phrases[i - ph::TABLE.len()];
            // a name the body reads that is not a hole is the program's; a
            // variable of the def the phrase is in would take its place
            // a given name is read from here: it must be a variable here
            for (g, gs) in &u.given {
                if !self.scope.contains(g) && !self.globals.contains(g) {
                    return Err(Diagnostic::error(format!("the phrase `{}` reads `{g}` from where it is used, and there is no `{g}` here", ph::display(&u.words)))
                        .with_primary(span, "used here")
                        .with_secondary(*gs, "given here")
                        .with_note(format!("use the phrase where `{g}` is a variable, or write the call it stands for")));
                }
            }
            for name in ph::free_names(&u.body) {
                if !holes.iter().any(|h| h.name == name) && !u.given.iter().any(|(g, _)| *g == name) && self.scope.contains(&name) {
                    return Err(Diagnostic::error(format!("the phrase `{}` reads the program's `{name}`, and `{name}` is a variable here", ph::display(&u.words)))
                        .with_primary(span, format!("`{name}` here is this def's"))
                        .with_secondary(u.body_span, "the phrase's body")
                        .with_note(format!("rename this def's `{name}`, or make `{name}` a word of the phrase so it is passed in")));
                }
            }
            let mut e = u.body.clone();
            ph::set_spans(&mut e, span);
            for h in &holes {
                ph::rename(&mut e, &h.name, &format!("viv__ph_{}", h.name));
            }
            (e, Some(u.body_span))
        };
        let mut uses: HashMap<String, usize> = HashMap::new();
        ph::walk_mut(&mut e, &mut |x: &mut Expr| {
            if let Expr::Var { name, .. } = x {
                *uses.entry(name.clone()).or_insert(0) += 1;
            }
        });
        let mut pre: Vec<Stmt> = Vec::new();
        let mut subst: HashMap<String, Expr> = HashMap::new();
        let mut record = Vec::new();
        let mut spread: HashMap<String, Expr> = HashMap::new();
        for h in holes {
            record.push(crate::phrases::Hole { name: h.name.clone(), span: h.span, text: h.text, var: h.var.clone() });
            match (h.kind, h.var) {
                (Kind::Call, _) => {
                    spread.insert(format!("viv__ph_{}", h.name), h.expr);
                }
                (Kind::Fn, Some(x)) => {
                    let u = self.hidden_name("u");
                    let mut inner = h.expr.clone();
                    ph::rename(&mut inner, &x, &u);
                    let d = Expr::Call {
                        callee: "deriv".to_string(),
                        args: vec![Expr::Lambda { params: vec![u], body: Box::new(inner), span: h.span }, Expr::Var { name: x.clone(), span: h.span }],
                        span: h.span,
                    };
                    subst.insert(format!("viv__phd_{}", h.name), Expr::Lambda { params: vec![x.clone()], body: Box::new(d), span: h.span });
                    if uses.keys().any(|k| k.starts_with("viv__phc")) {
                        self.cas_hole(&h.expr, h.span, &h.name, &x, span, &mut subst)?;
                    }
                    subst.insert(format!("viv__ph_{}", h.name), Expr::Lambda { params: vec![x], body: Box::new(h.expr), span: h.span });
                }
                _ => {
                    let key = format!("viv__ph_{}", h.name);
                    let value = if uses.get(&key).copied().unwrap_or(0) > 1 && !ph::trivial(&h.expr) {
                        let name = self.hidden_name("ph");
                        pre.push(Stmt::Assign { name: name.clone(), name_span: h.span, value: h.expr, span: h.span });
                        Expr::Var { name, span: h.span }
                    } else {
                        h.expr
                    };
                    subst.insert(key, value);
                }
            }
        }
        // a call hole is spread into the call it is an argument of:
        // `grad({f})` with f = `loss(w, X)` is `grad(loss, w, X)`
        if !spread.is_empty() {
            ph::walk_mut(&mut e, &mut |x: &mut Expr| {
                if let Expr::Call { args, .. } = x {
                    let mut out = Vec::with_capacity(args.len());
                    for a in args.drain(..) {
                        match &a {
                            Expr::Var { name, .. } if spread.contains_key(name) => {
                                if let Some(Expr::Call { callee, args: inner, span }) = spread.get(name) {
                                    out.push(Expr::Var { name: callee.clone(), span: *span });
                                    out.extend(inner.iter().cloned());
                                }
                            }
                            _ => out.push(a),
                        }
                    }
                    *args = out;
                }
            });
        }
        ph::walk_mut(&mut e, &mut |x: &mut Expr| {
            if let Expr::Var { name, .. } = x {
                if let Some(r) = subst.get(name.as_str()) {
                    *x = r.clone();
                }
            }
        });
        // the std defs the lowering calls
        let std = ph::std_def_names();
        let mut needs: Vec<String> = Vec::new();
        ph::walk_mut(&mut e, &mut |x: &mut Expr| {
            let n = match x {
                Expr::Call { callee, .. } => Some(callee.clone()),
                _ => None,
            };
            if let Some(n) = n.filter(|n| std.contains(n) && !needs.contains(n)) {
                needs.push(n);
            }
        });
        let pattern = (i < ph::TABLE.len()).then_some(i);
        let body_text = if i >= ph::TABLE.len() { self.user_phrases[i - ph::TABLE.len()].body_text.clone() } else { None };
        self.phrases.push(ph::PhraseUse { span, pattern, words: self.pat_words(i), body, body_text, holes: record, needs });
        if pre.is_empty() {
            Ok(e)
        } else {
            Ok(self.lift(pre, e, span))
        }
    }
}

impl Parser {
    /// A `fn` hole handed to the computer algebra system (`{f#text}` and the
    /// rest, crate::phrases): its text in the CAS's syntax, its variable, the
    /// other names it reads and their values. The call is lib/cas's
    /// `cas_deriv_at`, so the program uses lib/cas; whether it does is known
    /// only once its `#! use` lines are read, which tools that read a file on
    /// its own (lint, hash) do not do, so it is not checked here.
    fn cas_hole(&self, expr: &Expr, hspan: Span, hname: &str, x: &str, _span: Span, subst: &mut HashMap<String, Expr>) -> PResult<()> {
        let text = crate::cas_check::hole_text(expr).map_err(|why| Diagnostic::error(format!("the CAS cannot read this function: {why}")).with_primary(hspan, "the function"))?;
        let names: Vec<String> = crate::phrases::free_names(expr).into_iter().filter(|n| n != x && !n.starts_with(HIDDEN_PREFIX) && n != "pi").collect();
        let vals = if names.is_empty() {
            Expr::Call { callee: "zeros".into(), args: vec![Expr::Num { value: 0.0, span: hspan }], span: hspan }
        } else {
            Expr::ArrLit { elems: names.iter().map(|n| Expr::Var { name: n.clone(), span: hspan }).collect(), span: hspan }
        };
        subst.insert(format!("viv__phctext_{}", hname), Expr::Str { value: text, span: hspan });
        subst.insert(format!("viv__phcvar_{}", hname), Expr::Str { value: x.to_string(), span: hspan });
        subst.insert(format!("viv__phcnames_{}", hname), Expr::Str { value: names.join(","), span: hspan });
        subst.insert(format!("viv__phcvals_{}", hname), vals);
        Ok(())
    }
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Def => "`def`".into(),
        Tok::Return => "`return`".into(),
        Tok::If => "`if`".into(),
        Tok::Else => "`else`".into(),
        Tok::While => "`while`".into(),
        Tok::For => "`for`".into(),
        Tok::To => "`to`".into(),
        Tok::Break => "`break`".into(),
        Tok::Continue => "`continue`".into(),
        Tok::Narrate => "`narrate`".into(),
        Tok::Show => "`show`".into(),
        Tok::Test => "`test`".into(),
        Tok::Rung => "`rung`".into(),
        Tok::Parallel => "`parallel`".into(),
        Tok::Fn => "`fn`".into(),
        Tok::Record => "`record`".into(),
        Tok::Arrow => "`->`".into(),
        Tok::Colon => "`:`".into(),
        Tok::Dot => "`.`".into(),
        Tok::LParen => "`(`".into(),
        Tok::RParen => "`)`".into(),
        Tok::LBrace => "`{`".into(),
        Tok::RBrace => "`}`".into(),
        Tok::LBracket => "`[`".into(),
        Tok::RBracket => "`]`".into(),
        Tok::Comma => "`,`".into(),
        Tok::Eq => "`=`".into(),
        Tok::Plus => "`+`".into(),
        Tok::Minus => "`-`".into(),
        Tok::Star => "`*`".into(),
        Tok::Slash => "`/`".into(),
        Tok::At => "`@`".into(),
        Tok::PlusEq => "`+=`".into(),
        Tok::MinusEq => "`-=`".into(),
        Tok::StarEq => "`*=`".into(),
        Tok::SlashEq => "`/=`".into(),
        Tok::SlashSlashEq => "`//=`".into(),
        Tok::Percent => "`%`".into(),
        Tok::SlashSlash => "`//`".into(),
        Tok::Caret => "`^`".into(),
        Tok::FatArrow => "`=>`".into(),
        Tok::Question => "`?`".into(),
        Tok::Pipe => "`|`".into(),
        Tok::Str(_) => "a string".into(),
        Tok::FStr(_) => "an f-string".into(),
        Tok::Lt => "`<`".into(),
        Tok::Le => "`<=`".into(),
        Tok::Gt => "`>`".into(),
        Tok::Ge => "`>=`".into(),
        Tok::EqEq => "`==`".into(),
        Tok::Ne => "`!=`".into(),
        Tok::And => "`and`".into(),
        Tok::Or => "`or`".into(),
        Tok::Not => "`not`".into(),
        Tok::True => "`true`".into(),
        Tok::False => "`false`".into(),
        Tok::Ident(s) => format!("identifier `{}`", s),
        Tok::Num(v, _) => format!("number `{}`", v),
        Tok::Imag(v) => format!("imaginary literal `{}i`", v),
        Tok::Newline => "end of line".into(),
        Tok::Eof => "end of input".into(),
    }
}


/// What a default may read (crate::kwargs): the parameters before it and the
/// program's top-level names, and no call of a builtin with an effect.
fn check_default(param: &str, d: &Expr, later: &[String]) -> Result<(), String> {
    let mut bad: Option<String> = None;
    let mut walk = |e: &Expr| {
        if bad.is_some() {
            return;
        }
        if let Expr::Call { callee, .. } = e {
            if crate::surface::EFFECTS.contains(&callee.as_str()) || matches!(callee.as_str(), "randn" | "rand_matrix" | "randn_matrix") {
                bad = Some(format!("the default of `{param}` calls `{callee}`, which has an effect: a default is computed at each call that leaves it out"));
            }
        }
        if let Expr::Var { name, .. } = e {
            if later.contains(name) {
                bad = Some(format!("the default of `{param}` reads `{name}`, which is not before it: a default reads the parameters before it"));
            }
        }
    };
    let mut e = d.clone();
    crate::phrases::walk_mut(&mut e, &mut |x| walk(x));
    match bad {
        Some(m) => Err(m),
        None => Ok(()),
    }
}

/// A default with the parameters before it replaced by their arguments.
fn substitute(mut d: Expr, bound: &[(String, Expr)]) -> Expr {
    crate::phrases::walk_mut(&mut d, &mut |e| {
        if let Expr::Var { name, .. } = e {
            if let Some((_, v)) = bound.iter().rev().find(|(n, _)| n == name) {
                *e = v.clone();
            }
        }
    });
    d
}
