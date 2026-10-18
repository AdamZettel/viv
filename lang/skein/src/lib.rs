// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Skein: a small teaching Scheme that compiles to C through knotkit.
//! A fourth language on the kit, and the first *dynamic*
//! one: its values are tagged `scm` cells (int/flo/bool/nil/sym/str/pair/closure/
//! prim), heap-allocated by a bump allocator (leaks, like `knot_arr`), with
//! lambda-lifted closures. Pipeline: source -> reader (S-expressions) -> IR -> C.
//!
//! v1 scope: `quote if lambda define let let* begin cond`, closures (over-capturing
//! the enclosing locals), fixnums + flonums, list/arithmetic/predicate primitives,
//! `display`/`newline`. Proper tail calls (a trampoline) and GC are the next steps.

use std::collections::HashSet;

use knotkit::backend::{CEmit, CEmitCx, CExpr, CItem, CStmt, CType, Runtime, RuntimeFragment};
use knotkit::diag::Diagnostic;
use knotkit::ir::{Hasher, IrProgram, Node, NodeId, NodeKind};
use knotkit::span::{SourceId, Span};

pub fn build_ir(src: &str, source: SourceId) -> Result<IrProgram<Skein>, Vec<Diagnostic>> {
    let data = read_all(src, source).map_err(|d| vec![d])?;
    lower_program(&data, source)
}

/// A tree-walking interpreter — the browser execution path (run Scheme without a
/// C toolchain). The compiler (this file) remains the native path; they share the reader.
pub mod interp;

fn join_errs(ds: &[Diagnostic]) -> String {
    ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n")
}

/// Compile Scheme source to C text (for the "show the generated C" view).
pub fn compile_c(src: &str) -> Result<String, String> {
    use knotkit::backend::{Backend, CBackend};
    use knotkit::io::MemoryIo;
    let prog = build_ir(src, SourceId(0)).map_err(|ds| join_errs(&ds))?;
    let emission = CBackend.emit(&prog, &MemoryIo::new()).map_err(|ds| join_errs(&ds))?;
    Ok(emission.artifact.as_c().expect("C backend produces C").to_string())
}

/// Minimal wasm ABI for the browser page (no wasm-bindgen): the page writes
/// source bytes into a buffer from `skein_alloc`, calls `skein_run`/`skein_compile`,
/// and reads back a (ptr, len) packed into the i64 return value.
#[cfg(target_arch = "wasm32")]
mod wasm_api {
    #[no_mangle]
    pub extern "C" fn skein_alloc(len: usize) -> *mut u8 {
        let mut v = vec![0u8; len];
        let p = v.as_mut_ptr();
        std::mem::forget(v);
        p
    }
    fn input(ptr: *mut u8, len: usize) -> String {
        let v = unsafe { Vec::from_raw_parts(ptr, len, len) };
        String::from_utf8_lossy(&v).into_owned()
    }
    fn pack(s: String) -> u64 {
        let bytes = s.into_bytes();
        let len = bytes.len() as u64;
        let ptr = bytes.as_ptr() as u64;
        std::mem::forget(bytes);
        (ptr << 32) | len
    }
    #[no_mangle]
    pub extern "C" fn skein_run(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::interp::run(&input(ptr, len)).unwrap_or_else(|e| format!("error: {e}")))
    }
    #[no_mangle]
    pub extern "C" fn skein_compile(ptr: *mut u8, len: usize) -> u64 {
        pack(crate::compile_c(&input(ptr, len)).unwrap_or_else(|e| format!("error: {e}")))
    }
}

// ───────────────────────────────── reader ──────────────────────────────────

#[derive(Clone, Debug)]
pub enum Datum {
    Int(i64, Span),
    Flo(f64, Span),
    Bool(bool, Span),
    Sym(String, Span),
    Str(String, Span),
    List(Vec<Datum>, Span),
}

impl Datum {
    fn span(&self) -> Span {
        match self {
            Datum::Int(_, s) | Datum::Flo(_, s) | Datum::Bool(_, s) | Datum::Sym(_, s) | Datum::Str(_, s) | Datum::List(_, s) => *s,
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
    src: &'a str,
    source: SourceId,
}

fn read_all(src: &str, source: SourceId) -> Result<Vec<Datum>, Diagnostic> {
    let mut r = Reader { b: src.as_bytes(), i: 0, src, source };
    let mut out = Vec::new();
    loop {
        r.skip_ws();
        if r.i >= r.b.len() {
            break;
        }
        out.push(r.datum()?);
    }
    Ok(out)
}

impl<'a> Reader<'a> {
    fn span(&self, a: usize, b: usize) -> Span {
        Span::new(self.source, a as u32, b as u32)
    }
    fn err(&self, msg: impl Into<String>, a: usize, b: usize) -> Diagnostic {
        Diagnostic::error(msg).with_primary(self.span(a, b), "here")
    }
    fn skip_ws(&mut self) {
        while self.i < self.b.len() {
            let c = self.b[self.i] as char;
            if c == ';' {
                while self.i < self.b.len() && self.b[self.i] != b'\n' {
                    self.i += 1;
                }
            } else if c.is_whitespace() {
                self.i += 1;
            } else {
                break;
            }
        }
    }
    fn datum(&mut self) -> Result<Datum, Diagnostic> {
        self.skip_ws();
        let start = self.i;
        if self.i >= self.b.len() {
            return Err(self.err("unexpected end of input", start, start));
        }
        let c = self.b[self.i] as char;
        match c {
            '(' | '[' => {
                let close = if c == '(' { b')' } else { b']' };
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    if self.i >= self.b.len() {
                        return Err(self.err("unterminated list", start, self.i));
                    }
                    if self.b[self.i] == close {
                        self.i += 1;
                        break;
                    }
                    items.push(self.datum()?);
                }
                Ok(Datum::List(items, self.span(start, self.i)))
            }
            ')' | ']' => Err(self.err("unexpected `)`", start, start + 1)),
            '\'' => {
                // 'x  =>  (quote x)
                self.i += 1;
                let q = self.datum()?;
                let span = self.span(start, q.span().end as usize);
                Ok(Datum::List(vec![Datum::Sym("quote".into(), self.span(start, start + 1)), q], span))
            }
            '"' => {
                self.i += 1;
                let mut s = String::new();
                while self.i < self.b.len() && self.b[self.i] != b'"' {
                    if self.b[self.i] == b'\\' && self.i + 1 < self.b.len() {
                        self.i += 1;
                        s.push(match self.b[self.i] {
                            b'n' => '\n',
                            b't' => '\t',
                            o => o as char,
                        });
                    } else {
                        s.push(self.b[self.i] as char);
                    }
                    self.i += 1;
                }
                if self.i >= self.b.len() {
                    return Err(self.err("unterminated string", start, self.i));
                }
                self.i += 1;
                Ok(Datum::Str(s, self.span(start, self.i)))
            }
            _ => {
                // An atom: a maximal run of non-delimiter characters.
                while self.i < self.b.len() {
                    let d = self.b[self.i] as char;
                    if d.is_whitespace() || matches!(d, '(' | ')' | '[' | ']' | '"' | ';' | '\'') {
                        break;
                    }
                    self.i += 1;
                }
                let text = &self.src[start..self.i];
                let span = self.span(start, self.i);
                Ok(classify_atom(text, span))
            }
        }
    }
}

fn classify_atom(text: &str, span: Span) -> Datum {
    match text {
        "#t" => return Datum::Bool(true, span),
        "#f" => return Datum::Bool(false, span),
        _ => {}
    }
    if let Ok(i) = text.parse::<i64>() {
        return Datum::Int(i, span);
    }
    if let Ok(f) = text.parse::<f64>() {
        // Reject lone "." / "+" etc. parsing as float-ish; `parse::<f64>` is strict enough.
        return Datum::Flo(f, span);
    }
    Datum::Sym(text.to_string(), span)
}

// ─────────────────────────────── IR vocabulary ─────────────────────────────

/// Skein's IR. Every value is a tagged `scm` cell in C.
#[derive(Clone, Debug)]
pub enum Skein {
    /// Program root -> `main`. children: top-level statements (GlobalSet/ExprStmt).
    Program,
    /// A file-scope global cell `g_<name>`. leaf (declared, assigned in main).
    Global { name: String },
    /// `g_<name> = value` in main. children: [value].
    GlobalSet { name: String },
    /// A top-level expression evaluated for effect. children: [expr].
    ExprStmt,
    /// A lambda-lifted procedure `scm <fn>(scm* env, scm* args, int n)`.
    /// children: [body expr].
    LambdaDef { fn_name: String, nparams: usize, ncap: usize },
    // ── expressions ──
    Int(i64),
    Flo(f64),
    Bool(bool),
    Nil,
    Sym(String),
    Str(String),
    /// A procedure parameter, `args[i]`.
    Param(usize),
    /// A captured enclosing local, `env[j]`.
    Capture(usize),
    /// A reference to a global, `g_<name>`.
    GlobalRef { name: String },
    /// A primitive value (`+`, `cons`, ...).
    PrimRef { name: String },
    /// Build a closure: `scm_make_closure(<fn>, ncap, caps...)`. children: cap exprs.
    MakeClosure { fn_name: String, ncap: usize },
    /// Procedure application in *value* position. children: [callee, args...].
    /// Emits the trampolining `scm_apply` (loops on tail calls), yielding an `scm`.
    Apply,
    /// Procedure application in *tail* position. children: [callee, args...].
    /// Emits `scm_tail` (returns a pending-call `bounce` to the trampoline) — this
    /// is what gives proper tail calls (constant stack).
    TailApply,
    /// A value in tail position: wrap it in a `bounce` that says "done". child: [value].
    TailValue,
    /// `if`. children: [cond, then, els].
    If,
    /// `begin`. children: [exprs...].
    Seq,
    /// `cons` (used to build quoted lists). children: [car, cdr].
    Cons,
}

impl NodeKind for Skein {
    fn tag(&self) -> u32 {
        match self {
            Skein::Program => 1,
            Skein::Global { .. } => 2,
            Skein::GlobalSet { .. } => 3,
            Skein::ExprStmt => 4,
            Skein::LambdaDef { .. } => 5,
            Skein::Int(_) => 6,
            Skein::Flo(_) => 7,
            Skein::Bool(_) => 8,
            Skein::Nil => 9,
            Skein::Sym(_) => 10,
            Skein::Str(_) => 11,
            Skein::Param(_) => 12,
            Skein::Capture(_) => 13,
            Skein::GlobalRef { .. } => 14,
            Skein::PrimRef { .. } => 15,
            Skein::MakeClosure { .. } => 16,
            Skein::Apply => 17,
            Skein::If => 18,
            Skein::Seq => 19,
            Skein::Cons => 20,
            Skein::TailApply => 21,
            Skein::TailValue => 22,
        }
    }
    fn hash_payload(&self, h: &mut Hasher) {
        let s = |h: &mut Hasher, b: &[u8]| {
            h.update(&(b.len() as u32).to_le_bytes());
            h.update(b);
        };
        match self {
            Skein::Global { name } | Skein::GlobalSet { name } | Skein::GlobalRef { name } | Skein::PrimRef { name } | Skein::Sym(name) | Skein::Str(name) => s(h, name.as_bytes()),
            Skein::LambdaDef { fn_name, nparams, ncap } => {
                s(h, fn_name.as_bytes());
                h.update(&(*nparams as u32).to_le_bytes());
                h.update(&(*ncap as u32).to_le_bytes());
            }
            Skein::MakeClosure { fn_name, ncap } => {
                s(h, fn_name.as_bytes());
                h.update(&(*ncap as u32).to_le_bytes());
            }
            Skein::Int(i) => {
                h.update(&i.to_le_bytes());
            }
            Skein::Flo(f) => {
                h.update(&f.to_bits().to_le_bytes());
            }
            Skein::Bool(b) => {
                h.update(&[*b as u8]);
            }
            Skein::Param(i) | Skein::Capture(i) => {
                h.update(&(*i as u32).to_le_bytes());
            }
            Skein::Program | Skein::ExprStmt | Skein::Nil | Skein::Apply | Skein::TailApply | Skein::TailValue | Skein::If | Skein::Seq | Skein::Cons => {}
        }
    }
}

// ──────────────────────────────── lowering ─────────────────────────────────

/// The primitives, mapping the Scheme name to its C function.
fn prim_cfn(name: &str) -> Option<&'static str> {
    Some(match name {
        "+" => "scm_p_add",
        "-" => "scm_p_sub",
        "*" => "scm_p_mul",
        "=" => "scm_p_numeq",
        "<" => "scm_p_lt",
        ">" => "scm_p_gt",
        "eq?" => "scm_p_eq",
        "equal?" => "scm_p_eq",
        "null?" => "scm_p_null",
        "pair?" => "scm_p_pair",
        "cons" => "scm_p_cons",
        "car" => "scm_p_car",
        "cdr" => "scm_p_cdr",
        "not" => "scm_p_not",
        "list" => "scm_p_list",
        "display" => "scm_p_display",
        "newline" => "scm_p_newline",
        _ => return None,
    })
}

/// A lexical scope: parameter names and captured-local names (by position).
struct Scope {
    params: Vec<String>,
    captures: Vec<String>,
}

impl Scope {
    fn resolve(&self, prog: &IrProgram<Skein>, name: &str, span: Span, globals: &HashSet<String>) -> Option<Skein> {
        let _ = prog;
        let _ = span;
        if let Some(i) = self.params.iter().position(|p| p == name) {
            Some(Skein::Param(i))
        } else if let Some(j) = self.captures.iter().position(|c| c == name) {
            Some(Skein::Capture(j))
        } else if globals.contains(name) {
            Some(Skein::GlobalRef { name: name.to_string() })
        } else if prim_cfn(name).is_some() {
            Some(Skein::PrimRef { name: name.to_string() })
        } else {
            None
        }
    }
}

struct Lowerer<'a> {
    prog: IrProgram<Skein>,
    globals: &'a HashSet<String>,
    next_lam: u32,
}

fn lower_program(data: &[Datum], source: SourceId) -> Result<IrProgram<Skein>, Vec<Diagnostic>> {
    // Pre-scan top-level `define`s so forward references resolve as globals.
    let mut globals: HashSet<String> = HashSet::new();
    for d in data {
        if let Some((name, _)) = as_define(d) {
            globals.insert(name);
        }
    }
    let mut lw = Lowerer { prog: IrProgram::new(), globals: &globals, next_lam: 0 };
    let mut errs = Vec::new();

    // File-scope global cells.
    let mut gnames: Vec<&String> = globals.iter().collect();
    gnames.sort();
    for name in gnames {
        let g = lw.prog.add(Skein::Global { name: name.clone() }, Span::synthetic(source), vec![]);
        lw.prog.add_root(g);
    }

    let top = Scope { params: Vec::new(), captures: Vec::new() };
    let mut stmts = Vec::new();
    for d in data {
        match lw.lower_top(d, &top) {
            Ok(id) => stmts.push(id),
            Err(e) => errs.push(e),
        }
    }
    if !errs.is_empty() {
        return Err(errs);
    }
    let main = lw.prog.add(Skein::Program, Span::synthetic(source), stmts);
    lw.prog.add_root(main);
    Ok(lw.prog)
}

/// If `d` is a top-level define, return (name, is_function).
fn as_define(d: &Datum) -> Option<(String, bool)> {
    let Datum::List(items, _) = d else { return None };
    let [head, target, ..] = &items[..] else { return None };
    let Datum::Sym(h, _) = head else { return None };
    if h != "define" {
        return None;
    }
    match target {
        Datum::Sym(name, _) => Some((name.clone(), false)),
        Datum::List(sig, _) => match sig.first() {
            Some(Datum::Sym(name, _)) => Some((name.clone(), true)),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> Lowerer<'a> {
    fn lower_top(&mut self, d: &Datum, scope: &Scope) -> Result<NodeId, Diagnostic> {
        if let Some((name, is_fn)) = as_define(d) {
            let Datum::List(items, span) = d else { unreachable!() };
            let value = if is_fn {
                // (define (f a b) body...) => f = (lambda (a b) body...)
                let Datum::List(sig, _) = &items[1] else { unreachable!() };
                let params: Vec<String> = sig[1..].iter().map(sym_name).collect::<Result<_, _>>()?;
                self.lower_lambda(&params, &items[2..], scope, *span)?
            } else {
                if items.len() != 3 {
                    return Err(err(*span, "define expects `(define name value)`"));
                }
                self.lower_expr(&items[2], scope, false)?
            };
            return Ok(self.prog.add(Skein::GlobalSet { name }, *span, vec![value]));
        }
        // A top-level expression runs for value/effect — never a tail position.
        let e = self.lower_expr(d, scope, false)?;
        Ok(self.prog.add(Skein::ExprStmt, d.span(), vec![e]))
    }

    /// Wrap a value-producing node in a `bounce` if we are in tail position.
    fn finish(&mut self, value: NodeId, tail: bool, span: Span) -> NodeId {
        if tail {
            self.prog.add(Skein::TailValue, span, vec![value])
        } else {
            value
        }
    }

    /// Lower an expression. `tail` = is this in tail position (a procedure's result)?
    /// In tail position, applications become `TailApply` (bounce to the trampoline)
    /// and plain values become `TailValue`; otherwise everything is an `scm` value.
    fn lower_expr(&mut self, d: &Datum, scope: &Scope, tail: bool) -> Result<NodeId, Diagnostic> {
        let v = match d {
            Datum::Int(i, sp) => self.prog.add(Skein::Int(*i), *sp, vec![]),
            Datum::Flo(f, sp) => self.prog.add(Skein::Flo(*f), *sp, vec![]),
            Datum::Bool(b, sp) => self.prog.add(Skein::Bool(*b), *sp, vec![]),
            Datum::Str(s, sp) => self.prog.add(Skein::Str(s.clone()), *sp, vec![]),
            Datum::Sym(name, sp) => match scope.resolve(&self.prog, name, *sp, self.globals) {
                Some(node) => self.prog.add(node, *sp, vec![]),
                None => return Err(err(*sp, format!("unbound variable `{name}`"))),
            },
            // Control forms handle tail-ness themselves and return directly.
            Datum::List(items, sp) => return self.lower_list(items, *sp, scope, tail),
        };
        Ok(self.finish(v, tail, d.span()))
    }

    fn lower_list(&mut self, items: &[Datum], span: Span, scope: &Scope, tail: bool) -> Result<NodeId, Diagnostic> {
        if items.is_empty() {
            return Err(err(span, "empty application `()`"));
        }
        if let Datum::Sym(head, _) = &items[0] {
            match head.as_str() {
                "quote" => {
                    let q = self.lower_quote(&items[1], span)?;
                    return Ok(self.finish(q, tail, span));
                }
                "if" => return self.lower_if(items, span, scope, tail),
                "lambda" => {
                    let Datum::List(sig, _) = &items[1] else {
                        return Err(err(span, "lambda parameters must be a list"));
                    };
                    let params: Vec<String> = sig.iter().map(sym_name).collect::<Result<_, _>>()?;
                    let mk = self.lower_lambda(&params, &items[2..], scope, span)?;
                    return Ok(self.finish(mk, tail, span));
                }
                "begin" => return self.lower_seq(&items[1..], span, scope, tail),
                "let" => return self.lower_let(&items[1], &items[2..], span, scope, false, tail),
                "let*" => return self.lower_let(&items[1], &items[2..], span, scope, true, tail),
                "cond" => return self.lower_cond(&items[1..], span, scope, tail),
                "define" => return Err(err(span, "internal `define` is not supported (v1: top-level only)")),
                _ => {}
            }
        }
        // Application: callee and args are always evaluated to values.
        let callee = self.lower_expr(&items[0], scope, false)?;
        let mut children = vec![callee];
        for a in &items[1..] {
            children.push(self.lower_expr(a, scope, false)?);
        }
        let kind = if tail { Skein::TailApply } else { Skein::Apply };
        Ok(self.prog.add(kind, span, children))
    }

    fn lower_if(&mut self, items: &[Datum], span: Span, scope: &Scope, tail: bool) -> Result<NodeId, Diagnostic> {
        if items.len() != 3 && items.len() != 4 {
            return Err(err(span, "if expects `(if test then [else])`"));
        }
        let cond = self.lower_expr(&items[1], scope, false)?;
        // The branches inherit our tail-ness: they are the `if`'s result.
        let then = self.lower_expr(&items[2], scope, tail)?;
        let els = if items.len() == 4 {
            self.lower_expr(&items[3], scope, tail)?
        } else {
            let f = self.prog.add(Skein::Bool(false), span, vec![]);
            self.finish(f, tail, span)
        };
        Ok(self.prog.add(Skein::If, span, vec![cond, then, els]))
    }

    fn lower_seq(&mut self, body: &[Datum], span: Span, scope: &Scope, tail: bool) -> Result<NodeId, Diagnostic> {
        if body.is_empty() {
            return Err(err(span, "empty body / `begin`"));
        }
        let mut children = Vec::new();
        let last = body.len() - 1;
        for (i, d) in body.iter().enumerate() {
            // Only the last expression is in tail position.
            children.push(self.lower_expr(d, scope, tail && i == last)?);
        }
        Ok(self.prog.add(Skein::Seq, span, children))
    }

    /// `let`/`let*` desugar to immediate lambda application (a tail call if we are
    /// in tail position).
    fn lower_let(&mut self, binds: &Datum, body: &[Datum], span: Span, scope: &Scope, star: bool, tail: bool) -> Result<NodeId, Diagnostic> {
        let Datum::List(bs, _) = binds else {
            return Err(err(span, "let bindings must be a list"));
        };
        if star && bs.len() > 1 {
            // (let* ((a e1) (b e2) ...) body) => (let ((a e1)) (let* ((b e2)...) body))
            let inner = Datum::List(
                {
                    let mut v = vec![Datum::Sym("let*".into(), span), Datum::List(bs[1..].to_vec(), span)];
                    v.extend(body.iter().cloned());
                    v
                },
                span,
            );
            return self.lower_let(&Datum::List(bs[..1].to_vec(), span), &[inner], span, scope, false, tail);
        }
        let mut names = Vec::new();
        let mut inits = Vec::new();
        for b in bs {
            let Datum::List(pair, _) = b else {
                return Err(err(span, "each let binding must be `(name value)`"));
            };
            if pair.len() != 2 {
                return Err(err(span, "each let binding must be `(name value)`"));
            }
            names.push(sym_name(&pair[0])?);
            inits.push(pair[1].clone());
        }
        // closure = (lambda (names...) body...); apply it to the inits.
        let closure = self.lower_lambda(&names, body, scope, span)?;
        let mut children = vec![closure];
        for init in &inits {
            children.push(self.lower_expr(init, scope, false)?);
        }
        let kind = if tail { Skein::TailApply } else { Skein::Apply };
        Ok(self.prog.add(kind, span, children))
    }

    /// `cond` desugars to nested `if` (its clause bodies inherit tail-ness).
    fn lower_cond(&mut self, clauses: &[Datum], span: Span, scope: &Scope, tail: bool) -> Result<NodeId, Diagnostic> {
        let f = self.prog.add(Skein::Bool(false), span, vec![]);
        let mut result = self.finish(f, tail, span);
        for clause in clauses.iter().rev() {
            let Datum::List(parts, csp) = clause else {
                return Err(err(span, "each cond clause must be a list"));
            };
            if parts.is_empty() {
                return Err(err(*csp, "empty cond clause"));
            }
            let is_else = matches!(&parts[0], Datum::Sym(s, _) if s == "else");
            let body = self.lower_seq(&parts[1..], *csp, scope, tail)?;
            if is_else {
                result = body;
            } else {
                let test = self.lower_expr(&parts[0], scope, false)?;
                result = self.prog.add(Skein::If, *csp, vec![test, body, result]);
            }
        }
        Ok(result)
    }

    fn lower_quote(&mut self, d: &Datum, _span: Span) -> Result<NodeId, Diagnostic> {
        match d {
            Datum::Int(i, sp) => Ok(self.prog.add(Skein::Int(*i), *sp, vec![])),
            Datum::Flo(f, sp) => Ok(self.prog.add(Skein::Flo(*f), *sp, vec![])),
            Datum::Bool(b, sp) => Ok(self.prog.add(Skein::Bool(*b), *sp, vec![])),
            Datum::Str(s, sp) => Ok(self.prog.add(Skein::Str(s.clone()), *sp, vec![])),
            Datum::Sym(s, sp) => Ok(self.prog.add(Skein::Sym(s.clone()), *sp, vec![])),
            Datum::List(items, sp) => {
                // Build a cons chain, ending in nil.
                let mut acc = self.prog.add(Skein::Nil, *sp, vec![]);
                for item in items.iter().rev() {
                    let car = self.lower_quote(item, *sp)?;
                    acc = self.prog.add(Skein::Cons, *sp, vec![car, acc]);
                }
                Ok(acc)
            }
        }
    }

    /// Lower a lambda: lift it to a top-level C function and emit a closure that
    /// captures *all* enclosing locals (over-capture — simple and correct).
    fn lower_lambda(&mut self, params: &[String], body: &[Datum], enclosing: &Scope, span: Span) -> Result<NodeId, Diagnostic> {
        if body.is_empty() {
            return Err(err(span, "lambda body is empty"));
        }
        let captures: Vec<String> = enclosing.params.iter().chain(enclosing.captures.iter()).cloned().collect();
        let inner = Scope { params: params.to_vec(), captures: captures.clone() };
        // A procedure body is in tail position: its result drives the trampoline.
        let body_id = self.lower_seq(body, span, &inner, true)?;

        let fn_name = format!("scm_lam_{}", self.next_lam);
        self.next_lam += 1;
        let def = self.prog.add(Skein::LambdaDef { fn_name: fn_name.clone(), nparams: params.len(), ncap: captures.len() }, span, vec![body_id]);
        self.prog.add_root(def);

        // The closure captures the enclosing locals, in order: args[0..], env[0..].
        let mut cap_children = Vec::new();
        for i in 0..enclosing.params.len() {
            cap_children.push(self.prog.add(Skein::Param(i), span, vec![]));
        }
        for j in 0..enclosing.captures.len() {
            cap_children.push(self.prog.add(Skein::Capture(j), span, vec![]));
        }
        Ok(self.prog.add(Skein::MakeClosure { fn_name, ncap: captures.len() }, span, cap_children))
    }
}

fn sym_name(d: &Datum) -> Result<String, Diagnostic> {
    match d {
        Datum::Sym(s, _) => Ok(s.clone()),
        _ => Err(err(d.span(), "expected a name")),
    }
}

fn err(span: Span, msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(msg).with_primary(span, "here")
}

/// A user identifier (global/define name) -> a C global name. Scheme names allow
/// `- ? ! > ...`; map non-alphanumerics to `_`.
fn gname(name: &str) -> String {
    let mut out = String::from("g_");
    for c in name.chars() {
        if c.is_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out
}

// ─────────────────────────────── C emission ────────────────────────────────

fn scm() -> CType {
    CType::Named("scm".to_string())
}

impl CEmit for Skein {
    fn runtime() -> Runtime {
        Runtime::new().with(RuntimeFragment { name: "rt", includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <stdarg.h>", "#include <string.h>"], deps: &[], code: RUNTIME })
    }

    fn emit_item(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CItem {
        match &node.kind {
            Skein::Program => {
                cx.need("rt");
                CItem::Main { locals: Vec::new(), body: node.children.iter().map(|&c| cx.stmt(c)).collect() }
            }
            Skein::Global { name } => CItem::Global { ty: scm(), name: gname(name) },
            Skein::LambdaDef { fn_name, .. } => {
                let id = node.id;
                let body = cx.expr(node.children[0]);
                CItem::Func(knotkit::backend::CFunc {
                    origin: Some(id),
                    ret: CType::Named("bounce".to_string()),
                    name: fn_name.clone(),
                    params: vec![
                        (CType::Named("scm*".to_string()), "env".to_string()),
                        (CType::Named("scm*".to_string()), "args".to_string()),
                        (CType::Named("int".to_string()), "n".to_string()),
                    ],
                    locals: Vec::new(),
                    body: vec![
                        CStmt::expr(id, CExpr::raw(id, "(void)env")),
                        CStmt::expr(id, CExpr::raw(id, "(void)args")),
                        CStmt::expr(id, CExpr::raw(id, "(void)n")),
                        CStmt::ret(id, body),
                    ],
                })
            }
            _ => unreachable!("non-item node as root: {:?}", node.kind),
        }
    }

    fn emit_stmt(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CStmt {
        let id = node.id;
        match &node.kind {
            Skein::GlobalSet { name } => CStmt::assign(id, gname(name), cx.expr(node.children[0])),
            Skein::ExprStmt => CStmt::expr(id, cx.expr(node.children[0])),
            _ => unreachable!("non-statement node in statement position: {:?}", node.kind),
        }
    }

    fn emit_expr(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CExpr {
        let id = node.id;
        match &node.kind {
            Skein::Int(i) => CExpr::call(id, "scm_int", vec![CExpr::lit(id, i.to_string())]),
            Skein::Flo(f) => CExpr::call(id, "scm_flo", vec![CExpr::lit(id, format!("{f:?}"))]),
            Skein::Bool(b) => CExpr::call(id, "scm_bool", vec![CExpr::lit(id, if *b { "1" } else { "0" })]),
            Skein::Nil => CExpr::call(id, "scm_nil", vec![]),
            Skein::Sym(s) => CExpr::call(id, "scm_sym", vec![CExpr::lit(id, c_string(s))]),
            Skein::Str(s) => CExpr::call(id, "scm_str", vec![CExpr::lit(id, c_string(s))]),
            Skein::Param(i) => CExpr::raw(id, format!("args[{i}]")),
            Skein::Capture(j) => CExpr::raw(id, format!("env[{j}]")),
            Skein::GlobalRef { name } => CExpr::var(id, gname(name)),
            Skein::PrimRef { name } => CExpr::call(id, "scm_prim", vec![CExpr::raw(id, prim_cfn(name).expect("known primitive"))]),
            Skein::MakeClosure { fn_name, ncap } => {
                let mut args = vec![CExpr::raw(id, fn_name.clone()), CExpr::lit(id, ncap.to_string())];
                args.extend(node.children.iter().map(|&c| cx.expr(c)));
                CExpr::call(id, "scm_make_closure", args)
            }
            Skein::Apply | Skein::TailApply => {
                let nargs = node.children.len() - 1;
                let callee = cx.expr(node.children[0]);
                let mut args = vec![callee, CExpr::lit(id, nargs.to_string())];
                args.extend(node.children[1..].iter().map(|&c| cx.expr(c)));
                // Value position trampolines (`scm_apply` -> scm); tail position
                // returns a pending call (`scm_tail` -> bounce).
                let f = if matches!(node.kind, Skein::TailApply) { "scm_tail" } else { "scm_apply" };
                CExpr::call(id, f, args)
            }
            Skein::TailValue => CExpr::call(id, "scm_done", vec![cx.expr(node.children[0])]),
            Skein::If => {
                let cond = cx.expr(node.children[0]);
                let then = cx.expr(node.children[1]);
                let els = cx.expr(node.children[2]);
                CExpr::ternary(id, CExpr::call(id, "scm_truthy", vec![cond]), then, els)
            }
            Skein::Seq => CExpr::seq(id, node.children.iter().map(|&c| cx.expr(c)).collect()),
            Skein::Cons => {
                let a = cx.expr(node.children[0]);
                let d = cx.expr(node.children[1]);
                CExpr::call(id, "scm_cons", vec![a, d])
            }
            _ => unreachable!("non-expression node in expression position: {:?}", node.kind),
        }
    }
}

/// A C string literal for a (simple) Scheme symbol/string.
fn c_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

const RUNTIME: &str = r##"/* Skein runtime: tagged dynamic values, a bump heap (leaks), closures, prims. */
typedef struct scm scm;
struct scm_pair;
struct scm_clos;
struct scm {
    int tag; /* 0 int, 1 flo, 2 bool, 3 nil, 4 sym, 5 str, 6 pair, 7 clos, 8 prim */
    union {
        long long i;
        double f;
        const char* s;
        struct scm_pair* p;
        struct scm_clos* c;
        scm (*prim)(scm*, int);
    } as;
};
/* A trampoline result: either a final value, or a pending tail call. Compiled
   procedures return a `bounce`; `scm_apply` loops on tail calls (constant stack). */
typedef struct bounce { int tail; scm val; scm fn; scm args[16]; int n; } bounce;
struct scm_pair { scm car; scm cdr; };
struct scm_clos { bounce (*code)(scm*, scm*, int); scm* env; int nenv; };

static char scm_heap[1 << 26];
static long scm_hp = 0;
static void* scm_alloc(long n) {
    void* p = &scm_heap[scm_hp];
    scm_hp += (n + 7) & ~7L;
    if (scm_hp > (long)sizeof(scm_heap)) { fprintf(stderr, "skein: heap exhausted\n"); exit(1); }
    return p;
}

static scm scm_int(long long i) { scm v; v.tag = 0; v.as.i = i; return v; }
static scm scm_flo(double f) { scm v; v.tag = 1; v.as.f = f; return v; }
static scm scm_bool(int b) { scm v; v.tag = 2; v.as.i = b ? 1 : 0; return v; }
static scm scm_nil(void) { scm v; v.tag = 3; return v; }
static scm scm_sym(const char* s) { scm v; v.tag = 4; v.as.s = s; return v; }
static scm scm_str(const char* s) { scm v; v.tag = 5; v.as.s = s; return v; }
static scm scm_cons(scm a, scm d) { struct scm_pair* p = scm_alloc(sizeof *p); p->car = a; p->cdr = d; scm v; v.tag = 6; v.as.p = p; return v; }
static scm scm_prim(scm (*fn)(scm*, int)) { scm v; v.tag = 8; v.as.prim = fn; return v; }
static int scm_truthy(scm v) { return !(v.tag == 2 && v.as.i == 0); }
static double scm_nf(scm v) { return v.tag == 1 ? v.as.f : (double)v.as.i; }

static scm scm_make_closure(bounce (*code)(scm*, scm*, int), int ncap, ...) {
    struct scm_clos* c = scm_alloc(sizeof *c);
    c->code = code; c->nenv = ncap;
    c->env = ncap ? scm_alloc((long)sizeof(scm) * ncap) : 0;
    va_list ap; va_start(ap, ncap);
    for (int i = 0; i < ncap; i++) c->env[i] = va_arg(ap, scm);
    va_end(ap);
    scm v; v.tag = 7; v.as.c = c; return v;
}

/* A finished value (no further tail call). */
static bounce scm_done(scm v) { bounce b; b.tail = 0; b.val = v; b.n = 0; return b; }

/* A pending tail call: bounce back to scm_apply rather than recurse in C. */
static bounce scm_tail(scm fn, int n, ...) {
    bounce b; b.tail = 1; b.fn = fn; b.n = n;
    va_list ap; va_start(ap, n);
    for (int i = 0; i < n; i++) b.args[i] = va_arg(ap, scm);
    va_end(ap);
    return b;
}

/* The trampoline: invoke a procedure, then bounce on each tail call. Tail
   recursion thus runs in constant C stack; non-tail calls recurse normally. */
static scm scm_apply(scm fn, int n, ...) {
    scm args[16];
    va_list ap; va_start(ap, n);
    for (int i = 0; i < n; i++) args[i] = va_arg(ap, scm);
    va_end(ap);
    for (;;) {
        if (fn.tag == 8) return fn.as.prim(args, n);
        if (fn.tag != 7) { fprintf(stderr, "skein: attempt to call a non-procedure (tag %d)\n", fn.tag); exit(1); }
        bounce b = fn.as.c->code(fn.as.c->env, args, n);
        if (!b.tail) return b.val;
        fn = b.fn; n = b.n;
        for (int i = 0; i < n; i++) args[i] = b.args[i];
    }
}

static void scm_write(scm v) {
    switch (v.tag) {
        case 0: printf("%lld", v.as.i); break;
        case 1: printf("%g", v.as.f); break;
        case 2: printf(v.as.i ? "#t" : "#f"); break;
        case 3: printf("()"); break;
        case 4: case 5: printf("%s", v.as.s); break;
        case 6: {
            printf("(");
            scm cur = v; int first = 1;
            while (cur.tag == 6) { if (!first) printf(" "); first = 0; scm_write(cur.as.p->car); cur = cur.as.p->cdr; }
            if (cur.tag != 3) { printf(" . "); scm_write(cur); }
            printf(")");
        } break;
        default: printf("#<procedure>"); break;
    }
}

static scm scm_p_add(scm* a, int n) { int fl = 0; for (int i = 0; i < n; i++) if (a[i].tag == 1) fl = 1; if (fl) { double s = 0; for (int i = 0; i < n; i++) s += scm_nf(a[i]); return scm_flo(s); } long long s = 0; for (int i = 0; i < n; i++) s += a[i].as.i; return scm_int(s); }
static scm scm_p_mul(scm* a, int n) { int fl = 0; for (int i = 0; i < n; i++) if (a[i].tag == 1) fl = 1; if (fl) { double s = 1; for (int i = 0; i < n; i++) s *= scm_nf(a[i]); return scm_flo(s); } long long s = 1; for (int i = 0; i < n; i++) s *= a[i].as.i; return scm_int(s); }
static scm scm_p_sub(scm* a, int n) {
    if (n == 1) { return a[0].tag == 1 ? scm_flo(-a[0].as.f) : scm_int(-a[0].as.i); }
    int fl = 0; for (int i = 0; i < n; i++) if (a[i].tag == 1) fl = 1;
    if (fl) { double s = scm_nf(a[0]); for (int i = 1; i < n; i++) s -= scm_nf(a[i]); return scm_flo(s); }
    long long s = a[0].as.i; for (int i = 1; i < n; i++) s -= a[i].as.i; return scm_int(s);
}
static scm scm_p_numeq(scm* a, int n) { (void)n; return scm_bool(scm_nf(a[0]) == scm_nf(a[1])); }
static scm scm_p_lt(scm* a, int n) { (void)n; return scm_bool(scm_nf(a[0]) < scm_nf(a[1])); }
static scm scm_p_gt(scm* a, int n) { (void)n; return scm_bool(scm_nf(a[0]) > scm_nf(a[1])); }
static scm scm_p_eq(scm* a, int n) {
    (void)n; scm x = a[0], y = a[1];
    if (x.tag != y.tag) return scm_bool(0);
    switch (x.tag) {
        case 0: case 2: return scm_bool(x.as.i == y.as.i);
        case 1: return scm_bool(x.as.f == y.as.f);
        case 3: return scm_bool(1);
        case 4: case 5: return scm_bool(strcmp(x.as.s, y.as.s) == 0);
        case 6: return scm_bool(x.as.p == y.as.p);
        case 7: return scm_bool(x.as.c == y.as.c);
        default: return scm_bool(x.as.prim == y.as.prim);
    }
}
static scm scm_p_null(scm* a, int n) { (void)n; return scm_bool(a[0].tag == 3); }
static scm scm_p_pair(scm* a, int n) { (void)n; return scm_bool(a[0].tag == 6); }
static scm scm_p_cons(scm* a, int n) { (void)n; return scm_cons(a[0], a[1]); }
static scm scm_p_car(scm* a, int n) { (void)n; if (a[0].tag != 6) { fprintf(stderr, "skein: car of non-pair\n"); exit(1); } return a[0].as.p->car; }
static scm scm_p_cdr(scm* a, int n) { (void)n; if (a[0].tag != 6) { fprintf(stderr, "skein: cdr of non-pair\n"); exit(1); } return a[0].as.p->cdr; }
static scm scm_p_not(scm* a, int n) { (void)n; return scm_bool(!scm_truthy(a[0])); }
static scm scm_p_list(scm* a, int n) { scm r = scm_nil(); for (int i = n - 1; i >= 0; i--) r = scm_cons(a[i], r); return r; }
static scm scm_p_display(scm* a, int n) { (void)n; scm_write(a[0]); return scm_nil(); }
static scm scm_p_newline(scm* a, int n) { (void)a; (void)n; putchar('\n'); return scm_nil(); }
"##;
