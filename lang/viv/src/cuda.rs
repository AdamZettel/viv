// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! GPU kernels: `kernel def`, its emulation on the CPU, and
//! its CUDA C.
//!
//! A `kernel def` is one of two kinds:
//!
//! - an elementwise tree, `kernel def f(a: Num, x: Arr) -> Arr { return a * x
//!   + exp(x) }`: one thread per element, the tree fused;
//! - an explicit kernel, with `thread_id()`, `block_id()`, `block_size()`,
//!   `grid_size()`, `s = shared(N)` and `barrier()`.
//!
//! Both are called like any def, `y = f(x, y)`. The grid is computed from
//! the size of the first array parameter: 256 threads per block and as many
//! blocks as cover it. An explicit kernel that needs another grid says so in
//! a top-level line, `schedule f(blocks = 2, threads = 8)` (the schedule
//! line); either key may be left out, and a
//! value is an integer expression of literals, the kernel's Int parameters,
//! `len` of its array parameters and global constants.
//!
//! [`lower`] rewrites each one (in `crate::parse_before_grad`; a reader of
//! the text as written, `crate::parse_keeping_rungs`, sees the kernel as
//! written) into the def every engine runs: a loop over the blocks, and inside it the body cut at each
//! barrier, each piece a loop over the threads (MCUDA: Stratton, Stone and
//! Hwu, LCPC 2008). The interpreter, the C backend and the wasm backend see
//! only loops they already run, so the three agree, and the interpreter runs
//! (and steps through) a kernel thread by thread. The rewrite checks what a
//! GPU does not: indices out of range, shared memory read before it is
//! written, and races (two threads touching one element with no barrier
//! between, or two blocks touching one element); each stops the run with the
//! block, the thread and the element, at the line of the access.
//!
//! [`emit`] writes the CUDA C of the kernels as written (and of every
//! `parallel for` whose body is in the kernel subset), with Viv's libm for the
//! device and a host launcher per kernel. It is generated here and compiled
//! with nvcc elsewhere.

use crate::ast::*;
use knotkit::diag::Diagnostic;
use knotkit::span::Span;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};

/// The statement `kernel def` puts first in the def's body.
pub const MARK: &str = "viv__kernel";
/// Threads per block of an elementwise kernel, and of an explicit kernel
/// whose `schedule` line does not give `threads`.
pub const ELEMENTWISE_THREADS: i64 = 256;
/// The statement a `schedule f(blocks = ..., threads = ...)` line parses to
/// (the parser's `schedule`); [`lower`] and [`emit`] read and remove it.
pub const SCHEDULE: &str = "viv__schedule";
/// The keys a kernel's schedule line takes.
const GRID_KEYS: [&str; 3] = ["blocks", "threads", "cover"];
/// CUDA's limit on threads per block.
pub const MAX_THREADS: i64 = 1024;
/// CUDA's limit on the blocks of a grid along y.
pub const MAX_BLOCKS_Y: i64 = 65535;
/// The prefix of every name the rewrite makes.
const P: &str = "viv_k_";

const THREAD_ID: &str = "thread_id";
const BLOCK_ID: &str = "block_id";
const BLOCK_SIZE: &str = "block_size";
const GRID_SIZE: &str = "grid_size";
const SHARED: &str = "shared";
const BARRIER: &str = "barrier";
const DECLARE: &str = "viv__declare";
/// The functions a kernel may call: (Viv name, CUDA C name for a number).
const MATH: &[(&str, &str)] = &[
    ("abs", "fabs"),
    ("sqrt", "sqrt"),
    ("exp", "viv_exp_cr"),
    ("log", "viv_log"),
    ("sin", "viv_sin"),
    ("cos", "viv_cos"),
    ("tan", "viv_tan"),
    ("tanh", "viv_tanh"),
    ("floor", "floor"),
    ("ceil", "ceil"),
];

thread_local! {
    static RAW: Cell<bool> = const { Cell::new(false) };
    /// whether the kernel being rewritten has a 2-D grid (its builtins then
    /// split the flat thread and block numbers into x and y)
    static TWO_D: Cell<bool> = const { Cell::new(false) };
}

/// The axis of a grid builtin: `thread_id()` and `thread_id(0)` are x,
/// `thread_id(1)` is y; `None` for any other argument.
fn axis_of(args: &[Expr]) -> Option<usize> {
    match args {
        [] => Some(0),
        [Expr::Num { value, .. }] if *value == 0.0 => Some(0),
        [Expr::Num { value, .. }] if *value == 1.0 => Some(1),
        _ => None,
    }
}

/// Runs `f` with kernels left as written (for [`emit`], and while parsing
/// the rewrite's own templates).
pub fn raw<T>(f: impl FnOnce() -> T) -> T {
    RAW.with(|r| {
        let old = r.replace(true);
        let v = f();
        r.set(old);
        v
    })
}

/// Marks `def` as a kernel (the parser, at `kernel def`).
pub fn mark(def: &mut Def) {
    let span = def.name_span;
    def.body.insert(0, Stmt::ExprStmt { expr: Expr::Call { callee: MARK.into(), args: vec![], span }, span });
}

pub fn is_kernel(d: &Def) -> bool {
    matches!(d.body.first(), Some(Stmt::ExprStmt { expr: Expr::Call { callee, .. }, .. }) if callee == MARK)
}

fn has_kernels(p: &Program) -> bool {
    p.items.iter().any(|i| matches!(i, Item::Def(d) if is_kernel(d)))
}

fn err(span: Span, msg: impl Into<String>, label: &str) -> Diagnostic {
    Diagnostic::error(msg).with_primary(span, label)
}

// ---------------------------------------------------------------- the subset

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Num,
    Int,
    Bool,
    Arr,
    IntArr,
}

fn param_ty(p: &Param) -> Result<Ty, Diagnostic> {
    match &p.ty {
        Some(TypeExpr::Num) => Ok(Ty::Num),
        Some(TypeExpr::Int) => Ok(Ty::Int),
        Some(TypeExpr::Bool) => Ok(Ty::Bool),
        Some(TypeExpr::Arr) => Ok(Ty::Arr),
        Some(TypeExpr::IntList) => Ok(Ty::IntArr),
        None => Err(err(p.span, format!("a kernel's parameter needs a type: `{}: Arr`, `Num`, `Int`, `Bool` or `[Int]`", p.name), "no type")),
        Some(_) => Err(err(p.span, format!("`{}` cannot be passed to a GPU kernel: its parameters are `Arr`, `[Int]`, `Num`, `Int` or `Bool`", p.name), "this type")),
    }
}

/// What a kernel body may be: the facts the checks and the rewrite share.
struct Kern {
    name: String,
    params: Vec<(String, Ty)>,
    /// the array parameters the body writes, in parameter order
    written: Vec<String>,
    /// shared arrays and their sizes, in order of declaration
    shared: Vec<(String, Expr)>,
}

impl Kern {
    fn ty(&self, v: &str) -> Option<Ty> {
        self.params.iter().find(|(n, _)| n == v).map(|(_, t)| *t)
    }
    fn is_array(&self, v: &str) -> bool {
        matches!(self.ty(v), Some(Ty::Arr | Ty::IntArr)) || self.is_shared(v)
    }
    fn is_shared(&self, v: &str) -> bool {
        self.shared.iter().any(|(n, _)| n == v)
    }
    /// shared arrays and written parameters: the arrays whose races are checked
    fn tracked(&self, v: &str) -> bool {
        self.is_shared(v) || self.written.iter().any(|w| w == v)
    }
}

fn is_call(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::Call { callee, .. } if callee == name)
}

fn is_barrier(s: &Stmt) -> bool {
    matches!(s, Stmt::ExprStmt { expr, .. } if is_call(expr, BARRIER))
}

fn shared_decl(s: &Stmt) -> Option<(&str, &Expr)> {
    match s {
        Stmt::Assign { name, value: Expr::Call { callee, args, .. }, .. } if callee == SHARED && args.len() == 1 => Some((name, &args[0])),
        _ => None,
    }
}

fn holds_barrier(s: &Stmt) -> bool {
    match s {
        _ if is_barrier(s) => true,
        Stmt::If { then_body, else_body, .. } => then_body.iter().chain(else_body).any(holds_barrier),
        Stmt::While { body, .. } | Stmt::For { body, .. } => body.iter().any(holds_barrier),
        _ => false,
    }
}

/// Checks an expression of a kernel body against the subset.
fn check_expr(e: &Expr, k: &Kern, defs: &HashSet<String>, explicit: bool) -> Result<(), Diagnostic> {
    let what = |s: &str| err(e.span(), format!("a GPU kernel cannot use {s}"), "not in the kernel subset");
    match e {
        Expr::Num { .. } | Expr::Bool { .. } | Expr::Var { .. } => Ok(()),
        Expr::Str { .. } => Err(what("strings")),
        Expr::Imag { .. } => Err(what("complex numbers")),
        Expr::ArrLit { .. } => Err(what("an array literal (it allocates)").with_note("arrays come in as parameters, or `shared(N)` for a block's own")),
        Expr::Lambda { .. } => Err(what("a function value")),
        Expr::RecordLit { .. } | Expr::Field { .. } => Err(what("records")),
        Expr::Index { base, indices, .. } => {
            match &**base {
                Expr::Var { name, .. } if k.is_array(name) => {}
                _ => return Err(err(base.span(), "a kernel indexes its array parameters and its shared arrays only", "not one of them")),
            }
            if indices.len() != 1 {
                return Err(err(e.span(), "kernel arrays have one index in Phase 1", "more than one index"));
            }
            check_expr(&indices[0], k, defs, explicit)
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => check_expr(operand, k, defs, explicit),
        Expr::Binary { op, lhs, rhs, .. } => {
            if *op == BinOp::Matmul {
                return Err(what("`@`"));
            }
            check_expr(lhs, k, defs, explicit)?;
            check_expr(rhs, k, defs, explicit)
        }
        Expr::Call { callee, args, span } => {
            let c = callee.as_str();
            let builtin = [THREAD_ID, BLOCK_ID, BLOCK_SIZE, GRID_SIZE].contains(&c);
            if builtin && explicit && axis_of(args).is_some() {
                return Ok(());
            }
            if builtin && explicit {
                return Err(err(*span, format!("`{c}` takes no argument (the x axis) or an axis written as a number: `{c}(0)` for x, `{c}(1)` for y"), "not an axis"));
            }
            if c == "len" && args.len() == 1 && matches!(&args[0], Expr::Var { name, .. } if k.ty(name).is_some_and(|t| matches!(t, Ty::Arr | Ty::IntArr))) {
                return Ok(());
            }
            if (MATH.iter().any(|(n, _)| *n == c) && args.len() == 1) || (c == "pow" && args.len() == 2) {
                return args.iter().try_for_each(|a| check_expr(a, k, defs, explicit));
            }
            if c == SHARED {
                return Err(err(*span, "`shared(N)` is a statement of its own at the top of a kernel: `s = shared(N)`", "here"));
            }
            if c == BARRIER {
                return Err(err(*span, "`barrier()` is a statement of its own", "inside an expression"));
            }
            if builtin {
                return Err(err(*span, format!("`{c}()` is for explicit kernels (an elementwise kernel's threads are its elements)"), "here"));
            }
            if defs.contains(c) {
                return Err(err(*span, format!("a kernel cannot call the def `{c}` yet: device functions are Phase 2"), "a call of a def"));
            }
            let why = match c {
                "append" | "zeros" | "ones" | "arange" | "linspace" | "copy" => " (it allocates)",
                "print" | "show" | "write_file" | "read_file" | "read_csv" | "write_csv" => " (no I/O on a GPU)",
                "rand" | "randn" => " (no random numbers in Phase 1)",
                _ => "",
            };
            Err(err(*span, format!("`{c}` cannot run in a GPU kernel{why}"), "not in the kernel subset")
                .with_note("a kernel may call abs, sqrt, exp, log, sin, cos, tan, tanh, floor, ceil, pow and len"))
        }
    }
}

/// Checks a statement of an explicit kernel; `top` is whether it is at the
/// body's top level.
fn check_stmt(s: &Stmt, k: &Kern, defs: &HashSet<String>, top: bool) -> Result<(), Diagnostic> {
    let no = |what: &str| err(s.span(), format!("a GPU kernel cannot hold {what}"), "not in the kernel subset");
    match s {
        Stmt::Assign { name, name_span, value, .. } => {
            if k.ty(name).is_some() {
                return Err(err(*name_span, format!("a kernel cannot rebind its parameter `{name}`"), "a parameter"));
            }
            if let Some((_, size)) = shared_decl(s) {
                if !top {
                    return Err(err(s.span(), "`shared(N)` belongs at the top level of the kernel", "nested"));
                }
                return match size {
                    Expr::Num { value, .. } if *value >= 1.0 && value.fract() == 0.0 => Ok(()),
                    Expr::Var { .. } => Ok(()),
                    _ => Err(err(size.span(), "the size of a shared array is a whole-number literal or a global constant", "not a constant")),
                };
            }
            check_expr(value, k, defs, true)
        }
        Stmt::ExprStmt { expr, .. } => {
            if is_call(expr, BARRIER) || is_call(expr, DECLARE) {
                return Ok(());
            }
            Err(no("a call made for its effect"))
        }
        Stmt::SetIndex { base, indices, value, .. } => {
            match &**base {
                Expr::Var { name, .. } if k.is_array(name) => {}
                _ => return Err(err(base.span(), "a kernel writes its array parameters and its shared arrays only", "not one of them")),
            }
            if indices.len() != 1 {
                return Err(err(s.span(), "kernel arrays have one index in Phase 1", "more than one index"));
            }
            check_expr(&indices[0], k, defs, true)?;
            check_expr(value, k, defs, true)
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            check_expr(cond, k, defs, true)?;
            then_body.iter().chain(else_body).try_for_each(|t| check_stmt(t, k, defs, false))
        }
        Stmt::While { cond, body, .. } => {
            check_expr(cond, k, defs, true)?;
            body.iter().try_for_each(|t| check_stmt(t, k, defs, false))
        }
        Stmt::For { bound, body, parallel, .. } => {
            if *parallel {
                return Err(no("a `parallel for` (the threads are the parallel loop)"));
            }
            check_expr(bound, k, defs, true)?;
            body.iter().try_for_each(|t| check_stmt(t, k, defs, false))
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => Ok(()),
        Stmt::Return { .. } => Err(err(s.span(), "an explicit kernel does not return: it writes its array parameters, which the call returns", "a return")),
        Stmt::Show { .. } | Stmt::Narrate { .. } => Err(no("`show` or `narrate` (no output from a GPU)")),
        Stmt::Test { .. } => Err(no("a test block")),
        Stmt::SetField { .. } => Err(no("records")),
        Stmt::StepEnter { .. } | Stmt::StepExit { .. } => Err(no("a step")),
        Stmt::Lens { .. } => Err(no("a lens statement")),
    }
}

fn written_in(body: &[Stmt], out: &mut Vec<String>) {
    for s in body {
        match s {
            Stmt::SetIndex { base, .. } => {
                if let Expr::Var { name, .. } = &**base {
                    if !out.contains(name) {
                        out.push(name.clone());
                    }
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                written_in(then_body, out);
                written_in(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => written_in(body, out),
            _ => {}
        }
    }
}

fn kern_of(d: &Def) -> Result<Kern, Diagnostic> {
    let mut params = Vec::new();
    for p in &d.params {
        params.push((p.name.clone(), param_ty(p)?));
    }
    let body = &d.body[1..];
    let mut shared = Vec::new();
    for s in body {
        if let Some((n, size)) = shared_decl(s) {
            shared.push((n.to_string(), size.clone()));
        }
    }
    let mut w = Vec::new();
    written_in(body, &mut w);
    let written = params.iter().filter(|(n, _)| w.contains(n)).map(|(n, _)| n.clone()).collect();
    Ok(Kern { name: d.name.clone(), params, written, shared })
}

/// The kernel is an elementwise tree: its body is one `return`.
fn elementwise_body(d: &Def) -> Option<&Expr> {
    match &d.body[1..] {
        [Stmt::Return { value, .. }] => Some(value),
        _ => None,
    }
}

fn check_tree(e: &Expr, k: &Kern) -> Result<(), Diagnostic> {
    let bad = |label: &str| {
        err(e.span(), "an elementwise kernel's `return` is a tree of `+ - * /`, `**`, unary `-` and abs, sqrt, exp, log, sin, cos, tan, tanh, floor, ceil over its parameters", label)
    };
    match e {
        Expr::Num { .. } => Ok(()),
        Expr::Var { name, .. } => match k.ty(name) {
            Some(Ty::Arr | Ty::Num | Ty::Int) => Ok(()),
            _ => Err(bad("not an `Arr`, `Num` or `Int` parameter")),
        },
        Expr::Neg { operand, .. } => check_tree(operand, k),
        Expr::Binary { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, lhs, rhs, .. } => {
            check_tree(lhs, k)?;
            check_tree(rhs, k)
        }
        Expr::Call { callee, args, .. } if (args.len() == 1 && MATH.iter().any(|(n, _)| n == callee)) || (callee == "pow" && args.len() == 2) => {
            args.iter().try_for_each(|a| check_tree(a, k))
        }
        _ => Err(bad("not elementwise")),
    }
}

fn tree_arrays(e: &Expr, k: &Kern, out: &mut Vec<String>) {
    match e {
        Expr::Var { name, .. } if k.ty(name) == Some(Ty::Arr) && !out.contains(name) => out.push(name.clone()),
        Expr::Neg { operand, .. } => tree_arrays(operand, k, out),
        Expr::Binary { lhs, rhs, .. } => {
            tree_arrays(lhs, k, out);
            tree_arrays(rhs, k, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|a| tree_arrays(a, k, out)),
        _ => {}
    }
}

// ---------------------------------------------------------------- AST helpers

/// Parses Viv statements written by the rewrite, every span set to `span`.
fn tmpl(text: &str, span: Span) -> Vec<Stmt> {
    let src = format!("def {P}template() {{\n{text}\n}}\n");
    let toks = crate::lexer::lex(&src, span.source).unwrap_or_else(|d| panic!("cuda template does not lex: {d:?}\n{src}"));
    let prog = raw(|| crate::parser::parse(toks, span.source)).unwrap_or_else(|d| panic!("cuda template does not parse: {d:?}\n{src}"));
    let ints = prog.whole;
    let mut body = match prog.items.into_iter().next() {
        Some(Item::Def(d)) => d.body,
        _ => unreachable!("a template is one def"),
    };
    for s in &mut body {
        respan_stmt(s, span, &ints);
    }
    body
}

/// A def written as text, its spans fresh near `span` (as [`tmpl`]'s).
fn tmpl_def(text: &str, span: Span) -> Def {
    let toks = crate::lexer::lex(text, span.source).unwrap_or_else(|d| panic!("cuda template does not lex: {d:?}\n{text}"));
    let prog = raw(|| crate::parser::parse(toks, span.source)).unwrap_or_else(|d| panic!("cuda template does not parse: {d:?}\n{text}"));
    let ints = prog.whole;
    let mut d = match prog.items.into_iter().next() {
        Some(Item::Def(d)) => d,
        _ => unreachable!("a def template is one def"),
    };
    for s in &mut d.body {
        respan_stmt(s, span, &ints);
    }
    d.name_span = fresh_near(span);
    d.span = fresh_near(span);
    d
}

/// What the rewrite needs to give its nodes spans of their own: the source
/// text, and every span already taken.
#[derive(Default)]
struct Spans {
    src: String,
    used: HashSet<(u32, u32)>,
    /// for each start, the next end to try
    next_end: HashMap<u32, u32>,
}

thread_local! {
    static SPANS: std::cell::RefCell<Spans> = std::cell::RefCell::new(Spans::default());
    static WHOLE: std::cell::RefCell<Vec<((u32, u32), i64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A span no other node of the program has, inside the source text and on
/// character boundaries, starting at or after `near` (the statement the node
/// was made for, so an error in it points there). The interpreter matches
/// what the lowering decided about a node (an Int, say) by its span, so two
/// nodes must never share one; and the tools that read the source by span
/// must find text there. The same program gets the same spans every time it
/// is parsed.
fn fresh_near(near: Span) -> Span {
    SPANS.with(|cell| {
        let mut st = cell.borrow_mut();
        let len = st.src.len() as u32;
        let mut a = near.start.min(len);
        // outside a rewrite (the CUDA emitter's own nodes), or every span of
        // the text taken: the node shares `near`'s span
        let mut wrapped = false;
        loop {
            if len == 0 || (wrapped && a >= near.start.min(len)) {
                return near;
            }
            if a < len && st.src.is_char_boundary(a as usize) {
                let mut b = st.next_end.get(&a).copied().unwrap_or(a + 1);
                while b <= len {
                    if st.src.is_char_boundary(b as usize) && !st.used.contains(&(a, b)) {
                        st.used.insert((a, b));
                        st.next_end.insert(a, b + 1);
                        return Span { source: near.source, start: a, end: b };
                    }
                    b += 1;
                }
                st.next_end.insert(a, len + 1);
            }
            a += 1;
            if a >= len {
                // the text after `near` is used up: go on from its start
                a = 0;
                wrapped = true;
            }
        }
    })
}

fn spans_of_stmt(s: &Stmt, out: &mut HashSet<(u32, u32)>) {
    out.insert((s.span().start, s.span().end));
    match s {
        Stmt::Assign { name_span, value, .. } => {
            out.insert((name_span.start, name_span.end));
            spans_of_expr(value, out);
        }
        Stmt::Return { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => spans_of_expr(value, out),
        Stmt::SetIndex { base, indices, value, .. } => {
            spans_of_expr(base, out);
            indices.iter().for_each(|e| spans_of_expr(e, out));
            spans_of_expr(value, out);
        }
        Stmt::SetField { base, value, field_span, .. } => {
            out.insert((field_span.start, field_span.end));
            spans_of_expr(base, out);
            spans_of_expr(value, out);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            spans_of_expr(cond, out);
            then_body.iter().chain(else_body).for_each(|t| spans_of_stmt(t, out));
        }
        Stmt::While { cond, body, .. } => {
            spans_of_expr(cond, out);
            body.iter().for_each(|t| spans_of_stmt(t, out));
        }
        Stmt::For { var_span, bound, body, .. } => {
            out.insert((var_span.start, var_span.end));
            spans_of_expr(bound, out);
            body.iter().for_each(|t| spans_of_stmt(t, out));
        }
        Stmt::Test { body, .. } => body.iter().for_each(|t| spans_of_stmt(t, out)),
        _ => {}
    }
}

fn spans_of_expr(e: &Expr, out: &mut HashSet<(u32, u32)>) {
    out.insert((e.span().start, e.span().end));
    match e {
        Expr::ArrLit { elems, .. } => elems.iter().for_each(|x| spans_of_expr(x, out)),
        Expr::Index { base, indices, .. } => {
            spans_of_expr(base, out);
            indices.iter().for_each(|x| spans_of_expr(x, out));
        }
        Expr::Lambda { body, .. } => spans_of_expr(body, out),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => spans_of_expr(operand, out),
        Expr::Binary { lhs, rhs, .. } => {
            spans_of_expr(lhs, out);
            spans_of_expr(rhs, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|x| spans_of_expr(x, out)),
        Expr::RecordLit { fields, type_span, .. } => {
            out.insert((type_span.start, type_span.end));
            fields.iter().for_each(|(_, s, x)| {
                out.insert((s.start, s.end));
                spans_of_expr(x, out)
            });
        }
        Expr::Field { base, field_span, .. } => {
            out.insert((field_span.start, field_span.end));
            spans_of_expr(base, out);
        }
        _ => {}
    }
}

fn respan_stmt(s: &mut Stmt, user: Span, ints: &HashMap<(u32, u32), i64>) {
    match s {
        // the error a check stops with is shown at the user's statement
        Stmt::ExprStmt { expr: Expr::Call { callee, args, span: cs }, span } if callee == "stop" => {
            *span = user;
            *cs = user;
            args.iter_mut().for_each(|x| respan_expr(x, ints, user));
        }
        Stmt::Assign { name_span, value, span, .. } => {
            *name_span = fresh_near(user);
            *span = fresh_near(user);
            respan_expr(value, ints, user);
        }
        Stmt::Return { value, span } | Stmt::ExprStmt { expr: value, span } | Stmt::Show { expr: value, span } => {
            *span = fresh_near(user);
            respan_expr(value, ints, user);
        }
        Stmt::SetIndex { base, indices, value, span } => {
            *span = fresh_near(user);
            respan_expr(base, ints, user);
            indices.iter_mut().for_each(|x| respan_expr(x, ints, user));
            respan_expr(value, ints, user);
        }
        Stmt::If { cond, then_body, else_body, span } => {
            *span = fresh_near(user);
            respan_expr(cond, ints, user);
            then_body.iter_mut().chain(else_body.iter_mut()).for_each(|t| respan_stmt(t, user, ints));
        }
        Stmt::While { cond, body, span } => {
            *span = fresh_near(user);
            respan_expr(cond, ints, user);
            body.iter_mut().for_each(|t| respan_stmt(t, user, ints));
        }
        Stmt::For { var_span, bound, body, span, .. } => {
            *span = fresh_near(user);
            *var_span = fresh_near(user);
            respan_expr(bound, ints, user);
            body.iter_mut().for_each(|t| respan_stmt(t, user, ints));
        }
        Stmt::Break { span } | Stmt::Continue { span } | Stmt::Narrate { span, .. } => *span = fresh_near(user),
        _ => {}
    }
}

/// Gives every node of a template a span of its own (the interpreter keys
/// what it knows about a node, such as being an Int, by its span), and
/// records the whole-number literals as Ints where an Int is expected.
fn respan_expr(e: &mut Expr, ints: &HashMap<(u32, u32), i64>, user: Span) {
    match e {
        Expr::Num { span, .. } => {
            let int = ints.get(&(span.start, span.end)).copied();
            *span = fresh_near(user);
            if let Some(v) = int {
                WHOLE.with(|w| w.borrow_mut().push(((span.start, span.end), v)));
            }
        }
        Expr::Imag { span, .. } | Expr::Bool { span, .. } | Expr::Str { span, .. } | Expr::Var { span, .. } => *span = fresh_near(user),
        Expr::ArrLit { elems, span } => {
            *span = fresh_near(user);
            elems.iter_mut().for_each(|x| respan_expr(x, ints, user));
        }
        Expr::Index { base, indices, span } => {
            *span = fresh_near(user);
            respan_expr(base, ints, user);
            indices.iter_mut().for_each(|x| respan_expr(x, ints, user));
        }
        Expr::Lambda { body, span, .. } => {
            *span = fresh_near(user);
            respan_expr(body, ints, user);
        }
        Expr::Neg { operand, span } | Expr::Not { operand, span } => {
            *span = fresh_near(user);
            respan_expr(operand, ints, user);
        }
        Expr::Binary { lhs, rhs, span, .. } => {
            *span = fresh_near(user);
            respan_expr(lhs, ints, user);
            respan_expr(rhs, ints, user);
        }
        Expr::Call { args, span, .. } => {
            *span = fresh_near(user);
            args.iter_mut().for_each(|x| respan_expr(x, ints, user));
        }
        Expr::RecordLit { fields, span, type_span, .. } => {
            *span = fresh_near(user);
            *type_span = fresh_near(user);
            fields.iter_mut().for_each(|(_, s, x)| {
                *s = fresh_near(user);
                respan_expr(x, ints, user)
            });
        }
        Expr::Field { base, span, field_span, .. } => {
            *span = fresh_near(user);
            *field_span = fresh_near(user);
            respan_expr(base, ints, user);
        }
    }
}

fn var(name: &str, near: Span) -> Expr {
    Expr::Var { name: name.into(), span: fresh_near(near) }
}

/// The emulator's value of a grid builtin. The emulator numbers the threads
/// of a block and the blocks of a grid with one number each, x fastest, as
/// CUDA does (`t = threadIdx.y * blockDim.x + threadIdx.x`): a 1-D kernel
/// reads them as they are, a 2-D one splits them.
fn grid_builtin(callee: &str, axis: usize, sp: Span) -> Option<Expr> {
    let flat = !TWO_D.with(|c| c.get());
    let text = match (callee, axis) {
        (THREAD_ID, 0) if flat => format!("{P}t"),
        (BLOCK_ID, 0) if flat => format!("{P}b"),
        (BLOCK_SIZE, 0) if flat => format!("{P}threads"),
        (GRID_SIZE, 0) if flat => format!("{P}blocks"),
        (THREAD_ID, 0) => format!("{P}t % {P}tx"),
        (THREAD_ID, _) => format!("{P}t // {P}tx"),
        (BLOCK_ID, 0) => format!("{P}b % {P}bx"),
        (BLOCK_ID, _) => format!("{P}b // {P}bx"),
        (BLOCK_SIZE, 0) => format!("{P}tx"),
        (BLOCK_SIZE, _) => format!("{P}ty"),
        (GRID_SIZE, 0) => format!("{P}bx"),
        (GRID_SIZE, _) => format!("{P}by"),
        _ => return None,
    };
    match tmpl(&format!("{P}axis = {text}"), sp).pop() {
        Some(Stmt::Assign { value, .. }) => Some(value),
        _ => None,
    }
}

/// Replaces the thread and grid builtins with the emulator's variables.
fn subst_expr(e: &Expr) -> Expr {
    let sp = e.span();
    match e {
        Expr::Call { callee, args, .. } if [THREAD_ID, BLOCK_ID, BLOCK_SIZE, GRID_SIZE].contains(&callee.as_str()) && axis_of(args).is_some() => {
            grid_builtin(callee, axis_of(args).unwrap(), sp).unwrap_or_else(|| e.clone())
        }
        Expr::Call { callee, args, span } => Expr::Call { callee: callee.clone(), args: args.iter().map(subst_expr).collect(), span: *span },
        Expr::Index { base, indices, span } => Expr::Index { base: base.clone(), indices: indices.iter().map(subst_expr).collect(), span: *span },
        Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(subst_expr(operand)), span: *span },
        Expr::Not { operand, span } => Expr::Not { operand: Box::new(subst_expr(operand)), span: *span },
        Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op: *op, lhs: Box::new(subst_expr(lhs)), rhs: Box::new(subst_expr(rhs)), span: *span },
        _ => e.clone(),
    }
}

fn subst_stmt(s: &Stmt) -> Stmt {
    match s {
        Stmt::Assign { name, name_span, value, span } => Stmt::Assign { name: name.clone(), name_span: *name_span, value: subst_expr(value), span: *span },
        _ => s.clone(),
    }
}

fn reads(e: &Expr, out: &mut Vec<(String, Span)>) {
    match e {
        Expr::Var { name, span } => out.push((name.clone(), *span)),
        Expr::Index { base, indices, .. } => {
            reads(base, out);
            indices.iter().for_each(|x| reads(x, out));
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => reads(operand, out),
        Expr::Binary { lhs, rhs, .. } => {
            reads(lhs, out);
            reads(rhs, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|x| reads(x, out)),
        _ => {}
    }
}

fn calls(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Call { callee, args, .. } => callee == name || args.iter().any(|a| calls(a, name)),
        Expr::Index { base, indices, .. } => calls(base, name) || indices.iter().any(|a| calls(a, name)),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => calls(operand, name),
        Expr::Binary { lhs, rhs, .. } => calls(lhs, name) || calls(rhs, name),
        _ => false,
    }
}

fn has_index_of(e: &Expr, pred: &dyn Fn(&str) -> bool) -> bool {
    match e {
        Expr::Index { base, indices, .. } => matches!(&**base, Expr::Var { name, .. } if pred(name)) || indices.iter().any(|a| has_index_of(a, pred)),
        Expr::Call { args, .. } => args.iter().any(|a| has_index_of(a, pred)),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => has_index_of(operand, pred),
        Expr::Binary { lhs, rhs, .. } => has_index_of(lhs, pred) || has_index_of(rhs, pred),
        _ => false,
    }
}

fn assigned_in(s: &Stmt, out: &mut Vec<String>) {
    match s {
        Stmt::Assign { name, .. } => out.push(name.clone()),
        Stmt::If { then_body, else_body, .. } => then_body.iter().chain(else_body).for_each(|t| assigned_in(t, out)),
        Stmt::While { body, .. } => body.iter().for_each(|t| assigned_in(t, out)),
        Stmt::For { var, body, .. } => {
            out.push(var.clone());
            body.iter().for_each(|t| assigned_in(t, out));
        }
        _ => {}
    }
}

fn declared_name(s: &Stmt) -> Option<&str> {
    match s {
        Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } if callee == DECLARE => match args.first() {
            Some(Expr::Str { value, .. }) => Some(value),
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------- schedules

/// A kernel's `schedule` line: the grid it gives, each key one value per
/// axis (x, then y for a 2-D grid). `cover` is the index space the threads
/// must cover, from which the blocks are derived; it and `blocks` exclude
/// each other.
#[derive(Clone)]
struct Sched {
    blocks: Option<Vec<Expr>>,
    threads: Option<Vec<Expr>>,
    cover: Option<Vec<Expr>>,
    span: Span,
}

impl Sched {
    /// A 2-D grid: some key gives a y.
    fn two_d(&self) -> bool {
        [&self.blocks, &self.threads, &self.cover].iter().any(|v| v.as_ref().is_some_and(|v| v.len() == 2))
    }
}

/// The schedule of a kernel, if it has a 2-D grid.
fn sched_two_d(sched: Option<&Sched>) -> bool {
    sched.is_some_and(Sched::two_d)
}

fn sched_of(s: &Stmt) -> Option<(&[Expr], Span)> {
    match s {
        Stmt::ExprStmt { expr: Expr::Call { callee, args, span }, .. } if callee == SCHEDULE => Some((args, *span)),
        _ => None,
    }
}

/// The program's schedule lines by kernel name, checked against the
/// kernels they name.
fn schedules(p: &Program) -> Result<HashMap<String, Sched>, Diagnostic> {
    let kernels: HashMap<&str, &Def> = p.items.iter().filter_map(|i| match i {
        Item::Def(d) if is_kernel(d) => Some((d.name.as_str(), d)),
        _ => None,
    }).collect();
    let defs: HashSet<&str> = p.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.as_str()) } else { None }).collect();
    let consts = constants(p);
    let mut out: HashMap<String, Sched> = HashMap::new();
    for item in &p.items {
        let Item::Stmt(st) = item else { continue };
        let Some((args, span)) = sched_of(st) else { continue };
        let Some(Expr::Str { value: name, span: name_span }) = args.first() else { unreachable!("the parser writes the name first") };
        let Some(d) = kernels.get(name.as_str()) else {
            let why = if defs.contains(name.as_str()) { format!("`{name}` is a def, not a `kernel def`: a schedule line gives a kernel's grid") } else { format!("`schedule {name}(...)` names no kernel of this program") };
            return Err(err(*name_span, why, "not a kernel"));
        };
        if elementwise_body(d).is_some() {
            return Err(err(*name_span, format!("`{name}` is an elementwise kernel: one thread per element, {ELEMENTWISE_THREADS} to a block; a schedule line gives the grid of an explicit kernel"), "elementwise"));
        }
        if out.contains_key(name) {
            return Err(err(*name_span, format!("a second schedule line for `{name}`"), "again here"));
        }
        let k = kern_of(d)?;
        let mut sc = Sched { blocks: None, threads: None, cover: None, span };
        for pair in args[1..].chunks(2) {
            let (Expr::Str { value: key, span: key_span }, value) = (&pair[0], &pair[1]) else { unreachable!("the parser writes key, value") };
            let slot = match key.as_str() {
                "blocks" => &mut sc.blocks,
                "threads" => &mut sc.threads,
                "cover" => &mut sc.cover,
                _ => return Err(err(*key_span, format!("a kernel's schedule takes {}; `{key}` is not one of them", GRID_KEYS.map(|k| format!("`{k}`")).join(", ")), "unknown key")),
            };
            if slot.is_some() {
                return Err(err(*key_span, format!("`{key}` is given twice"), "again here"));
            }
            // one value, or one per axis: `[x, y]`
            let axes = match value {
                Expr::ArrLit { elems, .. } if (1..=2).contains(&elems.len()) => elems.clone(),
                Expr::ArrLit { span, .. } => return Err(err(*span, format!("`{key}` has one value per axis, x then y: `{key} = [x, y]`; grids have two axes here"), "not one or two values")),
                _ => vec![value.clone()],
            };
            for a in &axes {
                check_grid_expr(a, &k, &consts)?;
            }
            *slot = Some(axes);
        }
        if sc.blocks.is_some() && sc.cover.is_some() {
            return Err(err(span, format!("`schedule {name}` gives both `blocks` and `cover`: the blocks are derived from `cover`, so give one of them"), "both"));
        }
        out.insert(name.clone(), sc);
    }
    Ok(out)
}

/// A schedule's value: what the emulator and the CUDA launcher can both
/// compute before the launch.
fn check_grid_expr(e: &Expr, k: &Kern, consts: &HashMap<String, (f64, bool)>) -> Result<(), Diagnostic> {
    let bad = |sp: Span| err(sp, format!("a grid size is an integer expression of literals, `{}`'s Int parameters, `len` of its array parameters and global constants", k.name), "not one of those");
    match e {
        Expr::Num { .. } => Ok(()),
        Expr::Var { name, span } => match k.ty(name) {
            Some(Ty::Int) => Ok(()),
            Some(_) => Err(bad(*span)),
            None if consts.contains_key(name) => Ok(()),
            None => Err(bad(*span)),
        },
        Expr::Call { callee, args, span } if callee == "len" => match args.as_slice() {
            [Expr::Var { name, .. }] if matches!(k.ty(name), Some(Ty::Arr | Ty::IntArr)) => Ok(()),
            _ => Err(bad(*span)),
        },
        Expr::Binary { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv, lhs, rhs, .. } => {
            check_grid_expr(lhs, k, consts)?;
            check_grid_expr(rhs, k, consts)
        }
        _ => Err(bad(e.span())),
    }
}

/// The first array parameter: the one the default grid covers.
fn sized_by(k: &Kern) -> Option<&str> {
    k.params.iter().find(|(_, t)| matches!(t, Ty::Arr | Ty::IntArr)).map(|(n, _)| n.as_str())
}

// ---------------------------------------------------------------- the rewrite

/// Rewrites every `kernel def` of `program` into the def the engines run.
pub fn lower(mut program: Program, src: &str) -> Result<Program, Vec<Diagnostic>> {
    let any_schedule = program.items.iter().any(|i| matches!(i, Item::Stmt(st) if sched_of(st).is_some()));
    if RAW.with(|r| r.get()) || (!has_kernels(&program) && !any_schedule && !program_has_regions(&program)) {
        return Ok(program);
    }
    let kinds: HashMap<String, KInfo> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Def(d) if is_kernel(d) => kinfo_of(d).map(|k| (d.name.clone(), k)),
            _ => None,
        })
        .collect();
    let scheds = schedules(&program).map_err(|e| vec![e])?;
    program.items.retain(|i| !matches!(i, Item::Stmt(st) if sched_of(st).is_some()));
    let defs: HashSet<String> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    let mut used = HashSet::new();
    for item in &program.items {
        match item {
            Item::Def(d) => {
                used.insert((d.span.start, d.span.end));
                used.insert((d.name_span.start, d.name_span.end));
                d.params.iter().for_each(|p| {
                    used.insert((p.span.start, p.span.end));
                });
                d.body.iter().for_each(|s| spans_of_stmt(s, &mut used));
            }
            Item::Stmt(s) => spans_of_stmt(s, &mut used),
            _ => {}
        }
    }
    SPANS.with(|c| *c.borrow_mut() = Spans { src: src.to_string(), used, next_end: HashMap::new() });
    WHOLE.with(|w| w.borrow_mut().clear());
    let mut errs = Vec::new();
    for item in &mut program.items {
        if let Item::Def(d) = item {
            if is_kernel(d) {
                match lower_def(d, &defs, scheds.get(&d.name)) {
                    Ok(mut nd) => {
                        if let Some(k) = kinds.get(&d.name) {
                            let mut body = count_launch(k, d.name_span);
                            body.append(&mut nd.body);
                            nd.body = body;
                        }
                        *d = nd
                    }
                    Err(e) => errs.push(e),
                }
            } else {
                d.body = lower_regions(&d.body, &kinds, &defs, false, &mut errs);
            }
        }
    }
    // top-level `on device` blocks, and the counters every kernel and block
    // adds to
    let near = program.items.iter().find_map(|i| if let Item::Def(d) = i { Some(d.name_span) } else { None }).or_else(|| program.items.iter().find_map(|i| if let Item::Stmt(s) = i { Some(s.span()) } else { None }));
    let mut items = Vec::new();
    if let Some(near) = near {
        items.extend(tmpl(&format!("{XFER} = zeros(4)"), near).into_iter().map(Item::Stmt));
        if !defs.contains(TRANSFERS) {
            items.push(Item::Def(tmpl_def(&format!("def {TRANSFERS}() -> Arr {{\n    return {XFER}[0:3]\n}}"), near)));
        }
    }
    for item in std::mem::take(&mut program.items) {
        match item {
            Item::Stmt(st) => items.extend(lower_regions(std::slice::from_ref(&st), &kinds, &defs, false, &mut errs).into_iter().map(Item::Stmt)),
            other => items.push(other),
        }
    }
    program.items = items;
    WHOLE.with(|w| program.whole.extend(w.borrow_mut().drain(..)));
    if errs.is_empty() {
        Ok(program)
    } else {
        Err(errs)
    }
}

fn lower_def(d: &Def, defs: &HashSet<String>, sched: Option<&Sched>) -> Result<Def, Diagnostic> {
    let k = kern_of(d)?;
    if let Some(tree) = elementwise_body(d) {
        check_tree(tree, &k)?;
        return Ok(lower_elementwise(d, &k, tree));
    }
    for s in &d.body[1..] {
        check_stmt(s, &k, defs, true)?;
    }
    if k.written.is_empty() {
        return Err(err(d.name_span, format!("the kernel `{}` writes none of its array parameters, so it has no result", d.name), "writes nothing"));
    }
    Explicit::new(&k, d)?.lower(d, sched)
}

fn lower_elementwise(d: &Def, k: &Kern, tree: &Expr) -> Def {
    let sp = d.name_span;
    let mut arrays = Vec::new();
    tree_arrays(tree, k, &mut arrays);
    let name = &d.name;
    let mut text = String::new();
    match arrays.first() {
        None => text.push_str(&format!("stop(\"elementwise kernel {name}: its return reads no array\")\n")),
        Some(first) => {
            text.push_str(&format!("{P}n = len({first})\n"));
            for a in &arrays[1..] {
                text.push_str(&format!(
                    "if len({a}) != {P}n {{\n    stop(f\"elementwise kernel {name}: `{a}` has {{len({a})}} elements and `{first}` has {{{P}n}}\")\n}}\n"
                ));
            }
        }
    }
    let t = ELEMENTWISE_THREADS;
    text.push_str(&format!(
        "{P}out = zeros({P}n)\n{P}blocks = ({P}n + {t} - 1) // {t}\nfor {P}b to {P}blocks {{\n    for {P}t to {t} {{\n        {P}i = {P}b * {t} + {P}t\n        if {P}i < {P}n {{\n            {P}out[{P}i] = 0\n        }}\n    }}\n}}\nreturn {P}out\n"
    ));
    let mut body = tmpl(&text, sp);
    // the element's value: the tree with each array read at the thread's index
    fn at_i(e: &Expr, arrays: &[String]) -> Expr {
        let sp = e.span();
        match e {
            Expr::Var { name, .. } if arrays.contains(name) => Expr::Index { base: Box::new(var(name, sp)), indices: vec![var(&format!("{P}i"), sp)], span: fresh_near(sp) },
            Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(at_i(operand, arrays)), span: *span },
            Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op: *op, lhs: Box::new(at_i(lhs, arrays)), rhs: Box::new(at_i(rhs, arrays)), span: *span },
            Expr::Call { callee, args, span } => Expr::Call { callee: callee.clone(), args: args.iter().map(|a| at_i(a, arrays)).collect(), span: *span },
            _ => e.clone(),
        }
    }
    let value = at_i(tree, &arrays);
    fn place(body: &mut [Stmt], value: &Expr) -> bool {
        for s in body {
            match s {
                Stmt::SetIndex { value: v, .. } => {
                    *v = value.clone();
                    return true;
                }
                Stmt::If { then_body: b, .. } | Stmt::For { body: b, .. } => {
                    if place(b, value) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    place(&mut body, &value);
    Def { name: d.name.clone(), name_span: d.name_span, params: d.params.clone(), ret: d.ret.clone(), body, span: d.span }
}

/// How the rewrite sees one statement at the level of the block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Barrier,
    Shared,
    /// an `if`, `while` or `for` with a barrier inside
    Structural,
    /// every thread computes it alike: run once per block
    Uniform,
    /// run by each thread
    Thread,
}

struct Explicit<'a> {
    k: &'a Kern,
    /// variables whose value can differ between the threads of a block
    varying: HashSet<String>,
    /// per-thread variables computed again after each barrier, with their
    /// definitions in order
    remat: Vec<(String, Expr)>,
    /// every name the body assigns (any other name it reads is a global)
    assigned: HashSet<String>,
    /// counter for the rewrite's temporaries
    n: Cell<usize>,
}

impl<'a> Explicit<'a> {
    fn new(k: &'a Kern, d: &Def) -> Result<Self, Diagnostic> {
        let body = &d.body[1..];
        let mut assigned = Vec::new();
        body.iter().for_each(|s| assigned_in(s, &mut assigned));
        let mut x = Explicit { k, varying: HashSet::new(), remat: Vec::new(), assigned: assigned.into_iter().collect(), n: Cell::new(0) };
        // the varying variables: a fixed point, since a statement's kind
        // depends on the variables it reads
        loop {
            let mut more = Vec::new();
            x.collect_varying(body, &mut more);
            let before = x.varying.len();
            x.varying.extend(more);
            if x.varying.len() == before {
                break;
            }
        }
        x.check_structure(body)?;
        x.find_remat(body);
        x.check_crossings(body)?;
        Ok(x)
    }

    fn uniform(&self, e: &Expr) -> bool {
        let mut rs = Vec::new();
        reads(e, &mut rs);
        !calls(e, THREAD_ID) && !rs.iter().any(|(n, _)| self.varying.contains(n)) && !has_index_of(e, &|a| self.k.tracked(a))
    }

    fn kind(&self, s: &Stmt) -> Kind {
        if is_barrier(s) {
            return Kind::Barrier;
        }
        if shared_decl(s).is_some() {
            return Kind::Shared;
        }
        if holds_barrier(s) {
            return Kind::Structural;
        }
        match s {
            Stmt::Assign { name, value, .. } if !self.varying.contains(name) && self.uniform(value) => Kind::Uniform,
            Stmt::ExprStmt { .. } if declared_name(s).is_some_and(|n| !self.varying.contains(n)) => Kind::Uniform,
            _ => Kind::Thread,
        }
    }

    fn collect_varying(&self, body: &[Stmt], out: &mut Vec<String>) {
        for s in body {
            match self.kind(s) {
                Kind::Thread => assigned_in(s, out),
                Kind::Structural => match s {
                    Stmt::If { then_body, else_body, .. } => {
                        self.collect_varying(then_body, out);
                        self.collect_varying(else_body, out);
                    }
                    Stmt::While { body, .. } | Stmt::For { body, .. } => self.collect_varying(body, out),
                    _ => {}
                },
                _ => {}
            }
        }
    }

    /// A barrier only under conditions every thread of the block evaluates
    /// alike, and no `break` or `continue` beside one.
    fn check_structure(&self, body: &[Stmt]) -> Result<(), Diagnostic> {
        for s in body {
            if self.kind(s) != Kind::Structural {
                continue;
            }
            let first_barrier = |b: &[Stmt]| -> Span {
                fn find(b: &[Stmt]) -> Option<Span> {
                    b.iter().find_map(|s| {
                        if is_barrier(s) {
                            return Some(s.span());
                        }
                        match s {
                            Stmt::If { then_body, else_body, .. } => find(then_body).or_else(|| find(else_body)),
                            Stmt::While { body, .. } | Stmt::For { body, .. } => find(body),
                            _ => None,
                        }
                    })
                }
                find(b).unwrap_or(s.span())
            };
            let (cond, inner, what): (&Expr, Vec<&[Stmt]>, &str) = match s {
                Stmt::If { cond, then_body, else_body, .. } => (cond, vec![then_body, else_body], "an `if`"),
                Stmt::While { cond, body, .. } => (cond, vec![body], "a `while`"),
                Stmt::For { bound, body, .. } => (bound, vec![body], "a `for`"),
                _ => unreachable!(),
            };
            if !self.uniform(cond) {
                let at = inner.iter().map(|b| first_barrier(b)).next().unwrap_or(s.span());
                return Err(Diagnostic::error("`barrier()` under a condition that can differ between the threads of a block: the threads that skip it would never reach it")
                    .with_primary(at, "this barrier")
                    .with_secondary(cond.span(), format!("{what} whose condition reads `thread_id()`, an array, or a per-thread variable"))
                    .with_note("move the barrier out of the branch, or make the condition the same for every thread"));
            }
            if let Stmt::For { var, var_span, .. } = s {
                if self.varying.contains(var) {
                    return Err(err(*var_span, format!("`{var}` counts a loop that holds a barrier, so it must not be assigned by a thread elsewhere"), "here"));
                }
            }
            for b in inner {
                for t in b.iter() {
                    if matches!(t, Stmt::Break { .. } | Stmt::Continue { .. }) && !matches!(s, Stmt::If { .. }) {
                        return Err(err(t.span(), "`break` and `continue` cannot leave a loop that holds a barrier", "here"));
                    }
                }
                self.check_structure(b)?;
            }
        }
        Ok(())
    }

    /// Per-thread variables that are computed again after each barrier:
    /// assigned once, at the top level, from the thread and grid builtins,
    /// parameters, constants and others of their kind.
    fn find_remat(&mut self, body: &[Stmt]) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for s in body {
            let mut a = Vec::new();
            assigned_in(s, &mut a);
            for n in a {
                *counts.entry(n).or_default() += 1;
            }
        }
        // uniform variables assigned once at the top level, before any barrier loop
        let mut fixed: HashSet<String> = HashSet::new();
        for s in body {
            if let Stmt::Assign { name, value, .. } = s {
                if counts.get(name) != Some(&1) {
                    continue;
                }
                let ok = self.remat_safe(value, &fixed);
                if ok && !self.varying.contains(name) {
                    fixed.insert(name.clone());
                } else if ok {
                    fixed.insert(name.clone());
                    self.remat.push((name.clone(), value.clone()));
                }
            }
        }
    }

    fn remat_safe(&self, e: &Expr, known: &HashSet<String>) -> bool {
        match e {
            Expr::Num { .. } | Expr::Bool { .. } => true,
            Expr::Var { name, .. } => self.k.ty(name).is_some() || known.contains(name) || !self.assigned.contains(name),
            Expr::Call { callee, args, .. } => {
                ([THREAD_ID, BLOCK_ID, BLOCK_SIZE, GRID_SIZE, "len"].contains(&callee.as_str()) || MATH.iter().any(|(n, _)| n == callee) || callee == "pow")
                    && args.iter().all(|a| matches!(a, Expr::Var { name, .. } if self.k.is_array(name)) || self.remat_safe(a, known))
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.remat_safe(operand, known),
            Expr::Binary { lhs, rhs, .. } => self.remat_safe(lhs, known) && self.remat_safe(rhs, known),
            _ => false,
        }
    }

    fn is_remat(&self, v: &str) -> bool {
        self.remat.iter().any(|(n, _)| n == v)
    }

    /// A per-thread value read in a piece it was not set in, and not computed
    /// again, would have to survive a barrier: refused.
    fn check_crossings(&self, body: &[Stmt]) -> Result<(), Diagnostic> {
        let mut pieces: Vec<Vec<&Stmt>> = Vec::new();
        self.pieces(body, &mut pieces);
        for piece in pieces {
            let mut defined: HashSet<String> = HashSet::new();
            for s in piece {
                self.walk_defined(s, &mut defined)?;
            }
        }
        Ok(())
    }

    fn pieces<'s>(&self, body: &'s [Stmt], out: &mut Vec<Vec<&'s Stmt>>) {
        let mut cur: Vec<&Stmt> = Vec::new();
        for s in body {
            match self.kind(s) {
                Kind::Thread => cur.push(s),
                Kind::Structural => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    match s {
                        Stmt::If { then_body, else_body, .. } => {
                            self.pieces(then_body, out);
                            self.pieces(else_body, out);
                        }
                        Stmt::While { body, .. } | Stmt::For { body, .. } => self.pieces(body, out),
                        _ => {}
                    }
                }
                _ => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }

    fn check_reads(&self, e: &Expr, defined: &HashSet<String>) -> Result<(), Diagnostic> {
        let mut rs = Vec::new();
        reads(e, &mut rs);
        for (n, sp) in rs {
            if self.varying.contains(&n) && !defined.contains(&n) && !self.is_remat(&n) {
                return Err(err(sp, format!("`{n}` is a per-thread value set before a barrier (or before a statement every thread runs alike) and read after it"), "read here")
                    .with_note("a thread's variables do not survive `barrier()` in the emulator or in the CUDA it mirrors: write the value to a shared array before the barrier and read it back after"));
            }
        }
        Ok(())
    }

    fn walk_defined(&self, s: &Stmt, defined: &mut HashSet<String>) -> Result<(), Diagnostic> {
        match s {
            Stmt::Assign { name, value, .. } => {
                self.check_reads(value, defined)?;
                defined.insert(name.clone());
            }
            Stmt::SetIndex { indices, value, .. } => {
                indices.iter().try_for_each(|i| self.check_reads(i, defined))?;
                self.check_reads(value, defined)?;
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.check_reads(cond, defined)?;
                let mut a = defined.clone();
                for t in then_body {
                    self.walk_defined(t, &mut a)?;
                }
                let mut b = defined.clone();
                for t in else_body {
                    self.walk_defined(t, &mut b)?;
                }
                defined.extend(a);
                defined.extend(b);
            }
            Stmt::While { cond, body, .. } => {
                // a loop's body may read what an earlier pass of it set
                let mut a = defined.clone();
                for t in body {
                    assigned_in(t, &mut Vec::new());
                    let mut v = Vec::new();
                    assigned_in(t, &mut v);
                    a.extend(v);
                }
                self.check_reads(cond, &a)?;
                for t in body {
                    self.walk_defined(t, &mut a)?;
                }
                defined.extend(a);
            }
            Stmt::For { var, bound, body, .. } => {
                self.check_reads(bound, defined)?;
                let mut a = defined.clone();
                a.insert(var.clone());
                for t in body {
                    let mut v = Vec::new();
                    assigned_in(t, &mut v);
                    a.extend(v);
                }
                for t in body {
                    self.walk_defined(t, &mut a)?;
                }
                defined.extend(a);
            }
            _ => {}
        }
        Ok(())
    }

    fn tmp(&self, what: &str) -> String {
        let n = self.n.get();
        self.n.set(n + 1);
        format!("{P}{what}{n}")
    }

    fn lower(&self, d: &Def, sched: Option<&Sched>) -> Result<Def, Diagnostic> {
        let before = TWO_D.with(|c| c.replace(sched_two_d(sched)));
        let r = self.lower_grid(d, sched);
        TWO_D.with(|c| c.set(before));
        r
    }

    fn lower_grid(&self, d: &Def, sched: Option<&Sched>) -> Result<Def, Diagnostic> {
        let sp = d.name_span;
        let name = &self.k.name;
        // the grid: the schedule line's; by default 256 threads to a block and
        // the blocks that cover the first array parameter. Each axis is set
        // from the schedule's own expression, in place of the template's value.
        let sc = sched.cloned().unwrap_or(Sched { blocks: None, threads: None, cover: None, span: sp });
        let t = ELEMENTWISE_THREADS;
        let mut grid = format!("{P}tx: Int = {t}\n{P}ty: Int = 1\n");
        let mut values: Vec<Option<Expr>> = vec![sc.threads.as_ref().map(|v| v[0].clone()), sc.threads.as_ref().and_then(|v| v.get(1).cloned())];
        match &sc.blocks {
            Some(b) => {
                grid.push_str(&format!("{P}bx: Int = 1\n{P}by: Int = 1\n"));
                values.push(Some(b[0].clone()));
                values.push(b.get(1).cloned());
            }
            None => {
                let cover_x = match &sc.cover {
                    Some(c) => {
                        values.push(Some(c[0].clone()));
                        values.push(c.get(1).cloned());
                        "1".to_string()
                    }
                    None => {
                        let Some(first) = sized_by(self.k) else {
                            return Err(err(d.name_span, format!("the kernel `{name}` has no array parameter to size its grid from; give the grid in a top-level line, `schedule {name}(blocks = ..., threads = ...)` or `schedule {name}(cover = ...)`"), "no array parameter"));
                        };
                        values.push(None);
                        values.push(None);
                        format!("len({first})")
                    }
                };
                grid.push_str(&format!("{P}cx: Int = {cover_x}\n{P}cy: Int = 1\n{P}bx: Int = ({P}cx + {P}tx - 1) // {P}tx\n{P}by: Int = ({P}cy + {P}ty - 1) // {P}ty\n"));
            }
        }
        grid.push_str(&format!("{P}threads: Int = {P}tx * {P}ty\n{P}blocks: Int = {P}bx * {P}by\n"));
        let mut head = tmpl(&grid, sc.span);
        let assigns: Vec<usize> = head.iter().enumerate().filter(|(_, s)| matches!(s, Stmt::Assign { .. })).map(|(i, _)| i).collect();
        for (slot, v) in values.iter().enumerate() {
            if let (Some(v), Stmt::Assign { value, .. }) = (v, &mut head[assigns[slot]]) {
                *value = v.clone();
            }
        }
        if sc.two_d() {
            head.extend(tmpl(&format!("if {P}tx < 1 or {P}ty < 1 or {P}bx < 1 or {P}by < 1 {{\n    stop(f\"kernel {name}: a launch needs one block and one thread at least along each axis (got {{{P}bx}} x {{{P}by}} blocks of {{{P}tx}} x {{{P}ty}} threads)\")\n}}\nif {P}by > {MAX_BLOCKS_Y} {{\n    stop(f\"kernel {name}: {{{P}by}} blocks along y; CUDA allows {MAX_BLOCKS_Y}\")\n}}\n"), sp));
        }
        let mut text = format!(
            "if {P}blocks < 1 or {P}threads < 1 {{\n    stop(f\"kernel {name}: a launch needs one block and one thread at least (got {{{P}blocks}} blocks of {{{P}threads}} threads)\")\n}}\nif {P}threads > {MAX_THREADS} {{\n    stop(f\"kernel {name}: {{{P}threads}} threads per block; CUDA allows {MAX_THREADS}\")\n}}\n"
        );
        for g in &self.k.written {
            for s in ["gw", "gp", "gr", "grp"] {
                text.push_str(&format!("{P}{s}_{g} = zeros(len({g}))\n"));
            }
        }
        let mut body = head;
        body.extend(tmpl(&text, sp));
        let mut block = tmpl(&format!("{P}phase: Int = 0"), sp);
        block.extend(self.region(&d.body[1..])?);
        body.push(Stmt::For { var: format!("{P}b"), var_span: fresh_near(sp), bound: var(&format!("{P}blocks"), sp), body: block, parallel: false, span: fresh_near(sp) });
        let results: Vec<Expr> = self.k.written.iter().map(|g| var(g, sp)).collect();
        let value = if results.len() == 1 { results.into_iter().next().unwrap() } else { Expr::Call { callee: crate::tuples::TUPLE.into(), args: results, span: fresh_near(sp) } };
        body.push(Stmt::Return { value, span: fresh_near(sp) });
        Ok(Def { name: d.name.clone(), name_span: d.name_span, params: d.params.clone(), ret: None, body, span: d.span })
    }

    fn region(&self, body: &[Stmt]) -> Result<Vec<Stmt>, Diagnostic> {
        let mut out = Vec::new();
        let mut piece: Vec<&Stmt> = Vec::new();
        for s in body {
            let kind = self.kind(s);
            if kind != Kind::Thread && !piece.is_empty() {
                out.push(self.thread_loop(&std::mem::take(&mut piece))?);
            }
            match kind {
                Kind::Thread => piece.push(s),
                Kind::Barrier => out.extend(tmp_phase(s.span())),
                Kind::Shared => {
                    let (n, size) = shared_decl(s).unwrap();
                    let size = match size {
                        Expr::Num { value, .. } => format!("{}", *value as i64),
                        Expr::Var { name, .. } => name.clone(),
                        _ => unreachable!("checked"),
                    };
                    out.extend(tmpl(&format!("{n} = zeros({size})\n{P}sw_{n} = zeros({size})\n{P}sr_{n} = zeros({size})"), s.span()));
                }
                Kind::Uniform => out.push(subst_stmt(s)),
                Kind::Structural => out.push(match s {
                    Stmt::If { cond, then_body, else_body, span } => Stmt::If { cond: subst_expr(cond), then_body: self.region(then_body)?, else_body: self.region(else_body)?, span: *span },
                    Stmt::While { cond, body, span } => Stmt::While { cond: subst_expr(cond), body: self.region(body)?, span: *span },
                    Stmt::For { var, var_span, bound, body, span, .. } => Stmt::For { var: var.clone(), var_span: *var_span, bound: subst_expr(bound), body: self.region(body)?, parallel: false, span: *span },
                    _ => unreachable!(),
                }),
            }
        }
        if !piece.is_empty() {
            out.push(self.thread_loop(&piece)?);
        }
        Ok(out)
    }

    fn thread_loop(&self, piece: &[&Stmt]) -> Result<Stmt, Diagnostic> {
        let sp = piece[0].span();
        // the per-thread values this piece reads and does not set: computed again
        let mut need: HashSet<String> = HashSet::new();
        for s in piece {
            let mut rs = Vec::new();
            stmt_reads(s, &mut rs);
            need.extend(rs.into_iter().map(|(n, _)| n).filter(|n| self.is_remat(n)));
        }
        // and what those read, transitively
        loop {
            let before = need.len();
            for (n, e) in &self.remat {
                if need.contains(n) {
                    let mut rs = Vec::new();
                    reads(e, &mut rs);
                    need.extend(rs.into_iter().map(|(n, _)| n).filter(|n| self.is_remat(n)));
                }
            }
            if need.len() == before {
                break;
            }
        }
        let mut body = Vec::new();
        for (n, e) in &self.remat {
            // a piece that assigns it itself (the first) needs no copy
            if need.contains(n) && !piece.iter().any(|s| matches!(s, Stmt::Assign { name, .. } if name == n)) {
                body.push(Stmt::Assign { name: n.clone(), name_span: fresh_near(sp), value: subst_expr(e), span: fresh_near(sp) });
            }
        }
        for s in piece {
            body.extend(self.instr_stmt(s));
        }
        Ok(Stmt::For { var: format!("{P}t"), var_span: fresh_near(sp), bound: var(&format!("{P}threads"), sp), body, parallel: false, span: fresh_near(sp) })
    }

    // ------------------------------------------------ instrumentation

    fn instr_stmt(&self, s: &Stmt) -> Vec<Stmt> {
        let sp = s.span();
        match s {
            Stmt::Assign { name, name_span, value, span } => {
                let (mut pre, v) = self.instr_expr(value, sp);
                pre.push(Stmt::Assign { name: name.clone(), name_span: *name_span, value: v, span: *span });
                pre
            }
            Stmt::SetIndex { base, indices, value, span } => {
                let a = match &**base {
                    Expr::Var { name, .. } => name.clone(),
                    _ => unreachable!("checked"),
                };
                let (mut pre, i) = self.instr_expr(&indices[0], sp);
                let (pre2, v) = self.instr_expr(value, sp);
                pre.extend(pre2);
                let iv = self.tmp("i");
                pre.push(Stmt::Assign { name: iv.clone(), name_span: fresh_near(sp), value: i, span: fresh_near(sp) });
                pre.extend(self.bounds(&a, &iv, sp));
                if self.k.tracked(&a) {
                    pre.extend(self.on_write(&a, &iv, sp));
                }
                pre.push(Stmt::SetIndex { base: base.clone(), indices: vec![var(&iv, sp)], value: v, span: *span });
                pre
            }
            Stmt::If { cond, then_body, else_body, span } => {
                let (mut pre, c) = self.instr_expr(cond, sp);
                pre.push(Stmt::If { cond: c, then_body: then_body.iter().flat_map(|t| self.instr_stmt(t)).collect(), else_body: else_body.iter().flat_map(|t| self.instr_stmt(t)).collect(), span: *span });
                pre
            }
            Stmt::While { cond, body, span } => {
                let (pre, c) = self.instr_expr(cond, sp);
                let inner: Vec<Stmt> = body.iter().flat_map(|t| self.instr_stmt(t)).collect();
                if pre.is_empty() {
                    return vec![Stmt::While { cond: c, body: inner, span: *span }];
                }
                let mut b = pre;
                b.push(Stmt::If { cond: Expr::Not { operand: Box::new(c), span: fresh_near(sp) }, then_body: vec![Stmt::Break { span: fresh_near(sp) }], else_body: vec![], span: fresh_near(sp) });
                b.extend(inner);
                vec![Stmt::While { cond: Expr::Bool { value: true, span: fresh_near(sp) }, body: b, span: *span }]
            }
            Stmt::For { var, var_span, bound, body, span, .. } => {
                let (mut pre, b) = self.instr_expr(bound, sp);
                pre.push(Stmt::For { var: var.clone(), var_span: *var_span, bound: b, body: body.iter().flat_map(|t| self.instr_stmt(t)).collect(), parallel: false, span: *span });
                pre
            }
            _ => vec![s.clone()],
        }
    }

    /// The expression with each array read hoisted into a checked temporary.
    fn instr_expr(&self, e: &Expr, sp: Span) -> (Vec<Stmt>, Expr) {
        match e {
            Expr::Index { base, indices, span } => {
                let a = match &**base {
                    Expr::Var { name, .. } => name.clone(),
                    _ => unreachable!("checked"),
                };
                let (mut pre, i) = self.instr_expr(&indices[0], sp);
                let iv = self.tmp("i");
                pre.push(Stmt::Assign { name: iv.clone(), name_span: fresh_near(sp), value: i, span: fresh_near(sp) });
                pre.extend(self.bounds(&a, &iv, sp));
                if self.k.tracked(&a) {
                    pre.extend(self.on_read(&a, &iv, sp));
                }
                (pre, Expr::Index { base: base.clone(), indices: vec![var(&iv, sp)], span: *span })
            }
            Expr::Binary { op: op @ (BinOp::And | BinOp::Or), lhs, rhs, span } => {
                let (mut pl, l) = self.instr_expr(lhs, sp);
                let (pr, r) = self.instr_expr(rhs, sp);
                if pr.is_empty() {
                    return (pl, Expr::Binary { op: *op, lhs: Box::new(l), rhs: Box::new(r), span: *span });
                }
                // keep the short circuit: the right side's reads run only when it is evaluated
                let c = self.tmp("c");
                pl.push(Stmt::Assign { name: c.clone(), name_span: fresh_near(sp), value: l, span: fresh_near(sp) });
                let test = if *op == BinOp::And { var(&c, sp) } else { Expr::Not { operand: Box::new(var(&c, sp)), span: fresh_near(sp) } };
                let mut then_body = pr;
                then_body.push(Stmt::Assign { name: c.clone(), name_span: fresh_near(sp), value: r, span: fresh_near(sp) });
                pl.push(Stmt::If { cond: test, then_body, else_body: vec![], span: fresh_near(sp) });
                (pl, var(&c, sp))
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let (mut pl, l) = self.instr_expr(lhs, sp);
                let (pr, r) = self.instr_expr(rhs, sp);
                pl.extend(pr);
                (pl, Expr::Binary { op: *op, lhs: Box::new(l), rhs: Box::new(r), span: *span })
            }
            Expr::Neg { operand, span } => {
                let (p, o) = self.instr_expr(operand, sp);
                (p, Expr::Neg { operand: Box::new(o), span: *span })
            }
            Expr::Not { operand, span } => {
                let (p, o) = self.instr_expr(operand, sp);
                (p, Expr::Not { operand: Box::new(o), span: *span })
            }
            Expr::Call { callee, args, span } if !args.is_empty() && !([THREAD_ID, BLOCK_ID, BLOCK_SIZE, GRID_SIZE].contains(&callee.as_str()) && axis_of(args).is_some()) => {
                let mut pre = Vec::new();
                let mut out = Vec::new();
                for a in args {
                    let (p, x) = self.instr_expr(a, sp);
                    pre.extend(p);
                    out.push(x);
                }
                (pre, Expr::Call { callee: callee.clone(), args: out, span: *span })
            }
            _ => (Vec::new(), subst_expr(e)),
        }
    }

    fn who(&self) -> String {
        format!("kernel {}, block {{{P}b}}, thread {{{P}t}}", self.k.name)
    }

    fn bounds(&self, a: &str, i: &str, sp: Span) -> Vec<Stmt> {
        let who = self.who();
        tmpl(&format!("if {i} != floor({i}) {{\n    stop(f\"{who}: index {{{i}}} of `{a}` is not a whole number\")\n}}\nif {i} < 0 or {i} >= len({a}) {{\n    stop(f\"{who}: index {{{i}}} is outside `{a}` (length {{len({a})}})\")\n}}"), sp)
    }

    fn on_read(&self, a: &str, i: &str, sp: Span) -> Vec<Stmt> {
        let who = self.who();
        let text = if self.k.is_shared(a) {
            format!(
                r#"{P}w = {P}sw_{a}[{i}]
if {P}w == 0 {{
    stop(f"{who}: reads {a}[{{{i}}}], which no thread of the block has written yet (a barrier() missing between the write and this read?)")
}}
{P}wp = floor(({P}w - 1) / {P}threads)
{P}wt = {P}w - 1 - {P}wp * {P}threads
if {P}wp == {P}phase and {P}wt != {P}t {{
    stop(f"{who}: reads {a}[{{{i}}}], which thread {{{P}wt}} wrote with no barrier() between them")
}}
{P}me = 1.0 * ({P}phase * {P}threads + {P}t + 1)
{P}r = {P}sr_{a}[{i}]
if {P}r == 0 {{
    {P}sr_{a}[{i}] = {P}me
}}
if {P}r > 0 and {P}r != {P}me {{
    if floor(({P}r - 1) / {P}threads) == {P}phase {{
        {P}sr_{a}[{i}] = -1.0 - {P}phase
    }}
    else {{
        {P}sr_{a}[{i}] = {P}me
    }}
}}
if {P}r < 0 and -{P}r - 1 != {P}phase {{
    {P}sr_{a}[{i}] = {P}me
}}"#
            )
        } else {
            format!(
                r#"{P}me = 1.0 * ({P}b * {P}threads + {P}t + 1)
{P}w = {P}gw_{a}[{i}]
if {P}w > 0 and {P}w != {P}me {{
    {P}wb = floor(({P}w - 1) / {P}threads)
    if {P}wb != {P}b {{
        stop(f"{who}: reads {a}[{{{i}}}], which block {{{P}wb}} wrote in this launch: blocks cannot see each other's writes")
    }}
    if {P}gp_{a}[{i}] == {P}phase {{
        stop(f"{who}: reads {a}[{{{i}}}], which thread {{{P}w - 1 - {P}wb * {P}threads}} wrote with no barrier() between them")
    }}
}}
{P}r = {P}gr_{a}[{i}]
if {P}r == 0 {{
    {P}gr_{a}[{i}] = {P}me
}}
if {P}r > 0 and {P}r != {P}me {{
    if floor(({P}r - 1) / {P}threads) == {P}b {{
        {P}gr_{a}[{i}] = -1.0 - {P}b
    }}
    else {{
        {P}gr_{a}[{i}] = -1.0 - {P}blocks
    }}
}}
if {P}r < 0 and -{P}r - 1 != {P}b {{
    {P}gr_{a}[{i}] = -1.0 - {P}blocks
}}
{P}grp_{a}[{i}] = 1.0 * {P}phase"#
            )
        };
        tmpl(&text, sp)
    }

    fn on_write(&self, a: &str, i: &str, sp: Span) -> Vec<Stmt> {
        let who = self.who();
        let text = if self.k.is_shared(a) {
            format!(
                r#"{P}me = 1.0 * ({P}phase * {P}threads + {P}t + 1)
{P}w = {P}sw_{a}[{i}]
if {P}w > 0 {{
    {P}wp = floor(({P}w - 1) / {P}threads)
    {P}wt = {P}w - 1 - {P}wp * {P}threads
    if {P}wp == {P}phase and {P}wt != {P}t {{
        stop(f"{who}: writes {a}[{{{i}}}], which thread {{{P}wt}} also wrote with no barrier() between them")
    }}
}}
{P}r = {P}sr_{a}[{i}]
if {P}r < 0 and -{P}r - 1 == {P}phase {{
    stop(f"{who}: writes {a}[{{{i}}}], which other threads read with no barrier() between them")
}}
if {P}r > 0 and {P}r != {P}me and floor(({P}r - 1) / {P}threads) == {P}phase {{
    stop(f"{who}: writes {a}[{{{i}}}], which thread {{{P}r - 1 - {P}phase * {P}threads}} read with no barrier() between them")
}}
{P}sw_{a}[{i}] = {P}me"#
            )
        } else {
            format!(
                r#"{P}me = 1.0 * ({P}b * {P}threads + {P}t + 1)
{P}w = {P}gw_{a}[{i}]
if {P}w > 0 and {P}w != {P}me {{
    {P}wb = floor(({P}w - 1) / {P}threads)
    if {P}wb != {P}b {{
        stop(f"{who}: writes {a}[{{{i}}}], which block {{{P}wb}} also wrote in this launch")
    }}
    if {P}gp_{a}[{i}] == {P}phase {{
        stop(f"{who}: writes {a}[{{{i}}}], which thread {{{P}w - 1 - {P}wb * {P}threads}} also wrote with no barrier() between them")
    }}
}}
{P}r = {P}gr_{a}[{i}]
if {P}r == -1 - {P}blocks {{
    stop(f"{who}: writes {a}[{{{i}}}], which threads of other blocks read in this launch")
}}
if {P}r < 0 and {P}r != -1 - {P}blocks {{
    if -{P}r - 1 != {P}b {{
        stop(f"{who}: writes {a}[{{{i}}}], which threads of block {{-{P}r - 1}} read in this launch")
    }}
    if {P}grp_{a}[{i}] == {P}phase {{
        stop(f"{who}: writes {a}[{{{i}}}], which other threads read with no barrier() between them")
    }}
}}
if {P}r > 0 and {P}r != {P}me {{
    {P}rb = floor(({P}r - 1) / {P}threads)
    if {P}rb != {P}b {{
        stop(f"{who}: writes {a}[{{{i}}}], which block {{{P}rb}} read in this launch")
    }}
    if {P}grp_{a}[{i}] == {P}phase {{
        stop(f"{who}: writes {a}[{{{i}}}], which thread {{{P}r - 1 - {P}rb * {P}threads}} read with no barrier() between them")
    }}
}}
{P}gw_{a}[{i}] = {P}me
{P}gp_{a}[{i}] = 1.0 * {P}phase"#
            )
        };
        tmpl(&text, sp)
    }
}

fn tmp_phase(sp: Span) -> Vec<Stmt> {
    tmpl(&format!("{P}phase = {P}phase + 1"), sp)
}

fn stmt_reads(s: &Stmt, out: &mut Vec<(String, Span)>) {
    match s {
        Stmt::Assign { value, .. } => reads(value, out),
        Stmt::SetIndex { indices, value, .. } => {
            indices.iter().for_each(|i| reads(i, out));
            reads(value, out);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            reads(cond, out);
            then_body.iter().chain(else_body).for_each(|t| stmt_reads(t, out));
        }
        Stmt::While { cond, body, .. } => {
            reads(cond, out);
            body.iter().for_each(|t| stmt_reads(t, out));
        }
        Stmt::For { bound, body, .. } => {
            reads(bound, out);
            body.iter().for_each(|t| stmt_reads(t, out));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------- CUDA C

/// The C type of a value in a kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CTy {
    Double,
    Long,
    Bool,
    /// a whole-number literal: a `long long` beside one, a `double` alone
    Lit,
}

fn cty_join(a: CTy, b: CTy) -> CTy {
    match (a, b) {
        (CTy::Bool, _) | (_, CTy::Bool) => CTy::Bool,
        (CTy::Double, _) | (_, CTy::Double) => CTy::Double,
        (CTy::Long, _) | (_, CTy::Long) => CTy::Long,
        _ => CTy::Lit,
    }
}

fn cname(v: &str) -> String {
    format!("v_{v}")
}

struct Emit<'a> {
    /// parameters and locals with their types; arrays by element type
    vars: BTreeMap<String, CTy>,
    arrays: HashMap<String, CTy>,
    consts: &'a HashMap<String, (f64, bool)>,
    /// the literals written as whole numbers (`Program::whole`)
    whole: &'a HashMap<(u32, u32), i64>,
    n: usize,
}

impl<'a> Emit<'a> {
    fn ty(&self, e: &Expr) -> CTy {
        match e {
            Expr::Num { span, .. } if self.whole.contains_key(&(span.start, span.end)) => CTy::Lit,
            Expr::Num { .. } => CTy::Double,
            Expr::Bool { .. } => CTy::Bool,
            Expr::Var { name, .. } => match self.vars.get(name) {
                Some(t) => *t,
                None => match self.consts.get(name) {
                    Some((_, true)) => CTy::Long,
                    _ => CTy::Double,
                },
            },
            Expr::Index { base, .. } => match &**base {
                Expr::Var { name, .. } => *self.arrays.get(name).unwrap_or(&CTy::Double),
                _ => CTy::Double,
            },
            Expr::Neg { operand, .. } => self.ty(operand),
            Expr::Not { .. } => CTy::Bool,
            Expr::Binary { op, lhs, rhs, .. } => match op {
                BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::EqEq | BinOp::Ne | BinOp::And | BinOp::Or => CTy::Bool,
                BinOp::Div => CTy::Double,
                _ => cty_join(self.ty(lhs), self.ty(rhs)),
            },
            Expr::Call { callee, args, .. } => match callee.as_str() {
                THREAD_ID | BLOCK_ID | BLOCK_SIZE | GRID_SIZE | "len" => CTy::Long,
                "abs" if args.len() == 1 && self.ty(&args[0]) == CTy::Long => CTy::Long,
                _ => CTy::Double,
            },
            _ => CTy::Double,
        }
    }

    fn infer(&mut self, body: &[Stmt]) {
        // to a fixed point: a variable takes the join of what is assigned to it
        loop {
            let before = self.vars.clone();
            self.infer_pass(body);
            if self.vars == before {
                break;
            }
        }
    }

    fn infer_pass(&mut self, body: &[Stmt]) {
        for s in body {
            match s {
                Stmt::Assign { name, value, .. } if shared_decl(s).is_none() => {
                    let t = self.ty(value);
                    let t = if t == CTy::Lit { CTy::Double } else { t };
                    let cur = self.vars.get(name).copied();
                    let nt = match cur {
                        None => t,
                        Some(c) => cty_join(c, t),
                    };
                    self.vars.insert(name.clone(), nt);
                }
                Stmt::ExprStmt { .. } => {
                    if let (Some(n), Stmt::ExprStmt { expr: Expr::Call { args, .. }, .. }) = (declared_name(s), s) {
                        if let Some(Expr::Str { value, .. }) = args.get(1) {
                            let t = match value.as_str() {
                                "Int" => CTy::Long,
                                "Bool" => CTy::Bool,
                                _ => CTy::Double,
                            };
                            self.vars.insert(n.to_string(), t);
                        }
                    }
                }
                Stmt::If { then_body, else_body, .. } => {
                    self.infer_pass(then_body);
                    self.infer_pass(else_body);
                }
                Stmt::While { body, .. } => self.infer_pass(body),
                Stmt::For { var, body, .. } => {
                    self.vars.insert(var.clone(), CTy::Long);
                    self.infer_pass(body);
                }
                _ => {}
            }
        }
    }

    /// A C double literal that reads back to `v` (Rust's `{:e}` is the
    /// shortest such digits).
    fn num(v: f64) -> String {
        format!("{v:e}")
    }

    fn expr(&self, e: &Expr) -> String {
        match e {
            Expr::Num { value, .. } => {
                if self.ty(e) == CTy::Lit {
                    format!("{}LL", *value as i64)
                } else {
                    Self::num(*value)
                }
            }
            Expr::Bool { value, .. } => format!("{value}"),
            Expr::Var { name, .. } => match (self.vars.contains_key(name), self.consts.get(name)) {
                (false, Some((v, true))) => format!("{}LL", *v as i64),
                (false, Some((v, false))) => Self::num(*v),
                _ => cname(name),
            },
            Expr::Index { base, indices, .. } => format!("{}[(long long)({})]", self.expr(base), self.expr(&indices[0])),
            Expr::Neg { operand, .. } => format!("(-{})", self.expr(operand)),
            Expr::Not { operand, .. } => format!("(!{})", self.expr(operand)),
            Expr::Binary { op, lhs, rhs, .. } => {
                let (l, r) = (self.expr(lhs), self.expr(rhs));
                let int = cty_join(self.ty(lhs), self.ty(rhs)) != CTy::Double;
                match op {
                    BinOp::Div => format!("((double)({l}) / (double)({r}))"),
                    BinOp::FloorDiv if int => format!("viv_floordiv_ll({l}, {r})"),
                    BinOp::FloorDiv => format!("viv_floordiv_d({l}, {r})"),
                    BinOp::Mod if int => format!("viv_floormod_ll({l}, {r})"),
                    BinOp::Mod => format!("viv_floormod_d({l}, {r})"),
                    _ => format!("({l} {} {r})", op.c_op()),
                }
            }
            Expr::Call { callee, args, .. } => match callee.as_str() {
                THREAD_ID | BLOCK_ID | BLOCK_SIZE | GRID_SIZE => {
                    let v = match callee.as_str() {
                        THREAD_ID => "threadIdx",
                        BLOCK_ID => "blockIdx",
                        BLOCK_SIZE => "blockDim",
                        _ => "gridDim",
                    };
                    format!("(long long){v}.{}", if axis_of(args) == Some(1) { "y" } else { "x" })
                }
                "len" => match &args[0] {
                    Expr::Var { name, .. } => format!("{}_len", cname(name)),
                    _ => unreachable!("checked"),
                },
                "pow" => format!("viv_pow_sq((double)({}), (double)({}))", self.expr(&args[0]), self.expr(&args[1])),
                "abs" if self.ty(&args[0]) == CTy::Long => format!("llabs({})", self.expr(&args[0])),
                c => {
                    let f = MATH.iter().find(|(n, _)| *n == c).map(|(_, f)| *f).unwrap_or(c);
                    format!("{f}((double)({}))", self.expr(&args[0]))
                }
            },
            _ => "/* not in the kernel subset */".into(),
        }
    }

    fn stmts(&mut self, body: &[Stmt], ind: usize, out: &mut String) {
        let pad = "    ".repeat(ind);
        for s in body {
            match s {
                _ if is_barrier(s) => out.push_str(&format!("{pad}__syncthreads();\n")),
                _ if shared_decl(s).is_some() || declared_name(s).is_some() => {}
                Stmt::ExprStmt { expr, .. } if is_call(expr, MARK) => {}
                Stmt::Assign { name, value, .. } => out.push_str(&format!("{pad}{} = {};\n", cname(name), self.expr(value))),
                Stmt::SetIndex { base, indices, value, .. } => {
                    out.push_str(&format!("{pad}{}[(long long)({})] = {};\n", self.expr(base), self.expr(&indices[0]), self.expr(value)))
                }
                Stmt::If { cond, then_body, else_body, .. } => {
                    out.push_str(&format!("{pad}if ({}) {{\n", self.expr(cond)));
                    self.stmts(then_body, ind + 1, out);
                    if else_body.is_empty() {
                        out.push_str(&format!("{pad}}}\n"));
                    } else {
                        out.push_str(&format!("{pad}}} else {{\n"));
                        self.stmts(else_body, ind + 1, out);
                        out.push_str(&format!("{pad}}}\n"));
                    }
                }
                Stmt::While { cond, body, .. } => {
                    out.push_str(&format!("{pad}while ({}) {{\n", self.expr(cond)));
                    self.stmts(body, ind + 1, out);
                    out.push_str(&format!("{pad}}}\n"));
                }
                Stmt::For { var, bound, body, .. } => {
                    self.n += 1;
                    let b = format!("viv_bound{}", self.n);
                    let v = cname(var);
                    out.push_str(&format!("{pad}{{\n{pad}    long long {b} = (long long)({});\n{pad}    for ({v} = 0; {v} < {b}; {v}++) {{\n", self.expr(bound)));
                    self.stmts(body, ind + 2, out);
                    out.push_str(&format!("{pad}    }}\n{pad}}}\n"));
                }
                Stmt::Break { .. } => out.push_str(&format!("{pad}break;\n")),
                Stmt::Continue { .. } => out.push_str(&format!("{pad}continue;\n")),
                Stmt::Return { .. } => out.push_str(&format!("{pad}return;\n")),
                _ => out.push_str(&format!("{pad}/* not in the kernel subset */\n")),
            }
        }
    }

    fn decls(&self, skip: &HashSet<String>, ind: usize) -> String {
        let pad = "    ".repeat(ind);
        let mut out = String::new();
        for (n, t) in &self.vars {
            if skip.contains(n) {
                continue;
            }
            let c = match t {
                CTy::Long => "long long",
                CTy::Bool => "bool",
                _ => "double",
            };
            out.push_str(&format!("{pad}{c} {};\n", cname(n)));
        }
        out
    }
}

fn c_param(name: &str, t: Ty, written: bool) -> String {
    let c = cname(name);
    match t {
        Ty::Arr => format!("{}double *{c}, long long {c}_len", if written { "" } else { "const " }),
        Ty::IntArr => format!("{}long long *{c}, long long {c}_len", if written { "" } else { "const " }),
        Ty::Num => format!("double {c}"),
        Ty::Int => format!("long long {c}"),
        Ty::Bool => format!("bool {c}"),
    }
}

/// The launch configuration of a kernel whose grid C statements set
/// `blocks` and `threads` (and `blocks_y`, `threads_y` when `two_d`).
fn launch_cfg(two_d: bool) -> &'static str {
    if two_d {
        "<<<dim3((unsigned)blocks, (unsigned)blocks_y), dim3((unsigned)threads, (unsigned)threads_y)>>>"
    } else {
        "<<<(unsigned)blocks, (unsigned)threads>>>"
    }
}

/// A host launcher: compute the grid (`grid`, C statements that set
/// `blocks` and `threads`, and their `_y` twins for a 2-D grid), copy the
/// arrays in, launch, copy the written ones back.
fn launcher(name: &str, kernel: &str, params: &[(String, Ty)], written: &[String], grid: &str, two_d: bool) -> String {
    let mut sig = Vec::new();
    for (n, t) in params {
        sig.push(c_param(n, *t, written.contains(n)));
    }
    let mut out = format!("extern \"C\" int viv_launch_{name}({}) {{\n", sig.join(", "));
    out.push_str(grid);
    let mut args = Vec::new();
    for (n, t) in params {
        let c = cname(n);
        match t {
            Ty::Arr | Ty::IntArr => {
                let et = if *t == Ty::Arr { "double" } else { "long long" };
                out.push_str(&format!(
                    "    {et} *d_{n} = NULL;\n    VIV_CUDA_CHECK(cudaMalloc((void **)&d_{n}, sizeof({et}) * (size_t)({c}_len > 0 ? {c}_len : 1)));\n    VIV_CUDA_CHECK(cudaMemcpy(d_{n}, {c}, sizeof({et}) * (size_t){c}_len, cudaMemcpyHostToDevice));\n"
                ));
                args.push(format!("d_{n}, {c}_len"));
            }
            _ => args.push(c),
        }
    }
    out.push_str(&format!("    {kernel}{}({});\n", launch_cfg(two_d), args.join(", ")));
    out.push_str("    VIV_CUDA_CHECK(cudaGetLastError());\n    VIV_CUDA_CHECK(cudaDeviceSynchronize());\n");
    for (n, t) in params {
        if matches!(t, Ty::Arr | Ty::IntArr) {
            let c = cname(n);
            let et = if *t == Ty::Arr { "double" } else { "long long" };
            if written.contains(n) {
                out.push_str(&format!("    VIV_CUDA_CHECK(cudaMemcpy({c}, d_{n}, sizeof({et}) * (size_t){c}_len, cudaMemcpyDeviceToHost));\n"));
            }
            out.push_str(&format!("    cudaFree(d_{n});\n"));
        }
    }
    out.push_str("    return 0;\n}\n");
    out
}

/// Viv's libm for the device: vmath.c with each `static` function and table
/// marked `__device__`, less the complex functions (CUDA C++ has no
/// `_Complex`).
pub fn device_vmath() -> String {
    let src = include_str!("vmath.c");
    let mut out = String::new();
    let lines: Vec<&str> = src.lines().collect();
    let mut k = 0;
    while k < lines.len() {
        let l = lines[k];
        if (l.starts_with("static") || l.starts_with("typedef")) && l.contains("_Complex") {
            // skip the definition: to the line that closes its first brace
            let mut depth = 0i32;
            let mut opened = false;
            while k < lines.len() {
                for ch in lines[k].chars() {
                    if ch == '{' {
                        depth += 1;
                        opened = true;
                    } else if ch == '}' {
                        depth -= 1;
                    }
                }
                k += 1;
                if (opened && depth == 0) || (!opened && lines[k - 1].trim_end().ends_with(';')) {
                    break;
                }
            }
            continue;
        }
        if l.starts_with("#pragma STDC") || (l.starts_with("#define c") && l.contains("viv_c")) {
            k += 1;
            continue;
        }
        if let Some(rest) = l.strip_prefix("static inline ") {
            out.push_str(&format!("static __device__ inline {rest}\n"));
        } else if let Some(rest) = l.strip_prefix("static ") {
            out.push_str(&format!("static __device__ {rest}\n"));
        } else {
            out.push_str(l);
            out.push('\n');
        }
        k += 1;
    }
    out
}

const PRELUDE: &str = r#"#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define VIV_CUDA_CHECK(x) do { cudaError_t viv_e = (cudaError_t)(x); if (viv_e != cudaSuccess) { fprintf(stderr, "cuda: %s at %s:%d\n", cudaGetErrorString(viv_e), __FILE__, __LINE__); return (int)viv_e; } } while (0)

/* `a // b` and `a % b` floor, as in Viv (and Python) */
static __device__ inline long long viv_floordiv_ll(long long a, long long b) { long long q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1; return q; }
static __device__ inline long long viv_floormod_ll(long long a, long long b) { long long m = a % b; if (m != 0 && ((m < 0) != (b < 0))) m += b; return m; }
static __device__ inline double viv_floormod_d(double a, double b) { double m = fmod(a, b); if (m != 0.0) { if ((b < 0.0) != (m < 0.0)) m += b; } else m = copysign(0.0, b); return m; }
static __device__ inline double viv_floordiv_d(double a, double b) { double m = fmod(a, b); double d = (a - m) / b; if (m != 0.0 && (b < 0.0) != (m < 0.0)) d -= 1.0; if (d != 0.0) { double f = floor(d); if (d - f > 0.5) f += 1.0; return f; } return copysign(0.0, a / b); }
"#;

/// The global constants a kernel may name (`LEAF = 128` at the top level).
fn constants(p: &Program) -> HashMap<String, (f64, bool)> {
    fn eval(e: &Expr, known: &HashMap<String, f64>) -> Option<f64> {
        match e {
            Expr::Num { value, .. } => Some(*value),
            Expr::Var { name, .. } => known.get(name).copied(),
            Expr::Neg { operand, .. } => eval(operand, known).map(|v| -v),
            Expr::Binary { op, lhs, rhs, .. } => {
                let (a, b) = (eval(lhs, known)?, eval(rhs, known)?);
                match op {
                    BinOp::Add => Some(a + b),
                    BinOp::Sub => Some(a - b),
                    BinOp::Mul => Some(a * b),
                    BinOp::FloorDiv if b != 0.0 => Some((a / b).floor()),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    let mut vals: HashMap<String, f64> = HashMap::new();
    let mut ints: HashSet<String> = HashSet::new();
    let mut seen = HashSet::new();
    for i in &p.items {
        match i {
            Item::Stmt(s @ Stmt::ExprStmt { expr: Expr::Call { args, .. }, .. }) if declared_name(s).is_some() => {
                if matches!(args.get(1), Some(Expr::Str { value, .. }) if value == "Int") {
                    ints.insert(declared_name(s).unwrap().to_string());
                }
            }
            Item::Stmt(Stmt::Assign { name, value, .. }) => {
                // a name assigned twice at the top level is a variable, not a constant
                if !seen.insert(name.clone()) {
                    vals.remove(name);
                    continue;
                }
                if let Some(v) = eval(value, &vals) {
                    vals.insert(name.clone(), v);
                }
            }
            _ => {}
        }
    }
    vals.into_iter().map(|(n, v)| {
        let int = ints.contains(&n);
        (n, (v, int))
    }).collect()
}

fn line_of(src: &str, sp: Span) -> usize {
    src.get(..sp.start as usize).map(|s| s.matches('\n').count() + 1).unwrap_or(0)
}

/// The CUDA C of `program` (parsed with [`raw`]): its kernels and its
/// eligible `parallel for` loops. `src` gives the line numbers in comments.
pub fn emit(program: &Program, src: &str, file: &str) -> Result<String, Vec<Diagnostic>> {
    let consts = constants(program);
    let defs: HashSet<String> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    let mut out = format!(
        "// Generated by `viv cuda {file}`. Not compiled here.\n// Build for exact mode: nvcc -fmad=false -prec-div=true -prec-sqrt=true -ftz=false\n{PRELUDE}\n// ---- Viv's libm, for the device (lang/viv/src/vmath.c)\n{}\n",
        device_vmath()
    );
    let scheds = schedules(program).map_err(|e| vec![e])?;
    let mut errs = Vec::new();
    let mut skipped = Vec::new();
    for item in &program.items {
        let Item::Def(d) = item else { continue };
        if is_kernel(d) {
            match emit_kernel(d, &defs, &consts, &program.whole, src, scheds.get(&d.name)) {
                Ok(s) => out.push_str(&s),
                Err(e) => errs.push(e),
            }
        } else {
            emit_pfors(d, &defs, &consts, &program.whole, src, &mut out, &mut skipped);
        }
    }
    if !errs.is_empty() {
        return Err(errs);
    }
    // the `on device` blocks, after every launcher
    let kinds: HashMap<String, KInfo> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Def(d) if is_kernel(d) => kinfo_of(d).map(|k| (d.name.clone(), k)),
            _ => None,
        })
        .collect();
    for item in &program.items {
        if let Item::Def(d) = item {
            if !is_kernel(d) {
                emit_regions(&d.name, &d.params, &d.body, &kinds, &consts, &program.whole, src, &mut out, &mut skipped);
            }
        }
    }
    let top: Vec<Stmt> = program.items.iter().filter_map(|i| if let Item::Stmt(st) = i { Some(st.clone()) } else { None }).collect();
    emit_regions("main", &[], &top, &kinds, &consts, &program.whole, src, &mut out, &mut skipped);
    for s in skipped {
        out.push_str(&format!("// not emitted: {s}\n"));
    }
    Ok(out)
}

fn emit_kernel(d: &Def, defs: &HashSet<String>, consts: &HashMap<String, (f64, bool)>, whole: &HashMap<(u32, u32), i64>, src: &str, sched: Option<&Sched>) -> Result<String, Diagnostic> {
    let k = kern_of(d)?;
    let line = line_of(src, d.name_span);
    let mut e = Emit { vars: BTreeMap::new(), arrays: HashMap::new(), consts, whole, n: 0 };
    for (n, t) in &k.params {
        match t {
            Ty::Arr => {
                e.arrays.insert(n.clone(), CTy::Double);
            }
            Ty::IntArr => {
                e.arrays.insert(n.clone(), CTy::Long);
            }
            Ty::Num => {
                e.vars.insert(n.clone(), CTy::Double);
            }
            Ty::Int => {
                e.vars.insert(n.clone(), CTy::Long);
            }
            Ty::Bool => {
                e.vars.insert(n.clone(), CTy::Bool);
            }
        }
    }
    let params: HashSet<String> = k.params.iter().map(|(n, _)| n.clone()).collect();
    let name = &d.name;
    if let Some(tree) = elementwise_body(d) {
        check_tree(tree, &k)?;
        let mut arrays = Vec::new();
        tree_arrays(tree, &k, &mut arrays);
        fn at_i(e: &Expr, arrays: &[String]) -> Expr {
            let sp = e.span();
            match e {
                Expr::Var { name, .. } if arrays.contains(name) => Expr::Index { base: Box::new(var(name, sp)), indices: vec![var("viv_i", sp)], span: fresh_near(sp) },
                Expr::Neg { operand, span } => Expr::Neg { operand: Box::new(at_i(operand, arrays)), span: *span },
                Expr::Binary { op, lhs, rhs, span } => Expr::Binary { op: *op, lhs: Box::new(at_i(lhs, arrays)), rhs: Box::new(at_i(rhs, arrays)), span: *span },
                Expr::Call { callee, args, span } => Expr::Call { callee: callee.clone(), args: args.iter().map(|a| at_i(a, arrays)).collect(), span: *span },
                _ => e.clone(),
            }
        }
        e.vars.insert("viv_i".into(), CTy::Long);
        let value = e.expr(&at_i(tree, &arrays));
        let mut sig: Vec<String> = vec!["double *v_viv_out".into(), "long long v_viv_n".into()];
        sig.extend(k.params.iter().map(|(n, t)| c_param(n, *t, false)));
        let first = arrays.first().cloned().unwrap_or_default();
        let mut s = format!(
            "\n// ---- kernel {name} (line {line}): an elementwise tree, one thread per element\n__global__ void viv_kernel_{name}({}) {{\n    long long v_viv_i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n    if (v_viv_i < v_viv_n) {{\n        v_viv_out[v_viv_i] = {value};\n    }}\n}}\n",
            sig.join(", ")
        );
        let mut lp = vec![("viv_out".to_string(), Ty::Arr)];
        lp.extend(k.params.iter().cloned());
        let grid = format!(
            "    long long v_viv_n = {c}_len;\n    long long threads = {ELEMENTWISE_THREADS};\n    long long blocks = (v_viv_n + threads - 1) / threads;\n    if (blocks < 1) return 0;\n",
            c = cname(&first)
        );
        // the launcher takes the output array as its first argument
        let launch = launcher(name, &format!("viv_kernel_{name}"), &lp, &["viv_out".to_string()], &grid, false)
            .replace(&format!("viv_kernel_{name}<<<(unsigned)blocks, (unsigned)threads>>>(d_viv_out, v_viv_out_len, "), &format!("viv_kernel_{name}<<<(unsigned)blocks, (unsigned)threads>>>(d_viv_out, v_viv_n, "));
        s.push_str(&launch);
        s.push_str(&dlauncher(name, &format!("viv_kernel_{name}"), &lp, &["viv_out".to_string()], &grid, true, false));
        return Ok(s);
    }
    for st in &d.body[1..] {
        check_stmt(st, &k, defs, true)?;
    }
    let body = &d.body[1..];
    let mut shared = String::new();
    for (n, size) in &k.shared {
        let size = match size {
            Expr::Num { value, .. } => *value as i64,
            Expr::Var { name, span } => match consts.get(name) {
                Some((v, _)) => *v as i64,
                None => return Err(err(*span, format!("`{name}` is not a global constant (a number assigned at the top level)"), "the size of a shared array")),
            },
            _ => unreachable!("checked"),
        };
        shared.push_str(&format!("    __shared__ double {}[{size}];\n", cname(n)));
        e.arrays.insert(n.clone(), CTy::Double);
    }
    e.infer(body);
    let decls = e.decls(&params, 1);
    let mut code = String::new();
    e.stmts(body, 1, &mut code);
    let sig: Vec<String> = k.params.iter().map(|(n, t)| c_param(n, *t, k.written.contains(n))).collect();
    let mut s = format!("\n// ---- kernel {name} (line {line})\n__global__ void viv_kernel_{name}({}) {{\n{shared}{decls}{code}}}\n", sig.join(", "));
    // the grid, as the emulator computes it (Explicit::lower_grid)
    let two_d = sched_two_d(sched);
    let axis = |v: Option<&Vec<Expr>>, i: usize| v.and_then(|v| v.get(i)).map(|x| e.expr(x));
    let threads = axis(sched.and_then(|x| x.threads.as_ref()), 0).unwrap_or(format!("{ELEMENTWISE_THREADS}LL"));
    let threads_y = axis(sched.and_then(|x| x.threads.as_ref()), 1).unwrap_or("1LL".into());
    let (blocks, blocks_y) = match (sched.and_then(|x| x.blocks.as_ref()), sched.and_then(|x| x.cover.as_ref())) {
        (Some(b), _) => (e.expr(&b[0]), b.get(1).map_or("1LL".into(), |y| e.expr(y))),
        (None, Some(c)) => (format!("({} + threads - 1) / threads", e.expr(&c[0])), c.get(1).map_or("1LL".into(), |y| format!("({} + threads_y - 1) / threads_y", e.expr(y)))),
        (None, None) => match sized_by(&k) {
            Some(first) => (format!("({}_len + threads - 1) / threads", cname(first)), "1LL".into()),
            None => return Err(err(d.name_span, format!("the kernel `{name}` has no array parameter to size its grid from; give the grid in a top-level line, `schedule {name}(blocks = ..., threads = ...)`"), "no array parameter")),
        },
    };
    let grid = if two_d {
        format!(
            "    long long threads = {threads};\n    long long threads_y = {threads_y};\n    long long blocks = {blocks};\n    long long blocks_y = {blocks_y};\n    if (blocks < 1 || threads < 1 || blocks_y < 1 || threads_y < 1 || threads * threads_y > {MAX_THREADS} || blocks_y > {MAX_BLOCKS_Y}) return 1;\n"
        )
    } else {
        format!("    long long threads = {threads};\n    long long blocks = {blocks};\n    if (blocks < 1 || threads < 1 || threads > {MAX_THREADS}) return 1;\n")
    };
    s.push_str(&launcher(name, &format!("viv_kernel_{name}"), &k.params, &k.written, &grid, two_d));
    s.push_str(&dlauncher(name, &format!("viv_kernel_{name}"), &k.params, &k.written, &grid, false, two_d));
    Ok(s)
}

/// `parallel for` loops of a def: a kernel each, when the body is in the
/// subset and every variable it reads has a type the emitter can see.
fn emit_pfors(d: &Def, defs: &HashSet<String>, consts: &HashMap<String, (f64, bool)>, whole: &HashMap<(u32, u32), i64>, src: &str, out: &mut String, skipped: &mut Vec<String>) {
    // the def's variables and their types, from its annotations and assignments
    let mut types: HashMap<String, Ty> = HashMap::new();
    for p in &d.params {
        if let Ok(t) = param_ty(p) {
            types.insert(p.name.clone(), t);
        }
    }
    fn arrays_made(body: &[Stmt], types: &mut HashMap<String, Ty>) {
        for s in body {
            match s {
                Stmt::Assign { name, value: Expr::Call { callee, .. }, .. } if matches!(callee.as_str(), "zeros" | "ones" | "copy") => {
                    types.entry(name.clone()).or_insert(Ty::Arr);
                }
                Stmt::Assign { name, value: Expr::Call { callee, .. }, .. } if callee == "len" => {
                    types.entry(name.clone()).or_insert(Ty::Int);
                }
                Stmt::If { then_body, else_body, .. } => {
                    arrays_made(then_body, types);
                    arrays_made(else_body, types);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } => arrays_made(body, types),
                _ => {}
            }
        }
    }
    arrays_made(&d.body, &mut types);
    let mut n = 0;
    fn find<'b>(body: &'b [Stmt], out: &mut Vec<&'b Stmt>) {
        for s in body {
            match s {
                Stmt::For { parallel: true, .. } => out.push(s),
                Stmt::If { then_body, else_body, .. } => {
                    find(then_body, out);
                    find(else_body, out);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } => find(body, out),
                _ => {}
            }
        }
    }
    let mut loops = Vec::new();
    find(&d.body, &mut loops);
    for l in loops {
        let Stmt::For { var: iv, bound, body, span, .. } = l else { continue };
        n += 1;
        let line = line_of(src, *span);
        let kname = format!("{}_pfor{n}", d.name);
        // the loop's free variables
        let mut rs = Vec::new();
        for s in body {
            stmt_reads(s, &mut rs);
        }
        reads(bound, &mut rs);
        let mut assigned = vec![iv.clone()];
        for s in body {
            assigned_in(s, &mut assigned);
        }
        let mut written = Vec::new();
        written_in(body, &mut written);
        let mut free: Vec<(String, Ty)> = Vec::new();
        let mut why = None;
        for (v, _) in rs.iter().chain(written.iter().map(|w| (w.clone(), *span)).collect::<Vec<_>>().iter()) {
            if assigned.contains(v) || free.iter().any(|(f, _)| f == v) || consts.contains_key(v) {
                continue;
            }
            match types.get(v) {
                Some(t) => free.push((v.clone(), *t)),
                None => {
                    why = Some(format!("`{v}` has no type the emitter can see (annotate it as a parameter)"));
                    break;
                }
            }
        }
        let k = Kern { name: kname.clone(), params: free.clone(), written: written.clone(), shared: vec![] };
        if why.is_none() {
            if let Some(e) = body.iter().find_map(|s| check_stmt(s, &k, defs, false).err()) {
                why = Some(e.message.clone());
            }
        }
        if let Some(w) = why {
            skipped.push(format!("the parallel for of `{}` at line {line}: {w}", d.name));
            continue;
        }
        let mut e = Emit { vars: BTreeMap::new(), arrays: HashMap::new(), consts, whole, n: 0 };
        for (v, t) in &free {
            match t {
                Ty::Arr => e.arrays.insert(v.clone(), CTy::Double),
                Ty::IntArr => e.arrays.insert(v.clone(), CTy::Long),
                Ty::Num => e.vars.insert(v.clone(), CTy::Double),
                Ty::Int => e.vars.insert(v.clone(), CTy::Long),
                Ty::Bool => e.vars.insert(v.clone(), CTy::Bool),
            };
        }
        let skip: HashSet<String> = free.iter().map(|(v, _)| v.clone()).chain([iv.clone()]).collect();
        e.vars.insert(iv.clone(), CTy::Long);
        e.infer(body);
        let decls = e.decls(&skip, 1);
        let mut code = String::new();
        e.stmts(body, 1, &mut code);
        let ivc = cname(iv);
        let mut sig = vec!["long long viv_n".to_string()];
        sig.extend(free.iter().map(|(v, t)| c_param(v, *t, written.contains(v))));
        out.push_str(&format!(
            "\n// ---- the parallel for of `{}` (line {line}): one thread per pass\n__global__ void viv_kernel_{kname}({}) {{\n    long long {ivc} = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n    if ({ivc} >= viv_n) return;\n{decls}{code}}}\n",
            d.name,
            sig.join(", ")
        ));
        let mut lp = vec![("viv_n".to_string(), Ty::Int)];
        lp.extend(free.iter().cloned());
        let grid = format!("    long long threads = {ELEMENTWISE_THREADS};\n    long long blocks = (v_viv_n + threads - 1) / threads;\n    if (blocks < 1) return 0;\n");
        out.push_str(&launcher(&kname, &format!("viv_kernel_{kname}"), &lp, &written, &grid, false));
    }
}

// ---------------------------------------------------------------- `on device`

/// The condition an `on device a, b { ... }` block parses to (the parser's
/// `on_device`): `if viv__device(a, b) { ... }`, checked and rewritten by
/// [`lower`].
pub const DEVICE: &str = "viv__device";
/// The global the rewrite counts transfers in: to the device, to the host,
/// kernel launches, and how many `on device` blocks are running.
const XFER: &str = "viv_k_xfer";
/// The def that returns the first three counts.
pub const TRANSFERS: &str = "device_transfers";

/// What the `on device` checks need to know of a kernel.
struct KInfo {
    params: Vec<(String, Ty)>,
    written: Vec<String>,
    /// for an elementwise kernel, the parameter its grid covers
    elementwise: Option<String>,
}

fn kinfo_of(d: &Def) -> Option<KInfo> {
    let k = kern_of(d).ok()?;
    let elementwise = elementwise_body(d).map(|tree| {
        let mut arrays = Vec::new();
        tree_arrays(tree, &k, &mut arrays);
        arrays.first().cloned().unwrap_or_default()
    });
    Some(KInfo { params: k.params, written: k.written, elementwise })
}

/// The arrays of an `on device` block, from its condition.
fn device_arrays(cond: &Expr) -> Option<Vec<(String, Span)>> {
    match cond {
        Expr::Call { callee, args, .. } if callee == DEVICE => {
            Some(args.iter().filter_map(|a| if let Expr::Var { name, span } = a { Some((name.clone(), *span)) } else { None }).collect())
        }
        _ => None,
    }
}

fn has_regions(body: &[Stmt]) -> bool {
    body.iter().any(|s| match s {
        Stmt::If { cond, then_body, else_body, .. } => device_arrays(cond).is_some() || has_regions(then_body) || has_regions(else_body),
        Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => has_regions(body),
        _ => false,
    })
}

fn program_has_regions(p: &Program) -> bool {
    p.items.iter().any(|i| match i {
        Item::Def(d) => has_regions(&d.body),
        Item::Stmt(s) => has_regions(std::slice::from_ref(s)),
        _ => false,
    })
}

struct RegionCx<'a> {
    kinds: &'a HashMap<String, KInfo>,
    defs: &'a HashSet<String>,
    arrays: Vec<String>,
}

impl RegionCx<'_> {
    fn on_device(&self, n: &str) -> bool {
        self.arrays.iter().any(|a| a == n)
    }

    /// An expression the host evaluates inside the block: it names no device
    /// array (but in `len`), calls no def, and calls no kernel.
    fn host_expr(&self, e: &Expr) -> Result<(), Diagnostic> {
        match e {
            Expr::Var { name, span } if self.on_device(name) => Err(err(
                *span,
                format!("`{name}` is on the device inside `on device`: the host cannot read it here (it comes back at the end of the block)"),
                "a host read of a device array",
            )),
            Expr::Call { callee, args, .. } if callee == "len" && matches!(args.as_slice(), [Expr::Var { .. }]) => Ok(()),
            Expr::Call { callee, span, .. } if self.kinds.contains_key(callee) => Err(err(
                *span,
                format!("inside `on device` a kernel call is a statement of its own, its result one of the block's arrays: `v = {callee}(...)`"),
                "a kernel call in an expression",
            )),
            Expr::Call { callee, span, .. } if self.defs.contains(callee) && callee != TRANSFERS => Err(err(
                *span,
                format!("`{callee}` is a def: inside `on device` the block calls kernels only, so no def can read a device array"),
                "a call of a def",
            )),
            Expr::Call { args, .. } | Expr::ArrLit { elems: args, .. } => args.iter().try_for_each(|a| self.host_expr(a)),
            Expr::Index { base, indices, .. } => {
                self.host_expr(base)?;
                indices.iter().try_for_each(|a| self.host_expr(a))
            }
            Expr::Lambda { body, .. } => self.host_expr(body),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.host_expr(operand),
            Expr::Binary { lhs, rhs, .. } => {
                self.host_expr(lhs)?;
                self.host_expr(rhs)
            }
            Expr::RecordLit { fields, .. } => fields.iter().try_for_each(|(_, _, v)| self.host_expr(v)),
            Expr::Field { base, .. } => self.host_expr(base),
            _ => Ok(()),
        }
    }

    /// A kernel call inside the block: its array arguments are the block's
    /// arrays, its others host expressions, and no array it writes is
    /// passed twice.
    fn kernel_call<'k>(&'k self, callee: &str, args: &[Expr], span: Span) -> Result<&'k KInfo, Diagnostic> {
        let k = &self.kinds[callee];
        if args.len() != k.params.len() {
            return Err(err(span, format!("the kernel `{callee}` takes {} arguments; this call gives {}", k.params.len(), args.len()), "this call"));
        }
        for ((p, t), a) in k.params.iter().zip(args) {
            if matches!(t, Ty::Arr | Ty::IntArr) {
                match a {
                    Expr::Var { name, .. } if self.on_device(name) => {}
                    _ => {
                        return Err(err(
                            a.span(),
                            format!("inside `on device`, the array `{p}` of `{callee}` is one of the block's arrays ({}); name the array in `on device` to keep it there", self.arrays.join(", ")),
                            "not a device array",
                        ))
                    }
                }
            } else {
                self.host_expr(a)?;
            }
        }
        let name_of = |i: usize| if let Expr::Var { name, .. } = &args[i] { Some(name.as_str()) } else { None };
        for i in 0..args.len() {
            for j in i + 1..args.len() {
                let both = matches!(k.params[i].1, Ty::Arr | Ty::IntArr) && matches!(k.params[j].1, Ty::Arr | Ty::IntArr);
                let writes = k.written.contains(&k.params[i].0) || k.written.contains(&k.params[j].0);
                if both && writes && name_of(i).is_some() && name_of(i) == name_of(j) {
                    return Err(err(
                        args[j].span(),
                        format!("`{}` is passed to `{callee}` twice, and the kernel writes one of them: give it two arrays", name_of(i).unwrap_or_default()),
                        "the same device array again",
                    ));
                }
            }
        }
        Ok(k)
    }

    /// The array argument a kernel's written parameter `p` was given.
    fn arg_for<'e>(k: &KInfo, args: &'e [Expr], p: &str) -> Option<&'e str> {
        let i = k.params.iter().position(|(n, _)| n == p)?;
        if let Expr::Var { name, .. } = &args[i] {
            Some(name.as_str())
        } else {
            None
        }
    }

    /// Checks the block's body, with a length check put before each
    /// elementwise kernel call (the device writes into the target).
    fn body(&self, body: &[Stmt], loops: usize) -> Result<Vec<Stmt>, Diagnostic> {
        let mut out = Vec::new();
        for s in body {
            match s {
                Stmt::Assign { name, name_span, value: Expr::Call { callee, args, span }, .. } if self.kinds.contains_key(callee) => {
                    let k = self.kernel_call(callee, args, *span)?;
                    if !self.on_device(name) {
                        return Err(err(*name_span, format!("inside `on device` the result of `{callee}` goes to one of the block's arrays ({})", self.arrays.join(", ")), "not a device array"));
                    }
                    match &k.elementwise {
                        Some(g) => {
                            let given = Self::arg_for(k, args, g).unwrap_or_default();
                            out.extend(tmpl(
                                &format!("if len({name}) != len({given}) {{\n    stop(f\"on device: `{name}` has {{len({name})}} elements and `{callee}` makes {{len({given})}}\")\n}}"),
                                *span,
                            ));
                        }
                        None => {
                            if k.written.len() != 1 {
                                return Err(err(*span, format!("`{callee}` writes {} arrays ({}): unpack them, `{} = {callee}(...)`", k.written.len(), k.written.join(", "), k.written.join(", ")), "several results"));
                            }
                            let given = Self::arg_for(k, args, &k.written[0]).unwrap_or_default();
                            if given != name {
                                return Err(err(*name_span, format!("`{callee}` writes its `{}`, given `{given}` here, so on the device its result is `{given}`: write `{given} = {callee}(...)`", k.written[0]), "another array"));
                            }
                        }
                    }
                    out.push(s.clone());
                }
                Stmt::Assign { name, name_span, value, .. } => {
                    if self.on_device(name) {
                        return Err(err(*name_span, format!("`{name}` is on the device here: it can be given a kernel's result or swapped with another of the block's arrays, `u, v = v, u`"), "assigned on the host"));
                    }
                    self.host_expr(value)?;
                    out.push(s.clone());
                }
                Stmt::ExprStmt { expr: Expr::Call { callee, args, span }, .. } if callee == crate::tuples::UNPACK && args.len() == 2 => {
                    let targets: &[Expr] = match &args[0] {
                        Expr::ArrLit { elems, .. } => elems,
                        _ => &[],
                    };
                    let tnames: Vec<Option<&str>> = targets.iter().map(|t| if let Expr::Var { name, .. } = t { Some(name.as_str()) } else { None }).collect();
                    let any_dev = tnames.iter().any(|n| n.is_some_and(|n| self.on_device(n)));
                    match &args[1] {
                        Expr::Call { callee: k, args: kargs, span: ksp } if self.kinds.contains_key(k) => {
                            let info = self.kernel_call(k, kargs, *ksp)?;
                            let want: Vec<Option<&str>> = info.written.iter().map(|w| Self::arg_for(info, kargs, w)).collect();
                            if info.elementwise.is_some() || want != tnames {
                                let names: Vec<&str> = want.iter().map(|w| w.unwrap_or("?")).collect();
                                return Err(err(*span, format!("on the device `{k}` writes into the arrays it is given, so its results are `{}`", names.join(", ")), "other targets"));
                            }
                        }
                        Expr::Call { callee: t, args: srcs, .. } if t == crate::tuples::TUPLE && any_dev => {
                            let snames: Vec<Option<&str>> = srcs.iter().map(|t| if let Expr::Var { name, .. } = t { Some(name.as_str()) } else { None }).collect();
                            let all_dev = |ns: &[Option<&str>]| ns.iter().all(|n| n.is_some_and(|n| self.on_device(n)));
                            let mut a: Vec<_> = tnames.clone();
                            let mut b: Vec<_> = snames.clone();
                            a.sort();
                            b.sort();
                            let distinct = a.windows(2).all(|w| w[0] != w[1]);
                            if !(all_dev(&tnames) && all_dev(&snames) && distinct && a == b) {
                                return Err(err(*span, "inside `on device` the block's arrays are swapped among themselves, each named once on each side: `u, v = v, u`", "not a swap of device arrays"));
                            }
                        }
                        rhs => {
                            if any_dev {
                                return Err(err(*span, "a device array inside `on device` takes a kernel's result or a swap, `u, v = v, u`", "assigned on the host"));
                            }
                            targets.iter().try_for_each(|t| self.host_expr(t))?;
                            self.host_expr(rhs)?;
                        }
                    }
                    out.push(s.clone());
                }
                Stmt::ExprStmt { expr: Expr::Call { callee, args, span }, .. } if self.kinds.contains_key(callee) => {
                    self.kernel_call(callee, args, *span)?;
                    out.push(s.clone());
                }
                Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => {
                    self.host_expr(expr)?;
                    out.push(s.clone());
                }
                Stmt::If { cond, then_body, else_body, span } => {
                    if device_arrays(cond).is_some() {
                        return Err(err(cond.span(), "an `on device` block inside another: name every array in the outer one", "a second block"));
                    }
                    self.host_expr(cond)?;
                    out.push(Stmt::If { cond: cond.clone(), then_body: self.body(then_body, loops)?, else_body: self.body(else_body, loops)?, span: *span });
                }
                Stmt::While { cond, body, span } => {
                    self.host_expr(cond)?;
                    out.push(Stmt::While { cond: cond.clone(), body: self.body(body, loops + 1)?, span: *span });
                }
                Stmt::For { var, var_span, bound, body, parallel, span } => {
                    if *parallel {
                        return Err(err(*span, "a `parallel for` inside `on device` would run on the host; launch a kernel instead", "a host loop over the data"));
                    }
                    if self.on_device(var) {
                        return Err(err(*var_span, format!("`{var}` is one of the block's device arrays"), "a loop counter"));
                    }
                    self.host_expr(bound)?;
                    out.push(Stmt::For { var: var.clone(), var_span: *var_span, bound: bound.clone(), body: self.body(body, loops + 1)?, parallel: false, span: *span });
                }
                Stmt::Break { span } | Stmt::Continue { span } if loops == 0 => {
                    return Err(err(*span, "this would leave the `on device` block before its arrays come back", "leaves the block"));
                }
                Stmt::Return { span, .. } => {
                    return Err(err(*span, "a `return` inside `on device` would leave before the arrays come back: return after the block", "leaves the block"));
                }
                Stmt::Test { span, .. } => return Err(err(*span, "a test inside `on device`", "a test")),
                Stmt::SetIndex { base, indices, value, .. } => {
                    self.host_expr(base)?;
                    indices.iter().try_for_each(|i| self.host_expr(i))?;
                    self.host_expr(value)?;
                    out.push(s.clone());
                }
                Stmt::SetField { base, value, .. } => {
                    self.host_expr(base)?;
                    self.host_expr(value)?;
                    out.push(s.clone());
                }
                _ => out.push(s.clone()),
            }
        }
        Ok(out)
    }
}

/// One `on device` block, rewritten for the engines: each array copied (its
/// device buffer), the transfers counted, the body, the copies back counted.
fn expand_region(arrays: &[(String, Span)], body: &[Stmt], span: Span, kinds: &HashMap<String, KInfo>, defs: &HashSet<String>, in_parallel: bool) -> Result<Vec<Stmt>, Diagnostic> {
    if in_parallel {
        return Err(err(span, "`on device` inside a `parallel for`", "inside a parallel loop"));
    }
    let mut names: Vec<String> = Vec::new();
    for (n, sp) in arrays {
        if names.contains(n) {
            return Err(err(*sp, format!("`{n}` is named twice"), "again"));
        }
        names.push(n.clone());
    }
    let cx = RegionCx { kinds, defs, arrays: names.clone() };
    let checked = cx.body(body, 0)?;
    let count = names.len();
    let mut text = format!("{XFER}[3] = {XFER}[3] + 1.0\n");
    for n in &names {
        text.push_str(&format!("{n} = copy({n})\n"));
    }
    text.push_str(&format!("{XFER}[0] = {XFER}[0] + {count}.0\n"));
    let mut out = tmpl(&text, span);
    out.extend(checked);
    out.extend(tmpl(&format!("{XFER}[1] = {XFER}[1] + {count}.0\n{XFER}[3] = {XFER}[3] - 1.0"), span));
    Ok(out)
}

fn lower_regions(body: &[Stmt], kinds: &HashMap<String, KInfo>, defs: &HashSet<String>, in_parallel: bool, errs: &mut Vec<Diagnostic>) -> Vec<Stmt> {
    let mut out = Vec::new();
    for s in body {
        match s {
            Stmt::If { cond, then_body, span, .. } if device_arrays(cond).is_some() => match expand_region(&device_arrays(cond).unwrap(), then_body, *span, kinds, defs, in_parallel) {
                Ok(v) => out.extend(v),
                Err(e) => errs.push(e),
            },
            Stmt::If { cond, then_body, else_body, span } => out.push(Stmt::If {
                cond: cond.clone(),
                then_body: lower_regions(then_body, kinds, defs, in_parallel, errs),
                else_body: lower_regions(else_body, kinds, defs, in_parallel, errs),
                span: *span,
            }),
            Stmt::While { cond, body, span } => out.push(Stmt::While { cond: cond.clone(), body: lower_regions(body, kinds, defs, in_parallel, errs), span: *span }),
            Stmt::For { var, var_span, bound, body, parallel, span } => out.push(Stmt::For {
                var: var.clone(),
                var_span: *var_span,
                bound: bound.clone(),
                body: lower_regions(body, kinds, defs, in_parallel || *parallel, errs),
                parallel: *parallel,
                span: *span,
            }),
            Stmt::Test { name, body, span } => out.push(Stmt::Test { name: name.clone(), body: lower_regions(body, kinds, defs, in_parallel, errs), span: *span }),
            _ => out.push(s.clone()),
        }
    }
    out
}

/// The counting a lowered kernel does on entry: outside any `on device`
/// block, the copies of a Phase 1 call (every array parameter in, the
/// written ones out; an elementwise kernel's result in and out), and always
/// one launch.
fn count_launch(k: &KInfo, sp: Span) -> Vec<Stmt> {
    let arrays = k.params.iter().filter(|(_, t)| matches!(t, Ty::Arr | Ty::IntArr)).count();
    let (to_dev, to_host) = match k.elementwise {
        Some(_) => (arrays + 1, 1),
        None => (arrays, k.written.len()),
    };
    tmpl(
        &format!("if {XFER}[3] == 0.0 {{\n    {XFER}[0] = {XFER}[0] + {to_dev}.0\n    {XFER}[1] = {XFER}[1] + {to_host}.0\n}}\n{XFER}[2] = {XFER}[2] + 1.0"),
        sp,
    )
}

// ---------------------------------------------------------------- `on device`, CUDA C

/// A launcher on device pointers, for the kernel calls inside an `on
/// device` block: the grid, the launch, the launch checked; no copies and no
/// synchronization (the block synchronizes once, before its copies back).
/// An elementwise kernel's first two arguments are its output and its length.
fn dlauncher(name: &str, kernel: &str, params: &[(String, Ty)], written: &[String], grid: &str, elementwise: bool, two_d: bool) -> String {
    let sig: Vec<String> = params.iter().map(|(n, t)| c_param(n, *t, written.contains(n))).collect();
    let mut args = Vec::new();
    for (i, (n, t)) in params.iter().enumerate() {
        let c = cname(n);
        match t {
            Ty::Arr | Ty::IntArr if elementwise && i == 0 => args.push(format!("{c}, v_viv_n")),
            Ty::Arr | Ty::IntArr => args.push(format!("{c}, {c}_len")),
            _ => args.push(c),
        }
    }
    format!(
        "extern \"C\" int viv_dlaunch_{name}({}) {{\n{grid}    {kernel}{}({});\n    VIV_CUDA_CHECK(cudaGetLastError());\n    return 0;\n}}\n",
        sig.join(", "),
        launch_cfg(two_d),
        args.join(", ")
    )
}

/// Every name an expression or statement reads (calls included, as names).
fn all_reads_expr(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Var { name, .. } => out.push(name.clone()),
        Expr::Index { base, indices, .. } => {
            all_reads_expr(base, out);
            indices.iter().for_each(|x| all_reads_expr(x, out));
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => all_reads_expr(operand, out),
        Expr::Binary { lhs, rhs, .. } => {
            all_reads_expr(lhs, out);
            all_reads_expr(rhs, out);
        }
        Expr::Call { args, .. } | Expr::ArrLit { elems: args, .. } => args.iter().for_each(|x| all_reads_expr(x, out)),
        Expr::Field { base, .. } => all_reads_expr(base, out),
        _ => {}
    }
}

fn all_reads(body: &[Stmt], out: &mut Vec<String>) {
    for s in body {
        match s {
            Stmt::Assign { value, .. } | Stmt::Show { expr: value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Return { value, .. } => all_reads_expr(value, out),
            Stmt::SetIndex { base, indices, value, .. } => {
                all_reads_expr(base, out);
                indices.iter().for_each(|i| all_reads_expr(i, out));
                all_reads_expr(value, out);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                all_reads_expr(cond, out);
                all_reads(then_body, out);
                all_reads(else_body, out);
            }
            Stmt::While { cond, body, .. } => {
                all_reads_expr(cond, out);
                all_reads(body, out);
            }
            Stmt::For { bound, body, .. } => {
                all_reads_expr(bound, out);
                all_reads(body, out);
            }
            _ => {}
        }
    }
}

/// The `on device` blocks of a body, outermost only, in order.
fn find_regions<'b>(body: &'b [Stmt], out: &mut Vec<(Vec<(String, Span)>, &'b [Stmt], Span)>) {
    for s in body {
        match s {
            Stmt::If { cond, then_body, span, .. } if device_arrays(cond).is_some() => out.push((device_arrays(cond).unwrap(), then_body, *span)),
            Stmt::If { then_body, else_body, .. } => {
                find_regions(then_body, out);
                find_regions(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => find_regions(body, out),
            _ => {}
        }
    }
}

struct RegionEmit<'a> {
    e: Emit<'a>,
    kinds: &'a HashMap<String, KInfo>,
    arrays: Vec<String>,
}

impl RegionEmit<'_> {
    fn launch(&self, callee: &str, args: &[Expr], target: Option<&str>) -> Result<String, String> {
        let k = &self.kinds[callee];
        let mut a = Vec::new();
        if k.elementwise.is_some() {
            let t = target.ok_or_else(|| format!("a call of the elementwise kernel `{callee}` with no target"))?;
            a.push(format!("{c}, {c}_len", c = cname(t)));
        }
        for ((_, t), x) in k.params.iter().zip(args) {
            match (t, x) {
                (Ty::Arr | Ty::IntArr, Expr::Var { name, .. }) => a.push(format!("{c}, {c}_len", c = cname(name))),
                (Ty::Arr | Ty::IntArr, _) => return Err(format!("an array argument of `{callee}` that is not a device array")),
                _ => a.push(self.scalar(x)?),
            }
        }
        Ok(format!("VIV_CUDA_CHECK(viv_dlaunch_{callee}({}));", a.join(", ")))
    }

    fn scalar(&self, x: &Expr) -> Result<String, String> {
        let c = self.e.expr(x);
        if c.contains("not in the kernel subset") {
            return Err("an expression the host code cannot write".into());
        }
        Ok(c)
    }

    fn stmts(&mut self, body: &[Stmt], ind: usize, out: &mut String) -> Result<(), String> {
        let pad = "    ".repeat(ind);
        for s in body {
            match s {
                _ if declared_name(s).is_some() => {}
                Stmt::Assign { name, value: Expr::Call { callee, args, .. }, .. } if self.kinds.contains_key(callee) => {
                    out.push_str(&format!("{pad}{}\n", self.launch(callee, args, Some(name))?));
                }
                Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } if self.kinds.contains_key(callee) => {
                    out.push_str(&format!("{pad}{}\n", self.launch(callee, args, None)?));
                }
                Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } if callee == crate::tuples::UNPACK && args.len() == 2 => {
                    let names = |e: &Expr| -> Vec<String> {
                        match e {
                            Expr::ArrLit { elems, .. } | Expr::Call { args: elems, .. } => elems.iter().filter_map(|x| if let Expr::Var { name, .. } = x { Some(name.clone()) } else { None }).collect(),
                            _ => Vec::new(),
                        }
                    };
                    match &args[1] {
                        Expr::Call { callee, args: kargs, .. } if self.kinds.contains_key(callee) => {
                            out.push_str(&format!("{pad}{}\n", self.launch(callee, kargs, None)?));
                        }
                        Expr::Call { callee, .. } if callee == crate::tuples::TUPLE => {
                            let (t, sv) = (names(&args[0]), names(&args[1]));
                            if t.is_empty() || t.len() != sv.len() || !t.iter().all(|n| self.arrays.contains(n)) {
                                return Err("an unpacking that is not a swap of device arrays".into());
                            }
                            let ty = |n: &str| self.e.arrays.get(n).copied().unwrap_or(CTy::Double);
                            out.push_str(&format!("{pad}{{\n"));
                            for (i, src) in sv.iter().enumerate() {
                                let et = if ty(src) == CTy::Long { "long long" } else { "double" };
                                out.push_str(&format!("{pad}    {et} *viv_p{i} = {c}; long long viv_l{i} = {c}_len;\n", c = cname(src)));
                            }
                            for (i, tg) in t.iter().enumerate() {
                                out.push_str(&format!("{pad}    {c} = viv_p{i}; {c}_len = viv_l{i};\n", c = cname(tg)));
                            }
                            out.push_str(&format!("{pad}}}\n"));
                        }
                        _ => return Err("an unpacking the host code cannot write".into()),
                    }
                }
                Stmt::Assign { name, value, .. } => out.push_str(&format!("{pad}{} = {};\n", cname(name), self.scalar(value)?)),
                Stmt::If { cond, then_body, else_body, .. } => {
                    out.push_str(&format!("{pad}if ({}) {{\n", self.scalar(cond)?));
                    self.stmts(then_body, ind + 1, out)?;
                    if !else_body.is_empty() {
                        out.push_str(&format!("{pad}}} else {{\n"));
                        self.stmts(else_body, ind + 1, out)?;
                    }
                    out.push_str(&format!("{pad}}}\n"));
                }
                Stmt::While { cond, body, .. } => {
                    out.push_str(&format!("{pad}while ({}) {{\n", self.scalar(cond)?));
                    self.stmts(body, ind + 1, out)?;
                    out.push_str(&format!("{pad}}}\n"));
                }
                Stmt::For { var, bound, body, .. } => {
                    self.e.n += 1;
                    let b = format!("viv_bound{}", self.e.n);
                    let v = cname(var);
                    out.push_str(&format!("{pad}{{\n{pad}    long long {b} = (long long)({});\n{pad}    for ({v} = 0; {v} < {b}; {v}++) {{\n", self.scalar(bound)?));
                    self.stmts(body, ind + 2, out)?;
                    out.push_str(&format!("{pad}    }}\n{pad}}}\n"));
                }
                Stmt::Break { .. } => out.push_str(&format!("{pad}break;\n")),
                Stmt::Continue { .. } => out.push_str(&format!("{pad}continue;\n")),
                Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Narrate { .. } => {}
                _ => return Err("a statement the host code cannot write (only loops, `if`, scalar assignments, kernel calls and swaps)".into()),
            }
        }
        Ok(())
    }
}

/// With `-DVIV_REGION_TIMING`, each block's host function times its body (the
/// launches, not the copies at its ends) with CUDA events into `viv_region_ms`.
const TIMING_DECL: &str = "\n#ifdef VIV_REGION_TIMING\n/* the time of the last `on device` block's body, copies excluded, in ms */\nfloat viv_region_ms = 0.0f;\n#endif\n";
const TIMING_START: &str = "#ifdef VIV_REGION_TIMING\n    cudaEvent_t viv_t0, viv_t1;\n    VIV_CUDA_CHECK(cudaEventCreate(&viv_t0));\n    VIV_CUDA_CHECK(cudaEventCreate(&viv_t1));\n    VIV_CUDA_CHECK(cudaEventRecord(viv_t0));\n#endif\n";
const TIMING_STOP: &str = "#ifdef VIV_REGION_TIMING\n    VIV_CUDA_CHECK(cudaEventRecord(viv_t1));\n    VIV_CUDA_CHECK(cudaEventSynchronize(viv_t1));\n    VIV_CUDA_CHECK(cudaEventElapsedTime(&viv_region_ms, viv_t0, viv_t1));\n    cudaEventDestroy(viv_t0);\n    cudaEventDestroy(viv_t1);\n#endif\n";

/// The host function of each `on device` block in `body` (of the def
/// `owner`, or `main` for the top level): copies in, the body with its
/// launches on device pointers, one synchronization, copies out.
#[allow(clippy::too_many_arguments)]
fn emit_regions(owner: &str, params: &[Param], body: &[Stmt], kinds: &HashMap<String, KInfo>, consts: &HashMap<String, (f64, bool)>, whole: &HashMap<(u32, u32), i64>, src: &str, out: &mut String, skipped: &mut Vec<String>) {
    let mut regions = Vec::new();
    find_regions(body, &mut regions);
    // the scalars of the owner whose types are written down
    let mut scal: Vec<(String, CTy)> = Vec::new();
    for p in params {
        match param_ty(p) {
            Ok(Ty::Num) => scal.push((p.name.clone(), CTy::Double)),
            Ok(Ty::Int) => scal.push((p.name.clone(), CTy::Long)),
            Ok(Ty::Bool) => scal.push((p.name.clone(), CTy::Bool)),
            _ => {}
        }
    }
    fn declared(body: &[Stmt], out: &mut Vec<(String, CTy)>) {
        for s in body {
            if let (Some(n), Stmt::ExprStmt { expr: Expr::Call { args, .. }, .. }) = (declared_name(s), s) {
                let t = match args.get(1) {
                    Some(Expr::Str { value, .. }) if value == "Int" => CTy::Long,
                    Some(Expr::Str { value, .. }) if value == "Bool" => CTy::Bool,
                    _ => CTy::Double,
                };
                out.push((n.to_string(), t));
            }
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    declared(then_body, out);
                    declared(else_body, out);
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } => declared(body, out),
                _ => {}
            }
        }
    }
    declared(body, &mut scal);
    for (n, (arrays, rbody, span)) in regions.into_iter().enumerate() {
        let line = line_of(src, span);
        let fname = format!("viv_region_{owner}_{}", n + 1);
        let names: Vec<String> = arrays.iter().map(|(a, _)| a.clone()).collect();
        // each array's element type, from the kernels it is passed to
        let mut ety: HashMap<String, CTy> = names.iter().map(|a| (a.clone(), CTy::Double)).collect();
        fn elem_types(body: &[Stmt], kinds: &HashMap<String, KInfo>, ety: &mut HashMap<String, CTy>) {
            fn visit(callee: &str, args: &[Expr], kinds: &HashMap<String, KInfo>, ety: &mut HashMap<String, CTy>) {
                if let Some(k) = kinds.get(callee) {
                    for ((_, t), a) in k.params.iter().zip(args) {
                        if let (Ty::IntArr, Expr::Var { name, .. }) = (t, a) {
                            ety.insert(name.clone(), CTy::Long);
                        }
                    }
                }
            }
            for s in body {
                match s {
                    Stmt::Assign { value: Expr::Call { callee, args, .. }, .. } | Stmt::ExprStmt { expr: Expr::Call { callee, args, .. }, .. } => {
                        visit(callee, args, kinds, ety);
                        if let [_, Expr::Call { callee: c2, args: a2, .. }] = args.as_slice() {
                            visit(c2, a2, kinds, ety);
                        }
                    }
                    Stmt::If { then_body, else_body, .. } => {
                        elem_types(then_body, kinds, ety);
                        elem_types(else_body, kinds, ety);
                    }
                    Stmt::While { body, .. } | Stmt::For { body, .. } => elem_types(body, kinds, ety),
                    _ => {}
                }
            }
        }
        elem_types(rbody, kinds, &mut ety);
        let mut e = Emit { vars: BTreeMap::new(), arrays: ety.clone(), consts, whole, n: 0 };
        // the scalars the block reads and does not set: its parameters
        let mut assigned = Vec::new();
        rbody.iter().for_each(|s| assigned_in(s, &mut assigned));
        let mut rs = Vec::new();
        all_reads(rbody, &mut rs);
        let mut free: Vec<(String, CTy)> = Vec::new();
        let mut unknown = None;
        for r in rs {
            if names.contains(&r) || assigned.contains(&r) || free.iter().any(|(f, _)| *f == r) {
                continue;
            }
            match scal.iter().find(|(s, _)| *s == r) {
                Some((_, t)) => free.push((r.clone(), *t)),
                None if consts.contains_key(&r) => {}
                None => unknown = Some(r),
            }
        }
        if let Some(u) = unknown {
            skipped.push(format!("the `on device` block at line {line} ({owner}): `{u}` has no type the emitter can see"));
            continue;
        }
        free.sort_by_key(|(f, _)| params.iter().position(|p| p.name == *f).unwrap_or(usize::MAX));
        for (f, t) in &free {
            e.vars.insert(f.clone(), *t);
        }
        e.infer(rbody);
        for a in &names {
            e.vars.remove(a);
        }
        let skip: HashSet<String> = free.iter().map(|(f, _)| f.clone()).collect();
        let decls = e.decls(&skip, 1);
        let mut re = RegionEmit { e, kinds, arrays: names.clone() };
        let mut code = String::new();
        if let Err(why) = re.stmts(rbody, 1, &mut code) {
            skipped.push(format!("the `on device` block at line {line} ({owner}): {why}"));
            continue;
        }
        let mut sig = Vec::new();
        let mut enter = String::new();
        let mut leave = String::from("    VIV_CUDA_CHECK(cudaDeviceSynchronize());\n");
        let mut free_all = String::new();
        for a in &names {
            let c = cname(a);
            let et = if ety[a] == CTy::Long { "long long" } else { "double" };
            sig.push(format!("{et} *h_{a}, long long h_{a}_len"));
            enter.push_str(&format!(
                "    {et} *{c} = NULL;\n    long long {c}_len = h_{a}_len;\n    VIV_CUDA_CHECK(cudaMalloc((void **)&{c}, sizeof({et}) * (size_t)(h_{a}_len > 0 ? h_{a}_len : 1)));\n    VIV_CUDA_CHECK(cudaMemcpy({c}, h_{a}, sizeof({et}) * (size_t)h_{a}_len, cudaMemcpyHostToDevice));\n"
            ));
            leave.push_str(&format!(
                "    if ({c}_len != h_{a}_len) return -1;\n    VIV_CUDA_CHECK(cudaMemcpy(h_{a}, {c}, sizeof({et}) * (size_t)h_{a}_len, cudaMemcpyDeviceToHost));\n"
            ));
            free_all.push_str(&format!("    cudaFree({c});\n"));
        }
        for (f, t) in &free {
            let ct = match t {
                CTy::Long => "long long",
                CTy::Bool => "bool",
                _ => "double",
            };
            sig.push(format!("{ct} {}", cname(f)));
        }
        let k = names.len();
        if !out.contains(TIMING_DECL) {
            out.push_str(TIMING_DECL);
        }
        out.push_str(&format!(
            "\n// ---- on device {} (line {line}, in {owner}): {k} copies to the device on entry, {k} back on exit,\n// none in between; each h_ array comes back holding the array of its name at the end\nextern \"C\" int {fname}({}) {{\n{enter}{decls}{TIMING_START}{code}{TIMING_STOP}{leave}{free_all}    return 0;\n}}\n",
            names.join(", "),
            sig.join(", ")
        ));
    }
}
