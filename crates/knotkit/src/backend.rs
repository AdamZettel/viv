// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Backends (2.3) and the faithful C backend (2.6).
//!
//! A backend consumes an [`IrProgram`] and produces an [`Emission`]: the target
//! artifact plus a [`DebugMap`] recording where each IR node landed. The map is
//! what makes 2.6 ("each IR construct maps to a target construct via a known
//! rule") concrete, and what downstream tooling (3.3) consumes.
//!
//! The C backend owns a tiny C AST (expressions, statements, functions) and
//! renders it faithfully with position tracking. Knowing what C looks like is
//! intrinsic to a C backend, so this is not language-specific over-fitting. What
//! *is* language-specific — which C construct each IR node becomes, and what
//! runtime the language needs — is delegated to the [`CEmit`] trait, so the
//! backend stays generic over the source vocabulary and reusable by any DSL.

use crate::diag::Diagnostic;
use crate::io::IoEnvironment;
use crate::ir::{IrProgram, Node, NodeId, NodeKind};
use crate::span::Span;

/// What every backend produces.
pub struct Emission {
    pub artifact: Artifact,
    pub debug_map: DebugMap,
}

pub enum Artifact {
    CSource(String),
    WasmModule(Vec<u8>),
}

impl Artifact {
    /// The C source, if this is a C emission.
    pub fn as_c(&self) -> Option<&str> {
        match self {
            Artifact::CSource(s) => Some(s),
            _ => None,
        }
    }
}

/// A NodeId-keyed record of where each IR node's text/bytes landed in the
/// artifact. Multiple entries may share a node: a deduped subexpression emitted
/// in two places appears twice, which is the faithful answer.
#[derive(Default)]
pub struct DebugMap {
    pub entries: Vec<DebugEntry>,
}

pub struct DebugEntry {
    pub node: NodeId,
    pub span: Span,
    pub target: TargetLoc,
}

pub enum TargetLoc {
    /// C backend: 1-based line/column where the node's text begins. Exact.
    CText { line: u32, col: u32 },
    /// Direct WASM backend: byte offset in the code section. Exact.
    WasmOffset(u32),
    /// C→clang→wasm: a DWARF range. Approximate under -O3 (see design note).
    WasmRange { lo: u32, hi: u32 },
}

/// A backend over a particular IR vocabulary `K`. Generic over the source IR
/// (2.3); I/O is injected (1.8); errors are returned as data (1.7).
pub trait Backend<K: NodeKind> {
    fn name(&self) -> &str;
    fn emit(
        &self,
        program: &IrProgram<K>,
        io: &dyn IoEnvironment,
    ) -> Result<Emission, Vec<Diagnostic>>;
}

// ───────────────────────────── the C AST ───────────────────────────────────
//
// A minimal, faithful subset of C. Every node optionally carries the `NodeId`
// it originated from, so the renderer can populate the DebugMap (2.6).

#[derive(Clone, Debug)]
pub enum CType {
    /// `int64_t` (requires <stdint.h>).
    Int,
    /// `double`.
    Double,
    /// Any C type the language names itself (e.g. a struct/typedef declared in
    /// its prelude). This is what keeps the backend's type vocabulary open
    /// rather than Knot-shaped — a DSL whose values are `oa_op` uses
    /// `CType::Named("oa_op".into())`.
    Named(String),
}

impl CType {
    pub fn c_name(&self) -> &str {
        match self {
            CType::Int => "int64_t",
            CType::Double => "double",
            CType::Named(s) => s,
        }
    }

    /// A zero initializer valid for this type. Scalars take `0`; a named
    /// (possibly struct) type takes the universal aggregate zero `{0}`.
    pub fn zero_init(&self) -> &'static str {
        match self {
            CType::Int | CType::Double => "0",
            CType::Named(_) => "{0}",
        }
    }
}

pub struct CExpr {
    pub origin: Option<NodeId>,
    pub kind: CExprKind,
}

pub enum CExprKind {
    /// A literal token, already formatted as valid C (e.g. `"3.14159"`).
    Lit(String),
    /// A verbatim C expression the language composes itself — an escape hatch
    /// for target constructs the kit's C AST doesn't model (e.g. a compound
    /// literal for a language-specific closure type).
    Raw(String),
    /// A variable reference.
    Var(String),
    Unary { op: String, operand: Box<CExpr> },
    Binary { lhs: Box<CExpr>, op: String, rhs: Box<CExpr> },
    Call { callee: String, args: Vec<CExpr> },
    /// A lazy conditional expression: `(<cond> ? <then> : <els>)`. Only the taken
    /// branch is evaluated (needed by languages where `if` is an expression).
    Ternary { cond: Box<CExpr>, then: Box<CExpr>, els: Box<CExpr> },
    /// A comma sequence `(a, b, c)`: evaluates each in order, yields the last.
    Seq(Vec<CExpr>),
    /// Struct member access: `<base>.<field>`.
    Field { base: Box<CExpr>, field: String },
    /// A struct compound literal with designated initializers:
    /// `((<ty>){ .<field> = <expr>, ... })`. Field order is irrelevant to C.
    StructLit { ty: String, fields: Vec<(String, CExpr)> },
}

impl CExpr {
    pub fn lit(origin: NodeId, text: impl Into<String>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Lit(text.into()) }
    }
    pub fn raw(origin: NodeId, text: impl Into<String>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Raw(text.into()) }
    }
    pub fn var(origin: NodeId, name: impl Into<String>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Var(name.into()) }
    }
    pub fn unary(origin: NodeId, op: impl Into<String>, operand: CExpr) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Unary { op: op.into(), operand: Box::new(operand) } }
    }
    pub fn binary(origin: NodeId, lhs: CExpr, op: impl Into<String>, rhs: CExpr) -> Self {
        Self {
            origin: Some(origin),
            kind: CExprKind::Binary { lhs: Box::new(lhs), op: op.into(), rhs: Box::new(rhs) },
        }
    }
    pub fn call(origin: NodeId, callee: impl Into<String>, args: Vec<CExpr>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Call { callee: callee.into(), args } }
    }
    pub fn field(origin: NodeId, base: CExpr, field: impl Into<String>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Field { base: Box::new(base), field: field.into() } }
    }
    pub fn ternary(origin: NodeId, cond: CExpr, then: CExpr, els: CExpr) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Ternary { cond: Box::new(cond), then: Box::new(then), els: Box::new(els) } }
    }
    pub fn seq(origin: NodeId, parts: Vec<CExpr>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::Seq(parts) }
    }
    pub fn struct_lit(origin: NodeId, ty: impl Into<String>, fields: Vec<(String, CExpr)>) -> Self {
        Self { origin: Some(origin), kind: CExprKind::StructLit { ty: ty.into(), fields } }
    }
}

pub struct CStmt {
    pub origin: Option<NodeId>,
    pub kind: CStmtKind,
}

pub enum CStmtKind {
    /// Declare and initialize a local: `<ty> <name> = <init>;`.
    Decl { ty: CType, name: String, init: CExpr },
    /// Reassign an existing local: `<name> = <value>;`.
    Assign { name: String, value: CExpr },
    /// `return <expr>;`.
    Return(CExpr),
    /// An expression evaluated for effect: `<expr>;`.
    Expr(CExpr),
    /// Emits nothing — for source constructs with no runtime effect (e.g. a
    /// docstring or a narration the compiled program drops).
    Nop,
    /// `if (<cond>) { <then> } else { <else_body> }` (else omitted if empty).
    If { cond: CExpr, then_body: Vec<CStmt>, else_body: Vec<CStmt> },
    /// `while (<cond>) { <body> }`.
    While { cond: CExpr, body: Vec<CStmt> },
    /// `for (<init>; <cond>; <step>) { <body> }`. `init`/`step` are simple
    /// (Decl/Assign) and render inline in the loop header.
    For { init: Box<CStmt>, cond: CExpr, step: Box<CStmt>, body: Vec<CStmt> },
    Break,
    Continue,
    /// A braced compound statement `{ ... }`.
    Block(Vec<CStmt>),
    /// An OpenMP-parallel counted loop. Emits `#pragma omp parallel for` over an
    /// integer counter, assigns the (double) loop `var` from it, and privatizes
    /// per-iteration temporaries. The pragma is a no-op without `-fopenmp`, so
    /// the same C runs serial (e.g. wasm) or parallel (native).
    ParallelFor {
        var: String,
        bound: CExpr,
        body: Vec<CStmt>,
        /// Variables to make thread-private (per-iteration temporaries).
        privates: Vec<String>,
        /// Cross-iteration reductions, as (operator, variable).
        reductions: Vec<(String, String)>,
    },
}

impl CStmt {
    pub fn decl(origin: NodeId, ty: CType, name: impl Into<String>, init: CExpr) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Decl { ty, name: name.into(), init } }
    }
    pub fn assign(origin: NodeId, name: impl Into<String>, value: CExpr) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Assign { name: name.into(), value } }
    }
    pub fn ret(origin: NodeId, expr: CExpr) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Return(expr) }
    }
    pub fn expr(origin: NodeId, expr: CExpr) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Expr(expr) }
    }
    pub fn nop() -> Self {
        Self { origin: None, kind: CStmtKind::Nop }
    }
    pub fn block(origin: NodeId, stmts: Vec<CStmt>) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Block(stmts) }
    }
    pub fn parallel_for(
        origin: NodeId,
        var: impl Into<String>,
        bound: CExpr,
        body: Vec<CStmt>,
        privates: Vec<String>,
        reductions: Vec<(String, String)>,
    ) -> Self {
        Self {
            origin: Some(origin),
            kind: CStmtKind::ParallelFor { var: var.into(), bound, body, privates, reductions },
        }
    }
    pub fn if_(origin: NodeId, cond: CExpr, then_body: Vec<CStmt>, else_body: Vec<CStmt>) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::If { cond, then_body, else_body } }
    }
    pub fn while_(origin: NodeId, cond: CExpr, body: Vec<CStmt>) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::While { cond, body } }
    }
    pub fn for_(origin: NodeId, init: CStmt, cond: CExpr, step: CStmt, body: Vec<CStmt>) -> Self {
        Self {
            origin: Some(origin),
            kind: CStmtKind::For { init: Box::new(init), cond, step: Box::new(step), body },
        }
    }
    pub fn brk(origin: NodeId) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Break }
    }
    pub fn cont(origin: NodeId) -> Self {
        Self { origin: Some(origin), kind: CStmtKind::Continue }
    }
}

/// A C function with a statement-list body. `locals` are declared (zero-init)
/// at function entry — the kit's support for languages with function-scoped
/// variables, where a binding introduced in a branch must outlive it.
pub struct CFunc {
    pub origin: Option<NodeId>,
    pub ret: CType,
    pub name: String,
    pub params: Vec<(CType, String)>,
    pub locals: Vec<(CType, String)>,
    pub body: Vec<CStmt>,
}

/// A top-level C item: a file-scope global, a function, or the program body
/// that becomes `int main(void)`.
pub enum CItem {
    /// A user-defined value struct, emitted as a typedef before globals and
    /// functions so they can hold/pass it by value. Fields render in order.
    Struct { name: String, fields: Vec<(CType, String)> },
    /// A file-scope variable, emitted before functions so they can read it.
    /// C zero-initializes file-scope variables.
    Global { ty: CType, name: String },
    Func(CFunc),
    Main { locals: Vec<(CType, String)>, body: Vec<CStmt> },
    /// Declarations a language writes out as C text for one program (its
    /// tagged unions and their helpers, say): emitted after the runtime and
    /// before the structs, globals and functions, which may use them.
    Decls(String),
}

/// A named piece of a language's C runtime. Fragments are emitted on demand —
/// only those a program transitively references appear in the output — which
/// keeps generated code (and the eventual WASM) lean: a program using one
/// vector does not drag in tensor-contraction code.
pub struct RuntimeFragment {
    pub name: &'static str,
    /// `#include` lines this fragment needs (deduped across all emitted ones).
    pub includes: &'static [&'static str],
    /// Other fragments this one depends on (emitted before it).
    pub deps: &'static [&'static str],
    /// The C source for this fragment (typically one or more `static` defs).
    pub code: &'static str,
}

/// A language's catalog of runtime fragments. Each language brings its own
/// runtime (1.6, A.3); the kit imposes none.
#[derive(Default)]
pub struct Runtime {
    fragments: Vec<RuntimeFragment>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, fragment: RuntimeFragment) -> Self {
        self.fragments.push(fragment);
        self
    }
    fn get(&self, name: &str) -> Option<&RuntimeFragment> {
        self.fragments.iter().find(|f| f.name == name)
    }
    /// Every fragment of the catalog, for a language's own checks of its
    /// runtime (which functions it defines, and what they return).
    pub fn fragments(&self) -> &[RuntimeFragment] {
        &self.fragments
    }
}

/// How a language turns its IR nodes into C constructs. The backend drives the
/// walk; the language answers, per node, "what C is this?" (2.6, faithful).
pub trait CEmit: NodeKind {
    /// The language's runtime catalog. The backend emits only the fragments
    /// marked needed (via [`CEmitCx::need`]) during emission, plus their
    /// transitive dependencies. Default: no runtime.
    fn runtime() -> Runtime {
        Runtime::new()
    }

    /// Text written verbatim at the very top of the emitted file, before the
    /// runtime's `#include`s. It is where a language states translation-unit
    /// policy the kit has no opinion on -- a floating-point pragma, say.
    /// Default: nothing.
    fn preamble() -> &'static str {
        ""
    }

    /// Text written after the runtime's `#include`s and before its fragments:
    /// the first point at which every system header has been read, where a
    /// language can rename or forbid C library names for the rest of the
    /// file (a `#pragma GCC poison`, say). Default: nothing.
    fn after_includes() -> &'static str {
        ""
    }

    /// Called by [`CBackend::emit_split`] before the items of a translation
    /// unit are emitted and again before its runtime is rendered: `None` is
    /// the program's own unit, `Some(name)` a separately compiled one. A
    /// language that numbers things per program (source sites, literals)
    /// numbers them per unit from here on. Default: nothing.
    fn enter_unit(_unit: Option<&str>) {}

    /// A runtime fragment's code as it goes into a translation unit of
    /// `role`. Under [`UnitRole::Single`] it must be the code unchanged; for
    /// the split roles a language rewrites its file-scope state so that the
    /// program's unit defines it once and every other unit refers to it.
    /// Default: unchanged.
    fn unit_code(code: &'static str, _role: UnitRole) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(code)
    }

    /// Whether a program with no top-level statements still gets an empty
    /// `int main(void)`, so that a module of definitions alone links as a
    /// program. Default: no (the emission is then a library).
    fn always_main() -> bool {
        false
    }

    /// Emit a top-level item (function, or a program-level statement) for a root.
    fn emit_item(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CItem;

    /// Emit a statement for this node (used inside function/main bodies).
    fn emit_stmt(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CStmt;

    /// Emit a C expression for this node.
    fn emit_expr(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CExpr;
}

/// Context handed to a language during C emission. It owns the program so the
/// language can recurse into children by id without holding its own borrows,
/// and collects which runtime fragments the emission needs.
pub struct CEmitCx<'a, K: NodeKind> {
    program: &'a IrProgram<K>,
    needs: std::collections::HashSet<&'static str>,
}

impl<'a, K: CEmit> CEmitCx<'a, K> {
    /// Mark a runtime fragment as needed. The backend emits it (and its deps)
    /// exactly once, in dependency order.
    pub fn need(&mut self, fragment: &'static str) {
        self.needs.insert(fragment);
    }

    /// Emit the C statement for a child node, by id.
    pub fn stmt(&mut self, id: NodeId) -> CStmt {
        let node = self.program.get(id).clone();
        K::emit_stmt(&node, self)
    }

    /// The children of a node, by id (cloned so callers hold no borrow). Useful
    /// for emitting a block: map its children through [`Self::stmt`].
    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.program.get(id).children.clone()
    }

    /// The kind of a node, by id (cloned) — lets an item emitter inspect its
    /// children structurally (e.g. to find marker statements) before lowering.
    pub fn kind(&self, id: NodeId) -> K {
        self.program.get(id).kind.clone()
    }

    /// The whole program being emitted -- for facts an item needs about the
    /// others (what a callee returns, how a record is laid out).
    pub fn program(&self) -> &'a IrProgram<K> {
        self.program
    }

    /// Emit the C expression for a child node, by id.
    pub fn expr(&mut self, id: NodeId) -> CExpr {
        // Clone so this doesn't hold an immutable borrow of `self.program` while
        // passing `&mut self` along (cheap for v1).
        let node = self.program.get(id).clone();
        K::emit_expr(&node, self)
    }
}

// ───────────────────────────── the C backend ───────────────────────────────

pub struct CBackend;

impl<K: CEmit> Backend<K> for CBackend {
    fn name(&self) -> &str {
        "c"
    }

    fn emit(
        &self,
        program: &IrProgram<K>,
        _io: &dyn IoEnvironment,
    ) -> Result<Emission, Vec<Diagnostic>> {
        let mut cx = CEmitCx { program, needs: std::collections::HashSet::new() };
        let mut globals: Vec<(CType, String)> = Vec::new();
        let mut funcs = Vec::new();
        let mut structs: Vec<(String, Vec<(CType, String)>)> = Vec::new();
        let mut main: Option<(Vec<(CType, String)>, Vec<CStmt>)> = None;
        let mut decls: Vec<String> = Vec::new();
        for &root in program.roots() {
            let node = program.get(root).clone();
            match K::emit_item(&node, &mut cx) {
                CItem::Struct { name, fields } => structs.push((name, fields)),
                CItem::Global { ty, name } => globals.push((ty, name)),
                CItem::Func(f) => funcs.push(f),
                CItem::Main { locals, body } => main = Some((locals, body)),
                CItem::Decls(text) => decls.push(text),
            }
        }

        let mut w = CWriter::new();
        w.user_fns = funcs.iter().map(|f| f.name.clone()).collect();
        w.write(K::preamble());
        render_runtime_with(&mut w, &K::runtime(), &cx.needs, &|c| std::borrow::Cow::Borrowed(c), K::after_includes());
        for d in &decls {
            w.write(d);
            w.write("\n");
        }
        // User struct typedefs come before globals/functions that hold them.
        for (name, fields) in &structs {
            w.write("typedef struct {\n");
            for (ty, fname) in fields {
                w.write("    ");
                w.write(ty.c_name());
                w.write(" ");
                w.write(&escape_ident(fname));
                w.write(";\n");
            }
            w.write("} ");
            w.write(name);
            w.write(";\n");
        }
        if !structs.is_empty() {
            w.write("\n");
        }
        for (ty, name) in &globals {
            w.write(ty.c_name());
            w.write(" ");
            w.write(&escape_ident(name));
            w.write(";\n");
        }
        if !globals.is_empty() {
            w.write("\n");
        }
        // Prototypes first: function ORDER in the source is a presentation
        // choice, not a dependency constraint. Forward declarations make the
        // emitted C order-independent, matching the interpreter.
        if funcs.len() > 1 {
            for f in &funcs {
                render_prototype(&mut w, f);
            }
            w.write("\n");
        }
        for f in &funcs {
            render_func(&mut w, f);
            w.write("\n");
        }
        if main.is_none() && K::always_main() {
            main = Some((Vec::new(), Vec::new()));
        }
        if let Some((locals, body)) = &main {
            w.write("int main(void) {\n");
            render_locals(&mut w, locals, 1);
            for s in body {
                render_stmt(&mut w, s, 1);
            }
            w.write("    return 0;\n}\n");
        }

        let entries = w
            .marks
            .iter()
            .map(|&(node, line, col)| DebugEntry {
                node,
                span: program.get(node).span,
                target: TargetLoc::CText { line, col },
            })
            .collect();

        Ok(Emission {
            artifact: Artifact::CSource(w.buf),
            debug_map: DebugMap { entries },
        })
    }
}

/// The role of a translation unit in a split emission
/// ([`CBackend::emit_split`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnitRole {
    /// The whole program in one file, as [`Backend::emit`] writes it.
    Single,
    /// The program's own unit of a split emission: it defines the runtime's
    /// state, the globals and `main`.
    Main,
    /// A separately compiled unit: some of the program's functions, the
    /// runtime they use, and declarations of what they reach in other units.
    Unit,
}

/// A separately compiled translation unit of a split emission.
pub struct CUnit {
    /// The language's name for it (a module's path, say).
    pub name: String,
    /// Complete C: it compiles on its own, to an object linked with the
    /// program's unit. Its text depends only on the functions it holds and
    /// what they use, so equal text may be compiled once and reused.
    pub text: String,
}

/// The identifiers of a C text (an underestimate of nothing: every maximal
/// run of identifier characters that does not start with a digit).
fn c_idents(text: &str) -> std::collections::HashSet<&str> {
    c_words(text, false)
}

/// The identifiers of a C text that can name something at file scope: those
/// of `c_idents` less the member names, the words right after `.` or `->`
/// (`o.method`, `p->tag`, `.method = x` in an initializer). A struct field
/// `method` is not the program's global `method` (B260).
fn c_refs(text: &str) -> std::collections::HashSet<&str> {
    c_words(text, true)
}

fn c_words(text: &str, skip_members: bool) -> std::collections::HashSet<&str> {
    let mut out = std::collections::HashSet::new();
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let j0 = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let member = skip_members && {
                let mut k = j0;
                while k > 0 && (b[k - 1] == b' ' || b[k - 1] == b'\t' || b[k - 1] == b'\n') {
                    k -= 1;
                }
                k > 0 && (b[k - 1] == b'.' || (b[k - 1] == b'>' && k > 1 && b[k - 2] == b'-'))
            };
            if !member {
                out.insert(&text[j0..i]);
            }
        } else if c.is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
                i += 1;
            }
        } else if c == b'"' {
            // skip a string literal: its words are not references
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

impl CBackend {
    /// The program as several translation units: the functions of the roots
    /// that `unit_of` assigns to a unit go into that unit, everything else
    /// into the program's own unit, returned first. The program's unit holds
    /// `main`, the globals, the runtime's state and every function `unit_of`
    /// leaves to it, with prototypes for all; each other unit holds its
    /// functions with the runtime fragments they use (its state declared
    /// `extern` by [`CEmit::unit_code`]), the struct typedefs, enum
    /// declarations and prototypes they reach, and the globals they read as
    /// `extern`. The language is told which unit is current
    /// ([`CEmit::enter_unit`]) while it emits a unit's items and renders its
    /// runtime. Only functions can be placed in a unit; another kind of item
    /// stays in the program's unit.
    pub fn emit_split<K: CEmit>(
        &self,
        program: &IrProgram<K>,
        unit_of: &dyn Fn(NodeId) -> Option<String>,
    ) -> Result<(String, Vec<CUnit>), Vec<Diagnostic>> {
        use std::collections::{BTreeMap, HashSet};
        let mut cx = CEmitCx { program, needs: HashSet::new() };
        let mut globals: Vec<(CType, String)> = Vec::new();
        let mut funcs: Vec<CFunc> = Vec::new();
        let mut structs: Vec<(String, Vec<(CType, String)>)> = Vec::new();
        let mut main: Option<(Vec<(CType, String)>, Vec<CStmt>)> = None;
        let mut decls: Vec<String> = Vec::new();
        // the roots by unit, in program order within each
        let mut groups: BTreeMap<String, Vec<NodeId>> = BTreeMap::new();
        let mut own: Vec<NodeId> = Vec::new();
        for &root in program.roots() {
            match unit_of(root) {
                Some(u) => groups.entry(u).or_default().push(root),
                None => own.push(root),
            }
        }
        let mut place = |item: CItem, funcs: &mut Vec<CFunc>| match item {
            CItem::Struct { name, fields } => structs.push((name, fields)),
            CItem::Global { ty, name } => globals.push((ty, name)),
            CItem::Func(f) => funcs.push(f),
            CItem::Main { locals, body } => main = Some((locals, body)),
            CItem::Decls(text) => decls.push(text),
        };
        K::enter_unit(None);
        for root in own {
            let node = program.get(root).clone();
            let item = K::emit_item(&node, &mut cx);
            place(item, &mut funcs);
        }
        let mut units: Vec<(String, Vec<CFunc>, HashSet<&'static str>)> = Vec::new();
        for (name, roots) in &groups {
            K::enter_unit(Some(name));
            let mut ufuncs = Vec::new();
            let mut uneeds = HashSet::new();
            for &root in roots {
                let node = program.get(root).clone();
                let before = std::mem::take(&mut cx.needs);
                let item = K::emit_item(&node, &mut cx);
                let got = std::mem::replace(&mut cx.needs, before);
                match item {
                    CItem::Func(f) => {
                        uneeds.extend(got.iter().copied());
                        ufuncs.push(f);
                    }
                    other => place(other, &mut funcs),
                }
                // the program's unit defines the state of every fragment any unit uses
                cx.needs.extend(got);
            }
            units.push((name.clone(), ufuncs, uneeds));
        }

        let all_protos = |w: &mut CWriter, funcs: &[CFunc], units: &[(String, Vec<CFunc>, HashSet<&'static str>)]| {
            for f in funcs.iter().chain(units.iter().flat_map(|u| u.1.iter())) {
                render_prototype(w, f);
            }
        };

        // the program's unit
        K::enter_unit(None);
        let mut w = CWriter::new();
        w.user_fns = funcs.iter().chain(units.iter().flat_map(|u| u.1.iter())).map(|f| f.name.clone()).collect();
        w.write("#define VIV_UNITS_MAIN 1\n");
        w.write(K::preamble());
        render_runtime_with(&mut w, &K::runtime(), &cx.needs, &|c| K::unit_code(c, UnitRole::Main), K::after_includes());
        for d in &decls {
            w.write(d);
            w.write("\n");
        }
        for (name, fields) in &structs {
            render_struct(&mut w, name, fields);
        }
        for (ty, name) in &globals {
            w.write(ty.c_name());
            w.write(" ");
            w.write(&escape_ident(name));
            w.write(";\n");
        }
        all_protos(&mut w, &funcs, &units);
        w.write("\n");
        for f in &funcs {
            render_func(&mut w, f);
            w.write("\n");
        }
        if main.is_none() && K::always_main() {
            main = Some((Vec::new(), Vec::new()));
        }
        if let Some((locals, body)) = &main {
            w.write("int main(void) {\n");
            render_locals(&mut w, locals, 1);
            for s in body {
                render_stmt(&mut w, s, 1);
            }
            w.write("    return 0;\n}\n");
        }
        let main_text = w.buf;

        // what the declarations define: their words, less the runtime's
        let runtime_words: HashSet<String> = {
            let rt = K::runtime();
            rt.fragments().iter().flat_map(|f| c_idents(f.code)).map(str::to_string).collect()
        };
        let decl_ids: HashSet<String> = decls.iter().flat_map(|d| c_idents(d)).filter(|i| !runtime_words.contains(*i) && i.starts_with("viv_")).map(str::to_string).collect();

        // the other units
        let user_names: Vec<String> = funcs.iter().chain(units.iter().flat_map(|u| u.1.iter())).map(|f| f.name.clone()).collect();
        let mut out = Vec::new();
        for (name, ufuncs, uneeds) in &units {
            K::enter_unit(Some(name));
            // the functions' text first: what it names decides the rest
            let mut body = CWriter::new();
            body.user_fns = user_names.iter().cloned().collect();
            // what each function names, less its own parameters and locals
            // (a parameter `v` is not the program's global `v`)
            let mut reached: HashSet<String> = HashSet::new();
            for f in ufuncs {
                let at = body.buf.len();
                render_func(&mut body, f);
                body.write("\n");
                let own: HashSet<String> = f.params.iter().chain(f.locals.iter()).map(|(_, n)| escape_ident(n).into_owned()).collect();
                for id in c_refs(&body.buf[at..]) {
                    if !own.contains(id) {
                        reached.insert(id.to_string());
                    }
                }
            }
            // the types the prototypes and the globals they need name
            let mut decl_words = String::new();
            for f in funcs.iter().chain(units.iter().flat_map(|u| u.1.iter())) {
                if reached.contains(escape_ident(&f.name).as_ref()) {
                    decl_words.push_str(f.ret.c_name());
                    decl_words.push(' ');
                    for (ty, _) in &f.params {
                        decl_words.push_str(ty.c_name());
                        decl_words.push(' ');
                    }
                }
            }
            for (ty, gname) in &globals {
                if reached.contains(escape_ident(gname).as_ref()) {
                    decl_words.push_str(ty.c_name());
                    decl_words.push(' ');
                }
            }
            for id in c_idents(&decl_words) {
                reached.insert(id.to_string());
            }
            // struct typedefs, and the structs their fields hold, to a fixed point
            let mut want_structs: Vec<bool> = vec![false; structs.len()];
            loop {
                let mut grew = false;
                for (k, (sname, fields)) in structs.iter().enumerate() {
                    if !want_structs[k] && reached.contains(sname.as_str()) {
                        want_structs[k] = true;
                        grew = true;
                        for (ty, _) in fields {
                            for id in c_idents(ty.c_name()) {
                                reached.insert(id.to_string());
                            }
                        }
                    }
                }
                if !grew {
                    break;
                }
            }
            // a language's declarations (its enums) belong to the program's
            // unit; a unit that names one is refused (the language keeps such
            // functions in the program's unit)
            if let Some(id) = reached.iter().find(|i| decl_ids.contains(i.as_str())) {
                return Err(vec![Diagnostic::error(format!(
                    "the unit `{name}` uses `{id}`, which the program's declarations define; it cannot be compiled apart"
                ))]);
            }
            // the declarations the unit's functions need from elsewhere
            let mut head = CWriter::new();
            for (k, (sname, fields)) in structs.iter().enumerate() {
                if want_structs[k] {
                    render_struct(&mut head, sname, fields);
                }
            }
            for (ty, gname) in &globals {
                let g = escape_ident(gname);
                if reached.contains(g.as_ref()) {
                    head.write("extern ");
                    head.write(ty.c_name());
                    head.write(" ");
                    head.write(&g);
                    head.write(";\n");
                }
            }
            for f in funcs.iter().chain(units.iter().flat_map(|u| u.1.iter())) {
                if reached.contains(escape_ident(&f.name).as_ref()) {
                    render_prototype(&mut head, f);
                }
            }
            // a type those declarations name (a callee's parameter, say) comes
            // with the runtime fragment that defines it
            let rt = K::runtime();
            let mut needs = uneeds.clone();
            let head_ids = c_idents(&head.buf);
            for f in rt.fragments() {
                if typedef_names(f.code).iter().any(|t| head_ids.contains(t)) {
                    needs.insert(f.name);
                }
            }
            let mut w = CWriter::new();
            w.write("#define VIV_UNIT 1\n");
            w.write(K::preamble());
            render_runtime_with(&mut w, &rt, &needs, &|c| K::unit_code(c, UnitRole::Unit), K::after_includes());
            w.write(&head.buf);
            w.write("\n");
            w.write(&body.buf);
            out.push(CUnit { name: name.clone(), text: w.buf });
        }
        K::enter_unit(None);
        Ok((main_text, out))
    }
}

/// The names a C text's `typedef`s define: the last word before each
/// typedef's closing `;` outside braces.
fn typedef_names(code: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(k) = code[from..].find("typedef") {
        let start = from + k;
        let ok_before = start == 0 || !code.as_bytes()[start - 1].is_ascii_alphanumeric() && code.as_bytes()[start - 1] != b'_';
        let mut depth = 0i32;
        let mut end = None;
        for (i, ch) in code[start..].char_indices() {
            match ch {
                '{' | '(' => depth += 1,
                '}' | ')' => depth -= 1,
                ';' if depth == 0 => {
                    end = Some(start + i);
                    break;
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        if ok_before {
            let decl = &code[start..end];
            // `typedef T (*name)(...)`: the name is inside the first parentheses
            let word_src = if let Some(p) = decl.find("(*") { &decl[p + 2..] } else { decl };
            let word: &str = if decl.contains("(*") {
                word_src.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).next().unwrap_or("")
            } else {
                word_src.rsplit(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).find(|w| !w.is_empty()).unwrap_or("")
            };
            if !word.is_empty() {
                out.push(word);
            }
        }
        from = end + 1;
    }
    out
}

fn render_struct(w: &mut CWriter, name: &str, fields: &[(CType, String)]) {
    w.write("typedef struct {\n");
    for (ty, fname) in fields {
        w.write("    ");
        w.write(ty.c_name());
        w.write(" ");
        w.write(&escape_ident(fname));
        w.write(";\n");
    }
    w.write("} ");
    w.write(name);
    w.write(";\n");
}

fn render_prototype(w: &mut CWriter, f: &CFunc) {
    w.write(f.ret.c_name());
    w.write(" ");
    w.write(&escape_ident(&f.name));
    w.write("(");
    for (i, (ty, name)) in f.params.iter().enumerate() {
        if i > 0 {
            w.write(", ");
        }
        w.write(ty.c_name());
        w.write(" ");
        w.write(&escape_ident(name));
    }
    if f.params.is_empty() {
        w.write("void");
    }
    w.write(");\n");
}

/// Appends C text while tracking 1-based line/column, and records the position
/// at which each origin-tagged node's text begins.
struct CWriter {
    buf: String,
    line: u32,
    col: u32,
    marks: Vec<(NodeId, u32, u32)>,
    /// Functions this module defines. A callee is escaped only if it is one of
    /// these: `fmod` as a user `def` becomes `fmod_`, while the `fmod` the
    /// lowering emits for Viv's `mod` must stay `fmod` or it calls nothing.
    user_fns: std::collections::HashSet<String>,
}

impl CWriter {
    fn new() -> Self {
        Self {
            buf: String::new(),
            line: 1,
            col: 1,
            marks: Vec::new(),
            user_fns: std::collections::HashSet::new(),
        }
    }

    /// Render a callee: escaped when it is one of this module's own functions,
    /// verbatim when it is a runtime or libm symbol.
    fn callee(&self, name: &str) -> String {
        if self.user_fns.contains(name) {
            escape_ident(name).into_owned()
        } else {
            name.to_string()
        }
    }

    fn write(&mut self, s: &str) {
        for ch in s.chars() {
            if ch == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.buf.push(ch);
        }
    }

    fn indent(&mut self, levels: usize) {
        for _ in 0..levels {
            self.write("    ");
        }
    }

    fn mark(&mut self, id: NodeId) {
        self.marks.push((id, self.line, self.col));
    }
}

/// C reserved identifiers a user name must not collide with when emitted into C:
/// the C keywords, plus libc/libm symbols that read naturally as variable names
/// but are declared (as functions) in headers the runtime includes -- notably the
/// Bessel functions `y0`/`y1`/`yn`/`j0`/`j1`/`jn`, which are extremely common
/// numerical variable names. The math builtins (`exp`, `sqrt`, `floor`, ...) are
/// here too: a builtin call is emitted verbatim, and only a user's variable or
/// def of that name is renamed.
fn is_reserved_c_ident(name: &str) -> bool {
    matches!(
        name,
        // C89/C99 keywords
        "auto" | "break" | "case" | "char" | "const" | "continue" | "default" | "do"
            | "double" | "else" | "enum" | "extern" | "float" | "for" | "goto" | "if"
            | "inline" | "int" | "long" | "register" | "restrict" | "return" | "short"
            | "signed" | "sizeof" | "static" | "struct" | "switch" | "typedef" | "union"
            | "unsigned" | "void" | "volatile" | "while" | "_Bool" | "_Complex" | "_Imaginary"
            // libm Bessel functions (math.h) -- the common numerical-name collisions
            | "y0" | "y1" | "yn" | "j0" | "j1" | "jn"
            // other short libc/libm names that read naturally as variables
            | "abs" | "labs" | "div" | "ldiv" | "gamma" | "drem"
            // math.h names the LANGUAGE does not use, so they are free as Viv
            // variables and taken in C. `fmax` in particular: Viv spells the
            // builtin `max`, so `fmax = 0.0` is an ordinary user variable that
            // used to emit `double fmax;` on top of libm's declaration.
            | "fmax" | "fmin" | "fmod" | "fdim" | "fma" | "hypot" | "cbrt"
            | "trunc" | "rint" | "nearbyint" | "copysign" | "remainder" | "remquo"
            | "erfc" | "lgamma" | "tgamma" | "frexp" | "ldexp" | "modf"
            | "logb" | "ilogb" | "scalbn" | "scalbln" | "log1p" | "log2" | "log10"
            | "expm1" | "exp2" | "nan" | "nextafter" | "nexttoward"
            // complex.h (included whenever a program uses a complex scalar):
            // its functions and its MACROS. `I` expands to `_Complex_I`, so a
            // Viv variable `I` emitted as `viv_arr I = ...` did not compile.
            | "creal" | "cimag" | "cabs" | "carg" | "cproj" | "conj" | "cexp" | "clog"
            | "csqrt" | "cpow" | "csin" | "ccos" | "ctan"
            | "I" | "complex" | "imaginary" | "_Complex_I" | "_Imaginary_I" | "CMPLX"
            // the program's entry point: a user `def main` must not be C's main
            | "main"
            // C99/C11 keywords and stdbool/stddef/stdint/stdio/math/limits/float
            // macros and types that read naturally as names
            | "_Alignas" | "_Alignof" | "_Atomic" | "_Generic" | "_Noreturn"
            | "_Static_assert" | "_Thread_local" | "bool" | "true" | "false"
            // C23 keywords, and the extension keywords GCC and Clang accept in
            // their default (gnu) modes: `asm = "x"` emitted `asm` as a
            // variable and did not compile (B345)
            | "alignas" | "alignof" | "constexpr" | "nullptr" | "static_assert"
            | "thread_local" | "typeof" | "typeof_unqual" | "_BitInt" | "_Decimal32"
            | "_Decimal64" | "_Decimal128" | "_Pragma" | "asm" | "fortran"
            | "noreturn" | "unix" | "linux" | "i386"
            | "NULL" | "EOF" | "FILE" | "size_t" | "ptrdiff_t" | "int64_t" | "uint64_t"
            | "int32_t" | "uint32_t" | "uint8_t" | "intptr_t" | "assert"
            | "INFINITY" | "NAN" | "HUGE_VAL" | "M_PI" | "M_E" | "M_SQRT2" | "M_LN2"
            | "isnan" | "isinf" | "isfinite" | "signbit" | "fpclassify" | "isnormal"
            | "INT_MAX" | "INT_MIN" | "LONG_MAX" | "DBL_MAX" | "DBL_MIN" | "DBL_EPSILON"
            | "RAND_MAX" | "EXIT_SUCCESS" | "EXIT_FAILURE" | "BUFSIZ" | "EDOM" | "ERANGE"
            // stdlib.h / stdio.h names that read naturally as variables
            | "atof" | "atoi" | "atol" | "free" | "malloc" | "calloc" | "realloc"
            | "qsort" | "bsearch" | "system" | "getenv" | "abort" | "exit"
            | "printf" | "scanf" | "puts" | "remove" | "rename"
            | "strdup" | "memcpy" | "strlen" | "random" | "srandom" | "stdin" | "stdout"
            | "stderr" | "errno"
            // unistd.h (the runtime includes it for _exit), strings.h (through
            // string.h), signal.h and select.h: each of these, as a variable,
            // was a C compile error (study/compare/records/state.viv's `acct`)
            | "acct" | "access" | "alarm" | "brk" | "chdir" | "chown" | "chroot" | "close"
            | "dup" | "dup2" | "execl" | "execv" | "fchdir" | "fork" | "fsync" | "ftruncate"
            | "getcwd" | "gethostname" | "getlogin" | "getopt" | "getpid" | "isatty"
            | "link" | "lockf" | "lseek" | "nice" | "optarg" | "opterr" | "optind" | "optopt"
            | "pathconf" | "pause" | "pipe" | "read" | "readlink" | "rmdir" | "sbrk"
            | "sethostname" | "setsid" | "sleep" | "swab" | "symlink" | "sync" | "sysconf"
            | "truncate" | "ttyname" | "unlink" | "usleep" | "vfork" | "write" | "crypt"
            | "encrypt" | "index" | "rindex" | "bzero" | "bcopy" | "ffs" | "select" | "signal"
            // setjmp.h and signal.h, which an exported program's runtime
            // includes (lang/viv/src/embed.c)
            | "setjmp" | "longjmp" | "jmp_buf" | "sig_atomic_t" | "raise"
            // math.h names that are also Viv builtins: a builtin call is emitted
            // verbatim (`callee` escapes only user functions), so a variable of
            // the same name can take the `_`
            | "exp" | "log" | "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "atan2" | "sinh" | "cosh" | "tanh" | "erf" | "floor" | "ceil" | "round"
            | "pow" | "fma" | "fabs" | "rand" | "srand"
            // the rest of the C standard library's functions in the headers
            // the runtime includes (stdio.h, stdlib.h, string.h, time.h,
            // stdarg.h): a user def of one of these names, emitted as is,
            // clashed with the declaration or, through a wrapper that called
            // it by name, called libc's instead (B347: `def index` exported
            // through vipyr ran strings.h's index)
            | "fopen" | "fclose" | "fflush" | "freopen" | "fread" | "fwrite" | "fgetc" | "fgets"
            | "fputc" | "fputs" | "getc" | "getchar" | "gets" | "putc" | "putchar" | "ungetc"
            | "fprintf" | "sprintf" | "snprintf" | "vprintf" | "vfprintf" | "vsprintf" | "vsnprintf"
            | "fscanf" | "sscanf" | "fseek" | "ftell" | "rewind" | "fgetpos" | "fsetpos" | "feof"
            | "ferror" | "clearerr" | "perror" | "tmpfile" | "tmpnam" | "setbuf" | "setvbuf"
            | "fileno" | "popen" | "pclose" | "getline" | "dprintf"
            | "strtod" | "strtof" | "strtold" | "strtol" | "strtoll" | "strtoul" | "strtoull"
            | "atoll" | "llabs" | "lldiv" | "atexit" | "at_quick_exit" | "quick_exit" | "_Exit"
            | "mblen" | "mbtowc" | "wctomb" | "mbstowcs" | "wcstombs" | "aligned_alloc"
            | "posix_memalign" | "setenv" | "unsetenv" | "putenv" | "mkstemp" | "mktemp"
            | "realpath" | "drand48" | "erand48" | "lrand48" | "mrand48" | "srand48" | "seed48"
            | "arc4random"
            | "strcpy" | "strncpy" | "strcat" | "strncat" | "strcmp" | "strncmp" | "strcoll"
            | "strxfrm" | "strchr" | "strrchr" | "strspn" | "strcspn" | "strpbrk" | "strstr"
            | "strtok" | "strerror" | "strnlen" | "strndup" | "strsep" | "strcasecmp"
            | "strncasecmp" | "stpcpy" | "memmove" | "memcmp" | "memchr" | "memset"
            | "clock" | "time" | "difftime" | "mktime" | "strftime" | "gmtime" | "localtime"
            | "asctime" | "ctime" | "timegm" | "nanosleep" | "clock_gettime"
            | "va_start" | "va_end" | "va_arg" | "va_copy" | "va_list"
    )
}

/// Render a user identifier safely: append `_` if it would collide with a C
/// keyword or a libc/libm symbol, so e.g. the variable `y0` becomes `y0_` in C.
/// The mapping is deterministic, so definitions and uses stay consistent.
pub fn escape_ident(name: &str) -> std::borrow::Cow<'_, str> {
    if is_reserved_c_ident(name) {
        std::borrow::Cow::Owned(format!("{name}_"))
    } else {
        std::borrow::Cow::Borrowed(name)
    }
}

fn render_func(w: &mut CWriter, f: &CFunc) {
    if let Some(id) = f.origin {
        w.mark(id);
    }
    w.write(f.ret.c_name());
    w.write(" ");
    w.write(&escape_ident(&f.name));
    w.write("(");
    if f.params.is_empty() {
        w.write("void");
    } else {
        for (i, (ty, name)) in f.params.iter().enumerate() {
            if i > 0 {
                w.write(", ");
            }
            w.write(ty.c_name());
            w.write(" ");
            w.write(&escape_ident(name));
        }
    }
    w.write(") {\n");
    render_locals(w, &f.locals, 1);
    for s in &f.body {
        render_stmt(w, s, 1);
    }
    w.write("}\n");
}

/// Emit only the runtime fragments the program transitively needs: dependency
/// order, with `#include`s deduped and hoisted to the top. This is the on-demand
/// emission that keeps generated code (and WASM) lean. `after` (a language's
/// [`CEmit::after_includes`]) is written between the `#include`s and the
/// fragments; with no fragments needed, `after` alone.
fn render_runtime_with(
    w: &mut CWriter,
    runtime: &Runtime,
    needs: &std::collections::HashSet<&'static str>,
    code_of: &dyn Fn(&'static str) -> std::borrow::Cow<'static, str>,
    after: &str,
) {
    // Resolve the transitive closure in dependency order (deps before dependents).
    let mut order: Vec<&RuntimeFragment> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    // Deterministic: visit needed fragments in sorted order.
    let mut roots: Vec<&&'static str> = needs.iter().collect();
    roots.sort_unstable();
    for name in roots {
        visit_fragment(name, runtime, &mut seen, &mut order);
    }
    if order.is_empty() {
        w.write(after);
        return;
    }

    // Deduped includes, in first-seen order.
    let mut includes: Vec<&str> = Vec::new();
    for frag in &order {
        for inc in frag.includes {
            if !includes.contains(inc) {
                includes.push(inc);
            }
        }
    }
    for inc in &includes {
        w.write(inc);
        w.write("\n");
    }
    if !includes.is_empty() {
        w.write("\n");
    }
    w.write(after);
    for frag in &order {
        if frag.code.is_empty() {
            continue; // include-only fragment (e.g. just pulls a header)
        }
        let code = code_of(frag.code);
        w.write(&code);
        if !code.ends_with('\n') {
            w.write("\n");
        }
        w.write("\n");
    }
}

fn visit_fragment<'r>(
    name: &str,
    runtime: &'r Runtime,
    seen: &mut std::collections::HashSet<&'r str>,
    order: &mut Vec<&'r RuntimeFragment>,
) {
    let Some(frag) = runtime.get(name) else {
        return; // unknown fragment names are silently ignored
    };
    if !seen.insert(frag.name) {
        return;
    }
    for dep in frag.deps {
        visit_fragment(dep, runtime, seen, order);
    }
    order.push(frag);
}

/// Declare hoisted locals at the top of a body, zero-initialized.
fn render_locals(w: &mut CWriter, locals: &[(CType, String)], indent: usize) {
    for (ty, name) in locals {
        w.indent(indent);
        w.write(ty.c_name());
        w.write(" ");
        w.write(&escape_ident(name));
        w.write(" = ");
        w.write(ty.zero_init());
        w.write(";\n");
    }
}

fn render_stmt(w: &mut CWriter, s: &CStmt, indent: usize) {
    if matches!(s.kind, CStmtKind::Nop) {
        return; // emits nothing, not even indentation
    }
    w.indent(indent);
    if let Some(id) = s.origin {
        w.mark(id);
    }
    match &s.kind {
        CStmtKind::Decl { ty, name, init } => {
            w.write(ty.c_name());
            w.write(" ");
            w.write(&escape_ident(name));
            w.write(" = ");
            render_expr(w, init);
            w.write(";\n");
        }
        CStmtKind::Assign { name, value } => {
            w.write(&escape_ident(name));
            w.write(" = ");
            render_expr(w, value);
            w.write(";\n");
        }
        CStmtKind::Return(e) => {
            w.write("return ");
            render_expr(w, e);
            w.write(";\n");
        }
        CStmtKind::Expr(e) => {
            render_expr(w, e);
            w.write(";\n");
        }
        CStmtKind::If { cond, then_body, else_body } => {
            w.write("if (");
            render_cond(w, cond);
            w.write(") {\n");
            for s in then_body {
                render_stmt(w, s, indent + 1);
            }
            w.indent(indent);
            w.write("}");
            if !else_body.is_empty() {
                w.write(" else {\n");
                for s in else_body {
                    render_stmt(w, s, indent + 1);
                }
                w.indent(indent);
                w.write("}");
            }
            w.write("\n");
        }
        CStmtKind::While { cond, body } => {
            w.write("while (");
            render_cond(w, cond);
            w.write(") {\n");
            for s in body {
                render_stmt(w, s, indent + 1);
            }
            w.indent(indent);
            w.write("}\n");
        }
        CStmtKind::For { init, cond, step, body } => {
            w.write("for (");
            render_simple_inline(w, init);
            w.write("; ");
            render_cond(w, cond);
            w.write("; ");
            render_simple_inline(w, step);
            w.write(") {\n");
            for s in body {
                render_stmt(w, s, indent + 1);
            }
            w.indent(indent);
            w.write("}\n");
        }
        CStmtKind::Break => w.write("break;\n"),
        CStmtKind::Continue => w.write("continue;\n"),
        CStmtKind::Block(stmts) => {
            w.write("{\n");
            for s in stmts {
                render_stmt(w, s, indent + 1);
            }
            w.indent(indent);
            w.write("}\n");
        }
        CStmtKind::ParallelFor { var, bound, body, privates, reductions } => {
            // #pragma omp parallel for private(<var>, <privates>) reduction(...)
            w.write("#pragma omp parallel for");
            let mut privs: Vec<&str> = vec![var.as_str()];
            for p in privates {
                if !privs.contains(&p.as_str()) {
                    privs.push(p);
                }
            }
            w.write(" private(");
            for (i, p) in privs.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                w.write(p);
            }
            w.write(")");
            for (op, v) in reductions {
                w.write(" reduction(");
                w.write(op);
                w.write(":");
                w.write(v);
                w.write(")");
            }
            // Dynamic scheduling: lets fast cores grab more chunks so slow cores
            // (e.g. Apple Silicon efficiency cores) or uneven per-iteration work
            // don't become the critical path.
            w.write(" schedule(dynamic)");
            w.write("\n");
            w.indent(indent);
            // Integer canonical loop (OMP requires an integer counter); the
            // double loop variable is assigned from it inside the body.
            let idx = format!("knot_omp_{var}");
            w.write(&format!("for (long {idx} = 0; {idx} < (long)("));
            render_expr(w, bound);
            w.write(&format!("); {idx}++) {{\n"));
            w.indent(indent + 1);
            w.write(&format!("{var} = (double){idx};\n"));
            for s in body {
                render_stmt(w, s, indent + 1);
            }
            w.indent(indent);
            w.write("}\n");
        }
        CStmtKind::Nop => {} // handled by the early return above
    }
}

/// Render a Decl/Assign without indentation or a trailing `;` — for a `for`
/// loop header's init and step clauses.
fn render_simple_inline(w: &mut CWriter, s: &CStmt) {
    if let Some(id) = s.origin {
        w.mark(id);
    }
    match &s.kind {
        CStmtKind::Decl { ty, name, init } => {
            w.write(ty.c_name());
            w.write(" ");
            w.write(&escape_ident(name));
            w.write(" = ");
            render_expr(w, init);
        }
        CStmtKind::Assign { name, value } => {
            w.write(&escape_ident(name));
            w.write(" = ");
            render_expr(w, value);
        }
        _ => panic!("for-loop header must be a declaration or assignment"),
    }
}

/// Render an expression in condition position (`if (...)`, `while (...)`, the
/// `for` test). The C control-flow syntax already supplies the surrounding
/// parentheses, so a top-level binary is rendered without its own outer pair —
/// both cleaner and free of clang's `-Wparentheses-equality` warning on `==`.
fn render_cond(w: &mut CWriter, e: &CExpr) {
    if let CExprKind::Binary { lhs, op, rhs } = &e.kind {
        if let Some(id) = e.origin {
            w.mark(id);
        }
        render_expr(w, lhs);
        w.write(" ");
        w.write(op);
        w.write(" ");
        render_expr(w, rhs);
    } else {
        render_expr(w, e);
    }
}

fn render_expr(w: &mut CWriter, e: &CExpr) {
    if let Some(id) = e.origin {
        w.mark(id);
    }
    match &e.kind {
        CExprKind::Lit(s) | CExprKind::Raw(s) => w.write(s),
        CExprKind::Var(s) => w.write(&escape_ident(s)),
        // Parenthesize the operand so adjacent operator tokens cannot merge
        // (e.g. `-(-x)` must not render as `--x`, which C reads as predecrement).
        CExprKind::Unary { op, operand } => {
            w.write(op);
            w.write("(");
            render_expr(w, operand);
            w.write(")");
        }
        // Fully parenthesized: unambiguous and semantics-preserving (faithful).
        CExprKind::Binary { lhs, op, rhs } => {
            w.write("(");
            render_expr(w, lhs);
            w.write(" ");
            w.write(op);
            w.write(" ");
            render_expr(w, rhs);
            w.write(")");
        }
        CExprKind::Call { callee, args } => {
            // A user function's call site is escaped exactly as its definition
            // was; a runtime or libm callee (`viv_*`, `exp`, `fmod`) is emitted
            // verbatim even when the same spelling is reserved for variables.
            let rendered = w.callee(callee);
            w.write(&rendered);
            w.write("(");
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                render_expr(w, a);
            }
            w.write(")");
        }
        CExprKind::Ternary { cond, then, els } => {
            w.write("(");
            render_expr(w, cond);
            w.write(" ? ");
            render_expr(w, then);
            w.write(" : ");
            render_expr(w, els);
            w.write(")");
        }
        CExprKind::Seq(parts) => {
            w.write("(");
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                render_expr(w, p);
            }
            w.write(")");
        }
        CExprKind::Field { base, field } => {
            render_expr(w, base);
            w.write(".");
            w.write(&escape_ident(field));
        }
        CExprKind::StructLit { ty, fields } => {
            w.write("((");
            w.write(ty);
            w.write("){ ");
            for (i, (name, value)) in fields.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                w.write(".");
                w.write(&escape_ident(name));
                w.write(" = ");
                render_expr(w, value);
            }
            w.write(" })");
        }
    }
}
