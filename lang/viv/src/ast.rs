// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv's abstract syntax tree, aligned with the real Viv surface. This is the
//! transient parser output; it never escapes the parser-to-IR boundary (1.1).

use knotkit::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    And,
    Or,
    /// Matrix product (`@`). Always operates on arrays.
    Matmul,
    /// `a % b`: floored modulo, Python's rule for floats: the result takes the
    /// sign of `b`. Appended after `Matmul` so every earlier tag is unchanged.
    Mod,
    /// `a // b`: floor division, Python's rule for floats; `a == (a // b) * b + a % b`.
    FloorDiv,
}

impl BinOp {
    /// The C operator for the scalar form. Not meaningful for `Matmul` (which
    /// lowers to a runtime call, never a C operator).
    pub fn c_op(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::EqEq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::Matmul => "@",
            // lowered to runtime calls (viv_floormod / viv_floordiv), never a C operator
            BinOp::Mod => "%",
            BinOp::FloorDiv => "//",
        }
    }
    pub fn tag(self) -> u8 {
        self as u8
    }
}

pub struct Program {
    pub items: Vec<Item>,
    /// The debug lens written in the source: `debug { ... }` blocks, and the
    /// `.debug` file a driver appended as such blocks. They are not code: parsing keeps them here, apart from the
    /// items, and only `crate::debug_lens::instrument` turns them into the
    /// `Stmt::Lens` statements a debug build runs. Everything that reads the
    /// program as code (the body hashes, the doc tools) never sees them.
    pub debug: Vec<DebugBlock>,
    /// The `parallel for` loops a memory lens pins as disjoint
    /// (`crate::memory_lens`), as
    /// `crate::parse` found them: checked builds test each while it runs.
    /// Like the debug lens, not code: the body hashes never see them.
    pub memory: Vec<crate::parallel::Pinned>,
    /// The phrases the parser read (`crate::phrases`): where each is, the
    /// pattern it matched and the std defs its lowering calls, which
    /// `crate::parse` adds to the program. The phrases themselves are already
    /// ordinary calls in `items`; this is what the views and `viv phrases`
    /// read to show them as written.
    pub phrases: Vec<crate::phrases::PhraseUse>,
    /// Where the integer literals are, as (start, end) of their spans, with
    /// their exact values: the number literals written with no `.` and no
    /// exponent (`3`, not `3.0`) that fit in 64 bits (`Expr::Num` holds the
    /// nearest double, which past 2^53 is not the same number). Such a literal is
    /// an Int where an Int is expected, and a number everywhere else.
    /// Kept beside the tree, not in it, so
    /// every pass that reads a literal reads it as it did, and the tree a
    /// program without Ints prints is the tree it printed before.
    pub whole: std::collections::HashMap<(u32, u32), i64>,
}

// Written out so that `--emit ast` prints a program without a debug lens
// exactly as the derived impl did before the field existed.
impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Program");
        d.field("items", &self.items);
        if !self.debug.is_empty() {
            d.field("debug", &self.debug);
        }
        if !self.memory.is_empty() {
            d.field("memory", &self.memory);
        }
        if !self.phrases.is_empty() {
            d.field("phrases", &self.phrases);
        }
        d.finish()
    }
}

/// One `debug { ... }` block: what to watch, what must hold and what to
/// trace, for one def (`debug { }` directly before it, or `debug name { }`
/// anywhere at the top level) or for the whole program (`debug * { }`).
#[derive(Clone, Debug)]
pub struct DebugBlock {
    /// the def the block is about; `None` for `debug * { }`
    pub def: Option<String>,
    pub def_span: Span,
    pub items: Vec<DebugItem>,
    pub span: Span,
}

/// Where a watch or a hold is taken.
#[derive(Clone, Debug, PartialEq)]
pub enum DebugPos {
    /// `per iteration`, or nothing: at the end of every iteration of the
    /// def's iteration loops
    PerIteration,
    /// `at <step>`: at the end of that step
    At(String, Span),
}

#[derive(Clone, Debug)]
pub enum DebugItem {
    /// `watch [per iteration | at <step>] a, b, name = expr`. An item
    /// `name = expr` watches a value the code does not hold: the lens
    /// computes it there, in a debug build only, and later watches and
    /// holds of the def may read it by its name. `exprs[k]` is item k's
    /// expression, or `None` for a variable of the code.
    Watch { pos: DebugPos, names: Vec<(String, Span)>, exprs: Vec<Option<Expr>>, span: Span },
    /// `hold [per iteration | at <step>] <expr> [within <tol>] "<sentence>"`
    Hold { pos: DebugPos, check: HoldCheck, sentence: String, span: Span },
    /// `trace f, g`: record every call of these defs in the story
    Trace { names: Vec<(String, Span)>, span: Span },
}

/// What a `hold` checks.
#[derive(Clone, Debug)]
pub enum HoldCheck {
    /// a boolean expression that must be true
    Cond(Expr),
    /// `a == b within tol`: the largest |a - b| (over the elements, for
    /// arrays) must be at most `tol`. `diff` is `a - b`, built by the parser.
    Near { diff: Expr, tol: Expr },
}

/// A statement the debug lens adds to a def's body (`crate::debug_lens`), or
/// the body of a `placeholder def`. Only a debug build runs the lens ones:
/// the interpreter leaves them out of a release run, and the C backend guards
/// them with `VIV_STORY`, which `-DVIV_RELEASE` turns off. `def` is always the
/// def's name as written, so a defun specialization records its source def.
#[derive(Clone, Debug)]
pub enum Lens {
    /// The first statement of a def the story records calls of: records the
    /// call; both engines record the return on every way out of the def.
    Call { def: String },
    /// The first statement of an iteration loop's body. It adds one to
    /// `counter` (a variable the lens set to 0 just before the loop) and
    /// records the iteration's number. `end` runs at the end of every
    /// iteration, however the body is left (normally, `continue`, `break` or
    /// `return`), after the steps the body left open are closed: the
    /// per-iteration watches and holds, then `IterEnd`.
    Iter { def: String, counter: String, end: Vec<Stmt> },
    /// The last statement of an iteration's `end`.
    IterEnd { def: String },
    /// Record the values of `names` at `at` (a step id, or `iteration`).
    /// `vars`, when not empty, holds the variable each name is read from
    /// (one per name): a def inlined by `crate::grad` has its locals renamed
    /// apart, and its watches still record them under the names the source
    /// gives them.
    Watch { def: String, at: String, names: Vec<(String, Span)>, vars: Vec<String> },
    /// Check a hold at `at`. `sentence` is already JSON-escaped, so both
    /// engines copy the same bytes into the story.
    Hold { def: String, at: String, check: HoldCheck, sentence: String },
    /// The whole body of `placeholder def name(...)`: calling it is an error
    /// that names it.
    Placeholder { def: String },
    /// Stop the run with `message`, located at this statement, in every
    /// engine. The parser inserts it where sugar needs a run-time check, like
    /// `for i from a to b by s` with a step that is 0. `kind` is the error's
    /// kind (crate::errkind), the same in every engine.
    Stop { message: String, kind: crate::errkind::ErrKind },
    /// Where the body of a def inlined by `crate::grad` begins (`enter`) and
    /// ends, in the gradient def it was inlined into. Unlike the lens
    /// statements above, `crate::grad` writes these in every build: the
    /// interpreter counts the region as a call of `def` in its call graph,
    /// the compiled engines run nothing for them, and the debug lens
    /// instruments the region as a call of `def` (a `Call` after the enter
    /// marker, a `Ret` before the exit), so a gradient's forward pass shows
    /// the defs it was written in. `names` (on the enter marker) maps each
    /// name the def's source uses to the name the region gives it.
    Inline { def: String, enter: bool, names: Vec<(String, String)> },
    /// The return of a region `Inline` marks, recorded like the return of a
    /// call of `def`.
    Ret { def: String },
    /// `name = value` for a watch written `name = expr`. It lives only while
    /// `crate::debug_lens::instrument` works on a def: the last thing it does
    /// is turn each one into `if not release_mode() { name = value }`, so no
    /// engine sees it, and a release build computes nothing.
    Bind { name: String, value: Expr },
}

/// The element of a packed array: how it is stored.
/// Float kinds read as a `Float32` and round to nearest even on a store;
/// integer kinds read as an `Int` and check the range on a store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Store {
    F32,
    BF16,
    F16,
    I8,
    I16,
    I32,
    U8,
}

impl Store {
    pub const ALL: [Store; 7] = [Store::F32, Store::BF16, Store::F16, Store::I8, Store::I16, Store::I32, Store::U8];
    /// The type's name in source: `Float32`, `BFloat16`, ...
    pub fn name(self) -> &'static str {
        match self {
            Store::F32 => "Float32",
            Store::BF16 => "BFloat16",
            Store::F16 => "Float16",
            Store::I8 => "Int8",
            Store::I16 => "Int16",
            Store::I32 => "Int32",
            Store::U8 => "UInt8",
        }
    }
    /// The conversion function's name: the type's name in lower case.
    pub fn conv(self) -> &'static str {
        match self {
            Store::F32 => "float32",
            Store::BF16 => "bfloat16",
            Store::F16 => "float16",
            Store::I8 => "int8",
            Store::I16 => "int16",
            Store::I32 => "int32",
            Store::U8 => "uint8",
        }
    }
    /// `<conv>_zeros`, the constructor.
    pub fn zeros(self) -> &'static str {
        match self {
            Store::F32 => "float32_zeros",
            Store::BF16 => "bfloat16_zeros",
            Store::F16 => "float16_zeros",
            Store::I8 => "int8_zeros",
            Store::I16 => "int16_zeros",
            Store::I32 => "int32_zeros",
            Store::U8 => "uint8_zeros",
        }
    }
    pub fn of_name(s: &str) -> Option<Store> {
        Store::ALL.into_iter().find(|k| k.name() == s)
    }
    pub fn of_conv(s: &str) -> Option<Store> {
        Store::ALL.into_iter().find(|k| k.conv() == s)
    }
    pub fn of_zeros(s: &str) -> Option<Store> {
        Store::ALL.into_iter().find(|k| k.zeros() == s)
    }
    /// Bytes per element.
    pub fn width(self) -> usize {
        match self {
            Store::F32 | Store::I32 => 4,
            Store::BF16 | Store::F16 | Store::I16 => 2,
            Store::I8 | Store::U8 => 1,
        }
    }
    /// Reads as a Float32 (else as an Int).
    pub fn is_float(self) -> bool {
        matches!(self, Store::F32 | Store::BF16 | Store::F16)
    }
    /// A small code the runtimes pass along: the position in `ALL`.
    pub fn code(self) -> u8 {
        Store::ALL.iter().position(|k| *k == self).unwrap_or(0) as u8
    }
    pub fn of_code(c: u8) -> Store {
        Store::ALL.get(c as usize).copied().unwrap_or(Store::F32)
    }
}

/// A type annotation written in source: `Num`, `Arr`, or a record type name.
#[derive(Clone, Debug)]
pub enum TypeExpr {
    Num,
    Arr,
    /// `CArr`: an array of complex numbers.
    CArr,
    /// `Bool`: true or false.
    Bool,
    /// `Str`: a string.
    Str,
    /// `[Str]`: a list of strings.
    StrList,
    /// `Int`: a 64-bit integer.
    Int,
    /// `[Int]`: an array of Ints.
    IntList,
    /// `Float32`: an IEEE binary32 number.
    F32,
    /// `[Float32]`, `[BFloat16]`, `[Float16]`, `[Int8]`, `[Int16]`,
    /// `[Int32]`, `[UInt8]`: a packed rank-1 array.
    Packed(Store),
    /// `T?`: an optional T, `Some(x)` or `None`.
    Opt(Box<TypeExpr>),
    /// A named (record) type, e.g. `Mesh`.
    Named { name: String, span: Span },
    /// `[Block]`: an array of records of one type. A value, like the record it holds: `b = a` gives `b` its
    /// own array.
    RecList { name: String, span: Span },
    /// A function VALUE taking `arity` numbers. Synthesized by `defun` when it
    /// keeps a generic copy of a higher-order def, so the rest of the compiler
    /// sees an ordinary annotated parameter rather than a special case.
    Fn { arity: usize },
    /// A field of a record `crate::tuples` synthesizes for a def that
    /// returns several values: its type is inferred from the `return`s.
    Infer,
}

impl TypeExpr {
    pub fn span(&self) -> Option<Span> {
        match self {
            TypeExpr::Named { span, .. } | TypeExpr::RecList { span, .. } => Some(*span),
            _ => None,
        }
    }
}

/// A top-level item: a record declaration, a function definition, or a
/// program-level statement (assignment / bare expression) that runs in `main`.
#[derive(Clone, Debug)]
pub enum Item {
    Record(RecordDecl),
    /// `enum Name { A, B(T, U) }`: a sum type.
    Enum(EnumDecl),
    Def(Def),
    Stmt(Stmt),
    /// One step of a ladder: `rung "title" { ... }`. See `crate::ladder`.
    Rung(Rung),
}

/// A rung holds what a top level holds. The program at rung k is everything
/// outside any rung plus rungs 1..k, with a def in a later rung REPLACING an
/// earlier def of the same name -- that is what an edit is.
#[derive(Clone, Debug)]
pub struct Rung {
    pub title: String,
    pub items: Vec<Item>,
    /// from the opening `{` to the closing `}`: the rung's own text
    pub body_span: Span,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct RecordDecl {
    pub name: String,
    pub name_span: Span,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

/// `enum Name { A, B(T, U), ... }`: its variants in order, each with the
/// types of its payload (none for a variant like `A`).
#[derive(Clone, Debug)]
pub struct EnumDecl {
    pub name: String,
    pub name_span: Span,
    pub variants: Vec<VariantDecl>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct VariantDecl {
    pub name: String,
    pub name_span: Span,
    pub payload: Vec<TypeExpr>,
}

#[derive(Clone, Debug)]
pub struct FieldDecl {
    pub name: String,
    pub name_span: Span,
    pub ty: TypeExpr,
    /// `width: Int = 16`: the value a literal that leaves the field out gets
    /// a constant expression.
    pub default: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Def {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    /// Optional return-type annotation (`-> Type`); required when returning a record.
    pub ret: Option<TypeExpr>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub span: Span,
    /// Optional type annotation (`name: Type`); required for record-typed params.
    pub ty: Option<TypeExpr>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// `name = expr` — creates or rebinds a binding.
    Assign { name: String, name_span: Span, value: Expr, span: Span },
    /// `return expr`.
    Return { value: Expr, span: Span },
    /// A bare expression evaluated for effect, e.g. `print(x)`.
    ExprStmt { expr: Expr, span: Span },
    /// `narrate <expr>` — live algorithm narration. A string argument becomes a
    /// runtime progress line (`narrate: <text>`) in every backend — the program
    /// announcing its own flow chart ("building hamiltonian", "calculating
    /// ERIs"). A non-string argument is commentary: parsed and dropped (text
    /// stays empty, the statement compiles to nothing).
    Narrate { text: String, span: Span },
    /// `show <expr>` — print a value (prefixed by its name when it is a variable).
    Show { expr: Expr, span: Span },
    /// `test "name" { body }` — runs the body (the embedded asserts self-check).
    /// `test "name" { .. }`. The name is the reporting unit: asserts inside
    /// the block report against it, so the block prints one PASS/FAIL line.
    Test { name: String, body: Vec<Stmt>, span: Span },
    /// `base[indices...] = value` — assignment into an array element. `base` is
    /// the array expression (a variable, or a record field like `m.node_x`).
    SetIndex { base: Box<Expr>, indices: Vec<Expr>, value: Expr, span: Span },
    /// `base.field = value`: assignment into a field of a record, where
    /// `base` is a place: a variable, then fields and elements of arrays of
    /// records (`model.blocks[l].attn.Wq = W`). Records are values, so it
    /// changes the value the place's root variable holds and nothing else.
    /// The root must be a variable of the
    /// def it is written in (`crate::places`).
    SetField { base: Box<Expr>, field: String, field_span: Span, value: Expr, span: Span },
    /// `if cond { then } else { els }` (els empty if no else; an `else if`
    /// chains as a single `If` in the else branch).
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>, span: Span },
    /// `while cond { body }`.
    While { cond: Expr, body: Vec<Stmt>, span: Span },
    /// `for var to bound { body }` — `var` ranges 0..bound-1. `parallel` marks
    /// it for OpenMP parallelization (`parallel for ...`).
    For { var: String, var_span: Span, bound: Expr, body: Vec<Stmt>, parallel: bool, span: Span },
    Break { span: Span },
    Continue { span: Span },
    /// The opening of `step <id> "<text>" { ... }`: a named phase of a def.
    /// A step is a block, not a scope, so the parser writes it FLAT: this
    /// marker, the body's statements as siblings in the enclosing block, then
    /// `StepExit`. Every pass that walks statements therefore sees the body
    /// exactly as if the braces were not there, and only the passes that care
    /// about steps (the engines' step events, the doc tools) look at the
    /// markers. `def` is the enclosing def's name as written in the source
    /// (`<main>` at the top level), `depth` the step's nesting inside that def
    /// (1 for an outermost step), and `text` is empty for `step <id> { ... }`.
    StepEnter { def: String, id: String, text: String, depth: u32, span: Span },
    /// The closing brace of a step. A `return`, `break` or `continue` inside
    /// the step leaves it without reaching this marker; both engines record
    /// the exit on that path too.
    StepExit { def: String, id: String, depth: u32, span: Span },
    /// A debug-lens statement or a placeholder body (see `Lens`).
    Lens { lens: Box<Lens>, span: Span },
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Assign { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::ExprStmt { span, .. }
            | Stmt::Narrate { span, .. }
            | Stmt::Show { span, .. }
            | Stmt::Test { span, .. }
            | Stmt::SetIndex { span, .. }
            | Stmt::SetField { span, .. }
            | Stmt::If { span, .. }
            | Stmt::While { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Break { span }
            | Stmt::Continue { span }
            | Stmt::StepEnter { span, .. }
            | Stmt::StepExit { span, .. }
            | Stmt::Lens { span, .. } => *span,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    /// An imaginary literal: `2.5i`.
    Imag { value: f64, span: Span },
    /// `true` or `false`. A `Bool` is its own type: it is what a comparison
    /// yields, what `if` and `while` take, and it can be stored, passed and
    /// returned -- but it is not a number, and no arithmetic applies to it.
    Bool { value: bool, span: Span },
    /// A number literal. Its value is an `f64`; one written as an integer
    /// (`Program::whole`) is an Int where an Int is expected.
    Num { value: f64, span: Span },
    /// A string literal: a value, or, as a statement of
    /// its own, a docstring.
    Str { value: String, span: Span },
    Var { name: String, span: Span },
    /// An array literal: `[a, b, c]` (vector) or `[[…], […]]` (matrix). Elements
    /// are either all scalars (rank 1) or all equal-length array literals (rank 2+).
    ArrLit { elems: Vec<Expr>, span: Span },
    /// Array indexing: `base[i]`, `base[i, j]`, `base[i, j, k, l]`.
    Index { base: Box<Expr>, indices: Vec<Expr>, span: Span },
    /// An anonymous function `fn(params) -> body` (single-expression body).
    Lambda { params: Vec<String>, body: Box<Expr>, span: Span },
    /// Arithmetic negation (Num -> Num).
    Neg { operand: Box<Expr>, span: Span },
    /// Logical negation (Bool -> Bool).
    Not { operand: Box<Expr>, span: Span },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr>, span: Span },
    Call { callee: String, args: Vec<Expr>, span: Span },
    /// A record literal: `TypeName { field: expr, ... }`. Field order is irrelevant.
    RecordLit { type_name: String, type_span: Span, fields: Vec<(String, Span, Expr)>, span: Span },
    /// Field access: `base.field`.
    Field { base: Box<Expr>, field: String, field_span: Span, span: Span },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Imag { span, .. }
            | Expr::Bool { span, .. }
            | Expr::Num { span, .. }
            | Expr::Str { span, .. }
            | Expr::Var { span, .. }
            | Expr::ArrLit { span, .. }
            | Expr::Index { span, .. }
            | Expr::Lambda { span, .. }
            | Expr::Neg { span, .. }
            | Expr::Not { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Call { span, .. }
            | Expr::RecordLit { span, .. }
            | Expr::Field { span, .. } => *span,
        }
    }
}
