// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv's IR vocabulary, the checked lowering AST -> IR, and the C emission
//! rules.
//!
//! Values are now either numbers (`double`), transient booleans (conditions
//! only), or arrays (`viv_arr` — a unified rank-N type). Lowering infers a
//! `Ty` for every expression so the C backend can declare the right type and
//! the right runtime calls. Booleans are never stored; arrays are stored,
//! indexed, and built but (for this milestone) do not cross function boundaries
//! — function params and returns are numbers.
//!
//! Viv is function-scoped: every assigned name is hoisted to a zero-initialized
//! local at function entry (kit `CFunc.locals`), so a binding introduced in a
//! branch survives it. The AST is consumed here and never handed onward (1.1).

use std::collections::HashMap;

use knotkit::backend::{CEmit, CEmitCx, CExpr, CFunc, CItem, CStmt, CStmtKind, CType, Runtime, RuntimeFragment};
use knotkit::diag::Diagnostic;
use knotkit::ir::{Hasher, IrProgram, Node, NodeId, NodeKind};
use knotkit::span::{SourceId, Span};

use crate::ast::{BinOp, Def, Expr, HoldCheck, Item, Lens, Program, Stmt, TypeExpr};
pub use crate::interp::maps::Mk;

// Strings: their inference, lowering and C emission.
mod records;
mod strings;
// Integers: which literals are Ints, the Int operations,
// their C runtime.
pub(crate) mod ints;
// Enums, `match` and optionals: their registry, lowering
// and C.
pub(crate) mod enums;
// Integer loop counters and integer indices.
pub(crate) mod intidx;
// Float32 and the packed arrays: their lowering, the
// literals that become Float32s, their C runtime.
pub(crate) mod f32s;

/// Every runtime call strings make, with its argument and result kinds (the
/// wasm backend imports each). `float` of a number is never a call.
pub(crate) fn string_runtime() -> Vec<strings::RtFn> {
    let mut v: Vec<strings::RtFn> = crate::interp::strings::BUILTINS
        .iter()
        .filter(|b| b.c != "viv_num_float")
        .filter_map(|b| strings::rt_fn(b.c))
        .collect();
    v.dedup_by(|a, b| a.c == b.c);
    v.extend(strings::RUNTIME.iter().filter_map(|r| strings::rt_fn(r.c)));
    v
}

const PRINT_RUNTIME: &str = "viv_print_num";

/// The type of a Viv value. `Bool` is transient (conditions only); `Num` and
/// `Arr` are storable.
/// What an array's slots hold. Kept small and `Copy` so `Ty` stays `Copy` --
/// the same constraint records hit, solved the same way (an interned tag
/// rather than a boxed `Ty`). `Elem` does not nest: Viv arrays are rank-N, not
/// arrays of arrays.
/// What a function VALUE's callers need to know about it: how many arguments it
/// takes, and which of them are themselves functions. The mask matters because
/// arguments travel through a uniform ABI as a `viv_arg`, a union of a double
/// and a function pointer with no tag -- the wrapper reads `a[i].num` or
/// `a[i].fn` by this mask, so two functions that share an arity but not a mask
/// are NOT interchangeable and must not share a registry.
/// What one argument slot (or the result) of a function VALUE holds. These are
/// the three members of the C `viv_arg` union; the union has no tag, so this is
/// the only record of which member is live, and it is fixed at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgKind {
    Num,
    Fn,
    /// A real array (`viv_arr`), passed as its handle -- the same aliasing an
    /// ordinary array argument has.
    Arr,
}

impl ArgKind {
    fn of(t: Ty) -> Option<ArgKind> {
        match t {
            Ty::Num => Some(ArgKind::Num),
            Ty::Func(_) => Some(ArgKind::Fn),
            Ty::Arr(Elem::Num) => Some(ArgKind::Arr),
            _ => None,
        }
    }
    fn bits(self) -> u32 {
        match self {
            ArgKind::Num => 0,
            ArgKind::Fn => 1,
            ArgKind::Arr => 2,
        }
    }
    fn from_bits(b: u32) -> ArgKind {
        match b & 3 {
            1 => ArgKind::Fn,
            2 => ArgKind::Arr,
            _ => ArgKind::Num,
        }
    }
    fn member(self) -> &'static str {
        match self {
            ArgKind::Num => "num",
            ArgKind::Fn => "fn",
            ArgKind::Arr => "arr",
        }
    }
    /// The kind as a diagnostic spells it.
    fn word(self) -> &'static str {
        match self {
            ArgKind::Num => "a number",
            ArgKind::Fn => "a function",
            ArgKind::Arr => "an array",
        }
    }
    fn short(self) -> &'static str {
        match self {
            ArgKind::Num => "Num",
            ArgKind::Fn => "Fn",
            ArgKind::Arr => "Arr",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FnSig {
    pub arity: u8,
    /// Two bits per parameter (an `ArgKind`), parameter 0 in the low bits.
    pub kinds: u32,
    /// What the function returns: a number or an array.
    pub ret: ArgKind,
}

/// The most arguments one `apply` call can pass: the size of the C runtime's
/// `viv_arg` buffer and the number of parameter kinds a `FnSig` records.
pub const APPLY_MAX_ARGS: usize = 16;

/// The highest array rank the C runtime allocates and indexes (`zeros1` to
/// `zeros4`, `get1` to `get4`). The interpreter holds to the same limit, so a
/// program of higher rank is refused by both engines instead of running in one.
pub const MAX_RANK: usize = 4;

fn check_index_rank(n: usize, span: Span) -> Result<(), Diagnostic> {
    if n > MAX_RANK {
        return Err(Diagnostic::error(format!("an array takes at most {MAX_RANK} indices; this has {n}"))
            .with_primary(span, format!("arrays have rank {MAX_RANK} at most")));
    }
    Ok(())
}

impl FnSig {
    pub fn plain(arity: usize) -> FnSig {
        FnSig { arity: arity as u8, kinds: 0, ret: ArgKind::Num }
    }
    pub fn kind(self, i: usize) -> ArgKind {
        if i < 16 { ArgKind::from_bits(self.kinds >> (2 * i)) } else { ArgKind::Num }
    }
    fn ret_ty(self) -> Ty {
        match self.ret {
            ArgKind::Arr => Ty::Arr(Elem::Num),
            _ => Ty::Num,
        }
    }
    /// `(Arr, Num) -> Arr`: the signature as a registry diagnostic prints it.
    fn describe(self) -> String {
        let params: Vec<&str> = (0..self.arity as usize).map(|j| self.kind(j).short()).collect();
        format!("({}) -> {}", params.join(", "), self.ret.short())
    }
    fn hash_into(self, h: &mut Hasher) {
        h.update(&[self.arity, self.ret.bits() as u8]);
        h.update(&self.kinds.to_le_bytes());
    }
}

/// A function-bearing type is the one kind of type inference may REVISE: a
/// function's signature is computed from its parameter types, which can still be
/// settling when a registry literal naming it is first seen. Everything else is
/// fixed by its first binding.
fn revises(old: Ty, new: Ty) -> bool {
    match (old, new) {
        (Ty::Func(a), Ty::Func(b)) => a != b,
        (Ty::Arr(Elem::Func(a)), Ty::Arr(Elem::Func(b))) => a != b,
        // a variable guessed to be an array because it is indexed is a map
        // once a map reaches it; never the reverse
        (Ty::Arr(Elem::Num), Ty::Map(..)) => true,
        // a map's kinds only grow: nothing stored yet, then numbers, then
        // strings; a map that stores both kinds
        // of key is then refused where it stores the number
        (Ty::Map(k0, v0), Ty::Map(k1, v1)) => (k0, v0) != (k1, v1) && k0 <= k1 && v0 <= v1,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// f64 slots: `double*` in C, `Vec<f64>` in the interpreter. The default,
    /// and the only one that exists before the payloads land.
    Num,
    /// Complex slots: `double complex*` in C, interleaved (re, im) f64 pairs in
    /// the interpreter. Those are the same bytes -- C99 lays `double complex`
    /// out as two doubles -- which is what lets the engines agree to the bit.
    /// PACKED, like Num: no tag per slot and none of the boxed-value machinery.
    Cplx,
    /// Function slots: `viv_fn*` in C, `Val::Func` in the interpreter's boxed
    /// `VArr`. A `viv_fn` is a bare function pointer with no captures, so this
    /// is PACKED too -- an array of them is one contiguous block, not a tagged
    /// union per slot. It carries the shared arity because a C array has one
    /// element type: every function in one array must take the same number of
    /// arguments (and the same argument KINDS), and indexing has to know the
    /// signature to type the result.
    ///
    /// This is what a strategy REGISTRY is made of
    /// (`methods = [r_bisect, r_illinois, r_safe]`), which is how a contract
    /// holds its strategies as data instead of spelling them out. Only three
    /// operations are defined on it -- build a literal, index it, take its
    /// length -- because those are the three a registry needs. Arithmetic on a
    /// function array is a type error, not a silent reinterpretation of
    /// function pointers as doubles.
    Func(FnSig),
    /// String slots: a list of strings, `viv_strs` in C,
    /// `Val::Strs` in the interpreter. Rank 1, immutable.
    Str,
    /// Int slots: 8 bytes each, `int64_t` in the same
    /// `viv_arr` block a number array has. Rank 1.
    Int,
    /// Packed slots of the given width: `[Float32]`,
    /// `[BFloat16]`, `[Int8]`, ... In C a `viv_arr` whose `dims[0]` is the
    /// element count and whose `size` counts the doubles the bytes fill, so
    /// every copy that moves `size` doubles moves the bytes; in wasm a raw
    /// byte block (width 0) with `dims[0]` the element count; in the
    /// interpreter `Val::PArr`. Rank 1.
    P(crate::ast::Store),
}

impl Elem {
    pub fn name(self) -> &'static str {
        match self {
            Elem::Num => "number",
            Elem::Cplx => "complex number",
            Elem::Func(_) => "function",
            Elem::Str => "string",
            Elem::Int => "Int",
            Elem::P(k) => k.name(),
        }
    }
    /// The discriminant, for hashing. Written out rather than `self as u8`
    /// because `Func` carries a payload.
    fn tag(self) -> u8 {
        match self {
            Elem::Num => 0,
            Elem::Cplx => 1,
            Elem::Func(_) => 2,
            Elem::Str => 3,
            Elem::Int => 4,
            Elem::P(k) => 5 + k.code(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Num,
    Bool,
    Arr(Elem),
    /// A first-class function VALUE of the given arity (all arguments and the
    /// return are numbers, for now). Compiles to `viv_fn`, a function pointer
    /// with a uniform `(const double* args, int n) -> double` calling
    /// convention, so a def or lambda can be stored in a variable, returned,
    /// and (next) held in an array. Numbers stay raw `double`: only function
    /// values are boxed, so the numeric fast path is untouched.
    Func(FnSig),
    /// A complex scalar (C99 `double complex`). Arrays of them are `Arr(Elem::Cplx)`.
    Cplx,
    /// A string: `viv_str` in C, an i32 pointer in wasm.
    /// A list of strings is `Arr(Elem::Str)`.
    Str,
    /// The return type of a function with no `return` statement (e.g. one that
    /// only mutates arrays). Never the type of a storable value.
    Void,
    /// A nominal record type, identified by its (interned) declared name. Equality
    /// is by name (nominal). The field layout lives in the record registry (`Sigs`);
    /// the name is enough to emit the C struct type (`viv_rec_<name>`). Keeping the
    /// name as `&'static str` keeps `Ty: Copy`.
    Record(&'static str),
    /// A map: a handle changed in place, `viv_map*` in C
    /// (`viv_smap*` when its keys or values are strings), an i32 naming the
    /// runtime's map in wasm. The kinds of its keys and of its values; `Any`
    /// lives only in inference, which settles it to `Num`. Appended, so every
    /// earlier type keeps its tag.
    Map(Mk, Mk),
    /// An array of records of the named type (`[Block]`): rank 1, a value like the records it holds. `viv_recs` in C
    /// (the records packed one after another), a block of record pointers in
    /// wasm, a `VArr` of records in the interpreter. Appended, so every
    /// earlier type keeps its tag.
    Recs(&'static str),
    /// A 64-bit integer: `int64_t` in C, an i64 in wasm,
    /// `Val::Int` in the interpreter. Appended, so every earlier type keeps
    /// its tag.
    Int,
    /// An enum, or an optional (`Int?`), by its interned name:
    /// a pointer to an immutable node in C and wasm,
    /// `Val::Enum` in the interpreter. Its variants live in `enums`'s registry.
    Enum(&'static str),
    /// An IEEE binary32 number: `float` in C, an f32 in
    /// wasm, `Val::F32` in the interpreter. Appended, so every earlier type
    /// keeps its tag.
    F32,
}

/// Numbers to numbers: the map every map was before string keys.
pub const NUM_MAP: Ty = Ty::Map(Mk::Num, Mk::Num);

/// `t` with any map kind inference left open settled to numbers.
pub(crate) fn settle(t: Ty) -> Ty {
    match t {
        Ty::Map(k, v) => Ty::Map(k.settled(), v.settled()),
        t => t,
    }
}

/// The type of a map's key or value of kind `k`.
pub(crate) fn mk_ty(k: Mk) -> Ty {
    if k == Mk::Str { Ty::Str } else { Ty::Num }
}

/// A noun with its indefinite article: "an array", "an Int array", "a number".
/// Diagnostics say "this is {}" with it, so the article fits the type's name.
pub(crate) fn a_or_an(noun: &dyn std::fmt::Display) -> String {
    let s = noun.to_string();
    let vowel = s.chars().next().is_some_and(|c| "aeiouAEIOU".contains(c));
    format!("{} {s}", if vowel { "an" } else { "a" })
}

impl Ty {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Ty::Num => "number",
            Ty::Bool => "boolean",
            Ty::Arr(Elem::Num) => "array",
            Ty::Arr(Elem::Cplx) => "complex array",
            Ty::Arr(Elem::Func(_)) => "function array",
            Ty::Arr(Elem::Str) => "list of strings",
            Ty::Arr(Elem::Int) => "Int array",
            Ty::Arr(Elem::P(k)) => f32s::arr_name(k),
            Ty::F32 => "Float32",
            Ty::Int => "Int",
            Ty::Enum(n) => n,
            Ty::Str => "string",
            Ty::Cplx => "complex number",
            Ty::Void => "nothing",
            Ty::Func(_) => "function",
            Ty::Record(n) => n,
            Ty::Recs(_) => "array of records",
            Ty::Map(k, v) => match (k.settled(), v.settled()) {
                (Mk::Str, Mk::Num) => "map with string keys",
                (Mk::Num, Mk::Str) => "map with string values",
                (Mk::Str, Mk::Str) => "map of strings to strings",
                _ => "map",
            },
        }
    }
    fn tag(self) -> u8 {
        match self {
            Ty::Num => 0,
            Ty::Bool => 1,
            Ty::Arr(_) => 2,
            Ty::Void => 3,
            Ty::Record(_) => 4,
            Ty::Cplx => 5,
            Ty::Func(_) => 6,
            Ty::Map(..) => 7,
            Ty::Str => 8,
            Ty::Recs(_) => 9,
            Ty::Int => 10,
            Ty::Enum(_) => 11,
            Ty::F32 => 12,
        }
    }
    /// Hash this type's identity (the discriminant, plus the record name so two
    /// distinct record types never collide in the content-addressed NodeId).
    fn hash_into(self, h: &mut Hasher) {
        h.update(&[self.tag()]);
        // An array's ELEMENT type is part of its identity: without this,
        // `Arr(Cplx)` and `Arr(Num)` would share a content address, which is
        // the same failure the IR tags had when Scatter and MapMath shared 36.
        // Num contributes nothing, so every address minted before elements
        // existed still hashes to what it did -- the payload-only-when-needed
        // trick Record and Func already use below.
        if let Ty::Arr(e) = self {
            if e != Elem::Num {
                h.update(&[0xA0 | e.tag()]);
            }
            // A function array's ARITY is part of its identity too: an array of
            // one-argument strategies and one of three-argument strategies are
            // different types and must not share a content address.
            if let Elem::Func(sig) = e {
                sig.hash_into(h);
            }
        }
        if let Ty::Record(n) | Ty::Recs(n) | Ty::Enum(n) = self {
            h.update(&(n.len() as u32).to_le_bytes());
            h.update(n.as_bytes());
        }
        if let Ty::Func(sig) = self {
            sig.hash_into(h);
        }
        // a map of strings: its kinds (numbers to numbers adds nothing, so
        // every address minted before string keys hashes as it did)
        if let Ty::Map(k, v) = settle(self) {
            if (k, v) != (Mk::Num, Mk::Num) {
                h.update(&[0xB0 | ((k == Mk::Str) as u8) << 1 | (v == Mk::Str) as u8]);
            }
        }
    }
    /// The C type a value of this type is held in (`double`, `viv_arr`, ...).
    pub fn c_type_name(self) -> String {
        self.to_c().c_name().to_string()
    }

    fn to_c(self) -> CType {
        match self {
            Ty::Num | Ty::Bool => CType::Double,
            Ty::Int => CType::Named("long long".to_string()),
            Ty::F32 => CType::Named("float".to_string()),
            Ty::Enum(n) => CType::Named(format!("const struct viv_en_{}*", enums::c_name(n))),
            Ty::Cplx => CType::Named("double complex".to_string()),
            // Numbers and complex numbers share the struct: the element type is
            // static, so the emitted code knows the stride without a runtime
            // tag. Function slots are a different width and a different kind of
            // thing, so they get their own struct rather than a cast.
            Ty::Arr(Elem::Func(_)) => CType::Named("viv_fnarr".to_string()),
            Ty::Arr(Elem::Str) => CType::Named("viv_strs".to_string()),
            Ty::Str => CType::Named("viv_str".to_string()),
            Ty::Arr(_) => CType::Named("viv_arr".to_string()),
            Ty::Void => CType::Named("void".to_string()),
            Ty::Func(_) => CType::Named("viv_fn".to_string()),
            Ty::Record(n) => CType::Named(format!("viv_rec_{n}")),
            Ty::Recs(_) => CType::Named("viv_recs".to_string()),
            Ty::Map(..) if settle(self) == NUM_MAP => CType::Named("viv_map*".to_string()),
            Ty::Map(..) => CType::Named("viv_smap*".to_string()),
        }
    }
}

/// A resolved array operation. Lowering picks the right one by operand types,
/// so the backend just emits the corresponding runtime call (faithful, 2.6).
#[derive(Clone, Copy, Debug)]
pub enum ArrKind {
    /// elementwise a + b. children: [a, b].
    Add,
    /// elementwise a - b. children: [a, b].
    Sub,
    /// elementwise (Hadamard) a * b. children: [a, b].
    Had,
    /// scalar multiple a * s. children: [arr, scalar].
    Scale,
    /// a / s. children: [arr, scalar].
    DivS,
    /// matrix product a @ b. children: [a, b].
    Matmul,
    /// transpose. children: [a].
    Transpose,
    /// linspace(lo, hi, n) -> 1-D array. children: [lo, hi, n] (all scalars).
    Linspace,
    /// elementwise a + s. children: [arr, scalar].
    AddS,
    /// elementwise a - s. children: [arr, scalar].
    SubS,
    /// elementwise s - a. children: [arr, scalar].
    SubSR,
    /// elementwise a / b. children: [a, b]. APPENDED (the tag is `self as u8`),
    /// so every earlier kind keeps its content address. The interpreter always
    /// had this; the C backend refused it until a DFT code needed num/den.
    Div,
    /// elementwise s / a. children: [arr, scalar].
    DivSR,
    // ---- complex arrays. All appended; all packed (re, im) interleaved. ----
    /// complex a + b / a - b / a * b (Hadamard). children: [a, b], both complex.
    CAdd,
    CSub,
    CHad,
    /// complex a * real b, elementwise. children: [carr, rarr] (CHadRL: the
    /// other way round).
    CHadR,
    /// complex a * s, s a real or complex scalar. children: [carr, scalar].
    CScale,
    /// complex a / s. children: [carr, scalar].
    CDivS,
    /// complex matrix product. children: [a, b], both complex.
    CMatmul,
    /// transpose of a complex matrix (no conjugation). children: [a].
    CTranspose,
    /// elementwise max / min of two real arrays. children: [a, b].
    MaxA,
    MinA,
    /// elementwise max / min of a real array and a scalar. children: [arr, scalar].
    MaxS,
    MinS,
    /// elementwise a ^ s, a real array to a scalar power. children: [arr, scalar].
    PowS,
    /// elementwise comparison of two real arrays: 1.0 where it holds, 0.0
    /// where it does not. children: [a, b].
    LtA,
    LeA,
    GtA,
    GeA,
    EqA,
    NeA,
    /// the same against a scalar. children: [arr, scalar]; `s < a` is stored
    /// as `a > s`.
    LtS,
    LeS,
    GtS,
    GeS,
    EqS,
    NeS,
    /// -a of a complex array: both parts negated, signed zeros included (a
    /// scale by -1 + 0i turns the imaginary part of 0 + 0i into +0).
    /// children: [a].
    CNeg,
    /// a real array times a complex one, the real one written first: children
    /// [rarr, carr] in source order, so a shape mismatch names them in that
    /// order, as the interpreter does. Computed as CHadR.
    CHadRL,
}

impl ArrKind {
    fn tag(self) -> u8 {
        self as u8
    }
    fn runtime(self) -> &'static str {
        match self {
            ArrKind::Add => "viv_arr_add",
            ArrKind::Sub => "viv_arr_sub",
            ArrKind::Had => "viv_arr_had",
            ArrKind::Scale => "viv_arr_scale",
            ArrKind::DivS => "viv_arr_divs",
            ArrKind::Matmul => "viv_matmul",
            ArrKind::Transpose => "viv_transpose",
            ArrKind::Linspace => "viv_linspace",
            ArrKind::AddS => "viv_arr_adds",
            ArrKind::SubS => "viv_arr_subs",
            ArrKind::SubSR => "viv_arr_subsr",
            ArrKind::Div => "viv_arr_div",
            ArrKind::DivSR => "viv_arr_divsr",
            ArrKind::CAdd => "viv_carr_add",
            ArrKind::CSub => "viv_carr_sub",
            ArrKind::CHad => "viv_carr_had",
            ArrKind::CHadR | ArrKind::CHadRL => "viv_carr_hadr",
            ArrKind::CScale => "viv_carr_scale",
            ArrKind::CDivS => "viv_carr_divs",
            ArrKind::CNeg => "viv_carr_neg",
            ArrKind::CMatmul => "viv_cmatmul",
            ArrKind::CTranspose => "viv_ctranspose",
            ArrKind::MaxA => "viv_arr_max",
            ArrKind::MinA => "viv_arr_min",
            ArrKind::MaxS => "viv_arr_maxs",
            ArrKind::MinS => "viv_arr_mins",
            ArrKind::PowS => "viv_arr_pows",
            ArrKind::LtA => "viv_arr_lt",
            ArrKind::LeA => "viv_arr_le",
            ArrKind::GtA => "viv_arr_gt",
            ArrKind::GeA => "viv_arr_ge",
            ArrKind::EqA => "viv_arr_eq",
            ArrKind::NeA => "viv_arr_ne",
            ArrKind::LtS => "viv_arr_lts",
            ArrKind::LeS => "viv_arr_les",
            ArrKind::GtS => "viv_arr_gts",
            ArrKind::GeS => "viv_arr_ges",
            ArrKind::EqS => "viv_arr_eqs",
            ArrKind::NeS => "viv_arr_nes",
        }
    }
    /// How viv_arr_chk checks the two arrays of an operation (None: not two
    /// arrays): 0 the same shape or broadcast (real arrays), 1 the same shape
    /// with the operands swapped for the call, 2 `@` (batched), 3 `max`'s
    /// words, 4 `min`'s (both broadcast), 5 the same shape (complex arrays),
    /// 6 complex `@` (matrices and vectors).
    fn shape_check(self) -> Option<u8> {
        use ArrKind::*;
        match self {
            Add | Sub | Had | Div | LtA | LeA | GtA | GeA | EqA | NeA => Some(0),
            // complex arrays do not broadcast
            CAdd | CSub | CHad | CHadR => Some(5),
            CHadRL => Some(1),
            Matmul => Some(2),
            // complex @ takes matrices and vectors only
            CMatmul => Some(6),
            MaxA => Some(3),
            MinA => Some(4),
            _ => None,
        }
    }
    fn frag(self) -> &'static str {
        match self {
            ArrKind::Add => "arr_add",
            ArrKind::Sub => "arr_sub",
            ArrKind::Had => "arr_had",
            ArrKind::Scale => "arr_scale",
            ArrKind::DivS => "arr_divs",
            ArrKind::Matmul => "matmul",
            ArrKind::Transpose => "transpose",
            ArrKind::Linspace => "linspace",
            ArrKind::AddS => "arr_adds",
            ArrKind::SubS => "arr_subs",
            ArrKind::SubSR => "arr_subsr",
            ArrKind::Div => "arr_div",
            ArrKind::DivSR => "arr_divsr",
            ArrKind::CAdd | ArrKind::CSub | ArrKind::CHad | ArrKind::CHadR | ArrKind::CHadRL | ArrKind::CScale | ArrKind::CDivS | ArrKind::CNeg => "carr_arith",
            ArrKind::CMatmul => "cmatmul",
            ArrKind::CTranspose => "ctranspose",
            ArrKind::MaxA | ArrKind::MinA | ArrKind::MaxS | ArrKind::MinS => "arr_minmax",
            ArrKind::PowS => "arr_pows",
            ArrKind::LtA | ArrKind::LeA | ArrKind::GtA | ArrKind::GeA | ArrKind::EqA | ArrKind::NeA => "arr_cmp",
            ArrKind::LtS | ArrKind::LeS | ArrKind::GtS | ArrKind::GeS | ArrKind::EqS | ArrKind::NeS => "arr_cmps",
        }
    }
}

/// The animation verbs: for each, how many string
/// literals, arrays and numbers it takes, in that order. Inside a `figure()`
/// (no `stage`), hline, vline, xtick, label, rect and segment are static marks
/// on that figure's chart, drawn exactly as the record says. `opt` more numbers may
/// follow. One table, read by the checked lowering and by the interpreter, so
/// the two engines cannot disagree about a signature.
pub struct AnimVerb {
    pub name: &'static str,
    pub texts: usize,
    pub arrays: usize,
    pub nums: usize,
    pub opt: usize,
    pub usage: &'static str,
}
pub const ANIM_VERBS: &[AnimVerb] = &[
    AnimVerb { name: "stage", texts: 1, arrays: 0, nums: 4, opt: 0, usage: "stage(\"title\", xlo, xhi, ylo, yhi)" },
    AnimVerb { name: "axes", texts: 2, arrays: 0, nums: 0, opt: 0, usage: "axes(\"x label\", \"y label\")" },
    AnimVerb { name: "curve", texts: 1, arrays: 2, nums: 0, opt: 0, usage: "curve(\"id\", x, y)" },
    AnimVerb { name: "dots", texts: 1, arrays: 2, nums: 0, opt: 1, usage: "dots(\"id\", x, y) or dots(\"id\", x, y, radius)" },
    AnimVerb { name: "circle", texts: 1, arrays: 0, nums: 3, opt: 0, usage: "circle(\"id\", x, y, radius)" },
    AnimVerb { name: "rect", texts: 1, arrays: 0, nums: 4, opt: 1, usage: "rect(\"id\", x, y, width, height) or rect(\"id\", x, y, width, height, angle)" },
    AnimVerb { name: "segment", texts: 1, arrays: 0, nums: 4, opt: 0, usage: "segment(\"id\", x1, y1, x2, y2)" },
    AnimVerb { name: "poly", texts: 1, arrays: 2, nums: 0, opt: 0, usage: "poly(\"id\", xs, ys)" },
    AnimVerb { name: "frame", texts: 0, arrays: 0, nums: 1, opt: 0, usage: "frame(seconds)" },
    AnimVerb { name: "label", texts: 2, arrays: 0, nums: 2, opt: 0, usage: "label(\"id\", \"text\", x, y)" },
    AnimVerb { name: "hline", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "hline(\"id\", y)" },
    AnimVerb { name: "vline", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "vline(\"id\", x)" },
    AnimVerb { name: "xtick", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "xtick(\"text\", x)" },
    // settings of the open figure
    AnimVerb { name: "xscale", texts: 1, arrays: 0, nums: 0, opt: 0, usage: "xscale(\"log\") or xscale(\"linear\")" },
    AnimVerb { name: "yscale", texts: 1, arrays: 0, nums: 0, opt: 0, usage: "yscale(\"log\") or yscale(\"linear\")" },
    AnimVerb { name: "aspect", texts: 1, arrays: 0, nums: 0, opt: 0, usage: "aspect(\"equal\") or aspect(\"auto\")" },
    // a chart's data drawn as intervals: a filled band, error bars (full precision, like plot)
    AnimVerb { name: "fill_between", texts: 1, arrays: 3, nums: 0, opt: 0, usage: "fill_between(\"label\", x, lo, hi)" },
    AnimVerb { name: "rangebars", texts: 1, arrays: 3, nums: 0, opt: 0, usage: "rangebars(\"label\", x, lo, hi)" },
    AnimVerb { name: "yright", texts: 2, arrays: 0, nums: 0, opt: 0, usage: "yright(\"series label\", \"right axis label\")" },
    // a grid of figures: the next rows * cols figure() calls are its panels
    AnimVerb { name: "grid", texts: 1, arrays: 0, nums: 2, opt: 0, usage: "grid(\"title\", rows, cols)" },
    AnimVerb { name: "share", texts: 1, arrays: 0, nums: 0, opt: 0, usage: "share(\"x\"), share(\"y\") or share(\"xy\")" },
    AnimVerb { name: "ytick", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "ytick(\"text\", y)" },
    AnimVerb { name: "style", texts: 2, arrays: 0, nums: 0, opt: 0, usage: "style(\"series label\", \"dash s2 ring thick\")" },
    // a heatmap's settings: where row 0 goes, the colormap, the color limits, the contour levels
    AnimVerb { name: "origin", texts: 1, arrays: 0, nums: 0, opt: 0, usage: "origin(\"lower\") or origin(\"upper\")" },
    AnimVerb { name: "colormap", texts: 1, arrays: 0, nums: 0, opt: 0, usage: "colormap(\"diverging\") or colormap(\"viridis\")" },
    AnimVerb { name: "clim", texts: 0, arrays: 0, nums: 2, opt: 0, usage: "clim(lo, hi)" },
    AnimVerb { name: "levels", texts: 0, arrays: 1, nums: 0, opt: 0, usage: "levels(values)" },
    AnimVerb { name: "draw", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "draw(\"id\", seconds)" },
    AnimVerb { name: "fade_in", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "fade_in(\"id\", seconds)" },
    AnimVerb { name: "fade_out", texts: 1, arrays: 0, nums: 1, opt: 0, usage: "fade_out(\"id\", seconds)" },
    AnimVerb { name: "morph", texts: 1, arrays: 2, nums: 1, opt: 0, usage: "morph(\"id\", x, y, seconds)" },
    AnimVerb { name: "move_to", texts: 1, arrays: 0, nums: 3, opt: 0, usage: "move_to(\"id\", x, y, seconds)" },
    AnimVerb { name: "say", texts: 1, arrays: 0, nums: 0, opt: 1, usage: "say(\"caption\") or say(\"caption\", value)" },
    AnimVerb { name: "wait", texts: 0, arrays: 0, nums: 1, opt: 0, usage: "wait(seconds)" },
    AnimVerb { name: "together", texts: 0, arrays: 0, nums: 0, opt: 0, usage: "together()" },
    AnimVerb { name: "apart", texts: 0, arrays: 0, nums: 0, opt: 0, usage: "apart()" },
    // a diagram's marks: an outlined box by
    // its corners, a polyline with an arrowhead, a styled text
    AnimVerb { name: "box", texts: 2, arrays: 0, nums: 4, opt: 0, usage: "box(\"id\", \"style\", x0, y0, x1, y1)" },
    AnimVerb { name: "arrow", texts: 2, arrays: 2, nums: 0, opt: 0, usage: "arrow(\"id\", \"style\", xs, ys)" },
    AnimVerb { name: "note", texts: 3, arrays: 0, nums: 2, opt: 0, usage: "note(\"id\", \"text\", \"style\", x, y)" },
];
/// The verbs whose arrays are a chart's data, printed at full precision (the
/// shortest of %.15g, %.16g, %.17g that reads back, as `plot`'s are); every
/// other verb's arrays are `%g`. Read by the interpreter and the C emission.
pub fn anim_data_verb(verb: &str) -> bool {
    matches!(verb, "fill_between" | "rangebars")
}
pub fn anim_verb(name: &str) -> Option<&'static AnimVerb> {
    ANIM_VERBS.iter().find(|v| v.name == name)
}
/// A verb's text that is a keyword, not a label, must be one of its words: a
/// misspelled style is a compile error, never a record the page ignores.
/// `k` is the text's position among the verb's texts. Read by the checked
/// lowering and by the interpreter, like the table.
pub fn anim_text_check(verb: &str, k: usize, text: &str) -> Result<(), String> {
    match (verb, k) {
        ("xscale" | "yscale", 0) if text != "linear" && text != "log" => {
            Err(format!("`{verb}` takes \"log\" or \"linear\", not \"{text}\""))
        }
        ("style", 1) => style_check(text),
        ("box", 1) => words_check("box", &BOX_WORDS, text),
        ("arrow", 1) => words_check("arrow", &ARROW_WORDS, text),
        ("note", 2) => words_check("note", &NOTE_WORDS, text),
        ("share", 0) if !["x", "y", "xy"].contains(&text) => {
            Err(format!("`share` takes \"x\", \"y\" or \"xy\" (the axes the grid's panels share), not \"{text}\""))
        }
        ("grid", 0) if text.is_empty() => Err("a grid's title cannot be empty".to_string()),
        ("aspect", 0) if text != "equal" && text != "auto" => {
            Err(format!("`aspect` takes \"equal\" (one unit is as long on x as on y) or \"auto\", not \"{text}\""))
        }
        ("origin", 0) if text != "lower" && text != "upper" => {
            Err(format!("`origin` takes \"lower\" (row 0 at the bottom) or \"upper\", not \"{text}\""))
        }
        ("colormap", 0) if text != "viridis" && text != "diverging" => {
            Err(format!("`colormap` takes \"viridis\" or \"diverging\", not \"{text}\""))
        }
        _ => Ok(()),
    }
}

/// The words of a series style, one group per aspect.
pub const STYLE_WORDS: [(&str, &[&str]); 4] = [
    ("line", &["solid", "dash", "dot", "noline"]),
    ("color", &["s1", "s2", "s3", "s4", "s5", "s6"]),
    ("marker", &["circle", "ring", "square", "cross", "nomarker"]),
    ("width", &["thin", "thick"]),
];

/// A style is words from STYLE_WORDS, separated by single spaces, at most one
/// of each aspect.
fn style_check(text: &str) -> Result<(), String> {
    let mut seen = [false; 4];
    for w in text.split(' ') {
        let Some(k) = STYLE_WORDS.iter().position(|(_, ws)| ws.contains(&w)) else {
            let all: Vec<&str> = STYLE_WORDS.iter().flat_map(|(_, ws)| ws.iter().copied()).collect();
            return Err(format!("`style` does not know \"{w}\"; its words are {}", all.join(" ")));
        };
        if seen[k] {
            return Err(format!("`style` gives the {} twice", STYLE_WORDS[k].0));
        }
        seen[k] = true;
    }
    Ok(())
}

/// The words of a diagram mark's style,
/// one group per aspect: a box, an arrow, a note.
pub const BOX_WORDS: [(&str, &[&str]); 5] = [
    ("color", &["s1", "s2", "s3", "s4", "s5", "s6", "ink", "muted"]),
    ("fill", &["plain", "tint", "clear"]),
    ("line", &["solid", "dash", "dot", "noline"]),
    ("width", &["thin", "thick"]),
    ("corner", &["round", "square", "diamond"]),
];
pub const ARROW_WORDS: [(&str, &[&str]); 5] = [
    ("color", &["s1", "s2", "s3", "s4", "s5", "s6", "ink", "muted"]),
    ("line", &["solid", "dash", "dot"]),
    ("width", &["thin", "thick"]),
    ("head", &["head", "nohead"]),
    ("shape", &["straight", "curved"]),
];
pub const NOTE_WORDS: [(&str, &[&str]); 6] = [
    ("color", &["s1", "s2", "s3", "s4", "s5", "s6", "ink", "muted"]),
    ("align", &["left", "center", "right"]),
    ("anchor", &["top", "middle", "bottom"]),
    ("size", &["small", "normal", "large"]),
    ("weight", &["regular", "bold"]),
    ("font", &["sans", "mono"]),
];

/// A diagram mark's style: words from `groups`, separated by single spaces,
/// at most one of each aspect. An empty style takes every default.
fn words_check(verb: &str, groups: &[(&str, &[&str])], text: &str) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }
    let mut seen = vec![false; groups.len()];
    for w in text.split(' ') {
        let Some(k) = groups.iter().position(|(_, ws)| ws.contains(&w)) else {
            let all: Vec<&str> = groups.iter().flat_map(|(_, ws)| ws.iter().copied()).collect();
            return Err(format!("`{verb}`'s style does not know \"{w}\"; its words are {}", all.join(" ")));
        };
        if seen[k] {
            return Err(format!("`{verb}`'s style gives the {} twice", groups[k].0));
        }
        seen[k] = true;
    }
    Ok(())
}

/// The part of an animation record known at compile time: `verb: text ; text`.
pub fn anim_head(verb: &str, texts: &[String]) -> String {
    if texts.is_empty() { format!("{verb}:") } else { format!("{verb}: {}", texts.join(" ; ")) }
}

/// A C string literal for `s`. Rust's `{:?}` is NOT one: it writes a non-ASCII
/// character or a control byte as `\u{80}`, which C rejects ("universal character
/// name refers to a control character"). Printable ASCII goes through as it is,
/// so every program that compiled before compiles to the same bytes; anything
/// else is three-digit octal, which no C compiler and no source charset can
/// misread. A `?` after a `?` is escaped so `-std=c99` cannot form a trigraph.
fn c_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    let mut prev_q = false;
    for &b in s.as_bytes() {
        match b {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            b'?' if prev_q => out.push_str("\\?"),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\{b:03o}")),
        }
        prev_q = b == b'?';
    }
    out.push('"');
    out
}

/// One part of a `Viv::Print` line: text, or a value of a type in a format.
#[derive(Clone, Debug, PartialEq)]
pub enum PrintPart {
    Text(String),
    Num(crate::printfmt::Fmt),
    Arr(crate::printfmt::Fmt),
    Bool,
    Cplx,
    /// A string value, written as its text.
    Str,
    /// a map: its keys in the first format, its values in the second
    Map(crate::printfmt::Fmt, crate::printfmt::Fmt),
    /// a map whose keys or values are strings:
    /// the formats of its number keys and values
    SMap(crate::printfmt::Fmt, crate::printfmt::Fmt),
}

/// The print part of a map of type `t`.
pub(crate) fn map_part(t: Ty, kf: crate::printfmt::Fmt, vf: crate::printfmt::Fmt) -> PrintPart {
    if settle(t) == NUM_MAP { PrintPart::Map(kf, vf) } else { PrintPart::SMap(kf, vf) }
}

/// Viv's IR node vocabulary. Children are referenced positionally.
#[derive(Clone, Debug)]
pub enum Viv {
    /// A function. children: [body statements...].
    /// `arena`: some expression in the body has array or record type, so the
    /// function can allocate and the C backend brackets it with the array
    /// arena. DERIVED from the body and the types, hence deliberately not part
    /// of the content address.
    /// `src`: the def's name as written, which is what its step events are
    /// recorded under (the interpreter knows no other). It differs from `name`
    /// only in a defun specialization (`apply__sq` of `apply`), so it too is
    /// derived and not part of the content address: a def's hash stays
    /// independent of the names involved.
    Func { name: String, src: String, params: Vec<(String, Ty)>, ret: Ty, locals: Vec<(String, Ty)>, arena: bool },
    /// The program body that becomes `main`. children: [statements...].
    Main { locals: Vec<(String, Ty)>, globals: Vec<(String, Ty)> },
    /// A statement sequence (body of an if/while/for). children: [stmts...].
    Block,
    /// `name = value`. children: [value].
    Assign { name: String },
    /// `base[indices...] = value`. children: [base, idx..., value].
    /// `at` is where the indexing is in the source (span start + 1) when the
    /// program is compiled with `with_sites`, and 0 otherwise. It keeps two
    /// identical-looking index expressions on different lines apart, so each
    /// can report its own location when an index is out of bounds; at 0 it is
    /// left out of the hash, so node ids (and body hashes) are what they were.
    SetIndex { at: u32 },
    /// `return value`. children: [value].
    Return,
    /// A statement-position expression (e.g. `print(x)`). children: [expr].
    ExprStmt,
    /// A statement with no runtime effect (docstring, narration). No children.
    Nop,
    /// A file-scope global declaration. No children.
    Global { name: String, ty: Ty },
    /// A record type declaration; emits a C struct typedef. No children.
    /// `fields` are (name, type) in declaration order.
    RecordDef { name: String, fields: Vec<(String, Ty)> },
    /// A record literal `TypeName { ... }`. children: field value exprs, in the
    /// order given by `fields`. `ty_name` is the record's name (-> `viv_rec_<name>`).
    RecordLit { ty_name: String, fields: Vec<String> },
    /// Field access `base.field`. children: [base].
    Field { field: String },
    /// A lifted lambda emitted as a top-level function. children: [body]. `param`
    /// is the lambda's argument; `captures` are enclosing locals passed via the
    /// closure's `viv_caps` array.
    LambdaDef { name: String, param: String, captures: Vec<(String, Ty)> },
    /// A closure value: a reference to a lifted lambda plus its captured vars.
    Closure { name: String, captures: Vec<String> },
    /// A top-level `def` used as a first-class function VALUE (e.g. `g = sq`).
    /// Emits `(viv_fn){ viv_wrap_<name> }`; the wrapper is generated by
    /// `FnWrapDef`. No children.
    FnRef { name: String, sig: FnSig },
    /// A registry literal: `[r_bisect, r_illinois, r_safe]`. children: the
    /// function values, all of `arity`. Emits a `viv_fnarr` holding `n`
    /// function pointers.
    FnArrLit { n: usize, sig: FnSig },
    /// Reading a function out of a registry: `methods[i]`. children:
    /// [registry, index]. The result is a `viv_fn` that `apply` calls
    /// indirectly -- the dispatch monomorphization cannot do.
    FnIndex { sig: FnSig, at: u32 },
    /// `len(registry)`. children: [registry].
    FnLen,
    /// The generated uniform-ABI wrapper for a def used as a value:
    /// `static double viv_wrap_<name>(const double* a, int n) { return <name>(a[0], ...); }`.
    /// One per def-used-as-value; add_root'd so it lands as a top-level C item.
    FnWrapDef { name: String, sig: FnSig },
    /// `apply(f, args...)`: call a function value. children: [f, args...].
    /// `apply(f, args...)`: an indirect call through the uniform ABI. `kinds`
    /// says which arguments are functions rather than numbers, so each can be
    /// packed into the right member of the argument union.
    ApplyFn { sig: FnSig },
    /// `show <expr>`. children: [expr]. `label` is the variable name, or empty.
    Show { label: String, is_arr: bool },
    /// `narrate "text"` — a runtime progress line on stdout.
    Narrate { msg: String },
    /// `print(...)` with text, a format, or more than one value
    /// (crate::printfmt). children: one value per part that is not text.
    Print { parts: Vec<PrintPart> },
    /// Open the named `test` block. Asserts between this and TestEnd report
    /// against the test rather than printing their own lines.
    TestBegin { name: String },
    /// Close the innermost `test` block and print its one PASS/FAIL line.
    TestEnd,
    /// Entering (`enter`) or leaving a `step`: records one step event in a
    /// debug build and nothing under -DVIV_RELEASE. The def the step belongs
    /// to is the enclosing `Func`'s `src` (or `<main>`), supplied at emission,
    /// so the node's content address holds the step's id and depth only.
    /// `story`: the program carries a debug lens, so the event also goes
    /// into the story (`viv_story_step`). Hashed only when set, so a program
    /// without a debug lens hashes as before. `def`: the def the step was
    /// written in, when that is not the enclosing `Func`'s `src` (a step of a
    /// def `crate::grad` inlined into a gradient def); hashed only when set.
    StepMark { enter: bool, id: String, depth: u32, story: bool, def: Option<String> },
    /// `return value` from inside open steps: evaluate the value, record the
    /// exit of each open step (innermost first), then return -- the order the
    /// interpreter records them in. children: [value].
    /// Each exit is (the def it was written in when not the `Func`'s `src`,
    /// id, depth), as for `StepMark`.
    StepReturn { exits: Vec<(Option<String>, String, u32)>, ty: Ty },
    /// The end of `main` in a program that has steps: print the `__steps__`
    /// line, if any step ran (debug builds only).
    StepsFlush,
    /// The debug lens (built by
    /// `crate::debug_lens`). Every one of these records into the story in a
    /// debug build and does nothing under -DVIV_RELEASE (`VIV_STORY` is 0).
    ///
    /// `return value` inside open steps or iteration loops of a program with
    /// a debug lens: evaluate the value, run the ways out (step exits and
    /// iteration ends, innermost first, as the interpreter does), then
    /// return. children: [value, ways out...].
    LensReturn { ty: Ty },
    /// The first statement of a def the story records calls of. The def's C
    /// function also records its return, on every way out.
    StoryCall { def: String },
    /// The return of a region `crate::grad` inlined (ast `Lens::Ret`),
    /// recorded as a def's return is.
    StoryRet { def: String },
    /// The start of an iteration. children: [its number].
    StoryIter { def: String },
    /// The end of an iteration.
    StoryIterEnd { def: String },
    /// Watched values, by name and type. children: one value per name.
    StoryWatch { def: String, at: String, names: Vec<(String, Ty)> },
    /// A hold. `sentence` is a JSON string literal. `near`: `None` for a
    /// boolean check (children: [cond]); `Some(arr)` for `a == b within tol`
    /// (children: [a - b, tol], `arr` when a - b is an array).
    StoryHold { def: String, at: String, sentence: String, near: Option<bool> },
    /// The end of `main` in a program with a debug lens: print `__story__`
    /// and exit nonzero if a hold failed (debug builds only).
    StoryFlush,
    /// The body of a `placeholder def`: calling it is an error naming it.
    Placeholder { def: String },
    /// Stop the run with `message` (ast `Lens::Stop`). `at` is the site, as
    /// for an index (`site_at`), so the error carries its location; `kind`
    /// is the error's kind (crate::errkind).
    Stop { message: String, at: u32, kind: crate::errkind::ErrKind },
    /// `czeros(d1..d4)`: a complex array of zeros. Children are the dimensions.
    CZeros,
    /// Read one element of a complex array: [base, idx...] -> `double complex`.
    CIndex { at: u32 },
    /// Store into a complex array: [base, idx..., value]. A real value promotes.
    CSetIndex { at: u32 },
    /// `show` of a complex array.
    ShowCArr { label: String },
    /// `fft(a)` / `ifft(a)`: the transform of a complex array, in place.
    Fft { inverse: bool },
    /// `[1.0 + 2.0i, 3.0]`: a rank-1 complex literal; children are its elements.
    CArrLit { n: usize },
    /// re/im/abs over a complex array (-> real array), or conj (-> complex).
    CArrFn { name: String },
    /// `plot(x, y, "label")` — the explicit figure statement; children [x, y].
    Plot { label: String },
    Scatter { label: String, color: String },
    /// `figure(title, xlabel, ylabel[, xlo, xhi, ylo, yhi[, style]])` —
    /// presentation for the NEXT chart, carried by the program itself.
    /// children: the 0 or 4 window expressions. For heatmaps the window is the
    /// physical extent (axes get real ticks); style "contour" overlays contour
    /// lines on smooth fields.
    Figure { title: String, xlabel: String, ylabel: String, style: String },
    /// `checkpoint()` — a top-level state-snapshot marker. Interpreter: no-op.
    /// C backend: with VIV_CKPT_DIR set, the binary saves every top-level
    /// number/array to <dir>/ck_<i>.bin at the i-th marker; with VIV_RESUME=i
    /// and VIV_CKPT_IN=<file>, the statements before marker i are skipped and
    /// the state is loaded instead. One binary serves full runs and resumes —
    /// the basis of per-cell notebook caching.
    Ckpt,
    /// `plot2d(Z, "label")` — a matrix as a heatmap/surface; children [Z].
    Plot2d { label: String },
    /// `quiver(x, y, u, v, "label")` — a vector field as arrows; children [x, y, u, v].
    Quiver { label: String },
    /// `plot3d(x, y, z, "label")` — a curve in 3-space; children [x, y, z].
    Plot3d { label: String },
    /// `scene("title")` — opens a 3D scene; no children.
    Scene { title: String },
    /// `wire(x, y, z[, v], "group")` — a scene polyline; children are the arrays.
    Wire { group: String },
    /// `verts(x, y, z[, v], "group")` — scene point markers; children are the arrays.
    Verts { group: String },
    /// One animation record. `head` is `verb: text ; text`,
    /// fixed at compile time; children are `arrays` arrays, then the numbers.
    Anim { head: String, arrays: u8 },
    /// `bars(x, y, "label")` -- a bar chart; children [x, y]. The histogram's chart.
    Bars { label: String },
    /// A 1-argument math builtin mapped elementwise over an array — `sin(x)`
    /// where x is an array. `cname` is the C/libm function name. children: [arr].
    MapMath { cname: String },
    /// An imaginary literal: `2.5i`.
    Imag(f64),
    /// `show` of a complex value. children: [expr].
    ShowCplx { label: String },
    /// `true` / `false`.
    BoolLit(bool),
    /// `show` of a boolean: prints `true` or `false`. children: [expr].
    ShowBool { label: String },
    /// `if cond { then } else { els }`. children: [cond, then(Block), els(Block)].
    If,
    /// `while cond { body }`. children: [cond, body(Block)].
    While,
    /// `for var to bound { body }`. children: [bound, body(Block)].
    For { var: String },
    /// `parallel for var to bound { body }`. children: [bound, body(Block)].
    /// `privates` are per-iteration temporaries; `reductions` are (op, var).
    ///
    /// `pins`: the arrays a memory lens pins as written disjointly by the
    /// passes (`crate::memory_lens`), which a checked build tests while the
    /// loop runs; `pin_what` names the loop and the lens file in the run-time
    /// error. Hashed only when there are pins, so an unpinned loop keeps its
    /// content address; `build_ir_plain` (the body hashes) never has pins.
    ParallelFor { var: String, privates: Vec<String>, reductions: Vec<(String, String)>, pins: Vec<String>, pin_what: String },
    Break,
    Continue,
    Num(f64),
    Var { name: String },
    /// An array literal with the given dimensions. children: scalar elements in
    /// row-major order.
    ArrLit { dims: Vec<usize> },
    /// `zeros(d0, d1, ...)`. children: [dim exprs]; rank = children.len().
    Zeros,
    /// Array indexing. children: [base, idx...]; rank = children.len() - 1.
    /// `at` is where the indexing is in the source (span start + 1) when the
    /// program is compiled with `with_sites`, and 0 otherwise. It keeps two
    /// identical-looking index expressions on different lines apart, so each
    /// can report its own location when an index is out of bounds; at 0 it is
    /// left out of the hash, so node ids (and body hashes) are what they were.
    Index { at: u32 },
    /// Arithmetic negation. children: [operand].
    Neg,
    /// Logical negation. children: [operand].
    Not,
    /// children: [lhs, rhs]. `cplx`: `+ - * /` with a complex operand, which
    /// the C backend compiles to a call (vmath.c's viv_cadd, viv_csub, viv_cmul,
    /// viv_cdiv), not to C's operator on `double complex`. It is left out of
    /// the content hash: the operands' types already say it.
    Bin { op: BinOp, cplx: bool },
    /// A resolved array operation. children depend on `kind`.
    Arr { kind: ArrKind },
    /// A call. children: [args...].
    Call { callee: String },
    /// A string literal. No children.
    StrLit(String),
    // ---- arrays of records, and writes into records. Records and arrays of records are values: a write through
    // one name is never seen through another.
    /// `[a, b, c]` of records of type `rec` (`[]` when there are none).
    /// children: the records.
    RecArrLit { rec: String },
    /// `xs[i]` of an array of records. children: [xs, i].
    RecIndex { rec: String, at: u32 },
    /// `xs[i] = r` into an array of records. children: [xs, i, r], `xs` a
    /// place (a variable, then fields and elements).
    RecSetIndex { rec: String, at: u32 },
    /// `len(xs)` of an array of records. children: [xs].
    RecLen,
    /// `append(xs, r)`: a new array holding xs's records and then r.
    /// children: [xs, r].
    RecAppend { rec: String },
    /// `base.field = value`, the field of type `ty`. children: [base,
    /// value], `base` a place.
    SetField { field: String, ty: Ty },
    /// A record, or an array of records, read from a place (a variable, a
    /// field, an element) and stored somewhere else: assigned, returned, put
    /// in a record or an array. It becomes a value of its own, so that a
    /// later write through either name is not seen through the other. Only
    /// a program that writes into records or holds arrays of records has
    /// these (`Sigs::owned`). children: [value].
    Own { ty: Ty },
    /// One C helper of a record type, in a program whose records can be
    /// written (`kind`: 0 `viv_own_T`, a record's own copy; 1
    /// `viv_own_recs_T`, an array of them; 2 `viv_keep_T` and 3
    /// `viv_keep_recs_T`, keeping on the arena what one holds; 4
    /// `viv_recs_lit_T`, an array from its records; 5 `viv_recs_push_T`, an
    /// array with one more). No children.
    RecFns { rec: String, kind: u8, fields: Vec<(String, Ty)> },
    /// An Int literal. No children.
    Int(i64),
    /// The program's enums and the optionals it uses:
    /// each name with its variants' names and payload types. A root, after
    /// every def; the C backend writes their structs and helpers from it.
    EnumDefs { enums: Vec<(String, Vec<(String, Vec<Ty>)>)> },
    /// A value of variant `tag` of enum `ty`. children: its payload.
    EnumNew { ty: String, tag: u32 },
    /// Whether the value is variant `tag`. children: [value].
    EnumIs { tag: u32 },
    /// Payload field `k` of variant `tag` of enum `ty`. children: [value].
    EnumField { ty: String, tag: u32, k: u32 },
    /// `==` (or `!=`, `ne`) of two values of an enum without payloads.
    /// children: [a, b].
    EnumEq { ne: bool },
    /// An enum value as text. children: [value].
    EnumStr { ty: String },
}

impl NodeKind for Viv {
    fn tag(&self) -> u32 {
        match self {
            Viv::Func { .. } => 1,
            Viv::Main { .. } => 2,
            Viv::Block => 3,
            Viv::Assign { .. } => 4,
            Viv::SetIndex { .. } => 5,
            Viv::Return => 6,
            Viv::ExprStmt => 7,
            Viv::If => 8,
            Viv::While => 9,
            Viv::For { .. } => 10,
            Viv::Break => 11,
            Viv::Continue => 12,
            Viv::Num(_) => 13,
            Viv::Var { .. } => 14,
            Viv::ArrLit { .. } => 15,
            Viv::Zeros => 16,
            Viv::Index { .. } => 17,
            Viv::Neg => 18,
            Viv::Not => 19,
            Viv::Bin { .. } => 20,
            Viv::Arr { .. } => 21,
            Viv::Call { .. } => 22,
            Viv::Nop => 23,
            Viv::Global { .. } => 24,
            Viv::LambdaDef { .. } => 25,
            Viv::Closure { .. } => 26,
            Viv::Show { .. } => 27,
            Viv::ParallelFor { .. } => 28,
            Viv::RecordDef { .. } => 29,
            Viv::RecordLit { .. } => 30,
            Viv::Field { .. } => 31,
            Viv::Narrate { .. } => 32,
            Viv::Plot { .. } => 33,
            Viv::Plot2d { .. } => 34,
            Viv::Plot3d { .. } => 35,
            Viv::Scatter { .. } => 36,

            Viv::Imag(..) => 37,
            Viv::ShowCplx { .. } => 38,
            Viv::Figure { .. } => 39,
            Viv::Ckpt => 40,
            Viv::Quiver { .. } => 41,
            Viv::Scene { .. } => 42,
            Viv::Wire { .. } => 43,
            Viv::Verts { .. } => 44,
            Viv::FnRef { .. } => 45,
            Viv::FnWrapDef { .. } => 46,
            Viv::ApplyFn { .. } => 47,
            Viv::TestBegin { .. } => 48,
            Viv::TestEnd => 49,
            Viv::CZeros => 51,
            Viv::CIndex { .. } => 52,
            Viv::CSetIndex { .. } => 53,
            Viv::ShowCArr { .. } => 54,
            Viv::Fft { .. } => 55,
            Viv::CArrLit { .. } => 56,
            Viv::CArrFn { .. } => 57,
            Viv::BoolLit(..) => 58,
            Viv::ShowBool { .. } => 59,
            Viv::Anim { .. } => 60,
            Viv::Bars { .. } => 61,
            Viv::FnArrLit { .. } => 62,
            Viv::FnIndex { .. } => 63,
            Viv::FnLen => 64,
            Viv::StepMark { .. } => 65,
            Viv::StepReturn { .. } => 66,
            Viv::StepsFlush => 67,
            Viv::LensReturn { .. } => 68,
            Viv::StoryCall { .. } => 69,
            Viv::StoryIter { .. } => 70,
            Viv::StoryIterEnd { .. } => 71,
            Viv::StoryWatch { .. } => 72,
            Viv::StoryHold { .. } => 73,
            Viv::StoryFlush => 74,
            Viv::Placeholder { .. } => 75,
            Viv::Stop { .. } => 76,
            Viv::Print { .. } => 77,
            Viv::StrLit(..) => 78,
            Viv::StoryRet { .. } => 79,
            Viv::RecArrLit { .. } => 80,
            Viv::RecIndex { .. } => 81,
            Viv::RecSetIndex { .. } => 82,
            Viv::RecLen => 83,
            Viv::RecAppend { .. } => 84,
            Viv::SetField { .. } => 85,
            Viv::Own { .. } => 86,
            Viv::RecFns { .. } => 87,
            Viv::Int(_) => 88,
            Viv::EnumDefs { .. } => 89,
            Viv::EnumNew { .. } => 90,
            Viv::EnumIs { .. } => 91,
            Viv::EnumField { .. } => 92,
            Viv::EnumEq { .. } => 93,
            Viv::EnumStr { .. } => 94,
            // 50, not 36: MapMath shared Scatter's tag, so the two kinds were
            // distinguished only by their payload bytes happening to differ.
            Viv::MapMath { .. } => 50,
        }
    }

    fn hash_payload(&self, h: &mut Hasher) {
        let s = |h: &mut Hasher, bytes: &[u8]| {
            h.update(&(bytes.len() as u32).to_le_bytes());
            h.update(bytes);
        };
        match self {
            Viv::Func { name, src: _, params, ret, locals, arena: _ } => {
                s(h, name.as_bytes());
                for (n, t) in params {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
                ret.hash_into(h);
                h.update(&[0]);
                for (n, t) in locals {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
            }
            Viv::Main { locals, globals } => {
                for (n, t) in locals {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
                h.update(&[0]);
                for (n, t) in globals {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
            }
            Viv::Assign { name } | Viv::Var { name } | Viv::For { var: name } | Viv::Call { callee: name } => {
                s(h, name.as_bytes());
            }
            Viv::Global { name, ty } => {
                s(h, name.as_bytes());
                ty.hash_into(h);
            }
            Viv::RecordDef { name, fields } => {
                s(h, name.as_bytes());
                for (n, t) in fields {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
            }
            Viv::RecordLit { ty_name, fields } => {
                s(h, ty_name.as_bytes());
                for f in fields {
                    s(h, f.as_bytes());
                }
            }
            Viv::Field { field } => {
                s(h, field.as_bytes());
            }
            Viv::SetField { field, ty } => {
                s(h, field.as_bytes());
                ty.hash_into(h);
            }
            Viv::RecArrLit { rec } | Viv::RecAppend { rec } => s(h, rec.as_bytes()),
            Viv::RecIndex { rec, at } | Viv::RecSetIndex { rec, at } => {
                s(h, rec.as_bytes());
                if *at != 0 {
                    h.update(&at.to_le_bytes());
                }
            }
            Viv::Own { ty } => ty.hash_into(h),
            Viv::RecFns { rec, kind, fields } => {
                s(h, rec.as_bytes());
                h.update(&[*kind]);
                for (n, t) in fields {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
            }
            Viv::LambdaDef { name, param, captures } => {
                s(h, name.as_bytes());
                s(h, param.as_bytes());
                for (n, t) in captures {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
            }
            Viv::FnRef { name, sig } | Viv::FnWrapDef { name, sig } => {
                s(h, name.as_bytes());
                sig.hash_into(h);
            }
            Viv::FnArrLit { n, sig } => {
                h.update(&(*n as u32).to_le_bytes());
                sig.hash_into(h);
            }
            Viv::FnIndex { sig, at } => {
                sig.hash_into(h);
                if *at != 0 {
                    h.update(&at.to_le_bytes());
                }
            }
            // `at` is hashed only when it is set (see `Viv::Index`), so a
            // program compiled without sites has the ids it always had
            Viv::Index { at } | Viv::SetIndex { at } | Viv::CIndex { at } | Viv::CSetIndex { at } => {
                if *at != 0 {
                    h.update(&at.to_le_bytes());
                }
            }
            Viv::ApplyFn { sig } => {
                sig.hash_into(h);
            }
            Viv::Closure { name, captures } => {
                s(h, name.as_bytes());
                for c in captures {
                    s(h, c.as_bytes());
                }
            }
            Viv::Show { label, is_arr } => {
                s(h, label.as_bytes());
                h.update(&[*is_arr as u8]);
            }
            Viv::Narrate { msg } => {
                s(h, msg.as_bytes());
            }
            Viv::Print { parts } => {
                for p in parts {
                    match p {
                        PrintPart::Text(t) => {
                            h.update(&[0]);
                            s(h, t.as_bytes());
                        }
                        PrintPart::Num(f) | PrintPart::Arr(f) => {
                            h.update(&[if matches!(p, PrintPart::Num(_)) { 1 } else { 2 }]);
                            s(h, f.encode().as_bytes());
                        }
                        PrintPart::Bool => {
                            h.update(&[3]);
                        }
                        PrintPart::Cplx => {
                            h.update(&[4]);
                        }
                        PrintPart::Str => {
                            h.update(&[6]);
                        }
                        PrintPart::Map(kf, vf) | PrintPart::SMap(kf, vf) => {
                            h.update(&[if matches!(p, PrintPart::Map(..)) { 5 } else { 7 }]);
                            s(h, kf.encode().as_bytes());
                            s(h, vf.encode().as_bytes());
                        }
                    }
                }
            }
            Viv::StrLit(text) => {
                s(h, text.as_bytes());
            }
            Viv::TestBegin { name } => {
                s(h, name.as_bytes());
            }
            Viv::StepMark { enter, id, depth, story, def } => {
                h.update(&[*enter as u8]);
                s(h, id.as_bytes());
                h.update(&depth.to_le_bytes());
                if *story {
                    h.update(&[1]);
                }
                if let Some(d) = def {
                    h.update(&[2]);
                    s(h, d.as_bytes());
                }
            }
            Viv::LensReturn { ty } => ty.hash_into(h),
            Viv::StoryCall { def } | Viv::StoryRet { def } | Viv::StoryIter { def } | Viv::StoryIterEnd { def } | Viv::Placeholder { def } => {
                s(h, def.as_bytes());
            }
            // the kind is left out, so a def's hash is what it was before
            // stops carried one; the message decides it
            Viv::Stop { message, at, .. } => {
                s(h, message.as_bytes());
                h.update(&at.to_le_bytes());
            }
            Viv::StoryWatch { def, at, names } => {
                s(h, def.as_bytes());
                s(h, at.as_bytes());
                for (n, t) in names {
                    s(h, n.as_bytes());
                    t.hash_into(h);
                }
            }
            Viv::StoryHold { def, at, sentence, near } => {
                s(h, def.as_bytes());
                s(h, at.as_bytes());
                s(h, sentence.as_bytes());
                h.update(&[match near { None => 0, Some(false) => 1, Some(true) => 2 }]);
            }
            Viv::StepReturn { exits, ty } => {
                for (def, id, depth) in exits {
                    s(h, id.as_bytes());
                    h.update(&depth.to_le_bytes());
                    if let Some(d) = def {
                        h.update(&[2]);
                        s(h, d.as_bytes());
                    }
                }
                h.update(&[0]);
                ty.hash_into(h);
            }
            Viv::ShowCArr { label } => {
                s(h, label.as_bytes());
            }
            Viv::Fft { inverse } => {
                h.update(&[*inverse as u8]);
            }
            Viv::CArrLit { n } => {
                h.update(&(*n as u32).to_le_bytes());
            }
            Viv::CArrFn { name } => {
                s(h, name.as_bytes());
            }
            Viv::Plot { label } => {
                s(h, label.as_bytes());
            }
            Viv::Scatter { label, color } => {
                s(h, label.as_bytes());
                s(h, color.as_bytes());
            }
            Viv::Figure { title, xlabel, ylabel, style } => {
                s(h, title.as_bytes());
                s(h, xlabel.as_bytes());
                s(h, ylabel.as_bytes());
                s(h, style.as_bytes());
            }
            Viv::Plot2d { label } => {
                s(h, label.as_bytes());
            }
            Viv::Quiver { label } => {
                s(h, label.as_bytes());
            }
            Viv::Plot3d { label } => {
                s(h, label.as_bytes());
            }
            Viv::Scene { title } => {
                s(h, title.as_bytes());
            }
            Viv::Wire { group } => {
                s(h, group.as_bytes());
            }
            Viv::Verts { group } => {
                s(h, group.as_bytes());
            }
            Viv::Anim { head, arrays } => {
                s(h, head.as_bytes());
                h.update(&[*arrays]);
            }
            Viv::Bars { label } => {
                s(h, label.as_bytes());
            }
            Viv::MapMath { cname } => {
                s(h, cname.as_bytes());
            }
            Viv::Imag(v) => {
                s(h, &v.to_bits().to_le_bytes());
            }
            Viv::ShowCplx { label } => {
                s(h, label.as_bytes());
            }
            Viv::BoolLit(v) => {
                h.update(&[*v as u8]);
            }
            Viv::ShowBool { label } => {
                s(h, label.as_bytes());
            }
            Viv::ParallelFor { var, privates, reductions, pins, pin_what } => {
                s(h, var.as_bytes());
                for p in privates {
                    s(h, p.as_bytes());
                }
                h.update(&[0]);
                for (op, v) in reductions {
                    s(h, op.as_bytes());
                    s(h, v.as_bytes());
                }
                if !pins.is_empty() {
                    h.update(&[1]);
                    for p in pins {
                        s(h, p.as_bytes());
                    }
                    s(h, pin_what.as_bytes());
                }
            }
            Viv::Int(v) => {
                h.update(&v.to_le_bytes());
            }
            Viv::EnumDefs { enums } => {
                for (n, vs) in enums {
                    s(h, n.as_bytes());
                    for (v, p) in vs {
                        s(h, v.as_bytes());
                        h.update(&(p.len() as u32).to_le_bytes());
                        for t in p {
                            t.hash_into(h);
                        }
                    }
                }
            }
            Viv::EnumNew { ty, tag } => {
                s(h, ty.as_bytes());
                h.update(&tag.to_le_bytes());
            }
            Viv::EnumIs { tag } => {
                h.update(&tag.to_le_bytes());
            }
            Viv::EnumField { ty, tag, k } => {
                s(h, ty.as_bytes());
                h.update(&tag.to_le_bytes());
                h.update(&k.to_le_bytes());
            }
            Viv::EnumEq { ne } => {
                h.update(&[*ne as u8]);
            }
            Viv::EnumStr { ty } => s(h, ty.as_bytes()),
            Viv::Num(v) => {
                h.update(&v.to_bits().to_le_bytes());
            }
            Viv::Bin { op, .. } => {
                h.update(&[op.tag()]);
            }
            Viv::Arr { kind } => {
                h.update(&[kind.tag()]);
            }
            Viv::ArrLit { dims } => {
                for d in dims {
                    h.update(&(*d as u32).to_le_bytes());
                }
            }
            Viv::Block
            | Viv::Nop
            | Viv::Return
            | Viv::ExprStmt
            | Viv::If
            | Viv::While
            | Viv::Break
            | Viv::Continue
            | Viv::Zeros
            | Viv::Neg
            | Viv::Ckpt
            | Viv::FnLen
            | Viv::RecLen
            | Viv::TestEnd
            | Viv::StepsFlush
            | Viv::StoryFlush
            | Viv::CZeros
            | Viv::Not => {}
        }
    }
}

// ───────────────────────────── checked lowering ────────────────────────────

/// Per-function scope: name -> type, plus the order non-parameter locals were
/// introduced (for deterministic hoisted declarations).
struct Scope<'g> {
    /// Parameters and function-locals (empty for the top-level/main scope).
    bound: HashMap<String, Ty>,
    /// Non-parameter locals in introduction order (for hoisted declarations).
    locals: Vec<(String, Ty)>,
    /// File-scope globals (read-only lookup fallback).
    globals: &'g HashMap<String, Ty>,
    /// In a function, an assignment creates a local (shadowing any global). At
    /// the top level, an assignment writes the global itself.
    in_function: bool,
    /// Set once any expression lowered in this scope has array or record type.
    /// A function where it stays false cannot allocate an array, and compiles
    /// exactly as it did before the array arena existed.
    touches_arrays: std::cell::Cell<bool>,
    /// The steps open around the statement being lowered, innermost last, as
    /// (id, depth); `None` marks a loop boundary. A `return` inside steps
    /// closes all of them, a `break`/`continue` those inside its loop.
    open_steps: Vec<Open>,
    /// The types inference gave the def's locals, for a local bound to `[]`
    /// before the appends that say what it holds.
    inferred: Option<&'g HashMap<String, Ty>>,
    /// The def the steps of this body are recorded under by default (the
    /// `Func`'s `src`, or `<main>`); a step written in another def (inlined
    /// by `crate::grad`) carries its own.
    step_src: String,
}

/// What is open around a statement being lowered, innermost last.
#[derive(Clone)]
enum Open {
    /// a step, as (its def when not the enclosing def's `src`, id, depth)
    Step(Option<String>, String, u32),
    /// a loop; for an iteration loop of the debug lens, the statements that
    /// end each iteration (`Lens::Iter`'s `end`)
    Loop(Option<Vec<Stmt>>),
}

impl<'g> Scope<'g> {
    fn function(params: &[(String, Ty)], globals: &'g HashMap<String, Ty>) -> Self {
        Self { bound: params.iter().cloned().collect(), locals: Vec::new(), globals, in_function: true, touches_arrays: std::cell::Cell::new(false), open_steps: Vec::new(), inferred: None, step_src: String::new() }
    }
    fn top_level(globals: &'g HashMap<String, Ty>) -> Self {
        Self { bound: HashMap::new(), locals: Vec::new(), globals, in_function: false, touches_arrays: std::cell::Cell::new(false), open_steps: Vec::new(), inferred: None, step_src: crate::parser::MAIN_DEF.to_string() }
    }
    /// Resolve a name: local/param first, then file-scope global.
    fn get(&self, name: &str) -> Option<Ty> {
        self.bound.get(name).copied().or_else(|| self.globals.get(name).copied())
    }
    /// Whether the name is a parameter or local of the current function (i.e.
    /// must be captured by a nested closure, rather than referenced as a global).
    fn is_local(&self, name: &str) -> bool {
        self.bound.contains_key(name)
    }
    /// Bind a name to a type. Errors (returning the prior type) on a
    /// type-changing rebind — Viv has none.
    fn bind(&mut self, name: &str, ty: Ty) -> Result<(), Ty> {
        if self.in_function {
            match self.bound.get(name) {
                Some(&existing) if existing != ty => Err(existing),
                Some(_) => Ok(()),
                None => {
                    self.bound.insert(name.to_string(), ty);
                    self.locals.push((name.to_string(), ty));
                    Ok(())
                }
            }
        } else {
            // Top level: the binding is a global; its type is fixed by inference.
            match self.globals.get(name) {
                Some(&g) if g != ty => Err(g),
                _ => Ok(()),
            }
        }
    }
}

/// A declared record type: its interned name and ordered (field, type) layout.
struct RecordInfo {
    name: &'static str,
    fields: Vec<(String, Ty)>,
    /// For each field, whether its type is inferred (a record
    /// `crate::tuples` declared for a def that returns several values)
    /// and, if so, whether inference has found it yet. A field never
    /// found is a number, as an untyped return is.
    open: Vec<Option<bool>>,
}

/// Inferred function signatures: arity, parameter types, and return type.
/// `None` means "not yet determined" during inference (defaults to `Num`).
struct Sigs {
    arity: HashMap<String, usize>,
    params: HashMap<String, Vec<Option<Ty>>>,
    ret: HashMap<String, Option<Ty>>,
    /// Types of top-level variables (file-scope globals), readable in functions.
    globals: HashMap<String, Ty>,
    /// Declared record types, keyed by name (the registry, §4.1).
    records: HashMap<String, RecordInfo>,
    /// Whether indexing an unknown variable may default it to `Arr(Num)`. Off
    /// for the first fixpoint so call sites get to say `Arr(Cplx)` first.
    index_default: bool,
    /// The last fixpoint, run once every non-function type has settled. Only
    /// then is a function value's signature final, so only then does `apply`
    /// report its result type -- earlier, a provisional signature would fix the
    /// type of whatever the result is assigned to, and fix it wrongly.
    fn_final: bool,
    /// The program carries a debug lens: its step events also go into the
    /// story, and `main` ends by printing it.
    story: bool,
    /// The parameters (def, index) that real evidence never typed: still
    /// unknown after the first fixpoint, so their type is a default (indexing
    /// makes an array, the rest numbers), not something a call site said.
    guessed: std::collections::HashSet<(String, usize)>,
    /// The parameters (def, index) written without a type: one passed both
    /// Ints and numbers is a number (`infer_learn`).
    untyped: std::collections::HashSet<(String, usize)>,
    /// Each def's locals as inference last typed them.
    local_tys: HashMap<String, HashMap<String, Ty>>,
    /// The `parallel for` loops a memory lens pins (`Program::memory`).
    pinned: Vec<crate::parallel::Pinned>,
    /// The program writes into records or holds arrays of records, so a
    /// record read from a place and stored elsewhere is copied (`Viv::Own`).
    /// A program with neither is lowered exactly as before.
    owned: bool,
    /// The defs the lowering writes for operations over records
    /// (`lower::records`), by name, asked for while the program is lowered
    /// and made after it.
    rec_fns: std::cell::RefCell<std::collections::BTreeMap<String, records::RecFn>>,
}

impl Sigs {
    /// The signature a def presents when it is used as a VALUE: its arity, and
    /// which of its parameters are functions. The mask comes from the parameter
    /// types, which is where `defun` records what it found.
    fn fn_sig(&self, f: &str) -> Option<FnSig> {
        let arity = *self.arity.get(f)?;
        let mut kinds = 0u32;
        for j in 0..arity.min(16) {
            let k = ArgKind::of(self.param_ty(f, j)).unwrap_or(ArgKind::Num);
            kinds |= k.bits() << (2 * j);
        }
        let ret = match self.ret.get(f).copied().flatten() {
            Some(Ty::Arr(Elem::Num)) => ArgKind::Arr,
            _ => ArgKind::Num,
        };
        Some(FnSig { arity: arity as u8, kinds, ret })
    }
    fn param_ty(&self, f: &str, j: usize) -> Ty {
        self.params.get(f).and_then(|v| v.get(j)).copied().flatten().unwrap_or(Ty::Num)
    }
    fn ret_ty(&self, f: &str) -> Ty {
        self.ret.get(f).copied().flatten().unwrap_or(Ty::Num)
    }
    /// The nominal type of a declared record, by name.
    fn record_ty(&self, name: &str) -> Option<Ty> {
        self.records.get(name).map(|r| Ty::Record(r.name))
    }
}

/// Resolve a source type annotation to a `Ty`, using the record registry for
/// named (record) types. Errors if a named type was never declared.
fn resolve_type(t: &TypeExpr, records: &HashMap<String, RecordInfo>) -> Result<Ty, Diagnostic> {
    match t {
        TypeExpr::Num => Ok(Ty::Num),
        TypeExpr::Arr => Ok(Ty::Arr(Elem::Num)),
        TypeExpr::CArr => Ok(Ty::Arr(Elem::Cplx)),
        TypeExpr::Bool => Ok(Ty::Bool),
        TypeExpr::Str => Ok(Ty::Str),
        TypeExpr::StrList => Ok(Ty::Arr(Elem::Str)),
        TypeExpr::Int => Ok(Ty::Int),
        TypeExpr::IntList => Ok(Ty::Arr(Elem::Int)),
        TypeExpr::F32 => Ok(Ty::F32),
        TypeExpr::Packed(k) => Ok(Ty::Arr(Elem::P(*k))),
        TypeExpr::RecList { name, span } => records.get(name).map(|r| Ty::Recs(r.name)).ok_or_else(|| {
            // an array of enums is not supported yet
            if enums::is_enum(name) {
                Diagnostic::error(format!("an array of enum values (`[{name}]`) is not supported yet"))
                    .with_primary(*span, "an enum")
                    .with_note("enums are not array elements, map values or payload arrays yet; hold the values in a record's fields, or in an [Int] of their variant numbers")
            } else {
                Diagnostic::error(format!("unknown type `{name}`")).with_primary(*span, "not a declared record")
            }
        }),
        TypeExpr::Fn { arity } => Ok(Ty::Func(FnSig::plain(*arity))),
        // inferred later (`RecordInfo::open`)
        TypeExpr::Infer => Ok(Ty::Num),
        // `Cplx` is the name `viv features` and the reference give the type;
        // `Complex` was the only one the compiler took (k1's adapters use it)
        TypeExpr::Named { name, .. } if (name == "Cplx" || name == "Complex") && !records.contains_key(name) => Ok(Ty::Cplx),
        // its kinds come from what the program stores (inference)
        TypeExpr::Named { name, .. } if name == "Map" => Ok(Ty::Map(Mk::Any, Mk::Any)),
        TypeExpr::Named { name, .. } if !records.contains_key(name) && enums::is_enum(name) => Ok(Ty::Enum(enums::info(name).expect("an enum").name)),
        TypeExpr::Named { name, span } => records
            .get(name)
            .map(|r| Ty::Record(r.name))
            .ok_or_else(|| Diagnostic::error(format!("unknown type `{name}`")).with_primary(*span, "not a declared record or enum")),
        TypeExpr::Opt(t) => {
            let inner = resolve_type(t, records)?;
            if !matches!(inner, Ty::Num | Ty::Int | Ty::Bool | Ty::Str | Ty::Enum(_)) {
                return Err(Diagnostic::error(format!("an optional cannot hold {} yet", art(inner))).with_primary(Span::synthetic(SourceId(0)), "here"));
            }
            Ok(enums::opt(inner))
        }
    }
}


/// The program's record declarations as the interpreter checks them: each
/// record's fields in declaration order with their types, validated by
/// `build_records` (the same errors the compile gives).
pub(crate) fn record_layouts(program: &Program) -> Result<Vec<(String, Vec<(String, Option<Ty>)>)>, Vec<Diagnostic>> {
    let (records, errs) = build_records(program);
    if !errs.is_empty() {
        return Err(errs);
    }
    Ok(records
        .into_iter()
        .map(|(name, info)| (name, info.fields.into_iter().zip(info.open).map(|((f, t), open)| (f, if open.is_some() { None } else { Some(t) })).collect()))
        .collect())
}

/// The program's records in an order where each comes after the records its
/// fields name, so a field may name a record declared later in the file (as
/// a def may call a later def). Otherwise declaration order. A record that
/// holds itself, directly or through others, is left in place: resolving its
/// field then reports the type as unknown.
fn record_order(program: &Program) -> Vec<&crate::ast::RecordDecl> {
    fn names(t: &TypeExpr, out: &mut Vec<String>) {
        match t {
            TypeExpr::Named { name, .. } | TypeExpr::RecList { name, .. } => out.push(name.clone()),
            TypeExpr::Opt(inner) => names(inner, out),
            _ => {}
        }
    }
    let decls: Vec<&crate::ast::RecordDecl> = program.items.iter().filter_map(|i| if let Item::Record(r) = i { Some(r) } else { None }).collect();
    let index: HashMap<&str, usize> = decls.iter().enumerate().map(|(k, r)| (r.name.as_str(), k)).collect();
    let mut state = vec![0u8; decls.len()]; // 0 new, 1 on the path, 2 done
    let mut out = Vec::with_capacity(decls.len());
    fn visit<'a>(k: usize, decls: &[&'a crate::ast::RecordDecl], index: &HashMap<&str, usize>, state: &mut [u8], out: &mut Vec<&'a crate::ast::RecordDecl>) {
        if state[k] != 0 {
            return;
        }
        state[k] = 1;
        let mut deps = Vec::new();
        for f in &decls[k].fields {
            names(&f.ty, &mut deps);
        }
        for d in deps {
            if let Some(&j) = index.get(d.as_str()) {
                if state[j] == 0 {
                    visit(j, decls, index, state, out);
                }
            }
        }
        state[k] = 2;
        out.push(decls[k]);
    }
    for k in 0..decls.len() {
        visit(k, &decls, &index, &mut state, &mut out);
    }
    out
}

/// Build and validate the record registry from the program's `record`
/// declarations (errors per spec §7: redeclaration, duplicate/Bool fields,
/// unknown field types, empty records, recursive types).
fn build_records(program: &Program) -> (HashMap<String, RecordInfo>, Vec<Diagnostic>) {
    let mut records: HashMap<String, RecordInfo> = HashMap::new();
    let mut errs = Vec::new();
    for rd in record_order(program) {
        if records.contains_key(&rd.name) {
            errs.push(Diagnostic::error(format!("record `{}` is already defined", rd.name)).with_primary(rd.name_span, "redefined here"));
            continue;
        }
        if rd.fields.is_empty() {
            errs.push(Diagnostic::error(format!("record `{}` has no fields", rd.name)).with_primary(rd.name_span, "empty records are not allowed"));
            continue;
        }
        let mut fields: Vec<(String, Ty)> = Vec::new();
        let mut open = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for f in &rd.fields {
            if !seen.insert(f.name.clone()) {
                errs.push(Diagnostic::error(format!("duplicate field `{}` in record `{}`", f.name, rd.name)).with_primary(f.name_span, "already declared"));
                continue;
            }
            // Records may not store Bool (it is a transient type); resolve the rest
            // against the records ordered before this one (record_order), which
            // forbids recursive cycles.
            // a `Map` field takes its kinds from the maps it is built from,
            // as a `Map` parameter does
            let map_field = matches!(&f.ty, TypeExpr::Named { name, .. } if name == "Map");
            let ty = match resolve_type(&f.ty, &records) {
                Ok(t) if map_field => t,
                Ok(t) => settle(t),
                Err(e) => {
                    errs.push(e);
                    Ty::Num
                }
            };
            fields.push((f.name.clone(), ty));
            open.push((matches!(f.ty, TypeExpr::Infer) || map_field).then_some(false));
        }
        let name: &'static str = Box::leak(rd.name.clone().into_boxed_str());
        records.insert(rd.name.clone(), RecordInfo { name, fields, open });
    }
    (records, errs)
}

/// The function whose body inference is currently walking.
struct Cur<'a> {
    name: &'a str,
    params: &'a [String],
    /// Locals assigned so far in this pass whose type is not known yet. In a
    /// function an assignment creates a local that shadows any global, so until
    /// `S = prim_overlap(...)` is typed, a later `S` is "unknown" -- not the
    /// global overlap MATRIX of the same name, which is what it used to resolve
    /// to whenever the callee's return type arrived a pass late.
    pending: std::cell::RefCell<std::collections::HashSet<String>>,
}

/// Infer parameter and return types across the whole program.
///
/// Viv operators are overloaded (`a + b` is scalar or array depending on
/// operands), so types cannot be derived bottom-up from operators alone. They
/// originate at concrete sources (literals, `zeros`, indexing) and flow through
/// call sites into callee parameters. A fixpoint propagates them until stable.
fn infer_signatures(program: &Program, records: HashMap<String, RecordInfo>) -> Sigs {
    let mut sigs = Sigs { arity: HashMap::new(), params: HashMap::new(), ret: HashMap::new(), globals: HashMap::new(), records, index_default: false, fn_final: false, story: false, guessed: Default::default(), untyped: Default::default(), local_tys: HashMap::new(), pinned: Vec::new(), owned: false, rec_fns: Default::default() };
    for item in &program.items {
        if let Item::Def(d) = item {
            sigs.arity.entry(d.name.clone()).or_insert(d.params.len());
            // Seed parameter slots, pre-filled from any explicit `: Type` annotation
            // (record-typed params must be annotated; §4.4). Unresolvable annotations
            // are left to the checked lowering to report.
            let slots: Vec<Option<Ty>> = d
                .params
                .iter()
                .map(|p| p.ty.as_ref().and_then(|t| resolve_type(t, &sigs.records).ok()))
                .collect();
            for (i, p) in d.params.iter().enumerate() {
                if p.ty.is_none() {
                    sigs.untyped.insert((d.name.clone(), i));
                }
            }
            sigs.params.entry(d.name.clone()).or_insert(slots);
            // Return type: an explicit annotation wins; otherwise a function with no
            // `return` returns nothing (C `void`), and the rest are inferred below.
            let init_ret = match &d.ret {
                Some(t) => resolve_type(t, &sigs.records).ok(),
                None if has_return(&d.body) => None,
                // a placeholder without `-> T` returns a number, Viv's default
                None if is_placeholder(&d.body) => Some(Ty::Num),
                None => Some(Ty::Void),
            };
            sigs.ret.entry(d.name.clone()).or_insert(init_ret);
        }
    }
    // A global is a name the top level binds. A name the top level only reads
    // (`print(nosuch)`) is not one: its type would be guessed from the call it
    // is passed to, and the compiled program would read a zero where the
    // interpreter reports the unknown name (bug B299).
    let top_bound = top_level_names(program);
    // Three bounded fixpoints. The first propagates only real evidence; the
    // second also lets indexing default an unknown variable to `Arr(Num)`; the
    // third, with every other type settled, lets `apply` report the result
    // type its callee's (now final) signature promises.
    // Each converges in at most call-graph-depth iterations.
    for phase in 0..3 {
    if phase == 1 {
        sigs.guessed = sigs.params.iter().flat_map(|(f, slots)| slots.iter().enumerate().filter(|(_, t)| t.is_none()).map(move |(i, _)| (f.clone(), i))).collect();
    }
    sigs.index_default = phase >= 1;
    sigs.fn_final = phase == 2;
    for _ in 0..64 {
        let mut changed = false;
        // Top-level statements share one environment, seeded from known globals;
        // their bindings become file-scope globals.
        let mut tenv = sigs.globals.clone();
        for item in &program.items {
            if let Item::Stmt(s) = item {
                infer_stmt(s, &mut tenv, None, &mut sigs, &mut changed);
            }
        }
        for (k, v) in tenv {
            if !top_bound.contains(&k) && !sigs.globals.contains_key(&k) {
                continue;
            }
            let fresh = match sigs.globals.get(&k) {
                None => true,
                Some(&old) => revises(old, v),
            };
            if fresh {
                sigs.globals.insert(k, v);
                changed = true;
            }
        }
        // Function bodies: params seeded with current inferred types; free
        // variables resolve against globals.
        for item in &program.items {
            if let Item::Def(d) = item {
                let names: Vec<String> = d.params.iter().map(|p| p.name.clone()).collect();
                let mut env: HashMap<String, Ty> = HashMap::new();
                // a local `[]` that an earlier pass found to be a list of
                // strings stays one: the later phases default `[]` to numbers
                if let Some(prev) = sigs.local_tys.get(&d.name) {
                    for (k, t) in prev {
                        if matches!(t, Ty::Arr(Elem::Str) | Ty::Recs(_)) && !names.contains(k) {
                            env.insert(k.clone(), *t);
                        }
                    }
                }
                for (i, p) in d.params.iter().enumerate() {
                    if let Some(t) = sigs.params[&d.name][i] {
                        env.insert(p.name.clone(), t);
                    }
                }
                let cur = Cur { name: &d.name, params: &names, pending: Default::default() };
                for s in &d.body {
                    infer_stmt(s, &mut env, Some(&cur), &mut sigs, &mut changed);
                }
                sigs.local_tys.insert(d.name.clone(), env);
            }
        }
        if !changed {
            break;
        }
    }
    }
    // a map nothing stored into holds numbers
    for slots in sigs.params.values_mut() {
        for t in slots.iter_mut().flatten() {
            *t = settle(*t);
        }
    }
    for t in sigs.ret.values_mut().flatten() {
        *t = settle(*t);
    }
    for t in sigs.globals.values_mut() {
        *t = settle(*t);
    }
    for info in sigs.records.values_mut() {
        for (_, t) in info.fields.iter_mut() {
            *t = settle(*t);
        }
    }
    for env in sigs.local_tys.values_mut() {
        for t in env.values_mut() {
            *t = settle(*t);
        }
    }
    sigs
}

/// The known type of a callee's parameter `j` (builtin or user function), or
/// `None` if not yet determined. Used to push types backward into arguments.
fn callee_param_ty(callee: &str, j: usize, sigs: &Sigs) -> Option<Ty> {
    // a program's own def wins over a builtin of the same name
    if sigs.arity.contains_key(callee) {
        return sigs.params.get(callee).and_then(|v| v.get(j)).copied().flatten();
    }
    // the array vocabulary (interp/arrays.rs): what argument j is in every
    // form of the name that has one
    if let Some(t) = array_param_ty(callee, j) {
        return t;
    }
    match callee {
        // a math function that takes a complex number too says "a number"
        // only in the last pass, once every other type is settled: `zv, zpv =
        // f(z)` with f's complex results found late (lib/zfunc's, in the pass
        // that defaults indexing), then `abs(zv)`, typed zv a number before
        // the assignment's complex arrived ("type cannot change")
        "exp" | "sqrt" | "abs" | "sin" | "cos" | "tan" | "log" if !sigs.fn_final => None,
        "print" | "exp" | "sqrt" | "pow" | "assert_near" | "assert_true" | "assert_gt" | "assert_lt" | "srand" => Some(Ty::Num),
        "sin" | "cos" | "tan" | "log" | "abs" | "fabs" | "floor" | "ceil" | "min" | "max"
        | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh" | "erf" | "gamma" | "round" | "mod" | "fma"
        | "band" | "bor" | "bxor" | "bnot" | "shl" | "shr" | "sar" | "u32" | "i32" => Some(Ty::Num),
        "zeros" => Some(Ty::Num),
        "transpose" | "eig_sym" | "spectrum_sym" | "lin_solve" | "lin_solve_multi" | "cholesky" | "eig_general" | "sort" | "tridiag_solve" | "text" => Some(Ty::Arr(Elem::Num)),
        // linspace(lo, hi, n) takes three numbers
        "linspace" => Some(Ty::Num),
        "roll" => Some(if j == 0 { Ty::Arr(Elem::Num) } else { Ty::Num }),
        "copy" | "concat" | "hcat" | "vcat" => Some(Ty::Arr(Elem::Num)),
        "append" if j == 0 => Some(Ty::Arr(Elem::Num)),
        "extend" => Some(Ty::Arr(Elem::Num)),
        // a map first, then numbers
        c if crate::interp::maps::NAMES.contains(&c) => Some(if j == 0 { Ty::Map(Mk::Any, Mk::Any) } else { Ty::Num }),
        c if (c == SLICE_GET || c == SLICE_SET) && j < 2 => Some(Ty::Arr(Elem::Num)),
        "all" | "any" if !sigs.arity.contains_key(callee) => Some(Ty::Arr(Elem::Num)),
        "where" if j == 0 && !sigs.arity.contains_key(callee) => Some(Ty::Arr(Elem::Num)),
        c if crate::interp::strings::is_builtin(c) => strings::param_ty(c, j),
        _ => sigs.params.get(callee).and_then(|v| v.get(j)).copied().flatten(),
    }
}

/// What argument `j` of a builtin of the array vocabulary is, when every form
/// of the name with that many arguments agrees: `Some(None)` for an argument
/// that may be a number or an array, `None` when the name is not one of them.
fn array_param_ty(callee: &str, j: usize) -> Option<Option<Ty>> {
    use crate::interp::arrays::SIGS;
    // a name an older builtin also has keeps the older rule: `take` and
    // `transpose` take complex arrays too, so a parameter passed to one is not
    // thereby real
    if matches!(callee, "take" | "transpose" | "sum" | "any" | "all" | "linspace" | "rand" | "lin_solve" | "cholesky") {
        return None;
    }
    let kinds: Vec<char> = SIGS.iter().filter(|s| s.name == callee).filter_map(|s| s.params.chars().nth(j)).collect();
    let first = *kinds.first()?;
    if kinds.iter().any(|&k| k != first) {
        return Some(None);
    }
    Some(match first {
        'a' => Some(Ty::Arr(Elem::Num)),
        'n' => Some(Ty::Num),
        'b' => Some(Ty::Bool),
        's' => Some(Ty::Str),
        _ => None,
    })
}

/// The type a call of the array vocabulary gives (interp/arrays.rs, `SIGS`),
/// from its arguments' types; `None` while one it depends on is not known.
fn array_call_ty(sig: &crate::interp::arrays::Sig, args: &[Expr], arg_tys: &[Option<Ty>], sigs: &Sigs) -> Option<Ty> {
    match sig.ret {
        'a' => Some(Ty::Arr(Elem::Num)),
        'x' => match arg_tys.first().copied().flatten() {
            Some(Ty::Num) => Some(Ty::Num),
            Some(_) => Some(Ty::Arr(Elem::Num)),
            None if sigs.index_default => Some(Ty::Arr(Elem::Num)),
            None => None,
        },
        'e' => match args.first() {
            Some(Expr::Str { value, .. }) => match crate::interp::arrays::einsum_rank(value, args.len() - 1) {
                Ok(0) => Some(Ty::Num),
                _ => Some(Ty::Arr(Elem::Num)),
            },
            _ => Some(Ty::Arr(Elem::Num)),
        },
        _ => Some(Ty::Num),
    }
}

fn infer_learn(env: &mut HashMap<String, Ty>, cur: Option<&Cur>, sigs: &mut Sigs, changed: &mut bool, name: &str, ty: Ty) {
    // Only learn when the type is not already known — never override (e.g. don't
    // turn a known `Num` into `Arr` just because it was (mis)indexed; let the
    // checked lowering report that as an error).
    // a parameter with no type that holds an Int and is given a number
    // (`x = x * 0.5`) is a number (as one passed Ints and numbers is)
    let param_ix = cur.and_then(|c| c.params.iter().position(|p| p == name).map(|i| (c.name.to_string(), i)));
    let mixed = env.get(name) == Some(&Ty::Int) && ty == Ty::Num && param_ix.as_ref().is_some_and(|k| sigs.untyped.contains(k));
    if let Some(&old) = env.get(name) {
        if !revises(old, ty) && !mixed {
            return;
        }
    }
    env.insert(name.to_string(), ty);
    if let Some(c) = cur {
        if let Some(idx) = c.params.iter().position(|p| p == name) {
            let slot = &mut sigs.params.get_mut(c.name).unwrap()[idx];
            if slot.is_none() || slot.is_some_and(|o| revises(o, ty)) || mixed {
                *slot = Some(ty);
                *changed = true;
            }
        }
    }
}

/// The kind a map key or value of type `t` gives its map.
fn mk_of(t: Option<Ty>) -> Option<Mk> {
    match t {
        // an Int key or value is stored as the number it is (exactly)
        Some(Ty::Num | Ty::Int) => Some(Mk::Num),
        Some(Ty::Str) => Some(Mk::Str),
        _ => None,
    }
}

/// `name`, a map, stores (or looks up) a key of type `k` and a value of type
/// `v`: its kinds grow to take them.
fn infer_map_kinds(env: &mut HashMap<String, Ty>, cur: Option<&Cur>, sigs: &mut Sigs, changed: &mut bool, name: &str, k: Option<Ty>, v: Option<Ty>) {
    infer_map_grow(env, cur, sigs, changed, name, mk_of(k).unwrap_or(Mk::Any), mk_of(v).unwrap_or(Mk::Any));
}

/// `name`, a map, has keys of kind `k` and values of kind `v` (a def it is
/// passed to stores them): its kinds grow to take them.
fn infer_map_grow(env: &mut HashMap<String, Ty>, cur: Option<&Cur>, sigs: &mut Sigs, changed: &mut bool, name: &str, k: Mk, v: Mk) {
    let local = env.get(name).copied();
    // a def storing into a module's map: the global's kinds grow
    let shadowed = cur.is_some_and(|c| c.params.iter().any(|p| p == name) || c.pending.borrow().contains(name));
    let (Some(Ty::Map(k0, v0)), global) = (match local {
        Some(t) => (Some(t), false),
        None if !shadowed => (sigs.globals.get(name).copied(), true),
        None => (None, false),
    }) else {
        return;
    };
    let (k1, v1) = (k.max(k0), v.max(v0));
    if (k1, v1) == (k0, v0) {
        return;
    }
    if global {
        sigs.globals.insert(name.to_string(), Ty::Map(k1, v1));
    } else {
        infer_learn(env, cur, sigs, changed, name, Ty::Map(k1, v1));
    }
    *changed = true;
}

fn infer_stmt(stmt: &Stmt, env: &mut HashMap<String, Ty>, cur: Option<&Cur>, sigs: &mut Sigs, changed: &mut bool) {
    match stmt {
        Stmt::Assign { name, value, .. } => {
            // the value of `a if c else b` (a hidden variable the parser
            // writes): an integer literal on one side takes the other side's
            // type, an Int, so it is held back until that is known
            // (`return 0 if n == 0 else n`)
            if name.starts_with("viv__if") && !env.contains_key(name) && ints::adapts(value, sigs) {
                let prior = cur.and_then(|c| sigs.local_tys.get(c.name)).and_then(|m| m.get(name)).copied();
                if prior == Some(Ty::Int) {
                    env.insert(name.clone(), Ty::Int);
                    return;
                }
                if !sigs.index_default {
                    return;
                }
            }
            // A variable's type is fixed by its first binding (Viv forbids
            // type-changing rebinds); don't let a later assignment retype it.
            if let Some(t) = infer_expr(value, env, cur, sigs, changed) {
                match env.get(name) {
                    Some(&old) if revises(old, t) => {
                        env.insert(name.clone(), t);
                        *changed = true;
                    }
                    // a parameter written without a type, passed an Int and
                    // given a number here, is a number (infer_learn)
                    Some(&Ty::Int) if t == Ty::Num && cur.is_some_and(|c| c.params.iter().position(|p| p == name).is_some_and(|i| sigs.untyped.contains(&(c.name.to_string(), i)))) => {
                        infer_learn(env, cur, sigs, changed, name, t);
                        *changed = true;
                    }
                    Some(_) => {}
                    None => {
                        env.insert(name.clone(), t);
                    }
                }
            } else if let Some(c) = cur {
                if !env.contains_key(name) {
                    c.pending.borrow_mut().insert(name.clone());
                }
            }
        }
        Stmt::SetIndex { base, indices, value, .. } => {
            // A bare-variable base is known to be an array; otherwise infer it.
            // (A module's map a def stores into is the global, not a new local.)
            let global_map = |env: &HashMap<String, Ty>, name: &str| {
                env.get(name).is_none()
                    && cur.is_some_and(|c| !c.params.iter().any(|p| p == name) && !c.pending.borrow().contains(name))
                    && matches!(sigs.globals.get(name), Some(Ty::Map(..)))
            };
            if let Expr::Var { name, .. } = &**base {
                if !global_map(env, name) {
                    // `xs[i] = r` of a record makes `xs` an array of records
                    let elem = if env.contains_key(name) { None } else { infer_expr(value, env, cur, sigs, changed) };
                    // a def's result, or a field, may still turn out to be a
                    // record: "an array of numbers" is then a default held back
                    let may_be_record = matches!(value, Expr::Call { callee, .. } if sigs.arity.contains_key(callee.as_str())) || matches!(value, Expr::Field { .. });
                    let t = match elem {
                        Some(Ty::Record(n)) => Some(Ty::Recs(n)),
                        None if may_be_record && !sigs.index_default => None,
                        _ => Some(Ty::Arr(Elem::Num)),
                    };
                    if let Some(t) = t {
                        infer_learn(env, cur, sigs, changed, name, t);
                    }
                }
            } else {
                infer_expr(base, env, cur, sigs, changed);
            }
            let kts: Vec<Option<Ty>> = indices.iter().map(|idx| infer_expr(idx, env, cur, sigs, changed)).collect();
            let vt = infer_expr(value, env, cur, sigs, changed);
            // `m[k] = v` of a map: its kinds take k's and v's
            if let (Expr::Var { name, .. }, [kt]) = (&**base, kts.as_slice()) {
                infer_map_kinds(env, cur, sigs, changed, name, *kt, vt);
            }
        }
        Stmt::SetField { base, value, .. } => {
            infer_expr(base, env, cur, sigs, changed);
            infer_expr(value, env, cur, sigs, changed);
        }
        Stmt::Return { value, .. } => {
            let t = infer_expr(value, env, cur, sigs, changed);
            if let (Some(c), Some(t)) = (cur, t) {
                let slot = sigs.ret.get_mut(c.name).unwrap();
                if slot.is_none() || slot.is_some_and(|o| revises(o, t)) {
                    *slot = Some(t);
                    *changed = true;
                }
            }
        }
        Stmt::ExprStmt { expr, .. } => {
            // `x: T = e`: `x` is a T from here
            if let Expr::Call { callee, args, .. } = expr {
                if callee == crate::parser::DECLARE {
                    if let [Expr::Str { value: name, .. }, Expr::Str { value: text, span }] = args.as_slice() {
                        if let Ok(t) = resolve_type(&crate::parser::type_of_text(text, *span), &sigs.records) {
                            if env.get(name) != Some(&t) {
                                infer_learn(env, cur, sigs, changed, name, t);
                                env.insert(name.clone(), t);
                            }
                        }
                    }
                    return;
                }
            }
            infer_expr(expr, env, cur, sigs, changed);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            infer_expr(cond, env, cur, sigs, changed);
            for s in then_body.iter().chain(else_body) {
                infer_stmt(s, env, cur, sigs, changed);
            }
        }
        Stmt::While { cond, body, .. } => {
            infer_expr(cond, env, cur, sigs, changed);
            for s in body {
                infer_stmt(s, env, cur, sigs, changed);
            }
        }
        Stmt::For { var, bound, body, .. } => {
            // an Int bound counts in Ints, and so does a
            // quotient of two Ints (the count of `for i from a to b by s`)
            let mut bt = infer_expr(bound, env, cur, sigs, changed);
            if let Expr::Binary { op: BinOp::Div, lhs, rhs, .. } = bound {
                ints::INFER_TYS.with(|t| *t.borrow_mut() = env.clone());
                if ints::quotient_of_ints(lhs, rhs, sigs) {
                    bt = Some(Ty::Int);
                }
            }
            env.insert(var.clone(), if bt == Some(Ty::Int) { Ty::Int } else { Ty::Num });
            for s in body {
                infer_stmt(s, env, cur, sigs, changed);
            }
        }
        Stmt::Show { expr, .. } => {
            infer_expr(expr, env, cur, sigs, changed);
        }
        Stmt::Test { body, .. } => {
            for s in body {
                infer_stmt(s, env, cur, sigs, changed);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Narrate { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
    }
}

fn infer_expr(expr: &Expr, env: &mut HashMap<String, Ty>, cur: Option<&Cur>, sigs: &mut Sigs, changed: &mut bool) -> Option<Ty> {
    match expr {
        Expr::Num { .. } => Some(Ty::Num),
        Expr::Imag { .. } => Some(Ty::Cplx),
        Expr::Bool { .. } => Some(Ty::Bool),
        Expr::Str { .. } => Some(Ty::Str),
        Expr::Lambda { .. } => None,
        Expr::Var { name, .. } => {
            if let Some(t) = env.get(name) {
                Some(*t)
            } else if cur.map_or(false, |c| c.params.iter().any(|p| p == name) || c.pending.borrow().contains(name)) {
                // A parameter, or a local assigned above, whose type isn't
                // determined yet. It shadows any global of the same name, so
                // don't fall through to globals.
                None
            } else if let Some(t) = sigs.globals.get(name).copied() {
                Some(t)
            } else {
                // A bare top-level `def` name used as a value: its type is a
                // function of the def's arity (`g = sq`).
                sigs.fn_sig(name).map(Ty::Func)
            }
        }
        Expr::ArrLit { elems, .. } => {
            let mut any_cplx = false;
            let mut any_str = false;
            let mut fn_sig: Option<FnSig> = None;
            let mut all_fn = !elems.is_empty();
            // `[]`: its type comes from the appends that grow it (a list of
            // strings, or numbers), so it is a default held back to the second
            // phase, as indexing's is
            if elems.is_empty() && !sigs.index_default {
                return None;
            }
            let mut rec: Option<&'static str> = None;
            for e in elems {
                match infer_expr(e, env, cur, sigs, changed) {
                    Some(Ty::Record(n)) => {
                        rec.get_or_insert(n);
                        all_fn = false;
                    }
                    Some(Ty::Str) => {
                        any_str = true;
                        all_fn = false;
                    }
                    Some(Ty::Cplx) => {
                        any_cplx = true;
                        all_fn = false;
                    }
                    Some(Ty::Func(sig)) => {
                        if fn_sig.is_none() {
                            fn_sig = Some(sig);
                        }
                    }
                    _ => all_fn = false,
                }
            }
            if all_fn {
                // a registry of strategies; its element type carries the arity
                return Some(Ty::Arr(Elem::Func(fn_sig.unwrap_or(FnSig::plain(0)))));
            }
            // an array of records (the lowering checks they are all one type)
            if let Some(n) = rec {
                return Some(Ty::Recs(n));
            }
            if any_str {
                return Some(Ty::Arr(Elem::Str));
            }
            Some(Ty::Arr(if any_cplx { Elem::Cplx } else { Elem::Num }))
        }
        Expr::Index { base, indices, .. } => {
            // Indexing an unknown variable says "this is an array" but not of
            // WHAT. Guessing `Arr(Num)` immediately would beat a call site that
            // knows better (`f(spectrum)` with a complex array) whenever the
            // callee happens to be defined first. So the guess is held back
            // until a first fixpoint has let real evidence propagate, then
            // applied as the default it is. Order-independent.
            let bty = if let Expr::Var { name, .. } = &**base {
                // a module's map, read in a def, is not in the def's env
                let global_map = match env.get(name) {
                    None if cur.is_some() => infer_expr(base, env, cur, sigs, changed).filter(|t| matches!(t, Ty::Map(..))),
                    _ => None,
                };
                if global_map.is_some() {
                    global_map
                } else {
                    if env.get(name).is_none() && sigs.index_default {
                        infer_learn(env, cur, sigs, changed, name, Ty::Arr(Elem::Num));
                    }
                    env.get(name).copied()
                }
            } else {
                infer_expr(base, env, cur, sigs, changed)
            };
            let kts: Vec<Option<Ty>> = indices.iter().map(|idx| infer_expr(idx, env, cur, sigs, changed)).collect();
            // `m[k]` of a map: a string key makes its keys strings
            if let (Some(Ty::Map(..)), Expr::Var { name, .. }, [kt]) = (bty, &**base, kts.as_slice()) {
                infer_map_kinds(env, cur, sigs, changed, name, *kt, None);
            }
            match bty {
                Some(Ty::Arr(Elem::Cplx)) => Some(Ty::Cplx),
                Some(Ty::Arr(Elem::Num)) => Some(Ty::Num),
                Some(Ty::Arr(Elem::Int)) => Some(Ty::Int),
                Some(Ty::Arr(Elem::P(k))) => Some(if k.is_float() { Ty::F32 } else { Ty::Int }),
                // a string's character, a list's string
                Some(Ty::Str) | Some(Ty::Arr(Elem::Str)) => Some(Ty::Str),
                // An element of a registry is a function value. Calling it a
                // number (the default below) hid the signature from `apply`, so
                // `return apply(ms[k], v)` never learned it returns an array.
                Some(Ty::Arr(Elem::Func(sig))) => Some(Ty::Func(sig)),
                // an element of an array of records is a record
                Some(Ty::Recs(n)) => Some(Ty::Record(n)),
                // a map's value; not known while nothing is stored in it
                Some(Ty::Map(_, Mk::Any)) if !sigs.index_default => None,
                Some(Ty::Map(_, v)) => Some(mk_ty(v.settled())),
                _ if sigs.index_default => Some(Ty::Num),
                _ => None,
            }
        }
        Expr::Neg { operand, .. } => {
            let t = infer_expr(operand, env, cur, sigs, changed);
            match t {
                Some(Ty::Cplx) => Some(Ty::Cplx),
                Some(Ty::Int) => Some(Ty::Int),
                Some(Ty::F32) => Some(Ty::F32),
                Some(Ty::Arr(e)) => Some(Ty::Arr(e)),
                Some(t @ (Ty::Record(_) | Ty::Recs(_))) => Some(t),
                // an operand not typed yet: "a number" is a default held back
                None if !sigs.index_default => None,
                _ => Some(Ty::Num),
            }
        }
        Expr::Not { operand, .. } => {
            infer_expr(operand, env, cur, sigs, changed);
            Some(Ty::Bool)
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let lt = infer_expr(lhs, env, cur, sigs, changed);
            let rt = infer_expr(rhs, env, cur, sigs, changed);
            use BinOp::*;
            // An operand not typed yet (a parameter no call site has reached,
            // the result of a def whose return type has not arrived) could
            // still be anything, so the result is not known either -- unless the
            // other side already decides it (a complex array makes any
            // arithmetic a complex array). "A real array" and "a number" are
            // defaults held back until the first fixpoint has settled: answering
            // early stuck, and `r * factor` with `r` a complex array known only
            // later was typed as a real array.
            // a string or a list of strings on either side decides it
            if let Some(t) = strings::infer_binary(*op, lt, rt) {
                return Some(t);
            }
            // arithmetic over a record's arrays gives the record's type
            if let Some(t @ (Ty::Record(_) | Ty::Recs(_))) = lt.filter(|t| matches!(t, Ty::Record(_) | Ty::Recs(_))).or(rt.filter(|t| matches!(t, Ty::Record(_) | Ty::Recs(_)))) {
                return Some(t);
            }
            // a Float32 on either side
            if let Some(t) = f32s::infer_binary(*op, lt, rt) {
                return t;
            }
            // an Int on either side
            if let Some(t) = ints::infer_binary(*op, lt, rt, lhs, rhs, sigs) {
                return t;
            }
            let cplx_arr = lt == Some(Ty::Arr(Elem::Cplx)) || rt == Some(Ty::Arr(Elem::Cplx));
            let pending = (lt.is_none() || rt.is_none()) && !sigs.index_default;
            match op {
                Add | Sub | Mul | Div | Matmul if cplx_arr => Some(Ty::Arr(Elem::Cplx)),
                // `%` and `//` too: `b % 5` with b an Int not typed yet is an
                // Int once b is (B304)
                Lt | Le | Gt | Ge | EqEq | Ne | Add | Sub | Mul | Div | Mod | FloorDiv | Matmul if pending => None,
                // a comparison with a real array on either side is a 0/1 array
                Lt | Le | Gt | Ge | EqEq | Ne if lt == Some(Ty::Arr(Elem::Num)) || rt == Some(Ty::Arr(Elem::Num)) => Some(Ty::Arr(Elem::Num)),
                Lt | Le | Gt | Ge | EqEq | Ne | And | Or => Some(Ty::Bool),
                Mod | FloorDiv => Some(Ty::Num),
                Matmul => Some(Ty::Arr(if lt == Some(Ty::Arr(Elem::Cplx)) || rt == Some(Ty::Arr(Elem::Cplx)) { Elem::Cplx } else { Elem::Num })),
                Add | Sub | Mul | Div => {
                    // a complex array on either side makes the result one
                    if lt == Some(Ty::Arr(Elem::Cplx)) || rt == Some(Ty::Arr(Elem::Cplx)) {
                        Some(Ty::Arr(Elem::Cplx))
                    } else if lt == Some(Ty::Arr(Elem::Num)) || rt == Some(Ty::Arr(Elem::Num)) {
                        Some(Ty::Arr(Elem::Num))
                    } else if lt == Some(Ty::Cplx) || rt == Some(Ty::Cplx) {
                        Some(Ty::Cplx)
                    } else if lt == Some(Ty::Num) && rt == Some(Ty::Num) {
                        Some(Ty::Num)
                    } else {
                        None
                    }
                }
            }
        }
        Expr::Call { callee, args, .. } => {
            let arg_tys: Vec<Option<Ty>> = args.iter().map(|a| infer_expr(a, env, cur, sigs, changed)).collect();
            // `sorted`, `unique`, `del` of a real array: one (crate::stdlib)
            if crate::stdlib::number_form(callee).is_some() && !sigs.arity.contains_key(callee.as_str()) {
                let empty = matches!(args.first(), Some(Expr::ArrLit { elems, .. }) if elems.is_empty());
                match arg_tys.first().copied().flatten() {
                    Some(t @ Ty::Arr(Elem::Num)) if !empty => return Some(t),
                    Some(t @ Ty::Arr(Elem::Str)) if callee == crate::stdlib::DEL => return Some(t),
                    _ => {}
                }
            }
            // Forward: known argument types refine the callee's parameter types.
            if let Some(slots) = sigs.params.get_mut(callee) {
                for (j, at) in arg_tys.iter().enumerate() {
                    if let (Some(t), Some(slot)) = (at, slots.get_mut(j)) {
                        // a parameter with no type that is passed Ints and
                        // numbers is a number; the Ints are converted exactly
                        // at their calls
                        let mixed = *slot == Some(Ty::Int) && *t == Ty::Num && sigs.untyped.contains(&(callee.clone(), j));
                        if slot.is_none() || slot.is_some_and(|o| revises(o, *t)) || mixed {
                            *slot = Some(*t);
                            *changed = true;
                        }
                    }
                }
            }
            // Backward: a known callee parameter type teaches an unknown argument
            // variable (e.g. `x` is an array because it is passed to `dot`).
            for (j, arg) in args.iter().enumerate() {
                // a map passed to a def that stores strings in it holds strings
                if let (Expr::Var { name, .. }, Some(Ty::Map(..))) = (arg, arg_tys.get(j).copied().flatten()) {
                    if let Some(Some(Ty::Map(pk, pv))) = sigs.params.get(callee.as_str()).and_then(|v| v.get(j)).copied() {
                        infer_map_grow(env, cur, sigs, changed, name, pk, pv);
                    }
                }
                if let Expr::Var { name, .. } = arg {
                    // a module's map a def reads (`has(m, k)` in a def) is the
                    // global, as for `m[k] = v`: learning it here would make a
                    // local that hides the global, and the key the call
                    // teaches would reach the local only
                    let global_map = env.get(name).is_none()
                        && cur.is_some_and(|c| !c.params.iter().any(|p| p == name) && !c.pending.borrow().contains(name))
                        && matches!(sigs.globals.get(name.as_str()), Some(Ty::Map(..)));
                    if env.get(name).is_none() && !global_map {
                        // `transpose` accepts either element type, so "its argument is
                        // a real array" is a default held back like the indexing one.
                        // Likewise min / max, which take numbers OR arrays.
                        // ... and pow's base, and every math function that maps over an
                        // array: `cos(theta)` does not make `theta` a number. (It used to,
                        // and `def phase(theta) { return complex(cos(theta), sin(theta)) }`
                        // then refused the array it was written for.)
                        let overloaded = matches!(
                            callee.as_str(),
                            "transpose" | "min" | "max" | "exp" | "sqrt" | "sin" | "cos" | "tan" | "log" | "ln" | "abs" | "fabs" | "floor" | "ceil"
                                | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "erf" | "gamma" | "round"
                        ) || (callee == "pow" && j == 0)
                            // `print` takes a number, an array or a complex value:
                            // `q = f(); print(q)` must not make `q` a number
                            // before `f`'s return type has arrived
                            || callee == "print"
                            // `assert_true` takes a boolean or a number: a test's
                            // `assert_true(ok)` must not make `ok` a number before
                            // the comparison that assigns it is typed (bug B110)
                            || callee == "assert_true"
                            // a slice, a copy or a join keeps its array's element type
                            || matches!(callee.as_str(), "copy" | "concat" | "hcat" | "vcat")
                            // these take a record too (`lower::records`)
                            || matches!(callee.as_str(), "sum" | "zeros_like")
                            || ((callee == SLICE_GET || callee == SLICE_SET) && j == 0);
                        // a slice write takes a number or an array
                        if callee == SLICE_SET && j == 2 {
                            continue;
                        }
                        let overloaded = overloaded && !sigs.arity.contains_key(callee.as_str());
                        if overloaded && !sigs.index_default {
                            continue;
                        }
                        // `a = append(a, x)`: `a` holds what `x` is (a list
                        // of strings for a string), known once `x` is
                        if matches!(callee.as_str(), "append" | "extend") && j == 0 && !sigs.arity.contains_key(callee.as_str()) {
                            let t = match arg_tys.get(1).copied().flatten() {
                                // an Int array stays one (its Ints, or an error at the lowering)
                                _ if arg_tys.first().copied().flatten() == Some(Ty::Arr(Elem::Int)) => Some(Ty::Arr(Elem::Int)),
                                Some(Ty::Str) | Some(Ty::Arr(Elem::Str)) => Some(Ty::Arr(Elem::Str)),
                                // `blocks = append(blocks, b)` of a record
                                Some(Ty::Record(n)) if callee == "append" => Some(Ty::Recs(n)),
                                Some(_) => Some(Ty::Arr(Elem::Num)),
                                None if sigs.index_default => Some(Ty::Arr(Elem::Num)),
                                None => None,
                            };
                            if let Some(t) = t {
                                infer_learn(env, cur, sigs, changed, name, t);
                            }
                            continue;
                        }
                        // a map builtin's key and default: of the map's kinds,
                        // numbers once nothing else has been seen
                        if crate::interp::maps::NAMES.contains(&callee.as_str()) && j > 0 && !sigs.arity.contains_key(callee.as_str()) {
                            let kind = match arg_tys.first().copied().flatten() {
                                Some(Ty::Map(k, v)) => if j == 1 { k } else { v },
                                _ => Mk::Any,
                            };
                            if kind != Mk::Any || sigs.index_default {
                                infer_learn(env, cur, sigs, changed, name, mk_ty(kind.settled()));
                            }
                            continue;
                        }
                        if let Some(t) = callee_param_ty(callee, j, sigs) {
                            infer_learn(env, cur, sigs, changed, name, t);
                        }
                    }
                }
            }
            // a program's own def wins over a builtin of the same name
            if sigs.arity.contains_key(callee.as_str()) {
                return sigs.ret.get(callee).copied().flatten();
            }
            // over a record's arrays
            if let Some(t @ (Ty::Record(_) | Ty::Recs(_))) = arg_tys.first().copied().flatten() {
                if args.len() == 1 && (callee == "zeros_like" || REC_MAP_FNS.contains(&callee.as_str())) {
                    return Some(t);
                }
                if args.len() == 1 && callee == "sum" {
                    return Some(Ty::Num);
                }
            }
            // `zeros_like(x)` of an `x` not typed yet may be a record's
            if callee == "zeros_like" && args.len() == 1 && arg_tys[0].is_none() && !sigs.index_default {
                return None;
            }
            if callee == "for_leaves" {
                // the def takes a real array for each record, then the rest
                if let Some(Expr::Var { name: f, .. }) = args.first() {
                    let k = arg_tys[1..].iter().take_while(|t| matches!(t, Some(Ty::Record(_) | Ty::Recs(_)))).count();
                    if let Some(slots) = sigs.params.get_mut(f.as_str()) {
                        for (j, slot) in slots.iter_mut().enumerate() {
                            let want = if j < k { Some(Ty::Arr(Elem::Num)) } else { arg_tys.get(1 + j).copied().flatten() };
                            if let (None, Some(t)) = (*slot, want) {
                                *slot = Some(t);
                                *changed = true;
                            }
                        }
                    }
                }
                return Some(Ty::Num);
            }
            // `k in m` of a map teaches the map its key kind, as `has(m, k)`
            if callee == crate::interp::strings::IN {
                if let (Some(Expr::Var { name, .. }), Some(Some(Ty::Map(..)))) = (args.get(1), arg_tys.get(1)) {
                    infer_map_kinds(env, cur, sigs, changed, name, arg_tys.first().copied().flatten(), None);
                }
            }
            // enums and optionals, and the Int forms of the builtins
            if let Some(t) = enums::infer_call(callee, args, &arg_tys, sigs) {
                return t;
            }
            ints::INFER_TYS.with(|t| *t.borrow_mut() = env.clone());
            if let Some(t) = f32s::infer_call(callee, &arg_tys).filter(|_| !sigs.arity.contains_key(callee.as_str())) {
                return t;
            }
            if let Some(t) = ints::infer_call(callee, args, &arg_tys, sigs) {
                return t;
            }
            // strings: the registry's builtins, and those that take strings
            // among other values (str, fmt, len of a string, a slice of one, ...)
            if let Some(t) = strings::infer_call(callee, &arg_tys, sigs) {
                return t;
            }
            // the map builtins: a key or a default teaches
            // the map its kinds; a value, a key list or a value list has them
            if crate::interp::maps::NAMES.contains(&callee.as_str()) {
                if let (Some(Expr::Var { name, .. }), "has" | "get" | "remove") = (args.first(), callee.as_str()) {
                    infer_map_kinds(env, cur, sigs, changed, name, arg_tys.get(1).copied().flatten(), arg_tys.get(2).copied().flatten());
                }
                let (k, v) = match arg_tys.first().copied().flatten() {
                    Some(Ty::Map(k, v)) => (k, v),
                    _ => (Mk::Any, Mk::Any),
                };
                let open = |m: Mk| m == Mk::Any && !sigs.index_default;
                let elem = |m: Mk| if m.settled() == Mk::Str { Elem::Str } else { Elem::Num };
                return match callee.as_str() {
                    "map" => Some(Ty::Map(Mk::Any, Mk::Any)),
                    "has" => Some(Ty::Bool),
                    "clear" => Some(Ty::Num),
                    "keys" if open(k) => None,
                    "keys" => Some(Ty::Arr(elem(k))),
                    _ if open(v) => None,
                    "values" => Some(Ty::Arr(elem(v))),
                    _ => Some(mk_ty(v.settled())),
                };
            }
            // the array vocabulary (interp/arrays.rs)
            if let Some(sig) = crate::interp::arrays::sig(callee, args.len()) {
                return array_call_ty(sig, args, &arg_tys, sigs);
            }
            if callee == "apply" {
                if !sigs.fn_final {
                    return None;
                }
                return match arg_tys.first().copied().flatten() {
                    Some(Ty::Func(sig)) => Some(sig.ret_ty()),
                    _ => None,
                };
            }
            match callee.as_str() {
                "print" | "text" | "simpson" | "trapezoid" | "integrate" | "find_root" | "assert_near" | "assert_true" | "assert_gt" | "assert_lt" | "assert_allclose" | "srand" | "rand" | "sort" | "eig_sym" | "spectrum_sym" | "lin_solve" | "lin_solve_multi" | "cholesky" | "eig_general" | "eig_near" | "len" | "numel" | "deriv" | "fill" | "scatter" => Some(Ty::Num),
                "re" | "im" | "abs" | "fabs" | "conj"
                    if arg_tys.first().copied().flatten() == Some(Ty::Arr(Elem::Cplx)) =>
                {
                    Some(Ty::Arr(if callee == "conj" { Elem::Cplx } else { Elem::Num }))
                }
                // `complex(re, im)` packs two real arrays, or makes one scalar. Either
                // argument being an array settles it.
                "complex" if arg_tys.iter().any(|t| *t == Some(Ty::Arr(Elem::Num))) => Some(Ty::Arr(Elem::Cplx)),
                // These are scalar OR array functions, and which one depends on an
                // argument that may not be typed yet (`complex(u * u, ZERO)` with `u` a
                // parameter nobody has called yet). Answering "scalar" now would stick
                // -- a learned type is never overridden -- so an untyped argument means
                // "no answer yet", and "scalar" is a default held back to the last phase.
                "complex" | "conj" | "re" | "im" | "exp" | "sqrt" | "sin" | "cos" | "tan" | "log" | "abs" | "fabs" | "floor" | "ceil"
                | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "erf" | "gamma" | "round"
                    if arg_tys.iter().any(|t| t.is_none()) && !sigs.index_default =>
                {
                    None
                }
                "complex" | "conj" => Some(if callee == "conj" && arg_tys.first().copied().flatten() == Some(Ty::Num) { Ty::Num } else { Ty::Cplx }),
                "re" | "im" => Some(Ty::Num),
                // 1-argument math: elementwise over an array argument, scalar otherwise
                "exp" | "sqrt" | "sin" | "cos" | "tan" | "log" | "abs" | "fabs" | "floor" | "ceil"
        | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "erf" | "gamma" | "round" => {
                    if args.len() == 1 && arg_tys[0] == Some(Ty::Arr(Elem::Num)) {
                        Some(Ty::Arr(Elem::Num))
                    } else if args.len() == 1
                        && arg_tys[0] == Some(Ty::Cplx)
                        && matches!(callee.as_str(), "exp" | "sqrt" | "log" | "sin" | "cos" | "tan")
                    {
                        Some(Ty::Cplx)
                    } else {
                        Some(Ty::Num)
                    }
                }
                "min" | "max" if args.len() == 1 => Some(Ty::Num),
                // min / max of two, and pow, are scalar or elementwise by their
                // arguments; an argument not typed yet (a parameter no call site
                // has reached) means no answer yet, or `def gf(a, b) { return
                // max(a, b) }` is fixed as returning a number before `gf(u, v)`
                // with arrays is seen
                "min" | "max" | "pow" if arg_tys.iter().any(|t| t.is_none()) && !sigs.index_default => None,
                "min" | "max" => {
                    if arg_tys.iter().any(|t| *t == Some(Ty::Arr(Elem::Num))) { Some(Ty::Arr(Elem::Num)) } else { Some(Ty::Num) }
                }
                "pow" => {
                    if arg_tys.first().copied().flatten() == Some(Ty::Arr(Elem::Num)) { Some(Ty::Arr(Elem::Num)) } else { Some(Ty::Num) }
                }
                "atan2" | "fma" | "mod" | "band" | "bor" | "bxor" | "bnot" | "shl" | "shr" | "sar" | "u32" | "i32" => Some(Ty::Num),
                // these keep their first argument's element type
                "transpose" | "slab" | "take" => match arg_tys.first().copied().flatten() {
                    Some(Ty::Arr(e)) => Some(Ty::Arr(e)),
                    // an argument not typed yet may be a complex array
                    _ if sigs.index_default => Some(Ty::Arr(Elem::Num)),
                    _ => None,
                },
                "setslab" | "put" => Some(Ty::Num),
                // a slice, a copy or a join keeps its first array's element type
                c if c == SLICE_GET || matches!(c, "copy" | "concat" | "hcat" | "vcat") => match arg_tys.first().copied().flatten() {
                    Some(Ty::Arr(e)) => Some(Ty::Arr(e)),
                    Some(t @ Ty::Map(..)) if c == "copy" => Some(t),
                    _ if sigs.index_default => Some(Ty::Arr(Elem::Num)),
                    _ => None,
                },
                c if c == SLICE_SET => Some(Ty::Num),
                "append" => match (arg_tys.first().copied().flatten(), arg_tys.get(1).copied().flatten()) {
                    (Some(t @ Ty::Recs(_)), _) => Some(t),
                    (Some(t @ Ty::Arr(Elem::Int)), _) => Some(t),
                    (_, Some(Ty::Record(n))) => Some(Ty::Recs(n)),
                    _ => Some(Ty::Arr(Elem::Num)),
                },
                "extend" | "__ad_head" => Some(Ty::Arr(Elem::Num)),
                c if c == crate::growable::SEAL => arg_tys.first().copied().flatten(),
                "__ad_tail" => arg_tys.get(2).copied().flatten(),
                "__ad_cat" => Some(Ty::Arr(Elem::Num)),
                "__ad_extent" => Some(Ty::Num),
                "zeros" | "linspace" | "histogram" | "shape" => Some(Ty::Arr(Elem::Num)),
                "release_mode" => Some(Ty::Bool),
                "clock" => Some(Ty::Num),
                c if exporting() && export_builtin_ty(c).is_some() => export_builtin_ty(c),
                "tridiag_solve" | "roll" => Some(Ty::Arr(Elem::Num)),
                "where" if !sigs.arity.contains_key("where") => Some(Ty::Arr(Elem::Num)),
                // the autodiff helpers `crate::grad` emits (v4)
                "__ad_pop" | "__ad_zero" if !sigs.arity.contains_key(callee.as_str()) => arg_tys.first().copied().flatten(),
                "__ad_unb" if !sigs.arity.contains_key(callee.as_str()) => arg_tys.get(1).copied().flatten(),
                "__ad_cmp" if !sigs.arity.contains_key(callee.as_str()) => {
                    if arg_tys.iter().any(|t| *t == Some(Ty::Arr(Elem::Num))) {
                        Some(Ty::Arr(Elem::Num))
                    } else if arg_tys.iter().all(|t| *t == Some(Ty::Num)) {
                        Some(Ty::Num)
                    } else {
                        None
                    }
                }
                "__ad_push" | "__ad_restore" if !sigs.arity.contains_key(callee.as_str()) => Some(Ty::Num),
                "__ad_mm_a" | "__ad_mm_b" | "__ad_take" | "__ad_keep" | "__ad_like" | "__ad_prod" | "__ad_prodax" | "__ad_takeax" if !sigs.arity.contains_key(callee.as_str()) => {
                    Some(Ty::Arr(Elem::Num))
                }
                "all" | "any" if !sigs.arity.contains_key(callee.as_str()) => Some(Ty::Bool),
                "czeros" => Some(Ty::Arr(Elem::Cplx)),
                "fft" | "ifft" | "fft3" | "ifft3" | "eig_herm" | "sum" => Some(Ty::Num),
                _ => sigs.ret.get(callee).copied().flatten(),
            }
        }
        Expr::RecordLit { type_name, fields, .. } => {
            let tuple = crate::tuples::tuple_def(type_name).is_some();
            for (fname, _, value) in fields {
                let mut t = infer_expr(value, env, cur, sigs, changed);
                // `return x, int(x)`: a slot that holds what `int` gives is an
                // Int, as `n = int(x)` makes one (B305), unless another return
                // puts a number there
                let adapting = ints::adapts(value, sigs);
                let int_slot = tuple && t == Some(Ty::Num) && adapting && ints::has_int_call(value);
                if int_slot {
                    t = Some(Ty::Int);
                }
                // a field whose type is inferred learns it from the values
                // it is built from (`return a, b`)
                if let (Some(t), Some(info)) = (t, sigs.records.get_mut(type_name)) {
                    if let Some(k) = info.fields.iter().position(|(f, _)| f == fname) {
                        if let Some(found) = info.open[k] {
                            let to_num = tuple && found && info.fields[k].1 == Ty::Int && t == Ty::Num && !adapting;
                            if int_slot && found {
                                // only the first return decides an int() slot
                            } else if !found || to_num || revises(info.fields[k].1, t) {
                                if !found || info.fields[k].1 != t {
                                    *changed = true;
                                }
                                info.fields[k].1 = t;
                                info.open[k] = Some(true);
                            }
                        }
                    }
                }
            }
            sigs.record_ty(type_name)
        }
        Expr::Field { base, field, .. } => {
            let base_ty = infer_expr(base, env, cur, sigs, changed);
            if let Some(Ty::Record(name)) = base_ty {
                let info = sigs.records.get(name)?;
                let k = info.fields.iter().position(|(f, _)| f == field)?;
                if info.open[k] == Some(false) {
                    return None;
                }
                Some(info.fields[k].1)
            } else {
                None
            }
        }
    }
}

/// The tape and adjoint helpers of reverse-mode autodiff (`crate::grad`),
/// resolved by their arguments' types to the C runtime's `viv_ad_*`
/// (fragment "ad"). Only generated gradient defs call them.
/// The callees a slice is written as (`crate::interp::slices`).
const SLICE_GET: &str = crate::interp::slices::GET;
const SLICE_SET: &str = crate::interp::slices::SET;

/// `viv__slice_get(a, spec)`, `viv__slice_set(a, spec, v)`, `copy(a)` and
/// `concat` / `vcat` / `hcat` of one or more arrays (folded left, two at a
/// time, as the interpreter folds them).
fn lower_slice_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let shown = match callee {
        c if c == SLICE_GET || c == SLICE_SET => "a slice",
        c => c,
    };
    let array = |prog: &mut IrProgram<Viv>, e: &Expr| -> Result<(NodeId, Elem), Diagnostic> {
        let (id, ty) = lower_expr(prog, e, scope, sigs)?;
        match ty {
            Ty::Arr(el @ (Elem::Num | Elem::Cplx)) => Ok((id, el)),
            _ if shown == "a slice" => Err(Diagnostic::error("only an array can be sliced").with_primary(e.span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))),
            _ => Err(Diagnostic::error(format!("`{shown}` takes arrays")).with_primary(e.span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))),
        }
    };
    let width = |prog: &mut IrProgram<Viv>, el: Elem| prog.add(Viv::Num(if el == Elem::Cplx { 2.0 } else { 1.0 }), span, vec![]);
    let call = |prog: &mut IrProgram<Viv>, name: &str, kids: Vec<NodeId>| prog.add(Viv::Call { callee: name.to_string() }, span, kids);
    match callee {
        // a slice that `crate::fuse::substitute` reads in place where it is
        // used: here, only its bounds are checked
        c if c == crate::fuse::CHECK => {
            let (aid, _) = array(prog, &args[0])?;
            let (sid, sty) = lower_expr(prog, &args[1], scope, sigs)?;
            expect_arr(sty, args[1].span())?;
            Ok((call(prog, "viv_slicecheck", vec![aid, sid]), Ty::Num))
        }
        c if c == SLICE_GET || c == SLICE_SET || c == crate::fuse::VIEW => {
            if args.len() != if c == SLICE_SET { 3 } else { 2 } {
                return Err(Diagnostic::error(format!("`{c}` is written by the parser for a slice")).with_primary(span, "wrong number of arguments"));
            }
            // an Int array's slots are 8 bytes, as a real array's are: its
            // slices move them bit for bit (B343)
            let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
            if aty == Ty::Arr(Elem::Int) {
                let (sid, sty) = lower_expr(prog, &args[1], scope, sigs)?;
                expect_arr(sty, args[1].span())?;
                if c == SLICE_GET {
                    return Ok((call(prog, "viv_igetslice", vec![aid, sid]), aty));
                }
                let (vid, vty) = lower_expr(prog, &args[2], scope, sigs)?;
                if vty != aty {
                    return Err(Diagnostic::error("a slice of an Int array takes an Int array").with_primary(args[2].span(), format!("this is {}", crate::lower::a_or_an(&vty.name()))));
                }
                return Ok((call(prog, "viv_isetslice", vec![aid, sid, vid]), Ty::Num));
            }
            let el = match aty {
                Ty::Arr(el @ (Elem::Num | Elem::Cplx)) => el,
                _ => return Err(Diagnostic::error("only an array can be sliced").with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&aty.name())))),
            };
            let (sid, sty) = lower_expr(prog, &args[1], scope, sigs)?;
            expect_arr(sty, args[1].span())?;
            if c == SLICE_GET || c == crate::fuse::VIEW {
                let w = width(prog, el);
                let name = if c == SLICE_GET { "viv_getslice" } else { crate::fuse::VIEW_C };
                return Ok((call(prog, name, vec![aid, sid, w]), Ty::Arr(el)));
            }
            let (vid, vty) = lower_expr(prog, &args[2], scope, sigs)?;
            let name = match (el, vty) {
                (_, t) if t == Ty::Arr(el) => "viv_setslice",
                (Elem::Num, Ty::Num) => "viv_setslice_n",
                (Elem::Cplx, Ty::Cplx) => "viv_setslice_c",
                (Elem::Num, t) => {
                    return Err(Diagnostic::error("a slice of a real array takes a number or a real array")
                        .with_primary(args[2].span(), format!("this is {}", crate::lower::a_or_an(&t.name()))))
                }
                (_, t) => {
                    return Err(Diagnostic::error("a slice of a complex array takes a complex number or a complex array")
                        .with_primary(args[2].span(), format!("this is {}", crate::lower::a_or_an(&t.name()))))
                }
            };
            let mut kids = vec![aid, sid, vid];
            if name == "viv_setslice" {
                kids.push(width(prog, el));
            }
            Ok((call(prog, name, kids), Ty::Num))
        }
        "copy" => {
            if args.len() != 1 {
                return Err(Diagnostic::error("`copy` takes 1 array").with_primary(span, "wrong number of arguments"));
            }
            // a map is copied by the maps runtime
            let (aid, ty) = lower_expr(prog, &args[0], scope, sigs)?;
            if let Ty::Map(k, v) = ty {
                return Ok((call(prog, &crate::interp::maps::callee("copy", k, v), vec![aid]), ty));
            }
            let el = match ty {
                Ty::Arr(el @ (Elem::Num | Elem::Cplx)) => el,
                Ty::Recs(_) | Ty::Record(_) => {
                    return Err(Diagnostic::error("`copy` takes an array of numbers or a map").with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))
                        .with_note("records and arrays of records are values, so `b = a` is already a copy"))
                }
                t if strings::is_text(t) => {
                    return Err(Diagnostic::error("`copy` takes an array or a map").with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))
                        .with_note("a string or a list of strings is never changed in place, so `b = a` is already a copy"))
                }
                _ => return Err(Diagnostic::error("`copy` takes an array or a map").with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))),
            };
            let w = width(prog, el);
            Ok((call(prog, "viv_copy", vec![aid, w]), Ty::Arr(el)))
        }
        _ => {
            if args.is_empty() {
                return Err(Diagnostic::error(format!("`{callee}` takes one or more arrays")).with_primary(span, "no arguments"));
            }
            let mode = match callee {
                "concat" => 0.0,
                "vcat" => 1.0,
                _ => 2.0,
            };
            let (mut acc, el) = array(prog, &args[0])?;
            if args.len() == 1 {
                let m = prog.add(Viv::Num(mode), span, vec![]);
                let w = width(prog, el);
                acc = call(prog, "viv_cat1", vec![m, acc, w]);
            }
            for e in &args[1..] {
                let (bid, bel) = array(prog, e)?;
                if bel != el {
                    return Err(Diagnostic::error(format!("`{callee}`: the arrays must all be real or all complex"))
                        .with_primary(e.span(), format!("this is {}", crate::lower::a_or_an(&Ty::Arr(bel).name()))));
                }
                let m = prog.add(Viv::Num(mode), span, vec![]);
                let w = width(prog, el);
                acc = call(prog, "viv_cat", vec![m, acc, bid, w]);
            }
            Ok((acc, Ty::Arr(el)))
        }
    }
}

/// `append(a, x)` and `extend(a, b)` (interp/growable.rs), with the third
/// argument `crate::growable` adds to a rebinding it proved to be the only
/// holder of `a` (1: may grow in place; absent or 0: copies), and the pass's
/// `viv__seal(a)`.
fn lower_grow_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let call = |prog: &mut IrProgram<Viv>, name: &str, kids: Vec<NodeId>| prog.add(Viv::Call { callee: name.to_string() }, span, kids);
    if callee == crate::growable::SEAL {
        // only a real array has a handle to seal; anything else passes through
        // (and an append of it is refused below)
        let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
        return Ok(match ty {
            Ty::Arr(Elem::Num) => (call(prog, "viv_seal", vec![id]), ty),
            Ty::Arr(Elem::Int) => (call(prog, "viv_iseal", vec![id]), ty),
            _ => (id, ty),
        });
    }
    if args.len() != 2 && args.len() != 3 {
        return Err(Diagnostic::error(format!("`{callee}` takes 2 arguments")).with_primary(span, "wrong number of arguments"));
    }
    let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
    // `append(xs, r)` of an array of records: a new array (the flag the
    // growable pass adds is not needed: a record array is never grown in
    // place)
    if let Ty::Recs(rec) = aty {
        if callee != "append" {
            return Err(Diagnostic::error("`extend` joins two real vectors").with_primary(args[0].span(), "this is an array of records"));
        }
        let (xid, xty) = lower_expr(prog, &args[1], scope, sigs)?;
        if xty != Ty::Record(rec) {
            return Err(Diagnostic::error(format!("this array holds {rec} records, and this is {}", crate::lower::a_or_an(&xty.name()))).with_primary(args[1].span(), "wrong element type"));
        }
        let xid = owned(prog, xid, xty, &args[1], sigs);
        return Ok((prog.add(Viv::RecAppend { rec: rec.to_string() }, span, vec![aid, xid]), aty));
    }
    let own = match args.get(2) {
        Some(Expr::Num { value, .. }) => *value,
        _ => 0.0,
    };
    // an Int array grows by Ints: an expression that
    // adapts (`int(x)`, a literal) is computed as one
    if aty == Ty::Arr(Elem::Int) {
        if callee == "extend" {
            // two Int arrays (B343)
            let (xid, xty) = lower_expr(prog, &args[1], scope, sigs)?;
            if xty != aty {
                return Err(Diagnostic::error("`extend` of an Int array joins another Int array").with_primary(args[1].span(), format!("this is {}", crate::lower::a_or_an(&xty.name()))));
            }
            let oid = prog.add(Viv::Num(own), span, vec![]);
            return Ok((call(prog, "viv_iextend", vec![aid, xid, oid]), aty));
        }
        let (xid, xty) = match ints::lower_for(prog, &args[1], Some(Ty::Int), scope, sigs) {
            Some(r) => r?,
            None => lower_expr(prog, &args[1], scope, sigs)?,
        };
        if xty != Ty::Int {
            return Err(Diagnostic::error(format!("an Int array holds Ints, and this is {}", if xty == Ty::Num { "a number".to_string() } else { format!("a {}", xty.name()) }))
                .with_primary(args[1].span(), "not an Int")
                .with_note("a number becomes an Int only where you say how: `int(x)` (toward zero), `round(x)`, `floor(x)` or `ceil(x)`"));
        }
        let oid = prog.add(Viv::Num(own), span, vec![]);
        return Ok((call(prog, "viv_iappend", vec![aid, xid, oid]), aty));
    }
    if aty != Ty::Arr(Elem::Num) {
        return Err(Diagnostic::error(format!("`{callee}` takes a real array first")).with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&aty.name()))));
    }
    let (xid, xty) = lower_expr(prog, &args[1], scope, sigs)?;
    // an Int on a real array: the number it is, exactly (or the program stops)
    let (xid, xty) = ints::to_num(prog, xid, xty, args[1].span());
    let name = match (callee, xty) {
        ("append", Ty::Num) => "viv_append_n",
        ("append", Ty::Arr(Elem::Num)) => "viv_append_r",
        ("extend", Ty::Arr(Elem::Num)) => "viv_extend",
        ("extend", Ty::Num) => {
            return Err(Diagnostic::error("`extend` joins two vectors; add one number with `append`").with_primary(args[1].span(), "this is a number"))
        }
        ("extend", t) => return Err(Diagnostic::error("`extend` joins two real vectors").with_primary(args[1].span(), format!("this is {}", crate::lower::a_or_an(&t.name())))),
        (_, t) => return Err(Diagnostic::error("`append` adds a number, or a row (a real vector)").with_primary(args[1].span(), format!("this is {}", crate::lower::a_or_an(&t.name())))),
    };
    let oid = prog.add(Viv::Num(own), span, vec![]);
    Ok((call(prog, name, vec![aid, xid, oid]), Ty::Arr(Elem::Num)))
}

/// The map builtins (interp/maps.rs): `map()`, `has`, `get`,
/// `remove`, `keys`, `values`, `clear`. `len` and `copy` of a map are the
/// arrays' builtins, extended where they are lowered. A map of strings calls
/// the kinded runtime (`viv_map_get_sn`, maps::kinded_fns).
fn lower_map_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let call = |prog: &mut IrProgram<Viv>, name: String, kids: Vec<NodeId>| prog.add(Viv::Call { callee: name }, span, kids);
    let (usage, fits) = match callee {
        "map" => ("map()", args.is_empty()),
        "has" => ("has(m, k)", args.len() == 2),
        "get" => ("get(m, k, default)", args.len() == 3),
        "remove" => ("remove(m, k) or remove(m, k, default)", args.len() == 2 || args.len() == 3),
        "keys" => ("keys(m)", args.len() == 1),
        "values" => ("values(m)", args.len() == 1),
        _ => ("clear(m)", args.len() == 1),
    };
    if !fits {
        return Err(Diagnostic::error(format!("`{callee}` is written {usage}")).with_primary(span, "wrong number of arguments"));
    }
    if callee == "map" {
        return Ok((call(prog, "viv_map_new".into(), vec![]), NUM_MAP));
    }
    let (mid, mty) = lower_expr(prog, &args[0], scope, sigs)?;
    let Ty::Map(k, v) = settle(mty) else {
        return Err(Diagnostic::error(format!("`{callee}` takes a map first")).with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&mty.name()))));
    };
    let mut ids = vec![mid];
    for (j, a) in args.iter().enumerate().skip(1) {
        let (id, ty) = strings::lower_expected(prog, a, Some(mk_ty(if j == 1 { k } else { v })), scope, sigs)?;
        let (id, ty) = ints::to_num(prog, id, ty, a.span());
        if j == 1 {
            map_kind(ty, k, a.span(), "key", mty)?;
        } else {
            map_kind(ty, v, a.span(), "default", mty)?;
        }
        ids.push(id);
    }
    let op = match callee {
        "get" => "getd",
        "remove" if ids.len() == 3 => "removed",
        c => c,
    };
    let elem = |m: Mk| if m == Mk::Str { Elem::Str } else { Elem::Num };
    let ty = match callee {
        "has" => Ty::Bool,
        "get" | "remove" => mk_ty(v),
        "keys" => Ty::Arr(elem(k)),
        "values" => Ty::Arr(elem(v)),
        _ => Ty::Num,
    };
    Ok((call(prog, crate::interp::maps::callee(op, k, v), ids), ty))
}

/// The empty map a variable of type `want` holds: `map()` where a map of
/// strings is wanted makes one.
pub(super) fn lower_new_map(prog: &mut IrProgram<Viv>, want: Ty, span: Span) -> (NodeId, Ty) {
    let t = settle(want);
    let Ty::Map(k, v) = t else { unreachable!("a map type") };
    (prog.add(Viv::Call { callee: crate::interp::maps::callee("new", k, v) }, span, vec![]), t)
}

/// A `parallel for` body that uses a map: refused (section
/// 7), since its iterations run at once and an insert can rebuild the storage
/// under another iteration. A map named in the body, or made or read there by
/// a map builtin, is found; one a called def reaches as a global is not.
fn parallel_maps(body: &[Stmt], scope: &Scope<'_>, sigs: &Sigs) -> Result<(), Diagnostic> {
    let mut at = None;
    crate::defun::walk_block(body, &mut |e: &Expr| {
        let hit = match e {
            Expr::Var { name, .. } => matches!(scope.get(name), Some(Ty::Map(..))),
            Expr::Call { callee, .. } => crate::interp::maps::NAMES.contains(&callee.as_str()) && !sigs.arity.contains_key(callee.as_str()),
            _ => false,
        };
        if hit && at.is_none() {
            at = Some(e.span());
        }
    });
    match at {
        Some(span) => Err(Diagnostic::error("a `parallel for` cannot use a map")
            .with_primary(span, "a map")
            .with_note("its iterations run at once, and an insert can move the map's storage under another; fill the map in a plain `for`")),
        None => Ok(()),
    }
}

/// A map's key or value (or a `get` default) of type `ty`, where the map's
/// kind for it is `want`: a number or a
/// string, and one kind in each map.
fn map_kind(ty: Ty, want: Mk, at: Span, what: &str, map: Ty) -> Result<(), Diagnostic> {
    if ty == mk_ty(want) {
        return Ok(());
    }
    let (noun, plural) = match what {
        "key" => ("a map key", "keys"),
        "value" => ("a map value", "values"),
        _ => ("a map's default", "values"),
    };
    if matches!(ty, Ty::Num | Ty::Str) {
        let kind = if want == Mk::Str { "strings" } else { "numbers" };
        return Err(Diagnostic::error(format!("this map's {plural} are {kind}, and this {} is {}", if what == "key" { "key" } else { "value" }, crate::lower::a_or_an(&ty.name())))
            .with_primary(at, format!("this is {}", crate::lower::a_or_an(&ty.name())))
            .with_note(format!("a map has one kind of {}: all numbers or all strings (this one is {})", &plural[..plural.len() - 1], crate::lower::a_or_an(&map.name()))));
    }
    Err(Diagnostic::error(format!("{noun} must be a number or a string")).with_primary(at, format!("this is {}", crate::lower::a_or_an(&ty.name()))))
}

fn lower_ad(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let arity = match callee {
        "__ad_push" | "__ad_pop" | "__ad_restore" | "__ad_zero" => 1,
        "__ad_unb" | "__ad_cmp" => 2,
        "__ad_mm_a" | "__ad_mm_b" | "__ad_take" => 3,
        "__ad_extent" | "__ad_head" => 2,
        "__ad_tail" => 3,
        "__ad_cat" => 4,
        "__ad_keep" => 3,
        "__ad_like" => 2,
        "__ad_prod" => 1,
        "__ad_prodax" => 2,
        "__ad_takeax" => 4,
        _ => return Err(Diagnostic::error(format!("call to unknown function `{callee}`")).with_primary(span, "no such function")),
    };
    if args.len() != arity {
        return Err(Diagnostic::error(format!("`{callee}` takes {arity} argument(s)")).with_primary(span, "wrong number of arguments"));
    }
    let real = Ty::Arr(Elem::Num);
    let mut ids = Vec::with_capacity(args.len());
    let mut kinds = String::new();
    let mut tys = Vec::with_capacity(args.len());
    for a in args {
        let (id, ty) = lower_expr(prog, a, scope, sigs)?;
        kinds.push(match ty {
            Ty::Num | Ty::Bool => 'n',
            t if t == real => 'a',
            Ty::Record(r) => {
                // a record the differentiated code keeps across loop passes:
                // one `grad` could not split into variables (`grad::scalarize`),
                // because the def that returns it is recursive and not inlined
                let what = match crate::tuples::tuple_def(r) {
                    Some(def) => format!("the values `{def}` returns"),
                    None => format!("a {r} record"),
                };
                return Err(Diagnostic::error(format!("`grad` cannot keep {what} on its tape"))
                    .with_primary(a.span(), "overwritten in each pass of this loop")
                    .with_note("a recursive def is not inlined, so its record stays whole; call it outside the differentiated def and pass the values in"));
            }
            t => {
                return Err(Diagnostic::error(format!("`grad` differentiates numbers and real arrays; this is {}", crate::lower::a_or_an(&t.name())))
                    .with_primary(a.span(), "not a number or a real array"))
            }
        });
        ids.push(id);
        tys.push(ty);
    }
    let k: Vec<char> = kinds.chars().collect();
    let mut call = |name: &str, kids: Vec<NodeId>, ty: Ty| Ok((prog.add(Viv::Call { callee: name.to_string() }, span, kids), ty));
    match callee {
        "__ad_push" => call(if k[0] == 'a' { "viv_ad_push_a" } else { "viv_ad_push_n" }, ids, Ty::Num),
        // the argument only says the type of what comes off the tape
        "__ad_pop" => {
            let name = if k[0] == 'a' { "viv_ad_pop_a" } else if tys[0] == Ty::Bool { "viv_ad_pop_b" } else { "viv_ad_pop_n" };
            call(name, vec![], tys[0])
        }
        "__ad_restore" if k[0] == 'a' => call("viv_ad_restore", ids, Ty::Num),
        "__ad_zero" if k[0] == 'a' => call("viv_ad_zero", ids, real),
        "__ad_zero" if tys[0] == Ty::Num => Ok((prog.add(Viv::Num(0.0), span, vec![]), Ty::Num)),
        "__ad_unb" => match (k[0], k[1]) {
            ('a', 'n') => call("viv_ad_unb_an", ids, Ty::Num),
            ('n', 'a') => call("viv_ad_unb_na", ids, real),
            // an array broadcast against a larger one (interp/arrays.rs)
            ('a', 'a') => call("viv_ad_unb_aa", ids, real),
            _ => Ok((ids[0], tys[0])),
        },
        "__ad_cmp" => call(&format!("viv_ad_cmp_{kinds}"), ids, if kinds == "nn" { Ty::Num } else { real }),
        "__ad_extent" if kinds == "na" => call("viv_ad_extent", ids, Ty::Num),
        "__ad_cat" if kinds == "anna" => call("viv_ad_cat", ids, real),
        // the shares of an append's or an extend's adjoint (interp/growable.rs)
        "__ad_head" if kinds == "aa" => call("viv_grow_head", ids, real),
        "__ad_tail" if kinds == "aan" => call("viv_grow_tail_n", ids, Ty::Num),
        "__ad_tail" if kinds == "aaa" => call("viv_grow_tail_a", ids, real),
        "__ad_mm_a" | "__ad_mm_b" | "__ad_take" if kinds == "aaa" => call(&format!("viv{callee}").replace("viv__ad", "viv_ad"), ids, real),
        // the adjoints of the array vocabulary (interp/arrays.rs, "adjoints")
        "__ad_keep" if kinds == "aan" => call("viv_ad_keep", ids, real),
        "__ad_like" if kinds == "aa" => call("viv_ad_like", ids, real),
        "__ad_prod" if kinds == "a" => call("viv_ad_prod", ids, real),
        "__ad_prodax" if kinds == "an" => call("viv_ad_prodax", ids, real),
        "__ad_takeax" if kinds == "aaan" => call("viv_ad_takeax", ids, real),
        _ => Err(Diagnostic::error(format!("`{callee}`: arguments of the wrong kind ({kinds})")).with_primary(span, "here")),
    }
}

pub fn lower(program: &Program, source: SourceId) -> Result<IrProgram<Viv>, Vec<Diagnostic>> {
    let mut errs = Vec::new();
    // `n = int(x)` makes an Int (lower/ints.rs, infer_int_vars)
    ints::begin(program);
    let with_ints = ints::infer_int_vars(program);
    let program = with_ints.as_ref().unwrap_or(program);
    ints::begin(program);
    f32s::begin();

    // Reject duplicate function definitions.
    let mut seen = std::collections::HashSet::new();
    for item in &program.items {
        if let Item::Def(def) = item {
            if !seen.insert(def.name.clone()) {
                errs.push(
                    Diagnostic::error(format!("function `{}` is already defined", def.name))
                        .with_primary(def.name_span, "redefined here"),
                );
            }
        }
    }

    // Reject a read of a global inside a def that later makes a local of the
    // same name. See `global_reads_before_local`.
    errs.extend(check_global_shadowing(program));

    // Reject a top-level variable named like a def (`optimizer = ...` in a
    // program that uses lib/optim's `def optimizer`). See
    // `check_def_variable_collisions`.
    errs.extend(check_def_variable_collisions(program));

    // Reject a `def` whose name is one of the C names this backend EMITS for a
    // Viv builtin. Viv spells them `mod`/`min`/`max`/`gamma`, so `def fmod` is
    // not shadowing anything at the Viv level -- but the compiled output would
    // have the user's function and the emitted `fmod` call under one C name,
    // and the call would silently reach the wrong one. (Measured before this
    // check: `def fmod` made `mod(7, 3)` return 10 compiled and 1 interpreted.)
    // The interpreter is unaffected, so this is refused rather than allowed to
    // diverge. Proper fix, deferred: emit builtin calls through their own IR
    // node instead of by C name.
    for item in &program.items {
        if let Item::Def(def) = item {
            let viv_spelling = match def.name.as_str() {
                "fmod" => Some("mod"),
                "fmin" => Some("min"),
                "fmax" => Some("max"),
                "tgamma" => Some("gamma"),
                _ => None,
            };
            if let Some(viv) = viv_spelling {
                errs.push(
                    Diagnostic::error(format!(
                        "`{}` cannot be a function name: it is the C name Viv emits for `{}`",
                        def.name, viv
                    ))
                    .with_primary(def.name_span, "rename this function")
                    .with_note(format!(
                        "a variable named `{}` is fine -- only a `def` collides",
                        def.name
                    )),
                );
            }
        }
    }

    // Build/validate record types first; inference and lowering need the
    // registry. Enum names come before them (a record field may be an enum),
    // their variants after.
    errs.extend(enums::begin(program));
    let (records, record_errs) = build_records(program);
    errs.extend(record_errs);
    errs.extend(enums::resolve(program, &records));
    let mut sigs = infer_signatures(program, records);
    // Growable arrays (crate::growable): appends marked, the variables they
    // grow sealed where their handle could be copied. A program that grows
    // no array is lowered as it is.
    let grown = crate::growable::prepare(program, &|f: &str| matches!(sigs.ret.get(f).copied().flatten(), Some(Ty::Num | Ty::Bool | Ty::Cplx)));
    let program = match &grown {
        Some(p) => {
            sigs = infer_signatures(p, build_records(p).0);
            p
        }
        None => program,
    };
    sigs.story = !program.debug.is_empty();
    sigs.pinned = program.memory.clone();
    sigs.owned = writes_records(program, &sigs);

    if std::env::var("KNOT_DEBUG_SIGS").is_ok() {
        let mut names: Vec<&String> = sigs.ret.keys().collect();
        names.sort();
        for f in names {
            let ps: Vec<String> = sigs.params[f].iter().map(|t| format!("{:?}", t)).collect();
            eprintln!("SIG {f}({}) -> {:?}", ps.join(", "), sigs.ret[f]);
        }
    }

    let mut prog = IrProgram::new();
    // File-scope globals, in a deterministic order.
    let mut global_list: Vec<(&String, &Ty)> = sigs.globals.iter().collect();
    global_list.sort_by(|a, b| a.0.cmp(b.0));
    for (name, ty) in global_list {
        let g = prog.add(Viv::Global { name: name.clone(), ty: *ty }, Span::synthetic(source), vec![]);
        prog.add_root(g);
    }

    // Emit a struct typedef for each record type (deterministic: declaration
    // order, a record after the records its fields name).
    for rd in record_order(program) {
        {
            if let Some(info) = sigs.records.get(&rd.name) {
                let def = prog.add(
                    Viv::RecordDef { name: info.name.to_string(), fields: info.fields.clone() },
                    rd.span,
                    vec![],
                );
                prog.add_root(def);
                // a program whose records can be written copies and keeps
                // them through helpers of their own
                if sigs.owned {
                    for kind in 0..6 {
                        let fns = prog.add(Viv::RecFns { rec: info.name.to_string(), kind, fields: info.fields.clone() }, rd.span, vec![]);
                        prog.add_root(fns);
                    }
                }
            }
        }
    }

    let top: Vec<Stmt> = program.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    let top_lens = list_lens(&top, &[]);
    let mut main_scope = Scope::top_level(&sigs.globals);
    let mut main_stmts = Vec::new();
    let reached = reachable_defs(program);
    for item in &program.items {
        match item {
            Item::Record(_) | Item::Enum(_) => {}
            Item::Rung(r) => errs.push(Diagnostic::error("a rung reached the lowering: parse() flattens ladders").with_primary(r.span, "here")),
            // A def nothing calls, with a parameter nothing gave a type to
            // (a test-only helper in a `#! use`d module, whose tests are not
            // part of the program), cannot be typed from the program: its
            // parameters are guessed (real arrays where indexed, numbers
            // otherwise), and `conj(x) @ transpose(x)` in it then refused a
            // program that never runs it. When it does not type with the
            // guess but does with another (see `types_somehow`), it is left
            // out, as the interpreter never runs it; when no typing works, its
            // error is reported. Tries are lowered into a scratch program, so
            // a failed one leaves no nodes behind.
            Item::Def(def)
                if !reached.contains(&def.name)
                    && (0..def.params.len()).any(|i| sigs.guessed.contains(&(def.name.clone(), i)))
                    && lower_def(&mut IrProgram::new(), def, &sigs).is_err()
                    && types_somehow(def, &sigs) => {}
            Item::Def(def) => match lower_def(&mut prog, def, &sigs) {
                Ok(id) => prog.add_root(id),
                Err(e) => errs.push(e),
            },
            Item::Stmt(stmt) => match {
                LIST_LENS.with(|l| *l.borrow_mut() = top_lens.clone());
                lower_stmt(&mut prog, stmt, &mut main_scope, &sigs, None, 0)
            } {
                Ok(id) => main_stmts.push(id),
                Err(e) => errs.push(e),
            },
        }
    }
    if !main_stmts.is_empty() {
        // checkpoint() snapshots every top-level variable; refuse types the
        // snapshot format cannot carry rather than silently corrupting resumes.
        let has_ckpt = main_stmts.iter().any(|&id| {
            matches!(prog.get(id).kind, Viv::Ckpt)
                || (matches!(prog.get(id).kind, Viv::ExprStmt)
                    && prog.get(id).children.first().map(|&c| matches!(prog.get(c).kind, Viv::Ckpt)).unwrap_or(false))
        });
        if has_ckpt {
            for (name, ty) in &sigs.globals {
                if !matches!(ty, Ty::Num | Ty::Bool | Ty::Arr(Elem::Num) | Ty::Cplx) {
                    errs.push(Diagnostic::error(format!(
                        "`checkpoint()` cannot snapshot `{name}`: only numbers, complex scalars, and arrays are checkpointable in v1"
                    )));
                }
            }
        }
        // globals sorted by name: checkpoint save/load depends on a stable order
        let mut ck_globals: Vec<(String, Ty)> = sigs.globals.iter().map(|(n, t)| (n.clone(), *t)).collect();
        ck_globals.sort_by(|a, b| a.0.cmp(&b.0));
        if program_has_steps(program) {
            main_stmts.push(prog.add(Viv::StepsFlush, Span::synthetic(source), vec![]));
        }
        if sigs.story {
            main_stmts.push(prog.add(Viv::StoryFlush, Span::synthetic(source), vec![]));
        }
        let root = prog.add(
            Viv::Main { locals: main_scope.locals, globals: ck_globals },
            Span::synthetic(source),
            main_stmts,
        );
        prog.add_root(root);
    }

    // the defs written for operations over records (`lower::records`)
    if errs.is_empty() {
        if let Err(e) = records::emit(&mut prog, &mut sigs, &mut std::collections::HashSet::new()) {
            errs.push(e);
        }
    }
    // the program's enums and the optionals it used, after everything that
    // could name one
    let defs = enums::all();
    if !defs.is_empty() {
        let enums = defs.iter().map(|e| (e.name.to_string(), e.variants.clone())).collect();
        let d = prog.add(Viv::EnumDefs { enums }, Span::synthetic(source), vec![]);
        prog.add_root(d);
    }
    errs.extend(f32s::take_errors());
    if errs.is_empty() {
        Ok(prog)
    } else {
        // one error per place: an unpacking that fails reports once, not
        // once for each name it would have assigned
        let mut seen = std::collections::HashSet::new();
        errs.retain(|d| seen.insert((d.message.clone(), d.primary.as_ref().map(|l| (l.span.start, l.span.end)))));
        Err(errs)
    }
}

/// Whether a program can see two names share a record: it writes into a
/// field (`r.f = v`), or holds an array of records (whose elements can be
/// written). A program with neither is lowered as it was before records could
/// be written (`Sigs::owned`).
fn writes_records(program: &Program, sigs: &Sigs) -> bool {
    let mut field_write = false;
    for item in &program.items {
        match item {
            Item::Def(d) => walk_stmts(&d.body, &mut |s| field_write |= matches!(s, Stmt::SetField { .. })),
            Item::Stmt(s) => walk_stmts(std::slice::from_ref(s), &mut |s| field_write |= matches!(s, Stmt::SetField { .. })),
            _ => {}
        }
    }
    let recs = |t: &Ty| matches!(t, Ty::Recs(_));
    field_write
        || sigs.records.values().any(|r| r.fields.iter().any(|(_, t)| recs(t)))
        || sigs.globals.values().any(recs)
        || sigs.params.values().flatten().flatten().any(recs)
        || sigs.ret.values().flatten().any(recs)
        || sigs.local_tys.values().flat_map(|m| m.values()).any(recs)
}

/// The defs the top-level statements (tests included) reach: called, or
/// named as a value, directly or through other reached defs.
fn reachable_defs(program: &Program) -> std::collections::HashSet<String> {
    fn names_expr(e: &Expr, out: &mut Vec<String>) {
        match e {
            Expr::Var { name, .. } => out.push(name.clone()),
            Expr::Call { callee, args, .. } => {
                out.push(callee.clone());
                args.iter().for_each(|a| names_expr(a, out));
            }
            Expr::ArrLit { elems, .. } => elems.iter().for_each(|a| names_expr(a, out)),
            Expr::Index { base, indices, .. } => {
                names_expr(base, out);
                indices.iter().for_each(|a| names_expr(a, out));
            }
            Expr::Lambda { body, .. } => names_expr(body, out),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => names_expr(operand, out),
            Expr::Binary { lhs, rhs, .. } => {
                names_expr(lhs, out);
                names_expr(rhs, out);
            }
            Expr::RecordLit { fields, .. } => fields.iter().for_each(|(_, _, v)| names_expr(v, out)),
            Expr::Field { base, .. } => names_expr(base, out),
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }
    fn names_stmts(stmts: &[Stmt], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::Assign { value, .. } | Stmt::Return { value, .. } => names_expr(value, out),
                Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => names_expr(expr, out),
                Stmt::Test { body, .. } => names_stmts(body, out),
                Stmt::SetIndex { base, indices, value, .. } => {
                    names_expr(base, out);
                    indices.iter().for_each(|a| names_expr(a, out));
                    names_expr(value, out);
                }
                Stmt::SetField { base, value, .. } => {
                    names_expr(base, out);
                    names_expr(value, out);
                }
                Stmt::If { cond, then_body, else_body, .. } => {
                    names_expr(cond, out);
                    names_stmts(then_body, out);
                    names_stmts(else_body, out);
                }
                Stmt::While { cond, body, .. } => {
                    names_expr(cond, out);
                    names_stmts(body, out);
                }
                Stmt::For { bound, body, .. } => {
                    names_expr(bound, out);
                    names_stmts(body, out);
                }
                Stmt::StepEnter { def, .. } | Stmt::StepExit { def, .. } => out.push(def.clone()),
                Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Narrate { .. } | Stmt::Lens { .. } => {}
            }
        }
    }
    let defs: HashMap<&str, &Def> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some((d.name.as_str(), d)) } else { None }).collect();
    let mut todo = Vec::new();
    for item in &program.items {
        if let Item::Stmt(s) = item {
            names_stmts(std::slice::from_ref(s), &mut todo);
        }
    }
    let mut reached = std::collections::HashSet::new();
    while let Some(n) = todo.pop() {
        if let Some(d) = defs.get(n.as_str()) {
            if reached.insert(n) {
                names_stmts(&d.body, &mut todo);
            }
        }
    }
    reached
}

/// `a, b = e` of an `e` that is neither an array nor the values of a def that
/// returns several (`crate::tuples` reads it from a hidden variable).
fn unpack_refusal(e: &Expr, ty: Ty) -> Option<Diagnostic> {
    let Expr::Var { name, span } = e else { return None };
    if !name.starts_with(crate::tuples::UNPACKED) {
        return None;
    }
    let a = if ty.name().starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
    Some(
        Diagnostic::error(format!("cannot unpack {a} {} into several names", ty.name()))
            .with_primary(*span, format!("this is {a} {}", ty.name()))
            .with_note("`a, b = e` unpacks an array, or the values of a def that returns several (`return a, b`)"),
    )
}

/// `sort(a)` as a statement: an ascending, stable sort of the 1-D array `a` in
/// place (-0.0 before +0.0, NaN after every number; see the "sort" runtime
/// fragment and the interpreter's `sort_order`).
fn lower_sort(prog: &mut IrProgram<Viv>, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    if args.len() != 1 {
        return Err(Diagnostic::error("`sort` takes one array").with_primary(span, "expected 1 argument"));
    }
    let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
    if matches!(ty, Ty::Str | Ty::Arr(Elem::Str)) {
        let what = if ty == Ty::Str { "a string" } else { "a list of strings" };
        return Err(Diagnostic::error(format!("`sort` sorts an array of numbers in place, and {what} is never changed in place"))
            .with_primary(args[0].span(), format!("this is {what}"))
            .with_note("`sorted(words)` gives a new list in code point order, as Python's `sorted` does (of a string, its characters); `unique(words)` gives each string once"));
    }
    // an Int array sorts as Ints (B343)
    if ty == Ty::Arr(Elem::Int) {
        return Ok(prog.add(Viv::Call { callee: "viv_isort".to_string() }, span, vec![id]));
    }
    expect_arr(ty, args[0].span())?;
    Ok(prog.add(Viv::Call { callee: "viv_sort".to_string() }, span, vec![id]))
}

/// Whether a def nothing calls types with SOME choice for its guessed
/// parameters: all of them numbers, real arrays, complex numbers or complex
/// arrays, returning any of those or nothing. A def that fails every one has a
/// mistake no call could fix (an unknown name, `break` outside a loop), and it
/// is reported.
fn types_somehow(def: &Def, sigs: &Sigs) -> bool {
    let kinds = [Ty::Num, Ty::Arr(Elem::Num), Ty::Cplx, Ty::Arr(Elem::Cplx)];
    let rets = [None, Some(Ty::Num), Some(Ty::Arr(Elem::Num)), Some(Ty::Cplx), Some(Ty::Arr(Elem::Cplx)), Some(Ty::Void)];
    let guessed: Vec<usize> = (0..def.params.len()).filter(|&i| sigs.guessed.contains(&(def.name.clone(), i))).collect();
    let tries = |guess: &[Option<Ty>]| rets.iter().any(|&r| lower_def_as(&mut IrProgram::new(), def, sigs, guess, r).is_ok());
    let same = kinds.iter().any(|&k| {
        let guess: Vec<Option<Ty>> = (0..def.params.len()).map(|i| guessed.contains(&i).then_some(k)).collect();
        tries(&guess)
    });
    if same {
        return true;
    }
    // B281: one parameter an array (`len(f)`), another a number it passes to
    // a def of numbers. Every mix of numbers and real arrays, for a def of a
    // few guessed parameters (2^6 tries at most)
    if guessed.len() < 2 || guessed.len() > MIXED_GUESS_MAX {
        return false;
    }
    (1..(1u32 << guessed.len()) - 1).any(|bits| {
        let mut guess: Vec<Option<Ty>> = vec![None; def.params.len()];
        for (b, &i) in guessed.iter().enumerate() {
            guess[i] = Some(if bits >> b & 1 == 1 { Ty::Arr(Elem::Num) } else { Ty::Num });
        }
        tries(&guess)
    })
}

/// The most guessed parameters `types_somehow` mixes numbers and arrays over.
const MIXED_GUESS_MAX: usize = 6;

fn lower_def(prog: &mut IrProgram<Viv>, def: &Def, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    lower_def_as(prog, def, sigs, &[], None)
}

/// `lower_def`, with parameter `i` typed `params[i]` where that is given, and
/// the return type `ret` if given, in place of the inferred ones.
fn lower_def_as(prog: &mut IrProgram<Viv>, def: &Def, sigs: &Sigs, params: &[Option<Ty>], ret: Option<Ty>) -> Result<NodeId, Diagnostic> {
    LIST_LENS.with(|l| *l.borrow_mut() = list_lens(&def.body, &def.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>()));
    // An explicit `: Type` annotation is authoritative (and the only way to give a
    // param a record type); otherwise use the inferred type. Resolving here reports
    // an annotation that names an undeclared record.
    let mut typed_params: Vec<(String, Ty)> = Vec::with_capacity(def.params.len());
    for (i, p) in def.params.iter().enumerate() {
        let ty = match (&p.ty, params.get(i).copied().flatten()) {
            // `m: Map` takes its kinds from inference
            (Some(t), _) => match (resolve_type(t, &sigs.records)?, sigs.param_ty(&def.name, i)) {
                (Ty::Map(..), inferred @ Ty::Map(..)) => inferred,
                (t, _) => settle(t),
            },
            (None, Some(t)) => t,
            (None, None) => sigs.param_ty(&def.name, i),
        };
        typed_params.push((p.name.clone(), ty));
    }
    let ret = match (&def.ret, ret) {
        (Some(t), _) => match (resolve_type(t, &sigs.records)?, sigs.ret_ty(&def.name)) {
            (Ty::Map(..), inferred @ Ty::Map(..)) => inferred,
            (t, _) => settle(t),
        },
        (None, Some(t)) => t,
        (None, None) => sigs.ret_ty(&def.name),
    };
    let src = step_def_name(&def.body).unwrap_or_else(|| def.name.clone());
    let mut scope = Scope::function(&typed_params, &sigs.globals);
    scope.inferred = sigs.local_tys.get(&def.name);
    scope.step_src = src.clone();
    let mut body = Vec::with_capacity(def.body.len());
    for stmt in &def.body {
        body.push(lower_stmt(prog, stmt, &mut scope, sigs, Some(ret), 0)?);
    }
    // A def that returns a value and can reach its end without a `return`:
    // a def returning a number gives 0 there, in every engine; any other
    // type has no value to give, so the def is refused.
    if !matches!(ret, Ty::Void | Ty::Num) && can_finish(&def.body) {
        let n = ret.name();
        let a = if n.starts_with(['a', 'e', 'i', 'o', 'u', 'I']) { "an" } else { "a" };
        return Err(Diagnostic::error(format!("`{}` returns {a} {n} but can reach its end without a `return`", def.name))
            .with_primary(def.name_span, "the end of this def is reachable")
            .with_note("end every path with `return`, or with `stop(\"...\")` where no value makes sense"));
    }
    // array-typed parameters count: `sum(a * b)` allocates with no array local
    let arena = scope.touches_arrays.get() || typed_params.iter().any(|(_, t)| matches!(t, Ty::Arr(_) | Ty::Record(_) | Ty::Recs(_) | Ty::Str));
    Ok(prog.add(Viv::Func { name: def_symbol(&def.name), src, params: typed_params, ret, locals: scope.locals, arena }, def.span, body))
}

/// Whether the end of `stmts` can be reached: no statement on the way
/// returns, stops the run (`stop`, a placeholder's body), or loops forever. (A `match` without `_` ends in
/// its exhaustiveness check, which is never reached once it compiles.)
fn can_finish(stmts: &[Stmt]) -> bool {
    stmts.iter().all(|s| match s {
        Stmt::Return { .. } => false,
        Stmt::Lens { lens, .. } => !matches!(&**lens, Lens::Stop { .. } | Lens::Placeholder { .. }),
        Stmt::ExprStmt { expr: Expr::Call { callee, .. }, .. } => callee != "stop" && callee != crate::parser::MATCH_REST,
        Stmt::If { then_body, else_body, .. } => can_finish(then_body) || can_finish(else_body),
        Stmt::While { cond: Expr::Bool { value: true, .. }, body, .. } => breaks(body),
        _ => true,
    })
}

/// Whether a `break` in `stmts` leaves the loop they are the body of.
fn breaks(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Break { .. } => true,
        Stmt::If { then_body, else_body, .. } => breaks(then_body) || breaks(else_body),
        _ => false,
    })
}

fn lower_block(
    prog: &mut IrProgram<Viv>,
    stmts: &[Stmt],
    span: Span,
    scope: &mut Scope<'_>,
    sigs: &Sigs,
    ret_ty: Option<Ty>,
    loop_depth: u32,
) -> Result<NodeId, Diagnostic> {
    let mut ids = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        ids.push(lower_stmt(prog, stmt, scope, sigs, ret_ty, loop_depth)?);
    }
    Ok(prog.add(Viv::Block, span, ids))
}

fn lower_stmt(
    prog: &mut IrProgram<Viv>,
    stmt: &Stmt,
    scope: &mut Scope<'_>,
    sigs: &Sigs,
    ret_ty: Option<Ty>,
    loop_depth: u32,
) -> Result<NodeId, Diagnostic> {
    // a store into a packed array, `show` of a Float32 or a packed array:
    // one call, so this frame holds one result
    if let Some(r) = f32s::stmt_hook(prog, stmt, scope, sigs) {
        return r;
    }
    match stmt {
        Stmt::Assign { name, value, span, .. } => {
            // `[]` takes the type the variable holds (a list of strings, when
            // strings are appended to it)
            let want = scope.get(name).or_else(|| scope.inferred.and_then(|m| m.get(name).copied()));
            let (vid, ty) = strings::lower_expected(prog, value, want, scope, sigs)?;
            let (vid, ty) = match want {
                Some(w) => bind_conv(prog, vid, ty, w, value),
                None => (vid, ty),
            };
            scope.bind(name, ty).map_err(|prior| {
                let d = Diagnostic::error(format!("`{name}` is {} but is assigned {}", art(prior), art(ty))).with_primary(*span, "type cannot change");
                // a number where an Int is: say how to make one
                if (prior, ty) == (Ty::Int, Ty::Num) {
                    d.with_note("a number becomes an Int only where you say how: `int(x)` (toward zero), `round(x)`, `floor(x)` or `ceil(x)`; `a // b` is an Int when a and b are")
                } else {
                    d
                }
            })?;
            // `x = x` (the growable pass's seal) moves the value onto itself
            let to_itself = matches!(unseal(value), Expr::Var { name: n, .. } if n == name);
            let vid = if to_itself { vid } else { owned(prog, vid, ty, value, sigs) };
            Ok(prog.add(Viv::Assign { name: name.clone() }, *span, vec![vid]))
        }
        Stmt::SetIndex { base, indices, value, span } => {
            // The base (a variable or a record field like `m.node_x`) must be an
            // array; writes go through its shared `viv_arr` handle.
            let (bid, bty) = lower_expr(prog, base, scope, sigs)?;
            if bty == Ty::Str {
                return Err(Diagnostic::error(crate::interp::strings::STR_IMMUTABLE).with_primary(base.span(), "a string is not changed in place")
                    .with_note("build a new string: `s = s[:i] + c + s[i + 1:]`"));
            }
            if bty == Ty::Arr(Elem::Str) {
                return Err(Diagnostic::error(crate::interp::strings::STRS_IMMUTABLE).with_primary(base.span(), "a list of strings is not changed in place"));
            }
            // `m[k] = v` of a map: the key, then the value, then the store
            if let Ty::Map(mk, mv) = bty {
                let [k] = indices.as_slice() else {
                    return Err(Diagnostic::error(crate::interp::maps::ONE_KEY).with_primary(*span, format!("{} keys given", indices.len())));
                };
                let (kid, kty) = lower_expr(prog, k, scope, sigs)?;
                let (kid, kty) = ints::to_num(prog, kid, kty, k.span());
                map_kind(kty, mk, k.span(), "key", bty)?;
                let (vid, vty) = lower_expr(prog, value, scope, sigs)?;
                let (vid, vty) = ints::to_num(prog, vid, vty, value.span());
                map_kind(vty, mv, value.span(), "value", bty)?;
                let set = prog.add(Viv::Call { callee: crate::interp::maps::callee("set", mk, mv) }, *span, vec![bid, kid, vid]);
                return Ok(prog.add(Viv::ExprStmt, *span, vec![set]));
            }
            // `xs[i] = r` into an array of records
            if let Ty::Recs(rec) = bty {
                write_root(base, scope, "an element of")?;
                let [i] = indices.as_slice() else {
                    return Err(Diagnostic::error("an array of records is a list, so it takes one index")
                        .with_primary(*span, format!("{} indices given", indices.len())));
                };
                let (iid, ity) = lower_expr(prog, i, scope, sigs)?;
                expect_index(ity, i.span())?;
                let (vid, vty) = lower_expr(prog, value, scope, sigs)?;
                if vty != Ty::Record(rec) {
                    return Err(Diagnostic::error(format!("an element of this array is a {rec} but this value is {}", crate::lower::a_or_an(&vty.name())))
                        .with_primary(value.span(), "wrong element type"));
                }
                let vid = owned(prog, vid, vty, value, sigs);
                return Ok(prog.add(Viv::RecSetIndex { rec: rec.to_string(), at: site_at(*span) }, *span, vec![bid, iid, vid]));
            }
            let Ty::Arr(elem) = bty else {
                return Err(Diagnostic::error("can only assign into an array element")
                    .with_primary(base.span(), format!("this is {}", crate::lower::a_or_an(&bty.name()))));
            };
            check_index_rank(indices.len(), *span)?;
            let mut child_ids = vec![bid];
            for idx in indices {
                let (id, ty) = lower_expr(prog, idx, scope, sigs)?;
                expect_index(ty, idx.span())?;
                child_ids.push(id);
            }
            // an Int array's element: an Int (a literal adapts)
            if elem == Elem::Int {
                if child_ids.len() != 2 {
                    return Err(Diagnostic::error("an Int array has rank 1, so it takes one index").with_primary(*span, format!("{} indices given", child_ids.len() - 1)));
                }
                child_ids.push(ints::lower_as_int(prog, value, scope, sigs)?);
                let set = prog.add(Viv::Call { callee: "viv_iset".to_string() }, *span, child_ids);
                return Ok(prog.add(Viv::ExprStmt, *span, vec![set]));
            }
            let (vid, vty) = lower_expr(prog, value, scope, sigs)?;
            match elem {
                Elem::Num => {
                    if vty == Ty::Cplx {
                        return Err(Diagnostic::error("cannot store a complex number in a real array")
                            .with_primary(value.span(), "this is a complex number")
                            .with_note("allocate the array with `czeros` to hold complex values"));
                    }
                    expect_num(vty, value.span(), "an array element")?;
                    child_ids.push(vid);
                    Ok(prog.add(Viv::SetIndex { at: site_at(*span) }, *span, child_ids))
                }
                Elem::Cplx => {
                    // a real promotes, exactly as it does in scalar arithmetic
                    if vty != Ty::Cplx {
                        expect_num(vty, value.span(), "a complex array element")?;
                    }
                    child_ids.push(vid);
                    Ok(prog.add(Viv::CSetIndex { at: site_at(*span) }, *span, child_ids))
                }
                Elem::Func(_) => Err(Diagnostic::error("cannot assign into a function array")
                    .with_primary(base.span(), "a registry is built by its literal and then read")
                    .with_note("write the whole registry at once: `methods = [a, b, c]`")),
                Elem::Str => unreachable!("refused above"),
                Elem::Int | Elem::P(_) => unreachable!("lowered above"),
            }
        }
        Stmt::SetField { base, field, field_span, value, span } => {
            write_root(base, scope, "a field of")?;
            let (bid, bty) = lower_expr(prog, base, scope, sigs)?;
            let Ty::Record(rname) = bty else {
                return Err(Diagnostic::error(format!("cannot assign the field `{field}` of a {}", bty.name())).with_primary(base.span(), "not a record"));
            };
            let info = sigs.records.get(rname).expect("a record type is always registered");
            let fty = info
                .fields
                .iter()
                .find(|(f, _)| f == field)
                .map(|(_, t)| *t)
                .ok_or_else(|| Diagnostic::error(format!("record `{rname}` has no field `{field}`")).with_primary(*field_span, "unknown field"))?;
            let (vid, vty) = strings::lower_expected(prog, value, Some(fty), scope, sigs)?;
            if vty != fty {
                return Err(Diagnostic::error(format!("the field `{field}` of `{rname}` is {} but this value is {}", an(fty), an(vty)))
                    .with_primary(value.span(), "wrong field type"));
            }
            let vid = owned(prog, vid, vty, value, sigs);
            Ok(prog.add(Viv::SetField { field: field.clone(), ty: fty }, *span, vec![bid, vid]))
        }
        Stmt::Return { value, span } => {
            let Some(expected) = ret_ty else {
                return Err(Diagnostic::error("`return` outside of a function")
                    .with_primary(*span, "not allowed at the top level"));
            };
            let (vid, ty) = strings::lower_expected(prog, value, Some(expected), scope, sigs)?;
            let (vid, ty) = bind_conv(prog, vid, ty, expected, value);
            if ty != expected {
                return Err(Diagnostic::error(format!(
                    "this function returns {} but this value is {}",
                    art(expected),
                    art(ty)
                ))
                .with_primary(value.span(), "type does not match the function's other returns"));
            }
            let vid = owned(prog, vid, ty, value, sigs);
            let opens: Vec<Open> = scope.open_steps.iter().rev().cloned().collect();
            let exits: Vec<(Option<String>, String, u32)> = opens.iter().filter_map(|o| if let Open::Step(def, id, d) = o { Some((def.clone(), id.clone(), *d)) } else { None }).collect();
            let iterating = opens.iter().any(|o| matches!(o, Open::Loop(Some(_))));
            if sigs.story && (iterating || !exits.is_empty()) {
                // the debug lens's way out: step exits and iteration ends in
                // the order they close, each a node of its own
                let mut ids = vec![vid];
                for o in &opens {
                    match o {
                        Open::Step(def, id, depth) => ids.push(prog.add(Viv::StepMark { enter: false, id: id.clone(), depth: *depth, story: true, def: def.clone() }, *span, vec![])),
                        Open::Loop(Some(end)) => ids.push(lower_block(prog, end, *span, scope, sigs, ret_ty, loop_depth)?),
                        Open::Loop(None) => {}
                    }
                }
                return Ok(prog.add(Viv::LensReturn { ty }, *span, ids));
            }
            if !exits.is_empty() {
                return Ok(prog.add(Viv::StepReturn { exits, ty }, *span, vec![vid]));
            }
            Ok(prog.add(Viv::Return, *span, vec![vid]))
        }
        Stmt::ExprStmt { expr, span } => {
            // A bare string is a docstring: no runtime effect.
            if matches!(expr, Expr::Str { .. }) {
                return Ok(prog.add(Viv::Nop, *span, vec![]));
            }
            if let Expr::Call { callee, args, span: at } = expr {
                // the end of a `match` with no `_`: its arms must cover every variant
                if callee == crate::parser::MATCH_REST {
                    return enums::lower_rest(prog, args, *at, scope, sigs);
                }
                // `x: T = e`: the type of `x`
                if callee == crate::parser::DECLARE {
                    if let [Expr::Str { value: name, .. }, Expr::Str { value: text, span: tspan }] = args.as_slice() {
                        let t = settle(resolve_type(&crate::parser::type_of_text(text, *tspan), &sigs.records)?);
                        scope.bind(name, t).map_err(|prior| {
                            Diagnostic::error(format!("`{name}` is {} and cannot be declared a {}", crate::lower::a_or_an(&prior.name()), t.name())).with_primary(*at, "declared here")
                        })?;
                        return Ok(prog.add(Viv::Nop, *span, vec![]));
                    }
                }
                if callee == "sort" && !sigs.arity.contains_key("sort") {
                    let eid = lower_sort(prog, args, *at, scope, sigs)?;
                    return Ok(prog.add(Viv::ExprStmt, *span, vec![eid]));
                }
            }
            let (eid, _) = lower_expr(prog, expr, scope, sigs)?;
            Ok(prog.add(Viv::ExprStmt, *span, vec![eid]))
        }
        Stmt::Narrate { text, span } => {
            if text.is_empty() {
                Ok(prog.add(Viv::Nop, *span, vec![]))
            } else {
                Ok(prog.add(Viv::Narrate { msg: text.clone() }, *span, vec![]))
            }
        }
        Stmt::Show { expr, span } => {
            let (eid, ty) = lower_expr(prog, expr, scope, sigs)?;
            // a hidden variable (a comprehension's, crate::surface) stands for
            // the expression written, which has no label
            let label = match expr {
                Expr::Var { name, .. } if !name.starts_with(crate::parser::HIDDEN_PREFIX) => name.clone(),
                _ => String::new(),
            };
            let is_arr = match ty {
                // a string, or a list of strings: its repr
                Ty::Str | Ty::Arr(Elem::Str) => return Ok(strings::show(prog, &label, eid, ty, *span)),
                Ty::Arr(Elem::Num) => true,
                Ty::Arr(Elem::Cplx) => {
                    return Ok(prog.add(Viv::ShowCArr { label }, *span, vec![eid]));
                }
                Ty::Num => false,
                // an Int: its digits, as `print("n: ", n)` writes them
                Ty::Int => {
                    let mut parts = Vec::new();
                    if !label.is_empty() {
                        parts.push(PrintPart::Text(format!("{label}: ")));
                    }
                    parts.push(PrintPart::Str);
                    let (text, _) = ints::text_or_num(prog, eid, crate::printfmt::Fmt::BARE, *span)?;
                    let pid = prog.add(Viv::Print { parts }, *span, vec![text]);
                    return Ok(prog.add(Viv::ExprStmt, *span, vec![pid]));
                }
                Ty::Cplx => {
                    return Ok(prog.add(Viv::ShowCplx { label }, *span, vec![eid]));
                }
                Ty::Bool => {
                    return Ok(prog.add(Viv::ShowBool { label }, *span, vec![eid]));
                }
                // `show m` writes what `print("m: ", m)` writes
                Ty::Map(..) => {
                    let mut parts = Vec::new();
                    if !label.is_empty() {
                        parts.push(PrintPart::Text(format!("{label}: ")));
                    }
                    parts.push(map_part(ty, crate::printfmt::Fmt::BARE, crate::printfmt::Fmt::BARE));
                    let pid = prog.add(Viv::Print { parts }, *span, vec![eid]);
                    return Ok(prog.add(Viv::ExprStmt, *span, vec![pid]));
                }
                // an enum value: `Add(Num(1), Sym("x"))`
                Ty::Enum(_) => {
                    let mut parts = Vec::new();
                    if !label.is_empty() {
                        parts.push(PrintPart::Text(format!("{label}: ")));
                    }
                    parts.push(PrintPart::Str);
                    let text = enums::text(prog, eid, ty, *span);
                    let pid = prog.add(Viv::Print { parts }, *span, vec![text]);
                    return Ok(prog.add(Viv::ExprStmt, *span, vec![pid]));
                }
                Ty::F32 | Ty::Arr(Elem::P(_)) => unreachable!("lowered by f32s::stmt_hook"),
                // an Int array: [1, 2, 3]
                Ty::Arr(Elem::Int) => {
                    let mut parts = Vec::new();
                    if !label.is_empty() {
                        parts.push(PrintPart::Text(format!("{label}: ")));
                    }
                    parts.push(PrintPart::Str);
                    let text = prog.add(Viv::Call { callee: "viv_iarr_str".to_string() }, *span, vec![eid]);
                    let pid = prog.add(Viv::Print { parts }, *span, vec![text]);
                    return Ok(prog.add(Viv::ExprStmt, *span, vec![pid]));
                }
                Ty::Void | Ty::Record(_) | Ty::Recs(_) | Ty::Func(_) | Ty::Arr(Elem::Func(_)) => {
                    return Err(Diagnostic::error(format!("cannot `show` a {}", ty.name()))
                        .with_primary(expr.span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))))
                }
            };
            Ok(prog.add(Viv::Show { label, is_arr }, *span, vec![eid]))
        }
        Stmt::Test { name, body, span } => {
            // The body runs inline, bracketed by the test's begin/end so its
            // `assert_*` calls report against the test's NAME instead of each
            // printing an anonymous line.
            let mut ids = Vec::with_capacity(body.len() + 2);
            ids.push(prog.add(Viv::TestBegin { name: name.clone() }, *span, vec![]));
            for s in body {
                ids.push(lower_stmt(prog, s, scope, sigs, ret_ty, loop_depth)?);
            }
            ids.push(prog.add(Viv::TestEnd, *span, vec![]));
            Ok(prog.add(Viv::Block, *span, ids))
        }
        Stmt::If { cond, then_body, else_body, span } => {
            let cid = lower_cond(prog, cond, scope, sigs)?;
            let then_id = lower_block(prog, then_body, *span, scope, sigs, ret_ty, loop_depth)?;
            let else_id = lower_block(prog, else_body, *span, scope, sigs, ret_ty, loop_depth)?;
            Ok(prog.add(Viv::If, *span, vec![cid, then_id, else_id]))
        }
        Stmt::While { cond, body, span } => {
            let cid = lower_cond(prog, cond, scope, sigs)?;
            let end = iteration_end(body);
            scope.open_steps.push(Open::Loop(end.clone()));
            let body_id = lower_loop_body(prog, body, end.as_deref(), *span, scope, sigs, ret_ty, loop_depth + 1);
            scope.open_steps.pop();
            let body_id = body_id?;
            Ok(prog.add(Viv::While, *span, vec![cid, body_id]))
        }
        Stmt::For { var, bound, body, parallel, span, .. } => {
            // the pass count of `for i from a to b by s` with Int bounds is the
            // exact ceiling of the quotient, the passes the loop makes
            let (bid, bty) = match ints::int_quotient(bound, scope, sigs) {
                Some((l, r)) => (ints::lower_quotient(prog, l, r, true, bound.span(), scope, sigs)?, Ty::Int),
                None => lower_expr(prog, bound, scope, sigs)?,
            };
            // `for i to n` with an Int n counts in Ints
            if bty == Ty::Int {
                scope.bind(var, Ty::Int).map_err(|prior| {
                    Diagnostic::error(format!("`{var}` is {} and this loop counts in Ints", crate::lower::a_or_an(&prior.name()))).with_primary(bound.span(), "an Int bound")
                })?;
            } else {
                expect_num(bty, bound.span(), "a loop bound")?;
                scope.bind(var, Ty::Num).ok();
            }
            let end = iteration_end(body);
            scope.open_steps.push(Open::Loop(end.clone()));
            let body_id = lower_loop_body(prog, body, end.as_deref(), *span, scope, sigs, ret_ty, loop_depth + 1);
            scope.open_steps.pop();
            let body_id = body_id?;
            if *parallel {
                if strings::makes_strings(prog, body_id) {
                    return Err(Diagnostic::error("a `parallel for` cannot make strings").with_primary(*span, "this loop's body makes strings")
                        .with_note("make the strings in a plain `for`, and the numbers in the parallel one"));
                }
                parallel_maps(body, scope, sigs)?;
                let (privates, reductions) = analyze_parallel(body, *span)?;
                // a loop the memory lens pins (crate::parallel found it)
                let (pins, pin_what) = match sigs.pinned.iter().find(|p| p.span == *span) {
                    Some(p) => {
                        for a in &p.arrays {
                            if scope.get(a) != Some(Ty::Arr(Elem::Num)) {
                                return Err(Diagnostic::error(format!("a `disjoint` pin covers arrays of numbers, and `{a}` is not one"))
                                    .with_primary(*span, format!("this loop is pinned with `{a}`")));
                            }
                        }
                        (p.arrays.clone(), p.what.clone())
                    }
                    None => (Vec::new(), String::new()),
                };
                Ok(prog.add(Viv::ParallelFor { var: var.clone(), privates, reductions, pins, pin_what }, *span, vec![bid, body_id]))
            } else {
                Ok(prog.add(Viv::For { var: var.clone() }, *span, vec![bid, body_id]))
            }
        }
        Stmt::Break { span } => {
            if loop_depth == 0 {
                return Err(Diagnostic::error("`break` outside of a loop").with_primary(*span, "not in a loop"));
            }
            let jump = prog.add(Viv::Break, *span, vec![]);
            leave_loop_steps(prog, scope, sigs, ret_ty, loop_depth, jump, *span)
        }
        Stmt::Continue { span } => {
            if loop_depth == 0 {
                return Err(Diagnostic::error("`continue` outside of a loop").with_primary(*span, "not in a loop"));
            }
            let jump = prog.add(Viv::Continue, *span, vec![]);
            leave_loop_steps(prog, scope, sigs, ret_ty, loop_depth, jump, *span)
        }
        Stmt::StepEnter { def, id, depth, span, .. } => {
            let def = (*def != scope.step_src).then(|| def.clone());
            scope.open_steps.push(Open::Step(def.clone(), id.clone(), *depth));
            Ok(prog.add(Viv::StepMark { enter: true, id: id.clone(), depth: *depth, story: sigs.story, def }, *span, vec![]))
        }
        Stmt::StepExit { def, id, depth, span, .. } => {
            let def = (*def != scope.step_src).then(|| def.clone());
            scope.open_steps.pop();
            Ok(prog.add(Viv::StepMark { enter: false, id: id.clone(), depth: *depth, story: sigs.story, def }, *span, vec![]))
        }
        Stmt::Lens { lens, span } => lower_lens(prog, lens, *span, scope, sigs, ret_ty, loop_depth),
    }
}

/// A statement of the debug lens, or a placeholder body.
fn lower_lens(
    prog: &mut IrProgram<Viv>,
    lens: &Lens,
    span: Span,
    scope: &mut Scope<'_>,
    sigs: &Sigs,
    ret_ty: Option<Ty>,
    loop_depth: u32,
) -> Result<NodeId, Diagnostic> {
    match lens {
        Lens::Placeholder { def } => Ok(prog.add(Viv::Placeholder { def: def.clone() }, span, vec![])),
        // a check in a def the lowering wrote (`lower::records`) has no
        // place in the source of its own, and names none
        Lens::Stop { message, kind } => {
            let at = if span.start == 0 && span.end == 0 { 0 } else { site_at(span) };
            Ok(prog.add(Viv::Stop { message: message.clone(), at, kind: *kind }, span, vec![]))
        }
        Lens::Call { def } => Ok(prog.add(Viv::StoryCall { def: def.clone() }, span, vec![])),
        Lens::Ret { def } => Ok(prog.add(Viv::StoryRet { def: def.clone() }, span, vec![])),
        // a region `crate::grad` inlined: the compiled engines keep no call graph
        Lens::Inline { .. } => Ok(prog.add(Viv::Block, span, vec![])),
        Lens::Iter { def, counter, .. } => {
            // counter = counter + 1, then record it
            let var = Expr::Var { name: counter.clone(), span };
            let one = Expr::Num { value: 1.0, span };
            let bump = Stmt::Assign {
                name: counter.clone(),
                name_span: span,
                value: Expr::Binary { op: BinOp::Add, lhs: Box::new(var.clone()), rhs: Box::new(one), span },
                span,
            };
            let bump = lower_stmt(prog, &bump, scope, sigs, ret_ty, loop_depth)?;
            let (n, _) = lower_expr(prog, &var, scope, sigs)?;
            let rec = prog.add(Viv::StoryIter { def: def.clone() }, span, vec![n]);
            Ok(prog.add(Viv::Block, span, vec![bump, rec]))
        }
        Lens::IterEnd { def } => Ok(prog.add(Viv::StoryIterEnd { def: def.clone() }, span, vec![])),
        // the debug lens turns each one into an assignment before lowering
        Lens::Bind { name, .. } => Err(Diagnostic::error(format!("internal: the debug lens left its binding of `{name}` in the program")).with_primary(span, "here")),
        Lens::Watch { def, at, names, vars } => {
            let mut ids = Vec::with_capacity(names.len());
            let mut typed = Vec::with_capacity(names.len());
            for (k, (name, nspan)) in names.iter().enumerate() {
                let (id, ty) = lower_expr(prog, &Expr::Var { name: vars.get(k).unwrap_or(name).clone(), span: *nspan }, scope, sigs)?;
                if !matches!(ty, Ty::Num | Ty::Bool | Ty::Arr(Elem::Num)) {
                    return Err(Diagnostic::error(format!("cannot watch `{name}`: it is {}", crate::lower::a_or_an(&ty.name())))
                        .with_primary(*nspan, "a watch records numbers, booleans and real arrays"));
                }
                ids.push(id);
                typed.push((name.clone(), ty));
            }
            Ok(prog.add(Viv::StoryWatch { def: def.clone(), at: at.clone(), names: typed }, span, ids))
        }
        Lens::Hold { def, at, check, sentence } => match check {
            HoldCheck::Cond(e) => {
                let c = lower_cond(prog, e, scope, sigs)?;
                Ok(prog.add(Viv::StoryHold { def: def.clone(), at: at.clone(), sentence: sentence.clone(), near: None }, span, vec![c]))
            }
            HoldCheck::Near { diff, tol } => {
                let (d, dty) = lower_expr(prog, diff, scope, sigs)?;
                let arr = match dty {
                    Ty::Num => false,
                    Ty::Arr(Elem::Num) => true,
                    other => {
                        return Err(Diagnostic::error(format!("`within` compares numbers or real arrays, not a {}", other.name()))
                            .with_primary(diff.span(), format!("this is {}", crate::lower::a_or_an(&other.name()))))
                    }
                };
                let (t, tty) = lower_expr(prog, tol, scope, sigs)?;
                expect_num(tty, tol.span(), "a hold's tolerance")?;
                Ok(prog.add(Viv::StoryHold { def: def.clone(), at: at.clone(), sentence: sentence.clone(), near: Some(arr) }, span, vec![d, t]))
            }
        },
    }
}

/// The statements that end each iteration of a loop, when the debug lens made
/// it an iteration loop (its body starts with `Lens::Iter`).
fn iteration_end(body: &[Stmt]) -> Option<Vec<Stmt>> {
    match body.first() {
        Some(Stmt::Lens { lens, .. }) => match &**lens {
            Lens::Iter { end, .. } => Some(end.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// A loop body; an iteration loop's body ends with its iteration end.
#[allow(clippy::too_many_arguments)]
fn lower_loop_body(
    prog: &mut IrProgram<Viv>,
    body: &[Stmt],
    end: Option<&[Stmt]>,
    span: Span,
    scope: &mut Scope<'_>,
    sigs: &Sigs,
    ret_ty: Option<Ty>,
    loop_depth: u32,
) -> Result<NodeId, Diagnostic> {
    let Some(end) = end else {
        return lower_block(prog, body, span, scope, sigs, ret_ty, loop_depth);
    };
    let all: Vec<Stmt> = body.iter().chain(end.iter()).cloned().collect();
    lower_block(prog, &all, span, scope, sigs, ret_ty, loop_depth)
}

/// A `break` or `continue` inside steps that are themselves inside the loop
/// leaves those steps: record their exits, innermost first, then jump.
///
/// An iteration loop of the debug lens also ends its iteration, after those
/// exits and before the jump, as the interpreter does.
fn leave_loop_steps(
    prog: &mut IrProgram<Viv>,
    scope: &mut Scope<'_>,
    sigs: &Sigs,
    ret_ty: Option<Ty>,
    loop_depth: u32,
    jump: NodeId,
    span: Span,
) -> Result<NodeId, Diagnostic> {
    let mut ids: Vec<NodeId> = Vec::new();
    let mut end: Option<Vec<Stmt>> = None;
    for o in scope.open_steps.iter().rev() {
        match o {
            Open::Step(def, id, depth) => ids.push(prog.add(Viv::StepMark { enter: false, id: id.clone(), depth: *depth, story: sigs.story, def: def.clone() }, span, vec![])),
            Open::Loop(e) => {
                end = e.clone();
                break;
            }
        }
    }
    if let Some(end) = end {
        ids.push(lower_block(prog, &end, span, scope, sigs, ret_ty, loop_depth)?);
    }
    if ids.is_empty() {
        return Ok(jump);
    }
    ids.push(jump);
    Ok(prog.add(Viv::Block, span, ids))
}

/// The source name the steps in a def body were written under: the parser
/// records it on each step, and a defun specialization keeps it.
fn step_def_name(body: &[Stmt]) -> Option<String> {
    let mut found = None;
    walk_stmts(body, &mut |s| {
        if let (None, Stmt::StepEnter { def, .. }) = (&found, s) {
            found = Some(def.clone());
        }
    });
    found
}

/// Every statement in `body`, nested blocks included, in source order.
fn walk_stmts(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in body {
        f(s);
        match s {
            Stmt::If { then_body, else_body, .. } => {
                walk_stmts(then_body, f);
                walk_stmts(else_body, f);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => walk_stmts(body, f),
            _ => {}
        }
    }
}

/// Whether the program has a step anywhere, which is when `main` ends by
/// printing the step events.
fn program_has_steps(program: &Program) -> bool {
    let mut any = false;
    for item in &program.items {
        let body: &[Stmt] = match item {
            Item::Def(d) => &d.body,
            Item::Stmt(s) => std::slice::from_ref(s),
            _ => continue,
        };
        walk_stmts(body, &mut |s| any |= matches!(s, Stmt::StepEnter { .. }));
    }
    any
}

// ── source sites for run-time errors ─────────────────────────────────────
//
// An index out of bounds stops the compiled program with the interpreter's
// words AND its location: the same "  --> file:line:col", source line and
// caret the interpreter prints (interp::run_diagnosed). The C cannot compute
// that text, so the compiler renders it, once per indexing site, into a table
// (`viv_sites`, a runtime fragment built here) and each checked access passes
// its row number. Row 0 is empty: the location of an access compiled without
// a source (`compile_c`, the browser's "show the C" view).

struct Sites {
    /// The program as the engines see it, under its display name.
    map: knotkit::span::SourceMap,
    src: String,
    /// Rendered locations; the row number is what the C passes.
    rows: Vec<String>,
    index: HashMap<String, usize>,
    /// The tables of the other translation units of a split emission
    /// (`enter_unit`): each unit numbers its rows from 1 in its own table,
    /// so that its C does not depend on the rest of the program.
    parked: HashMap<Option<String>, (Vec<String>, HashMap<String, usize>)>,
    unit: Option<String>,
}

/// Make `unit` the translation unit whose site rows are numbered and
/// rendered (`None`: the program's own). See `CEmit::enter_unit`.
fn sites_enter(unit: Option<&str>) {
    SITES.with(|c| {
        let mut c = c.borrow_mut();
        let Some(s) = c.as_mut() else { return };
        let want = unit.map(str::to_string);
        if s.unit == want {
            return;
        }
        let cur = (std::mem::take(&mut s.rows), std::mem::take(&mut s.index));
        let old = std::mem::replace(&mut s.unit, want.clone());
        s.parked.insert(old, cur);
        let (rows, index) = s.parked.remove(&want).unwrap_or_else(|| (vec![String::new()], HashMap::new()));
        s.rows = rows;
        s.index = index;
    });
}

thread_local! {
    static SITES: std::cell::RefCell<Option<Sites>> = std::cell::RefCell::new(None);
}

/// Lower and emit a program inside `f` with its source known, so each array
/// access in the C carries its location. `name` is the program's name as
/// both engines print it (main.rs `display_path`: never an absolute path,
/// since this C is hashed and compiled into served wasm), and `src` the
/// whole program text (preludes and `#! use`d modules included) -- exactly
/// what the interpreter locates errors in.
pub fn with_sites<T>(name: &str, src: &str, f: impl FnOnce() -> T) -> T {
    let mut map = knotkit::span::SourceMap::new();
    map.add(name.to_string(), src.to_string());
    with_sites_map(map, src, f)
}

/// `with_sites`, with the program's source map given: `src` is its source 0,
/// and the origins it records (the modules a program uses, the Python file it
/// was translated from) are where the rendered locations point, as the
/// interpreter's errors do when `viv run` renders them through the same map.
pub fn with_sites_map<T>(map: knotkit::span::SourceMap, src: &str, f: impl FnOnce() -> T) -> T {
    SITES.with(|c| {
        *c.borrow_mut() = Some(Sites { map, src: src.to_string(), rows: vec![String::new()], index: HashMap::new(), parked: HashMap::new(), unit: None })
    });
    let r = f();
    SITES.with(|c| *c.borrow_mut() = None);
    r
}

/// The `at` of an index node: its span start + 1 while sites are on, else 0.
fn site_at(span: Span) -> u32 {
    SITES.with(|c| if c.borrow().is_some() { span.start + 1 } else { 0 })
}

/// The `viv_sites` row for an access at `span`: the location lines of the
/// interpreter's error, rendered by the same renderer. 0 with no sites.
fn site_row(span: Span) -> usize {
    use knotkit::diag::{PlainRenderer, Renderer};
    SITES.with(|c| {
        let mut c = c.borrow_mut();
        let Some(sites) = c.as_mut() else { return 0 };
        let d = Diagnostic::error("").with_primary(crate::interp::first_line_of(span, &sites.src), "here");
        let text = PlainRenderer.render(&d, &sites.map);
        // drop the headline ("error: \n"); the C prints its own message there
        let text = text.split_once('\n').map_or(String::new(), |(_, rest)| rest.to_string());
        if let Some(&row) = sites.index.get(&text) {
            return row;
        }
        sites.rows.push(text.clone());
        sites.index.insert(text, sites.rows.len() - 1);
        sites.rows.len() - 1
    })
}

/// A checked access with a known source site calls the `_at` form of its
/// accessor and passes its `viv_sites` row as the last argument; one without
/// (`at` 0: compiled with no source) calls the plain form, which reports
/// no location. Returns the name to call.
fn with_site(func: &str, at: u32, span: Span, args: &mut Vec<CExpr>, id: NodeId) -> String {
    if at == 0 {
        return func.to_string();
    }
    args.push(CExpr::raw(id, site_row(span).to_string()));
    format!("{func}_at")
}

/// `(*(viv_rec_T*)viv_recs_at(xs, i, sizeof(viv_rec_T)))`: the element of
/// an array of records, an lvalue, its index checked.
fn rec_elem(cx: &mut CEmitCx<Viv>, id: NodeId, rec: &str, base: NodeId, index: NodeId, at: u32, span: Span) -> CExpr {
    let mut args = vec![cx.expr(base), cx.expr(index), CExpr::raw(id, format!("sizeof(viv_rec_{rec})"))];
    let func = with_site("viv_recs_at", at, span, &mut args, id);
    let ptr = CExpr::call(id, &func, args);
    // parenthesized whole (a unary with no operator), so a field of it reads `(*(T*)(p)).f`
    CExpr::unary(id, "", CExpr::unary(id, format!("*(viv_rec_{rec}*)"), ptr))
}

/// One helper of the record type `rec` (`Viv::RecFns`): its own copy
/// (`viv_own_T`, `viv_own_recs_T`), keeping what it holds on the arena
/// (`viv_keep_T`, `viv_keep_recs_T`), and building arrays of it
/// (`viv_recs_lit_T`, `viv_recs_push_T`).
fn rec_fn_c(id: NodeId, rec: &str, kind: u8, fields: &[(String, Ty)], cx: &mut CEmitCx<Viv>) -> CItem {
    cx.need("recs");
    cx.need("arena_keep");
    if fields.iter().any(|f| strings::is_text(f.1)) {
        cx.need("str_list");
    }
    let records = record_registry(cx.program());
    let t = format!("viv_rec_{rec}");
    let named = |s: &str| CType::Named(s.to_string());
    let raw = |s: String| CStmt::expr(id, CExpr::raw(id, s));
    let elem = |a: &str| format!("((({t}*){a}.data)[viv_i])");
    let deep = fields.iter().any(|(_, ft)| holds_recs(*ft, &records));
    // a record reaching an enum: its keep helpers take the enum walk's mark
    let en = fields.iter().any(|(_, ft)| holds_enum(*ft, &records));
    let (ret, name, params, body) = match kind {
        0 => {
            let mut body = Vec::new();
            for (f, ft) in fields {
                let copy = match ft {
                    Ty::Recs(u) => format!("viv_own_recs_{u}"),
                    Ty::Record(g) if holds_recs(*ft, &records) => format!("viv_own_{g}"),
                    _ => continue,
                };
                let place = || CExpr::field(id, CExpr::var(id, "r"), f.clone());
                body.push(CStmt::expr(id, CExpr::binary(id, place(), "=", CExpr::call(id, copy, vec![place()]))));
            }
            body.push(CStmt::ret(id, CExpr::var(id, "r")));
            (named(&t), format!("viv_own_{rec}"), vec![(named(&t), "r".to_string())], body)
        }
        1 => {
            let mut body = vec![CStmt::decl(id, named("viv_recs"), "b", CExpr::raw(id, format!("viv_recs_from(a.data, a.size, sizeof({t}))")))];
            if deep {
                body.push(raw(format!("for (int viv_i = 0; viv_i < b.size; viv_i++) {} = viv_own_{rec}({})", elem("b"), elem("b"))));
            }
            body.push(CStmt::ret(id, CExpr::var(id, "b")));
            (named("viv_recs"), format!("viv_own_recs_{rec}"), vec![(named("viv_recs"), "a".to_string())], body)
        }
        2 => {
            let mut body = Vec::new();
            for (f, ft) in fields {
                keep_calls(id, "r", &mut vec![f.clone()], *ft, &records, "viv_kmark", &mut body);
            }
            let mut params = vec![(named(&t), "r".to_string())];
            if en {
                params.push((named("size_t"), "viv_kmark".to_string()));
            }
            (CType::Named("void".to_string()), format!("viv_keep_{rec}"), params, body)
        }
        3 => {
            let body = vec![
                raw("viv_arena_keep(a.data)".to_string()),
                raw(format!("for (int viv_i = 0; viv_i < a.size; viv_i++) viv_keep_{rec}({}{})", elem("a"), if en { ", viv_kmark" } else { "" })),
            ];
            let mut params = vec![(named("viv_recs"), "a".to_string())];
            if en {
                params.push((named("size_t"), "viv_kmark".to_string()));
            }
            (CType::Named("void".to_string()), format!("viv_keep_recs_{rec}"), params, body)
        }
        4 => {
            let body = vec![
                CStmt::decl(id, named("viv_recs"), "b", CExpr::raw(id, format!("viv_recs_new(n, sizeof({t}))"))),
                raw("va_list ap".to_string()),
                raw("va_start(ap, n)".to_string()),
                raw(format!("for (int viv_i = 0; viv_i < n; viv_i++) {} = va_arg(ap, {t})", elem("b"))),
                raw("va_end(ap)".to_string()),
                CStmt::ret(id, CExpr::var(id, "b")),
            ];
            (named("viv_recs"), format!("viv_recs_lit_{rec}"), vec![(CType::Named("int".to_string()), "n".to_string()), (named("..."), String::new())], body)
        }
        _ => {
            let body = vec![CStmt::ret(id, CExpr::raw(id, format!("viv_recs_append(a, &r, sizeof({t}))")))];
            (named("viv_recs"), format!("viv_recs_push_{rec}"), vec![(named("viv_recs"), "a".to_string()), (named(&t), "r".to_string())], body)
        }
    };
    CItem::Func(CFunc { origin: Some(id), ret, name, params, locals: Vec::new(), body })
}

/// Whether a value of type `t` holds an array of records, itself or in a
/// field at any depth: copying it then copies those arrays (`viv_own_T`).
/// Whether a value of type `t` reaches an enum value: its keep helpers then
/// take the mark the enum walk stops at.
fn holds_enum(t: Ty, records: &HashMap<String, Vec<(String, Ty)>>) -> bool {
    match t {
        Ty::Enum(_) => true,
        Ty::Record(r) | Ty::Recs(r) => records.get(r).is_some_and(|fs| fs.iter().any(|(_, ft)| holds_enum(*ft, records))),
        _ => false,
    }
}

fn holds_recs(t: Ty, records: &HashMap<String, Vec<(String, Ty)>>) -> bool {
    match t {
        Ty::Recs(_) => true,
        Ty::Record(r) => records.get(r).is_some_and(|fs| fs.iter().any(|(_, ft)| holds_recs(*ft, records))),
        _ => false,
    }
}

/// The program's record types, by name, from its `RecordDef` roots.
fn record_registry(prog: &IrProgram<Viv>) -> HashMap<String, Vec<(String, Ty)>> {
    let mut out = HashMap::new();
    for &r in prog.roots() {
        if let Viv::RecordDef { name, fields } = &prog.get(r).kind {
            out.insert(name.clone(), fields.clone());
        }
    }
    out
}

/// The `sites` runtime fragment: the location table, and the negative-index
/// note (crate::hints), which the interpreter prints under the location too.
/// Built when the runtime is rendered, after every item is emitted, so the
/// table holds every row. The strings live for the rest of the process: a
/// fragment is `'static`, and a compile renders one.
fn sites_fragment() -> RuntimeFragment {
    let rows = SITES.with(|c| c.borrow().as_ref().map_or_else(|| vec![String::new()], |s| s.rows.clone()));
    // VIV_BOUNDS is the one switch the checks test: on unless the build is a
    // release (-DVIV_RELEASE) or drops just the checks (-DVIV_NO_BOUNDS). The
    // table is inside it so a build without checks has no unused table.
    let mut code = String::from(
        "#if !defined(VIV_RELEASE) && !defined(VIV_NO_BOUNDS)\n\
         #define VIV_BOUNDS 1\n\
         /* Where each checked array access is in the source: the lines under the\n   \
         error, as the interpreter prints them. Row 0 is an access with no source. */\n\
         static const char* const viv_sites[] = {\n",
    );
    for r in &rows {
        code.push_str("    ");
        code.push_str(&c_str(r));
        code.push_str(",\n");
    }
    code.push_str("};\n");
    code.push_str("#endif");
    RuntimeFragment { name: "sites", includes: &[], deps: &[], code: Box::leak(code.into_boxed_str()) }
}

thread_local! {
    /// The def whose C body is being emitted, for its step events. Set by
    /// `emit_item` before it emits a function's (or main's) statements.
    static STEP_DEF: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
    /// What a loop in that body needs to free its iterations' arrays
    /// (`loop_release`). Set beside STEP_DEF.
    static LOOP_SCOPE: std::cell::RefCell<LoopScope> = std::cell::RefCell::new(LoopScope::default());
}

/// The types of the variables of the def (or main) being emitted, what every
/// def of the program returns, and every record's fields: enough to tell
/// whether a loop body can allocate and what its variables hold. The IR does
/// not keep an expression's type (a node is shared by every place its text
/// occurs), so this is recomputed from the declarations, by the rules the
/// direct wasm backend uses (`Em::allocates`).
#[derive(Default)]
struct LoopScope {
    vars: HashMap<String, Ty>,
    /// (program address, node count) the two maps below were built for
    key: (usize, usize),
    rets: std::rc::Rc<HashMap<String, Ty>>,
    records: std::rc::Rc<HashMap<String, Vec<(String, Ty)>>>,
    /// The defs a `parallel for` with a `disjoint` pin reaches: their element
    /// writes go through `viv_dj_set*`, which a checked build records.
    dj_defs: std::rc::Rc<std::collections::HashSet<String>>,
    /// The def being emitted (its C name), `<main>` for main.
    cur: String,
    /// Its variables that only ever hold an array it made (`dj_fresh`), when
    /// a pinned loop reaches it, and its parameters.
    dj_fresh: std::collections::HashSet<String>,
    dj_params: std::collections::HashSet<String>,
    /// Per def, the parameters a pinned array can be passed to (`dj_pinnable`).
    dj_pinnable: std::rc::Rc<HashMap<String, std::collections::HashSet<String>>>,
}

thread_local! {
    /// How many pinned `parallel for` bodies the emitter is inside.
    static DJ_BODY: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The defs a pinned `parallel for` reaches, directly or through the defs
/// they call: every call in a pinned loop's body, then every call in those
/// defs' bodies, to a fixed point.
pub(crate) fn dj_reached(prog: &IrProgram<Viv>) -> std::collections::HashSet<String> {
    fn calls(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<String>) {
        let n = prog.get(id);
        if let Viv::Call { callee } = &n.kind {
            out.push(callee.clone());
        }
        for &c in &n.children {
            calls(prog, c, out);
        }
    }
    fn pinned(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<String>) {
        let n = prog.get(id);
        if let Viv::ParallelFor { pins, .. } = &n.kind {
            if !pins.is_empty() {
                calls(prog, n.children[1], out);
                return;
            }
        }
        for &c in &n.children {
            pinned(prog, c, out);
        }
    }
    let mut funcs: HashMap<String, NodeId> = HashMap::new();
    let mut todo = Vec::new();
    for &r in prog.roots() {
        if let Viv::Func { name, .. } = &prog.get(r).kind {
            funcs.insert(name.clone(), r);
        }
        pinned(prog, r, &mut todo);
    }
    let mut seen = std::collections::HashSet::new();
    while let Some(f) = todo.pop() {
        if let Some(&id) = funcs.get(&f) {
            if seen.insert(f) {
                calls(prog, id, &mut todo);
            }
        }
    }
    seen
}

/// The variables of a def (or main) that can only hold an array the def made
/// itself: every assignment to one is a fresh allocation (a literal,
/// `zeros`, an array operation, a runtime call that gives a new array).
/// Such a variable cannot hold a pinned array, unless it is one of the loop's
/// own pinned names, so its element writes need no record: in the ERI code
/// the scratch arrays of `mderi_block` are most of the writes a pinned pass
/// makes. Parameters are never fresh.
pub(crate) fn dj_fresh(prog: &IrProgram<Viv>, body: NodeId, params: &[String]) -> std::collections::HashSet<String> {
    let funcs: std::collections::HashSet<String> = prog.roots().iter().filter_map(|&r| if let Viv::Func { name, .. } = &prog.get(r).kind { Some(name.clone()) } else { None }).collect();
    dj_fresh_in(prog, &funcs, body, params)
}

/// `dj_fresh`, with the program's def names given.
fn dj_fresh_in(prog: &IrProgram<Viv>, funcs: &std::collections::HashSet<String>, body: NodeId, params: &[String]) -> std::collections::HashSet<String> {
    fn assigns(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<(String, NodeId)>) {
        let n = prog.get(id);
        if let Viv::Assign { name } = &n.kind {
            out.push((name.clone(), n.children[0]));
        }
        for &c in &n.children {
            assigns(prog, c, out);
        }
    }
    let fresh_value = |v: NodeId| match &prog.get(v).kind {
        Viv::ArrLit { .. } | Viv::CArrLit { .. } | Viv::Zeros | Viv::CZeros | Viv::MapMath { .. } | Viv::CArrFn { .. } | Viv::Arr { .. } => true,
        Viv::Call { callee } => !funcs.contains(callee) && crate::builtins::runtime_allocates(callee),
        _ => false,
    };
    let mut list = Vec::new();
    assigns(prog, body, &mut list);
    let mut fresh: std::collections::HashSet<String> = list.iter().map(|(n, _)| n.clone()).filter(|n| !params.contains(n)).collect();
    for (n, v) in &list {
        if !fresh_value(*v) {
            fresh.remove(n);
        }
    }
    fresh
}

/// The parameters, per def, that can be given a pinned array (`pinnable`):
/// a pinned loop passes one of its pinned arrays, or a variable that may hold
/// one, and so on down the calls, to a fixed point. A write through any other
/// parameter (the scratch arrays `mderi_block` hands its kernels) cannot land
/// in a pinned array and is not recorded.
pub(crate) fn dj_pinnable(prog: &IrProgram<Viv>) -> HashMap<String, std::collections::HashSet<String>> {
    struct F {
        params: Vec<String>,
        fresh: std::collections::HashSet<String>,
        id: NodeId,
    }
    // a program without a pinned loop has nothing to find
    if !prog.roots().iter().any(|&r| {
        fn under(prog: &IrProgram<Viv>, id: NodeId) -> bool {
            let n = prog.get(id);
            matches!(&n.kind, Viv::ParallelFor { pins, .. } if !pins.is_empty()) || n.children.iter().any(|&c| under(prog, c))
        }
        under(prog, r)
    }) {
        return HashMap::new();
    }
    let names: std::collections::HashSet<String> = prog.roots().iter().filter_map(|&r| if let Viv::Func { name, .. } = &prog.get(r).kind { Some(name.clone()) } else { None }).collect();
    let mut funcs: HashMap<String, F> = HashMap::new();
    for &r in prog.roots() {
        if let Viv::Func { name, params, .. } = &prog.get(r).kind {
            let ps: Vec<String> = params.iter().map(|(p, _)| p.clone()).collect();
            let fresh = dj_fresh_in(prog, &names, r, &ps);
            funcs.insert(name.clone(), F { params: ps, fresh, id: r });
        }
    }
    // may the argument `a`, in a def whose fresh variables are `fresh` and
    // whose pinnable parameters are `pin`, hold a pinned array?
    fn may_hold(prog: &IrProgram<Viv>, a: NodeId, fresh: &std::collections::HashSet<String>, params: &[String], pin: &std::collections::HashSet<String>, pinned: &[String], funcs: &HashMap<String, F>) -> bool {
        match &prog.get(a).kind {
            Viv::Var { name } => {
                if pinned.contains(name) || pin.contains(name) {
                    true
                } else if params.contains(name) {
                    false
                } else {
                    !fresh.contains(name)
                }
            }
            Viv::Call { callee } => funcs.contains_key(callee),
            Viv::Field { .. } => true,
            _ => false,
        }
    }
    fn calls(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<NodeId>) {
        let n = prog.get(id);
        if let Viv::Call { .. } = &n.kind {
            out.push(id);
        }
        for &c in &n.children {
            calls(prog, c, out);
        }
    }
    let mut out: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    // the seeds: the calls in each pinned loop's body
    fn loops(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<(NodeId, Vec<String>)>) {
        let n = prog.get(id);
        if let Viv::ParallelFor { pins, .. } = &n.kind {
            if !pins.is_empty() {
                out.push((n.children[1], pins.clone()));
            }
        }
        for &c in &n.children {
            loops(prog, c, out);
        }
    }
    let empty = std::collections::HashSet::new();
    let mut todo: Vec<String> = Vec::new();
    for &r in prog.roots() {
        let (params, fresh) = match &prog.get(r).kind {
            Viv::Func { name, .. } => (funcs[name].params.clone(), funcs[name].fresh.clone()),
            Viv::Main { .. } => (Vec::new(), dj_fresh(prog, r, &[])),
            _ => continue,
        };
        let mut ls = Vec::new();
        loops(prog, r, &mut ls);
        for (body, pins) in ls {
            let mut cs = Vec::new();
            calls(prog, body, &mut cs);
            for c in cs {
                let Viv::Call { callee } = &prog.get(c).kind else { continue };
                let Some(f) = funcs.get(callee) else { continue };
                for (k, &a) in prog.get(c).children.iter().enumerate() {
                    if may_hold(prog, a, &fresh, &params, &empty, &pins, &funcs) {
                        if let Some(p) = f.params.get(k) {
                            if out.entry(callee.clone()).or_default().insert(p.clone()) {
                                todo.push(callee.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    // down the calls
    while let Some(g) = todo.pop() {
        let f = &funcs[&g];
        let pin = out.get(&g).cloned().unwrap_or_default();
        let mut cs = Vec::new();
        calls(prog, f.id, &mut cs);
        for c in cs {
            let Viv::Call { callee } = &prog.get(c).kind else { continue };
            let Some(h) = funcs.get(callee) else { continue };
            for (k, &a) in prog.get(c).children.iter().enumerate() {
                if may_hold(prog, a, &f.fresh, &f.params, &pin, &[], &funcs) {
                    if let Some(p) = h.params.get(k) {
                        if out.entry(callee.clone()).or_default().insert(p.clone()) {
                            todo.push(callee.clone());
                        }
                    }
                }
            }
        }
    }
    out
}

thread_local! {
    /// The pinned names of the pinned loops being emitted, innermost last.
    static DJ_PINS: std::cell::RefCell<Vec<Vec<String>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether an element write of `base` being emitted goes through
/// `viv_dj_set*`: it is in a pinned loop's reach, and its array may be a
/// pinned one.
fn dj_write(cx: &CEmitCx<Viv>, base: NodeId) -> bool {
    if !dj_here() {
        return false;
    }
    let Viv::Var { name } = cx.kind(base) else { return true };
    let pinned = DJ_PINS.with(|p| p.borrow().iter().any(|ps| ps.contains(&name)));
    pinned
        || LOOP_SCOPE.with(|s| {
            let s = s.borrow();
            if s.dj_params.contains(&name) {
                // a parameter: recorded when a pinned array can be passed to it
                s.dj_pinnable.get(&s.cur).is_some_and(|p| p.contains(&name))
            } else {
                !s.dj_fresh.contains(&name)
            }
        })
}

/// Whether a pinned `parallel for` is under `id`.
fn pinned_in(cx: &CEmitCx<Viv>, id: NodeId) -> bool {
    fn under(prog: &IrProgram<Viv>, id: NodeId) -> bool {
        let n = prog.get(id);
        matches!(&n.kind, Viv::ParallelFor { pins, .. } if !pins.is_empty()) || n.children.iter().any(|&c| under(prog, c))
    }
    under(cx.program(), id)
}

/// Whether an element write being emitted goes through `viv_dj_set*`.
fn dj_here() -> bool {
    DJ_BODY.with(|d| d.get() > 0) || LOOP_SCOPE.with(|s| {
        let s = s.borrow();
        s.dj_defs.contains(&s.cur)
    })
}

/// Enter a def (or main) for emission: its variables, and the program-wide
/// maps, rebuilt only when the program changes.
fn enter_loop_scope(cx: &CEmitCx<Viv>, vars: impl Iterator<Item = (String, Ty)>) {
    let prog = cx.program();
    let key = (prog as *const IrProgram<Viv> as usize, prog.len());
    LOOP_SCOPE.with(|s| {
        let mut s = s.borrow_mut();
        s.vars = vars.collect();
        if s.key != key {
            let mut rets = HashMap::new();
            let mut records = HashMap::new();
            for &r in prog.roots() {
                match &prog.get(r).kind {
                    Viv::Func { name, ret, .. } => {
                        rets.insert(name.clone(), *ret);
                    }
                    Viv::RecordDef { name, fields } => {
                        records.insert(name.clone(), fields.clone());
                    }
                    _ => {}
                }
            }
            s.key = key;
            s.rets = std::rc::Rc::new(rets);
            s.records = std::rc::Rc::new(records);
            s.dj_defs = std::rc::Rc::new(dj_reached(prog));
            s.dj_pinnable = std::rc::Rc::new(dj_pinnable(prog));
        }
    });
}

/// Whether evaluating `id` can allocate an array, a record, a string or a
/// map: the direct wasm backend's `allocates`, over the same node kinds, with
/// what each call gives from the same table (`crate::builtins`).
fn c_allocates(cx: &CEmitCx<Viv>, rets: &HashMap<String, Ty>, id: NodeId) -> bool {
    let here = match cx.kind(id) {
        Viv::ArrLit { .. }
        | Viv::CArrLit { .. }
        | Viv::Zeros
        | Viv::CZeros
        | Viv::Arr { .. }
        | Viv::MapMath { .. }
        | Viv::CArrFn { .. }
        | Viv::RecordLit { .. }
        | Viv::EnumNew { .. }
        | Viv::EnumStr { .. }
        | Viv::RecArrLit { .. }
        | Viv::RecAppend { .. }
        | Viv::Own { .. }
        | Viv::FnArrLit { .. } => true,
        // a def's call allocates when it returns a heap value; a runtime
        // call when the one table says it gives one (crate::builtins)
        Viv::Call { callee } => match rets.get(&callee) {
            Some(t) => crate::builtins::heap_ty(*t),
            None => crate::builtins::runtime_allocates(&callee),
        },
        Viv::ApplyFn { sig } => sig.ret == ArgKind::Arr,
        _ => false,
    };
    here || cx.children(id).into_iter().any(|c| c_allocates(cx, rets, c))
}

/// The names `id` assigns, in order of first assignment.
fn c_assigned(cx: &CEmitCx<Viv>, id: NodeId, out: &mut Vec<String>) {
    if let Viv::Assign { name } = cx.kind(id) {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    // a write into a record or an array of records changes the variable at
    // the root of its place, which may now hold what this pass made
    if let Viv::SetField { .. } | Viv::RecSetIndex { .. } = cx.kind(id) {
        if let Some(name) = place_var(cx.program(), cx.children(id)[0]) {
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    for c in cx.children(id) {
        c_assigned(cx, c, out);
    }
}

/// The variable at the root of a place node: `r` of `r`, `r.f`, `r.xs[i].g`.
pub(crate) fn place_var(prog: &IrProgram<Viv>, mut id: NodeId) -> Option<String> {
    loop {
        let n = prog.get(id);
        match &n.kind {
            Viv::Var { name } => return Some(name.clone()),
            Viv::Field { .. } | Viv::RecIndex { .. } => id = n.children[0],
            _ => return None,
        }
    }
}

/// `viv_arena_keep(...)` for each array a value of type `t` holds: the array
/// itself, or, through a record's fields, every array in it. The value is the
/// variable `var` followed by the fields in `path`.
fn keep_calls(id: NodeId, var: &str, path: &mut Vec<String>, t: Ty, records: &HashMap<String, Vec<(String, Ty)>>, mark: &str, out: &mut Vec<CStmt>) {
    let value = |path: &[String]| path.iter().fold(CExpr::var(id, var), |e, f| CExpr::field(id, e, f.clone()));
    match t {
        // a string keeps the block it lives in, a list of strings its block
        // and its chunks
        Ty::Str => out.push(CStmt::expr(id, CExpr::call(id, "viv_keep_str", vec![value(path)]))),
        Ty::Arr(Elem::Str) => out.push(CStmt::expr(id, CExpr::call(id, "viv_keep_strs", vec![value(path)]))),
        Ty::Arr(Elem::Func(_)) => {
            let data = CExpr::field(id, value(path), "data");
            out.push(CStmt::expr(id, CExpr::call(id, "viv_arena_keep", vec![CExpr::unary(id, "(const void*)", data)])));
        }
        Ty::Arr(_) => out.push(CStmt::expr(id, CExpr::call(id, "viv_arena_keep", vec![CExpr::field(id, value(path), "data")]))),
        // a map is filed under its tagged pointer (interp/maps.rs)
        Ty::Map(..) => {
            let key = if settle(t) == NUM_MAP { "VIV_MAP_KEY" } else { "VIV_SMAP_KEY" };
            out.push(CStmt::expr(id, CExpr::call(id, "viv_arena_keep", vec![CExpr::call(id, key, vec![value(path)])])))
        }
        Ty::Record(r) => {
            for (f, ft) in records.get(r).cloned().unwrap_or_default() {
                path.push(f);
                keep_calls(id, var, path, ft, records, mark, out);
                path.pop();
            }
        }
        // an enum value keeps every node it reaches
        Ty::Enum(n) => out.push(CStmt::expr(id, CExpr::call(id, format!("viv_keep_en_{}", enums::c_name(n)), vec![value(path), CExpr::var(id, mark)]))),
        // an array of records: its block and what each record holds
        Ty::Recs(r) => {
            let mut args = vec![value(path)];
            if holds_enum(t, records) {
                args.push(CExpr::var(id, mark));
            }
            out.push(CStmt::expr(id, CExpr::call(id, format!("viv_keep_recs_{r}"), args)))
        }
        _ => {}
    }
}

/// For a loop whose body can allocate, the variables whose arrays an
/// iteration keeps: those the body assigns (anything else it allocated is
/// garbage when the iteration ends, a variable's overwritten value included).
/// `None` for a body that cannot allocate, which pays nothing.
///
/// `cond` is a `while` loop's condition, which runs once a pass too: a
/// condition that allocates (`while sum(g(a)) < t`) makes the loop one that
/// allocates even when its body does not (study/memcheck/adversarial).
fn loop_keeps(cx: &mut CEmitCx<Viv>, body: NodeId, cond: Option<NodeId>) -> Option<Vec<(String, Ty)>> {
    let (vars, rets) = LOOP_SCOPE.with(|s| {
        let s = s.borrow();
        (s.vars.clone(), s.rets.clone())
    });
    if !c_allocates(cx, &rets, body) && !cond.is_some_and(|c| c_allocates(cx, &rets, c)) {
        return None;
    }
    cx.need("arena_keep");
    let mut names = Vec::new();
    c_assigned(cx, body, &mut names);
    Some(names.into_iter().filter_map(|n| vars.get(&n).map(|&t| (n, t))).collect())
}

thread_local! {
    /// Set by `viv export` (crate::export) while it compiles a program: the
    /// `__export_*` calls it writes are builtins only then.
    static EXPORTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Compile with the `__export_*` builtins on (`viv export`), or off.
pub fn set_exporting(on: bool) {
    EXPORTING.with(|e| e.set(on));
}

fn exporting() -> bool {
    EXPORTING.with(|e| e.get())
}

/// The type of an `__export_*` builtin: the parameter it reads, by kind.
fn export_builtin_ty(callee: &str) -> Option<Ty> {
    Some(match callee {
        "__export_given" | "__export_bool" => Ty::Bool,
        "__export_num" => Ty::Num,
        "__export_int" => Ty::Int,
        "__export_str" => Ty::Str,
        "__export_arr" => Ty::Arr(Elem::Num),
        "__export_iarr" => Ty::Arr(Elem::Int),
        _ => return None,
    })
}

/// The runtime of an exported program's parameters (crate::export).
const EXPORT_RUNTIME: &str = r#"/* An exported program's parameters (crate::export): the entry points these tables at the caller's values before
   each run, and the program reads a parameter where its deck assigns it:
   `ECUT = 10.0` is followed by `if __export_given(k) { ECUT = __export_num(j) }`. */
typedef struct { const void* data; int rank; long long dims[8]; } viv_export_arg;
static const unsigned char* viv_export_given_v = 0;
static const double* viv_export_num_v = 0;
static const long long* viv_export_int_v = 0;
static const char* const* viv_export_str_v = 0;
static const viv_export_arg* viv_export_arr_v = 0;
VIV_UNUSED static double viv_export_given(double k) { return viv_export_given_v[(int)k] ? 1.0 : 0.0; }
VIV_UNUSED static double viv_export_num(double k) { return viv_export_num_v[(int)k]; }
VIV_UNUSED static long long viv_export_int(double k) { return viv_export_int_v[(int)k]; }
VIV_UNUSED static double viv_export_bool(double k) { return viv_export_int_v[(int)k] ? 1.0 : 0.0; }"#;

/// A string parameter, as a string of the run's arena.
const EXPORT_STR_RUNTIME: &str = "static viv_str viv_export_str(double k) {\n    const char* s = viv_export_str_v[(int)k];\n    return viv_s_make(s, (long)strlen(s), 0);\n}";

/// An array parameter, copied into an array of the run's arena: numbers, or
/// Ints (eight bytes each too).
const EXPORT_ARR_RUNTIME: &str = "static viv_arr viv_export_arr(double k) {\n    const viv_export_arg* a = &viv_export_arr_v[(int)k];\n    int dims[8];\n    for (int i = 0; i < a->rank; i++) dims[i] = (int)a->dims[i];\n    viv_arr r = viv_arr_new(a->rank, dims);\n    if (r.size) memcpy(r.data, a->data, (size_t)r.size * sizeof(double));\n    return r;\n}";

/// Whether a loop polls for an interrupt at the top of each pass
/// (`VIV_POLL`, embed.c; nothing outside an exported library): every loop of
/// the top level, and every loop whose body holds another loop or a call, so
/// that a long run stops within one pass of a loop that does real work. An
/// innermost loop of plain arithmetic does not poll: the loop around it does,
/// and a load and branch per pass there would cost the hot loops the most.
/// Loops that allocate also poll where they release (`viv_arena_release_kept`).
fn poll_loop(cx: &CEmitCx<Viv>, body: NodeId) -> bool {
    fn heavy(cx: &CEmitCx<Viv>, id: NodeId) -> bool {
        cx.children(id).into_iter().any(|c| match cx.kind(c) {
            Viv::While | Viv::For { .. } | Viv::ParallelFor { .. } | Viv::ApplyFn { .. } => true,
            // a call of a def (the runtime's own calls are `viv_` names)
            Viv::Call { callee } if !callee.starts_with("viv_") => true,
            _ => heavy(cx, c),
        })
    }
    STEP_DEF.with(|d| *d.borrow() == crate::parser::MAIN_DEF) || heavy(cx, body)
}

fn poll_stmt(id: NodeId) -> CStmt {
    CStmt::expr(id, CExpr::raw(id, "VIV_POLL()"))
}

/// The statements that end an iteration: keep what `keeps` hold, free the
/// rest of what was filed on the arena since `viv_lmark`, which the loop's
/// block declares.
fn release_stmts(id: NodeId, keeps: &[(String, Ty)]) -> Vec<CStmt> {
    let records = LOOP_SCOPE.with(|s| s.borrow().records.clone());
    let mut out = Vec::new();
    for (n, t) in keeps {
        keep_calls(id, n, &mut Vec::new(), *t, &records, "viv_lmark", &mut out);
    }
    out.push(CStmt::expr(id, CExpr::call(id, "viv_arena_release_kept", vec![CExpr::var(id, "viv_lmark")])));
    // a pass that filed nothing on the arena has nothing to free: the keeps
    // and the scan are skipped, and only the poll the release makes is left
    // (measured: the skipped scan was a quarter of a hash group-by's time)
    let filed = CExpr::binary(id, CExpr::var(id, "viv_arena_top"), ">", CExpr::var(id, "viv_lmark"));
    vec![CStmt::if_(id, filed, out, vec![poll_stmt(id)])]
}

/// `{ size_t viv_lmark = viv_arena_top; <loop> <release> }` around a loop
/// whose every iteration starts with `release` (so a `continue` releases too),
/// and which releases once more when it ends.
fn loop_with_release(id: NodeId, lp: CStmt, release: Vec<CStmt>) -> CStmt {
    let mut stmts = vec![CStmt::decl(id, CType::Named("size_t".to_string()), "viv_lmark", CExpr::raw(id, "viv_arena_top"))];
    stmts.push(lp);
    stmts.extend(release);
    CStmt::block(id, stmts)
}

/// Whether a `continue` under `id` belongs to the loop `id` is the body of
/// (one inside a nested loop belongs to that loop).
fn own_continue(cx: &CEmitCx<Viv>, id: NodeId) -> bool {
    cx.children(id).into_iter().any(|c| match cx.kind(c) {
        Viv::Continue => true,
        Viv::While | Viv::For { .. } | Viv::ParallelFor { .. } => false,
        _ => own_continue(cx, c),
    })
}

/// The operators a record takes: + - * /, over its arrays.
pub(crate) const REC_OPERATORS: &str = "records take + - * / only, over their arrays";

/// The error of an operation over a record that is not a record of arrays.
pub(crate) fn not_arrays(word: &str, rec: &str, field: &str, what: &str) -> String {
    let a = if what.starts_with(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
    format!("{word} over a record acts on its arrays, and the field `{field}` of `{rec}` is {a} {what}")
}

/// The error of an operation on two arrays of records of different lengths.
pub(crate) fn rec_lengths_differ(word: &str, rec: &str) -> String {
    format!("{word} of two arrays of {rec} records needs them of one length")
}

/// The 1-argument math builtins that map over an array's elements, and so
/// over a record's arrays.
pub(crate) const REC_MAP_FNS: &[&str] =
    &["exp", "sqrt", "sin", "cos", "tan", "log", "ln", "abs", "fabs", "floor", "ceil", "round", "asin", "acos", "atan", "sinh", "cosh", "tanh", "erf", "gamma"];

/// `zeros_like(r)`, `sum(r)`, an elementwise math function of `r` and
/// `for_leaves(f, r1, ..., extra...)` with `r` a record or an array of
/// records: a call of the def `lower::records` writes. `None` when the call
/// is not one of these (its argument is not a record).
fn record_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    let is_rec = |t: Ty| matches!(t, Ty::Record(_) | Ty::Recs(_));
    if callee == "for_leaves" {
        return Some((|| {
            let [Expr::Var { name: f, span: fspan }, rest @ ..] = args else {
                return Err(Diagnostic::error("`for_leaves` takes a def first, then records of one type: `for_leaves(f, r1, r2, ...)`").with_primary(span, "no def named first"));
            };
            if !sigs.arity.contains_key(f.as_str()) || scope.get(f).is_some() {
                return Err(Diagnostic::error(format!("`for_leaves` calls a def by its name, and `{f}` is not a def of this program")).with_primary(*fspan, "not a def"));
            }
            let mut kids = Vec::with_capacity(rest.len());
            let mut tys = Vec::with_capacity(rest.len());
            for a in rest {
                let (id, t) = lower_expr(prog, a, scope, sigs)?;
                kids.push(id);
                tys.push((t, a.span()));
            }
            let k = tys.iter().take_while(|(t, _)| is_rec(*t)).count();
            let Some(&(ty, _)) = tys.first().filter(|_| k > 0) else {
                return Err(Diagnostic::error("`for_leaves` takes a def, then records of one type").with_primary(span, "no record after the def"));
            };
            for &(t, sp) in &tys[..k] {
                if t != ty {
                    return Err(Diagnostic::error(format!("`for_leaves` takes records of one type; the first is {} and this is {}", crate::lower::a_or_an(&ty.name()), crate::lower::a_or_an(&t.name()))).with_primary(sp, "a different type"));
                }
            }
            if sigs.arity.get(f.as_str()) != Some(&tys.len()) {
                return Err(Diagnostic::error(format!("`{f}` takes {} argument(s), and `for_leaves` passes it {}: an array of each of the {k} record(s), then the {} after them", sigs.arity[f.as_str()], tys.len(), tys.len() - k))
                    .with_primary(span, "the arguments `for_leaves` passes"));
            }
            let extra: Vec<Ty> = tys[k..].iter().map(|(t, _)| *t).collect();
            records::request(prog, sigs, records::RecFn::Leaves { f: f.clone(), ty, k, extra }, kids, span)
        })());
    }
    let one = callee == "zeros_like" || callee == "sum" || REC_MAP_FNS.contains(&callee);
    if !one || args.len() != 1 {
        return None;
    }
    let (id, ty) = match lower_expr(prog, &args[0], scope, sigs) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    if !is_rec(ty) {
        return None;
    }
    let rf = match callee {
        "zeros_like" => records::RecFn::Zeros { ty },
        "sum" => records::RecFn::Sum { ty },
        f => records::RecFn::Map { f: f.to_string(), ty },
    };
    Some(records::request(prog, sigs, rf, vec![id], span))
}

/// The variable a place is rooted at: `r` of `r`, `r.f` and `r.xs[i].g`.
pub(crate) fn place_root(e: &Expr) -> Option<(&str, Span)> {
    match e {
        Expr::Var { name, span } => Some((name, *span)),
        Expr::Field { base, .. } | Expr::Index { base, .. } => place_root(base),
        _ => None,
    }
}

/// `e` without the seal the growable pass may have wrapped around it.
fn unseal(e: &Expr) -> &Expr {
    match e {
        Expr::Call { callee, args, .. } if callee == crate::growable::SEAL && args.len() == 1 => unseal(&args[0]),
        e => e,
    }
}

/// Whether `e` reads its value out of a place (a variable, a field, an
/// element) rather than making a new one. A seal the growable pass wrapped
/// around it (`crate::growable`) reads what it wraps.
fn reads_place(e: &Expr) -> bool {
    match e {
        Expr::Var { .. } | Expr::Field { .. } | Expr::Index { .. } => true,
        Expr::Call { callee, args, .. } if callee == crate::growable::SEAL && args.len() == 1 => reads_place(&args[0]),
        _ => false,
    }
}

/// `vid`, the value of `e` of type `ty`, as it is stored somewhere new: a
/// record or an array of records read from a place becomes its own copy
/// (`Viv::Own`), so a write through one name is never seen through another.
/// In a program that writes into no record
/// and holds no array of records, sharing cannot be seen, and nothing is
/// added.
fn owned(prog: &mut IrProgram<Viv>, vid: NodeId, ty: Ty, e: &Expr, sigs: &Sigs) -> NodeId {
    if sigs.owned && matches!(ty, Ty::Record(_) | Ty::Recs(_)) && reads_place(e) {
        prog.add(Viv::Own { ty }, e.span(), vec![vid])
    } else {
        vid
    }
}

/// The rule for a write into a record or an array of records (`r.f = v`,
/// `xs[i] = r`): the place is rooted at a variable of the def it is written
/// in. A def's parameters are the caller's values and its globals are the
/// program's, and neither is changed by a def (a def's assignment to a
/// global's name makes a local, and records have no in-place parameters
/// yet). `what` is "a field of" or "an element of".
fn write_root(base: &Expr, scope: &Scope<'_>, what: &str) -> Result<(), Diagnostic> {
    let Some((root, rspan)) = place_root(base) else {
        return Err(Diagnostic::error(format!("can only assign into {what} a variable's record"))
            .with_primary(base.span(), "not a variable, a field or an element")
            .with_note("assign the value to a variable first, then change the variable"));
    };
    if !scope.in_function {
        return Ok(());
    }
    let local = scope.locals.iter().any(|(n, _)| n == root);
    if local {
        return Ok(());
    }
    if scope.bound.contains_key(root) {
        return Err(Diagnostic::error(format!("`{root}` is a parameter, and a def does not change the values it is given"))
            .with_primary(rspan, "a parameter")
            .with_note(format!("change a copy (`mine = {root}`) and return it; an array inside it can still be written in place (`{root}.w[i] = x`, `{root}.w[:] = v`)")));
    }
    Err(Diagnostic::error(format!("`{root}` is a top-level variable, and a def does not change those"))
        .with_primary(rspan, "a global")
        .with_note("pass it in and return the changed value, or change it at the top level"))
}

fn lower_cond(prog: &mut IrProgram<Viv>, cond: &Expr, scope: &mut Scope<'_>, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    let (cid, ty) = lower_expr(prog, cond, scope, sigs)?;
    if ty != Ty::Bool {
        let note = if ty == Ty::Arr(Elem::Num) { crate::hints::ARRAY_CONDITION } else { "use a comparison (e.g. `x < 0`) or `and`/`or`/`not`" };
        return Err(Diagnostic::error("condition must be a boolean")
            .with_primary(cond.span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))
            .with_note(note));
    }
    Ok(cid)
}

fn lower_expr(prog: &mut IrProgram<Viv>, expr: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let (id, ty) = lower_expr_inner(prog, expr, scope, sigs)?;
    if crate::builtins::heap_ty(ty) {
        scope.touches_arrays.set(true);
    }
    Ok((id, ty))
}

fn lower_expr_inner(prog: &mut IrProgram<Viv>, expr: &Expr, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    // Float32 and the packed arrays, in one call so this
    // recursive function's frame grows by one result, not one per form
    if let Some(r) = f32s::expr_hook(prog, expr, scope, sigs) {
        return r;
    }
    match expr {
        Expr::Num { value, span } => Ok((prog.add(Viv::Num(*value), *span, vec![]), Ty::Num)),
        Expr::Str { value, span } => Ok((prog.add(Viv::StrLit(value.clone()), *span, vec![]), Ty::Str)),
        Expr::Lambda { span, .. } => Err(Diagnostic::error("a lambda can be bound to a name, called, or passed to a def or to `simpson`/`trapezoid`/`deriv`; it cannot be returned or stored")
            .with_primary(*span, "a lambda used as a value")
            .with_note("a closure is resolved where it is used, so no function value outlives the def that makes it; return the values it needs, or write a def")),
        Expr::Var { name, span } => match scope.get(name) {
            Some(ty) => Ok((prog.add(Viv::Var { name: name.clone() }, *span, vec![]), ty)),
            None => {
                // Not a variable: a bare top-level `def` name used as a VALUE
                // (`g = sq`, `[r_bisect, ...]`). Emit a function value and make
                // sure its uniform-ABI wrapper is generated.
                if let Some(sig) = sigs.fn_sig(name) {
                    let wrap = prog.add(Viv::FnWrapDef { name: def_symbol(name), sig }, *span, vec![]);
                    // One wrapper per def, however many times the def is named
                    // as a value: the node is content-addressed, so the second
                    // mention lands on the same id and must not be rooted twice.
                    if !prog.roots().contains(&wrap) {
                        prog.add_root(wrap);
                    }
                    let r = prog.add(Viv::FnRef { name: def_symbol(name), sig }, *span, vec![]);
                    return Ok((r, Ty::Func(sig)));
                }
                Err(Diagnostic::error(format!("unknown variable `{}`", name))
                    .with_primary(*span, "not a parameter or local in scope"))
            }
        },
        Expr::ArrLit { elems, span } => lower_arr_lit(prog, elems, *span, scope, sigs),
        Expr::Index { base, indices, span } => {
            let (bid, bty) = lower_expr(prog, base, scope, sigs)?;
            if strings::is_text(bty) {
                return strings::lower_index(prog, bid, bty, indices, *span, scope, sigs);
            }
            // `m[k]` of a map: one key, a number or a string
            if let Ty::Map(mk, mv) = bty {
                let [k] = indices.as_slice() else {
                    return Err(Diagnostic::error(crate::interp::maps::ONE_KEY).with_primary(*span, format!("{} keys given", indices.len())));
                };
                let (kid, kty) = lower_expr(prog, k, scope, sigs)?;
                let (kid, kty) = ints::to_num(prog, kid, kty, k.span());
                map_kind(kty, mk, k.span(), "key", bty)?;
                return Ok((prog.add(Viv::Call { callee: crate::interp::maps::callee("get", mk, mv) }, *span, vec![bid, kid]), mk_ty(mv)));
            }
            // `xs[i]` of an array of records: one index, a record
            if let Ty::Recs(rec) = bty {
                let [i] = indices.as_slice() else {
                    return Err(Diagnostic::error("an array of records is a list, so it takes one index")
                        .with_primary(*span, format!("{} indices given", indices.len())));
                };
                let (iid, ity) = lower_expr(prog, i, scope, sigs)?;
                expect_index(ity, i.span())?;
                let id = prog.add(Viv::RecIndex { rec: rec.to_string(), at: site_at(*span) }, *span, vec![bid, iid]);
                return Ok((id, Ty::Record(rec)));
            }
            let Ty::Arr(elem) = bty else {
                if let Some(d) = unpack_refusal(base, bty) {
                    return Err(d);
                }
                return Err(Diagnostic::error("cannot index a non-array")
                    .with_primary(base.span(), format!("this is {}", crate::lower::a_or_an(&bty.name()))));
            };
            check_index_rank(indices.len(), *span)?;
            let mut child_ids = vec![bid];
            for idx in indices {
                let (id, ty) = lower_expr(prog, idx, scope, sigs)?;
                expect_index(ty, idx.span())?;
                child_ids.push(id);
            }
            // The element type picks the node: a complex array reads a complex,
            // and a function array reads a function VALUE of the array's shared
            // arity. That value is what `apply` then calls, which is the whole
            // point: `apply(methods[i], f, a, b)` is a call to a function the
            // compiler does not know, chosen at run time by an index.
            Ok(match elem {
                Elem::Num => (prog.add(Viv::Index { at: site_at(*span) }, *span, child_ids), Ty::Num),
                Elem::Cplx => (prog.add(Viv::CIndex { at: site_at(*span) }, *span, child_ids), Ty::Cplx),
                Elem::Func(sig) => {
                    if child_ids.len() != 2 {
                        return Err(Diagnostic::error("a function array is a list, so it takes one index")
                            .with_primary(*span, format!("{} indices given", child_ids.len() - 1)));
                    }
                    (prog.add(Viv::FnIndex { sig, at: site_at(*span) }, *span, child_ids), Ty::Func(sig))
                }
                Elem::Str => unreachable!("lowered above"),
                Elem::Int => {
                    if child_ids.len() != 2 {
                        return Err(Diagnostic::error("an Int array has rank 1, so it takes one index")
                            .with_primary(*span, format!("{} indices given", child_ids.len() - 1)));
                    }
                    (prog.add(Viv::Call { callee: "viv_iget".to_string() }, *span, child_ids), Ty::Int)
                }
                Elem::P(_) => unreachable!("lowered by f32s::expr_hook"),
            })
        }
        Expr::Imag { value, span } => Ok((prog.add(Viv::Imag(*value), *span, vec![]), Ty::Cplx)),
        Expr::Bool { value, span } => Ok((prog.add(Viv::BoolLit(*value), *span, vec![]), Ty::Bool)),
        Expr::Neg { operand, span } => {
            let (oid, ty) = lower_expr(prog, operand, scope, sigs)?;
            if ty == Ty::Int {
                return Ok((prog.add(Viv::Call { callee: "viv_i_neg".to_string() }, *span, vec![oid]), Ty::Int));
            }
            // each of a record's arrays negated (`lower::records`)
            if let Ty::Record(_) | Ty::Recs(_) = ty {
                return records::request(prog, sigs, records::RecFn::Neg { ty }, vec![oid], *span);
            }
            if ty == Ty::Cplx {
                return Ok((prog.add(Viv::Neg, *span, vec![oid]), Ty::Cplx));
            }
            // -a on a real array is the array scaled by -1, which in IEEE arithmetic
            // IS negation, bit for bit (signed zeros included). A complex scale by -1
            // is not (0 + 0i times -1 + 0i has imaginary part +0), so a complex
            // array is negated part by part.
            if let Ty::Arr(elem) = ty {
                if elem == Elem::Cplx {
                    return Ok((prog.add(Viv::Arr { kind: ArrKind::CNeg }, *span, vec![oid]), ty));
                }
                let minus_one = prog.add(Viv::Num(-1.0), *span, vec![]);
                return Ok((prog.add(Viv::Arr { kind: ArrKind::Scale }, *span, vec![oid, minus_one]), ty));
            }
            expect_num(ty, operand.span(), "the operand of `-`")?;
            Ok((prog.add(Viv::Neg, *span, vec![oid]), Ty::Num))
        }
        Expr::Not { operand, span } => {
            let (oid, ty) = lower_expr(prog, operand, scope, sigs)?;
            if ty != Ty::Bool {
                return Err(Diagnostic::error("`not` expects a boolean")
                    .with_primary(operand.span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
            }
            Ok((prog.add(Viv::Not, *span, vec![oid]), Ty::Bool))
        }
        Expr::Binary { op, lhs, rhs, span } => {
            // an Int on either side; a program with no
            // Int never takes this
            if let Some(r) = ints::lower_binary(prog, *op, lhs, rhs, *span, scope, sigs)? {
                return Ok(r);
            }
            let (lid, lty) = lower_expr(prog, lhs, scope, sigs)?;
            let (rid, rty) = lower_expr(prog, rhs, scope, sigs)?;
            if *op == BinOp::Add {
                list_join_trap(lhs, lty, rhs, rty, *span)?;
            }
            if matches!(lty, Ty::Enum(_)) || matches!(rty, Ty::Enum(_)) {
                return enums::lower_eq(prog, *op, (lid, lty, lhs.span()), (rid, rty, rhs.span()), *span);
            }
            // + - * / over a record's arrays (`lower::records`)
            if matches!(lty, Ty::Record(_) | Ty::Recs(_)) || matches!(rty, Ty::Record(_) | Ty::Recs(_)) {
                return records::binary(prog, sigs, *op, (lid, lty, lhs.span()), (rid, rty, rhs.span()), *span);
            }
            if (lty == Ty::Int || rty == Ty::Int) && !matches!(op, BinOp::And | BinOp::Or) {
                // an Int the look-ahead did not see: a literal beside it is an Int too
                let (lid, lty) = if lty == Ty::Num && rty == Ty::Int && ints::adapts(lhs, sigs) { (ints::lower_as_int(prog, lhs, scope, sigs)?, Ty::Int) } else { (lid, lty) };
                let (rid, rty) = if rty == Ty::Num && lty == Ty::Int && ints::adapts(rhs, sigs) { (ints::lower_as_int(prog, rhs, scope, sigs)?, Ty::Int) } else { (rid, rty) };
                return ints::combine(prog, *op, (lid, lty, lhs.span()), (rid, rty, rhs.span()), *span);
            }
            if strings::is_text(lty) || strings::is_text(rty) {
                return strings::lower_binop(prog, *op, (lid, lty, lhs.span()), (rid, rty, rhs.span()), *span);
            }
            lower_binop(prog, *op, (lid, lty, lhs.span()), (rid, rty, rhs.span()), *span)
        }
        Expr::Call { callee, args, span } => {
            // A program's own def wins over a builtin of the same name, in every
            // engine: the interpreter tries user defs first, so the builtins
            // below are only for names the program does not define. (Adding a
            // builtin never changes what an existing program computes.)
            'builtin: {
            if sigs.arity.contains_key(callee.as_str()) {
                break 'builtin;
            }
            // over a record's arrays (`lower::records`)
            if let Some(r) = record_call(prog, callee, args, *span, scope, sigs) {
                return r;
            }
            // enum values and the calls a `match` is written with
            if let Some(r) = enums::lower_call(prog, callee, args, *span, scope, sigs) {
                return r;
            }
            // the Int forms of the builtins: a program
            // with no Int takes none of these
            if callee == ints::NUM_EXACT {
                return ints::lower_num_exact(prog, args, *span, scope, sigs);
            }
            if let Some(r) = ints::lower_call(prog, callee, args, *span, scope, sigs) {
                return r;
            }
            // (an append's Int is its array's business: an Int array takes it as is)
            if !NATIVE_INT.contains(&callee.as_str()) && !strings::takes_ints(callee) && !crate::growable::GROW.contains(&callee.as_str()) {
                if let Some(args) = ints::promote_args(callee, args, *span, scope, sigs) {
                    return lower_expr_inner(prog, &Expr::Call { callee: callee.clone(), args, span: *span }, scope, sigs);
                }
            }
            // strings: the calls that take or make them
            // `sorted`, `unique` and `del` of a real array (or `del` of a list
            // of strings): the defs crate::stdlib added
            if let Some(form) = crate::stdlib::number_form(callee).filter(|_| !sigs.arity.contains_key(callee.as_str()) && !args.is_empty()) {
                let form = match lower_expr(&mut IrProgram::new(), &args[0], scope, sigs).ok().map(|(_, t)| t) {
                    Some(Ty::Arr(Elem::Num)) => Some(form),
                    Some(Ty::Arr(Elem::Str)) if callee == crate::stdlib::DEL => Some("viv__del_strs"),
                    _ => None,
                };
                if let Some(form) = form.filter(|f| sigs.arity.contains_key(*f)) {
                    return lower_expr_inner(prog, &Expr::Call { callee: form.to_string(), args: args.clone(), span: *span }, scope, sigs);
                }
            }
            if let Some(r) = strings::lower_call(prog, callee, args, *span, scope, sigs) {
                return r;
            }
            // the array vocabulary (interp/arrays.rs)
            if let Some(sig) = crate::interp::arrays::sig(callee, args.len()) {
                return lower_array_call(prog, sig, args, *span, scope, sigs);
            }
            // apply(f, args...): call a function VALUE. The first argument must
            // be a function; the rest are numbers. Lowers to an indirect call
            // through the uniform viv_fn ABI. Result is a number (Stage 1).
            if callee == "apply" {
                if args.is_empty() {
                    return Err(Diagnostic::error("`apply` takes a function and its arguments")
                        .with_primary(*span, "expected at least the function"));
                }
                let (fid, fty) = lower_expr(prog, &args[0], scope, sigs)?;
                let sig = match fty {
                    Ty::Func(s) => s,
                    _ => return Err(Diagnostic::error("`apply`'s first argument must be a function")
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&fty.name())))),
                };
                // viv_apply copies its arguments into a fixed viv_arg[16], and a
                // signature records the kinds of 16 parameters; a 17th argument
                // would be read from past the end and come back as garbage.
                if args.len() - 1 > APPLY_MAX_ARGS {
                    return Err(Diagnostic::error(format!(
                        "`apply` passes at most {APPLY_MAX_ARGS} arguments; this call passes {}", args.len() - 1))
                        .with_primary(*span, format!("more than {APPLY_MAX_ARGS} arguments -- pack some of them into an array")));
                }
                if args.len() - 1 != sig.arity as usize {
                    return Err(Diagnostic::error(format!(
                        "this function takes {} argument(s), got {}", sig.arity, args.len() - 1))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let mut child_ids = vec![fid];
                for (j, a) in args[1..].iter().enumerate() {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    // Every argument travels through one union slot that the
                    // wrapper reads by the callee's signature, so the kind given
                    // here must be the kind expected there -- a number read as a
                    // function pointer is not a type error at run time, it is a crash.
                    let want = sig.kind(j);
                    let given = ArgKind::of(ty).ok_or_else(|| {
                        Diagnostic::error(format!("a {} cannot be passed through `apply`", ty.name()))
                            .with_primary(a.span(), "only numbers, real arrays and functions can")
                    })?;
                    if given != want {
                        let name = |k: ArgKind| match k {
                            ArgKind::Num => "a number",
                            ArgKind::Fn => "a function",
                            ArgKind::Arr => "an array",
                        };
                        return Err(Diagnostic::error(format!(
                            "argument {} should be {} but is {}", j + 1, name(want), name(given)))
                            .with_primary(a.span(), "wrong kind of argument"));
                    }
                    child_ids.push(id);
                }
                return Ok((prog.add(Viv::ApplyFn { sig }, *span, child_ids), sig.ret_ty()));
            }
            if callee == "bars" {
                if args.len() < 2 || args.len() > 3 {
                    return Err(Diagnostic::error("`bars` takes (x, y) or (x, y, \"label\")").with_primary(*span, "wrong arguments"));
                }
                let label = match args.get(2) {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => return Err(Diagnostic::error("`bars`'s third argument must be a string label").with_primary(other.span(), "not a string literal")),
                    None => String::new(),
                };
                let mut ids = Vec::with_capacity(2);
                for a in &args[..2] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Arr(Elem::Num) {
                        return Err(Diagnostic::error("`bars` takes arrays for x and y").with_primary(a.span(), "not an array"));
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Bars { label }, *span, ids), Ty::Num));
            }
            // histogram(a, lo, hi, nbins): counts of a's entries in nbins equal bins over
            // [lo, hi]; the last bin closed on the right, entries outside not counted.
            // all(m) / any(m): a Bool from a real array (nonzero is true), and
            // where(m, a, b): a from where m is nonzero, b elsewhere; a and b are
            // each an array of m's shape or a number. A program's own def of
            // the same name wins, as in the interpreter.
            if matches!(callee.as_str(), "all" | "any") && !sigs.arity.contains_key(callee.as_str()) {
                if args.len() != 1 {
                    return Err(Diagnostic::error(format!("`{callee}` takes one array")).with_primary(*span, "expected 1 argument"));
                }
                let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if ty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error(format!("`{callee}` takes a real array (a mask such as `a > 0`)"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                }
                return Ok((prog.add(Viv::Call { callee: format!("viv_{callee}") }, *span, vec![id]), Ty::Bool));
            }
            if callee.starts_with("__ad_") && !sigs.arity.contains_key(callee.as_str()) {
                return lower_ad(prog, callee, args, *span, scope, sigs);
            }
            if callee == "where" && !sigs.arity.contains_key("where") {
                if args.len() != 3 {
                    return Err(Diagnostic::error("`where` takes (mask, a, b)").with_primary(*span, "expected 3 arguments"));
                }
                let (mid, mty) = lower_expr(prog, &args[0], scope, sigs)?;
                if mty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error("`where` takes a mask array first: `where(m, a, b)`")
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&mty.name()))));
                }
                let mut ids = vec![mid];
                let mut form = String::from("viv_where_");
                for a in &args[1..] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    form.push(match ty {
                        Ty::Num => 's',
                        t if t == Ty::Arr(Elem::Num) => 'a',
                        t => {
                            return Err(Diagnostic::error("`where` takes a number or a real array for each choice")
                                .with_primary(a.span(), format!("this is {}", crate::lower::a_or_an(&t.name()))))
                        }
                    });
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: form }, *span, ids), Ty::Arr(Elem::Num)));
            }
            if callee == "histogram" {
                if args.len() != 4 {
                    return Err(Diagnostic::error("`histogram` takes (values, lo, hi, nbins)").with_primary(*span, "expected 4 arguments"));
                }
                let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
                if aty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error("`histogram` counts a real array").with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&aty.name()))));
                }
                let mut ids = vec![aid];
                for a in &args[1..] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "a histogram argument")?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_histogram".to_string() }, *span, ids), Ty::Arr(Elem::Num)));
            }
            if callee == "plot" || callee == "scatter" {
                let max_args = if callee == "scatter" { 4 } else { 3 };
                if args.len() < 2 || args.len() > max_args {
                    return Err(Diagnostic::error("`plot` takes (x, y) or (x, y, \"label\"); `scatter` allows a fourth color-name string")
                        .with_primary(*span, "wrong arguments"));
                }
                let label = match args.get(2) {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => {
                        return Err(Diagnostic::error("`plot`'s third argument must be a string label")
                            .with_primary(other.span(), "not a string literal"));
                    }
                    None => String::new(),
                };
                let mut ids = Vec::with_capacity(2);
                for a in &args[..2] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Arr(Elem::Num) {
                        return Err(Diagnostic::error("`plot` takes arrays for x and y")
                            .with_primary(a.span(), "not an array"));
                    }
                    ids.push(id);
                }
                let color = match args.get(3) {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => {
                        return Err(Diagnostic::error("`scatter`'s fourth argument must be a color name string")
                            .with_primary(other.span(), "not a string literal"));
                    }
                    None => String::new(),
                };
                let node = if callee == "scatter" { Viv::Scatter { label, color } } else { Viv::Plot { label } };
                return Ok((prog.add(node, *span, ids), Ty::Num));
            }
            if callee == "checkpoint" {
                if !args.is_empty() {
                    return Err(Diagnostic::error("`checkpoint` takes no arguments")
                        .with_primary(*span, "unexpected arguments"));
                }
                return Ok((prog.add(Viv::Ckpt, *span, vec![]), Ty::Num));
            }
            if callee == "figure" {
                if args.len() != 3 && args.len() != 7 && args.len() != 8 {
                    return Err(Diagnostic::error("`figure` takes (title, xlabel, ylabel[, xlo, xhi, ylo, yhi[, style]])")
                        .with_primary(*span, "wrong arguments"));
                }
                // a title or a label that is a string value
                if args[..3].iter().any(|a| !matches!(a, Expr::Str { .. })) {
                    return strings::lower_figure(prog, args, *span, scope, sigs);
                }
                let mut strs = Vec::with_capacity(3);
                for a in &args[..3] {
                    match a {
                        Expr::Str { value, .. } => strs.push(value.clone()),
                        other => {
                            return Err(Diagnostic::error("`figure`'s first three arguments must be string literals: title, xlabel, ylabel")
                                .with_primary(other.span(), "not a string literal"));
                        }
                    }
                }
                let style = if args.len() == 8 {
                    match &args[7] {
                        Expr::Str { value, .. } => value.clone(),
                        other => {
                            return Err(Diagnostic::error("`figure`'s eighth argument must be a style string literal (e.g. \"contour\")")
                                .with_primary(other.span(), "not a string literal"));
                        }
                    }
                } else {
                    String::new()
                };
                let mut ids = Vec::new();
                for a in args.iter().skip(3).take(4) {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Num {
                        return Err(Diagnostic::error("`figure`'s window arguments must be numbers: xlo, xhi, ylo, yhi")
                            .with_primary(a.span(), "not a number"));
                    }
                    ids.push(id);
                }
                let node = Viv::Figure {
                    title: strs[0].clone(),
                    xlabel: strs[1].clone(),
                    ylabel: strs[2].clone(),
                    style,
                };
                return Ok((prog.add(node, *span, ids), Ty::Num));
            }
            if callee == "quiver" {
                if args.len() != 4 && args.len() != 5 {
                    return Err(Diagnostic::error("`quiver` takes (x, y, u, v) or (x, y, u, v, \"label\")")
                        .with_primary(*span, "wrong arguments"));
                }
                let label = match args.get(4) {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => {
                        return Err(Diagnostic::error("`quiver`'s fifth argument must be a string label")
                            .with_primary(other.span(), "not a string literal"));
                    }
                    None => String::new(),
                };
                let mut ids = Vec::with_capacity(4);
                for a in &args[..4] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Arr(Elem::Num) {
                        return Err(Diagnostic::error("`quiver` takes arrays for x, y, u, v")
                            .with_primary(a.span(), "not an array"));
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Quiver { label }, *span, ids), Ty::Num));
            }
            if callee == "plot2d" {
                if args.is_empty() || args.len() > 2 {
                    return Err(Diagnostic::error("`plot2d` takes (Z) or (Z, \"label\")")
                        .with_primary(*span, "expected a matrix and an optional string label"));
                }
                let label = match args.get(1) {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => {
                        return Err(Diagnostic::error("`plot2d`'s second argument must be a string label")
                            .with_primary(other.span(), "not a string literal"));
                    }
                    None => String::new(),
                };
                let (zid, zty) = lower_expr(prog, &args[0], scope, sigs)?;
                if zty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error("`plot2d` takes a 2-D array")
                        .with_primary(args[0].span(), "not an array"));
                }
                return Ok((prog.add(Viv::Plot2d { label }, *span, vec![zid]), Ty::Num));
            }
            if callee == "plot3d" {
                if args.len() != 3 && args.len() != 4 {
                    return Err(Diagnostic::error("`plot3d` takes (x, y, z) or (x, y, z, \"label\")")
                        .with_primary(*span, "expected 3 arrays and an optional string label"));
                }
                let label = match args.get(3) {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => {
                        return Err(Diagnostic::error("`plot3d`'s fourth argument must be a string label")
                            .with_primary(other.span(), "not a string literal"));
                    }
                    None => String::new(),
                };
                let mut ids = Vec::with_capacity(3);
                for a in &args[..3] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Arr(Elem::Num) {
                        return Err(Diagnostic::error("`plot3d` takes arrays for x, y, z")
                            .with_primary(a.span(), "not an array"));
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Plot3d { label }, *span, ids), Ty::Num));
            }
            // The animation verbs: string literals, then arrays, then numbers.
            // (A program's own `def draw(...)` wins: user functions are tried first.)
            if let Some(v) = anim_verb(callee).filter(|_| !sigs.arity.contains_key(callee.as_str())) {
                let fixed = v.texts + v.arrays + v.nums;
                if args.len() < fixed || args.len() > fixed + v.opt {
                    return Err(Diagnostic::error(format!("`{callee}` is written {}", v.usage))
                        .with_primary(*span, "wrong number of arguments"));
                }
                // text that is a string value
                if args[..v.texts].iter().any(|a| !matches!(a, Expr::Str { .. })) {
                    return strings::lower_anim(prog, v, args, *span, scope, sigs);
                }
                let mut texts = Vec::new();
                for (k, a) in args[..v.texts].iter().enumerate() {
                    let Expr::Str { value, .. } = a else {
                        return Err(Diagnostic::error(format!("`{callee}` is written {}", v.usage))
                            .with_primary(a.span(), "this must be a string literal"));
                    };
                    if value.contains('\n') || value.contains(" ; ") {
                        return Err(Diagnostic::error("text in an animation cannot contain a newline or ` ; `")
                            .with_primary(a.span(), "the record format uses these as separators"));
                    }
                    if let Err(m) = anim_text_check(callee, k, value) {
                        return Err(Diagnostic::error(m).with_primary(a.span(), "not one of the words it takes"));
                    }
                    texts.push(value.clone());
                }
                let mut ids = Vec::new();
                for (k, a) in args[v.texts..].iter().enumerate() {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if k < v.arrays {
                        if ty != Ty::Arr(Elem::Num) {
                            return Err(Diagnostic::error(format!("`{callee}` is written {}", v.usage))
                                .with_primary(a.span(), format!("this must be a real array, and is {}", crate::lower::a_or_an(&ty.name()))));
                        }
                    } else {
                        expect_num(ty, a.span(), &format!("`{callee}`'s argument"))?;
                    }
                    ids.push(id);
                }
                let node = Viv::Anim { head: anim_head(callee, &texts), arrays: v.arrays as u8 };
                return Ok((prog.add(node, *span, ids), Ty::Num));
            }
            if callee == "scene" {
                if args.len() != 1 {
                    return Err(Diagnostic::error("`scene` takes one string argument: the title")
                        .with_primary(*span, "expected scene(\"title\")"));
                }
                let Some(Expr::Str { value, .. }) = args.first() else {
                    return Err(Diagnostic::error("`scene`'s argument must be a string literal")
                        .with_primary(args[0].span(), "not a string literal"));
                };
                return Ok((prog.add(Viv::Scene { title: value.clone() }, *span, vec![]), Ty::Num));
            }
            if callee == "wire" || callee == "verts" {
                if args.len() != 4 && args.len() != 5 {
                    return Err(Diagnostic::error(format!(
                        "`{callee}` takes (x, y, z, \"group\") or (x, y, z, values, \"group\")"))
                        .with_primary(*span, "expected 3 or 4 arrays and a string group name"));
                }
                let group = match args.last() {
                    Some(Expr::Str { value, .. }) => value.clone(),
                    Some(other) => {
                        return Err(Diagnostic::error(format!("`{callee}`'s last argument must be a string group name"))
                            .with_primary(other.span(), "not a string literal"));
                    }
                    None => unreachable!(),
                };
                let n_arr = args.len() - 1;
                let mut ids = Vec::with_capacity(n_arr);
                for a in &args[..n_arr] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Arr(Elem::Num) {
                        return Err(Diagnostic::error(format!("`{callee}` takes arrays for x, y, z and values"))
                            .with_primary(a.span(), "not an array"));
                    }
                    ids.push(id);
                }
                let node = if callee == "wire" { Viv::Wire { group } } else { Viv::Verts { group } };
                return Ok((prog.add(node, *span, ids), Ty::Num));
            }
            if callee == "text" {
                if args.len() != 1 {
                    return Err(Diagnostic::error(format!("`text` expects 1 argument, got {}", args.len()))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let (aid, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if !matches!(ty, Ty::Arr(Elem::Num)) {
                    return Err(Diagnostic::error("`text` expects an array of character codes")
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                }
                return Ok((prog.add(Viv::Call { callee: "viv_text".to_string() }, *span, vec![aid]), Ty::Num));
            }
            if callee == "print" {
                // text and numbers on one line (crate::printfmt): a string
                // literal is text, `fmt(x, ...)` a value in a format, anything
                // else a value as `show` writes it
                let fmt_is_builtin = !sigs.arity.contains_key("fmt");
                let mut parts = Vec::with_capacity(args.len());
                let mut ids = Vec::new();
                for a in args {
                    let (x, f, vf, two) = match a {
                        Expr::Str { value, .. } => {
                            parts.push(PrintPart::Text(value.clone()));
                            continue;
                        }
                        Expr::Call { callee, args: fa, span } if callee == "fmt" && fmt_is_builtin => {
                            // `fmt` of a string or a list of strings is text of
                            // its own
                            // so is a number in a format whose width or precision
                            // is a value (`fmt(x, ".*f", p)`)
                            let starred = matches!(fa.get(1), Some(Expr::Str { value, .. }) if crate::printfmt::Fmt::parse_spec(value).is_ok_and(|f| f.stars() > 0));
                            if starred || fa.first().and_then(|x| strings::peek_text(x, scope, sigs)).is_some() {
                                let (id, _) = lower_expr(prog, a, scope, sigs)?;
                                parts.push(PrintPart::Str);
                                ids.push(id);
                                continue;
                            }
                            let (f, vf) = crate::printfmt::Fmt::of_map_args(fa, *span)
                                .map_err(|(at, m, label)| Diagnostic::error(m).with_primary(at, label))?;
                            (&fa[0], f, vf, fa.len() == 3)
                        }
                        e => (e, crate::printfmt::Fmt::BARE, crate::printfmt::Fmt::BARE, false),
                    };
                    let (id, ty) = lower_expr(prog, x, scope, sigs)?;
                    let bare = f.kind == crate::printfmt::Kind::Bare;
                    // an Int: its digits, or, in a float format, the number
                    // it is
                    let (id, ty, int_text) = match ty {
                        Ty::Int => {
                            let (t, tt) = ints::text_or_num(prog, id, f, x.span())?;
                            (t, tt, tt == Ty::Str)
                        }
                        Ty::Arr(Elem::Int) if bare => (prog.add(Viv::Call { callee: "viv_iarr_str".to_string() }, x.span(), vec![id]), Ty::Str, true),
                        Ty::Enum(_) if bare => (enums::text(prog, id, ty, x.span()), Ty::Str, true),
                        _ => (id, ty, false),
                    };
                    if int_text {
                        ids.push(id);
                        parts.push(PrintPart::Str);
                        continue;
                    }
                    if strings::is_text(ty) && bare {
                        ids.push(strings::print_part(prog, id, ty, x.span()));
                        parts.push(PrintPart::Str);
                        continue;
                    }
                    if two && !matches!(ty, Ty::Map(..)) {
                        return Err(Diagnostic::error(crate::interp::maps::TWO_FORMATS).with_primary(x.span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                    }
                    if let Ty::Map(mk, mv) = ty {
                        // a string key or value is written as its repr
                        for (kind, fm, which) in [(mk, f, "keys"), (mv, vf, "values")] {
                            if kind == Mk::Str && !matches!(fm.kind, crate::printfmt::Kind::Bare | crate::printfmt::Kind::Repr | crate::printfmt::Kind::Str) {
                                return Err(Diagnostic::error(format!("this map's {which} are strings, which `fmt` writes as their repr"))
                                    .with_primary(x.span(), format!("this is {}", crate::lower::a_or_an(&ty.name())))
                                    .with_note(format!("give \"\" as the format of its {which}")));
                            }
                        }
                    }
                    parts.push(match ty {
                        Ty::Map(..) => map_part(ty, f, vf),
                        Ty::Num => PrintPart::Num(f),
                        Ty::Arr(Elem::Num) => PrintPart::Arr(f),
                        Ty::Bool if bare => PrintPart::Bool,
                        Ty::Cplx if bare => PrintPart::Cplx,
                        Ty::Bool | Ty::Cplx => {
                            return Err(Diagnostic::error("`fmt` takes a real number or a real array")
                                .with_primary(x.span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))))
                        }
                        _ => {
                            return Err(Diagnostic::error(format!("`print` cannot write a {}", ty.name()))
                                .with_primary(x.span(), "print writes text, numbers, booleans, complex numbers and real arrays"))
                        }
                    });
                    ids.push(id);
                }
                // one bare number is the one-argument `print` it always was
                if let [PrintPart::Num(f)] = parts.as_slice() {
                    if f.kind == crate::printfmt::Kind::Bare {
                        return Ok((prog.add(Viv::Call { callee: PRINT_RUNTIME.to_string() }, *span, ids), Ty::Num));
                    }
                }
                return Ok((prog.add(Viv::Print { parts }, *span, ids), Ty::Num));
            }
            if callee == "czeros" {
                if args.is_empty() || args.len() > 4 {
                    return Err(Diagnostic::error("`czeros` takes 1 to 4 dimensions").with_primary(*span, "wrong number of dimensions"));
                }
                let mut dims = Vec::with_capacity(args.len());
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "a dimension")?;
                    dims.push(id);
                }
                return Ok((prog.add(Viv::CZeros, *span, dims), Ty::Arr(Elem::Cplx)));
            }
            if callee == "fft" || callee == "ifft" {
                if args.len() != 1 {
                    return Err(Diagnostic::error(format!("`{callee}` takes one complex array")).with_primary(*span, "wrong argument count"));
                }
                let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if ty != Ty::Arr(Elem::Cplx) {
                    // the same words the interpreter's run-time error uses
                    let what = if ty == Ty::Arr(Elem::Num) { "this one is real".to_string() } else { format!("this is {}", crate::lower::a_or_an(&ty.name())) };
                    let mut d = Diagnostic::error(format!("`{callee}` needs a complex array")).with_primary(args[0].span(), what);
                    if ty == Ty::Arr(Elem::Num) {
                        d = d.with_note("allocate with `czeros` and copy the samples in");
                    }
                    return Err(d);
                }
                return Ok((prog.add(Viv::Fft { inverse: callee == "ifft" }, *span, vec![id]), Ty::Num));
            }
            // re / im / abs over a complex ARRAY give a real array of the same
            // shape (a spectrum is |X|); conj gives another complex one.
            if matches!(callee.as_str(), "re" | "im" | "abs" | "fabs" | "conj") && args.len() == 1 {
                let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if ty == Ty::Arr(Elem::Cplx) {
                    let name = if callee == "fabs" { "abs" } else { callee.as_str() };
                    let out = if name == "conj" { Elem::Cplx } else { Elem::Num };
                    return Ok((prog.add(Viv::CArrFn { name: name.to_string() }, *span, vec![id]), Ty::Arr(out)));
                }
            }
            if callee == "zeros" {
                if args.is_empty() {
                    return Err(Diagnostic::error("`zeros` needs at least one dimension").with_primary(*span, "no dimensions"));
                }
                if args.len() > MAX_RANK {
                    return Err(Diagnostic::error(format!("`zeros` takes 1 to {MAX_RANK} dimensions; this call gives {}", args.len()))
                        .with_primary(*span, format!("arrays have rank {MAX_RANK} at most")));
                }
                let mut dims = Vec::with_capacity(args.len());
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "a dimension")?;
                    dims.push(id);
                }
                return Ok((prog.add(Viv::Zeros, *span, dims), Ty::Arr(Elem::Num)));
            }
            if callee == "linspace" {
                if args.len() != 3 {
                    return Err(Diagnostic::error("`linspace` takes (lo, hi, n)").with_primary(*span, "expected 3 arguments"));
                }
                let mut ids = Vec::with_capacity(3);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "a linspace argument")?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Arr { kind: ArrKind::Linspace }, *span, ids), Ty::Arr(Elem::Num)));
            }
            if callee == "transpose" {
                if args.len() != 1 {
                    return Err(Diagnostic::error("`transpose` takes 1 array").with_primary(*span, "wrong number of arguments"));
                }
                let (aid, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if ty == Ty::Arr(Elem::Cplx) {
                    return Ok((prog.add(Viv::Arr { kind: ArrKind::CTranspose }, *span, vec![aid]), Ty::Arr(Elem::Cplx)));
                }
                expect_arr(ty, args[0].span())?;
                return Ok((prog.add(Viv::Arr { kind: ArrKind::Transpose }, *span, vec![aid]), Ty::Arr(Elem::Num)));
            }
            // slab(A, k) / take(A, idx): copies out of an array of either element
            // type; the result keeps A's element type. One C function serves both
            // because the element width is an argument.
            if callee == "slab" || callee == "take" {
                if args.len() != 2 {
                    let sig = if callee == "slab" { "(array, index)" } else { "(array, indices)" };
                    return Err(Diagnostic::error(format!("`{callee}` takes {sig}")).with_primary(*span, "wrong number of arguments"));
                }
                let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
                let Ty::Arr(elem) = aty else {
                    return Err(Diagnostic::error(format!("`{callee}` takes an array first"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&aty.name()))));
                };
                let (kid, kty) = lower_expr(prog, &args[1], scope, sigs)?;
                if callee == "slab" {
                    expect_num(kty, args[1].span(), "`slab`'s index")?;
                } else if kty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error("`take`: the indices must be a real array")
                        .with_primary(args[1].span(), format!("this is {}", crate::lower::a_or_an(&kty.name()))));
                }
                let width = prog.add(Viv::Num(if elem == Elem::Cplx { 2.0 } else { 1.0 }), *span, vec![]);
                return Ok((prog.add(Viv::Call { callee: format!("viv_{callee}") }, *span, vec![aid, kid, width]), Ty::Arr(elem)));
            }
            // setslab(A, k, S) / put(A, idx, V): the in-place inverses.
            if callee == "setslab" || callee == "put" {
                if args.len() != 3 {
                    let sig = if callee == "setslab" { "(array, index, slab)" } else { "(array, indices, values)" };
                    return Err(Diagnostic::error(format!("`{callee}` takes {sig}")).with_primary(*span, "wrong number of arguments"));
                }
                let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
                let Ty::Arr(elem) = aty else {
                    return Err(Diagnostic::error(format!("`{callee}` takes an array first"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&aty.name()))));
                };
                let (kid, kty) = lower_expr(prog, &args[1], scope, sigs)?;
                if callee == "setslab" {
                    expect_num(kty, args[1].span(), "`setslab`'s index")?;
                } else if kty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error("`put`: the indices must be a real array")
                        .with_primary(args[1].span(), format!("this is {}", crate::lower::a_or_an(&kty.name()))));
                }
                let (vid, vty) = lower_expr(prog, &args[2], scope, sigs)?;
                if vty != aty {
                    return Err(Diagnostic::error(format!("`{callee}`: both arrays must be real, or both complex"))
                        .with_primary(args[2].span(), format!("this is {}, the target is {}", crate::lower::a_or_an(&vty.name()), crate::lower::a_or_an(&aty.name()))));
                }
                let width = prog.add(Viv::Num(if elem == Elem::Cplx { 2.0 } else { 1.0 }), *span, vec![]);
                return Ok((prog.add(Viv::Call { callee: format!("viv_{callee}") }, *span, vec![aid, kid, vid, width]), Ty::Num));
            }
            // Slices, copy and the joins (interp/slices.rs): copies, of either
            // element type; the C functions take the element width, and those
            // that can stop take their site (added when the C is emitted).
            if callee == SLICE_GET || callee == SLICE_SET || callee == crate::fuse::CHECK || callee == crate::fuse::VIEW || matches!(callee.as_str(), "copy" | "concat" | "hcat" | "vcat") {
                return lower_slice_call(prog, callee, args, *span, scope, sigs);
            }
            // maps (interp/maps.rs; a program's own def of the name wins)
            if crate::interp::maps::NAMES.contains(&callee.as_str()) && !sigs.arity.contains_key(callee.as_str()) {
                return lower_map_call(prog, callee, args, *span, scope, sigs);
            }
            // growable arrays (crate::growable, interp/growable.rs)
            if callee == crate::growable::SEAL || crate::growable::GROW.contains(&callee.as_str()) {
                return lower_grow_call(prog, callee, args, *span, scope, sigs);
            }
            // simpson/trapezoid: numerical integration of a closure over [a, b].
            // The first argument must be a `fn(x) -> ...` literal; it is lifted to
            // a top-level function and the captured locals become its closure env.
            if callee == "simpson" || callee == "trapezoid" {
                if args.len() != 4 {
                    return Err(Diagnostic::error(format!("`{callee}` takes (f, a, b, n)"))
                        .with_primary(*span, "expected 4 arguments"));
                }
                let Expr::Lambda { params, body, span: lspan } = &args[0] else {
                    return Err(Diagnostic::error(format!("the first argument of `{callee}` must be a `fn(x) -> ...`"))
                        .with_primary(args[0].span(), "expected a closure"));
                };
                if params.len() != 1 {
                    return Err(Diagnostic::error("this closure must take exactly 1 argument")
                        .with_primary(*lspan, "wrong arity"));
                }
                let lam_param = params[0].clone();
                let mut free = Vec::new();
                collect_free_vars(body, &lam_param, &mut free);
                let mut captures: Vec<(String, Ty)> = Vec::new();
                for fv in &free {
                    if scope.is_local(fv) {
                        let ty = scope.get(fv).unwrap();
                        if ty != Ty::Num {
                            return Err(Diagnostic::error("a closure can only capture numbers for now")
                                .with_primary(*lspan, format!("`{fv}` is {}", crate::lower::a_or_an(&ty.name()))));
                        }
                        captures.push((fv.clone(), Ty::Num));
                    }
                }
                let mut lam_params = vec![(lam_param.clone(), Ty::Num)];
                lam_params.extend(captures.iter().cloned());
                let lam_scope = Scope::function(&lam_params, &sigs.globals);
                let (body_id, body_ty) = lower_expr(prog, body, &lam_scope, sigs)?;
                expect_num(body_ty, body.span(), "a closure body")?;
                // The lifted function is named after WHAT IT IS -- its body's
                // content address, with its parameter and captures folded in --
                // and not after where it sits in the file. It used to be
                // `viv_lam_<byte offset>`, and since that name is hashed into the
                // enclosing function, inserting an unrelated line above `boys_F0`
                // changed `boys_F0`'s content address. (Found by the H2 ladder,
                // whose whole point is that an untouched function keeps its address.)
                let mut h = Hasher::new();
                h.update(body_id.as_bytes());
                h.update(lam_param.as_bytes());
                for (c, _) in &captures {
                    h.update(&[0]);
                    h.update(c.as_bytes());
                }
                let digest = h.finalize();
                let name = format!("viv_lam_{}", digest.to_hex().chars().take(8).collect::<String>());
                let lamdef = prog.add(
                    Viv::LambdaDef { name: name.clone(), param: lam_param, captures: captures.clone() },
                    *lspan,
                    vec![body_id],
                );
                // the same closure written twice is one function
                if !prog.roots().contains(&lamdef) {
                    prog.add_root(lamdef);
                }
                let cap_names: Vec<String> = captures.iter().map(|(n, _)| n.clone()).collect();
                let clos = prog.add(Viv::Closure { name, captures: cap_names }, *lspan, vec![]);
                let mut child_ids = vec![clos];
                for arg in &args[1..] {
                    let (id, ty) = lower_expr(prog, arg, scope, sigs)?;
                    expect_num(ty, arg.span(), "an integration argument")?;
                    child_ids.push(id);
                }
                let runtime = if callee == "simpson" { "viv_simpson" } else { "viv_trapezoid" };
                return Ok((prog.add(Viv::Call { callee: runtime.to_string() }, *span, child_ids), Ty::Num));
            }
            // eig_sym(A, vals_out, vecs_out): symmetric eigendecomposition that
            // writes through its array out-parameters (shared data pointers).
            if callee == "eig_sym" {
                if args.len() != 3 {
                    return Err(Diagnostic::error("`eig_sym` takes (A, vals_out, vecs_out)").with_primary(*span, "expected 3 arguments"));
                }
                let mut ids = Vec::with_capacity(3);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_arr(ty, a.span())?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_eig_sym".to_string() }, *span, ids), Ty::Num));
            }
            // spectrum_sym(A, vals_out): the CONTRACTED symmetric-spectrum
            // solve. The emitted runtime cross-checks two independent methods
            // (Jacobi; Householder + implicit-shift QL) and aborts loudly on
            // dissent unless the C is compiled with -DVIV_RELEASE, which
            // auto-selects the direct QL path instead. Debug builds verify,
            // release builds select: numerical CI at the language level.
            if callee == "spectrum_sym" {
                if args.len() != 2 {
                    return Err(Diagnostic::error("`spectrum_sym` takes (A, vals_out)").with_primary(*span, "expected 2 arguments"));
                }
                let mut ids = Vec::with_capacity(2);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_arr(ty, a.span())?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_spectrum_sym".to_string() }, *span, ids), Ty::Num));
            }
            // eig_near(A, sigma, k, re_out, im_out): shift-invert Arnoldi for
            // the k eigenvalues nearest the real shift sigma. Debug runs the
            // dual-depth agreement certificate; -DVIV_RELEASE skips it.
            if callee == "eig_near" {
                if args.len() != 5 {
                    return Err(Diagnostic::error("`eig_near` takes (A, sigma, k, re_out, im_out)")
                        .with_primary(*span, "expected 5 arguments"));
                }
                let mut ids = Vec::with_capacity(5);
                for (i, a) in args.iter().enumerate() {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if i == 1 || i == 2 {
                        if ty != Ty::Num {
                            return Err(Diagnostic::error("`eig_near`: sigma and k are numbers")
                                .with_primary(a.span(), "not a number"));
                        }
                    } else {
                        expect_arr(ty, a.span())?;
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_eig_near".to_string() }, *span, ids), Ty::Num));
            }
            // Dense linear-algebra builtins with all-array signatures:
            //   lin_solve(A, b, x_out)        LU + debug residual certificate
            //   cholesky(A, L_out)            A = L L^T, SPD checked by pivots
            //   eig_general(A, re_out, im_out) Hessenberg + Francis QR
            if matches!(callee.as_str(), "lin_solve" | "lin_solve_multi" | "cholesky" | "eig_general") {
                let want = if callee == "cholesky" { 2 } else { 3 };
                if args.len() != want {
                    return Err(Diagnostic::error(format!("`{}` takes {} array argument(s)", callee, want))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let mut ids = Vec::with_capacity(want);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_arr(ty, a.span())?;
                    ids.push(id);
                }
                let runtime = format!("viv_{}", callee);
                return Ok((prog.add(Viv::Call { callee: runtime }, *span, ids), Ty::Num));
            }
            // assert_near(a, b, tol): numeric assertion.
            if callee == "assert_near" {
                if args.len() != 3 {
                    return Err(Diagnostic::error("`assert_near` takes (a, b, tol)").with_primary(*span, "expected 3 arguments"));
                }
                let mut ids = Vec::with_capacity(3);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "an assert_near argument")?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_assert_near".to_string() }, *span, ids), Ty::Num));
            }
            // The inequality/boolean assert family (scalar arguments).
            if matches!(callee.as_str(), "assert_true" | "assert_gt" | "assert_lt") {
                let want = if callee == "assert_true" { 1 } else { 2 };
                if args.len() != want {
                    return Err(Diagnostic::error(format!("`{callee}` takes {want} argument(s)"))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let mut ids = Vec::with_capacity(want);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    // assert_true takes a condition, which the checker types
                    // as Bool; the sibling asserts compare numbers.
                    if !(matches!(ty, Ty::Num) || (callee == "assert_true" && matches!(ty, Ty::Bool))) {
                        expect_num(ty, a.span(), "an assert argument")?;
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: format!("viv_{}", callee) }, *span, ids), Ty::Num));
            }
            // assert_allclose(a, b, tol): elementwise array comparison.
            if callee == "assert_allclose" {
                if args.len() != 3 {
                    return Err(Diagnostic::error("`assert_allclose` takes (a, b, tol)").with_primary(*span, "expected 3 arguments"));
                }
                let mut ids = Vec::with_capacity(3);
                for (i, a) in args.iter().enumerate() {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if i == 2 { expect_num(ty, a.span(), "the tolerance")?; } else { expect_arr(ty, a.span())?; }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_assert_allclose".to_string() }, *span, ids), Ty::Num));
            }
            // clock(): the wall time in seconds, from a monotonic clock, in a
            // `#! timing` program only (crate::check_clock)
            if callee == "clock" && !sigs.arity.contains_key("clock") {
                if !args.is_empty() {
                    return Err(Diagnostic::error("`clock` takes no arguments").with_primary(*span, "too many arguments"));
                }
                return Ok((prog.add(Viv::Call { callee: "viv_clock".to_string() }, *span, vec![]), Ty::Num));
            }
            // an exported program's parameters (crate::export): `viv export`
            // writes these calls where a deck assigns a parameter, and no
            // other program can name them
            if exporting() {
                if let Some(ty) = export_builtin_ty(callee) {
                    let [Expr::Num { .. }] = args.as_slice() else {
                        return Err(Diagnostic::error(format!("`{callee}` takes one number, the parameter's slot")).with_primary(*span, "here"));
                    };
                    let (id, _) = lower_expr(prog, &args[0], scope, sigs)?;
                    let c = match callee.as_str() {
                        "__export_iarr" => "viv_export_arr".to_string(),
                        other => format!("viv_{}", &other[2..]),
                    };
                    return Ok((prog.add(Viv::Call { callee: c }, *span, vec![id]), ty));
                }
            }
            // release_mode(): the build-mode flag for Viv-level contract code.
            if callee == "release_mode" {
                if !args.is_empty() {
                    return Err(Diagnostic::error("`release_mode` takes no arguments").with_primary(*span, "too many arguments"));
                }
                return Ok((prog.add(Viv::Call { callee: "viv_release_mode".to_string() }, *span, vec![]), Ty::Bool));
            }
            // Seeded PRNG: srand(seed) / rand().
            if callee == "srand" || callee == "rand" {
                let want = if callee == "srand" { 1 } else { 0 };
                if args.len() != want {
                    return Err(Diagnostic::error(format!("`{callee}` takes {want} argument(s)"))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let mut ids = Vec::with_capacity(want);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "the seed")?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: format!("viv_{}", callee) }, *span, ids), Ty::Num));
            }
            // tridiag_solve(sub, diag, super, rhs) -> x, by the Thomas algorithm.
            if callee == "tridiag_solve" {
                if args.len() != 4 {
                    return Err(Diagnostic::error("`tridiag_solve` takes (sub, diag, super, rhs)").with_primary(*span, "expected 4 arrays"));
                }
                let mut ids = Vec::with_capacity(4);
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_arr(ty, a.span())?;
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_tridiag_solve".to_string() }, *span, ids), Ty::Arr(Elem::Num)));
            }
            // roll(a, k[, axis]): circular shift. The axis defaults to 0.
            if callee == "roll" {
                if args.len() != 2 && args.len() != 3 {
                    return Err(Diagnostic::error("`roll` takes (array, shift) or (array, shift, axis)").with_primary(*span, "wrong number of arguments"));
                }
                let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
                expect_arr(aty, args[0].span())?;
                let mut ids = vec![aid];
                for a in &args[1..] {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "`roll`'s shift and axis")?;
                    ids.push(id);
                }
                if args.len() == 2 {
                    ids.push(prog.add(Viv::Num(-1.0), *span, vec![]));   // "no axis given": the runtime insists on rank 1
                }
                return Ok((prog.add(Viv::Call { callee: "viv_roll".to_string() }, *span, ids), Ty::Arr(Elem::Num)));
            }
            // sort(a) sorts `a` in place and has no value: a statement of its
            // own is lowered by `lower_sort`. Anywhere else (`return sort(w)`,
            // `s = sort(w)`) it is refused here, where the mistake is, rather
            // than as the number it used to return, which surfaced as a type
            // error somewhere else.
            if callee == "sort" {
                let what = match args.first() {
                    Some(Expr::Var { name, .. }) => format!("`{name}`"),
                    _ => "its array".to_string(),
                };
                return Err(Diagnostic::error("`sort` sorts its array in place and has no value")
                    .with_primary(*span, format!("this sorts {what} and gives back nothing to use"))
                    .with_note(match args.first() {
                        Some(Expr::Var { name, .. }) => format!("for a sorted copy, copy it first: `s = copy({name})`, then `sort(s)` on a line of its own"),
                        _ => "for a sorted copy, copy the array first (`s = copy(a)`), then `sort(s)` on a line of its own".to_string(),
                    }));
            }
            // max(a) / min(a): the largest / smallest entry of a real array.
            if matches!(callee.as_str(), "max" | "min") && args.len() == 1 {
                let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                // of an Int array, an Int (B343)
                if ty == Ty::Arr(Elem::Int) {
                    return Ok((prog.add(Viv::Call { callee: format!("viv_i{callee}v") }, *span, vec![id]), Ty::Int));
                }
                if ty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error(format!("`{callee}` of one argument takes a real array or an Int array"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                }
                return Ok((prog.add(Viv::Call { callee: format!("viv_{callee}val") }, *span, vec![id]), Ty::Num));
            }
            // Bits -> the `bits` runtime fragment. Scalar in, scalar out; the
            // helpers are the interpreter's arithmetic in C.
            if matches!(callee.as_str(), "band" | "bor" | "bxor" | "bnot" | "shl" | "shr" | "sar" | "u32" | "i32") {
                let arity = if matches!(callee.as_str(), "bnot" | "u32" | "i32") { 1 } else { 2 };
                if args.len() != arity {
                    return Err(Diagnostic::error(format!("`{callee}` expects {arity} argument(s), got {}", args.len()))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let mut ids = Vec::new();
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != Ty::Num {
                        return Err(Diagnostic::error(format!("`{callee}` works on numbers, not {ty:?}"))
                            .with_primary(a.span(), "this is not a number"));
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: format!("viv_{callee}") }, *span, ids), Ty::Num));
            }
            // Math builtins -> libm. Viv names map to C names where they differ
            // (abs -> fabs, min/max -> fmin/fmax). Matches the interpreter's set.
            // fma(a, b, c): a*b + c rounded once, as IEEE 754's fusedMultiplyAdd
            // (vmath.c's viv_fma in C, f64::mul_add in the interpreter and wasm)
            if callee == "fma" {
                if args.len() != 3 {
                    return Err(Diagnostic::error(format!("`fma` expects 3 arguments, got {}", args.len()))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let mut kids = Vec::new();
                for a in args.iter() {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "`fma`'s argument")?;
                    kids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "fma".to_string() }, *span, kids), Ty::Num));
            }
            if matches!(
                callee.as_str(),
                "exp" | "sqrt" | "pow" | "sin" | "cos" | "tan" | "log" | "ln" | "abs" | "fabs" | "floor" | "ceil" | "min" | "max"
                | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh" | "erf" | "gamma" | "round" | "mod"
            ) {
                let arity = if matches!(callee.as_str(), "pow" | "min" | "max" | "atan2" | "mod") { 2 } else { 1 };
                if args.len() != arity {
                    return Err(Diagnostic::error(format!("`{callee}` expects {arity} argument(s), got {}", args.len()))
                        .with_primary(*span, "wrong number of arguments"));
                }
                let cname = match callee.as_str() {
                    "abs" => "fabs",
                    "ln" => "log",
                    "min" => "fmin",
                    "max" => "fmax",
                    "gamma" => "tgamma",
                    "mod" => "fmod",
                    other => other,
                };
                // min / max with an ARRAY operand are elementwise: `max(rho, 1.0e-30)`
                // floors a density, `min(a, b)` takes the smaller entry by entry.
                // A scalar may be on either side; children are [arr, scalar].
                if matches!(callee.as_str(), "min" | "max") {
                    let (lid, lty) = lower_expr(prog, &args[0], scope, sigs)?;
                    let (rid, rty) = lower_expr(prog, &args[1], scope, sigs)?;
                    let is_max = callee == "max";
                    let arr = Ty::Arr(Elem::Num);
                    let node = match (lty, rty) {
                        (a, b) if a == arr && b == arr => Some((if is_max { ArrKind::MaxA } else { ArrKind::MinA }, vec![lid, rid])),
                        (a, Ty::Num) if a == arr => Some((if is_max { ArrKind::MaxS } else { ArrKind::MinS }, vec![lid, rid])),
                        (Ty::Num, b) if b == arr => Some((if is_max { ArrKind::MaxS } else { ArrKind::MinS }, vec![rid, lid])),
                        _ => None,
                    };
                    if let Some((kind, kids)) = node {
                        return Ok((prog.add(Viv::Arr { kind }, *span, kids), arr));
                    }
                    expect_num(lty, args[0].span(), &format!("`{callee}`'s argument"))?;
                    expect_num(rty, args[1].span(), &format!("`{callee}`'s argument"))?;
                    return Ok((prog.add(Viv::Call { callee: cname.to_string() }, *span, vec![lid, rid]), Ty::Num));
                }
                // pow(array, s): every entry to the same scalar power, `pow(rho, 1.0 / 3.0)`.
                if callee == "pow" {
                    let (lid, lty) = lower_expr(prog, &args[0], scope, sigs)?;
                    let (rid, rty) = lower_expr(prog, &args[1], scope, sigs)?;
                    if lty == Ty::Arr(Elem::Num) {
                        expect_num(rty, args[1].span(), "`pow`'s exponent")?;
                        return Ok((prog.add(Viv::Arr { kind: ArrKind::PowS }, *span, vec![lid, rid]), Ty::Arr(Elem::Num)));
                    }
                    expect_num(lty, args[0].span(), "`pow`'s argument")?;
                    expect_num(rty, args[1].span(), "`pow`'s argument")?;
                    return Ok((prog.add(Viv::Call { callee: cname.to_string() }, *span, vec![lid, rid]), Ty::Num));
                }
                // 1-argument math on an ARRAY maps elementwise: y = sin(x)
                if arity == 1 {
                    let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                    if ty == Ty::Arr(Elem::Num) {
                        return Ok((prog.add(Viv::MapMath { cname: cname.to_string() }, *span, vec![id]), Ty::Arr(Elem::Num)));
                    }
                    // ... and on a COMPLEX scalar it goes through C99's c-functions
                    if ty == Ty::Cplx {
                        let (ccname, rty) = match cname {
                            "exp" => ("cexp", Ty::Cplx),
                            "sqrt" => ("csqrt", Ty::Cplx),
                            "log" => ("clog", Ty::Cplx),
                            "sin" => ("csin", Ty::Cplx),
                            "cos" => ("ccos", Ty::Cplx),
                            "tan" => ("ctan", Ty::Cplx),
                            "fabs" => ("cabs", Ty::Num),
                            other => {
                                return Err(Diagnostic::error(format!("`{other}` is not defined for complex numbers"))
                                    .with_primary(args[0].span(), "this is a complex number"));
                            }
                        };
                        return Ok((prog.add(Viv::Call { callee: ccname.to_string() }, *span, vec![id]), rty));
                    }
                    expect_num(ty, args[0].span(), "a math argument")?;
                    return Ok((prog.add(Viv::Call { callee: cname.to_string() }, *span, vec![id]), Ty::Num));
                }
                let mut child_ids = Vec::with_capacity(args.len());
                for a in args {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    expect_num(ty, a.span(), "a math argument")?;
                    child_ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: cname.to_string() }, *span, child_ids), Ty::Num));
            }

            // complex(re, im) and the complex accessors.
            if callee == "eig_herm" {
                if args.len() != 3 {
                    return Err(Diagnostic::error("`eig_herm` takes (A, vals, vecs)").with_primary(*span, "expected 3 arguments"));
                }
                let want = [Ty::Arr(Elem::Cplx), Ty::Arr(Elem::Num), Ty::Arr(Elem::Cplx)];
                let what = ["a complex n x n matrix (`czeros(n, n)`)", "a real n-vector (`zeros(n)`)", "a complex n x n matrix (`czeros(n, n)`)"];
                let mut ids = Vec::with_capacity(3);
                for (k, a) in args.iter().enumerate() {
                    let (id, ty) = lower_expr(prog, a, scope, sigs)?;
                    if ty != want[k] {
                        return Err(Diagnostic::error(format!("`eig_herm`: argument {} must be {}", k + 1, what[k]))
                            .with_primary(a.span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                    }
                    ids.push(id);
                }
                return Ok((prog.add(Viv::Call { callee: "viv_eig_herm".to_string() }, *span, ids), Ty::Num));
            }
            if callee == "fft3" || callee == "ifft3" {
                if args.len() != 1 {
                    return Err(Diagnostic::error(format!("`{callee}` takes one rank-3 complex array")).with_primary(*span, "wrong argument count"));
                }
                let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if ty != Ty::Arr(Elem::Cplx) {
                    return Err(Diagnostic::error(format!("`{callee}` needs a complex array"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                }
                return Ok((prog.add(Viv::Call { callee: format!("viv_{callee}") }, *span, vec![id]), Ty::Num));
            }
            if callee == "sum" {
                if args.len() != 1 {
                    return Err(Diagnostic::error("`sum` takes one array").with_primary(*span, "wrong argument count"));
                }
                let (id, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                // of an Int array, an Int, exactly (B343)
                if ty == Ty::Arr(Elem::Int) {
                    return Ok((prog.add(Viv::Call { callee: "viv_isum".to_string() }, *span, vec![id]), Ty::Int));
                }
                if ty != Ty::Arr(Elem::Num) {
                    return Err(Diagnostic::error("`sum` takes a real array or an Int array")
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                }
                return Ok((prog.add(Viv::Call { callee: "viv_sum".to_string() }, *span, vec![id]), Ty::Num));
            }
            if callee == "complex" {
                if args.len() != 2 {
                    return Err(Diagnostic::error("`complex` takes (re, im)").with_primary(*span, "wrong number of arguments"));
                }
                let (rid_, rt) = lower_expr(prog, &args[0], scope, sigs)?;
                // complex(re, im) over two real ARRAYS builds the complex array
                if rt == Ty::Arr(Elem::Num) {
                    let (iid, it) = lower_expr(prog, &args[1], scope, sigs)?;
                    if it != Ty::Arr(Elem::Num) {
                        return Err(Diagnostic::error("`complex(re, im)`: both must be real arrays, or both numbers")
                            .with_primary(args[1].span(), format!("this is {}", crate::lower::a_or_an(&it.name()))));
                    }
                    return Ok((prog.add(Viv::Call { callee: "viv_carr_from".to_string() }, *span, vec![rid_, iid]), Ty::Arr(Elem::Cplx)));
                }
                expect_num(rt, args[0].span(), "`complex`'s real part")?;
                let (iid, it) = lower_expr(prog, &args[1], scope, sigs)?;
                expect_num(it, args[1].span(), "`complex`'s imaginary part")?;
                return Ok((prog.add(Viv::Call { callee: "viv_cplx".to_string() }, *span, vec![rid_, iid]), Ty::Cplx));
            }
            if matches!(callee.as_str(), "re" | "im" | "conj") {
                if args.len() != 1 {
                    return Err(Diagnostic::error(format!("`{callee}` takes 1 argument")).with_primary(*span, "wrong number of arguments"));
                }
                let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
                return match (callee.as_str(), aty) {
                    ("re", Ty::Cplx) => Ok((prog.add(Viv::Call { callee: "creal".to_string() }, *span, vec![aid]), Ty::Num)),
                    ("im", Ty::Cplx) => Ok((prog.add(Viv::Call { callee: "cimag".to_string() }, *span, vec![aid]), Ty::Num)),
                    ("conj", Ty::Cplx) => Ok((prog.add(Viv::Call { callee: "conj".to_string() }, *span, vec![aid]), Ty::Cplx)),
                    // on a real number they are cheap identities (im is 0)
                    ("re", Ty::Num) => Ok((aid, Ty::Num)),
                    ("conj", Ty::Num) => Ok((aid, Ty::Num)),
                    ("im", Ty::Num) => Ok((prog.add(Viv::Num(0.0), *span, vec![]), Ty::Num)),
                    _ => Err(Diagnostic::error(format!("`{callee}` takes a number or a complex number"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&aty.name())))),
                };
            }

            // fill(arr, v): set every element to v, in place. The mutating
            // counterpart of broadcasting - out-parameters cannot be rebound,
            // so `fill(rho, 0.0)` replaces the reset loop.
            if callee == "fill" {
                if args.len() != 2 {
                    return Err(Diagnostic::error("`fill` takes (array, value)").with_primary(*span, "wrong number of arguments"));
                }
                let (aid, aty) = lower_expr(prog, &args[0], scope, sigs)?;
                if aty == Ty::Arr(Elem::Cplx) {
                    let (vid, vty) = lower_expr(prog, &args[1], scope, sigs)?;
                    if vty != Ty::Cplx {
                        expect_num(vty, args[1].span(), "`fill`'s value")?;
                    }
                    return Ok((prog.add(Viv::Call { callee: "viv_cfill".to_string() }, *span, vec![aid, vid]), Ty::Num));
                }
                expect_arr(aty, args[0].span())?;
                let (vid, vty) = lower_expr(prog, &args[1], scope, sigs)?;
                expect_num(vty, args[1].span(), "`fill`'s value")?;
                return Ok((prog.add(Viv::Call { callee: "viv_fill".to_string() }, *span, vec![aid, vid]), Ty::Num));
            }

            // len(arr): the number of elements in an array's first dimension.
            if callee == "len" {
                if args.len() != 1 {
                    return Err(Diagnostic::error("`len` takes 1 array").with_primary(*span, "wrong number of arguments"));
                }
                let (aid, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                // A registry is a different struct, so it needs its own reader;
                // the numeric and complex arrays share viv_len, which reads the
                // dims[0] both of them carry.
                if let Ty::Arr(Elem::Func(_)) = ty {
                    return Ok((prog.add(Viv::FnLen, *span, vec![aid]), Ty::Num));
                }
                if let Ty::Recs(_) = ty {
                    return Ok((prog.add(Viv::RecLen, *span, vec![aid]), Ty::Num));
                }
                // a string's code points, a list's strings
                if strings::is_text(ty) {
                    let f = if ty == Ty::Str { "viv_s_len" } else { "viv_ss_len" };
                    return Ok((prog.add(Viv::Call { callee: f.to_string() }, *span, vec![aid]), Ty::Num));
                }
                // the number of keys of a map
                if let Ty::Map(k, v) = ty {
                    return Ok((prog.add(Viv::Call { callee: crate::interp::maps::callee("len", k, v) }, *span, vec![aid]), Ty::Num));
                }
                if !matches!(ty, Ty::Arr(_)) {
                    // `a, b = e` of an `e` that is neither an array nor the
                    // values of a def that returns several (crate::tuples)
                    if let Some(d) = unpack_refusal(&args[0], ty) {
                        return Err(d);
                    }
                    expect_arr(ty, args[0].span())?;
                }
                return Ok((prog.add(Viv::Call { callee: "viv_len".to_string() }, *span, vec![aid]), Ty::Num));
            }

            // numel(A): the element count over every dimension; shape(A): the
            // dimensions as a rank-1 array. A generic routine needs both to
            // flatten or copy an array it did not allocate.
            if callee == "numel" || callee == "shape" {
                if args.len() != 1 {
                    return Err(Diagnostic::error(format!("`{callee}` takes 1 array")).with_primary(*span, "wrong number of arguments"));
                }
                let (aid, ty) = lower_expr(prog, &args[0], scope, sigs)?;
                if callee == "numel" {
                    if let Ty::Arr(Elem::Func(_)) = ty {
                        return Ok((prog.add(Viv::FnLen, *span, vec![aid]), Ty::Num));
                    }
                }
                if !matches!(ty, Ty::Arr(Elem::Num) | Ty::Arr(Elem::Cplx)) {
                    return Err(Diagnostic::error(format!("`{callee}` takes a real or complex array"))
                        .with_primary(args[0].span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
                }
                let (c, t) = if callee == "numel" { ("viv_numel", Ty::Num) } else { ("viv_shape", Ty::Arr(Elem::Num)) };
                return Ok((prog.add(Viv::Call { callee: c.to_string() }, *span, vec![aid]), t));
            }

            // deriv(fn(x) -> expr, x0): automatic forward-mode differentiation.
            // The closure body is dualized in place (chain rule over +,-,*,/,exp,
            // sqrt,pow), returning the *exact* derivative d/dx of expr at x0.
            if callee == "deriv" {
                if args.len() != 2 {
                    return Err(Diagnostic::error("`deriv` takes (fn(x) -> expr, x0)").with_primary(*span, "wrong number of arguments"));
                }
                let Expr::Lambda { params, body, span: lspan } = &args[0] else {
                    return Err(Diagnostic::error("`deriv`'s first argument must be a `fn(x) -> ...`").with_primary(args[0].span(), "expected a closure"));
                };
                if params.len() != 1 {
                    return Err(Diagnostic::error("`deriv`'s closure must take exactly 1 argument").with_primary(*lspan, "wrong arity"));
                }
                let (x0, x0ty) = lower_expr(prog, &args[1], scope, sigs)?;
                expect_num(x0ty, args[1].span(), "the point for `deriv`")?;
                let (_val, der) = lower_dual(prog, body, &params[0], x0, scope, sigs)?;
                return Ok((der, Ty::Num));
            }

            }

            let arity = *sigs.arity.get(callee).ok_or_else(|| {
                let d = Diagnostic::error(format!("call to unknown function `{}`", callee)).with_primary(*span, "no such function");
                if callee == "range" {
                    d.with_note(crate::hints::RANGE)
                } else if callee == "cas_deriv_at" {
                    // `... using the symbolic derivative` (crate::phrases)
                    d.with_note("the symbolic derivative is lib/cas's: add `import cas.cas` at the top of the program")
                } else {
                    d
                }
            })?;
            if args.len() != arity {
                return Err(Diagnostic::error(format!("`{}` expects {} argument(s), got {}", callee, arity, args.len()))
                    .with_primary(*span, "wrong number of arguments"));
            }
            let mut child_ids = Vec::with_capacity(args.len());
            for (j, arg) in args.iter().enumerate() {
                let want = sigs.param_ty(callee, j);
                let (id, ty) = strings::lower_expected(prog, arg, Some(want), scope, sigs)?;
                // an Int where a def takes a number: the number it is, exactly
                let (id, ty) = bind_conv(prog, id, ty, want, arg);
                if ty != want {
                    let msg = if crate::update::is_update_def(callee) {
                        // the parameters of a record update's def are its new
                        // values and, last, the record after `..`
                        if j + 1 == args.len() {
                            format!("the record after `..` should be {} but is {}", art(want), art(ty))
                        } else {
                            format!("this field should be {} but is {}", art(want), art(ty))
                        }
                    } else {
                        format!("argument {} of `{}` should be {} but is {}", j + 1, callee, art(want), art(ty))
                    };
                    let d = Diagnostic::error(msg).with_primary(arg.span(), "wrong argument type");
                    return Err(if (want, ty) == (Ty::Int, Ty::Num) {
                        d.with_note("a number becomes an Int only where you say how: `int(x)` (toward zero), `round(x)`, `floor(x)` or `ceil(x)`; a variable that holds a whole number can say so, `n: Int = 10`")
                    } else {
                        d
                    });
                }
                child_ids.push(id);
            }
            Ok((prog.add(Viv::Call { callee: def_symbol(callee) }, *span, child_ids), sigs.ret_ty(callee)))
        }
        Expr::RecordLit { type_name, type_span, fields, span } => {
            let info = sigs.records.get(type_name).ok_or_else(|| {
                Diagnostic::error(format!("unknown record type `{type_name}`")).with_primary(*type_span, "not a declared record")
            })?;
            // Collect provided fields, rejecting unknown names and duplicates.
            let mut provided: HashMap<&str, &Expr> = HashMap::new();
            for (fname, fspan, value) in fields {
                if !info.fields.iter().any(|(d, _)| d == fname) {
                    return Err(Diagnostic::error(format!("record `{type_name}` has no field `{fname}`")).with_primary(*fspan, "unknown field"));
                }
                if provided.insert(fname.as_str(), value).is_some() {
                    return Err(Diagnostic::error(format!("field `{fname}` is set more than once")).with_primary(*fspan, "duplicate field"));
                }
            }
            // Lower values in declaration order so children match the struct layout.
            let mut child_ids = Vec::with_capacity(info.fields.len());
            let mut field_order = Vec::with_capacity(info.fields.len());
            for (dfield, dty) in &info.fields {
                let Some(&value) = provided.get(dfield.as_str()) else {
                    return Err(Diagnostic::error(format!("missing field `{dfield}` in `{type_name}` literal")).with_primary(*span, "a field with no default must be given")
                        .with_note(format!("give `{dfield}: ...` here, or declare a default in the record: `{dfield}: Type = value`")));
                };
                let (vid, vty) = strings::lower_expected(prog, value, Some(*dty), scope, sigs)?;
                let (vid, vty) = bind_conv(prog, vid, vty, *dty, value);
                if vty != *dty {
                    if let Some(def) = crate::tuples::tuple_def(type_name) {
                        let k: usize = dfield[1..].parse().unwrap_or(0) + 1;
                        let a = |t: Ty| format!("{} {}", if t.name().starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" }, t.name());
                        return Err(Diagnostic::error(format!("value {k} that `{def}` returns is {} in one `return` and {} in another", a(*dty), a(vty)))
                            .with_primary(value.span(), a(vty))
                            .with_note("each value a def returns has one type"));
                    }
                    return Err(Diagnostic::error(format!("field `{dfield}` of `{type_name}` is {} but this value is {}", an(*dty), an(vty))).with_primary(value.span(), "wrong field type"));
                }
                child_ids.push(owned(prog, vid, vty, value, sigs));
                field_order.push(dfield.clone());
            }
            let rty = Ty::Record(info.name);
            Ok((prog.add(Viv::RecordLit { ty_name: info.name.to_string(), fields: field_order }, *span, child_ids), rty))
        }
        Expr::Field { base, field, field_span, span } => {
            let (bid, bty) = lower_expr(prog, base, scope, sigs)?;
            let Ty::Record(rname) = bty else {
                return Err(Diagnostic::error(format!("cannot access field `{field}` of a {}", bty.name())).with_primary(base.span(), "not a record"));
            };
            let info = sigs.records.get(rname).expect("a record type is always registered");
            let fty = info
                .fields
                .iter()
                .find(|(f, _)| f == field)
                .map(|(_, t)| *t)
                .ok_or_else(|| Diagnostic::error(format!("record `{rname}` has no field `{field}`")).with_primary(*field_span, "unknown field"))?;
            Ok((prog.add(Viv::Field { field: field.clone() }, *span, vec![bid]), fty))
        }
    }
}

/// Lower `[...]` into an `ArrLit` node. Rank 1 (vector of scalars) or rank 2
/// (matrix: rows that are equal-length scalar literals). Children are the
/// flattened scalar elements, row-major.
fn lower_arr_lit(prog: &mut IrProgram<Viv>, elems: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    if elems.is_empty() {
        // `[]`: an empty vector, the start of an array grown by `append`
        let n = prog.add(Viv::Num(0.0), span, vec![]);
        return Ok((prog.add(Viv::Zeros, span, vec![n]), Ty::Arr(Elem::Num)));
    }
    // `["a", "b"]`: a list of strings
    if matches!(elems[0], Expr::Str { .. }) || strings::peek_text(&elems[0], scope, sigs) == Some(Ty::Str) {
        return strings::lower_list_literal(prog, elems, span, scope, sigs);
    }
    // Matrix if the first element is itself an array literal.
    if matches!(elems[0], Expr::ArrLit { .. }) {
        let cols = match &elems[0] {
            Expr::ArrLit { elems, .. } => elems.len(),
            _ => unreachable!(),
        };
        let mut child_ids = Vec::new();
        for row in elems {
            let Expr::ArrLit { elems: row_elems, span: row_span } = row else {
                return Err(Diagnostic::error("matrix rows must be array literals").with_primary(row.span(), "expected `[...]`"));
            };
            if row_elems.len() != cols {
                return Err(Diagnostic::error("matrix rows must all have the same length").with_primary(*row_span, "ragged row"));
            }
            for e in row_elems {
                let (id, ty) = lower_expr(prog, e, scope, sigs)?;
                expect_num(ty, e.span(), "a matrix element")?;
                child_ids.push(id);
            }
        }
        Ok((prog.add(Viv::ArrLit { dims: vec![elems.len(), cols] }, span, child_ids), Ty::Arr(Elem::Num)))
    } else {
        let mut lowered = Vec::with_capacity(elems.len());
        for e in elems {
            lowered.push((lower_expr(prog, e, scope, sigs)?, e.span()));
        }
        // an array of records: every element a record of one type
        if let ((_, Ty::Record(rec)), _) = lowered[0] {
            let mut child_ids = Vec::with_capacity(elems.len());
            for (((id, ty), espan), e) in lowered.iter().zip(elems) {
                if *ty != Ty::Record(rec) {
                    return Err(Diagnostic::error(format!("an array of records holds records of one type; the first is a {rec} and this is {}", crate::lower::a_or_an(&ty.name())))
                        .with_primary(*espan, "a different type"));
                }
                child_ids.push(owned(prog, *id, *ty, e, sigs));
            }
            return Ok((prog.add(Viv::RecArrLit { rec: rec.to_string() }, span, child_ids), Ty::Recs(rec)));
        }
        if let Some((_, espan)) = lowered.iter().find(|((_, t), _)| matches!(t, Ty::Record(_))) {
            return Err(Diagnostic::error("an array holds numbers, functions or records of one type -- not a mixture")
                .with_primary(*espan, "a record"));
        }
        if let Some((_, espan)) = lowered.iter().find(|((_, t), _)| matches!(t, Ty::Map(..))) {
            return Err(Diagnostic::error(crate::interp::maps::NO_MAP_ARRAYS).with_primary(*espan, "a map"));
        }
        // A REGISTRY: every element a function, all of the same arity. Checked
        // before the numeric path, because a C array has one element type --
        // "all functions" is the only mixture of functions that has a packed
        // representation at all.
        if let Ty::Func(sig) = lowered[0].0 .1 {
            let mut child_ids = Vec::with_capacity(elems.len());
            for ((id, ty), espan) in &lowered {
                match ty {
                    Ty::Func(s2) if *s2 == sig => child_ids.push(*id),
                    Ty::Func(s2) if s2.arity != sig.arity => {
                        return Err(Diagnostic::error(format!(
                            "every function in one array takes the same number of arguments; this one takes {} where the first takes {}",
                            s2.arity, sig.arity))
                            .with_primary(*espan, "different signature")
                            .with_note("a registry is dispatched through one shared signature, so its members share it"))
                    }
                    Ty::Func(s2) => {
                        // Same arity, different kinds: name them, and the first
                        // slot that differs. The usual cause is a parameter one
                        // member never reads, which inference then calls a number.
                        let name_of = |e: &Expr| match e {
                            Expr::Var { name, .. } => format!("`{name}`"),
                            _ => "this function".to_string(),
                        };
                        let this = name_of(&elems[child_ids.len()]);
                        let first = name_of(&elems[0]);
                        let detail = match (0..sig.arity as usize).find(|&j| s2.kind(j) != sig.kind(j)) {
                            Some(j) => format!(
                                "parameter {} is {} in {this} but {} in {first}; if a function does not use a parameter, annotate it (`name: Arr`) so its kind matches",
                                j + 1, s2.kind(j).word(), sig.kind(j).word()),
                            None => format!("{this} returns {} but {first} returns {}", s2.ret.word(), sig.ret.word()),
                        };
                        return Err(Diagnostic::error(format!(
                            "every function in one array has the same signature; {this} is {} but {first} is {}",
                            s2.describe(), sig.describe()))
                            .with_primary(*espan, "different signature")
                            .with_note(detail))
                    }
                    _ => {
                        return Err(Diagnostic::error("an array holds numbers, or functions -- not a mixture")
                            .with_primary(*espan, format!("this is {}", crate::lower::a_or_an(&ty.name()))))
                    }
                }
            }
            return Ok((prog.add(Viv::FnArrLit { n: elems.len(), sig }, span, child_ids), Ty::Arr(Elem::Func(sig))));
        }
        let mut child_ids = Vec::with_capacity(elems.len());
        let mut any_cplx = false;
        for ((id, ty), espan) in lowered {
            // an Int among numbers: the number it is, exactly (an Int array's
            // literal is written where an [Int] is wanted, ints::lower_for)
            let (id, ty) = ints::to_num(prog, id, ty, espan);
            if ty == Ty::Cplx {
                any_cplx = true;
            } else if let Ty::Enum(name) = ty {
                // an array of enums is not supported yet
                return Err(Diagnostic::error(format!("an array of enum values (`[{name}]`) is not supported yet"))
                    .with_primary(espan, format!("this is a {name}"))
                    .with_note("enums are not array elements, map values or payload arrays yet; hold the values in a record's fields, or in an [Int] of their variant numbers"));
            } else {
                expect_num(ty, espan, "a vector element")?;
            }
            child_ids.push(id);
        }
        if any_cplx {
            // Numbers with a complex among them: the PACKED complex array, reals
            // promoting. Packed whenever it can be, same as the interpreter.
            Ok((prog.add(Viv::CArrLit { n: elems.len() }, span, child_ids), Ty::Arr(Elem::Cplx)))
        } else {
            Ok((prog.add(Viv::ArrLit { dims: vec![elems.len()] }, span, child_ids), Ty::Arr(Elem::Num)))
        }
    }
}

/// Type-directed resolution of a binary operator. The operand types pick a
/// scalar C operator (`Bin`) or a specialized array operation (`Arr`); the
/// backend then emits each faithfully without re-deriving types.
fn lower_binop(
    prog: &mut IrProgram<Viv>,
    op: BinOp,
    lhs: (NodeId, Ty, Span),
    rhs: (NodeId, Ty, Span),
    span: Span,
) -> Result<(NodeId, Ty), Diagnostic> {
    use BinOp::*;
    let (lid, lty, lspan) = lhs;
    let (rid, rty, rspan) = rhs;
    let mismatch = |verb: &str| {
        Diagnostic::error(format!("cannot {verb} a {} and a {}", lty.name(), rty.name()))
            .with_primary(span, "incompatible operand types")
    };
    match op {
        // two booleans can be compared for equality (and nothing else)
        EqEq | Ne if lty == Ty::Bool && rty == Ty::Bool => {
            Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Bool))
        }
        // a comparison with a real array on either side is elementwise and
        // gives an array of 1.0 and 0.0 (a mask is an array of numbers); a
        // scalar broadcasts from either side
        Lt | Le | Gt | Ge | EqEq | Ne if lty == Ty::Arr(Elem::Num) || rty == Ty::Arr(Elem::Num) => {
            let arr = Ty::Arr(Elem::Num);
            // `s < a` is `a > s`: the scalar always goes second
            let flip = |o: BinOp| match o {
                Lt => Gt,
                Le => Ge,
                Gt => Lt,
                Ge => Le,
                other => other,
            };
            let (kind_op, kids, scalar) = match (lty, rty) {
                (a, b) if a == arr && b == arr => (op, vec![lid, rid], false),
                (a, Ty::Num) if a == arr => (op, vec![lid, rid], true),
                (Ty::Num, b) if b == arr => (flip(op), vec![rid, lid], true),
                _ => return Err(mismatch("compare")),
            };
            let kind = match (kind_op, scalar) {
                (Lt, false) => ArrKind::LtA,
                (Le, false) => ArrKind::LeA,
                (Gt, false) => ArrKind::GtA,
                (Ge, false) => ArrKind::GeA,
                (EqEq, false) => ArrKind::EqA,
                (_, false) => ArrKind::NeA,
                (Lt, true) => ArrKind::LtS,
                (Le, true) => ArrKind::LeS,
                (Gt, true) => ArrKind::GtS,
                (Ge, true) => ArrKind::GeS,
                (EqEq, true) => ArrKind::EqS,
                (_, true) => ArrKind::NeS,
            };
            Ok((prog.add(Viv::Arr { kind }, span, kids), arr))
        }
        Lt | Le | Gt | Ge | EqEq | Ne => {
            expect_num(lty, lspan, "an operand")?;
            expect_num(rty, rspan, "an operand")?;
            Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Bool))
        }
        And | Or => {
            expect_bool(lty, lspan)?;
            expect_bool(rty, rspan)?;
            Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Bool))
        }
        // `%` and `//`: a runtime call, since a zero divisor stops the program
        // with its location. The third child is where the operator is (span
        // start + 1 with sites on, else 0): it keeps two identical-looking
        // `i % n` on different lines apart, so each reports its own line.
        Mod | FloorDiv => {
            let what = if op == Mod { "`%`'s operand" } else { "`//`'s operand" };
            expect_num(lty, lspan, what)?;
            expect_num(rty, rspan, what)?;
            let at = prog.add(Viv::Num(site_at(span) as f64), span, vec![]);
            let callee = if op == Mod { "viv_floormod" } else { "viv_floordiv" };
            Ok((prog.add(Viv::Call { callee: callee.to_string() }, span, vec![lid, rid, at]), Ty::Num))
        }
        Matmul => {
            // complex @ complex. A mixed product is refused rather than promoted:
            // the interpreter refuses it too, and says how to promote by hand.
            if lty == Ty::Arr(Elem::Cplx) || rty == Ty::Arr(Elem::Cplx) {
                if lty != rty {
                    return Err(Diagnostic::error("`@` on complex arrays needs BOTH operands complex")
                        .with_primary(span, format!("this is {} @ a {}", crate::lower::a_or_an(&lty.name()), rty.name()))
                        .with_note("promote the real one with `complex(a, zeros(..))`"));
                }
                return Ok((prog.add(Viv::Arr { kind: ArrKind::CMatmul }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx)));
            }
            expect_arr(lty, lspan)?;
            expect_arr(rty, rspan)?;
            Ok((prog.add(Viv::Arr { kind: ArrKind::Matmul }, span, vec![lid, rid]), Ty::Arr(Elem::Num)))
        }
        Add | Sub => match (lty, rty) {
            (Ty::Num, Ty::Num) => Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Num)),
            (Ty::Cplx, Ty::Cplx) | (Ty::Cplx, Ty::Num) | (Ty::Num, Ty::Cplx) => Ok(complex_arith(prog, op, span, lid, rid)),
            (Ty::Arr(Elem::Num), Ty::Arr(Elem::Num)) => {
                let kind = if op == Add { ArrKind::Add } else { ArrKind::Sub };
                Ok((prog.add(Viv::Arr { kind }, span, vec![lid, rid]), Ty::Arr(Elem::Num)))
            }
            // broadcasting: children stored [arr, scalar] in every case
            (Ty::Arr(Elem::Num), Ty::Num) => {
                let kind = if op == Add { ArrKind::AddS } else { ArrKind::SubS };
                Ok((prog.add(Viv::Arr { kind }, span, vec![lid, rid]), Ty::Arr(Elem::Num)))
            }
            (Ty::Num, Ty::Arr(Elem::Num)) => {
                let kind = if op == Add { ArrKind::AddS } else { ArrKind::SubSR };
                Ok((prog.add(Viv::Arr { kind }, span, vec![rid, lid]), Ty::Arr(Elem::Num)))
            }
            (Ty::Arr(Elem::Cplx), Ty::Arr(Elem::Cplx)) => {
                let kind = if op == Add { ArrKind::CAdd } else { ArrKind::CSub };
                Ok((prog.add(Viv::Arr { kind }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx)))
            }
            _ => Err(mismatch("add/subtract")),
        },
        Mul => match (lty, rty) {
            (Ty::Num, Ty::Num) => Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Num)),
            (Ty::Cplx, Ty::Cplx) | (Ty::Cplx, Ty::Num) | (Ty::Num, Ty::Cplx) => Ok(complex_arith(prog, op, span, lid, rid)),
            // Scale: store children as [array, scalar] regardless of source order.
            (Ty::Num, Ty::Arr(Elem::Num)) => Ok((prog.add(Viv::Arr { kind: ArrKind::Scale }, span, vec![rid, lid]), Ty::Arr(Elem::Num))),
            (Ty::Arr(Elem::Num), Ty::Num) => Ok((prog.add(Viv::Arr { kind: ArrKind::Scale }, span, vec![lid, rid]), Ty::Arr(Elem::Num))),
            (Ty::Arr(Elem::Num), Ty::Arr(Elem::Num)) => Ok((prog.add(Viv::Arr { kind: ArrKind::Had }, span, vec![lid, rid]), Ty::Arr(Elem::Num))),
            (Ty::Arr(Elem::Cplx), Ty::Arr(Elem::Cplx)) => Ok((prog.add(Viv::Arr { kind: ArrKind::CHad }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx))),
            // complex * real, elementwise: children [carr, rarr] (CHadR), or in
            // source order [rarr, carr] (CHadRL) when the real one came first
            (Ty::Arr(Elem::Cplx), Ty::Arr(Elem::Num)) => Ok((prog.add(Viv::Arr { kind: ArrKind::CHadR }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx))),
            (Ty::Arr(Elem::Num), Ty::Arr(Elem::Cplx)) => Ok((prog.add(Viv::Arr { kind: ArrKind::CHadRL }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx))),
            // complex array * scalar (real or complex): children [carr, scalar]
            (Ty::Arr(Elem::Cplx), Ty::Num) | (Ty::Arr(Elem::Cplx), Ty::Cplx) => Ok((prog.add(Viv::Arr { kind: ArrKind::CScale }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx))),
            (Ty::Num, Ty::Arr(Elem::Cplx)) | (Ty::Cplx, Ty::Arr(Elem::Cplx)) => Ok((prog.add(Viv::Arr { kind: ArrKind::CScale }, span, vec![rid, lid]), Ty::Arr(Elem::Cplx))),
            _ => Err(mismatch("multiply")),
        },
        Div => match (lty, rty) {
            (Ty::Num, Ty::Num) => Ok((prog.add(Viv::Bin { op, cplx: false }, span, vec![lid, rid]), Ty::Num)),
            (Ty::Cplx, Ty::Cplx) | (Ty::Cplx, Ty::Num) | (Ty::Num, Ty::Cplx) => Ok(complex_arith(prog, op, span, lid, rid)),
            (Ty::Arr(Elem::Num), Ty::Num) => Ok((prog.add(Viv::Arr { kind: ArrKind::DivS }, span, vec![lid, rid]), Ty::Arr(Elem::Num))),
            (Ty::Arr(Elem::Num), Ty::Arr(Elem::Num)) => Ok((prog.add(Viv::Arr { kind: ArrKind::Div }, span, vec![lid, rid]), Ty::Arr(Elem::Num))),
            // s / a: children are [arr, scalar], as for SubSR
            (Ty::Num, Ty::Arr(Elem::Num)) => Ok((prog.add(Viv::Arr { kind: ArrKind::DivSR }, span, vec![rid, lid]), Ty::Arr(Elem::Num))),
            (Ty::Arr(Elem::Cplx), Ty::Num) | (Ty::Arr(Elem::Cplx), Ty::Cplx) => Ok((prog.add(Viv::Arr { kind: ArrKind::CDivS }, span, vec![lid, rid]), Ty::Arr(Elem::Cplx))),
            _ => Err(mismatch("divide")),
        },
    }
}

/// Does a statement list contain a `return` (anywhere, including nested blocks)?
/// The body of a `placeholder def`.
fn is_placeholder(body: &[Stmt]) -> bool {
    matches!(body, [Stmt::Lens { lens, .. }] if matches!(**lens, Lens::Placeholder { .. }))
}

fn has_return(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Return { .. } => true,
        Stmt::If { then_body, else_body, .. } => has_return(then_body) || has_return(else_body),
        Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => has_return(body),
        _ => false,
    })
}

/// Does an expression reference the variable `name`?
fn expr_uses_var(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Var { name: n, .. } => n == name,
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => false,
        Expr::ArrLit { elems, .. } => elems.iter().any(|x| expr_uses_var(x, name)),
        Expr::Index { base, indices, .. } => expr_uses_var(base, name) || indices.iter().any(|x| expr_uses_var(x, name)),
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_uses_var(operand, name),
        Expr::Binary { lhs, rhs, .. } => expr_uses_var(lhs, name) || expr_uses_var(rhs, name),
        Expr::Call { args, .. } => args.iter().any(|x| expr_uses_var(x, name)),
        Expr::Lambda { body, .. } => expr_uses_var(body, name),
        Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, v)| expr_uses_var(v, name)),
        Expr::Field { base, .. } => expr_uses_var(base, name),
    }
}

/// If `value` is `name <op> rhs` for a reducible op (from a desugared `+=`/`*=`),
/// return the OpenMP reduction operator.
fn reduction_op(value: &Expr, name: &str) -> Option<&'static str> {
    if let Expr::Binary { op, lhs, .. } = value {
        if let Expr::Var { name: ln, .. } = &**lhs {
            if ln == name {
                return match op {
                    BinOp::Add => Some("+"),
                    BinOp::Mul => Some("*"),
                    _ => None,
                };
            }
        }
    }
    None
}

/// Record, per variable assigned in a loop body, whether it has a "fresh"
/// (non-self-referencing) assignment and/or a reducible self-assignment.
fn collect_par_vars(stmts: &[Stmt], info: &mut HashMap<String, (bool, Option<&'static str>)>) {
    for s in stmts {
        match s {
            Stmt::Assign { name, value, .. } => {
                let self_ref = expr_uses_var(value, name);
                let entry = info.entry(name.clone()).or_insert((false, None));
                if self_ref {
                    if let Some(op) = reduction_op(value, name) {
                        entry.1 = Some(op);
                    }
                } else {
                    entry.0 = true;
                }
            }
            Stmt::For { var, body, .. } => {
                info.entry(var.clone()).or_insert((false, None)).0 = true;
                collect_par_vars(body, info);
            }
            Stmt::While { body, .. } => collect_par_vars(body, info),
            Stmt::If { then_body, else_body, .. } => {
                collect_par_vars(then_body, info);
                collect_par_vars(else_body, info);
            }
            Stmt::Test { body, .. } => collect_par_vars(body, info),
            _ => {}
        }
    }
}

/// Classify a parallel loop's body variables into thread-private temporaries and
/// cross-iteration reductions. A variable with a fresh assignment is private
/// (re-initialized each iteration); one only ever `+=`/`*=`-updated is a
/// reduction; anything else is a loop-carried dependency we refuse to parallelize.
fn analyze_parallel(body: &[Stmt], span: Span) -> Result<(Vec<String>, Vec<(String, String)>), Diagnostic> {
    let mut info: HashMap<String, (bool, Option<&'static str>)> = HashMap::new();
    collect_par_vars(body, &mut info);
    let mut entries: Vec<(String, (bool, Option<&'static str>))> = info.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut privates = Vec::new();
    let mut reductions = Vec::new();
    for (name, (fresh, reduce)) in entries {
        if fresh {
            privates.push(name);
        } else if let Some(op) = reduce {
            reductions.push((op.to_string(), name));
        } else {
            return Err(Diagnostic::error(format!("cannot parallelize: `{name}` has a loop-carried dependency"))
                .with_primary(span, "this loop's iterations are not independent"));
        }
    }
    Ok((privates, reductions))
}

/// Collect the free variables of an expression (names referenced but not the
/// given bound name), in first-seen order. Used for closure capture analysis.
fn collect_free_vars(expr: &Expr, bound: &str, out: &mut Vec<String>) {
    match expr {
        Expr::Var { name, .. } => {
            if name != bound && !out.contains(name) {
                out.push(name.clone());
            }
        }
        Expr::Num { .. } | Expr::Imag { .. } | Expr::Bool { .. } | Expr::Str { .. } => {}
        Expr::ArrLit { elems, .. } => {
            for e in elems {
                collect_free_vars(e, bound, out);
            }
        }
        Expr::Index { base, indices, .. } => {
            collect_free_vars(base, bound, out);
            for i in indices {
                collect_free_vars(i, bound, out);
            }
        }
        Expr::Neg { operand, .. } | Expr::Not { operand, .. } => collect_free_vars(operand, bound, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_free_vars(lhs, bound, out);
            collect_free_vars(rhs, bound, out);
        }
        Expr::Call { args, .. } => {
            for a in args {
                collect_free_vars(a, bound, out);
            }
        }
        Expr::Lambda { body, .. } => collect_free_vars(body, bound, out),
        Expr::RecordLit { fields, .. } => {
            for (_, _, v) in fields {
                collect_free_vars(v, bound, out);
            }
        }
        Expr::Field { base, .. } => collect_free_vars(base, bound, out),
    }
}

// Node builders for the dual-AD lowering (free fns to sidestep borrow conflicts).
fn dnum(prog: &mut IrProgram<Viv>, sp: Span, v: f64) -> NodeId {
    prog.add(Viv::Num(v), sp, vec![])
}
fn dbin(prog: &mut IrProgram<Viv>, sp: Span, op: BinOp, a: NodeId, b: NodeId) -> NodeId {
    prog.add(Viv::Bin { op, cplx: false }, sp, vec![a, b])
}
fn dcall(prog: &mut IrProgram<Viv>, sp: Span, name: &str, args: Vec<NodeId>) -> NodeId {
    prog.add(Viv::Call { callee: name.to_string() }, sp, args)
}

/// Forward-mode AD of a scalar expression: returns IR nodes for (value, d value /
/// d `xname`), with `xname` seeded to `xval` and all other names treated as
/// constants. The chain rule is applied over `+ - * /`, `exp`, `sqrt`, and `pow`
/// with a constant exponent. (Automatic forward AD.)
fn lower_dual(prog: &mut IrProgram<Viv>, e: &Expr, xname: &str, xval: NodeId, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, NodeId), Diagnostic> {
    let sp = e.span();
    match e {
        Expr::Num { value, .. } => Ok((dnum(prog, sp, *value), dnum(prog, sp, 0.0))),
        Expr::Var { name, .. } => {
            if name == xname {
                Ok((xval, dnum(prog, sp, 1.0)))
            } else {
                // A value that does not depend on x: derivative 0.
                let (vid, ty) = lower_expr(prog, e, scope, sigs)?;
                expect_num(ty, sp, "a value in `deriv`")?;
                Ok((vid, dnum(prog, sp, 0.0)))
            }
        }
        Expr::Neg { operand, .. } => {
            let (av, ad) = lower_dual(prog, operand, xname, xval, scope, sigs)?;
            Ok((prog.add(Viv::Neg, sp, vec![av]), prog.add(Viv::Neg, sp, vec![ad])))
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let (av, ad) = lower_dual(prog, lhs, xname, xval, scope, sigs)?;
            let (bv, bd) = lower_dual(prog, rhs, xname, xval, scope, sigs)?;
            match op {
                BinOp::Add => Ok((dbin(prog, sp, BinOp::Add, av, bv), dbin(prog, sp, BinOp::Add, ad, bd))),
                BinOp::Sub => Ok((dbin(prog, sp, BinOp::Sub, av, bv), dbin(prog, sp, BinOp::Sub, ad, bd))),
                BinOp::Mul => {
                    let val = dbin(prog, sp, BinOp::Mul, av, bv);
                    let t1 = dbin(prog, sp, BinOp::Mul, av, bd);
                    let t2 = dbin(prog, sp, BinOp::Mul, ad, bv);
                    Ok((val, dbin(prog, sp, BinOp::Add, t1, t2)))
                }
                BinOp::Div => {
                    let val = dbin(prog, sp, BinOp::Div, av, bv);
                    let n1 = dbin(prog, sp, BinOp::Mul, ad, bv);
                    let n2 = dbin(prog, sp, BinOp::Mul, av, bd);
                    let numr = dbin(prog, sp, BinOp::Sub, n1, n2);
                    let den = dbin(prog, sp, BinOp::Mul, bv, bv);
                    Ok((val, dbin(prog, sp, BinOp::Div, numr, den)))
                }
                _ => Err(Diagnostic::error("`deriv` supports + - * / (not comparisons or logic)").with_primary(sp, "unsupported operator here")),
            }
        }
        // A program's own def wins over a builtin everywhere, and `deriv` knows
        // only the builtins' derivatives: a call to a def named `exp` must not
        // be differentiated as e^x. It is refused, like a call to any other def.
        Expr::Call { callee, .. } if sigs.arity.contains_key(callee.as_str()) => Err(Diagnostic::error(format!(
            "`deriv` cannot differentiate through `{callee}`: this program defines its own `{callee}`, and `deriv` knows only the builtins' derivatives"
        ))
        .with_primary(sp, format!("this calls the program's `{callee}`, not the builtin"))
        .with_note("rename the def, or write the closure with the expression it computes")),
        Expr::Call { callee, args, .. } => match callee.as_str() {
            "exp" if args.len() == 1 => {
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let ev = dcall(prog, sp, "exp", vec![av]);
                Ok((ev, dbin(prog, sp, BinOp::Mul, ev, ad)))
            }
            "sqrt" if args.len() == 1 => {
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let r = dcall(prog, sp, "sqrt", vec![av]);
                let two = dnum(prog, sp, 2.0);
                let twor = dbin(prog, sp, BinOp::Mul, two, r);
                Ok((r, dbin(prog, sp, BinOp::Div, ad, twor)))
            }
            "pow" if args.len() == 2 => {
                let c = match &args[1] {
                    Expr::Num { value, .. } => *value,
                    other => return Err(Diagnostic::error("`deriv` differentiates `pow(x, c)` only for a constant exponent").with_primary(other.span(), "exponent must be a number literal")),
                };
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let cnode = dnum(prog, sp, c);
                let val = dcall(prog, sp, "pow", vec![av, cnode]);
                let cm1 = dnum(prog, sp, c - 1.0);
                let p1 = dcall(prog, sp, "pow", vec![av, cm1]);
                let c2 = dnum(prog, sp, c);
                let t = dbin(prog, sp, BinOp::Mul, c2, p1);
                Ok((val, dbin(prog, sp, BinOp::Mul, t, ad)))
            }
            "sin" if args.len() == 1 => {
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let val = dcall(prog, sp, "sin", vec![av]);
                let c = dcall(prog, sp, "cos", vec![av]);
                Ok((val, dbin(prog, sp, BinOp::Mul, c, ad)))
            }
            "cos" if args.len() == 1 => {
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let val = dcall(prog, sp, "cos", vec![av]);
                let sn = dcall(prog, sp, "sin", vec![av]);
                let zero = dnum(prog, sp, 0.0);
                let neg = dbin(prog, sp, BinOp::Sub, zero, sn);
                Ok((val, dbin(prog, sp, BinOp::Mul, neg, ad)))
            }
            "log" if args.len() == 1 => {
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let val = dcall(prog, sp, "log", vec![av]);
                Ok((val, dbin(prog, sp, BinOp::Div, ad, av)))
            }
            "tan" if args.len() == 1 => {
                let (av, ad) = lower_dual(prog, &args[0], xname, xval, scope, sigs)?;
                let val = dcall(prog, sp, "tan", vec![av]);
                let one = dnum(prog, sp, 1.0);
                let t2 = dbin(prog, sp, BinOp::Mul, val, val);
                let sec2 = dbin(prog, sp, BinOp::Add, one, t2);
                Ok((val, dbin(prog, sp, BinOp::Mul, sec2, ad)))
            }
            _ => Err(Diagnostic::error(format!("`deriv` cannot differentiate through `{callee}` (supports exp/sqrt/pow/sin/cos/log/tan and + - * /)")).with_primary(sp, "unsupported call")),
        },
        _ => Err(Diagnostic::error("`deriv` supports scalar arithmetic expressions only").with_primary(sp, "unsupported expression")),
    }
}

/// A call of the array vocabulary (interp/arrays.rs): its arguments checked
/// against `SIGS`, lowered to the C runtime's `viv_ar_<name>_<n>` (with an
/// `n` on the end for the form that takes or gives a number where an array
/// could be).
fn lower_array_call(prog: &mut IrProgram<Viv>, sig: &crate::interp::arrays::Sig, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let name = sig.name;
    let mut ids = Vec::with_capacity(args.len());
    let mut scalar = false;
    let mut first = None;
    for (j, (a, kind)) in args.iter().zip(sig.params.chars()).enumerate() {
        if kind == 's' {
            let Expr::Str { value, .. } = a else {
                return Err(Diagnostic::error(format!("`{name}` takes its subscripts as a string literal"))
                    .with_primary(a.span(), "write the subscripts here, in quotes: \"ij,jk->ik\""));
            };
            if let Err(e) = crate::interp::arrays::einsum_rank(value, args.len() - 1) {
                return Err(Diagnostic::error(e).with_primary(a.span(), "these subscripts"));
            }
            ids.push(prog.add(Viv::StrLit(value.clone()), a.span(), vec![]));
            continue;
        }
        let (id, ty) = lower_expr(prog, a, scope, sigs)?;
        let ok = match kind {
            'a' => ty == Ty::Arr(Elem::Num),
            'n' => ty == Ty::Num,
            'b' => ty == Ty::Bool,
            _ => {
                scalar |= ty == Ty::Num;
                ty == Ty::Num || ty == Ty::Arr(Elem::Num)
            }
        };
        if !ok {
            let want = match kind {
                'a' => "a real array",
                'n' => "a number",
                'b' => "true or false",
                _ => "a number or a real array",
            };
            return Err(Diagnostic::error(format!("`{name}` takes {want} as argument {}", j + 1))
                .with_primary(a.span(), format!("this is {}", crate::lower::a_or_an(&ty.name()))));
        }
        if j == 0 {
            first = Some(ty);
        }
        ids.push(id);
    }
    let ret = match sig.ret {
        'a' => Ty::Arr(Elem::Num),
        'x' => first.unwrap_or(Ty::Num),
        'e' => {
            let Some(Expr::Str { value, .. }) = args.first() else { unreachable!("checked above") };
            if crate::interp::arrays::einsum_rank(value, args.len() - 1) == Ok(0) {
                scalar = true;
                Ty::Num
            } else {
                Ty::Arr(Elem::Num)
            }
        }
        _ => Ty::Num,
    };
    let callee = crate::interp::arrays::c_name(name, args.len(), scalar);
    Ok((prog.add(Viv::Call { callee }, span, ids), ret))
}

fn expect_arr(ty: Ty, span: Span) -> Result<(), Diagnostic> {
    if ty == Ty::Arr(Elem::Num) {
        Ok(())
    } else {
        Err(Diagnostic::error("expected an array").with_primary(span, format!("this is {}", crate::lower::a_or_an(&ty.name()))))
    }
}

/// A type with its article, as the type errors name it: "an Int", and for
/// every other type "a" and its name, as these messages always wrote it.
/// A type's name with the article its first letter takes ("an array",
/// "an Int", "a number"), as the interpreter writes a record field's type.
fn an(t: Ty) -> String {
    let n = t.name();
    if n.starts_with(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']) { format!("an {n}") } else { format!("a {n}") }
}

fn art(t: Ty) -> String {
    if matches!(t, Ty::Int | Ty::Arr(Elem::Int)) || t.name().starts_with("Int") {
        format!("an {}", t.name())
    } else {
        format!("a {}", t.name())
    }
}

/// An index: a number (exactly an integer, checked when it is used) or an Int.
fn expect_index(ty: Ty, span: Span) -> Result<(), Diagnostic> {
    if ty == Ty::Int {
        return Ok(());
    }
    expect_num(ty, span, "an index")
}

/// The builtins that take an Int as it is, rather than as a number:
/// `print` and the text of a value write its digits.
const NATIVE_INT: [&str; 4] = ["print", "show", "len", "assert_true"];

thread_local! {
    /// The variables of the def (or top level) being lowered that are only
    /// ever bound to list literals of one length, and that length
    /// (`list_lens`): `list_join_trap` reads them as the literals they are.
    static LIST_LENS: std::cell::RefCell<(HashMap<String, usize>, std::collections::HashSet<String>)> = std::cell::RefCell::new(Default::default());
}

/// The variables of `body` that every assignment binds to a flat list
/// literal of the same length, and are never grown, a parameter or a loop
/// variable.
fn list_lens(body: &[Stmt], params: &[String]) -> (HashMap<String, usize>, std::collections::HashSet<String>) {
    let mut lens: HashMap<String, Option<usize>> = HashMap::new();
    let mut grown = std::collections::HashSet::new();
    fn walk(body: &[Stmt], lens: &mut HashMap<String, Option<usize>>, grown: &mut std::collections::HashSet<String>) {
        for s in body {
            match s {
                // `xs = append(xs, x)`: a list grown as Python's are (a
                // comprehension is one too, its hidden variable)
                Stmt::Assign { name, value: Expr::Call { callee, .. }, .. } if callee == "append" => {
                    grown.insert(name.clone());
                    lens.insert(name.clone(), None);
                }
                Stmt::Assign { name, value, .. } => {
                    let n = match value {
                        Expr::ArrLit { elems, .. } if !elems.is_empty() && !elems.iter().any(|x| matches!(x, Expr::ArrLit { .. })) => Some(elems.len()),
                        _ => None,
                    };
                    let e = lens.entry(name.clone()).or_insert(n);
                    if *e != n {
                        *e = None;
                    }
                }
                Stmt::For { var, body, .. } => {
                    lens.insert(var.clone(), None);
                    walk(body, lens, grown);
                }
                Stmt::If { then_body, else_body, .. } => {
                    walk(then_body, lens, grown);
                    walk(else_body, lens, grown);
                }
                Stmt::While { body, .. } | Stmt::Test { body, .. } => walk(body, lens, grown),
                _ => {}
            }
        }
    }
    walk(body, &mut lens, &mut grown);
    // `row = viv__cmp1`, the comprehension's value: a list as its source is
    fn aliases(body: &[Stmt], out: &mut Vec<(String, String)>) {
        for s in body {
            match s {
                Stmt::Assign { name, value: Expr::Var { name: src, .. }, .. } => out.push((name.clone(), src.clone())),
                Stmt::If { then_body, else_body, .. } => {
                    aliases(then_body, out);
                    aliases(else_body, out);
                }
                Stmt::While { body, .. } | Stmt::Test { body, .. } | Stmt::For { body, .. } => aliases(body, out),
                _ => {}
            }
        }
    }
    let mut al = Vec::new();
    aliases(body, &mut al);
    loop {
        let before = grown.len();
        for (to, from) in &al {
            if grown.contains(from) {
                grown.insert(to.clone());
            }
        }
        if grown.len() == before {
            break;
        }
    }
    let lens = lens.into_iter().filter(|(n, _)| !params.contains(n)).filter_map(|(n, l)| l.map(|l| (n, l))).collect();
    (lens, grown)
}

/// `row + [1.0]`, `[1.0] + [row[k] + row[k + 1] for k to n]`: Python's
/// list join, which `+` on Viv's arrays is not -- it adds element by
/// element, and a one-element array is broadcast (NumPy's rule), so the
/// result is silently another array. Refused where the code says it is a
/// list: a one-element list literal (or a variable only ever bound to one)
/// beside a list the program grows with `append` or builds with a
/// comprehension, and two list literals of different lengths, neither of
/// one element (which could only stop). A one-element array beside any other
/// array is NumPy's broadcast and passes, as do equal lengths and scalars.
fn list_join_trap(lhs: &Expr, lty: Ty, rhs: &Expr, rty: Ty, span: Span) -> Result<(), Diagnostic> {
    if lty != Ty::Arr(Elem::Num) || rty != Ty::Arr(Elem::Num) {
        return Ok(());
    }
    let lit = |e: &Expr| match e {
        Expr::ArrLit { elems, .. } if !elems.is_empty() && !elems.iter().any(|x| matches!(x, Expr::ArrLit { .. })) => Some(elems.len()),
        // a variable only ever bound to list literals of one length
        Expr::Var { name, .. } => LIST_LENS.with(|l| l.borrow().0.get(name).copied()),
        _ => None,
    };
    let grown = |e: &Expr| matches!(e, Expr::Var { name, .. } if LIST_LENS.with(|l| l.borrow().1.contains(name)));
    // a one-element array beside an array is NumPy's broadcast (`[2.0] + x`),
    // and stays one; beside a list the program grew, it is a Python join
    let trap = match (lit(lhs), lit(rhs)) {
        (Some(a), Some(b)) => a != b && a != 1 && b != 1,
        (Some(1), None) => grown(rhs),
        (None, Some(1)) => grown(lhs),
        _ => false,
    };
    if !trap {
        return Ok(());
    }
    Err(Diagnostic::error("`+` adds arrays element by element; it does not join them, as Python's lists join")
        .with_primary(span, "a list literal of another length: this would broadcast or stop, not join")
        .with_note("to join, write `concat(a, b)` (`concat(row, [1.0])`); to add a number to every element, write `a + x`"))
}

/// An Int bound where a number is wanted (a def's parameter, a record
/// field, a return, a number variable): the number it is, exactly, or the
/// program stops. The interpreter converts the
/// same expressions (`ints::numed`).
/// A value bound where a `want` goes: an Int converted to the number it is
/// (`int_to_num_for`), a literal where a Float32 goes made one
/// (`f32s::adapt`, whose refusal waits in `f32s::take_errors`). One call, so
/// the recursive lowering's frame holds one result.
#[inline(never)]
fn bind_conv(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, want: Ty, e: &Expr) -> (NodeId, Ty) {
    let (id, ty) = int_to_num_for(prog, id, ty, want, e.span());
    f32s::adapt_or_note(prog, (id, ty), want, e)
}

fn int_to_num_for(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, want: Ty, span: Span) -> (NodeId, Ty) {
    if want == Ty::Num && ty == Ty::Int {
        ints::numed(span);
        ints::to_num(prog, id, ty, span)
    } else {
        (id, ty)
    }
}

fn expect_num(ty: Ty, span: Span, what: &str) -> Result<(), Diagnostic> {
    if ty == Ty::Num {
        Ok(())
    } else {
        Err(Diagnostic::error(format!("expected a number for {what}")).with_primary(span, format!("this is {}", crate::lower::a_or_an(&ty.name()))))
    }
}

/// `+ - * /` with a complex operand. The C backend emits it as a call to
/// vmath.c's viv_cadd, viv_csub, viv_cmul or viv_cdiv, not C's operators on
/// `double complex`: those follow Annex G for infinities and NaNs,
/// `double * double complex` skips the zero products the interpreter takes,
/// and `/` scales against overflow by a method the platform picks, so its
/// last bit depended on the machine. The direct wasm backend reads it as a
/// Bin like any other and lifts the real operand itself.
fn complex_arith(prog: &mut IrProgram<Viv>, op: BinOp, span: Span, lid: NodeId, rid: NodeId) -> (NodeId, Ty) {
    (prog.add(Viv::Bin { op, cplx: true }, span, vec![lid, rid]), Ty::Cplx)
}

fn expect_bool(ty: Ty, span: Span) -> Result<(), Diagnostic> {
    if ty == Ty::Bool {
        Ok(())
    } else {
        Err(Diagnostic::error("expected a boolean").with_primary(span, format!("this is {}", crate::lower::a_or_an(&ty.name()))))
    }
}

// ───────────────────────────── C emission ──────────────────────────────────

impl CEmit for Viv {
    /// A module of defs and records alone builds as a program that does
    /// nothing (B346); `viv export` renames this main as it renames any.
    fn always_main() -> bool {
        true
    }
    /// Floating-point contraction off for the whole file. A compiler is
    /// otherwise free to fuse `a * b + c` into one fused multiply-add, which
    /// rounds once where the interpreter rounds twice -- and clang on arm64
    /// does so by default. That costs the engines their agreement in the last
    /// bit of ordinary user arithmetic. (Measured: a 216-line low-bit checksum
    /// of every FFT method differed in 24 lines with contraction on and in
    /// none with it off.)
    ///
    /// Then Viv's libm (`vmath.c`): every program computes `sin`, `exp`, `pow`
    /// and the rest with the algorithm the interpreter uses (`crate::vmath`),
    /// not the platform's, which rounds differently in the last bit.
    ///
    /// Fast mode (`-DVIV_FAST`) turns contraction back on
    /// for the whole file and takes the platform libm (vmath.c skips its
    /// `#define sin viv_sin` block); the C text is the same in both modes.
    ///
    /// Before vmath.c, embed.c: how the runtime stops on an error and writes
    /// its output. In a program build that is the C library's calls, as
    /// always; in an exported library (`-DVIV_EMBED`, crate::export) a
    /// return to the caller and a line callback.
    fn preamble() -> &'static str {
        concat!(
            "#pragma STDC FP_CONTRACT OFF\n\n",
            include_str!("embed.c"),
            "\n",
            include_str!("vmath.c"),
            // after vmath.c, which sets contraction off again for its own code
            "\n#if defined(VIV_FAST) && !defined(VIV_FAST_NO_CONTRACT) && defined(__clang__)\n#pragma clang fp contract(fast)\n#elif defined(VIV_FAST) && !defined(VIV_FAST_NO_CONTRACT)\n#pragma STDC FP_CONTRACT ON\n#endif\n"
        )
    }

    /// An exported program (`-DVIV_EMBED`, crate::export) has no `main`:
    /// its body is the entry's to call. And from here on the C library's
    /// exits and stdout writes are forbidden, so a runtime site that does not
    /// go through `viv_stop` or `viv_printf` (embed.c) fails the build
    /// instead of ending the caller's process or printing past its callback.
    fn after_includes() -> &'static str {
        "#ifdef VIV_EMBED\n#define main viv_program_main\n#pragma GCC poison exit _exit printf puts putchar vprintf perror atexit\n#endif\n"
    }

    fn enter_unit(unit: Option<&str>) {
        sites_enter(unit);
        strings::lits_enter(unit);
    }

    fn unit_code(code: &'static str, role: knotkit::backend::UnitRole) -> std::borrow::Cow<'static, str> {
        split_state(code, role)
    }

    fn runtime() -> Runtime {
        // strings: their fragments, after everything else
        let mut rt = base_runtime();
        for f in strings::fragments() {
            rt = rt.with(f);
        }
        // Ints: the checked operations, their text, the
        // elements of an Int array
        rt.with(ints::fragment()).with(ints::str_fragment()).with(ints::arr_fragment()).with(ints::arr_ops_fragment()).with(ints::bytes_fragment())
            .with(ints::ivec_tab_fragment())
            .with(f32s::fragment())
            .with(f32s::arr_fragment())
    }

    fn emit_item(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CItem {
        emit_item_c(node, cx)
    }

    fn emit_stmt(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CStmt {
        emit_stmt_c(node, cx)
    }

    fn emit_expr(node: &Node<Self>, cx: &mut CEmitCx<Self>) -> CExpr {
        emit_expr_c(node, cx)
    }
}

/// A fused elementwise tree, or a reduction over a view or a tree, as one
/// call into the "fuse" fragment (crate::fuse).
fn fuse_emit_c(node: &Node<Viv>, cx: &mut CEmitCx<Viv>) -> Option<CExpr> {
    let id = node.id;
    let (plan, red) = match &node.kind {
        Viv::Arr { .. } | Viv::MapMath { .. } => (crate::fuse::plan_tree(cx.program(), id)?, None),
        Viv::Call { callee } => {
            let &(_, kind, arity) = crate::fuse::REDUCTIONS.iter().find(|r| r.0 == callee)?;
            (crate::fuse::plan_reduction(cx.program(), &node.children, arity)?, Some(kind))
        }
        _ => return None,
    };
    if !crate::fuse::enabled() {
        return None;
    }
    cx.need("fuse");
    let mut code = plan.code.clone();
    for (at, span) in &plan.sites {
        code[*at] = site_row(*span) as f64;
    }
    let text: Vec<String> = code.iter().map(|v| format!("{}", *v as i64)).collect();
    let fns = if plan.fns.is_empty() { "(const viv_fz_fn*)0".to_string() } else { format!("(const viv_fz_fn[]){{ {} }}", plan.fns.join(", ")) };
    let mut args = Vec::new();
    if let Some(kind) = red {
        args.push(CExpr::raw(id, kind.to_string()));
        args.push(CExpr::raw(id, site_row(node.span).to_string()));
    }
    args.push(CExpr::raw(id, format!("(const double[]){{ {} }}", text.join(", "))));
    args.push(CExpr::raw(id, code.len().to_string()));
    args.push(CExpr::raw(id, fns));
    args.push(CExpr::raw(id, plan.arrs.len().to_string()));
    args.push(CExpr::raw(id, plan.nums.len().to_string()));
    for &a in &plan.arrs {
        args.push(cx.expr(a));
    }
    for &n in &plan.nums {
        let e = cx.expr(n);
        args.push(CExpr::call(id, "(double)", vec![e]));
    }
    Some(CExpr::call(id, if red.is_some() { "viv_fz_red" } else { "viv_fz" }, args))
}

/// `name = tree` for a fused elementwise tree (crate::fuse) with the
/// element loop written out here, so the compiler sees one loop over plain
/// arrays and vectorizes it, as it does a hand-written loop. The runtime
/// still walks the program first (`viv_fz_prep`: the shape and slice checks,
/// in the unfused order, and the result's block); when the walk finds a
/// broadcast or an unevenly spaced slice, `viv_fz_run` runs the unfused
/// operations, as the call did. Each element is `crate::fuse::element_expr`,
/// the operations `viv_fz_block` does, so the bits are the same.
fn fuse_assign_c(id: NodeId, name: &str, value: NodeId, cx: &mut CEmitCx<Viv>) -> Option<CStmt> {
    if !matches!(cx.program().get(value).kind, Viv::Arr { .. } | Viv::MapMath { .. }) || !crate::fuse::enabled() {
        return None;
    }
    let plan = crate::fuse::plan_tree(cx.program(), value)?;
    let (unit, nl, sliced) = crate::fuse::element_expr(&plan, &|j| format!("viv_fzp{j}[viv_fzk]"))?;
    if nl == 0 || nl >= 24 || plan.arrs.is_empty() {
        return None;
    }
    cx.need("fuse");
    let mut code = plan.code.clone();
    for (at, span) in &plan.sites {
        code[*at] = site_row(*span) as f64;
    }
    let text: Vec<String> = code.iter().map(|v| format!("{}", *v as i64)).collect();
    let fns = if plan.fns.is_empty() { "(const viv_fz_fn*)0".to_string() } else { format!("(const viv_fz_fn[]){{ {} }}", plan.fns.join(", ")) };
    let mut body = Vec::new();
    // the operands, in the order the call took them
    for (k, &a) in plan.arrs.iter().enumerate() {
        let e = cx.expr(a);
        body.push(CStmt::decl(id, CType::Named("viv_arr".into()), format!("viv_fza{k}"), e));
    }
    for (k, &n) in plan.nums.iter().enumerate() {
        let e = cx.expr(n);
        body.push(CStmt::decl(id, CType::Double, format!("viv_fzn{k}"), CExpr::call(id, "(double)", vec![e])));
    }
    let arr_list: Vec<String> = (0..plan.arrs.len()).map(|k| format!("viv_fza{k}")).collect();
    let num_list: Vec<String> = (0..plan.nums.len()).map(|k| format!("viv_fzn{k}")).collect();
    let mut t = String::new();
    t.push_str(&format!("static const double viv_fzc[] = {{ {} }};
", text.join(", ")));
    t.push_str(&format!("const viv_arr viv_fza[] = {{ {} }};
", arr_list.join(", ")));
    t.push_str(&format!("const double viv_fzn[] = {{ {} }};
", if num_list.is_empty() { "0.0".to_string() } else { num_list.join(", ") }));
    t.push_str("viv_fz_v viv_fzl[VIV_FZ_DEPTH];
viv_arr viv_fzr;
");
    t.push_str(&format!("if (viv_fz_prep(viv_fzc, {}, viv_fza, viv_fzn, viv_fzl, &viv_fzr)) {{
", code.len()));
    t.push_str("double* restrict viv_fzo = viv_fzr.data;
long long viv_fzm = viv_fzr.size;
");
    for j in 0..nl {
        t.push_str(&format!("const double* restrict viv_fzp{j} = viv_fzl[{j}].p;
"));
    }
    let unit_loop = format!("for (long long viv_fzk = 0; viv_fzk < viv_fzm; viv_fzk++) viv_fzo[viv_fzk] = {unit};
");
    if sliced {
        let (strided, _, _) = crate::fuse::element_expr(&plan, &|j| format!("viv_fzp{j}[viv_fzk * viv_fzl[{j}].st]"))?;
        let unit_test: Vec<String> = (0..nl).map(|j| format!("viv_fzl[{j}].st == 1")).collect();
        t.push_str(&format!("if ({}) {unit_loop}", unit_test.join(" && ")));
        t.push_str(&format!("else for (long long viv_fzk = 0; viv_fzk < viv_fzm; viv_fzk++) viv_fzo[viv_fzk] = {strided};
"));
    } else {
        t.push_str(&unit_loop);
    }
    t.push_str(&format!("}} else viv_fzr = viv_fz_run(viv_fzc, {}, viv_fza, viv_fzn, {fns})", code.len()));
    body.push(CStmt::expr(id, CExpr::raw(id, t)));
    body.push(CStmt::assign(id, name.to_string(), CExpr::var(id, "viv_fzr")));
    Some(CStmt::block(id, body))
}

/// The part of the preamble that routes errors and output (embed.c), for a
/// test that builds runtime fragments without the rest of a program.
pub const EMBED_C: &str = include_str!("embed.c");

/// Viv's runtime, as on-demand fragments (1.6, A.3). Pure C99, no libstdc++,
/// emitted only as used — discipline for keeping generated/compiler WASM lean.
/// The whole C runtime, every fragment's code in the catalog's order: what
/// `crate::builtins`' tests read the runtime's prototypes from.
pub(crate) fn runtime_c_text() -> String {
    runtime_source()
}

/// `runtime_c_text`, for the tests outside the crate that read the runtime
/// (tests/embed.rs).
pub fn runtime_source() -> String {
    let rt = <Viv as CEmit>::runtime();
    rt.fragments().iter().map(|f| f.code).collect::<Vec<_>>().join("\n")
}

/// The runtime's file-scope state that a split build shares between its
/// translation units: the program's unit
/// defines each of these once and every other unit declares it `extern`.
/// `split_state_covers_the_runtime` fails when the runtime gains a mutable
/// file-scope variable that is in neither this list nor `PER_UNIT_STATE`.
pub(crate) const SHARED_STATE: &[&str] = &[
    "viv_db_conns", "viv_db_nconn", "viv_arena", "viv_arena_top", "viv_arena_cap",
    "viv_pool_p", "viv_pool_sz", "viv_pool_n", "viv_pool_bytes", "viv_live_p", "viv_live_sz", "viv_live_n", "viv_kept", "viv_kept_n", "viv_kept_cap", "viv_en_pass", "viv_fft_plans",
    "viv_depth_g", "viv_dj_t", "viv_dj_k", "viv_dj_gen", "viv_dj_live", "viv_dj_what", "viv_dj_mine", "viv_dj_at", "viv_ad_t",
    "viv_ad_n", "viv_ad_cap", "viv_fast_shadow_on", "viv_fast_shadow_f", "viv_step_ev", "viv_step_total", "viv_story_buf",
    "viv_story_n", "viv_story_cap", "viv_story_total", "viv_story_fails", "viv_story_keep", "viv_story_first", "viv_fails",
    "viv_test", "viv_test_depth", "viv_test_n", "viv_test_fails", "viv_test_detail", "viv_rng_state", "viv_err_site",
    "viv_export_given_v", "viv_export_num_v", "viv_export_int_v", "viv_export_str_v", "viv_export_arr_v",
];

/// Runtime functions a split build keeps one copy of, in the program's unit:
/// the step recorder, which a harness may hook by editing that one
/// definition (study/ports/harness/port.py's stage clocks).
pub(crate) const SHARED_FUNCS: &[&str] = &["viv_step"];

/// The shared function a runtime line starts the definition of.
fn shared_func_head(line: &str) -> Option<&'static str> {
    let rest = line.strip_prefix("static ")?;
    if !line.trim_end().ends_with('{') {
        return None;
    }
    SHARED_FUNCS.iter().copied().find(|f| rest.contains(&format!(" {f}(")))
}

/// File-scope tables each translation unit keeps for itself: the string
/// literals and the source sites its own code numbers.
pub(crate) const PER_UNIT_STATE: &[&str] = &["viv_lits", "viv_sites"];

/// The names a file-scope `static` data line declares, and whether it is a
/// mutable variable (not a `const` table): `None` for any other line.
pub(crate) fn static_data_line(line: &str) -> Option<(Vec<String>, bool)> {
    let rest = line.strip_prefix("static ")?;
    if line.contains('(') || !line.trim_end().ends_with(';') {
        return None;
    }
    // the declarators follow the type: after a struct's closing brace, or
    // else every comma-separated piece's last word before `[` or `=`
    let decl = match rest.rfind('}') {
        Some(k) => &rest[k + 1..],
        None => rest,
    };
    let decl = decl.trim_end().trim_end_matches(';');
    let mut names = Vec::new();
    let mut constant = rest.starts_with("const ") && !rest.contains('*');
    for (i, piece) in decl.split(',').enumerate() {
        let lhs = piece.split('=').next().unwrap_or("");
        let lhs = lhs.split('[').next().unwrap_or("").trim();
        let word = lhs.rsplit(|c: char| c.is_whitespace() || c == '*').next().unwrap_or("");
        if i == 0 && lhs.contains("* const") {
            constant = true;
        }
        if !word.is_empty() {
            names.push(word.to_string());
        }
    }
    if names.is_empty() {
        return None;
    }
    Some((names, !constant))
}

/// A runtime fragment's code for a translation unit of `role`: unchanged in
/// one file; in a split build, each line of shared state (`SHARED_STATE`) a
/// definition without `static` in the program's unit and `extern`
/// declarations, without initializers, in every other unit.
fn split_state(code: &'static str, role: knotkit::backend::UnitRole) -> std::borrow::Cow<'static, str> {
    use knotkit::backend::UnitRole;
    if role == UnitRole::Single
        || !code.lines().any(|l| shared_func_head(l).is_some() || static_data_line(l).is_some_and(|(n, m)| m && n.iter().any(|x| SHARED_STATE.contains(&x.as_str()))))
    {
        return std::borrow::Cow::Borrowed(code);
    }
    let mut out = String::with_capacity(code.len());
    let mut skipping = false;
    for line in code.split_inclusive('\n') {
        if skipping {
            // a shared function's body, which only the program's unit keeps
            if line.trim_end() == "}" {
                skipping = false;
            }
            continue;
        }
        if shared_func_head(line).is_some() {
            let head = line.trim_end().strip_prefix("static ").unwrap_or(line);
            if role == UnitRole::Main {
                out.push_str(head);
                out.push('\n');
            } else {
                out.push_str(head.trim_end_matches('{').trim_end());
                out.push_str(";\n");
                skipping = true;
            }
            continue;
        }
        let shared = static_data_line(line).filter(|(n, m)| *m && n.iter().all(|x| SHARED_STATE.contains(&x.as_str())));
        let Some((names, _)) = shared else {
            out.push_str(line);
            continue;
        };
        let body = line.trim_end().strip_prefix("static ").unwrap_or(line);
        if role == UnitRole::Main {
            out.push_str(body);
            out.push('\n');
            continue;
        }
        // the type is what precedes the first declarator; each name keeps its
        // own array bounds and pointer stars
        let decl_start = match body.rfind('}') {
            Some(k) => k + 1,
            None => {
                let first = body.split(',').next().unwrap_or(body);
                let lhs = first.split('=').next().unwrap_or(first).split('[').next().unwrap_or(first);
                let name = &names[0];
                let at = lhs.rfind(name.as_str()).unwrap_or(0);
                // back over the stars that belong to the declarator
                lhs[..at].trim_end_matches('*').len()
            }
        };
        let ty = body[..decl_start].trim_end();
        let decls = body[decl_start..].trim_end().trim_end_matches(';');
        for piece in decls.split(',') {
            let lhs = piece.split('=').next().unwrap_or(piece).trim();
            out.push_str(&format!("extern {ty} {lhs};\n"));
        }
    }
    std::borrow::Cow::Owned(out)
}

/// The C runtime of `disjoint` pins (fragment "disjoint"): see its comment.
const DJ_RUNTIME: &str = r#"/* `disjoint` pins (crate::memory_lens): a
   memory lens may say that the passes of a parallel for write disjoint
   elements of an array, for a reason the compiler cannot see. A checked
   build tests it: before the loop each pinned array gets a table of which
   pass wrote each element (-1: none yet), every element write in the loop's
   body and in the defs it reaches goes through viv_dj_set*, and a second
   pass writing an element stops the run naming both passes. The table entry
   is claimed with a compare-and-swap, so threads racing for one element are
   caught too. -DVIV_RELEASE trusts the pin: the writes are plain stores. */
#ifndef VIV_RELEASE
typedef struct { const double* data; int rank; int dims[8]; const char* name; long* owner; } viv_dj_arr;
static viv_dj_arr viv_dj_t[8];
static int viv_dj_k = 0;
static long viv_dj_gen = 0, viv_dj_live = 0;
static const char* viv_dj_what = "";
static VIV_TLS long viv_dj_mine = 0, viv_dj_at = -1;
#endif
static void viv_dj_begin(const char* what) {
#ifndef VIV_RELEASE
    viv_dj_what = what;
    viv_dj_k = 0;
    viv_dj_live = ++viv_dj_gen;
#else
    (void)what;
#endif
}
static void viv_dj_track(viv_arr a, const char* name) {
#ifndef VIV_RELEASE
    if (viv_dj_k == 8) return;
    viv_dj_arr* t = &viv_dj_t[viv_dj_k++];
    t->data = a.data; t->rank = a.rank; t->name = name;
    for (int d = 0; d < 8; d++) t->dims[d] = a.dims[d];
    t->owner = (long*)malloc(sizeof(long) * (size_t)(a.size ? a.size : 1));
    if (!t->owner) { viv_fflush(stdout); viv_errf("error: out of memory checking a disjoint pin\n"); viv_stop(VIV_E_MEMORY); }
    for (int e = 0; e < a.size; e++) t->owner[e] = -1;
#else
    (void)a; (void)name;
#endif
}
static void viv_dj_pass(double p) {
#ifndef VIV_RELEASE
    viv_dj_mine = viv_dj_live;
    viv_dj_at = (long)p;
#else
    (void)p;
#endif
}
#ifndef VIV_RELEASE
VIV_COLD static void viv_dj_fail(const viv_dj_arr* t, long e, long a, long b) {
    char ix[160]; int n = 0;
    long rest = e;
    long part[8];
    int rank = t->rank < 1 ? 1 : t->rank;
    for (int d = rank - 1; d >= 0; d--) {
        long dim = t->rank < 1 ? 1 : t->dims[d];
        part[d] = dim > 0 ? rest % dim : 0;
        rest = dim > 0 ? rest / dim : 0;
    }
    for (int d = 0; d < rank && n < 120; d++) n += snprintf(ix + n, sizeof ix - (size_t)n, "%s%ld", d ? ", " : "", part[d]);
    viv_fflush(stdout);
    viv_errf("error: passes %ld and %ld both write %s[%s] in %s\n", a < b ? a : b, a < b ? b : a, t->name, ix, viv_dj_what);
    viv_stop(VIV_E_ERROR);
}
#endif
/* whether `d` is a pinned array's storage: tested inline before every write
   in a pinned loop's reach, so a write to any other array (a def's scratch,
   most of them) costs two loads and a compare, and only a write to a pinned
   array goes out of line to be recorded */
VIV_ACCESS int viv_dj_hit(const double* d) {
#ifndef VIV_RELEASE
    for (int k = 0; k < viv_dj_k; k++) if (viv_dj_t[k].data == d) return 1;
#else
    (void)d;
#endif
    return 0;
}
VIV_COLD static void viv_dj_note(viv_arr a, int e) {
#ifndef VIV_RELEASE
    if (viv_dj_live == 0 || viv_dj_mine != viv_dj_live) return;
    for (int k = 0; k < viv_dj_k; k++) {
        if (viv_dj_t[k].data != a.data) continue;
        long seen = -1;
        if (!__atomic_compare_exchange_n(&viv_dj_t[k].owner[e], &seen, viv_dj_at, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED) && seen != viv_dj_at)
            viv_dj_fail(&viv_dj_t[k], e, seen, viv_dj_at);
        return;
    }
#else
    (void)a; (void)e;
#endif
}
static void viv_dj_end(void) {
#ifndef VIV_RELEASE
    for (int k = 0; k < viv_dj_k; k++) free(viv_dj_t[k].owner);
    viv_dj_k = 0;
    viv_dj_live = 0;
#endif
}
VIV_ACCESS void viv_dj_set1_at(viv_arr a, double i, double v, int site) {
#ifndef VIV_RELEASE
    viv_rank(a.rank, 1, site);
    int e = viv_ix(i, a.dims[0], 0, site);
    if (viv_dj_live && viv_dj_hit(a.data)) viv_dj_note(a, e);
    a.data[e] = v;
#else
    viv_set1_at(a, i, v, site);
#endif
}
VIV_ACCESS void viv_dj_set2_at(viv_arr a, double i, double j, double v, int site) {
#ifndef VIV_RELEASE
    viv_rank(a.rank, 2, site);
    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site);
    int e = ii * a.dims[1] + jj;
    if (viv_dj_live && viv_dj_hit(a.data)) viv_dj_note(a, e);
    a.data[e] = v;
#else
    viv_set2_at(a, i, j, v, site);
#endif
}
VIV_ACCESS void viv_dj_set3_at(viv_arr a, double i, double j, double k, double v, int site) {
#ifndef VIV_RELEASE
    viv_rank(a.rank, 3, site);
    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site);
    int e = (ii * a.dims[1] + jj) * a.dims[2] + kk;
    if (viv_dj_live && viv_dj_hit(a.data)) viv_dj_note(a, e);
    a.data[e] = v;
#else
    viv_set3_at(a, i, j, k, v, site);
#endif
}
VIV_ACCESS void viv_dj_set4_at(viv_arr a, double i, double j, double k, double l, double v, int site) {
#ifndef VIV_RELEASE
    viv_rank(a.rank, 4, site);
    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site); int ll = viv_ix(l, a.dims[3], 3, site);
    int e = ((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll;
    if (viv_dj_live && viv_dj_hit(a.data)) viv_dj_note(a, e);
    a.data[e] = v;
#else
    viv_set4_at(a, i, j, k, l, v, site);
#endif
}
VIV_ACCESS void viv_dj_set1(viv_arr a, double i, double v) { viv_dj_set1_at(a, i, v, 0); }
VIV_ACCESS void viv_dj_set2(viv_arr a, double i, double j, double v) { viv_dj_set2_at(a, i, j, v, 0); }
VIV_ACCESS void viv_dj_set3(viv_arr a, double i, double j, double k, double v) { viv_dj_set3_at(a, i, j, k, v, 0); }
VIV_ACCESS void viv_dj_set4(viv_arr a, double i, double j, double k, double l, double v) { viv_dj_set4_at(a, i, j, k, l, v, 0); }"#;

/// Fast mode's BLAS and LAPACK. With
/// `-DVIV_FAST -DVIV_BLAS` (what `viv build --fast` passes when it finds a
/// BLAS), `@` on a large product with a finite right operand is dgemm, and
/// `lin_solve` / `lin_solve_multi` on a matrix of order 32 or more are dgetrf
/// and dgetrs. Without VIV_BLAS, or in an exact build, this is empty. On macOS
/// the BLAS is Accelerate; elsewhere a CBLAS with the Fortran LAPACK symbols
/// (OpenBLAS: `-lopenblas`).
const FAST_LA: &str = r#"#if defined(VIV_FAST) && defined(VIV_BLAS)
#define VIV_FAST_LA 1
/* The BLAS and LAPACK entry points, declared here rather than through
   <Accelerate/Accelerate.h> or <cblas.h>: those headers bring names such as
   `wide`, `Fixed` and `Boolean` (MacTypes.h) that a program's own variables
   would collide with. Accelerate exports these symbols with these types, and
   so does OpenBLAS. CBLAS's enums are passed as their values. */
typedef int viv_la_int;
#define VIV_CBLAS_ROW_MAJOR 101
#define VIV_CBLAS_NO_TRANS 111
void cblas_dgemm(int order, int transa, int transb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc);
#ifdef __APPLE__
void dgetrf_(const viv_la_int* m, const viv_la_int* n, double* a, const viv_la_int* lda, viv_la_int* ipiv, viv_la_int* info);
void dgecon_(const char* norm, const viv_la_int* n, const double* a, const viv_la_int* lda, const double* anorm, double* rcond, double* work, viv_la_int* iwork, viv_la_int* info);
void dgetrs_(const char* trans, const viv_la_int* n, const viv_la_int* nrhs, const double* a, const viv_la_int* lda, const viv_la_int* ipiv, double* b, const viv_la_int* ldb, viv_la_int* info);
#define VIV_LA_STRLEN
#else
/* a Fortran LAPACK takes each character argument's length after the others */
#include <stddef.h>
void dgetrf_(const viv_la_int* m, const viv_la_int* n, double* a, const viv_la_int* lda, viv_la_int* ipiv, viv_la_int* info);
void dgecon_(const char* norm, const viv_la_int* n, const double* a, const viv_la_int* lda, const double* anorm, double* rcond, double* work, viv_la_int* iwork, viv_la_int* info, size_t norm_len);
void dgetrs_(const char* trans, const viv_la_int* n, const viv_la_int* nrhs, const double* a, const viv_la_int* lda, const viv_la_int* ipiv, double* b, const viv_la_int* ldb, viv_la_int* info, size_t trans_len);
#define VIV_LA_STRLEN , (size_t)1
#endif
/* the smallest m*k*n a dgemm call is made for, and the smallest order of a
   LAPACK solve: below them Viv's own loops are as fast */
#define VIV_FAST_MM_MIN 32768LL
#define VIV_FAST_SOLVE_MIN 32
static int viv_fast_finite(const double* x, long long n) {
    for (long long i = 0; i < n; i++) if (!(x[i] - x[i] == 0.0)) return 0;
    return 1;
}
/* VIV_FAST_SHADOW=<file> (what `viv check --fast` sets): each BLAS or LAPACK
   call also measures itself against its a-priori error bound and appends a
   line to the file. Nothing the program prints changes. */
static int viv_fast_shadow_on = -1;
static FILE* viv_fast_shadow_f = 0;
static FILE* viv_fast_shadow(void) {
    if (viv_fast_shadow_on < 0) {
        const char* p = getenv("VIV_FAST_SHADOW");
        viv_fast_shadow_f = (p && *p) ? fopen(p, "a") : 0;
        viv_fast_shadow_on = viv_fast_shadow_f != 0;
    }
    return viv_fast_shadow_f;
}
/* A X = B for row-major A (n x n), B and X (n x m), by LAPACK. Returns 0, and
   leaves the solve to Viv's own elimination, for a small or non-finite
   system, and for one whose smallest pivot is within a factor 100 of the
   threshold at which Viv's elimination calls the matrix singular: that
   elimination then decides, so the run stops (or not) where the exact build
   does. The row-major A is A^T to LAPACK, so the solve is A^T's transpose. */
static int viv_fast_solve(int n, const double* A, const double* B, int m, double* X, double scale) {
    if (n < VIV_FAST_SOLVE_MIN || !viv_fast_finite(A, (long long)n * n) || !viv_fast_finite(B, (long long)n * m)) return 0;
    double* lu = (double*)malloc((size_t)n * n * sizeof(double));
    double* bc = (double*)malloc((size_t)n * m * sizeof(double));
    viv_la_int* ipiv = (viv_la_int*)malloc((size_t)n * sizeof(viv_la_int));
    if (!lu || !bc || !ipiv) { free(lu); free(bc); free(ipiv); return 0; }
    for (long long i = 0; i < (long long)n * n; i++) lu[i] = A[i];
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) bc[(size_t)j * n + i] = B[(size_t)i * m + j];
    viv_la_int nn = n, mm = m, info = 0;
    dgetrf_(&nn, &nn, lu, &nn, ipiv, &info);
    int ok = info == 0;
    for (int k = 0; ok && k < n; k++) if (fabs(lu[(size_t)k * n + k]) < 1e-12 * scale) ok = 0;
    if (ok) {
        dgetrs_("T", &nn, &mm, lu, &nn, ipiv, bc, &nn, &info VIV_LA_STRLEN);
        ok = info == 0;
    }
    if (ok) for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) X[(size_t)i * m + j] = bc[(size_t)j * n + i];
    FILE* sh = ok ? viv_fast_shadow() : 0;
    if (sh) {
        /* the condition number in the infinity norm (LAPACK's 1-norm of the
           A^T it factored) and the forward-error bound 4 n u kappa that the
           exact and the fast solutions each meet, so their difference too */
        double anorm = 0.0;
        for (int i = 0; i < n; i++) { double r = 0.0; for (int j = 0; j < n; j++) r += fabs(A[(size_t)i * n + j]); if (r > anorm) anorm = r; }
        double rcond = 0.0;
        double* work = (double*)malloc((size_t)4 * n * sizeof(double));
        viv_la_int* iwork = (viv_la_int*)malloc((size_t)n * sizeof(viv_la_int));
        if (work && iwork) dgecon_("1", &nn, lu, &nn, &anorm, &rcond, work, iwork, &info VIV_LA_STRLEN);
        free(work); free(iwork);
        double kappa = rcond > 0.0 ? 1.0 / rcond : INFINITY;
        fprintf(sh, "solve n=%d nrhs=%d kappa=%.3e bound=%.3e\n", n, m, kappa, 4.0 * n * 0x1p-53 * kappa);
        fflush(sh);
    }
    free(lu); free(bc); free(ipiv);
    return ok;
}
#endif"#;

fn base_runtime() -> Runtime {
        Runtime::new()
            .with(RuntimeFragment {
                name: "fmt_g",
                includes: &["#include <stdio.h>", "#include <string.h>", "#include <float.h>"],
                deps: &[],
                code: crate::printfmt::C_FMT_G,
            })
            .with(RuntimeFragment {
                name: "print",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <stdarg.h>", "#include <float.h>", "#include <math.h>"],
                deps: &["arr_type", "fmt_g"],
                code: crate::printfmt::C_RUNTIME,
            })
            .with(RuntimeFragment {
                name: "print_num",
                includes: &["#include <stdio.h>"],
                deps: &["fmt_g"],
                code: "static void viv_print_num(double x) {\n    viv_printf(\"%s\\n\", viv_g(x));\n}",
            })
            .with(RuntimeFragment {
                name: "text",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type"],
                code: "static void viv_text(viv_arr a) {\n    /* Unicode scalar values out as UTF-8, matching the interpreter byte for\n       byte: a surrogate is not a scalar value, so it comes out as U+FFFD in\n       both, and a newline is escaped so that one record stays one line. */\n    viv_fputs(\"text: \", stdout);\n    for (int i = 0; i < a.size; i++) {\n        long code = (long)a.data[i];\n        if (code < 0) code = 0;\n        if (code > 0x10FFFF) code = 0x10FFFF;\n        if (code == 10) { viv_fputs(\"\\\\n\", stdout); continue; }\n        if (code == 9) { viv_fputs(\"\\\\t\", stdout); continue; }\n        unsigned int c = (unsigned int)code;\n        if (c >= 0xD800u && c <= 0xDFFFu) c = 0xFFFDu;\n        if (c < 0x80u) {\n            viv_putchar((int)c);\n        } else if (c < 0x800u) {\n            viv_putchar((int)(0xC0u | (c >> 6)));\n            viv_putchar((int)(0x80u | (c & 0x3Fu)));\n        } else if (c < 0x10000u) {\n            viv_putchar((int)(0xE0u | (c >> 12)));\n            viv_putchar((int)(0x80u | ((c >> 6) & 0x3Fu)));\n            viv_putchar((int)(0x80u | (c & 0x3Fu)));\n        } else {\n            viv_putchar((int)(0xF0u | (c >> 18)));\n            viv_putchar((int)(0x80u | ((c >> 12) & 0x3Fu)));\n            viv_putchar((int)(0x80u | ((c >> 6) & 0x3Fu)));\n            viv_putchar((int)(0x80u | (c & 0x3Fu)));\n        }\n    }\n    viv_putchar('\\n');\n}",
            })
            // Math builtins call libm directly; this fragment just pulls the header.
            .with(RuntimeFragment { name: "libm", includes: &["#include <math.h>"], deps: &[], code: "" })
            // fma: musl's integer fma (vmath_fma.c), only in programs that call it
            .with(RuntimeFragment { name: "fma", includes: &["#include <stdint.h>", "#include <float.h>"], deps: &[], code: include_str!("vmath_fma.c") })
            .with(RuntimeFragment {
                name: "arr_type",
                includes: &[],
                deps: &[],
                code: "typedef struct { int rank; int dims[8]; int size; double* data; } viv_arr;",
            })
            .with(RuntimeFragment {
                name: "arr_new",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <unistd.h>"],
                deps: &["arr_type", "arena"],
                code: "/* Every array's block is made here. The element count is multiplied out in\n   64 bits and may not pass INT_MAX (the struct's `size`): an index that\n   passed its dimension's check then gives an offset below `size`, so no\n   offset overflows. `zeros` and the array vocabulary refuse an\n   array past that first, with their own words (2147483647 elements, as the\n   interpreter does natively; the wasm engines stop at 16,000,000); this is\n   the last check. A failed calloc stops the run instead of handing out a null\n   block. `width` is 1 for numbers, 2 for complex numbers. */\nstatic void viv_arr_fail(int kind, const char* m, long long n) {\n    viv_fflush(stdout);\n    viv_errf(m, n);\n    viv_stop(kind);\n}\nstatic viv_arr viv_arr_alloc_z(int rank, const int* dims, int width, int zero) {\n    viv_arr a; a.rank = rank; a.dims[7] = 0;\n    long long n = 1;\n    for (int i = 0; i < rank; i++) {\n        a.dims[i] = dims[i] < 0 ? 0 : dims[i];\n        n *= a.dims[i];\n        if (n > 2147483647LL) viv_arr_fail(VIV_E_OVERFLOW, \"error: an array of more than %lld elements cannot be made\\n\", 2147483647LL);\n    }\n    a.size = (int)n;\n    a.data = (double*)(zero ? viv_calloc_pooled((size_t)n * (size_t)width, sizeof(double)) : viv_malloc_pooled((size_t)n * (size_t)width * sizeof(double)));\n    if (!a.data && n > 0) viv_arr_fail(VIV_E_MEMORY, \"error: out of memory making an array of %lld elements\\n\", n);\n    viv_arena_push(a.data);\n    return a;\n}\nstatic viv_arr viv_arr_alloc(int rank, const int* dims, int width) { return viv_arr_alloc_z(rank, dims, width, 1); }\nstatic viv_arr viv_arr_new(int rank, const int* dims) { return viv_arr_alloc(rank, dims, 1); }\n/* an array whose every element the caller writes before anything reads it */\nstatic viv_arr viv_arr_new_nz(int rank, const int* dims) { return viv_arr_alloc_z(rank, dims, 1, 0); }\n/* the dimensions `zeros` and `czeros` are given, read as the interpreter\n   reads them (interp.rs, zeros_shape): a negative or NaN one is 0, a\n   fraction is cut, a huge one saturates; the element count, a saturating\n   64-bit product, is checked against the limit before anything is made */\nstatic void viv_zeros_dims(int rank, const double* x, int* d, int width) {\n    unsigned long long total = 1;\n    for (int i = 0; i < rank; i++) {\n        double v = x[i];\n        unsigned long long u = !(v > 0) ? 0 : v >= 18446744073709551615.0 ? 18446744073709551615ULL : (unsigned long long)v;\n        total = (u != 0 && total > 18446744073709551615ULL / u) ? 18446744073709551615ULL : total * u;\n        d[i] = u > 2147483647ULL ? 2147483647 : (int)u;\n    }\n    if (width == 1 && total > 2147483647ULL) {\n        viv_fflush(stdout);\n        viv_errf(\"error: zeros: array of %llu elements exceeds the 2147483647-element limit\\n\", total);\n        viv_stop(VIV_E_OVERFLOW);\n    }\n    if (width == 2 && total > 1073741823ULL) {\n        viv_fflush(stdout);\n        viv_errf(\"error: czeros: complex array of %llu elements exceeds the 1073741823-element limit\\n\", total);\n        viv_stop(VIV_E_OVERFLOW);\n    }\n}",
            })
            .with(RuntimeFragment {
                name: "vec",
                includes: &["#include <stdarg.h>"],
                deps: &["arr_new"],
                code: "static viv_arr viv_vec(int n, ...) {\n    int dims[1]; dims[0] = n;\n    viv_arr a = viv_arr_new(1, dims);\n    va_list ap; va_start(ap, n);\n    for (int i = 0; i < n; i++) a.data[i] = va_arg(ap, double);\n    va_end(ap);\n    return a;\n}",
            })
            // A long constant literal is copied out of a C99 compound literal
            // instead of passed through a variadic call. clang's instruction
            // selection is superlinear in the number of call arguments: 18,145
            // varargs took 11 s at -O2, and the same values as an initializer
            // list take well under a second.
            .with(RuntimeFragment {
                name: "vec_tab",
                includes: &["#include <string.h>"],
                deps: &["arr_new"],
                code: "static viv_arr viv_vec_tab(int n, const double *t) {\n    int dims[1]; dims[0] = n;\n    viv_arr a = viv_arr_new(1, dims);\n    memcpy(a.data, t, (size_t)n * sizeof(double));\n    return a;\n}",
            })
            .with(RuntimeFragment {
                name: "mat_tab",
                includes: &["#include <string.h>"],
                deps: &["arr_new"],
                code: "static viv_arr viv_mat_tab(int rows, int cols, const double *t) {\n    int dims[2]; dims[0] = rows; dims[1] = cols;\n    viv_arr a = viv_arr_new(2, dims);\n    memcpy(a.data, t, (size_t)rows * (size_t)cols * sizeof(double));\n    return a;\n}",
            })
            .with(RuntimeFragment {
                name: "mat",
                includes: &["#include <stdarg.h>"],
                deps: &["arr_new"],
                code: "static viv_arr viv_mat(int rows, int cols, ...) {\n    int dims[2]; dims[0] = rows; dims[1] = cols;\n    viv_arr a = viv_arr_new(2, dims);\n    int total = rows * cols;\n    va_list ap; va_start(ap, cols);\n    for (int i = 0; i < total; i++) a.data[i] = va_arg(ap, double);\n    va_end(ap);\n    return a;\n}",
            })
            .with(RuntimeFragment { name: "arena", includes: &["#include <stdlib.h>", "#include <string.h>"], deps: &["arr_type"], code: "/* The array arena: every array allocation is pushed on a stack. A function\n   that can allocate notes the height on entry and frees, on return, everything\n   above it except the array it returns (which is re-filed under the caller).\n   Arrays made at the top level sit below every mark and live for the run.\n   With OpenMP each thread has its own stack, so a function called inside a\n   parallel loop frees only what that thread allocated. */\n#ifdef _OPENMP\n#define VIV_TLS __thread\n#else\n#define VIV_TLS\n#endif\nstatic VIV_TLS void** viv_arena = 0;\nstatic VIV_TLS size_t viv_arena_top = 0, viv_arena_cap = 0;\nstatic void viv_arena_push(void* p) {\n    if (viv_arena_top == viv_arena_cap) {\n        viv_arena_cap = viv_arena_cap ? 2 * viv_arena_cap : 256;\n        viv_arena = (void**)realloc(viv_arena, viv_arena_cap * sizeof(void*));\n    }\n    viv_arena[viv_arena_top++] = p;\n}\n/* A map is not one block: it is filed with a tag in its pointer's low bits\n   (1 a map of numbers, 2 a map with strings; every block malloc gives is\n   aligned past them) and freed by the function its runtime puts in\n   viv_arena_drop. */\nstatic void (*viv_arena_drop[4])(void*);\n/* An enum node is filed with tag 3 and\n   begins with this head: its place on the arena, and the number of the last\n   release pass that kept it. A pass keeps a node by marking it, so a release\n   asks the node instead of searching the kept list, and a walk stops at a\n   node it has marked, or at one below the mark it releases to. */\ntypedef struct { int tag; unsigned keep; long pos; } viv_en_head;\n/* The block pool: an array's block that is released is kept for the next\n   array of about its size, instead of going back to malloc. A program that\n   makes an array each pass of a loop (a fused expression, `@`, a gradient's\n   adjoints) then reuses the same few blocks: no malloc and free per pass, and\n   for a large block no fresh pages to fault in and no pages handed back to\n   the system on each free. A reused block is zeroed as calloc's is, so\n   nothing a program can see changes. Under AddressSanitizer (which must see\n   every free) there is none, and -DVIV_NO_POOL turns it off. It holds at\n   most VIV_POOL_BYTES bytes. With OpenMP each thread has its own. */\n#if defined(__has_feature)\n#if __has_feature(address_sanitizer)\n#define VIV_NO_POOL 1\n#endif\n#endif\n#if defined(__SANITIZE_ADDRESS__) && !defined(VIV_NO_POOL)\n#define VIV_NO_POOL 1\n#endif\n/* posix_memalign is POSIX's: without it (Windows, a wasm build) no pool */\n#if !defined(VIV_NO_POOL) && (defined(_WIN32) || defined(__EMSCRIPTEN__) || defined(__wasm__))\n#define VIV_NO_POOL 1\n#endif\n#ifdef VIV_NO_POOL\n#define VIV_POOL_ON 0\n#define viv_pool_aligned(pp, bytes) ((void)(pp), (void)(bytes), 1)\n#else\n#define VIV_POOL_ON 1\n#define viv_pool_aligned(pp, bytes) posix_memalign((pp), VIV_POOL_ALIGN, (bytes))\n#endif\n/* blocks are kept by size class (class k: sizes in [2^k, 2^(k+1))), each a\n   stack of at most VIV_POOL_N; a request of class k takes a block of class k\n   that fits, else the newest of class k + 1. A block of VIV_POOL_MIN bytes\n   or more that an array is made with is entered, with its size, in a table\n   of live blocks (open addressing), which is how a release knows its size\n   and that it may be kept; a block not in the table is freed. Such a block\n   is aligned to VIV_POOL_ALIGN, so a release of any other block (nearly\n   always unaligned) is freed without looking in the table. Smaller blocks\n   go straight to malloc, whose small-block path is as quick. */\n#define VIV_POOL_N 32\n#define VIV_POOL_MIN 65536\n#define VIV_POOL_ALIGN 4096\n#define VIV_POOL_CLASSES 40\n#define VIV_POOL_BYTES ((size_t)128 << 20)\n#define VIV_POOL_LIVE 8192\nstatic VIV_TLS void* viv_pool_p[VIV_POOL_CLASSES][VIV_POOL_N];\nstatic VIV_TLS size_t viv_pool_sz[VIV_POOL_CLASSES][VIV_POOL_N];\nstatic VIV_TLS int viv_pool_n[VIV_POOL_CLASSES];\nstatic VIV_TLS size_t viv_pool_bytes = 0;\nstatic VIV_TLS void* viv_live_p[VIV_POOL_LIVE];\nstatic VIV_TLS size_t viv_live_sz[VIV_POOL_LIVE];\nstatic VIV_TLS int viv_live_n = 0;\nstatic size_t viv_live_slot(const void* p) { return (size_t)((((unsigned long long)(size_t)p >> 4) * 0x9E3779B97F4A7C15ULL) >> 51) & (VIV_POOL_LIVE - 1); }\n/* enter a live block; a full table (three quarters) leaves it out */\nstatic void viv_live_add(void* p, size_t sz) {\n    if (!VIV_POOL_ON || !p || viv_live_n >= VIV_POOL_LIVE / 4 * 3) return;\n    size_t i = viv_live_slot(p);\n    while (viv_live_p[i]) {\n        if (viv_live_p[i] == p) { viv_live_sz[i] = sz; return; }\n        i = (i + 1) & (VIV_POOL_LIVE - 1);\n    }\n    viv_live_p[i] = p;\n    viv_live_sz[i] = sz;\n    viv_live_n++;\n}\n/* take a block out of the table: its size, or 0 when it is not there */\nstatic size_t viv_live_take(void* p) {\n    if (viv_live_n == 0) return 0;\n    size_t i = viv_live_slot(p);\n    while (viv_live_p[i] && viv_live_p[i] != p) i = (i + 1) & (VIV_POOL_LIVE - 1);\n    if (!viv_live_p[i]) return 0;\n    size_t sz = viv_live_sz[i];\n    viv_live_p[i] = 0;\n    viv_live_n--;\n    /* close the gap: move up each later entry of the run whose home slot\n       the gap now separates it from */\n    size_t j = i;\n    for (;;) {\n        j = (j + 1) & (VIV_POOL_LIVE - 1);\n        if (!viv_live_p[j]) break;\n        size_t h = viv_live_slot(viv_live_p[j]);\n        if ((i <= j) ? (i < h && h <= j) : (i < h || h <= j)) continue;\n        viv_live_p[i] = viv_live_p[j];\n        viv_live_sz[i] = viv_live_sz[j];\n        viv_live_p[j] = 0;\n        i = j;\n    }\n    return sz;\n}\nstatic int viv_pool_class(size_t sz) {\n    int k = 0;\n    while (k < VIV_POOL_CLASSES - 1 && (sz >> (k + 1)) != 0) k++;\n    return k;\n}\n/* a block of `bytes` from the pool, zeroed if asked, entered as live, or 0:\n   the most recently kept block that fits, the one most likely in the cache */\nstatic void* viv_pool_take(size_t bytes, int zero) {\n    if (!VIV_POOL_ON || bytes < VIV_POOL_MIN) return 0;\n    int k = viv_pool_class(bytes);\n    for (int c = k; c <= k + 1 && c < VIV_POOL_CLASSES; c++) {\n        for (int i = viv_pool_n[c] - 1; i >= 0; i--) {\n            size_t sz = viv_pool_sz[c][i];\n            if (sz < bytes) continue;\n            void* p = viv_pool_p[c][i];\n            viv_pool_bytes -= sz;\n            viv_pool_n[c]--;\n            for (int j = i; j < viv_pool_n[c]; j++) { viv_pool_p[c][j] = viv_pool_p[c][j + 1]; viv_pool_sz[c][j] = viv_pool_sz[c][j + 1]; }\n            if (zero) memset(p, 0, bytes);\n            viv_live_add(p, sz);\n            return p;\n        }\n    }\n    return 0;\n}\n/* keep a live block, or free it (not live, too large, its class full, or\n   the pool at its bytes) */\nstatic void viv_pool_put(void* p) {\n    if (!p) return;\n    if ((size_t)p & (VIV_POOL_ALIGN - 1)) { free(p); return; }\n    size_t sz = viv_live_take(p);\n    int k = viv_pool_class(sz);\n    if (sz < VIV_POOL_MIN || sz > VIV_POOL_BYTES / 8 || viv_pool_n[k] == VIV_POOL_N || viv_pool_bytes + sz > VIV_POOL_BYTES) { free(p); return; }\n    viv_pool_p[k][viv_pool_n[k]] = p;\n    viv_pool_sz[k][viv_pool_n[k]] = sz;\n    viv_pool_n[k]++;\n    viv_pool_bytes += sz;\n}\nstatic void* viv_calloc_pooled(size_t n, size_t each) {\n    size_t bytes = n * each;\n    void* p = viv_pool_take(bytes, 1);\n    if (p) return p;\n    if (!VIV_POOL_ON || bytes < VIV_POOL_MIN || viv_pool_aligned(&p, bytes) != 0) return calloc(n, each);\n    memset(p, 0, bytes);\n    viv_live_add(p, bytes);\n    return p;\n}\n/* a block whose every byte the caller writes before anything reads it */\nstatic void* viv_malloc_pooled(size_t bytes) {\n    void* p = viv_pool_take(bytes, 0);\n    if (p) return p;\n    if (!VIV_POOL_ON || bytes < VIV_POOL_MIN || viv_pool_aligned(&p, bytes) != 0) return malloc(bytes ? bytes : 1);\n    viv_live_add(p, bytes);\n    return p;\n}\nstatic void viv_arena_free(void* p) {\n    size_t tag = (size_t)p & 3;\n    if (tag == 3) free((char*)p - 3);\n    else if (tag) viv_arena_drop[tag]((char*)p - tag);\n    else viv_pool_put(p);\n}\nstatic void viv_arena_release(size_t mark, const void* keep) {\n    size_t w = mark;\n    for (size_t i = mark; i < viv_arena_top; i++) {\n        if (viv_arena[i] == keep) viv_arena[w++] = viv_arena[i];\n        else viv_arena_free(viv_arena[i]);\n    }\n    viv_arena_top = w;\n}" })
            .with(enums::fragment())
            .with(RuntimeFragment { name: "arena_keep", includes: &["#include <stdlib.h>"], deps: &["arena"], code: "/* The end of a loop iteration: the arrays the loop's\n   variables hold are kept, everything else filed since the loop began is\n   freed. So a loop allocating each time round runs in the memory of one\n   iteration, as it does in the direct wasm runtime (vrt_release_kept). A def\n   returning a record keeps the record's arrays the same way. */\nstatic VIV_TLS const void** viv_kept = 0;\nstatic VIV_TLS size_t viv_kept_n = 0, viv_kept_cap = 0;\nstatic void viv_arena_keep(const void* p) {\n    if (viv_kept_n == viv_kept_cap) {\n        viv_kept_cap = viv_kept_cap ? 2 * viv_kept_cap : 16;\n        viv_kept = (const void**)realloc((void*)viv_kept, viv_kept_cap * sizeof(void*));\n    }\n    viv_kept[viv_kept_n++] = p;\n}\n/* the number of this release pass; an enum node marked with it is kept */\nstatic VIV_TLS unsigned viv_en_pass = 1;\nstatic void viv_arena_release_kept(size_t mark) {\n    VIV_POLL();\n    size_t w = mark;\n    for (size_t i = mark; i < viv_arena_top; i++) {\n        int keep = 0;\n        if (((size_t)viv_arena[i] & 3) == 3) {\n            viv_en_head* h = (viv_en_head*)((char*)viv_arena[i] - 3);\n            if (h->keep == viv_en_pass) { keep = 1; h->pos = (long)w; }\n        } else {\n            for (size_t k = 0; k < viv_kept_n; k++) {\n                if (viv_kept[k] == viv_arena[i]) { keep = 1; break; }\n            }\n        }\n        if (keep) viv_arena[w++] = viv_arena[i];\n        else viv_arena_free(viv_arena[i]);\n    }\n    viv_arena_top = w;\n    viv_kept_n = 0;\n    viv_en_pass++;\n}" })
            .with(RuntimeFragment { name: "par_arena", includes: &[], deps: &["arena_keep"], code: "/* A parallel for whose body allocates: under\n   OpenMP each pass releases what it made on the thread that ran it; without\n   OpenMP the loop is released as an ordinary loop is. */\n#ifdef _OPENMP\n#define VIV_OMP 1\n#else\n#define VIV_OMP 0\n#endif" })
            .with(RuntimeFragment { name: "recs", includes: &["#include <stdlib.h>", "#include <string.h>", "#include <stdarg.h>", "#include <stdio.h>", "#include <unistd.h>"], deps: &["arena", "bounds"], code: "/* An array of records (`[Block]`): the\n   records one after another in one block on the arena. It is a value: a copy\n   has storage of its own (viv_own_recs_T), so a write through one name is\n   never seen through another. Its count is taken in 64 bits and may not pass\n   INT_MAX (the struct's `size`), as an array's may not (viv_arr_alloc), and a\n   failed calloc stops the run. */\ntypedef struct { int size; void* data; } viv_recs;\nstatic inline viv_recs viv_recs_new(long long n, size_t elem) {\n    if (n > 2147483647LL) {\n        viv_fflush(stdout);\n        viv_errf(\"error: an array of more than %lld elements cannot be made\\n\", 2147483647LL);\n        viv_stop(VIV_E_OVERFLOW);\n    }\n    viv_recs a; a.size = n > 0 ? (int)n : 0;\n    a.data = calloc(a.size > 0 ? (size_t)a.size : 1, elem);\n    if (!a.data) {\n        viv_fflush(stdout);\n        viv_errf(\"error: out of memory making an array of %lld elements\\n\", n);\n        viv_stop(VIV_E_MEMORY);\n    }\n    viv_arena_push(a.data);\n    return a;\n}\nstatic inline viv_recs viv_recs_from(const void* src, int n, size_t elem) {\n    viv_recs b = viv_recs_new(n, elem);\n    if (n > 0) memcpy(b.data, src, (size_t)n * elem);\n    return b;\n}\nstatic inline viv_recs viv_recs_append(viv_recs a, const void* r, size_t elem) {\n    viv_recs b = viv_recs_new((long long)a.size + 1, elem);\n    if (a.size > 0) memcpy(b.data, a.data, (size_t)a.size * elem);\n    memcpy((char*)b.data + (size_t)a.size * elem, r, elem);\n    return b;\n}\nVIV_ACCESS void* viv_recs_at_at(viv_recs a, double x, size_t elem, int site) { return (char*)a.data + (size_t)viv_ix(x, a.size, 0, site) * elem; }\nVIV_ACCESS void* viv_recs_at(viv_recs a, double x, size_t elem) { return viv_recs_at_at(a, x, elem, 0); }\nstatic inline double viv_recs_len(viv_recs a) { return (double)a.size; }" })
            .with(RuntimeFragment { name: "arr_free", includes: &["#include <stdlib.h>"], deps: &["arena"], code: "/* free one array now (runtime-internal temporaries), wherever it is filed */\nstatic void viv_arr_free(viv_arr a) {\n    for (size_t i = viv_arena_top; i-- > 0; ) {\n        if (viv_arena[i] == a.data) {\n            for (size_t j = i + 1; j < viv_arena_top; j++) viv_arena[j-1] = viv_arena[j];\n            viv_arena_top--;\n            break;\n        }\n    }\n    viv_pool_put(a.data);\n}" })
            .with(RuntimeFragment { name: "carr_new", includes: &["#include <stdlib.h>"], deps: &["arr_type", "arena", "arr_new"], code: "/* A complex array shares viv_arr. `data` holds interleaved (re, im) doubles,\n   2 * size of them -- which is exactly how C99 lays out `double complex`\n   (6.2.5p13), so viewing it as `double complex*` is exact, not a trick. */\nstatic viv_arr viv_carr_new(int rank, const int* dims) { return viv_arr_alloc(rank, dims, 2); }" })
            .with(RuntimeFragment { name: "czeros1", includes: &[], deps: &["carr_new"], code: "static viv_arr viv_czeros1(double a) { double x[1] = {a}; int d[1]; viv_zeros_dims(1, x, d, 2); return viv_carr_new(1, d); }" })
            .with(RuntimeFragment { name: "czeros2", includes: &[], deps: &["carr_new"], code: "static viv_arr viv_czeros2(double a, double b) { double x[2] = {a, b}; int d[2]; viv_zeros_dims(2, x, d, 2); return viv_carr_new(2, d); }" })
            .with(RuntimeFragment { name: "czeros3", includes: &[], deps: &["carr_new"], code: "static viv_arr viv_czeros3(double a, double b, double c) { double x[3] = {a, b, c}; int d[3]; viv_zeros_dims(3, x, d, 2); return viv_carr_new(3, d); }" })
            .with(RuntimeFragment { name: "czeros4", includes: &[], deps: &["carr_new"], code: "static viv_arr viv_czeros4(double a, double b, double c, double e) { double x[4] = {a, b, c, e}; int d[4]; viv_zeros_dims(4, x, d, 2); return viv_carr_new(4, d); }" })
            .with(RuntimeFragment { name: "cget1", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_cget1_at(viv_arr a, double i, int site) {\n    viv_rank(a.rank, 1, site);\n    return ((double complex*)a.data)[viv_ix(i, a.dims[0], 0, site)];\n}\nVIV_ACCESS double complex viv_cget1(viv_arr a, double i) { return viv_cget1_at(a, i, 0); }" })
            .with(RuntimeFragment { name: "cset1", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_cset1_at(viv_arr a, double i, double complex v, int site) {\n    viv_rank(a.rank, 1, site);\n    ((double complex*)a.data)[viv_ix(i, a.dims[0], 0, site)] = v;\n}\nVIV_ACCESS void viv_cset1(viv_arr a, double i, double complex v) { viv_cset1_at(a, i, v, 0); }" })
            .with(RuntimeFragment { name: "cget2", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_cget2_at(viv_arr a, double i, double j, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site);\n    return ((double complex*)a.data)[ii * a.dims[1] + jj];\n}\nVIV_ACCESS double complex viv_cget2(viv_arr a, double i, double j) { return viv_cget2_at(a, i, j, 0); }" })
            .with(RuntimeFragment { name: "cset2", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_cset2_at(viv_arr a, double i, double j, double complex v, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site);\n    ((double complex*)a.data)[ii * a.dims[1] + jj] = v;\n}\nVIV_ACCESS void viv_cset2(viv_arr a, double i, double j, double complex v) { viv_cset2_at(a, i, j, v, 0); }" })
            .with(RuntimeFragment { name: "cget3", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_cget3_at(viv_arr a, double i, double j, double k, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site);\n    return ((double complex*)a.data)[(ii * a.dims[1] + jj) * a.dims[2] + kk];\n}\nVIV_ACCESS double complex viv_cget3(viv_arr a, double i, double j, double k) { return viv_cget3_at(a, i, j, k, 0); }" })
            .with(RuntimeFragment { name: "cset3", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_cset3_at(viv_arr a, double i, double j, double k, double complex v, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site);\n    ((double complex*)a.data)[(ii * a.dims[1] + jj) * a.dims[2] + kk] = v;\n}\nVIV_ACCESS void viv_cset3(viv_arr a, double i, double j, double k, double complex v) { viv_cset3_at(a, i, j, k, v, 0); }" })
            .with(RuntimeFragment { name: "lcget1", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_lcget1_at(viv_arr a, long long i, int site) {\n    viv_rank(a.rank, 1, site);\n    int ii = viv_ixl(i, a.dims[0], 0, site);\n    return ((double complex*)a.data)[ii];\n}\nVIV_ACCESS double complex viv_lcget1(viv_arr a, long long i) { return viv_lcget1_at(a, i, 0); }" })
            .with(RuntimeFragment { name: "lcset1", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lcset1_at(viv_arr a, long long i, double complex v, int site) {\n    viv_rank(a.rank, 1, site);\n    int ii = viv_ixl(i, a.dims[0], 0, site);\n    ((double complex*)a.data)[ii] = v;\n}\nVIV_ACCESS void viv_lcset1(viv_arr a, long long i, double complex v) { viv_lcset1_at(a, i, v, 0); }" })
            .with(RuntimeFragment { name: "lcget2", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_lcget2_at(viv_arr a, long long i, long long j, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ixl(i, a.dims[0], 0, site); int jj = viv_ixl(j, a.dims[1], 1, site);\n    return ((double complex*)a.data)[ii * a.dims[1] + jj];\n}\nVIV_ACCESS double complex viv_lcget2(viv_arr a, long long i, long long j) { return viv_lcget2_at(a, i, j, 0); }" })
            .with(RuntimeFragment { name: "lcset2", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lcset2_at(viv_arr a, long long i, long long j, double complex v, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ixl(i, a.dims[0], 0, site); int jj = viv_ixl(j, a.dims[1], 1, site);\n    ((double complex*)a.data)[ii * a.dims[1] + jj] = v;\n}\nVIV_ACCESS void viv_lcset2(viv_arr a, long long i, long long j, double complex v) { viv_lcset2_at(a, i, j, v, 0); }" })
            .with(RuntimeFragment { name: "lcget3", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_lcget3_at(viv_arr a, long long i, long long j, long long k, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ixl(i, a.dims[0], 0, site); int jj = viv_ixl(j, a.dims[1], 1, site); int kk = viv_ixl(k, a.dims[2], 2, site);\n    return ((double complex*)a.data)[(ii * a.dims[1] + jj) * a.dims[2] + kk];\n}\nVIV_ACCESS double complex viv_lcget3(viv_arr a, long long i, long long j, long long k) { return viv_lcget3_at(a, i, j, k, 0); }" })
            .with(RuntimeFragment { name: "lcset3", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lcset3_at(viv_arr a, long long i, long long j, long long k, double complex v, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ixl(i, a.dims[0], 0, site); int jj = viv_ixl(j, a.dims[1], 1, site); int kk = viv_ixl(k, a.dims[2], 2, site);\n    ((double complex*)a.data)[(ii * a.dims[1] + jj) * a.dims[2] + kk] = v;\n}\nVIV_ACCESS void viv_lcset3(viv_arr a, long long i, long long j, long long k, double complex v) { viv_lcset3_at(a, i, j, k, v, 0); }" })
            .with(RuntimeFragment { name: "cget4", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double complex viv_cget4_at(viv_arr a, double i, double j, double k, double l, int site) {\n    viv_rank(a.rank, 4, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site); int ll = viv_ix(l, a.dims[3], 3, site);\n    return ((double complex*)a.data)[((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll];\n}\nVIV_ACCESS double complex viv_cget4(viv_arr a, double i, double j, double k, double l) { return viv_cget4_at(a, i, j, k, l, 0); }" })
            .with(RuntimeFragment { name: "cset4", includes: &["#include <complex.h>"], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_cset4_at(viv_arr a, double i, double j, double k, double l, double complex v, int site) {\n    viv_rank(a.rank, 4, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site); int ll = viv_ix(l, a.dims[3], 3, site);\n    ((double complex*)a.data)[((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll] = v;\n}\nVIV_ACCESS void viv_cset4(viv_arr a, double i, double j, double k, double l, double complex v) { viv_cset4_at(a, i, j, k, l, v, 0); }" })
            .with(RuntimeFragment { name: "cvec", includes: &["#include <complex.h>", "#include <stdarg.h>"], deps: &["carr_new"], code: "/* A rank-1 complex literal. Every element arrives as a `double complex`\n   (viv_cpromote has already promoted the reals), so va_arg is unambiguous. */\nstatic double complex viv_cpromote(double complex z) { return z; }\nstatic viv_arr viv_cvec(int n, ...) {\n    int dims[1]; dims[0] = n;\n    viv_arr a = viv_carr_new(1, dims);\n    va_list ap; va_start(ap, n);\n    for (int i = 0; i < n; i++) ((double complex*)a.data)[i] = va_arg(ap, double complex);\n    va_end(ap);\n    return a;\n}" })
            .with(RuntimeFragment { name: "show_carr", includes: &["#include <complex.h>", "#include <stdio.h>", "#include <string.h>", "#include <math.h>"], deps: &["arr_type", "fmt_g"], code: "/* One complex value, as the interpreter's format_cplx writes it: viv_g writes\n   a zero of either sign as \"0\" (FFT output is full of -0.0), and the sign\n   between is the imaginary part's unless it is a zero. */\nstatic void viv_put_cplx(double re, double im) {\n    viv_printf(\"%s%s%si\", viv_g(re), signbit(im) && im != 0.0 ? \"-\" : \"+\", viv_g(fabs(im)));\n}\nstatic void viv_show_carr(const char* label, viv_arr a) {\n    if (label[0]) viv_printf(\"%s: \", label);\n    if (a.rank == 2) {\n        int rows = a.dims[0], cols = a.dims[1];\n        int pad = label[0] ? (int)strlen(label) + 3 : 1;\n        viv_printf(\"[\");\n        for (int i = 0; i < rows; i++) {\n            if (i) viv_printf(\"%*s\", pad, \"\");\n            viv_printf(\"[\");\n            for (int j = 0; j < cols; j++) { if (j) viv_printf(\", \"); viv_put_cplx(a.data[2*(i*cols+j)], a.data[2*(i*cols+j)+1]); }\n            viv_printf(\"]\");\n            if (i < rows - 1) viv_printf(\",\\n\");\n        }\n        viv_printf(\"]\\n\");\n    } else {\n        viv_printf(\"[\");\n        for (int i = 0; i < a.size; i++) { if (i) viv_printf(\", \"); viv_put_cplx(a.data[2*i], a.data[2*i+1]); }\n        viv_printf(\"]\\n\");\n    }\n}" })
            .with(RuntimeFragment { name: "carr_fn", includes: &["#include <math.h>"], deps: &["arr_new", "carr_new"], code: "/* re / im / abs over a complex array give a REAL array of the same shape;\n   conj gives another complex one. abs is hypot, as the interpreter's is. */\nstatic viv_arr viv_carr_re(viv_arr a)  { viv_arr r = viv_arr_new(a.rank, a.dims); for (int i = 0; i < a.size; i++) r.data[i] = a.data[2*i]; return r; }\nstatic viv_arr viv_carr_im(viv_arr a)  { viv_arr r = viv_arr_new(a.rank, a.dims); for (int i = 0; i < a.size; i++) r.data[i] = a.data[2*i+1]; return r; }\nstatic viv_arr viv_carr_abs(viv_arr a) { viv_arr r = viv_arr_new(a.rank, a.dims); for (int i = 0; i < a.size; i++) r.data[i] = hypot(a.data[2*i], a.data[2*i+1]); return r; }\nstatic viv_arr viv_carr_conj(viv_arr a) { viv_arr r = viv_carr_new(a.rank, a.dims); for (int i = 0; i < a.size; i++) { r.data[2*i] = a.data[2*i]; r.data[2*i+1] = -a.data[2*i+1]; } return r; }" })
            // cos and sin of one angle as two separate calls, the way the
            // interpreter's cos_sin makes them: an optimizing C compiler merges
            // cos(x) next to sin(x) into one sincos call, whose last bit can
            // differ, and -O0 does not. Reading a volatile twice gives the two
            // calls arguments the compiler cannot prove equal.
            .with(RuntimeFragment { name: "cos_sin", includes: &["#include <math.h>"], deps: &[], code: "static void viv_cos_sin(double x, double* c, double* s) {\n    volatile double v = x;\n    *c = cos(v);\n    *s = sin(v);\n}" })
            .with(RuntimeFragment { name: "fft", includes: &["#include <math.h>", "#include <stdlib.h>"], deps: &["arr_type", "cos_sin"], code: "/* The discrete Fourier transform, in place, over interleaved (re, im) data.\n     forward:  X[k] = sum_n x[n] exp(-2 pi i k n / N)      (unscaled)\n     inverse:  x[n] = (1/N) sum_k X[k] exp(+2 pi i k n / N)\n   Powers of two: iterative radix-2. Lengths whose primes are all <= 7:\n   recursive mixed radix. Every other length: Bluestein's chirp-z, so\n   N log N holds for all N.\n   Everything that depends only on the length -- the twiddle table, the\n   factorization, Bluestein's chirp and its transform -- is a PLAN, built once\n   per length and kept. This is the interpreter's fft_in_place, line for\n   line: same order of operations, twiddles from cos/sin of the exact angle\n   rather than a recurrence (which would drift, and drift differently in each\n   engine), and contraction off so a fused multiply-add cannot change the low\n   bits. */\ntypedef struct viv_fft_plan {\n    int n, kind;            /* kind: 0 radix-2, 1 mixed radix, 2 Bluestein */\n    double* tw;             /* exp(-2 pi i k / n), k in 0..n, interleaved */\n    int nf, fac[64];        /* mixed radix: the prime factors */\n    int m;                  /* Bluestein: the convolution length */\n    double* chirp[2];       /* Bluestein: forward [0] and inverse [1] */\n    double* bfft[2];\n    double* work;           /* 2 * max(n, m) doubles of scratch */\n    struct viv_fft_plan* next;\n} viv_fft_plan;\nstatic viv_fft_plan* viv_fft_plans = 0;\nstatic void viv_fft_pow2(double* d, int n, const double* tw, int conj_) {\n#pragma STDC FP_CONTRACT OFF\n    int j = 0;\n    for (int i = 1; i < n; i++) {\n        int bit = n >> 1;\n        while (j & bit) { j ^= bit; bit >>= 1; }\n        j ^= bit;\n        if (i < j) {\n            double t = d[2*i]; d[2*i] = d[2*j]; d[2*j] = t;\n            t = d[2*i+1]; d[2*i+1] = d[2*j+1]; d[2*j+1] = t;\n        }\n    }\n    /* stage `len` wants exp(-2 pi i k / len) = tw[k * (n / len)] */\n    for (int len = 2; len <= n; len <<= 1) {\n        int half = len / 2, skip = n / len;\n        for (int k = 0; k < half; k++) {\n            double wr = tw[2*k*skip], wi = conj_ ? -tw[2*k*skip+1] : tw[2*k*skip+1];\n            for (int i = k; i < n; i += len) {\n                int a = 2 * i, b = 2 * (i + half);\n                double vr = d[b] * wr - d[b+1] * wi, vi = d[b] * wi + d[b+1] * wr;\n                double ur = d[a], ui = d[a+1];\n                d[a] = ur + vr; d[a+1] = ui + vi;\n                d[b] = ur - vr; d[b+1] = ui - vi;\n            }\n        }\n    }\n}\nstatic viv_fft_plan* viv_fft_plan_for(int n) {\n#pragma STDC FP_CONTRACT OFF\n    for (viv_fft_plan* q = viv_fft_plans; q; q = q->next) if (q->n == n) return q;\n    viv_fft_plan* pl = (viv_fft_plan*)calloc(1, sizeof(viv_fft_plan));\n    pl->n = n;\n    pl->tw = (double*)malloc((size_t)n * 2 * sizeof(double));\n    double step = -2.0 * 3.14159265358979323846 / (double)n;\n    for (int k = 0; k < n; k++) { viv_cos_sin(step * (double)k, &pl->tw[2*k], &pl->tw[2*k+1]); }\n    int worklen = n;\n    if ((n & (n - 1)) == 0) {\n        pl->kind = 0;\n    } else {\n        static const int primes[4] = { 2, 3, 5, 7 };\n        int rest = n;\n        for (int i = 0; i < 4; i++) while (rest % primes[i] == 0) { pl->fac[pl->nf++] = primes[i]; rest /= primes[i]; }\n        if (rest == 1) {\n            pl->kind = 1;\n        } else {\n            pl->kind = 2;\n            int m = 1;\n            while (m < 2 * n - 1) m <<= 1;\n            pl->m = m; worklen = m;\n            viv_fft_plan* inner = viv_fft_plan_for(m);\n            for (int s = 0; s < 2; s++) {\n                double sign = s ? 1.0 : -1.0;\n                double* chirp = (double*)calloc((size_t)n * 2, sizeof(double));\n                double* b = (double*)calloc((size_t)m * 2, sizeof(double));\n                for (int j = 0; j < n; j++) {\n                    /* chirp w[j] = exp(sign i pi j^2 / n); j^2 reduced mod 2n first, which\n                       is its period and keeps the angle small enough to stay accurate */\n                    double r = (double)(((unsigned long long)j * (unsigned long long)j) % (2ULL * (unsigned long long)n));\n                    double ang = sign * 3.14159265358979323846 * r / (double)n;\n                    double wr, wi; viv_cos_sin(ang, &wr, &wi);\n                    chirp[2*j] = wr; chirp[2*j+1] = wi;\n                    b[2*j] = wr; b[2*j+1] = -wi;\n                    if (j > 0) { b[2*(m-j)] = wr; b[2*(m-j)+1] = -wi; }\n                }\n                viv_fft_pow2(b, m, inner->tw, 0);\n                pl->chirp[s] = chirp; pl->bfft[s] = b;\n            }\n        }\n    }\n    pl->work = (double*)malloc((size_t)worklen * 2 * sizeof(double));\n    pl->next = viv_fft_plans; viv_fft_plans = pl;\n    return pl;\n}\n/* Recursive mixed radix (decimation in time). `out` receives the transform of\n   the n_sub samples inp[off], inp[off+stride], ...; stride * n_sub is always\n   n, so every level reads its twiddles out of the one length-n table. */\nVIV_SCALAR_LOOPS static void viv_fft_mixed(double* out, const double* inp, int off, int stride, const int* fac, int n_sub, const double* tw, int n, int conj_) {\n#pragma STDC FP_CONTRACT OFF\n    int p = fac[0], m = n_sub / p;\n    if (m == 1) {\n        for (int q = 0; q < p; q++) { out[2*q] = inp[2*(off + q*stride)]; out[2*q+1] = inp[2*(off + q*stride)+1]; }\n    } else {\n        for (int q = 0; q < p; q++) viv_fft_mixed(out + 2*q*m, inp, off + q*stride, stride * p, fac + 1, m, tw, n, conj_);\n    }\n    if (p == 2) {\n        for (int u = 0; u < m; u++) {\n            int t = stride * u;\n            double wr = tw[2*t], wi = conj_ ? -tw[2*t+1] : tw[2*t+1];\n            int a = 2 * u, b = 2 * (u + m);\n            double vr = out[b] * wr - out[b+1] * wi, vi = out[b] * wi + out[b+1] * wr;\n            double ur = out[a], ui = out[a+1];\n            out[a] = ur + vr; out[a+1] = ui + vi;\n            out[b] = ur - vr; out[b+1] = ui - vi;\n        }\n    } else if (p == 3) {\n        /* radix 3 in two multiplies per triple: with w = exp(-2 pi i / 3) =\n           -1/2 + i e, the outputs are F0 + s3 and F0 - s3/2 +- i e s0, where\n           s3 = s1 + s2 and s0 = s1 - s2 are the twiddled inputs' sum and difference */\n        double e = conj_ ? -tw[2*stride*m+1] : tw[2*stride*m+1];\n        for (int u = 0; u < m; u++) {\n            int t1 = stride * u, t2 = 2 * stride * u;\n            double w1r = tw[2*t1], w1i = conj_ ? -tw[2*t1+1] : tw[2*t1+1];\n            double w2r = tw[2*t2], w2i = conj_ ? -tw[2*t2+1] : tw[2*t2+1];\n            int i0 = 2 * u, i1 = 2 * (u + m), i2 = 2 * (u + 2*m);\n            double s1r = out[i1] * w1r - out[i1+1] * w1i, s1i = out[i1] * w1i + out[i1+1] * w1r;\n            double s2r = out[i2] * w2r - out[i2+1] * w2i, s2i = out[i2] * w2i + out[i2+1] * w2r;\n            double s3r = s1r + s2r, s3i = s1i + s2i;\n            double s0r = (s1r - s2r) * e, s0i = (s1i - s2i) * e;\n            double hr = out[i0] - 0.5 * s3r, hi = out[i0+1] - 0.5 * s3i;\n            out[i0] += s3r; out[i0+1] += s3i;\n            out[i2] = hr + s0i; out[i2+1] = hi - s0r;\n            out[i1] = hr - s0i; out[i1+1] = hi + s0r;\n        }\n    } else if (p == 5) {\n        /* radix 5 in four multiplies per quintuple, pairing q with 5 - q */\n        double yar = tw[2*stride*m], ybr = tw[2*2*stride*m];\n        double yai = conj_ ? -tw[2*stride*m+1] : tw[2*stride*m+1];\n        double ybi = conj_ ? -tw[2*2*stride*m+1] : tw[2*2*stride*m+1];\n        for (int u = 0; u < m; u++) {\n            double x[10];\n            for (int q = 1; q < 5; q++) {\n                int t = q * stride * u;\n                double wr = tw[2*t], wi = conj_ ? -tw[2*t+1] : tw[2*t+1];\n                int i = 2 * (u + q*m);\n                x[2*q] = out[i] * wr - out[i+1] * wi;\n                x[2*q+1] = out[i] * wi + out[i+1] * wr;\n            }\n            int i0 = 2 * u, i1 = 2 * (u + m), i2 = 2 * (u + 2*m), i3 = 2 * (u + 3*m), i4 = 2 * (u + 4*m);\n            double f0r = out[i0], f0i = out[i0+1];\n            double s7r = x[2] + x[8], s7i = x[3] + x[9], s10r = x[2] - x[8], s10i = x[3] - x[9];\n            double s8r = x[4] + x[6], s8i = x[5] + x[7], s9r = x[4] - x[6], s9i = x[5] - x[7];\n            out[i0] = f0r + (s7r + s8r);\n            out[i0+1] = f0i + (s7i + s8i);\n            double s5r = f0r + s7r * yar + s8r * ybr, s5i = f0i + s7i * yar + s8i * ybr;\n            double s6r = s10i * yai + s9i * ybi, s6i = -(s10r * yai) - s9r * ybi;\n            out[i1] = s5r - s6r; out[i1+1] = s5i - s6i;\n            out[i4] = s5r + s6r; out[i4+1] = s5i + s6i;\n            double s11r = f0r + s7r * ybr + s8r * yar, s11i = f0i + s7i * ybr + s8i * yar;\n            double s12r = -(s10i * ybi) + s9i * yai, s12i = s10r * ybi - s9r * yai;\n            out[i2] = s11r + s12r; out[i2+1] = s11i + s12i;\n            out[i3] = s11r - s12r; out[i3+1] = s11i - s12i;\n        }\n    } else {\n        double sc[14];\n        for (int u = 0; u < m; u++) {\n            for (int q1 = 0; q1 < p; q1++) { sc[2*q1] = out[2*(u + q1*m)]; sc[2*q1+1] = out[2*(u + q1*m)+1]; }\n            for (int q1 = 0; q1 < p; q1++) {\n                int k = u + q1 * m, t = 0;\n                double sr = sc[0], si = sc[1];\n                for (int q = 1; q < p; q++) {\n                    t += stride * k;\n                    if (t >= n) t -= n;\n                    double wr = tw[2*t], wi = conj_ ? -tw[2*t+1] : tw[2*t+1];\n                    sr += sc[2*q] * wr - sc[2*q+1] * wi;\n                    si += sc[2*q] * wi + sc[2*q+1] * wr;\n                }\n                out[2*k] = sr; out[2*k+1] = si;\n            }\n        }\n    }\n}\nVIV_SCALAR_LOOPS static void viv_fft_raw(double* data, int n, int inverse) {\n#pragma STDC FP_CONTRACT OFF\n    if (n < 2) return;\n    viv_fft_plan* pl = viv_fft_plan_for(n);\n    if (pl->kind == 0) {\n        viv_fft_pow2(data, n, pl->tw, inverse);\n    } else if (pl->kind == 1) {\n        for (int i = 0; i < 2 * n; i++) pl->work[i] = data[i];\n        viv_fft_mixed(data, pl->work, 0, 1, pl->fac, n, pl->tw, n, inverse);\n    } else {\n        int m = pl->m;\n        const double* chirp = pl->chirp[inverse ? 1 : 0];\n        const double* bfft = pl->bfft[inverse ? 1 : 0];\n        viv_fft_plan* inner = viv_fft_plan_for(m);\n        double* a = pl->work;\n        for (int j = 0; j < 2 * m; j++) a[j] = 0.0;\n        for (int j = 0; j < n; j++) {\n            double wr = chirp[2*j], wi = chirp[2*j+1];\n            double xr = data[2*j], xi = data[2*j+1];\n            a[2*j] = xr * wr - xi * wi;\n            a[2*j+1] = xr * wi + xi * wr;\n        }\n        viv_fft_pow2(a, m, inner->tw, 0);\n        for (int j = 0; j < m; j++) {\n            double ar = a[2*j], ai = a[2*j+1], br = bfft[2*j], bi = bfft[2*j+1];\n            a[2*j] = ar * br - ai * bi;\n            a[2*j+1] = ar * bi + ai * br;\n        }\n        viv_fft_pow2(a, m, inner->tw, 1);\n        double scale = 1.0 / (double)m;\n        for (int j = 0; j < n; j++) {\n            double wr = chirp[2*j], wi = chirp[2*j+1];\n            double cr = a[2*j] * scale, ci = a[2*j+1] * scale;\n            data[2*j] = cr * wr - ci * wi;\n            data[2*j+1] = cr * wi + ci * wr;\n        }\n    }\n    if (inverse) {\n        double s = 1.0 / (double)n;\n        for (int i = 0; i < 2 * n; i++) data[i] *= s;\n    }\n}\nVIV_SCALAR_LOOPS static double viv_fft(viv_arr arr, double inverse) {\n    viv_fft_raw(arr.data, arr.size, inverse != 0.0);\n    return 0.0;\n}" })
            .with(RuntimeFragment { name: "arr_extent", includes: &["#include <math.h>"], deps: &["arr_type"], code: "/* the largest / smallest entry; of an empty array, -inf / +inf */\nstatic double viv_maxval(viv_arr a) { double m = -INFINITY; for (int i = 0; i < a.size; i++) m = fmax(m, a.data[i]); return m; }\nstatic double viv_minval(viv_arr a) { double m = INFINITY; for (int i = 0; i < a.size; i++) m = fmin(m, a.data[i]); return m; }" })
            .with(RuntimeFragment { name: "arr_cmp", includes: &[], deps: &["arr_new"], code: "/* elementwise comparisons: 1.0 where it holds, 0.0 where it does not */\n#define VIV_CMP(name, op) static viv_arr name(viv_arr a, viv_arr b) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] op b.data[i] ? 1.0 : 0.0; return r; }\nVIV_CMP(viv_arr_lt, <) VIV_CMP(viv_arr_le, <=) VIV_CMP(viv_arr_gt, >) VIV_CMP(viv_arr_ge, >=) VIV_CMP(viv_arr_eq, ==) VIV_CMP(viv_arr_ne, !=)\n#undef VIV_CMP" })
            .with(RuntimeFragment { name: "arr_cmps", includes: &[], deps: &["arr_new"], code: "/* elementwise comparisons against a scalar (the scalar second) */\n#define VIV_CMPS(name, op) static viv_arr name(viv_arr a, double s) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] op s ? 1.0 : 0.0; return r; }\nVIV_CMPS(viv_arr_lts, <) VIV_CMPS(viv_arr_les, <=) VIV_CMPS(viv_arr_gts, >) VIV_CMPS(viv_arr_ges, >=) VIV_CMPS(viv_arr_eqs, ==) VIV_CMPS(viv_arr_nes, !=)\n#undef VIV_CMPS" })
            .with(RuntimeFragment { name: "all_any", includes: &[], deps: &["arr_type"], code: "/* all(m) / any(m): every / some entry nonzero; all of nothing is true */\nstatic double viv_all(viv_arr a) { for (int i = 0; i < a.size; i++) if (a.data[i] == 0.0) return 0.0; return 1.0; }\nstatic double viv_any(viv_arr a) { for (int i = 0; i < a.size; i++) if (a.data[i] != 0.0) return 1.0; return 0.0; }" })
            .with(RuntimeFragment {
                name: "where",
                includes: &["#include <stdio.h>", "#include <unistd.h>"],
                deps: &["arr_new", "arr_chk"],
                code: "/* where(m, a, b): a where m is nonzero, b elsewhere, each a number or an\n   array; the arrays broadcast to one shape as NumPy's do (interp/arrays.rs),\n   which is m's shape when they match it */\nstatic void viv_where_fail(viv_arr m, viv_arr a) {\n    viv_fflush(stdout);\n    viv_errf(\"error: array shape mismatch: [\");\n    for (int i = 0; i < m.rank; i++) viv_errf(i ? \", %d\" : \"%d\", m.dims[i]);\n    viv_errf(\"] vs [\");\n    for (int i = 0; i < a.rank; i++) viv_errf(i ? \", %d\" : \"%d\", a.dims[i]);\n    viv_errf(\"]\\n\");\n    viv_stop(VIV_E_VALUE);\n}\n/* the shape (rank, to) broadcast with a's, or a stop naming m and a */\nstatic void viv_where_grow(int* rank, int* to, viv_arr m, viv_arr a) {\n    int r = *rank > a.rank ? *rank : a.rank, t[8];\n    for (int d = 0; d < r; d++) {\n        int x = d + *rank >= r ? to[d + *rank - r] : 1, y = d + a.rank >= r ? a.dims[d + a.rank - r] : 1;\n        if (x != y && x != 1 && y != 1) viv_where_fail(m, a);\n        t[d] = x == 1 ? y : x;\n    }\n    *rank = r;\n    for (int d = 0; d < r; d++) to[d] = t[d];\n}\nstatic int viv_where_is(viv_arr a, int rank, const int* to) {\n    if (a.rank != rank) return 0;\n    for (int d = 0; d < rank; d++) if (a.dims[d] != to[d]) return 0;\n    return 1;\n}\nstatic viv_arr viv_where_to(viv_arr a, int rank, const int* to) {\n    if (viv_where_is(a, rank, to)) return a;\n    double total = 1.0;\n    for (int d = 0; d < rank; d++) total *= (double)to[d];\n    if (total > 2147483647.0) {\n        viv_fflush(stdout);\n        viv_errf(\"error: `where`: broadcasting gives %.0f elements, over the 2147483647-element limit\\n\", total);\n        viv_stop(VIV_E_OVERFLOW);\n    }\n    return viv_bc_expand(a, rank, to);\n}\nstatic viv_arr viv_where_aa(viv_arr m, viv_arr a, viv_arr b) {\n    int rank = m.rank, to[8];\n    for (int d = 0; d < rank; d++) to[d] = m.dims[d];\n    viv_where_grow(&rank, to, m, a); viv_where_grow(&rank, to, m, b);\n    m = viv_where_to(m, rank, to); a = viv_where_to(a, rank, to); b = viv_where_to(b, rank, to);\n    viv_arr r = viv_arr_new_nz(rank, to);\n    for (int i = 0; i < r.size; i++) r.data[i] = m.data[i] != 0.0 ? a.data[i] : b.data[i];\n    return r;\n}\nstatic viv_arr viv_where_as(viv_arr m, viv_arr a, double b) {\n    int rank = m.rank, to[8];\n    for (int d = 0; d < rank; d++) to[d] = m.dims[d];\n    viv_where_grow(&rank, to, m, a);\n    m = viv_where_to(m, rank, to); a = viv_where_to(a, rank, to);\n    viv_arr r = viv_arr_new_nz(rank, to);\n    for (int i = 0; i < r.size; i++) r.data[i] = m.data[i] != 0.0 ? a.data[i] : b;\n    return r;\n}\nstatic viv_arr viv_where_sa(viv_arr m, double a, viv_arr b) {\n    int rank = m.rank, to[8];\n    for (int d = 0; d < rank; d++) to[d] = m.dims[d];\n    viv_where_grow(&rank, to, m, b);\n    m = viv_where_to(m, rank, to); b = viv_where_to(b, rank, to);\n    viv_arr r = viv_arr_new_nz(rank, to);\n    for (int i = 0; i < r.size; i++) r.data[i] = m.data[i] != 0.0 ? a : b.data[i];\n    return r;\n}\nstatic viv_arr viv_where_ss(viv_arr m, double a, double b) { viv_arr r = viv_arr_new_nz(m.rank, m.dims); for (int i = 0; i < r.size; i++) r.data[i] = m.data[i] != 0.0 ? a : b; return r; }",
            })
            .with(RuntimeFragment { name: "floorops", includes: &["#include <stdio.h>", "#include <math.h>", "#include <unistd.h>"], deps: &["sites"], code: "/* `a % b` and `a // b`: Python's rules for floats (floatobject.c,\n   float_rem and float_floor_div), step for step as the interpreter's\n   floor_mod and floor_div. fmod is exact; the remainder then takes the\n   divisor's sign, and a == (a // b) * b + a % b. A zero divisor stops the\n   program with the interpreter's words and location. */\nstatic void viv_zerodiv(const char* op, int site) {\n    viv_fflush(stdout);\n#ifdef VIV_BOUNDS\n    viv_errf(\"error: division by zero in `a %s b`\\n%s\", op, viv_sites[site]);\n#else\n    (void)site;\n    viv_errf(\"error: division by zero in `a %s b`\\n\", op);\n#endif\n    viv_stop(VIV_E_ZERODIV);\n}\nstatic double viv_floormod(double a, double b, int site) {\n    if (b == 0.0) viv_zerodiv(\"%\", site);\n    double m = fmod(a, b);\n    if (m != 0.0) { if ((b < 0.0) != (m < 0.0)) m += b; }\n    else m = copysign(0.0, b);\n    return m;\n}\nstatic double viv_floordiv(double a, double b, int site) {\n    if (b == 0.0) viv_zerodiv(\"//\", site);\n    double m = fmod(a, b);\n    double d = (a - m) / b;\n    if (m != 0.0 && (b < 0.0) != (m < 0.0)) d -= 1.0;\n    if (d != 0.0) { double f = floor(d); if (d - f > 0.5) f += 1.0; return f; }\n    return copysign(0.0, a / b);\n}" })
            .with(RuntimeFragment { name: "bits", includes: &["#include <math.h>", "#include <stdint.h>"], deps: &[], code: "/* Bits of a double: truncate toward zero, wrap into [0, 2^32), and work on\n   that. The interpreter's u32_of is this same arithmetic, so both engines\n   agree on every input, negatives and huge magnitudes included. A shift of 32\n   or more takes everything away (and, for sar, leaves the sign). */\nstatic uint32_t viv_u32_of(double x) {\n    double t = fmod(trunc(x), 4294967296.0);\n    if (t < 0.0) t += 4294967296.0;\n    return (uint32_t)t;\n}\nstatic int viv_shift_ok(double k, int* out) { double t = trunc(k); *out = (int)t; return t >= 0.0 && t < 32.0; }\nstatic double viv_band(double a, double b) { return (double)(viv_u32_of(a) & viv_u32_of(b)); }\nstatic double viv_bor(double a, double b)  { return (double)(viv_u32_of(a) | viv_u32_of(b)); }\nstatic double viv_bxor(double a, double b) { return (double)(viv_u32_of(a) ^ viv_u32_of(b)); }\nstatic double viv_bnot(double a) { return (double)(~viv_u32_of(a)); }\nstatic double viv_shl(double a, double k) { int s; return viv_shift_ok(k, &s) ? (double)(viv_u32_of(a) << s) : 0.0; }\nstatic double viv_shr(double a, double k) { int s; return viv_shift_ok(k, &s) ? (double)(viv_u32_of(a) >> s) : 0.0; }\nstatic double viv_sar(double a, double k) {\n    int s;\n    int32_t v = (int32_t)viv_u32_of(a);\n    if (viv_shift_ok(k, &s)) return (double)(uint32_t)(v >> s);\n    return (double)(uint32_t)(v < 0 ? -1 : 0);\n}\nstatic double viv_u32(double a) { return (double)viv_u32_of(a); }\nstatic double viv_i32(double a) { return (double)(int32_t)viv_u32_of(a); }" })
            .with(RuntimeFragment { name: "sum", includes: &[], deps: &["arr_type"], code: "/* sum(a): NumPy's order, as the interpreter adds (interp/arrays.rs npsum):\n   from 0.0, blocks of 8192 entries in row-major order, each block by NumPy's\n   pairwise sum (below 8 entries left to right; to 128, eight running sums\n   combined as a tree; above, two halves, the first a multiple of 8 long) */\nstatic double viv_sum_pw(const double* f, long long n) {\n    if (n < 8) {\n        double s = 0.0;\n        for (long long i = 0; i < n; i++) s += f[i];\n        return s;\n    }\n    if (n <= 128) {\n        double r[8];\n        for (int j = 0; j < 8; j++) r[j] = f[j];\n        long long i = 8;\n        for (; i < n - n % 8; i += 8) for (int j = 0; j < 8; j++) r[j] += f[i + j];\n        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));\n        for (; i < n; i++) s += f[i];\n        return s;\n    }\n    long long n2 = n / 2;\n    n2 -= n2 % 8;\n    return viv_sum_pw(f, n2) + viv_sum_pw(f + n2, n - n2);\n}\nstatic double viv_sum(viv_arr a) {\n    double s = 0.0;\n    for (long long b = 0; b < a.size; b += 8192) s += viv_sum_pw(a.data + b, a.size - b < 8192 ? a.size - b : 8192);\n    return s;\n}" })
            .with(RuntimeFragment { name: "cfill", includes: &["#include <complex.h>"], deps: &["arr_type"], code: "static double viv_cfill(viv_arr a, double complex v) {\n    double re = creal(v), im = cimag(v);\n    for (int i = 0; i < a.size; i++) { a.data[2*i] = re; a.data[2*i+1] = im; }\n    return 0.0;\n}" })
            .with(RuntimeFragment { name: "carr_from", includes: &[], deps: &["carr_new"], code: "/* complex(re, im) over two real arrays of one shape: the complex array. */\nstatic viv_arr viv_carr_from(viv_arr re, viv_arr im) {\n    viv_arr r = viv_carr_new(re.rank, re.dims);\n    for (int i = 0; i < re.size; i++) { r.data[2*i] = re.data[i]; r.data[2*i+1] = im.data[i]; }\n    return r;\n}" })
            .with(RuntimeFragment { name: "fft3", includes: &["#include <stdlib.h>"], deps: &["fft"], code: "/* The 3-D transform is separable: three passes of the 1-D one over gathered\n   strides. It inherits every 1-D method, so a grid side can be any length\n   (2-3-5-7-smooth ones are the fast ones). */\nstatic double viv_fft3_do(viv_arr a, int inverse) {\n    int sh[3] = { a.dims[0], a.dims[1], a.dims[2] };\n    int st[3] = { a.dims[1] * a.dims[2], a.dims[2], 1 };\n    for (int axis = 0; axis < 3; axis++) {\n        int len = sh[axis];\n        if (len < 2) continue;\n        /* the two axes NOT being transformed index the lines */\n        int o1 = axis == 0 ? 1 : 0, o2 = axis == 2 ? 1 : 2;\n        int s = st[axis];\n        double* line = (double*)malloc((size_t)len * 2 * sizeof(double));\n        for (int p = 0; p < sh[o1]; p++) for (int q = 0; q < sh[o2]; q++) {\n            int base = p * st[o1] + q * st[o2];\n            int any = 0;\n            for (int t = 0; t < len; t++) {\n                line[2*t] = a.data[2*(base + t*s)]; line[2*t+1] = a.data[2*(base + t*s)+1];\n                any |= (line[2*t] != 0.0) | (line[2*t+1] != 0.0);\n            }\n            /* the transform of zeros is zeros, and a plane-wave sphere leaves\n               most of its box empty: skip those lines */\n            if (!any) continue;\n            viv_fft_raw(line, len, inverse);\n            for (int t = 0; t < len; t++) { a.data[2*(base + t*s)] = line[2*t]; a.data[2*(base + t*s)+1] = line[2*t+1]; }\n        }\n        free(line);\n    }\n    return 0.0;\n}\nstatic double viv_fft3(viv_arr a)  { return viv_fft3_do(a, 0); }\nstatic double viv_ifft3(viv_arr a) { return viv_fft3_do(a, 1); }" })
            .with(RuntimeFragment { name: "eig_herm", includes: &["#include <math.h>", "#include <stdlib.h>", "#include <stdio.h>", "#include <float.h>"], deps: &["arr_type", "tql2"], code: "/* The complex Hermitian eigenproblem: Householder reduction to a real\n   symmetric tridiagonal, implicit-shift QL on that, back-transformation.\n       A = Q T Q^H   (T Hermitian tridiagonal)     T = D S D^H   (D phases, S real)\n       S = Z diag(w) Z^T                           V = Q D Z\n   The interpreter's herm_eig, line for line. a and vecs are interleaved\n   (re, im), row-major n x n; only a's lower triangle is read. Ascending\n   eigenvalues; eigenvectors in the COLUMNS of vecs, as viv_eig_sym does. */\nstatic double viv_eig_herm(viv_arr A, viv_arr vals, viv_arr vecs) {\n#pragma STDC FP_CONTRACT OFF\n    int n = A.dims[0];\n#define VIX(i, j) (2 * ((i) * n + (j)))\n    double* a = (double*)malloc((size_t)n * n * 2 * sizeof(double));\n    double* q = (double*)calloc((size_t)n * n * 2, sizeof(double));\n    double* v = (double*)calloc((size_t)n * 2, sizeof(double));\n    double* pv = (double*)calloc((size_t)n * 2, sizeof(double));\n    double* d = (double*)calloc((size_t)n, sizeof(double));\n    double* e = (double*)calloc((size_t)n, sizeof(double));\n    double* ph = (double*)calloc((size_t)n * 2, sizeof(double));\n    double* z = (double*)calloc((size_t)n * n, sizeof(double));\n    int* order = (int*)malloc((size_t)n * sizeof(int));\n    for (int i = 0; i < 2 * n * n; i++) {\n        a[i] = A.data[i];\n        /* said plainly: otherwise the QL sweep spins to its cap and blames itself */\n        if (!isfinite(a[i])) { viv_errf(\"eig_herm: the matrix contains a NaN or an infinity\\n\"); viv_stop_exit(VIV_E_VALUE); }\n    }\n    for (int i = 0; i < n; i++) {\n        a[VIX(i, i) + 1] = 0.0;\n        for (int j = 0; j < i; j++) { a[VIX(j, i)] = a[VIX(i, j)]; a[VIX(j, i) + 1] = -a[VIX(i, j) + 1]; }\n        q[VIX(i, i)] = 1.0;\n    }\n    for (int k = 0; k + 2 < n; k++) {\n        double norm2 = 0.0, tail = 0.0;\n        for (int i = k + 1; i < n; i++) norm2 += a[VIX(i, k)] * a[VIX(i, k)] + a[VIX(i, k) + 1] * a[VIX(i, k) + 1];\n        double norm = sqrt(norm2);\n        for (int i = k + 2; i < n; i++) tail += a[VIX(i, k)] * a[VIX(i, k)] + a[VIX(i, k) + 1] * a[VIX(i, k) + 1];\n        if (tail == 0.0) continue;\n        double x0r = a[VIX(k + 1, k)], x0i = a[VIX(k + 1, k) + 1];\n        double x0 = hypot(x0r, x0i);\n        double pr = x0 == 0.0 ? 1.0 : x0r / x0, pi_ = x0 == 0.0 ? 0.0 : x0i / x0;\n        double alr = -pr * norm, ali = -pi_ * norm;\n        for (int i = 0; i < 2 * n; i++) v[i] = 0.0;\n        for (int i = k + 1; i < n; i++) { v[2*i] = a[VIX(i, k)]; v[2*i+1] = a[VIX(i, k) + 1]; }\n        v[2*(k+1)] -= alr; v[2*(k+1)+1] -= ali;\n        double vn = 0.0;\n        for (int i = k + 1; i < n; i++) vn += v[2*i] * v[2*i] + v[2*i+1] * v[2*i+1];\n        vn = sqrt(vn);\n        if (vn == 0.0) continue;\n        for (int i = k + 1; i < n; i++) { v[2*i] /= vn; v[2*i+1] /= vn; }\n        for (int i = 0; i < 2 * n; i++) pv[i] = 0.0;\n        for (int i = k; i < n; i++) {\n            double sr = 0.0, si = 0.0;\n            for (int j = k + 1; j < n; j++) {\n                double ar = a[VIX(i, j)], ai = a[VIX(i, j) + 1];\n                sr += ar * v[2*j] - ai * v[2*j+1];\n                si += ar * v[2*j+1] + ai * v[2*j];\n            }\n            pv[2*i] = sr; pv[2*i+1] = si;\n        }\n        double kk = 0.0;\n        for (int i = k + 1; i < n; i++) kk += v[2*i] * pv[2*i] + v[2*i+1] * pv[2*i+1];\n        for (int i = k; i < n; i++) { pv[2*i] -= kk * v[2*i]; pv[2*i+1] -= kk * v[2*i+1]; }\n        for (int i = k; i < n; i++) for (int j = k; j < n; j++) {\n            double vir = v[2*i], vii = v[2*i+1], wir = pv[2*i], wii = pv[2*i+1];\n            double vjr = v[2*j], vji = v[2*j+1], wjr = pv[2*j], wji = pv[2*j+1];\n            a[VIX(i, j)]     -= 2.0 * (vir * wjr + vii * wji + wir * vjr + wii * vji);\n            a[VIX(i, j) + 1] -= 2.0 * (vii * wjr - vir * wji + wii * vjr - wir * vji);\n        }\n        for (int i = 0; i < n; i++) {\n            double sr = 0.0, si = 0.0;\n            for (int j = k + 1; j < n; j++) {\n                double qr = q[VIX(i, j)], qi = q[VIX(i, j) + 1];\n                sr += qr * v[2*j] - qi * v[2*j+1];\n                si += qr * v[2*j+1] + qi * v[2*j];\n            }\n            for (int j = k + 1; j < n; j++) {\n                q[VIX(i, j)]     -= 2.0 * (sr * v[2*j] + si * v[2*j+1]);\n                q[VIX(i, j) + 1] -= 2.0 * (si * v[2*j] - sr * v[2*j+1]);\n            }\n        }\n    }\n    if (n > 0) ph[0] = 1.0;\n    for (int i = 0; i < n; i++) d[i] = a[VIX(i, i)];\n    for (int k = 0; k + 1 < n; k++) {\n        double er = a[VIX(k + 1, k)], ei = a[VIX(k + 1, k) + 1];\n        double m = hypot(er, ei);\n        e[k] = m;\n        double ur = m == 0.0 ? 1.0 : er / m, ui = m == 0.0 ? 0.0 : ei / m;\n        ph[2*(k+1)]   = ph[2*k] * ur - ph[2*k+1] * ui;\n        ph[2*(k+1)+1] = ph[2*k] * ui + ph[2*k+1] * ur;\n    }\n    for (int i = 0; i < n; i++) z[i * n + i] = 1.0;\n    if (!viv_tql2(n, d, e, z, 80)) { viv_errf(\"eig_herm: QL iteration did not converge\\n\"); viv_stop_exit(VIV_E_ERROR); }\n    /* ascending order: a stable insertion sort, matching the interpreter's */\n    for (int i = 0; i < n; i++) order[i] = i;\n    for (int i = 1; i < n; i++) {\n        int t = order[i], j = i - 1;\n        while (j >= 0 && d[order[j]] > d[t]) { order[j + 1] = order[j]; j--; }\n        order[j + 1] = t;\n    }\n    for (int c = 0; c < n; c++) vals.data[c] = d[order[c]];\n    for (int i = 0; i < n; i++) for (int c = 0; c < n; c++) {\n        int col = order[c];\n        double sr = 0.0, si = 0.0;\n        for (int j = 0; j < n; j++) {\n            double qr = q[VIX(i, j)], qi = q[VIX(i, j) + 1], dr = ph[2*j], di = ph[2*j+1], zj = z[j * n + col];\n            sr += (qr * dr - qi * di) * zj;\n            si += (qr * di + qi * dr) * zj;\n        }\n        vecs.data[VIX(i, c)] = sr; vecs.data[VIX(i, c) + 1] = si;\n    }\n#undef VIX\n    free(a); free(q); free(v); free(pv); free(d); free(e); free(ph); free(z); free(order);\n    return 0.0;\n}" })
            .with(RuntimeFragment { name: "zeros1", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_zeros1(double n) { double x[1] = {n}; int d[1]; viv_zeros_dims(1, x, d, 1); return viv_arr_new(1, d); }" })
            .with(RuntimeFragment { name: "zeros2", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_zeros2(double r, double c) { double x[2] = {r, c}; int d[2]; viv_zeros_dims(2, x, d, 1); return viv_arr_new(2, d); }" })
            .with(RuntimeFragment { name: "zeros3", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_zeros3(double a, double b, double c) { double x[3] = {a, b, c}; int d[3]; viv_zeros_dims(3, x, d, 1); return viv_arr_new(3, d); }" })
            .with(RuntimeFragment { name: "zeros4", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_zeros4(double a, double b, double c, double e) { double x[4] = {a, b, c, e}; int d[4]; viv_zeros_dims(4, x, d, 1); return viv_arr_new(4, d); }" })
            .with(RuntimeFragment { name: "len", includes: &[], deps: &["arr_type"], code: "static double viv_len(viv_arr a) { return (double)a.dims[0]; }" })
            .with(RuntimeFragment { name: "numel", includes: &[], deps: &["arr_type"], code: "static double viv_numel(viv_arr a) { return (double)a.size; }" })
            .with(RuntimeFragment { name: "shape", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_shape(viv_arr a) {\n    int d[1]; d[0] = a.rank;\n    viv_arr s = viv_arr_new(1, d);\n    for (int i = 0; i < a.rank; i++) s.data[i] = (double)a.dims[i];\n    return s;\n}" })
            .with(RuntimeFragment { name: "bounds", includes: &["#include <stdio.h>", "#include <limits.h>", "#include <math.h>", "#include <unistd.h>"], deps: &["sites", "fmt_g"], code: "/* Bounds checks: every array read and write goes through viv_ix, which\n   checks one index against its dimension, so an index out of bounds stops\n   the program instead of reading or writing memory it does not own. The\n   words and the location are the interpreter's, byte for byte.\n   An index is a double that must be exactly an integer: a fraction, NaN or\n   an infinity stops the program (\"index 2.5 is not an integer\"), as in\n   the interpreter (interp.rs, index_int). An integer is then read as Rust's\n   `as i64` reads it, saturating at the ends of the 64-bit range. A negative\n   index counts from the end, as in Python: a[-1] is the last element.\n   -DVIV_RELEASE compiles the checks out: viv_ix is then an (int) cast and\n   the step from the end for a negative index. -DVIV_NO_BOUNDS\n   drops only these checks and keeps the rest of a debug build (the solver\n   certificates, the story), which is how their cost is measured\n   (tools/bounds-cost.sh). */\n#if defined(__GNUC__)\n/* The accessors are always inlined: an accessor that is called takes its\n   viv_arr (48 bytes) by value, and with the checks in it the compiler\n   stopped inlining some of them, which alone made scf/coverage.viv take\n   1.8 times as long. The failing path is cold and never returns. */\n#define VIV_ACCESS static inline __attribute__((always_inline, unused))\n#define VIV_COLD __attribute__((cold, noinline, unused))\n#define VIV_LIKELY(c) __builtin_expect(!!(c), 1)\n#else\n#define VIV_ACCESS static inline\n#define VIV_COLD\n#define VIV_LIKELY(c) (c)\n#endif\n#ifndef VIV_BOUNDS\n/* (void)(site): the site of an accessor is still passed, and read, so a\n   release build has no unused parameter to warn about */\nstatic inline int viv_ix_wrap(int i, int n) { return i < 0 ? i + n : i; }\n/* unchecked, but never undefined: the double is clamped into int range\n   before it is converted (fmax sends NaN to INT_MIN) */\n#define viv_ix(x, n, k, site) ((void)(site), viv_ix_wrap((int)fmin(fmax((x), -2147483648.0), 2147483647.0), (n)))\n#define viv_rank(have, want, site) ((void)(site))\n/* an index computed as a 64-bit integer (lower/intidx.rs) */\nstatic inline int viv_ixl_wrap(long long i, int n) { return (int)(i < 0 ? i + n : i); }\n#define viv_ixl(i, n, k, site) ((void)(site), viv_ixl_wrap((i), (n)))\n#define viv_ixp(m, k, i, n, site) (((m) >> (k)) & 1 ? (int)(i) : viv_ixl((i), (n), (k), (site)))\n#else\n/* An array indexed with the wrong number of indices: the interpreter checks\n   the count before it reads any index, so this comes first in every\n   accessor, and a[5] on a 2 x 3 array stops with the rank, not an index. */\nVIV_COLD static void viv_rank_slow(int have, int want, int site) {\n    viv_fflush(stdout);\n    viv_errf(\"error: array has rank %d but was indexed with %d index/indices\\n%s\", have, want, viv_sites[site]);\n    viv_stop(VIV_E_INDEX);\n}\n#define viv_rank(have, want, site) do { if (!VIV_LIKELY((have) == (want))) viv_rank_slow((have), (want), (site)); } while (0)\n/* An index that failed the quick test: NaN, or a number no int holds, or an\n   int out of range. Read it as `as i64` does and stop if it is out. */\nVIV_COLD static int viv_ix_slow(double x, int n, int k, int site) {\n    if (x - floor(x) != 0.0) {\n        char b[40];\n        viv_rt_text(b, sizeof b, x);\n        viv_fflush(stdout);\n        viv_errf(\"error: index %s is not an integer\\n%s\", b, viv_sites[site]);\n        viv_stop(VIV_E_INDEX);\n    }\n    long long i = x != x ? 0\n                : x >= 9223372036854775807.0 ? LLONG_MAX\n                : x < -9223372036854775807.0 ? LLONG_MIN\n                : (long long)x;\n    if (i >= 0 && i < n) return (int)i;\n    /* a negative index counts from the end */\n    if (i < 0 && i >= -(long long)n) return (int)(i + n);\n    /* what the program printed comes first, as in the interpreter */\n    viv_fflush(stdout);\n    viv_errf(\"error: index %lld out of bounds for dimension %d (size %d)\\n%s\", i, k, n, viv_sites[site]);\n    /* _exit, not exit: the run stopped, so no \"N assertions FAILED\" summary\n       or story from the atexit handlers -- the interpreter prints neither */\n    viv_stop(VIV_E_INDEX);\n}\nVIV_ACCESS int viv_ix(double x, int n, int k, int site) {\n    /* The quick test is on the int, not the double: clamped into int range\n       first (fmax also sends NaN to INT_MIN), the conversion is defined for\n       every x and truncates toward zero as `as i64` does, and one unsigned\n       compare then takes both ends. Written as two double compares instead,\n       scf/coverage.viv took 1.5 times as long: the compiler turns a loop's\n       double counter into an int only when it is indexed through an int. */\n    int t = (int)fmin(fmax(x, -2147483648.0), 2147483647.0);\n    if (VIV_LIKELY((unsigned)t < (unsigned)n && (double)t == x)) return t;\n    return viv_ix_slow(x, n, k, site);\n}\n/* An index computed as a 64-bit integer (lower/intidx.rs): a loop counter,\n   an Int, or a number the compiler proved is always an integer, so there is\n   no fraction to test for, and one unsigned compare takes both ends. An Int\n   past 2^53 stops as the interpreter stops converting it to a number. */\nVIV_COLD static int viv_ixl_slow(long long i, int n, int k, int site) {\n    if (i > 9007199254740992LL || i < -9007199254740992LL) {\n        viv_fflush(stdout);\n        viv_errf(\"error: converting %lld to a number would change it; write float(n) to round it\\n%s\", i, viv_sites[site]);\n        viv_stop(VIV_E_OVERFLOW);\n    }\n    if (i < 0 && i >= -(long long)n) return (int)(i + n);\n    /* only an error flushes: the flush ends a test block (B344) */\n    viv_fflush(stdout);\n    viv_errf(\"error: index %lld out of bounds for dimension %d (size %d)\\n%s\", i, k, n, viv_sites[site]);\n    viv_stop(VIV_E_INDEX);\n}\nVIV_ACCESS int viv_ixl(long long i, int n, int k, int site) {\n    if (VIV_LIKELY((unsigned long long)i < (unsigned long long)(long long)n)) return (int)i;\n    return viv_ixl_slow(i, n, k, site);\n}\n/* bit k of m: the compiler proved index k inside its dimension (0 <= i <\n   n, lower/intidx.rs), so it is not tested */\n#define viv_ixp(m, k, i, n, site) (((m) >> (k)) & 1 ? (int)(i) : viv_ixl((i), (n), (k), (site)))\n#endif" })
            .with(sites_fragment())
            .with(RuntimeFragment { name: "get1", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_get1_at(viv_arr a, double i, int site) {\n    viv_rank(a.rank, 1, site);\n    return a.data[viv_ix(i, a.dims[0], 0, site)];\n}\nVIV_ACCESS double viv_get1(viv_arr a, double i) { return viv_get1_at(a, i, 0); }" })
            .with(RuntimeFragment { name: "get2", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_get2_at(viv_arr a, double i, double j, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site);\n    return a.data[ii * a.dims[1] + jj];\n}\nVIV_ACCESS double viv_get2(viv_arr a, double i, double j) { return viv_get2_at(a, i, j, 0); }" })
            .with(RuntimeFragment { name: "get3", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_get3_at(viv_arr a, double i, double j, double k, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site);\n    return a.data[(ii * a.dims[1] + jj) * a.dims[2] + kk];\n}\nVIV_ACCESS double viv_get3(viv_arr a, double i, double j, double k) { return viv_get3_at(a, i, j, k, 0); }" })
            .with(RuntimeFragment { name: "get4", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_get4_at(viv_arr a, double i, double j, double k, double l, int site) {\n    viv_rank(a.rank, 4, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site); int ll = viv_ix(l, a.dims[3], 3, site);\n    return a.data[((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll];\n}\nVIV_ACCESS double viv_get4(viv_arr a, double i, double j, double k, double l) { return viv_get4_at(a, i, j, k, l, 0); }" })
            .with(RuntimeFragment { name: "lget1", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_lget1_at(viv_arr a, long long i, int m, int site) {\n    viv_rank(a.rank, 1, site);\n    return a.data[viv_ixp(m, 0, i, a.dims[0], site)];\n}\nVIV_ACCESS double viv_lget1(viv_arr a, long long i, int m) { return viv_lget1_at(a, i, m, 0); }" })
            .with(RuntimeFragment { name: "lset1", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lset1_at(viv_arr a, long long i, double v, int m, int site) {\n    viv_rank(a.rank, 1, site);\n    a.data[viv_ixp(m, 0, i, a.dims[0], site)] = v;\n}\nVIV_ACCESS void viv_lset1(viv_arr a, long long i, double v, int m) { viv_lset1_at(a, i, v, m, 0); }" })
            .with(RuntimeFragment { name: "lget2", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_lget2_at(viv_arr a, long long i, long long j, int m, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site);\n    return a.data[ii * a.dims[1] + jj];\n}\nVIV_ACCESS double viv_lget2(viv_arr a, long long i, long long j, int m) { return viv_lget2_at(a, i, j, m, 0); }" })
            .with(RuntimeFragment { name: "lset2", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lset2_at(viv_arr a, long long i, long long j, double v, int m, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site);\n    a.data[ii * a.dims[1] + jj] = v;\n}\nVIV_ACCESS void viv_lset2(viv_arr a, long long i, long long j, double v, int m) { viv_lset2_at(a, i, j, v, m, 0); }" })
            .with(RuntimeFragment { name: "lget3", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_lget3_at(viv_arr a, long long i, long long j, long long k, int m, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site); int kk = viv_ixp(m, 2, k, a.dims[2], site);\n    return a.data[(ii * a.dims[1] + jj) * a.dims[2] + kk];\n}\nVIV_ACCESS double viv_lget3(viv_arr a, long long i, long long j, long long k, int m) { return viv_lget3_at(a, i, j, k, m, 0); }" })
            .with(RuntimeFragment { name: "lset3", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lset3_at(viv_arr a, long long i, long long j, long long k, double v, int m, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site); int kk = viv_ixp(m, 2, k, a.dims[2], site);\n    a.data[(ii * a.dims[1] + jj) * a.dims[2] + kk] = v;\n}\nVIV_ACCESS void viv_lset3(viv_arr a, long long i, long long j, long long k, double v, int m) { viv_lset3_at(a, i, j, k, v, m, 0); }" })
            .with(RuntimeFragment { name: "lget4", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS double viv_lget4_at(viv_arr a, long long i, long long j, long long k, long long l, int m, int site) {\n    viv_rank(a.rank, 4, site);\n    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site); int kk = viv_ixp(m, 2, k, a.dims[2], site); int ll = viv_ixp(m, 3, l, a.dims[3], site);\n    return a.data[((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll];\n}\nVIV_ACCESS double viv_lget4(viv_arr a, long long i, long long j, long long k, long long l, int m) { return viv_lget4_at(a, i, j, k, l, m, 0); }" })
            .with(RuntimeFragment { name: "lset4", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_lset4_at(viv_arr a, long long i, long long j, long long k, long long l, double v, int m, int site) {\n    viv_rank(a.rank, 4, site);\n    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site); int kk = viv_ixp(m, 2, k, a.dims[2], site); int ll = viv_ixp(m, 3, l, a.dims[3], site);\n    a.data[((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll] = v;\n}\nVIV_ACCESS void viv_lset4(viv_arr a, long long i, long long j, long long k, long long l, double v, int m) { viv_lset4_at(a, i, j, k, l, v, m, 0); }" })
            .with(RuntimeFragment { name: "export", includes: &[], deps: &[], code: EXPORT_RUNTIME })
            .with(RuntimeFragment { name: "export_str", includes: &["#include <string.h>"], deps: &["export", "str_core"], code: EXPORT_STR_RUNTIME })
            .with(RuntimeFragment { name: "export_arr", includes: &["#include <string.h>"], deps: &["export", "arr_new"], code: EXPORT_ARR_RUNTIME })
            .with(RuntimeFragment { name: "clock", includes: &["#include <time.h>"], deps: &[], code: "/* clock(): the wall time in seconds from a monotonic clock, for a\n   `#! timing` program; only the difference of two readings means anything */\nstatic double viv_clock(void) {\n    struct timespec t;\n    clock_gettime(CLOCK_MONOTONIC, &t);\n    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;\n}" })
            .with(RuntimeFragment { name: "depth", includes: &[], deps: &[], code: "/* the recursion counter (crate::recursion): the depth of nested calls of\n   recursive defs, as viv__depth[0] holds it in the interpreter */\nstatic long viv_depth_g = 0;" })
            .with(RuntimeFragment { name: "trip", includes: &["#include <math.h>"], deps: &[], code: "/* the passes of `for i to x` for a number x: the interpreter's double\n   counter runs while it is below x, so ceil(x) passes, none for 0, a\n   negative x or NaN; past 2^53 that counter stops advancing, and the\n   integer count stops there (lower/intidx.rs) */\nstatic inline long long viv_trip(double x) { return x > 0.0 ? (x < 9007199254740992.0 ? (long long)ceil(x) : 9007199254740992LL) : 0; }\n/* the guard of an integer index computed from a number variable: the\n   variable is an integer of magnitude at most g (false for NaN) */\nstatic inline int viv_wg(double x, double g) { return fabs(x) <= g && x == (double)(long long)x; }" })
            .with(RuntimeFragment { name: "set1", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_set1_at(viv_arr a, double i, double v, int site) {\n    viv_rank(a.rank, 1, site);\n    a.data[viv_ix(i, a.dims[0], 0, site)] = v;\n}\nVIV_ACCESS void viv_set1(viv_arr a, double i, double v) { viv_set1_at(a, i, v, 0); }" })
            .with(RuntimeFragment { name: "disjoint", includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <unistd.h>"], deps: &["arr_type", "bounds", "arena", "set1", "set2", "set3", "set4"], code: DJ_RUNTIME })
            .with(RuntimeFragment { name: "set2", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_set2_at(viv_arr a, double i, double j, double v, int site) {\n    viv_rank(a.rank, 2, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site);\n    a.data[ii * a.dims[1] + jj] = v;\n}\nVIV_ACCESS void viv_set2(viv_arr a, double i, double j, double v) { viv_set2_at(a, i, j, v, 0); }" })
            .with(RuntimeFragment { name: "set3", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_set3_at(viv_arr a, double i, double j, double k, double v, int site) {\n    viv_rank(a.rank, 3, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site);\n    a.data[(ii * a.dims[1] + jj) * a.dims[2] + kk] = v;\n}\nVIV_ACCESS void viv_set3(viv_arr a, double i, double j, double k, double v) { viv_set3_at(a, i, j, k, v, 0); }" })
            .with(RuntimeFragment { name: "set4", includes: &[], deps: &["arr_type", "bounds"], code: "VIV_ACCESS void viv_set4_at(viv_arr a, double i, double j, double k, double l, double v, int site) {\n    viv_rank(a.rank, 4, site);\n    int ii = viv_ix(i, a.dims[0], 0, site); int jj = viv_ix(j, a.dims[1], 1, site); int kk = viv_ix(k, a.dims[2], 2, site); int ll = viv_ix(l, a.dims[3], 3, site);\n    a.data[((ii * a.dims[1] + jj) * a.dims[2] + kk) * a.dims[3] + ll] = v;\n}\nVIV_ACCESS void viv_set4(viv_arr a, double i, double j, double k, double l, double v) { viv_set4_at(a, i, j, k, l, v, 0); }" })
            .with(RuntimeFragment { name: "arr_add", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_add(viv_arr a, viv_arr b) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] + b.data[i];\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_sub", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_sub(viv_arr a, viv_arr b) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] - b.data[i];\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_had", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_had(viv_arr a, viv_arr b) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] * b.data[i];\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_scale", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_scale(viv_arr a, double s) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] * s;\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_div", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_div(viv_arr a, viv_arr b) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] / b.data[i];\n    return r;\n}" })
            .with(RuntimeFragment { name: "carr_arith", includes: &["#include <complex.h>"], deps: &["carr_new"], code: "/* Complex array arithmetic, written out in real arithmetic on the packed\n   (re, im) pairs -- the same expressions, in the same order, as the\n   interpreter's, so the two agree to the last bit. */\nVIV_SCALAR_LOOPS static viv_arr viv_carr_add(viv_arr a, viv_arr b) {\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    for (int i = 0; i < 2 * r.size; i++) r.data[i] = a.data[i] + b.data[i];\n    return r;\n}\nVIV_SCALAR_LOOPS static viv_arr viv_carr_sub(viv_arr a, viv_arr b) {\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    for (int i = 0; i < 2 * r.size; i++) r.data[i] = a.data[i] - b.data[i];\n    return r;\n}\nVIV_SCALAR_LOOPS static viv_arr viv_carr_had(viv_arr a, viv_arr b) {\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) {\n        double ar = a.data[2*i], ai = a.data[2*i+1], br = b.data[2*i], bi = b.data[2*i+1];\n        r.data[2*i] = ar * br - ai * bi;\n        r.data[2*i+1] = ar * bi + ai * br;\n    }\n    return r;\n}\nVIV_SCALAR_LOOPS static viv_arr viv_carr_hadr(viv_arr a, viv_arr b) {\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) {\n        double ar = a.data[2*i], ai = a.data[2*i+1], br = b.data[i], bi = 0.0;\n        r.data[2*i] = ar * br - ai * bi;\n        r.data[2*i+1] = ar * bi + ai * br;\n    }\n    return r;\n}\nVIV_SCALAR_LOOPS static viv_arr viv_carr_scale(viv_arr a, double complex s) {\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    double br = creal(s), bi = cimag(s);\n    for (int i = 0; i < r.size; i++) {\n        double ar = a.data[2*i], ai = a.data[2*i+1];\n        r.data[2*i] = ar * br - ai * bi;\n        r.data[2*i+1] = ar * bi + ai * br;\n    }\n    return r;\n}\nVIV_SCALAR_LOOPS static viv_arr viv_carr_neg(viv_arr a) {\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    for (int i = 0; i < 2 * r.size; i++) r.data[i] = -a.data[i];\n    return r;\n}\nVIV_SCALAR_LOOPS static viv_arr viv_carr_divs(viv_arr a, double complex s) {\n    /* Smith's division (vmath.c), as the scalar `/` */\n    viv_arr r = viv_carr_new(a.rank, a.dims);\n    double br = creal(s), bi = cimag(s);\n    for (int i = 0; i < r.size; i++) viv_cdiv_parts(a.data[2*i], a.data[2*i+1], br, bi, &r.data[2*i], &r.data[2*i+1]);\n    return r;\n}" })
            .with(RuntimeFragment { name: "cmatmul", includes: &[], deps: &["carr_new"], code: "VIV_SCALAR_LOOPS static viv_arr viv_cmatmul(viv_arr a, viv_arr b) {\n    /* same shape rules as viv_matmul; same i-p-j order and zero skip as the\n       interpreter, so every element is the same sum in the same order */\n    int m = (a.rank == 2) ? a.dims[0] : 1;\n    int k = (a.rank == 2) ? a.dims[1] : a.dims[0];\n    int n = (b.rank == 2) ? b.dims[1] : 1;\n    viv_arr r;\n    if (a.rank == 1 || b.rank == 1) {\n        int dims[1]; dims[0] = m * n;\n        r = viv_carr_new(1, dims);\n    } else {\n        int dims[2]; dims[0] = m; dims[1] = n;\n        r = viv_carr_new(2, dims);\n    }\n    for (int i = 0; i < m; i++) for (int p = 0; p < k; p++) {\n        double ar = a.data[2*(i*k+p)], ai = a.data[2*(i*k+p)+1];\n        if (ar == 0.0 && ai == 0.0) continue;\n        const double* bp = b.data + 2*(size_t)p*n;\n        double* rp = r.data + 2*(size_t)i*n;\n        for (int j = 0; j < n; j++) {\n            double br = bp[2*j], bi = bp[2*j+1];\n            rp[2*j] += ar * br - ai * bi;\n            rp[2*j+1] += ar * bi + ai * br;\n        }\n    }\n    return r;\n}" })
            .with(RuntimeFragment { name: "ctranspose", includes: &[], deps: &["carr_new"], code: "static viv_arr viv_ctranspose(viv_arr a) {\n    int rows = a.dims[0], cols = a.dims[1];\n    int dims[2]; dims[0] = cols; dims[1] = rows;\n    viv_arr r = viv_carr_new(2, dims);\n    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) {\n        r.data[2*(j*rows+i)] = a.data[2*(i*cols+j)];\n        r.data[2*(j*rows+i)+1] = a.data[2*(i*cols+j)+1];\n    }\n    return r;\n}" })
            .with(RuntimeFragment {
                name: "slices",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <limits.h>", "#include <unistd.h>"],
                deps: &["arr_new", "carr_new", "sites"],
                code: crate::interp::slices::c_runtime(),
            })
            // fused elementwise trees and reductions over views (crate::fuse)
            .with(RuntimeFragment {
                name: "fuse",
                includes: &["#include <stdarg.h>", "#include <string.h>", "#include <math.h>"],
                deps: &["arr_add", "arr_sub", "arr_had", "arr_div", "arr_adds", "arr_subs", "arr_subsr", "arr_scale", "arr_divs", "arr_divsr", "arr_pows", "arr_map", "arr_chk", "slices", "sum", "arrays", "arr_extent"],
                code: crate::fuse::C_RUNTIME,
            })
            .with(RuntimeFragment { name: "slices_c", includes: &["#include <complex.h>"], deps: &["slices"], code: crate::interp::slices::C_CPLX })
            // the array vocabulary (interp/arrays.rs), and its parts that need
            // another fragment
            .with(RuntimeFragment {
                name: "arrays",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <limits.h>", "#include <unistd.h>", "#include <math.h>"],
                deps: &["arr_new", "sites", "fmt_g"],
                code: crate::interp::arrays::C_RUNTIME,
            })
            .with(RuntimeFragment { name: "ar_rand", includes: &[], deps: &["arrays", "rng"], code: crate::interp::arrays::C_RAND })
            .with(RuntimeFragment { name: "ar_einsum", includes: &[], deps: &["arrays"], code: crate::interp::arrays::C_EINSUM })
            .with(RuntimeFragment { name: "ar_solve", includes: &[], deps: &["arrays", "lin_solve", "lin_solve_multi"], code: crate::interp::arrays::C_SOLVE })
            .with(RuntimeFragment { name: "ar_chol", includes: &[], deps: &["arrays", "cholesky"], code: crate::interp::arrays::C_CHOL })
            .with(RuntimeFragment { name: "ar_svd", includes: &[], deps: &["arrays"], code: crate::interp::arrays::C_SVD })
            .with(RuntimeFragment { name: "ad_unb", includes: &[], deps: &["arr_new"], code: crate::interp::arrays::C_AD_UNB })
            .with(RuntimeFragment { name: "ad_arr", includes: &["#include <stdio.h>", "#include <stdlib.h>"], deps: &["arr_new", "fmt_g"], code: crate::interp::arrays::C_AD_ARR })
            .with(RuntimeFragment {
                name: "maps",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <stdarg.h>"],
                deps: &["slices", "arr_new", "print"],
                code: crate::interp::maps::C_RUNTIME,
            })
            .with(RuntimeFragment {
                name: "smaps",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <stdarg.h>"],
                deps: &["maps", "str_core", "str_list", "str_repr", "print", "arr_new"],
                code: crate::interp::maps::c_smaps(),
            })
            .with(RuntimeFragment {
                name: "growable",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>"],
                deps: &["slices", "arena", "arr_new"],
                code: crate::interp::growable::C_RUNTIME,
            })
            .with(RuntimeFragment { name: "slab", includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>"], deps: &["arr_new", "carr_new"], code: "/* slab / setslab: the k-th slab along the leading axis, copied out or\n   written back. `w` is the element width in doubles: 1 real, 2 complex. */\nstatic viv_arr viv_slab(viv_arr a, double kf, double wf) {\n    int w = (int)wf, k = (int)kf;\n    if (a.rank < 2 || kf < 0.0 || k >= a.dims[0]) { viv_errf(\"slab: index %d out of range, or rank below 2\\n\", k); viv_stop_exit(VIV_E_INDEX); }\n    viv_arr r = (w == 2) ? viv_carr_new(a.rank - 1, a.dims + 1) : viv_arr_new(a.rank - 1, a.dims + 1);\n    memcpy(r.data, a.data + (size_t)w * k * r.size, (size_t)w * r.size * sizeof(double));\n    return r;\n}\nstatic double viv_setslab(viv_arr a, double kf, viv_arr s, double wf) {\n    int w = (int)wf, k = (int)kf;\n    if (a.rank < 2 || kf < 0.0 || k >= a.dims[0]) { viv_errf(\"setslab: index %d out of range, or rank below 2\\n\", k); viv_stop_exit(VIV_E_INDEX); }\n    int len = a.size / a.dims[0];\n    if (s.size != len) { viv_errf(\"setslab: the slab has %d elements, the array's slabs have %d\\n\", s.size, len); viv_stop_exit(VIV_E_VALUE); }\n    memcpy(a.data + (size_t)w * k * len, s.data, (size_t)w * len * sizeof(double));\n    return 0.0;\n}" })
            .with(RuntimeFragment { name: "take_put", includes: &["#include <stdio.h>", "#include <stdlib.h>"], deps: &["arr_new", "carr_new", "fmt_g"], code: "/* take / put: gather and scatter through a real array of FLAT indices. */\nstatic viv_arr viv_take(viv_arr a, viv_arr ix, double wf) {\n    int w = (int)wf;\n    viv_arr r = (w == 2) ? viv_carr_new(ix.rank, ix.dims) : viv_arr_new(ix.rank, ix.dims);\n    for (int j = 0; j < ix.size; j++) {\n        double f = ix.data[j]; int k = (int)f;\n        if (f < 0.0 || k >= a.size) { viv_errf(\"take: index %s out of range (the array has %d elements)\\n\", viv_g(f), a.size); viv_stop_exit(VIV_E_INDEX); }\n        for (int c = 0; c < w; c++) r.data[w*j+c] = a.data[w*k+c];\n    }\n    return r;\n}\nstatic double viv_put(viv_arr a, viv_arr ix, viv_arr v, double wf) {\n    int w = (int)wf;\n    if (v.size != ix.size) { viv_errf(\"put: as many values as indices, please\\n\"); viv_stop_exit(VIV_E_VALUE); }\n    for (int j = 0; j < ix.size; j++) {\n        double f = ix.data[j]; int k = (int)f;\n        if (f < 0.0 || k >= a.size) { viv_errf(\"put: index %s out of range (the array has %d elements)\\n\", viv_g(f), a.size); viv_stop_exit(VIV_E_INDEX); }\n        for (int c = 0; c < w; c++) a.data[w*k+c] = v.data[w*j+c];\n    }\n    return 0.0;\n}" })
            .with(RuntimeFragment { name: "ad", includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>"], deps: &["arr_new", "matmul", "mm_adj", "transpose", "fmt_g", "arr_chk", "sum"], code: "/* The tape of reverse-mode autodiff (v4), and the adjoint\n   helpers the generated gradient defs call; the interpreter's ad_builtin is\n   their twin. The tape is a stack of doubles: a number is one entry, an array\n   its data, then its dims, then its rank. */\nstatic double* viv_ad_t = 0;\nstatic size_t viv_ad_n = 0, viv_ad_cap = 0;\nstatic void viv_ad_room(size_t k) {\n    if (viv_ad_n + k <= viv_ad_cap) return;\n    size_t cap = viv_ad_cap ? viv_ad_cap : 4096;\n    while (cap < viv_ad_n + k) cap *= 2;\n    viv_ad_t = (double*)realloc(viv_ad_t, cap * sizeof(double));\n    if (!viv_ad_t) { viv_fflush(stdout); viv_errf(\"error: out of memory for the autodiff tape\\n\"); viv_stop_exit(VIV_E_MEMORY); }\n    viv_ad_cap = cap;\n}\nstatic double viv_ad_push_n(double x) { viv_ad_room(1); viv_ad_t[viv_ad_n++] = x; return 0.0; }\nstatic double viv_ad_push_a(viv_arr a) {\n    viv_ad_room((size_t)a.size + (size_t)a.rank + 1);\n    if (a.size) memcpy(viv_ad_t + viv_ad_n, a.data, (size_t)a.size * sizeof(double));\n    viv_ad_n += (size_t)a.size;\n    for (int i = 0; i < a.rank; i++) viv_ad_t[viv_ad_n++] = (double)a.dims[i];\n    viv_ad_t[viv_ad_n++] = (double)a.rank;\n    return 0.0;\n}\nstatic double viv_ad_pop_n(void) { return viv_ad_t[--viv_ad_n]; }\nstatic double viv_ad_pop_b(void) { return viv_ad_t[--viv_ad_n]; }\nstatic viv_arr viv_ad_pop_a(void) {\n    int rank = (int)viv_ad_t[--viv_ad_n];\n    int dims[8];\n    for (int i = rank - 1; i >= 0; i--) dims[i] = (int)viv_ad_t[--viv_ad_n];\n    viv_arr r = viv_arr_new(rank, dims);\n    viv_ad_n -= (size_t)r.size;\n    if (r.size) memcpy(r.data, viv_ad_t + viv_ad_n, (size_t)r.size * sizeof(double));\n    return r;\n}\nstatic double viv_ad_restore(viv_arr a) {\n    int rank = (int)viv_ad_t[--viv_ad_n];\n    size_t size = 1;\n    for (int i = 0; i < rank; i++) size *= (size_t)viv_ad_t[--viv_ad_n];\n    viv_ad_n -= size;\n    if (size != (size_t)a.size) { viv_fflush(stdout); viv_errf(\"error: autodiff: an array changed size between the sweeps\\n\"); viv_stop_exit(VIV_E_ERROR); }\n    if (size) memcpy(a.data, viv_ad_t + viv_ad_n, size * sizeof(double));\n    return 0.0;\n}\nstatic viv_arr viv_ad_zero(viv_arr a) { return viv_arr_new(a.rank, a.dims); }\nstatic double viv_ad_unb_an(viv_arr g, double like) { (void)like; return viv_sum(g); }\nstatic viv_arr viv_ad_unb_na(double g, viv_arr like) { viv_arr r = viv_arr_new_nz(like.rank, like.dims); for (int i = 0; i < r.size; i++) r.data[i] = g; return r; }\n/* 1 where a > b, 1/2 where a == b, 0 elsewhere: the share of max(a, b) */\nstatic double viv_ad_share(double a, double b) { return a > b ? 1.0 : (a == b ? 0.5 : 0.0); }\nstatic double viv_ad_cmp_nn(double a, double b) { return viv_ad_share(a, b); }\nstatic viv_arr viv_ad_cmp_an(viv_arr a, double b) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = viv_ad_share(a.data[i], b); return r; }\nstatic viv_arr viv_ad_cmp_na(double a, viv_arr b) { viv_arr r = viv_arr_new_nz(b.rank, b.dims); for (int i = 0; i < r.size; i++) r.data[i] = viv_ad_share(a, b.data[i]); return r; }\nstatic viv_arr viv_ad_cmp_aa(viv_arr a, viv_arr b) {\n    int same = a.rank == b.rank;\n    for (int i = 0; same && i < a.rank; i++) same = a.dims[i] == b.dims[i];\n    if (!same) {\n        /* operands that were broadcast (interp/arrays.rs) */\n        int rank = a.rank > b.rank ? a.rank : b.rank, to[8];\n        for (int d = 0; d < rank; d++) {\n            int x = d + a.rank >= rank ? a.dims[d + a.rank - rank] : 1, y = d + b.rank >= rank ? b.dims[d + b.rank - rank] : 1;\n            if (x != y && x != 1 && y != 1) { viv_fflush(stdout); viv_errf(\"error: autodiff: array shape mismatch in max or min\\n\"); viv_stop_exit(VIV_E_VALUE); }\n            to[d] = x == 1 ? y : x;\n        }\n        a = viv_bc_expand(a, rank, to);\n        b = viv_bc_expand(b, rank, to);\n    }\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = viv_ad_share(a.data[i], b.data[i]);\n    return r;\n}\nstatic viv_arr viv_ad_outer(viv_arr u, viv_arr v) {\n    int dims[2]; dims[0] = u.size; dims[1] = v.size;\n    viv_arr r = viv_arr_new(2, dims);\n    for (int i = 0; i < u.size; i++) for (int j = 0; j < v.size; j++) r.data[i*v.size+j] = u.data[i] * v.data[j];\n    return r;\n}\n/* the adjoints of C = A @ B: dA = dC @ B^T and dB = A^T @ dC, or an outer\n   product where a vector operand makes C a vector */\nstatic viv_arr viv_ad_mm_a(viv_arr g, viv_arr a, viv_arr b) {\n    if (a.rank > 2 || b.rank > 2) return viv_mm_adj(g, a, b, 0);\n    if (a.rank == 2 && b.rank == 1) return viv_ad_outer(g, b);\n    return viv_matmul(g, viv_transpose(b));\n}\nstatic viv_arr viv_ad_mm_b(viv_arr g, viv_arr a, viv_arr b) {\n    if (a.rank > 2 || b.rank > 2) return viv_mm_adj(g, a, b, 1);\n    if (a.rank == 1) return viv_ad_outer(a, g);\n    return viv_matmul(viv_transpose(a), g);\n}\n/* the adjoint of take(A, idx): g added into A's shape at each flat index, in\n   order, so a repeated index collects every share */\nstatic viv_arr viv_ad_take(viv_arr g, viv_arr a, viv_arr ix) {\n    viv_arr r = viv_arr_new(a.rank, a.dims);\n    for (int j = 0; j < ix.size; j++) {\n        double f = ix.data[j]; int k = (int)f;\n        if (f < 0.0 || k >= r.size) { viv_fflush(stdout); viv_errf(\"take: index %s out of range (the array has %d elements)\\n\", viv_g(f), r.size); viv_stop_exit(VIV_E_INDEX); }\n        r.data[k] += g.data[j];\n    }\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_pows", includes: &["#include <math.h>"], deps: &["arr_new"], code: "/* every entry to the same scalar power */\nstatic viv_arr viv_arr_pows(viv_arr a, double s) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = pow(a.data[i], s); return r; }" })
            .with(RuntimeFragment { name: "arr_minmax", includes: &["#include <math.h>"], deps: &["arr_new"], code: "/* elementwise max / min: array with array, or array with scalar */\nstatic viv_arr viv_arr_max(viv_arr a, viv_arr b) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = fmax(a.data[i], b.data[i]); return r; }\nstatic viv_arr viv_arr_min(viv_arr a, viv_arr b) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = fmin(a.data[i], b.data[i]); return r; }\nstatic viv_arr viv_arr_maxs(viv_arr a, double s) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = fmax(a.data[i], s); return r; }\nstatic viv_arr viv_arr_mins(viv_arr a, double s) { viv_arr r = viv_arr_new_nz(a.rank, a.dims); for (int i = 0; i < r.size; i++) r.data[i] = fmin(a.data[i], s); return r; }" })
            .with(RuntimeFragment { name: "tridiag_solve", includes: &["#include <stdio.h>", "#include <stdlib.h>"], deps: &["arr_new"], code: "/* The Thomas algorithm: a tridiagonal solve in O(n), no pivoting -- for the\n   diagonally dominant systems implicit PDE schemes produce. a = sub-diagonal\n   (a[0] unused), b = diagonal, c = super-diagonal (c[n-1] unused). The\n   interpreter's arm, operation for operation. */\nstatic viv_arr viv_tridiag_solve(viv_arr a, viv_arr b, viv_arr c, viv_arr d) {\n    int n = b.size;\n    if (n == 0 || a.size != n || c.size != n || d.size != n) { viv_errf(\"tridiag_solve takes four 1-D arrays of one length: (sub, diag, super, rhs)\\n\"); viv_stop_exit(VIV_E_VALUE); }\n    int dims[1]; dims[0] = n;\n    viv_arr x = viv_arr_new(1, dims);\n    double* cp = (double*)malloc((size_t)n * sizeof(double));\n    double* dp = (double*)malloc((size_t)n * sizeof(double));\n    if (b.data[0] == 0.0) { viv_errf(\"tridiag_solve: zero pivot at row 0 (the system is not diagonally dominant)\\n\"); viv_stop_exit(VIV_E_VALUE); }\n    cp[0] = c.data[0] / b.data[0];\n    dp[0] = d.data[0] / b.data[0];\n    for (int i = 1; i < n; i++) {\n        double m = b.data[i] - a.data[i] * cp[i-1];\n        if (m == 0.0) { viv_errf(\"tridiag_solve: zero pivot at row %d (the system is not diagonally dominant)\\n\", i); viv_stop_exit(VIV_E_VALUE); }\n        cp[i] = c.data[i] / m;\n        dp[i] = (d.data[i] - a.data[i] * dp[i-1]) / m;\n    }\n    x.data[n-1] = dp[n-1];\n    for (int i = n - 2; i >= 0; i--) x.data[i] = dp[i] - cp[i] * x.data[i+1];\n    free(cp); free(dp);\n    return x;\n}" })
            .with(RuntimeFragment { name: "roll", includes: &["#include <stdio.h>", "#include <stdlib.h>"], deps: &["arr_new"], code: "/* roll(a, k, axis): circular shift -- element i moves to i + k, wrapping. axis < 0\n   means \"none given\", which is allowed for a 1-D array only. */\nstatic viv_arr viv_roll(viv_arr a, double kf, double axisf) {\n    int axis = (int)axisf;\n    if (axisf < 0.0) {\n        if (a.rank != 1) { viv_errf(\"roll(a, k) is for a 1-D array; give the axis for a matrix: roll(A, k, axis)\\n\"); viv_stop_exit(VIV_E_VALUE); }\n        axis = 0;\n    }\n    if (axis >= a.rank) { viv_errf(\"roll: axis %d out of range for a rank-%d array\\n\", axis, a.rank); viv_stop_exit(VIV_E_INDEX); }\n    int len = a.dims[axis], outer = 1, inner = 1;\n    for (int i = 0; i < axis; i++) outer *= a.dims[i];\n    for (int i = axis + 1; i < a.rank; i++) inner *= a.dims[i];\n    long long k = (long long)kf;\n    int sh = (int)(((k % len) + len) % len);\n    viv_arr r = viv_arr_new(a.rank, a.dims);\n    for (int o = 0; o < outer; o++) for (int j = 0; j < len; j++) {\n        int t = (j + sh) % len;\n        for (int q = 0; q < inner; q++) r.data[(o * len + t) * inner + q] = a.data[(o * len + j) * inner + q];\n    }\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_divsr", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_divsr(viv_arr a, double s) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = s / a.data[i];\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_divs", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_divs(viv_arr a, double s) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] / s;\n    return r;\n}" })
            .with(RuntimeFragment { name: "fastla", includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"], deps: &[], code: FAST_LA })
            .with(RuntimeFragment { name: "matmul", includes: &[], deps: &["arr_new", "fastla"], code: crate::interp::matmul::C_MATMUL })
            .with(RuntimeFragment { name: "mm_adj", includes: &[], deps: &["matmul", "ad_unb"], code: crate::interp::matmul::C_MM_ADJ })
            .with(RuntimeFragment { name: "histogram", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_histogram(viv_arr a, double lo, double hi, double nbd) {\n    int nb = (int)nbd; if (nb < 1) nb = 1;\n    int dims[1]; dims[0] = nb;\n    viv_arr r = viv_arr_new(1, dims);\n    for (int i = 0; i < nb; i++) r.data[i] = 0.0;\n    double w = (hi - lo) / (double)nb;\n    for (int i = 0; i < a.size; i++) {\n        double v = a.data[i];\n        if (v < lo || v > hi || w <= 0.0) continue;\n        int b = (int)((v - lo) / w);\n        if (b >= nb) b = nb - 1;\n        r.data[b] += 1.0;\n    }\n    return r;\n}" })
            .with(RuntimeFragment { name: "bars", includes: &["#include <stdio.h>"], deps: &["arr_type", "prt"], code: "static double viv_bars(const char* label, viv_arr x, viv_arr y) {\n    viv_printf(\"bars: %s ; [\", label);\n    for (int i = 0; i < x.size; i++) { if (i) viv_printf(\", \"); viv_prt(x.data[i]); }\n    viv_printf(\"] ; [\");\n    for (int i = 0; i < y.size; i++) { if (i) viv_printf(\", \"); viv_prt(y.data[i]); }\n    viv_printf(\"]\\n\");\n    return 0.0;\n}" })
            .with(RuntimeFragment { name: "linspace", includes: &[], deps: &["arr_new"], code: "/* NumPy's: n = 1 is [lo], n = 0 is empty */\nstatic viv_arr viv_linspace(double lo, double hi, double nd) {\n    int n = (int)nd;\n    if (n < 0) { viv_errf(\"error: `linspace`: the number of points %d is negative\\n\", n); viv_stop_exit(VIV_E_VALUE); }\n    int dims[1]; dims[0] = n;\n    viv_arr r = viv_arr_new(1, dims);\n    double h = n > 1 ? (hi - lo) / (double)(n - 1) : 0.0;\n    for (int i = 0; i < n; i++) r.data[i] = lo + h * (double)i;\n    if (n > 1) r.data[n-1] = hi;\n    return r;\n}" })
            .with(RuntimeFragment { name: "transpose", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_transpose(viv_arr a) {\n    if (a.rank != 2) {\n        /* any other rank: the axes reversed, a vector copied (interp/arrays.rs, `transpose_any`) */\n        int od[8]; long long st[8], idx[8];\n        for (int d = a.rank - 1; d >= 0; d--) st[d] = d == a.rank - 1 ? 1 : st[d + 1] * a.dims[d + 1];\n        for (int d = 0; d < a.rank; d++) { od[d] = a.dims[a.rank - 1 - d]; idx[d] = 0; }\n        viv_arr r = viv_arr_new(a.rank, od);\n        for (long long q = 0; q < r.size; q++) {\n            long long off = 0;\n            for (int d = 0; d < a.rank; d++) off += idx[d] * st[a.rank - 1 - d];\n            r.data[q] = a.data[off];\n            for (int d = a.rank - 1; d >= 0; d--) { if (++idx[d] < od[d]) break; idx[d] = 0; }\n        }\n        return r;\n    }\n    int rows = a.dims[0], cols = a.dims[1];\n    int dims[2]; dims[0] = cols; dims[1] = rows;\n    viv_arr r = viv_arr_new(2, dims);\n    for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) r.data[j*rows+i] = a.data[i*cols+j];\n    return r;\n}" })
            // A 1-arg double->double closure: function pointer + captured doubles.
            .with(RuntimeFragment { name: "clos1", includes: &[], deps: &[], code: "typedef struct { double (*fn)(const double*, double); const double* caps; } viv_clos1;" })
            // First-class function VALUES: a def or lambda stored in a variable
            // (and, later, an array). Uniform ABI: call takes (args[], n) and
            // returns a double. viv_apply is the variadic dispatcher the ApplyFn
            // node calls; each def used as a value gets a viv_wrap_<name>.
            .with(RuntimeFragment { name: "viv_fnarr", includes: &["#include <stdlib.h>", "#include <stdarg.h>"], deps: &["viv_fn", "arena", "bounds"], code: "/* A REGISTRY: an array of function values, which is how a contract holds its\n   strategies as data (`methods = [r_bisect, r_illinois, r_safe]`). A viv_fn is\n   a bare function pointer with no captures, so the array is packed -- one\n   contiguous block, no tag per slot -- exactly like the numeric arrays.\n   It rides the same arena as those: the arena holds `void*` and only ever\n   compares and frees what it is given, never reads through it. */\ntypedef struct { int size; viv_fn* data; } viv_fnarr;\nstatic viv_fnarr viv_fnvec(int n, ...) {\n    viv_fnarr a; a.size = n;\n    a.data = (viv_fn*)calloc((size_t)n, sizeof(viv_fn));\n    viv_arena_push((double*)a.data);\n    va_list ap; va_start(ap, n);\n    for (int i = 0; i < n; i++) a.data[i] = va_arg(ap, viv_fn);\n    va_end(ap);\n    return a;\n}\nVIV_ACCESS viv_fn viv_fnget_at(viv_fnarr a, double i, int site) { return a.data[viv_ix(i, a.size, 0, site)]; }\nVIV_ACCESS viv_fn viv_fnget(viv_fnarr a, double i) { return viv_fnget_at(a, i, 0); }\nstatic double viv_fnlen(viv_fnarr a) { return (double)a.size; }" })
            .with(RuntimeFragment { name: "viv_fn", includes: &["#include <stdarg.h>"], deps: &["arr_type"], code: "/* A function VALUE and the uniform ABI it is called through. Arguments travel\n   as viv_arg, a union of a number, a function and an array, because a strategy\n   takes the function it works on as its first argument and may be handed, or\n   hand back, a whole array. The union carries NO tag: which\n   member each slot holds is fixed by the callee\'s signature, which the\n   compiler knows at both ends, so the choice is made when the code is emitted\n   rather than looked up when it runs. */\ntypedef union viv_arg viv_arg;\ntypedef struct viv_fn { viv_arg (*call)(const viv_arg*, int); } viv_fn;\nunion viv_arg { double num; viv_fn fn; viv_arr arr; };\nstatic viv_arg viv_argn(double x) { viv_arg v; v.num = x; return v; }\nstatic viv_arg viv_argf(viv_fn f) { viv_arg v; v.fn = f; return v; }\nstatic viv_arg viv_arga(viv_arr x) { viv_arg v; v.arr = x; return v; }\nstatic viv_arg viv_apply(viv_fn f, int n, ...) {\n    viv_arg a[16];\n    va_list ap; va_start(ap, n);\n    for (int i = 0; i < n && i < 16; i++) a[i] = va_arg(ap, viv_arg);\n    va_end(ap);\n    return f.call(a, n);\n}" })
            // simpson/trapezoid mirror the reference stdlib formulas exactly so
            // results match bit-for-bit.
            .with(RuntimeFragment {
                name: "simpson",
                includes: &["#include <math.h>"],
                deps: &["clos1"],
                code: "static double viv_simpson(viv_clos1 f, double a, double b, double nd) {\n    double n = nd;\n    if (fmod(n, 2.0) != 0.0) { n += 1.0; }\n    double h = (b - a) / n;\n    double s = f.fn(f.caps, a) + f.fn(f.caps, b);\n    double k = 0.0;\n    while (k < n - 1.0) {\n        double x = a + (k + 1.0) * h;\n        if (fmod(k + 1.0, 2.0) == 1.0) { s += 4.0 * f.fn(f.caps, x); }\n        else { s += 2.0 * f.fn(f.caps, x); }\n        k += 1.0;\n    }\n    return s * h / 3.0;\n}",
            })
            .with(RuntimeFragment {
                name: "trapezoid",
                includes: &[],
                deps: &["clos1"],
                code: "static double viv_trapezoid(viv_clos1 f, double a, double b, double nd) {\n    double n = nd;\n    double h = (b - a) / n;\n    double s = (f.fn(f.caps, a) + f.fn(f.caps, b)) / 2.0;\n    double k = 0.0;\n    while (k < n - 1.0) {\n        s += f.fn(f.caps, a + (k + 1.0) * h);\n        k += 1.0;\n    }\n    return s * h;\n}",
            })
            .with(RuntimeFragment {
                name: "arr_chk",
                includes: &["#include <stdio.h>", "#include <unistd.h>"],
                deps: &["arr_type", "arr_new", "sites"],
                code: "/* An operation on two arrays checks their shapes first, as the interpreter\n   does, and a mismatch stops with its words and location: `u * v` of\n   lengths 3 and 2 used to read past v. `how`: 0 the same shape, 1 the same\n   shape with the operands swapped for the call (a real array times a complex\n   one), 2 `@`, 3 and 4 the same shape in `max`'s and `min`'s words. */\nstatic void viv_arr_chk_fail(int kind, const char* msg, int site) {\n    viv_fflush(stdout);\n#ifdef VIV_BOUNDS\n    viv_errf(\"error: %s\\n%s\", msg, viv_sites[site]);\n#else\n    (void)site;\n    viv_errf(\"error: %s\\n\", msg);\n#endif\n    viv_stop(kind);\n}\n/* a shape as Rust's {:?} writes a Vec<usize>: [2, 3] */\nstatic void viv_dims_text(char* b, viv_arr a) {\n    int n = sprintf(b, \"[\");\n    for (int i = 0; i < a.rank; i++) n += sprintf(b + n, i ? \", %d\" : \"%d\", a.dims[i]);\n    sprintf(b + n, \"]\");\n}\n/* a's entries repeated out to a broadcast shape, a row at a time (interp/arrays.rs, `expand`) */\nstatic viv_arr viv_bc_expand(viv_arr a, int rank, const int* to) {\n    long long st[8], idx[8];\n    int lead = rank - a.rank;\n    for (int d = a.rank - 1; d >= 0; d--) st[d] = d == a.rank - 1 ? 1 : st[d + 1] * a.dims[d + 1];\n    for (int d = 0; d < rank; d++) idx[d] = 0;\n    viv_arr r = viv_arr_new_nz(rank, to);\n    if (r.size == 0) return r;\n    long long last = to[rank - 1], w = 0;\n    int whole = a.dims[a.rank - 1] == last;\n    for (long long q = 0; q < r.size / last; q++) {\n        long long off = 0;\n        for (int d = lead; d < rank - 1; d++) if (a.dims[d - lead] != 1) off += idx[d] * st[d - lead];\n        if (whole) for (long long i = 0; i < last; i++) r.data[w++] = a.data[off + i];\n        else for (long long i = 0; i < last; i++) r.data[w++] = a.data[off];\n        for (int d = rank - 2; d >= 0; d--) { if (++idx[d] < to[d]) break; idx[d] = 0; }\n    }\n    return r;\n}\nstatic viv_arr viv_arr_chk(viv_arr (*f)(viv_arr, viv_arr), viv_arr a, viv_arr b, int how, int site) {\n    char m[320];\n    if (how == 6) {\n        /* complex @: matrices and vectors only */\n        int ra = a.rank, rb = b.rank;\n        if (ra == 1 && rb == 1)\n            viv_arr_chk_fail(VIV_E_VALUE, \"`@` of two vectors is refused: dot(u, v) is their dot product, outer(u, v) their outer product\", site);\n        if (!((ra == 2 || ra == 1) && (rb == 2 || rb == 1)))\n            viv_arr_chk_fail(VIV_E_VALUE, \"`@` needs arrays (2-D, or a 1-D vector operand)\", site);\n        int rows = ra == 2 ? a.dims[0] : 1, k = ra == 2 ? a.dims[1] : a.dims[0];\n        int k2 = b.dims[0], cols = rb == 2 ? b.dims[1] : 1;\n        if (k != k2) {\n            snprintf(m, sizeof m, \"`@` shape mismatch: %dx%d @ %dx%d\", rows, k, k2, cols);\n            viv_arr_chk_fail(VIV_E_VALUE, m, site);\n        }\n        return f(a, b);\n    }\n    if (how == 2) {\n        /* real @, batched over the axes before the last two (interp/matmul.rs, `plan`) */\n        int ra = a.rank, rb = b.rank;\n        if (ra == 1 && rb == 1)\n            viv_arr_chk_fail(VIV_E_VALUE, \"`@` of two vectors is refused: dot(u, v) is their dot product, outer(u, v) their outer product\", site);\n        if (ra == 0 || rb == 0)\n            viv_arr_chk_fail(VIV_E_VALUE, \"`@` needs arrays (2-D, or a 1-D vector operand)\", site);\n        int rows = ra == 1 ? 1 : a.dims[ra - 2], k = a.dims[ra - 1];\n        int k2 = rb == 1 ? b.dims[0] : b.dims[rb - 2], cols = rb == 1 ? 1 : b.dims[rb - 1];\n        if (k != k2) {\n            snprintf(m, sizeof m, \"`@` shape mismatch: %dx%d @ %dx%d\", rows, k, k2, cols);\n            viv_arr_chk_fail(VIV_E_VALUE, m, site);\n        }\n        int na = ra > 2 ? ra - 2 : 0, nb = rb > 2 ? rb - 2 : 0, nr = na > nb ? na : nb, ok = 1;\n        double count = 1.0;\n        for (int d = 0; d < nr; d++) {\n            int x = d + na >= nr ? a.dims[d + na - nr] : 1, y = d + nb >= nr ? b.dims[d + nb - nr] : 1;\n            if (x != y && x != 1 && y != 1) ok = 0;\n            count *= (double)(x == 1 ? y : x);\n        }\n        if (!ok) {\n            char da[128], db[128];\n            int n = sprintf(da, \"[\");\n            for (int i = 0; i < na; i++) n += sprintf(da + n, i ? \", %d\" : \"%d\", a.dims[i]);\n            sprintf(da + n, \"]\");\n            n = sprintf(db, \"[\");\n            for (int i = 0; i < nb; i++) n += sprintf(db + n, i ? \", %d\" : \"%d\", b.dims[i]);\n            sprintf(db + n, \"]\");\n            snprintf(m, sizeof m, \"`@`: the stacks of matrices have shapes %s and %s, which do not broadcast\", da, db);\n            viv_arr_chk_fail(VIV_E_VALUE, m, site);\n        }\n        int rr = nr + (ra > 1) + (rb > 1);\n        if (rr > 4) {\n            snprintf(m, sizeof m, \"`@`: the result would have rank %d; arrays have rank 4 at most\", rr);\n            viv_arr_chk_fail(VIV_E_VALUE, m, site);\n        }\n        if (count * (double)rows * (double)cols > 2147483647.0) {\n            if (nr == 0) snprintf(m, sizeof m, \"`@`: result of %dx%d exceeds the 2147483647-element limit\", rows, cols);\n            else snprintf(m, sizeof m, \"`@`: result of %.0f matrices of %dx%d exceeds the 2147483647-element limit\", count, rows, cols);\n            viv_arr_chk_fail(VIV_E_OVERFLOW, m, site);\n        }\n        return f(a, b);\n    }\n    int same = a.rank == b.rank;\n    for (int i = 0; same && i < a.rank; i++) same = a.dims[i] == b.dims[i];\n    if (!same && (how == 0 || how == 3 || how == 4)) {\n        /* real arrays broadcast as NumPy's do (interp/arrays.rs, `broadcast`) */\n        int rank = a.rank > b.rank ? a.rank : b.rank, to[8], ok = 1;\n        double total = 1.0;\n        for (int d = 0; d < rank; d++) {\n            int x = d + a.rank >= rank ? a.dims[d + a.rank - rank] : 1, y = d + b.rank >= rank ? b.dims[d + b.rank - rank] : 1;\n            if (x == y || y == 1) to[d] = x; else if (x == 1) to[d] = y; else ok = 0;\n            total *= (double)to[d];\n        }\n        if (ok) {\n            if (total > 2147483647.0) {\n                char da[128], db[128];\n                viv_dims_text(da, a);\n                viv_dims_text(db, b);\n                snprintf(m, sizeof m, \"broadcasting %s against %s gives %.0f elements, over the 2147483647-element limit\", da, db, total);\n                viv_arr_chk_fail(VIV_E_OVERFLOW, m, site);\n            }\n            return f(viv_bc_expand(a, rank, to), viv_bc_expand(b, rank, to));\n        }\n    }\n    if (!same) {\n        char da[128], db[128];\n        viv_dims_text(da, a);\n        viv_dims_text(db, b);\n        snprintf(m, sizeof m, \"%sarray shape mismatch: %s vs %s\", how == 3 ? \"`max`: \" : how == 4 ? \"`min`: \" : \"\", da, db);\n        viv_arr_chk_fail(VIV_E_VALUE, m, site);\n    }\n    return how == 1 ? f(b, a) : f(a, b);\n}",
            })
            .with(RuntimeFragment {
                name: "show_num",
                includes: &["#include <stdio.h>"],
                deps: &["fmt_g"],
                code: "static void viv_show_num(const char* label, double v) {\n    if (label[0]) viv_printf(\"%s: \", label);\n    viv_printf(\"%s\\n\", viv_g(v));\n}",
            })
            .with(RuntimeFragment {
                name: "show_bool",
                includes: &["#include <stdio.h>"],
                deps: &[],
                code: "static void viv_show_bool(const char* label, double v) {\n    if (label[0]) viv_printf(\"%s: \", label);\n    viv_printf(v != 0.0 ? \"true\\n\" : \"false\\n\");\n}",
            })
            .with(RuntimeFragment {
                // Step events, the C side of the interpreter's `__steps__` line:
                // the same first VIV_STEP_CAP events in the same text, and the
                // total. Under -DVIV_RELEASE both functions are empty, so the
                // calls cost nothing and nothing is printed.
                // The recorder is not thread-safe. `crate::parse` refuses any
                // step a `parallel for` can reach, so under OpenMP a step never
                // runs on a worker thread; if one ever did, the program stops
                // with an error instead of racing on the event array. The
                // prototype is declared rather than taken from <omp.h> so the
                // file builds wherever -fopenmp does, header path or not.
                name: "steps",
                includes: &["#include <stdio.h>", "#include <stdlib.h>"],
                deps: &[],
                code: "#define VIV_STEP_CAP 1000\n#if defined(_OPENMP) && !defined(VIV_RELEASE)\nint omp_in_parallel(void);\n#endif\n#ifndef VIV_RELEASE\nstatic struct { char kind; const char* def; const char* id; int depth; } viv_step_ev[VIV_STEP_CAP];\nstatic long viv_step_total = 0;\n#endif\nstatic void viv_step(int enter, const char* def, const char* id, int depth) {\n#ifndef VIV_RELEASE\n#ifdef _OPENMP\n    if (omp_in_parallel()) {\n        viv_errf(\"error: step `%s/%s` ran inside a parallel for\\n\", def, id);\n        viv_stop_exit(VIV_E_ERROR);\n    }\n#endif\n    if (viv_step_total < VIV_STEP_CAP) {\n        viv_step_ev[viv_step_total].kind = enter ? '+' : '-';\n        viv_step_ev[viv_step_total].def = def;\n        viv_step_ev[viv_step_total].id = id;\n        viv_step_ev[viv_step_total].depth = depth;\n    }\n    viv_step_total++;\n#else\n    (void)enter; (void)def; (void)id; (void)depth;\n#endif\n}\nstatic void viv_steps_flush(void) {\n#ifndef VIV_RELEASE\n    if (viv_step_total == 0) return;\n    long n = viv_step_total < VIV_STEP_CAP ? viv_step_total : VIV_STEP_CAP;\n    viv_printf(\"__steps__ {\\\"total\\\":%ld,\\\"events\\\":[\", viv_step_total);\n    for (long i = 0; i < n; i++)\n        viv_printf(\"%s\\\"%c%s/%s@%d\\\"\", i ? \",\" : \"\", viv_step_ev[i].kind, viv_step_ev[i].def, viv_step_ev[i].id, viv_step_ev[i].depth);\n    viv_printf(\"]}\\n\");\n#endif\n}",
            })
            .with(RuntimeFragment {
                name: "story",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <math.h>"],
                deps: &["arr_type"],
                code: "/* The story: the events of a debug run as\n   JSON text, printed after the run as one `__story__` line -- the bytes the\n   interpreter writes, event for event. Only the first VIV_STORY_CAP events\n   are kept; the total counts them all. Under -DVIV_RELEASE, VIV_STORY is 0:\n   every call site is guarded by it and the flush prints nothing. */\n#ifdef VIV_RELEASE\n#define VIV_STORY 0\n#else\n#define VIV_STORY 1\n#endif\n#define VIV_STORY_CAP 100000L\n#define VIV_STORY_ARR_CAP 4096\nstatic char* viv_story_buf = 0;\nstatic size_t viv_story_n = 0, viv_story_cap = 0;\nstatic long viv_story_total = 0;\nstatic int viv_story_fails = 0, viv_story_keep = 0, viv_story_first = 0;\nstatic void viv_story_put(const char* s) {\n    size_t k = strlen(s);\n    if (viv_story_n + k + 1 > viv_story_cap) {\n        size_t c = viv_story_cap ? viv_story_cap : 65536;\n        while (viv_story_n + k + 1 > c) c *= 2;\n        viv_story_buf = (char*)realloc(viv_story_buf, c);\n        viv_story_cap = c;\n    }\n    memcpy(viv_story_buf + viv_story_n, s, k + 1);\n    viv_story_n += k;\n}\n/* the interpreter's story_num: %.17e, and JSON strings for what JSON lacks */\nstatic void viv_story_num(double x) {\n    char b[48];\n    if (isnan(x)) viv_story_put(\"\\\"nan\\\"\");\n    else if (isinf(x)) viv_story_put(x < 0 ? \"\\\"-inf\\\"\" : \"\\\"inf\\\"\");\n    else { snprintf(b, sizeof b, \"%.17e\", x); viv_story_put(b); }\n}\nstatic int viv_story_begin(void) {\n    viv_story_total++;\n    if (viv_story_total > VIV_STORY_CAP) return 0;\n    if (viv_story_total > 1) viv_story_put(\",\");\n    return 1;\n}\nstatic void viv_story_mark(const char* kind, const char* def) {\n    if (!viv_story_begin()) return;\n    viv_story_put(\"[\\\"\"); viv_story_put(kind); viv_story_put(\"\\\",\\\"\"); viv_story_put(def); viv_story_put(\"\\\"]\");\n}\nstatic void viv_story_step(int enter, const char* def, const char* id, int depth) {\n    char b[32];\n    if (!viv_story_begin()) return;\n    viv_story_put(enter ? \"[\\\"enter\\\",\\\"\" : \"[\\\"exit\\\",\\\"\"); viv_story_put(def); viv_story_put(\"\\\",\\\"\"); viv_story_put(id);\n    snprintf(b, sizeof b, \"\\\",%d]\", depth); viv_story_put(b);\n}\nstatic void viv_story_iter(const char* def, double n) {\n    char b[48];\n    if (!viv_story_begin()) return;\n    viv_story_put(\"[\\\"iter\\\",\\\"\"); viv_story_put(def);\n    snprintf(b, sizeof b, \"\\\",%lld]\", (long long)n); viv_story_put(b);\n}\nstatic void viv_story_watch_begin(const char* def, const char* at) {\n    viv_story_keep = viv_story_begin();\n    viv_story_first = 1;\n    if (!viv_story_keep) return;\n    viv_story_put(\"[\\\"watch\\\",\\\"\"); viv_story_put(def); viv_story_put(\"\\\",\\\"\"); viv_story_put(at); viv_story_put(\"\\\",{\");\n}\nstatic void viv_story_wname(const char* name) {\n    if (!viv_story_first) viv_story_put(\",\");\n    viv_story_first = 0;\n    viv_story_put(\"\\\"\"); viv_story_put(name); viv_story_put(\"\\\":\");\n}\nstatic void viv_story_w_num(const char* name, double x) { if (!viv_story_keep) return; viv_story_wname(name); viv_story_num(x); }\nstatic void viv_story_w_bool(const char* name, double x) { if (!viv_story_keep) return; viv_story_wname(name); viv_story_put(x != 0.0 ? \"true\" : \"false\"); }\n/* an array whole up to VIV_STORY_ARR_CAP elements, else its shape, min, max,\n   norm and (square) trace -- summed in index order, as the interpreter does */\nstatic void viv_story_w_arr(const char* name, viv_arr a) {\n    char b[32];\n    if (!viv_story_keep) return;\n    viv_story_wname(name);\n    viv_story_put(\"{\\\"shape\\\":[\");\n    for (int i = 0; i < a.rank; i++) { snprintf(b, sizeof b, i ? \",%d\" : \"%d\", a.dims[i]); viv_story_put(b); }\n    if (a.size <= VIV_STORY_ARR_CAP) {\n        viv_story_put(\"],\\\"values\\\":[\");\n        for (int i = 0; i < a.size; i++) { if (i) viv_story_put(\",\"); viv_story_num(a.data[i]); }\n        viv_story_put(\"]}\");\n        return;\n    }\n    double lo = a.data[0], hi = a.data[0], ss = 0.0;\n    for (int i = 0; i < a.size; i++) { double x = a.data[i]; if (x < lo) lo = x; if (x > hi) hi = x; ss += x * x; }\n    viv_story_put(\"],\\\"min\\\":\"); viv_story_num(lo); viv_story_put(\",\\\"max\\\":\"); viv_story_num(hi); viv_story_put(\",\\\"norm\\\":\"); viv_story_num(sqrt(ss));\n    if (a.rank == 2 && a.dims[0] == a.dims[1]) {\n        int n = a.dims[0];\n        double tr = 0.0;\n        for (int i = 0; i < n; i++) tr += a.data[i * n + i];\n        viv_story_put(\",\\\"trace\\\":\"); viv_story_num(tr);\n    }\n    viv_story_put(\"}\");\n}\nstatic void viv_story_watch_end(void) { if (viv_story_keep) viv_story_put(\"}]\"); }\n/* the largest |x|; a NaN wins, so it is never hidden */\nstatic double viv_story_maxabs(viv_arr a) {\n    double w = 0.0;\n    for (int i = 0; i < a.size; i++) { double x = fabs(a.data[i]); if (x > w || isnan(x)) w = x; }\n    return w;\n}\nstatic void viv_story_hold(const char* def, const char* at, int ok, const char* sentence, int has_worst, double worst) {\n    if (!ok) viv_story_fails++;\n    if (!viv_story_begin()) return;\n    viv_story_put(\"[\\\"hold\\\",\\\"\"); viv_story_put(def); viv_story_put(\"\\\",\\\"\"); viv_story_put(at); viv_story_put(ok ? \"\\\",true,\" : \"\\\",false,\");\n    viv_story_put(sentence); viv_story_put(\",\");\n    if (has_worst) viv_story_num(worst); else viv_story_put(\"null\");\n    viv_story_put(\"]\");\n}\n/* the end of main: the story, and a failed hold makes the run fail */\nstatic void viv_story_flush(void) {\n    if (!VIV_STORY) return;\n#ifdef VIV_FAST\n    viv_printf(\"__story__ {\\\"mode\\\":\\\"fast\\\",\\\"total\\\":%ld,\\\"events\\\":[%s]}\\n\", viv_story_total, viv_story_buf ? viv_story_buf : \"\");\n#else\n    viv_printf(\"__story__ {\\\"total\\\":%ld,\\\"events\\\":[%s]}\\n\", viv_story_total, viv_story_buf ? viv_story_buf : \"\");\n#endif\n    if (viv_story_fails) {\n        viv_fflush(stdout);\n        viv_errf(\"%d hold%s FAILED\\n\", viv_story_fails, viv_story_fails == 1 ? \"\" : \"s\");\n        viv_stop_exit_failed();\n    }\n}",
            })
            .with(RuntimeFragment {
                name: "placeholder",
                includes: &["#include <stdio.h>", "#include <stdlib.h>"],
                deps: &[],
                code: "/* A call to a `placeholder def`: an error that names it, as the interpreter's. */\n#if defined(__GNUC__)\n__attribute__((noreturn))\n#endif\nstatic void viv_placeholder(const char* def) {\n    viv_fflush(stdout);\n    viv_errf(\"error: `%s` is a placeholder: it has a signature and no body yet\\n\", def);\n    viv_stop_exit(VIV_E_ERROR);\n}",
            })
            .with(RuntimeFragment {
                name: "stop",
                includes: &["#include <stdio.h>", "#include <unistd.h>"],
                deps: &["sites"],
                code: "/* A check the parser inserted (ast Lens::Stop) failed: stop with the\n   interpreter's words and, in a debug build, its location. */\n#if defined(__GNUC__)\n__attribute__((noreturn, unused))\n#endif\nstatic void viv_check_stop(int kind, const char* msg) {\n    viv_fflush(stdout);\n    viv_errf(\"error: %s\\n\", msg);\n    viv_stop(kind);\n}\n#if defined(__GNUC__)\n__attribute__((noreturn, unused))\n#endif\nstatic void viv_check_stop_at(int kind, const char* msg, int site) {\n#ifdef VIV_BOUNDS\n    viv_fflush(stdout);\n    viv_errf(\"error: %s\\n%s\", msg, viv_sites[site]);\n    viv_stop(kind);\n#else\n    (void)site;\n    viv_check_stop(kind, msg);\n#endif\n}",
            })
            .with(RuntimeFragment {
                name: "narrate",
                includes: &["#include <stdio.h>"],
                deps: &[],
                code: "static void viv_narrate(const char* msg) {\n    viv_printf(\"narrate: %s\\n\", msg);\n}",
            })
            .with(RuntimeFragment {
                name: "anim",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type", "prt", "fmt_g"],
                code: "/* one animation or figure record: `verb: text ; text ; [array] ; [array] ; [array] ; n n n`.\n   `full`: the arrays are a chart's data, printed at full precision (viv_prt), as plot's are */\nstatic double viv_anim(const char* head, int full, int na, viv_arr a0, viv_arr a1, viv_arr a2, int nn, double n0, double n1, double n2, double n3, double n4, double n5) {\n    viv_arr as[3]; as[0] = a0; as[1] = a1; as[2] = a2;\n    double ns[6]; ns[0] = n0; ns[1] = n1; ns[2] = n2; ns[3] = n3; ns[4] = n4; ns[5] = n5;\n    int bare = 1;\n    for (const char* c = head; *c; c++) if (*c == ':' && c[1]) bare = 0;\n    viv_printf(\"%s\", head);\n    for (int k = 0; k < na; k++) {\n        viv_printf(bare ? \" [\" : \" ; [\"); bare = 0;\n        for (int i = 0; i < as[k].size; i++) { if (i) viv_printf(\", \"); if (full) viv_prt(as[k].data[i]); else viv_fputs(viv_g(as[k].data[i]), stdout); }\n        viv_printf(\"]\");\n    }\n    if (nn) {\n        viv_printf(bare ? \"\" : \" ;\");\n        for (int k = 0; k < nn; k++) viv_printf(\" %s\", viv_g(ns[k]));\n    }\n    viv_printf(\"\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "prt",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <math.h>"],
                deps: &["fmt_g"],
                code: "/* one number of a chart's data: the shortest of %.15g, %.16g, %.17g that reads\n   back as the same double (the interpreter's format_rt, line for line), each\n   written by viv_fmt_g, not the C library's %g */\nstatic void viv_prt(double v) {\n    char b[40];\n    v += 0.0;\n    if (v != v) { viv_printf(\"nan\"); return; }\n    if (isinf(v)) { viv_printf(v < 0 ? \"-inf\" : \"inf\"); return; }\n    for (int p = 15; p <= 16; p++) {\n        viv_fmt_g(b, sizeof b, v, p);\n        if (strtod(b, 0) == v) { viv_fputs(b, stdout); return; }\n    }\n    viv_fputs(viv_gp(v, 17), stdout);\n}",
            })
            .with(RuntimeFragment {
                name: "plot",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type", "prt"],
                code: "static double viv_plot(const char* label, viv_arr x, viv_arr y) {\n    viv_printf(\"plot: %s ; [\", label);\n    for (int i = 0; i < x.size; i++) { if (i) viv_printf(\", \"); viv_prt(x.data[i]); }\n    viv_printf(\"] ; [\");\n    for (int i = 0; i < y.size; i++) { if (i) viv_printf(\", \"); viv_prt(y.data[i]); }\n    viv_printf(\"]\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "scatter",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type"],
                code: "static double viv_scatter(const char* label, const char* color, viv_arr x, viv_arr y) {\n    viv_printf(\"scatter: %s ; %s ; [\", label, color);\n    for (int i = 0; i < x.size; i++) { if (i) viv_printf(\", \"); viv_printf(\"%.17e\", x.data[i]); }\n    viv_printf(\"] ; [\");\n    for (int i = 0; i < y.size; i++) { if (i) viv_printf(\", \"); viv_printf(\"%.17e\", y.data[i]); }\n    viv_printf(\"]\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "plot2d",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type", "prt"],
                code: "static double viv_plot2d(const char* label, viv_arr z) {\n    int nx = z.rank == 2 ? z.dims[0] : 1, ny = z.rank == 2 ? z.dims[1] : z.size;\n    viv_printf(\"plot2d: %s ; %d %d ; [\", label, nx, ny);\n    for (int i = 0; i < z.size; i++) { if (i) viv_printf(\", \"); viv_prt(z.data[i]); }\n    viv_printf(\"]\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "quiver",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type", "fmt_g"],
                code: "static void viv_quiver_arr(viv_arr a) {\n    viv_printf(\" ; [\");\n    for (int i = 0; i < a.size; i++) { if (i) viv_printf(\", \"); viv_fputs(viv_g(a.data[i]), stdout); }\n    viv_printf(\"]\");\n}\nstatic double viv_quiver(const char* label, viv_arr x, viv_arr y, viv_arr u, viv_arr v) {\n    viv_printf(\"quiver: %s\", label);\n    viv_quiver_arr(x); viv_quiver_arr(y); viv_quiver_arr(u); viv_quiver_arr(v);\n    viv_printf(\"\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "ckpt",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <errno.h>", "#include <complex.h>"],
                deps: &["arr_type"],
                code: "static void viv_ck_die(const char* m) { viv_errf(\"ckpt: %s\\n\", m); viv_stop_exit(VIV_E_ERROR); }\nstatic void viv_ckw_head(FILE* f, const char* nm, unsigned char tag) {\n    unsigned int l = (unsigned int)strlen(nm);\n    if (fwrite(&l, 4, 1, f) != 1 || fwrite(nm, 1, l, f) != l || fwrite(&tag, 1, 1, f) != 1) viv_ck_die(\"write failed\");\n}\nstatic void viv_ckw_f64(FILE* f, const char* nm, double* v) { viv_ckw_head(f, nm, 0); if (fwrite(v, 8, 1, f) != 1) viv_ck_die(\"write failed\"); }\nstatic void viv_ckw_cplx(FILE* f, const char* nm, double complex* v) { viv_ckw_head(f, nm, 2); if (fwrite(v, 16, 1, f) != 1) viv_ck_die(\"write failed\"); }\nstatic void viv_ckw_arr(FILE* f, const char* nm, viv_arr* a) {\n    viv_ckw_head(f, nm, 1);\n    if (fwrite(&a->rank, 4, 1, f) != 1 || fwrite(a->dims, 4, 8, f) != 8 || fwrite(&a->size, 4, 1, f) != 1) viv_ck_die(\"write failed\");\n    if (a->size > 0 && fwrite(a->data, 8, (size_t)a->size, f) != (size_t)a->size) viv_ck_die(\"write failed\");\n}\nstatic int viv_ckr_head(FILE* f, char* nm, int cap, unsigned char* tag) {\n    unsigned int l;\n    if (fread(&l, 4, 1, f) != 1) return 0;\n    if (l >= (unsigned int)cap) viv_ck_die(\"bad name\");\n    if (fread(nm, 1, l, f) != l) viv_ck_die(\"short read\");\n    nm[l] = 0;\n    if (fread(tag, 1, 1, f) != 1) viv_ck_die(\"short read\");\n    return 1;\n}\nstatic void viv_ckr_f64(FILE* f, unsigned char tag, double* v) { if (tag != 0) viv_ck_die(\"type changed\"); if (fread(v, 8, 1, f) != 1) viv_ck_die(\"short read\"); }\nstatic void viv_ckr_cplx(FILE* f, unsigned char tag, double complex* v) { if (tag != 2) viv_ck_die(\"type changed\"); if (fread(v, 16, 1, f) != 1) viv_ck_die(\"short read\"); }\nstatic void viv_ckr_arr(FILE* f, unsigned char tag, viv_arr* a) {\n    if (tag != 1) viv_ck_die(\"type changed\");\n    if (fread(&a->rank, 4, 1, f) != 1 || fread(a->dims, 4, 8, f) != 8 || fread(&a->size, 4, 1, f) != 1) viv_ck_die(\"short read\");\n    if (a->size > 0) {\n        a->data = (double*)malloc((size_t)a->size * 8);\n        if (!a->data || fread(a->data, 8, (size_t)a->size, f) != (size_t)a->size) viv_ck_die(\"short read\");\n    } else { a->data = 0; }\n}\nstatic void viv_ckr_skip(FILE* f, unsigned char tag) {\n    if (tag == 0) { if (fseek(f, 8, SEEK_CUR)) viv_ck_die(\"seek\"); }\n    else if (tag == 2) { if (fseek(f, 16, SEEK_CUR)) viv_ck_die(\"seek\"); }\n    else if (tag == 1) {\n        int r, d[8], s;\n        if (fread(&r, 4, 1, f) != 1 || fread(d, 4, 8, f) != 8 || fread(&s, 4, 1, f) != 1) viv_ck_die(\"short read\");\n        if (fseek(f, (long)s * 8, SEEK_CUR)) viv_ck_die(\"seek\");\n    } else viv_ck_die(\"bad tag\");\n}",
            })
            .with(RuntimeFragment {
                name: "figure",
                includes: &["#include <stdio.h>"],
                deps: &["fmt_g"],
                code: "static double viv_figure(const char* t, const char* xl, const char* yl) {\n    viv_printf(\"figure: %s ; %s ; %s\\n\", t, xl, yl);\n    return 0.0;\n}\nstatic double viv_figure_win(const char* t, const char* xl, const char* yl, double a, double b, double c, double d, const char* st) {\n    if (st && st[0]) viv_printf(\"figure: %s ; %s ; %s ; %s %s %s %s ; %s\\n\", t, xl, yl, viv_g(a), viv_g(b), viv_g(c), viv_g(d), st);\n    else viv_printf(\"figure: %s ; %s ; %s ; %s %s %s %s\\n\", t, xl, yl, viv_g(a), viv_g(b), viv_g(c), viv_g(d));\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "plot3d",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type", "fmt_g"],
                code: "static double viv_plot3d(const char* label, viv_arr x, viv_arr y, viv_arr z) {\n    viv_printf(\"plot3d: %s ; [\", label);\n    for (int i = 0; i < x.size; i++) { if (i) viv_printf(\", \"); viv_fputs(viv_g(x.data[i]), stdout); }\n    viv_printf(\"] ; [\");\n    for (int i = 0; i < y.size; i++) { if (i) viv_printf(\", \"); viv_fputs(viv_g(y.data[i]), stdout); }\n    viv_printf(\"] ; [\");\n    for (int i = 0; i < z.size; i++) { if (i) viv_printf(\", \"); viv_fputs(viv_g(z.data[i]), stdout); }\n    viv_printf(\"]\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "scene3d",
                includes: &["#include <stdio.h>"],
                deps: &["arr_type", "fmt_g"],
                code: "static void viv_sc_arr(viv_arr a) {\n    viv_printf(\" ; [\");\n    for (int i = 0; i < a.size; i++) { if (i) viv_printf(\", \"); viv_fputs(viv_g(a.data[i]), stdout); }\n    viv_printf(\"]\");\n}\nstatic double viv_scene(const char* t) {\n    viv_printf(\"scene: %s\\n\", t);\n    return 0.0;\n}\nstatic double viv_sc3(const char* tag, const char* g, viv_arr x, viv_arr y, viv_arr z) {\n    viv_printf(\"%s: %s\", tag, g);\n    viv_sc_arr(x); viv_sc_arr(y); viv_sc_arr(z);\n    viv_printf(\"\\n\");\n    return 0.0;\n}\nstatic double viv_sc4(const char* tag, const char* g, viv_arr x, viv_arr y, viv_arr z, viv_arr v) {\n    viv_printf(\"%s: %s\", tag, g);\n    viv_sc_arr(x); viv_sc_arr(y); viv_sc_arr(z); viv_sc_arr(v);\n    viv_printf(\"\\n\");\n    return 0.0;\n}",
            })
            .with(RuntimeFragment { name: "arr_adds", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_adds(viv_arr a, double s) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] + s;\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_subs", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_subs(viv_arr a, double s) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = a.data[i] - s;\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_subsr", includes: &[], deps: &["arr_new"], code: "static viv_arr viv_arr_subsr(viv_arr a, double s) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = s - a.data[i];\n    return r;\n}" })
            .with(RuntimeFragment { name: "arr_map", includes: &["#include <math.h>"], deps: &["arr_new"], code: "static viv_arr viv_arr_map(double (*f)(double), viv_arr a) {\n    viv_arr r = viv_arr_new_nz(a.rank, a.dims);\n    for (int i = 0; i < r.size; i++) r.data[i] = f(a.data[i]);\n    return r;\n}" })
            .with(RuntimeFragment { name: "cplx", includes: &["#include <complex.h>", "#include <stdio.h>", "#include <math.h>"], deps: &["fmt_g"], code: "static double complex viv_cplx(double re, double im) { return viv_cmake(re, im); }\nstatic void viv_show_cplx(const char* label, double complex z) {\n    if (label[0]) viv_printf(\"%s: \", label);\n    viv_printf(\"%s%s%si\\n\", viv_g(creal(z)), signbit(cimag(z)) && cimag(z) != 0.0 ? \"-\" : \"+\", viv_g(fabs(cimag(z))));\n}" })
            .with(RuntimeFragment { name: "fill", includes: &[], deps: &["arr_type"], code: "static double viv_fill(viv_arr a, double v) {\n    for (int i = 0; i < a.size; i++) a.data[i] = v;\n    return 0.0;\n}" })
            .with(RuntimeFragment {
                name: "show_arr",
                // string.h for strlen: the matrix indent is the label's width.
                includes: &["#include <stdio.h>", "#include <string.h>"],
                deps: &["arr_type", "fmt_g"],
                code: "static void viv_show_arr(const char* label, viv_arr a) {\n    if (label[0]) viv_printf(\"%s: \", label);\n    if (a.rank == 2) {\n        int rows = a.dims[0], cols = a.dims[1];\n        /* \"label: [\" puts the inner bracket at strlen(label) + 3; with no\n           label the outer bracket is at column 0, so the inner is at 1.\n           Continuation rows pad to that column so the matrix lines up. */\n        int pad = label[0] ? (int)strlen(label) + 3 : 1;\n        viv_printf(\"[\");\n        for (int i = 0; i < rows; i++) {\n            if (i) viv_printf(\"%*s\", pad, \"\");\n            viv_printf(\"[\");\n            for (int j = 0; j < cols; j++) { if (j) viv_printf(\", \"); viv_fputs(viv_g(a.data[i*cols+j]), stdout); }\n            viv_printf(\"]\");\n            if (i < rows - 1) viv_printf(\",\\n\");\n        }\n        viv_printf(\"]\\n\");\n    } else {\n        viv_printf(\"[\");\n        for (int i = 0; i < a.size; i++) { if (i) viv_printf(\", \"); viv_fputs(viv_g(a.data[i]), stdout); }\n        viv_printf(\"]\\n\");\n    }\n}",
            })
            .with(RuntimeFragment {
                name: "assert_near",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <math.h>", "#include <unistd.h>"],
                deps: &["fmt_g"],
                code: "static int viv_fails = 0;\n/* The failed-assert summary is a diagnostic, not program output: it goes to\n   stderr after everything the program printed, as the interpreter writes it. */\n/* An exported program (-DVIV_EMBED) reports its tests in a table and does\n   not register the summary: atexit would end the caller's process. */\n#ifndef VIV_EMBED\nstatic void viv_fail_exit(void) { if (viv_fails) { viv_fflush(stdout); viv_errf(\"%d assertion%s FAILED\\n\", viv_fails, viv_fails == 1 ? \"\" : \"s\"); viv_stop_failed(); } }\nstatic void viv_fail(void) { if (!viv_fails) atexit(viv_fail_exit); viv_fails++; }\n#else\nstatic void viv_fail(void) { viv_fails++; }\n#endif\n/* The active `test` block. A test is the unit that passes or fails, so an\n   assert inside one records against it instead of printing its own line, and\n   the block prints one `PASS <name>` / `FAIL <name>` line when it ends. The\n   depth counter absorbs a nested test into the outer one. */\nstatic const char* viv_test = 0;\nstatic int viv_test_depth = 0;\nstatic int viv_test_n = 0;\nstatic int viv_test_fails = 0;\nstatic char viv_test_detail[256];\n#ifndef VIV_EMBED\n/* an error stopped the program inside a test: it FAILs (B344), as the\n   interpreter's test_stopped writes it */\nstatic void viv_test_stopped(void) {\n    viv_test_depth = 0;\n    viv_printf(\"FAIL %s -- an error stopped the program\\n\", viv_test ? viv_test : \"\");\n}\n#endif\nstatic void viv_test_begin(const char* name) {\n    if (viv_test_depth == 0) { viv_test = name; viv_test_n = 0; viv_test_fails = 0; viv_test_detail[0] = 0; }\n#ifndef VIV_EMBED\n    viv_fatal_hook = viv_test_stopped;\n#endif\n    viv_test_depth++;\n}\nstatic void viv_test_end(void) {\n    if (viv_test_depth > 0) viv_test_depth--;\n    if (viv_test_depth > 0) return;\n#ifndef VIV_EMBED\n    viv_fatal_hook = 0;\n#endif\n    if (viv_test_fails == 0 && viv_test_n == 0) viv_printf(\"PASS %s (no assertions)\\n\", viv_test ? viv_test : \"\");\n    else if (viv_test_fails == 0) viv_printf(\"PASS %s\\n\", viv_test ? viv_test : \"\");\n    else if (viv_test_fails == 1) viv_printf(\"FAIL %s -- %s\\n\", viv_test ? viv_test : \"\", viv_test_detail);\n    else viv_printf(\"FAIL %s -- %s (and %d more)\\n\", viv_test ? viv_test : \"\", viv_test_detail, viv_test_fails - 1);\n#ifdef VIV_EMBED\n    {\n        char d[320];\n        if (viv_test_fails > 1) snprintf(d, sizeof d, \"%s (and %d more)\", viv_test_detail, viv_test_fails - 1);\n        else snprintf(d, sizeof d, \"%s\", viv_test_fails ? viv_test_detail : \"\");\n        viv_embed_test(viv_test, viv_test_fails == 0, d);\n    }\n#endif\n    viv_test = 0;\n}\nstatic void viv_report(int pass, const char* what, const char* detail) {\n    if (!pass) viv_fail();\n    if (viv_test_depth > 0) {\n        viv_test_n++;\n        if (!pass) {\n            if (viv_test_fails == 0) snprintf(viv_test_detail, sizeof viv_test_detail, \"%s: %s\", what, detail);\n            viv_test_fails++;\n        }\n        return;\n    }\n    if (pass) viv_printf(\"%s PASS\\n\", what);\n    else viv_printf(\"%s FAIL: %s\\n\", what, detail);\n}\nstatic void viv_assert_near(double a, double b, double tol) {\n    char d[192]; int pass = fabs(a - b) <= tol;\n    if (!pass) snprintf(d, sizeof d, \"|%s - %s| = %s > %s\", viv_g(a), viv_g(b), viv_g(fabs(a - b)), viv_g(tol));\n    viv_report(pass, \"assert_near\", pass ? \"\" : d);\n}\nstatic void viv_assert_true(double c) {\n    viv_report(c != 0.0, \"assert_true\", \"condition is false\");\n}\nstatic void viv_assert_gt(double a, double b) {\n    char d[96]; int pass = a > b;\n    if (!pass) snprintf(d, sizeof d, \"%s <= %s\", viv_g(a), viv_g(b));\n    viv_report(pass, \"assert_gt\", pass ? \"\" : d);\n}\nstatic void viv_assert_lt(double a, double b) {\n    char d[96]; int pass = a < b;\n    if (!pass) snprintf(d, sizeof d, \"%s >= %s\", viv_g(a), viv_g(b));\n    viv_report(pass, \"assert_lt\", pass ? \"\" : d);\n}",
            })
            .with(RuntimeFragment {
                name: "assert_allclose",
                includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <math.h>"],
                deps: &["arr_type", "assert_near", "fmt_g"],
                code: "static void viv_assert_allclose(viv_arr a, viv_arr b, double tol) {\n    double worst = 0.0; int at = 0;\n    for (int i = 0; i < a.size; i++) { double d = fabs(a.data[i] - b.data[i]); if (d > worst) { worst = d; at = i; } }\n    char dd[128]; int pass = worst <= tol;\n    if (!pass) snprintf(dd, sizeof dd, \"max |a - b| = %s > %s at index %d\", viv_g(worst), viv_g(tol), at);\n    viv_report(pass, \"assert_allclose\", pass ? \"\" : dd);\n}",
            })
            .with(RuntimeFragment {
                name: "release_mode",
                includes: &[],
                deps: &[],
                code: "static double viv_release_mode(void) {\n#ifdef VIV_RELEASE\n    return 1.0;\n#else\n    return 0.0;\n#endif\n}",
            })
            .with(RuntimeFragment {
                name: "rng",
                includes: &[],
                deps: &[],
                code: "static double viv_rng_state = 12345.0;\nstatic double viv_srand(double s) { double m = 2147483646.0; double v = s < 0 ? -s : s; v = v - m * (double)(long long)(v / m); viv_rng_state = (double)(long long)v + 1.0; return 0.0; }\nstatic double viv_rand(void) {\n    double p = 16807.0 * viv_rng_state;\n    viv_rng_state = p - 2147483647.0 * (double)(long long)(p / 2147483647.0);\n    return viv_rng_state / 2147483647.0;\n}",
            })
            .with(RuntimeFragment {
                name: "sort",
                deps: &["arr_type"],
                // The interpreter's order (interp.rs, `sort_order`), and like
                // Rust's sort_by a stable sort: a bottom-up merge sort, not
                // qsort, whose order of equal keys is the C library's.
                includes: &["#include <stdlib.h>", "#include <string.h>", "#include <math.h>"],
                code: "/* a strictly before b: ascending, -0.0 before +0.0, NaN after every number */\nstatic int viv_sort_before(double a, double b) {\n    if (a != a) return 0;\n    if (b != b) return 1;\n    if (a < b) return 1;\n    if (a > b) return 0;\n    return signbit(a) && !signbit(b);\n}\nstatic double viv_sort(viv_arr a) {\n    size_t n = (size_t)a.size;\n    if (n < 2) return 0.0;\n    double* src = a.data;\n    double* dst = (double*)malloc(n * sizeof(double));\n    for (size_t w = 1; w < n; w *= 2) {\n        for (size_t lo = 0; lo < n; lo += 2 * w) {\n            size_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n;\n            size_t i = lo, j = mid, k = lo;\n            while (i < mid && j < hi) dst[k++] = viv_sort_before(src[j], src[i]) ? src[j++] : src[i++];\n            while (i < mid) dst[k++] = src[i++];\n            while (j < hi) dst[k++] = src[j++];\n        }\n        double* t = src; src = dst; dst = t;\n    }\n    if (src != a.data) { memcpy(a.data, src, n * sizeof(double)); dst = src; }\n    free(dst);\n    return 0.0;\n}",
            })
            .with(RuntimeFragment {
                name: "finite_chk",
                includes: &["#include <stdio.h>", "#include <unistd.h>"],
                deps: &["arr_type", "sites"],
                code: "/* eig_sym and spectrum_sym stop on a matrix with a NaN or an infinity, with\n   the interpreter's words and location (the order of the eigenvalues is\n   undefined; the interpreter's sort panicked on it). */\nstatic void viv_finite_chk(const char* who, viv_arr A, int site) {\n    for (int i = 0; i < A.size; i++) {\n        if (A.data[i] - A.data[i] == 0.0) continue;\n        viv_fflush(stdout);\n#ifdef VIV_BOUNDS\n        viv_errf(\"error: `%s`: the matrix has a NaN or an infinity\\n%s\", who, viv_sites[site]);\n#else\n        (void)site;\n        viv_errf(\"error: `%s`: the matrix has a NaN or an infinity\\n\", who);\n#endif\n        viv_stop(VIV_E_VALUE);\n    }\n}",
            })
            .with(RuntimeFragment {
                name: "eig_sym",
                includes: &["#include <stdlib.h>", "#include <math.h>"],
                deps: &["arr_type", "cos_sin", "finite_chk"],
                code: "static void viv_eig_sym(viv_arr A, viv_arr vals, viv_arr vecs) {\n    int n = A.dims[0];\n    double* M = (double*)malloc((size_t)(n*n) * sizeof(double));\n    double* V = (double*)malloc((size_t)(n*n) * sizeof(double));\n    for (int i = 0; i < n*n; i++) M[i] = A.data[i];\n    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) V[i*n+j] = (i == j) ? 1.0 : 0.0;\n    double scale = 0.0;\n    for (int i = 0; i < n*n; i++) scale += M[i]*M[i];\n    scale = sqrt(scale); if (scale < 1e-300) scale = 1e-300;\n    for (int sweep = 0; sweep < 100; sweep++) {\n        double off = 0.0;\n        for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) off += M[i*n+j]*M[i*n+j];\n        if (sqrt(off) < 1e-12 * scale) break;\n        for (int p = 0; p < n; p++) for (int q = p+1; q < n; q++) {\n            double apq = M[p*n+q];\n            if (fabs(apq) < 1e-15 * scale) continue;\n            double app = M[p*n+p], aqq = M[q*n+q];\n            double theta = 0.5 * atan2(2.0 * apq, app - aqq);\n            double c, s; viv_cos_sin(theta, &c, &s);\n            for (int k = 0; k < n; k++) if (k != p && k != q) {\n                double mkp = M[k*n+p], mkq = M[k*n+q];\n                double nkp = c*mkp + s*mkq, nkq = -s*mkp + c*mkq;\n                M[k*n+p] = nkp; M[p*n+k] = nkp; M[k*n+q] = nkq; M[q*n+k] = nkq;\n            }\n            M[p*n+p] = c*c*app + 2.0*s*c*apq + s*s*aqq;\n            M[q*n+q] = s*s*app - 2.0*s*c*apq + c*c*aqq;\n            M[p*n+q] = 0.0; M[q*n+p] = 0.0;\n            for (int k = 0; k < n; k++) {\n                double vkp = V[k*n+p], vkq = V[k*n+q];\n                V[k*n+p] = c*vkp + s*vkq; V[k*n+q] = -s*vkp + c*vkq;\n            }\n        }\n    }\n    int* idx = (int*)malloc((size_t)n * sizeof(int));\n    for (int i = 0; i < n; i++) idx[i] = i;\n    /* ascending order: a stable insertion sort, matching the interpreter's.\n       Equal eigenvalues keep their columns' order; the exchange sort this\n       replaces swapped them, so a degenerate spectrum came out with its\n       eigenvectors permuted, and a sum over them rounded differently. */\n    for (int i = 1; i < n; i++) {\n        int t = idx[i], j = i - 1;\n        while (j >= 0 && M[idx[j]*n+idx[j]] > M[t*n+t]) { idx[j + 1] = idx[j]; j--; }\n        idx[j + 1] = t;\n    }\n    for (int i = 0; i < n; i++) {\n        vals.data[i] = M[idx[i]*n+idx[i]];\n        for (int r = 0; r < n; r++) vecs.data[r*n+i] = V[r*n+idx[i]];\n    }\n    free(M); free(V); free(idx);\n}\nstatic void viv_eig_sym_at(viv_arr A, viv_arr vals, viv_arr vecs, int site) {\n    viv_finite_chk(\"eig_sym\", A, site);\n    viv_eig_sym(A, vals, vecs);\n}",
            })
            .with(RuntimeFragment {
                name: "tql2",
                includes: &["#include <math.h>", "#include <float.h>"],
                deps: &[],
                code: "/* Implicit QL with Wilkinson shifts on the symmetric tridiagonal with\n   diagonal d and subdiagonal e (e[k] couples k and k + 1; the last entry is\n   ignored): tql2 of EISPACK (Bowdler, Martin, Reinsch and Wilkinson 1968) in\n   the form the public-domain JAMA library gives it. The interpreter's tql2,\n   line for line. Eigenvalues land in d, unsorted. With z (row-major n x n)\n   the rotations are applied to its columns. Returns 0 when one eigenvalue\n   takes more than maxit QL steps. */\nstatic int viv_tql2(int n, double* d, double* e, double* z, int maxit) {\n#pragma STDC FP_CONTRACT OFF\n    if (n == 0) return 1;\n    e[n - 1] = 0.0;\n    double f = 0.0, tst1 = 0.0;\n    for (int l = 0; l < n; l++) {\n        double t1 = fabs(d[l]) + fabs(e[l]);\n        if (t1 > tst1) tst1 = t1;\n        int m = l;\n        while (m + 1 < n && fabs(e[m]) > DBL_EPSILON * tst1) m++;\n        if (m > l) {\n            int iter = 0;\n            for (;;) {\n                if (++iter > maxit) return 0;\n                /* Wilkinson shift from the 2 x 2 at l */\n                double g = d[l];\n                double p = (d[l + 1] - g) / (2.0 * e[l]);\n                double r = hypot(p, 1.0);\n                if (p < 0.0) r = -r;\n                d[l] = e[l] / (p + r);\n                d[l + 1] = e[l] * (p + r);\n                double dl1 = d[l + 1];\n                double h = g - d[l];\n                for (int i = l + 2; i < n; i++) d[i] -= h;\n                f += h;\n                /* the QL sweep from m up to l */\n                p = d[m];\n                double c = 1.0, c2 = 1.0, c3 = 1.0, s = 0.0, s2 = 0.0;\n                double el1 = e[l + 1];\n                for (int i = m - 1; i >= l; i--) {\n                    c3 = c2; c2 = c; s2 = s;\n                    g = c * e[i];\n                    h = c * p;\n                    r = hypot(p, e[i]);\n                    e[i + 1] = s * r;\n                    s = e[i] / r;\n                    c = p / r;\n                    p = c * d[i] - s * g;\n                    d[i + 1] = h + s * (c * g + s * d[i]);\n                    if (z) for (int k = 0; k < n; k++) {\n                        double zk = z[k * n + i + 1];\n                        z[k * n + i + 1] = s * z[k * n + i] + c * zk;\n                        z[k * n + i]     = c * z[k * n + i] - s * zk;\n                    }\n                }\n                p = -s * s2 * c3 * el1 * e[l] / dl1;\n                e[l] = s * p;\n                d[l] = c * p;\n                if (fabs(e[l]) <= DBL_EPSILON * tst1) break;\n            }\n        }\n        /* undo the accumulated shift */\n        d[l] += f;\n        e[l] = 0.0;\n    }\n    return 1;\n}",
            })
            .with(RuntimeFragment {
                name: "spectrum_sym",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"],
                deps: &["arr_type", "arr_new", "arr_free", "eig_sym", "fmt_g", "finite_chk", "tql2"],
                code: "static void viv_tridiag_eigvals(int n, const double* a_in, double* d) {\n    /* The reduction half of JAMA's tred2 (public domain, after EISPACK), with\n       no eigenvector accumulation, then tql2: the interpreter's\n       tridiag_eigvals, line for line. */\n#pragma STDC FP_CONTRACT OFF\n    if (n == 1) { d[0] = a_in[0]; return; }\n    double* v = (double*)malloc((size_t)(n*n) * sizeof(double));\n    double* e = (double*)calloc((size_t)n, sizeof(double));\n    for (int i = 0; i < n*n; i++) v[i] = a_in[i];\n    for (int j = 0; j < n; j++) d[j] = v[(n-1)*n + j];\n    for (int i = n - 1; i > 0; i--) {\n        double scale = 0.0, h = 0.0;\n        for (int k = 0; k < i; k++) scale += fabs(d[k]);\n        if (scale == 0.0) {\n            e[i] = d[i-1];\n            for (int j = 0; j < i; j++) { d[j] = v[(i-1)*n + j]; v[i*n + j] = 0.0; v[j*n + i] = 0.0; }\n        } else {\n            for (int k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }\n            double f = d[i-1];\n            double g = sqrt(h);\n            if (f > 0.0) g = -g;\n            e[i] = scale * g;\n            h -= f * g;\n            d[i-1] = f - g;\n            for (int j = 0; j < i; j++) e[j] = 0.0;\n            for (int j = 0; j < i; j++) {\n                f = d[j];\n                v[j*n + i] = f;\n                g = e[j] + v[j*n + j] * f;\n                for (int k = j + 1; k < i; k++) { g += v[k*n + j] * d[k]; e[k] += v[k*n + j] * f; }\n                e[j] = g;\n            }\n            f = 0.0;\n            for (int j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }\n            double hh = f / (h + h);\n            for (int j = 0; j < i; j++) e[j] -= hh * d[j];\n            for (int j = 0; j < i; j++) {\n                f = d[j]; g = e[j];\n                for (int k = j; k < i; k++) v[k*n + j] -= f * e[k] + g * d[k];\n                d[j] = v[(i-1)*n + j];\n                v[i*n + j] = 0.0;\n            }\n        }\n        d[i] = h;\n    }\n    for (int i = 0; i < n; i++) d[i] = v[i*n + i];\n    for (int i = 0; i < n - 1; i++) e[i] = e[i+1];\n    if (!viv_tql2(n, d, e, 0, 50)) { viv_errf(\"spectrum_sym: QL iteration failed to converge\\n\"); viv_stop_exit(VIV_E_ERROR); }\n    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++)\n        if (d[j] < d[i]) { double t = d[i]; d[i] = d[j]; d[j] = t; }\n    free(v); free(e);\n}\nstatic void viv_spectrum_sym(viv_arr A, viv_arr vals) {\n    int n = A.dims[0];\n#ifndef VIV_RELEASE\n    /* debug: verify the symmetry contract, run the method sweep, abort loudly on dissent */\n    double scale = 0.0;\n    for (int i = 0; i < n*n; i++) scale += A.data[i]*A.data[i];\n    scale = sqrt(scale); if (scale < 1e-300) scale = 1e-300;\n    double asym = 0.0;\n    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) {\n        double dlt = fabs(A.data[i*n+j] - A.data[j*n+i]);\n        if (dlt > asym) asym = dlt;\n    }\n    if (asym > 1e-9 * scale) {\n        viv_errf(\"spectrum_sym: contract violation - matrix not symmetric (max |A[i,j]-A[j,i]| = %s)\\n\", viv_g(asym));\n        viv_stop_exit(VIV_E_VALUE);\n    }\n    int dims1[1]; dims1[0] = n;\n    int dims2[2]; dims2[0] = n; dims2[1] = n;\n    viv_arr tvals = viv_arr_new(1, dims1);\n    viv_arr tvecs = viv_arr_new(2, dims2);\n    viv_eig_sym(A, tvals, tvecs);\n    double* ql = (double*)malloc((size_t)n * sizeof(double));\n    viv_tridiag_eigvals(n, A.data, ql);\n    double dev = 0.0;\n    for (int i = 0; i < n; i++) { double dlt = fabs(tvals.data[i] - ql[i]); if (dlt > dev) dev = dlt; }\n    if (dev > 1e-9 * scale) {\n        viv_errf(\"spectrum_sym: SOLVER SWEEP DISSENT\\n  jacobi:\");\n        for (int i = 0; i < n; i++) viv_errf(\" %s\", viv_gp(tvals.data[i], 12));\n        viv_errf(\"\\n  ql:    \");\n        for (int i = 0; i < n; i++) viv_errf(\" %s\", viv_gp(ql[i], 12));\n        viv_errf(\"\\n  max deviation %s exceeds tolerance %s\\n\", viv_g(dev), viv_g(1e-9*scale));\n        viv_stop_exit(VIV_E_ERROR);\n    }\n    for (int i = 0; i < n; i++) vals.data[i] = tvals.data[i];\n    viv_arr_free(tvals); viv_arr_free(tvecs); free(ql);\n#else\n    /* release: the auto-selected direct method (Householder + QL) */\n    double* d = (double*)malloc((size_t)n * sizeof(double));\n    viv_tridiag_eigvals(n, A.data, d);\n    for (int i = 0; i < n; i++) vals.data[i] = d[i];\n    free(d);\n#endif\n}\nstatic void viv_spectrum_sym_at(viv_arr A, viv_arr vals, int site) {\n    viv_finite_chk(\"spectrum_sym\", A, site);\n    viv_spectrum_sym(A, vals);\n}",
            })
            .with(RuntimeFragment {
                name: "solver_err",
                includes: &["#include <stdio.h>", "#include <unistd.h>"],
                deps: &["sites"],
                code: "/* A solver that stops says what the interpreter says, where it says it: the\n   `_at` form of the solver sets viv_err_site to the call's viv_sites row. */\nstatic int viv_err_site = 0;\nstatic void viv_solver_fail(int kind, const char* msg) {\n    viv_fflush(stdout);\n#ifdef VIV_BOUNDS\n    viv_errf(\"error: %s\\n%s\", msg, viv_sites[viv_err_site]);\n#else\n    viv_errf(\"error: %s\\n\", msg);\n#endif\n    viv_stop(kind);\n}",
            })
            .with(RuntimeFragment {
                name: "lin_solve",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"],
                deps: &["arr_type", "fmt_g", "solver_err", "fastla"],
                code: "static void viv_lin_solve(viv_arr A, viv_arr b, viv_arr x) {\n    int n = A.dims[0];\n    double* M = (double*)malloc((size_t)(n*n) * sizeof(double));\n    double* rhs = (double*)malloc((size_t)n * sizeof(double));\n    for (int i = 0; i < n*n; i++) M[i] = A.data[i];\n    for (int i = 0; i < n; i++) rhs[i] = b.data[i];\n    double scale = 0.0;\n    for (int i = 0; i < n*n; i++) scale += M[i]*M[i];\n    scale = sqrt(scale); if (scale < 1e-300) scale = 1e-300;\n#ifdef VIV_FAST_LA\n    if (viv_fast_solve(n, A.data, b.data, 1, x.data, scale)) goto certify;\n#endif\n    for (int k = 0; k < n; k++) {\n        int piv = k; double big = fabs(M[k*n+k]);\n        for (int i = k+1; i < n; i++) if (fabs(M[i*n+k]) > big) { big = fabs(M[i*n+k]); piv = i; }\n        if (big < 1e-14 * scale) {\n            char m[96];\n            snprintf(m, sizeof m, \"`lin_solve`: matrix is singular to working precision (pivot %d)\", k);\n            viv_solver_fail(VIV_E_VALUE, m);\n        }\n        if (piv != k) {\n            for (int j = 0; j < n; j++) { double t = M[k*n+j]; M[k*n+j] = M[piv*n+j]; M[piv*n+j] = t; }\n            double t = rhs[k]; rhs[k] = rhs[piv]; rhs[piv] = t;\n        }\n        for (int i = k+1; i < n; i++) {\n            double f = M[i*n+k] / M[k*n+k];\n            for (int j = k+1; j < n; j++) M[i*n+j] -= f * M[k*n+j];\n            rhs[i] -= f * rhs[k];\n        }\n    }\n    for (int i = n-1; i >= 0; i--) {\n        double s = rhs[i];\n        for (int j = i+1; j < n; j++) s -= M[i*n+j] * x.data[j];\n        x.data[i] = s / M[i*n+i];\n    }\ncertify: ;\n#ifndef VIV_RELEASE\n    /* residual certificate: ||Ax - b||_inf against the input scale */\n    double xnorm = 0.0, bnorm = 0.0, rmax = 0.0;\n    for (int i = 0; i < n; i++) { xnorm += x.data[i]*x.data[i]; bnorm += b.data[i]*b.data[i]; }\n    xnorm = sqrt(xnorm); bnorm = sqrt(bnorm);\n    for (int i = 0; i < n; i++) {\n        double s = -b.data[i];\n        for (int j = 0; j < n; j++) s += A.data[i*n+j] * x.data[j];\n        if (fabs(s) > rmax) rmax = fabs(s);\n    }\n    double tol = 1e-8 * (scale * xnorm + bnorm);\n    if (tol < 1e-300) tol = 1e-300;\n    if (rmax > tol) {\n        viv_errf(\"lin_solve: SOLVER CERTIFICATE FAILED - residual %s exceeds %s\\n\", viv_g(rmax), viv_g(tol));\n        viv_stop_exit(VIV_E_ERROR);\n    }\n#endif\n    free(M); free(rhs);\n}\nstatic void viv_lin_solve_at(viv_arr A, viv_arr b, viv_arr x, int site) {\n    viv_err_site = site;\n    viv_lin_solve(A, b, x);\n}",
            })
            .with(RuntimeFragment {
                name: "lin_solve_multi",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"],
                deps: &["arr_type", "fmt_g", "solver_err", "fastla"],
                code: "static void viv_lin_solve_multi(viv_arr A, viv_arr B, viv_arr X) {\n    int n = A.dims[0];\n    int m = B.dims[1];\n    double* lu = (double*)malloc((size_t)(n*n) * sizeof(double));\n    int* piv = (int*)malloc((size_t)n * sizeof(int));\n    double* col = (double*)malloc((size_t)n * sizeof(double));\n    for (int i = 0; i < n*n; i++) lu[i] = A.data[i];\n    double scale = 0.0;\n    for (int i = 0; i < n*n; i++) scale += lu[i]*lu[i];\n    scale = sqrt(scale); if (scale < 1e-300) scale = 1e-300;\n#ifdef VIV_FAST_LA\n    if (viv_fast_solve(n, A.data, B.data, m, X.data, scale)) goto certify;\n#endif\n    for (int k = 0; k < n; k++) {\n        int p = k; double big = fabs(lu[k*n+k]);\n        for (int i = k+1; i < n; i++) if (fabs(lu[i*n+k]) > big) { big = fabs(lu[i*n+k]); p = i; }\n        if (big < 1e-14 * scale) {\n            char m[96];\n            snprintf(m, sizeof m, \"`lin_solve_multi`: matrix is singular to working precision (pivot %d)\", k);\n            viv_solver_fail(VIV_E_VALUE, m);\n        }\n        piv[k] = p;\n        if (p != k) for (int j = 0; j < n; j++) { double t = lu[k*n+j]; lu[k*n+j] = lu[p*n+j]; lu[p*n+j] = t; }\n        for (int i = k+1; i < n; i++) {\n            double f = lu[i*n+k] / lu[k*n+k];\n            lu[i*n+k] = f;\n            for (int j = k+1; j < n; j++) lu[i*n+j] -= f * lu[k*n+j];\n        }\n    }\n    for (int j = 0; j < m; j++) {\n        for (int i = 0; i < n; i++) col[i] = B.data[i*m+j];\n        /* apply ALL pivots first (LAPACK order), then forward-substitute */\n        for (int k = 0; k < n; k++) { double t = col[k]; col[k] = col[piv[k]]; col[piv[k]] = t; }\n        for (int k = 0; k < n; k++) {\n            for (int i = k+1; i < n; i++) col[i] -= lu[i*n+k] * col[k];\n        }\n        for (int i = n-1; i >= 0; i--) {\n            double s = col[i];\n            for (int k2 = i+1; k2 < n; k2++) s -= lu[i*n+k2] * col[k2];\n            col[i] = s / lu[i*n+i];\n        }\n        for (int i = 0; i < n; i++) X.data[i*m+j] = col[i];\n    }\ncertify: ;\n#ifndef VIV_RELEASE\n    /* certificate: worst column residual, sampled over min(m, 8) evenly spaced columns\n       (a full check would double the O(n^2 m) cost) */\n    int nchk = (m < 8) ? m : 8;\n    for (int c = 0; c < nchk; c++) {\n        int j = (m == 1) ? 0 : (c * (m - 1)) / (nchk - 1 > 0 ? nchk - 1 : 1);\n        double xnorm = 0.0, bnorm = 0.0, rmax = 0.0;\n        for (int i = 0; i < n; i++) { xnorm += X.data[i*m+j]*X.data[i*m+j]; bnorm += B.data[i*m+j]*B.data[i*m+j]; }\n        for (int i = 0; i < n; i++) {\n            double s = -B.data[i*m+j];\n            for (int k = 0; k < n; k++) s += A.data[i*n+k] * X.data[k*m+j];\n            if (fabs(s) > rmax) rmax = fabs(s);\n        }\n        double tol = 1e-8 * (scale * sqrt(xnorm) + sqrt(bnorm));\n        if (tol < 1e-300) tol = 1e-300;\n        if (rmax > tol) {\n            viv_errf(\"lin_solve_multi: SOLVER CERTIFICATE FAILED - column %d residual %s exceeds %s\\n\", j, viv_g(rmax), viv_g(tol));\n            viv_stop_exit(VIV_E_ERROR);\n        }\n    }\n#endif\n    free(lu); free(piv); free(col);\n}\nstatic void viv_lin_solve_multi_at(viv_arr A, viv_arr B, viv_arr X, int site) {\n    viv_err_site = site;\n    viv_lin_solve_multi(A, B, X);\n}",
            })
            .with(RuntimeFragment {
                name: "cholesky",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"],
                deps: &["arr_type", "fmt_g"],
                code: "static void viv_cholesky(viv_arr A, viv_arr L) {\n    int n = A.dims[0];\n    for (int i = 0; i < n*n; i++) L.data[i] = 0.0;\n    for (int i = 0; i < n; i++) {\n        for (int j = 0; j <= i; j++) {\n            double s = A.data[i*n+j];\n            for (int k = 0; k < j; k++) s -= L.data[i*n+k] * L.data[j*n+k];\n            if (i == j) {\n                if (s <= 0.0) {\n                    viv_errf(\"cholesky: matrix is not positive definite (pivot %d = %s)\\n\", i, viv_g(s));\n                    viv_stop_exit(VIV_E_VALUE);\n                }\n                L.data[i*n+i] = sqrt(s);\n            } else {\n                L.data[i*n+j] = s / L.data[j*n+j];\n            }\n        }\n    }\n}",
            })
            .with(RuntimeFragment {
                name: "eig_general",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"],
                deps: &["arr_type", "fmt_g"],
                code: "static void viv_eig_general(viv_arr A, viv_arr re, viv_arr im) {\n    int n = A.dims[0];\n    double* a = (double*)malloc((size_t)(n*n) * sizeof(double));\n    double* wr = (double*)malloc((size_t)n * sizeof(double));\n    double* wi = (double*)malloc((size_t)n * sizeof(double));\n    for (int i = 0; i < n*n; i++) a[i] = A.data[i];\n    if (n == 1) { re.data[0] = a[0]; im.data[0] = 0.0; free(a); free(wr); free(wi); return; }\n    /* balanc: diagonal similarity scaling; badly scaled blocks stall QR */\n    for (;;) {\n        int last = 1;\n        for (int i = 0; i < n; i++) {\n            double r = 0.0, c = 0.0;\n            for (int j = 0; j < n; j++) if (j != i) { r += fabs(a[i*n+j]); c += fabs(a[j*n+i]); }\n            if (c != 0.0 && r != 0.0) {\n                double g = r / 2.0, f = 1.0, s = c + r;\n                while (c < g) { f *= 2.0; c *= 4.0; }\n                g = r * 2.0;\n                while (c > g) { f /= 2.0; c /= 4.0; }\n                if ((c + r) / f < 0.95 * s) {\n                    last = 0;\n                    double gi = 1.0 / f;\n                    for (int j = 0; j < n; j++) a[i*n+j] *= gi;\n                    for (int j = 0; j < n; j++) a[j*n+i] *= f;\n                }\n            }\n        }\n        if (last) break;\n    }\n    /* elmhes: reduce to upper Hessenberg by stabilized elimination */\n    for (int m = 1; m < n-1; m++) {\n        double x = 0.0; int ip = m;\n        for (int j = m; j < n; j++) if (fabs(a[j*n+m-1]) > fabs(x)) { x = a[j*n+m-1]; ip = j; }\n        if (ip != m) {\n            for (int j = m-1; j < n; j++) { double t = a[ip*n+j]; a[ip*n+j] = a[m*n+j]; a[m*n+j] = t; }\n            for (int j = 0; j < n; j++) { double t = a[j*n+ip]; a[j*n+ip] = a[j*n+m]; a[j*n+m] = t; }\n        }\n        if (x != 0.0) {\n            for (int i = m+1; i < n; i++) {\n                double y = a[i*n+m-1];\n                if (y != 0.0) {\n                    y /= x; a[i*n+m-1] = y;\n                    for (int j = m; j < n; j++) a[i*n+j] -= y * a[m*n+j];\n                    for (int j = 0; j < n; j++) a[j*n+m] += y * a[j*n+i];\n                }\n            }\n        }\n    }\n    /* hqr: Francis double-shift QR, eigenvalues only */\n    double anorm = 0.0;\n    for (int i = 0; i < n; i++) {\n        int lo = (i >= 1) ? i-1 : 0;\n        for (int j = lo; j < n; j++) anorm += fabs(a[i*n+j]);\n    }\n    int nn = n - 1; double t = 0.0;\n    while (nn >= 0) {\n        int its = 0;\n        for (;;) {\n            int l = nn;\n            while (l >= 1) {\n                double s = fabs(a[(l-1)*n+(l-1)]) + fabs(a[l*n+l]);\n                if (s == 0.0) s = anorm;\n                if (fabs(a[l*n+l-1]) + s == s) { a[l*n+l-1] = 0.0; break; }\n                l--;\n            }\n            double x = a[nn*n+nn];\n            if (l == nn) { wr[nn] = x + t; wi[nn] = 0.0; nn--; break; }\n            double y = a[(nn-1)*n+(nn-1)];\n            double w = a[nn*n+nn-1] * a[(nn-1)*n+nn];\n            if (l == nn-1) {\n                double p = 0.5*(y - x);\n                double q = p*p + w;\n                double z = sqrt(fabs(q));\n                x += t;\n                if (q >= 0.0) {\n                    z = p + (p >= 0.0 ? fabs(z) : -fabs(z));\n                    wr[nn-1] = wr[nn] = x + z;\n                    if (z != 0.0) wr[nn] = x - w/z;\n                    wi[nn-1] = wi[nn] = 0.0;\n                } else {\n                    wr[nn-1] = wr[nn] = x + p;\n                    wi[nn-1] = -z; wi[nn] = z;\n                }\n                nn -= 2; break;\n            }\n            if (its == 100) {\n                viv_errf(\"eig_general: too many QR iterations\\n\");\n                viv_stop_exit(VIV_E_ERROR);\n            }\n            if (its > 0 && its % 10 == 0) {\n                t += x;\n                for (int i = 0; i <= nn; i++) a[i*n+i] -= x;\n                double s = fabs(a[nn*n+nn-1]) + fabs(a[(nn-1)*n+nn-2]);\n                y = x = 0.75 * s;\n                w = -0.4375 * s * s;\n            }\n            its++;\n            int m = nn - 2;\n            double p = 0.0, q = 0.0, r = 0.0;\n            for (;;) {\n                double z = a[m*n+m];\n                double rr = x - z, ss = y - z;\n                p = (rr*ss - w) / a[(m+1)*n+m] + a[m*n+m+1];\n                q = a[(m+1)*n+(m+1)] - z - rr - ss;\n                r = a[(m+2)*n+(m+1)];\n                double s = fabs(p) + fabs(q) + fabs(r);\n                p /= s; q /= s; r /= s;\n                if (m == l) break;\n                double u = fabs(a[m*n+m-1]) * (fabs(q) + fabs(r));\n                double v = fabs(p) * (fabs(a[(m-1)*n+(m-1)]) + fabs(z) + fabs(a[(m+1)*n+(m+1)]));\n                if (u + v == v) break;\n                m--;\n            }\n            for (int i = m+2; i <= nn; i++) {\n                a[i*n+i-2] = 0.0;\n                if (i != m+2) a[i*n+i-3] = 0.0;\n            }\n            for (int k = m; k <= nn-1; k++) {\n                if (k != m) {\n                    p = a[k*n+k-1]; q = a[(k+1)*n+k-1];\n                    r = (k != nn-1) ? a[(k+2)*n+k-1] : 0.0;\n                    x = fabs(p) + fabs(q) + fabs(r);\n                    if (x != 0.0) { p /= x; q /= x; r /= x; }\n                }\n                double smag = sqrt(p*p + q*q + r*r);\n                double s = (p >= 0.0) ? fabs(smag) : -fabs(smag);\n                if (s != 0.0) {\n                    if (k == m) {\n                        if (l != m) a[k*n+k-1] = -a[k*n+k-1];\n                    } else {\n                        a[k*n+k-1] = -s * x;\n                    }\n                    p += s;\n                    x = p/s; y = q/s;\n                    double z = r/s;\n                    q /= p; r /= p;\n                    for (int j = k; j <= nn; j++) {\n                        double pj = a[k*n+j] + q*a[(k+1)*n+j];\n                        if (k != nn-1) { pj += r*a[(k+2)*n+j]; a[(k+2)*n+j] -= pj*z; }\n                        a[(k+1)*n+j] -= pj*y;\n                        a[k*n+j] -= pj*x;\n                    }\n                    int mmin = (nn < k+3) ? nn : k+3;\n                    for (int i = l; i <= mmin; i++) {\n                        double pi = x*a[i*n+k] + y*a[i*n+k+1];\n                        if (k != nn-1) { pi += z*a[i*n+k+2]; a[i*n+k+2] -= pi*r; }\n                        a[i*n+k+1] -= pi*q;\n                        a[i*n+k] -= pi;\n                    }\n                }\n            }\n        }\n    }\n    /* deterministic order: sort by (re, im) */\n    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++)\n        if (wr[j] < wr[i] || (wr[j] == wr[i] && wi[j] < wi[i])) {\n            double tr_ = wr[i]; wr[i] = wr[j]; wr[j] = tr_;\n            double ti_ = wi[i]; wi[i] = wi[j]; wi[j] = ti_;\n        }\n#ifndef VIV_RELEASE\n    /* trace-invariant certificates: tr(A) = sum re, tr(A^2) = sum(re^2 - im^2) */\n    double sc = 0.0;\n    for (int i = 0; i < n*n; i++) sc += A.data[i]*A.data[i];\n    sc = sqrt(sc); if (sc < 1e-300) sc = 1e-300;\n    double tr = 0.0, s1 = 0.0, tr2 = 0.0, s2 = 0.0;\n    for (int i = 0; i < n; i++) { tr += A.data[i*n+i]; s1 += wr[i]; s2 += wr[i]*wr[i] - wi[i]*wi[i]; }\n    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) tr2 += A.data[i*n+j]*A.data[j*n+i];\n    if (fabs(tr - s1) > 1e-8*n*sc || fabs(tr2 - s2) > 1e-8*n*sc*sc) {\n        viv_errf(\"eig_general: SOLVER CERTIFICATE FAILED - trace invariants do not close (|%s| vs %s, |%s| vs %s)\\n\",\n                viv_g(tr - s1), viv_g(1e-8*n*sc), viv_g(tr2 - s2), viv_g(1e-8*n*sc*sc));\n        viv_stop_exit(VIV_E_ERROR);\n    }\n#endif\n    for (int i = 0; i < n; i++) { re.data[i] = wr[i]; im.data[i] = wi[i]; }\n    free(a); free(wr); free(wi);\n}",
            })
            .with(RuntimeFragment {
                name: "eig_near",
                includes: &["#include <stdlib.h>", "#include <stdio.h>", "#include <math.h>"],
                deps: &["arr_type", "arr_new", "arr_free", "eig_general", "fmt_g"],
                code: "static void viv_eig_near_core(int n, const double* Adata, double sigma, int k, int m, double* outre, double* outim, int* outk) {\n    double* lu = (double*)malloc((size_t)(n*n) * sizeof(double));\n    int* piv = (int*)malloc((size_t)n * sizeof(int));\n    for (int i = 0; i < n*n; i++) lu[i] = Adata[i];\n    for (int i = 0; i < n; i++) lu[i*n+i] -= sigma;\n    double scale = 0.0;\n    for (int i = 0; i < n*n; i++) scale += lu[i]*lu[i];\n    scale = sqrt(scale); if (scale < 1e-300) scale = 1e-300;\n    for (int kk = 0; kk < n; kk++) {\n        int p = kk; double big = fabs(lu[kk*n+kk]);\n        for (int i = kk+1; i < n; i++) if (fabs(lu[i*n+kk]) > big) { big = fabs(lu[i*n+kk]); p = i; }\n        if (big < 1e-14 * scale) {\n            viv_errf(\"eig_near: sigma sits on the spectrum (singular shift); move it\\n\");\n            viv_stop_exit(VIV_E_VALUE);\n        }\n        piv[kk] = p;\n        if (p != kk) for (int j = 0; j < n; j++) { double t = lu[kk*n+j]; lu[kk*n+j] = lu[p*n+j]; lu[p*n+j] = t; }\n        for (int i = kk+1; i < n; i++) {\n            double f = lu[i*n+kk] / lu[kk*n+kk];\n            lu[i*n+kk] = f;\n            for (int j = kk+1; j < n; j++) lu[i*n+j] -= f * lu[kk*n+j];\n        }\n    }\n    if (m > n) m = n;\n    double* V = (double*)malloc((size_t)((m+1)*n) * sizeof(double));\n    double* H = (double*)malloc((size_t)((m+1)*m) * sizeof(double));\n    double* w = (double*)malloc((size_t)n * sizeof(double));\n    for (int i = 0; i < (m+1)*m; i++) H[i] = 0.0;\n    double nrm = 0.0;\n    for (int i = 0; i < n; i++) { V[i] = sin(1.0 + 0.7*i); nrm += V[i]*V[i]; }\n    nrm = sqrt(nrm);\n    for (int i = 0; i < n; i++) V[i] /= nrm;\n    int steps = m;\n    for (int j = 0; j < m; j++) {\n        for (int i = 0; i < n; i++) w[i] = V[j*n+i];\n        /* apply ALL pivots first (LAPACK order), then substitute */\n        for (int kk = 0; kk < n; kk++) { double t = w[kk]; w[kk] = w[piv[kk]]; w[piv[kk]] = t; }\n        for (int kk = 0; kk < n; kk++) for (int i = kk+1; i < n; i++) w[i] -= lu[i*n+kk] * w[kk];\n        for (int i = n-1; i >= 0; i--) {\n            double s = w[i];\n            for (int j2 = i+1; j2 < n; j2++) s -= lu[i*n+j2] * w[j2];\n            w[i] = s / lu[i*n+i];\n        }\n        for (int i = 0; i <= j; i++) {\n            double dot = 0.0;\n            for (int t = 0; t < n; t++) dot += V[i*n+t] * w[t];\n            H[i*m+j] = dot;\n            for (int t = 0; t < n; t++) w[t] -= dot * V[i*n+t];\n        }\n        double b = 0.0;\n        for (int t = 0; t < n; t++) b += w[t]*w[t];\n        b = sqrt(b);\n        H[(j+1)*m+j] = b;\n        if (b < 1e-13) { steps = j+1; break; }\n        for (int t = 0; t < n; t++) V[(j+1)*n+t] = w[t] / b;\n    }\n    int s = steps;\n    int dims2[2]; dims2[0] = s; dims2[1] = s;\n    int dims1[1]; dims1[0] = s;\n    viv_arr Hs = viv_arr_new(2, dims2);\n    viv_arr tr = viv_arr_new(1, dims1);\n    viv_arr ti = viv_arr_new(1, dims1);\n    for (int i = 0; i < s; i++) for (int j = 0; j < s; j++) Hs.data[i*s+j] = H[i*m+j];\n    viv_eig_general(Hs, tr, ti);\n    /* theta -> lambda = sigma + 1/theta; sort by |lambda - sigma| = 1/|theta| */\n    double* dist = (double*)malloc((size_t)s * sizeof(double));\n    double* lre = (double*)malloc((size_t)s * sizeof(double));\n    double* lim = (double*)malloc((size_t)s * sizeof(double));\n    int cnt = 0;\n    for (int i = 0; i < s; i++) {\n        double den = tr.data[i]*tr.data[i] + ti.data[i]*ti.data[i];\n        if (den < 1e-300) continue;\n        dist[cnt] = 1.0 / sqrt(den);\n        lre[cnt] = sigma + tr.data[i] / den;\n        lim[cnt] = -ti.data[i] / den;\n        cnt++;\n    }\n    for (int i = 0; i < cnt; i++) for (int j = i+1; j < cnt; j++) {\n        int swap = 0;\n        if (dist[j] < dist[i]) swap = 1;\n        else if (dist[j] == dist[i] && (lre[j] < lre[i] || (lre[j] == lre[i] && lim[j] < lim[i]))) swap = 1;\n        if (swap) {\n            double t;\n            t = dist[i]; dist[i] = dist[j]; dist[j] = t;\n            t = lre[i]; lre[i] = lre[j]; lre[j] = t;\n            t = lim[i]; lim[i] = lim[j]; lim[j] = t;\n        }\n    }\n    if (k > cnt) k = cnt;\n    for (int i = 0; i < k; i++) { outre[i] = lre[i]; outim[i] = lim[i]; }\n    *outk = k;\n    free(lu); free(piv); free(V); free(H); free(w);\n    viv_arr_free(Hs); viv_arr_free(tr); viv_arr_free(ti); free(dist); free(lre); free(lim);\n}\nstatic double viv_eig_near(viv_arr A, double sigma, double kf, viv_arr re, viv_arr im) {\n    int n = A.dims[0];\n    int k = (int)kf;\n    int m = 4*k + 20; if (m < 40) m = 40; if (m > n) m = n;\n    int got = 0;\n    viv_eig_near_core(n, A.data, sigma, k, m, re.data, im.data, &got);\n#ifndef VIV_RELEASE\n    if (m + 10 <= n) {\n        /* dual-depth agreement certificate */\n        double* r2 = (double*)malloc((size_t)k * sizeof(double));\n        double* i2 = (double*)malloc((size_t)k * sizeof(double));\n        int got2 = 0;\n        viv_eig_near_core(n, A.data, sigma, k, m + 10, r2, i2, &got2);\n        double scale = 0.0;\n        for (int i = 0; i < n*n; i++) scale += A.data[i]*A.data[i];\n        scale = sqrt(scale); if (scale < 1e-300) scale = 1e-300;\n        double dev = 0.0;\n        int kk = got < got2 ? got : got2;\n        for (int i = 0; i < kk; i++) {\n            double d1 = fabs(re.data[i] - r2[i]);\n            double d2 = fabs(im.data[i] - i2[i]);\n            if (d1 > dev) dev = d1;\n            if (d2 > dev) dev = d2;\n        }\n        if (dev > 1e-7 * scale) {\n            viv_errf(\"eig_near: SOLVER CERTIFICATE FAILED - Krylov depths %d and %d disagree by %s\\n\", m, m+10, viv_g(dev));\n            viv_stop_exit(VIV_E_ERROR);\n        }\n        free(r2); free(i2);\n    }\n#endif\n    return 0.0;\n}",
            })
    }

fn emit_item_c(node: &Node<Viv>, cx: &mut CEmitCx<Viv>) -> CItem {
        // an item that holds a string needs the string types
        let text = match &node.kind {
            Viv::Func { params, ret, locals, .. } => params.iter().chain(locals.iter()).map(|p| p.1).chain(std::iter::once(*ret)).any(strings::is_text),
            Viv::Main { locals, .. } => locals.iter().any(|p| strings::is_text(p.1)),
            Viv::Global { ty, .. } => strings::is_text(*ty),
            Viv::RecordDef { fields, .. } => fields.iter().any(|f| strings::is_text(f.1)),
            _ => false,
        };
        if text {
            cx.need("str_list");
        }
        // what the def's variables always hold (lower/intidx.rs)
        intidx::enter(cx.program(), match &node.kind {
            Viv::Func { name, .. } => Some(name.clone()),
            Viv::Main { .. } => Some(crate::parser::MAIN_DEF.to_string()),
            _ => None,
        });
        match &node.kind {
            Viv::Func { name, src, params, ret, locals, arena } => {
                STEP_DEF.with(|d| *d.borrow_mut() = src.clone());
                enter_loop_scope(cx, params.iter().chain(locals.iter()).cloned());
                LOOP_SCOPE.with(|s| {
                    let mut s = s.borrow_mut();
                    s.cur = name.clone();
                    let ps: Vec<String> = params.iter().map(|(p, _)| p.clone()).collect();
                    s.dj_fresh = if s.dj_defs.contains(name) || pinned_in(cx, node.id) { dj_fresh(cx.program(), node.id, &ps) } else { Default::default() };
                    s.dj_params = ps.into_iter().collect();
                });
                let mut body: Vec<CStmt> = node.children.iter().enumerate().map(|(t, &c)| {
                    intidx::at_top(t);
                    cx.stmt(c)
                }).collect();
                // A def the story records calls of (it starts with StoryCall)
                // records its return on every way out: after the value and the
                // step exits, before the arena release below.
                if let Some(Viv::StoryCall { def }) = node.children.first().map(|&c| cx.kind(c).clone()) {
                    let id = node.id;
                    let falls_off = !matches!(body.last().map(|s| &s.kind), Some(CStmtKind::Block(_)) | Some(CStmtKind::Return(_))) || matches!(ret, Ty::Void);
                    body = body.into_iter().map(|s| story_returns(s, id, *ret, &def)).collect();
                    if falls_off {
                        body.push(story_guard(id, story_mark(id, "ret", &def)));
                    }
                }
                // The array arena. A function that can allocate notes the arena's
                // height on entry and, on every way out, frees what was allocated
                // since -- except the array it returns, which becomes the caller's,
                // or every array in the record it returns.
                // (a def whose body can allocate nothing frees nothing: no mark)
                let rets = LOOP_SCOPE.with(|s| s.borrow().rets.clone());
                let allocates = *arena && node.children.iter().any(|&c| c_allocates(cx, &rets, c));
                if allocates {
                    cx.need("arena");
                    if matches!(ret, Ty::Record(_) | Ty::Recs(_)) {
                        cx.need("arena_keep");
                    }
                    let id = node.id;
                    body = body.into_iter().map(|s| arena_returns(s, id, *ret)).collect();
                    if !matches!(body.last().map(|s| &s.kind), Some(CStmtKind::Block(_)) | Some(CStmtKind::Return(_))) || matches!(ret, Ty::Void) {
                        body.push(CStmt::expr(id, CExpr::raw(id, "viv_arena_release(viv_mark, (const double*)0)")));
                    }
                    body.insert(0, CStmt::decl(id, CType::Named("size_t".to_string()), "viv_mark", CExpr::raw(id, "viv_arena_top")));
                }
                // a def returning a number that reaches its end gives 0, as in
                // the interpreter and the wasm backend (every other type is
                // refused there by the lowering, lower_def_as)
                if matches!(ret, Ty::Num) && !matches!(body.last().map(|s| &s.kind), Some(CStmtKind::Return(_))) {
                    body.push(CStmt::ret(node.id, CExpr::lit(node.id, "0.0")));
                }
                // An array parameter arrives as `p_viv_in` and is copied into a
                // local `p` on entry. The ABI passes a 48-byte struct through
                // memory, and clang then reloaded its dimensions after every
                // store into any array, inside every stencil loop: NPB MG's
                // resid and psinv took twice as long (gcc kept them in
                // registers). A local whose address is never taken is split
                // into registers. The copy is the struct only, not the data.
                let mut cparams = Vec::with_capacity(params.len());
                for (n, t) in params.iter() {
                    let ct = t.to_c();
                    if matches!(&ct, CType::Named(c) if c == "viv_arr") {
                        let incoming = format!("{n}_viv_in");
                        // `(void)p;`: a def that never reads the parameter (a placeholder) would
                        // otherwise warn that the local is unused
                        body.insert(0, CStmt::expr(node.id, CExpr::unary(node.id, "(void)", CExpr::var(node.id, n.clone()))));
                        body.insert(0, CStmt::decl(node.id, ct.clone(), n.clone(), CExpr::raw(node.id, incoming.clone())));
                        cparams.push((ct, incoming));
                    } else {
                        cparams.push((ct, n.clone()));
                    }
                }
                CItem::Func(CFunc {
                    origin: Some(node.id),
                    ret: ret.to_c(),
                    name: name.clone(),
                    params: cparams,
                    locals: locals.iter().map(|(n, t)| (t.to_c(), n.clone())).collect(),
                    body,
                })
            }
            Viv::Main { locals, globals } => {
                // checkpoint() arrives as an ExprStmt wrapping a Ckpt node
                fn is_ckpt(cx: &CEmitCx<Viv>, c: knotkit::ir::NodeId) -> bool {
                    matches!(cx.kind(c), Viv::Ckpt)
                        || (matches!(cx.kind(c), Viv::ExprStmt)
                            && cx.children(c).first().map(|&g| matches!(cx.kind(g), Viv::Ckpt)).unwrap_or(false))
                }
                let has_ckpt = node.children.iter().any(|&c| is_ckpt(cx, c));
                STEP_DEF.with(|d| *d.borrow_mut() = crate::parser::MAIN_DEF.to_string());
                enter_loop_scope(cx, locals.iter().chain(globals.iter()).cloned());
                LOOP_SCOPE.with(|s| {
                    let mut s = s.borrow_mut();
                    s.cur = crate::parser::MAIN_DEF.to_string();
                    s.dj_fresh = if pinned_in(cx, node.id) { dj_fresh(cx.program(), node.id, &[]) } else { Default::default() };
                    s.dj_params = Default::default();
                });
                if !has_ckpt {
                    return CItem::Main {
                        locals: locals.iter().map(|(n, t)| (t.to_c(), n.clone())).collect(),
                        body: node.children.iter().enumerate().map(|(t, &c)| {
                            intidx::at_top(t);
                            cx.stmt(c)
                        }).collect(),
                    };
                }
                // checkpoint mode: statement segments between markers run under
                // resume guards, and each marker saves (VIV_CKPT_DIR) or loads
                // (VIV_RESUME + VIV_CKPT_IN) every top-level number/array. One
                // binary serves full runs and per-cell resumes.
                cx.need("ckpt");
                let id = node.id;
                // name-tagged snapshot entries: the loader matches by NAME, so
                // a checkpoint survives edits to later cells that add or drop
                // variables — unknown entries are skipped, absent ones stay
                // zero-initialized (correct: they are defined by later cells)
                let saves = {
                    let mut s = String::new();
                    for (n, t) in globals {
                        match t {
                            Ty::Num | Ty::Bool => s.push_str(&format!("viv_ckw_f64(viv_ckf, \"{n}\", &{n}); ")),
                            Ty::Cplx => s.push_str(&format!("viv_ckw_cplx(viv_ckf, \"{n}\", &{n}); ")),
                            Ty::Arr(Elem::Num) => s.push_str(&format!("viv_ckw_arr(viv_ckf, \"{n}\", &{n}); ")),
                            _ => {}
                        }
                    }
                    s
                };
                let loads = {
                    let mut s = String::from("char viv_nm[256]; unsigned char viv_tg; while (viv_ckr_head(viv_ckf, viv_nm, 256, &viv_tg)) { if (0) {} ");
                    for (n, t) in globals {
                        match t {
                            Ty::Num | Ty::Bool => s.push_str(&format!("else if (strcmp(viv_nm, \"{n}\") == 0) viv_ckr_f64(viv_ckf, viv_tg, &{n}); ")),
                            Ty::Cplx => s.push_str(&format!("else if (strcmp(viv_nm, \"{n}\") == 0) viv_ckr_cplx(viv_ckf, viv_tg, &{n}); ")),
                            Ty::Arr(Elem::Num) => s.push_str(&format!("else if (strcmp(viv_nm, \"{n}\") == 0) viv_ckr_arr(viv_ckf, viv_tg, &{n}); ")),
                            _ => {}
                        }
                    }
                    s.push_str("else viv_ckr_skip(viv_ckf, viv_tg); } ");
                    s
                };
                let mut body: Vec<CStmt> = Vec::new();
                body.push(CStmt::expr(id, CExpr::raw(id, "const char* viv_ckpt_dir = getenv(\"VIV_CKPT_DIR\")")));
                body.push(CStmt::expr(id, CExpr::raw(id, "const char* viv_ckpt_in = getenv(\"VIV_CKPT_IN\")")));
                body.push(CStmt::expr(id, CExpr::raw(id, "int viv_resume = getenv(\"VIV_RESUME\") ? atoi(getenv(\"VIV_RESUME\")) : -1")));
                body.push(CStmt::expr(id, CExpr::raw(id,
                    "if (viv_resume >= 0 && !viv_ckpt_in) { viv_errf(\"VIV_RESUME set without VIV_CKPT_IN\\n\"); viv_stop_exit(VIV_E_ERROR); }")));
                let mut seg: Vec<CStmt> = Vec::new();
                let mut seg_idx: usize = 0;
                let mut marker: usize = 0;
                for &c in &node.children {
                    if is_ckpt(cx, c) {
                        body.push(CStmt {
                            origin: Some(id),
                            kind: CStmtKind::If {
                                cond: CExpr::raw(id, format!("viv_resume < {seg_idx}")),
                                then_body: std::mem::take(&mut seg),
                                else_body: Vec::new(),
                            },
                        });
                        body.push(CStmt::expr(id, CExpr::raw(id, format!(
                            "{{ if (viv_resume == {m}) {{ FILE* viv_ckf = fopen(viv_ckpt_in, \"rb\"); if (!viv_ckf) {{ viv_errf(\"viv ckpt in: %s\\n\", strerror(errno)); viv_stop_exit(VIV_E_ERROR); }} \
                             char viv_m4[4]; if (fread(viv_m4, 1, 4, viv_ckf) != 4 || memcmp(viv_m4, \"VCK2\", 4) != 0) {{ viv_errf(\"ckpt: format mismatch\\n\"); viv_stop_exit(VIV_E_ERROR); }} \
                             {loads}fclose(viv_ckf); }} \
                             if (viv_ckpt_dir && viv_resume <= {m}) {{ char viv_ckp[1024]; snprintf(viv_ckp, sizeof viv_ckp, \"%s/ck_%d.bin\", viv_ckpt_dir, {m}); \
                             FILE* viv_ckf = fopen(viv_ckp, \"wb\"); if (!viv_ckf) {{ viv_errf(\"viv ckpt out: %s\\n\", strerror(errno)); viv_stop_exit(VIV_E_ERROR); }} \
                             fwrite(\"VCK2\", 1, 4, viv_ckf); {saves}fclose(viv_ckf); }} }}",
                            m = marker,
                        ))));
                        marker += 1;
                        seg_idx += 1;
                    } else {
                        seg.push(cx.stmt(c));
                    }
                }
                body.push(CStmt {
                    origin: Some(id),
                    kind: CStmtKind::If {
                        cond: CExpr::raw(id, format!("viv_resume < {seg_idx}")),
                        then_body: seg,
                        else_body: Vec::new(),
                    },
                });
                CItem::Main {
                    locals: locals.iter().map(|(n, t)| (t.to_c(), n.clone())).collect(),
                    body,
                }
            }
            Viv::Global { name, ty } => CItem::Global { ty: ty.to_c(), name: name.clone() },
            Viv::RecFns { rec, kind, fields } => rec_fn_c(node.id, rec, *kind, fields, cx),
            Viv::RecordDef { name, fields } => CItem::Struct {
                name: format!("viv_rec_{name}"),
                fields: fields.iter().map(|(n, t)| (t.to_c(), n.clone())).collect(),
            },
            Viv::LambdaDef { name, param, captures } => {
                // Lifted to: double <name>(const double* viv_caps, double <param>)
                // { double <cap> = viv_caps[i]; ... return <body>; }
                let mut body = Vec::new();
                // Silence -Wunused-parameter for capture-free lambdas (the closure
                // signature is uniform whether or not captures are used).
                body.push(CStmt::expr(node.id, CExpr::raw(node.id, "(void)viv_caps")));
                for (i, (cap, _)) in captures.iter().enumerate() {
                    body.push(CStmt::decl(node.id, CType::Double, cap.clone(), CExpr::raw(node.id, format!("viv_caps[{i}]"))));
                }
                body.push(CStmt::ret(node.id, cx.expr(node.children[0])));
                CItem::Func(CFunc {
                    origin: Some(node.id),
                    ret: CType::Double,
                    name: name.clone(),
                    params: vec![
                        (CType::Named("const double*".to_string()), "viv_caps".to_string()),
                        (CType::Double, param.clone()),
                    ],
                    locals: Vec::new(),
                    body,
                })
            }
            Viv::FnWrapDef { name, sig } => {
                // static double viv_wrap_<name>(const viv_arg* a, int n) {
                //     (void)n; return <name>(a[0].fn, a[1].num, ...);
                // }
                // The `.fn` / `.num` choice is the signature's, made here at
                // compile time: the union carries no tag, because the wrapper
                // and the call site agree on the shape statically.
                cx.need("viv_fn");
                let callargs = (0..sig.arity as usize)
                    .map(|i| CExpr::raw(node.id, format!("a[{i}].{}", sig.kind(i).member())))
                    .collect::<Vec<_>>();
                let mut body = vec![CStmt::expr(node.id, CExpr::raw(node.id, "(void)n"))];
                // Every wrapper has the one signature a function pointer can
                // have, so the result is boxed too, into the member its kind names.
                let boxer = match sig.ret {
                    ArgKind::Arr => "viv_arga",
                    _ => "viv_argn",
                };
                body.push(CStmt::ret(node.id, CExpr::call(node.id, boxer, vec![CExpr::call(node.id, name.clone(), callargs)])));
                CItem::Func(CFunc {
                    origin: Some(node.id),
                    ret: CType::Named("viv_arg".to_string()),
                    name: format!("viv_wrap_{name}"),
                    params: vec![
                        (CType::Named("const viv_arg*".to_string()), "a".to_string()),
                        (CType::Named("int".to_string()), "n".to_string()),
                    ],
                    locals: Vec::new(),
                    body,
                })
            }
            // the program's enums: their structs and helpers, as C text
            Viv::EnumDefs { enums: defs } => {
                cx.need("enums");
                CItem::Decls(enums::c_decls(defs))
            }
            _ => unreachable!("non-item node as a root: {:?}", node.kind),
        }
    }

fn emit_stmt_c(node: &Node<Viv>, cx: &mut CEmitCx<Viv>) -> CStmt {
        let id = node.id;
        match &node.kind {
            // a checkpoint() that is not a direct child of main (e.g. inside a
            // loop) has no cell-boundary meaning; it compiles to nothing
            Viv::Ckpt => CStmt { origin: Some(id), kind: CStmtKind::Nop },
            Viv::Assign { name } => match fuse_assign_c(id, name, node.children[0], cx) {
                Some(s) => s,
                None => CStmt::assign(id, name.clone(), cx.expr(node.children[0])),
            },
            // `base.field = value` and `xs[i] = r`: the value first, into a
            // temporary, then the place (whose indices are checked as it is
            // reached), as the interpreter and the wasm module order them
            Viv::SetField { field, ty } => {
                let value = cx.expr(node.children[1]);
                let place = CExpr::field(id, cx.expr(node.children[0]), field.clone());
                CStmt::block(id, vec![
                    CStmt::decl(id, ty.to_c(), "viv_put", value),
                    CStmt::expr(id, CExpr::binary(id, place, "=", CExpr::var(id, "viv_put"))),
                ])
            }
            Viv::RecSetIndex { rec, at } => {
                cx.need("recs");
                let value = cx.expr(node.children[2]);
                let place = rec_elem(cx, id, rec, node.children[0], node.children[1], *at, node.span);
                CStmt::block(id, vec![
                    CStmt::decl(id, CType::Named(format!("viv_rec_{rec}")), "viv_put", value),
                    CStmt::expr(id, CExpr::binary(id, place, "=", CExpr::var(id, "viv_put"))),
                ])
            }
            Viv::CSetIndex { at } => {
                // children: [base, idx..., value]. A real `value` promotes to
                // `double complex` at the call, as C's conversions already do.
                let nidx = node.children.len() - 2;
                let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                // Int indices stay 64-bit integers (as viv_lgetN does for real arrays)
                let int_ix = nidx <= 3 && node.children[1..=nidx].iter().all(|&c| intidx::is_int_node(cx.program(), c));
                let (func, frag) = match (nidx, int_ix) {
                    (1, true) => ("viv_lcset1", "lcset1"),
                    (2, true) => ("viv_lcset2", "lcset2"),
                    (3, true) => ("viv_lcset3", "lcset3"),
                    (1, _) => ("viv_cset1", "cset1"),
                    (2, _) => ("viv_cset2", "cset2"),
                    (3, _) => ("viv_cset3", "cset3"),
                    (4, _) => ("viv_cset4", "cset4"),
                    _ => unreachable!("unsupported complex set rank {nidx}"),
                };
                cx.need(frag);
                let func = with_site(func, *at, node.span, &mut args, id);
                CStmt::expr(id, CExpr::call(id, &func, args))
            }
            Viv::ShowCArr { label } => {
                cx.need("show_carr");
                let value = cx.expr(node.children[0]);
                let label_lit = CExpr::raw(id, c_str(label));
                CStmt::expr(id, CExpr::call(id, "viv_show_carr", vec![label_lit, value]))
            }
            Viv::SetIndex { at } => {
                // children: [base, idx..., value]; emit viv_setN(base, idx..., value).
                let n = node.children.len();
                let nidx = n - 2;
                if is_depth_cell(cx.program(), node.children[0], &node.children[1..n - 1]) {
                    cx.need("depth");
                    // `viv__depth[0] = viv__depth[0] + 1` (or `- 1`), the only
                    // writes there are, as an integer step
                    return match depth_step(cx.program(), node.children[n - 1]) {
                        0 => CStmt::assign(id, "viv_depth_g", CExpr::unary(id, "(long)", cx.expr(node.children[n - 1]))),
                        d => CStmt::assign(id, "viv_depth_g", CExpr::binary(id, CExpr::var(id, "viv_depth_g"), if d > 0 { "+" } else { "-" }, CExpr::lit(id, "1"))),
                    };
                }
                if !dj_write(cx, node.children[0]) {
                    if let Some(fast) = int_access(cx, node, &node.children[1..n - 1], Some(node.children[n - 1]), *at) {
                        return CStmt::expr(id, fast);
                    }
                }
                intidx::note(&loc_text(node.span), nidx, "double", 0);
                let mut args = vec![cx.expr(node.children[0])];
                for &c in &node.children[1..n - 1] {
                    args.push(cx.expr(c));
                }
                args.push(cx.expr(node.children[n - 1]));
                let func = match nidx {
                    1 => "viv_set1",
                    2 => "viv_set2",
                    3 => "viv_set3",
                    4 => "viv_set4",
                    _ => unreachable!("unsupported array rank {} in set", nidx),
                };
                cx.need(match nidx { 1 => "set1", 2 => "set2", 3 => "set3", 4 => "set4", _ => unreachable!() });
                // in a pinned loop's reach, a checked build records the write
                let func = if dj_write(cx, node.children[0]) {
                    cx.need("disjoint");
                    match nidx {
                        1 => "viv_dj_set1",
                        2 => "viv_dj_set2",
                        3 => "viv_dj_set3",
                        _ => "viv_dj_set4",
                    }
                } else {
                    func
                };
                let func = with_site(func, *at, node.span, &mut args, id);
                CStmt::expr(id, CExpr::call(id, &func, args))
            }
            Viv::Return => CStmt::ret(id, cx.expr(node.children[0])),
            Viv::ExprStmt => CStmt::expr(id, cx.expr(node.children[0])),
            Viv::Nop => CStmt::nop(),
            Viv::If => {
                let cond = cx.expr(node.children[0]);
                let then_body = block(cx, node.children[1]);
                let else_body = block(cx, node.children[2]);
                CStmt::if_(id, cond, then_body, else_body)
            }
            Viv::While => {
                let cond = cx.expr(node.children[0]);
                let mut body = block(cx, node.children[1]);
                if poll_loop(cx, node.children[1]) {
                    body.insert(0, poll_stmt(id));
                }
                match loop_keeps(cx, node.children[1], Some(node.children[0])) {
                    Some(keeps) => {
                        body.splice(0..0, release_stmts(id, &keeps));
                        loop_with_release(id, CStmt::while_(id, cond, body), release_stmts(id, &keeps))
                    }
                    None => CStmt::while_(id, cond, body),
                }
            }
            Viv::For { var } => {
                // The interpreter's loop (and the direct wasm backend's): the
                // bound evaluated once, before the first pass, and a counter
                // of its own that the variable is set from at the top of each
                // pass. So a body that changes the bound, or assigns the loop
                // variable, does not change the trip count, and after the
                // loop the variable holds the last pass's value (or is left
                // alone when there was no pass).
                //
                // The counter is a 64-bit integer (lower/intidx.rs). A bound
                // that is a number gives the pass count `viv_trip(bound)`,
                // the passes the interpreter's double counter makes (ceil of
                // the bound; none for 0, a negative bound or NaN); an Int
                // bound is the pass count itself.
                //
                //     {
                //         double viv_end_i = n;
                //         long long viv_n_i = viv_trip(viv_end_i);
                //         for (long long viv_k_i = 0; viv_k_i < viv_n_i; viv_k_i = (viv_k_i + 1)) {
                //             i = (double)viv_k_i;
                //             ...
                //
                // The names are block-scoped, so a loop nested in another over
                // the same variable shadows them.
                let int = LOOP_SCOPE.with(|s| s.borrow().vars.get(var) == Some(&Ty::Int));
                let bound = cx.expr(node.children[0]);
                let k = format!("viv_k_{var}");
                let end = format!("viv_end_{var}");
                let trip = format!("viv_n_{var}");
                let ll = || CType::Named("long long".to_string());
                let mut pre = Vec::new();
                let limit = match cx.kind(node.children[0]) {
                    Viv::Num(v) => CExpr::lit(id, format!("{}LL", trip_of(v))),
                    Viv::Int(v) => CExpr::lit(id, format!("{v}LL")),
                    _ if int => {
                        pre.push(CStmt::decl(id, ll(), trip.clone(), bound));
                        CExpr::var(id, trip.clone())
                    }
                    _ => {
                        cx.need("trip");
                        pre.push(CStmt::decl(id, CType::Double, end.clone(), bound));
                        pre.push(CStmt::decl(id, ll(), trip.clone(), CExpr::call(id, "viv_trip", vec![CExpr::var(id, end)])));
                        CExpr::var(id, trip.clone())
                    }
                };
                // the highest counter value the bound allows, when the facts know it
                // (an Int loop has no 2^53 stop; its counter is then known only
                // to be at least 0, which `hi` past 2^53 says)
                let cap = if int { 4.0 * intidx::EXACT } else { intidx::EXACT - 1.0 };
                let hi = match intidx::fact(cx.program(), node.children[0]).bounds() {
                    Some((_, h)) => (h.ceil() - 1.0).clamp(0.0, cap),
                    None => cap,
                };
                let unassigned = !intidx::assigns(cx.program(), node.children[1], var);
                // a literal bound has no trip variable; its guard is never needed
                let trip_name = if pre.is_empty() { String::new() } else { trip.clone() };
                let ub = intidx::upper(cx.program(), node.children[0]);
                // loop versioning (intidx::version_arrays): an innermost loop
                // that indexes arrays at its counter is emitted twice, a
                // version without those checks, run when one test before the
                // loop shows every such array covers every pass, and the loop
                // as it was
                let vers = if unassigned { intidx::version_arrays(cx.program(), var, node.children[1], &ub) } else { Vec::new() };
                let limit_c = |cx: &mut CEmitCx<Viv>| match cx.kind(node.children[0]) {
                    Viv::Num(v) => CExpr::lit(id, format!("{}LL", trip_of(v))),
                    Viv::Int(v) => CExpr::lit(id, format!("{v}LL")),
                    _ => CExpr::var(id, trip.clone()),
                };
                let emit = |cx: &mut CEmitCx<Viv>, limit: CExpr, assumed: Vec<String>| -> CStmt {
                    intidx::push_loop(intidx::LoopCx { var: var.clone(), counter: k.clone(), trip: trip_name.clone(), unassigned, hi, ub: ub.clone(), body: node.children[1], assumed });
                    let mut body = block(cx, node.children[1]);
                    intidx::pop_loop();
                    let init = CStmt::decl(id, ll(), k.clone(), CExpr::lit(id, "0"));
                    let cond = CExpr::binary(id, CExpr::var(id, k.clone()), "<", limit);
                    let step = CStmt::assign(id, k.clone(), CExpr::binary(id, CExpr::var(id, k.clone()), "+", CExpr::lit(id, "1")));
                    let keeps = loop_keeps(cx, node.children[1], None);
                    if let Some(keeps) = &keeps {
                        body.splice(0..0, release_stmts(id, keeps));
                    }
                    if poll_loop(cx, node.children[1]) {
                        body.insert(0, poll_stmt(id));
                    }
                    let counter = if int { CExpr::var(id, k.clone()) } else { CExpr::unary(id, "(double)", CExpr::var(id, k.clone())) };
                    body.insert(0, CStmt::assign(id, var.clone(), counter));
                    let lp = CStmt::for_(id, init, cond, step, body);
                    let lp = match keeps {
                        Some(keeps) => loop_with_release(id, lp, release_stmts(id, &keeps)),
                        None => lp,
                    };
                    lp
                };
                let lp = if vers.is_empty() {
                    emit(cx, limit, Vec::new())
                } else {
                    let mut test: Option<CExpr> = None;
                    for (_, base) in &vers {
                        let a = cx.expr(*base);
                        let r = CExpr::binary(id, CExpr::field(id, a, "rank"), "==", CExpr::lit(id, "1"));
                        let a = cx.expr(*base);
                        let n = CExpr::binary(id, CExpr::unary(id, "(long long)", CExpr::field(id, a, "dims[0]")), ">=", limit_c(cx));
                        let both = CExpr::binary(id, r, "&&", n);
                        test = Some(match test {
                            Some(t) => CExpr::binary(id, t, "&&", both),
                            None => both,
                        });
                    }
                    let checked = emit(cx, limit, Vec::new());
                    let lim = limit_c(cx);
                    let unchecked = emit(cx, lim, vers.iter().map(|(a, _)| a.clone()).collect());
                    CStmt::if_(id, test.expect("a versioned loop tests its arrays"), vec![unchecked], vec![checked])
                };
                if pre.is_empty() {
                    lp
                } else {
                    pre.push(lp);
                    CStmt::block(id, pre)
                }
            }
            Viv::ParallelFor { var, privates, reductions, pins, pin_what } => {
                // The loop knotkit writes counts with an integer,
                // `for (long knot_omp_i = 0; knot_omp_i < (long)(<bound>); ...)`,
                // which OpenMP requires. Two things make it the interpreter's
                // loop: the bound is read once, into `viv_end_i`, before the
                // loop (a build without OpenMP would re-read it each pass),
                // and it is rounded up, since `i < 2.5` runs the pass i = 2
                // and `(long)2.5` is 2. A whole number written as a number
                // is used as it is.
                //
                // A body that allocates frees what each pass made.
                // Under OpenMP a pass marks
                // the arena of the thread running it and releases back to the
                // mark when it ends, so no worker thread keeps a pass's
                // arrays (the threads' arenas are never released by the def's
                // return). Without OpenMP the loop is an ordinary loop and is
                // released as one (`loop_keeps`): the variables the body
                // assigns keep their last pass's values, as the interpreter's
                // do. A `continue` would skip the end of the pass, so a body
                // with one runs inside a one-pass loop it leaves instead.
                if !pins.is_empty() {
                    DJ_BODY.with(|d| d.set(d.get() + 1));
                    DJ_PINS.with(|p| p.borrow_mut().push(pins.clone()));
                }
                let mut body = block(cx, node.children[1]);
                if !pins.is_empty() {
                    DJ_BODY.with(|d| d.set(d.get() - 1));
                    DJ_PINS.with(|p| p.borrow_mut().pop());
                    // each pass says which it is, for the record of who wrote what
                    cx.need("disjoint");
                    body.insert(0, CStmt::expr(id, CExpr::call(id, "viv_dj_pass", vec![CExpr::var(id, var.clone())])));
                }
                let keeps = loop_keeps(cx, node.children[1], None);
                if let Some(keeps) = &keeps {
                    cx.need("par_arena");
                    if own_continue(cx, node.children[1]) {
                        let once = "viv_once".to_string();
                        let init = CStmt::decl(id, CType::Named("int".to_string()), once.clone(), CExpr::lit(id, "0"));
                        let cond = CExpr::binary(id, CExpr::var(id, once.clone()), "<", CExpr::lit(id, "1"));
                        let step = CStmt::assign(id, once.clone(), CExpr::lit(id, "1"));
                        body = vec![CStmt::for_(id, init, cond, step, body)];
                    }
                    let serial = release_stmts(id, keeps);
                    body.insert(0, CStmt::if_(id, CExpr::raw(id, "!VIV_OMP"), serial, vec![]));
                    body.insert(0, CStmt::decl(id, CType::Named("size_t".to_string()), "viv_pmark", CExpr::raw(id, "viv_arena_top")));
                    body.push(CStmt::if_(id, CExpr::raw(id, "VIV_OMP"), vec![CStmt::expr(id, CExpr::raw(id, "viv_arena_release(viv_pmark, (const void*)0)"))], vec![]));
                }
                let whole = match cx.kind(node.children[0]) {
                    Viv::Num(v) if v.fract() == 0.0 => true,
                    _ => false,
                };
                let (lp, decl) = if whole {
                    let bound = cx.expr(node.children[0]);
                    (CStmt::parallel_for(id, var.clone(), bound, body, privates.clone(), reductions.clone()), None)
                } else {
                    cx.need("libm");
                    let end = format!("viv_end_{var}");
                    let decl = CStmt::decl(id, CType::Double, end.clone(), cx.expr(node.children[0]));
                    let bound = CExpr::call(id, "ceil", vec![CExpr::var(id, end)]);
                    (CStmt::parallel_for(id, var.clone(), bound, body, privates.clone(), reductions.clone()), Some(decl))
                };
                let lp = match &keeps {
                    Some(keeps) => {
                        let after = vec![CStmt::if_(id, CExpr::raw(id, "!VIV_OMP"), release_stmts(id, keeps), vec![])];
                        loop_with_release(id, lp, after)
                    }
                    None => lp,
                };
                // a pinned loop: the record of which pass wrote each element
                // of the pinned arrays, kept while it runs
                let lp = if pins.is_empty() {
                    lp
                } else {
                    let mut stmts = vec![CStmt::expr(id, CExpr::call(id, "viv_dj_begin", vec![CExpr::raw(id, c_str(pin_what))]))];
                    for a in pins {
                        stmts.push(CStmt::expr(id, CExpr::call(id, "viv_dj_track", vec![CExpr::var(id, a.clone()), CExpr::raw(id, c_str(a))])));
                    }
                    stmts.push(lp);
                    stmts.push(CStmt::expr(id, CExpr::call(id, "viv_dj_end", vec![])));
                    CStmt::block(id, stmts)
                };
                match decl {
                    Some(decl) => CStmt::block(id, vec![decl, lp]),
                    None => lp,
                }
            }
            Viv::Break => CStmt::brk(id),
            Viv::Continue => CStmt::cont(id),
            // A `Block` reaching emit_stmt is a `test` body (a compound statement).
            Viv::Block => {
                let stmts = node.children.iter().map(|&c| cx.stmt(c)).collect();
                CStmt::block(id, stmts)
            }
            Viv::Show { label, is_arr } => {
                let value = cx.expr(node.children[0]);
                let label_lit = CExpr::raw(id, c_str(label));
                if *is_arr {
                    cx.need("show_arr");
                    CStmt::expr(id, CExpr::call(id, "viv_show_arr", vec![label_lit, value]))
                } else {
                    cx.need("show_num");
                    CStmt::expr(id, CExpr::call(id, "viv_show_num", vec![label_lit, value]))
                }
            }
            Viv::ShowCplx { label } => {
                let value = cx.expr(node.children[0]);
                let label_lit = CExpr::raw(id, c_str(label));
                cx.need("cplx");
                CStmt::expr(id, CExpr::call(id, "viv_show_cplx", vec![label_lit, value]))
            }
            Viv::ShowBool { label } => {
                let value = cx.expr(node.children[0]);
                let label_lit = CExpr::raw(id, c_str(label));
                cx.need("show_bool");
                CStmt::expr(id, CExpr::call(id, "viv_show_bool", vec![label_lit, value]))
            }
            Viv::Narrate { msg } => {
                cx.need("narrate");
                let msg_lit = CExpr::raw(id, c_str(msg));
                CStmt::expr(id, CExpr::call(id, "viv_narrate", vec![msg_lit]))
            }
            Viv::TestBegin { name } => {
                cx.need("assert_near");
                let name_lit = CExpr::raw(id, c_str(name));
                CStmt::expr(id, CExpr::call(id, "viv_test_begin", vec![name_lit]))
            }
            Viv::TestEnd => {
                cx.need("assert_near");
                CStmt::expr(id, CExpr::call(id, "viv_test_end", vec![]))
            }
            Viv::StepMark { enter, id: step, depth, story, def } => {
                cx.need("steps");
                let def = def.as_deref();
                if !*story {
                    return CStmt::expr(id, step_call(id, *enter, step, *depth, def));
                }
                cx.need("story");
                let def = def.map(str::to_string).unwrap_or_else(|| STEP_DEF.with(|d| d.borrow().clone()));
                let rec = CExpr::call(id, "viv_story_step", vec![
                    CExpr::raw(id, if *enter { "1" } else { "0" }),
                    CExpr::raw(id, c_str(&def)),
                    CExpr::raw(id, c_str(step)),
                    CExpr::raw(id, depth.to_string()),
                ]);
                CStmt::block(id, vec![CStmt::expr(id, step_call(id, *enter, step, *depth, Some(&def))), story_guard(id, rec)])
            }
            Viv::LensReturn { ty } => {
                // { T viv_sret = value; <ways out>; return viv_sret; }
                let mut stmts = vec![CStmt::decl(id, ty.to_c(), "viv_sret", cx.expr(node.children[0]))];
                for &c in &node.children[1..] {
                    stmts.push(cx.stmt(c));
                }
                stmts.push(CStmt::ret(id, CExpr::raw(id, "viv_sret")));
                CStmt::block(id, stmts)
            }
            Viv::StoryCall { def } => {
                cx.need("story");
                story_guard(id, story_mark(id, "call", def))
            }
            Viv::StoryRet { def } => {
                cx.need("story");
                story_guard(id, story_mark(id, "ret", def))
            }
            Viv::StoryIterEnd { def } => {
                cx.need("story");
                story_guard(id, story_mark(id, "iter_end", def))
            }
            Viv::StoryIter { def } => {
                cx.need("story");
                let n = cx.expr(node.children[0]);
                story_guard(id, CExpr::call(id, "viv_story_iter", vec![CExpr::raw(id, c_str(def)), n]))
            }
            Viv::StoryWatch { def, at, names } => {
                cx.need("story");
                let mut stmts = vec![CStmt::expr(id, CExpr::call(id, "viv_story_watch_begin", vec![CExpr::raw(id, c_str(def)), CExpr::raw(id, c_str(at))]))];
                for (k, (name, ty)) in names.iter().enumerate() {
                    let f = match ty {
                        Ty::Bool => "viv_story_w_bool",
                        Ty::Arr(_) => "viv_story_w_arr",
                        _ => "viv_story_w_num",
                    };
                    let v = cx.expr(node.children[k]);
                    stmts.push(CStmt::expr(id, CExpr::call(id, f, vec![CExpr::raw(id, c_str(name)), v])));
                }
                stmts.push(CStmt::expr(id, CExpr::call(id, "viv_story_watch_end", vec![])));
                CStmt::if_(id, CExpr::raw(id, "VIV_STORY"), stmts, Vec::new())
            }
            Viv::StoryHold { def, at, sentence, near } => {
                cx.need("story");
                let (d, a, sn) = (CExpr::raw(id, c_str(def)), CExpr::raw(id, c_str(at)), CExpr::raw(id, c_str(sentence)));
                let stmts = match near {
                    None => {
                        let ok = cx.expr(node.children[0]);
                        vec![CStmt::expr(id, CExpr::call(id, "viv_story_hold", vec![d, a, ok, sn, CExpr::raw(id, "0"), CExpr::raw(id, "0.0")]))]
                    }
                    Some(arr) => {
                        let diff = cx.expr(node.children[0]);
                        let tol = cx.expr(node.children[1]);
                        let worst = if *arr { CExpr::call(id, "viv_story_maxabs", vec![diff]) } else { CExpr::call(id, "fabs", vec![diff]) };
                        vec![
                            CStmt::decl(id, CType::Double, "viv_hw", worst),
                            CStmt::decl(id, CType::Double, "viv_ht", tol),
                            CStmt::expr(id, CExpr::call(id, "viv_story_hold", vec![d, a, CExpr::raw(id, "viv_hw <= viv_ht"), sn, CExpr::raw(id, "1"), CExpr::raw(id, "viv_hw")])),
                        ]
                    }
                };
                CStmt::if_(id, CExpr::raw(id, "VIV_STORY"), stmts, Vec::new())
            }
            Viv::StoryFlush => {
                cx.need("story");
                CStmt::expr(id, CExpr::call(id, "viv_story_flush", vec![]))
            }
            Viv::Placeholder { def } => {
                cx.need("placeholder");
                CStmt::expr(id, CExpr::call(id, "viv_placeholder", vec![CExpr::raw(id, c_str(def))]))
            }
            Viv::Stop { message, at, kind } => {
                cx.need("stop");
                let mut args = vec![CExpr::raw(id, kind.c_name()), CExpr::raw(id, c_str(message))];
                let func = with_site("viv_check_stop", *at, node.span, &mut args, id);
                CStmt::expr(id, CExpr::call(id, &func, args))
            }
            Viv::StepReturn { exits, ty } => {
                // { T viv_sret = value; <exits>; return viv_sret; } -- the value
                // first, since computing it can itself run steps.
                cx.need("steps");
                let mut stmts = vec![CStmt::decl(id, ty.to_c(), "viv_sret", cx.expr(node.children[0]))];
                for (def, step, depth) in exits {
                    stmts.push(CStmt::expr(id, step_call(id, false, step, *depth, def.as_deref())));
                }
                stmts.push(CStmt::ret(id, CExpr::raw(id, "viv_sret")));
                CStmt::block(id, stmts)
            }
            Viv::StepsFlush => {
                cx.need("steps");
                CStmt::expr(id, CExpr::call(id, "viv_steps_flush", vec![]))
            }
            _ => unreachable!("non-statement node in statement position: {:?}", node.kind),
        }
    }

fn emit_expr_c(node: &Node<Viv>, cx: &mut CEmitCx<Viv>) -> CExpr {
    if let Some(e) = fuse_emit_c(node, cx) {
        return e;
    }
        let id = node.id;
        match &node.kind {
            Viv::Print { parts } => {
                // one call, `viv_print(n, tag, payload, ...)`, so every value
                // is computed before the line is written (crate::printfmt);
                // with a string among them, `viv_prints`, whose tag 6 is one
                cx.need("print");
                let with_strings = parts.contains(&PrintPart::Str);
                if with_strings {
                    cx.need("str_print");
                }
                let mut args = vec![CExpr::lit(id, parts.len().to_string())];
                let mut k = 0;
                for p in parts {
                    let fmt_args = |f: &crate::printfmt::Fmt| {
                        vec![
                            CExpr::lit(id, f.word().to_string()),
                            CExpr::lit(id, f.prec.to_string()),
                            CExpr::lit(id, f.width.to_string()),
                        ]
                    };
                    match p {
                        PrintPart::Text(t) => {
                            args.push(CExpr::lit(id, "0"));
                            args.push(CExpr::raw(id, c_str(t)));
                            continue;
                        }
                        PrintPart::Num(f) => {
                            args.push(CExpr::lit(id, "1"));
                            args.push(cx.expr(node.children[k]));
                            args.extend(fmt_args(f));
                        }
                        PrintPart::Arr(f) => {
                            args.push(CExpr::lit(id, "2"));
                            args.push(cx.expr(node.children[k]));
                            args.extend(fmt_args(f));
                        }
                        PrintPart::Bool => {
                            args.push(CExpr::lit(id, "3"));
                            args.push(cx.expr(node.children[k]));
                        }
                        PrintPart::Cplx => {
                            args.push(CExpr::lit(id, "4"));
                            args.push(cx.expr(node.children[k]));
                        }
                        PrintPart::Str => {
                            args.push(CExpr::lit(id, "6"));
                            args.push(cx.expr(node.children[k]));
                        }
                        PrintPart::Map(kf, vf) | PrintPart::SMap(kf, vf) => {
                            args.push(CExpr::lit(id, if matches!(p, PrintPart::Map(..)) { "5" } else { "7" }));
                            args.push(cx.expr(node.children[k]));
                            args.extend(fmt_args(kf));
                            args.extend(fmt_args(vf));
                        }
                    }
                    k += 1;
                }
                // a line with a map of strings is written by the smaps
                // fragment's viv_print_x, which has every tag
                if parts.iter().any(|p| matches!(p, PrintPart::SMap(..))) {
                    cx.need("smaps");
                    return CExpr::call(id, "viv_print_x", args);
                }
                // a line with a map in it is written by the maps fragment's
                // viv_print_m, one with a string by viv_prints, one with both
                // by viv_print_sm, so every other print compiles as it did
                let with_maps = parts.iter().any(|p| matches!(p, PrintPart::Map(..)));
                if with_maps {
                    cx.need("maps");
                }
                let f = match (with_strings, with_maps) {
                    (true, true) => {
                        cx.need("str_map_print");
                        "viv_print_sm"
                    }
                    (true, false) => "viv_prints",
                    (false, true) => "viv_print_m",
                    (false, false) => "viv_print",
                };
                CExpr::call(id, f, args)
            }
            Viv::StrLit(text) => {
                cx.need("str_core");
                cx.need("str_lits");
                strings::c_lit(id, text)
            }
            Viv::Plot { label } => {
                cx.need("plot");
                let label_lit = CExpr::raw(id, c_str(label));
                let x = cx.expr(node.children[0]);
                let y = cx.expr(node.children[1]);
                CExpr::call(id, "viv_plot", vec![label_lit, x, y])
            }
            Viv::Bars { label } => {
                cx.need("bars");
                let label_lit = CExpr::raw(id, c_str(label));
                let x = cx.expr(node.children[0]);
                let y = cx.expr(node.children[1]);
                CExpr::call(id, "viv_bars", vec![label_lit, x, y])
            }
            Viv::Scatter { label, color } => {
                cx.need("scatter");
                let label_lit = CExpr::raw(id, c_str(label));
                let color_lit = CExpr::raw(id, c_str(color));
                let x = cx.expr(node.children[0]);
                let y = cx.expr(node.children[1]);
                CExpr::call(id, "viv_scatter", vec![label_lit, color_lit, x, y])
            }
            Viv::Plot2d { label } => {
                cx.need("plot2d");
                let label_lit = CExpr::raw(id, c_str(label));
                let z = cx.expr(node.children[0]);
                CExpr::call(id, "viv_plot2d", vec![label_lit, z])
            }
            Viv::Quiver { label } => {
                cx.need("quiver");
                let mut args = vec![CExpr::raw(id, c_str(label))];
                for &c in &node.children {
                    args.push(cx.expr(c));
                }
                CExpr::call(id, "viv_quiver", args)
            }
            Viv::Figure { title, xlabel, ylabel, style } => {
                cx.need("figure");
                let t = CExpr::raw(id, c_str(title));
                let xl = CExpr::raw(id, c_str(xlabel));
                let yl = CExpr::raw(id, c_str(ylabel));
                if node.children.len() == 4 {
                    let mut args = vec![t, xl, yl];
                    for &c in &node.children {
                        args.push(cx.expr(c));
                    }
                    args.push(CExpr::raw(id, c_str(style)));
                    CExpr::call(id, "viv_figure_win", args)
                } else {
                    CExpr::call(id, "viv_figure", vec![t, xl, yl])
                }
            }
            Viv::Imag(v) => {
                cx.need("cplx");
                // (0.0, v), as the interpreter has it: `v * I` is a real times a
                // complex, which gives the real part v * 0.0 (-0.0 for a
                // negative v, NaN for an infinite one)
                CExpr::raw(id, format!("viv_cmake(0.0, {})", fmt_f64(*v)))
            }
            // A Bool lives in a C double as exactly 1.0 or 0.0: what a comparison
            // already yields, what `if` already accepts, and what the uniform
            // all-doubles function ABI can carry.
            Viv::BoolLit(v) => CExpr::lit(id, if *v { "1.0" } else { "0.0" }),
            Viv::MapMath { cname } => {
                cx.need("arr_map");
                let f = CExpr::raw(id, cname.clone());
                let a = cx.expr(node.children[0]);
                CExpr::call(id, "viv_arr_map", vec![f, a])
            }
            Viv::Plot3d { label } => {
                cx.need("plot3d");
                let label_lit = CExpr::raw(id, c_str(label));
                let x = cx.expr(node.children[0]);
                let y = cx.expr(node.children[1]);
                let z = cx.expr(node.children[2]);
                CExpr::call(id, "viv_plot3d", vec![label_lit, x, y, z])
            }
            Viv::Anim { head, arrays } => {
                cx.need("anim");
                let na = *arrays as usize;
                let nn = node.children.len() - na;
                let verb = head.split(':').next().unwrap_or("");
                let full = if anim_data_verb(verb) { "1" } else { "0" };
                let mut call = vec![CExpr::raw(id, c_str(head)), CExpr::lit(id, full.to_string()), CExpr::lit(id, na.to_string())];
                for k in 0..3 {
                    call.push(if k < na { cx.expr(node.children[k]) } else { CExpr::raw(id, "(viv_arr){0}".to_string()) });
                }
                call.push(CExpr::lit(id, nn.to_string()));
                for k in 0..6 {
                    call.push(if k < nn { cx.expr(node.children[na + k]) } else { CExpr::lit(id, "0.0".to_string()) });
                }
                CExpr::call(id, "viv_anim", call)
            }
            Viv::Scene { title } => {
                cx.need("scene3d");
                CExpr::call(id, "viv_scene", vec![CExpr::raw(id, c_str(title))])
            }
            Viv::Wire { group } => {
                cx.need("scene3d");
                let mut args = vec![CExpr::raw(id, "\"wire\"".to_string()), CExpr::raw(id, c_str(group))];
                for &c in &node.children {
                    args.push(cx.expr(c));
                }
                let f = if node.children.len() == 4 { "viv_sc4" } else { "viv_sc3" };
                CExpr::call(id, f, args)
            }
            Viv::Verts { group } => {
                cx.need("scene3d");
                let mut args = vec![CExpr::raw(id, "\"verts\"".to_string()), CExpr::raw(id, c_str(group))];
                for &c in &node.children {
                    args.push(cx.expr(c));
                }
                let f = if node.children.len() == 4 { "viv_sc4" } else { "viv_sc3" };
                CExpr::call(id, f, args)
            }
            Viv::Num(v) => CExpr::lit(id, fmt_f64(*v)),
            // enums: the helpers `enums::c_decls` writes
            Viv::EnumNew { ty, tag } => {
                cx.need("enums");
                let args = node.children.iter().map(|&c| cx.expr(c)).collect();
                CExpr::call(id, enums::c_ctor(ty, *tag), args)
            }
            Viv::EnumIs { tag } => {
                cx.need("enums");
                let m = cx.expr(node.children[0]);
                CExpr::binary(id, CExpr::call(id, "viv_en_tag", vec![m]), "==", CExpr::lit(id, tag.to_string()))
            }
            Viv::EnumField { ty, tag, k } => {
                let m = cx.expr(node.children[0]);
                CExpr::call(id, format!("viv_get_{}_{tag}_{k}", enums::c_name(ty)), vec![m])
            }
            Viv::EnumEq { ne } => {
                cx.need("enums");
                let a = cx.expr(node.children[0]);
                let b = cx.expr(node.children[1]);
                CExpr::binary(id, CExpr::call(id, "viv_en_tag", vec![a]), if *ne { "!=" } else { "==" }, CExpr::call(id, "viv_en_tag", vec![b]))
            }
            Viv::EnumStr { ty } => {
                cx.need("enums");
                let e = cx.expr(node.children[0]);
                CExpr::call(id, format!("viv_en_str_{}", enums::c_name(ty)), vec![e])
            }
            // C has no literal for -2^63: it is the expression below
            Viv::Int(v) if *v == i64::MIN => CExpr::lit(id, "(-9223372036854775807LL - 1)"),
            Viv::Int(v) if *v < 0 => CExpr::lit(id, format!("({v}LL)")),
            Viv::Int(v) => CExpr::lit(id, format!("{v}LL")),
            Viv::Var { name } => CExpr::var(id, name.clone()),
            Viv::ArrLit { dims } => {
                if let Some(table) = const_table(node, cx, dims.len()) {
                    // (const double[]){e0, e1, ...}: see the "vec_tab" fragment for why.
                    let (func, frag) = if dims.len() == 1 { ("viv_vec_tab", "vec_tab") } else { ("viv_mat_tab", "mat_tab") };
                    cx.need(frag);
                    let mut args: Vec<CExpr> = dims.iter().map(|d| CExpr::lit(id, d.to_string())).collect();
                    args.push(CExpr::raw(id, table));
                    return CExpr::call(id, func, args);
                }
                let elems: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                match dims.as_slice() {
                    [n] => {
                        cx.need("vec");
                        let mut args = vec![CExpr::lit(id, n.to_string())];
                        args.extend(elems);
                        CExpr::call(id, "viv_vec", args)
                    }
                    [r, c] => {
                        cx.need("mat");
                        let mut args = vec![CExpr::lit(id, r.to_string()), CExpr::lit(id, c.to_string())];
                        args.extend(elems);
                        CExpr::call(id, "viv_mat", args)
                    }
                    _ => unreachable!("unsupported literal rank {}", dims.len()),
                }
            }
            Viv::Zeros => {
                let args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                let func = match args.len() {
                    1 => "viv_zeros1",
                    2 => "viv_zeros2",
                    3 => "viv_zeros3",
                    4 => "viv_zeros4",
                    n => unreachable!("unsupported zeros rank {n}"),
                };
                cx.need(match args.len() { 1 => "zeros1", 2 => "zeros2", 3 => "zeros3", 4 => "zeros4", _ => unreachable!() });
                CExpr::call(id, func, args)
            }
            Viv::CZeros => {
                let args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                let (func, frag) = match args.len() {
                    1 => ("viv_czeros1", "czeros1"),
                    2 => ("viv_czeros2", "czeros2"),
                    3 => ("viv_czeros3", "czeros3"),
                    4 => ("viv_czeros4", "czeros4"),
                    n => unreachable!("unsupported czeros rank {n}"),
                };
                cx.need(frag);
                CExpr::call(id, func, args)
            }
            Viv::CIndex { at } => {
                let nidx = node.children.len() - 1;
                let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                // Int indices stay 64-bit integers: through a double and back
                // they cost NPB FT's radix-2 pass a third of its time
                let int_ix = nidx <= 3 && node.children[1..].iter().all(|&c| intidx::is_int_node(cx.program(), c));
                let (func, frag) = match (nidx, int_ix) {
                    (1, true) => ("viv_lcget1", "lcget1"),
                    (2, true) => ("viv_lcget2", "lcget2"),
                    (3, true) => ("viv_lcget3", "lcget3"),
                    (1, _) => ("viv_cget1", "cget1"),
                    (2, _) => ("viv_cget2", "cget2"),
                    (3, _) => ("viv_cget3", "cget3"),
                    (4, _) => ("viv_cget4", "cget4"),
                    _ => unreachable!("unsupported complex index rank {nidx}"),
                };
                cx.need(frag);
                let func = with_site(func, *at, node.span, &mut args, id);
                CExpr::call(id, &func, args)
            }
            Viv::CArrLit { n } => {
                // viv_cvec(n, e1, e2, ...), each element promoted to complex so
                // the variadic read on the other side is unambiguous.
                cx.need("cvec");
                let mut args = vec![CExpr::raw(id, n.to_string())];
                for &c in &node.children {
                    let e = cx.expr(c);
                    args.push(CExpr::call(id, "viv_cpromote", vec![e]));
                }
                CExpr::call(id, "viv_cvec", args)
            }
            Viv::CArrFn { name } => {
                cx.need("carr_fn");
                let a = cx.expr(node.children[0]);
                CExpr::call(id, &format!("viv_carr_{name}"), vec![a])
            }
            Viv::Fft { inverse } => {
                cx.need("fft");
                let a = cx.expr(node.children[0]);
                let flag = CExpr::raw(id, if *inverse { "1.0" } else { "0.0" }.to_string());
                CExpr::call(id, "viv_fft", vec![a, flag])
            }
            Viv::Index { at } => {
                let n = node.children.len();
                let nidx = n - 1;
                // the recursion counter (crate::recursion): a plain C global
                if is_depth_cell(cx.program(), node.children[0], &node.children[1..]) {
                    cx.need("depth");
                    return CExpr::unary(id, "(double)", CExpr::var(id, "viv_depth_g"));
                }
                // every index an exact integer: the integer accessor, behind
                // its guards (lower/intidx.rs)
                if let Some(fast) = int_access(cx, node, &node.children[1..], None, *at) {
                    return fast;
                }
                intidx::note(&loc_text(node.span), nidx, "double", 0);
                let base = cx.expr(node.children[0]);
                let mut args = vec![base];
                for &c in &node.children[1..] {
                    args.push(cx.expr(c));
                }
                let func = match nidx {
                    1 => "viv_get1",
                    2 => "viv_get2",
                    3 => "viv_get3",
                    4 => "viv_get4",
                    _ => unreachable!("unsupported index rank {nidx}"),
                };
                cx.need(match nidx { 1 => "get1", 2 => "get2", 3 => "get3", 4 => "get4", _ => unreachable!() });
                let func = with_site(func, *at, node.span, &mut args, id);
                CExpr::call(id, &func, args)
            }
            Viv::Neg => CExpr::unary(id, "-", cx.expr(node.children[0])),
            Viv::Not => CExpr::unary(id, "!", cx.expr(node.children[0])),
            Viv::Bin { op, cplx } => {
                // a comparison of two exact integers (loop counters, say)
                // compares them as integers (lower/intidx.rs)
                if !*cplx && matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::EqEq | BinOp::Ne) {
                    if let Some(c) = intidx::int_compare(cx, id, op.c_op(), node.children[0], node.children[1]) {
                        return c;
                    }
                }
                // `and` of comparisons that cannot fail: both sides, no branch
                let op_c = if !*cplx && *op == BinOp::And && intidx::branch_free_and(cx.program(), id) { "&" } else { op.c_op() };
                let lhs = cx.expr(node.children[0]);
                let rhs = cx.expr(node.children[1]);
                if *cplx {
                    cx.need("cplx");
                    let f = match op {
                        BinOp::Add => "viv_cadd",
                        BinOp::Sub => "viv_csub",
                        BinOp::Mul => "viv_cmul",
                        _ => "viv_cdiv",
                    };
                    return CExpr::call(id, f, vec![lhs, rhs]);
                }
                CExpr::binary(id, lhs, op_c, rhs)
            }
            Viv::Arr { kind } => {
                cx.need(kind.frag());
                let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                // Two arrays: their shapes are checked first, and a mismatch stops
                // with the interpreter's words and location (viv_arr_chk).
                if let Some(how) = kind.shape_check() {
                    cx.need("arr_chk");
                    args.insert(0, CExpr::raw(id, kind.runtime().to_string()));
                    args.push(CExpr::raw(id, how.to_string()));
                    args.push(CExpr::raw(id, site_row(node.span).to_string()));
                    return CExpr::call(id, "viv_arr_chk", args);
                }
                CExpr::call(id, kind.runtime(), args)
            }
            Viv::Closure { name, captures } => {
                cx.need("clos1");
                let caps = if captures.is_empty() {
                    "(const double*)0".to_string()
                } else {
                    format!("(double[]){{ {} }}", captures.join(", "))
                };
                CExpr::raw(id, format!("(viv_clos1){{ {name}, {caps} }}"))
            }
            Viv::FnRef { name, .. } => {
                cx.need("viv_fn");
                CExpr::raw(id, format!("(viv_fn){{ viv_wrap_{name} }}"))
            }
            Viv::FnArrLit { n, .. } => {
                // viv_fnvec(n, f1, f2, ...), the same shape as viv_vec and
                // viv_cvec. A viv_fn is a struct, and a struct is not subject to
                // the default argument promotions, so va_arg reads back exactly
                // the type that went in.
                cx.need("viv_fnarr");
                let mut args = vec![CExpr::raw(id, n.to_string())];
                for &c in &node.children {
                    args.push(cx.expr(c));
                }
                CExpr::call(id, "viv_fnvec", args)
            }
            Viv::FnIndex { at, .. } => {
                cx.need("viv_fnarr");
                let a = cx.expr(node.children[0]);
                let i = cx.expr(node.children[1]);
                let mut args = vec![a, i];
                let func = with_site("viv_fnget", *at, node.span, &mut args, id);
                CExpr::call(id, &func, args)
            }
            Viv::FnLen => {
                cx.need("viv_fnarr");
                let a = cx.expr(node.children[0]);
                CExpr::call(id, "viv_fnlen", vec![a])
            }
            // arrays of records (the `recs` fragment)
            Viv::RecArrLit { rec } => {
                cx.need("recs");
                let size = CExpr::raw(id, format!("sizeof(viv_rec_{rec})"));
                if node.children.is_empty() {
                    return CExpr::call(id, "viv_recs_new", vec![CExpr::raw(id, "0"), size]);
                }
                let _ = size;
                let mut args = vec![CExpr::raw(id, node.children.len().to_string())];
                args.extend(node.children.iter().map(|&c| cx.expr(c)));
                CExpr::call(id, format!("viv_recs_lit_{rec}"), args)
            }
            Viv::RecIndex { rec, at } => {
                cx.need("recs");
                rec_elem(cx, id, rec, node.children[0], node.children[1], *at, node.span)
            }
            Viv::RecLen => {
                cx.need("recs");
                let a = cx.expr(node.children[0]);
                CExpr::call(id, "viv_recs_len", vec![a])
            }
            Viv::RecAppend { rec } => {
                cx.need("recs");
                let a = cx.expr(node.children[0]);
                let r = cx.expr(node.children[1]);
                CExpr::call(id, format!("viv_recs_push_{rec}"), vec![a, r])
            }
            // a C struct is already copied by value; only an array of records
            // in it has storage that must be copied too
            Viv::Own { ty } => {
                let v = cx.expr(node.children[0]);
                let records = record_registry(cx.program());
                match ty {
                    Ty::Record(r) if holds_recs(*ty, &records) => CExpr::call(id, format!("viv_own_{r}"), vec![v]),
                    Ty::Recs(r) => CExpr::call(id, format!("viv_own_recs_{r}"), vec![v]),
                    _ => v,
                }
            }
            Viv::ApplyFn { sig } => {
                // viv_apply(f, n, args...): a variadic runtime helper collects
                // the arguments into an array and calls f through the uniform
                // ABI. Each argument is boxed by its kind -- the only place the
                // union is built -- and the result unboxed the same way.
                cx.need("viv_fn");
                let n = node.children.len() - 1;
                let mut args = vec![cx.expr(node.children[0])];
                args.push(CExpr::raw(id, n.to_string()));
                for (j, &c) in node.children[1..].iter().enumerate() {
                    let e = cx.expr(c);
                    let box_fn = match sig.kind(j) {
                        ArgKind::Num => "viv_argn",
                        ArgKind::Fn => "viv_argf",
                        ArgKind::Arr => "viv_arga",
                    };
                    args.push(CExpr::call(id, box_fn, vec![e]));
                }
                let call = CExpr::call(id, "viv_apply", args);
                CExpr::field(id, call, sig.ret.member())
            }
            Viv::Call { callee } => {
                // strings: the runtime's viv_s_* and viv_ss_*
                if let Some(e) = strings::emit_call(node, callee, cx) {
                    return e;
                }
                // Ints: the runtime's viv_i_*
                if let Some(e) = ints::emit_call(node, callee, cx) {
                    return e;
                }
                // Float32 and the packed arrays
                if let Some(e) = f32s::emit_call(node, callee, cx) {
                    return e;
                }
                match callee.as_str() {
                    PRINT_RUNTIME => cx.need("print_num"),
                    "viv_text" => cx.need("text"),
                    "exp" | "sqrt" | "pow" | "sin" | "cos" | "tan" | "log" | "fabs" | "floor" | "ceil" | "fmin" | "fmax"
                    | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh" | "erf" | "tgamma" | "round" | "fmod" => cx.need("libm"),
                    "fma" => cx.need("fma"),
                    "viv_simpson" => cx.need("simpson"),
                    "viv_trapezoid" => cx.need("trapezoid"),
                    // a solver that stops on a NaN or an infinity: its viv_sites row goes last
                    "viv_eig_sym" | "viv_spectrum_sym" => {
                        cx.need(if callee == "viv_eig_sym" { "eig_sym" } else { "spectrum_sym" });
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, format!("{callee}_at"), args);
                    }
                    "viv_eig_herm" => cx.need("eig_herm"),
                    "viv_fft3" | "viv_ifft3" => cx.need("fft3"),
                    "viv_sum" => cx.need("sum"),
                    "viv_band" | "viv_bor" | "viv_bxor" | "viv_bnot" | "viv_shl" | "viv_shr" | "viv_sar" | "viv_u32" | "viv_i32" => cx.need("bits"),
                    "viv_histogram" => cx.need("histogram"),
                    "viv_maxval" | "viv_minval" => cx.need("arr_extent"),
                    "viv_carr_from" => cx.need("carr_from"),
                    "viv_cfill" => cx.need("cfill"),
                    "viv_slab" | "viv_setslab" => cx.need("slab"),
                    "viv_take" | "viv_put" => cx.need("take_put"),
                    // a solver that can stop: its viv_sites row goes last (viv_solver_fail)
                    "viv_lin_solve" | "viv_lin_solve_multi" => {
                        cx.need(if callee == "viv_lin_solve" { "lin_solve" } else { "lin_solve_multi" });
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, format!("{callee}_at"), args);
                    }
                    "viv_cholesky" => cx.need("cholesky"),
                    "viv_eig_general" => cx.need("eig_general"),
                    "viv_eig_near" => cx.need("eig_near"),
                    "viv_fill" => cx.need("fill"),
                    "viv_cplx" | "creal" | "cimag" | "conj" | "cexp" | "csqrt" | "clog" | "csin" | "ccos" | "ctan" | "cabs" => cx.need("cplx"),
                    "viv_assert_near" | "viv_assert_true" | "viv_assert_gt" | "viv_assert_lt" => cx.need("assert_near"),
                    // viv_report lives in the assert_near fragment
                    "viv_assert_allclose" => { cx.need("assert_near"); cx.need("assert_allclose"); }
                    "viv_srand" | "viv_rand" => cx.need("rng"),
                    "viv_release_mode" => cx.need("release_mode"),
                    "viv_clock" => cx.need("clock"),
                    "viv_export_given" | "viv_export_num" | "viv_export_int" | "viv_export_bool" => cx.need("export"),
                    "viv_export_str" => cx.need("export_str"),
                    "viv_export_arr" => cx.need("export_arr"),
                    "viv_sort" => cx.need("sort"),
                    "viv_tridiag_solve" => cx.need("tridiag_solve"),
                    "viv_roll" => cx.need("roll"),
                    "viv_len" => cx.need("len"),
                    "viv_numel" => cx.need("numel"),
                    "viv_shape" => cx.need("shape"),
                    "viv_all" | "viv_any" => cx.need("all_any"),
                    "viv_where_aa" | "viv_where_as" | "viv_where_sa" | "viv_where_ss" => cx.need("where"),
                    "viv_ad_unb_aa" => cx.need("ad_unb"),
                    "viv_ad_keep" | "viv_ad_like" | "viv_ad_prod" | "viv_ad_prodax" | "viv_ad_takeax" => cx.need("ad_arr"),
                    c if c.starts_with("viv_ad_") => cx.need("ad"),
                    // a slice or a join can stop: its viv_sites row goes last
                    "viv_getslice" | "viv_getslice_v" | "viv_setslice" | "viv_setslice_n" | "viv_setslice_c" | "viv_cat" | "viv_cat1" => {
                        cx.need(if callee == "viv_setslice_c" { "slices_c" } else { "slices" });
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        let callee = if callee == crate::fuse::VIEW_C { "viv_getslice" } else { callee.as_str() };
                        return CExpr::call(id, callee, args);
                    }
                    "viv_copy" | "viv_ad_cat" | "viv_ad_extent" => cx.need("slices"),
                    "viv_slicecheck" => {
                        cx.need("fuse");
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, callee.clone(), args);
                    }
                    // the array vocabulary (interp/arrays.rs): its fragment, a
                    // string argument as a C string, and the viv_sites row last
                    c if crate::interp::arrays::of_c_name(c).is_some() => {
                        let (sig, _) = crate::interp::arrays::of_c_name(c).expect("checked");
                        cx.need(sig.frag);
                        let mut args: Vec<CExpr> = Vec::with_capacity(node.children.len() + 1);
                        for &k in &node.children {
                            args.push(match cx.kind(k) {
                                Viv::StrLit(text) => CExpr::raw(id, format!("{text:?}")),
                                _ => cx.expr(k),
                            });
                        }
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, callee.clone(), args);
                    }
                    // a map of strings (maps::kinded_fns): its read, store and
                    // remove can stop, and take the viv_sites row last
                    c if crate::interp::maps::kinded(c).is_some() => {
                        cx.need("smaps");
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        if crate::interp::maps::kinded(c).is_some_and(|f| f.stops) {
                            args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        }
                        return CExpr::call(id, callee.clone(), args);
                    }
                    // a map read, store or remove can stop: its viv_sites row goes last
                    "viv_map_get" | "viv_map_set" | "viv_map_remove" => {
                        cx.need("maps");
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, callee.clone(), args);
                    }
                    c if c.starts_with("viv_map_") => cx.need("maps"),
                    // an append can stop: its viv_sites row goes last
                    "viv_append_n" | "viv_append_r" | "viv_extend" => {
                        cx.need("growable");
                        let mut args: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, callee.clone(), args);
                    }
                    "viv_seal" | "viv_grow_head" | "viv_grow_tail_n" | "viv_grow_tail_a" => cx.need("growable"),
                    "viv_floormod" | "viv_floordiv" => {
                        // the operands, then the viv_sites row of the operator (0,
                        // no location, when compiled without sites); the third
                        // child only keeps the node apart
                        cx.need("floorops");
                        let mut args: Vec<CExpr> = node.children[..2].iter().map(|&c| cx.expr(c)).collect();
                        args.push(CExpr::raw(id, site_row(node.span).to_string()));
                        return CExpr::call(id, callee.clone(), args);
                    }
                    _ => {}
                }
                let args = node.children.iter().map(|&c| cx.expr(c)).collect();
                CExpr::call(id, callee.clone(), args)
            }
            Viv::RecordLit { ty_name, fields } => {
                // Children are field values in `fields` order; emit a designated
                // compound literal `((viv_rec_T){ .f = v, ... })`.
                let values: Vec<CExpr> = node.children.iter().map(|&c| cx.expr(c)).collect();
                let pairs = fields.iter().cloned().zip(values).collect();
                CExpr::struct_lit(id, format!("viv_rec_{ty_name}"), pairs)
            }
            Viv::Field { field } => CExpr::field(id, cx.expr(node.children[0]), field.clone()),
            // a checkpoint() used in expression position evaluates to 0.0 (the
            // marker semantics live in the Main emitter)
            Viv::Ckpt => CExpr::lit(id, "0.0"),
            _ => unreachable!("non-expression node in expression position: {:?}", node.kind),
        }
}

/// Rewrite every `return e;` under `s` into
/// `{ T viv_ret = e; viv_arena_release(viv_mark, keep); return viv_ret; }`.
/// The value is computed FIRST: `e` may read arrays the release would free.
/// A record is kept whole: every array it holds is kept, the rest freed.
fn arena_returns(s: CStmt, id: NodeId, ret: Ty) -> CStmt {
    let origin = s.origin;
    let each = |v: Vec<CStmt>| -> Vec<CStmt> { v.into_iter().map(|c| arena_returns(c, id, ret)).collect() };
    let kind = match s.kind {
        CStmtKind::Return(e) if matches!(ret, Ty::Record(_) | Ty::Recs(_) | Ty::Enum(_)) => {
            let records = LOOP_SCOPE.with(|s| s.borrow().records.clone());
            let mut stmts = vec![CStmt::decl(id, ret.to_c(), "viv_ret", e)];
            keep_calls(id, "viv_ret", &mut Vec::new(), ret, &records, "viv_mark", &mut stmts);
            stmts.push(CStmt::expr(id, CExpr::raw(id, "viv_arena_release_kept(viv_mark)")));
            stmts.push(CStmt::ret(id, CExpr::raw(id, "viv_ret")));
            CStmtKind::Block(stmts)
        }
        CStmtKind::Return(e) => {
            // a string keeps the block it lives in, a list its block and chunks
            let release = match ret {
                Ty::Str => "viv_arena_release(viv_mark, viv_ret ? viv_ret->owner : (void*)0)".to_string(),
                Ty::Arr(Elem::Str) => "viv_keep_strs(viv_ret), viv_arena_release_kept(viv_mark)".to_string(),
                Ty::Arr(_) => "viv_arena_release(viv_mark, viv_ret.data)".to_string(),
                // a map is filed under its tagged pointer (interp/maps.rs)
                Ty::Map(..) if settle(ret) == NUM_MAP => "viv_arena_release(viv_mark, VIV_MAP_KEY(viv_ret))".to_string(),
                Ty::Map(..) => "viv_arena_release(viv_mark, VIV_SMAP_KEY(viv_ret))".to_string(),
                _ => "viv_arena_release(viv_mark, (const double*)0)".to_string(),
            };
            CStmtKind::Block(vec![
                CStmt::decl(id, ret.to_c(), "viv_ret", e),
                CStmt::expr(id, CExpr::raw(id, release)),
                CStmt::ret(id, CExpr::raw(id, "viv_ret")),
            ])
        }
        CStmtKind::If { cond, then_body, else_body } => CStmtKind::If { cond, then_body: each(then_body), else_body: each(else_body) },
        CStmtKind::While { cond, body } => CStmtKind::While { cond, body: each(body) },
        CStmtKind::For { init, cond, step, body } => CStmtKind::For { init, cond, step, body: each(body) },
        CStmtKind::Block(body) => CStmtKind::Block(each(body)),
        CStmtKind::ParallelFor { var, bound, body, privates, reductions } => {
            CStmtKind::ParallelFor { var, bound, body: each(body), privates, reductions }
        }
        other => other,
    };
    CStmt { origin, kind }
}

/// `if (VIV_STORY) <call>;`: a story call, compiled out under -DVIV_RELEASE.
fn story_guard(id: NodeId, call: CExpr) -> CStmt {
    CStmt::if_(id, CExpr::raw(id, "VIV_STORY"), vec![CStmt::expr(id, call)], Vec::new())
}

/// `viv_story_mark("<kind>", "<def>")`.
fn story_mark(id: NodeId, kind: &str, def: &str) -> CExpr {
    CExpr::call(id, "viv_story_mark", vec![CExpr::raw(id, c_str(kind)), CExpr::raw(id, c_str(def))])
}

/// Rewrite every `return e;` under `s` into
/// `{ T viv_cret = e; if (VIV_STORY) viv_story_mark("ret", def); return viv_cret; }`.
fn story_returns(s: CStmt, id: NodeId, ret: Ty, def: &str) -> CStmt {
    let origin = s.origin;
    let each = |v: Vec<CStmt>| -> Vec<CStmt> { v.into_iter().map(|c| story_returns(c, id, ret, def)).collect() };
    let kind = match s.kind {
        CStmtKind::Return(e) => CStmtKind::Block(vec![
            CStmt::decl(id, ret.to_c(), "viv_cret", e),
            story_guard(id, story_mark(id, "ret", def)),
            CStmt::ret(id, CExpr::raw(id, "viv_cret")),
        ]),
        CStmtKind::If { cond, then_body, else_body } => CStmtKind::If { cond, then_body: each(then_body), else_body: each(else_body) },
        CStmtKind::While { cond, body } => CStmtKind::While { cond, body: each(body) },
        CStmtKind::For { init, cond, step, body } => CStmtKind::For { init, cond, step, body: each(body) },
        CStmtKind::Block(body) => CStmtKind::Block(each(body)),
        CStmtKind::ParallelFor { var, bound, body, privates, reductions } => {
            CStmtKind::ParallelFor { var, bound, body: each(body), privates, reductions }
        }
        other => other,
    };
    CStmt { origin, kind }
}

/// `viv_step(enter, "<def>", "<id>", depth)`: `def` is the def the step was
/// written in, or None for the def being emitted.
fn step_call(id: NodeId, enter: bool, step: &str, depth: u32, def: Option<&str>) -> CExpr {
    let def = def.map(str::to_string).unwrap_or_else(|| STEP_DEF.with(|d| d.borrow().clone()));
    CExpr::call(id, "viv_step", vec![
        CExpr::raw(id, if enter { "1" } else { "0" }),
        CExpr::raw(id, c_str(&def)),
        CExpr::raw(id, c_str(step)),
        CExpr::raw(id, depth.to_string()),
    ])
}

/// `a[i, j]` (or `a[i, j] = v` with `value`) through the integer accessor
/// `viv_lgetN` / `viv_lsetN` when every index has an exact integer form
/// (lower/intidx.rs). An index that needs a runtime fact (a pass count, a
/// magnitude) is guarded, and the access is then
/// `guard ? viv_lget2_at(a, i, j) : viv_get2_at(a, x, y)`: the unchanged
/// double path, with its checks and messages, whenever a guard fails.
fn int_access(cx: &mut CEmitCx<Viv>, node: &Node<Viv>, idx: &[NodeId], value: Option<NodeId>, at: u32) -> Option<CExpr> {
    let id = node.id;
    let r = idx.len();
    if !(1..=4).contains(&r) {
        return None;
    }
    let mut forms = Vec::new();
    for &i in idx {
        forms.push(intidx::ready(cx, i)?);
    }
    let guards: Vec<intidx::Guard> = forms.iter().flat_map(|f| f.guards.iter().cloned()).collect();
    // the dimensions an index is proved inside: 0 <= index, and
    // index < sym + off with off <= 0, sym the array's fixed dimension
    let mut mask = 0u32;
    for (d, f) in forms.iter().enumerate() {
        if r == 1 && intidx::assumed(cx.program(), node.children[0], idx[0]) {
            mask |= 1;
            continue;
        }
        if f.lo >= 0.0 {
            if let (Some((sym, off)), Some(dim)) = (&f.ub, intidx::base_dim(cx.program(), node.children[0], d)) {
                if *sym == dim && *off <= 0 {
                    mask |= 1 << d;
                }
            }
        }
    }
    intidx::note(&loc_text(node.span), r, if guards.is_empty() { "int" } else { "guarded" }, mask);
    let set = value.is_some();
    let mut fast = vec![cx.expr(node.children[0])];
    fast.extend(forms.into_iter().map(|f| f.c));
    if let Some(v) = value {
        fast.push(cx.expr(v));
    }
    fast.push(CExpr::lit(id, mask.to_string()));
    let name = format!("viv_l{}{r}", if set { "set" } else { "get" });
    cx.need(match (set, r) {
        (false, 1) => "lget1",
        (false, 2) => "lget2",
        (false, 3) => "lget3",
        (false, _) => "lget4",
        (true, 1) => "lset1",
        (true, 2) => "lset2",
        (true, 3) => "lset3",
        (true, _) => "lset4",
    });
    let func = with_site(&name, at, node.span, &mut fast, id);
    let fast = CExpr::call(id, &func, fast);
    let Some(test) = intidx::guard_c(id, &guards) else { return Some(fast) };
    // the double path, as it was
    let mut slow = vec![cx.expr(node.children[0])];
    for &i in idx {
        slow.push(cx.expr(i));
    }
    if let Some(v) = value {
        slow.push(cx.expr(v));
    }
    let name = format!("viv_{}{r}", if set { "set" } else { "get" });
    cx.need(match (set, r) {
        (false, 1) => "get1",
        (false, 2) => "get2",
        (false, 3) => "get3",
        (false, _) => "get4",
        (true, 1) => "set1",
        (true, 2) => "set2",
        (true, 3) => "set3",
        (true, _) => "set4",
    });
    cx.need("libm");
    cx.need("trip");
    let func = with_site(&name, at, node.span, &mut slow, id);
    Some(CExpr::ternary(id, test, fast, CExpr::call(id, &func, slow)))
}

/// `viv__depth[0]`, the recursion counter crate::recursion writes into every
/// recursive def: in C it is the global `viv_depth_g`, a double, so a call
/// counts its depth with no array access. The array itself is still made
/// (the program's first statement); nothing else reads it.
pub(crate) fn is_depth_cell(prog: &IrProgram<Viv>, base: NodeId, idx: &[NodeId]) -> bool {
    matches!(&prog.get(base).kind, Viv::Var { name } if *name == format!("{}depth", crate::parser::HIDDEN_PREFIX))
        && idx.len() == 1
        && matches!(prog.get(idx[0]).kind, Viv::Num(x) if x == 0.0)
}

/// `viv__depth[0] + 1` (+1) or `viv__depth[0] - 1` (-1): the two writes of the
/// recursion counter; 0 for any other value.
pub(crate) fn depth_step(prog: &IrProgram<Viv>, value: NodeId) -> i32 {
    let v = prog.get(value);
    let reads_cell = |c: NodeId| {
        let r = prog.get(c);
        matches!(r.kind, Viv::Index { .. }) && is_depth_cell(prog, r.children[0], &r.children[1..])
    };
    match &v.kind {
        Viv::Bin { op, cplx: false } if reads_cell(v.children[0]) && matches!(prog.get(v.children[1]).kind, Viv::Num(x) if x == 1.0) => match op {
            BinOp::Add => 1,
            BinOp::Sub => -1,
            _ => 0,
        },
        _ => 0,
    }
}

/// `file:line:col` of a span while sites are on (the access report).
fn loc_text(span: Span) -> String {
    if std::env::var_os("VIV_INDEX_REPORT").is_none() {
        return String::new();
    }
    let row = site_row(span);
    SITES.with(|c| {
        let c = c.borrow();
        let Some(sites) = c.as_ref() else { return String::new() };
        let text = sites.rows.get(row).cloned().unwrap_or_default();
        text.lines().find_map(|l| l.trim().strip_prefix("--> ").map(|x| x.to_string())).unwrap_or_default()
    })
}

/// The passes of `for i to v` for a bound written as a number: the
/// interpreter's double counter runs while it is below v, so ceil(v) passes,
/// none for 0, a negative v or NaN. A count past 2^53 is cut there: the
/// double counter stops advancing at 2^53 (`viv_trip`).
fn trip_of(v: f64) -> i64 {
    if v > 0.0 {
        v.ceil().min(intidx::EXACT) as i64
    } else {
        0
    }
}

fn block(cx: &mut CEmitCx<Viv>, block_id: NodeId) -> Vec<CStmt> {
    cx.children(block_id).into_iter().map(|c| cx.stmt(c)).collect()
}

fn fmt_f64(v: f64) -> String {
    // a string builtin's "not given" default is a NaN (interp/strings.rs); C
    // spells the special values with math.h's macros
    if v.is_nan() {
        "NAN".to_string()
    } else if v.is_infinite() {
        (if v > 0.0 { "HUGE_VAL" } else { "(-HUGE_VAL)" }).to_string()
    } else {
        format!("{:?}", v)
    }
}

/// Names the top level binds, by assignment or as a `for` variable, at any depth
/// of top-level control flow and inside `test` blocks: the globals a def can read.
fn top_level_names(program: &Program) -> std::collections::HashSet<String> {
    fn walk(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::Assign { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::For { var, body, .. } => {
                    out.insert(var.clone());
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
    let mut out = std::collections::HashSet::new();
    for item in &program.items {
        if let Item::Stmt(s) = item {
            walk(std::slice::from_ref(s), &mut out);
        }
    }
    out
}

/// A top-level variable with the name of a def. The interpreter keeps defs
/// and variables apart, but the C has one namespace for both, so the compile
/// used to fail in `cc` ("redefinition of 'optimizer' as different kind of
/// symbol"), naming neither Viv line. The def is often in a module a
/// `#! use` spliced in, so the error names both places. Public so the
/// interpreter refuses the same programs.
pub fn check_def_variable_collisions(program: &Program) -> Vec<Diagnostic> {
    let mut defs: HashMap<&str, Span> = HashMap::new();
    for item in &program.items {
        if let Item::Def(d) = item {
            defs.entry(d.name.as_str()).or_insert(d.name_span);
        }
    }
    let mut errs = Vec::new();
    let mut reported = std::collections::HashSet::new();
    fn walk<'a>(stmts: &'a [Stmt], out: &mut Vec<(&'a str, Span)>) {
        for s in stmts {
            match s {
                Stmt::Assign { name, name_span, .. } => out.push((name, *name_span)),
                Stmt::For { var, var_span, body, .. } => {
                    out.push((var, *var_span));
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
    let mut assigned = Vec::new();
    for item in &program.items {
        if let Item::Stmt(s) = item {
            walk(std::slice::from_ref(s), &mut assigned);
        }
    }
    for (name, span) in assigned {
        if let Some(def_span) = defs.get(name) {
            if reported.insert(name) {
                errs.push(
                    Diagnostic::error(format!("`{name}` is the name of a def and of a top-level variable"))
                        .with_primary(span, "the variable")
                        .with_secondary(*def_span, "the def")
                        .with_note("a def and a top-level variable share one namespace in the compiled program: rename the variable"),
                );
            }
        }
    }
    errs
}

/// The names a builtin's call is lowered to when it is not `viv_`-prefixed:
/// libm's names, which the C runtime also `#define`s to Viv's own libm
/// (vmath.c), and the complex functions. The direct wasm backend reads a call
/// to one of these as the builtin. A program's own def of one of these names
/// is emitted as `viv_def_<name>` instead (`def_symbol`), so its calls reach
/// the def in every engine.
const LOWERED_BUILTIN_NAMES: &[&str] = &[
    "sin", "cos", "tan", "exp", "expm1", "log", "log10", "pow", "atan", "atan2", "asin", "acos", "sinh", "cosh", "tanh", "hypot",
    "erf", "tgamma", "sqrt", "fabs", "floor", "ceil", "round", "fmod", "fmin", "fmax", "cabs", "cexp", "clog", "csqrt", "csin",
    "ccos", "ctan", "conj", "creal", "cimag",
];

/// The symbol a user def is compiled under: its own name, unless that name
/// is one a builtin's call is lowered to (see `LOWERED_BUILTIN_NAMES`).
pub fn def_symbol(name: &str) -> String {
    if LOWERED_BUILTIN_NAMES.contains(&name) {
        format!("viv_def_{name}")
    } else {
        name.to_string()
    }
}

/// Every def's reads of a global it later shadows with a local (see
/// `global_reads_before_local`). Public so the interpreter refuses the same
/// programs the compiler does.
pub fn check_global_shadowing(program: &Program) -> Vec<Diagnostic> {
    let globals = top_level_names(program);
    let mut errs = Vec::new();
    for item in &program.items {
        if let Item::Def(def) = item {
            errs.extend(global_reads_before_local(def, &globals));
        }
    }
    errs
}

/// Names a block assigns, by `=` or as a `for` variable, at any depth.
fn assigned_names(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::For { var, body, .. } => {
                out.insert(var.clone());
                assigned_names(body, out);
            }
            Stmt::If { then_body, else_body, .. } => {
                assigned_names(then_body, out);
                assigned_names(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::Test { body, .. } => assigned_names(body, out),
            _ => {}
        }
    }
}

/// A def that reads a global `k` and later assigns its own `k`.
///
/// The two engines disagree about such a read. The C code hoists every name a
/// def assigns to a local declared at entry, so the read sees that local: zero
/// if the types match, and C that clang rejects if they do not (`k + n` with `k`
/// a `viv_arr`). The interpreter reads the global until the first assignment.
/// So the read is refused: it is any read of such a name at a point where the
/// def has not yet definitely assigned it (both branches of an `if` must
/// assign; a loop body may run zero times).
///
/// The check is conservative in one case: a read that comes textually before
/// the assignment inside a loop, guarded so it runs only after the first
/// iteration, is refused although both engines would agree. Renaming the local
/// fixes it.
fn global_reads_before_local(def: &Def, globals: &std::collections::HashSet<String>) -> Vec<Diagnostic> {
    use std::collections::HashSet;
    let mut assigned = HashSet::new();
    assigned_names(&def.body, &mut assigned);
    let risky: HashSet<String> = assigned
        .into_iter()
        .filter(|n| globals.contains(n) && !def.params.iter().any(|p| &p.name == n))
        .collect();
    if risky.is_empty() {
        return Vec::new();
    }
    let mut w = DefAssign { def: &def.name, risky: &risky, errs: Vec::new(), reported: HashSet::new() };
    // `None` is "unreachable" (after return, break or continue).
    let mut state: Option<HashSet<String>> = Some(HashSet::new());
    w.block(&def.body, &mut state);
    w.errs
}

struct DefAssign<'a> {
    def: &'a str,
    risky: &'a std::collections::HashSet<String>,
    errs: Vec<Diagnostic>,
    reported: std::collections::HashSet<String>,
}

type Assigned = Option<std::collections::HashSet<String>>;

impl DefAssign<'_> {
    fn meet(a: Assigned, b: Assigned) -> Assigned {
        match (a, b) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(a.intersection(&b).cloned().collect()),
        }
    }
    fn block(&mut self, stmts: &[Stmt], st: &mut Assigned) {
        for s in stmts {
            self.stmt(s, st);
        }
    }
    fn stmt(&mut self, s: &Stmt, st: &mut Assigned) {
        match s {
            Stmt::Assign { name, value, .. } => {
                self.expr(value, st, &[]);
                if let Some(set) = st {
                    set.insert(name.clone());
                }
            }
            Stmt::Return { value, .. } => {
                self.expr(value, st, &[]);
                *st = None;
            }
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => self.expr(expr, st, &[]),
            Stmt::SetIndex { base, indices, value, .. } => {
                self.expr(base, st, &[]);
                for i in indices {
                    self.expr(i, st, &[]);
                }
                self.expr(value, st, &[]);
            }
            Stmt::SetField { base, value, .. } => {
                self.expr(base, st, &[]);
                self.expr(value, st, &[]);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.expr(cond, st, &[]);
                let mut t = st.clone();
                self.block(then_body, &mut t);
                let mut e = st.clone();
                self.block(else_body, &mut e);
                *st = Self::meet(t, e);
            }
            Stmt::While { cond, body, .. } => {
                self.expr(cond, st, &[]);
                let mut b = st.clone();
                self.block(body, &mut b);
            }
            Stmt::For { var, bound, body, .. } => {
                self.expr(bound, st, &[]);
                let mut b = st.clone();
                if let Some(set) = &mut b {
                    set.insert(var.clone());
                }
                self.block(body, &mut b);
            }
            Stmt::Test { body, .. } => {
                let mut b = st.clone();
                self.block(body, &mut b);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } => *st = None,
            // a stop ends the run: nothing after it is reached
            Stmt::Lens { lens, .. } if matches!(**lens, Lens::Stop { .. }) => *st = None,
            Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
            Stmt::Narrate { .. } => {}
        }
    }
    fn expr(&mut self, e: &Expr, st: &Assigned, shadow: &[String]) {
        match e {
            Expr::Var { name, span } => {
                let unassigned = st.as_ref().is_some_and(|s| !s.contains(name));
                if unassigned && self.risky.contains(name) && !shadow.contains(name) && self.reported.insert(name.clone()) {
                    self.errs.push(
                        Diagnostic::error(format!(
                            "`{name}` is read here before `{}` assigns its own `{name}`",
                            self.def
                        ))
                        .with_primary(*span, format!("this reads the global `{name}` in the interpreter, but the local `{name}` in compiled code"))
                        .with_note(format!(
                            "an assignment anywhere in a function makes `{name}` a local for the whole body; rename the local, or pass the global in as a parameter"
                        )),
                    );
                }
            }
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    self.expr(x, st, shadow);
                }
            }
            Expr::Index { base, indices, .. } => {
                self.expr(base, st, shadow);
                for x in indices {
                    self.expr(x, st, shadow);
                }
            }
            Expr::Lambda { params, body, .. } => {
                let mut inner = shadow.to_vec();
                inner.extend(params.iter().cloned());
                self.expr(body, st, &inner);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.expr(operand, st, shadow),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs, st, shadow);
                self.expr(rhs, st, shadow);
            }
            Expr::Call { args, .. } => {
                for x in args {
                    self.expr(x, st, shadow);
                }
            }
            Expr::RecordLit { fields, .. } => {
                for (_, _, x) in fields {
                    self.expr(x, st, shadow);
                }
            }
            Expr::Field { base, .. } => self.expr(base, st, shadow),
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }
}

/// Literals with at least this many elements are emitted as a table.
const LIT_TAB_MIN: usize = 16;

/// The initializer `(const double[]){...}` for an array literal of rank 1 or 2
/// whose elements are all numeric constants (`1.5` or `-1.5`) and that has at
/// least `LIT_TAB_MIN` of them; `None` otherwise, and the literal keeps the
/// variadic form. Short literals stay variadic so the generated C still reads
/// `viv_vec(3, 1.0, 2.0, 3.0)`.
fn const_table(node: &Node<Viv>, cx: &CEmitCx<Viv>, rank: usize) -> Option<String> {
    if rank > 2 || node.children.len() < LIT_TAB_MIN {
        return None;
    }
    let mut out = String::with_capacity(node.children.len() * 24 + 24);
    out.push_str("(const double[]){");
    for (i, &c) in node.children.iter().enumerate() {
        let v = match cx.kind(c) {
            Viv::Num(v) => v,
            Viv::Neg => match cx.children(c).as_slice() {
                [g] => match cx.kind(*g) {
                    Viv::Num(v) => -v,
                    _ => return None,
                },
                _ => return None,
            },
            _ => return None,
        };
        if !v.is_finite() {
            return None;
        }
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&fmt_f64(v));
    }
    out.push('}');
    Some(out)
}
