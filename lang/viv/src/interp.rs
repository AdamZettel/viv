// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! A tree-walking interpreter for Viv — the browser execution path (run Viv
//! without a C toolchain). The C backend remains the native/canonical path; this
//! shares the exact same lexer and parser (`crate::parse`), so only *evaluation*
//! is new, never parsing. Output matches the C runtime's `%g\n` formatting.
//!
//! It runs the whole language: numbers, arrays, records, lambdas and closures,
//! the numerical builtins. What the C lowering checks statically it checks as
//! it runs, with the same words, and `run_diagnosed` (`viv run`) lowers the
//! program first, so both engines refuse the same programs.

use crate::ast::{BinOp, Expr, HoldCheck, Item, Lens, Stmt, TypeExpr};
use knotkit::diag::Diagnostic;
use knotkit::span::{SourceId, Span};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use crate::vmath::VMath;

type Arr = Rc<RefCell<ArrData>>;

/// One argument of `print`: text, or a value and the format it is written in.
enum Part {
    Text(String),
    Val(crate::printfmt::Fmt, Val),
    /// a map, its keys in the first format and its values in the second
    Map(crate::printfmt::Fmt, crate::printfmt::Fmt, Val),
}

#[derive(Debug)]
struct ArrData {
    shape: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Clone, Debug)]
enum Val {
    Num(f64),
    /// `true` / `false`: what a comparison yields and what `if` and `while`
    /// take. Its own variant, not 1.0 and 0.0, so that `flag + 1.0` or
    /// `if count {` is refused here exactly as the typed C lowering refuses it.
    Bool(bool),
    Cplx(f64, f64),
    Arr(Arr),
    /// A packed COMPLEX array: same `ArrData`, but `data` is interleaved
    /// (re, im) pairs, so `data.len() == 2 * size`. That is exactly C99's
    /// `double complex` layout, which is what lets the two engines agree to
    /// the bit. A separate variant rather than a flag on `ArrData` on purpose:
    /// every existing `Val::Arr` site then CANNOT match a complex array and
    /// misread interleaved data as reals -- the type system is the guard,
    /// instead of an audit of thirty destructuring sites.
    CArr(Arr),
    Closure(Rc<Closure>),
    /// A named top-level `def` used as a value: `rk4(rhs, ...)` where `rhs` is a
    /// def. Carries the name so the call graph still records real edges.
    Func(String, Rc<Func>),
    /// A BOXED value array: holds arbitrary Vals (numbers, functions, records,
    /// other arrays), unlike the packed `Arr` which is `Vec<f64>` for speed.
    /// Built only when a `[ ... ]` literal is not uniformly numeric, so numeric
    /// arrays keep the fast path. This is the interpreter side of "arrays of
    /// arbitrary things".
    VArr(Rc<RefCell<Vec<Val>>>),
    /// A string: immutable, UTF-8, indexed by code point.
    Str(Rc<strings::Str>),
    /// A list of strings: immutable too (an append rebinds, growing in place
    /// when nothing else holds the list).
    Strs(strings::Strs),
    /// A record value: its type's name and its fields in
    /// declaration order. Shared by `Rc`: Phase 1 records have no field
    /// assignment, so a copy that shares the fields behaves exactly as the
    /// C backend's by-value struct copy -- scalars cannot be changed through
    /// either name, and an array field is one handle in both.
    Record(Rc<RecVal>),
    /// A map: a handle changed in place, so every holder of
    /// the `Rc` sees every change, as every holder of a C `viv_map*` does.
    Map(maps::Map),
    /// An Int.
    Int(i64),
    /// An array of Ints: rank 1, changed in place through its handle, as a
    /// number array is.
    IArr(Rc<RefCell<Vec<i64>>>),
    /// An enum value or an optional: immutable, so the
    /// `Rc` is shared freely, as the compiled engines share a node.
    Enum(Rc<EnumVal>),
    /// An array of records (`[Block]`). A
    /// value, as a record is: holders share the `Rc`, and a write copies
    /// first when another holder exists (`Rc::make_mut`), so a write through
    /// one name is never seen through another. Each element is a
    /// `Val::Record`.
    Recs(Rc<Vec<Val>>),
    /// A Float32.
    F32(f32),
    /// A packed array (`[Float32]`, `[BFloat16]`, `[Int8]`, ...): its kind
    /// and its bytes, changed in place through its handle as an Int array is.
    PArr(Rc<RefCell<PArr>>),
}

/// A packed array's kind and its elements' bytes, little-endian.
#[derive(Debug, Clone)]
pub(crate) struct PArr {
    pub k: crate::ast::Store,
    pub b: Vec<u8>,
}

fn parr(k: crate::ast::Store, b: Vec<u8>) -> Val {
    Val::PArr(Rc::new(RefCell::new(PArr { k, b })))
}

/// An enum value: its variant's name and its payload. The variant's name is
/// all the interpreter needs: the lowering has checked every use against
/// the enum's declaration.
#[derive(Debug)]
pub(crate) struct EnumVal {
    variant: Rc<str>,
    fields: Vec<Val>,
}

/// A list a million nodes long is dropped one node at a time, not by a
/// million nested drops: each payload this value held last is moved onto a
/// stack and dropped from there.
impl Drop for EnumVal {
    fn drop(&mut self) {
        let mut stack: Vec<Rc<EnumVal>> = Vec::new();
        for f in self.fields.drain(..) {
            if let Val::Enum(e) = f {
                stack.push(e);
            }
        }
        while let Some(e) = stack.pop() {
            if let Ok(mut inner) = Rc::try_unwrap(e) {
                for f in inner.fields.drain(..) {
                    if let Val::Enum(e) = f {
                        stack.push(e);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct RecVal {
    name: Rc<str>,
    fields: Vec<(Rc<str>, Val)>,
}

impl RecVal {
    fn get(&self, field: &str) -> Option<&Val> {
        self.fields.iter().find(|(f, _)| &**f == field).map(|(_, v)| v)
    }
}

/// What a record field holds, as the interpreter checks it: the C lowering's
/// field type (`lower::Ty`), or `None` for a tuple's field, whose type the
/// compiled engines infer from the returns.
#[derive(Debug)]
struct RecDecl {
    fields: Vec<(String, Option<crate::lower::Ty>)>,
}

/// An anonymous function value: a single-expression body plus a snapshot of the
/// bindings visible where the `fn(...) -> ...` was written (so it closes over
/// them). Used by the higher-order builtins `deriv`/`simpson`/`trapezoid` and by
/// closure-valued variables applied like `f(x)`.
#[derive(Debug)]
struct Closure {
    params: Vec<String>,
    body: Expr,
    captured: HashMap<String, Val>,
}

impl Val {
    fn as_num(&self, what: &str) -> Result<f64, String> {
        match self {
            Val::Num(n) => Ok(*n),
            // where a number is needed an Int converts exactly, or stops
            Val::Int(i) => ints::to_num(*i),
            Val::IArr(_) => Err(format!("expected a number for {what}, got an Int array")),
            Val::Enum(_) => Err(format!("expected a number for {what}, got an enum value")),
            Val::Bool(_) => Err(format!("expected a number for {what}, got a boolean")),
            Val::Cplx(..) => Err(format!("expected a number for {what}, got a complex number")),
            Val::Arr(_) => Err(format!("expected a number for {what}, got an array")),
            Val::CArr(_) => Err(format!("expected a number for {what}, got a complex array")),
            Val::Closure(_) | Val::Func(..) => {
                Err(format!("expected a number for {what}, got a function"))
            }
            Val::VArr(_) => Err(format!("expected a number for {what}, got a value array")),
            Val::Str(_) => Err(format!("expected a number for {what}, got a string")),
            Val::Strs(_) => Err(format!("expected a number for {what}, got a list of strings")),
            Val::Record(r) => Err(format!("expected a number for {what}, got a {}", r.name)),
            Val::Map(_) => Err(format!("expected a number for {what}, got a map")),
            Val::Recs(_) => Err(format!("expected a number for {what}, got an array of records")),
            Val::F32(_) => Err(format!("expected a number for {what}, got a Float32; write float(x)")),
            Val::PArr(p) => Err(format!("expected a number for {what}, got {}", crate::lower::Ty::Arr(crate::lower::Elem::P(p.borrow().k)).name())),
        }
    }
    fn truthy(&self) -> Result<bool, String> {
        match self {
            Val::Bool(b) => Ok(*b),
            Val::Arr(_) => Err(format!("condition must be a boolean, and this is an array; {}", crate::hints::ARRAY_CONDITION)),
            _ => Err("condition must be a boolean (a comparison, `and`/`or`/`not`, `true`/`false`, or a variable holding one)".into()),
        }
    }
}

/// Non-local control flow out of a statement.
enum Flow {
    Normal,
    Return(Val),
    Break,
    Continue,
}

#[derive(Debug)]
struct Func {
    params: Vec<String>,
    /// For each parameter, the record type its annotation names (`p: Vec3`),
    /// which a call checks its argument against, as the compiled engines do.
    rec_params: Vec<Option<String>>,
    body: Vec<Stmt>,
}

struct Interp {
    /// The program's record types, by name (`record` declarations and the
    /// tuples `crate::tuples` synthesizes).
    records: HashMap<String, Rc<RecDecl>>,
    /// The literals and calls computed as Ints.
    ints: RefCell<Rc<IntSites>>,
    /// Release mode: contracted solves (`spectrum_sym`) auto-select their
    /// method. Debug (default): they run the whole method sweep and fail
    /// loudly on dissent — numerical CI at the language level.
    release: bool,
    funcs: HashMap<String, Rc<Func>>,
    // Top-level bindings, visible (read-only for scalars) inside functions —
    // matching the C backend: a function can't rebind a scalar global, but it
    // can mutate a global array through its handle (`nn[0] = ...`), which is
    // what the tape-based examples (autodiff_reverse, mini_cas) rely on.
    globals: RefCell<HashMap<String, Val>>,
    out: RefCell<String>,
    steps: RefCell<u64>, // a budget so an infinite loop fails instead of hanging the tab
    // Call-graph instrumentation: a stack of active user-function names, the
    // (caller, callee) event sequence (bounded), and per-edge counts. Emitted
    // after the run as `__graph__`/`__trace__` lines for live visualizers.
    call_stack: RefCell<Vec<String>>,
    call_trace: RefCell<Vec<(String, String)>>,
    call_counts: RefCell<HashMap<(String, String), u64>>,
    // Assertion failures seen this run: `viv run` exits nonzero when > 0, so
    // CI and harnesses can gate on the exit code instead of grepping stdout.
    fails: std::cell::Cell<u32>,
    // The active `test` block. A test is the unit that passes or fails, so an
    // assert inside one records against it instead of printing its own line,
    // and the block prints `PASS <name>` / `FAIL <name>` when it ends. Depth
    // absorbs a nested test into the outer one rather than closing it early.
    test_name: RefCell<String>,
    test_depth: std::cell::Cell<u32>,
    test_asserts: std::cell::Cell<u32>,
    test_fails: std::cell::Cell<u32>,
    test_detail: RefCell<String>,
    // Seeded PRNG state for rand()/srand(): Park-Miller MINSTD (a = 16807,
    // m = 2^31 - 1), exact in f64 and portable to the C backend verbatim.
    rng: std::cell::Cell<f64>,
    // Step events: the first STEP_CAP
    // enter/exit events as `+def/id@depth` / `-def/id@depth`, how many there
    // were in all, and the steps open right now. Emitted after the run as one
    // `__steps__` line, and only when a step ran. Release runs record nothing.
    step_events: RefCell<Vec<String>>,
    step_total: std::cell::Cell<u64>,
    step_open: RefCell<Vec<(String, String, u32)>>,
    // Whether any step has been entered. Until one has, no step can be open,
    // and the loops and calls skip the bookkeeping: a program without steps
    // runs exactly the work it ran before steps existed.
    step_seen: std::cell::Cell<bool>,
    // The story: present only when the
    // program carries a debug lens and this is a debug run. Emitted after
    // the run as one `__story__` line, the same bytes the C backend's
    // `viv_story_*` runtime writes.
    story: RefCell<Option<Story>>,
    // Where the run stopped, for the error's source location: the span of the
    // innermost expression or statement that failed. Errors travel up as plain
    // strings; the first `eval`/`exec_stmt` to see one return records its own
    // span here, and the enclosing ones leave it alone. Nothing in the
    // interpreter catches an error and carries on, so the first span stays
    // right until the run ends.
    err_at: std::cell::Cell<Option<Span>>,
    // Notes for that error: hints for habits from other languages (a negative
    // index, say), printed under the location as the compiler's notes are.
    err_notes: RefCell<Vec<String>>,
    // The tape of reverse-mode autodiff (`crate::grad`): the values the forward sweep of a gradient saves and its reverse
    // sweep takes back, last in first out. Arrays are stored as copies.
    ad_tape: RefCell<Vec<Val>>,
    // The `parallel for` loops a memory lens pins as disjoint
    // (crate::memory_lens), and, while one runs in a debug run, the record of
    // which pass wrote each element of its pinned arrays: a second pass
    // writing one stops the run, as the C runtime's `viv_dj_*` does.
    pinned: RefCell<Vec<crate::parallel::Pinned>>,
    dj: RefCell<Option<Dj>>,
    dj_on: std::cell::Cell<bool>,
}

/// The record a pinned `parallel for` keeps while it runs: what to call it in
/// the error, the pass running, and per pinned array its storage, name,
/// shape and the pass that wrote each element.
struct Dj {
    what: String,
    pass: i64,
    arrays: Vec<(Rc<RefCell<ArrData>>, String, Vec<usize>, HashMap<usize, i64>)>,
}

/// The run-time error of a broken `disjoint` pin: the words the C runtime's
/// `viv_dj_fail` and the wasm runtime's `vrt_dj_note` print. The element is
/// named by its indices, from its flat position in an array of `shape`.
pub(crate) fn dj_error(a: i64, b: i64, name: &str, shape: &[usize], flat: usize, what: &str) -> String {
    let rank = shape.len().max(1);
    let mut part = vec![0usize; rank];
    let mut rest = flat;
    for d in (0..rank).rev() {
        let dim = shape.get(d).copied().unwrap_or(1);
        part[d] = if dim > 0 { rest % dim } else { 0 };
        rest = if dim > 0 { rest / dim } else { 0 };
    }
    let ix: Vec<String> = part.iter().map(|p| p.to_string()).collect();
    format!("passes {} and {} both write {name}[{}] in {what}", a.min(b), a.max(b), ix.join(", "))
}

/// The story being recorded: its events as JSON text, how many there were,
/// and how many holds failed. Only the first STORY_CAP events are kept.
struct Story {
    buf: String,
    total: u64,
    fails: u32,
}

// The C runtime's VIV_STORY_CAP.
const STORY_CAP: u64 = 100_000;
// Arrays up to this many elements are recorded whole; larger ones as their
// shape, min, max, norm and (when square) trace. The C runtime's
// VIV_STORY_ARR_CAP.
const STORY_ARR_CAP: usize = 4096;

const TRACE_CAP: usize = 400;
// The C backend's VIV_STEP_CAP: both engines keep the same first events.
const STEP_CAP: usize = 1000;

// 250M steps ~ 20 s of interpretation: big enough for real programs (the
// nn_symmetry search, the full TOV sweep), small enough to catch a hung loop.
// Infinite-loop guard, not a size police: big enough for a whole-paper
// notebook run (the MFD page's full browser chain is ~500M steps), small
// enough that a genuine hang still dies in well under a minute of wasm.
const STEP_BUDGET: u64 = 2_000_000_000;

/// The budget in force: STEP_BUDGET unless `set_step_budget` raised or lowered
/// it for this process (`viv run` reads VIV_STEP_BUDGET).
static STEP_LIMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(STEP_BUDGET);

/// Set the interpreter's step budget for every later run in this process: a
/// long run that is not a hang (the largest exported vivscf cases take more
/// than 2e9 steps). 0 restores the default.
pub fn set_step_budget(steps: u64) {
    STEP_LIMIT.store(if steps == 0 { STEP_BUDGET } else { steps }, std::sync::atomic::Ordering::Relaxed);
}

/// The step budget in force.
pub fn step_budget() -> u64 {
    STEP_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}
// Cap on elements in a single array allocation. Native code (the interpreter
// of `viv run`, and compiled C) holds up to 2^31 - 1 elements, the C runtime's
// `int` sizes: 16 GB of doubles, past which an array is refused before
// anything is allocated (and a failed allocation stops the run with "out of
// memory"). The wasm engines (the interpreter in a browser, and the runtime
// of the direct wasm backend, both in viv.wasm) keep 16,000,000 elements
// (128 MB): wasm32 addresses 4 GB in all, and a stray `zeros(100000, 100000)`
// in an edited cell must not take the browser tab down. A refusal there says
// so (`LIMIT_OF`). Native's limit rose from 16,000,000 on 2026-10-02
// (study/ports/pyscf: the packed integrals of benzene cc-pVDZ and (H2O)8
// 6-31G* need 21.1 and 54.5 million).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const MAX_ELEMS: usize = 2_147_483_647;
#[cfg(target_arch = "wasm32")]
pub(crate) const MAX_ELEMS: usize = 16_000_000;
/// What follows "the N-element limit" in a refusal: the engine, where the
/// limit is the wasm engines' smaller one.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const LIMIT_OF: &str = "";
#[cfg(target_arch = "wasm32")]
pub(crate) const LIMIT_OF: &str = " of the wasm engines (native code holds more)";

/// The dimensions `zeros` and `czeros` are given, read as every engine reads
/// them (a negative or NaN one is 0, a fraction is cut to its whole part, a
/// huge one saturates), and the element count, a product in 64 bits that
/// saturates. The count is checked against the element limit before anything
/// is allocated, in 64-bit arithmetic on every host (the wasm runtime's
/// `usize` is 32 bits), so `zeros(65536, 65537)` is refused with the same
/// count everywhere instead of wrapping to a small array.
pub(crate) fn zeros_shape(dims: &[f64]) -> (Vec<usize>, u64) {
    let ds: Vec<u64> = dims.iter().map(|n| n.max(0.0) as u64).collect();
    let total = ds.iter().fold(1u64, |acc, &d| acc.saturating_mul(d));
    (ds.iter().map(|&d| d.min(usize::MAX as u64) as usize).collect(), total)
}

/// The order `sort` puts numbers in, in every engine (the C runtime's
/// `viv_sort_before` is its twin): ascending, -0.0 before +0.0, and NaN after
/// every number, NaNs keeping their order. `sort_by` is stable, as the C's
/// merge sort is, so equal keys (only NaNs are equal without being the same
/// bits) come out in the same order in both.
/// `x = lin_solve(A, b)`, `L = cholesky(A)` and `X = inv(A)` as the older
/// builtins that fill an array (`lin_solve(A, b, x)`, `lin_solve_multi`,
/// `cholesky(A, L)`): the call to make, and the array it fills, which is the
/// value. Anything else comes back as it was.
fn returning(callee: &str, a: Vec<Val>) -> Result<(&str, Vec<Val>, Option<Val>), String> {
    let shape = |v: &Val| match v {
        Val::Arr(x) => Some(x.borrow().shape.clone()),
        _ => None,
    };
    let square = |v: &Val, who: &str| -> Result<usize, String> {
        match shape(v) {
            Some(s) if s.len() == 2 && s[0] == s[1] => Ok(s[0]),
            _ => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{who}` needs a square 2-D matrix"))),
        }
    };
    match (callee, a.len()) {
        ("lin_solve", 2) => {
            let b = shape(&a[1]).ok_or("`lin_solve` takes (A, b) arrays")?;
            if b.len() == 2 {
                let n = square(&a[0], "lin_solve_multi")?;
                if b[0] != n {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`lin_solve_multi`: B must be n x m"));
                }
                let x = new_arr(b.clone(), vec![0.0; b[0] * b[1]]);
                return Ok(("lin_solve_multi", vec![a[0].clone(), a[1].clone(), x.clone()], Some(x)));
            }
            let n = square(&a[0], "lin_solve")?;
            let x = new_arr(vec![n], vec![0.0; n]);
            Ok(("lin_solve", vec![a[0].clone(), a[1].clone(), x.clone()], Some(x)))
        }
        ("inv", 1) => {
            let n = square(&a[0], "lin_solve_multi")?;
            let mut id = vec![0.0; n * n];
            for i in 0..n {
                id[i * n + i] = 1.0;
            }
            let x = new_arr(vec![n, n], vec![0.0; n * n]);
            Ok(("lin_solve_multi", vec![a[0].clone(), new_arr(vec![n, n], id), x.clone()], Some(x)))
        }
        ("cholesky", 1) => {
            let n = square(&a[0], "cholesky")?;
            let l = new_arr(vec![n, n], vec![0.0; n * n]);
            Ok(("cholesky", vec![a[0].clone(), l.clone()], Some(l)))
        }
        _ => Ok((callee, a, None)),
    }
}

pub fn sort_order(x: &f64, y: &f64) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Equal,
        (true, false) => Greater,
        (false, true) => Less,
        _ => x.partial_cmp(y).unwrap().then_with(|| y.is_sign_negative().cmp(&x.is_sign_negative())),
    }
}

/// Whether the digits lens may treat an operand as counting when its value
/// is whole (crate::digits): anything but the value of a math function, so
/// `n - 1 - k` and `nelec + spin` stay exact and `1.0 - cos(x)` does not.
fn counting_expr(e: &Expr) -> bool {
    match e {
        Expr::Call { callee, .. } => !matches!(
            callee.as_str(),
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh" | "exp" | "expm1" | "log" | "log10"
                | "log1p" | "log2" | "sqrt" | "cbrt" | "pow" | "hypot" | "erf" | "erfc" | "gamma" | "lgamma"
        ),
        Expr::Neg { operand, .. } => counting_expr(operand),
        _ => true,
    }
}

pub fn run(src: &str) -> Result<String, String> {
    run_mode(src, false)
}

/// Release-mode run: contracted solves skip the cross-check sweep and use the
/// auto-selected method only.
pub fn run_release(src: &str) -> Result<String, String> {
    run_mode(src, true)
}

/// Split a program into its defs and its top-level statements. With `unique`
/// (a whole program, not a notebook cell), two programs the compiler refuses
/// are refused here too, so that a program runs in both engines or in neither:
/// - a second def of one name. Two `#! use`d modules that both define `f`
///   would otherwise run whichever came last here and fail to compile there.
/// - a def that reads a global and then assigns a local of the same name
///   (`lower::check_global_shadowing`), where the engines read different values.
///
/// Notebook cells pass `unique = false`: a later cell may redefine a def on
/// purpose, and a cell's globals are not all visible in its own source.
fn parse_items(src: &str, unique: bool) -> Result<(HashMap<String, Rc<Func>>, Vec<Stmt>, Records), String> {
    parse_items_lens(src, unique, false).map(|p| (p.funcs, p.main, p.records))
}

/// The record types of a program, by name.
type Records = HashMap<String, Rc<RecDecl>>;

/// A program split for the interpreter.
struct Parsed {
    /// the literals and calls the lowering computes as Ints (`IntSites`)
    ints: IntSites,
    funcs: HashMap<String, Rc<Func>>,
    main: Vec<Stmt>,
    /// whether a debug lens was compiled in (the run records a story)
    story: bool,
    records: Records,
    /// the loops a memory lens pins (`ast::Program::memory`)
    pinned: Vec<crate::parallel::Pinned>,
}

/// `parse_items`, with the debug lens compiled in when `lens` (a debug run
/// of a whole program). The flag says whether there was a lens to compile,
/// which is when the run records a story.
fn parse_items_lens(src: &str, unique: bool, lens: bool) -> Result<Parsed, String> {
    parse_items_diag(src, unique, lens).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))
}

/// `parse_items_lens`, with its errors as diagnostics (spans and all) for a
/// caller that prints them with their source locations.
fn parse_items_diag(src: &str, unique: bool, lens: bool) -> Result<Parsed, Vec<Diagnostic>> {
    let program = crate::parse(src, SourceId(0))?;
    let ints = IntSites::of(src, &program.whole);
    if unique {
        if let Some(d) = crate::lower::check_global_shadowing(&program).first() {
            return Err(vec![d.clone()]);
        }
        if let Some(d) = crate::lower::check_def_variable_collisions(&program).first() {
            return Err(vec![d.clone()]);
        }
    }
    // The record declarations, checked by the C lowering's own rules
    // (`lower::record_layouts`: redeclaration, duplicate fields, unknown field
    // types), so a declaration one engine refuses every engine refuses.
    let records: Records = crate::lower::record_layouts(&program)?
        .into_iter()
        .map(|(name, fields)| (name, Rc::new(RecDecl { fields })))
        .collect();
    // the recursion limit, as the compiled program has it (`crate::build_ir`)
    let program = crate::recursion::guard(program);
    let story = lens && !program.debug.is_empty();
    let program = if story { crate::debug_lens::instrument(program)? } else { program };
    let pinned = program.memory.clone();
    let mut funcs = HashMap::new();
    let mut main: Vec<Stmt> = Vec::new();
    for item in program.items {
        match item {
            Item::Def(d) => {
                if unique && funcs.contains_key(&d.name) {
                    return Err(vec![Diagnostic::error(format!("function `{}` is already defined", d.name)).with_primary(d.name_span, "defined again here")]);
                }
                let rec_params = d
                    .params
                    .iter()
                    .map(|p| match &p.ty {
                        Some(TypeExpr::Named { name, .. }) if records.contains_key(name) => Some(name.clone()),
                        _ => None,
                    })
                    .collect();
                let params = d.params.into_iter().map(|p| p.name).collect();
                funcs.insert(d.name, Rc::new(Func { params, rec_params, body: d.body }));
            }
            Item::Stmt(s) => main.push(s),
            Item::Rung(_) => return Err(vec![Diagnostic::error("a rung reached the interpreter: parse() flattens ladders")]),
            Item::Record(_) | Item::Enum(_) => {}
        }
    }
    Ok(Parsed { ints, funcs, main, story, records, pinned })
}

/// What the interpreter computes as Ints that the tree does not say: the
/// integer literals and the calls (`len`, `int`, `floor`, `ceil`, `round`)
/// the lowering computed as Ints because they stand in Int positions.
/// Empty for a program with no Int.
#[derive(Default)]
struct IntSites {
    lits: HashMap<(u32, u32), i64>,
    calls: std::collections::HashSet<(u32, u32)>,
    /// Int arguments the lowering converts to numbers (a def's parameter
    /// that is passed Ints and numbers)
    nums: std::collections::HashSet<(u32, u32)>,
    /// array literals the lowering made Int arrays
    arrs: std::collections::HashSet<(u32, u32)>,
    /// number literals the lowering made Float32s
    f32: std::collections::HashSet<(u32, u32)>,
}

impl IntSites {
    fn of(src: &str, whole: &HashMap<(u32, u32), i64>) -> IntSites {
        let decided = crate::lower::ints::int_spans(src);
        // (read after int_spans lowered `src`, and only when it did)
        let lowered = crate::lower::ints::mentions_ints(src);
        let nums = if lowered { crate::lower::ints::numed_spans() } else { Default::default() };
        let arrs = if lowered { crate::lower::ints::arr_spans() } else { Default::default() };
        let f32 = crate::lower::f32s::spans(src).0;
        let mut out = IntSites { nums, arrs, f32, ..IntSites::default() };
        for at in decided {
            match whole.get(&at) {
                Some(&v) => {
                    out.lits.insert(at, v);
                }
                None => {
                    out.calls.insert(at);
                }
            }
        }
        out
    }
}

fn fresh_interp(release: bool, funcs: HashMap<String, Rc<Func>>, globals: HashMap<String, Val>, records: Records) -> Interp {
    crate::errkind::reset();
    Interp {
        records,
        ints: RefCell::new(Rc::new(IntSites::default())),
        release,
        funcs,
        globals: RefCell::new(globals),
        out: RefCell::new(String::new()),
        steps: RefCell::new(0),
        call_stack: RefCell::new(Vec::new()),
        call_trace: RefCell::new(Vec::new()),
        call_counts: RefCell::new(HashMap::new()),
        fails: std::cell::Cell::new(0),
        test_name: RefCell::new(String::new()),
        test_depth: std::cell::Cell::new(0),
        test_asserts: std::cell::Cell::new(0),
        test_fails: std::cell::Cell::new(0),
        test_detail: RefCell::new(String::new()),
        rng: std::cell::Cell::new(12345.0),
        step_events: RefCell::new(Vec::new()),
        step_total: std::cell::Cell::new(0),
        step_open: RefCell::new(Vec::new()),
        step_seen: std::cell::Cell::new(false),
        story: RefCell::new(None),
        err_at: std::cell::Cell::new(None),
        err_notes: RefCell::new(Vec::new()),
        ad_tape: RefCell::new(Vec::new()),
        pinned: RefCell::new(Vec::new()),
        dj: RefCell::new(None),
        dj_on: std::cell::Cell::new(false),
    }
}

fn exec_main(interp: &Interp, main: &[Stmt]) -> Result<(), String> {
    let mut scope: HashMap<String, Val> = HashMap::new();
    match interp.exec_block(main, &mut scope, true)? {
        Flow::Normal | Flow::Return(_) => Ok(()),
        Flow::Break | Flow::Continue => Err("`break`/`continue` outside a loop".into()),
    }
}

fn render_output(interp: &Interp) -> String {
    let mut output = interp.out.borrow().clone();
    let counts = interp.call_counts.borrow();
    if !counts.is_empty() {
        let mut seen: Vec<String> = Vec::new();
        for (from, to) in counts.keys() {
            if !seen.contains(from) { seen.push(from.clone()); }
            if !seen.contains(to) { seen.push(to.clone()); }
        }
        let mut edges: Vec<(&(String, String), &u64)> = counts.iter().collect();
        edges.sort();
        let nodes_json = seen.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(",");
        let edges_json = edges
            .iter()
            .map(|((f, t), c)| format!("{{\"from\":\"{}\",\"to\":\"{}\",\"count\":{}}}", f, t, c))
            .collect::<Vec<_>>()
            .join(",");
        output.push_str(&format!("__graph__ {{\"nodes\":[{}],\"edges\":[{}]}}\n", nodes_json, edges_json));
        let trace = interp.call_trace.borrow();
        let trace_json = trace
            .iter()
            .map(|(f, t)| format!("\"{}>{}\"", f, t))
            .collect::<Vec<_>>()
            .join(",");
        output.push_str(&format!("__trace__ [{}]\n", trace_json));
    }
    if interp.step_total.get() > 0 {
        let events = interp.step_events.borrow().iter().map(|e| format!("\"{}\"", e)).collect::<Vec<_>>().join(",");
        output.push_str(&format!("__steps__ {{\"total\":{},\"events\":[{}]}}\n", interp.step_total.get(), events));
    }
    if let Some(st) = interp.story.borrow().as_ref() {
        output.push_str(&format!("__story__ {{\"total\":{},\"events\":[{}]}}\n", st.total, st.buf));
    }
    output
}

/// A whole-program interpreter, with the debug lens compiled in for a debug
/// run of a program that has one.
fn program_interp(src: &str, release: bool) -> Result<(Interp, Vec<Stmt>), String> {
    program_interp_diag(src, release).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))
}

/// `program_interp`, with its errors as diagnostics.
fn program_interp_diag(src: &str, release: bool) -> Result<(Interp, Vec<Stmt>), Vec<Diagnostic>> {
    let Parsed { ints, funcs, main, story, records, pinned } = parse_items_diag(src, true, !release)?;
    let interp = fresh_interp(release, funcs, HashMap::new(), records);
    *interp.pinned.borrow_mut() = pinned;
    *interp.ints.borrow_mut() = Rc::new(ints);
    if story {
        *interp.story.borrow_mut() = Some(Story { buf: String::new(), total: 0, fails: 0 });
    }
    Ok((interp, main))
}

fn run_mode(src: &str, release: bool) -> Result<String, String> {
    let (interp, main) = program_interp(src, release)?;
    exec_main(&interp, &main)?;
    Ok(render_output(&interp))
}

/// Like `run`, but a run that stops with an error keeps what the program
/// printed before it stopped: `Err((printed, error))`. The CLI's
/// `run_diagnosed` and the compiled engines keep that output too; this is the
/// same for the browser, which otherwise showed only the error.
pub fn run_keeping(src: &str, release: bool) -> Result<String, (String, String)> {
    let (interp, main) = program_interp(src, release).map_err(|e| (String::new(), e))?;
    match exec_main(&interp, &main) {
        Ok(()) => Ok(render_output(&interp)),
        Err(e) => Err((interp.out.borrow().clone(), e)),
    }
}

/// A whole-program run as the CLI reports it: the program's output, the
/// failed asserts and holds, and -- when the run stopped -- the diagnostics
/// that say why and where.
pub struct Run {
    /// What the program printed. On a clean run this includes the
    /// `__graph__`/`__trace__`/`__steps__`/`__story__` lines; when the run
    /// stopped with an error it is only what the program printed before it
    /// stopped -- the same text the compiled program has written by then.
    pub out: String,
    pub fails: u32,
    pub holds: u32,
    /// Empty on a clean run. A program that does not parse or check gets the
    /// front end's diagnostics; one that stops at run time gets one error,
    /// located at the innermost expression or statement that failed.
    pub errors: Vec<Diagnostic>,
    /// The kind of the run-time error the run stopped with (crate::errkind),
    /// the one an exported program returns for it; None on a clean run and
    /// for a program that does not parse or check.
    pub kind: Option<crate::errkind::ErrKind>,
}

/// The checks `run_diagnosed` makes before it runs anything: the compiler's
/// (the typed lowering every engine starts from) and the interpreter's own
/// (a def defined twice, a global shadowed by a local). Empty when the
/// program would run. Spans are in `SourceId(0)`, as in `run_diagnosed`.
pub fn check_diagnosed(src: &str) -> Vec<Diagnostic> {
    if let Err(errors) = crate::build_ir(src, SourceId(0)) {
        return errors;
    }
    match program_interp_diag(src, false) {
        Ok(_) => Vec::new(),
        Err(errors) => errors,
    }
}

/// Run a whole program and report it as diagnostics rather than a string, so
/// the caller can print the error with its file, line and column (and the
/// source line with a caret), or as JSON. Spans are in `SourceId(0)`: the
/// caller's source map must hold `src` under that id.
pub fn run_diagnosed(src: &str, release: bool) -> Run {
    // The compiler's checks come first, so `viv run` refuses what the
    // compile refuses, with the same diagnostics: a call to an unknown
    // function in a branch the run never takes, a variable given two
    // types (`for t to T`, then `t = draws(B)`), a lambda that captures a
    // local. The interpreter alone would run such a program, and the
    // compiled one would not build.
    if let Err(errors) = crate::build_ir(src, SourceId(0)) {
        return Run { out: String::new(), fails: 0, holds: 0, errors, kind: None };
    }
    let (interp, main) = match program_interp_diag(src, release) {
        Ok(x) => x,
        Err(errors) => return Run { out: String::new(), fails: 0, holds: 0, errors, kind: None },
    };
    match exec_main(&interp, &main) {
        Ok(()) => Run {
            out: render_output(&interp),
            fails: interp.fails.get(),
            holds: interp.story.borrow().as_ref().map_or(0, |s| s.fails),
            errors: Vec::new(),
            kind: None,
        },
        Err(e) => {
            let kind = Some(crate::errkind::kind_of(&e));
            let mut d = Diagnostic::error(e);
            if let Some(span) = interp.err_at.get() {
                d = d.with_primary(first_line_of(span, src), "here");
            }
            for n in interp.err_notes.borrow().iter() {
                d = d.with_note(n.clone());
            }
            Run { out: interp.out.borrow().clone(), fails: interp.fails.get(), holds: 0, errors: vec![d], kind }
        }
    }
}

/// A span cut back to the end of its first line. A statement can span many
/// lines (a loop, an `if`), and the caret goes under the line it starts on.
pub fn first_line_of(span: Span, src: &str) -> Span {
    let start = (span.start as usize).min(src.len());
    let end = (span.end as usize).min(src.len()).max(start);
    let cut = src[start..end].find('\n').map_or(end, |n| start + n);
    Span { end: cut.max(start + 1).min(src.len().max(start)) as u32, ..span }
}

/// Like `run`, but also reports how many assertions failed, so callers (the
/// CLI, harnesses, CI) can turn test failures into a nonzero exit code.
pub fn run_checked(src: &str, release: bool) -> Result<(String, u32), String> {
    run_checked_holds(src, release).map(|(out, fails, _)| (out, fails))
}

/// `run_checked`, plus how many of the debug lens's holds failed. A failed
/// hold is recorded in the story and never printed, so the program's output
/// is the same; the caller turns the count into a nonzero exit code.
pub fn run_checked_holds(src: &str, release: bool) -> Result<(String, u32, u32), String> {
    let (interp, main) = program_interp(src, release)?;
    exec_main(&interp, &main)?;
    let fails = interp.fails.get();
    let holds = interp.story.borrow().as_ref().map_or(0, |s| s.fails);
    Ok((render_output(&interp), fails, holds))
}

// ── incremental notebook sessions ───────────────────────────────────────────
// The notebook model re-runs a cell's full prefix on every run — honest, but
// quadratic across a Run-all. A session keeps the interpreter state (globals +
// defs) alive between calls, keyed by the HASH of the source that produced it:
// when a cell's prefix is exactly what the session already executed, only the
// cell itself runs and only its output is returned. Any mismatch (edited
// earlier cell, different notebook, re-run of an already-run cell) falls back
// to one cold rebuild of the prefix. Results are identical to a full re-run by
// construction; an error poisons the session so the next run starts cold.

struct Session {
    hash: u64,
    funcs: HashMap<String, Rc<Func>>,
    globals: HashMap<String, Val>,
    records: Records,
    /// rand()'s MINSTD state where the last cell left it, so the next cell draws
    /// the numbers the whole program would draw there
    rng: f64,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn src_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

pub fn run_cell(prefix: &str, cell: &str) -> Result<String, String> {
    run_cell_keeping(prefix, cell).map_err(|(_, e)| e)
}

/// `run_cell`, keeping what the cell printed before an error stopped it:
/// `Err((printed, error))`, as `run_keeping` does for a whole program.
pub fn run_cell_keeping(prefix: &str, cell: &str) -> Result<String, (String, String)> {
    let none = |e: String| (String::new(), e);
    let want = src_hash(prefix);
    // the NEXT cell's prefix, as the caller will build it (cells joined by \n)
    let next = if prefix.is_empty() {
        src_hash(cell)
    } else {
        src_hash(&format!("{prefix}\n{cell}"))
    };
    SESSION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let continues = matches!(slot.as_ref(), Some(s) if s.hash == want);
        if !continues {
            *slot = None;
            if prefix.trim().is_empty() {
                let rng = fresh_interp(false, HashMap::new(), HashMap::new(), HashMap::new()).rng.get();
                *slot = Some(Session { hash: want, funcs: HashMap::new(), globals: HashMap::new(), records: HashMap::new(), rng });
            } else {
                let (funcs, main, records) = parse_items(prefix, false).map_err(none)?;
                let interp = fresh_interp(false, funcs, HashMap::new(), records);
                exec_main(&interp, &main).map_err(none)?;
                let Interp { funcs, globals, rng, records, .. } = interp;
                *slot = Some(Session { hash: want, funcs, globals: globals.into_inner(), records, rng: rng.get() });
            }
        }
        let sess = slot.take().expect("session just ensured");
        // the earlier cells' defs that return several values, which the cell
        // may take apart (`a, b = f(x)`)
        let known: HashMap<String, usize> =
            sess.records.iter().filter_map(|(name, r)| crate::tuples::tuple_def(name).map(|d| (d.to_string(), r.fields.len()))).collect();
        // the names the earlier cells bound, which a cell's `pi`, `e` and the
        // rest of crate::stdlib's constants must not shadow
        let names: std::collections::HashSet<String> = sess.globals.keys().chain(sess.funcs.keys()).cloned().collect();
        let (new_funcs, main, new_records) =
            crate::stdlib::with_known(names, || crate::tuples::with_known(known, || parse_items(cell, false))).map_err(none)?;
        let mut funcs = sess.funcs;
        for (k, v) in new_funcs {
            funcs.insert(k, v);
        }
        let mut records = sess.records;
        records.extend(new_records);
        let interp = fresh_interp(false, funcs, sess.globals, records);
        interp.rng.set(sess.rng);
        // leave the session poisoned (None) if the cell errors: partially
        // mutated state must not be continued from
        if let Err(e) = exec_main(&interp, &main) {
            return Err((interp.out.borrow().clone(), e));
        }
        let out = render_output(&interp);
        let Interp { funcs, globals, rng, records, .. } = interp;
        *slot = Some(Session { hash: next, funcs, globals: globals.into_inner(), records, rng: rng.get() });
        Ok(out)
    })
}

impl Interp {
    fn tick(&self) -> Result<(), String> {
        let mut s = self.steps.borrow_mut();
        *s += 1;
        if *s > step_budget() {
            return Err(format!("interpreter step budget exceeded (possible infinite loop; {} steps, VIV_STEP_BUDGET raises it)", step_budget()));
        }
        Ok(())
    }

    // Bind a name the same way everywhere: at the top level it's a global,
    // inside a function body it's local. The C and wasm backends both treat a
    // top-level `for` variable as a global that persists after the loop, so the
    // loop variable MUST bind through here too — otherwise a later top-level
    // assignment to that name writes a global while reads still see the stale
    // loop value left in `scope` (the three backends would then disagree).
    /// `name = append(name, x)` or `extend`: the value `x` first, then the
    /// array taken out of its binding, so that its reference count says
    /// whether anything else -- an alias, a closure, a record -- holds it.
    /// Held by nothing else, it grows in place (`Vec` doubling, so a run of
    /// appends is O(n)); otherwise the append copies, and the other holders
    /// keep the array as it was. Either way the program sees a new array.
    fn grow_var(&self, name: &str, callee: &str, x: &Expr, span: Span, scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        let xv = self.eval(x, scope)?;
        let (cur, global) = match scope.remove(name) {
            Some(v) => (v, false),
            None => match self.globals.borrow_mut().remove(name) {
                Some(v) => (v, true),
                None => return Err(format!("undefined variable `{name}`")),
            },
        };
        let r = match &cur {
            // an array of records: a record on its end
            _ if append_record(&cur, &xv).is_some() => append_record(&cur, &xv).expect("checked"),
            // a list of strings: in place when nothing else holds it
            _ if strings::grows(&cur, &xv) => strings::grow(callee, &cur, &xv),
            Val::Arr(rc) if Rc::strong_count(rc) == 1 => growable::grow(callee, &mut rc.borrow_mut(), &xv).map(|()| cur.clone()),
            // an Int array: in place when nothing else holds it
            Val::IArr(rc) if Rc::strong_count(rc) == 1 && callee == "append" => growable::int_tail(&xv).map(|x| {
                rc.borrow_mut().push(x);
                cur.clone()
            }),
            _ => growable::builtin(callee, &[cur.clone(), xv]).map(|v| v.expect("append and extend are growable builtins")),
        };
        if r.is_err() {
            if global {
                self.globals.borrow_mut().insert(name.to_string(), cur);
            } else {
                scope.insert(name.to_string(), cur);
            }
            self.note_err_at(span);
        }
        r
    }

    /// Store `v` at a place: `r.f`, `xs[i]`, `model.blocks[l].attn.Wq`.
    /// Records and arrays of records are values, so each step of the way
    /// copies what another holder still shares before writing into it
    /// (`Rc::make_mut`); a holder never sees the write. The indices are
    /// numbers or variables (the parser computes any other index first).
    fn store_at(&self, place: &Expr, scope: &mut HashMap<String, Val>, top: bool, v: Val) -> Result<(), String> {
        enum Step {
            Field(String),
            Index(i64),
        }
        let mut steps = Vec::new();
        let mut e = place;
        let root = loop {
            match e {
                Expr::Var { name, .. } => break name.clone(),
                Expr::Field { base, field, .. } => {
                    steps.push(Step::Field(field.clone()));
                    e = base;
                }
                Expr::Index { base, indices, .. } => {
                    if indices.len() != 1 {
                        return Err("an array of records is a list, so it takes one index".into());
                    }
                    let i = index_int(self.eval(&indices[0], scope)?.as_num("an index")?)?;
                    steps.push(Step::Index(i));
                    e = base;
                }
                _ => return Err("can only assign into a field of a variable's record".into()),
            }
        };
        steps.reverse();
        // the lowering's rule (`lower::write_root`), checked as the write
        // runs: in a def, the place is rooted at a variable of the def
        if !top {
            let cur = self.call_stack.borrow().last().cloned();
            let param = cur.as_deref().and_then(|c| self.funcs.get(c)).is_some_and(|f| f.params.iter().any(|p| *p == root));
            if param {
                return Err(format!("`{root}` is a parameter, and a def does not change the values it is given"));
            }
            if !scope.contains_key(&root) {
                return Err(format!("`{root}` is a top-level variable, and a def does not change those"));
            }
        }
        fn put(it: &Interp, slot: &mut Val, steps: &[Step], v: Val) -> Result<(), String> {
            let Some((first, rest)) = steps.split_first() else {
                *slot = v;
                return Ok(());
            };
            match (first, slot) {
                (Step::Field(f), Val::Record(rc)) => {
                    let name = rc.name.clone();
                    let want = it.records.get(&*name).and_then(|d| d.fields.iter().find(|(g, _)| g == f).map(|(_, t)| *t));
                    let r = Rc::make_mut(rc);
                    let Some(s) = r.fields.iter_mut().find(|(g, _)| &**g == f.as_str()).map(|(_, s)| s) else {
                        return Err(format!("record `{name}` has no field `{f}`"));
                    };
                    // `[]` where an array of records goes is an empty one
                    let v = match (rest.is_empty(), want, v) {
                        (true, Some(Some(t)), v) if empty_as(t, &v).is_some() => empty_as(t, &v).expect("checked"),
                        // an enum's type is the lowering's to check, as in a literal
                        (true, Some(Some(t)), v) if !matches!(v, Val::Enum(_)) && val_kind(&v) != t.name() => {
                            return Err(format!("the field `{f}` of `{name}` is {} but this value is {}", an(&t.name()), an(&val_kind(&v))))
                        }
                        (_, _, v) => v,
                    };
                    put(it, s, rest, v)
                }
                (Step::Index(i), Val::Recs(rc)) => {
                    let at = it.check_index(&[rc.len()], 0, *i)?;
                    put(it, &mut Rc::make_mut(rc)[at], rest, v)
                }
                (Step::Field(f), other) => Err(format!("cannot access field `{f}` of a {}", val_kind(other))),
                (Step::Index(_), other) => Err(format!("cannot assign into an element of a {}", val_kind(other))),
            }
        }
        if let Some(slot) = scope.get_mut(&root) {
            return put(self, slot, &steps, v);
        }
        if top {
            if let Some(slot) = self.globals.borrow_mut().get_mut(&root) {
                return put(self, slot, &steps, v);
            }
        }
        Err(format!("undefined variable `{root}`"))
    }

    fn bind_var(&self, name: &str, v: Val, scope: &mut HashMap<String, Val>, top: bool) {
        if top {
            self.globals.borrow_mut().insert(name.to_string(), v);
        } else {
            scope.insert(name.to_string(), v);
        }
    }

    // `top` marks execution at the top level (main), where assignments create
    // globals; inside a function body assignments are always local.
    /// Report one assertion. Outside a `test` block this prints its own
    /// PASS/FAIL line, as it always has. Inside one it records against the
    /// test, which prints a single line naming itself when it ends.
    fn report_assert(&self, what: &str, pass: bool, detail: String) {
        if !pass {
            self.fails.set(self.fails.get() + 1);
        }
        if self.test_depth.get() > 0 {
            self.test_asserts.set(self.test_asserts.get() + 1);
            if !pass {
                if self.test_fails.get() == 0 {
                    *self.test_detail.borrow_mut() = format!("{what}: {detail}");
                }
                self.test_fails.set(self.test_fails.get() + 1);
            }
            return;
        }
        let msg = if pass {
            format!("{what} PASS\n")
        } else {
            format!("{what} FAIL: {detail}\n")
        };
        self.out.borrow_mut().push_str(&msg);
    }

    fn test_begin(&self, name: &str) {
        if self.test_depth.get() == 0 {
            *self.test_name.borrow_mut() = name.to_string();
            self.test_asserts.set(0);
            self.test_fails.set(0);
            self.test_detail.borrow_mut().clear();
        }
        self.test_depth.set(self.test_depth.get() + 1);
    }

    fn test_end(&self) {
        let d = self.test_depth.get().saturating_sub(1);
        self.test_depth.set(d);
        if d > 0 {
            return;
        }
        let name = self.test_name.borrow().clone();
        let (fails, n) = (self.test_fails.get(), self.test_asserts.get());
        let line = if fails == 0 && n == 0 {
            format!("PASS {name} (no assertions)\n")
        } else if fails == 0 {
            format!("PASS {name}\n")
        } else if fails == 1 {
            format!("FAIL {name} -- {}\n", self.test_detail.borrow())
        } else {
            format!("FAIL {name} -- {} (and {} more)\n", self.test_detail.borrow(), fails - 1)
        };
        self.out.borrow_mut().push_str(&line);
    }

    /// An error stops the program inside a test block: the test FAILs
    /// (B344), with the same line the C and wasm runtimes print before the
    /// error's message. Nothing when no test is running.
    pub(crate) fn test_stopped(&self) {
        if self.test_depth.get() == 0 {
            return;
        }
        self.test_depth.set(0);
        let line = format!("FAIL {} -- {}\n", self.test_name.borrow(), TEST_STOPPED);
        self.out.borrow_mut().push_str(&line);
    }

    /// A `for` loop's bound. A quotient of two Ints (the pass count of `for i
    /// from a to b by s` with Int bounds) is its exact ceiling, an Int, as
    /// the lowering computes it.
    fn loop_count(&self, bound: &Expr, scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        if let Expr::Binary { op: BinOp::Div, lhs, rhs, span } = bound {
            let l = self.eval(lhs, scope)?;
            let r = self.eval(rhs, scope)?;
            let r = match (&l, &r) {
                (Val::Int(a), Val::Int(b)) => ints::ceildiv(*a, *b).map(Val::Int),
                _ => self.binary(BinOp::Div, l, r),
            };
            if r.is_err() {
                self.note_err_at(*span);
            }
            return r;
        }
        self.eval(bound, scope)
    }

    /// The passes of a `for` loop of `n` passes (`dj`: the loop is pinned,
    /// and each pass says which it is to the record of writes).
    fn run_for(&self, var: &str, n: f64, body: &[Stmt], scope: &mut HashMap<String, Val>, top: bool, dj: bool) -> Result<Flow, String> {
        self.run_for_as(var, n, false, body, scope, top, dj)
    }

    /// `run_for`, with the variable an Int when `int` (an Int bound).
    #[allow(clippy::too_many_arguments)]
    fn run_for_as(&self, var: &str, n: f64, int: bool, body: &[Stmt], scope: &mut HashMap<String, Val>, top: bool, dj: bool) -> Result<Flow, String> {
        let mut i = 0.0_f64;
        while i < n {
            self.tick()?;
            self.bind_var(var, if int { Val::Int(i as i64) } else { Val::Num(i) }, scope, top);
            if dj {
                if let Some(d) = self.dj.borrow_mut().as_mut() {
                    d.pass = i as i64;
                }
            }
            let mark = self.steps_mark();
            let flow = self.exec_block(body, scope, top);
            self.steps_leave(mark);
            self.iteration_end(body, &flow, scope, top)?;
            match flow? {
                Flow::Normal | Flow::Continue => {}
                Flow::Break => break,
                ret @ Flow::Return(_) => return Ok(ret),
            }
            i += 1.0;
        }
        Ok(Flow::Normal)
    }

    /// An element write while a pinned `parallel for` runs: recorded, and a
    /// second pass writing the same element of a pinned array stops the run.
    fn dj_note(&self, arr: &Rc<RefCell<ArrData>>, flat: usize) -> Result<(), String> {
        if !self.dj_on.get() {
            return Ok(());
        }
        let mut dj = self.dj.borrow_mut();
        let Some(d) = dj.as_mut() else { return Ok(()) };
        let pass = d.pass;
        for (a, name, shape, owner) in d.arrays.iter_mut() {
            if !Rc::ptr_eq(a, arr) {
                continue;
            }
            match owner.get(&flat) {
                Some(&p) if p != pass => return Err(dj_error(p, pass, name, shape, flat, &d.what)),
                _ => {
                    owner.insert(flat, pass);
                }
            }
            return Ok(());
        }
        Ok(())
    }

    fn exec_block(&self, body: &[Stmt], scope: &mut HashMap<String, Val>, top: bool) -> Result<Flow, String> {
        for stmt in body {
            match self.exec_stmt(stmt, scope, top)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    /// Record where an error happened, unless an inner expression or
    /// statement already did. Synthetic spans (start = end = 0) point at no
    /// source, so they are passed over and the enclosing node's span is used.
    fn note_err_at(&self, span: Span) {
        if self.err_at.get().is_none() && !(span.start == 0 && span.end == 0) {
            self.err_at.set(Some(span));
        }
    }

    fn exec_stmt(&self, stmt: &Stmt, scope: &mut HashMap<String, Val>, top: bool) -> Result<Flow, String> {
        let r = self.exec_stmt_inner(stmt, scope, top);
        if r.is_err() {
            self.note_err_at(stmt.span());
        }
        r
    }

    fn exec_stmt_inner(&self, stmt: &Stmt, scope: &mut HashMap<String, Val>, top: bool) -> Result<Flow, String> {
        self.tick()?;
        match stmt {
            Stmt::Assign { name, value, .. } => {
                // `a = append(a, x)`: grown in place when nothing else holds `a`
                if let Expr::Call { callee, args, span } = value {
                    if crate::growable::GROW.contains(&callee.as_str())
                        && args.len() == 2
                        && !self.funcs.contains_key(callee.as_str())
                        && matches!(&args[0], Expr::Var { name: n, .. } if n == name)
                    {
                        let v = self.grow_var(name, callee, &args[1], *span, scope)?;
                        self.bind_var(name, v, scope, top);
                        return Ok(Flow::Normal);
                    }
                }
                let v = self.eval(value, scope)?;
                self.bind_var(name, v, scope, top);
                Ok(Flow::Normal)
            }
            Stmt::Return { value, .. } => Ok(Flow::Return(self.eval(value, scope)?)),
            Stmt::ExprStmt { expr, .. } => {
                // A bare string statement is a docstring: documentation, not a value.
                if !matches!(expr, Expr::Str { .. }) {
                    self.eval(expr, scope)?;
                }
                Ok(Flow::Normal)
            }
            Stmt::Narrate { text, .. } => {
                if !text.is_empty() {
                    let mut out = self.out.borrow_mut();
                    out.push_str("narrate: ");
                    out.push_str(text);
                    out.push('\n');
                }
                Ok(Flow::Normal)
            }
            Stmt::Show { expr, .. } => {
                // a hidden variable (a comprehension's, crate::surface) stands
                // for the expression written, which has no label
                let label = match expr {
                    Expr::Var { name, .. } if !name.starts_with(crate::parser::HIDDEN_PREFIX) => name.as_str(),
                    _ => "",
                };
                let v = self.eval(expr, scope)?;
                self.show(label, &v)?;
                Ok(Flow::Normal)
            }
            Stmt::Test { name, body, .. } => {
                self.test_begin(name);
                let r = self.exec_block(body, scope, top);
                if r.is_err() {
                    self.test_stopped();
                } else {
                    self.test_end();
                }
                r
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                if self.eval(cond, scope)?.truthy()? {
                    self.exec_block(then_body, scope, top)
                } else {
                    self.exec_block(else_body, scope, top)
                }
            }
            Stmt::While { cond, body, .. } => {
                while self.eval(cond, scope)?.truthy()? {
                    self.tick()?;
                    let mark = self.steps_mark();
                    let flow = self.exec_block(body, scope, top);
                    self.steps_leave(mark);
                    self.iteration_end(body, &flow, scope, top)?;
                    match flow? {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => break,
                        ret @ Flow::Return(_) => return Ok(ret),
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::For { var, bound, body, parallel, span, .. } => {
                let b = self.loop_count(bound, scope)?;
                let int = matches!(b, Val::Int(_));
                let n = b.as_num("a loop bound")?;
                // a `parallel for` a memory lens pins: a debug run records
                // which pass writes each element of the pinned arrays
                let pin = if *parallel && !self.release { self.pinned.borrow().iter().find(|p| p.span == *span).cloned() } else { None };
                // a pinned loop inside another keeps the outer one's record aside
                let outer = if pin.is_some() { self.dj.borrow_mut().take() } else { None };
                if let Some(p) = &pin {
                    let mut arrays = Vec::new();
                    for a in &p.arrays {
                        if let Val::Arr(arr) = self.eval(&Expr::Var { name: a.clone(), span: *span }, scope)? {
                            let shape = arr.borrow().shape.clone();
                            arrays.push((arr, a.clone(), shape, HashMap::new()));
                        }
                    }
                    *self.dj.borrow_mut() = Some(Dj { what: p.what.clone(), pass: -1, arrays });
                    self.dj_on.set(true);
                }
                let r = self.run_for_as(var, n, int, body, scope, top, pin.is_some());
                if pin.is_some() {
                    self.dj_on.set(outer.is_some());
                    *self.dj.borrow_mut() = outer;
                }
                r
            }
            Stmt::SetIndex { base, indices, value, .. } => {
                // The compiled store is one call, `viv_setN(a, i, j, value)`:
                // its indices and its value are computed first, and the
                // indices are checked inside it. So here too: the index
                // expressions, then the value (a read out of bounds in
                // `a[2] = a[2] + 1` is the read's error), then the check.
                let base_val = self.eval(base, scope)?;
                // `xs[i] = r` of an array of records: a value, written
                // through its place
                if let Val::Recs(_) = &base_val {
                    drop(base_val);
                    let v = self.eval(value, scope)?;
                    let place = Expr::Index { base: base.clone(), indices: indices.clone(), span: stmt.span() };
                    return self.store_at(&place, scope, top, v).map(|()| Flow::Normal);
                }
                // `m[k] = v`: the key, then the value, then the store
                if let Val::Map(m) = &base_val {
                    if indices.len() != 1 {
                        return Err(maps::ONE_KEY.into());
                    }
                    let k = self.eval(&indices[0], scope)?;
                    let v = self.eval(value, scope)?;
                    maps::store(m, &k, &v)?;
                    return Ok(Flow::Normal);
                }
                // (an array has rank 4 at most; more indices is a rank error)
                let mut fixed = [0.0f64; 8];
                for (k, ix) in indices.iter().enumerate() {
                    let v = self.eval(ix, scope)?.as_num("an index")?;
                    if k < fixed.len() {
                        fixed[k] = v;
                    }
                }
                let fixed = &fixed[..indices.len().min(8)];
                let v = self.eval(value, scope)?;
                match base_val {
                    Val::PArr(arr) => {
                        let k = arr.borrow().k;
                        let n = f32s::count(k, &arr.borrow().b);
                        let idx = self.flat_index_of(&[n], indices.len(), fixed)?;
                        let e = match v {
                            Val::F32(x) => f32s::Elt::F(x),
                            Val::Int(i) => f32s::Elt::I(i),
                            v => return Err(format!("a packed array holds Float32s or Ints, and this is {}", crate::lower::a_or_an(&val_kind(&v)))),
                        };
                        f32s::set(k, &mut arr.borrow_mut().b, idx, e)?;
                    }
                    Val::IArr(arr) => {
                        let n = arr.borrow().len();
                        let idx = self.flat_index_of(&[n], indices.len(), fixed)?;
                        let Val::Int(x) = v else { return Err(format!("an Int array holds Ints, and this is {}", crate::lower::a_or_an(&val_kind(&v)))) };
                        arr.borrow_mut()[idx] = x;
                    }
                    Val::Arr(arr) => {
                        let idx = self.flat_index_of(&arr.borrow().shape, indices.len(), fixed)?;
                        let v = match v {
                            Val::Cplx(..) => {
                                return Err("cannot store a complex number in a real array -- allocate it with `czeros`".into())
                            }
                            v => v.as_num("an array element")?,
                        };
                        self.dj_note(&arr, idx)?;
                        arr.borrow_mut().data[idx] = v;
                    }
                    Val::CArr(arr) => {
                        let idx = self.flat_index_of(&arr.borrow().shape, indices.len(), fixed)?;
                        // a real promotes, exactly as it does in scalar arithmetic
                        let (re, im) = match v {
                            Val::Cplx(re, im) => (re, im),
                            v => (v.as_num("a complex array element")?, 0.0),
                        };
                        let mut ab = arr.borrow_mut();
                        ab.data[2 * idx] = re;
                        ab.data[2 * idx + 1] = im;
                    }
                    Val::Str(_) => return Err(strings::STR_IMMUTABLE.into()),
                    Val::Strs(_) => return Err(strings::STRS_IMMUTABLE.into()),
                    _ => return Err("cannot index-assign into a non-array".into()),
                }
                Ok(Flow::Normal)
            }
            // `r.f = v`, `model.blocks[l].attn.Wq = W`: the value, then the
            // place
            Stmt::SetField { base, field, field_span, value, span } => {
                let v = self.eval(value, scope)?;
                let place = Expr::Field { base: base.clone(), field: field.clone(), field_span: *field_span, span: *span };
                self.store_at(&place, scope, top, v)?;
                Ok(Flow::Normal)
            }
            Stmt::Break { .. } => Ok(Flow::Break),
            Stmt::Continue { .. } => Ok(Flow::Continue),
            Stmt::StepEnter { def, id, depth, .. } => {
                if !self.release {
                    self.step_seen.set(true);
                    self.step_event('+', def, id, *depth);
                    self.step_open.borrow_mut().push((def.clone(), id.clone(), *depth));
                }
                Ok(Flow::Normal)
            }
            Stmt::StepExit { .. } => {
                if !self.release {
                    let open = self.step_open.borrow().len();
                    self.steps_close_to(open.saturating_sub(1));
                }
                Ok(Flow::Normal)
            }
            Stmt::Lens { lens, .. } => self.exec_lens(lens, scope, top),
        }
    }

    fn step_event(&self, kind: char, def: &str, id: &str, depth: u32) {
        let total = self.step_total.get();
        if (total as usize) < STEP_CAP {
            self.step_events.borrow_mut().push(format!("{kind}{def}/{id}@{depth}"));
        }
        self.step_total.set(total + 1);
        if self.story.borrow().is_some() {
            let kind = if kind == '+' { "enter" } else { "exit" };
            self.story_event(|| format!("[\"{kind}\",\"{def}\",\"{id}\",{depth}]"));
        }
    }

    /// Append one event to the story, if one is being recorded and it has
    /// room. `make` runs only for a kept event.
    fn story_event(&self, make: impl FnOnce() -> String) {
        let mut st = self.story.borrow_mut();
        let Some(st) = st.as_mut() else { return };
        st.total += 1;
        if st.total <= STORY_CAP {
            if st.total > 1 {
                st.buf.push(',');
            }
            st.buf.push_str(&make());
        }
    }

    /// A statement of the debug lens (`crate::debug_lens`), or a placeholder
    /// body. The lens statements exist only in a program instrumented for a
    /// debug run, so they always record.
    fn exec_lens(&self, lens: &Lens, scope: &mut HashMap<String, Val>, top: bool) -> Result<Flow, String> {
        match lens {
            Lens::Placeholder { def } => Err(format!("`{def}` is a placeholder: it has a signature and no body yet")),
            Lens::Stop { message, kind } => Err(crate::errkind::kinded(*kind, message.clone())),
            Lens::Call { def } => {
                self.story_event(|| format!("[\"call\",\"{def}\"]"));
                Ok(Flow::Normal)
            }
            Lens::Ret { def } => {
                self.story_event(|| format!("[\"ret\",\"{def}\"]"));
                Ok(Flow::Normal)
            }
            Lens::Inline { def, enter, .. } => {
                self.inline_mark(def, *enter);
                Ok(Flow::Normal)
            }
            Lens::Iter { def, counter, .. } => {
                let n = self.eval(&Expr::Var { name: counter.clone(), span: knotkit::span::Span::synthetic(SourceId(0)) }, scope)?.as_num("an iteration count")? + 1.0;
                self.bind_var(counter, Val::Num(n), scope, top);
                self.story_event(|| format!("[\"iter\",\"{def}\",{}]", n as i64));
                Ok(Flow::Normal)
            }
            Lens::IterEnd { def } => {
                self.story_event(|| format!("[\"iter_end\",\"{def}\"]"));
                Ok(Flow::Normal)
            }
            // the debug lens turns each one into an assignment before a run
            Lens::Bind { name, .. } => Err(format!("internal: the debug lens left its binding of `{name}` in the program")),
            Lens::Watch { def, at, names, vars } => {
                let mut vals = Vec::with_capacity(names.len());
                for (k, (name, span)) in names.iter().enumerate() {
                    let v = self.eval(&Expr::Var { name: vars.get(k).unwrap_or(name).clone(), span: *span }, scope)?;
                    vals.push((name.as_str(), story_value(name, &v)?));
                }
                self.story_event(|| {
                    let fields: Vec<String> = vals.iter().map(|(n, v)| format!("\"{n}\":{v}")).collect();
                    format!("[\"watch\",\"{def}\",\"{at}\",{{{}}}]", fields.join(","))
                });
                Ok(Flow::Normal)
            }
            Lens::Hold { def, at, check, sentence } => {
                let (ok, worst) = match check {
                    HoldCheck::Cond(e) => (self.eval(e, scope)?.truthy()?, None),
                    HoldCheck::Near { diff, tol } => {
                        let worst = match self.eval(diff, scope)? {
                            Val::Num(d) => d.abs(),
                            Val::Arr(a) => max_abs(&a.borrow().data),
                            _ => return Err("a hold with `within` compares numbers or real arrays".into()),
                        };
                        let tol = self.eval(tol, scope)?.as_num("a hold's tolerance")?;
                        (worst <= tol, Some(worst))
                    }
                };
                if !ok {
                    if let Some(st) = self.story.borrow_mut().as_mut() {
                        st.fails += 1;
                    }
                }
                self.story_event(|| {
                    let w = worst.map(story_num).unwrap_or_else(|| "null".to_string());
                    format!("[\"hold\",\"{def}\",\"{at}\",{ok},{sentence},{w}]")
                });
                Ok(Flow::Normal)
            }
        }
    }

    /// The end of one iteration of an iteration loop: its per-iteration
    /// watches and holds, then the iteration's end, however the body was left.
    /// The steps the body left open are closed first, as the C backend does.
    fn iteration_end(&self, body: &[Stmt], flow: &Result<Flow, String>, scope: &mut HashMap<String, Val>, top: bool) -> Result<(), String> {
        if flow.is_err() {
            return Ok(());
        }
        if let Some(Stmt::Lens { lens, .. }) = body.first() {
            if let Lens::Iter { end, .. } = &**lens {
                self.exec_block(end, scope, top)?;
            }
        }
        Ok(())
    }

    /// A region `crate::grad` inlined (ast `Lens::Inline`): its entry is a
    /// call of the def it came from in the call graph, as a call of that def
    /// is, and its exit the return.
    fn inline_mark(&self, def: &str, enter: bool) {
        if !enter {
            self.call_stack.borrow_mut().pop();
            return;
        }
        let caller = self.call_stack.borrow().last().cloned().unwrap_or_else(|| "<main>".to_string());
        *self.call_counts.borrow_mut().entry((caller.clone(), def.to_string())).or_insert(0) += 1;
        {
            let mut trace = self.call_trace.borrow_mut();
            if trace.len() < TRACE_CAP {
                trace.push((caller, def.to_string()));
            }
        }
        self.call_stack.borrow_mut().push(def.to_string());
    }

    /// How many steps are open, before a loop body or a call runs.
    fn steps_mark(&self) -> usize {
        if self.step_seen.get() { self.step_open.borrow().len() } else { 0 }
    }

    /// After the loop body or call: close the steps it left open.
    fn steps_leave(&self, mark: usize) {
        if self.step_seen.get() {
            self.steps_close_to(mark);
        }
    }

    /// Close every step opened after the first `n`, innermost first. A step's
    /// body can be left early -- `return` from the def, `break` or `continue`
    /// out of a loop around it -- and its exit is recorded all the same; the
    /// C backend emits the same exits before the same jumps.
    fn steps_close_to(&self, n: usize) {
        loop {
            let top = {
                let mut open = self.step_open.borrow_mut();
                if open.len() <= n {
                    return;
                }
                open.pop()
            };
            if let Some((def, id, depth)) = top {
                self.step_event('-', &def, &id, depth);
            }
        }
    }

    fn eval(&self, expr: &Expr, scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        let r = self.eval_inner(expr, scope);
        if r.is_err() {
            self.note_err_at(expr.span());
        }
        // an Int where the lowering bound a number (a def's parameter, a
        // field, a return, a number variable): the number it is, exactly
        if let Ok(Val::Int(i)) = r {
            let sp = expr.span();
            if self.ints.borrow().nums.contains(&(sp.start, sp.end)) {
                return Ok(Val::Num(ints::to_num(i)?));
            }
        }
        r
    }

    fn eval_inner(&self, expr: &Expr, scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        match expr {
            Expr::Num { value, span } => {
                // a literal the lowering computed as an Int
                if let Some(&v) = self.ints.borrow().lits.get(&(span.start, span.end)) {
                    // the number it holds, when exact; else the digits' value
                    let exact = value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0;
                    return Ok(Val::Int(if exact { *value as i64 } else { v }));
                }
                // a literal the lowering made a Float32
                if self.ints.borrow().f32.contains(&(span.start, span.end)) {
                    return Ok(Val::F32(*value as f32));
                }
                Ok(Val::Num(*value))
            }
            Expr::Var { name, .. } => scope
                .get(name)
                .cloned()
                .or_else(|| self.globals.borrow().get(name).cloned())
                // A bare def name is a function value: `rk4(rhs, y0, ...)`.
                .or_else(|| self.funcs.get(name).map(|f| Val::Func(name.clone(), f.clone())))
                .ok_or_else(|| format!("undefined variable `{name}`")),
            Expr::Imag { value, .. } => Ok(Val::Cplx(0.0, *value)),
            Expr::Bool { value, .. } => Ok(Val::Bool(*value)),
            Expr::Neg { operand, .. } => match self.eval(operand, scope)? {
                Val::Int(i) => Ok(Val::Int(ints::neg(i)?)),
                Val::F32(x) => Ok(Val::F32(-x)),
                Val::Cplx(re, im) => Ok(Val::Cplx(-re, -im)),
                // an array, real or complex: every element, scaled by -1 (what the C does)
                Val::Arr(a) => {
                    let ab = a.borrow();
                    Ok(new_arr(ab.shape.clone(), ab.data.iter().map(|x| x * -1.0).collect()))
                }
                Val::CArr(a) => {
                    let ab = a.borrow();
                    Ok(new_carr(ab.shape.clone(), ab.data.iter().map(|x| x * -1.0).collect()))
                }
                // a record: each of its arrays negated, as `-a` negates one
                v @ (Val::Record(_) | Val::Recs(_)) => self.rec_zip("`-`", &[v], &mut |_, x| {
                    Ok(match &x[0] {
                        Val::Arr(a) => new_arr(a.borrow().shape.clone(), a.borrow().data.iter().map(|x| x * -1.0).collect()),
                        Val::CArr(a) => new_carr(a.borrow().shape.clone(), a.borrow().data.iter().map(|x| x * -1.0).collect()),
                        other => other.clone(),
                    })
                }),
                v => Ok(Val::Num(-v.as_num("negation")?)),
            },
            Expr::Not { operand, .. } => {
                Ok(Val::Bool(!self.eval(operand, scope)?.truthy()?))
            }
            Expr::Binary { op, lhs, rhs, span } => {
                // the digits lens (crate::digits): `a*b + c` fused when the
                // run contracts, and the site of every operation
                if crate::digits::on() && matches!(op, BinOp::Add | BinOp::Sub) {
                    if let Some(v) = self.fused_add(*op, lhs, rhs, span.start, scope)? {
                        return Ok(v);
                    }
                }
                let l = self.eval(lhs, scope)?;
                // `and` / `or` short-circuit, as in C: the right side runs only
                // when the left one does not decide the answer.
                match (op, &l) {
                    (BinOp::And, Val::Bool(false)) => return Ok(Val::Bool(false)),
                    (BinOp::Or, Val::Bool(true)) => return Ok(Val::Bool(true)),
                    _ => {}
                }
                let r = self.eval(rhs, scope)?;
                if crate::digits::on() {
                    crate::digits::at(span.start);
                    crate::digits::counting(counting_expr(lhs) && counting_expr(rhs));
                }
                self.binary(*op, l, r)
            }
            // `[]` where the lowering made an empty Int array
            // an Int array's literal (the lowering made every element an Int)
            Expr::ArrLit { elems, span } if self.ints.borrow().arrs.contains(&(span.start, span.end)) => {
                let mut xs = Vec::with_capacity(elems.len());
                for e in elems {
                    match self.eval(e, scope)? {
                        Val::Int(i) => xs.push(i),
                        v => return Err(format!("an Int array holds Ints, not a {}", val_kind(&v))),
                    }
                }
                Ok(Val::IArr(Rc::new(RefCell::new(xs))))
            }
            Expr::ArrLit { elems, .. } => self.array_literal(elems, scope),
            Expr::Index { base, indices, .. } => {
                match self.eval(base, scope)? {
                    Val::IArr(a) => {
                        let n = a.borrow().len();
                        let idx = self.flat_index(&[n], indices, scope)?;
                        Ok(Val::Int(a.borrow()[idx]))
                    }
                    Val::PArr(a) => {
                        let (k, n) = (a.borrow().k, f32s::count(a.borrow().k, &a.borrow().b));
                        let idx = self.flat_index(&[n], indices, scope)?;
                        Ok(match f32s::get(k, &a.borrow().b, idx) {
                            f32s::Elt::F(x) => Val::F32(x),
                            f32s::Elt::I(i) => Val::Int(i),
                        })
                    }
                    // the shape is copied out before the indices are evaluated:
                    // an index may call a def that writes into this array (B303)
                    Val::Arr(a) => {
                        let shape = a.borrow().shape.clone();
                        let idx = self.flat_index(&shape, indices, scope)?;
                        Ok(Val::Num(a.borrow().data[idx]))
                    }
                    Val::CArr(a) => {
                        let shape = a.borrow().shape.clone();
                        let idx = self.flat_index(&shape, indices, scope)?;
                        let ab = a.borrow();
                        Ok(Val::Cplx(ab.data[2 * idx], ab.data[2 * idx + 1]))
                    }
                    // a string's character, a list's string: Python's index,
                    // negative from the end
                    Val::Str(st) => {
                        let i = self.one_index(indices, scope, "a string")?;
                        let v = st.view();
                        let (a, b) = strings::char_at(&v, i)?;
                        Ok(strings::sval(st.text[a..b].to_string()))
                    }
                    Val::Strs(l) => {
                        let i = self.one_index(indices, scope, "a list of strings")?;
                        let l = l.borrow();
                        let k = strings::py_index(l.len(), i).ok_or_else(|| crate::errkind::kinded(crate::errkind::ErrKind::Index, "list index out of range"))?;
                        Ok(Val::Str(l[k].clone()))
                    }
                    // a boxed value array: one integer index returns the stored Val
                    Val::VArr(v) => {
                        if indices.len() != 1 {
                            return Err("a value array takes a single index".into());
                        }
                        // read and checked as any other index is (flat_index), so
                        // `methods[-1]` is the last, as in the compiled program
                        let i = index_int(self.eval(&indices[0], scope)?.as_num("a value-array index")?)?;
                        let n = v.borrow().len();
                        let at = self.check_index(&[n], 0, i)?;
                        Ok(v.borrow()[at].clone())
                    }
                    // `m[k]`: one key, a number or a string
                    Val::Map(m) => {
                        if indices.len() != 1 {
                            return Err(maps::ONE_KEY.into());
                        }
                        let k = self.eval(&indices[0], scope)?;
                        maps::index(&m, &k)
                    }
                    // an element of an array of records: read and checked as
                    // any index is
                    Val::Recs(v) => {
                        if indices.len() != 1 {
                            return Err("an array of records is a list, so it takes one index".into());
                        }
                        let i = index_int(self.eval(&indices[0], scope)?.as_num("an index")?)?;
                        let at = self.check_index(&[v.len()], 0, i)?;
                        Ok(v[at].clone())
                    }
                    _ => Err("cannot index into a non-array".into()),
                }
            }
            Expr::Call { callee, args, span } => {
                let v = if crate::digits::on() {
                    // the digits lens perturbs reductions in the array
                    // builtins, so a slice is reduced the ordinary way here
                    crate::digits::push_call(span.start);
                    let v = self.call(callee, args, scope);
                    crate::digits::pop_call();
                    v?
                } else {
                    match self.view_reduce(callee, args, scope)? {
                        Some(v) => v,
                        None => self.call(callee, args, scope)?,
                    }
                };
                // a call the lowering computed as an Int: its number, exactly
                if !matches!(v, Val::Int(_)) && self.ints.borrow().calls.contains(&(span.start, span.end)) {
                    return Ok(Val::Int(ints::of_num(v.as_num("an Int")?)?));
                }
                Ok(v)
            }
            Expr::Str { value, .. } => Ok(strings::sval(value.clone())),
            Expr::Lambda { params, body, .. } => Ok(Val::Closure(Rc::new(Closure {
                params: params.clone(),
                body: (**body).clone(),
                captured: scope.clone(),
            }))),
            Expr::RecordLit { type_name, fields, .. } => self.record_literal(type_name, fields, scope),
            Expr::Field { base, field, .. } => match self.eval(base, scope)? {
                Val::Record(r) => r.get(field).cloned().ok_or_else(|| format!("record `{}` has no field `{field}`", r.name)),
                other => Err(format!("cannot access field `{field}` of a {}", val_kind(&other))),
            },
        }
    }

    /// A reduction of slices read in place (crate::fuse; the compiled engines'
    /// `viv_fz_red`): `sum`, `sum(.., ordered=true)`, `prod`, `mean`, `norm`
    /// of a slice, `dot` of slices and arrays, in the builtins' own orders.
    /// The slice's base and bounds are plain values; anything else (a bound
    /// out of range, a slice not evenly spaced, shapes `dot` refuses) is left
    /// to the builtin, which copies the slice and says what is wrong.
    fn view_reduce(&self, callee: &str, args: &[Expr], scope: &mut HashMap<String, Val>) -> Result<Option<Val>, String> {
        let kind = match (callee, args.len()) {
            ("sum", 1) => 0,
            ("__sum_ordered", 1) => 1,
            ("prod", 1) => 2,
            ("mean", 1) => 3,
            ("norm", 1) => 4,
            ("dot", 2) => 5,
            _ => return Ok(None),
        };
        let is_slice = |a: &Expr| matches!(a, Expr::Call { callee: c, args: sa, .. } if c == slices::GET && sa.len() == 2 && crate::fuse::quiet_ast(&sa[0]) && matches!(&sa[1], Expr::ArrLit { elems, .. } if elems.iter().all(crate::fuse::quiet_ast)));
        if self.funcs.contains_key(callee) || !args.iter().any(is_slice) || !args.iter().all(|a| is_slice(a) || matches!(a, Expr::Var { .. })) {
            return Ok(None);
        }
        let mut views = Vec::new();
        for a in args {
            let (base, spec) = match a {
                Expr::Call { args: sa, .. } => (self.eval(&sa[0], scope)?, Some(self.eval(&sa[1], scope)?)),
                _ => (self.eval(a, scope)?, None),
            };
            let Val::Arr(b) = base else { return Ok(None) };
            let (start, st, shape) = match spec {
                Some(Val::Arr(sp)) => {
                    let ax = match slices::axes(&b.borrow().shape, &sp.borrow().data) {
                        Ok(ax) => ax,
                        Err(_) => return Ok(None),
                    };
                    let Some((start, st)) = ax.affine() else { return Ok(None) };
                    (start, st, ax.shape)
                }
                Some(_) => return Ok(None),
                None => {
                    let shape = b.borrow().shape.clone();
                    (0, 1, shape)
                }
            };
            views.push((b, start, st, shape));
        }
        if kind == 5 && (views[0].3.len() != 1 || views[1].3.len() != 1 || views[0].3 != views[1].3) {
            return Ok(None);
        }
        let n: usize = views[0].3.iter().product();
        let at = |v: &(Rc<RefCell<ArrData>>, i64, i64, Vec<usize>), k: usize| -> f64 { v.0.borrow().data[(v.1 + k as i64 * v.2) as usize] };
        let gathered = |v: &(Rc<RefCell<ArrData>>, i64, i64, Vec<usize>)| -> Vec<f64> { (0..n).map(|k| at(v, k)).collect() };
        let t = match kind {
            0 | 3 => {
                let v = &views[0];
                let s = if v.2 == 1 && n > 0 { arrays::npsum(&v.0.borrow().data[v.1 as usize..v.1 as usize + n]) } else { arrays::npsum(&gathered(v)) };
                if kind == 3 { s / n as f64 } else { s }
            }
            1 => (0..n).fold(0.0, |t, k| t + at(&views[0], k)),
            2 => (0..n).fold(1.0, |t, k| t * at(&views[0], k)),
            4 => (0..n).fold(0.0, |t, k| {
                let x = at(&views[0], k);
                t + x * x
            })
            .sqrt(),
            _ => (0..n).fold(0.0, |t, k| t + at(&views[0], k) * at(&views[1], k)),
        };
        Ok(Some(Val::Num(t)))
    }

    /// `Name { field: value, ... }`, checked by the lowering's rules and
    /// with its messages. The values are computed in declaration order,
    /// whatever order they are written in: the compiled engines evaluate the
    /// struct's fields in that order, so a `rand()` in two fields draws the
    /// same numbers into the same fields everywhere.
    fn record_literal(&self, type_name: &str, fields: &[(String, Span, Expr)], scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        let decl = self.records.get(type_name).cloned().ok_or_else(|| format!("unknown record type `{type_name}`"))?;
        for (i, (fname, _, _)) in fields.iter().enumerate() {
            if !decl.fields.iter().any(|(d, _)| d == fname) {
                return Err(format!("record `{type_name}` has no field `{fname}`"));
            }
            if fields[..i].iter().any(|(g, _, _)| g == fname) {
                return Err(format!("field `{fname}` is set more than once"));
            }
        }
        let mut out = Vec::with_capacity(decl.fields.len());
        for (dfield, dty) in &decl.fields {
            let Some((_, _, value)) = fields.iter().find(|(g, _, _)| g == dfield) else {
                return Err(format!("missing field `{dfield}` in `{type_name}` literal"));
            };
            let mut v = self.eval(value, scope)?;
            // `[]` where an array of records, a list of strings or an Int
            // array goes is an empty one of that kind (B341)
            if let Some(e) = (*dty).and_then(|t| empty_as(t, &v)) {
                v = e;
            }
            // an enum's type is the lowering's to check: an optional's value
            // does not know which optional it is
            if let (Some(want), false) = (dty, matches!(v, Val::Enum(_))) {
                let got = val_kind(&v);
                if got != want.name() {
                    return Err(format!("field `{dfield}` of `{type_name}` is {} but this value is {}", an(&want.name()), an(&got)));
                }
            }
            out.push((Rc::from(dfield.as_str()), v));
        }
        Ok(Val::Record(Rc::new(RecVal { name: Rc::from(type_name), fields: out })))
    }

    /// A new value with the shape of the records among `vals` (the first
    /// gives it; numbers stand for every leaf): each array leaf is `leaf` of
    /// the leaves at that place, each record and array of records walked in
    /// declaration order. Each record is a record of arrays: any other field
    /// is refused, as the lowering refuses it (`lower::records`). `word`
    /// names the operation in the errors.
    fn rec_zip(&self, word: &str, vals: &[Val], leaf: &mut dyn FnMut(&Interp, &[Val]) -> Result<Val, String>) -> Result<Val, String> {
        let first = vals.iter().find(|v| matches!(v, Val::Record(_) | Val::Recs(_))).cloned().expect("a record among the operands");
        match first {
            Val::Recs(xs) => {
                let mut parts: Vec<&Vec<Val>> = Vec::new();
                for v in vals {
                    match v {
                        Val::Recs(ys) => {
                            if ys.len() != xs.len() {
                                return Err(crate::lower::rec_lengths_differ(word, &rec_name(vals)));
                            }
                            parts.push(ys);
                        }
                        Val::Num(_) => {}
                        other => return Err(format!("{word} takes records of one type, or a record and a number; this is {}", crate::lower::a_or_an(&val_kind(other)))),
                    }
                }
                let mut out = Vec::with_capacity(xs.len());
                for k in 0..xs.len() {
                    let row: Vec<Val> = vals.iter().map(|v| if let Val::Recs(ys) = v { ys[k].clone() } else { v.clone() }).collect();
                    out.push(self.rec_zip(word, &row, leaf)?);
                }
                let _ = parts;
                Ok(Val::Recs(Rc::new(out)))
            }
            Val::Record(r) => {
                for v in vals {
                    match v {
                        Val::Record(s) if s.name == r.name => {}
                        Val::Num(_) => {}
                        other => return Err(format!("{word} takes records of one type, or a record and a number; the first is {} and this is {}", crate::lower::a_or_an(&r.name), crate::lower::a_or_an(&val_kind(other)))),
                    }
                }
                let mut fields = Vec::with_capacity(r.fields.len());
                for (k, (fname, fv)) in r.fields.iter().enumerate() {
                    let at: Vec<Val> = vals.iter().map(|v| if let Val::Record(s) = v { s.fields[k].1.clone() } else { v.clone() }).collect();
                    let nv = match fv {
                        Val::Arr(_) => leaf(self, &at)?,
                        Val::CArr(_) if matches!(word, "`+`" | "`-`" | "`*`" | "`/`") => leaf(self, &at)?,
                        Val::Record(_) | Val::Recs(_) => self.rec_zip(word, &at, leaf)?,
                        other => return Err(crate::lower::not_arrays(word, &r.name, fname, &val_kind(other))),
                    };
                    fields.push((fname.clone(), nv));
                }
                Ok(Val::Record(Rc::new(RecVal { name: r.name.clone(), fields })))
            }
            _ => unreachable!("found above"),
        }
    }

    /// `zeros_like`, `sum`, an elementwise math function and `for_leaves`,
    /// when their argument is a record or an array of records; `None` when it
    /// is not.
    fn record_builtin(&self, callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
        let rec = |v: &Val| matches!(v, Val::Record(_) | Val::Recs(_));
        if callee == "for_leaves" && a.len() >= 2 && !self.funcs.contains_key("for_leaves") {
            let Val::Func(name, func) = &a[0] else {
                return Err("`for_leaves` takes a def first: `for_leaves(f, r1, r2, ...)`".into());
            };
            let k = a[1..].iter().take_while(|v| rec(v)).count();
            if k == 0 {
                return Err("`for_leaves` takes a def, then records of one type".into());
            }
            let (recs, extra) = a[1..].split_at(k);
            let mut rows: Vec<Vec<Val>> = Vec::new();
            self.rec_leaf_rows(recs, &mut rows)?;
            for row in rows {
                let mut args = row;
                args.extend(extra.iter().cloned());
                self.call_user(name, func, args)?;
            }
            return Ok(Some(Val::Num(0.0)));
        }
        if a.len() != 1 || !rec(&a[0]) || self.funcs.contains_key(callee) {
            return Ok(None);
        }
        if callee == "sum" {
            return self.rec_sum(&a[0]).map(|s| Some(Val::Num(s)));
        }
        if callee == "zeros_like" {
            return self
                .rec_zip("`zeros_like`", &a[..1], &mut |_, x| {
                    Ok(match &x[0] {
                        Val::Arr(v) => new_arr(v.borrow().shape.clone(), vec![0.0; v.borrow().data.len()]),
                        other => other.clone(),
                    })
                })
                .map(Some);
        }
        if let Some(f) = elementwise_fn(callee) {
            let word = format!("`{callee}`");
            return self
                .rec_zip(&word, &a[..1], &mut |_, x| match &x[0] {
                    Val::Arr(v) => {
                        let b = v.borrow();
                        Ok(new_arr(b.shape.clone(), b.data.iter().map(|e| f(*e)).collect()))
                    }
                    other => Ok(other.clone()),
                })
                .map(Some);
        }
        Ok(None)
    }

    /// `sum(r)`: the sums of `r`'s fields added in declaration order, a
    /// record field's (and each record of an array of records') own sum
    /// computed first, as the C and wasm helpers add them.
    fn rec_sum(&self, v: &Val) -> Result<f64, String> {
        let mut total = 0.0;
        match v {
            Val::Recs(xs) => {
                for x in xs.iter() {
                    total += self.rec_sum(x)?;
                }
            }
            Val::Record(r) => {
                for (name, f) in &r.fields {
                    match f {
                        Val::Arr(a) => total += arrays::npsum(&a.borrow().data),
                        Val::Record(_) | Val::Recs(_) => total += self.rec_sum(f)?,
                        other => return Err(crate::lower::not_arrays("`sum`", &r.name, name, &val_kind(other))),
                    }
                }
            }
            _ => unreachable!("a record"),
        }
        Ok(total)
    }

    /// The array leaves of `recs` (records of one type), one row per leaf in
    /// declaration order, each row holding that leaf of every record: what
    /// `for_leaves` calls its def with.
    fn rec_leaf_rows(&self, recs: &[Val], rows: &mut Vec<Vec<Val>>) -> Result<(), String> {
        match &recs[0] {
            Val::Recs(xs) => {
                for v in recs {
                    match v {
                        Val::Recs(ys) if ys.len() == xs.len() => {}
                        Val::Recs(_) => return Err(crate::lower::rec_lengths_differ("`for_leaves`", &rec_name(recs))),
                        other => return Err(format!("`for_leaves` takes records of one type; this is {}", crate::lower::a_or_an(&val_kind(other)))),
                    }
                }
                for k in 0..xs.len() {
                    let row: Vec<Val> = recs.iter().map(|v| if let Val::Recs(ys) = v { ys[k].clone() } else { unreachable!() }).collect();
                    self.rec_leaf_rows(&row, rows)?;
                }
                Ok(())
            }
            Val::Record(r) => {
                for v in recs {
                    match v {
                        Val::Record(s) if s.name == r.name => {}
                        other => return Err(format!("`for_leaves` takes records of one type; the first is {} and this is {}", crate::lower::a_or_an(&r.name), crate::lower::a_or_an(&val_kind(other)))),
                    }
                }
                for (k, (name, f)) in r.fields.iter().enumerate() {
                    let at: Vec<Val> = recs.iter().map(|v| if let Val::Record(s) = v { s.fields[k].1.clone() } else { unreachable!() }).collect();
                    match f {
                        Val::Arr(_) => rows.push(at),
                        Val::Record(_) | Val::Recs(_) => self.rec_leaf_rows(&at, rows)?,
                        other => return Err(crate::lower::not_arrays("`for_leaves`", &r.name, name, &val_kind(other))),
                    }
                }
                Ok(())
            }
            other => Err(format!("`for_leaves` takes records of one type; this is {}", crate::lower::a_or_an(&val_kind(other)))),
        }
    }

    /// The digits lens's FMA contraction: `a*b + c`, `c + a*b`, `a*b - c`
    /// and `c - a*b` over numbers, or arrays of one shape, in one rounding
    /// when this run contracts (crate::digits). `None`: not that shape, or
    /// the run leaves it alone; the caller evaluates it as written.
    fn fused_add(&self, op: BinOp, lhs: &Expr, rhs: &Expr, at: u32, scope: &mut HashMap<String, Val>) -> Result<Option<Val>, String> {
        let is_mul = |e: &Expr| matches!(e, Expr::Binary { op: BinOp::Mul, .. });
        let (prod, other, prod_left) = if is_mul(lhs) {
            (lhs, rhs, true)
        } else if is_mul(rhs) {
            (rhs, lhs, false)
        } else {
            return Ok(None);
        };
        crate::digits::at(at);
        if !crate::digits::fuse() {
            return Ok(None);
        }
        let Expr::Binary { lhs: pa, rhs: pb, .. } = prod else { unreachable!() };
        // operands in the order the program evaluates them
        let (x, y, c) = if prod_left {
            let x = self.eval(pa, scope)?;
            let y = self.eval(pb, scope)?;
            (x, y, self.eval(other, scope)?)
        } else {
            let c = self.eval(other, scope)?;
            let x = self.eval(pa, scope)?;
            (x, self.eval(pb, scope)?, c)
        };
        crate::digits::at(at);
        // a*b + c, a*b - c = a*b + (-c), c + a*b, c - a*b = (-a)*b + c
        let sub = matches!(op, BinOp::Sub);
        let one = |x: f64, y: f64, c: f64| -> f64 {
            match (sub, prod_left) {
                (false, _) => crate::digits::fused(x, y, c),
                (true, true) => crate::digits::fused(x, y, -c),
                (true, false) => crate::digits::fused(-x, y, c),
            }
        };
        let v = match (&x, &y, &c) {
            (Val::Num(x), Val::Num(y), Val::Num(c)) => Val::Num(one(*x, *y, *c)),
            (Val::Arr(_) | Val::Num(_), Val::Arr(_) | Val::Num(_), Val::Arr(_) | Val::Num(_)) => {
                let shape = [&x, &y, &c].iter().find_map(|v| if let Val::Arr(a) = v { Some(a.borrow().shape.clone()) } else { None });
                let Some(shape) = shape else { return Ok(None) };
                let n: usize = shape.iter().product();
                let get = |v: &Val, i: usize| -> Option<f64> {
                    match v {
                        Val::Num(s) => Some(*s),
                        Val::Arr(a) => {
                            let a = a.borrow();
                            (a.shape == shape).then(|| a.data[i])
                        }
                        _ => None,
                    }
                };
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    match (get(&x, i), get(&y, i), get(&c, i)) {
                        (Some(a), Some(b), Some(d)) => out.push(one(a, b, d)),
                        // broadcasting beyond scalars: as written
                        _ => return self.binary(op, if prod_left { self.binary(BinOp::Mul, x.clone(), y.clone())? } else { c.clone() }, if prod_left { c.clone() } else { self.binary(BinOp::Mul, x.clone(), y.clone())? }).map(Some),
                    }
                }
                new_arr(shape, out)
            }
            _ => {
                let p = self.binary(BinOp::Mul, x, y)?;
                return (if prod_left { self.binary(op, p, c) } else { self.binary(op, c, p) }).map(Some);
            }
        };
        Ok(Some(v))
    }

    fn binary(&self, op: BinOp, l: Val, r: Val) -> Result<Val, String> {
        if matches!(l, Val::Record(_) | Val::Recs(_)) || matches!(r, Val::Record(_) | Val::Recs(_)) {
            // + - * / over a record's arrays
            let word = match op {
                BinOp::Add => "`+`",
                BinOp::Sub => "`-`",
                BinOp::Mul => "`*`",
                BinOp::Div => "`/`",
                _ => return Err(crate::lower::REC_OPERATORS.into()),
            };
            for v in [&l, &r] {
                if !matches!(v, Val::Record(_) | Val::Recs(_) | Val::Num(_)) {
                    return Err(format!("{word} takes records of one type, or a record and a number; this is {}", crate::lower::a_or_an(&val_kind(v))));
                }
            }
            return self.rec_zip(word, &[l, r], &mut |it, x| it.binary(op, x[0].clone(), x[1].clone()));
        }
        // two values of an enum without payloads: the same variant or not
        if let (Val::Enum(a), Val::Enum(b)) = (&l, &r) {
            return match op {
                BinOp::EqEq => Ok(Val::Bool(a.variant == b.variant)),
                BinOp::Ne => Ok(Val::Bool(a.variant != b.variant)),
                _ => Err("enum values are compared with `==` and `!=` only".into()),
            };
        }
        // two Float32s: the lowering let no other mix through
        if let (Val::F32(a), Val::F32(b)) = (&l, &r) {
            use BinOp::*;
            let (a, b) = (*a, *b);
            return Ok(match op {
                Add => Val::F32(a + b),
                Sub => Val::F32(a - b),
                Mul => Val::F32(a * b),
                Div => Val::F32(a / b),
                Lt => Val::Bool(a < b),
                Le => Val::Bool(a <= b),
                Gt => Val::Bool(a > b),
                Ge => Val::Bool(a >= b),
                EqEq => Val::Bool(a == b),
                Ne => Val::Bool(a != b),
                _ => return Err(format!("{op:?} is not defined on Float32 in this version")),
            });
        }
        // an Int on either side: two Ints give an Int
        // (a number for `/`); an Int compares with a number exactly; anything
        // else takes the Int as a number, exactly
        if matches!(l, Val::Int(_)) || matches!(r, Val::Int(_)) {
            use BinOp::*;
            let cmp = |c: f64, op: BinOp| -> Val {
                Val::Bool(match op {
                    Lt => c < 0.0,
                    Le => c <= 0.0,
                    Gt => c > 0.0,
                    Ge => c >= 0.0,
                    EqEq => c == 0.0,
                    _ => c != 0.0,
                })
            };
            return match (&l, &r) {
                (Val::Int(a), Val::Int(b)) => Ok(match op {
                    Add => Val::Int(ints::add(*a, *b)?),
                    Sub => Val::Int(ints::sub(*a, *b)?),
                    Mul => Val::Int(ints::mul(*a, *b)?),
                    FloorDiv => Val::Int(ints::floordiv(*a, *b)?),
                    Mod => Val::Int(ints::floormod(*a, *b)?),
                    Div => Val::Num(ints::to_num(*a)? / ints::to_num(*b)?),
                    Lt => Val::Bool(a < b),
                    Le => Val::Bool(a <= b),
                    Gt => Val::Bool(a > b),
                    Ge => Val::Bool(a >= b),
                    EqEq => Val::Bool(a == b),
                    Ne => Val::Bool(a != b),
                    And | Or => return Err("`and` / `or` expect booleans on both sides".into()),
                    Matmul => return Err("`@` needs two 2-D arrays".into()),
                }),
                (Val::Int(a), Val::Num(x)) if matches!(op, Lt | Le | Gt | Ge | EqEq | Ne) => Ok(cmp(ints::cmp_num(*a, *x), op)),
                (Val::Num(x), Val::Int(a)) if matches!(op, Lt | Le | Gt | Ge | EqEq | Ne) => Ok(cmp(-ints::cmp_num(*a, *x), op)),
                _ => {
                    let num = |v: Val| -> Result<Val, String> { if let Val::Int(i) = v { Ok(Val::Num(ints::to_num(i)?)) } else { Ok(v) } };
                    self.binary(op, num(l)?, num(r)?)
                }
            };
        }
        if matches!(l, Val::Str(_) | Val::Strs(_)) || matches!(r, Val::Str(_) | Val::Strs(_)) {
            return strings::binary(op, &l, &r);
        }
        // `%` and `//`: numbers only, and a zero divisor stops the program
        if matches!(op, BinOp::Mod | BinOp::FloorDiv) {
            let sym = if op == BinOp::Mod { "%" } else { "//" };
            let (Val::Num(a), Val::Num(b)) = (&l, &r) else {
                return Err(format!("`{sym}` works on two numbers"));
            };
            if *b == 0.0 {
                return Err(zero_divisor(op));
            }
            return Ok(Val::Num(if op == BinOp::Mod { floor_mod(*a, *b) } else { floor_div(*a, *b) }));
        }
        // Complex arrays first: any operation with one on either side.
        if matches!(l, Val::CArr(_)) || matches!(r, Val::CArr(_)) {
            return self.binary_carray(op, l, r);
        }
        // Array operations: elementwise with scalar broadcast for + - * /.
        if matches!(l, Val::Arr(_)) || matches!(r, Val::Arr(_)) {
            return self.binary_array(op, l, r);
        }
        // Complex arithmetic, with real -> complex promotion.
        if matches!(l, Val::Cplx(..)) || matches!(r, Val::Cplx(..)) {
            let lift = |v: &Val| -> Result<(f64, f64), String> {
                match v {
                    Val::Cplx(a, b) => Ok((*a, *b)),
                    Val::Num(n) => Ok((*n, 0.0)),
                    _ => Err("cannot mix a complex number with this value".into()),
                }
            };
            let (ar, ai) = lift(&l)?;
            let (br, bi) = lift(&r)?;
            return match op {
                BinOp::Add => Ok(Val::Cplx(ar + br, ai + bi)),
                BinOp::Sub => Ok(Val::Cplx(ar - br, ai - bi)),
                // the formulas live in vmath, beside their C twins
                BinOp::Mul => {
                    let (cr, ci) = crate::vmath::cmul(ar, ai, br, bi);
                    Ok(Val::Cplx(cr, ci))
                }
                BinOp::Div => {
                    let (cr, ci) = crate::vmath::cdiv(ar, ai, br, bi);
                    Ok(Val::Cplx(cr, ci))
                }
                _ => Err("complex numbers cannot be ordered - compare abs(z) instead".into()),
            };
        }
        // booleans: `and` / `or`, and equality between two of them. Nothing else.
        if let (Val::Bool(a), Val::Bool(b)) = (&l, &r) {
            return match op {
                BinOp::And => Ok(Val::Bool(*a && *b)),
                BinOp::Or => Ok(Val::Bool(*a || *b)),
                BinOp::EqEq => Ok(Val::Bool(a == b)),
                BinOp::Ne => Ok(Val::Bool(a != b)),
                _ => Err("booleans can be combined with `and`, `or`, `==` and `!=` only".into()),
            };
        }
        if matches!(op, BinOp::And | BinOp::Or) {
            return Err("`and` / `or` expect booleans on both sides".into());
        }
        let (a, b) = (l.as_num("an operand")?, r.as_num("an operand")?);
        if crate::digits::on() && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
            let c = match op { BinOp::Add => '+', BinOp::Sub => '-', BinOp::Mul => '*', _ => '/' };
            return Ok(Val::Num(crate::digits::arith(c, a, b)));
        }
        Ok(match op {
            BinOp::Add => Val::Num(a + b),
            BinOp::Sub => Val::Num(a - b),
            BinOp::Mul => Val::Num(a * b),
            BinOp::Div => Val::Num(a / b),
            BinOp::Lt => Val::Bool(a < b),
            BinOp::Le => Val::Bool(a <= b),
            BinOp::Gt => Val::Bool(a > b),
            BinOp::Ge => Val::Bool(a >= b),
            BinOp::EqEq => Val::Bool(a == b),
            BinOp::Ne => Val::Bool(a != b),
            BinOp::And | BinOp::Or | BinOp::Mod | BinOp::FloorDiv => unreachable!(),
            BinOp::Matmul => return Err("`@` needs two 2-D arrays".into()),
        })
    }

    /// Arithmetic with a complex array on at least one side. Elementwise
    /// + - * between complex arrays of one shape; a REAL array of that shape
    /// promotes; a real or complex scalar broadcasts; `/` by a scalar. `@` is
    /// the complex matrix product, with the same shape rules as the real one.
    /// Interleaved (re, im) storage throughout.
    fn binary_carray(&self, op: BinOp, l: Val, r: Val) -> Result<Val, String> {
        // view either kind of operand as (shape, complex element getter)
        enum Side {
            C(Arr),
            R(Arr),
            S(f64, f64),
        }
        let side = |v: Val| -> Result<Side, String> {
            match v {
                Val::CArr(a) => Ok(Side::C(a)),
                Val::Arr(a) => Ok(Side::R(a)),
                Val::Num(x) => Ok(Side::S(x, 0.0)),
                Val::Cplx(x, y) => Ok(Side::S(x, y)),
                _ => Err("cannot combine a complex array with this value".into()),
            }
        };
        let (ls, rs) = (side(l)?, side(r)?);
        if matches!(op, BinOp::Matmul) {
            let (a, b) = match (&ls, &rs) {
                (Side::C(a), Side::C(b)) => (a.clone(), b.clone()),
                _ => return Err("`@` on complex arrays needs BOTH operands complex -- promote a real one with `complex(a, zeros(..))`".into()),
            };
            let (ab, bb) = (a.borrow(), b.borrow());
            let (ra, rb) = (ab.shape.len(), bb.shape.len());
            if ra == 1 && rb == 1 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, crate::interp::matmul::VECTORS));
            }
            if !((ra == 2 || ra == 1) && (rb == 2 || rb == 1)) {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`@` needs arrays (2-D, or a 1-D vector operand)"));
            }
            let (m, k) = if ra == 2 { (ab.shape[0], ab.shape[1]) } else { (1, ab.shape[0]) };
            let (k2, n) = if rb == 2 { (bb.shape[0], bb.shape[1]) } else { (bb.shape[0], 1) };
            if k != k2 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`@` shape mismatch: {m}x{k} @ {k2}x{n}")));
            }
            if m.saturating_mul(n) > MAX_ELEMS / 2 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`@`: result of {m}x{n} exceeds the element limit")));
            }
            let mut data = vec![0.0; 2 * m * n];
            for i in 0..m {
                for p in 0..k {
                    let (ar, ai) = (ab.data[2 * (i * k + p)], ab.data[2 * (i * k + p) + 1]);
                    if ar == 0.0 && ai == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        let (br, bi) = (bb.data[2 * (p * n + j)], bb.data[2 * (p * n + j) + 1]);
                        data[2 * (i * n + j)] += ar * br - ai * bi;
                        data[2 * (i * n + j) + 1] += ar * bi + ai * br;
                    }
                }
            }
            if ra == 1 || rb == 1 {
                return Ok(new_carr(vec![m * n], data));
            }
            return Ok(new_carr(vec![m, n], data));
        }
        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
            return Err("only + - * / and @ are supported on complex arrays".into());
        }
        let shape = match (&ls, &rs) {
            (Side::C(a), Side::C(b)) | (Side::C(a), Side::R(b)) | (Side::R(a), Side::C(b)) => {
                let (sa, sb) = (a.borrow().shape.clone(), b.borrow().shape.clone());
                if sa != sb {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("array shape mismatch: {sa:?} vs {sb:?}")));
                }
                sa
            }
            (Side::C(a), Side::S(..)) | (Side::S(..), Side::C(a)) => a.borrow().shape.clone(),
            _ => unreachable!("binary_carray is only entered with a complex array operand"),
        };
        if matches!(op, BinOp::Div) && !matches!(rs, Side::S(..)) {
            return Err("a complex array can be divided by a scalar, not by an array".into());
        }
        let n: usize = shape.iter().product();
        let get = |sd: &Side, k: usize| -> (f64, f64) {
            match sd {
                Side::C(a) => {
                    let d = a.borrow();
                    (d.data[2 * k], d.data[2 * k + 1])
                }
                Side::R(a) => (a.borrow().data[k], 0.0),
                Side::S(x, y) => (*x, *y),
            }
        };
        let mut data = Vec::with_capacity(2 * n);
        for k in 0..n {
            let ((ar, ai), (br, bi)) = (get(&ls, k), get(&rs, k));
            let (cr, ci) = match op {
                BinOp::Add => (ar + br, ai + bi),
                BinOp::Sub => (ar - br, ai - bi),
                BinOp::Mul => crate::vmath::cmul(ar, ai, br, bi),
                _ => crate::vmath::cdiv(ar, ai, br, bi),
            };
            data.push(cr);
            data.push(ci);
        }
        Ok(new_carr(shape, data))
    }

    /// The tape and the adjoint helpers the gradient defs of `crate::grad`
    /// call. The C runtime's `viv_ad_*` (lower.rs, fragment "ad") is their
    /// twin, operation for operation; the wasm runtime calls these.
    fn ad_builtin(&self, callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
        let real = |v: &Val, what: &str| -> Result<Arr, String> {
            match v {
                Val::Arr(x) => Ok(x.clone()),
                _ => Err(format!("autodiff: {what} must be a real array")),
            }
        };
        let transpose = |x: &Arr| -> Result<Val, String> {
            let b = x.borrow();
            if b.shape.len() != 2 {
                return Err("`transpose` takes a 2-D array".into());
            }
            let (rows, cols) = (b.shape[0], b.shape[1]);
            let mut data = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[j * rows + i] = b.data[i * cols + j];
                }
            }
            Ok(new_arr(vec![cols, rows], data))
        };
        let outer = |u: &Arr, v: &Arr| -> Val {
            let (ub, vb) = (u.borrow(), v.borrow());
            let (m, n) = (ub.data.len(), vb.data.len());
            let mut data = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    data[i * n + j] = ub.data[i] * vb.data[j];
                }
            }
            new_arr(vec![m, n], data)
        };
        // 1 where a > b, 1/2 where a == b, 0 elsewhere: the share of max(a, b)
        let share = |x: f64, y: f64| if x > y { 1.0 } else if x == y { 0.5 } else { 0.0 };
        Ok(Some(match (callee, a.len()) {
            ("__ad_push", 1) => {
                let v = match &a[0] {
                    Val::Num(_) | Val::Bool(_) => a[0].clone(),
                    Val::Arr(x) => {
                        let b = x.borrow();
                        new_arr(b.shape.clone(), b.data.clone())
                    }
                    _ => return Err("autodiff: `grad` differentiates through numbers, booleans and real arrays only".into()),
                };
                self.ad_tape.borrow_mut().push(v);
                Val::Num(0.0)
            }
            // the argument, if any, only says the type
            ("__ad_pop", _) => self.ad_tape.borrow_mut().pop().ok_or("autodiff: the tape is empty")?,
            ("__ad_restore", 1) => {
                let dst = real(&a[0], "a restored value")?;
                let src = self.ad_tape.borrow_mut().pop().ok_or("autodiff: the tape is empty")?;
                let src = real(&src, "a restored value")?;
                let data = src.borrow().data.clone();
                let mut d = dst.borrow_mut();
                if d.data.len() != data.len() {
                    return Err("autodiff: an array changed size between the sweeps".into());
                }
                d.data.copy_from_slice(&data);
                Val::Num(0.0)
            }
            ("__ad_zero", 1) => match &a[0] {
                Val::Num(_) => Val::Num(0.0),
                Val::Arr(x) => {
                    let b = x.borrow();
                    new_arr(b.shape.clone(), vec![0.0; b.data.len()])
                }
                _ => return Err("autodiff: `grad` differentiates numbers and real arrays only".into()),
            },
            // the adjoint of an operand that was broadcast: summed down to a
            // number, in `sum`'s order, or spread over the array's shape
            ("__ad_unb", 2) => match (&a[0], &a[1]) {
                (Val::Arr(g), Val::Num(_)) => Val::Num(arrays::npsum(&g.borrow().data)),
                (Val::Num(g), Val::Arr(like)) => {
                    let b = like.borrow();
                    new_arr(b.shape.clone(), vec![*g; b.data.len()])
                }
                // an array operand that was broadcast against a larger one
                (Val::Arr(g), Val::Arr(like)) if g.borrow().shape != like.borrow().shape => {
                    let (gb, lb) = (g.borrow(), like.borrow());
                    new_arr(lb.shape.clone(), arrays::unbroadcast(&gb.shape, &gb.data, &lb.shape))
                }
                _ => a[0].clone(),
            },
            ("__ad_cmp", 2) => match (&a[0], &a[1]) {
                (Val::Num(x), Val::Num(y)) => Val::Num(share(*x, *y)),
                (Val::Arr(x), Val::Num(y)) => map_arr(x, |v| share(v, *y)),
                (Val::Num(x), Val::Arr(y)) => map_arr(y, |v| share(*x, v)),
                (Val::Arr(x), Val::Arr(y)) => {
                    let (xb, yb) = (x.borrow(), y.borrow());
                    if xb.shape != yb.shape {
                        // operands that were broadcast (interp/arrays.rs)
                        let (to, p, q) = arrays::broadcast_pair(&xb.shape, &xb.data, &yb.shape, &yb.data, "")?;
                        return Ok(Some(new_arr(to, p.iter().zip(&q).map(|(&p, &q)| share(p, q)).collect())));
                    }
                    new_arr(xb.shape.clone(), xb.data.iter().zip(&yb.data).map(|(&p, &q)| share(p, q)).collect())
                }
                _ => return Err("autodiff: `max`/`min` of numbers or real arrays only".into()),
            },
            // the adjoints of C = A @ B: dA = dC @ B^T and dB = A^T @ dC, or
            // an outer product where the vector operand makes C a vector
            ("__ad_mm_a", 3) => {
                let (g, x, y) = (real(&a[0], "a gradient")?, real(&a[1], "a matrix product's operand")?, real(&a[2], "a matrix product's operand")?);
                let (rx, ry) = (x.borrow().shape.len(), y.borrow().shape.len());
                if rx > 2 || ry > 2 {
                    // a batched product (interp/matmul.rs)
                    let (gb, xb, yb) = (g.borrow(), x.borrow(), y.borrow());
                    new_arr(xb.shape.clone(), matmul::adjoint_a(&gb.data, &xb.shape, &yb.shape, &yb.data)?)
                } else if rx == 2 && ry == 1 {
                    outer(&g, &y)
                } else {
                    self.binary_array(BinOp::Matmul, Val::Arr(g), transpose(&y)?)?
                }
            }
            ("__ad_mm_b", 3) => {
                let (g, x, y) = (real(&a[0], "a gradient")?, real(&a[1], "a matrix product's operand")?, real(&a[2], "a matrix product's operand")?);
                let (rx, ry) = (x.borrow().shape.len(), y.borrow().shape.len());
                if rx > 2 || ry > 2 {
                    let (gb, xb, yb) = (g.borrow(), x.borrow(), y.borrow());
                    new_arr(yb.shape.clone(), matmul::adjoint_b(&gb.data, &xb.shape, &xb.data, &yb.shape)?)
                } else if rx == 1 {
                    outer(&x, &g)
                } else {
                    self.binary_array(BinOp::Matmul, transpose(&x)?, Val::Arr(g))?
                }
            }
            // the adjoints of the array vocabulary (interp/arrays.rs, "adjoints")
            ("__ad_keep", 3) => {
                let (g, x) = (real(&a[0], "a gradient")?, real(&a[1], "a reduced array")?);
                let k = match &a[2] {
                    Val::Num(k) => *k,
                    _ => return Err("autodiff: a reduction's axis must be a number".into()),
                };
                let shape = arrays::ad_keep_shape(&x.borrow().shape, k)?;
                let data = g.borrow().data.clone();
                if data.len() != shape.iter().product::<usize>() {
                    return Err("autodiff: a reduction's adjoint has the wrong number of entries".into());
                }
                new_arr(shape, data)
            }
            ("__ad_like", 2) => {
                let (g, x) = (real(&a[0], "a gradient")?, real(&a[1], "a reshaped array")?);
                let data = g.borrow().data.clone();
                let shape = x.borrow().shape.clone();
                if data.len() != shape.iter().product::<usize>() {
                    return Err("autodiff: a reshape's adjoint has the wrong number of entries".into());
                }
                new_arr(shape, data)
            }
            ("__ad_prod", 1) | ("__ad_prodax", 2) => {
                let x = real(&a[0], "`prod`'s array")?;
                let xb = x.borrow();
                let k = match a.get(1) {
                    Some(Val::Num(k)) => Some(arrays::ad_axis(*k, xb.shape.len())?),
                    Some(_) => return Err("autodiff: a reduction's axis must be a number".into()),
                    None => None,
                };
                new_arr(xb.shape.clone(), arrays::ad_prod_others(&xb.shape, &xb.data, k))
            }
            ("__ad_takeax", 4) => {
                let (g, x, ix) = (real(&a[0], "a gradient")?, real(&a[1], "`take`'s array")?, real(&a[2], "`take`'s indices")?);
                let k = match &a[3] {
                    Val::Num(k) => *k,
                    _ => return Err("autodiff: `take`'s axis must be a number".into()),
                };
                let shape = x.borrow().shape.clone();
                let data = arrays::ad_take_axis(&g.borrow().data, &shape, &ix.borrow().data, k)?;
                new_arr(shape, data)
            }
            // the adjoint of take(A, idx): each entry of g added into A's
            // shape at its flat index, in order, so repeated indices sum
            ("__ad_take", 3) => {
                let (g, x, ix) = (real(&a[0], "a gradient")?, real(&a[1], "`take`'s array")?, real(&a[2], "`take`'s indices")?);
                let (gb, xb, ib) = (g.borrow(), x.borrow(), ix.borrow());
                let mut data = vec![0.0; xb.data.len()];
                for (j, &f) in ib.data.iter().enumerate() {
                    let k = f as usize;
                    if f < 0.0 || k >= data.len() {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`take`: index {f} out of range (the array has {} elements)", data.len())));
                    }
                    data[k] += gb.data[j];
                }
                new_arr(xb.shape.clone(), data)
            }
            _ => return Ok(None),
        }))
    }

    fn binary_array(&self, op: BinOp, l: Val, r: Val) -> Result<Val, String> {
        // `@`: the matrix product, batched over leading axes as NumPy's
        // matmul (interp/matmul.rs, the rules and the order of the sums)
        if matches!(op, BinOp::Matmul) {
            let (a, b) = match (&l, &r) {
                (Val::Arr(a), Val::Arr(b)) => (a.clone(), b.clone()),
                _ => return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`@` needs arrays (2-D, or a 1-D vector operand)")),
            };
            let (ab, bb) = (a.borrow(), b.borrow());
            let (shape, data) = matmul::matmul(&ab.shape, &ab.data, &bb.shape, &bb.data)?;
            return Ok(new_arr(shape, data));
        }
        // + - * / elementwise, and the comparisons, which give 1.0 where the
        // comparison holds and 0.0 where it does not (a mask is an array of
        // numbers: Arr holds f64)
        let f: fn(f64, f64) -> f64 = match op {
            BinOp::Add => |a, b| a + b,
            BinOp::Sub => |a, b| a - b,
            BinOp::Mul => |a, b| a * b,
            BinOp::Div => |a, b| a / b,
            BinOp::Lt => |a, b| (a < b) as u8 as f64,
            BinOp::Le => |a, b| (a <= b) as u8 as f64,
            BinOp::Gt => |a, b| (a > b) as u8 as f64,
            BinOp::Ge => |a, b| (a >= b) as u8 as f64,
            BinOp::EqEq => |a, b| (a == b) as u8 as f64,
            BinOp::Ne => |a, b| (a != b) as u8 as f64,
            _ => {
                return Err("`and`, `or`, `%` and `//` take numbers or booleans, not arrays".into())
            }
        };
        // the digits lens: each element's operation through crate::digits
        let f: fn(f64, f64) -> f64 = if crate::digits::on() {
            match op {
                BinOp::Add => |a, b| crate::digits::arith_elem('+', a, b),
                BinOp::Sub => |a, b| crate::digits::arith_elem('-', a, b),
                BinOp::Mul => |a, b| crate::digits::arith_elem('*', a, b),
                BinOp::Div => |a, b| crate::digits::arith_elem('/', a, b),
                _ => f,
            }
        } else {
            f
        };
        let out = match (l, r) {
            (Val::Arr(a), Val::Num(s)) => map_arr(&a, |x| f(x, s)),
            (Val::Num(s), Val::Arr(a)) => map_arr(&a, |x| f(s, x)),
            (Val::Arr(a), Val::Arr(b)) => {
                let (ab, bb) = (a.borrow(), b.borrow());
                if ab.shape != bb.shape {
                    // NumPy's broadcasting (interp/arrays.rs), or the mismatch
                    let (to, x, y) = arrays::broadcast_pair(&ab.shape, &ab.data, &bb.shape, &bb.data, "")?;
                    let data = x.iter().zip(&y).map(|(&x, &y)| f(x, y)).collect();
                    return Ok(new_arr(to, data));
                }
                let data = ab.data.iter().zip(&bb.data).map(|(&x, &y)| f(x, y)).collect();
                new_arr(ab.shape.clone(), data)
            }
            _ => return Err("cannot combine an array with a function".into()),
        };
        Ok(out)
    }

    /// Call a user `def` with already-evaluated arguments (the shared path for
    /// direct calls and for calls through a function-valued variable).
    fn call_user(&self, name: &str, func: &Func, argvals: Vec<Val>) -> Result<Val, String> {
        if argvals.len() != func.params.len() {
            return Err(format!(
                "`{name}` expects {} argument(s), got {}",
                func.params.len(),
                argvals.len()
            ));
        }
        for (j, (want, v)) in func.rec_params.iter().zip(&argvals).enumerate() {
            if let Some(want) = want {
                let got = val_kind(v);
                if got != *want {
                    if crate::update::is_update_def(name) && j + 1 == func.rec_params.len() {
                        return Err(format!("the record after `..` should be a {want} but is a {got}"));
                    }
                    return Err(format!("argument {} of `{name}` should be a {want} but is a {got}", j + 1));
                }
            }
        }
        let mut local: HashMap<String, Val> = HashMap::new();
        for (p, v) in func.params.iter().zip(argvals) {
            local.insert(p.clone(), v);
        }
        // A def `crate::update` declared for a record update is part of the
        // expression that wrote it: the graph shows the caller, not it.
        if crate::update::is_expression_def(name) {
            let flow = self.exec_block(&func.body, &mut local, false);
            return match flow? {
                Flow::Return(v) => Ok(v),
                _ => Ok(Val::Num(0.0)),
            };
        }
        // Record the call edge for the live call graph.
        let caller = self
            .call_stack
            .borrow()
            .last()
            .cloned()
            .unwrap_or_else(|| "<main>".to_string());
        *self
            .call_counts
            .borrow_mut()
            .entry((caller.clone(), name.to_string()))
            .or_insert(0) += 1;
        {
            let mut trace = self.call_trace.borrow_mut();
            if trace.len() < TRACE_CAP {
                trace.push((caller, name.to_string()));
            }
        }
        self.call_stack.borrow_mut().push(name.to_string());
        let mark = self.steps_mark();
        let flow = self.exec_block(&func.body, &mut local, false);
        self.steps_leave(mark);
        if let Some(Stmt::Lens { lens, .. }) = func.body.first() {
            if let Lens::Call { def } = &**lens {
                self.story_event(|| format!("[\"ret\",\"{def}\"]"));
            }
        }
        self.call_stack.borrow_mut().pop();
        match flow? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Val::Num(0.0)), // fell off the end with no return
        }
    }

    fn call(&self, callee: &str, args: &[Expr], scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        // User-defined functions first.
        if let Some(func) = self.funcs.get(callee).cloned() {
            let mut argvals = Vec::with_capacity(args.len());
            for a in args {
                // (an Int where the def takes a number is converted by `eval`)
                argvals.push(self.eval(a, scope)?);
            }
            return self.call_user(callee, &func, argvals);
        }

        // `print(...)`: text and numbers on one line (crate::printfmt). A
        // string literal is text, `fmt(x, ...)` a value in a format, anything
        // else a value as `show` writes it. Every value is computed before a
        // byte is written, as the compiled program's one call does.
        if callee == "print" {
            let fmt_is_builtin = !self.funcs.contains_key("fmt");
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(match a {
                    Expr::Str { value, .. } => Part::Text(value.clone()),
                    Expr::Call { callee, args: fa, span } if callee == "fmt" && fmt_is_builtin => {
                        // a string, or a list of strings, in its own format
                        // a map in one or two
                        let v = match fa.first() {
                            Some(x) => self.eval(x, scope)?,
                            None => return Err(crate::printfmt::Fmt::of_args(fa, *span).map_err(|(_, m, _)| m).err().unwrap_or_default()),
                        };
                        if let Val::Str(_) | Val::Strs(_) = v {
                            let rest: Vec<Val> = fa[1..].iter().map(|e| self.eval(e, scope)).collect::<Result<_, _>>()?;
                            Part::Text(strings::fmt_value(&v, &rest)?)
                        } else {
                            let (f, vf) = crate::printfmt::Fmt::of_map_args(fa, *span).map_err(|(_, m, _)| m)?;
                            // a width or precision given as a value: `fmt(x, ".*f", p)`
                            let starred = f.stars() > 0;
                            let f = if starred {
                                let vals: Vec<Val> = fa[2..].iter().map(|e| self.eval(e, scope)).collect::<Result<_, _>>()?;
                                f.with_stars(&strings::star_values(&vals)?)?
                            } else {
                                f
                            };
                            match v {
                                m @ Val::Map(_) => Part::Map(f, vf, m),
                                _ if fa.len() == 3 && !starred => return Err(maps::TWO_FORMATS.into()),
                                v => Part::Val(f, v),
                            }
                        }
                    }
                    e => match self.eval(e, scope)? {
                        m @ Val::Map(_) => Part::Map(crate::printfmt::Fmt::BARE, crate::printfmt::Fmt::BARE, m),
                        v => Part::Val(crate::printfmt::Fmt::BARE, v),
                    },
                });
            }
            return self.print_parts(&parts);
        }
        // The wasm runtime's `print` (wasm_backend, Viv::Print): each part is
        // two arguments, a tag -- `t` for text, else the format's code -- then
        // the text or the value.
        if callee == "__print" {
            let mut parts = Vec::with_capacity(args.len() / 2);
            for pair in args.chunks(2) {
                let [Expr::Str { value: tag, .. }, payload] = pair else {
                    return Err("`__print` takes (tag, payload) pairs".into());
                };
                parts.push(match (tag.as_str(), payload) {
                    ("t", Expr::Str { value, .. }) => Part::Text(value.clone()),
                    // a map's two formats, `<keys>|<values>`
                    (code, e) if code.contains('|') => {
                        let (k, v) = code.split_once('|').expect("a `|`");
                        let f = |c: &str| crate::printfmt::Fmt::decode(c).ok_or("`__print`: a malformed format");
                        Part::Map(f(k)?, f(v)?, self.eval(e, scope)?)
                    }
                    (code, e) => {
                        let f = crate::printfmt::Fmt::decode(code).ok_or("`__print`: a malformed format")?;
                        Part::Val(f, self.eval(e, scope)?)
                    }
                });
            }
            return self.print_parts(&parts);
        }

        // `bars(x, y, "label")`: a bar chart record, the histogram's chart.
        if callee == "bars" && (args.len() == 2 || args.len() == 3) {
            let xv = self.eval(&args[0], scope)?;
            let yv = self.eval(&args[1], scope)?;
            let label = match args.get(2) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`bars`'s third argument must be a string label".into()),
                None => String::new(),
            };
            let (Val::Arr(xa), Val::Arr(ya)) = (&xv, &yv) else {
                return Err("`bars` takes two arrays (x, y) and an optional string label".into());
            };
            let (xb, yb) = (xa.borrow(), ya.borrow());
            let mut out = self.out.borrow_mut();
            out.push_str("bars: ");
            out.push_str(&label);
            out.push_str(" ; ");
            out.push_str(&format_data(&xb.data));
            out.push_str(" ; ");
            out.push_str(&format_data(&yb.data));
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `plot(x, y)` / `plot(x, y, "label")`: the explicit figure statement.
        // Emits a structured line the notebook/IDE renders as a chart. The
        // optional label is a string literal, read syntactically (strings are
        // not runtime values).
        if callee == "plot" && (args.len() == 2 || args.len() == 3) {
            let xv = self.eval(&args[0], scope)?;
            let yv = self.eval(&args[1], scope)?;
            let label = match args.get(2) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`plot`'s third argument must be a string label".into()),
                None => String::new(),
            };
            let (Val::Arr(xa), Val::Arr(ya)) = (&xv, &yv) else {
                return Err("`plot` takes two arrays (x, y) and an optional string label".into());
            };
            let (xb, yb) = (xa.borrow(), ya.borrow());
            let mut out = self.out.borrow_mut();
            out.push_str("plot: ");
            out.push_str(&label);
            out.push_str(" ; ");
            out.push_str(&format_data(&xb.data));
            out.push_str(" ; ");
            out.push_str(&format_data(&yb.data));
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `checkpoint()` — a cell-boundary state marker for the C backend's
        // per-cell caching; the interpreter treats it as a no-op.
        if callee == "checkpoint" && args.is_empty() {
            return Ok(Val::Num(0.0));
        }

        // `figure("title", "xlabel", "ylabel")` — optionally with a fixed axis
        // window: `figure(title, xlabel, ylabel, xlo, xhi, ylo, yhi)`. Emits a
        // structured line the renderer attaches to the NEXT chart, so the
        // program itself carries its presentation (titles, axes, windows) —
        // nothing lives outside the .viv source.
        if callee == "figure" && (args.len() == 3 || args.len() == 7 || args.len() == 8) {
            let mut strs = Vec::new();
            for a in &args[..3] {
                strs.push(self.text_arg(a, scope, "`figure`'s first three arguments must be strings: title, xlabel, ylabel")?);
            }
            let mut line = format!("figure: {} ; {} ; {}", strs[0], strs[1], strs[2]);
            if args.len() >= 7 {
                let mut nums = Vec::new();
                for a in &args[3..7] {
                    match self.eval(a, scope)? {
                        Val::Num(v) => nums.push(v),
                        _ => return Err("`figure`'s window arguments must be numbers: xlo, xhi, ylo, yhi".into()),
                    }
                }
                // the same %g the C runtime prints, so the two engines agree to the byte
                line.push_str(&format!(" ; {} {} {} {}", format_g(nums[0]), format_g(nums[1]), format_g(nums[2]), format_g(nums[3])));
            }
            if args.len() == 8 {
                match &args[7] {
                    Expr::Str { value, .. } => line.push_str(&format!(" ; {value}")),
                    _ => return Err("`figure`'s eighth argument must be a style string (e.g. \"contour\")".into()),
                }
            }
            let mut out = self.out.borrow_mut();
            out.push_str(&line);
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `scatter(x, y)` / `scatter(x, y, "label")`: points, not a line —
        // spectra in the complex plane, samples, anything unordered.
        // Consecutive scatter() calls overlay as series in one chart.
        if callee == "scatter" && (2..=4).contains(&args.len()) {
            let xv = self.eval(&args[0], scope)?;
            let yv = self.eval(&args[1], scope)?;
            let label = match args.get(2) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`scatter`'s third argument must be a string label".into()),
                None => String::new(),
            };
            let color = match args.get(3) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`scatter`'s fourth argument must be a color name string".into()),
                None => String::new(),
            };
            let (Val::Arr(xa), Val::Arr(ya)) = (&xv, &yv) else {
                return Err("`scatter` takes two arrays (x, y) and an optional string label".into());
            };
            let (xb, yb) = (xa.borrow(), ya.borrow());
            // Full precision: scatter carries figure DATA (round-off clouds at
            // 1e-15 are the point of some panels), unlike %g display text.
            let full = |data: &[f64]| {
                let mut s = String::from("[");
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&format_e17(*v));
                }
                s.push(']');
                s
            };
            let mut out = self.out.borrow_mut();
            out.push_str("scatter: ");
            out.push_str(&label);
            out.push_str(" ; ");
            out.push_str(&color);
            out.push_str(" ; ");
            out.push_str(&full(&xb.data));
            out.push_str(" ; ");
            out.push_str(&full(&yb.data));
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `quiver(x, y, u, v[, "label"])`: a vector field as arrows — one
        // arrow per point, at (x, y) with components (u, v).
        if callee == "quiver" && (args.len() == 4 || args.len() == 5) {
            let mut arrs = Vec::new();
            for a in &args[..4] {
                match self.eval(a, scope)? {
                    Val::Arr(x) => arrs.push(x),
                    _ => return Err("`quiver` takes four arrays (x, y, u, v) and an optional label".into()),
                }
            }
            let label = match args.get(4) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`quiver`'s fifth argument must be a string label".into()),
                None => String::new(),
            };
            let mut out = self.out.borrow_mut();
            out.push_str("quiver: ");
            out.push_str(&label);
            for a in &arrs {
                let b = a.borrow();
                out.push_str(" ; ");
                out.push_str(&format_arr(&[b.data.len()], &b.data));
            }
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `plot2d(Z)` / `plot2d(Z, "label")`: a matrix as a heatmap/surface.
        if callee == "plot2d" && (args.len() == 1 || args.len() == 2) {
            let zv = self.eval(&args[0], scope)?;
            let label = match args.get(1) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`plot2d`'s second argument must be a string label".into()),
                None => String::new(),
            };
            let Val::Arr(za) = &zv else {
                return Err("`plot2d` takes a 2-D array and an optional string label".into());
            };
            let zb = za.borrow();
            let (nx, ny) = match zb.shape.as_slice() {
                [r, c] => (*r, *c),
                [n] => (1, *n),
                _ => return Err("`plot2d` needs a rank-1 or rank-2 array".into()),
            };
            let mut out = self.out.borrow_mut();
            out.push_str("plot2d: ");
            out.push_str(&label);
            out.push_str(&format!(" ; {nx} {ny} ; "));
            out.push_str(&format_data(&zb.data));
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `plot3d(x, y, z)` / `plot3d(x, y, z, "label")`: a curve in 3-space.
        if callee == "plot3d" && (args.len() == 3 || args.len() == 4) {
            let xv = self.eval(&args[0], scope)?;
            let yv = self.eval(&args[1], scope)?;
            let zv = self.eval(&args[2], scope)?;
            let label = match args.get(3) {
                Some(Expr::Str { value, .. }) => value.clone(),
                Some(_) => return Err("`plot3d`'s fourth argument must be a string label".into()),
                None => String::new(),
            };
            let (Val::Arr(xa), Val::Arr(ya), Val::Arr(za)) = (&xv, &yv, &zv) else {
                return Err("`plot3d` takes three arrays (x, y, z) and an optional string label".into());
            };
            let (xb, yb, zb) = (xa.borrow(), ya.borrow(), za.borrow());
            let mut out = self.out.borrow_mut();
            out.push_str("plot3d: ");
            out.push_str(&label);
            out.push_str(" ; ");
            out.push_str(&format_arr(&xb.shape, &xb.data));
            out.push_str(" ; ");
            out.push_str(&format_arr(&yb.shape, &yb.data));
            out.push_str(" ; ");
            out.push_str(&format_arr(&zb.shape, &zb.data));
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `scene("title")` — opens a 3D scene; wire()/verts() output after it
        // belongs to this scene. The scene protocol is
        // the general geometry channel: schematics, FEM results, mechanisms.
        // The animation verbs: each prints one record,
        // `verb: text ; text ; [array] ; [array] ; n n n`, byte for byte what the
        // C runtime's viv_anim prints. A program's own `def draw(...)` wins.
        if let Some(v) = crate::lower::anim_verb(callee).filter(|_| !self.funcs.contains_key(callee)) {
            let fixed = v.texts + v.arrays + v.nums;
            if args.len() < fixed || args.len() > fixed + v.opt {
                return Err(format!("`{callee}` is written {}", v.usage));
            }
            let mut texts = Vec::new();
            for (k, a) in args[..v.texts].iter().enumerate() {
                let value = &self.text_arg(a, scope, &format!("`{callee}` is written {} -- the text must be a string", v.usage))?;
                if value.contains('\n') || value.contains(" ; ") {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "text in an animation cannot contain a newline or ` ; `"));
                }
                crate::lower::anim_text_check(callee, k, value)?;
                texts.push(value.clone());
            }
            let mut line = crate::lower::anim_head(callee, &texts);
            let mut bare = texts.is_empty();
            // a chart's data (fill_between, rangebars) at full precision, as plot's is
            let full = crate::lower::anim_data_verb(callee);
            let mut nums = Vec::new();
            for (k, a) in args[v.texts..].iter().enumerate() {
                let val = self.eval(a, scope)?;
                if k < v.arrays {
                    let Val::Arr(arr) = &val else {
                        return Err(format!("`{callee}` is written {} -- that argument must be a real array", v.usage));
                    };
                    line.push_str(if bare { " [" } else { " ; [" });
                    bare = false;
                    for (i, x) in arr.borrow().data.iter().enumerate() {
                        if i > 0 {
                            line.push_str(", ");
                        }
                        line.push_str(&if full { format_rt(*x + 0.0) } else { format_g(*x + 0.0) });
                    }
                    line.push(']');
                } else {
                    nums.push(val.as_num(&format!("`{callee}`'s argument"))?);
                }
            }
            if !nums.is_empty() {
                if !bare {
                    line.push_str(" ;");
                }
                for x in nums {
                    line.push(' ');
                    line.push_str(&format_g(x + 0.0));
                }
            }
            let mut out = self.out.borrow_mut();
            out.push_str(&line);
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        if callee == "scene" && args.len() == 1 {
            let Some(Expr::Str { value, .. }) = args.first() else {
                return Err("`scene` takes one string argument: the title".into());
            };
            let mut out = self.out.borrow_mut();
            out.push_str("scene: ");
            out.push_str(value);
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // `wire(x, y, z, "group")` / `wire(x, y, z, v, "group")` — a 3D
        // polyline appended to a named scene group; the optional v array is a
        // per-vertex scalar the viewer colormaps (stress, |B|, temperature).
        // `verts(...)` is identical but renders point markers.
        if (callee == "wire" || callee == "verts") && (args.len() == 4 || args.len() == 5) {
            let n_arr = args.len() - 1;
            let group = match args.last() {
                Some(Expr::Str { value, .. }) => value.clone(),
                _ => return Err(format!("`{callee}`'s last argument must be a string group name")),
            };
            let mut bufs: Vec<String> = Vec::with_capacity(n_arr);
            for e in &args[..n_arr] {
                let v = self.eval(e, scope)?;
                let Val::Arr(a) = &v else {
                    return Err(format!("`{callee}` takes 3 or 4 arrays (x, y, z[, values]) then a string group name"));
                };
                let ab = a.borrow();
                bufs.push(format_arr(&ab.shape, &ab.data));
            }
            let mut out = self.out.borrow_mut();
            out.push_str(callee);
            out.push_str(": ");
            out.push_str(&group);
            for b in &bufs {
                out.push_str(" ; ");
                out.push_str(b);
            }
            out.push('\n');
            return Ok(Val::Num(0.0));
        }

        // A function held in a variable, applied like `f(x)`: either a closure
        // (`fn(x) -> ...`) or a named def passed in as a value.
        match scope
            .get(callee)
            .cloned()
            .or_else(|| self.globals.borrow().get(callee).cloned())
        {
            Some(Val::Closure(clos)) => {
                let mut argvals = Vec::with_capacity(args.len());
                for e in args {
                    argvals.push(self.eval(e, scope)?);
                }
                return self.call_closure(&clos, argvals);
            }
            Some(Val::Func(name, func)) => {
                let mut argvals = Vec::with_capacity(args.len());
                for e in args {
                    argvals.push(self.eval(e, scope)?);
                }
                return self.call_user(&name, &func, argvals);
            }
            _ => {}
        }

        // `x: T = e` declares a type and runs nothing
        if callee == crate::parser::DECLARE {
            return Ok(Val::Num(0.0));
        }
        // `ceil(a / b)` and `floor(a / b)` of two Ints: the exact quotient;
        // of anything else, the quotient computed once, then the builtin
        let mut pre: Option<Vec<Val>> = None;
        if let ("ceil" | "floor", [Expr::Binary { op: BinOp::Div, lhs, rhs, span }]) = (callee, args) {
            if !self.funcs.contains_key(callee) {
                let l = self.eval(lhs, scope)?;
                let r = self.eval(rhs, scope)?;
                if let (Val::Int(a), Val::Int(b)) = (&l, &r) {
                    return if callee == "ceil" { ints::ceildiv(*a, *b) } else { ints::floordiv(*a, *b) }.map(Val::Int);
                }
                let q = self.binary(BinOp::Div, l, r);
                if q.is_err() {
                    self.note_err_at(*span);
                }
                pre = Some(vec![q?]);
            }
        }
        // Builtins.
        let a = match pre {
            Some(a) => a,
            None => args.iter().map(|e| self.eval(e, scope)).collect::<Result<Vec<Val>, String>>()?,
        };
        // `sorted`, `unique` and `del` of a real array: the defs of
        // lang/viv/std/builtins.viv the lowering calls too (crate::stdlib)
        if let Some(form) = crate::stdlib::number_form(callee) {
            let form = match a.first() {
                Some(Val::Arr(_)) => Some(form),
                Some(Val::Strs(_)) if callee == crate::stdlib::DEL => Some("viv__del_strs"),
                _ => None,
            };
            if let Some(func) = form.and_then(|f| self.funcs.get(f).cloned().map(|func| (f, func))) {
                return self.call_user(func.0, &func.1, a);
            }
        }
        // over a record's arrays: zeros_like, sum, the elementwise math, and
        // for_leaves
        if let Some(v) = self.record_builtin(callee, &a)? {
            return Ok(v);
        }
        // Float32 and the packed arrays; `fmt` of a
        // Float32 in a format spec formats the number it is
        let a = match a.first() {
            Some(Val::F32(x)) if callee == "fmt" && a.len() > 1 => {
                let mut w = a.clone();
                w[0] = Val::Num(*x as f64);
                w
            }
            _ => a,
        };
        if let Some(v) = f32_builtin(callee, &a)? {
            return Ok(v);
        }
        // the Int forms of the builtins
        if let Some(v) = int_builtin(callee, &a)? {
            return Ok(v);
        }
        // enum values, and the calls a `match` is written with
        if let Some(v) = enum_builtin(callee, &a)? {
            return Ok(v);
        }
        // strings: the registry, and the builtins that take
        // strings among other values (len, str, repr, fmt, slices, `in`, ...)
        if let Some(v) = strings::call(callee, &a)? {
            return Ok(v);
        }
        // The complex constructor and accessors.
        // complex(re, im) over two real ARRAYS of one shape builds the complex
        // array -- the way a real-space field (a density, a potential) gets
        // into a transform.
        if callee == "complex" && a.len() == 2 {
            if let (Val::Arr(x), Val::Arr(y)) = (&a[0], &a[1]) {
                let (xb, yb) = (x.borrow(), y.borrow());
                if xb.shape != yb.shape {
                    return Err("`complex(re, im)`: the two arrays must have the same shape".into());
                }
                let data = xb.data.iter().zip(yb.data.iter()).flat_map(|(&r, &i)| [r, i]).collect();
                return Ok(new_carr(xb.shape.clone(), data));
            }
        }
        if callee == "complex" && a.len() == 2 {
            return Ok(Val::Cplx(a[0].as_num("`complex`'s real part")?, a[1].as_num("`complex`'s imaginary part")?));
        }
        // ...and over a complex ARRAY: re/im/abs give a real array of the same
        // shape (what you plot as a spectrum), conj another complex one.
        if matches!(callee, "re" | "im" | "conj" | "abs" | "fabs") && a.len() == 1 {
            if let Val::CArr(arr) = &a[0] {
                let ab = arr.borrow();
                let n = ab.data.len() / 2;
                return Ok(match callee {
                    "re" => new_arr(ab.shape.clone(), (0..n).map(|k| ab.data[2 * k]).collect()),
                    "im" => new_arr(ab.shape.clone(), (0..n).map(|k| ab.data[2 * k + 1]).collect()),
                    "conj" => new_carr(
                        ab.shape.clone(),
                        (0..n).flat_map(|k| [ab.data[2 * k], -ab.data[2 * k + 1]]).collect(),
                    ),
                    _ => new_arr(
                        ab.shape.clone(),
                        (0..n).map(|k| ab.data[2 * k].vhypot(ab.data[2 * k + 1])).collect(),
                    ),
                });
            }
        }
        if matches!(callee, "re" | "im" | "conj") && a.len() == 1 {
            return match (&a[0], callee) {
                (Val::Cplx(re, _), "re") => Ok(Val::Num(*re)),
                (Val::Cplx(_, im), "im") => Ok(Val::Num(*im)),
                (Val::Cplx(re, im), "conj") => Ok(Val::Cplx(*re, -*im)),
                (Val::Num(n), "re") | (Val::Num(n), "conj") => Ok(Val::Num(*n)),
                (Val::Num(_), "im") => Ok(Val::Num(0.0)),
                _ => Err(format!("`{callee}` takes a number or a complex number")),
            };
        }
        // 1-argument math builtins on a COMPLEX scalar.
        if a.len() == 1 {
            if let Val::Cplx(re, im) = a[0] {
                let cplx = |(r, i): (f64, f64)| Val::Cplx(r, i);
                return match callee {
                    // the formulas live in vmath, beside their C twins
                    "abs" | "fabs" => Ok(Val::Num(crate::vmath::cabs(re, im))),
                    "exp" => Ok(cplx(crate::vmath::cexp(re, im))),
                    "log" | "ln" => Ok(cplx(crate::vmath::clog(re, im))),
                    "sqrt" => Ok(cplx(crate::vmath::csqrt(re, im))),
                    "sin" => Ok(cplx(crate::vmath::csin(re, im))),
                    "cos" => Ok(cplx(crate::vmath::ccos(re, im))),
                    "tan" => Ok(cplx(crate::vmath::ctan(re, im))),
                    other => Err(format!("`{other}` is not defined for complex numbers")),
                };
            }
        }
        // 1-argument math builtins map elementwise over arrays: y = sin(x)
        if a.len() == 1 {
            if let Val::Arr(arr) = &a[0] {
                if let Some(f) = elementwise_fn(callee) {
                    let ab = arr.borrow();
                    return Ok(new_arr(ab.shape.clone(), ab.data.iter().map(|v| f(*v)).collect()));
                }
            }
        }
        // fill(arr, v): set every element in place
        // sum(a): the sum of every element of a real array. The grid integrals
        // of a DFT code (E_xc, the density's norm) are sums over 10^4..10^6
        // points; as a Viv-level loop they would dominate the run.
        // max(a) / min(a): the largest / smallest entry. Folded with vmath's fmax/fmin
        // from -inf / +inf, which is what the C runtime's fmax/fmin loop does.
        if matches!(callee, "max" | "min") && a.len() == 1 {
            return match &a[0] {
                Val::Arr(arr) => {
                    let b = arr.borrow();
                    Ok(Val::Num(if callee == "max" {
                        b.data.iter().fold(f64::NEG_INFINITY, |m, &x| crate::vmath::fmax(m, x))
                    } else {
                        b.data.iter().fold(f64::INFINITY, |m, &x| crate::vmath::fmin(m, x))
                    }))
                }
                // of an Int array, an Int (B343)
                Val::IArr(x) => ints::extreme(&x.borrow(), callee == "max").map(Val::Int),
                _ => Err(format!("`{callee}` of one argument takes a real array or an Int array")),
            };
        }
        // all(m) / any(m): whether every / some entry of a real array is nonzero
        // (a mask from a comparison holds 1.0 and 0.0). all of an empty array is
        // true, any of one is false.
        if matches!(callee, "all" | "any") && a.len() == 1 {
            return match &a[0] {
                Val::Arr(arr) => {
                    let b = arr.borrow();
                    Ok(Val::Bool(if callee == "all" { b.data.iter().all(|&x| x != 0.0) } else { b.data.iter().any(|&x| x != 0.0) }))
                }
                _ => Err(format!("`{callee}` takes a real array (a mask such as `a > 0`)")),
            };
        }
        // where(m, a, b): a where m is nonzero and b where it is 0. a and b
        // are each a number or an array; the three broadcast to one shape as
        // NumPy's do (interp/arrays.rs), which is m's shape when they match it.
        if callee == "where" && a.len() == 3 {
            let Val::Arr(m) = &a[0] else {
                return Err("`where` takes a mask array first: `where(m, a, b)`".into());
            };
            let mb = m.borrow();
            let mut to = mb.shape.clone();
            for (v, what) in [(&a[1], "its second argument"), (&a[2], "its third argument")] {
                match v {
                    Val::Num(_) => {}
                    Val::Arr(arr) => {
                        let s = &arr.borrow().shape;
                        to = arrays::broadcast(&to, s).ok_or_else(|| crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("array shape mismatch: {:?} vs {:?}", mb.shape, s)))?;
                    }
                    _ => return Err(format!("`where` takes a number or a real array for {what}")),
                }
            }
            if to.iter().product::<usize>() > MAX_ELEMS {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("`where`: broadcasting gives {} elements, over the {MAX_ELEMS}-element limit{LIMIT_OF}", to.iter().product::<usize>())));
            }
            let spread = |shape: &[usize], data: &[f64]| if shape == to.as_slice() { data.to_vec() } else { arrays::expand(shape, data, &to) };
            let mask = spread(&mb.shape, &mb.data);
            let pick = |v: &Val| -> Box<dyn Fn(usize) -> f64> {
                match v {
                    Val::Num(x) => {
                        let x = *x;
                        Box::new(move |_| x)
                    }
                    Val::Arr(arr) => {
                        let ab = arr.borrow();
                        let d = spread(&ab.shape, &ab.data);
                        Box::new(move |k| d[k])
                    }
                    _ => unreachable!("checked above"),
                }
            };
            let (ya, yb) = (pick(&a[1]), pick(&a[2]));
            let data = mask.iter().enumerate().map(|(k, &c)| if c != 0.0 { ya(k) } else { yb(k) }).collect();
            return Ok(new_arr(to, data));
        }
        if callee == "sum" && a.len() == 1 {
            return match &a[0] {
                // NumPy's order (arrays::npsum), in every exact engine
                Val::Arr(arr) => Ok(Val::Num(arrays::npsum(&arr.borrow().data))),
                Val::CArr(_) => Err("`sum` of a complex array is not supported -- sum `re(a)` and `im(a)`".into()),
                // an Int array's sum is an Int, exactly (B343)
                Val::IArr(x) => ints::sum(&x.borrow()).map(Val::Int),
                _ => Err("`sum` takes one array".into()),
            };
        }
        if callee == "fill" && a.len() == 2 {
            // a complex array fills with a complex (or promoted real) value
            if let Val::CArr(arr) = &a[0] {
                let (re, im) = match &a[1] {
                    Val::Cplx(re, im) => (*re, *im),
                    v => (v.as_num("`fill`'s value")?, 0.0),
                };
                for pair in arr.borrow_mut().data.chunks_mut(2) {
                    pair[0] = re;
                    pair[1] = im;
                }
                return Ok(Val::Num(0.0));
            }
            let Val::Arr(arr) = &a[0] else {
                return Err("`fill` takes (array, value)".into());
            };
            let v = a[1].as_num("`fill`'s value")?;
            for x in arr.borrow_mut().data.iter_mut() {
                *x = v;
            }
            return Ok(Val::Num(0.0));
        }
        // `sort` of an Int array (B343)
        if let ("sort", [Val::IArr(x)]) = (callee, a.as_slice()) {
            x.borrow_mut().sort_unstable();
            return Ok(Val::Num(0.0));
        }
        // the map builtins, and `len` and `copy` of a map
        if let Some(v) = maps::builtin(callee, &a)? {
            return Ok(v);
        }
        if let Some(v) = slices::builtin(callee, &a, &self.err_notes)? {
            return Ok(v);
        }
        if callee == "append" && a.len() == 2 {
            if let Some(r) = append_record(&a[0], &a[1]) {
                return r;
            }
        }
        if let Some(v) = growable::builtin(callee, &a)? {
            return Ok(v);
        }
        let mut draw = || {
            let s = (16807.0 * self.rng.get()) % 2147483647.0;
            self.rng.set(s);
            s / 2147483647.0
        };
        if let Some(v) = arrays::builtin(callee, &a, &mut draw)? {
            return Ok(v);
        }
        // `x = lin_solve(A, b)`, `L = cholesky(A)`, `X = inv(A)`: the older
        // solvers, with the array they fill allocated here and returned
        let (callee, a, filled) = returning(callee, a)?;
        if callee.starts_with("__ad_") {
            if let Some(v) = self.ad_builtin(callee, &a)? {
                return Ok(v);
            }
        }
        let num = |i: usize| a[i].as_num("a builtin argument");
        let result = match (callee, a.len()) {
            // `text(a)` writes an array of Unicode scalar values as characters.
            // The counterpart of `chars`, and the only way a program says
            // anything that is not a number. One record per line, so a newline
            // inside the text is written as an escape rather than ending it.
            ("text", 1) => {
                let arr = match &a[0] {
                    Val::Arr(arr) => arr.clone(),
                    _ => return Err("`text` expects an array of character codes".into()),
                };
                let mut line = String::from("text: ");
                for v in arr.borrow().data.iter() {
                    let code = *v as i64;
                    match char::from_u32(code.clamp(0, 0x10_FFFF) as u32) {
                        Some('\n') => line.push_str("\\n"),
                        Some('\t') => line.push_str("\\t"),
                        Some(c) => line.push(c),
                        None => line.push('\u{fffd}'),
                    }
                }
                self.out.borrow_mut().push_str(&line);
                self.out.borrow_mut().push('\n');
                Ok(Val::Num(0.0))
            }
            ("exp", 1) => Ok(Val::Num(num(0)?.vexp())),
            ("sqrt", 1) => Ok(Val::Num(num(0)?.sqrt())),
            // pow(array, s): every entry to the same scalar power.
            ("pow", 2) if matches!(&a[0], Val::Arr(_)) => {
                let s = num(1)?;
                let Val::Arr(x) = &a[0] else { unreachable!() };
                let xb = x.borrow();
                let data = xb.data.iter().map(|&p| p.vpow(s)).collect();
                Ok(new_arr(xb.shape.clone(), data))
            }
            ("pow", 2) => Ok(Val::Num(num(0)?.vpow(num(1)?))),
            ("abs", 1) | ("fabs", 1) => Ok(Val::Num(num(0)?.abs())),
            ("log", 1) | ("ln", 1) => Ok(Val::Num(num(0)?.vln())),
            ("sin", 1) => Ok(Val::Num(num(0)?.vsin())),
            ("cos", 1) => Ok(Val::Num(num(0)?.vcos())),
            ("tan", 1) => Ok(Val::Num(num(0)?.vtan())),
            ("floor", 1) => Ok(Val::Num(num(0)?.floor())),
            ("ceil", 1) => Ok(Val::Num(num(0)?.ceil())),
            ("round", 1) => Ok(Val::Num(num(0)?.round())),
            ("asin", 1) => Ok(Val::Num(num(0)?.vasin())),
            ("acos", 1) => Ok(Val::Num(num(0)?.vacos())),
            ("atan", 1) => Ok(Val::Num(num(0)?.vatan())),
            ("atan2", 2) => Ok(Val::Num(num(0)?.vatan2(num(1)?))),
            ("fma", 3) => Ok(Val::Num(num(0)?.mul_add(num(1)?, num(2)?))),
            ("sinh", 1) => Ok(Val::Num(num(0)?.vsinh())),
            ("cosh", 1) => Ok(Val::Num(num(0)?.vcosh())),
            ("tanh", 1) => Ok(Val::Num(num(0)?.vtanh())),
            // fmod semantics, matching the C backend's fmod exactly.
            ("mod", 2) => Ok(Val::Num(num(0)? % num(1)?)),
            // Bits. Every Viv number is a double; these read the low 32 bits of
            // one (truncated toward zero, then wrapped) and hand back an
            // unsigned 32-bit value, so they compose without a cast. `i32`
            // reads the same bits as a signed number when that is what is
            // wanted. The C backend's helpers do this arithmetic verbatim.
            ("band", 2) => Ok(Val::Num(f64::from(u32_of(num(0)?) & u32_of(num(1)?)))),
            ("bor", 2) => Ok(Val::Num(f64::from(u32_of(num(0)?) | u32_of(num(1)?)))),
            ("bxor", 2) => Ok(Val::Num(f64::from(u32_of(num(0)?) ^ u32_of(num(1)?)))),
            ("bnot", 1) => Ok(Val::Num(f64::from(!u32_of(num(0)?)))),
            ("shl", 2) => {
                let (v, k) = (u32_of(num(0)?), shift_amount(num(1)?));
                Ok(Val::Num(f64::from(k.map_or(0, |k| v << k))))
            }
            ("shr", 2) => {
                let (v, k) = (u32_of(num(0)?), shift_amount(num(1)?));
                Ok(Val::Num(f64::from(k.map_or(0, |k| v >> k))))
            }
            ("sar", 2) => {
                let (v, k) = (u32_of(num(0)?) as i32, shift_amount(num(1)?));
                Ok(Val::Num(f64::from(match k {
                    Some(k) => (v >> k) as u32,
                    None => if v < 0 { u32::MAX } else { 0 },
                })))
            }
            ("u32", 1) => Ok(Val::Num(f64::from(u32_of(num(0)?)))),
            ("i32", 1) => Ok(Val::Num(f64::from(u32_of(num(0)?) as i32))),
            ("erf", 1) => Ok(Val::Num(erf_impl(num(0)?))),
            ("gamma", 1) => Ok(Val::Num(gamma_impl(num(0)?))),
            // min / max with an array operand are elementwise; a scalar may be on
            // either side. vmath's fmin/fmax and their C twins agree, NaN and -0.0 included.
            ("min", 2) | ("max", 2) if matches!(&a[0], Val::Arr(_)) || matches!(&a[1], Val::Arr(_)) => {
                let is_max = callee == "max";
                let pick = |x: f64, y: f64| if is_max { crate::vmath::fmax(x, y) } else { crate::vmath::fmin(x, y) };
                match (&a[0], &a[1]) {
                    (Val::Arr(x), Val::Arr(y)) => {
                        let (xb, yb) = (x.borrow(), y.borrow());
                        if xb.shape != yb.shape {
                            let (to, p, q) = arrays::broadcast_pair(&xb.shape, &xb.data, &yb.shape, &yb.data, &format!("`{callee}`: "))?;
                            return Ok(new_arr(to, p.iter().zip(&q).map(|(&p, &q)| pick(p, q)).collect()));
                        }
                        let data = xb.data.iter().zip(&yb.data).map(|(&p, &q)| pick(p, q)).collect();
                        Ok(new_arr(xb.shape.clone(), data))
                    }
                    (Val::Arr(x), Val::Num(s)) | (Val::Num(s), Val::Arr(x)) => {
                        let xb = x.borrow();
                        let data = xb.data.iter().map(|&p| pick(p, *s)).collect();
                        Ok(new_arr(xb.shape.clone(), data))
                    }
                    _ => Err(format!("`{callee}` takes numbers or real arrays")),
                }
            }
            ("min", 2) => Ok(Val::Num(crate::vmath::fmin(num(0)?, num(1)?))),
            ("max", 2) => Ok(Val::Num(crate::vmath::fmax(num(0)?, num(1)?))),
            ("len", 1) => match &a[0] {
                Val::Arr(arr) | Val::CArr(arr) => {
                    Ok(Val::Num(arr.borrow().shape.first().copied().unwrap_or(0) as f64))
                }
                Val::VArr(v) => Ok(Val::Num(v.borrow().len() as f64)),
                Val::Recs(v) => Ok(Val::Num(v.len() as f64)),
                _ => Err("`len` expects an array".into()),
            },
            // numel(A): every element, over all dimensions (a complex entry
            // counts once). shape(A): the dimensions, as a rank-1 array.
            ("numel", 1) => match &a[0] {
                Val::Arr(arr) | Val::CArr(arr) => Ok(Val::Num(arr.borrow().shape.iter().product::<usize>() as f64)),
                Val::VArr(v) => Ok(Val::Num(v.borrow().len() as f64)),
                _ => Err("`numel` expects an array".into()),
            },
            ("shape", 1) => match &a[0] {
                Val::Arr(arr) | Val::CArr(arr) => {
                    let dims: Vec<f64> = arr.borrow().shape.iter().map(|&d| d as f64).collect();
                    Ok(new_arr(vec![dims.len()], dims))
                }
                _ => Err("`shape` expects a real or complex array".into()),
            },
            // apply(f, args...): call a function VALUE (e.g. one fetched from a
            // value array) with the remaining arguments. This is how a strategy
            // registry dispatches today: `apply(strategies[i], f, a, b)`. Direct
            // `strategies[i](...)` call syntax is a separate parser feature.
            ("apply", _) if !a.is_empty() => {
                // the compiled `apply` has a 16-slot argument buffer; refuse here too
                if a.len() - 1 > crate::lower::APPLY_MAX_ARGS {
                    return Err(format!(
                        "`apply` passes at most {} arguments; this call passes {}",
                        crate::lower::APPLY_MAX_ARGS,
                        a.len() - 1
                    ));
                }
                let f = a[0].clone();
                let argvals = a[1..].to_vec();
                match f {
                    Val::Closure(clos) => self.call_closure(&clos, argvals),
                    Val::Func(name, func) => self.call_user(&name, &func, argvals),
                    _ => Err("`apply` expects a function as its first argument".into()),
                }
            }
            // czeros(d1..d4): a complex array of zeros. The complex twin of
            // `zeros`; `data` is interleaved (re, im), so twice the length.
            ("czeros", _) if !a.is_empty() => {
                if a.len() > crate::lower::MAX_RANK {
                    return Err(format!(
                        "`czeros` takes 1 to {} dimensions; this call gives {}",
                        crate::lower::MAX_RANK,
                        a.len()
                    ));
                }
                let dims: Vec<f64> = a.iter().map(|v| v.as_num("a dimension")).collect::<Result<_, _>>()?;
                let (shape, total) = zeros_shape(&dims);
                if total > (MAX_ELEMS / 2) as u64 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!(
                        "czeros: complex array of {total} elements exceeds the {}-element limit{LIMIT_OF}",
                        MAX_ELEMS / 2
                    )));
                }
                Ok(new_carr(shape, vec![0.0; 2 * total as usize]))
            }
            // fft(a) / ifft(a): the discrete Fourier transform of a complex
            // array, IN PLACE, over its whole flattened extent. Convention:
            //   fft :  X[k] = sum_n x[n] exp(-2 pi i k n / N)      (no scaling)
            //   ifft:  x[n] = (1/N) sum_k X[k] exp(+2 pi i k n / N)
            // so ifft(fft(x)) == x, matching numpy/FFTW.
            // fft3(a) / ifft3(a): the 3-D transform of a rank-3 complex array, in
            // place. Same convention as fft: forward unscaled, inverse carries
            // 1/(N0 N1 N2). Any grid dimensions.
            ("fft3", 1) | ("ifft3", 1) => {
                let arr = match &a[0] {
                    Val::CArr(arr) => arr.clone(),
                    Val::Arr(_) => return Err(format!("`{callee}` needs a complex array -- this one is real. Allocate with `czeros`")),
                    _ => return Err(format!("`{callee}` takes one rank-3 complex array")),
                };
                let shape = arr.borrow().shape.clone();
                if shape.len() != 3 {
                    return Err(format!("`{callee}` needs a rank-3 array; this one is rank {}", shape.len()));
                }
                fft3_in_place(&mut arr.borrow_mut().data, &shape, callee == "ifft3");
                Ok(Val::Num(0.0))
            }
            ("fft", 1) | ("ifft", 1) => {
                let arr = match &a[0] {
                    Val::CArr(arr) => arr.clone(),
                    Val::Arr(_) => {
                        return Err(format!(
                            "`{callee}` needs a complex array -- this one is real. Allocate with `czeros` and copy the samples in"
                        ))
                    }
                    _ => return Err(format!("`{callee}` takes one complex array")),
                };
                let inverse = callee == "ifft";
                fft_in_place(&mut arr.borrow_mut().data, inverse);
                Ok(Val::Num(0.0))
            }
            ("zeros", _) if !a.is_empty() => {
                if a.len() > crate::lower::MAX_RANK {
                    return Err(format!(
                        "`zeros` takes 1 to {} dimensions; this call gives {}",
                        crate::lower::MAX_RANK,
                        a.len()
                    ));
                }
                let dims: Vec<f64> = a.iter().map(|v| v.as_num("a dimension")).collect::<Result<_, _>>()?;
                let (shape, total) = zeros_shape(&dims);
                if total > MAX_ELEMS as u64 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!(
                        "zeros: array of {total} elements exceeds the {MAX_ELEMS}-element limit{LIMIT_OF}"
                    )));
                }
                Ok(new_arr(shape, vec![0.0; total as usize]))
            }
            ("assert_near", 3) => {
                let (x, y, tol) = (num(0)?, num(1)?, num(2)?);
                let pass = (x - y).abs() <= tol;
                self.report_assert("assert_near", pass, format!(
                    "|{} - {}| = {} > {}",
                    format_g(x), format_g(y), format_g((x - y).abs()), format_g(tol)
                ));
                Ok(Val::Num(0.0))
            }
            // The inequality/boolean assert family (same PASS/FAIL discipline
            // as assert_near, same effect on the exit code).
            ("assert_true", 1) => {
                // a boolean, or (as the C lowering also allows) a number, nonzero for true
                let ok = match &a[0] {
                    Val::Bool(b) => *b,
                    _ => num(0)? != 0.0,
                };
                self.report_assert("assert_true", ok, "condition is false".to_string());
                Ok(Val::Num(0.0))
            }
            ("assert_gt", 2) => {
                let (x, y) = (num(0)?, num(1)?);
                self.report_assert("assert_gt", x > y,
                    format!("{} <= {}", format_g(x), format_g(y)));
                Ok(Val::Num(0.0))
            }
            ("assert_lt", 2) => {
                let (x, y) = (num(0)?, num(1)?);
                self.report_assert("assert_lt", x < y,
                    format!("{} >= {}", format_g(x), format_g(y)));
                Ok(Val::Num(0.0))
            }
            // assert_allclose(a, b, tol): elementwise comparison of two arrays;
            // reports the worst element and its flat index on failure.
            ("assert_allclose", 3) => {
                let (xa, ya) = match (&a[0], &a[1]) {
                    (Val::Arr(x), Val::Arr(y)) => (x.clone(), y.clone()),
                    _ => return Err("`assert_allclose` takes (a, b, tol) with a and b arrays".into()),
                };
                let tol = num(2)?;
                let (xb, yb) = (xa.borrow(), ya.borrow());
                if xb.data.len() != yb.data.len() {
                    return Err(format!(
                        "`assert_allclose`: length mismatch ({} vs {})",
                        xb.data.len(), yb.data.len()
                    ));
                }
                let mut worst = 0.0f64;
                let mut at = 0usize;
                for (i, (x, y)) in xb.data.iter().zip(yb.data.iter()).enumerate() {
                    let d = (x - y).abs();
                    if d > worst { worst = d; at = i; }
                }
                self.report_assert("assert_allclose", worst <= tol, format!(
                    "max |a - b| = {} > {} at index {}",
                    format_g(worst), format_g(tol), at
                ));
                Ok(Val::Num(0.0))
            }
            // Seeded PRNG: Park-Miller MINSTD. state' = 16807 * state mod (2^31 - 1),
            // exact in f64 (products stay below 2^53); rand() in (0, 1).
            ("srand", 1) => {
                let s = num(0)?.abs().floor() % 2147483646.0 + 1.0;
                self.rng.set(s);
                Ok(Val::Num(0.0))
            }
            // release_mode(): true in release (viv run --release / -DVIV_RELEASE),
            // 0.0 in debug. Lets Viv-level contract code choose sweep-and-check
            // vs auto-select without the strategy logic living in Rust.
            ("release_mode", 0) => Ok(Val::Bool(self.release)),
            // clock(): the wall time in seconds from a monotonic clock (a
            // `#! timing` program only: crate::check_clock); the interpreter
            // in a browser has no clock to read
            ("clock", 0) => clock_seconds().map(Val::Num),
            ("rand", 0) => {
                let s = (16807.0 * self.rng.get()) % 2147483647.0;
                self.rng.set(s);
                Ok(Val::Num(s / 2147483647.0))
            }
            // tridiag_solve(a, b, c, d): solve a tridiagonal system by the Thomas
            // algorithm. a = sub-diagonal (a[0] unused), b = diagonal, c = super-
            // diagonal (c[n-1] unused), d = right-hand side. No pivoting: it is for
            // the diagonally dominant systems implicit PDE schemes produce. Same
            // operations in the same order as the C runtime's viv_tridiag_solve.
            ("tridiag_solve", 4) => {
                let mut v: Vec<Vec<f64>> = Vec::with_capacity(4);
                for (k, what) in ["sub-diagonal", "diagonal", "super-diagonal", "right-hand side"].iter().enumerate() {
                    match &a[k] {
                        Val::Arr(x) if x.borrow().shape.len() == 1 => v.push(x.borrow().data.clone()),
                        _ => return Err(format!("`tridiag_solve`: the {what} must be a 1-D real array")),
                    }
                }
                let n = v[1].len();
                if n == 0 || v.iter().any(|x| x.len() != n) {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`tridiag_solve` takes four 1-D arrays of one length: (sub, diag, super, rhs)"));
                }
                let (lo, di, up, rhs) = (&v[0], &v[1], &v[2], &v[3]);
                let (mut cp, mut dp, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                if di[0] == 0.0 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`tridiag_solve`: zero pivot at row 0 (the system is not diagonally dominant)"));
                }
                cp[0] = up[0] / di[0];
                dp[0] = rhs[0] / di[0];
                for i in 1..n {
                    let m = di[i] - lo[i] * cp[i - 1];
                    if m == 0.0 {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`tridiag_solve`: zero pivot at row {i} (the system is not diagonally dominant)")));
                    }
                    cp[i] = up[i] / m;
                    dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / m;
                }
                x[n - 1] = dp[n - 1];
                for i in (0..n - 1).rev() {
                    x[i] = dp[i] - cp[i] * x[i + 1];
                }
                Ok(new_arr(vec![n], x))
            }
            // roll(a, k[, axis]): circular shift -- element i moves to i + k, wrapping.
            // The periodic stencil in one line: u - c * (u - roll(u, 1)).
            ("roll", 2) | ("roll", 3) => {
                let Val::Arr(arr) = &a[0] else {
                    return Err("`roll` takes (array, shift) or (array, shift, axis)".into());
                };
                let ab = arr.borrow();
                let rank = ab.shape.len();
                let k = num(1)?;
                let axis = if a.len() == 3 { num(2)? as usize } else { 0 };
                if a.len() == 2 && rank != 1 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`roll(a, k)` is for a 1-D array; give the axis for a matrix: `roll(A, k, axis)`"));
                }
                if axis >= rank {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`roll`: axis {axis} out of range for a rank-{rank} array")));
                }
                let len = ab.shape[axis];
                let outer: usize = ab.shape[..axis].iter().product();
                let inner: usize = ab.shape[axis + 1..].iter().product();
                let sh = (((k as i64) % len as i64) + len as i64) as usize % len;
                let mut data = vec![0.0; ab.data.len()];
                for o in 0..outer {
                    for j in 0..len {
                        let t = (j + sh) % len;
                        for q in 0..inner {
                            data[(o * len + t) * inner + q] = ab.data[(o * len + j) * inner + q];
                        }
                    }
                }
                Ok(new_arr(ab.shape.clone(), data))
            }
            // sort(a): in-place ascending sort of a 1-D array.
            ("sort", 1) => {
                let arr = match &a[0] {
                    Val::Arr(x) => x.clone(),
                    _ => return Err("`sort` takes a 1-D array".into()),
                };
                let mut ab = arr.borrow_mut();
                if ab.shape.len() != 1 {
                    return Err("`sort` takes a 1-D array".into());
                }
                ab.data.sort_by(sort_order);
                Ok(Val::Num(0.0))
            }
            // Composite Simpson's rule, matching the C runtime exactly: an odd
            // interval count is bumped up to the next even number, then the
            // 1-4-2-…-4-1 weights over h = (b-a)/n.
            ("simpson", 4) => {
                let clos = self.as_closure(&a[0], "simpson")?;
                let (lo, hi, nreq) = (num(1)?, num(2)?, num(3)?);
                let mut n = nreq;
                if n % 2.0 != 0.0 {
                    n += 1.0;
                }
                let h = (hi - lo) / n;
                let mut s = self.apply1(&clos, lo)? + self.apply1(&clos, hi)?;
                let mut k = 0.0;
                while k < n - 1.0 {
                    let x = lo + (k + 1.0) * h;
                    if (k + 1.0) % 2.0 == 1.0 {
                        s += 4.0 * self.apply1(&clos, x)?;
                    } else {
                        s += 2.0 * self.apply1(&clos, x)?;
                    }
                    k += 1.0;
                }
                Ok(Val::Num(s * h / 3.0))
            }
            // Composite trapezoid rule, matching the C runtime: endpoints at
            // half weight, interior points at full weight, times h.
            ("trapezoid", 4) => {
                let clos = self.as_closure(&a[0], "trapezoid")?;
                let (lo, hi, n) = (num(1)?, num(2)?, num(3)?);
                let h = (hi - lo) / n;
                let mut s = (self.apply1(&clos, lo)? + self.apply1(&clos, hi)?) / 2.0;
                let mut k = 0.0;
                while k < n - 1.0 {
                    s += self.apply1(&clos, lo + (k + 1.0) * h)?;
                    k += 1.0;
                }
                Ok(Val::Num(s * h))
            }
            // Exact forward-mode (dual-number) derivative of the closure's body
            // at x0 — the same rule the C backend lowers (see `lower_dual`).
            ("deriv", 2) => {
                let clos = self.as_closure(&a[0], "deriv")?;
                if clos.params.len() != 1 {
                    return Err("`deriv`'s closure must take exactly 1 argument".into());
                }
                let x0 = num(1)?;
                let (_val, der) = self.eval_dual(&clos.body, &clos.params[0], x0, &clos.captured)?;
                Ok(Val::Num(der))
            }
            // linspace(lo, hi, n): n evenly spaced points, endpoints included,
            // as NumPy gives them -- mirrors the C runtime's `viv_linspace`
            // (n = 1 is [lo], n = 0 is empty, the last point is hi exactly).
            // histogram(a, lo, hi, nbins): counts per bin, like the C runtime's viv_histogram
            ("histogram", 4) => {
                let Val::Arr(arr) = &a[0] else { return Err("`histogram` counts a real array".into()) };
                let (lo, hi, nbd) = (num(1)?, num(2)?, num(3)?);
                let nb = if (nbd as i64) < 1 { 1usize } else { nbd as usize };
                let w = (hi - lo) / nb as f64;
                let mut data = vec![0.0; nb];
                for &v in arr.borrow().data.iter() {
                    if v < lo || v > hi || w <= 0.0 { continue; }
                    let mut b = ((v - lo) / w) as usize;
                    if b >= nb { b = nb - 1; }
                    data[b] += 1.0;
                }
                Ok(new_arr(vec![nb], data))
            }
            ("linspace", 3) => {
                let (lo, hi, nd) = (num(0)?, num(1)?, num(2)?);
                if (nd as i64) < 0 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`linspace`: the number of points {} is negative", nd as i64)));
                }
                let n = nd as usize;
                let h = if n > 1 { (hi - lo) / (n as f64 - 1.0) } else { 0.0 };
                let mut data: Vec<f64> = (0..n).map(|i| lo + h * i as f64).collect();
                if n > 1 {
                    data[n - 1] = hi;
                }
                Ok(new_arr(vec![n], data))
            }
            // 2-D transpose, mirroring the C runtime's `viv_transpose`.
            // transpose of a COMPLEX matrix: plain transpose, no conjugation
            // (the adjoint is conj(transpose(A)), spelled out where it is meant)
            ("transpose", 1) if matches!(&a[0], Val::CArr(_)) => {
                let Val::CArr(arr) = &a[0] else { unreachable!() };
                let ab = arr.borrow();
                if ab.shape.len() != 2 {
                    return Err("`transpose` takes a 2-D array".into());
                }
                let (rows, cols) = (ab.shape[0], ab.shape[1]);
                let mut data = vec![0.0; 2 * rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        data[2 * (j * rows + i)] = ab.data[2 * (i * cols + j)];
                        data[2 * (j * rows + i) + 1] = ab.data[2 * (i * cols + j) + 1];
                    }
                }
                Ok(new_carr(vec![cols, rows], data))
            }
            // slab(A, k): a COPY of the k-th slab along the leading axis -- row k
            // of a matrix, plane k of a rank-3 array. Rank drops by one. With
            // setslab it is how a program works on one row of a block of vectors
            // at native speed instead of element by element.
            ("slab", 2) => {
                let (arr, cplx) = match &a[0] {
                    Val::Arr(x) => (x.clone(), false),
                    Val::CArr(x) => (x.clone(), true),
                    _ => return Err("`slab` takes (array, index)".into()),
                };
                let ab = arr.borrow();
                if ab.shape.len() < 2 {
                    return Err("`slab` needs an array of rank 2 or more".into());
                }
                let k = num(1)? as usize;
                if k >= ab.shape[0] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`slab` index {k} out of range (leading dimension {})", ab.shape[0])));
                }
                let w = if cplx { 2 } else { 1 };
                let len: usize = ab.shape[1..].iter().product();
                let data = ab.data[w * k * len..w * (k + 1) * len].to_vec();
                let shape = ab.shape[1..].to_vec();
                Ok(if cplx { new_carr(shape, data) } else { new_arr(shape, data) })
            }
            // setslab(A, k, S): overwrite the k-th leading slab of A with S.
            ("setslab", 3) => {
                let (dst, src, cplx) = match (&a[0], &a[2]) {
                    (Val::Arr(x), Val::Arr(y)) => (x.clone(), y.clone(), false),
                    (Val::CArr(x), Val::CArr(y)) => (x.clone(), y.clone(), true),
                    _ => return Err("`setslab` takes (array, index, slab) with both arrays real or both complex".into()),
                };
                let k = num(1)? as usize;
                let sb = src.borrow().data.clone();
                let mut db = dst.borrow_mut();
                if db.shape.len() < 2 || k >= db.shape[0] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, "`setslab`: index out of range, or the array has rank below 2"));
                }
                let w = if cplx { 2 } else { 1 };
                let len: usize = db.shape[1..].iter().product();
                if sb.len() != w * len {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`setslab`: the slab has {} elements, the array's slabs have {len}", sb.len() / w)));
                }
                db.data[w * k * len..w * (k + 1) * len].copy_from_slice(&sb);
                Ok(Val::Num(0.0))
            }
            // take(A, idx): gather. A is viewed FLAT (row-major); idx is a real
            // array of flat indices; the result has idx's shape. put is its
            // inverse. Together they move plane-wave coefficients between a
            // sphere of G-vectors and an FFT box without a Viv-level loop.
            ("take", 2) => {
                let (arr, cplx) = match &a[0] {
                    Val::Arr(x) => (x.clone(), false),
                    Val::CArr(x) => (x.clone(), true),
                    _ => return Err("`take` takes (array, indices)".into()),
                };
                let Val::Arr(ix) = &a[1] else {
                    return Err("`take`: the indices must be a real array".into());
                };
                let (ab, ib) = (arr.borrow(), ix.borrow());
                let w = if cplx { 2 } else { 1 };
                let total = ab.data.len() / w;
                let mut data = Vec::with_capacity(w * ib.data.len());
                for &f in &ib.data {
                    let k = f as usize;
                    if f < 0.0 || k >= total {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`take`: index {f} out of range (the array has {total} elements)")));
                    }
                    data.extend_from_slice(&ab.data[w * k..w * (k + 1)]);
                }
                let shape = ib.shape.clone();
                Ok(if cplx { new_carr(shape, data) } else { new_arr(shape, data) })
            }
            ("put", 3) => {
                let (dst, src, cplx) = match (&a[0], &a[2]) {
                    (Val::Arr(x), Val::Arr(y)) => (x.clone(), y.clone(), false),
                    (Val::CArr(x), Val::CArr(y)) => (x.clone(), y.clone(), true),
                    _ => return Err("`put` takes (array, indices, values) with both arrays real or both complex".into()),
                };
                let Val::Arr(ix) = &a[1] else {
                    return Err("`put`: the indices must be a real array".into());
                };
                let (ib, sb) = (ix.borrow(), src.borrow().data.clone());
                let w = if cplx { 2 } else { 1 };
                if sb.len() != w * ib.data.len() {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`put`: as many values as indices, please"));
                }
                let mut db = dst.borrow_mut();
                let total = db.data.len() / w;
                for (j, &f) in ib.data.iter().enumerate() {
                    let k = f as usize;
                    if f < 0.0 || k >= total {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("`put`: index {f} out of range (the array has {total} elements)")));
                    }
                    db.data[w * k..w * (k + 1)].copy_from_slice(&sb[w * j..w * (j + 1)]);
                }
                Ok(Val::Num(0.0))
            }
            ("transpose", 1) => {
                let arr = match &a[0] {
                    Val::Arr(x) => x.clone(),
                    _ => return Err("`transpose` takes a 2-D array".into()),
                };
                let ab = arr.borrow();
                if ab.shape.len() != 2 {
                    return Err("`transpose` takes a 2-D array".into());
                }
                let (rows, cols) = (ab.shape[0], ab.shape[1]);
                let mut data = vec![0.0; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        data[j * rows + i] = ab.data[i * cols + j];
                    }
                }
                Ok(new_arr(vec![cols, rows], data))
            }
            // eig_sym(A, vals_out, vecs_out): symmetric eigendecomposition by
            // cyclic Jacobi with max-off-diagonal pivoting — a line-for-line
            // port of the C runtime's `viv_eig_sym` (100 sweeps, 1e-12 stop,
            // eigenvalues sorted ascending, eigenvectors in COLUMNS), writing
            // through the shared array handles like the C out-parameters.
            // lin_solve(A, b, x_out): linear solve by LU with partial pivoting.
            // Debug runs the residual CERTIFICATE ||Ax - b|| against the input
            // scale and fails loudly; release trusts the factorization.
            ("lin_solve", 3) => {
                let (aa, bb, xx) = match (&a[0], &a[1], &a[2]) {
                    (Val::Arr(x), Val::Arr(y), Val::Arr(z)) => (x.clone(), y.clone(), z.clone()),
                    _ => return Err("`lin_solve` takes (A, b, x_out) arrays".into()),
                };
                let ab = aa.borrow();
                let bv = bb.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`lin_solve` needs a square 2-D matrix"));
                }
                let n = ab.shape[0];
                if bv.data.len() < n {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`lin_solve`: b is too small"));
                }
                let x = lu_solve_vec(&ab.data, &bv.data, n)?;
                if !self.release {
                    let anorm: f64 = ab.data.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let xnorm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let bnorm: f64 = bv.data[..n].iter().map(|v| v * v).sum::<f64>().sqrt();
                    let mut rmax = 0.0f64;
                    for i in 0..n {
                        let mut s = -bv.data[i];
                        for j in 0..n {
                            s += ab.data[i * n + j] * x[j];
                        }
                        rmax = rmax.max(s.abs());
                    }
                    let tol = 1e-8 * (anorm * xnorm + bnorm).max(1e-300);
                    if rmax > tol {
                        return Err(format!(
                            "`lin_solve`: SOLVER CERTIFICATE FAILED — residual ||Ax-b||_inf = \
                             {rmax:.3e} exceeds {tol:.3e}; the system is ill-conditioned beyond \
                             working precision"
                        ));
                    }
                }
                let mut xb = xx.borrow_mut();
                if xb.data.len() < n {
                    return Err("`lin_solve`: x_out is too small".into());
                }
                xb.data[..n].copy_from_slice(&x[..n]);
                Ok(Val::Num(0.0))
            }
            // lin_solve_multi(A, B, X_out): solve A X = B for an n x m
            // right-hand-side block, factoring A ONCE (O(n^3 + n^2 m) instead
            // of one factorization per column). Debug verifies the worst
            // column residual as a certificate.
            ("lin_solve_multi", 3) => {
                let (aa, bb, xx) = match (&a[0], &a[1], &a[2]) {
                    (Val::Arr(x), Val::Arr(y), Val::Arr(z)) => (x.clone(), y.clone(), z.clone()),
                    _ => return Err("`lin_solve_multi` takes (A, B, X_out) arrays".into()),
                };
                let ab = aa.borrow();
                let bv = bb.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`lin_solve_multi` needs a square 2-D matrix"));
                }
                let n = ab.shape[0];
                if bv.shape.len() != 2 || bv.shape[0] != n {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`lin_solve_multi`: B must be n x m"));
                }
                let m = bv.shape[1];
                let mut lu = ab.data.clone();
                let piv = lu_factor(&mut lu, n).map_err(|e| format!("`lin_solve_multi`: {e}"))?;
                let mut x = vec![0.0f64; n * m];
                let mut col = vec![0.0f64; n];
                for j in 0..m {
                    for i in 0..n {
                        col[i] = bv.data[i * m + j];
                    }
                    lu_solve_factored(&lu, &piv, n, &mut col);
                    for i in 0..n {
                        x[i * m + j] = col[i];
                    }
                }
                if !self.release {
                    let anorm: f64 = ab.data.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let mut worst = 0.0f64;
                    let mut wtol = 0.0f64;
                    for j in 0..m {
                        let mut xnorm = 0.0f64;
                        let mut bnorm = 0.0f64;
                        let mut rmax = 0.0f64;
                        for i in 0..n {
                            xnorm += x[i * m + j] * x[i * m + j];
                            bnorm += bv.data[i * m + j] * bv.data[i * m + j];
                        }
                        for i in 0..n {
                            let mut s = -bv.data[i * m + j];
                            for k in 0..n {
                                s += ab.data[i * n + k] * x[k * m + j];
                            }
                            rmax = rmax.max(s.abs());
                        }
                        let tol = 1e-8 * (anorm * xnorm.sqrt() + bnorm.sqrt()).max(1e-300);
                        if rmax / tol > worst / wtol.max(1e-300) {
                            worst = rmax;
                            wtol = tol;
                        }
                    }
                    if worst > wtol {
                        return Err(format!(
                            "`lin_solve_multi`: SOLVER CERTIFICATE FAILED — worst column residual \
                             {worst:.3e} exceeds {wtol:.3e}"
                        ));
                    }
                }
                let mut xb = xx.borrow_mut();
                if xb.data.len() < n * m {
                    return Err("`lin_solve_multi`: X_out is too small".into());
                }
                xb.data[..n * m].copy_from_slice(&x[..n * m]);
                Ok(Val::Num(0.0))
            }
            // cholesky(A, L_out): A = L L^T, L lower triangular. The SPD
            // contract check is inherent (fails on a non-positive pivot).
            ("cholesky", 2) => {
                let (aa, ll) = match (&a[0], &a[1]) {
                    (Val::Arr(x), Val::Arr(y)) => (x.clone(), y.clone()),
                    _ => return Err("`cholesky` takes (A, L_out) arrays".into()),
                };
                let ab = aa.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`cholesky` needs a square 2-D matrix"));
                }
                let n = ab.shape[0];
                let l = cholesky_lower(&ab.data, n)?;
                let mut lb = ll.borrow_mut();
                if lb.data.len() < n * n {
                    return Err("`cholesky`: L_out is too small".into());
                }
                lb.data[..n * n].copy_from_slice(&l[..n * n]);
                Ok(Val::Num(0.0))
            }
            // eig_general(A, re_out, im_out): eigenvalues of a general real
            // matrix (Hessenberg + Francis double-shift QR). Complex values are
            // conjugate pairs split across the two output arrays; sorted by
            // (re, im). Debug verifies the trace invariants tr(A) = sum(re)
            // and tr(A^2) = sum(re^2 - im^2) as method-independent
            // certificates; release skips them.
            ("eig_general", 3) => {
                let (aa, rr, ii) = match (&a[0], &a[1], &a[2]) {
                    (Val::Arr(x), Val::Arr(y), Val::Arr(z)) => (x.clone(), y.clone(), z.clone()),
                    _ => return Err("`eig_general` takes (A, re_out, im_out) arrays".into()),
                };
                let ab = aa.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`eig_general` needs a square 2-D matrix"));
                }
                let n = ab.shape[0];
                let (wr, wi) = general_eigvals(&ab.data, n)?;
                if !self.release {
                    let anorm: f64 =
                        ab.data.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
                    let tr: f64 = (0..n).map(|i| ab.data[i * n + i]).sum();
                    let s1: f64 = wr.iter().sum();
                    let mut tr2 = 0.0f64;
                    for i in 0..n {
                        for j in 0..n {
                            tr2 += ab.data[i * n + j] * ab.data[j * n + i];
                        }
                    }
                    let s2: f64 = (0..n).map(|k| wr[k] * wr[k] - wi[k] * wi[k]).sum();
                    let tol1 = 1e-8 * (n as f64) * anorm;
                    let tol2 = 1e-8 * (n as f64) * anorm * anorm;
                    if (tr - s1).abs() > tol1 || (tr2 - s2).abs() > tol2 {
                        return Err(format!(
                            "`eig_general`: SOLVER CERTIFICATE FAILED — trace invariants do not \
                             close (|tr A - sum re| = {:.3e} vs {:.3e}, |tr A^2 - sum(re^2-im^2)| \
                             = {:.3e} vs {:.3e})",
                            (tr - s1).abs(),
                            tol1,
                            (tr2 - s2).abs(),
                            tol2
                        ));
                    }
                }
                {
                    let mut rb = rr.borrow_mut();
                    if rb.data.len() < n {
                        return Err("`eig_general`: re_out is too small".into());
                    }
                    rb.data[..n].copy_from_slice(&wr[..n]);
                }
                {
                    let mut ib = ii.borrow_mut();
                    if ib.data.len() < n {
                        return Err("`eig_general`: im_out is too small".into());
                    }
                    ib.data[..n].copy_from_slice(&wi[..n]);
                }
                Ok(Val::Num(0.0))
            }
            // eig_near(A, sigma, k, re_out, im_out): k CONVERGED eigenvalues
            // of A near the real shift sigma, by shift-invert Arnoldi:
            // O(n^3/3) for one LU + O(m n^2) solves instead of the full
            // O(10 n^3) QR spectrum. Debug certificate: the computation runs
            // at Krylov depths m and m+10 and must agree — actual
            // non-convergence fails loudly.
            //
            // HONEST LIMIT (measured on the 16^3 advective SAW operator, and
            // reproduced by the identical algorithm in numpy): for severely
            // NON-NORMAL matrices, eigenvalues with ill-conditioned
            // eigenvectors can be closer to sigma yet invisible to any Krylov
            // space — the returned pairs are true, converged eigenpairs, but
            // the global "k nearest" property is not certifiable from inside
            // the iteration. For non-normal stability questions, the full
            // spectrum (eig_general, or the compiled notebook twin at scale)
            // is the reliable tool.
            ("eig_near", 5) => {
                let (aa, rr, ii) = match (&a[0], &a[3], &a[4]) {
                    (Val::Arr(x), Val::Arr(y), Val::Arr(z)) => (x.clone(), y.clone(), z.clone()),
                    _ => return Err("`eig_near` takes (A, sigma, k, re_out, im_out)".into()),
                };
                let sigma = a[1].as_num("`eig_near`'s sigma")?;
                let kf = a[2].as_num("`eig_near`'s k")?;
                let ab = aa.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`eig_near` needs a square 2-D matrix"));
                }
                let n = ab.shape[0];
                let k = kf as usize;
                if k == 0 || k > n {
                    return Err("`eig_near`: k must be between 1 and n".into());
                }
                let m = n.min((4 * k + 20).max(40));
                let (wr, wi) = arnoldi_near(&ab.data, n, sigma, k, m)?;
                if !self.release && m + 10 <= n {
                    let scale: f64 =
                        ab.data.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
                    let (wr2, wi2) = arnoldi_near(&ab.data, n, sigma, k, m + 10)?;
                    let mut dev = 0.0f64;
                    for i in 0..k.min(wr2.len()).min(wr.len()) {
                        dev = dev.max((wr[i] - wr2[i]).abs()).max((wi[i] - wi2[i]).abs());
                    }
                    if dev > 1e-7 * scale {
                        return Err(format!(
                            "`eig_near`: SOLVER CERTIFICATE FAILED — Krylov depths {m} and {} \
                             disagree by {dev:.3e}; the requested eigenvalues have not converged \
                             (ask for fewer, or move sigma closer to them)",
                            m + 10
                        ));
                    }
                }
                {
                    let mut rb = rr.borrow_mut();
                    if rb.data.len() < k {
                        return Err("`eig_near`: re_out is too small".into());
                    }
                    rb.data[..wr.len()].copy_from_slice(&wr);
                }
                {
                    let mut ib = ii.borrow_mut();
                    if ib.data.len() < k {
                        return Err("`eig_near`: im_out is too small".into());
                    }
                    ib.data[..wi.len()].copy_from_slice(&wi);
                }
                Ok(Val::Num(0.0))
            }
            // spectrum_sym(A, vals_out): the CONTRACTED solve. The contract is
            // "A is symmetric; vals_out receives its full spectrum, ascending".
            // Debug (default): verify the symmetry contract, then run TWO
            // independent methods (cyclic Jacobi; Householder + implicit-shift
            // QL) and fail loudly if they dissent. Release: auto-select the
            // cheap direct method (QL) and skip the checks.
            ("spectrum_sym", 2) => {
                let (aa, vals) = match (&a[0], &a[1]) {
                    (Val::Arr(x), Val::Arr(y)) => (x.clone(), y.clone()),
                    _ => return Err("`spectrum_sym` takes (A, vals_out) arrays".into()),
                };
                let ab = aa.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`spectrum_sym` needs a square 2-D matrix"));
                }
                if ab.data.iter().any(|x| !x.is_finite()) {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`spectrum_sym`: the matrix has a NaN or an infinity"));
                }
                let n = ab.shape[0];
                let scale: f64 = ab.data.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
                let out = if self.release {
                    tridiag_eigvals(&ab.data, n)?
                } else {
                    // Contract check: symmetry.
                    let mut asym = 0.0f64;
                    for i in 0..n {
                        for j in (i + 1)..n {
                            asym = asym.max((ab.data[i * n + j] - ab.data[j * n + i]).abs());
                        }
                    }
                    if asym > 1e-9 * scale {
                        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!(
                            "`spectrum_sym`: contract violation — the matrix is not symmetric \
                             (max |A[i,j] - A[j,i]| = {asym:.3e}). Fix the matrix or use a \
                             general eigensolver."
                        )));
                    }
                    // The sweep: two independent algorithm families.
                    let jac = jacobi_eigvals(&ab.data, n);
                    let ql = tridiag_eigvals(&ab.data, n)?;
                    let mut dev = 0.0f64;
                    for i in 0..n {
                        dev = dev.max((jac[i] - ql[i]).abs());
                    }
                    if dev > 1e-9 * scale {
                        let fmt = |v: &[f64]| {
                            v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")
                        };
                        return Err(format!(
                            "`spectrum_sym`: SOLVER SWEEP DISSENT\n  jacobi: [{}]\n  ql:     [{}]\n  \
                             max deviation {dev:.3e} exceeds tolerance {:.3e}\n  a contracted solve \
                             failed its cross-check; one of the methods (or the matrix's declared \
                             structure) is wrong",
                            fmt(&jac),
                            fmt(&ql),
                            1e-9 * scale
                        ));
                    }
                    jac
                };
                let mut vb = vals.borrow_mut();
                if vb.data.len() < n {
                    return Err("`spectrum_sym`: vals_out is too small".into());
                }
                vb.data[..n].copy_from_slice(&out[..n]);
                Ok(Val::Num(0.0))
            }
            // eig_herm(A, vals, vecs): the complex Hermitian eigenproblem. A and
            // vecs are complex n x n, vals is a REAL n-vector (a Hermitian
            // matrix has real eigenvalues). Ascending; eigenvectors in columns,
            // the same contract as eig_sym. Householder + implicit QL.
            ("eig_herm", 3) => {
                let (aa, vals, vecs) = match (&a[0], &a[1], &a[2]) {
                    (Val::CArr(x), Val::Arr(y), Val::CArr(z)) => (x.clone(), y.clone(), z.clone()),
                    _ => {
                        return Err("`eig_herm` takes (A, vals, vecs): A and vecs complex n x n (`czeros(n, n)`), vals a real n-vector (`zeros(n)`)".into())
                    }
                };
                let n = {
                    let ab = aa.borrow();
                    if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                        return Err("`eig_herm` needs a square 2-D complex matrix".into());
                    }
                    ab.shape[0]
                };
                if vals.borrow().data.len() != n {
                    return Err(format!("`eig_herm`: vals has {} elements, the matrix is {n} x {n}", vals.borrow().data.len()));
                }
                if vecs.borrow().shape != vec![n, n] {
                    return Err(format!("`eig_herm`: vecs must be {n} x {n}"));
                }
                // The contract, checked in debug: A must actually be Hermitian.
                // Release skips it, as spectrum_sym skips its symmetry check.
                if !self.release {
                    let ab = aa.borrow();
                    let mut worst = 0.0f64;
                    let mut scale = 0.0f64;
                    for i in 0..n {
                        for j in 0..n {
                            let (p, q) = (2 * (i * n + j), 2 * (j * n + i));
                            worst = worst.max((ab.data[p] - ab.data[q]).abs()).max((ab.data[p + 1] + ab.data[q + 1]).abs());
                            scale = scale.max(ab.data[p].abs()).max(ab.data[p + 1].abs());
                        }
                    }
                    if worst > 1e-9 * scale.max(1e-300) {
                        return Err(format!("`eig_herm`: the matrix is not Hermitian (largest |A - A^H| entry {worst:.3e})"));
                    }
                }
                let (w, v) = herm_eig(&aa.borrow().data, n)?;
                vals.borrow_mut().data = w;
                vecs.borrow_mut().data = v;
                Ok(Val::Num(0.0))
            }
            ("eig_sym", 3) => {
                let (aa, vals, vecs) = match (&a[0], &a[1], &a[2]) {
                    (Val::Arr(x), Val::Arr(y), Val::Arr(z)) => (x.clone(), y.clone(), z.clone()),
                    _ => return Err("`eig_sym` takes (A, vals_out, vecs_out) arrays".into()),
                };
                let ab = aa.borrow();
                if ab.shape.len() != 2 || ab.shape[0] != ab.shape[1] {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`eig_sym` needs a square 2-D matrix"));
                }
                // a NaN would make the ordering of the eigenvalues undefined (it
                // panicked the sort); every engine stops here with these words
                if ab.data.iter().any(|x| !x.is_finite()) {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`eig_sym`: the matrix has a NaN or an infinity"));
                }
                let n = ab.shape[0];
                let mut m = ab.data.clone();
                let mut v = vec![0.0; n * n];
                for i in 0..n {
                    v[i * n + i] = 1.0;
                }
                // scale for the convergence test: the Frobenius norm of A
                let scale: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
                'sweeps: for _sweep in 0..100 {
                    // cyclic sweep: rotate every above-threshold (p, q) pair.
                    // (The old code did ONE rotation per "sweep" — fine below
                    // ~14x14, silent garbage above. Found by the HFB deck.)
                    let mut off = 0.0f64;
                    for i in 0..n {
                        for j in (i + 1)..n {
                            off += m[i * n + j] * m[i * n + j];
                        }
                    }
                    if off.sqrt() < 1e-12 * scale {
                        break 'sweeps;
                    }
                    for p in 0..n {
                        for q in (p + 1)..n {
                            let apq = m[p * n + q];
                            if apq.abs() < 1e-15 * scale {
                                continue;
                            }
                            let (app, aqq) = (m[p * n + p], m[q * n + q]);
                            let theta = 0.5 * (2.0 * apq).vatan2(app - aqq);
                            let (c, s) = (theta.vcos(), theta.vsin());
                            for k in 0..n {
                                if k != p && k != q {
                                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                                    let nkp = c * mkp + s * mkq;
                                    let nkq = -s * mkp + c * mkq;
                                    m[k * n + p] = nkp;
                                    m[p * n + k] = nkp;
                                    m[k * n + q] = nkq;
                                    m[q * n + k] = nkq;
                                }
                            }
                            m[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
                            m[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
                            m[p * n + q] = 0.0;
                            m[q * n + p] = 0.0;
                            for k in 0..n {
                                let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                                v[k * n + p] = c * vkp + s * vkq;
                                v[k * n + q] = -s * vkp + c * vkq;
                            }
                        }
                    }
                }
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&i, &j| m[i * n + i].partial_cmp(&m[j * n + j]).unwrap());
                {
                    let mut vb = vals.borrow_mut();
                    if vb.data.len() < n {
                        return Err("`eig_sym`: vals_out is too small".into());
                    }
                    for i in 0..n {
                        vb.data[i] = m[idx[i] * n + idx[i]];
                    }
                }
                {
                    let mut wb = vecs.borrow_mut();
                    if wb.data.len() < n * n {
                        return Err("`eig_sym`: vecs_out is too small".into());
                    }
                    for i in 0..n {
                        for r in 0..n {
                            wb.data[r * n + i] = v[r * n + idx[i]];
                        }
                    }
                }
                Ok(Val::Num(0.0))
            }
            _ => {
                if callee == "range" {
                    self.err_notes.borrow_mut().push(crate::hints::RANGE.to_string());
                }
                Err(format!("unknown function `{callee}` (or wrong argument count)"))
            }
        };
        match filled {
            Some(x) => result.map(|_| x),
            None => result,
        }
    }

    fn as_closure(&self, v: &Val, what: &str) -> Result<Rc<Closure>, String> {
        match v {
            Val::Closure(c) => Ok(c.clone()),
            Val::Func(name, _) => Err(format!(
                "`{what}`'s first argument must be a `fn(x) -> ...` lambda, not the def `{name}` \
                 (wrap it: `fn(x) -> {name}(x)`)"
            )),
            _ => Err(format!("`{what}`'s first argument must be a `fn(x) -> ...`")),
        }
    }

    /// Apply a closure to a single scalar argument (the shape the quadrature
    /// builtins need).
    fn apply1(&self, clos: &Closure, x: f64) -> Result<f64, String> {
        self.call_closure(clos, vec![Val::Num(x)])?.as_num("a closure result")
    }

    /// Call a closure: bind its parameters over a copy of its captured
    /// environment, then evaluate its (single-expression) body.
    fn call_closure(&self, clos: &Closure, args: Vec<Val>) -> Result<Val, String> {
        if args.len() != clos.params.len() {
            return Err(format!(
                "closure expects {} argument(s), got {}",
                clos.params.len(),
                args.len()
            ));
        }
        let mut local = clos.captured.clone();
        for (p, v) in clos.params.iter().zip(args) {
            local.insert(p.clone(), v);
        }
        self.eval(&clos.body, &mut local)
    }

    /// Forward-mode automatic differentiation of a scalar expression: returns
    /// `(value, d/dx value)` at the point where `xname == xval`. This mirrors the
    /// C backend's `lower_dual` exactly (so `deriv` gives an identical result):
    /// it supports `+ - * /`, `exp`, `sqrt`, and `pow(_, c)` for a constant `c`.
    fn eval_dual(
        &self,
        e: &Expr,
        xname: &str,
        xval: f64,
        env: &HashMap<String, Val>,
    ) -> Result<(f64, f64), String> {
        match e {
            Expr::Num { value, .. } => Ok((*value, 0.0)),
            Expr::Var { name, .. } => {
                if name == xname {
                    Ok((xval, 1.0))
                } else {
                    // A value that does not depend on x: derivative 0.
                    let v = env
                        .get(name)
                        .cloned()
                        .or_else(|| self.globals.borrow().get(name).cloned())
                        .ok_or_else(|| format!("undefined variable `{name}` in `deriv`"))?;
                    Ok((v.as_num("a value in `deriv`")?, 0.0))
                }
            }
            Expr::Neg { operand, .. } => {
                let (av, ad) = self.eval_dual(operand, xname, xval, env)?;
                Ok((-av, -ad))
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let (av, ad) = self.eval_dual(lhs, xname, xval, env)?;
                let (bv, bd) = self.eval_dual(rhs, xname, xval, env)?;
                match op {
                    BinOp::Add => Ok((av + bv, ad + bd)),
                    BinOp::Sub => Ok((av - bv, ad - bd)),
                    BinOp::Mul => Ok((av * bv, av * bd + ad * bv)),
                    BinOp::Div => Ok((av / bv, (ad * bv - av * bd) / (bv * bv))),
                    _ => Err("`deriv` supports + - * / (not comparisons or logic)".into()),
                }
            }
            // a program's own def wins over the builtin, and deriv knows only the
            // builtins' derivatives (lower_dual refuses the same call)
            Expr::Call { callee, .. } if self.funcs.contains_key(callee.as_str()) => Err(format!(
                "`deriv` cannot differentiate through `{callee}`: this program defines its own `{callee}`, and `deriv` knows only the builtins' derivatives"
            )),
            Expr::Call { callee, args, .. } => match (callee.as_str(), args.len()) {
                ("exp", 1) => {
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    let ev = av.vexp();
                    Ok((ev, ev * ad))
                }
                ("sqrt", 1) => {
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    let r = av.sqrt();
                    Ok((r, ad / (2.0 * r)))
                }
                ("pow", 2) => {
                    let c = match &args[1] {
                        Expr::Num { value, .. } => *value,
                        _ => return Err(
                            "`deriv` differentiates `pow(x, c)` only for a constant exponent".into(),
                        ),
                    };
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    Ok((av.vpow(c), c * av.vpow(c - 1.0) * ad))
                }
                ("sin", 1) => {
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    Ok((av.vsin(), av.vcos() * ad))
                }
                ("cos", 1) => {
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    Ok((av.vcos(), -av.vsin() * ad))
                }
                ("log", 1) => {
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    Ok((av.vln(), ad / av))
                }
                ("tan", 1) => {
                    let (av, ad) = self.eval_dual(&args[0], xname, xval, env)?;
                    let t = av.vtan();
                    Ok((t, (1.0 + t * t) * ad))
                }
                _ => Err(format!(
                    "`deriv` cannot differentiate through `{callee}` (supports exp/sqrt/pow/sin/cos/log/tan and + - * /)"
                )),
            },
            _ => Err("`deriv` supports scalar arithmetic expressions only".into()),
        }
    }

    fn array_literal(&self, elems: &[Expr], scope: &mut HashMap<String, Val>) -> Result<Val, String> {
        if elems.is_empty() {
            return Ok(new_arr(vec![0], vec![]));
        }
        // Rank-2 if the first element is itself an array literal.
        if matches!(elems[0], Expr::ArrLit { .. }) {
            let mut rows: Vec<Vec<f64>> = Vec::new();
            for e in elems {
                match self.eval(e, scope)? {
                    Val::Arr(a) => {
                        let ab = a.borrow();
                        if ab.shape.len() != 1 {
                            return Err("matrix rows must be 1-D arrays".into());
                        }
                        rows.push(ab.data.clone());
                    }
                    _ => return Err("inconsistent array literal (mixed scalars and rows)".into()),
                }
            }
            let cols = rows[0].len();
            if rows.iter().any(|r| r.len() != cols) {
                return Err("matrix rows must all have the same length".into());
            }
            let data: Vec<f64> = rows.into_iter().flatten().collect();
            Ok(new_arr(vec![elems.len(), cols], data))
        } else {
            // Evaluate every element first. If they are all numbers, build the
            // packed f64 array (the fast path). Otherwise box them into a VArr,
            // so `[r_bisect, r_illinois, r_safe]` -- a registry of strategies --
            // becomes a first-class value.
            let vals: Vec<Val> = elems
                .iter()
                .map(|e| self.eval(e, scope))
                .collect::<Result<_, _>>()?;
            // an array of records: every element a record of one type
            if let Val::Record(first) = &vals[0] {
                let name = first.name.clone();
                for v in &vals {
                    match v {
                        Val::Record(r) if r.name == name => {}
                        other => return Err(format!("an array of records holds records of one type; the first is a {name} and this is {}", crate::lower::a_or_an(&val_kind(other)))),
                    }
                }
                return Ok(Val::Recs(Rc::new(vals)));
            }
            if vals.iter().any(|v| matches!(v, Val::Record(_))) {
                return Err("an array holds numbers, functions or records of one type -- not a mixture".into());
            }
            if vals.iter().any(|v| matches!(v, Val::Map(_))) {
                return Err(maps::NO_MAP_ARRAYS.into());
            }
            if vals.iter().any(|v| matches!(v, Val::Str(_))) {
                return strings::list_literal(vals);
            }
            // an Int among numbers is the number it is, exactly (an Int
            // array's literal is decided by the lowering, above)
            let vals: Vec<Val> = vals.into_iter().map(|v| match v {
                Val::Int(i) => ints::to_num(i).map(Val::Num),
                v => Ok(v),
            }).collect::<Result<_, _>>()?;
            if vals.iter().all(|v| matches!(v, Val::Num(_))) {
                let data = vals.iter().map(|v| v.as_num("an array element").unwrap()).collect();
                Ok(new_arr(vec![vals.len()], data))
            } else if vals.iter().all(|v| matches!(v, Val::Num(_) | Val::Cplx(..))) {
                // Numbers with at least one complex among them: the PACKED
                // complex array, reals promoting. Packed whenever it can be --
                // only a genuinely mixed literal pays for boxing.
                let data = vals
                    .iter()
                    .flat_map(|v| match v {
                        Val::Cplx(re, im) => [*re, *im],
                        Val::Num(n) => [*n, 0.0],
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(new_carr(vec![vals.len()], data))
            } else {
                // Functions: a REGISTRY. Every member must take the same number
                // of arguments, because the compiled engine packs them into one
                // C array with one element type and reads each argument slot by
                // the shared signature. The interpreter could hold any mixture,
                // but a program that ran here and failed to compile would be a
                // worse bargain than this refusal.
                let arity = |v: &Val| match v {
                    Val::Func(_, f) => Some(f.params.len()),
                    Val::Closure(c) => Some(c.params.len()),
                    _ => None,
                };
                let first = arity(&vals[0]);
                if first.is_none() || vals.iter().any(|v| arity(v) != first) {
                    return Err(
                        "an array holds numbers, or functions that all take the same number of arguments -- not a mixture"
                            .into(),
                    );
                }
                Ok(Val::VArr(Rc::new(RefCell::new(vals))))
            }
        }
    }

    /// Text an argument gives: a literal, or a string value.
    fn text_arg(&self, a: &Expr, scope: &mut HashMap<String, Val>, what: &str) -> Result<String, String> {
        match a {
            Expr::Str { value, .. } => Ok(value.clone()),
            e => match self.eval(e, scope)? {
                Val::Str(s) => Ok(s.text.clone()),
                _ => Err(what.to_string()),
            },
        }
    }

    /// The one index of a string or a list of strings.
    fn one_index(&self, indices: &[Expr], scope: &mut HashMap<String, Val>, what: &str) -> Result<f64, String> {
        if indices.len() != 1 {
            return Err(format!("{what} takes one index; this has {}", indices.len()));
        }
        let x = self.eval(&indices[0], scope)?.as_num("an index")?;
        index_int(x)?;
        Ok(x)
    }

    fn flat_index(
        &self,
        shape: &[usize],
        indices: &[Expr],
        scope: &mut HashMap<String, Val>,
    ) -> Result<usize, String> {
        if indices.len() != shape.len() {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!(
                "array has rank {} but was indexed with {} index/indices",
                shape.len(),
                indices.len()
            )));
        }
        let mut flat = 0usize;
        for (k, ix) in indices.iter().enumerate() {
            let i = index_int(self.eval(ix, scope)?.as_num("an index")?)?;
            flat = flat * shape[k] + self.check_index(shape, k, i)?;
        }
        Ok(flat)
    }

    /// `flat_index` for indices already computed (`n` of them were written;
    /// `vals` holds them).
    fn flat_index_of(&self, shape: &[usize], n: usize, vals: &[f64]) -> Result<usize, String> {
        if n != shape.len() {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("array has rank {} but was indexed with {} index/indices", shape.len(), n)));
        }
        let mut flat = 0usize;
        for (k, v) in vals.iter().enumerate() {
            flat = flat * shape[k] + self.check_index(shape, k, index_int(*v)?)?;
        }
        Ok(flat)
    }

    /// Index `i` in dimension `k`, or the out-of-bounds error. A negative
    /// index counts from the end, as in Python: -1 is the last.
    fn check_index(&self, shape: &[usize], k: usize, i: i64) -> Result<usize, String> {
        let at = if i < 0 { i.saturating_add(shape[k] as i64) } else { i };
        if at < 0 || at as usize >= shape[k] {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("index {i} out of bounds for dimension {k} (size {})", shape[k])));
        }
        Ok(at as usize)
    }

    /// One `print` line: the parts in order, then a newline. The line is
    /// built first, so a value print cannot write stops the line before any
    /// of it is out, in every engine.
    fn print_parts(&self, parts: &[Part]) -> Result<Val, String> {
        use crate::printfmt::{self as pf, Kind};
        let mut line = String::new();
        for p in parts {
            match p {
                Part::Text(t) => line.push_str(t),
                Part::Map(kf, vf, v) => match v {
                    Val::Map(m) => line.push_str(&m.borrow().text(*kf, *vf)),
                    _ => return Err(maps::TWO_FORMATS.into()),
                },
                Part::Val(f, v) => match (v, f.kind) {
                    // an Int: its digits, or the number it is in a float format
                    (Val::Int(i), Kind::Bare | Kind::Repr | Kind::D | Kind::Str) => line.push_str(&crate::printfmt::int_text(*i, *f)),
                    (Val::Int(i), _) => line.push_str(&pf::number(ints::to_num(*i)?, *f)),
                    // a Float32: bare, its shortest float32 digits, or the number it is
                    (Val::F32(x), Kind::Bare) => line.push_str(&crate::printfmt::bare(*x as f64)),
                    (Val::F32(x), Kind::Repr | Kind::Str) => line.push_str(&f32s::repr(*x)),
                    (Val::F32(x), _) => line.push_str(&pf::number(*x as f64, *f)),
                    (Val::PArr(a), Kind::Bare) => line.push_str(&f32s::text(a.borrow().k, &a.borrow().b)),
                    (Val::IArr(arr), Kind::Bare) => line.push_str(&iarr_text(&arr.borrow())),
                    (Val::Enum(e), Kind::Bare) => enum_text(e, &mut line),
                    (Val::Str(st), Kind::Bare) => line.push_str(&st.text),
                    (Val::Strs(_), Kind::Bare) => line.push_str(&strings::to_str(v)?),
                    (Val::Str(_) | Val::Strs(_), _) => line.push_str(&strings::fmt_value(v, &[])?),
                    (Val::Num(x), _) => line.push_str(&pf::number(*x, *f)),
                    (Val::Arr(a), _) => {
                        let a = a.borrow();
                        line.push_str(&pf::array(&a.shape, &a.data, *f));
                    }
                    (Val::Bool(b), Kind::Bare) => line.push_str(if *b { "true" } else { "false" }),
                    (Val::Cplx(re, im), Kind::Bare) => line.push_str(&format_cplx(*re, *im)),
                    (Val::Bool(_) | Val::Cplx(..), _) => return Err("`fmt` takes a real number or a real array".into()),
                    (Val::CArr(_), _) => return Err("`print` cannot write a complex array; `show` it, or print `re` and `im` of it".into()),
                    (Val::Record(r), _) => return Err(format!("`print` cannot write a {}; print its fields", r.name)),
                    (Val::Recs(_), _) => return Err("`print` cannot write an array of records; print its records' fields".into()),
                    (Val::Map(m), _) => line.push_str(&m.borrow().text(*f, *f)),
                    _ => return Err("`print` writes text, numbers, booleans, complex numbers and real arrays, not a function".into()),
                },
            }
        }
        if crate::digits::on() {
            // the digits lens (crate::digits): the numbers at full precision
            // each named by the text before it: `energy: `, `x = `
            let mut name = String::new();
            for p in parts {
                match p {
                    Part::Text(t) => name = t.trim().trim_end_matches([':', '=']).trim().rsplit("  ").next().unwrap_or("").to_string(),
                    Part::Val(_, Val::Num(x)) => crate::digits::capture_as(*x, name.clone()),
                    Part::Val(_, Val::Cplx(re, im)) => {
                        crate::digits::capture_as(*re, format!("re {name}"));
                        crate::digits::capture_as(*im, format!("im {name}"));
                    }
                    Part::Val(_, Val::Arr(a)) => {
                        a.borrow().data.iter().enumerate().for_each(|(i, x)| crate::digits::capture_as(*x, format!("{name}[{i}]")));
                    }
                    _ => {}
                }
            }
            crate::digits::printed(&line);
        }
        line.push('\n');
        self.out.borrow_mut().push_str(&line);
        Ok(Val::Num(0.0))
    }

    fn show(&self, label: &str, v: &Val) -> Result<(), String> {
        if crate::digits::on() {
            // the digits lens (crate::digits): the numbers at full precision
            match v {
                Val::Num(x) => crate::digits::capture_as(*x, label.to_string()),
                Val::Cplx(re, im) => {
                    crate::digits::capture_as(*re, format!("re {label}"));
                    crate::digits::capture_as(*im, format!("im {label}"));
                }
                Val::Arr(a) | Val::CArr(a) => {
                    a.borrow().data.iter().enumerate().for_each(|(i, x)| crate::digits::capture_as(*x, format!("{label}[{i}]")));
                }
                _ => {}
            }
            crate::digits::printed(label);
        }
        if let Val::Record(r) = v {
            return Err(format!("cannot `show` a {}", r.name));
        }
        if let Val::Recs(_) = v {
            return Err("cannot `show` a array of records".into());
        }
        let mut out = self.out.borrow_mut();
        // a string, or a list of strings, as Python's repr: `s: 'text'`
        if let Val::Str(_) | Val::Strs(_) = v {
            if !label.is_empty() {
                out.push_str(label);
                out.push_str(": ");
            }
            out.push_str(&strings::repr_value(v)?);
            out.push('\n');
            return Ok(());
        }
        if !label.is_empty() {
            out.push_str(label);
            out.push_str(": ");
        }
        match v {
            Val::Num(n) => {
                out.push_str(&format_g(*n));
                out.push('\n');
            }
            Val::F32(x) => {
                out.push_str(&crate::printfmt::bare(*x as f64));
                out.push('\n');
            }
            Val::PArr(a) => {
                out.push_str(&f32s::text(a.borrow().k, &a.borrow().b));
                out.push('\n');
            }
            Val::Int(i) => {
                out.push_str(&ints::text(*i, 0));
                out.push('\n');
            }
            Val::Enum(e) => {
                enum_text(e, &mut out);
                out.push('\n');
            }
            Val::IArr(a) => {
                out.push_str(&iarr_text(&a.borrow()));
                out.push('\n');
            }
            Val::Bool(b) => {
                out.push_str(if *b { "true" } else { "false" });
                out.push('\n');
            }
            Val::Arr(a) => {
                let ab = a.borrow();
                // "label: [" puts the inner bracket at len(label) + 3; with no
                // label the outer bracket is at column 0, so the inner is at 1.
                let indent = if label.is_empty() { 1 } else { label.len() + 3 };
                out.push_str(&format_arr_at(&ab.shape, &ab.data, indent));
                out.push('\n');
            }
            Val::Cplx(re, im) => {
                out.push_str(&format_cplx(*re, *im));
                out.push('\n');
            }
            Val::Map(m) => {
                let bare = crate::printfmt::Fmt::BARE;
                out.push_str(&m.borrow().text(bare, bare));
                out.push('\n');
            }
            Val::CArr(a) => {
                let ab = a.borrow();
                let indent = if label.is_empty() { 1 } else { label.len() + 3 };
                out.push_str(&format_carr_at(&ab.shape, &ab.data, indent));
                out.push('\n');
            }
            Val::Closure(_) => {
                out.push_str("<function>\n");
            }
            Val::Func(name, _) => {
                out.push_str("<function ");
                out.push_str(name);
                out.push_str(">\n");
            }
            Val::VArr(vs) => {
                out.push('[');
                for (i, el) in vs.borrow().iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    match el {
                        Val::Num(n) => out.push_str(&format_g(*n)),
                        Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                        Val::Closure(_) => out.push_str("<function>"),
                        Val::Func(nm, _) => out.push_str(&format!("<function {nm}>")),
                        Val::VArr(_) => out.push_str("<array>"),
                        Val::Arr(_) | Val::CArr(_) => out.push_str("<array>"),
                        Val::Cplx(re, im) => out.push_str(&format_cplx(*re, *im)),
                        Val::Str(st) => out.push_str(&strings::repr(&st.text)),
                        Val::Strs(_) => out.push_str("<list of strings>"),
                        Val::Record(r) => out.push_str(&r.name),
                        Val::Map(_) => out.push_str("<map>"),
                        Val::Int(i) => out.push_str(&ints::text(*i, 0)),
                        Val::IArr(_) => out.push_str("<array>"),
                        Val::Enum(e) => enum_text(e, &mut out),
                        Val::Recs(_) => out.push_str("<array of records>"),
                        Val::F32(x) => out.push_str(&crate::printfmt::bare(*x as f64)),
                        Val::PArr(_) => out.push_str("<array>"),
                    }
                }
                out.push_str("]\n");
            }
            Val::Record(_) | Val::Recs(_) | Val::Str(_) | Val::Strs(_) => unreachable!("handled above"),
        }
        Ok(())
    }
}

// ---- the complex Hermitian eigenproblem ------------------------------------
//
// Householder reduction to a real symmetric tridiagonal, implicit-shift QL on
// that, and a back-transformation: the EISPACK route (htridi / tql2 / htribk),
// derived here rather than transcribed. It is ~4x cheaper than cyclic Jacobi
// and keeps its accuracy at sizes where Jacobi's sweep count starts to climb,
// which matters because the matrices it exists for -- a plane-wave
// Hamiltonian at one k-point -- are hundreds to thousands on a side.
//
//   A = Q T Q^H        Householder, T complex Hermitian tridiagonal
//   T = D S D^H        a diagonal phase matrix D makes the subdiagonal real
//   S = Z diag(w) Z^T  implicit QL, real arithmetic
//   V = Q D Z          eigenvectors of A, in columns
//
// `a` is row-major interleaved (re, im), n x n, Hermitian; only its lower
// triangle is read. Returns ascending eigenvalues and the eigenvectors in the
// COLUMNS of an interleaved n x n matrix, as eig_sym does for the real case.
fn herm_eig(a_in: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let ix = |i: usize, j: usize| 2 * (i * n + j);
    // Said plainly, because the alternative is the QL sweep spinning to its
    // iteration cap and blaming itself for someone else's NaN.
    if a_in.iter().any(|x| !x.is_finite()) {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "`eig_herm`: the matrix contains a NaN or an infinity"));
    }
    let mut a = a_in.to_vec();
    // symmetrize from the lower triangle so a caller's rounding noise in the
    // upper one cannot make the matrix non-Hermitian
    for i in 0..n {
        a[ix(i, i) + 1] = 0.0;
        for j in 0..i {
            a[ix(j, i)] = a[ix(i, j)];
            a[ix(j, i) + 1] = -a[ix(i, j) + 1];
        }
    }
    // q accumulates the reflectors; starts as the identity
    let mut q = vec![0.0f64; 2 * n * n];
    for i in 0..n {
        q[ix(i, i)] = 1.0;
    }
    let mut v = vec![0.0f64; 2 * n];
    let mut pv = vec![0.0f64; 2 * n];
    if n > 2 {
        for k in 0..n - 2 {
            // x = A[k+1.., k]; reflect it onto alpha * e1
            let mut norm2 = 0.0;
            for i in k + 1..n {
                norm2 += a[ix(i, k)] * a[ix(i, k)] + a[ix(i, k) + 1] * a[ix(i, k) + 1];
            }
            let norm = norm2.sqrt();
            // nothing below the subdiagonal already: skip
            let mut tail = 0.0;
            for i in k + 2..n {
                tail += a[ix(i, k)] * a[ix(i, k)] + a[ix(i, k) + 1] * a[ix(i, k) + 1];
            }
            if tail == 0.0 {
                continue;
            }
            let (x0r, x0i) = (a[ix(k + 1, k)], a[ix(k + 1, k) + 1]);
            let x0 = x0r.vhypot(x0i);
            // alpha = -e^{i arg x0} |x|: the sign that avoids cancellation
            let (pr, pi) = if x0 == 0.0 { (1.0, 0.0) } else { (x0r / x0, x0i / x0) };
            let (alr, ali) = (-pr * norm, -pi * norm);
            for t in v.iter_mut() {
                *t = 0.0;
            }
            for i in k + 1..n {
                v[2 * i] = a[ix(i, k)];
                v[2 * i + 1] = a[ix(i, k) + 1];
            }
            v[2 * (k + 1)] -= alr;
            v[2 * (k + 1) + 1] -= ali;
            let mut vn = 0.0;
            for i in k + 1..n {
                vn += v[2 * i] * v[2 * i] + v[2 * i + 1] * v[2 * i + 1];
            }
            let vn = vn.sqrt();
            if vn == 0.0 {
                continue;
            }
            for i in k + 1..n {
                v[2 * i] /= vn;
                v[2 * i + 1] /= vn;
            }
            // A <- H A H with H = I - 2 v v^H, as the Hermitian rank-2 update:
            //   p = A v,  K = v^H p (real),  w = p - K v,  A -= 2 (v w^H + w v^H)
            // Only rows/columns k.. can change: everything before k is already
            // tridiagonal and v is zero there, so p vanishes on it identically.
            for t in pv.iter_mut() {
                *t = 0.0;
            }
            for i in k..n {
                let (mut sr, mut si) = (0.0, 0.0);
                for j in k + 1..n {
                    let (ar, ai) = (a[ix(i, j)], a[ix(i, j) + 1]);
                    sr += ar * v[2 * j] - ai * v[2 * j + 1];
                    si += ar * v[2 * j + 1] + ai * v[2 * j];
                }
                pv[2 * i] = sr;
                pv[2 * i + 1] = si;
            }
            let mut kk = 0.0;
            for i in k + 1..n {
                kk += v[2 * i] * pv[2 * i] + v[2 * i + 1] * pv[2 * i + 1];
            }
            for i in k..n {
                pv[2 * i] -= kk * v[2 * i];
                pv[2 * i + 1] -= kk * v[2 * i + 1];
            }
            for i in k..n {
                for j in k..n {
                    // v_i conj(w_j) + w_i conj(v_j)
                    let (vir, vii, wir, wii) = (v[2 * i], v[2 * i + 1], pv[2 * i], pv[2 * i + 1]);
                    let (vjr, vji, wjr, wji) = (v[2 * j], v[2 * j + 1], pv[2 * j], pv[2 * j + 1]);
                    a[ix(i, j)] -= 2.0 * (vir * wjr + vii * wji + wir * vjr + wii * vji);
                    a[ix(i, j) + 1] -= 2.0 * (vii * wjr - vir * wji + wii * vjr - wir * vji);
                }
            }
            // Q <- Q H :  Q -= 2 (Q v) v^H
            for i in 0..n {
                let (mut sr, mut si) = (0.0, 0.0);
                for j in k + 1..n {
                    let (qr, qi) = (q[ix(i, j)], q[ix(i, j) + 1]);
                    sr += qr * v[2 * j] - qi * v[2 * j + 1];
                    si += qr * v[2 * j + 1] + qi * v[2 * j];
                }
                for j in k + 1..n {
                    q[ix(i, j)] -= 2.0 * (sr * v[2 * j] + si * v[2 * j + 1]);
                    q[ix(i, j) + 1] -= 2.0 * (si * v[2 * j] - sr * v[2 * j + 1]);
                }
            }
        }
    }
    // phases: d_0 = 1, d_{k+1} = d_k e_k / |e_k| makes every subdiagonal real
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    let mut ph = vec![0.0f64; 2 * n];
    if n > 0 {
        ph[0] = 1.0;
    }
    for i in 0..n {
        d[i] = a[ix(i, i)];
    }
    for k in 0..n.saturating_sub(1) {
        let (er, ei) = (a[ix(k + 1, k)], a[ix(k + 1, k) + 1]);
        let m = er.vhypot(ei);
        e[k] = m;
        let (ur, ui) = if m == 0.0 { (1.0, 0.0) } else { (er / m, ei / m) };
        ph[2 * (k + 1)] = ph[2 * k] * ur - ph[2 * k + 1] * ui;
        ph[2 * (k + 1) + 1] = ph[2 * k] * ui + ph[2 * k + 1] * ur;
    }
    // implicit-shift QL on the real tridiagonal (d, e), accumulating Z
    let mut z = vec![0.0f64; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }
    if !tql2(&mut d, &mut e, Some(&mut z), 80) {
        return Err("`eig_herm`: QL iteration did not converge".into());
    }
    // ascending order
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| d[x].partial_cmp(&d[y]).unwrap_or(std::cmp::Ordering::Equal));
    let vals: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    // V = Q D Z, columns reordered to match
    let mut vecs = vec![0.0f64; 2 * n * n];
    for i in 0..n {
        for (c, &col) in order.iter().enumerate() {
            let (mut sr, mut si) = (0.0, 0.0);
            for j in 0..n {
                // (Q D)_{ij} = q_ij * ph_j
                let (qr, qi) = (q[ix(i, j)], q[ix(i, j) + 1]);
                let (dr, di) = (ph[2 * j], ph[2 * j + 1]);
                let zj = z[j * n + col];
                sr += (qr * dr - qi * di) * zj;
                si += (qr * di + qi * dr) * zj;
            }
            vecs[ix(i, c)] = sr;
            vecs[ix(i, c) + 1] = si;
        }
    }
    Ok((vals, vecs))
}

// ---- the discrete Fourier transform --------------------------------------
//
// Data is interleaved (re, im), the layout C99 gives `double complex`, so this
// and the C runtime's copy walk identical memory and do identical arithmetic.
//
//   forward:  X[k] = sum_n x[n] exp(-2 pi i k n / N)        (unscaled)
//   inverse:  x[n] = (1/N) sum_k X[k] exp(+2 pi i k n / N)
//
// Powers of two go through iterative radix-2 Cooley-Tukey. Every other length
// goes through Bluestein's chirp-z identity, which turns a length-N transform
// into a convolution done with power-of-two transforms -- so no length is a
// special case the caller has to route around, and N log N holds for all N
// (a prime-length grid would otherwise fall off an O(N^2) cliff).

/// Everything about one transform length that does not depend on the data:
/// the twiddle table, how the length factors, and for Bluestein the chirp and
/// its transform. Built once per length and kept (see `fft_plan`) -- a
/// plane-wave code calls the same three lengths millions of times, and
/// recomputing cos/sin on each call was most of its run time.
struct FftPlan {
    /// forward twiddles exp(-2 pi i k / n) for k in 0..n, interleaved. Each
    /// one is cos/sin of the exact angle, never a recurrence: a recurrence
    /// drifts, and would drift DIFFERENTLY from the C runtime's.
    tw: Vec<f64>,
    kind: FftKind,
}

enum FftKind {
    /// n is a power of two: iterative radix-2.
    Pow2,
    /// n = product of these primes, all <= 7: recursive mixed radix.
    Mixed(Vec<usize>),
    /// anything else: Bluestein's chirp-z at power-of-two length `m`.
    /// `chirp[s]` and `bfft[s]` are for the forward (0) and inverse (1) sign.
    Bluestein { m: usize, chirp: [Vec<f64>; 2], bfft: [Vec<f64>; 2] },
}

thread_local! {
    static FFT_PLANS: RefCell<HashMap<usize, Rc<FftPlan>>> = RefCell::new(HashMap::new());
}

fn fft_plan(n: usize) -> Rc<FftPlan> {
    if let Some(p) = FFT_PLANS.with(|c| c.borrow().get(&n).cloned()) {
        return p;
    }
    let step = -2.0 * std::f64::consts::PI / n as f64;
    let mut tw = Vec::with_capacity(2 * n);
    for k in 0..n {
        tw.push((step * k as f64).vcos());
        tw.push((step * k as f64).vsin());
    }
    let kind = if n.is_power_of_two() {
        FftKind::Pow2
    } else {
        // trial-divide by 2, 3, 5, 7; whatever is left decides the method
        let (mut rest, mut factors) = (n, Vec::new());
        for p in [2usize, 3, 5, 7] {
            while rest % p == 0 {
                factors.push(p);
                rest /= p;
            }
        }
        if rest == 1 {
            FftKind::Mixed(factors)
        } else {
            // Bluestein: k*n = (k^2 + n^2 - (k-n)^2) / 2, so the transform is a
            // chirp multiply, a convolution with the conjugate chirp, and
            // another chirp multiply. The convolution runs at length m.
            let m = (2 * n - 1).next_power_of_two();
            let inner = fft_plan(m);
            let build = |sign: f64| -> (Vec<f64>, Vec<f64>) {
                let (mut chirp, mut b) = (vec![0.0f64; 2 * n], vec![0.0f64; 2 * m]);
                for j in 0..n {
                    // chirp w[j] = exp(sign * i * pi * j^2 / n). j^2 is reduced
                    // mod 2n first: the chirp has that period, and it keeps the
                    // angle small enough that cos/sin stay accurate at large j.
                    let r = ((j as u128 * j as u128) % (2 * n as u128)) as f64;
                    let ang = sign * std::f64::consts::PI * r / n as f64;
                    let (wr, wi) = (ang.vcos(), ang.vsin());
                    chirp[2 * j] = wr;
                    chirp[2 * j + 1] = wi;
                    // b holds the CONJUGATE chirp, wrapped so the convolution is cyclic
                    b[2 * j] = wr;
                    b[2 * j + 1] = -wi;
                    if j > 0 {
                        b[2 * (m - j)] = wr;
                        b[2 * (m - j) + 1] = -wi;
                    }
                }
                fft_pow2(&mut b, m, &inner.tw, false);
                (chirp, b)
            };
            let ((cf, bf), (ci, bi)) = (build(-1.0), build(1.0));
            FftKind::Bluestein { m, chirp: [cf, ci], bfft: [bf, bi] }
        }
    };
    let plan = Rc::new(FftPlan { tw, kind });
    FFT_PLANS.with(|c| c.borrow_mut().insert(n, plan.clone()));
    plan
}

/// In-place iterative radix-2 Cooley-Tukey over interleaved (re, im) data.
/// `tw` is the length-`n` plan's table; `conj` selects the inverse kernel.
/// `n` must be a power of two.
fn fft_pow2(d: &mut [f64], n: usize, tw: &[f64], conj: bool) {
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            d.swap(2 * i, 2 * j);
            d.swap(2 * i + 1, 2 * j + 1);
        }
    }
    // butterflies. Stage `len` wants exp(-2 pi i k / len) = tw[k * (n / len)].
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let skip = n / len;
        for k in 0..half {
            let (wr, wi) = (tw[2 * k * skip], if conj { -tw[2 * k * skip + 1] } else { tw[2 * k * skip + 1] });
            let mut i = k;
            while i < n {
                let (a, b) = (2 * i, 2 * (i + half));
                let (vr, vi) = (d[b] * wr - d[b + 1] * wi, d[b] * wi + d[b + 1] * wr);
                let (ur, ui) = (d[a], d[a + 1]);
                d[a] = ur + vr;
                d[a + 1] = ui + vi;
                d[b] = ur - vr;
                d[b + 1] = ui - vi;
                i += len;
            }
        }
        len <<= 1;
    }
}

/// Recursive mixed-radix transform (decimation in time). `out` receives the
/// transform of the `n_sub` input samples `inp[off], inp[off + stride], ...`;
/// `stride * n_sub` is always the full length `n`, which is what lets every
/// level read its twiddles out of the one length-`n` table.
#[allow(clippy::too_many_arguments)]
fn fft_mixed(out: &mut [f64], inp: &[f64], off: usize, stride: usize, factors: &[usize], n_sub: usize, tw: &[f64], n: usize, conj: bool) {
    let p = factors[0];
    let m = n_sub / p;
    if m == 1 {
        for q in 0..p {
            out[2 * q] = inp[2 * (off + q * stride)];
            out[2 * q + 1] = inp[2 * (off + q * stride) + 1];
        }
    } else {
        for q in 0..p {
            fft_mixed(&mut out[2 * q * m..2 * (q + 1) * m], inp, off + q * stride, stride * p, &factors[1..], m, tw, n, conj);
        }
    }
    // combine the p sub-transforms of length m: X[k] = sum_q S_q[k mod m] w^(k q)
    if p == 2 {
        for u in 0..m {
            let t = stride * u;
            let (wr, wi) = (tw[2 * t], if conj { -tw[2 * t + 1] } else { tw[2 * t + 1] });
            let (a, b) = (2 * u, 2 * (u + m));
            let (vr, vi) = (out[b] * wr - out[b + 1] * wi, out[b] * wi + out[b + 1] * wr);
            let (ur, ui) = (out[a], out[a + 1]);
            out[a] = ur + vr;
            out[a + 1] = ui + vi;
            out[b] = ur - vr;
            out[b + 1] = ui - vi;
        }
    } else if p == 3 {
        // radix 3 in two multiplies per triple: with w = exp(-2 pi i / 3) =
        // -1/2 + i e, the outputs are F0 + s3, and F0 - s3/2 +- i e s0, where
        // s3 = s1 + s2 and s0 = s1 - s2 are the twiddled inputs' sum and difference
        let e = if conj { -tw[2 * stride * m + 1] } else { tw[2 * stride * m + 1] };
        for u in 0..m {
            let (t1, t2) = (stride * u, 2 * stride * u);
            let (w1r, w1i) = (tw[2 * t1], if conj { -tw[2 * t1 + 1] } else { tw[2 * t1 + 1] });
            let (w2r, w2i) = (tw[2 * t2], if conj { -tw[2 * t2 + 1] } else { tw[2 * t2 + 1] });
            let (i0, i1, i2) = (2 * u, 2 * (u + m), 2 * (u + 2 * m));
            let (s1r, s1i) = (out[i1] * w1r - out[i1 + 1] * w1i, out[i1] * w1i + out[i1 + 1] * w1r);
            let (s2r, s2i) = (out[i2] * w2r - out[i2 + 1] * w2i, out[i2] * w2i + out[i2 + 1] * w2r);
            let (s3r, s3i) = (s1r + s2r, s1i + s2i);
            let (s0r, s0i) = ((s1r - s2r) * e, (s1i - s2i) * e);
            let (hr, hi) = (out[i0] - 0.5 * s3r, out[i0 + 1] - 0.5 * s3i);
            out[i0] += s3r;
            out[i0 + 1] += s3i;
            out[i2] = hr + s0i;
            out[i2 + 1] = hi - s0r;
            out[i1] = hr - s0i;
            out[i1 + 1] = hi + s0r;
        }
    } else if p == 5 {
        // radix 5 in four multiplies per quintuple, pairing q with 5 - q
        let (yar, ybr) = (tw[2 * stride * m], tw[2 * 2 * stride * m]);
        let (yai, ybi) = if conj {
            (-tw[2 * stride * m + 1], -tw[2 * 2 * stride * m + 1])
        } else {
            (tw[2 * stride * m + 1], tw[2 * 2 * stride * m + 1])
        };
        for u in 0..m {
            let mut x = [0.0f64; 10];
            for q in 1..5 {
                let t = q * stride * u;
                let (wr, wi) = (tw[2 * t], if conj { -tw[2 * t + 1] } else { tw[2 * t + 1] });
                let i = 2 * (u + q * m);
                x[2 * q] = out[i] * wr - out[i + 1] * wi;
                x[2 * q + 1] = out[i] * wi + out[i + 1] * wr;
            }
            let (i0, i1, i2, i3, i4) = (2 * u, 2 * (u + m), 2 * (u + 2 * m), 2 * (u + 3 * m), 2 * (u + 4 * m));
            let (f0r, f0i) = (out[i0], out[i0 + 1]);
            let (s7r, s7i, s10r, s10i) = (x[2] + x[8], x[3] + x[9], x[2] - x[8], x[3] - x[9]);
            let (s8r, s8i, s9r, s9i) = (x[4] + x[6], x[5] + x[7], x[4] - x[6], x[5] - x[7]);
            out[i0] = f0r + (s7r + s8r);
            out[i0 + 1] = f0i + (s7i + s8i);
            let (s5r, s5i) = (f0r + s7r * yar + s8r * ybr, f0i + s7i * yar + s8i * ybr);
            let (s6r, s6i) = (s10i * yai + s9i * ybi, -(s10r * yai) - s9r * ybi);
            out[i1] = s5r - s6r;
            out[i1 + 1] = s5i - s6i;
            out[i4] = s5r + s6r;
            out[i4 + 1] = s5i + s6i;
            let (s11r, s11i) = (f0r + s7r * ybr + s8r * yar, f0i + s7i * ybr + s8i * yar);
            let (s12r, s12i) = (-(s10i * ybi) + s9i * yai, s10r * ybi - s9r * yai);
            out[i2] = s11r + s12r;
            out[i2 + 1] = s11i + s12i;
            out[i3] = s11r - s12r;
            out[i3 + 1] = s11i - s12i;
        }
    } else {
        let mut sc = [0.0f64; 14];
        for u in 0..m {
            for q1 in 0..p {
                sc[2 * q1] = out[2 * (u + q1 * m)];
                sc[2 * q1 + 1] = out[2 * (u + q1 * m) + 1];
            }
            for q1 in 0..p {
                let k = u + q1 * m;
                let mut t = 0usize;
                let (mut sr, mut si) = (sc[0], sc[1]);
                for q in 1..p {
                    t += stride * k;
                    if t >= n {
                        t -= n;
                    }
                    let (wr, wi) = (tw[2 * t], if conj { -tw[2 * t + 1] } else { tw[2 * t + 1] });
                    sr += sc[2 * q] * wr - sc[2 * q + 1] * wi;
                    si += sc[2 * q] * wi + sc[2 * q + 1] * wr;
                }
                out[2 * k] = sr;
                out[2 * k + 1] = si;
            }
        }
    }
}

/// Transform interleaved complex `data` in place; any length.
fn fft_in_place(data: &mut [f64], inverse: bool) {
    let n = data.len() / 2;
    if n < 2 {
        return;
    }
    let plan = fft_plan(n);
    match &plan.kind {
        FftKind::Pow2 => fft_pow2(data, n, &plan.tw, inverse),
        FftKind::Mixed(factors) => {
            let inp = data.to_vec();
            fft_mixed(data, &inp, 0, 1, factors, n, &plan.tw, n, inverse);
        }
        FftKind::Bluestein { m, chirp, bfft } => {
            let m = *m;
            let (chirp, bfft) = (&chirp[inverse as usize], &bfft[inverse as usize]);
            let inner = fft_plan(m);
            let mut a = vec![0.0f64; 2 * m];
            for j in 0..n {
                let (wr, wi) = (chirp[2 * j], chirp[2 * j + 1]);
                let (xr, xi) = (data[2 * j], data[2 * j + 1]);
                a[2 * j] = xr * wr - xi * wi;
                a[2 * j + 1] = xr * wi + xi * wr;
            }
            fft_pow2(&mut a, m, &inner.tw, false);
            for j in 0..m {
                let (ar, ai, br, bi) = (a[2 * j], a[2 * j + 1], bfft[2 * j], bfft[2 * j + 1]);
                a[2 * j] = ar * br - ai * bi;
                a[2 * j + 1] = ar * bi + ai * br;
            }
            fft_pow2(&mut a, m, &inner.tw, true);
            let scale = 1.0 / m as f64;
            for j in 0..n {
                let (wr, wi) = (chirp[2 * j], chirp[2 * j + 1]);
                let (cr, ci) = (a[2 * j] * scale, a[2 * j + 1] * scale);
                data[2 * j] = cr * wr - ci * wi;
                data[2 * j + 1] = cr * wi + ci * wr;
            }
        }
    }
    if inverse {
        let s = 1.0 / n as f64;
        for v in data.iter_mut() {
            *v *= s;
        }
    }
}

/// Transform a rank-3 interleaved complex array along all three axes, in
/// place. The 3-D transform is separable, so it is three passes of the 1-D one
/// over gathered strides -- which inherits Bluestein for free, so a 3-D grid
/// need not have power-of-two sides (plane-wave grids usually do not).
fn fft3_in_place(data: &mut [f64], shape: &[usize], inverse: bool) {
    let strides = [shape[1] * shape[2], shape[2], 1];
    for axis in 0..3 {
        let len = shape[axis];
        if len < 2 {
            continue;
        }
        let st = strides[axis];
        // the two axes that are NOT being transformed index the lines
        let (o1, o2) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let mut line = vec![0.0f64; 2 * len];
        for a in 0..shape[o1] {
            for b in 0..shape[o2] {
                let base = a * strides[o1] + b * strides[o2];
                let mut any = false;
                for t in 0..len {
                    line[2 * t] = data[2 * (base + t * st)];
                    line[2 * t + 1] = data[2 * (base + t * st) + 1];
                    any |= line[2 * t] != 0.0 || line[2 * t + 1] != 0.0;
                }
                // the transform of zeros is zeros, and a plane-wave sphere
                // leaves most of its box empty: skip those lines
                if !any {
                    continue;
                }
                fft_in_place(&mut line, inverse);
                for t in 0..len {
                    data[2 * (base + t * st)] = line[2 * t];
                    data[2 * (base + t * st) + 1] = line[2 * t + 1];
                }
            }
        }
    }
}

/// LU factorization with partial pivoting; solves A x = b. Returns x, or an
/// error naming the (near-)singular pivot.
fn lu_solve_vec(a_in: &[f64], b_in: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    for k in 0..n {
        let mut piv = k;
        let mut big = a[k * n + k].abs();
        for i in (k + 1)..n {
            if a[i * n + k].abs() > big {
                big = a[i * n + k].abs();
                piv = i;
            }
        }
        if big < 1e-14 * scale {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`lin_solve`: matrix is singular to working precision (pivot {k})")));
        }
        if piv != k {
            for j in 0..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        for i in (k + 1)..n {
            let f = a[i * n + k] / a[k * n + k];
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    Ok(x)
}

/// LU factorization with partial pivoting, factor-once form: returns the
/// packed LU (L below the diagonal, unit-diagonal implied; U on and above)
/// plus the pivot row for each elimination step.
fn lu_factor(a: &mut [f64], n: usize) -> Result<Vec<usize>, String> {
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    let mut piv = vec![0usize; n];
    for k in 0..n {
        let mut p = k;
        let mut big = a[k * n + k].abs();
        for i in (k + 1)..n {
            if a[i * n + k].abs() > big {
                big = a[i * n + k].abs();
                p = i;
            }
        }
        if big < 1e-14 * scale {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("matrix is singular to working precision (pivot {k})")));
        }
        piv[k] = p;
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
        }
        for i in (k + 1)..n {
            let f = a[i * n + k] / a[k * n + k];
            a[i * n + k] = f;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
        }
    }
    Ok(piv)
}

/// Solve with a packed LU factorization (one right-hand side, in place).
/// The stored multipliers live in their post-swap rows, so ALL pivots must be
/// applied to b before forward substitution (LAPACK dgetrs order) — an
/// interleaved swap-and-eliminate is wrong the moment a swap actually fires.
fn lu_solve_factored(lu: &[f64], piv: &[usize], n: usize, b: &mut [f64]) {
    for k in 0..n {
        b.swap(k, piv[k]);
    }
    for k in 0..n {
        for i in (k + 1)..n {
            let f = lu[i * n + k];
            b[i] -= f * b[k];
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= lu[i * n + j] * b[j];
        }
        b[i] = s / lu[i * n + i];
    }
}

/// Cholesky factorization A = L L^T (lower triangular). Fails on a
/// non-positive pivot — the SPD contract check comes free with the algorithm.
fn cholesky_lower(a_in: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a_in[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!(
                        "`cholesky`: matrix is not positive definite (pivot {i} = {s:.3e})"
                    )));
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Eigenvalues of a general real matrix: stabilized-elimination reduction to
/// upper Hessenberg (elmhes) followed by Francis double-shift QR (hqr),
/// eigenvalues only. Complex values come out as conjugate pairs. Sorted by
/// (real, imag) for a deterministic order.
fn general_eigvals(a_in: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut a = a_in.to_vec();
    if n == 1 {
        return Ok((vec![a[0]], vec![0.0]));
    }
    // ---- balanc: diagonal similarity scaling so row and column norms match.
    // Badly scaled block matrices (e.g. a 2500x coupling block) stall the QR
    // iteration without this; eigenvalues are unchanged.
    let radix = 2.0f64;
    loop {
        let mut last = true;
        for i in 0..n {
            let mut r = 0.0f64;
            let mut c = 0.0f64;
            for j in 0..n {
                if j != i {
                    r += a[i * n + j].abs();
                    c += a[j * n + i].abs();
                }
            }
            if c != 0.0 && r != 0.0 {
                let mut g = r / radix;
                let mut f = 1.0f64;
                let s = c + r;
                let mut cc = c;
                while cc < g {
                    f *= radix;
                    cc *= radix * radix;
                }
                g = r * radix;
                while cc > g {
                    f /= radix;
                    cc /= radix * radix;
                }
                if (cc + r) / f < 0.95 * s {
                    last = false;
                    let ginv = 1.0 / f;
                    for j in 0..n {
                        a[i * n + j] *= ginv;
                    }
                    for j in 0..n {
                        a[j * n + i] *= f;
                    }
                }
            }
        }
        if last {
            break;
        }
    }
    // ---- elmhes: reduce to upper Hessenberg ----
    for m in 1..(n - 1) {
        let mut x = 0.0f64;
        let mut i_piv = m;
        for j in m..n {
            if a[j * n + m - 1].abs() > x.abs() {
                x = a[j * n + m - 1];
                i_piv = j;
            }
        }
        if i_piv != m {
            for j in (m - 1)..n {
                a.swap(i_piv * n + j, m * n + j);
            }
            for j in 0..n {
                a.swap(j * n + i_piv, j * n + m);
            }
        }
        if x != 0.0 {
            for i in (m + 1)..n {
                let mut y = a[i * n + m - 1];
                if y != 0.0 {
                    y /= x;
                    a[i * n + m - 1] = y;
                    for j in m..n {
                        a[i * n + j] -= y * a[m * n + j];
                    }
                    for j in 0..n {
                        a[j * n + m] += y * a[j * n + i];
                    }
                }
            }
        }
    }
    // ---- hqr: Francis double-shift QR on the Hessenberg matrix ----
    let mut wr = vec![0.0f64; n];
    let mut wi = vec![0.0f64; n];
    let mut anorm = 0.0f64;
    for i in 0..n {
        let lo = i.saturating_sub(1);
        for j in lo..n {
            anorm += a[i * n + j].abs();
        }
    }
    let mut nn: isize = n as isize - 1;
    let mut t = 0.0f64;
    while nn >= 0 {
        let mut its = 0;
        loop {
            let nnu = nn as usize;
            // Look for a single small subdiagonal element.
            let mut l = nnu;
            while l >= 1 {
                let mut s = a[(l - 1) * n + (l - 1)].abs() + a[l * n + l].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if a[l * n + l - 1].abs() + s == s {
                    a[l * n + l - 1] = 0.0;
                    break;
                }
                l -= 1;
            }
            let mut x = a[nnu * n + nnu];
            if l == nnu {
                // One root found.
                wr[nnu] = x + t;
                wi[nnu] = 0.0;
                nn -= 1;
                break;
            }
            let mut y = a[(nnu - 1) * n + (nnu - 1)];
            let mut w = a[nnu * n + nnu - 1] * a[(nnu - 1) * n + nnu];
            if l == nnu - 1 {
                // Two roots found.
                let p = 0.5 * (y - x);
                let q = p * p + w;
                let mut z = q.abs().sqrt();
                x += t;
                if q >= 0.0 {
                    z = p + z.copysign(if p >= 0.0 { 1.0 } else { -1.0 });
                    wr[nnu - 1] = x + z;
                    wr[nnu] = wr[nnu - 1];
                    if z != 0.0 {
                        wr[nnu] = x - w / z;
                    }
                    wi[nnu - 1] = 0.0;
                    wi[nnu] = 0.0;
                } else {
                    wr[nnu - 1] = x + p;
                    wr[nnu] = x + p;
                    wi[nnu - 1] = -z;
                    wi[nnu] = z;
                }
                nn -= 2;
                break;
            }
            // No roots yet: do a QR step.
            if its == 100 {
                return Err("`eig_general`: too many QR iterations".into());
            }
            if its > 0 && its % 10 == 0 {
                // Exceptional shift.
                t += x;
                for i in 0..=nnu {
                    a[i * n + i] -= x;
                }
                let s = a[nnu * n + nnu - 1].abs() + a[(nnu - 1) * n + nnu - 2].abs();
                y = 0.75 * s;
                x = y;
                w = -0.4375 * s * s;
            }
            its += 1;
            // Find two consecutive small subdiagonal elements.
            let mut m = nnu - 2;
            let (mut p, mut q, mut r) = (0.0f64, 0.0f64, 0.0f64);
            loop {
                let z = a[m * n + m];
                let rr = x - z;
                let ss = y - z;
                p = (rr * ss - w) / a[(m + 1) * n + m] + a[m * n + m + 1];
                q = a[(m + 1) * n + m + 1] - z - rr - ss;
                r = a[(m + 2) * n + m + 1];
                let s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                let u = a[m * n + m - 1].abs() * (q.abs() + r.abs());
                let v = p.abs()
                    * (a[(m - 1) * n + m - 1].abs() + z.abs() + a[(m + 1) * n + m + 1].abs());
                if u + v == v {
                    break;
                }
                m -= 1;
            }
            for i in (m + 2)..=nnu {
                a[i * n + i - 2] = 0.0;
                if i != m + 2 {
                    a[i * n + i - 3] = 0.0;
                }
            }
            // Double QR step on rows l..nn, columns m..nn.
            for k in m..=(nnu - 1) {
                if k != m {
                    p = a[k * n + k - 1];
                    q = a[(k + 1) * n + k - 1];
                    r = if k != nnu - 1 { a[(k + 2) * n + k - 1] } else { 0.0 };
                    x = p.abs() + q.abs() + r.abs();
                    if x != 0.0 {
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                }
                let s_mag = (p * p + q * q + r * r).sqrt();
                let s = s_mag.copysign(if p >= 0.0 { 1.0 } else { -1.0 });
                if s != 0.0 {
                    if k == m {
                        if l != m {
                            a[k * n + k - 1] = -a[k * n + k - 1];
                        }
                    } else {
                        a[k * n + k - 1] = -s * x;
                    }
                    p += s;
                    x = p / s;
                    y = q / s;
                    let z = r / s;
                    q /= p;
                    r /= p;
                    for j in k..=nnu {
                        let mut pj = a[k * n + j] + q * a[(k + 1) * n + j];
                        if k != nnu - 1 {
                            pj += r * a[(k + 2) * n + j];
                            a[(k + 2) * n + j] -= pj * z;
                        }
                        a[(k + 1) * n + j] -= pj * y;
                        a[k * n + j] -= pj * x;
                    }
                    let mmin = if nnu < k + 3 { nnu } else { k + 3 };
                    for i in l..=mmin {
                        let mut pi = x * a[i * n + k] + y * a[i * n + k + 1];
                        if k != nnu - 1 {
                            pi += z * a[i * n + k + 2];
                            a[i * n + k + 2] -= pi * r;
                        }
                        a[i * n + k + 1] -= pi * q;
                        a[i * n + k] -= pi;
                    }
                }
            }
        }
    }
    // Deterministic order: sort by (re, im).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        (wr[i], wi[i]).partial_cmp(&(wr[j], wi[j])).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok((idx.iter().map(|&i| wr[i]).collect(), idx.iter().map(|&i| wi[i]).collect()))
}

/// Shift-invert Arnoldi: the eigenvalues of A nearest the real shift sigma.
/// LU-factors (A - sigma I) once, builds an m-step Krylov space of the
/// resolvent with full modified-Gram-Schmidt orthogonalization, takes the
/// small Hessenberg's eigenvalues with the same Francis QR as `eig_general`,
/// and maps Ritz values back through lambda = sigma + 1/theta. Returns up to
/// k pairs sorted by |lambda - sigma| (ties by re, then im).
fn arnoldi_near(
    a_in: &[f64],
    n: usize,
    sigma: f64,
    k: usize,
    m: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut shifted = a_in.to_vec();
    for i in 0..n {
        shifted[i * n + i] -= sigma;
    }
    let piv = lu_factor(&mut shifted, n)
        .map_err(|e| format!("`eig_near`: shift sigma = {sigma} makes A - sigma*I {e}; move sigma off the spectrum"))?;
    let m = m.min(n);
    // Krylov basis, row j = v_j.
    let mut v = vec![0.0f64; (m + 1) * n];
    let mut h = vec![0.0f64; (m + 1) * m];
    let mut nrm = 0.0f64;
    for i in 0..n {
        v[i] = (1.0 + 0.7 * i as f64).vsin();
        nrm += v[i] * v[i];
    }
    nrm = nrm.sqrt();
    for i in 0..n {
        v[i] /= nrm;
    }
    let mut steps = m;
    let mut w = vec![0.0f64; n];
    for j in 0..m {
        w.copy_from_slice(&v[j * n..(j + 1) * n]);
        lu_solve_factored(&shifted, &piv, n, &mut w);
        // full MGS against every basis vector
        for i in 0..=j {
            let mut dot = 0.0f64;
            for t in 0..n {
                dot += v[i * n + t] * w[t];
            }
            h[i * m + j] = dot;
            for t in 0..n {
                w[t] -= dot * v[i * n + t];
            }
        }
        let mut b = 0.0f64;
        for t in 0..n {
            b += w[t] * w[t];
        }
        b = b.sqrt();
        h[(j + 1) * m + j] = b;
        if b < 1e-13 {
            // lucky breakdown: the Krylov space is exact at dimension j+1
            steps = j + 1;
            break;
        }
        for t in 0..n {
            v[(j + 1) * n + t] = w[t] / b;
        }
    }
    // eigenvalues of the leading steps x steps Hessenberg block
    let s = steps;
    let mut hs = vec![0.0f64; s * s];
    for i in 0..s {
        for j in 0..s {
            hs[i * s + j] = h[i * m + j];
        }
    }
    let (tr, ti) = general_eigvals(&hs, s)?;
    // theta -> lambda = sigma + 1/theta; sort by |lambda - sigma| = 1/|theta|
    let mut lam: Vec<(f64, f64, f64)> = Vec::with_capacity(s);
    for i in 0..s {
        let den = tr[i] * tr[i] + ti[i] * ti[i];
        if den < 1e-300 {
            continue;
        }
        let re = sigma + tr[i] / den;
        let im = -ti[i] / den;
        lam.push((1.0 / den.sqrt(), re, im));
    }
    lam.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.min(lam.len());
    Ok((lam[..k].iter().map(|t| t.1).collect(), lam[..k].iter().map(|t| t.2).collect()))
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations (values only,
/// ascending). Same sweep/thresholds as the `eig_sym` builtin.
fn jacobi_eigvals(a_in: &[f64], n: usize) -> Vec<f64> {
    let mut m = a_in.to_vec();
    let scale: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    'sweeps: for _ in 0..100 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off.sqrt() < 1e-12 * scale {
            break 'sweeps;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-15 * scale {
                    continue;
                }
                let (app, aqq) = (m[p * n + p], m[q * n + q]);
                let theta = 0.5 * (2.0 * apq).vatan2(app - aqq);
                let (c, s) = (theta.vcos(), theta.vsin());
                for k in 0..n {
                    if k != p && k != q {
                        let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                        let nkp = c * mkp + s * mkq;
                        let nkq = -s * mkp + c * mkq;
                        m[k * n + p] = nkp;
                        m[p * n + k] = nkp;
                        m[k * n + q] = nkq;
                        m[q * n + k] = nkq;
                    }
                }
                m[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
                m[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
            }
        }
    }
    let mut d: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    d
}

/// Implicit QL with Wilkinson shifts on the symmetric tridiagonal with
/// diagonal `d` and subdiagonal `e` (`e[k]` couples `k` and `k + 1`; the last
/// entry is ignored). This is `tql2` of EISPACK (Bowdler, Martin, Reinsch and
/// Wilkinson 1968) in the form the public-domain JAMA library gives it. On
/// return `d` holds the eigenvalues, unsorted. When `z` is given (row-major
/// n x n) the rotations are applied to its columns, so starting from the
/// identity its column `k` ends as the eigenvector of `d[k]`. Returns false
/// when one eigenvalue takes more than `maxit` QL steps.
fn tql2(d: &mut [f64], e: &mut [f64], mut z: Option<&mut [f64]>, maxit: usize) -> bool {
    let n = d.len();
    if n == 0 {
        return true;
    }
    e[n - 1] = 0.0;
    let eps = f64::EPSILON;
    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;
    for l in 0..n {
        // find a negligible subdiagonal entry at or below l
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m + 1 < n && e[m].abs() > eps * tst1 {
            m += 1;
        }
        if m > l {
            let mut iter = 0;
            loop {
                iter += 1;
                if iter > maxit {
                    return false;
                }
                // Wilkinson shift from the 2 x 2 at l
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.vhypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for v in d.iter_mut().take(n).skip(l + 2) {
                    *v -= h;
                }
                f += h;
                // the QL sweep from m up to l
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0f64, 1.0f64, 1.0f64);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0f64, 0.0f64);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.vhypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    if let Some(z) = z.as_deref_mut() {
                        for k in 0..n {
                            let zk = z[k * n + i + 1];
                            z[k * n + i + 1] = s * z[k * n + i] + c * zk;
                            z[k * n + i] = c * z[k * n + i] - s * zk;
                        }
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        // undo the accumulated shift
        d[l] += f;
        e[l] = 0.0;
    }
    true
}

/// Eigenvalues of a symmetric matrix by Householder tridiagonalization and
/// implicit QL: the reduction half of JAMA's `tred2` (public domain, after
/// EISPACK), with no eigenvector accumulation, then `tql2` above. A different
/// algorithm family from Jacobi, the second, independent method the
/// `spectrum_sym` sweep cross-checks against. Ascending.
fn tridiag_eigvals(a_in: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if n == 1 {
        return Ok(vec![a_in[0]]);
    }
    // JAMA's V, row-major; the reduction reads and writes only its lower part
    let mut v = a_in.to_vec();
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
    }
    for i in (1..n).rev() {
        // scale the row so h neither overflows nor underflows
        let mut scale = 0.0f64;
        let mut h = 0.0f64;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        } else {
            // the Householder vector; its sign avoids cancellation
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in 0..i {
                e[j] = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[j * n + i] = f;
                g = e[j] + v[j * n + j] * f;
                for k in (j + 1)..i {
                    g += v[k * n + j] * d[k];
                    e[k] += v[k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    // the reduced diagonal is what tred2's accumulation step would read off V
    for i in 0..n {
        d[i] = v[i * n + i];
    }
    // e[i] couples i - 1 and i; tql2 wants it coupling i and i + 1
    for i in 0..n - 1 {
        e[i] = e[i + 1];
    }
    if !tql2(&mut d, &mut e, None, 50) {
        return Err("`spectrum_sym`: QL iteration failed to converge".into());
    }
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Ok(d)
}

fn new_arr(shape: Vec<usize>, data: Vec<f64>) -> Val {
    Val::Arr(Rc::new(RefCell::new(ArrData { shape, data })))
}

/// A complex array; `data` is interleaved (re, im), twice the element count.
fn new_carr(shape: Vec<usize>, data: Vec<f64>) -> Val {
    debug_assert_eq!(data.len(), 2 * shape.iter().product::<usize>());
    Val::CArr(Rc::new(RefCell::new(ArrData { shape, data })))
}

/// `a % b`, floored: Python's rule for floats (Objects/floatobject.c,
/// `float_rem`). C's fmod first, which is exact; then, when the remainder and
/// the divisor have opposite signs, the divisor is added once. A zero
/// remainder takes the divisor's sign. So `-7 % 3` is 2 and `7 % -3` is -2.
/// The C runtime (`viv_floormod`) and the wasm runtime run these same steps.
pub fn floor_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// `a // b`, floor division: Python's rule for floats (`float_floor_div`).
/// The quotient is `(a - fmod(a, b)) / b`, one less when the remainder was
/// moved into the divisor's sign, then rounded to the nearest integer below
/// or at it (the division can land just off one). `a == (a // b) * b + a % b`
/// holds to rounding, and exactly for integers below 2^53.
pub fn floor_div(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        div -= 1.0;
    }
    if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 {
            f + 1.0
        } else {
            f
        }
    } else {
        0.0f64.copysign(a / b)
    }
}

/// The error for `a % 0` or `a // 0`: every engine stops with these words.
pub fn zero_divisor(op: BinOp) -> String {
    crate::errkind::kinded(crate::errkind::ErrKind::ZeroDiv, format!("division by zero in `a {} b`", if op == BinOp::Mod { "%" } else { "//" }))
}

fn map_arr(a: &Arr, f: impl Fn(f64) -> f64) -> Val {
    let ab = a.borrow();
    new_arr(ab.shape.clone(), ab.data.iter().map(|&x| f(x)).collect())
}

fn format_arr(shape: &[usize], data: &[f64]) -> String {
    format_arr_at(shape, data, 1)
}

/// One complex number, exactly as the C runtime's `%g%+gi` prints it.
fn format_cplx(re: f64, im: f64) -> String {
    let sign = if im.is_sign_negative() && im != 0.0 || im < 0.0 { "-" } else { "+" };
    format!("{}{}{}i", format_g(re), sign, format_g(im.abs()))
}

/// A complex array: interleaved (re, im) data, rows aligned like a real matrix.
fn format_carr_at(shape: &[usize], data: &[f64], indent: usize) -> String {
    let cell = |k: usize| format_cplx(data[2 * k], data[2 * k + 1]);
    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        let mut s = String::from("[");
        for i in 0..rows {
            if i > 0 {
                for _ in 0..indent {
                    s.push(' ');
                }
            }
            s.push('[');
            for j in 0..cols {
                if j > 0 {
                    s.push_str(", ");
                }
                s.push_str(&cell(i * cols + j));
            }
            s.push(']');
            if i < rows - 1 {
                s.push_str(",\n");
            }
        }
        s.push(']');
        s
    } else {
        let n = data.len() / 2;
        let inner: Vec<String> = (0..n).map(cell).collect();
        format!("[{}]", inner.join(", "))
    }
}

/// Format an array, indenting a rank-2 array's continuation rows by `indent`
/// columns so every row's `[` lines up with the first one's. The caller knows
/// what precedes the value on the line (`show M` writes "M: " first), so it
/// passes the column the opening bracket landed in.
fn format_arr_at(shape: &[usize], data: &[f64], indent: usize) -> String {
    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        let mut s = String::from("[");
        for i in 0..rows {
            if i > 0 {
                for _ in 0..indent {
                    s.push(' ');
                }
            }
            s.push('[');
            for j in 0..cols {
                if j > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format_g(data[i * cols + j]));
            }
            s.push(']');
            if i < rows - 1 {
                s.push_str(",\n");
            }
        }
        s.push(']');
        s
    } else {
        let inner: Vec<String> = data.iter().map(|&x| format_g(x)).collect();
        format!("[{}]", inner.join(", "))
    }
}

/// Replicate C `printf("%g", x)`: 6 significant digits, shortest of %e/%f, with
/// trailing zeros trimmed. Close enough that interpreter output matches the
/// compiled C backend on the programs the interpreter supports.
/// C's `%.17e`: Rust's `{:.17e}` has the same digits but writes the exponent as
/// `e0` where C writes `e+00`, and the two engines must print the same bytes.
fn format_e17(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".to_string() } else { "inf".to_string() };
    }
    let s = format!("{x:.17e}");
    let (mantissa, exp) = s.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("a decimal exponent");
    format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

/// A number in the story: C's `%.17e`, and a non-finite value as a JSON
/// string, since JSON has no NaN or infinity.
fn story_num(x: f64) -> String {
    if x.is_nan() {
        "\"nan\"".to_string()
    } else if x.is_infinite() {
        if x < 0.0 { "\"-inf\"".to_string() } else { "\"inf\"".to_string() }
    } else {
        format_e17(x)
    }
}

/// The largest |x| of an array; a NaN wins, so it is never hidden.
fn max_abs(data: &[f64]) -> f64 {
    let mut w = 0.0_f64;
    for &x in data {
        let a = x.abs();
        if a > w || a.is_nan() {
            w = a;
        }
    }
    w
}

/// A type's name with its article: "an Int", "a number".
fn an(name: &str) -> String {
    if name.starts_with(['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']) { format!("an {name}") } else { format!("a {name}") }
}

/// The record type of the arrays of records among `vals`, for messages
/// (`?` when every one is empty).
fn rec_name(vals: &[Val]) -> String {
    for v in vals {
        if let Val::Recs(xs) = v {
            if let Some(Val::Record(r)) = xs.first() {
                return r.name.to_string();
            }
        }
    }
    "?".to_string()
}

/// The 1-argument math builtins that map over an array's elements.
fn elementwise_fn(callee: &str) -> Option<fn(f64) -> f64> {
    match callee {
        "exp" => Some(crate::vmath::exp),
        "sqrt" => Some(f64::sqrt),
        "sin" => Some(crate::vmath::sin),
        "cos" => Some(crate::vmath::cos),
        "tan" => Some(crate::vmath::tan),
        "log" | "ln" => Some(crate::vmath::log),
        "abs" | "fabs" => Some(f64::abs),
        "floor" => Some(f64::floor),
        "ceil" => Some(f64::ceil),
        "round" => Some(f64::round),
        "asin" => Some(crate::vmath::asin),
        "acos" => Some(crate::vmath::acos),
        "atan" => Some(crate::vmath::atan),
        "sinh" => Some(crate::vmath::sinh),
        "cosh" => Some(crate::vmath::cosh),
        "tanh" => Some(crate::vmath::tanh),
        "erf" => Some(erf_impl),
        "gamma" => Some(gamma_impl),
        _ => None,
    }
}

/// `append(xs, r)` when it is about records: an array of records (or `[]`)
/// with `r` on its end, as a new value. `None` when neither side is a record.
fn append_record(cur: &Val, x: &Val) -> Option<Result<Val, String>> {
    let empty = |v: &Val| matches!(v, Val::Arr(a) if a.borrow().data.is_empty());
    match (cur, x) {
        (Val::Recs(v), Val::Record(r)) => {
            if let Some(Val::Record(first)) = v.first() {
                if first.name != r.name {
                    return Some(Err(format!("this array holds {} records, and this is {}", first.name, crate::lower::a_or_an(&r.name))));
                }
            }
            let mut out = (**v).clone();
            out.push(x.clone());
            Some(Ok(Val::Recs(Rc::new(out))))
        }
        (c, Val::Record(_)) if empty(c) => Some(Ok(Val::Recs(Rc::new(vec![x.clone()])))),
        (Val::Recs(_), other) => Some(Err(format!("this array holds records, and this is {}", crate::lower::a_or_an(&val_kind(other))))),
        _ => None,
    }
}

/// A value's type as the C lowering names it (`lower::Ty::name`), for the
/// messages both engines give.
/// The detail of a test block's FAIL line when an error stopped the program
/// inside it (the C runtime's `viv_test_stopped` writes the same).
pub(crate) const TEST_STOPPED: &str = "an error stopped the program";

/// `[]` (an empty real array, as the interpreter evaluates it) where a field
/// of type `t` wants an empty array of records, list of strings or Int array:
/// that empty value. None for anything else.
fn empty_as(t: crate::lower::Ty, v: &Val) -> Option<Val> {
    use crate::lower::{Elem, Ty};
    let Val::Arr(a) = v else { return None };
    if !a.borrow().data.is_empty() {
        return None;
    }
    match t {
        Ty::Recs(_) => Some(Val::Recs(Rc::new(Vec::new()))),
        Ty::Arr(Elem::Str) => Some(Val::Strs(Rc::new(RefCell::new(Vec::new())))),
        Ty::Arr(Elem::Int) => Some(Val::IArr(Rc::new(RefCell::new(Vec::new())))),
        _ => None,
    }
}

fn val_kind(v: &Val) -> String {
    match v {
        Val::Num(_) => "number".into(),
        Val::Bool(_) => "boolean".into(),
        Val::Cplx(..) => "complex number".into(),
        Val::Arr(_) => "array".into(),
        Val::CArr(_) => "complex array".into(),
        Val::Closure(_) | Val::Func(..) => "function".into(),
        Val::VArr(_) => "function array".into(),
        Val::Str(_) => "string".into(),
        Val::Strs(_) => "list of strings".into(),
        Val::Record(r) => r.name.to_string(),
        Val::Map(_) => "map".into(),
        Val::Int(_) => "Int".into(),
        Val::IArr(_) => "Int array".into(),
        Val::Enum(_) => "enum value".into(),
        Val::Recs(_) => "array of records".into(),
        Val::F32(_) => "Float32".into(),
        Val::PArr(p) => crate::lower::Ty::Arr(crate::lower::Elem::P(p.borrow().k)).name().into(),
    }
}

/// An enum value as `print`, `str` and `show` write it: `Add(Num(1), Sym('x'))`
/// with a stack of what is left to write
/// instead of recursion. The C runtime's `viv_en_put_*` and the wasm
/// runtime's `en_text` write the same.
pub(crate) fn enum_text(v: &EnumVal, out: &mut String) {
    enum Job<'a> {
        Text(&'static str),
        Val(&'a Val),
        Node(&'a EnumVal),
    }
    let mut jobs = vec![Job::Node(v)];
    while let Some(j) = jobs.pop() {
        match j {
            Job::Text(t) => out.push_str(t),
            Job::Node(v) => {
                out.push_str(&v.variant);
                if v.fields.is_empty() {
                    continue;
                }
                out.push('(');
                jobs.push(Job::Text(")"));
                for (k, f) in v.fields.iter().enumerate().rev() {
                    jobs.push(Job::Val(f));
                    if k > 0 {
                        jobs.push(Job::Text(", "));
                    }
                }
            }
            Job::Val(f) => match f {
                Val::Num(x) => out.push_str(&format_g(*x)),
                Val::Int(i) => out.push_str(&i.to_string()),
                Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                Val::Str(st) => out.push_str(&strings::repr(&st.text)),
                Val::Enum(e) => jobs.push(Job::Node(e)),
                other => out.push_str(&val_kind(other)),
            },
        }
    }
}

/// A watched value as story JSON: a number, a boolean, or an array (whole
/// when it has at most STORY_ARR_CAP elements, else summarized).
fn story_value(name: &str, v: &Val) -> Result<String, String> {
    match v {
        Val::Num(x) => Ok(story_num(*x)),
        Val::Bool(b) => Ok(b.to_string()),
        Val::Arr(a) => {
            let a = a.borrow();
            let shape = a.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            if a.data.len() <= STORY_ARR_CAP {
                let vals = a.data.iter().map(|&x| story_num(x)).collect::<Vec<_>>().join(",");
                return Ok(format!("{{\"shape\":[{shape}],\"values\":[{vals}]}}"));
            }
            let (mut lo, mut hi, mut ss) = (a.data[0], a.data[0], 0.0_f64);
            for &x in &a.data {
                if x < lo {
                    lo = x;
                }
                if x > hi {
                    hi = x;
                }
                ss += x * x;
            }
            let mut s = format!("{{\"shape\":[{shape}],\"min\":{},\"max\":{},\"norm\":{}", story_num(lo), story_num(hi), story_num(ss.sqrt()));
            if a.shape.len() == 2 && a.shape[0] == a.shape[1] {
                let n = a.shape[0];
                let mut tr = 0.0_f64;
                for i in 0..n {
                    tr += a.data[i * n + i];
                }
                s.push_str(&format!(",\"trace\":{}", story_num(tr)));
            }
            s.push('}');
            Ok(s)
        }
        _ => Err(format!("cannot watch `{name}`: a watch records numbers, booleans and real arrays")),
    }
}

/// C's `%g`, as `show` and `print` write a number: `crate::printfmt::bare`.
fn format_g(x: f64) -> String {
    crate::printfmt::bare(x)
}

/// C's `%.<p>g`: `crate::printfmt::g`.
fn format_gp(x: f64, p: usize) -> String {
    crate::printfmt::g(x, p as u32)
}

/// One number of a chart's data (plot, bars, plot2d): the shortest of C's
/// `%.15g`, `%.16g` and `%.17g` that reads back as the same double, so the
/// record carries the value itself, not six digits of it. The C runtime's
/// `viv_prt` is this, line for line; a nan prints as `nan` whatever its sign.
/// An index as the integer it must be. A number that is not exactly an
/// integer (a fraction, NaN, an infinity) stops the run: `v[2.9999999]` used
/// to read `v[2]`, `v[0.0 / 0.0]` read `v[0]`, and `w[0.29 * 100]` read
/// `w[28]`, silently, in every engine. The C runtime's `viv_ix` and the wasm
/// backend's index check stop with these words too.
/// Seconds from a monotonic clock, counted from the first reading's process
/// start (only differences mean anything).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn clock_seconds() -> Result<f64, String> {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let t0 = *START.get_or_init(std::time::Instant::now);
    Ok(t0.elapsed().as_secs_f64())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn clock_seconds() -> Result<f64, String> {
    Err("clock() has no clock to read in the interpreter in a browser; run the program natively or with the direct wasm engine".into())
}

pub(crate) fn index_int(x: f64) -> Result<i64, String> {
    if x - x.floor() != 0.0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("index {} is not an integer", format_rt(x))));
    }
    Ok(x as i64)
}

/// The builtins with a Float32 or packed form; `None`
/// for everything else. The lowering's `f32s::lower_call` picks the same.
fn f32_builtin(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    use crate::ast::Store;
    let mine = |v: Option<&Val>| matches!(v, Some(Val::F32(_) | Val::PArr(_)));
    let packed = |v: &Val| -> Option<(Store, Vec<u8>)> { if let Val::PArr(p) = v { Some((p.borrow().k, p.borrow().b.clone())) } else { None } };
    if let Some(k) = Store::of_zeros(callee) {
        let n = match a.first() {
            Some(Val::Int(i)) => *i as f64,
            Some(v) => v.as_num("a length")?,
            None => return Err(format!("`{callee}` takes one length")),
        };
        let n = zeros_shape(&[n]).0.first().copied().unwrap_or(0);
        return Ok(Some(parr(k, vec![0u8; n * k.width()])));
    }
    if let Some(k) = Store::of_conv(callee) {
        let Some(x) = a.first() else { return Err(format!("`{callee}` takes one argument")) };
        return Ok(Some(match (k, x) {
            (Store::F32, Val::Num(v)) => Val::F32(*v as f32),
            (Store::F32, Val::Int(i)) => Val::F32(*i as f32),
            (Store::F32, Val::F32(v)) => Val::F32(*v),
            (Store::BF16 | Store::F16, Val::F32(v)) => Val::F32(f32s::round_to(k, *v)),
            (_, Val::Arr(arr)) => parr(k, f32s::pack_floats(k, arr.borrow().data.iter().map(|x| *x as f32))),
            (_, Val::IArr(arr)) => parr(k, f32s::pack_ints(k, arr.borrow().iter().copied())?),
            (_, Val::PArr(p)) => {
                let p = p.borrow();
                parr(k, f32s::convert(p.k, k, &p.b)?)
            }
            (_, v) => return Err(format!("`{callee}` does not take a {}", val_kind(v))),
        }));
    }
    if !mine(a.first()) && !mine(a.get(1)) {
        return Ok(None);
    }
    Ok(Some(match (callee, a) {
        ("int64", [x]) => {
            let (k, b) = packed(x).ok_or("int64 takes an integer storage array")?;
            Val::IArr(Rc::new(RefCell::new((0..f32s::count(k, &b)).map(|i| if let f32s::Elt::I(n) = f32s::get(k, &b, i) { n } else { 0 }).collect())))
        }
        ("float", [Val::F32(x)]) => Val::Num(*x as f64),
        ("float", [x]) => {
            let (k, b) = packed(x).ok_or("float takes a Float32 or a float storage array")?;
            let d: Vec<f64> = f32s::floats(k, &b).into_iter().map(|x| x as f64).collect();
            new_arr(vec![d.len()], d)
        }
        ("len", [x]) => {
            let (k, b) = packed(x).ok_or("len of a packed array")?;
            Val::Num(f32s::count(k, &b) as f64)
        }
        ("sum", [x]) => {
            let (k, b) = packed(x).ok_or("sum of a packed array")?;
            if k.is_float() { Val::F32(f32s::sum(&f32s::floats(k, &b))) } else { Val::Int(f32s::isum(k, &b)?) }
        }
        ("dot", [x, y]) => {
            let (k1, b1) = packed(x).ok_or("dot of [Float32]")?;
            let (k2, b2) = packed(y).ok_or("dot of [Float32]")?;
            Val::F32(f32s::dot(&f32s::floats(k1, &b1), &f32s::floats(k2, &b2))?)
        }
        ("fmt" | "str", [Val::F32(x)]) => strings::sval(f32s::repr(*x)),
        ("min", [Val::F32(x), Val::F32(y)]) => Val::F32(f32s::min(*x, *y)),
        ("max", [Val::F32(x), Val::F32(y)]) => Val::F32(f32s::max(*x, *y)),
        ("sqrt", [Val::F32(x)]) => Val::F32(x.sqrt()),
        ("abs", [Val::F32(x)]) => Val::F32(x.abs()),
        (f, [Val::F32(x)]) if f32s::FUNCS.contains(&f) => Val::F32(f32s::func(f, *x).unwrap_or(f32::NAN)),
        _ => return Err(format!("`{callee}` is not defined on a Float32 or packed array in this version")),
    }))
}

/// The builtins with an Int form, when an argument is an Int (and the ones
/// that take Ints only); `None` for everything else.
/// The lowering's `ints::lower_call` picks the same forms.
fn int_builtin(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    use crate::printfmt::{Fmt, Kind};
    let int = |k: usize| -> Result<i64, String> {
        match a.get(k) {
            Some(Val::Int(i)) => Ok(*i),
            Some(v) => Err(format!("`{callee}` takes Ints, and this is {}", crate::lower::a_or_an(&val_kind(v)))),
            None => Err(format!("`{callee}` is missing an argument")),
        }
    };
    // an Int as text in a format: its digits, or the number it is in a float format
    let text = |i: i64, f: Fmt| -> Result<String, String> {
        Ok(match f.kind {
            Kind::Bare | Kind::Repr | Kind::D | Kind::Str => crate::printfmt::int_text(i, f),
            _ => crate::printfmt::number(ints::to_num(i)?, f),
        })
    };
    if let (Some(Val::IArr(arr)), "len", 1) = (a.first(), callee, a.len()) {
        return Ok(Some(Val::Num(arr.borrow().len() as f64)));
    }
    let any = a.iter().any(|v| matches!(v, Val::Int(_)));
    Ok(Some(match (callee, a.len()) {
        ("wrap_add" | "wrap_sub" | "wrap_mul" | "umulhi", 2) => Val::Int(ints::bits(callee, int(0)?, int(1)?)?),
        ("ult", 2) => Val::Bool(ints::ult(int(0)?, int(1)?)),
        ("ustr", 1) => strings::sval(ints::ustr(int(0)?)),
        ("fround", 1) => match &a[0] {
            Val::Num(x) => Val::Num(*x as f32 as f64),
            Val::Int(i) => Val::Num(ints::to_num(*i)? as f32 as f64),
            v => return Err(format!("`fround` takes a number, and this is {}", crate::lower::a_or_an(&val_kind(v)))),
        },
        // the little-endian readers: an Int array
        // of bytes and an offset
        (c, 2) if bytes::reader(c).is_some() => {
            let Some(Val::IArr(b)) = a.first() else {
                return Err(format!("`{c}` takes an array of Ints (bytes) and an offset"));
            };
            let off = match &a[1] {
                Val::Int(i) => *i,
                Val::Num(x) if x.floor() == *x && x.abs() < 9.0e15 => *x as i64,
                v => return Err(format!("`{c}` takes a whole offset, and this is {}", crate::lower::a_or_an(&val_kind(v)))),
            };
            match bytes::read(c, &b.borrow(), off)? {
                bytes::Read::Int(i) => Val::Int(i),
                bytes::Read::Num(x) => Val::Num(x),
            }
        }
        ("int_zeros", 1) => {
            let (_, total) = zeros_shape(&[a[0].as_num("a length")?]);
            if total > MAX_ELEMS as u64 {
                return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("zeros: array of {total} elements exceeds the {MAX_ELEMS}-element limit{LIMIT_OF}")));
            }
            Val::IArr(Rc::new(RefCell::new(vec![0; total as usize])))
        }
        _ if !any => return Ok(None),
        ("abs", 1) => Val::Int(ints::abs(int(0)?)?),
        ("floor" | "ceil" | "round" | "int", 1) => Val::Int(int(0)?),
        ("float", 1) => Val::Num(int(0)? as f64),
        ("min" | "max", 2) if a.iter().all(|v| matches!(v, Val::Int(_))) => {
            let (x, y) = (int(0)?, int(1)?);
            Val::Int(if callee == "min" { x.min(y) } else { x.max(y) })
        }
        ("pow", 2) if a.iter().all(|v| matches!(v, Val::Int(_))) => Val::Int(ints::pow(int(0)?, int(1)?)?),
        ("bnot", 1) => Val::Int(!int(0)?),
        ("band" | "bor" | "bxor" | "shl" | "shr" | "sar", 2) => Val::Int(ints::bits(callee, int(0)?, int(1)?)?),
        ("str" | "repr", 1) => strings::sval(ints::text(int(0)?, 0)),
        (c, 1) if c == strings::STR_OF || c == strings::REPR_OF => strings::sval(ints::text(int(0)?, 0)),
        (c, _) if (c == "fmt" || c == strings::FMT_OF) && matches!(a[0], Val::Int(_)) => {
            let f = match a.get(1) {
                None => Fmt::REPR,
                Some(Val::Str(st)) => {
                    let f = Fmt::parse_spec(&st.text)?;
                    if f.stars() != a.len() - 2 {
                        return Err("`fmt` takes a value and at most one format".into());
                    }
                    f.with_stars(&strings::star_values(&a[2..])?)?
                }
                Some(Val::Num(d)) => Fmt::digits(*d as u32),
                Some(v) => return Err(format!("`fmt`'s format is a string or a digit count, not a {}", val_kind(v))),
            };
            strings::sval(text(int(0)?, f)?)
        }
        _ => return Ok(None),
    }))
}

/// `Name.Variant(...)`, `Some(x)`, `None`, and the calls a `match` is written
/// with; `None` for every other callee.
fn enum_builtin(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    use crate::parser::{ENUM_FIELD, ENUM_IS, MATCH_REST};
    let make = |variant: &str, fields: Vec<Val>| Val::Enum(Rc::new(EnumVal { variant: Rc::from(variant), fields }));
    if let Some((_, v)) = callee.split_once('.') {
        return Ok(Some(make(v, a.to_vec())));
    }
    Ok(Some(match (callee, a) {
        ("Some", [x]) => make("Some", vec![x.clone()]),
        ("None", []) => make("None", Vec::new()),
        (c, [Val::Enum(e), Val::Str(v)]) if c == ENUM_IS => Val::Bool(*e.variant == *v.text),
        (c, [Val::Enum(e), Val::Str(_), k]) if c == ENUM_FIELD => e.fields.get(k.as_num("a field")? as usize).cloned().ok_or("internal: no such payload field")?,
        (c, _) if c == MATCH_REST => return Err("internal: no arm of this match matched".into()),
        // an enum value's text
        (c, [Val::Enum(e)]) if matches!(c, "str" | "repr" | "fmt") || c == strings::STR_OF || c == strings::REPR_OF || c == strings::FMT_OF => {
            let mut t = String::new();
            enum_text(e, &mut t);
            strings::sval(t)
        }
        _ => return Ok(None),
    }))
}

/// An Int array as `print` and `show` write it: `[1, 2, 3]`.
pub(crate) fn iarr_text(a: &[i64]) -> String {
    let parts: Vec<String> = a.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

pub(crate) fn format_rt(x: f64) -> String {
    let x = x + 0.0;
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".to_string() } else { "inf".to_string() };
    }
    if x == 0.0 {
        return "0".to_string();
    }
    for p in [15, 16] {
        let s = format_gp(x, p);
        if s.parse::<f64>() == Ok(x) {
            return s;
        }
    }
    format_gp(x, 17)
}

/// A chart's data array, flat, at full precision: `[a, b, c]`.
fn format_data(data: &[f64]) -> String {
    let inner: Vec<String> = data.iter().map(|&x| format_rt(x)).collect();
    format!("[{}]", inner.join(", "))
}

/// The low 32 bits of a Viv number: truncate toward zero, then wrap into
/// [0, 2^32). The C backend's viv_u32 is this arithmetic, line for line.
pub fn u32_of(x: f64) -> u32 {
    let t = x.trunc() % 4294967296.0;
    let t = if t < 0.0 { t + 4294967296.0 } else { t };
    t as u32
}

/// A shift distance: 0..=31, or None when the shift takes everything away.
fn shift_amount(k: f64) -> Option<u32> {
    let k = k.trunc();
    if (0.0..32.0).contains(&k) { Some(k as u32) } else { None }
}

/// erf and Γ live in `crate::vmath`, beside their C twins in vmath.c.
fn erf_impl(x: f64) -> f64 {
    crate::vmath::erf(x)
}

// Γ via the Lanczos approximation (g = 7, 9 terms), reflection below 1/2.
fn gamma_impl(x: f64) -> f64 {
    crate::vmath::gamma(x)
}

// The runtime a compiled program imports when it runs as a wasm module built
// by `crate::wasm_backend`. A child of the interpreter, so every builtin a
// compiled program calls runs this file's own code (see its header).
#[cfg(target_arch = "wasm32")]
mod wasm_rt;

// Slices, `copy`, `concat`, `hcat`, `vcat`: the rules, the interpreter's
// arms, and their C twin.
pub(crate) mod slices;

// Growable arrays, `append` and `extend`: the rules, the interpreter's arms,
// and their C twin.
pub(crate) mod growable;

// Strings: the rules, the interpreter's arms, the
// registry of builtins, and the character data they read.
pub(crate) mod strings;
/// The rules of `Int`, shared with the wasm runtime.
pub(crate) mod ints;
/// The rules of `Float32` and the packed arrays, shared
/// with the wasm runtime.
pub(crate) mod f32s;
/// Binary input: `read_bytes` and the little-endian readers
/// shared with the wasm runtime.
pub(crate) mod bytes;
pub(crate) mod unicode_tables;

// Maps: `map()`, `m[k] = v`, `has`, `get`, `remove`, `keys`, `values`: the
// rules, the interpreter's arms, and their C twin.
pub(crate) mod maps;

// The array vocabulary: shapes, reductions, broadcasting, gathers, einsum and
// the returning solvers; the rules, the interpreter's arms and their C twin
//.
pub(crate) mod arrays;
/// `@` on real arrays, batched, and its adjoints (one statement of the rules
/// for the interpreter, the wasm runtime and the C runtime).
pub(crate) mod matmul;
