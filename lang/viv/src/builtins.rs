// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! What each builtin does to memory, in one place.
//!
//! Three questions about a builtin decide memory policy and analysis, and
//! each used to be answered by a list of its own in the module that asked it:
//!
//! - **Which arguments does it write?** `graph.rs` (the program graph's write
//!   edges), `grad.rs` (which values the reverse sweep must save) and the
//!   `parallel for` check (`crate::parallel`) ask it of a Viv builtin by its
//!   Viv name. [`writes`] answers.
//! - **Is it pure** (a value from its arguments, nothing changed, no draw from
//!   the generator)? `grad.rs` re-evaluates pure calls in the reverse sweep,
//!   and a debug-lens `hold` may call only pure builtins. [`is_pure`].
//! - **Does its runtime call allocate?** The C emitter and the direct wasm
//!   backend free a loop's per-pass garbage only when the loop can allocate,
//!   and they ask it of the C function a call is lowered to
//!   (`viv_map_keys`, `viv_ar_cumsum_1`, ...). [`runtime_gives`] says what each
//!   runtime call returns, and [`runtime_allocates`] is "it returns an array,
//!   a string, a list of strings or a map" (a call of a def allocates when the
//!   def returns one of those or a record, which the emitters read from the
//!   def's declared type).
//!
//! The lists disagreed before they were merged: the graph had no `ifft` and
//! said `eig_near` writes from its second argument (it writes its fourth and
//! fifth), and the C emitter's list missed `viv_map_keys` and
//! `viv_map_values`, so a loop calling `keys(m)` freed nothing
//! (study/memcheck/scale.tsv). The tests at the end of this file check the
//! table against the runtime's C prototypes (every runtime function a `Call`
//! can name that returns a heap value is listed as allocating, and none that
//! returns a number), against the interpreter's dispatch (every builtin it
//! accepts has an entry: it writes, it is pure, it has another effect, or it
//! only reads), and against the array vocabulary's own `writes`; and that
//! `graph.rs` and `grad.rs` give the table's answers.

use crate::interp::maps::Mk;
use crate::lower::{Elem, Ty, NUM_MAP};

// ---- Viv-level builtins: what they write, and whether they are pure ----------------

/// Builtins that write into array (or map) arguments: the first and the last
/// argument written. A call with fewer arguments writes the ones it has, so
/// `lin_solve(A, b)` (which returns its solution) writes nothing, and
/// `lin_solve(A, b, x)` writes `x`. Checked against the interpreter's code
/// for each (`interp.rs`, `interp/arrays.rs`, `interp/maps.rs`).
pub const WRITERS: &[(&str, usize, usize)] = &[
    ("eig_sym", 1, 2),
    ("eig_herm", 1, 2),
    ("eig_general", 1, 2),
    ("eig_near", 3, 4),
    ("spectrum_sym", 1, 1),
    ("cholesky", 1, 1),
    ("lin_solve", 2, 2),
    ("lin_solve_multi", 2, 2),
    ("put", 0, 0),
    ("putmask", 0, 0),
    ("svd", 1, 3),
    ("setslab", 0, 0),
    (crate::interp::slices::SET, 0, 0),
    ("sort", 0, 0),
    ("fill", 0, 0),
    ("fft", 0, 0),
    ("ifft", 0, 0),
    ("fft3", 0, 0),
    ("ifft3", 0, 0),
    // maps: `m[k] = v` is a `SetIndex`, a write already
    ("remove", 0, 0),
    ("clear", 0, 0),
    // the autodiff tape (crate::grad): the reverse sweep puts an array's
    // saved contents back into it
    ("__ad_restore", 0, 0),
];

/// The argument positions a builtin writes into, if it writes any.
pub fn writes(callee: &str) -> Option<std::ops::RangeInclusive<usize>> {
    WRITERS.iter().find(|(n, _, _)| *n == callee).map(|(_, a, b)| *a..=*b)
}

/// The positions of the arguments a call of `callee` with `nargs` arguments
/// writes into.
pub fn written_args(callee: &str, nargs: usize) -> Vec<usize> {
    writes(callee).map(|r| r.filter(|&k| k < nargs).collect()).unwrap_or_default()
}

/// Builtins that compute a value from their arguments and change nothing (the
/// array vocabulary's are added by `interp::arrays::is_pure`). A builtin that
/// prints, plots, asserts, draws a random number or writes an argument is not
/// here.
pub const PURE: &[&str] = &[
    "exp", "log", "ln", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "fma", "sinh", "cosh", "tanh", "erf", "gamma",
    "abs", "fabs", "floor", "ceil", "round", "pow", "min", "max", "mod", "sum", "len", "numel", "shape", "zeros", "linspace",
    "transpose", "take", "slab", "where", "all", "any", "u32", "i32", "band", "bor", "bxor", "bnot", "shl", "shr", "sar", "re", "im",
    "conj", "complex", "czeros", "roll", "histogram", "tridiag_solve", crate::interp::slices::GET, "copy", "concat", "vcat", "hcat",
    "append", "extend",
    // the Int builtins
    "wrap_add", "wrap_sub", "wrap_mul", "umulhi", "ult", "ustr", "int_zeros",
    // the little-endian readers
    "le_u8", "le_i8", "le_u16", "le_i16", "le_u32", "le_i32", "le_u64", "le_i64", "le_f16", "le_bf16", "le_f32", "le_f64", "fround",
];

/// Whether `name` computes a value from its arguments and changes nothing.
pub fn is_pure(name: &str) -> bool {
    PURE.contains(&name) || crate::interp::arrays::is_pure(name)
}

/// Builtins with an effect that is not a write of an argument: output (text,
/// charts, figures, files), input (files), the state of a run (the random
/// generator, the test counters, the autodiff tape, checkpoints), or a call
/// of a function value whose effects are the function's. None of them writes
/// an argument.
pub const EFFECTS: &[&str] = &[
    "print", "__print", "text", "show", "narrate", "plot", "plot2d", "plot3d", "scatter", "bars", "quiver", "figure", "grid", "scene",
    "wire", "verts", "checkpoint", "assert_allclose", "assert_gt", "assert_lt", "assert_near", "assert_true", "rand",
    "srand", "randn", "release_mode", "clock", "apply", "deriv", "simpson", "trapezoid", "__ad_push", "__ad_pop", "read_text", "read_lines", "read_bytes",
    "write_text",
    // an exported program's parameters (crate::export): the run's input
    "__export_given", "__export_num", "__export_int", "__export_bool", "__export_str", "__export_arr", "__export_iarr",
    // calls the def it is given on a record's arrays, which that def may
    // write (crate::graph follows it; a `parallel for` refuses it)
    "for_leaves",
];

/// Builtins that read their arguments and write none, but are not listed
/// as pure for `grad` and the debug lens: they return a new value whose
/// identity matters (`inv`), or format and convert (`fmt`).
/// The strings registry (`interp::strings::BUILTINS`), the map builtins
/// (`interp::maps::NAMES`, apart from `remove` and `clear`) and the array
/// vocabulary (`interp::arrays::SIGS`, which says itself what each writes)
/// are read from their own tables.
pub const READERS: &[&str] = &["inv", "fmt", "__ad_zero", "__ad_unb", "__ad_cmp", "__ad_mm_a", "__ad_mm_b", "__ad_take", "__ad_cat", "__ad_extent", "__ad_head", "__ad_tail", "__ad_keep", "__ad_like", "__ad_prod", "__ad_prodax", "__ad_takeax"];

/// What a builtin does to its arguments, when this table (or one of the
/// registries it reads) knows the name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// writes the arguments `writes` gives
    Writes,
    /// computes a value from its arguments, changes nothing
    Pure,
    /// writes no argument, and is not pure (output, run state, a function value)
    Effect,
    /// writes no argument; reads them
    Reads,
}

/// The kind of the builtin `name`, or None for a name no table knows.
pub fn kind(name: &str) -> Option<Kind> {
    if writes(name).is_some() || crate::interp::arrays::SIGS.iter().any(|s| s.name == name && s.writes.is_some()) {
        return Some(Kind::Writes);
    }
    if is_pure(name) {
        return Some(Kind::Pure);
    }
    if EFFECTS.contains(&name) {
        return Some(Kind::Effect);
    }
    if READERS.contains(&name) || crate::interp::strings::is_builtin(name) || crate::interp::maps::NAMES.contains(&name) || crate::interp::arrays::is_name(name) {
        return Some(Kind::Reads);
    }
    None
}

// ---- runtime calls: what each returns ------------------------------------------------

/// What a runtime call (a `Viv::Call` callee after lowering: a C function of
/// the runtime, or a libm name) gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gives {
    Num,
    /// an Int
    Int,
    /// a Float32
    F32,
    Bool,
    Cplx,
    /// a new array of numbers, complex numbers or strings (a list of strings)
    Arr(Elem),
    Str,
    /// a new map
    Map(Mk, Mk),
    /// a new value of the type its argument `k` has (`viv_slab(a, i)` of a
    /// complex array is a complex array)
    Like(usize),
}

impl Gives {
    /// A value that lives on the arena: an array, a string, a list of
    /// strings or a map. Each one a call gives is a new allocation.
    pub fn heap(self) -> bool {
        matches!(self, Gives::Arr(_) | Gives::Str | Gives::Map(..) | Gives::Like(_))
    }

    /// As a type, with `arg(k)` the type of argument `k`.
    pub fn ty(self, arg: impl Fn(usize) -> Ty) -> Ty {
        match self {
            Gives::Num => Ty::Num,
            Gives::Int => Ty::Int,
            Gives::F32 => Ty::F32,
            Gives::Bool => Ty::Bool,
            Gives::Cplx => Ty::Cplx,
            Gives::Arr(e) => Ty::Arr(e),
            Gives::Str => Ty::Str,
            Gives::Map(k, v) => Ty::Map(k, v),
            Gives::Like(k) => arg(k),
        }
    }
}

/// What the runtime call `callee` gives. A name not listed gives a number
/// (the libm functions, the scalar helpers). `tests/builtins_table.rs` checks
/// that every runtime C function returning an array, a string, a list of
/// strings or a map is listed as giving one.
pub fn runtime_gives(callee: &str) -> Gives {
    use crate::interp::strings::K;
    // strings: the string runtime's own table
    match callee {
        "viv_s_catn" | "viv_s_str_cplx" | "viv_s_fmt_arr" => return Gives::Str,
        "viv_ss_of" => return Gives::Arr(Elem::Str),
        _ => {}
    }
    // the string runtime's calls and what each gives, gathered once: this is
    // asked of every call in every loop body the emitters look at
    static STRINGS: std::sync::OnceLock<std::collections::HashMap<&'static str, K>> = std::sync::OnceLock::new();
    let strings = STRINGS.get_or_init(|| crate::lower::string_runtime().into_iter().map(|f| (f.c, f.ret)).collect());
    if let Some(ret) = strings.get(callee) {
        return match ret {
            K::N => Gives::Num,
            K::B => Gives::Bool,
            K::S => Gives::Str,
            K::L => Gives::Arr(Elem::Str),
            K::A => Gives::Arr(Elem::Num),
            K::I => Gives::Arr(Elem::Int),
        };
    }
    // Ints: their runtime's own table
    if let Some(f) = crate::lower::ints::rt_fn(callee) {
        use crate::lower::ints::IK;
        return match f.ret {
            IK::I => Gives::Int,
            IK::N => Gives::Num,
            IK::B => Gives::Bool,
            IK::S => Gives::Str,
            IK::A => Gives::Arr(Elem::Int),
        };
    }
    // Float32 and the packed arrays: their own table
    if let Some(f) = crate::lower::f32s::rt_fn(callee) {
        use crate::lower::f32s::FK;
        return match f.ret {
            FK::F => Gives::F32,
            FK::N => Gives::Num,
            FK::I => Gives::Int,
            FK::S => Gives::Str,
            FK::A => match callee {
                "viv_p_to_arr" => Gives::Arr(Elem::Num),
                "viv_p_to_iarr" => Gives::Arr(Elem::Int),
                // the kind is the last argument; the emitters ask the
                // lowering's type, so any packed kind does here
                _ => Gives::Arr(Elem::P(crate::ast::Store::F32)),
            },
        };
    }
    match callee {
        "viv_iget" => return Gives::Int,
        "viv_ivec" | "viv_ivec_tab" => return Gives::Arr(Elem::Int),
        "viv_iarr_str" => return Gives::Str,
        // an exported program's parameters (crate::export): a string or an
        // array is a copy on the run's arena
        "viv_export_str" => return Gives::Str,
        "viv_export_arr" => return Gives::Arr(Elem::Num),
        "viv_export_int" => return Gives::Int,
        _ => {}
    }
    // a map of strings (maps::kinded_fns): the name says what it gives
    if let Some(f) = crate::interp::maps::kinded(callee) {
        use crate::interp::maps::Slot;
        return match f.ret {
            Slot::N => Gives::Num,
            Slot::B => Gives::Bool,
            Slot::S => Gives::Str,
            Slot::L => Gives::Arr(Elem::Str),
            Slot::A => Gives::Arr(Elem::Num),
            Slot::M => Gives::Map(f.key, f.val),
        };
    }
    // the array vocabulary (interp/arrays.rs)
    if crate::interp::arrays::of_c_name(callee).is_some() {
        return if crate::interp::arrays::c_gives_array(callee) { Gives::Arr(Elem::Num) } else { Gives::Num };
    }
    match callee {
        "viv_histogram" | "viv_tridiag_solve" | "viv_roll" | "viv_shape" | "viv_ad_unb_aa" => Gives::Arr(Elem::Num),
        // what the IR's array nodes (`ArrLit`, `Zeros`, `Arr { kind }`,
        // `MapMath`) are emitted as: listed so that the table is the runtime's
        // whole answer, though the emitters decide those nodes by their kind
        "viv_vec" | "viv_vec_tab" | "viv_mat" | "viv_mat_tab" | "viv_zeros1" | "viv_zeros2" | "viv_zeros3" | "viv_zeros4" | "viv_arr_add"
        | "viv_arr_sub" | "viv_arr_had" | "viv_arr_scale" | "viv_arr_div" | "viv_arr_pows" | "viv_arr_max" | "viv_arr_min" | "viv_arr_maxs"
        | "viv_arr_mins" | "viv_arr_divsr" | "viv_arr_divs" | "viv_arr_adds" | "viv_arr_subs" | "viv_arr_subsr" | "viv_arr_map" | "viv_arr_chk"
        | "viv_matmul" | "viv_linspace" | "viv_transpose" => Gives::Arr(Elem::Num),
        "viv_czeros1" | "viv_czeros2" | "viv_czeros3" | "viv_czeros4" | "viv_cvec" | "viv_carr_add" | "viv_carr_sub" | "viv_carr_had"
        | "viv_carr_hadr" | "viv_carr_scale" | "viv_carr_neg" | "viv_carr_divs" | "viv_cmatmul" | "viv_ctranspose" => Gives::Arr(Elem::Cplx),
        "viv_where_aa" | "viv_where_as" | "viv_where_sa" | "viv_where_ss" => Gives::Arr(Elem::Num),
        "viv_all" | "viv_any" => Gives::Bool,
        "viv_ad_pop_a" | "viv_ad_zero" | "viv_ad_unb_na" | "viv_ad_cmp_na" | "viv_ad_cmp_an" | "viv_ad_cmp_aa" | "viv_ad_mm_a" | "viv_ad_mm_b"
        | "viv_ad_take" | "viv_ad_cat" | "viv_ad_keep" | "viv_ad_like" | "viv_ad_prod" | "viv_ad_prodax" | "viv_ad_takeax" => Gives::Arr(Elem::Num),
        "viv_ad_pop_b" => Gives::Bool,
        "viv_slab" | "viv_take" | "viv_getslice" | "viv_getslice_v" | "viv_copy" => Gives::Like(0),
        // crate::fuse: a fused tree, a reduction over one, a slice's bounds
        "viv_fz" => Gives::Arr(Elem::Num),
        "viv_fz_red" | "viv_slicecheck" => Gives::Num,
        "viv_cat" | "viv_cat1" => Gives::Like(1),
        "viv_seal" | "viv_append_n" | "viv_append_r" | "viv_extend" | "viv_grow_head" | "viv_grow_tail_a" => Gives::Arr(Elem::Num),
        "viv_map_new" | "viv_map_copy" => match NUM_MAP {
            Ty::Map(k, v) => Gives::Map(k, v),
            _ => unreachable!("NUM_MAP is a map"),
        },
        "viv_map_keys" | "viv_map_values" => Gives::Arr(Elem::Num),
        "viv_map_has" => Gives::Bool,
        "viv_carr_from" => Gives::Arr(Elem::Cplx),
        "viv_cplx" | "conj" | "cexp" | "csqrt" | "clog" | "csin" | "ccos" | "ctan" => Gives::Cplx,
        "viv_release_mode" => Gives::Bool,
        _ => Gives::Num,
    }
}

/// Whether the runtime call `callee` makes a new heap value (an array, a
/// string, a list of strings or a map), which a loop must free at the end of
/// the pass that made it.
pub fn runtime_allocates(callee: &str) -> bool {
    runtime_gives(callee).heap()
}

/// Whether a value of type `t` lives on the arena (and so is freed, or kept,
/// by the arena's rules): an array, a record (its arrays), a string, a list
/// of strings, or a map.
pub fn heap_ty(t: Ty) -> bool {
    matches!(t, Ty::Arr(_) | Ty::Record(_) | Ty::Recs(_) | Ty::Str | Ty::Map(..) | Ty::Enum(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every C function of the runtime, with the type it returns: a
    /// definition `[qualifiers] static [inline] TYPE NAME(` at the start of a
    /// line.
    fn prototypes() -> Vec<(String, String)> {
        let text = crate::lower::runtime_c_text();
        let mut out = Vec::new();
        for line in text.lines() {
            let Some(open) = line.find('(') else { continue };
            let head = &line[..open];
            if !(head.contains("static ") || head.starts_with("VIV_ACCESS ")) || head.contains('=') || head.contains("#define") {
                continue;
            }
            let words: Vec<&str> = head.split_whitespace().collect();
            let Some(name) = words.last() else { continue };
            let name = name.trim_start_matches('*');
            if !name.starts_with("viv_") {
                continue;
            }
            let ret = words[..words.len() - 1].iter().filter(|w| !matches!(**w, "static" | "inline" | "VIV_ACCESS" | "VIV_SCALAR_LOOPS" | "VIV_SL_UNUSED" | "VIV_S_UNUSED" | "VIV_AR_UNUSED" | "VIV_COLD" | "const")).copied().collect::<Vec<_>>().join(" ");
            let ret = if words.last().is_some_and(|w| w.starts_with('*')) { format!("{ret}*") } else { ret };
            out.push((name.to_string(), ret));
        }
        out
    }

    /// The runtime's types that hold a heap value: an array, a string, a list
    /// of strings, a map, a registry.
    fn heap_c(ret: &str) -> bool {
        // (a registry, `viv_fnarr`, is only ever a `FnArrLit` node's)
        matches!(ret.replace(' ', "").as_str(), "viv_arr" | "viv_str" | "viv_strs" | "viv_map*" | "viv_smap*")
    }

    /// The runtime calls a `Viv::Call` node can name: every quoted `viv_...`
    /// name in the emitters' sources, and the registries' C names (strings,
    /// maps of strings, the array vocabulary at each arity it has).
    fn callees() -> std::collections::HashSet<String> {
        let mut out = std::collections::HashSet::new();
        for src in [include_str!("lower.rs"), include_str!("lower/strings.rs"), include_str!("lower/ints.rs"), include_str!("wasm_backend.rs"), include_str!("grad.rs")] {
            for (k, _) in src.match_indices("\"viv_") {
                let rest = &src[k + 1..];
                let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len());
                if rest[end..].starts_with('"') {
                    out.insert(rest[..end].to_string());
                }
            }
        }
        out.extend(crate::lower::string_runtime().into_iter().map(|f| f.c.to_string()));
        out.extend(crate::interp::maps::kinded_fns().iter().map(|f| f.name.clone()));
        // a map of numbers: `maps::callee` writes the name (`viv_map_keys`)
        for f in crate::interp::maps::kinded_fns() {
            out.insert(crate::interp::maps::callee(f.op, Mk::Num, Mk::Num));
        }
        for s in crate::interp::arrays::SIGS {
            out.insert(crate::interp::arrays::c_name(s.name, s.params.len(), false));
            out.insert(crate::interp::arrays::c_name(s.name, s.params.len(), true));
        }
        out
    }

    #[test]
    fn every_runtime_call_that_returns_a_heap_value_allocates() {
        let protos = prototypes();
        assert!(protos.len() > 200, "the runtime has its functions: {}", protos.len());
        let callable = callees();
        let mut wrong = Vec::new();
        let mut seen = 0;
        for (name, ret) in &protos {
            if !callable.contains(name) {
                continue;
            }
            seen += 1;
            if heap_c(ret) && !runtime_allocates(name) {
                wrong.push(format!("{name} returns {ret} and the table says it does not allocate"));
            }
            if matches!(ret.as_str(), "double" | "int" | "void" | "long") && runtime_allocates(name) {
                wrong.push(format!("{name} returns {ret} and the table says it allocates"));
            }
        }
        assert!(seen > 150, "the emitters call {seen} runtime functions");
        // the two the C emitter's own list had missed are among them
        assert!(callable.contains("viv_map_keys") && callable.contains("viv_map_values"));
        assert!(protos.iter().any(|(n, r)| n == "viv_map_keys" && r == "viv_arr"), "the scan reads the maps runtime");
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    fn keys_and_values_of_a_number_map_allocate() {
        // the leak study/memcheck/scale.tsv measured: 128 bytes a call
        assert!(runtime_allocates("viv_map_keys") && runtime_allocates("viv_map_values"));
        assert!(runtime_allocates("viv_map_new") && runtime_allocates("viv_map_copy"));
        assert!(!runtime_allocates("viv_map_len") && !runtime_allocates("viv_map_get"));
    }

    /// The builtins the interpreter dispatches by name in `interp.rs`.
    fn interp_names() -> Vec<String> {
        let src = include_str!("interp.rs");
        let mut out = Vec::new();
        for (k, _) in src.match_indices("(\"") {
            let rest = &src[k + 2..];
            let Some(end) = rest.find('"') else { continue };
            let name = &rest[..end];
            let after = &rest[end + 1..];
            if name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') && !name.is_empty() && after.starts_with(", ") && after[2..].starts_with(|c: char| c.is_ascii_digit() || c == '_') {
                out.push(name.to_string());
            }
        }
        for (k, _) in src.match_indices("callee == \"") {
            let rest = &src[k + 11..];
            if let Some(end) = rest.find('"') {
                out.push(rest[..end].to_string());
            }
        }
        out.retain(|n| !crate::hints::REFUSED_NAMES.contains(&n.as_str()));
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn the_table_covers_every_builtin_the_interpreter_dispatches() {
        let names = interp_names();
        assert!(names.len() > 80, "found {} names", names.len());
        let missing: Vec<&String> = names.iter().filter(|n| kind(n).is_none()).collect();
        assert!(missing.is_empty(), "builtins with no entry in crate::builtins: {missing:?}");
        // and the registries' own builtins
        for b in crate::interp::strings::BUILTINS {
            assert!(kind(b.name).is_some(), "{}", b.name);
        }
        for s in crate::interp::arrays::SIGS {
            assert!(kind(s.name).is_some(), "{}", s.name);
        }
        for n in crate::interp::maps::NAMES {
            assert!(kind(n).is_some(), "{n}");
        }
    }

    #[test]
    fn the_array_vocabulary_and_the_table_agree_on_what_is_written() {
        for s in crate::interp::arrays::SIGS {
            if let Some(first) = s.writes {
                let r = writes(s.name).unwrap_or_else(|| panic!("{} writes an argument and WRITERS does not list it", s.name));
                assert!(r.contains(&first), "{}: the vocabulary writes from argument {first}, the table {r:?}", s.name);
            }
        }
    }

    #[test]
    fn graph_and_grad_read_the_same_answers() {
        // the program graph's write edges
        let src = "A = zeros(3, 3)\nb = zeros(3)\nx = zeros(3)\nwr = zeros(3)\nwi = zeros(3)\nlin_solve(A, b, x)\neig_near(A, 1.0, 2, wr, wi)\nz = czeros(4)\nifft(z)\n";
        let program = crate::parse(src, knotkit::span::SourceId(0)).unwrap();
        let cx = crate::graph::Cx::of(&program);
        let mut eff = crate::graph::Eff::default();
        // the calls only (an assignment writes its name)
        for item in &program.items {
            if let crate::ast::Item::Stmt(s @ crate::ast::Stmt::ExprStmt { .. }) = item {
                crate::graph::stmt_eff(s, &cx, &mut eff);
            }
        }
        for w in ["x", "wr", "wi", "z"] {
            assert!(eff.writes.iter().any(|n| n == w), "graph: {w} written, {:?}", eff.writes);
        }
        for r in ["b", "A"] {
            assert!(!eff.writes.iter().any(|n| n == r), "graph: {r} is only read, {:?}", eff.writes);
        }
        // grad's view is the same function
        for (name, a, b) in WRITERS {
            assert_eq!(crate::grad::builtin_writes(name), Some(*a..=*b));
        }
        assert_eq!(written_args("lin_solve", 2), Vec::<usize>::new());
        assert_eq!(written_args("lin_solve", 3), vec![2]);
        assert_eq!(written_args("eig_near", 5), vec![3, 4]);
    }
}
