// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The direct WebAssembly backend: Viv IR -> a wasm module, with no C compiler
//! and no LLVM in between.
//!
//! It lowers from the same typed IR the C backend does (`crate::lower`), so the
//! two compiled engines share one front end: types, records, function values
//! and registries, complex numbers, contracts, `test` blocks and the debug lens
//! all arrive already checked. What a module computes itself is what a few
//! wasm instructions express -- arithmetic, comparisons, loops, calls,
//! function values (`call_indirect`), element access with bounds checks,
//! records and complex scalars. Everything else is a builtin, and a builtin is
//! a call into the runtime (`crate::interp::wasm_rt`, inside viv.wasm), which
//! runs the interpreter's own code for it. That is how the output of a compiled
//! module is byte-identical to the interpreter's: the text, the order of every
//! sum a builtin takes, and the words an error stops with come from one place.
//!
//! The module imports:
//!
//! - `rt.memory`, viv.wasm's linear memory, where arrays and records live;
//! - `rt.vrt_*`, the runtime (the table `RT` below; every module imports all
//!   of it, so the function indices are fixed);
//! - `host.flush`, called after each top-level statement so the host can show
//!   the output as it grows.
//!
//! It exports `run`, which executes the program.
//!
//! Value representation. A number or boolean is an `f64` (a boolean is 1.0 or
//! 0.0, as in C); a complex number is two `f64`s, real part first; an array,
//! record or registry is an `i32` pointer; a function value is an `i32` slot
//! in the module's table.

use std::collections::HashMap;

use crate::ast::BinOp;
use crate::lower::{ArgKind, ArrKind, Elem, FnSig, PrintPart, Ty, Viv};
use knotkit::ir::{IrProgram, Node, NodeId};

// ---- the runtime's imports ------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VT {
    I32,
    F64,
    /// an Int
    I64,
    /// a Float32
    F32,
}
use VT::{F32, F64, I32, I64};

impl VT {
    fn byte(self) -> u8 {
        match self {
            I32 => 0x7f,
            F64 => 0x7c,
            I64 => 0x7e,
            F32 => 0x7d,
        }
    }
}

/// Every runtime function a module may call: name, parameters, results. The
/// position is the import's function index.
const RT: &[(&str, &[VT], &[VT])] = &[
    ("vrt_init", &[I32], &[]),
    ("vrt_static", &[I32], &[I32]),
    ("vrt_block", &[I32], &[I32]),
    ("vrt_fnarr", &[I32], &[I32]),
    ("vrt_lit", &[I32, I32, I32], &[I32]),
    ("vrt_zeros", &[I32, F64, F64, F64, F64], &[I32]),
    ("vrt_czeros", &[I32, F64, F64, F64, F64], &[I32]),
    ("vrt_mark", &[], &[I32]),
    ("vrt_release", &[I32, I32], &[]),
    ("vrt_scratch", &[], &[I32]),
    ("vrt_oob", &[F64, I32, I32], &[]),
    ("vrt_rank", &[I32, I32], &[]),
    ("vrt_fnoob", &[F64, I32], &[]),
    ("vrt_placeholder", &[I32, I32], &[]),
    ("vrt_stop", &[I32, I32], &[]),
    ("vrt_exp", &[F64], &[F64]),
    ("vrt_ln", &[F64], &[F64]),
    ("vrt_sin", &[F64], &[F64]),
    ("vrt_cos", &[F64], &[F64]),
    ("vrt_tan", &[F64], &[F64]),
    ("vrt_asin", &[F64], &[F64]),
    ("vrt_acos", &[F64], &[F64]),
    ("vrt_atan", &[F64], &[F64]),
    ("vrt_sinh", &[F64], &[F64]),
    ("vrt_cosh", &[F64], &[F64]),
    ("vrt_tanh", &[F64], &[F64]),
    ("vrt_round", &[F64], &[F64]),
    ("vrt_erf", &[F64], &[F64]),
    ("vrt_gamma", &[F64], &[F64]),
    ("vrt_bnot", &[F64], &[F64]),
    ("vrt_u32", &[F64], &[F64]),
    ("vrt_i32", &[F64], &[F64]),
    ("vrt_pow", &[F64, F64], &[F64]),
    ("vrt_atan2", &[F64, F64], &[F64]),
    ("vrt_fma", &[F64, F64, F64], &[F64]),
    ("vrt_hypot", &[F64, F64], &[F64]),
    ("vrt_fmod", &[F64, F64], &[F64]),
    ("vrt_fmin", &[F64, F64], &[F64]),
    ("vrt_fmax", &[F64, F64], &[F64]),
    ("vrt_band", &[F64, F64], &[F64]),
    ("vrt_bor", &[F64, F64], &[F64]),
    ("vrt_bxor", &[F64, F64], &[F64]),
    ("vrt_shl", &[F64, F64], &[F64]),
    ("vrt_shr", &[F64, F64], &[F64]),
    ("vrt_sar", &[F64, F64], &[F64]),
    ("vrt_floormod", &[F64, F64], &[F64]),
    ("vrt_floordiv", &[F64, F64], &[F64]),
    ("vrt_cfn", &[I32, F64, F64], &[]),
    ("vrt_cdiv", &[F64, F64, F64, F64], &[]),
    ("vrt_print", &[F64], &[]),
    ("vrt_show_num", &[I32, I32, F64], &[]),
    ("vrt_show_bool", &[I32, I32, F64], &[]),
    ("vrt_show_cplx", &[I32, I32, F64, F64], &[]),
    ("vrt_show_arr", &[I32, I32, I32], &[]),
    ("vrt_narrate", &[I32, I32], &[]),
    ("vrt_test_begin", &[I32, I32], &[]),
    ("vrt_test_end", &[], &[]),
    ("vrt_step", &[I32, I32, I32, I32, I32, I32], &[]),
    ("vrt_steps_flush", &[], &[]),
    ("vrt_story_mark", &[I32, I32, I32, I32], &[]),
    ("vrt_story_iter", &[I32, I32, F64], &[]),
    ("vrt_story_watch_begin", &[], &[]),
    ("vrt_story_w_num", &[I32, I32, F64], &[]),
    ("vrt_story_w_bool", &[I32, I32, F64], &[]),
    ("vrt_story_w_arr", &[I32, I32, I32], &[]),
    ("vrt_story_watch_end", &[I32, I32, I32, I32], &[]),
    ("vrt_story_maxabs", &[I32], &[F64]),
    ("vrt_story_hold", &[I32, I32, I32, I32, I32, I32, I32, I32, F64], &[]),
    ("vrt_story_flush", &[], &[]),
    ("vrt_arg_num", &[F64], &[]),
    ("vrt_arg_bool", &[F64], &[]),
    ("vrt_arg_cplx", &[F64, F64], &[]),
    ("vrt_arg_arr", &[I32], &[]),
    ("vrt_arg_str", &[I32, I32], &[]),
    ("vrt_call_f", &[I32, I32], &[F64]),
    ("vrt_call_a", &[I32, I32], &[I32]),
    ("vrt_call_c", &[I32, I32], &[]),
    ("vrt_slab", &[I32, F64], &[I32]),
    ("vrt_setslab", &[I32, F64, I32], &[F64]),
    ("vrt_take", &[I32, I32], &[I32]),
    ("vrt_put", &[I32, I32, I32], &[F64]),
    ("vrt_getslice", &[I32, I32], &[I32]),
    ("vrt_setslice", &[I32, I32, I32], &[F64]),
    ("vrt_setslice_n", &[I32, I32, F64], &[F64]),
    ("vrt_setslice_c", &[I32, I32, F64, F64], &[F64]),
    ("vrt_copy", &[I32], &[I32]),
    ("vrt_seal", &[I32], &[I32]),
    // maps (interp/maps.rs): a map is an i32 naming the runtime's map
    ("vrt_map_new", &[], &[I32]),
    ("vrt_map_set", &[I32, F64, F64], &[F64]),
    ("vrt_map_get", &[I32, F64], &[F64]),
    ("vrt_map_has", &[I32, F64], &[F64]),
    ("vrt_map_getd", &[I32, F64, F64], &[F64]),
    ("vrt_map_remove", &[I32, F64], &[F64]),
    ("vrt_map_removed", &[I32, F64, F64], &[F64]),
    ("vrt_map_len", &[I32], &[F64]),
    ("vrt_map_keys", &[I32], &[I32]),
    ("vrt_map_values", &[I32], &[I32]),
    ("vrt_map_copy", &[I32], &[I32]),
    ("vrt_map_clear", &[I32], &[F64]),
    ("vrt_arg_map", &[I32], &[]),
    ("vrt_append_n", &[I32, F64, F64], &[I32]),
    ("vrt_append_r", &[I32, I32, F64], &[I32]),
    ("vrt_extend", &[I32, I32, F64], &[I32]),
    ("vrt_cat", &[F64, I32, I32], &[I32]),
    ("vrt_cat1", &[F64, I32], &[I32]),
    ("vrt_shape", &[I32], &[I32]),
    ("vrt_cmap", &[I32, I32], &[I32]),
    ("vrt_map", &[I32, I32], &[I32]),
    ("vrt_sum", &[I32], &[F64]),
    ("vrt_extreme", &[I32, I32], &[F64]),
    ("vrt_fill", &[I32, F64, F64], &[F64]),
    ("vrt_transpose", &[I32], &[I32]),
    ("vrt_cneg", &[I32], &[I32]),
    ("vrt_roll", &[I32, F64, F64, I32], &[I32]),
    ("vrt_linspace", &[F64, F64, F64], &[I32]),
    ("vrt_carr_from", &[I32, I32], &[I32]),
    ("vrt_arith", &[I32, I32, I32], &[I32]),
    ("vrt_arith_s", &[I32, I32, F64, I32], &[I32]),
    ("vrt_matmul", &[I32, I32], &[I32]),
    ("vrt_carith", &[I32, I32, I32], &[I32]),
    ("vrt_carith_s", &[I32, I32, F64, F64, I32], &[I32]),
    ("vrt_cmatmul", &[I32, I32], &[I32]),
    ("vrt_minmax", &[I32, I32, I32], &[I32]),
    ("vrt_minmax_s", &[I32, I32, F64], &[I32]),
    ("vrt_pows", &[I32, F64], &[I32]),
    ("vrt_keep", &[I32], &[]),
    // `disjoint` pins (crate::memory_lens): the record a pinned parallel for
    // keeps of which pass wrote each element
    ("vrt_dj_begin", &[I32, I32], &[]),
    ("vrt_dj_track", &[I32, I32, I32], &[]),
    ("vrt_dj_pass", &[F64], &[]),
    ("vrt_dj_note", &[I32], &[]),
    ("vrt_dj_lo", &[], &[I32]),
    ("vrt_dj_hi", &[], &[I32]),
    ("vrt_dj_end", &[], &[]),
    ("vrt_release_kept", &[I32], &[]),
    // `append(xs, r)` of an array of records
    ("vrt_recs_push", &[I32, I32], &[I32]),
    ("vrt_nonint", &[F64], &[]),
    // enums: a value's text, and its nodes kept, read
    // through the enum's descriptor (string table base, descriptor, value)
    ("vrt_en_str", &[I32, I32, I32], &[I32]),
    ("vrt_keep_en", &[I32, I32, I32, I32], &[]),
    ("vrt_en_new", &[I32], &[I32]),
    // fused trees and reductions over views (crate::fuse)
    ("vrt_slicecheck", &[I32, I32], &[F64]),
    ("vrt_fz_a", &[I32], &[]),
    ("vrt_fz_n", &[F64], &[]),
    ("vrt_fz", &[I32], &[I32]),
    ("vrt_fz_red", &[I32, I32], &[F64]),
];

type Import = (&'static str, &'static [VT], &'static [VT]);

/// `RT`, then the imports of strings (`interp/wasm_str.rs`),
/// read from the lowering's list of their runtime calls: `viv_s_x` is
/// imported as `vrt_s_x`, a string or a list as an i32, a number as an f64.
fn rt_table() -> &'static [Import] {
    static TABLE: std::sync::OnceLock<Vec<Import>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        use crate::interp::strings::K;
        let vt = |k: &K| if matches!(k, K::S | K::L | K::A | K::I) { I32 } else { F64 };
        let leak = |v: Vec<VT>| -> &'static [VT] { Box::leak(v.into_boxed_slice()) };
        let mut t: Vec<Import> = RT.to_vec();
        for f in crate::lower::string_runtime() {
            let name: &'static str = Box::leak(format!("vrt_{}", &f.c[4..]).into_boxed_str());
            t.push((name, leak(f.args.iter().map(vt).collect()), leak(vec![vt(&f.ret)])));
        }
        // Ints: `viv_i_x` is `vrt_i_x`, an Int an i64
        for f in crate::lower::ints::RUNTIME {
            use crate::lower::ints::IK;
            let vt = |k: &IK| match k {
                IK::I => I64,
                IK::N | IK::B => F64,
                IK::S | IK::A => I32,
            };
            let name: &'static str = Box::leak(f.c.replacen("viv_", "vrt_", 1).into_boxed_str());
            t.push((name, leak(f.args.iter().map(vt).collect()), leak(vec![vt(&f.ret)])));
        }
        // Float32 and the packed arrays: `viv_f_x` is
        // `vrt_f_x`, a Float32 an f32; the arithmetic is an instruction
        for f in crate::lower::f32s::RUNTIME.iter().filter(|f| crate::lower::f32s::wasm_op(f.c).is_none()) {
            use crate::lower::f32s::FK;
            let vt = |k: &FK| match k {
                FK::F => F32,
                FK::N => F64,
                FK::I => I64,
                FK::S | FK::A => I32,
            };
            let name: &'static str = Box::leak(f.c.replacen("viv_", "vrt_", 1).into_boxed_str());
            t.push((name, leak(f.args.iter().map(vt).collect()), leak(vec![vt(&f.ret)])));
        }
        // maps of strings (maps::kinded_fns): `viv_map_x_sn` is `vrt_map_x_sn`
        for f in crate::interp::maps::kinded_fns() {
            use crate::interp::maps::Slot;
            let vt = |x: &Slot| if matches!(x, Slot::N | Slot::B) { F64 } else { I32 };
            let name: &'static str = Box::leak(f.name.replacen("viv_", "vrt_", 1).into_boxed_str());
            t.push((name, leak(f.args.iter().map(vt).collect()), leak(vec![vt(&f.ret)])));
        }
        t.extend_from_slice(&[
            ("vrt_s_push", &[I32][..], &[][..]),
            ("vrt_s_catn", &[], &[I32]),
            ("vrt_ss_of", &[], &[I32]),
            ("vrt_s_str_cplx", &[F64, F64], &[I32]),
            ("vrt_s_fmt_arr", &[I32, F64, F64, F64], &[I32]),
            ("vrt_keep_str", &[I32], &[]),
            ("vrt_keep_strs", &[I32], &[]),
            ("vrt_arg_s", &[I32], &[]),
        ]);
        t
    })
}

/// The names of every runtime function a module may import -- what a host
/// must find among viv.wasm's exports.
pub fn runtime_imports() -> Vec<&'static str> {
    rt_table().iter().map(|r| r.0).collect()
}

thread_local! {
    /// The runtime functions this module imports, in import order. A module
    /// imports only what it calls: `emit_wasm` compiles twice, first with
    /// all of `RT` imported to learn which are called (`USED`), then with
    /// just those. (A small program's module is then 0.4 KB, not 2 KB, most
    /// of which was import names.)
    static IMPORTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
    static USED: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The function index of runtime import `name`, noting that it is used.
fn rt(name: &str) -> u32 {
    let i = IMPORTS.with(|im| im.borrow().iter().position(|n| *n == name));
    let Some(i) = i else { panic!("runtime import `{name}` not imported") };
    let name = rt_table().iter().find(|r| r.0 == name).map(|r| r.0).expect("a runtime import");
    USED.with(|u| {
        let mut u = u.borrow_mut();
        if !u.contains(&name) {
            u.push(name);
        }
    });
    i as u32
}

/// The host's one import, after the runtime's.
fn host_flush() -> u32 {
    IMPORTS.with(|im| im.borrow().len() as u32)
}

thread_local! {
    /// The program calls `clock()` (a `#! timing` program): the module also
    /// imports `host.clock`, after `host.flush`.
    static HOST_CLOCK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `host.clock`, the host's second import, in a module that reads the clock.
fn host_clock() -> u32 {
    host_flush() + 1
}

/// Imported functions in all; a module's own functions are numbered after them.
fn n_imports() -> u32 {
    host_flush() + 1 + HOST_CLOCK.with(|c| c.get()) as u32
}

/// Array header offsets (see `interp::wasm_rt`).
const H_SIZE: u32 = 4;
const H_RANK: u32 = 8;
const H_DIMS: u32 = 12;
const H_DATA: u32 = 32;

// ---- byte-level helpers ------------------------------------------------------

fn uleb(mut v: u64, o: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            o.push(b | 0x80)
        } else {
            o.push(b);
            break;
        }
    }
}
fn sleb(mut v: i64, o: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let sign = b & 0x40;
        if (v == 0 && sign == 0) || (v == -1 && sign != 0) {
            o.push(b);
            break;
        }
        o.push(b | 0x80);
    }
}
fn wname(s: &str, o: &mut Vec<u8>) {
    uleb(s.len() as u64, o);
    o.extend_from_slice(s.as_bytes());
}
fn section(id: u8, body: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    uleb(body.len() as u64, out);
    out.extend_from_slice(body);
}

// ---- types -------------------------------------------------------------------

/// How a Viv type is carried in wasm values.
fn parts(t: Ty) -> Vec<VT> {
    match t {
        Ty::Num | Ty::Bool => vec![F64],
        Ty::Int => vec![I64],
        Ty::F32 => vec![F32],
        Ty::Cplx => vec![F64, F64],
        Ty::Arr(_) | Ty::Func(_) | Ty::Record(_) | Ty::Recs(_) | Ty::Str | Ty::Map(..) | Ty::Enum(_) => vec![I32],
        Ty::Void => vec![],
    }
}

fn sig_ret(sig: FnSig) -> Ty {
    match sig.ret {
        ArgKind::Arr => Ty::Arr(Elem::Num),
        _ => Ty::Num,
    }
}

fn kind_ty(k: ArgKind) -> Ty {
    match k {
        ArgKind::Num => Ty::Num,
        ArgKind::Fn => Ty::Func(FnSig::plain(0)),
        ArgKind::Arr => Ty::Arr(Elem::Num),
    }
}

/// The wasm signature a function VALUE of `sig` is called through.
fn sig_type(sig: FnSig) -> (Vec<VT>, Vec<VT>) {
    let params = (0..sig.arity as usize).flat_map(|j| parts(kind_ty(sig.kind(j)))).collect();
    (params, parts(sig_ret(sig)))
}

#[derive(Default)]
struct Types {
    list: Vec<(Vec<VT>, Vec<VT>)>,
}
impl Types {
    fn idx(&mut self, params: Vec<VT>, results: Vec<VT>) -> u32 {
        let key = (params, results);
        if let Some(k) = self.list.iter().position(|t| *t == key) {
            return k as u32;
        }
        self.list.push(key);
        (self.list.len() - 1) as u32
    }
}

/// One record type: its fields in declaration order, each with its byte offset
/// in the record's block (8 bytes a slot, 16 for a complex number).
#[derive(Clone)]
struct Layout {
    fields: Vec<(String, Ty, u32)>,
    bytes: u32,
}

/// What the program declares: functions, globals, records, the table.
struct Unit {
    /// user defs: name -> (function index, params, return type)
    funcs: HashMap<String, (u32, Vec<(String, Ty)>, Ty)>,
    /// lifted lambdas: name -> (function index, param, captures)
    lambdas: HashMap<String, (u32, String, Vec<(String, Ty)>)>,
    /// globals: name -> (first wasm global index, type)
    globals: HashMap<String, (u32, Ty)>,
    records: HashMap<String, Layout>,
    /// enums: each one's variants, and its descriptor's
    /// offset in the string table
    enums: HashMap<String, (Vec<(String, Vec<Ty>)>, u32)>,
    /// def name -> its table slot, for every def used as a value
    slots: HashMap<String, u32>,
    /// the table: function indices of the wrappers, by slot
    table: Vec<u32>,
    /// generated helpers (simpson/trapezoid over a lambda): key -> function index
    helpers: HashMap<String, u32>,
    /// the bodies of generated helpers, emitted after everything else
    helper_code: Vec<(u32, u32, Vec<u8>)>,
    /// every function's type index, by function index - n_imports()
    fn_types: Vec<u32>,
    types: Types,
    strings: Vec<u8>,
    string_at: HashMap<String, (u32, u32)>,
    /// string literals as string objects in the table:
    /// the text -> the object's offset
    str_objs: HashMap<String, u32>,
    /// the bytes of every constant array literal, the module's data segment 1
    consts: Vec<u8>,
    story: bool,
    /// the defs a pinned `parallel for` reaches (`lower::dj_reached`): their
    /// element writes are recorded
    dj_defs: std::collections::HashSet<String>,
    /// how many pinned `parallel for` bodies the emitter is inside
    dj_body: usize,
    /// in a program with a pinned loop: two globals holding the address
    /// range of its pinned arrays while it runs (empty otherwise), so a write
    /// in the loop's reach calls the runtime only when it lands in that range
    dj_range: Option<(u32, u32)>,
    /// the pinned names of the pinned loops being emitted
    dj_pins: Vec<Vec<String>>,
    /// the i64 global that holds the recursion counter (crate::recursion),
    /// in a program with a recursive def
    depth_g: Option<u32>,
    /// per def, the parameters a pinned array can be passed to
    dj_pinnable: HashMap<String, std::collections::HashSet<String>>,
}

/// Global 0 holds the string table's address, global 1 the scratch pair's.
const G_STR: u32 = 0;
const G_SCRATCH: u32 = 1;

impl Unit {
    fn string(&mut self, s: &str) -> (u32, u32) {
        if let Some(&r) = self.string_at.get(s) {
            return r;
        }
        let r = (self.strings.len() as u32, s.len() as u32);
        self.strings.extend_from_slice(s.as_bytes());
        self.string_at.insert(s.to_string(), r);
        r
    }
    /// A string literal's object in the table (interp/wasm_str.rs's layout:
    /// len, nch, owner 0, the index cache, the bytes and a NUL).
    fn str_obj(&mut self, s: &str) -> u32 {
        if let Some(&o) = self.str_objs.get(s) {
            return o;
        }
        while self.strings.len() % 4 != 0 {
            self.strings.push(0);
        }
        let o = self.strings.len() as u32;
        let nch = crate::interp::strings::count_chars(s.as_bytes()) as u32;
        for w in [s.len() as u32, nch, 0, 0, 0] {
            self.strings.extend_from_slice(&w.to_le_bytes());
        }
        self.strings.extend_from_slice(s.as_bytes());
        self.strings.push(0);
        self.str_objs.insert(s.to_string(), o);
        o
    }
    fn new_fn(&mut self, params: Vec<VT>, results: Vec<VT>) -> u32 {
        let t = self.types.idx(params, results);
        self.fn_types.push(t);
        n_imports() + self.fn_types.len() as u32 - 1
    }
}

// ---- one function's code ----------------------------------------------------

/// A loop's labels, as structured-control depths: `brk` is the block a
/// `break` leaves, `cont` the block a `continue` leaves (the end of the body).
struct LoopLabels {
    brk: u32,
    cont: u32,
}

struct Fx {
    code: Vec<u8>,
    /// wasm types of every local after the parameters
    locals: Vec<VT>,
    n_param_slots: u32,
    /// a name -> its first local index and type
    names: HashMap<String, (u32, Ty)>,
    /// scratch locals not in use, by type
    free: HashMap<VT, Vec<u32>>,
    depth: u32,
    loops: Vec<LoopLabels>,
    ret: Ty,
    /// the arena mark's local, when the function can allocate
    mark: Option<u32>,
    /// the def the story records returns for, when it does
    story_def: Option<String>,
    /// the def step events are recorded under (`src`, or `<main>`)
    step_def: String,
    /// Array variables indexed in this function -> the first of five i32
    /// locals holding a copy of the array's rank and four dimensions. A
    /// header never changes once made, so the copy stays right until the
    /// variable is assigned, and it is refreshed there: element access then
    /// reads no header from memory. See `Em::cache_headers`.
    hdr: HashMap<String, u32>,
    /// a def a pinned `parallel for` reaches: its element writes are recorded
    dj: bool,
    /// its variables that only hold arrays it made (`lower::dj_fresh`): their
    /// writes are not recorded
    dj_fresh: std::collections::HashSet<String>,
    /// its parameters, and those a pinned array can be passed to
    dj_params: std::collections::HashSet<String>,
    dj_pinnable: std::collections::HashSet<String>,
    /// the i64 counters of the `for` loops being emitted, by their names in
    /// lower/intidx.rs (`viv_k_<var>`), innermost last
    ctrs: Vec<(String, u32)>,
}

impl Fx {
    fn new(params: &[(String, Ty)], ret: Ty, step_def: &str) -> Fx {
        let mut names = HashMap::new();
        let mut slot = 0u32;
        for (n, t) in params {
            names.insert(n.clone(), (slot, *t));
            slot += parts(*t).len() as u32;
        }
        Fx {
            code: Vec::new(),
            locals: Vec::new(),
            n_param_slots: slot,
            names,
            free: HashMap::new(),
            depth: 0,
            loops: Vec::new(),
            ret,
            mark: None,
            story_def: None,
            step_def: step_def.to_string(),
            hdr: HashMap::new(),
            dj: false,
            dj_fresh: Default::default(),
            dj_params: Default::default(),
            dj_pinnable: Default::default(),
            ctrs: Vec::new(),
        }
    }
    fn local(&mut self, vt: VT) -> u32 {
        self.locals.push(vt);
        self.n_param_slots + self.locals.len() as u32 - 1
    }
    fn declare(&mut self, name: &str, t: Ty) {
        if self.names.contains_key(name) {
            return;
        }
        let ps = parts(t);
        let first = self.n_param_slots + self.locals.len() as u32;
        for vt in ps {
            self.local(vt);
        }
        self.names.insert(name.to_string(), (first, t));
    }
    fn tmp(&mut self, vt: VT) -> u32 {
        if let Some(i) = self.free.get_mut(&vt).and_then(|v| v.pop()) {
            return i;
        }
        self.local(vt)
    }
    fn release(&mut self, vt: VT, i: u32) {
        self.free.entry(vt).or_default().push(i);
    }
    fn op(&mut self, b: u8) {
        self.code.push(b);
    }
    fn get(&mut self, i: u32) {
        self.code.push(0x20);
        uleb(i as u64, &mut self.code);
    }
    fn set(&mut self, i: u32) {
        self.code.push(0x21);
        uleb(i as u64, &mut self.code);
    }
    fn tee(&mut self, i: u32) {
        self.code.push(0x22);
        uleb(i as u64, &mut self.code);
    }
    fn gget(&mut self, i: u32) {
        self.code.push(0x23);
        uleb(i as u64, &mut self.code);
    }
    fn gset(&mut self, i: u32) {
        self.code.push(0x24);
        uleb(i as u64, &mut self.code);
    }
    fn call(&mut self, f: u32) {
        self.code.push(0x10);
        uleb(f as u64, &mut self.code);
    }
    fn rt(&mut self, name: &str) {
        self.call(rt(name));
    }
    fn f64c(&mut self, v: f64) {
        self.code.push(0x44);
        self.code.extend_from_slice(&v.to_le_bytes());
    }
    fn i32c(&mut self, v: i64) {
        self.code.push(0x41);
        sleb(v, &mut self.code);
    }
    fn i64c(&mut self, v: i64) {
        self.code.push(0x42);
        sleb(v, &mut self.code);
    }
    fn load_i64(&mut self, off: u32) {
        self.code.push(0x29);
        self.code.push(3);
        uleb(off as u64, &mut self.code);
    }
    fn store_i64(&mut self, off: u32) {
        self.code.push(0x37);
        self.code.push(3);
        uleb(off as u64, &mut self.code);
    }
    fn load_i32(&mut self, off: u32) {
        self.code.push(0x28);
        self.code.push(2);
        uleb(off as u64, &mut self.code);
    }
    fn load_f64(&mut self, off: u32) {
        self.code.push(0x2b);
        self.code.push(3);
        uleb(off as u64, &mut self.code);
    }
    fn store_i32(&mut self, off: u32) {
        self.code.push(0x36);
        self.code.push(2);
        uleb(off as u64, &mut self.code);
    }
    fn store_f64(&mut self, off: u32) {
        self.code.push(0x39);
        self.code.push(3);
        uleb(off as u64, &mut self.code);
    }
    /// A Float32 in an 8-byte slot: its 4 bytes first.
    fn store_f32(&mut self, off: u32) {
        self.code.push(0x38);
        self.code.push(2);
        uleb(off as u64, &mut self.code);
    }
    fn load_f32(&mut self, off: u32) {
        self.code.push(0x2a);
        self.code.push(2);
        uleb(off as u64, &mut self.code);
    }
    fn f32c(&mut self, v: f32) {
        self.code.push(0x43);
        self.code.extend_from_slice(&v.to_le_bytes());
    }
    /// Open a structured block (`block`, `loop` or `if`) with no result.
    fn open(&mut self, opcode: u8) -> u32 {
        self.code.push(opcode);
        self.code.push(0x40);
        self.depth += 1;
        self.depth
    }
    fn end(&mut self) {
        self.code.push(0x0b);
        self.depth -= 1;
    }
    fn br(&mut self, level: u32) {
        self.code.push(0x0c);
        uleb((self.depth - level) as u64, &mut self.code);
    }
    fn br_if(&mut self, level: u32) {
        self.code.push(0x0d);
        uleb((self.depth - level) as u64, &mut self.code);
    }
    /// `if <i32 on the stack> { <call>; unreachable }`: stop the program with a
    /// runtime error. The runtime traps first; `unreachable` tells wasm the
    /// path does not go on.
    /// Stop the program unless the index in local `x` is exactly an integer
    /// (`x - floor(x) == 0`, which is false for a fraction, NaN and the
    /// infinities): the interpreter's `index_int`.
    fn whole_check(&mut self, x: u32) {
        self.get(x);
        self.get(x);
        self.op(0x9c);
        self.op(0xa1);
        self.f64c(0.0);
        self.op(0x62);
        self.trap_if(|fx| {
            fx.get(x);
            fx.rt("vrt_nonint");
        });
    }
    fn trap_if(&mut self, emit_call: impl FnOnce(&mut Fx)) {
        self.open(0x04);
        emit_call(self);
        self.op(0x00);
        self.end();
    }
    /// Push a string from the table: its address, then its length.
    fn string(&mut self, at: (u32, u32)) {
        self.gget(G_STR);
        if at.0 != 0 {
            self.i32c(at.0 as i64);
            self.op(0x6a);
        }
        self.i32c(at.1 as i64);
    }
    /// Pop a value of type `t` into fresh scratch locals; returns them in order.
    fn stash(&mut self, t: Ty) -> Vec<(VT, u32)> {
        let ps = parts(t);
        let slots: Vec<(VT, u32)> = ps.iter().map(|&vt| (vt, self.tmp(vt))).collect();
        for &(_, i) in slots.iter().rev() {
            self.set(i);
        }
        slots
    }
    fn unstash(&mut self, slots: &[(VT, u32)]) {
        for &(_, i) in slots {
            self.get(i);
        }
    }
    fn free_all(&mut self, slots: Vec<(VT, u32)>) {
        for (vt, i) in slots {
            self.release(vt, i);
        }
    }
    fn drop_value(&mut self, t: Ty) {
        for _ in parts(t) {
            self.op(0x1a);
        }
    }
    /// Push the zero value of `t`: what a function that falls off its end
    /// returns (the interpreter's 0.0 for a number).
    fn zero(&mut self, t: Ty) {
        for vt in parts(t) {
            match vt {
                F64 => self.f64c(0.0),
                I32 => self.i32c(0),
                I64 => self.i64c(0),
                F32 => self.f32c(0.0),
            }
        }
    }
}

type R<T> = Result<T, String>;

fn unsupported(what: &str) -> String {
    format!("the direct wasm backend does not compile {what} yet")
}

// ---- the emitter ---------------------------------------------------------------

struct Em<'a> {
    ir: &'a IrProgram<Viv>,
    u: Unit,
}

/// The builtins a `Call` node names by their C spelling. The interpreter knows
/// them by their Viv names; `None` means the module computes it itself.
fn bridge_name(callee: &str) -> Option<&'static str> {
    Some(match callee {
        "viv_text" => "text",
        "viv_eig_sym" => "eig_sym",
        "viv_eig_herm" => "eig_herm",
        "viv_fft3" => "fft3",
        "viv_ifft3" => "ifft3",
        "viv_sum" => "sum",
        "viv_histogram" => "histogram",
        "viv_maxval" => "max",
        "viv_minval" => "min",
        "viv_carr_from" => "complex",
        "viv_cfill" | "viv_fill" => "fill",
        "viv_slab" => "slab",
        "viv_setslab" => "setslab",
        "viv_take" => "take",
        "viv_put" => "put",
        "viv_spectrum_sym" => "spectrum_sym",
        "viv_lin_solve" => "lin_solve",
        "viv_lin_solve_multi" => "lin_solve_multi",
        "viv_cholesky" => "cholesky",
        "viv_eig_general" => "eig_general",
        "viv_eig_near" => "eig_near",
        "viv_assert_near" => "assert_near",
        "viv_assert_true" => "assert_true",
        "viv_assert_gt" => "assert_gt",
        "viv_assert_lt" => "assert_lt",
        "viv_assert_allclose" => "assert_allclose",
        "viv_srand" => "srand",
        "viv_rand" => "rand",
        "viv_sort" => "sort",
        "viv_all" => "all",
        "viv_any" => "any",
        "viv_where_aa" | "viv_where_as" | "viv_where_sa" | "viv_where_ss" => "where",
        // the autodiff tape and adjoint helpers (lower.rs, `lower_ad`)
        "viv_ad_push_n" | "viv_ad_push_a" => "__ad_push",
        "viv_ad_pop_n" | "viv_ad_pop_b" | "viv_ad_pop_a" => "__ad_pop",
        "viv_ad_restore" => "__ad_restore",
        "viv_ad_zero" => "__ad_zero",
        "viv_ad_unb_an" | "viv_ad_unb_na" => "__ad_unb",
        "viv_ad_cmp_nn" | "viv_ad_cmp_na" | "viv_ad_cmp_an" | "viv_ad_cmp_aa" => "__ad_cmp",
        "viv_ad_mm_a" => "__ad_mm_a",
        "viv_ad_mm_b" => "__ad_mm_b",
        "viv_ad_take" => "__ad_take",
        "viv_ad_cat" => "__ad_cat",
        "viv_ad_extent" => "__ad_extent",
        "viv_grow_head" => "__ad_head",
        "viv_grow_tail_n" | "viv_grow_tail_a" => "__ad_tail",
        "viv_tridiag_solve" => "tridiag_solve",
        "viv_roll" => "roll",
        "viv_shape" => "shape",
        "viv_ad_unb_aa" => "__ad_unb",
        "viv_ad_keep" => "__ad_keep",
        "viv_ad_like" => "__ad_like",
        "viv_ad_prod" => "__ad_prod",
        "viv_ad_prodax" => "__ad_prodax",
        "viv_ad_takeax" => "__ad_takeax",
        // the array vocabulary (interp/arrays.rs): `viv_ar_<name>_<n>`
        c => return crate::interp::arrays::of_c_name(c).map(|(sig, _)| sig.name),
    })
}

/// The type a string runtime call gives, or None when `callee` is not one.
fn str_call_ty(callee: &str) -> Option<Ty> {
    use crate::interp::strings::K;
    let of = |k: K| match k {
        K::N => Ty::Num,
        K::B => Ty::Bool,
        K::S => Ty::Str,
        K::L => Ty::Arr(Elem::Str),
        K::A => Ty::Arr(Elem::Num),
        K::I => Ty::Arr(Elem::Int),
    };
    match callee {
        "viv_s_catn" | "viv_s_str_cplx" | "viv_s_fmt_arr" => Some(Ty::Str),
        "viv_ss_of" => Some(Ty::Arr(Elem::Str)),
        _ => crate::lower::string_runtime().into_iter().find(|f| f.c == callee).map(|f| of(f.ret)),
    }
}

/// The 1-argument math functions a `MapMath` node names, by their C names, in
/// the order of the runtime's `MAP_FNS`.
const MAP_FNS: [&str; 18] = [
    "exp", "sqrt", "sin", "cos", "tan", "log", "fabs", "floor", "ceil", "round", "asin", "acos", "atan", "sinh", "cosh", "tanh", "erf", "tgamma",
];

/// Scalar libm names the lowering emits, and the runtime import for each
/// (`None`: a single wasm instruction does it, bit for bit as Rust's).
fn math_import(callee: &str) -> Option<Option<&'static str>> {
    Some(match callee {
        "sqrt" | "fabs" | "floor" | "ceil" => None,
        "exp" => Some("vrt_exp"),
        "log" => Some("vrt_ln"),
        "sin" => Some("vrt_sin"),
        "cos" => Some("vrt_cos"),
        "tan" => Some("vrt_tan"),
        "asin" => Some("vrt_asin"),
        "acos" => Some("vrt_acos"),
        "atan" => Some("vrt_atan"),
        "sinh" => Some("vrt_sinh"),
        "cosh" => Some("vrt_cosh"),
        "tanh" => Some("vrt_tanh"),
        "round" => Some("vrt_round"),
        "erf" => Some("vrt_erf"),
        "tgamma" => Some("vrt_gamma"),
        "pow" => Some("vrt_pow"),
        "atan2" => Some("vrt_atan2"),
        "fma" => Some("vrt_fma"),
        "fmod" => Some("vrt_fmod"),
        "fmin" => Some("vrt_fmin"),
        "fmax" => Some("vrt_fmax"),
        "viv_band" => Some("vrt_band"),
        "viv_bor" => Some("vrt_bor"),
        "viv_bxor" => Some("vrt_bxor"),
        "viv_bnot" => Some("vrt_bnot"),
        "viv_shl" => Some("vrt_shl"),
        "viv_shr" => Some("vrt_shr"),
        "viv_sar" => Some("vrt_sar"),
        "viv_u32" => Some("vrt_u32"),
        "viv_i32" => Some("vrt_i32"),
        "viv_floormod" => Some("vrt_floormod"),
        "viv_floordiv" => Some("vrt_floordiv"),
        _ => return None,
    })
}

/// The complex functions of a complex scalar, by their C names, as the
/// runtime's `vrt_cfn` numbers them.
fn cfn_index(callee: &str) -> Option<i64> {
    Some(match callee {
        "cexp" => 0,
        "clog" => 1,
        "csqrt" => 2,
        "csin" => 3,
        "ccos" => 4,
        "ctan" => 5,
        _ => return None,
    })
}

fn is_complex_kind(k: ArrKind) -> bool {
    matches!(
        k,
        ArrKind::CAdd | ArrKind::CSub | ArrKind::CHad | ArrKind::CHadR | ArrKind::CHadRL | ArrKind::CScale | ArrKind::CDivS | ArrKind::CMatmul | ArrKind::CTranspose | ArrKind::CNeg
    )
}

impl<'a> Em<'a> {
    fn node(&self, id: NodeId) -> &'a Node<Viv> {
        self.ir.get(id)
    }
    fn kids(&self, id: NodeId) -> &'a [NodeId] {
        &self.ir.get(id).children
    }

    /// A record type's name as the `&'static str` `Ty::Record` holds.
    fn record_ty(&self, name: &str) -> Ty {
        Ty::Record(Box::leak(name.to_string().into_boxed_str()))
    }

    /// The type of payload field `k` of variant `tag` of enum `ty`.
    fn enum_field_ty(&self, ty: &str, tag: u32, k: u32) -> R<Ty> {
        self.u
            .enums
            .get(ty)
            .and_then(|e| e.0.get(tag as usize))
            .and_then(|(_, p)| p.get(k as usize).copied())
            .ok_or_else(|| format!("internal: no field {k} of variant {tag} of `{ty}`"))
    }

    fn var_ty(&self, fx: &Fx, name: &str) -> R<Ty> {
        if let Some(&(_, t)) = fx.names.get(name) {
            return Ok(t);
        }
        if let Some(&(_, t)) = self.u.globals.get(name) {
            return Ok(t);
        }
        Err(format!("internal: no variable `{name}`"))
    }

    /// The type an expression node has in this function. The lowering computed
    /// it once and did not keep it (a node is shared by every place its text
    /// occurs, possibly with different variable types), so it is recomputed
    /// here by the lowering's own rules.
    fn ty(&self, fx: &Fx, id: NodeId) -> R<Ty> {
        let n = self.node(id);
        let k = &n.children;
        Ok(match &n.kind {
            Viv::Num(_) => Ty::Num,
            Viv::Int(_) => Ty::Int,
            Viv::EnumNew { ty, .. } => Ty::Enum(Box::leak(ty.clone().into_boxed_str())),
            Viv::EnumIs { .. } | Viv::EnumEq { .. } => Ty::Bool,
            Viv::EnumField { ty, tag, k } => self.enum_field_ty(ty, *tag, *k)?,
            Viv::EnumStr { .. } => Ty::Str,
            Viv::BoolLit(_) => Ty::Bool,
            Viv::Imag(_) => Ty::Cplx,
            Viv::Var { name } => self.var_ty(fx, name)?,
            Viv::ArrLit { .. } | Viv::Zeros | Viv::MapMath { .. } => Ty::Arr(Elem::Num),
            Viv::CArrLit { .. } | Viv::CZeros => Ty::Arr(Elem::Cplx),
            Viv::Index { .. } | Viv::FnLen | Viv::Ckpt | Viv::Fft { .. } => Ty::Num,
            Viv::CIndex { .. } => Ty::Cplx,
            Viv::Neg => self.ty(fx, k[0])?,
            Viv::Not => Ty::Bool,
            Viv::Bin { op, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                    if self.ty(fx, k[0])? == Ty::Cplx || self.ty(fx, k[1])? == Ty::Cplx {
                        Ty::Cplx
                    } else {
                        Ty::Num
                    }
                }
                _ => Ty::Bool,
            },
            Viv::Arr { kind } => Ty::Arr(if is_complex_kind(*kind) { Elem::Cplx } else { Elem::Num }),
            Viv::CArrFn { name } => Ty::Arr(if name == "conj" { Elem::Cplx } else { Elem::Num }),
            Viv::Plot { .. }
            | Viv::Bars { .. }
            | Viv::Scatter { .. }
            | Viv::Plot2d { .. }
            | Viv::Quiver { .. }
            | Viv::Figure { .. }
            | Viv::Plot3d { .. }
            | Viv::Anim { .. }
            | Viv::Scene { .. }
            | Viv::Wire { .. }
            | Viv::Verts { .. } => Ty::Num,
            Viv::Call { callee } => self.call_ty(fx, callee, k)?,
            Viv::StrLit(_) => Ty::Str,
            Viv::RecordLit { ty_name, .. } => self.record_ty(ty_name),
            Viv::RecArrLit { rec } | Viv::RecAppend { rec } => Ty::Recs(Box::leak(rec.clone().into_boxed_str())),
            Viv::RecIndex { rec, .. } => self.record_ty(rec),
            Viv::RecLen => Ty::Num,
            Viv::Own { ty } => *ty,
            Viv::Field { field } => {
                let Ty::Record(r) = self.ty(fx, k[0])? else {
                    return Err("internal: a field of a non-record".into());
                };
                let l = self.u.records.get(r).ok_or_else(|| format!("internal: no record `{r}`"))?;
                l.fields.iter().find(|f| f.0 == *field).map(|f| f.1).ok_or_else(|| format!("internal: no field `{field}`"))?
            }
            Viv::FnRef { sig, .. } | Viv::FnIndex { sig, .. } => Ty::Func(*sig),
            Viv::FnArrLit { sig, .. } => Ty::Arr(Elem::Func(*sig)),
            Viv::ApplyFn { sig } => sig_ret(*sig),
            other => return Err(format!("internal: {other:?} is not an expression")),
        })
    }

    fn call_ty(&self, fx: &Fx, callee: &str, k: &[NodeId]) -> R<Ty> {
        if math_import(callee).is_none() && cfn_index(callee).is_none() {
            if let Some((_, _, ret)) = self.u.funcs.get(callee) {
                return Ok(*ret);
            }
        }
        // every runtime call's result, from the one table (crate::builtins)
        match crate::builtins::runtime_gives(callee) {
            crate::builtins::Gives::Like(j) => self.ty(fx, k[j]),
            g => Ok(g.ty(|_| unreachable!("only Like reads an argument"))),
        }
    }

    // ---- expressions -----------------------------------------------------

    /// Emit `id`, leaving its value on the stack; returns its type.
    fn expr(&mut self, fx: &mut Fx, id: NodeId) -> R<Ty> {
        if let Some(t) = self.fuse_expr(fx, id)? {
            return Ok(t);
        }
        let n = self.node(id);
        let k = &n.children;
        match &n.kind {
            Viv::Num(v) => {
                fx.f64c(*v);
                Ok(Ty::Num)
            }
            Viv::Int(v) => {
                fx.i64c(*v);
                Ok(Ty::Int)
            }
            // enums: a block of 8-byte slots, the tag
            // first, then the payload
            Viv::EnumNew { ty, tag } => {
                let held = self.hold_all(fx, k)?;
                let p = fx.tmp(I32);
                fx.i32c(8 + 8 * k.len() as i64);
                fx.rt("vrt_en_new");
                fx.set(p);
                fx.get(p);
                fx.i32c(*tag as i64);
                fx.store_i32(0);
                for (j, (_, slots)) in held.into_iter().enumerate() {
                    let off = 8 + 8 * j as u32;
                    for &(vt, i) in &slots {
                        fx.get(p);
                        fx.get(i);
                        match vt {
                            F64 => fx.store_f64(off),
                            I64 => fx.store_i64(off),
                            I32 => fx.store_i32(off),
                            F32 => fx.store_f32(off),
                        }
                    }
                    fx.free_all(slots);
                }
                fx.get(p);
                fx.release(I32, p);
                Ok(Ty::Enum(Box::leak(ty.clone().into_boxed_str())))
            }
            Viv::EnumIs { tag } => {
                self.expr(fx, k[0])?;
                fx.load_i32(0);
                fx.i32c(*tag as i64);
                fx.op(0x46);
                fx.op(0xb8);
                Ok(Ty::Bool)
            }
            Viv::EnumEq { ne } => {
                self.expr(fx, k[0])?;
                fx.load_i32(0);
                self.expr(fx, k[1])?;
                fx.load_i32(0);
                fx.op(if *ne { 0x47 } else { 0x46 });
                fx.op(0xb8);
                Ok(Ty::Bool)
            }
            Viv::EnumField { ty, tag, k: f } => {
                let t = self.enum_field_ty(ty, *tag, *f)?;
                self.expr(fx, k[0])?;
                let off = 8 + 8 * *f;
                match t {
                    Ty::Int => fx.load_i64(off),
                    Ty::F32 => fx.load_f32(off),
                    Ty::Num | Ty::Bool => fx.load_f64(off),
                    _ => fx.load_i32(off),
                }
                Ok(t)
            }
            Viv::EnumStr { ty } => {
                let desc = self.u.enums.get(ty).map(|e| e.1).ok_or_else(|| format!("internal: no enum `{ty}`"))?;
                let p = fx.tmp(I32);
                self.expr(fx, k[0])?;
                fx.set(p);
                fx.gget(G_STR);
                fx.i32c(desc as i64);
                fx.get(p);
                fx.release(I32, p);
                fx.rt("vrt_en_str");
                Ok(Ty::Str)
            }
            Viv::BoolLit(b) => {
                fx.f64c(if *b { 1.0 } else { 0.0 });
                Ok(Ty::Bool)
            }
            Viv::Imag(v) => {
                // the interpreter's Cplx(0.0, v)
                fx.f64c(0.0);
                fx.f64c(*v);
                Ok(Ty::Cplx)
            }
            Viv::Var { name } => self.read_var(fx, name),
            Viv::Neg => {
                let t = self.expr(fx, k[0])?;
                if t == Ty::Cplx {
                    let im = fx.tmp(F64);
                    fx.set(im);
                    fx.op(0x9a);
                    fx.get(im);
                    fx.op(0x9a);
                    fx.release(F64, im);
                } else {
                    fx.op(0x9a);
                }
                Ok(t)
            }
            Viv::Not => {
                self.cond(fx, id)?;
                fx.op(0xb8);
                Ok(Ty::Bool)
            }
            Viv::Bin { op, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => self.arith(fx, *op, k[0], k[1]),
                _ => {
                    self.cond(fx, id)?;
                    fx.op(0xb8);
                    Ok(Ty::Bool)
                }
            },
            Viv::Index { .. } => {
                // the recursion counter, in its global
                if let (Some(g), true) = (self.u.depth_g, crate::lower::is_depth_cell(self.ir, k[0], &k[1..])) {
                    fx.gget(g);
                    fx.op(0xb9);
                    return Ok(Ty::Num);
                }
                self.element_addr(fx, id, 3)?;
                fx.load_f64(H_DATA);
                Ok(Ty::Num)
            }
            Viv::CIndex { .. } => {
                self.element_addr(fx, id, 4)?;
                let a = fx.tmp(I32);
                fx.tee(a);
                fx.load_f64(H_DATA);
                fx.get(a);
                fx.load_f64(H_DATA + 8);
                fx.release(I32, a);
                Ok(Ty::Cplx)
            }
            Viv::ArrLit { dims } => {
                // a column count of -1 is a rank-1 literal: `[[]]` is a
                // 1 x 0 matrix, as it is in the interpreter and C
                let (r, c) = match dims.as_slice() {
                    [n] => (*n as i64, -1),
                    [r, c] => (*r as i64, *c as i64),
                    _ => return Err(unsupported("an array literal of rank above 2")),
                };
                let p = fx.tmp(I32);
                fx.i32c(1);
                fx.i32c(r);
                fx.i32c(c);
                fx.rt("vrt_lit");
                fx.set(p);
                if let Some(values) = self.const_values(k) {
                    // a table of constants (a basis set, a reference integral
                    // list) is data, copied in whole: 8 bytes a value in the
                    // module instead of ~17 of code
                    let off = self.u.consts.len();
                    for v in values {
                        self.u.consts.extend_from_slice(&v.to_le_bytes());
                    }
                    fx.get(p);
                    fx.i32c(H_DATA as i64);
                    fx.op(0x6a);
                    fx.i32c(off as i64);
                    fx.i32c(8 * k.len() as i64);
                    fx.code.extend_from_slice(&[0xfc, 0x08, 0x01, 0x00]);
                    fx.get(p);
                    fx.release(I32, p);
                    return Ok(Ty::Arr(Elem::Num));
                }
                for (j, &e) in k.iter().enumerate() {
                    fx.get(p);
                    self.expr(fx, e)?;
                    fx.store_f64(H_DATA + 8 * j as u32);
                }
                fx.get(p);
                fx.release(I32, p);
                Ok(Ty::Arr(Elem::Num))
            }
            Viv::CArrLit { n: len } => {
                let p = fx.tmp(I32);
                fx.i32c(2);
                fx.i32c(*len as i64);
                fx.i32c(-1);
                fx.rt("vrt_lit");
                fx.set(p);
                for (j, &e) in k.iter().enumerate() {
                    let t = self.expr(fx, e)?;
                    if t != Ty::Cplx {
                        fx.f64c(0.0);
                    }
                    let im = fx.tmp(F64);
                    let re = fx.tmp(F64);
                    fx.set(im);
                    fx.set(re);
                    fx.get(p);
                    fx.get(re);
                    fx.store_f64(H_DATA + 16 * j as u32);
                    fx.get(p);
                    fx.get(im);
                    fx.store_f64(H_DATA + 16 * j as u32 + 8);
                    fx.release(F64, im);
                    fx.release(F64, re);
                }
                fx.get(p);
                fx.release(I32, p);
                Ok(Ty::Arr(Elem::Cplx))
            }
            Viv::Zeros | Viv::CZeros => {
                fx.i32c(k.len() as i64);
                for &d in k {
                    self.expr(fx, d)?;
                }
                for _ in k.len()..4 {
                    fx.f64c(0.0);
                }
                let complex = matches!(n.kind, Viv::CZeros);
                fx.rt(if complex { "vrt_czeros" } else { "vrt_zeros" });
                Ok(Ty::Arr(if complex { Elem::Cplx } else { Elem::Num }))
            }
            Viv::Arr { kind } => self.arr_op(fx, *kind, k),
            Viv::MapMath { cname } => {
                let which = MAP_FNS.iter().position(|f| f == cname).ok_or_else(|| unsupported(&format!("`{cname}` over an array")))?;
                fx.i32c(which as i64);
                self.expr(fx, k[0])?;
                fx.rt("vrt_map");
                Ok(Ty::Arr(Elem::Num))
            }
            Viv::CArrFn { name } => {
                let which = ["re", "im", "abs", "conj"].iter().position(|f| f == name).ok_or_else(|| unsupported(&format!("`{name}` over a complex array")))?;
                fx.i32c(which as i64);
                self.expr(fx, k[0])?;
                fx.rt("vrt_cmap");
                Ok(Ty::Arr(if name == "conj" { Elem::Cplx } else { Elem::Num }))
            }
            Viv::Fft { inverse } => {
                self.bridge(fx, if *inverse { "ifft" } else { "fft" }, k, &[])?;
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            // an Int array's elements, read and written in place
            Viv::Call { callee } if callee == "viv_iget" => {
                self.element_addr(fx, id, 3)?;
                fx.load_i64(H_DATA);
                Ok(Ty::Int)
            }
            // an Int array's literal: its elements stored as i64 slots
            Viv::Call { callee } if callee == "viv_ivec" => {
                let held = self.hold_all(fx, k)?;
                let p = fx.tmp(I32);
                fx.i32c(1);
                fx.f64c(k.len() as f64);
                fx.f64c(0.0);
                fx.f64c(0.0);
                fx.f64c(0.0);
                fx.rt("vrt_zeros");
                fx.set(p);
                for (j, (_, slots)) in held.into_iter().enumerate() {
                    fx.get(p);
                    fx.unstash(&slots);
                    fx.store_i64(H_DATA + 8 * j as u32);
                    fx.free_all(slots);
                }
                fx.get(p);
                fx.release(I32, p);
                Ok(Ty::Arr(Elem::Int))
            }
            Viv::Call { callee } if callee == "viv_iset" => {
                self.element_addr(fx, id, 3)?;
                self.expr(fx, k[2])?;
                fx.store_i64(H_DATA);
                Ok(Ty::Void)
            }
            Viv::Call { callee } => self.call(fx, callee, k),
            Viv::StrLit(text) => {
                let o = self.u.str_obj(text);
                fx.gget(G_STR);
                if o != 0 {
                    fx.i32c(o as i64);
                    fx.op(0x6a);
                }
                Ok(Ty::Str)
            }
            Viv::RecordLit { ty_name, fields } => self.record_lit(fx, ty_name, fields, k),
            Viv::Field { field } => {
                let Ty::Record(r) = self.expr(fx, k[0])? else {
                    return Err("internal: a field of a non-record".into());
                };
                let l = self.u.records.get(r).ok_or_else(|| format!("internal: no record `{r}`"))?;
                let (_, t, off) = l.fields.iter().find(|f| f.0 == *field).cloned().ok_or_else(|| format!("internal: no field `{field}`"))?;
                match t {
                    Ty::Cplx => {
                        let p = fx.tmp(I32);
                        fx.tee(p);
                        fx.load_f64(off);
                        fx.get(p);
                        fx.load_f64(off + 8);
                        fx.release(I32, p);
                    }
                    Ty::Num | Ty::Bool => fx.load_f64(off),
                    Ty::Int => fx.load_i64(off),
                    Ty::F32 => fx.load_f32(off),
                    _ => fx.load_i32(off),
                }
                Ok(t)
            }
            Viv::FnRef { name, sig } => {
                let slot = *self.u.slots.get(name).ok_or_else(|| format!("internal: `{name}` has no wrapper"))?;
                fx.i32c(slot as i64);
                Ok(Ty::Func(*sig))
            }
            Viv::FnArrLit { n: len, sig } => {
                let held: Vec<u32> = {
                    let mut v = Vec::new();
                    for &e in k.iter() {
                        self.expr(fx, e)?;
                        let t = fx.tmp(I32);
                        fx.set(t);
                        v.push(t);
                    }
                    v
                };
                let p = fx.tmp(I32);
                fx.i32c(*len as i64);
                fx.rt("vrt_fnarr");
                fx.set(p);
                for (j, t) in held.into_iter().enumerate() {
                    fx.get(p);
                    fx.get(t);
                    fx.store_i32(H_DATA + 4 * j as u32);
                    fx.release(I32, t);
                }
                fx.get(p);
                fx.release(I32, p);
                Ok(Ty::Arr(Elem::Func(*sig)))
            }
            Viv::FnIndex { sig, .. } => {
                // read as any index is: toward zero, a negative one counting
                // from the end, and refused outside the registry
                let a = fx.tmp(I32);
                let x = fx.tmp(F64);
                let i = fx.tmp(I32);
                self.expr(fx, k[0])?;
                fx.set(a);
                self.expr(fx, k[1])?;
                fx.set(x);
                fx.whole_check(x);
                fx.get(x);
                fx.code.extend_from_slice(&[0xfc, 0x02]);
                fx.set(i);
                fx.get(i);
                fx.i32c(0);
                fx.op(0x48);
                fx.open(0x04);
                fx.get(i);
                fx.get(a);
                fx.load_i32(H_DIMS);
                fx.op(0x6a);
                fx.set(i);
                fx.end();
                fx.get(i);
                fx.get(a);
                fx.load_i32(H_DIMS);
                fx.op(0x4f);
                fx.trap_if(|fx| {
                    fx.get(x);
                    fx.get(a);
                    fx.load_i32(H_DIMS);
                    fx.rt("vrt_fnoob");
                });
                fx.get(a);
                fx.get(i);
                fx.i32c(2);
                fx.op(0x74);
                fx.op(0x6a);
                fx.load_i32(H_DATA);
                fx.release(I32, a);
                fx.release(F64, x);
                fx.release(I32, i);
                Ok(Ty::Func(*sig))
            }
            Viv::FnLen | Viv::RecLen => {
                self.expr(fx, k[0])?;
                fx.load_i32(H_DIMS);
                fx.op(0xb7);
                Ok(Ty::Num)
            }
            // an array of records: a block of record pointers, each record
            // owned by its slot
            Viv::RecArrLit { rec } => {
                let held = self.hold_all(fx, k)?;
                let p = fx.tmp(I32);
                fx.i32c(k.len() as i64);
                fx.rt("vrt_fnarr");
                fx.set(p);
                for (j, (_, slots)) in held.into_iter().enumerate() {
                    fx.get(p);
                    fx.get(slots[0].1);
                    fx.store_i32(H_DATA + 4 * j as u32);
                    fx.free_all(slots);
                }
                fx.get(p);
                fx.release(I32, p);
                Ok(Ty::Recs(Box::leak(rec.clone().into_boxed_str())))
            }
            Viv::RecIndex { rec, .. } => {
                self.rec_slot(fx, k[0], k[1])?;
                fx.load_i32(H_DATA);
                Ok(self.record_ty(rec))
            }
            Viv::RecAppend { rec } => {
                self.expr(fx, k[0])?;
                self.expr(fx, k[1])?;
                fx.rt("vrt_recs_push");
                Ok(Ty::Recs(Box::leak(rec.clone().into_boxed_str())))
            }
            // a copy of its own, so a later write through either name is not
            // seen through the other
            Viv::Own { ty } => {
                self.expr(fx, k[0])?;
                match ty {
                    Ty::Record(r) => {
                        let h = self.rec_helper(r, 0)?;
                        fx.call(h);
                    }
                    Ty::Recs(r) => {
                        let h = self.rec_helper(r, 1)?;
                        fx.call(h);
                    }
                    _ => {}
                }
                Ok(*ty)
            }
            Viv::ApplyFn { sig } => {
                let f = fx.tmp(I32);
                self.expr(fx, k[0])?;
                fx.set(f);
                for &a in &k[1..] {
                    self.expr(fx, a)?;
                }
                fx.get(f);
                fx.release(I32, f);
                let (p, r) = sig_type(*sig);
                let t = self.u.types.idx(p, r);
                fx.op(0x11);
                uleb(t as u64, &mut fx.code);
                fx.op(0x00);
                Ok(sig_ret(*sig))
            }
            Viv::Closure { .. } => Err(unsupported("a closure outside `simpson`/`trapezoid`")),
            Viv::Ckpt => {
                fx.f64c(0.0);
                Ok(Ty::Num)
            }
            // `print` with text or formats: the interpreter's `__print` writes
            // the line, from (tag, payload) pairs -- `t` and the text, or a
            // format's code and the value (crate::printfmt)
            Viv::Print { parts } => {
                let mut held = self.hold_all(fx, k)?.into_iter();
                for p in parts {
                    let tag = match p {
                        PrintPart::Text(_) => "t".to_string(),
                        PrintPart::Num(f) | PrintPart::Arr(f) => f.encode(),
                        PrintPart::Bool | PrintPart::Cplx => crate::printfmt::Fmt::BARE.encode(),
                        PrintPart::Str => "t".to_string(),
                        // a map's two formats (interp.rs, `__print`)
                        PrintPart::Map(kf, vf) | PrintPart::SMap(kf, vf) => format!("{}|{}", kf.encode(), vf.encode()),
                    };
                    let at = self.u.string(&tag);
                    fx.string(at);
                    fx.rt("vrt_arg_str");
                    if let PrintPart::Text(t) = p {
                        let at = self.u.string(t);
                        fx.string(at);
                        fx.rt("vrt_arg_str");
                    } else {
                        let one = held.next().expect("a value per part that is not text");
                        self.stage_held(fx, vec![one]);
                    }
                }
                let at = self.u.string("__print");
                fx.string(at);
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            // the figure statements: the interpreter's arm writes the record
            Viv::Plot { label } | Viv::Bars { label } | Viv::Plot2d { label } | Viv::Quiver { label } | Viv::Plot3d { label } => {
                let name = match &n.kind {
                    Viv::Plot { .. } => "plot",
                    Viv::Bars { .. } => "bars",
                    Viv::Plot2d { .. } => "plot2d",
                    Viv::Quiver { .. } => "quiver",
                    _ => "plot3d",
                };
                self.bridge(fx, name, k, &[label.as_str()])?;
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            Viv::Scatter { label, color } => {
                self.bridge(fx, "scatter", k, &[label.as_str(), color.as_str()])?;
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            Viv::Figure { title, xlabel, ylabel, style } => {
                // strings first, then the window, then the style: its order
                // in the source, which is the order the interpreter reads
                let mut strs = vec![title.as_str(), xlabel.as_str(), ylabel.as_str()];
                let held = self.hold_all(fx, k)?;
                for s in strs.drain(..) {
                    let at = self.u.string(s);
                    fx.string(at);
                    fx.rt("vrt_arg_str");
                }
                self.stage_held(fx, held);
                if !k.is_empty() && !style.is_empty() {
                    let at = self.u.string(style);
                    fx.string(at);
                    fx.rt("vrt_arg_str");
                }
                let at = self.u.string("figure");
                fx.string(at);
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            Viv::Anim { head, arrays: _ } => {
                // `verb: text ; text`, fixed at compile time; the texts go first
                let (verb, rest) = head.split_once(':').unwrap_or((head.as_str(), ""));
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                let texts: Vec<&str> = if rest.is_empty() { Vec::new() } else { rest.split(" ; ").collect() };
                let held = self.hold_all(fx, k)?;
                for t in texts {
                    let at = self.u.string(t);
                    fx.string(at);
                    fx.rt("vrt_arg_str");
                }
                self.stage_held(fx, held);
                let at = self.u.string(verb);
                fx.string(at);
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            Viv::Scene { title } => {
                self.bridge_strs_first(fx, "scene", &[title.as_str()], k)?;
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            Viv::Wire { group } | Viv::Verts { group } => {
                let name = if matches!(n.kind, Viv::Wire { .. }) { "wire" } else { "verts" };
                self.bridge(fx, name, k, &[group.as_str()])?;
                fx.rt("vrt_call_f");
                Ok(Ty::Num)
            }
            other => Err(format!("internal: {other:?} in expression position")),
        }
    }

    /// The values of an array literal whose every element is a number
    /// constant (`2.5` or `-2.5`), when it is long enough to be worth a table.
    fn const_values(&self, k: &[NodeId]) -> Option<Vec<f64>> {
        if k.len() < 8 {
            return None;
        }
        k.iter()
            .map(|&c| match &self.node(c).kind {
                Viv::Num(v) => Some(*v),
                // negation, as the interpreter evaluates `-2.5`
                Viv::Neg => match self.kids(c) {
                    [g] => match self.node(*g).kind {
                        Viv::Num(v) => Some(-v),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Give every array variable this function indexes a header cache
    /// (`Fx::hdr`), and fill the caches of parameters and globals, whose
    /// values are set before the body runs. A global cannot change while a
    /// function runs: only the top level assigns one.
    fn cache_headers(&mut self, fx: &mut Fx, body: &[NodeId], in_main: bool) -> R<()> {
        let mut bases = Vec::new();
        for &c in body {
            self.index_bases(c, &mut bases);
        }
        for name in bases {
            let Ok(Ty::Arr(Elem::Num | Elem::Cplx)) = self.var_ty(fx, &name) else { continue };
            let first = fx.local(I32);
            for _ in 0..4 {
                fx.local(I32);
            }
            fx.hdr.insert(name.clone(), first);
            let is_param = fx.names.get(&name).is_some_and(|&(i, _)| i < fx.n_param_slots);
            let is_global = !fx.names.contains_key(&name);
            if is_param || (is_global && !in_main) {
                self.refresh_header(fx, &name)?;
            }
        }
        Ok(())
    }

    fn index_bases(&self, id: NodeId, out: &mut Vec<String>) {
        let n = self.node(id);
        if matches!(n.kind, Viv::Index { .. } | Viv::CIndex { .. } | Viv::SetIndex { .. } | Viv::CSetIndex { .. }) {
            if let Viv::Var { name } = &self.node(n.children[0]).kind {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
        }
        for &c in &n.children {
            self.index_bases(c, out);
        }
    }

    /// Copy `name`'s header (rank, dims) into its cache, if it has one.
    fn refresh_header(&mut self, fx: &mut Fx, name: &str) -> R<()> {
        let Some(&c) = fx.hdr.get(name) else { return Ok(()) };
        for j in 0..5u32 {
            self.read_var(fx, name)?;
            fx.load_i32(H_RANK + 4 * j);
            fx.set(c + j);
        }
        Ok(())
    }

    fn read_var(&mut self, fx: &mut Fx, name: &str) -> R<Ty> {
        if let Some(&(i, t)) = fx.names.get(name) {
            for j in 0..parts(t).len() as u32 {
                fx.get(i + j);
            }
            return Ok(t);
        }
        if let Some(&(g, t)) = self.u.globals.get(name) {
            for j in 0..parts(t).len() as u32 {
                fx.gget(g + j);
            }
            return Ok(t);
        }
        Err(format!("internal: no variable `{name}`"))
    }

    fn write_var(&mut self, fx: &mut Fx, name: &str) -> R<()> {
        if let Some(&(i, t)) = fx.names.get(name) {
            for j in (0..parts(t).len() as u32).rev() {
                fx.set(i + j);
            }
            return Ok(());
        }
        if let Some(&(g, t)) = self.u.globals.get(name) {
            for j in (0..parts(t).len() as u32).rev() {
                fx.gset(g + j);
            }
            return Ok(());
        }
        Err(format!("internal: no variable `{name}`"))
    }

    /// `+ - * /` on numbers and complex numbers. A real operand of a complex
    /// operation is lifted to (x, 0.0) and the complex formula applied whole,
    /// which is what the interpreter does -- the zero products included.
    fn arith(&mut self, fx: &mut Fx, op: BinOp, l: NodeId, r: NodeId) -> R<Ty> {
        let (lt, rt_) = (self.ty(fx, l)?, self.ty(fx, r)?);
        if lt != Ty::Cplx && rt_ != Ty::Cplx {
            self.expr(fx, l)?;
            self.expr(fx, r)?;
            fx.op(match op {
                BinOp::Add => 0xa0,
                BinOp::Sub => 0xa1,
                BinOp::Mul => 0xa2,
                _ => 0xa3,
            });
            return Ok(Ty::Num);
        }
        self.expr(fx, l)?;
        if lt != Ty::Cplx {
            fx.f64c(0.0);
        }
        self.expr(fx, r)?;
        if rt_ != Ty::Cplx {
            fx.f64c(0.0);
        }
        let (ar, ai, br, bi) = (fx.tmp(F64), fx.tmp(F64), fx.tmp(F64), fx.tmp(F64));
        fx.set(bi);
        fx.set(br);
        fx.set(ai);
        fx.set(ar);
        let mul = |fx: &mut Fx, a: u32, b: u32| {
            fx.get(a);
            fx.get(b);
            fx.op(0xa2);
        };
        match op {
            BinOp::Add | BinOp::Sub => {
                let o = if op == BinOp::Add { 0xa0 } else { 0xa1 };
                fx.get(ar);
                fx.get(br);
                fx.op(o);
                fx.get(ai);
                fx.get(bi);
                fx.op(o);
            }
            BinOp::Mul => {
                // (ar br - ai bi, ar bi + ai br)
                mul(fx, ar, br);
                mul(fx, ai, bi);
                fx.op(0xa1);
                mul(fx, ar, bi);
                mul(fx, ai, br);
                fx.op(0xa0);
            }
            _ => {
                // Smith's algorithm, in the runtime: the interpreter's own
                // `vmath::cdiv`, so the branch and every rounding are its
                for t in [ar, ai, br, bi] {
                    fx.get(t);
                }
                fx.rt("vrt_cdiv");
                fx.gget(G_SCRATCH);
                fx.load_f64(0);
                fx.gget(G_SCRATCH);
                fx.load_f64(8);
            }
        }
        for t in [ar, ai, br, bi] {
            fx.release(F64, t);
        }
        Ok(Ty::Cplx)
    }

    /// Emit a condition, leaving an i32. `and` / `or` short-circuit, as C's
    /// `&&` / `||` and the interpreter do: the right side runs only when the
    /// left one does not decide the answer, so `j > 0 and a[j - 1] > v` never
    /// reads `a[-1]`.
    fn cond(&mut self, fx: &mut Fx, id: NodeId) -> R<()> {
        let n = self.node(id);
        let k = &n.children;
        match &n.kind {
            Viv::BoolLit(b) => fx.i32c(*b as i64),
            Viv::Not => {
                self.cond(fx, k[0])?;
                fx.op(0x45);
            }
            Viv::Bin { op: op @ (BinOp::And | BinOp::Or), .. } => {
                // and: if (result i32) <right> else 0 end
                // or:  if (result i32) 1 else <right> end
                self.cond(fx, k[0])?;
                fx.code.extend_from_slice(&[0x04, 0x7f]);
                fx.depth += 1;
                if *op == BinOp::And {
                    self.cond(fx, k[1])?;
                    fx.op(0x05);
                    fx.i32c(0);
                } else {
                    fx.i32c(1);
                    fx.op(0x05);
                    self.cond(fx, k[1])?;
                }
                fx.end();
            }
            Viv::Bin { op, .. } if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Matmul) => {
                // numbers, or two booleans (1.0 / 0.0) for == and !=, or two
                // Ints
                let int = self.expr(fx, k[0])? == Ty::Int;
                self.expr(fx, k[1])?;
                fx.op(match (op, int) {
                    (BinOp::Lt, false) => 0x63,
                    (BinOp::Le, false) => 0x65,
                    (BinOp::Gt, false) => 0x64,
                    (BinOp::Ge, false) => 0x66,
                    (BinOp::EqEq, false) => 0x61,
                    (_, false) => 0x62,
                    (BinOp::Lt, true) => 0x53,
                    (BinOp::Le, true) => 0x57,
                    (BinOp::Gt, true) => 0x55,
                    (BinOp::Ge, true) => 0x59,
                    (BinOp::EqEq, true) => 0x51,
                    (_, true) => 0x52,
                });
            }
            _ => {
                let t = self.expr(fx, id)?;
                if t != Ty::Bool && t != Ty::Num {
                    return Err(format!("internal: a condition of type {t:?}"));
                }
                fx.f64c(0.0);
                fx.op(0x62);
            }
        }
        Ok(())
    }

    /// The Int operations `lower::ints::unchecked` writes as plain C, as one
    /// wasm instruction each, on the same facts.
    fn int_unchecked(&mut self, fx: &mut Fx, callee: &str, k: &[NodeId]) -> R<Option<Ty>> {
        use crate::lower::intidx::{fact, Av, Fact, EXACT};
        let bounded = |a: Av, lim: f64| matches!(a, Av::F(Fact::Range { lo, hi, int: true }) if lo >= -lim && hi <= lim);
        let (ops, op, conv): (usize, u8, Option<u8>) = match callee {
            "viv_i_add" => (2, 0x7c, None),
            "viv_i_sub" => (2, 0x7d, None),
            "viv_i_mul" => (2, 0x7e, None),
            "viv_i_of" if bounded(fact(self.ir, k[0]), 4611686018427387904.0) => (1, 0, Some(0xb0)),
            "viv_i_num" if bounded(fact(self.ir, k[0]), EXACT) => (1, 0, Some(0xb9)),
            _ => return Ok(None),
        };
        if conv.is_none() {
            // the result's range, from the operands' facts
            let a = fact(self.ir, k[0]);
            let b = fact(self.ir, k[1]);
            let (Some((l1, h1)), Some((l2, h2))) = (if a.whole() { a.bounds() } else { None }, if b.whole() { b.bounds() } else { None }) else { return Ok(None) };
            let c = match callee {
                "viv_i_add" => [l1 + l2, h1 + h2, l1 + l2, h1 + h2],
                "viv_i_sub" => [l1 - h2, h1 - l2, l1 - h2, h1 - l2],
                _ => [l1 * l2, l1 * h2, h1 * l2, h1 * h2],
            };
            let lim = 4611686018427387904.0;
            if c.iter().any(|x| !(x.abs() <= lim)) {
                return Ok(None);
            }
        }
        for &a in &k[..ops] {
            self.expr(fx, a)?;
        }
        match conv {
            Some(c) => {
                fx.op(c);
                Ok(Some(if c == 0xb0 { Ty::Int } else { Ty::Num }))
            }
            None => {
                fx.op(op);
                Ok(Some(Ty::Int))
            }
        }
    }

    /// An exact integer form (lower/intidx.rs) as an i64 on the stack.
    fn iform(&mut self, fx: &mut Fx, f: &crate::lower::intidx::IForm) -> R<()> {
        use crate::lower::intidx::IForm;
        match f {
            IForm::Lit(c) => fx.i64c(*c),
            IForm::Ctr { counter, .. } => {
                let l = fx.ctrs.iter().rev().find(|c| &c.0 == counter).map(|c| c.1).ok_or("internal: a counter outside its loop")?;
                fx.get(l);
            }
            IForm::Cast { var, .. } => {
                self.read_var(fx, var)?;
                fx.op(0xb0); // i64.trunc_f64_s: an integer in range, by the facts
            }
            IForm::Int(id, ..) => {
                if self.expr(fx, *id)? != Ty::Int {
                    return Err("internal: an Int form that is not an Int".into());
                }
            }
            IForm::Neg(a) => {
                fx.i64c(0);
                self.iform(fx, a)?;
                fx.op(0x7d);
            }
            IForm::Bin(op, a, b) => {
                self.iform(fx, a)?;
                self.iform(fx, b)?;
                fx.op(match op {
                    BinOp::Add => 0x7c,
                    BinOp::Sub => 0x7d,
                    _ => 0x7e,
                });
            }
        }
        Ok(())
    }

    /// For `Index` / `CIndex` / `SetIndex` / `CSetIndex` nodes: evaluate the
    /// base and the indices, check them as the interpreter's `flat_index` does
    /// (the rank first, then each index in turn), and leave `base + flat <<
    /// shift` on the stack; the element is at that address plus `H_DATA`.
    fn element_addr(&mut self, fx: &mut Fx, id: NodeId, shift: i64) -> R<()> {
        let k = self.kids(id);
        let nidx = match &self.node(id).kind {
            Viv::Index { .. } | Viv::CIndex { .. } => k.len() - 1,
            Viv::Call { callee } if callee == "viv_iget" => k.len() - 1,
            _ => k.len() - 2,
        };
        // A variable is read in place; anything else goes through a scratch.
        // `cached`: the variable's rank and dims are in locals (`Fx::hdr`).
        let base = &self.node(k[0]).kind;
        let cached = match base {
            Viv::Var { name } => fx.hdr.get(name).copied(),
            _ => None,
        };
        let (a, owned) = match base {
            Viv::Var { name } if fx.names.contains_key(name) => (fx.names[name].0, false),
            _ => {
                let a = fx.tmp(I32);
                self.expr(fx, k[0])?;
                fx.set(a);
                (a, true)
            }
        };
        // the rank, then dimension `d`: from the cache, or the header
        let rank = move |fx: &mut Fx| match cached {
            Some(c) => fx.get(c),
            None => {
                fx.get(a);
                fx.load_i32(H_RANK);
            }
        };
        let dim = move |fx: &mut Fx, d: u32| match cached {
            Some(c) => fx.get(c + 1 + d),
            None => {
                fx.get(a);
                fx.load_i32(H_DIMS + 4 * d);
            }
        };
        rank(fx);
        fx.i32c(nidx as i64);
        fx.op(0x47);
        fx.trap_if(|fx| {
            rank(fx);
            fx.i32c(nidx as i64);
            fx.rt("vrt_rank");
        });
        let x = fx.tmp(F64);
        let i = fx.tmp(I32);
        let flat = if nidx > 1 { Some(fx.tmp(I32)) } else { None };
        for (d, &ix) in k[1..=nidx].iter().enumerate() {
            let d = d as u32;
            // an index with an exact integer form (a loop counter, an Int,
            // a number proved an integer): computed as an i64, with one
            // test for both ends (lower/intidx.rs)
            if let Some(f) = crate::lower::intidx::exact_form(self.ir, ix) {
                let int = matches!(f, crate::lower::intidx::IForm::Int(..));
                let xi = fx.tmp(I64);
                let t = fx.tmp(I64);
                self.iform(fx, &f)?;
                fx.tee(xi);
                fx.set(t);
                // a negative index counts from the end
                fx.get(t);
                fx.i64c(0);
                fx.op(0x53); // i64.lt_s
                fx.open(0x04);
                fx.get(t);
                dim(fx, d);
                fx.op(0xad); // i64.extend_i32_u
                fx.op(0x7c);
                fx.set(t);
                fx.end();
                fx.get(t);
                dim(fx, d);
                fx.op(0xad);
                fx.op(0x5a); // i64.ge_u
                fx.trap_if(|fx| {
                    // an Int past 2^53 stops where the interpreter converts it
                    if int {
                        fx.get(xi);
                        fx.rt("vrt_i_num");
                        fx.op(0x1a);
                    }
                    fx.get(xi);
                    fx.op(0xb9);
                    fx.i32c(d as i64);
                    dim(fx, d);
                    fx.rt("vrt_oob");
                });
                fx.get(t);
                fx.op(0xa7); // i32.wrap_i64
                fx.set(i);
                fx.release(I64, xi);
                fx.release(I64, t);
                if let Some(f) = flat {
                    if d == 0 {
                        fx.get(i);
                        fx.set(f);
                    } else {
                        fx.get(f);
                        dim(fx, d);
                        fx.op(0x6c);
                        fx.get(i);
                        fx.op(0x6a);
                        fx.set(f);
                    }
                }
                continue;
            }
            // an Int index is exact as a double below 2^53, and an index past
            // that is out of bounds whatever it rounds to
            if self.expr(fx, ix)? == Ty::Int {
                fx.op(0xb9);
            }
            fx.set(x);
            fx.whole_check(x);
            fx.get(x);
            fx.code.extend_from_slice(&[0xfc, 0x02]);
            fx.set(i);
            // a negative index counts from the end: i += dim when i < 0
            fx.get(i);
            fx.i32c(0);
            fx.op(0x48);
            fx.open(0x04);
            fx.get(i);
            dim(fx, d);
            fx.op(0x6a);
            fx.set(i);
            fx.end();
            fx.get(i);
            dim(fx, d);
            fx.op(0x4f);
            fx.trap_if(|fx| {
                fx.get(x);
                fx.i32c(d as i64);
                dim(fx, d);
                fx.rt("vrt_oob");
            });
            if let Some(f) = flat {
                if d == 0 {
                    fx.get(i);
                    fx.set(f);
                } else {
                    fx.get(f);
                    dim(fx, d);
                    fx.op(0x6c);
                    fx.get(i);
                    fx.op(0x6a);
                    fx.set(f);
                }
            }
        }
        fx.get(a);
        fx.get(flat.unwrap_or(i));
        fx.i32c(shift);
        fx.op(0x74);
        fx.op(0x6a);
        if let Some(f) = flat {
            fx.release(I32, f);
        }
        fx.release(I32, i);
        fx.release(F64, x);
        if owned {
            fx.release(I32, a);
        }
        Ok(())
    }

    /// Evaluate every node in `k` into scratch locals, in order. A bridged
    /// call stages its arguments only after ALL of them are computed, since
    /// computing one may itself make a bridged call.
    fn hold_all(&mut self, fx: &mut Fx, k: &[NodeId]) -> R<Vec<(Ty, Vec<(VT, u32)>)>> {
        let mut held = Vec::with_capacity(k.len());
        for &c in k {
            let t = self.expr(fx, c)?;
            let slots = fx.stash(t);
            held.push((t, slots));
        }
        Ok(held)
    }

    fn stage_held(&mut self, fx: &mut Fx, held: Vec<(Ty, Vec<(VT, u32)>)>) {
        for (t, slots) in held {
            fx.unstash(&slots);
            fx.rt(match t {
                Ty::Num => "vrt_arg_num",
                Ty::Bool => "vrt_arg_bool",
                Ty::Cplx => "vrt_arg_cplx",
                Ty::Str => "vrt_arg_s",
                Ty::Map(..) => "vrt_arg_map",
                _ => "vrt_arg_arr",
            });
            fx.free_all(slots);
        }
    }

    /// Stage `k` (values) and then `strs` (string literals) for the builtin
    /// `name`, and push the name, ready for one of the `vrt_call_*`.
    fn bridge(&mut self, fx: &mut Fx, name: &str, k: &[NodeId], strs: &[&str]) -> R<()> {
        let held = self.hold_all(fx, k)?;
        self.stage_held(fx, held);
        for s in strs {
            let at = self.u.string(s);
            fx.string(at);
            fx.rt("vrt_arg_str");
        }
        let at = self.u.string(name);
        fx.string(at);
        Ok(())
    }

    fn bridge_strs_first(&mut self, fx: &mut Fx, name: &str, strs: &[&str], k: &[NodeId]) -> R<()> {
        let held = self.hold_all(fx, k)?;
        for s in strs {
            let at = self.u.string(s);
            fx.string(at);
            fx.rt("vrt_arg_str");
        }
        self.stage_held(fx, held);
        let at = self.u.string(name);
        fx.string(at);
        Ok(())
    }

    /// A resolved array operation, by the runtime's copy of the interpreter's
    /// arm for it (`interp::wasm_rt`, "the hot builtins"). The lowering
    /// stores a broadcast's operands [array, scalar] whatever their order in
    /// the source; `s - a` and `s / a` put the order back.
    /// A fused elementwise tree, or a reduction over a view or a tree
    /// (crate::fuse): the operands pushed in the unfused order, then one
    /// runtime call with the program as an array of constants.
    fn fuse_expr(&mut self, fx: &mut Fx, id: NodeId) -> R<Option<Ty>> {
        let n = self.node(id);
        let (plan, red) = match &n.kind {
            Viv::Arr { .. } | Viv::MapMath { .. } => match crate::fuse::plan_tree(self.ir, id) {
                Some(p) => (p, None),
                None => return Ok(None),
            },
            Viv::Call { callee } => {
                let Some(&(_, kind, arity)) = crate::fuse::REDUCTIONS.iter().find(|r| r.0 == callee) else { return Ok(None) };
                match crate::fuse::plan_reduction(self.ir, &n.children, arity) {
                    Some(p) => (p, Some(kind)),
                    None => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        if !crate::fuse::enabled() {
            return Ok(None);
        }
        // the mapped functions by the runtime's numbers; the operands' types
        let mut code = plan.code.clone();
        let mut i = 0;
        while i < code.len() {
            let op = code[i] as u8;
            if op == crate::fuse::MAP as u8 {
                let name = &plan.fns[code[i + 1] as usize];
                let Some(w) = MAP_FNS.iter().position(|f| f == name) else { return Ok(None) };
                code[i + 1] = w as f64;
            }
            i += match op {
                0 | 1 | 3..=6 | 14 => 2,
                2 => 5,
                _ => 1,
            };
        }
        for &x in &plan.nums {
            if self.ty(fx, x)? != Ty::Num {
                return Ok(None);
            }
        }
        // the operands in the order the program reads them
        let (mut ia, mut inn, mut i) = (0, 0, 0);
        while i < code.len() {
            let op = code[i] as u8;
            match op {
                0 => {
                    self.expr(fx, plan.arrs[ia])?;
                    ia += 1;
                    fx.rt("vrt_fz_a");
                }
                1 => {
                    self.expr(fx, plan.nums[inn])?;
                    inn += 1;
                    fx.rt("vrt_fz_n");
                }
                2 => {
                    self.expr(fx, plan.arrs[ia])?;
                    ia += 1;
                    fx.rt("vrt_fz_a");
                    for _ in 0..4 * code[i + 2] as usize {
                        self.expr(fx, plan.nums[inn])?;
                        inn += 1;
                        fx.rt("vrt_fz_n");
                    }
                }
                _ => {}
            }
            i += match op {
                0 | 1 | 3..=6 | 14 => 2,
                2 => 5,
                _ => 1,
            };
        }
        if let Some(kind) = red {
            fx.i32c(kind as i64);
        }
        // the program: a constant array, copied in whole
        let p = fx.tmp(I32);
        fx.i32c(1);
        fx.i32c(code.len() as i64);
        fx.i32c(-1);
        fx.rt("vrt_lit");
        fx.set(p);
        let off = self.u.consts.len();
        for v in &code {
            self.u.consts.extend_from_slice(&v.to_le_bytes());
        }
        fx.get(p);
        fx.i32c(H_DATA as i64);
        fx.op(0x6a);
        fx.i32c(off as i64);
        fx.i32c(8 * code.len() as i64);
        fx.code.extend_from_slice(&[0xfc, 0x08, 0x01, 0x00]);
        fx.get(p);
        fx.release(I32, p);
        if red.is_some() {
            fx.rt("vrt_fz_red");
            Ok(Some(Ty::Num))
        } else {
            fx.rt("vrt_fz");
            Ok(Some(Ty::Arr(Elem::Num)))
        }
    }

    fn arr_op(&mut self, fx: &mut Fx, kind: ArrKind, k: &[NodeId]) -> R<Ty> {
        use ArrKind::*;
        let out = Ty::Arr(if is_complex_kind(kind) { Elem::Cplx } else { Elem::Num });
        let tag = |op: BinOp| op as u8 as i64;
        let args = |em: &mut Self, fx: &mut Fx| -> R<()> {
            for &c in k {
                let t = em.expr(fx, c)?;
                if matches!(kind, CScale | CDivS) {
                    if t == Ty::Num {
                        fx.f64c(0.0);
                    }
                }
            }
            Ok(())
        };
        match kind {
            Add | Sub | Had | Div | CAdd | CSub | CHad | CHadR | CHadRL => {
                let op = match kind {
                    Add | CAdd => BinOp::Add,
                    Sub | CSub => BinOp::Sub,
                    Div => BinOp::Div,
                    _ => BinOp::Mul,
                };
                fx.i32c(tag(op));
                args(self, fx)?;
                fx.rt(if is_complex_kind(kind) { "vrt_carith" } else { "vrt_arith" });
            }
            AddS | SubS | Scale | DivS | SubSR | DivSR => {
                let op = match kind {
                    AddS => BinOp::Add,
                    SubS | SubSR => BinOp::Sub,
                    Scale => BinOp::Mul,
                    _ => BinOp::Div,
                };
                fx.i32c(tag(op));
                args(self, fx)?;
                fx.i32c(matches!(kind, SubSR | DivSR) as i64);
                fx.rt("vrt_arith_s");
            }
            CScale | CDivS => {
                fx.i32c(tag(if matches!(kind, CScale) { BinOp::Mul } else { BinOp::Div }));
                args(self, fx)?;
                fx.i32c(0);
                fx.rt("vrt_carith_s");
            }
            Matmul | CMatmul => {
                args(self, fx)?;
                fx.rt(if matches!(kind, Matmul) { "vrt_matmul" } else { "vrt_cmatmul" });
            }
            Transpose | CTranspose => {
                args(self, fx)?;
                fx.rt("vrt_transpose");
            }
            CNeg => {
                args(self, fx)?;
                fx.rt("vrt_cneg");
            }
            Linspace => {
                args(self, fx)?;
                fx.rt("vrt_linspace");
            }
            MaxA | MinA => {
                fx.i32c(matches!(kind, MaxA) as i64);
                args(self, fx)?;
                fx.rt("vrt_minmax");
            }
            MaxS | MinS => {
                fx.i32c(matches!(kind, MaxS) as i64);
                args(self, fx)?;
                fx.rt("vrt_minmax_s");
            }
            PowS => {
                args(self, fx)?;
                fx.rt("vrt_pows");
            }
            // comparisons: the arithmetic arms with the comparison's tag
            LtA | LeA | GtA | GeA | EqA | NeA | LtS | LeS | GtS | GeS | EqS | NeS => {
                let op = match kind {
                    LtA | LtS => BinOp::Lt,
                    LeA | LeS => BinOp::Le,
                    GtA | GtS => BinOp::Gt,
                    GeA | GeS => BinOp::Ge,
                    EqA | EqS => BinOp::EqEq,
                    _ => BinOp::Ne,
                };
                fx.i32c(tag(op));
                args(self, fx)?;
                if matches!(kind, LtA | LeA | GtA | GeA | EqA | NeA) {
                    fx.rt("vrt_arith");
                } else {
                    fx.i32c(0);
                    fx.rt("vrt_arith_s");
                }
            }
        }
        Ok(out)
    }

    fn call(&mut self, fx: &mut Fx, callee: &str, k: &[NodeId]) -> R<Ty> {
        if let Some(t) = self.str_call(fx, callee, k)? {
            return Ok(t);
        }
        // Ints: an operation the facts prove cannot
        // overflow or fail is one instruction (lower/intidx.rs); the rest
        // are the runtime's checked operations
        if let Some(t) = self.int_unchecked(fx, callee, k)? {
            return Ok(t);
        }
        // Float32 and the packed arrays: an instruction,
        // or the runtime; an Int index converted to the number it is
        if let Some(f) = crate::lower::f32s::rt_fn(callee) {
            use crate::lower::f32s::FK;
            for (j, &a) in k.iter().enumerate() {
                let t = self.expr(fx, a)?;
                if f.args.get(j) == Some(&FK::N) && t == Ty::Int {
                    fx.op(0xb9);
                }
            }
            match crate::lower::f32s::wasm_op(callee) {
                Some(op) => fx.op(op),
                None => fx.rt(&callee.replacen("viv_", "vrt_", 1)),
            }
            return Ok(crate::builtins::runtime_gives(callee).ty(|_| Ty::Num));
        }
        if let Some(f) = crate::lower::ints::rt_fn(callee) {
            for &a in k {
                self.expr(fx, a)?;
            }
            fx.rt(&callee.replacen("viv_", "vrt_", 1));
            return Ok(crate::builtins::runtime_gives(callee).ty(|_| Ty::Num)).map(|t| if f.ret == crate::lower::ints::IK::A { Ty::Arr(Elem::Int) } else { t });
        }
        // scalar math: one instruction, or the runtime's copy of the function
        if let Some(imp) = math_import(callee) {
            // `%` and `//` carry their source position as a third child, which
            // only the C backend reads
            let k = if matches!(callee, "viv_floormod" | "viv_floordiv") { &k[..2] } else { k };
            for &a in k {
                self.expr(fx, a)?;
            }
            match imp {
                Some(f) => fx.rt(f),
                None => fx.op(match callee {
                    "sqrt" => 0x9f,
                    "fabs" => 0x99,
                    "floor" => 0x9c,
                    _ => 0x9b,
                }),
            }
            return Ok(Ty::Num);
        }
        if let Some(which) = cfn_index(callee) {
            fx.i32c(which);
            self.expr(fx, k[0])?;
            fx.rt("vrt_cfn");
            fx.gget(G_SCRATCH);
            fx.load_f64(0);
            fx.gget(G_SCRATCH);
            fx.load_f64(8);
            return Ok(Ty::Cplx);
        }
        match callee {
            "viv_print_num" => {
                self.expr(fx, k[0])?;
                fx.rt("vrt_print");
                fx.f64c(0.0);
                return Ok(Ty::Num);
            }
            "viv_cplx" => {
                self.expr(fx, k[0])?;
                self.expr(fx, k[1])?;
                return Ok(Ty::Cplx);
            }
            "creal" | "cimag" => {
                self.expr(fx, k[0])?;
                if callee == "creal" {
                    fx.op(0x1a);
                } else {
                    let im = fx.tmp(F64);
                    fx.set(im);
                    fx.op(0x1a);
                    fx.get(im);
                    fx.release(F64, im);
                }
                return Ok(Ty::Num);
            }
            "conj" => {
                self.expr(fx, k[0])?;
                fx.op(0x9a);
                return Ok(Ty::Cplx);
            }
            "cabs" => {
                self.expr(fx, k[0])?;
                fx.rt("vrt_hypot");
                return Ok(Ty::Num);
            }
            "viv_release_mode" => {
                // a debug run, as `viv run` and the default C build are
                fx.f64c(0.0);
                return Ok(Ty::Bool);
            }
            "viv_clock" => {
                fx.call(host_clock());
                return Ok(Ty::Num);
            }
            "viv_len" => {
                self.expr(fx, k[0])?;
                fx.load_i32(H_DIMS);
                fx.op(0xb8);
                return Ok(Ty::Num);
            }
            "viv_numel" => {
                self.expr(fx, k[0])?;
                fx.load_i32(H_SIZE);
                fx.op(0xb8);
                return Ok(Ty::Num);
            }
            "viv_simpson" | "viv_trapezoid" => return self.integrate(fx, callee == "viv_simpson", k),
            c if c.starts_with("viv_s_db_") => {
                return Err("the db_* builtins (SQLite) are not available in the browser yet: run the program natively (viv run, or the C engine)".to_string())
            }
            _ => {}
        }
        if let Some(t) = self.native_call(fx, callee, k)? {
            return Ok(t);
        }
        if let Some((f, params, ret)) = self.u.funcs.get(callee).cloned() {
            for (&a, (_, pt)) in k.iter().zip(&params) {
                let t = self.expr(fx, a)?;
                if parts(t) != parts(*pt) {
                    return Err(format!("internal: argument of type {t:?} for a {pt:?} parameter of `{callee}`"));
                }
            }
            fx.call(f);
            return Ok(ret);
        }
        if let Some(name) = bridge_name(callee) {
            // the C runtime's calling conventions the interpreter does not have:
            // an element width after slab/take/setslab/put, and an axis of -1
            // meaning "none given" for roll
            let args: &[NodeId] = match callee {
                "viv_slab" | "viv_take" | "viv_setslab" | "viv_put" => &k[..k.len() - 1],
                "viv_roll" if matches!(self.node(k[2]).kind, Viv::Num(v) if v == -1.0) => &k[..2],
                _ => k,
            };
            let ret = self.call_ty(fx, callee, k)?;
            self.bridge(fx, name, args, &[])?;
            match ret {
                Ty::Arr(_) => fx.rt("vrt_call_a"),
                Ty::Cplx => {
                    fx.rt("vrt_call_c");
                    fx.gget(G_SCRATCH);
                    fx.load_f64(0);
                    fx.gget(G_SCRATCH);
                    fx.load_f64(8);
                }
                _ => fx.rt("vrt_call_f"),
            }
            return Ok(ret);
        }
        Err(unsupported(&format!("a call to `{callee}`")))
    }

    /// A string runtime call: its arguments pushed in
    /// order, then its import; `None` when `callee` is not one.
    fn str_call(&mut self, fx: &mut Fx, callee: &str, k: &[NodeId]) -> R<Option<Ty>> {
        // a figure's title, an animation's text, as values: the interpreter's arm
        match callee {
            "viv_figure_s" | "viv_figure_win_s" => {
                self.bridge(fx, "figure", k, &[])?;
                fx.rt("vrt_call_f");
                return Ok(Some(Ty::Num));
            }
            "viv_anim_s" => {
                let Viv::StrLit(verb) = &self.node(k[0]).kind else { return Err("internal: an animation verb first".into()) };
                let verb = verb.clone();
                self.bridge(fx, &verb, &k[1..], &[])?;
                fx.rt("vrt_call_f");
                return Ok(Some(Ty::Num));
            }
            _ => {}
        }
        let Some(ret) = str_call_ty(callee) else { return Ok(None) };
        let import = format!("vrt_{}", &callee[4..]);
        match callee {
            // the parts are computed first, then staged, then joined
            "viv_s_catn" | "viv_ss_of" => {
                let held = self.hold_all(fx, k)?;
                for (_, slots) in held {
                    fx.unstash(&slots);
                    fx.rt("vrt_s_push");
                    fx.free_all(slots);
                }
            }
            _ => {
                for &c in k {
                    self.expr(fx, c)?;
                }
            }
        }
        fx.rt(&import);
        Ok(Some(ret))
    }

    /// The builtins the runtime does in place (see `arr_op`): args pushed in
    /// order, one call. `None` when `callee` is not one of them.
    fn native_call(&mut self, fx: &mut Fx, callee: &str, k: &[NodeId]) -> R<Option<Ty>> {
        let push = |em: &mut Self, fx: &mut Fx, nodes: &[NodeId]| -> R<Vec<Ty>> { nodes.iter().map(|&c| em.expr(fx, c)).collect() };
        Ok(Some(match callee {
            "viv_slab" | "viv_take" => {
                let t = push(self, fx, &k[..2])?;
                fx.rt(if callee == "viv_slab" { "vrt_slab" } else { "vrt_take" });
                t[0]
            }
            "viv_setslab" | "viv_put" => {
                push(self, fx, &k[..3])?;
                fx.rt(if callee == "viv_setslab" { "vrt_setslab" } else { "vrt_put" });
                Ty::Num
            }
            // slices, copy and the joins (interp/slices.rs); the element
            // width the C takes last is in the header here
            "viv_getslice" | "viv_getslice_v" => {
                let t = push(self, fx, &k[..2])?;
                fx.rt("vrt_getslice");
                t[0]
            }
            "viv_slicecheck" => {
                push(self, fx, &k[..2])?;
                fx.rt("vrt_slicecheck");
                Ty::Num
            }
            "viv_setslice" | "viv_setslice_n" | "viv_setslice_c" => {
                push(self, fx, &k[..3])?;
                fx.rt(match callee {
                    "viv_setslice" => "vrt_setslice",
                    "viv_setslice_n" => "vrt_setslice_n",
                    _ => "vrt_setslice_c",
                });
                Ty::Num
            }
            "viv_copy" => {
                let t = push(self, fx, &k[..1])?;
                fx.rt("vrt_copy");
                t[0]
            }
            // maps (interp/maps.rs): every one a runtime call of the same name
            c if c.starts_with("viv_map_") => {
                push(self, fx, k)?;
                fx.rt(&c.replacen("viv_", "vrt_", 1));
                self.call_ty(fx, callee, k)?
            }
            // growable arrays (interp/growable.rs), in place in the runtime
            "viv_seal" => {
                push(self, fx, &k[..1])?;
                fx.rt("vrt_seal");
                Ty::Arr(Elem::Num)
            }
            "viv_append_n" | "viv_append_r" | "viv_extend" => {
                push(self, fx, &k[..3])?;
                fx.rt(match callee {
                    "viv_append_n" => "vrt_append_n",
                    "viv_append_r" => "vrt_append_r",
                    _ => "vrt_extend",
                });
                Ty::Arr(Elem::Num)
            }
            "viv_cat" | "viv_cat1" => {
                let n = if callee == "viv_cat" { 3 } else { 2 };
                let t = push(self, fx, &k[..n])?;
                fx.rt(if callee == "viv_cat" { "vrt_cat" } else { "vrt_cat1" });
                t[1]
            }
            "viv_shape" => {
                push(self, fx, k)?;
                fx.rt("vrt_shape");
                Ty::Arr(Elem::Num)
            }
            "viv_sum" => {
                push(self, fx, k)?;
                fx.rt("vrt_sum");
                Ty::Num
            }
            "viv_maxval" | "viv_minval" => {
                fx.i32c((callee == "viv_maxval") as i64);
                push(self, fx, k)?;
                fx.rt("vrt_extreme");
                Ty::Num
            }
            "viv_fill" | "viv_cfill" => {
                let t = push(self, fx, k)?;
                if t[1] != Ty::Cplx {
                    fx.f64c(0.0);
                }
                fx.rt("vrt_fill");
                Ty::Num
            }
            "viv_roll" => {
                // an axis of -1, which only the lowering writes, means "none given"
                let given = !matches!(self.node(k[2]).kind, Viv::Num(v) if v == -1.0);
                push(self, fx, k)?;
                fx.i32c(given as i64);
                fx.rt("vrt_roll");
                Ty::Arr(Elem::Num)
            }
            "viv_carr_from" => {
                push(self, fx, k)?;
                fx.rt("vrt_carr_from");
                Ty::Arr(Elem::Cplx)
            }
            _ => return Ok(None),
        }))
    }

    /// The address of slot `idx` of the array of records `base`, less
    /// `H_DATA`, on the stack: the index read as any index is (toward zero, a
    /// negative one counting from the end) and refused outside the array,
    /// with the words a registry's index uses.
    fn rec_slot(&mut self, fx: &mut Fx, base: NodeId, idx: NodeId) -> R<()> {
        let a = fx.tmp(I32);
        let x = fx.tmp(F64);
        let i = fx.tmp(I32);
        self.expr(fx, base)?;
        fx.set(a);
        // an Int index converts as an array's does; a fraction stops the run
        if self.expr(fx, idx)? == Ty::Int {
            fx.op(0xb9);
        }
        fx.set(x);
        fx.whole_check(x);
        fx.get(x);
        fx.code.extend_from_slice(&[0xfc, 0x02]);
        fx.set(i);
        fx.get(i);
        fx.i32c(0);
        fx.op(0x48);
        fx.open(0x04);
        fx.get(i);
        fx.get(a);
        fx.load_i32(H_DIMS);
        fx.op(0x6a);
        fx.set(i);
        fx.end();
        fx.get(i);
        fx.get(a);
        fx.load_i32(H_DIMS);
        fx.op(0x4f);
        fx.trap_if(|fx| {
            fx.get(x);
            fx.get(a);
            fx.load_i32(H_DIMS);
            fx.rt("vrt_fnoob");
        });
        fx.get(a);
        fx.get(i);
        fx.i32c(2);
        fx.op(0x74);
        fx.op(0x6a);
        fx.release(I32, a);
        fx.release(F64, x);
        fx.release(I32, i);
        Ok(())
    }

    /// A helper function of the record type `rec`, made the first time it is
    /// asked for: `kind` 0 a record's own copy (`Viv::Own`): a new block with
    /// its slots, and a copy of each record and array of records in it; 1 an
    /// array of records' own copy, each record copied; 2 keeping what a
    /// record holds on the arena (its block, its arrays, strings and maps,
    /// and its records', at any depth); 3 the same for an array of records.
    /// Whether a value of type `t` reaches an enum value: its keep helpers
    /// (kinds 2 and 3) then take the mark the enum walk stops at as a second
    /// parameter.
    fn holds_enum(&self, t: Ty) -> bool {
        match t {
            Ty::Enum(_) => true,
            Ty::Record(r) | Ty::Recs(r) => self.u.records.get(r).is_some_and(|l| l.fields.iter().any(|f| self.holds_enum(f.1))),
            _ => false,
        }
    }

    fn rec_helper(&mut self, rec: &str, kind: u8) -> R<u32> {
        let key = format!("rec{kind}:{rec}");
        if let Some(&h) = self.u.helpers.get(&key) {
            return Ok(h);
        }
        let l = self.u.records.get(rec).cloned().ok_or_else(|| format!("internal: no record `{rec}`"))?;
        // the helpers this one calls, made first
        let mut inner: HashMap<String, u32> = HashMap::new();
        for (_, ft, _) in &l.fields {
            let want = match (kind, ft) {
                (0, Ty::Record(u)) => Some((u.to_string(), 0)),
                (0, Ty::Recs(u)) => Some((u.to_string(), 1)),
                (2, Ty::Record(u)) => Some((u.to_string(), 2)),
                (2, Ty::Recs(u)) => Some((u.to_string(), 3)),
                _ => None,
            };
            if let Some((u, kd)) = want {
                let h = self.rec_helper(&u, kd)?;
                inner.insert(format!("{kd}:{u}"), h);
            }
        }
        let element = match kind {
            1 => Some(self.rec_helper(rec, 0)?),
            3 => Some(self.rec_helper(rec, 2)?),
            _ => None,
        };
        let results = if kind < 2 { vec![I32] } else { vec![] };
        // a keep helper of a record reaching an enum takes the mark too
        let en = kind >= 2 && l.fields.iter().any(|f| self.holds_enum(f.1));
        let idx = self.u.new_fn(if en { vec![I32, I32] } else { vec![I32] }, results);
        self.u.helpers.insert(key, idx);
        let mut f = Fx::new(&[], Ty::Void, "");
        f.n_param_slots = if en { 2 } else { 1 };
        let arg = 0u32;
        let mark = 1u32;
        // `for i to n { body(i) }` over an array of records' slots
        let each = |f: &mut Fx, a: u32, body: &dyn Fn(&mut Fx, u32)| {
            let i = f.local(I32);
            f.i32c(0);
            f.set(i);
            let brk = f.open(0x02);
            let top = f.open(0x03);
            f.get(i);
            f.get(a);
            f.load_i32(H_DIMS);
            f.op(0x4e); // i32.ge_s
            f.br_if(brk);
            body(f, i);
            f.get(i);
            f.i32c(1);
            f.op(0x6a);
            f.set(i);
            f.br(top);
            f.end();
            f.end();
        };
        let slot_addr = |f: &mut Fx, a: u32, i: u32| {
            f.get(a);
            f.get(i);
            f.i32c(2);
            f.op(0x74);
            f.op(0x6a);
        };
        match kind {
            0 => {
                let q = f.local(I32);
                f.i32c(l.bytes as i64);
                f.rt("vrt_block");
                f.set(q);
                for j in 0..l.bytes / 8 {
                    f.get(q);
                    f.get(arg);
                    f.load_f64(8 * j);
                    f.store_f64(8 * j);
                }
                for (_, ft, off) in &l.fields {
                    let h = match ft {
                        Ty::Record(u) => inner[&format!("0:{u}")],
                        Ty::Recs(u) => inner[&format!("1:{u}")],
                        _ => continue,
                    };
                    f.get(q);
                    f.get(q);
                    f.load_i32(*off);
                    f.call(h);
                    f.store_i32(*off);
                }
                f.get(q);
            }
            1 => {
                let b = f.local(I32);
                let own = element.expect("made above");
                f.get(arg);
                f.load_i32(H_DIMS);
                f.rt("vrt_fnarr");
                f.set(b);
                each(&mut f, arg, &|f, i| {
                    slot_addr(f, b, i);
                    slot_addr(f, arg, i);
                    f.load_i32(H_DATA);
                    f.call(own);
                    f.store_i32(H_DATA);
                });
                f.get(b);
            }
            2 => {
                // the block (its header is just before the record's data)
                f.get(arg);
                f.i32c(H_DATA as i64);
                f.op(0x6b);
                f.rt("vrt_keep");
                for (_, ft, off) in &l.fields {
                    let call = match ft {
                        Ty::Str => "vrt_keep_str",
                        Ty::Arr(Elem::Str) => "vrt_keep_strs",
                        Ty::Arr(_) | Ty::Map(..) => "vrt_keep",
                        Ty::Record(u) => {
                            let h = inner[&format!("2:{u}")];
                            f.get(arg);
                            f.load_i32(*off);
                            if self.holds_enum(*ft) {
                                f.get(mark);
                            }
                            f.call(h);
                            continue;
                        }
                        Ty::Recs(u) => {
                            let h = inner[&format!("3:{u}")];
                            f.get(arg);
                            f.load_i32(*off);
                            if self.holds_enum(*ft) {
                                f.get(mark);
                            }
                            f.call(h);
                            continue;
                        }
                        // an enum value: every node it reaches
                        Ty::Enum(_) => {
                            f.get(arg);
                            f.load_i32(*off);
                            self.keep_value(&mut f, *ft, mark)?;
                            continue;
                        }
                        _ => continue,
                    };
                    f.get(arg);
                    f.load_i32(*off);
                    f.rt(call);
                }
            }
            _ => {
                let keep = element.expect("made above");
                f.get(arg);
                f.rt("vrt_keep");
                each(&mut f, arg, &|f, i| {
                    slot_addr(f, arg, i);
                    f.load_i32(H_DATA);
                    if en {
                        f.get(mark);
                    }
                    f.call(keep);
                });
            }
        }
        f.op(0x0b);
        let body = finish(f);
        self.u.helper_code.push((idx, self.u.fn_types[(idx - n_imports()) as usize], body));
        Ok(idx)
    }

    fn record_lit(&mut self, fx: &mut Fx, ty_name: &str, fields: &[String], k: &[NodeId]) -> R<Ty> {
        let held = self.hold_all(fx, k)?;
        let l = self.u.records.get(ty_name).ok_or_else(|| format!("internal: no record `{ty_name}`"))?;
        let offs: Vec<u32> = fields
            .iter()
            .map(|f| l.fields.iter().find(|d| d.0 == *f).map(|d| d.2).ok_or_else(|| format!("internal: no field `{f}`")))
            .collect::<R<_>>()?;
        let p = fx.tmp(I32);
        fx.i32c(l.bytes as i64);
        fx.rt("vrt_block");
        fx.set(p);
        for ((t, slots), off) in held.into_iter().zip(offs) {
            for (j, &(vt, i)) in slots.iter().enumerate() {
                fx.get(p);
                fx.get(i);
                match vt {
                    F64 => fx.store_f64(off + 8 * j as u32),
                    I32 => fx.store_i32(off),
                    I64 => fx.store_i64(off),
                    F32 => fx.store_f32(off),
                }
            }
            let _ = t;
            fx.free_all(slots);
        }
        fx.get(p);
        fx.release(I32, p);
        Ok(self.record_ty(ty_name))
    }

    /// `simpson` / `trapezoid` over a lifted lambda: a helper function per
    /// (rule, lambda), `(a, b, n, captures...) -> f64`, the interpreter's loop.
    fn integrate(&mut self, fx: &mut Fx, simpson: bool, k: &[NodeId]) -> R<Ty> {
        let Viv::Closure { name, captures } = &self.node(k[0]).kind else {
            return Err(unsupported("an integration over something other than a `fn(x) -> ...` literal"));
        };
        let key = format!("{}:{name}", if simpson { "simpson" } else { "trapezoid" });
        let helper = match self.u.helpers.get(&key) {
            Some(&h) => h,
            None => self.integrate_helper(simpson, name, captures.len())?,
        };
        self.u.helpers.insert(key, helper);
        for &a in &k[1..] {
            self.expr(fx, a)?;
        }
        for c in captures.clone() {
            self.read_var(fx, &c)?;
        }
        fx.call(helper);
        Ok(Ty::Num)
    }

    fn integrate_helper(&mut self, simpson: bool, lam: &str, ncaps: usize) -> R<u32> {
        let (lam_f, _, _) = self.u.lambdas.get(lam).cloned().ok_or_else(|| format!("internal: no lambda `{lam}`"))?;
        let params = vec![F64; 3 + ncaps];
        let idx = self.u.new_fn(params, vec![F64]);
        let (lo, hi, nreq) = (0u32, 1u32, 2u32);
        let caps: Vec<u32> = (3..3 + ncaps as u32).collect();
        let mut f = Fx::new(&[], Ty::Num, "");
        f.n_param_slots = 3 + ncaps as u32;
        let (n, h, s, kk, x) = (f.local(F64), f.local(F64), f.local(F64), f.local(F64), f.local(F64));
        // simpson: whether k is odd (starts 0: k = 0 is even)
        let odd = f.local(I32);
        let apply = |f: &mut Fx, arg: &dyn Fn(&mut Fx)| {
            arg(f);
            for &c in &caps {
                f.get(c);
            }
            f.call(lam_f);
        };
        // h = (hi - lo) / n
        let h_of = |f: &mut Fx, n: u32| {
            f.get(hi);
            f.get(lo);
            f.op(0xa1);
            f.get(n);
            f.op(0xa3);
            f.set(h);
        };
        // x = lo + (k + 1) * h
        let x_of = |f: &mut Fx| {
            f.get(lo);
            f.get(kk);
            f.f64c(1.0);
            f.op(0xa0);
            f.get(h);
            f.op(0xa2);
            f.op(0xa0);
        };
        if simpson {
            // n = nreq; if n % 2 != 0 { n += 1 }
            f.get(nreq);
            f.set(n);
            f.get(n);
            f.f64c(2.0);
            f.rt("vrt_fmod");
            f.f64c(0.0);
            f.op(0x62);
            f.open(0x04);
            f.get(n);
            f.f64c(1.0);
            f.op(0xa0);
            f.set(n);
            f.end();
            h_of(&mut f, n);
            // s = f(lo) + f(hi)
            apply(&mut f, &|f| f.get(lo));
            apply(&mut f, &|f| f.get(hi));
            f.op(0xa0);
            f.set(s);
        } else {
            f.get(nreq);
            f.set(n);
            h_of(&mut f, n);
            // s = (f(lo) + f(hi)) / 2
            apply(&mut f, &|f| f.get(lo));
            apply(&mut f, &|f| f.get(hi));
            f.op(0xa0);
            f.f64c(2.0);
            f.op(0xa3);
            f.set(s);
        }
        f.f64c(0.0);
        f.set(kk);
        // while k < n - 1
        let brk = f.open(0x02);
        let top = f.open(0x03);
        f.get(kk);
        f.get(n);
        f.f64c(1.0);
        f.op(0xa1);
        f.op(0x63);
        f.op(0x45);
        f.br_if(brk);
        if simpson {
            x_of(&mut f);
            f.set(x);
            // (k + 1) % 2 == 1 ? 4 : 2, times f(x), added to s. k counts 0,
            // 1, 2, ... exactly (a double is exact on integers below 2^53, and
            // a loop that long never ends in any engine), so (k + 1) % 2 == 1
            // is "k is even": a flag flipped each time round gives the same
            // answer without a software fmod per point, which was 16% of
            // geomopt_water's run.
            f.get(s);
            f.get(odd);
            f.op(0x45);
            f.code.extend_from_slice(&[0x04, 0x7c]);
            f.depth += 1;
            f.f64c(4.0);
            f.op(0x05);
            f.f64c(2.0);
            f.end();
            apply(&mut f, &|f| f.get(x));
            f.op(0xa2);
            f.op(0xa0);
            f.set(s);
        } else {
            f.get(s);
            apply(&mut f, &|f| x_of(f));
            f.op(0xa0);
            f.set(s);
        }
        f.get(kk);
        f.f64c(1.0);
        f.op(0xa0);
        f.set(kk);
        if simpson {
            f.get(odd);
            f.i32c(1);
            f.op(0x73);
            f.set(odd);
        }
        f.br(top);
        f.end();
        f.end();
        f.get(s);
        f.get(h);
        f.op(0xa2);
        if simpson {
            f.f64c(3.0);
            f.op(0xa3);
        }
        f.op(0x0b);
        let body = finish(f);
        self.u.helper_code.push((idx, self.u.fn_types[(idx - n_imports()) as usize], body));
        Ok(idx)
    }

    // ---- statements -------------------------------------------------------

    fn stmt(&mut self, fx: &mut Fx, id: NodeId) -> R<()> {
        let n = self.node(id);
        let k = &n.children;
        match &n.kind {
            // the C backend's per-pass release; wasm loops release their own
            Viv::Ckpt | Viv::Nop => {}
            Viv::Assign { name } => {
                self.expr(fx, k[0])?;
                self.write_var(fx, name)?;
                self.refresh_header(fx, name)?;
            }
            // `base.field = value`: the value, then the place; the record at
            // the place is the variable's own, so it is written where it is
            Viv::SetField { field, .. } => {
                let t = self.expr(fx, k[1])?;
                let slots = fx.stash(t);
                let Ty::Record(r) = self.expr(fx, k[0])? else {
                    return Err("internal: a field of a non-record".into());
                };
                let l = self.u.records.get(r).ok_or_else(|| format!("internal: no record `{r}`"))?;
                let (_, ft, off) = l.fields.iter().find(|f| f.0 == *field).cloned().ok_or_else(|| format!("internal: no field `{field}`"))?;
                let p = fx.tmp(I32);
                fx.set(p);
                for (j, &(vt, i)) in slots.iter().enumerate() {
                    fx.get(p);
                    fx.get(i);
                    match vt {
                        F64 => fx.store_f64(off + 8 * j as u32),
                        I32 => fx.store_i32(off),
                        I64 => fx.store_i64(off),
                        F32 => fx.store_f32(off),
                    }
                }
                let _ = ft;
                fx.release(I32, p);
                fx.free_all(slots);
            }
            Viv::RecSetIndex { .. } => {
                self.expr(fx, k[2])?;
                let v = fx.tmp(I32);
                fx.set(v);
                self.rec_slot(fx, k[0], k[1])?;
                fx.get(v);
                fx.store_i32(H_DATA);
                fx.release(I32, v);
            }
            Viv::SetIndex { .. } => {
                if let (Some(g), true) = (self.u.depth_g, crate::lower::is_depth_cell(self.ir, k[0], &k[1..k.len() - 1])) {
                    match crate::lower::depth_step(self.ir, k[k.len() - 1]) {
                        0 => {
                            self.expr(fx, k[k.len() - 1])?;
                            fx.code.extend_from_slice(&[0xfc, 0x06]);
                        }
                        d => {
                            fx.gget(g);
                            fx.i64c(1);
                            fx.op(if d > 0 { 0x7c } else { 0x7d });
                        }
                    }
                    fx.gset(g);
                    return Ok(());
                }
                // indices checked first, then the value computed: the order
                // of the interpreter's `flat_index` and `eval`
                self.element_addr(fx, id, 3)?;
                // a write to an array the def made itself needs no record
                let mine = match &self.node(k[0]).kind {
                    Viv::Var { name } if !self.u.dj_pins.iter().any(|p| p.contains(name)) => {
                        if fx.dj_params.contains(name) { !fx.dj_pinnable.contains(name) } else { fx.dj_fresh.contains(name) }
                    }
                    _ => false,
                };
                if let (true, Some((lo, hi))) = ((fx.dj || self.u.dj_body > 0) && !mine, self.u.dj_range) {
                    // in a pinned loop's reach: a write into the pinned
                    // arrays' range is recorded
                    let a = fx.tmp(I32);
                    fx.tee(a);
                    self.expr(fx, k[k.len() - 1])?;
                    fx.get(a);
                    fx.gget(lo);
                    fx.op(0x4f); // i32.ge_u
                    fx.get(a);
                    fx.gget(hi);
                    fx.op(0x49); // i32.lt_u
                    fx.op(0x71); // i32.and
                    fx.open(0x04);
                    fx.get(a);
                    fx.rt("vrt_dj_note");
                    fx.end();
                    fx.release(I32, a);
                } else {
                    self.expr(fx, k[k.len() - 1])?;
                }
                fx.store_f64(H_DATA);
            }
            Viv::CSetIndex { .. } => {
                self.element_addr(fx, id, 4)?;
                let a = fx.tmp(I32);
                fx.set(a);
                let t = self.expr(fx, k[k.len() - 1])?;
                if t != Ty::Cplx {
                    fx.f64c(0.0);
                }
                let slots = fx.stash(Ty::Cplx);
                fx.get(a);
                fx.get(slots[0].1);
                fx.store_f64(H_DATA);
                fx.get(a);
                fx.get(slots[1].1);
                fx.store_f64(H_DATA + 8);
                fx.free_all(slots);
                fx.release(I32, a);
            }
            Viv::Return => {
                let t = self.expr(fx, k[0])?;
                let slots = fx.stash(t);
                self.ret(fx, &slots)?;
                fx.free_all(slots);
            }
            Viv::StepReturn { exits, ty } => {
                let t = self.expr(fx, k[0])?;
                let _ = ty;
                let slots = fx.stash(t);
                for (def, step, depth) in exits.clone() {
                    self.step_event(fx, false, &step, depth, def.as_deref());
                }
                self.ret(fx, &slots)?;
                fx.free_all(slots);
            }
            Viv::LensReturn { .. } => {
                let t = self.expr(fx, k[0])?;
                let slots = fx.stash(t);
                for &c in &k[1..] {
                    self.stmt(fx, c)?;
                }
                self.ret(fx, &slots)?;
                fx.free_all(slots);
            }
            Viv::ExprStmt => {
                let t = self.expr(fx, k[0])?;
                fx.drop_value(t);
            }
            Viv::If => {
                self.cond(fx, k[0])?;
                fx.open(0x04);
                self.block(fx, k[1])?;
                if !self.kids(k[2]).is_empty() {
                    fx.op(0x05);
                    self.block(fx, k[2])?;
                }
                fx.end();
            }
            Viv::While => {
                let mark = self.loop_mark(fx, id)?;
                let keeps = self.loop_keeps(fx, id)?;
                let brk = fx.open(0x02);
                let top = fx.open(0x03);
                self.cond(fx, k[0])?;
                fx.op(0x45);
                fx.br_if(brk);
                let cont = fx.open(0x02);
                fx.loops.push(LoopLabels { brk, cont });
                self.block(fx, k[1])?;
                fx.loops.pop();
                fx.end();
                if let (Some(m), Some(keeps)) = (mark, &keeps) {
                    self.release_iteration(fx, m, keeps)?;
                }
                fx.br(top);
                fx.end();
                fx.end();
                // the last test of a condition that allocates (the one that
                // ended the loop) made garbage no pass released
                if let (Some(m), Some(keeps)) = (mark, &keeps) {
                    if self.allocates(fx, k[0])? {
                        self.release_iteration(fx, m, keeps)?;
                    }
                }
                if let Some(m) = mark {
                    fx.release(I32, m);
                }
            }
            Viv::For { var } | Viv::ParallelFor { var, .. } => {
                // the interpreter's loop: the bound evaluated once, a counter
                // of its own (so the body cannot change the trip count), and
                // the variable set from it each time round
                let (pins, pin_what) = match &n.kind {
                    Viv::ParallelFor { pins, pin_what, .. } => (pins.clone(), pin_what.clone()),
                    _ => (Vec::new(), String::new()),
                };
                // The counter is an i64 (lower/intidx.rs), as in C. An Int
                // bound is the pass count; a number bound gives the passes
                // the interpreter's double counter makes: ceil(bound), none
                // for 0, a negative bound or NaN, and at most 2^53 (where
                // that counter stops advancing):
                // i64.trunc_sat(ceil(min(bound, 2^53))).
                let bound = fx.tmp(I64);
                let i = fx.tmp(I64);
                let int = self.expr(fx, k[0])? == Ty::Int;
                if !int {
                    fx.f64c(crate::lower::intidx::EXACT);
                    fx.op(0xa4); // f64.min
                    fx.op(0x9b); // f64.ceil
                    fx.code.extend_from_slice(&[0xfc, 0x06]); // i64.trunc_sat_f64_s
                }
                fx.set(bound);
                fx.i64c(0);
                fx.set(i);
                let ctr = matches!(n.kind, Viv::For { .. });
                if ctr {
                    let unassigned = !crate::lower::intidx::assigns(self.ir, k[1], var);
                    let hi = match crate::lower::intidx::fact(self.ir, k[0]).bounds() {
                        Some((_, h)) => (h.ceil() - 1.0).clamp(0.0, if int { 4.0 * crate::lower::intidx::EXACT } else { crate::lower::intidx::EXACT - 1.0 }),
                        None => if int { 4.0 * crate::lower::intidx::EXACT } else { crate::lower::intidx::EXACT - 1.0 },
                    };
                    let counter = format!("viv_k_{var}");
                    fx.ctrs.push((counter.clone(), i));
                    crate::lower::intidx::push_loop(crate::lower::intidx::LoopCx { var: var.clone(), counter, trip: String::new(), unassigned, hi, ub: None, body: k[1], assumed: Vec::new() });
                }
                // a pinned loop: the record of which pass writes what
                if !pins.is_empty() {
                    let at = self.u.string(&pin_what);
                    fx.string(at);
                    fx.rt("vrt_dj_begin");
                    for a in &pins {
                        self.read_var(fx, a)?;
                        let at = self.u.string(a);
                        fx.string(at);
                        fx.rt("vrt_dj_track");
                    }
                    if let Some((lo, hi)) = self.u.dj_range {
                        fx.rt("vrt_dj_lo");
                        fx.gset(lo);
                        fx.rt("vrt_dj_hi");
                        fx.gset(hi);
                    }
                    self.u.dj_body += 1;
                    self.u.dj_pins.push(pins.clone());
                }
                let mark = self.loop_mark(fx, id)?;
                let keeps = self.loop_keeps(fx, id)?;
                let brk = fx.open(0x02);
                let top = fx.open(0x03);
                fx.get(i);
                fx.get(bound);
                fx.op(0x59); // i64.ge_s
                fx.br_if(brk);
                fx.get(i);
                // the variable's own type decides, not the bound's: an Int
                // variable counted to a literal bound (`for i to 3` after `for
                // i to n` with n an Int) stays an Int (bug B112)
                let var_int = fx.names.get(var.as_str()).map(|&(_, t)| t == Ty::Int)
                    .or_else(|| self.u.globals.get(var.as_str()).map(|&(_, t)| t == Ty::Int))
                    .unwrap_or(int);
                if !var_int {
                    fx.op(0xb9); // f64.convert_i64_s
                }
                self.write_var(fx, var)?;
                if !pins.is_empty() {
                    fx.get(i);
                    fx.op(0xb9);
                    fx.rt("vrt_dj_pass");
                }
                let cont = fx.open(0x02);
                fx.loops.push(LoopLabels { brk, cont });
                self.block(fx, k[1])?;
                fx.loops.pop();
                fx.end();
                if let (Some(m), Some(keeps)) = (mark, &keeps) {
                    self.release_iteration(fx, m, keeps)?;
                }
                fx.get(i);
                fx.i64c(1);
                fx.op(0x7c); // i64.add
                fx.set(i);
                fx.br(top);
                fx.end();
                fx.end();
                if ctr {
                    fx.ctrs.pop();
                    crate::lower::intidx::pop_loop();
                }
                if !pins.is_empty() {
                    self.u.dj_body -= 1;
                    self.u.dj_pins.pop();
                    fx.rt("vrt_dj_end");
                    if let Some((lo, hi)) = self.u.dj_range {
                        fx.i32c(0);
                        fx.gset(lo);
                        fx.i32c(0);
                        fx.gset(hi);
                    }
                }
                fx.release(I64, bound);
                fx.release(I64, i);
                if let Some(m) = mark {
                    fx.release(I32, m);
                }
            }
            Viv::Break => {
                let l = fx.loops.last().ok_or("internal: `break` outside a loop")?.brk;
                fx.br(l);
            }
            Viv::Continue => {
                let l = fx.loops.last().ok_or("internal: `continue` outside a loop")?.cont;
                fx.br(l);
            }
            Viv::Block => {
                for &c in k {
                    self.stmt(fx, c)?;
                }
            }
            Viv::Show { label, .. } | Viv::ShowCplx { label } | Viv::ShowBool { label } | Viv::ShowCArr { label } => {
                let at = self.u.string(label);
                fx.string(at);
                let t = self.expr(fx, k[0])?;
                fx.rt(match t {
                    Ty::Num => "vrt_show_num",
                    Ty::Bool => "vrt_show_bool",
                    Ty::Cplx => "vrt_show_cplx",
                    _ => "vrt_show_arr",
                });
            }
            Viv::Narrate { msg } => {
                let at = self.u.string(msg);
                fx.string(at);
                fx.rt("vrt_narrate");
            }
            Viv::TestBegin { name } => {
                let at = self.u.string(name);
                fx.string(at);
                fx.rt("vrt_test_begin");
            }
            Viv::TestEnd => fx.rt("vrt_test_end"),
            Viv::StepMark { enter, id: step, depth, def, .. } => {
                let (enter, step, depth, def) = (*enter, step.clone(), *depth, def.clone());
                self.step_event(fx, enter, &step, depth, def.as_deref());
            }
            Viv::StepsFlush => fx.rt("vrt_steps_flush"),
            Viv::StoryCall { def } => self.story_mark(fx, "call", def),
            Viv::StoryRet { def } => self.story_mark(fx, "ret", def),
            Viv::StoryIterEnd { def } => self.story_mark(fx, "iter_end", def),
            Viv::StoryIter { def } => {
                let at = self.u.string(def);
                fx.string(at);
                self.expr(fx, k[0])?;
                fx.rt("vrt_story_iter");
            }
            Viv::StoryWatch { def, at, names } => {
                fx.rt("vrt_story_watch_begin");
                for (j, (name, _)) in names.iter().enumerate() {
                    let s = self.u.string(name);
                    fx.string(s);
                    let t = self.expr(fx, k[j])?;
                    fx.rt(match t {
                        Ty::Bool => "vrt_story_w_bool",
                        Ty::Arr(_) => "vrt_story_w_arr",
                        _ => "vrt_story_w_num",
                    });
                }
                let (d, a) = (self.u.string(def), self.u.string(at));
                fx.string(d);
                fx.string(a);
                fx.rt("vrt_story_watch_end");
            }
            Viv::StoryHold { def, at, sentence, near } => {
                let (d, a, s) = (self.u.string(def), self.u.string(at), self.u.string(sentence));
                match near {
                    None => {
                        let ok = fx.tmp(I32);
                        self.cond(fx, k[0])?;
                        fx.set(ok);
                        fx.string(d);
                        fx.string(a);
                        fx.get(ok);
                        fx.string(s);
                        fx.i32c(0);
                        fx.f64c(0.0);
                        fx.release(I32, ok);
                    }
                    Some(arr) => {
                        let (w, t) = (fx.tmp(F64), fx.tmp(F64));
                        self.expr(fx, k[0])?;
                        if *arr {
                            fx.rt("vrt_story_maxabs");
                        } else {
                            fx.op(0x99);
                        }
                        fx.set(w);
                        self.expr(fx, k[1])?;
                        fx.set(t);
                        fx.string(d);
                        fx.string(a);
                        fx.get(w);
                        fx.get(t);
                        fx.op(0x65);
                        fx.string(s);
                        fx.i32c(1);
                        fx.get(w);
                        fx.release(F64, w);
                        fx.release(F64, t);
                    }
                }
                fx.rt("vrt_story_hold");
            }
            Viv::StoryFlush => fx.rt("vrt_story_flush"),
            Viv::Placeholder { def } => {
                let at = self.u.string(def);
                fx.string(at);
                fx.rt("vrt_placeholder");
                fx.op(0x00);
            }
            Viv::Stop { message, .. } => {
                let at = self.u.string(message);
                fx.string(at);
                fx.rt("vrt_stop");
                fx.op(0x00);
            }
            other => return Err(format!("internal: {other:?} in statement position")),
        }
        Ok(())
    }

    /// For a loop whose body can allocate: the arena's height before the loop,
    /// in a fresh local. Every iteration's end releases back to it, keeping
    /// what the loop's variables hold -- so a variable's previous value, filed
    /// by an earlier iteration and now unreachable, goes too.
    fn loop_mark(&mut self, fx: &mut Fx, lp: NodeId) -> R<Option<u32>> {
        if !self.loop_allocates(fx, lp)? {
            return Ok(None);
        }
        let m = fx.tmp(I32);
        fx.rt("vrt_mark");
        fx.set(m);
        Ok(Some(m))
    }

    /// For a loop whose body can allocate, the variables whose arrays and
    /// records an iteration must keep: those the body assigns (anything it
    /// allocated and did not assign is garbage when the iteration ends).
    /// `None` for a body that allocates nothing, which pays nothing.
    fn loop_keeps(&self, fx: &Fx, lp: NodeId) -> R<Option<Vec<(String, Ty)>>> {
        let body = self.kids(lp)[1];
        if !self.loop_allocates(fx, lp)? {
            return Ok(None);
        }
        let mut names = Vec::new();
        self.assigned(body, &mut names);
        let mut keeps = Vec::new();
        for n in names {
            let t = self.var_ty(fx, &n)?;
            if crate::builtins::heap_ty(t) {
                keeps.push((n, t));
            }
        }
        Ok(Some(keeps))
    }

    fn assigned(&self, id: NodeId, out: &mut Vec<String>) {
        if let Viv::Assign { name } = &self.node(id).kind {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        // a write into a record or an array of records changes the variable
        // at the root of its place
        if let Viv::SetField { .. } | Viv::RecSetIndex { .. } = &self.node(id).kind {
            if let Some(name) = crate::lower::place_var(self.ir, self.kids(id)[0]) {
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        for &c in self.kids(id) {
            self.assigned(c, out);
        }
    }

    /// Whether a pass of the loop `lp` can allocate: its body, or, for a
    /// `while`, its condition, which runs once a pass too.
    fn loop_allocates(&self, fx: &Fx, lp: NodeId) -> R<bool> {
        let k = self.kids(lp);
        if self.allocates(fx, k[1])? {
            return Ok(true);
        }
        Ok(matches!(self.node(lp).kind, Viv::While) && self.allocates(fx, k[0])?)
    }

    /// Whether evaluating `id` can allocate an array, a record, a string or
    /// a map.
    fn allocates(&self, fx: &Fx, id: NodeId) -> R<bool> {
        let n = self.node(id);
        let here = match &n.kind {
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
            Viv::Call { callee } => crate::builtins::heap_ty(self.call_ty(fx, callee, &n.children)?),
            Viv::ApplyFn { sig } => sig.ret == ArgKind::Arr,
            _ => false,
        };
        if here {
            return Ok(true);
        }
        for &c in &n.children {
            if self.allocates(fx, c)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The end of an iteration: keep what the loop's variables hold (a
    /// record's block and, through its fields, its arrays), free the rest of
    /// what the iteration allocated.
    fn release_iteration(&mut self, fx: &mut Fx, mark: u32, keeps: &[(String, Ty)]) -> R<()> {
        for (name, t) in keeps {
            self.read_var(fx, name)?;
            self.keep_value(fx, *t, mark)?;
        }
        fx.get(mark);
        fx.rt("vrt_release_kept");
        Ok(())
    }

    /// Keep the value on the stack (an array or record pointer); pops it.
    fn keep_value(&mut self, fx: &mut Fx, t: Ty, mark: u32) -> R<()> {
        match t {
            // an enum value: every node it reaches, through its descriptor
            Ty::Enum(n) => {
                let desc = self.u.enums.get(n).map(|e| e.1).ok_or_else(|| format!("internal: no enum `{n}`"))?;
                let p = fx.tmp(I32);
                fx.set(p);
                fx.gget(G_STR);
                fx.i32c(desc as i64);
                fx.get(p);
                fx.release(I32, p);
                fx.get(mark);
                fx.rt("vrt_keep_en");
            }
            Ty::Str => fx.rt("vrt_keep_str"),
            Ty::Arr(Elem::Str) => fx.rt("vrt_keep_strs"),
            // a map is a block on the arena too (interp/wasm_rt.rs, `map_handle`)
            Ty::Arr(_) | Ty::Map(..) => fx.rt("vrt_keep"),
            // an array of records: its block, and each record and what it holds
            Ty::Recs(r) => {
                let h = self.rec_helper(r, 3)?;
                if self.holds_enum(t) {
                    fx.get(mark);
                }
                fx.call(h);
            }
            Ty::Record(r) => {
                let p = fx.tmp(I32);
                fx.tee(p);
                // a record's value is its data; its header is just before
                fx.i32c(H_DATA as i64);
                fx.op(0x6b);
                fx.rt("vrt_keep");
                let fields: Vec<(Ty, u32)> = self.u.records.get(r).map(|l| l.fields.iter().map(|f| (f.1, f.2)).collect()).unwrap_or_default();
                for (ft, off) in fields {
                    if crate::builtins::heap_ty(ft) {
                        fx.get(p);
                        fx.load_i32(off);
                        self.keep_value(fx, ft, mark)?;
                    }
                }
                fx.release(I32, p);
            }
            _ => fx.op(0x1a),
        }
        Ok(())
    }

    fn block(&mut self, fx: &mut Fx, id: NodeId) -> R<()> {
        for &c in self.kids(id) {
            self.stmt(fx, c)?;
        }
        Ok(())
    }

    /// A step event; `def` is the def the step was written in, or None for
    /// the def being emitted.
    fn step_event(&mut self, fx: &mut Fx, enter: bool, step: &str, depth: u32, def: Option<&str>) {
        let def = def.map(str::to_string).unwrap_or_else(|| fx.step_def.clone());
        fx.i32c(enter as i64);
        let (d, s) = (self.u.string(&def), self.u.string(step));
        fx.string(d);
        fx.string(s);
        fx.i32c(depth as i64);
        fx.rt("vrt_step");
    }

    fn story_mark(&mut self, fx: &mut Fx, kind: &str, def: &str) {
        let (kd, d) = (self.u.string(kind), self.u.string(def));
        fx.string(kd);
        fx.string(d);
        fx.rt("vrt_story_mark");
    }

    /// Return the value held in `slots`: the story's `ret` event, then the
    /// arena released (keeping a returned array), as the C backend orders them.
    fn ret(&mut self, fx: &mut Fx, slots: &[(VT, u32)]) -> R<()> {
        if let Some(def) = fx.story_def.clone() {
            self.story_mark(fx, "ret", &def);
        }
        if let Some(m) = fx.mark {
            match fx.ret {
                Ty::Record(_) | Ty::Recs(_) | Ty::Str | Ty::Arr(Elem::Str) | Ty::Enum(_) => {
                    // a record keeps its block and, through its fields, its
                    // arrays and strings; a string its block; a list of
                    // strings its block and chunks; the rest is freed
                    let t = fx.ret;
                    fx.get(slots[0].1);
                    self.keep_value(fx, t, m)?;
                    fx.get(m);
                    fx.rt("vrt_release_kept");
                }
                Ty::Arr(_) | Ty::Map(..) => {
                    fx.get(m);
                    fx.get(slots[0].1);
                    fx.rt("vrt_release");
                }
                _ => {
                    fx.get(m);
                    fx.i32c(0);
                    fx.rt("vrt_release");
                }
            }
        }
        fx.unstash(slots);
        fx.op(0x0f);
        Ok(())
    }

    // ---- functions ----------------------------------------------------------

    fn function(&mut self, id: NodeId) -> R<Vec<u8>> {
        let n = self.node(id);
        let Viv::Func { name, src, params, ret, locals, arena } = &n.kind else { unreachable!() };
        crate::lower::intidx::enter(self.ir, Some(name.clone()));
        let mut fx = Fx::new(params, *ret, src);
        fx.dj = self.u.dj_defs.contains(name);
        if fx.dj || pinned_under(self.ir, id) {
            let ps: Vec<String> = params.iter().map(|(p, _)| p.clone()).collect();
            fx.dj_fresh = crate::lower::dj_fresh(self.ir, id, &ps);
            fx.dj_params = ps.into_iter().collect();
            fx.dj_pinnable = self.u.dj_pinnable.get(name).cloned().unwrap_or_default();
        }
        for (ln, lt) in locals {
            fx.declare(ln, *lt);
        }
        if let Some(Viv::StoryCall { def }) = n.children.first().map(|&c| &self.node(c).kind) {
            fx.story_def = Some(def.clone());
        }
        // A function that can allocate frees, on the way out, everything it
        // allocated except what it returns: the array, or the record and
        // every array in it.
        // (a def whose body can allocate nothing frees nothing: no mark)
        let mut allocates = false;
        if *arena {
            for &c in &n.children {
                allocates = allocates || self.allocates(&fx, c)?;
            }
        }
        if allocates {
            let m = fx.local(I32);
            fx.rt("vrt_mark");
            fx.set(m);
            fx.mark = Some(m);
        }
        self.cache_headers(&mut fx, &n.children, false)?;
        for &c in &n.children {
            self.stmt(&mut fx, c)?;
        }
        // falling off the end
        if let Some(def) = fx.story_def.clone() {
            self.story_mark(&mut fx, "ret", &def);
        }
        if let Some(m) = fx.mark {
            fx.get(m);
            fx.i32c(0);
            fx.rt("vrt_release");
        }
        fx.zero(*ret);
        fx.op(0x0b);
        Ok(finish(fx))
    }

    fn lambda(&mut self, id: NodeId) -> R<Vec<u8>> {
        let n = self.node(id);
        let Viv::LambdaDef { param, captures, .. } = &n.kind else { unreachable!() };
        crate::lower::intidx::enter(self.ir, None);
        let mut params = vec![(param.clone(), Ty::Num)];
        params.extend(captures.iter().cloned());
        let mut fx = Fx::new(&params, Ty::Num, "");
        let t = self.expr(&mut fx, n.children[0])?;
        if t != Ty::Num {
            return Err(format!("internal: a lambda body of type {t:?}"));
        }
        fx.op(0x0b);
        Ok(finish(fx))
    }

    /// The table entry for a def used as a value: the function-value signature
    /// in, the def called directly, its result passed back.
    fn wrapper(&mut self, id: NodeId) -> R<Vec<u8>> {
        let Viv::FnWrapDef { name, sig } = &self.node(id).kind else { unreachable!() };
        crate::lower::intidx::enter(self.ir, None);
        let (f, params, ret) = self.u.funcs.get(name).cloned().ok_or_else(|| format!("internal: no def `{name}`"))?;
        let mut fx = Fx::new(&[], Ty::Num, "");
        let (ps, _) = sig_type(*sig);
        fx.n_param_slots = ps.len() as u32;
        for (j, (_, pt)) in params.iter().enumerate() {
            let want = parts(kind_ty(sig.kind(j)));
            if parts(*pt) != want {
                return Err(unsupported(&format!("`{name}` as a function value (its parameter {} is a {pt:?})", j + 1)));
            }
            fx.get(j as u32);
        }
        fx.call(f);
        match (ret, sig.ret) {
            (Ty::Num | Ty::Bool, ArgKind::Num) | (Ty::Arr(Elem::Num), ArgKind::Arr) => {}
            (Ty::Void, ArgKind::Num) => fx.f64c(0.0),
            _ => return Err(unsupported(&format!("`{name}` as a function value (it returns a {ret:?})"))),
        }
        fx.op(0x0b);
        Ok(finish(fx))
    }

    fn main(&mut self, id: Option<NodeId>) -> R<Vec<u8>> {
        let mut fx = Fx::new(&[], Ty::Void, crate::parser::MAIN_DEF);
        crate::lower::intidx::enter(self.ir, Some(crate::parser::MAIN_DEF.to_string()));
        if let Some(id) = id {
            if pinned_under(self.ir, id) {
                fx.dj_fresh = crate::lower::dj_fresh(self.ir, id, &[]);
            }
            let n = self.node(id);
            let Viv::Main { locals, .. } = &n.kind else { unreachable!() };
            for (ln, lt) in locals {
                fx.declare(ln, *lt);
            }
            self.cache_headers(&mut fx, &n.children, true)?;
            for &c in &n.children {
                self.stmt(&mut fx, c)?;
                fx.call(host_flush());
            }
        }
        fx.op(0x0b);
        // The prologue goes in front last, when the string table is complete:
        // a fresh runtime, then the table copied into memory the runtime owns
        // (`memory.init` from the passive data segment).
        let body = std::mem::take(&mut fx.code);
        fx.i32c(self.u.story as i64);
        fx.rt("vrt_init");
        fx.rt("vrt_scratch");
        fx.gset(G_SCRATCH);
        let len = self.u.strings.len() as i64;
        fx.i32c(len);
        fx.rt("vrt_static");
        fx.gset(G_STR);
        fx.gget(G_STR);
        fx.i32c(0);
        fx.i32c(len);
        fx.code.extend_from_slice(&[0xfc, 0x08, 0x00, 0x00]);
        fx.code.extend_from_slice(&body);
        Ok(finish(fx))
    }
}

/// A function body: its locals declaration, then its code.
fn finish(fx: Fx) -> Vec<u8> {
    let mut body = Vec::new();
    // run-length groups of equal local types
    let mut groups: Vec<(u32, VT)> = Vec::new();
    for &vt in &fx.locals {
        match groups.last_mut() {
            Some((n, t)) if *t == vt => *n += 1,
            _ => groups.push((1, vt)),
        }
    }
    uleb(groups.len() as u64, &mut body);
    for (n, vt) in groups {
        uleb(n as u64, &mut body);
        body.push(vt.byte());
    }
    body.extend_from_slice(&fx.code);
    body
}

/// Compile a lowered Viv program to a wasm module.
pub fn emit_wasm(ir: &IrProgram<Viv>) -> Result<Vec<u8>, String> {
    // the db_* builtins need SQLite, which the browser runtime does not carry
    let mut stack: Vec<knotkit::ir::NodeId> = ir.roots().to_vec();
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let n = ir.get(id);
        if let crate::lower::Viv::Call { callee } = &n.kind {
            if callee.starts_with("viv_s_db_") {
                return Err("error: the db_* builtins (SQLite) are not available in the browser yet: run the program natively (viv run, or compiled to C)".to_string());
            }
        }
        stack.extend(n.children.iter().copied());
    }
    let clock = ir.roots().iter().any(|&r| calls(ir, r, "viv_clock"));
    HOST_CLOCK.with(|c| c.set(clock));
    // pass 1: everything imported, to learn what is called
    IMPORTS.with(|im| *im.borrow_mut() = rt_table().iter().map(|r| r.0).collect());
    USED.with(|u| u.borrow_mut().clear());
    emit_pass(ir)?;
    // pass 2: just that, in `RT`'s order
    let used = USED.with(|u| u.borrow().clone());
    IMPORTS.with(|im| *im.borrow_mut() = rt_table().iter().map(|r| r.0).filter(|n| used.contains(n)).collect());
    emit_pass(ir)
}

fn calls(ir: &IrProgram<Viv>, id: NodeId, callee: &str) -> bool {
    let n = ir.get(id);
    matches!(&n.kind, Viv::Call { callee: c } if c == callee) || n.children.iter().any(|&c| calls(ir, c, callee))
}

/// The descriptors of the program's enums, in the string table: for each
/// enum, u32 words (little endian): the number of variants, then for each
/// variant its name's offset and length in the table, its number of payload
/// fields, and each field's kind (0 a number, 1 an Int, 2 a boolean, 3 a
/// string, 4 an enum, followed by that enum's descriptor offset). The runtime
/// reads them to write a value as text and to keep its nodes
/// (`interp/wasm_rt.rs`, `en_text` and `vrt_keep_en`).
fn enum_descriptors(u: &mut Unit, enums: &[(String, Vec<(String, Vec<Ty>)>)]) {
    let size = |vs: &Vec<(String, Vec<Ty>)>| 4 + vs.iter().map(|(_, p)| 12 + p.iter().map(|t| if matches!(t, Ty::Enum(_)) { 8 } else { 4 }).sum::<usize>()).sum::<usize>();
    let mut offs = HashMap::new();
    for (name, vs) in enums {
        while u.strings.len() % 4 != 0 {
            u.strings.push(0);
        }
        offs.insert(name.clone(), u.strings.len() as u32);
        u.strings.extend(std::iter::repeat(0u8).take(size(vs)));
    }
    for (name, vs) in enums {
        let mut words: Vec<u32> = vec![vs.len() as u32];
        for (v, p) in vs {
            let (o, l) = u.string(v);
            words.extend([o, l, p.len() as u32]);
            for t in p {
                match t {
                    Ty::Num => words.push(0),
                    Ty::Int => words.push(1),
                    Ty::Bool => words.push(2),
                    Ty::Str => words.push(3),
                    Ty::Enum(n) => words.extend([4, offs[*n]]),
                    _ => words.push(0),
                }
            }
        }
        let at = offs[name] as usize;
        for (k, w) in words.iter().enumerate() {
            u.strings[at + 4 * k..at + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
        u.enums.insert(name.clone(), (vs.clone(), offs[name]));
    }
}

fn emit_pass(ir: &IrProgram<Viv>) -> Result<Vec<u8>, String> {
    let mut u = Unit {
        funcs: HashMap::new(),
        lambdas: HashMap::new(),
        globals: HashMap::new(),
        records: HashMap::new(),
        enums: HashMap::new(),
        slots: HashMap::new(),
        table: Vec::new(),
        helpers: HashMap::new(),
        helper_code: Vec::new(),
        fn_types: Vec::new(),
        types: Types::default(),
        strings: Vec::new(),
        string_at: HashMap::new(),
        str_objs: HashMap::new(),
        consts: Vec::new(),
        story: false,
        dj_defs: crate::lower::dj_reached(ir),
        dj_body: 0,
        dj_range: None,
        depth_g: None,
        dj_pins: Vec::new(),
        dj_pinnable: crate::lower::dj_pinnable(ir),
    };
    // the empty string at offset 0, so an empty label is never a special case
    u.string("");

    // globals 0 and 1 are the string table and the scratch pair
    let mut n_globals = 2u32;
    let mut global_types: Vec<VT> = vec![I32, I32];
    let mut defs = Vec::new();
    let mut main = None;
    for &r in ir.roots() {
        let n = ir.get(r);
        match &n.kind {
            Viv::Global { name, ty } => {
                u.globals.insert(name.clone(), (n_globals, *ty));
                for vt in parts(*ty) {
                    global_types.push(vt);
                    n_globals += 1;
                }
            }
            Viv::EnumDefs { enums } => enum_descriptors(&mut u, enums),
            Viv::RecordDef { name, fields } => {
                let mut off = 0;
                let mut fs = Vec::new();
                for (f, t) in fields {
                    fs.push((f.clone(), *t, off));
                    off += if *t == Ty::Cplx { 16 } else { 8 };
                }
                u.records.insert(name.clone(), Layout { fields: fs, bytes: off.max(8) });
            }
            Viv::Func { name, params, ret, .. } => {
                let ps = params.iter().flat_map(|(_, t)| parts(*t)).collect();
                let idx = u.new_fn(ps, parts(*ret));
                u.funcs.insert(name.clone(), (idx, params.clone(), *ret));
                defs.push(r);
            }
            Viv::LambdaDef { name, param, captures } => {
                let idx = u.new_fn(vec![F64; 1 + captures.len()], vec![F64]);
                u.lambdas.insert(name.clone(), (idx, param.clone(), captures.clone()));
                defs.push(r);
            }
            Viv::FnWrapDef { name, sig } => {
                let (p, res) = sig_type(*sig);
                let idx = u.new_fn(p, res);
                u.slots.insert(name.clone(), u.table.len() as u32);
                u.table.push(idx);
                defs.push(r);
            }
            Viv::Main { .. } => main = Some(r),
            // the C helpers of a record type; this backend makes its own
            // (`Em::rec_helper`) when they are used
            Viv::RecFns { .. } => {}
            other => return Err(format!("internal: {other:?} as a root")),
        }
    }
    // a program with a pinned loop: the two globals of its arrays' range
    if ir.roots().iter().any(|&r| pinned_under(ir, r)) {
        u.dj_range = Some((n_globals, n_globals + 1));
        global_types.push(I32);
        global_types.push(I32);
        n_globals += 2;
    }
    // a program with a recursive def: its depth counter, an i64 global
    if u.globals.contains_key(&format!("{}depth", crate::parser::HIDDEN_PREFIX)) {
        u.depth_g = Some(n_globals);
        global_types.push(I64);
        n_globals += 1;
    }
    // a program carries a debug lens when anything records into the story
    u.story = ir.roots().iter().any(|&r| story_under(ir, r));
    let run = u.new_fn(vec![], vec![]);

    let mut em = Em { ir, u };
    let mut bodies: Vec<(u32, Vec<u8>)> = Vec::new();
    for &d in &defs {
        let body = match &ir.get(d).kind {
            Viv::Func { name, .. } => {
                let f = em.u.funcs[name].0;
                (f, em.function(d).map_err(|e| format!("in `{name}`: {e}"))?)
            }
            Viv::LambdaDef { name, .. } => (em.u.lambdas[name].0, em.lambda(d)?),
            Viv::FnWrapDef { name, .. } => {
                let slot = em.u.slots[name];
                (em.u.table[slot as usize], em.wrapper(d)?)
            }
            _ => unreachable!(),
        };
        bodies.push(body);
    }
    let run_body = em.main(main)?;
    bodies.push((run, run_body));
    for (idx, _t, body) in std::mem::take(&mut em.u.helper_code) {
        bodies.push((idx, body));
    }
    bodies.sort_by_key(|b| b.0);
    let u = em.u;

    // ---- the module ----
    let mut out = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

    // types: the runtime's, then everything the module's own functions use
    let mut types = u.types;
    let imports: Vec<&(&str, &[VT], &[VT])> = IMPORTS.with(|im| im.borrow().iter().map(|n| rt_table().iter().find(|r| r.0 == *n).expect("a runtime import")).collect());
    let rt_types: Vec<u32> = imports.iter().map(|(_, p, r)| types.idx(p.to_vec(), r.to_vec())).collect();
    let t_void = types.idx(vec![], vec![]);
    let t_clock = types.idx(vec![], vec![F64]);
    let clock = HOST_CLOCK.with(|c| c.get());
    let mut sec = Vec::new();
    uleb(types.list.len() as u64, &mut sec);
    for (p, r) in &types.list {
        sec.push(0x60);
        uleb(p.len() as u64, &mut sec);
        sec.extend(p.iter().map(|v| v.byte()));
        uleb(r.len() as u64, &mut sec);
        sec.extend(r.iter().map(|v| v.byte()));
    }
    section(1, &sec, &mut out);

    // imports: the runtime's functions, host.flush, and the runtime's memory
    let mut sec = Vec::new();
    uleb(imports.len() as u64 + 2 + clock as u64, &mut sec);
    for (i, (name, _, _)) in imports.iter().enumerate() {
        wname("rt", &mut sec);
        wname(name, &mut sec);
        sec.push(0x00);
        uleb(rt_types[i] as u64, &mut sec);
    }
    wname("host", &mut sec);
    wname("flush", &mut sec);
    sec.push(0x00);
    uleb(t_void as u64, &mut sec);
    if clock {
        wname("host", &mut sec);
        wname("clock", &mut sec);
        sec.push(0x00);
        uleb(t_clock as u64, &mut sec);
    }
    wname("rt", &mut sec);
    wname("memory", &mut sec);
    sec.push(0x02);
    sec.push(0x00);
    uleb(1, &mut sec);
    section(2, &sec, &mut out);

    // functions
    let mut sec = Vec::new();
    uleb(u.fn_types.len() as u64, &mut sec);
    for &t in &u.fn_types {
        uleb(t as u64, &mut sec);
    }
    section(3, &sec, &mut out);

    // the table of function values, when there are any
    if !u.table.is_empty() {
        let mut sec = Vec::new();
        uleb(1, &mut sec);
        sec.push(0x70);
        sec.push(0x00);
        uleb(u.table.len() as u64, &mut sec);
        section(4, &sec, &mut out);
    }

    // globals
    let mut sec = Vec::new();
    uleb(global_types.len() as u64, &mut sec);
    for vt in &global_types {
        sec.push(vt.byte());
        sec.push(0x01);
        match vt {
            I32 => {
                sec.push(0x41);
                sec.push(0x00);
            }
            I64 => {
                sec.push(0x42);
                sec.push(0x00);
            }
            F64 => {
                sec.push(0x44);
                sec.extend_from_slice(&0f64.to_le_bytes());
            }
            F32 => {
                sec.push(0x43);
                sec.extend_from_slice(&0f32.to_le_bytes());
            }
        }
        sec.push(0x0b);
    }
    section(6, &sec, &mut out);

    // exports
    let mut sec = Vec::new();
    uleb(1, &mut sec);
    wname("run", &mut sec);
    sec.push(0x00);
    uleb(run as u64, &mut sec);
    section(7, &sec, &mut out);

    // elements: the table, filled at offset 0
    if !u.table.is_empty() {
        let mut sec = Vec::new();
        uleb(1, &mut sec);
        sec.push(0x00);
        sec.push(0x41);
        sec.push(0x00);
        sec.push(0x0b);
        uleb(u.table.len() as u64, &mut sec);
        for &f in &u.table {
            uleb(f as u64, &mut sec);
        }
        section(9, &sec, &mut out);
    }

    // data count (memory.init needs it before the code)
    let mut sec = Vec::new();
    uleb(2, &mut sec);
    section(12, &sec, &mut out);

    // code
    let mut sec = Vec::new();
    uleb(bodies.len() as u64, &mut sec);
    for (_, b) in &bodies {
        uleb(b.len() as u64, &mut sec);
        sec.extend_from_slice(b);
    }
    section(10, &sec, &mut out);

    // data, both passive: the string table (0), the constant tables (1)
    let mut sec = Vec::new();
    uleb(2, &mut sec);
    for seg in [&u.strings, &u.consts] {
        sec.push(0x01);
        uleb(seg.len() as u64, &mut sec);
        sec.extend_from_slice(seg);
    }
    section(11, &sec, &mut out);

    Ok(out)
}

/// Whether anything under `id` records into the story.
/// Whether a pinned `parallel for` is under `id`.
fn pinned_under(ir: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = ir.get(id);
    matches!(&n.kind, Viv::ParallelFor { pins, .. } if !pins.is_empty()) || n.children.iter().any(|&c| pinned_under(ir, c))
}

fn story_under(ir: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = ir.get(id);
    matches!(
        n.kind,
        Viv::StoryFlush | Viv::StoryCall { .. } | Viv::StoryRet { .. } | Viv::StoryIter { .. } | Viv::StoryWatch { .. } | Viv::StoryHold { .. }
    ) || matches!(n.kind, Viv::StepMark { story: true, .. })
        || n.children.iter().any(|&c| story_under(ir, c))
}
