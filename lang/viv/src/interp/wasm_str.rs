// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Strings in the wasm runtime: the imports a compiled
//! module calls for `viv_s_*` and `viv_ss_*`, over strings in linear memory.
//! Each runs the rule of the same name in `interp::strings`, the one the
//! interpreter runs, so the three engines agree by construction; C has its
//! twin in `strings.c`.
//!
//! Layouts, each in an arena block (the runtime's 32-byte header, width 0):
//!
//! ```text
//! a string (the value points here):  +0 len (bytes)  +4 nch (code points)
//!   +8 owner (the block it is in; 0 for a literal)  +12 ci  +16 cb (index
//!   cache)  +20 the bytes, then a NUL
//! a list of strings:  +0 n  +4 cap  +8 unsealed  +12 last chunk (a block)
//!   +16 cap item pointers
//! a chunk (strings a list holds):  +0 previous chunk  +4 cap  +8 used
//!   +12 (unused)  +16 cap bytes of strings, each 4-aligned
//! ```
//!
//! A literal is an object of the first layout in the module's string table.

use super::super::strings as rules;
use super::*;

const S_BYTES: u32 = 20;
const L_ITEMS: u32 = 16;
const C_DATA: u32 = 16;

// ---- reading and making strings ----------------------------------------------------

/// The text of the string at `v` (0: never assigned, read as "").
pub(crate) fn str_of<'a>(v: u32) -> &'a str {
    if v == 0 {
        return "";
    }
    unsafe {
        let len = rd(v) as usize;
        let b = std::slice::from_raw_parts((v + S_BYTES) as usize as *const u8, len);
        std::str::from_utf8_unchecked(b)
    }
}
fn nch_of(v: u32) -> usize {
    if v == 0 { 0 } else { unsafe { rd(v + 4) as usize } }
}
/// Run `f` on the string at `v` as the rules read it, its index cache kept.
fn with_sv<R>(v: u32, f: impl FnOnce(&rules::SV) -> R) -> R {
    let cache = std::cell::Cell::new(if v == 0 { (0, 0) } else { unsafe { (rd(v + 12) as usize, rd(v + 16) as usize) } });
    let r = f(&rules::SV { s: str_of(v), nch: nch_of(v), cache: &cache });
    if v != 0 {
        let (ci, cb) = cache.get();
        unsafe {
            wr(v + 12, ci as i32);
            wr(v + 16, cb as i32);
        }
    }
    r
}

thread_local! {
    /// The one-character ASCII strings, made once and kept for good.
    static ASCII: RefCell<[u32; 128]> = const { RefCell::new([0; 128]) };
}

/// A string outside the arena, for the rest of the run.
fn static_str(s: &str) -> u32 {
    let bytes = (S_BYTES as usize + s.len() + 1).max(8);
    let v = unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(bytes, 8).expect("a layout")) } as usize as u32;
    fill(v, s, 0);
    v
}

fn fill(v: u32, s: &str, owner: u32) {
    unsafe {
        wr(v, s.len() as i32);
        wr(v + 4, rules::count_chars(s.as_bytes()) as i32);
        wr(v + 8, owner as i32);
        wr(v + 12, 0);
        wr(v + 16, 0);
        std::ptr::copy_nonoverlapping(s.as_ptr(), (v + S_BYTES) as usize as *mut u8, s.len());
        *((v + S_BYTES) as usize as *mut u8).add(s.len()) = 0;
    }
}

/// A new string with this text, on the arena (a one-character ASCII string
/// is the shared one).
pub(crate) fn new_str(s: &str) -> u32 {
    if s.len() == 1 {
        let c = s.as_bytes()[0] as usize;
        return ASCII.with(|t| {
            let mut t = t.borrow_mut();
            if t[c] == 0 {
                t[c] = static_str(s);
            }
            t[c]
        });
    }
    if s.len() > rules::MAX_BYTES {
        fail(rules::too_long(s.len() as u128));
    }
    let p = alloc_with(0, &[], S_BYTES as usize + s.len() + 1, false);
    let v = p + HDR;
    fill(v, s, p);
    v
}

fn made(r: Result<String, String>) -> u32 {
    new_str(&or_fail(r))
}

// ---- lists ---------------------------------------------------------------------------

fn items<'a>(l: u32) -> &'a [u32] {
    if l == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts((l + L_ITEMS) as usize as *const u32, rd(l) as usize) }
}

/// A list with room for `cap` items, on the arena.
fn list_new(cap: usize) -> u32 {
    let cap = cap.max(1);
    let p = alloc_with(0, &[], L_ITEMS as usize + 4 * cap, false);
    let l = p + HDR;
    unsafe {
        wr(l, 0);
        wr(l + 4, cap as i32);
        wr(l + 8, 0);
        wr(l + 12, 0);
    }
    l
}

fn padded(len: usize) -> usize {
    (S_BYTES as usize + len + 1 + 3) & !3
}

/// A new chunk of `cap` bytes, after `prev`.
fn chunk_new(cap: usize, prev: u32) -> u32 {
    let c = alloc_with(0, &[], C_DATA as usize + cap, false);
    unsafe {
        wr(c + HDR, prev as i32);
        wr(c + HDR + 4, cap as i32);
        wr(c + HDR + 8, 0);
    }
    c
}

/// A copy of `s` in l's chunks (a new chunk, twice the last, when it has no
/// room); the item pointer.
fn store(l: u32, s: &str) -> u32 {
    let need = padded(s.len());
    unsafe {
        let mut c = rd(l + 12) as u32;
        if c == 0 || (rd(c + HDR + 4) - rd(c + HDR + 8)) < need as i32 {
            let cap = if c == 0 { 256 } else { 2 * rd(c + HDR + 4) as usize }.max(need);
            c = chunk_new(cap, c);
            wr(l + 12, c as i32);
        }
        let used = rd(c + HDR + 8) as u32;
        let v = c + HDR + C_DATA + used;
        wr(c + HDR + 8, (used as usize + need) as i32);
        fill(v, s, c);
        v
    }
}

/// A new list holding copies of these strings, in one chunk (with room for
/// `room` more items).
pub(crate) fn list_of(texts: &[&str], room: usize) -> u32 {
    let l = list_new(texts.len() + room);
    let bytes: usize = texts.iter().map(|t| padded(t.len())).sum();
    if bytes > 0 {
        let c = chunk_new(if room > 0 { 2 * bytes } else { bytes }, 0);
        unsafe { wr(l + 12, c as i32) };
    }
    for t in texts {
        let v = store(l, t);
        unsafe {
            let n = rd(l);
            wr(l + L_ITEMS + 4 * n as u32, v as i32);
            wr(l, n + 1);
        }
    }
    l
}

fn texts_of<'a>(l: u32) -> Vec<&'a str> {
    items(l).iter().map(|&v| str_of(v)).collect()
}

/// `a = append(a, s)` / `extend`: in place when `own` and the list is
/// unsealed (its block moved by a realloc when it needs room), else a copy
/// with room, unsealed (strings.c `viv_ss_grow`).
fn grow(l: u32, add: &[&str], own: f64) -> u32 {
    let l = if l == 0 { list_new(1) } else { l };
    unsafe {
        if own != 0.0 && rd(l + 8) != 0 {
            let (n, cap) = (rd(l) as usize, rd(l + 4) as usize);
            let mut l = l;
            if n + add.len() > cap {
                let newcap = (2 * cap).max(n + add.len());
                let p = l - HDR;
                let old = bytes_of(0, rd(p + 4));
                let size = L_ITEMS as usize + 4 * newcap;
                let q = std::alloc::realloc(p as usize as *mut u8, layout(old), bytes_of(0, size as i32)) as usize as u32;
                if q == 0 {
                    fail("out of memory growing a list of strings".into());
                }
                RT.with(|r| {
                    let mut r = r.borrow_mut();
                    let arena = &mut r.as_mut().expect("vrt_init runs first").arena;
                    match arena.iter().rposition(|&x| x == p) {
                        Some(i) => arena[i] = q,
                        None => arena.push(q),
                    }
                });
                wr(q, (q + HDR) as i32);
                wr(q + 4, size as i32);
                l = q + HDR;
                wr(l + 4, newcap as i32);
            }
            for t in add {
                let v = store(l, t);
                let n = rd(l);
                wr(l + L_ITEMS + 4 * n as u32, v as i32);
                wr(l, n + 1);
            }
            return l;
        }
    }
    let mut all = texts_of(l);
    let owned: Vec<String> = add.iter().map(|s| s.to_string()).collect();
    all.extend(owned.iter().map(|s| s.as_str()));
    let room = all.len().max(8);
    let r = list_of(&all, room);
    unsafe { wr(r + 8, 1) };
    r
}

// ---- keeping ---------------------------------------------------------------------------

/// Keep what a string variable holds: the block it is in.
#[no_mangle]
pub extern "C" fn vrt_keep_str(v: u32) {
    if v != 0 {
        let owner = unsafe { rd(v + 8) } as u32;
        if owner != 0 {
            vrt_keep(owner);
        }
    }
}
/// Keep a list: its block and its chunks.
#[no_mangle]
pub extern "C" fn vrt_keep_strs(l: u32) {
    if l == 0 {
        return;
    }
    vrt_keep(l - HDR);
    let mut c = unsafe { rd(l + 12) } as u32;
    while c != 0 {
        vrt_keep(c);
        c = unsafe { rd(c + HDR) } as u32;
    }
}

// ---- the runtime calls ------------------------------------------------------------------

fn b(x: bool) -> f64 {
    if x { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn vrt_s_cat(a: u32, c: u32) -> u32 {
    made(rules::concat(str_of(a), str_of(c)))
}
#[no_mangle]
pub extern "C" fn vrt_s_rep(s: u32, n: f64) -> u32 {
    made(rules::repeat(str_of(s), n))
}
#[no_mangle]
pub extern "C" fn vrt_s_eq(a: u32, c: u32) -> f64 {
    b(str_of(a) == str_of(c))
}
#[no_mangle]
pub extern "C" fn vrt_s_cmp(a: u32, c: u32) -> f64 {
    rules::cmp(str_of(a), str_of(c)) as i8 as f64
}
#[no_mangle]
pub extern "C" fn vrt_s_len(s: u32) -> f64 {
    nch_of(s) as f64
}
#[no_mangle]
pub extern "C" fn vrt_s_at(s: u32, i: f64) -> u32 {
    let (x, y) = or_fail(with_sv(s, |v| rules::char_at(v, i)));
    new_str(&str_of(s)[x..y])
}
#[no_mangle]
pub extern "C" fn vrt_s_slice(s: u32, kind: f64, lo: f64, hi: f64, step: f64) -> u32 {
    made(with_sv(s, |v| rules::slice(v, [kind, lo, hi, step])))
}
#[no_mangle]
pub extern "C" fn vrt_ss_len(l: u32) -> f64 {
    items(l).len() as f64
}
#[no_mangle]
pub extern "C" fn vrt_ss_at(l: u32, i: f64) -> u32 {
    or_fail(crate::interp::index_int(i));
    let it = items(l);
    match rules::py_index(it.len(), i) {
        Some(k) => it[k],
        None => fail("list index out of range".into()),
    }
}
#[no_mangle]
pub extern "C" fn vrt_ss_slice(l: u32, kind: f64, lo: f64, hi: f64, step: f64) -> u32 {
    let ks = or_fail(rules::slice_indices(items(l).len(), [kind, lo, hi, step]));
    let t = texts_of(l);
    list_of(&ks.iter().map(|&k| t[k]).collect::<Vec<_>>(), 0)
}
#[no_mangle]
pub extern "C" fn vrt_ss_cat(a: u32, c: u32) -> u32 {
    let mut t = texts_of(a);
    t.extend(texts_of(c));
    list_of(&t, 0)
}
#[no_mangle]
pub extern "C" fn vrt_ss_eq(a: u32, c: u32) -> f64 {
    b(texts_of(a) == texts_of(c))
}
#[no_mangle]
pub extern "C" fn vrt_ss_new() -> u32 {
    list_new(1)
}
#[no_mangle]
pub extern "C" fn vrt_ss_append(l: u32, s: u32, own: f64) -> u32 {
    let t = str_of(s).to_string();
    grow(l, &[&t], own)
}
#[no_mangle]
pub extern "C" fn vrt_ss_extend(l: u32, m: u32, own: f64) -> u32 {
    let t: Vec<String> = texts_of(m).iter().map(|s| s.to_string()).collect();
    grow(l, &t.iter().map(|s| s.as_str()).collect::<Vec<_>>(), own)
}
#[no_mangle]
pub extern "C" fn vrt_ss_seal(l: u32) -> u32 {
    if l != 0 {
        unsafe { wr(l + 8, 0) };
    }
    l
}
#[no_mangle]
pub extern "C" fn vrt_s_str_bool(x: f64) -> u32 {
    new_str(if x != 0.0 { "true" } else { "false" })
}
#[no_mangle]
pub extern "C" fn vrt_s_str_cplx(re: f64, im: f64) -> u32 {
    new_str(&super::super::format_cplx(re, im))
}
/// The format the compiled program passes as three numbers: the kind word
/// (the kind's letter and the flags, `Fmt::word`), the precision and the
/// width. A width or precision that came from a value is checked here.
fn fmt_of(kind: f64, prec: f64, width: f64) -> crate::printfmt::Fmt {
    let f = crate::printfmt::Fmt::of_word(kind as u32, 0, 0).unwrap_or(crate::printfmt::Fmt::REPR);
    let star = crate::printfmt::FL_STAR_WIDTH | crate::printfmt::FL_STAR_PREC;
    match (crate::printfmt::Fmt { flags: f.flags | star, ..f }).with_stars(&[width, prec]) {
        Ok(f) => crate::printfmt::Fmt { flags: f.flags & !star, ..f },
        Err(m) => fail(m),
    }
}
#[no_mangle]
pub extern "C" fn vrt_s_fmt_num(x: f64, kind: f64, prec: f64, width: f64) -> u32 {
    new_str(&crate::printfmt::number(x, fmt_of(kind, prec, width)))
}
#[no_mangle]
pub extern "C" fn vrt_s_fmt_arr(a: u32, kind: f64, prec: f64, width: f64) -> u32 {
    new_str(&crate::printfmt::array(&shape_of(a), data_of(a), fmt_of(kind, prec, width)))
}
#[no_mangle]
pub extern "C" fn vrt_s_fmt_str(s: u32, fill: f64, align: f64, width: f64, prec: f64) -> u32 {
    let mut spec = rules::StrSpec::decode([fill, align, width, prec]);
    spec.stars = 1 | if prec >= 0.0 { 2 } else { 0 };
    let spec = match spec.with_stars(&[width, prec]) {
        Ok(s) => s,
        Err(m) => fail(m),
    };
    new_str(&with_sv(s, |v| rules::format_str(v, spec)))
}
#[no_mangle]
pub extern "C" fn vrt_s_stop(s: u32) -> f64 {
    fail(str_of(s).to_string())
}

thread_local! {
    /// The parts of an f-string, or the items of a list literal, staged
    /// before the call that joins them.
    static STAGED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}
#[no_mangle]
pub extern "C" fn vrt_s_push(v: u32) {
    STAGED.with(|s| s.borrow_mut().push(v));
}
fn take_staged() -> Vec<u32> {
    STAGED.with(|s| std::mem::take(&mut *s.borrow_mut()))
}
#[no_mangle]
pub extern "C" fn vrt_s_catn() -> u32 {
    let parts = take_staged();
    let total: usize = parts.iter().map(|&p| str_of(p).len()).sum();
    if total > rules::MAX_BYTES {
        fail(rules::too_long(total as u128));
    }
    let mut out = String::with_capacity(total);
    for p in parts {
        out.push_str(str_of(p));
    }
    new_str(&out)
}
#[no_mangle]
pub extern "C" fn vrt_ss_of() -> u32 {
    let parts = take_staged();
    list_of(&parts.iter().map(|&p| str_of(p)).collect::<Vec<_>>(), 0)
}
/// A string's text as a bridge argument (a `print` part, a figure's title).
#[no_mangle]
pub extern "C" fn vrt_arg_s(v: u32) {
    stage(Arg::Str(str_of(v).to_string()));
}

/// The registry's builtins: each import takes its entry's arguments (a
/// string or a list as its pointer, a number as an f64) and runs the
/// interpreter's arm for it on values made from them.
fn registry(c: &str, args: &[u32], nums: &[f64]) -> Val {
    let b = rules::BUILTINS.iter().find(|b| b.c == c).expect("a registry builtin");
    let (mut ai, mut ni) = (0, 0);
    let vals: Vec<Val> = b
        .args
        .iter()
        .map(|k| match k {
            rules::K::S => {
                ai += 1;
                rules::sval(str_of(args[ai - 1]).to_string())
            }
            rules::K::L => {
                ai += 1;
                rules::strs(items(args[ai - 1]).iter().map(|&v| rules::Str::new(str_of(v).to_string())).collect())
            }
            rules::K::N | rules::K::B => {
                ni += 1;
                Val::Num(nums[ni - 1])
            }
            // the db_* builtins, refused before a module is made (crate::db)
            rules::K::A | rules::K::I => {
                ai += 1;
                Val::Num(f64::NAN)
            }
        })
        .collect();
    or_fail(rules::run(c, &vals))
}
fn out_ptr(v: Val) -> u32 {
    match v {
        Val::Str(s) => new_str(&s.text),
        Val::Strs(l) => {
            let l = l.borrow();
            list_of(&l.iter().map(|x| x.text.as_str()).collect::<Vec<_>>(), 0)
        }
        _ => fail("internal: a string builtin gave no string".into()),
    }
}
fn out_num(v: Val) -> f64 {
    match v {
        Val::Num(x) => x,
        Val::Bool(x) => b(x),
        _ => fail("internal: a string builtin gave no number".into()),
    }
}

// The hot ones without the copy through values: the rules run on the memory.
#[no_mangle]
pub extern "C" fn vrt_s_find(s: u32, sub: u32, a: f64, e: f64) -> f64 {
    with_sv(s, |v| with_sv(sub, |w| rules::find(v, w, a, e, false)))
}
#[no_mangle]
pub extern "C" fn vrt_s_rfind(s: u32, sub: u32, a: f64, e: f64) -> f64 {
    with_sv(s, |v| with_sv(sub, |w| rules::find(v, w, a, e, true)))
}
#[no_mangle]
pub extern "C" fn vrt_s_index(s: u32, sub: u32, a: f64, e: f64) -> f64 {
    or_fail(with_sv(s, |v| with_sv(sub, |w| rules::index(v, w, a, e, false))))
}
#[no_mangle]
pub extern "C" fn vrt_s_rindex(s: u32, sub: u32, a: f64, e: f64) -> f64 {
    or_fail(with_sv(s, |v| with_sv(sub, |w| rules::index(v, w, a, e, true))))
}
#[no_mangle]
pub extern "C" fn vrt_s_count(s: u32, sub: u32, a: f64, e: f64) -> f64 {
    with_sv(s, |v| with_sv(sub, |w| rules::count(v, w, a, e)))
}
#[no_mangle]
pub extern "C" fn vrt_s_contains(s: u32, sub: u32) -> f64 {
    b(str_of(s).contains(str_of(sub)))
}
#[no_mangle]
pub extern "C" fn vrt_s_startswith(s: u32, p: u32) -> f64 {
    b(str_of(s).starts_with(str_of(p)))
}
#[no_mangle]
pub extern "C" fn vrt_s_endswith(s: u32, p: u32) -> f64 {
    b(str_of(s).ends_with(str_of(p)))
}
#[no_mangle]
pub extern "C" fn vrt_s_float(s: u32) -> f64 {
    or_fail(rules::parse_float(str_of(s)))
}
#[no_mangle]
pub extern "C" fn vrt_s_int(s: u32, base: f64) -> f64 {
    or_fail(rules::parse_int(str_of(s), base))
}
#[no_mangle]
pub extern "C" fn vrt_num_int(x: f64) -> f64 {
    or_fail(rules::int_of_num(x))
}
#[no_mangle]
pub extern "C" fn vrt_s_ord(s: u32) -> f64 {
    or_fail(with_sv(s, rules::ord))
}
#[no_mangle]
pub extern "C" fn vrt_s_chr(n: f64) -> u32 {
    made(rules::chr(n))
}
#[no_mangle]
pub extern "C" fn vrt_s_split_ws(s: u32) -> u32 {
    list_of(&or_fail(rules::split(str_of(s), None, -1.0, false)), 0)
}
#[no_mangle]
pub extern "C" fn vrt_s_rsplit_ws(s: u32) -> u32 {
    list_of(&or_fail(rules::split(str_of(s), None, -1.0, true)), 0)
}
#[no_mangle]
pub extern "C" fn vrt_s_split(s: u32, sep: u32, m: f64) -> u32 {
    list_of(&or_fail(rules::split(str_of(s), Some(str_of(sep)), m, false)), 0)
}
#[no_mangle]
pub extern "C" fn vrt_s_rsplit(s: u32, sep: u32, m: f64) -> u32 {
    list_of(&or_fail(rules::split(str_of(s), Some(str_of(sep)), m, true)), 0)
}
#[no_mangle]
pub extern "C" fn vrt_s_splitlines(s: u32) -> u32 {
    list_of(&rules::splitlines(str_of(s)), 0)
}
#[no_mangle]
pub extern "C" fn vrt_s_join(sep: u32, l: u32) -> u32 {
    made(rules::join(str_of(sep), &texts_of(l)))
}
#[no_mangle]
pub extern "C" fn vrt_ss_contains(l: u32, s: u32) -> f64 {
    let t = str_of(s);
    b(items(l).iter().any(|&v| str_of(v) == t))
}

macro_rules! via_registry {
    ($($name:ident ($($a:ident: u32),*; $($x:ident: f64),*) -> $ret:ident = $c:expr;)*) => { $(
        #[no_mangle]
        pub extern "C" fn $name($($a: u32,)* $($x: f64),*) -> via_registry!(@t $ret) {
            let v = registry($c, &[$($a),*], &[$($x),*]);
            via_registry!(@o $ret, v)
        }
    )* };
    (@t ptr) => { u32 };
    (@t num) => { f64 };
    (@o ptr, $v:expr) => { out_ptr($v) };
    (@o num, $v:expr) => { out_num($v) };
}

via_registry! {
    vrt_s_upper(s: u32;) -> ptr = "viv_s_upper";
    vrt_s_lower(s: u32;) -> ptr = "viv_s_lower";
    vrt_s_isspace(s: u32;) -> num = "viv_s_isspace";
    vrt_s_isalpha(s: u32;) -> num = "viv_s_isalpha";
    vrt_s_isalnum(s: u32;) -> num = "viv_s_isalnum";
    vrt_s_isdigit(s: u32;) -> num = "viv_s_isdigit";
    vrt_s_isdecimal(s: u32;) -> num = "viv_s_isdecimal";
    vrt_s_isnumeric(s: u32;) -> num = "viv_s_isnumeric";
    vrt_s_isupper(s: u32;) -> num = "viv_s_isupper";
    vrt_s_islower(s: u32;) -> num = "viv_s_islower";
    vrt_s_isascii(s: u32;) -> num = "viv_s_isascii";
    vrt_s_isprintable(s: u32;) -> num = "viv_s_isprintable";
    vrt_s_strip(s: u32;) -> ptr = "viv_s_strip";
    vrt_s_lstrip(s: u32;) -> ptr = "viv_s_lstrip";
    vrt_s_rstrip(s: u32;) -> ptr = "viv_s_rstrip";
    vrt_s_strip_chars(s: u32, c: u32;) -> ptr = "viv_s_strip_chars";
    vrt_s_lstrip_chars(s: u32, c: u32;) -> ptr = "viv_s_lstrip_chars";
    vrt_s_rstrip_chars(s: u32, c: u32;) -> ptr = "viv_s_rstrip_chars";
    vrt_ss_index(l: u32, s: u32;) -> num = "viv_ss_index";
    vrt_ss_count(l: u32, s: u32;) -> num = "viv_ss_count";
    vrt_s_repr(s: u32;) -> ptr = "viv_s_repr";
    vrt_ss_repr(l: u32;) -> ptr = "viv_ss_repr";
    vrt_ss_sorted(l: u32;) -> ptr = "viv_ss_sorted";
    vrt_s_sorted(s: u32;) -> ptr = "viv_s_sorted";
    vrt_s_unique(s: u32;) -> ptr = "viv_s_unique";
    vrt_ss_unique(l: u32;) -> ptr = "viv_ss_unique";
    vrt_s_read_text(p: u32;) -> ptr = "viv_s_read_text";
    vrt_s_read_lines(p: u32;) -> ptr = "viv_s_read_lines";
    vrt_s_write_text(p: u32, t: u32;) -> num = "viv_s_write_text";
}

#[no_mangle]
pub extern "C" fn vrt_s_replace(s: u32, old: u32, new: u32, count: f64) -> u32 {
    made(with_sv(s, |v| rules::replace(v, str_of(old), str_of(new), count)))
}
#[no_mangle]
pub extern "C" fn vrt_s_ljust(s: u32, w: f64, f: u32) -> u32 {
    made(with_sv(s, |v| with_sv(f, |fv| rules::pad(v, w, fv, 0))))
}
#[no_mangle]
pub extern "C" fn vrt_s_rjust(s: u32, w: f64, f: u32) -> u32 {
    made(with_sv(s, |v| with_sv(f, |fv| rules::pad(v, w, fv, 1))))
}
#[no_mangle]
pub extern "C" fn vrt_s_center(s: u32, w: f64, f: u32) -> u32 {
    made(with_sv(s, |v| with_sv(f, |fv| rules::pad(v, w, fv, 2))))
}
#[no_mangle]
pub extern "C" fn vrt_s_zfill(s: u32, w: f64) -> u32 {
    made(with_sv(s, |v| rules::zfill(v, w)))
}

/// Put a file where a program running in the browser reads it.
/// `name` and `data` are bytes in memory the
/// host wrote (`viv_alloc`).
#[no_mangle]
pub extern "C" fn vrt_file_put(np: u32, nl: u32, dp: u32, dl: u32) {
    let name = string(np, nl);
    let data = unsafe { std::slice::from_raw_parts(dp as usize as *const u8, dl as usize) }.to_vec();
    rules::FILES.with(|f| f.borrow_mut().insert(name, data));
}
