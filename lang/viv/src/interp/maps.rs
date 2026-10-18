// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Maps: `map()`, `m[k] = v`, `m[k]`, `has`, `get`, `remove`, `keys`,
//! `values`, `copy`, `clear`, `len`.
//!
//! A map holds numbers under number keys in the order the keys were first
//! inserted, laid out as CPython's compact dict: a table of slots, a power of
//! two long, each empty, removed, or the position of an entry; and the entries
//! (key, value, live) in insertion order, a removed one left in place until
//! the next rebuild. The hash decides only where a key is looked for, never
//! anything a program prints.
//!
//! This file is the one statement of the rules: the interpreter's arms and the
//! wasm runtime call `MapData` here, and `C_RUNTIME` below is its C twin, check
//! for check and word for word.

use super::strings::{self, Str};
use super::Val;
use crate::printfmt::{self, Fmt};
use std::cell::RefCell;
use std::rc::Rc;

/// The builtins that take a map first (or make one). `len` and `copy` take a
/// map too, beside arrays; they are the arrays' builtins, extended.
pub const NAMES: [&str; 7] = ["map", "has", "get", "remove", "keys", "values", "clear"];

/// A slot that has never held an entry: a search for a key stops here.
const EMPTY: i32 = -1;
/// A slot whose entry was removed: a search passes over it, an insert may reuse it.
const GONE: i32 = -2;
/// The fewest slots a table has.
const MIN_SLOTS: usize = 8;

/// The message for a NaN key.
pub const NAN_KEY: &str = "a map key cannot be NaN";
/// `[m1, m2]`
pub const NO_MAP_ARRAYS: &str = "an array cannot hold maps";
/// `m[i, j]`
pub const ONE_KEY: &str = "a map takes one key: m[k]";
/// `fmt(x, "d", "r")` of something that is not a map
pub const TWO_FORMATS: &str = "`fmt` takes two formats only for a map (its keys, then its values)";

/// `hash(k)`: the bits of k, with -0.0 taken as 0.0 (they are one key),
/// through the splitmix64 finalizer. `viv_map_hash` in `C_RUNTIME` is this.
pub fn hash(k: f64) -> u64 {
    let k = if k == 0.0 { 0.0 } else { k };
    let mut z = k.to_bits();
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A key as the errors write it: `fmt(k)` without a trailing `.0`, since a
/// map does not know whether its key was an int (`3`, `2.5`, `-0`, `1e+300`).
pub fn key_text(k: f64) -> String {
    let s = printfmt::repr(k);
    match s.strip_suffix(".0") {
        Some(t) => t.to_string(),
        None => s,
    }
}

/// `the key 3 is not in the map`
pub fn missing(k: f64) -> String {
    format!("the key {} is not in the map", key_text(k))
}

/// What a map's keys, or its values, are. The
/// lowering fixes both kinds for each map (`Ty::Map(key, value)`); `Any` is
/// inference's "no store seen yet", which the lowering reads as `Num`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mk {
    Any,
    Num,
    Str,
}

impl Mk {
    /// `Any` settled: a map nothing stores into holds numbers.
    pub fn settled(self) -> Mk {
        if self == Mk::Any { Mk::Num } else { self }
    }
    /// `n` or `s`, as the kinded runtime names end (`viv_map_set_sn`).
    pub fn letter(self) -> char {
        if self == Mk::Str { 's' } else { 'n' }
    }
}

/// A key or a value: a number or a string.
#[derive(Debug, Clone)]
pub(crate) enum MV {
    N(f64),
    S(Rc<Str>),
}

/// A key as a lookup reads it: no string is made to look one up.
#[derive(Debug, Clone, Copy)]
pub(crate) enum KR<'a> {
    N(f64),
    S(&'a str),
}

impl MV {
    pub fn key(&self) -> KR<'_> {
        match self {
            MV::N(x) => KR::N(*x),
            MV::S(s) => KR::S(&s.text),
        }
    }
}

impl KR<'_> {
    fn hash(self) -> u64 {
        match self {
            KR::N(k) => hash(k),
            KR::S(s) => strings::hash(s),
        }
    }
    fn is_nan(self) -> bool {
        matches!(self, KR::N(k) if k.is_nan())
    }
}

/// `the key 'a' is not in the map` (a string key in its repr, as Python's
/// KeyError writes it).
pub fn missing_key(k: KR<'_>) -> String {
    match k {
        KR::N(x) => missing(x),
        KR::S(s) => format!("the key {} is not in the map", strings::repr(s)),
    }
}

/// A column of keys or of values: numbers, or strings. An empty map's
/// columns take the kind of the first key and value stored.
#[derive(Debug, Clone)]
pub(crate) enum Col {
    N(Vec<f64>),
    S(Vec<Rc<Str>>),
}

impl Col {
    fn len(&self) -> usize {
        match self {
            Col::N(v) => v.len(),
            Col::S(v) => v.len(),
        }
    }
    fn at(&self, e: usize) -> MV {
        match self {
            Col::N(v) => MV::N(v[e]),
            Col::S(v) => MV::S(v[e].clone()),
        }
    }
    fn is_key(&self, e: usize, k: KR<'_>) -> bool {
        match (self, k) {
            (Col::N(v), KR::N(k)) => v[e] == k,
            (Col::S(v), KR::S(k)) => v[e].text == k,
            _ => false,
        }
    }
    fn key_at(&self, e: usize) -> KR<'_> {
        match self {
            Col::N(v) => KR::N(v[e]),
            Col::S(v) => KR::S(&v[e].text),
        }
    }
    /// Store `x` at `e` (or push it, `e == len`), taking `x`'s kind when the
    /// column is empty. Err when the kinds differ (the lowering refuses a
    /// map that mixes them, so a run never meets this).
    fn put(&mut self, e: usize, x: MV, what: &str) -> Result<(), String> {
        if self.len() == 0 {
            match (&*self, &x) {
                (Col::N(_), MV::S(_)) => *self = Col::S(Vec::new()),
                (Col::S(_), MV::N(_)) => *self = Col::N(Vec::new()),
                _ => {}
            }
        }
        match (self, x) {
            (Col::N(v), MV::N(x)) => {
                if e == v.len() {
                    v.push(x)
                } else {
                    v[e] = x
                }
            }
            (Col::S(v), MV::S(x)) => {
                if e == v.len() {
                    v.push(x)
                } else {
                    v[e] = x
                }
            }
            _ => return Err(format!("a map's {what} are all numbers or all strings")),
        }
        Ok(())
    }
    fn empty_like(&self, cap: usize) -> Col {
        match self {
            Col::N(_) => Col::N(Vec::with_capacity(cap)),
            Col::S(_) => Col::S(Vec::with_capacity(cap)),
        }
    }
    fn text(&self, e: usize, f: Fmt) -> String {
        match self {
            Col::N(v) => printfmt::number(v[e], f),
            Col::S(v) => strings::repr(&v[e].text),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MapData {
    /// the slots: EMPTY, GONE, or an entry's position
    slots: Vec<i32>,
    keys: Col,
    vals: Col,
    live: Vec<bool>,
    /// live entries
    used: usize,
}

impl MapData {
    pub fn new() -> MapData {
        MapData { slots: vec![EMPTY; MIN_SLOTS], keys: Col::N(Vec::new()), vals: Col::N(Vec::new()), live: Vec::new(), used: 0 }
    }

    pub fn len(&self) -> usize {
        self.used
    }

    /// The slot holding `k`'s entry, or None. A NaN is never found.
    fn slot_of(&self, k: KR<'_>) -> Option<usize> {
        if k.is_nan() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut i = (k.hash() as usize) & mask;
        loop {
            let s = self.slots[i];
            if s == EMPTY {
                return None;
            }
            if s >= 0 && self.keys.is_key(s as usize, k) {
                return Some(i);
            }
            i = (i + 1) & mask;
        }
    }

    pub fn get(&self, k: KR<'_>) -> Option<MV> {
        self.slot_of(k).map(|i| self.vals.at(self.slots[i] as usize))
    }

    pub fn has(&self, k: KR<'_>) -> bool {
        self.slot_of(k).is_some()
    }

    /// `m[k] = v`: the value of a key already there changes in its place; a
    /// new key goes at the end. The first key inserted is the one kept, so
    /// `m[0.0]` then `m[-0.0]` keeps 0.0.
    pub fn set(&mut self, k: MV, v: MV) -> Result<(), String> {
        if k.key().is_nan() {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, NAN_KEY));
        }
        if let Some(i) = self.slot_of(k.key()) {
            let e = self.slots[i] as usize;
            return self.vals.put(e, v, "values");
        }
        // room for one more entry: at most two thirds of the slots in use
        if (self.keys.len() + 1) * 3 > self.slots.len() * 2 {
            self.rebuild();
        }
        let e = self.keys.len();
        if self.vals.len() == 0 && e == 0 {
            // the first entry sets both kinds; check the value's before the key goes in
            self.vals.put(0, v.clone(), "values")?;
            if let Err(msg) = self.keys.put(0, k.clone(), "keys") {
                self.vals = self.vals.empty_like(0);
                return Err(msg);
            }
        } else {
            self.keys.put(e, k.clone(), "keys")?;
            if let Err(msg) = self.vals.put(e, v, "values") {
                self.keys = self.truncated_keys(e);
                return Err(msg);
            }
        }
        let mask = self.slots.len() - 1;
        let mut i = (k.key().hash() as usize) & mask;
        while self.slots[i] >= 0 {
            i = (i + 1) & mask;
        }
        self.slots[i] = e as i32;
        self.live.push(true);
        self.used += 1;
        Ok(())
    }

    fn truncated_keys(&self, n: usize) -> Col {
        match &self.keys {
            Col::N(v) => Col::N(v[..n].to_vec()),
            Col::S(v) => Col::S(v[..n].to_vec()),
        }
    }

    /// Take `k` out and give its value, or None when it is not there.
    pub fn remove(&mut self, k: KR<'_>) -> Option<MV> {
        let i = self.slot_of(k)?;
        let e = self.slots[i] as usize;
        self.slots[i] = GONE;
        self.live[e] = false;
        self.used -= 1;
        Some(self.vals.at(e))
    }

    pub fn clear(&mut self) {
        *self = MapData::new();
    }

    /// The live entries copied down in order, into the smallest power of two
    /// of slots that is at least 3 * used (and MIN_SLOTS).
    fn rebuild(&mut self) {
        let mut cap = MIN_SLOTS;
        while cap < 3 * self.used {
            cap *= 2;
        }
        let (keys, vals) = (self.live_col(&self.keys), self.live_col(&self.vals));
        let mut slots = vec![EMPTY; cap];
        let mask = cap - 1;
        for e in 0..keys.len() {
            let mut i = (keys.key_at(e).hash() as usize) & mask;
            while slots[i] >= 0 {
                i = (i + 1) & mask;
            }
            slots[i] = e as i32;
        }
        self.live = vec![true; keys.len()];
        self.keys = keys;
        self.vals = vals;
        self.slots = slots;
    }

    /// A column's live entries, in order.
    fn live_col(&self, c: &Col) -> Col {
        match c {
            Col::N(v) => Col::N((0..v.len()).filter(|&e| self.live[e]).map(|e| v[e]).collect()),
            Col::S(v) => Col::S((0..v.len()).filter(|&e| self.live[e]).map(|e| v[e].clone()).collect()),
        }
    }

    /// The keys, in insertion order: numbers or strings.
    pub fn keys(&self) -> Col {
        self.live_col(&self.keys)
    }

    /// The values, in insertion order.
    pub fn values(&self) -> Col {
        self.live_col(&self.vals)
    }

    /// A new map with the same pairs in the same order (and no removed entries).
    pub fn copied(&self) -> MapData {
        let mut m = self.clone();
        m.rebuild();
        m
    }

    /// `{k: v, ...}`, number keys in `kf` and number values in `vf`; a
    /// string key or value is written as its repr, as Python writes a dict.
    pub fn text(&self, kf: Fmt, vf: Fmt) -> String {
        let mut s = String::from("{");
        let mut first = true;
        for e in (0..self.keys.len()).filter(|&e| self.live[e]) {
            if !first {
                s.push_str(", ");
            }
            first = false;
            s.push_str(&self.keys.text(e, kf));
            s.push_str(": ");
            s.push_str(&self.vals.text(e, vf));
        }
        s.push('}');
        s
    }
}

pub(super) type Map = Rc<RefCell<MapData>>;

pub(super) fn new_map(m: MapData) -> Val {
    Val::Map(Rc::new(RefCell::new(m)))
}

/// A key or a value: a number or a string (anything else is the lowering's
/// type error, in its words).
fn mv(v: &Val, callee: &str, what: &str) -> Result<MV, String> {
    match v {
        Val::Num(k) => Ok(MV::N(*k)),
        Val::Str(s) => Ok(MV::S(s.clone())),
        _ => Err(format!("`{callee}`: {what} is a number or a string")),
    }
}

fn key_ref<'a>(v: &'a Val, callee: &str) -> Result<KR<'a>, String> {
    match v {
        Val::Num(k) => Ok(KR::N(*k)),
        Val::Str(s) => Ok(KR::S(&s.text)),
        _ => Err(format!("`{callee}`: a map key is a number or a string")),
    }
}

fn val_of(x: MV) -> Val {
    match x {
        MV::N(n) => Val::Num(n),
        MV::S(s) => Val::Str(s),
    }
}

fn col_val(c: Col) -> Val {
    match c {
        Col::N(v) => super::new_arr(vec![v.len()], v),
        Col::S(v) => strings::strs(v),
    }
}

fn the_map<'a>(v: &'a Val, callee: &str) -> Result<&'a Map, String> {
    match v {
        Val::Map(m) => Ok(m),
        _ => Err(format!("`{callee}` takes a map first")),
    }
}

/// `m[k]`
/// An Int key or value is stored as the number it is, exactly (the lowering
/// converts it the same way).
fn int_as_num(v: &Val) -> Result<std::borrow::Cow<'_, Val>, String> {
    match v {
        Val::Int(i) => Ok(std::borrow::Cow::Owned(Val::Num(super::ints::to_num(*i)?))),
        v => Ok(std::borrow::Cow::Borrowed(v)),
    }
}

pub(super) fn index(m: &Map, k: &Val) -> Result<Val, String> {
    let k = int_as_num(k)?;
    let k = key_ref(&k, "a map read")?;
    m.borrow().get(k).map(val_of).ok_or_else(|| missing_key(k))
}

/// `m[k] = v`
pub(super) fn store(m: &Map, k: &Val, v: &Val) -> Result<(), String> {
    let (k, v) = (int_as_num(k)?, int_as_num(v)?);
    let k = mv(&k, "a map store", "a map key")?;
    let v = mv(&v, "a map store", "a map value")?;
    m.borrow_mut().set(k, v)
}

/// The interpreter's arms: the map builtins, and `len` and `copy` of a map.
/// `None`: not one of these (the arrays' `len` and `copy` are elsewhere).
pub(super) fn builtin(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    let conv: Vec<Val>;
    let a = if NAMES.contains(&callee) && a.iter().skip(1).any(|v| matches!(v, Val::Int(_))) {
        conv = a.iter().enumerate().map(|(j, v)| if j == 0 { Ok(v.clone()) } else { int_as_num(v).map(|c| c.into_owned()) }).collect::<Result<_, _>>()?;
        &conv[..]
    } else {
        a
    };
    let arity = |n: &[usize], usage: &str| -> Result<(), String> {
        if n.contains(&a.len()) {
            Ok(())
        } else {
            Err(format!("`{callee}` is written {usage}"))
        }
    };
    Ok(Some(match callee {
        "map" => {
            arity(&[0], "map()")?;
            new_map(MapData::new())
        }
        "len" | "copy" => {
            let [Val::Map(m)] = a else { return Ok(None) };
            if callee == "len" {
                Val::Num(m.borrow().len() as f64)
            } else {
                new_map(m.borrow().copied())
            }
        }
        "has" => {
            arity(&[2], "has(m, k)")?;
            let m = the_map(&a[0], callee)?;
            Val::Bool(m.borrow().has(key_ref(&a[1], callee)?))
        }
        "get" => {
            arity(&[3], "get(m, k, default)")?;
            let m = the_map(&a[0], callee)?;
            let k = key_ref(&a[1], callee)?;
            let d = mv(&a[2], callee, "a map's default")?;
            let got = m.borrow().get(k);
            val_of(got.unwrap_or(d))
        }
        "remove" => {
            arity(&[2, 3], "remove(m, k) or remove(m, k, default)")?;
            let m = the_map(&a[0], callee)?;
            let k = key_ref(&a[1], callee)?;
            let d = a.get(2).map(|d| mv(d, callee, "a map's default")).transpose()?;
            let got = m.borrow_mut().remove(k);
            match (got, d) {
                (Some(v), _) | (None, Some(v)) => val_of(v),
                (None, None) => return Err(missing_key(k)),
            }
        }
        "keys" | "values" => {
            arity(&[1], if callee == "keys" { "keys(m)" } else { "values(m)" })?;
            let m = the_map(&a[0], callee)?;
            let data = if callee == "keys" { m.borrow().keys() } else { m.borrow().values() };
            col_val(data)
        }
        "clear" => {
            arity(&[1], "clear(m)")?;
            the_map(&a[0], callee)?.borrow_mut().clear();
            Val::Num(0.0)
        }
        _ => return Ok(None),
    }))
}

/// The C runtime's twin (fragment "maps"; it stops through the slices
/// fragment's `viv_sl_fail`, so an error carries its location in a debug build
/// as a slice's does). A map is a `viv_map*`: a header that never moves, so
/// every copy of the pointer sees every change. A map is filed on the arena
/// with tag 1 in its pointer's low bits and freed by the arena's rules, like
/// an array: when the def that made it returns without returning it (or a
/// record holding it), or at the end of the loop pass that made it unless a
/// variable the loop assigns holds it.
pub const C_RUNTIME: &str = r#"/* Maps (lang/viv/src/interp/maps.rs has the rules; the layout is
   CPython's compact dict): slots, a power of two long, each -1 (empty), -2 (removed) or an
   entry's position; entries in insertion order, removed ones marked dead. */
typedef struct viv_map { int* slots; double* k; double* v; unsigned char* live; long cap, n, used, room; } viv_map;
VIV_SL_UNUSED static void* viv_map_mem(void* p, size_t bytes) {
    void* q = realloc(p, bytes ? bytes : 1);
    if (!q) { viv_fflush(stdout); viv_errf("error: out of memory growing a map\n"); viv_stop_exit(VIV_E_MEMORY); }
    return q;
}
VIV_SL_UNUSED static unsigned long long viv_map_hash(double k) {
    unsigned long long z;
    if (k == 0.0) k = 0.0;
    memcpy(&z, &k, sizeof z);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
/* entries room: the most a table of cap slots takes before a rebuild */
VIV_SL_UNUSED static void viv_map_alloc(viv_map* m, long cap) {
    m->cap = cap;
    m->slots = (int*)viv_map_mem(0, (size_t)cap * sizeof(int));
    for (long i = 0; i < cap; i++) m->slots[i] = -1;
    m->room = cap * 2 / 3 + 1;
    m->k = (double*)viv_map_mem(0, (size_t)m->room * sizeof(double));
    m->v = (double*)viv_map_mem(0, (size_t)m->room * sizeof(double));
    m->live = (unsigned char*)viv_map_mem(0, (size_t)m->room);
    m->n = 0; m->used = 0;
}
/* a map's blocks, freed when the arena drops it (tag 1) */
VIV_SL_UNUSED static void viv_map_drop(void* p) {
    viv_map* m = (viv_map*)p;
    free(m->slots); free(m->k); free(m->v); free(m->live);
    free(m);
}
VIV_SL_UNUSED static viv_map* viv_map_new(void) {
    viv_map* m = (viv_map*)viv_map_mem(0, sizeof(viv_map));
    viv_map_alloc(m, 8);
    viv_arena_drop[1] = viv_map_drop;
    viv_arena_push((char*)m + 1);
    return m;
}
/* what the arena files a map under: its pointer and its tag */
#define VIV_MAP_KEY(m) ((const void*)((size_t)(m) | 1))
VIV_SL_UNUSED static long viv_map_slot(viv_map* m, double k) {
    if (k != k) return -1;
    long mask = m->cap - 1, i = (long)(viv_map_hash(k) & (unsigned long long)mask);
    for (;;) {
        int s = m->slots[i];
        if (s == -1) return -1;
        if (s >= 0 && m->k[s] == k) return i;
        i = (i + 1) & mask;
    }
}
VIV_SL_UNUSED static void viv_map_rebuild(viv_map* m) {
    long cap = 8;
    while (cap < 3 * m->used) cap *= 2;
    viv_map old = *m;
    viv_map_alloc(m, cap);
    long mask = cap - 1;
    for (long e = 0; e < old.n; e++) {
        if (!old.live[e]) continue;
        long i = (long)(viv_map_hash(old.k[e]) & (unsigned long long)mask);
        while (m->slots[i] >= 0) i = (i + 1) & mask;
        m->slots[i] = (int)m->n;
        m->k[m->n] = old.k[e]; m->v[m->n] = old.v[e]; m->live[m->n] = 1;
        m->n++;
    }
    m->used = m->n;
    free(old.slots); free(old.k); free(old.v); free(old.live);
}
/* the key as the errors write it: repr without a trailing .0 */
VIV_SL_UNUSED static void viv_map_missing(double k, int site) {
    char b[64], msg[128];
    viv_repr(b, k);
    size_t n = strlen(b);
    if (n > 2 && strcmp(b + n - 2, ".0") == 0) b[n - 2] = 0;
    snprintf(msg, sizeof msg, "the key %s is not in the map", b);
    viv_sl_fail(VIV_E_ERROR, msg, 0, site);
}
VIV_SL_UNUSED static double viv_map_set(viv_map* m, double k, double v, int site) {
    if (k != k) viv_sl_fail(VIV_E_VALUE, "a map key cannot be NaN", 0, site);
    long i = viv_map_slot(m, k);
    if (i >= 0) { m->v[m->slots[i]] = v; return 0.0; }
    if ((m->n + 1) * 3 > m->cap * 2) viv_map_rebuild(m);
    long mask = m->cap - 1;
    i = (long)(viv_map_hash(k) & (unsigned long long)mask);
    while (m->slots[i] >= 0) i = (i + 1) & mask;
    m->slots[i] = (int)m->n;
    m->k[m->n] = k; m->v[m->n] = v; m->live[m->n] = 1;
    m->n++; m->used++;
    return 0.0;
}
VIV_SL_UNUSED static double viv_map_get(viv_map* m, double k, int site) {
    long i = viv_map_slot(m, k);
    if (i < 0) viv_map_missing(k, site);
    return m->v[m->slots[i]];
}
VIV_SL_UNUSED static double viv_map_has(viv_map* m, double k) { return viv_map_slot(m, k) >= 0 ? 1.0 : 0.0; }
VIV_SL_UNUSED static double viv_map_getd(viv_map* m, double k, double d) {
    long i = viv_map_slot(m, k);
    return i < 0 ? d : m->v[m->slots[i]];
}
/* take k out; its value, or d when it is not there and has_d, else stop */
VIV_SL_UNUSED static double viv_map_take(viv_map* m, double k, int has_d, double d, int site) {
    long i = viv_map_slot(m, k);
    if (i < 0) {
        if (!has_d) viv_map_missing(k, site);
        return d;
    }
    int e = m->slots[i];
    m->slots[i] = -2;
    m->live[e] = 0;
    m->used--;
    return m->v[e];
}
VIV_SL_UNUSED static double viv_map_remove(viv_map* m, double k, int site) { return viv_map_take(m, k, 0, 0.0, site); }
VIV_SL_UNUSED static double viv_map_removed(viv_map* m, double k, double d) { return viv_map_take(m, k, 1, d, 0); }
VIV_SL_UNUSED static double viv_map_len(viv_map* m) { return (double)m->used; }
VIV_SL_UNUSED static viv_arr viv_map_list(viv_map* m, int values) {
    int d[1]; d[0] = (int)m->used;
    viv_arr r = viv_arr_new(1, d);
    long j = 0;
    for (long e = 0; e < m->n; e++) if (m->live[e]) r.data[j++] = values ? m->v[e] : m->k[e];
    return r;
}
VIV_SL_UNUSED static viv_arr viv_map_keys(viv_map* m) { return viv_map_list(m, 0); }
VIV_SL_UNUSED static viv_arr viv_map_values(viv_map* m) { return viv_map_list(m, 1); }
VIV_SL_UNUSED static viv_map* viv_map_copy(viv_map* m) {
    viv_map* r = viv_map_new();
    for (long e = 0; e < m->n; e++) if (m->live[e]) viv_map_set(r, m->k[e], m->v[e], 0);
    return r;
}
VIV_SL_UNUSED static double viv_map_clear(viv_map* m) {
    free(m->slots); free(m->k); free(m->v); free(m->live);
    viv_map_alloc(m, 8);
    return 0.0;
}
/* {k: v, ...}, each number as viv_pnum writes it */
VIV_SL_UNUSED static void viv_pmap(viv_map* m, int kk, int kp, int kw, int vk, int vp, int vw) {
    int first = 1;
    viv_putchar('{');
    for (long e = 0; e < m->n; e++) {
        if (!m->live[e]) continue;
        if (!first) viv_fputs(", ", stdout);
        first = 0;
        viv_pnum(m->k[e], kk, kp, kw);
        viv_fputs(": ", stdout);
        viv_pnum(m->v[e], vk, vp, vw);
    }
    viv_putchar('}');
}
/* viv_print (printfmt.rs) with one more tag, 5: a map and two formats. A
   print with a map in it calls this one, so a program without maps compiles
   to the C it always did. */
VIV_SL_UNUSED static double viv_print_m(int n, ...) {
    va_list ap;
    va_start(ap, n);
    for (int k = 0; k < n; k++) {
        int tag = va_arg(ap, int);
        if (tag == 0) viv_fputs(va_arg(ap, const char*), stdout);
        else if (tag == 1) { double x = va_arg(ap, double); int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int); viv_pnum(x, kd, p, w); }
        else if (tag == 2) {
            viv_arr a = va_arg(ap, viv_arr);
            int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int);
            int one = a.size;
            viv_parr(a.data, a.rank <= 1 ? &one : a.dims, a.rank, kd, p, w);
        }
        else if (tag == 3) viv_fputs(va_arg(ap, double) != 0.0 ? "true" : "false", stdout);
        else if (tag == 5) {
            viv_map* m = va_arg(ap, viv_map*);
            int kk = va_arg(ap, int), kp = va_arg(ap, int), kw = va_arg(ap, int);
            int vk = va_arg(ap, int), vp = va_arg(ap, int), vw = va_arg(ap, int);
            viv_pmap(m, kk, kp, kw, vk, vp, vw);
        }
        else {
            _Complex double z = va_arg(ap, _Complex double);
            double re = ((double*)&z)[0] + 0.0, im = ((double*)&z)[1] + 0.0;
            viv_pnum(re, 'b', 6, 0);
            viv_fputs(im < 0.0 ? "-" : "+", stdout);
            viv_pnum(im < 0.0 ? -im : im, 'b', 6, 0);
            viv_putchar('i');
        }
    }
    va_end(ap);
    viv_putchar('\n');
    return 0.0;
}"#;

// ---- maps with string keys or values ---------------------------------------------------

/// A slot of a kinded map runtime call: what the C function and the wasm
/// import take or give.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// a number (f64)
    N,
    /// a boolean, as a number
    B,
    /// a string (a `viv_str`; an i32 in wasm)
    S,
    /// a list of strings
    L,
    /// a real array
    A,
    /// the map itself (a `viv_smap*`; an i32 in wasm)
    M,
}

/// One runtime call of a map whose keys or values are strings:
/// `viv_map_<op>_<k><v>` (`viv_map_set_sn`: string keys, number values). The
/// name carries the kinds, so the wasm backend types the call by it.
#[derive(Clone, Debug)]
pub struct MapFn {
    pub name: String,
    pub op: &'static str,
    pub key: Mk,
    pub val: Mk,
    pub args: Vec<Slot>,
    pub ret: Slot,
    /// the C function stops with a located error: its viv_sites row goes last
    pub stops: bool,
}

/// The three kinds of map the kinded calls serve (numbers to numbers is the
/// `viv_map_*` runtime above).
pub const KINDED: [(Mk, Mk); 3] = [(Mk::Str, Mk::Num), (Mk::Num, Mk::Str), (Mk::Str, Mk::Str)];

/// The runtime name of `op` on a map of these kinds: `viv_map_set` for
/// numbers to numbers, `viv_map_set_sn` for strings to numbers.
pub fn callee(op: &str, key: Mk, val: Mk) -> String {
    let (k, v) = (key.settled(), val.settled());
    if k == Mk::Num && v == Mk::Num {
        format!("viv_map_{op}")
    } else {
        format!("viv_map_{op}_{}{}", k.letter(), v.letter())
    }
}

/// Every kinded map call.
pub fn kinded_fns() -> &'static [MapFn] {
    static FNS: std::sync::OnceLock<Vec<MapFn>> = std::sync::OnceLock::new();
    FNS.get_or_init(|| {
        let slot = |k: Mk| if k == Mk::Str { Slot::S } else { Slot::N };
        let list = |k: Mk| if k == Mk::Str { Slot::L } else { Slot::A };
        let mut out = Vec::new();
        for (k, v) in KINDED {
            let (ks, vs) = (slot(k), slot(v));
            let ops: [(&'static str, Vec<Slot>, Slot, bool); 12] = [
                ("new", vec![], Slot::M, false),
                ("set", vec![Slot::M, ks, vs], Slot::N, true),
                ("get", vec![Slot::M, ks], vs, true),
                ("has", vec![Slot::M, ks], Slot::B, false),
                ("getd", vec![Slot::M, ks, vs], vs, false),
                ("remove", vec![Slot::M, ks], vs, true),
                ("removed", vec![Slot::M, ks, vs], vs, false),
                ("len", vec![Slot::M], Slot::N, false),
                ("keys", vec![Slot::M], list(k), false),
                ("values", vec![Slot::M], list(v), false),
                ("copy", vec![Slot::M], Slot::M, false),
                ("clear", vec![Slot::M], Slot::N, false),
            ];
            for (op, args, ret, stops) in ops {
                out.push(MapFn { name: callee(op, k, v), op, key: k, val: v, args, ret, stops });
            }
        }
        out
    })
}

/// The kinded map call named `callee`, if it is one.
pub fn kinded(callee: &str) -> Option<&'static MapFn> {
    if !callee.starts_with("viv_map_") {
        return None;
    }
    kinded_fns().iter().find(|f| f.name == callee)
}

/// The C runtime of maps with string keys or values (fragment "smaps"): the
/// rules of `MapData` over a `viv_smap`, whose keys and values are unions
/// of a number and a string. The map owns a copy of every string it holds,
/// made off the arena (`viv_s_static`) when the string goes in and freed
/// when the entry is removed, replaced or cleared, or when the map is; a
/// read hands out a new string on the arena. So
/// no string a map holds is on the arena; the map itself is, with tag 2.
pub fn c_smaps() -> &'static str {
    static CODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CODE.get_or_init(|| {
        let mut s = String::from(C_SMAPS);
        let cty = |x: Slot| match x {
            Slot::N | Slot::B => "double",
            Slot::S => "viv_str",
            Slot::L => "viv_strs",
            Slot::A => "viv_arr",
            Slot::M => "viv_smap*",
        };
        let wrap = |k: Mk, arg: &str| if k == Mk::Str { format!("viv_mvs({arg})") } else { format!("viv_mvn({arg})") };
        let pick = |k: Mk| if k == Mk::Str { ".s" } else { ".n" };
        for f in kinded_fns() {
            let mut params: Vec<String> = f.args.iter().enumerate().map(|(j, a)| format!("{} a{j}", cty(*a))).collect();
            if f.stops {
                params.push("int site".into());
            }
            let params = if params.is_empty() { "void".to_string() } else { params.join(", ") };
            let (k, v) = (f.key, f.val);
            let body = match f.op {
                "new" => format!("viv_smap_new({}, {})", (k == Mk::Str) as u8, (v == Mk::Str) as u8),
                "set" => format!("viv_smap_set(a0, {}, {}, site)", wrap(k, "a1"), wrap(v, "a2")),
                "get" => format!("viv_smap_get(a0, {}, site){}", wrap(k, "a1"), pick(v)),
                "has" => format!("viv_smap_has(a0, {})", wrap(k, "a1")),
                "getd" => format!("viv_smap_getd(a0, {}, {}){}", wrap(k, "a1"), wrap(v, "a2"), pick(v)),
                "remove" => format!("viv_smap_take(a0, {}, 0, viv_mvn(0.0), site){}", wrap(k, "a1"), pick(v)),
                "removed" => format!("viv_smap_take(a0, {}, 1, {}, 0){}", wrap(k, "a1"), wrap(v, "a2"), pick(v)),
                "len" => "(double)a0->used".to_string(),
                "keys" | "values" => {
                    let which = if f.op == "keys" { 0 } else { 1 };
                    let kind = if f.op == "keys" { k } else { v };
                    if kind == Mk::Str { format!("viv_smap_strs(a0, {which})") } else { format!("viv_smap_nums(a0, {which})") }
                }
                "copy" => "viv_smap_copy(a0)".to_string(),
                _ => "viv_smap_clear(a0)".to_string(),
            };
            s.push_str(&format!("VIV_S_UNUSED static {} {}({params}) {{ return {body}; }}\n", cty(f.ret), f.name));
        }
        s
    })
    .as_str()
}

const C_SMAPS: &str = r#"/* Maps with string keys or values (lang/viv/src/interp/maps.rs has the
   rules): the layout of viv_map, with
   each key and value a union of a number and a string; `ks` and `vs` say
   which. The map owns a copy of each string it holds (off the arena, freed
   when its entry goes); a read gives a new string on the arena. */
typedef union viv_mv { double n; viv_str s; } viv_mv;
typedef struct viv_smap { int* slots; viv_mv* k; viv_mv* v; unsigned char* live; long cap, n, used, room; int ks, vs; } viv_smap;
VIV_S_UNUSED static viv_mv viv_mvn(double x) { viv_mv r; r.n = x; return r; }
VIV_S_UNUSED static viv_mv viv_mvs(viv_str x) { viv_mv r; r.s = VIV_S(x); return r; }
/* the map's own copy, and a copy handed out on the arena */
VIV_S_UNUSED static viv_str viv_smap_own(viv_str x) { return viv_s_static(x->s, x->len, x->nch); }
VIV_S_UNUSED static viv_str viv_smap_out(viv_str x) { return viv_s_make(x->s, x->len, 0); }
VIV_S_UNUSED static unsigned long long viv_smap_hash(viv_smap* m, viv_mv k) { return m->ks ? viv_str_hash(k.s) : viv_map_hash(k.n); }
VIV_S_UNUSED static int viv_smap_is(viv_smap* m, viv_mv a, viv_mv b) { return m->ks ? viv_str_eq(a.s, b.s) : a.n == b.n; }
VIV_S_UNUSED static void viv_smap_alloc(viv_smap* m, long cap) {
    m->cap = cap;
    m->slots = (int*)viv_map_mem(0, (size_t)cap * sizeof(int));
    for (long i = 0; i < cap; i++) m->slots[i] = -1;
    m->room = cap * 2 / 3 + 1;
    m->k = (viv_mv*)viv_map_mem(0, (size_t)m->room * sizeof(viv_mv));
    m->v = (viv_mv*)viv_map_mem(0, (size_t)m->room * sizeof(viv_mv));
    m->live = (unsigned char*)viv_map_mem(0, (size_t)m->room);
    m->n = 0; m->used = 0;
}
/* a map's blocks and the strings it owns, freed when the arena drops it (tag 2) */
VIV_S_UNUSED static void viv_smap_drop(void* p) {
    viv_smap* m = (viv_smap*)p;
    for (long e = 0; e < m->n; e++) {
        if (!m->live[e]) continue;
        if (m->ks) free(m->k[e].s);
        if (m->vs) free(m->v[e].s);
    }
    free(m->slots); free(m->k); free(m->v); free(m->live);
    free(m);
}
VIV_S_UNUSED static viv_smap* viv_smap_new(int ks, int vs) {
    viv_smap* m = (viv_smap*)viv_map_mem(0, sizeof(viv_smap));
    m->ks = ks; m->vs = vs;
    viv_smap_alloc(m, 8);
    viv_arena_drop[2] = viv_smap_drop;
    viv_arena_push((char*)m + 2);
    return m;
}
#define VIV_SMAP_KEY(m) ((const void*)((size_t)(m) | 2))
VIV_S_UNUSED static long viv_smap_slot(viv_smap* m, viv_mv k) {
    if (!m->ks && k.n != k.n) return -1;
    long mask = m->cap - 1, i = (long)(viv_smap_hash(m, k) & (unsigned long long)mask);
    for (;;) {
        int s = m->slots[i];
        if (s == -1) return -1;
        if (s >= 0 && viv_smap_is(m, m->k[s], k)) return i;
        i = (i + 1) & mask;
    }
}
VIV_S_UNUSED static void viv_smap_rebuild(viv_smap* m) {
    long cap = 8;
    while (cap < 3 * m->used) cap *= 2;
    viv_smap old = *m;
    viv_smap_alloc(m, cap);
    long mask = cap - 1;
    for (long e = 0; e < old.n; e++) {
        if (!old.live[e]) continue;
        long i = (long)(viv_smap_hash(m, old.k[e]) & (unsigned long long)mask);
        while (m->slots[i] >= 0) i = (i + 1) & mask;
        m->slots[i] = (int)m->n;
        m->k[m->n] = old.k[e]; m->v[m->n] = old.v[e]; m->live[m->n] = 1;
        m->n++;
    }
    m->used = m->n;
    free(old.slots); free(old.k); free(old.v); free(old.live);
}
/* the key as the errors write it: a number as viv_map_missing, a string in its repr */
VIV_S_UNUSED static void viv_smap_missing(viv_smap* m, viv_mv k, int site) {
    if (!m->ks) viv_map_missing(k.n, site);
    viv_sb b = {0, 0, 0, site};
    viv_sb_cstr(&b, "the key ");
    viv_s_repr_into(&b, k.s);
    viv_sb_cstr(&b, " is not in the map");
    viv_sb_put(&b, "", 1);
    viv_s_fail(VIV_E_ERROR, b.p, site);
}
VIV_S_UNUSED static double viv_smap_set(viv_smap* m, viv_mv k, viv_mv v, int site) {
    if (!m->ks && k.n != k.n) viv_sl_fail(VIV_E_VALUE, "a map key cannot be NaN", 0, site);
    long i = viv_smap_slot(m, k);
    if (i >= 0) {
        int e = m->slots[i];
        if (m->vs) { free(m->v[e].s); v.s = viv_smap_own(v.s); }
        m->v[e] = v;
        return 0.0;
    }
    if ((m->n + 1) * 3 > m->cap * 2) viv_smap_rebuild(m);
    long mask = m->cap - 1;
    i = (long)(viv_smap_hash(m, k) & (unsigned long long)mask);
    while (m->slots[i] >= 0) i = (i + 1) & mask;
    m->slots[i] = (int)m->n;
    if (m->ks) k.s = viv_smap_own(k.s);
    if (m->vs) v.s = viv_smap_own(v.s);
    m->k[m->n] = k; m->v[m->n] = v; m->live[m->n] = 1;
    m->n++; m->used++;
    return 0.0;
}
/* a value handed out: a number as it is, a string copied onto the arena */
VIV_S_UNUSED static viv_mv viv_smap_give(viv_smap* m, viv_mv v) {
    if (m->vs) v.s = viv_smap_out(v.s);
    return v;
}
VIV_S_UNUSED static viv_mv viv_smap_get(viv_smap* m, viv_mv k, int site) {
    long i = viv_smap_slot(m, k);
    if (i < 0) viv_smap_missing(m, k, site);
    return viv_smap_give(m, m->v[m->slots[i]]);
}
VIV_S_UNUSED static double viv_smap_has(viv_smap* m, viv_mv k) { return viv_smap_slot(m, k) >= 0 ? 1.0 : 0.0; }
VIV_S_UNUSED static viv_mv viv_smap_getd(viv_smap* m, viv_mv k, viv_mv d) {
    long i = viv_smap_slot(m, k);
    return i < 0 ? d : viv_smap_give(m, m->v[m->slots[i]]);
}
/* take k out; its value, or d when it is not there and has_d, else stop */
VIV_S_UNUSED static viv_mv viv_smap_take(viv_smap* m, viv_mv k, int has_d, viv_mv d, int site) {
    long i = viv_smap_slot(m, k);
    if (i < 0) {
        if (!has_d) viv_smap_missing(m, k, site);
        return d;
    }
    int e = m->slots[i];
    m->slots[i] = -2;
    m->live[e] = 0;
    m->used--;
    viv_mv r = viv_smap_give(m, m->v[e]);
    if (m->ks) { free(m->k[e].s); m->k[e].s = 0; }
    if (m->vs) { free(m->v[e].s); m->v[e].s = 0; }
    return r;
}
/* keys (which 0) or values (1) as a list of strings, or as an array */
VIV_S_UNUSED static viv_strs viv_smap_strs(viv_smap* m, int which) {
    viv_str* t = (viv_str*)malloc(sizeof(viv_str) * (size_t)(m->used ? m->used : 1));
    if (!t) viv_s_nomem();
    long j = 0;
    for (long e = 0; e < m->n; e++) if (m->live[e]) t[j++] = which ? m->v[e].s : m->k[e].s;
    viv_strs l = viv_ss_copyof(t, j, 0);
    free(t);
    return l;
}
VIV_S_UNUSED static viv_arr viv_smap_nums(viv_smap* m, int which) {
    int d[1]; d[0] = (int)m->used;
    viv_arr r = viv_arr_new(1, d);
    long j = 0;
    for (long e = 0; e < m->n; e++) if (m->live[e]) r.data[j++] = which ? m->v[e].n : m->k[e].n;
    return r;
}
VIV_S_UNUSED static viv_smap* viv_smap_copy(viv_smap* m) {
    viv_smap* r = viv_smap_new(m->ks, m->vs);
    for (long e = 0; e < m->n; e++) if (m->live[e]) viv_smap_set(r, m->k[e], m->v[e], 0);
    return r;
}
VIV_S_UNUSED static double viv_smap_clear(viv_smap* m) {
    for (long e = 0; e < m->n; e++) {
        if (!m->live[e]) continue;
        if (m->ks) free(m->k[e].s);
        if (m->vs) free(m->v[e].s);
    }
    free(m->slots); free(m->k); free(m->v); free(m->live);
    viv_smap_alloc(m, 8);
    return 0.0;
}
/* one key or value: a number as viv_pnum writes it, a string in its repr */
VIV_S_UNUSED static void viv_smap_put1(int str, viv_mv x, int kd, int p, int w) {
    if (!str) { viv_pnum(x.n, kd, p, w); return; }
    viv_sb b = {0, 0, 0, 0};
    viv_s_repr_into(&b, x.s);
    viv_fwrite(b.p, 1, (size_t)b.n, stdout);
    free(b.p);
}
VIV_S_UNUSED static void viv_psmap(viv_smap* m, int kk, int kp, int kw, int vk, int vp, int vw) {
    int first = 1;
    viv_putchar('{');
    for (long e = 0; e < m->n; e++) {
        if (!m->live[e]) continue;
        if (!first) viv_fputs(", ", stdout);
        first = 0;
        viv_smap_put1(m->ks, m->k[e], kk, kp, kw);
        viv_fputs(": ", stdout);
        viv_smap_put1(m->vs, m->v[e], vk, vp, vw);
    }
    viv_putchar('}');
}
/* print with a map of strings in it: viv_print_sm's tags, and 7 for a
   viv_smap and its two formats */
VIV_S_UNUSED static double viv_print_x(int n, ...) {
    va_list ap;
    va_start(ap, n);
    for (int k = 0; k < n; k++) {
        int tag = va_arg(ap, int);
        if (tag == 0) viv_fputs(va_arg(ap, const char*), stdout);
        else if (tag == 1) { double x = va_arg(ap, double); int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int); viv_pnum(x, kd, p, w); }
        else if (tag == 2) {
            viv_arr a = va_arg(ap, viv_arr);
            int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int);
            int one = a.size;
            viv_parr(a.data, a.rank <= 1 ? &one : a.dims, a.rank, kd, p, w);
        }
        else if (tag == 3) viv_fputs(va_arg(ap, double) != 0.0 ? "true" : "false", stdout);
        else if (tag == 5 || tag == 7) {
            void* m = va_arg(ap, void*);
            int kk = va_arg(ap, int), kp = va_arg(ap, int), kw = va_arg(ap, int);
            int vk = va_arg(ap, int), vp = va_arg(ap, int), vw = va_arg(ap, int);
            if (tag == 5) viv_pmap((viv_map*)m, kk, kp, kw, vk, vp, vw);
            else viv_psmap((viv_smap*)m, kk, kp, kw, vk, vp, vw);
        }
        else if (tag == 6) { viv_str s = VIV_S(va_arg(ap, viv_str)); viv_fwrite(s->s, 1, (size_t)s->len, stdout); }
        else {
            _Complex double z = va_arg(ap, _Complex double);
            double re = ((double*)&z)[0] + 0.0, im = ((double*)&z)[1] + 0.0;
            viv_pnum(re, 'b', 6, 0);
            viv_fputs(im < 0.0 ? "-" : "+", stdout);
            viv_pnum(im < 0.0 ? -im : im, 'b', 6, 0);
            viv_putchar('i');
        }
    }
    va_end(ap);
    viv_putchar('\n');
    return 0.0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn n(x: f64) -> MV {
        MV::N(x)
    }
    fn st(x: &str) -> MV {
        MV::S(Str::new(x.to_string()))
    }
    fn nums(c: Col) -> Vec<f64> {
        match c {
            Col::N(v) => v,
            Col::S(_) => panic!("number keys expected"),
        }
    }

    #[test]
    fn insertion_order_survives_removal_and_rebuild() {
        let mut m = MapData::new();
        for i in 0..100 {
            m.set(n(i as f64), n((i * i) as f64)).unwrap();
        }
        for i in (0..100).step_by(3) {
            assert!(matches!(m.remove(KR::N(i as f64)), Some(MV::N(x)) if x == (i * i) as f64));
        }
        m.set(n(0.0), n(-1.0)).unwrap();
        let ks = nums(m.keys());
        assert_eq!(ks.len(), 67);
        assert_eq!(*ks.last().unwrap(), 0.0);
        assert!(ks.windows(2).take(65).all(|w| w[0] < w[1]));
        for _ in 0..1000 {
            m.set(n(1e9), n(1.0)).unwrap();
            m.remove(KR::N(1e9));
        }
        assert_eq!(m.len(), 67);
        assert!(m.slots.len() <= 256, "removed entries are reclaimed by the rebuild");
    }

    #[test]
    fn signed_zero_is_one_key_and_the_first_is_kept() {
        let mut m = MapData::new();
        m.set(n(-0.0), n(1.0)).unwrap();
        m.set(n(0.0), n(2.0)).unwrap();
        assert_eq!(m.len(), 1);
        assert!(nums(m.keys())[0].is_sign_negative());
        assert!(matches!(m.get(KR::N(0.0)), Some(MV::N(x)) if x == 2.0));
        assert!(m.set(n(f64::NAN), n(1.0)).is_err());
        assert!(!m.has(KR::N(f64::NAN)));
        assert_eq!(missing(3.0), "the key 3 is not in the map");
        assert_eq!(missing(-0.0), "the key -0 is not in the map");
        assert_eq!(missing(2.5), "the key 2.5 is not in the map");
    }

    #[test]
    fn string_keys_find_by_text_and_keep_their_order() {
        let mut m = MapData::new();
        let words = ["b", "a", "", "héllo", "a\u{0}b", "日本"];
        for (i, w) in words.iter().enumerate() {
            m.set(st(w), n(i as f64)).unwrap();
        }
        assert!(matches!(m.get(KR::S("")), Some(MV::N(x)) if x == 2.0));
        assert!(matches!(m.get(KR::S("日本")), Some(MV::N(x)) if x == 5.0));
        assert!(!m.has(KR::S("c")));
        assert!(m.set(n(1.0), n(1.0)).is_err(), "one kind of key per map");
        m.remove(KR::S("a"));
        m.set(st("a"), n(9.0)).unwrap();
        assert_eq!(m.text(Fmt::BARE, Fmt::BARE), "{'b': 0, '': 2, 'héllo': 3, 'a\\x00b': 4, '日本': 5, 'a': 9}");
        assert_eq!(missing_key(KR::S("it's")), "the key \"it's\" is not in the map");
        let mut v = MapData::new();
        v.set(n(1.0), st("x")).unwrap();
        assert!(v.set(n(2.0), n(2.0)).is_err(), "one kind of value per map");
        assert_eq!(v.len(), 1);
        assert_eq!(v.text(Fmt::BARE, Fmt::BARE), "{1: 'x'}");
    }
}
