// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Strings: the rules, the interpreter's arms, and the
//! registry of builtins the lowering and the wasm backend read.
//!
//! This file is the one statement of what every string builtin does. The
//! interpreter calls the functions below on its `Val::Str`s; the wasm runtime
//! (`interp::wasm_str`) calls the same functions on strings in linear memory;
//! `strings.c` is their C twin, function for function, and is held to them by
//! the three-engine tests (`lang/viv/tests/strings.rs`) and by CPython's own
//! outputs (`tools/strings/cases.py`).
//!
//! Positions are code points (Python's), the text is UTF-8, and a string never
//! holds anything but valid UTF-8. Character data (what is a space, a letter,
//! an upper-case letter; what `upper` maps a character to) comes from
//! `unicode_tables`, which is generated from CPython.

use super::unicode_tables as ut;
use super::Val;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// The calls the parser writes for strings: an f-string (the concatenation
/// of its arguments), `a in b`, and the three ways an f-string field is
/// written (str, repr, fmt), under names a program's own defs cannot take.
pub const FSTR: &str = "viv__fstr";
pub const IN: &str = "viv__in";
pub const STR_OF: &str = "viv__str";
pub const REPR_OF: &str = "viv__repr";
pub const FMT_OF: &str = "viv__fmt";

// ---- values ------------------------------------------------------------------

/// A string value in the interpreter: its text, its length in code points, and
/// the index cache (the code point last found and its byte offset), which
/// makes `s[i]` over a run of i O(1) a step on text that is not ASCII.
#[derive(Debug)]
pub struct Str {
    pub text: String,
    pub nch: usize,
    cache: Cell<(usize, usize)>,
}

impl Str {
    pub fn new(text: String) -> Rc<Str> {
        let nch = count_chars(text.as_bytes());
        Rc::new(Str { text, nch, cache: Cell::new((0, 0)) })
    }
    pub fn view(&self) -> SV<'_> {
        SV { s: &self.text, nch: self.nch, cache: &self.cache }
    }
}

/// A list of strings in the interpreter.
pub type Strs = Rc<RefCell<Vec<Rc<Str>>>>;

pub fn strs(v: Vec<Rc<Str>>) -> Val {
    Val::Strs(Rc::new(RefCell::new(v)))
}

pub fn sval(s: String) -> Val {
    Val::Str(Str::new(s))
}

/// A string as the rules below read it: the text, its length in code points,
/// and where to keep the index cache (a `Str`'s own, or a copy of the one in a
/// wasm header, written back by the caller).
pub struct SV<'a> {
    pub s: &'a str,
    pub nch: usize,
    pub cache: &'a Cell<(usize, usize)>,
}

impl<'a> SV<'a> {
    pub fn ascii(&self) -> bool {
        self.nch == self.s.len()
    }
    /// The byte offset of code point `ci` (0 <= ci <= nch): from whichever of
    /// the start, the cached point and the end is nearest, and the point found
    /// cached for the next call.
    pub fn byte_of(&self, ci: usize) -> usize {
        if self.ascii() {
            return ci;
        }
        let b = self.s.as_bytes();
        let off = byte_of(b, self.nch, ci, self.cache.get());
        self.cache.set((ci, off));
        off
    }
    /// The byte range of code points [lo, hi).
    pub fn range(&self, lo: usize, hi: usize) -> (usize, usize) {
        let a = self.byte_of(lo);
        (a, if hi == lo { a } else { self.byte_of(hi) })
    }
}

pub fn count_chars(b: &[u8]) -> usize {
    b.iter().filter(|&&c| (c & 0xC0) != 0x80).count()
}

/// The byte offset of code point `ci` in the UTF-8 text `b` of `nch` code
/// points, walking from the nearest of 0, the cached (index, offset) and the
/// end. The C twin is `viv_s_byteof`.
pub fn byte_of(b: &[u8], nch: usize, ci: usize, cache: (usize, usize)) -> usize {
    let (cc, cb) = if cache.0 <= nch && cache.1 <= b.len() { cache } else { (0, 0) };
    // candidates: from the start (distance ci), from the cache (|ci - cc|),
    // from the end (nch - ci)
    let from_start = ci;
    let from_cache = ci.abs_diff(cc);
    let from_end = nch - ci;
    let (mut k, mut p) = if from_start <= from_cache && from_start <= from_end {
        (0usize, 0usize)
    } else if from_cache <= from_end {
        (cc, cb)
    } else {
        (nch, b.len())
    };
    while k < ci {
        p += 1;
        while p < b.len() && (b[p] & 0xC0) == 0x80 {
            p += 1;
        }
        k += 1;
    }
    while k > ci {
        p -= 1;
        while p > 0 && (b[p] & 0xC0) == 0x80 {
            p -= 1;
        }
        k -= 1;
    }
    p
}

// ---- character data ------------------------------------------------------------

fn in_ranges(c: u32, t: &[(u32, u32)]) -> bool {
    // the last range starting at or before c
    let k = t.partition_point(|r| r.0 <= c);
    k > 0 && c <= t[k - 1].1
}

pub fn is_space(c: char) -> bool {
    in_ranges(c as u32, ut::SPACE)
}
pub fn is_alpha(c: char) -> bool {
    in_ranges(c as u32, ut::ALPHA)
}
pub fn decimal(c: char) -> Option<u32> {
    let c = c as u32;
    let k = ut::DECIMAL.partition_point(|r| r.0 <= c);
    (k > 0 && c <= ut::DECIMAL[k - 1].1).then(|| c - ut::DECIMAL[k - 1].2)
}
pub fn is_digit(c: char) -> bool {
    in_ranges(c as u32, ut::DIGIT)
}
pub fn is_numeric(c: char) -> bool {
    in_ranges(c as u32, ut::NUMERIC)
}
pub fn is_alnum(c: char) -> bool {
    is_alpha(c) || decimal(c).is_some() || is_digit(c) || is_numeric(c)
}
pub fn is_upper(c: char) -> bool {
    in_ranges(c as u32, ut::UPPER)
}
pub fn is_lower(c: char) -> bool {
    in_ranges(c as u32, ut::LOWER)
}
pub fn is_title(c: char) -> bool {
    in_ranges(c as u32, ut::TITLE)
}
pub fn is_printable(c: char) -> bool {
    in_ranges(c as u32, ut::PRINTABLE)
}
fn is_case_ignorable(c: char) -> bool {
    in_ranges(c as u32, ut::CASE_IGNORABLE)
}
fn is_cased(c: char) -> bool {
    in_ranges(c as u32, ut::CASED)
}

/// Push the full case mapping of `c` from `table` onto `out`.
fn map_case(c: char, table: &[(u32, u32, u8)], pool: &[u32], out: &mut String) {
    let cp = c as u32;
    match table.binary_search_by_key(&cp, |r| r.0) {
        Ok(k) => {
            let (_, t, n) = table[k];
            if n == 1 {
                out.push(char::from_u32(t).expect("a table maps to characters"));
            } else {
                for &p in &pool[t as usize..t as usize + n as usize] {
                    out.push(char::from_u32(p).expect("a table maps to characters"));
                }
            }
        }
        Err(_) => out.push(c),
    }
}

/// Python's `str.upper`: every character's full upper-case mapping.
pub fn upper(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        map_case(c, ut::TO_UPPER, ut::TO_UPPER_POOL, &mut out);
    }
    out
}

/// Python's `str.lower`: full lower-case mappings, and a capital sigma lowered
/// to the final form where it ends a word (CPython's `handle_capital_sigma`).
pub fn lower(s: &str) -> String {
    let cs: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in cs.iter().enumerate() {
        if c == '\u{3A3}' {
            out.push(if final_sigma(&cs, i) { '\u{3C2}' } else { '\u{3C3}' });
        } else {
            map_case(c, ut::TO_LOWER, ut::TO_LOWER_POOL, &mut out);
        }
    }
    out
}

fn final_sigma(cs: &[char], i: usize) -> bool {
    let mut j = i;
    let mut before = None;
    while j > 0 {
        j -= 1;
        if !is_case_ignorable(cs[j]) {
            before = Some(cs[j]);
            break;
        }
    }
    let Some(b) = before else { return false };
    if !is_cased(b) {
        return false;
    }
    for &c in &cs[i + 1..] {
        if !is_case_ignorable(c) {
            return !is_cased(c);
        }
    }
    true
}

// ---- predicates ----------------------------------------------------------------

/// `isspace`, `isalpha`, ...: every character has the property, and there is
/// at least one.
fn all_of(s: &str, f: fn(char) -> bool) -> bool {
    !s.is_empty() && s.chars().all(f)
}

pub fn pred(name: &str, s: &str) -> bool {
    match name {
        "isspace" => all_of(s, is_space),
        "isalpha" => all_of(s, is_alpha),
        "isalnum" => all_of(s, is_alnum),
        "isdigit" => all_of(s, is_digit),
        "isdecimal" => all_of(s, |c| decimal(c).is_some()),
        "isnumeric" => all_of(s, is_numeric),
        "isascii" => s.is_ascii(),
        "isprintable" => s.chars().all(is_printable),
        // CPython's unicode_isupper / unicode_islower
        "isupper" => {
            let mut cased = false;
            for c in s.chars() {
                if is_lower(c) || is_title(c) {
                    return false;
                }
                cased |= is_upper(c);
            }
            cased
        }
        "islower" => {
            let mut cased = false;
            for c in s.chars() {
                if is_upper(c) || is_title(c) {
                    return false;
                }
                cased |= is_lower(c);
            }
            cased
        }
        _ => unreachable!("not a predicate: {name}"),
    }
}

pub const PREDICATES: [&str; 10] = ["isspace", "isalpha", "isalnum", "isdigit", "isdecimal", "isnumeric", "isupper", "islower", "isascii", "isprintable"];

// ---- repr ----------------------------------------------------------------------

/// Python's `repr(str)`: in single quotes, or double quotes when the text has
/// a single quote and no double quote; backslash escapes for the backslash,
/// the quote, `\t` `\n` `\r`, and for every character that is not printable.
pub fn repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) < 0x80 => out.push(c),
            c if is_printable(c) => out.push(c),
            c if (c as u32) < 0x100 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) < 0x10000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(q);
    out
}

/// A list of strings as Python prints a list: `['a', 'b']`.
pub fn list_repr(items: &[&str]) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&repr(s));
    }
    out.push(']');
    out
}

// ---- positions -------------------------------------------------------------------

/// A number read as a position or a count: toward zero, NaN as 0, saturating
/// (Rust's `as`, which is how an array index is read).
pub fn as_int(x: f64) -> i64 {
    x as i64
}

/// Python's index into a sequence of `n`: negative counts from the end.
pub fn py_index(n: usize, i: f64) -> Option<usize> {
    let mut i = as_int(i);
    if i < 0 {
        i = i.saturating_add(n as i64);
    }
    (i >= 0 && (i as usize) < n).then_some(i as usize)
}

/// A slice as the parser writes it: `kind` 1 both bounds written, 2 no end,
/// 3 no start, 4 neither (3 and 4 only when a step that is not a positive
/// number is written; otherwise a missing start is 0). CPython's
/// PySlice_Unpack and PySlice_AdjustIndices: (start, step, count).
pub fn py_slice(n: usize, kind: f64, lo: f64, hi: f64, step: f64) -> Result<(i64, i64, usize), String> {
    let step = as_int(step);
    if step == 0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "slice step cannot be zero"));
    }
    let n = n as i64;
    let kind = kind as i64;
    let lo_given = kind == 1 || kind == 2;
    let hi_given = kind == 1 || kind == 3;
    let adjust = |x: Option<i64>, is_start: bool| -> i64 {
        match x {
            None => {
                if is_start {
                    if step < 0 { n - 1 } else { 0 }
                } else if step < 0 {
                    -1
                } else {
                    n
                }
            }
            Some(mut v) => {
                if v < 0 {
                    v = v.saturating_add(n);
                    if v < 0 {
                        v = if step < 0 { -1 } else { 0 };
                    }
                } else if v >= n {
                    v = if step < 0 { n - 1 } else { n };
                }
                v
            }
        }
    };
    let start = adjust(lo_given.then(|| as_int(lo)), true);
    let stop = adjust(hi_given.then(|| as_int(hi)), false);
    let count = if step < 0 {
        if stop < start { (start - stop - 1) / (-step) + 1 } else { 0 }
    } else if start < stop {
        (stop - start - 1) / step + 1
    } else {
        0
    };
    Ok((start, step, count as usize))
}

/// `s[i]` as a byte range of `v.s`.
pub fn char_at(v: &SV, i: f64) -> Result<(usize, usize), String> {
    crate::interp::index_int(i)?;
    let Some(k) = py_index(v.nch, i) else { return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, "string index out of range")) };
    Ok(v.range(k, k + 1))
}

/// `s[lo:hi:step]`.
pub fn slice(v: &SV, spec: [f64; 4]) -> Result<String, String> {
    let (start, step, count) = py_slice(v.nch, spec[0], spec[1], spec[2], spec[3])?;
    if count == 0 {
        return Ok(String::new());
    }
    if step == 1 {
        let (a, b) = v.range(start as usize, start as usize + count);
        return Ok(v.s[a..b].to_string());
    }
    if v.ascii() {
        let b = v.s.as_bytes();
        let mut out = String::with_capacity(count);
        for k in 0..count {
            out.push(b[(start + k as i64 * step) as usize] as char);
        }
        return Ok(out);
    }
    let cs: Vec<char> = v.s.chars().collect();
    let mut out = String::new();
    for k in 0..count {
        out.push(cs[(start + k as i64 * step) as usize]);
    }
    Ok(out)
}

/// The same for a list of `n` items: the indices taken, in order.
pub fn slice_indices(n: usize, spec: [f64; 4]) -> Result<Vec<usize>, String> {
    let (start, step, count) = py_slice(n, spec[0], spec[1], spec[2], spec[3])?;
    Ok((0..count).map(|k| (start + k as i64 * step) as usize).collect())
}

// ---- building ------------------------------------------------------------------

/// `s * n` (and `n * s`): n copies; none for n <= 0.
pub fn repeat(s: &str, n: f64) -> Result<String, String> {
    let n = as_int(n);
    if n <= 0 || s.is_empty() {
        return Ok(String::new());
    }
    let total = (s.len() as u128) * (n as u128);
    if total > MAX_BYTES as u128 {
        return Err(too_long(total));
    }
    Ok(s.repeat(n as usize))
}

/// The longest string any engine makes: 2^31 - 1 bytes less headroom, so a
/// length always fits the C runtime's `int` and wasm's i32 arithmetic.
pub const MAX_BYTES: usize = (1 << 30) - 1;

pub fn too_long(_n: u128) -> String {
    crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("the string would be longer than Viv's limit of {MAX_BYTES} bytes"))
}

pub fn concat(a: &str, b: &str) -> Result<String, String> {
    if a.len() + b.len() > MAX_BYTES {
        return Err(too_long((a.len() + b.len()) as u128));
    }
    let mut s = String::with_capacity(a.len() + b.len());
    s.push_str(a);
    s.push_str(b);
    Ok(s)
}

/// `chr(n)`: the character with code point n.
pub fn chr(n: f64) -> Result<String, String> {
    let k = as_int(n);
    if !(0..0x110000).contains(&k) {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "chr() arg not in range(0x110000)"));
    }
    match char::from_u32(k as u32) {
        Some(c) => Ok(c.to_string()),
        None => Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("chr({k:#x}) is a surrogate, which a Viv string (UTF-8) cannot hold"))),
    }
}

/// `ord(c)`: the code point of a one-character string.
pub fn ord(v: &SV) -> Result<f64, String> {
    if v.nch != 1 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("ord() expected a character, but string of length {} found", v.nch)));
    }
    Ok(v.s.chars().next().expect("one character") as u32 as f64)
}

// ---- whitespace, stripping, splitting ------------------------------------------------

/// What `strip` takes away: whitespace (Python's `str.isspace`), or the
/// characters of `chars`.
fn strip_set<'a>(chars: Option<&'a str>) -> impl Fn(char) -> bool + 'a {
    move |c| match chars {
        None => is_space(c),
        Some(set) => set.contains(c),
    }
}

/// `strip`, `lstrip`, `rstrip`: which = 0 both ends, 1 left, 2 right.
pub fn strip<'s>(s: &'s str, chars: Option<&str>, which: u8) -> &'s str {
    let f = strip_set(chars);
    let mut t = s;
    if which != 2 {
        t = t.trim_start_matches(|c| f(c));
    }
    if which != 1 {
        t = t.trim_end_matches(|c| f(c));
    }
    t
}

/// A `maxsplit` argument: the number of splits allowed (a negative number is
/// no limit).
fn max_count(n: f64) -> usize {
    let n = as_int(n);
    if n < 0 { usize::MAX } else { n as usize }
}

/// `split(s)` / `split(s, sep, maxsplit)` / `rsplit(...)`: CPython's
/// split_whitespace, split, rsplit_whitespace and rsplit. `sep` None is the
/// whitespace form.
pub fn split<'s>(s: &'s str, sep: Option<&str>, maxsplit: f64, right: bool) -> Result<Vec<&'s str>, String> {
    let mut max = max_count(maxsplit);
    let mut out: Vec<&str> = Vec::new();
    match sep {
        Some("") => return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "empty separator")),
        Some(sep) if !right => {
            let mut rest = s;
            while max > 0 {
                let Some(k) = rest.find(sep) else { break };
                out.push(&rest[..k]);
                rest = &rest[k + sep.len()..];
                max -= 1;
            }
            out.push(rest);
        }
        Some(sep) => {
            let mut rest = s;
            while max > 0 {
                let Some(k) = rest.rfind(sep) else { break };
                out.push(&rest[k + sep.len()..]);
                rest = &rest[..k];
                max -= 1;
            }
            out.push(rest);
            out.reverse();
        }
        None if !right => {
            // (byte positions of each character, its length)
            let b = s;
            let mut i = 0usize;
            let n = b.len();
            let sp = |i: usize| -> (bool, usize) {
                let c = b[i..].chars().next().expect("inside the text");
                (is_space(c), c.len_utf8())
            };
            while max > 0 {
                while i < n && sp(i).0 {
                    i += sp(i).1;
                }
                if i == n {
                    break;
                }
                let j = i;
                i += sp(i).1;
                while i < n && !sp(i).0 {
                    i += sp(i).1;
                }
                out.push(&b[j..i]);
                max -= 1;
            }
            if i < n {
                while i < n && sp(i).0 {
                    i += sp(i).1;
                }
                if i != n {
                    out.push(&b[i..]);
                }
            }
        }
        None => {
            let b = s;
            let mut i = b.len();
            let sp_before = |i: usize| -> (bool, usize) {
                let c = b[..i].chars().next_back().expect("inside the text");
                (is_space(c), c.len_utf8())
            };
            while max > 0 {
                while i > 0 && sp_before(i).0 {
                    i -= sp_before(i).1;
                }
                if i == 0 {
                    break;
                }
                let j = i;
                i -= sp_before(i).1;
                while i > 0 && !sp_before(i).0 {
                    i -= sp_before(i).1;
                }
                out.push(&b[i..j]);
                max -= 1;
            }
            if i > 0 {
                while i > 0 && sp_before(i).0 {
                    i -= sp_before(i).1;
                }
                if i != 0 {
                    out.push(&b[..i]);
                }
            }
            out.reverse();
        }
    }
    Ok(out)
}

/// Python's line boundaries (`str.splitlines`).
pub fn is_linebreak(c: char) -> bool {
    matches!(c, '\n' | '\u{b}' | '\u{c}' | '\r' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// `splitlines(s)`: CPython's stringlib splitlines, without the line ends.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let n = s.len();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n {
        loop {
            if i >= n {
                break;
            }
            let c = s[i..].chars().next().expect("inside the text");
            if is_linebreak(c) {
                break;
            }
            i += c.len_utf8();
        }
        let eol = i;
        if i < n {
            let c = s[i..].chars().next().expect("inside the text");
            if c == '\r' && i + 1 < n && b[i + 1] == b'\n' {
                i += 2;
            } else {
                i += c.len_utf8();
            }
        }
        out.push(&s[j..eol]);
        j = i;
    }
    out
}

/// `join(sep, items)`.
pub fn join(sep: &str, items: &[&str]) -> Result<String, String> {
    let total = items.iter().map(|s| s.len()).sum::<usize>() + sep.len() * items.len().saturating_sub(1);
    if total > MAX_BYTES {
        return Err(too_long(total as u128));
    }
    let mut out = String::with_capacity(total);
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        out.push_str(s);
    }
    Ok(out)
}

// ---- searching -------------------------------------------------------------------

/// CPython's ADJUST_INDICES: `start` and `end` of a search, in code points,
/// clamped into [0, len]. `end` NaN means "not given" (the length).
fn adjust(len: usize, start: f64, end: f64) -> (i64, i64) {
    let n = len as i64;
    let mut s = as_int(start);
    let mut e = if end.is_nan() { n } else { as_int(end) };
    if e > n {
        e = n;
    } else if e < 0 {
        e = e.saturating_add(n);
        if e < 0 {
            e = 0;
        }
    }
    if s < 0 {
        s = s.saturating_add(n);
        if s < 0 {
            s = 0;
        }
    }
    (s, e)
}

/// `find`, `rfind`: the code point index of the first (last) occurrence of
/// `sub` in `s[start:end]`, or -1.
pub fn find(v: &SV, sub: &SV, start: f64, end: f64, right: bool) -> f64 {
    let (s, e) = adjust(v.nch, start, end);
    if e - s < sub.nch as i64 {
        return -1.0;
    }
    let (a, b) = v.range(s as usize, e as usize);
    let hay = &v.s[a..b];
    let hit = if right { hay.rfind(sub.s) } else { hay.find(sub.s) };
    match hit {
        None => -1.0,
        Some(k) => (s as usize + count_chars(&hay.as_bytes()[..k])) as f64,
    }
}

/// `index`, `rindex`: `find`, but a miss stops the run.
pub fn index(v: &SV, sub: &SV, start: f64, end: f64, right: bool) -> Result<f64, String> {
    let k = find(v, sub, start, end, right);
    if k < 0.0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "substring not found"));
    }
    Ok(k)
}

/// `count`: non-overlapping occurrences of `sub` in `s[start:end]`.
pub fn count(v: &SV, sub: &SV, start: f64, end: f64) -> f64 {
    let (s, e) = adjust(v.nch, start, end);
    if e - s < sub.nch as i64 {
        return 0.0;
    }
    if sub.s.is_empty() {
        return (e - s + 1) as f64;
    }
    let (a, b) = v.range(s as usize, e as usize);
    v.s[a..b].matches(sub.s).count() as f64
}

/// `replace(s, old, new, count)`: CPython's; an empty `old` puts `new` before
/// every character and at the end.
pub fn replace(v: &SV, old: &str, new: &str, count: f64) -> Result<String, String> {
    let max = max_count(count);
    let s = v.s;
    if max == 0 {
        return Ok(s.to_string());
    }
    let mut out = String::new();
    let limit = |len: usize| -> Result<(), String> { if len > MAX_BYTES { Err(too_long(len as u128)) } else { Ok(()) } };
    if old.is_empty() {
        let mut left = max;
        for c in s.chars() {
            if left > 0 {
                out.push_str(new);
                left -= 1;
            }
            out.push(c);
            limit(out.len())?;
        }
        if left > 0 {
            out.push_str(new);
        }
        limit(out.len())?;
        return Ok(out);
    }
    let mut rest = s;
    let mut left = max;
    while left > 0 {
        let Some(k) = rest.find(old) else { break };
        out.push_str(&rest[..k]);
        out.push_str(new);
        limit(out.len())?;
        rest = &rest[k + old.len()..];
        left -= 1;
    }
    out.push_str(rest);
    limit(out.len())?;
    Ok(out)
}

// ---- padding ---------------------------------------------------------------------------

fn fill_char(fill: &SV) -> Result<char, String> {
    if fill.nch != 1 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "The fill character must be exactly one character long"));
    }
    Ok(fill.s.chars().next().expect("one character"))
}

/// `ljust`, `rjust`, `center`: which 0, 1, 2. CPython's pad(): `center` puts
/// the odd space on the left when the width is odd and the margin is odd.
pub fn pad(v: &SV, width: f64, fill: &SV, which: u8) -> Result<String, String> {
    let c = fill_char(fill)?;
    let w = as_int(width);
    let n = v.nch as i64;
    if w <= n {
        return Ok(v.s.to_string());
    }
    let marg = w - n;
    let (left, right) = match which {
        0 => (0, marg),
        1 => (marg, 0),
        _ => {
            let left = marg / 2 + (marg & w & 1);
            (left, marg - left)
        }
    };
    if (w as u128) * 4 > MAX_BYTES as u128 {
        return Err(too_long(w as u128));
    }
    let mut out = String::new();
    for _ in 0..left {
        out.push(c);
    }
    out.push_str(v.s);
    for _ in 0..right {
        out.push(c);
    }
    Ok(out)
}

/// `zfill(s, width)`: zeros on the left, after a leading sign.
pub fn zfill(v: &SV, width: f64) -> Result<String, String> {
    let w = as_int(width);
    let n = v.nch as i64;
    if w <= n {
        return Ok(v.s.to_string());
    }
    if w > MAX_BYTES as i64 {
        return Err(too_long(w as u128));
    }
    let fill = (w - n) as usize;
    let mut out = String::with_capacity(v.s.len() + fill);
    let (sign, rest) = match v.s.as_bytes().first() {
        Some(b'+') | Some(b'-') => (&v.s[..1], &v.s[1..]),
        _ => ("", v.s),
    };
    out.push_str(sign);
    for _ in 0..fill {
        out.push('0');
    }
    out.push_str(rest);
    Ok(out)
}

// ---- numbers from text -------------------------------------------------------------------

/// CPython's _PyUnicode_TransformDecimalAndSpaceToASCII: every character
/// above U+007E that is whitespace becomes a space and every decimal digit its
/// ASCII digit; at the first other one the text is cut and ends in `?` (which
/// no number reads).
fn to_ascii_number_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) < 127 {
            out.push(c);
        } else if is_space(c) {
            out.push(' ');
        } else if let Some(d) = decimal(c) {
            out.push((b'0' + d as u8) as char);
        } else {
            out.push('?');
            break;
        }
    }
    out
}

/// Python's Py_ISSPACE: the ASCII whitespace.
fn ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

/// CPython's _Py_string_to_number_with_underscores: an underscore only
/// between two digits; the text without them, or None.
fn drop_underscores(t: &str) -> Option<String> {
    if !t.contains('_') {
        return Some(t.to_string());
    }
    let mut out = String::with_capacity(t.len());
    let mut prev = 0u8;
    for &c in t.as_bytes() {
        if c == b'_' {
            if !prev.is_ascii_digit() {
                return None;
            }
        } else {
            out.push(c as char);
            if prev == b'_' && !c.is_ascii_digit() {
                return None;
            }
        }
        prev = c;
    }
    if prev == b'_' {
        return None;
    }
    Some(out)
}

/// The decimal a text holds, checked as _Py_dg_strtod and
/// _Py_parse_inf_or_nan read it, and written as `[-]digits[.digits][e[-]digits]`
/// for the correctly rounded parser (Rust's `parse`, C's `strtod`). None when
/// the text is not all a number.
pub fn plain_decimal(t: &str) -> Option<Result<String, f64>> {
    let b = t.as_bytes();
    let mut i = 0;
    let neg = match b.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let rest = &t[i..];
    let lower = rest.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(Err(if neg { f64::NEG_INFINITY } else { f64::INFINITY }));
    }
    if lower == "nan" {
        return Some(Err(if neg { -f64::NAN } else { f64::NAN }));
    }
    let mut int_digits = 0;
    let mut frac_digits = 0;
    let mut out = String::with_capacity(t.len() + 2);
    if neg {
        out.push('-');
    }
    while i < b.len() && b[i].is_ascii_digit() {
        out.push(b[i] as char);
        i += 1;
        int_digits += 1;
    }
    if int_digits == 0 {
        out.push('0');
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let mark = out.len();
        out.push('.');
        while i < b.len() && b[i].is_ascii_digit() {
            out.push(b[i] as char);
            i += 1;
            frac_digits += 1;
        }
        if frac_digits == 0 {
            out.truncate(mark);
        }
    }
    if int_digits + frac_digits == 0 {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        let mut exp = String::from("e");
        if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
            if b[j] == b'-' {
                exp.push('-');
            }
            j += 1;
        }
        let d0 = j;
        // leading zeros of the exponent go, so the text stays short
        while j < b.len() && b[j] == b'0' {
            j += 1;
        }
        let digits_from = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > d0 {
            let digits = &t[digits_from..j];
            if digits.len() > 9 {
                // an exponent this long is past every double: its sign decides
                exp.push_str("999999999");
            } else if digits.is_empty() {
                exp.push('0');
            } else {
                exp.push_str(digits);
            }
            out.push_str(&exp);
            i = j;
        }
    }
    if i != b.len() {
        return None;
    }
    Some(Ok(out))
}

/// `float(s)`: CPython's float(str), with its message.
pub fn parse_float(s: &str) -> Result<f64, String> {
    let bad = || crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("could not convert string to float: {}", repr(s)));
    let t = to_ascii_number_text(s);
    let t = drop_underscores(&t).ok_or_else(bad)?;
    let t = t.trim_matches(|c: char| c.is_ascii() && ascii_space(c as u8));
    if t.is_empty() {
        return Err(bad());
    }
    match plain_decimal(t) {
        None => Err(bad()),
        Some(Err(special)) => Ok(special),
        Some(Ok(text)) => Ok(text.parse::<f64>().expect("a plain decimal parses")),
    }
}

/// A digit's value in bases up to 36 (CPython's _PyLong_DigitValue), or 37.
fn digit_value(b: u8) -> u32 {
    match b {
        b'0'..=b'9' => (b - b'0') as u32,
        b'a'..=b'z' => (b - b'a') as u32 + 10,
        b'A'..=b'Z' => (b - b'A') as u32 + 10,
        _ => 37,
    }
}

/// CPython 3.11's limit on the digits it converts in a base that is not a
/// power of two (sys.int_info.default_max_str_digits).
pub const MAX_STR_DIGITS: usize = 4300;

/// `int(s, base)`: CPython's PyLong_FromString on the transformed text, the
/// integer rounded to the nearest double (ties to even).
pub fn parse_int(s: &str, base: f64) -> Result<f64, String> {
    let base_i = as_int(base);
    if base_i != 0 && !(2..=36).contains(&base_i) {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "int() base must be >= 2 and <= 36, or 0"));
    }
    let mut base = base_i as u32;
    let t = to_ascii_number_text(s);
    let b = t.as_bytes();
    let mut i = 0;
    let bad = |base: u32| {
        let r = repr(s);
        let r: String = if r.chars().count() > 200 { r.chars().take(200).collect() } else { r };
        crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("invalid literal for int() with base {base}: {r}"))
    };
    while i < b.len() && ascii_space(b[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    let at = |k: usize| b.get(k).copied().unwrap_or(0);
    let mut error_if_nonzero = false;
    if base == 0 {
        if at(i) != b'0' {
            base = 10;
        } else if matches!(at(i + 1), b'x' | b'X') {
            base = 16;
        } else if matches!(at(i + 1), b'o' | b'O') {
            base = 8;
        } else if matches!(at(i + 1), b'b' | b'B') {
            base = 2;
        } else {
            error_if_nonzero = true;
            base = 10;
        }
    }
    let shown = if error_if_nonzero { 0 } else { base };
    if at(i) == b'0'
        && ((base == 16 && matches!(at(i + 1), b'x' | b'X')) || (base == 8 && matches!(at(i + 1), b'o' | b'O')) || (base == 2 && matches!(at(i + 1), b'b' | b'B')))
    {
        i += 2;
        if at(i) == b'_' {
            i += 1;
        }
    }
    if at(i) == b'_' {
        return Err(bad(base_i as u32));
    }
    let mut digits: Vec<u8> = Vec::new();
    let mut prev = 0u8;
    while i < b.len() && (digit_value(b[i]) < base || b[i] == b'_') {
        if b[i] == b'_' {
            if prev == b'_' {
                return Err(bad(base_i as u32));
            }
        } else {
            digits.push(digit_value(b[i]) as u8);
        }
        prev = b[i];
        i += 1;
    }
    if prev == b'_' {
        return Err(bad(base_i as u32));
    }
    if digits.is_empty() {
        return Err(bad(base_i as u32));
    }
    if !base.is_power_of_two() && digits.len() > MAX_STR_DIGITS {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!(
            "Exceeds the limit ({MAX_STR_DIGITS} digits) for integer string conversion: value has {} digits; use sys.set_int_max_str_digits() to increase the limit",
            digits.len()
        )));
    }
    let zero = digits.iter().all(|&d| d == 0);
    if error_if_nonzero && !zero {
        return Err(bad(shown));
    }
    while i < b.len() && ascii_space(b[i]) {
        i += 1;
    }
    if i != b.len() {
        return Err(bad(shown));
    }
    let x = digits_to_f64(&digits, base).ok_or_else(|| crate::errkind::kinded(crate::errkind::ErrKind::Overflow, "int too large to convert to float"))?;
    Ok(if neg && x != 0.0 { -x } else { x })
}

/// The integer with these digits in `base`, rounded to the nearest double,
/// ties to even (CPython's int -> float), or None when that is beyond the
/// largest double: a big integer in 32-bit limbs, then its top 64 bits and a
/// sticky bit for the rest. The C twin is `viv_s_digits`.
pub fn digits_to_f64(digits: &[u8], base: u32) -> Option<f64> {
    // little-endian limbs
    let mut limbs: Vec<u32> = vec![0];
    for &d in digits {
        let mut carry = d as u64;
        for l in limbs.iter_mut() {
            let v = (*l as u64) * base as u64 + carry;
            *l = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            limbs.push(carry as u32);
        }
    }
    while limbs.len() > 1 && limbs[limbs.len() - 1] == 0 {
        limbs.pop();
    }
    let top = limbs.len() - 1;
    let bits = 32 * top as u32 + (32 - limbs[top].leading_zeros());
    if bits <= 64 {
        let mut v: u64 = 0;
        for (k, &l) in limbs.iter().enumerate() {
            v |= (l as u64) << (32 * k);
        }
        // `as` rounds a u64 to the nearest double, ties to even
        return Some(v as f64);
    }
    let bit = |k: u32| (limbs[(k / 32) as usize] >> (k % 32)) & 1;
    let mut m: u64 = 0;
    for k in 0..64 {
        m = (m << 1) | bit(bits - 1 - k) as u64;
    }
    let low = bits - 64;
    let sticky = (0..low).any(|k| bit(k) == 1);
    // m has its top bit set: keep 53 bits, round on the 11 below and the sticky bit
    let mut keep = m >> 11;
    let rest = m & 0x7ff;
    if rest > 0x400 || (rest == 0x400 && (sticky || keep & 1 == 1)) {
        keep += 1;
    }
    let e = low + 11;
    if e > 1100 {
        return None;
    }
    let x = keep as f64 * 2f64.powi(e as i32);
    x.is_finite().then_some(x)
}

/// `int(x)` of a number: toward zero, a zero without its sign.
pub fn int_of_num(x: f64) -> Result<f64, String> {
    if x.is_nan() {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "cannot convert float NaN to integer"));
    }
    if x.is_infinite() {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, "cannot convert float infinity to integer"));
    }
    Ok(x.trunc() + 0.0)
}

// ---- hashing and order ----------------------------------------------------------------

/// The hash of a string, in every engine: 64-bit FNV-1a of its UTF-8 bytes.
/// The C twin is `viv_str_hash`; a map with string keys probes by it.
pub fn hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Python's order of strings: by code point, which is the order of their
/// UTF-8 bytes.
pub fn cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.as_bytes().cmp(b.as_bytes())
}

// ---- format specs for strings -----------------------------------------------------------

/// A string format spec, `[[fill]align][width][.precision][s]` (Python's
/// format(str, spec)). Checked when the program is compiled; the defaults are
/// fill ' ', align '<'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrSpec {
    pub fill: char,
    pub align: u8,
    pub width: u32,
    /// u32::MAX when none is written
    pub prec: u32,
    /// the width (1) and the precision (2) are values, `*` in the spec
    /// (an f-string's nested field, `{s:>{w}}`)
    pub stars: u8,
}

impl StrSpec {
    pub fn parse(spec: &str) -> Result<StrSpec, String> {
        let cs: Vec<char> = spec.chars().collect();
        let mut i = 0;
        let mut fill = ' ';
        let mut align = b'<';
        let is_align = |c: char| matches!(c, '<' | '>' | '^' | '=');
        if cs.len() >= 2 && is_align(cs[1]) {
            fill = cs[0];
            align = cs[1] as u8;
            i = 2;
        } else if !cs.is_empty() && is_align(cs[0]) {
            align = cs[0] as u8;
            i = 1;
        }
        if align == b'=' {
            return Err("'=' alignment not allowed in string format specifier".into());
        }
        if i < cs.len() && matches!(cs[i], '+' | '-' | ' ') {
            return Err("Sign not allowed in string format specifier".into());
        }
        // the zero flag fills with zeros when no fill is written (Python:
        // f"{'ab':05}" is "ab000")
        if i < cs.len() && cs[i] == '0' && !(cs.len() >= 2 && is_align(cs[1])) {
            fill = '0';
            if cs.get(i + 1) == Some(&'*') {
                i += 1;
            }
        }
        let digits = |i: &mut usize| -> Option<u32> {
            let start = *i;
            while *i < cs.len() && cs[*i].is_ascii_digit() {
                *i += 1;
            }
            (*i > start).then(|| cs[start..*i].iter().collect::<String>().parse().unwrap_or(u32::MAX))
        };
        let mut stars = 0;
        let width = if cs.get(i) == Some(&'*') {
            i += 1;
            stars |= 1;
            0
        } else {
            digits(&mut i).unwrap_or(0)
        };
        let mut prec = u32::MAX;
        if i < cs.len() && cs[i] == '.' {
            i += 1;
            if cs.get(i) == Some(&'*') {
                i += 1;
                stars |= 2;
                prec = 0;
            } else {
                prec = digits(&mut i).ok_or("Format specifier missing precision")?;
            }
        }
        let rest: String = cs[i..].iter().collect();
        if !(rest.is_empty() || rest == "s") {
            return Err(format!("Unknown format code '{}' for object of type 'str'", rest.chars().next().expect("a code")));
        }
        if width > crate::printfmt::MAX_WIDTH {
            return Err(format!("format \"{spec}\": the width is at most {}", crate::printfmt::MAX_WIDTH));
        }
        Ok(StrSpec { fill, align, width, prec, stars })
    }
    pub fn star_count(&self) -> usize {
        (self.stars & 1) as usize + (self.stars >> 1 & 1) as usize
    }
    /// The spec with its `*` width and precision given values, checked as
    /// a number format's are (`Fmt::with_stars`).
    pub fn with_stars(mut self, vals: &[f64]) -> Result<StrSpec, String> {
        use crate::printfmt::{Fmt, FL_STAR_PREC, FL_STAR_WIDTH};
        let flags = if self.stars & 1 != 0 { FL_STAR_WIDTH } else { 0 } | if self.stars & 2 != 0 { FL_STAR_PREC } else { 0 };
        let f = Fmt { flags, ..Fmt::REPR }.with_limits(vals, crate::printfmt::STR_MAX_PREC)?;
        if self.stars & 1 != 0 {
            self.width = f.width;
        }
        if self.stars & 2 != 0 {
            self.prec = f.prec;
        }
        self.stars = 0;
        Ok(self)
    }
    /// The four numbers the compiled engines carry it as.
    pub fn encode(&self) -> [f64; 4] {
        [self.fill as u32 as f64, self.align as f64, self.width as f64, if self.prec == u32::MAX { -1.0 } else { self.prec as f64 }]
    }
    pub fn decode(v: [f64; 4]) -> StrSpec {
        StrSpec {
            fill: char::from_u32(v[0] as u32).unwrap_or(' '),
            align: v[1] as u8,
            width: v[2] as u32,
            prec: if v[3] < 0.0 { u32::MAX } else { v[3] as u32 },
            stars: 0,
        }
    }
}

/// `fmt(s, spec)`: truncated to the precision, then padded to the width.
pub fn format_str(v: &SV, spec: StrSpec) -> String {
    let (body, n) = if (spec.prec as usize) < v.nch {
        let (_, b) = v.range(0, spec.prec as usize);
        (&v.s[..b], spec.prec as usize)
    } else {
        (v.s, v.nch)
    };
    let w = spec.width as usize;
    if n >= w {
        return body.to_string();
    }
    let marg = w - n;
    let (left, right) = match spec.align {
        b'>' => (marg, 0),
        b'^' => (marg / 2, marg - marg / 2),
        _ => (0, marg),
    };
    let mut out = String::with_capacity(body.len() + marg * spec.fill.len_utf8());
    for _ in 0..left {
        out.push(spec.fill);
    }
    out.push_str(body);
    for _ in 0..right {
        out.push(spec.fill);
    }
    out
}

// ---- the registry: every string builtin a call can name --------------------------

/// What an argument or a result is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum K {
    /// a number
    N,
    /// a boolean
    B,
    /// a string
    S,
    /// a list of strings
    L,
    /// an array of numbers, rank 1 (the `db_*` builtins only, crate::db)
    A,
    /// an array of Ints (the `db_*` builtins only)
    I,
}

/// The value an optional argument takes when it is left out.
#[derive(Clone, Copy, Debug)]
pub enum D {
    N(f64),
    S(&'static str),
}

/// One string builtin: its Viv name, the kinds of its arguments (the last
/// `opt.len()` of them optional, with those defaults), what it gives back, and
/// the name of its C function, which is also the typed IR's callee and, with
/// `viv_` changed to `vrt_`, the wasm runtime's import.
#[derive(Debug)]
pub struct Builtin {
    pub name: &'static str,
    pub args: &'static [K],
    pub opt: &'static [D],
    pub ret: K,
    pub c: &'static str,
}

use K::{A as KA, B as KB, I as KI, L as KL, N as KN, S as KS};

const NOT_GIVEN: D = D::N(f64::NAN);

macro_rules! b {
    ($name:expr, [$($a:expr),*], [$($o:expr),*], $ret:expr, $c:expr) => {
        Builtin { name: $name, args: &[$($a),*], opt: &[$($o),*], ret: $ret, c: $c }
    };
}

pub const BUILTINS: &[Builtin] = &[
    b!("upper", [KS], [], KS, "viv_s_upper"),
    b!("lower", [KS], [], KS, "viv_s_lower"),
    b!("isspace", [KS], [], KB, "viv_s_isspace"),
    b!("isalpha", [KS], [], KB, "viv_s_isalpha"),
    b!("isalnum", [KS], [], KB, "viv_s_isalnum"),
    b!("isdigit", [KS], [], KB, "viv_s_isdigit"),
    b!("isdecimal", [KS], [], KB, "viv_s_isdecimal"),
    b!("isnumeric", [KS], [], KB, "viv_s_isnumeric"),
    b!("isupper", [KS], [], KB, "viv_s_isupper"),
    b!("islower", [KS], [], KB, "viv_s_islower"),
    b!("isascii", [KS], [], KB, "viv_s_isascii"),
    b!("isprintable", [KS], [], KB, "viv_s_isprintable"),
    b!("strip", [KS], [], KS, "viv_s_strip"),
    b!("strip", [KS, KS], [], KS, "viv_s_strip_chars"),
    b!("lstrip", [KS], [], KS, "viv_s_lstrip"),
    b!("lstrip", [KS, KS], [], KS, "viv_s_lstrip_chars"),
    b!("rstrip", [KS], [], KS, "viv_s_rstrip"),
    b!("rstrip", [KS, KS], [], KS, "viv_s_rstrip_chars"),
    b!("split", [KS], [], KL, "viv_s_split_ws"),
    b!("split", [KS, KS, KN], [D::N(-1.0)], KL, "viv_s_split"),
    b!("rsplit", [KS], [], KL, "viv_s_rsplit_ws"),
    b!("rsplit", [KS, KS, KN], [D::N(-1.0)], KL, "viv_s_rsplit"),
    b!("splitlines", [KS], [], KL, "viv_s_splitlines"),
    b!("join", [KS, KL], [], KS, "viv_s_join"),
    b!("find", [KS, KS, KN, KN], [D::N(0.0), NOT_GIVEN], KN, "viv_s_find"),
    b!("rfind", [KS, KS, KN, KN], [D::N(0.0), NOT_GIVEN], KN, "viv_s_rfind"),
    b!("index", [KS, KS, KN, KN], [D::N(0.0), NOT_GIVEN], KN, "viv_s_index"),
    b!("rindex", [KS, KS, KN, KN], [D::N(0.0), NOT_GIVEN], KN, "viv_s_rindex"),
    b!("count", [KS, KS, KN, KN], [D::N(0.0), NOT_GIVEN], KN, "viv_s_count"),
    b!("index", [KL, KS], [], KN, "viv_ss_index"),
    b!("count", [KL, KS], [], KN, "viv_ss_count"),
    b!("replace", [KS, KS, KS, KN], [D::N(-1.0)], KS, "viv_s_replace"),
    b!("startswith", [KS, KS], [], KB, "viv_s_startswith"),
    b!("endswith", [KS, KS], [], KB, "viv_s_endswith"),
    b!("contains", [KS, KS], [], KB, "viv_s_contains"),
    b!("contains", [KL, KS], [], KB, "viv_ss_contains"),
    b!("ljust", [KS, KN, KS], [D::S(" ")], KS, "viv_s_ljust"),
    b!("rjust", [KS, KN, KS], [D::S(" ")], KS, "viv_s_rjust"),
    b!("center", [KS, KN, KS], [D::S(" ")], KS, "viv_s_center"),
    b!("zfill", [KS, KN], [], KS, "viv_s_zfill"),
    b!("chr", [KN], [], KS, "viv_s_chr"),
    b!("ord", [KS], [], KN, "viv_s_ord"),
    b!("float", [KS], [], KN, "viv_s_float"),
    b!("float", [KN], [], KN, "viv_num_float"),
    b!("int", [KS, KN], [D::N(10.0)], KN, "viv_s_int"),
    b!("int", [KN], [], KN, "viv_num_int"),
    b!("repr", [KS], [], KS, "viv_s_repr"),
    b!("repr", [KL], [], KS, "viv_ss_repr"),
    b!("sorted", [KL], [], KL, "viv_ss_sorted"),
    b!("sorted", [KS], [], KL, "viv_s_sorted"),
    b!("unique", [KS], [], KL, "viv_s_unique"),
    b!("unique", [KL], [], KL, "viv_ss_unique"),
    b!("read_text", [KS], [], KS, "viv_s_read_text"),
    b!("read_lines", [KS], [], KL, "viv_s_read_lines"),
    b!("write_text", [KS, KS], [], KN, "viv_s_write_text"),
    // a file's bytes, each an Int: all of them, or
    // `count` from `offset` (count left out: to the end)
    b!("read_bytes", [KS, KN, KN], [D::N(0.0), NOT_GIVEN], KI, "viv_s_read_bytes"),
    // SQLite (crate::db): a database is a number,
    // its handle; parameters are bound, never pasted into the SQL
    b!("db_open", [KS], [], KN, "viv_s_db_open"),
    b!("db_close", [KN], [], KN, "viv_s_db_close"),
    b!("db_exec", [KN, KS], [], KN, "viv_s_db_exec0"),
    b!("db_exec", [KN, KS, KL], [], KN, "viv_s_db_exec"),
    b!("db_exec", [KN, KS, KA], [], KN, "viv_s_db_exec_n"),
    b!("db_col", [KN, KS], [], KA, "viv_s_db_col0"),
    b!("db_col", [KN, KS, KL], [], KA, "viv_s_db_col"),
    b!("db_col", [KN, KS, KA], [], KA, "viv_s_db_col_n"),
    b!("db_col_int", [KN, KS], [], KI, "viv_s_db_coli0"),
    b!("db_col_int", [KN, KS, KL], [], KI, "viv_s_db_coli"),
    b!("db_col_int", [KN, KS, KA], [], KI, "viv_s_db_coli_n"),
    b!("db_col_str", [KN, KS], [], KL, "viv_s_db_cols0"),
    b!("db_col_str", [KN, KS, KL], [], KL, "viv_s_db_cols"),
    b!("db_col_str", [KN, KS, KA], [], KL, "viv_s_db_cols_n"),
    b!("db_stage", [KN, KS, KA], [], KN, "viv_s_db_stage_f"),
    b!("db_stage", [KN, KS, KI], [], KN, "viv_s_db_stage_i"),
    b!("db_stage", [KN, KS, KL], [], KN, "viv_s_db_stage_s"),
    b!("db_insert", [KN, KS], [], KN, "viv_s_db_insert"),
];

/// Whether `name` is a string builtin (in the registry).
pub fn is_builtin(name: &str) -> bool {
    BUILTINS.iter().any(|b| b.name == name)
}

/// The entry a call of `name` with arguments of these kinds is, or None. A
/// `None` kind is an argument whose type is not known yet (it matches
/// anything). An empty `[]` may stand for an empty list.
pub fn resolve(name: &str, kinds: &[Option<K>]) -> Option<&'static Builtin> {
    BUILTINS.iter().find(|b| {
        b.name == name
            && kinds.len() <= b.args.len()
            && kinds.len() >= b.args.len() - b.opt.len()
            && kinds.iter().zip(b.args).all(|(k, want)| k.map_or(true, |k| k == *want))
    })
}

/// How a call of `name` is written, for the message when no entry fits.
pub fn usage(name: &str) -> String {
    let forms: Vec<String> = BUILTINS
        .iter()
        .filter(|b| b.name == name)
        .map(|b| {
            let req = b.args.len() - b.opt.len();
            let word = |k: &K| match k {
                KN => "number",
                KB => "boolean",
                KS => "string",
                KL => "list of strings",
                KA => "array of numbers",
                KI => "array of Ints",
            };
            let mut s = format!("{name}(");
            for (i, k) in b.args.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                if i >= req {
                    s.push('[');
                }
                s.push_str(word(k));
                if i >= req {
                    s.push(']');
                }
            }
            s.push(')');
            s
        })
        .collect();
    format!("`{name}` takes {}", forms.join(" or "))
}

// ---- the interpreter's arms ---------------------------------------------------------

fn kind_of(v: &Val) -> Option<K> {
    match v {
        Val::Num(_) => Some(KN),
        Val::Bool(_) => Some(KB),
        Val::Str(_) => Some(KS),
        Val::Strs(_) => Some(KL),
        _ => None,
    }
}

/// The kinds only the `db_*` builtins take: an array of numbers or of Ints.
fn db_kind(v: &Val) -> Option<K> {
    match v {
        Val::Arr(a) if a.borrow().shape.len() == 1 => Some(KA),
        Val::IArr(_) => Some(KI),
        _ => None,
    }
}

/// An empty real vector (`[]`) where a list of strings is wanted.
fn empty_arr(v: &Val) -> bool {
    matches!(v, Val::Arr(a) if a.borrow().data.is_empty() && a.borrow().shape.len() == 1)
}

fn s_of(v: &Val) -> &Rc<Str> {
    match v {
        Val::Str(s) => s,
        _ => unreachable!("resolved as a string"),
    }
}

fn items_of(v: &Val) -> Vec<Rc<Str>> {
    match v {
        Val::Strs(l) => l.borrow().clone(),
        _ => Vec::new(),
    }
}

fn to_strs(parts: Vec<&str>) -> Val {
    strs(parts.into_iter().map(|p| Str::new(p.to_string())).collect())
}

/// A registry builtin, run on evaluated arguments; `Ok(None)` when `callee`
/// is not a string builtin, or its arguments are not strings where it takes
/// strings (a numeric builtin of the same name may take them).
pub fn registry_call(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    if !is_builtin(callee) {
        return Ok(None);
    }
    // an Int where a number goes: the number it is, exactly (as the lowering
    // converts it, ints::promote_args)
    let conv: Vec<Val>;
    let a = if a.iter().any(|v| matches!(v, Val::Int(_))) {
        conv = a.iter().map(|v| match v {
            Val::Int(i) => super::ints::to_num(*i).map(Val::Num),
            v => Ok(v.clone()),
        }).collect::<Result<_, _>>()?;
        &conv[..]
    } else {
        a
    };
    let db = callee.starts_with("db_");
    let mut kinds: Vec<Option<K>> = a.iter().map(|v| kind_of(v).or_else(|| if db { db_kind(v) } else { None })).collect();
    // `[]` for an empty list of strings
    for (k, v) in kinds.iter_mut().zip(a) {
        if !db && k.is_none() && empty_arr(v) {
            *k = Some(KL);
        }
    }
    if kinds.iter().any(|k| k.is_none()) {
        return Err(usage(callee));
    }
    let Some(b) = resolve(callee, &kinds) else { return Err(usage(callee)) };
    // the defaults of the arguments left out
    let mut args: Vec<Val> = a.to_vec();
    for d in &b.opt[b.opt.len() - (b.args.len() - a.len())..] {
        args.push(match d {
            D::N(x) => Val::Num(*x),
            D::S(s) => sval(s.to_string()),
        });
    }
    run(b.c, &args).map(Some)
}

fn num(v: &Val) -> f64 {
    match v {
        Val::Num(x) => *x,
        _ => unreachable!("resolved as a number"),
    }
}

/// Run the builtin whose C name is `c` on arguments of the kinds its entry
/// says (defaults filled in).
pub fn run(c: &str, a: &[Val]) -> Result<Val, String> {
    let s = |i: usize| s_of(&a[i]);
    Ok(match c {
        "viv_s_upper" => sval(upper(&s(0).text)),
        "viv_s_lower" => sval(lower(&s(0).text)),
        _ if c.starts_with("viv_s_is") => Val::Bool(pred(&c["viv_s_".len()..], &s(0).text)),
        "viv_s_strip" | "viv_s_lstrip" | "viv_s_rstrip" | "viv_s_strip_chars" | "viv_s_lstrip_chars" | "viv_s_rstrip_chars" => {
            let which = if c.contains("_lstrip") { 1 } else if c.contains("_rstrip") { 2 } else { 0 };
            let chars = if c.ends_with("_chars") { Some(s(1).text.as_str()) } else { None };
            sval(strip(&s(0).text, chars, which).to_string())
        }
        "viv_s_split_ws" => to_strs(split(&s(0).text, None, -1.0, false)?),
        "viv_s_rsplit_ws" => to_strs(split(&s(0).text, None, -1.0, true)?),
        "viv_s_split" => to_strs(split(&s(0).text, Some(&s(1).text), num(&a[2]), false)?),
        "viv_s_rsplit" => to_strs(split(&s(0).text, Some(&s(1).text), num(&a[2]), true)?),
        "viv_s_splitlines" => to_strs(splitlines(&s(0).text)),
        "viv_s_join" => {
            let items = items_of(&a[1]);
            let parts: Vec<&str> = items.iter().map(|x| x.text.as_str()).collect();
            sval(join(&s(0).text, &parts)?)
        }
        "viv_s_find" => Val::Num(find(&s(0).view(), &s(1).view(), num(&a[2]), num(&a[3]), false)),
        "viv_s_rfind" => Val::Num(find(&s(0).view(), &s(1).view(), num(&a[2]), num(&a[3]), true)),
        "viv_s_index" => Val::Num(index(&s(0).view(), &s(1).view(), num(&a[2]), num(&a[3]), false)?),
        "viv_s_rindex" => Val::Num(index(&s(0).view(), &s(1).view(), num(&a[2]), num(&a[3]), true)?),
        "viv_s_count" => Val::Num(count(&s(0).view(), &s(1).view(), num(&a[2]), num(&a[3]))),
        "viv_ss_index" => {
            let items = items_of(&a[0]);
            match items.iter().position(|x| x.text == s(1).text) {
                Some(k) => Val::Num(k as f64),
                None => return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("{} is not in list", repr(&s(1).text)))),
            }
        }
        "viv_ss_count" => Val::Num(items_of(&a[0]).iter().filter(|x| x.text == s(1).text).count() as f64),
        "viv_s_replace" => sval(replace(&s(0).view(), &s(1).text, &s(2).text, num(&a[3]))?),
        "viv_s_startswith" => Val::Bool(s(0).text.starts_with(s(1).text.as_str())),
        "viv_s_endswith" => Val::Bool(s(0).text.ends_with(s(1).text.as_str())),
        "viv_s_contains" => Val::Bool(s(0).text.contains(s(1).text.as_str())),
        "viv_ss_contains" => Val::Bool(items_of(&a[0]).iter().any(|x| x.text == s(1).text)),
        "viv_s_ljust" => sval(pad(&s(0).view(), num(&a[1]), &s(2).view(), 0)?),
        "viv_s_rjust" => sval(pad(&s(0).view(), num(&a[1]), &s(2).view(), 1)?),
        "viv_s_center" => sval(pad(&s(0).view(), num(&a[1]), &s(2).view(), 2)?),
        "viv_s_zfill" => sval(zfill(&s(0).view(), num(&a[1]))?),
        "viv_s_chr" => sval(chr(num(&a[0]))?),
        "viv_s_ord" => Val::Num(ord(&s(0).view())?),
        "viv_s_float" => Val::Num(parse_float(&s(0).text)?),
        "viv_num_float" => Val::Num(num(&a[0])),
        "viv_s_int" => Val::Num(parse_int(&s(0).text, num(&a[1]))?),
        "viv_num_int" => Val::Num(int_of_num(num(&a[0]))?),
        "viv_s_repr" => sval(repr(&s(0).text)),
        "viv_ss_repr" => {
            let items = items_of(&a[0]);
            sval(list_repr(&items.iter().map(|x| x.text.as_str()).collect::<Vec<_>>()))
        }
        "viv_ss_sorted" => {
            let mut items = items_of(&a[0]);
            items.sort_by(|x, y| cmp(&x.text, &y.text));
            strs(items)
        }
        // Python's sorted(s) and sorted(set(s)) of a string: its characters
        // in code point order, each once for `unique`
        "viv_s_sorted" | "viv_s_unique" => {
            let mut cs: Vec<char> = s(0).text.chars().collect();
            cs.sort_unstable();
            if c == "viv_s_unique" {
                cs.dedup();
            }
            strs(cs.into_iter().map(|ch| Str::new(ch.to_string())).collect())
        }
        // sorted(set(words)): the distinct strings, in code point order
        "viv_ss_unique" => {
            let mut items = items_of(&a[0]);
            items.sort_by(|x, y| cmp(&x.text, &y.text));
            items.dedup_by(|x, y| x.text == y.text);
            strs(items)
        }
        "viv_s_read_text" => sval(read_text(&s(0).text)?),
        "viv_s_read_lines" => {
            let t = read_text(&s(0).text)?;
            to_strs(splitlines(&t))
        }
        "viv_s_write_text" => Val::Num(write_text(&s(0).text, &s(1).text)?),
        "viv_s_read_bytes" => Val::IArr(Rc::new(RefCell::new(super::bytes::read_bytes(&s(0).text, num(&a[1]), num(&a[2]))?))),
        c if c.starts_with("viv_s_db_") => db_run(c, a)?,
        other => return Err(format!("internal: no string builtin `{other}`")),
    })
}

// ---- SQLite (crate::db) -------------------------------------------------------------

/// A `db_*` builtin on the interpreter's values: crate::db does the work on
/// plain Rust values, the same rules as `db.c` in the C engine.
fn db_run(c: &str, a: &[Val]) -> Result<Val, String> {
    use crate::db::{self, Col, Params};
    let h = || num(&a[0]);
    let params = |i: usize| -> Params {
        match a.get(i) {
            None => Params::None,
            Some(Val::Strs(l)) => Params::Text(l.borrow().iter().map(|x| x.text.clone()).collect()),
            Some(Val::Arr(x)) => Params::Real(x.borrow().data.clone()),
            Some(_) => Params::None,
        }
    };
    let col = |v: &Val| -> Col {
        match v {
            Val::Arr(x) => Col::Real(x.borrow().data.clone()),
            Val::IArr(x) => Col::Int(x.borrow().clone()),
            Val::Strs(l) => Col::Text(l.borrow().iter().map(|x| x.text.clone()).collect()),
            _ => Col::Real(Vec::new()),
        }
    };
    let base = c.trim_end_matches("_n").trim_end_matches('0');
    Ok(match base {
        "viv_s_db_open" => Val::Num(db::open(&s_of(&a[0]).text)?),
        "viv_s_db_close" => Val::Num(db::close(h())?),
        "viv_s_db_exec" => Val::Num(db::exec(h(), &s_of(&a[1]).text, &params(2))?),
        "viv_s_db_col" => {
            let v = db::col_real(h(), &s_of(&a[1]).text, &params(2))?;
            super::new_arr(vec![v.len()], v)
        }
        "viv_s_db_coli" => Val::IArr(Rc::new(RefCell::new(db::col_int(h(), &s_of(&a[1]).text, &params(2))?))),
        "viv_s_db_cols" => strs(db::col_text(h(), &s_of(&a[1]).text, &params(2))?.into_iter().map(Str::new).collect()),
        "viv_s_db_stage_f" | "viv_s_db_stage_i" | "viv_s_db_stage_s" => Val::Num(db::stage(h(), &s_of(&a[1]).text, col(&a[2]))?),
        "viv_s_db_insert" => Val::Num(db::insert(h(), &s_of(&a[1]).text)?),
        other => return Err(format!("internal: no db builtin `{other}`")),
    })
}

// ---- files --------------------------------------------------------------------------

/// Python's universal newlines on reading: `\r\n` and a lone `\r` are `\n`.
pub fn universal_newlines(t: &str) -> String {
    if !t.contains('\r') {
        return t.to_string();
    }
    t.replace("\r\n", "\n").replace('\r', "\n")
}

/// The message for bytes that are not UTF-8, as Python's decoder words it.
pub fn not_utf8(path: &str, e: &std::str::Utf8Error) -> String {
    crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("{}: the file is not UTF-8 text (an invalid byte at offset {})", repr(path), e.valid_up_to()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_text(path: &str) -> Result<String, String> {
    let read = std::fs::read(path);
    crate::trace::note(std::path::Path::new(path), read.is_ok());
    let bytes = read.map_err(|e| file_error(path, &e))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| not_utf8(path, &e))?;
    Ok(universal_newlines(text))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_text(path: &str, text: &str) -> Result<f64, String> {
    std::fs::write(path, text).map_err(|e| file_error(path, &e))?;
    Ok(count_chars(text.as_bytes()) as f64)
}

/// Python's words for the usual file errors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn file_error(path: &str, e: &std::io::Error) -> String {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => format!("No such file or directory: {}", repr(path)),
        PermissionDenied => format!("Permission denied: {}", repr(path)),
        _ => format!("{}: {}", repr(path), e),
    }
}

// In the browser the files are a table the host fills.
#[cfg(target_arch = "wasm32")]
thread_local! {
    pub(crate) static FILES: RefCell<std::collections::HashMap<String, Vec<u8>>> = RefCell::new(std::collections::HashMap::new());
}

#[cfg(target_arch = "wasm32")]
pub fn read_text(path: &str) -> Result<String, String> {
    let bytes = FILES.with(|f| f.borrow().get(path).cloned());
    let Some(bytes) = bytes else {
        return Err(format!("No such file or directory: {} (in the browser a program reads the files attached to it)", repr(path)));
    };
    let text = std::str::from_utf8(&bytes).map_err(|e| not_utf8(path, &e))?;
    Ok(universal_newlines(text))
}

#[cfg(target_arch = "wasm32")]
pub fn write_text(path: &str, text: &str) -> Result<f64, String> {
    FILES.with(|f| f.borrow_mut().insert(path.to_string(), text.as_bytes().to_vec()));
    Ok(count_chars(text.as_bytes()) as f64)
}

// ---- values in the interpreter: operators, conversions, lists -----------------------

/// `s[i] = c`: a string is not changed in place.
pub const STR_IMMUTABLE: &str = "'str' object does not support item assignment";
/// `words[i] = w`: nor is a list of strings.
pub const STRS_IMMUTABLE: &str = "a list of strings cannot be changed in place; build a new one (`a = a[:i] + [w] + a[i + 1:]`)";

fn list_of(v: &Val) -> Option<Vec<Rc<Str>>> {
    match v {
        Val::Strs(l) => Some(l.borrow().clone()),
        v if empty_arr(v) => Some(Vec::new()),
        _ => None,
    }
}

/// An operator with a string, or a list of strings, on either side.
pub fn binary(op: crate::ast::BinOp, l: &Val, r: &Val) -> Result<Val, String> {
    use crate::ast::BinOp::*;
    match (op, l, r) {
        (Add, Val::Str(a), Val::Str(b)) => Ok(sval(concat(&a.text, &b.text)?)),
        (Mul, Val::Str(a), Val::Num(n)) | (Mul, Val::Num(n), Val::Str(a)) => Ok(sval(repeat(&a.text, *n)?)),
        (EqEq | Ne | Lt | Le | Gt | Ge, Val::Str(a), Val::Str(b)) => {
            let o = cmp(&a.text, &b.text);
            Ok(Val::Bool(match op {
                EqEq => o.is_eq(),
                Ne => o.is_ne(),
                Lt => o.is_lt(),
                Le => o.is_le(),
                Gt => o.is_gt(),
                _ => o.is_ge(),
            }))
        }
        (Add, _, _) if list_of(l).is_some() && list_of(r).is_some() && (matches!(l, Val::Strs(_)) || matches!(r, Val::Strs(_))) => {
            let mut v = list_of(l).expect("a list");
            v.extend(list_of(r).expect("a list"));
            Ok(strs(v))
        }
        (EqEq | Ne, Val::Strs(a), Val::Strs(b)) => {
            let (a, b) = (a.borrow(), b.borrow());
            let same = a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.text == y.text);
            Ok(Val::Bool(same == (op == EqEq)))
        }
        _ => Err(format!("`{}` does not apply to a {} and a {}", op_word(op), super::val_kind(l), super::val_kind(r))),
    }
}

fn op_word(op: crate::ast::BinOp) -> &'static str {
    op.c_op()
}

/// `str(x)`: a number as Python's repr(float), a boolean as `true` or
/// `false`, a complex number as `print` writes it, a real array as `fmt`
/// writes it, a string as itself, a list as Python prints one.
pub fn to_str(v: &Val) -> Result<String, String> {
    Ok(match v {
        Val::Num(x) => crate::printfmt::str_num(*x),
        Val::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Val::Cplx(re, im) => super::format_cplx(*re, *im),
        Val::Arr(a) => {
            let a = a.borrow();
            crate::printfmt::array(&a.shape, &a.data, crate::printfmt::Fmt::STR)
        }
        Val::Str(s) => s.text.clone(),
        Val::Strs(l) => {
            let l = l.borrow();
            list_repr(&l.iter().map(|x| x.text.as_str()).collect::<Vec<_>>())
        }
        other => return Err(format!("`str` cannot write a {}", super::val_kind(other))),
    })
}

/// `repr(x)`: a string in quotes, a list as Python prints it, anything else
/// as `str`.
pub fn repr_value(v: &Val) -> Result<String, String> {
    match v {
        Val::Str(s) => Ok(repr(&s.text)),
        Val::Num(x) => Ok(crate::printfmt::repr(*x)),
        Val::Arr(a) => {
            let a = a.borrow();
            Ok(crate::printfmt::array(&a.shape, &a.data, crate::printfmt::Fmt::REPR))
        }
        _ => to_str(v),
    }
}

/// The values of a format's `*` width and precision: numbers or Ints.
pub fn star_values(vals: &[Val]) -> Result<Vec<f64>, String> {
    vals.iter()
        .map(|v| match v {
            Val::Num(x) => Ok(*x),
            Val::Int(i) => Ok(*i as f64),
            v => Err(format!("a format's width or precision is a number, not a {}", super::val_kind(v))),
        })
        .collect()
}

/// `fmt(x, ...)` as a value.
pub fn fmt_value(v: &Val, rest: &[Val]) -> Result<String, String> {
    use crate::printfmt::{self as pf, Fmt};
    let sspec = |spec: &Str, stars: &[Val]| -> Result<StrSpec, String> {
        let sp = StrSpec::parse(&spec.text)?;
        if sp.star_count() != stars.len() {
            return Err("`fmt` takes a value and at most one format".into());
        }
        sp.with_stars(&star_values(stars)?)
    };
    match (v, rest) {
        (Val::Str(_) | Val::Strs(_), []) => repr_value(v),
        (Val::Str(s), [Val::Str(spec), stars @ ..]) => Ok(format_str(&s.view(), sspec(spec, stars)?)),
        (Val::Strs(_), [Val::Str(spec), stars @ ..]) => {
            let text = repr_value(v)?;
            let t = Str::new(text);
            Ok(format_str(&t.view(), sspec(spec, stars)?))
        }
        (Val::Str(_) | Val::Strs(_), _) => Err("`fmt` of a string takes a format string, such as \">10\" or \".3\"".into()),
        (_, _) => {
            let f = match rest {
                [] => Fmt::REPR,
                [Val::Num(d)] => {
                    if d.fract() == 0.0 && (1.0..=17.0).contains(d) {
                        Fmt::digits(*d as u32)
                    } else {
                        return Err("`fmt(x, digits)` takes 1 to 17 significant digits".into());
                    }
                }
                [Val::Str(spec), stars @ ..] => {
                    let f = Fmt::parse_spec(&spec.text)?;
                    if f.stars() != stars.len() {
                        return Err("`fmt` takes a value and at most one format".into());
                    }
                    f.with_stars(&star_values(stars)?)?
                }
                _ => return Err("`fmt` takes a value and at most one format".into()),
            };
            match v {
                Val::Num(x) => Ok(pf::number(*x, f)),
                Val::Arr(a) => {
                    let a = a.borrow();
                    Ok(pf::array(&a.shape, &a.data, f))
                }
                _ => Err("`fmt` takes a real number or a real array".into()),
            }
        }
    }
}

/// `[a, b, c]` of strings.
pub fn list_literal(vals: Vec<Val>) -> Result<Val, String> {
    let mut out = Vec::with_capacity(vals.len());
    for v in vals {
        match v {
            Val::Str(s) => out.push(s),
            other => return Err(format!("a list of strings cannot hold a {}", super::val_kind(&other))),
        }
    }
    Ok(strs(out))
}

/// Whether `a = append(a, x)` / `extend` is a list of strings' (the array
/// is one, or `[]` and `x` is a string or a list of strings).
pub fn grows(cur: &Val, x: &Val) -> bool {
    matches!(cur, Val::Strs(_)) || (empty_arr(cur) && matches!(x, Val::Str(_) | Val::Strs(_)))
}

/// `append(a, s)` and `extend(a, b)` of a list of strings: in place when the
/// list is held by nothing else (its reference count is 1), a copy otherwise.
pub fn grow(callee: &str, cur: &Val, x: &Val) -> Result<Val, String> {
    let add: Vec<Rc<Str>> = match (callee, x) {
        ("append", Val::Str(s)) => vec![s.clone()],
        ("extend", v) if list_of(v).is_some() => list_of(v).expect("a list"),
        ("append", other) => return Err(format!("`append` adds a string to a list of strings, not a {}", super::val_kind(other))),
        (_, other) => return Err(format!("`extend` joins two lists of strings, not a list and a {}", super::val_kind(other))),
    };
    match cur {
        Val::Strs(l) if Rc::strong_count(l) == 1 => {
            l.borrow_mut().extend(add);
            Ok(cur.clone())
        }
        _ => {
            let mut v = list_of(cur).expect("a list of strings or []");
            v.extend(add);
            Ok(strs(v))
        }
    }
}

/// The builtins that take strings along with other values, and the
/// registry's: `Ok(None)` when `callee` is none of them, or its arguments are
/// not strings where it would take them (so an array's `len` goes on to the
/// array builtins).
pub fn call(callee: &str, a: &[Val]) -> Result<Option<Val>, String> {
    let has = |v: &Val| matches!(v, Val::Str(_) | Val::Strs(_));
    Ok(Some(match (callee, a) {
        ("len", [Val::Str(s)]) => Val::Num(s.nch as f64),
        ("len", [Val::Strs(l)]) => Val::Num(l.borrow().len() as f64),
        (FSTR, parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Val::Str(s) => out.push_str(&s.text),
                    other => out.push_str(&to_str(other)?),
                }
                if out.len() > MAX_BYTES {
                    return Err(too_long(out.len() as u128));
                }
            }
            sval(out)
        }
        (IN, [x, Val::Str(s)]) => match x {
            Val::Str(sub) => Val::Bool(s.text.contains(sub.text.as_str())),
            other => return Err(format!("'in <string>' requires string as left operand, not {}", super::val_kind(other))),
        },
        (IN, [x, Val::Strs(l)]) => match x {
            Val::Str(w) => Val::Bool(l.borrow().iter().any(|e| e.text == w.text)),
            other => return Err(format!("a list of strings holds strings, not a {}", super::val_kind(other))),
        },
        // a number in an array: whether some element equals it, as NumPy's
        // `x in a` (a NaN equals nothing); a key in a map: `has(m, k)`
        (IN, [x, Val::Arr(a)]) => match x {
            Val::Num(v) => Val::Bool(a.borrow().data.iter().any(|e| e == v)),
            // `[]` where the lowering made an empty list of strings (`used: [Str] = []`)
            Val::Str(_) if a.borrow().data.is_empty() => Val::Bool(false),
            other => return Err(format!("`in` looks for a number in an array, not a {}", super::val_kind(other))),
        },
        (IN, [k, m @ Val::Map(_)]) => return super::maps::builtin("has", &[m.clone(), k.clone()]),
        (IN, [_, other]) => return Err(format!("`in` takes a string, a list of strings, a real array or a map on its right, not a {}", super::val_kind(other))),
        ("str" | STR_OF, [v]) => sval(to_str(v)?),
        ("repr" | REPR_OF, [v]) => sval(repr_value(v)?),
        ("fmt" | FMT_OF, [v, rest @ ..]) => sval(fmt_value(v, rest)?),
        ("stop", [Val::Str(s)]) => return Err(crate::errkind::kinded(crate::errkind::ErrKind::Stop, s.text.clone())),
        (c, [base, Val::Arr(spec)]) if c == super::slices::GET && has(base) => {
            let spec = spec.borrow().data.clone();
            if spec.len() != 4 {
                return Err(format!("{} takes one index or range; this has {}", if matches!(base, Val::Str(_)) { "a string" } else { "a list of strings" }, spec.len() / 4));
            }
            let spec = [spec[0], spec[1], spec[2], spec[3]];
            match base {
                Val::Str(s) if spec[0] == 0.0 => {
                    let (x, y) = char_at(&s.view(), spec[1])?;
                    sval(s.text[x..y].to_string())
                }
                Val::Str(s) => sval(slice(&s.view(), spec)?),
                Val::Strs(l) if spec[0] == 0.0 => {
                    let l = l.borrow();
                    crate::interp::index_int(spec[1])?;
                    let k = py_index(l.len(), spec[1]).ok_or_else(|| crate::errkind::kinded(crate::errkind::ErrKind::Index, "list index out of range"))?;
                    Val::Str(l[k].clone())
                }
                Val::Strs(l) => {
                    let l = l.borrow();
                    strs(slice_indices(l.len(), spec)?.into_iter().map(|k| l[k].clone()).collect())
                }
                _ => unreachable!("a string or a list"),
            }
        }
        (c, [Val::Str(_), ..]) if c == super::slices::SET => return Err(STR_IMMUTABLE.into()),
        (c, [Val::Strs(_), ..]) if c == super::slices::SET => return Err(STRS_IMMUTABLE.into()),
        ("append" | "extend", [cur, x]) if grows(cur, x) => grow(callee, cur, x)?,
        _ => return registry_call(callee, a),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    mod cases {
        include!("strings_cases.rs");
    }
    use cases::{A, ERRORS};

    fn sv_of(s: &str) -> Rc<Str> {
        Str::new(s.to_string())
    }

    /// Every call tools/strings/cases.py found to stop in CPython stops here
    /// with CPython's message.
    #[test]
    fn errors_have_cpythons_messages() {
        let mut bad = Vec::new();
        for (kind, args, want) in ERRORS {
            let s = |k: usize| match &args[k] {
                A::S(t) => t.to_string(),
                A::N(_) => panic!("{kind}: argument {k} is a number"),
            };
            let n = |k: usize| match &args[k] {
                A::N(x) => *x,
                A::S(_) => panic!("{kind}: argument {k} is a string"),
            };
            let got: Result<String, String> = match *kind {
                "float" => parse_float(&s(0)).map(|x| x.to_string()),
                "int" => parse_int(&s(0), n(1)).map(|x| x.to_string()),
                "split" => split(&s(0), Some(&s(1)), -1.0, false).map(|v| format!("{v:?}")),
                "ljust" => pad(&sv_of(&s(0)).view(), n(1), &sv_of(&s(2)).view(), 0),
                "chr" => chr(n(0)),
                "ord" => ord(&sv_of(&s(0)).view()).map(|x| x.to_string()),
                "at" => char_at(&sv_of(&s(0)).view(), n(1)).map(|r| format!("{r:?}")),
                "list_index" => py_index(5, n(0)).map(|k| k.to_string()).ok_or_else(|| "list index out of range".to_string()),
                "index" | "rindex" => index(&sv_of(&s(0)).view(), &sv_of(&s(1)).view(), 0.0, f64::NAN, *kind == "rindex").map(|x| x.to_string()),
                other => panic!("no case kind {other}"),
            };
            if got.as_ref().map_err(|e| e.as_str()) != Err(*want) {
                bad.push(format!("{kind}{:?}: want Err({want:?}), got {got:?}", args.iter().map(|a| match a { A::S(t) => format!("{t:?}"), A::N(x) => x.to_string() }).collect::<Vec<_>>()));
            }
        }
        assert!(bad.is_empty(), "{} of {}:\n{}", bad.len(), ERRORS.len(), bad.join("\n"));
    }

    #[test]
    fn digits_round_to_nearest_even() {
        // 2^53 + 1 is a tie between 2^53 and 2^53 + 2: the even one
        assert_eq!(digits_to_f64(&"9007199254740993".bytes().map(|b| b - b'0').collect::<Vec<_>>(), 10), Some(9007199254740992.0));
        assert_eq!(digits_to_f64(&"9007199254740995".bytes().map(|b| b - b'0').collect::<Vec<_>>(), 10), Some(9007199254740996.0));
        // past 64 bits: the sticky bit decides a near-tie
        let big = "1".to_string() + &"0".repeat(30);
        assert_eq!(digits_to_f64(&big.bytes().map(|b| b - b'0').collect::<Vec<_>>(), 10), Some(1e30));
        assert_eq!(digits_to_f64(&[1; 400], 10), None);
    }

    /// The map API's hash is FNV-1a of the UTF-8 bytes (the C twin,
    /// `viv_str_hash`, is the same loop): the published test values.
    #[test]
    fn hash_is_fnv1a_of_the_bytes() {
        assert_eq!(hash(""), 0xcbf29ce484222325);
        assert_eq!(hash("a"), 0xaf63dc4c8601ec8c);
        assert_eq!(hash("é"), 0x0ac21707b7181e01);
        assert_eq!(cmp("é", "z"), std::cmp::Ordering::Greater);
    }

    #[test]
    fn byte_of_walks_from_the_nearest_known_point() {
        let s = sv_of("aé€😀b");
        let v = s.view();
        for (ci, b) in [(0, 0), (1, 1), (2, 3), (3, 6), (4, 10), (5, 11), (3, 6), (1, 1), (4, 10)] {
            assert_eq!(v.byte_of(ci), b, "code point {ci}");
        }
    }
}
