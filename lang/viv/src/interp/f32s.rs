// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The rules of `Float32` and of the packed arrays:
//! rounding to bf16 and f16, the range checks of the integer storage types,
//! element reads and writes on the bytes, `sum` in NumPy's float32 order,
//! `dot`, the float32 functions and the shortest text of a Float32. The
//! interpreter calls these, and so does the wasm runtime (`wasm_rt.rs`); the
//! C runtime's fragment `f32s` (lower/f32s.rs) is their twin, line for line.

use crate::ast::Store;

/// What an element read gives: a Float32 or an Int.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Elt {
    F(f32),
    I(i64),
}

/// `x` rounded to bfloat16, round to nearest even: its 16 bits. A NaN stays
/// a NaN (quiet, its sign kept); an overflow gives an infinity. This is
/// `__float2bfloat16_rn`.
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    let lsb = (u >> 16) & 1;
    (u.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// A bfloat16 as the Float32 it is, exactly.
pub fn bf16_value(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// `x` rounded to IEEE binary16, round to nearest even: its 16 bits.
/// Subnormal results are rounded on their own grid (2^-24); a magnitude that
/// rounds past 65504 gives an infinity; a NaN stays a quiet NaN.
pub fn f16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let a = u & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | 0x7e00 | ((a >> 13) & 0x3ff) as u16;
    }
    if a == 0x7f80_0000 {
        return sign | 0x7c00;
    }
    // at or above 65520 (the midpoint between 65504 and 2^16) the result is
    // an infinity
    if a >= 0x477f_f000 {
        return sign | 0x7c00;
    }
    let e = (a >> 23) as i32;
    if e >= 113 {
        // normal in binary16: drop 13 fraction bits, round to nearest even
        let m = a - (112 << 23);
        let lsb = (m >> 13) & 1;
        return sign | ((m + 0x0fff + lsb) >> 13) as u16;
    }
    // subnormal in binary16 (or zero): the value in units of 2^-24, rounded
    // to nearest even; below 2^-25 that is 0
    if e < 102 {
        return sign;
    }
    let mant = (a & 0x007f_ffff) | 0x0080_0000;
    let shift = (126 - e) as u32; // value = mant * 2^(e - 150); units of 2^-24
    let q = mant >> shift;
    let rem = mant & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let q = if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q };
    sign | q as u16
}

/// A binary16 as the Float32 it is, exactly.
pub fn f16_value(h: u16) -> f32 {
    crate::interp::bytes::f16_value(h as u64) as f32
}

/// `x` rounded to the precision of storage kind `k` (`bfloat16(x)`,
/// `float16(x)`), still a Float32.
pub fn round_to(k: Store, x: f32) -> f32 {
    match k {
        Store::BF16 => bf16_value(bf16_bits(x)),
        Store::F16 => f16_value(f16_bits(x)),
        _ => x,
    }
}

/// The range an integer storage kind holds.
pub fn range(k: Store) -> (i64, i64) {
    match k {
        Store::I8 => (i8::MIN as i64, i8::MAX as i64),
        Store::I16 => (i16::MIN as i64, i16::MAX as i64),
        Store::I32 => (i32::MIN as i64, i32::MAX as i64),
        Store::U8 => (0, u8::MAX as i64),
        _ => (i64::MIN, i64::MAX),
    }
}

/// The message of an Int that does not fit where it is stored.
pub fn no_fit(k: Store, v: i64) -> String {
    let (lo, hi) = range(k);
    crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("{v} does not fit in {} {} ({lo} to {hi})", if k == Store::U8 { "a" } else { "an" }, k.name()))
}

/// `v` checked against the range of `k`.
pub fn narrow(k: Store, v: i64) -> Result<i64, String> {
    let (lo, hi) = range(k);
    if v < lo || v > hi {
        return Err(no_fit(k, v));
    }
    Ok(v)
}

/// Element `i` of a packed array of kind `k` held in `b`.
pub fn get(k: Store, b: &[u8], i: usize) -> Elt {
    let w = k.width();
    let s = &b[i * w..i * w + w];
    match k {
        Store::F32 => Elt::F(f32::from_le_bytes([s[0], s[1], s[2], s[3]])),
        Store::BF16 => Elt::F(bf16_value(u16::from_le_bytes([s[0], s[1]]))),
        Store::F16 => Elt::F(f16_value(u16::from_le_bytes([s[0], s[1]]))),
        Store::I8 => Elt::I(s[0] as i8 as i64),
        Store::U8 => Elt::I(s[0] as i64),
        Store::I16 => Elt::I(i16::from_le_bytes([s[0], s[1]]) as i64),
        Store::I32 => Elt::I(i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as i64),
    }
}

/// Write `v` as element `i`: a float kind rounds to nearest even, an
/// integer kind checks the range.
pub fn set(k: Store, b: &mut [u8], i: usize, v: Elt) -> Result<(), String> {
    let w = k.width();
    let s = &mut b[i * w..i * w + w];
    match (k, v) {
        (Store::F32, Elt::F(x)) => s.copy_from_slice(&x.to_le_bytes()),
        (Store::BF16, Elt::F(x)) => s.copy_from_slice(&bf16_bits(x).to_le_bytes()),
        (Store::F16, Elt::F(x)) => s.copy_from_slice(&f16_bits(x).to_le_bytes()),
        (Store::I8 | Store::U8, Elt::I(n)) => s[0] = narrow(k, n)? as u8,
        (Store::I16, Elt::I(n)) => s.copy_from_slice(&(narrow(k, n)? as i16).to_le_bytes()),
        (Store::I32, Elt::I(n)) => s.copy_from_slice(&(narrow(k, n)? as i32).to_le_bytes()),
        (_, Elt::F(_)) => return Err(format!("an {} array holds Ints, and this is a Float32", k.name())),
        (_, Elt::I(_)) => return Err(format!("a {} array holds Float32s, and this is an Int", k.name())),
    }
    Ok(())
}

/// The element count of `b` bytes of kind `k`.
pub fn count(k: Store, b: &[u8]) -> usize {
    b.len() / k.width()
}

/// The elements of a float kind, widened to Float32.
pub fn floats(k: Store, b: &[u8]) -> Vec<f32> {
    (0..count(k, b)).map(|i| match get(k, b, i) { Elt::F(x) => x, Elt::I(n) => n as f32 }).collect()
}

/// NumPy's pairwise sum (numpy/core/src/umath/loops_utils.h.src), in
/// float32: below 8 left to right, to 128 eight running sums combined as a
/// tree, above that two halves, the first a multiple of 8 long.
pub fn pairwise(f: &[f32]) -> f32 {
    let n = f.len();
    if n < 8 {
        let mut s = -0.0f32;
        for x in f {
            s += *x;
        }
        return s;
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&f[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += f[i + j];
            }
            i += 8;
        }
        let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            s += f[i];
            i += 1;
        }
        return s;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise(&f[..n2]) + pairwise(&f[n2..])
}

/// `sum` of a float array: from +0.0, blocks of 8192, each by the pairwise
/// sum; `numpy.sum` of a float32 array gives the same bits.
pub fn sum(f: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for b in f.chunks(BLOCK) {
        s += pairwise(b);
    }
    s
}

/// NumPy's reduction buffer: a sum is taken this many elements at a time.
pub const BLOCK: usize = 8192;

/// `sum` of an integer array: an Int, checked.
pub fn isum(k: Store, b: &[u8]) -> Result<i64, String> {
    let mut s: i64 = 0;
    for i in 0..count(k, b) {
        if let Elt::I(n) = get(k, b, i) {
            s = crate::interp::ints::add(s, n)?;
        }
    }
    Ok(s)
}

/// `dot(a, b)`: `s = s + a[i]*b[i]` for i ascending from +0.0, each product
/// and sum rounded to float32 (no fused multiply-add).
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("dot: the arrays have {} and {} elements", a.len(), b.len())));
    }
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let p = a[i] * b[i];
        s += p;
    }
    Ok(s)
}

/// `min` and `max` of two Float32s: `b` when it is strictly less (greater),
/// else `a` (so a NaN in `b` gives `a`).
pub fn min(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}
pub fn max(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// The inputs where Viv's double function rounded once is not the correctly
/// rounded float32: all of them, found by tools/float32/exhaustive.c over every
/// float32 input and decided by tools/float32/check.py (mpmath, 200 bits).
/// exp and tanh have none. The C runtime's `viv_f_*` carry the same table.
pub const EXCEPTIONS: &[(&str, u32, u32)] = &[
    ("log", 0x3c413d3a, 0xc08e158f),
    ("log", 0x41178feb, 0x400fe5e7),
    ("log", 0x4c5d65a5, 0x418f034b),
    ("log", 0x65d890d3, 0x4254d1f9),
    ("log", 0x6f31a8ec, 0x42845a89),
    ("sin", 0x46199998, 0xbeb1fa5d),
    ("sin", 0xc6199998, 0x3eb1fa5d),
    ("cos", 0x5f18b878, 0x3f7f14bb),
    ("cos", 0x6115cb11, 0x3f78142f),
    ("cos", 0xdf18b878, 0x3f7f14bb),
    ("cos", 0xe115cb11, 0x3f78142f),
];

/// The float32 functions: Viv's double function, rounded once.
/// The sweep in tools/float32/ checks every
/// input against a correctly rounded reference.
pub fn func(name: &str, x: f32) -> Option<f32> {
    if let Some(e) = EXCEPTIONS.iter().find(|e| e.1 == x.to_bits() && e.0 == name) {
        return Some(f32::from_bits(e.2));
    }
    let d = x as f64;
    let y = match name {
        "exp" => crate::vmath::exp_cr(d),
        "log" => crate::vmath::log(d),
        "sin" => crate::vmath::sin(d),
        "cos" => crate::vmath::cos(d),
        "tanh" => crate::vmath::tanh(d),
        _ => return None,
    };
    Some(y as f32)
}

/// The functions `func` knows.
pub const FUNCS: &[&str] = &["exp", "log", "sin", "cos", "tanh"];

/// The shortest decimal digits that read back as `a` (finite, positive) in
/// float32, and the decimal exponent of the first; `printfmt::shortest`'s
/// algorithm with `f32` reading.
fn shortest(a: f32) -> (String, i32) {
    for p in 1..=9u32 {
        let s = format!("{:.*e}", p as usize - 1, a as f64);
        let (m, ex) = s.split_once('e').expect("{:e} has an exponent");
        let mut n: u64 = m.bytes().filter(|c| c.is_ascii_digit()).fold(0, |n, c| n * 10 + (c - b'0') as u64);
        let mut ex: i32 = ex.parse().expect("a decimal exponent");
        let r: f32 = s.parse().expect("a decimal");
        if r != a {
            let lo = 10u64.pow(p - 1);
            let hi = lo * 10;
            if (r as f64) < a as f64 {
                n += 1;
                if n == hi {
                    n = lo;
                    ex += 1;
                }
            } else {
                n -= 1;
                if n < lo {
                    n = hi - 1;
                    ex -= 1;
                }
            }
            let t = format!("{}e{}", n, ex - (p as i32 - 1));
            if t.parse::<f32>() != Ok(a) {
                continue;
            }
        }
        let d = n.to_string();
        let d = d.trim_end_matches('0');
        return (if d.is_empty() { "0".into() } else { d.to_string() }, ex);
    }
    unreachable!("9 significant digits always read back in float32")
}

/// `fmt(x)` and `str(x)` of a Float32: the shortest decimal that reads back
/// as the same float32, laid out as `repr` lays out a number (NumPy's
/// `repr(np.float32(x))`).
pub fn repr(x: f32) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let (d, ex) = shortest(x.abs());
    crate::printfmt::repr_layout(x < 0.0, &d, ex)
}

/// A packed array as print and show write it: `[0.1, 0.25, 3]`, each
/// element as a bare number (a float kind widened) or as its digits.
pub fn text(k: Store, b: &[u8]) -> String {
    let mut s = String::from("[");
    for i in 0..count(k, b) {
        if i > 0 {
            s.push_str(", ");
        }
        match get(k, b, i) {
            Elt::F(x) => s.push_str(&crate::printfmt::bare(x as f64)),
            Elt::I(n) => s.push_str(&n.to_string()),
        }
    }
    s.push(']');
    s
}

/// A new packed array of kind `k` holding `xs` (Float32s), each rounded on
/// the store.
pub fn pack_floats(k: Store, xs: impl Iterator<Item = f32>) -> Vec<u8> {
    let xs: Vec<f32> = xs.collect();
    let mut b = vec![0u8; xs.len() * k.width()];
    for (i, x) in xs.into_iter().enumerate() {
        let _ = set(k, &mut b, i, Elt::F(x));
    }
    b
}

/// A new packed array of integer kind `k` holding `xs`, each checked.
pub fn pack_ints(k: Store, xs: impl Iterator<Item = i64>) -> Result<Vec<u8>, String> {
    let xs: Vec<i64> = xs.collect();
    let mut b = vec![0u8; xs.len() * k.width()];
    for (i, x) in xs.into_iter().enumerate() {
        set(k, &mut b, i, Elt::I(x))?;
    }
    Ok(b)
}

/// `float32(a)`, `bfloat16(a)`, `int8(a)`, ... of a packed array of kind
/// `from`: the elements converted to kind `to`. Float to float rounds,
/// integer to integer checks; the lowering refuses a mix.
pub fn convert(from: Store, to: Store, b: &[u8]) -> Result<Vec<u8>, String> {
    if from.is_float() {
        Ok(pack_floats(to, floats(from, b).into_iter()))
    }
    else {
        pack_ints(to, (0..count(from, b)).map(|i| match get(from, b, i) { Elt::I(n) => n, Elt::F(x) => x as i64 }))
    }
}

/// An index, checked with the interpreter's words (`interp::index_int`,
/// `check_index`): a whole number, a negative one counting from the end.
pub fn index(x: f64, n: usize) -> Result<usize, String> {
    let i = crate::interp::index_int(x)?;
    let at = if i < 0 { i.saturating_add(n as i64) } else { i };
    if at < 0 || at as usize >= n {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Index, format!("index {i} out of bounds for dimension 0 (size {n})")));
    }
    Ok(at as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1)
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: even is up
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        assert_eq!(bf16_bits(f32::MAX), 0x7f80);
        assert!(bf16_value(bf16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(65519.0), 0x7bff);
        assert_eq!(f16_bits(65520.0), 0x7c00);
        // the smallest subnormal, 2^-24, and half of it (ties to even: 0)
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f16_bits(2.980_232_2e-8), 0x0000);
        assert_eq!(f16_bits(-0.0), 0x8000);
        assert_eq!(f16_value(0x3555), 0.333_251_95);
        // every binary16 survives the round trip
        for h in 0..=0xffffu16 {
            let x = f16_value(h);
            if x.is_nan() {
                continue;
            }
            assert_eq!(f16_bits(x), h, "{h:#06x}");
        }
    }

    #[test]
    fn repr_is_numpys() {
        assert_eq!(repr(0.1), "0.1");
        assert_eq!(repr(1.0), "1.0");
        assert_eq!(repr(1.0 / 3.0), "0.33333334");
        assert_eq!(repr(16777216.0), "16777216.0");
        assert_eq!(repr(1e-5), "1e-05");
        assert_eq!(repr(f32::MAX), "3.4028235e+38");
    }

    #[test]
    fn narrowing_checks() {
        assert_eq!(narrow(Store::I8, 127), Ok(127));
        assert!(narrow(Store::I8, 128).is_err());
        assert!(narrow(Store::U8, -1).is_err());
        assert_eq!(narrow(Store::I32, i32::MIN as i64), Ok(i32::MIN as i64));
    }
}
