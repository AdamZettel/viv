// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The rules of `Int`: checked arithmetic, floor
//! division and modulo, conversions to and from numbers, comparison with a
//! number, the bit operations and their text. The interpreter calls these,
//! and so does the wasm runtime (`wasm_rt.rs`), so the two say the same
//! words; the C runtime's fragment `ints` (lower/ints.rs) is their twin, line
//! for line.

use crate::ast::BinOp;

/// The largest magnitude a number holds every integer up to: 2^53.
pub const EXACT: i64 = 1 << 53;

/// The message of an Int operation whose result leaves 64 bits.
pub fn overflow(a: i64, op: &str, b: i64) -> String {
    crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("integer overflow: {a} {op} {b} does not fit in an Int"))
}

fn ovf1(what: &str, a: i64) -> String {
    crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("integer overflow: {what}({a}) does not fit in an Int"))
}

pub fn add(a: i64, b: i64) -> Result<i64, String> {
    a.checked_add(b).ok_or_else(|| overflow(a, "+", b))
}

pub fn sub(a: i64, b: i64) -> Result<i64, String> {
    a.checked_sub(b).ok_or_else(|| overflow(a, "-", b))
}

pub fn mul(a: i64, b: i64) -> Result<i64, String> {
    a.checked_mul(b).ok_or_else(|| overflow(a, "*", b))
}

/// `a // b`: the floor of the exact quotient (Python's rule).
pub fn floordiv(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        return Err(crate::interp::zero_divisor(BinOp::FloorDiv));
    }
    if a == i64::MIN && b == -1 {
        return Err(overflow(a, "//", b));
    }
    let q = a / b;
    Ok(if a % b != 0 && ((a < 0) != (b < 0)) { q - 1 } else { q })
}

/// The ceiling of the exact quotient `a / b`.
pub fn ceildiv(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::ZeroDiv, "division by zero in `a / b`"));
    }
    if a == i64::MIN && b == -1 {
        return Err(overflow(a, "/", b));
    }
    let q = a / b;
    Ok(if a % b != 0 && ((a < 0) == (b < 0)) { q + 1 } else { q })
}

/// `a % b`: `a - (a // b) * b`, which takes the sign of `b` (Python's rule).
pub fn floormod(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        return Err(crate::interp::zero_divisor(BinOp::Mod));
    }
    if b == -1 {
        return Ok(0);
    }
    let r = a % b;
    Ok(if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r })
}

/// `a ^ k` for `k >= 0`, by squaring; a square is taken only when a higher
/// bit of `k` needs it, so a result that fits is never refused because a
/// square past it would not.
pub fn pow(a: i64, k: i64) -> Result<i64, String> {
    if k < 0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "an Int to a negative power is not an Int; write float(n) ^ k"));
    }
    let (mut r, mut b, mut e) = (1i64, a, k);
    while e > 0 {
        if e & 1 == 1 {
            r = r.checked_mul(b).ok_or_else(|| overflow(a, "^", k))?;
        }
        e >>= 1;
        if e > 0 {
            b = b.checked_mul(b).ok_or_else(|| overflow(a, "^", k))?;
        }
    }
    Ok(r)
}

pub fn neg(a: i64) -> Result<i64, String> {
    a.checked_neg().ok_or_else(|| ovf1("-", a))
}

pub fn abs(a: i64) -> Result<i64, String> {
    a.checked_abs().ok_or_else(|| ovf1("abs", a))
}

/// An Int where a number is needed: exact when its magnitude is at most
/// 2^53, refused otherwise.
pub fn to_num(a: i64) -> Result<f64, String> {
    if !(-EXACT..=EXACT).contains(&a) {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("converting {a} to a number would change it; write float(n) to round it")));
    }
    Ok(a as f64)
}

/// A number computed where an Int is needed (`len`, `int`, `floor`, `ceil`,
/// `round` in an Int position): it must be an integer that fits.
pub fn of_num(x: f64) -> Result<i64, String> {
    if x.is_nan() {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, "cannot convert float NaN to integer"));
    }
    if x.is_infinite() {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, "cannot convert float infinity to integer"));
    }
    if x - x.floor() != 0.0 {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("{} is not an integer", crate::interp::format_rt(x))));
    }
    // [-2^63, 2^63): the doubles that convert without leaving 64 bits
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&x) {
        return Err(crate::errkind::kinded(crate::errkind::ErrKind::Overflow, format!("{} does not fit in an Int", crate::interp::format_rt(x))));
    }
    Ok(x as i64)
}

/// `a` against the number `x`, exactly, for a comparison: -1, 0 or 1, or
/// NaN when `x` is NaN (so every ordered comparison is false and `!=` true).
pub fn cmp_num(a: i64, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if (-EXACT..=EXACT).contains(&a) {
        let y = a as f64;
        return if y < x { -1.0 } else if y > x { 1.0 } else { 0.0 };
    }
    // |a| > 2^53: a double out there is an integer, or out of range
    if x >= 9_223_372_036_854_775_808.0 {
        return -1.0;
    }
    if x < -9_223_372_036_854_775_808.0 {
        return 1.0;
    }
    let b = x as i64;
    if a < b { -1.0 } else if a > b { 1.0 } else { 0.0 }
}

/// An Int as `print`, `str` and `fmt(n, "d")` write it: its digits, right
/// aligned in `width` columns.
pub fn text(a: i64, width: usize) -> String {
    format!("{a:>width$}")
}

/// The Int forms of the bit builtins: the 64-bit two's-complement pattern.
/// A shift count from 0 to 63 shifts; 64 and more gives 0 (or, for `sar` of
/// a negative Int, -1); a negative count stops the program.
pub fn bits(name: &str, a: i64, b: i64) -> Result<i64, String> {
    let shift = |k: i64| -> Result<Option<u32>, String> {
        if k < 0 {
            return Err(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{name}` takes a shift count of 0 or more, not {k}")));
        }
        Ok(if k >= 64 { None } else { Some(k as u32) })
    };
    Ok(match name {
        "band" => a & b,
        "bor" => a | b,
        "bxor" => a ^ b,
        "shl" => shift(b)?.map_or(0, |k| ((a as u64) << k) as i64),
        "shr" => shift(b)?.map_or(0, |k| ((a as u64) >> k) as i64),
        "sar" => shift(b)?.map_or(if a < 0 { -1 } else { 0 }, |k| a >> k),
        "wrap_add" => a.wrapping_add(b),
        "wrap_sub" => a.wrapping_sub(b),
        "wrap_mul" => a.wrapping_mul(b),
        // the high 64 bits of the 128-bit product of the unsigned patterns
        "umulhi" => (((a as u64 as u128) * (b as u64 as u128)) >> 64) as u64 as i64,
        _ => return Err(format!("internal: no Int bit operation `{name}`")),
    })
}

/// `ult(a, b)`: `a < b` with both read as unsigned 64-bit patterns.
pub fn ult(a: i64, b: i64) -> bool {
    (a as u64) < (b as u64)
}

/// `ustr(a)`: the pattern as an unsigned decimal.
pub fn ustr(a: i64) -> String {
    (a as u64).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_division_and_modulo_follow_python() {
        // Python: -7 // 2 == -4, -7 % 2 == 1, 7 % -2 == -1, 7 // -2 == -4
        assert_eq!(floordiv(-7, 2), Ok(-4));
        assert_eq!(floormod(-7, 2), Ok(1));
        assert_eq!(floormod(7, -2), Ok(-1));
        assert_eq!(floordiv(7, -2), Ok(-4));
        assert_eq!(floormod(i64::MIN, -1), Ok(0));
        assert!(floordiv(i64::MIN, -1).is_err());
        assert!(floormod(1, 0).is_err());
    }

    #[test]
    fn powers_are_checked_without_a_spurious_square() {
        assert_eq!(pow(-2, 63), Ok(i64::MIN));
        assert_eq!(pow(3, 39), Ok(4052555153018976267));
        assert!(pow(3, 40).is_err());
        assert_eq!(pow(0, 0), Ok(1));
        assert!(pow(2, -1).is_err());
    }

    #[test]
    fn conversions_are_exact_or_refused() {
        assert_eq!(to_num(EXACT), Ok(9007199254740992.0));
        assert!(to_num(EXACT + 1).is_err());
        assert_eq!(of_num(-3.0), Ok(-3));
        assert!(of_num(2.5).is_err());
        assert!(of_num(9_223_372_036_854_775_808.0).is_err());
        assert_eq!(of_num(-9_223_372_036_854_775_808.0), Ok(i64::MIN));
        assert_eq!(cmp_num(EXACT + 1, EXACT as f64), 1.0);
        assert!(cmp_num(1, f64::NAN).is_nan());
    }

    #[test]
    fn unsigned_helpers() {
        assert_eq!(bits("umulhi", -1, -1), Ok(-2));
        assert_eq!(bits("shr", -1, 60), Ok(15));
        assert_eq!(bits("sar", -16, 2), Ok(-4));
        assert_eq!(bits("shl", 1, 64), Ok(0));
        assert!(ult(1, -1));
        assert_eq!(ustr(-1), "18446744073709551615");
    }
}

/// `sum(a)` of an Int array: exact, or the program stops on overflow (B343).
pub fn sum(x: &[i64]) -> Result<i64, String> {
    x.iter().try_fold(0i64, |s, &v| add(s, v))
}

/// `min(a)` / `max(a)` (`big`) of an Int array (B343).
pub fn extreme(x: &[i64], big: bool) -> Result<i64, String> {
    let m = if big { x.iter().max() } else { x.iter().min() };
    m.copied().ok_or_else(|| crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`{}` of an empty Int array", if big { "max" } else { "min" })))
}
