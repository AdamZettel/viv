// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Binary input: `read_bytes` and the little-endian readers `le_u8` ...
//! `le_f64`. A file's bytes are an `[Int]`, one
//! element per byte (0 to 255), and a reader takes the array and an offset
//! and gives the Int or the number those bytes encode, little-endian, as
//! NumPy's `frombuffer(..., dtype="<u4")` and the rest read them.
//!
//! The interpreter calls these, and so does the wasm runtime (`wasm_rt.rs`),
//! so the two say the same words; the C runtime's fragment `int_bytes`
//! (lower/ints.rs) is their twin, line for line.

use crate::errkind::{kinded, ErrKind};

/// Every reader: its Viv name, its width in bytes, and whether it gives a
/// number (a float format) rather than an Int.
pub const READERS: &[(&str, usize, bool)] = &[
    ("le_u8", 1, false),
    ("le_i8", 1, false),
    ("le_u16", 2, false),
    ("le_i16", 2, false),
    ("le_u32", 4, false),
    ("le_i32", 4, false),
    ("le_u64", 8, false),
    ("le_i64", 8, false),
    ("le_f16", 2, true),
    ("le_bf16", 2, true),
    ("le_f32", 4, true),
    ("le_f64", 8, true),
];

/// The reader `name`: its width and whether it gives a number.
pub fn reader(name: &str) -> Option<(usize, bool)> {
    READERS.iter().find(|r| r.0 == name).map(|r| (r.1, r.2))
}

/// The `width` bytes at `off`, as one unsigned 64-bit pattern, little-endian:
/// each element checked to be a byte, the window checked to be inside.
fn pattern(name: &str, b: &[i64], off: i64, width: usize) -> Result<u64, String> {
    let n = b.len() as i64;
    if off < 0 || off > n - width as i64 {
        return Err(kinded(ErrKind::Index, format!("`{name}`: {width} bytes at offset {off} are not inside an array of {n}")));
    }
    let mut u = 0u64;
    for k in 0..width {
        let at = off as usize + k;
        let x = b[at];
        if !(0..=255).contains(&x) {
            return Err(kinded(ErrKind::Value, format!("`{name}`: element {at} is {x}, not a byte (0 to 255)")));
        }
        u |= (x as u64) << (8 * k);
    }
    Ok(u)
}

/// An IEEE half (binary16) as the double it is, exactly: subnormals,
/// infinities and NaNs (payload and sign kept) included.
pub fn f16_value(h: u64) -> f64 {
    let sign = (h >> 15) & 1;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let bits = if e == 0 {
        if m == 0 {
            sign << 63
        }
        else {
            // m * 2^-24, normalized: shift the leading 1 into place
            let lead = 63 - m.leading_zeros() as u64; // 0..=9
            let exp = lead + 1023 - 24;
            (sign << 63) | (exp << 52) | ((m << (52 - lead)) & ((1u64 << 52) - 1))
        }
    }
    else if e == 0x1f {
        (sign << 63) | (0x7ffu64 << 52) | (m << 42)
    }
    else {
        (sign << 63) | ((e + 1023 - 15) << 52) | (m << 42)
    };
    f64::from_bits(bits)
}

/// A bfloat16 as the double it is: the top half of a binary32.
pub fn bf16_value(h: u64) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

/// What a reader gives: an Int or a number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Read {
    Int(i64),
    Num(f64),
}

/// `name(b, off)` for a reader `name`.
pub fn read(name: &str, b: &[i64], off: i64) -> Result<Read, String> {
    let Some((width, _)) = reader(name) else {
        return Err(format!("internal: no reader `{name}`"));
    };
    let u = pattern(name, b, off, width)?;
    Ok(match name {
        "le_u8" | "le_u16" | "le_u32" => Read::Int(u as i64),
        "le_i8" => Read::Int(u as u8 as i8 as i64),
        "le_i16" => Read::Int(u as u16 as i16 as i64),
        "le_i32" => Read::Int(u as u32 as i32 as i64),
        "le_i64" => Read::Int(u as i64),
        "le_u64" => {
            if u > i64::MAX as u64 {
                return Err(kinded(ErrKind::Overflow, format!("`le_u64`: {u} does not fit in an Int; le_i64 reads the same bits")));
            }
            Read::Int(u as i64)
        }
        "le_f16" => Read::Num(f16_value(u)),
        "le_bf16" => Read::Num(bf16_value(u)),
        "le_f32" => Read::Num(f32::from_bits(u as u32) as f64),
        _ => Read::Num(f64::from_bits(u)),
    })
}

/// The message for an offset or a count of `read_bytes` that is not a whole
/// number of 0 or more.
pub const BAD_WINDOW: &str = "`read_bytes` takes a whole offset and count of 0 or more";

/// The window `[off, off + count)` of a file of `n` bytes (`count` NaN: to
/// the end), as Python's `f.seek(off); f.read(count)` reads it: fewer bytes
/// where the file ends, none past its end.
pub fn window(n: u64, off: f64, count: f64) -> Result<(u64, u64), String> {
    let whole = |x: f64| x >= 0.0 && x.floor() == x && x < 9.0e15;
    if !whole(off) || !(count.is_nan() || whole(count)) {
        return Err(kinded(ErrKind::Value, BAD_WINDOW));
    }
    let o = (off as u64).min(n);
    let left = n - o;
    let c = if count.is_nan() { left } else { (count as u64).min(left) };
    Ok((o, c))
}

/// `read_bytes(path, off, count)`: the bytes, each an Int.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_bytes(path: &str, off: f64, count: f64) -> Result<Vec<i64>, String> {
    use std::io::{Read as _, Seek, SeekFrom};
    let open = std::fs::File::open(path);
    crate::trace::note(std::path::Path::new(path), open.is_ok());
    let mut f = open.map_err(|e| super::strings::file_error(path, &e))?;
    let n = f.metadata().map_err(|e| super::strings::file_error(path, &e))?.len();
    let (o, c) = window(n, off, count)?;
    f.seek(SeekFrom::Start(o)).map_err(|e| super::strings::file_error(path, &e))?;
    let mut buf = vec![0u8; c as usize];
    f.read_exact(&mut buf).map_err(|e| super::strings::file_error(path, &e))?;
    Ok(buf.into_iter().map(i64::from).collect())
}

/// In the browser: the bytes of a file the host attached (`vrt_file_put`).
#[cfg(target_arch = "wasm32")]
pub fn read_bytes(path: &str, off: f64, count: f64) -> Result<Vec<i64>, String> {
    let bytes = super::strings::FILES.with(|f| f.borrow().get(path).cloned());
    let Some(bytes) = bytes else {
        return Err(format!("No such file or directory: {} (in the browser a program reads the files attached to it)", super::strings::repr(path)));
    };
    let (o, c) = window(bytes.len() as u64, off, count)?;
    Ok(bytes[o as usize..(o + c) as usize].iter().map(|&x| i64::from(x)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_are_exact() {
        // every binary16 pattern against Rust's own widening of the f32 that
        // holds the same value (an f16 widens to f32 exactly)
        for h in 0u64..=0xffff {
            let v = f16_value(h);
            let e = (h >> 10) & 0x1f;
            let m = h & 0x3ff;
            let s = if h >> 15 == 1 { -1.0 } else { 1.0 };
            let want = if e == 0 {
                s * m as f64 * 2f64.powi(-24)
            }
            else if e == 31 {
                if m == 0 { s * f64::INFINITY } else { f64::NAN }
            }
            else {
                s * (1024 + m) as f64 * 2f64.powi(e as i32 - 25)
            };
            if want.is_nan() {
                assert!(v.is_nan(), "{h:#x}");
            }
            else {
                assert_eq!(v.to_bits(), want.to_bits(), "{h:#x}");
            }
        }
    }

    #[test]
    fn readers() {
        let b = [0x01, 0x02, 0x03, 0x04, 0xff, 0xff, 0xff, 0xff];
        assert_eq!(read("le_u32", &b, 0), Ok(Read::Int(0x04030201)));
        assert_eq!(read("le_i32", &b, 4), Ok(Read::Int(-1)));
        assert_eq!(read("le_u16", &b, 4), Ok(Read::Int(0xffff)));
        assert_eq!(read("le_i8", &b, 7), Ok(Read::Int(-1)));
        assert_eq!(read("le_f16", &[0x00, 0x3c], 0), Ok(Read::Num(1.0)));
        assert_eq!(read("le_bf16", &[0x80, 0x3f], 0), Ok(Read::Num(1.0)));
        assert!(read("le_u64", &b, 0).is_err());
        assert!(read("le_u32", &b, 5).is_err());
        assert!(read("le_u8", &[256], 0).is_err());
        assert!(read("le_u8", &[1], -1).is_err());
    }
}
