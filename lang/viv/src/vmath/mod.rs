// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2005-2020 Rich Felker et al. (musl)
// SPDX-FileCopyrightText: 2018 Jorge Aparicio and the libm crate contributors
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv's transcendental functions: one algorithm for every engine.
//!
//! `f64::sin` and the rest call the platform's libm, so the interpreter on a
//! Mac, the interpreter compiled to wasm (Rust's portable libm) and V8's
//! `Math` each round differently in the last bit, on 2 to 38% of inputs
//! (`tools/libm-sweep/run.sh`). A program whose stopping test sits near its
//! tolerance then prints different digits in different engines. So Viv carries
//! its own libm, and every engine calls it: the interpreter (native and in
//! wasm) and the direct wasm backend through this module, the C backend
//! through `vmath.c`, its line-for-line twin in C. `cargo test` compiles the
//! C twin and requires the two to agree bit for bit on a sweep.
//!
//! The code is the `libm` crate 0.2.8 (MIT OR Apache-2.0), a port of musl's
//! libm, which is FreeBSD's msun, which is Sun's fdlibm. It is the same code
//! Rust's standard library runs on wasm32-unknown-unknown, so the Run button's
//! numbers did not move when this replaced the standard library's calls.
//! `exp` is Viv's own and correctly rounded (exp_cr.rs; its C twin is
//! `viv_exp_cr`), since release 0.0.1 (bugs/ledger.jsonl B292). musl's exp
//! (exp.rs, here `exp_musl`) stays inside the musl code that calls it
//! (cosh, sinh and their __expo2) and at the ends of exp_cr's range; Viv's
//! own erf, gamma and complex functions call the correctly rounded one.
//! Only the functions Viv calls are here; the crate's tests are left out
//! (the sweep in `tests/vmath.rs` checks them against the C twin instead).
//! `sqrt`, `floor` and `fabs` are exact in IEEE arithmetic in every engine,
//! so they are the standard library's.
#![allow(clippy::all)]
#![allow(unused_unsafe, unused_parens, dead_code)]

macro_rules! force_eval {
    ($e:expr) => {
        unsafe { ::core::ptr::read_volatile(&$e) }
    };
}

// The crate's checked-index macro (its debug-build form: a bounds check, no
// `unsafe`).
macro_rules! i {
    ($array:expr, $index:expr) => {
        *$array.get($index).unwrap()
    };
    ($array:expr, $index:expr, = , $rhs:expr) => {
        *$array.get_mut($index).unwrap() = $rhs;
    };
    ($array:expr, $index:expr, -= , $rhs:expr) => {
        *$array.get_mut($index).unwrap() -= $rhs;
    };
    ($array:expr, $index:expr, += , $rhs:expr) => {
        *$array.get_mut($index).unwrap() += $rhs;
    };
    ($array:expr, $index:expr, &= , $rhs:expr) => {
        *$array.get_mut($index).unwrap() &= $rhs;
    };
    ($array:expr, $index:expr, == , $rhs:expr) => {
        *$array.get_mut($index).unwrap() == $rhs
    };
}

macro_rules! div {
    ($a:expr, $b:expr) => {
        $a / $b
    };
}

mod acos;
mod asin;
mod atan;
mod atan2;
mod cos;
mod cosh;
mod exp;
mod exp_cr;
mod expm1;
mod expo2;
mod hypot;
mod k_cos;
mod k_expo2;
mod k_sin;
mod k_tan;
mod log;
mod log10;
mod pow;
mod rem_pio2;
mod rem_pio2_large;
mod scalbn;
mod sin;
mod sinh;
mod tan;
mod tanh;

pub use self::acos::acos;
pub use self::asin::asin;
pub use self::atan::atan;
pub use self::atan2::atan2;
pub use self::cos::cos;
pub use self::cosh::cosh;
pub use self::exp::exp as exp_musl;
pub use self::exp_cr::exp;
pub use self::exp_cr::exp as exp_cr;
pub use self::expm1::expm1;
pub use self::hypot::hypot;
pub use self::log::log;
pub use self::log10::log10;
pub use self::pow::pow;
pub use self::sin::sin;
pub use self::sinh::sinh;
pub use self::tan::tan;
pub use self::tanh::tanh;

use self::expo2::expo2;
use self::k_cos::k_cos;
use self::k_expo2::k_expo2;
use self::k_sin::k_sin;
use self::k_tan::k_tan;
use self::rem_pio2::rem_pio2;
use self::rem_pio2_large::rem_pio2_large;
use self::scalbn::scalbn;

// ---- Viv's own: erf, gamma and the complex functions. Their C twins are the
// tail of vmath.c (tools/vmath/vmath_tail.c), line for line. ----

/// erf through the incomplete gamma function P(1/2, x^2): its series below
/// x^2 = 1.5, its continued fraction above.
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = 0.5_f64;
    let xx = x * x;
    // erf(±inf) is ±1 (and so is erf of any x whose square overflows): the
    // formulas below would take -inf + inf
    if xx > f64::MAX {
        return sign;
    }
    let gln = 0.572_364_942_924_700_1_f64; // ln Γ(1/2) = ln √π
    let p = if xx < 1.5 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..300 {
            ap += 1.0;
            del *= xx / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        sum * exp(-xx + a * log(xx) - gln)
    } else {
        let mut b = xx + 1.0 - a;
        let mut c = 1e300_f64;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..300 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-17 {
                break;
            }
        }
        1.0 - exp(-xx + a * log(xx) - gln) * h
    };
    sign * p
}

/// Γ via the Lanczos approximation (g = 7, 9 terms), reflection below 1/2.
pub fn gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let pi = std::f64::consts::PI;
    if x < 0.5 {
        pi / (sin(pi * x) * gamma(1.0 - x))
    } else if x > 172.0 {
        // Γ overflows past 171.62; the formula would take inf * 0 (gamma(inf))
        f64::INFINITY
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, &g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        (2.0 * pi).sqrt() * pow(t, x + 0.5) * exp(-t) * a
    }
}

/// max, the same in every engine and on every platform: a NaN gives the
/// other argument, and +0 is larger than -0. f64::max leaves the sign of a
/// tie of zeros to the target (compiled to wasm, max(-0.0, 0.0) was -0.0).
/// The C twin is viv_fmax.
pub fn fmax(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return y;
    }
    if y.is_nan() {
        return x;
    }
    if x > y {
        x
    } else if y > x {
        y
    } else if x.is_sign_negative() {
        y
    } else {
        x
    }
}

/// min, as `fmax`: a NaN gives the other argument, and -0 is smaller than +0.
pub fn fmin(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return y;
    }
    if y.is_nan() {
        return x;
    }
    if x < y {
        x
    } else if y < x {
        y
    } else if x.is_sign_negative() {
        x
    } else {
        y
    }
}

/// |re + i im|.
pub fn cabs(re: f64, im: f64) -> f64 {
    hypot(re, im)
}

/// e^z = e^re (cos im + i sin im).
pub fn cexp(re: f64, im: f64) -> (f64, f64) {
    let m = exp(re);
    (m * cos(im), m * sin(im))
}

/// The principal log: ln |z| + i arg z.
pub fn clog(re: f64, im: f64) -> (f64, f64) {
    (log(hypot(re, im)), atan2(im, re))
}

/// Above this a part is scaled by 1/4 before `|re| + |z|`, which could overflow.
const CSQRT_BIG: f64 = f64::MAX / 4.0;

/// The principal square root, by the half-angle formulas (Kahan, "Branch cuts
/// for complex elementary functions", 1987; C99 Annex G's csqrt): with
/// t = sqrt((|re| + |z|) / 2), the root is t + i im/(2t) when re >= 0 and
/// |im|/(2t) + i copysign(t, im) when re < 0. No angle is formed, so a root
/// on an axis is exact: sqrt(-1 + 0i) is 0 + 1i, not cos(pi/2) + i. The branch
/// cut is the negative real axis, and the sign of a zero imaginary part picks
/// the side (sqrt(-1 - 0i) is 0 - 1i). Every engine computes it with these
/// operations, in this order (vmath.c: viv_csqrt).
pub fn csqrt(re: f64, im: f64) -> (f64, f64) {
    if re == 0.0 && im == 0.0 {
        return (0.0, im);
    }
    if im.is_infinite() {
        return (f64::INFINITY, im);
    }
    let big = re.abs() > CSQRT_BIG || im.abs() > CSQRT_BIG;
    let (a, b, s) = if big { (re * 0.25, im * 0.25, 2.0) } else { (re, im, 1.0) };
    let t = ((a.abs() + hypot(a, b)) * 0.5).sqrt();
    if a >= 0.0 {
        (s * t, s * (b / (2.0 * t)))
    } else {
        (s * (b.abs() / (2.0 * t)), s * t.copysign(b))
    }
}

pub fn csin(re: f64, im: f64) -> (f64, f64) {
    (sin(re) * cosh(im), cos(re) * sinh(im))
}

pub fn ccos(re: f64, im: f64) -> (f64, f64) {
    (cos(re) * cosh(im), -sin(re) * sinh(im))
}

/// (ar + i ai)(br + i bi) = (ar br - ai bi) + i (ar bi + ai br): four
/// products, each rounded (Rust never fuses a multiply-add; the C twin says
/// FP_CONTRACT OFF). A real operand is lifted to (x, 0.0) by the caller, and
/// its zero products are taken, not skipped.
pub fn cmul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// (ar + i ai) / (br + i bi) by Smith's algorithm (R. L. Smith, "Algorithm
/// 116: Complex division", CACM 5(8), 1962): divide through by the larger
/// component of the divisor, so the divisor is never squared. The textbook
/// (a conj b) / |b|^2 overflows once |b| passes 1e154 and underflows below
/// 1e-154; this overflows only near where the quotient itself does. A real
/// divisor (bi = 0) gives r = 0 and d = br, so each part is one correctly
/// rounded division. A zero divisor gives NaN + NaN i. Every engine divides
/// with exactly these operations, in this order (vmath.c: viv_cdiv_parts).
pub fn cdiv(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    if br.abs() >= bi.abs() {
        let r = bi / br;
        let d = br + bi * r;
        ((ar + ai * r) / d, (ai - ar * r) / d)
    } else {
        let r = br / bi;
        let d = bi + br * r;
        ((ar * r + ai) / d, (ai * r - ar) / d)
    }
}

/// sin z / cos z, by `cdiv`.
pub fn ctan(re: f64, im: f64) -> (f64, f64) {
    let (sr, si) = (sin(re) * cosh(im), cos(re) * sinh(im));
    let (cr, ci) = (cos(re) * cosh(im), -sin(re) * sinh(im));
    cdiv(sr, si, cr, ci)
}

#[inline]
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[inline]
fn floor(x: f64) -> f64 {
    x.floor()
}

#[inline]
fn fabs(x: f64) -> f64 {
    x.abs()
}

#[inline]
fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn get_low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline]
fn with_set_high_word(f: f64, hi: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0x00000000_ffffffff;
    tmp |= (hi as u64) << 32;
    f64::from_bits(tmp)
}

#[inline]
fn with_set_low_word(f: f64, lo: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0xffffffff_00000000;
    tmp |= lo as u64;
    f64::from_bits(tmp)
}

#[inline]
fn combine_words(hi: u32, lo: u32) -> f64 {
    f64::from_bits((hi as u64) << 32 | lo as u64)
}

/// The methods the interpreter calls, spelled as `f64` methods so a call
/// site reads as before: `x.vsin()` is `vmath::sin(x)`. (A trait method
/// cannot take the standard library's names: an inherent method wins.)
pub trait VMath {
    fn vsin(self) -> f64;
    fn vcos(self) -> f64;
    fn vtan(self) -> f64;
    fn vexp(self) -> f64;
    fn vln(self) -> f64;
    fn vlog10(self) -> f64;
    fn vpow(self, y: f64) -> f64;
    fn vatan(self) -> f64;
    fn vatan2(self, x: f64) -> f64;
    fn vasin(self) -> f64;
    fn vacos(self) -> f64;
    fn vsinh(self) -> f64;
    fn vcosh(self) -> f64;
    fn vtanh(self) -> f64;
    fn vhypot(self, y: f64) -> f64;
}

impl VMath for f64 {
    #[inline]
    fn vsin(self) -> f64 {
        sin(self)
    }
    #[inline]
    fn vcos(self) -> f64 {
        cos(self)
    }
    #[inline]
    fn vtan(self) -> f64 {
        tan(self)
    }
    #[inline]
    fn vexp(self) -> f64 {
        exp(self)
    }
    #[inline]
    fn vln(self) -> f64 {
        log(self)
    }
    #[inline]
    fn vlog10(self) -> f64 {
        log10(self)
    }
    #[inline]
    fn vpow(self, y: f64) -> f64 {
        pow(self, y)
    }
    #[inline]
    fn vatan(self) -> f64 {
        atan(self)
    }
    #[inline]
    fn vatan2(self, x: f64) -> f64 {
        atan2(self, x)
    }
    #[inline]
    fn vasin(self) -> f64 {
        asin(self)
    }
    #[inline]
    fn vacos(self) -> f64 {
        acos(self)
    }
    #[inline]
    fn vsinh(self) -> f64 {
        sinh(self)
    }
    #[inline]
    fn vcosh(self) -> f64 {
        cosh(self)
    }
    #[inline]
    fn vtanh(self) -> f64 {
        tanh(self)
    }
    #[inline]
    fn vhypot(self, y: f64) -> f64 {
        hypot(self, y)
    }
}
