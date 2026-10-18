// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2005-2020 Rich Felker et al. (musl)
// SPDX-FileCopyrightText: 2018 Jorge Aparicio and the libm crate contributors
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `exp`, correctly rounded: the double nearest e^x. musl's `exp` (exp.rs,
//! fdlibm) is within one ulp, and off by one in the last bit on some inputs,
//! `exp(1.0)` among them (2.7182818284590455 where the nearest
//! double to e is 2.718281828459045, which CPython's math.exp(1) and math.e
//! give). The C twin is `viv_exp_cr` in vmath_tail.c, line for line.
//!
//! x = k ln2/128 + r, |r| <= ln2/256, and e^x = 2^(k/128) e^r with 2^(j/128)
//! from a table (j = k mod 128). The fast path sums 2^(j/128) (1 + r + r^2/2
//! + ... + r^6/720) with an error below 2^-66 relative (2^(j/128) r is formed
//! exactly, by Dekker's product) and keeps the rounding error of the last sum
//! exactly; when that leaves the result within the error bound of a midpoint
//! between two doubles (about one input in 5,000), the slow path redoes it in
//! double-double arithmetic (about 2^-100). It costs about 1.8 times musl's
//! exp (6 ns against 3.5 on an M-series core). Below |x| = 2^-54 the answer
//! is 1 + x. At or below x = -708 (results below 2^-1021, subnormal ones
//! among them) the slow path's double-double is rounded once to the grid of
//! 2^-1074 (`exp_tiny`); up to 709.79 the scaling by 2^e is done in two
//! exact steps where 2^e itself would overflow. Only past -746 (a result
//! that rounds to 0) and 709.79 (to inf) is it musl's. The constants come
//! from tools/vmath/exp_table.py; tools/vmath/check_exp.py checks the result
//! against mpmath.
use super::exp::exp as exp_musl;

const EXP_L1: u64 = 0x3f762e42fefa0000;
const EXP_L2: u64 = 0x3d0cf79abc9e3b3a;
const EXP_L3: u64 = 0xb9aff0342542fc33;
const EXP_INVL: u64 = 0x40671547652b82fe;
#[rustfmt::skip]
const EXP_T: [u64; 256] = [
    0x3ff0000000000000, 0x0000000000000000, 0x3ff0163da9fb3335, 0x3c9b61299ab8cdb7,
    0x3ff02c9a3e778061, 0xbc719083535b085d, 0x3ff04315e86e7f85, 0xbc90a31c1977c96e,
    0x3ff059b0d3158574, 0x3c8d73e2a475b465, 0x3ff0706b29ddf6de, 0xbc8c91dfe2b13c27,
    0x3ff0874518759bc8, 0x3c6186be4bb284ff, 0x3ff09e3ecac6f383, 0x3c91487818316136,
    0x3ff0b5586cf9890f, 0x3c98a62e4adc610b, 0x3ff0cc922b7247f7, 0x3c901edc16e24f71,
    0x3ff0e3ec32d3d1a2, 0x3c403a1727c57b53, 0x3ff0fb66affed31b, 0xbc6b9bedc44ebd7b,
    0x3ff11301d0125b51, 0xbc96c51039449b3a, 0x3ff12abdc06c31cc, 0xbc51b514b36ca5c7,
    0x3ff1429aaea92de0, 0xbc932fbf9af1369e, 0x3ff15a98c8a58e51, 0x3c82406ab9eeab0a,
    0x3ff172b83c7d517b, 0xbc819041b9d78a76, 0x3ff18af9388c8dea, 0xbc911023d1970f6c,
    0x3ff1a35beb6fcb75, 0x3c8e5b4c7b4968e4, 0x3ff1bbe084045cd4, 0xbc995386352ef607,
    0x3ff1d4873168b9aa, 0x3c9e016e00a2643c, 0x3ff1ed5022fcd91d, 0xbc91df98027bb78c,
    0x3ff2063b88628cd6, 0x3c8dc775814a8495, 0x3ff21f49917ddc96, 0x3c82a97e9494a5ee,
    0x3ff2387a6e756238, 0x3c99b07eb6c70573, 0x3ff251ce4fb2a63f, 0x3c8ac155bef4f4a4,
    0x3ff26b4565e27cdd, 0x3c82bd339940e9d9, 0x3ff284dfe1f56381, 0xbc9a4c3a8c3f0d7e,
    0x3ff29e9df51fdee1, 0x3c8612e8afad1255, 0x3ff2b87fd0dad990, 0xbc410adcd6381aa4,
    0x3ff2d285a6e4030b, 0x3c90024754db41d5, 0x3ff2ecafa93e2f56, 0x3c71ca0f45d52383,
    0x3ff306fe0a31b715, 0x3c86f46ad23182e4, 0x3ff32170fc4cd831, 0x3c8a9ce78e18047c,
    0x3ff33c08b26416ff, 0x3c932721843659a6, 0x3ff356c55f929ff1, 0xbc8b5cee5c4e4628,
    0x3ff371a7373aa9cb, 0xbc963aeabf42eae2, 0x3ff38cae6d05d866, 0xbc9e958d3c9904bd,
    0x3ff3a7db34e59ff7, 0xbc75e436d661f5e3, 0x3ff3c32dc313a8e5, 0xbc9efff8375d29c3,
    0x3ff3dea64c123422, 0x3c8ada0911f09ebc, 0x3ff3fa4504ac801c, 0xbc97d023f956f9f3,
    0x3ff4160a21f72e2a, 0xbc5ef3691c309278, 0x3ff431f5d950a897, 0xbc81c7dde35f7999,
    0x3ff44e086061892d, 0x3c489b7a04ef80d0, 0x3ff46a41ed1d0057, 0x3c9c944bd1648a76,
    0x3ff486a2b5c13cd0, 0x3c73c1a3b69062f0, 0x3ff4a32af0d7d3de, 0x3c99cb62f3d1be56,
    0x3ff4bfdad5362a27, 0x3c7d4397afec42e2, 0x3ff4dcb299fddd0d, 0x3c98ecdbbc6a7833,
    0x3ff4f9b2769d2ca7, 0xbc94b309d25957e3, 0x3ff516daa2cf6642, 0xbc8f768569bd93ef,
    0x3ff5342b569d4f82, 0xbc807abe1db13cad, 0x3ff551a4ca5d920f, 0xbc8d689cefede59b,
    0x3ff56f4736b527da, 0x3c99bb2c011d93ad, 0x3ff58d12d497c7fd, 0x3c8295e15b9a1de8,
    0x3ff5ab07dd485429, 0x3c96324c054647ad, 0x3ff5c9268a5946b7, 0x3c3c4b1b816986a2,
    0x3ff5e76f15ad2148, 0x3c9ba6f93080e65e, 0x3ff605e1b976dc09, 0xbc93e2429b56de47,
    0x3ff6247eb03a5585, 0xbc9383c17e40b497, 0x3ff6434634ccc320, 0xbc8c483c759d8933,
    0x3ff6623882552225, 0xbc9bb60987591c34, 0x3ff68155d44ca973, 0x3c6038ae44f73e65,
    0x3ff6a09e667f3bcd, 0xbc9bdd3413b26456, 0x3ff6c012750bdabf, 0xbc72895667ff0b0d,
    0x3ff6dfb23c651a2f, 0xbc6bbe3a683c88ab, 0x3ff6ff7df9519484, 0xbc883c0f25860ef6,
    0x3ff71f75e8ec5f74, 0xbc816e4786887a99, 0x3ff73f9a48a58174, 0xbc90a8d96c65d53c,
    0x3ff75feb564267c9, 0xbc90245957316dd3, 0x3ff780694fde5d3f, 0x3c9866b80a02162d,
    0x3ff7a11473eb0187, 0xbc841577ee04992f, 0x3ff7c1ed0130c132, 0x3c9f124cd1164dd6,
    0x3ff7e2f336cf4e62, 0x3c705d02ba15797e, 0x3ff80427543e1a12, 0xbc927c86626d972b,
    0x3ff82589994cce13, 0xbc9d4c1dd41532d8, 0x3ff8471a4623c7ad, 0xbc88d684a341cdfb,
    0x3ff868d99b4492ed, 0xbc9fc6f89bd4f6ba, 0x3ff88ac7d98a6699, 0x3c9994c2f37cb53a,
    0x3ff8ace5422aa0db, 0x3c96e9f156864b27, 0x3ff8cf3216b5448c, 0xbc70d55e32e9e3aa,
    0x3ff8f1ae99157736, 0x3c85cc13a2e3976c, 0x3ff9145b0b91ffc6, 0xbc9dd6792e582524,
    0x3ff93737b0cdc5e5, 0xbc675fc781b57ebc, 0x3ff95a44cbc8520f, 0xbc764b7c96a5f039,
    0x3ff97d829fde4e50, 0xbc9d185b7c1b85d1, 0x3ff9a0f170ca07ba, 0xbc9173bd91cee632,
    0x3ff9c49182a3f090, 0x3c7c7c46b071f2be, 0x3ff9e86319e32323, 0x3c7824ca78e64c6e,
    0x3ffa0c667b5de565, 0xbc9359495d1cd533, 0x3ffa309bec4a2d33, 0x3c96305c7ddc36ab,
    0x3ffa5503b23e255d, 0xbc9d2f6edb8d41e1, 0x3ffa799e1330b358, 0x3c9bcb7ecac563c7,
    0x3ffa9e6b5579fdbf, 0x3c90fac90ef7fd31, 0x3ffac36bbfd3f37a, 0xbc8f9234cae76cd0,
    0x3ffae89f995ad3ad, 0x3c97a1cd345dcc81, 0x3ffb0e07298db666, 0xbc9bdef54c80e425,
    0x3ffb33a2b84f15fb, 0xbc62805e3084d708, 0x3ffb59728de5593a, 0xbc9c71dfbbba6de3,
    0x3ffb7f76f2fb5e47, 0xbc75584f7e54ac3b, 0x3ffba5b030a1064a, 0xbc9efcd30e54292e,
    0x3ffbcc1e904bc1d2, 0x3c823dd07a2d9e84, 0x3ffbf2c25bd71e09, 0xbc9efdca3f6b9c73,
    0x3ffc199bdd85529c, 0x3c811065895048dd, 0x3ffc40ab5fffd07a, 0x3c9b4537e083c60a,
    0x3ffc67f12e57d14b, 0x3c92884dff483cad, 0x3ffc8f6d9406e7b5, 0x3c71acbc48805c44,
    0x3ffcb720dcef9069, 0x3c7503cbd1e949db, 0x3ffcdf0b555dc3fa, 0xbc8dd83b53829d72,
    0x3ffd072d4a07897c, 0xbc9cbc3743797a9c, 0x3ffd2f87080d89f2, 0xbc9d487b719d8578,
    0x3ffd5818dcfba487, 0x3c82ed02d75b3707, 0x3ffd80e316c98398, 0xbc911ec18beddfe8,
    0x3ffda9e603db3285, 0x3c9c2300696db532, 0x3ffdd321f301b460, 0x3c92da5778f018c3,
    0x3ffdfc97337b9b5f, 0xbc91a5cd4f184b5c, 0x3ffe264614f5a129, 0xbc97b627817a1496,
    0x3ffe502ee78b3ff6, 0x3c839e8980a9cc8f, 0x3ffe7a51fbc74c83, 0x3c92d522ca0c8de2,
    0x3ffea4afa2a490da, 0xbc9e9c23179c2893, 0x3ffecf482d8e67f1, 0xbc9c93f3b411ad8c,
    0x3ffefa1bee615a27, 0x3c9dc7f486a4b6b0, 0x3fff252b376bba97, 0x3c93a1a5bf0d8e43,
    0x3fff50765b6e4540, 0x3c99d3e12dd8a18b, 0x3fff7bfdad9cbe14, 0xbc9dbb12d006350a,
    0x3fffa7c1819e90d8, 0x3c874853f3a5931e, 0x3fffd3c22b8f71f1, 0x3c62eb74966579e7,
];
#[rustfmt::skip]
const EXP_C: [u64; 26] = [
    0x3ff0000000000000, 0x0000000000000000, 0x3ff0000000000000, 0x0000000000000000,
    0x3fe0000000000000, 0x0000000000000000, 0x3fc5555555555555, 0x3c65555555555555,
    0x3fa5555555555555, 0x3c45555555555555, 0x3f81111111111111, 0x3c01111111111111,
    0x3f56c16c16c16c17, 0xbbef49f49f49f49f, 0x3f2a01a01a01a01a, 0x3b6a01a01a01a01a,
    0x3efa01a01a01a01a, 0x3b3a01a01a01a01a, 0x3ec71de3a556c734, 0xbb6c154f8ddc6c00,
    0x3e927e4fb7789f5c, 0x3b3cbbc05b4fa99a, 0x3e5ae64567f544e4, 0xbafc062e06d1f209,
    0x3e21eed8eff8d898, 0xbac2aec959e14c06,
];

fn t(i: usize) -> f64 {
    f64::from_bits(EXP_T[i])
}

fn c(i: usize) -> f64 {
    f64::from_bits(EXP_C[i])
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

fn split(a: f64) -> (f64, f64) {
    let t = 134217729.0 * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

fn dd_mul(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (p, e) = two_prod(ah, bh);
    fast_two_sum(p, e + (ah * bl + al * bh))
}

fn dd_add(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (s, e) = two_sum(ah, bh);
    fast_two_sum(s, e + (al + bl))
}

/// x - k L1, exactly. k L1 is exact for |k| < 2^17 (L1 has 36 significant
/// bits); past that (x below -709.78 or above 709.77) k = 2m + b and
/// x - 2m L1, then - b L1, are each exact by Sterbenz's lemma.
fn reduce(x: f64, k: f64) -> f64 {
    let l1 = f64::from_bits(EXP_L1);
    if k.abs() < 131072.0 {
        return x - k * l1;
    }
    let m = (k * 0.5).floor();
    let b = k - 2.0 * m;
    (x - (2.0 * m) * l1) - b * l1
}

/// 2^(j/128) e^r in double-double (about 2^-100 relative): the slow path.
fn exp_slow_dd(x: f64, k: f64, j: usize) -> (f64, f64) {
    let t0 = reduce(x, k);
    let (p2h, p2l) = two_prod(k, f64::from_bits(EXP_L2));
    let p3 = k * f64::from_bits(EXP_L3);
    let (rh, rl) = two_sum(t0, -p2h);
    let (rh, rl) = fast_two_sum(rh, (rl - p2l) - p3);
    let mut ah = c(24);
    let mut al = c(25);
    for n in (0..12).rev() {
        let (mh, ml) = dd_mul(ah, al, rh, rl);
        let (sh, sl) = dd_add(mh, ml, c(2 * n), c(2 * n + 1));
        ah = sh;
        al = sl;
    }
    dd_mul(t(2 * j), t(2 * j + 1), ah, al)
}

fn exp_slow(x: f64, k: f64, j: usize) -> f64 {
    let (vh, vl) = exp_slow_dd(x, k, j);
    vh + vl
}

/// e^x below 2^-1021 (x <= -708): the slow path's double-double times 2^e,
/// rounded once to the grid of 2^-1074 that subnormals and the lowest binade
/// share. u = (vh + vl) 2^(e + 1074) is the result in units of 2^-1074; the
/// scalings by powers of two are exact, and so is everything after n0.
fn exp_tiny(x: f64, k: f64, j: usize, e: i64) -> f64 {
    let (vh, vl) = exp_slow_dd(x, k, j);
    let sc = f64::from_bits(((e + 1074 + 1023) as u64) << 52);
    let uh = vh * sc;
    let ul = vl * sc;
    let big = 4503599627370496.0; // 2^52: above it a double is a whole number
    let n0 = if uh >= big { uh } else { (uh + big) - big };
    let (fh, fl) = two_sum(uh - n0, ul);
    let n = if fh > 0.5 || (fh == 0.5 && fl > 0.0) {
        n0 + 1.0
    } else if fh < -0.5 || (fh == -0.5 && fl < 0.0) {
        n0 - 1.0
    } else {
        n0
    };
    n * f64::from_bits(1)
}

pub fn exp(x: f64) -> f64 {
    if !(x.abs() >= 5.551115123125783e-17) {
        // |x| < 2^-54, or NaN
        return 1.0 + x;
    }
    if !(x > -746.0 && x < 709.79) {
        // 0, or an overflow to inf (and the infinities themselves)
        return exp_musl(x);
    }
    let k = (x * f64::from_bits(EXP_INVL) + 0.5).floor();
    let ki = k as i64;
    let j = (ki & 127) as usize;
    let e = (ki - j as i64) / 128;
    if x <= -708.0 {
        return exp_tiny(x, k, j, e);
    }
    let t0 = reduce(x, k);
    let kl2 = k * f64::from_bits(EXP_L2);
    let r = t0 - kl2;
    let dr = (t0 - r) - kl2;
    let th = t(2 * j);
    let tl = t(2 * j + 1);
    let q = r * r * (c(4) + r * (c(6) + r * (c(8) + r * (c(10) + r * c(12)))));
    let (a, a_lo) = two_prod(th, r);
    let s = th + a;
    let s_lo = (th - s) + a;
    let lo = s_lo + (a_lo + (th * (q + dr) + tl * (1.0 + r)));
    let mut y = s + lo;
    let d = lo - (y - s);
    // the gap below y, which is never wider than the one above (y > 0)
    let gap = y - f64::from_bits(y.to_bits() - 1);
    if !(d.abs() <= 0.5 * gap - th * 1.3552527156068805e-20) {
        // within 2^-66 of a midpoint
        y = exp_slow(x, k, j);
    }
    if e > 1023 {
        // 2^e is past the largest double: scale in two exact steps, the
        // second of which overflows to inf exactly when e^x rounds to it
        return y * f64::from_bits(((e - 1 + 1023) as u64) << 52) * 2.0;
    }
    y * f64::from_bits(((e + 1023) as u64) << 52)
}
