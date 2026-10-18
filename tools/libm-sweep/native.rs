// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The native interpreter's side of the sweep: Viv's libm in Rust
// (lang/viv/src/vmath/) compiled for the host, against the same compiled from
// its C twin (v_<fn>.bin, what a native Viv program calls). Counts the inputs
// where they differ; run.sh expects 0 for all. (Before Viv had its own libm,
// the interpreter called f64::sin, the platform's; that agreed with C's libm
// on every input and with nothing else.)
#[path = "../../lang/viv/src/vmath/mod.rs"]
mod vmath;
use std::fs;

fn rd(f: &str) -> Vec<f64> {
    fs::read(f).unwrap().chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let x = rd("in.bin");
    let p = rd("pow.bin");
    let fns: [(&str, fn(f64) -> f64); 11] = [
        ("sin", vmath::sin), ("cos", vmath::cos), ("tan", vmath::tan), ("exp", vmath::exp), ("log", vmath::log),
        ("atan", vmath::atan), ("asin", vmath::asin), ("acos", vmath::acos), ("sinh", vmath::sinh),
        ("cosh", vmath::cosh), ("tanh", vmath::tanh),
    ];
    let differs = |a: f64, b: f64| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan());
    let mut bad = 0;
    for (name, f) in fns {
        let v = rd(&format!("v_{name}.bin"));
        let n = x.iter().zip(&v).filter(|(&a, &b)| differs(f(a), b)).count();
        bad += n;
        println!("native Rust vmath {name}: {n} of {} differ from vmath.c", x.len());
    }
    let v = rd("v_pow.bin");
    let n = (0..p.len() / 2).filter(|&i| differs(vmath::pow(p[2 * i], p[2 * i + 1]), v[i])).count();
    bad += n;
    println!("native Rust vmath pow: {n} of {} differ from vmath.c", p.len() / 2);
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
