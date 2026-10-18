// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The digits lens: which digits of a program's printed numbers are real.
//!
//! `viv digits prog.viv` runs the program in the interpreter (exact mode)
//! once as written, then again with its floating-point evaluation perturbed,
//! and counts, for every number the program prints, how many leading
//! significant digits stay the same. Three perturbations, each switchable:
//!
//! - reduction order: every built-in reduction (`sum`, `mean`, `dot`, the
//!   sums inside `@`) adds its terms pairwise, left to right, right to left,
//!   or in seeded random permutations;
//! - FMA contraction: `a*b + c` (scalar or elementwise), and the
//!   multiply-adds of `dot` and `@`, as one fused operation, which is what a
//!   C compiler may do under `-ffp-contract=fast`;
//! - Monte Carlo arithmetic, Verificarlo's method (Denis, de Oliveira
//!   Castro and Petit, ARITH 2016): each `+ - * /` gets random noise of
//!   half an ulp at 53 bits on its result (`rr`), its operands (`pb`) or both
//!   (`mca`, the default); over N samples the significant digits are
//!   `-log10(sigma / |mu|)`. An operation with a whole result and no
//!   rounding error is counting (an index, a size, a sum of charges) and
//!   stays exact, unless an operand is written as a math function's value:
//!   `1.0 - cos(x)` is perturbed even when cos(x) is 1.0. (A whole value
//!   that reaches the operation through a variable is taken as counting.)
//!
//! Attribution reruns the program with one source line perturbed at a time,
//! the lines that executed the most perturbable operations first.
//!
//! The runtime half is thread-local state the interpreter consults at its
//! arithmetic, reduction, matmul and print sites, only while `on()` holds;
//! a run with the identity setting computes exactly what `viv run` does.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

/// The most decimal digits a double carries: -log10(2^-53).
pub const FULL: f64 = 15.95;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    /// as the program does it
    Native,
    Pairwise,
    LeftToRight,
    Reversed,
    Random,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mca {
    Off,
    /// random rounding: noise on each result
    Rr,
    /// precision bounding: noise on each operand
    Pb,
    /// both
    Full,
}

impl Mca {
    pub fn parse(s: &str) -> Option<Mca> {
        Some(match s {
            "off" | "none" => Mca::Off,
            "rr" => Mca::Rr,
            "pb" => Mca::Pb,
            "mca" | "full" => Mca::Full,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        match self {
            Mca::Off => "off",
            Mca::Rr => "rr",
            Mca::Pb => "pb",
            Mca::Full => "mca",
        }
    }
}

/// One run's perturbation. `lines`: only the operations on these lines of
/// the program text (1-based, inclusive) are perturbed; `None`, all of them.
#[derive(Clone, Copy, Debug)]
pub struct Perturb {
    pub order: Order,
    pub fma: bool,
    pub mca: Mca,
    pub seed: u64,
    pub lines: Option<(u32, u32)>,
}

impl Perturb {
    pub const IDENTITY: Perturb = Perturb { order: Order::Native, fma: false, mca: Mca::Off, seed: 0, lines: None };
}

/// What one line of the program executed: arithmetic operations, terms of
/// reductions, multiply-adds that could be contracted.
#[derive(Clone, Copy, Default, Debug)]
pub struct Count {
    pub ops: u64,
    pub terms: u64,
    pub fmas: u64,
}

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static CFG: Cell<Perturb> = const { Cell::new(Perturb::IDENTITY) };
    static RNG: Cell<u64> = const { Cell::new(1) };
    static STARTS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static SITE: Cell<u32> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static COUNTS: RefCell<BTreeMap<u32, Count>> = const { RefCell::new(BTreeMap::new()) };
    static VALUES: RefCell<Vec<(usize, f64)>> = const { RefCell::new(Vec::new()) };
    static LABELS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static PRINTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

// ── the hooks the interpreter calls ─────────────────────────────────────────

#[inline]
pub fn on() -> bool {
    ON.with(|c| c.get())
}

/// The operation about to run is at byte `off` of the program text.
#[inline]
pub fn at(off: u32) {
    SITE.with(|s| s.set(off));
}

/// Whether the scalar operation about to run is written as counting
/// (`interp::counting_expr`): `i + 1`, `n - 1 - k`, `len(x) * 2`.
#[inline]
pub fn counting(yes: bool) {
    COUNTING.with(|c| c.set(yes));
}

/// One element of an array operation (an index table is an array of
/// whole numbers, so the counting rule is the scalar one).
pub fn arith_elem(op: char, a: f64, b: f64) -> f64 {
    arith(op, a, b)
}

pub fn push_call(off: u32) {
    CALLS.with(|c| c.borrow_mut().push(off));
}

pub fn pop_call() {
    CALLS.with(|c| {
        c.borrow_mut().pop();
    });
}

fn line_of(off: u32) -> u32 {
    STARTS.with(|s| {
        let s = s.borrow();
        match s.binary_search(&off) {
            Ok(i) => i as u32 + 1,
            Err(i) => i as u32,
        }
    })
}

/// Count an event at `off` and say whether this run perturbs it.
fn here(off: u32, f: impl FnOnce(&mut Count)) -> bool {
    let line = line_of(off);
    COUNTS.with(|c| f(c.borrow_mut().entry(line).or_default()));
    let cfg = CFG.with(|c| c.get());
    cfg.lines.map_or(true, |(a, b)| a <= line && line <= b)
}

fn cfg() -> Perturb {
    CFG.with(|c| c.get())
}

fn op_site() -> u32 {
    SITE.with(|s| s.get())
}

fn call_site() -> u32 {
    CALLS.with(|c| c.borrow().last().copied()).unwrap_or_else(op_site)
}

/// A uniform number in [0, 1) (xorshift64*).
fn uniform() -> f64 {
    RNG.with(|r| {
        let mut x = r.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        r.set(x);
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// The unit in the last place of `x` at 53 bits: 2^(e - 52) for x = 1.m 2^e.
fn ulp(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return 0.0;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7ff) as i32;
    if e == 0 {
        return f64::from_bits(1); // subnormal: the spacing is the smallest
    }
    2f64.powi(e - 1023 - 52)
}

/// Verificarlo's noise: ulp(x) times a uniform number in (-1/2, 1/2).
fn noise(x: f64) -> f64 {
    ulp(x) * (uniform() - 0.5)
}

/// `a op b` under Monte Carlo arithmetic: the operands' noise carried to
/// first order beside the rounded result, the rounding error added exactly
/// (error-free transformations), the result's noise, then one rounding.
fn mca_op(op: char, a: f64, b: f64, mode: Mca) -> f64 {
    let inb = matches!(mode, Mca::Pb | Mca::Full);
    let (da, db) = if inb { (noise(a), noise(b)) } else { (0.0, 0.0) };
    let (hi, lo) = match op {
        '+' | '-' => {
            let b = if op == '-' { -b } else { b };
            let db = if op == '-' { -db } else { db };
            let s = a + b;
            let bb = s - a;
            let err = (a - (s - bb)) + (b - bb);
            (s, err + da + db)
        }
        '*' => {
            let p = a * b;
            (p, a.mul_add(b, -p) + a * db + b * da)
        }
        _ => {
            let q = a / b;
            // the exact remainder of the rounded quotient, then first order
            let rem = (-q).mul_add(b, a);
            (q, (rem + da - q * db) / b)
        }
    };
    if !hi.is_finite() {
        return hi;
    }
    let lo = if matches!(mode, Mca::Rr | Mca::Full) { lo + noise(hi) } else { lo };
    hi + lo
}

/// A scalar `a op b` (op one of + - * /) at the current site.
pub fn arith(op: char, a: f64, b: f64) -> f64 {
    let cfg = cfg();
    let r = match op {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        _ => a / b,
    };
    // counting: a whole result with no rounding error (a loop index, a
    // size, an electron count, 2 * 25.5), which C would hold in an int,
    // unless an operand is a math function's value. It stays exact.
    if COUNTING.with(|c| c.get()) && whole(r) && exact(op, a, b, r) {
        return r;
    }
    let hit = here(op_site(), |c| c.ops += 1);
    if hit && cfg.mca != Mca::Off {
        return mca_op(op, a, b, cfg.mca);
    }
    r
}

/// Whether `r` is `a op b` with no rounding error.
fn exact(op: char, a: f64, b: f64, r: f64) -> bool {
    match op {
        '+' => r - a == b && r - b == a,
        '-' => a - r == b && r + b == a,
        '*' => a.mul_add(b, -r) == 0.0,
        _ => (-r).mul_add(b, a) == 0.0,
    }
}

fn whole(x: f64) -> bool {
    x.fract() == 0.0 && x.abs() < 9_007_199_254_740_992.0
}

/// Whether `a*b + c` at the current site is to be fused in this run.
pub fn fuse() -> bool {
    let cfg = cfg();
    here(op_site(), |c| c.fmas += 1) && cfg.fma
}

/// The add a perturbed reduction uses.
fn add(a: f64, b: f64, mode: Mca) -> f64 {
    if mode == Mca::Off {
        a + b
    }
    else {
        mca_op('+', a, b, mode)
    }
}

fn fold(xs: &[f64], mode: Mca) -> f64 {
    let mut t = 0.0;
    for &x in xs {
        t = add(t, x, mode);
    }
    t
}

fn pairwise(xs: &[f64], mode: Mca) -> f64 {
    match xs.len() {
        0 => 0.0,
        1 => xs[0],
        n => add(pairwise(&xs[..n / 2], mode), pairwise(&xs[n / 2..], mode), mode),
    }
}

/// Terms in the order `order` says, summed with `mode`'s adds; `native` is
/// the program's own order.
fn reorder(xs: &[f64], order: Order, mode: Mca, native: &dyn Fn(&[f64], Mca) -> f64) -> f64 {
    match order {
        Order::Native => native(xs, mode),
        Order::Pairwise => pairwise(xs, mode),
        Order::LeftToRight => fold(xs, mode),
        Order::Reversed => {
            let r: Vec<f64> = xs.iter().rev().copied().collect();
            fold(&r, mode)
        }
        Order::Random => {
            let mut r = xs.to_vec();
            for i in (1..r.len()).rev() {
                let j = (uniform() * (i + 1) as f64) as usize;
                r.swap(i, j.min(i));
            }
            fold(&r, mode)
        }
    }
}

fn perturbs(cfg: &Perturb) -> bool {
    cfg.order != Order::Native || cfg.mca != Mca::Off || cfg.fma
}

/// Whole terms whose sum is exact in any order: counting.
fn whole_sum(xs: &[f64]) -> bool {
    xs.iter().all(|x| whole(*x)) && xs.iter().map(|x| x.abs()).sum::<f64>() < 9_007_199_254_740_992.0
}

/// `sum` (and `mean`, `var`) over a contiguous run whose order is `native`.
pub fn sum(xs: &[f64], native: fn(&[f64]) -> f64) -> f64 {
    let cfg = cfg();
    let hit = here(call_site(), |c| c.terms += xs.len() as u64);
    if !hit || (cfg.order == Order::Native && cfg.mca == Mca::Off) || whole_sum(xs) {
        return native(xs);
    }
    reorder(xs, cfg.order, cfg.mca, &|v, m| if m == Mca::Off { native(v) } else { fold(v, m) })
}

/// `sum(a, ordered=true)`: left to right.
pub fn sum_ordered(xs: &[f64]) -> f64 {
    let cfg = cfg();
    let hit = here(call_site(), |c| c.terms += xs.len() as u64);
    if !hit || !perturbs(&cfg) || whole_sum(xs) {
        return fold(xs, Mca::Off);
    }
    reorder(xs, cfg.order, cfg.mca, &fold)
}

/// sum_k u[k] v[k], as the program writes it: left to right from 0, each
/// product rounded; fused when the run contracts.
fn dot_at(u: &[f64], v: &[f64], cfg: Perturb, hit: bool) -> f64 {
    if !hit || !perturbs(&cfg) {
        let mut t = 0.0;
        for i in 0..u.len() {
            t += u[i] * v[i];
        }
        return t;
    }
    let mut idx: Vec<usize> = (0..u.len()).collect();
    match cfg.order {
        Order::Reversed => idx.reverse(),
        Order::Random => {
            for i in (1..idx.len()).rev() {
                let j = (uniform() * (i + 1) as f64) as usize;
                idx.swap(i, j.min(i));
            }
        }
        _ => {}
    }
    if cfg.order == Order::Pairwise {
        let p: Vec<f64> = idx.iter().map(|&i| mul(u[i], v[i], cfg.mca)).collect();
        return pairwise(&p, cfg.mca);
    }
    let mut t = 0.0;
    for &i in &idx {
        t = if cfg.fma { fma(u[i], v[i], t, cfg.mca) } else { add(t, mul(u[i], v[i], cfg.mca), cfg.mca) };
    }
    t
}

fn mul(a: f64, b: f64, mode: Mca) -> f64 {
    if mode == Mca::Off {
        a * b
    }
    else {
        mca_op('*', a, b, mode)
    }
}

/// a*b + c in one rounding; under MCA, noise on the fused result.
fn fma(a: f64, b: f64, c: f64, mode: Mca) -> f64 {
    let r = a.mul_add(b, c);
    if matches!(mode, Mca::Rr | Mca::Full) && r.is_finite() {
        r + noise(r)
    }
    else {
        r
    }
}

/// `a*b + c` fused (the FMA perturbation) at the current site.
pub fn fused(a: f64, b: f64, c: f64) -> f64 {
    fma(a, b, c, cfg().mca)
}

/// `dot(u, v)`.
pub fn dot(u: &[f64], v: &[f64]) -> f64 {
    let cfg = cfg();
    let hit = here(call_site(), |c| {
        c.terms += u.len() as u64;
        c.fmas += u.len() as u64;
    });
    dot_at(u, v, cfg, hit && !(whole_sum(u) && whole_sum(v)))
}

/// The matmul kernel `r = a @ b` (m x k times k x n) at the current
/// operator site, or `None` when this run leaves it alone.
pub fn matmul(a: &[f64], b: &[f64], r: &mut [f64], m: usize, k: usize, n: usize) -> bool {
    let cfg = cfg();
    let hit = here(op_site(), |c| {
        c.terms += (m * k * n) as u64;
        c.fmas += (m * k * n) as u64;
    });
    if !hit || !perturbs(&cfg) || (whole_sum(a) && whole_sum(b)) {
        return false;
    }
    let mut col = vec![0.0; k];
    for j in 0..n {
        for (q, c) in col.iter_mut().enumerate() {
            *c = b[q * n + j];
        }
        for i in 0..m {
            r[i * n + j] = dot_at(&a[i * k..(i + 1) * k], &col, cfg, true);
        }
    }
    true
}

/// A number the program printed.
pub fn capture(x: f64) {
    capture_as(x, String::new());
}

/// A number the program printed, with the name it was printed under
/// (`MODEL_COEF[1]`, or the text before it on the line).
pub fn capture_as(x: f64, label: String) {
    let n = PRINTS.with(|p| p.borrow().len());
    VALUES.with(|v| v.borrow_mut().push((n, x)));
    LABELS.with(|l| l.borrow_mut().push(label));
}

/// The text of a printed line, after its numbers were captured.
/// A line with no number captured (the numbers were formatted into a
/// string) gives the numbers its text shows, to the digits it shows.
pub fn printed(line: &str) {
    let n = PRINTS.with(|p| p.borrow().len());
    if !VALUES.with(|v| v.borrow().last().is_some_and(|(k, _)| *k == n)) {
        // each named by the word before it
        let mut prev = "";
        for w in line.split(|c: char| c.is_whitespace() || matches!(c, ',' | '[' | ']' | '(' | ')' | '=' | ':' | ';')) {
            if w.is_empty() {
                continue;
            }
            match w.parse::<f64>() {
                Ok(x) if w.chars().any(|c| c.is_ascii_digit()) && w.contains(['.', 'e', 'E']) => {
                    VALUES.with(|v| v.borrow_mut().push((n, x)));
                    LABELS.with(|l| l.borrow_mut().push(prev.to_string()));
                }
                Ok(_) => {}
                Err(_) => prev = w,
            }
        }
    }
    PRINTS.with(|p| p.borrow_mut().push(line.to_string()));
}

// ── the driver ──────────────────────────────────────────────────────────────

/// What `viv digits` runs.
#[derive(Clone, Debug)]
pub struct Options {
    pub orders: bool,
    /// random permutations, besides pairwise / left to right / reversed
    pub randoms: usize,
    pub fma: bool,
    pub mca: Mca,
    pub samples: usize,
    pub seed: u64,
    /// how many lines to perturb one at a time
    pub attrib: usize,
    /// how many numbers whose lost digits no hot line explains to trace to
    /// a line by bisection
    pub trace: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { orders: true, randoms: 3, fma: true, mca: Mca::Full, samples: 16, seed: 1, attrib: 8, trace: 2 }
    }
}

/// One run's printed numbers, or why it stopped.
struct Outcome {
    values: Vec<(usize, f64)>,
    labels: Vec<String>,
    prints: Vec<String>,
    err: Option<String>,
}

fn run_once(src: &str, p: Perturb, run: &dyn Fn(&str) -> Result<String, String>) -> Outcome {
    CFG.with(|c| c.set(p));
    RNG.with(|r| r.set(p.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1));
    VALUES.with(|v| v.borrow_mut().clear());
    LABELS.with(|v| v.borrow_mut().clear());
    PRINTS.with(|v| v.borrow_mut().clear());
    CALLS.with(|v| v.borrow_mut().clear());
    ON.with(|o| o.set(true));
    let r = run(src);
    ON.with(|o| o.set(false));
    Outcome {
        values: VALUES.with(|v| std::mem::take(&mut *v.borrow_mut())),
        labels: LABELS.with(|v| std::mem::take(&mut *v.borrow_mut())),
        prints: PRINTS.with(|v| std::mem::take(&mut *v.borrow_mut())),
        err: r.err(),
    }
}

/// Leading significant digits two numbers share: -log10 of the relative
/// difference from `a`, between 0 and FULL.
pub fn agree(a: f64, b: f64) -> f64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return FULL;
    }
    if a == 0.0 || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    (-((a - b) / a).abs().log10()).clamp(0.0, FULL)
}

/// Verificarlo's significant digits of samples: -log10(sigma / |mu|).
pub fn sig_digits(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if xs.len() < 2 || xs.iter().all(|x| *x == xs[0]) {
        return FULL;
    }
    let mu = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    if sd == 0.0 {
        return FULL;
    }
    if mu == 0.0 || !mu.is_finite() {
        return 0.0;
    }
    (-(sd / mu.abs()).log10()).clamp(0.0, FULL)
}

/// Stable digits of one printed number under each perturbation.
#[derive(Clone, Debug)]
pub struct Digits {
    pub order: Option<f64>,
    pub fma: Option<f64>,
    pub mca: Option<f64>,
}

impl Digits {
    pub fn stable(&self) -> f64 {
        [self.order, self.fma, self.mca].iter().flatten().fold(FULL, |m, &x| m.min(x))
    }
}

#[derive(Clone, Debug)]
pub struct Output {
    /// which printed line (0-based) it is on, and that line's text
    pub print: usize,
    pub text: String,
    /// the name it was printed under, when the line gives one
    pub label: String,
    pub value: f64,
    pub digits: Digits,
}

/// One line's effect, perturbed alone: the digits each output keeps, the
/// fewest, and by which perturbation.
#[derive(Clone, Debug)]
pub struct Site {
    pub line: u32,
    pub count: Count,
    /// (output index, digits kept, perturbation)
    pub effects: Vec<(usize, f64, &'static str)>,
    /// found by bisection for this output, not by its operation count
    pub traced: Option<usize>,
}

pub struct Report {
    pub outputs: Vec<Output>,
    pub sites: Vec<Site>,
    pub runs: usize,
    /// why a perturbed run stopped, and the line of its last operation
    pub failed: Vec<(String, u32)>,
    pub opts: Options,
    /// lines that executed perturbable operations, with their counts
    pub counts: Vec<(u32, Count)>,
}

/// The runs a perturbation set makes, each with its kind.
fn plan(o: &Options, lines: Option<(u32, u32)>, samples: usize, randoms: usize) -> Vec<(&'static str, Perturb)> {
    let mut v = Vec::new();
    let base = Perturb { lines, ..Perturb::IDENTITY };
    if o.orders {
        for ord in [Order::Pairwise, Order::LeftToRight, Order::Reversed] {
            v.push(("order", Perturb { order: ord, ..base }));
        }
        for k in 0..randoms {
            v.push(("order", Perturb { order: Order::Random, seed: o.seed + k as u64, ..base }));
        }
    }
    if o.fma {
        v.push(("fma", Perturb { fma: true, ..base }));
    }
    if o.mca != Mca::Off {
        for k in 0..samples {
            v.push(("mca", Perturb { mca: o.mca, seed: o.seed + 1000 + k as u64, ..base }));
        }
    }
    v
}

/// Per output, the digits each kind of perturbation keeps.
fn score(base: &[(usize, f64)], runs: &[(&'static str, Outcome)]) -> Vec<Digits> {
    base.iter()
        .enumerate()
        .map(|(j, &(_, x0))| {
            let kind = |k: &str| -> Vec<Option<f64>> {
                runs.iter().filter(|(kk, _)| *kk == k).map(|(_, o)| if o.err.is_some() { None } else { o.values.get(j).map(|v| v.1) }).collect()
            };
            let det = |k: &str| -> Option<f64> {
                let xs = kind(k);
                if xs.is_empty() {
                    return None;
                }
                Some(xs.iter().map(|x| x.map_or(0.0, |x| agree(x0, x))).fold(FULL, f64::min))
            };
            let mca = {
                let xs = kind("mca");
                if xs.is_empty() {
                    None
                }
                else if xs.iter().any(|x| x.is_none()) {
                    Some(0.0)
                }
                else {
                    Some(sig_digits(&xs.iter().flatten().copied().collect::<Vec<_>>()))
                }
            };
            Digits { order: det("order"), fma: det("fma"), mca }
        })
        .collect()
}

/// Run the analysis. `run` executes the program text once in the
/// interpreter (on this thread, with a stack for it) and returns its output.
pub fn analyze(src: &str, o: &Options, run: &dyn Fn(&str) -> Result<String, String>) -> Result<Report, String> {
    let mut starts = vec![0u32];
    for (i, b) in src.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i as u32 + 1);
        }
    }
    STARTS.with(|s| *s.borrow_mut() = starts);
    COUNTS.with(|c| c.borrow_mut().clear());
    let base = run_once(src, Perturb::IDENTITY, run);
    if let Some(e) = base.err {
        return Err(e);
    }
    let counts: Vec<(u32, Count)> = COUNTS.with(|c| c.borrow().iter().map(|(k, v)| (*k, *v)).collect());
    let mut failed = Vec::new();
    let go = |plan: Vec<(&'static str, Perturb)>, failed: &mut Vec<(String, u32)>| -> Vec<(&'static str, Outcome)> {
        plan.into_iter()
            .map(|(k, p)| {
                let out = run_once(src, p, run);
                if let Some(e) = &out.err {
                    let what = match (k, p.order) {
                        ("order", o) => format!("order {o:?}"),
                        _ => format!("{k} seed {}", p.seed),
                    };
                    let only = p.lines.map_or(String::new(), |(a, b)| if a == b { format!(", line {a} alone") } else { format!(", lines {a} to {b}") });
                    failed.push((format!("{what}{only}: {}", e.lines().next().unwrap_or("")), line_of(op_site())));
                }
                (k, out)
            })
            .collect()
    };
    let all = go(plan(o, None, o.samples, o.randoms), &mut failed);
    let mut runs = all.len() + 1;
    let digits = score(&base.values, &all);
    let outputs: Vec<Output> = base
        .values
        .iter()
        .zip(digits)
        .enumerate()
        .map(|(k, (&(print, value), digits))| Output {
            print,
            text: base.prints.get(print).cloned().unwrap_or_default(),
            label: base.labels.get(k).cloned().unwrap_or_default(),
            value,
            digits,
        })
        .collect();
    // attribution: the most-executed lines first, each perturbed alone
    let mut hot: Vec<(u32, Count)> = counts.clone();
    hot.sort_by_key(|(l, c)| (std::cmp::Reverse(c.ops + c.terms + c.fmas), *l));
    let count_of = |line: u32| counts.iter().find(|(l, _)| *l == line).map_or(Count::default(), |(_, c)| *c);
    let mut sites: Vec<Site> = Vec::new();
    let unstable = outputs.iter().any(|x| x.digits.stable() < FULL);
    let alone = |line: u32, traced: Option<usize>, runs: &mut usize, failed: &mut Vec<(String, u32)>| -> Site {
        let rs = go(plan(o, Some((line, line)), o.samples.min(6), o.randoms.min(1)), failed);
        *runs += rs.len();
        let effects = score(&base.values, &rs).iter().enumerate().filter_map(|(j, dg)| worst(dg).filter(|w| w.1 < FULL).map(|(k, d)| (j, d, k))).collect();
        Site { line, count: count_of(line), effects, traced }
    };
    if unstable {
        for &(line, _) in hot.iter().take(o.attrib) {
            sites.push(alone(line, None, &mut runs, &mut failed));
        }
        // a number whose lost digits no line so far explains: bisect the
        // lines that ran perturbable operations, following the half that
        // moves it more, with the perturbation that moved it most
        let cand: Vec<u32> = counts.iter().map(|(l, _)| *l).collect();
        let mut order: Vec<usize> = (0..outputs.len()).filter(|&j| outputs[j].digits.stable() < FULL - 1.0).collect();
        order.sort_by(|a, b| outputs[*a].digits.stable().total_cmp(&outputs[*b].digits.stable()));
        let mut traced = 0;
        for j in order {
            if traced >= o.trace || cand.is_empty() {
                break;
            }
            let want = outputs[j].digits.stable() + 1.0;
            if sites.iter().any(|st| st.effects.iter().any(|(jj, d, _)| *jj == j && *d <= want)) {
                continue;
            }
            let Some((kind, _)) = worst(&outputs[j].digits) else { continue };
            let only = Options { orders: kind == "order", fma: kind == "fma", mca: if kind == "mca" { o.mca } else { Mca::Off }, ..o.clone() };
            let probe = |a: usize, b: usize, runs: &mut usize, failed: &mut Vec<(String, u32)>| -> f64 {
                let rs = go(plan(&only, Some((cand[a], cand[b])), o.samples.min(4), 1), failed);
                *runs += rs.len();
                score(&base.values, &rs).get(j).and_then(worst).map_or(FULL, |w| w.1)
            };
            let (mut lo, mut hi) = (0, cand.len() - 1);
            while lo < hi {
                let mid = (lo + hi) / 2;
                let left = probe(lo, mid, &mut runs, &mut failed);
                let right = probe(mid + 1, hi, &mut runs, &mut failed);
                if left <= right {
                    hi = mid;
                }
                else {
                    lo = mid + 1;
                }
            }
            traced += 1;
            if !sites.iter().any(|st| st.line == cand[lo]) {
                sites.push(alone(cand[lo], Some(j), &mut runs, &mut failed));
            }
        }
    }
    Ok(Report { outputs, sites, runs, failed, opts: o.clone(), counts })
}

// ── rendering ───────────────────────────────────────────────────────────────

/// The perturbation that kept the fewest digits, and how many.
fn worst(dg: &Digits) -> Option<(&'static str, f64)> {
    [("order", dg.order), ("fma", dg.fma), ("mca", dg.mca)]
        .into_iter()
        .filter_map(|(k, x)| x.map(|x| (k, x)))
        .fold(None, |m: Option<(&'static str, f64)>, (k, x)| match m {
            Some((_, y)) if y <= x => m,
            _ => Some((k, x)),
        })
}

/// `x` with its first `d` significant digits, a `|`, then the rest:
/// `5.000004|163e-11`.
pub fn split_digits(x: f64, d: f64) -> String {
    let s = format!("{:.16e}", x);
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    // all of them: no bar
    let keep = if d >= FULL { digits.len() } else { (d.floor() as usize).min(digits.len()) };
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    for (i, ch) in digits.chars().enumerate() {
        if i == keep {
            out.push('|');
        }
        out.push(ch);
        if i == 0 {
            out.push('.');
        }
    }
    let out = out.trim_end_matches('0').trim_end_matches('.').to_string();
    format!("{out}e{exp}")
}

fn opt(x: Option<f64>) -> String {
    x.map_or("-".into(), |x| if x >= FULL { "all".into() } else { format!("{x:.1}") })
}

fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn jnum(x: Option<f64>) -> String {
    x.map_or("null".into(), |x| format!("{:.2}", x))
}

impl Report {
    fn modes(&self) -> String {
        let o = &self.opts;
        let mut m = Vec::new();
        if o.orders {
            m.push(format!("reduction order (pairwise, left to right, reversed, {} random)", o.randoms));
        }
        if o.fma {
            m.push("FMA contraction".into());
        }
        if o.mca != Mca::Off {
            m.push(format!("Monte Carlo arithmetic ({}, 53 bits, {} samples)", o.mca.name(), o.samples));
        }
        m.join("; ")
    }

    /// The text tables. `loc` names a line of the program text
    /// (`file:line` and the line itself).
    pub fn text(&self, loc: &dyn Fn(u32) -> (String, String)) -> String {
        let mut s = String::new();
        s.push_str(&format!("{} runs: {}\n", self.runs, self.modes()));
        s.push_str("digits: the leading significant digits that stay the same; `|` marks where they stop\n\n");
        if self.outputs.is_empty() {
            s.push_str("the program printed no numbers\n");
        }
        let w = self.outputs.iter().map(|o| split_digits(o.value, o.digits.stable()).len()).max().unwrap_or(5).max(5);
        s.push_str(&format!("  #  {:<w$}  order    fma    mca  stable  printed as\n", "value"));
        for (j, o) in self.outputs.iter().enumerate() {
            let line = if o.label.is_empty() { o.text.chars().take(40).collect::<String>() } else { o.label.clone() };
            s.push_str(&format!(
                "{:>3}  {:<w$}  {:>5}  {:>5}  {:>5}  {:>6}  {}\n",
                j + 1,
                split_digits(o.value, o.digits.stable()),
                opt(o.digits.order),
                opt(o.digits.fma),
                opt(o.digits.mca),
                opt(Some(o.digits.stable())),
                line
            ));
        }
        if !self.sites.is_empty() {
            s.push_str("\nwhich lines move the digits (each line perturbed alone: the most-executed first, then lines traced by bisection)\n");
            for site in &self.sites {
                let (at, text) = loc(site.line);
                let n = site.count.ops + site.count.terms + site.count.fmas;
                let eff = if site.effects.is_empty() {
                    "moves no printed digit".to_string()
                }
                else {
                    let mut e: Vec<&(usize, f64, &str)> = site.effects.iter().collect();
                    e.sort_by(|a, b| a.1.total_cmp(&b.1));
                    let shown: Vec<String> = e.iter().take(4).map(|(j, d, k)| format!("#{} to {} ({k})", j + 1, opt(Some(*d)))).collect();
                    let more = if e.len() > 4 { format!(", {} more", e.len() - 4) } else { String::new() };
                    format!("{}{more}", shown.join(", "))
                };
                let how = site.traced.map_or(String::new(), |j| format!("  (traced from #{})", j + 1));
                s.push_str(&format!("  {at}  {n} ops  {eff}{how}\n      {}\n", text.trim()));
            }
        }
        if !self.failed.is_empty() {
            s.push_str(&format!("\n{} perturbed run(s) stopped with an error (their outputs count as 0 digits):\n", self.failed.len()));
            for (f, line) in self.failed.iter().take(5) {
                let (at, text) = loc(*line);
                s.push_str(&format!("  {f}\n      after {at}: {}\n", text.trim()));
            }
        }
        s
    }

    pub fn json(&self, loc: &dyn Fn(u32) -> (String, String)) -> String {
        let o = &self.opts;
        let mut s = format!(
            "{{\"runs\":{},\"modes\":{{\"orders\":{},\"randoms\":{},\"fma\":{},\"mca\":\"{}\",\"samples\":{},\"seed\":{}}},\"outputs\":[",
            self.runs,
            o.orders,
            o.randoms,
            o.fma,
            o.mca.name(),
            o.samples,
            o.seed
        );
        for (j, x) in self.outputs.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"print\":{},\"text\":\"{}\",\"label\":\"{}\",\"value\":{:e},\"order\":{},\"fma\":{},\"mca\":{},\"stable\":{:.2}}}",
                x.print,
                esc(&x.text),
                esc(&x.label),
                x.value,
                jnum(x.digits.order),
                jnum(x.digits.fma),
                jnum(x.digits.mca),
                x.digits.stable()
            ));
        }
        s.push_str("],\"lines\":[");
        let attributed: BTreeMap<u32, &Site> = self.sites.iter().map(|x| (x.line, x)).collect();
        for (i, (line, c)) in self.counts.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let (at, text) = loc(*line);
            s.push_str(&format!("{{\"at\":\"{}\",\"code\":\"{}\",\"ops\":{},\"terms\":{},\"fmas\":{}", esc(&at), esc(text.trim()), c.ops, c.terms, c.fmas));
            if let Some(site) = attributed.get(line) {
                s.push_str(",\"effects\":[");
                for (k, (j, d, by)) in site.effects.iter().enumerate() {
                    if k > 0 {
                        s.push(',');
                    }
                    s.push_str(&format!("{{\"output\":{},\"digits\":{:.2},\"by\":\"{by}\"}}", j, d));
                }
                s.push(']');
                if let Some(j) = site.traced {
                    s.push_str(&format!(",\"traced\":{j}"));
                }
            }
            s.push('}');
        }
        s.push_str("],\"failed\":[");
        s.push_str(&self.failed.iter().map(|(f, l)| format!("{{\"why\":\"{}\",\"at\":\"{}\"}}", esc(f), esc(&loc(*l).0))).collect::<Vec<_>>().join(","));
        s.push_str("]}");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agree_counts_shared_digits() {
        assert_eq!(agree(1.0, 1.0), FULL);
        assert!((agree(1.0, 1.0 + 1e-6) - 6.0).abs() < 1e-6);
        assert_eq!(agree(0.0, 1e-300), 0.0);
    }

    #[test]
    fn split_marks_the_boundary() {
        assert_eq!(split_digits(1.25, 2.4), "1.2|5e0");
        assert_eq!(split_digits(-2.5, FULL), "-2.5e0");
    }

    #[test]
    fn mca_noise_is_half_an_ulp_on_a_result() {
        CFG.with(|c| c.set(Perturb { mca: Mca::Rr, ..Perturb::IDENTITY }));
        RNG.with(|r| r.set(7));
        let xs: Vec<f64> = (0..200).map(|_| mca_op('+', 1.0, 1e-3, Mca::Rr)).collect();
        let exact = 1.0 + 1e-3;
        assert!(xs.iter().all(|x| (x - exact).abs() <= 2.0 * ulp(exact)));
        assert!(xs.iter().any(|x| *x != exact));
    }
}
