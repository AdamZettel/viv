// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Integer loop counters and integer indices in the generated code
//!.
//!
//! Viv's numbers are doubles, and a `for i to n` counter, an index read out of
//! a table, or `p = p + 1` is a double too. The interpreter is the spec: an
//! index must be exactly an integer, and is then read as Rust's `as i64` reads
//! it. This module proves, per def, which double variables always hold an
//! integer (and in what range), so the C backend can
//!
//! - count every `for` loop with a 64-bit integer counter (the variable is set
//!   from it each pass, as before), and
//! - compute an index in 64-bit integers when every value the double
//!   arithmetic could produce is an integer of magnitude at most 2^53: then
//!   the double computation is exact, and the two give the same number. An
//!   index that needs a runtime fact for that (a counter whose trip count is
//!   not known, a variable that is an integer of unknown size) is guarded: the
//!   integer path runs when the guard holds, and the unchanged double path
//!   (with its "is not an integer" check) runs otherwise.
//!
//! Nothing here changes the IR: the facts are read off it at emission, so the
//! content addresses, the AST and the IR stay what they were.

use std::collections::{HashMap, HashSet};

use knotkit::backend::{CEmitCx, CExpr};
use knotkit::ir::{IrProgram, NodeId};

use super::{BinOp, Ty, Viv};

/// 2^53: every integer of at most this magnitude is a double exactly.
pub(crate) const EXACT: f64 = 9007199254740992.0;
/// Past this a range is no longer tracked: an integer that large is only
/// known to be a whole number.
const WIDE: f64 = 4611686018427387904.0; // 2^62
/// How far an Int that only steps by one can get in a run (2^61).
const INT_STEPS: f64 = 2305843009213693952.0;
/// An array dimension, a length or a shape entry is at most this (the C
/// runtime's `int` sizes).
pub(crate) const DIM_MAX: f64 = 2147483647.0;

/// What is known about a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Fact {
    /// Always finite, in `[lo, hi]`; `int`: always an integer (then `lo` and
    /// `hi` are integers too).
    Range { lo: f64, hi: f64, int: bool },
    /// Always an integer, or not finite (NaN, an infinity): a double built
    /// from integers by `+ - *`, or rounded by `round`, `floor`, ...
    Whole,
}

/// A fact during the fixpoint: `Bot` before any value has reached it, `Top`
/// when nothing is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Av {
    Bot,
    F(Fact),
    Top,
}

fn range(lo: f64, hi: f64, int: bool) -> Av {
    if lo.is_nan() || hi.is_nan() {
        return Av::Top;
    }
    if lo < -WIDE || hi > WIDE {
        return if int { Av::F(Fact::Whole) } else { Av::Top };
    }
    Av::F(Fact::Range { lo, hi, int })
}

impl Av {
    fn join(self, o: Av) -> Av {
        match (self, o) {
            (Av::Bot, x) | (x, Av::Bot) => x,
            (Av::Top, _) | (_, Av::Top) => Av::Top,
            (Av::F(a), Av::F(b)) => match (a, b) {
                (Fact::Range { lo: l1, hi: h1, int: i1 }, Fact::Range { lo: l2, hi: h2, int: i2 }) => range(l1.min(l2), h1.max(h2), i1 && i2),
                (Fact::Whole, Fact::Whole) => Av::F(Fact::Whole),
                (Fact::Whole, Fact::Range { int, .. }) | (Fact::Range { int, .. }, Fact::Whole) => {
                    if int {
                        Av::F(Fact::Whole)
                    } else {
                        Av::Top
                    }
                }
            },
        }
    }
    /// The value is always an integer (or not finite, for `Whole`).
    pub(crate) fn whole(self) -> bool {
        matches!(self, Av::F(Fact::Whole) | Av::F(Fact::Range { int: true, .. }))
    }
    pub(crate) fn bounds(self) -> Option<(f64, f64)> {
        match self {
            Av::F(Fact::Range { lo, hi, .. }) => Some((lo, hi)),
            _ => None,
        }
    }
    /// Widen a value that keeps changing: a whole number stays whole.
    fn widen(self) -> Av {
        if self.whole() {
            Av::F(Fact::Whole)
        } else if self == Av::Bot {
            Av::Bot
        } else {
            Av::Top
        }
    }
}

fn arith(op: BinOp, a: Av, b: Av) -> Av {
    if a == Av::Bot || b == Av::Bot {
        return Av::Bot;
    }
    match (a, b) {
        (Av::F(Fact::Range { lo: l1, hi: h1, int: i1 }), Av::F(Fact::Range { lo: l2, hi: h2, int: i2 })) => match op {
            BinOp::Add => range(l1 + l2, h1 + h2, i1 && i2),
            BinOp::Sub => range(l1 - h2, h1 - l2, i1 && i2),
            BinOp::Mul => {
                let c = [l1 * l2, l1 * h2, h1 * l2, h1 * h2];
                let lo = c.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                range(lo, hi, i1 && i2)
            }
            BinOp::Div if l2 > 0.0 || h2 < 0.0 => {
                let c = [l1 / l2, l1 / h2, h1 / l2, h1 / h2];
                let lo = c.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // rounding of the quotient: widen by one ulp-ish margin
                range(lo - lo.abs() * 1e-15, hi + hi.abs() * 1e-15, false)
            }
            _ => Av::Top,
        },
        // integers stay integers under + - * (a double sum, difference or
        // product of two integers is an integer, or not finite)
        _ if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) && a.whole() && b.whole() => Av::F(Fact::Whole),
        _ => Av::Top,
    }
}

/// The variables of one def (or of main): what each always holds.
#[derive(Default, Clone)]
pub(crate) struct FnFacts {
    pub vars: HashMap<String, Av>,
    pub types: HashMap<String, Ty>,
    pub places: Places,
}

#[derive(Default)]
pub(crate) struct ProgFacts {
    pub fns: HashMap<String, FnFacts>,
    pub main: FnFacts,
    /// The defs' return types, by C name.
    pub rets: HashMap<String, Ty>,
    /// The arrays each def (or main) only ever assigns `shape(...)` to.
    pub shapes: HashMap<String, HashSet<String>>,
}

/// One assignment of a variable: a value, or a loop counter with its bound.
#[derive(Clone, Copy)]
enum Def {
    Value(NodeId),
    Counter(NodeId),
}

fn walk(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<(String, Def)>, calls: &mut Vec<NodeId>) {
    let n = prog.get(id);
    match &n.kind {
        Viv::Assign { name } => out.push((name.clone(), Def::Value(n.children[0]))),
        Viv::For { var } | Viv::ParallelFor { var, .. } => out.push((var.clone(), Def::Counter(n.children[0]))),
        Viv::Call { .. } => calls.push(id),
        _ => {}
    }
    for &c in &n.children {
        walk(prog, c, out, calls);
    }
}

struct Env<'a> {
    prog: &'a IrProgram<Viv>,
    vars: &'a HashMap<String, Av>,
    types: &'a HashMap<String, Ty>,
    globals: Option<&'a FnFacts>,
    shapes: &'a HashSet<String>,
    rets: &'a HashMap<String, Ty>,
}

impl Env<'_> {
    fn var(&self, name: &str) -> Av {
        if let Some(v) = self.vars.get(name) {
            return *v;
        }
        if let Some(g) = self.globals {
            if let Some(v) = g.vars.get(name) {
                return *v;
            }
        }
        Av::Top
    }
    fn ty(&self, name: &str) -> Option<Ty> {
        self.types.get(name).copied().or_else(|| self.globals.and_then(|g| g.types.get(name).copied()))
    }

    fn eval(&self, id: NodeId) -> Av {
        let n = self.prog.get(id);
        let k = &n.children;
        match &n.kind {
            Viv::Num(c) if c.is_finite() => range(*c, *c, c.fract() == 0.0),
            Viv::Int(c) => range(*c as f64, *c as f64, true),
            Viv::Var { name } => self.var(name),
            Viv::Neg => match self.eval(k[0]) {
                Av::F(Fact::Range { lo, hi, int }) => range(-hi, -lo, int),
                v => v,
            },
            Viv::Bin { op, cplx: false } => arith(*op, self.eval(k[0]), self.eval(k[1])),
            Viv::Index { .. } if k.len() == 2 => {
                // an entry of shape(a): a dimension
                let shape = match &self.prog.get(k[0]).kind {
                    Viv::Var { name } => self.shapes.contains(name),
                    Viv::Call { callee } => callee == "viv_shape",
                    _ => false,
                };
                if shape {
                    range(0.0, DIM_MAX, true)
                } else {
                    Av::Top
                }
            }
            Viv::Call { callee } => self.call(callee, k),
            _ => Av::Top,
        }
    }

    fn call(&self, callee: &str, k: &[NodeId]) -> Av {
        let arg = |i: usize| k.get(i).map(|&c| self.eval(c)).unwrap_or(Av::Top);
        let round_by = |f: fn(f64) -> f64| match arg(0) {
            Av::Bot => Av::Bot,
            Av::F(Fact::Range { lo, hi, .. }) => range(f(lo), f(hi), true),
            _ => Av::F(Fact::Whole),
        };
        match callee {
            "round" => round_by(f64::round),
            "floor" => round_by(f64::floor),
            "ceil" => round_by(f64::ceil),
            "trunc" | "viv_num_int" => round_by(f64::trunc),
            "viv_floordiv" => match (arg(0), arg(1)) {
                (Av::Bot, _) | (_, Av::Bot) => Av::Bot,
                _ => Av::F(Fact::Whole),
            },
            "viv_len" | "viv_numel" | "viv_s_len" | "viv_ss_len" => range(0.0, DIM_MAX, true),
            "fmin" | "fmax" => {
                let (a, b) = (arg(0), arg(1));
                match (a, b) {
                    (Av::Bot, _) | (_, Av::Bot) => Av::Bot,
                    (Av::F(Fact::Range { lo: l1, hi: h1, int: i1 }), Av::F(Fact::Range { lo: l2, hi: h2, int: i2 })) => {
                        if callee == "fmin" {
                            range(l1.min(l2), h1.min(h2), i1 && i2)
                        } else {
                            range(l1.max(l2), h1.max(h2), i1 && i2)
                        }
                    }
                    _ if a.whole() && b.whole() => Av::F(Fact::Whole),
                    _ => Av::Top,
                }
            }
            "fabs" => match arg(0) {
                Av::F(Fact::Range { lo, hi, int }) => {
                    let (a, b) = (lo.abs(), hi.abs());
                    let lo2 = if lo <= 0.0 && hi >= 0.0 { 0.0 } else { a.min(b) };
                    range(lo2, a.max(b), int)
                }
                v => v,
            },
            // the Int operations: an Int is an integer; its range when known
            "viv_i_add" => int_of(arith(BinOp::Add, arg(0), arg(1))),
            "viv_i_sub" => int_of(arith(BinOp::Sub, arg(0), arg(1))),
            "viv_i_mul" => int_of(arith(BinOp::Mul, arg(0), arg(1))),
            "viv_i_neg" => int_of(match arg(0) {
                Av::F(Fact::Range { lo, hi, int }) => range(-hi, -lo, int),
                v => v,
            }),
            "viv_i_min" | "viv_i_max" => {
                let (a, b) = (arg(0), arg(1));
                int_of(match (a.bounds(), b.bounds()) {
                    (Some((l1, h1)), Some((l2, h2))) => {
                        if callee == "viv_i_min" {
                            range(l1.min(l2), h1.min(h2), true)
                        } else {
                            range(l1.max(l2), h1.max(h2), true)
                        }
                    }
                    _ => Av::Top,
                })
            }
            // a number converted to an Int is an integer (or the run stopped)
            "viv_i_of" => int_of(match arg(0) {
                Av::F(Fact::Range { lo, hi, .. }) => range(lo.ceil(), hi.floor(), true),
                v => v,
            }),
            // an Int converted to a number exactly (or the run stopped)
            "viv_i_num" | "viv_i_float" => int_of(arg(0)),
            c if super::ints::rt_fn(c).map_or(false, |f| matches!(f.ret, super::ints::IK::I)) => Av::F(Fact::Whole),
            "viv_iget" => Av::F(Fact::Whole),
            c => match self.rets.get(c) {
                Some(Ty::Int) => Av::F(Fact::Whole),
                _ => Av::Top,
            },
        }
    }
}

/// An Int's fact: always an integer, so `Top` (nothing known) becomes `Whole`.
fn int_of(v: Av) -> Av {
    match v {
        Av::Top => Av::F(Fact::Whole),
        Av::F(Fact::Range { lo, hi, .. }) => range(lo, hi, true),
        v => v,
    }
}

/// `v + 1`, `1 + v` or `v - 1` of the variable `v`: +1, -1, else 0.
fn unit_step(prog: &IrProgram<Viv>, v: &str, id: NodeId) -> i32 {
    let n = prog.get(id);
    let is_var = |c: NodeId| matches!(&prog.get(c).kind, Viv::Var { name } if name == v);
    let is_one = |c: NodeId| matches!(prog.get(c).kind, Viv::Num(x) if x == 1.0) || matches!(prog.get(c).kind, Viv::Int(1));
    match &n.kind {
        Viv::Call { callee } if callee == "viv_i_add" && ((is_var(n.children[0]) && is_one(n.children[1])) || (is_one(n.children[0]) && is_var(n.children[1]))) => 1,
        Viv::Call { callee } if callee == "viv_i_sub" && is_var(n.children[0]) && is_one(n.children[1]) => -1,
        Viv::Bin { op: BinOp::Add, cplx: false } if (is_var(n.children[0]) && is_one(n.children[1])) || (is_one(n.children[0]) && is_var(n.children[1])) => 1,
        Viv::Bin { op: BinOp::Sub, cplx: false } if is_var(n.children[0]) && is_one(n.children[1]) => -1,
        _ => 0,
    }
}

fn mentions(prog: &IrProgram<Viv>, v: &str, id: NodeId) -> bool {
    let n = prog.get(id);
    if matches!(&n.kind, Viv::Var { name } if name == v) {
        return true;
    }
    n.children.iter().any(|&c| mentions(prog, v, c))
}

/// The bound's upper end, as the highest counter value: `ceil(hi) - 1`, at
/// most 2^53 - 1 (a counter cannot pass that in a run: see `trip`).
/// An Int loop counts exactly, without the 2^53 stop, so its cap is how far
/// it can count in a run (`INT_STEPS`).
fn counter_range(bound: Av, int: bool) -> Av {
    let cap = if int { INT_STEPS } else { EXACT } - 1.0;
    match bound {
        Av::Bot => Av::Bot,
        Av::F(Fact::Range { hi, .. }) => range(0.0, (hi.ceil() - 1.0).clamp(0.0, cap), true),
        _ => range(0.0, cap, true),
    }
}

struct Body {
    defs: Vec<(String, Def)>,
    calls: Vec<NodeId>,
    /// Each definition with the index of the def's top-level statement it is
    /// in, and whether it is that statement itself.
    places: Vec<(String, Def, usize, bool)>,
}

fn body_of(prog: &IrProgram<Viv>, items: &[NodeId]) -> Body {
    let mut defs = Vec::new();
    let mut calls = Vec::new();
    let mut places = Vec::new();
    for (t, &s) in items.iter().enumerate() {
        let from = defs.len();
        walk(prog, s, &mut defs, &mut calls);
        for (k, (v, d)) in defs[from..].iter().enumerate() {
            let top = k == 0 && matches!(&prog.get(s).kind, Viv::Assign { name } if name == v);
            places.push((v.clone(), *d, t, top));
        }
    }
    Body { defs, calls, places }
}

/// Where a def's variables are fixed, for proving accesses in bounds: the
/// variables assigned once, by a top-level statement (with its index and
/// value), the parameters never assigned (index -1), and every assignment of
/// each array variable (its value and top-level index).
#[derive(Default, Clone)]
pub(crate) struct Places {
    pub stable: HashMap<String, (i64, Option<NodeId>)>,
    pub arrs: HashMap<String, Vec<(NodeId, usize)>>,
    /// The def's parameters: a parameter starts as its argument, not as 0.
    pub params: HashSet<String>,
}

fn places_of(body: &Body, params: &[(String, Ty)]) -> Places {
    let mut count: HashMap<&str, usize> = HashMap::new();
    for (v, ..) in &body.places {
        *count.entry(v.as_str()).or_default() += 1;
    }
    let mut out = Places::default();
    out.params = params.iter().map(|(p, _)| p.clone()).collect();
    for (p, _) in params {
        if !count.contains_key(p.as_str()) {
            out.stable.insert(p.clone(), (-1, None));
        }
    }
    for (v, d, t, top) in &body.places {
        if let (Def::Value(e), true, Some(1)) = (d, *top, count.get(v.as_str())) {
            out.stable.insert(v.clone(), (*t as i64, Some(*e)));
        }
        if let Def::Value(e) = d {
            out.arrs.entry(v.clone()).or_default().push((*e, *t));
        }
    }
    out
}

/// The local fixpoint of one def: each variable holds 0 first (every local is
/// zero-initialized) and then what its assignments give.
fn solve(prog: &IrProgram<Viv>, body: &Body, start: HashMap<String, Av>, types: &HashMap<String, Ty>, globals: Option<&FnFacts>, shapes: &HashSet<String>, rets: &HashMap<String, Ty>) -> HashMap<String, Av> {
    let mut vars = start;
    // the step variables: every assignment is `v + 1`, `v - 1`, or a value
    // that does not mention v
    let mut steps: HashMap<String, (bool, bool)> = HashMap::new();
    let mut not_step: HashSet<String> = HashSet::new();
    for (v, d) in &body.defs {
        if let Def::Value(e) = d {
            match unit_step(prog, v, *e) {
                1 => steps.entry(v.clone()).or_default().0 = true,
                -1 => steps.entry(v.clone()).or_default().1 = true,
                _ => {
                    if mentions(prog, v, *e) {
                        not_step.insert(v.clone());
                    }
                }
            }
        }
    }
    steps.retain(|v, _| !not_step.contains(v));
    for round in 0..40 {
        let mut next = vars.clone();
        let mut normal: HashMap<String, Av> = steps.keys().map(|v| (v.clone(), Av::Bot)).collect();
        {
            let env = Env { prog, vars: &vars, types, globals, shapes, rets };
            for (v, d) in &body.defs {
                if !next.contains_key(v) {
                    continue;
                }
                let val = match d {
                    Def::Counter(b) => counter_range(env.eval(*b), types.get(v) == Some(&Ty::Int)),
                    Def::Value(e) => {
                        if steps.contains_key(v) && unit_step(prog, v, *e) != 0 {
                            continue;
                        }
                        env.eval(*e)
                    }
                };
                let e = normal.entry(v.clone()).or_insert(Av::Bot);
                *e = e.join(val);
            }
        }
        for (v, val) in normal {
            let Some(&old) = next.get(&v) else { continue };
            let mut new = old.join(val);
            // a step variable: its other values, then any number of +-1 steps,
            // which stay integers and stay within +-2^53 when they start there
            // (an Int step is exact, so the bound for one is how far unit
            // steps can go in a run: 2^61 of them would take 73 years at
            // one a nanosecond)
            if let Some(&(up, down)) = steps.get(&v) {
                let far = if types.get(&v) == Some(&Ty::Int) { INT_STEPS } else { EXACT };
                new = match new {
                    Av::F(Fact::Range { lo, hi, int: true }) if lo >= -far && hi <= far => {
                        range(if down { -far } else { lo }, if up { far } else { hi }, true)
                    }
                    x => x.widen(),
                };
            }
            if round >= 6 && new != old {
                new = new.widen();
            }
            next.insert(v, new);
        }
        if next == vars {
            break;
        }
        vars = next;
    }
    vars
}

/// The facts of every def in the program and of main, with the parameters'
/// facts read from the call sites.
pub(crate) fn analyze(prog: &IrProgram<Viv>) -> ProgFacts {
    let mut funcs: Vec<(String, Vec<(String, Ty)>, Vec<(String, Ty)>, Vec<NodeId>)> = Vec::new();
    let mut main: Option<(Vec<(String, Ty)>, Vec<NodeId>, bool)> = None;
    let mut by_value: HashSet<String> = HashSet::new();
    let mut rets = HashMap::new();
    for &r in prog.roots() {
        let n = prog.get(r);
        match &n.kind {
            Viv::Func { name, params, locals, ret, .. } => {
                rets.insert(name.clone(), *ret);
                funcs.push((name.clone(), params.clone(), locals.clone(), n.children.clone()));
            }
            Viv::Main { locals, globals } => {
                let mut vars = locals.clone();
                vars.extend(globals.iter().cloned());
                let ckpt = has_ckpt(prog, &n.children);
                main = Some((vars, n.children.clone(), ckpt));
            }
            Viv::FnWrapDef { name, .. } => {
                by_value.insert(name.clone());
            }
            _ => {}
        }
    }
    // a def reached as a value (`apply`, a registry) gets arguments no call
    // site shows
    let mut all = Vec::new();
    for &r in prog.roots() {
        collect_kind(prog, r, &mut all);
    }
    for name in all {
        by_value.insert(name);
    }
    let bodies: HashMap<String, Body> = funcs.iter().map(|(n, _, _, b)| (n.clone(), body_of(prog, b))).collect();
    let main_body = main.as_ref().map(|(_, b, _)| body_of(prog, b));
    let mut out = ProgFacts { rets: rets.clone(), ..Default::default() };
    // main first: the defs read its globals
    let shapes_of = |body: &Body| -> HashSet<String> {
        let mut yes: HashSet<String> = HashSet::new();
        let mut no: HashSet<String> = HashSet::new();
        for (v, d) in &body.defs {
            match d {
                Def::Value(e) if matches!(&prog.get(*e).kind, Viv::Call { callee } if callee == "viv_shape") => {
                    yes.insert(v.clone());
                }
                _ => {
                    no.insert(v.clone());
                }
            }
        }
        yes.retain(|v| !no.contains(v));
        yes
    };
    if let (Some((vars, _, ckpt)), Some(body)) = (&main, &main_body) {
        let types: HashMap<String, Ty> = vars.iter().cloned().collect();
        let shapes = if *ckpt { HashSet::new() } else { shapes_of(body) };
        let start: HashMap<String, Av> = vars.iter().filter(|(_, t)| tracked(*t)).map(|(n, _)| (n.clone(), range(0.0, 0.0, true))).collect();
        // a checkpoint resume loads main's variables from a file: nothing is known
        let facts = if *ckpt { HashMap::new() } else { solve(prog, body, start, &types, None, &shapes, &rets) };
        let places = if *ckpt { Places::default() } else { places_of(body, &[]) };
        out.main = FnFacts { vars: facts, types, places };
        out.shapes.insert(crate::parser::MAIN_DEF.to_string(), shapes);
    }
    let mut params: HashMap<String, Vec<Av>> = funcs
        .iter()
        .map(|(n, ps, _, _)| (n.clone(), ps.iter().map(|(_, t)| if by_value.contains(n) || !tracked(*t) { Av::Top } else { Av::Bot }).collect()))
        .collect();
    for round in 0..30 {
        let mut fns = HashMap::new();
        for (name, ps, locals, _) in &funcs {
            let mut types: HashMap<String, Ty> = ps.iter().cloned().collect();
            types.extend(locals.iter().cloned());
            let mut start: HashMap<String, Av> = locals.iter().filter(|(_, t)| tracked(*t)).map(|(n, _)| (n.clone(), range(0.0, 0.0, true))).collect();
            for (j, (p, t)) in ps.iter().enumerate() {
                if tracked(*t) {
                    let v = params[name][j];
                    start.insert(p.clone(), if *t == Ty::Int { int_of(v) } else { v });
                }
            }
            let body = &bodies[name];
            let shapes = shapes_of(body);
            let vars = solve(prog, body, start, &types, Some(&out.main), &shapes, &rets);
            out.shapes.insert(name.clone(), shapes);
            fns.insert(name.clone(), FnFacts { vars, types, places: places_of(body, ps) });
        }
        // the parameters: what every call site passes
        let mut next: HashMap<String, Vec<Av>> = params.iter().map(|(n, v)| (n.clone(), v.iter().map(|a| if *a == Av::Top { Av::Top } else { Av::Bot }).collect())).collect();
        let mut sites: Vec<(&FnFacts, &HashSet<String>, &Body)> = Vec::new();
        for (name, ..) in &funcs {
            sites.push((&fns[name], &out.shapes[name], &bodies[name]));
        }
        let main_shapes = out.shapes.get(crate::parser::MAIN_DEF).cloned().unwrap_or_default();
        if let Some(b) = &main_body {
            sites.push((&out.main, &main_shapes, b));
        }
        for (f, shapes, body) in sites {
            let env = Env { prog, vars: &f.vars, types: &f.types, globals: Some(&out.main), shapes, rets: &rets };
            for &c in &body.calls {
                let n = prog.get(c);
                let Viv::Call { callee } = &n.kind else { continue };
                let Some(slots) = next.get_mut(callee) else { continue };
                for (j, &a) in n.children.iter().enumerate() {
                    if j < slots.len() && slots[j] != Av::Top {
                        slots[j] = slots[j].join(env.eval(a));
                    }
                }
            }
        }
        // a def's own params changing across rounds: widen after a few
        if round >= 4 {
            for (n, v) in next.iter_mut() {
                for (j, a) in v.iter_mut().enumerate() {
                    if *a != params[n][j] {
                        *a = a.join(params[n][j]).widen();
                    }
                }
            }
        }
        let done = next == params;
        params = next;
        out.fns = fns;
        if done {
            break;
        }
    }
    out
}

fn collect_kind(prog: &IrProgram<Viv>, id: NodeId, out: &mut Vec<String>) {
    let n = prog.get(id);
    if let Viv::FnRef { name, .. } = &n.kind {
        out.push(name.clone());
    }
    for &c in &n.children {
        collect_kind(prog, c, out);
    }
}

fn has_ckpt(prog: &IrProgram<Viv>, items: &[NodeId]) -> bool {
    items.iter().any(|&c| {
        let n = prog.get(c);
        matches!(n.kind, Viv::Ckpt) || n.children.iter().any(|&g| matches!(prog.get(g).kind, Viv::Ckpt))
    })
}

/// The scalar types whose values this module follows.
fn tracked(t: Ty) -> bool {
    matches!(t, Ty::Num | Ty::Int)
}

// ---- emission ------------------------------------------------------------

/// A `for` loop being emitted: its variable, the C name of its counter, and
/// whether the body leaves the variable alone (then the variable equals the
/// counter everywhere in the body).
pub(crate) struct LoopCx {
    pub var: String,
    pub counter: String,
    pub trip: String,
    pub unassigned: bool,
    /// The highest counter value the bound allows, when known from facts.
    pub hi: f64,
    /// `counter < sym + off`, when the bound is known symbolically.
    pub ub: Option<(Sym, i64)>,
    /// The loop's body.
    pub body: NodeId,
    /// The arrays this pass of emission may take as covering the loop: the
    /// loop's checked-once version runs only after testing, before its first
    /// pass, that each is rank 1 with at least as many elements as the loop
    /// has passes (`version_arrays`), so a read or write of one at exactly
    /// the counter is inside it.
    pub assumed: Vec<String>,
}

/// A value that is fixed while a def runs past a point: a variable assigned
/// once (or a parameter never assigned), a dimension of an array whose
/// dimensions are fixed, or a whole number.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Sym {
    Var(String),
    Dim(String, usize),
    Lit(i64),
}

#[derive(Default)]
pub(crate) struct EmitState {
    pub key: (usize, usize),
    pub roots: Vec<NodeId>,
    pub facts: std::rc::Rc<ProgFacts>,
    /// The def being emitted: its facts, or None (a lambda, a wrapper).
    pub cur: Option<String>,
    pub loops: Vec<LoopCx>,
    /// The index of the def's top-level statement being emitted.
    pub top: usize,
}

thread_local! {
    pub(crate) static STATE: std::cell::RefCell<EmitState> = std::cell::RefCell::new(EmitState::default());
}

/// Enter a def (`Some(name)`), main (`Some(MAIN_DEF)`) or another item (None)
/// for emission; the program's facts are computed once per program.
pub(crate) fn enter(prog: &IrProgram<Viv>, cur: Option<String>) {
    // a program is known by its address, size and roots: two programs made
    // one after the other can share an address and a size
    let key = (prog as *const IrProgram<Viv> as usize, prog.len());
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if s.key != key || s.roots.as_slice() != prog.roots() {
            s.key = key;
            s.roots = prog.roots().to_vec();
            s.facts = std::rc::Rc::new(analyze(prog));
        }
        s.cur = cur;
        s.loops.clear();
        s.top = 0;
    });
}

/// The def's top-level statement `t` is being emitted.
pub(crate) fn at_top(t: usize) {
    STATE.with(|s| s.borrow_mut().top = t);
}

fn with_fn<T>(f: impl FnOnce(&FnFacts, usize) -> Option<T>) -> Option<T> {
    STATE.with(|s| {
        let s = s.borrow();
        let cur = s.cur.as_ref()?;
        let facts = &s.facts;
        let ff = if cur == crate::parser::MAIN_DEF { &facts.main } else { facts.fns.get(cur)? };
        f(ff, s.top)
    })
}

/// `f` run on the scope `name` belongs to at this point: the def's own
/// variables, or, for a global a def reads, main's, as of the end of main
/// (`at` past every statement). A def cannot assign a global, so a global
/// that main assigns once keeps that value in every def; a def that runs
/// before main assigned it reads the zero every variable starts as, which
/// gives a loop it bounds no pass and an array it names rank 0 (refused by
/// the rank check every access keeps). Main's names carry `main.` in a
/// `Sym`, so they never match a def's own.
fn in_scope<T>(name: &str, f: impl FnOnce(&FnFacts, i64, &str) -> Option<T>) -> Option<T> {
    STATE.with(|s| {
        let s = s.borrow();
        let cur = s.cur.as_ref()?;
        let facts = &s.facts;
        if cur == crate::parser::MAIN_DEF {
            return f(&facts.main, s.top as i64, "");
        }
        let ff = facts.fns.get(cur)?;
        if ff.types.contains_key(name) {
            f(ff, s.top as i64, "")
        } else if facts.main.types.contains_key(name) {
            f(&facts.main, i64::MAX, "main.")
        } else {
            None
        }
    })
}

/// A variable's fixed value as a `Sym`, when it is assigned before top-level
/// statement `at` and never again: through copies, `len(a)` and `shape(a)[k]`
/// to the dimension they read.
fn canon(prog: &IrProgram<Viv>, ff: &FnFacts, name: &str, at: i64, depth: u32, pre: &str) -> Option<Sym> {
    let &(pos, value) = ff.places.stable.get(name)?;
    if pos >= at || depth > 8 {
        return None;
    }
    // a fraction would make `zeros` and a loop disagree on the count
    if !ff.vars.get(name).map_or(false, |a| a.whole()) {
        return None;
    }
    let Some(v) = value else { return Some(Sym::Var(format!("{pre}{name}"))) };
    let n = prog.get(v);
    let k = &n.children;
    let read = match &n.kind {
        Viv::Var { name: m } => canon(prog, ff, m, pos, depth + 1, pre),
        Viv::Call { callee } if callee == "viv_i_of" => match &prog.get(k[0]).kind {
            Viv::Var { name: m } => canon(prog, ff, m, pos, depth + 1, pre),
            Viv::Call { callee } if callee == "viv_len" => len_sym(prog, ff, k[0], pos, depth, pre),
            _ => None,
        },
        Viv::Call { callee } if callee == "viv_len" => len_sym(prog, ff, v, pos, depth, pre),
        Viv::Index { .. } if k.len() == 2 => match (&prog.get(k[0]).kind, &prog.get(k[1]).kind) {
            (Viv::Var { name: sh }, Viv::Num(d)) if *d >= 0.0 && d.fract() == 0.0 => {
                let &(spos, sval) = ff.places.stable.get(sh)?;
                if spos >= pos {
                    return Some(Sym::Var(format!("{pre}{name}")));
                }
                match sval.map(|x| (&prog.get(x).kind, prog.get(x).children.clone())) {
                    Some((Viv::Call { callee }, kk)) if callee == "viv_shape" => match &prog.get(kk[0]).kind {
                        Viv::Var { name: a } => dim(prog, ff, a, *d as usize, spos, depth + 1, pre),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    };
    Some(read.unwrap_or_else(|| Sym::Var(format!("{pre}{name}"))))
}

fn len_sym(prog: &IrProgram<Viv>, ff: &FnFacts, call: NodeId, at: i64, depth: u32, pre: &str) -> Option<Sym> {
    match &prog.get(prog.get(call).children[0]).kind {
        Viv::Var { name: a } => dim(prog, ff, a, 0, at, depth + 1, pre),
        _ => None,
    }
}

/// Dimension `d` of the array variable `a` at any point past top-level
/// statement `at`: a parameter never assigned keeps its dimensions; a local
/// whose every assignment is `zeros(n1, ..)` of the same fixed values,
/// assigned after them, has those.
fn dim(prog: &IrProgram<Viv>, ff: &FnFacts, a: &str, d: usize, at: i64, depth: u32, pre: &str) -> Option<Sym> {
    // a number array's or an Int array's (int_zeros is a Zeros node too)
    if !matches!(ff.types.get(a), Some(Ty::Arr(super::Elem::Num | super::Elem::Int))) || depth > 8 {
        return None;
    }
    if let Some(&(-1, None)) = ff.places.stable.get(a) {
        return Some(Sym::Dim(format!("{pre}{a}"), d));
    }
    // a parameter the def assigns held its argument, of unknown dimensions,
    // until then
    if ff.places.params.contains(a) {
        return None;
    }
    let defs = ff.places.arrs.get(a)?;
    let mut out: Option<Sym> = None;
    for &(v, t) in defs {
        let n = prog.get(v);
        if !matches!(n.kind, Viv::Zeros) || d >= n.children.len() {
            return None;
        }
        let sym = match &prog.get(n.children[d]).kind {
            Viv::Num(c) if *c >= 0.0 && c.fract() == 0.0 && *c <= DIM_MAX => Sym::Lit(*c as i64),
            Viv::Var { name } => canon(prog, ff, name, t as i64, depth + 1, pre)?,
            _ => return None,
        };
        match &out {
            None => out = Some(sym),
            Some(o) if *o == sym => {}
            _ => return None,
        }
    }
    let _ = at;
    out
}

/// An upper end of a loop bound: `bound <= sym + off`.
pub(crate) fn upper(prog: &IrProgram<Viv>, id: NodeId) -> Option<(Sym, i64)> {
    let depth = STATE.with(|s| s.borrow().loops.len());
    upper_in(prog, id, depth, 0)
}

/// How many times `var` is assigned in the tree under `id`.
fn count_assigns(prog: &IrProgram<Viv>, id: NodeId, var: &str) -> usize {
    let n = prog.get(id);
    let here = matches!(&n.kind, Viv::Assign { name } | Viv::For { var: name } | Viv::ParallelFor { var: name, .. } if name == var) as usize;
    here + n.children.iter().map(|&c| count_assigns(prog, c, var)).sum::<usize>()
}

/// `upper`, reading only the loops `loops[..depth]`: the ones around the
/// point the expression is evaluated at.
fn upper_in(prog: &IrProgram<Viv>, id: NodeId, depth: usize, nest: u32) -> Option<(Sym, i64)> {
    if nest > 6 {
        return None;
    }
    let upper = |prog: &IrProgram<Viv>, id: NodeId| upper_in(prog, id, depth, nest + 1);
    let n = prog.get(id);
    let k = &n.children;
    let lit = |c: NodeId| match prog.get(c).kind {
        Viv::Num(x) if x.fract() == 0.0 && x.abs() <= DIM_MAX => Some(x as i64),
        Viv::Int(x) if x.abs() <= DIM_MAX as i64 => Some(x),
        _ => None,
    };
    match &n.kind {
        Viv::Num(_) | Viv::Int(_) => Some((Sym::Lit(lit(id)?), 0)),
        Viv::Var { name } => {
            let ctr = STATE.with(|s| s.borrow().loops[..depth].iter().rev().find(|l| &l.var == name).map(|l| (l.unassigned, l.ub.clone())));
            if let Some((true, ub)) = ctr {
                let (sym, off) = ub?;
                return Some((sym, off - 1));
            }
            if ctr.is_some() {
                return None;
            }
            if let Some(s) = in_scope(name, |ff, at, pre| canon(prog, ff, name, at, 0, pre)) {
                return Some((s, 0));
            }
            // a variable assigned only inside the body of a loop around this
            // point: each value it was given there is bounded with the loops
            // around that loop (its value at 0, before any assignment, gives
            // no pass to a loop it bounds)
            let total = with_fn(|ff, _| Some(ff.places.arrs.get(name).map_or(0, |v| v.len())))?;
            let (count_ok, inner) = STATE.with(|s| {
                let s = s.borrow();
                for (li, l) in s.loops[..depth].iter().enumerate().rev() {
                    if count_assigns(prog, l.body, name) == total {
                        return (true, li + 1);
                    }
                }
                (false, 0)
            });
            if !count_ok || total == 0 {
                return None;
            }
            let body = STATE.with(|s| s.borrow().loops[inner - 1].body);
            let mut values = Vec::new();
            collect_values(prog, body, name, &mut values);
            let mut out: Option<(Sym, i64)> = None;
            for v in values {
                let (sym, off) = upper_in(prog, v, inner, nest + 1)?;
                out = match out {
                    None => Some((sym, off)),
                    Some((s0, o0)) if s0 == sym => Some((s0, o0.max(off))),
                    _ => return None,
                };
            }
            out
        }
        Viv::Bin { op: BinOp::Add, cplx: false } => match (lit(k[0]), lit(k[1])) {
            (_, Some(c)) => upper(prog, k[0]).map(|(s, o)| (s, o + c)),
            (Some(c), _) => upper(prog, k[1]).map(|(s, o)| (s, o + c)),
            _ => None,
        },
        Viv::Bin { op: BinOp::Sub, cplx: false } => {
            let c = lit(k[1])?;
            upper(prog, k[0]).map(|(s, o)| (s, o - c))
        }
        Viv::Call { callee } if callee == "viv_i_add" => match (lit(k[0]), lit(k[1])) {
            (_, Some(c)) => upper(prog, k[0]).map(|(s, o)| (s, o + c)),
            (Some(c), _) => upper(prog, k[1]).map(|(s, o)| (s, o + c)),
            _ => None,
        },
        Viv::Call { callee } if callee == "viv_i_of" => upper(prog, k[0]),
        Viv::Call { callee } if callee == "viv_len" => match &prog.get(k[0]).kind {
            Viv::Var { name } => in_scope(name, |ff, at, pre| dim(prog, ff, name, 0, at, 0, pre)).map(|s| (s, 0)),
            _ => None,
        },
        Viv::Call { callee } if callee == "fmin" || callee == "viv_i_min" => upper(prog, k[0]).or_else(|| upper(prog, k[1])),
        _ => None,
    }
}

fn collect_values(prog: &IrProgram<Viv>, id: NodeId, var: &str, out: &mut Vec<NodeId>) {
    let n = prog.get(id);
    if let Viv::Assign { name } = &n.kind {
        if name == var {
            out.push(n.children[0]);
        }
    }
    for &c in &n.children {
        collect_values(prog, c, var, out);
    }
}

/// The fixed dimension `d` of the array `base` reads, at this point of the def.
pub(crate) fn base_dim(prog: &IrProgram<Viv>, base: NodeId, d: usize) -> Option<Sym> {
    match &prog.get(base).kind {
        Viv::Var { name } => in_scope(name, |ff, at, pre| dim(prog, ff, name, d, at, 0, pre)),
        _ => None,
    }
}

/// The fact of an expression in the def being emitted.
pub(crate) fn fact(prog: &IrProgram<Viv>, id: NodeId) -> Av {
    STATE.with(|s| {
        let s = s.borrow();
        let Some(cur) = &s.cur else { return Av::Top };
        let facts = &s.facts;
        let is_main = cur == crate::parser::MAIN_DEF;
        let f = if is_main { &facts.main } else {
            match facts.fns.get(cur) {
                Some(f) => f,
                None => return Av::Top,
            }
        };
        let empty = HashSet::new();
        let shapes = facts.shapes.get(cur).unwrap_or(&empty);
        let env = Env { prog, vars: &f.vars, types: &f.types, globals: if is_main { None } else { Some(&facts.main) }, shapes, rets: &facts.rets };
        env.eval(id)
    })
}

fn var_ty(name: &str) -> Option<Ty> {
    STATE.with(|s| {
        let s = s.borrow();
        let cur = s.cur.as_ref()?;
        let facts = &s.facts;
        if cur == crate::parser::MAIN_DEF {
            return facts.main.types.get(name).copied();
        }
        let f = facts.fns.get(cur)?;
        f.types.get(name).copied().or_else(|| facts.main.types.get(name).copied())
    })
}

/// Whether `var` is assigned anywhere in `body` (a `for` over it counts).
pub(crate) fn assigns(prog: &IrProgram<Viv>, body: NodeId, var: &str) -> bool {
    let n = prog.get(body);
    match &n.kind {
        Viv::Assign { name } | Viv::For { var: name } | Viv::ParallelFor { var: name, .. } if name == var => return true,
        _ => {}
    }
    n.children.iter().any(|&c| assigns(prog, c, var))
}

/// Whether `base[idx]` is an access at exactly the counter of a loop being
/// emitted in its checked-once version, to an array that version tested
/// covers every pass (`LoopCx::assumed`).
pub(crate) fn assumed(prog: &IrProgram<Viv>, base: NodeId, idx: NodeId) -> bool {
    let (Viv::Var { name: a }, Viv::Var { name: v }) = (&prog.get(base).kind, &prog.get(idx).kind) else { return false };
    STATE.with(|s| {
        let s = s.borrow();
        match s.loops.iter().rev().find(|l| &l.var == v) {
            Some(l) => l.unassigned && l.assumed.iter().any(|x| x == a),
            None => false,
        }
    })
}

/// Whether `base[idx]` can be read with no check failing: `idx` is the
/// counter of a loop being emitted that never assigns it, and the array is
/// one that loop's version tested (`assumed`, rank included), or an Int
/// array (always rank 1) whose fixed length is the loop's bound.
fn ctr_read_proved(prog: &IrProgram<Viv>, base: NodeId, idx: NodeId, int_arr: bool) -> bool {
    let (Viv::Var { name: a }, Viv::Var { name: v }) = (&prog.get(base).kind, &prog.get(idx).kind) else { return false };
    let lp = STATE.with(|s| s.borrow().loops.iter().rev().find(|l| &l.var == v).map(|l| (l.unassigned, l.assumed.iter().any(|x| x == a), l.ub.clone())));
    let Some((true, assumed, ub)) = lp else { return false };
    if assumed {
        return true;
    }
    if !int_arr {
        return false;
    }
    let d = in_scope(a, |ff, at, pre| dim(prog, ff, a, 0, at, 0, pre));
    matches!((d, ub), (Some(d), Some((sym, off))) if d == sym && off <= 0)
}

/// Whether `base[idx]` is inside `base`: `idx` is the never-assigned
/// counter of a loop being emitted, and the array is one that loop's
/// version tested, or its fixed length is the loop's bound. Says nothing of
/// the rank, which the accessor still checks.
pub(crate) fn ctr_access_proved(prog: &IrProgram<Viv>, base: NodeId, idx: NodeId) -> bool {
    ctr_read_proved(prog, base, idx, true)
}

/// A value whose evaluation cannot stop the program: a literal, a variable,
/// number arithmetic (`+ - *`, which never traps) of such values, or an
/// array read `ctr_read_proved` clears.
fn cannot_fail(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = prog.get(id);
    let k = &n.children;
    match &n.kind {
        Viv::Num(_) | Viv::Int(_) | Viv::Var { .. } => true,
        Viv::Bin { op: BinOp::Add | BinOp::Sub | BinOp::Mul, cplx: false } => cannot_fail(prog, k[0]) && cannot_fail(prog, k[1]),
        Viv::Index { .. } if k.len() == 2 => ctr_read_proved(prog, k[0], k[1], false),
        Viv::Call { callee } if callee == "viv_iget" && k.len() == 2 => ctr_read_proved(prog, k[0], k[1], true),
        _ => false,
    }
}

/// Whether `a and b` (the `And` node `id`) may evaluate both sides: every
/// leaf is a comparison of values that cannot fail (`cannot_fail`), joined
/// by `and`. Then `&` gives what `&&` gives -- nothing either side does can
/// be seen -- with no branch per comparison, which lets a filter loop be
/// vectorized (study/dataframe, TPC-H Q6).
pub(crate) fn branch_free_and(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = prog.get(id);
    let k = &n.children;
    match &n.kind {
        Viv::Bin { op: BinOp::And, cplx: false } => branch_free_and(prog, k[0]) && branch_free_and(prog, k[1]),
        Viv::Bin { op: BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::EqEq | BinOp::Ne, cplx: false } => cannot_fail(prog, k[0]) && cannot_fail(prog, k[1]),
        _ => false,
    }
}

fn has_loop(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = prog.get(id);
    if matches!(n.kind, Viv::For { .. } | Viv::ParallelFor { .. } | Viv::While) {
        return true;
    }
    n.children.iter().any(|&c| has_loop(prog, c))
}

fn ctr_accesses(prog: &IrProgram<Viv>, id: NodeId, var: &str, out: &mut Vec<(String, NodeId)>) {
    let n = prog.get(id);
    let k = &n.children;
    let at_ctr = |b: NodeId, i: NodeId| match (&prog.get(b).kind, &prog.get(i).kind) {
        (Viv::Var { name: a }, Viv::Var { name: v }) if v == var => Some((a.clone(), b)),
        _ => None,
    };
    let hit = match &n.kind {
        Viv::Index { .. } if k.len() == 2 => at_ctr(k[0], k[1]),
        Viv::SetIndex { .. } if k.len() == 3 => at_ctr(k[0], k[1]),
        Viv::Call { callee } if (callee == "viv_iget" && k.len() == 2) || (callee == "viv_iset" && k.len() == 3) => at_ctr(k[0], k[1]),
        _ => None,
    };
    if let Some(h) = hit {
        if !out.iter().any(|(a, _)| *a == h.0) {
            out.push(h);
        }
    }
    for &c in k {
        ctr_accesses(prog, c, var, out);
    }
}

/// Loop versioning: the arrays an innermost `for var` loop (counter never
/// assigned, `ub` its bound) reads or writes at exactly its counter whose
/// dimension is not already proved to cover it -- a def's own number or Int
/// array that the body never assigns. Each element is the array's name and
/// a node naming it. When there are any, lower.rs emits the loop twice: a
/// version run after testing once that every one of them is rank 1 with at
/// least as many elements as the loop has passes, where those accesses carry
/// no check (`assumed`), and the loop as it was otherwise. Both run the same
/// passes in the same order; only checks that cannot fail are gone.
pub(crate) fn version_arrays(prog: &IrProgram<Viv>, var: &str, body: NodeId, ub: &Option<(Sym, i64)>) -> Vec<(String, NodeId)> {
    if has_loop(prog, body) {
        return Vec::new();
    }
    let mut out = Vec::new();
    ctr_accesses(prog, body, var, &mut out);
    out.retain(|(a, _)| {
        let own = with_fn(|ff, _| Some(matches!(ff.types.get(a.as_str()), Some(Ty::Arr(super::Elem::Num | super::Elem::Int))))).unwrap_or(false);
        if !own || assigns(prog, body, a) {
            return false;
        }
        // already proved without a test
        let d = in_scope(a, |ff, at, pre| dim(prog, ff, a, 0, at, 0, pre));
        !matches!((d, ub), (Some(d), Some((sym, off))) if d == *sym && *off <= 0)
    });
    out
}

pub(crate) fn push_loop(l: LoopCx) {
    STATE.with(|s| s.borrow_mut().loops.push(l));
}
pub(crate) fn pop_loop() {
    STATE.with(|s| {
        s.borrow_mut().loops.pop();
    });
}

/// An index computed in 64-bit integers: its leaves and operations.
#[derive(Clone, Debug)]
pub(crate) enum IForm {
    Lit(i64),
    /// The counter of an enclosing loop, whose variable the body leaves
    /// alone; `trip` names its pass count; `hi` the highest value known.
    Ctr { counter: String, trip: String, hi: f64, ub: Option<(Sym, i64)> },
    /// A number variable that always holds an integer: `(long long)v`.
    Cast { var: String, fact: Av },
    /// An expression of type Int, as it is; `index < sym + off` when it is
    /// the counter of an enclosing loop.
    Int(NodeId, Av, Option<(Sym, i64)>),
    Bin(BinOp, Box<IForm>, Box<IForm>),
    Neg(Box<IForm>),
}

pub(crate) fn is_int_node(prog: &IrProgram<Viv>, id: NodeId) -> bool {
    match prog.get(id).kind.clone() {
        Viv::Int(_) => true,
        Viv::Var { name } => var_ty(&name) == Some(Ty::Int),
        Viv::Call { callee } => {
            if callee == "viv_iget" {
                return true;
            }
            if let Some(f) = super::ints::rt_fn(&callee) {
                return matches!(f.ret, super::ints::IK::I);
            }
            STATE.with(|s| s.borrow().facts.rets.get(&callee) == Some(&Ty::Int))
        }
        _ => false,
    }
}

/// The integer form of an index expression, if it has one.
pub(crate) fn form(prog: &IrProgram<Viv>, id: NodeId) -> Option<IForm> {
    let n = prog.get(id);
    if let Viv::Var { name } = &n.kind {
        let ctr = STATE.with(|s| {
            let s = s.borrow();
            s.loops.iter().rev().find(|l| &l.var == name).map(|l| (l.unassigned, l.counter.clone(), l.trip.clone(), l.hi, l.ub.clone()))
        });
        if let Some((true, counter, trip, hi, ub)) = ctr {
            // an Int counter is exact at any size: it is used as it is
            if var_ty(name) == Some(Ty::Int) {
                return Some(IForm::Int(id, range(0.0, hi.max(0.0), true), ub));
            }
            return Some(IForm::Ctr { counter, trip, hi, ub });
        }
    }
    if is_int_node(prog, id) {
        return Some(IForm::Int(id, fact(prog, id), None));
    }
    match &n.kind {
        Viv::Num(c) if c.fract() == 0.0 && c.abs() <= EXACT => Some(IForm::Lit(*c as i64)),
        Viv::Var { name } => {
            // a number variable: proved an integer, or tested at run time
            if var_ty(name) == Some(Ty::Num) {
                return Some(IForm::Cast { var: name.clone(), fact: fact(prog, id) });
            }
            None
        }
        Viv::Bin { op: op @ (BinOp::Add | BinOp::Sub | BinOp::Mul), cplx: false } => {
            let a = form(prog, n.children[0])?;
            let b = form(prog, n.children[1])?;
            // Int arithmetic is a call, never a Bin: these are numbers
            if matches!(a, IForm::Int(..)) || matches!(b, IForm::Int(..)) {
                return None;
            }
            Some(IForm::Bin(*op, Box::new(a), Box::new(b)))
        }
        Viv::Neg => {
            let a = form(prog, n.children[0])?;
            if matches!(a, IForm::Int(..)) {
                return None;
            }
            Some(IForm::Neg(Box::new(a)))
        }
        _ => None,
    }
}

/// The exact integer form of an index that needs no run-time guard (the
/// direct wasm backend's: it has no guarded path).
pub(crate) fn exact_form(prog: &IrProgram<Viv>, id: NodeId) -> Option<IForm> {
    let f = form(prog, id)?;
    if matches!(f, IForm::Int(..)) {
        return Some(f);
    }
    let mut guards = Vec::new();
    bound(&f, EXACT, &mut guards)?;
    if guards.is_empty() {
        Some(f)
    } else {
        None
    }
}

/// The range of a form when every guarded leaf is at most `g` in magnitude,
/// or None when some intermediate value could pass 2^53. Pushes each guard
/// the bound needs.
fn bound(f: &IForm, g: f64, guards: &mut Vec<Guard>) -> Option<(f64, f64)> {
    let ok = |lo: f64, hi: f64| if lo >= -EXACT && hi <= EXACT { Some((lo, hi)) } else { None };
    match f {
        IForm::Lit(c) => Some((*c as f64, *c as f64)),
        IForm::Ctr { trip, hi, .. } => {
            if *hi <= g || trip.is_empty() {
                Some((0.0, *hi))
            } else {
                guards.push(Guard::Trip(trip.clone(), g));
                Some((0.0, g))
            }
        }
        IForm::Cast { var, fact } => match fact.bounds() {
            Some((lo, hi)) if fact.whole() && lo >= -g && hi <= g => Some((lo, hi)),
            _ if fact.whole() => {
                guards.push(Guard::Abs(var.clone(), g));
                Some((-g, g))
            }
            _ => {
                guards.push(Guard::Whole(var.clone(), g));
                Some((-g, g))
            }
        },
        IForm::Int(..) => None,
        IForm::Neg(a) => {
            let (lo, hi) = bound(a, g, guards)?;
            ok(-hi, -lo)
        }
        IForm::Bin(op, a, b) => {
            let (l1, h1) = bound(a, g, guards)?;
            let (l2, h2) = bound(b, g, guards)?;
            match op {
                BinOp::Add => ok(l1 + l2, h1 + h2),
                BinOp::Sub => ok(l1 - h2, h1 - l2),
                _ => {
                    let c = [l1 * l2, l1 * h2, h1 * l2, h1 * h2];
                    ok(c.iter().cloned().fold(f64::INFINITY, f64::min), c.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Guard {
    /// the loop's pass count is at most this
    Trip(String, f64),
    /// the variable's magnitude is at most this (false for NaN)
    Abs(String, f64),
    /// the variable is an integer of magnitude at most this
    Whole(String, f64),
}

/// One index of an access, ready to emit: the integer C expression, the
/// guards it needs, and its lowest possible value (0 or more: no index from
/// the end).
pub(crate) struct Ready {
    pub c: CExpr,
    pub guards: Vec<Guard>,
    pub lo: f64,
    /// `index < sym + off`, when known
    pub ub: Option<(Sym, i64)>,
}

/// `index < sym + off` of a form: a counter, or a counter plus a whole number.
fn form_ub(f: &IForm) -> Option<(Sym, i64)> {
    match f {
        IForm::Ctr { ub, .. } => ub.clone(),
        IForm::Bin(BinOp::Add, a, b) => match (&**a, &**b) {
            (x, IForm::Lit(c)) | (IForm::Lit(c), x) => form_ub(x).map(|(s, o)| (s, o + c)),
            _ => None,
        },
        IForm::Bin(BinOp::Sub, a, b) => match &**b {
            IForm::Lit(c) => form_ub(a).map(|(s, o)| (s, o - c)),
            _ => None,
        },
        _ => None,
    }
}

/// The index `id` as a 64-bit integer, when it has an exact integer form.
pub(crate) fn ready(cx: &mut CEmitCx<Viv>, id: NodeId) -> Option<Ready> {
    let f = form(cx.program(), id)?;
    if let IForm::Int(i, fact, ub) = &f {
        let lo = fact.bounds().map_or(f64::NEG_INFINITY, |b| b.0);
        return Some(Ready { c: cx.expr(*i), guards: Vec::new(), lo, ub: ub.clone() });
    }
    for g in [2147483648.0f64, 67108864.0, 2097152.0, 65536.0] {
        let mut guards = Vec::new();
        if let Some((lo, _)) = bound(&f, g, &mut guards) {
            guards.dedup();
            let ub = form_ub(&f);
            return Some(Ready { c: render(cx, id, &f), guards, lo, ub });
        }
    }
    None
}

/// `a < b` (or another comparison) of two numbers that both have an exact
/// integer form with no guard: compared as 64-bit integers, which is the
/// same comparison. None for anything else, including two Ints (already
/// integers in C) and a constant on both sides.
pub(crate) fn int_compare(cx: &mut CEmitCx<Viv>, id: NodeId, op: &str, a: NodeId, b: NodeId) -> Option<CExpr> {
    let fa = form(cx.program(), a)?;
    let fb = form(cx.program(), b)?;
    if matches!(fa, IForm::Int(..)) || matches!(fb, IForm::Int(..)) {
        return None;
    }
    if !has_ctr(&fa) && !has_ctr(&fb) {
        return None;
    }
    let mut guards = Vec::new();
    bound(&fa, EXACT, &mut guards)?;
    bound(&fb, EXACT, &mut guards)?;
    if !guards.is_empty() {
        return None;
    }
    let l = render(cx, id, &fa);
    let r = render(cx, id, &fb);
    Some(CExpr::binary(id, l, op, r))
}

fn has_ctr(f: &IForm) -> bool {
    match f {
        IForm::Ctr { .. } => true,
        IForm::Bin(_, a, b) => has_ctr(a) || has_ctr(b),
        IForm::Neg(a) => has_ctr(a),
        _ => false,
    }
}

fn render(cx: &mut CEmitCx<Viv>, id: NodeId, f: &IForm) -> CExpr {
    match f {
        IForm::Lit(c) => CExpr::lit(id, format!("{c}LL")),
        IForm::Ctr { counter, .. } => CExpr::var(id, counter.clone()),
        IForm::Cast { var, .. } => CExpr::unary(id, "(long long)", CExpr::var(id, var.clone())),
        IForm::Int(i, ..) => cx.expr(*i),
        IForm::Neg(a) => CExpr::unary(id, "-", render(cx, id, a)),
        IForm::Bin(op, a, b) => {
            let l = render(cx, id, a);
            let r = render(cx, id, b);
            CExpr::binary(id, l, op.c_op(), r)
        }
    }
}

/// The C test of a list of guards, or None when there are none.
pub(crate) fn guard_c(id: NodeId, guards: &[Guard]) -> Option<CExpr> {
    let mut seen: Vec<&Guard> = Vec::new();
    let mut test: Option<CExpr> = None;
    for g in guards {
        if seen.contains(&g) {
            continue;
        }
        seen.push(g);
        let part = match g {
            Guard::Trip(t, g) => CExpr::binary(id, CExpr::var(id, t.clone()), "<=", CExpr::lit(id, format!("{}LL", *g as i64))),
            Guard::Abs(v, g) => CExpr::binary(id, CExpr::call(id, "fabs", vec![CExpr::var(id, v.clone())]), "<=", CExpr::lit(id, format!("{:.1}", g))),
            Guard::Whole(v, g) => CExpr::call(id, "viv_wg", vec![CExpr::var(id, v.clone()), CExpr::lit(id, format!("{:.1}", g))]),
        };
        test = Some(match test {
            None => part,
            Some(t) => CExpr::binary(id, t, "&&", part),
        });
    }
    test.map(|t| CExpr::call(id, "VIV_LIKELY", vec![t]))
}

// ---- the access report -----------------------------------------------------

/// With `VIV_INDEX_REPORT=<file>` set, every array access the C backend
/// emits is listed in that file, one line each: its location, its def, its
/// rank, and how it is indexed -- `int` (an integer index), `guarded` (an
/// integer index behind a run-time guard, the double path otherwise) or
/// `double` (the double path only) -- and which dimensions are proved in
/// bounds (`proved 0,1`). It is how the report of which bounds checks
/// remain in a program is made.
pub(crate) fn note(loc: &str, rank: usize, how: &str, mask: u32) {
    let Ok(path) = std::env::var("VIV_INDEX_REPORT") else { return };
    let def = STATE.with(|s| s.borrow().cur.clone().unwrap_or_else(|| "-".to_string()));
    let proved: Vec<String> = (0..rank).filter(|d| mask >> d & 1 == 1).map(|d| d.to_string()).collect();
    let line = format!("{loc}\t{def}\trank {rank}\t{how}\tproved {}\n", if proved.is_empty() { "-".to_string() } else { proved.join(",") });
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(line.as_bytes());
    }
}
