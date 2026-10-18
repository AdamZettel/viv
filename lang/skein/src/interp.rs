// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! A small tree-walking interpreter for Skein — evaluates the parsed `Datum`
//! AST directly. This is the *browser* execution path: the skein crate compiles
//! to wasm, and a page runs programs through `run` (no C toolchain needed). The
//! compiler (-> C -> native) remains the production path; the two share the reader.
//!
//! It is intentionally simple (recursive eval, no tail-call optimization — deep
//! tail recursion can exhaust the Rust stack; the *compiler* is the one with TCO).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use knotkit::span::SourceId;

use crate::{read_all, Datum};

#[derive(Clone)]
enum Val {
    Int(i64),
    Flo(f64),
    Bool(bool),
    Nil,
    Sym(String),
    Str(String),
    Pair(Rc<(Val, Val)>),
    Closure(Rc<Closure>),
    Prim(&'static str),
}

struct Closure {
    params: Vec<String>,
    body: Vec<Datum>,
    env: Env,
}

type Env = Rc<RefCell<Frame>>;

struct Frame {
    vars: HashMap<String, Val>,
    parent: Option<Env>,
}

fn child(parent: &Env) -> Env {
    Rc::new(RefCell::new(Frame { vars: HashMap::new(), parent: Some(parent.clone()) }))
}

fn lookup(env: &Env, name: &str) -> Option<Val> {
    let f = env.borrow();
    if let Some(v) = f.vars.get(name) {
        Some(v.clone())
    } else if let Some(p) = &f.parent {
        lookup(p, name)
    } else {
        None
    }
}

const PRIMS: &[&str] = &[
    "+", "-", "*", "=", "<", ">", "eq?", "equal?", "null?", "pair?", "cons", "car", "cdr", "not", "list", "display", "newline",
];

/// Evaluate a whole program, returning everything it printed (or an error).
pub fn run(src: &str) -> Result<String, String> {
    let data = read_all(src, SourceId(0)).map_err(|d| d.message)?;
    let global: Env = Rc::new(RefCell::new(Frame { vars: HashMap::new(), parent: None }));
    let mut out = String::new();
    for d in &data {
        eval(d, &global, &mut out)?;
    }
    Ok(out)
}

fn eval(d: &Datum, env: &Env, out: &mut String) -> Result<Val, String> {
    match d {
        Datum::Int(i, _) => Ok(Val::Int(*i)),
        Datum::Flo(f, _) => Ok(Val::Flo(*f)),
        Datum::Bool(b, _) => Ok(Val::Bool(*b)),
        Datum::Str(s, _) => Ok(Val::Str(s.clone())),
        Datum::Sym(name, _) => {
            if let Some(v) = lookup(env, name) {
                Ok(v)
            } else if let Some(p) = PRIMS.iter().find(|p| **p == name) {
                Ok(Val::Prim(p))
            } else {
                Err(format!("unbound variable `{name}`"))
            }
        }
        Datum::List(items, _) => eval_list(items, env, out),
    }
}

fn eval_list(items: &[Datum], env: &Env, out: &mut String) -> Result<Val, String> {
    if items.is_empty() {
        return Err("empty application ()".into());
    }
    if let Datum::Sym(head, _) = &items[0] {
        match head.as_str() {
            "quote" => return Ok(quote(&items[1])),
            "if" => {
                let c = eval(&items[1], env, out)?;
                return if truthy(&c) {
                    eval(&items[2], env, out)
                } else if items.len() > 3 {
                    eval(&items[3], env, out)
                } else {
                    Ok(Val::Bool(false))
                };
            }
            "lambda" => {
                let Datum::List(sig, _) = &items[1] else { return Err("lambda params must be a list".into()) };
                let params = sig.iter().map(sym).collect::<Result<_, _>>()?;
                return Ok(Val::Closure(Rc::new(Closure { params, body: items[2..].to_vec(), env: env.clone() })));
            }
            "define" => return eval_define(items, env, out),
            "begin" => return eval_body(&items[1..], env, out),
            "let" | "let*" => return eval_let(&items[1], &items[2..], env, out, head == "let*"),
            "cond" => return eval_cond(&items[1..], env, out),
            _ => {}
        }
    }
    // Application.
    let f = eval(&items[0], env, out)?;
    let mut args = Vec::with_capacity(items.len() - 1);
    for a in &items[1..] {
        args.push(eval(a, env, out)?);
    }
    apply(&f, args, out)
}

fn eval_define(items: &[Datum], env: &Env, out: &mut String) -> Result<Val, String> {
    match &items[1] {
        Datum::Sym(name, _) => {
            let v = eval(&items[2], env, out)?;
            env.borrow_mut().vars.insert(name.clone(), v);
        }
        Datum::List(sig, _) => {
            let name = sym(&sig[0])?;
            let params = sig[1..].iter().map(sym).collect::<Result<_, _>>()?;
            let clos = Val::Closure(Rc::new(Closure { params, body: items[2..].to_vec(), env: env.clone() }));
            env.borrow_mut().vars.insert(name, clos);
        }
        _ => return Err("malformed define".into()),
    }
    Ok(Val::Nil)
}

fn eval_let(binds: &Datum, body: &[Datum], env: &Env, out: &mut String, star: bool) -> Result<Val, String> {
    let Datum::List(bs, _) = binds else { return Err("let bindings must be a list".into()) };
    let scope = child(env);
    for b in bs {
        let Datum::List(pair, _) = b else { return Err("let binding must be (name value)".into()) };
        let name = sym(&pair[0])?;
        // let* sees earlier bindings; let evaluates in the outer env.
        let v = if star { eval(&pair[1], &scope, out)? } else { eval(&pair[1], env, out)? };
        scope.borrow_mut().vars.insert(name, v);
    }
    eval_body(body, &scope, out)
}

fn eval_cond(clauses: &[Datum], env: &Env, out: &mut String) -> Result<Val, String> {
    for clause in clauses {
        let Datum::List(parts, _) = clause else { return Err("cond clause must be a list".into()) };
        let is_else = matches!(&parts[0], Datum::Sym(s, _) if s == "else");
        if is_else {
            return eval_body(&parts[1..], env, out);
        }
        let t = eval(&parts[0], env, out)?;
        if truthy(&t) {
            return eval_body(&parts[1..], env, out);
        }
    }
    Ok(Val::Bool(false))
}

fn eval_body(body: &[Datum], env: &Env, out: &mut String) -> Result<Val, String> {
    let mut last = Val::Nil;
    for d in body {
        last = eval(d, env, out)?;
    }
    Ok(last)
}

fn apply(f: &Val, args: Vec<Val>, out: &mut String) -> Result<Val, String> {
    match f {
        Val::Closure(c) => {
            if args.len() != c.params.len() {
                return Err(format!("procedure expects {} args, got {}", c.params.len(), args.len()));
            }
            let scope = child(&c.env);
            for (p, a) in c.params.iter().zip(args) {
                scope.borrow_mut().vars.insert(p.clone(), a);
            }
            eval_body(&c.body, &scope, out)
        }
        Val::Prim(name) => prim(name, args, out),
        _ => Err("attempt to call a non-procedure".into()),
    }
}

fn prim(name: &str, a: Vec<Val>, out: &mut String) -> Result<Val, String> {
    let nums = |a: &[Val]| -> Result<Vec<f64>, String> { a.iter().map(num).collect() };
    let all_int = a.iter().all(|v| matches!(v, Val::Int(_)));
    match name {
        "+" => Ok(if all_int { Val::Int(a.iter().map(int).sum()) } else { Val::Flo(nums(&a)?.iter().sum()) }),
        "*" => Ok(if all_int { Val::Int(a.iter().map(int).product()) } else { Val::Flo(nums(&a)?.iter().product()) }),
        "-" => {
            if a.is_empty() {
                return Err("- needs an argument".into());
            }
            if all_int {
                let mut it = a.iter().map(int);
                let first = it.next().unwrap();
                Ok(Val::Int(if a.len() == 1 { -first } else { it.fold(first, |x, y| x - y) }))
            } else {
                let v = nums(&a)?;
                Ok(Val::Flo(if a.len() == 1 { -v[0] } else { v[1..].iter().fold(v[0], |x, y| x - y) }))
            }
        }
        "=" => Ok(Val::Bool(num(&a[0])? == num(&a[1])?)),
        "<" => Ok(Val::Bool(num(&a[0])? < num(&a[1])?)),
        ">" => Ok(Val::Bool(num(&a[0])? > num(&a[1])?)),
        "eq?" | "equal?" => Ok(Val::Bool(val_eq(&a[0], &a[1]))),
        "null?" => Ok(Val::Bool(matches!(a[0], Val::Nil))),
        "pair?" => Ok(Val::Bool(matches!(a[0], Val::Pair(_)))),
        "not" => Ok(Val::Bool(!truthy(&a[0]))),
        "cons" => Ok(Val::Pair(Rc::new((a[0].clone(), a[1].clone())))),
        "car" => match &a[0] {
            Val::Pair(p) => Ok(p.0.clone()),
            _ => Err("car of non-pair".into()),
        },
        "cdr" => match &a[0] {
            Val::Pair(p) => Ok(p.1.clone()),
            _ => Err("cdr of non-pair".into()),
        },
        "list" => {
            let mut r = Val::Nil;
            for v in a.into_iter().rev() {
                r = Val::Pair(Rc::new((v, r)));
            }
            Ok(r)
        }
        "display" => {
            write_val(&a[0], out);
            Ok(Val::Nil)
        }
        "newline" => {
            out.push('\n');
            Ok(Val::Nil)
        }
        _ => Err(format!("unknown primitive `{name}`")),
    }
}

fn quote(d: &Datum) -> Val {
    match d {
        Datum::Int(i, _) => Val::Int(*i),
        Datum::Flo(f, _) => Val::Flo(*f),
        Datum::Bool(b, _) => Val::Bool(*b),
        Datum::Str(s, _) => Val::Str(s.clone()),
        Datum::Sym(s, _) => Val::Sym(s.clone()),
        Datum::List(items, _) => {
            let mut r = Val::Nil;
            for item in items.iter().rev() {
                r = Val::Pair(Rc::new((quote(item), r)));
            }
            r
        }
    }
}

fn truthy(v: &Val) -> bool {
    !matches!(v, Val::Bool(false))
}
fn int(v: &Val) -> i64 {
    match v {
        Val::Int(i) => *i,
        Val::Flo(f) => *f as i64,
        _ => 0,
    }
}
fn num(v: &Val) -> Result<f64, String> {
    match v {
        Val::Int(i) => Ok(*i as f64),
        Val::Flo(f) => Ok(*f),
        _ => Err("expected a number".into()),
    }
}
fn val_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Flo(x), Val::Flo(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Nil, Val::Nil) => true,
        (Val::Sym(x), Val::Sym(y)) | (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Pair(x), Val::Pair(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}
fn sym(d: &Datum) -> Result<String, String> {
    match d {
        Datum::Sym(s, _) => Ok(s.clone()),
        _ => Err("expected a name".into()),
    }
}

fn write_val(v: &Val, out: &mut String) {
    match v {
        Val::Int(i) => out.push_str(&i.to_string()),
        Val::Flo(f) => out.push_str(&fmt_flo(*f)),
        Val::Bool(b) => out.push_str(if *b { "#t" } else { "#f" }),
        Val::Nil => out.push_str("()"),
        Val::Sym(s) | Val::Str(s) => out.push_str(s),
        Val::Pair(_) => {
            out.push('(');
            let mut cur = v.clone();
            let mut first = true;
            while let Val::Pair(p) = cur {
                if !first {
                    out.push(' ');
                }
                first = false;
                write_val(&p.0, out);
                cur = p.1.clone();
            }
            if !matches!(cur, Val::Nil) {
                out.push_str(" . ");
                write_val(&cur, out);
            }
            out.push(')');
        }
        Val::Closure(_) | Val::Prim(_) => out.push_str("#<procedure>"),
    }
}

/// Format a float like C's `%g` closely enough for the demos.
fn fmt_flo(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e15 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}
