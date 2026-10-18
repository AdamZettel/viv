// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv export` of a kernel collection:
//! several defs of a program's files, each its own C entry point, in one C
//! file with a ctypes wrapper.
//!
//! The collection shares the program track's pieces (crate::export): the
//! manifest's types and checks, the error boundary (`viv_embed_run`, a
//! `longjmp` back to the entry), the state restore and the wrapper's
//! errors and checks. What it adds:
//!
//! - **the program text**: one `import` line per file of the manifest, so
//!   each file is a module with its own top level (two files' test blocks
//!   do not share names), and the defs keep their names unless two modules
//!   define one (then `kernels_k1_norms__enorm`, crate::imports::mangle);
//! - **module initialization**: the program's top level (each module's
//!   constants, in import order, such as lib/zfunc's `ZF_A =
//!   zf_coefficients(...)`) runs once, at the first call or at `init`; the
//!   state after it is the snapshot each call starts from;
//! - **an entry per kernel**: an argument struct and a result struct; the
//!   entry checks the arguments (types, ranges, NaN and infinity unless the
//!   argument opts in, an array's rank and squareness), calls the def's C
//!   function inside the error boundary and copies its results out. A def
//!   with several results gives a struct of several fields, a tuple in
//!   Python. Complex numbers are `double _Complex` fields;
//! - **the agreement check**: each kernel's arguments are the def's
//!   parameters, by name, in order, with the type the def annotates; its
//!   results are the def's, by count and type.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::ast::{Item, Program, TypeExpr};
use crate::export::{c_ident, fmt_num, py_str, range_words, wrap, Kernel, Kind, Manifest, Param, RecArg, PY_BODY_1, PY_COMMON, PY_LOAD};
use crate::lower::{Elem, Ty};

/// A kernel's def as the compiled program has it.
#[derive(Clone, Debug)]
pub struct KDef {
    /// its C function
    pub cname: String,
    /// the C fields its result record holds, one per result (`_0`, `_1`,
    /// ... for `return a, b`; a record's own field names); empty for one
    /// plain value
    pub fields: Vec<String>,
    /// per record parameter (Kernel::rec_args), its C type
    pub rec_c: Vec<String>,
    /// the C type it returns
    pub ret_c: String,
    /// per argument: an array the entry lends the def without copying it
    /// (`borrowable`); false for every scalar
    pub borrow: Vec<bool>,
}

/// The program text of a collection: one `import` line per file, in the
/// manifest's order.
pub fn program_text(m: &Manifest) -> Result<String, String> {
    let mut t = format!("# `viv export` of the kernel collection {}: each file a module, imported in order\n", m.name);
    for f in &m.files {
        let Some(stem) = f.strip_suffix(".viv") else { return Err(format!("error: {f}: a kernel collection's files are .viv modules")) };
        let parts: Vec<&str> = stem.split('/').collect();
        if parts.iter().any(|p| !c_ident(p)) {
            return Err(format!("error: {f}: each part of a module's path is a name (letters, digits, `_`), to be imported"));
        }
        let _ = writeln!(t, "import {}", parts.join("."));
    }
    Ok(t)
}

/// The type a manifest's kind names in the compiled program.
fn ty_of(kind: &Kind) -> Option<Ty> {
    Some(match kind {
        Kind::Num => Ty::Num,
        Kind::Int => Ty::Int,
        Kind::Bool => Ty::Bool,
        Kind::Cplx => Ty::Cplx,
        Kind::Arr => Ty::Arr(Elem::Num),
        Kind::IntArr => Ty::Arr(Elem::Int),
        _ => return None,
    })
}

/// An array argument or result: numbers (`Arr`) or Ints (`[Int]`), which
/// share the runtime's storage (an Int array's slots read as `long long`).
fn is_arr(k: &Kind) -> bool {
    matches!(k, Kind::Arr | Kind::IntArr)
}

/// The `#|` docstring of `def name(` in a file's text: the comment lines
/// right above the def, without their `#| `, joined.
pub fn def_doc(text: &str, name: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let at = lines.iter().position(|l| l.starts_with(&format!("def {name}(")))?;
    let mut doc: Vec<&str> = Vec::new();
    // `#|` (what) and `#||` (how) are the docstring; `#|||` (why), `#~`
    // (tags) and `#:` (glossary) lines may sit among them and are skipped
    for l in lines[..at].iter().rev() {
        let t = l.trim_start();
        if t.starts_with("#|||") || t.starts_with("#~") || t.starts_with("#:") {
            continue;
        }
        match t.strip_prefix("#||").or_else(|| t.strip_prefix("#|")) {
            Some(t) => doc.push(t.trim()),
            None => break,
        }
    }
    doc.reverse();
    (!doc.is_empty()).then(|| doc.join(" "))
}

/// A type in the manifest's words.
fn ty_word(t: Ty) -> String {
    match t {
        Ty::Num => "Num".into(),
        Ty::Int => "Int".into(),
        Ty::Bool => "Bool".into(),
        Ty::Cplx => "Complex".into(),
        Ty::Arr(Elem::Num) => "Arr".into(),
        Ty::Arr(Elem::Cplx) => "CArr".into(),
        Ty::Arr(Elem::Int) => "[Int]".into(),
        Ty::Void => "nothing".into(),
        other => other.name().to_string(),
    }
}

/// An annotation's type, as far as a kernel's argument can be one.
fn annotated(t: &TypeExpr) -> Option<Ty> {
    Some(match t {
        TypeExpr::Num => Ty::Num,
        TypeExpr::Int => Ty::Int,
        TypeExpr::Bool => Ty::Bool,
        TypeExpr::Arr => Ty::Arr(Elem::Num),
        TypeExpr::CArr => Ty::Arr(Elem::Cplx),
        TypeExpr::IntList => Ty::Arr(Elem::Int),
        TypeExpr::Named { name, .. } if name == "Complex" || name == "Cplx" => Ty::Cplx,
        _ => return None,
    })
}

/// Check each kernel against its def and find the def's C function.
///
/// `files` holds each file of the manifest parsed on its own (its own defs
/// under their own names); `funcs` the compiled program's functions (C name
/// -> parameter types and result type); `records` its records' fields.
pub fn resolve(m: &Manifest, files: &[(String, Program)], funcs: &HashMap<String, (Vec<Ty>, Ty)>, records: &HashMap<String, Vec<(String, Ty)>>) -> Result<Vec<KDef>, String> {
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for k in &m.kernels {
        match resolve_one(k, files, funcs, records) {
            Ok(d) => out.push(d),
            Err(e) => errors.push(format!("error: kernel `{}` ({}:{}): {e}\n", k.name, k.file, k.def)),
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors.concat())
    }
}

fn resolve_one(k: &Kernel, files: &[(String, Program)], funcs: &HashMap<String, (Vec<Ty>, Ty)>, records: &HashMap<String, Vec<(String, Ty)>>) -> Result<KDef, String> {
    let prog = &files.iter().find(|(f, _)| *f == k.file).ok_or("its file was not read")?.1;
    let def = prog
        .items
        .iter()
        .find_map(|it| match it {
            Item::Def(d) if d.name == k.def => Some(d),
            _ => None,
        })
        .ok_or_else(|| format!("{} defines no `{}`", k.file, k.def))?;
    // the arguments are the def's parameters, by name and in order (a
    // record parameter's fields stand for it)
    let names: Vec<&str> = def.params.iter().map(|p| p.name.as_str()).collect();
    let want = def_params(k);
    if names != want.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>() {
        return Err(format!("the manifest's arguments are ({}), and the def's parameters are ({})", want.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "), names.join(", ")));
    }
    for (p, (_, ai)) in def.params.iter().zip(&want) {
        let Some(ai) = ai else { continue };
        let a = &k.args[*ai];
        let Some(t) = &p.ty else {
            return Err(format!("the parameter `{}` has no type; annotate it (`{}: {}`), so the def says what the entry takes", p.name, p.name, a.kind.word()));
        };
        let t = annotated(t).ok_or_else(|| format!("the parameter `{}` is of a type a kernel cannot take", p.name))?;
        if Some(t) != ty_of(&a.kind) {
            return Err(format!("the def's `{}` is {}, and the manifest says {}", p.name, ty_word(t), a.kind.word()));
        }
    }
    // the C function: the def's own name, or its module's (crate::imports)
    let stem = k.file.strip_suffix(".viv").unwrap_or(&k.file);
    let mangled = crate::imports::mangle(stem, &k.def);
    let (cname, (ptys, ret)) = [mangled, k.def.clone()]
        .into_iter()
        .find_map(|n| funcs.get(&n).map(|f| (n, f.clone())))
        .ok_or("the compiled program has no such function (internal)")?;
    let mut rec_c = Vec::new();
    for (pt, (pname, ai)) in ptys.iter().zip(&want) {
        match ai {
            Some(ai) if Some(*pt) != ty_of(&k.args[*ai].kind) => return Err(format!("the compiled `{pname}` takes {} (internal)", ty_word(*pt))),
            None if !matches!(pt, Ty::Record(_)) => return Err(format!("the compiled `{pname}` is not a record (internal)")),
            None => rec_c.push(pt.c_type_name()),
            _ => {}
        }
    }
    // its results: one value, or a record of several (crate::tuples), or a
    // record of the program's (its fields by name)
    let (results, fields): (Vec<Ty>, Vec<String>) = match ret {
        Ty::Record(n) if n.starts_with("viv__ret_") => {
            let r = records.get(n).ok_or("its results' record was not found (internal)")?;
            (r.iter().map(|f| f.1).collect(), (0..r.len()).map(|i| format!("_{i}")).collect())
        }
        Ty::Record(n) if k.rec_result.is_some() => {
            let r = records.get(n).ok_or("its result's record was not found (internal)")?;
            (r.iter().map(|f| f.1).collect(), r.iter().map(|f| f.0.clone()).collect())
        }
        Ty::Void => return Err("the def returns nothing; a kernel gives a result".into()),
        t => (vec![t], Vec::new()),
    };
    if results.len() != k.results.len() {
        return Err(format!("the def gives {} result{}, and the manifest names {}", results.len(), if results.len() == 1 { "" } else { "s" }, k.results.len()));
    }
    for (t, r) in results.iter().zip(&k.results) {
        if Some(*t) != ty_of(&r.kind) {
            return Err(format!("the result `{}` is {} in the def, and the manifest says {}", r.name, ty_word(*t), r.kind.word()));
        }
    }
    let lend = borrowable(prog, &k.def);
    let at = def_index(k);
    let borrow = k.args.iter().enumerate().map(|(i, a)| is_arr(&a.kind) && lend.get(at[i]).copied().unwrap_or(false)).collect();
    Ok(KDef { cname, fields, rec_c, ret_c: ret.c_type_name(), borrow })
}

/// The def's parameters as the kernel's arguments stand for them: each
/// name, with its argument's index, or None for a record parameter (whose
/// fields are several arguments).
fn def_params(k: &Kernel) -> Vec<(String, Option<usize>)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < k.args.len() {
        if let Some(r) = k.rec_args.iter().find(|r| r.first == i) {
            out.push((r.param.clone(), None));
            i += r.fields.len().max(1);
        } else {
            out.push((k.args[i].name.clone(), Some(i)));
            i += 1;
        }
    }
    out
}

/// Each argument's position among the def's parameters.
fn def_index(k: &Kernel) -> Vec<usize> {
    let mut out = Vec::new();
    for (j, (_, ai)) in def_params(k).iter().enumerate() {
        match ai {
            Some(_) => out.push(j),
            None => {
                let n = k.rec_args.iter().find(|r| r.first == out.len()).map_or(1, |r| r.fields.len());
                out.extend(std::iter::repeat(j).take(n));
            }
        }
    }
    out
}

// ---------------------------------------------------------------- C

fn c_arg_field(kind: &Kind, pre: &str) -> String {
    match kind {
        Kind::Num => "double".into(),
        Kind::Int => "int64_t".into(),
        Kind::Bool => "int".into(),
        Kind::Cplx => "double _Complex".into(),
        _ => format!("{pre}arg"),
    }
}

fn c_res_field(kind: &Kind, pre: &str) -> String {
    match kind {
        Kind::Num => "double".into(),
        Kind::Int => "int64_t".into(),
        Kind::Bool => "int".into(),
        Kind::Cplx => "double _Complex".into(),
        _ => format!("{pre}array"),
    }
}

/// A kernel's call in words: `boys(n_max, T) -> F`.
fn signature(k: &Kernel) -> String {
    let args: Vec<&str> = k.args.iter().map(|a| a.name.as_str()).collect();
    let res: Vec<&str> = k.results.iter().map(|r| r.name.as_str()).collect();
    format!("{}({}) -> {}", k.name, args.join(", "), res.join(", "))
}

/// An argument in words: type, rank, range.
fn arg_words(a: &Param) -> String {
    let mut t = match a.kind {
        Kind::Num => "float".to_string(),
        Kind::Int => "int".to_string(),
        Kind::Bool => "bool".to_string(),
        Kind::Cplx => "complex".to_string(),
        Kind::IntArr => match a.rank {
            Some(r) => format!("int64 array of rank {r}"),
            None => "int64 array".to_string(),
        },
        _ => match (a.square, a.rank) {
            (true, _) => "square float64 matrix".to_string(),
            (false, Some(r)) => format!("float64 array of rank {r}"),
            (false, None) => "float64 array".to_string(),
        },
    };
    t.push_str(&range_words(a, if is_arr(&a.kind) { "each element " } else { "" }));
    if a.nonfinite && matches!(a.kind, Kind::Num | Kind::Cplx | Kind::Arr) {
        t.push_str(", NaN and infinity allowed");
    }
    t
}

/// `k1.h`: the collection's declarations.
pub fn header(m: &Manifest, version: &str) -> String {
    let (p, pre) = (&m.name, &m.prefix);
    let up = p.to_uppercase();
    let mut h = String::new();
    let _ = writeln!(h, "/* {p}.h -- written by `viv export`; do not edit.\n   {version}\n   A kernel collection. */");
    let _ = writeln!(h, "#ifndef {up}_H\n#define {up}_H\n#include <stddef.h>\n#include <stdint.h>\n");
    let _ = writeln!(h, "#define {up}_ABI {}\n", m.abi);
    let _ = writeln!(h, "/* an array argument, borrowed for the call: row-major, `dims[0..rank)` */");
    let _ = writeln!(h, "typedef struct {{ const double* data; int64_t rank; const int64_t* dims; }} {pre}arg;");
    let _ = writeln!(h, "/* an array result, allocated by the library and freed by the kernel's _free */");
    let _ = writeln!(h, "typedef struct {{ double* data; int64_t rank; int64_t dims[8]; }} {pre}array;");
    let _ = writeln!(h, "/* each printed line, without its newline; stream 1 is output, 2 a diagnostic */");
    let _ = writeln!(h, "typedef void (*{pre}line_fn)(const char* line, long n, int stream, void* ctx);\n");
    let _ = writeln!(h, "enum {{ {up}_OK = 0, {up}_ERROR = 1, {up}_INTERRUPTED = 2, {up}_BAD_PARAM = 3, {up}_BUSY = 4 }};");
    let _ = writeln!(h, "enum {{ {up}_E_ERROR = 1, {up}_E_INDEX = 2, {up}_E_VALUE = 3, {up}_E_OVERFLOW = 4, {up}_E_ZERODIV = 5,\n       {up}_E_RECURSION = 6, {up}_E_MEMORY = 7, {up}_E_INTERRUPT = 8, {up}_E_STOP = 9 }};\n");
    for k in &m.kernels {
        let _ = writeln!(h, "/* {}", signature(k));
        for a in &k.args {
            let _ = writeln!(h, "     {}: {}", a.name, arg_words(a).replace("*/", "* /"));
        }
        for r in &k.results {
            let _ = writeln!(h, "     -> {}: {}{}", r.name, r.kind.word(), r.doc.as_ref().map(|d| format!(", {}", d.replace("*/", "* /"))).unwrap_or_default());
        }
        let _ = writeln!(h, "   0 on success; otherwise msg (cap bytes) holds the error and r->error_kind its kind. */");
        if k.args.is_empty() {
            let _ = writeln!(h, "typedef struct {{ char unused_; }} {pre}{}_args;", k.name);
        } else {
            let fields: Vec<String> = k.args.iter().map(|a| format!("{} {};", c_arg_field(&a.kind, pre), a.name)).collect();
            let _ = writeln!(h, "typedef struct {{ {} }} {pre}{}_args;", fields.join(" "), k.name);
        }
        let fields: Vec<String> = k.results.iter().map(|r| format!("{} {};", c_res_field(&r.kind, pre), r.name)).collect();
        let _ = writeln!(h, "typedef struct {{ {} int error_kind; }} {pre}{}_results;", fields.join(" "), k.name);
        let _ = writeln!(h, "int {pre}{k}(const {pre}{k}_args* a, {pre}{k}_results* r, char* msg, size_t cap);", k = k.name);
        let _ = writeln!(h, "void {pre}{k}_free({pre}{k}_results* r);   /* the arrays of r; safe on a zeroed r and twice */\n", k = k.name);
    }
    let _ = writeln!(h, "/* Run the program's top level once: its modules' constants, in import\n   order, and its test blocks. The first call of a kernel does it if this\n   was not called. Its status is kept: a failed init fails every call. */");
    let _ = writeln!(h, "int {pre}init({pre}line_fn on_line, void* ctx, char* msg, size_t cap);");
    let _ = writeln!(h, "/* where the lines a kernel prints go; NULL (the default) drops them */");
    let _ = writeln!(h, "void {pre}set_output({pre}line_fn on_line, void* ctx);");
    let _ = writeln!(h, "/* the test blocks init ran: their count, and each one's name, result and first failure */");
    let _ = writeln!(h, "int {pre}ntests(void);\nconst char* {pre}test_name(int i);\nint {pre}test_passed(int i);\nconst char* {pre}test_detail(int i);");
    let _ = writeln!(h, "/* stop the call in progress at its next loop pass; from any thread */");
    let _ = writeln!(h, "void {pre}interrupt(void);\nint {pre}abi(void);\nconst char* {pre}version(void);");
    let _ = writeln!(h, "#endif");
    h
}

/// The checks of one argument before any work: `x` is the argument (`a->T`).
fn checks(e: &mut String, a: &Param, x: &str) {
    let f = &a.name;
    let bad = |cond: &str, fmt: &str, args: &str| format!("        if ({cond}) return vivk_bad(msg, cap, \"{f}{fmt}\"{args});\n");
    let range = |e: &mut String, v: &str, fmt: &str| {
        if let Some(lo) = a.min {
            let (op, words) = if a.min_excl { (">", "above") } else { (">=", "at least") };
            e.push_str(&bad(&format!("!({v} {op} {lo:?})"), &format!(" is {fmt}; it must be {words} {}", fmt_num(lo)), &format!(", {v}")));
        }
        if let Some(hi) = a.max {
            let (op, words) = if a.max_excl { ("<", "below") } else { ("<=", "at most") };
            e.push_str(&bad(&format!("!({v} {op} {hi:?})"), &format!(" is {fmt}; it must be {words} {}", fmt_num(hi)), &format!(", {v}")));
        }
    };
    match a.kind {
        Kind::Num => {
            if !a.nonfinite {
                e.push_str(&bad(&format!("!isfinite({x})"), " is %g; it must be a finite number", &format!(", {x}")));
            }
            range(e, x, "%.17g");
        }
        Kind::Int => range(e, &format!("(double){x}"), "%.17g"),
        Kind::Bool => e.push_str(&bad(&format!("{x} != 0 && {x} != 1"), " is %d; a Bool is 0 or 1", &format!(", {x}"))),
        Kind::Cplx => {
            if !a.nonfinite {
                e.push_str(&bad(&format!("!isfinite(creal({x})) || !isfinite(cimag({x}))"), " is %g%+gi; it must be finite", &format!(", creal({x}), cimag({x})")));
            }
        }
        _ => {
            let want = a.rank.unwrap_or(0);
            let _ = writeln!(e, "        {{\n            char why[96];\n            long long dims[8];\n            if (!vivk_arg_ok({x}.rank, {x}.dims, (const void*){x}.data, dims, {want}, why, sizeof why)) return vivk_bad(msg, cap, \"{f} %s\", why);\n        }}");
            if a.square {
                e.push_str(&bad(&format!("{x}.dims[0] != {x}.dims[1]"), " is %lld by %lld; it must be square", &format!(", (long long){x}.dims[0], (long long){x}.dims[1]")));
            }
            let lo_ck = a.min.map(|lo| format!("!(v {} {lo:?})", if a.min_excl { ">" } else { ">=" }));
            let hi_ck = a.max.map(|hi| format!("!(v {} {hi:?})", if a.max_excl { "<" } else { "<=" }));
            let mut each = Vec::new();
            if !a.nonfinite && a.kind == Kind::Arr {
                each.push(("!isfinite(v)".to_string(), "it must hold finite numbers".to_string()));
            }
            if let Some(c) = lo_ck {
                each.push((c, format!("each element must be {} {}", if a.min_excl { "above" } else { "at least" }, fmt_num(a.min.unwrap_or(0.0)))));
            }
            if let Some(c) = hi_ck {
                each.push((c, format!("each element must be {} {}", if a.max_excl { "below" } else { "at most" }, fmt_num(a.max.unwrap_or(0.0)))));
            }
            if !each.is_empty() {
                let _ = writeln!(e, "        {{\n            long long n = 1;\n            for (int i = 0; i < {x}.rank; i++) n *= {x}.dims[i];\n            for (long long i = 0; i < n; i++) {{\n                double v = {};", if a.kind == Kind::IntArr { format!("(double)((const long long*){x}.data)[i]") } else { format!("{x}.data[i]") });
                for (c, why) in each {
                    let _ = writeln!(e, "                if ({c}) return vivk_bad(msg, cap, \"{f}[%lld] (flat) is %g; {why}\", i, v);");
                }
                let _ = writeln!(e, "            }}\n        }}");
            }
        }
    }
}

/// The C after the program: the state restore, the init, an entry per kernel.
pub fn entry(m: &Manifest, defs: &[KDef], c: &str, globals: &HashMap<String, Ty>, version: &str) -> Result<String, String> {
    let (pre, up) = (&m.prefix, m.name.to_uppercase());
    let mut e = String::new();
    let has = |w: &str| c.contains(w);
    let arena = has("viv_arena_top") && has("static void viv_arena_release(");
    // an argument the entry copies (not one it lends) is made by viv_arr_new
    let copies = m.kernels.iter().zip(defs).any(|(k, d)| k.args.iter().enumerate().any(|(i, a)| is_arr(&a.kind) && !d.borrow.get(i).copied().unwrap_or(false)));
    // a program that makes no array of its own has no viv_arr_new: the
    // entry then copies into blocks of its own, freed after the call
    let own_copies = copies && !has("static viv_arr viv_arr_new(");
    let _ = writeln!(e, "/* ---- the exported entry points (`viv export`) ---- */");
    let _ = writeln!(e, "#undef main");
    // a program of defs alone has an empty `main` (B346), renamed
    // viv_program_main above; this covers C written without one
    if !c.contains("int main(void)") {
        let _ = writeln!(e, "static int viv_program_main(void) {{ return 0; }}");
    }
    let _ = writeln!(e, "#if defined(__GNUC__)\n#define {up}_EXPORT __attribute__((visibility(\"default\")))\n#else\n#define {up}_EXPORT\n#endif");
    let _ = writeln!(e, "static int vivk_busy = 0;\nstatic {pre}line_fn vivk_on_line = 0;\nstatic void* vivk_ctx = 0;");
    // the state after init, restored before every call
    let state = crate::export::per_run_state(c, globals);
    let _ = writeln!(e, "\n/* The program's state after init: restored before each call, so a call\n   depends only on its arguments. Kept across calls instead: viv export's\n   KEEP_ACROSS_RUNS (lang/viv/src/export.rs). */");
    let _ = writeln!(e, "static int vivk_loaded = 0;");
    for (i, n) in state.iter().enumerate() {
        let _ = writeln!(e, "static unsigned char vivk_s{i}[sizeof({n})];");
    }
    let _ = writeln!(e, "static void vivk_restore(void) {{\n    if (!vivk_loaded) {{");
    for (i, n) in state.iter().enumerate() {
        let _ = writeln!(e, "        memcpy(vivk_s{i}, (const void*)&{n}, sizeof({n}));");
    }
    let _ = writeln!(e, "        vivk_loaded = 1;\n        return;\n    }}");
    for (i, n) in state.iter().enumerate() {
        let _ = writeln!(e, "    memcpy((void*)&{n}, vivk_s{i}, sizeof({n}));");
    }
    let _ = writeln!(e, "}}\n");
    // helpers
    let _ = writeln!(e, "static int vivk_bad(char* msg, size_t cap, const char* fmt, ...) {{\n    if (msg && cap) {{\n        va_list ap;\n        va_start(ap, fmt);\n        vsnprintf(msg, cap, fmt, ap);\n        va_end(ap);\n    }}\n    vivk_busy = 0;\n    return {up}_BAD_PARAM;\n}}");
    let _ = writeln!(e, "VIV_UNUSED static int vivk_arg_ok(int64_t rank, const int64_t* dims, const void* data, long long dst[8], int want, char* why, size_t cap) {{\n    long long n = 1;\n    if (rank < 1 || rank > 8 || (want && rank != want)) {{ snprintf(why, cap, \"has rank %lld%s\", (long long)rank, want ? \"; it must have the manifest's rank\" : \"\"); return 0; }}\n    if (!dims) {{ snprintf(why, cap, \"has no dims\"); return 0; }}\n    for (int i = 0; i < rank; i++) {{\n        if (dims[i] < 0 || dims[i] > 2147483647LL) {{ snprintf(why, cap, \"has dimension %lld\", (long long)dims[i]); return 0; }}\n        n *= dims[i];\n        if (n > 2147483647LL) {{ snprintf(why, cap, \"has more than 2147483647 elements\"); return 0; }}\n        dst[i] = dims[i];\n    }}\n    if (n > 0 && !data) {{ snprintf(why, cap, \"has no data\"); return 0; }}\n    return 1;\n}}");
    if own_copies {
        let nmax = m.kernels.iter().map(|k| k.args.iter().filter(|a| is_arr(&a.kind)).count()).max().unwrap_or(1).max(1);
        let _ = writeln!(e, "/* an array argument, copied into a block of the entry's, freed after the call\n   (the program makes no array of its own, so its runtime has no viv_arr_new) */\nstatic double* vivk_tmp[{nmax}];\nstatic int vivk_ntmp = 0;\nstatic viv_arr vivk_in_arr(const {pre}arg* g) {{\n    viv_arr r;\n    long long n = 1;\n    r.rank = (int)g->rank;\n    for (int i = 0; i < 8; i++) r.dims[i] = 0;\n    for (int i = 0; i < g->rank; i++) {{ r.dims[i] = (int)g->dims[i]; n *= g->dims[i]; }}\n    r.size = (int)n;\n    r.data = (double*)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));\n    if (!r.data) {{\n        viv_errf(\"error: out of memory copying an argument of %lld elements\\n\", n);\n        viv_stop(VIV_E_MEMORY);\n    }}\n    vivk_tmp[vivk_ntmp++] = r.data;\n    if (n > 0) memcpy(r.data, g->data, (size_t)n * sizeof(double));\n    return r;\n}}");
    } else if copies {
        let _ = writeln!(e, "/* an array argument, copied into an array of the call's arena */\nstatic viv_arr vivk_in_arr(const {pre}arg* g) {{\n    int dims[8];\n    for (int i = 0; i < g->rank; i++) dims[i] = (int)g->dims[i];\n    viv_arr r = viv_arr_new((int)g->rank, dims);\n    if (r.size) memcpy(r.data, g->data, (size_t)r.size * sizeof(double));\n    return r;\n}}");
    }
    if defs.iter().any(|d| d.borrow.iter().any(|&b| b)) {
        let _ = writeln!(e, "/* an array argument the def only reads (export_kernels::borrowable): the\n   caller's buffer itself, not filed on the arena, so nothing frees it */\nstatic viv_arr vivk_lend_arr(const {pre}arg* g) {{\n    viv_arr r;\n    long long n = 1;\n    r.rank = (int)g->rank;\n    for (int i = 0; i < 8; i++) r.dims[i] = 0;\n    for (int i = 0; i < g->rank; i++) {{ r.dims[i] = (int)g->dims[i]; n *= g->dims[i]; }}\n    r.size = (int)n;\n    r.data = (double*)g->data;\n    return r;\n}}");
    }
    if m.kernels.iter().any(|k| k.results.iter().any(|r| is_arr(&r.kind))) {
        let _ = writeln!(e, "/* an array result, copied out of the arena into a block the caller frees */\nstatic void vivk_out_arr({pre}array* d, viv_arr s) {{\n    d->rank = s.rank;\n    for (int i = 0; i < s.rank && i < 8; i++) d->dims[i] = s.dims[i];\n    d->data = (double*)malloc((size_t)(s.size > 0 ? s.size : 1) * sizeof(double));\n    if (!d->data) {{\n        viv_errf(\"error: out of memory copying a result of %d elements\\n\", s.size);\n        viv_stop(VIV_E_MEMORY);\n    }}\n    if (s.size > 0) memcpy(d->data, s.data, (size_t)s.size * sizeof(double));\n}}");
    }

    // init: the program's top level, once
    let _ = writeln!(e, "\nstatic int vivk_inited = 0;\nstatic int vivk_init_status = 0;\nstatic int vivk_init_kind = 0;\nstatic char vivk_init_msg[1024];\nstatic viv_embed_test_row* vivk_tests = 0;\nstatic int vivk_ntests = 0;");
    let _ = writeln!(e, "static int vivk_do_init({pre}line_fn on_line, void* ctx, char* msg, size_t cap) {{\n    if (!vivk_inited) {{\n        vivk_inited = 1;\n        viv_embed_stop_flag = 0;\n        int code = viv_embed_run(viv_program_main, on_line ? on_line : vivk_on_line, on_line ? ctx : vivk_ctx, vivk_init_msg, sizeof vivk_init_msg);\n        vivk_tests = viv_embed_take_tests(&vivk_ntests);\n        vivk_init_kind = code;\n        vivk_init_status = code == 0 ? {up}_OK : code == VIV_E_INTERRUPT ? {up}_INTERRUPTED : {up}_ERROR;\n        /* the snapshot each call starts from */\n        if (code == 0) vivk_restore();\n    }}\n    if (msg && cap) snprintf(msg, cap, \"%s\", vivk_init_msg);\n    return vivk_init_status;\n}}");
    let _ = writeln!(e, "\n{up}_EXPORT int {pre}init({pre}line_fn on_line, void* ctx, char* msg, size_t cap) {{\n    if (msg && cap) msg[0] = 0;\n    if (vivk_busy) return {up}_BUSY;\n    vivk_busy = 1;\n    int s = vivk_do_init(on_line, ctx, msg, cap);\n    vivk_busy = 0;\n    return s;\n}}");
    let _ = writeln!(e, "{up}_EXPORT void {pre}set_output({pre}line_fn on_line, void* ctx) {{ vivk_on_line = on_line; vivk_ctx = ctx; }}");
    let _ = writeln!(e, "{up}_EXPORT int {pre}ntests(void) {{ return vivk_ntests; }}");
    let _ = writeln!(e, "{up}_EXPORT const char* {pre}test_name(int i) {{ return i >= 0 && i < vivk_ntests ? vivk_tests[i].name : 0; }}");
    let _ = writeln!(e, "{up}_EXPORT int {pre}test_passed(int i) {{ return i >= 0 && i < vivk_ntests ? vivk_tests[i].passed : 0; }}");
    let _ = writeln!(e, "{up}_EXPORT const char* {pre}test_detail(int i) {{ return i >= 0 && i < vivk_ntests ? vivk_tests[i].detail : 0; }}");
    let _ = writeln!(e, "{up}_EXPORT int {pre}abi(void) {{ return {up}_ABI; }}");
    let _ = writeln!(e, "{up}_EXPORT const char* {pre}version(void) {{ return {version:?}; }}");
    let _ = writeln!(e, "{up}_EXPORT void {pre}interrupt(void) {{ viv_embed_stop_flag = 1; }}");

    for (k, d) in m.kernels.iter().zip(defs) {
        let n = &k.name;
        let _ = writeln!(e, "\n/* {} */", signature(k));
        let _ = writeln!(e, "static {pre}{n}_args vivk_a_{n};\nstatic {pre}{n}_results* vivk_r_{n};");
        // the body, inside the error boundary: the arguments as Viv values,
        // the def, the results copied out
        let _ = writeln!(e, "static int vivk_body_{n}(void) {{\n    const {pre}{n}_args* a = &vivk_a_{n};\n    {pre}{n}_results* r = vivk_r_{n};\n    (void)a;");
        let mut vals = Vec::new();
        for (i, a) in k.args.iter().enumerate() {
            let v = format!("x{i}");
            let _ = match a.kind {
                Kind::Num => writeln!(e, "    double {v} = a->{};", a.name),
                Kind::Int => writeln!(e, "    long long {v} = (long long)a->{};", a.name),
                Kind::Bool => writeln!(e, "    double {v} = a->{} ? 1.0 : 0.0;", a.name),
                Kind::Cplx => writeln!(e, "    double complex {v} = a->{};", a.name),
                _ if d.borrow.get(i).copied().unwrap_or(false) => writeln!(e, "    viv_arr {v} = vivk_lend_arr(&a->{});", a.name),
                _ => writeln!(e, "    viv_arr {v} = vivk_in_arr(&a->{});", a.name),
            };
            vals.push(v);
        }
        // a record parameter: its fields' arrays, made into the record
        let mut call = Vec::new();
        let mut ri = 0;
        for (pname, ai) in def_params(k) {
            match ai {
                Some(i) => call.push(vals[i].clone()),
                None => {
                    let r = &k.rec_args[ri];
                    let inits: Vec<String> = r.fields.iter().enumerate().map(|(j, f)| format!(".{f} = {}", vals[r.first + j])).collect();
                    let _ = writeln!(e, "    {0} rec_{pname} = ({0}){{ {1} }};", d.rec_c[ri], inits.join(", "));
                    call.push(format!("rec_{pname}"));
                    ri += 1;
                }
            }
        }
        // `vivk_res`, not `res`: a def may be called res
        // the def as the C backend named it: `index` is emitted `index_` (B347)
        let _ = writeln!(e, "    {} vivk_res = {}({});", d.ret_c, knotkit::backend::escape_ident(&d.cname), call.join(", "));
        for (i, r) in k.results.iter().enumerate() {
            let src = if d.fields.is_empty() { "vivk_res".to_string() } else { format!("vivk_res.{}", d.fields[i]) };
            let _ = match r.kind {
                Kind::Num | Kind::Cplx => writeln!(e, "    r->{} = {src};", r.name),
                Kind::Int => writeln!(e, "    r->{} = (int64_t){src};", r.name),
                Kind::Bool => writeln!(e, "    r->{} = {src} != 0.0;", r.name),
                _ => writeln!(e, "    vivk_out_arr(&r->{}, {src});", r.name),
            };
        }
        let _ = writeln!(e, "    return 0;\n}}");
        // free
        let _ = writeln!(e, "{up}_EXPORT void {pre}{n}_free({pre}{n}_results* r) {{\n    if (!r) return;");
        for r in k.results.iter().filter(|r| is_arr(&r.kind)) {
            let _ = writeln!(e, "    free(r->{0}.data);\n    r->{0}.data = 0;", r.name);
        }
        let _ = writeln!(e, "}}");
        // the entry
        let _ = writeln!(e, "{up}_EXPORT int {pre}{n}(const {pre}{n}_args* a, {pre}{n}_results* r, char* msg, size_t cap) {{");
        let _ = writeln!(e, "    if (msg && cap) msg[0] = 0;\n    if (!a || !r) return {up}_BAD_PARAM;\n    memset(r, 0, sizeof *r);\n    if (vivk_busy) return {up}_BUSY;\n    vivk_busy = 1;");
        let _ = writeln!(e, "    if (!vivk_inited || vivk_init_status) {{\n        int s = vivk_do_init(0, 0, msg, cap);\n        if (s) {{\n            r->error_kind = vivk_init_kind;\n            vivk_busy = 0;\n            return s;\n        }}\n    }}");
        let _ = writeln!(e, "    {{");
        for a in &k.args {
            checks(&mut e, a, &format!("a->{}", a.name));
        }
        let _ = writeln!(e, "    }}");
        let _ = writeln!(e, "    vivk_a_{n} = *a;\n    vivk_r_{n} = r;\n    vivk_restore();");
        if arena {
            let _ = writeln!(e, "    size_t mark = viv_arena_top;");
        }
        let _ = writeln!(e, "    viv_embed_stop_flag = 0;\n    int code = viv_embed_run(vivk_body_{n}, vivk_on_line, vivk_ctx, msg, cap);");
        let _ = writeln!(e, "    {{\n        int nt = 0;\n        viv_embed_test_row* t = viv_embed_take_tests(&nt);\n        viv_embed_free_tests(t, nt);\n    }}");
        if arena {
            let _ = writeln!(e, "    viv_arena_release(mark, 0);");
        }
        if own_copies {
            let _ = writeln!(e, "    while (vivk_ntmp > 0) free(vivk_tmp[--vivk_ntmp]);");
        }
        let _ = writeln!(e, "    vivk_busy = 0;\n    if (code == 0) return {up}_OK;\n    {pre}{n}_free(r);\n    r->error_kind = code;\n    return code == VIV_E_INTERRUPT ? {up}_INTERRUPTED : {up}_ERROR;\n}}");
    }
    Ok(e)
}

/// The whole `k1.c`: the header, the program, the entry points.
pub fn c_file(m: &Manifest, defs: &[KDef], c: &str, globals: &HashMap<String, Ty>, version: &str) -> Result<String, String> {
    let p = &m.name;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "/* {p}.c -- written by `viv export`; do not edit.\n   {version}\n\n   A kernel collection: one file, C99 with GNU attributes, the C library\n   only. Build it:\n\n       cc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}.so -lm\n\n   (exact: the interpreter's bits; README.md has the fast build). The\n   interface is {p}.h, pasted below. */"
    );
    let _ = writeln!(out, "#ifndef VIV_EMBED\n#define VIV_EMBED 1\n#endif");
    out.push_str(&header(m, version));
    out.push_str("#include <stdarg.h>\n#include <string.h>\n#include <stdlib.h>\n#include <math.h>\n#include <complex.h>\n\n");
    out.push_str(c);
    out.push('\n');
    out.push_str(&entry(m, defs, c, globals, version)?);
    Ok(out)
}

// ------------------------------------------------------------- Python

/// `k1.py`: the ctypes wrapper, a function per kernel.
pub fn python(m: &Manifest, about: &str, version: &str) -> String {
    let (p, pre) = (&m.name, &m.prefix);
    let up = p.to_uppercase();
    let mut s = String::new();
    let names: Vec<&str> = m.kernels.iter().map(|k| k.name.as_str()).collect();
    let _ = writeln!(s, "\"\"\"{p}: a Viv kernel collection, exported by `viv export`.\n");
    if !about.is_empty() {
        s.push_str(&wrap(about, 76, ""));
        s.push('\n');
    }
    let _ = writeln!(s, "    import {p}\n    {p}.{}(...)\n", names.first().copied().unwrap_or(""));
    let _ = writeln!(s, "The kernels: {}. help({p}.<kernel>) gives each one's arguments and results.\n", names.join(", "));
    let _ = writeln!(s, "The library lib{p}.so (.dylib, .dll) is built from {p}.c beside this file on\nfirst import when it is missing or older than {p}.c, with $CC (default cc):\n\n    cc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}.so -lm\n\nSet {up}_FAST=1 for the fast build, {up}_LIB to load a library built elsewhere.\nImporting runs the collection's module initialization once (each module's\nconstants and its test blocks; tests() gives them). A kernel's printed lines\ngo to the logger \"{p}\" at INFO. One call at a time per process.\n\nWritten by `viv export`; do not edit. {version}\n\"\"\"");
    s.push_str(PY_BODY_1);
    let _ = writeln!(s, "\nNAME = {p:?}\nPREFIX = {pre:?}\nVERSION = {}\nABI = {}", py_str(version), m.abi);
    s.push_str(PY_LOAD);
    // the kernels: (args, results)
    let _ = writeln!(s, "\n\n# each kernel: its arguments (name, kind, min, max, rank, nonfinite, min exclusive,\n# max exclusive, square) and its results (name, kind)\nKERNELS = {{");
    let num = |x: Option<f64>| x.map_or("None".to_string(), |v| format!("{v:?}"));
    let tf = |b: bool| if b { "True" } else { "False" };
    for k in &m.kernels {
        let args: Vec<String> = k.args.iter().map(|a| format!("({:?}, {:?}, {}, {}, {}, {}, {}, {}, {})", a.name, a.kind.word(), num(a.min), num(a.max), a.rank.map_or("None".to_string(), |r| r.to_string()), tf(a.nonfinite), tf(a.min_excl), tf(a.max_excl), tf(a.square))).collect();
        let res: Vec<String> = k.results.iter().map(|r| format!("({:?}, {:?})", r.name, r.kind.word())).collect();
        let _ = writeln!(s, "    {:?}: ([{}], [{}]),", k.name, args.join(", "), res.join(", "));
    }
    let _ = writeln!(s, "}}\n__all__ = list(KERNELS)");
    // record parameters (param, first argument, fields) and record results
    let _ = writeln!(s, "\n# record parameters: (parameter, its first argument, its fields); record results\nRECORD_ARGS = {{");
    for k in m.kernels.iter().filter(|k| !k.rec_args.is_empty()) {
        let rs: Vec<String> = k.rec_args.iter().map(|r| format!("({:?}, {}, [{}])", r.param, r.first, r.fields.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(", "))).collect();
        let _ = writeln!(s, "    {:?}: [{}],", k.name, rs.join(", "));
    }
    let _ = writeln!(s, "}}\nRECORD_RESULTS = {{{}}}", m.kernels.iter().filter_map(|k| k.rec_result.as_ref().map(|r| format!("{:?}: {r:?}", k.name))).collect::<Vec<_>>().join(", "));
    s.push_str(PY_KBIND);
    s.push_str(PY_COMMON);
    s.push_str(PY_KCALL);
    for k in &m.kernels {
        let dp = def_params(k);
        let args: Vec<&str> = dp.iter().map(|(n, _)| n.as_str()).collect();
        let doc = kernel_doc(k);
        let _ = writeln!(s, "\n\ndef {}({}):\n    \"\"\"{}    \"\"\"\n    return _call({:?}, ({}{}))", k.name, args.join(", "), doc.replace("\"\"\"", "'''").replace('\\', "\\\\"), k.name, args.join(", "), if args.len() == 1 { "," } else { "" });
    }
    s
}

/// A kernel's docstring: its call, the def's `#|` text, each argument's
/// type and range, its results and the manifest's reference.
fn kernel_doc(k: &Kernel) -> String {
    let mut doc = String::new();
    let what = k.info.iter().find(|(key, _)| key == "doc").map(|(_, v)| v.clone());
    let _ = writeln!(doc, "{}", signature(k));
    if let Some(w) = what {
        doc.push('\n');
        doc.push_str(&wrap(&w, 72, "    "));
    }
    doc.push('\n');
    for a in &k.args {
        let _ = writeln!(doc, "    {} ({})", a.name, arg_words(a));
    }
    let several = k.results.len() > 1;
    let _ = writeln!(doc, "\n    Returns {}:", if let Some(r) = &k.rec_result { format!("a dict of the record {r}'s fields") } else if several { "a tuple".into() } else { "one value".into() });
    for r in &k.results {
        let ty = match r.kind {
            Kind::Num => "float",
            Kind::Int => "int",
            Kind::Bool => "bool",
            Kind::Cplx => "complex",
            Kind::IntArr => "int64 array",
            _ => "float64 array",
        };
        let _ = writeln!(doc, "    {} ({ty}){}", r.name, r.doc.as_ref().map(|d| format!(": {d}")).unwrap_or_default());
    }
    for (key, v) in &k.info {
        if matches!(key.as_str(), "reference" | "tolerance" | "domain") {
            doc.push('\n');
            doc.push_str(&wrap(&format!("{key}: {v}"), 72, "    "));
        }
    }
    doc
}

/// The library's entry points, bound by name, and the module's init.
const PY_KBIND: &str = r#"

_KT = {"Num": _C.c_double, "Int": _C.c_int64, "Bool": _C.c_int, "Cplx": _C.c_double * 2, "Arr": _Arg, "IntArr": _Arg}
_KR = {"Num": _C.c_double, "Int": _C.c_int64, "Bool": _C.c_int, "Cplx": _C.c_double * 2, "Arr": _Array, "IntArr": _Array}

_lib = _C.CDLL(_lib_path)


def _sym(n):
    return getattr(_lib, PREFIX + n)


_sym("version").restype = _C.c_char_p
_sym("abi").restype = _C.c_int
if _sym("abi")() != ABI:
    raise ImportError("%s: %s was built from another version of %s.c (ABI %d, this wrapper %d); delete it to rebuild"
                      % (NAME, _lib_path, NAME, _sym("abi")(), ABI))
_entries = {}
for _k, (_args, _res) in KERNELS.items():
    _A = type("_Args_" + _k, (_C.Structure,), {"_fields_": [(a[0], _KT[a[1]]) for a in _args] or [("unused_", _C.c_char)]})
    _R = type("_Results_" + _k, (_C.Structure,), {"_fields_": [(r[0], _KR[r[1]]) for r in _res] + [("error_kind", _C.c_int)]})
    _f = _sym(_k)
    _f.argtypes = [_C.POINTER(_A), _C.POINTER(_R), _C.c_char_p, _C.c_size_t]
    _f.restype = _C.c_int
    _fr = _sym(_k + "_free")
    _fr.argtypes = [_C.POINTER(_R)]
    _fr.restype = None
    _entries[_k] = (_A, _R, _f, _fr)


def _on_line(raw, n, stream, _ctx):
    text = _C.string_at(raw, n).decode("utf-8", "replace")
    if stream == 2:
        _log.warning("%s", text)
    elif text.startswith("__"):
        _log.debug("%s", text)
    else:
        _log.info("%s", text)


_line_cb = _LINE(_on_line)              # kept for the library's lifetime
_sym("set_output").argtypes = [_LINE, _C.c_void_p]
_sym("set_output").restype = None
_sym("set_output")(_line_cb, None)
_sym("init").argtypes = [_LINE, _C.c_void_p, _C.c_char_p, _C.c_size_t]
_sym("init").restype = _C.c_int
_sym("test_name").restype = _C.c_char_p
_sym("test_detail").restype = _C.c_char_p
_init_msg = _C.create_string_buffer(1024)
if _sym("init")(_line_cb, None, _init_msg, len(_init_msg)) != 0:
    raise ImportError("%s: the module initialization failed: %s" % (NAME, _init_msg.value.decode("utf-8", "replace")))


def version():
    """The compiler commit and program hash the library was generated from."""
    return _sym("version")().decode()


def tests():
    """The test blocks the module initialization ran: (name, passed, detail)."""
    return [(_sym("test_name")(i).decode(), bool(_sym("test_passed")(i)), _sym("test_detail")(i).decode())
            for i in range(_sym("ntests")())]
"#;

/// A kernel's call: the arguments checked and laid out, the results back as
/// Python values, the errors raised.
const PY_KCALL: &str = r#"

def _field(param, v, f):
    if hasattr(v, "keys") and hasattr(v, "__getitem__"):
        if f not in v.keys():
            raise TypeError("%s has no field %r" % (param, f))
        return v[f]
    if not hasattr(v, f):
        raise TypeError("%s has no field %r" % (param, f))
    return getattr(v, f)


def _expand(kernel, vals):
    """A record parameter (a mapping, or an object with the fields as
    attributes) as one argument per field."""
    rec = RECORD_ARGS.get(kernel)
    if not rec:
        return vals
    groups = {first: (param, fields) for param, first, fields in rec}
    nargs = len(KERNELS[kernel][0])
    flat, i, it = [], 0, iter(vals)
    while i < nargs:
        v = next(it)
        if i in groups:
            param, fields = groups[i]
            flat += [_field(param, v, f) for f in fields]
            i += len(fields)
        else:
            flat.append(v)
            i += 1
    return tuple(flat)


def _call(kernel, vals):
    A, R, f, free = _entries[kernel]
    args, res = KERNELS[kernel]
    vals = _expand(kernel, vals)
    if len(vals) != len(args):
        raise TypeError("%s takes %d arguments, not %d" % (kernel, len(args), len(vals)))
    a = A()
    keep = []                        # what the C borrows during the call
    for (name, kind, lo, hi, rank, nonfinite, lo_excl, hi_excl, square), v in zip(args, vals):
        v = _check(name, kind, lo, hi, rank, None, nonfinite, v, None, lo_excl, hi_excl)
        if kind in ("Arr", "IntArr"):
            if square and v.shape[0] != v.shape[1]:
                raise ValueError("%s is %d by %d; it must be square" % (name, v.shape[0], v.shape[1]))
            dims = (_C.c_int64 * v.ndim)(*v.shape)
            keep += [v, dims]
            setattr(a, name, _Arg(v.ctypes.data_as(_C.c_void_p), v.ndim, dims))
        elif kind == "Cplx":
            setattr(a, name, (_C.c_double * 2)(v.real, v.imag))
        else:
            setattr(a, name, v)
    r = R()
    msg = _C.create_string_buffer(1024)
    code = f(_C.byref(a), _C.byref(r), msg, len(msg))
    try:
        if code == 4:
            raise RuntimeError("%s: a call is already in progress in this process" % NAME)
        if code == 3:
            raise ValueError(msg.value.decode("utf-8", "replace"))
        if code == 2:
            raise KeyboardInterrupt("%s.%s was interrupted" % (NAME, kernel))
        if code != 0:
            raise _KINDS.get(r.error_kind, VivError)(msg.value.decode("utf-8", "replace"))
        out = []
        for name, kind in res:
            x = getattr(r, name)
            if kind == "Num":
                out.append(float(x))
            elif kind == "Int":
                out.append(int(x))
            elif kind == "Bool":
                out.append(bool(x))
            elif kind == "Cplx":
                out.append(complex(x[0], x[1]))
            elif kind == "IntArr":
                out.append(_array(x, _C.c_int64, _np.int64))
            else:
                out.append(_array(x, _C.c_double, _np.float64))
        if kernel in RECORD_RESULTS:
            return {name: x for (name, _), x in zip(res, out)}
        return out[0] if len(out) == 1 else tuple(out)
    finally:
        free(_C.byref(r))


def results_hash(kernel, outs):
    """SHA-256 of a kernel's results over several calls (`outs`, one value or
    tuple per call): each result's name, then its bytes (a number as an IEEE
    double, an int as 8 bytes, a complex number as two doubles, an array's
    rank, shape and elements)."""
    import hashlib
    import struct
    h = hashlib.sha256()
    names = [r[0] for r in KERNELS[kernel][1]]
    for out in outs:
        vals = [out[n] for n in names] if isinstance(out, dict) else out if len(names) > 1 else (out,)
        for name, v in zip(names, vals):
            h.update(name.encode() + b"\0")
            if isinstance(v, (bool, _np.bool_)):
                h.update(struct.pack("<d", float(v)))
            elif isinstance(v, (int, _np.integer)):
                h.update(struct.pack("<q", int(v)))
            elif isinstance(v, (float, _np.floating)):
                h.update(struct.pack("<d", float(v)))
            elif isinstance(v, (complex, _np.complexfloating)):
                h.update(struct.pack("<dd", v.real, v.imag))
            else:
                a = _np.asarray(v)
                h.update(struct.pack("<q", a.ndim) + b"".join(struct.pack("<q", d) for d in a.shape))
                h.update(a.astype("<f8").tobytes())
    return h.hexdigest()
"#;

/// `README.md` of a kernel collection.
pub fn readme(m: &Manifest, about: &str, version: &str) -> String {
    let (p, pre) = (&m.name, &m.prefix);
    let up = p.to_uppercase();
    let mut s = String::new();
    let _ = writeln!(s, "# {p}\n");
    if !about.is_empty() {
        let _ = writeln!(s, "{}\n", wrap(about, 78, "").trim_end());
    }
    let _ = writeln!(s, "A collection of kernels written in Viv and exported as one C file with a\nPython wrapper. The files here are written by `viv export`; edit the Viv\nsource, not these.\n");
    let _ = writeln!(s, "- `{p}.c`: the kernels and the part of Viv's runtime they use. C99 with GNU\n  attributes; it needs a C compiler and the C library, nothing else.\n- `{p}.h`: the C interface (`{pre}<kernel>`), also pasted at the top of `{p}.c`.\n- `{p}.py`: the Python interface (ctypes and NumPy), a function per kernel.\n- `test_{p}.py`: its tests.\n");
    let _ = writeln!(s, "## Build\n\n```sh\ncc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC {p}.c -o lib{p}.so -lm\n```\n");
    let _ = writeln!(s, "`import {p}` runs that line itself (with `$CC`, default `cc`) when the library\nis missing or older than `{p}.c`. `-ffp-contract=off` must stay: this exact\nbuild gives the bits Viv's interpreter gives, on every platform and compiler.\n`{up}_FAST=1` builds and loads the fast one (`-O3 -DVIV_FAST -ffp-contract=fast`),\nwhose results differ in the last bits.\n");
    let _ = writeln!(s, "## The kernels\n");
    for k in &m.kernels {
        let _ = writeln!(s, "### `{}`\n", signature(k));
        if let Some((_, d)) = k.info.iter().find(|(key, _)| key == "doc") {
            let _ = writeln!(s, "{}", wrap(d, 78, "").trim_end());
            s.push('\n');
        }
        for a in &k.args {
            let _ = writeln!(s, "- `{}`: {}", a.name, arg_words(a));
        }
        for r in &k.results {
            let _ = writeln!(s, "- returns `{}`: {}{}", r.name, r.kind.word(), r.doc.as_ref().map(|d| format!(", {d}")).unwrap_or_default());
        }
        for (key, v) in k.info.iter().filter(|(key, _)| key != "doc") {
            let _ = writeln!(s, "- {key}: {v}");
        }
        s.push('\n');
    }
    let _ = writeln!(s, "## Use\n\nAn argument is checked before any work: its type, its range, its rank, and\nNaN or infinity, which are refused unless the manifest allows them. A bad\nargument raises `ValueError` or `TypeError`; an error inside a kernel raises\n`{p}.VivError` (also an `IndexError`, `ValueError`, ... by kind) with Viv's\nmessage and line, and the next call works. Each call starts from the state\nthe module initialization left. One call at a time per process; a call\nruns in the calling thread, so Ctrl-C waits for it.\n");
    if m.cases.is_some() {
        let _ = writeln!(s, "## Cases\n\n`cases.json` holds each kernel's calls (`{{\"kernel\": [[arg, ...], ...]}}`, a\ncomplex number as `[re, im]`). `test_{p}.py` makes them and, in the exact\nbuild, compares the results' hash with `exact_hashes.json`, recorded from\nthe library and Viv's interpreter.\n");
    }
    let _ = writeln!(s, "## Version\n\n{version}");
    s
}

/// `test_k1.py`: init's tests pass; each kernel's calls in `cases.json` give
/// the recorded bits; a NaN and a wrong count of arguments are refused.
pub fn test_py(m: &Manifest) -> String {
    let p = &m.name;
    let mut s = String::new();
    let _ = writeln!(s, "\"\"\"Tests of the exported kernel collection {p}: python -m pytest test_{p}.py, or python test_{p}.py.\nWritten by `viv export`; do not edit.\"\"\"");
    let _ = writeln!(s, "import json\nimport math\nimport os\nimport sys\n\nimport numpy as np\n\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport {p}  # noqa: E402\n");
    let _ = writeln!(s, "HERE = os.path.dirname(os.path.abspath(__file__))\nHASHES = os.path.join(HERE, \"exact_hashes.json\")\nCASES = os.path.join(HERE, \"cases.json\")\n");
    s.push_str(&TEST_K.replace("{p}", p));
    s
}

const TEST_K: &str = r#"
def decode(kind, v):
    """A value of cases.json as the kernel takes it."""
    if kind == "Cplx":
        return complex(v[0], v[1])
    if kind == "Arr":
        return np.array(v, dtype=float)
    return v


def calls(kernel):
    cases = json.load(open(CASES)) if os.path.exists(CASES) else {}
    args = {p}.KERNELS[kernel][0]
    return [[decode(a[1], v) for a, v in zip(args, call)] for call in cases.get(kernel, [])]


def test_the_module_initialized_and_its_tests_pass():
    failed = [t for t in {p}.tests() if not t[1]]
    assert not failed, failed


def test_every_kernel_gives_the_recorded_bits():
    want = json.load(open(HASHES)) if os.path.exists(HASHES) else {}
    for kernel in {p}.KERNELS:
        cs = calls(kernel)
        outs = [getattr({p}, kernel)(*c) for c in cs]
        if kernel in want and not {p}._FAST:
            assert {p}.results_hash(kernel, outs) == want[kernel], kernel
        print("ok", kernel, len(cs), "calls", flush=True)


def test_a_nan_argument_is_refused_and_the_next_call_works():
    for kernel, (args, _) in {p}.KERNELS.items():
        cs = calls(kernel)
        for i, a in enumerate(args):
            if not cs or a[5] or a[1] not in ("Num", "Cplx", "Arr"):
                continue
            bad = list(cs[0])
            bad[i] = math.nan if a[1] == "Num" else complex(math.nan, 0.0) if a[1] == "Cplx" else np.full_like(bad[i], math.nan)
            if a[1] == "Arr" and bad[i].size == 0:
                continue
            try:
                getattr({p}, kernel)(*bad)
            except ValueError:
                pass
            else:
                raise AssertionError("%s: a NaN %s was not refused" % (kernel, a[0]))
            getattr({p}, kernel)(*cs[0])


def test_a_wrong_count_of_arguments_is_refused():
    kernel, (args, _) = next(iter({p}.KERNELS.items()))
    try:
        {p}._call(kernel, (0,) * (len(args) + 1))
    except TypeError:
        return
    raise AssertionError("no TypeError")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
"#;

// ------------------------------------------------------------ inference

/// The Python names the generated module defines itself: a kernel of one of
/// these names would replace it.
const MODULE_NAMES: &[&str] = &["build", "version", "tests", "results_hash", "normalize"];

/// Python's keywords: a def or a parameter of one of these names cannot be
/// a Python function or argument.
const PY_RESERVED: &[&str] = &["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"];

/// An annotation as a kernel argument's kind, or why it cannot be one.
fn kind_of_annotation(t: &TypeExpr) -> Result<Kind, String> {
    Ok(match t {
        TypeExpr::Num => Kind::Num,
        TypeExpr::Int => Kind::Int,
        TypeExpr::Bool => Kind::Bool,
        TypeExpr::Arr => Kind::Arr,
        TypeExpr::IntList => Kind::IntArr,
        TypeExpr::Named { name, .. } if name == "Complex" || name == "Cplx" => Kind::Cplx,
        TypeExpr::CArr => return Err("complex arrays (CArr) cannot cross the boundary yet".into()),
        TypeExpr::Str | TypeExpr::StrList => return Err("strings cannot be a kernel's argument".into()),
        TypeExpr::Named { name, .. } | TypeExpr::RecList { name, .. } => return Err(format!("`{name}` is a record; a kernel takes numbers and arrays")),
        TypeExpr::Opt(_) => return Err("an optional cannot be a kernel's argument".into()),
        _ => return Err("its type cannot cross the boundary".into()),
    })
}

/// A compiled result type as a kernel result's kind.
fn kind_of_ty(t: Ty) -> Option<Kind> {
    Some(match t {
        Ty::Num => Kind::Num,
        Ty::Int => Kind::Int,
        Ty::Bool => Kind::Bool,
        Ty::Cplx => Kind::Cplx,
        Ty::Arr(Elem::Num) => Kind::Arr,
        Ty::Arr(Elem::Int) => Kind::IntArr,
        _ => return None,
    })
}

/// A def's bare-string docstring: a string literal as its first statement.
fn string_doc(d: &crate::ast::Def) -> Option<String> {
    match d.body.first() {
        Some(crate::ast::Stmt::ExprStmt { expr: crate::ast::Expr::Str { value, .. }, .. }) => Some(value.trim().to_string()),
        _ => None,
    }
}

/// `viv export --infer file.viv`, first half: a kernel collection of the
/// file's top-level defs whose parameters are all annotated with a type that
/// crosses the boundary (Num, Int, Bool, Complex, Arr, [Int]). Each other def
/// is left out with a note saying why. The results are filled in by
/// `infer_results` once the program is compiled.
///
/// `name` is the collection's name (the file's stem) and `file` its path as
/// a manifest would name it.
pub fn infer(name: &str, file: &str, prog: &Program, text: &str) -> (Manifest, Vec<String>) {
    let mut m = Manifest {
        name: name.to_string(),
        about: None,
        files: vec![file.to_string()],
        deck_param: "material".into(),
        decks: Vec::new(),
        params: Vec::new(),
        groups: Vec::new(),
        results: Vec::new(),
        cases: None,
        reference: Vec::new(),
        is_kernels: true,
        prefix: format!("{name}_"),
        abi: 1,
        kernels: Vec::new(),
        page: Default::default(),
    };
    let mut notes = Vec::new();
    let recs: HashMap<&str, &crate::ast::RecordDecl> = prog.items.iter().filter_map(|i| if let Item::Record(r) = i { Some((r.name.as_str(), r)) } else { None }).collect();
    let plain = |name: String, kind: Kind| Param { name, kind, min: None, max: None, min_excl: false, max_excl: false, rank: None, nonfinite: true, square: false, shape: None, doc: None, normalize: None, units: None, range: None };
    for it in &prog.items {
        let Item::Def(d) = it else { continue };
        let skip = |why: String| format!("{file}: `{}` is not exported: {why}", d.name);
        if d.name.starts_with('_') {
            notes.push(skip("its name starts with `_`, so it is private to the module".into()));
            continue;
        }
        if MODULE_NAMES.contains(&d.name.as_str()) || PY_RESERVED.contains(&d.name.as_str()) {
            notes.push(skip("the Python module uses that name itself; rename the def".into()));
            continue;
        }
        let mut args = Vec::new();
        let mut rec_args = Vec::new();
        let mut why = None;
        for p in &d.params {
            let Some(t) = &p.ty else {
                why = Some(format!("the parameter `{}` has no type; annotate it (`{}: Num`, or Arr, Int, Bool, Complex, [Int])", p.name, p.name));
                break;
            };
            // a record of array fields: one argument per field (named columns)
            if let TypeExpr::Named { name, .. } = t {
                if let Some(r) = recs.get(name.as_str()) {
                    let bad = r.fields.iter().find(|f| !matches!(kind_of_annotation(&f.ty), Ok(Kind::Arr | Kind::IntArr)));
                    if let Some(f) = bad {
                        why = Some(format!("the parameter `{}` is the record `{name}`, whose field `{}` is not Arr or [Int]; a record crosses as named columns of arrays", p.name, f.name));
                        break;
                    }
                    if r.fields.is_empty() || PY_RESERVED.contains(&p.name.as_str()) {
                        why = Some(format!("the parameter `{}` (the record `{name}`) cannot cross: it has no fields, or its name is a Python keyword", p.name));
                        break;
                    }
                    rec_args.push(RecArg { param: p.name.clone(), record: name.clone(), first: args.len(), fields: r.fields.iter().map(|f| f.name.clone()).collect() });
                    for f in &r.fields {
                        let kind = kind_of_annotation(&f.ty).unwrap_or(Kind::Arr);
                        args.push(plain(format!("{}__{}", p.name, f.name), kind));
                    }
                    continue;
                }
            }
            match kind_of_annotation(t) {
                Ok(_) if PY_RESERVED.contains(&p.name.as_str()) => {
                    why = Some(format!("the parameter `{}` is a Python keyword", p.name));
                    break;
                }
                Ok(kind) => args.push(plain(p.name.clone(), kind)),
                Err(e) => {
                    why = Some(format!("the parameter `{}`: {e}", p.name));
                    break;
                }
            }
        }
        if let Some(w) = why {
            notes.push(skip(w));
            continue;
        }
        let mut info = Vec::new();
        if let Some(doc) = def_doc(text, &d.name).or_else(|| string_doc(d)) {
            info.push(("doc".to_string(), doc));
        }
        let k = Kernel { name: d.name.clone(), file: file.to_string(), def: d.name.clone(), args, results: Vec::new(), info, rec_args, rec_result: None };
        // the manifest's own checks (names the C struct or C itself reserves),
        // with a placeholder result
        let mut probe = m.clone();
        let mut kp = k.clone();
        kp.results.push(crate::export::KResult { name: "result".into(), kind: Kind::Num, doc: None });
        probe.kernels = vec![kp];
        if let Err(e) = crate::export::check_kernels_manifest(&probe) {
            notes.push(skip(e));
            continue;
        }
        m.kernels.push(k);
    }
    (m, notes)
}

/// `viv export --infer`, second half: each kernel's results from its compiled
/// function. A def whose results cannot cross the boundary (nothing, a
/// string, a record, a complex array) is left out with a note.
pub fn infer_results(m: &mut Manifest, funcs: &HashMap<String, (Vec<Ty>, Ty)>, records: &HashMap<String, Vec<(String, Ty)>>, notes: &mut Vec<String>) {
    let stem = m.files.first().map(|f| f.strip_suffix(".viv").unwrap_or(f).to_string()).unwrap_or_default();
    let mut keep = Vec::new();
    for mut k in std::mem::take(&mut m.kernels) {
        let skip = |why: &str| format!("{}: `{}` is not exported: {why}", k.file, k.def);
        let f = [crate::imports::mangle(&stem, &k.def), k.def.clone()].into_iter().find_map(|n| funcs.get(&n).cloned());
        let Some((_, ret)) = f else {
            notes.push(skip("the compiled program has no such function"));
            continue;
        };
        let mut names: Option<Vec<String>> = None;
        let tys: Vec<Ty> = match ret {
            Ty::Void => {
                notes.push(skip("it returns nothing; a kernel gives a result"));
                continue;
            }
            Ty::Record(n) if n.starts_with("viv__ret_") => records.get(n).map(|r| r.iter().map(|f| f.1).collect()).unwrap_or_default(),
            // a record of the program's: its fields come back by name
            Ty::Record(n) if records.contains_key(n) => {
                let r = &records[n];
                if let Some(f) = r.iter().find(|f| f.0 == "error_kind" || PY_RESERVED.contains(&f.0.as_str())) {
                    notes.push(skip(&format!("its result record `{n}` has a field `{}`, a name the wrapper reserves", f.0)));
                    continue;
                }
                names = Some(r.iter().map(|f| f.0.clone()).collect());
                k.rec_result = Some(n.to_string());
                r.iter().map(|f| f.1).collect()
            }
            t => vec![t],
        };
        let kinds: Option<Vec<Kind>> = tys.iter().map(|t| kind_of_ty(*t)).collect();
        let Some(kinds) = kinds.filter(|k| !k.is_empty()) else {
            let words: Vec<String> = tys.iter().map(|t| ty_word(*t)).collect();
            notes.push(skip(&format!("it returns {}; a kernel's results are Num, Int, Bool, Complex, Arr or [Int]", words.join(", "))));
            continue;
        };
        let several = kinds.len() > 1;
        k.results = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| crate::export::KResult { name: match &names { Some(n) => n[i].clone(), None if several => format!("result{i}"), None => "result".into() }, kind, doc: None })
            .collect();
        keep.push(k);
    }
    m.kernels = keep;
}

// ---------------------------------------------------------------- stubs

/// A kernel argument's type in a stub.
fn py_arg_type(k: &Kind) -> &'static str {
    match k {
        Kind::Num => "float",
        Kind::Int => "int",
        Kind::Bool => "bool",
        Kind::Cplx => "complex",
        _ => "npt.ArrayLike",
    }
}

/// A kernel result's type in a stub.
fn py_res_type(k: &Kind) -> &'static str {
    match k {
        Kind::Num => "float",
        Kind::Int => "int",
        Kind::Bool => "bool",
        Kind::Cplx => "complex",
        Kind::IntArr => "npt.NDArray[np.int64]",
        _ => "npt.NDArray[np.float64]",
    }
}

/// `<name>.pyi`: the collection's type stub, a signature and docstring per
/// kernel, for editors and type checkers.
pub fn stub(m: &Manifest, about: &str, version: &str) -> String {
    let q3 = "\"\"\"";
    let mut s = String::new();
    let _ = writeln!(s, "# {}.pyi -- written by `viv export`; do not edit. {version}", m.name);
    if !about.is_empty() {
        let _ = writeln!(s, "{q3}{}{q3}", about.replace(q3, "'''"));
    }
    let _ = writeln!(s, "from typing import Any, Dict, List, Mapping, Tuple\n\nimport numpy as np\nimport numpy.typing as npt\n");
    let names: Vec<String> = m.kernels.iter().map(|k| format!("{:?}", k.name)).collect();
    let _ = writeln!(s, "__all__ = [{}]\n", names.join(", "));
    let _ = writeln!(s, "NAME: str\nVERSION: str\n");
    let errors = [
        ("VivError", "RuntimeError"),
        ("VivIndexError", "VivError, IndexError"),
        ("VivValueError", "VivError, ValueError"),
        ("VivOverflowError", "VivError, OverflowError"),
        ("VivZeroDivisionError", "VivError, ZeroDivisionError"),
        ("VivRecursionError", "VivError, RecursionError"),
        ("VivMemoryError", "VivError, MemoryError"),
        ("VivStop", "VivError"),
    ];
    for (c, base) in errors {
        let _ = writeln!(s, "class {c}({base}):\n    kind: str\n");
    }
    let _ = writeln!(s, "def version() -> str: ...\ndef tests() -> List[Tuple[str, bool, str]]: ...\n");
    for k in &m.kernels {
        let args: Vec<String> = def_params(k).iter().map(|(n, ai)| format!("{n}: {}", ai.map_or("Mapping[str, npt.ArrayLike]", |i| py_arg_type(&k.args[i].kind)))).collect();
        let ret = if k.rec_result.is_some() {
            "Dict[str, npt.NDArray[Any]]".to_string()
        } else if k.results.len() == 1 {
            py_res_type(&k.results[0].kind).to_string()
        } else {
            format!("Tuple[{}]", k.results.iter().map(|r| py_res_type(&r.kind)).collect::<Vec<_>>().join(", "))
        };
        let doc = kernel_doc(k).replace(q3, "'''").replace('\\', "\\\\");
        let _ = writeln!(s, "def {}({}) -> {ret}:\n    {q3}{}    {q3}\n    ...\n", k.name, args.join(", "), doc);
    }
    s
}

// ---------------------------------------------------------------- borrowing

/// The lending of each kernel's arguments, read again over the whole
/// program (every imported module flattened in, so a call into
/// lib/minpack is followed rather than taken as unknown). `resolve` read
/// each kernel's own file only.
pub fn lend_over(m: &Manifest, defs: &mut [KDef], whole: &Program) {
    for (k, d) in m.kernels.iter().zip(defs.iter_mut()) {
        let name = whole.items.iter().find_map(|i| match i {
            Item::Def(x) if x.name == d.cname || crate::lower::def_symbol(&x.name) == d.cname => Some(x.name.clone()),
            _ => None,
        });
        let Some(name) = name else { continue };
        let lend = borrowable(whole, &name);
        let at = def_index(k);
        for (i, a) in k.args.iter().enumerate() {
            if is_arr(&a.kind) {
                d.borrow[i] = d.borrow[i] || lend.get(at[i]).copied().unwrap_or(false);
            }
        }
    }
}

/// Which parameters of `def` the entry may lend the caller's buffer instead
/// of copying it: those whose arrays
/// nothing in the program can write. Arrays alias in Viv (`b = a` shares
/// storage, a def writes its caller's array), so a parameter is lent only
/// when no name that may hold its storage is written in place.
///
/// The names that may hold a parameter's storage are found by a fixed point
/// over the def's assignments: the parameter, then every variable assigned
/// (or bound by a loop) from an expression that mentions one of them. That
/// over-approximates, since the array builtins copy (`reshape`, `rows`,
/// slices). A parameter is not lent when one of those names is
///
/// - written in place: an element or field write, a builtin that writes
///   that argument (`sort`, `fill`, `fft`), or an argument of a def of the
///   file at a position that def does not lend (a fixed point over the
///   call graph);
/// - passed to anything else that is not a builtin known to write nothing:
///   a def of another module, a function value;
/// - assigned to a top-level name, which outlives the call.
///
/// Rebinding a parameter (`a = a * 2`, `a = append(a, x)`, which copies a
/// parameter's storage first, crate::growable) leaves the buffer alone.
pub fn borrowable(prog: &Program, def: &str) -> Vec<bool> {
    use crate::ast::{Def, Expr, Stmt};
    use std::collections::HashSet;

    let defs: Vec<&Def> = prog.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d) } else { None }).collect();
    let globals: HashSet<&str> = prog.items.iter().filter_map(|i| if let Item::Stmt(Stmt::Assign { name, .. }) = i { Some(name.as_str()) } else { None }).collect();

    fn mentions(e: &Expr, names: &HashSet<String>) -> bool {
        match e {
            Expr::Var { name, .. } => names.contains(name),
            Expr::ArrLit { elems, .. } => elems.iter().any(|x| mentions(x, names)),
            Expr::Index { base, indices, .. } => mentions(base, names) || indices.iter().any(|x| mentions(x, names)),
            Expr::Lambda { body, .. } => mentions(body, names),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => mentions(operand, names),
            Expr::Binary { lhs, rhs, .. } => mentions(lhs, names) || mentions(rhs, names),
            Expr::Call { args, .. } => args.iter().any(|x| mentions(x, names)),
            Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| mentions(x, names)),
            Expr::Field { base, .. } => mentions(base, names),
            _ => false,
        }
    }
    fn root(e: &Expr) -> Option<&str> {
        match e {
            Expr::Var { name, .. } => Some(name),
            Expr::Field { base, .. } | Expr::Index { base, .. } => root(base),
            _ => None,
        }
    }
    // the names that may hold `start`'s storage, by a fixed point
    fn aliases(body: &[Stmt], start: &str) -> HashSet<String> {
        fn pass(body: &[Stmt], t: &mut HashSet<String>) -> bool {
            let mut grew = false;
            for s in body {
                match s {
                    Stmt::Assign { name, value, .. } if !t.contains(name) && mentions(value, t) => {
                        t.insert(name.clone());
                        grew = true;
                    }
                    Stmt::For { var, bound, body, .. } => {
                        if !t.contains(var) && mentions(bound, t) {
                            t.insert(var.clone());
                            grew = true;
                        }
                        grew |= pass(body, t);
                    }
                    Stmt::If { then_body, else_body, .. } => {
                        grew |= pass(then_body, t);
                        grew |= pass(else_body, t);
                    }
                    Stmt::While { body, .. } => grew |= pass(body, t),
                    _ => {}
                }
            }
            grew
        }
        let mut t = HashSet::from([start.to_string()]);
        while pass(body, &mut t) {}
        t
    }
    // whether an expression writes, or hands where it cannot follow, a name of `t`
    fn expr_bad(e: &Expr, t: &HashSet<String>, lent: &HashMap<String, Vec<bool>>) -> bool {
        match e {
            Expr::Call { callee, args, .. } => {
                if args.iter().any(|a| expr_bad(a, t, lent)) {
                    return true;
                }
                if let Some(lend) = lent.get(callee) {
                    // a def of this file: an argument it does not lend
                    return args.iter().enumerate().any(|(k, a)| !lend.get(k).copied().unwrap_or(false) && mentions(a, t));
                }
                let written = crate::builtins::written_args(callee, args.len());
                if written.iter().any(|&k| args.get(k).and_then(root).is_some_and(|r| t.contains(r))) {
                    return true;
                }
                let known = crate::builtins::is_pure(callee) || crate::builtins::EFFECTS.contains(&callee.as_str()) || !written.is_empty();
                !known && args.iter().any(|a| mentions(a, t))
            }
            Expr::ArrLit { elems, .. } => elems.iter().any(|x| expr_bad(x, t, lent)),
            Expr::Index { base, indices, .. } => expr_bad(base, t, lent) || indices.iter().any(|x| expr_bad(x, t, lent)),
            Expr::Lambda { body, .. } => expr_bad(body, t, lent),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => expr_bad(operand, t, lent),
            Expr::Binary { lhs, rhs, .. } => expr_bad(lhs, t, lent) || expr_bad(rhs, t, lent),
            Expr::RecordLit { fields, .. } => fields.iter().any(|(_, _, x)| expr_bad(x, t, lent)),
            Expr::Field { base, .. } => expr_bad(base, t, lent),
            _ => false,
        }
    }
    fn stmts_bad(body: &[Stmt], t: &HashSet<String>, lent: &HashMap<String, Vec<bool>>, globals: &HashSet<&str>) -> bool {
        body.iter().any(|s| match s {
            Stmt::Assign { name, value, .. } => (t.contains(name) && globals.contains(name.as_str())) || expr_bad(value, t, lent),
            Stmt::SetIndex { base, indices, value, .. } => root(base).is_some_and(|r| t.contains(r)) || expr_bad(base, t, lent) || indices.iter().any(|x| expr_bad(x, t, lent)) || expr_bad(value, t, lent),
            Stmt::SetField { base, value, .. } => root(base).is_some_and(|r| t.contains(r)) || expr_bad(base, t, lent) || expr_bad(value, t, lent),
            Stmt::Return { value, .. } | Stmt::ExprStmt { expr: value, .. } | Stmt::Show { expr: value, .. } => expr_bad(value, t, lent),
            Stmt::If { cond, then_body, else_body, .. } => expr_bad(cond, t, lent) || stmts_bad(then_body, t, lent, globals) || stmts_bad(else_body, t, lent, globals),
            Stmt::While { cond, body, .. } => expr_bad(cond, t, lent) || stmts_bad(body, t, lent, globals),
            Stmt::For { var, bound, body, .. } => (t.contains(var) && globals.contains(var.as_str())) || expr_bad(bound, t, lent) || stmts_bad(body, t, lent, globals),
            _ => false,
        })
    }
    // per def, per parameter: lent. Start from "every parameter lent" and
    // take away until nothing changes (a greatest fixed point, so recursion
    // that only reads stays lent).
    let mut lend: HashMap<String, Vec<bool>> = defs.iter().map(|d| (d.name.clone(), vec![true; d.params.len()])).collect();
    loop {
        let mut changed = false;
        for d in &defs {
            for (k, p) in d.params.iter().enumerate() {
                if !lend[&d.name][k] {
                    continue;
                }
                let t = aliases(&d.body, &p.name);
                if stmts_bad(&d.body, &t, &lend, &globals) {
                    lend.get_mut(&d.name).unwrap()[k] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    lend.remove(def).unwrap_or_default()
}
