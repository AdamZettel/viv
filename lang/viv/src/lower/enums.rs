// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Enums, `match` and optionals in the checked lowering and the C backend.
//!
//! An enum value is a pointer to an immutable node: a tag, then the payload
//! of its variant. In C, `struct viv_en_<Name>` per enum, written out per
//! program (`CItem::Decls`), with a constructor per variant, a keep walker
//! for the arena and a text writer; in wasm, a block of 8-byte slots read
//! through a descriptor (`wasm_backend`, `interp/wasm_rt.rs`). A `match` reaches
//! the lowering as the `if` chain the parser wrote: `viv__enum_is`,
//! `viv__enum_field` and `viv__match_rest`, which checks that the arms cover
//! every variant. A program with no enum reaches none of this.

use super::*;
use std::rc::Rc;

/// A declared enum, or an optional (`Int?`), as the lowering knows it.
#[derive(Clone, Debug, PartialEq)]
pub struct EnumInfo {
    pub name: &'static str,
    /// each variant's name and payload types, in declaration order (the tag
    /// is the position)
    pub variants: Vec<(String, Vec<Ty>)>,
}

impl EnumInfo {
    pub fn tag(&self, variant: &str) -> Option<usize> {
        self.variants.iter().position(|(v, _)| v == variant)
    }
    /// No variant has a payload: the enum is a set of names, and `==` compares them.
    pub fn plain(&self) -> bool {
        self.variants.iter().all(|(_, p)| p.is_empty())
    }
}

thread_local! {
    /// The enums of the program being lowered, and the optionals it uses.
    static ENUMS: std::cell::RefCell<HashMap<String, Rc<EnumInfo>>> = std::cell::RefCell::new(HashMap::new());
}

/// The enum (or optional) named `name`.
pub fn info(name: &str) -> Option<Rc<EnumInfo>> {
    ENUMS.with(|e| e.borrow().get(name).cloned())
}

/// Every enum of the program being lowered, by name.
pub fn all() -> Vec<Rc<EnumInfo>> {
    let mut v: Vec<Rc<EnumInfo>> = ENUMS.with(|e| e.borrow().values().cloned().collect());
    v.sort_by(|a, b| a.name.cmp(b.name));
    v
}

/// Whether `name` is a type the program declares with `enum`, or an optional.
pub fn is_enum(name: &str) -> bool {
    ENUMS.with(|e| e.borrow().contains_key(name))
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// The types a payload may hold.
fn payload_ok(t: Ty) -> bool {
    matches!(t, Ty::Num | Ty::Int | Ty::Bool | Ty::Str | Ty::Enum(_))
}

/// `T?`, registered on first use: `None`, then `Some(T)`.
pub fn opt(t: Ty) -> Ty {
    let name = format!("{}?", t.name());
    if let Some(i) = info(&name) {
        return Ty::Enum(i.name);
    }
    let n = leak(&name);
    ENUMS.with(|e| e.borrow_mut().insert(name, Rc::new(EnumInfo { name: n, variants: vec![("None".to_string(), vec![]), ("Some".to_string(), vec![t])] })));
    Ty::Enum(n)
}

/// Start lowering `program`: its enum types, by name, so record fields and
/// annotations can name them; `resolve` fills in their variants.
pub(super) fn begin(program: &Program) -> Vec<Diagnostic> {
    ENUMS.with(|e| e.borrow_mut().clear());
    let mut errs = Vec::new();
    for item in &program.items {
        let Item::Enum(d) = item else { continue };
        if is_enum(&d.name) {
            errs.push(Diagnostic::error(format!("the enum `{}` is already defined", d.name)).with_primary(d.name_span, "defined again here"));
            continue;
        }
        let n = leak(&d.name);
        ENUMS.with(|e| e.borrow_mut().insert(d.name.clone(), Rc::new(EnumInfo { name: n, variants: Vec::new() })));
    }
    errs
}

/// The variants of `program`'s enums, their payloads resolved (an enum may
/// hold itself, or another declared before or after it).
pub(super) fn resolve(program: &Program, records: &HashMap<String, RecordInfo>) -> Vec<Diagnostic> {
    let mut errs = Vec::new();
    for item in &program.items {
        let Item::Enum(d) = item else { continue };
        if records.contains_key(&d.name) {
            errs.push(Diagnostic::error(format!("`{}` is a record and an enum", d.name)).with_primary(d.name_span, "one type per name"));
            continue;
        }
        let mut variants = Vec::new();
        for v in &d.variants {
            let mut payload = Vec::new();
            for t in &v.payload {
                match resolve_type(t, records) {
                    Ok(t) if payload_ok(t) => payload.push(t),
                    Ok(t) => errs.push(
                        Diagnostic::error(format!("a variant's payload cannot be {} yet", art(t)))
                            .with_primary(v.name_span, format!("`{}` holds {}", v.name, art(t)))
                            .with_note("a payload is a number, an Int, a boolean, a string, an enum or an optional"),
                    ),
                    Err(e) => errs.push(e),
                }
            }
            variants.push((v.name.clone(), payload));
        }
        if let Some(i) = info(&d.name) {
            ENUMS.with(|e| e.borrow_mut().insert(d.name.clone(), Rc::new(EnumInfo { name: i.name, variants })));
        }
    }
    errs
}

/// The C name of an enum type: its name, with `?` spelled `_opt`.
pub fn c_name(name: &str) -> String {
    name.replace('?', "_opt")
}

// ---- inference ----------------------------------------------------------------

/// The type of a call that builds or reads an enum, or `None` when `callee`
/// is not one (`Some(None)`: one, not typed yet).
pub(super) fn infer_call(callee: &str, args: &[Expr], arg_tys: &[Option<Ty>], sigs: &Sigs) -> Option<Option<Ty>> {
    if sigs.arity.contains_key(callee) {
        return None;
    }
    if let Some((e, _)) = callee.split_once('.') {
        if let Some(i) = info(e) {
            return Some(Some(Ty::Enum(i.name)));
        }
    }
    Some(match callee {
        "Some" if args.len() == 1 => arg_tys[0].map(opt),
        "None" if args.is_empty() => None,
        c if c == crate::parser::ENUM_IS => Some(Ty::Bool),
        c if c == crate::parser::MATCH_REST => Some(Ty::Num),
        c if c == crate::parser::ENUM_FIELD => {
            let (Some(Some(Ty::Enum(e))), Some(Expr::Str { value: v, .. }), Some(Expr::Num { value: k, .. })) = (arg_tys.first(), args.get(1), args.get(2)) else {
                return Some(None);
            };
            info(e).and_then(|i| i.tag(v).map(|t| (i, t))).and_then(|(i, t)| i.variants[t].1.get(*k as usize).copied())
        }
        _ => return None,
    })
}

// ---- lowering -----------------------------------------------------------------

/// `e` lowered for a binding of type `want` when `e` builds an optional:
/// `None` takes the optional's type, and `Some(x)` passes the payload's type
/// down (so `Some(3)` is an `Int?` where one is bound).
pub(super) fn lower_for(prog: &mut IrProgram<Viv>, e: &Expr, want: Option<Ty>, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    let Some(Ty::Enum(name)) = want else { return None };
    let Expr::Call { callee, args, span } = e else { return None };
    if sigs.arity.contains_key(callee.as_str()) || !name.ends_with('?') {
        return None;
    }
    let i = info(name)?;
    match (callee.as_str(), args.as_slice()) {
        ("None", []) => Some(Ok((prog.add(Viv::EnumNew { ty: name.to_string(), tag: 0 }, *span, vec![]), Ty::Enum(name)))),
        ("Some", [x]) => {
            let t = i.variants[1].1[0];
            Some(strings::lower_expected(prog, x, Some(t), scope, sigs).and_then(|(id, xt)| {
                if xt != t {
                    return Err(Diagnostic::error(format!("this optional holds {} and `Some` is given {}", art(t), art(xt))).with_primary(x.span(), format!("this is {}", art(xt))));
                }
                Ok((prog.add(Viv::EnumNew { ty: name.to_string(), tag: 1 }, *span, vec![id]), Ty::Enum(name)))
            }))
        }
        _ => None,
    }
}

/// The calls that build and take apart enum values, or `None` when `callee`
/// is not one.
pub(super) fn lower_call(prog: &mut IrProgram<Viv>, callee: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Option<Result<(NodeId, Ty), Diagnostic>> {
    if sigs.arity.contains_key(callee) {
        return None;
    }
    if let Some((e, v)) = callee.split_once('.') {
        let i = info(e)?;
        return Some(construct(prog, &i, v, args, span, scope, sigs));
    }
    let r = (|| -> Result<Option<(NodeId, Ty)>, Diagnostic> {
        Ok(Some(match callee {
            "Some" => {
                let [x] = args else { return Err(Diagnostic::error("`Some` takes one value").with_primary(span, "wrong number of arguments")) };
                let (id, t) = lower_expr(prog, x, scope, sigs)?;
                if !payload_ok(t) {
                    return Err(Diagnostic::error(format!("an optional cannot hold {} yet", art(t))).with_primary(x.span(), format!("this is {}", art(t))));
                }
                let ty = opt(t);
                (prog.add(Viv::EnumNew { ty: ty.name().to_string(), tag: 1 }, span, vec![id]), ty)
            }
            "None" if args.is_empty() => {
                return Err(Diagnostic::error("`None` of what type? An optional's type comes from where it goes")
                    .with_primary(span, "no optional type here")
                    .with_note("write the type where it is bound: `x: Int? = None`, a parameter `x: Int?`, or `-> Int?` on the def"));
            }
            c if c == crate::parser::ENUM_IS => {
                let (m, i, tag) = variant_of(prog, args, span, scope, sigs)?;
                let _ = i;
                (prog.add(Viv::EnumIs { tag: tag as u32 }, span, vec![m]), Ty::Bool)
            }
            c if c == crate::parser::ENUM_FIELD => {
                let (m, i, tag) = variant_of(prog, args, span, scope, sigs)?;
                let Some(Expr::Num { value: k, .. }) = args.get(2) else { return Err(Diagnostic::error("internal: a payload field without its position").with_primary(span, "here")) };
                let k = *k as usize;
                let Some(&t) = i.variants[tag].1.get(k) else {
                    let (v, p) = &i.variants[tag];
                    return Err(Diagnostic::error(format!("`{v}` holds {} value{}, and this pattern names more", p.len(), if p.len() == 1 { "" } else { "s" }))
                        .with_primary(args[1].span(), "too many names for its payload"));
                };
                (prog.add(Viv::EnumField { ty: i.name.to_string(), tag: tag as u32, k: k as u32 }, span, vec![m]), t)
            }
            _ => return Ok(None),
        }))
    })();
    match r {
        Ok(Some(x)) => Some(Ok(x)),
        Ok(None) => None,
        Err(d) => Some(Err(d)),
    }
}

/// `viv__enum_is(m, "V")` and `viv__enum_field(m, "V", k)`: the scrutinee,
/// its enum, and the tag of `V`, which must be one of its variants.
fn variant_of(prog: &mut IrProgram<Viv>, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Rc<EnumInfo>, usize), Diagnostic> {
    let (Some(m), Some(Expr::Str { value: v, span: vspan })) = (args.first(), args.get(1)) else {
        return Err(Diagnostic::error("internal: a match without its scrutinee").with_primary(span, "here"));
    };
    let (id, t) = lower_expr(prog, m, scope, sigs)?;
    let Ty::Enum(name) = t else {
        return Err(Diagnostic::error(format!("`match` takes an enum or an optional apart, and this is {}", art(t))).with_primary(m.span(), format!("this is {}", art(t))));
    };
    let i = info(name).expect("a lowered enum type is registered");
    let tag = i.tag(v).ok_or_else(|| {
        let names: Vec<&str> = i.variants.iter().map(|(n, _)| n.as_str()).collect();
        Diagnostic::error(format!("`{name}` has no variant `{v}`")).with_primary(*vspan, "not one of its variants").with_note(format!("its variants are {}", names.join(", ")))
    })?;
    Ok((id, i, tag))
}

/// `Name.V(args)`: the payload checked against the variant's, each value
/// lowered for its slot (a literal is an Int in an Int slot).
fn construct(prog: &mut IrProgram<Viv>, i: &EnumInfo, v: &str, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<(NodeId, Ty), Diagnostic> {
    let Some(tag) = i.tag(v) else {
        let names: Vec<&str> = i.variants.iter().map(|(n, _)| n.as_str()).collect();
        return Err(Diagnostic::error(format!("`{}` has no variant `{v}`", i.name)).with_primary(span, "not one of its variants").with_note(format!("its variants are {}", names.join(", "))));
    };
    let payload = &i.variants[tag].1;
    if args.len() != payload.len() {
        return Err(Diagnostic::error(format!("`{}.{v}` holds {} value{}, and {} {} given", i.name, payload.len(), if payload.len() == 1 { "" } else { "s" }, args.len(), if args.len() == 1 { "is" } else { "are" }))
            .with_primary(span, "wrong number of values"));
    }
    let mut ids = Vec::with_capacity(args.len());
    for (a, &want) in args.iter().zip(payload) {
        let (id, t) = strings::lower_expected(prog, a, Some(want), scope, sigs)?;
        if t != want {
            return Err(Diagnostic::error(format!("`{}.{v}` holds {} here, and this is {}", i.name, art(want), art(t))).with_primary(a.span(), format!("this is {}", art(t))));
        }
        ids.push(id);
    }
    Ok((prog.add(Viv::EnumNew { ty: i.name.to_string(), tag: tag as u32 }, span, ids), Ty::Enum(i.name)))
}

/// `viv__match_rest(m, "A|B")` at the end of a `match` with no `_`: every
/// variant must be covered. The statement can then never run; it stops the
/// program in case a later pass breaks that.
pub(super) fn lower_rest(prog: &mut IrProgram<Viv>, args: &[Expr], span: Span, scope: &Scope<'_>, sigs: &Sigs) -> Result<NodeId, Diagnostic> {
    let (Some(m), Some(Expr::Str { value: covered, .. })) = (args.first(), args.get(1)) else {
        return Err(Diagnostic::error("internal: a match rest without its scrutinee").with_primary(span, "here"));
    };
    let (_, t) = lower_expr(prog, m, scope, sigs)?;
    let Ty::Enum(name) = t else {
        return Err(Diagnostic::error(format!("`match` takes an enum or an optional apart, and this is {}", art(t))).with_primary(m.span(), format!("this is {}", art(t))));
    };
    let i = info(name).expect("a lowered enum type is registered");
    let covered: Vec<&str> = covered.split('|').collect();
    let missing: Vec<&str> = i.variants.iter().map(|(v, _)| v.as_str()).filter(|v| !covered.contains(v)).collect();
    if !missing.is_empty() {
        let list = missing.iter().map(|v| format!("`{v}`")).collect::<Vec<_>>().join(", ");
        return Err(Diagnostic::error(format!("this `match` does not cover {list}"))
            .with_primary(span, format!("a value of `{name}` can be {list}, and no arm takes it"))
            .with_note("add an arm for each, or end with `_ => ...`"));
    }
    Ok(prog.add(Viv::Stop { message: "internal: no arm of this match matched".to_string(), at: site_at(span), kind: crate::errkind::ErrKind::Error }, span, vec![]))
}

/// `a == b` and `a != b` of two values of one enum with no payloads.
pub(super) fn lower_eq(prog: &mut IrProgram<Viv>, op: BinOp, l: (NodeId, Ty, Span), r: (NodeId, Ty, Span), span: Span) -> Result<(NodeId, Ty), Diagnostic> {
    let (lid, lty, _) = l;
    let (rid, rty, rspan) = r;
    if lty != rty {
        return Err(Diagnostic::error(format!("cannot compare {} and {}", art(lty), art(rty))).with_primary(span, "two different types"));
    }
    let Ty::Enum(name) = lty else { unreachable!("an enum") };
    if !matches!(op, BinOp::EqEq | BinOp::Ne) {
        return Err(Diagnostic::error(format!("enum values are compared with `==` and `!=` only, and this is `{}`", op.c_op())).with_primary(span, "not an equality"));
    }
    let i = info(name).expect("registered");
    if !i.plain() {
        return Err(Diagnostic::error(format!("`==` compares enums without payloads, and `{name}` has them"))
            .with_primary(rspan, "a value with a payload")
            .with_note("take the values apart with `match`"));
    }
    Ok((prog.add(Viv::EnumEq { ne: op == BinOp::Ne }, span, vec![lid, rid]), Ty::Bool))
}

/// An enum value as text (`print`, `str`, `show`).
pub(super) fn text(prog: &mut IrProgram<Viv>, id: NodeId, ty: Ty, span: Span) -> NodeId {
    prog.add(Viv::EnumStr { ty: ty.name().to_string() }, span, vec![id])
}

// ---- the C backend ------------------------------------------------------------

/// The C type of a payload slot.
fn c_slot(t: Ty) -> String {
    match t {
        Ty::Num | Ty::Bool => "double".into(),
        Ty::Int => "long long".into(),
        Ty::Str => "viv_str".into(),
        Ty::Enum(n) => format!("const struct viv_en_{}*", c_name(n)),
        _ => unreachable!("payload_ok"),
    }
}

/// The C of the program's enums: each a tagged struct that begins with the
/// arena's node head (`viv_en_head`), a constructor per variant (a variant
/// with no payload is one static node), the keep walker the arena calls and
/// the text writer. Both walk the last field of a variant in a loop when it
/// is of the same enum, so a list a million nodes long is not a million calls
/// deep.
pub(super) fn c_decls(defs: &[(String, Vec<(String, Vec<Ty>)>)]) -> String {
    let enums: Vec<EnumInfo> = defs.iter().map(|(n, vs)| EnumInfo { name: leak(n), variants: vs.clone() }).collect();
    let mut c = String::from("/* the program's enums */\n");
    for e in &enums {
        let m = c_name(e.name);
        c.push_str(&format!("struct viv_en_{m} {{\n    int tag;\n    unsigned keep;\n    long pos;\n    union {{\n        char none_;\n"));
        for (k, (_, p)) in e.variants.iter().enumerate() {
            if p.is_empty() {
                continue;
            }
            c.push_str("        struct {");
            for (j, t) in p.iter().enumerate() {
                c.push_str(&format!(" {} f{j};", c_slot(*t)));
            }
            c.push_str(&format!(" }} v{k};\n"));
        }
        c.push_str("    } u;\n};\n");
    }
    for e in &enums {
        let m = c_name(e.name);
        c.push_str(&format!("static void viv_keep_en_{m}(const struct viv_en_{m}* p, size_t mark);\nstatic void viv_en_put_{m}(viv_ebuf* b, const struct viv_en_{m}* p);\n"));
    }
    // the last field of a variant, when it holds the same enum: walked in a loop
    let tail = |e: &EnumInfo, p: &[Ty]| p.last().is_some_and(|t| *t == Ty::Enum(e.name));
    for e in &enums {
        let m = c_name(e.name);
        for (k, (_, p)) in e.variants.iter().enumerate() {
            if p.is_empty() {
                c.push_str(&format!(
                    "static struct viv_en_{m} viv_en_{m}_s{k} = {{ {k}, 0, 0, {{ 0 }} }};\nstatic const struct viv_en_{m}* viv_mk_{m}_{k}(void) {{ return &viv_en_{m}_s{k}; }}\n"
                ));
                continue;
            }
            let params: Vec<String> = p.iter().enumerate().map(|(j, t)| format!("{} a{j}", c_slot(*t))).collect();
            c.push_str(&format!("static const struct viv_en_{m}* viv_mk_{m}_{k}({}) {{\n    struct viv_en_{m}* p = (struct viv_en_{m}*)viv_en_alloc(sizeof *p);\n    p->tag = {k};\n", params.join(", ")));
            for j in 0..p.len() {
                c.push_str(&format!("    p->u.v{k}.f{j} = a{j};\n"));
            }
            c.push_str("    return p;\n}\n");
            for (j, t) in p.iter().enumerate() {
                c.push_str(&format!("static {} viv_get_{m}_{k}_{j}(const struct viv_en_{m}* p) {{ return p->u.v{k}.f{j}; }}\n", c_slot(*t)));
            }
        }
        // the keep walker: marks each node with the pass, stops at a marked
        // node and at one below the mark (owned by an older region)
        c.push_str(&format!(
            "static void viv_keep_en_{m}(const struct viv_en_{m}* p, size_t mark) {{\n    while (p && p->pos >= (long)mark && p->keep != viv_en_pass) {{\n        ((struct viv_en_{m}*)p)->keep = viv_en_pass;\n        switch (p->tag) {{\n"
        ));
        for (k, (_, p)) in e.variants.iter().enumerate() {
            let last_same = tail(e, p);
            let mut body = String::new();
            for (j, t) in p.iter().enumerate() {
                if last_same && j + 1 == p.len() {
                    continue;
                }
                match t {
                    Ty::Str => body.push_str(&format!(" viv_keep_str(p->u.v{k}.f{j});")),
                    Ty::Enum(n) => body.push_str(&format!(" viv_keep_en_{}(p->u.v{k}.f{j}, mark);", c_name(n))),
                    _ => {}
                }
            }
            if last_same {
                body.push_str(&format!(" p = p->u.v{k}.f{}; continue;", p.len() - 1));
            }
            if !body.is_empty() {
                c.push_str(&format!("        case {k}:{body} break;\n"));
            }
        }
        c.push_str("        default: break;\n        }\n        return;\n    }\n}\n");
        // the text writer
        c.push_str(&format!("static void viv_en_put_{m}(viv_ebuf* b, const struct viv_en_{m}* p) {{\n    long close = 0;\n    for (;;) {{\n        switch (p->tag) {{\n"));
        for (k, (v, p)) in e.variants.iter().enumerate() {
            c.push_str(&format!("        case {k}:\n            viv_ebuf_put(b, \"{v}\");\n"));
            if !p.is_empty() {
                c.push_str("            viv_ebuf_put(b, \"(\");\n");
                for (j, t) in p.iter().enumerate() {
                    if j > 0 {
                        c.push_str("            viv_ebuf_put(b, \", \");\n");
                    }
                    let f = format!("p->u.v{k}.f{j}");
                    if tail(e, p) && j + 1 == p.len() {
                        c.push_str(&format!("            p = {f};\n            close++;\n            continue;\n"));
                        continue;
                    }
                    c.push_str(&match t {
                        Ty::Num => format!("            viv_ebuf_put(b, viv_g({f}));\n"),
                        Ty::Bool => format!("            viv_ebuf_put(b, {f} != 0.0 ? \"true\" : \"false\");\n"),
                        Ty::Int => format!("            viv_ebuf_int(b, {f});\n"),
                        Ty::Str => format!("            viv_ebuf_repr(b, {f});\n"),
                        Ty::Enum(n) => format!("            viv_en_put_{}(b, {f});\n", c_name(n)),
                        _ => unreachable!("payload_ok"),
                    });
                }
                if !tail(e, p) {
                    c.push_str("            viv_ebuf_put(b, \")\");\n");
                }
            }
            c.push_str("            break;\n");
        }
        c.push_str("        }\n        break;\n    }\n    while (close-- > 0) viv_ebuf_put(b, \")\");\n}\n");
        c.push_str(&format!(
            "static viv_str viv_en_str_{m}(const struct viv_en_{m}* p) {{\n    viv_ebuf b = {{ 0, 0, 0 }};\n    viv_en_put_{m}(&b, p);\n    viv_str r = viv_s_make(b.p ? b.p : \"\", b.n, 0);\n    free(b.p);\n    return r;\n}}\n"
        ));
    }
    c
}

/// What every program with enums shares: a node allocated and filed on the
/// arena with tag 3 (its head says where), and the text buffer.
pub(super) fn fragment() -> RuntimeFragment {
    RuntimeFragment {
        name: "enums",
        includes: &["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <unistd.h>"],
        deps: &["arena", "arena_keep", "str_core", "str_repr", "fmt_g"],
        code: r#"/* enum nodes: immutable, filed on the arena */
static void* viv_en_alloc(size_t bytes) {
    viv_en_head* p = (viv_en_head*)calloc(1, bytes);
    if (!p) { viv_fflush(stdout); viv_errf("error: out of memory making an enum value\n"); viv_stop(VIV_E_MEMORY); }
    p->pos = (long)viv_arena_top;
    viv_arena_push((char*)p + 3);
    return p;
}
/* every node begins with its tag */
static int viv_en_tag(const void* p) { return *(const int*)p; }
typedef struct { char* p; long n, cap; } viv_ebuf;
static void viv_ebuf_putn(viv_ebuf* b, const char* s, long n) {
    if (b->n + n + 1 > b->cap) {
        long cap = b->cap ? 2 * b->cap : 64;
        while (cap < b->n + n + 1) cap *= 2;
        char* q = (char*)realloc(b->p, (size_t)cap);
        if (!q) { viv_fflush(stdout); viv_errf("error: out of memory writing an enum value\n"); viv_stop(VIV_E_MEMORY); }
        b->p = q;
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, (size_t)n);
    b->n += n;
}
static void viv_ebuf_put(viv_ebuf* b, const char* s) { viv_ebuf_putn(b, s, (long)strlen(s)); }
static void viv_ebuf_int(viv_ebuf* b, long long v) { char t[24]; int n = snprintf(t, sizeof t, "%lld", v); viv_ebuf_putn(b, t, n); }
static void viv_ebuf_repr(viv_ebuf* b, viv_str s) { viv_str r = VIV_S(viv_s_repr(s, 0)); viv_ebuf_putn(b, r->s, r->len); }"#,
    }
}

/// The C call that builds variant `tag` of `ty`.
pub(super) fn c_ctor(ty: &str, tag: u32) -> String {
    format!("viv_mk_{}_{tag}", c_name(ty))
}
