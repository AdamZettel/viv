// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The memory lens, as far as it is built: what each def does with the values it is given and the values it
//! makes, inferred from the code and shown beside it, never written in it.
//!
//! `viv view <file> --lens memory [--def d] [--json]` prints, per def:
//!
//! - **in place**: the parameters whose arrays the def writes, so the caller
//!   sees the change (an element or slice write, through a record's field or
//!   an element of an array of records, or through a def or `for_leaves`
//!   that writes them), each with the first line that does it; the other
//!   parameters are only read;
//! - **makes**: the def's own variables that hold arrays, records, arrays of
//!   records, strings or maps;
//! - **writes into records**: `r.f = v` and `xs[i] = r`, each into a
//!   variable of the def (the compiler refuses one into a parameter or a
//!   global);
//! - **copies**: each place a record, or an array of records, read from a
//!   variable, a field or an element is stored somewhere else and becomes a
//!   value of its own (`Viv::Own`), with the line;
//! - **loops**: each loop whose passes can allocate, and what a pass keeps
//!   when it ends (the variables it assigns, or writes into, that hold heap
//!   values); everything else a pass made is freed then;
//! - **returns**: what the caller is handed, kept on the arena with all it
//!   holds; everything else the def made is freed when it returns.
//!
//! The facts come from the typed IR the backends emit from (copies, loops,
//! the variables' types) and from the program graph's write effects
//! (`crate::graph`), so the lens shows what the compiled code does.

use std::collections::{BTreeMap, HashMap};

use knotkit::ir::{IrProgram, NodeId};
use knotkit::span::{SourceId, Span};

use crate::ast::{Def, Item, Stmt};
use crate::lower::{Ty, Viv};

/// One def's memory facts.
#[derive(Debug)]
pub struct DefMemory {
    pub name: String,
    pub at: String,
    pub params: Vec<(String, Ty)>,
    /// parameter name -> the first line that writes into its arrays
    pub in_place: BTreeMap<String, String>,
    pub makes: Vec<(String, Ty)>,
    pub record_writes: Vec<(String, String)>,
    pub copies: Vec<(String, String)>,
    pub loops: Vec<(String, Vec<String>)>,
    pub returns: Ty,
}

impl DefMemory {
    fn new(name: String, at: String, params: Vec<(String, Ty)>, returns: Ty) -> Self {
        DefMemory { name, at, params, in_place: BTreeMap::new(), makes: Vec::new(), record_writes: Vec::new(), copies: Vec::new(), loops: Vec::new(), returns }
    }
}

/// Where a span is, as `file:line` in the file it was written in.
pub type Locate<'a> = &'a dyn Fn(Span) -> String;

/// The memory facts of every def of `src` (and `<main>`), in program order.
pub fn facts(src: &str, locate: Locate) -> Result<Vec<DefMemory>, String> {
    let ast = crate::parse(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| format!("error: {}", d.message)).collect::<Vec<_>>().join("\n"))?;
    let ir = crate::build_ir_plain(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| format!("error: {}", d.message)).collect::<Vec<_>>().join("\n"))?;
    let cx = crate::graph::Cx::of(&ast);
    let defs: HashMap<String, &Def> = ast.items.iter().filter_map(|i| if let Item::Def(d) = i { Some((crate::lower::def_symbol(&d.name), d)) } else { None }).collect();
    let mut out = Vec::new();
    for &r in ir.roots() {
        let n = ir.get(r);
        match &n.kind {
            Viv::Func { name, src: _, params, ret, locals, .. } => {
                // the defs the lowering writes (records' arithmetic) are the
                // compiler's, not the program's
                if name.starts_with("viv__") {
                    continue;
                }
                let Some(def) = defs.get(name) else { continue };
                let mut m = DefMemory::new(def.name.clone(), locate(def.name_span), params.clone(), *ret);
                m.makes = locals.iter().filter(|(n, t)| crate::builtins::heap_ty(*t) && !n.starts_with("viv__")).cloned().collect();
                if let Some(w) = cx.mutates.get(&def.name) {
                    for &k in w {
                        if let Some(p) = def.params.get(k) {
                            let line = first_write(&def.body, &p.name, &cx).map(|s| locate(s)).unwrap_or_default();
                            m.in_place.insert(p.name.clone(), line);
                        }
                    }
                }
                let types: HashMap<String, Ty> = params.iter().chain(locals.iter()).cloned().collect();
                walk(&ir, &n.children, &types, locate, &mut m);
                out.push(m);
            }
            Viv::Main { locals, globals } => {
                let mut m = DefMemory::new(crate::parser::MAIN_DEF.to_string(), String::new(), Vec::new(), Ty::Void);
                m.makes = globals.iter().chain(locals.iter()).filter(|(n, t)| crate::builtins::heap_ty(*t) && !n.starts_with("viv__")).cloned().collect();
                let types: HashMap<String, Ty> = globals.iter().chain(locals.iter()).cloned().collect();
                walk(&ir, &n.children, &types, locate, &mut m);
                out.push(m);
            }
            _ => {}
        }
    }
    Ok(out)
}

/// The first statement of `body` that writes `name`'s arrays, by the graph's
/// write effects.
fn first_write(body: &[Stmt], name: &str, cx: &crate::graph::Cx) -> Option<Span> {
    for s in body {
        let inner: Vec<&Vec<Stmt>> = match s {
            Stmt::If { then_body, else_body, .. } => vec![then_body, else_body],
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Test { body, .. } => vec![body],
            _ => vec![],
        };
        if inner.is_empty() {
            let mut e = crate::graph::Eff::default();
            crate::graph::stmt_eff(s, cx, &mut e);
            if e.writes.iter().any(|w| w == name) {
                return Some(s.span());
            }
        } else {
            for b in inner {
                if let Some(sp) = first_write(b, name, cx) {
                    return Some(sp);
                }
            }
        }
    }
    None
}

/// The text of a place node: `m.blocks[l].attn`.
fn place_text(ir: &IrProgram<Viv>, id: NodeId) -> String {
    let n = ir.get(id);
    match &n.kind {
        Viv::Var { name } => name.clone(),
        Viv::Field { field } => format!("{}.{field}", place_text(ir, n.children[0])),
        Viv::RecIndex { .. } => format!("{}[..]", place_text(ir, n.children[0])),
        _ => "a value".to_string(),
    }
}

fn walk(ir: &IrProgram<Viv>, kids: &[NodeId], types: &HashMap<String, Ty>, locate: Locate, m: &mut DefMemory) {
    for &c in kids {
        let n = ir.get(c);
        match &n.kind {
            Viv::Own { .. } => m.copies.push((locate(n.span), place_text(ir, n.children[0]))),
            Viv::SetField { field, .. } => {
                let root = crate::lower::place_var(ir, n.children[0]).unwrap_or_default();
                m.record_writes.push((locate(n.span), format!("{}.{field}, into {root}", place_text(ir, n.children[0]))));
            }
            Viv::RecSetIndex { .. } => {
                let root = crate::lower::place_var(ir, n.children[0]).unwrap_or_default();
                m.record_writes.push((locate(n.span), format!("{}[..], into {root}", place_text(ir, n.children[0]))));
            }
            Viv::For { .. } | Viv::While => {
                if allocates(ir, c) {
                    let mut names = Vec::new();
                    assigned(ir, n.children[1], &mut names);
                    let keeps = names.into_iter().filter(|v| !v.starts_with("viv__") && types.get(v).is_some_and(|t| crate::builtins::heap_ty(*t))).collect();
                    m.loops.push((locate(n.span), keeps));
                }
            }
            _ => {}
        }
        walk(ir, &n.children, types, locate, m);
    }
}

/// Whether a loop's pass can allocate: any node in it makes an array, a
/// record, a string or a map (the backends' rule, `lower::c_allocates`).
fn allocates(ir: &IrProgram<Viv>, id: NodeId) -> bool {
    let n = ir.get(id);
    let here = matches!(
        n.kind,
        Viv::ArrLit { .. } | Viv::CArrLit { .. } | Viv::Zeros | Viv::CZeros | Viv::Arr { .. } | Viv::MapMath { .. } | Viv::CArrFn { .. } | Viv::RecordLit { .. } | Viv::RecArrLit { .. } | Viv::RecAppend { .. } | Viv::Own { .. } | Viv::FnArrLit { .. } | Viv::Call { .. } | Viv::StrLit(_)
    );
    here || n.children.iter().any(|&c| allocates(ir, c))
}

/// The variables a loop body assigns or writes into, in order.
fn assigned(ir: &IrProgram<Viv>, id: NodeId, out: &mut Vec<String>) {
    let n = ir.get(id);
    let name = match &n.kind {
        Viv::Assign { name } => Some(name.clone()),
        Viv::SetField { .. } | Viv::RecSetIndex { .. } => crate::lower::place_var(ir, n.children[0]),
        _ => None,
    };
    if let Some(v) = name {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    for &c in &n.children {
        assigned(ir, c, out);
    }
}

fn ty_text(t: Ty) -> String {
    match t {
        Ty::Record(r) if r.starts_with("viv__ret_") => "several values".to_string(),
        Ty::Record(r) => r.to_string(),
        Ty::Recs(r) => format!("[{r}]"),
        Ty::Num => "number".to_string(),
        t => t.name().to_string(),
    }
}

/// The lens as text: one block per def, or only `only`'s.
pub fn text(facts: &[DefMemory], only: Option<&str>) -> String {
    let mut out = String::new();
    for m in facts.iter().filter(|m| only.map_or(true, |d| d == m.name)) {
        let params: Vec<String> = m.params.iter().map(|(n, t)| format!("{n}: {}", ty_text(*t))).collect();
        let head = if m.name == crate::parser::MAIN_DEF { m.name.clone() } else { format!("{}({}) -> {}", m.name, params.join(", "), ty_text(m.returns)) };
        out.push_str(&head);
        if !m.at.is_empty() {
            out.push_str(&format!("    {}", m.at));
        }
        out.push('\n');
        let heap_params: Vec<&(String, Ty)> = m.params.iter().filter(|(_, t)| crate::builtins::heap_ty(*t)).collect();
        for (p, line) in &m.in_place {
            out.push_str(&format!("  in place      {p}: its arrays are written, the caller sees it ({line})\n"));
        }
        let read: Vec<&str> = heap_params.iter().map(|(n, _)| n.as_str()).filter(|n| !m.in_place.contains_key(*n)).collect();
        if !read.is_empty() {
            out.push_str(&format!("  reads         {}\n", read.join(", ")));
        }
        if !m.makes.is_empty() {
            let made: Vec<String> = m.makes.iter().map(|(n, t)| format!("{n} ({})", ty_text(*t))).collect();
            out.push_str(&format!("  makes         {}\n", made.join(", ")));
        }
        for (line, what) in &m.record_writes {
            out.push_str(&format!("  writes        {what}, a variable of this def ({line})\n"));
        }
        for (line, what) in &m.copies {
            out.push_str(&format!("  copies        {what}, read from a place and stored as its own value ({line})\n"));
        }
        for (line, keeps) in &m.loops {
            let kept = if keeps.is_empty() { "nothing".to_string() } else { keeps.join(", ") };
            out.push_str(&format!("  loop          a pass keeps {kept}, and frees the rest of what it made ({line})\n"));
        }
        if crate::builtins::heap_ty(m.returns) {
            let t = ty_text(m.returns);
            let a = if t.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
            let what = if t == "several values" { t.clone() } else { format!("{a} {t}") };
            out.push_str(&format!("  returns       {what}, with all it holds; the rest of what the def made is freed\n"));
        }
        out.push('\n');
    }
    out
}

/// The lens as JSON, for tools and agents.
pub fn json(facts: &[DefMemory], only: Option<&str>) -> String {
    let s = crate::debug_lens::json_str;
    let defs: Vec<String> = facts
        .iter()
        .filter(|m| only.map_or(true, |d| d == m.name))
        .map(|m| {
            let params: Vec<String> = m
                .params
                .iter()
                .map(|(n, t)| format!("{{\"name\":{},\"type\":{},\"in_place\":{}}}", s(n), s(&ty_text(*t)), m.in_place.get(n).map_or("null".to_string(), |l| s(l))))
                .collect();
            let makes: Vec<String> = m.makes.iter().map(|(n, t)| format!("{{\"name\":{},\"type\":{}}}", s(n), s(&ty_text(*t)))).collect();
            let pairs = |v: &[(String, String)]| v.iter().map(|(l, w)| format!("{{\"at\":{},\"what\":{}}}", s(l), s(w))).collect::<Vec<_>>().join(",");
            let loops: Vec<String> = m.loops.iter().map(|(l, k)| format!("{{\"at\":{},\"keeps\":[{}]}}", s(l), k.iter().map(|x| s(x)).collect::<Vec<_>>().join(","))).collect();
            format!(
                "{{\"def\":{},\"at\":{},\"params\":[{}],\"makes\":[{}],\"record_writes\":[{}],\"copies\":[{}],\"loops\":[{}],\"returns\":{}}}",
                s(&m.name),
                s(&m.at),
                params.join(","),
                makes.join(","),
                pairs(&m.record_writes),
                pairs(&m.copies),
                loops.join(","),
                s(&ty_text(m.returns))
            )
        })
        .collect();
    format!("{{\"lens\":\"memory\",\"defs\":[{}]}}", defs.join(","))
}
