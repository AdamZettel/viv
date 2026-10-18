// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The code map of a program (`viv graph --map <file> [--entry <def>]`): every
//! def reachable from an entry, which module each one is written in and where,
//! and the calls between them, read from the code. The vipyr docs build writes
//! one per public function (python/vipyr/_codemap.py) and the page
//! /vipyr/map/<function> draws it.
//!
//! It is the static half of what graph.rs gives the explorer: the same parse
//! of the expanded program, the same steps (graph.rs's nodes) and the same map
//! from the expanded lines back to each module's own lines (`line_map`).
//! Beyond graph.rs it records each call site: its line, how many loops it sits
//! in, and whether the callee is a def, a function value (a parameter or a
//! local that holds a def) or a builtin. Builtins are counted per def, not
//! drawn as nodes.
//!
//! The entry is a def of the program, or, when none is named, the top level
//! (`<main>`): the program as `viv run` runs it, its `test` blocks included.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use knotkit::span::SourceId;

use crate::ast::*;
use knotkit::span::Span;
use crate::debug_lens::json_str;
use crate::graph::{graph, line_map, Lines};

/// One call site.
struct Site {
    callee: String,
    line: usize,
    /// loops around the call inside its def
    depth: usize,
    /// the callee is named as a value (passed, stored), not called
    value: bool,
}

/// What one def (or the top level) calls, holds and loops over.
#[derive(Default)]
struct Body {
    sites: Vec<Site>,
    /// calls through a function value: (the name called, line, depth)
    indirect: Vec<(String, usize, usize)>,
    builtins: BTreeMap<String, usize>,
    /// (first line, last line, depth, header)
    loops: Vec<(usize, usize, usize, String)>,
}

struct Walk<'a> {
    defs: &'a HashSet<String>,
    records: &'a HashSet<String>,
    lines: &'a Lines,
    src: &'a str,
    /// names bound in the def being walked: its parameters and locals
    locals: HashSet<String>,
    out: Body,
}

impl Walk<'_> {
    fn expr(&mut self, e: &Expr, depth: usize) {
        match e {
            Expr::Var { name, span } => {
                if self.defs.contains(name) && !self.locals.contains(name) {
                    self.out.sites.push(Site { callee: name.clone(), line: self.lines.line(span.start), depth, value: true });
                }
            }
            Expr::Call { callee, args, span } => {
                let line = self.lines.line(span.start);
                if self.locals.contains(callee) {
                    self.out.indirect.push((callee.clone(), line, depth));
                } else if self.defs.contains(callee) {
                    if !crate::update::is_expression_def(callee) {
                        self.out.sites.push(Site { callee: callee.clone(), line, depth, value: false });
                    }
                } else if !self.records.contains(callee) && !callee.starts_with("__") && !callee.starts_with("viv__") {
                    *self.out.builtins.entry(callee.clone()).or_insert(0) += 1;
                }
                for a in args {
                    self.expr(a, depth);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs, depth);
                self.expr(rhs, depth);
            }
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.expr(operand, depth),
            Expr::Index { base, indices, .. } => {
                self.expr(base, depth);
                for i in indices {
                    self.expr(i, depth);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for x in elems {
                    self.expr(x, depth);
                }
            }
            Expr::Lambda { params, body, .. } => {
                let added: Vec<String> = params.iter().filter(|p| self.locals.insert((*p).clone())).cloned().collect();
                self.expr(body, depth);
                for p in added {
                    self.locals.remove(&p);
                }
            }
            Expr::Field { base, .. } => self.expr(base, depth),
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    self.expr(v, depth);
                }
            }
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } => {}
        }
    }

    fn header(&self, span: Span) -> String {
        let start = span.start as usize;
        let rest = &self.src[start.min(self.src.len())..];
        let line = rest.lines().next().unwrap_or("");
        line.trim().trim_end_matches('{').trim().to_string()
    }

    fn stmts(&mut self, body: &[Stmt], depth: usize) {
        for s in body {
            self.stmt(s, depth);
        }
    }

    fn stmt(&mut self, s: &Stmt, depth: usize) {
        match s {
            Stmt::Assign { name, value, .. } => {
                self.expr(value, depth);
                self.locals.insert(name.clone());
            }
            Stmt::Return { value, .. } => self.expr(value, depth),
            Stmt::ExprStmt { expr, .. } | Stmt::Show { expr, .. } => self.expr(expr, depth),
            Stmt::Test { body, .. } => self.stmts(body, depth),
            Stmt::SetIndex { base, indices, value, .. } => {
                self.expr(base, depth);
                for i in indices {
                    self.expr(i, depth);
                }
                self.expr(value, depth);
            }
            Stmt::SetField { base, value, .. } => {
                self.expr(base, depth);
                self.expr(value, depth);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                self.expr(cond, depth);
                self.stmts(then_body, depth);
                self.stmts(else_body, depth);
            }
            Stmt::While { cond, body, span } => {
                self.loop_at(*span, depth);
                self.expr(cond, depth + 1);
                self.stmts(body, depth + 1);
            }
            Stmt::For { var, bound, body, span, .. } => {
                self.loop_at(*span, depth);
                self.expr(bound, depth);
                self.locals.insert(var.clone());
                self.stmts(body, depth + 1);
            }
            Stmt::Narrate { .. } | Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. } => {}
        }
    }

    fn loop_at(&mut self, span: Span, depth: usize) {
        let first = self.lines.line(span.start);
        let last = self.lines.line(span.end.saturating_sub(1));
        let head = self.header(span);
        self.out.loops.push((first, last, depth + 1, head));
    }
}

/// The comment block right above line `line` (1-based) of `text`: the `#`
/// lines with no blank line between them and the def, `#!` lines left out.
fn doc_above(text: &str, line: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut k = line.saturating_sub(1);
    while k > 0 {
        let l = lines.get(k - 1).map(|l| l.trim()).unwrap_or("");
        if !l.starts_with('#') || l.starts_with("#!") || l.starts_with("###") {
            break;
        }
        // `#| text` and `#: name  text` are doc lines, `# text` a comment: all
        // read as the def's
        let body = l.trim_start_matches('#').trim_start_matches(['|', ':']);
        out.push(body.strip_prefix(' ').unwrap_or(body));
        k -= 1;
    }
    out.reverse();
    out.join("\n")
}

fn list_json(xs: &[String]) -> String {
    format!("[{}]", xs.iter().map(|x| json_str(x)).collect::<Vec<_>>().join(","))
}

/// The map as JSON. `entry` is a def of the program, or `None` for the top
/// level.
pub fn map_json(input: &crate::lens::Input, entry: Option<&str>) -> Result<String, String> {
    let src = &input.expanded;
    let mut program = crate::parse(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    program.debug.clear();
    let g = graph(&program, src);
    let lines = Lines::new(src);
    let defs: HashSet<String> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
    let records: HashSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Record(r) => Some(r.name.clone()),
            Item::Enum(e) => Some(e.name.clone()),
            _ => None,
        })
        .collect();
    if let Some(e) = entry {
        if !defs.contains(e) {
            return Err(format!("viv graph --map: no def `{e}` in {}", input.root.name));
        }
    }
    let walk = |params: &[String], body: &[Stmt]| -> Body {
        let mut w = Walk { defs: &defs, records: &records, lines: &lines, src, locals: params.iter().cloned().collect(), out: Body::default() };
        w.stmts(body, 0);
        w.out
    };
    // each def's body, and the top level's
    let mut bodies: HashMap<String, (Body, Option<&Def>)> = HashMap::new();
    for item in &program.items {
        if let Item::Def(d) = item {
            let params: Vec<String> = d.params.iter().map(|p| p.name.clone()).collect();
            bodies.insert(d.name.clone(), (walk(&params, &d.body), Some(d)));
        }
    }
    let top: Vec<Stmt> = program.items.iter().filter_map(|i| if let Item::Stmt(s) = i { Some(s.clone()) } else { None }).collect();
    bodies.insert("<main>".into(), (walk(&[], &top), None));

    // the defs the entry reaches, nearest first, each with the caller it is
    // first reached from (the path the page shows)
    let start = entry.unwrap_or("<main>").to_string();
    let mut order: Vec<String> = vec![start.clone()];
    let mut via: HashMap<String, (Option<String>, usize)> = HashMap::from([(start.clone(), (None, 0))]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(f) = queue.pop_front() {
        let dist = via[&f].1;
        let Some((b, _)) = bodies.get(&f) else { continue };
        for s in &b.sites {
            if !via.contains_key(&s.callee) {
                via.insert(s.callee.clone(), (Some(f.clone()), dist + 1));
                order.push(s.callee.clone());
                queue.push_back(s.callee.clone());
            }
        }
    }

    // the lines back to their modules
    let mut texts: HashMap<String, String> = HashMap::new();
    for m in &input.used {
        texts.insert(m.name.clone(), m.src.clone());
    }
    let (names, lmap) = line_map(&input.root.src, &input.root.name, src, &texts);
    let module_text = |k: usize| -> &str {
        if k == 0 {
            &input.root.src
        } else {
            input.used.iter().find(|m| m.name == names[k]).map(|m| m.src.as_str()).unwrap_or("")
        }
    };
    let module_path = |k: usize| -> String {
        if k == 0 {
            input.root.path.clone()
        } else {
            input.used.iter().find(|m| m.name == names[k]).map(|m| m.path.clone()).unwrap_or_default()
        }
    };
    // a line of the expanded program as (module, line there); a line the
    // expansion wrote itself goes to the nearest line before it that maps
    let at = |line: usize| -> Option<(usize, usize)> {
        let mut l = line.min(lmap.len().saturating_sub(1));
        while l > 0 {
            if let Some(x) = lmap[l] {
                return Some(x);
            }
            l -= 1;
        }
        None
    };
    let local = |line: usize| at(line).map(|(_, n)| n).unwrap_or(0);

    // each def's intent: its `what`, from the `.intent` entry or the `#:`
    // lines above it (the lens tools' merge); none when the program does not
    // load for them
    let loaded = crate::lens::load(input).ok();
    let intent_of = |name: &str| -> String {
        loaded.as_ref().and_then(|l| l.find(name)).map(|(m, d)| crate::lens::what_of(m, d)).unwrap_or_default()
    };

    let mut used_modules: Vec<usize> = Vec::new();
    let mut def_json: Vec<String> = Vec::new();
    for name in &order {
        let Some((b, d)) = bodies.get(name) else { continue };
        let (module, first, last, params) = match d {
            Some(d) => {
                let (m, first) = at(lines.line(d.span.start)).unwrap_or((0, 0));
                let last = local(lines.line(d.span.end.saturating_sub(1)));
                (m, first, last, d.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>())
            }
            None => (0, 0, 0, Vec::new()),
        };
        if d.is_some() && !used_modules.contains(&module) {
            used_modules.push(module);
        }
        if d.is_none() {
            for st in &top {
                if let Some((m, _)) = at(lines.line(st.span().start)) {
                    if !used_modules.contains(&m) {
                        used_modules.push(m);
                    }
                }
            }
        }
        let doc = if d.is_some() { doc_above(module_text(module), first) } else { String::new() };
        let steps: Vec<String> = g
            .nodes
            .iter()
            .filter(|n| n.kind == "step" && n.def == *name)
            .map(|n| format!("{{\"id\":{},\"text\":{},\"line\":{},\"end\":{}}}", json_str(&n.name), json_str(&n.text), local(n.line), local(n.end)))
            .collect();
        let loops: Vec<String> = b.loops.iter().map(|(a, z, dp, h)| format!("{{\"line\":{},\"end\":{},\"depth\":{dp},\"head\":{}}}", local(*a), local(*z), json_str(h))).collect();
        let calls: Vec<String> = b
            .sites
            .iter()
            .map(|s| format!("[{},{},{},{}]", json_str(&s.callee), local(s.line), s.depth, if s.value { 1 } else { 0 }))
            .collect();
        let indirect: Vec<String> = b.indirect.iter().map(|(n, l, dp)| format!("[{},{},{dp}]", json_str(n), local(*l))).collect();
        let builtins: Vec<String> = b.builtins.iter().map(|(n, c)| format!("{}:{c}", json_str(n))).collect();
        let (from, dist) = via[name].clone();
        // where the top level is: the root module's lines outside every def
        let main_lines = if d.is_none() {
            // the top level's statements, each as [module, first, last]
            let spans: Vec<String> = top
                .iter()
                .filter(|s| !matches!(s, Stmt::StepEnter { .. } | Stmt::StepExit { .. } | Stmt::Lens { .. }))
                .filter_map(|s| {
                    let sp = s.span();
                    let (m, a) = at(lines.line(sp.start))?;
                    let z = local(lines.line(sp.end.saturating_sub(1)));
                    Some(format!("[{m},{a},{}]", z.max(a)))
                })
                .collect();
            format!(",\"main\":true,\"top\":[{}]", spans.join(","))
        } else {
            String::new()
        };
        // a def documented by `#: name  text` lines states its intent there
        let mut intent = if d.is_some() { intent_of(name) } else { String::new() };
        if intent.is_empty() {
            if let Some(rest) = doc.strip_prefix(name.as_str()).filter(|r| r.starts_with(' ')) {
                intent = rest.split_whitespace().collect::<Vec<_>>().join(" ");
            }
        }
        def_json.push(format!(
            "{{\"id\":{},\"module\":{},\"line\":{first},\"end\":{last},\"params\":{},\"doc\":{},\"intent\":{},\"dist\":{dist},\"from\":{},\"steps\":[{}],\"loops\":[{}],\"calls\":[{}],\"indirect\":[{}],\"builtins\":{{{}}}{main_lines}}}",
            json_str(name),
            module,
            list_json(&params),
            json_str(&doc),
            json_str(&intent),
            from.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
            steps.join(","),
            loops.join(","),
            calls.join(","),
            indirect.join(","),
            builtins.join(",")
        ));
    }
    if !used_modules.contains(&0) {
        used_modules.insert(0, 0);
    }
    let contracts = contracts_json(names.len(), &module_text);
    let modules: Vec<String> = (0..names.len())
        .map(|k| {
            let text = module_text(k);
            let src_field = if used_modules.contains(&k) { format!(",\"src\":{}", json_str(text)) } else { String::new() };
            format!("{{\"name\":{},\"path\":{},\"lines\":{}{src_field}}}", json_str(&names[k]), json_str(&module_path(k)), text.lines().count())
        })
        .collect();
    Ok(format!(
        "{{\"root\":{},\"entry\":{},\"defsInProgram\":{},\"modules\":[{}],\"defs\":[{}],\"contracts\":[{}]}}",
        json_str(&input.root.name),
        json_str(&start),
        defs.len(),
        modules.join(","),
        def_json.join(","),
        contracts.join(",")
    ))
}

/// The contracts the program's modules declare: each
/// one's JSON from `crate::contracts::json` with the module it is in and its
/// line there.
fn contracts_json<'a>(modules: usize, module_text: &dyn Fn(usize) -> &'a str) -> Vec<String> {
    let mut out = Vec::new();
    for m in 0..modules {
        let text = module_text(m);
        for c in crate::contracts::scan(text) {
            let line = text[..(c.span.start as usize).min(text.len())].matches('\n').count() + 1;
            let j = crate::contracts::json(&c, line);
            out.push(format!("{{\"module\":{m},{}", &j[1..]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(src: &str) -> crate::lens::Input {
        let root = crate::lens::ModuleText { name: "t".into(), path: "t.viv".into(), src: src.into(), intent_path: String::new(), intent: None };
        crate::lens::Input { root, used: Vec::new(), expanded: src.into(), companion_tests: Vec::new() }
    }

    const SRC: &str = "# the leaf\ndef leaf(x: Num) -> Num {\n    return sqrt(x) + sqrt(x)\n}\n\ndef mid(n: Int) -> Num {\n    s = 0.0\n    for i to n {\n        for j to n {\n            s = s + leaf(float(i * j))\n        }\n    }\n    return s\n}\n\ndef unused() -> Num {\n    return 1.0\n}\n\nshow mid(3)\n";

    #[test]
    fn reaches_from_the_top_level_only() {
        let j = map_json(&input(SRC), None).unwrap();
        let v = crate::story::parse_json(&j).unwrap();
        let crate::story::Json::Arr(defs) = v.get("defs").unwrap() else { panic!("defs") };
        let ids: Vec<&str> = defs.iter().map(|d| d.get("id").unwrap().str().unwrap()).collect();
        assert_eq!(ids, ["<main>", "mid", "leaf"]);
        assert!(j.contains("\"defsInProgram\":3"));
    }

    #[test]
    fn records_lines_loops_and_builtins() {
        let j = map_json(&input(SRC), Some("mid")).unwrap();
        // the call of leaf is on line 10, inside two loops
        assert!(j.contains("\"calls\":[[\"leaf\",10,2,0]]"), "{j}");
        assert!(j.contains("\"loops\":[{\"line\":8,\"end\":12,\"depth\":1,\"head\":\"for i to n\"},{\"line\":9,\"end\":11,\"depth\":2,\"head\":\"for j to n\"}]"), "{j}");
        // leaf's doc comment and its builtin, counted per call
        assert!(j.contains("\"doc\":\"the leaf\""), "{j}");
        assert!(j.contains("\"builtins\":{\"sqrt\":2}"), "{j}");
        assert!(map_json(&input(SRC), Some("nope")).is_err());
    }

    #[test]
    fn finds_a_contract_and_its_methods() {
        let src = "#: power  x^2 by multiplying\ndef power(x: Num) -> Num {\n    return x * x\n}\n\n#: by_pow  x^2 by pow\ndef by_pow(x: Num) -> Num {\n    return pow(x, 2.0)\n}\n\ncontract square(x: Num) -> value: Num {\n    intent \"the square of a number\"\n    methods power, by_pow\n    agree value within absolute 1.0e-12\n}\n\ny = square(3.0)\nshow y\n";
        let j = map_json(&input(src), None).unwrap();
        assert!(j.contains("\"contracts\":[{\"module\":0,\"name\":\"square\",\"line\":11,\"intent\":\"the square of a number\""), "{j}");
        assert!(j.contains("\"methods\":[\"power\",\"by_pow\"],\"groups\":[],\"default\":\"power\",\"using\":\"power\""), "{j}");
        assert!(j.contains("\"doc\":\"power  x^2 by multiplying\",\"intent\":\"x^2 by multiplying\""), "{j}");
    }
}
