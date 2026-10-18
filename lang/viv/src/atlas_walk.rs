// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The walk's recording, made by `viv atlas` on the command line.
//!
//! The site's walk (www/src/components/ZoomWalk.jsx) records a run in the
//! reader's browser: viv.wasm's `viv_explore_run` runs the program with every
//! def traced, and a second recording takes the passes and calls worth reading
//! line by line. This is the same procedure, natively, calling the function
//! the wasm export calls (`crate::graph::explore_run_json`) with the same
//! requests, so the page `viv atlas` writes replays the recording the site
//! would show, without viv.wasm.
//!
//! The pieces are ports of the site's pure functions and keep their names:
//!
//!   trace_source        lib/zoomwalk.js traceSource (with parallelReached)
//!   hot_defs            lib/zoomwalk.js hotDefs
//!   Tree::build         lib/explore.js buildTree, the parts the second pass reads
//!   notable             lib/explore.js notableIterations (which passes, not why)
//!   second_pass_*       lib/explore.js secondPassIters, secondPassCalls
//!   whole_run_request   lib/explore.js wholeRunRequest
//!
//! The result is the story line the run printed (`{"total":..,"events":[..]}`,
//! byte for byte) with the counts of failed asserts and holds.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::story::{parse_json, Json};

/// A traced run with more events than this is recorded again without
/// tracing its hot defs (lib/zoomwalk.js TRACE_LIMIT).
pub const TRACE_LIMIT: u64 = 150_000;
/// A def called more often than this in the kept events, with no steps of its own, is hot (HOT_CALLS).
pub const HOT_CALLS: usize = 400;
/// lib/explore.js WHOLE_RUN_EVENTS and WHOLE_RUN_PASSES.
pub const WHOLE_RUN_EVENTS: u64 = 12_000;
pub const WHOLE_RUN_PASSES: usize = 40;

/// One recording, as lib/explore.js parseRecording reads it.
pub struct Recording {
    /// the `__story__` line's JSON, as the run printed it
    pub story: String,
    pub total: u64,
    pub fails: u64,
    pub holds: u64,
    events: Vec<Json>,
}

/// The walk's recording of a run, ready for the page.
pub struct Walk {
    pub rec: Recording,
    /// what the walk leaves out, as ZoomWalk.jsx says it
    pub note: Option<String>,
    pub seconds: f64,
}

fn arr(j: Option<&Json>) -> &[Json] {
    match j {
        Some(Json::Arr(xs)) => xs,
        _ => &[],
    }
}

fn s(j: Option<&Json>) -> &str {
    j.and_then(Json::str).unwrap_or("")
}

fn is_ident(w: &str) -> bool {
    let mut cs = w.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The names followed by `(` in a line: `/([A-Za-z_]\w*)\s*\(/g`.
fn called_names(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut out = Vec::new();
    let mut k = 0;
    while k < b.len() {
        if b[k].is_ascii_alphabetic() || b[k] == b'_' {
            let start = k;
            while k < b.len() && word(b[k]) {
                k += 1;
            }
            let mut j = k;
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'(' {
                out.push(&line[start..k]);
                k = j + 1;
            }
        } else {
            k += 1;
        }
    }
    out
}

/// A line with its strings emptied (`"(?:[^"\\]|\\.)*"` -> `""`) and its comment cut.
fn code_of(line: &str) -> String {
    let cs: Vec<char> = line.chars().collect();
    let mut out = String::new();
    let mut k = 0;
    while k < cs.len() {
        let c = cs[k];
        if c == '"' {
            // the closing quote, past escapes; an unclosed quote stays as it is
            let mut j = k + 1;
            let mut end = None;
            while j < cs.len() {
                if cs[j] == '\\' {
                    j += 2;
                    continue;
                }
                if cs[j] == '"' {
                    end = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(e) = end {
                out.push_str("\"\"");
                k = e + 1;
                continue;
            }
            out.push(c);
        } else if c == '#' {
            break;
        } else {
            out.push(c);
        }
        k += 1;
    }
    out
}

/// The defs a `parallel for` reaches (lib/zoomwalk.js parallelReached).
fn parallel_reached(bundle: &Json) -> HashSet<String> {
    let names: HashSet<&str> = arr(bundle.get("nodes")).iter().filter(|n| s(n.get("kind")) == "def").map(|n| s(n.get("id"))).collect();
    let lines: Vec<&str> = s(bundle.get("program")).split('\n').collect();
    let mut todo: Vec<String> = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        let t = l.trim_start();
        let Some(rest) = t.strip_prefix("parallel") else { continue };
        let rest2 = rest.trim_start();
        if rest.len() == rest2.len() || !rest2.starts_with("for") || rest2[3..].chars().next().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let mut depth: i64 = 0;
        for (j, line) in lines.iter().enumerate().skip(k) {
            let code = code_of(line);
            for m in called_names(&code) {
                if names.contains(m) {
                    todo.push(m.to_string());
                }
            }
            for ch in code.chars() {
                depth += match ch {
                    '{' => 1,
                    '}' => -1,
                    _ => 0,
                };
            }
            if depth <= 0 && j > k {
                break;
            }
            if depth <= 0 && code.contains('}') {
                break;
            }
        }
    }
    let mut callees: HashMap<&str, Vec<&str>> = HashMap::new();
    for c in arr(bundle.get("calls")) {
        if let Json::Arr(x) = c {
            callees.entry(s(x.first())).or_default().push(s(x.get(1)));
        }
    }
    let mut out = HashSet::new();
    while let Some(d) = todo.pop() {
        if !out.insert(d.clone()) {
            continue;
        }
        for c in callees.get(d.as_str()).into_iter().flatten() {
            todo.push(c.to_string());
        }
    }
    out
}

/// The program with a `trace` line naming every def (lib/zoomwalk.js traceSource).
pub fn trace_source(bundle: &Json, also: &HashSet<String>) -> String {
    let program = s(bundle.get("program"));
    let mut skip = parallel_reached(bundle);
    skip.extend(also.iter().cloned());
    let defs: Vec<&str> = arr(bundle.get("nodes"))
        .iter()
        .filter(|n| s(n.get("kind")) == "def")
        .map(|n| s(n.get("id")))
        .filter(|id| *id != "<main>" && is_ident(id) && !skip.contains(*id))
        .collect();
    if defs.is_empty() {
        return program.to_string();
    }
    let line = format!("  trace {}\n", defs.join(", "));
    let head = "debug * {\n";
    let at = if program.starts_with(head) { Some(0) } else { program.find(&format!("\n{head}")).map(|k| k + 1) };
    match at {
        Some(k) => format!("{}{line}{}", &program[..k + head.len()], &program[k + head.len()..]),
        None => format!("{}\ndebug * {{\n{line}}}\n", program.trim_end_matches('\n')),
    }
}

fn names_json(xs: &BTreeSet<String>) -> String {
    format!("[{}]", xs.iter().map(|x| crate::debug_lens::json_str(x)).collect::<Vec<_>>().join(","))
}

fn by_def_json(m: &BTreeMap<String, Vec<i64>>) -> String {
    let rows: Vec<String> = m.iter().map(|(d, ns)| format!("{}:[{}]", crate::debug_lens::json_str(d), ns.iter().map(i64::to_string).collect::<Vec<_>>().join(","))).collect();
    format!("{{{}}}", rows.join(","))
}

/// What a recording records beyond its defaults (the worker's message).
#[derive(Default)]
struct Request {
    iters: BTreeMap<String, Vec<i64>>,
    calls: BTreeMap<String, Vec<i64>>,
    every: BTreeSet<String>,
}

impl Request {
    fn is_empty(&self) -> bool {
        self.iters.is_empty() && self.calls.is_empty() && self.every.is_empty()
    }
}

/// One recording: what public/explore-worker.js asks viv.wasm for.
fn record(src: &str, defs: &BTreeSet<String>, req: &Request) -> Result<Recording, String> {
    let request = format!(
        "{{\"src\":{},\"iters\":{},\"calls\":{},\"defs\":{},\"every\":{}}}",
        crate::debug_lens::json_str(src),
        by_def_json(&req.iters),
        by_def_json(&req.calls),
        names_json(defs),
        names_json(&req.every)
    );
    let out = crate::graph::explore_run_json(&request);
    if let Some(e) = out.strip_prefix("error:") {
        return Err(e.trim().to_string());
    }
    let Some(line) = out.lines().find(|l| l.starts_with("__story__ ")) else { return Err("the run printed no story".into()) };
    let story = line["__story__ ".len()..].to_string();
    let j = parse_json(&story)?;
    let total = j.get("total").and_then(Json::num).unwrap_or(0.0) as u64;
    let events = match j {
        Json::Obj(fs) => fs.into_iter().find(|(k, _)| k == "events").map(|(_, v)| v),
        _ => None,
    };
    let events = match events {
        Some(Json::Arr(xs)) => xs,
        _ => Vec::new(),
    };
    let (mut fails, mut holds) = (0, 0);
    if let Some(ex) = out.lines().find_map(|l| l.strip_prefix("__explore__ ")) {
        if let Ok(c) = parse_json(ex) {
            fails = c.get("fails").and_then(Json::num).unwrap_or(0.0) as u64;
            holds = c.get("holds").and_then(Json::num).unwrap_or(0.0) as u64;
        }
    }
    Ok(Recording { story, total, fails, holds, events })
}

/// The defs called more than HOT_CALLS times in the kept events, without steps (hotDefs).
fn hot_defs(rec: &Recording, bundle: &Json) -> HashSet<String> {
    let mut n: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &rec.events {
        if let Json::Arr(x) = e {
            if s(x.first()) == "call" {
                *n.entry(s(x.get(1))).or_insert(0) += 1;
            }
        }
    }
    let stepped: HashSet<&str> = arr(bundle.get("nodes")).iter().filter(|g| matches!(g.get("stepped"), Some(Json::Bool(true)))).map(|g| s(g.get("id"))).collect();
    n.into_iter().filter(|(d, k)| *k > HOT_CALLS && !stepped.contains(d)).map(|(d, _)| d.to_string()).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Call,
    Step,
    Loop,
    Iter,
    Line,
}

struct Node {
    kind: Kind,
    def: String,
    id: Json,
    n: i64,
    parent: Option<usize>,
    children: Vec<usize>,
    implicit: bool,
    /// the values of `watch` events (and of a line node, its own)
    watch: Vec<(usize, Vec<(String, Json)>)>,
    holds_failed: usize,
    branches: Vec<(i64, bool)>,
}

/// The run as a tree (lib/explore.js buildTree): calls, steps, loops,
/// iterations and lines, with what the second pass reads of them.
pub struct Tree {
    nodes: Vec<Node>,
}

fn values(j: Option<&Json>) -> Vec<(String, Json)> {
    match j {
        Some(Json::Obj(fs)) => fs.clone(),
        _ => Vec::new(),
    }
}

impl Tree {
    fn mk(&mut self, kind: Kind, def: &str, _i: usize, parent: Option<usize>) -> usize {
        self.nodes.push(Node { kind, def: def.to_string(), id: Json::Null, n: 0, parent, children: Vec::new(), implicit: false, watch: Vec::new(), holds_failed: 0, branches: Vec::new() });
        let k = self.nodes.len() - 1;
        if let Some(p) = parent {
            self.nodes[p].children.push(k);
        }
        k
    }

    fn call_of(&self, mut n: usize) -> usize {
        loop {
            if self.nodes[n].kind == Kind::Call {
                return n;
            }
            match self.nodes[n].parent {
                Some(p) => n = p,
                None => return n,
            }
        }
    }

    pub fn build(events: &[Json], line_defs: &HashSet<String>) -> Tree {
        let mut t = Tree { nodes: Vec::new() };
        let root = t.mk(Kind::Call, "<main>", 0, None);
        let mut stack = vec![root];
        let close_to = |t: &Tree, stack: &mut Vec<usize>, pred: &dyn Fn(&Node) -> bool| {
            for k in (1..stack.len()).rev() {
                if pred(&t.nodes[stack[k]]) {
                    stack.truncate(k);
                    return;
                }
            }
        };
        for (i, e) in events.iter().enumerate() {
            let Json::Arr(x) = e else { continue };
            let kind = s(x.first());
            let def = s(x.get(1));
            while stack.len() > 1 && t.nodes[*stack.last().unwrap()].implicit && def != t.nodes[*stack.last().unwrap()].def && kind != "call" {
                stack.pop();
            }
            if kind == "watch" && line_defs.contains(def) && s(x.get(2)).starts_with("line:") && t.nodes[t.call_of(*stack.last().unwrap())].def != def {
                let n = t.mk(Kind::Call, def, i, stack.last().copied());
                t.nodes[n].implicit = true;
                stack.push(n);
            }
            let p = *stack.last().unwrap();
            match kind {
                "call" => {
                    let n = t.mk(Kind::Call, def, i, Some(p));
                    stack.push(n);
                }
                "ret" => close_to(&t, &mut stack, &|n| n.kind == Kind::Call && n.def == def),
                "enter" => {
                    let n = t.mk(Kind::Step, def, i, Some(p));
                    t.nodes[n].id = x.get(2).cloned().unwrap_or(Json::Null);
                    stack.push(n);
                }
                "exit" => {
                    let id = x.get(2).cloned().unwrap_or(Json::Null);
                    close_to(&t, &mut stack, &|n| n.kind == Kind::Step && n.def == def && n.id == id)
                }
                "iter" => {
                    let n_iter = x.get(2).and_then(|j| j.num().or_else(|| j.str().and_then(|s| s.parse().ok()))).unwrap_or(0.0) as i64;
                    let last = t.nodes[p].children.last().copied();
                    let loop_ = match last {
                        Some(l) if t.nodes[l].kind == Kind::Loop && t.nodes[l].def == def && n_iter == t.nodes[l].children.len() as i64 + 1 && n_iter > 1 => l,
                        _ => t.mk(Kind::Loop, def, i, Some(p)),
                    };
                    let it = t.mk(Kind::Iter, def, i, Some(loop_));
                    t.nodes[it].n = n_iter;
                    stack.push(it);
                }
                "iter_end" => close_to(&t, &mut stack, &|n| n.kind == Kind::Iter && n.def == def),
                "watch" => {
                    let at = s(x.get(2));
                    if at.starts_with("line:") {
                        let n = t.mk(Kind::Line, def, i, Some(p));
                        t.nodes[n].watch.push((i, values(x.get(3))));
                    } else if let Some(rest) = at.strip_prefix("br:") {
                        let mut parts = rest.split(':');
                        let line = parts.next().and_then(|l| l.parse().ok()).unwrap_or(0);
                        t.nodes[p].branches.push((line, parts.next() == Some("1")));
                    } else {
                        t.nodes[p].watch.push((i, values(x.get(3))));
                    }
                }
                "hold" => {
                    if !matches!(x.get(3), Some(Json::Bool(true))) {
                        t.nodes[p].holds_failed += 1;
                    }
                }
                _ => {}
            }
        }
        t
    }

    fn all_of(&self, kind: Kind) -> Vec<usize> {
        let mut out = Vec::new();
        let mut todo = vec![0];
        while let Some(n) = todo.pop() {
            if self.nodes[n].kind == kind {
                out.push(n);
            }
            todo.extend(self.nodes[n].children.iter().rev());
        }
        out
    }

    fn failed_holds(&self, n: usize) -> bool {
        self.nodes[n].holds_failed > 0 || self.nodes[n].children.iter().any(|&c| self.failed_holds(c))
    }

    /// pathParts's keys: the branches of the pass, and its steps with theirs.
    fn path_keys(&self, n: usize, out: &mut BTreeSet<String>) {
        for (l, taken) in &self.nodes[n].branches {
            out.insert(format!("b{l}{}", if *taken { 't' } else { 'f' }));
        }
        for &c in &self.nodes[n].children {
            if self.nodes[c].kind == Kind::Step {
                let id = match &self.nodes[c].id {
                    Json::Str(x) => x.clone(),
                    j => j.text(),
                };
                out.insert(format!("s{id}"));
                self.path_keys(c, out);
            }
        }
    }

    /// iterValues: the last value of each watched name in a pass (no objects).
    fn iter_values(&self, n: usize, v: &mut HashMap<String, Json>) {
        let mut evs: Vec<&(usize, Vec<(String, Json)>)> = self.nodes[n].watch.iter().collect();
        for &c in &self.nodes[n].children {
            if self.nodes[c].kind == Kind::Line {
                evs.extend(self.nodes[c].watch.iter());
            }
        }
        evs.sort_by_key(|e| e.0);
        for (_, vals) in evs {
            for (k, x) in vals {
                if !matches!(x, Json::Obj(_) | Json::Arr(_) | Json::Null) {
                    v.insert(k.clone(), x.clone());
                }
            }
        }
        for &c in &self.nodes[n].children {
            if matches!(self.nodes[c].kind, Kind::Step | Kind::Iter) {
                self.iter_values(c, v);
            }
        }
    }

    /// The passes of a loop worth showing (notableIterations), by number.
    fn notable(&self, l: usize) -> BTreeSet<i64> {
        let its = &self.nodes[l].children;
        let mut out = BTreeSet::new();
        let Some(&first) = its.first() else { return out };
        out.insert(self.nodes[first].n);
        let mut seen: HashSet<String> = HashSet::new();
        let mut prev: Option<HashMap<String, Json>> = None;
        for &it in its {
            let mut keys = BTreeSet::new();
            self.path_keys(it, &mut keys);
            let sig = keys.into_iter().collect::<Vec<_>>().join(" ");
            if !seen.is_empty() && !seen.contains(&sig) {
                out.insert(self.nodes[it].n);
            }
            seen.insert(sig);
            if self.failed_holds(it) {
                out.insert(self.nodes[it].n);
            }
            let mut vals = HashMap::new();
            self.iter_values(it, &mut vals);
            if let Some(p) = &prev {
                if vals.iter().any(|(k, x)| matches!(x, Json::Bool(_)) && p.get(k).is_some_and(|y| y != x)) {
                    out.insert(self.nodes[it].n);
                }
            }
            prev = Some(vals);
        }
        out.insert(self.nodes[*its.last().unwrap()].n);
        out
    }

    /// The calls a call node stands for, by def in run order: real calls only.
    fn calls_by_def(&self) -> BTreeMap<String, Vec<usize>> {
        let mut by: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for n in self.all_of(Kind::Call) {
            if self.nodes[n].parent.is_some() && !self.nodes[n].implicit {
                by.entry(self.nodes[n].def.clone()).or_default().push(n);
            }
        }
        by
    }

    /// secondPassCalls: the last call of each def, and any where a hold failed.
    fn second_pass_calls(&self) -> BTreeMap<String, Vec<i64>> {
        let mut out = BTreeMap::new();
        for (d, calls) in self.calls_by_def() {
            if calls.len() < 2 {
                continue;
            }
            let mut pick: BTreeSet<i64> = BTreeSet::from([calls.len() as i64]);
            for (k, &c) in calls.iter().enumerate() {
                if self.failed_holds(c) {
                    pick.insert(k as i64 + 1);
                }
            }
            out.insert(d, pick.into_iter().filter(|&k| k > 1).take(4).collect());
        }
        out
    }

    /// secondPassIters: the notable passes of each def's loops, at most six.
    fn second_pass_iters(&self) -> BTreeMap<String, Vec<i64>> {
        let mut by: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
        for l in self.all_of(Kind::Loop) {
            for n in self.notable(l) {
                if n > 1 {
                    by.entry(self.nodes[l].def.clone()).or_default().insert(n);
                }
            }
        }
        by.into_iter()
            .map(|(d, ns)| {
                let ns: Vec<i64> = ns.into_iter().collect();
                let mut v: Vec<i64> = ns.iter().take(5).copied().collect();
                let last = *ns.last().unwrap();
                if !v.contains(&last) {
                    v.push(last);
                }
                (d, v)
            })
            .collect()
    }

    /// What the second recording asks for: the whole run's request, else the passes and calls worth reading.
    fn second_request(&self, total: u64) -> Request {
        self.whole_run_request(total).unwrap_or_else(|| Request { iters: self.second_pass_iters(), calls: self.second_pass_calls(), every: BTreeSet::new() })
    }

    /// wholeRunRequest: every pass and call line by line, for a short run.
    fn whole_run_request(&self, total: u64) -> Option<Request> {
        if total > WHOLE_RUN_EVENTS {
            return None;
        }
        let mut passes: BTreeMap<String, usize> = BTreeMap::new();
        for l in self.all_of(Kind::Loop) {
            let e = passes.entry(self.nodes[l].def.clone()).or_insert(0);
            *e = (*e).max(self.nodes[l].children.len());
        }
        let calls: BTreeMap<String, usize> = self.calls_by_def().into_iter().map(|(d, v)| (d, v.len())).collect();
        if passes.values().any(|&k| k > WHOLE_RUN_PASSES) {
            return None;
        }
        let upto = |k: usize| -> Vec<i64> { (2..=k.min(WHOLE_RUN_PASSES) as i64).collect() };
        let mut r = Request::default();
        for (d, &k) in &passes {
            if k > 1 {
                r.iters.insert(d.clone(), upto(k));
            }
        }
        for (d, &k) in &calls {
            if k > 1 {
                r.calls.insert(d.clone(), upto(k));
            }
        }
        r.every = passes.keys().chain(calls.keys()).cloned().collect();
        Some(r)
    }
}

/// The second recording's request for a run, from its first recording's
/// story (`{"total":..,"events":[..]}`): `{"iters":..,"calls":..,"every":..}`,
/// what ZoomWalk.jsx sends (wholeRunRequest, else secondPassIters and
/// secondPassCalls). Pure, so the site's functions can be checked against it.
pub fn second_request_json(story: &str, line_defs: &[String]) -> Result<String, String> {
    let j = parse_json(story)?;
    let total = j.get("total").and_then(Json::num).unwrap_or(0.0) as u64;
    let defs: HashSet<String> = line_defs.iter().cloned().collect();
    let tree = Tree::build(arr(j.get("events")), &defs);
    let req = tree.second_request(total);
    Ok(format!("{{\"iters\":{},\"calls\":{},\"every\":{}}}", by_def_json(&req.iters), by_def_json(&req.calls), names_json(&req.every)))
}

/// Records the walk of a program from its explorer bundle (`viv graph
/// --bundle`), as ZoomWalk.jsx does in the browser. A second recording
/// longer than `cap` bytes is dropped for the first, with a note.
pub fn record_walk(bundle_text: &str, cap: usize) -> Result<Walk, String> {
    let t0 = std::time::Instant::now();
    let bundle = parse_json(bundle_text)?;
    let program = s(bundle.get("program")).to_string();
    let defs: BTreeSet<String> = arr(bundle.get("lineDefs")).iter().filter_map(Json::str).map(String::from).collect();
    let line_defs: HashSet<String> = defs.iter().cloned().collect();
    let none = Request::default();
    let mut traced = true;
    let mut note = None;
    let mut hot = HashSet::new();
    let mut rec = record(&trace_source(&bundle, &hot), &defs, &none);
    if let Ok(r) = &rec {
        if r.total > TRACE_LIMIT {
            hot = hot_defs(r, &bundle);
            rec = record(&trace_source(&bundle, &hot), &defs, &none);
            let mut names: Vec<&String> = hot.iter().collect();
            names.sort();
            note = Some(format!("Calls of {} are not traced: the run made too many to keep.", names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
    }
    let mut rec = match rec {
        Ok(r) => r,
        Err(e) => {
            traced = false;
            note = Some(format!("This run could not be traced call by call ({e}); the walk shows calls of the defs with steps only."));
            record(&program, &defs, &none)?
        }
    };
    let req = Tree::build(&rec.events, &line_defs).second_request(rec.total);
    if !req.is_empty() {
        let src = if traced { trace_source(&bundle, &hot) } else { program };
        match record(&src, &defs, &req) {
            // a second recording past the cap leaves the first, whose loops
            // and calls are line by line in their first pass only
            Ok(r2) if r2.story.len() > cap && rec.story.len() <= cap => {
                let more = format!("The passes after the first are not recorded line by line: that recording ({:.1} MB) is more than a page keeps.", r2.story.len() as f64 / 1e6);
                note = Some(match note {
                    Some(n) => format!("{n} {more}"),
                    None => more,
                });
            }
            Ok(r2) => rec = r2,
            Err(_) => {}
        }
    }
    Ok(Walk { rec, note, seconds: t0.elapsed().as_secs_f64() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn called_names_are_the_words_before_a_paren() {
        assert_eq!(called_names("x = f(a) + g (b) * h"), vec!["f", "g"]);
        assert_eq!(code_of("y = f(\"a(b\") # g(x)"), "y = f(\"\") ");
    }

    #[test]
    fn the_trace_line_goes_into_the_debug_block() {
        let b = parse_json(r#"{"program":"def f(x) {\n    return x\n}\nprint(f(1.0))\n","nodes":[{"id":"<main>","kind":"def"},{"id":"f","kind":"def"}]}"#).unwrap();
        assert_eq!(trace_source(&b, &HashSet::new()), "def f(x) {\n    return x\n}\nprint(f(1.0))\ndebug * {\n  trace f\n}\n");
        let b = parse_json(r#"{"program":"x = 1.0\ndebug * {\n  watch x\n}\n","nodes":[{"id":"g","kind":"def"}]}"#).unwrap();
        assert_eq!(trace_source(&b, &HashSet::new()), "x = 1.0\ndebug * {\n  trace g\n  watch x\n}\n");
    }
}
